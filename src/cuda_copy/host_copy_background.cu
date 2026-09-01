#include "copy_common.cuh"
#include "temporal_stats.cuh"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kWarmupIterations = 10;
constexpr size_t kDefaultBytes = 255ULL * 1024ULL * 1024ULL;

enum class Direction {
  kD2H,
  kH2D,
};

struct Options {
  std::vector<int> devices{0, 1};
  Direction direction = Direction::kD2H;
  size_t bytes = kDefaultBytes;
  double duty_cycle = 1.0;
  int report_seconds = 2;
  std::string ready_file;
  std::string stop_file;
  std::string output;
};

struct Counter {
  std::atomic<unsigned long long> bytes{0};
  std::vector<unsigned long long> operation_begin_ns;
  std::vector<unsigned long long> operation_end_ns;
};

struct SharedState {
  std::atomic<int> ready{0};
  std::atomic<bool> run{false};
  std::atomic<bool> stop{false};
  std::atomic<bool> failed{false};
  std::mutex error_mutex;
  std::string error;
};

volatile std::sig_atomic_t g_signal_stop = 0;

void signalHandler(int) {
  g_signal_stop = 1;
}

unsigned long long steadyNowNs() {
  return static_cast<unsigned long long>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

const char* directionName(Direction direction) {
  return direction == Direction::kD2H ? "d2h" : "h2d";
}

void printUsage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --devList=0,1                         GPUs that generate traffic\n"
      << "  --direction=d2h|h2d                  transfer direction\n"
      << "  --size=255M                            one cudaMemcpyAsync size\n"
      << "  --dutyCycle=1.0                        target wall active duty in (0,1]\n"
      << "  --readyFile=<path>                     optional ready marker\n"
      << "  --stopFile=<path>                      optional stop marker\n"
      << "  --reportSec=2                          progress report interval\n"
      << "  --output=<path>                        optional JSON output path\n"
      << "  --help                                 show this message\n"
      << "Defaults: devList=0,1, direction=d2h, warmup=10 (fixed), "
         "size=255M, dutyCycle=1.0, loop=forever\n";
}

double parseDutyCycle(const std::string& value) {
  size_t consumed = 0;
  double duty = 0.0;
  try {
    duty = std::stod(value, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("--dutyCycle must be a decimal in (0,1]");
  }
  if (consumed != value.size() || !std::isfinite(duty) || duty <= 0.0 ||
      duty > 1.0) {
    throw std::invalid_argument("--dutyCycle must be a decimal in (0,1]");
  }
  return duty;
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    }

    std::string value;
    if (copybench::consumeOption(index, argc, argv,
                                 {"--devList", "--dev-list"}, &value)) {
      options.devices = copybench::parseDeviceList(value, "--devList");
    } else if (copybench::consumeOption(index, argc, argv,
                                        {"--direction"}, &value)) {
      const std::string direction = copybench::lower(value);
      if (direction == "d2h") {
        options.direction = Direction::kD2H;
      } else if (direction == "h2d") {
        options.direction = Direction::kH2D;
      } else {
        throw std::invalid_argument("--direction must be d2h or h2d");
      }
    } else if (copybench::consumeOption(index, argc, argv, {"--size"},
                                        &value)) {
      options.bytes = copybench::parseSize(value, "--size");
    } else if (copybench::consumeOption(index, argc, argv, {"--dutyCycle"},
                                        &value)) {
      options.duty_cycle = parseDutyCycle(value);
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--readyFile", "--ready-file"},
                   &value)) {
      options.ready_file = value;
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--stopFile", "--stop-file"},
                   &value)) {
      options.stop_file = value;
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--reportSec", "--report-sec"},
                   &value)) {
      options.report_seconds = copybench::parsePositiveInt(value, "--reportSec");
    } else if (copybench::consumeOption(index, argc, argv, {"--output"},
                                        &value)) {
      options.output = value;
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  return options;
}

void setWorkerFailure(SharedState* state, const std::exception& error) {
  {
    std::lock_guard<std::mutex> lock(state->error_mutex);
    if (state->error.empty()) state->error = error.what();
  }
  state->failed.store(true, std::memory_order_release);
  state->stop.store(true, std::memory_order_release);
}

void cleanupWorker(int device, void* host_buffer, void* device_buffer,
                   cudaStream_t stream) noexcept {
  cudaSetDevice(device);
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (device_buffer != nullptr) cudaFree(device_buffer);
  if (host_buffer != nullptr) cudaFreeHost(host_buffer);
}

void worker(int device, Direction direction, size_t bytes, double duty_cycle,
            Counter* counter,
            SharedState* state) {
  void* host_buffer = nullptr;
  void* device_buffer = nullptr;
  cudaStream_t stream = nullptr;
  try {
    COPYBENCH_CUDA_CHECK(cudaSetDevice(device));
    COPYBENCH_CUDA_CHECK(cudaMallocHost(&host_buffer, bytes));
    COPYBENCH_CUDA_CHECK(cudaMalloc(&device_buffer, bytes));
    std::memset(host_buffer, device & 0xff, bytes);
    COPYBENCH_CUDA_CHECK(cudaMemset(device_buffer, device & 0xff, bytes));
    COPYBENCH_CUDA_CHECK(cudaStreamCreate(&stream));

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
      if (direction == Direction::kD2H) {
        COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
            host_buffer, device_buffer, bytes, cudaMemcpyDeviceToHost, stream));
      } else {
        COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
            device_buffer, host_buffer, bytes, cudaMemcpyHostToDevice, stream));
      }
      COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    state->ready.fetch_add(1, std::memory_order_acq_rel);
    while (!state->run.load(std::memory_order_acquire) &&
           !state->stop.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    while (!state->stop.load(std::memory_order_acquire)) {
      const unsigned long long begin_ns = steadyNowNs();
      if (direction == Direction::kD2H) {
        COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
            host_buffer, device_buffer, bytes, cudaMemcpyDeviceToHost, stream));
      } else {
        COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
            device_buffer, host_buffer, bytes, cudaMemcpyHostToDevice, stream));
      }
      COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
      const unsigned long long end_ns = steadyNowNs();
      counter->operation_begin_ns.push_back(begin_ns);
      counter->operation_end_ns.push_back(end_ns);
      counter->bytes.fetch_add(bytes, std::memory_order_relaxed);
      if (duty_cycle < 1.0) {
        const double active_seconds =
            static_cast<double>(end_ns - begin_ns) / 1.0e9;
        const double idle_seconds =
            active_seconds * (1.0 / duty_cycle - 1.0);
        std::this_thread::sleep_for(std::chrono::duration<double>(idle_seconds));
      }
    }
  } catch (const std::exception& error) {
    setWorkerFailure(state, error);
  }
  cleanupWorker(device, host_buffer, device_buffer, stream);
}

unsigned long long totalBytes(
    const std::vector<std::unique_ptr<Counter>>& counters) {
  unsigned long long total = 0;
  for (const auto& counter : counters) {
    total += counter->bytes.load(std::memory_order_relaxed);
  }
  return total;
}

struct DeviceTimingSummary {
  double wall_active_sec = 0.0;
  copybench::TimingStats duration;
  copybench::TimingStats submit_interval;
  copybench::TimingStats idle_gap;
};

DeviceTimingSummary summarizeTiming(const Counter& counter) {
  const std::vector<unsigned long long> durations =
      copybench::durationNanoseconds(counter.operation_begin_ns,
                                     counter.operation_end_ns);
  const std::vector<unsigned long long> submit_intervals =
      copybench::submitIntervalsNanoseconds(counter.operation_begin_ns);
  const std::vector<unsigned long long> idle_gaps =
      copybench::idleGapsNanoseconds(counter.operation_begin_ns,
                                     counter.operation_end_ns);
  DeviceTimingSummary summary;
  summary.wall_active_sec = copybench::sumNanoseconds(durations);
  summary.duration = copybench::summarizeNanoseconds(durations);
  summary.submit_interval = copybench::summarizeNanoseconds(submit_intervals);
  summary.idle_gap = copybench::summarizeNanoseconds(idle_gaps);
  return summary;
}

std::string makeJson(const Options& options, unsigned long long bytes,
                     double elapsed_seconds, double aggregate_gbps,
                     const std::vector<std::unique_ptr<Counter>>& counters,
                     const std::vector<std::string>& device_names) {
  std::ostringstream json;
  json << std::fixed << std::setprecision(6);
  json << "{\n"
       << "  \"program\": \"host_copy_background\",\n"
       << "  \"direction\": "
       << copybench::jsonQuote(directionName(options.direction)) << ",\n"
       << "  \"devices\": [";
  for (size_t index = 0; index < options.devices.size(); ++index) {
    if (index != 0) json << ", ";
    json << options.devices[index];
  }
  json << "],\n"
       << "  \"deviceNames\": [";
  for (size_t index = 0; index < device_names.size(); ++index) {
    if (index != 0) json << ", ";
    json << copybench::jsonQuote(device_names[index]);
  }
  json << "],\n"
       << "  \"warmup\": " << kWarmupIterations << ",\n"
       << "  \"bytesPerMemcpy\": " << options.bytes << ",\n"
       << "  \"dutyCycleTarget\": " << options.duty_cycle << ",\n"
       << "  \"totalBytes\": " << bytes << ",\n"
       << "  \"elapsedSec\": " << elapsed_seconds << ",\n"
       << "  \"perDeviceBytes\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    json << counters[index]->bytes.load(std::memory_order_relaxed);
  }
  json << "],\n"
       << "  \"perDeviceGBps\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    const unsigned long long device_bytes =
        counters[index]->bytes.load(std::memory_order_relaxed);
    const double device_gbps = elapsed_seconds > 0.0
                                   ? static_cast<double>(device_bytes) /
                                         elapsed_seconds / 1e9
                                   : 0.0;
    json << device_gbps;
  }
  json << "],\n  \"perDeviceOperations\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    json << counters[index]->operation_begin_ns.size();
  }
  json << "],\n  \"perDeviceWallActiveSec\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    json << summarizeTiming(*counters[index]).wall_active_sec;
  }
  json << "],\n  \"perDeviceWallActiveDuty\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    const double active_sec = summarizeTiming(*counters[index]).wall_active_sec;
    json << (elapsed_seconds > 0.0 ? active_sec / elapsed_seconds : 0.0);
  }
  json << "],\n  \"perDeviceGpuActivitySec\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    json << "null";
  }
  json << "],\n  \"perDeviceGpuActivityDuty\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    json << "null";
  }
  json << "],\n  \"perDeviceOperationDurationMs\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    copybench::writeTimingStats(json, summarizeTiming(*counters[index]).duration);
  }
  json << "],\n  \"perDeviceSubmitIntervalMs\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    copybench::writeTimingStats(json,
                                summarizeTiming(*counters[index]).submit_interval);
  }
  json << "],\n  \"perDeviceIdleGapMs\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    copybench::writeTimingStats(json,
                                summarizeTiming(*counters[index]).idle_gap);
  }
  json << "],\n"
       << "  \"aggregateGBps\": " << aggregate_gbps << "\n"
       << "}\n";
  return json.str();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  SharedState state;
  std::vector<std::unique_ptr<Counter>> counters;
  std::vector<std::thread> workers;
  try {
    options = parseOptions(argc, argv);
    int device_count = 0;
    COPYBENCH_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::vector<std::string> device_names;
    device_names.reserve(options.devices.size());
    for (int device : options.devices) {
      if (device >= device_count) {
        std::ostringstream message;
        message << "requested GPU " << device << ", but only "
                << device_count << " CUDA devices are available";
        throw std::invalid_argument(message.str());
      }
      cudaDeviceProp properties{};
      COPYBENCH_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
      device_names.emplace_back(properties.name);
    }

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    counters.reserve(options.devices.size());
    for (size_t index = 0; index < options.devices.size(); ++index) {
      counters.emplace_back(std::make_unique<Counter>());
    }

    workers.reserve(options.devices.size());
    for (size_t index = 0; index < options.devices.size(); ++index) {
      workers.emplace_back(worker, options.devices[index], options.direction,
                           options.bytes, options.duty_cycle,
                           counters[index].get(), &state);
    }

    while (state.ready.load(std::memory_order_acquire) <
               static_cast<int>(options.devices.size()) &&
           !state.failed.load(std::memory_order_acquire) && !g_signal_stop) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    bool measurement_started = false;
    std::chrono::steady_clock::time_point measurement_start;
    if (g_signal_stop) state.stop.store(true, std::memory_order_release);
    if (!state.failed.load(std::memory_order_acquire) && !g_signal_stop) {
      if (!options.ready_file.empty()) {
        copybench::writeTextFile(options.ready_file, "ready\n");
      }
      std::cout << "PCIe background copy\n"
                << "devices=" << copybench::joinDevices(options.devices)
                << " direction=" << directionName(options.direction)
                << " bytes=" << options.bytes
                << " dutyCycleTarget=" << options.duty_cycle
                << " warmup=" << kWarmupIterations << '\n'
                << "loop=forever; stop with SIGINT/SIGTERM or --stopFile\n";
      std::cout.flush();

      measurement_start = std::chrono::steady_clock::now();
      measurement_started = true;
      state.run.store(true, std::memory_order_release);
      auto last_report = measurement_start;
      unsigned long long last_bytes = 0;
      while (!state.stop.load(std::memory_order_acquire)) {
        if (g_signal_stop || copybench::fileExists(options.stop_file)) {
          state.stop.store(true, std::memory_order_release);
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto now = std::chrono::steady_clock::now();
        const double since_report =
            std::chrono::duration<double>(now - last_report).count();
        if (since_report >= options.report_seconds) {
          const unsigned long long current_bytes = totalBytes(counters);
          const double aggregate_gbps =
              static_cast<double>(current_bytes - last_bytes) /
              since_report / 1e9;
          std::cout << std::fixed << std::setprecision(3)
                    << "window_bytes=" << (current_bytes - last_bytes)
                    << " time_sec=" << since_report
                    << " aggregateGBps=" << aggregate_gbps << '\n';
          std::cout.flush();
          last_bytes = current_bytes;
          last_report = now;
        }
      }

      state.stop.store(true, std::memory_order_release);
    } else {
      state.stop.store(true, std::memory_order_release);
    }

    for (std::thread& thread : workers) {
      if (thread.joinable()) thread.join();
    }

    if (state.failed.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(state.error_mutex);
      throw std::runtime_error(state.error.empty() ? "background worker failed"
                                                   : state.error);
    }

    if (measurement_started) {
      const auto end = std::chrono::steady_clock::now();
      const double elapsed_seconds =
          std::chrono::duration<double>(end - measurement_start).count();
      const unsigned long long bytes = totalBytes(counters);
      const double aggregate_gbps = elapsed_seconds > 0.0
                                        ? static_cast<double>(bytes) /
                                              elapsed_seconds / 1e9
                                        : 0.0;
      std::cout << std::fixed << std::setprecision(3)
                << "total_bytes=" << bytes
                << " elapsed_sec=" << elapsed_seconds
                << " aggregateGBps=" << aggregate_gbps << '\n';
      copybench::writeTextFile(
          options.output,
          makeJson(options, bytes, elapsed_seconds, aggregate_gbps,
                   counters, device_names));
    }
    return 0;
  } catch (const std::exception& error) {
    state.stop.store(true, std::memory_order_release);
    for (std::thread& thread : workers) {
      if (thread.joinable()) thread.join();
    }
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
