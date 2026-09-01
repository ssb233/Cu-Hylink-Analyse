#include "copy_common.cuh"
#include "temporal_stats.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
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
constexpr size_t kL2WorkingSetBytes = 4ULL * 1024ULL * 1024ULL;
constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 4096;

enum class Path {
  kLocalD2DCE,
  kStreamingHbmRead,
  kStreamingHbmWrite,
  kL2ResidentRead,
};

struct Options {
  std::vector<int> devices{0, 1, 2};
  Path path = Path::kLocalD2DCE;
  size_t bytes = kDefaultBytes;
  double target_gbps = 4.0;
  double duty_cycle = 1.0;
  int iterations = 0;
  int report_seconds = 2;
  std::string ready_file;
  std::string stop_file;
  std::string output;
};

struct Counter {
  std::atomic<unsigned long long> bytes{0};
  std::atomic<unsigned long long> operations{0};
  std::atomic<unsigned long long> active_ns{0};
  std::vector<unsigned long long> operation_begin_ns;
  std::vector<unsigned long long> operation_end_ns;
};

struct SharedState {
  std::atomic<int> ready{0};
  std::atomic<bool> run{false};
  std::atomic<bool> stop{false};
  std::atomic<bool> failed{false};
  std::atomic<int> completed_workers{0};
  int worker_count = 0;
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

const char* pathName(Path path) {
  switch (path) {
    case Path::kLocalD2DCE: return "local-d2d-ce";
    case Path::kStreamingHbmRead: return "streaming-hbm-read";
    case Path::kStreamingHbmWrite: return "streaming-hbm-write";
    case Path::kL2ResidentRead: return "l2-resident-read";
  }
  return "unknown";
}

double parseDouble(const std::string& text, const std::string& option_name) {
  const std::string value = copybench::trim(text);
  if (value.empty()) {
    throw std::invalid_argument(option_name + " requires a number");
  }
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(value.c_str(), &end);
  if (errno == ERANGE || end == value.c_str() || *end != '\0' ||
      !std::isfinite(parsed)) {
    throw std::invalid_argument(option_name + " has an invalid number: " +
                                text);
  }
  return parsed;
}

void printUsage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --deviceList=0,1,2                    GPUs generating background\n"
      << "  --path=local-d2d-ce|streaming-hbm-read|streaming-hbm-write|l2-resident-read\n"
      << "  --size=255M                            requested working-set size\n"
      << "  --targetGBps=4.0                       per-GPU pacing target\n"
      << "  --dutyCycle=1.0                        active-time fraction (0,1]\n"
      << "  --iterations=0                         bounded operations per GPU; 0 means forever\n"
      << "  --readyFile=<path>                     ready marker\n"
      << "  --stopFile=<path>                      stop marker\n"
      << "  --reportSec=2                          progress interval\n"
      << "  --output=<path>                        JSON output path\n"
      << "  --help                                 show this message\n"
      << "Defaults: deviceList=0,1,2, path=local-d2d-ce, size=255M, "
         "targetGBps=4.0, dutyCycle=1.0, warmup=10 (fixed), loop=forever\n";
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
                                 {"--deviceList", "--devList", "--device-list"},
                                 &value)) {
      options.devices = copybench::parseDeviceList(value, "--deviceList");
    } else if (copybench::consumeOption(index, argc, argv, {"--path"},
                                        &value)) {
      const std::string path = copybench::lower(value);
      if (path == "local-d2d-ce") {
        options.path = Path::kLocalD2DCE;
      } else if (path == "streaming-hbm-read") {
        options.path = Path::kStreamingHbmRead;
      } else if (path == "streaming-hbm-write") {
        options.path = Path::kStreamingHbmWrite;
      } else if (path == "l2-resident-read") {
        options.path = Path::kL2ResidentRead;
      } else {
        throw std::invalid_argument(
            "--path must be local-d2d-ce, streaming-hbm-read, "
            "streaming-hbm-write, or l2-resident-read");
      }
    } else if (copybench::consumeOption(index, argc, argv, {"--size"},
                                        &value)) {
      options.bytes = copybench::parseSize(value, "--size");
    } else if (copybench::consumeOption(index, argc, argv, {"--targetGBps"},
                                        &value)) {
      options.target_gbps = parseDouble(value, "--targetGBps");
    } else if (copybench::consumeOption(index, argc, argv, {"--dutyCycle"},
                                        &value)) {
      options.duty_cycle = parseDouble(value, "--dutyCycle");
    } else if (copybench::consumeOption(index, argc, argv, {"--iterations"},
                                        &value)) {
      options.iterations = copybench::parseNonNegativeInt(value, "--iterations");
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
  if (options.devices.empty()) {
    throw std::invalid_argument("--deviceList cannot be empty");
  }
  if (options.target_gbps < 0.0 || !std::isfinite(options.target_gbps)) {
    throw std::invalid_argument("--targetGBps must be non-negative");
  }
  if (options.duty_cycle <= 0.0 || options.duty_cycle > 1.0 ||
      !std::isfinite(options.duty_cycle)) {
    throw std::invalid_argument("--dutyCycle must be in (0,1]");
  }
  return options;
}

__global__ void streamingReadKernel(const unsigned char* source,
                                    unsigned long long* partial,
                                    size_t bytes) {
  __shared__ unsigned long long block_sum[kThreadsPerBlock];
  unsigned long long local_sum = 0;
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < bytes; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    local_sum += source[index];
  }
  block_sum[threadIdx.x] = local_sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) block_sum[threadIdx.x] += block_sum[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = block_sum[0];
}

__global__ void streamingWriteKernel(unsigned char* destination,
                                      unsigned long long* partial,
                                      size_t bytes) {
  unsigned long long local_sum = 0;
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < bytes; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const unsigned char value = static_cast<unsigned char>(index + blockIdx.x);
    destination[index] = value;
    local_sum += value;
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = local_sum;
}

struct WorkerResources {
  int device = -1;
  void* source = nullptr;
  void* destination = nullptr;
  unsigned long long* partial = nullptr;
  cudaStream_t stream = nullptr;
};

size_t operationBytes(const Options& options) {
  return options.path == Path::kL2ResidentRead
             ? std::min(options.bytes, kL2WorkingSetBytes)
             : options.bytes;
}

int blockCount(size_t bytes) {
  const size_t needed =
      (bytes + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return static_cast<int>(std::min<size_t>(std::max<size_t>(needed, 1),
                                           kMaxBlocks));
}

void issueOperation(const Options& options, WorkerResources* resource,
                    size_t bytes) {
  switch (options.path) {
    case Path::kLocalD2DCE:
      COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
          resource->destination, resource->source, bytes,
          cudaMemcpyDeviceToDevice, resource->stream));
      break;
    case Path::kStreamingHbmRead:
    case Path::kL2ResidentRead:
      streamingReadKernel<<<blockCount(bytes), kThreadsPerBlock, 0,
                            resource->stream>>>(
          static_cast<const unsigned char*>(resource->source),
          resource->partial, bytes);
      COPYBENCH_CUDA_CHECK(cudaGetLastError());
      break;
    case Path::kStreamingHbmWrite:
      streamingWriteKernel<<<blockCount(bytes), kThreadsPerBlock, 0,
                             resource->stream>>>(
          static_cast<unsigned char*>(resource->destination),
          resource->partial, bytes);
      COPYBENCH_CUDA_CHECK(cudaGetLastError());
      break;
  }
}

void setWorkerFailure(SharedState* state, const std::exception& error) {
  {
    std::lock_guard<std::mutex> lock(state->error_mutex);
    if (state->error.empty()) state->error = error.what();
  }
  state->failed.store(true, std::memory_order_release);
  state->stop.store(true, std::memory_order_release);
}

void cleanupWorker(WorkerResources* resource) noexcept {
  cudaSetDevice(resource->device);
  if (resource->stream != nullptr) cudaStreamDestroy(resource->stream);
  if (resource->partial != nullptr) cudaFree(resource->partial);
  if (resource->destination != nullptr) cudaFree(resource->destination);
  if (resource->source != nullptr) cudaFree(resource->source);
  resource->stream = nullptr;
  resource->partial = nullptr;
  resource->destination = nullptr;
  resource->source = nullptr;
}

void worker(const Options& options, int device, Counter* counter,
            SharedState* state) {
  WorkerResources resource;
  resource.device = device;
  const size_t bytes = operationBytes(options);
  try {
    COPYBENCH_CUDA_CHECK(cudaSetDevice(device));
    COPYBENCH_CUDA_CHECK(cudaMalloc(&resource.source, bytes));
    COPYBENCH_CUDA_CHECK(cudaMalloc(&resource.destination, bytes));
    COPYBENCH_CUDA_CHECK(cudaMalloc(
        &resource.partial,
        static_cast<size_t>(kMaxBlocks) * sizeof(unsigned long long)));
    COPYBENCH_CUDA_CHECK(cudaMemset(resource.source, device & 0xff, bytes));
    COPYBENCH_CUDA_CHECK(cudaMemset(resource.destination, 0, bytes));
    COPYBENCH_CUDA_CHECK(cudaStreamCreate(&resource.stream));

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
      issueOperation(options, &resource, bytes);
      COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(resource.stream));
    }

    state->ready.fetch_add(1, std::memory_order_acq_rel);
    while (!state->run.load(std::memory_order_acquire) &&
           !state->stop.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    int iterations = 0;
    while (!state->stop.load(std::memory_order_acquire) &&
           (options.iterations == 0 || iterations < options.iterations)) {
      const unsigned long long begin_ns = steadyNowNs();
      issueOperation(options, &resource, bytes);
      COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(resource.stream));
      const unsigned long long end_ns = steadyNowNs();
      const unsigned long long active_ns = end_ns >= begin_ns
                                               ? end_ns - begin_ns
                                               : 0;
      counter->operation_begin_ns.push_back(begin_ns);
      counter->operation_end_ns.push_back(end_ns);
      counter->bytes.fetch_add(bytes, std::memory_order_relaxed);
      counter->operations.fetch_add(1, std::memory_order_relaxed);
      counter->active_ns.fetch_add(active_ns, std::memory_order_relaxed);
      ++iterations;

      if (options.target_gbps > 0.0 || options.duty_cycle < 1.0) {
        const double target_seconds =
            options.target_gbps > 0.0
                ? static_cast<double>(bytes) / options.target_gbps / 1.0e9
                : 0.0;
        const double duty_seconds =
            static_cast<double>(active_ns) / 1.0e9 /
            options.duty_cycle;
        const double sleep_seconds =
            std::max(target_seconds, duty_seconds) -
            static_cast<double>(active_ns) / 1.0e9;
        if (sleep_seconds > 0.0) {
          std::this_thread::sleep_for(
              std::chrono::duration<double>(sleep_seconds));
        }
      }
    }
    if (options.iterations > 0 && iterations >= options.iterations &&
        state->completed_workers.fetch_add(1, std::memory_order_acq_rel) + 1 >=
            state->worker_count) {
      state->stop.store(true, std::memory_order_release);
    }
  } catch (const std::exception& error) {
    setWorkerFailure(state, error);
  }
  cleanupWorker(&resource);
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

std::string makeJson(const Options& options,
                    const std::vector<std::unique_ptr<Counter>>& counters,
                    const std::vector<std::string>& device_names,
                    size_t bytes, double elapsed_seconds,
                    unsigned long long bytes_total) {
  std::ostringstream json;
  json << std::fixed << std::setprecision(6);
  json << "{\n"
       << "  \"program\": \"copy_path_background\",\n"
       << "  \"path\": " << copybench::jsonQuote(pathName(options.path))
       << ",\n"
       << "  \"devices\": [";
  for (size_t index = 0; index < options.devices.size(); ++index) {
    if (index != 0) json << ", ";
    json << options.devices[index];
  }
  json << "],\n  \"deviceNames\": [";
  for (size_t index = 0; index < device_names.size(); ++index) {
    if (index != 0) json << ", ";
    json << copybench::jsonQuote(device_names[index]);
  }
  json << "],\n"
       << "  \"requestedBytes\": " << options.bytes << ",\n"
       << "  \"workingSetBytes\": " << bytes << ",\n"
       << "  \"targetGBps\": " << options.target_gbps << ",\n"
       << "  \"dutyCycle\": " << options.duty_cycle << ",\n"
       << "  \"warmup\": " << kWarmupIterations << ",\n"
       << "  \"elapsedSec\": " << elapsed_seconds << ",\n"
       << "  \"totalBytes\": " << bytes_total << ",\n"
       << "  \"aggregateGBps\": "
       << (elapsed_seconds > 0.0
               ? static_cast<double>(bytes_total) / elapsed_seconds / 1e9
               : 0.0)
       << ",\n  \"perDeviceBytes\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    json << counters[index]->bytes.load(std::memory_order_relaxed);
  }
  json << "],\n  \"perDeviceOperations\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    json << counters[index]->operations.load(std::memory_order_relaxed);
  }
  json << "],\n  \"perDeviceActiveSec\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    json << static_cast<double>(counters[index]->active_ns.load(
                                    std::memory_order_relaxed)) /
               1.0e9;
  }
  json << "],\n  \"perDeviceGBps\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    const unsigned long long device_bytes =
        counters[index]->bytes.load(std::memory_order_relaxed);
    json << (elapsed_seconds > 0.0
                 ? static_cast<double>(device_bytes) / elapsed_seconds / 1e9
                 : 0.0);
  }
  json << "],\n  \"perDeviceWallActiveSec\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    json << summarizeTiming(*counters[index]).wall_active_sec;
  }
  json << "],\n  \"perDeviceWallActiveDuty\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    const double active_seconds =
        summarizeTiming(*counters[index]).wall_active_sec;
    json << (elapsed_seconds > 0.0 ? active_seconds / elapsed_seconds : 0.0);
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
    copybench::writeTimingStats(
        json, summarizeTiming(*counters[index]).submit_interval);
  }
  json << "],\n  \"perDeviceIdleGapMs\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    copybench::writeTimingStats(json, summarizeTiming(*counters[index]).idle_gap);
  }
  json << "],\n  \"aggregateActiveDutyCycle\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    const double active_seconds =
        static_cast<double>(counters[index]->active_ns.load(
                                std::memory_order_relaxed)) /
        1.0e9;
    json << (elapsed_seconds > 0.0 ? active_seconds / elapsed_seconds : 0.0);
  }
  json << "]\n}\n";
  return json.str();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  SharedState state;
  std::vector<std::unique_ptr<Counter>> counters;
  std::vector<std::thread> workers;
  bool measurement_started = false;
  try {
    options = parseOptions(argc, argv);
    state.worker_count = static_cast<int>(options.devices.size());
    int device_count = 0;
    COPYBENCH_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::vector<std::string> device_names;
    device_names.reserve(options.devices.size());
    for (int device : options.devices) {
      if (device < 0 || device >= device_count) {
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
    workers.reserve(options.devices.size());
    for (size_t index = 0; index < options.devices.size(); ++index) {
      counters.emplace_back(std::make_unique<Counter>());
      workers.emplace_back(worker, std::cref(options), options.devices[index],
                           counters[index].get(), &state);
    }

    while (state.ready.load(std::memory_order_acquire) <
               static_cast<int>(options.devices.size()) &&
           !state.failed.load(std::memory_order_acquire) && !g_signal_stop) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (state.failed.load(std::memory_order_acquire) || g_signal_stop) {
      state.stop.store(true, std::memory_order_release);
    } else {
      copybench::writeTextFile(options.ready_file, "ready\n");
      std::cout << "Copy-path background\n"
                << "path=" << pathName(options.path)
                << " devices=" << copybench::joinDevices(options.devices)
                << " requestedBytes=" << options.bytes
                << " workingSetBytes=" << operationBytes(options)
                << " targetGBps=" << options.target_gbps
                << " dutyCycle=" << options.duty_cycle
                << " warmup=" << kWarmupIterations << '\n';
      std::cout.flush();
      state.run.store(true, std::memory_order_release);
      const auto measurement_start = std::chrono::steady_clock::now();
      measurement_started = true;
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
          const double rate = static_cast<double>(current_bytes - last_bytes) /
                              since_report / 1e9;
          std::cout << std::fixed << std::setprecision(3)
                    << "window_bytes=" << (current_bytes - last_bytes)
                    << " time_sec=" << since_report
                    << " aggregateGBps=" << rate << '\n';
          std::cout.flush();
          last_bytes = current_bytes;
          last_report = now;
        }
      }
      state.stop.store(true, std::memory_order_release);
      for (std::thread& worker_thread : workers) {
        if (worker_thread.joinable()) worker_thread.join();
      }
      const auto measurement_end = std::chrono::steady_clock::now();
      const double elapsed_seconds =
          std::chrono::duration<double>(measurement_end - measurement_start)
              .count();
      const unsigned long long bytes_total = totalBytes(counters);
      const double aggregate_gbps =
          elapsed_seconds > 0.0
              ? static_cast<double>(bytes_total) / elapsed_seconds / 1e9
              : 0.0;
      std::cout << std::fixed << std::setprecision(3)
                << "total_bytes=" << bytes_total
                << " elapsed_sec=" << elapsed_seconds
                << " aggregateGBps=" << aggregate_gbps << '\n';
      copybench::writeTextFile(
          options.output,
          makeJson(options, counters, device_names, operationBytes(options),
                   elapsed_seconds, bytes_total));
      measurement_started = false;
    }

    for (std::thread& worker_thread : workers) {
      if (worker_thread.joinable()) worker_thread.join();
    }
    if (state.failed.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(state.error_mutex);
      throw std::runtime_error(state.error.empty() ? "background worker failed"
                                                   : state.error);
    }
    if (!measurement_started) return 0;
    return 0;
  } catch (const std::exception& error) {
    state.stop.store(true, std::memory_order_release);
    state.run.store(true, std::memory_order_release);
    for (std::thread& worker_thread : workers) {
      if (worker_thread.joinable()) worker_thread.join();
    }
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
