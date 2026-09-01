#include "copy_common.cuh"
#include "temporal_stats.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>
#include <sys/sysinfo.h>

namespace {

constexpr size_t kDefaultBytes = 255ULL * 1024ULL * 1024ULL;
constexpr int kDefaultWarmup = 20;
constexpr unsigned long long kDefaultIterations = 100;
constexpr size_t kMaxStoredSamples = 200000;

enum class Mode {
  kD2H,
  kH2D,
  kRelay,
};

struct Edge {
  int src = 0;
  int dst = 1;
};

struct Options {
  Mode mode = Mode::kRelay;
  std::vector<Edge> edges{{0, 1}};
  size_t bytes = kDefaultBytes;
  int warmup = kDefaultWarmup;
  unsigned long long iterations = kDefaultIterations;
  double duty_cycle = 1.0;
  unsigned long long report_ms = 250;
  std::string ready_file;
  std::string start_file;
  std::string stop_file;
  std::string telemetry_file;
  std::string output;
};

struct Counter {
  std::atomic<unsigned long long> operations{0};
  std::atomic<unsigned long long> bytes_completed{0};
  std::atomic<unsigned long long> active_ns{0};
  std::atomic<unsigned long long> latest_begin_ns{0};
  std::atomic<unsigned long long> latest_end_ns{0};
  std::atomic<unsigned long long> latest_d2h_ns{0};
  std::atomic<unsigned long long> latest_transition_gap_ns{0};
  std::atomic<unsigned long long> latest_h2d_ns{0};
  std::atomic<unsigned long long> latest_end_to_end_ns{0};
  unsigned long long first_begin_ns = 0;
  unsigned long long last_end_ns = 0;
  unsigned long long previous_begin_ns = 0;
  bool has_begin = false;
  std::vector<unsigned long long> d2h_ns;
  std::vector<unsigned long long> transition_gap_ns;
  std::vector<unsigned long long> h2d_ns;
  std::vector<unsigned long long> end_to_end_ns;
  std::vector<unsigned long long> interval_ns;
};

struct SharedState {
  std::atomic<int> ready{0};
  std::atomic<int> finished{0};
  std::atomic<bool> run{false};
  std::atomic<bool> stop{false};
  std::atomic<bool> failed{false};
  std::mutex error_mutex;
  std::string error;
};

struct PreflightInfo {
  unsigned long long required_host_bytes = 0;
  unsigned long long available_host_bytes = 0;
  unsigned long long memlock_limit_bytes = 0;
  bool memlock_unlimited = false;
};

struct WorkerResources {
  void* host_buffer = nullptr;
  void* source_buffer = nullptr;
  void* destination_buffer = nullptr;
  cudaStream_t d2h_stream = nullptr;
  cudaStream_t h2d_stream = nullptr;
};

struct Measurement {
  unsigned long long begin_ns = 0;
  unsigned long long end_ns = 0;
  unsigned long long d2h_ns = 0;
  unsigned long long transition_gap_ns = 0;
  unsigned long long h2d_ns = 0;
  unsigned long long end_to_end_ns = 0;
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

const char* modeName(Mode mode) {
  switch (mode) {
    case Mode::kD2H: return "d2h";
    case Mode::kH2D: return "h2d";
    case Mode::kRelay: return "relay";
  }
  return "unknown";
}

void printUsage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --mode=relay|d2h|h2d                  transfer mode\n"
      << "  --edges=0:1,2:3                       source:destination edges\n"
      << "  --size=255M                            bytes per iteration\n"
      << "  --warmup=20                            warmup iterations\n"
      << "  --iterations=100                       measured iterations; 0=forever\n"
      << "  --dutyCycle=1.0                        target active duty in (0,1]\n"
      << "  --readyFile=<path>                     write marker after warmup\n"
      << "  --startFile=<path>                     wait for marker before measuring\n"
      << "  --stopFile=<path>                      stop when marker appears\n"
      << "  --reportMs=250                         telemetry report interval\n"
      << "  --reportSec=2                          legacy report interval in seconds\n"
      << "  --telemetryFile=<path>                 append JSONL counter snapshots\n"
      << "  --output=<path>                        JSON output path\n"
      << "  --help                                 show this message\n"
      << "D2H uses each edge source, H2D uses each edge destination; relay\n"
      << "strictly waits for D2H completion before submitting H2D.\n";
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

unsigned long long parseCount(const std::string& value,
                              const std::string& option_name) {
  const long long parsed = copybench::parseInteger(value, option_name);
  if (parsed < 0) {
    throw std::invalid_argument(option_name + " must be non-negative");
  }
  return static_cast<unsigned long long>(parsed);
}

std::vector<Edge> parseEdges(const std::string& value) {
  std::vector<Edge> edges;
  std::stringstream list(value);
  std::string item;
  while (std::getline(list, item, ',')) {
    const size_t separator = item.find(':');
    if (separator == std::string::npos ||
        item.find(':', separator + 1) != std::string::npos) {
      throw std::invalid_argument(
          "--edges must contain comma-separated source:destination pairs");
    }
    const int src = copybench::parseNonNegativeInt(
        copybench::trim(item.substr(0, separator)), "--edges source");
    const int dst = copybench::parseNonNegativeInt(
        copybench::trim(item.substr(separator + 1)), "--edges destination");
    for (const Edge& existing : edges) {
      if (existing.src == src && existing.dst == dst) {
        throw std::invalid_argument("--edges contains a duplicate edge");
      }
    }
    edges.push_back({src, dst});
  }
  if (edges.empty()) {
    throw std::invalid_argument("--edges cannot be empty");
  }
  return edges;
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
    if (copybench::consumeOption(index, argc, argv, {"--mode"}, &value)) {
      const std::string mode = copybench::lower(value);
      if (mode == "d2h") {
        options.mode = Mode::kD2H;
      } else if (mode == "h2d") {
        options.mode = Mode::kH2D;
      } else if (mode == "relay") {
        options.mode = Mode::kRelay;
      } else {
        throw std::invalid_argument("--mode must be relay, d2h, or h2d");
      }
    } else if (copybench::consumeOption(index, argc, argv,
                                        {"--edges", "--edgeList"}, &value)) {
      options.edges = parseEdges(value);
    } else if (copybench::consumeOption(index, argc, argv, {"--size"},
                                        &value)) {
      options.bytes = copybench::parseSize(value, "--size");
    } else if (copybench::consumeOption(index, argc, argv, {"--warmup"},
                                        &value)) {
      const unsigned long long warmup = parseCount(value, "--warmup");
      if (warmup > static_cast<unsigned long long>(
                       std::numeric_limits<int>::max())) {
        throw std::invalid_argument("--warmup is too large");
      }
      options.warmup = static_cast<int>(warmup);
    } else if (copybench::consumeOption(index, argc, argv, {"--iterations"},
                                        &value)) {
      options.iterations = parseCount(value, "--iterations");
    } else if (copybench::consumeOption(index, argc, argv, {"--dutyCycle"},
                                        &value)) {
      options.duty_cycle = parseDutyCycle(value);
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--readyFile", "--ready-file"},
                   &value)) {
      options.ready_file = value;
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--startFile", "--start-file"},
                   &value)) {
      options.start_file = value;
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--stopFile", "--stop-file"},
                   &value)) {
      options.stop_file = value;
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--reportMs", "--report-ms"},
                   &value)) {
      options.report_ms = static_cast<unsigned long long>(
          copybench::parsePositiveInt(value, "--reportMs"));
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--reportSec", "--report-sec"},
                   &value)) {
      const unsigned long long seconds = static_cast<unsigned long long>(
          copybench::parsePositiveInt(value, "--reportSec"));
      if (seconds > std::numeric_limits<unsigned long long>::max() / 1000ULL) {
        throw std::invalid_argument("--reportSec is too large");
      }
      options.report_ms = seconds * 1000ULL;
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--telemetryFile", "--telemetry-file"},
                   &value)) {
      options.telemetry_file = value;
    } else if (copybench::consumeOption(index, argc, argv, {"--output"},
                                        &value)) {
      options.output = value;
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }

  if (options.mode == Mode::kRelay) {
    for (const Edge& edge : options.edges) {
      if (edge.src == edge.dst) {
        throw std::invalid_argument(
            "relay edges must use distinct source and destination GPUs");
      }
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

void appendSample(std::vector<unsigned long long>* samples,
                  unsigned long long value) {
  if (samples->size() < kMaxStoredSamples) samples->push_back(value);
}

void cleanupWorker(const Edge& edge, Mode mode, WorkerResources* resources) noexcept {
  if (mode == Mode::kD2H) {
    cudaSetDevice(edge.src);
    if (resources->d2h_stream != nullptr) cudaStreamDestroy(resources->d2h_stream);
    if (resources->source_buffer != nullptr) cudaFree(resources->source_buffer);
  } else if (mode == Mode::kH2D) {
    cudaSetDevice(edge.dst);
    if (resources->h2d_stream != nullptr) cudaStreamDestroy(resources->h2d_stream);
    if (resources->destination_buffer != nullptr) {
      cudaFree(resources->destination_buffer);
    }
  } else {
    cudaSetDevice(edge.src);
    if (resources->d2h_stream != nullptr) cudaStreamDestroy(resources->d2h_stream);
    if (resources->source_buffer != nullptr) cudaFree(resources->source_buffer);
    cudaSetDevice(edge.dst);
    if (resources->h2d_stream != nullptr) cudaStreamDestroy(resources->h2d_stream);
    if (resources->destination_buffer != nullptr) {
      cudaFree(resources->destination_buffer);
    }
  }
  if (resources->host_buffer != nullptr) cudaFreeHost(resources->host_buffer);
  *resources = WorkerResources{};
}

unsigned char patternForEdge(const Edge& edge) {
  return static_cast<unsigned char>((edge.src * 37 + edge.dst * 13 + 0x5a) &
                                    0xff);
}

void validateHostBuffer(const void* host_buffer, size_t bytes,
                        unsigned char expected, const Edge& edge) {
  const unsigned char* data = static_cast<const unsigned char*>(host_buffer);
  for (size_t offset = 0; offset < bytes; ++offset) {
    if (data[offset] != expected) {
      std::ostringstream message;
      message << "validation failed for edge " << edge.src << "->" << edge.dst
              << " at offset " << offset << ": expected "
              << static_cast<unsigned int>(expected) << ", actual "
              << static_cast<unsigned int>(data[offset]);
      throw std::runtime_error(message.str());
    }
  }
}

void validateDeviceBuffer(int device, cudaStream_t stream, void* device_buffer,
                          void* host_buffer, size_t bytes,
                          unsigned char expected, const Edge& edge) {
  COPYBENCH_CUDA_CHECK(cudaSetDevice(device));
  COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(host_buffer, device_buffer, bytes,
                                       cudaMemcpyDeviceToHost, stream));
  COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
  validateHostBuffer(host_buffer, bytes, expected, edge);
}

void initializeResources(const Options& options, const Edge& edge,
                         WorkerResources* resources) {
  const unsigned char expected = patternForEdge(edge);
  COPYBENCH_CUDA_CHECK(cudaMallocHost(&resources->host_buffer, options.bytes));
  std::memset(resources->host_buffer, expected, options.bytes);

  if (options.mode == Mode::kD2H) {
    COPYBENCH_CUDA_CHECK(cudaSetDevice(edge.src));
    COPYBENCH_CUDA_CHECK(cudaMalloc(&resources->source_buffer, options.bytes));
    COPYBENCH_CUDA_CHECK(cudaMemset(resources->source_buffer, expected,
                                    options.bytes));
    COPYBENCH_CUDA_CHECK(cudaStreamCreate(&resources->d2h_stream));
  } else if (options.mode == Mode::kH2D) {
    COPYBENCH_CUDA_CHECK(cudaSetDevice(edge.dst));
    COPYBENCH_CUDA_CHECK(cudaMalloc(&resources->destination_buffer,
                                    options.bytes));
    COPYBENCH_CUDA_CHECK(cudaMemset(resources->destination_buffer, 0,
                                    options.bytes));
    COPYBENCH_CUDA_CHECK(cudaStreamCreate(&resources->h2d_stream));
  } else {
    COPYBENCH_CUDA_CHECK(cudaSetDevice(edge.src));
    COPYBENCH_CUDA_CHECK(cudaMalloc(&resources->source_buffer, options.bytes));
    COPYBENCH_CUDA_CHECK(cudaMemset(resources->source_buffer, expected,
                                    options.bytes));
    COPYBENCH_CUDA_CHECK(cudaStreamCreate(&resources->d2h_stream));

    COPYBENCH_CUDA_CHECK(cudaSetDevice(edge.dst));
    COPYBENCH_CUDA_CHECK(cudaMalloc(&resources->destination_buffer,
                                    options.bytes));
    COPYBENCH_CUDA_CHECK(cudaMemset(resources->destination_buffer, 0,
                                    options.bytes));
    COPYBENCH_CUDA_CHECK(cudaStreamCreate(&resources->h2d_stream));
  }
}

Measurement executeOne(const Options& options, const Edge& edge,
                       WorkerResources* resources) {
  Measurement measurement;
  if (options.mode == Mode::kD2H) {
    COPYBENCH_CUDA_CHECK(cudaSetDevice(edge.src));
    const unsigned long long begin = steadyNowNs();
    COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
        resources->host_buffer, resources->source_buffer, options.bytes,
        cudaMemcpyDeviceToHost, resources->d2h_stream));
    COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(resources->d2h_stream));
    const unsigned long long end = steadyNowNs();
    measurement.begin_ns = begin;
    measurement.end_ns = end;
    measurement.d2h_ns = end - begin;
    measurement.end_to_end_ns = measurement.d2h_ns;
    return measurement;
  }

  if (options.mode == Mode::kH2D) {
    COPYBENCH_CUDA_CHECK(cudaSetDevice(edge.dst));
    const unsigned long long begin = steadyNowNs();
    COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
        resources->destination_buffer, resources->host_buffer, options.bytes,
        cudaMemcpyHostToDevice, resources->h2d_stream));
    COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(resources->h2d_stream));
    const unsigned long long end = steadyNowNs();
    measurement.begin_ns = begin;
    measurement.end_ns = end;
    measurement.h2d_ns = end - begin;
    measurement.end_to_end_ns = measurement.h2d_ns;
    return measurement;
  }

  COPYBENCH_CUDA_CHECK(cudaSetDevice(edge.src));
  const unsigned long long d2h_begin = steadyNowNs();
  COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
      resources->host_buffer, resources->source_buffer, options.bytes,
      cudaMemcpyDeviceToHost, resources->d2h_stream));
  COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(resources->d2h_stream));
  const unsigned long long d2h_end = steadyNowNs();
  COPYBENCH_CUDA_CHECK(cudaSetDevice(edge.dst));
  const unsigned long long h2d_begin = steadyNowNs();
  COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
      resources->destination_buffer, resources->host_buffer, options.bytes,
      cudaMemcpyHostToDevice, resources->h2d_stream));
  COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(resources->h2d_stream));
  const unsigned long long h2d_end = steadyNowNs();
  measurement.begin_ns = d2h_begin;
  measurement.end_ns = h2d_end;
  measurement.d2h_ns = d2h_end - d2h_begin;
  measurement.transition_gap_ns = h2d_begin - d2h_end;
  measurement.h2d_ns = h2d_end - h2d_begin;
  measurement.end_to_end_ns = h2d_end - d2h_begin;
  return measurement;
}

void validateAfterWarmup(const Options& options, const Edge& edge,
                         WorkerResources* resources) {
  const unsigned char expected = patternForEdge(edge);
  if (options.mode == Mode::kD2H) {
    validateHostBuffer(resources->host_buffer, options.bytes, expected, edge);
  } else {
    validateDeviceBuffer(edge.dst, resources->h2d_stream,
                         resources->destination_buffer, resources->host_buffer,
                         options.bytes, expected, edge);
    std::memset(resources->host_buffer, expected, options.bytes);
  }
}

void recordMeasurement(Counter* counter, const Measurement& measurement,
                       size_t bytes) {
  const unsigned long long begin = measurement.begin_ns;
  if (!counter->has_begin) {
    counter->first_begin_ns = begin;
    counter->has_begin = true;
  } else {
    appendSample(&counter->interval_ns, begin - counter->previous_begin_ns);
  }
  counter->previous_begin_ns = begin;
  counter->last_end_ns = measurement.end_ns;
  counter->latest_begin_ns.store(measurement.begin_ns, std::memory_order_relaxed);
  counter->latest_end_ns.store(measurement.end_ns, std::memory_order_relaxed);
  counter->latest_d2h_ns.store(measurement.d2h_ns, std::memory_order_relaxed);
  counter->latest_transition_gap_ns.store(measurement.transition_gap_ns,
                                          std::memory_order_relaxed);
  counter->latest_h2d_ns.store(measurement.h2d_ns, std::memory_order_relaxed);
  counter->latest_end_to_end_ns.store(measurement.end_to_end_ns,
                                      std::memory_order_relaxed);
  counter->operations.fetch_add(1, std::memory_order_relaxed);
  counter->bytes_completed.fetch_add(static_cast<unsigned long long>(bytes),
                                     std::memory_order_relaxed);
  counter->active_ns.fetch_add(measurement.end_to_end_ns,
                               std::memory_order_relaxed);
  appendSample(&counter->d2h_ns, measurement.d2h_ns);
  appendSample(&counter->transition_gap_ns, measurement.transition_gap_ns);
  appendSample(&counter->h2d_ns, measurement.h2d_ns);
  appendSample(&counter->end_to_end_ns, measurement.end_to_end_ns);
}

void worker(const Options& options, const Edge& edge, Counter* counter,
            SharedState* state) {
  WorkerResources resources;
  try {
    int device_count = 0;
    COPYBENCH_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (edge.src >= device_count || edge.dst >= device_count) {
      std::ostringstream message;
      message << "edge " << edge.src << "->" << edge.dst
              << " exceeds CUDA device count " << device_count;
      throw std::invalid_argument(message.str());
    }

    initializeResources(options, edge, &resources);
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      executeOne(options, edge, &resources);
    }
    validateAfterWarmup(options, edge, &resources);

    state->ready.fetch_add(1, std::memory_order_acq_rel);
    while (!state->run.load(std::memory_order_acquire) &&
           !state->stop.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (!state->stop.load(std::memory_order_acquire)) {
      unsigned long long iteration = 0;
      while (!state->stop.load(std::memory_order_acquire) &&
             (options.iterations == 0 || iteration < options.iterations)) {
        const Measurement measurement = executeOne(options, edge, &resources);
        recordMeasurement(counter, measurement, options.bytes);
        ++iteration;
        if (options.duty_cycle < 1.0) {
          const double active_seconds =
              static_cast<double>(measurement.end_to_end_ns) / 1.0e9;
          const double idle_seconds =
              active_seconds * (1.0 / options.duty_cycle - 1.0);
          std::this_thread::sleep_for(
              std::chrono::duration<double>(idle_seconds));
        }
      }
      validateAfterWarmup(options, edge, &resources);
    }
  } catch (const std::exception& error) {
    setWorkerFailure(state, error);
  }
  cleanupWorker(edge, options.mode, &resources);
  state->finished.fetch_add(1, std::memory_order_acq_rel);
}

unsigned long long totalBytes(
    const std::vector<std::unique_ptr<Counter>>& counters) {
  unsigned long long total = 0;
  for (const auto& counter : counters) {
    total += counter->bytes_completed.load(std::memory_order_relaxed);
  }
  return total;
}

void writeTelemetryEvent(std::ostream* telemetry, const char* event,
                         unsigned long long timestamp_ns) {
  if (telemetry == nullptr || !telemetry->good()) return;
  *telemetry << "{\"event\":\"" << event
             << "\",\"timestampNs\":" << timestamp_ns << "}\n";
  telemetry->flush();
}

void writeTelemetrySnapshot(
    std::ostream* telemetry, unsigned long long timestamp_ns,
    const std::vector<Edge>& edges,
    const std::vector<std::unique_ptr<Counter>>& counters) {
  if (telemetry == nullptr || !telemetry->good()) return;
  unsigned long long aggregate_operations = 0;
  unsigned long long aggregate_bytes = 0;
  *telemetry << "{\"event\":\"snapshot\",\"timestampNs\":"
             << timestamp_ns << ",\"perEdge\":[";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) *telemetry << ',';
    const Counter& counter = *counters[index];
    const unsigned long long operations =
        counter.operations.load(std::memory_order_relaxed);
    const unsigned long long bytes =
        counter.bytes_completed.load(std::memory_order_relaxed);
    aggregate_operations += operations;
    aggregate_bytes += bytes;
    *telemetry << "{\"src\":" << edges[index].src
               << ",\"dst\":" << edges[index].dst
               << ",\"operations\":" << operations
               << ",\"bytesCompleted\":" << bytes
               << ",\"latest\":{";
    *telemetry << "\"beginNs\":"
               << counter.latest_begin_ns.load(std::memory_order_relaxed)
               << ",\"endNs\":"
               << counter.latest_end_ns.load(std::memory_order_relaxed)
               << ",\"d2hMs\":"
               << static_cast<double>(counter.latest_d2h_ns.load(
                                          std::memory_order_relaxed)) /
                      1.0e6
               << ",\"transitionGapMs\":"
               << static_cast<double>(counter.latest_transition_gap_ns.load(
                                          std::memory_order_relaxed)) /
                      1.0e6
               << ",\"h2dMs\":"
               << static_cast<double>(counter.latest_h2d_ns.load(
                                          std::memory_order_relaxed)) /
                      1.0e6
               << ",\"endToEndMs\":"
               << static_cast<double>(counter.latest_end_to_end_ns.load(
                                          std::memory_order_relaxed)) /
                      1.0e6
               << "}}";
  }
  *telemetry << "],\"aggregateOperations\":" << aggregate_operations
             << ",\"aggregateBytesCompleted\":" << aggregate_bytes
             << "}\n";
  telemetry->flush();
}

PreflightInfo checkHostBudget(const Options& options) {
  if (options.edges.size() >
      std::numeric_limits<unsigned long long>::max() / options.bytes) {
    throw std::invalid_argument("total pinned host allocation overflows");
  }
  PreflightInfo info;
  info.required_host_bytes =
      static_cast<unsigned long long>(options.bytes * options.edges.size());

  struct sysinfo system_info {};
  if (sysinfo(&system_info) == 0) {
    info.available_host_bytes =
        static_cast<unsigned long long>(system_info.freeram) *
        static_cast<unsigned long long>(system_info.mem_unit);
    if (info.required_host_bytes > info.available_host_bytes) {
      throw std::runtime_error("pinned host allocation exceeds available memory");
    }
  }

  struct rlimit memlock_limit {};
  if (getrlimit(RLIMIT_MEMLOCK, &memlock_limit) == 0) {
    if (memlock_limit.rlim_cur == RLIM_INFINITY) {
      info.memlock_unlimited = true;
    } else {
      info.memlock_limit_bytes =
          static_cast<unsigned long long>(memlock_limit.rlim_cur);
      if (info.required_host_bytes > info.memlock_limit_bytes) {
        std::cerr << "WARNING: requested pinned host bytes "
                  << info.required_host_bytes
                  << " exceed RLIMIT_MEMLOCK " << info.memlock_limit_bytes
                  << "; CUDA allocation will determine whether this is allowed\n";
      }
    }
  }
  return info;
}

void writeStats(std::ostringstream* json,
                const std::vector<unsigned long long>& samples) {
  copybench::writeTimingStats(*json, copybench::summarizeNanoseconds(samples));
}

double elapsedSeconds(const Counter& counter) {
  if (!counter.has_begin || counter.last_end_ns <= counter.first_begin_ns) {
    return 0.0;
  }
  return static_cast<double>(counter.last_end_ns - counter.first_begin_ns) /
         1.0e9;
}

std::string makeJson(const Options& options, const PreflightInfo& preflight,
                     const std::vector<std::unique_ptr<Counter>>& counters,
                     unsigned long long ready_ns,
                     unsigned long long measurement_start_ns,
                     unsigned long long measurement_end_ns) {
  std::ostringstream json;
  json << std::fixed << std::setprecision(9);
  json << "{\n"
       << "  \"program\": \"host_relay\",\n"
       << "  \"mode\": " << copybench::jsonQuote(modeName(options.mode))
       << ",\n"
       << "  \"bytesPerIteration\": " << options.bytes << ",\n"
       << "  \"warmup\": " << options.warmup << ",\n"
       << "  \"iterationsRequested\": " << options.iterations << ",\n"
       << "  \"dutyCycleTarget\": " << options.duty_cycle << ",\n"
       << "  \"reportIntervalMs\": " << options.report_ms << ",\n"
       << "  \"telemetryFile\": "
       << copybench::jsonQuote(options.telemetry_file) << ",\n"
       << "  \"readyTimestampNs\": " << ready_ns << ",\n"
       << "  \"measurementStartNs\": " << measurement_start_ns << ",\n"
       << "  \"measurementEndNs\": " << measurement_end_ns << ",\n"
       << "  \"hostBufferBytesTotal\": " << preflight.required_host_bytes
       << ",\n"
       << "  \"availableHostBytesAtStart\": "
       << preflight.available_host_bytes << ",\n"
       << "  \"memlockUnlimited\": "
       << (preflight.memlock_unlimited ? "true" : "false") << ",\n"
       << "  \"memlockLimitBytes\": " << preflight.memlock_limit_bytes
       << ",\n"
       << "  \"edges\": [";
  for (size_t index = 0; index < options.edges.size(); ++index) {
    if (index != 0) json << ", ";
    json << "{\"src\": " << options.edges[index].src
         << ", \"dst\": " << options.edges[index].dst << "}";
  }
  json << "],\n  \"perEdge\": [";

  unsigned long long aggregate_bytes = 0;
  double aggregate_elapsed = 0.0;
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ",";
    const Counter& counter = *counters[index];
    const double elapsed = elapsedSeconds(counter);
    const double useful_gbps =
        elapsed > 0.0 ? static_cast<double>(counter.bytes_completed.load(
                             std::memory_order_relaxed)) /
                           elapsed / 1.0e9
                       : 0.0;
    const double traffic_gbps = useful_gbps *
                                (options.mode == Mode::kRelay ? 2.0 : 1.0);
    aggregate_bytes +=
        counter.bytes_completed.load(std::memory_order_relaxed);
    aggregate_elapsed = std::max(aggregate_elapsed, elapsed);
    json << "\n    {\n"
         << "      \"src\": " << options.edges[index].src << ",\n"
         << "      \"dst\": " << options.edges[index].dst << ",\n"
         << "      \"operations\": "
         << counter.operations.load(std::memory_order_relaxed) << ",\n"
         << "      \"bytesCompleted\": "
         << counter.bytes_completed.load(std::memory_order_relaxed) << ",\n"
         << "      \"storedSamples\": " << counter.end_to_end_ns.size()
         << ",\n"
         << "      \"d2hMs\": ";
    writeStats(&json, counter.d2h_ns);
    json << ",\n      \"transitionGapMs\": ";
    writeStats(&json, counter.transition_gap_ns);
    json << ",\n      \"h2dMs\": ";
    writeStats(&json, counter.h2d_ns);
    json << ",\n      \"endToEndMs\": ";
    writeStats(&json, counter.end_to_end_ns);
    json << ",\n      \"iterationIntervalMs\": ";
    writeStats(&json, counter.interval_ns);
    json << ",\n"
         << "      \"wallActiveSec\": "
         << static_cast<double>(counter.active_ns.load(
                                    std::memory_order_relaxed)) /
                1.0e9
         << ",\n"
         << "      \"elapsedSec\": " << elapsed << ",\n"
         << "      \"actualDuty\": "
         << (elapsed > 0.0
                 ? static_cast<double>(counter.active_ns.load(
                       std::memory_order_relaxed)) /
                       static_cast<double>(counter.last_end_ns -
                                           counter.first_begin_ns)
                 : 0.0)
         << ",\n"
         << "      \"usefulGBps\": " << useful_gbps << ",\n"
         << "      \"trafficGBps\": " << traffic_gbps << "\n"
         << "    }";
  }

  const double aggregate_useful_gbps =
      aggregate_elapsed > 0.0
          ? static_cast<double>(aggregate_bytes) / aggregate_elapsed / 1.0e9
          : 0.0;
  json << "\n  ],\n"
       << "  \"aggregateBytesCompleted\": " << aggregate_bytes << ",\n"
       << "  \"aggregateElapsedSec\": " << aggregate_elapsed << ",\n"
       << "  \"aggregateUsefulGBps\": " << aggregate_useful_gbps << ",\n"
       << "  \"aggregateTrafficGBps\": "
       << aggregate_useful_gbps * (options.mode == Mode::kRelay ? 2.0 : 1.0)
       << "\n}\n";
  return json.str();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  SharedState state;
  std::vector<std::unique_ptr<Counter>> counters;
  std::vector<std::thread> workers;
  PreflightInfo preflight;
  std::ofstream telemetry;
  unsigned long long ready_ns = 0;
  unsigned long long measurement_start_ns = 0;
  unsigned long long measurement_end_ns = 0;
  try {
    options = parseOptions(argc, argv);
    preflight = checkHostBudget(options);

    if (!options.telemetry_file.empty()) {
      telemetry.open(options.telemetry_file, std::ios::out | std::ios::trunc);
      if (!telemetry.is_open()) {
        throw std::runtime_error("failed to open telemetry file: " +
                                 options.telemetry_file);
      }
      writeTelemetryEvent(&telemetry, "process_start", steadyNowNs());
    }

    int device_count = 0;
    COPYBENCH_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    for (const Edge& edge : options.edges) {
      if (edge.src >= device_count || edge.dst >= device_count) {
        std::ostringstream message;
        message << "edge " << edge.src << "->" << edge.dst
                << " exceeds CUDA device count " << device_count;
        throw std::invalid_argument(message.str());
      }
    }

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    counters.reserve(options.edges.size());
    for (size_t index = 0; index < options.edges.size(); ++index) {
      counters.emplace_back(std::make_unique<Counter>());
    }
    workers.reserve(options.edges.size());
    for (size_t index = 0; index < options.edges.size(); ++index) {
      workers.emplace_back(worker, std::cref(options), options.edges[index],
                           counters[index].get(), &state);
    }

    while (state.ready.load(std::memory_order_acquire) <
               static_cast<int>(options.edges.size()) &&
           !state.failed.load(std::memory_order_acquire) && !g_signal_stop) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (g_signal_stop || state.failed.load(std::memory_order_acquire)) {
      state.stop.store(true, std::memory_order_release);
    } else {
      ready_ns = steadyNowNs();
      copybench::writeTextFile(options.ready_file, "ready\n");
      writeTelemetryEvent(&telemetry, "ready", ready_ns);
      std::cout << "CUDA CE host relay\n"
                << "mode=" << modeName(options.mode)
                << " edges=";
      for (size_t index = 0; index < options.edges.size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << options.edges[index].src << "->" << options.edges[index].dst;
      }
      std::cout << " bytes=" << options.bytes << " warmup=" << options.warmup
                << " iterations=" << options.iterations
                << " dutyCycleTarget=" << options.duty_cycle << '\n';
      std::cout.flush();

      while (!options.start_file.empty() &&
             !copybench::fileExists(options.start_file) && !g_signal_stop &&
             !copybench::fileExists(options.stop_file)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      if (g_signal_stop || copybench::fileExists(options.stop_file)) {
        state.stop.store(true, std::memory_order_release);
      } else {
        measurement_start_ns = steadyNowNs();
        writeTelemetryEvent(&telemetry, "measurement_start",
                            measurement_start_ns);
        writeTelemetrySnapshot(&telemetry, measurement_start_ns, options.edges,
                               counters);
        state.run.store(true, std::memory_order_release);
      }
    }

    const auto report_start = std::chrono::steady_clock::now();
    auto last_report = report_start;
    unsigned long long last_bytes = 0;
    while (state.finished.load(std::memory_order_acquire) <
               static_cast<int>(options.edges.size()) &&
           !state.failed.load(std::memory_order_acquire)) {
      if (g_signal_stop || copybench::fileExists(options.stop_file)) {
        state.stop.store(true, std::memory_order_release);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      const auto now = std::chrono::steady_clock::now();
      const double since_report =
          std::chrono::duration<double>(now - last_report).count();
      if (since_report * 1000.0 >= static_cast<double>(options.report_ms)) {
        const unsigned long long current_bytes = totalBytes(counters);
        const double aggregate_gbps =
            static_cast<double>(current_bytes - last_bytes) / since_report /
            1.0e9;
        const unsigned long long snapshot_ns = steadyNowNs();
        writeTelemetrySnapshot(&telemetry, snapshot_ns, options.edges, counters);
        std::cout << std::fixed << std::setprecision(3)
                  << "window_bytes=" << (current_bytes - last_bytes)
                  << " time_sec=" << since_report
                  << " aggregateUsefulGBps=" << aggregate_gbps << '\n';
        std::cout.flush();
        last_bytes = current_bytes;
        last_report = now;
      }
    }
    state.stop.store(true, std::memory_order_release);
    for (std::thread& thread : workers) {
      if (thread.joinable()) thread.join();
    }

    measurement_end_ns = steadyNowNs();
    if (measurement_start_ns != 0) {
      writeTelemetrySnapshot(&telemetry, measurement_end_ns, options.edges,
                             counters);
      writeTelemetryEvent(&telemetry, "measurement_end", measurement_end_ns);
    }

    if (state.failed.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(state.error_mutex);
      throw std::runtime_error(state.error.empty() ? "worker failed"
                                                   : state.error);
    }

    const std::string json = makeJson(options, preflight, counters, ready_ns,
                                      measurement_start_ns,
                                      measurement_end_ns);
    copybench::writeTextFile(options.output, json);
    std::cout << "completed_bytes=" << totalBytes(counters) << '\n';
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
