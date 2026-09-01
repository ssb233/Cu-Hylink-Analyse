#include "copy_common.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
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

enum class Topology {
  kSingleTwoCopy,
  kEdgeIndependent,
};

enum class Background {
  kNone,
  kD2H,
};

struct Options {
  std::vector<int> devices{0, 1, 2};
  Topology topology = Topology::kSingleTwoCopy;
  Background background = Background::kNone;
  size_t d2d_bytes = kDefaultBytes;
  size_t background_bytes = kDefaultBytes;
  int repeats = 300;
  std::string output;
};

struct Edge {
  int source_index = -1;
  int destination_index = -1;
};

struct DeviceResources {
  int device = -1;
  void* source = nullptr;
};

struct StreamResource {
  int device = -1;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
};

struct SourceResult {
  int device = -1;
  int outgoing = 0;
  float elapsed_ms = 0.0f;
  double gbps = 0.0;
};

struct BackgroundCounter {
  std::atomic<unsigned long long> bytes{0};
};

struct BackgroundState {
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> measure{false};
  std::atomic<bool> stop{false};
  std::atomic<bool> failed{false};
  std::mutex error_mutex;
  std::string error;
};

const char* topologyName(Topology topology) {
  return topology == Topology::kSingleTwoCopy ? "single-two-copy"
                                               : "edge-independent";
}

const char* backgroundName(Background background) {
  return background == Background::kD2H ? "d2h" : "none";
}

void printUsage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --deviceList=0,1,2                    GPUs used by this process\n"
      << "  --topology=single-two-copy|edge-independent\n"
      << "  --background=none|d2h                 same-process background mode\n"
      << "  --size=255M                            D2D P2P size\n"
      << "  --backgroundSize=255M                  D2H size\n"
      << "  --repeats=300                          measured D2D repetitions\n"
      << "  --output=<path>                        JSON output path\n"
      << "  --help                                 show this message\n"
      << "Defaults: deviceList=0,1,2, topology=single-two-copy, "
         "background=none, size=255M, backgroundSize=255M, repeats=300, "
         "warmup=10 (fixed)\n";
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
                                 {"--deviceList", "--device-list"},
                                 &value)) {
      options.devices = copybench::parseDeviceList(value, "--deviceList");
    } else if (copybench::consumeOption(index, argc, argv, {"--topology"},
                                        &value)) {
      const std::string topology = copybench::lower(value);
      if (topology == "single-two-copy" || topology == "single") {
        options.topology = Topology::kSingleTwoCopy;
      } else if (topology == "edge-independent" || topology == "edge") {
        options.topology = Topology::kEdgeIndependent;
      } else {
        throw std::invalid_argument(
            "--topology must be single-two-copy or edge-independent");
      }
    } else if (copybench::consumeOption(index, argc, argv, {"--background"},
                                        &value)) {
      const std::string background = copybench::lower(value);
      if (background == "none" || background == "clean") {
        options.background = Background::kNone;
      } else if (background == "d2h") {
        options.background = Background::kD2H;
      } else {
        throw std::invalid_argument("--background must be none or d2h");
      }
    } else if (copybench::consumeOption(index, argc, argv, {"--size"},
                                        &value)) {
      options.d2d_bytes = copybench::parseSize(value, "--size");
    } else if (copybench::consumeOption(index, argc, argv,
                                        {"--backgroundSize"}, &value)) {
      options.background_bytes =
          copybench::parseSize(value, "--backgroundSize");
    } else if (copybench::consumeOption(index, argc, argv, {"--repeats"},
                                        &value)) {
      options.repeats = copybench::parsePositiveInt(value, "--repeats");
    } else if (copybench::consumeOption(index, argc, argv, {"--output"},
                                        &value)) {
      options.output = value;
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }

  if (options.devices.size() < 2) {
    throw std::invalid_argument("at least two devices are required");
  }
  return options;
}

std::vector<Edge> makeEdges(const Options& options) {
  std::vector<Edge> edges;
  const int count = static_cast<int>(options.devices.size());
  edges.reserve(count * (count - 1));
  for (int source = 0; source < count; ++source) {
    for (int destination = 0; destination < count; ++destination) {
      if (source != destination) edges.push_back({source, destination});
    }
  }
  return edges;
}

int streamIndexForEdge(const Options& options, const Edge& edge,
                       int edge_index) {
  if (options.topology == Topology::kEdgeIndependent) return edge_index;
  return edge.source_index;
}

void enablePeerAccess(int source_device, int destination_device) {
  int can_access = 0;
  COPYBENCH_CUDA_CHECK(
      cudaDeviceCanAccessPeer(&can_access, source_device, destination_device));
  if (!can_access) {
    std::ostringstream message;
    message << "GPU " << source_device << " cannot access GPU "
            << destination_device;
    throw std::runtime_error(message.str());
  }
  COPYBENCH_CUDA_CHECK(cudaSetDevice(source_device));
  const cudaError_t status = cudaDeviceEnablePeerAccess(destination_device, 0);
  if (status == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
  } else {
    COPYBENCH_CUDA_CHECK(status);
  }
}

void cleanupD2D(std::vector<DeviceResources>& resources,
               std::vector<StreamResource>& streams,
               std::vector<std::vector<void*>>& destinations) noexcept {
  for (StreamResource& resource : streams) {
    if (resource.device < 0) continue;
    cudaSetDevice(resource.device);
    if (resource.stream != nullptr) cudaStreamSynchronize(resource.stream);
  }
  for (StreamResource& resource : streams) {
    if (resource.device < 0) continue;
    cudaSetDevice(resource.device);
    if (resource.start != nullptr) cudaEventDestroy(resource.start);
    if (resource.stop != nullptr) cudaEventDestroy(resource.stop);
    if (resource.stream != nullptr) cudaStreamDestroy(resource.stream);
    resource.start = nullptr;
    resource.stop = nullptr;
    resource.stream = nullptr;
  }
  for (DeviceResources& resource : resources) {
    if (resource.device < 0) continue;
    cudaSetDevice(resource.device);
    if (resource.source != nullptr) cudaFree(resource.source);
    resource.source = nullptr;
  }
  for (size_t destination_index = 0; destination_index < destinations.size();
       ++destination_index) {
    if (resources[destination_index].device < 0) continue;
    cudaSetDevice(resources[destination_index].device);
    for (void*& destination : destinations[destination_index]) {
      if (destination != nullptr) cudaFree(destination);
      destination = nullptr;
    }
  }
}

void setBackgroundFailure(BackgroundState* state,
                          const std::exception& error) {
  {
    std::lock_guard<std::mutex> lock(state->error_mutex);
    if (state->error.empty()) state->error = error.what();
  }
  state->failed.store(true, std::memory_order_release);
  state->stop.store(true, std::memory_order_release);
}

void cleanupBackgroundWorker(int device, void* host_buffer,
                             void* device_buffer, cudaStream_t stream) noexcept {
  cudaSetDevice(device);
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (device_buffer != nullptr) cudaFree(device_buffer);
  if (host_buffer != nullptr) cudaFreeHost(host_buffer);
}

void backgroundWorker(int device, size_t bytes, BackgroundCounter* counter,
                      BackgroundState* state) {
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
      COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
          host_buffer, device_buffer, bytes, cudaMemcpyDeviceToHost, stream));
      COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    state->ready.fetch_add(1, std::memory_order_acq_rel);
    while (!state->start.load(std::memory_order_acquire) &&
           !state->stop.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    while (!state->stop.load(std::memory_order_acquire)) {
      COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
          host_buffer, device_buffer, bytes, cudaMemcpyDeviceToHost, stream));
      COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
      if (state->measure.load(std::memory_order_acquire)) {
        counter->bytes.fetch_add(bytes, std::memory_order_relaxed);
      }
    }
  } catch (const std::exception& error) {
    setBackgroundFailure(state, error);
  }
  cleanupBackgroundWorker(device, host_buffer, device_buffer, stream);
}

void stopBackgroundWorkers(BackgroundState* state,
                           std::vector<std::thread>* workers) noexcept {
  state->start.store(true, std::memory_order_release);
  state->stop.store(true, std::memory_order_release);
  for (std::thread& worker : *workers) {
    if (worker.joinable()) worker.join();
  }
}

void issueRound(const Options& options,
                const std::vector<DeviceResources>& resources,
                const std::vector<StreamResource>& streams,
                const std::vector<std::vector<void*>>& destinations,
                const std::vector<Edge>& edges) {
  for (size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    const Edge& edge = edges[edge_index];
    const int source_device = options.devices[edge.source_index];
    const int destination_device = options.devices[edge.destination_index];
    const int stream_index =
        streamIndexForEdge(options, edge, static_cast<int>(edge_index));
    COPYBENCH_CUDA_CHECK(cudaSetDevice(source_device));
    COPYBENCH_CUDA_CHECK(cudaMemcpyPeerAsync(
        destinations[edge.destination_index][edge.source_index],
        destination_device, resources[edge.source_index].source, source_device,
        options.d2d_bytes, streams[stream_index].stream));
  }
}

unsigned long long backgroundTotal(
    const std::vector<std::unique_ptr<BackgroundCounter>>& counters) {
  unsigned long long total = 0;
  for (const auto& counter : counters) {
    total += counter->bytes.load(std::memory_order_relaxed);
  }
  return total;
}

std::string makeJson(
    const Options& options, const std::vector<SourceResult>& source_results,
    double d2d_gbps, const std::vector<std::string>& device_names,
    const std::vector<std::unique_ptr<BackgroundCounter>>& counters,
    double background_elapsed_sec, unsigned long long background_bytes) {
  std::ostringstream json;
  json << std::fixed << std::setprecision(6);
  json << "{\n"
       << "  \"program\": \"d2d_with_background_bw\",\n"
       << "  \"contextMode\": \"same-process\",\n"
       << "  \"topology\": "
       << copybench::jsonQuote(topologyName(options.topology)) << ",\n"
       << "  \"background\": "
       << copybench::jsonQuote(backgroundName(options.background)) << ",\n"
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
       << "  \"streamMode\": "
       << copybench::jsonQuote(options.topology == Topology::kSingleTwoCopy
                                   ? "per-source"
                                   : "per-edge")
       << ",\n"
       << "  \"streamsPerSource\": "
       << (options.topology == Topology::kSingleTwoCopy ? 1 : 3) << ",\n"
       << "  \"repeats\": " << options.repeats << ",\n"
       << "  \"warmup\": " << kWarmupIterations << ",\n"
       << "  \"bytesPerMemcpy\": " << options.d2d_bytes << ",\n"
       << "  \"backgroundBytesPerMemcpy\": "
       << options.background_bytes << ",\n"
       << "  \"aggregateGBps\": " << d2d_gbps << ",\n"
       << "  \"backgroundTotalBytes\": " << background_bytes << ",\n"
       << "  \"backgroundElapsedSec\": " << background_elapsed_sec << ",\n"
       << "  \"backgroundAggregateGBps\": "
       << (background_elapsed_sec > 0.0
               ? static_cast<double>(background_bytes) /
                     background_elapsed_sec / 1e9
               : 0.0)
       << ",\n"
       << "  \"backgroundPerDeviceBytes\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    json << counters[index]->bytes.load(std::memory_order_relaxed);
  }
  json << "],\n"
       << "  \"backgroundPerDeviceGBps\": [";
  for (size_t index = 0; index < counters.size(); ++index) {
    if (index != 0) json << ", ";
    const unsigned long long bytes =
        counters[index]->bytes.load(std::memory_order_relaxed);
    json << (background_elapsed_sec > 0.0
                 ? static_cast<double>(bytes) / background_elapsed_sec / 1e9
                 : 0.0);
  }
  json << "],\n"
       << "  \"sourceResults\": [\n";
  for (size_t index = 0; index < source_results.size(); ++index) {
    if (index != 0) json << ",\n";
    json << "    {\"device\": " << source_results[index].device
         << ", \"outgoing\": " << source_results[index].outgoing
         << ", \"elapsedMs\": " << source_results[index].elapsed_ms
         << ", \"GBps\": " << source_results[index].gbps << '}';
  }
  json << "\n  ]\n}\n";
  return json.str();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::vector<DeviceResources> resources;
  std::vector<StreamResource> streams;
  std::vector<std::vector<void*>> destinations;
  BackgroundState background_state;
  std::vector<std::unique_ptr<BackgroundCounter>> counters;
  std::vector<std::thread> background_workers;
  bool background_started = false;

  try {
    options = parseOptions(argc, argv);
    int device_count = 0;
    COPYBENCH_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    for (int device : options.devices) {
      if (device < 0 || device >= device_count) {
        std::ostringstream message;
        message << "requested GPU " << device << ", but only "
                << device_count << " CUDA devices are available";
        throw std::invalid_argument(message.str());
      }
    }

    resources.resize(options.devices.size());
    destinations.assign(options.devices.size(),
                        std::vector<void*>(options.devices.size(), nullptr));
    for (size_t index = 0; index < options.devices.size(); ++index) {
      resources[index].device = options.devices[index];
    }

    const std::vector<Edge> edges = makeEdges(options);
    for (const Edge& edge : edges) {
      enablePeerAccess(options.devices[edge.source_index],
                       options.devices[edge.destination_index]);
    }

    for (DeviceResources& resource : resources) {
      COPYBENCH_CUDA_CHECK(cudaSetDevice(resource.device));
      COPYBENCH_CUDA_CHECK(cudaMalloc(&resource.source, options.d2d_bytes));
      COPYBENCH_CUDA_CHECK(
          cudaMemset(resource.source, resource.device & 0xff, options.d2d_bytes));
    }
    for (size_t destination_index = 0;
         destination_index < options.devices.size(); ++destination_index) {
      COPYBENCH_CUDA_CHECK(cudaSetDevice(options.devices[destination_index]));
      for (size_t source_index = 0; source_index < options.devices.size();
           ++source_index) {
        if (source_index == destination_index) continue;
        COPYBENCH_CUDA_CHECK(cudaMalloc(
            &destinations[destination_index][source_index], options.d2d_bytes));
        COPYBENCH_CUDA_CHECK(cudaMemset(
            destinations[destination_index][source_index], 0,
            options.d2d_bytes));
      }
    }

    const size_t stream_count = options.topology == Topology::kSingleTwoCopy
                                    ? options.devices.size()
                                    : edges.size();
    streams.resize(stream_count);
    for (size_t index = 0; index < streams.size(); ++index) {
      const int source_index =
          options.topology == Topology::kSingleTwoCopy
              ? static_cast<int>(index)
              : edges[index].source_index;
      streams[index].device = options.devices[source_index];
      COPYBENCH_CUDA_CHECK(cudaSetDevice(streams[index].device));
      COPYBENCH_CUDA_CHECK(cudaStreamCreate(&streams[index].stream));
      COPYBENCH_CUDA_CHECK(cudaEventCreate(&streams[index].start));
      COPYBENCH_CUDA_CHECK(cudaEventCreate(&streams[index].stop));
    }

    std::vector<std::string> device_names;
    device_names.reserve(options.devices.size());
    for (int device : options.devices) {
      cudaDeviceProp properties{};
      COPYBENCH_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
      device_names.emplace_back(properties.name);
    }

    counters.reserve(options.devices.size());
    for (size_t index = 0; index < options.devices.size(); ++index) {
      counters.emplace_back(std::make_unique<BackgroundCounter>());
    }

    if (options.background == Background::kD2H) {
      background_workers.reserve(options.devices.size());
      for (size_t index = 0; index < options.devices.size(); ++index) {
        background_workers.emplace_back(
            backgroundWorker, options.devices[index], options.background_bytes,
            counters[index].get(), &background_state);
      }
      while (background_state.ready.load(std::memory_order_acquire) <
                 static_cast<int>(options.devices.size()) &&
             !background_state.failed.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      if (background_state.failed.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(background_state.error_mutex);
        throw std::runtime_error(background_state.error.empty()
                                     ? "background worker failed"
                                     : background_state.error);
      }
      background_state.start.store(true, std::memory_order_release);
      background_started = true;
    }

    std::cout << "D2D with same-process background benchmark\n"
              << "contextMode=same-process topology="
              << topologyName(options.topology)
              << " background=" << backgroundName(options.background)
              << " devices=" << copybench::joinDevices(options.devices)
              << " repeats=" << options.repeats
              << " warmup=" << kWarmupIterations
              << " bytes=" << options.d2d_bytes
              << " backgroundBytes=" << options.background_bytes
              << " directions=" << edges.size() << '\n';
    std::cout.flush();

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
      issueRound(options, resources, streams, destinations, edges);
      for (const StreamResource& resource : streams) {
        COPYBENCH_CUDA_CHECK(cudaSetDevice(resource.device));
        COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(resource.stream));
      }
    }

    if (options.background == Background::kD2H) {
      background_state.measure.store(true, std::memory_order_release);
    }
    const auto host_start = std::chrono::steady_clock::now();
    for (StreamResource& resource : streams) {
      COPYBENCH_CUDA_CHECK(cudaSetDevice(resource.device));
      COPYBENCH_CUDA_CHECK(cudaEventRecord(resource.start, resource.stream));
    }
    for (int iteration = 0; iteration < options.repeats; ++iteration) {
      issueRound(options, resources, streams, destinations, edges);
    }
    for (StreamResource& resource : streams) {
      COPYBENCH_CUDA_CHECK(cudaSetDevice(resource.device));
      COPYBENCH_CUDA_CHECK(cudaEventRecord(resource.stop, resource.stream));
    }

    std::vector<float> stream_elapsed_ms(streams.size(), 0.0f);
    for (size_t index = 0; index < streams.size(); ++index) {
      StreamResource& resource = streams[index];
      COPYBENCH_CUDA_CHECK(cudaSetDevice(resource.device));
      COPYBENCH_CUDA_CHECK(cudaEventSynchronize(resource.stop));
      COPYBENCH_CUDA_CHECK(cudaEventElapsedTime(
          &stream_elapsed_ms[index], resource.start, resource.stop));
      if (stream_elapsed_ms[index] <= 0.0f) {
        throw std::runtime_error("CUDA event resolution was insufficient");
      }
    }
    if (options.background == Background::kD2H) {
      background_state.stop.store(true, std::memory_order_release);
      for (std::thread& worker : background_workers) {
        if (worker.joinable()) worker.join();
      }
      background_started = false;
      if (background_state.failed.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(background_state.error_mutex);
        throw std::runtime_error(background_state.error.empty()
                                     ? "background worker failed"
                                     : background_state.error);
      }
    }
    const auto host_end = std::chrono::steady_clock::now();

    std::vector<SourceResult> source_results(options.devices.size());
    for (size_t source_index = 0; source_index < options.devices.size();
         ++source_index) {
      SourceResult& result = source_results[source_index];
      result.device = options.devices[source_index];
      for (const Edge& edge : edges) {
        if (edge.source_index == static_cast<int>(source_index)) {
          ++result.outgoing;
        }
      }
      float elapsed_ms = 0.0f;
      if (options.topology == Topology::kSingleTwoCopy) {
        elapsed_ms = stream_elapsed_ms[source_index];
      } else {
        for (size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
          if (edges[edge_index].source_index ==
              static_cast<int>(source_index)) {
            elapsed_ms = std::max(elapsed_ms, stream_elapsed_ms[edge_index]);
          }
        }
      }
      result.elapsed_ms = elapsed_ms;
      const double seconds = static_cast<double>(elapsed_ms) / 1000.0;
      result.gbps = static_cast<double>(options.d2d_bytes) * options.repeats *
                    result.outgoing / seconds / 1e9;
    }

    float max_elapsed_ms = 0.0f;
    for (const SourceResult& result : source_results) {
      max_elapsed_ms = std::max(max_elapsed_ms, result.elapsed_ms);
    }
    const double d2d_gbps =
        static_cast<double>(options.d2d_bytes) * options.repeats * edges.size() /
        (static_cast<double>(max_elapsed_ms) / 1000.0) / 1e9;

    for (const SourceResult& result : source_results) {
      std::cout << std::fixed << std::setprecision(3)
                << "source[" << result.device << "] outgoing="
                << result.outgoing << " elapsed_ms=" << result.elapsed_ms
                << " GBps=" << result.gbps << '\n';
    }
    const unsigned long long background_bytes = backgroundTotal(counters);
    const double background_elapsed_sec =
        options.background == Background::kD2H
            ? std::chrono::duration<double>(host_end - host_start).count()
            : 0.0;
    const double background_gbps = background_elapsed_sec > 0.0
                                       ? static_cast<double>(background_bytes) /
                                             background_elapsed_sec / 1e9
                                       : 0.0;
    std::cout << std::fixed << std::setprecision(3)
              << "aggregateGBps=" << d2d_gbps << '\n'
              << "backgroundTotalBytes=" << background_bytes
              << " backgroundElapsedSec=" << background_elapsed_sec
              << " backgroundAggregateGBps=" << background_gbps << '\n';

    copybench::writeTextFile(
        options.output,
        makeJson(options, source_results, d2d_gbps, device_names, counters,
                 background_elapsed_sec, background_bytes));
    cleanupD2D(resources, streams, destinations);
    return 0;
  } catch (const std::exception& error) {
    if (background_started || !background_workers.empty()) {
      stopBackgroundWorkers(&background_state, &background_workers);
    }
    cleanupD2D(resources, streams, destinations);
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
