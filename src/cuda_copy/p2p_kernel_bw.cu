#include "copy_common.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kWarmupIterations = 10;
constexpr size_t kDefaultBytes = 255ULL * 1024ULL * 1024ULL;
constexpr int kThreadsPerBlock = 256;
constexpr int kMaxBlocks = 4096;

enum class Topology {
  kSingleTwoCopy,
  kEdgeIndependent,
};

enum class VictimMode {
  kLocalD2DCE,
  kPeerRead,
  kPeerWrite,
};

struct Options {
  std::vector<int> devices{0, 1, 2};
  Topology topology = Topology::kSingleTwoCopy;
  VictimMode victim = VictimMode::kLocalD2DCE;
  size_t bytes = kDefaultBytes;
  int repeats = 300;
  std::string output;
};

struct Edge {
  int source_index = -1;
  int destination_index = -1;
};

struct DeviceResource {
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

const char* topologyName(Topology topology) {
  return topology == Topology::kSingleTwoCopy ? "single-two-copy"
                                               : "edge-independent";
}

const char* victimName(VictimMode victim) {
  switch (victim) {
    case VictimMode::kLocalD2DCE: return "local-d2d-ce";
    case VictimMode::kPeerRead: return "peer-read";
    case VictimMode::kPeerWrite: return "peer-write";
  }
  return "unknown";
}

void printUsage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --deviceList=0,1,2                    GPUs in the allpairs victim\n"
      << "  --topology=single-two-copy|edge-independent\n"
      << "  --victimMode=local-d2d-ce|peer-read|peer-write\n"
      << "  --size=255M                            bytes per edge operation\n"
      << "  --repeats=300                          measured repetitions\n"
      << "  --output=<path>                        JSON output path\n"
      << "  --help                                 show this message\n"
      << "Defaults: deviceList=0,1,2, topology=single-two-copy, "
         "victimMode=local-d2d-ce, size=255M, repeats=300, warmup=10 (fixed)\n";
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
    } else if (copybench::consumeOption(index, argc, argv,
                                        {"--victimMode", "--victim-mode"},
                                        &value)) {
      const std::string victim = copybench::lower(value);
      if (victim == "local-d2d-ce") {
        options.victim = VictimMode::kLocalD2DCE;
      } else if (victim == "peer-read") {
        options.victim = VictimMode::kPeerRead;
      } else if (victim == "peer-write") {
        options.victim = VictimMode::kPeerWrite;
      } else {
        throw std::invalid_argument(
            "--victimMode must be local-d2d-ce, peer-read, or peer-write");
      }
    } else if (copybench::consumeOption(index, argc, argv, {"--size"},
                                        &value)) {
      options.bytes = copybench::parseSize(value, "--size");
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

bool launchesOnDestination(VictimMode victim) {
  return victim == VictimMode::kPeerRead;
}

int streamIndexForEdge(const Options& options, const Edge& edge,
                       int edge_index) {
  if (options.topology == Topology::kEdgeIndependent) return edge_index;
  return launchesOnDestination(options.victim) ? edge.destination_index
                                                : edge.source_index;
}

int streamDeviceForEdge(const Options& options, const Edge& edge) {
  const int index = launchesOnDestination(options.victim)
                        ? edge.destination_index
                        : edge.source_index;
  return options.devices[index];
}

void enablePeerAccess(int requester, int peer) {
  int can_access = 0;
  COPYBENCH_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, requester, peer));
  if (!can_access) {
    std::ostringstream message;
    message << "GPU " << requester << " cannot access GPU " << peer;
    throw std::runtime_error(message.str());
  }
  COPYBENCH_CUDA_CHECK(cudaSetDevice(requester));
  const cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
  if (status == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
  } else {
    COPYBENCH_CUDA_CHECK(status);
  }
}

__global__ void peerReadKernel(const unsigned char* source,
                               unsigned char* destination, size_t bytes) {
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < bytes; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const unsigned char value = source[index];
    destination[index] = static_cast<unsigned char>(value ^ 0x5a);
  }
}

__global__ void peerWriteKernel(const unsigned char* source,
                                unsigned char* destination, size_t bytes) {
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < bytes; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const unsigned char value = source[index];
    destination[index] = static_cast<unsigned char>(value ^ 0xa5);
  }
}

int blockCount(size_t bytes) {
  const size_t needed = (bytes + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return static_cast<int>(std::min<size_t>(std::max<size_t>(needed, 1),
                                           kMaxBlocks));
}

void cleanup(std::vector<DeviceResource>& resources,
             std::vector<StreamResource>& streams,
             std::vector<std::vector<void*>>& destinations,
             std::vector<std::vector<void*>>& local_destinations) noexcept {
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
  for (DeviceResource& resource : resources) {
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
  for (size_t source_index = 0; source_index < local_destinations.size();
       ++source_index) {
    if (resources[source_index].device < 0) continue;
    cudaSetDevice(resources[source_index].device);
    for (void*& destination : local_destinations[source_index]) {
      if (destination != nullptr) cudaFree(destination);
      destination = nullptr;
    }
  }
}

void issueRound(const Options& options,
                const std::vector<DeviceResource>& resources,
                const std::vector<StreamResource>& streams,
                const std::vector<std::vector<void*>>& destinations,
                const std::vector<std::vector<void*>>& local_destinations,
                const std::vector<Edge>& edges) {
  for (size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    const Edge& edge = edges[edge_index];
    const int source_device = options.devices[edge.source_index];
    const int destination_device = options.devices[edge.destination_index];
    const int stream_index =
        streamIndexForEdge(options, edge, static_cast<int>(edge_index));
    const int launch_device = streamDeviceForEdge(options, edge);
    COPYBENCH_CUDA_CHECK(cudaSetDevice(launch_device));
    void* destination =
        options.victim == VictimMode::kLocalD2DCE
            ? local_destinations[edge.source_index][edge.destination_index]
            : destinations[edge.destination_index][edge.source_index];
    const void* source = resources[edge.source_index].source;
    if (options.victim == VictimMode::kLocalD2DCE) {
      COPYBENCH_CUDA_CHECK(cudaMemcpyAsync(
          destination, source, options.bytes, cudaMemcpyDeviceToDevice,
          streams[stream_index].stream));
    } else if (options.victim == VictimMode::kPeerRead) {
      peerReadKernel<<<blockCount(options.bytes), kThreadsPerBlock, 0,
                       streams[stream_index].stream>>>(
          static_cast<const unsigned char*>(source),
          static_cast<unsigned char*>(destination), options.bytes);
      COPYBENCH_CUDA_CHECK(cudaGetLastError());
    } else {
      peerWriteKernel<<<blockCount(options.bytes), kThreadsPerBlock, 0,
                        streams[stream_index].stream>>>(
          static_cast<const unsigned char*>(source),
          static_cast<unsigned char*>(destination), options.bytes);
      COPYBENCH_CUDA_CHECK(cudaGetLastError());
    }
    (void)source_device;
    (void)destination_device;
  }
}

std::string makeJson(const Options& options,
                    const std::vector<SourceResult>& source_results,
                    const std::vector<StreamResource>& streams,
                    const std::vector<float>& stream_elapsed_ms,
                    double aggregate_gbps,
                    const std::vector<std::string>& device_names) {
  std::ostringstream json;
  json << std::fixed << std::setprecision(6);
  json << "{\n"
       << "  \"program\": \"p2p_kernel_bw\",\n"
       << "  \"victimMode\": "
       << copybench::jsonQuote(victimName(options.victim)) << ",\n"
       << "  \"topology\": "
       << copybench::jsonQuote(topologyName(options.topology)) << ",\n"
       << "  \"launchDeviceRole\": "
       << copybench::jsonQuote(launchesOnDestination(options.victim)
                                   ? "destination"
                                   : "source")
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
       << "  \"bytesPerEdge\": " << options.bytes << ",\n"
       << "  \"repeats\": " << options.repeats << ",\n"
       << "  \"warmup\": " << kWarmupIterations << ",\n"
       << "  \"streamCount\": " << streams.size() << ",\n"
       << "  \"aggregateGBps\": " << aggregate_gbps << ",\n"
       << "  \"sourceResults\": [\n";
  for (size_t index = 0; index < source_results.size(); ++index) {
    if (index != 0) json << ",\n";
    json << "    {\"device\": " << source_results[index].device
         << ", \"outgoing\": " << source_results[index].outgoing
         << ", \"elapsedMs\": " << source_results[index].elapsed_ms
         << ", \"GBps\": " << source_results[index].gbps << '}';
  }
  json << "\n  ],\n  \"streamResults\": [\n";
  for (size_t index = 0; index < streams.size(); ++index) {
    if (index != 0) json << ",\n";
    const double seconds = static_cast<double>(stream_elapsed_ms[index]) / 1000.0;
    json << "    {\"stream\": " << index
         << ", \"device\": " << streams[index].device
         << ", \"elapsedMs\": " << stream_elapsed_ms[index]
         << ", \"GBps\": "
         << (static_cast<double>(options.bytes) * options.repeats *
             (options.topology == Topology::kSingleTwoCopy ? 2 : 1)) /
                seconds / 1e9
         << '}';
  }
  json << "\n  ]\n}\n";
  return json.str();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::vector<DeviceResource> resources;
  std::vector<StreamResource> streams;
  std::vector<std::vector<void*>> destinations;
  std::vector<std::vector<void*>> local_destinations;
  try {
    options = parseOptions(argc, argv);
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

    const std::vector<Edge> edges = makeEdges(options);
    if (options.victim == VictimMode::kPeerRead) {
      for (const Edge& edge : edges) {
        enablePeerAccess(options.devices[edge.destination_index],
                         options.devices[edge.source_index]);
      }
    } else if (options.victim == VictimMode::kPeerWrite) {
      for (const Edge& edge : edges) {
        enablePeerAccess(options.devices[edge.source_index],
                         options.devices[edge.destination_index]);
      }
    }

    resources.resize(options.devices.size());
    destinations.assign(options.devices.size(),
                        std::vector<void*>(options.devices.size(), nullptr));
    local_destinations.assign(
        options.devices.size(),
        std::vector<void*>(options.devices.size(), nullptr));
    for (size_t index = 0; index < options.devices.size(); ++index) {
      resources[index].device = options.devices[index];
      COPYBENCH_CUDA_CHECK(cudaSetDevice(resources[index].device));
      COPYBENCH_CUDA_CHECK(cudaMalloc(&resources[index].source, options.bytes));
      COPYBENCH_CUDA_CHECK(cudaMemset(resources[index].source,
                                      resources[index].device & 0xff,
                                      options.bytes));
    }
    if (options.victim == VictimMode::kLocalD2DCE) {
      for (size_t source_index = 0; source_index < options.devices.size();
           ++source_index) {
        COPYBENCH_CUDA_CHECK(cudaSetDevice(options.devices[source_index]));
        for (size_t destination_index = 0;
             destination_index < options.devices.size(); ++destination_index) {
          if (destination_index == source_index) continue;
          COPYBENCH_CUDA_CHECK(cudaMalloc(
              &local_destinations[source_index][destination_index],
              options.bytes));
          COPYBENCH_CUDA_CHECK(cudaMemset(
              local_destinations[source_index][destination_index], 0,
              options.bytes));
        }
      }
    } else {
      for (size_t destination_index = 0;
           destination_index < options.devices.size(); ++destination_index) {
        COPYBENCH_CUDA_CHECK(cudaSetDevice(options.devices[destination_index]));
        for (size_t source_index = 0; source_index < options.devices.size();
             ++source_index) {
          if (destination_index == source_index) continue;
          COPYBENCH_CUDA_CHECK(cudaMalloc(
              &destinations[destination_index][source_index], options.bytes));
          COPYBENCH_CUDA_CHECK(cudaMemset(
              destinations[destination_index][source_index], 0,
              options.bytes));
        }
      }
    }

    const size_t stream_count = options.topology == Topology::kSingleTwoCopy
                                    ? options.devices.size()
                                    : edges.size();
    streams.resize(stream_count);
    for (size_t index = 0; index < streams.size(); ++index) {
      const Edge* edge = options.topology == Topology::kEdgeIndependent
                             ? &edges[index]
                             : nullptr;
      const int device_index =
          edge == nullptr
              ? static_cast<int>(index)
              : (launchesOnDestination(options.victim) ? edge->destination_index
                                                         : edge->source_index);
      streams[index].device = options.devices[device_index];
      COPYBENCH_CUDA_CHECK(cudaSetDevice(streams[index].device));
      COPYBENCH_CUDA_CHECK(cudaStreamCreate(&streams[index].stream));
      COPYBENCH_CUDA_CHECK(cudaEventCreate(&streams[index].start));
      COPYBENCH_CUDA_CHECK(cudaEventCreate(&streams[index].stop));
    }

    std::cout << "P2P kernel victim benchmark\n"
              << "victimMode=" << victimName(options.victim)
              << " topology=" << topologyName(options.topology)
              << " launchDeviceRole="
              << (launchesOnDestination(options.victim) ? "destination"
                                                         : "source")
              << " devices=" << copybench::joinDevices(options.devices)
              << " repeats=" << options.repeats
              << " warmup=" << kWarmupIterations
              << " bytes=" << options.bytes
              << " directions=" << edges.size() << '\n';
    std::cout.flush();

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
      issueRound(options, resources, streams, destinations, local_destinations,
                 edges);
      for (const StreamResource& resource : streams) {
        COPYBENCH_CUDA_CHECK(cudaSetDevice(resource.device));
        COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(resource.stream));
      }
    }
    for (StreamResource& resource : streams) {
      COPYBENCH_CUDA_CHECK(cudaSetDevice(resource.device));
      COPYBENCH_CUDA_CHECK(cudaEventRecord(resource.start, resource.stream));
    }
    for (int iteration = 0; iteration < options.repeats; ++iteration) {
      issueRound(options, resources, streams, destinations, local_destinations,
                 edges);
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

    std::vector<SourceResult> source_results(options.devices.size());
    for (size_t source_index = 0; source_index < options.devices.size();
         ++source_index) {
      SourceResult& result = source_results[source_index];
      result.device = options.devices[source_index];
      for (const Edge& edge : edges) {
        if (edge.source_index == static_cast<int>(source_index)) ++result.outgoing;
      }
      float elapsed_ms = 0.0f;
      for (size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
        if (edges[edge_index].source_index == static_cast<int>(source_index)) {
          const int stream_index = streamIndexForEdge(
              options, edges[edge_index], static_cast<int>(edge_index));
          elapsed_ms = std::max(elapsed_ms, stream_elapsed_ms[stream_index]);
        }
      }
      result.elapsed_ms = elapsed_ms;
      const double seconds = static_cast<double>(elapsed_ms) / 1000.0;
      result.gbps = static_cast<double>(options.bytes) * options.repeats *
                    result.outgoing / seconds / 1e9;
    }

    float max_elapsed_ms = 0.0f;
    for (float elapsed : stream_elapsed_ms) {
      max_elapsed_ms = std::max(max_elapsed_ms, elapsed);
    }
    const double aggregate_gbps =
        static_cast<double>(options.bytes) * options.repeats * edges.size() /
        (static_cast<double>(max_elapsed_ms) / 1000.0) / 1e9;
    for (const SourceResult& result : source_results) {
      std::cout << std::fixed << std::setprecision(3)
                << "source[" << result.device << "] outgoing="
                << result.outgoing << " elapsed_ms=" << result.elapsed_ms
                << " GBps=" << result.gbps << '\n';
    }
    std::cout << std::fixed << std::setprecision(3)
              << "aggregateGBps=" << aggregate_gbps << '\n';
    copybench::writeTextFile(
        options.output,
        makeJson(options, source_results, streams, stream_elapsed_ms,
                 aggregate_gbps, device_names));
    cleanup(resources, streams, destinations, local_destinations);
    return 0;
  } catch (const std::exception& error) {
    cleanup(resources, streams, destinations, local_destinations);
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
