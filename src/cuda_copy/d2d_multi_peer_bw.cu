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

enum class Pattern {
  kRing,
  kAllPairs,
};

enum class EdgeOrder {
  kSourceMajor,
  kDestinationMajor,
};

enum class StreamMode {
  kPerSource,
  kPerEdge,
};

enum class StreamDependency {
  kNone,
  kSourceChain,
};

struct Options {
  std::vector<int> devices{0, 1, 2, 3};
  Pattern pattern = Pattern::kAllPairs;
  EdgeOrder edge_order = EdgeOrder::kSourceMajor;
  std::vector<int> edge_permutation;
  bool edge_permutation_explicit = false;
  StreamMode stream_mode = StreamMode::kPerSource;
  StreamDependency stream_dependency = StreamDependency::kNone;
  int streams_per_source = 1;
  bool stream_mode_explicit = false;
  bool streams_per_source_explicit = false;
  std::vector<int> stream_assignment;
  bool stream_assignment_explicit = false;
  std::vector<int> source_offsets_us;
  bool source_offsets_explicit = false;
  int repeats = 20;
  int chunk_repeats = 0;
  bool sync_each_iteration = false;
  size_t bytes = kDefaultBytes;
  std::string output;
};

struct DeviceResources {
  int device = -1;
  void* source = nullptr;
};

struct StreamResources {
  int device = -1;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  std::vector<cudaEvent_t> chunk_starts;
  std::vector<cudaEvent_t> chunk_stops;
  std::vector<cudaEvent_t> dependency_events;
};

__device__ __forceinline__ unsigned long long readGlobalTimerNs() {
  unsigned long long value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__global__ void sourceOffsetDelay(unsigned long long delay_ns) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const unsigned long long begin = readGlobalTimerNs();
  while (readGlobalTimerNs() - begin < delay_ns) {
  }
}

void enqueueSourceOffsets(const Options& options,
                          const std::vector<StreamResources>& streams) {
  for (size_t source_index = 0; source_index < options.devices.size();
       ++source_index) {
    const int offset_us = options.source_offsets_us[source_index];
    if (offset_us == 0) continue;
    const StreamResources& stream_resource = streams[source_index];
    COPYBENCH_CUDA_CHECK(cudaSetDevice(stream_resource.device));
    const unsigned long long delay_ns =
        static_cast<unsigned long long>(offset_us) * 1000ULL;
    sourceOffsetDelay<<<1, 1, 0, stream_resource.stream>>>(delay_ns);
    COPYBENCH_CUDA_CHECK(cudaGetLastError());
  }
}

struct ChunkResult {
  int index = 0;
  int repeats = 0;
  float elapsed_ms = 0.0f;
  double gbps = 0.0;
};

struct SourceResult {
  int device = -1;
  int outgoing = 0;
  float elapsed_ms = 0.0f;
  double gbps = 0.0;
  std::vector<ChunkResult> chunks;
};

struct EdgeResult {
  int source = -1;
  int destination = -1;
  int order = -1;
  int stream = -1;
  float elapsed_ms = 0.0f;
  double gbps = 0.0;
  std::vector<ChunkResult> chunks;
};

struct Edge {
  int source_index = -1;
  int destination_index = -1;
};

const char* patternName(Pattern pattern) {
  return pattern == Pattern::kRing ? "ring" : "allpairs";
}

const char* edgeOrderName(EdgeOrder order) {
  return order == EdgeOrder::kDestinationMajor ? "destination-major"
                                               : "source-major";
}

const char* streamModeName(StreamMode mode) {
  return mode == StreamMode::kPerEdge ? "per-edge" : "per-source";
}

const char* streamDependencyName(StreamDependency dependency) {
  return dependency == StreamDependency::kSourceChain ? "source-chain"
                                                      : "none";
}

int chunkCount(const Options& options) {
  if (options.chunk_repeats == 0) return 0;
  return (options.repeats - 1) / options.chunk_repeats + 1;
}

int chunkLength(const Options& options, int chunk_index) {
  const int completed = chunk_index * options.chunk_repeats;
  return std::min(options.chunk_repeats, options.repeats - completed);
}

bool parseBoolean01(const std::string& text, const std::string& option_name) {
  const int parsed = copybench::parseNonNegativeInt(text, option_name);
  if (parsed > 1) {
    throw std::invalid_argument(option_name + " must be 0 or 1");
  }
  return parsed == 1;
}

std::vector<int> parsePermutation(const std::string& text,
                                  const std::string& option_name) {
  std::vector<int> permutation;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    permutation.push_back(
        copybench::parseNonNegativeInt(copybench::trim(item), option_name));
  }
  if (permutation.empty()) {
    throw std::invalid_argument(option_name + " cannot be empty");
  }
  return permutation;
}

std::vector<int> parseNonNegativeList(const std::string& text,
                                      const std::string& option_name) {
  std::vector<int> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    values.push_back(
        copybench::parseNonNegativeInt(copybench::trim(item), option_name));
  }
  if (values.empty()) {
    throw std::invalid_argument(option_name + " cannot be empty");
  }
  return values;
}

void printUsage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --pattern=ring|allpairs                D2D traffic pattern\n"
      << "  --deviceList=0,1,2,3                   visible GPU list\n"
      << "  --devices=4                            use GPUs 0 through N-1\n"
      << "  --edgeOrder=source-major|destination-major\n"
      << "                                           order edges are submitted\n"
      << "  --edgePermutation=0,1,2                 per-source FIFO permutation\n"
      << "  --sourceOffsetsUs=0,250,500,750         one-time source delays\n"
      << "  --streamMode=per-source|per-edge        logical CUDA stream topology\n"
      << "  --streamDependency=none|source-chain    per-edge dependency model\n"
      << "  --streamsPerSource=1|2|3               eligible streams per source\n"
      << "  --streamAssignment=0,1,1|1,0,1|1,1,0   one-vs-two stream mapping\n"
      << "  --repeats=20                           measured repetitions\n"
      << "  --chunkRepeats=0                       optional chunk timing; 0 disables\n"
      << "  --syncEachIteration=0|1                synchronize D2D streams after each round\n"
      << "  --size=255M                            one peer memcpy size\n"
      << "  --output=<path>                         optional JSON output path\n"
      << "  --help                                  show this message\n"
      << "Defaults: pattern=allpairs, deviceList=0,1,2,3, "
         "edgeOrder=source-major, streamMode=per-source, "
         "streamDependency=none, streamsPerSource=1, repeats=20, "
         "chunkRepeats=0, warmup=10 (fixed), size=255M\n";
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
    if (copybench::consumeOption(index, argc, argv, {"--pattern"}, &value)) {
      const std::string pattern = copybench::lower(value);
      if (pattern == "ring") {
        options.pattern = Pattern::kRing;
      } else if (pattern == "allpairs" || pattern == "all-pairs" ||
                 pattern == "fullmesh") {
        options.pattern = Pattern::kAllPairs;
      } else {
        throw std::invalid_argument("--pattern must be ring or allpairs");
      }
    } else if (copybench::consumeOption(index, argc, argv, {"--edgeOrder"},
                                        &value)) {
      const std::string order = copybench::lower(value);
      if (order == "source-major" || order == "source") {
        options.edge_order = EdgeOrder::kSourceMajor;
      } else if (order == "destination-major" || order == "destination") {
        options.edge_order = EdgeOrder::kDestinationMajor;
      } else {
        throw std::invalid_argument(
            "--edgeOrder must be source-major or destination-major");
      }
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--edgePermutation"}, &value)) {
      options.edge_permutation =
          parsePermutation(value, "--edgePermutation");
      options.edge_permutation_explicit = true;
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--sourceOffsetsUs"}, &value)) {
      options.source_offsets_us =
          parseNonNegativeList(value, "--sourceOffsetsUs");
      options.source_offsets_explicit = true;
    } else if (copybench::consumeOption(index, argc, argv, {"--streamMode"},
                                        &value)) {
      options.stream_mode_explicit = true;
      const std::string mode = copybench::lower(value);
      if (mode == "per-source" || mode == "source") {
        options.stream_mode = StreamMode::kPerSource;
      } else if (mode == "per-edge" || mode == "edge") {
        options.stream_mode = StreamMode::kPerEdge;
      } else {
        throw std::invalid_argument(
            "--streamMode must be per-source or per-edge");
      }
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--streamsPerSource"}, &value)) {
      options.streams_per_source =
          copybench::parsePositiveInt(value, "--streamsPerSource");
      options.streams_per_source_explicit = true;
      if (options.streams_per_source > 3) {
        throw std::invalid_argument(
            "--streamsPerSource must be 1, 2, or 3");
      }
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--streamAssignment"}, &value)) {
      options.stream_assignment =
          parseNonNegativeList(value, "--streamAssignment");
      options.stream_assignment_explicit = true;
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--streamDependency"}, &value)) {
      const std::string dependency = copybench::lower(value);
      if (dependency == "none") {
        options.stream_dependency = StreamDependency::kNone;
      } else if (dependency == "source-chain" || dependency == "sourcechain") {
        options.stream_dependency = StreamDependency::kSourceChain;
      } else {
        throw std::invalid_argument(
            "--streamDependency must be none or source-chain");
      }
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--deviceList", "--device-list"},
                   &value)) {
      options.devices = copybench::parseDeviceList(value, "--deviceList");
    } else if (copybench::consumeOption(index, argc, argv, {"--devices"},
                                        &value)) {
      const int count = copybench::parsePositiveInt(value, "--devices");
      options.devices.clear();
      for (int device = 0; device < count; ++device) {
        options.devices.push_back(device);
      }
    } else if (copybench::consumeOption(index, argc, argv, {"--repeats"},
                                        &value)) {
      options.repeats = copybench::parsePositiveInt(value, "--repeats");
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--chunkRepeats", "--chunk-repeats"},
                   &value)) {
      options.chunk_repeats =
          copybench::parseNonNegativeInt(value, "--chunkRepeats");
    } else if (copybench::consumeOption(
                   index, argc, argv,
                   {"--syncEachIteration", "--sync-each-iteration"},
                   &value)) {
      options.sync_each_iteration =
          parseBoolean01(value, "--syncEachIteration");
    } else if (copybench::consumeOption(index, argc, argv, {"--size"},
                                        &value)) {
      options.bytes = copybench::parseSize(value, "--size");
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
  if (options.edge_permutation_explicit) {
    if (options.edge_order != EdgeOrder::kSourceMajor) {
      throw std::invalid_argument(
          "--edgePermutation requires --edgeOrder=source-major");
    }
    const int expected_length = options.pattern == Pattern::kAllPairs
                                    ? static_cast<int>(options.devices.size()) - 1
                                    : 1;
    if (static_cast<int>(options.edge_permutation.size()) != expected_length) {
      throw std::invalid_argument(
          "--edgePermutation length must equal outgoing edges per source");
    }
    std::vector<bool> seen(expected_length, false);
    for (int position : options.edge_permutation) {
      if (position < 0 || position >= expected_length || seen[position]) {
        throw std::invalid_argument(
            "--edgePermutation must contain each valid position exactly once");
      }
      seen[position] = true;
    }
  }
  if (!options.streams_per_source_explicit) {
    options.streams_per_source =
        options.stream_mode == StreamMode::kPerEdge ? 3 : 1;
  } else {
    if (options.stream_mode_explicit &&
        ((options.stream_mode == StreamMode::kPerEdge &&
          options.streams_per_source != 3) ||
         (options.stream_mode == StreamMode::kPerSource &&
          options.streams_per_source == 3))) {
      throw std::invalid_argument(
          "--streamMode conflicts with --streamsPerSource");
    }
    options.stream_mode = options.streams_per_source == 3
                              ? StreamMode::kPerEdge
                              : StreamMode::kPerSource;
  }
  if (options.source_offsets_explicit) {
    if (options.streams_per_source != 1) {
      throw std::invalid_argument(
          "--sourceOffsetsUs requires one stream per source");
    }
    if (options.source_offsets_us.size() != options.devices.size()) {
      throw std::invalid_argument(
          "--sourceOffsetsUs length must equal the device count");
    }
  } else {
    options.source_offsets_us.assign(options.devices.size(), 0);
  }
  const int expected_edges_per_source =
      options.pattern == Pattern::kAllPairs
          ? static_cast<int>(options.devices.size()) - 1
          : 1;
  if (options.stream_assignment_explicit) {
    if (options.stream_mode != StreamMode::kPerSource ||
        options.streams_per_source != 2) {
      throw std::invalid_argument(
          "--streamAssignment requires per-source streamsPerSource=2");
    }
    if (expected_edges_per_source != 3 ||
        static_cast<int>(options.stream_assignment.size()) !=
            expected_edges_per_source) {
      throw std::invalid_argument(
          "--streamAssignment requires exactly three allpairs edges per source");
    }
    int slot_counts[2] = {0, 0};
    for (int stream_slot : options.stream_assignment) {
      if (stream_slot < 0 || stream_slot >= options.streams_per_source) {
        throw std::invalid_argument(
            "--streamAssignment values must be 0 or 1");
      }
      ++slot_counts[stream_slot];
    }
    if (!((slot_counts[0] == 1 && slot_counts[1] == 2) ||
          (slot_counts[0] == 2 && slot_counts[1] == 1))) {
      throw std::invalid_argument(
          "--streamAssignment must place one edge on one stream and two on the other");
    }
  } else {
    options.stream_assignment.resize(expected_edges_per_source, 0);
    if (options.streams_per_source == 2) {
      for (int position = 0; position < expected_edges_per_source;
           ++position) {
        options.stream_assignment[position] = position == 1 ? 1 : 0;
      }
    } else if (options.streams_per_source == 3) {
      for (int position = 0; position < expected_edges_per_source;
           ++position) {
        options.stream_assignment[position] = position;
      }
    }
  }
  if (options.stream_dependency == StreamDependency::kSourceChain &&
      options.streams_per_source != 3) {
    throw std::invalid_argument(
        "--streamDependency=source-chain requires three per-source streams");
  }
  return options;
}

std::vector<Edge> makeEdges(const Options& options) {
  const int device_count = static_cast<int>(options.devices.size());
  std::vector<Edge> edges;
  if (options.pattern == Pattern::kRing) {
    edges.reserve(device_count);
    for (int source = 0; source < device_count; ++source) {
      edges.push_back({source, (source + 1) % device_count});
    }
  } else {
    edges.reserve(device_count * (device_count - 1));
    for (int source = 0; source < device_count; ++source) {
      for (int destination = 0; destination < device_count; ++destination) {
        if (source != destination) edges.push_back({source, destination});
      }
    }
  }
  if (options.edge_permutation_explicit) {
    const int edges_per_source =
        options.pattern == Pattern::kAllPairs ? device_count - 1 : 1;
    std::vector<Edge> permuted;
    permuted.reserve(edges.size());
    for (int source = 0; source < device_count; ++source) {
      const int source_begin = source * edges_per_source;
      for (int position : options.edge_permutation) {
        permuted.push_back(edges[source_begin + position]);
      }
    }
    edges.swap(permuted);
  }
  if (options.edge_order == EdgeOrder::kDestinationMajor) {
    std::stable_sort(edges.begin(), edges.end(), [](const Edge& left,
                                                    const Edge& right) {
      if (left.destination_index != right.destination_index) {
        return left.destination_index < right.destination_index;
      }
      return left.source_index < right.source_index;
    });
  }
  return edges;
}

std::vector<std::vector<int>> makeSourceEdgeLists(
    const Options& options, const std::vector<Edge>& edges) {
  std::vector<std::vector<int>> source_edges(options.devices.size());
  for (size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    source_edges[edges[edge_index].source_index].push_back(
        static_cast<int>(edge_index));
  }
  return source_edges;
}

int streamIndexForEdge(const Options& options, int edge_index,
                       const Edge& edge,
                       const std::vector<int>& edge_positions) {
  if (options.stream_mode == StreamMode::kPerEdge) {
    return edge_index;
  }
  if (options.streams_per_source == 1) {
    return edge.source_index;
  }

  const int stream_slot = options.stream_assignment[edge_positions[edge_index]];
  return edge.source_index * options.streams_per_source + stream_slot;
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
  const cudaError_t status =
      cudaDeviceEnablePeerAccess(destination_device, 0);
  if (status == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
  } else {
    COPYBENCH_CUDA_CHECK(status);
  }
}

void cleanup(std::vector<DeviceResources>& resources,
             std::vector<StreamResources>& streams,
             std::vector<std::vector<void*>>& destinations) noexcept {
  for (StreamResources& stream_resource : streams) {
    if (stream_resource.device < 0) continue;
    cudaSetDevice(stream_resource.device);
    if (stream_resource.stream != nullptr) {
      cudaStreamSynchronize(stream_resource.stream);
    }
  }

  for (StreamResources& stream_resource : streams) {
    if (stream_resource.device < 0) continue;
    cudaSetDevice(stream_resource.device);
    if (stream_resource.start != nullptr) {
      cudaEventDestroy(stream_resource.start);
    }
    if (stream_resource.stop != nullptr) {
      cudaEventDestroy(stream_resource.stop);
    }
    for (cudaEvent_t event : stream_resource.chunk_starts) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    for (cudaEvent_t event : stream_resource.chunk_stops) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    for (cudaEvent_t event : stream_resource.dependency_events) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    stream_resource.chunk_starts.clear();
    stream_resource.chunk_stops.clear();
    stream_resource.dependency_events.clear();
    if (stream_resource.stream != nullptr) {
      cudaStreamDestroy(stream_resource.stream);
    }
    stream_resource.start = nullptr;
    stream_resource.stop = nullptr;
    stream_resource.stream = nullptr;
  }

  for (DeviceResources& resource : resources) {
    if (resource.device < 0) continue;
    cudaSetDevice(resource.device);
    if (resource.source != nullptr) cudaFree(resource.source);
    resource.source = nullptr;
  }

  for (size_t destination_index = 0;
       destination_index < destinations.size(); ++destination_index) {
    const int destination_device =
        resources[destination_index].device;
    if (destination_device < 0) continue;
    cudaSetDevice(destination_device);
    for (void*& buffer : destinations[destination_index]) {
      if (buffer != nullptr) cudaFree(buffer);
      buffer = nullptr;
    }
  }
}

void issueRound(const Options& options,
                const std::vector<DeviceResources>& resources,
                const std::vector<StreamResources>& streams,
                const std::vector<std::vector<void*>>& destinations,
                const std::vector<Edge>& edges,
                const std::vector<std::vector<int>>& source_edges,
                const std::vector<int>& edge_positions, int iteration) {
  for (size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    const Edge& edge = edges[edge_index];
    const DeviceResources& source = resources[edge.source_index];
    const int stream_index = streamIndexForEdge(
        options, static_cast<int>(edge_index), edge, edge_positions);
    const int source_device = options.devices[edge.source_index];
    const int destination_device = options.devices[edge.destination_index];
    COPYBENCH_CUDA_CHECK(cudaSetDevice(source_device));
    if (options.stream_dependency == StreamDependency::kSourceChain) {
      const std::vector<int>& edges_for_source =
          source_edges[edge.source_index];
      const int position = edge_positions[edge_index];
      int dependency_edge_index = -1;
      int dependency_iteration = iteration;
      if (position > 0) {
        dependency_edge_index = edges_for_source[position - 1];
      } else if (iteration > 0) {
        dependency_edge_index = edges_for_source.back();
        dependency_iteration = iteration - 1;
      }
      if (dependency_edge_index >= 0) {
        COPYBENCH_CUDA_CHECK(cudaStreamWaitEvent(
            streams[stream_index].stream,
            streams[dependency_edge_index]
                .dependency_events[dependency_iteration],
            0));
      }
    }
    COPYBENCH_CUDA_CHECK(cudaMemcpyPeerAsync(
        destinations[edge.destination_index][edge.source_index],
        destination_device, source.source, source_device, options.bytes,
        streams[stream_index].stream));
    if (options.stream_dependency == StreamDependency::kSourceChain) {
      COPYBENCH_CUDA_CHECK(cudaEventRecord(
          streams[stream_index].dependency_events[iteration],
          streams[stream_index].stream));
    }
  }
}

std::string makeJson(const Options& options,
                     const std::vector<SourceResult>& results,
                     const std::vector<EdgeResult>& edge_results,
                     const std::vector<Edge>& edges,
                     int edge_count, double aggregate_gbps,
                     const std::vector<std::string>& device_names) {
  std::ostringstream json;
  json << std::fixed << std::setprecision(6);
  json << "{\n"
       << "  \"program\": \"d2d_multi_peer_bw\",\n"
       << "  \"pattern\": "
       << copybench::jsonQuote(patternName(options.pattern)) << ",\n"
       << "  \"edgeOrder\": "
       << copybench::jsonQuote(edgeOrderName(options.edge_order)) << ",\n"
       << "  \"streamMode\": "
       << copybench::jsonQuote(streamModeName(options.stream_mode)) << ",\n"
       << "  \"streamDependency\": "
       << copybench::jsonQuote(
              streamDependencyName(options.stream_dependency))
       << ",\n"
       << "  \"streamsPerSource\": " << options.streams_per_source
       << ",\n"
       << "  \"streamAssignment\": [";
  for (size_t index = 0; index < options.stream_assignment.size(); ++index) {
    if (index != 0) json << ", ";
    json << options.stream_assignment[index];
  }
  json << "],\n"
       << "  \"edgePermutation\": [";
  for (size_t index = 0; index < options.edge_permutation.size(); ++index) {
    if (index != 0) json << ", ";
    json << options.edge_permutation[index];
  }
  json << "],\n"
       << "  \"devices\": [";
  for (size_t index = 0; index < options.devices.size(); ++index) {
    if (index != 0) json << ", ";
    json << options.devices[index];
  }
  json << "],\n"
       << "  \"sourceEdgeOrder\": [\n";
  for (size_t source_index = 0; source_index < options.devices.size();
       ++source_index) {
    if (source_index != 0) json << ",\n";
    json << "    {\"source\": " << options.devices[source_index]
         << ", \"destinations\": [";
    bool first_destination = true;
    for (const Edge& edge : edges) {
      if (edge.source_index != static_cast<int>(source_index)) continue;
      if (!first_destination) json << ", ";
      json << options.devices[edge.destination_index];
      first_destination = false;
    }
    json << "]}";
  }
  json << "],\n"
       << "  \"sourceOffsetsUs\": [";
  for (size_t index = 0; index < options.source_offsets_us.size(); ++index) {
    if (index != 0) json << ", ";
    json << options.source_offsets_us[index];
  }
  json << "],\n"
       << "  \"repeats\": " << options.repeats << ",\n"
       << "  \"warmup\": " << kWarmupIterations << ",\n"
       << "  \"bytesPerMemcpy\": " << options.bytes << ",\n"
       << "  \"totalBytes\": "
       << static_cast<unsigned long long>(options.bytes) * options.repeats *
              edge_count
       << ",\n"
       << "  \"directions\": " << edge_count << ",\n"
       << "  \"aggregateGBps\": " << aggregate_gbps << ",\n";
  if (options.chunk_repeats > 0) {
    json << "  \"chunkRepeats\": " << options.chunk_repeats << ",\n"
         << "  \"chunkCount\": " << chunkCount(options) << ",\n";
  }
  if (options.sync_each_iteration) {
    json << "  \"syncEachIteration\": true,\n";
  }
  json << "  \"deviceNames\": [";
  for (size_t index = 0; index < device_names.size(); ++index) {
    if (index != 0) json << ", ";
    json << copybench::jsonQuote(device_names[index]);
  }
  json << "],\n"
       << "  \"sourceResults\": [\n";
  for (size_t index = 0; index < results.size(); ++index) {
    if (index != 0) json << ",\n";
    json << "    {\"device\": " << results[index].device
         << ", \"outgoing\": " << results[index].outgoing
         << ", \"elapsedMs\": " << results[index].elapsed_ms
         << ", \"GBps\": " << results[index].gbps;
    if (options.chunk_repeats > 0) {
      json << ", \"chunks\": [";
      for (size_t chunk_index = 0;
           chunk_index < results[index].chunks.size(); ++chunk_index) {
        if (chunk_index != 0) json << ", ";
        const ChunkResult& chunk = results[index].chunks[chunk_index];
        json << "{\"index\": " << chunk.index
             << ", \"repeats\": " << chunk.repeats
             << ", \"elapsedMs\": " << chunk.elapsed_ms
             << ", \"GBps\": " << chunk.gbps << '}';
      }
      json << ']';
    }
    json << '}';
  }
  if (options.stream_mode == StreamMode::kPerEdge) {
    json << "\n  ],\n  \"edgeResults\": [\n";
    for (size_t index = 0; index < edge_results.size(); ++index) {
      if (index != 0) json << ",\n";
      const EdgeResult& result = edge_results[index];
      json << "    {\"source\": " << result.source
           << ", \"destination\": " << result.destination
           << ", \"order\": " << result.order
           << ", \"stream\": " << result.stream
           << ", \"elapsedMs\": " << result.elapsed_ms
           << ", \"GBps\": " << result.gbps;
      if (options.chunk_repeats > 0) {
        json << ", \"chunks\": [";
        for (size_t chunk_index = 0;
             chunk_index < result.chunks.size(); ++chunk_index) {
          if (chunk_index != 0) json << ", ";
          const ChunkResult& chunk = result.chunks[chunk_index];
          json << "{\"index\": " << chunk.index
               << ", \"repeats\": " << chunk.repeats
               << ", \"elapsedMs\": " << chunk.elapsed_ms
               << ", \"GBps\": " << chunk.gbps << '}';
        }
        json << ']';
      }
      json << '}';
    }
    json << "\n  ]\n}\n";
  } else {
    json << "\n  ]\n}\n";
  }
  return json.str();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::vector<DeviceResources> resources;
  std::vector<StreamResources> streams;
  std::vector<std::vector<void*>> destinations;
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
    const std::vector<std::vector<int>> source_edges =
        makeSourceEdgeLists(options, edges);
    std::vector<int> edge_positions(edges.size(), 0);
    for (const std::vector<int>& edges_for_source : source_edges) {
      for (size_t position = 0; position < edges_for_source.size();
           ++position) {
        edge_positions[edges_for_source[position]] =
            static_cast<int>(position);
      }
    }
    for (const Edge& edge : edges) {
      enablePeerAccess(options.devices[edge.source_index],
                       options.devices[edge.destination_index]);
    }

    for (DeviceResources& resource : resources) {
      COPYBENCH_CUDA_CHECK(cudaSetDevice(resource.device));
      COPYBENCH_CUDA_CHECK(cudaMalloc(&resource.source, options.bytes));
      COPYBENCH_CUDA_CHECK(
          cudaMemset(resource.source, resource.device & 0xff, options.bytes));
    }

    for (size_t destination_index = 0;
         destination_index < options.devices.size(); ++destination_index) {
      COPYBENCH_CUDA_CHECK(cudaSetDevice(options.devices[destination_index]));
      for (size_t source_index = 0; source_index < options.devices.size();
           ++source_index) {
        if (source_index == destination_index) continue;
        COPYBENCH_CUDA_CHECK(cudaMalloc(
            &destinations[destination_index][source_index], options.bytes));
        COPYBENCH_CUDA_CHECK(cudaMemset(
            destinations[destination_index][source_index], 0, options.bytes));
      }
    }

    const size_t stream_count = options.stream_mode == StreamMode::kPerEdge
                                    ? edges.size()
                                    : resources.size() *
                                          options.streams_per_source;
    streams.resize(stream_count);
    for (size_t stream_index = 0; stream_index < streams.size();
         ++stream_index) {
      const int source_index = options.stream_mode == StreamMode::kPerEdge
                                   ? edges[stream_index].source_index
                                   : static_cast<int>(
                                         stream_index /
                                         options.streams_per_source);
      streams[stream_index].device = options.devices[source_index];
      COPYBENCH_CUDA_CHECK(cudaSetDevice(streams[stream_index].device));
      COPYBENCH_CUDA_CHECK(cudaStreamCreate(&streams[stream_index].stream));
      COPYBENCH_CUDA_CHECK(cudaEventCreate(&streams[stream_index].start));
      COPYBENCH_CUDA_CHECK(cudaEventCreate(&streams[stream_index].stop));
      const int chunk_count = chunkCount(options);
      streams[stream_index].chunk_starts.resize(chunk_count, nullptr);
      streams[stream_index].chunk_stops.resize(chunk_count, nullptr);
      for (int chunk = 0; chunk < chunk_count; ++chunk) {
        COPYBENCH_CUDA_CHECK(
            cudaEventCreate(&streams[stream_index].chunk_starts[chunk]));
        COPYBENCH_CUDA_CHECK(
            cudaEventCreate(&streams[stream_index].chunk_stops[chunk]));
      }
    }
    if (options.stream_dependency == StreamDependency::kSourceChain) {
      const int total_iterations = kWarmupIterations + options.repeats;
      for (StreamResources& stream_resource : streams) {
        COPYBENCH_CUDA_CHECK(cudaSetDevice(stream_resource.device));
        stream_resource.dependency_events.resize(total_iterations, nullptr);
        for (int iteration = 0; iteration < total_iterations; ++iteration) {
          COPYBENCH_CUDA_CHECK(cudaEventCreateWithFlags(
              &stream_resource.dependency_events[iteration],
              cudaEventDisableTiming));
        }
      }
    }

    std::vector<SourceResult> results(options.devices.size());
    for (size_t index = 0; index < results.size(); ++index) {
      results[index].device = options.devices[index];
      for (const Edge& edge : edges) {
        if (edge.source_index == static_cast<int>(index)) {
          ++results[index].outgoing;
        }
      }
    }

    std::vector<EdgeResult> edge_results;
    if (options.stream_mode == StreamMode::kPerEdge) {
      edge_results.resize(edges.size());
      for (size_t index = 0; index < edges.size(); ++index) {
        edge_results[index].source = options.devices[edges[index].source_index];
        edge_results[index].destination =
            options.devices[edges[index].destination_index];
        edge_results[index].order = static_cast<int>(index);
        edge_results[index].stream = static_cast<int>(index);
      }
    }

    std::cout << "D2D multi-GPU peer memcpy benchmark\n"
              << "pattern=" << patternName(options.pattern)
              << " edgeOrder=" << edgeOrderName(options.edge_order)
              << " streamMode=" << streamModeName(options.stream_mode)
              << " streamDependency="
              << streamDependencyName(options.stream_dependency)
              << " streamsPerSource=" << options.streams_per_source
              << " devices=" << copybench::joinDevices(options.devices)
              << " repeats=" << options.repeats
              << " warmup=" << kWarmupIterations
              << " bytes=" << options.bytes
              << " directions=" << edges.size();
    if (options.stream_assignment_explicit) {
      std::cout << " streamAssignment=";
      for (size_t index = 0; index < options.stream_assignment.size();
           ++index) {
        if (index != 0) std::cout << ',';
        std::cout << options.stream_assignment[index];
      }
    }
    if (options.edge_permutation_explicit) {
      std::cout << " edgePermutation=";
      for (size_t index = 0; index < options.edge_permutation.size();
           ++index) {
        if (index != 0) std::cout << ',';
        std::cout << options.edge_permutation[index];
      }
    }
    if (options.source_offsets_explicit) {
      std::cout << " sourceOffsetsUs=";
      for (size_t index = 0; index < options.source_offsets_us.size();
           ++index) {
        if (index != 0) std::cout << ',';
        std::cout << options.source_offsets_us[index];
      }
    }
    if (options.chunk_repeats > 0) {
      std::cout << " chunkRepeats=" << options.chunk_repeats
                << " chunkCount=" << chunkCount(options);
    }
    if (options.sync_each_iteration) {
      std::cout << " syncEachIteration=1";
    }
    std::cout << '\n';
    std::cout.flush();

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
      issueRound(options, resources, streams, destinations, edges,
                 source_edges, edge_positions, iteration);
      for (const StreamResources& stream_resource : streams) {
        COPYBENCH_CUDA_CHECK(cudaSetDevice(stream_resource.device));
        COPYBENCH_CUDA_CHECK(
          cudaStreamSynchronize(stream_resource.stream));
      }
    }

    enqueueSourceOffsets(options, streams);
    for (StreamResources& stream_resource : streams) {
      COPYBENCH_CUDA_CHECK(cudaSetDevice(stream_resource.device));
      COPYBENCH_CUDA_CHECK(
          cudaEventRecord(stream_resource.start, stream_resource.stream));
    }
    if (options.chunk_repeats == 0) {
      for (int iteration = 0; iteration < options.repeats; ++iteration) {
        issueRound(options, resources, streams, destinations, edges,
                   source_edges, edge_positions,
                   kWarmupIterations + iteration);
        if (options.sync_each_iteration) {
          for (const StreamResources& stream_resource : streams) {
            COPYBENCH_CUDA_CHECK(cudaSetDevice(stream_resource.device));
            COPYBENCH_CUDA_CHECK(
                cudaStreamSynchronize(stream_resource.stream));
          }
        }
      }
    } else {
      for (int chunk = 0; chunk < chunkCount(options); ++chunk) {
        for (StreamResources& stream_resource : streams) {
          COPYBENCH_CUDA_CHECK(cudaSetDevice(stream_resource.device));
          COPYBENCH_CUDA_CHECK(cudaEventRecord(
              stream_resource.chunk_starts[chunk], stream_resource.stream));
        }
        for (int iteration = 0; iteration < chunkLength(options, chunk);
             ++iteration) {
          issueRound(options, resources, streams, destinations, edges,
                     source_edges, edge_positions,
                     kWarmupIterations + chunk * options.chunk_repeats +
                         iteration);
          if (options.sync_each_iteration) {
            for (const StreamResources& stream_resource : streams) {
              COPYBENCH_CUDA_CHECK(cudaSetDevice(stream_resource.device));
              COPYBENCH_CUDA_CHECK(
                  cudaStreamSynchronize(stream_resource.stream));
            }
          }
        }
        for (StreamResources& stream_resource : streams) {
          COPYBENCH_CUDA_CHECK(cudaSetDevice(stream_resource.device));
          COPYBENCH_CUDA_CHECK(cudaEventRecord(
              stream_resource.chunk_stops[chunk], stream_resource.stream));
        }
      }
    }
    for (StreamResources& stream_resource : streams) {
      COPYBENCH_CUDA_CHECK(cudaSetDevice(stream_resource.device));
      COPYBENCH_CUDA_CHECK(
          cudaEventRecord(stream_resource.stop, stream_resource.stream));
    }

    std::vector<float> stream_elapsed_ms(streams.size(), 0.0f);
    std::vector<std::vector<ChunkResult>> stream_chunks(streams.size());
    for (size_t index = 0; index < streams.size(); ++index) {
      StreamResources& stream_resource = streams[index];
      COPYBENCH_CUDA_CHECK(cudaSetDevice(stream_resource.device));
      COPYBENCH_CUDA_CHECK(cudaEventSynchronize(stream_resource.stop));
      COPYBENCH_CUDA_CHECK(cudaEventElapsedTime(
          &stream_elapsed_ms[index], stream_resource.start,
          stream_resource.stop));
      if (stream_elapsed_ms[index] <= 0.0f) {
        throw std::runtime_error(
            "CUDA event resolution was insufficient for the selected size/repeats");
      }

      if (options.chunk_repeats > 0) {
        stream_chunks[index].reserve(chunkCount(options));
        for (int chunk = 0; chunk < chunkCount(options); ++chunk) {
          COPYBENCH_CUDA_CHECK(
              cudaEventSynchronize(stream_resource.chunk_stops[chunk]));
          float elapsed_ms = 0.0f;
          COPYBENCH_CUDA_CHECK(cudaEventElapsedTime(
              &elapsed_ms, stream_resource.chunk_starts[chunk],
              stream_resource.chunk_stops[chunk]));
          if (elapsed_ms <= 0.0f) {
            throw std::runtime_error(
                "CUDA event resolution was insufficient for chunk timing");
          }
          const int chunk_repeats = chunkLength(options, chunk);
          const double chunk_seconds =
              static_cast<double>(elapsed_ms) / 1000.0;
          stream_chunks[index].push_back(
              {chunk, chunk_repeats, elapsed_ms,
               static_cast<double>(options.bytes) * chunk_repeats /
                   chunk_seconds / 1e9});
        }
      }
    }

    if (options.stream_mode == StreamMode::kPerEdge) {
      for (size_t index = 0; index < edge_results.size(); ++index) {
        edge_results[index].elapsed_ms = stream_elapsed_ms[index];
        const double seconds =
            static_cast<double>(stream_elapsed_ms[index]) / 1000.0;
        edge_results[index].gbps =
            static_cast<double>(options.bytes) * options.repeats / seconds /
            1e9;
        edge_results[index].chunks = stream_chunks[index];
      }
    }

    float max_elapsed_ms = 0.0f;
    for (size_t source_index = 0; source_index < results.size();
         ++source_index) {
      SourceResult& result = results[source_index];
      if (options.stream_mode == StreamMode::kPerSource &&
          options.streams_per_source == 1) {
        result.elapsed_ms = stream_elapsed_ms[source_index];
        result.chunks = stream_chunks[source_index];
      } else {
        for (size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
          if (edges[edge_index].source_index ==
              static_cast<int>(source_index)) {
            const int stream_index = streamIndexForEdge(
                options, static_cast<int>(edge_index), edges[edge_index],
                edge_positions);
            result.elapsed_ms =
                std::max(result.elapsed_ms, stream_elapsed_ms[stream_index]);
          }
        }
        if (options.chunk_repeats > 0) {
          result.chunks.reserve(chunkCount(options));
          for (int chunk = 0; chunk < chunkCount(options); ++chunk) {
            float elapsed_ms = 0.0f;
            for (size_t edge_index = 0; edge_index < edges.size();
                 ++edge_index) {
              if (edges[edge_index].source_index ==
                  static_cast<int>(source_index)) {
                const int stream_index = streamIndexForEdge(
                    options, static_cast<int>(edge_index), edges[edge_index],
                    edge_positions);
                elapsed_ms = std::max(
                    elapsed_ms, stream_chunks[stream_index][chunk].elapsed_ms);
              }
            }
            const int chunk_repeats = chunkLength(options, chunk);
            const double chunk_seconds =
                static_cast<double>(elapsed_ms) / 1000.0;
            result.chunks.push_back(
                {chunk, chunk_repeats, elapsed_ms,
                 static_cast<double>(options.bytes) * chunk_repeats *
                     result.outgoing / chunk_seconds / 1e9});
          }
        }
      }
      if (result.elapsed_ms <= 0.0f) {
        throw std::runtime_error(
            "CUDA event resolution was insufficient for source timing");
      }
      max_elapsed_ms = std::max(max_elapsed_ms, result.elapsed_ms);
      const double seconds = static_cast<double>(result.elapsed_ms) / 1000.0;
      result.gbps = static_cast<double>(options.bytes) * options.repeats *
                    result.outgoing / seconds / 1e9;
    }
    if (max_elapsed_ms <= 0.0f) {
      throw std::runtime_error(
          "CUDA event resolution was insufficient for aggregate timing");
    }

    const double aggregate_gbps =
        static_cast<double>(options.bytes) * options.repeats * edges.size() /
        (static_cast<double>(max_elapsed_ms) / 1000.0) / 1e9;
    for (const SourceResult& result : results) {
      std::cout << std::fixed << std::setprecision(3)
                << "source[" << result.device << "] outgoing="
                << result.outgoing << " elapsed_ms=" << result.elapsed_ms
                << " GBps=" << result.gbps << '\n';
    }
    std::cout << std::fixed << std::setprecision(3)
              << "aggregateGBps=" << aggregate_gbps << '\n';

    std::vector<std::string> device_names;
    device_names.reserve(options.devices.size());
    for (int device : options.devices) {
      cudaDeviceProp properties{};
      COPYBENCH_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
      device_names.emplace_back(properties.name);
    }
    copybench::writeTextFile(
        options.output,
        makeJson(options, results, edge_results, edges,
                 static_cast<int>(edges.size()), aggregate_gbps,
                 device_names));
    cleanup(resources, streams, destinations);
    return 0;
  } catch (const std::exception& error) {
    cleanup(resources, streams, destinations);
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
