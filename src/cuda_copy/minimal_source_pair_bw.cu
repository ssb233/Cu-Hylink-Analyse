#include "copy_common.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kWarmupIterations = 10;
constexpr size_t kDefaultBytes = 255ULL * 1024ULL * 1024ULL;

enum class StreamMode {
  kShared,
  kIndependent,
};

enum class StreamDependency {
  kNone,
  kSourceChain,
};

struct Edge {
  int source = 0;
  int destination = 0;
  int order = 0;
};

struct Options {
  std::vector<int> devices{0, 1, 2};
  std::vector<Edge> edges;
  StreamMode stream_mode = StreamMode::kShared;
  StreamDependency stream_dependency = StreamDependency::kNone;
  int repeats = 300;
  size_t bytes = kDefaultBytes;
  bool diagnostic = false;
  std::string output;
};

struct EdgeResources {
  std::vector<cudaEvent_t> operation_starts;
  std::vector<cudaEvent_t> operation_stops;
  std::vector<cudaEvent_t> dependency_events;
};

struct StreamResources {
  cudaStream_t stream = nullptr;
  cudaEvent_t block_start = nullptr;
  cudaEvent_t block_stop = nullptr;
};

struct TimingStats {
  int count = 0;
  double p50 = 0.0;
  double p90 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
};

const char* streamModeName(StreamMode mode) {
  return mode == StreamMode::kShared ? "shared" : "independent";
}

const char* dependencyName(StreamDependency dependency) {
  return dependency == StreamDependency::kSourceChain ? "source-chain"
                                                       : "none";
}

std::string compact(const std::string& input) {
  std::string result;
  result.reserve(input.size());
  for (char character : input) {
    if (!std::isspace(static_cast<unsigned char>(character))) {
      result.push_back(character);
    }
  }
  return result;
}

std::vector<Edge> parseEdgeOrder(const std::string& text,
                                 const std::vector<int>& devices) {
  if (devices.size() != 3 || devices[0] != 0 || devices[1] != 1 ||
      devices[2] != 2) {
    throw std::invalid_argument(
        "minimal source diagnostic requires --deviceList=0,1,2");
  }
  const std::string value = compact(text);
  if (value == "0->1,0->2") {
    return {{0, 1, 0}, {0, 2, 1}};
  }
  if (value == "0->2,0->1") {
    return {{0, 2, 0}, {0, 1, 1}};
  }
  throw std::invalid_argument(
      "--edgeOrder must be 0->1,0->2 or 0->2,0->1");
}

void printUsage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --deviceList=0,1,2                    fixed source/destination GPUs\n"
      << "  --edgeOrder=0->1,0->2|0->2,0->1       issue order per source\n"
      << "  --streamMode=shared|independent        one stream or one per edge\n"
      << "  --streamDependency=none|source-chain   optional explicit edge chain\n"
      << "  --repeats=300                          measured repetitions\n"
      << "  --size=255M                            bytes per edge copy\n"
      << "  --diagnostic=0|1                       record per-operation CUDA events\n"
      << "  --output=<path>                        JSON output path\n"
      << "  --help                                 show this message\n"
      << "Defaults: deviceList=0,1,2, edgeOrder=0->1,0->2, streamMode=shared, "
         "streamDependency=none, repeats=300, warmup=10, size=255M, "
         "diagnostic=0\n";
}

Options parseOptions(int argc, char** argv) {
  Options options;
  std::string edge_order_text = "0->1,0->2";
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
    } else if (copybench::consumeOption(index, argc, argv, {"--edgeOrder"},
                                        &value)) {
      edge_order_text = value;
    } else if (copybench::consumeOption(index, argc, argv, {"--streamMode"},
                                        &value)) {
      const std::string mode = copybench::lower(value);
      if (mode == "shared") {
        options.stream_mode = StreamMode::kShared;
      } else if (mode == "independent") {
        options.stream_mode = StreamMode::kIndependent;
      } else {
        throw std::invalid_argument(
            "--streamMode must be shared or independent");
      }
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
    } else if (copybench::consumeOption(index, argc, argv, {"--repeats"},
                                        &value)) {
      options.repeats = copybench::parsePositiveInt(value, "--repeats");
    } else if (copybench::consumeOption(index, argc, argv, {"--size"},
                                        &value)) {
      options.bytes = copybench::parseSize(value, "--size");
    } else if (copybench::consumeOption(index, argc, argv, {"--diagnostic"},
                                        &value)) {
      if (value == "0") {
        options.diagnostic = false;
      } else if (value == "1") {
        options.diagnostic = true;
      } else {
        throw std::invalid_argument("--diagnostic must be 0 or 1");
      }
    } else if (copybench::consumeOption(index, argc, argv, {"--output"},
                                        &value)) {
      options.output = value;
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  options.edges = parseEdgeOrder(edge_order_text, options.devices);
  return options;
}

void enablePeerAccess(int source, int destination) {
  int can_access = 0;
  COPYBENCH_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, source, destination));
  if (!can_access) {
    std::ostringstream message;
    message << "GPU " << source << " cannot access GPU " << destination;
    throw std::runtime_error(message.str());
  }
  COPYBENCH_CUDA_CHECK(cudaSetDevice(source));
  const cudaError_t status = cudaDeviceEnablePeerAccess(destination, 0);
  if (status == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
  } else {
    COPYBENCH_CUDA_CHECK(status);
  }
}

void cleanup(std::vector<StreamResources>* streams,
             std::vector<EdgeResources>* edge_resources, void* source,
             std::vector<void*>* destinations) noexcept {
  cudaSetDevice(0);
  for (StreamResources& resource : *streams) {
    if (resource.stream != nullptr) cudaStreamSynchronize(resource.stream);
  }
  for (EdgeResources& resource : *edge_resources) {
    for (cudaEvent_t event : resource.operation_starts) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    for (cudaEvent_t event : resource.operation_stops) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    for (cudaEvent_t event : resource.dependency_events) {
      if (event != nullptr) cudaEventDestroy(event);
    }
    resource.operation_starts.clear();
    resource.operation_stops.clear();
    resource.dependency_events.clear();
  }
  for (StreamResources& resource : *streams) {
    if (resource.block_start != nullptr) cudaEventDestroy(resource.block_start);
    if (resource.block_stop != nullptr) cudaEventDestroy(resource.block_stop);
    if (resource.stream != nullptr) cudaStreamDestroy(resource.stream);
    resource.block_start = nullptr;
    resource.block_stop = nullptr;
    resource.stream = nullptr;
  }
  if (source != nullptr) cudaFree(source);
  for (size_t index = 0; index < destinations->size(); ++index) {
    if ((*destinations)[index] == nullptr) continue;
    cudaSetDevice(static_cast<int>(index + 1));
    cudaFree((*destinations)[index]);
    (*destinations)[index] = nullptr;
  }
}

double nearestRank(const std::vector<double>& sorted_values, double quantile) {
  if (sorted_values.empty()) return 0.0;
  const size_t rank = static_cast<size_t>(std::ceil(
      quantile * static_cast<double>(sorted_values.size())));
  const size_t index = std::min(sorted_values.size() - 1, std::max<size_t>(1, rank) - 1);
  return sorted_values[index];
}

TimingStats summarize(std::vector<double> values) {
  TimingStats result;
  result.count = static_cast<int>(values.size());
  if (values.empty()) return result;
  std::sort(values.begin(), values.end());
  result.p50 = nearestRank(values, 0.50);
  result.p90 = nearestRank(values, 0.90);
  result.p99 = nearestRank(values, 0.99);
  result.max = values.back();
  return result;
}

void writeStats(std::ostringstream& json, const TimingStats& stats) {
  json << "{\"count\": " << stats.count << ", \"p50\": " << stats.p50
       << ", \"p90\": " << stats.p90 << ", \"p99\": " << stats.p99
       << ", \"max\": " << stats.max << '}';
}

void issueEdge(const Options& options, const Edge& edge, int edge_index,
               int iteration, int measured_index,
               const std::vector<StreamResources>& streams,
               const std::vector<EdgeResources>& edge_resources,
               void* source, const std::vector<void*>& destinations) {
  const int stream_index = options.stream_mode == StreamMode::kShared
                               ? 0
                               : edge_index;
  cudaStream_t stream = streams[stream_index].stream;
  if (options.stream_dependency == StreamDependency::kSourceChain) {
    cudaEvent_t dependency = nullptr;
    if (edge_index > 0) {
      dependency = edge_resources[edge_index - 1]
                        .dependency_events[iteration];
    } else if (iteration > 0) {
      dependency = edge_resources[options.edges.size() - 1]
                        .dependency_events[iteration - 1];
    }
    if (dependency != nullptr) {
      COPYBENCH_CUDA_CHECK(cudaStreamWaitEvent(stream, dependency, 0));
    }
  }
  if (options.diagnostic && measured_index >= 0) {
    COPYBENCH_CUDA_CHECK(cudaEventRecord(
        edge_resources[edge_index].operation_starts[measured_index], stream));
  }
  COPYBENCH_CUDA_CHECK(cudaMemcpyPeerAsync(
      destinations[edge.destination - 1], edge.destination, source, edge.source,
      options.bytes, stream));
  if (options.diagnostic && measured_index >= 0) {
    COPYBENCH_CUDA_CHECK(cudaEventRecord(
        edge_resources[edge_index].operation_stops[measured_index], stream));
  }
  if (options.stream_dependency == StreamDependency::kSourceChain) {
    COPYBENCH_CUDA_CHECK(cudaEventRecord(
        edge_resources[edge_index].dependency_events[iteration], stream));
  }
}

void synchronizeStreams(const std::vector<StreamResources>& streams) {
  for (const StreamResources& resource : streams) {
    COPYBENCH_CUDA_CHECK(cudaSetDevice(0));
    COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(resource.stream));
  }
}

std::string makeJson(
    const Options& options, const std::vector<StreamResources>& streams,
    const std::vector<EdgeResources>& edge_resources,
    const std::vector<float>& stream_elapsed_ms,
    const std::vector<double>& edge_elapsed_ms,
    const std::vector<TimingStats>& edge_duration_stats,
    double source_elapsed_ms, double aggregate_gbps) {
  std::ostringstream json;
  json << std::fixed << std::setprecision(6);
  json << "{\n"
       << "  \"program\": \"minimal_source_pair_bw\",\n"
       << "  \"devices\": [0, 1, 2],\n"
       << "  \"source\": 0,\n"
       << "  \"destinations\": [1, 2],\n"
       << "  \"edgeOrder\": [";
  for (size_t index = 0; index < options.edges.size(); ++index) {
    if (index != 0) json << ", ";
    json << copybench::jsonQuote(
        std::to_string(options.edges[index].source) + "->" +
        std::to_string(options.edges[index].destination));
  }
  json << "],\n"
       << "  \"streamMode\": "
       << copybench::jsonQuote(streamModeName(options.stream_mode)) << ",\n"
       << "  \"streamDependency\": "
       << copybench::jsonQuote(dependencyName(options.stream_dependency))
       << ",\n"
       << "  \"diagnostic\": " << (options.diagnostic ? "true" : "false")
       << ",\n"
       << "  \"streams\": " << streams.size() << ",\n"
       << "  \"repeats\": " << options.repeats << ",\n"
       << "  \"warmup\": " << kWarmupIterations << ",\n"
       << "  \"bytesPerMemcpy\": " << options.bytes << ",\n"
       << "  \"totalBytes\": "
       << static_cast<unsigned long long>(options.bytes) * options.repeats *
              options.edges.size()
       << ",\n"
       << "  \"sourceElapsedMs\": " << source_elapsed_ms << ",\n"
       << "  \"aggregateGBps\": " << aggregate_gbps << ",\n"
       << "  \"sourceResults\": [{\"device\": 0, \"outgoing\": 2, "
          "\"elapsedMs\": "
       << source_elapsed_ms << ", \"GBps\": " << aggregate_gbps << "}],\n"
       << "  \"streamElapsedMs\": [";
  for (size_t index = 0; index < stream_elapsed_ms.size(); ++index) {
    if (index != 0) json << ", ";
    json << stream_elapsed_ms[index];
  }
  json << "],\n  \"edgeResults\": [\n";
  for (size_t index = 0; index < options.edges.size(); ++index) {
    if (index != 0) json << ",\n";
    const Edge& edge = options.edges[index];
    const int stream_index = options.stream_mode == StreamMode::kShared
                                 ? 0
                                 : static_cast<int>(index);
    const double seconds = edge_elapsed_ms[index] / 1000.0;
    const double gbps = seconds > 0.0
                            ? static_cast<double>(options.bytes) *
                                  options.repeats / seconds / 1.0e9
                            : 0.0;
    json << "    {\"source\": " << edge.source
         << ", \"destination\": " << edge.destination
         << ", \"order\": " << edge.order
         << ", \"stream\": " << stream_index
         << ", \"operations\": " << options.repeats;
    if (options.diagnostic) {
      json << ", \"elapsedMs\": " << edge_elapsed_ms[index]
           << ", \"GBps\": " << gbps << ", \"operationDurationMs\": ";
      writeStats(json, edge_duration_stats[index]);
    } else {
      json << ", \"elapsedMs\": null, \"GBps\": null, "
              "\"operationDurationMs\": null";
    }
    json << '}';
  }
  json << "\n  ]\n}\n";
  return json.str();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::vector<StreamResources> streams;
  std::vector<EdgeResources> edge_resources;
  std::vector<void*> destinations(2, nullptr);
  void* source = nullptr;
  try {
    options = parseOptions(argc, argv);
    int device_count = 0;
    COPYBENCH_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 3) {
      throw std::runtime_error("minimal source diagnostic requires three GPUs");
    }
    for (int device : options.devices) {
      if (device < 0 || device >= device_count) {
        throw std::invalid_argument("requested device is not available");
      }
    }

    enablePeerAccess(0, 1);
    enablePeerAccess(0, 2);
    COPYBENCH_CUDA_CHECK(cudaSetDevice(0));
    COPYBENCH_CUDA_CHECK(cudaMalloc(&source, options.bytes));
    COPYBENCH_CUDA_CHECK(cudaMemset(source, 0, options.bytes));
    for (int destination = 1; destination <= 2; ++destination) {
      COPYBENCH_CUDA_CHECK(cudaSetDevice(destination));
      COPYBENCH_CUDA_CHECK(cudaMalloc(&destinations[destination - 1],
                                      options.bytes));
      COPYBENCH_CUDA_CHECK(cudaMemset(destinations[destination - 1], 0,
                                      options.bytes));
    }

    const size_t stream_count = options.stream_mode == StreamMode::kShared ? 1 : 2;
    streams.resize(stream_count);
    COPYBENCH_CUDA_CHECK(cudaSetDevice(0));
    for (StreamResources& resource : streams) {
      COPYBENCH_CUDA_CHECK(cudaStreamCreate(&resource.stream));
      COPYBENCH_CUDA_CHECK(cudaEventCreate(&resource.block_start));
      COPYBENCH_CUDA_CHECK(cudaEventCreate(&resource.block_stop));
    }
    edge_resources.resize(options.edges.size());
    if (options.diagnostic) {
      for (EdgeResources& resource : edge_resources) {
        resource.operation_starts.resize(options.repeats, nullptr);
        resource.operation_stops.resize(options.repeats, nullptr);
        for (int iteration = 0; iteration < options.repeats; ++iteration) {
          COPYBENCH_CUDA_CHECK(
              cudaEventCreate(&resource.operation_starts[iteration]));
          COPYBENCH_CUDA_CHECK(
              cudaEventCreate(&resource.operation_stops[iteration]));
        }
      }
    }
    if (options.stream_dependency == StreamDependency::kSourceChain) {
      const int total_iterations = kWarmupIterations + options.repeats;
      for (EdgeResources& resource : edge_resources) {
        resource.dependency_events.resize(total_iterations, nullptr);
        for (int iteration = 0; iteration < total_iterations; ++iteration) {
          COPYBENCH_CUDA_CHECK(cudaEventCreateWithFlags(
              &resource.dependency_events[iteration], cudaEventDisableTiming));
        }
      }
    }

    std::cout << "Minimal source-local peer memcpy benchmark\n"
              << "devices=0,1,2 edgeOrder="
              << (options.edges[0].destination == 1 ? "0->1,0->2" : "0->2,0->1")
              << " streamMode=" << streamModeName(options.stream_mode)
              << " streamDependency=" << dependencyName(options.stream_dependency)
              << " repeats=" << options.repeats
              << " warmup=" << kWarmupIterations
              << " bytes=" << options.bytes << '\n';
    std::cout.flush();

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
      for (size_t edge_index = 0; edge_index < options.edges.size(); ++edge_index) {
        issueEdge(options, options.edges[edge_index], static_cast<int>(edge_index),
                  iteration, -1, streams, edge_resources, source, destinations);
      }
      synchronizeStreams(streams);
    }

    for (StreamResources& resource : streams) {
      COPYBENCH_CUDA_CHECK(cudaEventRecord(resource.block_start, resource.stream));
    }
    for (int iteration = 0; iteration < options.repeats; ++iteration) {
      for (size_t edge_index = 0; edge_index < options.edges.size(); ++edge_index) {
        issueEdge(options, options.edges[edge_index], static_cast<int>(edge_index),
                  kWarmupIterations + iteration, iteration, streams,
                  edge_resources, source, destinations);
      }
    }
    for (StreamResources& resource : streams) {
      COPYBENCH_CUDA_CHECK(cudaEventRecord(resource.block_stop, resource.stream));
    }
    synchronizeStreams(streams);

    std::vector<float> stream_elapsed_ms(streams.size(), 0.0f);
    for (size_t index = 0; index < streams.size(); ++index) {
      COPYBENCH_CUDA_CHECK(cudaEventElapsedTime(
          &stream_elapsed_ms[index], streams[index].block_start,
          streams[index].block_stop));
    }
    std::vector<double> edge_elapsed_ms(options.edges.size(), 0.0);
    std::vector<TimingStats> edge_duration_stats(options.edges.size());
    if (options.diagnostic) {
      for (size_t edge_index = 0; edge_index < options.edges.size(); ++edge_index) {
        const EdgeResources& resource = edge_resources[edge_index];
        std::vector<double> durations;
        durations.reserve(options.repeats);
        for (int iteration = 0; iteration < options.repeats; ++iteration) {
          float duration_ms = 0.0f;
          COPYBENCH_CUDA_CHECK(cudaEventElapsedTime(
              &duration_ms, resource.operation_starts[iteration],
              resource.operation_stops[iteration]));
          durations.push_back(duration_ms);
        }
        edge_duration_stats[edge_index] = summarize(durations);
        float span_ms = 0.0f;
        COPYBENCH_CUDA_CHECK(cudaEventElapsedTime(
            &span_ms, resource.operation_starts.front(),
            resource.operation_stops.back()));
        edge_elapsed_ms[edge_index] = span_ms;
      }
    }
    const double source_elapsed_ms =
        *std::max_element(stream_elapsed_ms.begin(), stream_elapsed_ms.end());
    const double aggregate_gbps =
        static_cast<double>(options.bytes) * options.repeats * options.edges.size() /
        (source_elapsed_ms / 1000.0) / 1.0e9;
    if (options.diagnostic) {
      for (size_t edge_index = 0; edge_index < options.edges.size(); ++edge_index) {
        const Edge& edge = options.edges[edge_index];
        std::cout << std::fixed << std::setprecision(3)
                  << "edge[" << edge.source << "->" << edge.destination
                  << "] order=" << edge.order
                  << " elapsed_ms=" << edge_elapsed_ms[edge_index]
                  << " GBps="
                  << static_cast<double>(options.bytes) * options.repeats /
                         (edge_elapsed_ms[edge_index] / 1000.0) / 1.0e9
                  << '\n';
      }
    }
    std::cout << std::fixed << std::setprecision(3)
              << "source[0] elapsed_ms=" << source_elapsed_ms
              << " aggregateGBps=" << aggregate_gbps << '\n';
    copybench::writeTextFile(
        options.output,
        makeJson(options, streams, edge_resources, stream_elapsed_ms,
                 edge_elapsed_ms, edge_duration_stats, source_elapsed_ms,
                 aggregate_gbps));
    cleanup(&streams, &edge_resources, source, &destinations);
    return 0;
  } catch (const std::exception& error) {
    cleanup(&streams, &edge_resources, source, &destinations);
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
