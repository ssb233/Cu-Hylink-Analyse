#include "copy_common.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

constexpr int kWarmupIterations = 10;
constexpr size_t kDefaultBytes = 255ULL * 1024ULL * 1024ULL;

enum class Mode {
  kUnidirectional,
  kBidirectional,
};

struct Options {
  Mode mode = Mode::kBidirectional;
  int src_device = 0;
  int dst_device = 1;
  int repeats = 20;
  size_t bytes = kDefaultBytes;
  std::string output;
};

struct DeviceResources {
  int device = -1;
  void* src = nullptr;
  void* dst = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
};

struct DirectionResult {
  int src = -1;
  int dst = -1;
  float elapsed_ms = 0.0f;
  double gbps = 0.0;
};

const char* modeName(Mode mode) {
  return mode == Mode::kBidirectional ? "bidirectional" : "unidirectional";
}

void printUsage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --mode=unidirectional|bidirectional  D2D traffic mode\n"
      << "  --srcDev=0                            source GPU\n"
      << "  --dstDev=1                            destination GPU\n"
      << "  --repeats=20                          measured repetitions\n"
      << "  --size=255M                            one cudaMemcpyPeerAsync size\n"
      << "  --output=<path>                        optional JSON output path\n"
      << "  --help                                 show this message\n"
      << "Defaults: mode=bidirectional, srcDev=0, dstDev=1, repeats=20, "
         "warmup=10 (fixed), size=255M\n";
}

int resourceIndex(const std::array<DeviceResources, 2>& resources, int device) {
  for (size_t index = 0; index < resources.size(); ++index) {
    if (resources[index].device == device) return static_cast<int>(index);
  }
  throw std::logic_error("device resource was not allocated");
}

void cleanup(std::array<DeviceResources, 2>& resources) noexcept {
  for (DeviceResources& resource : resources) {
    if (resource.device < 0) continue;
    cudaSetDevice(resource.device);
    if (resource.start != nullptr) cudaEventDestroy(resource.start);
    if (resource.stop != nullptr) cudaEventDestroy(resource.stop);
    if (resource.stream != nullptr) cudaStreamDestroy(resource.stream);
    if (resource.src != nullptr) cudaFree(resource.src);
    if (resource.dst != nullptr) cudaFree(resource.dst);
    resource.start = nullptr;
    resource.stop = nullptr;
    resource.stream = nullptr;
    resource.src = nullptr;
    resource.dst = nullptr;
  }
}

void enablePeerAccess(int src_device, int dst_device) {
  int can_access = 0;
  COPYBENCH_CUDA_CHECK(
      cudaDeviceCanAccessPeer(&can_access, src_device, dst_device));
  if (!can_access) {
    std::ostringstream message;
    message << "GPU " << src_device << " cannot access GPU " << dst_device;
    throw std::runtime_error(message.str());
  }

  COPYBENCH_CUDA_CHECK(cudaSetDevice(src_device));
  const cudaError_t status = cudaDeviceEnablePeerAccess(dst_device, 0);
  if (status == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
  } else {
    COPYBENCH_CUDA_CHECK(status);
  }
}

void issueCopy(const std::array<DeviceResources, 2>& resources, int src_device,
               int dst_device, size_t bytes) {
  const DeviceResources& source = resources[resourceIndex(resources, src_device)];
  const DeviceResources& destination =
      resources[resourceIndex(resources, dst_device)];
  COPYBENCH_CUDA_CHECK(cudaSetDevice(src_device));
  COPYBENCH_CUDA_CHECK(cudaMemcpyPeerAsync(
      destination.dst, dst_device, source.src, src_device, bytes,
      source.stream));
}

void synchronizeDirection(const std::array<DeviceResources, 2>& resources,
                          int src_device) {
  const DeviceResources& source = resources[resourceIndex(resources, src_device)];
  COPYBENCH_CUDA_CHECK(cudaSetDevice(src_device));
  COPYBENCH_CUDA_CHECK(cudaStreamSynchronize(source.stream));
}

void recordEvent(const std::array<DeviceResources, 2>& resources, int src_device,
                 cudaEvent_t event) {
  const DeviceResources& source = resources[resourceIndex(resources, src_device)];
  COPYBENCH_CUDA_CHECK(cudaSetDevice(src_device));
  COPYBENCH_CUDA_CHECK(cudaEventRecord(event, source.stream));
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
                                 {"--mode", "--pattern"}, &value)) {
      if (value == "unidirectional" || value == "one-way" ||
          value == "oneway") {
        options.mode = Mode::kUnidirectional;
      } else if (value == "bidirectional" || value == "allpairs" ||
                 value == "two-way" || value == "twoway") {
        options.mode = Mode::kBidirectional;
      } else {
        throw std::invalid_argument(
            "--mode must be unidirectional or bidirectional");
      }
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--srcDev", "--src-dev"}, &value)) {
      options.src_device = copybench::parseNonNegativeInt(value, "--srcDev");
    } else if (copybench::consumeOption(
                   index, argc, argv, {"--dstDev", "--dst-dev"}, &value)) {
      options.dst_device = copybench::parseNonNegativeInt(value, "--dstDev");
    } else if (copybench::consumeOption(index, argc, argv, {"--repeats"},
                                        &value)) {
      options.repeats = copybench::parsePositiveInt(value, "--repeats");
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

  if (options.src_device == options.dst_device) {
    throw std::invalid_argument("--srcDev and --dstDev must be different");
  }
  return options;
}

std::string makeJson(const Options& options,
                     const std::array<DirectionResult, 2>& results,
                     int direction_count, double aggregate_gbps,
                     const std::string& src_name,
                     const std::string& dst_name) {
  std::ostringstream json;
  json << std::fixed << std::setprecision(6);
  json << "{\n"
       << "  \"program\": \"d2d_peer_bw\",\n"
       << "  \"mode\": " << copybench::jsonQuote(modeName(options.mode))
       << ",\n"
       << "  \"srcDev\": " << options.src_device << ",\n"
       << "  \"dstDev\": " << options.dst_device << ",\n"
       << "  \"repeats\": " << options.repeats << ",\n"
       << "  \"warmup\": " << kWarmupIterations << ",\n"
       << "  \"bytes\": " << options.bytes << ",\n"
       << "  \"directions\": " << direction_count << ",\n"
       << "  \"aggregateGBps\": " << aggregate_gbps << ",\n"
       << "  \"devices\": {\n"
       << "    \"src\": " << copybench::jsonQuote(src_name) << ",\n"
       << "    \"dst\": " << copybench::jsonQuote(dst_name) << "\n"
       << "  },\n"
       << "  \"results\": [\n";
  for (int index = 0; index < direction_count; ++index) {
    if (index != 0) json << ",\n";
    json << "    {\"srcDev\": " << results[index].src
         << ", \"dstDev\": " << results[index].dst
         << ", \"elapsedMs\": " << results[index].elapsed_ms
         << ", \"GBps\": " << results[index].gbps << '}';
  }
  json << "\n  ]\n}\n";
  return json.str();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::array<DeviceResources, 2> resources{};
  try {
    options = parseOptions(argc, argv);

    int device_count = 0;
    COPYBENCH_CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (options.src_device >= device_count || options.dst_device >= device_count) {
      std::ostringstream message;
      message << "requested devices " << options.src_device << " and "
              << options.dst_device << ", but only " << device_count
              << " CUDA devices are available";
      throw std::invalid_argument(message.str());
    }

    enablePeerAccess(options.src_device, options.dst_device);
    if (options.mode == Mode::kBidirectional) {
      enablePeerAccess(options.dst_device, options.src_device);
    }

    resources[0].device = options.src_device;
    resources[1].device = options.dst_device;
    for (DeviceResources& resource : resources) {
      COPYBENCH_CUDA_CHECK(cudaSetDevice(resource.device));
      COPYBENCH_CUDA_CHECK(cudaMalloc(&resource.src, options.bytes));
      COPYBENCH_CUDA_CHECK(cudaMalloc(&resource.dst, options.bytes));
      COPYBENCH_CUDA_CHECK(
          cudaMemset(resource.src, resource.device & 0xff, options.bytes));
      COPYBENCH_CUDA_CHECK(cudaMemset(resource.dst, 0, options.bytes));
      COPYBENCH_CUDA_CHECK(cudaStreamCreate(&resource.stream));
      COPYBENCH_CUDA_CHECK(cudaEventCreate(&resource.start));
      COPYBENCH_CUDA_CHECK(cudaEventCreate(&resource.stop));
    }

    cudaDeviceProp src_properties{};
    cudaDeviceProp dst_properties{};
    COPYBENCH_CUDA_CHECK(
        cudaGetDeviceProperties(&src_properties, options.src_device));
    COPYBENCH_CUDA_CHECK(
        cudaGetDeviceProperties(&dst_properties, options.dst_device));

    std::cout << "D2D peer memcpy benchmark\n"
              << "mode=" << modeName(options.mode)
              << " srcDev=" << options.src_device
              << " dstDev=" << options.dst_device
              << " repeats=" << options.repeats
              << " warmup=" << kWarmupIterations
              << " bytes=" << options.bytes
              << " directions="
              << (options.mode == Mode::kBidirectional ? 2 : 1) << '\n'
              << "srcName=" << src_properties.name
              << " dstName=" << dst_properties.name << '\n';

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
      issueCopy(resources, options.src_device, options.dst_device, options.bytes);
      if (options.mode == Mode::kBidirectional) {
        issueCopy(resources, options.dst_device, options.src_device, options.bytes);
      }
      synchronizeDirection(resources, options.src_device);
      if (options.mode == Mode::kBidirectional) {
        synchronizeDirection(resources, options.dst_device);
      }
    }

    const int direction_count = options.mode == Mode::kBidirectional ? 2 : 1;
    std::array<DirectionResult, 2> results{};
    results[0].src = options.src_device;
    results[0].dst = options.dst_device;
    recordEvent(resources, options.src_device,
                resources[resourceIndex(resources, options.src_device)].start);
    if (options.mode == Mode::kBidirectional) {
      results[1].src = options.dst_device;
      results[1].dst = options.src_device;
      recordEvent(resources, options.dst_device,
                  resources[resourceIndex(resources, options.dst_device)].start);
    }

    for (int iteration = 0; iteration < options.repeats; ++iteration) {
      issueCopy(resources, options.src_device, options.dst_device, options.bytes);
      if (options.mode == Mode::kBidirectional) {
        issueCopy(resources, options.dst_device, options.src_device, options.bytes);
      }
    }

    recordEvent(resources, options.src_device,
                resources[resourceIndex(resources, options.src_device)].stop);
    if (options.mode == Mode::kBidirectional) {
      recordEvent(resources, options.dst_device,
                  resources[resourceIndex(resources, options.dst_device)].stop);
    }

    for (int index = 0; index < direction_count; ++index) {
      const int source_device = results[index].src;
      const DeviceResources& source =
          resources[resourceIndex(resources, source_device)];
      COPYBENCH_CUDA_CHECK(cudaSetDevice(source_device));
      COPYBENCH_CUDA_CHECK(cudaEventSynchronize(source.stop));
      COPYBENCH_CUDA_CHECK(cudaEventElapsedTime(
          &results[index].elapsed_ms, source.start, source.stop));
      const double seconds = static_cast<double>(results[index].elapsed_ms) / 1000.0;
      if (seconds <= 0.0) {
        throw std::runtime_error(
            "CUDA event resolution was insufficient for the selected size/repeats");
      }
      results[index].gbps =
          static_cast<double>(options.bytes) * options.repeats / seconds / 1e9;
    }

    const double max_ms = std::max(results[0].elapsed_ms,
                                   direction_count == 2 ? results[1].elapsed_ms
                                                        : results[0].elapsed_ms);
    if (max_ms <= 0.0) {
      throw std::runtime_error(
          "CUDA event resolution was insufficient for aggregate timing");
    }
    const double aggregate_gbps =
        static_cast<double>(options.bytes) * options.repeats * direction_count /
        (static_cast<double>(max_ms) / 1000.0) / 1e9;

    std::cout << std::fixed << std::setprecision(3)
              << "direction[0]=" << results[0].src << "->" << results[0].dst
              << " elapsed_ms=" << results[0].elapsed_ms
              << " GBps=" << results[0].gbps << '\n';
    if (direction_count == 2) {
      std::cout << "direction[1]=" << results[1].src << "->" << results[1].dst
                << " elapsed_ms=" << results[1].elapsed_ms
                << " GBps=" << results[1].gbps << '\n';
    }
    std::cout << "aggregateGBps=" << aggregate_gbps << '\n';

    const std::string json = makeJson(options, results, direction_count,
                                      aggregate_gbps, src_properties.name,
                                      dst_properties.name);
    copybench::writeTextFile(options.output, json);
    cleanup(resources);
    return 0;
  } catch (const std::exception& error) {
    cleanup(resources);
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
