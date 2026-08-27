#pragma once

#include <cuda_runtime.h>

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace copybench {

inline void checkCuda(cudaError_t status, const char* expression,
                      const char* file, int line) {
  if (status == cudaSuccess) return;
  std::ostringstream message;
  message << file << ':' << line << " CUDA call failed: " << expression
          << " -> " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

#define COPYBENCH_CUDA_CHECK(expression) \
  ::copybench::checkCuda((expression), #expression, __FILE__, __LINE__)

inline std::string trim(const std::string& input) {
  size_t begin = 0;
  while (begin < input.size() &&
         std::isspace(static_cast<unsigned char>(input[begin]))) {
    ++begin;
  }
  size_t end = input.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }
  return input.substr(begin, end - begin);
}

inline std::string lower(std::string input) {
  for (char& character : input) {
    character = static_cast<char>(
        std::tolower(static_cast<unsigned char>(character)));
  }
  return input;
}

inline bool consumeOption(int& index, int argc, char** argv,
                           std::initializer_list<const char*> names,
                           std::string* value) {
  const std::string argument(argv[index]);
  for (const char* raw_name : names) {
    const std::string name(raw_name);
    const std::string prefix = name + '=';
    if (argument.compare(0, prefix.size(), prefix) == 0) {
      *value = argument.substr(prefix.size());
      if (value->empty()) {
        throw std::invalid_argument(name + " requires a value");
      }
      return true;
    }
    if (argument == name) {
      if (index + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value");
      }
      *value = argv[++index];
      if (value->empty()) {
        throw std::invalid_argument(name + " requires a value");
      }
      return true;
    }
  }
  return false;
}

inline long long parseInteger(const std::string& text,
                              const std::string& option_name) {
  const std::string value = trim(text);
  if (value.empty()) {
    throw std::invalid_argument(option_name + " requires an integer");
  }
  errno = 0;
  char* end = nullptr;
  const long long parsed = std::strtoll(value.c_str(), &end, 10);
  if (errno == ERANGE || end == value.c_str() || *end != '\0') {
    throw std::invalid_argument(option_name + " has an invalid integer: " +
                                text);
  }
  return parsed;
}

inline int parseNonNegativeInt(const std::string& text,
                               const std::string& option_name) {
  const long long parsed = parseInteger(text, option_name);
  if (parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(option_name + " must be non-negative");
  }
  return static_cast<int>(parsed);
}

inline int parsePositiveInt(const std::string& text,
                            const std::string& option_name) {
  const long long parsed = parseInteger(text, option_name);
  if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(option_name + " must be positive");
  }
  return static_cast<int>(parsed);
}

inline size_t parseSize(const std::string& text,
                        const std::string& option_name) {
  const std::string value = trim(text);
  if (value.empty()) {
    throw std::invalid_argument(option_name + " requires a size");
  }

  errno = 0;
  char* end = nullptr;
  const double amount = std::strtod(value.c_str(), &end);
  if (errno == ERANGE || end == value.c_str() || amount <= 0.0 ||
      !std::isfinite(amount)) {
    throw std::invalid_argument(option_name + " has an invalid size: " +
                                text);
  }

  const std::string suffix = lower(trim(std::string(end)));
  long double multiplier = 1.0L;
  if (suffix == "k" || suffix == "kb" || suffix == "kib") {
    multiplier = 1024.0L;
  } else if (suffix == "m" || suffix == "mb" || suffix == "mib") {
    multiplier = 1024.0L * 1024.0L;
  } else if (suffix == "g" || suffix == "gb" || suffix == "gib") {
    multiplier = 1024.0L * 1024.0L * 1024.0L;
  } else if (suffix.empty() || suffix == "b") {
    multiplier = 1.0L;
  } else {
    throw std::invalid_argument(option_name + " has an unknown suffix: " +
                                suffix);
  }

  const long double bytes = static_cast<long double>(amount) * multiplier;
  if (bytes < 1.0L ||
      bytes > static_cast<long double>(std::numeric_limits<size_t>::max())) {
    throw std::invalid_argument(option_name + " is outside the supported range");
  }
  return static_cast<size_t>(bytes);
}

inline std::vector<int> parseDeviceList(const std::string& text,
                                        const std::string& option_name) {
  std::vector<int> devices;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const int device = parseNonNegativeInt(trim(item), option_name);
    for (int existing : devices) {
      if (existing == device) {
        throw std::invalid_argument(option_name + " contains duplicate device " +
                                    std::to_string(device));
      }
    }
    devices.push_back(device);
  }
  if (devices.empty()) {
    throw std::invalid_argument(option_name + " cannot be empty");
  }
  return devices;
}

inline std::string jsonQuote(const std::string& input) {
  std::ostringstream output;
  output << '"';
  for (unsigned char character : input) {
    switch (character) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << static_cast<int>(character)
                 << std::dec;
        } else {
          output << character;
        }
    }
  }
  output << '"';
  return output.str();
}

inline void writeTextFile(const std::string& path, const std::string& text) {
  if (path.empty() || path == "-") return;
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot open output file: " + path);
  }
  output << text;
  if (!output) {
    throw std::runtime_error("cannot write output file: " + path);
  }
}

inline bool fileExists(const std::string& path) {
  if (path.empty()) return false;
  std::ifstream input(path);
  return static_cast<bool>(input);
}

inline std::string joinDevices(const std::vector<int>& devices) {
  std::ostringstream output;
  for (size_t index = 0; index < devices.size(); ++index) {
    if (index != 0) output << ',';
    output << devices[index];
  }
  return output.str();
}

}  // namespace copybench
