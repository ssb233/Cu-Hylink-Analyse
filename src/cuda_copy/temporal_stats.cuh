#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

namespace copybench {

struct TimingStats {
  size_t count = 0;
  double p50_ms = 0.0;
  double p90_ms = 0.0;
  double p99_ms = 0.0;
  double max_ms = 0.0;
};

inline double nearestRankMs(std::vector<unsigned long long> values,
                            double quantile) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double rank = std::ceil(quantile * static_cast<double>(values.size()));
  const size_t index = static_cast<size_t>(std::max(1.0, rank)) - 1;
  return static_cast<double>(values[std::min(index, values.size() - 1)]) /
         1.0e6;
}

inline TimingStats summarizeNanoseconds(
    const std::vector<unsigned long long>& values) {
  TimingStats stats;
  stats.count = values.size();
  if (values.empty()) return stats;
  stats.p50_ms = nearestRankMs(values, 0.50);
  stats.p90_ms = nearestRankMs(values, 0.90);
  stats.p99_ms = nearestRankMs(values, 0.99);
  stats.max_ms = static_cast<double>(*std::max_element(values.begin(),
                                                        values.end())) /
                 1.0e6;
  return stats;
}

inline std::vector<unsigned long long> durationNanoseconds(
    const std::vector<unsigned long long>& begin_ns,
    const std::vector<unsigned long long>& end_ns) {
  const size_t count = std::min(begin_ns.size(), end_ns.size());
  std::vector<unsigned long long> values;
  values.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    values.push_back(end_ns[index] >= begin_ns[index]
                         ? end_ns[index] - begin_ns[index]
                         : 0);
  }
  return values;
}

inline std::vector<unsigned long long> submitIntervalsNanoseconds(
    const std::vector<unsigned long long>& begin_ns) {
  std::vector<unsigned long long> values;
  if (begin_ns.size() < 2) return values;
  values.reserve(begin_ns.size() - 1);
  for (size_t index = 1; index < begin_ns.size(); ++index) {
    values.push_back(begin_ns[index] >= begin_ns[index - 1]
                         ? begin_ns[index] - begin_ns[index - 1]
                         : 0);
  }
  return values;
}

inline std::vector<unsigned long long> idleGapsNanoseconds(
    const std::vector<unsigned long long>& begin_ns,
    const std::vector<unsigned long long>& end_ns) {
  const size_t count = std::min(begin_ns.size(), end_ns.size());
  std::vector<unsigned long long> values;
  if (count < 2) return values;
  values.reserve(count - 1);
  for (size_t index = 1; index < count; ++index) {
    values.push_back(begin_ns[index] > end_ns[index - 1]
                         ? begin_ns[index] - end_ns[index - 1]
                         : 0);
  }
  return values;
}

inline double sumNanoseconds(
    const std::vector<unsigned long long>& values) {
  unsigned long long total = 0;
  for (unsigned long long value : values) total += value;
  return static_cast<double>(total) / 1.0e9;
}

inline void writeTimingStats(std::ostream& output,
                             const TimingStats& stats) {
  output << "{\"count\": " << stats.count << ", \"p50\": "
         << stats.p50_ms << ", \"p90\": " << stats.p90_ms
         << ", \"p99\": " << stats.p99_ms << ", \"max\": "
         << stats.max_ms << "}";
}

}  // namespace copybench
