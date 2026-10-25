// P6 portable statistics. No MPI, GPU, or Tpetra dependency.
#ifndef TPETRA_PERFORMANCE_NVSHMEM_P6_STATS_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P6_STATS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace NvshmemP6Stats {
struct Summary {
  double min = 0;
  double p10 = 0;
  double median = 0;
  double p90 = 0;
  double max = 0;
};
inline double percentile(const std::vector<double>& sorted, double p) {
  if (sorted.empty() || p < 0.0 || p > 1.0)
    throw std::runtime_error("P6 percentile input invalid");
  const double i = p * static_cast<double>(sorted.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(i));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(i));
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}
inline Summary summarize(std::vector<double> x) {
  if (x.empty()) throw std::runtime_error("P6 empty sample series");
  for (double d : x)
    if (!std::isfinite(d) || d < 0.0)
      throw std::runtime_error("P6 invalid/nonfinite timing sample");
  std::sort(x.begin(), x.end());
  return {x.front(), percentile(x, 0.10), percentile(x, 0.50),
          percentile(x, 0.90), x.back()};
}
inline std::vector<double> maxRankByIteration(
    const std::vector<double>& rankMajor, int ranks, int reps) {
  if (ranks <= 0 || reps <= 0 ||
      rankMajor.size() != static_cast<std::size_t>(ranks) *
                          static_cast<std::size_t>(reps))
    throw std::runtime_error("P6 rank-major sample dimensions invalid");
  std::vector<double> result(static_cast<std::size_t>(reps), 0.0);
  for (int rank = 0; rank < ranks; ++rank)
    for (int i = 0; i < reps; ++i) {
      const double d = rankMajor[static_cast<std::size_t>(rank) * reps + i];
      if (!std::isfinite(d) || d < 0.0)
        throw std::runtime_error("P6 invalid rank sample");
      result[static_cast<std::size_t>(i)] =
          std::max(result[static_cast<std::size_t>(i)], d);
    }
  return result;
}
} // namespace NvshmemP6Stats
#endif
