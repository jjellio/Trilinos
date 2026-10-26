// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
// P7.1 fixed-size, POD, GPU instrumentation records and pure-host helpers.
// This header intentionally has NO dependency on CUDA, MPI, NVSHMEM or Tpetra.
#ifndef TPETRA_PERFORMANCE_NVSHMEM_P71_STATS_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P71_STATS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace NvshmemP71Stats {
using U64 = std::uint64_t;

// Sender: first=group gather+sync, second=PUT_SIGNAL API call/return.
// Receiver: first=signal-test loop, second=cooperative inbox scatter+sync.
// Local: first=one scalar copy, second=0.
// All clock64 intervals must be from the SAME CUDA thread, not two SMs.
struct Probe {
  U64 firstCycles;
  U64 secondCycles;
  U64 pollAttempts;
};

struct Sampling {
  std::size_t length = 0;
  std::size_t stride = 1;
  std::size_t samples = 0;
};

inline Sampling planSampling(std::size_t items, std::size_t cap) {
  if (cap == 0) throw std::invalid_argument("P7.1 sample capacity must be positive");
  if (items == 0) return {0, 1, 0};
  const std::size_t stride = items / cap + (items % cap != 0 ? 1U : 0U);
  const std::size_t samples = (items - 1) / stride + 1;
  if (samples > cap) throw std::logic_error("P7.1 sampling capacity exceeded");
  return {items, stride, samples};
}

inline std::size_t sampledId(const Sampling& plan, std::size_t sampleIndex) {
  if (sampleIndex >= plan.samples)
    throw std::out_of_range("P7.1 probe index outside sample range");
  return sampleIndex * plan.stride;
}

// cudaDeviceProp.clockRate is in kHz; clock64 represents GPU cycles.
// These are approximate on architectures with dynamic SM clocks and are
// unsuitable for comparing absolute timestamps across different SMs.
inline double cyclesToMicroseconds(U64 cycles, int clockKHz) {
  if (clockKHz <= 0) throw std::invalid_argument("P7.1 invalid GPU clock rate");
  return 1000.0 * static_cast<double>(cycles) / static_cast<double>(clockKHz);
}

inline double quantile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  if (p < 0.0 || p > 1.0) throw std::invalid_argument("P7.1 quantile outside [0,1]");
  std::sort(values.begin(), values.end());
  const double pos = p * static_cast<double>(values.size() - 1);
  const std::size_t low = static_cast<std::size_t>(pos);
  const std::size_t high = std::min(low + 1, values.size() - 1);
  return values[low] + (values[high] - values[low]) * (pos - static_cast<double>(low));
}

// Correct barrier-inclusive rank maximum: reduce (barrier + operation) per
// rank *before* taking each iteration's maximum. Do not sum independent
// per-stage rank maxima or medians.
inline std::vector<double> pairedFullIteration(
    const std::vector<double>& barriers, const std::vector<double>& operations) {
  if (barriers.size() != operations.size())
    throw std::invalid_argument("P7.1 barrier/operation sample count mismatch");
  std::vector<double> result(barriers.size());
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = barriers[i] + operations[i];
  return result;
}
} // namespace NvshmemP71Stats
#endif
