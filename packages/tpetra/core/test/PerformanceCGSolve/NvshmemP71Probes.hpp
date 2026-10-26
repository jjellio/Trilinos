// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
// P7.1 INSTRUMENTED CLONES of the P6.2 kernels. Keep B0 kernels unchanged.
// No protocol alteration: exactly one PUT_SIGNAL per descriptor and one
// independent NVSHMEM_TEST before any inbox scatter; no quiet/fence/ACK.
// Additional clocks, warp/block syncs, probes and CUDA events perturb GPU
// scheduling, so these timings are diagnostic, not production throughput.
#ifndef TPETRA_PERFORMANCE_NVSHMEM_P71_PROBES_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P71_PROBES_HPP

#include "NvshmemP71Stats.hpp"
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cstddef>
#include <cstdint>

namespace NvshmemP71Probes {
using U64 = std::uint64_t;
using NvshmemP71Stats::Probe;

template<class Send, class Local, int Mode, class XView, class YView>
static __global__ void packAndPutProbed(
    const Send* sends, std::size_t nSend, std::size_t sendBlocks,
    const U64* sourceLids, double* packet, double* remoteInbox, U64* remoteReady,
    const Local* localEntries, std::size_t nLocal,
    XView x, YView y, U64 generation,
    Probe* sendProbes, std::size_t sendStride,
    Probe* localProbes, std::size_t localStride) {
  if (static_cast<std::size_t>(blockIdx.x) >= sendBlocks) {
    const std::size_t first =
        (static_cast<std::size_t>(blockIdx.x) - sendBlocks) * blockDim.x;
    const std::size_t i = first + threadIdx.x;
    if (i < nLocal) {
      const bool sample = i % localStride == 0;
      const U64 before = sample ? clock64() : 0;
      const Local e = localEntries[i];
      y(e.destinationLid, 0) = x(e.sourceLid, 0);
      if (sample) localProbes[i / localStride] = {clock64() - before, 0, 0};
    }
    return;
  }
  if constexpr (Mode == 0) {
    const std::size_t id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < nSend) {
      const Send d = sends[id];
      const bool sample = id % sendStride == 0;
      const U64 before = sample ? clock64() : 0;
      for (std::uint32_t j = 0; j < d.count; ++j)
        packet[d.packedOffset + j] = x(sourceLids[d.sourceOffset + j], 0);
      const U64 afterPack = sample ? clock64() : 0;
      nvshmem_double_put_signal(remoteInbox + d.remoteInboxOffset,
          packet + d.packedOffset, static_cast<std::size_t>(d.count),
          remoteReady + d.signalIndex, generation, NVSHMEM_SIGNAL_SET, d.pe);
      if (sample) sendProbes[id / sendStride] =
          {afterPack - before, clock64() - afterPack, 0};
    }
  } else if constexpr (Mode == 1) {
    const unsigned int lane = threadIdx.x & 31U;
    const std::size_t group = static_cast<std::size_t>(threadIdx.x) / 32U;
    const std::size_t id = static_cast<std::size_t>(blockIdx.x) *
                               (blockDim.x / 32U) + group;
    if (id < nSend) { // group-uniform
      const Send d = sends[id];
      const bool sample = id % sendStride == 0;
      U64 before = 0;
      if (sample && lane == 0U) before = clock64();
      // Synchronize the sampled leader's start with the other pack lanes.
      if (sample) __syncwarp();
      for (std::uint32_t j = lane; j < d.count; j += 32U)
        packet[d.packedOffset + j] = x(sourceLids[d.sourceOffset + j], 0);
      __syncwarp(); // identical pack-completion requirement as B0
      U64 afterPack = 0;
      if (sample && lane == 0U) afterPack = clock64();
      nvshmemx_double_put_signal_warp(remoteInbox + d.remoteInboxOffset,
          packet + d.packedOffset, static_cast<std::size_t>(d.count),
          remoteReady + d.signalIndex, generation, NVSHMEM_SIGNAL_SET, d.pe);
      if (sample && lane == 0U) sendProbes[id / sendStride] =
          {afterPack - before, clock64() - afterPack, 0};
    }
  } else { // Mode==2, block-uniform
    const std::size_t id = static_cast<std::size_t>(blockIdx.x);
    if (id < nSend) {
      const Send d = sends[id];
      const bool sample = id % sendStride == 0;
      U64 before = 0;
      if (sample && threadIdx.x == 0U) before = clock64();
      if (sample) __syncthreads();
      for (std::uint32_t j = threadIdx.x; j < d.count; j += blockDim.x)
        packet[d.packedOffset + j] = x(sourceLids[d.sourceOffset + j], 0);
      __syncthreads(); // identical pack-completion requirement as B0
      U64 afterPack = 0;
      if (sample && threadIdx.x == 0U) afterPack = clock64();
      nvshmemx_double_put_signal_block(remoteInbox + d.remoteInboxOffset,
          packet + d.packedOffset, static_cast<std::size_t>(d.count),
          remoteReady + d.signalIndex, generation, NVSHMEM_SIGNAL_SET, d.pe);
      if (sample && threadIdx.x == 0U) sendProbes[id / sendStride] =
          {afterPack - before, clock64() - afterPack, 0};
    }
  }
}

template<class Receive, class YView>
static __global__ void waitAndUnpackProbed(
    const Receive* receives, std::size_t nRecv,
    const U64* destinationLids, const double* inbox, U64* ready,
    YView y, U64 generation, Probe* receiveProbes, std::size_t receiveStride) {
  const unsigned int lane = threadIdx.x & 31U;
  const std::size_t warpsPerBlock = static_cast<std::size_t>(blockDim.x) / 32U;
  const std::size_t warp = static_cast<std::size_t>(blockIdx.x) * warpsPerBlock +
                            static_cast<std::size_t>(threadIdx.x) / 32U;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * warpsPerBlock;
  for (std::size_t id = warp; id < nRecv; id += stride) {
    const Receive d = receives[id];
    const bool sample = id % receiveStride == 0;
    U64 before = 0, afterWait = 0, attempts = 0;
    if (lane == 0U) {
      if (sample) before = clock64();
      // Identical NVSHMEM predicate and snooze as the P6.2 receiver.
      // Count ONLY sampled descriptors; no extra signal operations for others.
      if (sample) {
        do {
          ++attempts;
          if (nvshmem_uint64_test(ready + d.signalIndex,
                                  NVSHMEM_CMP_EQ, generation)) break;
          __nanosleep(64U);
        } while (true);
        afterWait = clock64();
      } else {
        while (!nvshmem_uint64_test(ready + d.signalIndex,
                                    NVSHMEM_CMP_EQ, generation))
          __nanosleep(64U);
      }
    }
    __syncwarp(); // packet may be read only after its signal is observed
    for (std::uint32_t j = lane; j < d.count; j += 32U)
      y(destinationLids[d.destinationOffset + j], 0) = inbox[d.inboxOffset + j];
    __syncwarp();
    if (sample && lane == 0U) receiveProbes[id / receiveStride] =
        {afterWait - before, clock64() - afterWait, attempts};
  }
}
} // namespace NvshmemP71Probes
#endif
