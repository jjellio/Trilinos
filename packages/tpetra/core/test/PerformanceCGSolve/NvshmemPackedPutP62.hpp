// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
// P6.2: two ordinary CUDA kernels, two independent CUDA streams, and a
// CALLER-ISSUED MPI_Barrier immediately before each Import. No ACK, no
// credit/ack buffer, no MPI payload, and no intermediate device fence.
// Experimental nonblocking NVSHMEM-test receiver: GPU progress and occupancy
// MUST be validated on the real hardware; it is NOT an asynchronous Import.
#ifndef TPETRA_PERFORMANCE_NVSHMEM_PACKED_PUT_P62_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_PACKED_PUT_P62_HPP

#include "NvshmemPackedPut.hpp" // cudaCheck, P3::upload and common baseline
#include "PackedPutP62Plan.hpp"
#include "NvshmemP71Probes.hpp"
#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace NvshmemPackedPutP62 {
using U64 = PackedPutP51::U64;
using Send = PackedPutP62::SendDescriptor;
using Receive = PackedPutP62::ReceiveDescriptor;
using Local = PackedPutP51::LocalEntry;
using NvshmemPackedPut::cudaCheck;
using NvshmemPackedPut::require;

enum class PutMode { Thread, Warp, Block };
inline PutMode parseMode(const std::string& name) {
  require(name == "thread" || name == "warp" || name == "block",
          "P6.2 PUT mode must be thread|warp|block");
  if (name == "warp") return PutMode::Warp;
  if (name == "block") return PutMode::Block;
  return PutMode::Thread;
}

// One descriptor is one NVSHMEM PUT+signal and one independent completion.
// The sender kernel has a disjoint set of blocks for local copies, so that
// the ENTIRE local+remote send-side work is fused into one native kernel.
// All warp/block collective calls are executed uniformly by their group.
// The ordinary Tpetra X/Y views remain non-symmetric CUDA allocations.
template<int Mode, class XView, class YView>
static __global__ void packAndPutKernel(
    const Send* sends, std::size_t nSend, std::size_t sendBlocks,
    const U64* sourceLids, double* packet, double* remoteInbox, U64* remoteReady,
    const Local* localEntries, std::size_t nLocal,
    XView x, YView y, U64 generation) {
  if (static_cast<std::size_t>(blockIdx.x) >= sendBlocks) {
    const std::size_t first =
        (static_cast<std::size_t>(blockIdx.x) - sendBlocks) * blockDim.x;
    const std::size_t i = first + threadIdx.x;
    if (i < nLocal) {
      const Local e = localEntries[i];
      y(e.destinationLid, 0) = x(e.sourceLid, 0);
    }
    return;
  }
  if constexpr (Mode == 0) {
    const std::size_t id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < nSend) {
      const Send d = sends[id];
      for (std::uint32_t j = 0; j < d.count; ++j)
        packet[d.packedOffset + j] = x(sourceLids[d.sourceOffset + j], 0);
      nvshmem_double_put_signal(remoteInbox + d.remoteInboxOffset,
          packet + d.packedOffset, static_cast<std::size_t>(d.count),
          remoteReady + d.signalIndex, generation, NVSHMEM_SIGNAL_SET, d.pe);
    }
  } else if constexpr (Mode == 1) {
    const unsigned int lane = threadIdx.x & 31U;
    const std::size_t group = static_cast<std::size_t>(threadIdx.x) / 32U;
    const std::size_t id = static_cast<std::size_t>(blockIdx.x) *
                               (blockDim.x / 32U) + group;
    if (id < nSend) { // branch is warp-uniform
      const Send d = sends[id];
      for (std::uint32_t j = lane; j < d.count; j += 32U)
        packet[d.packedOffset + j] = x(sourceLids[d.sourceOffset + j], 0);
      __syncwarp(); // all 32 lanes finish gathering before the warp PUT
      nvshmemx_double_put_signal_warp(remoteInbox + d.remoteInboxOffset,
          packet + d.packedOffset, static_cast<std::size_t>(d.count),
          remoteReady + d.signalIndex, generation, NVSHMEM_SIGNAL_SET, d.pe);
    }
  } else { // Mode == 2, one complete 128-thread block per descriptor
    const std::size_t id = static_cast<std::size_t>(blockIdx.x);
    if (id < nSend) { // branch is block-uniform
      const Send d = sends[id];
      for (std::uint32_t j = threadIdx.x; j < d.count; j += blockDim.x)
        packet[d.packedOffset + j] = x(sourceLids[d.sourceOffset + j], 0);
      __syncthreads();
      nvshmemx_double_put_signal_block(remoteInbox + d.remoteInboxOffset,
          packet + d.packedOffset, static_cast<std::size_t>(d.count),
          remoteReady + d.signalIndex, generation, NVSHMEM_SIGNAL_SET, d.pe);
    }
  }
}

// Fixed SMALL GPU footprint: at most four blocks by default, 4 warps each.
// One independently signaled receive descriptor is owned by each warp.
// Lane 0 uses a documented *nonblocking* symmetric-object test (not the
// blocking nvshmem_signal_wait_until API); all lanes then scatter the chunk.
// The spin loop is a progress experiment, NOT a proven portable solution for
// concurrent GPU kernel scheduling. Real two-node tests are mandatory.
// No remote consumer ACK is published: the NEXT MPI_Barrier, after every PE's
// cudaDeviceSynchronize, makes reuse of this inbox generation safe.
template<class YView>
static __global__ void waitAndUnpackKernel(
    const Receive* receives, std::size_t nRecv,
    const U64* destinationLids, const double* inbox, U64* ready,
    YView y, U64 generation) {
  const unsigned int lane = threadIdx.x & 31U;
  const std::size_t warpsPerBlock = static_cast<std::size_t>(blockDim.x) / 32U;
  const std::size_t warp = static_cast<std::size_t>(blockIdx.x) * warpsPerBlock +
                            static_cast<std::size_t>(threadIdx.x) / 32U;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * warpsPerBlock;
  for (std::size_t id = warp; id < nRecv; id += stride) {
    const Receive d = receives[id];
    if (lane == 0U) {
      while (!nvshmem_uint64_test(ready + d.signalIndex,
                                  NVSHMEM_CMP_EQ, generation))
        __nanosleep(64U);
    }
    __syncwarp(); // release waiter lane before any lane reads the inbox
    for (std::uint32_t j = lane; j < d.count; j += 32U)
      y(destinationLids[d.destinationOffset + j], 0) = inbox[d.inboxOffset + j];
    __syncwarp(); // finish this descriptor's scatter before reusing this warp
  }
}

struct HostTimings {
  double producerLaunchUs = 0.0;
  double receiverLaunchUs = 0.0;
  double completeUs = 0.0;
  double totalUs = 0.0;
};

// Opt-in P7.1 probe results, sampled only in the separate instrumented
// kernels. GPU event timings measure kernel residency on each individual
// stream (including waiting); they are NOT NIC transfer times. The clock64
// per-descriptor results are approximate SM-cycle intervals, NOT a global
// cross-SM timestamp or an additive critical-path decomposition.
struct P71ProfileResults {
  HostTimings host;
  double sendKernelEventUs = 0.0;
  double receiveKernelEventUs = 0.0;
  double readbackUs = 0.0; // excluded from the completed GPU Import time
  int clockKHz = 0;
  NvshmemP71Stats::Sampling sendSampling;
  NvshmemP71Stats::Sampling receiveSampling;
  NvshmemP71Stats::Sampling localSampling;
  std::vector<NvshmemP71Stats::Probe> send;
  std::vector<NvshmemP71Stats::Probe> receive;
  std::vector<NvshmemP71Stats::Probe> local;
};

struct Counters {
  U64 imports = 0;
  U64 producerKernelLaunches = 0;
  U64 receiverKernelLaunches = 0;
  U64 putCalls = 0;
  U64 putDoubles = 0;
  U64 expectedSignals = 0;
  U64 localCopied = 0;
  U64 acknowledgementSignals = 0; // invariant zero
};

template<class LO, class GO, class Node>
class Exchange {
 public:
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  Exchange(const PackedPutP51::Plan& base,
           const PackedPutP62::Plan& plan, PutMode mode,
           int requestedReceiverBlocks = 4,
           std::size_t p71SampleCap = 0)
      : base_(base), plan_(plan), mode_(mode),
        nSend_(plan.sends.size()), nRecv_(plan.receives.size()),
        nLocal_(base.local.size()) {
    require(base.finalized && plan.finalized &&
            requestedReceiverBlocks >= 1 && requestedReceiverBlocks <= 16,
            "P6.2 invalid finalized plan or receiver blocks");
    require(plan.sourceLids.size() == base.packElements &&
            plan.destinationLids.size() == base.inboxElements &&
            nRecv_ == plan.localReadyElements &&
            plan.symmetricReadyElements >= plan.localReadyElements,
            "P6.2 inconsistent persistent packet/ready plan");
    require(base.symmetricPackElements > 0 && base.symmetricInboxElements > 0 &&
            plan.symmetricReadyElements > 0, "P6.2 symmetric capacity is zero");
    for (const auto& d : plan.sends) {
      require(d.pe >= 0 && d.signalIndex < plan.symmetricReadyElements &&
              d.count > 0 && d.count <= plan.chunkDoubles &&
              d.sourceOffset <= base.packElements &&
              d.count <= base.packElements - d.sourceOffset &&
              d.packedOffset <= base.symmetricPackElements &&
              d.count <= base.symmetricPackElements - d.packedOffset &&
              d.remoteInboxOffset <= base.symmetricInboxElements &&
              d.count <= base.symmetricInboxElements - d.remoteInboxOffset,
              "P6.2 invalid outgoing descriptor");
    }
    for (const auto& d : plan.receives) {
      require(d.signalIndex < plan.localReadyElements &&
              d.count > 0 && d.count <= plan.chunkDoubles &&
              d.inboxOffset <= base.inboxElements &&
              d.count <= base.inboxElements - d.inboxOffset &&
              d.destinationOffset <= base.inboxElements &&
              d.count <= base.inboxElements - d.destinationOffset,
              "P6.2 invalid incoming descriptor");
    }
    int device = -1, smCount = 0;
    cudaCheck(cudaGetDevice(&device), "P6.2 query CUDA device");
    cudaCheck(cudaDeviceGetAttribute(&smCount,
              cudaDevAttrMultiProcessorCount, device), "P6.2 query SM count");
    // Reserve SM headroom for the producer. A bounded polling kernel is
    // necessary but not sufficient for portable cross-stream progress.
    receiverBlocks_ = static_cast<unsigned int>(std::min(
        requestedReceiverBlocks, std::max(1, smCount - 1)));
    cudaCheck(cudaStreamCreateWithFlags(&producerStream_, cudaStreamNonBlocking),
              "P6.2 create producer stream");
    cudaCheck(cudaStreamCreateWithFlags(&receiverStream_, cudaStreamNonBlocking),
              "P6.2 create receiver stream");
    // Every allocation is collective, identical size and call order across
    // all PEs, performed ONCE outside every measured Import.
    packed_ = static_cast<double*>(nvshmem_malloc(
        PackedPutP51::checkedBytes(base.symmetricPackElements,
                                   sizeof(double), "P6.2 packed storage")));
    require(packed_ != nullptr, "P6.2 nvshmem_malloc(pack) failed");
    inbox_ = static_cast<double*>(nvshmem_malloc(
        PackedPutP51::checkedBytes(base.symmetricInboxElements,
                                   sizeof(double), "P6.2 inbox storage")));
    require(inbox_ != nullptr, "P6.2 nvshmem_malloc(inbox) failed");
    ready_ = static_cast<U64*>(nvshmem_calloc(
        PackedPutP51::asSize(plan.symmetricReadyElements, "P6.2 ready slots"),
        sizeof(U64)));
    require(ready_ != nullptr, "P6.2 nvshmem_calloc(ready) failed");
    sendsDevice_ = NvshmemP3::upload("P6.2 PUT descriptors", plan.sends);
    receivesDevice_ = NvshmemP3::upload("P6.2 receive descriptors", plan.receives);
    sourceDevice_ = NvshmemP3::upload("P6.2 GID-sorted source LIDs", plan.sourceLids);
    destinationDevice_ = NvshmemP3::upload(
        "P6.2 GID-sorted destination LIDs", plan.destinationLids);
    localDevice_ = NvshmemP3::upload("P6.2 local copies", base.local);
    if (p71SampleCap != 0) {
      require(p71SampleCap <= 4096, "P7.1 sample cap exceeds 4096");
      sendSampling_ = NvshmemP71Stats::planSampling(nSend_, p71SampleCap);
      receiveSampling_ = NvshmemP71Stats::planSampling(nRecv_, p71SampleCap);
      localSampling_ = NvshmemP71Stats::planSampling(nLocal_, p71SampleCap);
      sendProbeDevice_ = Kokkos::View<NvshmemP71Stats::Probe*, Kokkos::CudaSpace>(
          "P7.1 send samples", std::max<std::size_t>(1, sendSampling_.samples));
      receiveProbeDevice_ = Kokkos::View<NvshmemP71Stats::Probe*, Kokkos::CudaSpace>(
          "P7.1 receive samples", std::max<std::size_t>(1, receiveSampling_.samples));
      localProbeDevice_ = Kokkos::View<NvshmemP71Stats::Probe*, Kokkos::CudaSpace>(
          "P7.1 local copy samples", std::max<std::size_t>(1, localSampling_.samples));
      clockKHz_ = 0;
      cudaCheck(cudaDeviceGetAttribute(&clockKHz_, cudaDevAttrClockRate, device), "P7.1 device clock properties");
      require(clockKHz_ > 0, "P7.1 cannot determine GPU SM clock rate");
      cudaCheck(cudaEventCreate(&sendStart_), "P7.1 CUDA producer start event");
      cudaCheck(cudaEventCreate(&sendEnd_), "P7.1 CUDA producer end event");
      cudaCheck(cudaEventCreate(&receiveStart_), "P7.1 CUDA receiver start event");
      cudaCheck(cudaEventCreate(&receiveEnd_), "P7.1 CUDA receiver end event");
      p71Enabled_ = true;
    }
    Kokkos::fence("P6.2 persistent GPU descriptors ready");
  }
  Exchange(const Exchange&) = delete;
  Exchange& operator=(const Exchange&) = delete;
  ~Exchange() {
    if (packed_ && std::uncaught_exceptions() == 0) release();
  }
  std::size_t sendDescriptorCount() const { return nSend_; }
  std::size_t receiveDescriptorCount() const { return nRecv_; }
  unsigned int receiverBlocks() const { return receiverBlocks_; }
  const Counters& counters() const { return counters_; }
  U64 generation() const { return generation_; }
  std::size_t symmetricBytes() const {
    return PackedPutP51::checkedBytes(base_.symmetricPackElements, sizeof(double),
               "P6.2 pack") +
           PackedPutP51::checkedBytes(base_.symmetricInboxElements, sizeof(double),
               "P6.2 inbox") +
           PackedPutP51::checkedBytes(plan_.symmetricReadyElements, sizeof(U64),
               "P6.2 ready");
  }

  // CONTRACT: caller MUST finish its prior iteration and call MPI_Barrier on
  // the same communicator on ALL PEs immediately before this invocation.
  // (The P6.2 benchmark enforces it for all warmup/direct/stress iterations.)
  // The barrier is deliberately outside the timer so timings are comparable
  // to Tpetra's fence/barrier/start/doImport/fence/stop measurement.
  // No intermediate cuda{Device,Stream}Synchronize or on-stream NVSHMEM wait.
  // Exactly TWO CUDA kernels are launched, even for an all-empty plan.
  void importAfterBarrier(const Vector& X, Vector& Y, HostTimings* times = nullptr) {
    require(packed_ && inbox_ && ready_, "P6.2 Import after release");
    require(generation_ < std::numeric_limits<U64>::max(),
            "P6.2 generation overflow: collectively recreate exchange");
    require(X.getMap()->getLocalNumElements() == base_.ownedLength &&
            Y.getMap()->getLocalNumElements() == base_.targetLength,
            "P6.2 vector dimensions differ from plan");
    const U64 epoch = generation_ + 1;
    auto x = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
    auto y = Y.getLocalViewDevice(Tpetra::Access::ReadWrite);
    using Clock = std::chrono::steady_clock;
    const auto wallStart = times ? Clock::now() : Clock::time_point{};
    constexpr unsigned int threads = 128U;
    std::size_t sendBlocks = 0;
    switch (mode_) {
      case PutMode::Thread: sendBlocks = (nSend_ + threads - 1) / threads; break;
      case PutMode::Warp: sendBlocks = (nSend_ + 3) / 4; break;
      case PutMode::Block: sendBlocks = nSend_; break;
    }
    const std::size_t localBlocks = (nLocal_ + threads - 1) / threads;
    const std::size_t totalBlocks = std::max<std::size_t>(1, sendBlocks + localBlocks);
    require(totalBlocks <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
            "P6.2 producer CUDA grid overflow");
#define P62_SEND_KERNEL(M) \
    packAndPutKernel<M><<<static_cast<unsigned int>(totalBlocks), threads, 0, producerStream_>>>( \
        sendsDevice_.data(), nSend_, sendBlocks, sourceDevice_.data(), \
        packed_, inbox_, ready_, localDevice_.data(), nLocal_, x, y, epoch)
    switch (mode_) {
      case PutMode::Thread: P62_SEND_KERNEL(0); break;
      case PutMode::Warp:   P62_SEND_KERNEL(1); break;
      case PutMode::Block:  P62_SEND_KERNEL(2); break;
    }
#undef P62_SEND_KERNEL
    cudaCheck(cudaGetLastError(), "P6.2 launch descriptor pack+PUT kernel");
    const auto producerSubmitted = times ? Clock::now() : Clock::time_point{};
    // This receiver may begin before, during, or after remote PUTs. Its
    // completion is independent of PUT arrival order and of sender PE count.
    waitAndUnpackKernel<<<receiverBlocks_, threads, 0, receiverStream_>>>(
        receivesDevice_.data(), nRecv_, destinationDevice_.data(),
        inbox_, ready_, y, epoch);
    cudaCheck(cudaGetLastError(), "P6.2 launch bounded receive/unpack kernel");
    const auto receiverSubmitted = times ? Clock::now() : Clock::time_point{};
    cudaCheck(cudaDeviceSynchronize(), "P6.2 two-stream device completion");
    if (times) {
      const auto completed = Clock::now();
      using Micros = std::chrono::duration<double, std::micro>;
      times->producerLaunchUs = Micros(producerSubmitted - wallStart).count();
      times->receiverLaunchUs = Micros(receiverSubmitted - producerSubmitted).count();
      times->completeUs = Micros(completed - receiverSubmitted).count();
      times->totalUs = Micros(completed - wallStart).count();
    }
    generation_ = epoch;
    ++counters_.imports;
    ++counters_.producerKernelLaunches;
    ++counters_.receiverKernelLaunches;
    counters_.putCalls += static_cast<U64>(nSend_);
    counters_.putDoubles += base_.packElements;
    counters_.expectedSignals += static_cast<U64>(nRecv_);
    counters_.localCopied += static_cast<U64>(nLocal_);
  }

  // P7.1 diagnostics only. This path does NOT replace B0 and its extra
  // cudaEventRecord/clock64 instructions necessarily change scheduling.
  // Neither a PUT API-return timestamp nor the receiver's first ready test
  // proves raw network latency. Source/target/GID semantics are unchanged.
  // CONTRACT: same caller-enforced start MPI_Barrier as importAfterBarrier().
  void importProfiledAfterBarrier(const Vector& X, Vector& Y,
                                  P71ProfileResults& result) {
    require(p71Enabled_ && packed_ && inbox_ && ready_,
            "P7.1 profiled Import requires persistent probes and symmetric memory");
    require(generation_ < std::numeric_limits<U64>::max(),
            "P7.1 generation overflow");
    require(X.getMap()->getLocalNumElements() == base_.ownedLength &&
            Y.getMap()->getLocalNumElements() == base_.targetLength,
            "P7.1 vector dimensions differ from plan");
    const U64 epoch = generation_ + 1;
    auto x = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
    auto y = Y.getLocalViewDevice(Tpetra::Access::ReadWrite);
    constexpr unsigned int threads = 128U;
    std::size_t sendBlocks = 0;
    switch (mode_) {
      case PutMode::Thread: sendBlocks = (nSend_ + threads - 1) / threads; break;
      case PutMode::Warp: sendBlocks = (nSend_ + 3) / 4; break;
      case PutMode::Block: sendBlocks = nSend_; break;
    }
    const std::size_t localBlocks = (nLocal_ + threads - 1) / threads;
    const std::size_t totalBlocks = std::max<std::size_t>(1, sendBlocks + localBlocks);
    require(totalBlocks <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
            "P7.1 producer CUDA grid overflow");
    using Clock = std::chrono::steady_clock;
    using Micros = std::chrono::duration<double, std::micro>;
    const auto wallStart = Clock::now();
    cudaCheck(cudaEventRecord(sendStart_, producerStream_),
              "P7.1 record sender begin");
#define P71_SEND_KERNEL(M) \
    NvshmemP71Probes::packAndPutProbed<Send, Local, M> \
      <<<static_cast<unsigned int>(totalBlocks), threads, 0, producerStream_>>>( \
        sendsDevice_.data(), nSend_, sendBlocks, sourceDevice_.data(), \
        packed_, inbox_, ready_, localDevice_.data(), nLocal_, x, y, epoch, \
        sendProbeDevice_.data(), sendSampling_.stride, \
        localProbeDevice_.data(), localSampling_.stride)
    switch (mode_) {
      case PutMode::Thread: P71_SEND_KERNEL(0); break;
      case PutMode::Warp: P71_SEND_KERNEL(1); break;
      case PutMode::Block: P71_SEND_KERNEL(2); break;
    }
#undef P71_SEND_KERNEL
    cudaCheck(cudaGetLastError(), "P7.1 launch instrumented producer");
    cudaCheck(cudaEventRecord(sendEnd_, producerStream_), "P7.1 record sender end");
    const auto producerSubmitted = Clock::now();
    cudaCheck(cudaEventRecord(receiveStart_, receiverStream_),
              "P7.1 record receiver begin");
    NvshmemP71Probes::waitAndUnpackProbed<Receive>
      <<<receiverBlocks_, threads, 0, receiverStream_>>>(
        receivesDevice_.data(), nRecv_, destinationDevice_.data(),
        inbox_, ready_, y, epoch, receiveProbeDevice_.data(), receiveSampling_.stride);
    cudaCheck(cudaGetLastError(), "P7.1 launch instrumented receiver");
    cudaCheck(cudaEventRecord(receiveEnd_, receiverStream_),
              "P7.1 record receiver end");
    const auto receiverSubmitted = Clock::now();
    cudaCheck(cudaDeviceSynchronize(), "P7.1 two-stream device completion");
    const auto completed = Clock::now();
    result.host.producerLaunchUs = Micros(producerSubmitted - wallStart).count();
    result.host.receiverLaunchUs = Micros(receiverSubmitted - producerSubmitted).count();
    result.host.completeUs = Micros(completed - receiverSubmitted).count();
    result.host.totalUs = Micros(completed - wallStart).count();
    float sendMs = 0.0F, receiveMs = 0.0F;
    cudaCheck(cudaEventElapsedTime(&sendMs, sendStart_, sendEnd_),
              "P7.1 producer event elapsed");
    cudaCheck(cudaEventElapsedTime(&receiveMs, receiveStart_, receiveEnd_),
              "P7.1 receiver event elapsed");
    result.sendKernelEventUs = static_cast<double>(sendMs) * 1000.0;
    result.receiveKernelEventUs = static_cast<double>(receiveMs) * 1000.0;
    result.clockKHz = clockKHz_;
    result.sendSampling = sendSampling_;
    result.receiveSampling = receiveSampling_;
    result.localSampling = localSampling_;
    const auto readbackStart = Clock::now();
    // A SINGLE post-completion readback per sampling class: no transfers or
    // host-device synchronizations inside the descriptor loops.
    const auto retrieve = [](std::vector<NvshmemP71Stats::Probe>& output,
                             const Kokkos::View<NvshmemP71Stats::Probe*, Kokkos::CudaSpace>& source,
                             std::size_t count, const char* stage) {
      output.resize(count);
      if (count)
        cudaCheck(cudaMemcpy(output.data(), source.data(),
                             count * sizeof(NvshmemP71Stats::Probe),
                             cudaMemcpyDeviceToHost), stage);
    };
    retrieve(result.send, sendProbeDevice_, sendSampling_.samples, "P7.1 send probe readback");
    retrieve(result.receive, receiveProbeDevice_, receiveSampling_.samples,
             "P7.1 receive probe readback");
    retrieve(result.local, localProbeDevice_, localSampling_.samples,
             "P7.1 local probe readback");
    result.readbackUs = Micros(Clock::now() - readbackStart).count();
    generation_ = epoch;
    ++counters_.imports;
    ++counters_.producerKernelLaunches;
    ++counters_.receiverKernelLaunches;
    counters_.putCalls += static_cast<U64>(nSend_);
    counters_.putDoubles += base_.packElements;
    counters_.expectedSignals += static_cast<U64>(nRecv_);
    counters_.localCopied += static_cast<U64>(nLocal_);
  }

  // Collective retirement; all PEs have completed their last Import before
  // nvshmem_free barriers can retire ready slots/inboxes safely.
  void release() {
    if (!packed_) return;
    cudaCheck(cudaDeviceSynchronize(), "P6.2 drain streams before free");
    if (p71Enabled_) {
      cudaCheck(cudaEventDestroy(receiveEnd_), "P7.1 destroy receiver end event");
      cudaCheck(cudaEventDestroy(receiveStart_), "P7.1 destroy receiver start event");
      cudaCheck(cudaEventDestroy(sendEnd_), "P7.1 destroy sender end event");
      cudaCheck(cudaEventDestroy(sendStart_), "P7.1 destroy sender start event");
      p71Enabled_ = false;
    }
    nvshmem_free(ready_);
    nvshmem_free(inbox_);
    nvshmem_free(packed_);
    ready_ = nullptr;
    inbox_ = nullptr;
    packed_ = nullptr;
    cudaCheck(cudaStreamDestroy(receiverStream_), "P6.2 destroy receiver stream");
    cudaCheck(cudaStreamDestroy(producerStream_), "P6.2 destroy producer stream");
    receiverStream_ = nullptr;
    producerStream_ = nullptr;
  }
 private:
  const PackedPutP51::Plan& base_;
  const PackedPutP62::Plan& plan_;
  PutMode mode_;
  std::size_t nSend_, nRecv_, nLocal_;
  unsigned int receiverBlocks_ = 1;
  double* packed_ = nullptr;
  double* inbox_ = nullptr;
  U64* ready_ = nullptr;
  cudaStream_t producerStream_ = nullptr;
  cudaStream_t receiverStream_ = nullptr;
  U64 generation_ = 0;
  Counters counters_;
  Kokkos::View<Send*, Kokkos::CudaSpace> sendsDevice_;
  Kokkos::View<Receive*, Kokkos::CudaSpace> receivesDevice_;
  Kokkos::View<U64*, Kokkos::CudaSpace> sourceDevice_;
  Kokkos::View<U64*, Kokkos::CudaSpace> destinationDevice_;
  Kokkos::View<Local*, Kokkos::CudaSpace> localDevice_;
  bool p71Enabled_ = false;
  int clockKHz_ = 0;
  NvshmemP71Stats::Sampling sendSampling_;
  NvshmemP71Stats::Sampling receiveSampling_;
  NvshmemP71Stats::Sampling localSampling_;
  Kokkos::View<NvshmemP71Stats::Probe*, Kokkos::CudaSpace> sendProbeDevice_;
  Kokkos::View<NvshmemP71Stats::Probe*, Kokkos::CudaSpace> receiveProbeDevice_;
  Kokkos::View<NvshmemP71Stats::Probe*, Kokkos::CudaSpace> localProbeDevice_;
  cudaEvent_t sendStart_ = nullptr, sendEnd_ = nullptr;
  cudaEvent_t receiveStart_ = nullptr, receiveEnd_ = nullptr;
};
} // namespace NvshmemPackedPutP62
#endif
