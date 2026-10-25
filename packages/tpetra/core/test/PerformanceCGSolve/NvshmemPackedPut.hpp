// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
// P5.2: GPU-initiated, sender-packed NVSHMEM PUT+signal with a per-peer,
// generation-tagged consumption-credit protocol. NO MPI, collective CUDA
// kernel launch, or world barrier in import(). Collective symmetric malloc /
// free happens only at setup/retirement. See P52_POINT_TO_POINT_README.md.

#ifndef TPETRA_PERFORMANCE_NVSHMEM_PACKED_PUT_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_PACKED_PUT_HPP

#include "NvshmemP3.hpp"       // ordinary Kokkos::CudaSpace metadata upload
#include "PackedPutPlan.hpp" // authoritative P5.1 LID/packet mappings
#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <chrono>
#include <thread>
#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace NvshmemPackedPut {

inline void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error("NVSHMEM P5.2: " + message);
}
inline void cudaCheck(cudaError_t code, const char* what) {
  require(code == cudaSuccess,
          std::string(what) + ": " + cudaGetErrorString(code));
}

using U64 = PackedPutP51::U64;

// NVSHMEM's GPU RMA/signal operations are __device__ APIs. KOKKOS_LAMBDA
// expands to a __host__ __device__ operator(), which nvcc cannot compile
// when that operator directly calls a device-only API. Keep ONLY these two
// point-to-point communication operations in native device-only kernels.
// Neither kernel performs an NVSHMEM wait or collective; ordinary <<<>>>
// launches are correct and do NOT require nvshmemx_collective_launch.
//
// The static P5.1 descriptors already live in ordinary Kokkos::CudaSpace.
// Each send peer owns a disjoint destination inbox segment and ready slot.
static __global__ void packedPutSignalKernel(
    const PackedPutP51::SendPeer* sends, std::size_t nSend,
    double* remoteInbox, const double* packed, U64* remoteReady, U64 generation) {
  const std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = tid; i < nSend; i += stride) {
    const PackedPutP51::SendPeer e = sends[i];
    nvshmem_double_put_signal(
        remoteInbox + e.remoteInboxOffset, packed + e.packOffset,
        static_cast<std::size_t>(e.count), remoteReady + e.remoteSignalIndex,
        generation, NVSHMEM_SIGNAL_SET, e.pe);
  }
}

// P6.1 variant B: one *complete warp* per destination PE packet. The
// nvshmemx_*_warp call is collective within THAT warp only. All 32 lanes
// execute the same call with identical arguments; an unused entire warp
// skips it uniformly. It is NOT an inter-PE or cooperative-grid collective.
// One CUDA block contains four independent 32-lane warps.
static __global__ void packedPutSignalWarpKernel(
    const PackedPutP51::SendPeer* sends, std::size_t nSend,
    double* remoteInbox, const double* packed, U64* remoteReady, U64 generation) {
  const std::size_t warpsPerBlock = static_cast<std::size_t>(blockDim.x) / 32U;
  const std::size_t warpInBlock = static_cast<std::size_t>(threadIdx.x) / 32U;
  const std::size_t peer = static_cast<std::size_t>(blockIdx.x) * warpsPerBlock
                             + warpInBlock;
  if (peer < nSend) { // uniform branch across the entire warp
    const PackedPutP51::SendPeer e = sends[peer];
    nvshmemx_double_put_signal_warp(
        remoteInbox + e.remoteInboxOffset, packed + e.packOffset,
        static_cast<std::size_t>(e.count), remoteReady + e.remoteSignalIndex,
        generation, NVSHMEM_SIGNAL_SET, e.pe);
  }
}

// P6.1 variant C: one *complete CUDA block* per destination PE packet.
// Every thread in a 128-thread block participates in the same NVSHMEM call.
// The block-scoped API must NEVER be called by only threadIdx.x==0.
static __global__ void packedPutSignalBlockKernel(
    const PackedPutP51::SendPeer* sends, std::size_t nSend,
    double* remoteInbox, const double* packed, U64* remoteReady, U64 generation) {
  const std::size_t peer = static_cast<std::size_t>(blockIdx.x);
  if (peer < nSend) { // uniform branch across the entire CUDA block
    const PackedPutP51::SendPeer e = sends[peer];
    nvshmemx_double_put_signal_block(
        remoteInbox + e.remoteInboxOffset, packed + e.packOffset,
        static_cast<std::size_t>(e.count), remoteReady + e.remoteSignalIndex,
        generation, NVSHMEM_SIGNAL_SET, e.pe);
  }
}

// One remote writer per credit slot; SET needs neither CAS nor a lock.
// This kernel only publishes acknowledgments after unpackCuda() has
// finished all local inbox reads.
static __global__ void acknowledgeConsumedKernel(
    const PackedPutP51::ReceivePeer* receives, std::size_t nRecv,
    U64* remoteCredits, U64 receiverPe, U64 generation) {
  const std::size_t tid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = tid; i < nRecv; i += stride) {
    const PackedPutP51::ReceivePeer e = receives[i];
    nvshmemx_signal_op(remoteCredits + receiverPe, generation,
                       NVSHMEM_SIGNAL_SET, e.pe);
  }
}

// Optional P6 timing of the *same production algorithm* used by import().
// The uninstrumented path does not sample a clock. These are CPU wall-clock
// durations, each enclosing the stage's normal CUDA launch/fence or NVSHMEMx
// stream wait/synchronization. They are NOT isolated NIC wire latencies.
enum Phase : std::size_t {
  CreditWait = 0, Pack = 1, PutSignal = 2, ReadyWait = 3,
  Unpack = 4, Local = 5, Ack = 6, PhaseCount = 7
};
struct PhaseTimings {
  std::array<std::uint64_t, PhaseCount> nanoseconds{};
  std::uint64_t totalNanoseconds = 0;
};

struct Counters {
  U64 imports = 0;
  U64 packLaunches = 0;
  U64 sendLaunches = 0;
  U64 unpackLaunches = 0;
  U64 localLaunches = 0;
  U64 ackLaunches = 0;
  U64 putCalls = 0;         // one GPU-initiated put_signal per destination peer
  U64 putElements = 0;      // doubles, NOT transport packet count
  U64 readyWaits = 0;       // one stream wait per incoming peer per Import
  U64 creditWaits = 0;      // one stream wait per outgoing peer AFTER first Import
  U64 ackSignals = 0;       // one GPU-initiated signal_op per incoming peer
  U64 localElements = 0;
  U64 creditWaitNanos = 0; // measured local stream-wait wall time
  U64 maxCreditWaitNanos = 0;
  U64 globalSyncsInImport = 0; // invariant: zero (not an estimated cost)
};

// Persistent source/destination Maps and exact per-peer layout are reused
// across imports. GPU-only NVSHMEM operations run in the device-only kernels
// above. Remaining Kokkos CUDA lambdas use only ordinary CUDA load/stores.
//
// ready_[source PE]: receiver-owned slot written by PUT_SIGNAL on source.
// credit_[dest PE]: sender-owned slot written by receiver ACK on source.
// Both arrays use actual NVSHMEM PE indices, not MPI ranks. Each slot has
// exactly one remote writer; CAS/locks are therefore unnecessary.
//
// Single slot per peer, generation starts at 1 after calloc. Source A may
// issue PUT(g) to B only after its credit_[B] == g-1 (except g=1). Receiver B
// writes ACK(g) only AFTER its GPU unpack of A's segment is finished. Thus
// PUT(g+1) cannot overwrite B's still-unconsumed inbox for generation g.
template<class LO, class GO, class Node>
class Exchange {
 public:
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  using Plan = PackedPutP51::Plan;
  using Local = PackedPutP51::LocalEntry;
  using Send = PackedPutP51::SendPeer;
  using Receive = PackedPutP51::ReceivePeer;
  using SymmetricDoubles = Kokkos::View<
      double*, Kokkos::CudaSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  Exchange(const Plan& plan, const NvshmemP0::Session& session)
      : plan_(plan), session_(session),
        owned_(PackedPutP51::asSize(plan.ownedLength, "P5.2 owned")),
        target_(PackedPutP51::asSize(plan.targetLength, "P5.2 target")),
        packElements_(PackedPutP51::asSize(plan.packElements, "P5.2 pack")),
        inboxElements_(PackedPutP51::asSize(plan.inboxElements, "P5.2 inbox")),
        packCapacity_(PackedPutP51::asSize(plan.symmetricPackElements, "P5.2 pack capacity")),
        inboxCapacity_(PackedPutP51::asSize(plan.symmetricInboxElements, "P5.2 inbox capacity")),
        nLocal_(plan.local.size()), nSend_(plan.sendPeers.size()),
        nRecv_(plan.receivePeers.size()) {
    // Validate BEFORE the first collective allocation, so that corrupt
    // metadata cannot trigger a remote out-of-bounds PUT.
    require(plan.finalized && plan.rank == session.mpiRank() &&
            plan.pe == session.pe(), "P5.1 plan not finalized for this PE");
    require(plan.symmetricSignalElements == static_cast<U64>(session.numPes()) &&
            packCapacity_ >= packElements_ && inboxCapacity_ >= inboxElements_ &&
            packCapacity_ > 0 && inboxCapacity_ > 0 &&
            plan.packedSourceLids.size() == packElements_ &&
            plan.unpackDestinationLids.size() == inboxElements_ &&
            nLocal_ + inboxElements_ == target_, "invalid plan capacities/partition");
    for (const auto& e : plan.sendPeers) {
      require(e.count != 0 && e.pe >= 0 && e.pe < session.numPes() &&
              session.mpiRankForPe(e.pe) == e.rank &&
              e.remoteSignalIndex == static_cast<U64>(session.pe()) &&
              e.packOffset <= plan.packElements &&
              e.count <= plan.packElements - e.packOffset &&
              e.remoteInboxOffset <= plan.symmetricInboxElements &&
              e.count <= plan.symmetricInboxElements - e.remoteInboxOffset,
              "invalid send peer/remote inbox segment");
    }
    for (const auto& e : plan.receivePeers) {
      require(e.count != 0 && e.pe >= 0 && e.pe < session.numPes() &&
              session.mpiRankForPe(e.pe) == e.rank &&
              e.localSignalIndex == static_cast<U64>(e.pe) &&
              e.inboxOffset <= plan.inboxElements &&
              e.count <= plan.inboxElements - e.inboxOffset,
              "invalid receive peer/inbox segment");
    }
    for (const U64 lid : plan.packedSourceLids)
      require(lid < plan.ownedLength, "invalid source pack LID");
    for (const U64 lid : plan.unpackDestinationLids)
      require(lid < plan.targetLength, "invalid remote destination LID");
    for (const auto& e : plan.local)
      require(e.sourceLid < plan.ownedLength && e.destinationLid < plan.targetLength,
              "invalid local source/target LIDs");

    // Symmetric allocations are collective, with identical byte counts and
    // call order on every PE. Their implicit NVSHMEM barriers are OUTSIDE
    // import(), which has no world operations whatsoever.
    packed_ = static_cast<double*>(nvshmem_malloc(PackedPutP51::checkedBytes(
        plan.symmetricPackElements, sizeof(double), "P5.2 symmetric pack")));
    require(packed_ != nullptr, "collective nvshmem_malloc(pack) failed");
    inbox_ = static_cast<double*>(nvshmem_malloc(PackedPutP51::checkedBytes(
        plan.symmetricInboxElements, sizeof(double), "P5.2 symmetric inbox")));
    require(inbox_ != nullptr, "collective nvshmem_malloc(inbox) failed");
    ready_ = static_cast<U64*>(nvshmem_calloc(
        static_cast<std::size_t>(session.numPes()), sizeof(U64)));
    require(ready_ != nullptr, "collective nvshmem_calloc(ready) failed");
    credit_ = static_cast<U64*>(nvshmem_calloc(
        static_cast<std::size_t>(session.numPes()), sizeof(U64)));
    require(credit_ != nullptr, "collective nvshmem_calloc(credit) failed");

    localDevice_ = NvshmemP3::upload("P5.2 local copy descriptors", plan.local);
    packLidsDevice_ = NvshmemP3::upload("P5.2 original export source LIDs", plan.packedSourceLids);
    unpackLidsDevice_ = NvshmemP3::upload("P5.2 inbox-slot target LIDs", plan.unpackDestinationLids);
    sendDevice_ = NvshmemP3::upload("P5.2 sender peer packets", plan.sendPeers);
    recvDevice_ = NvshmemP3::upload("P5.2 receiver peer packets", plan.receivePeers);
    Kokkos::fence("P5.2 static descriptors ready");
  }

  Exchange(const Exchange&) = delete;
  Exchange& operator=(const Exchange&) = delete;

  ~Exchange() {
    // A rank-local failure is handled by the driver with MPI_Abort. Do not
    // enter an NVSHMEM collective free while stack-unwinding one PE.
    if (packed_ && std::uncaught_exceptions() == 0) release();
  }

  std::size_t packedCount() const { return packElements_; }
  std::size_t inboxCount() const { return inboxElements_; }
  std::size_t sendPeerCount() const { return nSend_; }
  std::size_t receivePeerCount() const { return nRecv_; }
  U64 generation() const { return generation_; }
  const Counters& counters() const { return counts_; }

  std::size_t symmetricBytes() const {
    const auto pack = PackedPutP51::checkedBytes(
        plan_.symmetricPackElements, sizeof(double), "pack bytes");
    const auto inbox = PackedPutP51::checkedBytes(
        plan_.symmetricInboxElements, sizeof(double), "inbox bytes");
    const auto signals = PackedPutP51::checkedBytes(
        plan_.symmetricSignalElements, sizeof(U64), "signal bytes");
    require(inbox <= std::numeric_limits<std::size_t>::max() - pack,
            "combined symmetric allocations overflow size_t");
    require(signals <= (std::numeric_limits<std::size_t>::max() - pack - inbox) / 2,
            "ready+credit allocations overflow size_t");
    return pack + inbox + 2 * signals;
  }

  // Consumer credits: only outgoing peer relationships matter. This is
  // point-to-point credit control, not a barrier or a collective launch.
  // Each producer finished all sends for g-1 before returning its Import.
  void waitPreviousCredits(U64 previous) {
    if (previous == 0 || nSend_ == 0) return;
    const cudaStream_t stream = 0;
    // Include both NVSHMEMx wait submission and local stream completion in
    // measured time. The submission itself can cost host time depending on
    // the installed NVSHMEM transport; this is not a GPU timing benchmark.
    const auto started = std::chrono::steady_clock::now();
    for (const auto& e : plan_.sendPeers)
      (void)nvshmemx_signal_wait_until_on_stream(
          credit_ + static_cast<std::size_t>(e.pe), NVSHMEM_CMP_EQ,
          previous, stream);
    cudaCheck(cudaStreamSynchronize(stream), "P5.2 wait consumed credits on stream");
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    if (elapsed > 0) {
      const U64 ns = static_cast<U64>(elapsed);
      counts_.creditWaitNanos += ns;
      counts_.maxCreditWaitNanos = std::max(counts_.maxCreditWaitNanos, ns);
    }
  }

  void packCuda(const Vector& X) const {
    require(X.getMap()->getLocalNumElements() == owned_,
            "X does not have the domain Map local length");
    if (packElements_ == 0) return;
    const auto lids = packLidsDevice_;
    const SymmetricDoubles packet(packed_, packCapacity_);
    {
      auto x = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
      Kokkos::parallel_for("P5.2 Kokkos pack into per-PE symmetric scratch",
        Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, packElements_),
        KOKKOS_LAMBDA(const std::size_t i) {
          packet(i) = x(lids(i), 0);
        });
      Kokkos::fence("P5.2 GPU pack finished before GPU PUT reads scratch");
    }
  }

  void issuePutsCuda(U64 generation) const {
    if (nSend_ == 0) return;
    constexpr unsigned int threads = 128;
    const unsigned int blocks = static_cast<unsigned int>(
        (nSend_ + threads - 1) / threads);
    // Device-only kernel: KOKKOS_LAMBDA cannot call nvshmem_double_put_signal.
    // This is NOT an NVSHMEM collective launch. Only ranks with sends launch.
    packedPutSignalKernel<<<blocks, threads, 0, 0>>>(
        sendDevice_.data(), nSend_, inbox_, packed_, ready_, generation);
    cudaCheck(cudaGetLastError(), "P5.2 launch per-peer GPU PUT+signal");
    // A local CUDA completion, not a global rendezvous. The blocking PUT
    // call has finished reading its scratch packet on return; receivers
    // independently wait for their own 'ready' signal to observe delivery.
    cudaCheck(cudaStreamSynchronize(0),
              "P5.2 all local GPU PUT+signal calls finished");
  }

  // P6.1 B: preserve issuePutsCuda (one thread/peer) as the baseline.
  // This path changes ONLY the GPU PUT issuance granularity. All other
  // synchronization, scratch layout, unpack and ACK work remains identical.
  void issuePutsWarpCuda(U64 generation) const {
    if (nSend_ == 0) return;
    constexpr unsigned int threads = 128; // 4 complete warps per CUDA block
    constexpr std::size_t warpsPerBlock = threads / 32;
    const unsigned int blocks = static_cast<unsigned int>(
        (nSend_ + warpsPerBlock - 1) / warpsPerBlock);
    packedPutSignalWarpKernel<<<blocks, threads, 0, 0>>>(
        sendDevice_.data(), nSend_, inbox_, packed_, ready_, generation);
    cudaCheck(cudaGetLastError(), "P6.1 launch per-peer warp PUT+signal");
    cudaCheck(cudaStreamSynchronize(0),
              "P6.1 all local warp PUT+signal calls finished");
  }

  // P6.1 C: one 128-thread CUDA block for each destination peer packet.
  // No more than one logical PUT or ready signal is issued per peer.
  void issuePutsThreadBlockCuda(U64 generation) const {
    if (nSend_ == 0) return;
    constexpr unsigned int threads = 128;
    const unsigned int blocks = static_cast<unsigned int>(nSend_);
    packedPutSignalBlockKernel<<<blocks, threads, 0, 0>>>(
        sendDevice_.data(), nSend_, inbox_, packed_, ready_, generation);
    cudaCheck(cudaGetLastError(), "P6.1 launch per-peer block PUT+signal");
    cudaCheck(cudaStreamSynchronize(0),
              "P6.1 all local block PUT+signal calls finished");
  }

  // NVSHMEMx host-issued on-stream wait is the documented alternative to a
  // kernel calling nvshmem_signal_wait_until, and does not require an
  // nvshmemx_collective_launch. Only actual incoming source PEs are waited.
  void waitReadyOnStream(U64 generation) const {
    if (nRecv_ == 0) return;
    const cudaStream_t stream = 0;
    for (const auto& e : plan_.receivePeers)
      (void)nvshmemx_signal_wait_until_on_stream(
          ready_ + static_cast<std::size_t>(e.pe), NVSHMEM_CMP_EQ,
          generation, stream);
    // This is *local* stream completion. It orders signal visibility before
    // the subsequent independent Kokkos inbox-scatter kernel. It never waits
    // for an unrelated rank or a non-peer.
    cudaCheck(cudaStreamSynchronize(stream), "P5.2 wait ready signals on stream");
  }

  void unpackCuda(Vector& Y) const {
    require(Y.getMap()->getLocalNumElements() == target_,
            "Y does not have the column Map local length");
    if (inboxElements_ == 0) return;
    const auto destination = unpackLidsDevice_;
    const SymmetricDoubles inbox(inbox_, inboxCapacity_);
    {
      auto y = Y.getLocalViewDevice(Tpetra::Access::ReadWrite);
      Kokkos::parallel_for("P5.2 GPU inbox scatter to exact target LIDs",
        Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, inboxElements_),
        KOKKOS_LAMBDA(const std::size_t i) {
          y(destination(i), 0) = inbox(i);
        });
      Kokkos::fence("P5.2 last receiver GPU inbox read finished");
    }
  }

  void localCuda(const Vector& X, Vector& Y) const {
    if (nLocal_ == 0) return;
    const auto entries = localDevice_;
    {
      auto x = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
      auto y = Y.getLocalViewDevice(Tpetra::Access::ReadWrite);
      Kokkos::parallel_for("P5.2 GPU local/permute domain-to-column copies",
        Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, nLocal_),
        KOKKOS_LAMBDA(const std::size_t i) {
          const Local e = entries(i);
          y(e.destinationLid, 0) = x(e.sourceLid, 0);
        });
      Kokkos::fence("P5.2 local output writes finished");
    }
  }

  // Each receiver has finished accessing its own inbox BEFORE publishing its
  // ACK to the corresponding source. Exactly one writer per sender's
  // credit_[receiver PE], so an atomic SET suffices; no CAS/spinlock.
  void acknowledgeCuda(U64 generation) const {
    if (nRecv_ == 0) return;
    constexpr unsigned int threads = 128;
    const unsigned int blocks = static_cast<unsigned int>(
        (nRecv_ + threads - 1) / threads);
    // Device-only kernel: nvshmemx_signal_op is also __device__ only.
    acknowledgeConsumedKernel<<<blocks, threads, 0, 0>>>(
        recvDevice_.data(), nRecv_, credit_,
        static_cast<U64>(session_.pe()), generation);
    cudaCheck(cudaGetLastError(), "P5.2 launch consumed-generation ACK");
    cudaCheck(cudaStreamSynchronize(0),
              "P5.2 receiver ACK operations locally issued");
  }

  // Test-only local delay: after all GPU inbox reads, before publishing
  // the consumption ACK. Never enabled by normal P5.2.
  void setStressAckDelayMs(int ms) {
    require(ms >= 0 && ms <= 1000, "invalid P5.2.1 ACK delay");
    stressAckDelayMs_ = ms;
  }

  // There are ZERO world collectives in this function. Only peer waits can
  // block, exactly as per-peer MPI requests + Waitall would. No receiver GPU
  // kernel needs to be resident when a remote PUT+signal arrives.
  void import(const Vector& X, Vector& Y, PhaseTimings* phases = nullptr) {
    require(packed_ && inbox_ && ready_ && credit_, "Import after release");
    require(generation_ < std::numeric_limits<U64>::max(),
            "generation overflow; collectively rebuild exchange");
    const U64 current = generation_ + 1;
    // All measurements are stack-only and opt-in. No GPU/host/symmetric
    // allocation is performed here. The uninstrumented direct comparison
    // follows exactly the same code, with no clock sampling.
    using Timer = std::chrono::steady_clock;
    Timer::time_point last;
    if (phases) { *phases = PhaseTimings{}; last = Timer::now(); }
    const auto record = [&](const Phase phase) {
      if (!phases) return;
      const auto now = Timer::now();
      phases->nanoseconds[static_cast<std::size_t>(phase)] =
          static_cast<U64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
              now - last).count());
      last = now;
    };
    // No ready-wait can stall another PE's *unissued* send for this epoch:
    // finish every local PUT first, then wait for incoming peer signals.
    waitPreviousCredits(generation_);
    record(CreditWait);
    packCuda(X);
    record(Pack);
    issuePutsCuda(current);
    record(PutSignal);
    waitReadyOnStream(current);
    record(ReadyWait);
    unpackCuda(Y);
    record(Unpack);
    localCuda(X, Y);
    record(NvshmemPackedPut::Local);
    // P5.2.1: exercise sender credit backpressure while the receiver
    // has already consumed the inbox but has not sent ACK(current).
    // No MPI or collective operations are introduced here.
    if (stressAckDelayMs_ > 0 && nRecv_ > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(stressAckDelayMs_));
    acknowledgeCuda(current);
    record(Ack);
    generation_ = current;

    ++counts_.imports;
    counts_.packLaunches += (packElements_ != 0);
    counts_.sendLaunches += (nSend_ != 0);
    counts_.unpackLaunches += (inboxElements_ != 0);
    counts_.localLaunches += (nLocal_ != 0);
    counts_.ackLaunches += (nRecv_ != 0);
    counts_.putCalls += static_cast<U64>(nSend_);
    counts_.putElements += static_cast<U64>(packElements_);
    counts_.readyWaits += static_cast<U64>(nRecv_);
    counts_.creditWaits += (current == 1 ? 0 : static_cast<U64>(nSend_));
    counts_.ackSignals += static_cast<U64>(nRecv_);
    counts_.localElements += static_cast<U64>(nLocal_);
    if (phases) {
      // Sum disjoint stage durations; tiny counter-bookkeeping tail excluded.
      // The caller also times the total Import with its own host clock.
      phases->totalNanoseconds = 0;
      for (const auto ns : phases->nanoseconds)
        phases->totalNanoseconds += ns;
    }
  }

  // P6.1 B and C: explicit entry points for the two new PUT granularities.
  // Existing Exchange::import() and its single-thread PUT path are unchanged.
  // The common protocol helper below duplicates the proven P5.2 ordering
  // rather than rewriting its production baseline code.
  void importWarp(const Vector& X, Vector& Y, PhaseTimings* phases = nullptr) {
    importCooperative<true>(X, Y, phases);
  }
  void importThreadBlock(const Vector& X, Vector& Y,
                         PhaseTimings* phases = nullptr) {
    importCooperative<false>(X, Y, phases);
  }

  // Collective symmetric retirement is a lifecycle event, not an Import
  // synchronization operation. Flush GPU-issued AMOs before freeing remote
  // signal targets; nvshmem_free itself has an implicit entry barrier.
  void release() {
    if (!packed_) return;
    Kokkos::fence("P5.2 no in-flight local GPU kernels before symmetric free");
    const cudaStream_t stream = 0;
    nvshmemx_quiet_on_stream(stream); // include GPU-issued ACK signal AMOs
    cudaCheck(cudaStreamSynchronize(stream), "P5.2 quiet GPU AMOs before free");
    nvshmem_free(credit_);
    nvshmem_free(ready_);
    nvshmem_free(inbox_);
    nvshmem_free(packed_);
    credit_ = nullptr;
    ready_ = nullptr;
    inbox_ = nullptr;
    packed_ = nullptr;
  }

 private:
  // The two new entry points differ ONLY in this PUT stage. No change is
  // made to the original import() implementation or its single-thread API.
  template<bool Warp>
  void importCooperative(const Vector& X, Vector& Y, PhaseTimings* phases) {
    require(packed_ && inbox_ && ready_ && credit_, "Import after release");
    require(generation_ < std::numeric_limits<U64>::max(),
            "generation overflow; collectively rebuild exchange");
    const U64 current = generation_ + 1;
    using Timer = std::chrono::steady_clock;
    Timer::time_point last;
    if (phases) { *phases = PhaseTimings{}; last = Timer::now(); }
    const auto record = [&](const Phase phase) {
      if (!phases) return;
      const auto now = Timer::now();
      phases->nanoseconds[static_cast<std::size_t>(phase)] =
          static_cast<U64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
              now - last).count());
      last = now;
    };
    waitPreviousCredits(generation_);
    record(CreditWait);
    packCuda(X);
    record(Pack);
    if constexpr (Warp) issuePutsWarpCuda(current);
    else issuePutsThreadBlockCuda(current);
    record(PutSignal);
    waitReadyOnStream(current);
    record(ReadyWait);
    unpackCuda(Y);
    record(Unpack);
    localCuda(X, Y);
    record(NvshmemPackedPut::Local);
    if (stressAckDelayMs_ > 0 && nRecv_ > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(stressAckDelayMs_));
    acknowledgeCuda(current);
    record(Ack);
    generation_ = current;

    ++counts_.imports;
    counts_.packLaunches += (packElements_ != 0);
    counts_.sendLaunches += (nSend_ != 0);
    counts_.unpackLaunches += (inboxElements_ != 0);
    counts_.localLaunches += (nLocal_ != 0);
    counts_.ackLaunches += (nRecv_ != 0);
    counts_.putCalls += static_cast<U64>(nSend_);
    counts_.putElements += static_cast<U64>(packElements_);
    counts_.readyWaits += static_cast<U64>(nRecv_);
    counts_.creditWaits += (current == 1 ? 0 : static_cast<U64>(nSend_));
    counts_.ackSignals += static_cast<U64>(nRecv_);
    counts_.localElements += static_cast<U64>(nLocal_);
    if (phases) {
      phases->totalNanoseconds = 0;
      for (const auto ns : phases->nanoseconds)
        phases->totalNanoseconds += ns;
    }
  }

  const Plan& plan_;
  const NvshmemP0::Session& session_;
  std::size_t owned_, target_, packElements_, inboxElements_;
  std::size_t packCapacity_, inboxCapacity_, nLocal_, nSend_, nRecv_;
  double* packed_ = nullptr;
  double* inbox_ = nullptr;
  U64* ready_ = nullptr;
  U64* credit_ = nullptr;
  U64 generation_ = 0;
  int stressAckDelayMs_ = 0; // 0 during all normal P5.2 Imports
  Counters counts_;
  Kokkos::View<Local*, Kokkos::CudaSpace> localDevice_;
  Kokkos::View<U64*, Kokkos::CudaSpace> packLidsDevice_;
  Kokkos::View<U64*, Kokkos::CudaSpace> unpackLidsDevice_;
  Kokkos::View<Send*, Kokkos::CudaSpace> sendDevice_;
  Kokkos::View<Receive*, Kokkos::CudaSpace> recvDevice_;
};

} // namespace NvshmemPackedPut
#endif // TPETRA_PERFORMANCE_NVSHMEM_PACKED_PUT_HPP
