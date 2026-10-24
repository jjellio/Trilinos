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
  U64 globalSyncsInImport = 0; // invariant: zero (not an estimated cost)
};

// Persistent source/destination Maps and exact per-peer layout are reused
// across imports. All CUDA launch methods are public: nvcc's KOKKOS_LAMBDA
// extended lambdas must not be enclosed inside private member functions.
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
  void waitPreviousCredits(U64 previous) const {
    if (previous == 0 || nSend_ == 0) return;
    const cudaStream_t stream = 0;
    for (const auto& e : plan_.sendPeers)
      (void)nvshmemx_signal_wait_until_on_stream(
          credit_ + static_cast<std::size_t>(e.pe), NVSHMEM_CMP_EQ,
          previous, stream);
    cudaCheck(cudaStreamSynchronize(stream), "P5.2 wait consumed credits on stream");
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
    const auto sends = sendDevice_;
    double* const inbox = inbox_;
    const double* const packed = packed_;
    U64* const ready = ready_;
    Kokkos::parallel_for("P5.2 GPU PUT+signal per actual destination PE",
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, nSend_),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Send e = sends(i);
        nvshmem_double_put_signal(
            inbox + e.remoteInboxOffset, packed + e.packOffset,
            static_cast<std::size_t>(e.count), ready + e.remoteSignalIndex,
            generation, NVSHMEM_SIGNAL_SET, e.pe);
      });
    // The blocking put_signal API makes sender scratch safe to reuse when
    // each issuing thread returns. Fence the *local* CUDA kernel only.
    Kokkos::fence("P5.2 all local GPU PUT+signal calls finished");
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
    const auto receives = recvDevice_;
    U64* const credits = credit_;
    const U64 self = static_cast<U64>(session_.pe());
    Kokkos::parallel_for("P5.2 GPU per-source consumed generation ACK",
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, nRecv_),
      KOKKOS_LAMBDA(const std::size_t i) {
        const Receive e = receives(i);
        nvshmemx_signal_op(credits + self, generation, NVSHMEM_SIGNAL_SET, e.pe);
      });
    Kokkos::fence("P5.2 receiver ACK operations locally issued");
  }

  // There are ZERO world collectives in this function. Only peer waits can
  // block, exactly as per-peer MPI requests + Waitall would. No receiver GPU
  // kernel needs to be resident when a remote PUT+signal arrives.
  void import(const Vector& X, Vector& Y) {
    require(packed_ && inbox_ && ready_ && credit_, "Import after release");
    require(generation_ < std::numeric_limits<U64>::max(),
            "generation overflow; collectively rebuild exchange");
    const U64 current = generation_ + 1;
    // No ready-wait can stall another PE's *unissued* send for this epoch:
    // finish every local PUT first, then wait for incoming peer signals.
    waitPreviousCredits(generation_);
    packCuda(X);
    issuePutsCuda(current);
    waitReadyOnStream(current);
    unpackCuda(Y);
    localCuda(X, Y);
    acknowledgeCuda(current);
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
  const Plan& plan_;
  const NvshmemP0::Session& session_;
  std::size_t owned_, target_, packElements_, inboxElements_;
  std::size_t packCapacity_, inboxCapacity_, nLocal_, nSend_, nRecv_;
  double* packed_ = nullptr;
  double* inbox_ = nullptr;
  U64* ready_ = nullptr;
  U64* credit_ = nullptr;
  U64 generation_ = 0;
  Counters counts_;
  Kokkos::View<Local*, Kokkos::CudaSpace> localDevice_;
  Kokkos::View<U64*, Kokkos::CudaSpace> packLidsDevice_;
  Kokkos::View<U64*, Kokkos::CudaSpace> unpackLidsDevice_;
  Kokkos::View<Send*, Kokkos::CudaSpace> sendDevice_;
  Kokkos::View<Receive*, Kokkos::CudaSpace> recvDevice_;
};

} // namespace NvshmemPackedPut
#endif // TPETRA_PERFORMANCE_NVSHMEM_PACKED_PUT_HPP
