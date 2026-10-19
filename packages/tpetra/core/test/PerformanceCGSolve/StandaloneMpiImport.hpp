// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P2: one-column Kokkos CUDA reproduction of Tpetra MultiVector's
// non-overlapped Import data path.  The existing Importer/Distributor owns
// all interprocess packet ordering; we do not invent MPI message schedules.
//
// All communication payload buffers are ordinary Kokkos::CudaSpace Views,
// NEVER NVSHMEM symmetric allocations.  The Distributor chooses the MPI
// transport according to the configured Tpetra/MPI implementation.

#ifndef TPETRA_PERFORMANCE_STANDALONE_MPI_IMPORT_HPP
#define TPETRA_PERFORMANCE_STANDALONE_MPI_IMPORT_HPP

#include "Tpetra_Access.hpp"
#include "Tpetra_Distributor.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Vector.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace StandaloneMpiP2 {

inline void check(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error("NVSHMEM P2: " + why);
}

struct Timings {
  double local = 0.0;     // copy same IDs and local permutations, CUDA complete
  double pack = 0.0;      // pack exports in Tpetra's exact export-LID order
  double exchange = 0.0;  // Distributor forward doPostsAndWaits, CUDA complete
  double unpack = 0.0;    // place received values at Tpetra remote LIDs
  double total = 0.0;     // complete column-map destination
};

// Copy the Importer's existing LID list to normal CudaSpace, once before
// measurements. There are no translated or newly sorted IDs.
template <class LO>
Kokkos::View<LO*, Kokkos::CudaSpace>
uploadLids(const std::string& name, const Teuchos::ArrayView<const LO>& lids) {
  const std::size_t n = static_cast<std::size_t>(lids.size());
  Kokkos::View<LO*, Kokkos::CudaSpace> device(name, n);
  if (n != 0) {
    Kokkos::View<LO*, Kokkos::HostSpace> host(name + " host", n);
    for (std::size_t i = 0; i < n; ++i) host(i) = lids[i];
    Kokkos::deep_copy(device, host);
  }
  return device;
}

template<class LO, class GO, class Node>
class ImportReplica {
 public:
  using Map = Tpetra::Map<LO, GO, Node>;
  using Import = Tpetra::Import<LO, GO, Node>;
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  using DeviceDoubles = Kokkos::View<double*, Kokkos::CudaSpace>;
  using DeviceLids = Kokkos::View<LO*, Kokkos::CudaSpace>;

  ImportReplica(const Teuchos::RCP<const Map>& sourceMap,
                const Teuchos::RCP<const Map>& targetMap,
                const Import& importer)
      : sourceMap_(sourceMap), targetMap_(targetMap), importer_(importer) {
    check(!sourceMap_.is_null() && !targetMap_.is_null(), "null Map");
    check(importer_.getSourceMap()->isSameAs(*sourceMap_) &&
          importer_.getTargetMap()->isSameAs(*targetMap_),
          "Importer Maps do not match the vectors");

    numSame_ = importer_.getNumSameIDs();
    numPermute_ = importer_.getNumPermuteIDs();
    numExports_ = importer_.getNumExportIDs();
    numRemote_ = importer_.getNumRemoteIDs();
    const auto from = importer_.getPermuteFromLIDs();
    const auto to = importer_.getPermuteToLIDs();
    const auto exportLids = importer_.getExportLIDs();
    const auto remoteLids = importer_.getRemoteLIDs();

    check(static_cast<std::size_t>(from.size()) == numPermute_ &&
          static_cast<std::size_t>(to.size()) == numPermute_ &&
          static_cast<std::size_t>(exportLids.size()) == numExports_ &&
          static_cast<std::size_t>(remoteLids.size()) == numRemote_,
          "Importer LID list and count mismatch");
    check(numSame_ <= sourceMap_->getLocalNumElements() &&
          numSame_ <= targetMap_->getLocalNumElements() &&
          numSame_ + numPermute_ + numRemote_ == targetMap_->getLocalNumElements(),
          "Importer does not cover the target Map");
    check(importer_.getDistributor().getTotalReceiveLength() == numRemote_,
          "Distributor receive packet count differs from remoteLIDs");

    exportLids_ = uploadLids("P2 Tpetra export LIDs", exportLids);
    remoteLids_ = uploadLids("P2 Tpetra remote LIDs", remoteLids);
    permuteFrom_ = uploadLids("P2 Tpetra permute-from LIDs", from);
    permuteTo_ = uploadLids("P2 Tpetra permute-to LIDs", to);
    exports_ = DeviceDoubles("P2 CudaSpace MPI exports", numExports_);
    imports_ = DeviceDoubles("P2 CudaSpace MPI imports", numRemote_);
    Kokkos::fence("P2 Importer metadata staged to CUDA");

  }

  std::size_t exported() const { return numExports_; }
  std::size_t remote() const { return numRemote_; }
  std::size_t numSame() const { return numSame_; }
  std::size_t numPermute() const { return numPermute_; }

  // Public by design. NVCC requires member functions enclosing extended
  // KOKKOS_LAMBDA (__host__ __device__) to be publicly accessible. This
  // explicitly preserves the uploaded fix-p2-private-cuda-lambda patch.
  void copyLocalCuda(const Vector& X, Vector& Y) const {
    const std::size_t same = numSame_;
    const std::size_t permute = numPermute_;
    if (same == 0 && permute == 0) return;
    {
      auto source = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
      auto target = Y.getLocalViewDevice(Tpetra::Access::OverwriteAll);
      const auto from = permuteFrom_;
      const auto to = permuteTo_;
      Kokkos::parallel_for(
          "P2 copy Tpetra same IDs",
          Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, same),
          KOKKOS_LAMBDA(const std::size_t i) {
            target(i, 0) = source(i, 0);
          });
      Kokkos::parallel_for(
          "P2 copy Tpetra local permutations",
          Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, permute),
          KOKKOS_LAMBDA(const std::size_t i) {
            target(to(i), 0) = source(from(i), 0);
          });
      Kokkos::fence("P2 same and permute destination entries ready");
    }
  }

  void packCuda(const Vector& X) const {
    if (numExports_ == 0) return;
    {
      auto source = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
      const auto lids = exportLids_;
      const auto buffer = exports_;
      Kokkos::parallel_for(
          "P2 pack exactly in Importer exportLIDs order",
          Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, numExports_),
          KOKKOS_LAMBDA(const std::size_t i) {
            buffer(i) = source(lids(i), 0);
          });
      Kokkos::fence("P2 exported CudaSpace values ready for Distributor");
    }
  }

  void placeCuda(Vector& Y) const {
    if (numRemote_ == 0) return;
    {
      auto target = Y.getLocalViewDevice(Tpetra::Access::ReadWrite);
      const auto lids = remoteLids_;
      const auto received = imports_;
      Kokkos::parallel_for(
          "P2 unpack exactly in Importer remoteLIDs order",
          Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, numRemote_),
          KOKKOS_LAMBDA(const std::size_t i) {
            target(lids(i), 0) = received(i);
          });
      Kokkos::fence("P2 received values written to column Map destination");
    }
  }

  // The authoritative Distributor plan owns the send/receive message
  // ordering, including any internal indicesTo permutation or self message.
  // We never reorder the export packet list or construct another plan.
  void exchangeCuda() const {
    // Call on *every* rank, including ranks with no local sends/receives.
    // This is also safe for Distributor plans using collective algorithms;
    // ranks must not independently decide to skip such a call.
    importer_.getDistributor().doPostsAndWaits(exports_, std::size_t(1), imports_);
    Kokkos::fence("P2 Distributor completed CudaSpace communication");
  }

  // Match the standard, non-overlapped DistObject::beginTransfer/endTransfer
  // order: local copy, pack, transfer, and unpack. No epochs, control flags,
  // additional GID messages, or host-staging MPI buffers are introduced.
  Timings import(const Vector& X, Vector& Y) const {
    Timings t;
    const double begin = MPI_Wtime();
    copyLocalCuda(X, Y);
    const double copied = MPI_Wtime();
    packCuda(X);
    const double packed = MPI_Wtime();
    exchangeCuda();
    const double received = MPI_Wtime();
    placeCuda(Y);
    const double finished = MPI_Wtime();
    t.local = copied - begin;
    t.pack = packed - copied;
    t.exchange = received - packed;
    t.unpack = finished - received;
    t.total = finished - begin;
    return t;
  }

 private:
  Teuchos::RCP<const Map> sourceMap_, targetMap_;
  const Import& importer_;
  std::size_t numSame_ = 0, numPermute_ = 0, numExports_ = 0, numRemote_ = 0;
  DeviceLids exportLids_, remoteLids_, permuteFrom_, permuteTo_;
  DeviceDoubles exports_, imports_;
};

} // namespace StandaloneMpiP2

#endif // TPETRA_PERFORMANCE_STANDALONE_MPI_IMPORT_HPP

