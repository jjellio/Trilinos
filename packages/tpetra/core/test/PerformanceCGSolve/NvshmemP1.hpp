// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P1 only: Tpetra Import oracle, host-built plan, Kokkos local reconstruction.
// No standalone MPI import and no NVSHMEM gets are performed in this file.

#ifndef TPETRA_PERFORMANCE_NVSHMEM_P1_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P1_HPP

#include "ImportGatherPlan.hpp"
#include "NvshmemP0.hpp"
#include "Tpetra_Access.hpp"
#include "Tpetra_Vector.hpp"
#include "Teuchos_OrdinalTraits.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace NvshmemP1 {

// Encode (owning MPI rank, owner-local LID) in a finite, exactly representable
// double without assuming uniform rank-local Map lengths or contiguous GIDs.
// This makes the imported remote reference independently checkable from the
// (owner,sourceLID) descriptors, without doing our own remote value transfer.
struct ValueEncoding {
  std::uint64_t stride = 0;
  static constexpr std::uint64_t exactLimit = (1ULL << 53) - 1ULL;

  ValueEncoding(std::size_t localLength, const NvshmemP0::Session& session) {
    const unsigned long long mine = static_cast<unsigned long long>(localLength);
    unsigned long long maximum = 0;
    ImportGatherP1::checkMpi(
        MPI_Allreduce(&mine, &maximum, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, session.comm()),
        "MPI_Allreduce(max owned elements for exact value encoding)");
    ImportGatherP1::check(maximum < exactLimit,
                          "owner-local Map too large for exact double encoding");
    stride = static_cast<std::uint64_t>(maximum) + 1;
    ImportGatherP1::check(
        static_cast<std::uint64_t>(session.numPes()) <= exactLimit / stride,
        "rank x source-LID encoding would exceed double integer precision");
  }

  double value(int mpiRank, std::uint64_t localLid) const {
    ImportGatherP1::check(mpiRank >= 0 &&
                              localLid < stride &&
                              static_cast<std::uint64_t>(mpiRank) * stride + localLid + 1 <= exactLimit,
                          "invalid (owner,LID) in double encoding");
    return static_cast<double>(static_cast<std::uint64_t>(mpiRank) * stride +
                               localLid + 1);
  }
};

template<class LO, class GO, class Node>
Teuchos::RCP<const Tpetra::Import<LO, GO, Node>>
ensureImporter(const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domainMap,
               const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& colMap,
               Teuchos::RCP<const Tpetra::Import<LO, GO, Node>> importer,
               MPI_Comm mpiComm) {
  const int localHas = importer.is_null() ? 0 : 1;
  int minHas = -1, maxHas = -1;
  ImportGatherP1::checkMpi(MPI_Allreduce(&localHas, &minHas, 1, MPI_INT, MPI_MIN, mpiComm),
                           "MPI_Allreduce(min graph importer)");
  ImportGatherP1::checkMpi(MPI_Allreduce(&localHas, &maxHas, 1, MPI_INT, MPI_MAX, mpiComm),
                           "MPI_Allreduce(max graph importer)");
  ImportGatherP1::check(minHas == maxHas,
                        "graph Importer is not uniformly present across MPI ranks");
  if (importer.is_null()) {
    // Some fill-complete graphs have no Importer (the common case is an
    // identical column/domain Map). Build an equivalent public Tpetra Import
    // from the two Maps, collectively, rather than assuming there are no
    // remote entries merely because the graph returned a null importer.
    importer = Teuchos::rcp(new Tpetra::Import<LO, GO, Node>(domainMap, colMap));
  }
  return importer;
}

template<class LO, class GO, class Node>
void runCase(const std::string& name,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domainMap,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& colMap,
             const Teuchos::RCP<const Tpetra::Import<LO, GO, Node>>& importer,
             const NvshmemP0::Session& session,
             bool requireRemote) {
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  using Entry = ImportGatherP1::LocalCopyEntry;
  const int rank = session.mpiRank();
  const std::size_t nOwned = domainMap->getLocalNumElements();
  ValueEncoding encoding(nOwned, session);

  Vector X(domainMap);
  const std::uint64_t base = static_cast<std::uint64_t>(rank) * encoding.stride;
  // Release the device view before doImport(): Tpetra may access X through
  // its host view during the import, and WrappedDualView forbids overlapping
  // live device and host views of the same MultiVector.
  {
    auto owned = X.getLocalViewDevice(Tpetra::Access::OverwriteAll);
    Kokkos::parallel_for(
        "NvshmemP1::fill_distinct_owned_values",
        Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, nOwned),
        KOKKOS_LAMBDA(const std::size_t i) {
          owned(i, 0) = static_cast<double>(base + static_cast<std::uint64_t>(i) + 1ULL);
        });
    Kokkos::fence("NVSHMEM P1 finish filling canonical Tpetra Vector");
  }

  Vector X_reference(colMap);
  X_reference.putScalar(std::numeric_limits<double>::quiet_NaN());
  X_reference.doImport(X, *importer, Tpetra::INSERT);

  std::vector<int> peForRank(static_cast<std::size_t>(session.numPes()), -1);
  for (int pe = 0; pe < session.numPes(); ++pe) {
    peForRank[static_cast<std::size_t>(session.mpiRankForPe(pe))] = pe;
  }
  const auto plan = ImportGatherP1::buildImportGatherPlan(
      domainMap, colMap, *importer, peForRank, session.comm());

  // Local reconstruction uses a Kokkos CUDA loop.  Remote destination LIDs
  // intentionally remain NaN: P1 does not issue gets or independent MPI data
  // transfers. The host plan is staged in device memory only for this check.
  Vector X_localOnly(colMap);
  X_localOnly.putScalar(std::numeric_limits<double>::quiet_NaN());
  const std::size_t nCopy = plan.local.size();
  Kokkos::View<Entry*, Kokkos::HostSpace> localHost("P1 local descriptors host", nCopy);
  for (std::size_t i = 0; i < nCopy; ++i) localHost(i) = plan.local[i];
  Kokkos::View<Entry*, Kokkos::CudaSpace> localDevice("P1 local descriptors device", nCopy);
  Kokkos::deep_copy(localDevice, localHost);
  // Reacquire a read-only device view of X for the local copy. Both views
  // must leave scope before the host-side correctness checks below.
  {
    auto owned = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
    auto localResult = X_localOnly.getLocalViewDevice(Tpetra::Access::ReadWrite);
    Kokkos::parallel_for(
        "NvshmemP1::local_copy",
        Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, nCopy),
        KOKKOS_LAMBDA(const std::size_t i) {
          const Entry e = localDevice(i);
          localResult(e.destinationLid, 0) = owned(e.sourceLid, 0);
        });
    Kokkos::fence("NVSHMEM P1 finish Kokkos local reconstruction");
  }

  // Checking each reference value against rank/LID encoding validates remote
  // descriptor contents (not just counts) without manually exchanging values.
  const auto refHost = X_reference.getLocalViewHost(Tpetra::Access::ReadOnly);
  const auto scratchHost = X_localOnly.getLocalViewHost(Tpetra::Access::ReadOnly);
  const auto ownedHost = X.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (const auto& e : plan.local) {
    const std::size_t dst = static_cast<std::size_t>(e.destinationLid);
    const std::size_t src = static_cast<std::size_t>(e.sourceLid);
    const double expected = encoding.value(rank, e.sourceLid);
    ImportGatherP1::check(ownedHost(src, 0) == expected &&
                              refHost(dst, 0) == expected &&
                              scratchHost(dst, 0) == expected,
                          name + ": local reference/reconstruction mismatch at target LID " +
                              std::to_string(dst));
  }
  for (const auto& e : plan.remote) {
    const std::size_t dst = static_cast<std::size_t>(e.destinationLid);
    const double expected = encoding.value(e.ownerRank, e.sourceLid);
    ImportGatherP1::check(refHost(dst, 0) == expected,
                          name + ": remote owner/source-LID disagrees with Tpetra doImport at target LID " +
                              std::to_string(dst));
    ImportGatherP1::check(std::isnan(scratchHost(dst, 0)),
                          name + ": local-only copy wrote a remote slot");
  }

  const unsigned long long remoteCount =
      static_cast<unsigned long long>(plan.remote.size());
  const unsigned long long exportCount =
      static_cast<unsigned long long>(plan.exportLids.size());
  unsigned long long totals[2] = {0ULL, 0ULL};
  const unsigned long long localCounts[2] = {remoteCount, exportCount};
  ImportGatherP1::checkMpi(MPI_Allreduce(localCounts, totals, 2,
                                          MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                                          session.comm()),
                            "MPI_Allreduce(P1 counts)");
  ImportGatherP1::check(totals[0] == totals[1],
                        name + ": total remote requests do not equal total exports");
  ImportGatherP1::check(!requireRemote || totals[0] > 0,
                        name + ": remote communication required but no remote entries exist");

  std::cout << "[NVSHMEM P1] case=" << name
            << " rank=" << rank
            << " owned=" << plan.sourceLength
            << " target=" << plan.targetLength
            << " same=" << plan.numSame
            << " permute=" << plan.numPermute
            << " remote=" << plan.remote.size()
            << " exports=" << plan.exportLids.size()
            << " sourcePEs=" << plan.numRemotePeers
            << " remoteBytes=" << (remoteCount * sizeof(double))
            << " status=PASS\n";
  if (rank == 0) {
    std::cout << "[NVSHMEM P1] case=" << name
              << " globalRemote=" << totals[0]
              << " globalExports=" << totals[1]
              << " globalRemoteBytes=" << (totals[0] * sizeof(double))
              << " status=PASS\n";
  }
}

// Synthetic Maps exercise permutations and noncontiguous/uneven ownership,
// and isolate zero-remote and export-only ranks.  They are separate from A:
// the production matrix and its domain/column Maps are never changed.
enum class SyntheticKind { Mixed, OneWay, LocalOnly };

template<class GO>
GO syntheticGid(int rank, int lid) {
  return static_cast<GO>(7 + static_cast<long long>(rank) * 1024 +
                          static_cast<long long>(lid) * 11);
}

template<class LO, class GO, class Node>
void runSynthetic(SyntheticKind kind, const NvshmemP0::Session& session) {
  using Map = Tpetra::Map<LO, GO, Node>;
  using Import = Tpetra::Import<LO, GO, Node>;
  const int rank = session.mpiRank();
  const int size = session.numPes();
  ImportGatherP1::check(
      static_cast<long long>(size) * 1024 + 100 <
          static_cast<long long>(std::numeric_limits<GO>::max()),
      "too many ranks for synthetic noncontiguous GIDs");
  const int nOwned = 3 + rank % 3;  // uneven ownership across MPI ranks
  Teuchos::Array<GO> ownGids;
  for (int i = 0; i < nOwned; ++i) ownGids.push_back(syntheticGid<GO>(rank, i));

  // Keep one same ID, interleave off-rank data, then reverse the remaining
  // owned GIDs. The off-rank target is NOT necessarily the last target LID.
  Teuchos::Array<GO> targetGids;
  targetGids.push_back(ownGids[0]);
  if (kind == SyntheticKind::Mixed && size > 1) {
    targetGids.push_back(syntheticGid<GO>((rank + 1) % size, 1));
  } else if (kind == SyntheticKind::OneWay && size > 1 && rank == 0) {
    targetGids.push_back(syntheticGid<GO>(1, 1));
  }
  for (int i = nOwned - 1; i >= 1; --i) targetGids.push_back(ownGids[i]);

  const auto comm = Tpetra::getDefaultComm();
  const auto invalidGlobal = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  const auto domain = Teuchos::rcp(new Map(invalidGlobal, ownGids(),
                                           static_cast<GO>(7), comm));
  const auto target = Teuchos::rcp(new Map(invalidGlobal, targetGids(),
                                           static_cast<GO>(7), comm));
  Teuchos::RCP<const Map> domainConst = domain;
  Teuchos::RCP<const Map> targetConst = target;
  Teuchos::RCP<const Import> importer =
      Teuchos::rcp(new Import(domainConst, targetConst));
  if (kind == SyntheticKind::Mixed) {
    runCase("synthetic-mixed", domainConst, targetConst, importer,
            session, size > 1);
  } else if (kind == SyntheticKind::OneWay) {
    runCase("synthetic-one-way", domainConst, targetConst, importer,
            session, size > 1);
  } else {
    runCase("synthetic-local-only", domainConst, targetConst, importer,
            session, false);
  }
}

} // namespace NvshmemP1

#endif // TPETRA_PERFORMANCE_NVSHMEM_P1_HPP
