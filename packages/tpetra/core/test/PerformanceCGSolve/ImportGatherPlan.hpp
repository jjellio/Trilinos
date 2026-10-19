// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P1: read-only Tpetra Import metadata, independent of NVSHMEM and packing.
// The host descriptors are also the input layout for the eventual P2/P3 paths.

#ifndef TPETRA_PERFORMANCE_IMPORT_GATHER_PLAN_HPP
#define TPETRA_PERFORMANCE_IMPORT_GATHER_PLAN_HPP

#include "Tpetra_Import.hpp"
#include "Tpetra_Map.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_OrdinalTraits.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ImportGatherP1 {

inline void check(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error("NVSHMEM P1: " + message);
}

inline void checkMpi(int error, const char* where) {
  if (error != MPI_SUCCESS) {
    throw std::runtime_error(std::string("NVSHMEM P1: MPI error in ") + where);
  }
}

// Do not let invalid signed LIDs turn into enormous unsigned device offsets.
template <class LO>
std::size_t checkedLid(LO lid, std::size_t length, const char* what) {
  static_assert(std::is_integral<LO>::value, "Tpetra local ordinal must be integral");
  static_assert(sizeof(LO) <= sizeof(std::uint64_t), "LO wider than device offset");
  check(!std::numeric_limits<LO>::is_signed || lid >= LO(0),
        std::string(what) + ": negative LID");
  const std::uint64_t value = static_cast<std::uint64_t>(lid);
  check(value < static_cast<std::uint64_t>(length),
        std::string(what) + ": LID outside Map length");
  return static_cast<std::size_t>(value);
}

struct LocalCopyEntry {
  std::uint64_t sourceLid;
  std::uint64_t destinationLid;
};

struct RemoteGatherEntry {
  int ownerRank;            // rank on the Map's MPI communicator
  int sourcePe;             // NVSHMEM PE, never implicitly MPI rank
  std::uint64_t sourceLid; // owner-domain-Map LID, offset into symmetric X
  std::uint64_t destinationLid; // local column-Map target LID
};

template <class LO, class GO, class Node>
struct ImportGatherPlan {
  std::vector<LocalCopyEntry> local;
  std::vector<RemoteGatherEntry> remote;
  std::vector<LO> exportLids; // original Tpetra pack order (NOT sorted)
  std::vector<int> exportPids;
  std::size_t sourceLength = 0;
  std::size_t targetLength = 0;
  std::size_t numSame = 0;
  std::size_t numPermute = 0;
  std::size_t numRemotePeers = 0;
  std::vector<unsigned long long> sourceLengthByRank;
};

// Called by all ranks, including ranks with no remote GIDs.  Neither this
// function nor the resulting descriptors modify the Maps or Importer.
// `peForRank` comes from the actual NVSHMEM/MPI rank permutation established
// in P0 Session; this function itself does not depend on NVSHMEM.
template <class LO, class GO, class Node>
ImportGatherPlan<LO, GO, Node> buildImportGatherPlan(
    const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domainMap,
    const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& colMap,
    const Tpetra::Import<LO, GO, Node>& importer,
    const std::vector<int>& peForRank,
    MPI_Comm mpiComm) {
  using Plan = ImportGatherPlan<LO, GO, Node>;
  Plan plan;
  int rank = -1, nRanks = 0;
  checkMpi(MPI_Comm_rank(mpiComm, &rank), "MPI_Comm_rank");
  checkMpi(MPI_Comm_size(mpiComm, &nRanks), "MPI_Comm_size");
  check(!domainMap.is_null() && !colMap.is_null(), "missing source/target Map");
  check(static_cast<int>(peForRank.size()) == nRanks, "bad rank-to-PE table length");
  {
    std::set<int> uniquePes(peForRank.begin(), peForRank.end());
    check(static_cast<int>(uniquePes.size()) == nRanks,
          "rank-to-PE table contains duplicate entries");
    for (int pe : peForRank) check(pe >= 0 && pe < nRanks, "invalid NVSHMEM PE");
  }

  // An empty global source Map has no GIDs that could have duplicate
  // owners.  Some Tpetra Directory implementations nevertheless report
  // isOneToOne() == false for an all-empty, locally replicated Map.
  // Only exempt *globally* empty Maps: a rank with zero local elements in
  // a nonempty global Map must still pass the normal one-to-one check.
  // The empty target check also rules out requests for nonexistent GIDs.
  // Map::isOneToOne and the nonempty directory lookup may be collective.
  const bool globallyEmptySource = (domainMap->getGlobalNumElements() == 0);
  if (globallyEmptySource) {
    check(colMap->getGlobalNumElements() == 0,
          "globally empty source Map has nonempty target Map");
  } else {
    check(domainMap->isOneToOne(), "source domain Map is not one-to-one");
  }
  check(importer.getSourceMap()->isSameAs(*domainMap),
        "Importer source Map differs from domain Map");
  check(importer.getTargetMap()->isSameAs(*colMap),
        "Importer target Map differs from column Map");

  plan.sourceLength = domainMap->getLocalNumElements();
  plan.targetLength = colMap->getLocalNumElements();
  check(plan.sourceLength <= static_cast<std::size_t>(std::numeric_limits<LO>::max()) &&
            plan.targetLength <= static_cast<std::size_t>(std::numeric_limits<LO>::max()),
        "local Map size exceeds local ordinal range");
  check(plan.sourceLength <= static_cast<std::size_t>(std::numeric_limits<unsigned long long>::max()),
        "source length cannot be represented by MPI length type");
  const unsigned long long mine = static_cast<unsigned long long>(plan.sourceLength);
  plan.sourceLengthByRank.resize(static_cast<std::size_t>(nRanks));
  checkMpi(MPI_Allgather(&mine, 1, MPI_UNSIGNED_LONG_LONG,
                         plan.sourceLengthByRank.data(), 1,
                         MPI_UNSIGNED_LONG_LONG, mpiComm),
           "MPI_Allgather(source lengths)");

  Teuchos::Array<GO> remoteGids;
  std::vector<std::size_t> remoteDestinations;
  plan.local.reserve(plan.targetLength);
  for (std::size_t dst = 0; dst < plan.targetLength; ++dst) {
    const GO gid = colMap->getGlobalElement(static_cast<LO>(dst));
    if (domainMap->isNodeGlobalElement(gid)) {
      const std::size_t src = checkedLid(domainMap->getLocalElement(gid),
                                         plan.sourceLength, "local source");
      plan.local.push_back(LocalCopyEntry{static_cast<std::uint64_t>(src),
                                          static_cast<std::uint64_t>(dst)});
    } else {
      remoteGids.push_back(gid);
      remoteDestinations.push_back(dst);
    }
  }

  // The three-argument lookup supplies actual *owner-local* LIDs, not just
  // owner ranks.  Always call it collectively, even if remoteGids is empty.
  Teuchos::Array<int> owners(remoteGids.size());
  Teuchos::Array<LO> ownerLids(remoteGids.size());
  const Tpetra::LookupStatus status =
      domainMap->getRemoteIndexList(remoteGids(), owners(), ownerLids());
  check(status == Tpetra::AllIDsPresent,
        "a target column GID is missing from the domain Map");

  plan.remote.reserve(remoteDestinations.size());
  std::set<int> distinctPeers;
  std::vector<unsigned long long> requestedFromRank(
      static_cast<std::size_t>(nRanks), 0ULL);
  for (std::size_t k = 0; k < remoteDestinations.size(); ++k) {
    const int owner = owners[static_cast<Tpetra::Array_size_type>(k)];
    check(owner >= 0 && owner < nRanks && owner != rank,
          "invalid/locally owned remote entry owner");
    const std::size_t src = checkedLid(
        ownerLids[static_cast<Tpetra::Array_size_type>(k)],
        static_cast<std::size_t>(plan.sourceLengthByRank[static_cast<std::size_t>(owner)]),
        "remote owner's source");
    plan.remote.push_back(RemoteGatherEntry{
        owner, peForRank[static_cast<std::size_t>(owner)],
        static_cast<std::uint64_t>(src),
        static_cast<std::uint64_t>(remoteDestinations[k])});
    distinctPeers.insert(owner);
    ++requestedFromRank[static_cast<std::size_t>(owner)];
  }
  plan.numRemotePeers = distinctPeers.size();

  // Independently partition the target LIDs according to Importer's
  // same IDs, local permutations and remote IDs.  Every destination must
  // occur exactly once, and its expected source LID must match the Map.
  plan.numSame = importer.getNumSameIDs();
  plan.numPermute = importer.getNumPermuteIDs();
  check(plan.numSame <= plan.sourceLength && plan.numSame <= plan.targetLength,
        "Importer numSameIDs outside Map bounds");
  const auto permuteFrom = importer.getPermuteFromLIDs();
  const auto permuteTo = importer.getPermuteToLIDs();
  const auto remoteLids = importer.getRemoteLIDs();
  check(static_cast<std::size_t>(permuteFrom.size()) == plan.numPermute &&
            static_cast<std::size_t>(permuteTo.size()) == plan.numPermute,
        "Importer permute list lengths disagree");
  check(static_cast<std::size_t>(remoteLids.size()) == plan.remote.size() &&
            importer.getNumRemoteIDs() == plan.remote.size(),
        "Importer remote count disagrees with Map classification");
  check(plan.numSame + plan.numPermute + plan.remote.size() == plan.targetLength,
        "Importer does not account for every target LID");

  std::vector<int> kind(plan.targetLength, -1); // 0 same, 1 permute, 2 remote
  std::vector<std::size_t> importerLocalSource(plan.targetLength, 0);
  for (std::size_t i = 0; i < plan.numSame; ++i) {
    check(domainMap->getGlobalElement(static_cast<LO>(i)) ==
              colMap->getGlobalElement(static_cast<LO>(i)),
          "Importer same-ID prefix GIDs disagree");
    kind[i] = 0;
    importerLocalSource[i] = i;
  }
  for (std::size_t i = 0; i < plan.numPermute; ++i) {
    const std::size_t dst = checkedLid(permuteTo[static_cast<Tpetra::Array_size_type>(i)],
                                        plan.targetLength, "permute-to");
    const std::size_t src = checkedLid(permuteFrom[static_cast<Tpetra::Array_size_type>(i)],
                                        plan.sourceLength, "permute-from");
    check(kind[dst] == -1, "duplicate same/permute destination LID");
    check(domainMap->getGlobalElement(static_cast<LO>(src)) ==
              colMap->getGlobalElement(static_cast<LO>(dst)),
          "Importer permuted source and target GIDs disagree");
    kind[dst] = 1;
    importerLocalSource[dst] = src;
  }
  for (std::size_t i = 0; i < static_cast<std::size_t>(remoteLids.size()); ++i) {
    const std::size_t dst = checkedLid(remoteLids[static_cast<Tpetra::Array_size_type>(i)],
                                        plan.targetLength, "Importer remote LID");
    check(kind[dst] == -1, "duplicate remote destination LID");
    kind[dst] = 2;
  }
  for (const auto& e : plan.local) {
    const std::size_t dst = static_cast<std::size_t>(e.destinationLid);
    check((kind[dst] == 0 || kind[dst] == 1) &&
              importerLocalSource[dst] == static_cast<std::size_t>(e.sourceLid),
          "Map local copy differs from Importer same/permute metadata");
  }
  for (const auto& e : plan.remote) {
    check(kind[static_cast<std::size_t>(e.destinationLid)] == 2,
          "Map remote LID set differs from Importer remote LID set");
  }
  for (std::size_t dst = 0; dst < plan.targetLength; ++dst) {
    check(kind[dst] != -1, "Importer left a target LID unclassified");
  }

  const auto expLids = importer.getExportLIDs();
  const auto expPids = importer.getExportPIDs();
  check(static_cast<std::size_t>(expLids.size()) ==
            static_cast<std::size_t>(expPids.size()) &&
            static_cast<std::size_t>(expLids.size()) == importer.getNumExportIDs(),
        "Importer export LID and PID metadata differ");
  plan.exportLids.assign(expLids.begin(), expLids.end());
  plan.exportPids.assign(expPids.begin(), expPids.end());
  std::vector<unsigned long long> exportedToRank(
      static_cast<std::size_t>(nRanks), 0ULL);
  for (std::size_t i = 0; i < plan.exportLids.size(); ++i) {
    const int destRank = plan.exportPids[i];
    check(destRank >= 0 && destRank < nRanks && destRank != rank,
          "Importer export has invalid destination MPI rank");
    (void)checkedLid(plan.exportLids[i], plan.sourceLength, "export source");
    ++exportedToRank[static_cast<std::size_t>(destRank)];
  }

  // Cross-check the Importer's outgoing message schedule against the exact
  // incoming peer requests.  This catches receive-only / export-only ranks.
  std::vector<unsigned long long> peerRequestedFromMe(
      static_cast<std::size_t>(nRanks), 0ULL);
  checkMpi(MPI_Alltoall(requestedFromRank.data(), 1, MPI_UNSIGNED_LONG_LONG,
                         peerRequestedFromMe.data(), 1, MPI_UNSIGNED_LONG_LONG,
                         mpiComm), "MPI_Alltoall(import peer counts)");
  check(exportedToRank == peerRequestedFromMe,
        "Importer export counts differ from peers' remote requests");
  return plan;
}

} // namespace ImportGatherP1

#endif // TPETRA_PERFORMANCE_IMPORT_GATHER_PLAN_HPP
