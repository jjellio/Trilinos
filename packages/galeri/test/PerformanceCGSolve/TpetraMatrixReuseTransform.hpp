#pragma once

#include <Kokkos_Core.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_OrdinalTraits.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Distributor.hpp>
#include <Tpetra_Map.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace TpetraMatrixTools {

// Semantics:
//
//   maxReuse = N
//
// means that a synthetic X scalar may be referenced at most N+1 times.
// Thus maxReuse == 0 means no reuse: every matrix entry references a distinct
// synthetic X scalar (apart from unused domain entries, which are retained as
// unused columns so the transformed domain never becomes smaller merely due to
// structurally empty original columns).
//
// A negative maxReuse disables the transformation and returns the input RCP.
struct ReuseTransformStats {
  int requestedReuse = -1;
  long long intrinsicMaxReuse = -1;

  Tpetra::global_size_t originalDomainSize = 0;
  Tpetra::global_size_t transformedDomainSize = 0;
  Tpetra::global_size_t globalNnz = 0;
  Tpetra::global_size_t unreferencedOriginalDomainEntries = 0;

  bool transformed = false;
};

namespace detail {

inline long long ceilDivPositive(const long long n, const long long d) {
  if (n < 0 || d <= 0) {
    throw std::invalid_argument(
        "TpetraMatrixTools::ceilDivPositive expects n >= 0 and d > 0");
  }
  return n == 0 ? 0 : 1 + (n - 1) / d;
}

template <class Packet>
std::vector<Packet> distributorForward(Tpetra::Distributor& distributor,
                                       const std::vector<Packet>& exports,
                                       const std::size_t numImports) {
  static_assert(std::is_trivially_copyable<Packet>::value,
                "Distributor packet must be trivially copyable");

  std::vector<Packet> imports(numImports);

  Kokkos::View<const Packet*, Kokkos::HostSpace> exportView(
      exports.data(), exports.size());
  Kokkos::View<Packet*, Kokkos::HostSpace> importView(imports.data(),
                                                       imports.size());

  distributor.doPostsAndWaits(exportView, 1, importView);
  return imports;
}

template <class GO>
GO checkedLongLongToGO(const long long value, const char* what) {
  static_assert(std::is_integral<GO>::value,
                "Tpetra GlobalOrdinal must be integral");

  if (value < 0) {
    throw std::overflow_error(std::string(what) + " is negative");
  }

  using limits = std::numeric_limits<GO>;
  if constexpr (limits::is_signed) {
    if (value > static_cast<long long>(limits::max())) {
      throw std::overflow_error(std::string(what) +
                                " exceeds GlobalOrdinal range");
    }
  } else {
    using unsigned_ll = unsigned long long;
    if (static_cast<unsigned_ll>(value) >
        static_cast<unsigned_ll>(limits::max())) {
      throw std::overflow_error(std::string(what) +
                                " exceeds GlobalOrdinal range");
    }
  }

  return static_cast<GO>(value);
}

}  // namespace detail

// Collective reuse-limiting graph transform.
//
// Input requirements:
//   * A is fill-complete.
//   * A's domain Map is one-to-one.  It need not be contiguous.
//
// Output invariants when a transformation is performed:
//   * same row Map
//   * same range Map
//   * same number of rows
//   * same global/local number of matrix entries
//   * same values and same row lengths
//   * each original X owner rank continues to own all synthetic copies of
//     that original X value
//   * every synthetic X value is referenced at most maxReuse+1 times globally
//
// Consequently, the rank-to-rank reference-count matrix R(i,j) is preserved,
// while the unique-X matrix U(i,j) increases as reuse is removed.
//
// The returned matrix generally has a wider domain Map.  For maxReuse == 0,
// every referenced matrix entry gets a distinct synthetic X scalar globally.
// If every original domain entry was referenced at least once, the transformed
// global domain size is then exactly the global NNZ count.
//
// A negative maxReuse means "disabled" and simply returns A.
template <class CrsMatrix>
Teuchos::RCP<CrsMatrix> limitReuse(
    const Teuchos::RCP<CrsMatrix>& A,
    const int maxReuse,
    ReuseTransformStats* stats = nullptr) {
  using Teuchos::Array;
  using Teuchos::RCP;
  using Teuchos::rcp;

  using Scalar = typename CrsMatrix::scalar_type;
  using LO = typename CrsMatrix::local_ordinal_type;
  using GO = typename CrsMatrix::global_ordinal_type;
  using map_type = typename CrsMatrix::map_type;

  if (A.is_null()) {
    throw std::invalid_argument(
        "TpetraMatrixTools::limitReuse received a null matrix");
  }

  if (!A->isFillComplete()) {
    throw std::runtime_error(
        "TpetraMatrixTools::limitReuse requires a fill-complete matrix");
  }

  const auto rowMap = A->getRowMap();
  const auto rangeMap = A->getRangeMap();
  const auto domainMap = A->getDomainMap();
  const auto colMap = A->getColMap();

  if (rowMap.is_null() || rangeMap.is_null() || domainMap.is_null() ||
      colMap.is_null()) {
    throw std::runtime_error(
        "TpetraMatrixTools::limitReuse requires row/range/domain/column maps");
  }

  ReuseTransformStats localStats;
  localStats.requestedReuse = maxReuse;
  localStats.originalDomainSize = domainMap->getGlobalNumElements();
  localStats.transformedDomainSize = localStats.originalDomainSize;
  localStats.globalNnz = A->getGlobalNumEntries();

  if (maxReuse < 0) {
    if (stats != nullptr) *stats = localStats;
    return A;
  }

  const auto comm = domainMap->getComm();
  const int rank = comm->getRank();
  const int numRanks = comm->getSize();

  if (!domainMap->isOneToOne()) {
    throw std::runtime_error(
        "TpetraMatrixTools::limitReuse requires a one-to-one domain Map");
  }

  const long long referencesPerSynthetic =
      static_cast<long long>(maxReuse) + 1LL;
  if (referencesPerSynthetic <= 0) {
    throw std::overflow_error(
        "TpetraMatrixTools::limitReuse reuse value overflowed");
  }

  // -----------------------------------------------------------------------
  // 1. Count how often this rank references each local column-map entry.
  // -----------------------------------------------------------------------
  const auto localA = A->getLocalMatrixHost();
  const std::size_t numLocalCols = colMap->getLocalNumElements();
  const std::size_t localNnz = static_cast<std::size_t>(localA.nnz());

  std::vector<long long> localUseCount(numLocalCols, 0);
  for (std::size_t k = 0; k < localNnz; ++k) {
    const LO colLid = localA.graph.entries(k);
    if (colLid < static_cast<LO>(0) ||
        static_cast<std::size_t>(colLid) >= numLocalCols) {
      throw std::runtime_error(
          "TpetraMatrixTools::limitReuse found a CRS column LID outside "
          "the column Map");
    }
    ++localUseCount[static_cast<std::size_t>(colLid)];
  }

  // Resolve each column-map GID to its original domain-map owner.  This is the
  // ownership that the synthetic copies will preserve.
  const auto colGidList = colMap->getLocalElementList();
  Array<GO> colGids(numLocalCols);
  Array<int> colOwners(numLocalCols);
  std::fill(colOwners.begin(), colOwners.end(), -1);

  for (std::size_t lid = 0; lid < numLocalCols; ++lid) {
    colGids[lid] = colGidList[lid];
  }

  const Tpetra::LookupStatus lookup =
      domainMap->getRemoteIndexList(colGids(), colOwners());
  if (lookup != Tpetra::AllIDsPresent) {
    throw std::runtime_error(
        "TpetraMatrixTools::limitReuse: column Map contains a GID not "
        "present in the domain Map");
  }

  // One request per (source rank, original X GID), carrying this rank's total
  // number of references to that GID.  This is typically column-map sized,
  // not NNZ sized.
  Array<int> exportProcIDs;
  std::vector<GO> exportGids;
  std::vector<long long> exportCounts;
  std::vector<int> exportSources;
  std::vector<long long> exportRequestIds;

  exportProcIDs.reserve(numLocalCols);
  exportGids.reserve(numLocalCols);
  exportCounts.reserve(numLocalCols);
  exportSources.reserve(numLocalCols);
  exportRequestIds.reserve(numLocalCols);

  const std::size_t noRequest = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> requestIndexByColLid(numLocalCols, noRequest);

  for (std::size_t lid = 0; lid < numLocalCols; ++lid) {
    if (localUseCount[lid] == 0) continue;

    const int owner = colOwners[lid];
    if (owner < 0 || owner >= numRanks) {
      throw std::runtime_error(
          "TpetraMatrixTools::limitReuse resolved an invalid X owner rank");
    }

    const std::size_t requestIndex = exportGids.size();
    requestIndexByColLid[lid] = requestIndex;

    if (requestIndex >
        static_cast<std::size_t>(std::numeric_limits<long long>::max())) {
      throw std::overflow_error(
          "TpetraMatrixTools::limitReuse request index exceeds long long");
    }

    exportProcIDs.push_back(owner);
    exportGids.push_back(colGids[lid]);
    exportCounts.push_back(localUseCount[lid]);
    exportSources.push_back(rank);
    exportRequestIds.push_back(static_cast<long long>(requestIndex));
  }

  // -----------------------------------------------------------------------
  // 2. Send reference counts to the original X owners.
  // -----------------------------------------------------------------------
  Tpetra::Distributor distributor(comm);
  const std::size_t numImports = distributor.createFromSends(exportProcIDs());

  const std::vector<GO> importGids =
      detail::distributorForward(distributor, exportGids, numImports);
  const std::vector<long long> importCounts =
      detail::distributorForward(distributor, exportCounts, numImports);
  const std::vector<int> importSources =
      detail::distributorForward(distributor, exportSources, numImports);
  const std::vector<long long> importRequestIds =
      detail::distributorForward(distributor, exportRequestIds, numImports);

  // Every imported GID must be locally owned in the original domain Map.
  const std::size_t localDomainSize = domainMap->getLocalNumElements();
  std::vector<long long> totalReferencesByDomainLid(localDomainSize, 0);
  std::vector<std::size_t> requestDomainLid(numImports, 0);

  for (std::size_t i = 0; i < numImports; ++i) {
    const GO gid = importGids[i];
    if (!domainMap->isNodeGlobalElement(gid)) {
      throw std::runtime_error(
          "TpetraMatrixTools::limitReuse owner received a GID it does not own");
    }

    const LO lid = domainMap->getLocalElement(gid);
    if (lid < static_cast<LO>(0) ||
        static_cast<std::size_t>(lid) >= localDomainSize) {
      throw std::runtime_error(
          "TpetraMatrixTools::limitReuse got invalid owner-local domain LID");
    }

    const std::size_t domainLid = static_cast<std::size_t>(lid);
    requestDomainLid[i] = domainLid;

    if (importCounts[i] <= 0) {
      throw std::runtime_error(
          "TpetraMatrixTools::limitReuse received a nonpositive reference count");
    }

    if (totalReferencesByDomainLid[domainLid] >
        std::numeric_limits<long long>::max() - importCounts[i]) {
      throw std::overflow_error(
          "TpetraMatrixTools::limitReuse reference count overflow");
    }
    totalReferencesByDomainLid[domainLid] += importCounts[i];
  }

  // -----------------------------------------------------------------------
  // 3. On each owner, determine intrinsic reuse and the minimal number of
  //    synthetic clones needed to enforce the requested cap.
  //
  //    Unreferenced original domain entries retain one unused synthetic column
  //    so the transform does not silently delete structurally empty columns.
  // -----------------------------------------------------------------------
  long long localIntrinsicMaxReuse = 0;
  long long localUnreferenced = 0;
  long long localSyntheticCount = 0;

  std::vector<long long> cloneBaseLocal(localDomainSize, 0);

  for (std::size_t lid = 0; lid < localDomainSize; ++lid) {
    const long long refs = totalReferencesByDomainLid[lid];
    if (refs == 0) {
      ++localUnreferenced;
    } else {
      localIntrinsicMaxReuse = std::max(localIntrinsicMaxReuse, refs - 1);
    }

    const long long clones =
        refs == 0 ? 1 : detail::ceilDivPositive(refs, referencesPerSynthetic);

    cloneBaseLocal[lid] = localSyntheticCount;

    if (localSyntheticCount >
        std::numeric_limits<long long>::max() - clones) {
      throw std::overflow_error(
          "TpetraMatrixTools::limitReuse synthetic domain size overflow");
    }
    localSyntheticCount += clones;
  }

  long long globalIntrinsicMaxReuse = 0;
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, localIntrinsicMaxReuse,
                     Teuchos::outArg(globalIntrinsicMaxReuse));

  long long globalUnreferenced = 0;
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, localUnreferenced,
                     Teuchos::outArg(globalUnreferenced));

  localStats.intrinsicMaxReuse = globalIntrinsicMaxReuse;
  localStats.unreferencedOriginalDomainEntries =
      static_cast<Tpetra::global_size_t>(globalUnreferenced);

  // If the original graph already satisfies the requested cap, preserve it
  // exactly rather than gratuitously renumbering its domain.
  if (static_cast<long long>(maxReuse) >= globalIntrinsicMaxReuse) {
    if (stats != nullptr) *stats = localStats;
    return A;
  }

  // -----------------------------------------------------------------------
  // 4. Assign a deterministic global occurrence prefix for every
  //    (source rank, original GID) request.
  //
  //    Sorting by owner-local original domain LID, then source rank, makes the
  //    partition into reuse groups deterministic.  The owner sends each source
  //    two numbers back:
  //      * global occurrence prefix within the original X value
  //      * owner-local base LID of that original value's synthetic clone range
  // -----------------------------------------------------------------------
  std::vector<std::size_t> requestOrder(numImports);
  std::iota(requestOrder.begin(), requestOrder.end(), std::size_t{0});

  std::sort(requestOrder.begin(), requestOrder.end(),
            [&](const std::size_t a, const std::size_t b) {
              if (requestDomainLid[a] != requestDomainLid[b])
                return requestDomainLid[a] < requestDomainLid[b];
              if (importSources[a] != importSources[b])
                return importSources[a] < importSources[b];
              return a < b;
            });

  std::vector<long long> responseOccurrencePrefix(numImports, 0);
  std::vector<long long> responseCloneBaseLocal(numImports, 0);

  std::size_t currentLid = std::numeric_limits<std::size_t>::max();
  long long runningOccurrence = 0;

  for (const std::size_t request : requestOrder) {
    const std::size_t lid = requestDomainLid[request];
    if (lid != currentLid) {
      currentLid = lid;
      runningOccurrence = 0;
    }

    responseOccurrencePrefix[request] = runningOccurrence;
    responseCloneBaseLocal[request] = cloneBaseLocal[lid];

    if (runningOccurrence >
        std::numeric_limits<long long>::max() - importCounts[request]) {
      throw std::overflow_error(
          "TpetraMatrixTools::limitReuse occurrence prefix overflow");
    }
    runningOccurrence += importCounts[request];
  }

  // -----------------------------------------------------------------------
  // 5. Build a nonuniform contiguous synthetic domain Map.  Each rank owns all
  //    clones of the original X values that it owned, so source->owner traffic
  //    counts are preserved exactly.
  // -----------------------------------------------------------------------
  std::vector<long long> localSyntheticByRank(
      static_cast<std::size_t>(numRanks), 0);
  std::vector<long long> globalSyntheticByRank(
      static_cast<std::size_t>(numRanks), 0);
  localSyntheticByRank[static_cast<std::size_t>(rank)] = localSyntheticCount;

  Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, numRanks,
                     localSyntheticByRank.data(),
                     globalSyntheticByRank.data());

  std::vector<long long> ownerGlobalBase(static_cast<std::size_t>(numRanks),
                                         0);
  long long globalSyntheticCount = 0;
  for (int p = 0; p < numRanks; ++p) {
    ownerGlobalBase[static_cast<std::size_t>(p)] = globalSyntheticCount;
    const long long n = globalSyntheticByRank[static_cast<std::size_t>(p)];
    if (n < 0 || globalSyntheticCount >
                     std::numeric_limits<long long>::max() - n) {
      throw std::overflow_error(
          "TpetraMatrixTools::limitReuse global synthetic domain overflow");
    }
    globalSyntheticCount += n;
  }

  if (globalSyntheticCount < 0) {
    throw std::overflow_error(
        "TpetraMatrixTools::limitReuse negative synthetic domain size");
  }

  // Ensure the synthetic GIDs fit in GO.
  if (globalSyntheticCount > 0) {
    (void)detail::checkedLongLongToGO<GO>(globalSyntheticCount - 1,
                                          "synthetic domain GID");
  }

  if (static_cast<unsigned long long>(globalSyntheticCount) >
      static_cast<unsigned long long>(
          std::numeric_limits<Tpetra::global_size_t>::max())) {
    throw std::overflow_error(
        "TpetraMatrixTools::limitReuse synthetic domain exceeds global_size_t");
  }
  if (static_cast<unsigned long long>(localSyntheticCount) >
      static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(
        "TpetraMatrixTools::limitReuse local synthetic domain exceeds size_t");
  }

  const Tpetra::global_size_t globalSyntheticCountTpetra =
      static_cast<Tpetra::global_size_t>(globalSyntheticCount);
  const std::size_t localSyntheticCountSizeT =
      static_cast<std::size_t>(localSyntheticCount);

  RCP<const map_type> newDomainMap = rcp(new map_type(
      globalSyntheticCountTpetra, localSyntheticCountSizeT,
      static_cast<GO>(0), comm));

  localStats.transformedDomainSize = globalSyntheticCountTpetra;

  // -----------------------------------------------------------------------
  // 6. Return owner assignments to the source ranks.
  //
  // Do not rely on Distributor reverse-order semantics here.  Each request
  // carries an explicit source-local request ID; the owner sends that ID back
  // with the assignment, and the source places the response by ID.
  // -----------------------------------------------------------------------
  Array<int> responseProcIDs;
  std::vector<long long> responseRequestIds;
  responseProcIDs.reserve(numImports);
  responseRequestIds.reserve(numImports);

  for (std::size_t i = 0; i < numImports; ++i) {
    const int source = importSources[i];
    if (source < 0 || source >= numRanks) {
      throw std::runtime_error(
          "TpetraMatrixTools::limitReuse received an invalid source rank");
    }
    responseProcIDs.push_back(source);
    responseRequestIds.push_back(importRequestIds[i]);
  }

  Tpetra::Distributor responseDistributor(comm);
  const std::size_t numResponses =
      responseDistributor.createFromSends(responseProcIDs());

  const std::vector<long long> receivedRequestIds =
      detail::distributorForward(responseDistributor, responseRequestIds,
                                 numResponses);
  const std::vector<long long> receivedOccurrencePrefix =
      detail::distributorForward(responseDistributor,
                                 responseOccurrencePrefix, numResponses);
  const std::vector<long long> receivedCloneBaseLocal =
      detail::distributorForward(responseDistributor,
                                 responseCloneBaseLocal, numResponses);

  if (numResponses != exportGids.size()) {
    throw std::runtime_error(
        "TpetraMatrixTools::limitReuse response count does not match "
        "the number of source requests");
  }

  std::vector<long long> exportOccurrencePrefix(exportGids.size(), -1);
  std::vector<long long> exportCloneBaseLocal(exportGids.size(), -1);
  std::vector<unsigned char> responseSeen(exportGids.size(), 0);

  for (std::size_t i = 0; i < numResponses; ++i) {
    const long long requestIdLL = receivedRequestIds[i];
    if (requestIdLL < 0 ||
        static_cast<unsigned long long>(requestIdLL) >=
            static_cast<unsigned long long>(exportGids.size())) {
      throw std::runtime_error(
          "TpetraMatrixTools::limitReuse received an invalid request ID");
    }

    const std::size_t requestId = static_cast<std::size_t>(requestIdLL);
    if (responseSeen[requestId] != 0) {
      throw std::runtime_error(
          "TpetraMatrixTools::limitReuse received a duplicate response ID");
    }

    responseSeen[requestId] = 1;
    exportOccurrencePrefix[requestId] = receivedOccurrencePrefix[i];
    exportCloneBaseLocal[requestId] = receivedCloneBaseLocal[i];
  }

  for (std::size_t requestId = 0; requestId < responseSeen.size();
       ++requestId) {
    if (responseSeen[requestId] == 0) {
      throw std::runtime_error(
          "TpetraMatrixTools::limitReuse did not receive every assignment "
          "response");
    }
  }

  // -----------------------------------------------------------------------
  // 7. Rebuild the matrix using the same rows, row lengths, and values, but
  //    remap each original X reference to the appropriate synthetic clone.
  // -----------------------------------------------------------------------
  RCP<CrsMatrix> B =
      rcp(new CrsMatrix(rowMap, A->getLocalMaxNumRowEntries()));

  std::vector<long long> localOccurrence(numLocalCols, 0);

  const std::size_t numLocalRows = A->getLocalNumRows();
  for (std::size_t rowLid = 0; rowLid < numLocalRows; ++rowLid) {
    const GO rowGid = rowMap->getGlobalElement(static_cast<LO>(rowLid));

    const std::size_t begin =
        static_cast<std::size_t>(localA.graph.row_map(rowLid));
    const std::size_t end =
        static_cast<std::size_t>(localA.graph.row_map(rowLid + 1));

    Array<GO> newColumns;
    Array<Scalar> rowValues;
    newColumns.reserve(end - begin);
    rowValues.reserve(end - begin);

    for (std::size_t k = begin; k < end; ++k) {
      const LO oldColLid = localA.graph.entries(k);
      const std::size_t oldCol = static_cast<std::size_t>(oldColLid);

      if (oldCol >= requestIndexByColLid.size()) {
        throw std::runtime_error(
            "TpetraMatrixTools::limitReuse encountered invalid old column LID");
      }

      const std::size_t request = requestIndexByColLid[oldCol];
      if (request == noRequest) {
        throw std::runtime_error(
            "TpetraMatrixTools::limitReuse missing request metadata for used column");
      }

      const long long occurrence = localOccurrence[oldCol]++;
      const long long globalOccurrence =
          exportOccurrencePrefix[request] + occurrence;
      const long long cloneOrdinal =
          globalOccurrence / referencesPerSynthetic;

      const int owner = exportProcIDs[static_cast<Teuchos::Ordinal>(request)];
      const long long syntheticGidLL =
          ownerGlobalBase[static_cast<std::size_t>(owner)] +
          exportCloneBaseLocal[request] + cloneOrdinal;

      const GO syntheticGid = detail::checkedLongLongToGO<GO>(
          syntheticGidLL, "synthetic column GID");

      newColumns.push_back(syntheticGid);
      rowValues.push_back(localA.values(k));
    }

    B->insertGlobalValues(rowGid, newColumns(), rowValues());
  }

  for (std::size_t lid = 0; lid < numLocalCols; ++lid) {
    if (localOccurrence[lid] != localUseCount[lid]) {
      throw std::runtime_error(
          "TpetraMatrixTools::limitReuse internal occurrence-count mismatch");
    }
  }

  B->fillComplete(newDomainMap, rangeMap);

  // Validate that every synthetic column GID actually belongs to the
  // synthetic domain Map.  This catches numbering/communication bugs here,
  // before a downstream analyzer or SpMV sees an inconsistent operator.
  {
    const auto newColMap = B->getColMap();
    const auto newColGidList = newColMap->getLocalElementList();
    Array<GO> newColGids(newColGidList.size());
    Array<int> newColOwners(newColGidList.size());
    std::fill(newColOwners.begin(), newColOwners.end(), -1);

    for (std::size_t lid = 0; lid < newColGidList.size(); ++lid) {
      newColGids[lid] = newColGidList[lid];
    }

    const Tpetra::LookupStatus transformedLookup =
        newDomainMap->getRemoteIndexList(newColGids(), newColOwners());

    if (transformedLookup != Tpetra::AllIDsPresent) {
      GO missingGid = Teuchos::OrdinalTraits<GO>::invalid();
      for (std::size_t lid = 0; lid < newColOwners.size(); ++lid) {
        if (newColOwners[lid] < 0) {
          missingGid = newColGids[lid];
          break;
        }
      }
      throw std::runtime_error(
          "TpetraMatrixTools::limitReuse generated a synthetic column GID "
          "that is absent from the synthetic domain Map; first missing GID=" +
          std::to_string(static_cast<long long>(missingGid)));
    }
  }

  if (B->getGlobalNumEntries() != A->getGlobalNumEntries()) {
    throw std::runtime_error(
        "TpetraMatrixTools::limitReuse changed the global NNZ count; "
        "this usually indicates duplicate synthetic columns were merged");
  }

  if (B->getGlobalNumRows() != A->getGlobalNumRows()) {
    throw std::runtime_error(
        "TpetraMatrixTools::limitReuse changed the global row count");
  }

  localStats.transformed = true;
  if (stats != nullptr) *stats = localStats;
  return B;
}

}  // namespace TpetraMatrixTools

