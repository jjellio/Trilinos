#pragma once

#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace TpetraMatrixInfo {

// Signed 64-bit-style counter.  long long is used deliberately because it is
// a native Teuchos communication type on MPI and non-MPI builds.
using count_type = long long;

struct AccessAnalysis {
  int numRanks = 0;
  int myRank   = 0;

  // Row-major dense matrices indexed as [sourceRank][xOwnerRank].
  //
  // references(i,j): number of CRS scalar entries in rows owned by rank i
  //                  whose X entry is owned by rank j.
  //
  // uniqueX(i,j):    number of distinct X entries owned by rank j that are
  //                  referenced at least once by rows owned by rank i.
  std::vector<count_type> references;
  std::vector<count_type> uniqueX;

  // Additional per-rank metadata useful when interpreting the matrices.
  std::vector<count_type> localRows;
  std::vector<count_type> localDomainEntries;

  count_type reference(const int sourceRank, const int ownerRank) const {
    return references[index(sourceRank, ownerRank)];
  }

  count_type unique(const int sourceRank, const int ownerRank) const {
    return uniqueX[index(sourceRank, ownerRank)];
  }

  double reuse(const int sourceRank, const int ownerRank) const {
    const count_type u = unique(sourceRank, ownerRank);
    return u == 0 ? 0.0
                  : static_cast<double>(reference(sourceRank, ownerRank)) /
                        static_cast<double>(u);
  }

  count_type totalReferences(const int sourceRank) const {
    count_type sum = 0;
    for (int j = 0; j < numRanks; ++j) sum += reference(sourceRank, j);
    return sum;
  }

  count_type localReferences(const int sourceRank) const {
    return reference(sourceRank, sourceRank);
  }

  count_type remoteReferences(const int sourceRank) const {
    return totalReferences(sourceRank) - localReferences(sourceRank);
  }

  count_type totalUniqueX(const int sourceRank) const {
    count_type sum = 0;
    for (int j = 0; j < numRanks; ++j) sum += unique(sourceRank, j);
    return sum;
  }

  count_type localUniqueX(const int sourceRank) const {
    return unique(sourceRank, sourceRank);
  }

  count_type remoteUniqueX(const int sourceRank) const {
    return totalUniqueX(sourceRank) - localUniqueX(sourceRank);
  }

  double remoteReferenceFraction(const int sourceRank) const {
    const count_type total = totalReferences(sourceRank);
    return total == 0 ? 0.0
                      : static_cast<double>(remoteReferences(sourceRank)) /
                            static_cast<double>(total);
  }

  // Aggregate reuse among all off-rank references made by sourceRank.
  // This is not the average of pairwise reuse values; it is
  //   sum(remote references) / sum(unique remote X entries).
  double remoteReuse(const int sourceRank) const {
    const count_type u = remoteUniqueX(sourceRank);
    return u == 0 ? 0.0
                  : static_cast<double>(remoteReferences(sourceRank)) /
                        static_cast<double>(u);
  }

 private:
  std::size_t index(const int sourceRank, const int ownerRank) const {
    if (sourceRank < 0 || sourceRank >= numRanks ||
        ownerRank < 0 || ownerRank >= numRanks) {
      throw std::out_of_range("TpetraMatrixInfo::AccessAnalysis rank index out of range");
    }
    return static_cast<std::size_t>(sourceRank) *
               static_cast<std::size_t>(numRanks) +
           static_cast<std::size_t>(ownerRank);
  }
};

// Analyze the X-access pattern implied by a fill-complete Tpetra::CrsMatrix.
//
// This routine is collective over A.getDomainMap()->getComm().
//
// It does NOT depend on VMM, Import, Export, Distributor, CUDA, or MPI APIs.
// Teuchos communication is used only to resolve Map ownership and to assemble
// the per-rank rows into collective dense matrices.
//
// The domain Map must be one-to-one so every X GID has a unique owner.  The
// Map need not be contiguous or uniformly distributed.
template <class CrsMatrix>
AccessAnalysis analyzeAccessPattern(const CrsMatrix& A) {
  using LO = typename CrsMatrix::local_ordinal_type;
  using GO = typename CrsMatrix::global_ordinal_type;

  if (!A.isFillComplete()) {
    throw std::runtime_error(
        "TpetraMatrixInfo::analyzeAccessPattern requires a fill-complete matrix");
  }

  const auto domainMap = A.getDomainMap();
  const auto colMap    = A.getColMap();

  if (domainMap.is_null() || colMap.is_null()) {
    throw std::runtime_error(
        "TpetraMatrixInfo::analyzeAccessPattern requires domain and column maps");
  }

  const auto comm = domainMap->getComm();
  const int rank  = comm->getRank();
  const int size  = comm->getSize();

  // The ownership matrix is only well-defined if each X entry has one owner.
  // isOneToOne() is itself collective.
  if (!domainMap->isOneToOne()) {
    throw std::runtime_error(
        "TpetraMatrixInfo::analyzeAccessPattern requires a one-to-one domain Map");
  }

  const std::size_t nRanks = static_cast<std::size_t>(size);
  if (nRanks != 0 &&
      nRanks > static_cast<std::size_t>(std::numeric_limits<int>::max()) / nRanks) {
    throw std::runtime_error(
        "TpetraMatrixInfo::analyzeAccessPattern communicator too large for dense analysis");
  }

  const std::size_t denseSize = nRanks * nRanks;
  if (denseSize > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "TpetraMatrixInfo::analyzeAccessPattern dense matrix exceeds Teuchos count range");
  }

  // Resolve every local column-map GID to the rank that owns that X entry in
  // the domain Map.  For noncontiguous maps this may communicate, but it is a
  // one-time analysis/setup operation and uses Tpetra's normal Map directory.
  const auto colGidView = colMap->getLocalElementList();
  const std::size_t numLocalCols = colGidView.size();

  Teuchos::Array<GO> colGids(numLocalCols);
  Teuchos::Array<int> owners(numLocalCols);
  std::fill(owners.begin(), owners.end(), -1);

  for (std::size_t lid = 0; lid < numLocalCols; ++lid) {
    colGids[lid] = colGidView[lid];
  }

  const Tpetra::LookupStatus lookupStatus =
      domainMap->getRemoteIndexList(colGids(), owners());

  if (lookupStatus != Tpetra::AllIDsPresent) {
    throw std::runtime_error(
        "TpetraMatrixInfo::analyzeAccessPattern: column Map contains GIDs "
        "that are not present in the domain Map");
  }

  for (std::size_t lid = 0; lid < numLocalCols; ++lid) {
    if (owners[lid] < 0 || owners[lid] >= size) {
      throw std::runtime_error(
          "TpetraMatrixInfo::analyzeAccessPattern: invalid domain-map owner rank");
    }
  }

  // Scan only the local CRS graph.  Values are irrelevant to this analysis.
  const auto localA = A.getLocalMatrixHost();
  const std::size_t localNnz = static_cast<std::size_t>(localA.nnz());

  std::vector<count_type> refsByOwner(nRanks, 0);
  std::vector<count_type> uniqueByOwner(nRanks, 0);
  std::vector<unsigned char> seenColumn(numLocalCols, 0);

  for (std::size_t k = 0; k < localNnz; ++k) {
    const LO colLid = localA.graph.entries(k);
    if (colLid < static_cast<LO>(0) ||
        static_cast<std::size_t>(colLid) >= numLocalCols) {
      throw std::runtime_error(
          "TpetraMatrixInfo::analyzeAccessPattern: local CRS column LID "
          "is outside the column Map");
    }

    const std::size_t lid = static_cast<std::size_t>(colLid);
    const int owner       = owners[lid];

    ++refsByOwner[static_cast<std::size_t>(owner)];

    if (seenColumn[lid] == 0) {
      seenColumn[lid] = 1;
      ++uniqueByOwner[static_cast<std::size_t>(owner)];
    }
  }

  // Each rank contributes exactly one row to each dense matrix.  A SUM
  // all-reduce is therefore equivalent to gathering all rows, while keeping
  // the full result available on every rank for later heuristic decisions.
  std::vector<count_type> localReferences(denseSize, 0);
  std::vector<count_type> localUniqueX(denseSize, 0);
  std::vector<count_type> globalReferences(denseSize, 0);
  std::vector<count_type> globalUniqueX(denseSize, 0);

  const std::size_t rowOffset =
      static_cast<std::size_t>(rank) * nRanks;

  for (int owner = 0; owner < size; ++owner) {
    const std::size_t j = static_cast<std::size_t>(owner);
    localReferences[rowOffset + j] = refsByOwner[j];
    localUniqueX[rowOffset + j]    = uniqueByOwner[j];
  }

  const int denseCount = static_cast<int>(denseSize);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, denseCount,
                     localReferences.data(), globalReferences.data());
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, denseCount,
                     localUniqueX.data(), globalUniqueX.data());

  // Collect a small amount of row/domain metadata using the same pattern.
  std::vector<count_type> localMeta(2 * nRanks, 0);
  std::vector<count_type> globalMeta(2 * nRanks, 0);

  localMeta[static_cast<std::size_t>(rank)] =
      static_cast<count_type>(A.getLocalNumRows());
  localMeta[nRanks + static_cast<std::size_t>(rank)] =
      static_cast<count_type>(domainMap->getLocalNumElements());

  Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM,
                     static_cast<int>(localMeta.size()),
                     localMeta.data(), globalMeta.data());

  AccessAnalysis result;
  result.numRanks = size;
  result.myRank   = rank;
  result.references = std::move(globalReferences);
  result.uniqueX    = std::move(globalUniqueX);
  result.localRows.assign(globalMeta.begin(),
                          globalMeta.begin() + static_cast<std::ptrdiff_t>(nRanks));
  result.localDomainEntries.assign(
      globalMeta.begin() + static_cast<std::ptrdiff_t>(nRanks),
      globalMeta.end());

  // Strong consistency check: one reference is counted for every local CRS
  // entry, therefore each dense row sum must equal that rank's local NNZ.
  // Only the calling rank's local nnz is immediately known here; the global
  // row data were assembled collectively, so validate the calling row now.
  if (result.totalReferences(rank) != static_cast<count_type>(localNnz)) {
    throw std::runtime_error(
        "TpetraMatrixInfo::analyzeAccessPattern internal error: reference "
        "row sum does not equal local nnz");
  }

  return result;
}

inline void printCountMatrix(std::ostream& os,
                             const std::string& title,
                             const AccessAnalysis& a,
                             const std::vector<count_type>& matrix) {
  const int width = 14;

  os << title << '\n';
  os << std::setw(width) << "rank\\X owner";
  for (int j = 0; j < a.numRanks; ++j) os << std::setw(width) << j;
  os << std::setw(width) << "row sum" << '\n';

  for (int i = 0; i < a.numRanks; ++i) {
    os << std::setw(width) << i;
    count_type rowSum = 0;
    for (int j = 0; j < a.numRanks; ++j) {
      const count_type v = matrix[static_cast<std::size_t>(i) *
                                      static_cast<std::size_t>(a.numRanks) +
                                  static_cast<std::size_t>(j)];
      rowSum += v;
      os << std::setw(width) << v;
    }
    os << std::setw(width) << rowSum << '\n';
  }
}

inline void printReuseMatrix(std::ostream& os, const AccessAnalysis& a) {
  const int width = 14;

  os << "Reuse factor R/U (references per distinct X entry)\n";
  os << std::setw(width) << "rank\\X owner";
  for (int j = 0; j < a.numRanks; ++j) os << std::setw(width) << j;
  os << '\n';

  const std::ios::fmtflags oldFlags = os.flags();
  const std::streamsize oldPrecision = os.precision();
  os << std::fixed << std::setprecision(2);

  for (int i = 0; i < a.numRanks; ++i) {
    os << std::setw(width) << i;
    for (int j = 0; j < a.numRanks; ++j) {
      if (a.unique(i, j) == 0) {
        os << std::setw(width) << "-";
      } else {
        os << std::setw(width) << a.reuse(i, j);
      }
    }
    os << '\n';
  }

  os.flags(oldFlags);
  os.precision(oldPrecision);
}

inline void printRankSummary(std::ostream& os, const AccessAnalysis& a) {
  const int width = 16;

  os << "Per-rank access summary\n";
  os << std::setw(6) << "rank"
     << std::setw(width) << "rows"
     << std::setw(width) << "domain X"
     << std::setw(width) << "refs"
     << std::setw(width) << "local refs"
     << std::setw(width) << "remote refs"
     << std::setw(width) << "remote unique"
     << std::setw(width) << "remote frac"
     << std::setw(width) << "remote reuse"
     << '\n';

  const std::ios::fmtflags oldFlags = os.flags();
  const std::streamsize oldPrecision = os.precision();
  os << std::fixed << std::setprecision(4);

  for (int i = 0; i < a.numRanks; ++i) {
    os << std::setw(6) << i
       << std::setw(width) << a.localRows[static_cast<std::size_t>(i)]
       << std::setw(width) << a.localDomainEntries[static_cast<std::size_t>(i)]
       << std::setw(width) << a.totalReferences(i)
       << std::setw(width) << a.localReferences(i)
       << std::setw(width) << a.remoteReferences(i)
       << std::setw(width) << a.remoteUniqueX(i)
       << std::setw(width) << a.remoteReferenceFraction(i)
       << std::setw(width) << a.remoteReuse(i)
       << '\n';
  }

  os.flags(oldFlags);
  os.precision(oldPrecision);
}

// Print the complete analysis on rootRank.  This function is not collective;
// analyzeAccessPattern() has already made the full data available on all ranks.
inline void print(std::ostream& os,
                  const AccessAnalysis& a,
                  const int rootRank = 0) {
  if (a.myRank != rootRank) return;

  printCountMatrix(os, "Reference matrix R: CRS scalar X references", a,
                   a.references);
  os << '\n';
  printCountMatrix(os, "Unique-X matrix U: distinct X entries referenced", a,
                   a.uniqueX);
  os << '\n';
  printReuseMatrix(os, a);
  os << '\n';
  printRankSummary(os, a);
}

// Long-form CSV is convenient for plotting / post-processing and avoids
// awkward N-column CSV schemas as process counts change.
inline void writeCsv(std::ostream& os,
                     const AccessAnalysis& a,
                     const int rootRank = 0) {
  if (a.myRank != rootRank) return;

  os << "source_rank,owner_rank,references,unique_x,reuse,is_remote\n";
  os << std::setprecision(17);

  for (int i = 0; i < a.numRanks; ++i) {
    for (int j = 0; j < a.numRanks; ++j) {
      os << i << ','
         << j << ','
         << a.reference(i, j) << ','
         << a.unique(i, j) << ','
         << a.reuse(i, j) << ','
         << (i == j ? 0 : 1) << '\n';
    }
  }
}

}  // namespace TpetraMatrixInfo

