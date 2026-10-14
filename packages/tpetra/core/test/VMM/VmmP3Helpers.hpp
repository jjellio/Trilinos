// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// P3: distributed Map fixtures, independent numerical oracles, and a
// test-only phase switch.  Do not add these checks to the SpMV hot path.
#ifndef TPETRA_VMM_P3_HELPERS_HPP
#define TPETRA_VMM_P3_HELPERS_HPP

#include "VmmP2Helpers.hpp"

#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace TpetraVmmP3 {
using namespace TpetraVmmP2;

// The same sources serve as the pre-refactor baseline and post-refactor
// acceptance tests.  This variable affects *tests only*, not Tpetra::Behavior.
inline bool expectGeneralizedMaps() {
  const char* raw = std::getenv("TPETRA_VMM_TEST_MAP_MODE");
  if (raw == nullptr || std::string(raw) == "baseline") return false;
  if (std::string(raw) == "generalized") return true;
  throw std::invalid_argument(
      "TPETRA_VMM_TEST_MAP_MODE must be baseline or generalized");
}

inline auto explicitMap(std::initializer_list<go_type> rank0,
                        std::initializer_list<go_type> rank1,
                        const Tpetra::global_size_t globalEntries) {
  const auto comm = Tpetra::getDefaultComm();
  const auto& local = comm->getRank() == 0 ? rank0 : rank1;
  Teuchos::Array<go_type> gids;
  for (const auto gid : local) gids.push_back(gid);
  return Teuchos::rcp(new map_type(globalEntries, gids(), go_type(0), comm));
}

inline auto interleavedMap() {
  return explicitMap({0, 2, 4, 6}, {1, 3, 5, 7}, Tpetra::global_size_t(8));
}

inline auto permutedMap() {
  // The same ownership partition as G01, with *different local LIDs*.
  return explicitMap({4, 0, 6, 2}, {7, 3, 1, 5}, Tpetra::global_size_t(8));
}

inline auto overlappingMap() {
  // Nine global slots but eight distinct GIDs. GID 3 is held by both ranks.
  // Each local occurrence remains a separate, valid MultiVector storage slot.
  return explicitMap({0, 1, 2, 3}, {3, 4, 5, 6, 7},
                     Tpetra::global_size_t(9));
}

inline bool checkFixtureGids(const Teuchos::RCP<const map_type>& map,
                             std::initializer_list<go_type> rank0,
                             std::initializer_list<go_type> rank1,
                             const std::string& caseName,
                             std::ostream& out, bool& success) {
  // Perform an all-rank check *before* any collective construction, so a
  // malformed rank-local fixture does not let its peer enter an MPI call.
  const auto comm = Tpetra::getDefaultComm();
  const auto& want = comm->getRank() == 0 ? rank0 : rank1;
  int localOk = map->getLocalNumElements() == want.size() ? 1 : 0;
  if (localOk) {
    std::size_t lid = 0;
    for (const auto gid : want) {
      if (map->getGlobalElement(static_cast<lo_type>(lid)) != gid) localOk = 0;
      ++lid;
    }
  }
  int globalOk = 0;
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &localOk, &globalOk);
  if (!globalOk) {
    success = false;
    out << "P3 invalid test fixture: " << caseName << " rank=" << comm->getRank()
        << " expected local GID ordering differs\n";
    return false;
  }
  return true;
}

inline scalar_type neighborResult(const go_type gid) {
  // Independent analytic oracle, not derived from the assembled CRS entries.
  return 2.0 * input(gid) + 3.0 * input((gid + 1) % 8);
}

inline auto makeNeighborMatrix(const Teuchos::RCP<const map_type>& map,
                               const std::string& label) {
  auto A = Teuchos::rcp(new matrix_type(map, std::size_t(2)));
  A->setObjectLabel(label);
  for (std::size_t lid = 0; lid < map->getLocalNumElements(); ++lid) {
    const go_type gid = map->getGlobalElement(static_cast<lo_type>(lid));
    Teuchos::Array<go_type> cols(2);
    Teuchos::Array<scalar_type> vals(2);
    cols[0] = gid;
    cols[1] = (gid + 1) % 8;  // Remote on both G01 and G02 partitions.
    vals[0] = 2.0;
    vals[1] = 3.0;
    A->insertGlobalValues(gid, cols(), vals());
  }
  A->fillComplete(map, map);
  return A;
}

inline auto makeDiagonalWithDomain(const Teuchos::RCP<const map_type>& rowMap,
                                   const Teuchos::RCP<const map_type>& domainMap,
                                   const std::string& label) {
  auto A = Teuchos::rcp(new matrix_type(rowMap, std::size_t(1)));
  A->setObjectLabel(label);
  for (std::size_t lid = 0; lid < rowMap->getLocalNumElements(); ++lid) {
    const go_type gid = rowMap->getGlobalElement(static_cast<lo_type>(lid));
    Teuchos::Array<go_type> cols(1);
    Teuchos::Array<scalar_type> vals(1);
    cols[0] = gid;
    vals[0] = 2.0;
    A->insertGlobalValues(gid, cols(), vals());
  }
  A->fillComplete(domainMap, rowMap);
  return A;
}

// Matrix fillComplete must preserve the historical rejection diagnostics
// in baseline mode; generalized mode specifies the desired future result.
// Capture only around *construction*, not around an MPI or device epoch.
template <class MatrixFactory>
auto makeAndCheckMatrix(const MatrixFactory& factory,
                        const std::string& label,
                        const std::string& expectedReason,
                        std::ostream& out, bool& success) {
  using matrix_rcp = Teuchos::RCP<matrix_type>;
  matrix_rcp A;
  if (Tpetra::Details::Behavior::experimentalVmmStatistics()) {
    std::string captured;
    {
      ScopedApplyOutput capture;
      A = factory();
      captured = capture.str();
    }
    std::cout << captured << std::flush;
    if (Tpetra::getDefaultComm()->getRank() == 0 &&
        Tpetra::Details::Behavior::experimentalVmm()) {
      const auto rejectionPrefix =
          std::string("[TPETRA_MATRIX] event=vmmSkip");
      const auto reasonText = "reason=" + expectedReason;
      const bool hasSkip = captured.find(rejectionPrefix) != std::string::npos;
      const bool hasLabel = captured.find("label=\"" + label + "\"") !=
                            std::string::npos;
      if ((!expectedReason.empty() &&
           (!hasSkip || !hasLabel || captured.find(reasonText) == std::string::npos)) ||
          (expectedReason.empty() && hasSkip)) {
        success = false;
        out << "P3 matrix eligibility mismatch label=" << label
            << " expected_reason=" << (expectedReason.empty() ? "<none>" : expectedReason)
            << " observed=" << captured << '\n';
      }
    }
  } else {
    A = factory();
  }
  return A;
}

inline void prepareX(vector_type& X) {
  setHost(X, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(X);
}

inline void executeAndCheck(const matrix_type& A, const vector_type& X,
                            vector_type& Y, const std::string& path,
                            const std::string& label,
                            const bool diagonal,
                            std::ostream& out, bool& success) {
  Y.putScalar(-315.0);
  applyWithExpectedPath(A, X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        path, label, out, success);
  checkHost(Y, [diagonal](go_type gid, std::size_t) {
    return diagonal ? 2.0 * input(gid) : neighborResult(gid);
  }, label, out, success);
}

inline std::string standardOrVmmPath(const bool eligibleInGeneralizedMode) {
  return nativeExpected() && eligibleInGeneralizedMode ? "vmm" : "classic";
}

}  // namespace TpetraVmmP3
#endif  // TPETRA_VMM_P3_HELPERS_HPP
