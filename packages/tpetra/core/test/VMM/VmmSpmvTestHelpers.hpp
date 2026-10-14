// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
//
// P1: small two-rank, closed-form correctness fixtures.  Intentionally do not
// use a second CrsMatrix::apply() as the oracle: a common error in classic and
// direct VMM kernels should not make the reference answer appear correct.

#ifndef TPETRA_VMM_SPMV_TEST_HELPERS_HPP
#define TPETRA_VMM_SPMV_TEST_HELPERS_HPP

#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "Tpetra_Details_Behavior.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstddef>
#include <limits>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace TpetraVmmP1 {

using scalar_type = double;
using lo_type = typename Tpetra::Map<>::local_ordinal_type;
using go_type = typename Tpetra::Map<>::global_ordinal_type;
// Explicitly test the CUDA instantiation, regardless of the default Node.
using node_type = Tpetra::KokkosCompat::KokkosCudaWrapperNode;
using map_type = Tpetra::Map<lo_type, go_type, node_type>;
using matrix_type = Tpetra::CrsMatrix<scalar_type, lo_type, go_type, node_type>;
using vector_type = Tpetra::MultiVector<scalar_type, lo_type, go_type, node_type>;
using row_entry_type = std::pair<go_type, scalar_type>;

enum class Pattern { Diagonal, Boundary, AllRemote, Stencil };

// Input values are assigned by GID, not LID.  This is intentional groundwork
// for P3's noncontiguous and reordered-Map regression cases.
inline scalar_type valueOfX(const go_type gid) {
  return static_cast<scalar_type>(gid + go_type(1));
}

inline scalar_type oldY(const go_type gid) {
  return static_cast<scalar_type>(go_type(7) + gid);
}

// Assemble the exact matrix, using hand-picked GIDs and weights.
inline std::vector<row_entry_type> entriesForRow(const Pattern pattern,
                                                  const go_type g) {
  std::vector<row_entry_type> entries;
  switch (pattern) {
    case Pattern::Diagonal:
      entries.emplace_back(g, static_cast<scalar_type>(g + go_type(2)));
      break;
    case Pattern::Boundary:
      entries.emplace_back(g, 2.0);
      if (g == go_type(3)) entries.emplace_back(go_type(4), 3.0);
      if (g == go_type(4)) entries.emplace_back(go_type(3), 3.0);
      break;
    case Pattern::AllRemote:
      entries.emplace_back(g, 2.0);
      entries.emplace_back((g + go_type(4)) % go_type(8), 3.0);
      break;
    case Pattern::Stencil:
      entries.emplace_back(g, 4.0);
      if (g > go_type(0)) entries.emplace_back(g - go_type(1), -1.0);
      if (g < go_type(7)) entries.emplace_back(g + go_type(1), -1.0);
      break;
  }
  return entries;
}

// Independent closed-form answers: do not call entriesForRow() here.
inline scalar_type exactAx(const Pattern pattern, const go_type g) {
  const scalar_type xg = valueOfX(g);
  switch (pattern) {
    case Pattern::Diagonal:
      return static_cast<scalar_type>(g + go_type(2)) * xg;
    case Pattern::Boundary:
      return 2.0 * xg + (g == go_type(3) ? 3.0 * valueOfX(go_type(4)) : 0.0) +
             (g == go_type(4) ? 3.0 * valueOfX(go_type(3)) : 0.0);
    case Pattern::AllRemote:
      return 2.0 * xg + 3.0 * valueOfX((g + go_type(4)) % go_type(8));
    case Pattern::Stencil: {
      scalar_type result = 4.0 * xg;
      if (g > go_type(0)) result -= valueOfX(g - go_type(1));
      if (g < go_type(7)) result -= valueOfX(g + go_type(1));
      return result;
    }
  }
  return std::numeric_limits<scalar_type>::quiet_NaN();
}

inline scalar_type exactY(const Pattern pattern, const go_type g,
                           const scalar_type alpha, const scalar_type beta) {
  // Sparse BLAS must not read old Y when beta == 0.  In that case the test
  // deliberately initializes Y to NaNs to reveal an incorrect implementation.
  if (alpha == 0.0) {
    return beta == 0.0 ? 0.0 : beta * oldY(g);
  }
  const scalar_type product = alpha * exactAx(pattern, g);
  return beta == 0.0 ? product : product + beta * oldY(g);
}

struct Coefficients {
  scalar_type alpha;
  scalar_type beta;
};

// Redirect stdout *only around an apply call* in statistics-enabled tests.
// This lets the unit test itself verify its rank's selected execution path,
// rather than relying on CTest's ability to parse interleaved MPI stdout.
// The captured text is replayed after restoration for the standalone runner.
class ScopedApplyOutput {
 public:
  ScopedApplyOutput() : previous_(std::cout.rdbuf(buffer_.rdbuf())) {}
  ~ScopedApplyOutput() { std::cout.rdbuf(previous_); }
  ScopedApplyOutput(const ScopedApplyOutput&) = delete;
  ScopedApplyOutput& operator=(const ScopedApplyOutput&) = delete;
  std::string str() const { return buffer_.str(); }

 private:
  std::ostringstream buffer_;
  std::streambuf* previous_;
};

inline std::size_t occurrences(const std::string& haystack,
                               const std::string& needle) {
  std::size_t count = 0;
  std::size_t offset = 0;
  while ((offset = haystack.find(needle, offset)) != std::string::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

inline void runPattern(const Pattern pattern, const std::string& matrixLabel,
                       const std::vector<Coefficients>& coefficients,
                       std::ostream& out, bool& success) {
  const auto comm = Tpetra::getDefaultComm();
  TEST_EQUALITY(comm->getSize(), 2);
  if (comm->getSize() != 2) return;

  // Exactly four contiguous GIDs per rank: [0,1,2,3] and [4,5,6,7].
  constexpr Tpetra::global_size_t globalRows = 8;
  const auto map = Teuchos::rcp(new map_type(globalRows, go_type(0), comm));
  const auto localRows = map->getLocalNumElements();
  TEST_EQUALITY(localRows, std::size_t(4));
  TEST_ASSERT(map->isContiguous());
  TEST_ASSERT(map->isOneToOne());
  if (localRows != std::size_t(4)) return;

  matrix_type A(map, std::size_t(3));
  A.setObjectLabel(matrixLabel);
  for (std::size_t lid = 0; lid < localRows; ++lid) {
    const go_type gid = map->getGlobalElement(static_cast<lo_type>(lid));
    const auto entries = entriesForRow(pattern, gid);
    Teuchos::Array<go_type> columns;
    Teuchos::Array<scalar_type> values;
    for (const auto& entry : entries) {
      columns.push_back(entry.first);
      values.push_back(entry.second);
    }
    A.insertGlobalValues(gid, columns(), values());
  }
  A.fillComplete(map, map);

  vector_type X(map, 1);
  vector_type Y(map, 1);
  // Storage configuration is a construction-time fact.  Assert it even in
  // stats=0 runs, where no allocation diagnostics are supposed to print.
  // This deliberately fails a native-VMM test when the VMM implementation
  // was not compiled in, instead of silently accepting classic fallback.
  const bool expectedVmmBacked =
      Tpetra::Details::Behavior::experimentalVmm() &&
      Tpetra::Details::Behavior::experimentalVmmNativeMv();
  TEST_EQUALITY(X.isVmmBacked(), expectedVmmBacked);
  TEST_EQUALITY(Y.isVmmBacked(), expectedVmmBacked);

  {
    auto xHost = X.getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (std::size_t lid = 0; lid < localRows; ++lid) {
      xHost(lid, 0) = valueOfX(map->getGlobalElement(static_cast<lo_type>(lid)));
    }
  }
  {
    // Explicitly synchronize X host writes to the ordinary device view.
    // For a native VMM MV, this device view aliases its peer-visible arena.
    auto xDevice = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
    (void)xDevice;
  }

  for (const auto& coeff : coefficients) {
    {
      auto yHost = Y.getLocalViewHost(Tpetra::Access::OverwriteAll);
      for (std::size_t lid = 0; lid < localRows; ++lid) {
        const go_type gid = map->getGlobalElement(static_cast<lo_type>(lid));
        yHost(lid, 0) = coeff.beta == 0.0
                            ? std::numeric_limits<scalar_type>::quiet_NaN()
                            : oldY(gid);
      }
    }
    {
      auto yDevice = Y.getLocalViewDevice(Tpetra::Access::ReadOnly);
      (void)yDevice;
    }

    // Distributed producer / consumer epochs are the caller's responsibility.
    // Do NOT move these fences/barriers into CrsMatrix::apply().
    Kokkos::fence();
    comm->barrier();

    const bool statsEnabled = Tpetra::Details::Behavior::experimentalVmmStatistics();
    std::string captured;
    if (statsEnabled) {
      {
        ScopedApplyOutput capture;
        A.apply(X, Y, Teuchos::NO_TRANS, coeff.alpha, coeff.beta);
        captured = capture.str();
      }
      // Re-emit once to leave auditable per-rank events in the MPI log.
      std::cout << captured << std::flush;
      const char* path = coeff.alpha == 0.0 ? "alpha_zero" :
                         expectedVmmBacked ? "vmm" : "classic";
      const std::string expectedLine =
          std::string("[TPETRA_MATRIX] event=apply path=") + path +
          " rank=" + std::to_string(comm->getRank()) +
          " label=\"" + matrixLabel + "\"";
      if (captured.find(expectedLine) == std::string::npos ||
          occurrences(captured, "[TPETRA_MATRIX] event=apply path=") != 1) {
        success = false;
        out << "P1 apply path mismatch: rank=" << comm->getRank()
            << " label=" << matrixLabel << " alpha=" << coeff.alpha
            << " expected=" << expectedLine << " captured=" << captured << '\n';
      }
    } else {
      A.apply(X, Y, Teuchos::NO_TRANS, coeff.alpha, coeff.beta);
    }
    Kokkos::fence();
    comm->barrier();

    {
      auto yHost = Y.getLocalViewHost(Tpetra::Access::ReadOnly);
      for (std::size_t lid = 0; lid < localRows; ++lid) {
        const go_type gid = map->getGlobalElement(static_cast<lo_type>(lid));
        const scalar_type expected = exactY(pattern, gid, coeff.alpha, coeff.beta);
        const scalar_type actual = yHost(lid, 0);
        // Unlike a simple relative check, this catches a NaN result even if
        // the comparison's normal relative-error formula would not.
        if (!std::isfinite(actual) ||
            std::abs(actual - expected) > 1.0e-12 * (1.0 + std::abs(expected))) {
          success = false;
          out << "P1 mismatch " << matrixLabel << " rank=" << comm->getRank()
              << " gid=" << gid << " alpha=" << coeff.alpha
              << " beta=" << coeff.beta << " got=" << actual
              << " expected=" << expected << '\n';
        }
      }
    }
  }
}

}  // namespace TpetraVmmP1

#endif  // TPETRA_VMM_SPMV_TEST_HELPERS_HPP
