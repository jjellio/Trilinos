// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// P2: focused MultiVector ownership and classical-fallback fixtures.
// Uses P1's CUDA instantiation and rank-local apply trace capture.
#ifndef TPETRA_VMM_P2_HELPERS_HPP
#define TPETRA_VMM_P2_HELPERS_HPP

#include "VmmSpmvTestHelpers.hpp"

#include "Teuchos_Range1D.hpp"
#include "Teuchos_CommHelpers.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace TpetraVmmP2 {

using namespace TpetraVmmP1;

inline auto distributedMap() {
  const auto comm = Tpetra::getDefaultComm();
  return Teuchos::rcp(new map_type(Tpetra::global_size_t(8), go_type(0), comm));
}

inline bool nativeExpected() {
  return Tpetra::Details::Behavior::experimentalVmm() &&
         Tpetra::Details::Behavior::experimentalVmmNativeMv();
}

inline bool twoRanks(std::ostream& out, bool& success) {
  const int np = Tpetra::getDefaultComm()->getSize();
  if (np != 2) {
    success = false;
    out << "P2 requires two MPI ranks, found " << np << '\n';
  }
  return np == 2;
}

inline void producerReady() {
  // Caller-side epochs only: not part of Tpetra::CrsMatrix::apply().
  Kokkos::fence();
  Tpetra::getDefaultComm()->barrier();
}

inline void consumerComplete() {
  Kokkos::fence();
  Tpetra::getDefaultComm()->barrier();
}

inline scalar_type input(const go_type gid, const std::size_t col = 0) {
  return static_cast<scalar_type>(gid + 1) *
         static_cast<scalar_type>(col + 1);
}

inline scalar_type allRemoteResult(const go_type gid, const std::size_t col = 0) {
  return 2.0 * input(gid, col) + 3.0 * input((gid + 4) % 8, col);
}

template <class Formula>
void setHost(vector_type& mv, const Formula& formula) {
  const auto map = mv.getMap();
  auto h = mv.getLocalViewHost(Tpetra::Access::OverwriteAll);
  for (std::size_t lid = 0; lid < mv.getLocalLength(); ++lid) {
    const go_type gid = map->getGlobalElement(static_cast<lo_type>(lid));
    for (std::size_t col = 0; col < mv.getNumVectors(); ++col) {
      h(lid, col) = formula(gid, col);
    }
  }
}

inline void synchronizeToDevice(vector_type& mv) {
  // Mirrors updated on host must be published to the ordinary device view.
  auto d = mv.getLocalViewDevice(Tpetra::Access::ReadOnly);
  (void)d;
}

template <class Formula>
void checkHost(vector_type& mv, const Formula& expected,
               const std::string& label, std::ostream& out, bool& success) {
  const auto map = mv.getMap();
  auto h = mv.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (std::size_t lid = 0; lid < mv.getLocalLength(); ++lid) {
    const go_type gid = map->getGlobalElement(static_cast<lo_type>(lid));
    for (std::size_t col = 0; col < mv.getNumVectors(); ++col) {
      const scalar_type actual = h(lid, col);
      const scalar_type want = expected(gid, col);
      if (!std::isfinite(actual) ||
          std::abs(actual - want) > 1.0e-12 * (1.0 + std::abs(want))) {
        success = false;
        out << "P2 mismatch " << label << " rank=" << Tpetra::getDefaultComm()->getRank()
            << " gid=" << gid << " col=" << col
            << " got=" << actual << " expected=" << want << '\n';
      }
    }
  }
}

inline void checkStorage(const vector_type& mv, bool vmmExpected,
                         const std::string& label,
                         std::ostream& out, bool& success) {
  if (mv.isVmmBacked() != vmmExpected ||
      (mv.getVmmGlobalPtr() != nullptr) != vmmExpected ||
      (mv.getVmmLayout() != nullptr) != vmmExpected) {
    success = false;
    out << "P2 storage mismatch " << label
        << " rank=" << Tpetra::getDefaultComm()->getRank()
        << " expected_vmm=" << vmmExpected
        << " backed=" << mv.isVmmBacked()
        << " ptr=" << static_cast<const void*>(mv.getVmmGlobalPtr())
        << " layout=" << static_cast<const void*>(mv.getVmmLayout()) << '\n';
  }
}

inline Teuchos::RCP<matrix_type> makeMatrix(
    const Teuchos::RCP<const map_type>& rowMap,
    const Teuchos::RCP<const map_type>& domainMap,
    const Teuchos::RCP<const map_type>& rangeMap,
    const std::string& label, bool weightedTranspose = false) {
  auto A = Teuchos::rcp(new matrix_type(rowMap, std::size_t(2)));
  A->setObjectLabel(label);
  for (std::size_t lid = 0; lid < rowMap->getLocalNumElements(); ++lid) {
    const go_type g = rowMap->getGlobalElement(static_cast<lo_type>(lid));
    Teuchos::Array<go_type> cols(2);
    Teuchos::Array<scalar_type> vals(2);
    cols[0] = g;
    cols[1] = (g + 4) % 8;
    vals[0] = 2.0;
    vals[1] = weightedTranspose ? 3.0 + static_cast<scalar_type>(g) : 3.0;
    A->insertGlobalValues(g, cols(), vals());
  }
  A->fillComplete(domainMap, rangeMap);
  return A;
}

// Deliberately permutes *rank ownership* while preserving the GID set.
// Provides a nontrivial row-to-range Exporter for F03.
inline auto swappedRangeMap() {
  const auto comm = Tpetra::getDefaultComm();
  Teuchos::Array<go_type> gids(4);
  for (int i = 0; i < 4; ++i) {
    gids[static_cast<std::size_t>(i)] =
        static_cast<go_type>((comm->getRank() == 0 ? 4 : 0) + i);
  }
  return Teuchos::rcp(new map_type(Tpetra::global_size_t(8), gids(), go_type(0), comm));
}

inline auto replicatedRangeMap() {
  const auto comm = Tpetra::getDefaultComm();
  return Teuchos::rcp(new map_type(Tpetra::global_size_t(8), go_type(0),
                                   comm, Tpetra::LocallyReplicated));
}

// P1 records only nontranspose apply paths; transpose is validated by its
// distinct mathematical result and the known classic applyTranspose route.
// In stats mode, verify the *rank-local* path inside the C++ test, not only
// in a postprocessed MPI log.
inline void applyWithExpectedPath(const matrix_type& A, const vector_type& X,
                                  vector_type& Y, const Teuchos::ETransp mode,
                                  const scalar_type alpha, const scalar_type beta,
                                  const std::string& expectedPath,
                                  const std::string& label,
                                  std::ostream& out, bool& success) {
  producerReady();
  const bool stats = Tpetra::Details::Behavior::experimentalVmmStatistics();
  if (stats && mode == Teuchos::NO_TRANS) {
    std::string captured;
    {
      TpetraVmmP1::ScopedApplyOutput cap;
      A.apply(X, Y, mode, alpha, beta);
      captured = cap.str();
    }
    std::cout << captured << std::flush;
    const std::string expectedLine =
        "[TPETRA_MATRIX] event=apply path=" + expectedPath +
        " rank=" + std::to_string(Tpetra::getDefaultComm()->getRank()) +
        " label=\"" + label + "\"";
    if (captured.find(expectedLine) == std::string::npos ||
        TpetraVmmP1::occurrences(captured, "[TPETRA_MATRIX] event=apply path=") != 1) {
      success = false;
      out << "P2 apply path mismatch: rank=" << Tpetra::getDefaultComm()->getRank()
          << " expected=" << expectedLine << " got=" << captured << '\n';
    }
  } else {
    A.apply(X, Y, mode, alpha, beta);
  }
  consumerComplete();
}

}  // namespace TpetraVmmP2
#endif // TPETRA_VMM_P2_HELPERS_HPP
