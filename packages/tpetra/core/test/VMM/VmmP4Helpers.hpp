// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// P4: very small contiguous, highly skewed Maps and debug-test utilities.
#ifndef TPETRA_VMM_P4_HELPERS_HPP
#define TPETRA_VMM_P4_HELPERS_HPP

#include "VmmP3Helpers.hpp"

#include <exception>
#include <cstdint>
#include <string>

namespace TpetraVmmP4 {
using namespace TpetraVmmP3;

// Both Maps use the rank-local-count Map constructor. Unlike the P3
// explicit-GID maps, these are contiguous and one-to-one even when one rank
// contributes zero entries. Global IDs occupy [0,8) in local-LID order.
inline Teuchos::RCP<const map_type> countMap(std::size_t rank0Count) {
  const auto comm = Tpetra::getDefaultComm();
  const std::size_t localCount = comm->getRank() == 0 ? rank0Count : 8 - rank0Count;
  return Teuchos::rcp(new map_type(Tpetra::global_size_t(8), localCount,
                                  go_type(0), comm));
}

inline Teuchos::RCP<const map_type> emptyRankMap() { return countMap(8); }
inline Teuchos::RCP<const map_type> skewedMap() { return countMap(1); }

inline scalar_type neighborExpected(go_type gid) {
  return 2.0 * input(gid) + 3.0 * input((gid + 1) % go_type(8));
}

// The test intentionally compares real exception content, not just whether
// something threw.  No MPI operations occur until all ranks have returned
// from the (uniformly invalid) operation under test.
inline void checkDebugException(const std::string& message,
                                const std::string& requiredText,
                                const std::string& name,
                                std::ostream& out, bool& success) {
  const int rank = Tpetra::getDefaultComm()->getRank();
  const bool correct = !message.empty() &&
                       message.find(requiredText) != std::string::npos;
  if (!correct) {
    success = false;
    out << "P4 " << name << " rank=" << rank
        << " expected exception containing '" << requiredText
        << "', got: " << (message.empty() ? "<no exception>" : message) << '\n';
  } else {
    std::cout << "[VMM_P4] event=debugCaught case=" << name
              << " rank=" << rank << " result=PASS\n" << std::flush;
  }
}

}  // namespace TpetraVmmP4
#endif  // TPETRA_VMM_P4_HELPERS_HPP
