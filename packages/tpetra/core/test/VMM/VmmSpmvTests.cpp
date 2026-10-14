// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
//
// P1 numerical reference cases.  No random input or Matrix Market fixtures.
// The two-rank matrix and input are deliberately small and analytically known.

#include "VmmSpmvTestHelpers.hpp"

TEUCHOS_UNIT_TEST(VMM_SpMV, S01_Diagonal) {
  TpetraVmmP1::runPattern(TpetraVmmP1::Pattern::Diagonal,
                         "P1_S01_Diagonal", {{1.0, 0.0}}, out, success);
}

TEUCHOS_UNIT_TEST(VMM_SpMV, S02_Boundary) {
  TpetraVmmP1::runPattern(TpetraVmmP1::Pattern::Boundary,
                         "P1_S02_Boundary", {{1.0, 0.0}}, out, success);
}

TEUCHOS_UNIT_TEST(VMM_SpMV, S03_AllRemote) {
  // Rank 0 expected [17,22,27,32]; rank 1 expected [13,18,23,28].
  TpetraVmmP1::runPattern(TpetraVmmP1::Pattern::AllRemote,
                         "P1_S03_AllRemote", {{1.0, 0.0}}, out, success);
}

TEUCHOS_UNIT_TEST(VMM_SpMV, S04_Stencil) {
  TpetraVmmP1::runPattern(TpetraVmmP1::Pattern::Stencil,
                         "P1_S04_Stencil", {{1.0, 0.0}}, out, success);
}

TEUCHOS_UNIT_TEST(VMM_SpMV, S06_AlphaBeta) {
  // The last two calls bypass SpMV entirely in ordinary Tpetra, since
  // alpha == 0.  The diagnostic log must say path=alpha_zero, not path=vmm.
  TpetraVmmP1::runPattern(
      TpetraVmmP1::Pattern::AllRemote, "P1_S06_AlphaBeta",
      {{1.0, 0.0}, {2.0, 0.0}, {1.0, 1.0}, {-0.5, 2.0},
       {0.0, -1.0}, {0.0, 0.0}},
      out, success);
}
