// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// P2 F01-F07: deliberate classical paths with independent analytic answers.

#include "VmmP2Helpers.hpp"

TEUCHOS_UNIT_TEST(VMM_Fallback, F01_OrdinaryInputAgainstVmmMatrix) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  auto map = distributedMap();
  auto A = makeMatrix(map, map, map, "P2_F01_OrdinaryInput");

  // A three-column parent can only use standard allocation in M4.1.
  // Its one-column view is therefore *not* VMM-backed, even in full VMM mode.
  vector_type parent(map, 3);
  setHost(parent, [](go_type gid, std::size_t col) { return input(gid, col); });
  synchronizeToDevice(parent);
  auto X = parent.subViewNonConst(Teuchos::Range1D(0, 0));
  vector_type Y(map, 1);
  checkStorage(*X, false, "F01 deliberately standard X", out, success);
  checkStorage(Y, nativeExpected(), "F01 Y", out, success);
  Y.putScalar(-7.0);
  applyWithExpectedPath(*A, *X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        "classic", "P2_F01_OrdinaryInput", out, success);
  checkHost(Y, [](go_type gid, std::size_t) {
    return allRemoteResult(gid);
  }, "F01", out, success);
}

TEUCHOS_UNIT_TEST(VMM_Fallback, F02_ThreeColumns) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  auto map = distributedMap();
  auto A = makeMatrix(map, map, map, "P2_F02_ThreeColumns");
  vector_type X(map, 3), Y(map, 3);
  checkStorage(X, false, "F02 X", out, success);
  checkStorage(Y, false, "F02 Y", out, success);
  setHost(X, [](go_type gid, std::size_t col) { return input(gid, col); });
  synchronizeToDevice(X);
  Y.putScalar(-7.0);
  applyWithExpectedPath(*A, X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        "classic", "P2_F02_ThreeColumns", out, success);
  checkHost(Y, [](go_type gid, std::size_t col) {
    return allRemoteResult(gid, col);
  }, "F02", out, success);
}

TEUCHOS_UNIT_TEST(VMM_Fallback, F03_NontrivialExporter) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  auto rowMap = distributedMap();
  auto rangeMap = swappedRangeMap();
  auto A = makeMatrix(rowMap, rowMap, rangeMap, "P2_F03_Exporter");
  if (A->getGraph()->getExporter().is_null()) {
    success = false;
    out << "F03 fixture did not create a nontrivial Exporter\n";
    return;
  }
  vector_type X(rowMap, 1), Y(rangeMap, 1);
  checkStorage(X, nativeExpected(), "F03 input", out, success);
  setHost(X, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(X);
  Y.putScalar(-7.0);
  applyWithExpectedPath(*A, X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        "classic", "P2_F03_Exporter", out, success);
  checkHost(Y, [](go_type gid, std::size_t) {
    return allRemoteResult(gid);
  }, "F03 reordered range", out, success);
}

TEUCHOS_UNIT_TEST(VMM_Fallback, F04_LocallyReplicatedRange) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  auto rowMap = distributedMap();
  auto replicated = replicatedRangeMap();
  if (replicated->isDistributed()) {
    success = false;
    out << "F04 fixture is not locally replicated\n";
    return;
  }
  auto A = makeMatrix(rowMap, rowMap, replicated, "P2_F04_ReplicatedRange");
  if (A->getGraph()->getExporter().is_null()) {
    success = false;
    out << "F04 fixture did not create its required Exporter\n";
    return;
  }
  vector_type X(rowMap, 1), Y(replicated, 1);
  checkStorage(X, nativeExpected(), "F04 input", out, success);
  setHost(X, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(X);
  Y.putScalar(-7.0);
  applyWithExpectedPath(*A, X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        "classic", "P2_F04_ReplicatedRange", out, success);
  if (Y.getLocalLength() != std::size_t(8)) {
    success = false;
    out << "F04 expected 8 locally replicated Y entries, got "
        << Y.getLocalLength() << '\n';
  }
  checkHost(Y, [](go_type gid, std::size_t) {
    return allRemoteResult(gid);
  }, "F04 replicated on both ranks", out, success);
}

TEUCHOS_UNIT_TEST(VMM_Fallback, F05_Transpose) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  auto map = distributedMap();
  // A(i,i)=2, A(i,(i+4)%8)=3+i, so A^T != A.
  auto A = makeMatrix(map, map, map, "P2_F05_Transpose", true);
  vector_type X(map, 1), Y(map, 1);
  checkStorage(X, nativeExpected(), "F05 input", out, success);
  setHost(X, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(X);
  Y.putScalar(-7.0);
  applyWithExpectedPath(*A, X, Y, Teuchos::TRANS, 1.0, 0.0,
                        "classic", "P2_F05_Transpose", out, success);
  checkHost(Y, [](go_type gid, std::size_t) {
    const go_type remote = (gid + 4) % 8;
    return 2.0 * input(gid) +
           (3.0 + static_cast<scalar_type>(remote)) * input(remote);
  }, "F05 A-transpose", out, success);
}

TEUCHOS_UNIT_TEST(VMM_Fallback, F06_AlphaZeroBypass) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  auto map = distributedMap();
  auto A = makeMatrix(map, map, map, "P2_F06_AlphaZero");
  vector_type X(map, 1), Y(map, 1);
  setHost(Y, [](go_type gid, std::size_t) {
    return 7.0 + static_cast<scalar_type>(gid);
  });
  synchronizeToDevice(Y);
  // X is deliberately not used when alpha==0.
  applyWithExpectedPath(*A, X, Y, Teuchos::NO_TRANS, 0.0, -2.0,
                        "alpha_zero", "P2_F06_AlphaZero", out, success);
  checkHost(Y, [](go_type gid, std::size_t) {
    return -2.0 * (7.0 + static_cast<scalar_type>(gid));
  }, "F06 alpha-zero", out, success);
}

TEUCHOS_UNIT_TEST(VMM_Fallback, F07_GlobalVmmDisabled) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  if (Tpetra::Details::Behavior::experimentalVmm()) {
    out << "F07: not applicable in global-VMM-enabled scenario\n";
    return;
  }
  auto map = distributedMap();
  auto A = makeMatrix(map, map, map, "P2_F07_GlobalDisabled");
  vector_type X(map, 1), Y(map, 1);
  checkStorage(X, false, "F07 X", out, success);
  checkStorage(Y, false, "F07 Y", out, success);
  setHost(X, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(X);
  Y.putScalar(-7.0);
  applyWithExpectedPath(*A, X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        "classic", "P2_F07_GlobalDisabled", out, success);
  checkHost(Y, [](go_type gid, std::size_t) {
    return allRemoteResult(gid);
  }, "F07 global off", out, success);
}
