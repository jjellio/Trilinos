// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// P4: zero-local-entry and one-versus-seven distributed Maps.

#include "VmmP4Helpers.hpp"
#include "Tpetra_Details_VmmLayout.hpp"

#include <iostream>
#include <string>

TEUCHOS_UNIT_TEST(VMM_Boundary, B01_EmptyRankAllocation) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  const auto comm = Tpetra::getDefaultComm();
  const auto map = emptyRankMap();
  const std::size_t expectedLocal = comm->getRank() == 0 ? 8 : 0;
  TEST_EQUALITY(map->getLocalNumElements(), expectedLocal);
  TEST_EQUALITY(map->getGlobalNumElements(), Tpetra::global_size_t(8));
  TEST_ASSERT(map->isContiguous());
  TEST_ASSERT(map->isOneToOne());

  vector_type X(map, 1);
  checkStorage(X, nativeExpected(), "B01 empty rank storage", out, success);
  TEST_EQUALITY(X.getLocalLength(), expectedLocal);
  if (nativeExpected() && X.getVmmLayout() != nullptr) {
    const auto* layout = X.getVmmLayout();
    TEST_EQUALITY(layout->logicalCounts().at(0), std::uint64_t(8));
    TEST_EQUALITY(layout->logicalCounts().at(1), std::uint64_t(0));
    TEST_ASSERT(layout->bytesByRank().at(1) > 0);
    TEST_ASSERT(layout->baseElements().at(1) > layout->baseElements().at(0));
  }
  setHost(X, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(X);
  producerReady();
  checkHost(X, [](go_type gid, std::size_t) { return input(gid); },
            "B01 zero-local rows", out, success);
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_Boundary, B02_EmptyRankDiagonalApply) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  const auto comm = Tpetra::getDefaultComm();
  const auto map = emptyRankMap();
  auto A = makeDiagonalWithDomain(map, map, "P4_B02_EmptyRank");
  vector_type X(map, 1), Y(map, 1);
  checkStorage(X, nativeExpected(), "B02 X", out, success);
  checkStorage(Y, nativeExpected(), "B02 Y", out, success);
  prepareX(X);
  Y.putScalar(-99.0);
  producerReady();

  const bool stats = Tpetra::Details::Behavior::experimentalVmmStatistics();
  std::string captured;
  if (stats) {
    {
      ScopedApplyOutput capture;
      A->apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0);
      captured = capture.str();
    }
    std::cout << captured << std::flush;
  } else {
    A->apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0);
  }
  consumerComplete();

  checkHost(Y, [](go_type gid, std::size_t) { return 2.0 * input(gid); },
            "B02 empty-rank SpMV", out, success);

  // With an empty rank, Tpetra's own isDistributed() / local structure
  // determines whether this optimization is applicable. We do not prescribe
  // a new production policy in P4. We DO assert all ranks chose the same
  // path; mixed MPI and direct-VMM execution is never acceptable.
  if (stats) {
    const std::string common = " rank=" + std::to_string(comm->getRank()) +
                               " label=\"P4_B02_EmptyRank\"";
    const bool didVmm = captured.find("event=apply path=vmm" + common) !=
                        std::string::npos;
    const bool didClassic = captured.find("event=apply path=classic" + common) !=
                            std::string::npos;
    const int localPath = didVmm ? 1 : (didClassic ? 0 : -1);
    const int count = occurrences(captured, "[TPETRA_MATRIX] event=apply path=");
    TEST_EQUALITY(count, 1);
    TEST_ASSERT(didVmm != didClassic);
    int lowest = -1, highest = -1;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &localPath, &lowest);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &localPath, &highest);
    TEST_EQUALITY(lowest, highest);
    if (localPath >= 0) {
      std::cout << "[VMM_P4] event=zeroRowsDispatch rank=" << comm->getRank()
                << " path=" << (localPath == 1 ? "vmm" : "classic")
                << '\n' << std::flush;
    }
  }
}

TEUCHOS_UNIT_TEST(VMM_Boundary, B03_OneVsSevenRemoteApply) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  const auto comm = Tpetra::getDefaultComm();
  const auto map = skewedMap();
  TEST_EQUALITY(map->getLocalNumElements(),
                std::size_t(comm->getRank() == 0 ? 1 : 7));
  TEST_ASSERT(map->isContiguous());
  TEST_ASSERT(map->isOneToOne());

  const auto A = makeMatrix(map, map, map, "P4_B03_SkewedRemote");
  vector_type X(map, 1), Y(map, 1);
  checkStorage(X, nativeExpected(), "B03 X", out, success);
  checkStorage(Y, nativeExpected(), "B03 Y", out, success);
  if (nativeExpected() && X.getVmmLayout() != nullptr) {
    const auto* layout = X.getVmmLayout();
    TEST_EQUALITY(layout->logicalCounts().at(0), std::uint64_t(1));
    TEST_EQUALITY(layout->logicalCounts().at(1), std::uint64_t(7));
    TEST_ASSERT(layout->baseElements().at(1) > layout->baseElements().at(0));
  }
  prepareX(X);
  Y.putScalar(-99.0);
  applyWithExpectedPath(*A, X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        nativeExpected() ? "vmm" : "classic",
                        "P4_B03_SkewedRemote", out, success);
  checkHost(Y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "B03 one-versus-seven", out, success);
}
