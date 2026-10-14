// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// P4: opt-in synchronized diagnostics measure timer invocation counts,
// not time thresholds. Normal P0-P3 SpMV tests still use async semantics.

#include "VmmP4Helpers.hpp"
#include "Teuchos_TimeMonitor.hpp"

TEUCHOS_UNIT_TEST(VMM_Timer, T01_SynchronizedPathCounters) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  TEST_ASSERT(Tpetra::Details::Behavior::synchronizeApply());
  TEST_ASSERT(Tpetra::Details::Behavior::experimentalVmmStatistics());

  const auto map = distributedMap();
  const auto A = makeMatrix(map, map, map, "P4_T01_TimedRemote");
  vector_type X(map, 1), Y(map, 1);
  prepareX(X);
  Y.putScalar(-17.0);

  // getNewTimer is a named counter registry: it returns an existing timer
  // with this name. Read counts immediately before/after one A.apply().
  // This is far more deterministic than comparing elapsed milliseconds.
  const auto vmmTimer = Teuchos::TimeMonitor::getNewTimer("Tpetra VMM: direct spmv");
  const auto classicTimer = Teuchos::TimeMonitor::getNewTimer("Tpetra Classic: spmv");
  const int vmmBefore = vmmTimer->numCalls();
  const int classicBefore = classicTimer->numCalls();

  const bool useVmm = nativeExpected();
  applyWithExpectedPath(*A, X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        useVmm ? "vmm" : "classic",
                        "P4_T01_TimedRemote", out, success);
  checkHost(Y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "T01 timed SpMV", out, success);

  const int vmmDelta = vmmTimer->numCalls() - vmmBefore;
  const int classicDelta = classicTimer->numCalls() - classicBefore;
  TEST_EQUALITY(vmmDelta, useVmm ? 1 : 0);
  TEST_EQUALITY(classicDelta, useVmm ? 0 : 1);
  std::cout << "[VMM_P4] event=timer rank=" << Tpetra::getDefaultComm()->getRank()
            << " path=" << (useVmm ? "vmm" : "classic")
            << " direct_delta=" << vmmDelta
            << " classic_delta=" << classicDelta << '\n' << std::flush;
}
