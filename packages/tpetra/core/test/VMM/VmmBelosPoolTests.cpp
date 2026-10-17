// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
//
// B2: integration tests of Belos::MultiVecTraits with the experimental
// native-VMM-aware Belos MultiVecPool (B1).
//
// This file intentionally lives with the Tpetra VMM tests.  The optional
// BelosTpetra test target compiles it from packages/belos/tpetra/test so
// TpetraCore never acquires a dependency on Belos.
//
// Run the same tests under independent MPI processes for all Behavior flag
// combinations: Behavior caches environment variables after first access.
// All pool checkouts and releases have identical control flow on both ranks.

#include "VmmP2Helpers.hpp"
#include "BelosMultiVecTraits_Tpetra.hpp"
#include "Teuchos_Range1D.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace TpetraVmmBelosPool {

using namespace TpetraVmmP2;
using traits_type = Belos::MultiVecTraits<
    scalar_type, vector_type, Teuchos::SerialDenseMatrix<int, scalar_type>>;

// Belos's B1 cache emits these records only when the existing Tpetra VMM
// statistics Behavior is on.  Inspect local stdout rather than merge MPI logs.
inline bool statisticsOn() {
  return Tpetra::Details::Behavior::experimentalVmmStatistics();
}

inline std::size_t poolEvents(const std::string& trace, const char* event) {
  return TpetraVmmP1::occurrences(
      trace, std::string("[BELOS_MV_POOL] event=") + event + " ");
}

inline void replayTrace(const std::string& trace) {
  if (!trace.empty()) std::cout << trace << std::flush;
}

// Capture output *only* during a Belos clone, not while executing other MPI
// operations.  The native VMM constructor may itself use MPI collectives;
// ScopedApplyOutput only redirects this rank's std::cout, not MPI traffic.
inline Teuchos::RCP<vector_type> cloneWithTrace(
    const vector_type& source, const int nvec, std::string& trace) {
  if (!statisticsOn()) return traits_type::Clone(source, nvec);
  Teuchos::RCP<vector_type> result;
  {
    TpetraVmmP1::ScopedApplyOutput capture;
    result = traits_type::Clone(source, nvec);
    trace = capture.str();
  }
  replayTrace(trace);
  return result;
}

inline const void* devicePtr(vector_type& mv) {
  // Local pointer only.  Never compare device or VMM virtual addresses
  // between MPI ranks (the address spaces are independent).
  auto d = mv.getLocalViewDevice(Tpetra::Access::ReadOnly);
  return static_cast<const void*>(d.data());
}

inline void checkOneColumnPoolStorage(const vector_type& mv,
                                      const std::string& label,
                                      std::ostream& out, bool& success) {
  TEST_EQUALITY(mv.getNumVectors(), std::size_t(1));
  TEST_ASSERT(mv.isConstantStride());
  checkStorage(mv, nativeExpected(), label, out, success);
}

}  // namespace TpetraVmmBelosPool

TEUCHOS_UNIT_TEST(VMM_BelosPool, B01_Clone_OneColumn_Storage) {
  using namespace TpetraVmmBelosPool;
  if (!twoRanks(out, success)) return;

  const auto map = distributedMap();
  vector_type source(map, 1);
  auto cloned = traits_type::Clone(source, 1);

  TEST_ASSERT(!cloned.is_null());
  TEST_ASSERT(cloned.getRawPtr() != &source);
  TEST_ASSERT(cloned->getMap().getRawPtr() == map.getRawPtr());
  TEST_EQUALITY(cloned->getLocalLength(), map->getLocalNumElements());
  checkOneColumnPoolStorage(*cloned, "B01 Belos Clone", out, success);

  // Clone returns unspecified values: explicitly overwrite before reading.
  setHost(*cloned, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(*cloned);
  producerReady();
  checkHost(*cloned, [](go_type gid, std::size_t) { return input(gid); },
            "B01 initialized clone", out, success);
  consumerComplete();
  cloned = Teuchos::null;
}

TEUCHOS_UNIT_TEST(VMM_BelosPool, B02_Clone_ReleaseAndReuse) {
  using namespace TpetraVmmBelosPool;
  if (!twoRanks(out, success)) return;

  const auto map = distributedMap();
  vector_type source(map, 1);
  std::string firstTrace, secondTrace;

  auto first = cloneWithTrace(source, 1, firstTrace);
  checkOneColumnPoolStorage(*first, "B02 first checkout", out, success);
  const void* firstObject = static_cast<const void*>(first.getRawPtr());
  const auto firstGlobal = first->getVmmGlobalPtr();
  setHost(*first, [](go_type gid, std::size_t) { return 10.0 + input(gid); });
  synchronizeToDevice(*first);
  const void* firstDevice = devicePtr(*first);
  TEST_ASSERT(firstDevice != nullptr);
  consumerComplete();
  first = Teuchos::null; // Return entire MV, not just its DualView.

  auto second = cloneWithTrace(source, 1, secondTrace);
  checkOneColumnPoolStorage(*second, "B02 second checkout", out, success);
  TEST_ASSERT(devicePtr(*second) == firstDevice);
  if (nativeExpected()) {
    // The VMM cache retains an entire MV; the conventional pool retains
    // only DualViews and creates a fresh MV wrapper on checkout.
    TEST_ASSERT(static_cast<const void*>(second.getRawPtr()) == firstObject);
    TEST_ASSERT(second->getVmmGlobalPtr() == firstGlobal);
    TEST_ASSERT(firstGlobal != nullptr);
  }

  if (statisticsOn()) {
    if (nativeExpected()) {
      // New Map identity in each Teuchos test: the first request must miss,
      // and the second request must hit the same native VMM arena.
      TEST_EQUALITY(poolEvents(firstTrace, "miss"), std::size_t(1));
      TEST_EQUALITY(poolEvents(firstTrace, "hit"), std::size_t(0));
      TEST_EQUALITY(poolEvents(secondTrace, "hit"), std::size_t(1));
      TEST_EQUALITY(poolEvents(secondTrace, "miss"), std::size_t(0));
    } else {
      TEST_EQUALITY(TpetraVmmP1::occurrences(firstTrace, "[BELOS_MV_POOL]"),
                    std::size_t(0));
      TEST_EQUALITY(TpetraVmmP1::occurrences(secondTrace, "[BELOS_MV_POOL]"),
                    std::size_t(0));
    }
  }

  // A reused slot must accept fresh data; it must not expose stale values.
  setHost(*second, [](go_type gid, std::size_t) { return 50.0 + input(gid); });
  synchronizeToDevice(*second);
  producerReady();
  checkHost(*second, [](go_type gid, std::size_t) { return 50.0 + input(gid); },
            "B02 rewritten reuse", out, success);
  consumerComplete();
  second = Teuchos::null;
}

TEUCHOS_UNIT_TEST(VMM_BelosPool, B03_Clone_ConcurrentSlots) {
  using namespace TpetraVmmBelosPool;
  if (!twoRanks(out, success)) return;

  const auto map = distributedMap();
  vector_type source(map, 1);
  auto first = traits_type::Clone(source, 1);
  auto second = traits_type::Clone(source, 1);
  checkOneColumnPoolStorage(*first, "B03 first live", out, success);
  checkOneColumnPoolStorage(*second, "B03 second live", out, success);

  const auto firstGlobal = first->getVmmGlobalPtr();
  const auto secondGlobal = second->getVmmGlobalPtr();
  const void* firstObject = static_cast<const void*>(first.getRawPtr());
  const void* secondObject = static_cast<const void*>(second.getRawPtr());
  TEST_ASSERT(firstObject != secondObject);
  if (nativeExpected()) {
    TEST_ASSERT(firstGlobal != nullptr);
    TEST_ASSERT(secondGlobal != nullptr);
    TEST_ASSERT(firstGlobal != secondGlobal);
  }

  setHost(*first, [](go_type gid, std::size_t) { return input(gid); });
  setHost(*second, [](go_type gid, std::size_t) { return 100.0 + input(gid); });
  synchronizeToDevice(*first);
  synchronizeToDevice(*second);
  TEST_ASSERT(devicePtr(*first) != devicePtr(*second));
  producerReady();
  checkHost(*first, [](go_type gid, std::size_t) { return input(gid); },
            "B03 independent first", out, success);
  checkHost(*second, [](go_type gid, std::size_t) { return 100.0 + input(gid); },
            "B03 independent second", out, success);
  consumerComplete();

  // Identical release order on all ranks; the LIFO VMM list must return
  // the second allocation first.  Never intentionally skew this test.
  first = Teuchos::null;
  second = Teuchos::null;
  auto reused = traits_type::Clone(source, 1);
  checkOneColumnPoolStorage(*reused, "B03 reused", out, success);
  if (nativeExpected()) {
    TEST_ASSERT(static_cast<const void*>(reused.getRawPtr()) == secondObject);
    TEST_ASSERT(reused->getVmmGlobalPtr() == secondGlobal);
  }
  setHost(*reused, [](go_type gid, std::size_t) { return 200.0 + input(gid); });
  synchronizeToDevice(*reused);
  producerReady();
  checkHost(*reused, [](go_type gid, std::size_t) { return 200.0 + input(gid); },
            "B03 reused slot", out, success);
  consumerComplete();
  reused = Teuchos::null;
}

TEUCHOS_UNIT_TEST(VMM_BelosPool, B04_CloneCopy_DeepCopy) {
  using namespace TpetraVmmBelosPool;
  if (!twoRanks(out, success)) return;

  const auto map = distributedMap();
  vector_type source(map, 1);
  setHost(source, [](go_type gid, std::size_t) { return 10.0 + input(gid); });
  synchronizeToDevice(source);
  producerReady();

  auto copied = traits_type::CloneCopy(source);
  checkOneColumnPoolStorage(*copied, "B04 CloneCopy", out, success);
  TEST_ASSERT(copied.getRawPtr() != &source);
  TEST_ASSERT(devicePtr(*copied) != devicePtr(source));
  producerReady();
  checkHost(*copied, [](go_type gid, std::size_t) { return 10.0 + input(gid); },
            "B04 initial copy", out, success);

  setHost(source, [](go_type gid, std::size_t) { return 30.0 + input(gid); });
  synchronizeToDevice(source);
  producerReady();
  checkHost(*copied, [](go_type gid, std::size_t) { return 10.0 + input(gid); },
            "B04 copy after source modification", out, success);

  const void* oldCopyObject = static_cast<const void*>(copied.getRawPtr());
  const auto oldCopyGlobal = copied->getVmmGlobalPtr();
  consumerComplete();
  copied = Teuchos::null;
  auto recopy = traits_type::CloneCopy(source);
  checkOneColumnPoolStorage(*recopy, "B04 recycled CloneCopy", out, success);
  if (nativeExpected()) {
    TEST_ASSERT(static_cast<const void*>(recopy.getRawPtr()) == oldCopyObject);
    TEST_ASSERT(recopy->getVmmGlobalPtr() == oldCopyGlobal);
  }
  producerReady();
  checkHost(*recopy, [](go_type gid, std::size_t) { return 30.0 + input(gid); },
            "B04 recopy", out, success);
  consumerComplete();
  recopy = Teuchos::null;

  // Exercise both single-column selection overloads of CloneCopy.
  // A three-column source is intentionally standard-backed.
  vector_type many(map, 3);
  setHost(many, [](go_type gid, std::size_t col) { return input(gid, col); });
  synchronizeToDevice(many);
  producerReady();
  auto byIndex = traits_type::CloneCopy(many, std::vector<int>{1});
  auto byRange = traits_type::CloneCopy(many, Teuchos::Range1D(2, 2));
  checkOneColumnPoolStorage(*byIndex, "B04 index CloneCopy", out, success);
  checkOneColumnPoolStorage(*byRange, "B04 range CloneCopy", out, success);
  TEST_ASSERT(devicePtr(*byIndex) != devicePtr(*byRange));
  producerReady();
  checkHost(*byIndex, [](go_type gid, std::size_t) { return 2.0 * input(gid); },
            "B04 index copy values", out, success);
  checkHost(*byRange, [](go_type gid, std::size_t) { return 3.0 * input(gid); },
            "B04 range copy values", out, success);
  consumerComplete();
  byIndex = Teuchos::null;
  byRange = Teuchos::null;
}

TEUCHOS_UNIT_TEST(VMM_BelosPool, B05_PooledX_DirectSpMV) {
  using namespace TpetraVmmBelosPool;
  if (!twoRanks(out, success)) return;

  const auto map = distributedMap();
  const std::string label("B05_BelosPool_AllRemote");
  auto A = makeMatrix(map, map, map, label);
  vector_type source(map, 1);
  vector_type Y(map, 1);
  const std::string path = nativeExpected() ? "vmm" : "classic";

  // Perform one apply on a fresh pooled X and another on the same cached
  // complete MV after release/recheckout.  The expected result is analytical
  // (rank 0: 17/22/27/32, rank 1: 13/18/23/28).
  const void* firstObject = nullptr;
  const scalar_type* firstGlobal = nullptr;
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto X = traits_type::Clone(source, 1);
    checkOneColumnPoolStorage(*X, "B05 pooled X", out, success);
    if (iteration == 0) {
      firstObject = static_cast<const void*>(X.getRawPtr());
      firstGlobal = X->getVmmGlobalPtr();
    } else {
      if (nativeExpected()) {
        TEST_ASSERT(static_cast<const void*>(X.getRawPtr()) == firstObject);
        TEST_ASSERT(X->getVmmGlobalPtr() == firstGlobal);
      }
    }

    setHost(*X, [](go_type gid, std::size_t) { return input(gid); });
    synchronizeToDevice(*X);
    setHost(Y, [](go_type, std::size_t) { return -9.0; });
    synchronizeToDevice(Y);

    // In stats mode, applyWithExpectedPath asserts exactly one per-rank
    // `path=vmm` / `path=classic` event for the labeled forward apply.
    // Correct numerical results via the classic fallback do not pass B05.
    applyWithExpectedPath(*A, *X, Y, Teuchos::NO_TRANS,
                          1.0, 0.0, path, label, out, success);
    checkHost(Y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
              "B05 all-remote Ax", out, success);
    consumerComplete();
    X = Teuchos::null;
  }
}

TEUCHOS_UNIT_TEST(VMM_BelosPool, B06_Ineligible_PreservesStandardPool) {
  using namespace TpetraVmmBelosPool;
  if (!twoRanks(out, success)) return;

  const auto map = distributedMap();
  vector_type source(map, 1);
  std::string firstTrace, secondTrace;

  auto first = cloneWithTrace(source, 3, firstTrace);
  TEST_EQUALITY(first->getNumVectors(), std::size_t(3));
  checkStorage(*first, false, "B06 three columns", out, success);
  setHost(*first, [](go_type gid, std::size_t col) { return input(gid, col); });
  synchronizeToDevice(*first);
  const void* firstDevice = devicePtr(*first);
  producerReady();
  checkHost(*first, [](go_type gid, std::size_t col) { return input(gid, col); },
            "B06 three-column values", out, success);
  consumerComplete();
  first = Teuchos::null;

  auto second = cloneWithTrace(source, 3, secondTrace);
  TEST_ASSERT(devicePtr(*second) == firstDevice);
  checkStorage(*second, false, "B06 standard cache reuse", out, success);
  second = Teuchos::null;
  if (statisticsOn()) {
    TEST_EQUALITY(TpetraVmmP1::occurrences(firstTrace, "[BELOS_MV_POOL]"),
                  std::size_t(0));
    TEST_EQUALITY(TpetraVmmP1::occurrences(secondTrace, "[BELOS_MV_POOL]"),
                  std::size_t(0));
  }

  // With either VMM flag disabled, single-column requests also take the
  // original standard DualView pool.  They are tested as fresh CTest MPI
  // processes, not by changing Behavior variables within this process.
  if (!nativeExpected()) {
    std::string offTrace;
    auto off = cloneWithTrace(source, 1, offTrace);
    checkOneColumnPoolStorage(*off, "B06 single-column flags off", out, success);
    if (statisticsOn()) {
      TEST_EQUALITY(TpetraVmmP1::occurrences(offTrace, "[BELOS_MV_POOL]"),
                    std::size_t(0));
    }
    off = Teuchos::null;
  }
}

// B07 directly verifies that Belos's column-view adapters preserve native
// VMM identity without creating new VMM allocations.  Both overloads route
// through Tpetra::MultiVector::subView(0,0).
TEUCHOS_UNIT_TEST(VMM_BelosPool, B07_CloneViewsRetainCanonicalVmm) {
  using namespace TpetraVmmBelosPool;
  if (!twoRanks(out, success)) return;

  const auto map = distributedMap();
  vector_type source(map, 1);
  setHost(source, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(source);

  auto byIndex = traits_type::CloneViewNonConst(source, std::vector<int>{0});
  auto byRange = traits_type::CloneViewNonConst(source, Teuchos::Range1D(0, 0));
  auto readIndex = traits_type::CloneView(source, std::vector<int>{0});
  auto readRange = traits_type::CloneView(source, Teuchos::Range1D(0, 0));
  checkOneColumnPoolStorage(*byIndex, "B07 Belos mutable index", out, success);
  checkOneColumnPoolStorage(*byRange, "B07 Belos mutable range", out, success);
  checkOneColumnPoolStorage(*readIndex, "B07 Belos const index", out, success);
  checkOneColumnPoolStorage(*readRange, "B07 Belos const range", out, success);

  const auto global = source.getVmmGlobalPtr();
  const auto layout = source.getVmmLayout();
  TEST_ASSERT(byIndex->getVmmGlobalPtr() == global);
  TEST_ASSERT(byRange->getVmmGlobalPtr() == global);
  TEST_ASSERT(readIndex->getVmmGlobalPtr() == global);
  TEST_ASSERT(readRange->getVmmGlobalPtr() == global);
  TEST_ASSERT(byIndex->getVmmLayout() == layout);
  TEST_ASSERT(byRange->getVmmLayout() == layout);

  // Mutable views alias source data; neither modifies source Map ownership.
  setHost(*byRange, [](go_type gid, std::size_t) { return 90.0 + input(gid); });
  synchronizeToDevice(*byRange);
  producerReady();
  checkHost(source, [](go_type gid, std::size_t) { return 90.0 + input(gid); },
            "B07 Belos view alias", out, success);
  consumerComplete();

  // Belos's read-only full-column view must also drive VMM SpMV, not just
  // expose the correct metadata.
  const std::string label = "B07_BelosCloneView_AllRemote";
  auto A = makeMatrix(map, map, map, label);
  vector_type Y(map, 1);
  const std::string path = nativeExpected() ? "vmm" : "classic";
  applyWithExpectedPath(*A, *readIndex, Y, Teuchos::NO_TRANS,
                        1.0, 0.0, path, label, out, success);
  checkHost(Y, [](go_type gid, std::size_t) {
    return 5.0 * 90.0 + allRemoteResult(gid);
  }, "B07 Belos read-only view apply", out, success);
  consumerComplete();
}
