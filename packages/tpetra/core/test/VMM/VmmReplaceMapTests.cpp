// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// Experiment A hotfix: replaceMap() must remain noncollective.  Preserve
// native VMM identity for the identical Map object only; invalidate when
// any different Map object is supplied.  These tests include rank-subset
// execution resembling MueLu repartitioning.

#include "VmmP2Helpers.hpp"

namespace {

using namespace TpetraVmmP2;

Teuchos::RCP<const map_type> reversedLocalMap() {
  const auto comm = Tpetra::getDefaultComm();
  const go_type first = static_cast<go_type>(comm->getRank() * 4);
  Teuchos::Array<go_type> gids(4);
  for (int i = 0; i < 4; ++i) {
    gids[static_cast<std::size_t>(i)] = first + (3 - i);
  }
  return Teuchos::rcp(new map_type(Tpetra::global_size_t(8), gids(),
                                   go_type(0), comm));
}

// Change the Map only on rank 1; rank 0 continues to use its original
// Map object.  Noncollective replaceMap() makes a local decision.
Teuchos::RCP<const map_type> changedOnRankOne(
    const Teuchos::RCP<const map_type>& original) {
  // Map construction is collective even if only one rank subsequently uses
  // the new object.  Construct a valid rank-ordered distribution on both
  // ranks, then keep the original Map object on rank 0.
  const auto comm = Tpetra::getDefaultComm();
  const int rank = comm->getRank();
  const go_type first = static_cast<go_type>(rank * 4);
  Teuchos::Array<go_type> gids(4);
  for (int i = 0; i < 4; ++i) {
    gids[static_cast<std::size_t>(i)] =
        first + (rank == 0 ? i : 3 - i);
  }
  const auto changed = Teuchos::rcp(new map_type(
      Tpetra::global_size_t(8), gids(), go_type(0), comm));
  if (rank == 0) return original;
  return changed;
}

void verifyPreserved(const vector_type& x, const bool expected,
                     const void* originalGlobalPtr,
                     const void* originalLayout,
                     const std::string& label,
                     std::ostream& out, bool& success) {
  checkStorage(x, expected, label, out, success);
  if (expected &&
      (x.getVmmGlobalPtr() != originalGlobalPtr ||
       x.getVmmLayout() != originalLayout)) {
    success = false;
    out << label << ": VMM allocation/layout changed during replaceMap\n";
  }
}

void checkOriginalValues(vector_type& x, const char* name,
                         std::ostream& out, bool& success) {
  checkHost(x, [](go_type gid, std::size_t) { return input(gid); },
            name, out, success);
}

} // namespace

TEUCHOS_UNIT_TEST(VMM_ReplaceMap, A01_IdenticalMapPreservesArena) {
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  vector_type x(map, 1);
  const bool expected = nativeExpected();
  const void* const globalPtr = x.getVmmGlobalPtr();
  const void* const layout = x.getVmmLayout();
  setHost(x, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(x);
  producerReady();

  for (int k = 0; k < 4; ++k) {
    x.replaceMap(map);
    verifyPreserved(x, expected, globalPtr, layout,
                    "A01 identical Map", out, success);
  }
  checkOriginalValues(x, "A01 content", out, success);
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_ReplaceMap, A02_DistinctEquivalentMapInvalidates) {
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto equivalent = distributedMap();
  TEST_ASSERT(map.getRawPtr() != equivalent.getRawPtr());
  TEST_ASSERT(map->isSameAs(*equivalent));

  vector_type x(map, 1);
  setHost(x, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(x);
  producerReady();

  x.replaceMap(equivalent);
  checkStorage(x, false, "A02 distinct equivalent Map", out, success);
  checkOriginalValues(x, "A02 equivalent content", out, success);
  x.replaceMap(map);
  checkStorage(x, false, "A02 restore must not resurrect arena", out,
               success);
  checkOriginalValues(x, "A02 original content", out, success);
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_ReplaceMap, A03_LocalPermutationInvalidates) {
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto permuted = reversedLocalMap();
  TEST_ASSERT(!map->isSameAs(*permuted));
  vector_type x(map, 1);
  setHost(x, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(x);
  producerReady();

  x.replaceMap(permuted);
  checkStorage(x, false, "A03 permutation", out, success);
  x.replaceMap(map);
  checkStorage(x, false, "A03 restore must not resurrect old arena",
               out, success);
  checkOriginalValues(x, "A03 content", out, success);
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_ReplaceMap, A04_AsymmetricMapChangeNoncollective) {
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto partlyChanged = changedOnRankOne(map);
  const int rank = Tpetra::getDefaultComm()->getRank();
  vector_type x(map, 1);
  producerReady();

  // Rank 0 performs a no-op, while rank 1 changes its Map.  This must not
  // communicate.  Until all ranks again agree on distributed identity,
  // callers must not dispatch VMM SpMV using this MV.
  x.replaceMap(partlyChanged);
  checkStorage(x, rank == 0 && nativeExpected(),
               "A04 asymmetric change", out, success);
  x.replaceMap(map);
  checkStorage(x, rank == 0 && nativeExpected(),
               "A04 asymmetric restore", out, success);
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_ReplaceMap, A05_ExcludedRankInvalidatesLocally) {
  if (!twoRanks(out, success)) return;
  const auto original = distributedMap();
  const auto comm = Tpetra::getDefaultComm();
  vector_type x(original, 1);
  producerReady();

  // A valid Map projection: rank 0 keeps its original four GIDs while rank 1
  // is removed from the new communicator.  Both ranks provide a changed Map
  // and invalidate without communicating inside replaceMap().
  auto subcomm = comm->split(comm->getRank() == 0 ? 0 : -1, comm->getRank());
  Teuchos::RCP<const map_type> subsetMap = Teuchos::null;
  if (comm->getRank() == 0) {
    subsetMap = Teuchos::rcp(new map_type(Tpetra::global_size_t(4),
                                         go_type(0), subcomm));
  }
  x.replaceMap(subsetMap);
  checkStorage(x, false, "A05 subset", out, success);
  TEST_ASSERT(x.getMap().is_null() == (comm->getRank() == 1));

  // Re-entering the original communicator must not claim the old VMM
  // allocation: this path may allocate a new, conventional local DualView.
  x.replaceMap(original);
  checkStorage(x, false, "A05 restored distribution", out, success);
  TEST_ASSERT(!x.getMap().is_null());
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_ReplaceMap, A06_SpMVBeforeAfterMapChanges) {
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto equivalent = distributedMap();
  const auto permuted = reversedLocalMap();
  const auto A = makeMatrix(map, map, map, "ReplaceMap_A06");
  vector_type x(map, 1), y(map, 1);
  setHost(x, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(x);
  producerReady();

  const std::string directOrClassic = nativeExpected() ? "vmm" : "classic";
  x.replaceMap(map);
  checkStorage(x, nativeExpected(), "A06 same Map before apply", out, success);
  applyWithExpectedPath(*A, x, y, Teuchos::NO_TRANS, 1.0, 0.0,
                        directOrClassic, "ReplaceMap_A06", out, success);
  checkHost(y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "A06 same Map", out, success);

  // Distinct-but-equivalent Maps must now use classic fallback.  A previous
  // attempt to verify distributed equivalence inside replaceMap() deadlocked
  // when MueLu entered the routine on a subset of ranks.
  x.replaceMap(equivalent);
  checkStorage(x, false, "A06 equivalent before apply", out, success);
  applyWithExpectedPath(*A, x, y, Teuchos::NO_TRANS, 1.0, 0.0,
                        "classic", "ReplaceMap_A06", out, success);
  checkHost(y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "A06 distinct equivalent Map", out, success);

  x.replaceMap(map);
  checkStorage(x, false, "A06 original before apply", out, success);
  applyWithExpectedPath(*A, x, y, Teuchos::NO_TRANS, 1.0, 0.0,
                        "classic", "ReplaceMap_A06", out, success);
  checkHost(y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "A06 restored Map", out, success);

  x.replaceMap(permuted);
  checkStorage(x, false, "A06 invalid after permutation", out, success);
  x.replaceMap(map);
  checkStorage(x, false, "A06 invalid after restore", out, success);
  applyWithExpectedPath(*A, x, y, Teuchos::NO_TRANS, 1.0, 0.0,
                        "classic", "ReplaceMap_A06", out, success);
  checkHost(y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "A06 classic after invalidation", out, success);
}

TEUCHOS_UNIT_TEST(VMM_ReplaceMap, A07_StandardMultiColumnUnaffected) {
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto equivalent = distributedMap();
  vector_type x(map, 3);
  checkStorage(x, false, "A07 initial", out, success);
  setHost(x, [](go_type gid, std::size_t col) { return input(gid, col); });
  synchronizeToDevice(x);

  x.replaceMap(equivalent);
  checkStorage(x, false, "A07 equivalent", out, success);
  checkHost(x, [](go_type gid, std::size_t col) { return input(gid, col); },
            "A07 values", out, success);
  x.replaceMap(map);
  checkStorage(x, false, "A07 original", out, success);
}

TEUCHOS_UNIT_TEST(VMM_ReplaceMap, A08_DifferentCommunicatorInvalidates) {
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto duplicateComm = Tpetra::getDefaultComm()->duplicate();
  const auto otherCommMap = Teuchos::rcp(new map_type(
      Tpetra::global_size_t(8), go_type(0), duplicateComm));
  vector_type x(map, 1);
  producerReady();

  // Even if the GIDs match, we conservatively require the original Comm
  // object, since the VMM arena was collectively established on that Comm.
  TEST_ASSERT(map->getComm().getRawPtr() !=
              otherCommMap->getComm().getRawPtr());
  x.replaceMap(otherCommMap);
  checkStorage(x, false, "A08 changed communicator", out, success);
  x.replaceMap(map);
  checkStorage(x, false, "A08 restored communicator", out, success);
  consumerComplete();
}

// Rank-mixed Map identities exercise the noncollective contract.  We do not
// run SpMV while different ranks retain different arena states.
TEUCHOS_UNIT_TEST(VMM_ReplaceMap, A09_MixedMapIdentityNoncollective) {
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto equivalent = distributedMap();
  const auto comm = Tpetra::getDefaultComm();
  TEST_ASSERT(map.getRawPtr() != equivalent.getRawPtr());
  TEST_ASSERT(map->isSameAs(*equivalent));

  for (int k = 0; k < 4; ++k) {
    // New native allocations are collective; both ranks enter each iteration.
    vector_type x(map, 1);
    const bool useOriginal = (comm->getRank() + k) % 2 == 0;
    producerReady();
    x.replaceMap(useOriginal ? map : equivalent);
    checkStorage(x, useOriginal && nativeExpected(),
                 "A09 mixed pointer identity", out, success);
    x.replaceMap(map);
    checkStorage(x, useOriginal && nativeExpected(),
                 "A09 mixed restore", out, success);
    consumerComplete();
  }
}

// Reproduce MueLu's level-2 pattern: after repartitioning, only rank 0
// has an Ac and executes replaceMap on an MV whose old communicator includes
// both ranks.  The other rank DOES NOT call replaceMap at all.  An embedded
// MPI reduction therefore hangs (or mismatches the next world collective).
TEUCHOS_UNIT_TEST(VMM_ReplaceMap, A10_SubsetOnlyCallsReplaceMap) {
  if (!twoRanks(out, success)) return;
  const auto original = distributedMap();
  const auto comm = Tpetra::getDefaultComm();
  const int rank = comm->getRank();
  vector_type x(original, 1);
  setHost(x, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(x);

  auto subcomm = comm->split(rank == 0 ? 0 : -1, rank);
  Teuchos::RCP<const map_type> subsetMap = Teuchos::null;
  if (rank == 0) {
    subsetMap = Teuchos::rcp(new map_type(Tpetra::global_size_t(4),
                                         go_type(0), subcomm));
  }
  producerReady();

  if (rank == 0) {
    x.replaceMap(subsetMap);
    checkStorage(x, false, "A10 subset rank remap", out, success);
    TEST_ASSERT(x.getLocalLength() == 4);
    x.replaceMap(original);
    checkStorage(x, false, "A10 subset rank restored", out, success);
    checkOriginalValues(x, "A10 restored values", out, success);
  } else {
    // This rank skips replaceMap altogether, just as MueLu does when Ac is
    // null.  The original arena remains valid locally until the owner dies.
    checkStorage(x, nativeExpected(), "A10 excluded rank untouched", out,
                 success);
  }
  // If rank 0 enters a collective inside replaceMap(), this barrier cannot
  // complete, and CTest's timeout catches the regression.
  consumerComplete();
}
