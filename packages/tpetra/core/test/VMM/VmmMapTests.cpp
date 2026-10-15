// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// P3 G01-G05: VMM Map contracts across development milestones.
// "generalized" tests M5.4's Map-identity-independent direct dispatch;
// older profiles preserve the earlier allocation and indexing expectations.

#include "VmmP3Helpers.hpp"

TEUCHOS_UNIT_TEST(VMM_Map, G01_Interleaved) {
  using namespace TpetraVmmP3;
  if (!twoRanks(out, success)) return;
  const auto map = interleavedMap();
  if (!checkFixtureGids(map, {0, 2, 4, 6}, {1, 3, 5, 7},
                        "G01", out, success)) return;
  TEST_ASSERT(!map->isContiguous());
  TEST_ASSERT(map->isOneToOne());

  const bool matrixGeneralized = expectMatrixGeneralized();
  const auto A = makeAndCheckMatrix(
      [&]() { return makeNeighborMatrix(map, "P3_G01_Interleaved"); },
      "P3_G01_Interleaved", matrixGeneralized ? "" : "map_not_contiguous",
      out, success);
  vector_type X(map, 1), Y(map, 1);
  checkStorage(X, nativeExpected() && expectGeneralizedMvAllocation(),
               "G01 X", out, success);
  checkStorage(Y, nativeExpected() && expectGeneralizedMvAllocation(),
               "G01 Y", out, success);
  prepareX(X);
  executeAndCheck(*A, X, Y, standardOrVmmPath(matrixGeneralized),
                  "P3_G01_Interleaved", false, out, success);
}

TEUCHOS_UNIT_TEST(VMM_Map, G02_PermutedLocalGids) {
  using namespace TpetraVmmP3;
  if (!twoRanks(out, success)) return;
  const auto map = permutedMap();
  if (!checkFixtureGids(map, {4, 0, 6, 2}, {7, 3, 1, 5},
                        "G02", out, success)) return;
  TEST_ASSERT(!map->isContiguous());
  TEST_ASSERT(map->isOneToOne());

  const bool matrixGeneralized = expectMatrixGeneralized();
  const auto A = makeAndCheckMatrix(
      [&]() { return makeNeighborMatrix(map, "P3_G02_Permuted"); },
      "P3_G02_Permuted", matrixGeneralized ? "" : "map_not_contiguous",
      out, success);
  vector_type X(map, 1), Y(map, 1);
  checkStorage(X, nativeExpected() && expectGeneralizedMvAllocation(),
               "G02 X", out, success);
  checkStorage(Y, nativeExpected() && expectGeneralizedMvAllocation(),
               "G02 Y", out, success);
  prepareX(X);
  executeAndCheck(*A, X, Y, standardOrVmmPath(matrixGeneralized),
                  "P3_G02_Permuted", false, out, success);
}

TEUCHOS_UNIT_TEST(VMM_Map, G03_EquivalentDistinctMaps) {
  using namespace TpetraVmmP3;
  if (!twoRanks(out, success)) return;
  auto primary = distributedMap();
  auto second = distributedMap();
  TEST_ASSERT(primary.getRawPtr() != second.getRawPtr());
  // Test-only collective equality: never add isSameAs() to the SpMV hot path.
  TEST_ASSERT(primary->isSameAs(*second));

  const auto A = makeAndCheckMatrix(
      [&]() { return makeNeighborMatrix(primary, "P3_G03_XDistinct"); },
      "P3_G03_XDistinct", "", out, success);
  vector_type primaryX(primary, 1), distinctX(second, 1);
  vector_type primaryY(primary, 1), distinctY(second, 1);
  checkStorage(primaryX, nativeExpected(), "G03 primary X", out, success);
  checkStorage(distinctX, nativeExpected(), "G03 distinct X", out, success);
  checkStorage(primaryY, nativeExpected(), "G03 primary Y", out, success);
  checkStorage(distinctY, nativeExpected(), "G03 distinct Y", out, success);
  prepareX(primaryX);
  prepareX(distinctX);

  const auto wanted = standardOrVmmPath(expectGeneralizedMaps());
  // A is constructed once with the primary domain/range Maps.  Change only
  // the RCP identity of X and/or Y while preserving exact distribution.
  // Distinct labels require a separate matrix label for each trace; changing
  // a label does not change the graph, allocation, or map semantics.
  A->setObjectLabel("P3_G03_XDistinct");
  executeAndCheck(*A, distinctX, primaryY, wanted, "P3_G03_XDistinct",
                  false, out, success);
  A->setObjectLabel("P3_G03_YDistinct");
  executeAndCheck(*A, primaryX, distinctY, wanted, "P3_G03_YDistinct",
                  false, out, success);
  A->setObjectLabel("P3_G03_BothDistinct");
  executeAndCheck(*A, distinctX, distinctY, wanted, "P3_G03_BothDistinct",
                  false, out, success);
}

// Combines M5.2's noncontiguous/locally permuted domain addressing with
// M5.4's distinct-but-equivalent Map dispatch.  Both independently
// constructed Maps must supply identical owner-local-index ordering.
TEUCHOS_UNIT_TEST(VMM_Map, G03_DistinctPermutedMaps) {
  using namespace TpetraVmmP3;
  if (!twoRanks(out, success)) return;
  const auto primary = permutedMap();
  const auto distinct = permutedMap();
  TEST_ASSERT(primary.getRawPtr() != distinct.getRawPtr());
  TEST_ASSERT(primary->isSameAs(*distinct));
  TEST_ASSERT(!primary->isContiguous());

  const bool matrixGeneralized = expectMatrixGeneralized();
  const auto A = makeAndCheckMatrix(
      [&]() { return makeNeighborMatrix(primary, "P3_G03_DistinctPermuted"); },
      "P3_G03_DistinctPermuted", matrixGeneralized ? "" : "map_not_contiguous",
      out, success);
  vector_type X(distinct, 1), Y(distinct, 1);
  checkStorage(X, nativeExpected() && expectGeneralizedMvAllocation(),
               "G03 permuted distinct X", out, success);
  checkStorage(Y, nativeExpected() && expectGeneralizedMvAllocation(),
               "G03 permuted distinct Y", out, success);
  prepareX(X);
  executeAndCheck(*A, X, Y, standardOrVmmPath(expectGeneralizedMaps()),
                  "P3_G03_DistinctPermuted", false, out, success);
}

TEUCHOS_UNIT_TEST(VMM_Map, G04_OverlappingMV) {
  using namespace TpetraVmmP3;
  if (!twoRanks(out, success)) return;
  const auto map = overlappingMap();
  if (!checkFixtureGids(map, {0, 1, 2, 3}, {3, 4, 5, 6, 7},
                        "G04", out, success)) return;
  TEST_ASSERT(!map->isOneToOne());
  TEST_EQUALITY(map->getGlobalNumElements(), Tpetra::global_size_t(9));

  // Only allocation is being tested. An overlapping MV is a valid storage
  // object, but it does not imply unique owner addresses for a VMM SpMV.
  vector_type X(map, 1);
  checkStorage(X, nativeExpected() && expectGeneralizedMvAllocation(),
               "G04 overlapping MV", out, success);
  prepareX(X);
  producerReady();
  checkHost(X, [](go_type gid, std::size_t) { return input(gid); },
            "G04 independently stored duplicate GID", out, success);
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_Map, G05_NonOneToOneDomain) {
  using namespace TpetraVmmP3;
  if (!twoRanks(out, success)) return;
  const auto row = distributedMap();
  const auto domain = overlappingMap();
  if (!checkFixtureGids(domain, {0, 1, 2, 3}, {3, 4, 5, 6, 7},
                        "G05", out, success)) return;
  TEST_ASSERT(!domain->isOneToOne());
  TEST_ASSERT(row->isOneToOne());

  // Deliberately avoid a global producer/consumer ambiguity: both local
  // copies of GID 3 receive the same X value, and the diagonal only reads
  // elements available locally. This isolates the matrix eligibility rule.
  // A one-to-one domain is an intended VMM *addressing* requirement, not a
  // MultiVector allocation requirement (compare G04).
  const bool matrixGeneralized = expectMatrixGeneralized();
  const auto A = makeAndCheckMatrix(
      [&]() { return makeDiagonalWithDomain(row, domain, "P3_G05_NonOneToOneDomain"); },
      "P3_G05_NonOneToOneDomain",
      matrixGeneralized ? "domain_map_not_one_to_one" : "map_not_contiguous",
      out, success);
  vector_type X(domain, 1), Y(row, 1);
  checkStorage(X, nativeExpected() && expectGeneralizedMvAllocation(),
               "G05 domain X", out, success);
  checkStorage(Y, nativeExpected(), "G05 range Y", out, success);
  prepareX(X);
  executeAndCheck(*A, X, Y, "classic", "P3_G05_NonOneToOneDomain",
                  true, out, success);
}
