// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// P4 G06/G07: DistObject and VMM-only Map contract diagnostics.
// These deliberately invalid calls run only with TPETRA_DEBUG=1.  New G07
// cases use no Importer/Exporter and are intentionally skipped in classic-only
// jobs, where no Map check is guaranteed.

#include "VmmP4Helpers.hpp"
#include "Tpetra_Details_Behavior.hpp"

#include <exception>
#include <string>

TEUCHOS_UNIT_TEST(VMM_Debug, G06_InvalidXDomain) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  TEST_ASSERT(Tpetra::Details::Behavior::debug("DistObject"));
  TEST_ASSERT(Tpetra::Details::Behavior::debug("CrsMatrix"));

  const auto correct = distributedMap();
  const auto wrong = swappedRangeMap();
  TEST_ASSERT(!correct->isSameAs(*wrong));
  auto A = makeMatrix(correct, correct, correct, "P4_G06_InvalidX");
  TEST_ASSERT(!A->getGraph()->getImporter().is_null());
  vector_type X(wrong, 1), Y(correct, 1);
  setHost(X, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(X);
  Y.putScalar(-50.0);
  producerReady();

  std::string caught;
  try {
    A->apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0);
  } catch (const std::exception& e) {
    caught = e.what();
  }
  consumerComplete();
  // The direct candidate now reports the violated mathematical Map contract
  // before a VMM kernel or classic Import.  Classic-only still reports from
  // DistObject::beginTransfer(), as it did before M5.3.
  checkDebugException(caught, nativeExpected() ? "domain-X Map mismatch" :
                      "source DistObject's Map must be the same",
                      "G06_InvalidX", out, success);
}

TEUCHOS_UNIT_TEST(VMM_Debug, G06_InvalidYRange) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  TEST_ASSERT(Tpetra::Details::Behavior::debug("DistObject"));

  const auto row = distributedMap();
  const auto range = swappedRangeMap();
  TEST_ASSERT(!row->isSameAs(*range));
  auto A = makeMatrix(row, row, range, "P4_G06_InvalidY");
  TEST_ASSERT(!A->getGraph()->getExporter().is_null());
  vector_type X(row, 1), Y(row, 1);  // Incorrect Y range Map.
  setHost(X, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(X);
  Y.putScalar(-50.0);
  producerReady();

  std::string caught;
  try {
    A->apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0);
  } catch (const std::exception& e) {
    caught = e.what();
  }
  consumerComplete();
  checkDebugException(caught, "target DistObject's Map must be the same",
                      "G06_InvalidY", out, success);
}

TEUCHOS_UNIT_TEST(VMM_Debug, G06_EquivalentButDistinctValid) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  TEST_ASSERT(Tpetra::Details::Behavior::debug("DistObject"));

  const auto primary = distributedMap();
  const auto equivalent = distributedMap();
  TEST_ASSERT(primary.getRawPtr() != equivalent.getRawPtr());
  TEST_ASSERT(primary->isSameAs(*equivalent));

  auto A = makeMatrix(primary, primary, primary, "P4_G06_EquivalentValid");
  TEST_ASSERT(!A->getGraph()->getImporter().is_null());
  vector_type X(equivalent, 1), Y(primary, 1);
  prepareX(X);
  Y.putScalar(-50.0);
  const auto expectedPath = standardOrVmmPath(expectGeneralizedMaps());
  applyWithExpectedPath(*A, X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        expectedPath, "P4_G06_EquivalentValid", out, success);
  checkHost(Y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "G06 sameAs but distinct", out, success);
}

// G07 cases verify that VMM debug does not rely on an Importer or Exporter
// to discover wrong Maps.  All invalid Maps have the correct local length.
TEUCHOS_UNIT_TEST(VMM_Debug, G07_NoImporterInvalidX) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  TEST_ASSERT(Tpetra::Details::Behavior::debug("CrsMatrix"));
  if (!requireDirectVmmDebug("G07_NoImporterX")) return;

  const auto correct = distributedMap();
  const auto wrong = swappedRangeMap();
  const auto A = makeDiagonalWithDomain(correct, correct, "P4_G07_NoImporterX");
  TEST_ASSERT(A->getGraph()->getImporter().is_null());
  TEST_ASSERT(A->getGraph()->getExporter().is_null());
  TEST_EQUALITY(correct->getLocalNumElements(), wrong->getLocalNumElements());
  vector_type X(wrong, 1), Y(correct, 1);
  prepareX(X);
  Y.putScalar(-50.0);
  producerReady();
  std::string caught;
  try {
    A->apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0);
  } catch (const std::exception& e) {
    caught = e.what();
  }
  consumerComplete();
  checkDebugException(caught, "domain-X Map mismatch", "G07_NoImporterX",
                      out, success);
}

TEUCHOS_UNIT_TEST(VMM_Debug, G07_NoExporterInvalidY) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  TEST_ASSERT(Tpetra::Details::Behavior::debug("CrsMatrix"));
  if (!requireDirectVmmDebug("G07_NoExporterY")) return;

  const auto correct = distributedMap();
  const auto wrong = swappedRangeMap();
  const auto A = makeDiagonalWithDomain(correct, correct, "P4_G07_NoExporterY");
  TEST_ASSERT(A->getGraph()->getImporter().is_null());
  TEST_ASSERT(A->getGraph()->getExporter().is_null());
  TEST_EQUALITY(correct->getLocalNumElements(), wrong->getLocalNumElements());
  vector_type X(correct, 1), Y(wrong, 1);
  prepareX(X);
  Y.putScalar(-50.0);
  producerReady();
  std::string caught;
  try {
    A->apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0);
  } catch (const std::exception& e) {
    caught = e.what();
  }
  consumerComplete();
  checkDebugException(caught, "range-Y Map mismatch", "G07_NoExporterY",
                      out, success);
}

TEUCHOS_UNIT_TEST(VMM_Debug, G07_LocalGidOrder) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  TEST_ASSERT(Tpetra::Details::Behavior::debug("CrsMatrix"));
  if (!requireDirectVmmDebug("G07_LocalOrdering")) return;

  const auto correct = distributedMap();
  const auto wrong = explicitMap({3, 2, 1, 0}, {7, 6, 5, 4},
                                 Tpetra::global_size_t(8));
  const auto A = makeDiagonalWithDomain(correct, correct, "P4_G07_LocalOrdering");
  TEST_EQUALITY(correct->getLocalNumElements(), wrong->getLocalNumElements());
  TEST_ASSERT(!correct->isSameAs(*wrong));
  vector_type X(wrong, 1), Y(correct, 1);
  prepareX(X);
  Y.putScalar(-50.0);
  producerReady();
  std::string caught;
  try {
    A->apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0);
  } catch (const std::exception& e) {
    caught = e.what();
  }
  consumerComplete();
  checkDebugException(caught, "domain-X Map mismatch", "G07_LocalOrdering",
                      out, success);
}

TEUCHOS_UNIT_TEST(VMM_Debug, G07_RankLocalXMapMismatch) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  TEST_ASSERT(Tpetra::Details::Behavior::debug("CrsMatrix"));
  if (!requireDirectVmmDebug("G07_RankLocalX")) return;

  const auto correct = distributedMap();
  const auto wrong = swappedRangeMap();
  const auto A = makeDiagonalWithDomain(correct, correct, "P4_G07_RankLocalX");
  TEST_ASSERT(A->getGraph()->getImporter().is_null());
  TEST_ASSERT(A->getGraph()->getExporter().is_null());
  TEST_EQUALITY(correct->getLocalNumElements(), wrong->getLocalNumElements());
  // Only rank 1 passes a different Map.  Rank 0 passes the exact RCP used
  // by the matrix: isSameAs() would return early on rank 0 and may reduce
  // on rank 1 unless the VMM debug path handles mixed Map identity.
  const auto xMap = Tpetra::getDefaultComm()->getRank() == 0 ? correct : wrong;
  vector_type X(xMap, 1), Y(correct, 1);
  prepareX(X);
  Y.putScalar(-50.0);
  producerReady();
  std::string caught;
  try {
    A->apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0);
  } catch (const std::exception& e) {
    caught = e.what();
  }
  consumerComplete();
  checkDebugException(caught, "domain-X Map mismatch", "G07_RankLocalX",
                      out, success);
}

TEUCHOS_UNIT_TEST(VMM_Debug, G07_EquivalentNoImporterValid) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  TEST_ASSERT(Tpetra::Details::Behavior::debug("CrsMatrix"));
  const auto primary = distributedMap();
  const auto equivalent = distributedMap();
  TEST_ASSERT(primary.getRawPtr() != equivalent.getRawPtr());
  TEST_ASSERT(primary->isSameAs(*equivalent));
  const auto A = makeDiagonalWithDomain(primary, primary,
                                        "P4_G07_EquivalentNoImporter");
  TEST_ASSERT(A->getGraph()->getImporter().is_null());
  TEST_ASSERT(A->getGraph()->getExporter().is_null());
  vector_type X(equivalent, 1), Y(equivalent, 1);
  prepareX(X);
  Y.putScalar(-50.0);
  applyWithExpectedPath(*A, X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        standardOrVmmPath(expectGeneralizedMaps()),
                        "P4_G07_EquivalentNoImporter", out, success);
  checkHost(Y, [](go_type gid, std::size_t) { return 2.0 * input(gid); },
            "G07 equivalent no Importer", out, success);
}

TEUCHOS_UNIT_TEST(VMM_Debug, G07_RankMixedMapClasses) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  TEST_ASSERT(Tpetra::Details::Behavior::debug("CrsMatrix"));
  if (!requireDirectVmmDebug("G07_RankMixedClasses")) return;

  const auto correct = distributedMap();
  const auto equivalent = distributedMap();
  const auto wrong = swappedRangeMap();
  const auto A = makeDiagonalWithDomain(correct, correct,
                                        "P4_G07_RankMixedClasses");
  TEST_ASSERT(A->getGraph()->getImporter().is_null());
  TEST_ASSERT(A->getGraph()->getExporter().is_null());
  // Both ranks pass distinct Map objects.  On rank 0 it is another
  // uniform/contiguous Map; on rank 1 it is a nonuniform explicit Map.
  // Map::isSameAs() alone could return on rank 0 and reduce on rank 1.
  const auto xMap = Tpetra::getDefaultComm()->getRank() == 0 ? equivalent : wrong;
  vector_type X(xMap, 1), Y(correct, 1);
  prepareX(X);
  Y.putScalar(-50.0);
  producerReady();
  std::string caught;
  try {
    A->apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0);
  } catch (const std::exception& e) {
    caught = e.what();
  }
  consumerComplete();
  checkDebugException(caught, "domain-X Map mismatch", "G07_RankMixedClasses",
                      out, success);
}
