// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// P4 G06: invalid-MAP diagnostics in normal DistObject Import/Export flows.
// These deliberately invalid calls are TESTED ONLY when TPETRA_DEBUG=1 and
// both ranks have identical mismatches. Never run these cases with debug off.

#include "VmmP4Helpers.hpp"
#include "Tpetra_Details_Behavior.hpp"

#include <exception>
#include <string>

TEUCHOS_UNIT_TEST(VMM_Debug, G06_InvalidXDomain) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  TEST_ASSERT(Tpetra::Details::Behavior::debug("DistObject"));

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
  checkDebugException(caught, "source DistObject's Map must be the same",
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
