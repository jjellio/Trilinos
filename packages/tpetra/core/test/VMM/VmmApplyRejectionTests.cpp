// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
//
// Runtime diagnostics for classic CrsMatrix::apply fallbacks.  These tests
// exercise the actual apply interface on two ranks and capture each rank's
// stdout around the apply call (not around the surrounding MPI epochs).

#include "VmmP2Helpers.hpp"

#include <string>

namespace TpetraVmmApplyRejection {

using namespace TpetraVmmP2;

// Expected reason when an otherwise eligible VMM dispatch is suppressed by
// process-wide Behavior flags.  In full VMM mode, return the local predicate.
inline const char* expectedReason(const char* fullVmmReason) {
  if (!Tpetra::Details::Behavior::experimentalVmm()) return "vmm_disabled";
  if (!Tpetra::Details::Behavior::experimentalVmmNativeMv())
    return "native_mv_disabled";
  return fullVmmReason;
}

inline void applyAndCheckTrace(const matrix_type& A, const vector_type& X,
                               vector_type& Y, Teuchos::ETransp mode,
                               scalar_type alpha, scalar_type beta,
                               const char* expectedPath,
                               const char* expectedSkipReason,
                               std::ostream& out, bool& success) {
  producerReady();
  std::string captured;
  {
    TpetraVmmP1::ScopedApplyOutput capture;
    A.apply(X, Y, mode, alpha, beta);
    captured = capture.str();
  }
  std::cout << captured << std::flush;
  consumerComplete();

  const bool stats = Tpetra::Details::Behavior::experimentalVmmStatistics();
  const int rank = Tpetra::getDefaultComm()->getRank();
  const std::string applyTag = "[TPETRA_MATRIX] event=apply path=";
  const std::string skipTag = "[TPETRA_MATRIX] event=vmmSkip phase=apply";
  const std::size_t applyEvents = TpetraVmmP1::occurrences(captured, applyTag);
  const std::size_t skipEvents = TpetraVmmP1::occurrences(captured, skipTag);

  const bool transpose = mode != Teuchos::NO_TRANS;
  const std::size_t expectedApplyEvents = (stats && !transpose) ? 1 : 0;
  const std::size_t expectedSkipEvents =
      (stats && expectedSkipReason != nullptr) ? 1 : 0;
  bool valid = applyEvents == expectedApplyEvents &&
               skipEvents == expectedSkipEvents;

  if (stats && !transpose) {
    const std::string expected =
        applyTag + expectedPath + " rank=" + std::to_string(rank) +
        " label=\"" + A.getObjectLabel() + "\"";
    valid = valid && captured.find(expected) != std::string::npos;
  }
  if (stats && expectedSkipReason != nullptr) {
    const std::string expected = skipTag +
        " reason=" + expectedSkipReason + " rank=" + std::to_string(rank);
    valid = valid && captured.find(expected) != std::string::npos &&
            captured.find("label=\"" + A.getObjectLabel() + "\"") !=
                std::string::npos;
    if (!transpose) {
      // These fields allow us to distinguish actual lost VMM storage from
      // an unrelated matrix or output restriction in a MueLu hierarchy.
      valid = valid && captured.find(" x_object=") != std::string::npos &&
              captured.find(" x_vmm=") != std::string::npos &&
              captured.find(" y_object=") != std::string::npos &&
              captured.find(" y_vmm=") != std::string::npos;
    } else {
      valid = valid && captured.find(" mode=trans") != std::string::npos;
    }
  }
  if (!valid) {
    success = false;
    out << "Apply rejection diagnostics mismatch rank=" << rank
        << " matrix=" << A.getObjectLabel()
        << " expectedPath=" << expectedPath
        << " expectedReason=" << (expectedSkipReason ? expectedSkipReason : "none")
        << " stats=" << stats
        << " applyEvents=" << applyEvents
        << " skipEvents=" << skipEvents << "\n"
        << "Captured output:\n" << captured << '\n';
  }
}

}  // namespace TpetraVmmApplyRejection

TEUCHOS_UNIT_TEST(VMM_ApplyRejection, R01_OrdinaryX) {
  using namespace TpetraVmmApplyRejection;
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto A = makeMatrix(map, map, map, "ApplyReject_OrdinaryX");

  // Deliberately non-VMM input even in native mode.
  vector_type parent(map, 3);
  setHost(parent, [](go_type gid, std::size_t col) { return input(gid, col); });
  synchronizeToDevice(parent);
  const auto X = parent.subViewNonConst(Teuchos::Range1D(0, 0));
  vector_type Y(map, 1);
  Y.putScalar(-7.0);
  applyAndCheckTrace(*A, *X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                     "classic", expectedReason("x_not_vmm_backed"), out, success);
  checkHost(Y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "R01 ordinary input", out, success);
}

TEUCHOS_UNIT_TEST(VMM_ApplyRejection, R02_Exporter) {
  using namespace TpetraVmmApplyRejection;
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto rangeMap = swappedRangeMap();
  const auto A = makeMatrix(map, map, rangeMap, "ApplyReject_Exporter");
  TEST_ASSERT(!A->getGraph()->getExporter().is_null());
  vector_type X(map, 1), Y(rangeMap, 1);
  setHost(X, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(X);
  Y.putScalar(-7.0);
  applyAndCheckTrace(*A, X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                     "classic", expectedReason("has_exporter"), out, success);
  checkHost(Y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "R02 exporter", out, success);
}

TEUCHOS_UNIT_TEST(VMM_ApplyRejection, R03_DirectVmmNoSkip) {
  using namespace TpetraVmmApplyRejection;
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto A = makeMatrix(map, map, map, "ApplyReject_Direct");
  vector_type X(map, 1), Y(map, 1);
  setHost(X, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(X);
  Y.putScalar(-7.0);
  const bool direct = nativeExpected();
  applyAndCheckTrace(*A, X, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                     direct ? "vmm" : "classic",
                     direct ? nullptr : expectedReason("unclassified_rejection"),
                     out, success);
  checkHost(Y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "R03 direct", out, success);
}

TEUCHOS_UNIT_TEST(VMM_ApplyRejection, R04_Transpose) {
  using namespace TpetraVmmApplyRejection;
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto A = makeMatrix(map, map, map, "ApplyReject_Transpose", true);
  vector_type X(map, 1), Y(map, 1);
  setHost(X, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(X);
  Y.putScalar(-7.0);
  applyAndCheckTrace(*A, X, Y, Teuchos::TRANS, 1.0, 0.0,
                     "none", "transpose_unsupported", out, success);
  checkHost(Y, [](go_type gid, std::size_t) {
    const go_type remote = (gid + 4) % 8;
    return 2.0 * input(gid) +
           (3.0 + static_cast<scalar_type>(remote)) * input(remote);
  }, "R04 transpose", out, success);
}

TEUCHOS_UNIT_TEST(VMM_ApplyRejection, R05_AlphaZeroDoesNotReject) {
  using namespace TpetraVmmApplyRejection;
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const auto A = makeMatrix(map, map, map, "ApplyReject_AlphaZero");
  vector_type X(map, 1), Y(map, 1);
  setHost(Y, [](go_type gid, std::size_t) {
    return 7.0 + static_cast<scalar_type>(gid);
  });
  synchronizeToDevice(Y);
  applyAndCheckTrace(*A, X, Y, Teuchos::NO_TRANS, 0.0, -2.0,
                     "alpha_zero", nullptr, out, success);
  checkHost(Y, [](go_type gid, std::size_t) {
    return -2.0 * (7.0 + static_cast<scalar_type>(gid));
  }, "R05 alpha zero", out, success);
}
