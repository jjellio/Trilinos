// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// Full-column view preservation for canonical one-column VMM MultiVectors.
// Tests use ordinary public Tpetra APIs; no new MPI collective is permitted
// in MultiVector::subView().

#include "VmmP2Helpers.hpp"
#include "VmmP4Helpers.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_Range1D.hpp"

#include <cstddef>
#include <string>

namespace TpetraVmmSubview {

using namespace TpetraVmmP2;

inline Teuchos::Array<std::size_t> columnZero() {
  Teuchos::Array<std::size_t> cols(1);
  cols[0] = 0;
  return cols;
}

inline void checkSameAllocation(const vector_type& parent,
                                const vector_type& view,
                                const std::string& label,
                                std::ostream& out, bool& success) {
  checkStorage(view, nativeExpected(), label, out, success);
  TEST_EQUALITY(view.getNumVectors(), std::size_t(1));
  TEST_EQUALITY(view.getLocalLength(), parent.getLocalLength());
  TEST_ASSERT(view.isConstantStride());
  TEST_ASSERT(view.getMap().getRawPtr() == parent.getMap().getRawPtr());
  TEST_ASSERT(view.getVmmGlobalPtr() == parent.getVmmGlobalPtr());
  TEST_ASSERT(view.getVmmLayout() == parent.getVmmLayout());
  const auto parentDevice = parent.getLocalViewDevice(Tpetra::Access::ReadOnly);
  const auto viewDevice = view.getLocalViewDevice(Tpetra::Access::ReadOnly);
  if (parent.getLocalLength() != 0) {
    TEST_ASSERT(parentDevice.data() == viewDevice.data());
  }
}

}  // namespace TpetraVmmSubview

TEUCHOS_UNIT_TEST(VMM_Subview, S01_RangeConstAndNonConst) {
  using namespace TpetraVmmSubview;
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  vector_type owner(map, 1);
  setHost(owner, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(owner);

  const auto readOnly = owner.subView(Teuchos::Range1D(0, 0));
  auto writable = owner.subViewNonConst(Teuchos::Range1D(0, 0));
  checkSameAllocation(owner, *readOnly, "S01 const Range1D", out, success);
  checkSameAllocation(owner, *writable, "S01 nonconst Range1D", out, success);

  producerReady();
  checkHost(*writable, [](go_type gid, std::size_t) { return input(gid); },
            "S01 range values", out, success);
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_Subview, S02_ArrayAndNestedViews) {
  using namespace TpetraVmmSubview;
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  vector_type owner(map, 1);
  const auto cols = columnZero();
  auto mutableArray = owner.subViewNonConst(cols());
  const auto constArray = owner.subView(cols());
  auto nested = mutableArray->subViewNonConst(Teuchos::Range1D(0, 0));

  checkSameAllocation(owner, *mutableArray, "S02 ArrayView nonconst", out, success);
  checkSameAllocation(owner, *constArray, "S02 ArrayView const", out, success);
  checkSameAllocation(owner, *nested, "S02 nested Range1D", out, success);

  setHost(*nested, [](go_type gid, std::size_t) { return 10.0 + input(gid); });
  synchronizeToDevice(*nested);
  producerReady();
  checkHost(owner, [](go_type gid, std::size_t) { return 10.0 + input(gid); },
            "S02 write through nested view", out, success);
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_Subview, S03_ViewOutlivesOriginalMV) {
  using namespace TpetraVmmSubview;
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  auto original = Teuchos::rcp(new vector_type(map, 1));
  setHost(*original, [](go_type gid, std::size_t) { return input(gid) + 20.0; });
  synchronizeToDevice(*original);

  auto first = original->subViewNonConst(Teuchos::Range1D(0, 0));
  const auto nested = first->subView(columnZero()());
  const auto global = original->getVmmGlobalPtr();
  const auto layout = original->getVmmLayout();
  checkSameAllocation(*original, *first, "S03 first", out, success);
  checkSameAllocation(*original, *nested, "S03 nested", out, success);

  original = Teuchos::null;
  first = Teuchos::null;
  checkStorage(*nested, nativeExpected(), "S03 after owner destruction", out, success);
  TEST_ASSERT(nested->getVmmGlobalPtr() == global);
  TEST_ASSERT(nested->getVmmLayout() == layout);
  producerReady();
  const auto h = nested->getLocalViewHost(Tpetra::Access::ReadOnly);
  for (std::size_t lid = 0; lid < nested->getLocalLength(); ++lid) {
    const go_type gid = map->getGlobalElement(static_cast<lo_type>(lid));
    const scalar_type expected = input(gid) + 20.0;
    if (std::abs(h(lid, 0) - expected) > 1.0e-12) {
      success = false;
      out << "S03 lifetime value mismatch gid=" << gid << '\n';
    }
  }
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_Subview, S04_AliasingVsDeepCopy) {
  using namespace TpetraVmmSubview;
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  vector_type owner(map, 1);
  setHost(owner, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(owner);

  auto view = owner.subViewNonConst(Teuchos::Range1D(0, 0));
  auto deep = owner.subCopy(Teuchos::Range1D(0, 0));
  checkSameAllocation(owner, *view, "S04 view", out, success);
  checkStorage(*deep, nativeExpected(), "S04 deep copy", out, success);
  // Device views must go out of scope before setHost() requests host access.
  // In the classic DualView path, WrappedDualView rejects host access while
  // a device view remains alive, even when that view is read-only.
  {
    const auto ownerDevice = owner.getLocalViewDevice(Tpetra::Access::ReadOnly);
    const auto deepDevice = deep->getLocalViewDevice(Tpetra::Access::ReadOnly);
    TEST_ASSERT(ownerDevice.data() != deepDevice.data());
    if (nativeExpected()) {
      TEST_ASSERT(owner.getVmmGlobalPtr() != deep->getVmmGlobalPtr());
    }
  }

  setHost(*view, [](go_type gid, std::size_t) { return 100.0 + input(gid); });
  synchronizeToDevice(*view);
  producerReady();
  checkHost(owner, [](go_type gid, std::size_t) { return 100.0 + input(gid); },
            "S04 alias reflects modification", out, success);
  checkHost(*deep, [](go_type gid, std::size_t) { return input(gid); },
            "S04 deep copy remains independent", out, success);
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_Subview, S05_NoncanonicalViewsStayStandard) {
  using namespace TpetraVmmSubview;
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  // A two-column MV is deliberately not VMM-backed.  Selecting one of its
  // columns must not invent an arena or advertise a canonical global pointer.
  vector_type multi(map, 2);
  auto zero = multi.subViewNonConst(Teuchos::Range1D(0, 0));
  auto one = multi.subViewNonConst(Teuchos::Range1D(1, 1));
  checkStorage(multi, false, "S05 multi-column parent", out, success);
  checkStorage(*zero, false, "S05 column zero of standard MV", out, success);
  checkStorage(*one, false, "S05 shifted column of standard MV", out, success);

  // Row slicing genuinely changes the distributed logical layout, even when
  // the source is VMM-backed.  It may alias the bytes but must not expose the
  // canonical VMM descriptor as a full input vector.
  vector_type native(map, 1);
  Teuchos::Array<go_type> gids(3);
  for (int i = 0; i < 3; ++i) {
    gids[static_cast<std::size_t>(i)] =
        map->getGlobalElement(static_cast<lo_type>(i + 1));
  }
  const auto shorter = Teuchos::rcp(new map_type(
      Tpetra::global_size_t(6), gids(), go_type(0), Tpetra::getDefaultComm()));
  auto rowSlice = native.offsetViewNonConst(shorter, std::size_t(1));
  checkStorage(*rowSlice, false, "S05 row-offset view", out, success);
  checkStorage(native, nativeExpected(), "S05 original remains canonical", out, success);
}

TEUCHOS_UNIT_TEST(VMM_Subview, S06_ViewsDispatchDirectSpMV) {
  using namespace TpetraVmmSubview;
  if (!twoRanks(out, success)) return;
  const auto map = distributedMap();
  const std::string label = "S06_VmmSubview_AllRemote";
  auto A = makeMatrix(map, map, map, label);
  auto owner = Teuchos::rcp(new vector_type(map, 1));
  vector_type Y(map, 1);
  setHost(*owner, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(*owner);

  const auto range = owner->subView(Teuchos::Range1D(0, 0));
  const auto array = owner->subView(columnZero()());
  checkSameAllocation(*owner, *range, "S06 range input", out, success);
  checkSameAllocation(*owner, *array, "S06 array input", out, success);
  const std::string path = nativeExpected() ? "vmm" : "classic";
  owner = Teuchos::null;

  // Verify dispatch (when statistics are on) and the independently known
  // result, including cross-rank dependencies, for each subview overload.
  applyWithExpectedPath(*A, *range, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        path, label, out, success);
  checkHost(Y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "S06 Range1D apply", out, success);
  applyWithExpectedPath(*A, *array, Y, Teuchos::NO_TRANS, 1.0, 0.0,
                        path, label, out, success);
  checkHost(Y, [](go_type gid, std::size_t) { return allRemoteResult(gid); },
            "S06 ArrayView apply", out, success);
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_Subview, S07_ZeroLocalRowsRetainArena) {
  using namespace TpetraVmmSubview;
  if (!twoRanks(out, success)) return;
  const auto map = TpetraVmmP4::emptyRankMap();
  vector_type owner(map, 1);
  const auto range = owner.subView(Teuchos::Range1D(0, 0));
  const auto array = owner.subView(columnZero()());

  checkSameAllocation(owner, *range, "S07 zero-row Range1D", out, success);
  checkSameAllocation(owner, *array, "S07 zero-row ArrayView", out, success);
  if (Tpetra::getDefaultComm()->getRank() == 1) {
    TEST_EQUALITY(owner.getLocalLength(), std::size_t(0));
  }
  // Nonzero-length ranks must alias the same local bytes; empty ranks have
  // no meaningful local data pointer, but retain the same global arena.
  producerReady();
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_Subview, S08_OnlyOneRankTakesView) {
  using namespace TpetraVmmSubview;
  if (!twoRanks(out, success)) return;
  const auto comm = Tpetra::getDefaultComm();
  const auto map = distributedMap();
  vector_type owner(map, 1);

  // Reproduce rank-subset control flow: unlike the VMM allocation above,
  // merely forming a view of an existing arena must not add communication
  // in this test's TPETRA_DEBUG=0 configuration.
  if (comm->getRank() == 0) {
    auto view = owner.subViewNonConst(Teuchos::Range1D(0, 0));
    checkSameAllocation(owner, *view, "S08 rank-subset view", out, success);
  }
  comm->barrier();
}
