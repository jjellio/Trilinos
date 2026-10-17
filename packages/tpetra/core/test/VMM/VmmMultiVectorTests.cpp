// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// P2 M01-M07: independent MultiVector allocation and lifetime tests.

#include "VmmP2Helpers.hpp"

#include <Kokkos_Core.hpp>

TEUCHOS_UNIT_TEST(VMM_MultiVector, M01_OneColumnStorage) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  auto map = distributedMap();
  vector_type x(map, 1);
  checkStorage(x, nativeExpected(), "M01 one column", out, success);
  TEST_EQUALITY(x.getNumVectors(), std::size_t(1));
  setHost(x, [](go_type gid, std::size_t) { return input(gid); });
  synchronizeToDevice(x);
  producerReady();
  checkHost(x, [](go_type gid, std::size_t) { return input(gid); },
            "M01 initialized", out, success);
}

TEUCHOS_UNIT_TEST(VMM_MultiVector, M02_ThreeColumnsStandard) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  auto map = distributedMap();
  vector_type x(map, 3);
  TEST_EQUALITY(x.getNumVectors(), std::size_t(3));
  checkStorage(x, false, "M02 three columns", out, success);
  setHost(x, [](go_type gid, std::size_t col) { return input(gid, col); });
  synchronizeToDevice(x);
  producerReady();
  checkHost(x, [](go_type gid, std::size_t col) { return input(gid, col); },
            "M02 initialized", out, success);
}

TEUCHOS_UNIT_TEST(VMM_MultiVector, M03_NativeFlagOff) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  if (Tpetra::Details::Behavior::experimentalVmmNativeMv()) {
    out << "M03: not applicable in native-MV-enabled scenario\n";
    return;
  }
  auto map = distributedMap();
  vector_type x(map, 1);
  checkStorage(x, false, "M03 native disabled", out, success);
}

TEUCHOS_UNIT_TEST(VMM_MultiVector, M04_GlobalFlagOff) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  if (Tpetra::Details::Behavior::experimentalVmm()) {
    out << "M04: not applicable in global-VMM-enabled scenario\n";
    return;
  }
  auto map = distributedMap();
  vector_type x(map, 1);
  checkStorage(x, false, "M04 global disabled", out, success);
}

TEUCHOS_UNIT_TEST(VMM_MultiVector, M05_CopyViewAndSubviewLifetime) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  auto map = distributedMap();

  Teuchos::RCP<vector_type> owner = Teuchos::rcp(new vector_type(map, 1));
  checkStorage(*owner, nativeExpected(), "M05 owner", out, success);
  setHost(*owner, [](go_type gid, std::size_t) { return input(gid) + 11.0; });
  synchronizeToDevice(*owner);
  producerReady();

  // Teuchos::View is shallow; Teuchos::Copy uses independent storage.
  Teuchos::RCP<vector_type> shallow =
      Teuchos::rcp(new vector_type(*owner, Teuchos::View));
  Teuchos::RCP<vector_type> deep =
      Teuchos::rcp(new vector_type(*owner, Teuchos::Copy));
  Teuchos::RCP<vector_type> sub = owner->subViewNonConst(Teuchos::Range1D(0, 0));
  checkStorage(*shallow, nativeExpected(), "M05 Teuchos::View", out, success);
  checkStorage(*deep, nativeExpected(), "M05 Teuchos::Copy", out, success);
  // A full-column, one-column subview has the same canonical VMM identity
  // as its parent (not merely a WrappedDualView external owner token).
  checkStorage(*sub, nativeExpected(), "M05 full-column subview", out, success);
  if (nativeExpected()) {
    TEST_ASSERT(shallow->getVmmGlobalPtr() == owner->getVmmGlobalPtr());
    TEST_ASSERT(sub->getVmmGlobalPtr() == owner->getVmmGlobalPtr());
    TEST_ASSERT(sub->getVmmLayout() == owner->getVmmLayout());
    TEST_ASSERT(deep->getVmmGlobalPtr() != owner->getVmmGlobalPtr());
  }
  owner = Teuchos::null;
  checkHost(*shallow, [](go_type gid, std::size_t) { return input(gid) + 11.0; },
            "M05 shallow after owner release", out, success);
  checkHost(*sub, [](go_type gid, std::size_t) { return input(gid) + 11.0; },
            "M05 subview after owner release", out, success);

  // Changing the shallow view changes the subview, but not the deep copy.
  setHost(*shallow, [](go_type gid, std::size_t) { return input(gid) + 41.0; });
  synchronizeToDevice(*shallow);
  producerReady();
  checkHost(*deep, [](go_type gid, std::size_t) { return input(gid) + 11.0; },
            "M05 independent deep copy", out, success);
  shallow = Teuchos::null;
  checkHost(*sub, [](go_type gid, std::size_t) { return input(gid) + 41.0; },
            "M05 subview retains arena", out, success);
  sub = Teuchos::null;
  deep = Teuchos::null;
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_MultiVector, M06_HostDeviceCoherence) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  auto map = distributedMap();
  vector_type x(map, 1);
  checkStorage(x, nativeExpected(), "M06", out, success);
  const int gidOffset = 4 * Tpetra::getDefaultComm()->getRank();
  {
    // Device writes are intentionally made *through the standard Tpetra
    // view*, not through a VMM pointer or a test-only publish operation.
    auto d = x.getLocalViewDevice(Tpetra::Access::OverwriteAll);
    Kokkos::parallel_for(
        "VMM_P2_M06_DeviceWrite",
        Kokkos::RangePolicy<Kokkos::Cuda>(0, static_cast<int>(x.getLocalLength())),
        KOKKOS_LAMBDA(const int lid) {
          d(lid, 0) = static_cast<scalar_type>(20 + gidOffset + lid);
        });
  }
  producerReady();
  checkHost(x, [](go_type gid, std::size_t) {
    return 20.0 + static_cast<scalar_type>(gid);
  }, "M06 device-to-host", out, success);

  setHost(x, [](go_type gid, std::size_t) {
    return 40.0 + static_cast<scalar_type>(gid);
  });
  synchronizeToDevice(x);
  {
    auto d = x.getLocalViewDevice(Tpetra::Access::ReadOnly);
    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
    for (std::size_t lid = 0; lid < x.getLocalLength(); ++lid) {
      const go_type gid = map->getGlobalElement(static_cast<lo_type>(lid));
      const scalar_type expected = 40.0 + static_cast<scalar_type>(gid);
      if (std::abs(mirror(lid, 0) - expected) > 1.0e-12) {
        success = false;
        out << "M06 host-to-device mismatch gid=" << gid
            << " actual=" << mirror(lid, 0)
            << " expected=" << expected << '\n';
      }
    }
  }
  consumerComplete();
}

TEUCHOS_UNIT_TEST(VMM_MultiVector, M07_RepeatedAllocationAndRelease) {
  using namespace TpetraVmmP2;
  if (!twoRanks(out, success)) return;
  auto map = distributedMap();

  // Exercise collective allocator order, distinct concurrent arenas, and
  // destruction/reuse.  Deliberately small: intended for correctness, not
  // measuring allocator throughput.
  for (int iteration = 0; iteration < 12; ++iteration) {
    auto a = Teuchos::rcp(new vector_type(map, 1));
    auto b = Teuchos::rcp(new vector_type(map, 1));
    checkStorage(*a, nativeExpected(), "M07 a", out, success);
    checkStorage(*b, nativeExpected(), "M07 b", out, success);
    if (nativeExpected()) {
      TEST_ASSERT(a->getVmmGlobalPtr() != b->getVmmGlobalPtr());
    }
    setHost(*a, [iteration](go_type gid, std::size_t) {
      return static_cast<scalar_type>(100 * iteration + gid);
    });
    setHost(*b, [iteration](go_type gid, std::size_t) {
      return static_cast<scalar_type>(100 * iteration + gid + 50);
    });
    synchronizeToDevice(*a);
    synchronizeToDevice(*b);
    producerReady();
    checkHost(*a, [iteration](go_type gid, std::size_t) {
      return static_cast<scalar_type>(100 * iteration + gid);
    }, "M07 a", out, success);
    // First allocation is destroyed while the second remains live.
    a = Teuchos::null;
    checkHost(*b, [iteration](go_type gid, std::size_t) {
      return static_cast<scalar_type>(100 * iteration + gid + 50);
    }, "M07 b survives a", out, success);
    consumerComplete();
    b = Teuchos::null;
  }
}
