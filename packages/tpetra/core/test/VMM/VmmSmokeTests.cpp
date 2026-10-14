// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
//
// P0 smoke test.  More detailed SpMV, MultiVector, fallback and Map tests
// belong to the subsequent P1-P4 stages.

#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include "Teuchos_Array.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include <Kokkos_Core.hpp>
#include <limits>

TEUCHOS_UNIT_TEST(VMM_Smoke, AllRemoteTwoRanks) {
  using scalar_type = double;
  // Explicitly instantiate the CUDA Node: a CUDA-enabled Trilinos build
  // may still select Serial or OpenMP as its default Tpetra Node.
  using lo_type     = typename Tpetra::Map<>::local_ordinal_type;
  using go_type     = typename Tpetra::Map<>::global_ordinal_type;
  using node_type   = Tpetra::KokkosCompat::KokkosCudaWrapperNode;
  using map_type    = Tpetra::Map<lo_type, go_type, node_type>;
  using matrix_type = Tpetra::CrsMatrix<scalar_type, lo_type, go_type, node_type>;
  using mv_type     = Tpetra::MultiVector<scalar_type, lo_type, go_type, node_type>;

  const auto comm = Tpetra::getDefaultComm();
  TEST_EQUALITY(comm->getSize(), 2);
  if (comm->getSize() != 2) {
    return;  // P0's analytical fixture is deliberately specific to NP=2.
  }

  constexpr Tpetra::global_size_t globalRows = 8;
  const auto map = Teuchos::rcp(new map_type(globalRows, go_type(0), comm));
  TEST_EQUALITY(map->getLocalNumElements(), std::size_t(4));
  TEST_ASSERT(map->isContiguous());
  TEST_ASSERT(map->isOneToOne());

  matrix_type A(map, std::size_t(2));
  const auto localRows = map->getLocalNumElements();
  for (std::size_t lid = 0; lid < localRows; ++lid) {
    const go_type gid = map->getGlobalElement(static_cast<lo_type>(lid));
    const go_type remoteGid = (gid + go_type(4)) % go_type(8);
    Teuchos::Array<go_type> columns(2);
    Teuchos::Array<scalar_type> values(2);
    columns[0] = gid;
    columns[1] = remoteGid;
    values[0]  = 2.0;
    values[1]  = 3.0;
    A.insertGlobalValues(gid, columns(), values());
  }
  A.fillComplete(map, map);

  mv_type X(map, 1);
  mv_type Y(map, 1);
  {
    // Write X(lid)=gid(lid)+1, not rank*4+lid: this convention extends
    // naturally to the forthcoming noncontiguous-Map fixtures.
    auto xHost = X.getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (std::size_t lid = 0; lid < localRows; ++lid) {
      const go_type gid = map->getGlobalElement(static_cast<lo_type>(lid));
      xHost(lid, 0) = static_cast<scalar_type>(gid + 1);
    }
  }
  // Ensure the host writes reach the actual device view before peer GPUs
  // may read X.  This is caller-side synchronization, never a library change.
  {
    auto xDevice = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
    (void)xDevice;
  }
  Y.putScalar(-99.0);
  Kokkos::fence();
  comm->barrier();

  A.apply(X, Y, Teuchos::NO_TRANS, 1.0, 0.0);

  // Completion of peer reads is also a caller responsibility.
  Kokkos::fence();
  comm->barrier();
  {
    auto yHost = Y.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (std::size_t lid = 0; lid < localRows; ++lid) {
      const go_type gid = map->getGlobalElement(static_cast<lo_type>(lid));
      const go_type remoteGid = (gid + go_type(4)) % go_type(8);
      const scalar_type expected =
          2.0 * static_cast<scalar_type>(gid + 1) +
          3.0 * static_cast<scalar_type>(remoteGid + 1);
      TEST_FLOATING_EQUALITY(yHost(lid, 0), expected,
                             100.0 * std::numeric_limits<scalar_type>::epsilon());
    }
  }
  if (comm->getRank() == 0) {
    out << "P0 smoke fixture: 8 rows; rank 0 owns [0..3]; rank 1 owns [4..7];"
           " all rows have remote references.\n";
  }
}
