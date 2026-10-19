// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P0 only. Header-only Kokkos/NVSHMEM smoke test used by PerformanceCGSolve.
// The actual Import packing comparison is a subsequent milestone.

#ifndef TPETRA_PERFORMANCE_NVSHMEM_P0_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P0_HPP

#include "Tpetra_Access.hpp"
#include "Kokkos_Core.hpp"
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace NvshmemP0 {

inline void checkMpi(int error, const char* where) {
  if (error != MPI_SUCCESS) {
    throw std::runtime_error(std::string("NVSHMEM P0: MPI error in ") + where);
  }
}

// Used before Kokkos initialization. Kokkos selects the CUDA GPU; there is
// intentionally no cudaSetDevice, cudaMalloc, cudaMemcpy or CUDA kernel launch.
inline int nodeLocalMpiRank(MPI_Comm world) {
  MPI_Comm local = MPI_COMM_NULL;
  checkMpi(MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                               &local), "MPI_Comm_split_type");
  int rank = -1;
  checkMpi(MPI_Comm_rank(local, &rank), "MPI_Comm_rank(shared)");
  checkMpi(MPI_Comm_free(&local), "MPI_Comm_free(shared)");
  return rank;
}

// NVSHMEM is initialized AFTER Kokkos selected/initialized its CUDA device,
// and finalized BEFORE Kokkos and MPI are finalized.
class Session {
 public:
  explicit Session(MPI_Comm comm) : comm_(comm) {
    checkMpi(MPI_Comm_rank(comm_, &rank_), "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(comm_, &numRanks_), "MPI_Comm_size");

    nvshmemx_init_attr_t attr = {};
    MPI_Comm world = comm_;
    attr.mpi_comm = &world;
    const int rc = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    if (rc != 0) {
      throw std::runtime_error("NVSHMEM P0: nvshmemx_init_attr failed: " +
                               std::to_string(rc));
    }
    initialized_ = true;

    pe_ = nvshmem_my_pe();
    const int nPes = nvshmem_n_pes();
    if (nPes != numRanks_) {
      throw std::runtime_error("NVSHMEM P0: MPI and NVSHMEM sizes differ");
    }
    std::vector<int> peForRank(static_cast<std::size_t>(numRanks_));
    rankForPe_.assign(static_cast<std::size_t>(nPes), -1);
    checkMpi(MPI_Allgather(&pe_, 1, MPI_INT, peForRank.data(), 1, MPI_INT,
                            comm_), "MPI_Allgather(PE)");
    for (int r = 0; r < numRanks_; ++r) {
      const int otherPe = peForRank[static_cast<std::size_t>(r)];
      if (otherPe < 0 || otherPe >= nPes ||
          rankForPe_[static_cast<std::size_t>(otherPe)] != -1) {
        throw std::runtime_error("NVSHMEM P0: invalid MPI rank / PE mapping");
      }
      rankForPe_[static_cast<std::size_t>(otherPe)] = r;
    }
  }

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  ~Session() {
    if (initialized_) nvshmem_finalize();
  }

  void describe(int selectedDeviceId) const {
    int major = 0, minor = 0;
    nvshmem_info_get_version(&major, &minor);
    for (int r = 0; r < numRanks_; ++r) {
      checkMpi(MPI_Barrier(comm_), "MPI_Barrier(P0 diagnostics)");
      if (rank_ == r) {
        std::cout << "[NVSHMEM P0] mpi_rank=" << rank_
                  << " nvshmem_pe=" << pe_
                  << " kokkos_selected_device=" << selectedDeviceId
                  << " nvshmem_api=" << major << "." << minor
                  << std::endl;
      }
    }
    checkMpi(MPI_Barrier(comm_), "MPI_Barrier(P0 diagnostics completion)");
  }

  MPI_Comm comm() const { return comm_; }
  int mpiRank() const { return rank_; }
  int pe() const { return pe_; }
  int numPes() const { return numRanks_; }
  int mpiRankForPe(int requestedPe) const {
    return rankForPe_.at(static_cast<std::size_t>(requestedPe));
  }

 private:
  MPI_Comm comm_ = MPI_COMM_NULL;
  int rank_ = -1;
  int numRanks_ = 0;
  int pe_ = -1;
  bool initialized_ = false;
  std::vector<int> rankForPe_;
};

// A pure-Kokkos P0 smoke test. The original Tpetra Vector remains ordinary
// Tpetra/Kokkos allocated memory; the NVSHMEM symmetric clone is an unmanaged
// Kokkos view. 'nvshmem_double_g' executes IN the Kokkos CUDA parallel_for.
//
// The input vector's local entries must equal MPI rank + 1. The test does
// not attempt to reproduce a Tpetra Import plan yet (that is P1/P3).
template<class Vector>
int smokeTest(const Vector& source, const Session& session) {
  using scalar_type = typename Vector::scalar_type;
  static_assert(std::is_same<scalar_type, double>::value,
                "NVSHMEM P0 supports Tpetra Vector<double> only");

  if (source.getNumVectors() != 1) {
    throw std::runtime_error("NVSHMEM P0 requires one vector column");
  }

  const std::size_t nLocal = source.getLocalLength();
  if (nLocal > static_cast<std::size_t>(
                   std::numeric_limits<unsigned long long>::max())) {
    throw std::overflow_error("NVSHMEM P0 local length is too large");
  }
  const unsigned long long length = static_cast<unsigned long long>(nLocal);
  unsigned long long maxLength = 0;
  checkMpi(MPI_Allreduce(&length, &maxLength, 1, MPI_UNSIGNED_LONG_LONG,
                          MPI_MAX, session.comm()), "MPI_Allreduce(max length)");
  maxLength = std::max(1ULL, maxLength); // symmetric allocation even on empty ranks
  if (maxLength > static_cast<unsigned long long>(
                      std::numeric_limits<std::size_t>::max() / sizeof(double))) {
    throw std::overflow_error("NVSHMEM P0 symmetric allocation would overflow");
  }

  // NVSHMEM symmetric allocations are collective; all PEs use maxLength.
  double* symmetricPointer = static_cast<double*>(
      nvshmem_malloc(static_cast<std::size_t>(maxLength) * sizeof(double)));
  if (symmetricPointer == nullptr) {
    throw std::runtime_error("NVSHMEM P0: nvshmem_malloc failed");
  }

  using SymmetricView = Kokkos::View<
      double*, Kokkos::CudaSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  SymmetricView symmetric(symmetricPointer, static_cast<std::size_t>(maxLength));
  auto owned = source.getLocalViewDevice(Tpetra::Access::ReadOnly);
  const double marker = static_cast<double>(session.mpiRank() + 1);

  // Clone the stock Tpetra input into its local symmetric allocation, using
  // Kokkos only. Rank-local length may differ between PEs. The first slot of
  // a zero-length rank is populated with a smoke-test sentinel.
  const std::size_t cloneCount = std::max(std::size_t(1), nLocal);
  Kokkos::parallel_for(
      "NvshmemP0::clone",
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(
          0, cloneCount),
      KOKKOS_LAMBDA(const std::size_t i) {
        symmetric(i) = i < nLocal ? owned(i, 0) : marker;
      });
  Kokkos::fence("NVSHMEM P0 finish Kokkos source clone");

  // This barrier is intentionally outside any future gather benchmark. It
  // protects producer readiness; it is NOT the intended hot-path algorithm.
  nvshmem_barrier_all();

  // Select the next PE in PE-number order and determine how much real X it
  // owns. MPI ranks and NVSHMEM PE numbers are not assumed identical.
  const int sourcePe = (session.pe() + 1) % session.numPes();
  const int sourceRank = session.mpiRankForPe(sourcePe);
  std::vector<unsigned long long> allLengths(
      static_cast<std::size_t>(session.numPes()));
  checkMpi(MPI_Allgather(&length, 1, MPI_UNSIGNED_LONG_LONG,
                          allLengths.data(), 1, MPI_UNSIGNED_LONG_LONG,
                          session.comm()), "MPI_Allgather(local length)");
  const unsigned long long remoteLength =
      allLengths.at(static_cast<std::size_t>(sourceRank));
  const std::size_t sampleCount = static_cast<std::size_t>(
      std::min(32ULL, std::max(1ULL, remoteLength)));

  Kokkos::View<double*, Kokkos::CudaSpace> gathered(
      "NVSHMEM P0 remotely gathered values", sampleCount);
  const double* symmetricBase = symmetricPointer;

  // GPU-initiated communication, launched and indexed entirely by Kokkos.
  Kokkos::parallel_for(
      "NvshmemP0::remote_get",
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(
          0, sampleCount),
      KOKKOS_LAMBDA(const std::size_t i) {
        gathered(i) = nvshmem_double_g(symmetricBase + i, sourcePe);
      });
  Kokkos::fence("NVSHMEM P0 complete GPU-initiated NVSHMEM gets");

  auto gatheredHost = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), gathered);
  const double expected = static_cast<double>(sourceRank + 1);
  int localFailed = 0;
  for (std::size_t i = 0; i < sampleCount; ++i) {
    if (gatheredHost(i) != expected) {
      localFailed = 1;
      std::cerr << "[NVSHMEM P0] MPI rank " << session.mpiRank()
                << " PE " << sourcePe << " entry " << i
                << " got " << gatheredHost(i)
                << " expected " << expected << "\n";
      break;
    }
  }

  // Do not free symmetric source memory until every PE's get has completed.
  nvshmem_barrier_all();
  nvshmem_free(symmetricPointer);

  int globalFailed = 0;
  checkMpi(MPI_Allreduce(&localFailed, &globalFailed, 1, MPI_INT, MPI_MAX,
                          session.comm()), "MPI_Allreduce(result)");
  std::cout << "[NVSHMEM P0] rank=" << session.mpiRank()
            << " pe=" << session.pe()
            << " sourcePe=" << sourcePe
            << " localLength=" << length
            << " paddedLength=" << maxLength
            << " sampledElements=" << sampleCount
            << " status=" << (localFailed ? "FAIL" : "PASS") << std::endl;
  if (session.mpiRank() == 0) {
    std::cout << "[NVSHMEM P0] " << (globalFailed ? "FAIL" : "PASS")
              << " (" << (session.numPes() > 1 ? "remote GPU get" : "one-PE local smoke")
              << "; no Import benchmark yet)" << std::endl;
  }
  return globalFailed ? EXIT_FAILURE : EXIT_SUCCESS;
}

} // namespace NvshmemP0

#endif // TPETRA_PERFORMANCE_NVSHMEM_P0_HPP

