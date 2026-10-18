// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P0 only: a header-local NVSHMEM bootstrap and a symmetric-buffer smoke test.
// This file is NOT part of the Tpetra library. Include it only in the
// PerformanceCGSolve experiment when PERFORMANCE_CGSOLVE_NVSHMEM_P0 is enabled.
// The small GPU kernel is compiled separately in NvshmemP0.cu.

#ifndef PERFORMANCE_CGSOLVE_NVSHMEM_P0_HPP
#define PERFORMANCE_CGSOLVE_NVSHMEM_P0_HPP

#include "Tpetra_Access.hpp"
#include "Kokkos_Core.hpp"

#include <mpi.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <vector>

namespace NvshmemP0 {

inline void checkCuda(cudaError_t code, const char* action) {
  if (code != cudaSuccess) {
    std::ostringstream os;
    os << "NVSHMEM P0 " << action << ": " << cudaGetErrorString(code);
    throw std::runtime_error(os.str());
  }
}

inline void checkMpi(int code, const char* action) {
  if (code != MPI_SUCCESS) {
    std::ostringstream os;
    os << "NVSHMEM P0 " << action << " failed (MPI error " << code << ")";
    throw std::runtime_error(os.str());
  }
}

struct DeviceBinding {
  int device = -1;
  int localRank = -1;
  int localSize = 0;
  int visibleDevices = 0;
  std::string pciBusId;
};

// Call after MPI initialization and BEFORE Kokkos::initialize and NVSHMEM init.
// Handles both a shared CUDA_VISIBLE_DEVICES list and one GPU exposed per rank.
// Initial P0 rejects use of the same physical PCIe GPU by two ranks on a node.
inline DeviceBinding bindCudaDevice(MPI_Comm world) {
  MPI_Comm shared = MPI_COMM_NULL;
  checkMpi(MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                               &shared), "MPI_Comm_split_type");
  DeviceBinding b;
  checkMpi(MPI_Comm_rank(shared, &b.localRank), "local rank");
  checkMpi(MPI_Comm_size(shared, &b.localSize), "local size");
  checkCuda(cudaGetDeviceCount(&b.visibleDevices), "cudaGetDeviceCount");
  if (b.visibleDevices < 1) {
    throw std::runtime_error("NVSHMEM P0 requires a visible CUDA GPU");
  }
  b.device = b.localRank % b.visibleDevices;
  checkCuda(cudaSetDevice(b.device), "cudaSetDevice");

  std::array<char, 64> bus = {{0}};
  checkCuda(cudaDeviceGetPCIBusId(bus.data(), static_cast<int>(bus.size()),
                                  b.device), "cudaDeviceGetPCIBusId");
  b.pciBusId = bus.data();
  std::vector<char> allBusIds(static_cast<size_t>(b.localSize) * bus.size());
  checkMpi(MPI_Allgather(bus.data(), static_cast<int>(bus.size()), MPI_CHAR,
                          allBusIds.data(), static_cast<int>(bus.size()), MPI_CHAR,
                          shared), "allgather GPU identity");
  for (int i = 0; i < b.localSize; ++i) {
    for (int j = i + 1; j < b.localSize; ++j) {
      const char* left  = allBusIds.data() + static_cast<size_t>(i) * bus.size();
      const char* right = allBusIds.data() + static_cast<size_t>(j) * bus.size();
      if (std::string(left) == std::string(right)) {
        std::ostringstream os;
        os << "P0 needs one rank per physical GPU; node-local ranks " << i
           << " and " << j << " both selected " << left
           << ". Check GPU binding or CUDA_VISIBLE_DEVICES.";
        throw std::runtime_error(os.str());
      }
    }
  }
  checkMpi(MPI_Comm_free(&shared), "free local communicator");
  return b;
}

// Defined in NvshmemP0.cu; this is intentionally a genuine DEVICE-issued RMA.
cudaError_t launchScalarGet(const double* symmetricSource,
                            int sourcePe,
                            double* localDeviceResult,
                            cudaStream_t stream);

// MPI owns communicator lifetime. This session must be destroyed BEFORE
// Kokkos::finalize and MPI_Finalize. No MPI collectives occur in the destructor.
class Session {
 public:
  Session() = default;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  void initialize(MPI_Comm comm) {
    if (initialized_) throw std::logic_error("NVSHMEM P0 initialized twice");
    comm_ = comm;
    checkMpi(MPI_Comm_rank(comm_, &rank_), "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(comm_, &size_), "MPI_Comm_size");

    // Initializer macro is available in newer NVSHMEM, but support earlier
    // releases where zero initialization was the documented interface.
#ifdef NVSHMEMX_INIT_ATTR_INITIALIZER
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
#else
    nvshmemx_init_attr_t attr = {};
#endif
    MPI_Comm commHandle = comm_;
    attr.mpi_comm = &commHandle;
    const int rc = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    if (rc != 0) {
      std::ostringstream os;
      os << "nvshmemx_init_attr returned " << rc;
      throw std::runtime_error(os.str());
    }
    initialized_ = true;
    myPe_ = nvshmem_my_pe();
    nPes_ = nvshmem_n_pes();
    if (nPes_ != size_) throw std::runtime_error("MPI size and NVSHMEM PE count differ");
    peToRank_.assign(static_cast<size_t>(nPes_), -1);
    std::vector<int> peForRank(static_cast<size_t>(size_));
    checkMpi(MPI_Allgather(&myPe_, 1, MPI_INT, peForRank.data(), 1, MPI_INT,
                           comm_), "collect NVSHMEM PE mapping");
    for (int r = 0; r < size_; ++r) {
      const int pe = peForRank[static_cast<size_t>(r)];
      if (pe < 0 || pe >= nPes_ || peToRank_[static_cast<size_t>(pe)] != -1) {
        throw std::runtime_error("NVSHMEM PE mapping is not a permutation of MPI ranks");
      }
      peToRank_[static_cast<size_t>(pe)] = r;
    }
  }

  ~Session() {
    if (initialized_) nvshmem_finalize();
  }

  int rank() const { return rank_; }
  int pe() const { return myPe_; }
  int npes() const { return nPes_; }
  int rankForPe(int pe) const { return peToRank_.at(static_cast<size_t>(pe)); }
  MPI_Comm comm() const { return comm_; }

  void printConfiguration(const DeviceBinding& binding) const {
    int driver = 0, runtime = 0;
    checkCuda(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    checkCuda(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, binding.device), "cudaGetDeviceProperties");

    char processor[MPI_MAX_PROCESSOR_NAME] = {};
    int nameLength = 0;
    checkMpi(MPI_Get_processor_name(processor, &nameLength), "processor name");
    int specMajor = 0, specMinor = 0;
    nvshmem_info_get_version(&specMajor, &specMinor);
    char vendor[NVSHMEM_MAX_NAME_LEN] = {};
    nvshmem_info_get_name(vendor);

    // Deterministic startup output, outside all timing regions.
    for (int r = 0; r < size_; ++r) {
      checkMpi(MPI_Barrier(comm_), "logging barrier");
      if (rank_ == r) {
        std::cout << "[NVSHMEM P0] mpi_rank=" << rank_ << "/" << size_
                  << " nvshmem_pe=" << myPe_ << "/" << nPes_
                  << " host=" << processor
                  << " local_rank=" << binding.localRank << "/" << binding.localSize
                  << " cuda_device=" << binding.device
                  << " gpu='" << prop.name << "' bus=" << binding.pciBusId
                  << " cuda_driver=" << driver << " cuda_runtime=" << runtime
                  << " nvshmem_vendor='" << vendor << "'"
                  << " nvshmem_spec=" << specMajor << "." << specMinor
#ifdef NVSHMEM_VENDOR_MAJOR_VERSION
                  << " nvshmem_version=" << NVSHMEM_VENDOR_MAJOR_VERSION
#ifdef NVSHMEM_VENDOR_MINOR_VERSION
                  << "." << NVSHMEM_VENDOR_MINOR_VERSION
#endif
#ifdef NVSHMEM_VENDOR_PATCH_VERSION
                  << "." << NVSHMEM_VENDOR_PATCH_VERSION
#endif
#endif
                  << std::endl;
      }
    }
    checkMpi(MPI_Barrier(comm_), "final logging barrier");
    if (rank_ == 0) {
      char version[MPI_MAX_LIBRARY_VERSION_STRING] = {};
      int len = 0;
      checkMpi(MPI_Get_library_version(version, &len), "MPI library version");
      std::cout << "[NVSHMEM P0] MPI library: "
                << std::string(version, static_cast<size_t>(len)) << std::endl;
    }
  }

 private:
  MPI_Comm comm_ = MPI_COMM_NULL;
  bool initialized_ = false;
  int rank_ = -1;
  int size_ = 0;
  int myPe_ = -1;
  int nPes_ = 0;
  std::vector<int> peToRank_;
};

// This validates P0 plumbing. This is NOT a Tpetra Import benchmark yet.
// The input must be a rank-local, one-column double Tpetra Vector with a
// deterministic value (MPI rank + 1) in every owned slot.
template<class Vector>
int smokeTest(const Vector& source, MPI_Comm comm) {
  int rank = -1;
  int nRanks = 0;
  checkMpi(MPI_Comm_rank(comm, &rank), "MPI rank in smoke test");
  checkMpi(MPI_Comm_size(comm, &nRanks), "MPI size in smoke test");
  const int myPe = nvshmem_my_pe();
  const int nPes = nvshmem_n_pes();
  if (nRanks != nPes) throw std::runtime_error("MPI/NVSHMEM size mismatch");
  std::vector<int> peForRank(static_cast<size_t>(nRanks));
  std::vector<int> rankForPe(static_cast<size_t>(nPes), -1);
  checkMpi(MPI_Allgather(&myPe, 1, MPI_INT, peForRank.data(), 1,
                          MPI_INT, comm), "smoke PE map");
  for (int r = 0; r < nRanks; ++r) {
    const int pe = peForRank[static_cast<size_t>(r)];
    if (pe < 0 || pe >= nPes || rankForPe[static_cast<size_t>(pe)] != -1)
      throw std::runtime_error("inconsistent MPI-rank / NVSHMEM-PE mapping");
    rankForPe[static_cast<size_t>(pe)] = r;
  }
  static_assert(std::is_same<typename Vector::impl_scalar_type, double>::value,
                "P0 symmetric clone currently supports only native double");
  if (source.getNumVectors() != 1) {
    throw std::runtime_error("NVSHMEM P0 smoke requires exactly one vector column");
  }
  const size_t nLocal = source.getLocalLength();
  if (nLocal > static_cast<size_t>(std::numeric_limits<unsigned long long>::max())) {
    throw std::overflow_error("local source length exceeds MPI reduction type");
  }
  const unsigned long long localSize = static_cast<unsigned long long>(nLocal);
  unsigned long long maxSize = 0;
  checkMpi(MPI_Allreduce(&localSize, &maxSize, 1, MPI_UNSIGNED_LONG_LONG,
                          MPI_MAX, comm), "maximum local vector length");
  maxSize = std::max(1ULL, maxSize);
  if (maxSize > static_cast<unsigned long long>(SIZE_MAX / sizeof(double))) {
    throw std::overflow_error("NVSHMEM P0 symmetric allocation size overflow");
  }

  double* symmetric = static_cast<double*>(
      nvshmem_malloc(static_cast<size_t>(maxSize) * sizeof(double)));
  if (symmetric == nullptr) throw std::runtime_error("nvshmem_malloc failed");

  Kokkos::fence("NVSHMEM P0: finish source-vector initialization");
  auto sourceDevice = source.getLocalViewDevice(Tpetra::Access::ReadOnly);
  if (sourceDevice.extent(0) != nLocal || sourceDevice.extent(1) != 1) {
    throw std::runtime_error("unexpected source device view shape");
  }
  if (nLocal != 0) {
    checkCuda(cudaMemcpy(symmetric, sourceDevice.data(), nLocal * sizeof(double),
                         cudaMemcpyDeviceToDevice), "clone Tpetra Vector into symmetric source");
  } else {
    // All PEs allocate >= 1 element; an empty rank's slot 0 is a test sentinel.
    const double sentinel = static_cast<double>(rank + 1);
    checkCuda(cudaMemcpy(symmetric, &sentinel, sizeof(double), cudaMemcpyHostToDevice),
              "seed empty-rank sentinel");
  }
  checkCuda(cudaDeviceSynchronize(), "finish symmetric source publication");
  nvshmem_barrier_all();  // P0 readiness only; not a timed gather or a future fence design.

  double cloneFirst = 0;
  checkCuda(cudaMemcpy(&cloneFirst, symmetric, sizeof(double),
                       cudaMemcpyDeviceToHost), "validate symmetric clone");
  bool cloneOk = (cloneFirst == static_cast<double>(rank + 1));
  if (nLocal > 1) {
    double cloneLast = 0;
    checkCuda(cudaMemcpy(&cloneLast, symmetric + (nLocal - 1), sizeof(double),
                         cudaMemcpyDeviceToHost), "check last symmetric clone entry");
    cloneOk = cloneOk && (cloneLast == static_cast<double>(rank + 1));
  }

  const int sourcePe = (myPe + 1) % nPes;
  const double expected = static_cast<double>(rankForPe.at(static_cast<size_t>(sourcePe)) + 1);
  double* deviceResult = nullptr;
  checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceResult), sizeof(double)),
            "allocate local result");
  checkCuda(launchScalarGet(symmetric, sourcePe, deviceResult, nullptr),
            "launch GPU-initiated nvshmem_double_g");
  checkCuda(cudaDeviceSynchronize(), "complete GPU-initiated get");
  double observed = 0;
  checkCuda(cudaMemcpy(&observed, deviceResult, sizeof(double),
                       cudaMemcpyDeviceToHost), "read remote-get result");
  checkCuda(cudaFree(deviceResult), "free local result");

  // No PE may free its symmetric source while another PE still reads it.
  nvshmem_barrier_all();
  nvshmem_free(symmetric);

  const int failure = (!cloneOk || observed != expected) ? 1 : 0;
  int anyFailure = 0;
  checkMpi(MPI_Allreduce(&failure, &anyFailure, 1, MPI_INT,
                          MPI_MAX, comm), "smoke correctness reduction");
  std::cout << "[NVSHMEM P0] rank=" << rank
            << " source_pe=" << sourcePe
            << " local_elements=" << localSize
            << " symmetric_elements=" << maxSize
            << " got=" << observed << " expected=" << expected
            << " status=" << (failure == 0 ? "PASS" : "FAIL") << std::endl;
  if (rank == 0) {
    std::cout << "[NVSHMEM P0] "
              << (anyFailure == 0 ? "PASS" : "FAIL")
              << " (" << (nPes > 1 ? "remote GPU get" : "one-PE local smoke")
              << "; P1 plan/import and P2/P3 performance are not implemented)"
              << std::endl;
  }
  return anyFailure == 0 ? 0 : 1;
}

}  // namespace NvshmemP0

#endif  // PERFORMANCE_CGSOLVE_NVSHMEM_P0_HPP
