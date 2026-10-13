// Experimental, allocation-free layout for a distributed CUDA VMM vector.
// All ranks compute the same padded byte offsets from rank counts, CUDA
// granularities, scalar size and IPC mode; no physical or virtual memory is
// allocated here. The arena and (eventually) CrsMatrix use this same layout.
#ifndef TPETRA_EXPERIMENT_VMM_LAYOUT_HPP
#define TPETRA_EXPERIMENT_VMM_LAYOUT_HPP

#ifdef HAVE_TPETRA_INST_CUDA

#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"
#ifdef HAVE_MPI
#include "Teuchos_DefaultMpiComm.hpp"
#include <mpi.h>
#endif
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace VmmExperiment {

// Value object; copying shares the same Teuchos communicator reference.
// No process-global communicator or rank is assumed.
struct VmmComm {
  explicit VmmComm(Teuchos::RCP<const Teuchos::Comm<int>> communicator)
      : comm(std::move(communicator)) {
    if (comm.is_null()) {
      throw std::invalid_argument("VMM context requires a non-null Teuchos communicator");
    }
#ifdef HAVE_MPI
    // Fail early rather than silently substituting a process-wide MPI comm for a
    // SerialComm or an unsupported Comm implementation.
    (void)Teuchos::getRawMpiComm(*comm);
#endif
  }

  int rank() const { return comm->getRank(); }
  int size() const { return comm->getSize(); }

#ifdef HAVE_MPI
  // Get the raw handle on demand: its lifetime belongs to the RCP above.
  MPI_Comm mpi() const { return Teuchos::getRawMpiComm(*comm); }
#endif

  Teuchos::RCP<const Teuchos::Comm<int>> comm;
};

[[noreturn]] inline void abortWithMessage(const VmmComm& comm,
                                          const std::string& msg) {
  std::fprintf(stderr, "rank %d: %s\n", comm.rank(), msg.c_str());
  std::fflush(stderr);
#ifdef HAVE_MPI
  MPI_Abort(comm.mpi(), 1);
#endif
  std::abort();
}

inline void checkCudaDriver(const VmmComm& comm, CUresult err,
                            const char* expr, const char* file, int line) {
  if (err == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* text = nullptr;
  cuGetErrorName(err, &name);
  cuGetErrorString(err, &text);
  char buf[2048];
  std::snprintf(buf, sizeof(buf), "%s:%d: %s failed: %s (%s)", file, line, expr,
                name ? name : "CUDA_ERROR", text ? text : "unknown");
  abortWithMessage(comm, buf);
}

inline void checkCudaRuntime(const VmmComm& comm, cudaError_t err,
                             const char* expr, const char* file, int line) {
  if (err == cudaSuccess) return;
  char buf[2048];
  std::snprintf(buf, sizeof(buf), "%s:%d: %s failed: %s", file, line, expr,
                cudaGetErrorString(err));
  abortWithMessage(comm, buf);
}

#define VMM_CU_CHECK(comm, call) \
    ::VmmExperiment::checkCudaDriver((comm), (call), #call, __FILE__, __LINE__)
#define VMM_CUDART_CHECK(comm, call) \
    ::VmmExperiment::checkCudaRuntime((comm), (call), #call, __FILE__, __LINE__)

enum class IpcMode { Posix, Fabric };

inline IpcMode parseIpcMode(const std::string& text) {
  if (text == "posix") return IpcMode::Posix;
  if (text == "fabric") return IpcMode::Fabric;
  throw std::runtime_error("--vmm-ipc must be 'posix' or 'fabric'");
}

// The physical layout is parameterized by the *stored* scalar type (in Tpetra,
// impl_scalar_type). CUDA addresses are bytes; CRS addressing is in elements.
// Unlike Tpetra's LO, all prefix/element offsets here are 64-bit.
template <class Scalar>
class VmmLayout {
 public:
  // Collective metadata discovery, but NO cuMemCreate/AddressReserve/Map.
  // Both the matrix and the vector allocator can call this independently.
  static VmmLayout compute(std::size_t logicalLocalCount, IpcMode mode,
                           const VmmComm& comm) {
#ifndef HAVE_MPI
    (void)logicalLocalCount;
    (void)mode;
    (void)comm;
    throw std::runtime_error("VMM distributed layout requires MPI in this experiment");
#else
    VMM_CU_CHECK(comm, cuInit(0));
    int runtimeDev = 0;
    VMM_CUDART_CHECK(comm, cudaGetDevice(&runtimeDev));
    CUdevice device = 0;
    VMM_CU_CHECK(comm, cuDeviceGet(&device, runtimeDev));

    CUmemAllocationProp prop {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = requestedHandleType(mode);

    std::size_t localGran = 0;
    VMM_CU_CHECK(comm, cuMemGetAllocationGranularity(
        &localGran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    // Exchange *logical* sizes and native granularities. The padded sizes,
    // offsets, and total are computed identically on every rank below.
    const auto localCountUll = static_cast<unsigned long long>(logicalLocalCount);
    const auto localGranUll = static_cast<unsigned long long>(localGran);
    std::vector<unsigned long long> counts(comm.size());
    std::vector<unsigned long long> granularities(comm.size());
    MPI_Allgather(&localCountUll, 1, MPI_UNSIGNED_LONG_LONG,
                  counts.data(), 1, MPI_UNSIGNED_LONG_LONG, comm.mpi());
    MPI_Allgather(&localGranUll, 1, MPI_UNSIGNED_LONG_LONG,
                  granularities.data(), 1, MPI_UNSIGNED_LONG_LONG, comm.mpi());

    return fromRankMetadata(comm, mode, device,
                            std::vector<std::uint64_t>(counts.begin(), counts.end()),
                            std::vector<std::uint64_t>(granularities.begin(), granularities.end()));
#endif
  }

  // Pure layout computation. Useful for testing without a GPU or MPI exchange;
  // all ranks get identical offsets when given identical rank-ordered inputs.
  static VmmLayout fromRankMetadata(
      const VmmComm& comm, IpcMode mode, CUdevice device,
      const std::vector<std::uint64_t>& logicalCounts,
      const std::vector<std::uint64_t>& cudaGranularities) {
    if (comm.size() < 1 || comm.rank() < 0 || comm.rank() >= comm.size() ||
        logicalCounts.size() != static_cast<std::size_t>(comm.size()) ||
        cudaGranularities.size() != logicalCounts.size()) {
      abortWithMessage(comm, "VMM layout metadata does not match communicator ranks");
    }
    // Check the policy even for layouts generated from supplied metadata.
    (void)requestedHandleType(mode);

    VmmLayout layout(comm, mode, device);
    layout.logicalCounts_ = logicalCounts;

    // CUDA VA reservation requires CUDA alignment, not necessarily an LCM
    // including sizeof(Scalar). Both are powers of two in the common case.
    std::uint64_t cudaGran = 1;
    for (const auto gran : cudaGranularities) {
      if (gran == 0) abortWithMessage(comm, "zero CUDA VMM allocation granularity");
      cudaGran = checkedLcm(cudaGran, gran, comm);
    }
    layout.granularity_ = checkedSize(cudaGran, comm);

    // The rank boundaries must also be Scalar-aligned. This extra LCM matters
    // for valid scalar types whose sizeof(Scalar) is not a divisor of CUDA's
    // granularity; it does not change the current layout for common types.
    const std::uint64_t alignBytes =
        checkedLcm(cudaGran, static_cast<std::uint64_t>(sizeof(Scalar)), comm);
    layout.allocationAlignment_ = checkedSize(alignBytes, comm);

    layout.bytesByRank_.resize(comm.size());
    layout.baseElements_.resize(comm.size());
    std::uint64_t runningBytes = 0;
    for (int p = 0; p < comm.size(); ++p) {
      const auto requestedBytes = checkedMul(
          logicalCounts[p], static_cast<std::uint64_t>(sizeof(Scalar)), comm);
      // CUDA cannot create a zero-byte allocation. Retain one padded slot
      // even for ranks with no logical entries (as in the original arena).
      const auto bytes = std::max(alignBytes,
                                  checkedRoundUp(requestedBytes, alignBytes, comm));
      if (runningBytes % sizeof(Scalar) != 0) {
        abortWithMessage(comm, "VMM byte offset is not Scalar-aligned");
      }
      layout.baseElements_[p] = runningBytes / sizeof(Scalar);
      layout.bytesByRank_[p] = checkedSize(bytes, comm);
      runningBytes = checkedAdd(runningBytes, bytes, comm);
    }
    if (runningBytes % sizeof(Scalar) != 0) {
      abortWithMessage(comm, "VMM total byte count is not Scalar-aligned");
    }
    layout.totalBytes_ = checkedSize(runningBytes, comm);
    layout.totalElements_ = runningBytes / sizeof(Scalar);
    layout.localBytes_ = layout.bytesByRank_[comm.rank()];
    layout.logicalLocalCount_ = checkedSize(logicalCounts[comm.rank()], comm);
    return layout;
  }

  const VmmComm& comm() const { return comm_; }
  IpcMode ipcMode() const { return ipcMode_; }
  CUdevice device() const { return cuDevice_; }
  CUmemAllocationHandleType handleType() const { return requestedHandleType(ipcMode_); }
  int rank() const { return comm_.rank(); }
  int size() const { return comm_.size(); }
  std::size_t logicalLocalCount() const { return logicalLocalCount_; }
  std::size_t granularity() const { return granularity_; }
  std::size_t allocationAlignment() const { return allocationAlignment_; }
  std::size_t localBytes() const { return localBytes_; }
  std::size_t totalBytes() const { return totalBytes_; }
  std::uint64_t totalElements() const { return totalElements_; }
  const std::vector<std::uint64_t>& logicalCounts() const { return logicalCounts_; }
  const std::vector<std::size_t>& bytesByRank() const { return bytesByRank_; }
  const std::vector<std::uint64_t>& baseElements() const { return baseElements_; }

  std::uint64_t globalIndex(int owner, std::uint64_t localIndex) const {
    if (owner < 0 || owner >= size() || localIndex >= logicalCounts_[owner]) {
      abortWithMessage(comm_, "VMM layout globalIndex received an invalid owner/local index");
    }
    return checkedAdd(baseElements_[owner], localIndex, comm_);
  }

 private:
  VmmLayout(const VmmComm& comm, IpcMode mode, CUdevice device)
      : comm_(comm), ipcMode_(mode), cuDevice_(device) {}

  static CUmemAllocationHandleType requestedHandleType(IpcMode mode) {
    if (mode == IpcMode::Fabric) {
#if CUDA_VERSION >= 12040
      return CU_MEM_HANDLE_TYPE_FABRIC;
#else
      throw std::runtime_error("--vmm-ipc=fabric requires CUDA 12.4+ headers");
#endif
    }
    return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }

  static std::uint64_t checkedAdd(std::uint64_t a, std::uint64_t b,
                                  const VmmComm& comm) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
      abortWithMessage(comm, "VMM layout byte/element addition overflow");
    }
    return a + b;
  }

  static std::uint64_t checkedMul(std::uint64_t a, std::uint64_t b,
                                  const VmmComm& comm) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
      abortWithMessage(comm, "VMM layout byte multiplication overflow");
    }
    return a * b;
  }

  static std::uint64_t checkedLcm(std::uint64_t a, std::uint64_t b,
                                  const VmmComm& comm) {
    return checkedMul(a / std::gcd(a, b), b, comm);
  }

  static std::uint64_t checkedRoundUp(std::uint64_t n, std::uint64_t a,
                                      const VmmComm& comm) {
    const auto remainder = n % a; // a is positive (validated above).
    return remainder == 0 ? n : checkedAdd(n, a - remainder, comm);
  }

  static std::size_t checkedSize(std::uint64_t n, const VmmComm& comm) {
    if (n > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      abortWithMessage(comm, "VMM layout exceeds size_t/CUDA allocation size");
    }
    return static_cast<std::size_t>(n);
  }

  VmmComm comm_;
  IpcMode ipcMode_;
  CUdevice cuDevice_ = 0;
  std::size_t logicalLocalCount_ = 0;
  std::size_t granularity_ = 0;
  std::size_t allocationAlignment_ = 0;
  std::size_t localBytes_ = 0;
  std::size_t totalBytes_ = 0;
  std::uint64_t totalElements_ = 0;
  std::vector<std::uint64_t> logicalCounts_;
  std::vector<std::size_t> bytesByRank_;
  std::vector<std::uint64_t> baseElements_;
};

}  // namespace VmmExperiment

#endif  // HAVE_TPETRA_INST_CUDA
#endif  // TPETRA_EXPERIMENT_VMM_LAYOUT_HPP
