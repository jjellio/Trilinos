// @HEADER
// *****************************************************************************
// Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// Experimental derivative of
//   packages/tpetra/core/test/PerformanceCGSolve/cg_solve_file.cpp
//
// Adds a CUDA-VMM direct-address SpMV path while retaining the original
// Tpetra::CrsMatrix::apply() path as the default baseline.
//
// IMPORTANT FIRST-VERSION LIMITATION
// ----------------------------------
// CG's p vector is still a normal Tpetra::Vector.  Before a VMM SpMV, its
// local device slice is copied into this rank's VMM-owned X allocation and
// all ranks synchronize.  That work is reported separately as
// "CG: vmm publish".  The direct sparse operation itself is reported as
// "CG: vmm spmv".  The total CG timer includes both.  A later step should
// make solver vectors natively VMM-backed so this mirror copy disappears.
//
// Current VMM addressing also requires a contiguous Tpetra domain Map.
// The built-in miniFE matrix generator satisfies this requirement.

#ifdef FENCE_TIMERS
#pragma message("FENCING TIMERS")
#endif

#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_Version.hpp"
#include "TpetraUtils_MatrixGenerator.hpp"
#include "MatrixMarket_Tpetra.hpp"

#include "KokkosSparse_spmv.hpp"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_CommHelpers.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_TPETRA_INST_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include "GaleriHelper.hpp"
#include "TpetraMatrixInfo.hpp"
#include "TpetraMatrixReuseTransform.hpp"

std::string matrixName = "miniFE";
int reuse = -1;
bool noReuse = false;

namespace CGParams {
int nsize = 20;
bool printMatrix = false;
bool verbose = false;
int niters = 100;
double tolerance = 1.0e-2;
std::string filename;
std::string filename_vector;
std::string testarchive("Tpetra_PerformanceTests.xml");
std::string hostname;
double tol_small = 0.05;
double tol_large = 0.10;

bool useVmm = false;
bool validateVmm = true;
std::string vmmIpc = "posix";  // posix on current H200/HGX node, fabric on NVL72/IMEX
}  // namespace CGParams

#ifdef HAVE_TPETRA_INST_CUDA

namespace VmmExperiment {

static int g_rank = -1;

[[noreturn]] static void abortWithMessage(const std::string& msg) {
  std::fprintf(stderr, "rank %d: %s\n", g_rank, msg.c_str());
  std::fflush(stderr);
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  std::abort();
}

static void checkCudaDriver(CUresult err, const char* expr, const char* file, int line) {
  if (err == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* text = nullptr;
  cuGetErrorName(err, &name);
  cuGetErrorString(err, &text);
  char buf[2048];
  std::snprintf(buf, sizeof(buf), "%s:%d: %s failed: %s (%s)", file, line, expr,
                name ? name : "CUDA_ERROR", text ? text : "unknown");
  abortWithMessage(buf);
}

static void checkCudaRuntime(cudaError_t err, const char* expr, const char* file, int line) {
  if (err == cudaSuccess) return;
  char buf[2048];
  std::snprintf(buf, sizeof(buf), "%s:%d: %s failed: %s", file, line, expr,
                cudaGetErrorString(err));
  abortWithMessage(buf);
}

#define VMM_CU_CHECK(call) ::VmmExperiment::checkCudaDriver((call), #call, __FILE__, __LINE__)
#define VMM_CUDART_CHECK(call) ::VmmExperiment::checkCudaRuntime((call), #call, __FILE__, __LINE__)

static std::uint64_t roundUp(std::uint64_t n, std::uint64_t a) {
  return ((n + a - 1) / a) * a;
}

static void setCloseOnExec(int fd) {
  const int flags = fcntl(fd, F_GETFD);
  if (flags < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0) {
    abortWithMessage(std::string("fcntl(FD_CLOEXEC) failed: ") + std::strerror(errno));
  }
}

static void sendFdPacket(int sock, int ownerRank, int fd) {
  char control[CMSG_SPACE(sizeof(int))] = {};
  struct iovec iov {};
  iov.iov_base = &ownerRank;
  iov.iov_len  = sizeof(ownerRank);

  struct msghdr msg {};
  msg.msg_iov        = &iov;
  msg.msg_iovlen     = 1;
  msg.msg_control    = control;
  msg.msg_controllen = sizeof(control);

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type  = SCM_RIGHTS;
  cmsg->cmsg_len   = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));

  ssize_t n;
  do {
    n = sendmsg(sock, &msg, 0);
  } while (n < 0 && errno == EINTR);

  if (n != static_cast<ssize_t>(sizeof(ownerRank))) {
    abortWithMessage(std::string("sendmsg(SCM_RIGHTS) failed: ") + std::strerror(errno));
  }
}

static int recvFdPacket(int sock, int* ownerRank) {
  char control[CMSG_SPACE(sizeof(int))] = {};
  struct iovec iov {};
  iov.iov_base = ownerRank;
  iov.iov_len  = sizeof(*ownerRank);

  struct msghdr msg {};
  msg.msg_iov        = &iov;
  msg.msg_iovlen     = 1;
  msg.msg_control    = control;
  msg.msg_controllen = sizeof(control);

  ssize_t n;
  do {
    n = recvmsg(sock, &msg, 0);
  } while (n < 0 && errno == EINTR);

  if (n != static_cast<ssize_t>(sizeof(*ownerRank)) || (msg.msg_flags & MSG_CTRUNC)) {
    abortWithMessage(std::string("recvmsg(SCM_RIGHTS) failed/truncated: ") +
                     std::strerror(errno));
  }

  for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr;
       cmsg = CMSG_NXTHDR(&msg, cmsg)) {
    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
      int fd = -1;
      std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
      setCloseOnExec(fd);
      return fd;
    }
  }

  abortWithMessage("recvmsg did not contain an SCM_RIGHTS file descriptor");
}

#ifdef HAVE_MPI
// Same-OS VMM IPC.  MPI cannot transfer an FD by copying the integer value, so
// rank 0 brokers the actual descriptors using SCM_RIGHTS.
static std::vector<int> exchangePosixFds(int localFd, int nranks) {
  std::vector<int> fds(nranks, -1);
  fds[g_rank] = localFd;
  if (nranks == 1) return fds;

  MPI_Comm sharedComm = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &sharedComm);
  int localSize = 0;
  MPI_Comm_size(sharedComm, &localSize);
  MPI_Comm_free(&sharedComm);
  if (localSize != nranks) {
    abortWithMessage("--vmm-ipc=posix requires all ranks in one OS/shared-memory domain; use fabric+IMEX across OS instances");
  }

  char socketPath[sizeof(sockaddr_un{}.sun_path)] = {};
  int listenFd = -1;

  if (g_rank == 0) {
    std::snprintf(socketPath, sizeof(socketPath), "/tmp/tpetra_vmm_%u_%ld.sock",
                  static_cast<unsigned>(getuid()), static_cast<long>(getpid()));

    listenFd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (listenFd < 0) {
      abortWithMessage(std::string("socket(AF_UNIX) failed: ") + std::strerror(errno));
    }
    setCloseOnExec(listenFd);

    unlink(socketPath);
    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socketPath, sizeof(addr.sun_path) - 1);
    if (bind(listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      abortWithMessage(std::string("bind(") + socketPath + ") failed: " + std::strerror(errno));
    }
    chmod(socketPath, S_IRUSR | S_IWUSR);
    if (listen(listenFd, nranks) != 0) {
      abortWithMessage(std::string("listen failed: ") + std::strerror(errno));
    }
  }

  MPI_Bcast(socketPath, sizeof(socketPath), MPI_CHAR, 0, MPI_COMM_WORLD);

  if (g_rank == 0) {
    std::vector<int> clients(nranks, -1);
    std::vector<bool> seen(nranks, false);
    seen[0] = true;

    for (int i = 1; i < nranks; ++i) {
      int s;
      do {
        s = accept(listenFd, nullptr, nullptr);
      } while (s < 0 && errno == EINTR);
      if (s < 0) abortWithMessage(std::string("accept failed: ") + std::strerror(errno));
      setCloseOnExec(s);

      int owner = -1;
      int fd = recvFdPacket(s, &owner);
      if (owner <= 0 || owner >= nranks || seen[owner]) {
        abortWithMessage("invalid or duplicate rank in POSIX FD broker handshake");
      }
      seen[owner] = true;
      clients[owner] = s;
      fds[owner] = fd;
    }

    for (int r = 1; r < nranks; ++r) {
      for (int p = 0; p < nranks; ++p) {
        if (p == r) continue;
        sendFdPacket(clients[r], p, fds[p]);
      }
    }

    for (int r = 1; r < nranks; ++r) close(clients[r]);
    close(listenFd);
    unlink(socketPath);
  } else {
    int s = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (s < 0) abortWithMessage(std::string("socket(AF_UNIX) failed: ") + std::strerror(errno));
    setCloseOnExec(s);

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socketPath, sizeof(addr.sun_path) - 1);
    if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      abortWithMessage(std::string("connect(") + socketPath + ") failed: " + std::strerror(errno));
    }

    sendFdPacket(s, g_rank, localFd);
    for (int i = 0; i < nranks - 1; ++i) {
      int owner = -1;
      int fd = recvFdPacket(s, &owner);
      if (owner < 0 || owner >= nranks || owner == g_rank || fds[owner] != -1) {
        abortWithMessage("invalid or duplicate FD received from POSIX FD broker");
      }
      fds[owner] = fd;
    }
    close(s);
  }

  for (int p = 0; p < nranks; ++p) {
    if (fds[p] < 0) abortWithMessage("POSIX FD exchange left a missing rank allocation");
  }
  return fds;
}
#endif

enum class IpcMode { Posix, Fabric };

static IpcMode parseIpcMode(const std::string& text) {
  if (text == "posix") return IpcMode::Posix;
  if (text == "fabric") return IpcMode::Fabric;
  throw std::runtime_error("--vmm-ipc must be 'posix' or 'fabric'");
}

template <class Scalar>
class DistributedVmmArena {
 public:
  DistributedVmmArena(std::size_t logicalLocalCount, IpcMode ipcMode)
      : ipcMode_(ipcMode), logicalLocalCount_(logicalLocalCount) {
#ifndef HAVE_MPI
    throw std::runtime_error("VMM distributed arena requires MPI in this experiment");
#else
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    g_rank = rank_;

    VMM_CU_CHECK(cuInit(0));
    int runtimeDev = 0;
    VMM_CUDART_CHECK(cudaGetDevice(&runtimeDev));
    VMM_CU_CHECK(cuDeviceGet(&cuDevice_, runtimeDev));

    CUmemAllocationHandleType shareType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    if (ipcMode_ == IpcMode::Fabric) {
#if CUDA_VERSION >= 12040
      shareType = CU_MEM_HANDLE_TYPE_FABRIC;
#else
      throw std::runtime_error("--vmm-ipc=fabric requires CUDA 12.4+ headers");
#endif
    }

    CUmemAllocationProp prop {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cuDevice_;
    prop.requestedHandleTypes = shareType;

    std::size_t localGran = 0;
    VMM_CU_CHECK(cuMemGetAllocationGranularity(&localGran, &prop,
                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    unsigned long long localGranUll = static_cast<unsigned long long>(localGran);
    std::vector<unsigned long long> allGran(size_);
    MPI_Allgather(&localGranUll, 1, MPI_UNSIGNED_LONG_LONG,
                  allGran.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    std::uint64_t commonGran = 1;
    for (const auto g : allGran) commonGran = std::lcm(commonGran, static_cast<std::uint64_t>(g));
    granularity_ = static_cast<std::size_t>(commonGran);

    const std::uint64_t requestedBytes =
        static_cast<std::uint64_t>(logicalLocalCount_) * sizeof(Scalar);
    localBytes_ = static_cast<std::size_t>(roundUp(requestedBytes, commonGran));
    // cuMemCreate does not accept a zero-byte physical allocation.  Preserve a
    // valid padded slot even for an empty rank; logicalLocalCount_ remains 0.
    if (localBytes_ == 0) localBytes_ = granularity_;

    unsigned long long localBytesUll = static_cast<unsigned long long>(localBytes_);
    std::vector<unsigned long long> bytesByRankUll(size_);
    MPI_Allgather(&localBytesUll, 1, MPI_UNSIGNED_LONG_LONG,
                  bytesByRankUll.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);

    bytesByRank_.resize(size_);
    baseElements_.resize(size_);
    std::uint64_t runningBytes = 0;
    for (int p = 0; p < size_; ++p) {
      bytesByRank_[p] = static_cast<std::size_t>(bytesByRankUll[p]);
      if (runningBytes % sizeof(Scalar) != 0) abortWithMessage("VMM byte offset is not Scalar-aligned");
      baseElements_[p] = runningBytes / sizeof(Scalar);
      runningBytes += bytesByRank_[p];
    }
    totalBytes_ = static_cast<std::size_t>(runningBytes);
    totalElements_ = runningBytes / sizeof(Scalar);

    VMM_CU_CHECK(cuMemCreate(&localHandle_, localBytes_, &prop, 0));

    handles_.resize(size_);
    handles_[rank_] = localHandle_;

    if (ipcMode_ == IpcMode::Posix) {
      int localFd = -1;
      VMM_CU_CHECK(cuMemExportToShareableHandle(&localFd, localHandle_,
                                                 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
      std::vector<int> fds = exchangePosixFds(localFd, size_);

      for (int p = 0; p < size_; ++p) {
        if (p == rank_) continue;
        const int fd = fds[p];
        if (fcntl(fd, F_GETFD) == -1) {
          abortWithMessage("received POSIX FD is invalid before CUDA import");
        }

        // NOTE: On the H200 system used to develop this experiment, the
        // working import convention matches NVIDIA NCCL: encode the numeric
        // FD value in the void* argument.  This differs from some CUDA guide
        // examples that show &fd.
        VMM_CU_CHECK(cuMemImportFromShareableHandle(
            &handles_[p], reinterpret_cast<void*>(static_cast<uintptr_t>(fd)),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
      }

      MPI_Barrier(MPI_COMM_WORLD);
      for (const int fd : fds) if (fd >= 0) close(fd);
    } else {
#if CUDA_VERSION >= 12040
      // NVL72 / IMEX NOTE:
      // On Vera Rubin NVL72, ranks may span independent OS instances.  Use
      // --vmm-ipc=fabric and configure every process with access to the same
      // NVIDIA IMEX channel.  If cuMemCreate(...FABRIC...) returns
      // CUDA_ERROR_NOT_PERMITTED, check IMEX service/channel configuration and
      // /dev/nvidia-caps-imex-channels permissions.
      CUmemFabricHandle localFabric {};
      VMM_CU_CHECK(cuMemExportToShareableHandle(&localFabric, localHandle_,
                                                 CU_MEM_HANDLE_TYPE_FABRIC, 0));
      std::vector<CUmemFabricHandle> allFabric(size_);
      MPI_Allgather(&localFabric, sizeof(CUmemFabricHandle), MPI_BYTE,
                    allFabric.data(), sizeof(CUmemFabricHandle), MPI_BYTE,
                    MPI_COMM_WORLD);
      for (int p = 0; p < size_; ++p) {
        if (p == rank_) continue;
        VMM_CU_CHECK(cuMemImportFromShareableHandle(
            &handles_[p], &allFabric[p], CU_MEM_HANDLE_TYPE_FABRIC));
      }
#else
      throw std::runtime_error("fabric IPC not compiled with this CUDA version");
#endif
    }

    VMM_CU_CHECK(cuMemAddressReserve(&baseVa_, totalBytes_, granularity_, 0, 0));

    std::uint64_t byteOffset = 0;
    for (int p = 0; p < size_; ++p) {
      VMM_CU_CHECK(cuMemMap(baseVa_ + byteOffset, bytesByRank_[p], 0, handles_[p], 0));
      byteOffset += bytesByRank_[p];
    }

    CUmemAccessDesc access {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cuDevice_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    VMM_CU_CHECK(cuMemSetAccess(baseVa_, totalBytes_, &access, 1));

    globalPtr_ = reinterpret_cast<Scalar*>(baseVa_);
    localPtr_  = globalPtr_ + baseElements_[rank_];

    if (rank_ == 0) {
      std::cout << "VMM arena: granularity=" << granularity_
                << " bytes, total mapped=" << totalBytes_
                << " bytes, IPC=" << (ipcMode_ == IpcMode::Posix ? "posix" : "fabric")
                << std::endl;
    }
#endif
  }

  ~DistributedVmmArena() {
#ifdef HAVE_MPI
    if (baseVa_ == 0) return;
    Kokkos::fence("VMM arena teardown fence");
    MPI_Barrier(MPI_COMM_WORLD);

    std::uint64_t byteOffset = 0;
    for (int p = 0; p < size_; ++p) {
      // Destructor must not throw.  Print failures and continue best-effort.
      CUresult e = cuMemUnmap(baseVa_ + byteOffset, bytesByRank_[p]);
      if (e != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(e, &s);
        std::fprintf(stderr, "rank %d: cuMemUnmap failed during teardown: %s\n",
                     rank_, s ? s : "unknown");
      }
      byteOffset += bytesByRank_[p];
    }
    cuMemAddressFree(baseVa_, totalBytes_);
    baseVa_ = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    for (auto h : handles_) cuMemRelease(h);
#endif
  }

  DistributedVmmArena(const DistributedVmmArena&) = delete;
  DistributedVmmArena& operator=(const DistributedVmmArena&) = delete;

  Scalar* globalPtr() const { return globalPtr_; }
  Scalar* localPtr() const { return localPtr_; }
  std::size_t logicalLocalCount() const { return logicalLocalCount_; }
  std::uint64_t totalElements() const { return totalElements_; }
  const std::vector<std::uint64_t>& baseElements() const { return baseElements_; }
  int rank() const { return rank_; }
  int size() const { return size_; }

 private:
  IpcMode ipcMode_;
  int rank_ = 0;
  int size_ = 1;
  CUdevice cuDevice_ = 0;
  std::size_t logicalLocalCount_ = 0;
  std::size_t granularity_ = 0;
  std::size_t localBytes_ = 0;
  std::size_t totalBytes_ = 0;
  std::uint64_t totalElements_ = 0;
  CUmemGenericAllocationHandle localHandle_ {};
  std::vector<CUmemGenericAllocationHandle> handles_;
  std::vector<std::size_t> bytesByRank_;
  std::vector<std::uint64_t> baseElements_;
  CUdeviceptr baseVa_ = 0;
  Scalar* globalPtr_ = nullptr;
  Scalar* localPtr_ = nullptr;
};

template <class CrsMatrix, class Vector>
class VmmSpmvContext {
 public:
  using scalar_type = typename CrsMatrix::scalar_type;
  using LO          = typename CrsMatrix::local_ordinal_type;
  using GO          = typename CrsMatrix::global_ordinal_type;
  using local_matrix_type = typename CrsMatrix::local_matrix_device_type;
  using device_type = typename local_matrix_type::device_type;
  using memory_space = typename device_type::memory_space;
  using entries_type = typename local_matrix_type::index_type::non_const_type;
  using unmanaged_vector_type =
      Kokkos::View<scalar_type*, device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  static_assert(std::is_same<scalar_type, double>::value,
                "This initial PerformanceCGSolve VMM experiment supports Scalar=double only");
  static_assert(std::is_integral<LO>::value && std::is_signed<LO>::value,
                "KokkosSparse local ordinal must be a signed integer");

  VmmSpmvContext(const Teuchos::RCP<CrsMatrix>& A, IpcMode ipcMode)
      : A_(A),
        arena_(A->getDomainMap()->getLocalNumElements(), ipcMode),
        localA_(A->getLocalMatrixDevice()) {
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
#endif
    buildAddressedMatrix();

    xGlobal_ = unmanaged_vector_type(arena_.globalPtr(), arena_.totalElements());
    xLocal_  = unmanaged_vector_type(arena_.localPtr(), arena_.logicalLocalCount());
  }

  // Publish the current Tpetra vector into this rank's physical VMM allocation.
  // This is scaffolding for the first implementation; a native VMM-backed
  // Tpetra/solver vector would eliminate this deep copy.
  void publish(const Vector& x) {
    auto x2d = x.getLocalViewDevice(Tpetra::Access::ReadOnly);
    if (x2d.extent(1) != 1 || x2d.extent(0) != xLocal_.extent(0)) {
      throw std::runtime_error("VMM publish: unexpected Tpetra vector local shape");
    }
    auto x1d = Kokkos::subview(x2d, Kokkos::ALL(), 0);
    Kokkos::deep_copy(xLocal_, x1d);
    Kokkos::fence("VMM publish local X fence");
#ifdef HAVE_MPI
    // Establish a simple global epoch: all owner writes are complete before any
    // rank starts ordinary remote loads.  This is intentionally conservative.
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  void applyPublished(Vector& y) const {
    auto y2d = y.getLocalViewDevice(Tpetra::Access::OverwriteAll);
    if (y2d.extent(1) != 1 || y2d.extent(0) != static_cast<std::size_t>(vmmA_.numRows())) {
      throw std::runtime_error("VMM SpMV: unexpected Tpetra output vector local shape");
    }
    auto y1d = Kokkos::subview(y2d, Kokkos::ALL(), 0);

    const scalar_type one  = static_cast<scalar_type>(1.0);
    const scalar_type zero = static_cast<scalar_type>(0.0);
    KokkosSparse::spmv("N", one, vmmA_, xGlobal_, zero, y1d);
    Kokkos::fence("VMM direct SpMV fence");
  }

  void apply(const Vector& x, Vector& y) {
    publish(x);
    applyPublished(y);
  }

 private:
  void buildAddressedMatrix() {
    auto domainMap = A_->getDomainMap();
    auto colMap    = A_->getColMap();

    if (!domainMap->isContiguous()) {
      throw std::runtime_error(
          "VMM prototype currently requires a contiguous Tpetra domain Map. "
          "The built-in miniFE generator satisfies this; general Maps can be "
          "supported later with Tpetra owner/LID lookup metadata.");
    }

    const std::size_t myCount = domainMap->getLocalNumElements();
    auto myGids = domainMap->getLocalElementList();
    long long myFirst = 0;
    if (myCount != 0) myFirst = static_cast<long long>(myGids[0]);

    std::vector<unsigned long long> counts(size_);
    std::vector<long long> firstGid(size_);
    unsigned long long myCountUll = static_cast<unsigned long long>(myCount);
#ifdef HAVE_MPI
    MPI_Allgather(&myCountUll, 1, MPI_UNSIGNED_LONG_LONG,
                  counts.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    MPI_Allgather(&myFirst, 1, MPI_LONG_LONG,
                  firstGid.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
#else
    counts[0] = myCountUll;
    firstGid[0] = myFirst;
#endif

    if (arena_.totalElements() > static_cast<std::uint64_t>(std::numeric_limits<LO>::max())) {
      throw std::runtime_error(
          "Padded VMM arena exceeds the 32-bit/signed LocalOrdinal range. "
          "Use a wider VMM ordinal for this configuration.");
    }

    const std::size_t numColLids = colMap->getLocalNumElements();
    std::vector<LO> colLidToVmm(numColLids);
    auto colGids = colMap->getLocalElementList();
    const auto& base = arena_.baseElements();

    for (std::size_t lid = 0; lid < numColLids; ++lid) {
      const long long gid = static_cast<long long>(colGids[lid]);
      int owner = -1;
      std::uint64_t ownerLocal = 0;
      for (int p = 0; p < size_; ++p) {
        if (counts[p] == 0) continue;
        const long long begin = firstGid[p];
        const long long end   = begin + static_cast<long long>(counts[p]);
        if (gid >= begin && gid < end) {
          owner = p;
          ownerLocal = static_cast<std::uint64_t>(gid - begin);
          break;
        }
      }
      if (owner < 0) {
        throw std::runtime_error("Failed to resolve a column-map GID to a domain-map owner");
      }
      const std::uint64_t addressOrdinal = base[owner] + ownerLocal;
      if (addressOrdinal > static_cast<std::uint64_t>(std::numeric_limits<LO>::max())) {
        throw std::runtime_error("VMM address ordinal exceeds LocalOrdinal range");
      }
      colLidToVmm[lid] = static_cast<LO>(addressOrdinal);
    }

    const std::size_t nnz = localA_.nnz();
    vmmEntries_ = entries_type("VMM direct column ordinals", nnz);

    auto entriesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                            localA_.graph.entries);
    auto vmmEntriesHost = Kokkos::create_mirror_view(vmmEntries_);
    for (std::size_t k = 0; k < nnz; ++k) {
      const LO colLid = entriesHost(k);
      if (colLid < 0 || static_cast<std::size_t>(colLid) >= colLidToVmm.size()) {
        throw std::runtime_error("Local CRS column ordinal is outside the Tpetra column Map");
      }
      vmmEntriesHost(k) = colLidToVmm[static_cast<std::size_t>(colLid)];
    }
    Kokkos::deep_copy(vmmEntries_, vmmEntriesHost);
    Kokkos::fence("build VMM column ordinals");

    const LO numRows = static_cast<LO>(localA_.numRows());
    const LO numCols = static_cast<LO>(arena_.totalElements());
    vmmA_ = local_matrix_type("VMM addressed local matrix",
                              numRows, numCols, localA_.nnz(),
                              localA_.values, localA_.graph.row_map, vmmEntries_);

    if (rank_ == 0) {
      std::cout << "VMM CRS translation: reusing Tpetra row_map + values; "
                << "translated local column ordinals now index padded global VMM X"
                << std::endl;
    }
  }

  Teuchos::RCP<CrsMatrix> A_;
  DistributedVmmArena<scalar_type> arena_;
  local_matrix_type localA_;
  entries_type vmmEntries_;
  local_matrix_type vmmA_;
  unmanaged_vector_type xGlobal_;
  unmanaged_vector_type xLocal_;
  int rank_ = 0;
  int size_ = 1;
};

}  // namespace VmmExperiment

#endif  // HAVE_TPETRA_INST_CUDA

template <class CrsMatrix, class Vector>
static void applyCgOperator(const Teuchos::RCP<CrsMatrix>& A,
                            const Teuchos::RCP<Vector>& x,
                            const Teuchos::RCP<Vector>& y
#ifdef HAVE_TPETRA_INST_CUDA
                            , VmmExperiment::VmmSpmvContext<CrsMatrix, Vector>* vmm
#endif
                            ) {
  using Teuchos::TimeMonitor;
#ifdef HAVE_TPETRA_INST_CUDA
  if (vmm != nullptr) {
    {
      TimeMonitor t(*TimeMonitor::getNewTimer("CG: vmm publish"));
      vmm->publish(*x);
    }
    {
      TimeMonitor t(*TimeMonitor::getNewTimer("CG: vmm spmv"));
      vmm->applyPublished(*y);
    }
    return;
  }
#endif
  {
    #ifdef FENCE_TIMERS
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    TimeMonitor t(*TimeMonitor::getNewTimer("CG: spmv"));
    A->apply(*x, *y);

    #ifdef FENCE_TIMERS
    Kokkos::fence("CG Tpetra SpMV timing fence");
    #endif
  }
}

template <class CrsMatrix, class Vector>
bool cg_solve(Teuchos::RCP<CrsMatrix> A,
              Teuchos::RCP<Vector> b,
              Teuchos::RCP<Vector> x,
              int myproc,
              double tolerance,
              int max_iter
#ifdef HAVE_TPETRA_INST_CUDA
              , VmmExperiment::VmmSpmvContext<CrsMatrix, Vector>* vmm
#endif
              ) {
  using Teuchos::TimeMonitor;
  const std::string addTimerName = "CG: axpby";
  const std::string dotTimerName = "CG: dot";

  static_assert(std::is_same<typename CrsMatrix::scalar_type,
                             typename Vector::scalar_type>::value,
                "The CrsMatrix and Vector template parameters must have the same scalar_type.");

  using ScalarType    = typename Vector::scalar_type;
  using magnitude_type = typename Vector::mag_type;
  using LO             = typename Vector::local_ordinal_type;

  auto r  = Tpetra::createVector<ScalarType>(A->getRangeMap());
  auto p  = Tpetra::createVector<ScalarType>(A->getRangeMap());
  auto Ap = Tpetra::createVector<ScalarType>(A->getRangeMap());

  magnitude_type normr = 0;
  magnitude_type rtrans = 0;
  magnitude_type oldrtrans = 0;
  LO print_freq = max_iter / 10;
  print_freq = std::min(print_freq, static_cast<LO>(50));
  print_freq = std::max(print_freq, static_cast<LO>(1));

  {
    TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
    p->update(1.0, *x, 0.0, *x, 0.0);
    #ifdef FENCE_TIMERS
    Kokkos::fence("CG axpby p timing fence");
    #endif
  }

  applyCgOperator(A, p, Ap
#ifdef HAVE_TPETRA_INST_CUDA
                  , vmm
#endif
                  );

  {
    TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
    r->update(1.0, *b, -1.0, *Ap, 0.0);
    #ifdef FENCE_TIMERS
    Kokkos::fence("CG axpby r timing fence");
    #endif
  }
  {
    TimeMonitor t(*TimeMonitor::getNewTimer(dotTimerName));
    rtrans = r->dot(*r);
    #ifdef FENCE_TIMERS
    Kokkos::fence("CG dot timing fence");
    #endif
  }

  normr = std::sqrt(rtrans);
  if (myproc == 0) std::cout << "Initial Residual = " << normr << std::endl;

  magnitude_type brkdown_tol = std::numeric_limits<magnitude_type>::epsilon();
  LO k;
  for (k = 1; k <= max_iter && normr > tolerance; ++k) {
    if (k == 1) {
      TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
      p->update(1.0, *r, 0.0);
    } else {
      oldrtrans = rtrans;
      {
        TimeMonitor t(*TimeMonitor::getNewTimer(dotTimerName));
        rtrans = r->dot(*r);
      }
      {
        TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
        const magnitude_type beta = rtrans / oldrtrans;
        p->update(1.0, *r, beta);
      }
    }

    normr = std::sqrt(rtrans);
    if (myproc == 0 && (k % print_freq == 0 || k == max_iter)) {
      std::cout << "Iteration = " << k << " Residual = " << normr << std::endl;
    }

    magnitude_type alpha = 0;
    magnitude_type p_ap_dot = 0;

    applyCgOperator(A, p, Ap
#ifdef HAVE_TPETRA_INST_CUDA
                    , vmm
#endif
                    );

    {
      TimeMonitor t(*TimeMonitor::getNewTimer(dotTimerName));
      p_ap_dot = Ap->dot(*p);
    }
    {
      TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
      if (p_ap_dot < brkdown_tol) {
        if (p_ap_dot < 0) {
          std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!" << std::endl;
          return false;
        } else {
          brkdown_tol = 0.1 * p_ap_dot;
        }
      }
      alpha = rtrans / p_ap_dot;
      x->update(alpha, *p, 1.0);
      r->update(-alpha, *Ap, 1.0);
    }
  }

  {
    TimeMonitor t(*TimeMonitor::getNewTimer(dotTimerName));
    rtrans = r->dot(*r);
  }
  normr = std::sqrt(rtrans);
  return true;
}

template <class Node>
int run() {
  using namespace CGParams;
  using std::cout;
  using std::endl;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::StackedTimer;
  using Teuchos::TimeMonitor;

  using Scalar = Tpetra::Vector<>::scalar_type;
  using LO = typename Tpetra::Map<>::local_ordinal_type;
  using GO = typename Tpetra::Map<>::global_ordinal_type;
  using crs_matrix_type = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
  using vec_type = Tpetra::Vector<Scalar, LO, GO, Node>;
  using map_type = Tpetra::Map<LO, GO, Node>;

  RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
  const int myRank = comm->getRank();

#ifdef HAVE_TPETRA_INST_CUDA
  VmmExperiment::g_rank = myRank;
#endif

  if (verbose) {
    if (myRank == 0) cout << "Comm info: ";
    cout << *comm;
  }


  RCP<crs_matrix_type> A;
  
  if (!filename.empty()) {
    A = Tpetra::MatrixMarket::Reader<crs_matrix_type>::readSparseFile(
        filename, comm);
  }
  else if (matrixName == "miniFE") {
    A = Tpetra::Utils::MatrixGenerator<crs_matrix_type>::
        generate_miniFE_matrix(nsize, comm);
  }
  else {
    A = my_helper::get_galeri_matrix<Node>(
        matrixName,
        nsize,
        comm);
  }
  if (printMatrix) {
    RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(cout));
    A->describe(*fos, Teuchos::VERB_EXTREME);
  } else if (verbose) {
    cout << endl << A->description() << endl << endl;
  }

  if (!A->getRangeMap()->isSameAs(*(A->getDomainMap()))) {
    throw std::runtime_error("The matrix must have domain and range maps that are the same.");
  }

  /* Galeri / miniFE / file creation ... */
  
  if (reuse >= 0) {
      TpetraMatrixTools::ReuseTransformStats reuseStats;
  
      A = TpetraMatrixTools::limitReuse(
          A,
          reuse,
          &reuseStats);
  
      if (myRank == 0) {
          std::cout
              << "Reuse transform:"
              << " requested=" << reuseStats.requestedReuse
              << " intrinsic-max=" << reuseStats.intrinsicMaxReuse
              << " domain=" << reuseStats.originalDomainSize
              << " -> " << reuseStats.transformedDomainSize
              << " nnz=" << reuseStats.globalNnz
              << std::endl;
      }
  }


  RCP<const map_type> map = A->getRangeMap();
  RCP<vec_type> b;

  if (nsize < 0) {
    using reader_type =
        Tpetra::MatrixMarket::Reader<crs_matrix_type>;
  
    b = reader_type::readVectorFile(
        filename_vector,
        map->getComm(),
        map);
  }
  else if (matrixName == "miniFE") {
    using gen_type =
        Tpetra::Utils::MatrixGenerator<crs_matrix_type>;
  
    b = gen_type::generate_miniFE_vector(
        nsize,
        map->getComm());
  }
  else {
    b = rcp(new vec_type(map));
    b->putScalar(1.0);
  }

  const Tpetra::global_size_t ng = map->getGlobalNumElements();
  if (myRank == 0) {
    std::cout << "Matrix = " << matrixName << std::endl;
    std::cout << "Global matrix size = " << ng << std::endl;
    std::cout << "SpMV backend = "
              << (useVmm ? "CUDA VMM direct-address" : "Tpetra apply")
              << std::endl;
  }

  const auto access = TpetraMatrixInfo::analyzeAccessPattern(*A);

  TpetraMatrixInfo::print(std::cout, access);

  RCP<vec_type> x(new vec_type(A->getDomainMap()));

#ifdef HAVE_TPETRA_INST_CUDA
  std::unique_ptr<VmmExperiment::VmmSpmvContext<crs_matrix_type, vec_type>> vmm;
  if (useVmm) {
    using exec_space = typename crs_matrix_type::device_type::execution_space;
    if (!std::is_same<exec_space, Kokkos::Cuda>::value) {
      throw std::runtime_error("--vmm requires the CUDA Tpetra node");
    }
    const auto ipcMode = VmmExperiment::parseIpcMode(vmmIpc);
    vmm.reset(new VmmExperiment::VmmSpmvContext<crs_matrix_type, vec_type>(A, ipcMode));

    if (validateVmm) {
      auto yRef = Tpetra::createVector<Scalar>(A->getRangeMap());
      auto yVmm = Tpetra::createVector<Scalar>(A->getRangeMap());
      auto diff = Tpetra::createVector<Scalar>(A->getRangeMap());

      A->apply(*b, *yRef);
      vmm->apply(*b, *yVmm);
      Tpetra::deep_copy(*diff, *yVmm);
      diff->update(-1.0, *yRef, 1.0);
      const auto err = diff->norm2();
      const auto ref = yRef->norm2();
      const double rel = ref == 0 ? static_cast<double>(err)
                                  : static_cast<double>(err / ref);
      if (myRank == 0) {
        std::cout << "VMM validation relative ||Yvmm-Ytpetra||2/||Ytpetra||2 = "
                  << rel << std::endl;
      }
      if (!(rel < 1.0e-10)) {
        throw std::runtime_error("VMM direct SpMV validation against Tpetra::apply failed");
      }
    }
  }
#else
  if (useVmm) throw std::runtime_error("This Tpetra build has no CUDA instantiation for --vmm");
#endif

  // Untimed warm-up apply using the selected backend.
#ifdef HAVE_TPETRA_INST_CUDA
  if (vmm) vmm->apply(*b, *x);
  else A->apply(*b, *x);
#else
  A->apply(*b, *x);
#endif


Kokkos::fence("warmup complete");
x->putScalar(0);
Kokkos::fence("x reset complete");

// Align the starting line.
MPI_Barrier(MPI_COMM_WORLD);

// you must start a stacked timer somewhere...
RCP<StackedTimer> timer = rcp(new StackedTimer("CG: global"));
TimeMonitor::setStackedTimer(timer);

const double t0 = MPI_Wtime();

  const bool success = cg_solve(A, b, x, myRank, tolerance, niters
#ifdef HAVE_TPETRA_INST_CUDA
                                , vmm.get()
#endif
                                );

// Define solver completion as device work completed.
Kokkos::fence("CG completion");

const double local_elapsed = MPI_Wtime() - t0;

timer->stopBaseTimer();
double critical_elapsed = 0.0;
MPI_Reduce(&local_elapsed,
           &critical_elapsed,
           1,
           MPI_DOUBLE,
           MPI_MAX,
           0,
           MPI_COMM_WORLD);



 

  StackedTimer::OutputOptions options;
  options.print_warnings = false;
  options.output_proc_minmax = true;
  options.output_fraction = options.output_histogram = options.output_minmax = true;
  timer->report(std::cout, comm, options);

  const std::string testBaseName = std::string("Tpetra CGSolve ") +
      (useVmm ? "VMM " : "") +
      (Tpetra::Details::Behavior::cudaLaunchBlocking() ? "CUDA_LAUNCH_BLOCKING " : "");
  auto xmlOut = timer->reportWatchrXML(testBaseName + std::to_string(comm->getSize()) + " ranks", comm);

  if (myRank == 0) {
    std::cout << "CG Solve Critical Max Time: " << std::setprecision(std::numeric_limits<double>::max_digits10) << critical_elapsed << "\n";
    if (xmlOut.length()) std::cout << "\nAlso created Watchr performance report " << xmlOut << '\n';
    if (success) std::cout << "End Result: TEST PASSED\n";
    else std::cout << "End Result: TEST FAILED\n";
  }

  // Ensure VMM resources are destroyed before Kokkos/MPI teardown.
#ifdef HAVE_TPETRA_INST_CUDA
  vmm.reset();
#endif

  return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
  using namespace CGParams;
  using default_exec = Tpetra::Details::DefaultTypes::execution_space;

  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

  int myRank = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#endif

  int numthreads = 1;
  int numgpus = 1;
  if (const char* raw = std::getenv("KOKKOS_NUM_DEVICES")) numgpus = std::atoi(raw);

  bool useCuda = false;

  Teuchos::CommandLineProcessor cmdp(false, true);
  cmdp.setOption("verbose", "quiet", &verbose, "Print messages and results.");
  cmdp.setOption("numthreads", &numthreads, "Number of threads per thread team.");
  cmdp.setOption("numgpus", &numgpus, "Number of GPUs visible to this process.");
  cmdp.setOption("hostname", &hostname, "Override of hostname for PerfTest entry.");
  cmdp.setOption("testarchive", &testarchive, "Set filename for Performance Test archive.");
  cmdp.setOption("matrixType", &matrixName,
      "Matrix to generate: miniFE (default), or any supported Galeri matrix name.");
  cmdp.setOption("filename", &filename, "Filename for test matrix.");
  cmdp.setOption("filename_vector", &filename_vector, "Filename for test matrix vector.");
  cmdp.setOption("tolerance", &tolerance, "Relative residual tolerance used for solver.");
  cmdp.setOption("iterations", &niters, "Maximum number of iterations.");
  cmdp.setOption("printMatrix", "noPrintMatrix", &printMatrix,
                 "Print the full matrix after reading it.");
  cmdp.setOption("size", &nsize, "Generate miniFE matrix with X^3 elements.");
  cmdp.setOption("tol_small", &tol_small, "Tolerance for total CG-Time and final residual.");
  cmdp.setOption("tol_large", &tol_large, "Tolerance for individual times.");
  cmdp.setOption("vmm", "no-vmm", &useVmm,
                 "Use experimental CUDA-VMM direct-address SpMV instead of Tpetra::apply.");
  cmdp.setOption("vmm-validate", "no-vmm-validate", &validateVmm,
                 "Validate one VMM SpMV against Tpetra::apply before CG.");
  cmdp.setOption("vmm-ipc", &vmmIpc,
                 "VMM IPC backend: posix (same OS) or fabric (NVL72/IMEX).");

  cmdp.setOption(
      "reuse",
      &reuse,
      "Maximum reuse of a synthetic X scalar; 0 means no reuse.");
  
  cmdp.setOption(
      "no-reuse",
      "allow-reuse",
      &noReuse,
      "Equivalent to --reuse=0.");
#ifdef HAVE_TPETRA_INST_CUDA
  cmdp.setOption("cuda", "no-cuda", &useCuda, "Use Cuda node");
#endif

  if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return EXIT_FAILURE;
  }
  if (noReuse) {
      reuse = 0;
  }

#ifdef HAVE_TPETRA_INST_CUDA
  if (!useCuda) {
    if (std::is_same<default_exec, Kokkos::Cuda>::value) {
      if (myRank == 0) std::cout << "No node specified; using default CUDA node\n";
      useCuda = true;
    }
  }
#else
  if (useVmm) {
    if (myRank == 0) std::cerr << "--vmm requested but Tpetra CUDA instantiation is disabled\n";
    return EXIT_FAILURE;
  }
#endif

  if (numgpus <= 0) numgpus = 1;

  Kokkos::InitializationSettings kokkosArgs;
  kokkosArgs.set_num_threads(numthreads);
  // With the binding wrapper used in this project, each MPI rank normally sees
  // exactly one GPU, so --numgpus=1 makes this select CUDA device 0 in each
  // process (which is a distinct physical H200 after CUDA_VISIBLE_DEVICES).
  kokkosArgs.set_device_id(myRank % numgpus);
  kokkosArgs.set_disable_warnings(!verbose);
  Kokkos::initialize(kokkosArgs);

  int rc = EXIT_FAILURE;
  {
#ifdef HAVE_TPETRA_INST_CUDA
    if (useCuda) {
      rc = run<Tpetra::KokkosCompat::KokkosCudaWrapperNode>();
    } else
#endif
    {
      if (myRank == 0) std::cerr << "Error: CUDA node was not enabled. CG was not run.\n";
    }
  }

  Kokkos::finalize();
  return rc;
}


