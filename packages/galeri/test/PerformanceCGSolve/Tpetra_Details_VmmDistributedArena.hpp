// Experimental CUDA VMM arena extracted from PerformanceCGSolve.
// Owns only the VMM allocation / mapping and MPI-scoped IPC resources.
#ifndef TPETRA_EXPERIMENT_VMM_DISTRIBUTED_ARENA_HPP
#define TPETRA_EXPERIMENT_VMM_DISTRIBUTED_ARENA_HPP

#ifdef HAVE_TPETRA_INST_CUDA

#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"
#ifdef HAVE_MPI
#include "Teuchos_DefaultMpiComm.hpp"
#include <mpi.h>
#endif
#include <Kokkos_Core.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>
#include <utility>

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

inline std::uint64_t roundUp(std::uint64_t n, std::uint64_t a) {
  return ((n + a - 1) / a) * a;
}

inline void setCloseOnExec(int fd, const VmmComm& comm) {
  const int flags = fcntl(fd, F_GETFD);
  if (flags < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0) {
    abortWithMessage(comm, std::string("fcntl(FD_CLOEXEC) failed: ") + std::strerror(errno));
  }
}

inline void sendFdPacket(int sock, int ownerRank, int fd, const VmmComm& comm) {
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
    abortWithMessage(comm, std::string("sendmsg(SCM_RIGHTS) failed: ") + std::strerror(errno));
  }
}

inline int recvFdPacket(int sock, int* ownerRank, const VmmComm& comm) {
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
    abortWithMessage(comm, std::string("recvmsg(SCM_RIGHTS) failed/truncated: ") +
                     std::strerror(errno));
  }

  for (struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr;
       cmsg = CMSG_NXTHDR(&msg, cmsg)) {
    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
      int fd = -1;
      std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
      setCloseOnExec(fd, comm);
      return fd;
    }
  }

  abortWithMessage(comm, "recvmsg did not contain an SCM_RIGHTS file descriptor");
}

#ifdef HAVE_MPI
// Same-OS VMM IPC.  MPI cannot transfer an FD by copying the integer value, so
// rank 0 brokers the actual descriptors using SCM_RIGHTS.
inline std::vector<int> exchangePosixFds(int localFd, const VmmComm& comm) {
  const int rank = comm.rank();
  const int nranks = comm.size();
  std::vector<int> fds(nranks, -1);
  fds[rank] = localFd;
  if (nranks == 1) return fds;

  MPI_Comm sharedComm = MPI_COMM_NULL;
  MPI_Comm_split_type(comm.mpi(), MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &sharedComm);
  int localSize = 0;
  MPI_Comm_size(sharedComm, &localSize);
  MPI_Comm_free(&sharedComm);
  if (localSize != nranks) {
    abortWithMessage(comm, "--vmm-ipc=posix requires all ranks in one OS/shared-memory domain; use fabric+IMEX across OS instances");
  }

  char socketPath[sizeof(sockaddr_un{}.sun_path)] = {};
  int listenFd = -1;

  if (rank == 0) {
    std::snprintf(socketPath, sizeof(socketPath), "/tmp/tpetra_vmm_%u_%ld.sock",
                  static_cast<unsigned>(getuid()), static_cast<long>(getpid()));

    listenFd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (listenFd < 0) {
      abortWithMessage(comm, std::string("socket(AF_UNIX) failed: ") + std::strerror(errno));
    }
    setCloseOnExec(listenFd, comm);

    unlink(socketPath);
    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socketPath, sizeof(addr.sun_path) - 1);
    if (bind(listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      abortWithMessage(comm, std::string("bind(") + socketPath + ") failed: " + std::strerror(errno));
    }
    chmod(socketPath, S_IRUSR | S_IWUSR);
    if (listen(listenFd, nranks) != 0) {
      abortWithMessage(comm, std::string("listen failed: ") + std::strerror(errno));
    }
  }

  MPI_Bcast(socketPath, sizeof(socketPath), MPI_CHAR, 0, comm.mpi());

  if (rank == 0) {
    std::vector<int> clients(nranks, -1);
    std::vector<bool> seen(nranks, false);
    seen[0] = true;

    for (int i = 1; i < nranks; ++i) {
      int s;
      do {
        s = accept(listenFd, nullptr, nullptr);
      } while (s < 0 && errno == EINTR);
      if (s < 0) abortWithMessage(comm, std::string("accept failed: ") + std::strerror(errno));
      setCloseOnExec(s, comm);

      int owner = -1;
      int fd = recvFdPacket(s, &owner, comm);
      if (owner <= 0 || owner >= nranks || seen[owner]) {
        abortWithMessage(comm, "invalid or duplicate rank in POSIX FD broker handshake");
      }
      seen[owner] = true;
      clients[owner] = s;
      fds[owner] = fd;
    }

    for (int r = 1; r < nranks; ++r) {
      for (int p = 0; p < nranks; ++p) {
        if (p == r) continue;
        sendFdPacket(clients[r], p, fds[p], comm);
      }
    }

    for (int r = 1; r < nranks; ++r) close(clients[r]);
    close(listenFd);
    unlink(socketPath);
  } else {
    int s = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (s < 0) abortWithMessage(comm, std::string("socket(AF_UNIX) failed: ") + std::strerror(errno));
    setCloseOnExec(s, comm);

    sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socketPath, sizeof(addr.sun_path) - 1);
    if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      abortWithMessage(comm, std::string("connect(") + socketPath + ") failed: " + std::strerror(errno));
    }

    sendFdPacket(s, rank, localFd, comm);
    for (int i = 0; i < nranks - 1; ++i) {
      int owner = -1;
      int fd = recvFdPacket(s, &owner, comm);
      if (owner < 0 || owner >= nranks || owner == rank || fds[owner] != -1) {
        abortWithMessage(comm, "invalid or duplicate FD received from POSIX FD broker");
      }
      fds[owner] = fd;
    }
    close(s);
  }

  for (int p = 0; p < nranks; ++p) {
    if (fds[p] < 0) abortWithMessage(comm, "POSIX FD exchange left a missing rank allocation");
  }
  return fds;
}
#endif

enum class IpcMode { Posix, Fabric };

inline IpcMode parseIpcMode(const std::string& text) {
  if (text == "posix") return IpcMode::Posix;
  if (text == "fabric") return IpcMode::Fabric;
  throw std::runtime_error("--vmm-ipc must be 'posix' or 'fabric'");
}

template <class Scalar>
class DistributedVmmArena {
 public:
  DistributedVmmArena(std::size_t logicalLocalCount, IpcMode ipcMode,
                      const VmmComm& comm)
      : comm_(comm), ipcMode_(ipcMode), logicalLocalCount_(logicalLocalCount) {
#ifndef HAVE_MPI
    throw std::runtime_error("VMM distributed arena requires MPI in this experiment");
#else
    rank_ = comm_.rank();
    size_ = comm_.size();

    VMM_CU_CHECK(comm_, cuInit(0));
    int runtimeDev = 0;
    VMM_CUDART_CHECK(comm_, cudaGetDevice(&runtimeDev));
    VMM_CU_CHECK(comm_, cuDeviceGet(&cuDevice_, runtimeDev));

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
    VMM_CU_CHECK(comm_, cuMemGetAllocationGranularity(&localGran, &prop,
                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    unsigned long long localGranUll = static_cast<unsigned long long>(localGran);
    std::vector<unsigned long long> allGran(size_);
    MPI_Allgather(&localGranUll, 1, MPI_UNSIGNED_LONG_LONG,
                  allGran.data(), 1, MPI_UNSIGNED_LONG_LONG, comm_.mpi());

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
                  bytesByRankUll.data(), 1, MPI_UNSIGNED_LONG_LONG, comm_.mpi());

    bytesByRank_.resize(size_);
    baseElements_.resize(size_);
    std::uint64_t runningBytes = 0;
    for (int p = 0; p < size_; ++p) {
      bytesByRank_[p] = static_cast<std::size_t>(bytesByRankUll[p]);
      if (runningBytes % sizeof(Scalar) != 0) abortWithMessage(comm_, "VMM byte offset is not Scalar-aligned");
      baseElements_[p] = runningBytes / sizeof(Scalar);
      runningBytes += bytesByRank_[p];
    }
    totalBytes_ = static_cast<std::size_t>(runningBytes);
    totalElements_ = runningBytes / sizeof(Scalar);

    VMM_CU_CHECK(comm_, cuMemCreate(&localHandle_, localBytes_, &prop, 0));

    handles_.resize(size_);
    handles_[rank_] = localHandle_;

    if (ipcMode_ == IpcMode::Posix) {
      int localFd = -1;
      VMM_CU_CHECK(comm_, cuMemExportToShareableHandle(&localFd, localHandle_,
                                                 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
      std::vector<int> fds = exchangePosixFds(localFd, comm_);

      for (int p = 0; p < size_; ++p) {
        if (p == rank_) continue;
        const int fd = fds[p];
        if (fcntl(fd, F_GETFD) == -1) {
          abortWithMessage(comm_, "received POSIX FD is invalid before CUDA import");
        }

        // NOTE: On the H200 system used to develop this experiment, the
        // working import convention matches NVIDIA NCCL: encode the numeric
        // FD value in the void* argument.  This differs from some CUDA guide
        // examples that show &fd.
        VMM_CU_CHECK(comm_, cuMemImportFromShareableHandle(
            &handles_[p], reinterpret_cast<void*>(static_cast<uintptr_t>(fd)),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
      }

      MPI_Barrier(comm_.mpi());
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
      VMM_CU_CHECK(comm_, cuMemExportToShareableHandle(&localFabric, localHandle_,
                                                 CU_MEM_HANDLE_TYPE_FABRIC, 0));
      std::vector<CUmemFabricHandle> allFabric(size_);
      MPI_Allgather(&localFabric, sizeof(CUmemFabricHandle), MPI_BYTE,
                    allFabric.data(), sizeof(CUmemFabricHandle), MPI_BYTE,
                    comm_.mpi());
      for (int p = 0; p < size_; ++p) {
        if (p == rank_) continue;
        VMM_CU_CHECK(comm_, cuMemImportFromShareableHandle(
            &handles_[p], &allFabric[p], CU_MEM_HANDLE_TYPE_FABRIC));
      }
#else
      throw std::runtime_error("fabric IPC not compiled with this CUDA version");
#endif
    }

    VMM_CU_CHECK(comm_, cuMemAddressReserve(&baseVa_, totalBytes_, granularity_, 0, 0));

    std::uint64_t byteOffset = 0;
    for (int p = 0; p < size_; ++p) {
      VMM_CU_CHECK(comm_, cuMemMap(baseVa_ + byteOffset, bytesByRank_[p], 0, handles_[p], 0));
      byteOffset += bytesByRank_[p];
    }

    CUmemAccessDesc access {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cuDevice_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    VMM_CU_CHECK(comm_, cuMemSetAccess(baseVa_, totalBytes_, &access, 1));

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
  VmmComm comm_;
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

}  // namespace VmmExperiment

#endif  // HAVE_TPETRA_INST_CUDA
#endif  // TPETRA_EXPERIMENT_VMM_DISTRIBUTED_ARENA_HPP

