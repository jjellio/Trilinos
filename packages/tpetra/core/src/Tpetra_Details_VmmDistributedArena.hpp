// Experimental CUDA VMM arena extracted from PerformanceCGSolve.
// Owns only the VMM allocation / mapping and MPI-scoped IPC resources.
#ifndef TPETRA_EXPERIMENT_VMM_DISTRIBUTED_ARENA_HPP
#define TPETRA_EXPERIMENT_VMM_DISTRIBUTED_ARENA_HPP

#ifdef HAVE_TPETRA_INST_CUDA

#include "Tpetra_Details_VmmLayout.hpp"
#include <Kokkos_Core.hpp>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

namespace VmmExperiment {

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

template <class Scalar>
class DistributedVmmArena {
 public:
  // Compatibility constructor for the current SpmvContext. The new layout
  // constructor below allows the matrix and arena to compute layout separately.
  DistributedVmmArena(std::size_t logicalLocalCount, IpcMode ipcMode,
                      const VmmComm& comm)
      : DistributedVmmArena(VmmLayout<Scalar>::compute(logicalLocalCount,
                                                       ipcMode, comm)) {}

  explicit DistributedVmmArena(VmmLayout<Scalar> layout)
      : layout_(std::move(layout)), comm_(layout_.comm()),
        ipcMode_(layout_.ipcMode()) {
#ifndef HAVE_MPI
    throw std::runtime_error("VMM distributed arena requires MPI in this experiment");
#else
    rank_ = layout_.rank();
    size_ = layout_.size();
    cuDevice_ = layout_.device();

    // A layout can be computed ahead of arena construction. Verify that the
    // active GPU has not changed before allocating with its stored device id.
    VMM_CU_CHECK(comm_, cuInit(0));
    int runtimeDev = 0;
    VMM_CUDART_CHECK(comm_, cudaGetDevice(&runtimeDev));
    CUdevice activeDevice = 0;
    VMM_CU_CHECK(comm_, cuDeviceGet(&activeDevice, runtimeDev));
    if (activeDevice != cuDevice_) {
      abortWithMessage(comm_, "VMM layout was computed for a different CUDA device");
    }

    CUmemAllocationProp prop {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cuDevice_;
    prop.requestedHandleTypes = layout_.handleType();

    // Layout is already finalized: no rank counts, CUDA granularities,
    // padding, or address offsets are recomputed by the arena.
    VMM_CU_CHECK(comm_, cuMemCreate(&localHandle_, layout_.localBytes(), &prop, 0));

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

    VMM_CU_CHECK(comm_, cuMemAddressReserve(&baseVa_, layout_.totalBytes(), layout_.granularity(), 0, 0));

    std::uint64_t byteOffset = 0;
    for (int p = 0; p < size_; ++p) {
      VMM_CU_CHECK(comm_, cuMemMap(baseVa_ + byteOffset, layout_.bytesByRank()[p], 0, handles_[p], 0));
      byteOffset += layout_.bytesByRank()[p];
    }

    CUmemAccessDesc access {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cuDevice_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    VMM_CU_CHECK(comm_, cuMemSetAccess(baseVa_, layout_.totalBytes(), &access, 1));

    globalPtr_ = reinterpret_cast<Scalar*>(baseVa_);
    localPtr_  = globalPtr_ + layout_.baseElements()[rank_];

    if (rank_ == 0) {
      std::cout << "VMM arena: granularity=" << layout_.granularity()
                << " bytes, total mapped=" << layout_.totalBytes()
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
      CUresult e = cuMemUnmap(baseVa_ + byteOffset, layout_.bytesByRank()[p]);
      if (e != CUDA_SUCCESS) {
        const char* s = nullptr;
        cuGetErrorString(e, &s);
        std::fprintf(stderr, "rank %d: cuMemUnmap failed during teardown: %s\n",
                     rank_, s ? s : "unknown");
      }
      byteOffset += layout_.bytesByRank()[p];
    }
    cuMemAddressFree(baseVa_, layout_.totalBytes());
    baseVa_ = 0;

    for (auto h : handles_) cuMemRelease(h);
#endif
  }

  DistributedVmmArena(const DistributedVmmArena&) = delete;
  DistributedVmmArena& operator=(const DistributedVmmArena&) = delete;

  Scalar* globalPtr() const { return globalPtr_; }
  Scalar* localPtr() const { return localPtr_; }
  std::size_t logicalLocalCount() const { return layout_.logicalLocalCount(); }
  std::uint64_t totalElements() const { return layout_.totalElements(); }
  const std::vector<std::uint64_t>& baseElements() const { return layout_.baseElements(); }
  const VmmLayout<Scalar>& layout() const { return layout_; }
  int rank() const { return rank_; }
  int size() const { return size_; }

 private:
  VmmLayout<Scalar> layout_;
  VmmComm comm_;
  IpcMode ipcMode_;
  int rank_ = 0;
  int size_ = 1;
  CUdevice cuDevice_ = 0;
  CUmemGenericAllocationHandle localHandle_ {};
  std::vector<CUmemGenericAllocationHandle> handles_;
  CUdeviceptr baseVa_ = 0;
  Scalar* globalPtr_ = nullptr;
  Scalar* localPtr_ = nullptr;
};

}  // namespace VmmExperiment

#endif  // HAVE_TPETRA_INST_CUDA
#endif  // TPETRA_EXPERIMENT_VMM_DISTRIBUTED_ARENA_HPP

