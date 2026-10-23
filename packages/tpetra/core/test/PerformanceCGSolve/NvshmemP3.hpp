// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P3: one GPU-initiated NVSHMEM double get per remote column-map entry.
// Reuse P1's actual (sourcePE, sourceLID, destinationLID) descriptors.
// Reuse P2's independent CUDA-space Distributor import for the MPI oracle.
// No NVSHMEM allocation is ever passed to MPI/Tpetra::Distributor.

#ifndef TPETRA_PERFORMANCE_NVSHMEM_P3_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P3_HPP

#include "NvshmemP0.hpp"
#include "NvshmemP2.hpp"
#include "ImportGatherPlan.hpp"
#include "Tpetra_Access.hpp"
#include "Tpetra_Vector.hpp"
#include "Teuchos_OrdinalTraits.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <nvshmem.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace NvshmemP3 {

inline void check(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error("NVSHMEM P3: " + message);
}
inline void mpiCheck(int error, const char* operation) {
  if (error != MPI_SUCCESS)
    throw std::runtime_error(std::string("NVSHMEM P3: MPI error in ") + operation);
}

struct Options {
  bool runSynthetic = true;
  bool benchmarkBaselines = true;
  bool dumpPeerStats = false;  // setup diagnostics; no effect on timed regions
  int warmup = 2;
  int repetitions = 10;
};
inline void checkOptions(const Options& opts) {
  check(opts.warmup >= 0 && opts.repetitions > 0,
        "warmup must be nonnegative and repetitions must be positive");
  check(static_cast<std::size_t>(opts.repetitions) <=
            static_cast<std::size_t>(std::numeric_limits<int>::max() / 3),
        "too many repetitions for MPI timing reduction");
}

template<class T>
Kokkos::View<T*, Kokkos::CudaSpace>
upload(const std::string& name, const std::vector<T>& source) {
  static_assert(std::is_trivially_copyable<T>::value,
                "P3 gather descriptors must be trivially copyable");
  Kokkos::View<T*, Kokkos::CudaSpace> device(name, source.size());
  if (!source.empty()) {
    Kokkos::View<T*, Kokkos::HostSpace> host(name + " host", source.size());
    for (std::size_t i = 0; i < source.size(); ++i) host(i) = source[i];
    Kokkos::deep_copy(device, host);
  }
  return device;
}

struct Timings {
  double local = 0.0;
  double remote = 0.0;
  double total = 0.0; // from ready symmetric X to completed column-map target
};

// NVSHMEM's symmetric allocation is a separate copy of stock Tpetra X.
// The maximum rank-local length is allocated collectively across all PEs.
// Local and remote plan entries are uploaded to *ordinary* Kokkos CudaSpace.
// In contrast to the P2 Distributor, neither the symmetric source pointer
// nor any of its contents is passed to MPI.
template<class LO, class GO, class Node>
class Gather {
 public:
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  using Plan = ImportGatherP1::ImportGatherPlan<LO, GO, Node>;
  using LocalEntry = ImportGatherP1::LocalCopyEntry;
  using RemoteEntry = ImportGatherP1::RemoteGatherEntry;
  using SymmetricView = Kokkos::View<
      double*, Kokkos::CudaSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  Gather(const Plan& plan, const NvshmemP0::Session& session)
      : sourceLength_(plan.sourceLength), targetLength_(plan.targetLength),
        numLocal_(plan.local.size()), numRemote_(plan.remote.size()) {
    check(plan.sourceLengthByRank.size() == static_cast<std::size_t>(session.numPes()),
          "P1 rank-local source lengths are unavailable");
    for (const auto& e : plan.local) {
      check(e.sourceLid < sourceLength_ && e.destinationLid < targetLength_,
            "P1 local-copy descriptor outside a Map");
    }
    for (const auto& e : plan.remote) {
      check(e.ownerRank >= 0 && e.ownerRank < session.numPes() &&
            e.ownerRank != session.mpiRank() &&
            e.sourcePe >= 0 && e.sourcePe < session.numPes() &&
            session.mpiRankForPe(e.sourcePe) == e.ownerRank &&
            e.sourceLid < plan.sourceLengthByRank[static_cast<std::size_t>(e.ownerRank)] &&
            e.destinationLid < targetLength_,
            "invalid P1 remote source PE/source LID/destination LID");
    }

    const unsigned long long localLength =
        static_cast<unsigned long long>(sourceLength_);
    unsigned long long maximum = 0;
    mpiCheck(MPI_Allreduce(&localLength, &maximum, 1,
                           MPI_UNSIGNED_LONG_LONG, MPI_MAX, session.comm()),
             "MPI_Allreduce(P3 maximum owned length)");
    maximum = std::max(1ULL, maximum);
    check(maximum <= static_cast<unsigned long long>(
                         std::numeric_limits<std::size_t>::max() / sizeof(double)),
          "collective symmetric allocation size overflows size_t");
    paddedLength_ = static_cast<std::size_t>(maximum);

    // Identical byte count and collective order on all PEs. NVSHMEM VMM
    // allocations are never touched by the MPI payload path.
    symmetric_ = static_cast<double*>(nvshmem_malloc(paddedLength_ * sizeof(double)));
    check(symmetric_ != nullptr, "nvshmem_malloc failed");
    localDevice_ = upload("P3 local LID descriptors", plan.local);
    remoteDevice_ = upload("P3 remote PE/LID descriptors", plan.remote);
    Kokkos::fence("P3 gather descriptors on CUDA");
  }

  Gather(const Gather&) = delete;
  Gather& operator=(const Gather&) = delete;

  ~Gather() {
    // The CGSolve driver immediately calls MPI_Abort on an exception. Do not
    // enter collective nvshmem_free() during stack unwinding on one rank.
    if (symmetric_ && std::uncaught_exceptions() == 0) {
      Kokkos::fence("P3 destructor: GPU remote gets finished");
      nvshmem_free(symmetric_);
    }
  }

  std::size_t sourceLength() const { return sourceLength_; }
  std::size_t paddedLength() const { return paddedLength_; }
  std::size_t localCount() const { return numLocal_; }
  std::size_t remoteCount() const { return numRemote_; }

  // CUDA kernel-launching member functions MUST remain PUBLIC: NVCC requires
  // the enclosing function of the extended KOKKOS_LAMBDA to be accessible.
  // This preserves the fix-p2-private-cuda-lambda requirement.
  void cloneCuda(const Vector& X) const {
    check(X.getMap()->getLocalNumElements() == sourceLength_,
          "source Vector has an unexpected local length");
    const auto n = sourceLength_;
    const SymmetricView symmetric(symmetric_, paddedLength_);
    {
      auto source = X.getLocalViewDevice(Tpetra::Access::ReadOnly);
      Kokkos::parallel_for(
          "P3 clone ordinary Tpetra CUDA X to symmetric X",
          Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, n),
          KOKKOS_LAMBDA(const std::size_t i) {
            symmetric(i) = source(i, 0);
          });
      Kokkos::fence("P3 source CUDA clone completed");
    }
  }

  // Call on ALL PEs after their producer kernels have completed. Source
  // readiness belongs outside the gather-only hot-path measurement.
  void publishReady() const { nvshmem_barrier_all(); }

  // Source is symmetric and remains immutable while any rank is gathering.
  // Each destination LID is written exactly once across the local and remote
  // descriptor lists already exhaustively validated in P1.
  void copyLocalCuda(Vector& Y) const {
    if (numLocal_ == 0) return;
    const auto local = localDevice_;
    const SymmetricView symmetric(symmetric_, paddedLength_);
    {
      auto target = Y.getLocalViewDevice(Tpetra::Access::ReadWrite);
      Kokkos::parallel_for(
          "P3 place local column-map LIDs from symmetric X",
          Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, numLocal_),
          KOKKOS_LAMBDA(const std::size_t i) {
            const LocalEntry e = local(i);
            target(e.destinationLid, 0) = symmetric(e.sourceLid);
          });
      Kokkos::fence("P3 local column-map copies completed");
    }
  }

  void gatherRemoteCuda(Vector& Y) const {
    if (numRemote_ == 0) return;
    const auto remote = remoteDevice_;
    const double* symmetricBase = symmetric_;
    {
      auto target = Y.getLocalViewDevice(Tpetra::Access::ReadWrite);
      Kokkos::parallel_for(
          "P3 GPU-initiated 8-byte NVSHMEM double gets",
          Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, numRemote_),
          KOKKOS_LAMBDA(const std::size_t i) {
            const RemoteEntry e = remote(i);
            target(e.destinationLid, 0) =
                nvshmem_double_g(symmetricBase + e.sourceLid, e.sourcePe);
          });
      Kokkos::fence("P3 NVSHMEM scalar gets completed on CUDA");
    }
  }

  Timings gatherPrepared(Vector& Y) const {
    const double started = MPI_Wtime();
    copyLocalCuda(Y);
    const double copied = MPI_Wtime();
    gatherRemoteCuda(Y);
    const double gathered = MPI_Wtime();
    return Timings{copied - started, gathered - copied, gathered - started};
  }

  // Invoke collectively after all PEs have completed every remote get, and
  // before any PE can leave the symmetric source's lifetime. Not timed.
  void release() {
    if (!symmetric_) return;
    Kokkos::fence("P3 all remote GPU gets finished before symmetric free");
    nvshmem_barrier_all();
    nvshmem_free(symmetric_);
    symmetric_ = nullptr;
  }

 private:
  std::size_t sourceLength_ = 0, targetLength_ = 0;
  std::size_t paddedLength_ = 0, numLocal_ = 0, numRemote_ = 0;
  double* symmetric_ = nullptr;
  Kokkos::View<LocalEntry*, Kokkos::CudaSpace> localDevice_;
  Kokkos::View<RemoteEntry*, Kokkos::CudaSpace> remoteDevice_;
};

// First mismatch includes the Tpetra GID and the remote PE/source LID.
// P3 output must agree exactly with BOTH the Tpetra and P2 results.
template<class LO, class GO, class Node>
void assertMatchesBoth(const std::string& name,
                       const ImportGatherP1::ImportGatherPlan<LO, GO, Node>& plan,
                       const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
                       const NvshmemP0::Session& session,
                       const Tpetra::Vector<double, LO, GO, Node>& reference,
                       const Tpetra::Vector<double, LO, GO, Node>& mpiReplica,
                       const Tpetra::Vector<double, LO, GO, Node>& nvshmemGather) {
  const auto ref = reference.getLocalViewHost(Tpetra::Access::ReadOnly);
  const auto mpi = mpiReplica.getLocalViewHost(Tpetra::Access::ReadOnly);
  const auto got = nvshmemGather.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (std::size_t dst = 0; dst < plan.targetLength; ++dst) {
    const double expected = ref(dst, 0);
    if (std::isfinite(expected) && mpi(dst, 0) == expected && got(dst, 0) == expected)
      continue;
    int sourcePe = session.pe();
    std::uint64_t sourceLid = 0;
    const char* category = "not_found";
    for (const auto& e : plan.local) {
      if (e.destinationLid == dst) {
        category = "local";
        sourceLid = e.sourceLid;
        break;
      }
    }
    for (const auto& e : plan.remote) {
      if (e.destinationLid == dst) {
        category = "remote";
        sourcePe = e.sourcePe;
        sourceLid = e.sourceLid;
        break;
      }
    }
    std::cerr << "[NVSHMEM P3 MISMATCH] case=" << name
              << " rank=" << session.mpiRank()
              << " pe=" << session.pe()
              << " destinationLID=" << dst
              << " GID=" << column->getGlobalElement(static_cast<LO>(dst))
              << " kind=" << category
              << " sourcePE=" << sourcePe
              << " sourceLID=" << sourceLid
              << " tpetra=" << expected
              << " p2=" << mpi(dst, 0)
              << " nvshmem=" << got(dst, 0) << "\n";
    throw std::runtime_error("NVSHMEM P3: " + name +
                             " full target disagrees at LID " + std::to_string(dst));
  }
}

// Single-shot clone/readiness timing includes the real rendezvous cost.
// The repeated prepared-gather timing below excludes it by design.
inline void reportFirstImport(const std::string& name,
                              const NvshmemP0::Session& session,
                              double clone, double ready, const Timings& t) {
  const double local[6] = {clone, ready, t.local, t.remote, t.total,
                            clone + ready + t.total};
  double maxRank[6] = {0, 0, 0, 0, 0, 0};
  mpiCheck(MPI_Allreduce(local, maxRank, 6, MPI_DOUBLE, MPI_MAX, session.comm()),
           "MPI_Allreduce(P3 first import stages)");
  if (session.mpiRank() != 0) return;
  const auto flags = std::cout.flags();
  const auto precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(2)
            << "[NVSHMEM P3 SETUP] case=" << name
            << " scope=max_rank single_shot=1 unit=us"
            << " clone=" << maxRank[0] * 1.e6
            << " readiness=" << maxRank[1] * 1.e6
            << " local=" << maxRank[2] * 1.e6
            << " remote=" << maxRank[3] * 1.e6
            << " prepared_total=" << maxRank[4] * 1.e6
            << " with_clone_total=" << maxRank[5] * 1.e6 << "\n";
  std::cout.flags(flags);
  std::cout.precision(precision);
}

template<class Action>
void measurePrepared(const std::string& name, const Options& opts,
                     const NvshmemP0::Session& session, Action action) {
  for (int i = 0; i < opts.warmup; ++i) (void)action();
  std::vector<double> local(static_cast<std::size_t>(3 * opts.repetitions));
  for (int i = 0; i < opts.repetitions; ++i) {
    const Timings t = action();
    const std::size_t k = static_cast<std::size_t>(3 * i);
    local[k] = t.local;
    local[k + 1] = t.remote;
    local[k + 2] = t.total;
  }
  std::vector<double> maxRank(local.size());
  mpiCheck(MPI_Allreduce(local.data(), maxRank.data(),
                         static_cast<int>(maxRank.size()), MPI_DOUBLE,
                         MPI_MAX, session.comm()),
           "MPI_Allreduce(P3 prepared timing)");
  if (session.mpiRank() != 0) return;
  const auto flags = std::cout.flags();
  const auto precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(2)
            << "[NVSHMEM P3 TIMING] case=" << name
            << " mode=prepared_symmetric_double_g"
            << " scope=max_rank_per_iteration unit=us"
            << " reps=" << opts.repetitions << " warmup=" << opts.warmup;
  const char* labels[3] = {"local", "remote", "total"};
  for (int field = 0; field < 3; ++field) {
    std::vector<double> samples(static_cast<std::size_t>(opts.repetitions));
    for (int i = 0; i < opts.repetitions; ++i)
      samples[static_cast<std::size_t>(i)] =
          maxRank[static_cast<std::size_t>(3 * i + field)] * 1.e6;
    const auto q = NvshmemP2::summarize(std::move(samples));
    std::cout << " " << labels[field] << "_median=" << q.median;
    if (field == 2)
      std::cout << " total_p10=" << q.p10 << " total_p90=" << q.p90;
  }
  std::cout << "\n";
  std::cout.flags(flags);
  std::cout.precision(precision);
}

// Collect the actual P1 Importer remote descriptors into a directed
// receiver-rank x source-rank table. The reported volume is unique remote
// *column-Map values* (one 8-byte scalar per element), NOT matrix nonzeros
// or InfiniBand wire overhead. This is setup diagnostics, outside timers.
// The hostnames come from the actual MPI processes, not guessed rank layout.
template<class LO, class GO, class Node>
void dumpPeerStats(const std::string& caseName,
                   const ImportGatherP1::ImportGatherPlan<LO, GO, Node>& plan,
                   const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
                   const NvshmemP0::Session& session) {
  const int nRanks = session.numPes();
  const int me = session.mpiRank();
  check(nRanks > 0, "invalid MPI rank count in P3 peer diagnostics");
  std::vector<unsigned long long> fromOwner(static_cast<std::size_t>(nRanks), 0ULL);
  for (const auto& e : plan.remote) {
    check(e.ownerRank >= 0 && e.ownerRank < nRanks, "invalid peer in P3 descriptors");
    ++fromOwner[static_cast<std::size_t>(e.ownerRank)];
  }

  std::vector<unsigned long long> matrix(
      static_cast<std::size_t>(nRanks) * static_cast<std::size_t>(nRanks), 0ULL);
  mpiCheck(MPI_Allgather(fromOwner.data(), nRanks, MPI_UNSIGNED_LONG_LONG,
                         matrix.data(), nRanks, MPI_UNSIGNED_LONG_LONG, session.comm()),
           "MPI_Allgather(P3 actual receiver/source peer matrix)");

  char myHostname[MPI_MAX_PROCESSOR_NAME] = {};
  int hostnameLength = 0;
  mpiCheck(MPI_Get_processor_name(myHostname, &hostnameLength),
           "MPI_Get_processor_name(P3 peer diagnostics)");
  std::vector<char> allHostnames(
      static_cast<std::size_t>(nRanks) * MPI_MAX_PROCESSOR_NAME, '\0');
  mpiCheck(MPI_Allgather(myHostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         allHostnames.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         session.comm()),
           "MPI_Allgather(P3 peer hostnames)");

  // If any nonadjacent rank requests values from rank zero, print the actual
  // GID/source-LID here. This diagnoses the extra 15 values in the size-100
  // result without assuming what produced those references.
  for (const auto& e : plan.remote) {
    if (e.ownerRank == 0 && me > 1) {
      std::cout << "[NVSHMEM P3 PEER GID] case=" << caseName
                << " receiver=" << me << " source=0"
                << " gid=" << column->getGlobalElement(static_cast<LO>(e.destinationLid))
                << " sourceLID=" << e.sourceLid
                << " destinationLID=" << e.destinationLid << "\n";
    }
  }
  if (me != 0) return;

  unsigned long long total = 0, sameNode = 0, otherNode = 0;
  std::size_t nonzeroPairs = 0;
  std::cout << "[NVSHMEM P3 PEER MATRIX] case=" << caseName
            << " receiver=rows source=columns unit=doubles source=data=P1_remote_descriptors\n";
  for (int dst = 0; dst < nRanks; ++dst) {
    const std::string dstHost(
        &allHostnames[static_cast<std::size_t>(dst) * MPI_MAX_PROCESSOR_NAME]);
    for (int src = 0; src < nRanks; ++src) {
      const auto n = matrix[static_cast<std::size_t>(dst) * nRanks + src];
      if (n == 0) continue;
      const std::string srcHost(
          &allHostnames[static_cast<std::size_t>(src) * MPI_MAX_PROCESSOR_NAME]);
      const bool localNode = dstHost == srcHost;
      ++nonzeroPairs;
      total += n;
      if (localNode) sameNode += n;
      else otherNode += n;
      std::cout << "[NVSHMEM P3 PEER] case=" << caseName
                << " source=" << src << " receiver=" << dst
                << " doubles=" << n << " bytes=" << (n * sizeof(double))
                << " locality=" << (localNode ? "same-node" : "cross-node")
                << " sourceHost=" << srcHost << " receiverHost=" << dstHost << "\n";
    }
  }
  std::cout << "[NVSHMEM P3 PEER TOTAL] case=" << caseName
            << " remoteDoubles=" << total
            << " packedPayloadBytes=" << (total * sizeof(double))
            << " directedPairs=" << nonzeroPairs
            << " sameNodeDoubles=" << sameNode
            << " crossNodeDoubles=" << otherNode
            << " sameNodeBytes=" << (sameNode * sizeof(double))
            << " crossNodeBytes=" << (otherNode * sizeof(double)) << "\n";
}

template<class LO, class GO, class Node>
void runCase(const std::string& name,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domain,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
             const Teuchos::RCP<const Tpetra::Import<LO, GO, Node>>& importer,
             const NvshmemP0::Session& session, const Options& opts,
             bool requireRemote, bool benchmark) {
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  checkOptions(opts);
  check(!importer.is_null(), name + ": missing Importer");
  NvshmemP1::ValueEncoding encoding(domain->getLocalNumElements(), session);
  std::vector<int> peForRank(static_cast<std::size_t>(session.numPes()), -1);
  for (int pe = 0; pe < session.numPes(); ++pe)
    peForRank[static_cast<std::size_t>(session.mpiRankForPe(pe))] = pe;
  const auto plan = ImportGatherP1::buildImportGatherPlan(
      domain, column, *importer, peForRank, session.comm());
  StandaloneMpiP2::ImportReplica<LO, GO, Node> replica(domain, column, *importer);

  Vector X(domain);
  Vector X_reference(column);
  Vector X_mpi(column);
  Vector X_nvshmem(column);
  NvshmemP2::fillOwned(X, encoding, session);
  X_reference.doImport(X, *importer, Tpetra::INSERT);
  NvshmemP2::verifyOracle(name, plan, encoding, session, X_reference);
  (void)replica.import(X, X_mpi);
  NvshmemP2::assertEqual(name + " P2", X_reference, X_mpi);

  Gather<LO, GO, Node> gather(plan, session);
  // Initialize the entire target once to expose any missing local or remote
  // writes. Resetting it is excluded from all gather timings.
  X_nvshmem.putScalar(std::numeric_limits<double>::quiet_NaN());
  const double beforeClone = MPI_Wtime();
  gather.cloneCuda(X);
  const double afterClone = MPI_Wtime();
  gather.publishReady();  // all producers finished before any remote reader
  const double afterReady = MPI_Wtime();
  const Timings first = gather.gatherPrepared(X_nvshmem);
  assertMatchesBoth(name, plan, column, session, X_reference, X_mpi, X_nvshmem);

  const unsigned long long counts[2] = {
      static_cast<unsigned long long>(plan.remote.size()),
      static_cast<unsigned long long>(plan.exportLids.size())};
  unsigned long long global[2] = {0ULL, 0ULL};
  mpiCheck(MPI_Allreduce(counts, global, 2, MPI_UNSIGNED_LONG_LONG,
                         MPI_SUM, session.comm()), "MPI_Allreduce(P3 counts)");
  check(global[0] == global[1], name + ": total remote/export counts differ");
  check(!requireRemote || global[0] > 0, name + ": expected remote entries");
  if (opts.dumpPeerStats && benchmark)
    dumpPeerStats(name, plan, column, session);

  std::cout << "[NVSHMEM P3] case=" << name
            << " rank=" << session.mpiRank()
            << " pe=" << session.pe()
            << " owned=" << plan.sourceLength
            << " padded=" << gather.paddedLength()
            << " target=" << plan.targetLength
            << " local=" << plan.local.size()
            << " remote=" << plan.remote.size()
            << " exports=" << plan.exportLids.size()
            << " sourcePEs=" << plan.numRemotePeers
            << " remoteBytes=" << counts[0] * sizeof(double)
            << " sourceMemory=NVSHMEM_symmetric"
            << " transport=GPU_double_g"
            << " oracle=Tpetra_and_P2"
            << " status=PASS\n";
  if (session.mpiRank() == 0)
    std::cout << "[NVSHMEM P3] case=" << name
              << " globalRemote=" << global[0]
              << " globalExports=" << global[1]
              << " globalRemoteBytes=" << global[0] * sizeof(double)
              << " status=PASS\n";

  if (benchmark) {
    reportFirstImport(name, session, afterClone - beforeClone,
                      afterReady - afterClone, first);
    if (opts.benchmarkBaselines) {
      NvshmemP2::Options p2opts;
      p2opts.warmup = opts.warmup;
      p2opts.repetitions = opts.repetitions;
      NvshmemP2::measure(name, "tpetra_doImport", p2opts, session, [&]() {
        StandaloneMpiP2::Timings t;
        const double started = MPI_Wtime();
        X_reference.doImport(X, *importer, Tpetra::INSERT);
        Kokkos::fence("P3 Tpetra comparison import completed");
        t.total = MPI_Wtime() - started;
        return t;
      });
      NvshmemP2::measure(name, "cuda_pack_distributor_cuda_unpack", p2opts,
                         session, [&]() { return replica.import(X, X_mpi); });
    }
    // The symmetric copy is immutable for all warmup and timed repeats.
    // There is NO repeated clone and NO barrier inside this hot-path loop.
    measurePrepared(name, opts, session,
                    [&]() { return gather.gatherPrepared(X_nvshmem); });
    assertMatchesBoth(name + " timed", plan, column, session,
                      X_reference, X_mpi, X_nvshmem);
  }

  // Every PE participates before any symmetric source is deallocated.
  gather.release();
}

// Same five cases as the passing P2 test. P3 does not modify P2's test
// implementation; it builds equivalent synthetic Maps locally.
enum class SyntheticKind { Mixed, OneWay, LocalOnly, Interleaved };

template<class LO, class GO, class Node>
void runSynthetic(SyntheticKind kind, const NvshmemP0::Session& session,
                  const Options& opts) {
  using Map = Tpetra::Map<LO, GO, Node>;
  using Import = Tpetra::Import<LO, GO, Node>;
  const int rank = session.mpiRank();
  const int nRanks = session.numPes();
  check(static_cast<long long>(nRanks) * 1024 + 100 <
            static_cast<long long>(std::numeric_limits<GO>::max()),
        "too many ranks for the synthetic noncontiguous GIDs");
  const int nOwned = 3 + rank % 3;
  Teuchos::Array<GO> ownGids;
  for (int i = 0; i < nOwned; ++i)
    ownGids.push_back(NvshmemP1::syntheticGid<GO>(rank, i));
  Teuchos::Array<GO> targetGids;
  targetGids.push_back(ownGids[0]);
  if (nRanks > 1) {
    const int next = (rank + 1) % nRanks;
    if (kind == SyntheticKind::Mixed)
      targetGids.push_back(NvshmemP1::syntheticGid<GO>(next, 1));
    else if (kind == SyntheticKind::OneWay && rank == 0)
      targetGids.push_back(NvshmemP1::syntheticGid<GO>(1, 1));
    else if (kind == SyntheticKind::Interleaved)
      targetGids.push_back(NvshmemP1::syntheticGid<GO>(next, 2));
  }
  targetGids.push_back(ownGids[nOwned - 1]);
  if (kind == SyntheticKind::Interleaved && nRanks > 1) {
    const int other = nRanks > 2 ? (rank + 2) % nRanks : (rank + 1) % nRanks;
    targetGids.push_back(NvshmemP1::syntheticGid<GO>(other, 1));
  }
  for (int i = nOwned - 2; i >= 1; --i) targetGids.push_back(ownGids[i]);

  const auto comm = Tpetra::getDefaultComm();
  const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  auto own = Teuchos::rcp(new Map(invalid, ownGids(), static_cast<GO>(7), comm));
  auto tgt = Teuchos::rcp(new Map(invalid, targetGids(), static_cast<GO>(7), comm));
  Teuchos::RCP<const Map> domain = own;
  Teuchos::RCP<const Map> column = tgt;
  Teuchos::RCP<const Import> importer = Teuchos::rcp(new Import(domain, column));
  std::string name;
  if (kind == SyntheticKind::Mixed) name = "synthetic-mixed";
  else if (kind == SyntheticKind::OneWay) name = "synthetic-one-way";
  else if (kind == SyntheticKind::LocalOnly) name = "synthetic-local-only";
  else name = "synthetic-interleaved";
  runCase(name, domain, column, importer, session, opts,
          kind != SyntheticKind::LocalOnly && nRanks > 1, false);
}

}  // namespace NvshmemP3

#endif // TPETRA_PERFORMANCE_NVSHMEM_P3_HPP
