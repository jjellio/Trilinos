// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P4: correctness and robustness only.  No timings and no performance gates.
// Reuse the production P3 Gather, the P2 MPI Distributor replica, and the
// stock Tpetra doImport oracle.  Exercise changing source values, symmetric
// memory refresh/retirement, and difficult Map ownership/target layouts.

#ifndef TPETRA_PERFORMANCE_NVSHMEM_P4_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P4_HPP

#include "NvshmemP3.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "TpetraUtils_MatrixGenerator.hpp"
#include "MatrixMarket_Tpetra.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace NvshmemP4 {

inline void check(bool good, const std::string& what) {
  if (!good) throw std::runtime_error("NVSHMEM P4: " + what);
}
inline void mpiCheck(int value, const char* operation) {
  if (value != MPI_SUCCESS)
    throw std::runtime_error(std::string("NVSHMEM P4 MPI failure in ") + operation);
}

struct Options {
  std::string only = "all";       // all, matrix, or a named synthetic case
  int epochs = 4;                 // changed source values every epoch
  int cycles = 2;                 // repeated symmetric malloc/free lifetimes
  int minimumNodes = 1;          // opt-in physical topology verification
  int maximumNodes = 0;          // 0 means unlimited; 1 requires one physical node
  bool requireCrossNode = false;  // require genuine inter-node remote reads
  bool negativeChecks = true;     // reject invalid descriptors BEFORE RMA
};

inline const std::vector<std::string>& cases() {
  static const std::vector<std::string> names = {
      "matrix", "identity", "permute", "sparse-mixed", "one-way", "uneven",
      "zero-owner", "all-empty", "multi-owner", "fanout"};
  return names;
}
inline void checkOptions(const Options& opts) {
  check(opts.epochs >= 2 && opts.epochs <= 1000,
        "--nvshmem-p4-epochs must be within [2,1000]");
  check(opts.cycles >= 2 && opts.cycles <= 100,
        "--nvshmem-p4-cycles must be within [2,100]");
  check(opts.minimumNodes >= 1, "minimum node count must be positive");
  check(opts.maximumNodes == 0 || opts.maximumNodes >= opts.minimumNodes,
        "maximum node count must be 0 (unlimited) or >= minimum node count");
  check(opts.only == "all" ||
            std::find(cases().begin(), cases().end(), opts.only) != cases().end(),
        "unknown --nvshmem-p4-case: " + opts.only);
}

struct Topology {
  std::vector<std::string> names;
  int distinctNodes = 0;
};
inline Topology getTopology(const NvshmemP0::Session& session) {
  char hostname[MPI_MAX_PROCESSOR_NAME] = {};
  int length = 0;
  mpiCheck(MPI_Get_processor_name(hostname, &length), "MPI_Get_processor_name");
  check(length > 0 && length < MPI_MAX_PROCESSOR_NAME, "invalid MPI processor name");
  std::vector<char> names(static_cast<std::size_t>(session.numPes()) *
                          MPI_MAX_PROCESSOR_NAME, '\0');
  mpiCheck(MPI_Allgather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, session.comm()),
           "MPI_Allgather(P4 processor names)");
  Topology topology;
  std::set<std::string> unique;
  for (int rank = 0; rank < session.numPes(); ++rank) {
    const char* name = names.data() + static_cast<std::size_t>(rank) * MPI_MAX_PROCESSOR_NAME;
    topology.names.emplace_back(name);
    unique.insert(name);
  }
  topology.distinctNodes = static_cast<int>(unique.size());
  return topology;
}

// Source GIDs are intentionally sparse, scrambled in LID order, and never
// equal to owner-local LIDs. The rank ranges are disjoint, also when rank 0
// has no ownership. There is no contiguous-global-index assumption.
template<class GO>
GO gid(int rank, int lid) {
  const long long value = 113LL + static_cast<long long>(rank) * 1000000LL +
                          ((static_cast<long long>(lid) * 37LL + 11LL) % 101LL) * 13LL;
  return static_cast<GO>(value);
}

inline int countOwned(const std::string& name, int rank) {
  if (name == "all-empty") return 0;
  if (name == "zero-owner") return rank == 0 ? 0 : 3 + rank % 4;
  if (name == "uneven") return rank == 0 ? 1 : rank == 1 ? 23 : 2 + rank % 3;
  if (name == "identity") return 3 + rank % 3;
  if (name == "permute") return 4 + rank % 3;
  if (name == "sparse-mixed") return 5 + rank % 4;
  if (name == "one-way") return 3 + rank % 2;
  if (name == "multi-owner") return 4 + rank % 4;
  if (name == "fanout") return 4 + rank % 3;
  throw std::runtime_error("NVSHMEM P4: unknown synthetic case " + name);
}

inline int minimumPes(const std::string& name) {
  if (name == "sparse-mixed" || name == "one-way" || name == "uneven" ||
      name == "zero-owner") return 2;
  if (name == "multi-owner" || name == "fanout") return 3;
  return 1;
}

template<class GO>
struct MapSpec {
  Teuchos::Array<GO> owned;
  Teuchos::Array<GO> target;
};

template<class GO>
MapSpec<GO> buildSpec(const std::string& name, int rank, int nRanks) {
  MapSpec<GO> result;
  const int nOwned = countOwned(name, rank);
  for (int i = 0; i < nOwned; ++i) result.owned.push_back(gid<GO>(rank, i));
  if (name == "all-empty" || (name == "zero-owner" && rank == 0)) {
    if (name == "zero-owner") {
      result.target.push_back(gid<GO>(1, 0));
      result.target.push_back(gid<GO>(1, countOwned(name, 1) - 1));
    }
  } else if (name == "identity") {
    result.target = result.owned;
  } else {
    result.target.push_back(result.owned[0]);
    // Interleave remote entries with local permutations, rather than
    // assuming that remote destination slots form a contiguous suffix.
    if (name == "sparse-mixed") {
      const int next = (rank + 1) % nRanks;
      const int prev = (rank + nRanks - 1) % nRanks;
      result.target.push_back(gid<GO>(next, countOwned(name, next) - 1));
      result.target.push_back(result.owned[nOwned - 1]);
      result.target.push_back(gid<GO>(prev, 0));
      for (int i = nOwned - 2; i >= 1; --i) result.target.push_back(result.owned[i]);
    } else if (name == "multi-owner") {
      const int next = (rank + 1) % nRanks;
      const int other = (rank + 2) % nRanks;
      result.target.push_back(gid<GO>(next, 1));
      result.target.push_back(result.owned[nOwned - 1]);
      result.target.push_back(gid<GO>(other, 2));
      for (int i = nOwned - 2; i >= 1; --i) result.target.push_back(result.owned[i]);
    } else if (name == "one-way" && rank == 0) {
      result.target.push_back(gid<GO>(1, 1));
      for (int i = nOwned - 1; i >= 1; --i) result.target.push_back(result.owned[i]);
    } else if (name == "uneven") {
      if (rank == 0) {
        result.target.push_back(gid<GO>(1, 22));
        result.target.push_back(gid<GO>(1, 0));
        result.target.push_back(gid<GO>(1, 11));
      } else {
        result.target.push_back(gid<GO>(rank == 1 ? 0 : 1,
                                          rank == 1 ? 0 : 22));
        for (int i = nOwned - 1; i >= 1; --i) result.target.push_back(result.owned[i]);
      }
    } else if (name == "fanout" && rank != 0) {
      result.target.push_back(gid<GO>(0, 1));
      for (int i = nOwned - 1; i >= 1; --i) result.target.push_back(result.owned[i]);
    } else {
      for (int i = nOwned - 1; i >= 1; --i) result.target.push_back(result.owned[i]);
    }
  }
  std::set<GO> localUnique(result.owned.begin(), result.owned.end());
  std::set<GO> targetUnique(result.target.begin(), result.target.end());
  check(localUnique.size() == result.owned.size(), name + ": duplicate owned GID");
  check(targetUnique.size() == result.target.size(), name + ": duplicate target GID");
  return result;
}

// Fractional, signed and epoch-distinct finite doubles, represented exactly
// in binary64. The encoded tuple allows independent owner/LID validation;
// it cannot accidentally pass by comparing two identically stale buffers.
struct Pattern {
  std::uint64_t stride = 0;
  std::uint64_t rankCount = 0;
  static constexpr std::uint64_t maxExactInteger = (1ULL << 53) - 1ULL;

  Pattern(std::size_t owned, const NvshmemP0::Session& session, int epochs) {
    const unsigned long long mine = static_cast<unsigned long long>(owned);
    unsigned long long maximum = 0;
    mpiCheck(MPI_Allreduce(&mine, &maximum, 1, MPI_UNSIGNED_LONG_LONG,
                           MPI_MAX, session.comm()), "MPI_Allreduce(P4 maximum source length)");
    check(maximum < maxExactInteger, "P4 source length exceeds exact double encoding");
    stride = maximum + 1;
    rankCount = static_cast<std::uint64_t>(session.numPes());
    check(rankCount <= maxExactInteger / stride &&
              static_cast<std::uint64_t>(epochs) <=
                  maxExactInteger / (stride * rankCount),
          "P4 epoch/rank/LID values exceed exact double integer representation");
  }

  double value(int epoch, int ownerRank, std::uint64_t lid) const {
    const std::uint64_t id =
        (static_cast<std::uint64_t>(epoch) * rankCount +
         static_cast<std::uint64_t>(ownerRank)) * stride + lid + 1ULL;
    const double fraction = static_cast<double>(id) * 0.125; // exact binary fraction
    return ((static_cast<std::uint64_t>(epoch) + ownerRank + lid) & 1ULL)
        ? -fraction : fraction;
  }
};

template<class LO, class GO, class Node>
void fillOwned(Tpetra::Vector<double, LO, GO, Node>& X,
               const Pattern& pattern, const NvshmemP0::Session& session, int epoch) {
  const std::size_t n = X.getMap()->getLocalNumElements();
  const std::uint64_t rank = static_cast<std::uint64_t>(session.mpiRank());
  const std::uint64_t base =
      (static_cast<std::uint64_t>(epoch) * pattern.rankCount + rank) * pattern.stride;
  {
    auto owned = X.getLocalViewDevice(Tpetra::Access::OverwriteAll);
    Kokkos::parallel_for("P4 distinct signed fractional source values",
      Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<std::size_t>>(0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        const double magnitude = static_cast<double>(base + i + 1ULL) * 0.125;
        owned(i, 0) = ((static_cast<std::uint64_t>(epoch) +
                         rank + i) & 1ULL)
            ? -magnitude : magnitude;
      });
    Kokkos::fence("P4 new source values visible in ordinary CUDA storage");
  }
}

// Fail before making a remote get; never attempt an intentionally invalid
// NVSHMEM device address. Both mutations must throw on EVERY rank before the
// first collective in the Gather constructor, so no rank can deadlock here.
template<class LO, class GO, class Node>
void rejectBadDescriptors(const ImportGatherP1::ImportGatherPlan<LO, GO, Node>& plan,
                          const NvshmemP0::Session& session) {
  typedef NvshmemP3::Gather<LO, GO, Node> Gather;
  {
    auto bad = plan;
    bad.local.push_back(ImportGatherP1::LocalCopyEntry{
        static_cast<std::uint64_t>(bad.sourceLength), 0ULL});
    bool rejected = false;
    try { Gather invalid(bad, session); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "out-of-range local LID was accepted");
  }
  {
    auto bad = plan;
    bad.remote.push_back(ImportGatherP1::RemoteGatherEntry{
        session.numPes(), 0, 0ULL, 0ULL});
    bool rejected = false;
    try { Gather invalid(bad, session); }
    catch (const std::runtime_error&) { rejected = true; }
    check(rejected, "invalid owner rank was accepted");
  }
}

template<class LO, class GO, class Node>
void verify(const std::string& name, int cycle, int epoch,
            const Pattern& pattern,
            const ImportGatherP1::ImportGatherPlan<LO, GO, Node>& plan,
            const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& target,
            const NvshmemP0::Session& session,
            const Tpetra::Vector<double, LO, GO, Node>& Xref,
            const Tpetra::Vector<double, LO, GO, Node>& Xmpi,
            const Tpetra::Vector<double, LO, GO, Node>& Xnvshmem) {
  // No Kokkos device view of any of these Tpetra vectors remains alive.
  const auto ref = Xref.getLocalViewHost(Tpetra::Access::ReadOnly);
  const auto mpi = Xmpi.getLocalViewHost(Tpetra::Access::ReadOnly);
  const auto shmem = Xnvshmem.getLocalViewHost(Tpetra::Access::ReadOnly);
  std::vector<unsigned char> checked(plan.targetLength, 0);
  const auto visit = [&](std::size_t destination, int owner, std::uint64_t lid,
                         const char* kind) {
    check(destination < plan.targetLength && checked[destination] == 0,
          name + ": duplicate/out-of-range destination in P1 plan");
    checked[destination] = 1;
    const double expected = pattern.value(epoch, owner, lid);
    if (std::isfinite(ref(destination, 0)) &&
        ref(destination, 0) == expected &&
        mpi(destination, 0) == expected &&
        shmem(destination, 0) == expected) return;
    std::cerr << "[NVSHMEM P4 MISMATCH] case=" << name
              << " rank=" << session.mpiRank() << " cycle=" << cycle
              << " epoch=" << epoch << " kind=" << kind
              << " destinationLID=" << destination
              << " GID=" << target->getGlobalElement(static_cast<LO>(destination))
              << " ownerRank=" << owner << " sourceLID=" << lid
              << " expected=" << expected
              << " tpetra=" << ref(destination, 0)
              << " p2=" << mpi(destination, 0)
              << " p3=" << shmem(destination, 0) << "\n";
    throw std::runtime_error("NVSHMEM P4: " + name +
                             " mismatch at destination " + std::to_string(destination));
  };
  for (const auto& e : plan.local)
    visit(static_cast<std::size_t>(e.destinationLid), session.mpiRank(),
          e.sourceLid, "local");
  for (const auto& e : plan.remote)
    visit(static_cast<std::size_t>(e.destinationLid), e.ownerRank,
          e.sourceLid, "remote");
  for (std::size_t i = 0; i < checked.size(); ++i)
    check(checked[i] == 1, name + ": missing target destination " + std::to_string(i));
}

struct SuiteStats {
  int passed = 0;
  int skipped = 0;
  bool negativeDone = false;
  unsigned long long crossNodeReads = 0;
  unsigned long long remoteReads = 0;
};

template<class LO, class GO, class Node>
void runCase(const std::string& name,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domain,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
             const Teuchos::RCP<const Tpetra::Import<LO, GO, Node>>& importer,
             const NvshmemP0::Session& session,
             const Topology& topology, const Options& opts, SuiteStats& suite,
             bool requireRemote = false) {
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  check(!importer.is_null(), name + ": importer must be valid");
  std::vector<int> rankToPe(static_cast<std::size_t>(session.numPes()), -1);
  for (int pe = 0; pe < session.numPes(); ++pe)
    rankToPe[static_cast<std::size_t>(session.mpiRankForPe(pe))] = pe;
  const auto plan = ImportGatherP1::buildImportGatherPlan(
      domain, column, *importer, rankToPe, session.comm());
  if (opts.negativeChecks && !suite.negativeDone) {
    rejectBadDescriptors(plan, session);
    suite.negativeDone = true;
    if (session.mpiRank() == 0)
      std::cout << "[NVSHMEM P4] case=invalid-descriptors"
                   " checks=local_LID,remote_owner status=PASS\n";
  }
  const Pattern pattern(plan.sourceLength, session, opts.epochs);
  StandaloneMpiP2::ImportReplica<LO, GO, Node> mpiReplica(
      domain, column, *importer);
  Vector X(domain), Xref(column), Xmpi(column), Xnvshmem(column);

  // This is correctness-only. A cycle owns a distinct symmetric allocation;
  // epochs within a cycle change source values in the SAME allocation.
  const double sentinel = std::numeric_limits<double>::quiet_NaN();
  for (int cycle = 0; cycle < opts.cycles; ++cycle) {
    NvshmemP3::Gather<LO, GO, Node> gather(plan, session);
    for (int epoch = 0; epoch < opts.epochs; ++epoch) {
      fillOwned(X, pattern, session, epoch);
      Xref.putScalar(sentinel);
      Xmpi.putScalar(sentinel);
      Xnvshmem.putScalar(sentinel);
      Xref.doImport(X, *importer, Tpetra::INSERT);
      (void)mpiReplica.import(X, Xmpi);
      gather.cloneCuda(X);
      gather.publishReady();
      // Call the existing P3 CUDA operations, not the timed P3 wrapper.
      gather.copyLocalCuda(Xnvshmem);
      gather.gatherRemoteCuda(Xnvshmem);
      verify(name, cycle, epoch, pattern, plan, column,
             session, Xref, Xmpi, Xnvshmem);
      // GPU kernel completion is local. Before the *next* epoch may overwrite
      // its symmetric source, ALL other PEs must have completed prior reads.
      nvshmem_barrier_all();
    }
    // release() also establishes a collective retirement barrier before
    // nvshmem_free; repeat to catch symmetric allocation lifetime problems.
    gather.release();
  }

  const unsigned long long localCounts[5] = {
      static_cast<unsigned long long>(plan.sourceLength),
      static_cast<unsigned long long>(plan.targetLength),
      static_cast<unsigned long long>(plan.remote.size()),
      static_cast<unsigned long long>(plan.exportLids.size()),
      static_cast<unsigned long long>(plan.numPermute)};
  unsigned long long totalCounts[5] = {0, 0, 0, 0, 0};
  mpiCheck(MPI_Allreduce(localCounts, totalCounts, 5, MPI_UNSIGNED_LONG_LONG,
                         MPI_SUM, session.comm()), "MPI_Allreduce(P4 case counts)");
  check(totalCounts[2] == totalCounts[3], name + ": remote/export totals differ");
  unsigned long long localCross = 0;
  for (const auto& e : plan.remote) {
    if (topology.names[static_cast<std::size_t>(e.ownerRank)] !=
        topology.names[static_cast<std::size_t>(session.mpiRank())]) ++localCross;
  }
  unsigned long long globalCross = 0;
  mpiCheck(MPI_Allreduce(&localCross, &globalCross, 1, MPI_UNSIGNED_LONG_LONG,
                         MPI_SUM, session.comm()), "MPI_Allreduce(P4 cross-node reads)");
  suite.crossNodeReads += globalCross;
  suite.remoteReads += totalCounts[2];
  if (name == "identity" || name == "permute" || name == "all-empty")
    check(totalCounts[2] == 0, name + ": unexpected remote communication");
  if (name == "one-way" || name == "fanout" || name == "zero-owner") {
    const unsigned long long wanted = name == "one-way" ? 1ULL :
        name == "zero-owner" ? 2ULL : static_cast<unsigned long long>(session.numPes() - 1);
    check(totalCounts[2] == wanted, name + ": unexpected remote count");
  }
  if (name == "multi-owner") {
    unsigned long long localPeers = plan.numRemotePeers;
    unsigned long long minPeers = 0;
    mpiCheck(MPI_Allreduce(&localPeers, &minPeers, 1, MPI_UNSIGNED_LONG_LONG,
                           MPI_MIN, session.comm()), "MPI_Allreduce(P4 minimum source peers)");
    check(minPeers >= 2, "multi-owner did not exercise two distinct source PEs per rank");
  }
  if (name == "all-empty")
    check(totalCounts[0] == 0 && totalCounts[1] == 0, "all-empty Map is not empty");
  if (name == "permute" || name == "sparse-mixed" || name == "multi-owner")
    check(totalCounts[4] > 0, name + ": expected genuine local permutations");
  if (name == "zero-owner" && session.mpiRank() == 0)
    check(plan.sourceLength == 0 && plan.local.empty() && plan.remote.size() == 2,
          "zero-owner rank 0 did not exercise receive-only with zero local source");
  if (name == "one-way") {
    if (session.mpiRank() == 0)
      check(plan.remote.size() == 1 && plan.exportLids.empty(),
            "one-way rank 0 must receive without exporting");
    if (session.mpiRank() == 1)
      check(plan.remote.empty() && plan.exportLids.size() == 1,
            "one-way rank 1 must export without receiving");
  }
  if (name == "fanout") {
    for (const auto& e : plan.remote)
      check(session.mpiRank() != 0 && e.ownerRank == 0 && e.sourceLid == 1,
            "fanout must read the identical rank-0 source LID on every receiver");
  }
  const unsigned long long mine = static_cast<unsigned long long>(plan.sourceLength);
  unsigned long long minimumOwned = 0, maximumOwned = 0;
  mpiCheck(MPI_Allreduce(&mine, &minimumOwned, 1, MPI_UNSIGNED_LONG_LONG,
                         MPI_MIN, session.comm()), "MPI_Allreduce(P4 minimum owned)");
  mpiCheck(MPI_Allreduce(&mine, &maximumOwned, 1, MPI_UNSIGNED_LONG_LONG,
                         MPI_MAX, session.comm()), "MPI_Allreduce(P4 maximum owned)");
  if (name == "uneven" || name == "zero-owner")
    check(maximumOwned > minimumOwned, name + ": source lengths were unexpectedly equal");
  if ((requireRemote || name == "sparse-mixed" || name == "uneven") &&
      session.numPes() > 1)
    check(totalCounts[2] > 0, name + ": expected nonzero remote imports");

  ++suite.passed;
  if (session.mpiRank() == 0) {
    std::cout << "[NVSHMEM P4] case=" << name
              << " ranks=" << session.numPes()
              << " epochs=" << opts.epochs
              << " cycles=" << opts.cycles
              << " globalOwned=" << totalCounts[0]
              << " globalTarget=" << totalCounts[1]
              << " minOwned=" << minimumOwned
              << " maxOwned=" << maximumOwned
              << " globalRemote=" << totalCounts[2]
              << " globalExports=" << totalCounts[3]
              << " globalPermute=" << totalCounts[4]
              << " crossNodeRemote=" << globalCross
              << " communication=" << (totalCounts[2] == 0 ? "NO_REMOTE_COMM" : "REMOTE")
              << " status=PASS\n";
  }
}

template<class LO, class GO, class Node>
void runSynthetic(const std::string& name, const NvshmemP0::Session& session,
                  const Topology& topology, const Options& opts, SuiteStats& suite) {
  using Map = Tpetra::Map<LO, GO, Node>;
  using Import = Tpetra::Import<LO, GO, Node>;
  if (session.numPes() < minimumPes(name)) {
    ++suite.skipped;
    if (session.mpiRank() == 0)
      std::cout << "[NVSHMEM P4] case=" << name << " status=SKIP"
                << " reason=requires_" << minimumPes(name) << "_PEs\n";
    return;
  }
  check(static_cast<std::uint64_t>(session.numPes()) <=
            (static_cast<std::uint64_t>(std::numeric_limits<GO>::max()) - 20000ULL) /
                1000000ULL,
        "global-ordinal range too small for P4 sparse GIDs");
  auto spec = buildSpec<GO>(name, session.mpiRank(), session.numPes());
  auto comm = Tpetra::getDefaultComm();
  const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  Teuchos::RCP<const Map> domain =
      Teuchos::rcp(new Map(invalid, spec.owned(), static_cast<GO>(0), comm));
  Teuchos::RCP<const Map> target =
      Teuchos::rcp(new Map(invalid, spec.target(), static_cast<GO>(0), comm));
  Teuchos::RCP<const Import> importer =
      Teuchos::rcp(new Import(domain, target));
  runCase(name, domain, target, importer, session, topology, opts, suite);
}

template<class LO, class GO, class Node>
void runMatrix(const NvshmemP0::Session& session, const Topology& topology,
               const Options& opts, SuiteStats& suite,
               int nsize, const std::string& filename) {
  using Matrix = Tpetra::CrsMatrix<double, LO, GO, Node>;
  using Map = Tpetra::Map<LO, GO, Node>;
  using Import = Tpetra::Import<LO, GO, Node>;
  auto comm = Tpetra::getDefaultComm();
  Teuchos::RCP<Matrix> A;
  if (!filename.empty())
    A = Tpetra::MatrixMarket::Reader<Matrix>::readSparseFile(filename, comm);
  else
    A = Tpetra::Utils::MatrixGenerator<Matrix>::generate_miniFE_matrix(nsize, comm);
  check(A->isFillComplete(), "matrix must be fill complete");
  Teuchos::RCP<const Map> domain = A->getDomainMap();
  Teuchos::RCP<const Map> column = A->getColMap();
  Teuchos::RCP<const Import> importer = NvshmemP1::ensureImporter(
      domain, column, A->getGraph()->getImporter(), session.comm());
  // An arbitrary MatrixMarket matrix may legitimately have no remote data.
  // The generated miniFE matrix must exercise communication for >=2 PEs.
  runCase("matrix", domain, column, importer, session, topology, opts, suite,
          filename.empty() && session.numPes() > 1);
}

template<class LO, class GO, class Node>
void run(const NvshmemP0::Session& session, const Options& opts,
         int nsize, const std::string& filename) {
  checkOptions(opts);
  const auto topology = getTopology(session);
  check(topology.distinctNodes >= opts.minimumNodes,
        "requested " + std::to_string(opts.minimumNodes) +
        " distinct physical nodes, but MPI reports " +
        std::to_string(topology.distinctNodes));
  check(opts.maximumNodes == 0 || topology.distinctNodes <= opts.maximumNodes,
        "requested at most " + std::to_string(opts.maximumNodes) +
        " distinct nodes, but MPI reports " + std::to_string(topology.distinctNodes));
  SuiteStats suite;
  if (session.mpiRank() == 0) {
    const char* transport = std::getenv("NVSHMEM_REMOTE_TRANSPORT");
    const char* ibgda = std::getenv("NVSHMEM_IB_ENABLE_IBGDA");
    std::cout << "[NVSHMEM P4] BEGIN ranks=" << session.numPes()
              << " nodes=" << topology.distinctNodes
              << " transport=" << (transport ? transport : "default")
              << " ibgda=" << (ibgda ? ibgda : "default")
              << " correctness_only=1\n";
  }
  for (const auto& name : cases()) {
    if (opts.only != "all" && opts.only != name) continue;
    if (name == "matrix")
      runMatrix<LO, GO, Node>(session, topology, opts, suite, nsize, filename);
    else
      runSynthetic<LO, GO, Node>(name, session, topology, opts, suite);
  }
  check(suite.passed + suite.skipped > 0, "no P4 cases selected");
  if (suite.passed == 0 && !opts.requireCrossNode) {
    if (session.mpiRank() == 0)
      std::cout << "[NVSHMEM P4] SUMMARY status=SKIP casesPassed=0"
                << " casesSkipped=" << suite.skipped
                << " reason=no_applicable_cases measuredPerformance=NO\n";
    return;
  }
  if (opts.requireCrossNode)
    check(suite.crossNodeReads > 0,
          "no inter-node remote scalar gets were tested; rank placement or cases are wrong");
  if (session.mpiRank() == 0)
    std::cout << "[NVSHMEM P4] SUMMARY status=PASS"
              << " casesPassed=" << suite.passed
              << " casesSkipped=" << suite.skipped
              << " epochs=" << opts.epochs
              << " cycles=" << opts.cycles
              << " crossNodeRemote=" << suite.crossNodeReads
              << " measuredPerformance=NO\n";
}

} // namespace NvshmemP4

#endif // TPETRA_PERFORMANCE_NVSHMEM_P4_HPP
