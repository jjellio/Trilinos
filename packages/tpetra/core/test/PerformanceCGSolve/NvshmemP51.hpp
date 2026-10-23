// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P5.1 only: build/validate read-only sender/receiver PUT metadata.
// Does not allocate symmetric buffers, transfer vector values, issue RMA,
// launch CUDA kernels, or make throughput claims.

#ifndef TPETRA_PERFORMANCE_NVSHMEM_P51_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P51_HPP

#include "NvshmemP4.hpp"  // reuse already accepted synthetic Map cases
#include "PackedPutPlanMpi.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace NvshmemP51 {

struct Options {
  std::string only = "all";
  bool showPeers = true;
  int minimumNodes = 1;
  int maximumNodes = 0;
  bool requireCrossNode = false;
};

struct Suite {
  int passed = 0;
  int skipped = 0;
  std::uint64_t allExports = 0;
  std::uint64_t allRemote = 0;
  std::uint64_t allSendPeers = 0;
  std::uint64_t crossNodeElements = 0;
};

inline void check(bool valid, const std::string& what) {
  if (!valid) throw std::runtime_error("NVSHMEM P5.1: " + what);
}

inline void checkOptions(const Options& opts) {
  const auto& cases = NvshmemP4::cases();
  check(opts.only == "all" ||
        std::find(cases.begin(), cases.end(), opts.only) != cases.end(),
        "unknown --nvshmem-p51-case: " + opts.only);
  check(opts.minimumNodes >= 1 &&
            (opts.maximumNodes == 0 || opts.maximumNodes >= opts.minimumNodes),
        "invalid requested minimum/maximum physical node count");
}

template<class LO, class GO, class Node>
void runCase(const std::string& name,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domain,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
             const Teuchos::RCP<const Tpetra::Import<LO, GO, Node>>& importer,
             const NvshmemP0::Session& session,
             const NvshmemP4::Topology& topology,
             const Options& options, Suite& suite, bool needRemote = false) {
  check(!importer.is_null(), name + ": missing Importer");
  std::vector<int> rankToPe(static_cast<std::size_t>(session.numPes()), -1);
  for (int pe = 0; pe < session.numPes(); ++pe)
    rankToPe[static_cast<std::size_t>(session.mpiRankForPe(pe))] = pe;
  const auto p1 = ImportGatherP1::buildImportGatherPlan(
      domain, column, *importer, rankToPe, session.comm());
  const auto plan = PackedPutP51::build(p1, session);
  check(plan.finalized && plan.packElements == p1.exportLids.size() &&
            plan.inboxElements == p1.remote.size() &&
            plan.local.size() + plan.unpackDestinationLids.size() ==
                p1.targetLength,
        name + ": P5.1 plan does not reproduce P1 destination partition");

  std::uint64_t localStats[5] = {
      plan.packElements, plan.inboxElements,
      static_cast<std::uint64_t>(plan.sendPeers.size()),
      static_cast<std::uint64_t>(plan.receivePeers.size()), 0};
  for (const auto& peer : plan.sendPeers) {
    if (topology.names[static_cast<std::size_t>(peer.rank)] !=
        topology.names[static_cast<std::size_t>(session.mpiRank())])
      localStats[4] = PackedPutP51::add(localStats[4], peer.count,
                                         "cross-node exported elements");
  }
  std::uint64_t globalStats[5] = {};
  PackedPutP51::mpiCheck(
      MPI_Allreduce(localStats, globalStats, 5, MPI_UINT64_T, MPI_SUM,
                    session.comm()), "MPI_Allreduce(P5.1 plan accounting)");
  check(globalStats[0] == globalStats[1] &&
            globalStats[2] == globalStats[3],
        name + ": global incoming/outgoing packet counts disagree");
  if (needRemote)
    check(globalStats[1] > 0, name + ": expected actual remote requests");
  if (name == "all-empty" || name == "identity" || name == "permute")
    check(globalStats[0] == 0 && globalStats[1] == 0,
          name + ": unexpected remote communication");
  if (name == "zero-owner" && session.mpiRank() == 0)
    check(plan.ownedLength == 0 && plan.packElements == 0 &&
              plan.inboxElements == 2,
          "zero-owner did not exercise a receive-only rank");
  if (name == "one-way") {
    if (session.mpiRank() == 0)
      check(plan.packElements == 0 && plan.inboxElements == 1,
            "one-way receiver has incorrect packet plan");
    if (session.mpiRank() == 1)
      check(plan.packElements == 1 && plan.inboxElements == 0,
            "one-way sender has incorrect packet plan");
  }
  if (name == "fanout")
    check(globalStats[2] == static_cast<std::uint64_t>(session.numPes() - 1),
          "fanout did not preserve one packet to each receiving rank");

  // Rank-by-rank diagnostic ordering makes independent cluster logs usable.
  for (int rank = 0; rank < session.numPes(); ++rank) {
    PackedPutP51::mpiCheck(MPI_Barrier(session.comm()),
                            "MPI_Barrier(P5.1 ordered diagnostics)");
    if (session.mpiRank() != rank) continue;
    std::cout << "[NVSHMEM P5.1 RANK] case=" << name
              << " rank=" << rank << " pe=" << session.pe()
              << " local=" << plan.local.size()
              << " exports=" << plan.packElements
              << " imports=" << plan.inboxElements
              << " sendPeers=" << plan.sendPeers.size()
              << " receivePeers=" << plan.receivePeers.size()
              << " packCapacity=" << plan.symmetricPackElements
              << " inboxCapacity=" << plan.symmetricInboxElements
              << " signalSlots=" << plan.symmetricSignalElements
              << " status=PASS\n";
    if (options.showPeers) {
      for (const auto& peer : plan.sendPeers) {
        std::cout << "[NVSHMEM P5.1 SEND] case=" << name
                  << " rank=" << rank << " dstRank=" << peer.rank
                  << " dstPe=" << peer.pe
                  << " count=" << peer.count << " bytes=" << peer.count * sizeof(double)
                  << " packOffset=" << peer.packOffset
                  << " remoteInboxOffset=" << peer.remoteInboxOffset
                  << " remoteSignalIndex=" << peer.remoteSignalIndex << "\n";
      }
      for (const auto& peer : plan.receivePeers) {
        std::cout << "[NVSHMEM P5.1 RECV] case=" << name
                  << " rank=" << rank << " srcRank=" << peer.rank
                  << " srcPe=" << peer.pe
                  << " count=" << peer.count << " bytes=" << peer.count * sizeof(double)
                  << " inboxOffset=" << peer.inboxOffset
                  << " localSignalIndex=" << peer.localSignalIndex << "\n";
      }
    }
    std::cout.flush();
  }
  PackedPutP51::mpiCheck(MPI_Barrier(session.comm()),
                          "MPI_Barrier(P5.1 end diagnostics)");
  suite.passed++;
  suite.allExports = PackedPutP51::add(suite.allExports, globalStats[0], "suite exports");
  suite.allRemote = PackedPutP51::add(suite.allRemote, globalStats[1], "suite remote");
  suite.allSendPeers = PackedPutP51::add(suite.allSendPeers, globalStats[2], "suite peers");
  suite.crossNodeElements = PackedPutP51::add(
      suite.crossNodeElements, globalStats[4], "suite cross-node elements");
  if (session.mpiRank() == 0)
    std::cout << "[NVSHMEM P5.1 CASE] case=" << name
              << " globalExports=" << globalStats[0]
              << " globalImports=" << globalStats[1]
              << " senderPeerRelationships=" << globalStats[2]
              << " receiverPeerRelationships=" << globalStats[3]
              << " crossNodeElements=" << globalStats[4]
              << " planOnly=YES rmaIssued=0 performanceMeasured=NO"
              << " status=PASS\n";
}

template<class LO, class GO, class Node>
void run(const NvshmemP0::Session& session, const Options& opts,
         int nsize, const std::string& filename) {
  checkOptions(opts);
  const auto topology = NvshmemP4::getTopology(session);
  check(topology.distinctNodes >= opts.minimumNodes,
        "fewer physical nodes than --nvshmem-p51-min-nodes");
  check(opts.maximumNodes == 0 || topology.distinctNodes <= opts.maximumNodes,
        "more physical nodes than --nvshmem-p51-max-nodes");
  const auto comm = Tpetra::getDefaultComm();
  check(comm->getRank() == session.mpiRank() &&
            comm->getSize() == session.numPes(),
        "Tpetra and session MPI communicator ranks differ");

  Suite suite;
  if (session.mpiRank() == 0)
    std::cout << "[NVSHMEM P5.1] BEGIN ranks=" << session.numPes()
              << " nodes=" << topology.distinctNodes
              << " planOnly=YES rmaIssued=0 performanceMeasured=NO\n";
  for (const auto& name : NvshmemP4::cases()) {
    if (opts.only != "all" && opts.only != name) continue;
    if (name == "matrix") {
      using Matrix = Tpetra::CrsMatrix<double, LO, GO, Node>;
      using Map = Tpetra::Map<LO, GO, Node>;
      using Import = Tpetra::Import<LO, GO, Node>;
      Teuchos::RCP<Matrix> A;
      if (filename.empty())
        A = Tpetra::Utils::MatrixGenerator<Matrix>::generate_miniFE_matrix(nsize, comm);
      else
        A = Tpetra::MatrixMarket::Reader<Matrix>::readSparseFile(filename, comm);
      check(A->isFillComplete(), "matrix must be fill complete");
      Teuchos::RCP<const Map> domain = A->getDomainMap();
      Teuchos::RCP<const Map> column = A->getColMap();
      Teuchos::RCP<const Import> importer = NvshmemP1::ensureImporter(
          domain, column, A->getGraph()->getImporter(), session.comm());
      runCase(name, domain, column, importer, session, topology, opts, suite,
              filename.empty() && session.numPes() > 1);
    } else {
      if (session.numPes() < NvshmemP4::minimumPes(name)) {
        ++suite.skipped;
        if (session.mpiRank() == 0)
          std::cout << "[NVSHMEM P5.1] case=" << name << " status=SKIP"
                    << " reason=requires_" << NvshmemP4::minimumPes(name)
                    << "_PEs\n";
        continue;
      }
      check(static_cast<std::uint64_t>(session.numPes()) <=
                (static_cast<std::uint64_t>(std::numeric_limits<GO>::max()) - 20000ULL) /
                    1000000ULL,
            "GO range is too small for P4 sparse Map GIDs");
      using Map = Tpetra::Map<LO, GO, Node>;
      using Import = Tpetra::Import<LO, GO, Node>;
      const auto spec = NvshmemP4::buildSpec<GO>(name, session.mpiRank(),
                                                 session.numPes());
      const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
      Teuchos::RCP<const Map> domain =
          Teuchos::rcp(new Map(invalid, spec.owned(), static_cast<GO>(0), comm));
      Teuchos::RCP<const Map> target =
          Teuchos::rcp(new Map(invalid, spec.target(), static_cast<GO>(0), comm));
      Teuchos::RCP<const Import> importer =
          Teuchos::rcp(new Import(domain, target));
      runCase(name, domain, target, importer, session, topology, opts, suite);
    }
  }
  check(suite.passed + suite.skipped > 0, "no P5.1 cases selected");
  if (opts.requireCrossNode)
    check(suite.crossNodeElements > 0,
          "no actual inter-node sender/receiver packet relationship found");
  if (session.mpiRank() == 0)
    std::cout << "[NVSHMEM P5.1] SUMMARY status="
              << (suite.passed == 0 ? "SKIP" : "PASS")
              << " casesPassed=" << suite.passed
              << " casesSkipped=" << suite.skipped
              << " globalExports=" << suite.allExports
              << " globalImports=" << suite.allRemote
              << " senderPeerRelationships=" << suite.allSendPeers
              << " crossNodeElements=" << suite.crossNodeElements
              << " planOnly=YES rmaIssued=0 performanceMeasured=NO\n";
}

} // namespace NvshmemP51

#endif // TPETRA_PERFORMANCE_NVSHMEM_P51_HPP
