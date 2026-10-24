// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
// P5.2: general per-peer GPU packed PUT, generation/credit synchronization,
// and exact full-column Tpetra + P2 comparison. No timings/performance gates.

#ifndef TPETRA_PERFORMANCE_NVSHMEM_P52_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P52_HPP

#include "NvshmemP4.hpp"           // all ten accepted Map layouts and owner/LID encoding
#include "PackedPutPlanMpi.hpp"    // one-time P5.1 host-only metadata setup
#include "NvshmemPackedPut.hpp"   // RMA-only, no MPI within import()

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace NvshmemP52 {
using U64 = PackedPutP51::U64;
inline void require(bool good, const std::string& why) {
  if (!good) throw std::runtime_error("NVSHMEM P5.2: " + why);
}
inline void mpiCheck(int err, const char* op) {
  require(err == MPI_SUCCESS, std::string("MPI failure at ") + op);
}

struct Options {
  std::string only = "all";
  int epochs = 4;
  int cycles = 2;
  bool showPeers = true;
  int minimumNodes = 1;
  int maximumNodes = 0;
  bool requireCrossNode = false;
  int skewMillis = 0;  // deliberately delay rank 0 before its first PUT Import
};
inline void checkOptions(const Options& o) {
  const auto& cases = NvshmemP4::cases();
  require(o.only == "all" ||
              std::find(cases.begin(), cases.end(), o.only) != cases.end(),
          "unknown --nvshmem-p52-case " + o.only);
  require(o.epochs >= 4 && o.epochs <= 1000, "epochs must be [4,1000]");
  require(o.cycles >= 2 && o.cycles <= 100, "cycles must be [2,100]");
  require(o.minimumNodes >= 1 &&
              (o.maximumNodes == 0 || o.maximumNodes >= o.minimumNodes),
          "invalid --nvshmem-p52-min-nodes/max-nodes");
  require(o.skewMillis >= 0 && o.skewMillis <= 3000,
          "--nvshmem-p52-skew-ms must be [0,3000]");
}

struct Suite {
  int passed = 0;
  int skipped = 0;
  U64 putCalls = 0;
  U64 putElements = 0;
  U64 readyWaits = 0;
  U64 creditWaits = 0;
  U64 ackSignals = 0;
  U64 crossNodePutCalls = 0;
  U64 crossNodePutElements = 0;
};

// Independent source-pattern oracle. Checks all target LIDs, not just peer
// totals, against Tpetra doImport(), P2 ordinary CudaSpace MPI and GPU PUT.
template<class LO, class GO, class Node>
void verify(const std::string& name, int cycle, int epoch,
            const NvshmemP4::Pattern& pattern,
            const PackedPutP51::Plan& plan,
            const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
            const NvshmemP0::Session& session,
            const Tpetra::Vector<double, LO, GO, Node>& reference,
            const Tpetra::Vector<double, LO, GO, Node>& mpiResult,
            const Tpetra::Vector<double, LO, GO, Node>& putResult) {
  const auto ref = reference.getLocalViewHost(Tpetra::Access::ReadOnly);
  const auto mpi = mpiResult.getLocalViewHost(Tpetra::Access::ReadOnly);
  const auto put = putResult.getLocalViewHost(Tpetra::Access::ReadOnly);
  std::vector<unsigned char> visited(PackedPutP51::asSize(plan.targetLength, "P5.2 coverage"), 0);
  const auto one = [&](U64 destination, int owner, U64 lid, const char* kind) {
    require(destination < plan.targetLength &&
                visited[static_cast<std::size_t>(destination)] == 0,
            name + ": invalid/duplicate destination in original P1 plan");
    visited[static_cast<std::size_t>(destination)] = 1;
    const double expected = pattern.value(epoch, owner, lid);
    if (std::isfinite(ref(destination, 0)) &&
        expected == ref(destination, 0) &&
        expected == mpi(destination, 0) &&
        expected == put(destination, 0)) return;
    std::cerr << std::setprecision(17)
              << "[NVSHMEM P5.2 MISMATCH] case=" << name
              << " rank=" << session.mpiRank() << " pe=" << session.pe()
              << " cycle=" << cycle << " epoch=" << epoch
              << " kind=" << kind << " destinationLID=" << destination
              << " GID=" << column->getGlobalElement(static_cast<LO>(destination))
              << " ownerRank=" << owner << " ownerLID=" << lid
              << " expected=" << expected << " Tpetra=" << ref(destination, 0)
              << " P2=" << mpi(destination, 0) << " P52=" << put(destination, 0) << '\n';
    throw std::runtime_error("NVSHMEM P5.2: " + name +
                             " wrong value at destination LID=" +
                             std::to_string(destination));
  };
  for (const auto& e : plan.local)
    one(e.destinationLid, session.mpiRank(), e.sourceLid, "local");
  for (const auto& e : plan.expectedRemote)
    one(e.destinationLid, e.ownerRank, e.sourceLid, "remote");
  for (const auto valid : visited)
    require(valid == 1, name + ": missing destination in verification");
}

template<class LO, class GO, class Node>
void runCase(const std::string& name,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domain,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
             const Teuchos::RCP<const Tpetra::Import<LO, GO, Node>>& importer,
             const NvshmemP0::Session& session,
             const NvshmemP4::Topology& topology,
             const Options& opts, Suite& suite, bool requireRemote = false) {
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  require(!importer.is_null(), name + ": no Importer");
  std::vector<int> rankToPe(static_cast<std::size_t>(session.numPes()), -1);
  for (int pe = 0; pe < session.numPes(); ++pe)
    rankToPe[static_cast<std::size_t>(session.mpiRankForPe(pe))] = pe;
  const auto p1 = ImportGatherP1::buildImportGatherPlan(
      domain, column, *importer, rankToPe, session.comm());
  const auto plan = PackedPutP51::build(p1, session);
  require(plan.finalized && plan.packElements == p1.exportLids.size() &&
          plan.inboxElements == p1.remote.size() &&
          plan.local.size() + plan.unpackDestinationLids.size() == p1.targetLength,
          name + ": P5.1 plan does not reproduce Importer");

  U64 crossCalls = 0, crossElements = 0;
  for (const auto& s : plan.sendPeers)
    if (topology.names[static_cast<std::size_t>(s.rank)] !=
        topology.names[static_cast<std::size_t>(session.mpiRank())]) {
      ++crossCalls;
      crossElements += s.count;
    }
  const U64 localPlan[5] = {
      plan.packElements, plan.inboxElements,
      static_cast<U64>(plan.sendPeers.size()),
      static_cast<U64>(plan.receivePeers.size()), crossElements};
  U64 globalPlan[5] = {};
  mpiCheck(MPI_Allreduce(localPlan, globalPlan, 5, MPI_UINT64_T,
                         MPI_SUM, session.comm()), "P5.2 setup packet accounting");
  require(globalPlan[0] == globalPlan[1] && globalPlan[2] == globalPlan[3],
          name + ": peer counts do not balance");
  if (requireRemote) require(globalPlan[0] > 0, name + ": missing matrix remote data");
  if (name == "identity" || name == "permute" || name == "all-empty")
    require(globalPlan[0] == 0 && globalPlan[2] == 0,
            name + ": local-only test has network packets");
  if (name == "one-way") {
    if (session.mpiRank() == 0)
      require(plan.sendPeers.empty() && plan.receivePeers.size() == 1,
              name + ": one-way receiver has an outgoing packet");
    if (session.mpiRank() == 1)
      require(plan.receivePeers.empty() && plan.sendPeers.size() == 1,
              name + ": one-way sender has an incoming packet");
  }
  if (name == "zero-owner" && session.mpiRank() == 0)
    require(plan.ownedLength == 0 && plan.packElements == 0 &&
            plan.inboxElements == 2, name + ": zero-owned rank case not exercised");
  if (name == "fanout")
    require(globalPlan[2] == static_cast<U64>(session.numPes() - 1),
            name + ": incorrect fanout peer count");

  if (opts.showPeers) {
    for (const auto& s : plan.sendPeers)
      std::cout << "[NVSHMEM P5.2 SEND] case=" << name
                << " rank=" << session.mpiRank() << " pe=" << session.pe()
                << " destRank=" << s.rank << " destPe=" << s.pe
                << " count=" << s.count << " packOffset=" << s.packOffset
                << " inboxOffset=" << s.remoteInboxOffset
                << " readySlot=" << s.remoteSignalIndex
                << " creditSlot=" << s.pe << '\n';
    for (const auto& r : plan.receivePeers)
      std::cout << "[NVSHMEM P5.2 RECV] case=" << name
                << " rank=" << session.mpiRank() << " pe=" << session.pe()
                << " sourceRank=" << r.rank << " sourcePe=" << r.pe
                << " count=" << r.count << " inboxOffset=" << r.inboxOffset
                << " readySlot=" << r.localSignalIndex
                << " remoteCreditSlot=" << session.pe() << '\n';
  }

  const int totalEpochs = opts.epochs * opts.cycles;
  const NvshmemP4::Pattern pattern(
      PackedPutP51::asSize(plan.ownedLength, "P5.2 source size"),
      session, totalEpochs);
  StandaloneMpiP2::ImportReplica<LO, GO, Node> mpiReplica(domain, column, *importer);
  Vector X(domain), reference(column), mpiResult(column), putResult(column);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  U64 localStats[13] = {}; // sum over cycles, counted in import(), not estimated
  std::size_t symmetricBytes = 0;
  for (int cycle = 0; cycle < opts.cycles; ++cycle) {
    // Four collective symmetric allocations only at this lifecycle boundary.
    NvshmemPackedPut::Exchange<LO, GO, Node> exchange(plan, session);
    symmetricBytes = exchange.symmetricBytes();
    for (int localEpoch = 0; localEpoch < opts.epochs; ++localEpoch) {
      const int valueEpoch = cycle * opts.epochs + localEpoch;
      NvshmemP4::fillOwned(X, pattern, session, valueEpoch);
      reference.putScalar(nan);
      mpiResult.putScalar(nan);
      putResult.putScalar(nan);
      reference.doImport(X, *importer, Tpetra::INSERT);
      (void)mpiReplica.import(X, mpiResult); // ordinary MPI CUDA-space buffers ONLY

      // Optional skew tests arrival before the receiver enters its Import.
      // It is placed AFTER both MPI reference operations, so P5.2 remains
      // responsible for completion, with no implicit MPI synchronization.
      if (cycle == 0 && localEpoch == 0 && session.mpiRank() == 0 &&
          opts.skewMillis > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(opts.skewMillis));
      exchange.import(X, putResult); // NVSHMEM only; zero global barriers
      require(exchange.generation() == static_cast<U64>(localEpoch + 1),
              name + ": generation did not advance exactly once");
      verify(name, cycle, valueEpoch, pattern, plan, column, session,
             reference, mpiResult, putResult);
    }
    const auto& c = exchange.counters();
    require(c.imports == static_cast<U64>(opts.epochs) &&
            c.putCalls == static_cast<U64>(opts.epochs) * plan.sendPeers.size() &&
            c.putElements == static_cast<U64>(opts.epochs) * plan.packElements &&
            c.readyWaits == static_cast<U64>(opts.epochs) * plan.receivePeers.size() &&
            c.ackSignals == c.readyWaits &&
            c.creditWaits == static_cast<U64>(opts.epochs - 1) * plan.sendPeers.size() &&
            c.globalSyncsInImport == 0,
            name + ": per-peer PUT/generation/ACK counters do not match plan");
    localStats[0] += c.putCalls;
    localStats[1] += c.putElements;
    localStats[2] += c.readyWaits;
    localStats[3] += c.creditWaits;
    localStats[4] += c.ackSignals;
    localStats[5] += c.packLaunches;
    localStats[6] += c.sendLaunches;
    localStats[7] += c.unpackLaunches;
    localStats[8] += c.localLaunches;
    localStats[9] += c.ackLaunches;
    localStats[10] += c.localElements;
    localStats[11] += c.globalSyncsInImport;
    localStats[12] += c.imports;
    // Collective retirement is NOT part of the Import hot path. It is
    // required by NVSHMEM symmetric allocation's lifetime contract.
    exchange.release();
  }
  U64 globalStats[13] = {};
  mpiCheck(MPI_Allreduce(localStats, globalStats, 13, MPI_UINT64_T,
                         MPI_SUM, session.comm()), "P5.2 post-test operation accounting");
  const U64 nEpochs = static_cast<U64>(totalEpochs);
  require(globalStats[0] == globalPlan[2] * nEpochs &&
          globalStats[1] == globalPlan[0] * nEpochs &&
          globalStats[2] == globalPlan[3] * nEpochs &&
          globalStats[3] == globalPlan[2] * static_cast<U64>(
                                opts.cycles * (opts.epochs - 1)) &&
          globalStats[4] == globalStats[2] &&
          globalStats[11] == 0 &&
          globalStats[12] == static_cast<U64>(session.numPes()) * nEpochs,
          name + ": global per-peer dynamic counters do not match plan");
  const U64 localCross[2] = {crossCalls * nEpochs, crossElements * nEpochs};
  U64 globalCross[2] = {};
  mpiCheck(MPI_Allreduce(localCross, globalCross, 2, MPI_UINT64_T,
                         MPI_SUM, session.comm()), "P5.2 post-test inter-node accounting");
  suite.putCalls += globalStats[0];
  suite.putElements += globalStats[1];
  suite.readyWaits += globalStats[2];
  suite.creditWaits += globalStats[3];
  suite.ackSignals += globalStats[4];
  suite.crossNodePutCalls += globalCross[0];
  suite.crossNodePutElements += globalCross[1];
  ++suite.passed;
  if (session.mpiRank() == 0)
    std::cout << "[NVSHMEM P5.2 CASE] case=" << name
              << " status=PASS cycles=" << opts.cycles
              << " epochsPerCycle=" << opts.epochs
              << " peerPutsPerEpoch=" << globalPlan[2]
              << " importElementsPerEpoch=" << globalPlan[1]
              << " gpuPutCalls=" << globalStats[0]
              << " gpuPutElements=" << globalStats[1]
              << " gpuPutBytes=" << globalStats[1] * sizeof(double)
              << " readyStreamWaits=" << globalStats[2]
              << " creditStreamWaits=" << globalStats[3]
              << " gpuAckSignals=" << globalStats[4]
              << " packKernels=" << globalStats[5]
              << " sendKernels=" << globalStats[6]
              << " unpackKernels=" << globalStats[7]
              << " localKernels=" << globalStats[8]
              << " ackKernels=" << globalStats[9]
              << " localCopies=" << globalStats[10]
              << " globalSyncsInImport=" << globalStats[11]
              << " crossNodePutCalls=" << globalCross[0]
              << " crossNodePutBytes=" << globalCross[1] * sizeof(double)
              << " symmetricBytesPerPE=" << symmetricBytes
              << " payloadTransport=NVSHMEM_PUT_SIGNAL"
              << " sync=POINT_TO_POINT_GENERATION_CREDIT"
              << " oracles=Tpetra_doImport,P2_CudaSpace"
              << " performanceMeasured=NO\n";
}

template<class LO, class GO, class Node>
void run(const NvshmemP0::Session& session, const Options& options,
         int matrixSize, const std::string& filename) {
  checkOptions(options);
  const auto topology = NvshmemP4::getTopology(session);
  require(topology.distinctNodes >= options.minimumNodes &&
          (options.maximumNodes == 0 || topology.distinctNodes <= options.maximumNodes),
          "physical node count outside requested topology");
  const auto comm = Tpetra::getDefaultComm();
  require(comm->getRank() == session.mpiRank() &&
          comm->getSize() == session.numPes(),
          "Tpetra and NVSHMEM communicator ranks disagree");
  Suite suite;
  if (session.mpiRank() == 0) {
    const char* transport = std::getenv("NVSHMEM_REMOTE_TRANSPORT");
    const char* ibgda = std::getenv("NVSHMEM_IB_ENABLE_IBGDA");
    std::cout << "[NVSHMEM P5.2] BEGIN ranks=" << session.numPes()
              << " physicalNodes=" << topology.distinctNodes
              << " epochs=" << options.epochs << " cycles=" << options.cycles
              << " skewRank0Ms=" << options.skewMillis
              << " transport=" << (transport ? transport : "default")
              << " ibgda=" << (ibgda ? ibgda : "default")
              << " readyWait=nvshmemx_signal_wait_until_on_stream"
              << " reuse=PER_PEER_CREDIT_ACK"
              << " perImportGlobalSync=NONE performanceMeasured=NO\n";
  }
  for (const auto& name : NvshmemP4::cases()) {
    if (options.only != "all" && options.only != name) continue;
    if (name == "matrix") {
      using Matrix = Tpetra::CrsMatrix<double, LO, GO, Node>;
      using Map = Tpetra::Map<LO, GO, Node>;
      using Import = Tpetra::Import<LO, GO, Node>;
      Teuchos::RCP<Matrix> A;
      if (filename.empty())
        A = Tpetra::Utils::MatrixGenerator<Matrix>::generate_miniFE_matrix(matrixSize, comm);
      else
        A = Tpetra::MatrixMarket::Reader<Matrix>::readSparseFile(filename, comm);
      require(A->isFillComplete(), "matrix is not fill complete");
      const Teuchos::RCP<const Map> domain = A->getDomainMap();
      const Teuchos::RCP<const Map> column = A->getColMap();
      const Teuchos::RCP<const Import> importer = NvshmemP1::ensureImporter(
          domain, column, A->getGraph()->getImporter(), session.comm());
      runCase(name, domain, column, importer, session, topology, options, suite,
              filename.empty() && session.numPes() > 1);
    } else {
      if (session.numPes() < NvshmemP4::minimumPes(name)) {
        ++suite.skipped;
        if (session.mpiRank() == 0)
          std::cout << "[NVSHMEM P5.2] case=" << name << " status=SKIP"
                    << " requiresPEs=" << NvshmemP4::minimumPes(name) << '\n';
        continue;
      }
      require(static_cast<U64>(session.numPes()) <=
                  (static_cast<U64>(std::numeric_limits<GO>::max()) - 20000ULL) /
                      1000000ULL, "GO too narrow for synthetic sparse GIDs");
      using Map = Tpetra::Map<LO, GO, Node>;
      using Import = Tpetra::Import<LO, GO, Node>;
      const auto spec = NvshmemP4::buildSpec<GO>(name, session.mpiRank(), session.numPes());
      const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
      const Teuchos::RCP<const Map> domain = Teuchos::rcp(
          new Map(invalid, spec.owned(), static_cast<GO>(0), comm));
      const Teuchos::RCP<const Map> column = Teuchos::rcp(
          new Map(invalid, spec.target(), static_cast<GO>(0), comm));
      const Teuchos::RCP<const Import> importer = Teuchos::rcp(new Import(domain, column));
      runCase(name, domain, column, importer, session, topology, options, suite);
    }
  }
  require(suite.passed + suite.skipped > 0, "no P5.2 case was selected");
  if (options.requireCrossNode)
    require(suite.crossNodePutCalls != 0,
            "selected cases performed no inter-node GPU PUTs");
  if (session.mpiRank() == 0)
    std::cout << "[NVSHMEM P5.2] SUMMARY status=" << (suite.passed ? "PASS" : "SKIP")
              << " casesPassed=" << suite.passed
              << " casesSkipped=" << suite.skipped
              << " gpuPutCalls=" << suite.putCalls
              << " gpuPutElements=" << suite.putElements
              << " readyStreamWaits=" << suite.readyWaits
              << " creditStreamWaits=" << suite.creditWaits
              << " gpuAckSignals=" << suite.ackSignals
              << " crossNodePutCalls=" << suite.crossNodePutCalls
              << " crossNodePutElements=" << suite.crossNodePutElements
              << " globalSyncsInImport=0"
              << " sync=POINT_TO_POINT_GENERATION_CREDIT"
              << " performanceMeasured=NO\n";
}
} // namespace NvshmemP52
#endif // TPETRA_PERFORMANCE_NVSHMEM_P52_HPP
