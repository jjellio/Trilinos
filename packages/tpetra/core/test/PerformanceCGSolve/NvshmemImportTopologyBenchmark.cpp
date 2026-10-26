// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
// P7.1b dedicated matrix-free Tpetra Import vs P6.2/P7.1 NVSHMEM driver.
// No CrsMatrix, no MiniFE; identical source/target Maps for both methods.
#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Vector.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_OrdinalTraits.hpp"
#include "SyntheticImportGraph.hpp"
#include "NvshmemP62.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef HAVE_TPETRA_INST_CUDA
#error "Matrix-free NVSHMEM Import topology benchmark requires CUDA Tpetra"
#endif

namespace SyntheticImportBenchmark {
namespace S = SyntheticImportGraph;
using U64 = S::U64;
struct Options {
  int owned = 65536;
  int offDegree = 1;
  int intraDegree = 0;
  std::string volume = "per-peer";
  int valuesPerPeer = 1024;
  int totalRemoteValues = 8192;
  int localValues = 0;
  std::string source = "contiguous";
  int sourceStride = 17;
  std::string target = "remote-only";
  int seed = 12345;
  int expectedPpn = 0; // 0 = infer uniform node placement
  NvshmemP62::Options reference;
};

inline void mpiCheck(int result, const char* where) {
  if (result != MPI_SUCCESS)
    throw std::runtime_error(std::string("P7.1b MPI error in ") + where);
}
inline S::Config makeConfig(const Options& o) {
  S::check(o.owned >= 1 && o.valuesPerPeer >= 0 && o.totalRemoteValues >= 0 &&
           o.localValues >= 0 && o.sourceStride >= 1 && o.seed >= 0 &&
           o.offDegree >= 0 && o.intraDegree >= 0 && o.expectedPpn >= 0,
           "negative map size, degree, stride, seed or expected PPN");
  S::Config c;
  c.ownedPerRank = static_cast<U64>(o.owned);
  c.offNodeDegree = o.offDegree;
  c.intraNodeDegree = o.intraDegree;
  if (o.volume == "per-peer") c.volume = S::TransferVolume::PerPeer;
  else if (o.volume == "fixed-total") c.volume = S::TransferVolume::FixedTotal;
  else S::check(false, "volume must be per-peer|fixed-total");
  c.valuesPerPeer = static_cast<U64>(o.valuesPerPeer);
  c.remoteValuesTotal = static_cast<U64>(o.totalRemoteValues);
  c.localValues = static_cast<U64>(o.localValues);
  c.source = S::parseSourcePattern(o.source);
  c.target = S::parseTargetLayout(o.target);
  c.sourceStride = static_cast<U64>(o.sourceStride);
  c.seed = static_cast<U64>(o.seed);
  return c;
}

inline void outputTraffic(const S::Graph& g, const NvshmemP0::Session& session,
                          const std::string& prefix) {
  const auto& p = g.placement;
  if (session.mpiRank() != 0) return;
  std::cout << "[NVSHMEM TOPOLOGY] nranks=" << p.nranks << " nodes=" << p.nnodes
            << " ppn=" << p.ppn << " offDegree=" << g.config.offNodeDegree
            << " intraDegree=" << g.config.intraNodeDegree
            << " source=" << S::sourceName(g.config.source)
            << " ownedPerRank=" << g.config.ownedPerRank
            << " remoteDoublesPerRank=" << g.doublesPerRank
            << " crossNodeDoublesPerRank=" << g.crossNodeDoublesPerRank
            << " localValues=" << g.config.localValues
            << " remoteBytesPerRank=" << S::product(g.doublesPerRank, 8, "remote bytes")
            << " crossNodeBytesPerRank=" << S::product(g.crossNodeDoublesPerRank, 8, "cross-node bytes")
            << " status=PLAN_PASS\n";
  std::ofstream matrix, peerInfo;
  if (!prefix.empty()) {
    matrix.open(prefix + "_traffic.csv", std::ios::trunc);
    peerInfo.open(prefix + "_rank_topology.csv", std::ios::trunc);
    S::check(matrix.good() && peerInfo.good(), "cannot create traffic CSV files");
    matrix << "sender_rank,receiver_rank,sender_node,receiver_node,route,doubles,bytes\n";
    peerInfo << "rank,node_name,node_index,local_slot,send_peers,receive_peers,"
                "send_doubles,receive_doubles,cross_node_send_doubles,cross_node_receive_doubles\n";
  }
  for (int rank = 0; rank < p.nranks; ++rank) {
    const auto ni = p.nodeOfRank[static_cast<std::size_t>(rank)];
    std::cout << "[NVSHMEM TOPOLOGY RANK] rank=" << rank
              << " hostname=" << p.nodeNames[static_cast<std::size_t>(ni)]
              << " slot=" << p.localSlotOfRank[static_cast<std::size_t>(rank)]
              << " sendPeers=" << g.outgoing[static_cast<std::size_t>(rank)].size()
              << " recvPeers=" << g.incoming[static_cast<std::size_t>(rank)].size()
              << " sendDoubles=" << g.doublesPerRank
              << " recvDoubles=" << g.doublesPerRank
              << " crossNodeSendDoubles=" << g.crossNodeDoublesPerRank
              << " crossNodeRecvDoubles=" << g.crossNodeDoublesPerRank << '\n';
    if (peerInfo.is_open())
      peerInfo << rank << ',' << p.nodeNames[static_cast<std::size_t>(ni)] << ','
               << ni << ',' << p.localSlotOfRank[static_cast<std::size_t>(rank)] << ','
               << g.outgoing[static_cast<std::size_t>(rank)].size() << ','
               << g.incoming[static_cast<std::size_t>(rank)].size() << ','
               << g.doublesPerRank << ',' << g.doublesPerRank << ','
               << g.crossNodeDoublesPerRank << ',' << g.crossNodeDoublesPerRank << '\n';
    for (int receiver = 0; receiver < p.nranks; ++receiver) {
      const U64 n = g.count(rank, receiver);
      const bool cross = ni != p.nodeOfRank[static_cast<std::size_t>(receiver)];
      if (matrix.is_open())
        matrix << rank << ',' << receiver << ','
               << p.nodeNames[static_cast<std::size_t>(ni)] << ','
               << p.nodeNames[static_cast<std::size_t>(p.nodeOfRank[static_cast<std::size_t>(receiver)])]
               << ',' << (rank == receiver ? "self" : (cross ? "off-node" : "intra-node"))
               << ',' << n << ',' << S::product(n,8,"traffic matrix bytes") << '\n';
      if (n)
        std::cout << "[NVSHMEM TOPOLOGY EDGE] sender=" << rank
                  << " receiver=" << receiver << " route=" << (cross ? "off-node" : "intra-node")
                  << " doubles=" << n << " bytes=" << S::product(n,8,"traffic bytes") << '\n';
    }
  }
  if (matrix.is_open()) S::check(matrix.good() && peerInfo.good(), "traffic CSV write error");
}

template<class LO, class GO, class Node>
void run(const NvshmemP0::Session& session, const Options& options) {
  using Map = Tpetra::Map<LO,GO,Node>;
  using Import = Tpetra::Import<LO,GO,Node>;
  const auto topology = NvshmemP4::getTopology(session);
  const S::Placement placement(topology.names);
  S::check(placement.nnodes >= options.reference.minimumNodes &&
           (options.reference.maximumNodes == 0 || placement.nnodes <= options.reference.maximumNodes),
           "physical node count violates requested bounds");
  S::check(!options.expectedPpn || placement.ppn == options.expectedPpn,
           "actual PPN differs from requested PPN");
  const auto cfg = makeConfig(options);
  const S::Graph graph(placement, cfg);
  S::check(!options.reference.requireCrossNode || graph.crossNodeDoublesPerRank > 0,
           "require-cross-node set but chosen graph has no off-node traffic");
  S::check(S::product(static_cast<U64>(session.numPes()), cfg.ownedPerRank, "global source GID range")
             <= static_cast<U64>(std::numeric_limits<GO>::max()),
           "rank*ownedPerRank would exceed the Tpetra global ordinal range");
  S::check(cfg.ownedPerRank <= static_cast<U64>(std::numeric_limits<LO>::max()) &&
           S::add(graph.doublesPerRank, cfg.localValues, "target size") <=
             static_cast<U64>(std::numeric_limits<LO>::max()),
           "local Map size exceeds Tpetra LO capacity");
  const auto spec = S::mapsForRank(graph,session.mpiRank());
  Teuchos::Array<GO> owned, target;
  using teuchos_size_type = Teuchos::Array<GO>::size_type;
  S::check(spec.owned.size() <= static_cast<std::size_t>(std::numeric_limits<teuchos_size_type>::max()) &&
           spec.target.size() <= static_cast<std::size_t>(std::numeric_limits<teuchos_size_type>::max()),
           "Teuchos::Array ordinal/array length overflow");
  owned.reserve(static_cast<teuchos_size_type>(spec.owned.size()));
  target.reserve(static_cast<teuchos_size_type>(spec.target.size()));
  for (U64 v : spec.owned) owned.push_back(static_cast<GO>(v));
  for (U64 v : spec.target) target.push_back(static_cast<GO>(v));
  const auto comm = Tpetra::getDefaultComm();
  S::check(comm->getRank() == session.mpiRank() && comm->getSize() == session.numPes(),
           "Tpetra communicator does not match MPI/NVSHMEM communicator");
  const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  Teuchos::RCP<const Map> domain = Teuchos::rcp(new Map(invalid, owned(), static_cast<GO>(0), comm));
  Teuchos::RCP<const Map> column = Teuchos::rcp(new Map(invalid, target(), static_cast<GO>(0), comm));
  const Teuchos::RCP<const Import> importer = Teuchos::rcp(new Import(domain, column));

  // Check the REAL Importer's actual owner-local LIDs and send/receive peers
  // against the synthetic graph before performing or timing any transfer.
  std::vector<int> rankToPe(static_cast<std::size_t>(session.numPes()), -1);
  for (int pe = 0; pe < session.numPes(); ++pe)
    rankToPe[static_cast<std::size_t>(session.mpiRankForPe(pe))] = pe;
  const auto actual = ImportGatherP1::buildImportGatherPlan(
      domain, column, *importer, rankToPe, session.comm());
  const int rank = session.mpiRank();
  S::check(actual.sourceLength == cfg.ownedPerRank &&
           actual.targetLength == graph.doublesPerRank + cfg.localValues &&
           actual.local.size() == cfg.localValues &&
           actual.remote.size() == graph.doublesPerRank &&
           actual.numRemotePeers == graph.incoming[static_cast<std::size_t>(rank)].size() &&
           actual.exportLids.size() == graph.doublesPerRank &&
           actual.exportPids.size() == graph.doublesPerRank,
           "actual Importer does not match requested graph sizes or degrees");
  std::vector<U64> from(static_cast<std::size_t>(session.numPes()), 0),
                   to(static_cast<std::size_t>(session.numPes()), 0);
  for (const auto& e : actual.remote) {
    S::check(e.ownerRank != rank && e.sourceLid < cfg.ownedPerRank,
             "unexpected owner or source LID in Importer");
    ++from[static_cast<std::size_t>(e.ownerRank)];
  }
  for (int pe : actual.exportPids) {
    S::check(pe >= 0 && pe < session.numPes(), "invalid actual Importer export rank");
    ++to[static_cast<std::size_t>(pe)];
  }
  for (int other = 0; other < session.numPes(); ++other) {
    S::check(from[static_cast<std::size_t>(other)] == graph.count(other,rank),
             "actual received Map traffic differs from requested peer matrix");
    S::check(to[static_cast<std::size_t>(other)] == graph.count(rank,other),
             "actual sender export traffic differs from requested peer matrix");
  }
  const unsigned long long passed = 1ULL;
  unsigned long long allPassed = 0;
  mpiCheck(MPI_Allreduce(&passed, &allPassed, 1, MPI_UNSIGNED_LONG_LONG,
                         MPI_SUM, session.comm()), "P7.1b Importer validation");
  S::check(allPassed == static_cast<unsigned long long>(session.numPes()),
           "MPI global validation mismatch");
  outputTraffic(graph,session,options.reference.outputPrefix);
  if (rank == 0)
    std::cout << "[NVSHMEM TOPOLOGY IMPORTER] status=PASS matchedRanks=" << allPassed
              << " matrixConstructed=NO independentImporter=YES\n";

  // Benchmark controls and verification stay exactly those of the P7.1/P6.2
  // reference. Source remains ordinary Tpetra CUDA-space memory; NVSHMEM's
  // symmetric allocations are the unchanged P6.2 scratch and inbox buffers.
  const auto result = NvshmemP62::benchmarkCase(
      "topology", domain, column, importer, session, topology,
      options.reference, graph.doublesPerRank > 0);
  if (rank == 0) {
    std::cout << "[NVSHMEM TOPOLOGY] SUMMARY status=PASS nodes=" << placement.nnodes
              << " ppn=" << placement.ppn << " degree="
              << (cfg.offNodeDegree + cfg.intraNodeDegree)
              << " doublePerRank=" << graph.doublesPerRank
              << " crossNodeDoublePerRank=" << graph.crossNodeDoublesPerRank
              << " p62CrossNodeDoublesGlobal=" << result.crossNodeDoubles
              << " verifiedOutputs=" << result.validations << '\n';
  }
  S::check(result.crossNodeDoubles ==
            S::product(graph.crossNodeDoublesPerRank,static_cast<U64>(session.numPes()),
                       "global cross-node doubles"),
           "P6.2 measured cross-node export bytes differ from requested topology");
}
} // namespace SyntheticImportBenchmark

int main(int argc, char** argv) {
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);
  int mpiRank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  SyntheticImportBenchmark::Options opts;
  int numGpus = 1;
  Teuchos::CommandLineProcessor parser(false, true);
  parser.setOption("numgpus", &numGpus, "GPU count visible per rank or local node (see GPU binding wrapper)");
  parser.setOption("owned", &opts.owned, "Owned source doubles per PE, equal on all ranks");
  parser.setOption("offnode-degree", &opts.offDegree, "Exact number of distinct off-node send/receive peers per PE");
  parser.setOption("intranode-degree", &opts.intraDegree, "Exact number of distinct same-node send/receive peers per PE");
  parser.setOption("volume", &opts.volume, "per-peer|fixed-total");
  parser.setOption("values-per-peer", &opts.valuesPerPeer, "Doubles per directed peer relation in per-peer mode");
  parser.setOption("remote-values-total", &opts.totalRemoteValues, "Doubles per PE across all remote peers in fixed-total mode");
  parser.setOption("local-values", &opts.localValues, "Number of local source entries also requested in target Map");
  parser.setOption("source-pattern", &opts.source, "contiguous|strided|permuted");
  parser.setOption("source-stride", &opts.sourceStride, "For strided mode: positive stride coprime with owned");
  parser.setOption("target-layout", &opts.target, "remote-only|prefix|interleave|shuffle");
  parser.setOption("seed", &opts.seed, "Deterministic GID sampling and target shuffle seed");
  parser.setOption("expected-ppn", &opts.expectedPpn, "Assert physical ranks per node, or 0 to infer");
  parser.setOption("put-variant", &opts.reference.putVariant, "thread|warp|block P6.2 PUT collective");
  parser.setOption("chunk", &opts.reference.chunkDoubles, "Doubles per P6.2 PUT descriptor");
  parser.setOption("receiver-blocks", &opts.reference.receiverBlocks, "Bounded P6.2 GPU signal polling blocks");
  parser.setOption("mode", &opts.reference.mode, "direct|diagnostics|both|p71");
  parser.setOption("warmup", &opts.reference.warmup, "Warmups per method");
  parser.setOption("reps", &opts.reference.repetitions, "Measured direct repetitions per method");
  parser.setOption("diagnostic-reps", &opts.reference.diagnosticRepetitions, "P6.2 diagnostics repetitions");
  parser.setOption("p71-reps", &opts.reference.p71ProfileRepetitions, "P7.1 instrumented repetitions");
  parser.setOption("p71-max-samples", &opts.reference.p71MaxSamples, "P7.1 work units sampled per direction");
  parser.setOption("verify-every", &opts.reference.verifyEvery, "0=last warmup/final, 1=every epoch");
  parser.setOption("out", &opts.reference.outputPrefix, "Output CSV prefix, mandatory for p71 mode");
  parser.setOption("min-nodes", &opts.reference.minimumNodes, "Minimum distinct physical nodes");
  parser.setOption("max-nodes", &opts.reference.maximumNodes, "Maximum distinct physical nodes (0 unlimited)");
  parser.setOption("require-cross-node", "no-require-cross-node", &opts.reference.requireCrossNode,
                   "Require a genuine off-node transfer");
  if (parser.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) return EXIT_FAILURE;
  if (numGpus <= 0) {
    if (mpiRank == 0) std::cerr << "--numgpus must be positive\n";
    return EXIT_FAILURE;
  }
  // Avoid initializing NVSHMEM before Kokkos selects and initializes the CUDA device.
  const int localRank = NvshmemP0::nodeLocalMpiRank(MPI_COMM_WORLD);
  const int device = localRank % numGpus;
  Kokkos::InitializationSettings init;
  init.set_device_id(device);
  Kokkos::initialize(init);
  int rc = EXIT_FAILURE;
  try {
    NvshmemP0::Session session(MPI_COMM_WORLD);
    session.describe(device);
    // Validate options and local constraints *before* allocating any symmetric
    // heap. All ranks must participate in the same P6.2 plan collectives.
    SyntheticImportBenchmark::makeConfig(opts);
    NvshmemP62::checkOptions(opts.reference); // keep reference.only=matrix
    using LO = Tpetra::Map<>::local_ordinal_type;
    using GO = Tpetra::Map<>::global_ordinal_type;
    using Node = Tpetra::KokkosCompat::KokkosCudaWrapperNode;
    SyntheticImportBenchmark::run<LO,GO,Node>(session,opts);
    rc = EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "ImportTopologyBenchmark rank " << mpiRank << ": " << e.what() << '\n';
    // A peer may be inside NVSHMEM collective symmetric allocation.
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  Kokkos::finalize();
  return rc;
}
