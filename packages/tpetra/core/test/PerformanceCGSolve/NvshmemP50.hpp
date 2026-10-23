// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P5.0: one-time, host-only Import structure analysis. No vector payload,
// NVSHMEM allocation, NVSHMEM RMA, or timed kernel. Reuse the passing P1
// descriptors and the P4 Map cases without altering P0-P4 implementations.

#ifndef TPETRA_PERFORMANCE_NVSHMEM_P50_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P50_HPP

#include "ImportStructureAnalyzer.hpp"
#include "NvshmemP4.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace NvshmemP50 {

struct Options {
  std::string only = "all";       // all, matrix, or any named P4 case
  bool showPeers = true;          // ordered per-incoming-peer records
  bool showOutgoing = true;       // ordered per-outgoing-rank records
  std::string csvPrefix;          // optional: writes .rank/.recv/.send/.global.csv on rank 0
};

inline void mpiCheck(int error, const char* operation) {
  ImportStructureP50::require(error == MPI_SUCCESS,
                               std::string("MPI failure in ") + operation);
}
inline void checkOptions(const Options& opts) {
  ImportStructureP50::require(opts.only == "all" ||
      std::find(NvshmemP4::cases().begin(), NvshmemP4::cases().end(), opts.only) !=
          NvshmemP4::cases().end(), "unrecognized --nvshmem-p50-case: " + opts.only);
}
inline std::uint64_t elementBytes(std::uint64_t elements) {
  ImportStructureP50::require(elements <=
      std::numeric_limits<std::uint64_t>::max() / sizeof(double),
      "byte accounting would overflow 64-bit unsigned integer");
  return elements * sizeof(double);
}
inline std::string fraction(std::uint64_t numerator, std::uint64_t denominator) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(5)
      << (denominator ? static_cast<double>(numerator) / static_cast<double>(denominator) : 0.0);
  return out.str();
}
inline std::string ownerOrder(const std::vector<int>& rankOrder) {
  std::ostringstream out;
  if (rankOrder.empty()) return "-";
  for (std::size_t i = 0; i < rankOrder.size(); ++i) {
    if (i) out << ':';
    out << rankOrder[i];
  }
  return out.str();
}

// MPI_Gatherv preserves numerical rank order; only rank zero writes output.
// Rank-order printout is deterministic even when OpenMPI multiplexes stdout.
inline std::string gatherToRoot(const std::string& local, const NvshmemP0::Session& session) {
  const int rank = session.mpiRank();
  const int nRanks = session.numPes();
  ImportStructureP50::require(local.size() <=
      static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "per-rank report too large for MPI_Gatherv");
  const int mine = static_cast<int>(local.size());
  std::vector<int> counts(rank == 0 ? static_cast<std::size_t>(nRanks) : 0);
  mpiCheck(MPI_Gather(&mine, 1, MPI_INT, rank == 0 ? counts.data() : nullptr,
                      1, MPI_INT, 0, session.comm()), "MPI_Gather(P5.0 text lengths)");
  std::vector<int> offsets(counts.size());
  std::size_t total = 0;
  if (rank == 0) {
    for (int i = 0; i < nRanks; ++i) {
      ImportStructureP50::require(counts[static_cast<std::size_t>(i)] >= 0,
                                   "negative gathered text length");
      offsets[static_cast<std::size_t>(i)] = static_cast<int>(total);
      total += static_cast<std::size_t>(counts[static_cast<std::size_t>(i)]);
      ImportStructureP50::require(total <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()),
          "global report too large for MPI_Gatherv");
    }
  }
  std::string all(rank == 0 ? total : 0, '\0');
  mpiCheck(MPI_Gatherv(mine ? local.data() : nullptr, mine, MPI_CHAR,
                        rank == 0 && !all.empty() ? &all[0] : nullptr,
                        rank == 0 ? counts.data() : nullptr,
                        rank == 0 ? offsets.data() : nullptr,
                        MPI_CHAR, 0, session.comm()), "MPI_Gatherv(P5.0 ordered text)");
  return all;
}

inline std::string recvCsvHeader() {
  return "case,rank,ownerRank,sourcePe,requests,uniqueSources,duplicates,sourceRuns,"
         "multiSourceRuns,multiSourceElements,maxSourceRun,destinationRuns,"
         "multiDestinationRuns,multiDestinationElements,maxDestinationRun,"
         "alignedSourceRuns,alignedSourceElements,directRuns,"
         "directMultiRuns,directMultiElements,directFallbackCalls,scatterElements,"
         "ownerLength,minSourceLid,maxSourceLid,windowElements,windowExtraElements,"
         "uniqueWindowDensity,maxScratchElements,sourceTouchesOwnerEnd\n";
}
inline std::string sendCsvHeader() {
  return "case,rank,destinationRank,exportPackets,uniqueExportLids,duplicatePackets,"
         "minExportLid,maxExportLid,adjacentNumericRank\n";
}
inline std::string rankCsvHeader() {
  return "case,rank,pe,owned,target,same,permute,remote,exports,sourcePeers,outgoingPeers,"
         "localPrefix,remoteSuffix,remoteExactSuffix,ownerGrouped,ownerGroupedSuffix,ownerRankOrder,"
         "ownerVisitOrder,sourceRuns,destinationRuns,sourceMultiElements,"
         "alignedSourceRuns,alignedSourceElements,directRuns,"
         "directBulkRuns,directBulkElements,directLogicalGets,sourceRunLogicalGets,"
         "scatterIfSourceRuns,duplicateRemoteRequests,windowExtraElements,"
         "maxScratchElements,exportPidSegments,adjacentRankExports,"
         "fanoutSourceLids,fanoutExtraPackets,adjacentFanoutSourceLids,maxFanoutPeers\n";
}
inline std::string globalCsvHeader() {
  return "case,ranks,globalOwned,globalTarget,globalSame,globalRemote,globalExports,"
         "sourcePeerRelationships,sourceRuns,destinationRuns,sourceMultiElements,"
         "alignedSourceRuns,alignedSourceElements,directRuns,"
         "directBulkRuns,directBulkElements,directLogicalGets,sourceRunLogicalGets,"
         "potentialSenderPackedGets,scatterIfSourceRuns,duplicateRemoteRequests,"
         "fullWindowExtraElements,ranksRemoteExactSuffix,ranksRemoteOwnerGrouped\n";
}

inline void writeRank(const std::string& name, const ImportStructureP50::Rank& a,
                      const Options& opts, std::ostringstream& log,
                      std::ostringstream& recvCsv, std::ostringstream& sendCsv,
                      std::ostringstream& rankCsv) {
  log << "[NVSHMEM P5.0 RANK] case=" << name << " rank=" << a.rank
      << " pe=" << a.pe << " owned=" << a.sourceLength
      << " target=" << a.targetLength << " same=" << a.numSame
      << " permute=" << a.numPermute << " remote=" << a.requests
      << " exports=" << a.exports << " sourcePEs=" << a.sourcePeers
      << " outboundPeers=" << a.outgoingPeers
      << " localPrefix=" << a.localPrefix
      << " samePrefixFraction=" << fraction(a.numSame, a.targetLength)
      << " localPrefixFraction=" << fraction(a.localPrefix, a.targetLength)
      << " remoteSuffix=" << a.remoteSuffix
      << " remoteSuffixFraction=" << fraction(a.remoteSuffix, a.targetLength)
      << " remoteExactSuffix=" << static_cast<int>(a.remoteIsExactSuffix)
      << " ownerGroupedSuffix=" << static_cast<int>(a.remoteIsExactSuffix && a.remoteOwnerGrouped)
      << " remoteOwnerGrouped=" << static_cast<int>(a.remoteOwnerGrouped)
      << " remoteRankOrder=" << a.remoteOwnerRankOrder
      << " remoteOwnerVisits=" << ownerOrder(a.remoteOwnerOrder)
      << " sourceRuns=" << a.sourceRuns
      << " destinationRuns=" << a.destinationRuns
      << " alignedSourceRuns=" << a.alignedSourceRuns
      << " alignedSourceElements=" << a.alignedSourceElements
      << " directRuns=" << a.directRuns
      << " directBulkRuns=" << a.directMultiRuns
      << " directBulkElements=" << a.directMultiElements
      << " directLogicalGets=" << a.directLogicalGets
      << " sourceRunLogicalGets=" << a.sourceRunLogicalGets
      << " scatterIfSourceRuns=" << a.scatterIfSourceRuns
      << " duplicateRemoteRequests=" << a.duplicates
      << " windowExtraElements=" << a.windowExtraElements
      << " maxScratchElements=" << a.largestScratchElements
      << " exportsToNumericNeighbors=" << a.adjacentRankExports
      << " fanoutSourceLids=" << a.fanoutSourceLids
      << " adjacentFanoutLids=" << a.adjacentFanoutSourceLids
      << " status=PASS\n";
  rankCsv << name << ',' << a.rank << ',' << a.pe << ',' << a.sourceLength << ','
          << a.targetLength << ',' << a.numSame << ',' << a.numPermute << ','
          << a.requests << ',' << a.exports << ',' << a.sourcePeers << ','
          << a.outgoingPeers << ',' << a.localPrefix << ',' << a.remoteSuffix << ','
          << static_cast<int>(a.remoteIsExactSuffix) << ','
          << static_cast<int>(a.remoteOwnerGrouped) << ','
          << static_cast<int>(a.remoteIsExactSuffix && a.remoteOwnerGrouped) << ','
          << a.remoteOwnerRankOrder << ','
          << ownerOrder(a.remoteOwnerOrder) << ',' << a.sourceRuns << ','
          << a.destinationRuns << ',' << a.sourceMultiElements << ','
          << a.alignedSourceRuns << ',' << a.alignedSourceElements << ','
          << a.directRuns << ',' << a.directMultiRuns << ',' << a.directMultiElements << ','
          << a.directLogicalGets << ',' << a.sourceRunLogicalGets << ','
          << a.scatterIfSourceRuns << ',' << a.duplicates << ','
          << a.windowExtraElements << ',' << a.largestScratchElements << ','
          << a.exportPidSegments << ',' << a.adjacentRankExports << ','
          << a.fanoutSourceLids << ',' << a.fanoutExtraPackets << ','
          << a.adjacentFanoutSourceLids << ',' << a.maxFanoutDestinations << '\n';

  for (const auto& peer : a.peers) {
    if (opts.showPeers) {
      log << "[NVSHMEM P5.0 PEER] case=" << name << " rank=" << a.rank
          << " ownerRank=" << peer.ownerRank << " sourcePE=" << peer.sourcePe
          << " requests=" << peer.requests
          << " uniqueSources=" << peer.uniqueSources
          << " duplicates=" << peer.duplicates
          << " sourceRuns=" << peer.sourceRunCount
          << " multiSourceRuns=" << peer.sourceMultiRuns
          << " multiSourceElements=" << peer.sourceMultiElements
          << " maxSourceRun=" << peer.longestSourceRun
          << " destinationRuns=" << peer.destinationRunCount
          << " multiDestinationRuns=" << peer.destinationMultiRuns
          << " maxDestinationRun=" << peer.longestDestinationRun
          << " alignedSourceRuns=" << peer.alignedSourceRuns
          << " alignedSourceElements=" << peer.alignedSourceElements
          << " directRuns=" << peer.directRuns
          << " directBulkRuns=" << peer.directMultiRuns
          << " directBulkElements=" << peer.directMultiElements
          << " directScalarFallback=" << peer.directFallbackScalarCalls
          << " sourceRunScatterElements=" << peer.scatterIfSourceRuns
          << " sourceMin=" << peer.minSourceLid
          << " sourceMax=" << peer.maxSourceLid
          << " ownerLength=" << peer.ownerLength
          << " ownerLastTouched=" << static_cast<int>(peer.sourceTouchesLastOwned)
          << " windowElements=" << peer.windowElements
          << " uniqueWindowDensity=" << fraction(peer.uniqueSources, peer.windowElements)
          << " fullWindowExtraElements=" << peer.extraWindowElements
          << " fullWindowExtraBytes=" << elementBytes(peer.extraWindowElements)
          << " maxScratchElements=" << peer.largestScratchElements << '\n';
    }
    recvCsv << name << ',' << a.rank << ',' << peer.ownerRank << ',' << peer.sourcePe << ','
            << peer.requests << ',' << peer.uniqueSources << ',' << peer.duplicates << ','
            << peer.sourceRunCount << ',' << peer.sourceMultiRuns << ','
            << peer.sourceMultiElements << ',' << peer.longestSourceRun << ','
            << peer.destinationRunCount << ',' << peer.destinationMultiRuns << ','
            << peer.destinationMultiElements << ',' << peer.longestDestinationRun << ','
            << peer.alignedSourceRuns << ',' << peer.alignedSourceElements << ','
            << peer.directRuns << ',' << peer.directMultiRuns << ',' << peer.directMultiElements << ','
            << peer.directFallbackScalarCalls << ',' << peer.scatterIfSourceRuns << ','
            << peer.ownerLength << ',' << peer.minSourceLid << ','
            << peer.maxSourceLid << ',' << peer.windowElements << ','
            << peer.extraWindowElements << ',' << fraction(peer.uniqueSources, peer.windowElements)
            << ',' << peer.largestScratchElements << ','
            << static_cast<int>(peer.sourceTouchesLastOwned) << '\n';
  }
  for (const auto& out : a.outgoing) {
    if (opts.showOutgoing)
      log << "[NVSHMEM P5.0 SEND] case=" << name << " rank=" << a.rank
          << " destinationRank=" << out.destinationRank
          << " exportPackets=" << out.requests
          << " uniqueSourceLids=" << out.uniqueSourceLids
          << " duplicates=" << out.duplicateExports
          << " minExportLid=" << out.minSourceLid
          << " maxExportLid=" << out.maxSourceLid
          << " adjacentNumericRank=" << static_cast<int>(out.adjacentNumericRank)
          << '\n';
    sendCsv << name << ',' << a.rank << ',' << out.destinationRank << ','
            << out.requests << ',' << out.uniqueSourceLids << ',' << out.duplicateExports
            << ',' << out.minSourceLid << ',' << out.maxSourceLid << ','
            << static_cast<int>(out.adjacentNumericRank) << '\n';
  }
}

// Fields reduced across ranks. Their order is mirrored in the summary text.
enum Counter : std::size_t {
  Owned, Target, Same, Remote, Export, SourcePeers, OutgoingPeers, SourceRuns,
  DestinationRuns, MultiSourceElems, AlignedRuns, AlignedElems, AllDirectRuns,
  DirectRuns, DirectElems, DirectGets,
  SourceGets, ScatterElems, Duplicates, WindowExtra, LocalPrefix, RemoteSuffix,
  ExactSuffixRanks, GroupedRanks, AdjacentExports, FanoutLids, FanoutExtra,
  CounterCount
};

inline std::array<unsigned long long, CounterCount>
localCounters(const ImportStructureP50::Rank& a) {
  return {{a.sourceLength, a.targetLength, a.numSame, a.requests, a.exports,
           a.sourcePeers, a.outgoingPeers, a.sourceRuns, a.destinationRuns,
           a.sourceMultiElements, a.alignedSourceRuns, a.alignedSourceElements,
           a.directRuns, a.directMultiRuns, a.directMultiElements,
           a.directLogicalGets, a.sourceRunLogicalGets, a.scatterIfSourceRuns,
           a.duplicates, a.windowExtraElements, a.localPrefix, a.remoteSuffix,
           static_cast<unsigned long long>(a.remoteIsExactSuffix),
           static_cast<unsigned long long>(a.remoteOwnerGrouped),
           a.adjacentRankExports, a.fanoutSourceLids, a.fanoutExtraPackets}};
}

inline void writeGlobal(const std::string& name,
                        const std::array<unsigned long long, CounterCount>& t,
                        int nRanks, std::ostream& log, std::ostream& csv) {
  ImportStructureP50::require(t[Remote] == t[Export],
                               name + ": global remote/export accounting mismatch");
  ImportStructureP50::require(t[SourcePeers] == t[OutgoingPeers],
                               name + ": incoming source-peer and outgoing destination-peer relationships differ");
  ImportStructureP50::require(t[DirectGets] <= t[Remote] &&
                               t[SourceGets] <= t[Remote] &&
                               t[DirectElems] <= t[Remote],
                               name + ": optimizer candidate accounting invalid");
  log << "[NVSHMEM P5.0 GLOBAL] case=" << name << " ranks=" << nRanks
      << " globalOwned=" << t[Owned] << " globalTarget=" << t[Target]
      << " globalRemote=" << t[Remote] << " globalExports=" << t[Export]
      << " sourcePeerRelationships=" << t[SourcePeers]
      << " scalarGetCallsBaseline=" << t[Remote]
      << " sourceRuns=" << t[SourceRuns]
      << " destinationRuns=" << t[DestinationRuns]
      << " sourceMultiElements=" << t[MultiSourceElems]
      << " alignedSourceRuns=" << t[AlignedRuns]
      << " alignedSourceElements=" << t[AlignedElems]
      << " directRuns=" << t[AllDirectRuns]
      << " directBulkRuns=" << t[DirectRuns]
      << " directBulkElements=" << t[DirectElems]
      << " directLogicalGets=" << t[DirectGets]
      << " sourceRunLogicalGets=" << t[SourceGets]
      << " senderPackedLogicalGets=" << t[SourcePeers]
      << " scatterIfSourceRuns=" << t[ScatterElems]
      << " duplicateRemoteRequests=" << t[Duplicates]
      << " fullWindowExtraElements=" << t[WindowExtra]
      << " fullWindowExtraBytes=" << elementBytes(t[WindowExtra])
      << " exactSuffixRanks=" << t[ExactSuffixRanks]
      << " remoteOwnerGroupedRanks=" << t[GroupedRanks]
      << " analysisOnly=YES rmaIssued=0 status=PASS\n";
  csv << name << ',' << nRanks << ',' << t[Owned] << ',' << t[Target] << ','
      << t[Same] << ',' << t[Remote] << ',' << t[Export] << ','
      << t[SourcePeers] << ',' << t[SourceRuns] << ',' << t[DestinationRuns] << ','
      << t[MultiSourceElems] << ',' << t[AlignedRuns] << ',' << t[AlignedElems] << ','
      << t[AllDirectRuns] << ',' << t[DirectRuns] << ',' << t[DirectElems] << ','
      << t[DirectGets] << ',' << t[SourceGets] << ',' << t[SourcePeers] << ','
      << t[ScatterElems] << ',' << t[Duplicates] << ',' << t[WindowExtra] << ','
      << t[ExactSuffixRanks] << ',' << t[GroupedRanks] << '\n';
}

template<class LO, class GO, class Node>
void runCase(const std::string& name,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domain,
             const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
             const Teuchos::RCP<const Tpetra::Import<LO, GO, Node>>& importer,
             const NvshmemP0::Session& session, const Options& opts,
             std::ostringstream& rankCsvRoot, std::ostringstream& recvCsvRoot,
             std::ostringstream& sendCsvRoot, std::ostringstream& globalCsvRoot) {
  ImportStructureP50::require(!importer.is_null(), "null Tpetra Importer");
  std::vector<int> peForRank(static_cast<std::size_t>(session.numPes()), -1);
  for (int p = 0; p < session.numPes(); ++p)
    peForRank[static_cast<std::size_t>(session.mpiRankForPe(p))] = p;
  const auto plan = ImportGatherP1::buildImportGatherPlan(
      domain, column, *importer, peForRank, session.comm());
  const auto a = ImportStructureP50::analyze(
      plan, session.mpiRank(), session.pe(), peForRank);

  std::ostringstream log, rankCsv, recvCsv, sendCsv;
  writeRank(name, a, opts, log, recvCsv, sendCsv, rankCsv);
  const std::string ordered = gatherToRoot(log.str(), session);
  if (session.mpiRank() == 0) std::cout << ordered;
  if (!opts.csvPrefix.empty()) {
    const std::string rankData = gatherToRoot(rankCsv.str(), session);
    const std::string recvData = gatherToRoot(recvCsv.str(), session);
    const std::string sendData = gatherToRoot(sendCsv.str(), session);
    if (session.mpiRank() == 0) {
      rankCsvRoot << rankData; recvCsvRoot << recvData; sendCsvRoot << sendData;
    }
  }

  const auto local = localCounters(a);
  std::array<unsigned long long, CounterCount> total{};
  mpiCheck(MPI_Allreduce(local.data(), total.data(), static_cast<int>(CounterCount),
                         MPI_UNSIGNED_LONG_LONG, MPI_SUM, session.comm()),
           "MPI_Allreduce(P5.0 metrics)");
  if (session.mpiRank() == 0)
    writeGlobal(name, total, session.numPes(), std::cout, globalCsvRoot);
}

template<class LO, class GO, class Node>
void runSynthetic(const std::string& name, const NvshmemP0::Session& session,
                  const Options& opts, std::ostringstream& rankCsvRoot,
                  std::ostringstream& recvCsvRoot, std::ostringstream& sendCsvRoot,
                  std::ostringstream& globalCsvRoot) {
  using Map = Tpetra::Map<LO, GO, Node>;
  using Import = Tpetra::Import<LO, GO, Node>;
  if (session.numPes() < NvshmemP4::minimumPes(name)) {
    if (session.mpiRank() == 0)
      std::cout << "[NVSHMEM P5.0] case=" << name << " status=SKIP"
                << " reason=requires_" << NvshmemP4::minimumPes(name) << "_PEs\n";
    return;
  }
  ImportStructureP50::require(static_cast<std::uint64_t>(session.numPes()) <=
      (static_cast<std::uint64_t>(std::numeric_limits<GO>::max()) - 20000ULL) / 1000000ULL,
      "global-ordinal range too small for P4 synthetic GIDs");
  const auto spec = NvshmemP4::buildSpec<GO>(name, session.mpiRank(), session.numPes());
  const auto comm = Tpetra::getDefaultComm();
  const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
  Teuchos::RCP<const Map> domain = Teuchos::rcp(
      new Map(invalid, spec.owned(), static_cast<GO>(0), comm));
  Teuchos::RCP<const Map> column = Teuchos::rcp(
      new Map(invalid, spec.target(), static_cast<GO>(0), comm));
  Teuchos::RCP<const Import> importer = Teuchos::rcp(new Import(domain, column));
  runCase(name, domain, column, importer, session, opts,
          rankCsvRoot, recvCsvRoot, sendCsvRoot, globalCsvRoot);
}

template<class LO, class GO, class Node>
void runMatrix(const NvshmemP0::Session& session, const Options& opts,
               int nsize, const std::string& filename,
               std::ostringstream& rankCsvRoot, std::ostringstream& recvCsvRoot,
               std::ostringstream& sendCsvRoot, std::ostringstream& globalCsvRoot) {
  using Matrix = Tpetra::CrsMatrix<double, LO, GO, Node>;
  using Map = Tpetra::Map<LO, GO, Node>;
  using Import = Tpetra::Import<LO, GO, Node>;
  const auto comm = Tpetra::getDefaultComm();
  Teuchos::RCP<Matrix> A;
  if (!filename.empty())
    A = Tpetra::MatrixMarket::Reader<Matrix>::readSparseFile(filename, comm);
  else
    A = Tpetra::Utils::MatrixGenerator<Matrix>::generate_miniFE_matrix(nsize, comm);
  ImportStructureP50::require(A->isFillComplete(), "matrix must be fill complete");
  Teuchos::RCP<const Map> domain = A->getDomainMap();
  Teuchos::RCP<const Map> column = A->getColMap();
  Teuchos::RCP<const Import> importer = NvshmemP1::ensureImporter(
      domain, column, A->getGraph()->getImporter(), session.comm());
  runCase("matrix", domain, column, importer, session, opts,
          rankCsvRoot, recvCsvRoot, sendCsvRoot, globalCsvRoot);
}

inline void writeCsv(const std::string& path, const std::string& value) {
  std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc);
  ImportStructureP50::require(out.is_open(), "cannot create CSV file " + path);
  out << value;
  out.flush();
  ImportStructureP50::require(out.good(), "failed to write CSV file " + path);
}

template<class LO, class GO, class Node>
void run(const NvshmemP0::Session& session, const Options& opts,
         int nsize, const std::string& filename) {
  checkOptions(opts);
  const auto comm = Tpetra::getDefaultComm();
  ImportStructureP50::require(comm->getRank() == session.mpiRank() &&
      comm->getSize() == session.numPes(), "Tpetra and NVSHMEM/MPI groups differ");
  if (session.mpiRank() == 0)
    std::cout << "[NVSHMEM P5.0] BEGIN ranks=" << session.numPes()
              << " only=" << opts.only
              << " analysisOnly=YES rmaIssued=0 performanceMeasured=NO\n";

  std::ostringstream rankCsv, recvCsv, sendCsv, globalCsv;
  rankCsv << rankCsvHeader(); recvCsv << recvCsvHeader();
  sendCsv << sendCsvHeader(); globalCsv << globalCsvHeader();
  int passed = 0, skipped = 0;
  for (const auto& name : NvshmemP4::cases()) {
    if (opts.only != "all" && opts.only != name) continue;
    if (name != "matrix" && session.numPes() < NvshmemP4::minimumPes(name)) {
      ++skipped;
      if (session.mpiRank() == 0)
        std::cout << "[NVSHMEM P5.0] case=" << name << " status=SKIP"
                  << " reason=requires_" << NvshmemP4::minimumPes(name) << "_PEs\n";
      continue;
    }
    if (name == "matrix")
      runMatrix<LO, GO, Node>(session, opts, nsize, filename,
                             rankCsv, recvCsv, sendCsv, globalCsv);
    else
      runSynthetic<LO, GO, Node>(name, session, opts,
                                rankCsv, recvCsv, sendCsv, globalCsv);
    ++passed;
  }
  ImportStructureP50::require(passed + skipped > 0, "no P5.0 cases selected");
  if (session.mpiRank() == 0) {
    if (!opts.csvPrefix.empty()) {
      writeCsv(opts.csvPrefix + ".rank.csv", rankCsv.str());
      writeCsv(opts.csvPrefix + ".recv.csv", recvCsv.str());
      writeCsv(opts.csvPrefix + ".send.csv", sendCsv.str());
      writeCsv(opts.csvPrefix + ".global.csv", globalCsv.str());
      std::cout << "[NVSHMEM P5.0] CSV prefix=" << opts.csvPrefix
                << " files=rank,recv,send,global\n";
    }
    std::cout << "[NVSHMEM P5.0] SUMMARY status=" << (passed ? "PASS" : "SKIP")
              << " casesAnalyzed=" << passed << " casesSkipped=" << skipped
              << " analysisOnly=YES rmaIssued=0 performanceMeasured=NO\n";
  }
}

} // namespace NvshmemP50
#endif // TPETRA_PERFORMANCE_NVSHMEM_P50_HPP
