// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
// P6: allocation-excluded full Import comparison and stage-level timing.
// Maintains one persistent domain/column Map pair, Tpetra Importer, ordinary
// X/Y vectors, P5.1 packet plan, and NVSHMEM Exchange per selected case.
// Phase measurements reuse the SAME Exchange::import implementation.
#ifndef TPETRA_PERFORMANCE_NVSHMEM_P6_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P6_HPP

#include "NvshmemP4.hpp"
#include "PackedPutPlanMpi.hpp"
#include "NvshmemPackedPut.hpp"
#include "NvshmemP6Stats.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace NvshmemP6 {
using U64 = PackedPutP51::U64;
using Clock = std::chrono::steady_clock;
inline void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("NVSHMEM P6: " + what);
}
inline void mpiCheck(int code, const char* what) {
  require(code == MPI_SUCCESS, std::string("MPI call failed in ") + what);
}
inline double elapsedMicroseconds(const Clock::time_point& begin) {
  return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

struct Options {
  std::string only = "matrix"; // all P4 layouts optional; primary is miniFE matrix
  std::string mode = "both";    // both, direct, phases
  std::string putVariant = "thread"; // P6.1 thread (original)|warp|block
  int warmup = 8;
  int repetitions = 100;
  int phaseRepetitions = 100;
  int minimumNodes = 1;
  int maximumNodes = 0;
  bool requireCrossNode = false;
  std::string outputPrefix;  // optional rank-0 raw .csv and summary .csv
};
inline void checkOptions(const Options& opts) {
  const auto& cases = NvshmemP4::cases();
  require(opts.only == "all" ||
              std::find(cases.begin(), cases.end(), opts.only) != cases.end(),
          "invalid --nvshmem-p6-case");
  require(opts.mode == "both" || opts.mode == "direct" ||
              opts.mode == "phases", "--nvshmem-p6-mode must be both|direct|phases");
  require(opts.putVariant == "thread" || opts.putVariant == "warp" ||
              opts.putVariant == "block",
          "--nvshmem-p6-put-variant must be thread|warp|block");
  require(opts.warmup >= 2 && opts.warmup <= 1000, "warmup must be 2..1000");
  require(opts.repetitions >= 5 && opts.repetitions <= 5000,
          "repetitions must be 5..5000");
  require(opts.phaseRepetitions >= 5 && opts.phaseRepetitions <= 5000,
          "phase repetitions must be 5..5000");
  require(opts.minimumNodes >= 1 &&
              (opts.maximumNodes == 0 || opts.maximumNodes >= opts.minimumNodes),
          "invalid physical node count constraint");
}

// Root-only rank-major timing samples. Single MPI_Gather is performed only
// AFTER the entire phase sequence, not between individual phase iterations.
inline std::vector<double> gatherSamples(const std::vector<double>& local,
                                          const NvshmemP0::Session& session) {
  const int size = static_cast<int>(local.size());
  std::vector<double> root;
  if (session.mpiRank() == 0)
    root.resize(local.size() * static_cast<std::size_t>(session.numPes()));
  mpiCheck(MPI_Gather(local.data(), size, MPI_DOUBLE,
                      session.mpiRank() == 0 ? root.data() : nullptr,
                      size, MPI_DOUBLE, 0, session.comm()),
           "MPI_Gather(post-timing per-rank samples)");
  return root;
}

inline const char* phaseName(const std::size_t index) {
  static constexpr const char* names[] = {
      "credit_wait", "pack", "put_signal", "ready_wait", "unpack", "local_copy", "ack"};
  require(index < NvshmemPackedPut::PhaseCount, "invalid P6 phase index");
  return names[index];
}

struct CaseStats {
  bool direct = false;
  bool phases = false;
  NvshmemP6Stats::Summary tpetra;
  NvshmemP6Stats::Summary put;
  NvshmemP6Stats::Summary phaseTotal;
  U64 sends = 0;
  U64 doubles = 0;
  U64 crossNodeDoubles = 0;
  U64 validations = 0;
};

// The source X is changed OUTSIDE each timed section, using P4's exactly
// representable signed owner/rank/LID epoch encoding. The measured operations
// start with GPU-ready X and end with GPU-ready, fully populated column Y.
// No setup, symmetric malloc/free, source refill, verification or result
// reduction is inside a timed region.
template<class LO, class GO, class Node>
CaseStats benchmarkCase(
    const std::string& name,
    const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domain,
    const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
    const Teuchos::RCP<const Tpetra::Import<LO, GO, Node>>& importer,
    const NvshmemP0::Session& session,
    const NvshmemP4::Topology& topology,
    const Options& opts,
    bool assertNonzeroRemote) {
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  require(!importer.is_null(), name + ": missing Tpetra Importer");
  const auto comm = session.comm();
  const int ranks = session.numPes();
  const int me = session.mpiRank();
  std::vector<int> rankToPe(static_cast<std::size_t>(ranks), -1);
  for (int pe = 0; pe < ranks; ++pe)
    rankToPe[static_cast<std::size_t>(session.mpiRankForPe(pe))] = pe;
  const auto p1 = ImportGatherP1::buildImportGatherPlan(
      domain, column, *importer, rankToPe, comm);
  const auto plan = PackedPutP51::build(p1, session);
  require(plan.finalized, name + ": P5.1 packet plan not finalized");

  U64 crossLocal = 0;
  for (const auto& peer : plan.sendPeers) {
    if (topology.names[static_cast<std::size_t>(peer.rank)] !=
        topology.names[static_cast<std::size_t>(me)])
      crossLocal += peer.count;
  }
  const U64 localCounts[4] = {plan.packElements, plan.inboxElements,
      static_cast<U64>(plan.sendPeers.size()), crossLocal};
  U64 globalCounts[4] = {};
  mpiCheck(MPI_Allreduce(localCounts, globalCounts, 4, MPI_UINT64_T,
                         MPI_SUM, comm), "MPI_Allreduce(P6 setup message counts)");
  require(globalCounts[0] == globalCounts[1], name + ": unbalanced packet plan");
  if (assertNonzeroRemote)
    require(globalCounts[0] > 0, name + ": matrix unexpectedly has no remote Import");

  // One pattern allocation and one X, Ytpetra, Yput object for the ENTIRE
  // case, including warmups, direct comparison and instrumented phases.
  const bool direct = opts.mode != "phases";
  const bool phases = opts.mode != "direct";
  const int directCount = direct ? opts.repetitions : 0;
  const int phaseCount = phases ? opts.phaseRepetitions : 0;
  const int maxEpochs = opts.warmup + directCount + phaseCount + 2;
  const NvshmemP4::Pattern pattern(
      PackedPutP51::asSize(plan.ownedLength, "P6 owned X"), session, maxEpochs);
  Vector X(domain), Ytpetra(column), Yput(column);
  using PutExchange = NvshmemPackedPut::Exchange<LO, GO, Node>;
  PutExchange exchange(plan, session);
  exchange.setStressAckDelayMs(0);
  // Resolve a pointer to the requested, distinct Import entry point ONCE,
  // before warmup. The same dispatch cost applies to every variant and both
  // instrumented and direct measurements; no branch in the timed hot path.
  using PutMethod = void (PutExchange::*)(
      const Vector&, Vector&, NvshmemPackedPut::PhaseTimings*);
  PutMethod putMethod = &PutExchange::import; // untouched P5.2 thread baseline
  if (opts.putVariant == "warp") putMethod = &PutExchange::importWarp;
  if (opts.putVariant == "block") putMethod = &PutExchange::importThreadBlock;
  const auto callPut = [&](NvshmemPackedPut::PhaseTimings* timing = nullptr) {
    (exchange.*putMethod)(X, Yput, timing);
  };
  const char* putLabel = "nvshmem_packed_put"; // preserve historical P6 CSV
  if (opts.putVariant == "warp") putLabel = "nvshmem_packed_put_warp";
  if (opts.putVariant == "block") putLabel = "nvshmem_packed_put_block";

  // Allocate ALL harness timing arrays before the first warmup or timed call;
  // no vector growth or file I/O occurs within the measured loops.
  std::vector<double> localT(static_cast<std::size_t>(directCount), 0.0);
  std::vector<double> localP(static_cast<std::size_t>(directCount), 0.0);
  std::array<std::vector<double>, NvshmemPackedPut::PhaseCount + 1> localPhases;
  for (auto& v : localPhases)
    v.resize(static_cast<std::size_t>(phaseCount), 0.0);

  // Same ordinary source vector, same immutable Importer, identical target
  // requirements. Alternate order during warmup and direct measurements.
  for (int i = 0; i < opts.warmup; ++i) {
    NvshmemP4::fillOwned(X, pattern, session, i);
    if ((i & 1) == 0) {
      Ytpetra.doImport(X, *importer, Tpetra::INSERT);
      callPut();
    } else {
      callPut();
      Ytpetra.doImport(X, *importer, Tpetra::INSERT);
    }
    Kokkos::fence("P6 warmup completion and Tpetra Distributor scratch preparation");
  }

  // Spot check once outside the timers before sampling. The separate P5.2.1
  // acceptance test already checks ALL changing values on EVERY generation.
  const auto verify = [&](const Vector& Y, int epoch, const char* label) {
    auto got = Y.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (const auto& e : plan.local) {
      const double expected = pattern.value(epoch, me, e.sourceLid);
      require(got(e.destinationLid, 0) == expected,
              name + ": " + label + " local target value mismatch");
    }
    for (const auto& e : plan.expectedRemote) {
      const double expected = pattern.value(epoch, e.ownerRank, e.sourceLid);
      require(got(e.destinationLid, 0) == expected,
              name + ": " + label + " remote target value mismatch");
    }
  };
  verify(Ytpetra, opts.warmup - 1, "Tpetra warmup");
  verify(Yput, opts.warmup - 1, "NVSHMEM warmup");
  U64 validations = 2;
  // The host-side validation may switch Tpetra DualView residency. Restore
  // the ordinary CUDA destination views with an additional UNTImed Import
  // from the SAME source before the first measured sample.
  Ytpetra.doImport(X, *importer, Tpetra::INSERT);
  callPut();
  Kokkos::fence("P6 after oracle verification: device-resident warmup");

  if (me == 0)
    std::cout << "[NVSHMEM P6 SETUP] case=" << name
              << " ranks=" << ranks << " nodes=" << topology.distinctNodes
              << " outgoingPeers=" << globalCounts[2]
              << " packedDoublesPerImport=" << globalCounts[0]
              << " crossNodeDoublesPerImport=" << globalCounts[3]
              << " symmetricBytesPerPE=" << exchange.symmetricBytes()
              << " allocationScope=ONCE_PER_CASE"
              << " warmup=" << opts.warmup
              << " directReps=" << directCount << " phaseReps=" << phaseCount
              << " X=ORDINARY_CUDASPACE mode=" << opts.mode
              << " putVariant=" << opts.putVariant << '\n';

  if (direct) {
    for (int i = 0; i < directCount; ++i) {
      const int epoch = opts.warmup + i;
      NvshmemP4::fillOwned(X, pattern, session, epoch); // OUTSIDE timer
      for (int modeOrdinal = 0; modeOrdinal < 2; ++modeOrdinal) {
        const bool tpetra = ((modeOrdinal + i) % 2) == 0;
        // The boundary specifically requested by the benchmark specification.
        Kokkos::fence("P6 direct pre-Import device completion");
        mpiCheck(MPI_Barrier(comm), "MPI_Barrier(P6 direct start, OUTSIDE timer)");
        const auto started = Clock::now();
        if (tpetra) Ytpetra.doImport(X, *importer, Tpetra::INSERT);
        else callPut();
        Kokkos::fence("P6 direct post-Import complete Y");
        const double us = elapsedMicroseconds(started);
        (tpetra ? localT : localP)[static_cast<std::size_t>(i)] = us;
        // No MPI operations between the timed GPU finish and time capture.
        // Next iteration's barrier is OUTSIDE this sample.
      }
    }
    const int lastEpoch = opts.warmup + directCount - 1;
    verify(Ytpetra, lastEpoch, "Tpetra direct");
    verify(Yput, lastEpoch, "NVSHMEM direct");
    validations += 2;
  }

  // Verifying the final direct result may make Yput host-resident. Perform
  // one additional untimed NVSHMEM Import to restore its GPU destination
  // view before starting the separate instrumented experiment.
  if (direct && phases) {
    callPut();
    Kokkos::fence("P6 restore device residency before phase measurements");
  }

  // Different experiment: NO MPI or barrier between repetitions, exposing
  // natural peer skew, ready-wait latency and consumption-credit stalls.
  // Still measure wall time (including CUDA completion), now by exact stages.
  if (phases) {
    for (int i = 0; i < phaseCount; ++i) {
      const int epoch = opts.warmup + directCount + i;
      NvshmemP4::fillOwned(X, pattern, session, epoch); // OUTSIDE timer
      Kokkos::fence("P6 instrumented source ready (local only)");
      NvshmemPackedPut::PhaseTimings measurements;
      const auto started = Clock::now();
      callPut(&measurements);
      Kokkos::fence("P6 instrumented complete target");
      localPhases[NvshmemPackedPut::PhaseCount][static_cast<std::size_t>(i)] =
          elapsedMicroseconds(started);
      for (std::size_t k = 0; k < NvshmemPackedPut::PhaseCount; ++k)
        localPhases[k][static_cast<std::size_t>(i)] =
            static_cast<double>(measurements.nanoseconds[k]) / 1000.0;
    }
    verify(Yput, opts.warmup + directCount + phaseCount - 1, "NVSHMEM phases");
    ++validations;
  }

  // One Exchange allocation has serviced every warmup and measured Import.
  const U64 expectedImports = static_cast<U64>(opts.warmup + 1 +
      directCount + phaseCount + ((direct && phases) ? 1 : 0));
  require(exchange.counters().imports == expectedImports &&
          exchange.counters().globalSyncsInImport == 0,
          name + ": unexpected NVSHMEM Import count or global synchronization");

  // All timed Imports have completed before reductions/file I/O. No MPI
  // reduction exists in the instrumented repeated-Import loop.
  std::vector<double> rootT, rootP;
  if (direct) {
    rootT = gatherSamples(localT, session);
    rootP = gatherSamples(localP, session);
  }
  std::array<std::vector<double>, NvshmemPackedPut::PhaseCount + 1> rootPhases;
  if (phases)
    for (std::size_t k = 0; k < rootPhases.size(); ++k)
      rootPhases[k] = gatherSamples(localPhases[k], session);

  U64 totalValidations = 0;
  mpiCheck(MPI_Allreduce(&validations, &totalValidations, 1, MPI_UINT64_T,
                         MPI_SUM, comm), "MPI_Allreduce(P6 post-run verification)");
  CaseStats stats;
  stats.sends = globalCounts[2];
  stats.doubles = globalCounts[0];
  stats.crossNodeDoubles = globalCounts[3];
  stats.validations = totalValidations;
  stats.direct = direct;
  stats.phases = phases;
  if (me == 0) {
    std::ofstream raw;
    if (!opts.outputPrefix.empty()) {
      raw.open(opts.outputPrefix + "_" + name + "_samples.csv", std::ios::trunc);
      require(raw.good(), "cannot write P6 raw per-rank CSV");
      raw << "case,section,method,iteration,rank,us\n";
      raw << std::setprecision(12);
    }
    const auto report = [&](const char* section, const char* method,
                            const std::vector<double>& gathered, int reps) {
      if (raw.is_open())
        for (int r = 0; r < ranks; ++r)
          for (int i = 0; i < reps; ++i)
            raw << name << ',' << section << ',' << method << ',' << i << ',' << r << ','
                << gathered[static_cast<std::size_t>(r) * reps + i] << '\n';
      const auto worstPerIteration = NvshmemP6Stats::maxRankByIteration(
          gathered, ranks, reps);
      const auto s = NvshmemP6Stats::summarize(worstPerIteration);
      std::cout << std::fixed << std::setprecision(3)
                << "[NVSHMEM P6 " << section << "] case=" << name
                << " mode=" << method << " ranks=" << ranks
                << " scope=max_rank_per_iteration unit=us reps=" << reps
                << " min=" << s.min << " p10=" << s.p10
                << " median=" << s.median << " p90=" << s.p90
                << " max=" << s.max << '\n';
      return s;
    };
    if (direct) {
      stats.tpetra = report("DIRECT", "tpetra_doImport", rootT, directCount);
      stats.put = report("DIRECT", putLabel, rootP, directCount);
      // Compare the two methods on each matched source epoch as well as
      // reporting the ratio of their individual distribution medians.
      const auto pairedT = NvshmemP6Stats::maxRankByIteration(rootT, ranks, directCount);
      const auto pairedP = NvshmemP6Stats::maxRankByIteration(rootP, ranks, directCount);
      std::vector<double> pairedRatios(static_cast<std::size_t>(directCount), 0.0);
      for (int i = 0; i < directCount; ++i) {
        require(pairedP[static_cast<std::size_t>(i)] > 0.0,
                "P6 zero-time NVSHMEM sample prevents valid paired ratio");
        pairedRatios[static_cast<std::size_t>(i)] =
            pairedT[static_cast<std::size_t>(i)] /
            pairedP[static_cast<std::size_t>(i)];
      }
      const auto paired = NvshmemP6Stats::summarize(pairedRatios);
      std::cout << "[NVSHMEM P6 SPEEDUP] case=" << name
                << " putVariant=" << opts.putVariant
                << " metric=median_of_max_rank_tpetra_over_packed_put ratio="
                << (stats.put.median > 0.0 ? stats.tpetra.median / stats.put.median : 0.0)
                << " pairedMedian=" << paired.median
                << " pairedP10=" << paired.p10 << " pairedP90=" << paired.p90
                << " boundary=fence_barrier_start_import_fence_stop\n";
    }
    if (phases) {
      for (std::size_t k = 0; k < NvshmemPackedPut::PhaseCount; ++k)
        (void)report("PHASE", phaseName(k), rootPhases[k], phaseCount);
      stats.phaseTotal = report("PHASE", "nvshmem_total",
                                rootPhases[NvshmemPackedPut::PhaseCount], phaseCount);
    }
    if (raw.is_open()) raw.close();
    std::cout << "[NVSHMEM P6 CASE] case=" << name
              << " status=PASS verifiedOutputs=" << totalValidations
              << " importPayload=FULL_COLUMN_MAP doubleBytesPerImport="
              << (globalCounts[0] * sizeof(double))
              << " putVariant=" << opts.putVariant
              << " oneTimeSymmetricAllocation=YES"
              << " setupInTiming=NO sourceRefreshInTiming=NO"
              << " steadyStateCollectivesInExchange=0\n";
  }
  // Because nvshmem_free is collective, every rank leaves this function only
  // after completing its local measurements and post-run reporting.
  exchange.release();
  return stats;
}

template<class LO, class GO, class Node>
void run(const NvshmemP0::Session& session, const Options& options,
         int matrixSize, const std::string& filename) {
  checkOptions(options);
  const auto topology = NvshmemP4::getTopology(session);
  require(topology.distinctNodes >= options.minimumNodes &&
          (options.maximumNodes == 0 || topology.distinctNodes <= options.maximumNodes),
          "physical node count outside P6 requested topology");
  const auto comm = Tpetra::getDefaultComm();
  require(comm->getRank() == session.mpiRank() &&
          comm->getSize() == session.numPes(),
          "Tpetra and NVSHMEM communicator ranks disagree");
  if (session.mpiRank() == 0) {
    const char* transport = std::getenv("NVSHMEM_REMOTE_TRANSPORT");
    const char* ibgda = std::getenv("NVSHMEM_IB_ENABLE_IBGDA");
    std::cout << "[NVSHMEM P6] BEGIN ranks=" << session.numPes()
              << " physicalNodes=" << topology.distinctNodes
              << " case=" << options.only << " mode=" << options.mode
              << " putVariant=" << options.putVariant
              << " warmup=" << options.warmup
              << " repetitions=" << options.repetitions
              << " phaseRepetitions=" << options.phaseRepetitions
              << " transport=" << (transport ? transport : "default")
              << " ibgda=" << (ibgda ? ibgda : "default")
              << " direct=fence_barrier_start_import_fence_stop"
              << " phase=NO_PER_ITERATION_MPI"
              << " allocations=ONCE_PER_CASE\n";
  }
  int passed = 0;
  int skipped = 0;
  U64 crossDoubles = 0;
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
      require(A->isFillComplete(), "matrix not fill complete");
      const Teuchos::RCP<const Map> domain = A->getDomainMap();
      const Teuchos::RCP<const Map> column = A->getColMap();
      const Teuchos::RCP<const Import> importer = NvshmemP1::ensureImporter(
          domain, column, A->getGraph()->getImporter(), session.comm());
      const auto result = benchmarkCase(
          name, domain, column, importer, session, topology, options,
          filename.empty() && session.numPes() > 1);
      crossDoubles += result.crossNodeDoubles;
      ++passed;
    } else {
      if (session.numPes() < NvshmemP4::minimumPes(name)) {
        ++skipped;
        continue;
      }
      require(static_cast<U64>(session.numPes()) <=
                  (static_cast<U64>(std::numeric_limits<GO>::max()) - 20000ULL) /
                      1000000ULL,
              "synthetic sparse GIDs exceed GlobalOrdinal range");
      using Map = Tpetra::Map<LO, GO, Node>;
      using Import = Tpetra::Import<LO, GO, Node>;
      const auto spec = NvshmemP4::buildSpec<GO>(name, session.mpiRank(), session.numPes());
      const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
      const Teuchos::RCP<const Map> domain = Teuchos::rcp(
          new Map(invalid, spec.owned(), static_cast<GO>(0), comm));
      const Teuchos::RCP<const Map> column = Teuchos::rcp(
          new Map(invalid, spec.target(), static_cast<GO>(0), comm));
      const Teuchos::RCP<const Import> importer = Teuchos::rcp(new Import(domain, column));
      const auto result = benchmarkCase(
          name, domain, column, importer, session, topology, options, false);
      crossDoubles += result.crossNodeDoubles;
      ++passed;
    }
  }
  require(passed > 0, "P6 no case could be run");
  if (options.requireCrossNode)
    require(crossDoubles > 0, "P6 no actual cross-node packed PUTs in selected cases");
  if (session.mpiRank() == 0)
    std::cout << "[NVSHMEM P6] SUMMARY status=PASS casesPassed=" << passed
              << " casesSkipped=" << skipped << " ranks=" << session.numPes()
              << " mode=" << options.mode
              << " putVariant=" << options.putVariant
              << " sampleScope=max_rank_per_iteration"
              << " setupInTiming=NO perImportSymmetricAllocations=0\n";
}
} // namespace NvshmemP6
#endif // TPETRA_PERFORMANCE_NVSHMEM_P6_HPP
