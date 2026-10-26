// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
// P6.2 controlled comparison: MPI barrier before EVERY P6.2 exchange.
// Two GPU kernels / two streams, no ACK. Outside stopwatch: source update,
// start barrier, persistent plan construction, host validation, output.
#ifndef TPETRA_PERFORMANCE_NVSHMEM_P62_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P62_HPP
#include "NvshmemP6.hpp"  // P6 reference clock, per-rank gathering and stats
#include "PackedPutP62PlanMpi.hpp"
#include "NvshmemPackedPutP62.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace NvshmemP62 {
using U64 = PackedPutP51::U64;
using Clock = NvshmemP6::Clock;
using NvshmemP6::require;
using NvshmemP6::mpiCheck;
using NvshmemP6::elapsedMicroseconds;

struct Options {
  std::string only = "matrix";
  std::string mode = "direct"; // direct, diagnostics, both, p71
  std::string putVariant = "warp";
  int chunkDoubles = 512;
  int receiverBlocks = 4;
  int warmup = 5;
  int repetitions = 30;
  int diagnosticRepetitions = 30;
  int p71ProfileRepetitions = 12;   // only in --nvshmem-p62-mode=p71
  int p71MaxSamples = 64;          // sampled work units per category and PE
  int verifyEvery = 0; // 1 for every epoch, 0 for warmup and final check
  int minimumNodes = 1;
  int maximumNodes = 0;
  bool requireCrossNode = false;
  std::string outputPrefix;
};
inline void checkOptions(const Options& o) {
  const auto& cases = NvshmemP4::cases();
  require(o.only == "all" ||
              std::find(cases.begin(), cases.end(), o.only) != cases.end(),
          "P6.2 invalid case");
  require(o.mode == "direct" || o.mode == "diagnostics" || o.mode == "both" ||
              o.mode == "p71", "P6.2 mode must be direct|diagnostics|both|p71");
  if (o.mode == "p71")
    require(!o.outputPrefix.empty(),
            "P7.1 profiling requires --nvshmem-p62-out for rank-local trace files");
  (void)NvshmemPackedPutP62::parseMode(o.putVariant);
  require(o.chunkDoubles > 0,
          "P6.2 chunk size must be a positive integer");
  require(o.receiverBlocks >= 1 && o.receiverBlocks <= 16,
          "P6.2 receiver blocks must be 1..16");
  require(o.warmup >= 2 && o.warmup <= 1000 &&
          o.repetitions >= 5 && o.repetitions <= 5000 &&
          o.diagnosticRepetitions >= 5 && o.diagnosticRepetitions <= 5000,
          "P6.2 warmup/repetition count outside permitted range");
  require(o.p71ProfileRepetitions >= 2 && o.p71ProfileRepetitions <= 5000 &&
          o.p71MaxSamples >= 1 && o.p71MaxSamples <= 4096,
          "P7.1 repetitions must be 2..5000, sampled descriptors 1..4096");
  require(o.verifyEvery == 0 || o.verifyEvery == 1,
          "P6.2 verify-every must be 0 or 1");
  require(o.minimumNodes >= 1 &&
          (o.maximumNodes == 0 || o.maximumNodes >= o.minimumNodes),
          "P6.2 physical-node constraint invalid");
}

struct CaseStats {
  U64 crossNodeDoubles = 0;
  U64 validations = 0;
};

template<class LO, class GO, class Node>
CaseStats benchmarkCase(
    const std::string& name,
    const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domain,
    const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
    const Teuchos::RCP<const Tpetra::Import<LO, GO, Node>>& importer,
    const NvshmemP0::Session& session,
    const NvshmemP4::Topology& topology,
    const Options& opts,
    bool requireRemote) {
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  require(!importer.is_null(), "P6.2 missing Tpetra Importer");
  const int ranks = session.numPes();
  const int me = session.mpiRank();
  const MPI_Comm comm = session.comm();
  std::vector<int> rankToPe(static_cast<std::size_t>(ranks), -1);
  for (int pe = 0; pe < ranks; ++pe)
    rankToPe[static_cast<std::size_t>(session.mpiRankForPe(pe))] = pe;
  const auto p1 = ImportGatherP1::buildImportGatherPlan(
      domain, column, *importer, rankToPe, comm);
  const auto p51 = PackedPutP51::build(p1, session);
  const auto plan = PackedPutP62::build(
      p51, *domain, *column, opts.chunkDoubles, session);
  require(plan.finalized, "P6.2 descriptor plan was not finalized");
  U64 localCross = 0;
  for (const auto& peer : p51.sendPeers)
    if (topology.names[static_cast<std::size_t>(peer.rank)] !=
        topology.names[static_cast<std::size_t>(me)])
      localCross += peer.count;
  const U64 localCounts[5] = {
      p51.packElements, p51.inboxElements, static_cast<U64>(plan.sends.size()),
      static_cast<U64>(plan.receives.size()), localCross};
  U64 globalCounts[5] = {};
  mpiCheck(MPI_Allreduce(localCounts, globalCounts, 5, MPI_UINT64_T,
                         MPI_SUM, comm), "P6.2 setup descriptor counts");
  require(globalCounts[0] == globalCounts[1] &&
          globalCounts[2] == globalCounts[3],
          "P6.2 global descriptor/payload counts disagree");
  if (requireRemote)
    require(globalCounts[0] > 0, "P6.2 matrix unexpectedly contains no remote exports");

  const bool p71 = opts.mode == "p71";
  const bool direct = opts.mode != "diagnostics";
  const bool diagnostics = opts.mode == "diagnostics" || opts.mode == "both";
  const int directCount = direct ? opts.repetitions : 0;
  const int diagCount = diagnostics ? opts.diagnosticRepetitions : 0;
  const int profileCount = p71 ? opts.p71ProfileRepetitions : 0;
  const int maxEpoch = opts.warmup + directCount + diagCount + profileCount + 3;
  const NvshmemP4::Pattern pattern(
      PackedPutP51::asSize(p51.ownedLength, "P6.2 X length"), session, maxEpoch);
  Vector X(domain), Ytpetra(column), Yput(column);
  NvshmemPackedPutP62::Exchange<LO, GO, Node> exchange(
      p51, plan, NvshmemPackedPutP62::parseMode(opts.putVariant),
      opts.receiverBlocks, p71 ? static_cast<std::size_t>(opts.p71MaxSamples) : 0);

  // Every call performs Kokkos::fence() then MPI_Barrier() on the same
  // communicator before launching a new generation. This is the only
  // inter-PE synchronization in the steady-state P6.2 path.
  const auto barrierForImport = [&]() {
    Kokkos::fence("P6.2 prior GPU activity complete before start barrier");
    const auto started = Clock::now();
    mpiCheck(MPI_Barrier(comm), "P6.2 start-of-Import MPI_Barrier");
    return elapsedMicroseconds(started);
  };
  const auto restoreDeviceView = [&](Vector& Y) {
    // Host-value verification makes Y host-resident. Rehydrate it OUTSIDE
    // the next timing interval, before launch on nonblocking CUDA streams.
    (void)Y.getLocalViewDevice(Tpetra::Access::ReadWrite);
    Kokkos::fence("P6.2 restore CUDA destination after oracle comparison");
  };
  const auto verify = [&](const Vector& Y, int epoch, const char* method) {
    auto got = Y.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (const auto& e : p51.local) {
      const double expected = pattern.value(epoch, me, e.sourceLid);
      require(got(e.destinationLid, 0) == expected,
              name + ": P6.2 " + method + " local LID mismatch");
    }
    for (const auto& e : p51.expectedRemote) {
      const double expected = pattern.value(epoch, e.ownerRank, e.sourceLid);
      require(got(e.destinationLid, 0) == expected,
              name + ": P6.2 " + method + " remote LID mismatch");
    }
  };
  U64 validations = 0;

  for (int epoch = 0; epoch < opts.warmup; ++epoch) {
    NvshmemP4::fillOwned(X, pattern, session, epoch);
    // The Tpetra oracle and P6.2 produce independent, disjoint Y vectors.
    Ytpetra.doImport(X, *importer, Tpetra::INSERT);
    (void)barrierForImport();
    exchange.importAfterBarrier(X, Yput);
    if (opts.verifyEvery) {
      verify(Ytpetra, epoch, "Tpetra warmup");
      verify(Yput, epoch, "P6.2 warmup");
      validations += 2;
      restoreDeviceView(Ytpetra);
      restoreDeviceView(Yput);
    }
  }
  verify(Ytpetra, opts.warmup - 1, "Tpetra final warmup");
  verify(Yput, opts.warmup - 1, "P6.2 final warmup");
  validations += 2;
  restoreDeviceView(Ytpetra);
  restoreDeviceView(Yput);

  if (me == 0)
    std::cout << "[NVSHMEM P6.2 SETUP] case=" << name
              << " chunkDoubles=" << opts.chunkDoubles
              << " putVariant=" << opts.putVariant
              << " receiverBlocks=" << exchange.receiverBlocks()
              << " totalPutDescriptors=" << globalCounts[2]
              << " totalReceiveSignals=" << globalCounts[3]
              << " packedDoubles=" << globalCounts[0]
              << " crossNodeDoubles=" << globalCounts[4]
              << " symmetricBytesPerPE=" << exchange.symmetricBytes()
              << " persistentAllocations=YES twoCudaKernels=YES noACK=YES"
              << " mpiBarrierBeforeEveryImport=YES\n";

  std::vector<double> localT(static_cast<std::size_t>(directCount));
  std::vector<double> localP(static_cast<std::size_t>(directCount));
  std::vector<double> localBarrier(static_cast<std::size_t>(directCount));
  // P7.1: rank-matched *barrier+Import* samples, not sums of separate
  // maximum-rank medians. Separate per-method barriers prevent mispairing.
  std::vector<double> localTFull(static_cast<std::size_t>(directCount));
  std::vector<double> localPFull(static_cast<std::size_t>(directCount));
  std::vector<double> localTBarrier(static_cast<std::size_t>(directCount));
  // Diagnostic timings are host wall-clock stage durations and include
  // CUDA launch submissions and final wait; NOT device/NIC wire durations.
  std::array<std::vector<double>, 5> localDiag;
  for (auto& series : localDiag)
    series.resize(static_cast<std::size_t>(diagCount), 0.0);

  if (direct) {
    for (int i = 0; i < directCount; ++i) {
      const int epoch = opts.warmup + i;
      NvshmemP4::fillOwned(X, pattern, session, epoch);
      for (int method = 0; method < 2; ++method) {
        const bool tpetra = ((method + i) & 1) == 0;
        const double barrierUs = barrierForImport();
        const auto started = Clock::now();
        if (tpetra) Ytpetra.doImport(X, *importer, Tpetra::INSERT);
        else exchange.importAfterBarrier(X, Yput);
        Kokkos::fence("P6.2 complete Import for matched measurement");
        const double duration = elapsedMicroseconds(started);
        (tpetra ? localT : localP)[static_cast<std::size_t>(i)] = duration;
        (tpetra ? localTFull : localPFull)[static_cast<std::size_t>(i)] =
            duration + barrierUs;
        if (!tpetra) localBarrier[static_cast<std::size_t>(i)] = barrierUs;
        else localTBarrier[static_cast<std::size_t>(i)] = barrierUs;
        if (opts.verifyEvery) {
          verify(tpetra ? Ytpetra : Yput, epoch,
                 tpetra ? "Tpetra direct" : "P6.2 direct");
          ++validations;
          restoreDeviceView(tpetra ? Ytpetra : Yput);
        }
      }
    }
    const int lastEpoch = opts.warmup + directCount - 1;
    verify(Ytpetra, lastEpoch, "Tpetra final direct");
    verify(Yput, lastEpoch, "P6.2 final direct");
    validations += 2;
    restoreDeviceView(Ytpetra);
    restoreDeviceView(Yput);
  }

  if (diagnostics) {
    for (int i = 0; i < diagCount; ++i) {
      const int epoch = opts.warmup + directCount + i;
      NvshmemP4::fillOwned(X, pattern, session, epoch);
      const double barrierUs = barrierForImport();
      NvshmemPackedPutP62::HostTimings times;
      const auto started = Clock::now();
      exchange.importAfterBarrier(X, Yput, &times);
      Kokkos::fence("P6.2 instrumented device completion");
      localDiag[0][static_cast<std::size_t>(i)] = barrierUs;
      localDiag[1][static_cast<std::size_t>(i)] = times.producerLaunchUs;
      localDiag[2][static_cast<std::size_t>(i)] = times.receiverLaunchUs;
      localDiag[3][static_cast<std::size_t>(i)] = times.completeUs;
      localDiag[4][static_cast<std::size_t>(i)] = elapsedMicroseconds(started);
      if (opts.verifyEvery) {
        verify(Yput, epoch, "P6.2 diagnostics");
        ++validations;
        restoreDeviceView(Yput);
      }
    }
    verify(Yput, opts.warmup + directCount + diagCount - 1,
           "P6.2 final diagnostics");
    ++validations;
  }

  // P7.1: diagnostic-only, separately instrumented clones of the frozen B0
  // GPU kernels. The direct, non-instrumented paired comparison has already
  // completed before entering this section. Never compare instrumented GPU
  // timing directly to Tpetra as though the probes had zero overhead.
  std::array<std::vector<double>, 8> localP71;
  for (auto& series : localP71)
    series.resize(static_cast<std::size_t>(profileCount), 0.0);
  std::ostringstream p71Raw;
  if (p71) {
    p71Raw << std::setprecision(12);
    p71Raw << "case,rank,epoch,kind,descriptor_id,peer_rank,peer_pe,route,"
              "doubles,first_us,second_us,poll_attempts,clock_khz,sample_stride\n";
    for (int i = 0; i < profileCount; ++i) {
      const int epoch = opts.warmup + directCount + diagCount + i;
      NvshmemP4::fillOwned(X, pattern, session, epoch);
      const double barrierUs = barrierForImport();
      NvshmemPackedPutP62::P71ProfileResults result;
      exchange.importProfiledAfterBarrier(X, Yput, result);
      // The profiled import returns only after BOTH streams complete, and
      // collects probes after that completion. Readback is not in the event
      // durations or the recorded completed-GPU-Import host interval.
      localP71[0][static_cast<std::size_t>(i)] = barrierUs;
      localP71[1][static_cast<std::size_t>(i)] = result.host.producerLaunchUs;
      localP71[2][static_cast<std::size_t>(i)] = result.host.receiverLaunchUs;
      localP71[3][static_cast<std::size_t>(i)] = result.host.completeUs;
      localP71[4][static_cast<std::size_t>(i)] = result.sendKernelEventUs;
      localP71[5][static_cast<std::size_t>(i)] = result.receiveKernelEventUs;
      localP71[6][static_cast<std::size_t>(i)] = result.host.totalUs;
      localP71[7][static_cast<std::size_t>(i)] = result.readbackUs;
      const auto emitProbe = [&](const char* kind, std::size_t id,
                                 int peerPe, U64 count,
                                 const NvshmemP71Stats::Probe& probe,
                                 std::size_t stride) {
        const int peerRank = peerPe >= 0 ? session.mpiRankForPe(peerPe) : me;
        require(peerRank >= 0 && peerRank < ranks, "P7.1 invalid peer in probe");
        const char* route = peerRank == me ? "local" :
            topology.names[static_cast<std::size_t>(peerRank)] ==
            topology.names[static_cast<std::size_t>(me)] ? "intra" : "inter";
        p71Raw << name << ',' << me << ',' << epoch << ',' << kind << ',' << id
               << ',' << peerRank << ',' << peerPe << ',' << route << ',' << count
               << ',' << NvshmemP71Stats::cyclesToMicroseconds(
                          probe.firstCycles, result.clockKHz)
               << ',' << NvshmemP71Stats::cyclesToMicroseconds(
                          probe.secondCycles, result.clockKHz)
               << ',' << probe.pollAttempts << ',' << result.clockKHz
               << ',' << stride << '\n';
      };
      for (std::size_t j = 0; j < result.send.size(); ++j) {
        const std::size_t id = NvshmemP71Stats::sampledId(result.sendSampling, j);
        emitProbe("pack_and_put_api_return", id, plan.sends[id].pe,
                  plan.sends[id].count, result.send[j], result.sendSampling.stride);
      }
      for (std::size_t j = 0; j < result.receive.size(); ++j) {
        const std::size_t id = NvshmemP71Stats::sampledId(result.receiveSampling, j);
        emitProbe("signal_poll_and_scatter", id, plan.receives[id].sourcePe,
                  plan.receives[id].count, result.receive[j], result.receiveSampling.stride);
      }
      for (std::size_t j = 0; j < result.local.size(); ++j) {
        const std::size_t id = NvshmemP71Stats::sampledId(result.localSampling, j);
        emitProbe("local_copy", id, session.pe(), 1, result.local[j],
                  result.localSampling.stride);
      }
      if (opts.verifyEvery) {
        verify(Yput, epoch, "P7.1 profiled GPU Import");
        ++validations;
        restoreDeviceView(Yput);
      }
    }
    verify(Yput, opts.warmup + directCount + diagCount + profileCount - 1,
           "P7.1 final profiled GPU Import");
    ++validations;
    // One file per rank, unique filenames: no concurrent MPI filesystem
    // writes to rank 0's output; no I/O during timed GPU work.
    const std::string path = opts.outputPrefix + "_" + name +
        "_p71_rank" + std::to_string(me) + "_probes.csv";
    std::ofstream file(path, std::ios::trunc);
    require(file.good(), "P7.1 could not open local probe CSV " + path);
    file << p71Raw.str();
    require(file.good(), "P7.1 could not write local probe CSV " + path);
  }

  const U64 expectedImports = static_cast<U64>(opts.warmup + directCount + diagCount + profileCount);
  const auto& c = exchange.counters();
  require(c.imports == expectedImports &&
          c.producerKernelLaunches == expectedImports &&
          c.receiverKernelLaunches == expectedImports &&
          c.putCalls == expectedImports * static_cast<U64>(plan.sends.size()) &&
          c.expectedSignals == expectedImports * static_cast<U64>(plan.receives.size()) &&
          c.acknowledgementSignals == 0,
          name + ": P6.2 launch/signal counters violate two-kernel model");

  std::vector<double> rootT, rootP, rootBarrier, rootTFull, rootPFull, rootTBarrier;
  if (direct) {
    rootT = NvshmemP6::gatherSamples(localT, session);
    rootP = NvshmemP6::gatherSamples(localP, session);
    rootBarrier = NvshmemP6::gatherSamples(localBarrier, session);
    if (p71) {
      rootTFull = NvshmemP6::gatherSamples(localTFull, session);
      rootPFull = NvshmemP6::gatherSamples(localPFull, session);
      rootTBarrier = NvshmemP6::gatherSamples(localTBarrier, session);
    }
  }
  std::array<std::vector<double>, 5> rootDiag;
  if (diagnostics)
    for (std::size_t i = 0; i < rootDiag.size(); ++i)
      rootDiag[i] = NvshmemP6::gatherSamples(localDiag[i], session);
  std::array<std::vector<double>, 8> rootP71;
  if (p71)
    for (std::size_t i = 0; i < rootP71.size(); ++i)
      rootP71[i] = NvshmemP6::gatherSamples(localP71[i], session);
  U64 globalValidations = 0;
  mpiCheck(MPI_Allreduce(&validations, &globalValidations, 1, MPI_UINT64_T,
                         MPI_SUM, comm), "P6.2 verification count");
  if (me == 0) {
    std::ofstream raw;
    if (!opts.outputPrefix.empty()) {
      raw.open(opts.outputPrefix + "_" + name + "_p62_samples.csv", std::ios::trunc);
      require(raw.good(), "P6.2 cannot create raw timing CSV");
      raw << "case,section,method,iteration,rank,us\n";
      raw << std::setprecision(12);
    }
    const auto report = [&](const char* section, const char* method,
                            const std::vector<double>& samples, int count) {
      if (raw.is_open())
        for (int r = 0; r < ranks; ++r)
          for (int i = 0; i < count; ++i)
            raw << name << ',' << section << ',' << method << ',' << i << ',' << r
                << ',' << samples[static_cast<std::size_t>(r) * count + i] << '\n';
      const auto s = NvshmemP6Stats::summarize(
          NvshmemP6Stats::maxRankByIteration(samples, ranks, count));
      std::cout << std::fixed << std::setprecision(3)
                << "[NVSHMEM P6.2 " << section << "] case=" << name
                << " method=" << method << " putVariant=" << opts.putVariant
                << " chunkDoubles=" << opts.chunkDoubles
                << " scope=max_rank_per_iteration unit=us reps=" << count
                << " median=" << s.median << " p10=" << s.p10
                << " p90=" << s.p90 << " min=" << s.min
                << " max=" << s.max << '\n';
      return s;
    };
    if (direct) {
      const auto t = report("DIRECT", "tpetra_doImport", rootT, directCount);
      const auto p = report("DIRECT", "descriptor_two_stream", rootP, directCount);
      (void)report("BARRIER", "outside_gpu_import_timing", rootBarrier, directCount);
      const auto ts = NvshmemP6Stats::maxRankByIteration(rootT, ranks, directCount);
      const auto ps = NvshmemP6Stats::maxRankByIteration(rootP, ranks, directCount);
      std::vector<double> ratio(static_cast<std::size_t>(directCount));
      for (int i = 0; i < directCount; ++i) {
        require(ps[static_cast<std::size_t>(i)] > 0, "P6.2 invalid zero duration");
        ratio[static_cast<std::size_t>(i)] =
            ts[static_cast<std::size_t>(i)] / ps[static_cast<std::size_t>(i)];
      }
      const auto paired = NvshmemP6Stats::summarize(ratio);
      std::cout << "[NVSHMEM P6.2 SPEEDUP] case=" << name
                << " tpetraMedianOverP62Median="
                << (p.median > 0 ? t.median / p.median : 0)
                << " pairedMedian=" << paired.median
                << " pairedP10=" << paired.p10
                << " pairedP90=" << paired.p90
                << " boundary=fence_barrier_outside_start_import_fence_stop\n";
    }
    if (p71) {
      (void)report("P71_FULL", "tpetra_barrier_plus_import", rootTFull, directCount);
      (void)report("P71_FULL", "p62_barrier_plus_import", rootPFull, directCount);
      (void)report("P71_BARRIER", "tpetra_start_barrier", rootTBarrier, directCount);
      // Direct reference samples above are entirely uninstrumented.
      const auto a = NvshmemP6Stats::summarize(
          NvshmemP6Stats::maxRankByIteration(rootT, ranks, directCount));
      const auto b = NvshmemP6Stats::summarize(
          NvshmemP6Stats::maxRankByIteration(rootPFull, ranks, directCount));
      std::cout << "[NVSHMEM P7.1 CONTROL] case=" << name
                << " mpiImportOnlyMedian=" << a.median
                << " p62WithBarrierMedian=" << b.median
                << " p62FullOverMpiImportOnly="
                << (a.median > 0 ? b.median / a.median : 0)
                << " deployable=NO_gpu_diagnostic_only\n";
      static const char* p71Labels[] = {
          "pre_start_barrier", "producer_host_issue_with_events",
          "receiver_host_issue_with_events", "cudaDeviceSynchronize_host_wait",
          "producer_cuda_kernel_event", "receiver_cuda_kernel_event",
          "instrumented_gpu_import_host", "post_import_probe_readback"};
      for (std::size_t k = 0; k < rootP71.size(); ++k)
        (void)report("P71_PROFILE", p71Labels[k], rootP71[k], profileCount);
      std::cout << "[NVSHMEM P7.1 PROFILE] case=" << name
                << " protocol=unchanged_put_signal_per_descriptor"
                << " gpuProbeClock=clock64_approximate"
                << " cudaEvents=per_stream_not_remote_network_latency"
                << " quietCalls=0 fenceCalls=0 ackCalls=0"
                << " receiverTestsPerEpoch=" << globalCounts[3]
                << " gpuKernelLaunchesPerEpoch=2"
                << " probeFiles=per_rank_profiler_output"
                << " status=PASS\n";
    }
    if (diagnostics) {
      static const char* labels[] = {"pre_start_mpi_barrier", "producer_host_issue",
          "receiver_host_issue", "device_completion_host_wait", "total_gpu_import"};
      for (std::size_t k = 0; k < rootDiag.size(); ++k)
        (void)report("DIAGNOSTIC", labels[k], rootDiag[k], diagCount);
    }
    std::cout << "[NVSHMEM P6.2 CASE] case=" << name
              << " status=PASS verifiedOutputs=" << globalValidations
              << " putVariant=" << opts.putVariant
              << " chunkDoubles=" << opts.chunkDoubles
              << " putsPerImport=" << globalCounts[2]
              << " readySignalsPerImport=" << globalCounts[3]
              << " totalDoublesPerImport=" << globalCounts[0]
              << " twoKernelLaunchesPerPE=2 ackSignals=0"
              << " mpiBarrierPerP62Import=1 setupInTiming=NO\n";
  }
  exchange.release();
  return {globalCounts[4], globalValidations};
}

template<class LO, class GO, class Node>
void run(const NvshmemP0::Session& session, const Options& options,
         int matrixSize, const std::string& filename) {
  checkOptions(options);
  const auto topology = NvshmemP4::getTopology(session);
  require(topology.distinctNodes >= options.minimumNodes &&
          (options.maximumNodes == 0 || topology.distinctNodes <= options.maximumNodes),
          "P6.2 physical node requirement not met");
  const auto comm = Tpetra::getDefaultComm();
  require(comm->getRank() == session.mpiRank() &&
          comm->getSize() == session.numPes(),
          "P6.2 Tpetra and NVSHMEM MPI rank order disagree");
  if (session.mpiRank() == 0) {
    std::cout << "[NVSHMEM P6.2] BEGIN case=" << options.only
              << " mode=" << options.mode
              << " putVariant=" << options.putVariant
              << " chunkDoubles=" << options.chunkDoubles
              << " receiverBlocks=" << options.receiverBlocks
              << " ranks=" << session.numPes()
              << " nodes=" << topology.distinctNodes
              << " warmup=" << options.warmup
              << " reps=" << options.repetitions
              << " diagnosticReps=" << options.diagnosticRepetitions
              << " p71ProfileReps=" << options.p71ProfileRepetitions
              << " p71MaxSamples=" << options.p71MaxSamples
              << " verifyEvery=" << options.verifyEvery
              << " MPI_Barrier=START_EACH_EXCHANGE"
              << " ACK=NONE twoKernels=YES\n";
  }
  U64 crossNodeDoubles = 0;
  int passed = 0, skipped = 0;
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
      require(A->isFillComplete(), "P6.2 matrix is not fill complete");
      const Teuchos::RCP<const Map> domain = A->getDomainMap();
      const Teuchos::RCP<const Map> column = A->getColMap();
      const Teuchos::RCP<const Import> importer = NvshmemP1::ensureImporter(
          domain, column, A->getGraph()->getImporter(), session.comm());
      const auto s = benchmarkCase(name, domain, column, importer,
          session, topology, options, filename.empty() && session.numPes() > 1);
      crossNodeDoubles += s.crossNodeDoubles;
      ++passed;
    } else {
      if (session.numPes() < NvshmemP4::minimumPes(name)) {
        ++skipped;
        continue;
      }
      require(static_cast<U64>(session.numPes()) <=
                  (static_cast<U64>(std::numeric_limits<GO>::max()) - 20000ULL) /
                      1000000ULL,
              "P6.2 synthetic GIDs would exceed ordinal range");
      using Map = Tpetra::Map<LO, GO, Node>;
      using Import = Tpetra::Import<LO, GO, Node>;
      const auto spec = NvshmemP4::buildSpec<GO>(name, session.mpiRank(), session.numPes());
      const auto invalid = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
      const Teuchos::RCP<const Map> domain = Teuchos::rcp(
          new Map(invalid, spec.owned(), static_cast<GO>(0), comm));
      const Teuchos::RCP<const Map> column = Teuchos::rcp(
          new Map(invalid, spec.target(), static_cast<GO>(0), comm));
      const Teuchos::RCP<const Import> importer = Teuchos::rcp(new Import(domain, column));
      const auto s = benchmarkCase(name, domain, column, importer,
          session, topology, options, false);
      crossNodeDoubles += s.crossNodeDoubles;
      ++passed;
    }
  }
  require(passed > 0, "P6.2 no case could run");
  if (options.requireCrossNode)
    require(crossNodeDoubles > 0, "P6.2 no actual cross-node payload in chosen cases");
  if (session.mpiRank() == 0)
    std::cout << "[NVSHMEM P6.2] SUMMARY status=PASS"
              << " casesPassed=" << passed << " casesSkipped=" << skipped
              << " ranks=" << session.numPes() << " chunkDoubles=" << options.chunkDoubles
              << " putVariant=" << options.putVariant
              << " barrierBetweenGenerations=YES ACK=NONE\n";
}
} // namespace NvshmemP62
#endif
