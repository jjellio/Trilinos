// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
// P5.2.1 NVSHMEM-only, independently advancing PE/epoch stress harness.
// No Tpetra::doImport, MPI send/recv, MPI collectives or barriers are called
// inside the repeated GPU Import loop. P5.1 plan construction and symmetric
// NVSHMEM allocation/free are setup/retirement operations outside that loop.
#ifndef TPETRA_PERFORMANCE_NVSHMEM_P521_STRESS_HPP
#define TPETRA_PERFORMANCE_NVSHMEM_P521_STRESS_HPP

#include "NvshmemP4.hpp"
#include "NvshmemPackedPut.hpp"
#include "PackedPutPlan.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace NvshmemP521 {
using U64 = PackedPutP51::U64;

struct Config {
  int epochs = 128;
  int cycles = 2;
  int entryDelayMs = 3;  // delayed entry (signals may arrive before receive work)
  int ackDelayMs = 5;    // receiver stalls after inbox unpack, BEFORE ACK(g)
  int checkEvery = 1;    // compare ALL target LIDs at every kth epoch
  int seed = 17;
};

struct Result {
  U64 putCalls = 0;
  U64 putElements = 0;
  U64 readyWaits = 0;
  U64 creditWaits = 0;
  U64 ackSignals = 0;
  U64 checkedValues = 0;
  U64 checkedEpochs = 0;
  U64 injectedEntryDelays = 0;
  U64 injectedAckDelays = 0;
  U64 creditWaitNanos = 0;
  U64 maxCreditWaitNanos = 0;
  U64 globalSyncsInImport = 0;
  std::size_t symmetricBytesPerPE = 0;
};

inline void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("NVSHMEM P5.2.1: " + what);
}

// Explicit, deterministic, rank-dependent scheduling. This is stable across
// runs and never calls a collective. The rank's sequence is independent of
// its neighbors' progress. Varying seed changes which PE is delayed.
inline U64 mix(U64 x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
inline bool delayEntry(int rank, int epoch, int seed) {
  return (mix(static_cast<U64>(rank + 1) * 239017ULL +
              static_cast<U64>(epoch + 1) * 18617ULL +
              static_cast<U64>(seed)) % 4ULL) == 0ULL;
}
inline bool delayAck(int rank, int epoch, int seed, bool hasReceives) {
  if (!hasReceives) return false;
  // Rank zero is an intentionally slow receiver on some epochs. All other
  // receiver PEs take turns so fanout, multi-owner and uneven patterns also
  // exercise real consumption backpressure rather than just entry skew.
  if (rank == 0 && epoch % 4 == 0) return true;
  return (mix(static_cast<U64>(rank + 3) * 14591ULL +
              static_cast<U64>(epoch + 5) * 19073ULL +
              static_cast<U64>(seed)) % 5ULL) == 0ULL;
}

inline void checkConfig(const Config& cfg) {
  require(cfg.epochs >= 4 && cfg.epochs <= 1000, "epochs must be [4,1000]");
  require(cfg.cycles >= 2 && cfg.cycles <= 100, "cycles must be [2,100]");
  require(cfg.entryDelayMs >= 0 && cfg.entryDelayMs <= 1000,
          "entry delay must be [0,1000] ms");
  require(cfg.ackDelayMs >= 0 && cfg.ackDelayMs <= 1000,
          "ACK delay must be [0,1000] ms");
  require(cfg.checkEvery >= 1 && cfg.checkEvery <= cfg.epochs,
          "verification stride must be [1,epochs]");
  require(cfg.seed >= 0, "stress seed must be nonnegative");
}

// Independent oracle constructed from the original P1 owner/owner-local-LID
// descriptors and the known exact-in-binary64 P4 pattern. Crucially there is
// no MPI reference exchange or global collective anywhere in this function.
// The output is checked against an immutable expected source tuple, not
// against a communication method that could accidentally share stale data.
template<class LO, class GO, class Node>
std::size_t checkOneEpoch(const std::string& name, int cycle, int epoch,
                          const NvshmemP4::Pattern& pattern,
                          const PackedPutP51::Plan& plan,
                          const NvshmemP0::Session& session,
                          const Tpetra::Vector<double, LO, GO, Node>& target) {
  const auto got = target.getLocalViewHost(Tpetra::Access::ReadOnly);
  std::vector<unsigned char> visited(
      PackedPutP51::asSize(plan.targetLength, "P5.2.1 target coverage"), 0);
  const auto inspect = [&](U64 dest, int owner, U64 source, const char* kind) {
    require(dest < plan.targetLength && !visited[static_cast<std::size_t>(dest)],
            name + ": duplicated or invalid target column LID");
    visited[static_cast<std::size_t>(dest)] = 1;
    const double expected = pattern.value(epoch, owner, source);
    const double actual = got(static_cast<std::size_t>(dest), 0);
    if (std::isfinite(actual) && actual == expected) return;
    std::cerr << std::setprecision(17)
              << "[NVSHMEM P5.2.1 MISMATCH] case=" << name
              << " rank=" << session.mpiRank() << " cycle=" << cycle
              << " epoch=" << epoch << " kind=" << kind
              << " destinationLID=" << dest
              << " ownerRank=" << owner << " sourceLID=" << source
              << " expected=" << expected << " actual=" << actual << '\n';
    throw std::runtime_error("NVSHMEM P5.2.1: mismatched column value");
  };
  for (const auto& e : plan.local)
    inspect(e.destinationLid, session.mpiRank(), e.sourceLid, "local");
  for (const auto& e : plan.expectedRemote)
    inspect(e.destinationLid, e.ownerRank, e.sourceLid, "remote");
  for (unsigned char v : visited)
    require(v == 1, name + ": target column has a hole in the P1 partition");
  return visited.size();
}

// Called only *after* P5.1 has finished one-time MPI metadata construction.
// There is deliberately no reference Tpetra Import or P2 replica anywhere in
// this function, and no MPI call in either of the two epoch loops.
template<class LO, class GO, class Node>
Result runStressCase(const std::string& name,
                     const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& domain,
                     const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& column,
                     const PackedPutP51::Plan& plan,
                     const NvshmemP0::Session& session,
                     const Config& cfg) {
  using Vector = Tpetra::Vector<double, LO, GO, Node>;
  checkConfig(cfg);
  require(plan.finalized, "P5.1 packet plan must be finalized");
  const int totalEpochs = cfg.epochs * cfg.cycles;
  // P4::Pattern invokes MPI_Allreduce only HERE (before repeated imports)
  // to establish a globally collision-free owner/LID encoding.
  const NvshmemP4::Pattern pattern(
      PackedPutP51::asSize(plan.ownedLength, "P5.2.1 owned"),
      session, totalEpochs);
  Vector X(domain), Y(column);
  const double sentinel = std::numeric_limits<double>::quiet_NaN();
  Result result;

  for (int cycle = 0; cycle < cfg.cycles; ++cycle) {
    // NVSHMEM symmetric allocation/free is collective by API contract,
    // but occurs ONLY at lifecycle boundaries, not between import epochs.
    NvshmemPackedPut::Exchange<LO, GO, Node> exchange(plan, session);
    result.symmetricBytesPerPE = exchange.symmetricBytes();
    for (int i = 0; i < cfg.epochs; ++i) {
      const int epoch = cycle * cfg.epochs + i;
      if (cfg.entryDelayMs && delayEntry(session.mpiRank(), epoch, cfg.seed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg.entryDelayMs));
        ++result.injectedEntryDelays;
      }
      NvshmemP4::fillOwned(X, pattern, session, epoch); // GPU only
      Y.putScalar(sentinel);                            // local GPU only

      // Inject backpressure AFTER the GPU unpack but BEFORE per-peer ACK.
      // This is deliberately inside Exchange::import() and is only enabled
      // through its stress-only test hook, never during normal P5.2 usage.
      const bool holdAck = delayAck(session.mpiRank(), epoch, cfg.seed,
                                    !plan.receivePeers.empty());
      exchange.setStressAckDelayMs(holdAck ? cfg.ackDelayMs : 0);
      if (holdAck && cfg.ackDelayMs) ++result.injectedAckDelays;

      exchange.import(X, Y);  // NO MPI, barrier, collective launch, or oracle
      require(exchange.generation() == static_cast<U64>(i + 1),
              name + ": generation skipped or repeated");
      if (i % cfg.checkEvery == 0 || i + 1 == cfg.epochs) {
        result.checkedValues += checkOneEpoch<LO, GO, Node>(
            name, cycle, epoch, pattern, plan, session, Y);
        ++result.checkedEpochs;
      }
    }

    const auto& c = exchange.counters();
    const U64 epochs = static_cast<U64>(cfg.epochs);
    require(c.imports == epochs &&
            c.putCalls == epochs * plan.sendPeers.size() &&
            c.putElements == epochs * plan.packElements &&
            c.readyWaits == epochs * plan.receivePeers.size() &&
            c.ackSignals == c.readyWaits &&
            c.creditWaits == (epochs - 1) * plan.sendPeers.size() &&
            c.globalSyncsInImport == 0,
            name + ": PUT/ACK/credit counts disagree with P5.1 plan");
    result.putCalls += c.putCalls;
    result.putElements += c.putElements;
    result.readyWaits += c.readyWaits;
    result.creditWaits += c.creditWaits;
    result.ackSignals += c.ackSignals;
    result.creditWaitNanos += c.creditWaitNanos;
    result.maxCreditWaitNanos = std::max(
        result.maxCreditWaitNanos, c.maxCreditWaitNanos);
    result.globalSyncsInImport += c.globalSyncsInImport;
    // Collective retirement is distinct from the repeated Import path.
    exchange.release();
  }
  // Every verification pass compares the entire target, including every
  // remote and local LID; reject any silent reduction in verification rate.
  U64 checksPerCycle = 0;
  for (int i = 0; i < cfg.epochs; ++i)
    if (i % cfg.checkEvery == 0 || i + 1 == cfg.epochs) ++checksPerCycle;
  require(result.checkedEpochs == checksPerCycle * static_cast<U64>(cfg.cycles) &&
          result.checkedValues == result.checkedEpochs * plan.targetLength,
          name + ": stress verification coverage does not match configuration");
  return result;
}
} // namespace NvshmemP521
#endif // TPETRA_PERFORMANCE_NVSHMEM_P521_STRESS_HPP
