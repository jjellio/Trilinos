// P5.1 plan plus P5.2 asynchronously scheduled per-peer credit protocol model.
// CPU-only (no MPI, Trilinos, CUDA or NVSHMEM): hardware acceptance still required.
#include "PackedPutPlan.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace PackedPutP51;

void verify(bool ok, const std::string& text) {
  if (!ok) throw std::runtime_error("P5.1 UNIT: " + text);
}
template<class F> void expectError(F f, const std::string& text) {
  bool threw = false;
  try { f(); } catch (const std::runtime_error&) { threw = true; }
  verify(threw, "invalid metadata accepted: " + text);
}

struct Trial {
  std::vector<RankInput> inputs;
  std::vector<Plan> localPlans;
  std::vector<PeerMetadata> metadata;
};

Trial construct(const std::vector<RankInput>& input) {
  Trial t;
  t.inputs = input;
  const std::size_t n = input.size();
  verify(n > 0, "empty trial group");
  for (const auto& rank : input) t.localPlans.push_back(buildLocal(rank));
  U64 maxPack = 0, maxInbox = 0;
  for (const auto& p : t.localPlans) {
    maxPack = std::max(maxPack, p.packElements);
    maxInbox = std::max(maxInbox, p.inboxElements);
  }
  t.metadata.resize(n);
  for (std::size_t receiver = 0; receiver < n; ++receiver) {
    auto& m = t.metadata[receiver];
    m.sourceLidsFromRank.resize(n);
    m.inboxOffsetAtRank.resize(n, noOffset);
    m.inboxLengthAtRank.resize(n);
    m.maxPackElements = maxPack;
    m.maxInboxElements = maxInbox;
    for (std::size_t sender = 0; sender < n; ++sender) {
      m.inboxLengthAtRank[sender] = t.localPlans[sender].inboxElements;
      for (const auto& send : t.localPlans[sender].sendPeers) {
        if (send.rank != static_cast<int>(receiver)) continue;
        for (U64 i = 0; i < send.count; ++i)
          m.sourceLidsFromRank[sender].push_back(
              t.localPlans[sender].packedSourceLids[static_cast<std::size_t>(
                  send.packOffset + i)]);
      }
    }
    for (const auto& send : t.localPlans[receiver].sendPeers) {
      const std::size_t remoteReceiver = static_cast<std::size_t>(send.rank);
      bool found = false;
      for (const auto& recv : t.localPlans[remoteReceiver].receivePeers) {
        if (recv.rank != static_cast<int>(receiver)) continue;
        verify(!found && recv.count == send.count, "nonunique peer/count mismatch");
        m.inboxOffsetAtRank[remoteReceiver] = recv.inboxOffset;
        found = true;
      }
      verify(found, "missing receiving PE segment for sender");
    }
  }
  return t;
}

U64 expectedValue(int ownerRank, U64 sourceLid) {
  return static_cast<U64>(ownerRank + 1) * 1000000ULL + sourceLid;
}

// Non-collective simulation of the exact P5.2 ready/consumed protocol. Rank
// schedules are independently and adversarially interleaved. A sender waits
// only for the destinations to which it exports, and a receiver only for its
// actual source PEs. A PE with no traffic may run arbitrarily far ahead.
// Per-slot stamps catch premature overwrite; stale-generation checks catch
// accepting a previous PUT as the next epoch; full owner/LID comparison
// catches packet ordering and incorrect remote/local scatter.
inline double changingValue(int epoch, int owner, U64 lid) {
  const U64 integer = static_cast<U64>(epoch + 1) * 100000000ULL +
                      static_cast<U64>(owner + 1) * 10000ULL + lid;
  const double magnitude = static_cast<double>(integer) * 0.125;
  return ((static_cast<U64>(epoch) + owner + lid) & 1ULL) ? -magnitude : magnitude;
}

struct SimRank {
  int epoch = 1;              // 1..8, corresponds to one allocation lifetime
  int phase = 0;              // wait credit, pack, send, wait ready, unpack, ACK
  std::vector<double> packed;
  std::vector<double> output;
};

void exerciseCreditProtocol(const Trial& trial) {
  const std::size_t n = trial.localPlans.size();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // Two independent collective malloc/free cycles; both zero signal arrays.
  for (int cycle = 0; cycle < 2; ++cycle) {
    const int maxEpoch = 4;
    std::vector<SimRank> states(n);
    std::vector<std::vector<double>> inbox(n);
    std::vector<std::vector<int>> writeStamp(n), readStamp(n);
    // Here the rank->PE table is deliberately permuted by the test generator.
    std::vector<std::vector<int>> ready(n, std::vector<int>(n, 0));
    std::vector<std::vector<int>> credit(n, std::vector<int>(n, 0));
    for (std::size_t rank = 0; rank < n; ++rank) {
      const auto& plan = trial.localPlans[rank];
      inbox[rank].assign(asSize(plan.symmetricInboxElements, "model inbox"), nan);
      writeStamp[rank].assign(asSize(plan.inboxElements, "model stamp"), 0);
      readStamp[rank].assign(asSize(plan.inboxElements, "model stamp"), 0);
    }
    std::mt19937_64 schedule(0x52A11C0EDULL + n * 33 + cycle * 97 +
                             trial.inputs[0].targetLength);
    const std::size_t maxSteps = n * static_cast<std::size_t>(maxEpoch) * 6 + 5;
    std::size_t completed = 0;
    std::size_t scheduled = 0;
    while (completed != n) {
      verify(scheduled <= maxSteps, "P5.2 credit/ready deadlock or infinite loop");
      std::vector<std::size_t> runnable;
      for (std::size_t rank = 0; rank < n; ++rank) {
        auto& state = states[rank];
        if (state.epoch > maxEpoch) continue;
        const auto& plan = trial.localPlans[rank];
        if (state.phase == 0 && state.epoch > 1) {
          bool creditsReady = true;
          for (const auto& send : plan.sendPeers)
            creditsReady &= (credit[rank][static_cast<std::size_t>(send.pe)] ==
                             state.epoch - 1);
          if (!creditsReady) continue;
        }
        if (state.phase == 3) {
          bool packetsReady = true;
          for (const auto& recv : plan.receivePeers)
            packetsReady &= (ready[rank][static_cast<std::size_t>(recv.pe)] ==
                             state.epoch);
          if (!packetsReady) continue;
        }
        runnable.push_back(rank);
      }
      verify(!runnable.empty(), "P5.2 model cyclic wait / no runnable rank");
      const auto rank = runnable[static_cast<std::size_t>(schedule() % runnable.size())];
      auto& state = states[rank];
      const auto& plan = trial.localPlans[rank];
      const int generation = state.epoch;
      const int patternEpoch = cycle * maxEpoch + generation - 1;
      switch (state.phase) {
        case 0: // credit waited ONLY for send peers, not all ranks
          if (generation > 1)
            for (const auto& send : plan.sendPeers)
              verify(credit[rank][static_cast<std::size_t>(send.pe)] == generation - 1,
                     "sender advanced without receiver-consumed ACK");
          state.phase = 1;
          break;
        case 1: // ordinary X -> symmetric sender scratch
          state.packed.assign(asSize(plan.packElements, "model pack"), nan);
          for (std::size_t i = 0; i < state.packed.size(); ++i)
            state.packed[i] = changingValue(patternEpoch, static_cast<int>(rank),
                                          plan.packedSourceLids[i]);
          state.phase = 2;
          break;
        case 2: // PUT+signal to each actual destination PE
          for (const auto& send : plan.sendPeers) {
            auto& receiver = trial.localPlans[static_cast<std::size_t>(send.rank)];
            verify(receiver.pe == send.pe, "P5.2 bad destination PE mapping");
            for (U64 j = 0; j < send.count; ++j) {
              const std::size_t dst = static_cast<std::size_t>(send.remoteInboxOffset + j);
              const std::size_t src = static_cast<std::size_t>(send.packOffset + j);
              verify(dst < receiver.inboxElements && src < state.packed.size(),
                     "P5.2 simulated PUT outside logical packet");
              // Never overwrite an inbox slot the receiver still needs.
              verify(readStamp[static_cast<std::size_t>(send.rank)][dst] ==
                         generation - 1,
                     "P5.2 PUT overwrote inbox before reader acknowledgment");
              verify(writeStamp[static_cast<std::size_t>(send.rank)][dst] ==
                         generation - 1,
                     "P5.2 duplicated or out-of-order packet PUT");
              inbox[static_cast<std::size_t>(send.rank)][dst] = state.packed[src];
              writeStamp[static_cast<std::size_t>(send.rank)][dst] = generation;
            }
            const std::size_t sourcePe = static_cast<std::size_t>(plan.pe);
            verify(ready[static_cast<std::size_t>(send.rank)][sourcePe] ==
                       generation - 1,
                   "P5.2 ready signal updated out of order");
            ready[static_cast<std::size_t>(send.rank)][sourcePe] = generation;
          }
          state.phase = 3;
          break;
        case 3: // expect ready[sourcePE] == generation for each inbound source
          for (const auto& recv : plan.receivePeers)
            verify(ready[rank][static_cast<std::size_t>(recv.pe)] == generation,
                   "P5.2 stale/missing completion signal accepted");
          state.phase = 4;
          break;
        case 4: // all packet-to-column scatters and local copies
          state.output.assign(asSize(plan.targetLength, "model target"), nan);
          for (std::size_t slot = 0; slot < plan.unpackDestinationLids.size(); ++slot) {
            verify(writeStamp[rank][slot] == generation,
                   "P5.2 inbox read before matching PUT+signal");
            const std::size_t dst = asSize(plan.unpackDestinationLids[slot], "target LID");
            state.output[dst] = inbox[rank][slot];
            readStamp[rank][slot] = generation;
          }
          for (const auto& local : plan.local)
            state.output[asSize(local.destinationLid, "local target")] =
                changingValue(patternEpoch, static_cast<int>(rank), local.sourceLid);
          for (const auto& local : plan.local)
            verify(state.output[asSize(local.destinationLid, "local target")] ==
                       changingValue(patternEpoch, static_cast<int>(rank), local.sourceLid),
                   "P5.2 wrong local owner/source LID");
          for (const auto& remote : plan.expectedRemote)
            verify(state.output[asSize(remote.destinationLid, "remote target")] ==
                       changingValue(patternEpoch, remote.ownerRank, remote.sourceLid),
                   "P5.2 wrong remote owner/source LID");
          for (const double v : state.output)
            verify(std::isfinite(v), "P5.2 target retained unwritten sentinel");
          state.phase = 5;
          break;
        case 5: // GPU signal_op(credit + selfPE, g, sourcePE), AFTER scatter
          for (const auto& recv : plan.receivePeers) {
            const std::size_t source = static_cast<std::size_t>(recv.rank);
            const std::size_t receiverPe = static_cast<std::size_t>(plan.pe);
            verify(credit[source][receiverPe] == generation - 1,
                   "P5.2 duplicate/stale consumed ACK");
            credit[source][receiverPe] = generation;
          }
          ++state.epoch;
          state.phase = 0;
          if (state.epoch > maxEpoch) ++completed;
          break;
        default: throw std::runtime_error("P5.2 invalid model scheduling phase");
      }
      ++scheduled;
    }
    verify(scheduled == n * static_cast<std::size_t>(maxEpoch) * 6,
           "P5.2 model lost rank/phase tasks");
    for (std::size_t receiver = 0; receiver < n; ++receiver)
      for (const auto& recv : trial.localPlans[receiver].receivePeers) {
        verify(ready[receiver][static_cast<std::size_t>(recv.pe)] == maxEpoch,
               "P5.2 final ready generation missing");
        verify(credit[static_cast<std::size_t>(recv.rank)][
                   static_cast<std::size_t>(trial.localPlans[receiver].pe)] == maxEpoch,
               "P5.2 final consumed acknowledgment missing");
      }
  }
}

void finishAndCheck(Trial t) {
  const std::size_t n = t.localPlans.size();
  U64 globalSend = 0, globalReceive = 0;
  U64 sendPeers = 0, receivePeers = 0;
  for (std::size_t rank = 0; rank < n; ++rank) {
    auto& p = t.localPlans[rank];
    finalizeLocal(p, t.metadata[rank]);
    verify(p.finalized, "unfinalized plan");
    verify(p.symmetricPackElements >= 1 && p.symmetricInboxElements >= 1 &&
               p.symmetricSignalElements == n, "invalid symmetric capacities");
    verify(p.packElements == t.inputs[rank].exportLids.size(), "lost exports");
    verify(p.inboxElements == t.inputs[rank].remote.size(), "lost imports");
    globalSend += p.packElements;
    globalReceive += p.inboxElements;
    sendPeers += p.sendPeers.size();
    receivePeers += p.receivePeers.size();

    // Verify that synthetic values from the original sender/source LID land
    // at exactly the P1-designated destination, even with interleaving.
    std::vector<U64> reconstructed(static_cast<std::size_t>(p.targetLength), 0);
    std::vector<U64> expected(reconstructed.size(), 0);
    for (const auto& e : t.inputs[rank].local) {
      expected[e.destinationLid] = expectedValue(p.rank, e.sourceLid);
      reconstructed[e.destinationLid] = expected[e.destinationLid];
    }
    for (const auto& e : t.inputs[rank].remote)
      expected[e.destinationLid] = expectedValue(e.ownerRank, e.sourceLid);
    for (const auto& receive : p.receivePeers) {
      const auto& sourceLids = t.metadata[rank].sourceLidsFromRank[
          static_cast<std::size_t>(receive.rank)];
      verify(sourceLids.size() == receive.count, "wire length mismatch");
      for (U64 i = 0; i < receive.count; ++i) {
        const U64 dst = p.unpackDestinationLids[static_cast<std::size_t>(
            receive.inboxOffset + i)];
        reconstructed[dst] = expectedValue(receive.rank, sourceLids[i]);
      }
    }
    verify(reconstructed == expected, "incorrect full column-map reconstruction");
    for (const auto& s : p.sendPeers) {
      verify(s.remoteSignalIndex == static_cast<U64>(p.pe) &&
                 s.remoteInboxOffset != noOffset, "wrong remote signal/offset");
      verify(s.remoteInboxOffset + s.count <=
                 t.localPlans[static_cast<std::size_t>(s.rank)].inboxElements,
             "remote inbox out of bounds");
      // Stable per-recipient projection of the original export order.
      std::vector<U64> original;
      for (std::size_t i = 0; i < t.inputs[rank].exportLids.size(); ++i)
        if (t.inputs[rank].exportPids[i] == s.rank)
          original.push_back(t.inputs[rank].exportLids[i]);
      std::vector<U64> packed;
      for (U64 i = 0; i < s.count; ++i)
        packed.push_back(p.packedSourceLids[static_cast<std::size_t>(
            s.packOffset + i)]);
      verify(original == packed, "original per-peer export order changed");
    }
    expectError([&] { finalizeLocal(p, t.metadata[rank]); }, "double finalize");
  }
  exerciseCreditProtocol(t);
  verify(globalSend == globalReceive && sendPeers == receivePeers,
         "global message/element counts do not balance");
}

std::vector<RankInput> generate(std::mt19937_64& gen, int nRanks, bool empty,
                                bool oneWay) {
  std::vector<int> peForRank(static_cast<std::size_t>(nRanks));
  for (int i = 0; i < nRanks; ++i) peForRank[i] = i;
  std::shuffle(peForRank.begin(), peForRank.end(), gen);
  std::vector<U64> owned(static_cast<std::size_t>(nRanks), 0);
  for (int i = 0; i < nRanks; ++i)
    owned[i] = empty ? 0 : (oneWay ? 2 : gen() % 12);
  if (oneWay && nRanks >= 2) owned[0] = 0; // receives without owning
  std::vector<RankInput> inputs(static_cast<std::size_t>(nRanks));
  for (int rank = 0; rank < nRanks; ++rank) {
    auto& in = inputs[static_cast<std::size_t>(rank)];
    in.rank = rank;
    in.pe = peForRank[rank];
    in.peForRank = peForRank;
    in.ownedLengthsByRank = owned;
    in.ownedLength = owned[rank];
    std::vector<std::pair<int,U64>> entries;
    if (!empty && !oneWay) {
      for (U64 lid = 0; lid < owned[rank]; ++lid)
        if ((gen() % 3) != 0) entries.push_back({rank, lid});
    }
    if (!empty && oneWay && rank != 0) {
      for (U64 lid = 0; lid < owned[rank]; ++lid)
        entries.push_back({rank, lid});
    }
    const int requests = empty ? 0 : oneWay ? (rank == 0 ? 2 : 0) : gen() % 35;
    for (int j = 0; j < requests; ++j) {
      int owner = -1;
      if (oneWay) owner = 1;
      else {
        for (int tries = 0; tries < nRanks * 3; ++tries) {
          const int candidate = static_cast<int>(gen() % nRanks);
          if (candidate != rank && owned[candidate] != 0) {
            owner = candidate;
            break;
          }
        }
      }
      if (owner >= 0 && owned[owner] != 0) {
        const U64 src = oneWay ? 1ULL : gen() % owned[owner];
        entries.push_back({owner, src});
      }
    }
    std::shuffle(entries.begin(), entries.end(), gen);
    in.targetLength = static_cast<U64>(entries.size());
    for (U64 dst = 0; dst < in.targetLength; ++dst) {
      const auto& owner = entries[static_cast<std::size_t>(dst)];
      if (owner.first == rank)
        in.local.push_back(LocalEntry{owner.second, dst});
      else {
        in.remote.push_back(RemoteEntry{
            owner.first, peForRank[owner.first], owner.second, dst});
        auto& sender = inputs[static_cast<std::size_t>(owner.first)];
        sender.exportPids.push_back(rank);
        sender.exportLids.push_back(owner.second);
      }
    }
  }
  // Some senders are constructed before their own rank's input is filled;
  // recreate only the rank-local metadata without discarding received exports.
  for (int rank = 0; rank < nRanks; ++rank) {
    auto& in = inputs[static_cast<std::size_t>(rank)];
    in.rank = rank;
    in.pe = peForRank[rank];
    in.peForRank = peForRank;
    in.ownedLengthsByRank = owned;
    in.ownedLength = owned[rank];
    std::vector<std::pair<U64,int>> exports;
    for (std::size_t i = 0; i < in.exportLids.size(); ++i)
      exports.push_back({in.exportLids[i], in.exportPids[i]});
    std::shuffle(exports.begin(), exports.end(), gen);
    in.exportLids.clear(); in.exportPids.clear();
    for (const auto& e : exports) {
      in.exportLids.push_back(e.first);
      in.exportPids.push_back(e.second);
    }
  }
  return inputs;
}

int main() {
  std::mt19937_64 gen(0x5171c0ffeeULL);
  {
    auto t = construct(generate(gen, 1, true, false));
    finishAndCheck(t);
    verify(t.localPlans[0].packElements == 0, "empty source exports");
  }
  {
    auto t = construct(generate(gen, 5, true, false));
    finishAndCheck(t);
  }
  {
    auto t = construct(generate(gen, 2, false, true));
    finishAndCheck(t);
    verify(t.localPlans[0].packElements == 0 &&
               t.localPlans[0].inboxElements == 2 &&
               t.localPlans[1].inboxElements == 0 &&
               t.localPlans[1].packElements == 2,
           "one-way/zero-owner case not exercised");
  }
  for (int ranks : {1, 2, 3, 4, 8, 16}) {
    for (int repeat = 0; repeat < 150; ++repeat)
      finishAndCheck(construct(generate(gen, ranks, false, false)));
  }
  {
    auto t = construct(generate(gen, 2, false, true));
    auto bad = t.metadata[0];
    bad.sourceLidsFromRank[1][0] = 55555;
    expectError([&] { auto p = t.localPlans[0]; finalizeLocal(p, bad); },
                "wrong source LID with matching packet count");
    bad = t.metadata[0];
    bad.sourceLidsFromRank[1].pop_back();
    expectError([&] { auto p = t.localPlans[0]; finalizeLocal(p, bad); },
                "missing received packet slot");
    auto receiver = t.localPlans[1];
    bad = t.metadata[1];
    bad.inboxOffsetAtRank[0] = 0;
    // Rank 1 sends to rank 0, but receiver rank 0 has only two slots.
    bad.inboxLengthAtRank[0] = 1;
    expectError([&] { finalizeLocal(receiver, bad); },
                "remote inbox overrun");
  }
  {
    auto inputs = generate(gen, 2, false, true);
    auto bad = inputs[0];
    bad.peForRank[1] = bad.peForRank[0];
    expectError([&] { (void)buildLocal(bad); }, "duplicate NVSHMEM PE");
    bad = inputs[0];
    bad.remote[0].destinationLid = bad.targetLength;
    expectError([&] { (void)buildLocal(bad); }, "out-of-range target LID");
    bad = inputs[0];
    bad.remote[0].sourcePe = bad.pe;
    expectError([&] { (void)buildLocal(bad); }, "owner rank vs PE mismatch");
    bad = inputs[0];
    bad.remote[1].destinationLid = bad.remote[0].destinationLid;
    expectError([&] { (void)buildLocal(bad); }, "duplicate destination LID");
    bad = inputs[1];
    bad.exportLids[0] = bad.ownedLength;
    expectError([&] { (void)buildLocal(bad); }, "export outside owner source Map");
    auto t = construct(inputs);
    auto overflow = t.metadata[0];
    overflow.maxInboxElements = noOffset;
    expectError([&] { auto p = t.localPlans[0]; finalizeLocal(p, overflow); },
                "overflow of symmetric allocation size");
  }
  std::cout << "P5.1 plan + P5.2 asynchronous per-peer credit PUT model: PASS "
               "(900 randomized layouts, 1/2/3/4/8/16 ranks, 2 allocation cycles, "
               "4 changed-value epochs per cycle, adversarial rank scheduling, "
               "PE permutations, zero/one-way and corrupt metadata)\n";
}
