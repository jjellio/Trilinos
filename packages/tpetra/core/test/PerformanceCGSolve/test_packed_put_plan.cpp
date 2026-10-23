// P5.1 deterministic CPU-only plan tests: no MPI, Trilinos, CUDA or NVSHMEM.
#include "PackedPutPlan.hpp"

#include <algorithm>
#include <cstdint>
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
  std::cout << "P5.1 packed PUT plan: PASS (all-empty, one-way, zero-owner, "
               "PE permutation, randomized 1/2/3/4/8/16 ranks, corrupt metadata)\n";
}
