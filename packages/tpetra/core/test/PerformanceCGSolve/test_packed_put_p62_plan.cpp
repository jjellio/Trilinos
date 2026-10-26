// CPU-only P6.2 descriptor model. Compile with g++ -std=c++17 -O2.
#include "PackedPutP62Plan.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using U64 = PackedPutP51::U64;
using Input = PackedPutP51::RankInput;
using Old = PackedPutP51::Plan;
using Plan = PackedPutP62::Plan;
using PackedPutP51::LocalEntry;
using PackedPutP51::RemoteEntry;

static void check(bool ok, const char* reason) {
  if (!ok) throw std::runtime_error(reason);
}

static std::array<Input, 3> inputs() {
  std::array<Input, 3> in;
  const std::vector<int> peForRank = {2, 0, 1}; // deliberately nonidentity
  const std::vector<U64> lengths = {4, 3, 4};
  const std::vector<U64> targetLengths = {6, 5, 3};
  for (int r = 0; r < 3; ++r) {
    in[r].rank = r;
    in[r].pe = peForRank[static_cast<std::size_t>(r)];
    in[r].peForRank = peForRank;
    in[r].ownedLengthsByRank = lengths;
    in[r].ownedLength = lengths[static_cast<std::size_t>(r)];
    in[r].targetLength = targetLengths[static_cast<std::size_t>(r)];
    in[r].local.push_back({0, 0});
  }
  // Rank 0: remote arrivals from ranks 1 and 2, repeated GID 325.
  in[0].remote = {{2, 1, 3, 1}, {1, 0, 1, 2}, {2, 1, 2, 3},
                  {1, 0, 2, 4}, {2, 1, 3, 5}};
  in[0].exportLids = {2, 0, 1, 3, 2};
  in[0].exportPids = {1, 2, 1, 2, 1};
  // Rank 1: repeated GID 50 from rank 0, one packet from rank 2.
  in[1].remote = {{0, 2, 2, 1}, {2, 1, 1, 2},
                  {0, 2, 1, 3}, {0, 2, 2, 4}};
  in[1].exportLids = {2, 1};
  in[1].exportPids = {0, 0};
  // Rank 2: one-way sends to rank 1 and rank 0; rank 1 sends none to rank 2.
  in[2].remote = {{0, 2, 0, 1}, {0, 2, 3, 2}};
  in[2].exportLids = {3, 1, 2, 3};
  in[2].exportPids = {0, 1, 0, 0};
  return in;
}

static std::array<Old, 3> makeOld() {
  const auto in = inputs();
  std::array<Old, 3> plans;
  U64 maxPack = 0, maxInbox = 0;
  for (int r = 0; r < 3; ++r) {
    plans[r] = PackedPutP51::buildLocal(in[r]);
    maxPack = std::max(maxPack, plans[r].packElements);
    maxInbox = std::max(maxInbox, plans[r].inboxElements);
  }
  for (int r = 0; r < 3; ++r) {
    PackedPutP51::PeerMetadata metadata;
    metadata.sourceLidsFromRank.resize(3);
    metadata.inboxOffsetAtRank.assign(3, PackedPutP51::noOffset);
    metadata.inboxLengthAtRank.resize(3);
    metadata.maxPackElements = maxPack;
    metadata.maxInboxElements = maxInbox;
    for (int s = 0; s < 3; ++s) {
      metadata.inboxLengthAtRank[s] = plans[s].inboxElements;
      // Our outgoing send plan needs the receiver's offset for our source PE.
      for (const auto& recv : plans[s].receivePeers)
        if (recv.rank == r)
          metadata.inboxOffsetAtRank[s] = recv.inboxOffset;
      // Our incoming receive plan needs the sender's *original* source LIDs.
      for (const auto& send : plans[s].sendPeers)
        if (send.rank == r)
          for (U64 i = 0; i < send.count; ++i)
            metadata.sourceLidsFromRank[s].push_back(
                plans[s].packedSourceLids[static_cast<std::size_t>(send.packOffset + i)]);
    }
    PackedPutP51::finalizeLocal(plans[r], metadata);
  }
  return plans;
}

static void verify(U64 chunk) {
  const auto olds = makeOld();
  const std::array<std::vector<std::int64_t>, 3> domains = {
      std::vector<std::int64_t>{100, 10, 50, 5},
      std::vector<std::int64_t>{200, 150, 175},
      std::vector<std::int64_t>{300, 275, 250, 325}};
  const std::array<std::vector<std::int64_t>, 3> targets = {
      std::vector<std::int64_t>{100, 325, 150, 250, 175, 325},
      std::vector<std::int64_t>{200, 50, 275, 10, 50},
      std::vector<std::int64_t>{300, 100, 5}};
  std::array<Plan, 3> plans;
  U64 maxReady = 0;
  for (int r = 0; r < 3; ++r) {
    plans[r] = PackedPutP62::buildLocal(olds[r], domains[r], targets[r], chunk);
    maxReady = std::max(maxReady, plans[r].localReadyElements);
    // GID order is monotonically increasing within each packet. The first
    // peer might have no traffic, including rank 2's missing incoming rank 1.
    for (const auto& recv : olds[r].receivePeers) {
      for (U64 j = 1; j < recv.count; ++j) {
        const U64 prev = plans[r].destinationLids[static_cast<std::size_t>(recv.inboxOffset + j - 1)];
        const U64 next = plans[r].destinationLids[static_cast<std::size_t>(recv.inboxOffset + j)];
        check(targets[r][static_cast<std::size_t>(prev)] <=
              targets[r][static_cast<std::size_t>(next)], "receiver GID order incorrect");
      }
    }
  }
  for (int r = 0; r < 3; ++r) {
    std::vector<U64> remoteBases(3, PackedPutP51::noOffset);
    for (int destination = 0; destination < 3; ++destination) {
      remoteBases[destination] = plans[destination].readyBaseForSourceRank[r];
      check(plans[r].sendChunksForRank[destination] ==
            plans[destination].receiveChunksFromRank[r],
            "sender/receiver chunk counts differ");
      check(plans[r].sendGidDigestForRank[destination] ==
            plans[destination].recvGidDigestFromRank[r],
            "sorted sender/receiver GID digest differs");
    }
    PackedPutP62::finalize(plans[r], olds[r], remoteBases, maxReady);
    check(plans[r].finalized, "plan not finalized");
    std::set<U64> uniqueSlots;
    for (const auto& d : plans[r].receives)
      check(uniqueSlots.insert(d.signalIndex).second, "duplicate receiver signal slot");
    check(uniqueSlots.size() == plans[r].localReadyElements,
          "wrong number of distinct receive signal slots");
  }
  // Exact mapping: every sender lane writes the GID expected at the
  // receiving column LID, even with duplicated/unsorted requests.
  for (int s = 0; s < 3; ++s) {
    for (const auto& d : plans[s].sends) {
      const auto& peToRank = olds[s].peForRank;
      int destination = -1;
      for (int r = 0; r < 3; ++r)
        if (peToRank[r] == d.pe) destination = r;
      check(destination >= 0, "bad PE mapping");
      const auto& rd = plans[destination];
      std::size_t matched = 0;
      for (const auto& incoming : rd.receives) {
        if (incoming.signalIndex != d.signalIndex ||
            incoming.sourcePe != olds[s].pe) continue;
        ++matched;
        check(incoming.inboxOffset == d.remoteInboxOffset &&
              incoming.count == d.count, "remote inbox offset/count differs");
        for (U64 j = 0; j < d.count; ++j) {
          const U64 sourceLid = plans[s].sourceLids[static_cast<std::size_t>(d.sourceOffset + j)];
          const U64 destLid = rd.destinationLids[static_cast<std::size_t>(incoming.destinationOffset + j)];
          check(domains[s][static_cast<std::size_t>(sourceLid)] ==
                targets[destination][static_cast<std::size_t>(destLid)],
                "descriptor would deliver wrong GID/value");
        }
      }
      check(matched == 1, "each PUT must match exactly one receiver signal");
    }
  }
  // Mutation must be detected by the MPI adapter's advertised GID digest.
  auto wrongTarget = targets[1];
  wrongTarget[1] = 9999;
  const auto wrong = PackedPutP62::buildLocal(olds[1], domains[1], wrongTarget, chunk);
  check(plans[0].sendGidDigestForRank[1] != wrong.recvGidDigestFromRank[0],
        "GID mismatch fingerprint failure");
  std::cout << "[NVSHMEM P6.2 PLAN] chunk=" << chunk << " ranks=3"
            << " rankPermutation=YES duplicateGids=YES oneWay=YES status=PASS\n";
}
static void verifyEmptyAndLocalOnly() {
  for (int localCount : {0, 3}) {
    Input input;
    input.rank = 0;
    input.pe = 0;
    input.peForRank = {0};
    input.ownedLengthsByRank = {static_cast<U64>(localCount)};
    input.ownedLength = static_cast<U64>(localCount);
    input.targetLength = static_cast<U64>(localCount);
    for (int i = 0; i < localCount; ++i)
      input.local.push_back({static_cast<U64>(i), static_cast<U64>(i)});
    auto old = PackedPutP51::buildLocal(input);
    PackedPutP51::PeerMetadata m;
    m.sourceLidsFromRank.resize(1);
    m.inboxOffsetAtRank = {PackedPutP51::noOffset};
    m.inboxLengthAtRank = {0};
    PackedPutP51::finalizeLocal(old, m);
    std::vector<std::int64_t> gids;
    for (int i = 0; i < localCount; ++i) gids.push_back(5 + i * 7);
    auto plan = PackedPutP62::buildLocal(old, gids, gids, 2);
    check(plan.sends.empty() && plan.receives.empty() &&
          plan.localReadyElements == 0, "empty/local-only plan created remote work");
    PackedPutP62::finalize(plan, old, {PackedPutP51::noOffset}, 0);
    check(plan.finalized && plan.symmetricReadyElements == 1,
          "empty plan should retain one symmetric ready padding slot");
  }
  std::cout << "[NVSHMEM P6.2 PLAN] emptyLocalOnly=YES status=PASS\n";
}

int main() {
  try {
    verifyEmptyAndLocalOnly();
    verify(1);
    verify(2);
    verify(3);
    verify(8);
    const auto old = makeOld();
    bool bad = false;
    try { (void)PackedPutP62::buildLocal(old[0], std::vector<std::int64_t>{100,10,50,5},
                                          std::vector<std::int64_t>{100,325,150,250,175,325}, 0); }
    catch (const std::runtime_error&) { bad = true; }
    check(bad, "zero chunk must be rejected");
    std::cout << "[NVSHMEM P6.2 PLAN] RESULT=PASS\n";
  } catch (const std::exception& e) {
    std::cerr << "[NVSHMEM P6.2 PLAN] RESULT=FAIL: " << e.what() << '\n';
    return 1;
  }
}
