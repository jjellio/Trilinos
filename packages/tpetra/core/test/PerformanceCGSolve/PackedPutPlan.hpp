// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P5.1: pure-host sender-packed NVSHMEM PUT packet/inbox plan.
// NO MPI, Tpetra, CUDA, NVSHMEM, payload movement, or allocation in this file.
// Separate local construction from one-time peer metadata reconciliation so
// this algorithm is testable in ordinary C++17 on machines without GPUs.

#ifndef TPETRA_PERFORMANCE_PACKED_PUT_PLAN_HPP
#define TPETRA_PERFORMANCE_PACKED_PUT_PLAN_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace PackedPutP51 {

using U64 = std::uint64_t;
static constexpr U64 noOffset = std::numeric_limits<U64>::max();

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("NVSHMEM P5.1: " + message);
}

inline U64 add(U64 a, U64 b, const char* what) {
  require(b <= noOffset - a, std::string("integer overflow in ") + what);
  return a + b;
}

inline std::size_t asSize(U64 value, const char* what) {
  require(value <= static_cast<U64>(std::numeric_limits<std::size_t>::max()),
          std::string("size_t overflow in ") + what);
  return static_cast<std::size_t>(value);
}

inline std::size_t checkedBytes(U64 count, std::size_t itemSize, const char* what) {
  require(itemSize != 0 && count <= static_cast<U64>(
              std::numeric_limits<std::size_t>::max() / itemSize),
          std::string("allocation size overflow in ") + what);
  return asSize(count, what) * itemSize;
}

struct LocalEntry {
  U64 sourceLid = 0;
  U64 destinationLid = 0;
};
struct RemoteEntry {
  int ownerRank = -1;
  int sourcePe = -1;
  U64 sourceLid = 0;
  U64 destinationLid = 0;
};

struct RankInput {
  int rank = -1;
  int pe = -1;
  std::vector<int> peForRank;
  std::vector<U64> ownedLengthsByRank;
  U64 ownedLength = 0;
  U64 targetLength = 0;
  std::vector<LocalEntry> local;
  std::vector<RemoteEntry> remote;
  // Original Tpetra Importer order. These arrays are NOT sorted in place.
  std::vector<U64> exportLids;
  std::vector<int> exportPids;
};

struct SendPeer {
  int rank = -1;
  int pe = -1;
  U64 packOffset = 0;
  U64 count = 0;
  U64 remoteInboxOffset = noOffset; // filled during finalizeLocal()
  U64 remoteSignalIndex = noOffset; // our actual NVSHMEM PE
};
struct ReceivePeer {
  int rank = -1;
  int pe = -1;
  U64 inboxOffset = 0;
  U64 count = 0;
  U64 localSignalIndex = noOffset; // actual sending NVSHMEM PE
};

struct Plan {
  int rank = -1;
  int pe = -1;
  std::vector<int> peForRank;
  U64 ownedLength = 0;
  U64 targetLength = 0;
  std::vector<LocalEntry> local;
  std::vector<RemoteEntry> expectedRemote;
  std::vector<SendPeer> sendPeers;      // ascending destination PE
  std::vector<ReceivePeer> receivePeers; // ascending source PE
  std::vector<U64> packedSourceLids;   // contiguous per-peer pack order
  std::vector<U64> unpackDestinationLids; // index = local symmetric inbox slot
  U64 packElements = 0;
  U64 inboxElements = 0;
  U64 symmetricPackElements = 0;       // collective padded capacity >= 1
  U64 symmetricInboxElements = 0;      // collective padded capacity >= 1
  U64 symmetricSignalElements = 0;     // one per actual PE
  bool finalized = false;
};

// Called on every rank before metadata exchange. Source and destination LIDs
// are arbitrary; only sender per-peer order is stable in its original sequence.
inline Plan buildLocal(const RankInput& in) {
  const int nRanks = static_cast<int>(in.peForRank.size());
  require(nRanks > 0 && in.rank >= 0 && in.rank < nRanks &&
              in.pe >= 0 && in.pe < nRanks,
          "invalid local rank/PE or rank-to-PE table");
  require(static_cast<int>(in.ownedLengthsByRank.size()) == nRanks,
          "owned lengths do not match communicator size");
  std::vector<int> rankForPe(static_cast<std::size_t>(nRanks), -1);
  for (int rank = 0; rank < nRanks; ++rank) {
    const int pe = in.peForRank[static_cast<std::size_t>(rank)];
    require(pe >= 0 && pe < nRanks &&
                rankForPe[static_cast<std::size_t>(pe)] == -1,
            "rank-to-PE table is not a permutation");
    rankForPe[static_cast<std::size_t>(pe)] = rank;
  }
  require(in.peForRank[static_cast<std::size_t>(in.rank)] == in.pe,
          "local rank-to-PE mapping is inconsistent");
  require(in.ownedLengthsByRank[static_cast<std::size_t>(in.rank)] == in.ownedLength,
          "local source length disagrees with owned-length table");
  require(in.exportLids.size() == in.exportPids.size(),
          "export source LIDs and PIDs have different lengths");
  (void)checkedBytes(in.ownedLength, sizeof(double), "source length");
  (void)checkedBytes(in.targetLength, sizeof(double), "target length");

  Plan p;
  p.rank = in.rank;
  p.pe = in.pe;
  p.peForRank = in.peForRank;
  p.ownedLength = in.ownedLength;
  p.targetLength = in.targetLength;
  p.local = in.local;
  p.expectedRemote = in.remote;

  require(add(static_cast<U64>(in.local.size()),
              static_cast<U64>(in.remote.size()), "destination partition") ==
              in.targetLength,
          "local and remote destination counts do not cover target Map");
  std::vector<unsigned char> covered(asSize(in.targetLength, "coverage"), 0);
  for (const auto& e : in.local) {
    require(e.sourceLid < in.ownedLength && e.destinationLid < in.targetLength,
            "local source/destination LID out of range");
    auto& slot = covered[asSize(e.destinationLid, "local destination")];
    require(slot == 0, "duplicate local destination LID");
    slot = 1;
  }
  std::map<int, U64> incomingCounts; // key = actual NVSHMEM source PE
  for (const auto& e : in.remote) {
    require(e.ownerRank >= 0 && e.ownerRank < nRanks &&
                e.ownerRank != in.rank &&
                e.sourcePe == in.peForRank[static_cast<std::size_t>(e.ownerRank)],
            "invalid remote owner / source PE mapping");
    require(e.sourceLid < in.ownedLengthsByRank[static_cast<std::size_t>(e.ownerRank)] &&
                e.destinationLid < in.targetLength,
            "remote source/destination LID out of range");
    auto& slot = covered[asSize(e.destinationLid, "remote destination")];
    require(slot == 0, "duplicate remote destination LID");
    slot = 1;
    incomingCounts[e.sourcePe] = add(incomingCounts[e.sourcePe], 1,
                                     "incoming peer element count");
  }
  for (unsigned char value : covered)
    require(value == 1, "destination partition has a hole");

  std::map<int, std::vector<U64>> outgoing; // stable subsequence per destination PE
  for (std::size_t i = 0; i < in.exportLids.size(); ++i) {
    const int destinationRank = in.exportPids[i];
    require(destinationRank >= 0 && destinationRank < nRanks &&
                destinationRank != in.rank,
            "invalid export destination rank");
    const U64 src = in.exportLids[i];
    require(src < in.ownedLength, "export source LID outside domain Map");
    outgoing[in.peForRank[static_cast<std::size_t>(destinationRank)]].push_back(src);
  }
  for (const auto& kv : outgoing) {
    require(!kv.second.empty(), "empty export peer unexpectedly created");
    const int destinationPe = kv.first;
    const U64 count = static_cast<U64>(kv.second.size());
    p.sendPeers.push_back(SendPeer{rankForPe[static_cast<std::size_t>(destinationPe)],
                                   destinationPe, p.packElements, count,
                                   noOffset, static_cast<U64>(in.pe)});
    p.packElements = add(p.packElements, count, "sender packet offset");
    p.packedSourceLids.insert(p.packedSourceLids.end(),
                              kv.second.begin(), kv.second.end());
  }
  for (const auto& kv : incomingCounts) {
    require(kv.second > 0, "empty receive peer unexpectedly created");
    const int sourcePe = kv.first;
    p.receivePeers.push_back(ReceivePeer{
        rankForPe[static_cast<std::size_t>(sourcePe)], sourcePe,
        p.inboxElements, kv.second, static_cast<U64>(sourcePe)});
    p.inboxElements = add(p.inboxElements, kv.second, "receiver inbox offset");
  }
  require(p.packElements == static_cast<U64>(p.packedSourceLids.size()),
          "packed source count mismatch");
  require(p.inboxElements == static_cast<U64>(p.expectedRemote.size()),
          "incoming packet count mismatch");
  (void)checkedBytes(p.packElements, sizeof(double), "pack data");
  (void)checkedBytes(p.inboxElements, sizeof(double), "inbox data");
  p.unpackDestinationLids.assign(asSize(p.inboxElements, "unpack map"), noOffset);
  return p;
}

// Data communicated only ONCE by the MPI host-side adapter: the exact
// export-source LID sequence for each sender, and the destination's inbox
// offset/capacity. No vector values cross MPI in this protocol.
struct PeerMetadata {
  std::vector<std::vector<U64>> sourceLidsFromRank;
  std::vector<U64> inboxOffsetAtRank;    // our segment in each receiving rank
  std::vector<U64> inboxLengthAtRank;    // logical receive size at each rank
  U64 maxPackElements = 0;
  U64 maxInboxElements = 0;
};

inline void finalizeLocal(Plan& p, const PeerMetadata& m) {
  require(!p.finalized, "plan finalized twice");
  const auto nRanks = p.peForRank.size();
  require(m.sourceLidsFromRank.size() == nRanks &&
              m.inboxOffsetAtRank.size() == nRanks &&
              m.inboxLengthAtRank.size() == nRanks,
          "peer metadata dimensions differ from communicator");
  require(m.maxPackElements >= p.packElements &&
              m.maxInboxElements >= p.inboxElements,
          "collective symmetric capacity is too small");
  p.symmetricPackElements = std::max<U64>(1, m.maxPackElements);
  p.symmetricInboxElements = std::max<U64>(1, m.maxInboxElements);
  p.symmetricSignalElements = static_cast<U64>(nRanks);
  (void)checkedBytes(p.symmetricPackElements, sizeof(double), "symmetric pack");
  (void)checkedBytes(p.symmetricInboxElements, sizeof(double), "symmetric inbox");
  (void)checkedBytes(p.symmetricSignalElements, sizeof(U64), "symmetric signals");

  std::vector<unsigned char> sentToRank(nRanks, 0);
  for (auto& peer : p.sendPeers) {
    const std::size_t rank = static_cast<std::size_t>(peer.rank);
    const U64 offset = m.inboxOffsetAtRank[rank];
    require(offset != noOffset && peer.count > 0,
            "missing receiver inbox offset for nonempty send peer");
    const U64 finish = add(offset, peer.count, "remote inbox range");
    require(finish <= m.inboxLengthAtRank[rank] &&
                finish <= p.symmetricInboxElements,
            "remote PUT would address inbox padding/outside remote segment");
    peer.remoteInboxOffset = offset;
    sentToRank[rank] = 1;
  }
  for (std::size_t rank = 0; rank < nRanks; ++rank) {
    if (!sentToRank[rank])
      require(m.inboxOffsetAtRank[rank] == noOffset,
              "unexpected receiver inbox offset for empty sender relationship");
  }

  std::vector<unsigned char> receivedFromRank(nRanks, 0);
  for (const auto& peer : p.receivePeers) {
    const std::size_t rank = static_cast<std::size_t>(peer.rank);
    receivedFromRank[rank] = 1;
    const auto& wire = m.sourceLidsFromRank[rank];
    require(static_cast<U64>(wire.size()) == peer.count,
            "sender/receiver element count mismatch for rank " +
                std::to_string(peer.rank));

    // The destination's P1 lookup establishes the authoritative mapping
    // (sending rank, owner-local source LID) -> actual target column LIDs.
    // A FIFO of destinations for each source LID also handles repeated
    // requests for the same source LID without reordering exports.
    std::map<U64, std::vector<U64>> expectedBySource;
    for (const auto& e : p.expectedRemote)
      if (e.ownerRank == peer.rank)
        expectedBySource[e.sourceLid].push_back(e.destinationLid);
    std::map<U64, std::size_t> consumed;
    for (U64 i = 0; i < peer.count; ++i) {
      const U64 lid = wire[asSize(i, "wire source index")];
      const auto it = expectedBySource.find(lid);
      require(it != expectedBySource.end(),
              "sender exported a source LID not requested by receiver");
      auto& index = consumed[lid];
      require(index < it->second.size(),
              "sender exported a source LID too many times to receiver");
      const U64 packetSlot = add(peer.inboxOffset, i, "inbox packet slot");
      require(packetSlot < p.inboxElements,
              "receiver packet slot outside logical inbox");
      auto& destination = p.unpackDestinationLids[asSize(packetSlot, "unpack slot")];
      require(destination == noOffset, "duplicate inbox slot assignment");
      destination = it->second[index++];
    }
    for (const auto& kv : expectedBySource) {
      require(consumed[kv.first] == kv.second.size(),
              "sender did not export every requested source-LID occurrence");
    }
  }
  for (std::size_t rank = 0; rank < nRanks; ++rank)
    if (!receivedFromRank[rank])
      require(m.sourceLidsFromRank[rank].empty(),
              "unexpected sender packet for non-receiving rank");

  std::vector<unsigned char> visited(asSize(p.targetLength, "coverage"), 0);
  for (const auto& e : p.local) {
    require(e.destinationLid < p.targetLength &&
                !visited[asSize(e.destinationLid, "local target")],
            "duplicate or invalid local target during finalization");
    visited[asSize(e.destinationLid, "local target")] = 1;
  }
  for (U64 dst : p.unpackDestinationLids) {
    require(dst != noOffset && dst < p.targetLength &&
                !visited[asSize(dst, "remote target")],
            "missing, duplicate or invalid remote destination packet mapping");
    visited[asSize(dst, "remote target")] = 1;
  }
  for (unsigned char v : visited)
    require(v == 1, "unpack/local destination partition is incomplete");
  p.finalized = true;
}

} // namespace PackedPutP51

#endif // TPETRA_PERFORMANCE_PACKED_PUT_PLAN_HPP
