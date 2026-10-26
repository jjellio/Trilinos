// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
// P6.2 host-only descriptor construction. No CUDA/NVSHMEM/Tpetra/MPI here.
// Reorders a FINALIZED P5.1 plan once by (peer, actual global ID), then
// partitions each ordered peer packet into independent signal-bearing PUTs.
#ifndef TPETRA_PERFORMANCE_PACKED_PUT_P62_PLAN_HPP
#define TPETRA_PERFORMANCE_PACKED_PUT_P62_PLAN_HPP

#include "PackedPutPlan.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace PackedPutP62 {
using U64 = PackedPutP51::U64;
using PackedPutP51::require;
using PackedPutP51::add;
using PackedPutP51::asSize;
using PackedPutP51::checkedBytes;

struct SendDescriptor {
  int pe = -1;
  U64 sourceOffset = 0; // offset in the GID-ordered sourceLids[]
  U64 packedOffset = 0; // source symmetric scratch offset
  U64 remoteInboxOffset = 0;
  U64 signalIndex = PackedPutP51::noOffset; // offset in destination's ready[]
  std::uint32_t count = 0;
};
struct ReceiveDescriptor {
  int sourcePe = -1;
  U64 inboxOffset = 0;
  U64 destinationOffset = 0; // offset in GID-ordered destinationLids[]
  U64 signalIndex = 0;      // offset in MY ready[]
  std::uint32_t count = 0;
};
struct Plan {
  std::vector<SendDescriptor> sends;
  std::vector<ReceiveDescriptor> receives;
  std::vector<U64> sourceLids;      // same number of entries as P5.1 pack
  std::vector<U64> destinationLids; // same number of entries as P5.1 inbox
  std::vector<U64> sendChunksForRank;
  std::vector<U64> receiveChunksFromRank;
  std::vector<U64> readyBaseForSourceRank; // advertised to sender by MPI setup
  std::vector<U64> sendGidDigestForRank;
  std::vector<U64> recvGidDigestFromRank;
  U64 chunkDoubles = 0;
  U64 localReadyElements = 0;
  U64 symmetricReadyElements = 0;
  bool finalized = false;
};

// Sorted-GID metadata fingerprint: a SETUP-time check, not a correctness
// substitute for the authoritative P5.1 reconciliation or runtime tests.
// All PEs use the same GO representation and compiler in this experiment.
inline U64 digestStep(U64 digest, U64 value) {
  return (digest ^ value) * UINT64_C(1099511628211);
}
template<class GO>
inline U64 fingerprint(const std::vector<GO>& sorted) {
  U64 hash = UINT64_C(14695981039346656037);
  hash = digestStep(hash, static_cast<U64>(sorted.size()));
  for (GO gid : sorted)
    hash = digestStep(hash, static_cast<U64>(gid));
  return hash;
}

// Caller supplies the exact domain/column Map GID for each corresponding
// owner-local source / receiver-local destination LID, respectively.
// Sorting equal GIDs is stable; their scalar values are identical for a
// unique owner (possibly requested repeatedly at different target LIDs).
template<class GO>
Plan buildLocal(const PackedPutP51::Plan& old,
                const std::vector<GO>& ownedGids,
                const std::vector<GO>& targetGids,
                U64 chunkDoubles) {
  static_assert(std::is_integral<GO>::value && sizeof(GO) <= sizeof(U64),
                "P6.2 GIDs must be <=64-bit integers");
  require(old.finalized, "P6.2 requires finalized P5.1 plan");
  require(chunkDoubles > 0 && chunkDoubles <=
          static_cast<U64>(std::numeric_limits<std::uint32_t>::max()),
          "P6.2 chunk size must fit uint32_t and be positive");
  require(ownedGids.size() == asSize(old.ownedLength, "P6.2 owned GIDs") &&
          targetGids.size() == asSize(old.targetLength, "P6.2 target GIDs"),
          "P6.2 Map GID sizes differ from original plan");
  const std::size_t nRanks = old.peForRank.size();
  Plan out;
  out.chunkDoubles = chunkDoubles;
  out.sendChunksForRank.assign(nRanks, 0);
  out.receiveChunksFromRank.assign(nRanks, 0);
  out.readyBaseForSourceRank.assign(nRanks, PackedPutP51::noOffset);
  out.sendGidDigestForRank.assign(nRanks, 0);
  out.recvGidDigestFromRank.assign(nRanks, 0);
  out.sourceLids.resize(asSize(old.packElements, "P6.2 source count"));
  out.destinationLids.resize(asSize(old.inboxElements, "P6.2 destination count"));
  for (const auto& peer : old.sendPeers) {
    require(peer.rank >= 0 && static_cast<std::size_t>(peer.rank) < nRanks &&
            peer.count > 0 && peer.packOffset <= old.packElements &&
            peer.count <= old.packElements - peer.packOffset &&
            peer.remoteInboxOffset != PackedPutP51::noOffset,
            "P6.2 invalid outgoing P5.1 peer");
    struct Element { GO gid; U64 sourceLid; };
    std::vector<Element> elements;
    elements.reserve(asSize(peer.count, "sender peer count"));
    for (U64 i = 0; i < peer.count; ++i) {
      const U64 lid = old.packedSourceLids[asSize(peer.packOffset + i, "sender LID")];
      require(lid < old.ownedLength, "P6.2 sender LID out of range");
      elements.push_back({ownedGids[asSize(lid, "sender GID")], lid});
    }
    std::stable_sort(elements.begin(), elements.end(),
        [](const Element& a, const Element& b) { return a.gid < b.gid; });
    std::vector<GO> sortedGids;
    sortedGids.reserve(elements.size());
    for (U64 i = 0; i < peer.count; ++i) {
      out.sourceLids[asSize(peer.packOffset + i, "ordered sender LID")] =
          elements[asSize(i, "sender element")].sourceLid;
      sortedGids.push_back(elements[asSize(i, "ordered GID")].gid);
    }
    out.sendGidDigestForRank[static_cast<std::size_t>(peer.rank)] = fingerprint(sortedGids);
    U64 sequence = 0;
    for (U64 offset = 0; offset < peer.count;) {
      const U64 count = std::min(chunkDoubles, peer.count - offset);
      const U64 oldStart = add(peer.packOffset, offset, "P6.2 sender descriptor");
      const U64 remoteStart = add(peer.remoteInboxOffset, offset, "P6.2 remote range");
      require(remoteStart <= old.symmetricInboxElements &&
              count <= old.symmetricInboxElements - remoteStart,
              "P6.2 sender PUT would exceed remote symmetric inbox");
      out.sends.push_back({peer.pe, oldStart, oldStart, remoteStart,
                           PackedPutP51::noOffset, static_cast<std::uint32_t>(count)});
      offset = add(offset, count, "sender partition advance");
      ++sequence;
    }
    out.sendChunksForRank[static_cast<std::size_t>(peer.rank)] = sequence;
  }
  for (const auto& peer : old.receivePeers) {
    require(peer.rank >= 0 && static_cast<std::size_t>(peer.rank) < nRanks &&
            peer.count > 0 && peer.inboxOffset <= old.inboxElements &&
            peer.count <= old.inboxElements - peer.inboxOffset,
            "P6.2 invalid incoming P5.1 peer");
    struct Element { GO gid; U64 destinationLid; };
    std::vector<Element> elements;
    elements.reserve(asSize(peer.count, "receiver peer count"));
    for (U64 i = 0; i < peer.count; ++i) {
      const U64 lid = old.unpackDestinationLids[asSize(peer.inboxOffset + i, "target LID")];
      require(lid < old.targetLength, "P6.2 target LID out of range");
      elements.push_back({targetGids[asSize(lid, "target GID")], lid});
    }
    std::stable_sort(elements.begin(), elements.end(),
        [](const Element& a, const Element& b) { return a.gid < b.gid; });
    std::vector<GO> sortedGids;
    sortedGids.reserve(elements.size());
    for (U64 i = 0; i < peer.count; ++i) {
      out.destinationLids[asSize(peer.inboxOffset + i, "ordered dest LID")] =
          elements[asSize(i, "receiver element")].destinationLid;
      sortedGids.push_back(elements[asSize(i, "receiver GID")].gid);
    }
    out.recvGidDigestFromRank[static_cast<std::size_t>(peer.rank)] = fingerprint(sortedGids);
    out.readyBaseForSourceRank[static_cast<std::size_t>(peer.rank)] = out.localReadyElements;
    U64 sequence = 0;
    for (U64 offset = 0; offset < peer.count;) {
      const U64 count = std::min(chunkDoubles, peer.count - offset);
      const U64 inboxStart = add(peer.inboxOffset, offset, "P6.2 inbox descriptor");
      out.receives.push_back({peer.pe, inboxStart, inboxStart,
                              add(out.localReadyElements, sequence, "receive slot"),
                              static_cast<std::uint32_t>(count)});
      offset = add(offset, count, "receiver partition advance");
      ++sequence;
    }
    out.localReadyElements = add(out.localReadyElements, sequence, "P6.2 ready count");
    out.receiveChunksFromRank[static_cast<std::size_t>(peer.rank)] = sequence;
  }
  U64 outgoingChunks = 0;
  for (U64 n : out.sendChunksForRank)
    outgoingChunks = add(outgoingChunks, n, "P6.2 outgoing chunk count");
  require(out.sends.size() == asSize(outgoingChunks, "P6.2 outgoing descriptors"),
          "P6.2 outgoing chunk count mismatch");
  require(out.receives.size() == asSize(out.localReadyElements, "P6.2 ready count"),
          "P6.2 incoming chunk count mismatch");
  return out;
}

// remoteBaseForDestinationRank is obtained by MPI_Alltoall of the receiver's
// readyBaseForSourceRank. No MPI or peer handshake in a P6.2 Import.
inline void finalize(Plan& out, const PackedPutP51::Plan& old,
                     const std::vector<U64>& remoteBaseForDestinationRank,
                     U64 maximumReceiveDescriptors) {
  require(!out.finalized && old.finalized, "P6.2 finalized twice or invalid P5.1 plan");
  require(remoteBaseForDestinationRank.size() == old.peForRank.size() &&
          maximumReceiveDescriptors >= out.localReadyElements,
          "P6.2 signal slot metadata size/capacity invalid");
  out.symmetricReadyElements = std::max<U64>(1, maximumReceiveDescriptors);
  (void)checkedBytes(out.symmetricReadyElements, sizeof(U64), "P6.2 ready memory");
  std::vector<U64> nextChunk(old.peForRank.size(), 0);
  std::vector<int> rankForPe(old.peForRank.size(), -1);
  for (std::size_t r = 0; r < rankForPe.size(); ++r)
    rankForPe[static_cast<std::size_t>(old.peForRank[r])] = static_cast<int>(r);
  for (auto& transfer : out.sends) {
    require(transfer.pe >= 0 && static_cast<std::size_t>(transfer.pe) < rankForPe.size(),
            "P6.2 send to invalid PE");
    const int rank = rankForPe[static_cast<std::size_t>(transfer.pe)];
    const U64 base = remoteBaseForDestinationRank[static_cast<std::size_t>(rank)];
    require(base != PackedPutP51::noOffset, "P6.2 missing advertised receiver signal base");
    const U64 index = add(base, nextChunk[static_cast<std::size_t>(rank)]++,
                          "P6.2 advertised signal offset");
    require(index < out.symmetricReadyElements, "P6.2 remote signal out of bounds");
    transfer.signalIndex = index;
  }
  for (std::size_t rank = 0; rank < nextChunk.size(); ++rank) {
    require(nextChunk[rank] == out.sendChunksForRank[rank],
            "P6.2 sender signal indexing differs from chunk counts");
    if (nextChunk[rank] == 0)
      require(remoteBaseForDestinationRank[rank] == PackedPutP51::noOffset,
              "P6.2 signal base advertised for nonexistent peer");
  }
  out.finalized = true;
}
} // namespace PackedPutP62
#endif
