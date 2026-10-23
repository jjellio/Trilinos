// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P5.1 one-time metadata adapter. MPI exchanges ONLY source LIDs, counts,
// inbox offsets and capacities, never vector payload or NVSHMEM allocations.

#ifndef TPETRA_PERFORMANCE_PACKED_PUT_PLAN_MPI_HPP
#define TPETRA_PERFORMANCE_PACKED_PUT_PLAN_MPI_HPP

#include "PackedPutPlan.hpp"
#include "ImportGatherPlan.hpp"
#include "NvshmemP0.hpp"

#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace PackedPutP51 {

inline void mpiCheck(int err, const char* operation) {
  require(err == MPI_SUCCESS, std::string("MPI error in ") + operation);
}

inline int asMpiCount(U64 count, const char* what) {
  require(count <= static_cast<U64>(std::numeric_limits<int>::max()),
          std::string("MPI_Alltoallv int count/displacement overflow: ") + what);
  return static_cast<int>(count);
}

template<class LO, class GO, class Node>
RankInput fromP1(const ImportGatherP1::ImportGatherPlan<LO, GO, Node>& p1,
                 const NvshmemP0::Session& session) {
  RankInput in;
  in.rank = session.mpiRank();
  in.pe = session.pe();
  const int nRanks = session.numPes();
  in.peForRank.assign(static_cast<std::size_t>(nRanks), -1);
  for (int pe = 0; pe < nRanks; ++pe)
    in.peForRank[static_cast<std::size_t>(session.mpiRankForPe(pe))] = pe;
  in.ownedLengthsByRank.assign(p1.sourceLengthByRank.begin(),
                                p1.sourceLengthByRank.end());
  in.ownedLength = static_cast<U64>(p1.sourceLength);
  in.targetLength = static_cast<U64>(p1.targetLength);
  in.local.reserve(p1.local.size());
  for (const auto& e : p1.local)
    in.local.push_back(LocalEntry{e.sourceLid, e.destinationLid});
  in.remote.reserve(p1.remote.size());
  for (const auto& e : p1.remote)
    in.remote.push_back(RemoteEntry{e.ownerRank, e.sourcePe,
                                    e.sourceLid, e.destinationLid});
  in.exportLids.reserve(p1.exportLids.size());
  for (const LO lid : p1.exportLids) {
    in.exportLids.push_back(static_cast<U64>(
        ImportGatherP1::checkedLid(lid, p1.sourceLength,
                                   "P5.1 export source LID")));
  }
  in.exportPids = p1.exportPids;
  return in;
}

// The only collective transfers performed here are setup metadata. In
// particular, MPI_Alltoallv does NOT transport double vector payloads.
inline Plan reconcileMpi(RankInput input, MPI_Comm comm) {
  int mpiRank = -1, nRanks = -1;
  mpiCheck(MPI_Comm_rank(comm, &mpiRank), "MPI_Comm_rank");
  mpiCheck(MPI_Comm_size(comm, &nRanks), "MPI_Comm_size");
  require(input.rank == mpiRank &&
              static_cast<int>(input.peForRank.size()) == nRanks,
          "MPI communicator differs from local P5.1 input");

  Plan plan = buildLocal(input);
  const auto n = static_cast<std::size_t>(nRanks);
  std::vector<U64> sendCounts(n, 0), recvCounts(n, 0);
  for (const auto& peer : plan.sendPeers)
    sendCounts[static_cast<std::size_t>(peer.rank)] = peer.count;
  mpiCheck(MPI_Alltoall(sendCounts.data(), 1, MPI_UINT64_T,
                        recvCounts.data(), 1, MPI_UINT64_T, comm),
           "MPI_Alltoall(P5.1 peer packet counts)");
  std::vector<U64> expectedCounts(n, 0);
  for (const auto& peer : plan.receivePeers)
    expectedCounts[static_cast<std::size_t>(peer.rank)] = peer.count;
  require(recvCounts == expectedCounts,
          "sender and receiver packet counts differ across MPI ranks");

  std::vector<int> sendN(n, 0), recvN(n, 0);
  std::vector<int> sendDispls(n, 0), recvDispls(n, 0);
  U64 sendTotal = 0, recvTotal = 0;
  for (std::size_t rank = 0; rank < n; ++rank) {
    sendDispls[rank] = asMpiCount(sendTotal, "send displacement");
    recvDispls[rank] = asMpiCount(recvTotal, "receive displacement");
    sendN[rank] = asMpiCount(sendCounts[rank], "send element count");
    recvN[rank] = asMpiCount(recvCounts[rank], "receive element count");
    sendTotal = add(sendTotal, sendCounts[rank], "metadata send total");
    recvTotal = add(recvTotal, recvCounts[rank], "metadata receive total");
    (void)asMpiCount(sendTotal, "total metadata send extent");
    (void)asMpiCount(recvTotal, "total metadata receive extent");
  }
  require(sendTotal == plan.packElements && recvTotal == plan.inboxElements,
          "MPI wire metadata sizes do not match local packet plan");

  // Alltoallv segments are in MPI-rank order. Each segment preserves the
  // *original* exportLID order for that destination, even though the local
  // pack scratch is grouped by NVSHMEM PE order.
  std::vector<U64> sent(asSize(sendTotal, "metadata send"), 0);
  std::vector<U64> received(asSize(recvTotal, "metadata receive"), 0);
  for (const auto& peer : plan.sendPeers) {
    const std::size_t dst = static_cast<std::size_t>(peer.rank);
    const U64 start = static_cast<U64>(sendDispls[dst]);
    for (U64 i = 0; i < peer.count; ++i) {
      sent[asSize(add(start, i, "wire send slot"), "wire send slot")] =
          plan.packedSourceLids[asSize(add(peer.packOffset, i,
                                            "packed send slot"), "packed send slot")];
    }
  }
  U64 dummySend = 0, dummyRecv = 0;
  mpiCheck(MPI_Alltoallv(sent.empty() ? &dummySend : sent.data(),
                         sendN.data(), sendDispls.data(), MPI_UINT64_T,
                         received.empty() ? &dummyRecv : received.data(),
                         recvN.data(), recvDispls.data(), MPI_UINT64_T, comm),
           "MPI_Alltoallv(P5.1 source-LID metadata only)");

  PeerMetadata metadata;
  metadata.sourceLidsFromRank.resize(n);
  for (std::size_t rank = 0; rank < n; ++rank) {
    const auto begin = received.begin() + recvDispls[rank];
    metadata.sourceLidsFromRank[rank].assign(begin, begin + recvN[rank]);
  }

  // A receiver owns disjoint inbox segments, indexed by *source PE*.
  // It advertises its exact local segment offset to each sender.
  std::vector<U64> inboxOffsetForSource(n, noOffset);
  for (const auto& peer : plan.receivePeers)
    inboxOffsetForSource[static_cast<std::size_t>(peer.rank)] = peer.inboxOffset;
  metadata.inboxOffsetAtRank.resize(n, noOffset);
  mpiCheck(MPI_Alltoall(inboxOffsetForSource.data(), 1, MPI_UINT64_T,
                        metadata.inboxOffsetAtRank.data(), 1, MPI_UINT64_T,
                        comm), "MPI_Alltoall(P5.1 receiver inbox offsets)");

  metadata.inboxLengthAtRank.resize(n, 0);
  mpiCheck(MPI_Allgather(&plan.inboxElements, 1, MPI_UINT64_T,
                         metadata.inboxLengthAtRank.data(), 1, MPI_UINT64_T,
                         comm), "MPI_Allgather(P5.1 receiver logical inbox sizes)");
  const U64 localCapacity[2] = {plan.packElements, plan.inboxElements};
  U64 maxCapacity[2] = {0, 0};
  mpiCheck(MPI_Allreduce(localCapacity, maxCapacity, 2, MPI_UINT64_T,
                         MPI_MAX, comm),
           "MPI_Allreduce(P5.1 symmetric allocation capacities)");
  metadata.maxPackElements = maxCapacity[0];
  metadata.maxInboxElements = maxCapacity[1];
  finalizeLocal(plan, metadata);
  return plan;
}

template<class LO, class GO, class Node>
Plan build(const ImportGatherP1::ImportGatherPlan<LO, GO, Node>& p1,
           const NvshmemP0::Session& session) {
  return reconcileMpi(fromP1(p1, session), session.comm());
}

} // namespace PackedPutP51

#endif // TPETRA_PERFORMANCE_PACKED_PUT_PLAN_MPI_HPP
