// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// P5.0: PURE HOST-side, read-only analysis of the P1 ImportGatherPlan.
// No MPI, CUDA, NVSHMEM, packet transfer, or Tpetra library dependency here.
// Source and destination permutations refer to ORIGINAL P1 remote indices.

#ifndef TPETRA_PERFORMANCE_IMPORT_STRUCTURE_ANALYZER_HPP
#define TPETRA_PERFORMANCE_IMPORT_STRUCTURE_ANALYZER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ImportStructureP50 {

inline void require(bool condition, const std::string& reason) {
  if (!condition) throw std::runtime_error("NVSHMEM P5.0: " + reason);
}

inline std::uint64_t add(std::uint64_t lhs, std::uint64_t rhs, const char* field) {
  require(rhs <= std::numeric_limits<std::uint64_t>::max() - lhs,
          std::string("counter overflow: ") + field);
  return lhs + rhs;
}

struct SourceGroup {
  std::uint64_t sourceLid = 0;
  std::size_t beginSourceOrder = 0;  // inclusive, indexing sourceOrder
  std::size_t endSourceOrder = 0;    // exclusive
  std::size_t requests() const { return endSourceOrder - beginSourceOrder; }
};

struct SourceRun {
  std::uint64_t beginSourceLid = 0;
  std::uint64_t endSourceLid = 0; // exclusive; legal within owner-local X
  std::size_t firstGroup = 0;     // inclusive, indexing uniqueSourceGroups
  std::size_t lastGroup = 0;      // exclusive
  std::size_t requests = 0;      // duplicates included
  bool sameOrderDirect = false;   // EXACT 1:1 contiguous destination mapping
  std::uint64_t destinationBegin = 0; // meaningful only when sameOrderDirect
  std::size_t elements() const { return lastGroup - firstGroup; }
};

struct Peer {
  int ownerRank = -1;
  int sourcePe = -1;
  std::uint64_t ownerLength = 0;
  // Stable original indices; P1 plan.remote remains unchanged.
  std::vector<std::size_t> sourceOrder;
  std::vector<std::size_t> destinationOrder;
  std::vector<SourceGroup> uniqueSourceGroups;
  std::vector<SourceRun> sourceRuns;

  std::uint64_t requests = 0, uniqueSources = 0, duplicates = 0;
  std::uint64_t sourceRunCount = 0, sourceMultiRuns = 0;
  std::uint64_t sourceMultiElements = 0, longestSourceRun = 0;
  std::uint64_t destinationRunCount = 0, destinationMultiRuns = 0;
  std::uint64_t destinationMultiElements = 0, longestDestinationRun = 0;
  std::uint64_t alignedSourceRuns = 0, alignedSourceElements = 0;
  std::uint64_t directRuns = 0, directMultiRuns = 0, directMultiElements = 0;
  std::uint64_t directFallbackScalarCalls = 0;
  std::uint64_t scatterIfSourceRuns = 0; // layout-only; legality handled in P5.1
  std::uint64_t largestScratchElements = 0;
  std::uint64_t minSourceLid = 0, maxSourceLid = 0;
  std::uint64_t windowElements = 0, extraWindowElements = 0;
  bool sourceTouchesLastOwned = false;
};

struct Outgoing {
  int destinationRank = -1;
  std::uint64_t requests = 0, uniqueSourceLids = 0;
  std::uint64_t duplicateExports = 0;
  std::uint64_t minSourceLid = 0, maxSourceLid = 0;
  bool adjacentNumericRank = false;
};

struct Rank {
  int rank = -1, pe = -1, nRanks = 0;
  std::uint64_t sourceLength = 0, targetLength = 0;
  std::uint64_t numSame = 0, numPermute = 0;
  std::uint64_t localPrefix = 0, remoteSuffix = 0;
  std::uint64_t requests = 0, exports = 0;
  std::uint64_t sourcePeers = 0, outgoingPeers = 0;
  std::uint64_t sourceRuns = 0, destinationRuns = 0;
  std::uint64_t sourceMultiElements = 0;
  std::uint64_t alignedSourceRuns = 0, alignedSourceElements = 0;
  std::uint64_t directRuns = 0, directMultiRuns = 0, directMultiElements = 0;
  std::uint64_t directLogicalGets = 0, sourceRunLogicalGets = 0;
  std::uint64_t scatterIfSourceRuns = 0, duplicates = 0;
  std::uint64_t windowExtraElements = 0, largestScratchElements = 0;
  std::uint64_t exportPidSegments = 0, adjacentRankExports = 0;
  std::uint64_t fanoutSourceLids = 0, fanoutExtraPackets = 0;
  std::uint64_t adjacentFanoutSourceLids = 0, maxFanoutDestinations = 0;
  bool remoteIsExactSuffix = true, remoteOwnerGrouped = true;
  std::string remoteOwnerRankOrder = "NONE"; // ASC/DESC/MIXED/SINGLE/NONE
  std::vector<int> remoteOwnerOrder; // target-LID order, owner MPI ranks
  std::vector<Peer> peers;           // sorted by source PE
  std::vector<Outgoing> outgoing;    // sorted by destination MPI rank
};

// The analyzer does not reorder or modify any P1 descriptor. It checks
// destination coverage and source bounds again so it can be used safely by
// subsequent P5 algorithms without silently relying on special miniFE maps.
template<class Plan>
Rank analyze(const Plan& plan, int rank, int pe, const std::vector<int>& peForRank) {
  Rank result;
  result.rank = rank; result.pe = pe;
  const std::size_t size = peForRank.size();
  require(size > 0 && size <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
          "invalid MPI rank count");
  result.nRanks = static_cast<int>(size);
  require(rank >= 0 && rank < result.nRanks && pe >= 0 && pe < result.nRanks &&
          peForRank[static_cast<std::size_t>(rank)] == pe,
          "MPI rank / PE mapping inconsistent");
  require(plan.sourceLengthByRank.size() == size,
          "sourceLengthByRank is missing owner bounds");
  {
    const std::set<int> pes(peForRank.begin(), peForRank.end());
    require(pes.size() == size, "rank-to-PE table is not a permutation");
  }
  for (int p : peForRank)
    require(p >= 0 && p < result.nRanks, "rank-to-PE entry out of range");
  result.sourceLength = static_cast<std::uint64_t>(plan.sourceLength);
  result.targetLength = static_cast<std::uint64_t>(plan.targetLength);
  result.numSame = static_cast<std::uint64_t>(plan.numSame);
  result.numPermute = static_cast<std::uint64_t>(plan.numPermute);
  result.requests = static_cast<std::uint64_t>(plan.remote.size());
  result.exports = static_cast<std::uint64_t>(plan.exportLids.size());
  require(plan.sourceLengthByRank[static_cast<std::size_t>(rank)] == result.sourceLength,
          "local owner length disagrees with P1");
  require(result.numSame <= result.sourceLength && result.numSame <= result.targetLength,
          "same-ID prefix exceeds source/target bounds");
  require(result.numSame + result.numPermute == plan.local.size() &&
          plan.local.size() + plan.remote.size() == plan.targetLength,
          "P1 local/remote descriptors do not cover target Map");
  require(plan.exportLids.size() == plan.exportPids.size(),
          "P1 export LID/PID lengths disagree");

  // 1 local, 2 remote. A target Map LID must be assigned exactly once.
  std::vector<unsigned char> classification(plan.targetLength, 0);
  for (const auto& e : plan.local) {
    require(e.sourceLid < result.sourceLength && e.destinationLid < result.targetLength,
            "local descriptor outside source or target Map");
    auto& slot = classification[static_cast<std::size_t>(e.destinationLid)];
    require(slot == 0, "duplicate local destination");
    slot = 1;
  }

  std::map<int, std::vector<std::size_t>> peerIndices;
  std::map<int, int> ownerForPe;
  for (std::size_t i = 0; i < plan.remote.size(); ++i) {
    const auto& e = plan.remote[i];
    require(e.ownerRank >= 0 && e.ownerRank < result.nRanks && e.ownerRank != rank &&
            e.sourcePe >= 0 && e.sourcePe < result.nRanks &&
            peForRank[static_cast<std::size_t>(e.ownerRank)] == e.sourcePe,
            "remote descriptor ownerRank/sourcePe is invalid");
    require(e.sourceLid < plan.sourceLengthByRank[static_cast<std::size_t>(e.ownerRank)] &&
            e.destinationLid < result.targetLength,
            "remote descriptor source/target LID out of bounds");
    auto& slot = classification[static_cast<std::size_t>(e.destinationLid)];
    require(slot == 0, "duplicate remote/local destination");
    slot = 2;
    peerIndices[e.sourcePe].push_back(i);
    ownerForPe[e.sourcePe] = e.ownerRank;
  }
  require(std::find(classification.begin(), classification.end(), 0) ==
              classification.end(), "unclassified target LID");
  require(peerIndices.size() == plan.numRemotePeers, "P1 source-peer count changed");

  while (result.localPrefix < result.targetLength &&
         classification[static_cast<std::size_t>(result.localPrefix)] == 1)
    ++result.localPrefix;
  while (result.remoteSuffix < result.targetLength &&
         classification[static_cast<std::size_t>(result.targetLength - 1 - result.remoteSuffix)] == 2)
    ++result.remoteSuffix;
  result.remoteIsExactSuffix = result.remoteSuffix == result.requests;

  // Remote owner order is measured in actual increasing target LID order.
  // The remote-only subsequence may group peers even when local entries are
  // interleaved: remoteIsExactSuffix separately records that distinction.
  std::vector<std::size_t> remoteTargetOrder(plan.remote.size());
  for (std::size_t i = 0; i < remoteTargetOrder.size(); ++i) remoteTargetOrder[i] = i;
  std::sort(remoteTargetOrder.begin(), remoteTargetOrder.end(), [&](std::size_t a, std::size_t b) {
    return plan.remote[a].destinationLid < plan.remote[b].destinationLid;
  });
  std::set<int> completedOwners;
  int previousOwner = -1;
  for (std::size_t i : remoteTargetOrder) {
    const int owner = plan.remote[i].ownerRank;
    if (owner == previousOwner) continue;
    if (previousOwner != -1) completedOwners.insert(previousOwner);
    if (completedOwners.count(owner)) result.remoteOwnerGrouped = false;
    result.remoteOwnerOrder.push_back(owner);
    previousOwner = owner;
  }
  if (result.remoteOwnerOrder.empty()) result.remoteOwnerRankOrder = "NONE";
  else if (result.remoteOwnerOrder.size() == 1) result.remoteOwnerRankOrder = "SINGLE";
  else {
    bool ascending = true, descending = true;
    for (std::size_t i = 1; i < result.remoteOwnerOrder.size(); ++i) {
      ascending &= result.remoteOwnerOrder[i] > result.remoteOwnerOrder[i-1];
      descending &= result.remoteOwnerOrder[i] < result.remoteOwnerOrder[i-1];
    }
    result.remoteOwnerRankOrder = ascending ? "ASC" : descending ? "DESC" : "MIXED";
    if (!result.remoteOwnerGrouped) result.remoteOwnerRankOrder = "REVISITS";
  }

  for (const auto& element : peerIndices) {
    Peer out;
    out.sourcePe = element.first;
    out.ownerRank = ownerForPe.at(out.sourcePe);
    out.ownerLength = plan.sourceLengthByRank[static_cast<std::size_t>(out.ownerRank)];
    out.sourceOrder = element.second;
    out.destinationOrder = element.second;
    std::sort(out.sourceOrder.begin(), out.sourceOrder.end(), [&](std::size_t a, std::size_t b) {
      const auto& x = plan.remote[a]; const auto& y = plan.remote[b];
      if (x.sourceLid != y.sourceLid) return x.sourceLid < y.sourceLid;
      return x.destinationLid < y.destinationLid;
    });
    std::sort(out.destinationOrder.begin(), out.destinationOrder.end(), [&](std::size_t a, std::size_t b) {
      return plan.remote[a].destinationLid < plan.remote[b].destinationLid;
    });
    out.requests = static_cast<std::uint64_t>(out.sourceOrder.size());
    require(out.requests > 0, "empty incoming source peer");
    for (std::size_t i = 0; i < out.sourceOrder.size();) {
      const std::uint64_t source = plan.remote[out.sourceOrder[i]].sourceLid;
      const std::size_t begin = i;
      while (i < out.sourceOrder.size() &&
             plan.remote[out.sourceOrder[i]].sourceLid == source) ++i;
      out.uniqueSourceGroups.push_back(SourceGroup{source, begin, i});
    }
    out.uniqueSources = static_cast<std::uint64_t>(out.uniqueSourceGroups.size());
    out.duplicates = out.requests - out.uniqueSources;
    out.minSourceLid = out.uniqueSourceGroups.front().sourceLid;
    out.maxSourceLid = out.uniqueSourceGroups.back().sourceLid;
    require(out.maxSourceLid < out.ownerLength, "source window exceeds owner length");
    out.sourceTouchesLastOwned = out.maxSourceLid + 1 == out.ownerLength;
    out.windowElements = out.maxSourceLid - out.minSourceLid + 1;
    out.extraWindowElements = out.windowElements - out.uniqueSources;

    // Unique sources in increasing owner-local LID order; duplicates of one
    // source are recorded separately, not misclassified as additional runs.
    for (std::size_t i = 0; i < out.uniqueSourceGroups.size();) {
      const std::size_t start = i;
      std::size_t requestCount = 0;
      while (i < out.uniqueSourceGroups.size()) {
        const auto& g = out.uniqueSourceGroups[i];
        requestCount += g.requests();
        ++i;
        if (i == out.uniqueSourceGroups.size() ||
            out.uniqueSourceGroups[i].sourceLid != g.sourceLid + 1) break;
      }
      const auto first = out.uniqueSourceGroups[start].sourceLid;
      const auto last = out.uniqueSourceGroups[i-1].sourceLid;
      SourceRun r;
      r.beginSourceLid = first; r.endSourceLid = last + 1;
      r.firstGroup = start; r.lastGroup = i; r.requests = requestCount;
      const auto& firstGroup = out.uniqueSourceGroups[start];
      r.destinationBegin = plan.remote[out.sourceOrder[firstGroup.beginSourceOrder]].destinationLid;
      r.sameOrderDirect = true;
      for (std::size_t k = start; k < i; ++k) {
        const auto& g = out.uniqueSourceGroups[k];
        if (g.requests() != 1 ||
            plan.remote[out.sourceOrder[g.beginSourceOrder]].destinationLid !=
                r.destinationBegin + (k-start)) {
          r.sameOrderDirect = false;
          break;
        }
      }
      out.sourceRuns.push_back(r);
      if (r.sameOrderDirect) {
        ++out.alignedSourceRuns;
        out.alignedSourceElements += static_cast<std::uint64_t>(r.elements());
      }
      out.largestScratchElements = std::max(out.largestScratchElements,
                                             static_cast<std::uint64_t>(r.elements()));
      out.longestSourceRun = std::max(out.longestSourceRun,
                                      static_cast<std::uint64_t>(r.elements()));
      if (r.elements() > 1) {
        ++out.sourceMultiRuns;
        out.sourceMultiElements += static_cast<std::uint64_t>(r.elements());
      }
      if (!r.sameOrderDirect) out.scatterIfSourceRuns += r.requests;
    }
    out.sourceRunCount = static_cast<std::uint64_t>(out.sourceRuns.size());

    // Independent destination adjacency, regardless of source-LID order.
    for (std::size_t i = 0; i < out.destinationOrder.size();) {
      std::size_t j = i + 1;
      while (j < out.destinationOrder.size() &&
             plan.remote[out.destinationOrder[j]].destinationLid ==
                 plan.remote[out.destinationOrder[j-1]].destinationLid + 1) ++j;
      const std::uint64_t len = static_cast<std::uint64_t>(j-i);
      out.longestDestinationRun = std::max(out.longestDestinationRun, len);
      ++out.destinationRunCount;
      if (len > 1) {++out.destinationMultiRuns; out.destinationMultiElements += len;}
      i = j;
    }

    // Maximal runs which are SIMULTANEOUSLY consecutive in source and
    // destination, in the same order, with exactly one target per source.
    // These are layout candidates for direct bulk GET, not a claim that the
    // installed NVSHMEM API can write directly to arbitrary CudaSpace Y.
    for (std::size_t i = 0; i < out.uniqueSourceGroups.size();) {
      if (out.uniqueSourceGroups[i].requests() != 1) { ++i; continue; }
      std::size_t j = i + 1;
      while (j < out.uniqueSourceGroups.size() &&
             out.uniqueSourceGroups[j].requests() == 1 &&
             out.uniqueSourceGroups[j].sourceLid ==
                 out.uniqueSourceGroups[j-1].sourceLid + 1 &&
             plan.remote[out.sourceOrder[out.uniqueSourceGroups[j].beginSourceOrder]].destinationLid ==
                 plan.remote[out.sourceOrder[out.uniqueSourceGroups[j-1].beginSourceOrder]].destinationLid + 1) ++j;
      ++out.directRuns;
      if (j - i > 1) {
        ++out.directMultiRuns;
        out.directMultiElements += static_cast<std::uint64_t>(j-i);
      }
      i = j;
    }
    out.directFallbackScalarCalls = out.requests - out.directMultiElements;

    result.sourceRuns += out.sourceRunCount;
    result.destinationRuns += out.destinationRunCount;
    result.sourceMultiElements += out.sourceMultiElements;
    result.alignedSourceRuns += out.alignedSourceRuns;
    result.alignedSourceElements += out.alignedSourceElements;
    result.directRuns += out.directRuns;
    result.directMultiRuns += out.directMultiRuns;
    result.directMultiElements += out.directMultiElements;
    result.directLogicalGets += out.directMultiRuns + out.directFallbackScalarCalls;
    result.sourceRunLogicalGets += out.sourceRunCount;
    result.scatterIfSourceRuns += out.scatterIfSourceRuns;
    result.windowExtraElements += out.extraWindowElements;
    result.duplicates += out.duplicates;
    result.largestScratchElements = std::max(result.largestScratchElements,
                                               out.largestScratchElements);
    result.peers.push_back(std::move(out));
  }
  result.sourcePeers = static_cast<std::uint64_t>(result.peers.size());

  std::map<int, std::vector<std::uint64_t>> outgoingLids;
  std::map<std::uint64_t, std::set<int>> fanoutByLid;
  int previousPid = -1;
  for (std::size_t i = 0; i < plan.exportLids.size(); ++i) {
    const int pid = plan.exportPids[i];
    require(pid >= 0 && pid < result.nRanks && pid != rank,
            "invalid export destination rank");
    using Lid = typename std::decay<decltype(plan.exportLids[i])>::type;
    const Lid lid = plan.exportLids[i];
    require(!std::numeric_limits<Lid>::is_signed || lid >= Lid(0),
            "negative exported source LID");
    const std::uint64_t src = static_cast<std::uint64_t>(lid);
    require(src < result.sourceLength, "export LID beyond local source length");
    outgoingLids[pid].push_back(src);
    fanoutByLid[src].insert(pid);
    if (pid != previousPid) ++result.exportPidSegments;
    previousPid = pid;
    const long long distance = static_cast<long long>(pid) - rank;
    if (distance == 1 || distance == -1) ++result.adjacentRankExports;
  }
  for (const auto& item : outgoingLids) {
    Outgoing out;
    out.destinationRank = item.first;
    out.requests = static_cast<std::uint64_t>(item.second.size());
    const std::set<std::uint64_t> unique(item.second.begin(), item.second.end());
    out.uniqueSourceLids = static_cast<std::uint64_t>(unique.size());
    out.duplicateExports = out.requests - out.uniqueSourceLids;
    out.minSourceLid = *unique.begin(); out.maxSourceLid = *unique.rbegin();
    const long long distance = static_cast<long long>(item.first) - rank;
    out.adjacentNumericRank = distance == 1 || distance == -1;
    result.outgoing.push_back(out);
  }
  result.outgoingPeers = static_cast<std::uint64_t>(result.outgoing.size());
  for (const auto& item : fanoutByLid) {
    const auto& targets = item.second;
    result.maxFanoutDestinations = std::max(result.maxFanoutDestinations,
                                           static_cast<std::uint64_t>(targets.size()));
    if (targets.size() > 1) {
      ++result.fanoutSourceLids;
      result.fanoutExtraPackets += static_cast<std::uint64_t>(targets.size() - 1);
    }
    int adjacentTargets = 0;
    for (int pid : targets) {
      const long long distance = static_cast<long long>(pid) - rank;
      if (distance == 1 || distance == -1) ++adjacentTargets;
    }
    if (adjacentTargets >= 2) ++result.adjacentFanoutSourceLids;
  }
  return result;
}

} // namespace ImportStructureP50
#endif // TPETRA_PERFORMANCE_IMPORT_STRUCTURE_ANALYZER_HPP
