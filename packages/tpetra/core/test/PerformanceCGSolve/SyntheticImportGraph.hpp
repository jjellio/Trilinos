// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
// P7.1b: portable, MPI/Tpetra/NVSHMEM-free, matrix-free Import graph generator.
// Includes real-node-aware directed permutations with equal in/out degrees.
#ifndef TPETRA_PERFORMANCE_SYNTHETIC_IMPORT_GRAPH_HPP
#define TPETRA_PERFORMANCE_SYNTHETIC_IMPORT_GRAPH_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace SyntheticImportGraph {
using U64 = std::uint64_t;

inline void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("P7.1b Import graph: " + message);
}
inline U64 add(U64 a, U64 b, const char* why) {
  check(b <= std::numeric_limits<U64>::max() - a, std::string(why) + " overflow");
  return a + b;
}
inline U64 product(U64 a, U64 b, const char* why) {
  check(!a || b <= std::numeric_limits<U64>::max() / a,
        std::string(why) + " overflow");
  return a * b;
}
inline std::size_t asSize(U64 n, const char* why) {
  check(n <= static_cast<U64>(std::numeric_limits<std::size_t>::max()),
        std::string(why) + " exceeds size_t");
  return static_cast<std::size_t>(n);
}

struct Placement {
  int nranks = 0;
  int nnodes = 0;
  int ppn = 0;
  std::vector<std::string> nodeNames;
  std::vector<int> nodeOfRank, localSlotOfRank;
  std::vector<std::vector<int>> ranksByNode;

  explicit Placement(const std::vector<std::string>& mpiHostnames) {
    check(!mpiHostnames.empty(), "no MPI ranks");
    check(mpiHostnames.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
          "too many MPI ranks");
    nranks = static_cast<int>(mpiHostnames.size());
    std::map<std::string, std::vector<int>> onHost;
    for (int r = 0; r < nranks; ++r) {
      check(!mpiHostnames[static_cast<std::size_t>(r)].empty(), "empty MPI hostname");
      onHost[mpiHostnames[static_cast<std::size_t>(r)]].push_back(r);
    }
    nnodes = static_cast<int>(onHost.size());
    ppn = static_cast<int>(onHost.begin()->second.size());
    check(ppn > 0, "node without ranks");
    nodeOfRank.resize(nranks, -1);
    localSlotOfRank.resize(nranks, -1);
    for (const auto& host : onHost) {
      check(static_cast<int>(host.second.size()) == ppn,
            "unequal MPI ranks per physical node: " + host.first);
      const int node = static_cast<int>(nodeNames.size());
      nodeNames.push_back(host.first);
      ranksByNode.push_back(host.second); // sorted by MPI rank, not assumed rank contiguous
      for (int i = 0; i < ppn; ++i) {
        const int r = host.second[static_cast<std::size_t>(i)];
        nodeOfRank[static_cast<std::size_t>(r)] = node;
        localSlotOfRank[static_cast<std::size_t>(r)] = i;
      }
    }
    check(nnodes * ppn == nranks, "invalid node/slot bijection");
  }
  int rankAt(int node, int localSlot) const {
    check(node >= 0 && node < nnodes && localSlot >= 0 && localSlot < ppn,
          "rankAt node or slot is out of range");
    return ranksByNode[static_cast<std::size_t>(node)][static_cast<std::size_t>(localSlot)];
  }
};

enum class TransferVolume { PerPeer, FixedTotal };
enum class SourcePattern { Contiguous, Strided, Permuted };
enum class TargetLayout { RemoteOnly, PrefixLocal, InterleavedLocal, ShuffledLocal };

inline SourcePattern parseSourcePattern(const std::string& name) {
  if (name == "contiguous") return SourcePattern::Contiguous;
  if (name == "strided") return SourcePattern::Strided;
  if (name == "permuted") return SourcePattern::Permuted;
  throw std::runtime_error("P7.1b source pattern must be contiguous|strided|permuted");
}
inline TargetLayout parseTargetLayout(const std::string& name) {
  if (name == "remote-only") return TargetLayout::RemoteOnly;
  if (name == "prefix") return TargetLayout::PrefixLocal;
  if (name == "interleave") return TargetLayout::InterleavedLocal;
  if (name == "shuffle") return TargetLayout::ShuffledLocal;
  throw std::runtime_error("P7.1b target layout must be remote-only|prefix|interleave|shuffle");
}
inline const char* sourceName(SourcePattern p) {
  switch(p) {
    case SourcePattern::Contiguous: return "contiguous";
    case SourcePattern::Strided: return "strided";
    default: return "permuted";
  }
}

struct Config {
  U64 ownedPerRank = 65536;
  int offNodeDegree = 1;
  int intraNodeDegree = 0;
  TransferVolume volume = TransferVolume::PerPeer;
  U64 valuesPerPeer = 1024;
  U64 remoteValuesTotal = 8192;
  U64 localValues = 0;
  SourcePattern source = SourcePattern::Contiguous;
  TargetLayout target = TargetLayout::RemoteOnly;
  U64 sourceStride = 17;
  U64 seed = 12345;
};

struct Edge {
  int sender = -1;
  int receiver = -1;
  int layer = -1;    // permutation layer, 0..degree-1
  U64 values = 0;   // same count at both endpoints
  bool offNode = false;
};

struct Graph {
  Placement placement;
  Config config;
  std::vector<std::vector<Edge>> outgoing, incoming;
  U64 doublesPerRank = 0;
  U64 crossNodeDoublesPerRank = 0;
  explicit Graph(const Placement& p, const Config& cfg)
      : placement(p), config(cfg), outgoing(p.nranks), incoming(p.nranks) {
    check(cfg.ownedPerRank > 0, "ownedPerRank must be positive");
    check(cfg.offNodeDegree >= 0 && cfg.intraNodeDegree >= 0,
          "peer degrees must be nonnegative");
    check(cfg.offNodeDegree <= (p.nnodes - 1) * p.ppn,
          "off-node degree exceeds available remote GPUs");
    check(cfg.intraNodeDegree <= p.ppn - 1,
          "intra-node degree exceeds available peer GPUs");
    check(cfg.localValues <= cfg.ownedPerRank, "localValues exceeds ownedPerRank");
    if (cfg.target == TargetLayout::RemoteOnly)
      check(cfg.localValues == 0, "remote-only layout requires localValues=0");
    const int totalDegree = cfg.offNodeDegree + cfg.intraNodeDegree;
    check(totalDegree > 0 || cfg.localValues > 0,
          "empty workload: choose a peer degree or localValues > 0");
    if (cfg.volume == TransferVolume::PerPeer)
      check(cfg.valuesPerPeer > 0 || totalDegree == 0,
            "valuesPerPeer must be positive for remote traffic");
    else
      check(totalDegree > 0 ?
                cfg.remoteValuesTotal >= static_cast<U64>(totalDegree) :
                cfg.remoteValuesTotal == 0,
            "fixed-total requires at least one double per peer, or zero with no peers");
    if (cfg.source == SourcePattern::Strided)
      check(cfg.sourceStride > 0 && std::gcd(cfg.ownedPerRank, cfg.sourceStride) == 1,
            "sourceStride must be nonzero and coprime with ownedPerRank");
    const U64 nLayers = static_cast<U64>(totalDegree);
    for (int layer = 0; layer < totalDegree; ++layer) {
      const bool offNode = layer < cfg.offNodeDegree;
      const int relative = offNode ? layer : layer - cfg.offNodeDegree;
      const int nodeDelta = offNode ? (relative / p.ppn) + 1 : 0;
      const int slotDelta = offNode ? relative % p.ppn : relative + 1;
      U64 count = cfg.valuesPerPeer;
      if (cfg.volume == TransferVolume::FixedTotal) {
        count = cfg.remoteValuesTotal / nLayers;
        if (static_cast<U64>(layer) < cfg.remoteValuesTotal % nLayers) ++count;
      }
      check(count > 0 && count <= cfg.ownedPerRank,
            "per-peer request count must be between 1 and ownedPerRank");
      for (int sender = 0; sender < p.nranks; ++sender) {
        const int node = p.nodeOfRank[static_cast<std::size_t>(sender)];
        const int slot = p.localSlotOfRank[static_cast<std::size_t>(sender)];
        const int destNode = (node + nodeDelta) % p.nnodes;
        const int destSlot = (slot + slotDelta) % p.ppn;
        const int receiver = p.rankAt(destNode, destSlot);
        check(receiver != sender, "self edge generated");
        Edge edge{sender, receiver, layer, count, offNode};
        outgoing[static_cast<std::size_t>(sender)].push_back(edge);
        incoming[static_cast<std::size_t>(receiver)].push_back(edge);
      }
      doublesPerRank = add(doublesPerRank, count, "total remote values");
      if (offNode)
        crossNodeDoublesPerRank = add(crossNodeDoublesPerRank, count, "cross-node values");
    }
    check(cfg.localValues <= static_cast<U64>(std::numeric_limits<std::size_t>::max()) &&
          add(doublesPerRank, cfg.localValues, "target length") <=
            static_cast<U64>(std::numeric_limits<std::size_t>::max()),
          "target length too large");
    for (int rank = 0; rank < p.nranks; ++rank) {
      const auto& send = outgoing[static_cast<std::size_t>(rank)];
      const auto& recv = incoming[static_cast<std::size_t>(rank)];
      check(static_cast<int>(send.size()) == totalDegree &&
            static_cast<int>(recv.size()) == totalDegree,
            "permutation graph not degree-balanced");
      std::set<int> sendPeers, recvPeers;
      U64 ns = 0, nr = 0;
      for (const auto& edge : send) {
        check(sendPeers.insert(edge.receiver).second, "duplicate send peer");
        ns = add(ns, edge.values, "rank send volume");
      }
      for (const auto& edge : recv) {
        check(recvPeers.insert(edge.sender).second, "duplicate receive peer");
        nr = add(nr, edge.values, "rank receive volume");
      }
      check(ns == doublesPerRank && nr == doublesPerRank,
            "rank's sent and received volumes differ");
    }
    if (cfg.volume == TransferVolume::FixedTotal && totalDegree > 0)
      check(doublesPerRank == cfg.remoteValuesTotal,
            "fixed-volume total differs from requested value count");
  }
  U64 count(int sender, int receiver) const {
    check(sender >= 0 && sender < placement.nranks && receiver >= 0 &&
          receiver < placement.nranks, "peer rank out of bounds");
    for (const auto& e : outgoing[static_cast<std::size_t>(sender)])
      if (e.receiver == receiver) return e.values;
    return 0;
  }
};

inline U64 hash(U64 input) { // deterministic avalanche, not for cryptography
  input += UINT64_C(0x9e3779b97f4a7c15);
  input = (input ^ (input >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  input = (input ^ (input >> 27)) * UINT64_C(0x94d049bb133111eb);
  return input ^ (input >> 31);
}
inline U64 pickCoprimeStep(U64 extent, U64 seed) {
  if (extent <= 1) return 1;
  U64 candidate = seed % extent;
  if (candidate == 0) candidate = 1;
  while (std::gcd(candidate, extent) != 1) {
    candidate = (candidate + 1) % extent;
    if (candidate == 0) candidate = 1;
  }
  return candidate;
}
inline U64 selectedSourceLid(const Config& cfg, const Edge& edge, U64 ordinal) {
  check(ordinal < edge.values, "request ordinal out of bounds");
  const U64 n = cfg.ownedPerRank;
  const U64 key = hash(cfg.seed ^ (static_cast<U64>(edge.sender) << 32) ^
                       static_cast<U64>(edge.receiver));
  if (cfg.source == SourcePattern::Contiguous) {
    const U64 span = n - edge.values + 1;
    return (key % span) + ordinal; // no wrap; all neighboring source LIDs
  }
  const U64 step = cfg.source == SourcePattern::Strided ? cfg.sourceStride :
                   pickCoprimeStep(n, (key >> 10) + n / 3);
  // Avoid multiplication overflow even when n is near uint64 limits.
  // Accumulate modular steps without overflows (fast shortcut for common sizes).
  const U64 start = key % n;
  if (n <= (1ULL << 32)) return (start + ((ordinal * step) % n)) % n;
  U64 lid = start;
  for (U64 i = 0; i < ordinal; ++i) {
    if (lid >= n - step) lid -= (n - step);
    else lid += step;
  }
  return lid;
}

inline U64 gid(int ownerRank, U64 localLid, U64 ownedPerRank) {
  check(ownerRank >= 0 && localLid < ownedPerRank, "invalid owner/LID");
  return add(product(static_cast<U64>(ownerRank), ownedPerRank, "global GID"),
             localLid, "global GID");
}

struct LocalMaps {
  std::vector<U64> owned, target;
};

inline LocalMaps mapsForRank(const Graph& graph, int rank) {
  check(rank >= 0 && rank < graph.placement.nranks, "mapsForRank invalid rank");
  const auto& cfg = graph.config;
  LocalMaps maps;
  const U64 nOwned = cfg.ownedPerRank;
  maps.owned.reserve(asSize(nOwned, "source GID count"));
  for (U64 lid = 0; lid < nOwned; ++lid)
    maps.owned.push_back(gid(rank, lid, nOwned));
  std::vector<U64> remote;
  remote.reserve(asSize(graph.doublesPerRank, "incoming remote GIDs"));
  for (const auto& edge : graph.incoming[static_cast<std::size_t>(rank)])
    for (U64 i = 0; i < edge.values; ++i)
      remote.push_back(gid(edge.sender, selectedSourceLid(cfg, edge, i), nOwned));
  std::vector<U64> local;
  local.reserve(asSize(cfg.localValues, "local target GIDs"));
  for (U64 i = 0; i < cfg.localValues; ++i)
    local.push_back(gid(rank, i, nOwned));
  maps.target.reserve(remote.size() + local.size());
  if (cfg.target == TargetLayout::InterleavedLocal) {
    const U64 total = static_cast<U64>(remote.size() + local.size());
    U64 li = 0, ri = 0;
    for (U64 i = 0; i < total; ++i) {
      // Proportional distribution, including when local count > remote count.
      const U64 current = product(i + 1, static_cast<U64>(local.size()),
                                  "interleaved local/remote mapping") / total;
      if (li < current) maps.target.push_back(local[asSize(li++, "local index")]);
      else maps.target.push_back(remote[asSize(ri++, "remote index")]);
    }
    check(li == local.size() && ri == remote.size(), "interleaved target lost entries");
  } else {
    maps.target.insert(maps.target.end(), local.begin(), local.end());
    maps.target.insert(maps.target.end(), remote.begin(), remote.end());
    if (cfg.target == TargetLayout::ShuffledLocal) {
      U64 state = hash(cfg.seed ^ static_cast<U64>(rank));
      for (std::size_t i = maps.target.size(); i > 1; --i) {
        state = hash(state);
        std::swap(maps.target[i-1], maps.target[static_cast<std::size_t>(state % i)]);
      }
    }
  }
  std::set<U64> unique(maps.target.begin(), maps.target.end());
  check(unique.size() == maps.target.size(), "target Map has duplicate GIDs");
  return maps;
}
} // namespace SyntheticImportGraph
#endif
