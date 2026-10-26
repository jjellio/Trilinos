// Standalone CPU/C++17 unit test for matrix-free P7.1b Import graph design.
#include "SyntheticImportGraph.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace S = SyntheticImportGraph;
using S::U64;
static void test(const std::vector<std::string>& hosts, int off, int intra,
                 S::TransferVolume volume, U64 extent, U64 size,
                 S::SourcePattern source, S::TargetLayout layout, U64 local) {
  S::Placement p(hosts);
  S::Config cfg;
  cfg.ownedPerRank = extent;
  cfg.offNodeDegree = off;
  cfg.intraNodeDegree = intra;
  cfg.volume = volume;
  cfg.valuesPerPeer = size;
  cfg.remoteValuesTotal = size;
  cfg.source = source;
  cfg.target = layout;
  cfg.localValues = local;
  cfg.sourceStride = 17;
  S::Graph graph(p,cfg);
  for (int rank = 0; rank < p.nranks; ++rank) {
    auto maps = S::mapsForRank(graph,rank);
    assert(maps.owned.size() == extent);
    assert(maps.target.size() == graph.doublesPerRank + local);
    std::map<int,U64> srcCounts;
    std::map<int,U64> dstCounts;
    for (const auto& edge : graph.outgoing[rank]) {
      assert(edge.sender == rank && edge.receiver != rank);
      assert((p.nodeOfRank[rank] != p.nodeOfRank[edge.receiver]) == edge.offNode);
      srcCounts[edge.receiver] += edge.values;
      assert(graph.count(rank,edge.receiver) == edge.values);
    }
    for (const auto& edge : graph.incoming[rank])
      dstCounts[edge.sender] += edge.values;
    assert(static_cast<int>(srcCounts.size()) == off + intra);
    assert(static_cast<int>(dstCounts.size()) == off + intra);
    for (int other = 0; other < p.nranks; ++other) {
      if (other == rank) continue;
      U64 count = 0;
      for (const auto& e : graph.outgoing[other]) if (e.receiver == rank) count += e.values;
      const auto found = dstCounts.find(other);
      assert(count == (found == dstCounts.end() ? 0 : found->second));
    }
    std::map<int,U64> actual;
    for (U64 gid : maps.target) {
      int owner = static_cast<int>(gid / extent);
      assert(owner >= 0 && owner < p.nranks);
      assert(gid % extent < extent);
      if (owner != rank) actual[owner]++;
    }
    assert(actual == dstCounts);
    auto maps2 = S::mapsForRank(graph,rank);
    assert(maps.owned == maps2.owned && maps.target == maps2.target);
  }
}
static void errors() {
  auto rejects = [](const auto& action) {
    bool caught = false;
    try { action(); } catch (const std::runtime_error&) { caught = true; }
    assert(caught);
  };
  rejects([]{ S::Placement({"A","B","B"}); });
  rejects([]{ S::Config c; c.offNodeDegree=2; S::Graph(S::Placement({"A","B"}),c); });
  rejects([]{ S::Config c; c.offNodeDegree=0; c.localValues=0; S::Graph(S::Placement({"A"}),c); });
  rejects([]{ S::Config c; c.offNodeDegree=1; c.ownedPerRank=128; c.valuesPerPeer=129; S::Graph(S::Placement({"A","B"}),c); });
  rejects([]{ S::Config c; c.offNodeDegree=1; c.ownedPerRank=128; c.source=S::SourcePattern::Strided; c.sourceStride=2; S::Graph(S::Placement({"A","B"}),c); });
  rejects([]{ S::Config c; c.offNodeDegree=1; c.volume=S::TransferVolume::FixedTotal; c.remoteValuesTotal=0; S::Graph(S::Placement({"A","B"}),c); });
  rejects([]{ S::Config c; c.offNodeDegree=1; c.localValues=2; S::Graph(S::Placement({"A","B"}),c); });
  rejects([]{ (void)S::parseSourcePattern("unknown"); });
  rejects([]{ (void)S::parseTargetLayout("unknown"); });
}
int main() {
  try {
    errors();
    for (int nodes : {2,3,4}) {
      for (int ppn : {1,4}) {
        // Interleave ranks across hosts to ensure there is no rank-contiguity assumption.
        std::vector<std::string> names;
        for (int slot = 0; slot < ppn; ++slot)
          for (int n = 0; n < nodes; ++n)
            names.push_back("node" + std::to_string(n));
        const int offMax = (nodes - 1)*ppn;
        for (int off : {0,1,2,3,4,8,12}) {
          if (off > offMax) continue;
          for (int intra : {0,1,3}) {
            if (intra >= ppn || off+intra == 0) continue;
            for (auto mode : {S::TransferVolume::PerPeer, S::TransferVolume::FixedTotal}) {
              for (auto pattern : {S::SourcePattern::Contiguous,
                                   S::SourcePattern::Strided, S::SourcePattern::Permuted}) {
                for (auto layout : {S::TargetLayout::RemoteOnly,
                                    S::TargetLayout::PrefixLocal,
                                    S::TargetLayout::InterleavedLocal,
                                    S::TargetLayout::ShuffledLocal}) {
                  const U64 local = layout == S::TargetLayout::RemoteOnly ? 0 : 5;
                  test(names,off,intra,mode,127,off+intra == 0 ? 0 : 23,
                       pattern,layout,local);
                }
              }
            }
          }
        }
        std::cout << "[P7.1b GRAPH] nodes=" << nodes << " ppn=" << ppn
                  << " mixedRankLayout=YES status=PASS\n";
      }
    }
    // Realistic large 16-PE/4-node controlled Import. No matrix is made.
    {
      std::vector<std::string> hosts;
      for (int r = 0; r < 16; ++r)
        hosts.push_back("gpu-node-" + std::to_string(r / 4));
      S::Config c;
      c.ownedPerRank = 65536;
      c.offNodeDegree = 12;
      c.valuesPerPeer = 8192;
      c.source = S::SourcePattern::Permuted;
      S::Graph g(S::Placement(hosts),c);
      assert(g.doublesPerRank == 98304);
      assert(g.crossNodeDoublesPerRank == 98304);
      auto local = S::mapsForRank(g,0);
      assert(local.owned.size() == 65536 && local.target.size() == 98304);
      std::set<U64> unique(local.target.begin(),local.target.end());
      assert(unique.size() == local.target.size());
      c.volume = S::TransferVolume::FixedTotal;
      c.remoteValuesTotal = 8192;
      S::Graph fixed(S::Placement(hosts),c);
      assert(fixed.doublesPerRank == 8192);
      assert(S::mapsForRank(fixed,0).target.size() == 8192);
      std::cout << "[P7.1b GRAPH] large16PE=YES fixedTotalVsPerPeer=YES status=PASS\n";
    }
    // Deliberately unsorted hostnames and odd node count with degree 1.
    test({"zz","aa","mm"},1,0,S::TransferVolume::PerPeer,64,7,
         S::SourcePattern::Contiguous,S::TargetLayout::RemoteOnly,0);
    std::cout << "[P7.1b GRAPH] RESULT=PASS\n";
  } catch (const std::exception& e) {
    std::cerr << "[P7.1b GRAPH] RESULT=FAIL " << e.what() << '\n';
    return 1;
  }
}
