// Host-only P5.0 unit tests: no Trilinos, CUDA, MPI or NVSHMEM required.
#include "ImportStructureAnalyzer.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ImportStructureP50;
struct Local {std::uint64_t sourceLid, destinationLid;};
struct Remote {int ownerRank, sourcePe; std::uint64_t sourceLid, destinationLid;};
struct Plan {
  std::vector<Local> local;
  std::vector<Remote> remote;
  std::vector<int> exportLids, exportPids;
  std::size_t sourceLength=0, targetLength=0, numSame=0, numPermute=0, numRemotePeers=0;
  std::vector<unsigned long long> sourceLengthByRank;
};

template<class Fn> void expectThrow(Fn fn) {
  bool rejected=false;
  try {fn();} catch(const std::runtime_error&) {rejected=true;}
  assert(rejected && "invalid descriptor was accepted");
}
Plan localOnly(std::size_t sourceLen, std::size_t targetLen) {
  assert(sourceLen >= targetLen);
  Plan p;
  p.sourceLength=sourceLen;p.targetLength=targetLen;p.numSame=targetLen;
  p.sourceLengthByRank={static_cast<unsigned long long>(sourceLen)};
  for(std::size_t i=0;i<targetLen;++i)p.local.push_back({i,i});
  return p;
}
Plan multiPeer() {
  Plan p; p.sourceLength=5;p.targetLength=9;p.numSame=2;p.numPermute=2;
  p.sourceLengthByRank={5,10,10};
  p.local={{0,0},{1,1},{4,3},{3,6}};
  p.remote={{1,1,6,4},{2,2,4,8},{1,1,5,2},{2,2,3,7},{1,1,7,5}};
  p.numRemotePeers=2;
  p.exportLids={0,0,1};p.exportPids={1,2,2};
  return p;
}
int main() {
  {
    const auto p=localOnly(0,0);
    const auto a=analyze(p,0,0,{0});
    assert(a.requests==0 && a.sourcePeers==0 && a.sourceRuns==0);
    assert(a.remoteIsExactSuffix && a.remoteOwnerGrouped && a.localPrefix==0);
  }
  {
    Plan p=localOnly(5,5);
    const auto a=analyze(p,0,0,{0});
    assert(a.localPrefix==5 && a.numSame==5 && a.remoteSuffix==0);
    assert(a.sourcePeers==0 && a.outgoingPeers==0);
  }
  {
    Plan p=multiPeer();
    const auto original=p.remote;
    const auto a=analyze(p,0,0,{0,1,2});
    assert(a.requests==5 && a.sourcePeers==2 && a.sourceRuns==2);
    assert(a.localPrefix==2 && a.remoteSuffix==2 && !a.remoteIsExactSuffix);
    assert(a.remoteOwnerGrouped && a.remoteOwnerRankOrder=="ASC");
    assert(a.remoteOwnerOrder == std::vector<int>({1,2}));
    assert(a.directMultiRuns==2 && a.directMultiElements==4 && a.directLogicalGets==3);
    assert(a.sourceRunLogicalGets==2 && a.scatterIfSourceRuns==3);
    assert(a.adjacentRankExports==1 && a.fanoutSourceLids==1);
    assert(a.fanoutExtraPackets==1 && a.maxFanoutDestinations==2);
    assert(a.peers[0].sourceRunCount==1 && a.peers[0].longestSourceRun==3);
    assert(a.peers[0].destinationRunCount==2 && a.peers[0].longestDestinationRun==2);
    assert(a.peers[0].sourceRuns.size()==1 && !a.peers[0].sourceRuns[0].sameOrderDirect);
    assert(a.peers[1].sourceRuns[0].sameOrderDirect &&
           a.peers[1].sourceRuns[0].destinationBegin==7);
    assert(p.remote.size()==original.size());
    for(std::size_t i=0;i<p.remote.size();++i) {
      assert(p.remote[i].destinationLid==original[i].destinationLid &&
             p.remote[i].sourceLid==original[i].sourceLid);
    }
    assert(a.peers[0].sourceOrder == std::vector<std::size_t>({2,0,4}));
    assert(a.peers[0].destinationOrder == std::vector<std::size_t>({2,0,4}));
  }
  {
    Plan p; p.sourceLength=5;p.targetLength=8;p.numSame=5;
    p.sourceLengthByRank={5,6};p.numRemotePeers=1;
    for(unsigned i=0;i<5;++i)p.local.push_back({i,i});
    p.remote={{1,1,3,6},{1,1,4,7},{1,1,2,5}};
    const auto a=analyze(p,0,0,{0,1});
    assert(a.localPrefix==5 && a.remoteSuffix==3 && a.remoteIsExactSuffix);
    assert(a.sourceRuns==1 && a.directMultiRuns==1 && a.directMultiElements==3);
    assert(a.directLogicalGets==1 && a.scatterIfSourceRuns==0);
    assert(a.peers[0].sourceRuns[0].sameOrderDirect &&
           a.peers[0].sourceRuns[0].destinationBegin==5);
  }
  {
    Plan p; p.sourceLength=3;p.targetLength=6;p.numSame=3;
    p.sourceLengthByRank={3,12};p.numRemotePeers=1;
    for(unsigned i=0;i<3;++i)p.local.push_back({i,i});
    p.remote={{1,1,4,3},{1,1,4,4},{1,1,5,5}};
    const auto a=analyze(p,0,0,{0,1});
    assert(a.duplicates==1 && a.sourceRunLogicalGets==1);
    assert(a.directMultiRuns==0 && a.directLogicalGets==3);
    assert(a.peers[0].uniqueSources==2 && a.peers[0].sourceRuns.size()==1);
    assert(a.peers[0].scatterIfSourceRuns==3);
  }
  {
    Plan p; p.sourceLength=1;p.targetLength=4;p.numSame=1;
    p.sourceLengthByRank={1,23};p.numRemotePeers=1;p.local={{0,0}};
    p.remote={{1,1,22,3},{1,1,0,1},{1,1,11,2}};
    const auto a=analyze(p,0,0,{0,1});
    assert(a.peers[0].windowElements==23 && a.peers[0].extraWindowElements==20);
    assert(a.peers[0].sourceTouchesLastOwned && a.sourceRuns==3);
    p.remote[0].sourceLid=23;
    expectThrow([&](){(void)analyze(p,0,0,{0,1});});
  }
  {
    Plan p; p.sourceLength=0;p.targetLength=2;p.sourceLengthByRank={0,4};
    p.remote={{1,1,3,0},{1,1,0,1}};p.numRemotePeers=1;
    const auto a=analyze(p,0,0,{0,1});
    assert(a.localPrefix==0 && a.requests==2 && a.remoteSuffix==2);
    assert(a.peers[0].sourceTouchesLastOwned && a.peers[0].windowElements==4);
  }
  {
    Plan p=multiPeer();
    p.remote[0].sourcePe=2;
    expectThrow([&](){(void)analyze(p,0,0,{0,1,2});});
    p=multiPeer();p.remote[0].destinationLid=1;
    expectThrow([&](){(void)analyze(p,0,0,{0,1,2});});
    p=multiPeer();p.exportLids[0]=5;
    expectThrow([&](){(void)analyze(p,0,0,{0,1,2});});
    p=multiPeer();p.remote.pop_back();
    expectThrow([&](){(void)analyze(p,0,0,{0,1,2});});
    p=multiPeer();
    expectThrow([&](){(void)analyze(p,0,0,{0,0,2});});
  }
  {
    // Deterministic randomized one-peer mappings: run accounting and
    // read-only permutation properties independent of actual Tpetra APIs.
    std::mt19937 engine(20261019);
    for(int repeat=0;repeat<400;++repeat) {
      const std::size_t owned=1+engine()%6;
      const std::size_t count=1+engine()%40;
      Plan p;p.sourceLength=owned;p.targetLength=owned+count;
      p.numSame=owned;p.sourceLengthByRank={owned,64};p.numRemotePeers=1;
      for(std::size_t i=0;i<owned;++i)p.local.push_back({i,i});
      for(std::size_t i=0;i<count;++i)
        p.remote.push_back({1,1,static_cast<std::uint64_t>(engine()%64),owned+i});
      std::shuffle(p.remote.begin(),p.remote.end(),engine);
      auto snapshot=p.remote;
      const auto a=analyze(p,0,0,{0,1});
      assert(a.sourcePeers==1 && a.requests==count);
      assert(a.peers[0].sourceRunCount<=a.peers[0].uniqueSources);
      assert(a.peers[0].uniqueSources<=count);
      assert(a.directLogicalGets<=a.requests && a.sourceRunLogicalGets<=a.requests);
      assert(a.peers[0].extraWindowElements+a.peers[0].uniqueSources ==
             a.peers[0].windowElements);
      for(std::size_t i=0;i<count;++i)
        assert(snapshot[i].sourceLid==p.remote[i].sourceLid &&
               snapshot[i].destinationLid==p.remote[i].destinationLid);
    }
  }
  std::cout << "P5.0 independent analyzer tests: PASS\n";
}
