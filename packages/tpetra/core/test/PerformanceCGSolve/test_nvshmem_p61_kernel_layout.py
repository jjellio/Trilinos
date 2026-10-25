#!/usr/bin/env python3
"""Compile/execute the actual P6.1 CUDA kernel bodies as sequential C++ mocks.

This validates that warp/block group participation and all per-peer arguments
remain uniform, with exactly 32/128 participants per packet. It is NOT NVCC,
GPU-memory-order, signal delivery or transport validation.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys

source = (Path(__file__).parent / "NvshmemPackedPut.hpp").read_text()
first = source.index("static __global__ void packedPutSignalWarpKernel(")
last = source.index("// One remote writer per credit slot", first)
kernels = source[first:last]
assert kernels.count("nvshmemx_double_put_signal_warp(") == 1
assert kernels.count("nvshmemx_double_put_signal_block(") == 1
cpp = r'''
#include "PackedPutPlan.hpp"
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <vector>
#include <map>
#include <stdexcept>
#define __global__
#define NVSHMEM_SIGNAL_SET 1
using U64 = PackedPutP51::U64;
struct Dim {std::size_t x=0;};
Dim blockIdx, threadIdx, blockDim, gridDim;
struct Call {
  std::size_t src,dst,slot,count,block,lane;
  U64 generation;
  int pe;
};
std::vector<Call> calls;
std::vector<double> source(30000), inbox(30000);
std::vector<U64> ready(128);
void note(double* dst,const double* src,std::size_t count,U64* sig,U64 generation,int op,int pe) {
  assert(op==NVSHMEM_SIGNAL_SET);
  calls.push_back({static_cast<std::size_t>(src-source.data()),
                   static_cast<std::size_t>(dst-inbox.data()),
                   static_cast<std::size_t>(sig-ready.data()),count,
                   blockIdx.x,threadIdx.x,generation,pe});
}
void nvshmemx_double_put_signal_warp(double* d,const double* s,
    std::size_t n,U64* sig,U64 gen,int op,int pe) {
  note(d,s,n,sig,gen,op,pe);
}
void nvshmemx_double_put_signal_block(double* d,const double* s,
    std::size_t n,U64* sig,U64 gen,int op,int pe) {
  note(d,s,n,sig,gen,op,pe);
}
''' + kernels + r'''
void test(std::size_t peers,bool warp) {
  std::vector<PackedPutP51::SendPeer> entries(peers);
  for(std::size_t p=0;p<peers;++p) {
    auto& e=entries[p];
    e.rank=static_cast<int>(p+1);e.pe=static_cast<int>(p+1);
    e.packOffset=p*64;e.remoteInboxOffset=p*64;
    e.remoteSignalIndex=p%17;e.count=1+(p*17)%63;
  }
  calls.clear();blockDim.x=128;
  gridDim.x=warp ? (peers+3)/4 : peers;
  for(blockIdx.x=0;blockIdx.x<gridDim.x;++blockIdx.x)
    for(threadIdx.x=0;threadIdx.x<blockDim.x;++threadIdx.x) {
      if(warp)packedPutSignalWarpKernel(entries.data(),peers,
          inbox.data(),source.data(),ready.data(),34);
      else packedPutSignalBlockKernel(entries.data(),peers,
          inbox.data(),source.data(),ready.data(),34);
    }
  const std::size_t participants=warp?32:128;
  assert(calls.size()==peers*participants);
  std::vector<std::size_t> counts(peers,0);
  std::vector<std::size_t> first(peers,calls.size());
  for(std::size_t idx=0;idx<calls.size();++idx) {
    const auto& c=calls[idx];
    assert(c.pe>=1 && static_cast<std::size_t>(c.pe)<=peers);
    const std::size_t peer=static_cast<std::size_t>(c.pe)-1;
    const auto& e=entries[peer];
    assert(c.count==e.count && c.src==e.packOffset &&
           c.dst==e.remoteInboxOffset && c.slot==e.remoteSignalIndex &&
           c.generation==34);
    const std::size_t expectedBlock=warp?peer/4:peer;
    assert(c.block==expectedBlock);
    if(warp)assert(c.lane/32==peer%4);
    if(first[peer]==calls.size())first[peer]=idx;
    else {
      const auto& f=calls[first[peer]];
      assert(f.src==c.src && f.dst==c.dst && f.slot==c.slot &&
             f.count==c.count && f.pe==c.pe && f.generation==c.generation);
    }
    counts[peer]++;
  }
  for(auto count:counts)assert(count==participants);
}
int main() {
  for(std::size_t n : std::vector<std::size_t>{0,1,2,3,4,5,7,8,15,16,33,64,127,255}) {
    test(n,true);test(n,false);
  }
  std::cout << "P6.1 real kernel source mapping mock: PASS, 14 peer degrees x warp/block" << std::endl;
}
'''
with tempfile.TemporaryDirectory(prefix="p61_geometry_") as tmp:
    cpp_path = Path(tmp) / "test_kernel_layout.cpp"
    cpp_path.write_text(cpp)
    compilers = [x for x in ("g++", "clang++") if shutil.which(x)]
    if not compilers:
        raise RuntimeError("no host C++ compiler available")
    for compiler in compilers:
        exe = Path(tmp) / (compiler.replace("+", "p") + "_layout")
        subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra",
                        "-Werror", "-pedantic", "-I", str(Path(__file__).parent),
                        str(cpp_path), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
print("This is a group-participation/descriptor test, NOT native NVSHMEM correctness.")
