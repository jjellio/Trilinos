#!/usr/bin/env python3
"""CPU-only static P6.2 source sanity checks; NOT CUDA or GPU correctness."""
from pathlib import Path

p = Path(__file__).with_name("NvshmemPackedPutP62.hpp")
src = p.read_text()
checks = {
    "two_native_kernel_invocations": src.count("<<<") == 2,
    "native_fused_gather_put": "packAndPutKernel" in src and "nvshmem_double_put_signal(" in src,
    "warp_collective": "nvshmemx_double_put_signal_warp(" in src and "__syncwarp()" in src,
    "block_collective": "nvshmemx_double_put_signal_block(" in src and "__syncthreads()" in src,
    "per_descriptor_test": "nvshmem_uint64_test(" in src,
    "two_independent_streams": src.count("cudaStreamCreateWithFlags(") == 2,
    "bounded_receiver": "std::max(1, smCount - 1)" in src,
    "local_terminal_join": "cudaDeviceSynchronize()" in src,
    "no_ack_kernel": "acknowledgeConsumedKernel<<<" not in src,
    "no_mpi_inside_exchange": "MPI_Barrier(" not in src.split("void importAfterBarrier(")[1].split("void release()")[0],
}
for name, ok in checks.items():
    print(f"[NVSHMEM P6.2 STATIC] {name}={'PASS' if ok else 'FAIL'}")
if not all(checks.values()):
    raise SystemExit(1)
