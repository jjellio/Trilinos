#!/usr/bin/env python3
"""P7.1 static invariants. Does NOT replace a CUDA/NVSHMEM compile or GPU run."""
from pathlib import Path
import hashlib
import re

here=Path(__file__).resolve().parent
b0=(here/'NvshmemPackedPutP62.hpp').read_text()
probes=(here/'NvshmemP71Probes.hpp').read_text()
harness=(here/'NvshmemP62.hpp').read_text()
cli=(here/'cg_solve_file.cpp').read_text()
body=b0.split('template<int Mode, class XView, class YView>')[1].split('struct HostTimings')[0]
# Fingerprint of the ORIGINAL uploaded P7.0 P6.2 producer/receiver functions.
original='ad7c63ad65f1db9aa3bc16aa7b2b4ae93c3d9ee5d9f14f8069e0faa53177dc87'
checks={
 'unchanged_B0_gpu_producer_and_receiver': hashlib.sha256(body.encode()).hexdigest()==original,
 'B0_exactly_two_launches': b0.count('<<<')==4,  # 2 original + 2 P7.1 clones
 'profiling_disabled_by_default': 'std::size_t p71SampleCap = 0' in b0,
 'two_P71_launches': b0.count('<<<')==4 and 'importProfiledAfterBarrier' in b0,
 'warp_and_block_PUT_unchanged_operation': all(x in probes for x in ('nvshmem_double_put_signal(', 'nvshmemx_double_put_signal_warp(', 'nvshmemx_double_put_signal_block(')),
 'P71_same_signal_test': 'nvshmem_uint64_test(ready + d.signalIndex' in probes,
 'P71_no_quiet_or_NBI': all(x not in probes for x in ('nvshmem_quiet(', 'nvshmemx_quiet_on_stream(', 'nvshmem_putmem_nbi(')),
 'sampled_gpu_copy_only_after_completion': 'cudaDeviceSynchronize(), "P7.1 two-stream device completion"' in b0 and 'P7.1 send probe readback' in b0,
 'no_intermediate_device_sync': b0.split('void importProfiledAfterBarrier(')[1].split('void release()')[0].count('cudaDeviceSynchronize()')==1,
 'barrier_rank_paired': '(tpetra ? localTFull : localPFull)' in harness and 'duration + barrierUs' in harness,
 'P71_control_is_uninstrumented': 'if (tpetra) Ytpetra.doImport(' in harness and 'else exchange.importAfterBarrier(X, Yput);' in harness,
 'P71_requires_output_prefix': 'P7.1 profiling requires --nvshmem-p62-out' in harness,
 'P71_cli_exposed': all(x in cli for x in ('nvshmem-p62-p71-reps','nvshmem-p62-p71-max-samples')),
}
for name,ok in checks.items():
 print(f'[NVSHMEM P7.1 STATIC] {name}={"PASS" if ok else "FAIL"}')
if not all(checks.values()):
 raise SystemExit(1)
