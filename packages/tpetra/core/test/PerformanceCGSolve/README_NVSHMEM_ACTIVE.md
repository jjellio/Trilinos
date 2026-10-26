# Active NVSHMEM Import experiment — post-archive working tree

The active executable modes are **ordinary Tpetra/CG** and `--nvshmem-p62`.
The historical P0, P1, P2, P3, P4, P5.0, P5.1, P5.2, P6 and P6.1 *standalone CLI modes* are archived in the recovery Git tag made by `git_rm_legacy_nvshmem.sh`. They are **not** enabled here. Original code is recoverable through Git, not by guessing a new executable option.

## Reproducibility / files intentionally retained

- `NvshmemP62.hpp`, `NvshmemPackedPutP62.hpp`, `PackedPutP62Plan.hpp`, `PackedPutP62PlanMpi.hpp`: **unchanged** P6.2 reference; two-stream, per-descriptor `put_signal`, experimental pre-Import MPI barrier, matched Tpetra benchmark.
- `PackedPutPlan.hpp`, `PackedPutPlanMpi.hpp`, `ImportGatherPlan.hpp`: authoritative Importer source/destination/LID metadata used by P6.2. `test_packed_put_plan.cpp` and `test_packed_put_p62_plan.cpp` validate host planning.
- `NvshmemP0.hpp`, `NvshmemP1.hpp`, `NvshmemP2.hpp`, `NvshmemP3.hpp`, `NvshmemP4.hpp`, `NvshmemP6.hpp`, `NvshmemPackedPut.hpp`, `NvshmemP6Stats.hpp`, `StandaloneMpiImport.hpp`: **still needed transitively by P6.2**, despite their historical names. Removing these would break compilation. They will be split into stable utilities only in a later separate, tested change.
- `P62_DESCRIPTOR_TWO_STREAM_README.md`, `run_p62_sweep.sh`, `test_nvshmem_p62_layout.py`, `test_nvshmem_p6_stats.cpp`: current benchmark method, repeatable sweep and controls.
- `P7_COMPLETION_PROTOCOL_WORKPLAN.md`: workplan to decouple data transfer from completion, with correctness and go/no-go gates.

## Building and smoke tests

The original TriBITS target `Performance-CGSolve` and its ordinary strong-scaling CTest registrations remain. The existing project default `PERFORMANCE_CGSOLVE_NVSHMEM=ON` remains unchanged; to build stock Tpetra without NVSHMEM, configure with `-DPERFORMANCE_CGSOLVE_NVSHMEM=OFF`. The only NVSHMEM CTest option still exposed is `PERFORMANCE_CGSOLVE_NVSHMEM_P62_CTEST=ON`, which requires the NVSHMEM build switch. Hardware-dependent testing remains opt-in.

After rebuilding the usual Trilinos executable, run the same P6.2 smoke test as before:

```bash
mpirun -np 1 "$HOME/cronus/gpu_bind.sh" \
  ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe \
  --cuda --numgpus=1 --size=8 --nvshmem-p62 --nvshmem-p62-case=all \
  --nvshmem-p62-chunk=128 --nvshmem-p62-put-variant=warp \
  --nvshmem-p62-verify-every=1 --nvshmem-p62-warmup=2 --nvshmem-p62-reps=5
```

For two-node correctness use the existing `--nvshmem-p62-min-nodes=2 --nvshmem-p62-cross-node` controls. The P6.2 MPI barrier **is part of the experimental correctness protocol**, even though the existing completed-Import clock excludes it; do not treat the barrier-excluded time as a full production Import cost.

Host-only regression checks:

```bash
cd packages/tpetra/core/test/PerformanceCGSolve
g++ -std=c++17 -O2 test_packed_put_plan.cpp -o /tmp/packed_plan_test && /tmp/packed_plan_test
g++ -std=c++17 -O2 test_packed_put_p62_plan.cpp -o /tmp/p62_plan_test && /tmp/p62_plan_test
g++ -std=c++17 -O2 test_nvshmem_p6_stats.cpp -o /tmp/p6_stats_test && /tmp/p6_stats_test
python3 test_nvshmem_p62_layout.py
```

**P7 alternatives are only planned, not implemented.** No production solver integration or performance claim is made by the cleanup.

## P7.1 instrumentation (opt-in; current source)

`--nvshmem-p62-mode=p71` adds a **separate** sampled GPU diagnostic section *after* an uninstrumented, alternating MPI/P6.2 comparison. The original P6.2 CUDA kernel bodies, packet plan and PUT-with-signal protocol are unchanged. The new mode measures per-stream CUDA event durations, sampled `clock64` gather/PUT-call-return/ready-poll/scatter intervals and correctly paired barrier-plus-Import totals. It writes one rank-local probe CSV per rank and a rank0 combined timing CSV; `--nvshmem-p62-out=PREFIX` is required.

See `P71_COMPLETION_COST_README.md` for the exact CLI/sweep, analysis methodology, transport controls and current limitations. Host-only validation adds `test_nvshmem_p71_stats.cpp` and `test_nvshmem_p71_layout.py`; `PERFORMANCE_CGSOLVE_NVSHMEM_P71_CTEST=ON` optionally registers a single-PE smoke. This is **instrumentation only**: no NBI, quiet/fence, or aggregate-completion variant exists yet, and no GPU/runtime validation is claimed without cluster runs.

## P7.1b matrix-free topology benchmark

The **separate executable** `TpetraCore_ImportTopologyBenchmark.exe` builds balanced,
parameterized source/target Maps and a stock Tpetra Importer **without a matrix**.
It varies off-node/intra-node peer degree, per-peer versus total packet volume,
contiguous/strided/permuted source GIDs, and target local/remote interleaving.
Actual Importer peer relationships are checked against the graph before any
benchmarking. It uses the **same frozen P6.2 / P7.1 GPU kernels and matched
completed-Import comparison**, not a new or faster completion protocol.
See `P71B_MATRIX_FREE_IMPORT_BENCHMARK.md` and `run_import_topology_sweep.sh`.
The existing `Performance-CGSolve.exe` entry points remain unchanged.
