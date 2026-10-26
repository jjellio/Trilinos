# P7.1 — Measure the current PUT-with-signal completion costs

**Status (October 25, 2026):** Source implementation complete; local host tests and host-stub kernel-instantiation checks pass. **CUDA NVCC compilation, NVSHMEM runtime execution, transport-progress correctness, and Nsight traces remain to be done on Cronus.** No P7.2 or P7.3 transfer protocol is implemented by these files.

## Goal and frozen reference

P7.1 diagnoses B0 (the P6.2 per-descriptor PUT-with-signal protocol) before considering `put_nbi`, `quiet`, or fewer completion signals. The **original P6.2 `packAndPutKernel`, `waitAndUnpackKernel`, and `importAfterBarrier` methods are unchanged**. In particular, the active baseline still performs exactly **one PUT-with-signal and one receiver-side ready test per descriptor**, two CUDA kernels on independent streams and the mandatory start-of-epoch MPI barrier.

`--nvshmem-p62-mode=p71` runs two separate sections per case:

1. **Uninstrumented paired DIRECT:** alternating `Tpetra::doImport()` and unchanged P6.2 B0. Every operation receives a pre-start barrier. Report completed Import without the barrier, separate barrier time, and an independent **barrier+Import** sample computed *per rank/per iteration before the max-rank reduction*.
2. **Instrumented P7.1 PROFILE:** **cloned** producer and receiver kernels with sampled GPU clock readings, per-stream CUDA events, and post-completion device-to-host probe retrieval. These clones use **the same PUT-with-signal, signal predicate, descriptor mappings, and synchronization** as B0, but timing instructions perturb scheduling; **never infer uninstrumented speedups from this section**. No per-descriptor host fences or temporary allocations are introduced.

Both sections validate against the same Tpetra reference values (warmup and final always, optionally every iteration). Source data are regenerated for each iteration as in P6.2.

## Instrumentation fields and limitations

| Stage/field | What is measured | What it does **not** prove |
|---|---|---|
| `pack_and_put_api_return / first_us` | Same-thread CUDA SM cycles around descriptor gather; for warp/block, includes the local pack barrier | Standalone NIC packing overhead or global pack kernel time |
| `pack_and_put_api_return / second_us` | CUDA SM cycles around the `nvshmem_*_put_signal` call, until return to issuing thread/lane | **Remote data arrival** or separate signal cost |
| `signal_poll_and_scatter / first_us` | Time from first receiver test until `nvshmem_uint64_test` reports the generation; `poll_attempts` counts only sampled descriptor tests | Pure NIC/network latency; includes inter-stream scheduling and producer start delays |
| `signal_poll_and_scatter / second_us` | Same-thread cycles around warp scatter, including its final `__syncwarp` | Isolated memcpy throughput or per-transfer remote latency |
| `local_copy / first_us` | A sampled scalar local copy within fused producer | Full local-copy kernel duration |
| `producer_cuda_kernel_event` | Kernel duration on producer stream, including all issue/work | Per-transfer completion or isolated network time |
| `receiver_cuda_kernel_event` | Kernel duration on listener stream, including wait and scatter | Independent critical-path component: streams can overlap |
| `cudaDeviceSynchronize_host_wait` | CPU blocking time after both kernel submissions | GPU network transfer time alone |
| `post_import_probe_readback` | Host D2H transfer of sampled instrumentation, **after completed Import** | Part of the uninstrumented Import or timing boundary |
| `P71_FULL` | Correct per-rank (barrier + finished Import), reduced as max-rank per iteration | Deployable asynchronous Import cost (P6.2 still needs barrier) |

The application converts `clock64` cycles to approximate microseconds using `cudaDeviceProp.clockRate` (kHz). GPU clocks may change dynamically. **Never subtract timestamps from different SMs or add packing/PUT/wait/scatter medians as if they form one serialized critical path.** NVSHMEM `put_signal` call-return need not mean that the receiver observed the data. CUDA kernel event durations on different streams cannot establish actual overlap; capture an Nsight Systems GPU timeline to verify simultaneous execution and stalls.

No `quiet`, `fence`, `put_nbi`, ACK, or extra completion signal is added. For B0, the counters for NBI, quiet and fence are correctly **zero**. Isolating the intrinsic *signal cost* requires the later controlled B2 experiment with the same packet schedule.

## CLI / build

The active executable remains `TpetraCore_Performance-CGSolve.exe`. **No second NVSHMEM command mode is introduced**; invoke the opt-in profile mode through the existing P6.2 option:

```bash
--nvshmem-p62 --nvshmem-p62-mode=p71
--nvshmem-p62-p71-reps=12
--nvshmem-p62-p71-max-samples=64
--nvshmem-p62-out=PREFIX
```

`--nvshmem-p62-out` is **required in p71 mode**. One rank-0 all-rank timing CSV (`PREFIX_matrix_p62_samples.csv`) and one **rank-local** sampled probe CSV (`PREFIX_matrix_p71_rankR_probes.csv` for each R) are created. The rank-local files are written only **after** the profiling loop, not inside measured GPU work.

The P7.1 sample capacity can be set to 1..4096 per PE and per category (send/receive/local); it deterministically selects descriptor indices at a stride `ceil(n/cap)` in each category. This limits GPU probe overhead but is **not a random representative sample**. Every descriptor is still transmitted and waited upon; only the timing instructions are sampled. GPU memory and CUDA events are allocated in the Exchange constructor, once, only for mode=p71. The B0 default continues to allocate **none** of these profiling resources.

The build/CMake interface is unchanged apart from adding optional `PERFORMANCE_CGSOLVE_NVSHMEM_P71_CTEST=ON`. The test runs a 1-PE/zero-network instrumentation smoke and **does not validate InfiniBand completion**.

### First one-PE smoke (build directory)

```bash
mpirun -np 1 "$HOME/cronus/gpu_bind.sh" \
  ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe \
  --cuda --numgpus=1 --size=8 --nvshmem-p62 --nvshmem-p62-case=all \
  --nvshmem-p62-mode=p71 --nvshmem-p62-put-variant=warp \
  --nvshmem-p62-chunk=128 --nvshmem-p62-verify-every=1 \
  --nvshmem-p62-warmup=2 --nvshmem-p62-reps=5 \
  --nvshmem-p62-p71-reps=2 --nvshmem-p62-p71-max-samples=8 \
  --nvshmem-p62-out=p71-smoke
```

### 2 physical nodes, one PE per node

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=0
mpirun --map-by ppr:1:node --bind-to none -np 2 \
  "$HOME/cronus/gpu_bind.sh" \
  ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe \
  --cuda --numgpus=1 --size=100 --nvshmem-p62 --nvshmem-p62-case=matrix \
  --nvshmem-p62-mode=p71 --nvshmem-p62-put-variant=block \
  --nvshmem-p62-chunk=1024 --nvshmem-p62-verify-every=1 \
  --nvshmem-p62-warmup=5 --nvshmem-p62-reps=30 \
  --nvshmem-p62-p71-reps=12 --nvshmem-p62-p71-max-samples=64 \
  --nvshmem-p62-min-nodes=2 --nvshmem-p62-max-nodes=2 \
  --nvshmem-p62-cross-node --nvshmem-p62-out=p71-inter2-block1024
```

Run with a scheduler/job timeout: the original P6.2 bounded receive polling remains an experimental CUDA progress strategy.

## Recommended controlled P7.1 sweep

From the **build directory**, put `run_p71_completion_profile.sh` and `analyze_p71_profiles.py` together under the source tree as supplied, then:

```bash
# 2 physical nodes, one GPU/PE per node
P71_EXE=./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe \
P71_BIND="$HOME/cronus/gpu_bind.sh" \
  /path/to/Trilinos/packages/tpetra/core/test/PerformanceCGSolve/run_p71_completion_profile.sh inter2 100

# 2 physical nodes, 4 GPUs/PEs per node
P71_BIND="$HOME/cronus/gpu_bind.sh" \
  /path/to/Trilinos/packages/tpetra/core/test/PerformanceCGSolve/run_p71_completion_profile.sh inter8 100
```

Default sweep: `{warp,block} × {512,1024,2048} doubles × {IBRC, IBGDA with **CPU NIC handler**} × 3 independent MPI launches` = **36 jobs per topology**. Alternate launch order across passes; keep batch and RC tuning variables unset. `P71_DRY_RUN=1` plans the entire sweep without MPI/GPU. The script checks actual hostnames for node count, records executable/binder hashes, captures logs/environment and aborts on missing PASS markers or timeout. Re-run independently at the desired topology; do not pool inter2 and inter8 results.

For the IBGDA/CPU profile, the script sets `NVSHMEM_REMOTE_TRANSPORT=ibrc`, `NVSHMEM_IB_ENABLE_IBGDA=1`, and `NVSHMEM_IBGDA_NIC_HANDLER=cpu`; IBRC sets `NVSHMEM_IB_ENABLE_IBGDA=0` with NIC handler unset. Select `P71_VERBOSE_TRANSPORT=1` on a startup-validation pass to inspect actual transport and handler, then turn it off during timing. Do **not** claim GPU NIC doorbell handling on Cronus; documented attempted GPU NIC-mapping failed on this system.

For the complete suite of synthetic Maps, run the executable manually with `--nvshmem-p62-case=all --nvshmem-p62-mode=p71` and `--verify-every=1`. The sweep analyzer intentionally restricts itself to the `matrix` case.

After the sweep, the script invokes its companion analyzer, or run:

```bash
python3 /path/to/Trilinos/packages/tpetra/core/test/PerformanceCGSolve/analyze_p71_profiles.py \
  ./p71-profiles-DATE
```

Output: `runs.csv`, `summary.csv` (one row per configuration, independent-launch medians), `probe_distributions.csv` (sampled packet distributions by intra/inter-node), and `analysis.md`. It validates rank CSVs against the log medians, rejects missing rank-local probes, and does not treat 30 iterations or many sampled descriptors as independent MPI launches.

## Nsight GPU timeline

Use an otherwise identical P7.1 job with `nsys profile --trace=cuda,nvtx,osrt` inserted **after GPU binding and before the executable** (or configure your site's supported `nsys` wrapper). Nsight Systems should show one producer and one receiver kernel in the baseline direct section and the two separately named `packAndPutProbed` and `waitAndUnpackProbed` kernels in the profiled section. Confirm their actual residency/overlap, submission delays, polling occupancy, remote-transfer stalls, and whether each PE's waiting kernel blocks producer work. A CUDA stream distinction alone is **not proof of overlap**.

## Completion gate before P7.2/P7.3

1. Host stat/layout tests pass, native NVCC compile/link succeeds, two-node correctness passes with `--verify-every=1` for all test Maps.
2. At least 3 independent runs per (topology, IBGDA state, variant, K), under a job timeout. Confirm runtime NIC handler and no unexpected fallback.
3. Report **unchanged B0 vs MPI** completed Import plus B0 barrier-inclusive cost, and P7.1 GPU diagnostic phases separately. Show raw per-rank CSVs and rank maxima.
4. Obtain Nsight timeline(s) on inter2 and inter8 that actually explain where time is spent (pack/PUT/receiver poll/scatter/kernel scheduling/barrier). Do not conflate API return with remote completion.
5. Only then change **one** receiver or completion mechanism for P7.2 or P7.3.

**Files added:** `NvshmemP71Stats.hpp`, `NvshmemP71Probes.hpp`, `test_nvshmem_p71_stats.cpp`, `test_nvshmem_p71_layout.py`, `run_p71_completion_profile.sh`, `analyze_p71_profiles.py`, `P71_COMPLETION_COST_README.md`. **Files modified:** `NvshmemPackedPutP62.hpp` (optional profiled method and allocations; original kernels untouched), `NvshmemP62.hpp` (mode and report), `cg_solve_file.cpp` (two options), `CMakeLists.txt` (optional smoke), `README_NVSHMEM_ACTIVE.md`, `test_nvshmem_p62_layout.py` (scope its launch count to the unmodified B0 body), and file manifest. Baseline P6.2 algorithm remains B0.
