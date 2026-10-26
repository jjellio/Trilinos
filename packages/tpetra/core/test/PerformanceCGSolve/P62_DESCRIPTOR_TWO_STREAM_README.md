# P6.2 — Barrier-delimited descriptor PUT / independent signal tests

**Status:** Implemented in the uploaded source snapshot; host-only plan unit tests pass. **CUDA compilation, NVSHMEM runtime progress, intra-node and two-node correctness, and performance are not verified here.** The implementation is deliberately a temporary barrier-synchronized experiment, **not** a drop-in asynchronous Tpetra `doImport()`.

## Experiment hypothesis

P6.1 issues one PUT-with-signal per destination PE, using one thread/warp/block collectively. P6.2 changes the unit of work to a **descriptor**, a chunk of up to `K` doubles for one peer. It tests whether a much larger number of independent GPU PUTs increases utilization enough to overcome per-operation overhead. The source and receiver agree on all descriptors and per-transfer signal slots in a one-time setup step.

For a peer with `N` doubles, `ceil(N/K)` PUTs and **the same number of distinct receive signals** are required. P6.2 doesn't use one signal per PE, aggregation counters, remote ACKs or credit waits.

## Import timeline

```text
Host:      finish previous GPU work → MPI_Barrier(MPI_COMM_WORLD)
Stream A:                                 pack+PUT+signal kernel (thread/warp/block)
Stream B:                                 nonblocking signal-test+scatter kernel
Host:      cudaDeviceSynchronize() → caller may start next iteration
```

The benchmark's matched Tpetra and NVSHMEM clocks **start after their respective pre-start barrier**. It additionally records P6.2 barrier durations separately. Thus the two methods share the P6 timing boundary `fence; MPI_Barrier; start; Import; fence; stop`. In real use the barrier is part of this experimental algorithm and cannot be omitted. Add the separately reported barrier cost if a full-iteration comparison is desired. The optional diagnostic section measures host submission time for each kernel plus the final device wait; those times are **not wire times** or a profiler proof of overlap.

Exactly **two CUDA kernel launches per PE per P6.2 Import** (even if there is no remote traffic):

1. **Producer**: local domain-to-column copies in separate CUDA blocks plus independent gather-to-symmetric-scratch, GPU PUT-with-signal work. It uses a thread, an entire 32-thread warp, or an entire 128-thread block per transfer, depending on `--nvshmem-p62-put-variant`. Warp/block participating threads share descriptor arguments and synchronize their cooperative gathers before calling the matching collective PUT API. Packets from the same PE use distinct source and destination ranges. No external pack kernel or intermediate fence is needed.
2. **Receiver**: a deliberately occupancy-bounded kernel (default **four 128-thread blocks**, automatically capped below the device's SM count) with one **logical signal test per chunk**, followed by a warp-cooperative inbox scatter to the correct column LIDs. It uses `nvshmem_uint64_test(..., NVSHMEM_CMP_EQ, generation)` until its particular PUT's delivery signal arrives; this is not a blocking `nvshmem_signal_wait_until`. It does **not** require signals to arrive in descriptor order.

The producer and receiver occupy separate `cudaStreamNonBlocking` streams, with no intermediate `cudaStreamSynchronize`, device barrier, or MPI synchronization between them. Only the final local `cudaDeviceSynchronize` joins them.

## Why the inter-iteration barrier is essential

`cudaDeviceSynchronize()` returns only after **this PE's** producer completed AND **this PE's** receiver observed every incoming chunk and finished all inbox-to-target reads. The next iteration's **MPI_Barrier** waits for that condition at all PEs before any PE starts another PUT. Therefore a subsequent generation cannot overwrite a remote inbox that is still being read from the previous generation. This deliberately replaces the previous P5.2.1 ACK/credit protocol for the experimental run.

Signal arrays are zero-initialized once, and every chunk gets one stable signal slot with a unique source writer. Successful PUTs set that slot to the monotonically increasing epoch. No per-iteration signal reset or atomic arrival counter is necessary. A zero-traffic PE still participates in the iteration's barrier.

**Never call `Exchange::importAfterBarrier()` directly without first establishing that ALL PEs have finished the prior generation and entered an MPI barrier together.** The method is intentionally named to make that contract explicit. This is *not* a valid fully asynchronous operation.

## One-time plan creation

- Build and finalize the P5.1 authoritative Importer packet/LID plan unchanged.
- Build the P6.2 plan once by sorting each sender's exports by **actual GID**, with a stable tie-breaker for duplicates, while the receiver sorts the corresponding source-peer destinations by the column Map's GID. This is not a sort by source-LID or a repeated sort in the hot loop. Repeated requested GIDs have identical source values and can appear at several distinct destination LIDs.
- Partition each peer segment into `K`-double descriptors; the final chunk is shorter, and no padding descriptors are transmitted.
- Once per setup, `MPI_Alltoall` exchanges the exact number of sends to be received, checks sorted-GID fingerprints, and advertises the receiver's contiguous signal-slot base to every sender. `MPI_Allreduce(MAX)` sizes the symmetric ready array identically on all PEs.
- Allocate symmetric pack, inbox, and ready arrays once. Upload descriptor, ordered source-LID, destination-LID and local copy arrays once. No per-Import MPI payload, allocation, sort or metadata exchange takes place. The original `Exchange::import`, `importWarp`, and `importThreadBlock` remain untouched.

Source storage remains the ordinary Tpetra CUDA-space Vector and target storage is an ordinary Tpetra CUDA-space Vector. Symmetric memory is a communication scratch/inbox **proxy**, not a new `Tpetra::MultiVector` allocation.

## GPU progress caveat — must validate before performance claims

NVIDIA warns that waiting kernels can starve their producer and that synchronizing NVSHMEM CUDA kernels may require its collective-launch mechanism. P6.2 does **not** introduce a collective launch: it bounds the receiver to at most a few blocks and repeatedly calls the documented nonblocking `nvshmem_uint64_test`. This is an *experimental progress strategy*, not a proof that CUDA can always co-schedule the two kernels or that the configuration is deadlock-free on every NVSHMEM release/transport. **Run the two-node progress and correctness checks under an MPI/Slurm job timeout.** A hang/failure must not be construed as successful overlap. Nsight Systems is needed to verify actual concurrency. The transport with IBGDA CPU NIC handler must be explicitly checked in the NVSHMEM startup log on Cronus.

## Host-only test and new command-line mode

The host plan test needs no NVSHMEM, CUDA, MPI, or Trilinos installation:

```bash
cd packages/tpetra/core/test/PerformanceCGSolve
g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
    test_packed_put_p62_plan.cpp -o /tmp/p62_plan_test
/tmp/p62_plan_test
```

In an existing Trilinos CUDA/NVSHMEM build (`PERFORMANCE_CGSOLVE_NVSHMEM=ON`), compile the `TpetraCore_Performance-CGSolve.exe` target as before. P6.2 has a completely separate option: `--nvshmem-p62`. P6.1 still uses `--nvshmem-p6 --nvshmem-p6-put-variant=...`.

**Single GPU wiring smoke, no remote traffic:**

```bash
mpirun -np 1 "$HOME/cronus/gpu_bind.sh" \
  ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe \
  --cuda --numgpus=1 --size=8 --nvshmem-p62 \
  --nvshmem-p62-case=all --nvshmem-p62-chunk=128 \
  --nvshmem-p62-put-variant=warp --nvshmem-p62-verify-every=1 \
  --nvshmem-p62-warmup=2 --nvshmem-p62-reps=5
```

**Two physical nodes, one PE/GPU each, including an actual cross-node path:**

```bash
mpirun --map-by ppr:1:node --bind-to none -np 2 \
  "$HOME/cronus/gpu_bind.sh" \
  ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe \
  --cuda --numgpus=1 --size=32 --nvshmem-p62 \
  --nvshmem-p62-case=all --nvshmem-p62-chunk=128 \
  --nvshmem-p62-put-variant=warp --nvshmem-p62-verify-every=1 \
  --nvshmem-p62-warmup=3 --nvshmem-p62-reps=10 \
  --nvshmem-p62-min-nodes=2 --nvshmem-p62-max-nodes=2 \
  --nvshmem-p62-cross-node
```

Or run the controlled sweep inside an existing allocation:

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
P62_REPS=50 P62_WARMUP=8 P62_PASSES=2 \
  P62_CHUNKS='1 128 512 2048' \
  ./packages/tpetra/core/test/PerformanceCGSolve/run_p62_sweep.sh \
  inter8 100 direct
```

Other settings: `--nvshmem-p62-mode=direct|diagnostics|both`, `--nvshmem-p62-receiver-blocks=1..16`, `--nvshmem-p62-diagnostic-reps`, `--nvshmem-p62-out=PREFIX`, `--nvshmem-p62-case=matrix|all|...`, `--nvshmem-p62-min-nodes`, `--nvshmem-p62-max-nodes`. Use `--nvshmem-p62-verify-every=1` for correctness/stress runs; it validates every epoch **after the timer**, then re-establishes CUDA destination residency before the next sample. Benchmark sweeps default to final-epoch validation instead.

Raw `PREFIX_CASE_p62_samples.csv` contains every rank's timings. Summary lines use max-rank-per-iteration medians/p10/p90 and paired ratios. Compare **end-to-end completed Import** with stock Tpetra; separately run the unchanged P6.1 thread/warp/block controls. Do not compare isolated PUT issue time to full `doImport`.

## Acceptance matrix for actual cluster runs

1. Compile with the installed NVSHMEM headers/device library and the existing CUDA/RDC flags; record the precise NVSHMEM version.
2. Run the single-GPU smoke and the host plan test.
3. Cross-node 2-PE, intra-node 2/4-PE, and mixed 8-PE 2-node maps. Confirm hostnames. Run `--nvshmem-p62-case=all`, including zero-owner, all-empty, one-way, fanout, uneven and permute cases; check every epoch with multiple chunk sizes `1, 128, 512, 2048` in each thread/warp/block mode.
4. Repeat with many changing epochs and multiple symmetric allocation cycles by launching separate runs. Test generation recycling and deliberately skewed peers under job timeout. Do NOT remove the MPI barrier when stressing buffer reuse.
5. Only after correctness and progress pass, use `--size=100` matched completed-Import measurements and an Nsight Systems GPU timeline. Record the actual NIC handler and test IBRC versus IBGDA CPU handler separately. Show whether producer and receiver really overlap.

## Source files

- `PackedPutP62Plan.hpp`: pure-host sorted-GID chunk descriptors / signal mapping.
- `PackedPutP62PlanMpi.hpp`: one-time MPI descriptor-count, fingerprint and signal offset reconciliation.
- `NvshmemPackedPutP62.hpp`: standalone CUDA producer/receiver and persistent Exchange, **no modifications to P6.1 code**.
- `NvshmemP62.hpp`: end-to-end correctness, Tpetra comparison, timings/CSV and all existing cases.
- `test_packed_put_p62_plan.cpp`: host-only 3-rank simulation (nonidentity rank/PE, repeated GIDs, one-way, four chunk sizes and remote PUT/receiver slot matching).
- `run_p62_sweep.sh`: reproducible transport/mode/chunk sweep script.
- `cg_solve_file.cpp`, `CMakeLists.txt`: new opt-in entry point and opt-in single-GPU CTest smoke.

Do not report the design as native CUDA validated until the cluster's NVCC build and GPU/InfiniBand run succeed.
