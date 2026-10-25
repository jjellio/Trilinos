# P6: stock Tpetra `doImport` versus NVSHMEM sender-packed PUT

**Status: benchmark SOURCE IMPLEMENTED and host-tested; GPU-cluster timing
results PENDING.** This is a performance study, not a correctness
replacement for the accepted P5.2.1 16-rank NVSHMEM-only stress suite.

P6 implements the two independent measurements requested:

1. **Direct end-to-end comparison:** `Kokkos::fence(); MPI_Barrier();
   start wall timer; X_column.doImport(X, importer, Tpetra::INSERT)` **or**
   `exchange.import(X, Y_put); Kokkos::fence(); stop wall timer`.
   The barrier and the preceding fence are outside the timed region. The
   post-Import fence is inside. Each PE stores its own duration. Rank zero
   reports per-repetition `max_rank` elapsed times, including median, p10,
   p90, min, max, plus a Tpetra / NVSHMEM median ratio. Both methods process
   the same newly populated ordinary Tpetra source X. Order alternates by
   repetition to mitigate second-run cache/clock drift bias.
2. **Separately instrumented NVSHMEM exchange:** the *identical*
   `NvshmemPackedPut::Exchange::import` accepts an optional POD timing
   pointer (null in the direct comparison). Its seven host-wall-clock stage
   durations enclose the **original** kernel launches, CUDA/Kokkos fences
   and NVSHMEMx signal waits. There is NO MPI or barrier between the
   instrumented Imports, so per-peer credit waits and natural skew remain
   observable. The stage samples are stored per rank, gathered only after
   all repetitions, and summarized as per-iteration rank maxima.

## Allocation and source-data scope

For **each Map/case** the harness constructs these objects *once*, before
warmup and both measured experiments:

- domain and column Maps, Tpetra Importer, P1/P5.1 packet/scatter metadata;
- one ordinary-CudaSpace Tpetra X and two ordinary-CudaSpace destination
  vectors, `Y_tpetra` and `Y_nvshmem`;
- one persistent `NvshmemPackedPut::Exchange`, including all symmetric
  pack/inbox/ready/credit allocations, Kokkos descriptors, and remote offsets;
- fixed-size vectors for all local direct and stage timing samples.

Every warmed-up and measured operation reuses exactly these objects.
There is **no explicit allocation, symmetric free, file I/O, MPI reduction,
Map construction or packet-plan setup in either timed region**. Symmetric
free is collective **after** the reporting, not per Import.

`NvshmemP4::fillOwned` updates **the same X** with new, signed, exactly
representable double values before each direct sample pair and each
instrumented repetition, and fences local CUDA completion. This source
update is *outside* the stopwatch for BOTH implementations. We are timing
`Import(X already GPU-ready)`, not time to compute X. Packing **is** inside
NVSHMEM Import, and stock Tpetra's Import includes its own packing.

Warmup exercises Tpetra's internal Distributor and per-object caches, plus
NVSHMEM scratch state, before recording samples. A second untimed Import
restores GPU destination view residency after initial correctness checking.
**Important limitation:** the benchmark can guarantee no harness or
NVSHMEM allocation inside a timed Import, but the external stock Tpetra
`doImport` implementation may manage temporary storage internally. It is
not possible to certify zero internal allocation from this harness alone;
for a strict allocation-free attribution, trace `cudaMalloc`/`cudaFree` and
host allocators with Nsight Systems or an allocator interposer on the target
Trilinos build. If Tpetra allocates on every call, its measured value still
includes that cost; do not label it an allocation-free comparison.

Correctness is checked against the independent owner-rank/owner-LID encoding
before and after each measured experiment. Validation is OUTSIDE the timers.
P5.2.1 remains the required every-epoch concurrency correctness test.

## Run with your established two-node/eight-PE-per-node binding

Build from the latest P5.2.1 checkout (`nvshmem-pack-latest-p521-tar6.xz`)
with the accompanying P6 patch applied. Preserve the current Kokkos CUDA
RDC, NVSHMEM, and Tpetra double settings. The new executable option is
`--nvshmem-p6`; it is mutually exclusive with `--nvshmem-p52`.

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
mkdir -p ./p6-results

mpirun --map-by ppr:8:node --bind-to none -np 16 \
  ~/cronus/gpu_bind.sh \
  ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe \
  --cuda --numgpus=1 --size=100 \
  --nvshmem-p6 --nvshmem-p6-case=matrix \
  --nvshmem-p6-mode=both \
  --nvshmem-p6-warmup=20 \
  --nvshmem-p6-reps=100 \
  --nvshmem-p6-phase-reps=100 \
  --nvshmem-p6-min-nodes=2 --nvshmem-p6-max-nodes=2 \
  --nvshmem-p6-cross-node \
  --nvshmem-p6-out=./p6-results/inter16_miniFE100
```

The `--nvshmem-p6-out=PREFIX` option writes
`PREFIX_matrix_samples.csv` on rank zero. Each row is one
`(case, section, method, iteration, MPI rank, wall microseconds)` sample.
The runner also prints tagged `P6 DIRECT`, `P6 SPEEDUP`, `P6 PHASE`, and
`P6 CASE` records suitable for later log aggregation.

For a more reproducible set of job commands (inside a matching GPU
allocation), use:

```bash
export P6_EXE=./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe
export P6_BIND=$HOME/cronus/gpu_bind.sh
export P6_OUT_DIR=./p6-results
./packages/tpetra/core/test/PerformanceCGSolve/run_p6_comparison.sh inter16 100 both
```

The wrapper verifies allocation size by hostname, records transport
settings and exact launch command to `.env.txt`, and preserves MPI logs
and full raw per-rank `.csv` files. The output directory must exist for a
manual invocation; the wrapper creates it automatically.

## P6 options

| Option | Default | Meaning |
|---|---|---|
| `--nvshmem-p6` | off | Select P6 benchmark rather than P5.2 correctness suite |
| `--nvshmem-p6-case` | `matrix` | Any P4 named case, or `all` for all ten |
| `--nvshmem-p6-mode` | `both` | `direct`, `phases` or both sequentially in one process |
| `--nvshmem-p6-warmup` | 8 | Untimed Tpetra/NVSHMEM pairs (2–1000) |
| `--nvshmem-p6-reps` | 100 | Measured repetitions **per method** in direct comparison (5–5000) |
| `--nvshmem-p6-phase-reps` | 100 | NVSHMEM-only instrumented repetitions (5–5000) |
| `--nvshmem-p6-min-nodes` | 1 | Reject too few actual physical nodes |
| `--nvshmem-p6-max-nodes` | 0 | Reject too many nodes (0 means no limit) |
| `--nvshmem-p6-cross-node` | off | Require real outgoing cross-node packet(s) |
| `--nvshmem-p6-out` | empty | Save raw rank-major samples to `<prefix>_<case>_samples.csv` |

The standalone wrapper also accepts environment overrides: `P6_CASE`,
`P6_WARMUP`, `P6_REPS`, `P6_PHASE_REPS`, `P6_LABEL`, `P6_EXE`, `P6_BIND`,
`P6_OUT_DIR`, and `P6_MPIRUN`.

## Phase interpretation (NVSHMEM only)

| Phase field | Host wall interval, including normal GPU completion |
|---|---|
| `credit_wait` | Wait on relevant outgoing destinations' previous consumed-generation ACKs; zero for first generation |
| `pack` | Gather the required ordinary X source LIDs into symmetric per-peer scratch, including device fence |
| `put_signal` | Launch and complete GPU-initiated PUT+signal calls for all outgoing packets; **origin-side stage**, not isolated NIC time |
| `ready_wait` | Wait for exactly the expected inbound ready-generation signals, and complete the local control stream |
| `unpack` | Scatter received symmetric inbox entries to arbitrary column LIDs, including device fence |
| `local_copy` | Direct same/permuted local X-to-column mapping, including device fence |
| `ack` | GPU acknowledgment kernel initiation and local CUDA completion; not proof that a remote sender has observed the credit |
| `nvshmem_total` | Wall time for the entire instrumented Exchange Import and trailing Kokkos fence |

**Do not sum medians of per-stage rank maxima** to reconstruct total.
The critical rank can differ per phase and the instruments themselves add
small host clock overhead. Actual remote network delivery can overlap
`put_signal` and `ready_wait`; these numbers are causal stage wall times,
not a physical dissection of individual fabric packet latency.

## Experiment matrix and data analysis

Start with `inter16 size=100 matrix`, then repeat three independent runs
(with different `P6_LABEL` values). Follow with `intra4 size=100`,
`inter2 size=100`, and `inter16 size=32/64` to expose the packet-size trend.
Explore all ten P4 cases separately with `P6_CASE=all` for tiny-message,
one-way and fanout differences. The `--nvshmem-p6-cross-node` flag should
only be required if the chosen case has actual cross-node peer traffic.

Run the same matrix with `NVSHMEM_IB_ENABLE_IBGDA=0` and `=1`, recording
configuration and the known GPU-handler CPU-fallback warnings. Do not
claim a different NIC execution path solely from the environment variable.

To summarize saved per-rank CSV files (one invocation can span multiple
cases), use:

```bash
python3 packages/tpetra/core/test/PerformanceCGSolve/analyze_p6_samples.py \
    ./p6-results/*_samples.csv > ./p6-results/summaries.csv
```

This tool independently validates complete rank/iteration grids and
computes min, p10, median, p90 and max of the **maximum rank per repetition**.
It does not need pandas or another Python package. Preserve raw files for
comparing variability rather than publishing only medians.

### Acceptance criteria

- Native NVCC/Trilinos build passes, including CUDA RDC and NVSHMEM device
  linking; ordinary P5.2.1 correctness test still passes unchanged.
- Rank and topology requirements are satisfied, and miniFE/P5.1 pack
  element and sender-peer totals agree with the known workload.
- Full-vector values are correct before and after the timed runs.
- All allocation owned by this benchmark occurs outside timed regions;
  optional native allocator tracing resolves any Tpetra-internal allocations.
- At least 100 usable direct samples per method and 100 instrumented samples
  per rank are saved for every reported configuration; repeat complete jobs
  for confidence intervals / run-to-run variation.
- Compare **completed full Import** timing (not isolated sender-side PUT
  timing) with the same source, result, device bindings and measured packet
  distribution. No speedup or performance threshold is required for P6
  source acceptance.

Native cluster results have NOT yet been produced for this P6 patch.
