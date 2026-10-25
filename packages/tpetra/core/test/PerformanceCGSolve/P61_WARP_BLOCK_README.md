# P6.1 — A/B/C comparison of GPU PUT-with-signal issuance granularity

**Status: implemented and host-checked; native CUDA/NVSHMEM build and GPU
measurements pending.** P6.1 extends the P6 timing experiment. It does NOT
change the P5.2/P5.2.1 transport baseline or the generation/credit protocol.
The user-provided prior P6 result had Tpetra ~99.5 us and the GPU single-thread
packed PUT ~1164.5 us at 16 PEs, `--size=100`; these results are NOT retimed
by the new patch.

## Exactly what differs

Each outgoing destination PE continues to receive **one contiguous packet**
with the same P5.1 `packOffset`, `remoteInboxOffset`, `remoteSignalIndex`,
packet length, and generation. The receiver stream-waits on the same
`ready[sourcePE]` values and sends the same per-source consumption credits.

| CLI `--nvshmem-p6-put-variant` | Method | GPU PUT issuance per peer |
|---|---|---|
| `thread` (default) | `exchange.import()` | Original **one GPU thread** calls `nvshmem_double_put_signal` |
| `warp` | `exchange.importWarp()` | One **32-lane CUDA warp** calls `nvshmemx_double_put_signal_warp` collectively |
| `block` | `exchange.importThreadBlock()` | One **128-thread CUDA block** calls `nvshmemx_double_put_signal_block` collectively |

No on-stream PUT variant is included; that is reserved for a later milestone.
The `_warp` / `_block` routines are GPU-thread-group collectives **within a
single PE**, not inter-PE collectives. Every lane/thread of the participating
warp or block must call with the same arguments, which the kernels guarantee.
They do not require `nvshmemx_collective_launch` or any MPI/NVSHMEM world
barrier. NVIDIA documents the call requirements in the NVSHMEM API overview.

### Code shape

`NvshmemPackedPut.hpp` retains the original `packedPutSignalKernel`,
`issuePutsCuda` and `Exchange::import` **byte-for-byte**. P6.1 only adds:

- `packedPutSignalWarpKernel`: four complete warps per 128-thread CUDA block;
  `warpId = blockIdx.x*4 + threadIdx.x/32`. The final unused warp(s) skip
  the collective uniformly; all 32 lanes within an active warp participate.
- `packedPutSignalBlockKernel`: exactly one block per peer, all 128 threads
  call the same block-scoped PUT+signal.
- `issuePutsWarpCuda` and `issuePutsThreadBlockCuda`: launch and locally
  synchronize their respective kernels, matching the original fence boundary.
- `importWarp` and `importThreadBlock`: explicit user-facing methods; a new
  common **cooperative-only** protocol helper performs the same credit, pack,
  send, ready, unpack, local-copy, ACK sequence and optional phase timing.
  The original `Exchange::import` is not refactored or replaced.

The P6 runner selects the target member-function pointer **before** warmup
and uses that method for **all** warmup, untimed validation-recovery, direct,
and instrumented phase samples. No timed allocation or setup is added.
It preserves P6's direct boundary (pre-fence + MPI barrier outside timing,
then Import + post-fence inside timing) and phase boundary (no per-iteration
MPI exchange). The Tpetra path is unchanged. Thread mode preserves the
existing P6 output method label `nvshmem_packed_put`; additional labels are
`nvshmem_packed_put_warp` and `nvshmem_packed_put_block`.

**Interpretation caveat:** NVSHMEM may use many GPU lanes on P2P/NVLink,
while an InfiniBand peer can still be issued by just one lane internally.
A warp/block variant is not guaranteed to accelerate large inter-node PUTs.
Also, applying all 32 or 128 lanes to an 8-byte packet may cost more than a
single thread. P6.1 deliberately does not add packet-size thresholds; that
would be an extra algorithmic variable beyond the requested A/B/C test.

## Build / compile gate

Apply from Trilinos source root to the P6 implementation **with the previously
provided `LocalCopy` naming fix already applied**:

```bash
git apply --check /path/to/nvshmem-p61-warp-block.patch
git apply /path/to/nvshmem-p61-warp-block.patch
```

Use the same NVCC, NVSHMEM static device library, CUDA RDC, Tpetra and IBGDA
configuration that passed P6. Ensure the installed `nvshmemx.h` provides
`nvshmemx_double_put_signal_warp` and `nvshmemx_double_put_signal_block`.
The local artifact environment has neither NVCC nor your cluster libraries,
so it cannot establish native compilation.

## Direct comparison and instrumented phases

Set the same environment used for the baseline:

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1
```

For a single two-node/16-PE test, use the same launcher and GPU binding as P6;
only vary `--nvshmem-p6-put-variant`:

```bash
mpirun --map-by ppr:8:node --bind-to none -np 16 \
  ~/cronus/gpu_bind.sh \
  ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe \
  --cuda --numgpus=1 --size=100 --nvshmem-p6 --nvshmem-p6-case=matrix \
  --nvshmem-p6-mode=both \
  --nvshmem-p6-put-variant=warp \
  --nvshmem-p6-warmup=20 --nvshmem-p6-reps=100 \
  --nvshmem-p6-phase-reps=100 \
  --nvshmem-p6-min-nodes=2 --nvshmem-p6-max-nodes=2 \
  --nvshmem-p6-cross-node --nvshmem-p6-out=./p61-results/inter16_warp
```

Substitute `thread` and `block` to obtain the other two comparable runs,
with **different `--nvshmem-p6-out` prefixes** to avoid overwrites.
The wrapper `run_p61_variants.sh` automates all three without contention:

```bash
mkdir -p p61-results
export P6_OUT_DIR=./p61-results
export P6_EXE=./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe
export P6_BIND="$HOME/cronus/gpu_bind.sh"
export P6_WARMUP=20 P6_REPS=100 P6_PHASE_REPS=100
export P61_PASSES=2 # run A,B,C then C,B,A to check drift
./packages/tpetra/core/test/PerformanceCGSolve/run_p61_variants.sh inter16 100 both
```

Runs are **sequential** and each process allocation is created outside its
own timer. The script varies only GPU issuance granularity; it does not
change the packet plan, source values or the direct/phase timing boundaries.
Both `analyze_p6_samples.py` and the new
`analyze_p61_variants.py` understand the unchanged six-column raw CSV format.
The latter calculates median rank-max Tpetra-to-PUT speedups and matches each
warp/block run to its **same-pass** thread baseline using the unique filename
variant token; it also reports the median `put_signal` phase. For multi-pass comparisons, compare medians *within the same pass*
first and inspect pass-to-pass variation before concluding that any variant
improves performance.

To produce one CSV of matched variants after the sweep:

```bash
python3 ./packages/tpetra/core/test/PerformanceCGSolve/analyze_p61_variants.py \
  p61-results/*_samples.csv > p61-results/variant_comparison.csv
```

## Recommended correctness and performance acceptance

1. Run two PEs on one node and two PEs on two nodes. Verify the 2-PE
   miniFE Map contains nonempty peer packets, then run each of the three
   variants with `--nvshmem-p6-case=matrix --nvshmem-p6-mode=both`.
2. Run 16 PEs across two nodes (eight PEs/node), with `--size=100`.
   Compare all three, collecting the raw per-rank phase samples.
3. For both new variants, check source/target correctness before and after
   measured runs; then separately stress repeated generations and mixed
   incoming/outgoing traffic (`--nvshmem-p6-case=sparse-mixed`, `multi-owner`,
   `uneven` and `fanout`), so we do not infer correctness from speed.
4. Compare direct median of per-iteration rank maxima, paired median ratios,
   and especially the `put_signal` phase. The per-phase *rank maxima* are not
   additive because each phase can peak on a different rank.
5. Confirm the baseline `thread` method's performance roughly reproduces
   the previous P6 run under the same IBGDA/NIC mapping. A large deviation
   indicates environment/benchmark drift, not necessarily a warp/block gain.

The benchmark verifies the final warmup output and the final direct/phase
results against independent owner-rank/source-LID values, outside timers.
It is **not** a full every-epoch transport stress test for warp/block. The
existing P5.2.1 default stress still uses the untouched thread baseline;
for exhaustive cooperative transport correctness a later isolated
`--nvshmem-p52` variant stress hook can be added without changing P6.1's
baseline kernel. This caveat must be recorded when interpreting benchmark PASS.

## Source constraints

The release patch is based on the generated P6 full-source snapshot
`nvshmem-p6-timing-full-source.tar.xz` **plus**
`nvshmem-p6-localcopy-compile-fix.patch`. If your working Trilinos source
contains further changes beyond that snapshot, run `git apply --check`
before applying it. The bundle also includes the changed-file overlay.
