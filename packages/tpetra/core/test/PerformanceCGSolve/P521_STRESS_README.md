# P5.2.1 — NVSHMEM-only, asynchronous per-peer stress test

## Baseline and installation

This change is **constructed against the actual** `nvshmem-pack-latest.tar(6).xz`
source snapshot. It retains that snapshot's **native device-only**
`packedPutSignalKernel` / `acknowledgeConsumedKernel`, and the existing P5.2
`nvshmemx_signal_wait_until_on_stream` point-to-point credit protocol. It does
not replace those kernels, reintroduce a collective kernel launch, or add a
per-Import MPI/NVSHMEM world barrier.

From the **Trilinos source root** (the directory containing `packages/`):

```bash
git apply --check /path/to/nvshmem-p521-tar6.patch
git apply /path/to/nvshmem-p521-tar6.patch
```

The patch was generated from the bytes in tar(6) and successfully tested against
an untouched second extraction. Do **not** apply the previous `p521-stress-
point-to-point.patch` first. If your actual checkout differs from tar(6), use
the supplied changed-files overlay instead of forcing hunks into an unrelated
file. You can restore original files from your usual source control first.

## What the P5.2.1 test changes

Ordinary `--nvshmem-p52` remains the original P5.2 Tpetra/P2/P5.2
three-oracle test. Add `--nvshmem-p52-stress-only` to instead exercise a
persistent P5.1 plan across repeated NVSHMEM packed-PUT Imports. For each
source epoch on each PE, independently:

1. Optionally delay entry to skew peer execution.
2. Fill ordinary Tpetra device X with changing, signed, exactly representable
   owner-rank/source-LID/epoch values. Reset Y to a NaN sentinel.
3. Perform **only** `Exchange::import(X,Y)`: wait on relevant outgoing peer
   credits, GPU pack/PUT+ready-signal, on-stream wait for incoming peers,
   GPU unpack/local copy, and GPU consumption acknowledgments.
4. Optionally inject a receiver delay **after the GPU finishes unpacking but
   before acknowledging consumption**, to hold the sender's reuse credit.
5. Verify *every column-map LID* against the independent expected value,
   every generation by default. No Tpetra `doImport`, P2 MPI exchange,
   MPI collectives, or barriers occur **between repeated Imports**.

The P5.1 metadata setup and `Pattern` constructor use one-time MPI
collectives before the stress loop; NVSHMEM symmetric malloc/free is
collective at **cycle boundaries**; final MPI reductions aggregate diagnostics
**after** the loop. These operations remain outside the consecutive-Import
stress interval. Full-case mode also has separate setup/retirement between
Map cases. This is not a claim that the entire executable uses no MPI.

The source changes in `NvshmemPackedPut.hpp` add a test-only receiver ACK
hold and credit-wait wall-time counters. Normal P5.2 behavior has a zero
receiver hold. Measured credit wait includes submitting the per-peer waits and
synchronizing this PE's control CUDA stream; it is not a NIC latency metric.

## New options

| Flag | Default | Purpose |
|---|---:|---|
| `--nvshmem-p52-stress-only` | off | Select the NVSHMEM-only repeated-Import path |
| `--nvshmem-p52-epochs=N` | 4 | Changed source epochs **per** symmetric allocation, 4–1000 |
| `--nvshmem-p52-cycles=N` | 2 | Collective alloc/free cycles, 2–100 |
| `--nvshmem-p52-stress-entry-ms=N` | 3 | Selected PE delays before Import, 0–1000 ms |
| `--nvshmem-p52-stress-ack-ms=N` | 5 | Selected receiver delays after unpack and before ACK, 0–1000 ms |
| `--nvshmem-p52-stress-verify-every=N` | 1 | Full LID verification every Nth epoch **plus the final epoch** |
| `--nvshmem-p52-stress-seed=N` | 17 | Deterministic but rank/epoch-dependent skew seed, nonnegative |
| `--nvshmem-p52-case=all` | all | All ten P4 Map/distribution cases, or select one |

All existing P5.2 topology checks, `--numgpus`, and `--size` remain supported.
P5.2.1 emits `[NVSHMEM P5.2.1 CASE]` records and the original
`[NVSHMEM P5.2] SUMMARY`. The stress runner verifies the exact per-case
PUT/ready/ACK/credit counts and all expected case records. `checkedValues`
counts independent full-target comparisons; `maxCreditWaitUs` is the largest
observed rank-local sender credit-wait wall duration.

## Recommended native GPU test

Preserve your established GPU/NIC binding and two-node allocation. With each
PE restricted to one CUDA-visible GPU, pass `--numgpus=1`:

```bash
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1

mpirun --map-by ppr:8:node --bind-to none -np 16 \
  ~/cronus/gpu_bind.sh \
  ./packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe \
  --cuda --numgpus=1 --size=100 --nvshmem-p52 \
  --nvshmem-p52-case=all --nvshmem-p52-stress-only \
  --nvshmem-p52-epochs=128 --nvshmem-p52-cycles=2 \
  --nvshmem-p52-stress-entry-ms=3 \
  --nvshmem-p52-stress-ack-ms=5 \
  --nvshmem-p52-stress-verify-every=1 \
  --nvshmem-p52-stress-seed=17 \
  --nvshmem-p52-min-nodes=2 --nvshmem-p52-max-nodes=2 \
  --nvshmem-p52-cross-node --no-nvshmem-p52-peers
```

The previously accepted 16-PE P5.1 plan (size=100) has **141** directed
sender/receiver relationships and **309,205** exported doubles summed across
the ten P4 layouts per generation. At 128 epochs times two cycles, **expected**
all-case totals are 36,096 logical GPU PUT calls, 79,156,480 transferred
doubles, 36,096 ready waits/ACKs, and 35,814 previous-generation credit waits.
These are assertions against the known plan, **not** observed P5.2.1 results.
IBGDA's GPU NIC-handler initialization is known to emit fallback warnings on
this cluster; successful IBGDA-requested execution is not proof of GPU NIC
handler selection.

For an explicit 2-PE backpressure test, use a two-node allocation and
`--nvshmem-p52-case=one-way --nvshmem-p52-stress-ack-ms=25` so the
receive-only rank 0 holds ACKs sent to rank 1. Verify a nontrivial
`maxCreditWaitUs` in the output. For longer tests, select 512 epochs per
allocation, two cycles, and different seeds. Do not set epochs above the
program's validation maximum of 1000 without changing source.

## CTest stress matrix

The separate `ctest/` directory in the patch bundle contains CMake and a
Python MPI-launch/validator. This suite **owns its mpirun**, so do not nest
it in Trilinos's MPI-wrapped `TRIBITS_ADD_TEST`.

```bash
EXE=/absolute/path/to/TpetraCore_Performance-CGSolve.exe
BIND=$HOME/cronus/gpu_bind.sh

# On a one-node GPU allocation (3 tests):
cmake -S /path/to/p521-bundle/ctest -B ~/p521-one \
  -DP521_EXE="$EXE" -DP521_BIND_SCRIPT="$BIND" -DP521_NUMGPUS=1 \
  -DP521_ENABLE_ONE_NODE=ON -DP521_ENABLE_TWO_NODE=OFF
ctest --test-dir ~/p521-one -j1 --output-on-failure

# On a TWO-node allocation (5 normal + 2 optional long tests):
cmake -S /path/to/p521-bundle/ctest -B ~/p521-two \
  -DP521_EXE="$EXE" -DP521_BIND_SCRIPT="$BIND" -DP521_NUMGPUS=1 \
  -DP521_ENABLE_ONE_NODE=OFF -DP521_ENABLE_TWO_NODE=ON -DP521_ENABLE_LONG=ON
ctest --test-dir ~/p521-two -j1 --output-on-failure
```

The two-node `inter2/one-way` test forces an ACK delay of **at least 25 ms**
and requires an observed sender credit-wait interval of at least 1 ms, making
the backpressure test more discriminating. The runner preserves complete
`.log` and `.json` files, validates returned exit status, run placement,
full-case coverage, the known miniFE counts, and generation signal totals.
Tests have timeouts, but the actual multi-GPU stress can only be verified on
your cluster.

## Acceptance and limitations

**Source-level checks completed:** patch applies cleanly to an untouched copy
of the supplied tar(6) source; the P5.0 analyzer and P5.1/P5.2 standalone
model regressions pass GCC/Clang C++17; the P5.2.1 stress harness passes
mocked host Tpetra/Exchange template-execution checks with GCC and Clang;
standalone CTest configuration/validation has been checked offline.

**Still required:** native Trilinos/NVCC/NVSHMEM compilation and execution of
the full one-node and two-node stress matrix (including explicit credit stalls),
all changed-value comparisons, and the long 512-epoch tests. A single-rank
mock is not evidence of real asynchronous remote progress.
