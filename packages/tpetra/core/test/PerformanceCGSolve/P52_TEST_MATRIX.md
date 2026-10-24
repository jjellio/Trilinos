# P5.2 point-to-point packed PUT — correctness acceptance procedure

**Purpose:** Run the *reimplemented* P5.2 `--nvshmem-p52` transport against
all meaningful MPI/GPU/physical-node configurations, verify stock Tpetra
`doImport()` and the P2 CUDA-space MPI replica, and collect reproducible
logs. This is a **correctness** procedure, not a throughput benchmark.

**Source baseline:** The attached `nvshmem-pack-latest.tar(5).xz` ends at
P5.1. Apply/build the **point-to-point P5.2** patch before following these
instructions. The patch is expected to add `NvshmemPackedPut.hpp`,
`NvshmemP52.hpp`, and the `--nvshmem-p52` CLI options. The earlier globally
synchronized P5.2 implementation is **not** suitable for this acceptance.

## 1. Preflight: GPU selection, MPI and build

Use the same MPI/NVSHMEM/CUDA module stack and GPU affinity wrapper that
passed P5.1. `nvcc`/NVSHMEM device-code linkage must be enabled (Kokkos CUDA
relocatable device code). `--cuda` is mandatory. Use a working two-node
allocation for `inter2` and `inter16`; the MPI mapping options do **not**
create the allocation for you. The tests fail when the actual number of
physical nodes differs from the specified topology.

```bash
# From the root of the compiled Trilinos checkout.
EXE="$PWD/packages/tpetra/core/test/PerformanceCGSolve/TpetraCore_Performance-CGSolve.exe"
BIND="$HOME/cronus/gpu_bind.sh"
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_IB_ENABLE_IBGDA=1

# IMPORTANT: choose one of these device-selection modes.
# A. Your gpu_bind.sh restricts CUDA_VISIBLE_DEVICES to ONE GPU per rank:
NUMGPUS=1
# B. No per-rank CUDA_VISIBLE_DEVICES restriction, all 8 GPUs visible:
#    OMIT the BIND wrapper in the launch, and set NUMGPUS=8.

"$EXE" --help 2>&1 | grep nvshmem-p52
```

`--numgpus` is the number of **CUDA devices visible to each rank after
binding**, not necessarily the physical GPUs installed per node. The
previous P5.1 log showed `kokkos_selected_device=0` on every rank, which
is consistent with a per-rank restricted GPU view; do not blindly combine
`gpu_bind.sh` with `--numgpus=8`.

Run the independently compiled host model first:

```bash
cd packages/tpetra/core/test/PerformanceCGSolve
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
    test_packed_put_plan.cpp -o /tmp/test_packed_put_plan
/tmp/test_packed_put_plan
```

## 2. Exact topology/Map sweep

The ten cases are `matrix`, `identity`, `permute`, `sparse-mixed`, `one-way`,
`uneven`, `zero-owner`, `all-empty`, `multi-owner`, `fanout`.

| Topology | Ranks | Physical nodes | Applicable cases | `--case=all` expected summary |
|---|---:|---:|---|---|
| `smoke1` | 1 | 1 | matrix, identity, permute, all-empty | `casesPassed=4 casesSkipped=6` |
| `intra2` | 2 | 1 | preceding four + sparse-mixed, one-way, uneven, zero-owner | `casesPassed=8 casesSkipped=2` |
| `intra4` | 4 | 1 | all ten | `casesPassed=10 casesSkipped=0` |
| `inter2` | 2 | 2 | eight cases as for intra2 | `casesPassed=8 casesSkipped=2` |
| `inter16` | 16 | 2 (eight per node) | all ten | `casesPassed=10 casesSkipped=0` |

**Quick acceptance** — one MPI invocation per topology, exercising all
applicable cases. Run inside the appropriate scheduler allocation; for the
examples, the GPU wrapper exposes one GPU per rank:

```bash
# Single node: 1, 2 and 4 rank allocations/placements.
mpirun -np 1 --bind-to none "$BIND" "$EXE" \
  --cuda --numgpus=1 --size=8 --nvshmem-p52 --nvshmem-p52-max-nodes=1

mpirun --map-by ppr:2:node --bind-to none -np 2 "$BIND" "$EXE" \
  --cuda --numgpus=1 --size=8 --nvshmem-p52 --nvshmem-p52-max-nodes=1

mpirun --map-by ppr:4:node --bind-to none -np 4 "$BIND" "$EXE" \
  --cuda --numgpus=1 --size=8 --nvshmem-p52 --nvshmem-p52-max-nodes=1

# Two allocated physical nodes: 1 rank per node.
mpirun --map-by ppr:1:node --bind-to none -np 2 "$BIND" "$EXE" \
  --cuda --numgpus=1 --size=8 --nvshmem-p52 \
  --nvshmem-p52-min-nodes=2 --nvshmem-p52-max-nodes=2 \
  --nvshmem-p52-cross-node

# Two allocated physical nodes: 8 ranks on each; reference size=100.
mpirun --map-by ppr:8:node --bind-to none -np 16 "$BIND" "$EXE" \
  --cuda --numgpus=1 --size=100 --nvshmem-p52 \
  --nvshmem-p52-min-nodes=2 --nvshmem-p52-max-nodes=2 \
  --nvshmem-p52-cross-node
```

For diagnostics use `--nvshmem-p52-case=one-way` (or any other applicable
case), `--nvshmem-p52-peers` for full peer layouts, and
`--no-nvshmem-p52-peers` for reduced logs. Case names need not be run
separately for logical coverage if the `--case=all` summary is validated,
but **separate CTest tests are preferable** for pinpointing failures.

**Important:** do **not** use `--nvshmem-p52-cross-node` indiscriminately
on every 16-rank case. That switch demands an actual *cross-node PUT*.
At inter16, `identity`, `permute`, `all-empty`, `one-way`, and `zero-owner`
produce **no** cross-node PUTs, even though 16 ranks are spread across
physical nodes. At inter2, `one-way` and `zero-owner` *do* cross the node
boundary because ranks 0 and 1 are on different nodes. The supplied runner
sets this switch only when meaningful, and always enforces `min/max-nodes`.

## 3. Explicit early-signal / delayed-receiver validation

The `--nvshmem-p52-skew-ms=250` option delays rank 0 **after** completing
both MPI reference oracles but **before** entering its first P5.2 Import.
This tests a destination receiving a PUT+signal before it launches its
own waiting work. The first example is deliberately the rank 1 -> rank 0
`one-way` pattern:

```bash
# Two nodes, one rank per node: rank 0 is receive-only.
mpirun --map-by ppr:1:node --bind-to none -np 2 "$BIND" "$EXE" \
  --cuda --numgpus=1 --size=8 --nvshmem-p52 \
  --nvshmem-p52-case=one-way --nvshmem-p52-epochs=4 \
  --nvshmem-p52-cycles=2 --nvshmem-p52-skew-ms=250 \
  --nvshmem-p52-min-nodes=2 --nvshmem-p52-max-nodes=2 \
  --nvshmem-p52-cross-node

# Two nodes, 16 ranks, matrix size=100: rank 0 has an incoming peer.
mpirun --map-by ppr:8:node --bind-to none -np 16 "$BIND" "$EXE" \
  --cuda --numgpus=1 --size=100 --nvshmem-p52 \
  --nvshmem-p52-case=matrix --nvshmem-p52-skew-ms=250 \
  --nvshmem-p52-min-nodes=2 --nvshmem-p52-max-nodes=2 \
  --nvshmem-p52-cross-node
```

`--nvshmem-p52-skew-ms` supports **0..3000 ms**; the 250-ms test is a
correctness check, not a latency measurement. CTest registers four skew
tests by default when two-node registration is enabled.

## 4. Epoch/credit stress

Each Import gets different owned `X` values. Defaults are **4 epochs per
symmetric allocation and 2 allocation/free cycles**. A sender must receive
a consumed-credit signal from its receiver before reusing that receiver's
inbox segment for the next generation. Exercise much longer runs after
baseline correctness passes:

```bash
# All-cases inter16 stress; can be expensive with size=100.
mpirun --map-by ppr:8:node --bind-to none -np 16 "$BIND" "$EXE" \
  --cuda --numgpus=1 --size=32 --nvshmem-p52 \
  --nvshmem-p52-epochs=32 --nvshmem-p52-cycles=4 \
  --nvshmem-p52-min-nodes=2 --nvshmem-p52-max-nodes=2 \
  --nvshmem-p52-cross-node
```

This tests buffer reuse and changing generations, including asymmetric,
fanout, zero-owner and empty patterns. It does **not** measure P5.2
performance. If a deadlock is suspected, limit the test duration using
CTest `TIMEOUT` or the runner's `--timeout` so a hung MPI job does not
consume a node allocation indefinitely.

## 5. Comprehensive CTest: recommended usage

`P52_CTestSweep.cmake` registers topology-aware MPI invocations using the
included `p52_case_runner.py`. **The runner starts exactly one `mpirun`**;
do not invoke `ctest` itself through `mpirun`, or nest it beneath
`TRIBITS_ADD_TEST(... NUM_MPI_PROCS ...)`.

The independent CTest project works against a *finished executable* and
requires no Trilinos reconfiguration:

```bash
# Suppose the supplied suite is extracted to ~/p52-sweep.
# First, in an allocation with at least one suitable GPU node:
cmake -S ~/p52-sweep -B ~/p52-sweep/build-one \
  -DP52_EXE="$EXE" -DP52_BIND_SCRIPT="$BIND" -DP52_NUMGPUS=1 \
  -DP52_ENABLE_SINGLE_NODE=ON -DP52_ENABLE_TWO_NODE=OFF
ctest --test-dir ~/p52-sweep/build-one -N
ctest --test-dir ~/p52-sweep/build-one -j1 --output-on-failure
# Expected: 22 CTest case tests (4+8+10).

# Then, inside a scheduler allocation spanning two nodes:
cmake -S ~/p52-sweep -B ~/p52-sweep/build-two \
  -DP52_EXE="$EXE" -DP52_BIND_SCRIPT="$BIND" -DP52_NUMGPUS=1 \
  -DP52_ENABLE_SINGLE_NODE=OFF -DP52_ENABLE_TWO_NODE=ON \
  -DP52_ENABLE_SKEW=ON
ctest --test-dir ~/p52-sweep/build-two -N
ctest --test-dir ~/p52-sweep/build-two -j1 --output-on-failure
# Expected: 22 CTest tests (8+10 baseline cases +4 skew).
```

Combining the two configurations exercises **all 40 case/topology pairs,
plus four deliberate late-receiver tests**. If CTest is configured with
both single-node and two-node tests enabled in a single two-node allocation,
`ctest -N` reports **44 tests**. Every test has `RUN_SERIAL`, a per-test
timeout, and separately archived stdout/stderr plus JSON validation output
in `build-*/p52-test-logs/`.

For a faster coarse sweep, configure `-DP52_ENABLE_CASE_TESTS=OFF
-DP52_ENABLE_AGGREGATE=ON`. That registers one `--case=all` test per
topology instead of individual cases; CTest cannot isolate the failing Map
until you rerun the individual case. Add `-DP52_ENABLE_STRESS=ON` to
register longer credit/epoch stress cases (two intra-node, three inter-node).
CTest labels include `p52-one-node`, `p52-two-node`, `p52-skew`, `p52-stress`,
`p52-case`, and `p52-quick`:

```bash
ctest --test-dir ~/p52-sweep/build-two -L p52-two-node -j1 --output-on-failure
ctest --test-dir ~/p52-sweep/build-two -L p52-skew -j1 --output-on-failure
ctest --test-dir ~/p52-sweep/build-two -R 'P52_inter16_matrix' -j1 --output-on-failure
```

### Optional: register in Trilinos's own CTest

The accompanying `p52-trilinos-ctest-integration.patch` adds
`PERFORMANCE_CGSOLVE_NVSHMEM_P52_SWEEP_CTEST=ON` to the
`PerformanceCGSolve/CMakeLists.txt` and installs the runner, registration
module and this guide inside the Trilinos source directory. Reconfigure
Trilinos with your existing NVSHMEM options *plus*:

```bash
-DPERFORMANCE_CGSOLVE_NVSHMEM_P52_SWEEP_CTEST=ON \
-DP52_BIND_SCRIPT="$HOME/cronus/gpu_bind.sh" \
-DP52_NUMGPUS=1 \
-DP52_ENABLE_TWO_NODE=ON
```

Two-node tests will only pass inside a genuine two-node MPI allocation.
This test-registration patch is based on the **last attached P5.1 source
CMakeLists**, and appends a separate opt-in block. Check application against
your P5.2-patched tree with `git apply --check` before using it.

## 6. Acceptance criteria and expected summary

Each CTest job fails if the `mpirun` process fails, times out, emits no
P5.2 summary, has missing/duplicate case results, reports the wrong
physical node count or rank set, reports nonzero per-Import world sync,
or has mismatched GPU PUT/ready/ACK totals. The runner records full logs
and structured per-test JSON so failures can be reproduced by exact
command line. It also verifies a known P5.1-derived matrix fixture for
16 ranks, two nodes, size=100:

| Metric | Per Import | Default 8 Imports (4 × 2) |
|---|---:|---:|
| GPU PUT+signal peer calls | 44 | 352 |
| GPU PUT payload doubles | 309,105 | 2,472,840 |
| GPU PUT payload bytes | 2,472,840 | 19,782,720 |
| Ready waits / consumption ACKs | 44 each | 352 each |
| Credit waits (only after first generation per allocation) | 44 for generations 2–4 | 264 |
| Cross-node PUT calls | 10 | 80 |
| Cross-node payload bytes | 164,912 | 1,319,296 |

The aggregate 16-rank all-case suite had **141 directed peer relationships**
and **309,205 double values per Import** in the previously accepted P5.1
plan. The runner also checks those counts for the size=100 `--case=all`
fixture (when case/topology counts are unchanged).

**A passing test means the code's own full-vector comparison against
Tpetra and P2 passed and its reported metadata/operation accounting is
consistent.** It does not establish a performance speedup or independently
prove the internal stream/remote-memory ordering beyond those checks.
The GPU-native build, actual node placement, and runtime correctness still
must be verified on the real cluster.

## 7. CLI options quick reference

| CLI | Meaning | Default / restriction |
|---|---|---|
| `--nvshmem-p52` | Select corrected P5.2 | Off unless specified |
| `--nvshmem-p52-case=X` | Case name, or `all` | `all` |
| `--nvshmem-p52-epochs=N` | Different X generations per allocation | 4, range 4–1000 |
| `--nvshmem-p52-cycles=N` | Collective malloc/free cycles | 2, range 2–100 |
| `--nvshmem-p52-peers` | Print sender/receiver records | On |
| `--no-nvshmem-p52-peers` | Reduce log volume | Optional |
| `--nvshmem-p52-min-nodes=N` | Assert placement lower bound | 1 |
| `--nvshmem-p52-max-nodes=N` | Assert placement upper bound | 0 (unbounded) |
| `--nvshmem-p52-cross-node` | Require actual cross-node PUT relationship | Off |
| `--nvshmem-p52-skew-ms=N` | Delay rank 0 before first PUT Import | 0, range 0–3000 |
| `--size=N` | miniFE matrix dimension parameter | 8 quick, 100 acceptance inter16 |
| `--cuda --numgpus=N` | Required Tpetra CUDA backend / local device selection | Match GPUs visible to each rank |

### Caveat about the existing CTest entry

The patch's `PERFORMANCE_CGSOLVE_NVSHMEM_P52_CTEST` option registers **one
MPI-rank smoke** only. This is useful to prove the binary can initialize
and run local-only cases, but cannot validate PUT, remote signal arrival,
consumed-credit flow, or any network transport. The new comprehensive
registration is deliberately **separate** from that smoke option.
