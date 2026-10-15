# VMM–Tpetra Testing: Overview and Developer Guide

**Scope:** In-tree regression tests for the experimental CUDA Virtual Memory Management (VMM) implementation in Tpetra.  
**Recommended repository location:** `packages/tpetra/core/test/VMM/TESTING_OVERVIEW.md`  
**Validation baseline:** Milestone 4.1 implementation, P0–P4 tests, two MPI ranks on two GPUs (October 2026).

## Why this suite exists

The VMM prototype gives a Tpetra `MultiVector` GPU memory that other GPUs on the same node can address. An alternative, locally stored CRS representation can then refer directly to those values, allowing SpMV to avoid the conventional MPI packing, transfer, and unpacking of remote vector elements when the direct VMM path is eligible.

The optimization must not change the mathematical meaning of `CrsMatrix::apply()`. It must also retain Tpetra's classical path for cases the experimental implementation does not support. This test suite was built *before* relaxing the original Map restrictions and runtime eligibility checks, so it can distinguish an intentional expansion of VMM support from a numerical, ownership, or dispatch regression.

The suite has two complementary requirements:

1. **Numerical correctness:** small, deterministic matrices and vectors are compared with independent analytical answers—not merely with a second Tpetra `apply()` call.
2. **Execution-path correctness:** where instrumentation is enabled, the tests establish whether an operation actually used direct VMM, classical Tpetra, or an `alpha == 0` early exit. VMM allocation alone does **not** prove that the direct kernel ran.

## Test organization

Tests live inside Trilinos, using **Teuchos unit tests**, the **TriBITS/CTest** build integration, and dedicated shell runners for MPI environment sweeps. Trilinos is not installed separately for these tests. The primary test executable is `TpetraCore_VmmRegression.exe` (the exact build path depends on the TriBITS configuration).

| Milestone | Purpose | What it protects |
|---|---|---|
| **P0 — Harness** | Two-rank CUDA/MPI smoke test; physical GPU identity check; launch and logging infrastructure | A working in-tree build, distinct GPU assignments, basic remote references, and repeatable configuration sweeps |
| **P1 — SpMV correctness** | Diagonal, boundary-crossing, all-remote and stencil matrices; `alpha`/`beta` combinations | Numerical semantics and proof of direct VMM versus classical dispatch on **both** ranks |
| **P2 — MultiVector and fallback** | Allocation policy; multicolumn fallback; shallow/deep copies and views; host/device coherence; allocation lifetime; Exporter, replicated-output and transpose cases | Storage ownership, host/device correctness, and preserved classical behavior when VMM is ineligible |
| **P3 — Map contracts** | Interleaved and locally permuted GIDs; equivalent-but-distinct Map objects; overlapping ownership; non-one-to-one domain Map | A measurable **pre-refactor baseline** and a separate **post-refactor acceptance target** for generalized Maps |
| **P4 — Debug and boundaries** | Zero-local-row and skewed partitions; invalid X/Y Map diagnostics; timer counters; optional Fabric probe | Empty partitions, rank-consistent selection, existing DistObject diagnostics, M5.3 VMM Map semantic checks, and diagnostic observability |

The milestone-specific documents are the source of detail: [P1](P1_README.md), [P2](P2_README.md), [P3](P3_README.md), and [P4](P4_README.md). The directory's [README](README.md) describes the P0 harness. P3 also carries `p3_expected_paths.tsv` for present-versus-future expectations. The C++ test sources provide the definitive fixtures and assertions.

### Intentional design choices

- **Two ranks, explicit data:** most fixtures have eight global rows, with four per rank unless a test deliberately changes the distribution. Small cases make remote addressing errors reproducible and the answers easy to inspect.
- **One GPU per rank:** the harness checks physical PCI bus identities, not only CUDA device ordinals (which may both appear as device 0 under rank-local `CUDA_VISIBLE_DEVICES`).
- **Separate MPI jobs per environment:** `Tpetra::Details::Behavior` caches settings; switching flags within a process is not a meaningful configuration sweep.
- **Caller-managed completion:** test fixtures explicitly synchronize producers before cross-rank reads and complete readers before reusing storage. They do **not** add barriers or fences to production SpMV. Dedicated P4 timer runs enable Tpetra's existing synchronization setting for diagnostics only.
- **No silent fallbacks in path-sensitive tests:** a correct result produced by classical MPI does not count as a successful direct-VMM test when direct VMM was required.

## Building and running

**Prerequisites:** an in-tree Trilinos build with Tpetra tests, MPI, CUDA, the required scalar/Node instantiations, and the experimental VMM code enabled; two MPI processes on one GPU-enabled node with access to two distinct GPUs. P0–P4 primarily exercise POSIX IPC.

Enable the test directory with `-DTpetra_ENABLE_VMM_TESTS=ON` when configuring Trilinos. The MPI launcher may already be configured with `MPI_EXEC_PRE_NUMPROCS_FLAGS` and `MPI_EXEC_POST_NUMPROCS_FLAGS` to supply `--bind-to none` and the site's rank-to-GPU wrapper. The expected launch shape is:

```bash
mpirun --bind-to none -np 2 \
  "$HOME/cronus/gpu_bind.sh" \
  /path/to/TpetraCore_VmmRegression.exe <test-arguments>
```

After reconfiguring when test source files or registrations change:

```bash
cmake --build /path/to/trilinos-build \
  --target TpetraCore_VmmRegression --parallel

# Discover the registered cases.
ctest --test-dir /path/to/trilinos-build -N -R VMM

# Run the full registered P0–P4 suite without overlapping GPU jobs.
ctest --test-dir /path/to/trilinos-build -j1 \
  -R 'VMM_(Smoke|P[1-4])_NP2' --output-on-failure
```

For full environment matrices, per-scenario logs, and additional path assertions, use the standalone runners from the **build directory**:

```bash
EXE=./packages/tpetra/core/test/VMM/TpetraCore_VmmRegression.exe
TESTS=/path/to/trilinos/packages/tpetra/core/test/VMM

"$TESTS/run_vmm_tests.sh"    --exe "$EXE" --full --outdir ./vmm-p0-results
"$TESTS/run_vmm_p1_tests.sh" --exe "$EXE" --full --outdir ./vmm-p1-results
"$TESTS/run_vmm_p2_tests.sh" --exe "$EXE" --full --outdir ./vmm-p2-results
"$TESTS/run_vmm_p3_tests.sh" --exe "$EXE" --full --outdir ./vmm-p3-results --mode matrix_generalized
"$TESTS/run_vmm_p4_tests.sh" --exe "$EXE" --full --outdir ./vmm-p4-results --mode matrix_generalized
```

The runners default to the site's `~/cronus/gpu_bind.sh`; supply `--gpu-bind /path/to/wrapper` if different. Each creates an environment-specific log and a machine-readable `results.tsv`; timeouts, failed numerical assertions, incorrect path events, and unexpected job exits are failures. See each runner's `--help` for available filters.

## Configuration and diagnostics

| Setting | Meaning for these tests |
|---|---|
| `TPETRA_EXPERIMENTAL_VMM` | Enables/disables the experimental VMM feature globally |
| `TPETRA_EXPERIMENTAL_VMM_NATIVE_MV` | Enables/disables native VMM-backed MultiVector allocation |
| `TPETRA_EXPERIMENTAL_VMM_IPC=posix` | Standard tested cross-process memory-sharing mechanism |
| `TPETRA_EXPERIMENTAL_VMM_STATISTICS=1` | Enables allocation, skip-reason and `event=apply` diagnostic records; `0` suppresses them |
| `TPETRA_DEBUG=1` | Dedicated P4 debug tests validate both existing DistObject diagnostics and new M5.3 CrsMatrix VMM Map semantics; ordinary runs remain debug off |
| `TPETRA_SYNCHRONIZE_APPLY=1` | Used only in explicit P4 timer diagnostics; ordinary correctness configurations use `0` |
| `VMM_TEST_MAP_MODE=baseline\|allocation_only\|matrix_generalized\|generalized` | **Test-only** expectations for M4.1 / M5.1 / M5.2 / final M5.4; not a production VMM switch |

**Environment namespace:** `Tpetra::Details::Behavior` rejects unrecognized environment variables prefixed `TPETRA_` or `Tpetra_`. Test-only controls must avoid those prefixes. In particular, use `VMM_TEST_MAP_MODE`, **not** the obsolete `TPETRA_VMM_TEST_MAP_MODE`. The CMake cache setting `Tpetra_VMM_P3_MAP_MODE` is distinct from a process environment variable.

Selected statistics-enabled records look like:

```text
[TPETRA_MV] event=allocate ... storage=vmm
[TPETRA_MATRIX] event=vmmSkip ... reason=domain_map_not_one_to_one
[TPETRA_MATRIX] event=apply path=vmm rank=0 label="P1_S03_AllRemote"
[TPETRA_MATRIX] event=apply path=classic rank=1 label="P2_F03_Exporter"
[TPETRA_MATRIX] event=apply path=alpha_zero rank=0 label="P1_S06_AlphaBeta"
```

These messages are for correctness and investigation, **not performance measurements**: statistics and log I/O can perturb timings. P4's `Tpetra VMM: direct spmv` and classical timer-count checks independently corroborate dispatch without requiring particular elapsed times.

## Validated baseline

The following results were reviewed from the supplied two-GPU run archives on **October 14, 2026**. Counts are *executed Teuchos tests across configuration sweeps*, not counts of unique test cases or claims about other hardware.

| Suite | Observed executions | Result |
|---|---:|---|
| P0 harness | 6/6 | PASS; GPU preflight passed |
| P1 numerical/dispatch | 30/30 | PASS |
| P2 MultiVector/fallback | 84/84 | PASS |
| P3 Map behavior, **baseline mode** | 30/30 | PASS |
| P4 debug/boundary/timer/Fabric probe | 18/18 | PASS; optional Fabric execution **SKIP** |

The P4 Fabric capability probe reported `supported=0` on both ranks (`reason=fabric_allocation`). The development node lacks the required IMEX/Fabric capability; the optional Fabric IPC execution was therefore correctly **skipped**, not counted as a VMM correctness failure. The validated path on this system is **POSIX IPC**.

This is a focused **NP=2, two-GPU prototype baseline**. It establishes correctness and dispatch behavior for the exercised fixtures, not comprehensive production qualification, performance scaling, or 4/8-GPU coverage. M5.2 passed the user's full 32/32 CTest suite on the two-GPU POSIX node. M5.3 adds debug-only semantic checks to noncontiguous, one-to-one matrix addressing, pending validation on the two-GPU node.

## Using the suite during the Map/eligibility refactor

1. **Keep P1, P2, and P4 green.** These protect the numerical kernel, MultiVector ownership, fallback routes, and debug/boundary semantics as implementation details change. P0 continues to protect the harness and launcher.
2. **Use P3 `matrix_generalized` for M5.2.** G01/G02 have native VMM storage and now execute direct VMM SpMV through directory-derived owner/local indices; G03 still uses classic SpMV because it passes distinct Map objects; G05 has an ambiguous domain Map and remains classical. The historical `baseline` and `allocation_only` modes are retained for earlier code revisions and intentionally differ from M5.2.
3. **Reserve P3 `generalized` for M5.4.** This final acceptance specification is not an enable flag. It additionally expects direct VMM for equivalent-but-distinct Map objects after debug validation and pointer-check cleanup; ambiguous non-one-to-one domain Maps remain a classical fallback.
4. **Investigate failures in layers:** GPU binding → object allocation (`storage=vmm` or skip reason) → matrix VMM context → rank-local `event=apply` → analytic numerical result. Do not interpret an allocator success as proof of kernel dispatch.
5. **Preserve asynchronous semantics.** New compatibility checks should follow the agreed Tpetra debug contract where appropriate; tests should not force collectives or expensive Map comparisons into normal `CrsMatrix::apply()`.

For test-specific formulas, exception text, lifecycle details, individual CTest cases, and expected diagnostic counts, refer to the corresponding milestone README and test source rather than extending this overview.

