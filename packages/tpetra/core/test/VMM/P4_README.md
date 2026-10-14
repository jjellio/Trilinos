# P4 — Debug Map diagnostics, boundary distributions, timed-path checks

P4 is **test-only** and builds in the existing TpetraCore regression executable
from P0–P3. It requires the **corrected P3 installation** (test-only environment
variable `VMM_TEST_MAP_MODE`, not the reserved `TPETRA_VMM_TEST_MAP_MODE`).

No production `CrsMatrix`, `MultiVector`, `Behavior`, or VMM-allocator files
change. As in P0–P3, caller-side fences and communicator barriers in test
fixtures are deliberate; ordinary `apply()` remains asynchronous. A separate
explicit **diagnostic** test uses existing `TPETRA_SYNCHRONIZE_APPLY=1` to
check timer invocation counts, not runtime performance.

## Added tests

| Group / ID | Two-rank fixture | Contract checked |
|---|---|---|
| `VMM_Boundary` B01 | Rank 0 has all 8 GIDs, rank 1 has 0 | Nonzero padded VMM allocation on zero-local-entry rank, storage/content |
| `VMM_Boundary` B02 | Same 8/0 rank distribution, diagonal A | Numerical result, rank agreement on VMM versus classic dispatch (no fixed requirement that the empty-rank configuration take direct VMM) |
| `VMM_Boundary` B03 | Rank 0 owns 1 GID, rank 1 owns 7 | Remote X access, unequal logical sizes, native VMM and direct path when enabled |
| `VMM_Debug` G06a | Wrong X domain Map of the same local length, matrix with Importer | With `TPETRA_DEBUG=1`, `DistObject` throws *source Map* diagnostic before MPI transfer |
| `VMM_Debug` G06b | Wrong Y range Map of the same local length, matrix with Exporter | With `TPETRA_DEBUG=1`, `DistObject` throws *target Map* diagnostic before MPI transfer |
| `VMM_Debug` G06c | X Map has a different RCP identity but `isSameAs()` | Valid classical Import works under debug; preserves P3's baseline/future distinction for direct VMM eligibility |
| `VMM_Timer` T01 | Balanced 4/4, all-remote CRS | With opt-in synchronized instrumentation, direct or classic named Teuchos timer increments by exactly one on each rank; analytical result correct |
| `VMM_FabricProbe` F01 | Both GPU ranks, driver-only capability query | Fabric allocation probe succeeds or clearly reports unsupported, without production VMM abort |

The G06 tests deliberately use **nontrivial Importer/Exporter paths**, because
classic `CrsMatrix::apply()` does not guarantee full Map checking without
communication. All ranks use the same invalid fixture. These are *debug-only*
negative tests; never run them with `TPETRA_DEBUG=0`.

## CMake / CTest

After applying `vmm-tpetra-P4.patch` at the Trilinos repository root,
**reconfigure CMake** to register the additional source files and tests.
Continue using `Tpetra_ENABLE_VMM_TESTS=ON` and your existing
`MPI_EXEC_PRE_NUMPROCS_FLAGS` / `MPI_EXEC_POST_NUMPROCS_FLAGS` GPU-binding
configuration. The tests use a dedicated `-np 2` MPI launcher.

```sh
cmake --build /path/to/build --target TpetraCore_VmmRegression --parallel
ctest --test-dir /path/to/build -j1 -R VMM_P4_NP2 -V
```

Eight default registrations:

```text
VMM_P4_NP2_BoundaryClassic
VMM_P4_NP2_BoundaryVmmNoStats
VMM_P4_NP2_BoundaryVmmStats
VMM_P4_NP2_DebugClassic
VMM_P4_NP2_DebugVmm
VMM_P4_NP2_TimerClassic
VMM_P4_NP2_TimerVmm
VMM_P4_NP2_FabricProbe
```

`TPETRA_DEBUG=1` appears only in the two debug runs. The ordinary boundary
jobs keep `TPETRA_SYNCHRONIZE_APPLY=0`; timer jobs set it to 1 *only in those
fresh MPI executions*.

An **optional ninth CTest registration**, `VMM_P4_NP2_FabricTimer`, is included
only when `-DTpetra_VMM_P4_ENABLE_FABRIC_TESTS=ON` is explicitly set. Enable
that only after confirming the machine has Fabric VMM handle support and IPC
permissions; the arena currently aborts the MPI job if CUDA driver mapping
calls fail. The capability probe itself is always safe to run.

`VMM_TEST_MAP_MODE=baseline|allocation_only|generalized` is the test-only
expectation switch from P3, propagated into the P4 debug-equivalence test.
`allocation_only` is the M5.1 default; P4's expected debug SpMV path is the
same as the historical baseline until M5.4 removes Map-pointer checks.
`generalized` remains a future acceptance target, not a feature-enabling flag.

## Full standalone sweep

```sh
~/cronus/trilinos/packages/tpetra/core/test/VMM/run_vmm_p4_tests.sh \
  --exe ./packages/tpetra/core/test/VMM/TpetraCore_VmmRegression.exe \
  --gpu-bind "$HOME/cronus/gpu_bind.sh" \
  --outdir ./vmm-p4-results --full --mode allocation_only
```

The script launches eight independent `mpirun --bind-to none -np 2` jobs,
validates exact Teuchos summaries, captures separate rank paths and exception
markers, and saves `results.tsv` and per-scenario logs. `--quick` uses five
jobs. The optional `--fabric` flag probes CUDA driver capability and
conditionally runs an additional Fabric timer test; unsupported Fabric is
explicitly **SKIP**, never silently changed to POSIX. A subsequent Fabric
**mapping/import** failure is a real **FAIL**.

The test-only variables and output markers use the `VMM_TEST_` and `VMM_P4`
namespaces. **No unrecognized `TPETRA_` or `Tpetra_` environment variable is
introduced.** The production variables used by these tests are registered
Tpetra `Behavior` settings.

## Failure interpretation

- B01 or B02 failure on rank 1 indicates a zero-length local view/allocation
  edge case, not necessarily a Map identity problem.
- B03 failure with valid native allocation signals incorrect padded-offset or
  cross-owner addressing.
- G06a/G06b should fail their test **if the expected debug exception did not
  occur**; a caught and validated exception means the test passes.
- T01 checks `Teuchos::Time::numCalls()` deltas. It is not a performance
  benchmark and does not assert any elapsed-time threshold.
- Unsupported Fabric is a capability result, not a regression.

P4 is designed to finish pre-refactor NP=2 test coverage. The separate P5
milestone can extend the suite to 4/8 GPUs and reporting/CI workflows.
