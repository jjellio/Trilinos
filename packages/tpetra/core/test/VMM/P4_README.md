# P4 — Debug Map diagnostics, boundary distributions, timed-path checks

P4 is an in-tree regression group in the existing TpetraCore executable.
M5.3 extended its debug-only tests to validate the VMM Map contract. M5.4
retains those diagnostics while allowing equivalent distinct Maps to dispatch
directly in both debug and release modes. It requires the **corrected P3 installation** (test-only environment
variable `VMM_TEST_MAP_MODE`, not the reserved `TPETRA_VMM_TEST_MAP_MODE`).

M5.4 modifies the release dispatch in `Tpetra_CrsMatrix_def.hpp` and adds a
constant-time physical-layout check in `Tpetra_Details_VmmSpmvContext.hpp`.
The M5.3 debug guard, `MultiVector`, `Behavior`, allocator, and matrix-address
translation are unchanged. As in P0–P3, caller-side fences and communicator barriers in test
fixtures are deliberate; ordinary `apply()` remains asynchronous. A separate
explicit **diagnostic** test uses existing `TPETRA_SYNCHRONIZE_APPLY=1` to
check timer invocation counts, not runtime performance.

## Added tests

| Group / ID | Two-rank fixture | Contract checked |
|---|---|---|
| `VMM_Boundary` B01 | Rank 0 has all 8 GIDs, rank 1 has 0 | Nonzero padded VMM allocation on zero-local-entry rank, storage/content |
| `VMM_Boundary` B02 | Same 8/0 rank distribution, diagonal A | Numerical result, rank agreement on VMM versus classic dispatch (no fixed requirement that the empty-rank configuration take direct VMM) |
| `VMM_Boundary` B03 | Rank 0 owns 1 GID, rank 1 owns 7 | Remote X access, unequal logical sizes, native VMM and direct path when enabled |
| `VMM_Debug` G06a | Wrong X domain Map of the same local length, matrix with Importer | In native-VMM debug, CrsMatrix reports `domain-X Map mismatch` before dispatch; classic-only retains DistObject source-Map diagnostic |
| `VMM_Debug` G06b | Wrong Y range Map of the same local length, matrix with Exporter | With `TPETRA_DEBUG=1`, `DistObject` throws *target Map* diagnostic before MPI transfer |
| `VMM_Debug` G06c | X Map has a different RCP identity but `isSameAs()` | Valid Map accepted under debug; direct VMM dispatch in native mode at M5.4 |
| `VMM_Debug` G07a | Wrong X Map, diagonal matrix with **no Importer** | Native-VMM debug catches domain-X mismatch before the otherwise unchecked classical local SpMV |
| `VMM_Debug` G07b | Wrong Y Map, diagonal matrix with **no Exporter** | Native-VMM debug catches range-Y mismatch before an otherwise unchecked local output kernel |
| `VMM_Debug` G07c | Same per-rank counts and GID set, different local GID order | Semantic Map check detects reordered LIDs rather than mistaking identical physical layout for Map equality |
| `VMM_Debug` G07d | Only rank 1 has a wrong X Map; rank 0 uses the matrix's exact Map object | All ranks reject without `isSameAs()` taking a rank-divergent collective path |
| `VMM_Debug` G07e | Distinct but equivalent X/Y Maps, no Importer or Exporter | Debug accepts valid Map semantics; direct VMM dispatch in native mode at M5.4 |
| `VMM_Debug` G07f | Rank 0 has equivalent uniform X Map; rank 1 has a different explicit X Map | All ranks reject despite different Map::isSameAs() early-return behaviors |
| `VMM_Timer` T01 | Balanced 4/4, all-remote CRS | With opt-in synchronized instrumentation, direct or classic named Teuchos timer increments by exactly one on each rank; analytical result correct |
| `VMM_FabricProbe` F01 | Both GPU ranks, driver-only capability query | Fabric allocation probe succeeds or clearly reports unsupported, without production VMM abort |

The G06 tests retain classic Importer/Exporter diagnostics. G07 invalid-Map
fixtures exercise the cases classic local SpMV may not reject: under VMM native
and `TPETRA_DEBUG=1`, M5.3 checks domain/X and range/Y semantic equivalence
**before the direct kernel**. Classic-only G07 negative operations are
intentionally not executed; they emit `debugSkipped ...
reason=direct_vmm_unavailable`. G07e is a positive, numerical test in both
configurations. The debug guard runs on all ranks, handles mixed Map object
identity, and throws uniform diagnostics only after completing collectives.
Neither Map comparison nor synchronization occurs in normal release SpMV.

## CMake / CTest

Apply `M5_4_dispatch_cleanup.patch` at the Trilinos repository root
**after the validated M5.3 implementation**. Existing P4 CTest registrations are unchanged, but the debug
Teuchos group now contains nine cases; rebuild the regression executable.
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

`VMM_TEST_MAP_MODE=baseline|allocation_only|matrix_generalized|generalized`
is the P3 test-only switch, propagated into P4. For **M5.4**, use
`generalized` as the default. Equivalent-but-distinct Maps are valid and take
direct VMM SpMV when native VMM is enabled. No new `TPETRA_` environment
variable is introduced. The test profile selects expectations, never features.

## Full standalone sweep

```sh
~/cronus/trilinos/packages/tpetra/core/test/VMM/run_vmm_p4_tests.sh \
  --exe ./packages/tpetra/core/test/VMM/TpetraCore_VmmRegression.exe \
  --gpu-bind "$HOME/cronus/gpu_bind.sh" \
  --outdir ./vmm-p4-results --full --mode generalized
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
- G06/G07 invalid-map tests must catch the **specific** CrsMatrix or
  DistObject diagnostic appropriate to that configuration; any other
  exception or no exception is a test failure.
- T01 checks `Teuchos::Time::numCalls()` deltas. It is not a performance
  benchmark and does not assert any elapsed-time threshold.
- Unsupported Fabric is a capability result, not a regression.

P4 also serves as the M5.3/M5.4 semantic-debug acceptance suite for NP=2. The separate P5
milestone can extend the suite to 4/8 GPUs and reporting/CI workflows.
