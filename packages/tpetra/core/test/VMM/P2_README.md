# P2 — MultiVector allocation/lifecycle and classical fallback regression

P2 is an incremental extension of the working **P0 + P1** source tree. It
adds two Teuchos unit-test groups to `TpetraCore_VmmRegression` and twelve
independent two-rank CTest jobs. It uses the P1 rank-local `event=apply`
diagnostic but **does not modify any production source files**.

## Source files

- `VmmP2Helpers.hpp`: CUDA type aliases shared with P1; two-rank Maps,
  explicit CRS assembly, host/device initialization and numerical oracles,
  caller-side fences/barriers, MV eligibility checks, and rank-local capture
  of P1's apply dispatch event.
- `VmmMultiVectorTests.cpp`: seven tests M01–M07 (below).
- `VmmFallbackTests.cpp`: seven tests F01–F07 (below).
- `verify_vmm_p2_log.py`: checks exactly seven Teuchos test cases per group,
  and the expected apply path on **each** MPI rank when statistics are enabled.
- `run_vmm_p2_tests.sh`: GPU preflight, timeout-protected six-configuration
  environment sweep, split per-suite/per-scenario logs, `results.tsv`.
- `CMakeLists.txt`: adds both sources to the existing executable and registers
  six independent CTest jobs for each test group (12 total).

## MultiVector tests

| Test | Check |
|---|---|
| M01_OneColumnStorage | 1-col CUDA MV allocation agrees with VMM/native flags, global pointer/layout accessors, and initialized values |
| M02_ThreeColumnsStandard | 3-col MV stays standard, with three independently initialized columns |
| M03_NativeFlagOff | native disabled scenario forces standard allocation (not applicable otherwise) |
| M04_GlobalFlagOff | global VMM disabled scenario forces standard allocation (not applicable otherwise) |
| M05_CopyViewAndSubviewLifetime | `Teuchos::View` shares arena, `Teuchos::Copy` owns independent storage, and a subview remains valid when owner and shallow view are destroyed |
| M06_HostDeviceCoherence | Kokkos device kernel writes to native MV's **ordinary** device view; subsequent host and device reads agree after intentional sync |
| M07_RepeatedAllocationAndRelease | 12 create/destroy cycles, two simultaneous arenas, independence and reuse safety |

## Fallback tests

All matrices have eight global GIDs, with `[0,1,2,3]` owned by rank 0 and
`[4,5,6,7]` by rank 1 unless the test explicitly changes the range Map.
The default matrix has `A(i,i)=2` and `A(i,(i+4)%8)=3`, with `X(i)=i+1`.
Numerical results are compared with independent analytical formulas.

| Test | Reason for classic Tpetra path |
|---|---|
| F01_OrdinaryInputAgainstVmmMatrix | X is a one-column subview of a standard three-column MV, while Y may be VMM-backed |
| F02_ThreeColumns | 3-col input and output are unsupported by direct VMM |
| F03_NontrivialExporter | row Map distributed normally, range Map owns the opposite rank's four GIDs: required Exporter |
| F04_LocallyReplicatedRange | range Map locally replicates all eight GIDs on both ranks, requiring classical reduce/export behavior |
| F05_Transpose | explicit nonsymmetric matrix `A(i,(i+4)%8)=3+i`; `A^T x` has a different analytical result from `Ax` |
| F06_AlphaZeroBypass | `alpha=0`, `beta=-2` must scale Y without multiplying X; P1 trace `path=alpha_zero` |
| F07_GlobalVmmDisabled | global VMM disabled forces classical SpMV (not applicable in VMM-enabled scenarios) |

**Path verification:** For F01–F04, and active F07, each statistics-enabled
rank verifies `path=classic`. F06 verifies `path=alpha_zero`. F05 goes
through the existing `applyTranspose()` route, which P1 does not instrument;
it is validated by an independent transpose oracle, **not** by a fabricated
path event. The standalone validator enforces exactly five nontranspose events
per rank in VMM-enabled stats runs, and six per rank when global VMM is off.
When statistics are off, no `event=apply` diagnostics are expected.

## CMake / CTest integration

Apply `vmm-tpetra-P2.patch` from the root of your Trilinos source checkout,
**after** the P1 patch. Your optional six-configuration P0 smoke follow-up
is compatible: this P2 patch targets only P1's stable test-source and
registration lines. It does not alter the existing P0 or P1 tests.

The same in-tree test configuration is used, including
`-DTpetra_ENABLE_VMM_TESTS=ON` and the already-configured MPI GPU binding
flags (`MPI_EXEC_PRE_NUMPROCS_FLAGS` / `MPI_EXEC_POST_NUMPROCS_FLAGS`).

```bash
cmake --build /path/to/build --target TpetraCore_VmmRegression --parallel
ctest --test-dir /path/to/build -j1 -R VMM_P2_NP2 -V
```

The command selects **12 CTest launches**, not two: each Teuchos group runs
independently in each of the six environments below. Each MultiVector test
group has seven cases; each fallback group has seven. These CTest entries
all use `RUN_SERIAL`, `NUM_MPI_PROCS 2`, and a 180-second timeout.

| Environment | VMM | native MV | statistics |
|---|---:|---:|---:|
| Classic | 0 | 0 | 0 |
| ClassicStats | 0 | 0 | 1 |
| MatrixOnly | 1 | 0 | 1 |
| NativeRequestedGlobalOff | 0 | 1 | 1 |
| VmmNoStats | 1 | 1 | 0 |
| VmmStats | 1 | 1 | 1 |

All scenarios use POSIX IPC and `TPETRA_SYNCHRONIZE_APPLY=0`. No new MPI
barriers or Kokkos fences have been added to the library's `apply()`: P2's
fixture adds them **only as downstream caller responsibilities** to safely
stage a new X and complete readers before reusing storage.

## Standalone runner

From the build directory:

```bash
~/cronus/trilinos/packages/tpetra/core/test/VMM/run_vmm_p2_tests.sh \
  --exe ./packages/tpetra/core/test/VMM/TpetraCore_VmmRegression.exe \
  --gpu-bind "$HOME/cronus/gpu_bind.sh" \
  --outdir ./vmm-p2-results --full
```

`--quick` executes the three core scenarios (classic, VMM no stats, VMM
stats); `--full` executes all six. `--suite mv` or `--suite fallback` isolates
one group. The runner's physical GPU preflight is inherited from P0. Each
scenario/suite is launched through its own `mpirun --bind-to none -np 2`
process. Any timeout, Teuchos numerical failure, missing expected event,
or rank-specific fallback divergence returns a nonzero exit code.

## Deferred to P3–P4

The present P2 suite assumes NP=2 and one-to-one contiguous domain Maps.
Noncontiguous maps, equivalent distinct Map identities, overlapping maps,
zero-local-row ranks, and intentionally incompatible debug-only Map pairs
remain P3/P4 work. P2 does not change the VMM memory allocation policy,
dispatch eligibility checks, or the matrix's addressed CRS translation.
