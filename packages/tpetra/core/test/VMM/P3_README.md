# P3 — Distributed Map eligibility and identity regression (NP=2)

P3 extends the user-validated P0/P1/P2 test suite with **five concrete Map
fixtures** and a test-only three-stage expectation switch: `baseline` (M4.1),
`allocation_only` (M5.1, current default), and `generalized` (final target).
The fixture does not change production behavior; the expected outcomes are
specified in `p3_expected_paths.tsv`.

## Purpose

Distinguish the implemented M5.1 MV allocation policy from the still
contiguous-only matrix indexing policy, while retaining the historical M4.1
baseline and the future M5.2–M5.4 direct-SpMV acceptance requirements.
Also validate that equivalent, separately constructed Maps eventually use
direct VMM dispatch without raw-pointer Map comparisons.

The modes have different assertions:

- `baseline`: the **historical milestone-4.1 behavior**. Noncontiguous and
  overlapping Maps select standard MV storage; equivalent-but-distinct Maps
  execute classical SpMV. This mode is *not expected to pass* on M5.1 when
  native VMM is enabled, because the allocator policy has intentionally changed.
- `allocation_only` (**M5.1 default**): G01/G02's noncontiguous Maps and
  G04/G05's overlapping Maps allocate native VMM storage when enabled. The
  matrix still requires contiguous domain Maps, so G01/G02/G05 execute
  classical SpMV with their existing matrix skip reason. G03 remains classic
  for distinct Map objects. The numerical oracle is unchanged.
- `generalized`: the **future M5 acceptance specification**. G01/G02 and G03
  execute direct VMM SpMV; G04 allocates VMM storage, while G05's non-one-to-one
  domain remains a matrix-side rejection and classical fallback.

No test mode changes library behavior; `VMM_TEST_MAP_MODE` is only consulted
by the C++ fixture and the external runner.

## Fixtures and checks

| Test | Rank 0 GIDs | Rank 1 GIDs | Main assertion |
|---|---|---|---|
| G01_Interleaved | `[0,2,4,6]` | `[1,3,5,7]` | Remote columns, noncontiguous one-to-one Map; classic now / direct later |
| G02_PermutedLocalGids | `[4,0,6,2]` | `[7,3,1,5]` | Preserve owner-local-index order despite scrambled GIDs |
| G03_EquivalentDistinctMaps | `[0,1,2,3]` | `[4,5,6,7]` | Distinct Map RCP objects whose `isSameAs()` is true; independently test X-only, Y-only, and X/Y mismatches |
| G04_OverlappingMV | `[0,1,2,3]` | `[3,4,5,6,7]` | Duplicate GID 3, nine global *slots*; MV allocation not inherently one-to-one |
| G05_NonOneToOneDomain | row `[0..3]`, domain `[0..3]` | row `[4..7]`, domain `[3..7]` | Matrix uses overlapping domain (GID 3 duplicated); no direct VMM; diagonal analytical result |

For G01/G02/G03, each row has `A(g,g)=2` and `A(g,(g+1)%8)=3`, with
`X(g)=g+1`, so `Y(g)=2*(g+1)+3*(((g+1)%8)+1)`. The successor GID is remote
on **every** G01/G02 row. This catches a wrong owner-local-index mapping,
not just missing communication. G05 uses a diagonal with `A(g,g)=2` and
both copies of the duplicated input GID initialized identically. All five
cases execute in classic and VMM-enabled configurations and check their own
analytical results; G04 tests MV allocation and values, not SpMV.

`Map::isSameAs()` is invoked **only** as a collective test assertion for
G03 and never within the library SpMV path. Tests also check actual
`isVmmBacked()` and VMM pointer/layout accessors.

## Existing instrumentation

P3 reuses P1's statistics-gated apply path record:

```
[TPETRA_MATRIX] event=apply path=classic rank=0 label="P3_G01_Interleaved"
[TPETRA_MATRIX] event=apply path=classic rank=1 label="P3_G01_Interleaved"
```

Five tests emit **six** nontranspose apply events per rank: G01, G02, three
G03 subcases, and G05. G04 emits none. The C++ tests check native storage
independently of matrix SpMV path. Both the in-test trace assertions and the
standalone validator check rank-local dispatch. In `allocation_only`, the
matrix still emits `reason=map_not_contiguous` for G01/G02/G05, but the
MultiVector allocator must no longer reject those Maps with that reason or
`map_not_one_to_one`. The later desired matrix-side G05 reason is
`domain_map_not_one_to_one` (M5.2). With statistics off no apply records are
expected.

## Build and CTest

Apply `vmm-tpetra-P3.patch` at the Trilinos source root **after P2**, keeping
`Tpetra_ENABLE_VMM_TESTS=ON` and the existing GPU-aware
`MPI_EXEC_PRE_NUMPROCS_FLAGS` / `MPI_EXEC_POST_NUMPROCS_FLAGS`. Reconfigure
CMake to discover `VmmMapTests.cpp` and the six new CTest registrations, then:

```bash
cmake --build /path/to/build --target TpetraCore_VmmRegression --parallel
ctest --test-dir /path/to/build -j1 -R VMM_P3_NP2 -V
```

This selects **six independent, two-rank CTest jobs**, each running all five
Map tests in a fresh process. The new CMake default is `allocation_only`.
For an **existing build directory**, explicitly update the cached test mode:

```text
-DTpetra_VMM_P3_MAP_MODE=allocation_only
```

After M5.2–M5.4, change the test oracle to `generalized`; it must not be used
as an M5.1 pass criterion.

The suite intentionally requires both distinct GPUs and all-rank execution
with the same Map construction order. All test producer/read fences and MPI
barriers are **caller-side only**; `apply()` remains asynchronous.

## Standalone runner

```bash
~/cronus/trilinos/packages/tpetra/core/test/VMM/run_vmm_p3_tests.sh \
  --exe ./packages/tpetra/core/test/VMM/TpetraCore_VmmRegression.exe \
  --gpu-bind "$HOME/cronus/gpu_bind.sh" \
  --outdir ./vmm-p3-results --full --mode allocation_only
```

Use `--quick` for classic, full VMM quiet and full VMM+statistics. Full mode
runs all six P0–P2 configuration combinations. Each run uses POSIX IPC,
`TPETRA_SYNCHRONIZE_APPLY=0`, and `TPETRA_DEBUG=0`; no MPI launcher binding
flags need to be changed beyond those already configured.

Use `--mode generalized` only after matrix indexing and dispatch are updated.
The historical `baseline` profile intentionally does not represent M5.1's
native-MV storage behavior. Every scenario writes a log and one TSV status.

## Scope / caveats

G05 uses a deliberately non-one-to-one domain Map to establish matrix-side
VMM ineligibility. This is an uncommon mathematical operator configuration,
not a recommended way to build typical Tpetra systems. The fixture only
accesses its local diagonal and initializes both occurrences of the shared
GID identically. If classic Tpetra itself rejects such a domain Map during
`fillComplete()`, that is a distinct API-contract failure, and the test must
be revisited rather than weakening its acceptance criteria to a silent skip.

No new production VMM eligibility checks or Map comparison logic are in P3.
The tests need actual Trilinos/CUDA/MPI compilation and two-GPU execution on
your system; mock launcher and validator checks cannot establish GPU behavior.
