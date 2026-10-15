# P3 — Distributed Map eligibility and identity regression (NP=2)

P3 extends the P0/P1/P2 suite with **five concrete Map fixtures** and a
test-only four-stage expectation switch: `baseline` (M4.1),
`allocation_only` (M5.1), `matrix_generalized` (M5.2/M5.3 historical),
and `generalized` (**M5.4 default**).
The fixture does not change production behavior; the expected outcomes are
specified in `p3_expected_paths.tsv`.

## Purpose

Validate M5.4's Map-identity-independent direct VMM dispatch on top of M5.2
noncontiguous indexing and M5.3 debug Map diagnostics. Retain the M4.1,
M5.1, and M5.2/M5.3 test profiles for historical comparisons. The new
G03 permutation case exercises both generalizations together.

The modes have different assertions:

- `baseline`: the **historical milestone-4.1 behavior**. Noncontiguous and
  overlapping Maps select standard MV storage; equivalent-but-distinct Maps
  execute classical SpMV. This mode is *not expected to pass* on M5.1 when
  native VMM is enabled, because the allocator policy has intentionally changed.
- `allocation_only` (**M5.1 historical**): G01/G02's noncontiguous Maps and
  G04/G05's overlapping Maps allocate native VMM storage when enabled. The
  matrix still requires contiguous domain Maps, so G01/G02/G05 execute
  classical SpMV with their existing matrix skip reason. G03 remains classic
  for distinct Map objects. The numerical oracle is unchanged.
- `matrix_generalized` (**M5.2/M5.3 historical**): G01/G02 now construct VMM-addressed
  CRS and execute direct VMM SpMV using directory-resolved owner-local LIDs.
  G03 remains classical because `tryApplyVmm()` still checks Map pointer
  identity. G04 remains VMM allocation-only; G05 rejects non-one-to-one
  matrix ownership at `fillComplete()` with
  `reason=domain_map_not_one_to_one` and executes classical SpMV.
- `generalized` (**M5.4 default**): G01/G02 and all G03 cases execute direct
  VMM SpMV, including independently constructed equivalent noncontiguous Maps
  with locally permuted GIDs. G04 allocates VMM storage; G05's non-one-to-one
  domain remains a matrix-side rejection and classical fallback.

No test mode changes library behavior; `VMM_TEST_MAP_MODE` is only consulted
by the C++ fixture and the external runner.

## Fixtures and checks

| Test | Rank 0 GIDs | Rank 1 GIDs | Main assertion |
|---|---|---|---|
| G01_Interleaved | `[0,2,4,6]` | `[1,3,5,7]` | Remote columns, noncontiguous one-to-one Map; direct at M5.2 |
| G02_PermutedLocalGids | `[4,0,6,2]` | `[7,3,1,5]` | Preserve owner-local-index order despite scrambled GIDs |
| G03_EquivalentDistinctMaps | `[0,1,2,3]` | `[4,5,6,7]` | Distinct Map RCP objects whose `isSameAs()` is true; independently test X-only, Y-only, and X/Y identity |
| G03_DistinctPermutedMaps | `[4,0,6,2]` | `[7,3,1,5]` | Distinct but equivalent noncontiguous Maps with locally permuted GIDs and remote reads; tests both M5.2 and M5.4 |
| G04_OverlappingMV | `[0,1,2,3]` | `[3,4,5,6,7]` | Duplicate GID 3, nine global *slots*; MV allocation not inherently one-to-one |
| G05_NonOneToOneDomain | row `[0..3]`, domain `[0..3]` | row `[4..7]`, domain `[3..7]` | Matrix uses overlapping domain (GID 3 duplicated); no direct VMM; diagonal analytical result |

For G01/G02/G03 (including the permuted G03 case), each row has `A(g,g)=2` and `A(g,(g+1)%8)=3`, with
`X(g)=g+1`, so `Y(g)=2*(g+1)+3*(((g+1)%8)+1)`. The successor GID is remote
on **every** G01/G02 row. This catches a wrong owner-local-index mapping,
not just missing communication. G05 uses a diagonal with `A(g,g)=2` and
both copies of the duplicated input GID initialized identically. All six
cases execute in classic and VMM-enabled configurations and check their own
analytical results; G04 tests MV allocation and values, not SpMV.

`Map::isSameAs()` is invoked as a collective test assertion for G03.
M5.3 also invokes semantic Map validation **only under**
`Behavior::debug("CrsMatrix")` for a collective direct-VMM candidate;
release-mode `apply()` has no Map comparisons. Tests also check actual
`isVmmBacked()` and VMM pointer/layout accessors.

## Existing instrumentation

P3 reuses P1's statistics-gated apply path record:

```
[TPETRA_MATRIX] event=apply path=classic rank=0 label="P3_G01_Interleaved"
[TPETRA_MATRIX] event=apply path=classic rank=1 label="P3_G01_Interleaved"
```

Six tests emit **seven** nontranspose apply events per rank: G01, G02, four
G03 variants (X-only, Y-only, X/Y distinct, and permuted), and G05. G04 emits none. The C++ tests check native storage
independently of matrix SpMV path. Both the in-test trace assertions and the
standalone validator check rank-local dispatch. In `generalized`, G01/G02 and G03's permuted case must not emit a matrix VMM skip; G05
must emit `reason=domain_map_not_one_to_one` when VMM is enabled. The
MultiVector allocator never rejects these Maps for contiguity or unique
ownership. With statistics off no apply records are expected.

## Build and CTest

Apply `M5_4_dispatch_cleanup.patch` at the Trilinos source root **after M5.3**, keeping
`Tpetra_ENABLE_VMM_TESTS=ON` and the existing GPU-aware
`MPI_EXEC_PRE_NUMPROCS_FLAGS` / `MPI_EXEC_POST_NUMPROCS_FLAGS`. Reconfigure
CMake to select the `generalized` profile for the existing six CTest registrations, then:

```bash
cmake --build /path/to/build --target TpetraCore_VmmRegression --parallel
ctest --test-dir /path/to/build -j1 -R VMM_P3_NP2 -V
```

This selects **six independent, two-rank CTest jobs**, each running all six
Map tests in a fresh process. The new CMake default is `generalized`.
For an **existing build directory**, explicitly update the cached test mode:

```text
-DTpetra_VMM_P3_MAP_MODE=generalized
```

Use `generalized` for M5.4 acceptance. `matrix_generalized` remains the
historical M5.2/M5.3 oracle and will deliberately fail G03 native-VMM path
checks against M5.4 source.

The suite intentionally requires both distinct GPUs and all-rank execution
with the same Map construction order. All test producer/read fences and MPI
barriers are **caller-side only**; `apply()` remains asynchronous.

## Standalone runner

```bash
~/cronus/trilinos/packages/tpetra/core/test/VMM/run_vmm_p3_tests.sh \
  --exe ./packages/tpetra/core/test/VMM/TpetraCore_VmmRegression.exe \
  --gpu-bind "$HOME/cronus/gpu_bind.sh" \
  --outdir ./vmm-p3-results --full --mode generalized
```

Use `--quick` for classic, full VMM quiet and full VMM+statistics. Full mode
runs all six P0–P2 configuration combinations. Each run uses POSIX IPC,
`TPETRA_SYNCHRONIZE_APPLY=0`, and `TPETRA_DEBUG=0`; no MPI launcher binding
flags need to be changed beyond those already configured.

The historical `baseline`, `allocation_only`, and `matrix_generalized` profiles
are not expected to pass all native-VMM path assertions against M5.4 source. Every scenario writes a log and one TSV status.

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
