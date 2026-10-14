# P1 — Numerical SpMV and dispatch regression tests

P1 extends the already-working P0 in-tree Teuchos/TriBITS test executable.
It includes analytic numerical references for S01–S04 and S06, covering both
local and remote X reads. No random matrices, Matrix Market files, downstream
install, or extra Trilinos dependency is introduced.

## Installation and scope

Apply `vmm-tpetra-P1.patch` **after the original P0 patch** to the same Trilinos
checkout. The optional P0 six-configuration CTest follow-up patch can already
be present or absent; the P1 registration is independent of it. In-tree
configuration stays `-DTpetra_ENABLE_VMM_TESTS=ON` with the previously working
MPI launcher flags. Build target `TpetraCore_VmmRegression` as before.

P1 changes exactly one production file:
`packages/tpetra/core/src/Tpetra_CrsMatrix_def.hpp`, adding an event emitted
for every rank on non-transpose `applyNonTranspose()`:

```
[TPETRA_MATRIX] event=apply path=vmm rank=0 label="P1_S03_AllRemote"
[TPETRA_MATRIX] event=apply path=vmm rank=1 label="P1_S03_AllRemote"
```

Other possible `path` values are `classic` and `alpha_zero` (the latter is the
ordinary Tpetra alpha==0 early return, not a SpMV dispatch). Emission occurs
**only** when the experimental VMM code is compiled and
`TPETRA_EXPERIMENTAL_VMM_STATISTICS=1`. It performs no MPI collective, CUDA
fence, or barrier. No `tryApplyVmm()` eligibility conditions were changed.
The trace is diagnostic instrumentation and should **not** be enabled when
measuring SpMV performance.

Each statistics-enabled P1 unit test captures its **own rank's** event during
`apply()`, verifies the expected path, and then replays the event for human
logs and the standalone runner. Thus a CTest numerical pass in stats mode
also proves execution-path selection on both ranks. With statistics off,
`isVmmBacked()` and the analytical numerical result are checked, but no
`event=apply` output is expected.

## Cases

| Teuchos test | Rank-local CRS construction | Analytical reference |
|---|---|---|
| `VMM_SpMV_S01_Diagonal` | `A(i,i)=i+2` | `(i+2)(i+1)` |
| `VMM_SpMV_S02_Boundary` | `A(i,i)=2`, `A(3,4)=A(4,3)=3` | two boundary cross-GPU accesses |
| `VMM_SpMV_S03_AllRemote` | `A(i,i)=2`, `A(i,(i+4)%8)=3` | rank 0 `[17,22,27,32]`; rank 1 `[13,18,23,28]` |
| `VMM_SpMV_S04_Stencil` | tridiagonal `4,-1,-1` | `4*x(i)-x(i-1)-x(i+1)` with endpoint terms omitted |
| `VMM_SpMV_S06_AlphaBeta` | reuse S03 | `(alpha,beta)`: `(1,0)`, `(2,0)`, `(1,1)`, `(-0.5,2)`, `(0,-1)`, `(0,0)` |

`x(i)=i+1` and `oldY(i)=7+i` for all cases. When `beta=0`, Y is deliberately
filled with NaN before SpMV, verifying that the BLAS `beta=0` implementation
never reads stale output values. All tests use exactly 8 global rows with rank
0 GIDs `[0,1,2,3]` and rank 1 GIDs `[4,5,6,7]`. Each test independently
computes expected output without invoking a second `CrsMatrix::apply()`.

The test *caller* fences and barriers before reading remote X and after SpMV.
The library's `apply()` remains asynchronous and does not gain a barrier or
fence. Tests compute on the CUDA Node regardless of other instantiations in
the same Trilinos build.

## Run via existing TriBITS / CTest integration

```bash
cmake --build /path/to/trilinos-build --target TpetraCore_VmmRegression --parallel
ctest --test-dir /path/to/trilinos-build -j1 -R VMM_P1_NP2 -V
```

Six independent CTest launches appear, one per scenario:

| Scenario suffix | VMM | native MV | statistics | Expected S01–S04/S06 nonzero-alpha path |
|---|---:|---:|---:|---|
| `Classic` | 0 | 0 | 0 | classical |
| `ClassicStats` | 0 | 0 | 1 | classical |
| `MatrixOnly` | 1 | 0 | 1 | classical |
| `NativeRequestedGlobalOff` | 0 | 1 | 1 | classical |
| `VmmNoStats` | 1 | 1 | 0 | VMM, without event messages |
| `VmmStats` | 1 | 1 | 1 | direct VMM, per-rank event validated |

The P0 tests remain registered and still use `--group=VMM_Smoke`.
The P1 tests use `--group=VMM_SpMV`. Both groups reside in the same executable.

## Full standalone verification

```bash
packages/tpetra/core/test/VMM/run_vmm_p1_tests.sh \
  --exe /path/to/build/packages/tpetra/core/test/VMM/TpetraCore_VmmRegression.exe \
  --gpu-bind "$HOME/cronus/gpu_bind.sh" \
  --outdir ./vmm-p1-results --full
```

`--quick` runs three core configurations; `--full` runs all six. This runner
uses your demonstrated MPI command shape (`mpirun --bind-to none -np 2
<gpu_bind.sh> <exe>`), starts each environment scenario in its own process,
verifies physical GPU uniqueness through P0's preflight, stores separate logs,
and summarizes results in `results.tsv`.

`verify_vmm_p1_log.py` checks that Teuchos ran/passed exactly five P1 tests,
then checks every expected path event **per rank and per matrix**:

- Four single SpMV tests (`S01`, `S02`, `S03`, `S04`), once each.
- Four nonzero-alpha `S06` calls.
- Two `S06` `alpha_zero` calls (never counted as direct SpMV).

Thus for a stats-enabled run, each MPI rank must generate exactly eight
matching `path=vmm` or `path=classic` events plus two `path=alpha_zero`
events. All unexpected paths, missing events, and wrong-rank events fail.
For stats-off runs, any `event=apply` trace is considered an error.

## Validation status

P1 was syntax-checked for shell/Python, registered through a synthetic TriBITS
CMake harness, and the runner was exercised with simulated two-rank logs,
including an injected rank-specific path mismatch which it correctly rejected.
An actual in-tree C++ build and execution on two CUDA GPUs must be performed
on the target Trilinos machine; the attached source snapshot is not a complete
Trilinos checkout or CUDA/MPI build.

P1 deliberately does *not* add S05 (uneven ownership), S07 (repeated apply
with changing X), S08 (independent X arena lifetime), map-generalization
cases, or fallback/lifecycle tests. They remain P2–P4 work items.
