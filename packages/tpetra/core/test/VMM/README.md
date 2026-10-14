# Tpetra VMM regression tests: P0 harness

This directory is a self-contained test scaffold for **in-tree** Trilinos builds
with the experimental Tpetra CUDA VMM implementation. Nothing here requires an
installed Trilinos. The source changes from milestone 4.1 remain untouched.

## P0 scope

- One custom `Tpetra::ScopeGuard` + `Teuchos::UnitTestRepository` executable.
- GPU binding preflight on *every* invocation, comparing physical GPU PCI bus
  IDs across MPI ranks; rejects duplicate GPUs or CUDA device-query failures.
- A two-rank, eight-row CRS test whose every row references the other rank's X.
  The reference answer is analytical, not a second call to Tpetra.
- An opt-in TriBITS CTest registration (`VMM_Smoke_NP2`, two MPI ranks), plus a
  standalone runner that launches separate jobs for VMM environment settings.
- Rank-global numerical test results, caller-side producer/consumer fencing,
  timeout-protected jobs, and persistent per-case logs + `results.tsv`.

**Not in P0:** proof of direct VMM kernel dispatch, instrumentation changes,
noncontiguous Maps, lifetime tests, debug-negative tests or NP=4/8 tests. Those
are covered by P1–P5 of the test suite plan. Native VMM allocation is checked
from statistics in `vmm_stats`, but **allocation is not proof of dispatch**.

## Integrating with an existing Trilinos checkout

This directory belongs at `packages/tpetra/core/test/VMM/`. The parent
`packages/tpetra/core/test/CMakeLists.txt` receives an opt-in block:

```cmake
OPTION(Tpetra_ENABLE_VMM_TESTS "Build experimental MPI/CUDA VMM tests" OFF)
IF(Tpetra_ENABLE_VMM_TESTS)
  IF(NOT TPL_ENABLE_MPI OR NOT Tpetra_INST_CUDA OR NOT Tpetra_INST_DOUBLE)
    MESSAGE(FATAL_ERROR "Tpetra_ENABLE_VMM_TESTS needs MPI, CUDA and double")
  ENDIF()
  ADD_SUBDIRECTORIES(VMM)
ENDIF()
```

Apply the accompanying `vmm-tpetra-P0.patch` at the Trilinos source root to
install the files and register the directory. It uses only in-tree Tpetra,
Teuchos, Kokkos, and (for PCI identity) the CUDA runtime supplied by the CUDA
configuration. The executable is named `TpetraCore_VmmRegression.exe` in the usual TriBITS TpetraCore subpackage
configuration (with Trilinos' `.exe` suffix convention); inspect the actual build output.

With your existing MPI/CUDA-enabled Trilinos build configuration (and
`TpetraCore` tests enabled, e.g. `Trilinos_ENABLE_TESTS=ON`), add:

```text
-DTpetra_ENABLE_VMM_TESTS=ON
-DMPI_EXEC_PRE_NUMPROCS_FLAGS="--bind-to;none"
-DMPI_EXEC_POST_NUMPROCS_FLAGS="$HOME/cronus/gpu_bind.sh"
```

These global launcher flags also affect *other* TriBITS MPI tests in that build.
Use a dedicated GPU test build, as planned. Keep the C++ VMM enable macro
`TPETRA_ENABLE_EXPERIMENTAL_VMM` configured in the same way you already build
milestone 4.1. `Tpetra_ENABLE_VMM_TESTS` enables only the test registration; it
does not enable the implementation itself.

Build just the test executable (the exact target name may vary with TriBITS
package naming):

```sh
cmake --build /path/to/trilinos-build --target TpetraCore_VmmRegression --parallel
ctest --test-dir /path/to/trilinos-build -R 'VMM_Smoke_NP2' -V
```

The configured CTest MPI launcher must put the GPU binding wrapper between
`-np 2` and the executable, as in:

```sh
mpirun --bind-to none -np 2 \
    "$HOME/cronus/gpu_bind.sh" \
    /path/to/TpetraCore_VmmRegression.exe --group=VMM_Smoke
```

## Standalone sweep

```sh
packages/tpetra/core/test/VMM/run_vmm_tests.sh \
  --exe /path/to/TpetraCore_VmmRegression.exe \
  --gpu-bind "$HOME/cronus/gpu_bind.sh" \
  --outdir ./vmm-p0-results --quick
```

`--full` adds classic+stats, matrix-only, native-requested/global-off, and
full VMM without stats. Use `--help` for other options. Each scenario is a
fresh `mpirun` job because `Tpetra::Details::Behavior` caches environment
variables. All cases use `TPETRA_EXPERIMENTAL_VMM_IPC=posix` and leave
`TPETRA_SYNCHRONIZE_APPLY=0` to preserve normal asynchronous semantics.

A preflight-only invocation is also available:

```sh
mpirun --bind-to none -np 2 "$HOME/cronus/gpu_bind.sh" \
  /path/to/TpetraCore_VmmRegression.exe --vmm-gpu-preflight-only
```

The launcher will fail if it detects the same physical PCI bus ID on multiple
ranks. It does not equate CUDA ordinal `0` in two isolated visible-device lists
with the same physical GPU.

## Reference values and synchronization

Rank 0 owns GIDs `[0,1,2,3]`; rank 1 owns `[4,5,6,7]`. Every row contains
`A(i,i)=2`, `A(i,(i+4)%8)=3`. Initialize `X(g)=g+1`. Then:

```text
rank 0: Y = [17,22,27,32]
rank 1: Y = [13,18,23,28]
```

The fixture writes X on the host, requests a synchronized device read view,
fences Kokkos, and barriers **before** A.apply(). It fences and barriers again
before reading Y. These are *caller-side test synchronization points*. They
must **not** be added to Tpetra's normal `apply()` path.

The test reports a nonzero result if any rank fails its numerical checks. The
script additionally checks for a real Teuchos test summary; in `vmm_stats` it
requires a VMM allocation event. A passing P0 test does not by itself prove
that the direct VMM kernel ran: execution-path assertions are part of P1.

## Local validation notes

P0 can be reviewed and shell-tested without CUDA hardware. Full compilation
and an actual two-GPU MPI run require your CUDA-enabled Trilinos checkout; the
source archive contains changed files, not the complete Trilinos build tree.
