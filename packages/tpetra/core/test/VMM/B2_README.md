# B2 — Belos VMM-backed MultiVector pool tests

This optional test suite exercises **public** `Belos::MultiVecTraits` methods
against the B1 `MultiVecPool` implementation.  It is located in Tpetra's
experimental VMM test directory; CMake registration is placed in
`packages/belos/tpetra/test/CMakeLists.txt` to avoid a TpetraCore → Belos
reverse dependency.

## Build prerequisites

- B1 rebased patch applied to `packages/belos/tpetra/src/BelosMultiVecTraits_Tpetra.hpp`
- `Trilinos_ENABLE_Tpetra=ON`, `Trilinos_ENABLE_Belos=ON`
- `Belos_ENABLE_Tpetra=ON`, `Belos_ENABLE_TESTS=ON`
- `Tpetra_ENABLE_VMM_TESTS=ON` (the Tpetra test parent declares the option)
- MPI, CUDA, double instantiations: `TPL_ENABLE_MPI=ON`, `Tpetra_INST_CUDA=ON`, `Tpetra_INST_DOUBLE=ON`
- Native VMM built/enabled as in the existing Tpetra VMM regression suite.
- Two distinct CUDA devices, assigned to the two MPI ranks via your existing GPU-binding launcher wrapper.

The single new BelosTpetra executable compiles `VmmBelosPoolTests.cpp` and
reuses `VmmTestMain.cpp` and `VmmP2Helpers.hpp`.  No MueLu dependency.

## Run

```bash
# Discover the TriBITS-prefixed target and exact CTest names
ctest --test-dir /path/to/build -N -R VMM_BelosPool_NP2

# All five two-rank configurations, sequentially
ctest --test-dir /path/to/build -j1 -R VMM_BelosPool_NP2 -V --output-on-failure

# Isolate the VMM+native+statistics case
ctest --test-dir /path/to/build -j1 -R VMM_BelosPool_NP2_VmmStats -V --output-on-failure
```

### Tests

| ID | Test | Contract |
|---|---|---|
| B01 | Clone_OneColumn_Storage | Belos clone, exact Map identity, canonical VMM backing only when both flags enabled |
| B02 | Clone_ReleaseAndReuse | Same whole MV / global VMM pointer on hit; standard mode only promises device storage reuse; stats check one miss followed by one hit |
| B03 | Clone_ConcurrentSlots | Two outstanding clones have distinct storage; after rank-consistent LIFO release the same VMM slot is reused |
| B04 | CloneCopy_DeepCopy | Independent deep copy, recycled CloneCopy, one-column index/range copy overloads |
| B05 | PooledX_DirectSpMV | Two-rank all-remote 8x8 CRS numerical oracle; on stats runs require exactly one `path=vmm` (or `path=classic`) on **each rank** on both applies |
| B06 | Ineligible_PreservesStandardPool | Three-column DV pool reuse; native/global VMM disabled configurations retain the original path |

### CTest configurations

| Name suffix | VMM | Native MV | Statistics | Expected checkout |
|---|---:|---:|---:|---|
| Classic | 0 | 0 | 0 | Standard |
| MatrixOnly | 1 | 0 | 1 | Standard |
| NativeRequestedGlobalOff | 0 | 1 | 1 | Standard |
| VmmStats | 1 | 1 | 1 | VMM pool + verified direct SpMV |
| VmmNoStats | 1 | 1 | 0 | VMM pool; no logging dependency |

All five jobs use `TPETRA_EXPERIMENTAL_VMM_IPC=posix`,
`TPETRA_SYNCHRONIZE_APPLY=0`, and `TPETRA_DEBUG=0`.  The native allocation
and pool diagnostics are gated behind *existing recognized* Behavior flags;
there are no newly introduced `TPETRA_` test-only environment variables.

## Known experimental limitations

The B1 implementation is **rank-local**, and expects identical clone and
release sequencing across MPI ranks.  No intentionally asymmetric cache tests
are included.  Belos subviews and MueLu `replaceMap()` remain separate work.

Assertions about VMM apply path require `TPETRA_EXPERIMENTAL_VMM_STATISTICS=1`.
The no-statistics job tests storage identity and numerical correctness, but
cannot independently assert a logging-disabled dispatch path.  Every cloned
MV is explicitly released before `Tpetra::ScopeGuard`/Kokkos finalize; the
pool's finalize hook then deletes its cached native arenas.

## Validation caveat

The source is prepared for your Trilinos CUDA/MPI build.  The full build and
multi-GPU runtime checks must be performed on your node.  In particular, test
B05 is deliberately strict: numerical correctness through the classic path is
*not* sufficient to pass the VmmStats configuration.
