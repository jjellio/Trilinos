# Experimental VMM Tpetra regression tests.  This directory is only included
# by the parent when CUDA, MPI and double ETI are enabled and the opt-in test
# switch is ON.  It is compiled inside the normal Trilinos/Tpetra build.

TRIBITS_ADD_EXECUTABLE(
  VmmRegression
  SOURCES VmmTestMain.cpp VmmSmokeTests.cpp VmmSpmvTests.cpp
          VmmMultiVectorTests.cpp VmmFallbackTests.cpp
          VmmMapTests.cpp
  COMM mpi
)

# P1 runs a different Teuchos test group in the SAME executable.  Each
# configuration is a fresh MPI process because Behavior caches environment
# variables on first use.  These registrations are independent of the P0
# follow-up patch that adds five more VMM_Smoke_NP2 cases.
MACRO(TPETRA_VMM_REGISTER_P1_CASE caseName vmmEnabled nativeMv statsEnabled)
  TRIBITS_ADD_TEST(
    VmmRegression
    NAME VMM_P1_NP2_${caseName}
    ARGS "--group=VMM_SpMV --globally-reduce-test-result"
    COMM mpi
    NUM_MPI_PROCS 2
    ENVIRONMENT
      "TPETRA_EXPERIMENTAL_VMM=${vmmEnabled}"
      "TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=${nativeMv}"
      "TPETRA_EXPERIMENTAL_VMM_STATISTICS=${statsEnabled}"
      "TPETRA_EXPERIMENTAL_VMM_IPC=posix"
      "TPETRA_SYNCHRONIZE_APPLY=0"
      "TPETRA_DEBUG=0"
    RUN_SERIAL
    TIMEOUT 120
    STANDARD_PASS_OUTPUT
  )
ENDMACRO()

# All six environment configurations also appear in CTest, so a user can
# run the entire P1 matrix with `ctest -j1 -R VMM_P1_NP2 -V`.
# The standalone runner additionally verifies exact per-rank dispatch counts.
TPETRA_VMM_REGISTER_P1_CASE(Classic                   0 0 0)
TPETRA_VMM_REGISTER_P1_CASE(ClassicStats              0 0 1)
TPETRA_VMM_REGISTER_P1_CASE(MatrixOnly                1 0 1)
TPETRA_VMM_REGISTER_P1_CASE(NativeRequestedGlobalOff  0 1 1)
TPETRA_VMM_REGISTER_P1_CASE(VmmNoStats                1 1 0)
TPETRA_VMM_REGISTER_P1_CASE(VmmStats                  1 1 1)

# P2: two Teuchos groups, each independently launched under all six
# environment settings.  RUN_SERIAL prevents accidental overlapping jobs
# on the same two GPUs.  Use `ctest -j1 -R VMM_P2_NP2 -V` for the complete
# allocation/lifetime and classical-fallback regression matrix.
MACRO(TPETRA_VMM_REGISTER_P2_CASE suite group caseName vmmEnabled nativeMv statsEnabled)
  TRIBITS_ADD_TEST(
    VmmRegression
    NAME VMM_P2_NP2_${suite}_${caseName}
    ARGS "--group=${group} --globally-reduce-test-result"
    COMM mpi
    NUM_MPI_PROCS 2
    ENVIRONMENT
      "TPETRA_EXPERIMENTAL_VMM=${vmmEnabled}"
      "TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=${nativeMv}"
      "TPETRA_EXPERIMENTAL_VMM_STATISTICS=${statsEnabled}"
      "TPETRA_EXPERIMENTAL_VMM_IPC=posix"
      "TPETRA_SYNCHRONIZE_APPLY=0"
      "TPETRA_DEBUG=0"
    RUN_SERIAL
    TIMEOUT 180
    STANDARD_PASS_OUTPUT
  )
ENDMACRO()

MACRO(TPETRA_VMM_REGISTER_P2_GROUP suite group)
  TPETRA_VMM_REGISTER_P2_CASE(${suite} ${group} Classic                  0 0 0)
  TPETRA_VMM_REGISTER_P2_CASE(${suite} ${group} ClassicStats             0 0 1)
  TPETRA_VMM_REGISTER_P2_CASE(${suite} ${group} MatrixOnly               1 0 1)
  TPETRA_VMM_REGISTER_P2_CASE(${suite} ${group} NativeRequestedGlobalOff 0 1 1)
  TPETRA_VMM_REGISTER_P2_CASE(${suite} ${group} VmmNoStats               1 1 0)
  TPETRA_VMM_REGISTER_P2_CASE(${suite} ${group} VmmStats                 1 1 1)
ENDMACRO()

TPETRA_VMM_REGISTER_P2_GROUP(MultiVector VMM_MultiVector)
TPETRA_VMM_REGISTER_P2_GROUP(Fallback VMM_Fallback)

# P3 Map-generalization *test* mode.  This is not a production VMM flag.
# Default baseline asserts M4.1 behavior; generalized defines acceptance
# criteria for the later Map refactor.  Reconfigure CMake to switch mode.
SET(Tpetra_VMM_P3_MAP_MODE "baseline" CACHE STRING
    "VMM P3 Map test mode: baseline or generalized")
SET_PROPERTY(CACHE Tpetra_VMM_P3_MAP_MODE PROPERTY STRINGS baseline generalized)
IF(NOT Tpetra_VMM_P3_MAP_MODE STREQUAL "baseline" AND
   NOT Tpetra_VMM_P3_MAP_MODE STREQUAL "generalized")
  MESSAGE(FATAL_ERROR "Tpetra_VMM_P3_MAP_MODE must be baseline or generalized")
ENDIF()

MACRO(TPETRA_VMM_REGISTER_P3_CASE caseName vmmEnabled nativeMv statsEnabled)
  TRIBITS_ADD_TEST(
    VmmRegression
    NAME VMM_P3_NP2_${caseName}
    ARGS "--group=VMM_Map --globally-reduce-test-result"
    COMM mpi
    NUM_MPI_PROCS 2
    ENVIRONMENT
      "TPETRA_EXPERIMENTAL_VMM=${vmmEnabled}"
      "TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=${nativeMv}"
      "TPETRA_EXPERIMENTAL_VMM_STATISTICS=${statsEnabled}"
      "TPETRA_EXPERIMENTAL_VMM_IPC=posix"
      "TPETRA_SYNCHRONIZE_APPLY=0"
      "TPETRA_DEBUG=0"
      "TPETRA_VMM_TEST_MAP_MODE=${Tpetra_VMM_P3_MAP_MODE}"
    RUN_SERIAL
    TIMEOUT 180
    STANDARD_PASS_OUTPUT
  )
ENDMACRO()

# Six separate MPI launches exercise all map fixtures.  Every configuration
# reruns the five Teuchos tests with an independent Behavior environment.
TPETRA_VMM_REGISTER_P3_CASE(Classic                   0 0 0)
TPETRA_VMM_REGISTER_P3_CASE(ClassicStats              0 0 1)
TPETRA_VMM_REGISTER_P3_CASE(MatrixOnly                1 0 1)
TPETRA_VMM_REGISTER_P3_CASE(NativeRequestedGlobalOff  0 1 1)
TPETRA_VMM_REGISTER_P3_CASE(VmmNoStats                1 1 0)
TPETRA_VMM_REGISTER_P3_CASE(VmmStats                  1 1 1)

# The configured TriBITS MPI launcher may supply:
#   MPI_EXEC_PRE_NUMPROCS_FLAGS  = --bind-to;none
#   MPI_EXEC_POST_NUMPROCS_FLAGS = /path/to/gpu_bind.sh
# All CTest jobs use separate GPUs and should not overlap with other GPU tests.
TRIBITS_ADD_TEST(
  VmmRegression
  NAME VMM_Smoke_NP2
  ARGS "--group=VMM_Smoke --globally-reduce-test-result"
  COMM mpi
  NUM_MPI_PROCS 2
  ENVIRONMENT
    "TPETRA_EXPERIMENTAL_VMM=1"
    "TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=1"
    "TPETRA_EXPERIMENTAL_VMM_STATISTICS=1"
    "TPETRA_EXPERIMENTAL_VMM_IPC=posix"
    "TPETRA_SYNCHRONIZE_APPLY=0"
    "TPETRA_DEBUG=0"
  RUN_SERIAL
  TIMEOUT 120
  STANDARD_PASS_OUTPUT
)

# The original VMM_Smoke_NP2 registration above is the VMM+native-MV+stats
# case.  Register the five other configurations from run_vmm_tests.sh --full
# as separate MPI launches: Tpetra::Details::Behavior caches environment
# variables, so they cannot be swept inside one test process.
#
# Every name starts with VMM_Smoke_NP2 so the existing command
#     ctest -j1 -R VMM_Smoke_NP2 -V
# now runs the entire six-configuration smoke matrix.
MACRO(TPETRA_VMM_REGISTER_SMOKE_CASE caseName vmmEnabled nativeMv statsEnabled)
  TRIBITS_ADD_TEST(
    VmmRegression
    NAME VMM_Smoke_NP2_${caseName}
    ARGS "--group=VMM_Smoke --globally-reduce-test-result"
    COMM mpi
    NUM_MPI_PROCS 2
    ENVIRONMENT
      "TPETRA_EXPERIMENTAL_VMM=${vmmEnabled}"
      "TPETRA_EXPERIMENTAL_VMM_NATIVE_MV=${nativeMv}"
      "TPETRA_EXPERIMENTAL_VMM_STATISTICS=${statsEnabled}"
      "TPETRA_EXPERIMENTAL_VMM_IPC=posix"
      "TPETRA_SYNCHRONIZE_APPLY=0"
      "TPETRA_DEBUG=0"
    RUN_SERIAL
    TIMEOUT 120
    STANDARD_PASS_OUTPUT
  )
ENDMACRO()

TPETRA_VMM_REGISTER_SMOKE_CASE(Classic                   0 0 0)
TPETRA_VMM_REGISTER_SMOKE_CASE(ClassicStats              0 0 1)
TPETRA_VMM_REGISTER_SMOKE_CASE(MatrixOnly                1 0 1)
TPETRA_VMM_REGISTER_SMOKE_CASE(NativeRequestedGlobalOff  0 1 1)
TPETRA_VMM_REGISTER_SMOKE_CASE(VmmNoStats                1 1 0)
