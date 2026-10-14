# Experimental VMM Tpetra regression tests.  This directory is only included
# by the parent when CUDA, MPI and double ETI are enabled and the opt-in test
# switch is ON.  It is compiled inside the normal Trilinos/Tpetra build.

TRIBITS_ADD_EXECUTABLE(
  VmmRegression
  SOURCES VmmTestMain.cpp VmmSmokeTests.cpp
  COMM mpi
)

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
