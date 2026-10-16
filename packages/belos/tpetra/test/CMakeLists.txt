INCLUDE(BelosAddTest)

ADD_SUBDIRECTORY(GCRODR)
ADD_SUBDIRECTORY(RCG)
ADD_SUBDIRECTORY(BiCGStab)
ADD_SUBDIRECTORY(BlockCG)
ADD_SUBDIRECTORY(BlockGmres)
ADD_SUBDIRECTORY(TFQMR)
ADD_SUBDIRECTORY(MINRES)
ADD_SUBDIRECTORY(FixedPoint)
ADD_SUBDIRECTORY(LinearSolverFactory)
ADD_SUBDIRECTORY(MultiMatrixSolve)
ADD_SUBDIRECTORY(MultipleSolves)
ADD_SUBDIRECTORY(MVOPTester)
ADD_SUBDIRECTORY(Native)
ADD_SUBDIRECTORY(OrthoManager)
ADD_SUBDIRECTORY(PCPG)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Issue_3235
  SOURCES Issue_3235.cpp
  COMM mpi
  NUM_MPI_PROCS 2
  )


TRIBITS_ADD_EXECUTABLE(
  Issue_14389
  SOURCES Issue_14389.cpp
)


TRIBITS_ADD_Test(
  Issue_14389
  NAME "Issue_14389_pass"
  ARGS
  "--case=0"
  "--case=1"
  COMM mpi
  NUM_MPI_PROCS 1
  )

TRIBITS_ADD_Test(
  Issue_14389
  NAME "Issue_14389_fail"
  ARGS "--case=2"
  WILL_FAIL
  COMM mpi
  NUM_MPI_PROCS 1
  )

# B2: Test source remains under Tpetra/core/test/VMM; registration lives in
# BelosTpetra because TpetraCore must not depend on Belos.  This path is only
# configured when Belos_ENABLE_Tpetra and Belos tests are enabled.
IF(Tpetra_ENABLE_VMM_TESTS AND Belos_ENABLE_Tpetra AND
   TPL_ENABLE_MPI AND Tpetra_INST_CUDA AND Tpetra_INST_DOUBLE)
  SET(BELOS_TPETRA_VMM_TEST_DIR
      "${PROJECT_SOURCE_DIR}/packages/tpetra/core/test/VMM")
  TRIBITS_INCLUDE_DIRECTORIES(${BELOS_TPETRA_VMM_TEST_DIR})

  TRIBITS_ADD_EXECUTABLE(
    VmmBelosPoolRegression
    SOURCES
      ${BELOS_TPETRA_VMM_TEST_DIR}/VmmTestMain.cpp
      ${BELOS_TPETRA_VMM_TEST_DIR}/VmmBelosPoolTests.cpp
    DEPLIBS belostpetra
    COMM mpi
  )

  MACRO(BELOS_TPETRA_REGISTER_VMM_POOL_CASE caseName vmmEnabled nativeMv statsEnabled)
    TRIBITS_ADD_TEST(
      VmmBelosPoolRegression
      NAME VMM_BelosPool_NP2_${caseName}
      ARGS "--group=VMM_BelosPool --globally-reduce-test-result"
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

  # Behavior flags are cached. Each scenario must be a fresh MPI process.
  BELOS_TPETRA_REGISTER_VMM_POOL_CASE(Classic                   0 0 0)
  BELOS_TPETRA_REGISTER_VMM_POOL_CASE(MatrixOnly                1 0 1)
  BELOS_TPETRA_REGISTER_VMM_POOL_CASE(NativeRequestedGlobalOff  0 1 1)
  BELOS_TPETRA_REGISTER_VMM_POOL_CASE(VmmStats                  1 1 1)
  BELOS_TPETRA_REGISTER_VMM_POOL_CASE(VmmNoStats                1 1 0)
ENDIF()
