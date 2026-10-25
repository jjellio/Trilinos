TRIBITS_SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR})
# The performance test currently depends on KokkosCore for its timers.
# Otherwise, it does not use anything else in KokkosCore directly.
#

# Isolated experiment. OFF preserves the ordinary Trilinos test target.
OPTION(PERFORMANCE_CGSOLVE_NVSHMEM
  "Enable opt-in NVSHMEM P0-P6 Import experiments in Performance-CGSolve"
  ON)

# Hardware-dependent CTest registration is opt-in. Ordinary Trilinos builds
# must not assume that a CUDA GPU or NVSHMEM fabric is available.
OPTION(PERFORMANCE_CGSOLVE_NVSHMEM_P4_CTEST
  "Register single-PE P4 correctness smoke test (needs a CUDA GPU/NVSHMEM runtime)"
  ON)
OPTION(PERFORMANCE_CGSOLVE_NVSHMEM_P51_CTEST
  "Register single-PE P5.1 plan-only smoke test (needs a CUDA GPU/NVSHMEM initialization)"
  ON)
OPTION(PERFORMANCE_CGSOLVE_NVSHMEM_P6_CTEST
  "Register 1-PE P6 timer-wiring smoke (not a transport performance result)"
  OFF)
IF(PERFORMANCE_CGSOLVE_NVSHMEM_P6_CTEST AND NOT PERFORMANCE_CGSOLVE_NVSHMEM)
  MESSAGE(FATAL_ERROR "P6 CTest requires PERFORMANCE_CGSOLVE_NVSHMEM=ON")
ENDIF()
OPTION(PERFORMANCE_CGSOLVE_NVSHMEM_P521_CTEST
  "Register P5.2.1 repeated NVSHMEM-only Import smoke (needs GPU/NVSHMEM)"
  OFF)
IF(PERFORMANCE_CGSOLVE_NVSHMEM_P521_CTEST AND NOT PERFORMANCE_CGSOLVE_NVSHMEM)
  MESSAGE(FATAL_ERROR "P5.2.1 CTest requires PERFORMANCE_CGSOLVE_NVSHMEM=ON")
ENDIF()
OPTION(PERFORMANCE_CGSOLVE_NVSHMEM_P52_CTEST
  "Register single-PE P5.2 point-to-point PUT+signal correctness smoke (CUDA/NVSHMEM)"
  ON)
IF(PERFORMANCE_CGSOLVE_NVSHMEM_P52_CTEST AND NOT PERFORMANCE_CGSOLVE_NVSHMEM)
  MESSAGE(FATAL_ERROR
    "PERFORMANCE_CGSOLVE_NVSHMEM_P52_CTEST requires PERFORMANCE_CGSOLVE_NVSHMEM=ON")
ENDIF()

IF(PERFORMANCE_CGSOLVE_NVSHMEM_P51_CTEST AND NOT PERFORMANCE_CGSOLVE_NVSHMEM)
  MESSAGE(FATAL_ERROR
    "PERFORMANCE_CGSOLVE_NVSHMEM_P51_CTEST requires PERFORMANCE_CGSOLVE_NVSHMEM=ON")
ENDIF()

IF(PERFORMANCE_CGSOLVE_NVSHMEM_P4_CTEST AND NOT PERFORMANCE_CGSOLVE_NVSHMEM)
  MESSAGE(FATAL_ERROR
    "PERFORMANCE_CGSOLVE_NVSHMEM_P4_CTEST requires PERFORMANCE_CGSOLVE_NVSHMEM=ON")
ENDIF()

IF(PERFORMANCE_CGSOLVE_NVSHMEM)
  IF(NOT Tpetra_ENABLE_CUDA OR NOT Tpetra_INST_DOUBLE)
    MESSAGE(FATAL_ERROR
      "PERFORMANCE_CGSOLVE_NVSHMEM requires Tpetra CUDA and double")
  ENDIF()
  # NVSHMEM device API lives in a static device library; a Kokkos CUDA build
  # with relocatable device code is needed for cross-translation-unit linking.
  IF(DEFINED Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
     AND NOT Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
    MESSAGE(FATAL_ERROR
      "NVSHMEM device gets require Kokkos CUDA relocatable device code; "
      "configure Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON")
  ENDIF()
ENDIF()

IF(Tpetra_INST_DOUBLE)
  MESSAGE(STATUS "Tpetra: Performance-CGSolve test ENABLED")

  TRIBITS_ADD_EXECUTABLE(
    Performance-CGSolve
    SOURCES cg_solve_file.cpp
    COMM mpi
    CATEGORIES BASIC PERFORMANCE
  )

  IF(PERFORMANCE_CGSOLVE_NVSHMEM)
    SET(_cg_target ${PACKAGE_NAME}_Performance-CGSolve)
    TARGET_COMPILE_DEFINITIONS(${_cg_target} PRIVATE
      PERFORMANCE_CGSOLVE_NVSHMEM)
    MESSAGE(STATUS "Tpetra: NVSHMEM P0-P6 experiments enabled")
  ENDIF()

  IF(PERFORMANCE_CGSOLVE_NVSHMEM_P4_CTEST)
    TRIBITS_ADD_TEST(
      Performance-CGSolve
      NAME P4_Correctness_Smoke_1PE
      ARGS "--nvshmem-p4 --size=8 --nvshmem-p4-epochs=2 --nvshmem-p4-cycles=2 --nvshmem-p4-max-nodes=1"
      COMM mpi
      NUM_MPI_PROCS 1
      RUN_SERIAL
      CATEGORIES BASIC
    )
  ENDIF()

  IF(PERFORMANCE_CGSOLVE_NVSHMEM_P51_CTEST)
    TRIBITS_ADD_TEST(
      Performance-CGSolve
      NAME P51_PackedPutPlan_Smoke_1PE
      ARGS "--nvshmem-p51 --size=8 --nvshmem-p51-max-nodes=1"
      COMM mpi
      NUM_MPI_PROCS 1
      RUN_SERIAL
      CATEGORIES BASIC
    )
  ENDIF()

  IF(PERFORMANCE_CGSOLVE_NVSHMEM_P52_CTEST)
    TRIBITS_ADD_TEST(
      Performance-CGSolve
      NAME P52_PeerCreditPut_Smoke_1PE
      ARGS "--nvshmem-p52 --size=8 --nvshmem-p52-epochs=4 --nvshmem-p52-cycles=2 --nvshmem-p52-max-nodes=1"
      COMM mpi
      NUM_MPI_PROCS 1
      RUN_SERIAL
      CATEGORIES BASIC
    )
  ENDIF()

  # Single-PE integration smoke for the P5.2.1 stress path. The separate
  # P5.2.1 CTest sweep deliberately includes multi-PE and inter-node cases.
  IF(PERFORMANCE_CGSOLVE_NVSHMEM_P521_CTEST)
    TRIBITS_ADD_TEST(
      Performance-CGSolve
      NAME P521_NVSHMEM_Only_Smoke_1PE
      ARGS "--nvshmem-p52 --nvshmem-p52-case=all --size=8 --nvshmem-p52-epochs=16 --nvshmem-p52-cycles=2 --nvshmem-p52-stress-only"
      COMM mpi
      NUM_MPI_PROCS 1
      RUN_SERIAL
      CATEGORIES BASIC
    )
  ENDIF()

  IF(PERFORMANCE_CGSOLVE_NVSHMEM_P6_CTEST)
    TRIBITS_ADD_TEST(
      Performance-CGSolve
      NAME P6_ImportTiming_Wiring_1PE
      ARGS "--nvshmem-p6 --nvshmem-p6-case=matrix --size=8 --nvshmem-p6-warmup=3 --nvshmem-p6-reps=5 --nvshmem-p6-phase-reps=5 --nvshmem-p6-max-nodes=1"
      COMM mpi
      NUM_MPI_PROCS 1
      RUN_SERIAL
      CATEGORIES BASIC
    )
  ENDIF()

  SET(COUNTER 1)
  SET(MAX_COUNT 11)
  WHILE(COUNTER LESS MAX_COUNT)
    MATH(EXPR NP "${COUNTER} * ${COUNTER}")

    TRIBITS_ADD_TEST(
      Performance-CGSolve
      NAME Performance_StrongScaling_CGSolve
      ARGS "--size=200"
      COMM mpi
      NUM_MPI_PROCS ${NP}
      STANDARD_PASS_OUTPUT
      RUN_SERIAL
      CATEGORIES PERFORMANCE
    )

    MATH(EXPR COUNTER "${COUNTER} + 1")
  ENDWHILE()
  IF(Tpetra_ENABLE_CUDA)
    MESSAGE(STATUS "Tpetra: Performance-CGSolve CUDA_LAUNCH_BLOCKING test ENABLED")
    SET(COUNTER 1)
    SET(MAX_COUNT 11)

    WHILE(COUNTER LESS MAX_COUNT)
      MATH(EXPR NP "${COUNTER} * ${COUNTER}")
      TRIBITS_ADD_TEST(
        Performance-CGSolve
        NAME Performance_StrongScaling_CGSolve_CUDA_LAUNCH_BLOCKING
        ARGS "--size=200"
        COMM mpi
        NUM_MPI_PROCS ${NP}
        ENVIRONMENT CUDA_LAUNCH_BLOCKING=1
        STANDARD_PASS_OUTPUT
        RUN_SERIAL
        CATEGORIES PERFORMANCE
      )

      MATH(EXPR COUNTER "${COUNTER} + 1")
    ENDWHILE()

  ENDIF()

ENDIF()

# Complete P5.2 GPU/cluster correctness sweep. The earlier P5.2 CTest smoke
# covers only one PE. This opt-in suite controls MPI placement with a single
# mpirun per CTest case and thus must NOT be nested inside TRIBITS_ADD_TEST's
# own MPI launcher. Only enable the two-node battery in a real two-node job.
  IF(NOT PERFORMANCE_CGSOLVE_NVSHMEM OR NOT Tpetra_INST_DOUBLE)
    MESSAGE(FATAL_ERROR
      "P5.2 sweep requires PERFORMANCE_CGSOLVE_NVSHMEM=ON and Tpetra double")
  ENDIF()
  IF(NOT TARGET ${PACKAGE_NAME}_Performance-CGSolve)
    MESSAGE(FATAL_ERROR "P5.2 CTest sweep requires the Performance-CGSolve target")
  ENDIF()
  # P52_CTestSweep.cmake obtains this target's actual generated binary path.
  INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/P52_CTestSweep.cmake")
