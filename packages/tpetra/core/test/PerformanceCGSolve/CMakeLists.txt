TRIBITS_SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR})
# A single opt-in NVSHMEM benchmark mode. Keep ordinary CG and its scaling
# registrations unchanged. The enabled default matches the supplied checkout.
OPTION(PERFORMANCE_CGSOLVE_NVSHMEM
  "Enable the opt-in NVSHMEM P6.2 Import reference (and future P7)"
  ON)
OPTION(PERFORMANCE_CGSOLVE_NVSHMEM_P62_CTEST
  "Register 1-PE P6.2 correctness smoke (not a transport measurement)"
  OFF)
OPTION(PERFORMANCE_CGSOLVE_NVSHMEM_P71_CTEST
  "Register 1-PE P7.1 instrumented GPU probe smoke (not a transport result)"
  OFF)
OPTION(PERFORMANCE_CGSOLVE_IMPORT_TOPOLOGY_CTEST
  "Register a 1-PE matrix-free Import topology smoke test (no off-node performance data)"
  OFF)
IF(PERFORMANCE_CGSOLVE_IMPORT_TOPOLOGY_CTEST AND NOT PERFORMANCE_CGSOLVE_NVSHMEM)
  MESSAGE(FATAL_ERROR "Import topology CTest requires PERFORMANCE_CGSOLVE_NVSHMEM=ON")
ENDIF()
IF(PERFORMANCE_CGSOLVE_NVSHMEM_P71_CTEST AND NOT PERFORMANCE_CGSOLVE_NVSHMEM)
  MESSAGE(FATAL_ERROR "P7.1 CTest requires PERFORMANCE_CGSOLVE_NVSHMEM=ON")
ENDIF()
IF(PERFORMANCE_CGSOLVE_NVSHMEM_P62_CTEST AND NOT PERFORMANCE_CGSOLVE_NVSHMEM)
  MESSAGE(FATAL_ERROR "P6.2 CTest requires PERFORMANCE_CGSOLVE_NVSHMEM=ON")
ENDIF()

IF(PERFORMANCE_CGSOLVE_NVSHMEM)
  IF(NOT Tpetra_ENABLE_CUDA OR NOT Tpetra_INST_DOUBLE)
    MESSAGE(FATAL_ERROR
      "PERFORMANCE_CGSOLVE_NVSHMEM requires Tpetra CUDA and double")
  ENDIF()
  IF(DEFINED Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
     AND NOT Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
    MESSAGE(FATAL_ERROR
      "NVSHMEM GPU operations require Kokkos CUDA relocatable device code; "
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
    MESSAGE(STATUS "Tpetra: NVSHMEM P6.2 Import reference enabled")
  ENDIF()

  IF(PERFORMANCE_CGSOLVE_NVSHMEM)
    # P7.1b: separate executable so the CG solver CLI remains clean.
    # Reuse the frozen P6.2 and P7.1 kernel/benchmark headers, not the
    # MiniFE matrix or the actual CG solver.
    TRIBITS_ADD_EXECUTABLE(
      ImportTopologyBenchmark
      SOURCES NvshmemImportTopologyBenchmark.cpp
      COMM mpi
      CATEGORIES BASIC PERFORMANCE
    )
    SET(_topology_target ${PACKAGE_NAME}_ImportTopologyBenchmark)
    TARGET_COMPILE_DEFINITIONS(${_topology_target} PRIVATE
      PERFORMANCE_CGSOLVE_NVSHMEM)
  ENDIF()
  IF(PERFORMANCE_CGSOLVE_IMPORT_TOPOLOGY_CTEST)
    TRIBITS_ADD_TEST(
      ImportTopologyBenchmark
      NAME MatrixFree_LocalOnly_1PE
      ARGS "--numgpus=1 --owned=1024 --offnode-degree=0 --intranode-degree=0 --local-values=128 --target-layout=interleave --values-per-peer=0 --mode=direct --chunk=128 --put-variant=warp --warmup=2 --reps=5 --verify-every=1 --expected-ppn=1 --min-nodes=1 --max-nodes=1"
      COMM mpi
      NUM_MPI_PROCS 1
      RUN_SERIAL
      CATEGORIES BASIC
    )
  ENDIF()

  IF(PERFORMANCE_CGSOLVE_NVSHMEM_P62_CTEST)
    TRIBITS_ADD_TEST(
      Performance-CGSolve
      NAME P62_DescriptorTwoStream_Smoke_1PE
      ARGS "--cuda --numgpus=1 --nvshmem-p62 --nvshmem-p62-case=all --size=8 --nvshmem-p62-put-variant=warp --nvshmem-p62-chunk=128 --nvshmem-p62-verify-every=1 --nvshmem-p62-warmup=2 --nvshmem-p62-reps=5 --nvshmem-p62-max-nodes=1"
      COMM mpi
      NUM_MPI_PROCS 1
      RUN_SERIAL
      CATEGORIES BASIC
    )
  ENDIF()

  IF(PERFORMANCE_CGSOLVE_NVSHMEM_P71_CTEST)
    TRIBITS_ADD_TEST(
      Performance-CGSolve
      NAME P71_CUDA_Probe_Smoke_1PE
      ARGS "--cuda --numgpus=1 --nvshmem-p62 --nvshmem-p62-mode=p71 --nvshmem-p62-case=all --size=8 --nvshmem-p62-put-variant=warp --nvshmem-p62-chunk=128 --nvshmem-p62-verify-every=1 --nvshmem-p62-warmup=2 --nvshmem-p62-reps=5 --nvshmem-p62-p71-reps=2 --nvshmem-p62-p71-max-samples=8 --nvshmem-p62-out=p71-smoke --nvshmem-p62-max-nodes=1"
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
