TRIBITS_SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR})
# Existing Trilinos PerformanceCGSolve CMake target; NVSHMEM P0 adds no new
# TriBITS target and is completely absent in the default OFF configuration.
OPTION(PERFORMANCE_CGSOLVE_NVSHMEM_P0
  "Enable isolated NVSHMEM P0 smoke mode in the existing Performance-CGSolve executable"
  OFF)

IF(PERFORMANCE_CGSOLVE_NVSHMEM_P0)
  IF(NOT Tpetra_ENABLE_CUDA)
    MESSAGE(FATAL_ERROR "PERFORMANCE_CGSOLVE_NVSHMEM_P0 requires Tpetra_ENABLE_CUDA")
  ENDIF()
  IF(NOT Tpetra_INST_DOUBLE)
    MESSAGE(FATAL_ERROR "PERFORMANCE_CGSOLVE_NVSHMEM_P0 requires Tpetra_INST_DOUBLE")
  ENDIF()
  # The normal CG driver remains a C++ file. Only the tiny device kernel is
  # CUDA source; don't force nvcc to compile the large Tpetra CG driver.
  ENABLE_LANGUAGE(CUDA)
  FIND_PACKAGE(CUDAToolkit REQUIRED)
  FIND_PACKAGE(NVSHMEM CONFIG REQUIRED)
  SET(_cg_sources cg_solve_file.cpp NvshmemP0.cu)
ELSE()
  SET(_cg_sources cg_solve_file.cpp)
ENDIF()

IF(Tpetra_INST_DOUBLE)
  MESSAGE(STATUS "Tpetra: Performance-CGSolve test ENABLED")
  TRIBITS_ADD_EXECUTABLE(
    Performance-CGSolve
    SOURCES ${_cg_sources}
    COMM mpi
    CATEGORIES BASIC PERFORMANCE
  )

  IF(PERFORMANCE_CGSOLVE_NVSHMEM_P0)
    # TriBITS names executable targets ${PACKAGE_NAME}_<root-name>.
    SET(_cg_target ${PACKAGE_NAME}_Performance-CGSolve)
    TARGET_COMPILE_DEFINITIONS(${_cg_target} PRIVATE PERFORMANCE_CGSOLVE_NVSHMEM_P0)
    TARGET_LINK_LIBRARIES(${_cg_target} PRIVATE
      nvshmem::nvshmem_host
      nvshmem::nvshmem_device
      CUDA::cudart)
    SET_TARGET_PROPERTIES(${_cg_target} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    MESSAGE(STATUS "Tpetra: NVSHMEM P0 smoke enabled in ${_cg_target}")
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

