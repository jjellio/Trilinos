TRIBITS_SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR})

# The performance test currently depends on KokkosCore for its timers.
# Otherwise, it does not use anything else in KokkosCore directly.
#
IF(Tpetra_INST_DOUBLE)
  MESSAGE(STATUS "Tpetra: Performance-CGSolve test ENABLED")

  if (NOT Trilinos_ENABLE_Galeri)
    message(FATAL_ERROR
      "Performance-CGSolve requires Galeri. "
      "Configure Trilinos with -DTrilinos_ENABLE_Galeri=ON"
    )
  endif()
 
  if (KokkosKernels_ENABLE_TPL_CUSPARSE)
    set(CG_SOLVE_VARIANT "cusparse")
  else()
    set(CG_SOLVE_VARIANT "NOcusparse")
  endif()
  
  TRIBITS_ADD_EXECUTABLE(
    Performance-CGSolve
    NOEXESUFFIX
    SOURCES cg_solve_file.cpp
    COMM mpi
    CATEGORIES BASIC PERFORMANCE
    ADDED_EXE_TARGET_NAME_OUT CG_SOLVE_TARGET
  )
  
  if (CG_SOLVE_TARGET)
    set_target_properties(
      ${CG_SOLVE_TARGET}
      PROPERTIES
        OUTPUT_NAME "${CG_SOLVE_TARGET}.exe+${CG_SOLVE_VARIANT}"
    )
  endif()
 
  TRIBITS_ADD_EXECUTABLE(
    Performance-CGSolve-fenced
    NOEXESUFFIX
    SOURCES cg_solve_file.cpp
    COMM mpi
    CATEGORIES BASIC PERFORMANCE
    ADDED_EXE_TARGET_NAME_OUT CG_SOLVE_FENCED_TARGET
  )

  if (CG_SOLVE_FENCED_TARGET)
    TARGET_COMPILE_DEFINITIONS(
      ${CG_SOLVE_FENCED_TARGET}
      PRIVATE FENCE_TIMERS=1
    )
    set_target_properties(
      ${CG_SOLVE_FENCED_TARGET}
      PROPERTIES
        OUTPUT_NAME "${CG_SOLVE_FENCED_TARGET}.exe+${CG_SOLVE_VARIANT}"
    )
  endif()
  

  TRIBITS_ADD_EXECUTABLE(
    Performance-CGSolve-shadow
    NOEXESUFFIX
    SOURCES cg_solve_file_shadow.cpp
    COMM mpi
    CATEGORIES BASIC PERFORMANCE
    ADDED_EXE_TARGET_NAME_OUT CG_SOLVE_SHADOW_TARGET
  )

  if (CG_SOLVE_SHADOW_TARGET)
    TARGET_COMPILE_DEFINITIONS(
      ${CG_SOLVE_SHADOW_TARGET}
      PRIVATE FENCE_TIMERS=1
    )
    set_target_properties(
      ${CG_SOLVE_SHADOW_TARGET}
      PROPERTIES
        OUTPUT_NAME "${CG_SOLVE_SHADOW_TARGET}.exe+${CG_SOLVE_VARIANT}"
    )
  endif()


  SET(COUNTER 1)
  SET(MAX_COUNT 11)

  WHILE(COUNTER LESS MAX_COUNT)
    MATH(EXPR NP "${COUNTER} * ${COUNTER}")

    TRIBITS_ADD_TEST(
      ${CG_SOLVE_EXE_NAME}
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
