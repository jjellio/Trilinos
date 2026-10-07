TRIBITS_SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR})

# The performance test currently depends on KokkosCore for its timers.
# Otherwise, it does not use anything else in KokkosCore directly.
#
IF(Tpetra_INST_DOUBLE)
  MESSAGE(STATUS "Tpetra: Performance-CGSolve test ENABLED")

  if (KokkosKernels_ENABLE_TPL_CUSPARSE)
    set(CG_SOLVE_VARIANT "cusparse")
  else()
    set(CG_SOLVE_VARIANT "NOcusparse")
  endif()

  TRIBITS_ADD_EXECUTABLE(
    Performance-CGSolve-orig
    NOEXEPREFIX
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


ENDIF()
