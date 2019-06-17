
ASSERT_DEFINED (Tpetra_INST_DOUBLE)
IF (Tpetra_INST_DOUBLE)
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    kernel_compare
    SOURCES main.cpp
            descriptive_stats/descriptive_stats.cpp
    ARGS 
    COMM serial mpi
    STANDARD_PASS_OUTPUT
  )
ENDIF ()

