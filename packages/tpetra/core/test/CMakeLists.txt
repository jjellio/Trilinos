# for TpetraTestingUtilities.hpp
TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})

# The experimental VMM regression suite is an explicit opt-in.  This keeps
# normal CUDA Trilinos test configurations (without the VMM experiment) intact.
OPTION(Tpetra_ENABLE_VMM_TESTS "Build experimental MPI/CUDA VMM tests" OFF)
IF(Tpetra_ENABLE_VMM_TESTS)
  IF(NOT TPL_ENABLE_MPI OR NOT Tpetra_INST_CUDA OR NOT Tpetra_INST_DOUBLE)
    MESSAGE(FATAL_ERROR
      "Tpetra_ENABLE_VMM_TESTS requires MPI, Tpetra_INST_CUDA and Tpetra_INST_DOUBLE")
  ENDIF()
  ADD_SUBDIRECTORIES(VMM)
ENDIF()

ADD_SUBDIRECTORIES(
  BasicPerfTest
  Behavior
  Blas
  Block
  BugTests
  Comm
  Core
  CrsGraph
  CrsGraphTransposer
  CrsMatrix
  Directory
  Distributor
  Extra
  FECrsGraph
  FECrsMatrix
  FEMultiVector
  Filter
  HashTable
  ImportExport
  ImportExport2
  inout
  LinearProblem
  Map
  MatrixMatrix
  Merge
  MPIAdvance
  MultiVector
  Node
  Operator
  PerformanceCGSolve
  Sort
  Spaces
  Utils
  RowMatrixTransposer
  Tsqr
  ReindexTransform
  SolverMapTransform
  SingletonFiltering
)
