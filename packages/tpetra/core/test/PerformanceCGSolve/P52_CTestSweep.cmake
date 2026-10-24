# P5.2 correctness CTest registrations. Can be included from both the
# standalone p52-sweep/CMakeLists.txt and a GPU-enabled Trilinos CMakeLists.
# The tests launch mpirun DIRECTLY to control physical placement; do not wrap
# them in TRIBITS_ADD_TEST(NUM_MPI_PROCS) or an outer mpirun.
# All registration is opt-in, all cluster tests RUN_SERIAL and fail rather
# than silently skip if fewer nodes than requested are allocated.

message("HERE!!!")
set(_P52_SWEEP_TOOL_DIR "${CMAKE_CURRENT_LIST_DIR}")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(P52_EXE "${P52_EXE}" CACHE STRING
    "Absolute compiled P5.2 Performance-CGSolve executable, or generator expression")
if(NOT P52_EXE AND DEFINED PACKAGE_NAME AND
   TARGET "${PACKAGE_NAME}_Performance-CGSolve")
  set(P52_EXE "$<TARGET_FILE:${PACKAGE_NAME}_Performance-CGSolve>")
endif()
if(NOT P52_EXE)
  message(FATAL_ERROR "P52_EXE must be the rebuilt executable containing --nvshmem-p52")
endif()

if(DEFINED MPIEXEC_EXECUTABLE AND MPIEXEC_EXECUTABLE)
  set(_P52_DEFAULT_MPIEXEC "${MPIEXEC_EXECUTABLE}")
else()
  find_program(_P52_DEFAULT_MPIEXEC NAMES mpirun mpiexec)
  if(NOT _P52_DEFAULT_MPIEXEC)
    set(_P52_DEFAULT_MPIEXEC "mpirun")
  endif()
endif()
set(P52_MPIEXEC "${_P52_DEFAULT_MPIEXEC}" CACHE FILEPATH
    "Open MPI launcher (run ONCE per CTest case)")
set(P52_BIND_SCRIPT "/home/jjellio/cronus/gpu_bind.sh" CACHE FILEPATH
    "Optional existing GPU binding wrapper, applied to EACH MPI rank")
set(P52_NUMGPUS 1 CACHE STRING
    "Number of CUDA devices visible to EACH rank after the binding wrapper")
set(P52_HOSTFILE "" CACHE FILEPATH
    "Optional MPI hostfile; otherwise use active scheduler allocation")
set(P52_MPI_OPTIONS "" CACHE STRING "Other options to OpenMPI mpirun")
set(P52_LOG_DIR "${CMAKE_CURRENT_BINARY_DIR}/p52-test-logs" CACHE PATH
    "Full stdout/stderr logs plus per-test JSON validation records")
set(P52_EPOCHS 4 CACHE STRING "Changed source values per allocation (>=4)")
set(P52_CYCLES 2 CACHE STRING "Collective symmetric allocation cycles (>=2)")
set(P52_SKEW_MS 250 CACHE STRING "Delayed receiver stress, milliseconds")
set(P52_TIMEOUT 600 CACHE STRING "Per-test timeout seconds")
option(P52_ENABLE_SINGLE_NODE "Register smoke1/intra2/intra4 tests" ON)
option(P52_ENABLE_TWO_NODE "Register inter2/inter16 tests (requires a two-node scheduler allocation)" ON)
option(P52_ENABLE_CASE_TESTS "Register one CTest per applicable Map case per topology" ON)
option(P52_ENABLE_AGGREGATE "Register one aggregate --case=all per topology as a quick suite" OFF)
option(P52_ENABLE_SKEW "Register delayed-receiver tests when P52_ENABLE_TWO_NODE is ON" ON)
option(P52_ENABLE_STRESS "Register longer repeated-generation tests (opt-in)" ON)

if(NOT P52_ENABLE_CASE_TESTS AND NOT P52_ENABLE_AGGREGATE)
  message(FATAL_ERROR "Enable at least one of P52_ENABLE_CASE_TESTS, P52_ENABLE_AGGREGATE")
endif()
if(P52_EPOCHS LESS 4 OR P52_CYCLES LESS 2 OR P52_NUMGPUS LESS 1)
  message(FATAL_ERROR "P52_EPOCHS>=4, P52_CYCLES>=2, and P52_NUMGPUS>=1 are required")
endif()
if(P52_ENABLE_TWO_NODE AND NOT P52_MPIEXEC)
  message(FATAL_ERROR "P52_ENABLE_TWO_NODE requires P52_MPIEXEC")
endif()
if(NOT P52_BIND_SCRIPT)
  message(STATUS "P5.2: no binding wrapper; require GPU binding through scheduler or --numgpus=GPUs-visible-per-rank")
else()
  message(STATUS "P5.2: each MPI rank launched through ${P52_BIND_SCRIPT}; --numgpus=${P52_NUMGPUS}")
endif()

set(_P52_CASES_1 matrix identity permute all-empty)
set(_P52_CASES_2 matrix identity permute sparse-mixed one-way uneven zero-owner all-empty)
set(_P52_CASES_4 matrix identity permute sparse-mixed one-way uneven zero-owner all-empty multi-owner fanout)
set(_P52_CASES_16 ${_P52_CASES_4})

function(_p52_add_test topo case flavor epochs cycles size skew)
  # Each test name is unique, including stress and skew variants.
  set(_test_name "P52_${topo}_${case}")
  if(NOT "${flavor}" STREQUAL "case")
    string(APPEND _test_name "_${flavor}")
  endif()
  set(_cmd "${Python3_EXECUTABLE}" "${_P52_SWEEP_TOOL_DIR}/p52_case_runner.py"
      --exe "${P52_EXE}" --mpirun "${P52_MPIEXEC}"
      --topology "${topo}" --case "${case}"
      --numgpus "${P52_NUMGPUS}" --epochs "${epochs}" --cycles "${cycles}"
      --size "${size}" --skew-ms "${skew}"
      --timeout "${P52_TIMEOUT}" --log-dir "${P52_LOG_DIR}")
  if(P52_BIND_SCRIPT)
    list(APPEND _cmd --bind-script "${P52_BIND_SCRIPT}")
  endif()
  if(P52_HOSTFILE)
    list(APPEND _cmd --hostfile "${P52_HOSTFILE}")
  endif()
  if(P52_MPI_OPTIONS)
    list(APPEND _cmd "--mpi-options=${P52_MPI_OPTIONS}")
  endif()
  add_test(NAME "${_test_name}" COMMAND ${_cmd})

  set(_labels p52 p52-gpu "p52-${topo}")
  if(topo MATCHES "^inter")
    list(APPEND _labels p52-two-node)
  else()
    list(APPEND _labels p52-one-node)
  endif()
  if(flavor STREQUAL "case")
    list(APPEND _labels p52-case)
  elseif(flavor STREQUAL "aggregate")
    list(APPEND _labels p52-quick)
  elseif(flavor STREQUAL "skew")
    list(APPEND _labels p52-skew)
  elseif(flavor STREQUAL "stress")
    list(APPEND _labels p52-stress)
  endif()
  # Never overlap tests using the same GPUs/NICs. This is about CTest
  # scheduling only; the algorithm itself has no per-Import world barrier.
  set_tests_properties("${_test_name}" PROPERTIES
      LABELS "${_labels}" RUN_SERIAL TRUE
      TIMEOUT "${P52_TIMEOUT}" PROCESSORS 16)
endfunction()

if(P52_ENABLE_SINGLE_NODE)
  foreach(_topo smoke1 intra2 intra4)
    if(_topo STREQUAL "smoke1")
      set(_cases ${_P52_CASES_1})
    elseif(_topo STREQUAL "intra2")
      set(_cases ${_P52_CASES_2})
    else()
      set(_cases ${_P52_CASES_4})
    endif()
    if(P52_ENABLE_CASE_TESTS)
      foreach(_case IN LISTS _cases)
        _p52_add_test("${_topo}" "${_case}" case ${P52_EPOCHS} ${P52_CYCLES} 8 0)
      endforeach()
    endif()
    if(P52_ENABLE_AGGREGATE)
      _p52_add_test("${_topo}" all aggregate ${P52_EPOCHS} ${P52_CYCLES} 8 0)
    endif()
  endforeach()
endif()

if(P52_ENABLE_TWO_NODE)
  foreach(_topo inter2 inter16)
    if(_topo STREQUAL "inter2")
      set(_cases ${_P52_CASES_2})
      set(_size 8)
    else()
      set(_cases ${_P52_CASES_16})
      set(_size 100)
    endif()
    if(P52_ENABLE_CASE_TESTS)
      foreach(_case IN LISTS _cases)
        if(_case STREQUAL "matrix")
          set(_case_size ${_size})
        else()
          set(_case_size 8)
        endif()
        _p52_add_test("${_topo}" "${_case}" case ${P52_EPOCHS} ${P52_CYCLES} ${_case_size} 0)
      endforeach()
    endif()
    if(P52_ENABLE_AGGREGATE)
      _p52_add_test("${_topo}" all aggregate ${P52_EPOCHS} ${P52_CYCLES} ${_size} 0)
    endif()
  endforeach()
  if(P52_ENABLE_SKEW)
    # rank 0 is a receiver in these cases. Specifically, one-way is rank 1
    # -> rank 0, so this is an unambiguous early-PUT/late-receiver test.
    _p52_add_test(inter2 one-way skew ${P52_EPOCHS} ${P52_CYCLES} 8 ${P52_SKEW_MS})
    _p52_add_test(inter2 matrix skew ${P52_EPOCHS} ${P52_CYCLES} 8 ${P52_SKEW_MS})
    _p52_add_test(inter16 matrix skew ${P52_EPOCHS} ${P52_CYCLES} 100 ${P52_SKEW_MS})
    _p52_add_test(inter16 sparse-mixed skew ${P52_EPOCHS} ${P52_CYCLES} 8 ${P52_SKEW_MS})
  endif()
endif()

if(P52_ENABLE_STRESS)
  if(P52_ENABLE_SINGLE_NODE)
    _p52_add_test(intra4 fanout stress 64 4 8 0)
    _p52_add_test(intra4 one-way stress 64 4 8 0)
  endif()
  if(P52_ENABLE_TWO_NODE)
    _p52_add_test(inter16 matrix stress 32 4 32 0)
    _p52_add_test(inter16 fanout stress 64 4 8 0)
    _p52_add_test(inter2 one-way stress 64 4 8 250)
  endif()
endif()

message(STATUS "P5.2: CTest topology-aware GPU suite registered (intranode=${P52_ENABLE_SINGLE_NODE}, internode=${P52_ENABLE_TWO_NODE}, cases=${P52_ENABLE_CASE_TESTS}, aggregate=${P52_ENABLE_AGGREGATE}, skew=${P52_ENABLE_SKEW}, stress=${P52_ENABLE_STRESS})")
