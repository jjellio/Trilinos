// @HEADER
// *****************************************************************************
// Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// Experimental derivative of
//   packages/tpetra/core/test/PerformanceCGSolve/cg_solve_file.cpp
//
// Adds a CUDA-VMM direct-address SpMV path while retaining the original
// Tpetra::CrsMatrix::apply() path as the default baseline.
//
// IMPORTANT FIRST-VERSION LIMITATION
// ----------------------------------
// CG's p vector is still a normal Tpetra::Vector.  Before a VMM SpMV, its
// local device slice is copied into this rank's VMM-owned X allocation and
// all ranks synchronize.  That work is reported separately as
// "CG: vmm publish".  The direct sparse operation itself is reported as
// "CG: vmm spmv".  The total CG timer includes both.  A later step should
// make solver vectors natively VMM-backed so this mirror copy disappears.
//
// Current VMM addressing also requires a contiguous Tpetra domain Map.
// The built-in miniFE matrix generator satisfies this requirement.

#ifdef FENCE_TIMERS
#pragma message("FENCING TIMERS")
#endif

#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_Version.hpp"
#include "TpetraUtils_MatrixGenerator.hpp"
#include "MatrixMarket_Tpetra.hpp"

#include "KokkosSparse_spmv.hpp"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_CommHelpers.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_TPETRA_INST_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include "GaleriHelper.hpp"
#include "TpetraMatrixInfo.hpp"
#include "TpetraMatrixReuseTransform.hpp"
#include "VmmSpmvContext.hpp"

std::string matrixName = "miniFE";
int reuse = -1;
bool noReuse = false;
std::string saveGaleri = "";


namespace CGParams {
int nsize = 20;
bool printMatrix = false;
bool verbose = false;
int niters = 100;
double tolerance = 1.0e-2;
std::string filename;
std::string filename_vector;
std::string testarchive("Tpetra_PerformanceTests.xml");
std::string hostname;
double tol_small = 0.05;
double tol_large = 0.10;

bool useVmm = false;
bool validateVmm = true;
bool spmvOnly = false;
std::string vmmIpc = "posix";  // posix on current H200/HGX node, fabric on NVL72/IMEX
}  // namespace CGParams


// Local shorthand for existing CG driver templates.  No new instantiation machinery.
#ifdef HAVE_TPETRA_INST_CUDA
template <class CrsMatrix>
using VmmContextFor = VmmExperiment::VmmSpmvContext<
    typename CrsMatrix::scalar_type,
    typename CrsMatrix::local_ordinal_type,
    typename CrsMatrix::global_ordinal_type,
    typename CrsMatrix::node_type>;
#endif



template <class CrsMatrix, class Vector>
static void applyCgOperator(const Teuchos::RCP<CrsMatrix>& A,
                            const Teuchos::RCP<Vector>& x,
                            const Teuchos::RCP<Vector>& y
#ifdef HAVE_TPETRA_INST_CUDA
                            , VmmContextFor<CrsMatrix>* vmm
#endif
                            ) {
  using Teuchos::TimeMonitor;
#ifdef HAVE_TPETRA_INST_CUDA
  if (vmm != nullptr) {
    {
      TimeMonitor t(*TimeMonitor::getNewTimer("CG: vmm publish"));
      vmm->publish(*x);
    }
    {
      TimeMonitor t(*TimeMonitor::getNewTimer("CG: vmm spmv"));
      vmm->applyPublished(*y);
    }
    return;
  }
#endif
  {
    #ifdef FENCE_TIMERS
    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    TimeMonitor t(*TimeMonitor::getNewTimer("CG: spmv"));
    A->apply(*x, *y);

    #ifdef FENCE_TIMERS
    Kokkos::fence("CG Tpetra SpMV timing fence");
    #endif
  }
}

template <class CrsMatrix, class Vector>
bool cg_solve(Teuchos::RCP<CrsMatrix> A,
              Teuchos::RCP<Vector> b,
              Teuchos::RCP<Vector> x,
              int myproc,
              double tolerance,
              int max_iter
#ifdef HAVE_TPETRA_INST_CUDA
              , VmmContextFor<CrsMatrix>* vmm
#endif
              ) {
  using Teuchos::TimeMonitor;
  const std::string addTimerName = "CG: axpby";
  const std::string dotTimerName = "CG: dot";

  static_assert(std::is_same<typename CrsMatrix::scalar_type,
                             typename Vector::scalar_type>::value,
                "The CrsMatrix and Vector template parameters must have the same scalar_type.");

  using ScalarType    = typename Vector::scalar_type;
  using magnitude_type = typename Vector::mag_type;
  using LO             = typename Vector::local_ordinal_type;

  auto r  = Tpetra::createVector<ScalarType>(A->getRangeMap());
  auto p  = Tpetra::createVector<ScalarType>(A->getRangeMap());
  auto Ap = Tpetra::createVector<ScalarType>(A->getRangeMap());

  magnitude_type normr = 0;
  magnitude_type rtrans = 0;
  magnitude_type oldrtrans = 0;
  LO print_freq = max_iter / 10;
  print_freq = std::min(print_freq, static_cast<LO>(50));
  print_freq = std::max(print_freq, static_cast<LO>(1));

  {
    TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
    p->update(1.0, *x, 0.0, *x, 0.0);
    #ifdef FENCE_TIMERS
    Kokkos::fence("CG axpby p timing fence");
    #endif
  }

  applyCgOperator(A, p, Ap
#ifdef HAVE_TPETRA_INST_CUDA
                  , vmm
#endif
                  );

  {
    TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
    r->update(1.0, *b, -1.0, *Ap, 0.0);
    #ifdef FENCE_TIMERS
    Kokkos::fence("CG axpby r timing fence");
    #endif
  }
  {
    TimeMonitor t(*TimeMonitor::getNewTimer(dotTimerName));
    rtrans = r->dot(*r);
    #ifdef FENCE_TIMERS
    Kokkos::fence("CG dot timing fence");
    #endif
  }

  normr = std::sqrt(rtrans);
  if (myproc == 0) std::cout << "Initial Residual = " << normr << std::endl;

  magnitude_type brkdown_tol = std::numeric_limits<magnitude_type>::epsilon();
  LO k;
  for (k = 1; k <= max_iter && normr > tolerance; ++k) {
    if (k == 1) {
      TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
      p->update(1.0, *r, 0.0);
    } else {
      oldrtrans = rtrans;
      {
        TimeMonitor t(*TimeMonitor::getNewTimer(dotTimerName));
        rtrans = r->dot(*r);
      }
      {
        TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
        const magnitude_type beta = rtrans / oldrtrans;
        p->update(1.0, *r, beta);
      }
    }

    normr = std::sqrt(rtrans);
    if (myproc == 0 && (k % print_freq == 0 || k == max_iter)) {
      std::cout << "Iteration = " << k << " Residual = " << normr << std::endl;
    }

    magnitude_type alpha = 0;
    magnitude_type p_ap_dot = 0;

    applyCgOperator(A, p, Ap
#ifdef HAVE_TPETRA_INST_CUDA
                    , vmm
#endif
                    );

    {
      TimeMonitor t(*TimeMonitor::getNewTimer(dotTimerName));
      p_ap_dot = Ap->dot(*p);
    }
    {
      TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
      if (p_ap_dot < brkdown_tol) {
        if (p_ap_dot < 0) {
          std::cerr << "miniFE::cg_solve ERROR, numerical breakdown!" << std::endl;
          return false;
        } else {
          brkdown_tol = 0.1 * p_ap_dot;
        }
      }
      alpha = rtrans / p_ap_dot;
      x->update(alpha, *p, 1.0);
      r->update(-alpha, *Ap, 1.0);
    }
  }

  {
    TimeMonitor t(*TimeMonitor::getNewTimer(dotTimerName));
    rtrans = r->dot(*r);
  }
  normr = std::sqrt(rtrans);
  return true;
}

template <class Node>
int run() {
  using namespace CGParams;
  using std::cout;
  using std::endl;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::StackedTimer;
  using Teuchos::TimeMonitor;

  using Scalar = Tpetra::Vector<>::scalar_type;
  using LO = typename Tpetra::Map<>::local_ordinal_type;
  using GO = typename Tpetra::Map<>::global_ordinal_type;
  using crs_matrix_type = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
  using vec_type = Tpetra::Vector<Scalar, LO, GO, Node>;
  using map_type = Tpetra::Map<LO, GO, Node>;

  RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
  const int myRank = comm->getRank();

  if (verbose) {
    if (myRank == 0) cout << "Comm info: ";
    cout << *comm;
  }


  RCP<crs_matrix_type> A;
  
  if (!filename.empty()) {
    A = Tpetra::MatrixMarket::Reader<crs_matrix_type>::readSparseFile(
        filename, comm);
  }
  else if (matrixName == "miniFE") {
    A = Tpetra::Utils::MatrixGenerator<crs_matrix_type>::
        generate_miniFE_matrix(nsize, comm);
  }
  else {
    A = my_helper::get_galeri_matrix<Node>(
        matrixName,
        nsize,
        comm,
        saveGaleri);
  }
  if (printMatrix) {
    RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(cout));
    A->describe(*fos, Teuchos::VERB_EXTREME);
  } else if (verbose) {
    cout << endl << A->description() << endl << endl;
  }

  /* Optional synthetic reuse transform. */
  if (reuse >= 0) {
      TpetraMatrixTools::ReuseTransformStats reuseStats;
  
      A = TpetraMatrixTools::limitReuse(
          A,
          reuse,
          &reuseStats);
  
      if (myRank == 0) {
          std::cout
              << "Reuse transform:"
              << " requested=" << reuseStats.requestedReuse
              << " intrinsic-max=" << reuseStats.intrinsicMaxReuse
              << " domain=" << reuseStats.originalDomainSize
              << " -> " << reuseStats.transformedDomainSize
              << " nnz=" << reuseStats.globalNnz
              << std::endl;
      }
  }


  RCP<const map_type> domainMap = A->getDomainMap();
  RCP<const map_type> rangeMap  = A->getRangeMap();
  const bool square = rangeMap->isSameAs(*domainMap);

  if (!spmvOnly && !square) {
    throw std::runtime_error(
        "CG requires identical domain and range Maps. "
        "The matrix is rectangular; use --spmv-only.");
  }

  // Keep CG vectors and pure-SpMV vectors distinct.  A reuse-transformed
  // matrix may be rectangular:
  //
  //   A : range x domain
  //   x : domain
  //   y : range
  RCP<vec_type> b;
  RCP<vec_type> x;
  RCP<vec_type> spmvX;
  RCP<vec_type> spmvY;

  if (spmvOnly) {
    spmvX = rcp(new vec_type(domainMap));
    spmvY = rcp(new vec_type(rangeMap));
    spmvX->putScalar(static_cast<Scalar>(1.0));
    spmvY->putScalar(static_cast<Scalar>(0.0));
  } else {
    if (nsize < 0) {
      using reader_type =
          Tpetra::MatrixMarket::Reader<crs_matrix_type>;

      b = reader_type::readVectorFile(
          filename_vector,
          rangeMap->getComm(),
          rangeMap);
    }
    else if (matrixName == "miniFE") {
      using gen_type =
          Tpetra::Utils::MatrixGenerator<crs_matrix_type>;

      b = gen_type::generate_miniFE_vector(
          nsize,
          rangeMap->getComm());
    }
    else {
      b = rcp(new vec_type(rangeMap));
      b->putScalar(1.0);
    }

    x = rcp(new vec_type(domainMap));
  }

  const Tpetra::global_size_t globalRows =
      rangeMap->getGlobalNumElements();
  const Tpetra::global_size_t globalCols =
      domainMap->getGlobalNumElements();

  if (myRank == 0) {
    std::cout << "Matrix = " << matrixName << std::endl;
    // Preserve this line for the existing sweep parser.
    std::cout << "Global matrix size = " << globalRows << std::endl;
    std::cout << "Global matrix shape = "
              << globalRows << " x " << globalCols << std::endl;
    std::cout << "Execution mode = "
              << (spmvOnly ? "SpMV only" : "CG") << std::endl;
    std::cout << "SpMV backend = "
              << (useVmm ? "CUDA VMM direct-address" : "Tpetra apply")
              << std::endl;
  }

  const auto access = TpetraMatrixInfo::analyzeAccessPattern(*A);
  TpetraMatrixInfo::print(std::cout, access);

#ifdef HAVE_TPETRA_INST_CUDA
  std::unique_ptr<VmmContextFor<crs_matrix_type>> vmm;
  if (useVmm) {
    using exec_space = typename crs_matrix_type::device_type::execution_space;
    if (!std::is_same<exec_space, Kokkos::Cuda>::value) {
      throw std::runtime_error("--vmm requires the CUDA Tpetra node");
    }
    const auto ipcMode = VmmExperiment::parseIpcMode(vmmIpc);
    vmm.reset(new VmmContextFor<crs_matrix_type>(*A, ipcMode));

    if (validateVmm) {
      auto yRef = Tpetra::createVector<Scalar>(A->getRangeMap());
      auto yVmm = Tpetra::createVector<Scalar>(A->getRangeMap());
      auto diff = Tpetra::createVector<Scalar>(A->getRangeMap());

      const RCP<vec_type>& validationX = spmvOnly ? spmvX : b;
      A->apply(*validationX, *yRef);
      vmm->apply(*validationX, *yVmm);
      Tpetra::deep_copy(*diff, *yVmm);
      diff->update(-1.0, *yRef, 1.0);
      const auto err = diff->norm2();
      const auto ref = yRef->norm2();
      const double rel = ref == 0 ? static_cast<double>(err)
                                  : static_cast<double>(err / ref);
      if (myRank == 0) {
        std::cout << "VMM validation relative ||Yvmm-Ytpetra||2/||Ytpetra||2 = "
                  << rel << std::endl;
      }
      if (!(rel < 1.0e-10)) {
        throw std::runtime_error("VMM direct SpMV validation against Tpetra::apply failed");
      }
    }
  }
#else
  if (useVmm) throw std::runtime_error("This Tpetra build has no CUDA instantiation for --vmm");
#endif

  // Untimed warm-up apply using the selected backend.
#ifdef HAVE_TPETRA_INST_CUDA
  if (spmvOnly) {
    if (vmm) vmm->apply(*spmvX, *spmvY);
    else A->apply(*spmvX, *spmvY);
  } else {
    if (vmm) vmm->apply(*b, *x);
    else A->apply(*b, *x);
  }
#else
  if (spmvOnly) A->apply(*spmvX, *spmvY);
  else A->apply(*b, *x);
#endif

  Kokkos::fence("warmup complete");
  if (spmvOnly) {
    spmvY->putScalar(0);
  } else {
    x->putScalar(0);
  }
  Kokkos::fence("post-warmup reset complete");

// Align the starting line.
MPI_Barrier(MPI_COMM_WORLD);

// you must start a stacked timer somewhere...
RCP<StackedTimer> timer = rcp(new StackedTimer(spmvOnly ? "SpMV: global" : "CG: global"));
TimeMonitor::setStackedTimer(timer);

const double t0 = MPI_Wtime();

  bool success = true;

  if (spmvOnly) {
    for (int iter = 0; iter < niters; ++iter) {
      applyCgOperator(
          A,
          spmvX,
          spmvY
#ifdef HAVE_TPETRA_INST_CUDA
          , vmm.get()
#endif
      );
    }
  } else {
    success = cg_solve(
        A, b, x, myRank, tolerance, niters
#ifdef HAVE_TPETRA_INST_CUDA
        , vmm.get()
#endif
    );
  }

// Define benchmark completion as device work completed.
Kokkos::fence(spmvOnly ? "SpMV completion" : "CG completion");

const double local_elapsed = MPI_Wtime() - t0;

timer->stopBaseTimer();
double critical_elapsed = 0.0;
MPI_Reduce(&local_elapsed,
           &critical_elapsed,
           1,
           MPI_DOUBLE,
           MPI_MAX,
           0,
           MPI_COMM_WORLD);



 

  StackedTimer::OutputOptions options;
  options.print_warnings = false;
  options.output_proc_minmax = true;
  options.output_fraction = options.output_histogram = options.output_minmax = true;
  timer->report(std::cout, comm, options);

  const std::string testBaseName =
      std::string(spmvOnly ? "Tpetra SpMV " : "Tpetra CGSolve ") +
      (useVmm ? "VMM " : "") +
      (Tpetra::Details::Behavior::cudaLaunchBlocking() ? "CUDA_LAUNCH_BLOCKING " : "");
  auto xmlOut = timer->reportWatchrXML(
      testBaseName + std::to_string(comm->getSize()) + " ranks", comm);

  if (myRank == 0) {
    std::cout
        << (spmvOnly ? "SpMV Critical Max Time: " : "CG Solve Critical Max Time: ")
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << critical_elapsed << "\n";
    if (xmlOut.length()) {
      std::cout << "\nAlso created Watchr performance report "
                << xmlOut << '\n';
    }
    if (success) std::cout << "End Result: TEST PASSED\n";
    else std::cout << "End Result: TEST FAILED\n";
  }

  // Ensure VMM resources are destroyed before Kokkos/MPI teardown.
#ifdef HAVE_TPETRA_INST_CUDA
  vmm.reset();
#endif

  return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
  using namespace CGParams;
  using default_exec = Tpetra::Details::DefaultTypes::execution_space;

  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

  int myRank = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#endif

  int numthreads = 1;
  int numgpus = 1;
  if (const char* raw = std::getenv("KOKKOS_NUM_DEVICES")) numgpus = std::atoi(raw);

  bool useCuda = false;

  Teuchos::CommandLineProcessor cmdp(false, true);
  cmdp.setOption("verbose", "quiet", &verbose, "Print messages and results.");
  cmdp.setOption("numthreads", &numthreads, "Number of threads per thread team.");
  cmdp.setOption("numgpus", &numgpus, "Number of GPUs visible to this process.");
  cmdp.setOption("hostname", &hostname, "Override of hostname for PerfTest entry.");
  cmdp.setOption("testarchive", &testarchive, "Set filename for Performance Test archive.");
  cmdp.setOption("matrixType", &matrixName,
      "Matrix to generate: miniFE (default), or any supported Galeri matrix name.");
  cmdp.setOption("filename", &filename, "Filename for test matrix.");
  cmdp.setOption("filename_vector", &filename_vector, "Filename for test matrix vector.");
  cmdp.setOption("tolerance", &tolerance, "Relative residual tolerance used for solver.");
  cmdp.setOption("iterations", &niters, "Maximum number of iterations / SpMV repetitions.");
  cmdp.setOption(
      "spmv-only", "cg", &spmvOnly,
      "Run exactly --iterations SpMVs instead of CG; supports rectangular matrices.");
  cmdp.setOption("printMatrix", "noPrintMatrix", &printMatrix,
                 "Print the full matrix after reading it.");
  cmdp.setOption("size", &nsize, "Generate miniFE matrix with X^3 elements.");
  cmdp.setOption("tol_small", &tol_small, "Tolerance for total CG-Time and final residual.");
  cmdp.setOption("tol_large", &tol_large, "Tolerance for individual times.");
  cmdp.setOption("vmm", "no-vmm", &useVmm,
                 "Use experimental CUDA-VMM direct-address SpMV instead of Tpetra::apply.");
  cmdp.setOption("vmm-validate", "no-vmm-validate", &validateVmm,
                 "Validate one VMM SpMV against Tpetra::apply before CG.");
  cmdp.setOption("vmm-ipc", &vmmIpc,
                 "VMM IPC backend: posix (same OS) or fabric (NVL72/IMEX).");

  cmdp.setOption(
      "reuse",
      &reuse,
      "Maximum reuse of a synthetic X scalar; 0 means no reuse.");
  
  cmdp.setOption(
      "no-reuse",
      "allow-reuse",
      &noReuse,
      "Equivalent to --reuse=0.");

  cmdp.setOption(
      "saveGaleri",
      &saveGaleri,
      "If nonempty, write <prefix>.mtx and "
      "<prefix>.logical.mtx for Galeri validation");

#ifdef HAVE_TPETRA_INST_CUDA
  cmdp.setOption("cuda", "no-cuda", &useCuda, "Use Cuda node");
#endif

  if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return EXIT_FAILURE;
  }
  if (noReuse) {
      reuse = 0;
  }

#ifdef HAVE_TPETRA_INST_CUDA
  if (!useCuda) {
    if (std::is_same<default_exec, Kokkos::Cuda>::value) {
      if (myRank == 0) std::cout << "No node specified; using default CUDA node\n";
      useCuda = true;
    }
  }
#else
  if (useVmm) {
    if (myRank == 0) std::cerr << "--vmm requested but Tpetra CUDA instantiation is disabled\n";
    return EXIT_FAILURE;
  }
#endif

  if (numgpus <= 0) numgpus = 1;

  Kokkos::InitializationSettings kokkosArgs;
  kokkosArgs.set_num_threads(numthreads);
  // With the binding wrapper used in this project, each MPI rank normally sees
  // exactly one GPU, so --numgpus=1 makes this select CUDA device 0 in each
  // process (which is a distinct physical H200 after CUDA_VISIBLE_DEVICES).
  kokkosArgs.set_device_id(myRank % numgpus);
  kokkosArgs.set_disable_warnings(!verbose);
  Kokkos::initialize(kokkosArgs);

  int rc = EXIT_FAILURE;
  {
#ifdef HAVE_TPETRA_INST_CUDA
    if (useCuda) {
      rc = run<Tpetra::KokkosCompat::KokkosCudaWrapperNode>();
    } else
#endif
    {
      if (myRank == 0) std::cerr << "Error: CUDA node was not enabled. CG was not run.\n";
    }
  }

  Kokkos::finalize();
  return rc;
}



