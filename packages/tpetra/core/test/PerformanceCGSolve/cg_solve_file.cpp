// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_Version.hpp"

#include "TpetraUtils_MatrixGenerator.hpp"
#include "MatrixMarket_Tpetra.hpp"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_XMLPerfTestArchive.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_StackedTimer.hpp"

#include <algorithm>
#include <functional>

#ifdef PERFORMANCE_CGSOLVE_NVSHMEM
// P6.2 is the fixed reference for the P7 completion-protocol experiment.
// Legacy shared utility headers remain transitively required for now.
#include "NvshmemP0.hpp"
#include "NvshmemP62.hpp"
#ifndef HAVE_TPETRA_INST_CUDA
#error "PERFORMANCE_CGSOLVE_NVSHMEM requires a CUDA-enabled Tpetra instantiation"
#endif
#endif

template <class CrsMatrix, class Vector>
bool cg_solve(Teuchos::RCP<CrsMatrix> A, Teuchos::RCP<Vector> b, Teuchos::RCP<Vector> x, int myproc, double tolerance, int max_iter) {
  using Teuchos::TimeMonitor;
  std::string addTimerName    = "CG: axpby";
  std::string matvecTimerName = "CG: spmv";
  std::string dotTimerName    = "CG: dot";
  static_assert(std::is_same<typename CrsMatrix::scalar_type, typename Vector::scalar_type>::value,
                "The CrsMatrix and Vector template parameters must have the same scalar_type.");

  typedef typename Vector::scalar_type ScalarType;
  typedef typename Vector::mag_type magnitude_type;
  typedef typename Vector::local_ordinal_type LO;
  Teuchos::RCP<Vector> r, p, Ap;
  // int max_iter=200;
  // double tolerance = 1e-8;
  r  = Tpetra::createVector<ScalarType>(A->getRangeMap());
  p  = Tpetra::createVector<ScalarType>(A->getRangeMap());
  Ap = Tpetra::createVector<ScalarType>(A->getRangeMap());

  magnitude_type normr     = 0;
  magnitude_type rtrans    = 0;
  magnitude_type oldrtrans = 0;

  LO print_freq = max_iter / 10;
  print_freq    = std::min(print_freq, 50);
  print_freq    = std::max(print_freq, 1);
  {
    TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
    p->update(1.0, *x, 0.0, *x, 0.0);
  }
  {
    TimeMonitor t(*TimeMonitor::getNewTimer(matvecTimerName));
    A->apply(*p, *Ap);
  }
  {
    TimeMonitor t(*TimeMonitor::getNewTimer(addTimerName));
    r->update(1.0, *b, -1.0, *Ap, 0.0);
  }
  {
    TimeMonitor t(*TimeMonitor::getNewTimer(dotTimerName));
    rtrans = r->dot(*r);
  }

  normr = std::sqrt(rtrans);

  if (myproc == 0) {
    std::cout << "Initial Residual = " << normr << std::endl;
  }

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
        magnitude_type beta = rtrans / oldrtrans;
        p->update(1.0, *r, beta);
      }
    }
    normr = std::sqrt(rtrans);
    if (myproc == 0 && (k % print_freq == 0 || k == max_iter)) {
      std::cout << "Iteration = " << k << "   Residual = " << normr << std::endl;
    }

    magnitude_type alpha    = 0;
    magnitude_type p_ap_dot = 0;
    {
      TimeMonitor t(*TimeMonitor::getNewTimer(matvecTimerName));
      A->apply(*p, *Ap);
    }
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
        } else
          brkdown_tol = 0.1 * p_ap_dot;
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

namespace CGParams {
int nsize        = 20;
bool printMatrix = false;
bool verbose     = false;
int niters       = 100;
double tolerance = 1.0e-2;
std::string filename;
std::string filename_vector;
std::string testarchive("Tpetra_PerformanceTests.xml");
std::string hostname;

double tol_small = 0.05;
double tol_large = 0.10;
}  // namespace CGParams

template <class Node>
int run() {
  using namespace CGParams;
  using std::cout;
  using std::endl;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::StackedTimer;
  using Teuchos::TimeMonitor;
  using Teuchos::tuple;

  typedef Tpetra::Vector<>::scalar_type Scalar;
  typedef typename Tpetra::Map<>::local_ordinal_type LO;
  typedef typename Tpetra::Map<>::global_ordinal_type GO;

  typedef Tpetra::CrsMatrix<Scalar, LO, GO, Node> crs_matrix_type;
  typedef Tpetra::Vector<Scalar, LO, GO, Node> vec_type;
  typedef Tpetra::Map<LO, GO, Node> map_type;

  //
  // Get the communicator and node
  //
  RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int myRank                          = comm->getRank();

  //
  // Say hello, print some communicator info
  //
  if (verbose) {
    if (myRank == 0) {
      cout << "Comm info: ";
    }
    cout << *comm;
  }

  // Read Tpetra::CrsMatrix from file
  //
  RCP<crs_matrix_type> A;
  if (!filename.empty()) {
    A = Tpetra::MatrixMarket::Reader<crs_matrix_type>::readSparseFile(filename, comm);
  } else {
    A = Tpetra::Utils::MatrixGenerator<crs_matrix_type>::generate_miniFE_matrix(nsize, comm);
  }

  if (printMatrix) {
    RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(cout));
    A->describe(*fos, Teuchos::VERB_EXTREME);
  } else if (verbose) {
    cout << endl
         << A->description() << endl
         << endl;
  }

  // This is a collective over the Maps' communicator.
  if (!A->getRangeMap()->isSameAs(*(A->getDomainMap()))) {
    throw std::runtime_error("The matrix must have domain and range maps that are the same.");
  }

  // Either read the right-hand side b of the linear system from a
  // file, or generate it.
  RCP<const map_type> map = A->getRangeMap();
  RCP<vec_type> b;
  if (nsize < 0) {
    typedef Tpetra::MatrixMarket::Reader<crs_matrix_type> reader_type;
    b = reader_type::readVectorFile(filename_vector, map->getComm(),
                                    map);
  } else {
    typedef Tpetra::Utils::MatrixGenerator<crs_matrix_type> gen_type;
    b = gen_type::generate_miniFE_vector(nsize, map->getComm());
  }

  // Output the problem size
  Tpetra::global_size_t ng = map->getGlobalNumElements();
  if (!myRank)
    std::cout << "Global matrix size = " << ng << std::endl;

  // The vector x on input is the initial guess for the CG solve.
  // On output, it is the approximate solution.
  RCP<vec_type> x(new vec_type(A->getDomainMap()));

  // Untimed warm-up apply
  A->apply(*b, *x);
  // Zero out x again
  x->putScalar(0);

  // Solve the linear system Ax=b using CG.
  RCP<StackedTimer> timer = rcp(new StackedTimer("CG: global"));
  TimeMonitor::setStackedTimer(timer);

  bool success = cg_solve(A, b, x, myRank, tolerance, niters);
  timer->stopBaseTimer();
  StackedTimer::OutputOptions options;
  options.print_warnings  = false;
  options.output_fraction = options.output_histogram = options.output_minmax = true;
  timer->report(std::cout, comm, options);

  std::string testBaseName = "Tpetra CGSolve ";
  if (Tpetra::Details::Behavior::cudaLaunchBlocking()) testBaseName += "CUDA_LAUNCH_BLOCKING ";

  auto xmlOut = timer->reportWatchrXML(testBaseName + std::to_string(comm->getSize()) + " ranks", comm);
  if (myRank == 0) {
    if (xmlOut.length())
      std::cout << "\nAlso created Watchr performance report " << xmlOut << '\n';
    if (success)
      std::cout << "End Result: TEST PASSED\n";
    else
      std::cout << "End Result: TEST FAILED\n";
  }

  return EXIT_SUCCESS;
}


#ifdef PERFORMANCE_CGSOLVE_NVSHMEM
// The archived P0-P6.1 entry points are no longer linked into this program.
int runNvshmemP62(const NvshmemP0::Session& session,
                  const NvshmemP62::Options& opts) {
  using namespace CGParams;
  using Node = Tpetra::KokkosCompat::KokkosCudaWrapperNode;
  using LO = Tpetra::Map<>::local_ordinal_type;
  using GO = Tpetra::Map<>::global_ordinal_type;
  NvshmemP62::run<LO, GO, Node>(session, opts, nsize, filename);
  return EXIT_SUCCESS;
}
#endif

int main(int argc, char* argv[]) {
  using namespace CGParams;
  using default_exec = Tpetra::Details::DefaultTypes::execution_space;
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

  int myRank = 0;
#ifdef HAVE_MPI
  (void)MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#endif  // HAVE_MPI

  //
  // Get example parameters from command-line processor
  //
  int numthreads = 1;
  int numgpus    = 1;
  // Default for #gpus is $KOKKOS_NUM_DEVICES
  const char* rawKokkosNumDevices = std::getenv("KOKKOS_NUM_DEVICES");
  if (rawKokkosNumDevices)
    numgpus = std::atoi(rawKokkosNumDevices);

  bool useSYCL    = false;
  bool useHIP     = false;
  bool useCuda    = false;
  bool useOpenMP  = false;
  bool useThreads = false;
  bool useSerial  = false;

#ifdef PERFORMANCE_CGSOLVE_NVSHMEM
  bool nvshmemP62 = false;
  NvshmemP62::Options nvshmemP62Options;
#endif

  Teuchos::CommandLineProcessor cmdp(false, true);
  cmdp.setOption("verbose", "quiet", &verbose, "Print messages and results.");
  cmdp.setOption("numthreads", &numthreads, "Number of threads per thread team.");
  cmdp.setOption("numgpus", &numgpus, "Number of GPUs.");
  cmdp.setOption("hostname", &hostname, "Override of hostname for PerfTest entry.");
  cmdp.setOption("testarchive", &testarchive, "Set filename for Performance Test archive.");
  cmdp.setOption("filename", &filename, "Filename for test matrix.");
  cmdp.setOption("filename_vector", &filename_vector, "Filename for test matrix vector.");
  cmdp.setOption("tolerance", &tolerance, "Relative residual tolerance used for solver.");
  cmdp.setOption("iterations", &niters, "Maximum number of iterations.");
  cmdp.setOption("printMatrix", "noPrintMatrix", &printMatrix, "Print the full matrix after reading it.");
  cmdp.setOption("size", &nsize, "Generate miniFE matrix with X^3 elements.");
  cmdp.setOption("tol_small", &tol_small, "Tolerance for total CG-Time and final residual.");
  cmdp.setOption("tol_large", &tol_large, "Tolerance for individual times.");
#ifdef PERFORMANCE_CGSOLVE_NVSHMEM
  cmdp.setOption("nvshmem-p62", "no-nvshmem-p62", &nvshmemP62,
                 "P6.2 descriptor-driven two-kernel packed PUT with MPI start barrier, no ACK.");
  cmdp.setOption("nvshmem-p62-case", &nvshmemP62Options.only,
                 "matrix|all|identity|permute|sparse-mixed|one-way|uneven|zero-owner|all-empty|multi-owner|fanout");
  cmdp.setOption("nvshmem-p62-mode", &nvshmemP62Options.mode,
                 "direct|diagnostics|both; barrier always outside GPU Import timing.");
  cmdp.setOption("nvshmem-p62-put-variant", &nvshmemP62Options.putVariant,
                 "thread|warp|block (one independent PUT/signal per descriptor).");
  cmdp.setOption("nvshmem-p62-chunk", &nvshmemP62Options.chunkDoubles,
                 "Maximum doubles per independent PUT descriptor (default 512).");
  cmdp.setOption("nvshmem-p62-receiver-blocks", &nvshmemP62Options.receiverBlocks,
                 "Persistent receiver polling blocks, capped below SM count (default 4).");
  cmdp.setOption("nvshmem-p62-warmup", &nvshmemP62Options.warmup,
                 "Warmup iterations (default 5).");
  cmdp.setOption("nvshmem-p62-reps", &nvshmemP62Options.repetitions,
                 "Direct comparisons per method (default 30).");
  cmdp.setOption("nvshmem-p62-diagnostic-reps", &nvshmemP62Options.diagnosticRepetitions,
                 "Two-stream host submission/completion diagnostic samples (default 30).");
  cmdp.setOption("nvshmem-p62-verify-every", &nvshmemP62Options.verifyEvery,
                 "Validate every epoch after the timer: 0=no (default), 1=yes.");
  cmdp.setOption("nvshmem-p62-out", &nvshmemP62Options.outputPrefix,
                 "Optional rank0 raw per-rank timing CSV prefix.");
  cmdp.setOption("nvshmem-p62-min-nodes", &nvshmemP62Options.minimumNodes,
                 "Required minimum distinct physical nodes.");
  cmdp.setOption("nvshmem-p62-max-nodes", &nvshmemP62Options.maximumNodes,
                 "Required maximum distinct physical nodes (0 unlimited).");
  cmdp.setOption("nvshmem-p62-cross-node", "no-nvshmem-p62-cross-node",
                 &nvshmemP62Options.requireCrossNode,
                 "Require actual cross-node packed sends in the requested cases.");
#endif
  // Only provide the option to use Node types that are actually enabled in this build
#ifdef HAVE_TPETRA_INST_SYCL
  cmdp.setOption("sycl", "no-sycl", &useSYCL, "Use SYCL node");
#endif
#ifdef HAVE_TPETRA_INST_HIP
  cmdp.setOption("hip", "no-hip", &useHIP, "Use HIP node");
#endif
#ifdef HAVE_TPETRA_INST_CUDA
  cmdp.setOption("cuda", "no-cuda", &useCuda, "Use Cuda node");
#endif
#ifdef HAVE_TPETRA_INST_OPENMP
  cmdp.setOption("openmp", "no-openmp", &useOpenMP, "Use OpenMP node");
#endif
#ifdef HAVE_TPETRA_INST_PTHREAD
  cmdp.setOption("threads", "no-threads", &useThreads, "Use Pthreads node");
#endif
#ifdef HAVE_TPETRA_INST_SERIAL
  cmdp.setOption("serial", "no-serial", &useSerial, "Use Serial node");
#endif
  if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return EXIT_FAILURE;
  }

  // If no node type was explicitly requested, use Tpetra's default node.

  if (!useSYCL && !useHIP && !useCuda && !useOpenMP && !useThreads && !useSerial) {
#ifdef HAVE_TPETRA_INST_SYCL
    if (std::is_same<default_exec, Kokkos::SYCL>::value) {
      if (myRank == 0) std::cout << "No node specified in command-line args, so using default (SYCL)\n";
      useSYCL = true;
    }
#endif
#ifdef HAVE_TPETRA_INST_HIP
    if (std::is_same<default_exec, Kokkos::HIP>::value) {
      if (myRank == 0) std::cout << "No node specified in command-line args, so using default (HIP)\n";
      useHIP = true;
    }
#endif
#ifdef HAVE_TPETRA_INST_CUDA
    if (std::is_same<default_exec, Kokkos::Cuda>::value) {
      if (myRank == 0) std::cout << "No node specified in command-line args, so using default (Cuda)\n";
      useCuda = true;
    }
#endif
#ifdef HAVE_TPETRA_INST_OPENMP
    if (std::is_same<default_exec, Kokkos::OpenMP>::value) {
      if (myRank == 0) std::cout << "No node specified in command-line args, so using default (OpenMP)\n";
      useOpenMP = true;
    }
#endif
#ifdef HAVE_TPETRA_INST_PTHREAD
    if (std::is_same<default_exec, Kokkos::Threads>::value) {
      if (myRank == 0) std::cout << "No node specified in command-line args, so using default (Pthreads)\n";
      useThreads = true;
    }
#endif
#ifdef HAVE_TPETRA_INST_SERIAL
    if (std::is_same<default_exec, Kokkos::Serial>::value) {
      if (myRank == 0) std::cout << "No node specified in command-line args, so using default (Serial)\n";
      useSerial = true;
    }
#endif
  }

#ifdef PERFORMANCE_CGSOLVE_NVSHMEM
  if (nvshmemP62 && (!useCuda || useSYCL || useHIP || useOpenMP ||
                     useThreads || useSerial)) {
    if (myRank == 0)
      std::cerr << "--nvshmem-p62 requires the CUDA Tpetra backend "
                   "(use --cuda and --numgpus=<GPUs visible per rank>).\n";
    return EXIT_FAILURE;
  }
#endif

  if (numgpus < 1) {
    if (myRank == 0) std::cerr << "--numgpus must be at least one.\n";
    return EXIT_FAILURE;
  }

  Kokkos::InitializationSettings kokkosArgs;
  kokkosArgs.set_num_threads(numthreads);
  kokkosArgs.set_disable_warnings(!verbose);
  int deviceRank = myRank;
#ifdef PERFORMANCE_CGSOLVE_NVSHMEM
  if (nvshmemP62) {
    // Kokkos owns CUDA device initialization. Use local MPI rank to choose
    // a GPU when each node exposes multiple GPUs to its local ranks.
    deviceRank = NvshmemP0::nodeLocalMpiRank(MPI_COMM_WORLD);
  }
#endif
  kokkosArgs.set_device_id(deviceRank % numgpus);

  Kokkos::initialize(kokkosArgs);
  int status = EXIT_FAILURE;
  try {
#ifdef PERFORMANCE_CGSOLVE_NVSHMEM
    if (nvshmemP62) {
      // Retire NVSHMEM before Kokkos::finalize and MPI_Finalize.
      NvshmemP0::Session session(MPI_COMM_WORLD);
      session.describe(deviceRank % numgpus);
      status = runNvshmemP62(session, nvshmemP62Options);
    } else
#endif
    {
#ifdef HAVE_TPETRA_INST_SYCL
      if (useSYCL) status = run<Tpetra::KokkosCompat::KokkosSYCLWrapperNode>();
      else
#endif
#ifdef HAVE_TPETRA_INST_HIP
      if (useHIP) status = run<Tpetra::KokkosCompat::KokkosHIPWrapperNode>();
      else
#endif
#ifdef HAVE_TPETRA_INST_CUDA
      if (useCuda) status = run<Tpetra::KokkosCompat::KokkosCudaWrapperNode>();
      else
#endif
#ifdef HAVE_TPETRA_INST_OPENMP
      if (useOpenMP) status = run<Tpetra::KokkosCompat::KokkosOpenMPWrapperNode>();
      else
#endif
#ifdef HAVE_TPETRA_INST_PTHREAD
      if (useThreads) status = run<Tpetra::KokkosCompat::KokkosThreadsWrapperNode>();
      else
#endif
#ifdef HAVE_TPETRA_INST_SERIAL
      if (useSerial) status = run<Tpetra::KokkosCompat::KokkosSerialWrapperNode>();
      else
#endif
      {
        if (myRank == 0)
          std::cerr << "Error: no node type was enabled. CG was not run.\n";
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "PerformanceCGSolve rank " << myRank << ": "
              << e.what() << std::endl;
#ifdef HAVE_MPI
    // A peer may be inside nvshmem_malloc() or another collective.
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    status = EXIT_FAILURE;
  }
  Kokkos::finalize();
  return status;
}
