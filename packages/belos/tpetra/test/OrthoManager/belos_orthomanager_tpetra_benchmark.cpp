//@HEADER
// ************************************************************************
//
//                 Belos: Block Linear Solvers Package
//                  Copyright 2004 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER

/// \file belos_orthomanager_tpetra_benchmark.cpp
/// \brief Benchmark (Mat)OrthoManager subclass(es) with Tpetra
///
/// Benchmark various subclasses of (Mat)OrthoManager, using
/// Tpetra::MultiVector as the multivector implementation, and
/// Tpetra::Operator as the operator implementation.
///
#include "belos_orthomanager_tpetra_util.hpp"
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <regex>

#include <Teuchos_LAPACK_wrappers.hpp>
//
// These typedefs make main() as generic as possible.
//
typedef double scalar_type;
typedef Tpetra::Map<>::local_ordinal_type local_ordinal_type;
typedef Tpetra::Map<>::global_ordinal_type global_ordinal_type;
typedef Tpetra::Map<>::node_type node_type;

typedef Teuchos::ScalarTraits<scalar_type> SCT;
typedef SCT::magnitudeType magnitude_type;
typedef Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type> MV;
typedef Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type, node_type> OP;
typedef Belos::MultiVecTraits<scalar_type, MV> MVT;
typedef Belos::OperatorTraits<scalar_type, MV, OP> OPT;
typedef Teuchos::SerialDenseMatrix<int, scalar_type> serial_matrix_type;
typedef Tpetra::Map<local_ordinal_type, global_ordinal_type, node_type> map_type;
typedef Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type> sparse_matrix_type;

/* ******************************************************************* */

extern "C" {
  void F77_BLAS_MANGLE(ilaver,ILAVER) (int* major,int* minor,int* patch);
}

std::string
getLapackVersion ();

std::string
getBlasLibVersion ();

std::string
getDateTime ();

std::string
getHostname ();


std::string
getOMP_WAIT_POLICY ();

std::string
getOMP_PLACES ();

std::string
quote (const std::string& str);

void
fineGrainMultiplyTimersToCSV (const std::string& identifier,
                              const int rank,
                              const std::vector<double>& timings,
                              std::stringstream& ss,
                              const bool ignore_first=true);

void
writeDetailsFile (const std::string& filename, const std::stringstream& data, const std::string& header);

void
gatherStrings (const std::string& str, std::stringstream& ss);

std::string
fineGrainMultiplyTimersToCSV ();

/* ******************************************************************* */

/// \fn main
/// \brief Benchmark driver for (Mat)OrthoManager subclasses
int
main (int argc, char *argv[])
{
  using std::endl;
  using Belos::OrthoManager;
  using Belos::OrthoManagerFactory;
  using Belos::OutputManager;
  using Teuchos::CommandLineProcessor;
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &std::cout);

  bool success = false;
  bool verbose = false; // Verbosity of output
  try {
    RCP<const Teuchos::Comm<int> > pComm =
      Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
    // This factory object knows how to make a (Mat)OrthoManager
    // subclass, given a name for the subclass.  The name is not the
    // same as the class' syntactic name: e.g., "TSQR" is the name of
    // TsqrOrthoManager.
    OrthoManagerFactory<scalar_type, MV, OP> factory;

    // The name of the (Mat)OrthoManager subclass to instantiate.
    std::string orthoManName (factory.defaultName());

    // For SimpleOrthoManager: the normalization method to use.  Valid
    // values: "MGS", "CGS".
    std::string normalization ("CGS");

    // Name of the Harwell-Boeing sparse matrix file from which to read
    // the inner product operator matrix.  If name is "" or not provided
    // at the command line, use the standard Euclidean inner product.
    std::string filename;

    bool debug = false;   // Whether to print debugging-level output
    // Whether or not to run the benchmark.  If false, we let this
    // "test" pass trivially.
    bool benchmark = false;

    // Whether to display benchmark results compactly (in a CSV format),
    // or in a human-readable table.
    bool displayResultsCompactly = false;

    // Default _local_ (per MPI process) number of rows.  This will
    // change if a sparse matrix is loaded in as an inner product
    // operator.  Regardless, the number of rows per MPI process must be
    // no less than numCols*numBlocks in order for TSQR to work.  To
    // ensure that the test always passes with default parameters, we
    // scale by the number of processes.  The default value below may be
    // changed by a command-line parameter with a corresponding name.
    int numRowsPerProcess = 100;
    int numRowsGlobal = 100;

    // The OrthoManager is benchmarked with numBlocks multivectors of
    // width numCols each, for numTrials trials.  The values below are
    // defaults and may be changed by the corresponding command-line
    // arguments.
    int numCols = 10;
    int numBlocks = 5;
    int numTrials = 3;

    CommandLineProcessor cmdp (false, true);
    cmdp.setOption ("benchmark", "nobenchmark", &benchmark,
        "Whether to run the benchmark.  If not, this \"test\" "
        "passes trivially.");
    cmdp.setOption ("verbose", "quiet", &verbose,
        "Print messages and results.");
    cmdp.setOption ("debug", "nodebug", &debug,
        "Print debugging information.");
    cmdp.setOption ("compact", "human", &displayResultsCompactly,
        "Whether to display benchmark results compactly (in a "
        "CSV format), or in a human-readable table.");
    cmdp.setOption ("filename", &filename,
        "Filename of a Harwell-Boeing sparse matrix, used as the "
        "inner product operator by the orthogonalization manager."
        "  If not provided, no matrix is read and the Euclidean "
        "inner product is used.");
    {
      std::ostringstream os;
      const int numValid = factory.numOrthoManagers();
      const bool plural = numValid > 1 || numValid == 0;

      os << "OrthoManager subclass to benchmark.  There ";
      os << (plural ? "are " : "is ") << numValid << (plural ? "s: " : ": ");
      factory.printValidNames (os);
      os << ".  If none is provided, the test trivially passes.";
      cmdp.setOption ("ortho", &orthoManName, os.str().c_str());
    }
    cmdp.setOption ("normalization", &normalization,
        "For SimpleOrthoManager (--ortho=Simple): the normalization "
        "method to use.  Valid values: \"MGS\", \"CGS\".");
    cmdp.setOption ("numRowsPerProcess", &numRowsPerProcess,
        "Number of rows per MPI process in the test multivectors.  "
        "If an input matrix is given, this value is ignored, since "
        "the vectors must be commensurate with the dimensions of "
        "the matrix.");
    cmdp.setOption ("numRowsGlobal", &numRowsGlobal,
        "Global number of rows.");
    cmdp.setOption ("numCols", &numCols,
        "Number of columns in the input multivector (>= 1).");
    cmdp.setOption ("numBlocks", &numBlocks,
        "Number of block(s) to benchmark (>= 1).");
    cmdp.setOption ("numTrials", &numTrials,
        "Number of trial(s) per timing run (>= 1).");

    // Parse the command-line arguments.
    {
      const CommandLineProcessor::EParseCommandLineReturn parseResult = cmdp.parse (argc,argv);
      // If the caller asks us to print the documentation, or does not
      // explicitly say to run the benchmark, we let this "test" pass
      // trivially.
      if (! benchmark || parseResult == CommandLineProcessor::PARSE_HELP_PRINTED)
      {
        if (Teuchos::rank(*pComm) == 0)
          std::cout << "End Result: TEST PASSED" << endl;
        return EXIT_SUCCESS;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(parseResult != CommandLineProcessor::PARSE_SUCCESSFUL,
          std::invalid_argument,
          "Failed to parse command-line arguments");
    }

    // Total number of rows in the test vector(s).
    // This may be changed if we load in a sparse matrix.
    numRowsPerProcess = numRowsGlobal / pComm->getSize();
    int numRows = numRowsGlobal;
    //
    // Validate command-line arguments
    //
    TEUCHOS_TEST_FOR_EXCEPTION(numRowsPerProcess <= 0, std::invalid_argument,
        "numRowsPerProcess <= 0 is not allowed");
    TEUCHOS_TEST_FOR_EXCEPTION(numCols <= 0, std::invalid_argument,
        "numCols <= 0 is not allowed");
    TEUCHOS_TEST_FOR_EXCEPTION(numBlocks <= 0, std::invalid_argument,
        "numBlocks <= 0 is not allowed");

    // Declare an output manager for handling local output.  Initialize,
    // using the caller's desired verbosity level.
    RCP<OutputManager<scalar_type> > outMan =
      Belos::Test::makeOutputManager<scalar_type> (verbose, debug);

    // Stream for debug output.  If debug output is not enabled, then
    // this stream doesn't print anything sent to it (it's a "black
    // hole" stream).
    std::ostream& debugOut = outMan->stream(Belos::Debug);
    Belos::Test::printVersionInfo (debugOut);

    // Load the inner product operator matrix from the given filename.
    // If filename == "", use the identity matrix as the inner product
    // operator (the Euclidean inner product), and leave M as
    // Teuchos::null.  Also return an appropriate Map (which will
    // always be initialized; it should never be Teuchos::null).
    RCP<map_type> map;
    RCP<sparse_matrix_type> M;
    {
      using Belos::Test::loadSparseMatrix;
      // If the sparse matrix is loaded successfully, this call will
      // modify numRows to be the total number of rows in the sparse
      // matrix.  Otherwise, it will leave numRows alone.
      std::pair<RCP<map_type>, RCP<sparse_matrix_type> > results =
        loadSparseMatrix<local_ordinal_type, global_ordinal_type, node_type> (pComm, filename, numRows, debugOut);
      map = results.first;
      M = results.second;
    }
    TEUCHOS_TEST_FOR_EXCEPTION(map.is_null(), std::logic_error,
        "Error: (Mat)OrthoManager test code failed to "
        "initialize the Map");
    if (M.is_null())
    {
      // Number of rows per process has to be >= number of rows.
      TEUCHOS_TEST_FOR_EXCEPTION(numRowsPerProcess <= numCols,
          std::invalid_argument,
          "numRowsPerProcess <= numCols is not allowed");
    }
    // Loading the sparse matrix may have changed numRows, so check
    // again that the number of rows per process is >= numCols.
    // getNodeNumElements() returns a size_t, which is unsigned, and you
    // shouldn't compare signed and unsigned values.
    if (map->getNodeNumElements() < static_cast<size_t>(numCols))
    {
      std::ostringstream os;
      os << "The number of elements on this process " << pComm->getRank()
        << " is too small for the number of columns that you want to test."
        << "  There are " << map->getNodeNumElements() << " elements on "
        "this process, but the normalize() method of the MatOrthoManager "
        "subclass will need to process a multivector with " << numCols
        << " columns.  Not all MatOrthoManager subclasses can handle a "
        "local row block with fewer rows than columns.";
      // QUESTION (mfh 26 Jan 2011) Should this be a logic error
      // instead?  It's really TSQR's fault that it can't handle a
      // local number of elements less than the number of columns.
      throw std::invalid_argument(os.str());
    }

    // Using the factory object, instantiate the specified OrthoManager
    // subclass to be tested.  Specify "fast" parameters for a fair
    // benchmark comparison, but override the fast parameters to get the
    // desired normalization method for SimpleOrthoManaager.
    RCP<OrthoManager<scalar_type, MV> > orthoMan;
    {
      std::string label (orthoManName);
      RCP<ParameterList> params =
        parameterList (*(factory.getFastParameters (orthoManName)));
      if (orthoManName == "Simple") {
        params->set ("Normalization", normalization);
        label = label + " (" + normalization + " normalization)";
      }
      orthoMan = factory.makeOrthoManager (orthoManName, M, outMan, label, params);
    }

    // "Prototype" multivector.  The test code will use this (via
    // Belos::MultiVecTraits) to clone other multivectors as necessary.
    // (This means the test code doesn't need the Map, and it also makes
    // the test code independent of the idea of a Map.)  We only have to
    // allocate one column, because the entries are S are not even read.
    // (We could allocate zero columns, if the MV object allows it.  We
    // play it safe and allocate 1 column instead.)
    RCP<MV> X = rcp (new MV (map, 1));

    // "Compact" mode means that we have to override
    // TimeMonitor::summarize(), which both handles multiple MPI
    // processes correctly (only Rank 0 prints to std::cout), and prints
    // verbosely in a table form.  We deal with the former by making an
    // ostream which is std::cout on Rank 0, and prints nothing (is a
    // "bit bucket") elsewhere.  We deal with the latter inside the
    // benchmark itself.
    Teuchos::oblackholestream bitBucket;
    std::ostream& resultStream =
      (displayResultsCompactly && Teuchos::rank(*pComm) != 0) ? bitBucket : std::cout;

    // Benchmark the OrthoManager subclass.
    typedef Belos::Test::OrthoManagerBenchmarker<scalar_type, MV> benchmarker_type;
    benchmarker_type::benchmark (orthoMan, orthoManName, normalization, X,
        numCols, numBlocks, numTrials,
        outMan, resultStream, displayResultsCompactly);

    const auto timer_map_vectors = Teuchos::TimeMonitor::getFineGrainTiming ();

    const bool ignore_first = true;

    int commSize;
    int rank;
    MPI_Comm_size (MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    std::stringstream ss;

    for (const auto& kv : timer_map_vectors) {
      // pack the timer data into text form
      fineGrainMultiplyTimersToCSV (kv.first,rank, kv.second, ss, ignore_first);
    }

    std::stringstream all_strings;

    // collective gather, rank 0 will fill all_strings
    gatherStrings (ss.str (), all_strings);

    if (rank == 0)
    {
      const std::string header = fineGrainMultiplyTimersToCSV ();
      writeDetailsFile ("details.csv", all_strings, header);
    }



    success = true;

    // Only Rank 0 gets to write to cout.
    if (Teuchos::rank(*pComm) == 0)
      std::cout << "End Result: TEST PASSED" << endl;
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}

extern "C" {
  void F77_BLAS_MANGLE(ilaver,ILAVER) (int* major,int* minor,int* patch);
}

std::string getLapackVersion ()
{
  int major, minor, patch;

  F77_BLAS_MANGLE(ilaver,ILAVER) (&major, &minor, &patch);

  std::stringstream ss;
  ss << major << "." << minor << "." << patch;

  return ss.str ();
}

#ifdef MKL_LIB

  #include <mkl_version.h>

  std::string getBlasLibVersion ()
  {
    /*
        #define __INTEL_MKL_BUILD_DATE 20151022

        #define __INTEL_MKL__ 11
        #define __INTEL_MKL_MINOR__ 3
        #define __INTEL_MKL_UPDATE__ 1
     */

    std::stringstream ss;

    ss << "MKL " << __INTEL_MKL__ << "." << __INTEL_MKL_MINOR__ << "." << __INTEL_MKL_UPDATE__ << "." << __INTEL_MKL_BUILD_DATE;
    return ss.str ();
  }
#elif OPENBLAS_LIB

  #include <openblas_config.h>

  std::string getBlasLibVersion ()
  {
    /*
        #define OPENBLAS_VERSION " OpenBLAS 0.2.19.dev "
     */

    std::stringstream ss;

    ss << OPENBLAS_VERSION;
    return ss.str ();
  }
#elif LIBSCI_LIB
  // No API for libsci... Compilation should pass the version
  // -DLIBSCI_VERSION=\"${LIBSCI_VERSION}\"
  std::string getBlasLibVersion ()
  {
    char * libsci_ver = LIBSCI_VERSION;

    std::stringstream ss;
    ss << "Libsci " << libsci_ver;
    return ss.str ();
  }

#else
  #warning("No define is specified for the blas lib being linked. Please Define one of: -DMKL_LIB, -DOPENBLAS_LIB")
  std::string getBlasLibVersion ()
  {
    return "UNKNOWN";
  }
#endif


std::string getDateTime ()
{
  using std::chrono::system_clock;

  auto current_time_point = system_clock::now();

  auto current_ctime = system_clock::to_time_t(current_time_point);
  std::tm now_tm = *std::localtime(&current_ctime);

  char s[1000];
  std::strftime(s, 1000, "%c", &now_tm);

  return std::string(s);
}

std::string getHostname ()
{
  // get the host this process ran on
  char name[MPI_MAX_PROCESSOR_NAME];
  int len;
  MPI_Get_processor_name( name, &len );

  return std::string(name);
}

std::string getOMP_WAIT_POLICY ()
{
  // query the OMP env
  if(const char* env_omp_wait = std::getenv("OMP_WAIT_POLICY"))
  {
    return std::string(env_omp_wait);
  }
  else
  {
    return std::string("NOT SET");
  }
}

std::string getOMP_PLACES ()
{
  // query the OMP env
  if(const char* env_omp_places = std::getenv("OMP_PLACES"))
  {
    return std::string(env_omp_places);
  }
  else
  {
    return std::string("NOT SET");
  }
}

std::string quote (const std::string& str)
{
  return "\"" + str + "\"";
}


void
fineGrainMultiplyTimersToCSV (const std::string& identifier,
                              const int rank,
                              const std::vector<double>& timings,
                              std::stringstream& ss,
                              const bool ignore_first)
{
  // Tpetra::Multiply::alpha_1_beta_0_A_C_x_B_N::global::14929920x9_x_14929920x1::local::3732480x9_x_3732480x1::pre_gemm
  using std::cout;
  using std::endl;

  int a_dim1_glb, a_dim2_glb, b_dim1_glb, b_dim2_glb;
  int a_dim1_lcl, a_dim2_lcl, b_dim1_lcl, b_dim2_lcl;
  double alpha, beta;
  std::string A_trans;
  std::string B_trans;
  std::string stage;
  std::string operation;
  int numProcs;

  const std::string hostname    = quote (getHostname ());
  const std::string timestamp   = quote (getDateTime ());
  const std::string lapack_ver  = quote (getLapackVersion ());
  const std::string blaslib     = quote (getBlasLibVersion ());
  const std::string wait_policy = quote (getOMP_WAIT_POLICY ());
  const std::string omp_places  = quote (getOMP_PLACES ());

  std::stringstream regex_ss;
  std::string numeric_regex = R"(([-+]?[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?))";
  std::string integer_regex = R"((\d+))";
  std::string text_regex = R"((\w+))";
  regex_ss << "Tpetra::"
           << text_regex
           << "::alpha_"
           << numeric_regex
           << "_beta_"
           << numeric_regex
           << "_A_([CTN])_x_B_([CTN])::global::"
           << integer_regex
           << "x"
           << integer_regex
           << "_x_"
           << integer_regex
           << "x"
           << integer_regex
           << "::local::"
           << integer_regex
           << "x"
           << integer_regex
           << "_x_"
           << integer_regex
           << "x"
           << integer_regex
           << "::"
           << text_regex;
  std::regex e (regex_ss.str ());
  std::smatch sm; // match into strings
  std::regex_match (identifier,sm,e);


  if ( sm.size () != (14+1))
  {
    std::stringstream err_ss;
    err_ss   << "Regex failed to read 14 arguments"
             << endl
             << "Read " << sm.size ()
             << endl
             << "Label:"
             << endl
             << identifier
             << endl
             << "Regex:"
             << endl
             << regex_ss.str ()
             << endl;

    for (int i=0; i < sm.size (); ++i)
    {
      err_ss << "Match[" << i << "] "
             << sm[i]
             << endl;
    }

    std::cerr << err_ss.str ();
    MPI_Abort(MPI_COMM_WORLD, -1);
    return;
  }
  else
  {
    using std::stod;
    using std::stoi;

    operation = sm[1];
    alpha = stod(sm[2]);
    beta  = stod(sm[3]);
    A_trans = sm[4];
    B_trans = sm[5];
    a_dim1_glb = stoi(sm[6]);
    a_dim2_glb = stoi(sm[7]);
    b_dim1_glb = stoi(sm[8]);
    b_dim2_glb = stoi(sm[9]);
    a_dim1_lcl = stoi(sm[10]);
    a_dim2_lcl = stoi(sm[11]);
    b_dim1_lcl = stoi(sm[12]);
    b_dim2_lcl = stoi(sm[13]);
    stage = sm[14];
  }

  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  std::stringstream ss_detail_label;
  ss_detail_label << "Tpetra::" << operation << "::" << stage;
  const std::string detail_label = quote (ss_detail_label.str ());
  const std::string label = quote (
                            (alpha == 1.0 && beta == 0.0) ? "innerProduct"
                          : (alpha == -1.0 && beta == 1.0) ? "Update"
                          : detail_label
                          );
  const int m = a_dim1_lcl;
  const int n = a_dim2_lcl; // global and local are the same

  // Label, hostname, rank, np, OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PLACES, m (local), n, time (ns), Date, blaslib, lapack_version
  // , detail_label, a_dim1_global, a_dim2_global, b_dim1_global, b_dim2_global
  //               , a_dim1_local, a_dim2_local, b_dim1_local, b_dim2_local

  // skip the first?
  auto timing = timings.cbegin () + ((ignore_first) ? 1 : 0);

  // create a huge string blob
  for (; timing != timings.cend (); ++timing)
  {
    ss << label
        << ","
       << hostname
       << ","
       << rank
       << ","
       << numProcs
       << ","
       << omp_get_max_threads ()
       << ","
       << wait_policy
       << ","
       << omp_places
       << ","
       << m
       << ","
       << n
       << ","
       << *timing
       << ","
       << timestamp
       << ","
       << blaslib
       << ","
       << lapack_ver
       << ","
       << detail_label
       << ","
       << a_dim1_glb
       << ","
       << a_dim2_glb
       << ","
       << b_dim1_glb
       << ","
       << b_dim2_glb
       << ","
       << a_dim1_lcl
       << ","
       << a_dim2_lcl
       << ","
       << b_dim1_lcl
       << ","
       << b_dim2_lcl
       << endl;
  }
}

void writeDetailsFile (const std::string& filename, const std::stringstream& data, const std::string& header)
{
  using std::ofstream;
  ofstream ofs (filename.c_str (), std::ofstream::trunc | std::ofstream::out);

  // no error checking for now

  ofs << header
      << data.str ();

  ofs.close ();
}

void
gatherStrings (const std::string& str, std::stringstream& ss)
{
  int commSize;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> string_lengths;
  if (rank == 0)
  {
    string_lengths.reserve (commSize);
    string_lengths.resize (commSize);
  }

  const char * cstr = str.c_str ();
  int length = std::char_traits<char>::length(cstr);

  MPI_Gather(&length, 1, MPI_INT, &string_lengths[0], 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs;
  if (rank == 0)
  {
    displs.reserve (commSize);
    displs.resize (commSize);
  }

  size_t total_size = 0;

  if (rank == 0)
  {
    for (int i=0; i < commSize; ++i)
    {
      displs[i] = total_size;
      total_size += string_lengths[i];
    }
  }

  if (total_size > INT_MAX)
  {
    MPI_Abort (MPI_COMM_WORLD, -1);
  }

  std::vector<char> cstrs;
  if (rank == 0)
  {
    // one more byte for the null character
    cstrs.reserve (total_size+1);
    cstrs.resize (total_size+1);
    cstrs[total_size] = '\0';
  }

  MPI_Gatherv (const_cast<char *> (cstr), length, MPI_CHAR,
               &cstrs[0], &string_lengths[0], &displs[0], MPI_CHAR, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    const char * tmp = &cstrs[0];
    ss << tmp;
  }
}


std::string
fineGrainMultiplyTimersToCSV ()
{
  // Tpetra::Multiply::alpha_1_beta_0_A_C_x_B_N::global::14929920x9_x_14929920x1::local::3732480x9_x_3732480x1::pre_gemm
  using std::cout;
  using std::endl;

  // Label, hostname, rank, np, OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PLACES, m (local), n, time (ns), Date, blaslib, lapack_version
  // , detail_label, a_dim1_global, a_dim2_global, b_dim1_global, b_dim2_global
  //               , a_dim1_local, a_dim2_local, b_dim1_local, b_dim2_local

  std::stringstream ss;
  ss << "Label, hostname, rank, np, OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PLACES, m (local), n, time (ns),"
     << "Date, blaslib, lapack_version, detail_label, a_dim1_global, a_dim2_global, b_dim1_global, b_dim2_global, "
     << "a_dim1_local, a_dim2_local, b_dim1_local, b_dim2_local"
     << endl;
  return ss.str ();
}

