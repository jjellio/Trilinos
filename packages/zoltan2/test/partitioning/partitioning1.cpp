#include <iostream>
#include <limits>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Vector.hpp>
#include <MatrixMarket_Tpetra.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_XpetraCrsMatrixInput.hpp>
#include <Zoltan2_XpetraVectorInput.hpp>

//#include <Zoltan2_Memory.hpp>  KDD User app wouldn't include our memory mgr.

#include <useMueLuGallery.hpp>

#ifdef SHOW_LINUX_MEMINFO
extern "C"{
static char *meminfo=NULL;
extern void Zoltan_get_linux_meminfo(char *msg, char **result);
}
#endif

using Teuchos::RCP;
using namespace std;

/////////////////////////////////////////////////////////////////////////////
// Program to demonstrate use of Zoltan2 to partition a TPetra matrix 
// (read from a MatrixMarket file or generated by MueLuGallery).
// Usage:
//     a.out [--inputFile=filename.mtx] [--outputFile=outfile.mtx] [--verbose] 
//           [--x=#] [--y=#] [--z=#] [--matrix={Laplace1D,Laplace2D,Laplace3D}
// Karen Devine, 2011
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Eventually want to use Teuchos unit tests to vary z2TestLO and
// GO.  For now, we set them at compile time.
typedef int  z2TestLO;
typedef long z2TestGO;

typedef double Scalar;
typedef Kokkos::DefaultNode::DefaultNodeType Node;
typedef Tpetra::CrsMatrix<Scalar, z2TestLO, z2TestGO> SparseMatrix;
typedef Tpetra::Vector<Scalar, z2TestLO, z2TestGO> Vector;

typedef Zoltan2::XpetraCrsMatrixInput<SparseMatrix> SparseMatrixAdapter;
typedef Zoltan2::XpetraVectorInput<Vector> VectorAdapter;

#define epsilon 0.00000001

/////////////////////////////////////////////////////////////////////////////
int main(int narg, char** arg)
{
  std::string inputFile = "";            // Matrix Market file to read
  std::string outputFile = "";           // Matrix Market file to write
  bool verbose = false;                  // Verbosity of output
  int testReturn = 0;

  ////// Establish session.
  Teuchos::GlobalMPISession mpiSession(&narg, &arg, NULL);
  RCP<const Teuchos::Comm<int> > comm =
    Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  int me = comm->getRank();

  // Read run-time options.
  Teuchos::CommandLineProcessor cmdp (false, false);
  cmdp.setOption("inputFile", &inputFile,
                 "Name of the Matrix Market sparse matrix file to read; "
                 "if not specified, a matrix will be generated by MueLu.");
  cmdp.setOption("outputFile", &outputFile,
                 "Name of the Matrix Market sparse matrix file to write, "
                 "echoing the input/generated matrix.");
  cmdp.setOption("verbose", "quiet", &verbose,
                 "Print messages and results.");

  //////////////////////////////////
  // Even with cmdp option "true", I get errors for having these
  //   arguments on the command line.  (On redsky build)
  // KDDKDD Should just be warnings, right?  Code should still work with these
  // KDDKDD params in the create-a-matrix file.  Better to have them where
  // KDDKDD they are used.
  int xdim=10;
  int ydim=10;
  int zdim=10;
  std::string matrixType("Laplace3D");

  cmdp.setOption("x", &xdim,
                "number of gridpoints in X dimension for "
                "mesh used to generate matrix.");
  cmdp.setOption("y", &ydim,
                "number of gridpoints in Y dimension for "
                "mesh used to generate matrix.");
  cmdp.setOption("z", &zdim,              
                "number of gridpoints in Z dimension for "
                "mesh used to generate matrix.");
  cmdp.setOption("matrix", &matrixType,
                "Matrix type: Laplace1D, Laplace2D, or Laplace3D");
  //////////////////////////////////

  cmdp.parse(narg, arg);


#ifdef SHOW_LINUX_MEMINFO
  if (me == 0){
    Zoltan_get_linux_meminfo("Before creating matrix", &meminfo);
    if (meminfo){
      std::cout << "Rank " << me << ": " << meminfo << std::endl;
      free(meminfo);
      meminfo=NULL;
    }
  }
#endif

  ////// Read or construct a matrix using Tpetra or MueLu

  RCP<SparseMatrix> origMatrix;
  if (inputFile != "") { // Input file specified; read a matrix

    // Need a node for the MatrixMarket reader.
    Teuchos::ParameterList defaultParameters;
    RCP<Node> node = rcp(new Node(defaultParameters));

    origMatrix =
      Tpetra::MatrixMarket::Reader<SparseMatrix>::readSparseFile(
                                         inputFile, comm, node, 
                                         true, false, true);
  }
  else { // Let MueLu generate a matrix
    origMatrix = useMueLuGallery<Scalar, z2TestLO, z2TestGO>(narg, arg, comm);
  }

  if (outputFile != "") {
    // Just a sanity check.
    Tpetra::MatrixMarket::Writer<SparseMatrix>::writeSparseFile(outputFile,
                                                origMatrix, verbose);
  }

  if (me == 0) 
    cout << "NumRows     = " << origMatrix->getGlobalNumRows() << endl
         << "NumNonzeros = " << origMatrix->getGlobalNumEntries() << endl
         << "NumProcs = " << comm->getSize() << endl;

#ifdef SHOW_LINUX_MEMINFO
  if (me == 0){
    Zoltan_get_linux_meminfo("After creating matrix", &meminfo);
    if (meminfo){
      std::cout << "Rank " << me << ": " << meminfo << std::endl;
      free(meminfo);
      meminfo=NULL;
    }
  }
#endif

  ////// Create a vector to use with the matrix.
  RCP<Vector> origVector, origProd;
  origProd   = Tpetra::createVector<Scalar,z2TestLO,z2TestGO>(
                                    origMatrix->getRangeMap());
  origVector = Tpetra::createVector<Scalar,z2TestLO,z2TestGO>(
                                    origMatrix->getDomainMap());
  origVector->randomize();

  ////// Specify problem parameters
  Teuchos::ParameterList params;
  Teuchos::ParameterList partitioningParams = 
    params.sublist("partitioning");
  
  partitioningParams.set("APPROACH", "PARTITION");
  partitioningParams.set("ALGORITHM", "SCOTCH");

  ////// Create an input adapter for the Tpetra matrix.
  SparseMatrixAdapter adapter(origMatrix);

  ////// Create and solve partitioning problem
  Zoltan2::PartitioningProblem<SparseMatrixAdapter> problem(&adapter, &params);

#ifdef SHOW_LINUX_MEMINFO
  if (me == 0){
    Zoltan_get_linux_meminfo("After creating problem", &meminfo);
    if (meminfo){
      std::cout << "Rank " << me << ": " << meminfo << std::endl;
      free(meminfo);
      meminfo=NULL;
    }
  }
#endif

  try {
    if (me == 0) cout << "Calling solve() " << endl;
    problem.solve();
    if (me == 0) cout << "Done solve() " << endl;
  }
  catch (std::runtime_error &e) {
    cout << "Runtime exception returned from solve(): " << e.what();
    if (!strncmp(e.what(), "BUILD ERROR", 11)) {
      // Catching build errors as exceptions is OK in the tests
      cout << " PASS" << endl;
      return 0;
    }
    else {
      // All other runtime_errors are failures
      cout << " FAIL" << endl;
      return -1;
    }
  }
  catch (std::logic_error &e) {
    cout << "Logic exception returned from solve(): " << e.what()
         << " FAIL" << endl;
    return -1;
  }
  catch (std::bad_alloc &e) {
    cout << "Bad_alloc exception returned from solve(): " << e.what()
         << " FAIL" << endl;
    return -1;
  }
  catch (std::exception &e) {
    cout << "Unknown exception returned from solve(). " << e.what()
         << " FAIL" << endl;
    return -1;
  }

  ////// Basic metric checking of the partitioning solution
  ////// Not ordinarily done in application code; just doing it for testing here.
  size_t checkNparts = problem.getSolution().getNumParts();
  size_t checkLength;
  size_t  *checkParts = problem.getSolution().getParts(&checkLength);

  // Check number of parts
  if (me == 0) 
    cout << "Number of parts:  " << checkNparts 
         << (checkNparts != comm->getSize()
                         ? "Wrong number of parts; FAIL"
                         : " ")
         << endl;

  // Check for load balance
  size_t *countPerPart = new size_t[checkNparts];
  size_t *globalCountPerPart = new size_t[checkNparts];
  for (size_t i = 0; i < checkNparts; i++) countPerPart[i] = 0;
  for (size_t i = 0; i < checkLength; i++) {
    if (checkParts[i] >= checkNparts) cout << "Invalid Part:  FAIL" << endl;
    countPerPart[checkParts[i]]++;
  }
  Teuchos::reduceAll<int, size_t>(*comm, Teuchos::REDUCE_SUM, checkNparts,
                                  countPerPart, globalCountPerPart);

  size_t min = std::numeric_limits<std::size_t>::max();
  size_t max = 0;
  size_t sum = 0;
  size_t minrank = 0, maxrank = 0;
  for (size_t i = 0; i < checkNparts; i++) {
    if (globalCountPerPart[i] < min) {min = globalCountPerPart[i]; minrank = i;}
    if (globalCountPerPart[i] > max) {max = globalCountPerPart[i]; maxrank = i;}
    sum += globalCountPerPart[i];
  }
  delete [] countPerPart;
  delete [] globalCountPerPart;

  if (me == 0) {
    float avg = (float) sum / (float) checkNparts;
    cout << "Minimum load:  " << min << " on rank " << minrank << endl;
    cout << "Maximum load:  " << max << " on rank " << maxrank << endl;
    cout << "Average load:  " << avg << endl;
    cout << "Total load:    " << sum 
         << (sum != origMatrix->getGlobalNumRows()
                 ? "Work was lost; FAIL"
                 : " ")
         << endl;
    cout << "Imbalance:     " << max / avg << endl;
  }

  ////// Redistribute matrix and vector into new matrix and vector.
  if (me == 0) cout << "Redistributing matrix..." << endl;
  SparseMatrix *redistribMatrix;
  adapter.applyPartitioningSolution(*origMatrix, redistribMatrix,
                                    problem.getSolution());

  if (me == 0) cout << "Redistributing vectors..." << endl;
  Vector *redistribVector;
  VectorAdapter adapterVector(origVector);
  adapterVector.applyPartitioningSolution(*origVector, redistribVector,
                                          problem.getSolution());

  RCP<Vector> redistribProd;
  redistribProd = Tpetra::createVector<Scalar,z2TestLO,z2TestGO>(
                                       redistribMatrix->getRangeMap());


  ////// Verify that redistribution is "correct"; perform matvec with 
  ////// original and redistributed matrices/vectors and compare norms.

  if (me == 0) cout << "Matvec original..." << endl;
  origMatrix->apply(*origVector, *origProd);
  Scalar origNorm = origProd->norm2();
  if (me == 0)
    cout << "Norm of Original matvec prod:       " << origNorm << endl;

  if (me == 0) cout << "Matvec redistributed..." << endl;
  redistribMatrix->apply(*redistribVector, *redistribProd);
  Scalar redistribNorm = redistribProd->norm2();
  if (me == 0)
    cout << "Norm of Redistributed matvec prod:  " << redistribNorm << endl;

  if (redistribNorm > origNorm+epsilon || redistribNorm < origNorm-epsilon) 
    testReturn = 1;

  delete redistribVector;
  delete redistribMatrix;

  if (me == 0) {
    if (testReturn)
      std::cout << "Mat-Vec product changed; FAIL" << std::endl;
    else
      std::cout << "PASS" << std::endl;
  }

  return testReturn;
}
