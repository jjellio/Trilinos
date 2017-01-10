// @HEADER
// ************************************************************************
//
//                           Intrepid Package
//                 Copyright (2007) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
// USA
// Questions? Contact Pavel Bochev  (pbboche@sandia.gov),
//                    Denis Ridzal  (dridzal@sandia.gov),
//                    Kara Peterson (kjpeter@sandia.gov).
//
// ************************************************************************
// @HEADER

#ifndef __TrilinosCouplings_IntrepidPoissonExample_SolveWithBelos_hpp
#define __TrilinosCouplings_IntrepidPoissonExample_SolveWithBelos_hpp

/// \file TrilinosCouplings_IntrepidPoissonExample_SolveWithBelos.hpp
/// \brief Generic Belos solver for the Intrepid Poisson test problem example.

#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosSolverFactory.hpp"

// used for matching solver name
#include <string>
#include <regex>

namespace TrilinosCouplings {
namespace IntrepidPoissonExample {


/// \brief report the linear solvers available in Belos
///
/// Report the names and parameter options available for each
/// solver.
template<class ST, class MV, class OP>
void
reportBelosSolvers (Teuchos::RCP<Teuchos::FancyOStream>& pOut)
{
  using std::string;
  using Teuchos::Array;
  using Belos::SolverFactory;
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using Teuchos::parameterList;

  typedef Teuchos::Array<string> string_array_type;
  typedef Belos::SolverFactory< ST, MV, OP > belos_solver_factory_type;
  typedef Belos::SolverManager<ST , MV, OP> belos_solver_type;

  belos_solver_factory_type factory;
  string_array_type supportedSolvers = factory.supportedSolverNames ();

  string_array_type::iterator it;
  for (it = supportedSolvers.begin (); it != supportedSolvers.end (); ++it)
  {
    const string& solverName = *it;
    RCP<ParameterList> solverParams = parameterList ();

    RCP<belos_solver_type> aSolver = factory.create ( solverName, solverParams);

    *pOut << std::string(80, '-')
          << std::endl
          << solverName
          << std::endl;

    RCP< const ParameterList > solverParams1 = aSolver->getValidParameters ();
    solverParams1->print(*pOut);
  }

}


/// \brief Solve the linear system(s) AX=B with Belos.
///
/// This is a generic solve function: you can use any scalar,
/// multivector, and operator types that Belos supports.  We in turn
/// use this as the implementation of solveWithBelos() functions for a
/// specific Scalar, multivector, and operator type combination.  This
/// hopefully increases parallelism in the build.
///
/// \tparam ST The type of entries in the matrix and vectors.  For
///   Epetra objects, this is always double.  For Tpetra objects, this
///   corresponds to the \c Scalar template parameter of
///   Tpetra::MultiVector and Tpetra::Operator.
///
/// \tparam MV The type of multivectors: for example,
///   Epetra_MultiVector or a Tpetra::MultiVector specialization.
///
/// \tparam OP The type of operators (from multivectors to
///   multivectors): for example, Epetra_Operator or a
///   Tpetra::Operator specialization.
///
/// The X and B arguments are both multivectors, meaning that you may
/// ask Belos to solve more than one linear system at a time.  X and B
/// must have the same number of columns.
///
/// This interface will change in the future to accept the name of the
/// Belos solver to use.  For now, the solver is hard-coded to
/// Pseudoblock CG (implemented by Belos::PseudoBlockCGSolMgr).
///
/// \param converged [out] Whether Belos reported that the iterative
///   method converged (solved the linear system to the desired
///   tolerance).
///
/// \param numItersPerformed [out] Number of iterations that the Belos
///   solver performed.
///
/// \param solverName [in] Name of Belos solver to use.  You may use
///   any name that Belos::SolverFactory understands.
///
/// \param tol [in] Convergence tolerance for the iterative method.
///   The meaning of this depends on the particular iterative method.
///
/// \param maxNumIters [in] Maximum number of iterations that the
///   iterative method should perform, regardless of whether it
///   converged.
///
/// \param restartLength [in] The restart length of the solver (default is
///   no restart).
///
/// \param num_steps [in] Number of "time steps", i.e., the number of
//    times the solver is called in a fake time-step loop.
///
/// \param X [in/out] On input: the initial guess(es) for the iterative
///   method.  On output: the computed approximate solution.
///
/// \param A [in] The matrix in the linear system(s) AX=B to solve.
///
/// \param B [in] The right-hand side(s) in the linear system AX=B to solve.
///
/// \param M_left [in] If nonnull, a left preconditioner that the
///   iterative method may use.  If null, the iterative method will
///   not use a left preconditioner.
///
/// \param M_right [in] If nonnull, a right preconditioner that the
///   iterative method may use.  If null, the iterative method will
///   not use a right preconditioner.
template<class ST, class MV, class OP>
void
solveWithBelos (bool& converged,
                int& numItersPerformed,
                const std::string& solverName,
                const typename Teuchos::ScalarTraits<ST>::magnitudeType& tol,
                const int maxNumIters,
                const int restartLength,
                const int num_steps,
                const Teuchos::RCP<MV>& X,
                const Teuchos::RCP<const OP>& A,
                const Teuchos::RCP<const MV>& B,
                const Teuchos::RCP<const OP>& M_left,
                const Teuchos::RCP<const OP>& M_right)
{
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  typedef Belos::LinearProblem<ST, MV, OP > problem_type;
  typedef Belos::SolverFactory<ST, MV, OP> solver_factory_type;
  typedef Belos::SolverManager<ST, MV, OP> solver_type;
  typedef Belos::MultiVecTraits<ST, MV> MVT;

  // Set these in advance, so that if the Belos solver throws an
  // exception for some reason, these will have sensible values.
  converged = false;
  numItersPerformed = 0;

  TEUCHOS_TEST_FOR_EXCEPTION(A.is_null () || X.is_null () || B.is_null (),
    std::invalid_argument, "solveWithBelos: The A, X, and B arguments must all "
    "be nonnull.");
  const int numColsB = MVT::GetNumberVecs (*B);
  const int numColsX = MVT::GetNumberVecs (*X);
  TEUCHOS_TEST_FOR_EXCEPTION(numColsB != numColsX, std::invalid_argument,
    "solveWithBelos: X and B must have the same number of columns.  X has "
    << numColsX << " columns, but B has " << numColsB << " columns.");

  RCP<ParameterList> belosParams = parameterList ();

  std::stringstream solverLabel;
  solverLabel << "Belos:" << solverName << "(" << restartLength << ")";
  belosParams->set ("Timer Label", solverLabel.str());

  belosParams->set ("Block Size", numColsB);
  belosParams->set ("Maximum Iterations", maxNumIters);
  // belosParams->set ("Num Blocks", maxNumIters);
  // configure restart length to be smaller than maxIt (200)
  belosParams->set ("Num Blocks", restartLength);
  // ensure the solver can restart enough times
  belosParams->set ("Maximum Restarts", maxNumIters);

  belosParams->set ("Convergence Tolerance", tol);

  // safe Gmres match, ignores case and matches substrings
  // e.g., FGmres Block GMRES, etc..
  std::regex gmres_regex ("gmres", std::regex_constants::icase );
  std::smatch sm; // match into strings
  std::regex_match (solverName,sm,gmres_regex);

  //if (solverName == "GMRES") {
  belosParams->set ("Orthogonalization", "ICGS");
  belosParams->set ("maxNumOrthogPasses", 2);
  
belosParams->set ("Output Frequency", 10);
  belosParams->set ("Output Style", 1);
  belosParams->set ("Verbosity", 33);

  RCP<problem_type> problem = rcp (new problem_type (A, X, B));
  if (! M_left.is_null ()) {
    problem->setLeftPrec (M_left);
  }
  if (! M_right.is_null ()) {
    problem->setRightPrec (M_right);
  }

  problem->setLabel ( solverLabel.str () );
  // Create solver
  RCP<solver_type> solver;
  {
    solver_factory_type factory;
    solver = factory.create (solverName, belosParams);
  }

  // Enter "time step" loop -- we're really solving the same system repeatedly
  converged = true;
  numItersPerformed = 0;
  for (int step = 0; step < num_steps; ++step) {
    // Set x
    MVT::MvInit(*X);

    // Reset problem
    const bool set = problem->setProblem ();
    TEUCHOS_TEST_FOR_EXCEPTION(
      ! set, std::runtime_error, "solveWithBelos: The "
      "Belos::LinearProblem's setProblem() method returned false.  "
      "This probably indicates that there is something wrong with A, X, or B.");
    solver->setProblem (problem);

    // Solve
    Belos::ReturnType result = solver->solve ();

    converged = converged && (result == Belos::Converged);
    numItersPerformed += solver->getNumIters ();
  }
}

} // namespace IntrepidPoissonExample
} // namespace TrilinosCouplings

#endif // __TrilinosCouplings_IntrepidPoissonExample_SolveWithBelos_hpp
