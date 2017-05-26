/*
// @HEADER
// ***********************************************************************
//
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// ***********************************************************************
// @HEADER
*/

#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_Version.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

using namespace Teuchos;

// Global Timers
RCP<Time> CompTime = TimeMonitor::getNewCounter("Quad Inner");
RCP<Time> FactTime = TimeMonitor::getNewCounter("Factorial Inner");

// Quadratic function declaration.
double quadFunc( double x );

// Factorial function declaration.
double factFunc( int x );


int main(int argc, char* argv[])
{
  std::cout << Teuchos::Teuchos_Version() << std::endl << std::endl;

  int i;
  double x;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  RCP<Time> TotalTime = TimeMonitor::getNewCounter("Total Time");
  RCP<Time> CompTime2 = TimeMonitor::getNewCounter("Quad Outer");
  RCP<Time> FacTime2 = TimeMonitor::getNewCounter("Factorial Outer");
  int constexpr MAX_EXP = 1500;
  {
    Teuchos::TimeMonitor LocalTimer(*TotalTime);
    for (int k=0; k < MAX_EXP; ++k) {

      MPI_Barrier(MPI_COMM_WORLD);
      {
        Teuchos::TimeMonitor LocalTimer1(*CompTime2);
        // Apply the quadratic function.
        for( i=-5000; i<5000; i++ ) {
          x = quadFunc( (double) i );
          (void)x; // Not used!
        }
      }

      {
        Teuchos::TimeMonitor LocalTimer2(*FacTime2);
        // Apply the factorial function.
        for( i=0; i<100; i++ ) {
          x = factFunc( i );
          (void)x; // Not used!
        }
      }

      if (k % 100 == 0) {
        // Get a summary from the time monitor.
        TimeMonitor::summarize();
      }
    }
  }

  // Get a summary from the time monitor.
  TimeMonitor::summarize();

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}

/* Evaluate a quadratic function at point x */
double quadFunc( double x )
{
  // Construct a local time monitor, this starts the CompTime timer and will stop when leaving scope.
  Teuchos::TimeMonitor LocalTimer(*CompTime);

  double v = 0.0;
  // Evaluate the quadratic function.
  for (int i=1; i<10001; ++i) {
    v += (x*x -1.0)/10000.0;
  }
  return ( v );
}

/* Compute the factorial of x */
double factFunc( int x )
{
  // Construct a local time monitor, this starts the FactTime timer and will stop when leaving scope.
  Teuchos::TimeMonitor LocalTimer(*FactTime);

  // Special returns for specific cases.
  if( x == 0 ) return 0.0;
  if( x == 1 ) return 1.0;

  // Evaluate the factorial function.
  return ( (double) x * factFunc(x-1) );
}
