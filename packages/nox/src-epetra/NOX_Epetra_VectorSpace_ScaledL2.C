// $Id$
// $Source$

//@HEADER
// ************************************************************************
//
//            NOX: An Object-Oriented Nonlinear Solver Package
//                 Copyright (2002) Sandia Corporation
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
// Questions? Contact Roger Pawlowski (rppawlo@sandia.gov) or
// Eric Phipps (etphipp@sandia.gov), Sandia National Laboratories.
// ************************************************************************
//  CVS Information
//  $Source$
//  $Author$
//  $Date$
//  $Revision$
// ************************************************************************
//@HEADER

#include "NOX_Epetra_VectorSpace_ScaledL2.H"
#include "Epetra_Vector.h"

NOX::Epetra::VectorSpaceScaledL2::
VectorSpaceScaledL2(const Teuchos::RCP<NOX::Epetra::Scaling>& s,
            NOX::Epetra::Scaling::ScaleType st) :
  scalingPtr(s),
  scaleType(st)
{

}

NOX::Epetra::VectorSpaceScaledL2::~VectorSpaceScaledL2()
{

}

double NOX::Epetra::VectorSpaceScaledL2::
innerProduct(const Epetra_Vector& a, const Epetra_Vector& b) const
{
  if ( Teuchos::is_null(tmpVectorPtr) )
    tmpVectorPtr = Teuchos::rcp(new Epetra_Vector(a));

  *tmpVectorPtr = a;

  if (scaleType == NOX::Epetra::Scaling::Left) {
    // Do twice on a instead of once on a and once on b.
    scalingPtr->applyLeftScaling(*tmpVectorPtr, *tmpVectorPtr);
    scalingPtr->applyLeftScaling(*tmpVectorPtr, *tmpVectorPtr);
  }
  else {
    // Do twice on a instead of once on a and once on b.
    scalingPtr->applyRightScaling(*tmpVectorPtr, *tmpVectorPtr);
    scalingPtr->applyRightScaling(*tmpVectorPtr, *tmpVectorPtr);
  }

  double dot;
  tmpVectorPtr->Dot(b, &dot);
  return dot;
}

double NOX::Epetra::VectorSpaceScaledL2::
norm(const Epetra_Vector& a, NOX::Abstract::Vector::NormType type) const
{
  if ( Teuchos::is_null(tmpVectorPtr) )
    tmpVectorPtr = Teuchos::rcp(new Epetra_Vector(a));

  *tmpVectorPtr = a;

  if (scaleType == NOX::Epetra::Scaling::Left) {
    scalingPtr->applyLeftScaling(*tmpVectorPtr, *tmpVectorPtr);
  }
  else {
    scalingPtr->applyRightScaling(*tmpVectorPtr, *tmpVectorPtr);
  }

  double value;
  switch (type) {
  case NOX::Abstract::Vector::MaxNorm:
    tmpVectorPtr->NormInf(&value);
    break;
  case NOX::Abstract::Vector::OneNorm:
    tmpVectorPtr->Norm1(&value);
    break;
  case NOX::Abstract::Vector::TwoNorm:
  default:
   tmpVectorPtr->Norm2(&value);
   break;
  }
  return value;
}
