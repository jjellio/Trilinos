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

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_as.hpp"
#include "Teuchos_Hashtable.hpp"


namespace {


TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( Hashtable, test0, Key, Value )
{
  using Teuchos::as;

  Teuchos::Hashtable<Key,Value> hashtable;

  hashtable.put(as<Key>(1), as<Value>(1));
  hashtable.put(as<Key>(3), as<Value>(9));
  hashtable.put(as<Key>(5), as<Value>(7));

  TEST_EQUALITY( hashtable.size(), 3 );

  TEST_EQUALITY( hashtable.containsKey(as<Key>(3)), true );
  TEST_EQUALITY( hashtable.containsKey(as<Key>(4)), false );

  TEST_EQUALITY( hashtable.get(as<Key>(5)), as<Value>(7) );
}

//
// Instantiations
//


#define UNIT_TEST_GROUP( K, V ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( Hashtable, test0, K, V )

UNIT_TEST_GROUP(int, int)
UNIT_TEST_GROUP(int, float)

} // namespace
