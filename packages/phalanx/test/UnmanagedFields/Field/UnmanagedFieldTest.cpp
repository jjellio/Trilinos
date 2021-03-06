// @HEADER
// ************************************************************************
//
//        Phalanx: A Partial Differential Equation Field Evaluation 
//       Kernel for Flexible Management of Complex Dependency Chains
//                    Copyright 2008 Sandia Corporation
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
// Questions? Contact Roger Pawlowski (rppawlo@sandia.gov), Sandia
// National Laboratories.
//
// ************************************************************************
// @HEADER

#include <limits>

#include "Phalanx_KokkosDeviceTypes.hpp"
#include "Phalanx_Field_UnmanagedAllocator.hpp"
#include "Phalanx_DataLayout_DynamicLayout.hpp"
#include "Phalanx_FieldTag_Tag.hpp"
#include "Phalanx_FieldManager.hpp"
#include "Phalanx_TypeStrings.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_Assert.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_TimeMonitor.hpp"

#include "MyTraits.hpp"

#include "Field_TestEvaluators.hpp"

TEUCHOS_UNIT_TEST(unmanaged_fields, basic)
{
  using namespace std;
  using namespace Teuchos;
  using namespace PHX;

  // Put the field manager outside all evalaution code so that we can
  // test that memory management is handled correctly.
  FieldManager<MyTraits> fm;

  RCP<DataLayout> dl = rcp(new Layout("H-Grad<CELL,BASIS>",100,4));

  const auto tol = std::numeric_limits<double>::epsilon() * 100.0;

  {
    // Register evaluators
    {
      auto plist = Teuchos::parameterList("EvalUnmanaged"); 
      using EU = EvalUnmanaged<MyTraits::Residual,MyTraits>;
      RCP<EU> e = rcp(new EU(*plist));
      e->setName("EvalUnmanaged");
      fm.registerEvaluator<MyTraits::Residual>(e);
    }
    {
      using ED = EvalDummy<MyTraits::Residual,MyTraits>;
      auto plist = Teuchos::parameterList("EvalDummy"); 
      RCP<ED> e = rcp(new ED(*plist));
      e->setName("EvalDummy");
      fm.registerEvaluator<MyTraits::Residual>(e);
    }

    // Require fields
    {
      PHX::Tag<MyTraits::Residual::ScalarT> tag_a("a",dl);
      fm.requireField<MyTraits::Residual>(tag_a);
      
      PHX::Tag<MyTraits::Residual::ScalarT> tag_c("c",dl);
      fm.requireField<MyTraits::Residual>(tag_c);
    }
    

    Field<double,2> unmanaged_a = allocateUnmanagedField<double,2>("a",dl);
    Field<double,2> unmanaged_b = allocateUnmanagedField<double,2>("b",dl);
    Field<double,2> unmanaged_c = allocateUnmanagedField<double,2>("c",dl);
    Field<double,2> unmanaged_d = allocateUnmanagedField<double,2>("d",dl);
    unmanaged_a.deep_copy(0.0);
    unmanaged_b.deep_copy(5.0);
    unmanaged_c.deep_copy(0.0);
    unmanaged_d.deep_copy(6.0);

    for (int cell = 0; cell < unmanaged_a.extent_int(0); ++cell) {
      for (int basis = 0; basis < unmanaged_a.extent_int(1); ++basis) {
        TEST_FLOATING_EQUALITY(unmanaged_a(cell,basis),0.0,tol);
        TEST_FLOATING_EQUALITY(unmanaged_b(cell,basis),5.0,tol);
        TEST_FLOATING_EQUALITY(unmanaged_c(cell,basis),0.0,tol);
        TEST_FLOATING_EQUALITY(unmanaged_d(cell,basis),6.0,tol);
      }
    }
    
    // Register some unmanaged fields before postRegistrationSetup()
    // is called and some unmanaged fields after. There are two
    // branches through the code base depending on whether
    // postRegistrationSetup() has been called and we want to cover
    // both branches in testing.
    fm.setUnmanagedField<MyTraits::Residual>(unmanaged_a);
    fm.setUnmanagedField<MyTraits::Residual>(unmanaged_b);
    fm.postRegistrationSetup(0);
    fm.setUnmanagedField<MyTraits::Residual>(unmanaged_c);
    fm.setUnmanagedField<MyTraits::Residual>(unmanaged_d);
    fm.evaluateFields<MyTraits::Residual>(0);
    
    for (int cell = 0; cell < unmanaged_a.extent_int(0); ++cell) {
      for (int basis = 0; basis < unmanaged_a.extent_int(1); ++basis) {
        TEST_FLOATING_EQUALITY(unmanaged_a(cell,basis),5.0,tol);
        TEST_FLOATING_EQUALITY(unmanaged_b(cell,basis),5.0,tol);
        TEST_FLOATING_EQUALITY(unmanaged_c(cell,basis),6.0,tol);
        TEST_FLOATING_EQUALITY(unmanaged_d(cell,basis),6.0,tol);
      }
    }
  }

  // Once outside unmanaged_view scope, grab the views again and make
  // sure the internally stored views in the field manager are
  // pointing to the same memory.
  Field<double,2> a("a",dl);
  Field<double,2> b("b",dl);
  Field<const double,2> c("c",dl); // test const version
  Field<const double,2> d("d",dl); // test const version
  
  fm.getFieldData<MyTraits::Residual>(a);
  fm.getFieldData<MyTraits::Residual>(b);
  fm.getFieldData<MyTraits::Residual>(c);
  fm.getFieldData<MyTraits::Residual>(d);

  for (int cell = 0; cell < a.extent_int(0); ++cell) {
    for (int basis = 0; basis < a.extent_int(1); ++basis) {
      TEST_FLOATING_EQUALITY(a(cell,basis),5.0,tol);
      TEST_FLOATING_EQUALITY(b(cell,basis),5.0,tol);
      TEST_FLOATING_EQUALITY(c(cell,basis),6.0,tol);
      TEST_FLOATING_EQUALITY(d(cell,basis),6.0,tol);
    }
  }

}
