TRIBITS_SUBPACKAGE(Unit_tests)

ADD_SUBDIRECTORY(stk_mesh_fixtures)
TRIBITS_ADD_TEST_DIRECTORIES(stk_mesh)
TRIBITS_ADD_TEST_DIRECTORIES(stk_ngp)
TRIBITS_ADD_TEST_DIRECTORIES(stk_ngp_test)

IF ( TPL_ENABLE_BoostLib )
  TRIBITS_ADD_TEST_DIRECTORIES(stk_topology)
ENDIF()

TRIBITS_ADD_TEST_DIRECTORIES(stk_io)
TRIBITS_ADD_TEST_DIRECTORIES(stk_util)
TRIBITS_ADD_TEST_DIRECTORIES(stk_transfer)
TRIBITS_ADD_TEST_DIRECTORIES(stk_tools)

TRIBITS_SUBPACKAGE_POSTPROCESS()
