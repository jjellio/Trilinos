// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
// P4 optional Fabric capability discovery.  Probe is deliberately nonfatal:
// a supported CUDA configuration can run this test and report capability
// without attempting an MPI-peer VMM mapping or calling the aborting arena.

#include "VmmP4Helpers.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>

namespace {

// CUDA 12.4 added CU_MEM_HANDLE_TYPE_FABRIC. A successful individual
// cuMemCreate is a *preflight*, not a guarantee that all IPC peers can import
// one another's allocations (fabric permissions/topology still matter).
std::string fabricProbeReason() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12040
  if (cuInit(0) != CUDA_SUCCESS) return "cu_init";
  int runtimeDevice = 0;
  if (cudaGetDevice(&runtimeDevice) != cudaSuccess) return "get_device";
  CUdevice device = 0;
  if (cuDeviceGet(&device, runtimeDevice) != CUDA_SUCCESS) return "cu_device";
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  std::size_t bytes = 0;
  if (cuMemGetAllocationGranularity(&bytes, &prop,
                                   CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS ||
      bytes == 0) return "fabric_granularity";
  CUmemGenericAllocationHandle handle = 0;
  if (cuMemCreate(&handle, bytes, &prop, 0) != CUDA_SUCCESS)
    return "fabric_allocation";
  if (cuMemRelease(handle) != CUDA_SUCCESS) return "fabric_release";
  return "ok";
#else
  return "cuda_headers_pre_12_4";
#endif
}

}  // namespace

TEUCHOS_UNIT_TEST(VMM_FabricProbe, F01_FabricCapability) {
  using namespace TpetraVmmP4;
  if (!twoRanks(out, success)) return;
  const auto comm = Tpetra::getDefaultComm();
  const std::string reason = fabricProbeReason();
  const int localAvailable = reason == "ok" ? 1 : 0;
  int globallyAvailable = 0;
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1,
                     &localAvailable, &globallyAvailable);
  std::cout << "[VMM_P4] event=fabricProbe rank=" << comm->getRank()
            << " supported=" << localAvailable
            << " reason=" << reason << '\n' << std::flush;
  if (comm->getRank() == 0) {
    out << "Fabric individual-GPU capability: "
        << (globallyAvailable ? "YES" : "NO (optional test skipped)")
        << "; peer IPC is tested only by an explicit Fabric run.\n";
  }
  // The capability probe itself PASSes even when unsupported, enabling an
  // honest SKIP from the optional standalone runner.
}
