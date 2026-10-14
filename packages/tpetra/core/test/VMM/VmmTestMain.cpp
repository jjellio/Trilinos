// @HEADER
// ****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
// SPDX-License-Identifier: BSD-3-Clause
// ****************************************************************************
// @HEADER
//
// P0 test main: initialize Tpetra before the Teuchos test repository and
// reject accidental placement of two MPI ranks on the same physical GPU.

#include "Tpetra_Core.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_UnitTestRepository.hpp"

#include <Kokkos_Core.hpp>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

// All ranks enter this function, including on a rank-local CUDA failure.
// Collective error handling prevents a successful rank from continuing into
// an MPI-based unit test after another rank has failed its GPU checks.
int checkGpuBinding() {
  const auto comm = Tpetra::getDefaultComm();
  const int rank  = comm->getRank();
  const int size  = comm->getSize();

  constexpr int idBytes = 64;
  std::array<char, idBytes> pciBusId{};
  int localOk = 1;
  int device  = -1;

#if defined(KOKKOS_ENABLE_CUDA)
  cudaError_t error = cudaGetDevice(&device);
  if (error == cudaSuccess) {
    error = cudaDeviceGetPCIBusId(pciBusId.data(), idBytes, device);
  }
  if (error != cudaSuccess || pciBusId[0] == '\0') {
    std::cerr << "[VMM_TEST] event=gpuBinding rank=" << rank
              << " result=FAIL reason=cuda_device_query error="
              << cudaGetErrorString(error) << std::endl;
    localOk = 0;
  }
#else
  std::cerr << "[VMM_TEST] event=gpuBinding rank=" << rank
            << " result=FAIL reason=kokkos_cuda_not_enabled" << std::endl;
  localOk = 0;
#endif

  int everyRankOk = 0;
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &localOk, &everyRankOk);
  if (!everyRankOk) {
    if (rank == 0) {
      std::cerr << "[VMM_TEST] event=gpuPreflight result=FAIL"
                << " reason=device_query_failed" << std::endl;
    }
    return 2;
  }

  const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
  std::cout << "[VMM_TEST] event=gpuBinding rank=" << rank
            << " device=" << device << " pci_bus_id=" << pciBusId.data()
            << " cuda_visible_devices=" << (visible ? visible : "<unset>")
            << std::endl;

  // CUDA device ordinals may all be zero when each rank has its own
  // CUDA_VISIBLE_DEVICES.  Compare PCI bus IDs (physical GPU identity), not
  // ordinals or CUDA_VISIBLE_DEVICES strings.
  std::vector<char> allIds(static_cast<std::size_t>(size) * idBytes, '\0');
  Teuchos::gatherAll(*comm, idBytes, pciBusId.data(), size * idBytes, allIds.data());

  int duplicate = 0;
  for (int other = 0; other < size; ++other) {
    if (other != rank &&
        std::strcmp(pciBusId.data(), allIds.data() + other * idBytes) == 0) {
      duplicate = 1;
    }
  }
  int anyDuplicate = 0;
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &duplicate, &anyDuplicate);
  if (anyDuplicate) {
    if (rank == 0) {
      std::cerr << "[VMM_TEST] event=gpuPreflight result=FAIL"
                << " reason=shared_physical_gpu ranks=" << size << std::endl;
    }
    return 2;
  }

  if (rank == 0) {
    std::cout << "[VMM_TEST] event=gpuPreflight result=PASS ranks="
              << size << std::endl;
  }
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  // This switch is consumed here, not passed to the Teuchos CLI parser.
  bool preflightOnly = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--vmm-gpu-preflight-only") == 0) {
      preflightOnly = true;
      for (int j = i; j + 1 < argc; ++j) {
        argv[j] = argv[j + 1];
      }
      --argc;
      --i;
    }
  }

  Tpetra::ScopeGuard tpetraScope(&argc, &argv);
  const int gpuStatus = checkGpuBinding();
  if (gpuStatus != 0 || preflightOnly) {
    return gpuStatus;
  }

  // Every MPI rank must observe a failed assertion as a failed MPI job.
  Teuchos::UnitTestRepository::setGloballyReduceTestResult(true);
  return Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);
}
