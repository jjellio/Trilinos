// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <cuda_runtime.h>
#include <nvshmem.h>

namespace NvshmemP0 {
namespace {
__global__ void scalarGetKernel(const double* symmetricSource,
                                int sourcePe, double* localDeviceResult) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    localDeviceResult[0] = nvshmem_double_g(symmetricSource, sourcePe);
  }
}
}  // namespace

cudaError_t launchScalarGet(const double* symmetricSource,
                            int sourcePe,
                            double* localDeviceResult,
                            cudaStream_t stream) {
  scalarGetKernel<<<1, 1, 0, stream>>>(symmetricSource, sourcePe, localDeviceResult);
  return cudaGetLastError();
}
}  // namespace NvshmemP0
