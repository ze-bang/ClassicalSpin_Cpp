#ifndef CUDA_CONTRACTIONS_CUH
#define CUDA_CONTRACTIONS_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>

// CUDA kernel for tensor contraction: 'aijk, aj, ak->ai'
template<typename T, size_t N, size_t M>
__global__ void tensorContractKernel(
    const T* tensor4d,    // Shape: (N, M, M, M)
    const T* tensor2d_1,  // Shape: (N, M)
    const T* tensor2d_2,  // Shape: (N, M)
    T* result            // Shape: (N, M)
);

// Host function to launch the contraction kernel
template<typename T, size_t N, size_t M>
void tensorContract(
    const T* tensor4d,    // Shape: (N, M, M, M)
    const T* tensor2d_1,  // Shape: (N, M)
    const T* tensor2d_2,  // Shape: (N, M)
    T* result,            // Shape: (N, M)
    cudaStream_t stream
);

#endif // CUDA_CONTRACTIONS_CUH