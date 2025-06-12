#include <cuda_contractions.cuh>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>

// Specialized kernel for M=3 with full unrolling
template<typename T, size_t N>
__global__ void tensorContractKernelM3(
    const T* __restrict__ tensor4d,    // Shape: (N, 3, 3, 3)
    const T* __restrict__ tensor2d_1,  // Shape: (N, 3)
    const T* __restrict__ tensor2d_2,  // Shape: (N, 3)
    T* __restrict__ result            // Shape: (N, 3)
) {
    const int a = blockIdx.x;
    const int i = threadIdx.x;
    
    if (a >= N || i >= 3) return;
    
    // Direct register storage for M=3
    T val1_0 = tensor2d_1[a * 3 + 0];
    T val1_1 = tensor2d_1[a * 3 + 1];
    T val1_2 = tensor2d_1[a * 3 + 2];
    
    T val2_0 = tensor2d_2[a * 3 + 0];
    T val2_1 = tensor2d_2[a * 3 + 1];
    T val2_2 = tensor2d_2[a * 3 + 2];
    
    const int base = a * 27 + i * 9;
    
    // Fully unrolled loops for M=3
    T sum = tensor4d[base + 0] * val1_0 * val2_0 +
            tensor4d[base + 1] * val1_0 * val2_1 +
            tensor4d[base + 2] * val1_0 * val2_2 +
            tensor4d[base + 3] * val1_1 * val2_0 +
            tensor4d[base + 4] * val1_1 * val2_1 +
            tensor4d[base + 5] * val1_1 * val2_2 +
            tensor4d[base + 6] * val1_2 * val2_0 +
            tensor4d[base + 7] * val1_2 * val2_1 +
            tensor4d[base + 8] * val1_2 * val2_2;
    
    result[a * 3 + i] = sum;
}

// Specialized kernel for M=8 with partial unrolling
template<typename T, size_t N>
__global__ void tensorContractKernelM8(
    const T* __restrict__ tensor4d,    // Shape: (N, 8, 8, 8)
    const T* __restrict__ tensor2d_1,  // Shape: (N, 8)
    const T* __restrict__ tensor2d_2,  // Shape: (N, 8)
    T* __restrict__ result            // Shape: (N, 8)
) {
    const int a = blockIdx.x;
    const int i = threadIdx.x;
    
    if (a >= N || i >= 8) return;
    
    // Use registers for small arrays
    T val1[8], val2[8];
    
    #pragma unroll
    for (int idx = 0; idx < 8; ++idx) {
        val1[idx] = tensor2d_1[a * 8 + idx];
        val2[idx] = tensor2d_2[a * 8 + idx];
    }
    
    T sum = 0;
    const int base = a * 512 + i * 64;
    
    // Unroll outer loop completely, inner loop partially
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        T temp = 0;
        #pragma unroll 4
        for (int k = 0; k < 8; ++k) {
            temp += tensor4d[base + j * 8 + k] * val2[k];
        }
        sum += val1[j] * temp;
    }
    
    result[a * 8 + i] = sum;
}

// Generic kernel for arbitrary M
template<typename T, size_t N, size_t M>
__global__ void tensorContractKernel(
    const T* __restrict__ tensor4d,    // Shape: (N, M, M, M)
    const T* __restrict__ tensor2d_1,  // Shape: (N, M)
    const T* __restrict__ tensor2d_2,  // Shape: (N, M)
    T* __restrict__ result            // Shape: (N, M)
) {
    int a = blockIdx.z;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (a >= N || i >= M) return;
    
    // Shared memory for tensor2d_1 and tensor2d_2
    extern __shared__ T sharedMem[];
    T* s_tensor2d_1 = sharedMem;
    T* s_tensor2d_2 = &s_tensor2d_1[M];
    
    // Collaboratively load tensor2d_1 and tensor2d_2 into shared memory
    for (int idx = threadIdx.x; idx < M; idx += blockDim.x) {
        s_tensor2d_1[idx] = tensor2d_1[a * M + idx];
        s_tensor2d_2[idx] = tensor2d_2[a * M + idx];
    }
    
    __syncthreads();
    
    T sum = 0;
    int tensor4d_base = a * M * M * M + i * M * M;
    
    // Compute contraction with improved memory access pattern
    for (int j = 0; j < M; ++j) {
        T val1 = s_tensor2d_1[j];
        T temp_sum = 0;
        int jM = j * M;
        
        #pragma unroll 4
        for (int k = 0; k < M; ++k) {
            temp_sum += tensor4d[tensor4d_base + jM + k] * s_tensor2d_2[k];
        }
        sum += val1 * temp_sum;
    }
    
    result[a * M + i] = sum;
}

// Specialized dispatch for M=3
template<typename T, size_t N>
void tensorContractM3(
    const T* tensor4d,
    const T* tensor2d_1,
    const T* tensor2d_2,
    T* result,
    cudaStream_t stream
) {
    dim3 grid(N);
    dim3 block(3);
    
    tensorContractKernelM3<T, N><<<grid, block, 0, stream>>>(
        tensor4d, tensor2d_1, tensor2d_2, result
    );
}

// Specialized dispatch for M=8
template<typename T, size_t N>
void tensorContractM8(
    const T* tensor4d,
    const T* tensor2d_1,
    const T* tensor2d_2,
    T* result,
    cudaStream_t stream
) {
    dim3 grid(N);
    dim3 block(8);
    
    tensorContractKernelM8<T, N><<<grid, block, 0, stream>>>(
        tensor4d, tensor2d_1, tensor2d_2, result
    );
}

// Host function to launch the contraction kernel with specializations
template<typename T, size_t N, size_t M>
void tensorContract(
    const T* tensor4d,    // Shape: (N, M, M, M)
    const T* tensor2d_1,  // Shape: (N, M)
    const T* tensor2d_2,  // Shape: (N, M)
    T* result,            // Shape: (N, M)
    cudaStream_t stream
) {
    if constexpr (M == 3) {
        tensorContractM3<T, N>(tensor4d, tensor2d_1, tensor2d_2, result, stream);
    } else if constexpr (M == 8) {
        tensorContractM8<T, N>(tensor4d, tensor2d_1, tensor2d_2, result, stream);
    } else {
        // Generic implementation
        const int BLOCK_SIZE = (M <= 32) ? 32 : ((M <= 64) ? 64 : 128);
        
        dim3 block(BLOCK_SIZE);
        dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, N);
        
        size_t sharedMemSize = 2 * M * sizeof(T);
        
        tensorContractKernel<T, N, M><<<grid, block, sharedMemSize, stream>>>(
            tensor4d, tensor2d_1, tensor2d_2, result
        );
    }
}

// Explicitly instantiate the templates for the sizes used in the project
template void tensorContract<double, 2048, 3>(const double*, const double*, const double*, double*, cudaStream_t);
template void tensorContract<double, 2048, 8>(const double*, const double*, const double*, double*, cudaStream_t);

