#include <cuda_contractions.cuh>
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
    
    // Compute contraction sum_{j,k} tensor4d[a,i,j,k] * tensor2d_1[a,j] * tensor2d_2[a,k]
    for (int j = 0; j < M; ++j) {
        T val1 = s_tensor2d_1[j];
        int jM = j * M;
        
        for (int k = 0; k < M; ++k) {
            sum += tensor4d[tensor4d_base + jM + k] * val1 * s_tensor2d_2[k];
        }
    }
    
    result[a * M + i] = sum;
}

// Host function to launch the contraction kernel
template<typename T, size_t N, size_t M>
void tensorContract(
    const T* tensor4d,    // Shape: (N, M, M, M)
    const T* tensor2d_1,  // Shape: (N, M)
    const T* tensor2d_2,  // Shape: (N, M)
    T* result,            // Shape: (N, M)
    cudaStream_t stream
) {
    // Optimize thread block size based on GPU architecture
    const int BLOCK_SIZE = 256;
    
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, N);
    
    // Shared memory size: space for both 2D tensors
    size_t sharedMemSize = 2 * M * sizeof(T);
    
    tensorContractKernel<T, N, M> <<<grid, block, sharedMemSize, stream>>>(
        tensor4d, tensor2d_1, tensor2d_2, result
    );
}

// Explicitly instantiate the templates for the sizes used in the project
// For double type with N=3, M=2048
template void tensorContract<double, 3, 2048>(const double*, const double*, const double*, double*, cudaStream_t);

// For double type with N=8, M=2048
template void tensorContract<double, 8, 2048>(const double*, const double*, const double*, double*, cudaStream_t);

