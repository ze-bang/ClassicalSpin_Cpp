#ifndef CUDA_WRAPPER_OPTIMIZED_H
#define CUDA_WRAPPER_OPTIMIZED_H

#include "cuda_contractions.cuh"
#include <vector>

// Device memory manager for persistent structure tensors
class DeviceStructureTensorManager {
private:
    double *d_SU2_structure_tensor = nullptr;
    double *d_SU3_structure_tensor = nullptr;
    size_t SU2_tensor_size = 0;
    size_t SU3_tensor_size = 0;

public:
    DeviceStructureTensorManager() = default;

    ~DeviceStructureTensorManager() {
        cleanup();
    }

    void cleanup() {
        if (d_SU2_structure_tensor) {
            cudaFree(d_SU2_structure_tensor);
            d_SU2_structure_tensor = nullptr;
        }

        if (d_SU3_structure_tensor) {
            cudaFree(d_SU3_structure_tensor);
            d_SU3_structure_tensor = nullptr;
        }
    }

    // Initialize SU2 structure tensor on GPU
    bool initSU2(const std::vector<double>& host_SU2_tensor, size_t N_SU2, size_t N_ATOMS_SU2, 
                size_t dim1, size_t dim2, size_t dim3) {
        SU2_tensor_size = N_ATOMS_SU2 * dim1 * dim2 * dim3 * N_SU2 * N_SU2 * N_SU2 * sizeof(double);
        
        // Allocate GPU memory
        if (cudaMalloc(&d_SU2_structure_tensor, SU2_tensor_size) != cudaSuccess) {
            return false;
        }

        // Copy to GPU
        if (cudaMemcpy(d_SU2_structure_tensor, host_SU2_tensor.data(), SU2_tensor_size, 
                     cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_SU2_structure_tensor);
            d_SU2_structure_tensor = nullptr;
            return false;
        }
        
        return true;
    }

    // Initialize SU3 structure tensor on GPU
    bool initSU3(const std::vector<double>& host_SU3_tensor, size_t N_SU3, size_t N_ATOMS_SU3, 
                size_t dim1, size_t dim2, size_t dim3) {
        SU3_tensor_size = N_ATOMS_SU3 * dim1 * dim2 * dim3 * N_SU3 * N_SU3 * N_SU3 * sizeof(double);
        
        // Allocate GPU memory
        if (cudaMalloc(&d_SU3_structure_tensor, SU3_tensor_size) != cudaSuccess) {
            return false;
        }

        // Copy to GPU
        if (cudaMemcpy(d_SU3_structure_tensor, host_SU3_tensor.data(), SU3_tensor_size, 
                     cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_SU3_structure_tensor);
            d_SU3_structure_tensor = nullptr;
            return false;
        }
        
        return true;
    }

    // Wrapper function for SU2 structure tensor contractions
    template<typename T, size_t N, size_t M>
    bool contractSU2(const std::vector<T>& local_fields_flat,
                   const std::vector<T>& spins_flat,
                   std::vector<T>& result_flat) {
        if (d_SU2_structure_tensor == nullptr) {
            return false;  // not initialized
        }

        T *d_tensor2d_1, *d_tensor2d_2, *d_result;
        const size_t tensor2d_size = N * M * sizeof(T);
        
        // Allocate GPU memory for fields, spins, and result
        if (cudaMalloc(&d_tensor2d_1, tensor2d_size) != cudaSuccess ||
            cudaMalloc(&d_tensor2d_2, tensor2d_size) != cudaSuccess ||
            cudaMalloc(&d_result, tensor2d_size) != cudaSuccess) {
            // Handle allocation failure
            if (d_tensor2d_1) cudaFree(d_tensor2d_1);
            if (d_tensor2d_2) cudaFree(d_tensor2d_2);
            if (d_result) cudaFree(d_result);
            return false;
        }
        
        // Copy fields and spins to GPU
        if (cudaMemcpy(d_tensor2d_1, local_fields_flat.data(), tensor2d_size, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_tensor2d_2, spins_flat.data(), tensor2d_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            // Handle copy failure
            cudaFree(d_tensor2d_1);
            cudaFree(d_tensor2d_2);
            cudaFree(d_result);
            return false;
        }
        
        // Launch kernel with default stream (0)
        // Use the pre-allocated SU2 structure tensor
        tensorContract<T, N, M>(d_SU2_structure_tensor, d_tensor2d_1, d_tensor2d_2, d_result, 0);
        
        // Copy result back
        if (cudaMemcpy(result_flat.data(), d_result, tensor2d_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
            // Handle copy failure
            cudaFree(d_tensor2d_1);
            cudaFree(d_tensor2d_2);
            cudaFree(d_result);
            return false;
        }
        
        // Cleanup temporary allocations
        cudaFree(d_tensor2d_1);
        cudaFree(d_tensor2d_2);
        cudaFree(d_result);
        
        return true;
    }

    // Wrapper function for SU3 structure tensor contractions
    template<typename T, size_t N, size_t M>
    bool contractSU3(const std::vector<T>& local_fields_flat,
                   const std::vector<T>& spins_flat,
                   std::vector<T>& result_flat) {
        if (d_SU3_structure_tensor == nullptr) {
            return false;  // not initialized
        }

        T *d_tensor2d_1, *d_tensor2d_2, *d_result;
        const size_t tensor2d_size = N * M * sizeof(T);
        
        // Allocate GPU memory for fields, spins, and result
        if (cudaMalloc(&d_tensor2d_1, tensor2d_size) != cudaSuccess ||
            cudaMalloc(&d_tensor2d_2, tensor2d_size) != cudaSuccess ||
            cudaMalloc(&d_result, tensor2d_size) != cudaSuccess) {
            // Handle allocation failure
            if (d_tensor2d_1) cudaFree(d_tensor2d_1);
            if (d_tensor2d_2) cudaFree(d_tensor2d_2);
            if (d_result) cudaFree(d_result);
            return false;
        }
        
        // Copy fields and spins to GPU
        if (cudaMemcpy(d_tensor2d_1, local_fields_flat.data(), tensor2d_size, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_tensor2d_2, spins_flat.data(), tensor2d_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            // Handle copy failure
            cudaFree(d_tensor2d_1);
            cudaFree(d_tensor2d_2);
            cudaFree(d_result);
            return false;
        }
        
        // Launch kernel with default stream (0)
        // Use the pre-allocated SU3 structure tensor
        tensorContract<T, N, M>(d_SU3_structure_tensor, d_tensor2d_1, d_tensor2d_2, d_result, 0);
        
        // Copy result back
        if (cudaMemcpy(result_flat.data(), d_result, tensor2d_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
            // Handle copy failure
            cudaFree(d_tensor2d_1);
            cudaFree(d_tensor2d_2);
            cudaFree(d_result);
            return false;
        }
        
        // Cleanup temporary allocations
        cudaFree(d_tensor2d_1);
        cudaFree(d_tensor2d_2);
        cudaFree(d_result);
        
        return true;
    }


};

// The original CUDA wrapper function - kept for backward compatibility
template<typename T, size_t N, size_t M>
void cuda_tensor_contraction_wrapper(
    const std::vector<T>& structure_tensor_flat,
    const std::vector<T>& local_fields_flat,
    const std::vector<T>& spins_flat,
    std::vector<T>& result_flat
) {
    T *d_tensor4d, *d_tensor2d_1, *d_tensor2d_2, *d_result;
    
    const size_t tensor4d_size = N * M * M * M * sizeof(T);
    const size_t tensor2d_size = N * M * sizeof(T);
    
    // Allocate GPU memory
    cudaMalloc(&d_tensor4d, tensor4d_size);
    cudaMalloc(&d_tensor2d_1, tensor2d_size);
    cudaMalloc(&d_tensor2d_2, tensor2d_size);
    cudaMalloc(&d_result, tensor2d_size);
    
    // Copy to GPU
    cudaMemcpy(d_tensor4d, structure_tensor_flat.data(), tensor4d_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tensor2d_1, local_fields_flat.data(), tensor2d_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tensor2d_2, spins_flat.data(), tensor2d_size, cudaMemcpyHostToDevice);
    
    // Launch kernel with default stream (0)
    tensorContract<T, N, M>(d_tensor4d, d_tensor2d_1, d_tensor2d_2, d_result, 0);
    
    // Copy result back
    cudaMemcpy(result_flat.data(), d_result, tensor2d_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_tensor4d);
    cudaFree(d_tensor2d_1);
    cudaFree(d_tensor2d_2);
    cudaFree(d_result);
}

#endif // CUDA_WRAPPER_OPTIMIZED_H
