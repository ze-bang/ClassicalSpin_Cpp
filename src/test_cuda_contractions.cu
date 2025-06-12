#include <cuda_contractions.cuh>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cmath>

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// CPU reference implementation for verification
template<typename T, size_t N, size_t M>
void tensorContractCPU(
    const T* tensor4d,    // Shape: (N, M, M, M)
    const T* tensor2d_1,  // Shape: (N, M)
    const T* tensor2d_2,  // Shape: (N, M)
    T* result            // Shape: (N, M)
) {
    for (size_t a = 0; a < N; ++a) {
        for (size_t i = 0; i < M; ++i) {
            T sum = 0;
            for (size_t j = 0; j < M; ++j) {
                for (size_t k = 0; k < M; ++k) {
                    size_t idx4d = a * M * M * M + i * M * M + j * M + k;
                    size_t idx2d_1 = a * M + j;
                    size_t idx2d_2 = a * M + k;
                    sum += tensor4d[idx4d] * tensor2d_1[idx2d_1] * tensor2d_2[idx2d_2];
                }
            }
            result[a * M + i] = sum;
        }
    }
}

// Function to generate random test data
template<typename T>
void generateRandomData(std::vector<T>& data, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& val : data) {
        val = static_cast<T>(dist(gen));
    }
}

// Function to compare results with tolerance
template<typename T>
bool compareResults(const std::vector<T>& gpu_result, const std::vector<T>& cpu_result, T tolerance = 1e-6) {
    if (gpu_result.size() != cpu_result.size()) {
        std::cerr << "Size mismatch: GPU " << gpu_result.size() << " vs CPU " << cpu_result.size() << std::endl;
        return false;
    }
    
    T max_diff = 0;
    size_t max_diff_idx = 0;
    
    for (size_t i = 0; i < gpu_result.size(); ++i) {
        T diff = std::abs(gpu_result[i] - cpu_result[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        if (diff > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU " << gpu_result[i] 
                      << " vs CPU " << cpu_result[i] << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    
    std::cout << "Results match! Max difference: " << max_diff << " at index " << max_diff_idx << std::endl;
    return true;
}

// Test function template
template<typename T, size_t N, size_t M>
bool testTensorContraction(int num_iterations = 10) {
    std::cout << "\n=== Testing Tensor Contraction: N=" << N << ", M=" << M << " ===" << std::endl;
    
    // Calculate sizes
    const size_t tensor4d_size = N * M * M * M;
    const size_t tensor2d_size = N * M;
    const size_t result_size = N * M;
    
    std::cout << "Tensor4D size: " << tensor4d_size << " elements (" 
              << (tensor4d_size * sizeof(T)) / (1024*1024) << " MB)" << std::endl;
    std::cout << "Tensor2D size: " << tensor2d_size << " elements each" << std::endl;
    
    // Initialize random number generator
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    
    // Allocate host memory
    std::vector<T> h_tensor4d(tensor4d_size);
    std::vector<T> h_tensor2d_1(tensor2d_size);
    std::vector<T> h_tensor2d_2(tensor2d_size);
    std::vector<T> h_result_gpu(result_size);
    std::vector<T> h_result_cpu(result_size);
    
    // Generate random test data
    generateRandomData(h_tensor4d, gen);
    generateRandomData(h_tensor2d_1, gen);
    generateRandomData(h_tensor2d_2, gen);
    
    // Allocate device memory
    T *d_tensor4d, *d_tensor2d_1, *d_tensor2d_2, *d_result;
    CUDA_CHECK(cudaMalloc(&d_tensor4d, tensor4d_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tensor2d_1, tensor2d_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tensor2d_2, tensor2d_size * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_result, result_size * sizeof(T)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_tensor4d, h_tensor4d.data(), tensor4d_size * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tensor2d_1, h_tensor2d_1.data(), tensor2d_size * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tensor2d_2, h_tensor2d_2.data(), tensor2d_size * sizeof(T), cudaMemcpyHostToDevice));
    
    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Warm-up run
    tensorContract<T, N, M>(d_tensor4d, d_tensor2d_1, d_tensor2d_2, d_result, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Performance measurement
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        tensorContract<T, N, M>(d_tensor4d, d_tensor2d_1, d_tensor2d_2, d_result, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (1000.0 * num_iterations);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_result_gpu.data(), d_result, result_size * sizeof(T), cudaMemcpyDeviceToHost));
    
    // Compute CPU reference result
    std::cout << "Computing CPU reference..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    tensorContractCPU<T, N, M>(h_tensor4d.data(), h_tensor2d_1.data(), h_tensor2d_2.data(), h_result_cpu.data());
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    double cpu_time_ms = cpu_duration.count() / 1000.0;
    
    // Verify correctness
    bool correct = compareResults(h_result_gpu, h_result_cpu, static_cast<T>(1e-5));
    
    // Performance results
    std::cout << "Performance Results:" << std::endl;
    std::cout << "  GPU average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  CPU time: " << cpu_time_ms << " ms" << std::endl;
    std::cout << "  Speedup: " << (cpu_time_ms / avg_time_ms) << "x" << std::endl;
    
    // Calculate theoretical operations and bandwidth
    size_t ops = N * M * M * M * 3;  // 3 operations per inner loop iteration
    double gflops = (ops * num_iterations) / (duration.count() * 1e3);
    std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
    
    size_t bytes_read = (tensor4d_size + 2 * tensor2d_size) * sizeof(T);
    size_t bytes_write = result_size * sizeof(T);
    double bandwidth = ((bytes_read + bytes_write) * num_iterations) / (duration.count() * 1e-3) / (1024*1024*1024);
    std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_tensor4d));
    CUDA_CHECK(cudaFree(d_tensor2d_1));
    CUDA_CHECK(cudaFree(d_tensor2d_2));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return correct;
}

int main() {
    std::cout << "CUDA Tensor Contraction Test Suite" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Print GPU information
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << std::endl;
    
    bool all_passed = true;
    
    // Test the instantiated template cases
    if (!testTensorContraction<double, 2048, 3>(5)) {
        std::cout << "❌ Test case (N=2048, M=3) FAILED!" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Test case (N=2048, M=3) PASSED!" << std::endl;
    }
    
    if (!testTensorContraction<double, 2048, 8>(5)) {
        std::cout << "❌ Test case (N=2048, M=8) FAILED!" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Test case (N=2048, M=8) PASSED!" << std::endl;
    }
    
    
    std::cout << "\n===================================" << std::endl;
    if (all_passed) {
        std::cout << "🎉 ALL TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "💥 SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}