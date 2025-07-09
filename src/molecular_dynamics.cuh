#ifndef MIXED_LATTICE_CUDA_CUH
#define MIXED_LATTICE_CUDA_CUH

#include "mixed_lattice.h"
#include "unitcell.h"
#include "simple_linear_alg.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <random>
#include <chrono>
#include <math.h>
#include <tuple>
#include <cuda_contractions.cuh>

// CUDA device array structures
template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
struct mixed_lattice_spin_cuda {
    double* spins_SU2;  // Flattened array: [lattice_size_SU2 * N_SU2]
    double* spins_SU3;  // Flattened array: [lattice_size_SU3 * N_SU3]
    
    __host__ __device__
    mixed_lattice_spin_cuda() : spins_SU2(nullptr), spins_SU3(nullptr) {}
    
    __host__
    void allocate() {
        cudaMalloc(&spins_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&spins_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
        cudaMemset(spins_SU2, 0, lattice_size_SU2 * N_SU2 * sizeof(double));
        cudaMemset(spins_SU3, 0, lattice_size_SU3 * N_SU3 * sizeof(double));
    }
    
    __host__
    void deallocate() {
        if (spins_SU2) cudaFree(spins_SU2);
        if (spins_SU3) cudaFree(spins_SU3);
        spins_SU2 = nullptr;
        spins_SU3 = nullptr;
    }
    
    __device__
    double& spin_SU2(size_t site, size_t component) {
        return spins_SU2[site * N_SU2 + component];
    }
    
    __device__
    double& spin_SU3(size_t site, size_t component) {
        return spins_SU3[site * N_SU3 + component];
    }
    
    __device__
    const double& spin_SU2(size_t site, size_t component) const {
        return spins_SU2[site * N_SU2 + component];
    }
    
    __device__
    const double& spin_SU3(size_t site, size_t component) const {
        return spins_SU3[site * N_SU3 + component];
    }
};

template<size_t lattice_size_SU2, size_t lattice_size_SU3>
struct mixed_lattice_pos_cuda {
    double* pos_SU2;  // Flattened array: [lattice_size_SU2 * 3]
    double* pos_SU3;  // Flattened array: [lattice_size_SU3 * 3]
    
    __host__ __device__
    mixed_lattice_pos_cuda() : pos_SU2(nullptr), pos_SU3(nullptr) {}
    
    __host__
    void allocate() {
        cudaMalloc(&pos_SU2, lattice_size_SU2 * 3 * sizeof(double));
        cudaMalloc(&pos_SU3, lattice_size_SU3 * 3 * sizeof(double));
        cudaMemset(pos_SU2, 0, lattice_size_SU2 * 3 * sizeof(double));
        cudaMemset(pos_SU3, 0, lattice_size_SU3 * 3 * sizeof(double));
    }
    
    __host__
    void deallocate() {
        if (pos_SU2) cudaFree(pos_SU2);
        if (pos_SU3) cudaFree(pos_SU3);
        pos_SU2 = nullptr;
        pos_SU3 = nullptr;
    }
    
    __device__
    double& pos_SU2_coord(size_t site, size_t coord) {
        return pos_SU2[site * 3 + coord];
    }
    
    __device__
    double& pos_SU3_coord(size_t site, size_t coord) {
        return pos_SU3[site * 3 + coord];
    }
};

// Device helper function declarations
__device__ double dot_device(const double* a, const double* b, size_t n);
__device__ double contract_device(const double* spin1, const double* matrix, const double* spin2, size_t n);
__device__ double contract_trilinear_device(const double* tensor, const double* spin1, const double* spin2, const double* spin3, size_t n);
__device__ void multiply_matrix_vector_device(double* result, const double* matrix, const double* vector, size_t n);
__device__ void contract_trilinear_field_device(double* result, const double* tensor, const double* spin1, const double* spin2, size_t n);
__device__ void cross_product_SU2_device(double* result, const double* a, const double* b);
__device__ void cross_product_SU3_device(double* result, const double* a, const double* b);

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
class mixed_lattice_cuda;

// CUDA version of mixed_lattice that inherits from the CPU version
template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
class mixed_lattice_cuda : public mixed_lattice<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>
{
public:
    static constexpr size_t lattice_size_SU2 = dim1 * dim2 * dim * N_ATOMS_SU2;
    static constexpr size_t lattice_size_SU3 = dim1 * dim2 * dim * N_ATOMS_SU3;
    
    // Device data structures
    mixed_lattice_spin_cuda<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> d_spins;
    mixed_lattice_pos_cuda<lattice_size_SU2, lattice_size_SU3> d_site_pos;
    
    // Device arrays for lookup tables
    double* d_field_SU2;                    // [lattice_size_SU2 * N_SU2]
    double* d_field_SU3;                    // [lattice_size_SU3 * N_SU3]
    double* d_field_drive_1_SU2;            // [N_ATOMS_SU2 * N_SU2]
    double* d_field_drive_2_SU2;            // [N_ATOMS_SU2 * N_SU2]
    double* d_field_drive_1_SU3;            // [N_ATOMS_SU3 * N_SU3]
    double* d_field_drive_2_SU3;            // [N_ATOMS_SU3 * N_SU3]
    
    double* d_onsite_interaction_SU2;       // [lattice_size_SU2 * N_SU2 * N_SU2]
    double* d_onsite_interaction_SU3;       // [lattice_size_SU3 * N_SU3 * N_SU3]
    
    // Device arrays for interaction parameters (simplified flat arrays)
    double* d_bilinear_interaction_SU2;     // [lattice_size_SU2 * max_neighbors * N_SU2 * N_SU2]
    double* d_bilinear_interaction_SU3;     // [lattice_size_SU3 * max_neighbors * N_SU3 * N_SU3]
    size_t* d_bilinear_partners_SU2;        // [lattice_size_SU2 * max_neighbors]
    size_t* d_bilinear_partners_SU3;        // [lattice_size_SU3 * max_neighbors]
    
    double* d_trilinear_interaction_SU2;    // [lattice_size_SU2 * max_tri_neighbors * N_SU2^3]
    double* d_trilinear_interaction_SU3;    // [lattice_size_SU3 * max_tri_neighbors * N_SU3^3]
    size_t* d_trilinear_partners_SU2;       // [lattice_size_SU2 * max_tri_neighbors * 2]
    size_t* d_trilinear_partners_SU3;       // [lattice_size_SU3 * max_tri_neighbors * 2]
    
    // Mixed interaction arrays
    double* d_mixed_bilinear_interaction_SU2;   // [lattice_size_SU2 * max_mixed_neighbors * N_SU2 * N_SU3]
    size_t* d_mixed_bilinear_partners_SU2;      // [lattice_size_SU2 * max_mixed_neighbors * 2]
    size_t* d_mixed_bilinear_partners_SU3;      // [lattice_size_SU3 * max_mixed_neighbors * 2]
    
    double* d_mixed_trilinear_interaction_SU2;  // [lattice_size_SU2 * max_mixed_tri * N_SU2^2 * N_SU3]
    double* d_mixed_trilinear_interaction_SU3;  // [lattice_size_SU3 * max_mixed_tri * N_SU2^2 * N_SU3]
    size_t* d_mixed_trilinear_partners_SU2;     // [lattice_size_SU2 * max_mixed_tri * 2]
    size_t* d_mixed_trilinear_partners_SU3;     // [lattice_size_SU3 * max_mixed_tri * 2]
    
    // Device constants copied from host
    double d_spin_length_SU2, d_spin_length_SU3;
    double d_field_drive_freq_SU2, d_field_drive_amp_SU2, d_field_drive_width_SU2;
    double d_t_B_1_SU2, d_t_B_2_SU2;
    double d_field_drive_freq_SU3, d_field_drive_amp_SU3, d_field_drive_width_SU3;
    double d_t_B_1_SU3, d_t_B_2_SU3;
    
    size_t d_num_bi_SU2, d_num_tri_SU2, d_num_bi_SU3, d_num_tri_SU3, d_num_tri_SU2_SU3;
    
    // Maximum neighbor counts (determined from host data)
    size_t max_bilinear_neighbors_SU2;
    size_t max_trilinear_neighbors_SU2;
    size_t max_mixed_bilinear_neighbors_SU2;
    size_t max_mixed_trilinear_neighbors_SU2;

    size_t max_bilinear_neighbors_SU3;
    size_t max_trilinear_neighbors_SU3;
    size_t max_mixed_bilinear_neighbors_SU3;
    size_t max_mixed_trilinear_neighbors_SU3;

    // Structure tensors for SU2 and SU3
    double *d_SU2_structure_tensor;
    double *d_SU3_structure_tensor;
    size_t SU2_tensor_size;
    size_t SU3_tensor_size;

    // Random state for CUDA kernels
    curandState* d_rng_states;
    
    // Constructor - inherits from base class and sets up CUDA data
    __host__
    mixed_lattice_cuda(mixed_UnitCell<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3>* atoms, 
                       double spin_length_SU2_in, double spin_length_SU3_in)
        : mixed_lattice<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>(atoms, spin_length_SU2_in, spin_length_SU3_in)
    {
        // Initialize device pointers
        d_field_SU2 = nullptr;
        d_field_SU3 = nullptr;
        d_field_drive_1_SU2 = nullptr;
        d_field_drive_2_SU2 = nullptr;
        d_field_drive_1_SU3 = nullptr;
        d_field_drive_2_SU3 = nullptr;
        d_onsite_interaction_SU2 = nullptr;
        d_onsite_interaction_SU3 = nullptr;
        d_bilinear_interaction_SU2 = nullptr;
        d_bilinear_interaction_SU3 = nullptr;
        d_bilinear_partners_SU2 = nullptr;
        d_bilinear_partners_SU3 = nullptr;
        d_trilinear_interaction_SU2 = nullptr;
        d_trilinear_interaction_SU3 = nullptr;
        d_trilinear_partners_SU2 = nullptr;
        d_trilinear_partners_SU3 = nullptr;
        d_mixed_bilinear_interaction_SU2 = nullptr;
        d_mixed_bilinear_partners_SU2 = nullptr;
        d_mixed_bilinear_partners_SU3 = nullptr;
        d_mixed_trilinear_interaction_SU2 = nullptr;
        d_mixed_trilinear_interaction_SU3 = nullptr;
        d_mixed_trilinear_partners_SU2 = nullptr;
        d_mixed_trilinear_partners_SU3 = nullptr;
        d_rng_states = nullptr;
        
        // Copy scalar values
        d_spin_length_SU2 = this->spin_length_SU2;
        d_spin_length_SU3 = this->spin_length_SU3;
        d_field_drive_freq_SU2 = this->field_drive_freq_SU2;
        d_field_drive_amp_SU2 = this->field_drive_amp_SU2;
        d_field_drive_width_SU2 = this->field_drive_width_SU2;
        d_t_B_1_SU2 = this->t_B_1_SU2;
        d_t_B_2_SU2 = this->t_B_2_SU2;
        d_field_drive_freq_SU3 = this->field_drive_freq_SU3;
        d_field_drive_amp_SU3 = this->field_drive_amp_SU3;
        d_field_drive_width_SU3 = this->field_drive_width_SU3;
        d_t_B_1_SU3 = this->t_B_1_SU3;
        d_t_B_2_SU3 = this->t_B_2_SU3;
        d_num_bi_SU2 = this->num_bi_SU2;
        d_num_tri_SU2 = this->num_tri_SU2;
        d_num_bi_SU3 = this->num_bi_SU3;
        d_num_tri_SU3 = this->num_tri_SU3;
        d_num_tri_SU2_SU3 = this->num_tri_SU2_SU3;

        // Print all parameters of the base class
        std::cout << "\n----- Base Class Parameters -----\n";
        
        // Scalar parameters
        std::cout << "lattice_size_SU2: " << this->lattice_size_SU2 << std::endl;
        std::cout << "lattice_size_SU3: " << this->lattice_size_SU3 << std::endl;
        std::cout << "spin_length_SU2: " << this->spin_length_SU2 << std::endl;
        std::cout << "spin_length_SU3: " << this->spin_length_SU3 << std::endl;
        
        // Field drive parameters SU2
        std::cout << "field_drive_freq_SU2: " << this->field_drive_freq_SU2 << std::endl;
        std::cout << "field_drive_amp_SU2: " << this->field_drive_amp_SU2 << std::endl;
        std::cout << "field_drive_width_SU2: " << this->field_drive_width_SU2 << std::endl;
        std::cout << "t_B_1_SU2: " << this->t_B_1_SU2 << std::endl;
        std::cout << "t_B_2_SU2: " << this->t_B_2_SU2 << std::endl;
        
        // Field drive parameters SU3
        std::cout << "field_drive_freq_SU3: " << this->field_drive_freq_SU3 << std::endl;
        std::cout << "field_drive_amp_SU3: " << this->field_drive_amp_SU3 << std::endl;
        std::cout << "field_drive_width_SU3: " << this->field_drive_width_SU3 << std::endl;
        std::cout << "t_B_1_SU3: " << this->t_B_1_SU3 << std::endl;
        std::cout << "t_B_2_SU3: " << this->t_B_2_SU3 << std::endl;
        
        // Interaction counts
        std::cout << "num_bi_SU2: " << this->num_bi_SU2 << std::endl;
        std::cout << "num_tri_SU2: " << this->num_tri_SU2 << std::endl;
        std::cout << "num_bi_SU3: " << this->num_bi_SU3 << std::endl;
        std::cout << "num_tri_SU3: " << this->num_tri_SU3 << std::endl;
        std::cout << "num_tri_SU2_SU3: " << this->num_tri_SU2_SU3 << std::endl;
        
        // Unit cell info
        std::cout << "Unit cell atoms SU2: " << N_ATOMS_SU2 << std::endl;
        std::cout << "Unit cell atoms SU3: " << N_ATOMS_SU3 << std::endl;
        std::cout << "Lattice dimensions: " << dim1 << "x" << dim2 << "x" << dim << std::endl;
        
        std::cout << "----- End Base Class Parameters -----\n\n";
        
        // Calculate maximum neighbor counts
        calculate_max_neighbors();

        // Print the maximum neighbor counts
        std::cout << "\n----- Maximum Neighbor Counts -----\n";
        std::cout << "max_bilinear_neighbors_SU2: " << max_bilinear_neighbors_SU2 << std::endl;
        std::cout << "max_trilinear_neighbors_SU2: " << max_trilinear_neighbors_SU2 << std::endl;
        std::cout << "max_mixed_bilinear_neighbors_SU2: " << max_mixed_bilinear_neighbors_SU2 << std::endl;
        std::cout << "max_mixed_trilinear_neighbors_SU2: " << max_mixed_trilinear_neighbors_SU2 << std::endl;
        std::cout << "max_bilinear_neighbors_SU3: " << max_bilinear_neighbors_SU3 << std::endl;
        std::cout << "max_trilinear_neighbors_SU3: " << max_trilinear_neighbors_SU3 << std::endl;
        std::cout << "max_mixed_bilinear_neighbors_SU3: " << max_mixed_bilinear_neighbors_SU3 << std::endl;
        std::cout << "max_mixed_trilinear_neighbors_SU3: " << max_mixed_trilinear_neighbors_SU3 << std::endl;
        std::cout << "----- End Maximum Neighbor Counts -----\n\n";
        
        // Allocate and initialize CUDA memory
        allocate_cuda_memory();
        copy_data_to_device();
        
        std::cout << "CUDA mixed lattice initialized with " << lattice_size_SU2 << " SU2 sites and " 
                  << lattice_size_SU3 << " SU3 sites" << std::endl;
    }
    
    // Destructor
    __host__
    ~mixed_lattice_cuda() {
        cleanup_cuda_memory();
    }
    
private:
    // Calculate maximum number of neighbors for each interaction type
    __host__
    void calculate_max_neighbors() {
        max_bilinear_neighbors_SU2 = 0;
        max_trilinear_neighbors_SU2 = 0;
        max_mixed_bilinear_neighbors_SU2 = 0;
        max_mixed_trilinear_neighbors_SU2 = 0;

        max_bilinear_neighbors_SU3 = 0;
        max_trilinear_neighbors_SU3 = 0;
        max_mixed_bilinear_neighbors_SU3 = 0;
        max_mixed_trilinear_neighbors_SU3 = 0;

        // Calculate max bilinear neighbors for SU2
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            max_bilinear_neighbors_SU2 = std::max(max_bilinear_neighbors_SU2, this->bilinear_partners_SU2[i].size());
            max_trilinear_neighbors_SU2 = std::max(max_trilinear_neighbors_SU2, this->trilinear_partners_SU2[i].size());
            max_mixed_bilinear_neighbors_SU2 = std::max(max_mixed_bilinear_neighbors_SU2, this->mixed_bilinear_partners_SU2[i].size());
            max_mixed_trilinear_neighbors_SU2 = std::max(max_mixed_trilinear_neighbors_SU2, this->mixed_trilinear_partners_SU2[i].size());
        }
        
        // Calculate max bilinear neighbors for SU3
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            max_bilinear_neighbors_SU3 = std::max(max_bilinear_neighbors_SU3, this->bilinear_partners_SU3[i].size());
            max_trilinear_neighbors_SU3 = std::max(max_trilinear_neighbors_SU3, this->trilinear_partners_SU3[i].size());
            max_mixed_bilinear_neighbors_SU3 = std::max(max_mixed_bilinear_neighbors_SU3, this->mixed_bilinear_partners_SU3[i].size());
            max_mixed_trilinear_neighbors_SU3 = std::max(max_mixed_trilinear_neighbors_SU3, this->mixed_trilinear_partners_SU3[i].size());
        }
        
    }
    
    // Allocate CUDA memory for all arrays
    __host__
    void allocate_cuda_memory() {
        // Allocate spin and position arrays
        d_spins.allocate();
        d_site_pos.allocate();
        
        // Allocate field arrays
        cudaMalloc(&d_field_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&d_field_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
        cudaMalloc(&d_field_drive_1_SU2, N_ATOMS_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&d_field_drive_2_SU2, N_ATOMS_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&d_field_drive_1_SU3, N_ATOMS_SU3 * N_SU3 * sizeof(double));
        cudaMalloc(&d_field_drive_2_SU3, N_ATOMS_SU3 * N_SU3 * sizeof(double));
        
        // Allocate onsite interaction arrays
        cudaMalloc(&d_onsite_interaction_SU2, lattice_size_SU2 * N_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&d_onsite_interaction_SU3, lattice_size_SU3 * N_SU3 * N_SU3 * sizeof(double));
        // Allocate bilinear interaction arrays
        if (max_bilinear_neighbors_SU2 > 0) {
            cudaMalloc(&d_bilinear_interaction_SU2, lattice_size_SU2 * max_bilinear_neighbors_SU2 * N_SU2 * N_SU2 * sizeof(double));
            cudaMalloc(&d_bilinear_partners_SU2, lattice_size_SU2 * max_bilinear_neighbors_SU2 * sizeof(size_t));
        }
        if (max_bilinear_neighbors_SU3 > 0) {
            cudaMalloc(&d_bilinear_interaction_SU3, lattice_size_SU3 * max_bilinear_neighbors_SU3 * N_SU3 * N_SU3 * sizeof(double));
            cudaMalloc(&d_bilinear_partners_SU3, lattice_size_SU3 * max_bilinear_neighbors_SU3 * sizeof(size_t));
        }
        
        // Allocate trilinear interaction arrays
        if (max_trilinear_neighbors_SU2 > 0) {
            cudaMalloc(&d_trilinear_interaction_SU2, lattice_size_SU2 * max_trilinear_neighbors_SU2 * N_SU2 * N_SU2 * N_SU2 * sizeof(double));
            cudaMalloc(&d_trilinear_partners_SU2, lattice_size_SU2 * max_trilinear_neighbors_SU2 * 2 * sizeof(size_t));
        }

        if (max_trilinear_neighbors_SU3 > 0) {
            cudaMalloc(&d_trilinear_interaction_SU3, lattice_size_SU3 * max_trilinear_neighbors_SU3 * N_SU3 * N_SU3 * N_SU3 * sizeof(double));
            cudaMalloc(&d_trilinear_partners_SU3, lattice_size_SU3 * max_trilinear_neighbors_SU3 * 2 * sizeof(size_t));
        }

        // Allocate mixed interaction arrays
        if (max_mixed_bilinear_neighbors_SU2 > 0) {
            cudaMalloc(&d_mixed_bilinear_interaction_SU2, lattice_size_SU2 * max_mixed_bilinear_neighbors_SU2 * N_SU2 * N_SU3 * sizeof(double));
            cudaMalloc(&d_mixed_bilinear_partners_SU2, lattice_size_SU2 * max_mixed_bilinear_neighbors_SU2 * 2 * sizeof(size_t));
            cudaMalloc(&d_mixed_bilinear_partners_SU3, lattice_size_SU3 * max_mixed_bilinear_neighbors_SU3 * 2 * sizeof(size_t));
        }
        
        if (max_mixed_trilinear_neighbors_SU2 > 0) {
            cudaMalloc(&d_mixed_trilinear_interaction_SU2, lattice_size_SU2 * max_mixed_trilinear_neighbors_SU2 * N_SU2 * N_SU2 * N_SU3 * sizeof(double));
            cudaMalloc(&d_mixed_trilinear_interaction_SU3, lattice_size_SU3 * max_mixed_trilinear_neighbors_SU3 * N_SU2 * N_SU2 * N_SU3 * sizeof(double));
            cudaMalloc(&d_mixed_trilinear_partners_SU2, lattice_size_SU2 * max_mixed_trilinear_neighbors_SU2 * 2 * sizeof(size_t));
            cudaMalloc(&d_mixed_trilinear_partners_SU3, lattice_size_SU3 * max_mixed_trilinear_neighbors_SU3 * 2 * sizeof(size_t));
        }
        
        // Allocate structure tensors
        SU2_tensor_size = N_ATOMS_SU2 * dim1 * dim2 * dim * N_SU2 * N_SU2 * N_SU2 * sizeof(double);
        SU3_tensor_size = N_ATOMS_SU3 * dim1 * dim2 * dim * N_SU3 * N_SU3 * N_SU3 * sizeof(double);
        cudaMalloc(&d_SU2_structure_tensor, SU2_tensor_size);
        cudaMalloc(&d_SU3_structure_tensor, SU3_tensor_size);

        // Allocate random states
        cudaMalloc(&d_rng_states, (lattice_size_SU2 + lattice_size_SU3) * sizeof(curandState));
    }
    
    // Copy data from host arrays to device arrays
    __host__
    void copy_data_to_device() {
        // Copy spins
        std::vector<double> temp_spins_SU2(lattice_size_SU2 * N_SU2);
        std::vector<double> temp_spins_SU3(lattice_size_SU3 * N_SU3);
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t j = 0; j < N_SU2; ++j) {
                temp_spins_SU2[i * N_SU2 + j] = this->spins.spins_SU2[i][j];
            }
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t j = 0; j < N_SU3; ++j) {
                temp_spins_SU3[i * N_SU3 + j] = this->spins.spins_SU3[i][j];
            }
        }
        
        cudaMemcpy(d_spins.spins_SU2, temp_spins_SU2.data(), lattice_size_SU2 * N_SU2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_spins.spins_SU3, temp_spins_SU3.data(), lattice_size_SU3 * N_SU3 * sizeof(double), cudaMemcpyHostToDevice);
        
        // Copy positions
        std::vector<double> temp_pos_SU2(lattice_size_SU2 * 3);
        std::vector<double> temp_pos_SU3(lattice_size_SU3 * 3);
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                temp_pos_SU2[i * 3 + j] = this->site_pos.pos_SU2[i][j];
            }
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                temp_pos_SU3[i * 3 + j] = this->site_pos.pos_SU3[i][j];
            }
        }
        
        cudaMemcpy(d_site_pos.pos_SU2, temp_pos_SU2.data(), lattice_size_SU2 * 3 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_site_pos.pos_SU3, temp_pos_SU3.data(), lattice_size_SU3 * 3 * sizeof(double), cudaMemcpyHostToDevice);
        
        // Copy fields
        std::vector<double> temp_field_SU2(lattice_size_SU2 * N_SU2);
        std::vector<double> temp_field_SU3(lattice_size_SU3 * N_SU3);
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t j = 0; j < N_SU2; ++j) {
                temp_field_SU2[i * N_SU2 + j] = this->field_SU2[i][j];
            }
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t j = 0; j < N_SU3; ++j) {
                temp_field_SU3[i * N_SU3 + j] = this->field_SU3[i][j];
            }
        }
        
        cudaMemcpy(d_field_SU2, temp_field_SU2.data(), lattice_size_SU2 * N_SU2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_field_SU3, temp_field_SU3.data(), lattice_size_SU3 * N_SU3 * sizeof(double), cudaMemcpyHostToDevice);
        
        // Copy drive fields
        std::vector<double> temp_drive1_SU2(N_ATOMS_SU2 * N_SU2);
        std::vector<double> temp_drive2_SU2(N_ATOMS_SU2 * N_SU2);
        std::vector<double> temp_drive1_SU3(N_ATOMS_SU3 * N_SU3);
        std::vector<double> temp_drive2_SU3(N_ATOMS_SU3 * N_SU3);
        
        for (size_t i = 0; i < N_ATOMS_SU2; ++i) {
            for (size_t j = 0; j < N_SU2; ++j) {
                temp_drive1_SU2[i * N_SU2 + j] = this->field_drive_1_SU2[i][j];
                temp_drive2_SU2[i * N_SU2 + j] = this->field_drive_2_SU2[i][j];
            }
        }
        for (size_t i = 0; i < N_ATOMS_SU3; ++i) {
            for (size_t j = 0; j < N_SU3; ++j) {
                temp_drive1_SU3[i * N_SU3 + j] = this->field_drive_1_SU3[i][j];
                temp_drive2_SU3[i * N_SU3 + j] = this->field_drive_2_SU3[i][j];
            }
        }
        
        cudaMemcpy(d_field_drive_1_SU2, temp_drive1_SU2.data(), N_ATOMS_SU2 * N_SU2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_field_drive_2_SU2, temp_drive2_SU2.data(), N_ATOMS_SU2 * N_SU2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_field_drive_1_SU3, temp_drive1_SU3.data(), N_ATOMS_SU3 * N_SU3 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_field_drive_2_SU3, temp_drive2_SU3.data(), N_ATOMS_SU3 * N_SU3 * sizeof(double), cudaMemcpyHostToDevice);
        
        // Copy onsite interactions
        std::vector<double> temp_onsite_SU2(lattice_size_SU2 * N_SU2 * N_SU2);
        std::vector<double> temp_onsite_SU3(lattice_size_SU3 * N_SU3 * N_SU3);
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t j = 0; j < N_SU2 * N_SU2; ++j) {
                temp_onsite_SU2[i * N_SU2 * N_SU2 + j] = this->onsite_interaction_SU2[i][j];
            }
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t j = 0; j < N_SU3 * N_SU3; ++j) {
                temp_onsite_SU3[i * N_SU3 * N_SU3 + j] = this->onsite_interaction_SU3[i][j];
            }
        }
        
        cudaMemcpy(d_onsite_interaction_SU2, temp_onsite_SU2.data(), lattice_size_SU2 * N_SU2 * N_SU2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_onsite_interaction_SU3, temp_onsite_SU3.data(), lattice_size_SU3 * N_SU3 * N_SU3 * sizeof(double), cudaMemcpyHostToDevice);

        // Copy structure tensors
        cudaMemcpy(d_SU2_structure_tensor, this->SU2_structure_tensor.data(), SU2_tensor_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_SU3_structure_tensor, this->SU3_structure_tensor.data(), SU3_tensor_size, cudaMemcpyHostToDevice);

        // Copy interaction arrays
        copy_interaction_arrays_to_device();
    }
    
    // Helper function to copy complex interaction arrays
    __host__
    void copy_interaction_arrays_to_device() {
        if (max_bilinear_neighbors_SU2 > 0) {
            std::vector<double> temp_bi_SU2(lattice_size_SU2 * max_bilinear_neighbors_SU2 * N_SU2 * N_SU2, 0.0);
            std::vector<size_t> temp_bi_partners_SU2(lattice_size_SU2 * max_bilinear_neighbors_SU2, 0);
            for (size_t site = 0; site < lattice_size_SU2; ++site) {
                const auto& interactions = this->bilinear_interaction_SU2[site];
                const auto& partners = this->bilinear_partners_SU2[site];
                for (size_t i = 0; i < max_bilinear_neighbors_SU2; ++i) {
                    size_t base_idx = site * max_bilinear_neighbors_SU2 * N_SU2 * N_SU2 + i * N_SU2 * N_SU2;
                    for (size_t j = 0; j < N_SU2 * N_SU2; ++j) {
                        temp_bi_SU2[base_idx + j] = interactions[i][j];
                    }
                    temp_bi_partners_SU2[site * max_bilinear_neighbors_SU2 + i] = partners[i];
                }
            }
            cudaMemcpy(d_bilinear_interaction_SU2, temp_bi_SU2.data(), temp_bi_SU2.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bilinear_partners_SU2, temp_bi_partners_SU2.data(), temp_bi_partners_SU2.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        }

        if (max_bilinear_neighbors_SU3 > 0) {
            std::vector<double> temp_bi_SU3(lattice_size_SU3 * max_bilinear_neighbors_SU3 * N_SU3 * N_SU3, 0.0);
            std::vector<size_t> temp_bi_partners_SU3(lattice_size_SU3 * max_bilinear_neighbors_SU3, 0);         
            for (size_t site = 0; site < lattice_size_SU3; ++site) {
                const auto& interactions = this->bilinear_interaction_SU3[site];
                const auto& partners = this->bilinear_partners_SU3[site];

                for (size_t i = 0; i < max_bilinear_neighbors_SU3; ++i) {
                    size_t base_idx = site * max_bilinear_neighbors_SU3 * N_SU3 * N_SU3 + i * N_SU3 * N_SU3;
                    for (size_t j = 0; j < N_SU3 * N_SU3; ++j) {
                        temp_bi_SU3[base_idx + j] = interactions[i][j];
                    }
                    temp_bi_partners_SU3[site * max_bilinear_neighbors_SU3 + i] = partners[i];
                }
            }

            cudaMemcpy(d_bilinear_interaction_SU3, temp_bi_SU3.data(), temp_bi_SU3.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bilinear_partners_SU3, temp_bi_partners_SU3.data(), temp_bi_partners_SU3.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        }
        
        // Similar implementations would be needed for mixed interactions
        if (max_trilinear_neighbors_SU2 > 0) {
            // Allocate and copy trilinear interactions
            std::vector<double> temp_tri_SU2(lattice_size_SU2 * max_trilinear_neighbors_SU2 * N_SU2 * N_SU2 * N_SU2, 0.0);
            std::vector<size_t> temp_tri_partners_SU2(lattice_size_SU2 * max_trilinear_neighbors_SU2 * 2, 0);
            
            // Fill trilinear data for SU2
            for (size_t site = 0; site < lattice_size_SU2; ++site) {
                const auto& interactions = this->trilinear_interaction_SU2[site];
                const auto& partners = this->trilinear_partners_SU2[site];

                for (size_t i = 0; i < max_trilinear_neighbors_SU2; ++i) {
                    size_t base_idx = site * max_trilinear_neighbors_SU2 * N_SU2 * N_SU2 * N_SU2 + i * N_SU2 * N_SU2 * N_SU2;
                    for (size_t j = 0; j < N_SU2 * N_SU2 * N_SU2; ++j) {
                        temp_tri_SU2[base_idx + j] = interactions[i][j];
                    }
                    temp_tri_partners_SU2[site * max_trilinear_neighbors_SU2 * 2 + i * 2] = partners[i][0];
                    temp_tri_partners_SU2[site * max_trilinear_neighbors_SU2 * 2 + i * 2 + 1] = partners[i][1];
                }
            }
            cudaMemcpy(d_trilinear_interaction_SU2, temp_tri_SU2.data(), temp_tri_SU2.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_trilinear_partners_SU2, temp_tri_partners_SU2.data(), temp_tri_partners_SU2.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        }

        if (max_trilinear_neighbors_SU3 > 0) {
            // Allocate and copy trilinear interactions
            std::vector<double> temp_tri_SU3(lattice_size_SU3 * max_trilinear_neighbors_SU3 * N_SU3 * N_SU3 * N_SU3, 0.0);
            std::vector<size_t> temp_tri_partners_SU3(lattice_size_SU3 * max_trilinear_neighbors_SU3 * 2, 0);
            
            // Fill trilinear data for SU3
            for (size_t site = 0; site < lattice_size_SU3; ++site) {
                const auto& interactions = this->trilinear_interaction_SU3[site];
                const auto& partners = this->trilinear_partners_SU3[site];

                for (size_t i = 0; i < max_trilinear_neighbors_SU3; ++i) {
                    size_t base_idx = site * max_trilinear_neighbors_SU3 * N_SU3 * N_SU3 * N_SU3 + i * N_SU3 * N_SU3 * N_SU3;
                    for (size_t j = 0; j < N_SU3 * N_SU3 * N_SU3; ++j) {
                        temp_tri_SU3[base_idx + j] = interactions[i][j];
                    }
                    temp_tri_partners_SU3[site * max_trilinear_neighbors_SU3 * 2 + i * 2] = partners[i][0];
                    temp_tri_partners_SU3[site * max_trilinear_neighbors_SU3 * 2 + i * 2 + 1] = partners[i][1];
                }
            }

            cudaMemcpy(d_trilinear_interaction_SU3, temp_tri_SU3.data(), temp_tri_SU3.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_trilinear_partners_SU3, temp_tri_partners_SU3.data(), temp_tri_partners_SU3.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        }   

        if (max_mixed_bilinear_neighbors_SU2 > 0){ 
            std::vector<double> temp_mixed_bi_SU2(lattice_size_SU2 * max_mixed_bilinear_neighbors_SU2 * N_SU2 * N_SU3, 0.0);
            std::vector<size_t> temp_mixed_bi_partners_SU2(lattice_size_SU2 * max_mixed_bilinear_neighbors_SU2, 0);
            std::vector<size_t> temp_mixed_bi_partners_SU3(lattice_size_SU3 * max_mixed_bilinear_neighbors_SU3, 0);
            // Fill mixed bilinear data for SU2
            for (size_t site = 0; site < lattice_size_SU2; ++site) {
                const auto& interactions = this->mixed_bilinear_interaction_SU2[site];
                const auto& partners_SU2 = this->mixed_bilinear_partners_SU2[site];
                const auto& partners_SU3 = this->mixed_bilinear_partners_SU3[site];

                for (size_t i = 0; i < max_mixed_bilinear_neighbors_SU2; ++i) {
                    size_t base_idx = site * max_mixed_bilinear_neighbors_SU2 * N_SU2 * N_SU3 + i * N_SU2 * N_SU3;
                    for (size_t j = 0; j < N_SU2 * N_SU3; ++j) {
                        temp_mixed_bi_SU2[base_idx + j] = interactions[i][j];
                    }
                    temp_mixed_bi_partners_SU2[site * max_mixed_bilinear_neighbors_SU2 + i] = partners_SU2[i];
                    temp_mixed_bi_partners_SU3[site * max_mixed_bilinear_neighbors_SU3 + i] = partners_SU3[i];
                }
            }

            cudaMemcpy(d_mixed_bilinear_interaction_SU2, temp_mixed_bi_SU2.data(), temp_mixed_bi_SU2.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_mixed_bilinear_partners_SU2, temp_mixed_bi_partners_SU2.data(), temp_mixed_bi_partners_SU2.size() * sizeof(size_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_mixed_bilinear_partners_SU3, temp_mixed_bi_partners_SU3.data(), temp_mixed_bi_partners_SU3.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        }

        if (max_mixed_trilinear_neighbors_SU2 > 0){
            std::vector<double> temp_mixed_tri_SU2(lattice_size_SU2 * max_mixed_trilinear_neighbors_SU2 * N_SU2 * N_SU2 * N_SU3, 0.0);
            std::vector<double> temp_mixed_tri_SU3(lattice_size_SU3 * max_mixed_trilinear_neighbors_SU3 * N_SU2 * N_SU2 * N_SU3, 0.0);
            std::vector<size_t> temp_mixed_tri_partners_SU2(lattice_size_SU2 * max_mixed_trilinear_neighbors_SU2 * 2, 0);
            std::vector<size_t> temp_mixed_tri_partners_SU3(lattice_size_SU3 * max_mixed_trilinear_neighbors_SU3 * 2, 0);
            // Fill mixed trilinear data for SU2
            for (size_t site = 0; site < lattice_size_SU2; ++site) {
                const auto& interactions = this->mixed_trilinear_interaction_SU2[site];
                const auto& partners = this->mixed_trilinear_partners_SU2[site];

                for (size_t i = 0; i < max_mixed_trilinear_neighbors_SU2; ++i) {
                    size_t base_idx = site * max_mixed_trilinear_neighbors_SU2 * N_SU2 * N_SU2 * N_SU3 + i * N_SU2 * N_SU2 * N_SU3;
                    for (size_t j = 0; j < N_SU2 * N_SU2 * N_SU3; ++j) {
                        temp_mixed_tri_SU2[base_idx + j] = interactions[i][j];
                    }
                    temp_mixed_tri_partners_SU2[site * max_mixed_trilinear_neighbors_SU2 * 2 + i * 2] = partners[i][0];
                    temp_mixed_tri_partners_SU2[site * max_mixed_trilinear_neighbors_SU2 * 2 + i * 2 + 1] = partners[i][1];
                }
            }

            for (size_t site = 0; site < lattice_size_SU3; ++site) {
                const auto& interactions = this->mixed_trilinear_interaction_SU3[site];
                const auto& partners = this->mixed_trilinear_partners_SU3[site];

                for (size_t i = 0; i < max_mixed_trilinear_neighbors_SU3; ++i) {
                    size_t base_idx = site * max_mixed_trilinear_neighbors_SU3 * N_SU3 * N_SU2 * N_SU2 + i * N_SU3 * N_SU2 * N_SU2;
                    for (size_t j = 0; j < N_SU2 * N_SU2 * N_SU3; ++j) {
                        temp_mixed_tri_SU3[base_idx + j] = interactions[i][j];
                    }
                    temp_mixed_tri_partners_SU3[site * max_mixed_trilinear_neighbors_SU3 * 2 + i * 2] = partners[i][0];
                    temp_mixed_tri_partners_SU3[site * max_mixed_trilinear_neighbors_SU3 * 2 + i * 2 + 1] = partners[i][1];
                }
            }
            cudaMemcpy(d_mixed_trilinear_interaction_SU2, temp_mixed_tri_SU2.data(), temp_mixed_tri_SU2.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_mixed_trilinear_interaction_SU3, temp_mixed_tri_SU3.data(), temp_mixed_tri_SU3.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_mixed_trilinear_partners_SU2, temp_mixed_tri_partners_SU2.data(), temp_mixed_tri_partners_SU2.size() * sizeof(size_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_mixed_trilinear_partners_SU3, temp_mixed_tri_partners_SU3.data(), temp_mixed_tri_partners_SU3.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        }
    
    }
    // Cleanup CUDA memory
    __host__
    void cleanup_cuda_memory() {
        // First safely deallocate the structures that have their own deallocation methods
        d_spins.deallocate();
        d_site_pos.deallocate();
        
        // Free all other device pointers and set them to nullptr to avoid double free
        if (d_field_SU2) {
            cudaFree(d_field_SU2);
            d_field_SU2 = nullptr;
        }
        if (d_field_SU3) {
            cudaFree(d_field_SU3);
            d_field_SU3 = nullptr;
        }
        if (d_field_drive_1_SU2) {
            cudaFree(d_field_drive_1_SU2);
            d_field_drive_1_SU2 = nullptr;
        }
        if (d_field_drive_2_SU2) {
            cudaFree(d_field_drive_2_SU2);
            d_field_drive_2_SU2 = nullptr;
        }
        if (d_field_drive_1_SU3) {
            cudaFree(d_field_drive_1_SU3);
            d_field_drive_1_SU3 = nullptr;
        }
        if (d_field_drive_2_SU3) {
            cudaFree(d_field_drive_2_SU3);
            d_field_drive_2_SU3 = nullptr;
        }
        if (d_onsite_interaction_SU2) {
            cudaFree(d_onsite_interaction_SU2);
            d_onsite_interaction_SU2 = nullptr;
        }
        if (d_onsite_interaction_SU3) {
            cudaFree(d_onsite_interaction_SU3);
            d_onsite_interaction_SU3 = nullptr;
        }
        if (d_bilinear_interaction_SU2) {
            cudaFree(d_bilinear_interaction_SU2);
            d_bilinear_interaction_SU2 = nullptr;
        }
        if (d_bilinear_interaction_SU3) {
            cudaFree(d_bilinear_interaction_SU3);
            d_bilinear_interaction_SU3 = nullptr;
        }
        if (d_bilinear_partners_SU2) {
            cudaFree(d_bilinear_partners_SU2);
            d_bilinear_partners_SU2 = nullptr;
        }
        if (d_bilinear_partners_SU3) {
            cudaFree(d_bilinear_partners_SU3);
            d_bilinear_partners_SU3 = nullptr;
        }
        if (d_trilinear_interaction_SU2) {
            cudaFree(d_trilinear_interaction_SU2);
            d_trilinear_interaction_SU2 = nullptr;
        }
        if (d_trilinear_interaction_SU3) {
            cudaFree(d_trilinear_interaction_SU3);
            d_trilinear_interaction_SU3 = nullptr;
        }
        if (d_trilinear_partners_SU2) {
            cudaFree(d_trilinear_partners_SU2);
            d_trilinear_partners_SU2 = nullptr;
        }
        if (d_trilinear_partners_SU3) {
            cudaFree(d_trilinear_partners_SU3);
            d_trilinear_partners_SU3 = nullptr;
        }
        if (d_mixed_bilinear_interaction_SU2) {
            cudaFree(d_mixed_bilinear_interaction_SU2);
            d_mixed_bilinear_interaction_SU2 = nullptr;
        }
        if (d_mixed_bilinear_partners_SU2) {
            cudaFree(d_mixed_bilinear_partners_SU2);
            d_mixed_bilinear_partners_SU2 = nullptr;
        }
        if (d_mixed_bilinear_partners_SU3) {
            cudaFree(d_mixed_bilinear_partners_SU3);
            d_mixed_bilinear_partners_SU3 = nullptr;
        }
        if (d_mixed_trilinear_interaction_SU2) {
            cudaFree(d_mixed_trilinear_interaction_SU2);
            d_mixed_trilinear_interaction_SU2 = nullptr;
        }
        if (d_mixed_trilinear_interaction_SU3) {
            cudaFree(d_mixed_trilinear_interaction_SU3);
            d_mixed_trilinear_interaction_SU3 = nullptr;
        }
        if (d_mixed_trilinear_partners_SU2) {
            cudaFree(d_mixed_trilinear_partners_SU2);
            d_mixed_trilinear_partners_SU2 = nullptr;
        }
        if (d_mixed_trilinear_partners_SU3) {
            cudaFree(d_mixed_trilinear_partners_SU3);
            d_mixed_trilinear_partners_SU3 = nullptr;
        }
        if (d_rng_states) {
            cudaFree(d_rng_states);
            d_rng_states = nullptr;
        }
        
        // Synchronize device to make sure all deallocations are complete
        cudaDeviceSynchronize();
    }

public:

    // Print information about the device lattice and CUDA implementation
    __host__
    void print_lattice_info_cuda(const string& filename = "lattice_info_GPU.txt") const {
        ofstream outfile(filename);
        if (!outfile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << endl;
            return;
        }

        outfile << "Lattice size SU2: " << lattice_size_SU2 << endl;
        outfile << "Lattice size SU3: " << lattice_size_SU3 << endl;
        outfile << "Number of bilinear interactions SU2: " << d_num_bi_SU2 << endl;
        outfile << "Number of trilinear interactions SU2: " << d_num_tri_SU2 << endl;
        outfile << "Number of bilinear interactions SU3: " << d_num_bi_SU3 << endl;
        outfile << "Number of trilinear interactions SU3: " << d_num_tri_SU3 << endl;
        outfile << "Number of mixed trilinear interactions SU2-SU3: " << d_num_tri_SU2_SU3 << endl;

        // Get data from device for display
        std::vector<double> h_onsite_SU2(lattice_size_SU2 * N_SU2 * N_SU2);
        std::vector<double> h_onsite_SU3(lattice_size_SU3 * N_SU3 * N_SU3);
        
        // Copy onsite interactions from device to host
        cudaMemcpy(h_onsite_SU2.data(), d_onsite_interaction_SU2, h_onsite_SU2.size() * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_onsite_SU3.data(), d_onsite_interaction_SU3, h_onsite_SU3.size() * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Print SU2 interactions
        outfile << "\nSU2 Interactions:" << endl;
        outfile << "-----------------" << endl;

        // Print onsite interactions for first few sites
        outfile << "Onsite interactions (first 3 sites):" << endl;
        for (size_t i = 0; i < min(size_t(3), lattice_size_SU2); ++i) {
            outfile << "Site " << i << ": ";
            for (size_t j = 0; j < N_SU2 * N_SU2; ++j) {
                outfile << h_onsite_SU2[i * N_SU2 * N_SU2 + j] << " ";
            }
            outfile << endl;
        }

        // For bilinear interactions, we need partner indices and interaction matrices
        if (max_bilinear_neighbors_SU2 > 0 && d_bilinear_partners_SU2 != nullptr) {
            std::vector<size_t> h_bi_partners_SU2(lattice_size_SU2 * max_bilinear_neighbors_SU2);
            std::vector<double> h_bi_interaction_SU2(lattice_size_SU2 * max_bilinear_neighbors_SU2 * N_SU2 * N_SU2);
            
            cudaMemcpy(h_bi_partners_SU2.data(), d_bilinear_partners_SU2, h_bi_partners_SU2.size() * sizeof(size_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_bi_interaction_SU2.data(), d_bilinear_interaction_SU2, h_bi_interaction_SU2.size() * sizeof(double), cudaMemcpyDeviceToHost);

            // Print bilinear interactions for first few sites
            outfile << "\nBilinear interactions (first 3 sites):" << endl;
            for (size_t i = 0; i < min(size_t(3), lattice_size_SU2); ++i) {
                outfile << "Site " << i << " has " << max_bilinear_neighbors_SU2 << " bilinear partners:" << endl;
                for (size_t j = 0; j < max_bilinear_neighbors_SU2; ++j) {
                    size_t partner = h_bi_partners_SU2[i * max_bilinear_neighbors_SU2 + j];
                    if (partner < lattice_size_SU2) {  // Valid partner index
                        outfile << "  Partner: " << partner << ", Matrix: ";
                        for (size_t k = 0; k < N_SU2 * N_SU2; ++k) {
                            outfile << h_bi_interaction_SU2[i * max_bilinear_neighbors_SU2 * N_SU2 * N_SU2 + j * N_SU2 * N_SU2 + k] << " ";
                        }
                        outfile << endl;
                    }
                }
            }
        }

        // For trilinear interactions, we need partner pairs and interaction tensors
        if (max_trilinear_neighbors_SU2 > 0 && d_trilinear_partners_SU2 != nullptr) {
            std::vector<size_t> h_tri_partners_SU2(lattice_size_SU2 * max_trilinear_neighbors_SU2 * 2);
            std::vector<double> h_tri_interaction_SU2(lattice_size_SU2 * max_trilinear_neighbors_SU2 * N_SU2 * N_SU2 * N_SU2);
            
            cudaMemcpy(h_tri_partners_SU2.data(), d_trilinear_partners_SU2, h_tri_partners_SU2.size() * sizeof(size_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_tri_interaction_SU2.data(), d_trilinear_interaction_SU2, h_tri_interaction_SU2.size() * sizeof(double), cudaMemcpyDeviceToHost);
            
            // Print trilinear interactions for first few sites
            outfile << "\nTrilinear interactions (first 3 sites):" << endl;
            for (size_t i = 0; i < min(size_t(3), lattice_size_SU2); ++i) {
                outfile << "Site " << i << " has " << max_trilinear_neighbors_SU2 << " trilinear partner pairs:" << endl;
                for (size_t j = 0; j < max_trilinear_neighbors_SU2; ++j) {
                    size_t partner1 = h_tri_partners_SU2[i * max_trilinear_neighbors_SU2 * 2 + j * 2];
                    size_t partner2 = h_tri_partners_SU2[i * max_trilinear_neighbors_SU2 * 2 + j * 2 + 1];
                    if (partner1 < lattice_size_SU2 && partner2 < lattice_size_SU2) {  // Valid partners
                        outfile << "  Partners: (" << partner1 << ", " << partner2 << ")" << endl;
                    }
                }
            }
        }

        // Print SU3 interactions
        outfile << "\nSU3 Interactions:" << endl;
        outfile << "-----------------" << endl;

        // Print onsite interactions for first few sites
        outfile << "Onsite interactions (first 3 sites):" << endl;
        for (size_t i = 0; i < min(size_t(3), lattice_size_SU3); ++i) {
            outfile << "Site " << i << ": ";
            for (size_t j = 0; j < N_SU3 * N_SU3; ++j) {
                outfile << h_onsite_SU3[i * N_SU3 * N_SU3 + j] << " ";
            }
            outfile << endl;
        }

        // Similar blocks for SU3 bilinear and trilinear interactions
        if (max_bilinear_neighbors_SU3 > 0 && d_bilinear_partners_SU3 != nullptr) {
            std::vector<size_t> h_bi_partners_SU3(lattice_size_SU3 * max_bilinear_neighbors_SU3);
            std::vector<double> h_bi_interaction_SU3(lattice_size_SU3 * max_bilinear_neighbors_SU3 * N_SU3 * N_SU3);
            
            cudaMemcpy(h_bi_partners_SU3.data(), d_bilinear_partners_SU3, h_bi_partners_SU3.size() * sizeof(size_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_bi_interaction_SU3.data(), d_bilinear_interaction_SU3, h_bi_interaction_SU3.size() * sizeof(double), cudaMemcpyDeviceToHost);
            
            outfile << "\nBilinear interactions (first 3 sites):" << endl;
            for (size_t i = 0; i < min(size_t(3), lattice_size_SU3); ++i) {
                outfile << "Site " << i << " has " << max_bilinear_neighbors_SU3 << " bilinear partners:" << endl;
                for (size_t j = 0; j < max_bilinear_neighbors_SU3; ++j) {
                    size_t partner = h_bi_partners_SU3[i * max_bilinear_neighbors_SU3 + j];
                    if (partner < lattice_size_SU3) {  // Valid partner index
                        outfile << "  Partner: " << partner << ", Matrix: ";
                        for (size_t k = 0; k < N_SU3 * N_SU3; ++k) {
                            outfile << h_bi_interaction_SU3[i * max_bilinear_neighbors_SU3 * N_SU3 * N_SU3 + j * N_SU3 * N_SU3 + k] << " ";
                        }
                        outfile << endl;
                    }
                }
            }
        }

        // Mixed interactions
        if (max_mixed_trilinear_neighbors_SU2 > 0 && d_mixed_trilinear_partners_SU2 != nullptr) {
            std::vector<size_t> h_mixed_tri_partners_SU2(lattice_size_SU2 * max_mixed_trilinear_neighbors_SU2 * 2);
            std::vector<double> h_mixed_tri_interaction_SU2(lattice_size_SU2 * max_mixed_trilinear_neighbors_SU2 * N_SU2 * N_SU2 * N_SU3);
            std::vector<size_t> h_mixed_tri_partners_SU3(lattice_size_SU3 * max_mixed_trilinear_neighbors_SU3 * 2);
            std::vector<double> h_mixed_tri_interaction_SU3(lattice_size_SU3 * max_mixed_trilinear_neighbors_SU3 * N_SU2 * N_SU2 * N_SU3);
            
            cudaMemcpy(h_mixed_tri_partners_SU2.data(), d_mixed_trilinear_partners_SU2, h_mixed_tri_partners_SU2.size() * sizeof(size_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_mixed_tri_interaction_SU2.data(), d_mixed_trilinear_interaction_SU2, h_mixed_tri_interaction_SU2.size() * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_mixed_tri_partners_SU3.data(), d_mixed_trilinear_partners_SU3, h_mixed_tri_partners_SU3.size() * sizeof(size_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_mixed_tri_interaction_SU3.data(), d_mixed_trilinear_interaction_SU3, h_mixed_tri_interaction_SU3.size() * sizeof(double), cudaMemcpyDeviceToHost);


            outfile << "\nMixed SU2-SU3 Interactions:" << endl;
            outfile << "---------------------------" << endl;
            outfile << "Mixed trilinear interactions from SU2 sites (first 3 sites):" << endl;
            
            for (size_t i = 0; i < min(size_t(3), lattice_size_SU2); ++i) {
                outfile << "SU2 Site " << i << " has " << max_mixed_trilinear_neighbors_SU2 << " mixed partner:" << endl;
                for (size_t j = 0; j < max_mixed_trilinear_neighbors_SU2; ++j) {
                    size_t partner1 = h_mixed_tri_partners_SU2[i * max_mixed_trilinear_neighbors_SU2 * 2 + j * 2];
                    size_t partner2 = h_mixed_tri_partners_SU2[i * max_mixed_trilinear_neighbors_SU2 * 2 + j * 2 + 1];
                    if (partner1 < lattice_size_SU2 && partner2 < lattice_size_SU3) {  // Valid partners
                        outfile << "  Partners: (SU2 site " << partner1 << ", SU3 site " << partner2 << "), Tensor: ";
                        for (size_t k = 0; k < N_SU2 * N_SU2 * N_SU3; ++k) {
                            outfile << h_mixed_tri_interaction_SU2[i * max_trilinear_neighbors_SU2 * N_SU2 * N_SU2 * N_SU3 + j * N_SU2 * N_SU2 * N_SU3 + k] << " ";
                        }
                        outfile << endl;
                    }
                }
            }

            for (size_t i = 0; i < min(size_t(3), lattice_size_SU3); ++i) {
                outfile << "SU3 Site " << i << " has " << max_mixed_trilinear_neighbors_SU3 << " mixed partner:" << endl;
                for (size_t j = 0; j < max_mixed_trilinear_neighbors_SU3; ++j) {
                    size_t partner1 = h_mixed_tri_partners_SU3[i * max_mixed_trilinear_neighbors_SU3 * 2 + j * 2];
                    size_t partner2 = h_mixed_tri_partners_SU3[i * max_mixed_trilinear_neighbors_SU3 * 2 + j * 2 + 1];
                    if (partner1 < lattice_size_SU3 && partner2 < lattice_size_SU2) {  // Valid partners
                        outfile << "  Partners: (SU2 site " << partner1 << ", SU2 site " << partner2 << "), Tensor: ";
                        for (size_t k = 0; k < N_SU2 * N_SU2 * N_SU3; ++k) {
                            outfile << h_mixed_tri_interaction_SU3[i * max_trilinear_neighbors_SU3 * N_SU2 * N_SU2 * N_SU3 + j * N_SU2 * N_SU2 * N_SU3 + k] << " ";
                        }
                        outfile << endl;
                    }
                }
            }
        }

        outfile.close();
        cout << "CUDA lattice information written to " << filename << endl;
    }

    // Host wrapper for computing total energy on GPU
    __host__
    double total_energy_cuda();
    
    __host__
    double energy_density_cuda() {
        return total_energy_cuda() / (lattice_size_SU2 + lattice_size_SU3);
    }
    
    // Host wrapper function for SSPRK53 time step
    __host__
    void SSPRK53_step_cuda(double step_size, double curr_time, double tol);

    __host__
    void euler_step_cuda(double step_size, double curr_time, double tol);

    __host__
    void get_local_field_cuda(double step_size, double curr_time, double tol);

    __host__
    void molecular_dynamics_cuda(double T_start, double T_end, double step_size, string dir_name, 
                                 size_t output_frequency = 1, bool use_adaptive_stepping = false, bool verbose = false) {
        if (dir_name != "") {
            filesystem::create_directory(dir_name);
        }
        
        // Pre-allocate time vector with estimated size
        const size_t estimated_steps = static_cast<size_t>((T_end - T_start) / step_size) + 1;
        vector<double> time;
        time.reserve(estimated_steps);
        
        // Copy initial spins to device if not already there
        copy_spins_to_device();
        
        // Open file handles once instead of reopening for each write
        ofstream mag_file_f, mag_file, mag_file_f_SU3, mag_file_SU3;
        ofstream spin_file_SU2, spin_file_SU3;
        
        if (dir_name != "") {
            mag_file_f.open(dir_name + "/M_t_f.txt", ios::out | ios::trunc);
            mag_file.open(dir_name + "/M_t.txt", ios::out | ios::trunc);
            mag_file_f_SU3.open(dir_name + "/M_t_f_SU3.txt", ios::out | ios::trunc);
            mag_file_SU3.open(dir_name + "/M_t_SU3.txt", ios::out | ios::trunc);
            
            if (verbose) {
                spin_file_SU2.open(dir_name + "/spin_t_SU2.txt", ios::out | ios::trunc);
                spin_file_SU3.open(dir_name + "/spin_t_SU3.txt", ios::out | ios::trunc);
            }
            
            // Set up buffering for better I/O performance
            constexpr size_t BUFFER_SIZE = 8192;
            mag_file_f.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
            mag_file.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
            mag_file_f_SU3.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
            mag_file_SU3.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
            
            if (verbose) {
                spin_file_SU2.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
                spin_file_SU3.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
            }
        }
        
        double current_time = T_start;
        size_t step_count = 0;
        
        // Initial magnetization
        if (dir_name != "") {
            copy_spins_to_host();
            
            // Write initial magnetization
            auto mag_f = this->magnetization_local(this->spins);
            auto mag = this->magnetization_local_antiferromagnetic(this->spins);
            auto mag_SU3 = this->magnetization_local_SU3(this->spins);
            auto mag_afm_SU3 = this->magnetization_local_antiferromagnetic_SU3(this->spins);
            
            for (size_t j = 0; j < N_SU2; ++j) {
                mag_file_f << mag_f[j] << " ";
                mag_file << mag[j] << " ";
            }
            mag_file_f << "\n";
            mag_file << "\n";
            
            for (size_t j = 0; j < N_SU3; ++j) {
                mag_file_f_SU3 << mag_SU3[j] << " ";
                mag_file_SU3 << mag_afm_SU3[j] << " ";
            }
            mag_file_f_SU3 << "\n";
            mag_file_SU3 << "\n";
        }
        
        // Main time evolution loop
        while (current_time < T_end) {
            // Perform time step on GPU
            time.push_back(current_time);
            SSPRK53_step_cuda(step_size, current_time, 1e-6);
            
            current_time += step_size;
            step_count++;
            
            // Print progress
            if (step_count % 100 == 0) {
                std::cout << "Step: " << step_count << ", Time: " << current_time << std::endl;
            }
            
            // Periodically copy data back to host for output
            if (dir_name != "" && step_count % output_frequency == 0) {
                copy_spins_to_host();
                
                // Write output
                auto mag_f = this->magnetization_local(this->spins);
                auto mag = this->magnetization_local_antiferromagnetic(this->spins);
                auto mag_SU3 = this->magnetization_local_SU3(this->spins);
                auto mag_afm_SU3 = this->magnetization_local_antiferromagnetic_SU3(this->spins);
                
                for (size_t j = 0; j < N_SU2; ++j) {
                    mag_file_f << mag_f[j] << " ";
                    mag_file << mag[j] << " ";
                }
                mag_file_f << "\n";
                mag_file << "\n";
                
                for (size_t j = 0; j < N_SU3; ++j) {
                    mag_file_f_SU3 << mag_SU3[j] << " ";
                    mag_file_SU3 << mag_afm_SU3[j] << " ";
                }
                mag_file_f_SU3 << "\n";
                mag_file_SU3 << "\n";
                
                // Write spin states if verbose
                if (verbose) {
                    for (size_t i = 0; i < lattice_size_SU2; ++i) {
                        for (size_t j = 0; j < N_SU2; ++j) {
                            spin_file_SU2 << this->spins.spins_SU2[i][j] << " ";
                        }
                        spin_file_SU2 << "\n";
                    }
                    
                    for (size_t i = 0; i < lattice_size_SU3; ++i) {
                        for (size_t j = 0; j < N_SU3; ++j) {
                            spin_file_SU3 << this->spins.spins_SU3[i][j] << " ";
                        }
                        spin_file_SU3 << "\n";
                    }
                }
            }
        }
        
        // Close output files
        if (dir_name != "") {
            mag_file_f.close();
            mag_file.close();
            mag_file_f_SU3.close();
            mag_file_SU3.close();
            
            if (verbose) {
                spin_file_SU2.close();
                spin_file_SU3.close();
            }
            
            // Write time steps
            ofstream time_sections(dir_name + "/Time_steps.txt", ios::out | ios::trunc);
            time_sections.rdbuf()->pubsetbuf(nullptr, 8192);
            for (const auto& t : time) {
                time_sections << t << "\n";
            }
            time_sections.close();
        }
    }
    // CUDA implementation of M_B_t
    __host__
    void M_B_t_cuda(array<array<double,N_SU2>, N_ATOMS_SU2> &field_in, double t_B, 
                    double pulse_amp, double pulse_width, double pulse_freq, 
                    double T_start, double T_end, double step_size, string dir_name,
                    size_t output_frequency = 1, bool use_adaptive_stepping = false) {
                        
        // Set pulse parameters for SU2
        this->set_pulse_SU2(field_in, t_B, {{0}}, 0, pulse_amp, pulse_width, pulse_freq);

        // Update device parameters
        d_field_drive_freq_SU2 = this->field_drive_freq_SU2;
        d_field_drive_amp_SU2 = this->field_drive_amp_SU2;
        d_field_drive_width_SU2 = this->field_drive_width_SU2;
        d_t_B_1_SU2 = this->t_B_1_SU2;
        d_t_B_2_SU2 = this->t_B_2_SU2;

        // Copy field drive arrays to device
        std::vector<double> temp_drive1_SU2(N_ATOMS_SU2 * N_SU2);
        std::vector<double> temp_drive2_SU2(N_ATOMS_SU2 * N_SU2);
        
        for (size_t i = 0; i < N_ATOMS_SU2; ++i) {
            for (size_t j = 0; j < N_SU2; ++j) {
                temp_drive1_SU2[i * N_SU2 + j] = this->field_drive_1_SU2[i][j];
                temp_drive2_SU2[i * N_SU2 + j] = this->field_drive_2_SU2[i][j];
            }
        }

        std::vector<double> temp_drive1_SU3(N_ATOMS_SU3 * N_SU3);
        std::vector<double> temp_drive2_SU3(N_ATOMS_SU3 * N_SU3);
        for (size_t i = 0; i < N_ATOMS_SU3; ++i) {
            for (size_t j = 0; j < N_SU3; ++j) {
                temp_drive1_SU3[i * N_SU3 + j] = this->field_drive_1_SU3[i][j];
                temp_drive2_SU3[i * N_SU3 + j] = this->field_drive_2_SU3[i][j];
            }
        }
        
        cudaMemcpy(d_field_drive_1_SU2, temp_drive1_SU2.data(), N_ATOMS_SU2 * N_SU2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_field_drive_2_SU2, temp_drive2_SU2.data(), N_ATOMS_SU2 * N_SU2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_field_drive_1_SU3, temp_drive1_SU3.data(), N_ATOMS_SU3 * N_SU3 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_field_drive_2_SU3, temp_drive2_SU3.data(), N_ATOMS_SU3 * N_SU3 * sizeof(double), cudaMemcpyHostToDevice);


        std::cout << "Running M_B_t_cuda with T_start: " << T_start << ", T_end: " << T_end 
                  << ", step_size: " << step_size << ", dir_name: " << dir_name 
                  << ", output_frequency: " << output_frequency 
                  << ", use_adaptive_stepping: " << use_adaptive_stepping << std::endl;
        
        // Run molecular dynamics with CUDA
        molecular_dynamics_cuda(T_start, T_end, step_size, dir_name, output_frequency, use_adaptive_stepping);
    }

    // CUDA implementation of M_BA_BB_t
    __host__
    void M_BA_BB_t_cuda(array<array<double,N_SU2>, N_ATOMS_SU2> &field_in_1, double t_B_1, 
                        array<array<double,N_SU2>, N_ATOMS_SU2> &field_in_2, double t_B_2, 
                        double pulse_amp, double pulse_width, double pulse_freq,
                        double T_start, double T_end, double step_size, string dir_name,
                        size_t output_frequency = 1, bool use_adaptive_stepping = false) {
        // Set both pulses for SU2
        this->set_pulse_SU2(field_in_1, t_B_1, field_in_2, t_B_2, pulse_amp, pulse_width, pulse_freq);

        // Update device parameters
        d_field_drive_freq_SU2 = this->field_drive_freq_SU2;
        d_field_drive_amp_SU2 = this->field_drive_amp_SU2;
        d_field_drive_width_SU2 = this->field_drive_width_SU2;
        d_t_B_1_SU2 = this->t_B_1_SU2;
        d_t_B_2_SU2 = this->t_B_2_SU2;

        // Copy field drive arrays to device
        std::vector<double> temp_drive1_SU2(N_ATOMS_SU2 * N_SU2);
        std::vector<double> temp_drive2_SU2(N_ATOMS_SU2 * N_SU2);
        
        for (size_t i = 0; i < N_ATOMS_SU2; ++i) {
            for (size_t j = 0; j < N_SU2; ++j) {
                temp_drive1_SU2[i * N_SU2 + j] = this->field_drive_1_SU2[i][j];
                temp_drive2_SU2[i * N_SU2 + j] = this->field_drive_2_SU2[i][j];
            }
        }

        std::vector<double> temp_drive1_SU3(N_ATOMS_SU3 * N_SU3);
        std::vector<double> temp_drive2_SU3(N_ATOMS_SU3 * N_SU3);
        for (size_t i = 0; i < N_ATOMS_SU3; ++i) {
            for (size_t j = 0; j < N_SU3; ++j) {
                temp_drive1_SU3[i * N_SU3 + j] = this->field_drive_1_SU3[i][j];
                temp_drive2_SU3[i * N_SU3 + j] = this->field_drive_2_SU3[i][j];
            }
        }
        
        cudaMemcpy(d_field_drive_1_SU2, temp_drive1_SU2.data(), N_ATOMS_SU2 * N_SU2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_field_drive_2_SU2, temp_drive2_SU2.data(), N_ATOMS_SU2 * N_SU2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_field_drive_1_SU3, temp_drive1_SU3.data(), N_ATOMS_SU3 * N_SU3 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_field_drive_2_SU3, temp_drive2_SU3.data(), N_ATOMS_SU3 * N_SU3 * sizeof(double), cudaMemcpyHostToDevice);


        // Run molecular dynamics with CUDA
        molecular_dynamics_cuda(T_start, T_end, step_size, dir_name, output_frequency, use_adaptive_stepping);
    }

    // Copy data from device back to host
    __host__
    void copy_spins_to_host() {
        std::vector<double> temp_spins_SU2(lattice_size_SU2 * N_SU2);
        std::vector<double> temp_spins_SU3(lattice_size_SU3 * N_SU3);
        
        cudaDeviceSynchronize();

        cudaMemcpy(temp_spins_SU2.data(), d_spins.spins_SU2, lattice_size_SU2 * N_SU2 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(temp_spins_SU3.data(), d_spins.spins_SU3, lattice_size_SU3 * N_SU3 * sizeof(double), cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t j = 0; j < N_SU2; ++j) {
                this->spins.spins_SU2[i][j] = temp_spins_SU2[i * N_SU2 + j];
            }
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t j = 0; j < N_SU3; ++j) {
                this->spins.spins_SU3[i][j] = temp_spins_SU3[i * N_SU3 + j];
            }
        }
    }
    
    // Copy data from host to device
    __host__
    void copy_spins_to_device() {
        std::vector<double> temp_spins_SU2(lattice_size_SU2 * N_SU2);
        std::vector<double> temp_spins_SU3(lattice_size_SU3 * N_SU3);
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t j = 0; j < N_SU2; ++j) {
                temp_spins_SU2[i * N_SU2 + j] = this->spins.spins_SU2[i][j];
            }
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t j = 0; j < N_SU3; ++j) {
                temp_spins_SU3[i * N_SU3 + j] = this->spins.spins_SU3[i][j];
            }
        }
        
        cudaMemcpy(d_spins.spins_SU2, temp_spins_SU2.data(), lattice_size_SU2 * N_SU2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_spins.spins_SU3, temp_spins_SU3.data(), lattice_size_SU3 * N_SU3 * sizeof(double), cudaMemcpyHostToDevice);
    }
};

template <size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
__device__
void compute_local_field_SU2(
    double* out,
    int site_index,
    const double* spins_SU2,
    const double* field_SU2,
    const double* onsite_interaction_SU2,
    const double* bilinear_interaction_SU2,
    const size_t* bilinear_partners_SU2,
    const double* trilinear_interaction_SU2,
    const size_t* trilinear_partners_SU2,
    const double* mixed_bilinear_interaction_SU2,
    const size_t* mixed_bilinear_partners_SU2,
    const double* mixed_trilinear_interaction_SU2,
    const size_t* mixed_trilinear_partners_SU2,
    const double* spins_SU3,
    size_t num_bi_SU2,
    size_t num_tri_SU2,
    size_t num_bi_SU2_SU3,
    size_t num_tri_SU2_SU3,
    size_t max_bi_neighbors,
    size_t max_tri_neighbors,
    size_t max_mixed_bi_neighbors,
    size_t max_mixed_tri_neighbors
) {
    double local_field[N_SU2] = {0.0};

    // Onsite contribution with better memory access pattern
    const size_t onsite_base = site_index * N_SU2 * N_SU2;
    const double* spin_site = &spins_SU2[site_index * N_SU2];
    
    #pragma unroll
    for (size_t i = 0; i < N_SU2; ++i) {
        double sum = 0.0;
        #pragma unroll
        for (size_t j = 0; j < N_SU2; ++j) {
            sum += onsite_interaction_SU2[onsite_base + i * N_SU2 + j] * spin_site[j];
        }
        local_field[i] = sum;
    }
    
    // Bilinear contributions with improved memory access
    if (num_bi_SU2 > 0) {
        const size_t partner_base = site_index * max_bi_neighbors;
        const size_t interaction_base = partner_base * N_SU2 * N_SU2;
        
        for (size_t i = 0; i < num_bi_SU2; ++i) {
            const size_t partner = bilinear_partners_SU2[partner_base + i];
            if (partner < lattice_size_SU2) {
                const double* spin_partner = &spins_SU2[partner * N_SU2];
                const size_t i_base = interaction_base + i * N_SU2 * N_SU2;
                
                #pragma unroll
                for (size_t j = 0; j < N_SU2; ++j) {
                    double sum = 0.0;
                    #pragma unroll
                    for (size_t k = 0; k < N_SU2; ++k) {
                        sum += bilinear_interaction_SU2[i_base + j * N_SU2 + k] * spin_partner[k];
                    }
                    local_field[j] += sum;
                }
            }
        }
    }
    
    // Trilinear contributions with optimized access patterns
    if (num_tri_SU2 > 0) {
        const size_t partner_base = site_index * max_tri_neighbors * 2;
        const size_t interaction_base = site_index * max_tri_neighbors * N_SU2 * N_SU2 * N_SU2;
        
        for (size_t i = 0; i < num_tri_SU2; ++i) {
            const size_t partner1 = trilinear_partners_SU2[partner_base + i * 2];
            const size_t partner2 = trilinear_partners_SU2[partner_base + i * 2 + 1];
            
            if (partner1 < lattice_size_SU2 && partner2 < lattice_size_SU2) {
                double temp[N_SU2] = {0.0};
                contract_trilinear_field_device(temp,
                    &trilinear_interaction_SU2[interaction_base + i * N_SU2 * N_SU2 * N_SU2],
                    &spins_SU2[partner1 * N_SU2], 
                    &spins_SU2[partner2 * N_SU2], 
                    N_SU2);
                
                #pragma unroll
                for (size_t j = 0; j < N_SU2; ++j) {
                    local_field[j] += temp[j];
                }
            }
        }
    }

    // Mixed bilinear contributions with improved memory access
    if (num_bi_SU2_SU3 > 0) {
        const size_t partner_base = site_index * max_mixed_bi_neighbors;
        const size_t interaction_base = partner_base * N_SU2 * N_SU3;
        for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
            const size_t partner = mixed_bilinear_partners_SU2[partner_base + i];
            if (partner < lattice_size_SU3) {
                const double* spin_partner = &spins_SU3[partner * N_SU3];
                const size_t i_base = interaction_base + i * N_SU2 * N_SU3;
                
                #pragma unroll
                for (size_t j = 0; j < N_SU2; ++j) {
                    double sum = 0.0;
                    #pragma unroll
                    for (size_t k = 0; k < N_SU3; ++k) {
                        sum += mixed_bilinear_interaction_SU2[i_base + j * N_SU3 + k] * spin_partner[k];
                    }
                    local_field[j] += sum;
                }
            }
        }
    }

    // Mixed trilinear contributions with improved memory access
    if (num_tri_SU2_SU3 > 0) {
        const size_t partner_base = site_index * max_mixed_tri_neighbors * 2;
        const size_t interaction_base = site_index * max_mixed_tri_neighbors * N_SU2 * N_SU2 * N_SU3;
        
        for (size_t i = 0; i < num_tri_SU2_SU3; ++i) {
            const size_t partner1 = mixed_trilinear_partners_SU2[partner_base + i * 2];
            const size_t partner2 = mixed_trilinear_partners_SU2[partner_base + i * 2 + 1];
            
            if (partner1 < lattice_size_SU2 && partner2 < lattice_size_SU3) {
                const double* spin1_ptr = &spins_SU2[partner1 * N_SU2];
                const double* spin3_ptr = &spins_SU3[partner2 * N_SU3];
                const size_t i_interaction_base = interaction_base + i * N_SU2 * N_SU2 * N_SU3;
                
                // Rearranged loops for better cache usage
                #pragma unroll
                for (size_t a = 0; a < N_SU2; ++a) {
                    double temp = 0.0;
                    const size_t a_base = i_interaction_base + a * N_SU2 * N_SU3;
                    
                    // Inner loops reordered for coalesced memory access
                    #pragma unroll
                    for (size_t c = 0; c < N_SU3; ++c) {
                        const double spin3_c = spin3_ptr[c];
                        double inner_sum = 0.0;
                        
                        #pragma unroll
                        for (size_t b = 0; b < N_SU2; ++b) {
                            inner_sum += mixed_trilinear_interaction_SU2[a_base + b * N_SU3 + c] * spin1_ptr[b];
                        }
                        temp += inner_sum * spin3_c;
                    }
                    local_field[a] += temp;
                }
            }
        }
    }
    
    // Subtract external field with vectorized operation
    const double* field_ptr = &field_SU2[site_index * N_SU2];
    #pragma unroll
    for (size_t i = 0; i < N_SU2; ++i) {
        out[site_index * N_SU2 + i] = local_field[i] - field_ptr[i];
    }
}

template <size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
__device__
void compute_local_field_SU3(
    double* out,
    int site_index,
    const double* spins_SU3,
    const double* field_SU3,
    const double* onsite_interaction_SU3,
    const double* bilinear_interaction_SU3,
    const size_t* bilinear_partners_SU3,
    const double* trilinear_interaction_SU3,
    const size_t* trilinear_partners_SU3,
    const double* mixed_bilinear_interaction_SU3,
    const size_t* mixed_bilinear_partners_SU3,
    const double* mixed_trilinear_interaction_SU3,
    const size_t* mixed_trilinear_partners_SU3,
    const double* spins_SU2,
    size_t num_bi_SU3,
    size_t num_tri_SU3,
    size_t num_bi_SU2_SU3,
    size_t num_tri_SU2_SU3,
    size_t max_bi_neighbors,
    size_t max_tri_neighbors,
    size_t max_mixed_bi_neighbors,
    size_t max_mixed_tri_neighbors
) {
    double local_field[N_SU3] = {0.0};

    // Onsite contribution with better memory access pattern
    const size_t onsite_base = site_index * N_SU3 * N_SU3;
    const double* spin_site = &spins_SU3[site_index * N_SU3];
    
    #pragma unroll
    for (size_t i = 0; i < N_SU3; ++i) {
        double sum = 0.0;
        #pragma unroll
        for (size_t j = 0; j < N_SU3; ++j) {
            sum += onsite_interaction_SU3[onsite_base + i * N_SU3 + j] * spin_site[j];
        }
        local_field[i] = sum;
    }
    
    // Bilinear contributions with improved memory access
    if (num_bi_SU3 > 0) {
        const size_t partner_base = site_index * max_bi_neighbors;
        const size_t interaction_base = partner_base * N_SU3 * N_SU3;
        
        for (size_t i = 0; i < num_bi_SU3; ++i) {
            const size_t partner = bilinear_partners_SU3[partner_base + i];
            if (partner < lattice_size_SU3) {
                const double* spin_partner = &spins_SU3[partner * N_SU3];
                const size_t i_base = interaction_base + i * N_SU3 * N_SU3;
                
                #pragma unroll
                for (size_t j = 0; j < N_SU3; ++j) {
                    double sum = 0.0;
                    #pragma unroll
                    for (size_t k = 0; k < N_SU3; ++k) {
                        sum += bilinear_interaction_SU3[i_base + j * N_SU3 + k] * spin_partner[k];
                    }
                    local_field[j] += sum;
                }
            }
        }
    }
    
    // Trilinear contributions with optimized access patterns
    if (num_tri_SU3 > 0) {
        const size_t partner_base = site_index * max_tri_neighbors * 2;
        const size_t interaction_base = site_index * max_tri_neighbors * N_SU3 * N_SU3 * N_SU3;
        
        for (size_t i = 0; i < num_tri_SU3; ++i) {
            const size_t partner1 = trilinear_partners_SU3[partner_base + i * 2];
            const size_t partner2 = trilinear_partners_SU3[partner_base + i * 2 + 1];
            
            if (partner1 < lattice_size_SU3 && partner2 < lattice_size_SU3) {
                double temp[N_SU3] = {0.0};
                contract_trilinear_field_device(temp,
                    &trilinear_interaction_SU3[interaction_base + i * N_SU3 * N_SU3 * N_SU3],
                    &spins_SU3[partner1 * N_SU3], 
                    &spins_SU3[partner2 * N_SU3], 
                    N_SU3);
                
                #pragma unroll
                for (size_t j = 0; j < N_SU3; ++j) {
                    local_field[j] += temp[j];
                }
            }
        }
    }

    // Mixed bilinear contributions with improved memory access
    if (num_bi_SU2_SU3 > 0) {
        const size_t partner_base = site_index * max_mixed_bi_neighbors;
        const size_t interaction_base = partner_base * N_SU2 * N_SU3;
        for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
            const size_t partner = mixed_bilinear_partners_SU3[partner_base + i];
            if (partner < lattice_size_SU2) {
                const double* spin_partner = &spins_SU2[partner * N_SU2];
                const size_t i_base = interaction_base + i * N_SU2 * N_SU3;
                
                #pragma unroll
                for (size_t j = 0; j < N_SU3; ++j) {
                    double sum = 0.0;
                    #pragma unroll
                    for (size_t k = 0; k < N_SU2; ++k) {
                        sum += mixed_bilinear_interaction_SU3[i_base + j * N_SU2 + k] * spin_partner[k];
                    }
                    local_field[j] += sum;
                }
            }
        }
    }
    // Mixed trilinear contributions with improved memory access
    if (num_tri_SU2_SU3 > 0) {
        const size_t partner_base = site_index * max_mixed_tri_neighbors * 2;
        const size_t interaction_base = site_index * max_mixed_tri_neighbors * N_SU2 * N_SU2 * N_SU3;
        
        for (size_t i = 0; i < num_tri_SU2_SU3; ++i) {
            const size_t partner1 = mixed_trilinear_partners_SU3[partner_base + i * 2];
            const size_t partner2 = mixed_trilinear_partners_SU3[partner_base + i * 2 + 1];
            
            if (partner1 < lattice_size_SU2 && partner2 < lattice_size_SU2) {
                const double* spin1_ptr = &spins_SU2[partner1 * N_SU2];
                const double* spin2_ptr = &spins_SU2[partner2 * N_SU2];
                const size_t i_interaction_base = interaction_base + i * N_SU2 * N_SU2 * N_SU3;
                
                // Rearranged loops for better cache usage
                #pragma unroll
                for (size_t a = 0; a < N_SU3; ++a) {
                    double temp = 0.0;
                    const size_t a_base = i_interaction_base + a * N_SU2 * N_SU2;
                    
                    // Inner loops reordered for coalesced memory access
                    #pragma unroll
                    for (size_t c = 0; c < N_SU2; ++c) {
                        const double spin2_c = spin2_ptr[c];
                        double inner_sum = 0.0;
                        
                        #pragma unroll
                        for (size_t b = 0; b < N_SU2; ++b) {
                            inner_sum += mixed_trilinear_interaction_SU3[a_base + b * N_SU2 + c] * spin1_ptr[b];
                        }
                        temp += inner_sum * spin2_c;
                    }
                    local_field[a] += temp;
                }
            }
        }
    }
    
    // Subtract external field with vectorized operation
    const double* field_ptr = &field_SU3[site_index * N_SU3];
    #pragma unroll
    for (size_t i = 0; i < N_SU3; ++i) {
        out[site_index * N_SU3 + i] = local_field[i] - field_ptr[i];
    }
}


template <size_t N_SU2, size_t N_ATOMS_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t lattice_size_SU3>
__device__
void drive_field_T_SU2(
    double* out, double currT, int site_index,
    double* d_field_drive_1_SU2, double* d_field_drive_2_SU2, 
    double d_field_drive_amp_SU2, double d_field_drive_width_SU2, 
    double d_field_drive_freq_SU2, double d_t_B_1_SU2, double d_t_B_2_SU2, 
    size_t max_mixed_tri_neighbors, double* mixed_trilinear_interaction_SU2, size_t* mixed_trilinear_partners_SU2,
    double* d_spins_SU3)
{
    // Pre-compute common exponential terms
    const double dt1 = currT - d_t_B_1_SU2;
    const double dt2 = currT - d_t_B_2_SU2;
    const double inv_2width_sq = 1.0 / (4.0 * d_field_drive_width_SU2 * d_field_drive_width_SU2);
    const double exp1 = exp(-dt1 * dt1 * inv_2width_sq);
    const double exp2 = exp(-dt2 * dt2 * inv_2width_sq);
    const double omega = 2.0 * M_PI * d_field_drive_freq_SU2;
    
    // Compute factors once
    const double factor1_SU2 = d_field_drive_amp_SU2 * exp1 * cos(omega * dt1);
    const double factor2_SU2 = d_field_drive_amp_SU2 * exp2 * cos(omega * dt2);
    
    // Cache sublattice index for site
    const int site_sublattice = site_index % N_ATOMS_SU2;
    const size_t site_sublattice_base = site_sublattice * N_SU2;

    if (factor1_SU2 < 1e-14 && factor2_SU2 < 1e-14) return;
    
    // Initialize output with direct field contribution
    #pragma unroll
    for (size_t i = 0; i < N_SU2; ++i) {
        out[site_index * N_SU2 + i] -= (d_field_drive_1_SU2[site_sublattice_base + i] * factor1_SU2 + 
                                        d_field_drive_2_SU2[site_sublattice_base + i] * factor2_SU2);
    }
}

template <size_t N_SU2, size_t N_ATOMS_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t lattice_size_SU3>
__device__
void drive_field_T_SU3(
    double* out, double currT, int site_index,
    double* d_field_drive_1_SU3, double* d_field_drive_2_SU3,
    double d_field_drive_amp_SU2, double d_field_drive_width_SU2,
    double d_field_drive_freq_SU2, double d_t_B_1_SU2, double d_t_B_2_SU2,
    size_t max_mixed_tri_neighbors, double* mixed_trilinear_interaction_SU3,
    size_t* mixed_trilinear_partners_SU3, double* d_spins_SU2)
{
    // Pre-compute common exponential terms
    const double dt1 = currT - d_t_B_1_SU2;
    const double dt2 = currT - d_t_B_2_SU2;
    const double inv_2width_sq = 1.0 / (4.0 * d_field_drive_width_SU2 * d_field_drive_width_SU2);
    const double exp1 = exp(-dt1 * dt1 * inv_2width_sq);
    const double exp2 = exp(-dt2 * dt2 * inv_2width_sq);
    const double omega = 2.0 * M_PI * d_field_drive_freq_SU2;
    
    // Compute factors once
    const double factor1_SU2 = d_field_drive_amp_SU2 * exp1 * cos(omega * dt1);
    const double factor2_SU2 = d_field_drive_amp_SU2 * exp2 * cos(omega * dt2);
    
    // Early exit if factors are small
    if (factor1_SU2 < 1e-14 && factor2_SU2 < 1e-14) return;

    // Initialize output with direct field contribution
    #pragma unroll
    for (size_t i = 0; i < N_SU3; ++i) {
        out[site_index * N_SU3 + i] -= (d_field_drive_1_SU3[site_sublattice_base + i] * factor1_SU2 + 
                                        d_field_drive_2_SU3[site_sublattice_base + i] * factor2_SU2);
    }
}

__global__
void compute_site_energy_SU2_kernel(
    double* d_energies,
    const double* spins_SU2,
    const double* field_SU2,
    const double* onsite_interaction_SU2,
    const double* bilinear_interaction_SU2,
    const size_t* bilinear_partners_SU2,
    const double* trilinear_interaction_SU2,
    const size_t* trilinear_partners_SU2,
    const double* mixed_trilinear_interaction_SU2,
    const size_t* mixed_trilinear_partners_SU2,
    const double* spins_SU3,
    size_t num_bi_SU2,
    size_t num_tri_SU2,
    size_t num_tri_SU2_SU3,
    size_t max_bi_neighbors,
    size_t max_tri_neighbors,
    size_t max_mixed_tri_neighbors
);

// CUDA kernel for computing site energy for SU3
__global__
void compute_site_energy_SU3_kernel(
    double* d_energies,
    const double* spins_SU3,
    const double* field_SU3,
    const double* onsite_interaction_SU3,
    const double* bilinear_interaction_SU3,
    const size_t* bilinear_partners_SU3,
    const double* trilinear_interaction_SU3,
    const size_t* trilinear_partners_SU3,
    const double* mixed_trilinear_interaction_SU3,
    const size_t* mixed_trilinear_partners_SU3,
    const double* spins_SU2,
    size_t num_bi_SU3,
    size_t num_tri_SU3,
    size_t num_tri_SU2_SU3,
    size_t max_bi_neighbors,
    size_t max_tri_neighbors,
    size_t max_mixed_tri_neighbors
);


template <size_t N_SU2, size_t lattice_size_SU2>
__device__
void landau_Lifshitz_SU2(double* out, int site_index, double* spins, const double* local_field) {
    double spin[N_SU2], local_field_here[N_SU2];
    for (size_t i = 0; i < N_SU2; ++i) {
        spin[i] = spins[site_index * N_SU2 + i];
        local_field_here[i] = local_field[site_index * N_SU2 + i];
    }

    // Compute the cross product with the local field
    double cross[N_SU2];
    cross_product_SU2_device(cross, spin, local_field_here);

    // Update the spins using the Landau-Lifshitz equation
    for (size_t i = 0; i < N_SU2; ++i) {
        out[site_index * N_SU2 + i] = cross[i];
    }
}

template <size_t N_SU3, size_t lattice_size_SU3>
__device__
void landau_Lifshitz_SU3(double* out, int site_index, double* spins, const double* local_field) {
    double spin[N_SU3], local_field_here[N_SU3];
    for (size_t i = 0; i < N_SU3; ++i) {
        spin[i] = spins[site_index * N_SU3 + i];
        local_field_here[i] = local_field[site_index * N_SU3 + i];
    }

    // Compute the cross product with the local field
    double cross[N_SU3];
    cross_product_SU3_device(cross, spin, local_field_here);

    // Update the spins using the Landau-Lifshitz equation
    for (size_t i = 0; i < N_SU3; ++i) {
        out[site_index * N_SU3 + i] = cross[i];
    }
}


template<size_t N_SU2, size_t N_ATOMS_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t lattice_size_SU3>
__global__
void LLG_kernel(
    double* k_SU2, double* k_SU3,
    double* d_spins_SU2, double* d_spins_SU3,
    double* d_local_field_SU2, double* d_local_field_SU3,
    double* d_field_SU2, double* d_field_SU3,
    double* d_onsite_interaction_SU2, double* d_onsite_interaction_SU3,
    double* d_bilinear_interaction_SU2, double* d_bilinear_interaction_SU3,
    size_t* d_bilinear_partners_SU2, size_t* d_bilinear_partners_SU3,
    double* d_trilinear_interaction_SU2, double* d_trilinear_interaction_SU3,
    size_t* d_trilinear_partners_SU2, size_t* d_trilinear_partners_SU3,
    double* d_mixed_bilinear_interaction_SU2, double* d_mixed_bilinear_interaction_SU3,
    size_t* d_mixed_bilinear_partners_SU2, size_t* d_mixed_bilinear_partners_SU3,
    double* d_mixed_trilinear_interaction_SU2, double* d_mixed_trilinear_interaction_SU3,
    size_t* d_mixed_trilinear_partners_SU2, size_t* d_mixed_trilinear_partners_SU3,
    size_t num_bi_SU2, size_t num_tri_SU2, size_t num_bi_SU3, size_t num_tri_SU3, size_t num_bi_SU2_SU3, size_t num_tri_SU2_SU3,
    size_t max_bi_neighbors_SU2, size_t max_tri_neighbors_SU2, size_t max_mixed_bi_neighbors_SU2, size_t max_mixed_tri_neighbors_SU2,
    size_t max_bi_neighbors_SU3, size_t max_tri_neighbors_SU3, size_t max_mixed_bi_neighbors_SU3, size_t max_mixed_tri_neighbors_SU3,
    double* d_field_drive_1_SU2, double* d_field_drive_2_SU2, double* d_field_drive_1_SU3, double* d_field_drive_2_SU3,
    double d_field_drive_amp_SU2, double d_field_drive_width_SU2, double d_field_drive_freq_SU2, double d_t_B_1_SU2, double d_t_B_2_SU2,
    double curr_time, double dt);


template <size_t N_SU2, size_t N_ATOMS_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t lattice_size_SU3>
__host__
void SSPRK53_step_kernel(
    double* d_spins_SU2, double* d_spins_SU3,
    double* d_local_field_SU2, double* d_local_field_SU3,
    double* d_field_SU2, double* d_field_SU3,
    double* d_onsite_interaction_SU2, double* d_onsite_interaction_SU3,
    double* d_bilinear_interaction_SU2, double* d_bilinear_interaction_SU3,
    size_t* d_bilinear_partners_SU2, size_t* d_bilinear_partners_SU3,
    double* d_trilinear_interaction_SU2, double* d_trilinear_interaction_SU3,
    size_t* d_trilinear_partners_SU2, size_t* d_trilinear_partners_SU3,
    double* d_mixed_trilinear_interaction_SU2, double* d_mixed_trilinear_interaction_SU3,
    size_t* d_mixed_trilinear_partners_SU2, size_t* d_mixed_trilinear_partners_SU3,
    size_t num_bi_SU2, size_t num_tri_SU2, size_t num_bi_SU3, size_t num_tri_SU3, size_t num_tri_SU2_SU3,
    size_t max_bi_neighbors_SU2, size_t max_tri_neighbors_SU2, size_t max_mixed_tri_neighbors_SU2,
    size_t max_bi_neighbors_SU3, size_t max_tri_neighbors_SU3, size_t max_mixed_tri_neighbors_SU3,
    double* d_field_drive_1_SU2, double* d_field_drive_2_SU2, double* d_field_drive_1_SU3, double* d_field_drive_2_SU3,
    double d_field_drive_amp_SU2, double d_field_drive_width_SU2, double d_field_drive_freq_SU2, double d_t_B_1_SU2, double d_t_B_2_SU2,
    double curr_time, double dt, double spin_length_SU2, double spin_length_SU3,
    double* work_SU2_1, double* work_SU2_2, double* work_SU2_3,
    double* work_SU3_1, double* work_SU3_2, double* work_SU3_3);

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t lattice_size_SU3>
__host__
void euler_step_kernel(
    double* d_spins_SU2, double* d_spins_SU3,
    double* d_local_field_SU2, double* d_local_field_SU3,
    double* d_field_SU2, double* d_field_SU3,
    double* d_onsite_interaction_SU2, double* d_onsite_interaction_SU3,
    double* d_bilinear_interaction_SU2, double* d_bilinear_interaction_SU3,
    size_t* d_bilinear_partners_SU2, size_t* d_bilinear_partners_SU3,
    double* d_trilinear_interaction_SU2, double* d_trilinear_interaction_SU3,
    size_t* d_trilinear_partners_SU2, size_t* d_trilinear_partners_SU3,
    double* d_mixed_trilinear_interaction_SU2, double* d_mixed_trilinear_interaction_SU3,
    size_t* d_mixed_trilinear_partners_SU2, size_t* d_mixed_trilinear_partners_SU3,
    size_t num_bi_SU2, size_t num_tri_SU2, size_t num_bi_SU3, size_t num_tri_SU3, size_t num_tri_SU2_SU3,
    size_t max_bi_neighbors_SU2, size_t max_tri_neighbors_SU2, size_t max_mixed_tri_neighbors_SU2,
    size_t max_bi_neighbors_SU3, size_t max_tri_neighbors_SU3, size_t max_mixed_tri_neighbors_SU3,
    double* d_field_drive_1_SU2, double* d_field_drive_2_SU2, double* d_field_drive_1_SU3, double* d_field_drive_2_SU3,
    double d_field_drive_amp_SU2, double d_field_drive_width_SU2, double d_field_drive_freq_SU2, double d_t_B_1_SU2, double d_t_B_2_SU2,
    double curr_time, double dt, double spin_length_SU2, double spin_length_SU3);

#endif // MIXED_LATTICE_CUDA_CUH