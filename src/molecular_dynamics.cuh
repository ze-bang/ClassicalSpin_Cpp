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

// CUDA kernels for random number generation
__global__
void init_rng_states(curandState* states, size_t n_states, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_states) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ 
double curand_uniform_lehman(curandState* state) {
    return curand_uniform_double(state);
}

__device__
void generate_random_spin_SU2_device(double* spin, double spin_length, curandState* state) {
    const int N_SU2 = 3; // Assuming SU2 has 3 components
    double z = 2.0 * curand_uniform_double(state) - 1.0;
    double r = sqrt(1.0 - z * z);
    
    if (N_SU2 >= 3) {
        double phi = 2.0 * M_PI * curand_uniform_double(state);
        spin[0] = r * cos(phi);
        spin[1] = r * sin(phi);
        spin[N_SU2-1] = z;
        
        // Normalize and scale
        double norm = 0;
        for (int i = 0; i < N_SU2; i++) {
            norm += spin[i] * spin[i];
        }
        norm = sqrt(norm);
        for (int i = 0; i < N_SU2; i++) {
            spin[i] = spin[i] / norm * spin_length;
        }
    }
}

__device__
void generate_random_spin_SU3_device(double* spin, double spin_length, curandState* state) {
    const int N_SU3 = 8; // Assuming SU3 has 8 components
    double z = 2.0 * curand_uniform_double(state) - 1.0;
    double r = sqrt(1.0 - z * z);
    
    if (N_SU3 >= 8) {
        // Generate random angles for higher dimensional sphere
        for (int i = 0; i < N_SU3 - 2; i++) {
            double angle = 2.0 * M_PI * curand_uniform_double(state);
            spin[i] = r;
            for (int j = 0; j < i; j++) {
                spin[i] *= sin(angle);
            }
            if (i == N_SU3 - 3) {
                spin[i + 1] = spin[i] * sin(angle);
            }
            spin[i] *= cos(angle);
        }
        spin[N_SU3-1] = z;
        
        // Normalize and scale
        double norm = 0;
        for (int i = 0; i < N_SU3; i++) {
            norm += spin[i] * spin[i];
        }
        norm = sqrt(norm);
        for (int i = 0; i < N_SU3; i++) {
            spin[i] = spin[i] / norm * spin_length;
        }
    }
}

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
        // initialize_random_states();
        
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
            max_mixed_trilinear_neighbors_SU2 = std::max(max_mixed_trilinear_neighbors_SU2, this->mixed_trilinear_partners_SU2[i].size()/2);
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
                for (size_t i = 0; i < interactions.size() && i < max_bilinear_neighbors_SU2; ++i) {
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
                
                for (size_t i = 0; i < interactions.size() && i < max_bilinear_neighbors_SU3; ++i) {
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

                for (size_t i = 0; i < interactions.size() && i < max_trilinear_neighbors_SU2; ++i) {
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

                for (size_t i = 0; i < interactions.size() && i < max_trilinear_neighbors_SU3; ++i) {
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

                for (size_t i = 0; i < interactions.size() && i < max_mixed_bilinear_neighbors_SU2; ++i) {
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

                for (size_t i = 0; i < interactions.size() && i < max_mixed_trilinear_neighbors_SU2; ++i) {
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

                for (size_t i = 0; i < interactions.size() && i < max_mixed_trilinear_neighbors_SU3; ++i) {
                    size_t base_idx = site * max_mixed_trilinear_neighbors_SU3 * N_SU3 * N_SU2 * N_SU2 + i * N_SU3 * N_SU2 * N_SU2;
                    for (size_t j = 0; j < N_SU2 * N_SU2 * N_SU3; ++j) {
                        temp_mixed_tri_SU2[base_idx + j] = interactions[i][j];
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
    
    // Initialize random number generator states
    __host__
    void initialize_random_states() {
        dim3 block(256);
        dim3 grid((lattice_size_SU2 + lattice_size_SU3 + block.x - 1) / block.x);
        
        // Launch kernel to initialize random states
        init_rng_states<<<grid, block>>>(d_rng_states, lattice_size_SU2 + lattice_size_SU3, time(NULL));
        cudaDeviceSynchronize();
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
    private:
        // CUDA device functions for vector operations
        __device__
        double dot_device(const double* a, const double* b, size_t n) {
            double result = 0.0;
            for (size_t i = 0; i < n; ++i) {
                result += a[i] * b[i];
            }
            return result;
        }
        
        __device__
        double contract_device(const double* spin1, const double* matrix, const double* spin2, size_t n) {
            double result = 0.0;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    result += spin1[i] * matrix[i * n + j] * spin2[j];
                }
            }
            return result;
        }
        
        __device__
        double contract_trilinear_device(const double* tensor, const double* spin1, const double* spin2, const double* spin3, size_t n) {
            double result = 0.0;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    for (size_t k = 0; k < n; ++k) {
                        result += tensor[i * n * n + j * n + k] * spin1[i] * spin2[j] * spin3[k];
                    }
                }
            }
            return result;
        }
        
        __device__
        void multiply_matrix_vector_device(double* result, const double* matrix, const double* vector, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                result[i] = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    result[i] += matrix[i * n + j] * vector[j];
                }
            }
        }
        
        __device__
        void contract_trilinear_field_device(double* result, const double* tensor, const double* spin1, const double* spin2, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                result[i] = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    for (size_t k = 0; k < n; ++k) {
                        result[i] += tensor[i * n * n + j * n + k] * spin1[j] * spin2[k];
                    }
                }
            }
        }

        __device__
        void cross_product_SU2_device(double* result, const double* a, const double* b) {
            result[0] = a[1] * b[2] - a[2] * b[1];
            result[1] = a[2] * b[0] - a[0] * b[2];
            result[2] = a[0] * b[1] - a[1] * b[0];
        }

        __device__
        void cross_product_SU3_device(double* result, const double* a, const double* b) {
            // Access the structure tensor for this site
            const double* f_tensor = &d_SU3_structure_tensor[0];

            // Compute cross product: result[alpha] = f[alpha][beta][gamma] * a[beta] * b[gamma]
            for (size_t alpha = 0; alpha < N_SU3; ++alpha) {
                result[alpha] = 0.0;
                for (size_t beta = 0; beta < N_SU3; ++beta) {
                    for (size_t gamma = 0; gamma < N_SU3; ++gamma) {
                        result[alpha] += f_tensor[alpha * N_SU3 * N_SU3 + beta * N_SU3 + gamma] * a[beta] * b[gamma];
                    }
                }
            }
        }
        


    public:
        // CUDA kernel for computing site energy for SU2
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
            size_t max_mixed_tri_neighbors,
            size_t lattice_size_SU2
        ) {
            int site_index = blockIdx.x * blockDim.x + threadIdx.x;
            if (site_index >= lattice_size_SU2) return;
            
            double energy = 0.0;
            const double* spin_here = &spins_SU2[site_index * N_SU2];
            
            // Field energy
            for (size_t i = 0; i < N_SU2; ++i) {
                energy -= spin_here[i] * field_SU2[site_index * N_SU2 + i];
            }
            
            // Onsite energy
            energy += contract_device(spin_here, &onsite_interaction_SU2[site_index * N_SU2 * N_SU2], spin_here, N_SU2);
            
            // Bilinear interactions
            for (size_t i = 0; i < num_bi_SU2 && i < max_bi_neighbors; ++i) {
                size_t partner = bilinear_partners_SU2[site_index * max_bi_neighbors + i];
                if (partner < lattice_size_SU2) {
                    energy += contract_device(spin_here, 
                        &bilinear_interaction_SU2[site_index * max_bi_neighbors * N_SU2 * N_SU2 + i * N_SU2 * N_SU2],
                        &spins_SU2[partner * N_SU2], N_SU2);
                }
            }
            
            // Trilinear interactions
            for (size_t i = 0; i < num_tri_SU2 && i < max_tri_neighbors; ++i) {
                size_t partner1 = trilinear_partners_SU2[site_index * max_tri_neighbors * 2 + i * 2];
                size_t partner2 = trilinear_partners_SU2[site_index * max_tri_neighbors * 2 + i * 2 + 1];
                if (partner1 < lattice_size_SU2 && partner2 < lattice_size_SU2) {
                    energy += contract_trilinear_device(
                        &trilinear_interaction_SU2[site_index * max_tri_neighbors * N_SU2 * N_SU2 * N_SU2 + i * N_SU2 * N_SU2 * N_SU2],
                        spin_here, &spins_SU2[partner1 * N_SU2], &spins_SU2[partner2 * N_SU2], N_SU2);
                }
            }
            
            // Mixed trilinear interactions
            for (size_t i = 0; i < num_tri_SU2_SU3 && i < max_mixed_tri_neighbors; ++i) {
                size_t partner1 = mixed_trilinear_partners_SU2[site_index * max_mixed_tri_neighbors * 2 + i * 2];
                size_t partner2 = mixed_trilinear_partners_SU2[site_index * max_mixed_tri_neighbors * 2 + i * 2 + 1];
                if (partner1 < lattice_size_SU2) {
                    // Note: mixed interaction tensor has dimensions N_SU2 x N_SU2 x N_SU3
                    double partial_energy = 0.0;
                    for (size_t a = 0; a < N_SU2; ++a) {
                        for (size_t b = 0; b < N_SU2; ++b) {
                            for (size_t c = 0; c < N_SU3; ++c) {
                                partial_energy += mixed_trilinear_interaction_SU2[
                                    site_index * max_mixed_tri_neighbors * N_SU2 * N_SU2 * N_SU3 + 
                                    i * N_SU2 * N_SU2 * N_SU3 + a * N_SU2 * N_SU3 + b * N_SU3 + c] *
                                    spin_here[a] * spins_SU2[partner1 * N_SU2 + b] * spins_SU3[partner2 * N_SU3 + c];
                            }
                        }
                    }
                    energy += partial_energy;
                }
            }
            
            d_energies[site_index] = energy;
        }
        
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
            size_t max_mixed_tri_neighbors,
            size_t lattice_size_SU3,
            size_t lattice_size_SU2
        ) {
            int site_index = blockIdx.x * blockDim.x + threadIdx.x;
            if (site_index >= lattice_size_SU3) return;
            
            double energy = 0.0;
            const double* spin_here = &spins_SU3[site_index * N_SU3];
            
            // Field energy
            for (size_t i = 0; i < N_SU3; ++i) {
                energy -= spin_here[i] * field_SU3[site_index * N_SU3 + i];
            }
            
            // Onsite energy
            energy += contract_device(spin_here, &onsite_interaction_SU3[site_index * N_SU3 * N_SU3], spin_here, N_SU3);
            
            // Bilinear interactions
            for (size_t i = 0; i < num_bi_SU3 && i < max_bi_neighbors; ++i) {
                size_t partner = bilinear_partners_SU3[site_index * max_bi_neighbors + i];
                if (partner < lattice_size_SU3) {
                    energy += contract_device(spin_here, 
                        &bilinear_interaction_SU3[site_index * max_bi_neighbors * N_SU3 * N_SU3 + i * N_SU3 * N_SU3],
                        &spins_SU3[partner * N_SU3], N_SU3);
                }
            }
            
            // Trilinear interactions
            for (size_t i = 0; i < num_tri_SU3 && i < max_tri_neighbors; ++i) {
                size_t partner1 = trilinear_partners_SU3[site_index * max_tri_neighbors * 2 + i * 2];
                size_t partner2 = trilinear_partners_SU3[site_index * max_tri_neighbors * 2 + i * 2 + 1];
                if (partner1 < lattice_size_SU3 && partner2 < lattice_size_SU3) {
                    energy += contract_trilinear_device(
                        &trilinear_interaction_SU3[site_index * max_tri_neighbors * N_SU3 * N_SU3 * N_SU3 + i * N_SU3 * N_SU3 * N_SU3],
                        spin_here, &spins_SU3[partner1 * N_SU3], &spins_SU3[partner2 * N_SU3], N_SU3);
                }
            }
            
            // Mixed trilinear interactions
            for (size_t i = 0; i < num_tri_SU2_SU3 && i < max_mixed_tri_neighbors; ++i) {
                size_t partner1 = mixed_trilinear_partners_SU3[site_index * max_mixed_tri_neighbors * 2 + i * 2];
                size_t partner2 = mixed_trilinear_partners_SU3[site_index * max_mixed_tri_neighbors * 2 + i * 2 + 1];
                if (partner1 < lattice_size_SU2 && partner2 < lattice_size_SU2) {
                    // Note: mixed interaction tensor has dimensions N_SU3 x N_SU2 x N_SU2
                    double partial_energy = 0.0;
                    for (size_t a = 0; a < N_SU3; ++a) {
                        for (size_t b = 0; b < N_SU2; ++b) {
                            for (size_t c = 0; c < N_SU2; ++c) {
                                partial_energy += mixed_trilinear_interaction_SU3[
                                    site_index * max_mixed_tri_neighbors * N_SU3 * N_SU2 * N_SU2 + 
                                    i * N_SU3 * N_SU2 * N_SU2 + a * N_SU2 * N_SU2 + b * N_SU2 + c] *
                                    spin_here[a] * spins_SU2[partner1 * N_SU2 + b] * spins_SU2[partner2 * N_SU2 + c];
                            }
                        }
                    }
                    energy += partial_energy;
                }
            }
            
            d_energies[site_index] = energy;
        }
        
        // Host wrapper for computing total energy on GPU
        __host__
        double total_energy_cuda() {
            double total_energy = 0.0;
            
            // Allocate device memory for energies
            double *d_energies_SU2, *d_energies_SU3;
            cudaMalloc(&d_energies_SU2, lattice_size_SU2 * sizeof(double));
            cudaMalloc(&d_energies_SU3, lattice_size_SU3 * sizeof(double));
            
            // Launch kernels
            dim3 block(256);
            dim3 grid_SU2((lattice_size_SU2 + block.x - 1) / block.x);
            dim3 grid_SU3((lattice_size_SU3 + block.x - 1) / block.x);
            
            compute_site_energy_SU2_kernel<<<grid_SU2, block>>>(
                d_energies_SU2, d_spins.spins_SU2, d_field_SU2, d_onsite_interaction_SU2,
                d_bilinear_interaction_SU2, d_bilinear_partners_SU2,
                d_trilinear_interaction_SU2, d_trilinear_partners_SU2,
                d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2,
                d_spins.spins_SU3, d_num_bi_SU2, d_num_tri_SU2, d_num_tri_SU2_SU3,
                max_bilinear_neighbors_SU2, max_trilinear_neighbors_SU2, max_mixed_trilinear_neighbors_SU2,
                lattice_size_SU2
            );
            
            compute_site_energy_SU3_kernel<<<grid_SU3, block>>>(
                d_energies_SU3, d_spins.spins_SU3, d_field_SU3, d_onsite_interaction_SU3,
                d_bilinear_interaction_SU3, d_bilinear_partners_SU3,
                d_trilinear_interaction_SU3, d_trilinear_partners_SU3,
                d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3,
                d_spins.spins_SU2, d_num_bi_SU3, d_num_tri_SU3, d_num_tri_SU2_SU3,
                max_bilinear_neighbors_SU3, max_trilinear_neighbors_SU3, max_mixed_trilinear_neighbors_SU3,
                lattice_size_SU3, lattice_size_SU2
            );
            
            // Reduce energies
            thrust::device_ptr<double> d_ptr_SU2(d_energies_SU2);
            thrust::device_ptr<double> d_ptr_SU3(d_energies_SU3);
            
            double energy_SU2 = thrust::reduce(d_ptr_SU2, d_ptr_SU2 + lattice_size_SU2);
            double energy_SU3 = thrust::reduce(d_ptr_SU3, d_ptr_SU3 + lattice_size_SU3);
            
            // Cleanup
            cudaFree(d_energies_SU2);
            cudaFree(d_energies_SU3);
            
            // Account for double counting
            return energy_SU2 / 2.0 + energy_SU3 / 2.0;
        }
        
        __host__
        double energy_density_cuda() {
            return total_energy_cuda() / (lattice_size_SU2 + lattice_size_SU3);
        }
        
        // CUDA kernel for computing local fields
        __global__
        void compute_local_field_SU2_kernel(
            double* d_local_fields,
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
            size_t max_mixed_tri_neighbors,
            size_t lattice_size_SU2
        ) {
            int site_index = blockIdx.x * blockDim.x + threadIdx.x;
            if (site_index >= lattice_size_SU2) return;
            
            double local_field[N_SU2];
            
            // Onsite contribution
            multiply_matrix_vector_device(local_field, &onsite_interaction_SU2[site_index * N_SU2 * N_SU2], 
                                        &spins_SU2[site_index * N_SU2], N_SU2);
            
            // Bilinear contributions
            for (size_t i = 0; i < num_bi_SU2 && i < max_bi_neighbors; ++i) {
                size_t partner = bilinear_partners_SU2[site_index * max_bi_neighbors + i];
                if (partner < lattice_size_SU2) {
                    double temp[N_SU2];
                    multiply_matrix_vector_device(temp, 
                        &bilinear_interaction_SU2[site_index * max_bi_neighbors * N_SU2 * N_SU2 + i * N_SU2 * N_SU2],
                        &spins_SU2[partner * N_SU2], N_SU2);
                    for (size_t j = 0; j < N_SU2; ++j) {
                        local_field[j] += temp[j];
                    }
                }
            }
            
            // Trilinear contributions
            for (size_t i = 0; i < num_tri_SU2 && i < max_tri_neighbors; ++i) {
                size_t partner1 = trilinear_partners_SU2[site_index * max_tri_neighbors * 2 + i * 2];
                size_t partner2 = trilinear_partners_SU2[site_index * max_tri_neighbors * 2 + i * 2 + 1];
                if (partner1 < lattice_size_SU2 && partner2 < lattice_size_SU2) {
                    double temp[N_SU2];
                    contract_trilinear_field_device(temp,
                        &trilinear_interaction_SU2[site_index * max_tri_neighbors * N_SU2 * N_SU2 * N_SU2 + i * N_SU2 * N_SU2 * N_SU2],
                        &spins_SU2[partner1 * N_SU2], &spins_SU2[partner2 * N_SU2], N_SU2);
                    for (size_t j = 0; j < N_SU2; ++j) {
                        local_field[j] += temp[j];
                    }
                }
            }
            
            // Mixed trilinear contributions
            for (size_t i = 0; i < num_tri_SU2_SU3 && i < max_mixed_tri_neighbors; ++i) {
                size_t partner1 = mixed_trilinear_partners_SU2[site_index * max_mixed_tri_neighbors * 2 + i * 2];
                size_t partner2 = mixed_trilinear_partners_SU2[site_index * max_mixed_tri_neighbors * 2 + i * 2 + 1];
                if (partner1 < lattice_size_SU2) {
                    for (size_t a = 0; a < N_SU2; ++a) {
                        double temp = 0.0;
                        for (size_t b = 0; b < N_SU2; ++b) {
                            for (size_t c = 0; c < N_SU3; ++c) {
                                temp += mixed_trilinear_interaction_SU2[
                                    site_index * max_mixed_tri_neighbors * N_SU2 * N_SU2 * N_SU3 + 
                                    i * N_SU2 * N_SU2 * N_SU3 + a * N_SU2 * N_SU3 + b * N_SU3 + c] *
                                    spins_SU2[partner1 * N_SU2 + b] * spins_SU3[partner2 * N_SU3 + c];
                            }
                        }
                        local_field[a] += temp;
                    }
                }
            }
            
            // Subtract external field
            for (size_t i = 0; i < N_SU2; ++i) {
                d_local_fields[site_index * N_SU2 + i] = local_field[i] - field_SU2[site_index * N_SU2 + i];
            }
        }

        __global__
        void compute_local_field_SU3_kernel(
            double* d_local_fields,
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
            size_t max_mixed_tri_neighbors,
            size_t lattice_size_SU3
        ) {
            int site_index = blockIdx.x * blockDim.x + threadIdx.x;
            if (site_index >= lattice_size_SU3) return;
            
            double local_field[N_SU3];
            
            // Onsite contribution
            multiply_matrix_vector_device(local_field, &onsite_interaction_SU3[site_index * N_SU3 * N_SU3], 
                                        &spins_SU3[site_index * N_SU3], N_SU3);
            
            // Bilinear contributions
            for (size_t i = 0; i < num_bi_SU3 && i < max_bi_neighbors; ++i) {
                size_t partner = bilinear_partners_SU3[site_index * max_bi_neighbors + i];
                if (partner < lattice_size_SU3) {
                    double temp[N_SU3];
                    multiply_matrix_vector_device(temp, 
                        &bilinear_interaction_SU3[site_index * max_bi_neighbors * N_SU3 * N_SU3 + i * N_SU3 * N_SU3],
                        &spins_SU3[partner * N_SU3], N_SU3);
                    for (size_t j = 0; j < N_SU3; ++j) {
                        local_field[j] += temp[j];
                    }
                }
            }
            
            // Trilinear contributions
            for (size_t i = 0; i < num_tri_SU3 && i < max_tri_neighbors; ++i) {
                size_t partner1 = trilinear_partners_SU3[site_index * max_tri_neighbors * 2 + i * 2];
                size_t partner2 = trilinear_partners_SU3[site_index * max_tri_neighbors * 2 + i * 2 + 1];
                if (partner1 < lattice_size_SU3 && partner2 < lattice_size_SU3) {
                    double temp[N_SU3];
                    contract_trilinear_field_device(temp,
                        &trilinear_interaction_SU3[site_index * max_tri_neighbors * N_SU3 * N_SU3 * N_SU3 + i * N_SU3 * N_SU3 * N_SU3],
                        &spins_SU3[partner1 * N_SU3], &spins_SU3[partner2 * N_SU3], N_SU3);
                    for (size_t j = 0; j < N_SU3; ++j) {
                        local_field[j] += temp[j];
                    }
                }
            }
            // Mixed trilinear contributions
            for (size_t i = 0; i < num_tri_SU2_SU3 && i < max_mixed_tri_neighbors; ++i) {
                size_t partner1 = mixed_trilinear_partners_SU3[site_index * max_mixed_tri_neighbors * 2 + i * 2];
                size_t partner2 = mixed_trilinear_partners_SU3[site_index * max_mixed_tri_neighbors * 2 + i * 2 + 1];
                if (partner1 < lattice_size_SU2 && partner2 < lattice_size_SU2) {
                    for (size_t a = 0; a < N_SU3; ++a) {
                        double temp = 0.0;
                        for (size_t b = 0; b < N_SU2; ++b) {
                            for (size_t c = 0; c < N_SU2; ++c) {
                                temp += mixed_trilinear_interaction_SU3[
                                    site_index * max_mixed_tri_neighbors * N_SU3 * N_SU2 * N_SU2 + 
                                    i * N_SU3 * N_SU2 * N_SU2 + a * N_SU2 * N_SU2 + b * N_SU2 + c] *
                                    spins_SU2[partner1 * N_SU2 + b] * spins_SU2[partner2 * N_SU2 + c];
                            }
                        }
                        local_field[a] += temp;
                    }
                }
            }

            // Subtract external field
            for (size_t i = 0; i < N_SU3; ++i) {
                d_local_fields[site_index * N_SU3 + i] = local_field[i] - field_SU3[site_index * N_SU3 + i];
            }

        }
        
        __device__
        void drive_field_T_SU2_kernel(double currT, double* curr_field){

            // Compute the driving field at time currT for SU2
            int ind = blockIdx.x * blockDim.x + threadIdx.x;
            if (ind >= lattice_size_SU2) return;

            double factor1_SU2 = double(d_field_drive_amp_SU2*exp(-pow((currT-d_t_B_1_SU2)/(2*d_field_drive_width_SU2),2))*cos(2*M_PI*d_field_drive_freq_SU2*(currT-d_t_B_1_SU2)));
            double factor2_SU2 = double(d_field_drive_amp_SU2*exp(-pow((currT-d_t_B_2_SU2)/(2*d_field_drive_width_SU2),2))*cos(2*M_PI*d_field_drive_freq_SU2*(currT-d_t_B_2_SU2)));

            double temp_field[N_SU2];
            for (size_t i = 0; i < N_SU2; ++i) {
                curr_field[ind * N_SU2 + i] = d_field_drive_1_SU2[i] * factor1_SU2 + d_field_drive_2_SU2[i] * factor2_SU2;
            }
        }

        __device__
        void drive_field_T_SU3_kernel(double currT, double* curr_field){

            // Compute the driving field at time currT for SU3
            int ind = blockIdx.x * blockDim.x + threadIdx.x;
            if (ind >= lattice_size_SU3) return;

            double factor1_SU3 = double(d_field_drive_amp_SU3*exp(-pow((currT-d_t_B_1_SU3)/(2*d_field_drive_width_SU3),2))*cos(2*M_PI*d_field_drive_freq_SU3*(currT-d_t_B_1_SU3)));
            double factor2_SU3 = double(d_field_drive_amp_SU3*exp(-pow((currT-d_t_B_2_SU3)/(2*d_field_drive_width_SU3),2))*cos(2*M_PI*d_field_drive_freq_SU3*(currT-d_t_B_2_SU3)));

            double temp_field[N_SU3];

            for (size_t i = 0; i < N_SU3; ++i) {
                curr_field[ind * N_SU3 + i] = d_field_drive_1_SU3[i] * factor1_SU3 + d_field_drive_2_SU3[i] * factor2_SU3;
            }

        }

        __device__
        void landau_Lifshitz_SU2_kernel(double* spins, const double* local_field, double dt) {
            int ind = blockIdx.x * blockDim.x + threadIdx.x;
            if (ind >= lattice_size_SU2) return;

            double spin[N_SU2];
            for (size_t i = 0; i < N_SU2; ++i) {
                spin[i] = spins[ind * N_SU2 + i];
            }

            // Compute the cross product with the local field
            double cross[N_SU2];
            cross_product_SU2_device(cross, spin, local_field + ind * N_SU2);

            // Update the spins using the Landau-Lifshitz equation
            for (size_t i = 0; i < N_SU2; ++i) {
                spins[ind * N_SU2 + i] += dt * cross[i];
            }
        }

        __device__
        void landau_Lifshitz_SU3_kernel(double* spins, const double* local_field, double dt) {
            int ind = blockIdx.x * blockDim.x + threadIdx.x;
            if (ind >= lattice_size_SU3) return;

            double spin[N_SU3];
            for (size_t i = 0; i < N_SU3; ++i) {
                spin[i] = spins[ind * N_SU3 + i];
            }

            // Compute the cross product with the local field
            double cross[N_SU3];
            cross_product_SU3_device(cross, spin, local_field + ind * N_SU3);

            // Update the spins using the Landau-Lifshitz equation
            for (size_t i = 0; i < N_SU3; ++i) {
                spins[ind * N_SU3 + i] += dt * cross[i];
            }
        }

    __global__
    void SSPRK53_step_kernel(
        double* spins_SU2,
        double* spins_SU3,
        const double* field_SU2,
        const double* field_SU3,
        const double* onsite_interaction_SU2,
        const double* onsite_interaction_SU3,
        const double* bilinear_interaction_SU2,
        const size_t* bilinear_partners_SU2,
        const double* bilinear_interaction_SU3,
        const size_t* bilinear_partners_SU3,
        const double* trilinear_interaction_SU2,
        const size_t* trilinear_partners_SU2,
        const double* trilinear_interaction_SU3,
        const size_t* trilinear_partners_SU3,
        const double* mixed_trilinear_interaction_SU2,
        const size_t* mixed_trilinear_partners_SU2,
        const double* mixed_trilinear_interaction_SU3,
        const size_t* mixed_trilinear_partners_SU3,
        double* temp_spins_SU2_1,
        double* temp_spins_SU2_2,
        double* temp_spins_SU3_1,
        double* temp_spins_SU3_2,
        double* local_fields_SU2,
        double* local_fields_SU3,
        double* drive_field_SU2,
        double* drive_field_SU3,
        size_t num_bi_SU2,
        size_t num_tri_SU2,
        size_t num_bi_SU3,
        size_t num_tri_SU3,
        size_t num_tri_SU2_SU3,
        size_t max_bi_neighbors_SU2,
        size_t max_tri_neighbors_SU2,
        size_t max_mixed_tri_neighbors_SU2,
        size_t max_bi_neighbors_SU3,
        size_t max_tri_neighbors_SU3,
        size_t max_mixed_tri_neighbors_SU3,
        size_t lattice_size_SU2,
        size_t lattice_size_SU3,
        double step_size,
        double curr_time,
        int stage
    ) {
        // SSPRK53 coefficients
        const double alpha[5] = {0.0, 0.355909775063327, 0.644090224936674, 0.0, 0.0};
        const double beta[5] = {0.0, 0.0, 0.0, 0.368410593050371, 0.631589406949629};
        const double gamma[5] = {0.0, 0.0, 0.136670099875755, 0.0, 0.218866822769240};
        const double c[5] = {0.0, 0.377268915331368, 0.420351733105640, 0.413514300428344, 1.0};
        
        // Get thread indices
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (stage == 0) {
            // Stage 1: u^(1) = u^n + dt * L(u^n)
            // Just copy initial state to temp_1
            if (tid < lattice_size_SU2) {
                for (int i = 0; i < N_SU2; ++i) {
                    temp_spins_SU2_1[tid * N_SU2 + i] = spins_SU2[tid * N_SU2 + i];
                }
            } else if (tid - lattice_size_SU2 < lattice_size_SU3) {
                int sid = tid - lattice_size_SU2;
                for (int i = 0; i < N_SU3; ++i) {
                    temp_spins_SU3_1[sid * N_SU3 + i] = spins_SU3[sid * N_SU3 + i];
                }
            }
        }
        else if (stage == 1) {
            // Compute local fields and apply Landau-Lifshitz
            if (tid < lattice_size_SU2) {
                compute_local_field_SU2_kernel(local_fields_SU2, spins_SU2, field_SU2, onsite_interaction_SU2,
                    bilinear_interaction_SU2, bilinear_partners_SU2, trilinear_interaction_SU2, trilinear_partners_SU2,
                    mixed_trilinear_interaction_SU2, mixed_trilinear_partners_SU2, spins_SU3,
                    num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3, max_bi_neighbors_SU2, max_tri_neighbors_SU2,
                    max_mixed_tri_neighbors_SU2, lattice_size_SU2);
                
                // Add drive field
                drive_field_T_SU2_kernel(curr_time + c[0] * step_size, drive_field_SU2);
                for (int i = 0; i < N_SU2; ++i) {
                    local_fields_SU2[tid * N_SU2 + i] += drive_field_SU2[tid * N_SU2 + i];
                }
                
                // Update spins: u^(1) = u^n + dt * L(u^n)
                landau_Lifshitz_SU2_kernel(temp_spins_SU2_1, local_fields_SU2, step_size);
            } else if (tid - lattice_size_SU2 < lattice_size_SU3) {
                int sid = tid - lattice_size_SU2;
                compute_local_field_SU3_kernel(local_fields_SU3, spins_SU3, field_SU3, onsite_interaction_SU3,
                    bilinear_interaction_SU3, bilinear_partners_SU3, trilinear_interaction_SU3, trilinear_partners_SU3,
                    mixed_trilinear_interaction_SU3, mixed_trilinear_partners_SU3, spins_SU2,
                    num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3, max_bi_neighbors_SU3, max_tri_neighbors_SU3,
                    max_mixed_tri_neighbors_SU3, lattice_size_SU3);
                
                // Add drive field
                drive_field_T_SU3_kernel(curr_time + c[0] * step_size, drive_field_SU3);
                for (int i = 0; i < N_SU3; ++i) {
                    local_fields_SU3[sid * N_SU3 + i] += drive_field_SU3[sid * N_SU3 + i];
                }
                
                landau_Lifshitz_SU3_kernel(temp_spins_SU3_1, local_fields_SU3, step_size);
            }
        }
        else if (stage == 2) {
            // Stage 2: u^(2) = alpha[1]*u^n + alpha[2]*u^(1) + beta[2]*dt*L(u^(1)) + gamma[2]*dt*L(u^n)
            if (tid < lattice_size_SU2) {
                // Compute L(u^(1))
                compute_local_field_SU2_kernel(local_fields_SU2, temp_spins_SU2_1, field_SU2, onsite_interaction_SU2,
                    bilinear_interaction_SU2, bilinear_partners_SU2, trilinear_interaction_SU2, trilinear_partners_SU2,
                    mixed_trilinear_interaction_SU2, mixed_trilinear_partners_SU2, temp_spins_SU3_1,
                    num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3, max_bi_neighbors_SU2, max_tri_neighbors_SU2,
                    max_mixed_tri_neighbors_SU2, lattice_size_SU2);
                
                drive_field_T_SU2_kernel(curr_time + c[1] * step_size, drive_field_SU2);
                for (int i = 0; i < N_SU2; ++i) {
                    local_fields_SU2[tid * N_SU2 + i] += drive_field_SU2[tid * N_SU2 + i];
                }
                
                // Compute cross product for L(u^(1))
                double cross1[N_SU2];
                cross_product_SU2_device(cross1, &temp_spins_SU2_1[tid * N_SU2], &local_fields_SU2[tid * N_SU2]);
                
                // Compute L(u^n) - recompute local field for original spins
                compute_local_field_SU2_kernel(local_fields_SU2, spins_SU2, field_SU2, onsite_interaction_SU2,
                    bilinear_interaction_SU2, bilinear_partners_SU2, trilinear_interaction_SU2, trilinear_partners_SU2,
                    mixed_trilinear_interaction_SU2, mixed_trilinear_partners_SU2, spins_SU3,
                    num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3, max_bi_neighbors_SU2, max_tri_neighbors_SU2,
                    max_mixed_tri_neighbors_SU2, lattice_size_SU2);
                
                drive_field_T_SU2_kernel(curr_time + c[0] * step_size, drive_field_SU2);
                for (int i = 0; i < N_SU2; ++i) {
                    local_fields_SU2[tid * N_SU2 + i] += drive_field_SU2[tid * N_SU2 + i];
                }
                
                double cross_n[N_SU2];
                cross_product_SU2_device(cross_n, &spins_SU2[tid * N_SU2], &local_fields_SU2[tid * N_SU2]);
                
                // Combine: u^(2) = alpha[1]*u^n + alpha[2]*u^(1) + beta[2]*dt*L(u^(1)) + gamma[2]*dt*L(u^n)
                for (int i = 0; i < N_SU2; ++i) {
                    temp_spins_SU2_2[tid * N_SU2 + i] = alpha[1] * spins_SU2[tid * N_SU2 + i] + 
                                                         alpha[2] * temp_spins_SU2_1[tid * N_SU2 + i] + 
                                                         step_size * gamma[2] * cross_n[i];
                }
            } else if (tid - lattice_size_SU2 < lattice_size_SU3) {
                int sid = tid - lattice_size_SU2;
                // Similar for SU3
                compute_local_field_SU3_kernel(local_fields_SU3, temp_spins_SU3_1, field_SU3, onsite_interaction_SU3,
                    bilinear_interaction_SU3, bilinear_partners_SU3, trilinear_interaction_SU3, trilinear_partners_SU3,
                    mixed_trilinear_interaction_SU3, mixed_trilinear_partners_SU3, temp_spins_SU2_1,
                    num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3, max_bi_neighbors_SU3, max_tri_neighbors_SU3,
                    max_mixed_tri_neighbors_SU3, lattice_size_SU3);
                
                drive_field_T_SU3_kernel(curr_time + c[1] * step_size, drive_field_SU3);
                for (int i = 0; i < N_SU3; ++i) {
                    local_fields_SU3[sid * N_SU3 + i] += drive_field_SU3[sid * N_SU3 + i];
                }
                
                double cross1[N_SU3];
                cross_product_SU3_device(cross1, &temp_spins_SU3_1[sid * N_SU3], &local_fields_SU3[sid * N_SU3]);
                
                compute_local_field_SU3_kernel(local_fields_SU3, spins_SU3, field_SU3, onsite_interaction_SU3,
                    bilinear_interaction_SU3, bilinear_partners_SU3, trilinear_interaction_SU3, trilinear_partners_SU3,
                    mixed_trilinear_interaction_SU3, mixed_trilinear_partners_SU3, spins_SU2,
                    num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3, max_bi_neighbors_SU3, max_tri_neighbors_SU3,
                    max_mixed_tri_neighbors_SU3, lattice_size_SU3);
                
                drive_field_T_SU3_kernel(curr_time + c[0] * step_size, drive_field_SU3);
                for (int i = 0; i < N_SU3; ++i) {
                    local_fields_SU3[sid * N_SU3 + i] += drive_field_SU3[sid * N_SU3 + i];
                }
                
                double cross_n[N_SU3];
                cross_product_SU3_device(cross_n, &spins_SU3[sid * N_SU3], &local_fields_SU3[sid * N_SU3]);
                
                for (int i = 0; i < N_SU3; ++i) {
                    temp_spins_SU3_2[sid * N_SU3 + i] = alpha[1] * spins_SU3[sid * N_SU3 + i] + 
                                                         alpha[2] * temp_spins_SU3_1[sid * N_SU3 + i] + 
                                                         step_size * gamma[2] * cross_n[i];
                }
            }
        }
        else if (stage == 3) {
            // Stage 3: u^(3) = alpha[3]*u^(1) + beta[3]*dt*L(u^(2)) + gamma[3]*dt*L(u^n)
            if (tid < lattice_size_SU2) {
                // Compute L(u^(2))
                compute_local_field_SU2_kernel(local_fields_SU2, temp_spins_SU2_2, field_SU2, onsite_interaction_SU2,
                    bilinear_interaction_SU2, bilinear_partners_SU2, trilinear_interaction_SU2, trilinear_partners_SU2,
                    mixed_trilinear_interaction_SU2, mixed_trilinear_partners_SU2, temp_spins_SU3_2,
                    num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3, max_bi_neighbors_SU2, max_tri_neighbors_SU2,
                    max_mixed_tri_neighbors_SU2, lattice_size_SU2);
                
                drive_field_T_SU2_kernel(curr_time + c[2] * step_size, drive_field_SU2);
                for (int i = 0; i < N_SU2; ++i) {
                    local_fields_SU2[tid * N_SU2 + i] += drive_field_SU2[tid * N_SU2 + i];
                }
                
                double cross2[N_SU2];
                cross_product_SU2_device(cross2, &temp_spins_SU2_2[tid * N_SU2], &local_fields_SU2[tid * N_SU2]);
                
                // Update temp_spins_1 to store u^(3)
                for (int i = 0; i < N_SU2; ++i) {
                    temp_spins_SU2_1[tid * N_SU2 + i] = temp_spins_SU2_1[tid * N_SU2 + i] + 
                                                         step_size * beta[3] * cross2[i];
                }
            } else if (tid - lattice_size_SU2 < lattice_size_SU3) {
                int sid = tid - lattice_size_SU2;
                compute_local_field_SU3_kernel(local_fields_SU3, temp_spins_SU3_2, field_SU3, onsite_interaction_SU3,
                    bilinear_interaction_SU3, bilinear_partners_SU3, trilinear_interaction_SU3, trilinear_partners_SU3,
                    mixed_trilinear_interaction_SU3, mixed_trilinear_partners_SU3, temp_spins_SU2_2,
                    num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3, max_bi_neighbors_SU3, max_tri_neighbors_SU3,
                    max_mixed_tri_neighbors_SU3, lattice_size_SU3);
                
                drive_field_T_SU3_kernel(curr_time + c[2] * step_size, drive_field_SU3);
                for (int i = 0; i < N_SU3; ++i) {
                    local_fields_SU3[sid * N_SU3 + i] += drive_field_SU3[sid * N_SU3 + i];
                }
                
                double cross2[N_SU3];
                cross_product_SU3_device(cross2, &temp_spins_SU3_2[sid * N_SU3], &local_fields_SU3[sid * N_SU3]);
                
                for (int i = 0; i < N_SU3; ++i) {
                    temp_spins_SU3_1[sid * N_SU3 + i] = temp_spins_SU3_1[sid * N_SU3 + i] + 
                                                         step_size * beta[3] * cross2[i];
                }
            }
        }
        else if (stage == 4) {
            // Stage 4: u^(n+1) = alpha[4]*u^(2) + beta[4]*dt*L(u^(3)) + gamma[4]*dt*L(u^n)
            if (tid < lattice_size_SU2) {
                // Compute L(u^(3))
                compute_local_field_SU2_kernel(local_fields_SU2, temp_spins_SU2_1, field_SU2, onsite_interaction_SU2,
                    bilinear_interaction_SU2, bilinear_partners_SU2, trilinear_interaction_SU2, trilinear_partners_SU2,
                    mixed_trilinear_interaction_SU2, mixed_trilinear_partners_SU2, temp_spins_SU3_1,
                    num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3, max_bi_neighbors_SU2, max_tri_neighbors_SU2,
                    max_mixed_tri_neighbors_SU2, lattice_size_SU2);
                
                drive_field_T_SU2_kernel(curr_time + c[3] * step_size, drive_field_SU2);
                for (int i = 0; i < N_SU2; ++i) {
                    local_fields_SU2[tid * N_SU2 + i] += drive_field_SU2[tid * N_SU2 + i];
                }
                
                double cross3[N_SU2];
                cross_product_SU2_device(cross3, &temp_spins_SU2_1[tid * N_SU2], &local_fields_SU2[tid * N_SU2]);
                
                // Compute L(u^n) again
                compute_local_field_SU2_kernel(local_fields_SU2, spins_SU2, field_SU2, onsite_interaction_SU2,
                    bilinear_interaction_SU2, bilinear_partners_SU2, trilinear_interaction_SU2, trilinear_partners_SU2,
                    mixed_trilinear_interaction_SU2, mixed_trilinear_partners_SU2, spins_SU3,
                    num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3, max_bi_neighbors_SU2, max_tri_neighbors_SU2,
                    max_mixed_tri_neighbors_SU2, lattice_size_SU2);
                
                drive_field_T_SU2_kernel(curr_time + c[0] * step_size, drive_field_SU2);
                for (int i = 0; i < N_SU2; ++i) {
                    local_fields_SU2[tid * N_SU2 + i] += drive_field_SU2[tid * N_SU2 + i];
                }
                
                double cross_n[N_SU2];
                cross_product_SU2_device(cross_n, &spins_SU2[tid * N_SU2], &local_fields_SU2[tid * N_SU2]);
                
                // Final update
                for (int i = 0; i < N_SU2; ++i) {
                    spins_SU2[tid * N_SU2 + i] = temp_spins_SU2_2[tid * N_SU2 + i] + 
                                                  step_size * (beta[4] * cross3[i] + gamma[4] * cross_n[i]);
                }
            } else if (tid - lattice_size_SU2 < lattice_size_SU3) {
                int sid = tid - lattice_size_SU2;
                compute_local_field_SU3_kernel(local_fields_SU3, temp_spins_SU3_1, field_SU3, onsite_interaction_SU3,
                    bilinear_interaction_SU3, bilinear_partners_SU3, trilinear_interaction_SU3, trilinear_partners_SU3,
                    mixed_trilinear_interaction_SU3, mixed_trilinear_partners_SU3, temp_spins_SU2_1,
                    num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3, max_bi_neighbors_SU3, max_tri_neighbors_SU3,
                    max_mixed_tri_neighbors_SU3, lattice_size_SU3);
                
                drive_field_T_SU3_kernel(curr_time + c[3] * step_size, drive_field_SU3);
                for (int i = 0; i < N_SU3; ++i) {
                    local_fields_SU3[sid * N_SU3 + i] += drive_field_SU3[sid * N_SU3 + i];
                }
                
                double cross3[N_SU3];
                cross_product_SU3_device(cross3, &temp_spins_SU3_1[sid * N_SU3], &local_fields_SU3[sid * N_SU3]);
                
                compute_local_field_SU3_kernel(local_fields_SU3, spins_SU3, field_SU3, onsite_interaction_SU3,
                    bilinear_interaction_SU3, bilinear_partners_SU3, trilinear_interaction_SU3, trilinear_partners_SU3,
                    mixed_trilinear_interaction_SU3, mixed_trilinear_partners_SU3, spins_SU2,
                    num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3, max_bi_neighbors_SU3, max_tri_neighbors_SU3,
                    max_mixed_tri_neighbors_SU3, lattice_size_SU3);
                
                drive_field_T_SU3_kernel(curr_time + c[0] * step_size, drive_field_SU3);
                for (int i = 0; i < N_SU3; ++i) {
                    local_fields_SU3[sid * N_SU3 + i] += drive_field_SU3[sid * N_SU3 + i];
                }
                
                double cross_n[N_SU3];
                cross_product_SU3_device(cross_n, &spins_SU3[sid * N_SU3], &local_fields_SU3[sid * N_SU3]);
                
                for (int i = 0; i < N_SU3; ++i) {
                    spins_SU3[sid * N_SU3 + i] = temp_spins_SU3_2[sid * N_SU3 + i] + 
                                                  step_size * (beta[4] * cross3[i] + gamma[4] * cross_n[i]);
                }
            }
        }
    }

    // Host wrapper function for SSPRK53 time step
    __host__
    void SSPRK53_step_cuda(double step_size, double curr_time, double tol) {
        // Allocate temporary arrays for intermediate stages
        double *d_temp_spins_SU2_1, *d_temp_spins_SU2_2;
        double *d_temp_spins_SU3_1, *d_temp_spins_SU3_2;
        double *d_local_fields_SU2, *d_local_fields_SU3;
        double *d_drive_field_SU2, *d_drive_field_SU3;
        
        cudaMalloc(&d_temp_spins_SU2_1, lattice_size_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&d_temp_spins_SU2_2, lattice_size_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&d_temp_spins_SU3_1, lattice_size_SU3 * N_SU3 * sizeof(double));
        cudaMalloc(&d_temp_spins_SU3_2, lattice_size_SU3 * N_SU3 * sizeof(double));
        cudaMalloc(&d_local_fields_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&d_local_fields_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
        cudaMalloc(&d_drive_field_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&d_drive_field_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
        
        // Configure kernel launch parameters
        dim3 block(256);
        dim3 grid((lattice_size_SU2 + lattice_size_SU3 + block.x - 1) / block.x);
        
        // Execute SSPRK53 stages
        for (int stage = 0; stage < 5; ++stage) {
            SSPRK53_step_kernel<<<grid, block>>>(
                d_spins.spins_SU2, d_spins.spins_SU3,
                d_field_SU2, d_field_SU3,
                d_onsite_interaction_SU2, d_onsite_interaction_SU3,
                d_bilinear_interaction_SU2, d_bilinear_partners_SU2,
                d_bilinear_interaction_SU3, d_bilinear_partners_SU3,
                d_trilinear_interaction_SU2, d_trilinear_partners_SU2,
                d_trilinear_interaction_SU3, d_trilinear_partners_SU3,
                d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2,
                d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3,
                d_temp_spins_SU2_1, d_temp_spins_SU2_2,
                d_temp_spins_SU3_1, d_temp_spins_SU3_2,
                d_local_fields_SU2, d_local_fields_SU3,
                d_drive_field_SU2, d_drive_field_SU3,
                d_num_bi_SU2, d_num_tri_SU2, d_num_bi_SU3, d_num_tri_SU3, d_num_tri_SU2_SU3,
                max_bilinear_neighbors_SU2, max_trilinear_neighbors_SU2, max_mixed_trilinear_neighbors_SU2,
                max_bilinear_neighbors_SU3, max_trilinear_neighbors_SU3, max_mixed_trilinear_neighbors_SU3,
                lattice_size_SU2, lattice_size_SU3,
                step_size, curr_time, stage
            );
            cudaDeviceSynchronize();
        }
        
        // Cleanup temporary arrays
        cudaFree(d_temp_spins_SU2_1);
        cudaFree(d_temp_spins_SU2_2);
        cudaFree(d_temp_spins_SU3_1);
        cudaFree(d_temp_spins_SU3_2);
        cudaFree(d_local_fields_SU2);
        cudaFree(d_local_fields_SU3);
        cudaFree(d_drive_field_SU2);
        cudaFree(d_drive_field_SU3);
    }

    __host__
    void molecular_dynamics_cuda(double T_start, double T_end, double step_size, string dir_name, 
                                 size_t output_frequency = 100, bool use_adaptive_stepping = false) {
        if (dir_name != "") {
            filesystem::create_directory(dir_name);
            this->write_to_file_spin(dir_name + "/spin_initial");
            this->write_to_file_pos(dir_name + "/spin_pos");
        }
        
        // Copy initial spins to device if not already there
        copy_spins_to_device();
        
        double current_time = T_start;
        size_t step_count = 0;
        
        while (current_time < T_end) {
            // Perform SSPRK53 time step on GPU
            SSPRK53_step_cuda(step_size, current_time, 1e-6);
            
            current_time += step_size;
            step_count++;
            
            // Periodically copy data back to host for output
            if (step_count % output_frequency == 0 && dir_name != "") {
                copy_spins_to_host();
                
                // Write output
                ofstream energy_file(dir_name + "/energy.txt", ios::app);
                energy_file << current_time << " " << this->energy_density(this->spins) << endl;
                energy_file.close();
                
                this->write_to_file(dir_name + "/spin_evolution", this->spins);
            }
        }
        
        // Copy final state back to host
        copy_spins_to_host();
        
        if (dir_name != "") {
            this->write_to_file_spin(dir_name + "/spin_final");
        }
    }
    
    // Compute magnetization M_B(t) for a given external field direction
    __host__
    mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> 
    M_B_t(const array<array<double,N_SU2>, N_ATOMS_SU2> &B_direction_SU2,
          const array<array<double,N_SU3>, N_ATOMS_SU3> &B_direction_SU3) {
        
        // Copy current spins from device to host
        copy_spins_to_host();
        
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> magnetization;
        
        // Calculate magnetization for SU2 sites
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim; ++k) {
                    for (size_t l = 0; l < N_ATOMS_SU2; ++l) {
                        size_t site_index = this->flatten_index(i, j, k, l, N_ATOMS_SU2);
                        // Project spin onto B direction for this sublattice
                        double projection = dot(this->spins.spins_SU2[site_index], B_direction_SU2[l]);
                        magnetization.spins_SU2[site_index] = B_direction_SU2[l] * projection;
                    }
                }
            }
        }
        
        // Calculate magnetization for SU3 sites
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim; ++k) {
                    for (size_t l = 0; l < N_ATOMS_SU3; ++l) {
                        size_t site_index = this->flatten_index(i, j, k, l, N_ATOMS_SU3);
                        // Project spin onto B direction for this sublattice
                        double projection = dot(this->spins.spins_SU3[site_index], B_direction_SU3[l]);
                        magnetization.spins_SU3[site_index] = B_direction_SU3[l] * projection;
                    }
                }
            }
        }
        
        return magnetization;
    }
    
    // Compute magnetizations M_BA(t) and M_BB(t) for two different field directions
    __host__
    tuple<mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim>,
          mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim>>
    M_BA_BB_t(const array<array<double,N_SU2>, N_ATOMS_SU2> &B_A_direction_SU2,
              const array<array<double,N_SU3>, N_ATOMS_SU3> &B_A_direction_SU3,
              const array<array<double,N_SU2>, N_ATOMS_SU2> &B_B_direction_SU2,
              const array<array<double,N_SU3>, N_ATOMS_SU3> &B_B_direction_SU3) {
        
        // Copy current spins from device to host
        copy_spins_to_host();
        
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> magnetization_A;
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> magnetization_B;
        
        // Calculate magnetizations for both directions for SU2 sites
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim; ++k) {
                    for (size_t l = 0; l < N_ATOMS_SU2; ++l) {
                        size_t site_index = this->flatten_index(i, j, k, l, N_ATOMS_SU2);
                        
                        // Project spin onto B_A direction
                        double projection_A = dot(this->spins.spins_SU2[site_index], B_A_direction_SU2[l]);
                        magnetization_A.spins_SU2[site_index] = B_A_direction_SU2[l] * projection_A;
                        
                        // Project spin onto B_B direction
                        double projection_B = dot(this->spins.spins_SU2[site_index], B_B_direction_SU2[l]);
                        magnetization_B.spins_SU2[site_index] = B_B_direction_SU2[l] * projection_B;
                    }
                }
            }
        }
        
        // Calculate magnetizations for both directions for SU3 sites
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim; ++k) {
                    for (size_t l = 0; l < N_ATOMS_SU3; ++l) {
                        size_t site_index = this->flatten_index(i, j, k, l, N_ATOMS_SU3);
                        
                        // Project spin onto B_A direction
                        double projection_A = dot(this->spins.spins_SU3[site_index], B_A_direction_SU3[l]);
                        magnetization_A.spins_SU3[site_index] = B_A_direction_SU3[l] * projection_A;
                        
                        // Project spin onto B_B direction
                        double projection_B = dot(this->spins.spins_SU3[site_index], B_B_direction_SU3[l]);
                        magnetization_B.spins_SU3[site_index] = B_B_direction_SU3[l] * projection_B;
                    }
                }
            }
        }
        
        return make_tuple(magnetization_A, magnetization_B);
    }

public:
    
    // Copy data from device back to host
    __host__
    void copy_spins_to_host() {
        std::vector<double> temp_spins_SU2(lattice_size_SU2 * N_SU2);
        std::vector<double> temp_spins_SU3(lattice_size_SU3 * N_SU3);
        
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

#endif // MIXED_LATTICE_CUDA_CUH