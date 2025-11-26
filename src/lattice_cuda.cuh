#ifndef LATTICE_CUDA_CUH
#define LATTICE_CUDA_CUH

#include "lattice.h"
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

// CUDA device array structures
template<size_t N, size_t lattice_size>
struct lattice_spin_cuda {
    double* spins;  // Flattened array: [lattice_size * N]
    
    __host__ __device__
    lattice_spin_cuda() : spins(nullptr) {}
    
    __host__
    void allocate() {
        cudaMalloc(&spins, lattice_size * N * sizeof(double));
    }
    
    __host__
    void deallocate() {
        if (spins) cudaFree(spins);
        spins = nullptr;
    }
    
    __device__
    double& spin(size_t site, size_t component) {
        return spins[site * N + component];
    }
    
    __device__
    const double& spin(size_t site, size_t component) const {
        return spins[site * N + component];
    }
};

template<size_t lattice_size>
struct lattice_pos_cuda {
    double* pos;  // Flattened array: [lattice_size * 3]
    
    __host__ __device__
    lattice_pos_cuda() : pos(nullptr) {}
    
    __host__
    void allocate() {
        cudaMalloc(&pos, lattice_size * 3 * sizeof(double));
    }
    
    __host__
    void deallocate() {
        if (pos) cudaFree(pos);
        pos = nullptr;
    }
    
    __device__
    double& pos_coord(size_t site, size_t coord) {
        return pos[site * 3 + coord];
    }
};

// Struct definitions for organizing kernel parameters
template<size_t N>
struct InteractionData {
    double* field;
    double* onsite_interaction;
    double* bilinear_interaction;
    size_t* bilinear_partners;
    double* trilinear_interaction;
    size_t* trilinear_partners;
};

struct NeighborCounts {
    size_t num_bi;
    size_t num_tri;
    size_t max_bi_neighbors;
    size_t max_tri_neighbors;
};

template<size_t N, size_t N_ATOMS>
struct FieldDriveParams {
    double* field_drive_1;
    double* field_drive_2;
    double field_drive_amp;
    double field_drive_width;
    double field_drive_freq;
    double t_B_1;
    double t_B_2;
};

struct TimeStepParams {
    double curr_time;
    double dt;
    double spin_length;
};

// Device helper function declarations (static inline to avoid multiple definition errors)
static __device__ double dot_device(const double* a, const double* b, size_t n);
static __device__ double contract_device(const double* spin1, const double* matrix, const double* spin2, size_t n);
static __device__ double contract_trilinear_device(const double* tensor, const double* spin1, const double* spin2, const double* spin3, size_t n);
static __device__ void multiply_matrix_vector_device(double* result, const double* matrix, const double* vector, size_t n);
static __device__ void contract_trilinear_field_device(double* result, const double* tensor, const double* spin1, const double* spin2, size_t n);
static __device__ void cross_product_device(double* result, const double* a, const double* b, size_t n);

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
class lattice_cuda;

// CUDA version of lattice that inherits from the CPU version
template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
class lattice_cuda : public lattice<N, N_ATOMS, dim1, dim2, dim>
{
public:
    static constexpr size_t lattice_size = dim1 * dim2 * dim * N_ATOMS;
    
    // Device data structures
    lattice_spin_cuda<N, lattice_size> d_spins;
    lattice_pos_cuda<lattice_size> d_site_pos;
    
    // Device arrays for lookup tables
    double* d_field;                    // [lattice_size * N]
    double* d_field_drive_1;            // [N_ATOMS * N]
    double* d_field_drive_2;            // [N_ATOMS * N]
    
    double* d_onsite_interaction;       // [lattice_size * N * N]
    
    // Device arrays for interaction parameters (simplified flat arrays)
    double* d_bilinear_interaction;     // [lattice_size * max_neighbors * N * N]
    size_t* d_bilinear_partners;        // [lattice_size * max_neighbors]
    
    double* d_trilinear_interaction;    // [lattice_size * max_tri_neighbors * N^3]
    size_t* d_trilinear_partners;       // [lattice_size * max_tri_neighbors * 2]
    
    // Device constants copied from host
    double d_spin_length;
    double d_field_drive_freq, d_field_drive_amp, d_field_drive_width;
    double d_t_B_1, d_t_B_2;
    
    size_t d_num_bi, d_num_tri;
    
    // Maximum neighbor counts (determined from host data)
    size_t max_bilinear_neighbors;
    size_t max_trilinear_neighbors;

    // Random state for CUDA kernels
    curandState* d_rng_states;
    
    // Constructor - inherits from base class and sets up CUDA data
    __host__
    lattice_cuda(const UnitCell<N, N_ATOMS>* atoms, 
                 float spin_length_in = 1, 
                 bool periodic = true)
        : lattice<N, N_ATOMS, dim1, dim2, dim>(atoms, spin_length_in, periodic),
          d_field(nullptr), d_field_drive_1(nullptr), d_field_drive_2(nullptr),
          d_onsite_interaction(nullptr), d_bilinear_interaction(nullptr),
          d_bilinear_partners(nullptr), d_trilinear_interaction(nullptr),
          d_trilinear_partners(nullptr), d_rng_states(nullptr)
    {
        d_spin_length = spin_length_in;
        d_field_drive_freq = 0.0;
        d_field_drive_amp = 0.0;
        d_field_drive_width = 0.0;
        d_t_B_1 = 0.0;
        d_t_B_2 = 0.0;
        
        // Calculate maximum neighbors
        calculate_max_neighbors();
        
        // Allocate CUDA memory
        allocate_cuda_memory();
        
        // Copy data from host to device
        copy_data_to_device();
        
        std::cout << "CUDA lattice initialized successfully" << std::endl;
        std::cout << "  Lattice size: " << lattice_size << std::endl;
        std::cout << "  N (spin dimension): " << N << std::endl;
        std::cout << "  Max bilinear neighbors: " << max_bilinear_neighbors << std::endl;
        std::cout << "  Max trilinear neighbors: " << max_trilinear_neighbors << std::endl;
    }
    
    // Destructor
    __host__
    ~lattice_cuda() {
        cleanup_cuda_memory();
    }
    
private:
    // Calculate maximum number of neighbors for each interaction type
    __host__
    void calculate_max_neighbors();
    
    // Allocate CUDA memory for all arrays
    __host__
    void allocate_cuda_memory();
    
    // Copy data from host arrays to device arrays
    __host__
    void copy_data_to_device();
    
    // Helper function to copy complex interaction arrays
    __host__
    void copy_interaction_arrays_to_device();
    
    // Cleanup CUDA memory
    __host__
    void cleanup_cuda_memory();

public:

    // Print information about the device lattice and CUDA implementation
    __host__
    void print_lattice_info_cuda(const std::string& filename = "lattice_info_GPU.txt") const;

    // Host wrapper for computing total energy on GPU
    __host__
    double total_energy_cuda();
    
    __host__
    double energy_density_cuda() {
        return total_energy_cuda() / lattice_size;
    }
    
    // Host wrapper function for SSPRK53 time step
    __host__
    void SSPRK53_step_cuda(double step_size, double curr_time, double tol);

    __host__
    void euler_step_cuda(double step_size, double curr_time, double tol);

    __host__
    void get_local_field_cuda(double step_size, double curr_time, double tol);
    
    // GPU magnetization computation (avoids large host-device memory transfers)
    __host__
    void compute_magnetization_cuda(double* d_mag_local, double* d_mag_global);

    __host__
    void molecular_dynamics_cuda(double T_start, double T_end, double step_size, 
                                 std::string dir_name, bool save_all = false, 
                                 size_t save_interval = 100);

    // Copy data from device back to host
    __host__
    void copy_spins_to_host();
    
    // Copy data from host to device
    __host__
    void copy_spins_to_device();
};

// Device function to compute local field at a site
template <size_t N, size_t lattice_size>
__device__
void compute_local_field(
    double* out,
    int site_index,
    const double* spins,
    const InteractionData<N>& interactions,
    const NeighborCounts& neighbors
);

// Device function for Landau-Lifshitz dynamics
template <size_t N, size_t lattice_size>
__device__
void landau_Lifshitz(double* out, int site_index, double* spins, const double* local_field);

// Device function for drive field
template <size_t N, size_t N_ATOMS, size_t lattice_size>
__device__
void drive_field_T(
    double* out, double currT, int site_index,
    double* d_field_drive_1, double* d_field_drive_2, 
    double d_field_drive_amp, double d_field_drive_width, 
    double d_field_drive_freq, double d_t_B_1, double d_t_B_2, 
    size_t max_tri_neighbors, double* trilinear_interaction, 
    size_t* trilinear_partners, double* d_spins
);

// CUDA kernel for computing site energy
template<size_t N, size_t lattice_size>
__global__
void compute_site_energy_kernel(
    double* d_energies,
    const double* spins,
    const double* field,
    const double* onsite_interaction,
    const double* bilinear_interaction,
    const size_t* bilinear_partners,
    const double* trilinear_interaction,
    const size_t* trilinear_partners,
    size_t num_bi,
    size_t num_tri,
    size_t max_bi_neighbors,
    size_t max_tri_neighbors
);

// CUDA kernel for Landau-Lifshitz-Gilbert equation
template<size_t N, size_t N_ATOMS, size_t lattice_size>
__global__
void LLG_kernel(
    double* k, double* d_spins, double* d_local_field,
    InteractionData<N> interactions,
    NeighborCounts neighbors,
    FieldDriveParams<N, N_ATOMS> field_drive,
    TimeStepParams time_params
);

// Working array struct for SSPRK53
struct WorkingArrays {
    double* work_1;
    double* work_2;
    double* work_3;
};

// SSPRK53 step kernel
template <size_t N, size_t N_ATOMS, size_t lattice_size>
__host__
void SSPRK53_step_kernel(
    double* d_spins, double* d_local_field,
    InteractionData<N> interactions,
    NeighborCounts neighbors,
    FieldDriveParams<N, N_ATOMS> field_drive,
    TimeStepParams time_params,
    WorkingArrays work_arrays
);

// Euler step kernel
template<size_t N, size_t N_ATOMS, size_t lattice_size>
__host__
void euler_step_kernel(
    double* d_spins, double* d_local_field,
    InteractionData<N> interactions,
    NeighborCounts neighbors,
    FieldDriveParams<N, N_ATOMS> field_drive,
    TimeStepParams time_params
);

#endif // LATTICE_CUDA_CUH
