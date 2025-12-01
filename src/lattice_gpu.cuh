#ifndef LATTICE_GPU_CUH
#define LATTICE_GPU_CUH

/**
 * GPU-accelerated lattice simulation header
 * 
 * This file contains:
 * - GPU data structures for spin configurations
 * - Kernel parameter structures
 * - Device function declarations
 * - Host-side GPU lattice class declaration
 * 
 * The implementation is in lattice_gpu.cu
 * 
 * Architecture:
 * - lattice_gpu.cuh: Header with declarations and GPU structures
 * - lattice_gpu.cu: CUDA kernel implementations
 * - lattice.h: CPU Lattice class, includes this header when CUDA_ENABLED
 * 
 * The gpu:: namespace provides the interface between CPU Lattice and GPU kernels.
 */

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
#include <vector>
#include <utility>

// ======================= GPU Data Structures =======================

/**
 * GPU spin configuration storage
 * Flattened arrays for coalesced memory access
 */
struct LatticeSpinsCuda {
    double* spins;           // [lattice_size * spin_dim]
    size_t lattice_size;
    size_t spin_dim;
    
    __host__ __device__
    LatticeSpinsCuda() : spins(nullptr), lattice_size(0), spin_dim(0) {}
    
    __host__
    void allocate(size_t n_sites, size_t n_dim) {
        lattice_size = n_sites;
        spin_dim = n_dim;
        cudaMalloc(&spins, lattice_size * spin_dim * sizeof(double));
        cudaMemset(spins, 0, lattice_size * spin_dim * sizeof(double));
    }
    
    __host__
    void deallocate() {
        if (spins) cudaFree(spins);
        spins = nullptr;
    }
    
    __device__
    double& operator()(size_t site, size_t component) {
        return spins[site * spin_dim + component];
    }
    
    __device__
    const double& operator()(size_t site, size_t component) const {
        return spins[site * spin_dim + component];
    }
};

/**
 * Interaction data for GPU kernels
 */
struct InteractionDataGPU {
    double* field;                      // External field [lattice_size * spin_dim]
    double* onsite_interaction;         // On-site matrix [lattice_size * spin_dim^2]
    double* bilinear_interaction;       // Bilinear coupling [lattice_size * max_neighbors * spin_dim^2]
    size_t* bilinear_partners;          // Partner indices [lattice_size * max_neighbors]
    double* trilinear_interaction;      // Trilinear coupling [lattice_size * max_tri * spin_dim^3]
    size_t* trilinear_partners;         // Partner pairs [lattice_size * max_tri * 2]
    size_t max_bilinear_neighbors;
    size_t max_trilinear_neighbors;
};

/**
 * Field drive parameters for time-dependent simulations
 */
struct FieldDriveParamsGPU {
    double* field_drive_1;      // First pulse component [N_atoms * spin_dim]
    double* field_drive_2;      // Second pulse component [N_atoms * spin_dim]
    double amplitude;
    double width;
    double frequency;
    double t_pulse_1;
    double t_pulse_2;
};

/**
 * Time stepping parameters
 */
struct TimeStepParamsGPU {
    double curr_time;
    double dt;
    double spin_length;
};

/**
 * Neighbor count information
 */
struct NeighborCountsGPU {
    size_t num_bilinear;
    size_t num_trilinear;
    size_t max_bilinear;
    size_t max_trilinear;
};

/**
 * Working arrays for ODE integration (pre-allocated to avoid per-step allocation)
 */
struct WorkingArraysGPU {
    double* work_1;
    double* work_2;
    double* work_3;
    double* local_field;
    size_t array_size;
    bool allocated;
    
    __host__
    WorkingArraysGPU() : work_1(nullptr), work_2(nullptr), work_3(nullptr), 
                         local_field(nullptr), array_size(0), allocated(false) {}
    
    __host__
    void allocate(size_t size) {
        if (allocated) deallocate();
        array_size = size;
        cudaError_t err;
        err = cudaMalloc(&work_1, size * sizeof(double));
        if (err != cudaSuccess) { cleanup_on_error(); return; }
        err = cudaMalloc(&work_2, size * sizeof(double));
        if (err != cudaSuccess) { cleanup_on_error(); return; }
        err = cudaMalloc(&work_3, size * sizeof(double));
        if (err != cudaSuccess) { cleanup_on_error(); return; }
        err = cudaMalloc(&local_field, size * sizeof(double));
        if (err != cudaSuccess) { cleanup_on_error(); return; }
        allocated = true;
    }
    
    __host__
    void deallocate() {
        if (work_1) cudaFree(work_1);
        if (work_2) cudaFree(work_2);
        if (work_3) cudaFree(work_3);
        if (local_field) cudaFree(local_field);
        work_1 = work_2 = work_3 = local_field = nullptr;
        allocated = false;
    }
    
private:
    __host__
    void cleanup_on_error() {
        if (work_1) cudaFree(work_1);
        if (work_2) cudaFree(work_2);
        if (work_3) cudaFree(work_3);
        if (local_field) cudaFree(local_field);
        work_1 = work_2 = work_3 = local_field = nullptr;
        allocated = false;
    }
};

// ======================= Device Function Declarations =======================

/**
 * Dot product of two vectors
 */
__device__
double dot_device(const double* a, const double* b, size_t n);

/**
 * Matrix-vector-vector contraction: spin1^T * matrix * spin2
 */
__device__
double contract_device(const double* spin1, const double* matrix, const double* spin2, size_t n);

/**
 * Trilinear tensor contraction: T_ijk * spin1_i * spin2_j * spin3_k
 */
__device__
double contract_trilinear_device(const double* tensor, const double* spin1, 
                                  const double* spin2, const double* spin3, size_t n);

/**
 * Matrix-vector multiplication: result = matrix * vector
 */
__device__
void multiply_matrix_vector_device(double* result, const double* matrix, const double* vector, size_t n);

/**
 * Trilinear field contraction: result_i = T_ijk * spin1_j * spin2_k
 */
__device__
void contract_trilinear_field_device(double* result, const double* tensor, 
                                      const double* spin1, const double* spin2, size_t n);

/**
 * SU(2) cross product (3D Levi-Civita)
 */
__device__
void cross_product_SU2_device(double* result, const double* a, const double* b);

/**
 * SU(3) cross product using Gell-Mann structure constants
 */
__device__
void cross_product_SU3_device(double* result, const double* a, const double* b);

// ======================= Shared Device Functions for Local Field Computation =======================

/**
 * Initialize local field with external field
 */
__device__
void init_local_field_device(double* local_field, const double* external_field, size_t spin_dim);

/**
 * Add on-site interaction contribution: local_field -= A * spin
 */
__device__
void add_onsite_contribution_device(double* local_field, const double* onsite_matrix, 
                                     const double* spin, double* temp, size_t spin_dim);

/**
 * Add bilinear interaction contribution: local_field -= J * partner_spin
 */
__device__
void add_bilinear_contribution_device(double* local_field, const double* J_matrix,
                                       const double* partner_spin, double* temp, size_t spin_dim);

/**
 * Add trilinear interaction contribution: local_field -= T * spin1 * spin2
 */
__device__
void add_trilinear_contribution_device(double* local_field, const double* T_tensor,
                                        const double* spin1, const double* spin2, 
                                        double* temp, size_t spin_dim);

/**
 * Add drive field contribution to local field
 */
__device__
void add_drive_field_device(double* local_field,
                            const double* field_drive_1, const double* field_drive_2,
                            size_t atom, double amplitude, double width, double frequency,
                            double t_pulse_1, double t_pulse_2, double curr_time,
                            size_t spin_dim);

/**
 * Compute Landau-Lifshitz derivative: dsdt = spin × local_field
 */
__device__
void compute_ll_derivative_device(double* dsdt, const double* spin, 
                                   const double* local_field, size_t spin_dim);

/**
 * Compute local field for a single site (unified function)
 * Computes: H_local = H_ext - A*S - sum_j J_ij*S_j - sum_jk T_ijk*S_j*S_k
 */
__device__
void compute_local_field_device(
    double* local_field,
    const double* d_spins,
    int site,
    const double* field,
    const double* onsite_interaction,
    const double* bilinear_interaction,
    const size_t* bilinear_partners,
    const double* trilinear_interaction,
    const size_t* trilinear_partners,
    size_t num_bilinear,
    size_t max_bilinear,
    size_t num_trilinear,
    size_t max_trilinear,
    size_t lattice_size,
    size_t spin_dim
);

// ======================= Kernel Declarations =======================

/**
 * Compute local field at each site
 */
__global__
void compute_local_field_kernel(
    double* d_local_field,
    const double* d_spins,
    InteractionDataGPU interactions,
    NeighborCountsGPU neighbors,
    size_t lattice_size,
    size_t spin_dim
);

/**
 * Add time-dependent drive field
 */
__global__
void add_drive_field_kernel(
    double* d_local_field,
    FieldDriveParamsGPU field_drive,
    double curr_time,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms
);

/**
 * Compute Landau-Lifshitz derivatives: dS/dt = S × H_eff
 */
__global__
void landau_lifshitz_kernel(
    double* d_dsdt,
    const double* d_spins,
    const double* d_local_field,
    size_t lattice_size,
    size_t spin_dim
);

/**
 * Combined LLG kernel: compute local field and derivatives in one pass
 */
__global__
void LLG_kernel(
    double* d_dsdt,
    const double* d_spins,
    double* d_local_field,
    InteractionDataGPU interactions,
    NeighborCountsGPU neighbors,
    FieldDriveParamsGPU field_drive,
    TimeStepParamsGPU time_params,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms
);

/**
 * Update arrays: out = a1 * in1 + a2 * in2
 */
__global__
void update_arrays_kernel(
    double* out,
    const double* in1, double a1,
    const double* in2, double a2,
    size_t size
);

/**
 * Update arrays (three inputs): out = a1 * in1 + a2 * in2 + a3 * in3
 */
__global__
void update_arrays_three_kernel(
    double* out,
    const double* in1, double a1,
    const double* in2, double a2,
    const double* in3, double a3,
    size_t size
);

/**
 * Normalize spin vectors to specified length
 */
__global__
void normalize_spins_kernel(
    double* d_spins,
    double spin_length,
    size_t lattice_size,
    size_t spin_dim
);

/**
 * Compute site energies
 */
__global__
void compute_site_energy_kernel(
    double* d_energies,
    const double* d_spins,
    InteractionDataGPU interactions,
    NeighborCountsGPU neighbors,
    size_t lattice_size,
    size_t spin_dim
);

/**
 * Compute magnetization per component (with shared memory reduction)
 */
__global__
void compute_magnetization_kernel(
    const double* d_spins,
    double* d_mag_local,
    double* d_mag_staggered,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms
);

// ======================= Host Integration Functions =======================

/**
 * Perform a single Euler step on GPU
 */
void euler_step_gpu(
    double* d_spins,
    InteractionDataGPU& interactions,
    NeighborCountsGPU& neighbors,
    FieldDriveParamsGPU& field_drive,
    TimeStepParamsGPU& time_params,
    WorkingArraysGPU& work_arrays,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms
);

/**
 * Perform a single SSPRK53 step on GPU (5-stage, 3rd order SSP Runge-Kutta)
 * This is highly optimized for spin dynamics.
 */
void SSPRK53_step_gpu(
    double* d_spins,
    InteractionDataGPU& interactions,
    NeighborCountsGPU& neighbors,
    FieldDriveParamsGPU& field_drive,
    TimeStepParamsGPU& time_params,
    WorkingArraysGPU& work_arrays,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms
);

/**
 * Perform a single RK4 step on GPU
 */
void RK4_step_gpu(
    double* d_spins,
    InteractionDataGPU& interactions,
    NeighborCountsGPU& neighbors,
    FieldDriveParamsGPU& field_drive,
    TimeStepParamsGPU& time_params,
    WorkingArraysGPU& work_arrays,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms
);

/**
 * Compute total energy on GPU
 */
double total_energy_gpu(
    const double* d_spins,
    InteractionDataGPU& interactions,
    NeighborCountsGPU& neighbors,
    size_t lattice_size,
    size_t spin_dim
);

/**
 * Compute magnetization on GPU (returns local and staggered)
 */
void compute_magnetization_gpu(
    const double* d_spins,
    double* h_mag_local,
    double* h_mag_staggered,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms
);

// ======================= GPU Lattice Wrapper Class =======================

/**
 * Forward declaration of CPU Lattice class
 */
class Lattice;

/**
 * LatticeGPU: GPU-accelerated version of Lattice
 * 
 * Manages GPU memory and provides GPU-accelerated simulation methods.
 * Can be constructed from a CPU Lattice instance.
 */
class LatticeGPU {
public:
    // Core properties (mirrored from CPU Lattice)
    size_t spin_dim;
    size_t N_atoms;
    size_t dim1, dim2, dim3;
    size_t lattice_size;
    float spin_length;
    
    // GPU data
    LatticeSpinsCuda d_spins;
    InteractionDataGPU d_interactions;
    NeighborCountsGPU d_neighbors;
    FieldDriveParamsGPU d_field_drive;
    WorkingArraysGPU work_arrays;
    
    // Host-side copies for I/O
    std::vector<double> h_spins;
    
    /**
     * Construct from CPU Lattice
     */
    LatticeGPU(const Lattice& cpu_lattice);
    
    /**
     * Destructor - cleanup GPU memory
     */
    ~LatticeGPU();
    
    /**
     * Copy spins from GPU to host buffer
     */
    void copy_spins_to_host();
    
    /**
     * Copy spins from host buffer to GPU
     */
    void copy_spins_to_device();
    
    /**
     * Sync spins with CPU Lattice (both directions)
     */
    void sync_with_cpu(Lattice& cpu_lattice, bool to_gpu);
    
    /**
     * Run molecular dynamics entirely on GPU
     * 
     * @param T_start Start time
     * @param T_end End time
     * @param dt Time step
     * @param method Integration method: "euler", "rk4", "ssprk53"
     * @param out_dir Output directory (empty = no output)
     * @param save_interval Steps between saves
     */
    void molecular_dynamics(
        double T_start, double T_end, double dt,
        const std::string& method = "ssprk53",
        const std::string& out_dir = "",
        size_t save_interval = 100
    );
    
    /**
     * Single integration step
     */
    void step(double dt, double curr_time, const std::string& method = "ssprk53");
    
    /**
     * Compute total energy
     */
    double total_energy();
    
    /**
     * Compute magnetization
     */
    void compute_magnetization(double* mag_local, double* mag_staggered);
    
    /**
     * Normalize all spins to spin_length
     */
    void normalize_spins();

private:
    /**
     * Initialize GPU memory from CPU Lattice data
     */
    void initialize_from_cpu(const Lattice& cpu_lattice);
    
    /**
     * Cleanup GPU memory
     */
    void cleanup();
    
    /**
     * Flatten host interaction data for GPU transfer
     */
    void flatten_interactions(const Lattice& cpu_lattice);
    
    // Flattened host buffers for interactions (used during initialization)
    std::vector<double> h_field;
    std::vector<double> h_onsite;
    std::vector<double> h_bilinear;
    std::vector<size_t> h_bilinear_partners;
    std::vector<double> h_trilinear;
    std::vector<size_t> h_trilinear_partners;
    std::vector<double> h_field_drive_1;
    std::vector<double> h_field_drive_2;
};

// ======================= gpu:: Namespace Interface =======================
// This namespace provides the interface between CPU Lattice class and GPU kernels.
// It is used by lattice.h for GPU-accelerated molecular dynamics.

namespace gpu {

/**
 * GPU state vector type (thrust device vector)
 */
using GPUState = thrust::device_vector<double>;

/**
 * GPU lattice data structure - holds all interaction data on device
 */
struct GPULatticeData {
    // Spin configuration
    thrust::device_vector<double> spins;
    
    // Fields
    thrust::device_vector<double> field;
    thrust::device_vector<double> onsite;
    
    // Bilinear interactions
    thrust::device_vector<double> bilinear_vals;
    thrust::device_vector<size_t> bilinear_idx;
    thrust::device_vector<size_t> bilinear_counts;
    
    // Trilinear interactions (optional)
    thrust::device_vector<double> trilinear_vals;
    thrust::device_vector<size_t> trilinear_idx;
    thrust::device_vector<size_t> trilinear_counts;
    
    // Field drive pulse parameters
    thrust::device_vector<double> field_drive;
    double pulse_amp = 0.0;
    double pulse_width = 1.0;
    double pulse_freq = 0.0;
    double t_pulse_1 = 0.0;
    double t_pulse_2 = 0.0;
    
    // Lattice dimensions
    size_t lattice_size = 0;
    size_t spin_dim = 0;
    size_t N_atoms = 0;
    size_t max_bilinear = 0;
    size_t max_trilinear = 0;
    
    // Working arrays for integration (pre-allocated)
    thrust::device_vector<double> work_1;
    thrust::device_vector<double> work_2;
    thrust::device_vector<double> work_3;
    thrust::device_vector<double> local_field;
    
    bool initialized = false;
};

/**
 * GPU ODE system functor for Landau-Lifshitz integration
 */
struct GPUODESystem {
    GPULatticeData& data;
    
    GPUODESystem(GPULatticeData& d) : data(d) {}
    
    /**
     * Compute dS/dt = S × H_eff on GPU
     * This is called by the integration routine
     */
    void operator()(const GPUState& x, GPUState& dxdt, double t) const;
};

/**
 * Create GPU lattice data from host arrays
 */
GPULatticeData create_gpu_lattice_data(
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms,
    size_t max_bilinear,
    const std::vector<double>& flat_field,
    const std::vector<double>& flat_onsite,
    const std::vector<double>& flat_bilinear,
    const std::vector<size_t>& flat_partners,
    const std::vector<size_t>& num_bilinear_per_site
);

/**
 * Set pulse parameters on GPU
 */
void set_gpu_pulse(
    GPULatticeData& data,
    const std::vector<double>& flat_field_drive,
    double pulse_amp,
    double pulse_width,
    double pulse_freq,
    double t_pulse_1,
    double t_pulse_2
);

/**
 * Perform GPU integration using specified method
 * 
 * Available methods (matching CPU Boost.Odeint options):
 * - "euler": Explicit Euler (1st order, 1 stage)
 * - "rk2" or "midpoint": Modified midpoint (2nd order, 2 stages)
 * - "rk4": Classic Runge-Kutta (4th order, 4 stages)
 * - "rk5" or "rkck54": Cash-Karp 5(4) (5th order, 6 stages)
 * - "dopri5": Dormand-Prince 5(4) (5th order, 7 stages) - recommended
 * - "rk78" or "rkf78": Fehlberg 7(8) (8th order, 13 stages)
 * - "ssprk53": SSP RK 5-stage 3rd order (optimized for spin dynamics) - default
 * - "bulirsch_stoer" or "bs": Modified midpoint extrapolation (high accuracy)
 * 
 * @param system GPU ODE system
 * @param state Device state vector (modified in-place)
 * @param T_start Start time
 * @param T_end End time
 * @param dt Step size
 * @param save_interval Steps between trajectory saves
 * @param trajectory Output: (time, state) pairs saved at intervals
 * @param method Integration method (default: ssprk53)
 */
void integrate_gpu(
    GPUODESystem& system,
    GPUState& state,
    double T_start,
    double T_end,
    double dt,
    size_t save_interval,
    std::vector<std::pair<double, std::vector<double>>>& trajectory,
    const std::string& method = "ssprk53"
);

/**
 * Single integration step on GPU
 * 
 * Available methods (matching CPU options):
 * - "euler": Explicit Euler (1st order)
 * - "rk2" or "midpoint": Modified midpoint (2nd order)
 * - "rk4": Classic Runge-Kutta (4th order)
 * - "rk5" or "rkck54": Cash-Karp RK5 (5th order)
 * - "dopri5": Dormand-Prince 5(4) (5th order)
 * - "rk78" or "rkf78": Fehlberg 7(8) (8th order)
 * - "ssprk53": SSP RK 5-stage 3rd order (optimized for spin dynamics)
 * - "bulirsch_stoer" or "bs": Modified midpoint extrapolation
 * 
 * @param system GPU ODE system
 * @param state Device state vector (modified in-place)
 * @param t Current time
 * @param dt Step size
 * @param method Integration method (default: ssprk53)
 */
void step_gpu(
    GPUODESystem& system,
    GPUState& state,
    double t,
    double dt,
    const std::string& method = "ssprk53"
);

/**
 * Compute total energy on GPU
 */
double compute_energy_gpu(const GPULatticeData& data, const GPUState& state);

/**
 * Normalize spins on GPU
 */
void normalize_spins_gpu(GPUState& state, size_t lattice_size, size_t spin_dim, double spin_length);

} // namespace gpu

#endif // LATTICE_GPU_CUH

