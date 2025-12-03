#ifndef MIXED_LATTICE_GPU_CUH
#define MIXED_LATTICE_GPU_CUH

/**
 * GPU-accelerated mixed lattice simulation header
 * 
 * This file contains:
 * - GPU data structures for mixed SU(2)/SU(3) spin configurations
 * - Kernel parameter structures
 * - Device function declarations
 * - Host-side GPU integration functions
 * 
 * The implementation is in mixed_lattice_gpu.cu
 * 
 * Architecture:
 * - mixed_lattice_gpu.cuh: Header with declarations and GPU structures
 * - mixed_lattice_gpu.cu: CUDA kernel implementations
 * - mixed_lattice.h: CPU MixedLattice class, includes this header when CUDA_ENABLED
 * 
 * The mixed_gpu:: namespace provides the interface between CPU MixedLattice and GPU kernels.
 */

#include "lattice_gpu.cuh"  // Include base GPU functions and structures
#include <string>

// ======================= GPU Data Structures for Mixed Lattice =======================

namespace mixed_gpu {

/**
 * GPU state vector type
 */
using GPUState = thrust::device_vector<double>;

/**
 * Interaction data for SU(2) sublattice on GPU
 */
struct InteractionDataSU2GPU {
    double* field;                      // External field [lattice_size_SU2 * spin_dim_SU2]
    double* onsite_interaction;         // On-site matrix [lattice_size_SU2 * spin_dim_SU2^2]
    double* bilinear_interaction;       // Bilinear coupling [total_bilinear * spin_dim_SU2^2]
    size_t* bilinear_partners;          // Partner indices [total_bilinear]
    size_t* bilinear_counts;            // Number of neighbors per site [lattice_size_SU2]
    double* trilinear_interaction;      // Trilinear coupling (optional)
    size_t* trilinear_partners;         // Partner pairs (optional)
    size_t* trilinear_counts;           // Number of trilinear per site
    size_t max_bilinear_neighbors;
    size_t max_trilinear_neighbors;
};

/**
 * Interaction data for SU(3) sublattice on GPU
 */
struct InteractionDataSU3GPU {
    double* field;                      // External field [lattice_size_SU3 * spin_dim_SU3]
    double* onsite_interaction;         // On-site matrix [lattice_size_SU3 * spin_dim_SU3^2]
    double* bilinear_interaction;       // Bilinear coupling [total_bilinear * spin_dim_SU3^2]
    size_t* bilinear_partners;          // Partner indices [total_bilinear]
    size_t* bilinear_counts;            // Number of neighbors per site [lattice_size_SU3]
    double* trilinear_interaction;      // Trilinear coupling (optional)
    size_t* trilinear_partners;         // Partner pairs (optional)
    size_t* trilinear_counts;           // Number of trilinear per site
    size_t max_bilinear_neighbors;
    size_t max_trilinear_neighbors;
};

/**
 * Mixed interaction data (SU2-SU3 coupling) on GPU
 */
struct MixedInteractionDataGPU {
    // Mixed bilinear: J_ab * S2_a * S3_b
    double* bilinear_interaction;       // [total_mixed_bilinear * spin_dim_SU2 * spin_dim_SU3]
    size_t* bilinear_partners_SU2;      // SU(2) site indices
    size_t* bilinear_partners_SU3;      // SU(3) site indices
    size_t* bilinear_counts_SU2;        // Number of mixed neighbors per SU(2) site
    size_t* bilinear_counts_SU3;        // Number of mixed neighbors per SU(3) site
    size_t max_mixed_bilinear;
};

/**
 * Field drive parameters for time-dependent simulations (SU(2))
 */
struct FieldDriveParamsSU2GPU {
    double* field_drive_1;      // First pulse [N_atoms_SU2 * spin_dim_SU2]
    double* field_drive_2;      // Second pulse [N_atoms_SU2 * spin_dim_SU2]
    double amplitude;
    double width;
    double frequency;
    double t_pulse_1;
    double t_pulse_2;
};

/**
 * Field drive parameters for time-dependent simulations (SU(3))
 */
struct FieldDriveParamsSU3GPU {
    double* field_drive_1;      // First pulse [N_atoms_SU3 * spin_dim_SU3]
    double* field_drive_2;      // Second pulse [N_atoms_SU3 * spin_dim_SU3]
    double amplitude;
    double width;
    double frequency;
    double t_pulse_1;
    double t_pulse_2;
};

/**
 * Time stepping parameters for mixed lattice
 */
struct TimeStepParamsMixedGPU {
    double curr_time;
    double dt;
    double spin_length_SU2;
    double spin_length_SU3;
};

/**
 * Lattice dimension parameters for mixed lattice
 */
struct LatticeDimsMixedGPU {
    size_t lattice_size_SU2;
    size_t lattice_size_SU3;
    size_t spin_dim_SU2;
    size_t spin_dim_SU3;
    size_t N_atoms_SU2;
    size_t N_atoms_SU3;
    size_t SU3_offset;          // Offset in combined state vector
};

/**
 * Neighbor counts for mixed lattice
 */
struct NeighborCountsMixedGPU {
    size_t max_bilinear_SU2;
    size_t max_bilinear_SU3;
    size_t max_trilinear_SU2;
    size_t max_trilinear_SU3;
    size_t max_mixed_bilinear;
};

/**
 * Complete GPU data for mixed lattice
 */
struct GPUMixedLatticeData {
    // Spin state on device
    thrust::device_vector<double> spins;  // [lattice_size_SU2 * spin_dim_SU2 + lattice_size_SU3 * spin_dim_SU3]
    
    // SU(2) interactions
    thrust::device_vector<double> field_SU2;
    thrust::device_vector<double> onsite_SU2;
    thrust::device_vector<double> bilinear_vals_SU2;
    thrust::device_vector<size_t> bilinear_idx_SU2;
    thrust::device_vector<size_t> bilinear_counts_SU2;
    thrust::device_vector<double> trilinear_vals_SU2;
    thrust::device_vector<size_t> trilinear_idx_SU2;
    thrust::device_vector<size_t> trilinear_counts_SU2;
    
    // SU(3) interactions
    thrust::device_vector<double> field_SU3;
    thrust::device_vector<double> onsite_SU3;
    thrust::device_vector<double> bilinear_vals_SU3;
    thrust::device_vector<size_t> bilinear_idx_SU3;
    thrust::device_vector<size_t> bilinear_counts_SU3;
    thrust::device_vector<double> trilinear_vals_SU3;
    thrust::device_vector<size_t> trilinear_idx_SU3;
    thrust::device_vector<size_t> trilinear_counts_SU3;
    
    // Mixed SU(2)-SU(3) interactions
    thrust::device_vector<double> mixed_bilinear_vals;
    thrust::device_vector<size_t> mixed_bilinear_idx_SU2;
    thrust::device_vector<size_t> mixed_bilinear_idx_SU3;
    thrust::device_vector<size_t> mixed_bilinear_counts_SU2;
    
    // Field drive (pulses)
    thrust::device_vector<double> field_drive_SU2;  // [2 * N_atoms_SU2 * spin_dim_SU2]
    thrust::device_vector<double> field_drive_SU3;  // [2 * N_atoms_SU3 * spin_dim_SU3]
    double pulse_amp_SU2 = 0.0, pulse_width_SU2 = 1.0, pulse_freq_SU2 = 0.0;
    double t_pulse_1_SU2 = 0.0, t_pulse_2_SU2 = 0.0;
    double pulse_amp_SU3 = 0.0, pulse_width_SU3 = 1.0, pulse_freq_SU3 = 0.0;
    double t_pulse_1_SU3 = 0.0, t_pulse_2_SU3 = 0.0;
    
    // Lattice dimensions
    size_t lattice_size_SU2 = 0;
    size_t lattice_size_SU3 = 0;
    size_t spin_dim_SU2 = 3;    // Default SU(2) = 3 components
    size_t spin_dim_SU3 = 8;    // Default SU(3) = 8 components
    size_t N_atoms_SU2 = 0;
    size_t N_atoms_SU3 = 0;
    size_t max_bilinear_SU2 = 0;
    size_t max_bilinear_SU3 = 0;
    size_t max_trilinear_SU2 = 0;
    size_t max_trilinear_SU3 = 0;
    size_t max_mixed_bilinear = 0;
    
    // Working arrays for integration (pre-allocated)
    thrust::device_vector<double> work_1;
    thrust::device_vector<double> work_2;
    thrust::device_vector<double> work_3;
    thrust::device_vector<double> local_field;
    
    // RK stage storage (pre-allocated for high-order methods)
    thrust::device_vector<double> k1, k2, k3, k4, k5, k6, k7;
    thrust::device_vector<double> tmp_state;
    
    // CUDA streams for concurrent SU2/SU3 kernel execution
    cudaStream_t stream_SU2 = nullptr;
    cudaStream_t stream_SU3 = nullptr;
    bool streams_initialized = false;
    
    bool initialized = false;
    
    /**
     * Initialize CUDA streams for concurrent execution
     */
    void init_streams() {
        if (!streams_initialized) {
            cudaStreamCreate(&stream_SU2);
            cudaStreamCreate(&stream_SU3);
            streams_initialized = true;
        }
    }
    
    /**
     * Destroy CUDA streams
     */
    void destroy_streams() {
        if (streams_initialized) {
            cudaStreamDestroy(stream_SU2);
            cudaStreamDestroy(stream_SU3);
            streams_initialized = false;
            stream_SU2 = nullptr;
            stream_SU3 = nullptr;
        }
    }
    
    /**
     * Get total state size
     */
    size_t state_size() const {
        return lattice_size_SU2 * spin_dim_SU2 + lattice_size_SU3 * spin_dim_SU3;
    }
    
    /**
     * Get offset where SU(3) spins start in the state vector
     */
    size_t SU3_offset() const {
        return lattice_size_SU2 * spin_dim_SU2;
    }
    
    /**
     * Allocate RK working arrays based on method needs
     */
    void ensure_rk_arrays(size_t array_size, int stages) {
        if (k1.size() < array_size) {
            k1.resize(array_size, 0.0);
            k2.resize(array_size, 0.0);
            tmp_state.resize(array_size, 0.0);
        }
        if (stages >= 4 && k3.size() < array_size) {
            k3.resize(array_size, 0.0);
            k4.resize(array_size, 0.0);
        }
        if (stages >= 6 && k5.size() < array_size) {
            k5.resize(array_size, 0.0);
            k6.resize(array_size, 0.0);
        }
        if (stages >= 7 && k7.size() < array_size) {
            k7.resize(array_size, 0.0);
        }
    }
    
    /**
     * Build lattice dimensions struct for kernel calls
     */
    LatticeDimsMixedGPU get_dims() const {
        LatticeDimsMixedGPU dims;
        dims.lattice_size_SU2 = lattice_size_SU2;
        dims.lattice_size_SU3 = lattice_size_SU3;
        dims.spin_dim_SU2 = spin_dim_SU2;
        dims.spin_dim_SU3 = spin_dim_SU3;
        dims.N_atoms_SU2 = N_atoms_SU2;
        dims.N_atoms_SU3 = N_atoms_SU3;
        dims.SU3_offset = SU3_offset();
        return dims;
    }
    
    /**
     * Build neighbor counts struct for kernel calls
     */
    NeighborCountsMixedGPU get_neighbor_counts() const {
        NeighborCountsMixedGPU neighbors;
        neighbors.max_bilinear_SU2 = max_bilinear_SU2;
        neighbors.max_bilinear_SU3 = max_bilinear_SU3;
        neighbors.max_trilinear_SU2 = max_trilinear_SU2;
        neighbors.max_trilinear_SU3 = max_trilinear_SU3;
        neighbors.max_mixed_bilinear = max_mixed_bilinear;
        return neighbors;
    }
    
    /**
     * Build time step params struct for kernel calls
     */
    TimeStepParamsMixedGPU get_time_params(double t, double dt = 0.0,
                                            double spin_len_SU2 = 1.0, 
                                            double spin_len_SU3 = 1.0) const {
        TimeStepParamsMixedGPU time_params;
        time_params.curr_time = t;
        time_params.dt = dt;
        time_params.spin_length_SU2 = spin_len_SU2;
        time_params.spin_length_SU3 = spin_len_SU3;
        return time_params;
    }
    
    /**
     * Build SU(2) interaction data struct for kernel calls
     */
    InteractionDataSU2GPU get_interactions_SU2() const {
        InteractionDataSU2GPU interactions;
        interactions.field = const_cast<double*>(thrust::raw_pointer_cast(field_SU2.data()));
        interactions.onsite_interaction = const_cast<double*>(thrust::raw_pointer_cast(onsite_SU2.data()));
        interactions.bilinear_interaction = const_cast<double*>(thrust::raw_pointer_cast(bilinear_vals_SU2.data()));
        interactions.bilinear_partners = const_cast<size_t*>(thrust::raw_pointer_cast(bilinear_idx_SU2.data()));
        interactions.bilinear_counts = const_cast<size_t*>(thrust::raw_pointer_cast(bilinear_counts_SU2.data()));
        interactions.trilinear_interaction = const_cast<double*>(thrust::raw_pointer_cast(trilinear_vals_SU2.data()));
        interactions.trilinear_partners = const_cast<size_t*>(thrust::raw_pointer_cast(trilinear_idx_SU2.data()));
        interactions.trilinear_counts = const_cast<size_t*>(thrust::raw_pointer_cast(trilinear_counts_SU2.data()));
        interactions.max_bilinear_neighbors = max_bilinear_SU2;
        interactions.max_trilinear_neighbors = max_trilinear_SU2;
        return interactions;
    }
    
    /**
     * Build SU(3) interaction data struct for kernel calls
     */
    InteractionDataSU3GPU get_interactions_SU3() const {
        InteractionDataSU3GPU interactions;
        interactions.field = const_cast<double*>(thrust::raw_pointer_cast(field_SU3.data()));
        interactions.onsite_interaction = const_cast<double*>(thrust::raw_pointer_cast(onsite_SU3.data()));
        interactions.bilinear_interaction = const_cast<double*>(thrust::raw_pointer_cast(bilinear_vals_SU3.data()));
        interactions.bilinear_partners = const_cast<size_t*>(thrust::raw_pointer_cast(bilinear_idx_SU3.data()));
        interactions.bilinear_counts = const_cast<size_t*>(thrust::raw_pointer_cast(bilinear_counts_SU3.data()));
        interactions.trilinear_interaction = const_cast<double*>(thrust::raw_pointer_cast(trilinear_vals_SU3.data()));
        interactions.trilinear_partners = const_cast<size_t*>(thrust::raw_pointer_cast(trilinear_idx_SU3.data()));
        interactions.trilinear_counts = const_cast<size_t*>(thrust::raw_pointer_cast(trilinear_counts_SU3.data()));
        interactions.max_bilinear_neighbors = max_bilinear_SU3;
        interactions.max_trilinear_neighbors = max_trilinear_SU3;
        return interactions;
    }
    
    /**
     * Build mixed interaction data struct for kernel calls
     */
    MixedInteractionDataGPU get_mixed_interactions() const {
        MixedInteractionDataGPU mixed;
        mixed.bilinear_interaction = const_cast<double*>(thrust::raw_pointer_cast(mixed_bilinear_vals.data()));
        mixed.bilinear_partners_SU2 = const_cast<size_t*>(thrust::raw_pointer_cast(mixed_bilinear_idx_SU2.data()));
        mixed.bilinear_partners_SU3 = const_cast<size_t*>(thrust::raw_pointer_cast(mixed_bilinear_idx_SU3.data()));
        mixed.bilinear_counts_SU2 = const_cast<size_t*>(thrust::raw_pointer_cast(mixed_bilinear_counts_SU2.data()));
        mixed.bilinear_counts_SU3 = const_cast<size_t*>(thrust::raw_pointer_cast(mixed_bilinear_counts_SU2.data()));  // TODO: separate counts
        mixed.max_mixed_bilinear = max_mixed_bilinear;
        return mixed;
    }
    
    /**
     * Build SU(2) field drive params struct for kernel calls
     */
    FieldDriveParamsSU2GPU get_field_drive_SU2() const {
        FieldDriveParamsSU2GPU drive;
        drive.field_drive_1 = const_cast<double*>(thrust::raw_pointer_cast(field_drive_SU2.data()));
        drive.field_drive_2 = const_cast<double*>(thrust::raw_pointer_cast(field_drive_SU2.data())) + 
                              N_atoms_SU2 * spin_dim_SU2;
        drive.amplitude = pulse_amp_SU2;
        drive.width = pulse_width_SU2;
        drive.frequency = pulse_freq_SU2;
        drive.t_pulse_1 = t_pulse_1_SU2;
        drive.t_pulse_2 = t_pulse_2_SU2;
        return drive;
    }
    
    /**
     * Build SU(3) field drive params struct for kernel calls
     */
    FieldDriveParamsSU3GPU get_field_drive_SU3() const {
        FieldDriveParamsSU3GPU drive;
        drive.field_drive_1 = const_cast<double*>(thrust::raw_pointer_cast(field_drive_SU3.data()));
        drive.field_drive_2 = const_cast<double*>(thrust::raw_pointer_cast(field_drive_SU3.data())) + 
                              N_atoms_SU3 * spin_dim_SU3;
        drive.amplitude = pulse_amp_SU3;
        drive.width = pulse_width_SU3;
        drive.frequency = pulse_freq_SU3;
        drive.t_pulse_1 = t_pulse_1_SU3;
        drive.t_pulse_2 = t_pulse_2_SU3;
        return drive;
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
 * Matrix-vector multiplication: result = matrix * vector
 */
__device__
void multiply_matrix_vector_device(double* result, const double* matrix, const double* vector, size_t n);

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

/**
 * Mixed matrix-vector multiplication: result = matrix * vector (non-square)
 */
__device__
void multiply_mixed_matrix_vector_device(double* result, const double* matrix, 
                                          const double* vector, 
                                          size_t n_rows, size_t n_cols);

/**
 * Mixed matrix-vector multiplication with transpose: result = matrix^T * vector
 */
__device__
void multiply_mixed_matrix_transpose_vector_device(double* result, const double* matrix, 
                                                    const double* vector, 
                                                    size_t n_rows, size_t n_cols);

/**
 * Add mixed SU(2)-SU(3) bilinear contribution to local field
 */
__device__
void add_mixed_bilinear_contribution_device(double* local_field, const double* J_mixed,
                                             const double* partner_spin, double* temp,
                                             size_t spin_dim_out, size_t spin_dim_in);

/**
 * Add mixed SU(3)-SU(2) bilinear contribution (transpose) to local field
 */
__device__
void add_mixed_bilinear_transpose_contribution_device(double* local_field, const double* J_mixed,
                                                       const double* partner_spin, double* temp,
                                                       size_t spin_dim_rows, size_t spin_dim_cols);

/**
 * Compute local field for SU(2) site in mixed lattice (unified function)
 */
__device__
void compute_local_field_SU2_device(
    double* local_field,
    const double* d_spins_SU2,
    const double* d_spins_SU3,
    int site,
    const double* field,
    const double* onsite_interaction,
    const double* bilinear_interaction,
    const size_t* bilinear_partners,
    const size_t* bilinear_counts,
    const double* mixed_bilinear_interaction,
    const size_t* mixed_bilinear_partners_SU3,
    const size_t* mixed_bilinear_counts_SU2,
    size_t max_bilinear,
    size_t max_mixed_bilinear,
    size_t lattice_size_SU2,
    size_t lattice_size_SU3,
    size_t spin_dim_SU2,
    size_t spin_dim_SU3
);

/**
 * Compute local field for SU(3) site in mixed lattice (unified function)
 */
__device__
void compute_local_field_SU3_device(
    double* local_field,
    const double* d_spins_SU2,
    const double* d_spins_SU3,
    int site,
    const double* field,
    const double* onsite_interaction,
    const double* bilinear_interaction,
    const size_t* bilinear_partners,
    const size_t* bilinear_counts,
    const double* mixed_bilinear_interaction,
    const size_t* mixed_bilinear_partners_SU2,
    const size_t* mixed_bilinear_counts_SU3,
    size_t max_bilinear,
    size_t max_mixed_bilinear,
    size_t lattice_size_SU2,
    size_t lattice_size_SU3,
    size_t spin_dim_SU2,
    size_t spin_dim_SU3
);

// ======================= Kernel Declarations =======================

/**
 * Compute Landau-Lifshitz derivatives for SU(2) spins
 * dS/dt = S × H_eff
 */
__global__
void LLG_SU2_kernel(
    double* d_dsdt,
    const double* d_spins_SU2,
    const double* d_spins_SU3,
    double* d_local_field,
    InteractionDataSU2GPU interactions_SU2,
    MixedInteractionDataGPU mixed_interactions,
    FieldDriveParamsSU2GPU field_drive,
    TimeStepParamsMixedGPU time_params,
    LatticeDimsMixedGPU dims,
    NeighborCountsMixedGPU neighbors
);

/**
 * Compute Landau-Lifshitz derivatives for SU(3) spins
 */
__global__
void LLG_SU3_kernel(
    double* d_dsdt,
    const double* d_spins_SU2,
    const double* d_spins_SU3,
    double* d_local_field,
    InteractionDataSU3GPU interactions_SU3,
    MixedInteractionDataGPU mixed_interactions,
    FieldDriveParamsSU3GPU field_drive,
    TimeStepParamsMixedGPU time_params,
    LatticeDimsMixedGPU dims,
    NeighborCountsMixedGPU neighbors
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
 * Compute magnetization per component
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

// ======================= GPU ODE System Class =======================

/**
 * GPU ODE system functor for mixed lattice Landau-Lifshitz integration
 * 
 * This class evaluates dS/dt = S × H_eff on GPU without host-device transfers
 */
class GPUMixedODESystem {
public:
    GPUMixedLatticeData& data;
    
    GPUMixedODESystem(GPUMixedLatticeData& d) : data(d) {}
    
    /**
     * Compute dS/dt = S × H_eff on GPU
     * This is called by the integration routine
     * 
     * @param x Current state (device vector)
     * @param dxdt Output derivatives (device vector)
     * @param t Current time
     */
    void operator()(const GPUState& x, GPUState& dxdt, double t) const;
};

// ======================= Host Integration Functions =======================

/**
 * Create GPU mixed lattice data from host arrays
 * Internal function - use the API version (returning a handle) for C++ code
 */
GPUMixedLatticeData create_gpu_mixed_lattice_data_internal(
    size_t lattice_size_SU2, size_t spin_dim_SU2, size_t N_atoms_SU2,
    size_t lattice_size_SU3, size_t spin_dim_SU3, size_t N_atoms_SU3,
    size_t max_bilinear_SU2, size_t max_bilinear_SU3, size_t max_mixed_bilinear,
    const std::vector<double>& flat_field_SU2,
    const std::vector<double>& flat_onsite_SU2,
    const std::vector<double>& flat_bilinear_SU2,
    const std::vector<size_t>& flat_partners_SU2,
    const std::vector<size_t>& num_bilinear_per_site_SU2,
    const std::vector<double>& flat_field_SU3,
    const std::vector<double>& flat_onsite_SU3,
    const std::vector<double>& flat_bilinear_SU3,
    const std::vector<size_t>& flat_partners_SU3,
    const std::vector<size_t>& num_bilinear_per_site_SU3,
    const std::vector<double>& flat_mixed_bilinear,
    const std::vector<size_t>& flat_mixed_partners_SU2,
    const std::vector<size_t>& flat_mixed_partners_SU3,
    const std::vector<size_t>& num_mixed_per_site_SU2
);

/**
 * Set pulse parameters for SU(2) sublattice
 */
void set_gpu_pulse_SU2(
    GPUMixedLatticeData& data,
    const std::vector<double>& flat_field_drive,
    double pulse_amp, double pulse_width, double pulse_freq,
    double t_pulse_1, double t_pulse_2
);

/**
 * Set pulse parameters for SU(3) sublattice
 */
void set_gpu_pulse_SU3(
    GPUMixedLatticeData& data,
    const std::vector<double>& flat_field_drive,
    double pulse_amp, double pulse_width, double pulse_freq,
    double t_pulse_1, double t_pulse_2
);

/**
 * Perform GPU integration with selectable method
 * 
 * Available methods:
 * - "euler": Explicit Euler (1st order)
 * - "rk2" or "midpoint": Modified midpoint (2nd order)
 * - "rk4": Classic Runge-Kutta (4th order)
 * - "rk5" or "rkck54": Cash-Karp 5(4)
 * - "dopri5": Dormand-Prince 5(4) - recommended
 * - "rk78" or "rkf78": Fehlberg 7(8)
 * - "ssprk53": SSP RK 5-stage 3rd order (default, optimized for spin dynamics)
 * - "bulirsch_stoer" or "bs": Bulirsch-Stoer
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
void integrate_mixed_gpu(
    GPUMixedODESystem& system,
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
 */
void step_mixed_gpu(
    GPUMixedODESystem& system,
    GPUState& state,
    double t,
    double dt,
    const std::string& method = "ssprk53"
);

/**
 * Compute magnetization on GPU (for both sublattices)
 */
void compute_magnetization_mixed_gpu(
    const GPUMixedLatticeData& data,
    const GPUState& state,
    double* h_mag_SU2,
    double* h_mag_staggered_SU2,
    double* h_mag_SU3,
    double* h_mag_staggered_SU3
);

/**
 * Normalize spins on GPU
 */
void normalize_spins_mixed_gpu(
    GPUState& state,
    size_t lattice_size_SU2, size_t spin_dim_SU2, double spin_length_SU2,
    size_t lattice_size_SU3, size_t spin_dim_SU3, double spin_length_SU3
);

} // namespace mixed_gpu

#endif // MIXED_LATTICE_GPU_CUH
