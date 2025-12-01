#include "mixed_lattice_gpu.cuh"

namespace mixed_gpu {

// ======================= Double atomicAdd for older architectures =======================

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ inline double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#define ATOMIC_ADD_DOUBLE(addr, val) atomicAdd_double(addr, val)
#else
#define ATOMIC_ADD_DOUBLE(addr, val) atomicAdd(addr, val)
#endif

// ======================= Device Helper Functions =======================

__device__
double dot_device(const double* a, const double* b, size_t n) {
    double result = 0.0;
    if (n <= 4) {
        switch (n) {
            case 4: result += a[3] * b[3]; [[fallthrough]];
            case 3: result += a[2] * b[2]; [[fallthrough]];
            case 2: result += a[1] * b[1]; [[fallthrough]];
            case 1: result += a[0] * b[0]; break;
            default: break;
        }
    } else {
        #pragma unroll 8
        for (size_t i = 0; i < n; ++i) {
            result += a[i] * b[i];
        }
    }
    return result;
}

__device__
double contract_device(const double* spin1, const double* matrix, const double* spin2, size_t n) {
    double result = 0.0;
    if (n == 3) {
        result = spin1[0] * (matrix[0] * spin2[0] + matrix[1] * spin2[1] + matrix[2] * spin2[2]) +
                 spin1[1] * (matrix[3] * spin2[0] + matrix[4] * spin2[1] + matrix[5] * spin2[2]) +
                 spin1[2] * (matrix[6] * spin2[0] + matrix[7] * spin2[1] + matrix[8] * spin2[2]);
    } else if (n == 8) {
        double temp_results[8];
        #pragma unroll
        for (size_t i = 0; i < 8; ++i) {
            temp_results[i] = 0.0;
            #pragma unroll
            for (size_t j = 0; j < 8; ++j) {
                temp_results[i] += matrix[i * 8 + j] * spin2[j];
            }
            result += spin1[i] * temp_results[i];
        }
    } else {
        #pragma unroll 4
        for (size_t i = 0; i < n; ++i) {
            double temp = 0.0;
            #pragma unroll 4
            for (size_t j = 0; j < n; ++j) {
                temp += matrix[i * n + j] * spin2[j];
            }
            result += spin1[i] * temp;
        }
    }
    return result;
}

__device__
void multiply_matrix_vector_device(double* result, const double* matrix, const double* vector, size_t n) {
    if (n == 3) {
        result[0] = matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2];
        result[1] = matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2];
        result[2] = matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2];
    } else if (n == 8) {
        #pragma unroll
        for (size_t i = 0; i < 8; ++i) {
            double temp = 0.0;
            #pragma unroll
            for (size_t j = 0; j < 8; ++j) {
                temp += matrix[i * 8 + j] * vector[j];
            }
            result[i] = temp;
        }
    } else {
        #pragma unroll 4
        for (size_t i = 0; i < n; ++i) {
            result[i] = 0.0;
            #pragma unroll 4
            for (size_t j = 0; j < n; ++j) {
                result[i] += matrix[i * n + j] * vector[j];
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
    constexpr double sqrt3_2 = 0.86602540378443864676372317075294; // sqrt(3)/2
    
    // Component 0 (λ_1): f_{123}, f_{147}, f_{156}
    result[0] = a[1]*b[2] - a[2]*b[1] + 0.5*(a[3]*b[6] - a[6]*b[3]) - 0.5*(a[4]*b[5] - a[5]*b[4]);
    
    // Component 1 (λ_2): f_{213}, f_{246}, f_{257}
    result[1] = a[2]*b[0] - a[0]*b[2] + 0.5*(a[3]*b[5] - a[5]*b[3]) + 0.5*(a[4]*b[6] - a[6]*b[4]);
    
    // Component 2 (λ_3): f_{312}, f_{345}, f_{367}
    result[2] = a[0]*b[1] - a[1]*b[0] + 0.5*(a[3]*b[4] - a[4]*b[3]) - 0.5*(a[5]*b[6] - a[6]*b[5]);
    
    // Component 3 (λ_4): f_{147}, f_{246}, f_{345}, f_{458}
    result[3] = 0.5*(a[6]*b[0] - a[0]*b[6]) + 0.5*(a[5]*b[1] - a[1]*b[5]) + 
                0.5*(a[4]*b[2] - a[2]*b[4]) + sqrt3_2*(a[4]*b[7] - a[7]*b[4]);
    
    // Component 4 (λ_5): f_{156}, f_{257}, f_{345}, f_{458}
    result[4] = -0.5*(a[5]*b[0] - a[0]*b[5]) + 0.5*(a[6]*b[1] - a[1]*b[6]) + 
                0.5*(a[2]*b[3] - a[3]*b[2]) + sqrt3_2*(a[7]*b[3] - a[3]*b[7]);
    
    // Component 5 (λ_6): f_{156}, f_{246}, f_{367}, f_{678}
    result[5] = -0.5*(a[0]*b[4] - a[4]*b[0]) + 0.5*(a[1]*b[3] - a[3]*b[1]) - 
                0.5*(a[6]*b[2] - a[2]*b[6]) + sqrt3_2*(a[6]*b[7] - a[7]*b[6]);
    
    // Component 6 (λ_7): f_{147}, f_{257}, f_{367}, f_{678}
    result[6] = 0.5*(a[0]*b[3] - a[3]*b[0]) + 0.5*(a[1]*b[4] - a[4]*b[1]) - 
                0.5*(a[2]*b[5] - a[5]*b[2]) + sqrt3_2*(a[7]*b[5] - a[5]*b[7]);
    
    // Component 7 (λ_8): f_{458}, f_{678}
    result[7] = sqrt3_2*(a[3]*b[4] - a[4]*b[3]) + sqrt3_2*(a[5]*b[6] - a[6]*b[5]);
}

/**
 * Mixed bilinear contraction for SU(2)-SU(3) coupling
 * result_i = sum_j J_{ij} * spin3_j  (result has spin_dim_SU2 components)
 */
__device__
void multiply_mixed_matrix_vector_device(double* result, const double* matrix, 
                                          const double* vector, 
                                          size_t n_out, size_t n_in) {
    // matrix is n_out x n_in
    for (size_t i = 0; i < n_out; ++i) {
        result[i] = 0.0;
        for (size_t j = 0; j < n_in; ++j) {
            result[i] += matrix[i * n_in + j] * vector[j];
        }
    }
}

// ======================= Kernel Implementations =======================

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
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= dims.lattice_size_SU2) return;
    
    const double* spin_here = &d_spins_SU2[site * dims.spin_dim_SU2];
    double* local_field = &d_local_field[site * dims.spin_dim_SU2];
    double* dsdt = &d_dsdt[site * dims.spin_dim_SU2];
    double temp[8];  // Max dimension
    
    // Initialize with external field
    for (size_t i = 0; i < dims.spin_dim_SU2; ++i) {
        local_field[i] = interactions_SU2.field[site * dims.spin_dim_SU2 + i];
    }
    
    // On-site interaction: H_local -= A * S
    multiply_matrix_vector_device(temp, 
        &interactions_SU2.onsite_interaction[site * dims.spin_dim_SU2 * dims.spin_dim_SU2], 
        spin_here, dims.spin_dim_SU2);
    for (size_t i = 0; i < dims.spin_dim_SU2; ++i) {
        local_field[i] -= temp[i];
    }
    
    // Bilinear SU(2)-SU(2) interactions: H_local -= sum_j J_ij * S_j
    size_t num_neighbors = interactions_SU2.bilinear_counts[site];
    for (size_t n = 0; n < num_neighbors && n < neighbors.max_bilinear_SU2; ++n) {
        size_t partner = interactions_SU2.bilinear_partners[site * neighbors.max_bilinear_SU2 + n];
        if (partner < dims.lattice_size_SU2) {
            const double* partner_spin = &d_spins_SU2[partner * dims.spin_dim_SU2];
            const double* J = &interactions_SU2.bilinear_interaction[
                (site * neighbors.max_bilinear_SU2 + n) * dims.spin_dim_SU2 * dims.spin_dim_SU2];
            
            multiply_matrix_vector_device(temp, J, partner_spin, dims.spin_dim_SU2);
            for (size_t i = 0; i < dims.spin_dim_SU2; ++i) {
                local_field[i] -= temp[i];
            }
        }
    }
    
    // Mixed SU(2)-SU(3) bilinear interactions: H_local -= sum_j J_mixed * S3_j
    size_t num_mixed = mixed_interactions.bilinear_counts_SU2[site];
    for (size_t n = 0; n < num_mixed && n < neighbors.max_mixed_bilinear; ++n) {
        size_t partner_SU3 = mixed_interactions.bilinear_partners_SU3[site * neighbors.max_mixed_bilinear + n];
        if (partner_SU3 < dims.lattice_size_SU3) {
            const double* partner_spin = &d_spins_SU3[partner_SU3 * dims.spin_dim_SU3];
            const double* J_mixed = &mixed_interactions.bilinear_interaction[
                (site * neighbors.max_mixed_bilinear + n) * dims.spin_dim_SU2 * dims.spin_dim_SU3];
            
            // Contract: result_i = sum_j J_{ij} * S3_j
            for (size_t i = 0; i < dims.spin_dim_SU2; ++i) {
                double contrib = 0.0;
                for (size_t j = 0; j < dims.spin_dim_SU3; ++j) {
                    contrib += J_mixed[i * dims.spin_dim_SU3 + j] * partner_spin[j];
                }
                local_field[i] -= contrib;
            }
        }
    }
    
    // Add drive field if pulse amplitude is non-zero
    if (field_drive.amplitude > 0.0) {
        size_t atom = site % dims.N_atoms_SU2;
        double t1_diff = time_params.curr_time - field_drive.t_pulse_1;
        double t2_diff = time_params.curr_time - field_drive.t_pulse_2;
        double env1 = exp(-t1_diff * t1_diff / (2.0 * field_drive.width * field_drive.width));
        double env2 = exp(-t2_diff * t2_diff / (2.0 * field_drive.width * field_drive.width));
        double osc = cos(field_drive.frequency * time_params.curr_time);
        
        for (size_t i = 0; i < dims.spin_dim_SU2; ++i) {
            double drive1 = field_drive.field_drive_1[atom * dims.spin_dim_SU2 + i];
            double drive2 = field_drive.field_drive_2[atom * dims.spin_dim_SU2 + i];
            local_field[i] += field_drive.amplitude * osc * (env1 * drive1 + env2 * drive2);
        }
    }
    
    // Compute Landau-Lifshitz derivative: dS/dt = S × H_eff
    cross_product_SU2_device(dsdt, spin_here, local_field);
}

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
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= dims.lattice_size_SU3) return;
    
    const double* spin_here = &d_spins_SU3[site * dims.spin_dim_SU3];
    double* local_field = &d_local_field[site * dims.spin_dim_SU3];
    double* dsdt = &d_dsdt[dims.SU3_offset + site * dims.spin_dim_SU3];
    double temp[8];
    
    // Initialize with external field
    for (size_t i = 0; i < dims.spin_dim_SU3; ++i) {
        local_field[i] = interactions_SU3.field[site * dims.spin_dim_SU3 + i];
    }
    
    // On-site interaction: H_local -= A * S
    multiply_matrix_vector_device(temp, 
        &interactions_SU3.onsite_interaction[site * dims.spin_dim_SU3 * dims.spin_dim_SU3], 
        spin_here, dims.spin_dim_SU3);
    for (size_t i = 0; i < dims.spin_dim_SU3; ++i) {
        local_field[i] -= temp[i];
    }
    
    // Bilinear SU(3)-SU(3) interactions
    size_t num_neighbors = interactions_SU3.bilinear_counts[site];
    for (size_t n = 0; n < num_neighbors && n < neighbors.max_bilinear_SU3; ++n) {
        size_t partner = interactions_SU3.bilinear_partners[site * neighbors.max_bilinear_SU3 + n];
        if (partner < dims.lattice_size_SU3) {
            const double* partner_spin = &d_spins_SU3[partner * dims.spin_dim_SU3];
            const double* J = &interactions_SU3.bilinear_interaction[
                (site * neighbors.max_bilinear_SU3 + n) * dims.spin_dim_SU3 * dims.spin_dim_SU3];
            
            multiply_matrix_vector_device(temp, J, partner_spin, dims.spin_dim_SU3);
            for (size_t i = 0; i < dims.spin_dim_SU3; ++i) {
                local_field[i] -= temp[i];
            }
        }
    }
    
    // Mixed SU(3)-SU(2) bilinear interactions: H_local -= sum_j J_mixed^T * S2_j
    // Note: For SU(3) sites, we use the transpose of the mixed coupling
    size_t num_mixed = mixed_interactions.bilinear_counts_SU3[site];
    for (size_t n = 0; n < num_mixed && n < neighbors.max_mixed_bilinear; ++n) {
        size_t partner_SU2 = mixed_interactions.bilinear_partners_SU2[site * neighbors.max_mixed_bilinear + n];
        if (partner_SU2 < dims.lattice_size_SU2) {
            const double* partner_spin = &d_spins_SU2[partner_SU2 * dims.spin_dim_SU2];
            const double* J_mixed = &mixed_interactions.bilinear_interaction[
                (site * neighbors.max_mixed_bilinear + n) * dims.spin_dim_SU2 * dims.spin_dim_SU3];
            
            // Contract transpose: result_j = sum_i J_{ij}^T * S2_i = sum_i J_{ji} * S2_i
            // J is stored as spin_dim_SU2 x spin_dim_SU3, so transpose is spin_dim_SU3 x spin_dim_SU2
            for (size_t j = 0; j < dims.spin_dim_SU3; ++j) {
                double contrib = 0.0;
                for (size_t i = 0; i < dims.spin_dim_SU2; ++i) {
                    contrib += J_mixed[i * dims.spin_dim_SU3 + j] * partner_spin[i];
                }
                local_field[j] -= contrib;
            }
        }
    }
    
    // Add drive field
    if (field_drive.amplitude > 0.0) {
        size_t atom = site % dims.N_atoms_SU3;
        double t1_diff = time_params.curr_time - field_drive.t_pulse_1;
        double t2_diff = time_params.curr_time - field_drive.t_pulse_2;
        double env1 = exp(-t1_diff * t1_diff / (2.0 * field_drive.width * field_drive.width));
        double env2 = exp(-t2_diff * t2_diff / (2.0 * field_drive.width * field_drive.width));
        double osc = cos(field_drive.frequency * time_params.curr_time);
        
        for (size_t i = 0; i < dims.spin_dim_SU3; ++i) {
            double drive1 = field_drive.field_drive_1[atom * dims.spin_dim_SU3 + i];
            double drive2 = field_drive.field_drive_2[atom * dims.spin_dim_SU3 + i];
            local_field[i] += field_drive.amplitude * osc * (env1 * drive1 + env2 * drive2);
        }
    }
    
    // Compute Landau-Lifshitz derivative
    cross_product_SU3_device(dsdt, spin_here, local_field);
}

__global__
void update_arrays_kernel(
    double* out,
    const double* in1, double a1,
    const double* in2, double a2,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = a1 * in1[idx] + a2 * in2[idx];
}

__global__
void update_arrays_three_kernel(
    double* out,
    const double* in1, double a1,
    const double* in2, double a2,
    const double* in3, double a3,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = a1 * in1[idx] + a2 * in2[idx] + a3 * in3[idx];
}

__global__
void normalize_spins_kernel(
    double* d_spins,
    double spin_length,
    size_t lattice_size,
    size_t spin_dim
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    double* spin = &d_spins[site * spin_dim];
    double norm = 0.0;
    for (size_t i = 0; i < spin_dim; ++i) {
        norm += spin[i] * spin[i];
    }
    norm = sqrt(norm);
    
    if (norm > 1e-10) {
        for (size_t i = 0; i < spin_dim; ++i) {
            spin[i] = spin[i] * spin_length / norm;
        }
    }
}

__global__
void compute_magnetization_kernel(
    const double* d_spins,
    double* d_mag_local,
    double* d_mag_staggered,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms
) {
    __shared__ double s_mag_local;
    __shared__ double s_mag_staggered;
    
    int tid = threadIdx.x;
    int component = blockIdx.x;
    
    if (tid == 0) {
        s_mag_local = 0.0;
        s_mag_staggered = 0.0;
    }
    __syncthreads();
    
    if (component >= spin_dim) return;
    
    double local_sum = 0.0;
    double staggered_sum = 0.0;
    
    for (size_t i = tid; i < lattice_size; i += blockDim.x) {
        double spin_val = d_spins[i * spin_dim + component];
        local_sum += spin_val;
        int sign = 1 - 2 * ((i % N_atoms) & 1);
        staggered_sum += spin_val * sign;
    }
    
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        staggered_sum += __shfl_down_sync(0xffffffff, staggered_sum, offset);
    }
    
    if ((tid % warpSize) == 0) {
        ATOMIC_ADD_DOUBLE(&s_mag_local, local_sum);
        ATOMIC_ADD_DOUBLE(&s_mag_staggered, staggered_sum);
    }
    __syncthreads();
    
    if (tid == 0) {
        d_mag_local[component] = s_mag_local / double(lattice_size);
        d_mag_staggered[component] = s_mag_staggered / double(lattice_size);
    }
}

// ======================= GPUMixedODESystem Implementation =======================

void GPUMixedODESystem::operator()(const GPUState& x, GPUState& dxdt, double t) const {
    const int BLOCK_SIZE = 256;
    
    // Get pointers
    const double* d_x = thrust::raw_pointer_cast(x.data());
    double* d_dxdt = thrust::raw_pointer_cast(dxdt.data());
    
    // SU(2) spins are at the beginning
    const double* d_spins_SU2 = d_x;
    // SU(3) spins start at offset
    const double* d_spins_SU3 = d_x + data.SU3_offset();
    
    // Working arrays for local fields (separate for SU2 and SU3)
    double* d_local_field_SU2 = thrust::raw_pointer_cast(data.local_field.data());
    double* d_local_field_SU3 = d_local_field_SU2 + data.lattice_size_SU2 * data.spin_dim_SU2;
    
    // Build shared structs using helper methods
    LatticeDimsMixedGPU dims = data.get_dims();
    NeighborCountsMixedGPU neighbors = data.get_neighbor_counts();
    TimeStepParamsMixedGPU time_params = data.get_time_params(t);
    MixedInteractionDataGPU mixed_interactions = data.get_mixed_interactions();
    
    // Launch SU(2) kernel
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((data.lattice_size_SU2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        InteractionDataSU2GPU interactions_SU2 = data.get_interactions_SU2();
        FieldDriveParamsSU2GPU field_drive_SU2 = data.get_field_drive_SU2();
        
        LLG_SU2_kernel<<<grid, block>>>(
            d_dxdt,
            d_spins_SU2,
            d_spins_SU3,
            d_local_field_SU2,
            interactions_SU2,
            mixed_interactions,
            field_drive_SU2,
            time_params,
            dims,
            neighbors
        );
    }
    
    // Launch SU(3) kernel
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((data.lattice_size_SU3 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        InteractionDataSU3GPU interactions_SU3 = data.get_interactions_SU3();
        FieldDriveParamsSU3GPU field_drive_SU3 = data.get_field_drive_SU3();
        
        LLG_SU3_kernel<<<grid, block>>>(
            d_dxdt,
            d_spins_SU2,
            d_spins_SU3,
            d_local_field_SU3,
            interactions_SU3,
            mixed_interactions,
            field_drive_SU3,
            time_params,
            dims,
            neighbors
        );
    }
    
    cudaDeviceSynchronize();
}

// ======================= Host Integration Functions =======================

GPUMixedLatticeData create_gpu_mixed_lattice_data(
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
) {
    GPUMixedLatticeData data;
    
    // Set dimensions
    data.lattice_size_SU2 = lattice_size_SU2;
    data.lattice_size_SU3 = lattice_size_SU3;
    data.spin_dim_SU2 = spin_dim_SU2;
    data.spin_dim_SU3 = spin_dim_SU3;
    data.N_atoms_SU2 = N_atoms_SU2;
    data.N_atoms_SU3 = N_atoms_SU3;
    data.max_bilinear_SU2 = max_bilinear_SU2;
    data.max_bilinear_SU3 = max_bilinear_SU3;
    data.max_mixed_bilinear = max_mixed_bilinear;
    
    size_t total_size = data.state_size();
    
    // Copy SU(2) data to device
    data.field_SU2 = thrust::device_vector<double>(flat_field_SU2.begin(), flat_field_SU2.end());
    data.onsite_SU2 = thrust::device_vector<double>(flat_onsite_SU2.begin(), flat_onsite_SU2.end());
    data.bilinear_vals_SU2 = thrust::device_vector<double>(flat_bilinear_SU2.begin(), flat_bilinear_SU2.end());
    data.bilinear_idx_SU2 = thrust::device_vector<size_t>(flat_partners_SU2.begin(), flat_partners_SU2.end());
    data.bilinear_counts_SU2 = thrust::device_vector<size_t>(num_bilinear_per_site_SU2.begin(), num_bilinear_per_site_SU2.end());
    
    // Copy SU(3) data to device
    data.field_SU3 = thrust::device_vector<double>(flat_field_SU3.begin(), flat_field_SU3.end());
    data.onsite_SU3 = thrust::device_vector<double>(flat_onsite_SU3.begin(), flat_onsite_SU3.end());
    data.bilinear_vals_SU3 = thrust::device_vector<double>(flat_bilinear_SU3.begin(), flat_bilinear_SU3.end());
    data.bilinear_idx_SU3 = thrust::device_vector<size_t>(flat_partners_SU3.begin(), flat_partners_SU3.end());
    data.bilinear_counts_SU3 = thrust::device_vector<size_t>(num_bilinear_per_site_SU3.begin(), num_bilinear_per_site_SU3.end());
    
    // Copy mixed interaction data
    data.mixed_bilinear_vals = thrust::device_vector<double>(flat_mixed_bilinear.begin(), flat_mixed_bilinear.end());
    data.mixed_bilinear_idx_SU2 = thrust::device_vector<size_t>(flat_mixed_partners_SU2.begin(), flat_mixed_partners_SU2.end());
    data.mixed_bilinear_idx_SU3 = thrust::device_vector<size_t>(flat_mixed_partners_SU3.begin(), flat_mixed_partners_SU3.end());
    data.mixed_bilinear_counts_SU2 = thrust::device_vector<size_t>(num_mixed_per_site_SU2.begin(), num_mixed_per_site_SU2.end());
    
    // Initialize working arrays
    data.work_1.resize(total_size, 0.0);
    data.work_2.resize(total_size, 0.0);
    data.work_3.resize(total_size, 0.0);
    data.local_field.resize(total_size, 0.0);
    
    // Initialize field drive to zeros
    data.field_drive_SU2.resize(2 * N_atoms_SU2 * spin_dim_SU2, 0.0);
    data.field_drive_SU3.resize(2 * N_atoms_SU3 * spin_dim_SU3, 0.0);
    
    data.initialized = true;
    return data;
}

void set_gpu_pulse_SU2(
    GPUMixedLatticeData& data,
    const std::vector<double>& flat_field_drive,
    double pulse_amp, double pulse_width, double pulse_freq,
    double t_pulse_1, double t_pulse_2
) {
    data.field_drive_SU2 = thrust::device_vector<double>(flat_field_drive.begin(), flat_field_drive.end());
    data.pulse_amp_SU2 = pulse_amp;
    data.pulse_width_SU2 = pulse_width;
    data.pulse_freq_SU2 = pulse_freq;
    data.t_pulse_1_SU2 = t_pulse_1;
    data.t_pulse_2_SU2 = t_pulse_2;
}

void set_gpu_pulse_SU3(
    GPUMixedLatticeData& data,
    const std::vector<double>& flat_field_drive,
    double pulse_amp, double pulse_width, double pulse_freq,
    double t_pulse_1, double t_pulse_2
) {
    data.field_drive_SU3 = thrust::device_vector<double>(flat_field_drive.begin(), flat_field_drive.end());
    data.pulse_amp_SU3 = pulse_amp;
    data.pulse_width_SU3 = pulse_width;
    data.pulse_freq_SU3 = pulse_freq;
    data.t_pulse_1_SU3 = t_pulse_1;
    data.t_pulse_2_SU3 = t_pulse_2;
}

void step_mixed_gpu(
    GPUMixedODESystem& system,
    GPUState& state,
    double t,
    double dt,
    const std::string& method
) {
    GPUMixedLatticeData& data = system.data;
    size_t array_size = data.state_size();
    
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((array_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    double* d_state = thrust::raw_pointer_cast(state.data());
    
    if (method == "euler") {
        // Euler: y_{n+1} = y_n + h * f(t_n, y_n)
        GPUState k(array_size);
        double* d_k = thrust::raw_pointer_cast(k.data());
        
        system(state, k, t);
        update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k, dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk2" || method == "midpoint") {
        // RK2: midpoint method
        GPUState k1(array_size), k2(array_size), tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        system(state, k1, t);
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        
        system(tmp, k2, t + 0.5 * dt);
        update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k2, dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk4") {
        // Classic RK4
        GPUState k1(array_size), k2(array_size), k3(array_size), k4(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        system(state, k1, t);
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        
        system(tmp, k2, t + 0.5 * dt);
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k2, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        
        system(tmp, k3, t + 0.5 * dt);
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k3, dt, array_size);
        cudaDeviceSynchronize();
        
        system(tmp, k4, t + dt);
        
        // Combine: y_{n+1} = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        update_arrays_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k2, 2.0, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k3, 2.0, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k4, 1.0, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, dt / 6.0, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "dopri5") {
        // Dormand-Prince 5(4)
        GPUState k1(array_size), k2(array_size), k3(array_size), k4(array_size);
        GPUState k5(array_size), k6(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_k5 = thrust::raw_pointer_cast(k5.data());
        double* d_k6 = thrust::raw_pointer_cast(k6.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        // Dormand-Prince coefficients
        constexpr double a21 = 1.0/5.0;
        constexpr double a31 = 3.0/40.0, a32 = 9.0/40.0;
        constexpr double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
        constexpr double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
        constexpr double a61 = 9017.0/3168.0, a62 = -355.0/33.0, a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;
        constexpr double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 4.0/5.0, c5 = 8.0/9.0, c6 = 1.0;
        constexpr double b1 = 35.0/384.0, b3 = 500.0/1113.0, b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0;
        
        system(state, k1, t);
        
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a21 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k2, t + c2 * dt);
        
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a31 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a32 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k3, t + c3 * dt);
        
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a41 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a42 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a43 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k4, t + c4 * dt);
        
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a51 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a52 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a53 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a54 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k5, t + c5 * dt);
        
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a61 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a62 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a63 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a64 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, a65 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k6, t + c6 * dt);
        
        // Final combination
        update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, b1 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k3, b3 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k4, b4 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k5, b5 * dt, array_size);
        cudaDeviceSynchronize();
        update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k6, b6 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "ssprk53") {
        // SSPRK53: Strong Stability Preserving RK, 5 stages, 3rd order
        constexpr double a30 = 0.355909775063327;
        constexpr double a32 = 0.644090224936674;
        constexpr double a40 = 0.367933791638137;
        constexpr double a43 = 0.632066208361863;
        constexpr double a52 = 0.237593836598569;
        constexpr double a54 = 0.762406163401431;
        constexpr double b10 = 0.377268915331368;
        constexpr double b21 = 0.377268915331368;
        constexpr double b32 = 0.242995220537396;
        constexpr double b43 = 0.238458932846290;
        constexpr double b54 = 0.287632146308408;
        constexpr double c1 = 0.377268915331368;
        constexpr double c2 = 0.754537830662736;
        constexpr double c3 = 0.728985661612188;
        constexpr double c4 = 0.699226135931670;
        
        GPUState k(array_size), tmp(array_size), u(array_size);
        double* d_k = thrust::raw_pointer_cast(k.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        double* d_u = thrust::raw_pointer_cast(u.data());
        
        // Stage 1
        system(state, k, t);
        update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k, b10 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 2
        system(tmp, k, t + c1 * dt);
        update_arrays_kernel<<<grid, block>>>(d_u, d_tmp, 1.0, d_k, b21 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 3
        system(u, k, t + c2 * dt);
        update_arrays_three_kernel<<<grid, block>>>(d_tmp, d_state, a30, d_u, a32, d_k, b32 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 4
        system(tmp, k, t + c3 * dt);
        update_arrays_three_kernel<<<grid, block>>>(d_tmp, d_state, a40, d_tmp, a43, d_k, b43 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 5 (final)
        system(tmp, k, t + c4 * dt);
        update_arrays_three_kernel<<<grid, block>>>(d_state, d_u, a52, d_tmp, a54, d_k, b54 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else {
        // Default to SSPRK53
        std::cerr << "Warning: Unknown GPU integration method '" << method << "', using ssprk53" << std::endl;
        step_mixed_gpu(system, state, t, dt, "ssprk53");
    }
}

void integrate_mixed_gpu(
    GPUMixedODESystem& system,
    GPUState& state,
    double T_start,
    double T_end,
    double dt,
    size_t save_interval,
    std::vector<std::pair<double, std::vector<double>>>& trajectory,
    const std::string& method
) {
    double t = T_start;
    size_t step = 0;
    
    while (t < T_end) {
        // Save trajectory at intervals
        if (step % save_interval == 0) {
            thrust::host_vector<double> h_state = state;
            std::vector<double> state_vec(h_state.begin(), h_state.end());
            trajectory.push_back({t, state_vec});
        }
        
        // Perform one step
        step_mixed_gpu(system, state, t, dt, method);
        
        t += dt;
        step++;
    }
    
    // Save final state
    thrust::host_vector<double> h_state = state;
    std::vector<double> state_vec(h_state.begin(), h_state.end());
    trajectory.push_back({t, state_vec});
}

void compute_magnetization_mixed_gpu(
    const GPUMixedLatticeData& data,
    const GPUState& state,
    double* h_mag_SU2,
    double* h_mag_staggered_SU2,
    double* h_mag_SU3,
    double* h_mag_staggered_SU3
) {
    thrust::device_vector<double> d_mag_local_SU2(data.spin_dim_SU2);
    thrust::device_vector<double> d_mag_staggered_SU2(data.spin_dim_SU2);
    thrust::device_vector<double> d_mag_local_SU3(data.spin_dim_SU3);
    thrust::device_vector<double> d_mag_staggered_SU3(data.spin_dim_SU3);
    
    const double* d_state = thrust::raw_pointer_cast(state.data());
    
    // Compute SU(2) magnetization
    const int threads_per_block = 256;
    compute_magnetization_kernel<<<data.spin_dim_SU2, threads_per_block>>>(
        d_state,
        thrust::raw_pointer_cast(d_mag_local_SU2.data()),
        thrust::raw_pointer_cast(d_mag_staggered_SU2.data()),
        data.lattice_size_SU2,
        data.spin_dim_SU2,
        data.N_atoms_SU2
    );
    
    // Compute SU(3) magnetization
    compute_magnetization_kernel<<<data.spin_dim_SU3, threads_per_block>>>(
        d_state + data.SU3_offset(),
        thrust::raw_pointer_cast(d_mag_local_SU3.data()),
        thrust::raw_pointer_cast(d_mag_staggered_SU3.data()),
        data.lattice_size_SU3,
        data.spin_dim_SU3,
        data.N_atoms_SU3
    );
    
    cudaDeviceSynchronize();
    
    // Copy results back to host
    thrust::copy(d_mag_local_SU2.begin(), d_mag_local_SU2.end(), h_mag_SU2);
    thrust::copy(d_mag_staggered_SU2.begin(), d_mag_staggered_SU2.end(), h_mag_staggered_SU2);
    thrust::copy(d_mag_local_SU3.begin(), d_mag_local_SU3.end(), h_mag_SU3);
    thrust::copy(d_mag_staggered_SU3.begin(), d_mag_staggered_SU3.end(), h_mag_staggered_SU3);
}

void normalize_spins_mixed_gpu(
    GPUState& state,
    size_t lattice_size_SU2, size_t spin_dim_SU2, double spin_length_SU2,
    size_t lattice_size_SU3, size_t spin_dim_SU3, double spin_length_SU3
) {
    const int BLOCK_SIZE = 256;
    double* d_state = thrust::raw_pointer_cast(state.data());
    
    // Normalize SU(2) spins
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((lattice_size_SU2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        normalize_spins_kernel<<<grid, block>>>(d_state, spin_length_SU2, lattice_size_SU2, spin_dim_SU2);
    }
    
    // Normalize SU(3) spins
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((lattice_size_SU3 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        size_t SU3_offset = lattice_size_SU2 * spin_dim_SU2;
        normalize_spins_kernel<<<grid, block>>>(d_state + SU3_offset, spin_length_SU3, lattice_size_SU3, spin_dim_SU3);
    }
    
    cudaDeviceSynchronize();
}

} // namespace mixed_gpu
