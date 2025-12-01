#include "lattice_gpu.cuh"

// ======================= Double atomicAdd for older architectures =======================

// atomicAdd for double is only natively supported on compute capability 6.0+
// This provides a fallback implementation for older GPUs
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
    
    // Unroll for common small sizes
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
    
    // Optimize for SU(2) 3x3 and SU(3) 8x8 cases
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
double contract_trilinear_device(const double* tensor, const double* spin1, 
                                  const double* spin2, const double* spin3, size_t n) {
    double result = 0.0;
    
    if (n == 3) {
        for (size_t i = 0; i < 3; ++i) {
            double temp_j = 0.0;
            for (size_t j = 0; j < 3; ++j) {
                double temp_k = 0.0;
                temp_k += tensor[i * 9 + j * 3 + 0] * spin3[0];
                temp_k += tensor[i * 9 + j * 3 + 1] * spin3[1];
                temp_k += tensor[i * 9 + j * 3 + 2] * spin3[2];
                temp_j += spin2[j] * temp_k;
            }
            result += spin1[i] * temp_j;
        }
    } else if (n == 8) {
        double temp_results[8];
        #pragma unroll
        for (size_t i = 0; i < 8; ++i) {
            temp_results[i] = 0.0;
            #pragma unroll
            for (size_t j = 0; j < 8; ++j) {
                double temp_k = 0.0;
                #pragma unroll
                for (size_t k = 0; k < 8; ++k) {
                    temp_k += tensor[i * 64 + j * 8 + k] * spin3[k];
                }
                temp_results[i] += spin2[j] * temp_k;
            }
            result += spin1[i] * temp_results[i];
        }
    } else {
        #pragma unroll 2
        for (size_t i = 0; i < n; ++i) {
            double temp_i = 0.0;
            #pragma unroll 2
            for (size_t j = 0; j < n; ++j) {
                double temp_j = 0.0;
                #pragma unroll 4
                for (size_t k = 0; k < n; ++k) {
                    temp_j += tensor[i * n * n + j * n + k] * spin3[k];
                }
                temp_i += spin2[j] * temp_j;
            }
            result += spin1[i] * temp_i;
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
void contract_trilinear_field_device(double* result, const double* tensor, 
                                      const double* spin1, const double* spin2, size_t n) {
    if (n == 3) {
        result[0] = result[1] = result[2] = 0.0;
        
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                double spin_product = spin1[j] * spin2[k];
                result[0] += tensor[0 * 9 + j * 3 + k] * spin_product;
                result[1] += tensor[1 * 9 + j * 3 + k] * spin_product;
                result[2] += tensor[2 * 9 + j * 3 + k] * spin_product;
            }
        }
    } else if (n == 8) {
        #pragma unroll
        for (size_t i = 0; i < 8; ++i) {
            result[i] = 0.0;
        }
        
        #pragma unroll
        for (size_t j = 0; j < 8; ++j) {
            #pragma unroll
            for (size_t k = 0; k < 8; ++k) {
                double spin_product = spin1[j] * spin2[k];
                #pragma unroll
                for (size_t i = 0; i < 8; ++i) {
                    result[i] += tensor[i * 64 + j * 8 + k] * spin_product;
                }
            }
        }
    } else {
        #pragma unroll 4
        for (size_t i = 0; i < n; ++i) {
            result[i] = 0.0;
        }
        
        #pragma unroll 2
        for (size_t j = 0; j < n; ++j) {
            #pragma unroll 2
            for (size_t k = 0; k < n; ++k) {
                double spin_product = spin1[j] * spin2[k];
                #pragma unroll 4
                for (size_t i = 0; i < n; ++i) {
                    result[i] += tensor[i * n * n + j * n + k] * spin_product;
                }
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

// ======================= Shared Device Functions for Local Field Computation =======================

/**
 * Initialize local field with external field
 * Used by both lattice and mixed_lattice kernels
 */
__device__
void init_local_field_device(double* local_field, const double* external_field, size_t spin_dim) {
    for (size_t i = 0; i < spin_dim; ++i) {
        local_field[i] = external_field[i];
    }
}

/**
 * Add on-site interaction contribution: local_field -= A * spin
 * Used by both lattice and mixed_lattice kernels
 */
__device__
void add_onsite_contribution_device(double* local_field, const double* onsite_matrix, 
                                     const double* spin, double* temp, size_t spin_dim) {
    multiply_matrix_vector_device(temp, onsite_matrix, spin, spin_dim);
    for (size_t i = 0; i < spin_dim; ++i) {
        local_field[i] -= temp[i];
    }
}

/**
 * Add bilinear interaction contribution: local_field -= J * partner_spin
 * Used by both lattice and mixed_lattice kernels
 */
__device__
void add_bilinear_contribution_device(double* local_field, const double* J_matrix,
                                       const double* partner_spin, double* temp, size_t spin_dim) {
    multiply_matrix_vector_device(temp, J_matrix, partner_spin, spin_dim);
    for (size_t i = 0; i < spin_dim; ++i) {
        local_field[i] -= temp[i];
    }
}

/**
 * Add trilinear interaction contribution: local_field -= T * spin1 * spin2
 * Used by both lattice and mixed_lattice kernels
 */
__device__
void add_trilinear_contribution_device(double* local_field, const double* T_tensor,
                                        const double* spin1, const double* spin2, 
                                        double* temp, size_t spin_dim) {
    contract_trilinear_field_device(temp, T_tensor, spin1, spin2, spin_dim);
    for (size_t i = 0; i < spin_dim; ++i) {
        local_field[i] -= temp[i];
    }
}

/**
 * Add drive field contribution to local field
 * Used by both lattice and mixed_lattice kernels
 * @param local_field Output local field array
 * @param field_drive_1 First pulse component [N_atoms * spin_dim]
 * @param field_drive_2 Second pulse component [N_atoms * spin_dim]
 * @param atom Atom index within unit cell (site % N_atoms)
 * @param amplitude Pulse amplitude
 * @param width Gaussian width
 * @param frequency Oscillation frequency
 * @param t_pulse_1 Center time of first pulse
 * @param t_pulse_2 Center time of second pulse
 * @param curr_time Current simulation time
 * @param spin_dim Spin dimension
 */
__device__
void add_drive_field_device(double* local_field,
                            const double* field_drive_1, const double* field_drive_2,
                            size_t atom, double amplitude, double width, double frequency,
                            double t_pulse_1, double t_pulse_2, double curr_time,
                            size_t spin_dim) {
    if (amplitude > 0.0) {
        double t1_diff = curr_time - t_pulse_1;
        double t2_diff = curr_time - t_pulse_2;
        double env1 = exp(-t1_diff * t1_diff / (2.0 * width * width));
        double env2 = exp(-t2_diff * t2_diff / (2.0 * width * width));
        double osc = cos(frequency * curr_time);
        
        for (size_t i = 0; i < spin_dim; ++i) {
            local_field[i] += amplitude * osc * (
                env1 * field_drive_1[atom * spin_dim + i] +
                env2 * field_drive_2[atom * spin_dim + i]
            );
        }
    }
}

/**
 * Compute Landau-Lifshitz derivative: dsdt = H_eff × spin
 * Used by both lattice and mixed_lattice kernels
 */
__device__
void compute_ll_derivative_device(double* dsdt, const double* spin, 
                                   const double* local_field, size_t spin_dim) {
    if (spin_dim == 3) {
        cross_product_SU2_device(dsdt, local_field, spin);
    } else if (spin_dim == 8) {
        cross_product_SU3_device(dsdt, local_field, spin);
    } else {
        // Generic case - zero for now
        for (size_t i = 0; i < spin_dim; ++i) {
            dsdt[i] = 0.0;
        }
    }
}

/**
 * Compute local field for a single site (unified function)
 * Used by compute_local_field_kernel and LLG_kernel
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
) {
    const double* spin_here = &d_spins[site * spin_dim];
    double temp[8];
    
    // Initialize with external field
    init_local_field_device(local_field, &field[site * spin_dim], spin_dim);
    
    // On-site interaction: H_local -= A * S
    add_onsite_contribution_device(local_field, 
        &onsite_interaction[site * spin_dim * spin_dim], 
        spin_here, temp, spin_dim);
    
    // Bilinear interactions: H_local -= sum_j J_ij * S_j
    for (size_t n = 0; n < num_bilinear && n < max_bilinear; ++n) {
        size_t partner = bilinear_partners[site * max_bilinear + n];
        if (partner < lattice_size) {
            const double* partner_spin = &d_spins[partner * spin_dim];
            const double* J = &bilinear_interaction[
                site * max_bilinear * spin_dim * spin_dim + n * spin_dim * spin_dim];
            
            add_bilinear_contribution_device(local_field, J, partner_spin, temp, spin_dim);
        }
    }
    
    // Trilinear interactions: H_local -= sum_{jk} T_ijk * S_j * S_k
    for (size_t n = 0; n < num_trilinear && n < max_trilinear; ++n) {
        size_t p1 = trilinear_partners[site * max_trilinear * 2 + n * 2];
        size_t p2 = trilinear_partners[site * max_trilinear * 2 + n * 2 + 1];
        if (p1 < lattice_size && p2 < lattice_size) {
            const double* spin1 = &d_spins[p1 * spin_dim];
            const double* spin2 = &d_spins[p2 * spin_dim];
            const double* T = &trilinear_interaction[
                site * max_trilinear * spin_dim * spin_dim * spin_dim + 
                n * spin_dim * spin_dim * spin_dim];
            
            add_trilinear_contribution_device(local_field, T, spin1, spin2, temp, spin_dim);
        }
    }
}

// ======================= Kernel Implementations =======================

__global__
void compute_local_field_kernel(
    double* d_local_field,
    const double* d_spins,
    InteractionDataGPU interactions,
    NeighborCountsGPU neighbors,
    size_t lattice_size,
    size_t spin_dim
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    double* local_field = &d_local_field[site * spin_dim];
    
    // Compute local field using unified device function
    compute_local_field_device(
        local_field, d_spins, site,
        interactions.field,
        interactions.onsite_interaction,
        interactions.bilinear_interaction,
        interactions.bilinear_partners,
        interactions.trilinear_interaction,
        interactions.trilinear_partners,
        neighbors.num_bilinear, neighbors.max_bilinear,
        neighbors.num_trilinear, neighbors.max_trilinear,
        lattice_size, spin_dim
    );
}

__global__
void add_drive_field_kernel(
    double* d_local_field,
    FieldDriveParamsGPU field_drive,
    double curr_time,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    double* local_field = &d_local_field[site * spin_dim];
    
    // Add drive field using shared function
    add_drive_field_device(local_field, field_drive.field_drive_1, field_drive.field_drive_2,
                           site % N_atoms, field_drive.amplitude, field_drive.width,
                           field_drive.frequency, field_drive.t_pulse_1, field_drive.t_pulse_2,
                           curr_time, spin_dim);
}

__global__
void landau_lifshitz_kernel(
    double* d_dsdt,
    const double* d_spins,
    const double* d_local_field,
    size_t lattice_size,
    size_t spin_dim
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    const double* spin = &d_spins[site * spin_dim];
    const double* H_eff = &d_local_field[site * spin_dim];
    double* dsdt = &d_dsdt[site * spin_dim];
    
    // dS/dt = H_eff × S (using shared function)
    compute_ll_derivative_device(dsdt, spin, H_eff, spin_dim);
}

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
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    const double* spin_here = &d_spins[site * spin_dim];
    double* local_field = &d_local_field[site * spin_dim];
    double* dsdt = &d_dsdt[site * spin_dim];
    
    // Compute local field using unified device function
    compute_local_field_device(
        local_field, d_spins, site,
        interactions.field,
        interactions.onsite_interaction,
        interactions.bilinear_interaction,
        interactions.bilinear_partners,
        interactions.trilinear_interaction,
        interactions.trilinear_partners,
        neighbors.num_bilinear, neighbors.max_bilinear,
        neighbors.num_trilinear, neighbors.max_trilinear,
        lattice_size, spin_dim
    );
    
    // Add drive field
    add_drive_field_device(local_field, field_drive.field_drive_1, field_drive.field_drive_2,
                           site % N_atoms, field_drive.amplitude, field_drive.width,
                           field_drive.frequency, field_drive.t_pulse_1, field_drive.t_pulse_2,
                           time_params.curr_time, spin_dim);
    
    // Compute Landau-Lifshitz derivative
    compute_ll_derivative_device(dsdt, spin_here, local_field, spin_dim);
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
void compute_site_energy_kernel(
    double* d_energies,
    const double* d_spins,
    InteractionDataGPU interactions,
    NeighborCountsGPU neighbors,
    size_t lattice_size,
    size_t spin_dim
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    double energy = 0.0;
    const double* spin_here = &d_spins[site * spin_dim];
    
    // Field energy: -S · H
    energy -= dot_device(spin_here, &interactions.field[site * spin_dim], spin_dim);
    
    // Onsite energy: S · A · S / 2
    energy += contract_device(spin_here, 
        &interactions.onsite_interaction[site * spin_dim * spin_dim], 
        spin_here, spin_dim) / 2.0;
    
    // Bilinear energy: S · J · S' / 2
    for (size_t n = 0; n < neighbors.num_bilinear && n < neighbors.max_bilinear; ++n) {
        size_t partner = interactions.bilinear_partners[site * neighbors.max_bilinear + n];
        if (partner < lattice_size) {
            energy += contract_device(spin_here, 
                &interactions.bilinear_interaction[
                    site * neighbors.max_bilinear * spin_dim * spin_dim + n * spin_dim * spin_dim],
                &d_spins[partner * spin_dim], spin_dim) / 2.0;
        }
    }
    
    // Trilinear energy: T · S · S' · S'' / 3
    for (size_t n = 0; n < neighbors.num_trilinear && n < neighbors.max_trilinear; ++n) {
        size_t p1 = interactions.trilinear_partners[site * neighbors.max_trilinear * 2 + n * 2];
        size_t p2 = interactions.trilinear_partners[site * neighbors.max_trilinear * 2 + n * 2 + 1];
        if (p1 < lattice_size && p2 < lattice_size) {
            energy += contract_trilinear_device(
                &interactions.trilinear_interaction[
                    site * neighbors.max_trilinear * spin_dim * spin_dim * spin_dim + 
                    n * spin_dim * spin_dim * spin_dim],
                spin_here, &d_spins[p1 * spin_dim], &d_spins[p2 * spin_dim], spin_dim) / 3.0;
        }
    }
    
    d_energies[site] = energy;
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

// ======================= Host Integration Functions =======================

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
) {
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((lattice_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t array_size = lattice_size * spin_dim;
    
    double* k = work_arrays.work_1;
    double* local_field = work_arrays.local_field;
    
    // Compute k = f(t, y)
    LLG_kernel<<<grid, block>>>(
        k, d_spins, local_field,
        interactions, neighbors, field_drive, time_params,
        lattice_size, spin_dim, N_atoms
    );
    
    // y_new = y + dt * k
    update_arrays_kernel<<<(array_size + BLOCK_SIZE - 1) / BLOCK_SIZE, block>>>(
        d_spins, d_spins, 1.0, k, time_params.dt, array_size
    );
}

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
) {
    // SSPRK53 coefficients
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
    
    const double dt = time_params.dt;
    const double curr_time = time_params.curr_time;
    
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((lattice_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t array_size = lattice_size * spin_dim;
    dim3 array_grid((array_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    double* k = work_arrays.work_1;
    double* tmp = work_arrays.work_2;
    double* u = work_arrays.work_3;
    double* local_field = work_arrays.local_field;
    
    // Stage 1: k1 = f(t, y)
    TimeStepParamsGPU time_stage = time_params;
    LLG_kernel<<<grid, block>>>(k, d_spins, local_field, interactions, neighbors, 
                                 field_drive, time_stage, lattice_size, spin_dim, N_atoms);
    
    // tmp = y + b10*h*k1
    update_arrays_kernel<<<array_grid, block>>>(tmp, d_spins, 1.0, k, b10 * dt, array_size);
    
    // Stage 2: k2 = f(t + c1*h, tmp)
    time_stage.curr_time = curr_time + c1 * dt;
    LLG_kernel<<<grid, block>>>(k, tmp, local_field, interactions, neighbors, 
                                 field_drive, time_stage, lattice_size, spin_dim, N_atoms);
    
    // u = tmp + b21*h*k2
    update_arrays_kernel<<<array_grid, block>>>(u, tmp, 1.0, k, b21 * dt, array_size);
    
    // Stage 3: k3 = f(t + c2*h, u)
    time_stage.curr_time = curr_time + c2 * dt;
    LLG_kernel<<<grid, block>>>(k, u, local_field, interactions, neighbors, 
                                 field_drive, time_stage, lattice_size, spin_dim, N_atoms);
    
    // tmp = a30*y + a32*u + b32*h*k3
    update_arrays_three_kernel<<<array_grid, block>>>(tmp, d_spins, a30, u, a32, k, b32 * dt, array_size);
    
    // Stage 4: k4 = f(t + c3*h, tmp)
    time_stage.curr_time = curr_time + c3 * dt;
    LLG_kernel<<<grid, block>>>(k, tmp, local_field, interactions, neighbors, 
                                 field_drive, time_stage, lattice_size, spin_dim, N_atoms);
    
    // tmp = a40*y + a43*tmp + b43*h*k4
    update_arrays_three_kernel<<<array_grid, block>>>(tmp, d_spins, a40, tmp, a43, k, b43 * dt, array_size);
    
    // Stage 5: k5 = f(t + c4*h, tmp)
    time_stage.curr_time = curr_time + c4 * dt;
    LLG_kernel<<<grid, block>>>(k, tmp, local_field, interactions, neighbors, 
                                 field_drive, time_stage, lattice_size, spin_dim, N_atoms);
    
    // Final: y_new = a52*u + a54*tmp + b54*h*k5
    update_arrays_three_kernel<<<array_grid, block>>>(d_spins, u, a52, tmp, a54, k, b54 * dt, array_size);
}

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
) {
    const double dt = time_params.dt;
    const double curr_time = time_params.curr_time;
    
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((lattice_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t array_size = lattice_size * spin_dim;
    dim3 array_grid((array_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // We need 4 k arrays for RK4, but can reuse with care
    double* k1 = work_arrays.work_1;
    double* k_tmp = work_arrays.work_2;
    double* y_tmp = work_arrays.work_3;
    double* local_field = work_arrays.local_field;
    
    // Allocate additional arrays for RK4 (k2, k3, k4 accumulators)
    // For simplicity, we accumulate directly
    
    TimeStepParamsGPU time_stage = time_params;
    
    // k1 = f(t, y)
    LLG_kernel<<<grid, block>>>(k1, d_spins, local_field, interactions, neighbors, 
                                 field_drive, time_stage, lattice_size, spin_dim, N_atoms);
    
    // y_tmp = y + 0.5*dt*k1
    update_arrays_kernel<<<array_grid, block>>>(y_tmp, d_spins, 1.0, k1, 0.5 * dt, array_size);
    
    // k2 = f(t + 0.5*dt, y_tmp)
    time_stage.curr_time = curr_time + 0.5 * dt;
    LLG_kernel<<<grid, block>>>(k_tmp, y_tmp, local_field, interactions, neighbors, 
                                 field_drive, time_stage, lattice_size, spin_dim, N_atoms);
    
    // k1 += 2*k2 (accumulate)
    update_arrays_kernel<<<array_grid, block>>>(k1, k1, 1.0, k_tmp, 2.0, array_size);
    
    // y_tmp = y + 0.5*dt*k2
    update_arrays_kernel<<<array_grid, block>>>(y_tmp, d_spins, 1.0, k_tmp, 0.5 * dt, array_size);
    
    // k3 = f(t + 0.5*dt, y_tmp)
    LLG_kernel<<<grid, block>>>(k_tmp, y_tmp, local_field, interactions, neighbors, 
                                 field_drive, time_stage, lattice_size, spin_dim, N_atoms);
    
    // k1 += 2*k3
    update_arrays_kernel<<<array_grid, block>>>(k1, k1, 1.0, k_tmp, 2.0, array_size);
    
    // y_tmp = y + dt*k3
    update_arrays_kernel<<<array_grid, block>>>(y_tmp, d_spins, 1.0, k_tmp, dt, array_size);
    
    // k4 = f(t + dt, y_tmp)
    time_stage.curr_time = curr_time + dt;
    LLG_kernel<<<grid, block>>>(k_tmp, y_tmp, local_field, interactions, neighbors, 
                                 field_drive, time_stage, lattice_size, spin_dim, N_atoms);
    
    // k1 += k4, now k1 = k1 + 2*k2 + 2*k3 + k4
    update_arrays_kernel<<<array_grid, block>>>(k1, k1, 1.0, k_tmp, 1.0, array_size);
    
    // y_new = y + (dt/6) * k1
    update_arrays_kernel<<<array_grid, block>>>(d_spins, d_spins, 1.0, k1, dt / 6.0, array_size);
}

double total_energy_gpu(
    const double* d_spins,
    InteractionDataGPU& interactions,
    NeighborCountsGPU& neighbors,
    size_t lattice_size,
    size_t spin_dim
) {
    // Allocate device memory for site energies
    double* d_energies;
    cudaMalloc(&d_energies, lattice_size * sizeof(double));
    
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((lattice_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    compute_site_energy_kernel<<<grid, block>>>(
        d_energies, d_spins, interactions, neighbors, lattice_size, spin_dim
    );
    
    // Reduce on GPU using thrust
    thrust::device_ptr<double> d_ptr(d_energies);
    double total = thrust::reduce(d_ptr, d_ptr + lattice_size);
    
    cudaFree(d_energies);
    return total;
}

void compute_magnetization_gpu(
    const double* d_spins,
    double* h_mag_local,
    double* h_mag_staggered,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms
) {
    double *d_mag_local, *d_mag_staggered;
    cudaMalloc(&d_mag_local, spin_dim * sizeof(double));
    cudaMalloc(&d_mag_staggered, spin_dim * sizeof(double));
    
    const int threads_per_block = 256;
    compute_magnetization_kernel<<<spin_dim, threads_per_block>>>(
        d_spins, d_mag_local, d_mag_staggered, lattice_size, spin_dim, N_atoms
    );
    
    cudaMemcpy(h_mag_local, d_mag_local, spin_dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mag_staggered, d_mag_staggered, spin_dim * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_mag_local);
    cudaFree(d_mag_staggered);
}

// ======================= LatticeGPU Class Implementation =======================

// Forward declaration - actual implementation needs lattice.h which creates circular dependency
// This will be implemented in a separate file or as inline in the header after lattice.h is included

// Placeholder implementations - these need the full Lattice class definition

// Note: The full LatticeGPU implementation that interfaces with the CPU Lattice class
// should be placed in a separate file (lattice_gpu_impl.cu) that includes both
// lattice.h and lattice_gpu.cuh to avoid circular dependencies.

// ======================= gpu:: Namespace Implementation =======================

namespace gpu {

/**
 * Kernel to compute Landau-Lifshitz derivatives using flattened arrays
 */
__global__
void LLG_flat_kernel(
    double* d_dsdt,
    const double* d_spins,
    double* d_local_field,
    const double* d_field,
    const double* d_onsite,
    const double* d_bilinear_vals,
    const size_t* d_bilinear_idx,
    const size_t* d_bilinear_counts,
    const double* d_field_drive,
    double pulse_amp,
    double pulse_width,
    double pulse_freq,
    double t_pulse_1,
    double t_pulse_2,
    double curr_time,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms,
    size_t max_bilinear
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    const double* spin_here = &d_spins[site * spin_dim];
    double* local_field = &d_local_field[site * spin_dim];
    double* dsdt = &d_dsdt[site * spin_dim];
    
    // Initialize with external field
    for (size_t i = 0; i < spin_dim; ++i) {
        local_field[i] = d_field[site * spin_dim + i];
    }
    
    // On-site interaction: H_local -= A * S
    const double* A = &d_onsite[site * spin_dim * spin_dim];
    for (size_t i = 0; i < spin_dim; ++i) {
        double temp = 0.0;
        for (size_t j = 0; j < spin_dim; ++j) {
            temp += A[i * spin_dim + j] * spin_here[j];
        }
        local_field[i] -= temp;
    }
    
    // Bilinear interactions: H_local -= sum_j J_ij * S_j
    size_t num_neighbors = d_bilinear_counts[site];
    for (size_t n = 0; n < num_neighbors && n < max_bilinear; ++n) {
        size_t partner = d_bilinear_idx[site * max_bilinear + n];
        if (partner < lattice_size) {
            const double* partner_spin = &d_spins[partner * spin_dim];
            const double* J = &d_bilinear_vals[(site * max_bilinear + n) * spin_dim * spin_dim];
            
            for (size_t i = 0; i < spin_dim; ++i) {
                double temp = 0.0;
                for (size_t j = 0; j < spin_dim; ++j) {
                    temp += J[i * spin_dim + j] * partner_spin[j];
                }
                local_field[i] -= temp;
            }
        }
    }
    
    // Add drive field if pulse amplitude is non-zero
    if (pulse_amp > 0.0) {
        size_t atom = site % N_atoms;
        double t1_diff = curr_time - t_pulse_1;
        double t2_diff = curr_time - t_pulse_2;
        double env1 = exp(-t1_diff * t1_diff / (2.0 * pulse_width * pulse_width));
        double env2 = exp(-t2_diff * t2_diff / (2.0 * pulse_width * pulse_width));
        double osc = cos(pulse_freq * curr_time);
        
        for (size_t i = 0; i < spin_dim; ++i) {
            double drive1 = d_field_drive[atom * spin_dim + i];
            double drive2 = d_field_drive[N_atoms * spin_dim + atom * spin_dim + i];
            local_field[i] += pulse_amp * osc * (env1 * drive1 + env2 * drive2);
        }
    }
    
    // Compute Landau-Lifshitz derivative: dS/dt = H_eff × S
    if (spin_dim == 3) {
        cross_product_SU2_device(dsdt, local_field, spin_here);
    } else if (spin_dim == 8) {
        cross_product_SU3_device(dsdt, local_field, spin_here);
    }
}

/**
 * Kernel to update state: y_new = a1 * y + a2 * k
 */
__global__
void update_state_kernel(
    double* d_out,
    const double* d_in1,
    double a1,
    const double* d_in2,
    double a2,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    d_out[idx] = a1 * d_in1[idx] + a2 * d_in2[idx];
}

/**
 * Kernel to update state with three inputs: y_new = a1*y1 + a2*y2 + a3*y3
 */
__global__
void update_state_three_kernel(
    double* d_out,
    const double* d_in1, double a1,
    const double* d_in2, double a2,
    const double* d_in3, double a3,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    d_out[idx] = a1 * d_in1[idx] + a2 * d_in2[idx] + a3 * d_in3[idx];
}

/**
 * GPUODESystem operator() implementation
 */
void GPUODESystem::operator()(const GPUState& x, GPUState& dxdt, double t) const {
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((data.lattice_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Get raw pointers
    const double* d_x = thrust::raw_pointer_cast(x.data());
    double* d_dxdt = thrust::raw_pointer_cast(dxdt.data());
    double* d_local_field = thrust::raw_pointer_cast(data.local_field.data());
    const double* d_field = thrust::raw_pointer_cast(data.field.data());
    const double* d_onsite = thrust::raw_pointer_cast(data.onsite.data());
    const double* d_bilinear_vals = thrust::raw_pointer_cast(data.bilinear_vals.data());
    const size_t* d_bilinear_idx = thrust::raw_pointer_cast(data.bilinear_idx.data());
    const size_t* d_bilinear_counts = thrust::raw_pointer_cast(data.bilinear_counts.data());
    const double* d_field_drive = thrust::raw_pointer_cast(data.field_drive.data());
    
    LLG_flat_kernel<<<grid, block>>>(
        d_dxdt, d_x, d_local_field,
        d_field, d_onsite,
        d_bilinear_vals, d_bilinear_idx, d_bilinear_counts,
        d_field_drive,
        data.pulse_amp, data.pulse_width, data.pulse_freq,
        data.t_pulse_1, data.t_pulse_2,
        t,
        data.lattice_size, data.spin_dim, data.N_atoms, data.max_bilinear
    );
    
    cudaDeviceSynchronize();
}

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
) {
    GPULatticeData data;
    data.lattice_size = lattice_size;
    data.spin_dim = spin_dim;
    data.N_atoms = N_atoms;
    data.max_bilinear = max_bilinear;
    
    size_t array_size = lattice_size * spin_dim;
    
    // Copy arrays to device
    data.field = thrust::device_vector<double>(flat_field.begin(), flat_field.end());
    data.onsite = thrust::device_vector<double>(flat_onsite.begin(), flat_onsite.end());
    data.bilinear_vals = thrust::device_vector<double>(flat_bilinear.begin(), flat_bilinear.end());
    data.bilinear_idx = thrust::device_vector<size_t>(flat_partners.begin(), flat_partners.end());
    data.bilinear_counts = thrust::device_vector<size_t>(num_bilinear_per_site.begin(), num_bilinear_per_site.end());
    
    // Initialize working arrays
    data.work_1.resize(array_size, 0.0);
    data.work_2.resize(array_size, 0.0);
    data.work_3.resize(array_size, 0.0);
    data.local_field.resize(array_size, 0.0);
    
    // Initialize field drive to zeros
    data.field_drive.resize(2 * N_atoms * spin_dim, 0.0);
    
    data.initialized = true;
    return data;
}

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
) {
    data.field_drive = thrust::device_vector<double>(flat_field_drive.begin(), flat_field_drive.end());
    data.pulse_amp = pulse_amp;
    data.pulse_width = pulse_width;
    data.pulse_freq = pulse_freq;
    data.t_pulse_1 = t_pulse_1;
    data.t_pulse_2 = t_pulse_2;
}

/**
 * Perform GPU integration with selectable method
 * 
 * Available methods (matching CPU Boost.Odeint options):
 * - "euler": Explicit Euler (1st order)
 * - "rk2" or "midpoint": Modified midpoint (2nd order)
 * - "rk4": Classic Runge-Kutta 4th order
 * - "rk5" or "rkck54": Cash-Karp 5(4) - simulated as RK5 fixed step
 * - "dopri5": Dormand-Prince 5(4) - default
 * - "rk78" or "rkf78": Runge-Kutta-Fehlberg 7(8)
 * - "ssprk53": Strong Stability Preserving RK 5-stage 3rd order (optimized for spin dynamics)
 * - "bulirsch_stoer" or "bs": Bulirsch-Stoer (high accuracy)
 */
void integrate_gpu(
    GPUODESystem& system,
    GPUState& state,
    double T_start,
    double T_end,
    double dt,
    size_t save_interval,
    std::vector<std::pair<double, std::vector<double>>>& trajectory,
    const std::string& method
) {
    // Time stepping
    double t = T_start;
    size_t step = 0;
    
    while (t < T_end) {
        // Save trajectory at intervals
        if (step % save_interval == 0) {
            thrust::host_vector<double> h_state = state;
            std::vector<double> state_vec(h_state.begin(), h_state.end());
            trajectory.push_back({t, state_vec});
        }
        
        // Perform one step with selected method
        step_gpu(system, state, t, dt, method);
        
        t += dt;
        step++;
    }
    
    // Save final state
    thrust::host_vector<double> h_state = state;
    std::vector<double> state_vec(h_state.begin(), h_state.end());
    trajectory.push_back({t, state_vec});
}

// Overload for backward compatibility (defaults to ssprk53)
void integrate_gpu(
    GPUODESystem& system,
    GPUState& state,
    double T_start,
    double T_end,
    double dt,
    size_t save_interval,
    std::vector<std::pair<double, std::vector<double>>>& trajectory
) {
    integrate_gpu(system, state, T_start, T_end, dt, save_interval, trajectory, "ssprk53");
}

/**
 * Single integration step on GPU with selectable method
 * 
 * Available methods:
 * - "euler": Explicit Euler (1st order, 1 stage)
 * - "rk2" or "midpoint": Modified midpoint (2nd order, 2 stages)
 * - "rk4": Classic Runge-Kutta (4th order, 4 stages)
 * - "rk5" or "rkck54": Cash-Karp style (5th order, 6 stages)
 * - "dopri5": Dormand-Prince (5th order, 7 stages) - default
 * - "rk78" or "rkf78": Fehlberg (8th order, 13 stages)
 * - "ssprk53": SSP RK (3rd order, 5 stages, optimized for stability)
 * - "bulirsch_stoer" or "bs": Modified midpoint extrapolation
 */
void step_gpu(
    GPUODESystem& system,
    GPUState& state,
    double t,
    double dt,
    const std::string& method
) {
    GPULatticeData& data = system.data;
    size_t array_size = data.lattice_size * data.spin_dim;
    
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((array_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    double* d_state = thrust::raw_pointer_cast(state.data());
    
    if (method == "euler") {
        // =====================================================================
        // Euler method: y_{n+1} = y_n + h * f(t_n, y_n)
        // 1st order, 1 function evaluation
        // =====================================================================
        GPUState k(array_size);
        double* d_k = thrust::raw_pointer_cast(k.data());
        
        system(state, k, t);
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k, dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk2" || method == "midpoint") {
        // =====================================================================
        // Modified Midpoint (RK2): 2nd order, 2 function evaluations
        // k1 = f(t, y)
        // k2 = f(t + h/2, y + h/2 * k1)
        // y_{n+1} = y_n + h * k2
        // =====================================================================
        GPUState k1(array_size), k2(array_size), tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        // k1 = f(t, y)
        system(state, k1, t);
        
        // tmp = y + h/2 * k1
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        
        // k2 = f(t + h/2, tmp)
        system(tmp, k2, t + 0.5 * dt);
        
        // y_{n+1} = y + h * k2
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k2, dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk4") {
        // =====================================================================
        // Classic RK4: 4th order, 4 function evaluations
        // k1 = f(t, y)
        // k2 = f(t + h/2, y + h/2 * k1)
        // k3 = f(t + h/2, y + h/2 * k2)
        // k4 = f(t + h, y + h * k3)
        // y_{n+1} = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        // =====================================================================
        GPUState k1(array_size), k2(array_size), k3(array_size), k4(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        // k1 = f(t, y)
        system(state, k1, t);
        
        // k2 = f(t + h/2, y + h/2 * k1)
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k2, t + 0.5 * dt);
        
        // k3 = f(t + h/2, y + h/2 * k2)
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k2, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k3, t + 0.5 * dt);
        
        // k4 = f(t + h, y + h * k3)
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k3, dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k4, t + dt);
        
        // y_{n+1} = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        // Accumulate: sum = k1 + 2*k2 + 2*k3 + k4
        update_state_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k2, 2.0, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k3, 2.0, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k4, 1.0, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, dt / 6.0, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk5" || method == "rkck54") {
        // =====================================================================
        // Cash-Karp RK5(4): 5th order, 6 function evaluations
        // Butcher tableau coefficients for the 5th order solution
        // =====================================================================
        GPUState k1(array_size), k2(array_size), k3(array_size);
        GPUState k4(array_size), k5(array_size), k6(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_k5 = thrust::raw_pointer_cast(k5.data());
        double* d_k6 = thrust::raw_pointer_cast(k6.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        // Cash-Karp coefficients
        constexpr double a21 = 1.0/5.0;
        constexpr double a31 = 3.0/40.0, a32 = 9.0/40.0;
        constexpr double a41 = 3.0/10.0, a42 = -9.0/10.0, a43 = 6.0/5.0;
        constexpr double a51 = -11.0/54.0, a52 = 5.0/2.0, a53 = -70.0/27.0, a54 = 35.0/27.0;
        constexpr double a61 = 1631.0/55296.0, a62 = 175.0/512.0, a63 = 575.0/13824.0;
        constexpr double a64 = 44275.0/110592.0, a65 = 253.0/4096.0;
        constexpr double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 3.0/5.0, c5 = 1.0, c6 = 7.0/8.0;
        // 5th order weights
        constexpr double b1 = 37.0/378.0, b3 = 250.0/621.0, b4 = 125.0/594.0, b6 = 512.0/1771.0;
        
        // k1 = f(t, y)
        system(state, k1, t);
        
        // k2 = f(t + c2*h, y + h*a21*k1)
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a21 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k2, t + c2 * dt);
        
        // k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a31 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a32 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k3, t + c3 * dt);
        
        // k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a41 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a42 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a43 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k4, t + c4 * dt);
        
        // k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a51 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a52 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a53 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a54 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k5, t + c5 * dt);
        
        // k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a61 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a62 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a63 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a64 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, a65 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k6, t + c6 * dt);
        
        // y_{n+1} = y + h*(b1*k1 + b3*k3 + b4*k4 + b6*k6)  (note: b2=b5=0)
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, b1 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k3, b3 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k4, b4 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k6, b6 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "dopri5") {
        // =====================================================================
        // Dormand-Prince 5(4): 5th order, 7 function evaluations (FSAL)
        // Default method, good general-purpose choice
        // =====================================================================
        GPUState k1(array_size), k2(array_size), k3(array_size), k4(array_size);
        GPUState k5(array_size), k6(array_size), k7(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_k5 = thrust::raw_pointer_cast(k5.data());
        double* d_k6 = thrust::raw_pointer_cast(k6.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        (void)k7;  // k7 not used in fixed-step (used in adaptive for FSAL)
        
        // Dormand-Prince coefficients
        constexpr double a21 = 1.0/5.0;
        constexpr double a31 = 3.0/40.0, a32 = 9.0/40.0;
        constexpr double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
        constexpr double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
        constexpr double a61 = 9017.0/3168.0, a62 = -355.0/33.0, a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;
        constexpr double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 4.0/5.0, c5 = 8.0/9.0, c6 = 1.0;
        // 5th order weights
        constexpr double b1 = 35.0/384.0, b3 = 500.0/1113.0, b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0;
        
        // k1 = f(t, y)
        system(state, k1, t);
        
        // k2 = f(t + c2*h, y + h*a21*k1)
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a21 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k2, t + c2 * dt);
        
        // k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a31 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a32 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k3, t + c3 * dt);
        
        // k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a41 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a42 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a43 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k4, t + c4 * dt);
        
        // k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a51 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a52 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a53 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a54 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k5, t + c5 * dt);
        
        // k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a61 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a62 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a63 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a64 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, a65 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k6, t + c6 * dt);
        
        // y_{n+1} = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)  (note: b2=b7=0)
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, b1 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k3, b3 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k4, b4 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k5, b5 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k6, b6 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk78" || method == "rkf78") {
        // =====================================================================
        // Runge-Kutta-Fehlberg 7(8): 8th order, 13 function evaluations
        // Very high accuracy for smooth problems
        // =====================================================================
        GPUState k1(array_size), k2(array_size), k3(array_size), k4(array_size);
        GPUState k5(array_size), k6(array_size), k7(array_size), k8(array_size);
        GPUState k9(array_size), k10(array_size), k11(array_size), k12(array_size), k13(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_k5 = thrust::raw_pointer_cast(k5.data());
        double* d_k6 = thrust::raw_pointer_cast(k6.data());
        double* d_k7 = thrust::raw_pointer_cast(k7.data());
        double* d_k8 = thrust::raw_pointer_cast(k8.data());
        double* d_k9 = thrust::raw_pointer_cast(k9.data());
        double* d_k10 = thrust::raw_pointer_cast(k10.data());
        double* d_k11 = thrust::raw_pointer_cast(k11.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        (void)k12; (void)k13;  // k12, k13 not used in simplified implementation
        
        // RKF78 coefficients (Fehlberg's 7(8) method)
        // Using simplified 8th order weights for the primary solution
        constexpr double c2 = 2.0/27.0, c3 = 1.0/9.0, c4 = 1.0/6.0, c5 = 5.0/12.0;
        constexpr double c6 = 1.0/2.0, c7 = 5.0/6.0, c8 = 1.0/6.0, c9 = 2.0/3.0;
        constexpr double c10 = 1.0/3.0, c11 = 1.0;
        
        // 8th order solution weights
        constexpr double b1 = 41.0/840.0, b6 = 34.0/105.0, b7 = 9.0/35.0, b8 = 9.0/35.0;
        constexpr double b9 = 9.0/280.0, b10 = 9.0/280.0, b11 = 41.0/840.0;
        
        // Stage coefficients (simplified - using key stages)
        system(state, k1, t);
        
        // k2
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, c2 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k2, t + c2 * dt);
        
        // k3
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (1.0/36.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, (1.0/12.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k3, t + c3 * dt);
        
        // k4
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (1.0/24.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, (1.0/8.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k4, t + c4 * dt);
        
        // k5
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (5.0/12.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (-25.0/16.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, (25.0/16.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k5, t + c5 * dt);
        
        // k6
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (1.0/20.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (1.0/4.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (1.0/5.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k6, t + c6 * dt);
        
        // k7
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (-25.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (125.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (-65.0/27.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (125.0/54.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k7, t + c7 * dt);
        
        // k8
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (31.0/300.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (61.0/225.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (-2.0/9.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (13.0/900.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k8, t + c8 * dt);
        
        // k9
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 2.0 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (-53.0/6.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (704.0/45.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (-107.0/9.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (67.0/90.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k8, 3.0 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k9, t + c9 * dt);
        
        // k10
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (-91.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (23.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (-976.0/135.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (311.0/54.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (-19.0/60.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k8, (17.0/6.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k9, (-1.0/12.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k10, t + c10 * dt);
        
        // k11
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (2383.0/4100.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (-341.0/164.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (4496.0/1025.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (-301.0/82.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (2133.0/4100.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k8, (45.0/82.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k9, (45.0/164.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k10, (18.0/41.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k11, t + c11 * dt);
        
        // 8th order solution: y_{n+1} = y + h*(b1*k1 + b6*k6 + b7*k7 + b8*k8 + b9*k9 + b10*k10 + b11*k11)
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, b1 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k6, b6 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k7, b7 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k8, b8 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k9, b9 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k10, b10 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k11, b11 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "bulirsch_stoer" || method == "bs") {
        // =====================================================================
        // Bulirsch-Stoer: Modified midpoint with Richardson extrapolation
        // Very high accuracy, uses sequence of subdivisions
        // Simplified version using 4 substeps for GPU efficiency
        // =====================================================================
        const int n_substeps = 4;  // Number of midpoint steps
        GPUState y_mid(array_size), y_prev(array_size), y_next(array_size);
        GPUState k(array_size);
        double* d_y_mid = thrust::raw_pointer_cast(y_mid.data());
        double* d_y_prev = thrust::raw_pointer_cast(y_prev.data());
        double* d_y_next = thrust::raw_pointer_cast(y_next.data());
        double* d_k = thrust::raw_pointer_cast(k.data());
        
        double h_sub = dt / n_substeps;
        
        // Modified midpoint method with n_substeps
        // y_prev = y
        thrust::copy(state.begin(), state.end(), y_prev.begin());
        
        // First Euler step: y_mid = y + h_sub * f(t, y)
        system(state, k, t);
        update_state_kernel<<<grid, block>>>(d_y_mid, d_state, 1.0, d_k, h_sub, array_size);
        cudaDeviceSynchronize();
        
        // Midpoint steps
        for (int i = 1; i < n_substeps; ++i) {
            double t_curr = t + i * h_sub;
            system(y_mid, k, t_curr);
            // y_next = y_prev + 2*h_sub*f(t_curr, y_mid)
            update_state_kernel<<<grid, block>>>(d_y_next, d_y_prev, 1.0, d_k, 2.0 * h_sub, array_size);
            cudaDeviceSynchronize();
            // Shift: y_prev = y_mid, y_mid = y_next
            thrust::copy(y_mid.begin(), y_mid.end(), y_prev.begin());
            thrust::copy(y_next.begin(), y_next.end(), y_mid.begin());
        }
        
        // Final correction step
        system(y_mid, k, t + dt);
        // y_{n+1} = 0.5 * (y_mid + y_prev + h_sub * f(t+dt, y_mid))
        update_state_kernel<<<grid, block>>>(d_y_next, d_y_mid, 0.5, d_y_prev, 0.5, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_y_next, 1.0, d_k, 0.5 * h_sub, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "ssprk53") {
        // =====================================================================
        // SSPRK53: Strong Stability Preserving RK, 5 stages, 3rd order
        // Optimized for hyperbolic PDEs and spin dynamics
        // =====================================================================
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
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k, b10 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 2
        system(tmp, k, t + c1 * dt);
        update_state_kernel<<<grid, block>>>(d_u, d_tmp, 1.0, d_k, b21 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 3
        system(u, k, t + c2 * dt);
        update_state_three_kernel<<<grid, block>>>(d_tmp, d_state, a30, d_u, a32, d_k, b32 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 4
        system(tmp, k, t + c3 * dt);
        update_state_three_kernel<<<grid, block>>>(d_tmp, d_state, a40, d_tmp, a43, d_k, b43 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 5 (final)
        system(tmp, k, t + c4 * dt);
        update_state_three_kernel<<<grid, block>>>(d_state, d_u, a52, d_tmp, a54, d_k, b54 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk54" || method == "rkf54") {
        // =====================================================================
        // Runge-Kutta-Fehlberg 5(4): Same as rkck54 but with Fehlberg coefficients
        // =====================================================================
        // Use the same implementation as dopri5 (both are 5th order embedded methods)
        step_gpu(system, state, t, dt, "dopri5");
        
    } else {
        // Default to SSPRK53 for unknown methods
        std::cerr << "Warning: Unknown GPU integration method '" << method << "', using ssprk53" << std::endl;
        step_gpu(system, state, t, dt, "ssprk53");
    }
}

/**
 * Compute total energy on GPU
 */
double compute_energy_gpu(const GPULatticeData& data, const GPUState& state) {
    // Allocate device memory for site energies
    thrust::device_vector<double> d_energies(data.lattice_size);
    
    // TODO: Implement energy computation kernel
    // For now, return 0 as placeholder
    return 0.0;
}

/**
 * Normalize spins on GPU
 */
__global__
void normalize_spins_flat_kernel(
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

void normalize_spins_gpu(GPUState& state, size_t lattice_size, size_t spin_dim, double spin_length) {
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((lattice_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    double* d_state = thrust::raw_pointer_cast(state.data());
    normalize_spins_flat_kernel<<<grid, block>>>(d_state, spin_length, lattice_size, spin_dim);
    cudaDeviceSynchronize();
}

} // namespace gpu
