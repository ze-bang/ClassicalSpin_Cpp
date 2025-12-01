#ifndef GPU_COMMON_HELPERS_CUH
#define GPU_COMMON_HELPERS_CUH

#include <cuda_runtime.h>
#include <cstddef>

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

// ======================= Device Helper Functions (inline for cross-TU use) =======================

/**
 * Compute dot product of two vectors
 */
__device__ __forceinline__
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

/**
 * Compute bilinear contraction: spin1 · matrix · spin2
 */
__device__ __forceinline__
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

/**
 * Compute trilinear contraction: tensor · spin1 · spin2 · spin3
 */
__device__ __forceinline__
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

/**
 * Matrix-vector multiplication: result = matrix * vector
 */
__device__ __forceinline__
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

/**
 * Compute trilinear field contribution: result = T · spin1 · spin2 for each component
 */
__device__ __forceinline__
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

/**
 * Cross product for SU(2) spins (3 components)
 */
__device__ __forceinline__
void cross_product_SU2_device(double* result, const double* a, const double* b) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

/**
 * Cross product for SU(3) spins (8 components using Gell-Mann structure constants)
 */
__device__ __forceinline__
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
 * Initialize local field with negative external field: H = -B
 */
__device__ __forceinline__
void init_local_field_device(double* local_field, const double* external_field, size_t spin_dim) {
    for (size_t i = 0; i < spin_dim; ++i) {
        local_field[i] = -external_field[i];
    }
}

/**
 * Add onsite contribution to local field: H += scale * J_onsite * S
 * For quadratic onsite energy E = S^T A S, the derivative gives scale=2
 */
__device__ __forceinline__
void add_onsite_contribution_device(double* local_field, const double* onsite_matrix, 
                                     const double* spin, double* temp, size_t spin_dim,
                                     double scale = 2.0) {
    multiply_matrix_vector_device(temp, onsite_matrix, spin, spin_dim);
    for (size_t i = 0; i < spin_dim; ++i) {
        local_field[i] += scale * temp[i];
    }
}

/**
 * Add bilinear contribution to local field: H += J * S_partner
 */
__device__ __forceinline__
void add_bilinear_contribution_device(double* local_field, const double* bilinear_matrix,
                                       const double* partner_spin, double* temp, size_t spin_dim) {
    multiply_matrix_vector_device(temp, bilinear_matrix, partner_spin, spin_dim);
    for (size_t i = 0; i < spin_dim; ++i) {
        local_field[i] += temp[i];
    }
}

/**
 * Add trilinear contribution to local field: H += T * S1 * S2
 */
__device__ __forceinline__
void add_trilinear_contribution_device(double* local_field, const double* tensor,
                                        const double* spin1, const double* spin2,
                                        double* temp, size_t spin_dim) {
    contract_trilinear_field_device(temp, tensor, spin1, spin2, spin_dim);
    for (size_t i = 0; i < spin_dim; ++i) {
        local_field[i] += temp[i];
    }
}

/**
 * Add drive field contribution to local field
 */
__device__ __forceinline__
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
            local_field[i] -= amplitude * osc * (
                env1 * field_drive_1[atom * spin_dim + i] +
                env2 * field_drive_2[atom * spin_dim + i]
            );
        }
    }
}

/**
 * Compute Landau-Lifshitz derivative: dsdt = H_eff × spin
 */
__device__ __forceinline__
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

// ======================= Common GPU Kernels =======================

/**
 * Update array: out = a1*in1 + a2*in2
 */
__global__
void update_arrays_kernel(
    double* out,
    const double* in1, double a1,
    const double* in2, double a2,
    size_t size
);

/**
 * Update array: out = a1*in1 + a2*in2 + a3*in3
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
 * Normalize spins to specified length
 */
__global__
void normalize_spins_kernel(
    double* d_spins,
    double spin_length,
    size_t lattice_size,
    size_t spin_dim
);

/**
 * Compute magnetization (local and staggered) per component
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

#endif // GPU_COMMON_HELPERS_CUH
