#include "molecular_dynamics.cuh"
#include "kernel_params.cuh"

// CUDA device functions for vector operations
__device__
double dot_device(const double* a, const double* b, size_t n) {
    double result = 0.0;
    
    // Unroll loop for small, compile-time known sizes
    if (n <= 4) {
        switch (n) {
            case 4: result += a[3] * b[3]; [[fallthrough]];
            case 3: result += a[2] * b[2]; [[fallthrough]];
            case 2: result += a[1] * b[1]; [[fallthrough]];
            case 1: result += a[0] * b[0]; break;
            default: break;
        }
    } else {
        // Standard loop for larger sizes, use #pragma unroll for small constant sizes
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
    
    // Optimize for common small matrix sizes (SU(2) = 3x3, SU(3) = 8x8)
    if (n == 3) {
        // Manually unrolled 3x3 matrix contraction
        result = spin1[0] * (matrix[0] * spin2[0] + matrix[1] * spin2[1] + matrix[2] * spin2[2]) +
                 spin1[1] * (matrix[3] * spin2[0] + matrix[4] * spin2[1] + matrix[5] * spin2[2]) +
                 spin1[2] * (matrix[6] * spin2[0] + matrix[7] * spin2[1] + matrix[8] * spin2[2]);
    } else if (n == 8) {
        // Use register blocking for 8x8 matrix
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
        // General case with loop unrolling
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
double contract_trilinear_device(const double* tensor, const double* spin1, const double* spin2, const double* spin3, size_t n) {
    double result = 0.0;
    
    if (n == 3) {
        // Fully unrolled 3x3x3 trilinear contraction for SU(2)
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
        // Register-blocked 8x8x8 trilinear contraction for SU(3)
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
        // General case with optimized memory access pattern
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
        // Optimized 3x3 matrix-vector multiplication
        result[0] = matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2];
        result[1] = matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2];
        result[2] = matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2];
    } else if (n == 8) {
        // Optimized 8x8 matrix-vector multiplication with unrolling
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
        // General case with pragma unroll
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
void contract_trilinear_field_device(double* result, const double* tensor, const double* spin1, const double* spin2, size_t n) {
    if (n == 3) {
        // Optimized 3x3x3 trilinear field contraction for SU(2)
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
        // Register-blocked 8x8x8 trilinear field contraction for SU(3)
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
        // General case with optimized access pattern
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
    constexpr double sqrt3_2 = 0.86602540378443864676372317075294; // sqrt(3)/2 - more precision
    
    // Component 0 (λ_1): receives contributions from f_{123}, f_{147}, f_{156}
    result[0] = a[1]*b[2] - a[2]*b[1] + 0.5*(a[3]*b[6] - a[6]*b[3]) - 0.5*(a[4]*b[5] - a[5]*b[4]);
    
    // Component 1 (λ_2): receives contributions from f_{213}, f_{246}, f_{257}
    result[1] = a[2]*b[0] - a[0]*b[2] + 0.5*(a[3]*b[5] - a[5]*b[3]) + 0.5*(a[4]*b[6] - a[6]*b[4]);
    
    // Component 2 (λ_3): receives contributions from f_{312}, f_{345}, f_{367}
    result[2] = a[0]*b[1] - a[1]*b[0] + 0.5*(a[3]*b[4] - a[4]*b[3]) - 0.5*(a[5]*b[6] - a[6]*b[5]);
    
    // Component 3 (λ_4): receives contributions from f_{147}, f_{246}, f_{345}, f_{458}
    result[3] = 0.5*(a[6]*b[0] - a[0]*b[6]) + 0.5*(a[5]*b[1] - a[1]*b[5]) + 0.5*(a[4]*b[2] - a[2]*b[4]) + sqrt3_2*(a[4]*b[7] - a[7]*b[4]);
    
    // Component 4 (λ_5): receives contributions from f_{156}, f_{257}, f_{345}, f_{458}
    result[4] = -0.5*(a[5]*b[0] - a[0]*b[5]) + 0.5*(a[6]*b[1] - a[1]*b[6]) + 0.5*(a[2]*b[3] - a[3]*b[2]) + sqrt3_2*(a[7]*b[3] - a[3]*b[7]);
    
    // Component 5 (λ_6): receives contributions from f_{156}, f_{246}, f_{367}, f_{678}
    result[5] = -0.5*(a[0]*b[4] - a[4]*b[0]) + 0.5*(a[1]*b[3] - a[3]*b[1]) - 0.5*(a[6]*b[2] - a[2]*b[6]) + sqrt3_2*(a[6]*b[7] - a[7]*b[6]);
    
    // Component 6 (λ_7): receives contributions from f_{147}, f_{257}, f_{367}, f_{678}
    result[6] = 0.5*(a[0]*b[3] - a[3]*b[0]) + 0.5*(a[1]*b[4] - a[4]*b[1]) - 0.5*(a[2]*b[5] - a[5]*b[2]) + sqrt3_2*(a[7]*b[5] - a[5]*b[7]);
    
    // Component 7 (λ_8): receives contributions from f_{458}, f_{678}
    result[7] = sqrt3_2*(a[3]*b[4] - a[4]*b[3]) + sqrt3_2*(a[5]*b[6] - a[6]*b[5]);
}



template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
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
    energy += contract_device(spin_here, &onsite_interaction_SU2[site_index * N_SU2 * N_SU2], spin_here, N_SU2)/2;
    
    // Bilinear interactions
    for (size_t i = 0; i < num_bi_SU2 && i < max_bi_neighbors; ++i) {
        size_t partner = bilinear_partners_SU2[site_index * max_bi_neighbors + i];
        if (partner < lattice_size_SU2) {
            energy += contract_device(spin_here, 
                &bilinear_interaction_SU2[site_index * max_bi_neighbors * N_SU2 * N_SU2 + i * N_SU2 * N_SU2],
                &spins_SU2[partner * N_SU2], N_SU2)/2;
        }
    }
    
    // Trilinear interactions
    for (size_t i = 0; i < num_tri_SU2 && i < max_tri_neighbors; ++i) {
        size_t partner1 = trilinear_partners_SU2[site_index * max_tri_neighbors * 2 + i * 2];
        size_t partner2 = trilinear_partners_SU2[site_index * max_tri_neighbors * 2 + i * 2 + 1];
        if (partner1 < lattice_size_SU2 && partner2 < lattice_size_SU2) {
            energy += contract_trilinear_device(
                &trilinear_interaction_SU2[site_index * max_tri_neighbors * N_SU2 * N_SU2 * N_SU2 + i * N_SU2 * N_SU2 * N_SU2],
                spin_here, &spins_SU2[partner1 * N_SU2], &spins_SU2[partner2 * N_SU2], N_SU2)/3;
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
                            spin_here[a] * spins_SU2[partner1 * N_SU2 + b] * spins_SU3[partner2 * N_SU3 + c]/3;
                    }
                }
            }
            energy += partial_energy;
        }
    }
    
    d_energies[site_index] = energy;
}

template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
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
    energy += contract_device(spin_here, &onsite_interaction_SU3[site_index * N_SU3 * N_SU3], spin_here, N_SU3)/2;

    // Bilinear interactions
    for (size_t i = 0; i < num_bi_SU3 && i < max_bi_neighbors; ++i) {
        size_t partner = bilinear_partners_SU3[site_index * max_bi_neighbors + i];
        if (partner < lattice_size_SU3) {
            energy += contract_device(spin_here, 
                &bilinear_interaction_SU3[site_index * max_bi_neighbors * N_SU3 * N_SU3 + i * N_SU3 * N_SU3],
                &spins_SU3[partner * N_SU3], N_SU3)/2;
        }
    }
    
    // Trilinear interactions
    for (size_t i = 0; i < num_tri_SU3 && i < max_tri_neighbors; ++i) {
        size_t partner1 = trilinear_partners_SU3[site_index * max_tri_neighbors * 2 + i * 2];
        size_t partner2 = trilinear_partners_SU3[site_index * max_tri_neighbors * 2 + i * 2 + 1];
        if (partner1 < lattice_size_SU3 && partner2 < lattice_size_SU3) {
            energy += contract_trilinear_device(
                &trilinear_interaction_SU3[site_index * max_tri_neighbors * N_SU3 * N_SU3 * N_SU3 + i * N_SU3 * N_SU3 * N_SU3],
                spin_here, &spins_SU3[partner1 * N_SU3], &spins_SU3[partner2 * N_SU3], N_SU3)/3;
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
                            spin_here[a] * spins_SU2[partner1 * N_SU2 + b] * spins_SU2[partner2 * N_SU2 + c]/3;
                    }
                }
            }
            energy += partial_energy;
        }
    }
    
    d_energies[site_index] = energy;
}

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
__host__
double mixed_lattice_cuda<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>::total_energy_cuda() {
    double total_energy = 0.0;
    
    // Allocate device memory for energies
    double *d_energies_SU2, *d_energies_SU3;
    cudaMalloc(&d_energies_SU2, lattice_size_SU2 * sizeof(double));
    cudaMalloc(&d_energies_SU3, lattice_size_SU3 * sizeof(double));
    
    // Launch kernels
    dim3 block(256);
    dim3 grid_SU2((lattice_size_SU2 + block.x - 1) / block.x);
    dim3 grid_SU3((lattice_size_SU3 + block.x - 1) / block.x);
    
    compute_site_energy_SU2_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> <<<grid_SU2, block>>>(
        d_energies_SU2, d_spins.spins_SU2, d_field_SU2, d_onsite_interaction_SU2,
        d_bilinear_interaction_SU2, d_bilinear_partners_SU2,
        d_trilinear_interaction_SU2, d_trilinear_partners_SU2,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2,
        d_spins.spins_SU3, d_num_bi_SU2, d_num_tri_SU2, d_num_tri_SU2_SU3,
        max_bilinear_neighbors_SU2, max_trilinear_neighbors_SU2, max_mixed_trilinear_neighbors_SU2
    );
    
    compute_site_energy_SU3_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> <<<grid_SU3, block>>>(
        d_energies_SU3, d_spins.spins_SU3, d_field_SU3, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU3, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU3, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3,
        d_spins.spins_SU2, d_num_bi_SU3, d_num_tri_SU3, d_num_tri_SU2_SU3,
        max_bilinear_neighbors_SU3, max_trilinear_neighbors_SU3, max_mixed_trilinear_neighbors_SU3
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
    return energy_SU2 + energy_SU3;
}


// Kernel to update arrays with linear combination
template <size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
__global__ void update_arrays_kernel(double* out_SU2, const double* in1_SU2, double a1_SU2, 
                                    const double* in2_SU2, double a2_SU2,
                                    double* out_SU3, const double* in1_SU3, double a1_SU3, 
                                    const double* in2_SU3, double a2_SU3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Ensure we do not exceed the bounds of either SU2 or SU3 arrays
    if (idx >= lattice_size_SU2 + lattice_size_SU3) return;
    if (idx < lattice_size_SU2) {
        // Update SU2 array
        for (int i = 0; i < N_SU2; ++i) {
            out_SU2[idx * N_SU2 + i] = a1_SU2 * in1_SU2[idx * N_SU2 + i] + a2_SU2 * in2_SU2[idx * N_SU2 + i];
        }
    }
    else{
        int idx_SU3 = idx - lattice_size_SU2; // Adjust index for SU3
        for (int i = 0; i < N_SU3; ++i) {
            out_SU3[idx_SU3 * N_SU3 + i] = a1_SU3 * in1_SU3[idx_SU3 * N_SU3 + i] + a2_SU3 * in2_SU3[idx_SU3 * N_SU3 + i];
        }
    }
}

// Kernel to update arrays with linear combination of three arrays
template <size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
__global__ void update_arrays_three_kernel(double* out_SU2, 
                                          const double* in1_SU2, double a1_SU2,
                                          const double* in2_SU2, double a2_SU2,
                                          const double* in3_SU2, double a3_SU2,
                                          double* out_SU3, 
                                          const double* in1_SU3, double a1_SU3,
                                          const double* in2_SU3, double a2_SU3,
                                          const double* in3_SU3, double a3_SU3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Ensure we do not exceed the bounds of either SU2 or SU3 arrays
    if (idx >= lattice_size_SU2 + lattice_size_SU3) return;
    if (idx < lattice_size_SU2) {
        for (int i = 0; i < N_SU2; ++i) {
            out_SU2[idx * N_SU2 + i] = a1_SU2 * in1_SU2[idx * N_SU2 + i] + 
                                       a2_SU2 * in2_SU2[idx * N_SU2 + i] + 
                                       a3_SU2 * in3_SU2[idx * N_SU2 + i];
        }
    }
    else{
        int idx_SU3 = idx - lattice_size_SU2; // Adjust index for SU3
        for (int i = 0; i < N_SU3; ++i) {
            out_SU3[idx_SU3 * N_SU3 + i] = a1_SU3 * in1_SU3[idx_SU3 * N_SU3 + i] + 
                                        a2_SU3 * in2_SU3[idx_SU3 * N_SU3 + i] + 
                                        a3_SU3 * in3_SU3[idx_SU3 * N_SU3 + i];
        }
    }
}

// Kernel to normalize spins for SU2
template <size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
__global__ void normalize_spins_kernel(double* spins_SU2, double spin_length_SU2,
                                       double* spins_SU3, double spin_length_SU3) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site < lattice_size_SU2) {
        double* spin = &spins_SU2[site * N_SU2];
        double norm = 0.0;
        for (int i = 0; i < N_SU2; i++) {
            norm += spin[i] * spin[i];
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (int i = 0; i < N_SU2; i++) {
                spin[i] = spin[i] * spin_length_SU2 / norm;
            }
        }
    }
    else{
        site -= lattice_size_SU2; // Adjust index for SU3
        double* spin = &spins_SU3[site * N_SU3];
        double norm = 0.0;
        for (int i = 0; i < N_SU3; i++) {
            norm += spin[i] * spin[i];
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (int i = 0; i < N_SU3; i++) {
                spin[i] = spin[i] * spin_length_SU3 / norm;
            }
        }
    }
}


template<size_t N_SU2, size_t N_ATOMS_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t lattice_size_SU3>
__global__
void LLG_kernel(
    RKWorkArrays<N_SU2, N_SU3> work_arrays,
    SU2_DeviceParams<N_SU2, lattice_size_SU2> su2_params,
    SU3_DeviceParams<N_SU3, lattice_size_SU3> su3_params,
    MixedInteractionParams<N_SU2, N_SU3> mixed_params,
    DriveFieldParams drive_params)
{
    int site_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (site_index > lattice_size_SU2 + lattice_size_SU3) return;

    if (site_index < lattice_size_SU2) {
        compute_local_field_SU2<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3>(
            su2_params.local_field, site_index, su2_params.spins, su2_params.field, su2_params.onsite_interaction,
            su2_params.bilinear_interaction, su2_params.bilinear_partners,
            su2_params.trilinear_interaction, su2_params.trilinear_partners,
            mixed_params.mixed_bilinear_interaction_SU2, mixed_params.mixed_bilinear_partners_SU2,
            mixed_params.mixed_trilinear_interaction_SU2, mixed_params.mixed_trilinear_partners_SU2,
            su3_params.spins, su2_params.num_bi, su2_params.num_tri, mixed_params.num_bi_SU2_SU3, mixed_params.num_tri_SU2_SU3,
            su2_params.max_bi_neighbors, su2_params.max_tri_neighbors, 
            mixed_params.max_mixed_bi_neighbors_SU2, mixed_params.max_mixed_tri_neighbors_SU2);

        // Add drive field if applicable
        drive_field_T_SU2<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3>(
            su2_params.local_field, drive_params.curr_time, site_index, su2_params.field_drive_1, su2_params.field_drive_2, 
            drive_params.amp, drive_params.width, drive_params.freq, drive_params.t_B_1, drive_params.t_B_2,
            mixed_params.max_mixed_tri_neighbors_SU2, mixed_params.mixed_trilinear_interaction_SU2, 
            mixed_params.mixed_trilinear_partners_SU2, su3_params.spins);
        
        // Compute derivatives (dS/dt) using Landau-Lifshitz equation
        landau_Lifshitz_SU2<N_SU2, lattice_size_SU2>(
            work_arrays.k_SU2, site_index, su2_params.local_field, su2_params.spins);
    
    } else {
        size_t site_index_SU3 = site_index - lattice_size_SU2;        
        compute_local_field_SU3<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3>(
            su3_params.local_field, site_index_SU3, su3_params.spins, su3_params.field, su3_params.onsite_interaction,
            su3_params.bilinear_interaction, su3_params.bilinear_partners,
            su3_params.trilinear_interaction, su3_params.trilinear_partners,
            mixed_params.mixed_bilinear_interaction_SU3, mixed_params.mixed_bilinear_partners_SU3,
            mixed_params.mixed_trilinear_interaction_SU3, mixed_params.mixed_trilinear_partners_SU3,
            su2_params.spins, su3_params.num_bi, su3_params.num_tri, mixed_params.num_bi_SU2_SU3, mixed_params.num_tri_SU2_SU3,
            su3_params.max_bi_neighbors, su3_params.max_tri_neighbors,
            mixed_params.max_mixed_bi_neighbors_SU3, mixed_params.max_mixed_tri_neighbors_SU3);

        drive_field_T_SU3<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3>(
            su3_params.local_field, drive_params.curr_time, site_index_SU3, su3_params.field_drive_1, su3_params.field_drive_2, 
            drive_params.amp, drive_params.width, drive_params.freq, drive_params.t_B_1, drive_params.t_B_2,
            mixed_params.max_mixed_tri_neighbors_SU3, mixed_params.mixed_trilinear_interaction_SU3, 
            mixed_params.mixed_trilinear_partners_SU3, su2_params.spins);
            
        landau_Lifshitz_SU3<N_SU3, lattice_size_SU3>(
            work_arrays.k_SU3, site_index_SU3, su3_params.local_field, su3_params.spins);
    }
}




// Memory pool class for SSPRK53 working arrays - avoids repeated allocation/deallocation
template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
class SSPRK53_MemoryPool {
private:
    double* work_SU2_1;
    double* work_SU2_2;
    double* work_SU2_3;
    double* work_SU3_1;
    double* work_SU3_2;
    double* work_SU3_3;
    bool allocated;

public:
    SSPRK53_MemoryPool() : allocated(false) {
        allocate();
    }
    
    ~SSPRK53_MemoryPool() {
        deallocate();
    }
    
    void allocate() {
        if (allocated) return;
        
        cudaMalloc(&work_SU2_1, lattice_size_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&work_SU2_2, lattice_size_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&work_SU2_3, lattice_size_SU2 * N_SU2 * sizeof(double));
        cudaMalloc(&work_SU3_1, lattice_size_SU3 * N_SU3 * sizeof(double));
        cudaMalloc(&work_SU3_2, lattice_size_SU3 * N_SU3 * sizeof(double));
        cudaMalloc(&work_SU3_3, lattice_size_SU3 * N_SU3 * sizeof(double));
        allocated = true;
    }
    
    void deallocate() {
        if (!allocated) return;
        
        cudaFree(work_SU2_1);
        cudaFree(work_SU2_2);
        cudaFree(work_SU2_3);
        cudaFree(work_SU3_1);
        cudaFree(work_SU3_2);
        cudaFree(work_SU3_3);
        allocated = false;
    }
    
    // Getters for the working arrays
    double* get_work_SU2_1() const { return work_SU2_1; }
    double* get_work_SU2_2() const { return work_SU2_2; }
    double* get_work_SU2_3() const { return work_SU2_3; }
    double* get_work_SU3_1() const { return work_SU3_1; }
    double* get_work_SU3_2() const { return work_SU3_2; }
    double* get_work_SU3_3() const { return work_SU3_3; }
};


// Optimized SSPRK53 kernel that reduces memory allocations and synchronizations
template <size_t N_SU2, size_t N_ATOMS_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t lattice_size_SU3>
__host__
void SSPRK53_step_kernel(
    SU2_DeviceParams<N_SU2, lattice_size_SU2> su2_params,
    SU3_DeviceParams<N_SU3, lattice_size_SU3> su3_params,
    MixedInteractionParams<N_SU2, N_SU3> mixed_params,
    DriveFieldParams drive_params,
    double spin_length_SU2, double spin_length_SU3,
    RKWorkArrays<N_SU2, N_SU3> work_arrays)
{
    // SSPRK53 coefficients - now constexpr for compile-time optimization
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

    // Pre-compute all time step multiplications
    const double dt = drive_params.dt;
    const double b10_h = b10 * dt;
    const double b21_h = b21 * dt;
    const double b32_h = b32 * dt;
    const double b43_h = b43 * dt;
    const double b54_h = b54 * dt;
    const double c1_h = c1 * dt;
    const double c2_h = c2 * dt;
    const double c3_h = c3 * dt;
    const double c4_h = c4 * dt;

    // Store original spins pointers
    double* orig_spins_SU2 = su2_params.spins;
    double* orig_spins_SU3 = su3_params.spins;

    // Use pre-allocated working arrays
    double* k_SU2 = work_arrays.k_SU2;
    double* tmp_SU2 = work_arrays.work_SU2_2;
    double* u_SU2 = work_arrays.work_SU2_3;
    double* k_SU3 = work_arrays.k_SU3;
    double* tmp_SU3 = work_arrays.work_SU3_2;
    double* u_SU3 = work_arrays.work_SU3_3;

    // Optimize grid configuration - compute once with better occupancy
    constexpr int BLOCK_SIZE = 256;
    constexpr dim3 block_size(BLOCK_SIZE);
    const dim3 grid_size((lattice_size_SU2 + lattice_size_SU3 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Build work arrays for LLG kernel
    RKWorkArrays<N_SU2, N_SU3> llg_work;
    llg_work.k_SU2 = k_SU2;
    llg_work.k_SU3 = k_SU3;
    
    // Keep original time for incremental updates
    double base_time = drive_params.curr_time;
    
    // Stage 1: Compute k1 = f(t, y)
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        llg_work, su2_params, su3_params, mixed_params, drive_params);

    // Stage 1 update: tmp = y + b10*h*k1
    update_arrays_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size, block_size>>>
    (tmp_SU2, orig_spins_SU2, 1.0, k_SU2, b10_h, 
     tmp_SU3, orig_spins_SU3, 1.0, k_SU3, b10_h);
    
    // Stage 2: Compute k2 = f(t + c1*h, tmp)
    su2_params.spins = tmp_SU2;
    su3_params.spins = tmp_SU3;
    drive_params.curr_time = base_time + c1_h;
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        llg_work, su2_params, su3_params, mixed_params, drive_params);
        
    // Stage 2 update: u = tmp + b21*h*k2
    update_arrays_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size, block_size>>>
    (u_SU2, tmp_SU2, 1.0, k_SU2, b21_h, 
     u_SU3, tmp_SU3, 1.0, k_SU3, b21_h);
    
    // Stage 3: Compute k3 = f(t + c2*h, u)
    su2_params.spins = u_SU2;
    su3_params.spins = u_SU3;
    drive_params.curr_time = base_time + c2_h;
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        llg_work, su2_params, su3_params, mixed_params, drive_params);

    // Stage 3 update: tmp = a30*y + a32*u + b32*h*k3
    update_arrays_three_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size, block_size>>>
    (tmp_SU2, orig_spins_SU2, a30, u_SU2, a32, k_SU2, b32_h,
     tmp_SU3, orig_spins_SU3, a30, u_SU3, a32, k_SU3, b32_h);

    // Stage 4: Compute k4 = f(t + c3*h, tmp)
    su2_params.spins = tmp_SU2;
    su3_params.spins = tmp_SU3;
    drive_params.curr_time = base_time + c3_h;
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        llg_work, su2_params, su3_params, mixed_params, drive_params);

    // Stage 4 update: Reuse tmp arrays for efficiency (tmp = a40*y + a43*tmp + b43*h*k4)
    update_arrays_three_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> <<<grid_size, block_size>>>
    (tmp_SU2, orig_spins_SU2, a40, tmp_SU2, a43, k_SU2, b43_h,
     tmp_SU3, orig_spins_SU3, a40, tmp_SU3, a43, k_SU3, b43_h);

    // Stage 5: Compute k5 = f(t + c4*h, tmp)
    // tmp is already set from stage 4
    drive_params.curr_time = base_time + c4_h;
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        llg_work, su2_params, su3_params, mixed_params, drive_params);

    // Final update: y_new = a52*u + a54*tmp + b54*h*k5
    update_arrays_three_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size, block_size>>>
    (orig_spins_SU2, u_SU2, a52, tmp_SU2, a54, k_SU2, b54_h,
     orig_spins_SU3, u_SU3, a52, tmp_SU3, a54, k_SU3, b54_h);

    // Single final synchronization only - no intermediate sync needed
    //cudaDeviceSynchronize();
}

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
__host__
void mixed_lattice_cuda<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>::SSPRK53_step_cuda(double step_size, double curr_time, double tol) {
    // Static memory pool to persist across multiple calls
    static SSPRK53_MemoryPool<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> memory_pool;
    
    // Allocate temporary arrays for intermediate stages
    double *d_local_fields_SU2 = nullptr, *d_local_fields_SU3 = nullptr;
    
    // Error checking for memory allocation
    cudaError_t err = cudaMalloc(&d_local_fields_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate d_local_fields_SU2 (Error code: %s)\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&d_local_fields_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate d_local_fields_SU3 (Error code: %s)\n", cudaGetErrorString(err));
        cudaFree(d_local_fields_SU2);
        return;
    }

    // Check if any CUDA arrays are NULL before passing to the kernel
    if (d_spins.spins_SU2 == nullptr || d_spins.spins_SU3 == nullptr) {
        fprintf(stderr, "CUDA error: NULL pointer in d_spins detected\n");
        cudaFree(d_local_fields_SU2);
        cudaFree(d_local_fields_SU3);
        return;
    }

    // Build parameter structures
    auto su2_params = get_su2_params();
    su2_params.local_field = d_local_fields_SU2;
    
    auto su3_params = get_su3_params();
    su3_params.local_field = d_local_fields_SU3;
    
    auto mixed_params = get_mixed_params();
    auto drive_params = get_drive_params(curr_time, step_size);
    
    // Build work arrays structure
    RKWorkArrays<N_SU2, N_SU3> work_arrays;
    work_arrays.k_SU2 = memory_pool.get_work_SU2_1();
    work_arrays.k_SU3 = memory_pool.get_work_SU3_1();
    work_arrays.work_SU2_1 = memory_pool.get_work_SU2_1();
    work_arrays.work_SU2_2 = memory_pool.get_work_SU2_2();
    work_arrays.work_SU2_3 = memory_pool.get_work_SU2_3();
    work_arrays.work_SU3_1 = memory_pool.get_work_SU3_1();
    work_arrays.work_SU3_2 = memory_pool.get_work_SU3_2();
    work_arrays.work_SU3_3 = memory_pool.get_work_SU3_3();
    
    // Execute optimized SSPRK53 step with pre-allocated working arrays
    SSPRK53_step_kernel<N_SU2, N_ATOMS_SU2, N_ATOMS_SU2*dim*dim1*dim2, N_SU3, N_ATOMS_SU3, N_ATOMS_SU3*dim*dim1*dim2>(
        su2_params, su3_params, mixed_params, drive_params, 
        d_spin_length_SU2, d_spin_length_SU3, work_arrays);    
    
    // Check for errors in kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to launch SSPRK53_step_kernel (Error code: %s)\n", cudaGetErrorString(err));
    }
    
    // Cleanup temporary arrays (working arrays are managed by static memory pool)
    cudaFree(d_local_fields_SU2);
    cudaFree(d_local_fields_SU3);
}

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t lattice_size_SU3>
__host__
void euler_step_kernel(
    SU2_DeviceParams<N_SU2, lattice_size_SU2> su2_params,
    SU3_DeviceParams<N_SU3, lattice_size_SU3> su3_params,
    MixedInteractionParams<N_SU2, N_SU3> mixed_params,
    DriveFieldParams drive_params,
    double spin_length_SU2, double spin_length_SU3)
{
    // Allocate device memory for intermediate results
    double* k_SU2 = nullptr;
    double* k_SU3 = nullptr;
    
    cudaError_t err = cudaMalloc(&k_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate k_SU2 (Error code: %s)\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&k_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate k_SU3 (Error code: %s)\n", cudaGetErrorString(err));
        cudaFree(k_SU2);
        return;
    }
    
    // Set up grid and block dimensions
    dim3 block_size(256);
    dim3 grid_size((lattice_size_SU2 + lattice_size_SU3 + block_size.x - 1) / block_size.x);
    
    // Build work arrays structure
    RKWorkArrays<N_SU2, N_SU3> work_arrays;
    work_arrays.k_SU2 = k_SU2;
    work_arrays.k_SU3 = k_SU3;
    
    // Compute derivatives using LLG kernel
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        work_arrays, su2_params, su3_params, mixed_params, drive_params);

    // Update spins using Euler step: S(t+dt) = S(t) + dt * (dS/dt)
    update_arrays_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        su2_params.spins, su2_params.spins, 1.0, k_SU2, drive_params.dt, 
        su3_params.spins, su3_params.spins, 1.0, k_SU3, drive_params.dt);

    // Free temporary arrays
    cudaFree(k_SU2);
    cudaFree(k_SU3);
}

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
__host__
void mixed_lattice_cuda<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>::euler_step_cuda(double step_size, double curr_time, double tol) {
    // Allocate temporary arrays for local fields
    double *d_local_fields_SU2 = nullptr, *d_local_fields_SU3 = nullptr;
    
    // Error checking for memory allocation
    cudaError_t err = cudaMalloc(&d_local_fields_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate d_local_fields_SU2 (Error code: %s)\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&d_local_fields_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate d_local_fields_SU3 (Error code: %s)\n", cudaGetErrorString(err));
        cudaFree(d_local_fields_SU2);
        return;
    }
    
    // Check if any CUDA arrays are NULL before passing to the kernel
    if (d_spins.spins_SU2 == nullptr || d_spins.spins_SU3 == nullptr) {
        fprintf(stderr, "CUDA error: NULL pointer in d_spins detected\n");
        cudaFree(d_local_fields_SU2);
        cudaFree(d_local_fields_SU3);
        return;
    }
    
    // Build parameter structures
    auto su2_params = get_su2_params();
    su2_params.local_field = d_local_fields_SU2;
    
    auto su3_params = get_su3_params();
    su3_params.local_field = d_local_fields_SU3;
    
    auto mixed_params = get_mixed_params();
    auto drive_params = get_drive_params(curr_time, step_size);
    
    // Execute Euler step
    euler_step_kernel<N_SU2, N_ATOMS_SU2, N_ATOMS_SU2*dim*dim1*dim2, N_SU3, N_ATOMS_SU3, N_ATOMS_SU3*dim*dim1*dim2>(
        su2_params, su3_params, mixed_params, drive_params,
        d_spin_length_SU2, d_spin_length_SU3);
    
    // Check for errors in kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to launch euler_step_kernel (Error code: %s)\n", cudaGetErrorString(err));
    }
    
    // Make sure everything is synchronized
    //cudaDeviceSynchronize();
    
    // Cleanup temporary arrays
    cudaFree(d_local_fields_SU2);
    cudaFree(d_local_fields_SU3);
}

// ==================== Magnetization Computation Kernels ====================

// Helper function for atomic add on double (works on all CUDA versions)
__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Kernel to compute local magnetization for SU2 (simple average)
template <size_t N_SU2, size_t lattice_size_SU2>
__global__
void compute_magnetization_local_SU2_kernel(const double* spins, double* mag_out) {
    // Use shared memory for reduction
    __shared__ double shared_mag[N_SU2];
    
    int tid = threadIdx.x;
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (tid < N_SU2) {
        shared_mag[tid] = 0.0;
    }
    __syncthreads();
    
    // Each thread accumulates its contribution
    if (site < lattice_size_SU2) {
        for (size_t j = 0; j < N_SU2; ++j) {
            atomicAddDouble(&shared_mag[j], spins[site * N_SU2 + j]);
        }
    }
    __syncthreads();
    
    // Thread 0 writes block result to global memory
    if (tid == 0) {
        for (size_t j = 0; j < N_SU2; ++j) {
            atomicAddDouble(&mag_out[j], shared_mag[j]);
        }
    }
}

// Kernel to compute local magnetization for SU3 (simple average)
template <size_t N_SU3, size_t lattice_size_SU3>
__global__
void compute_magnetization_local_SU3_kernel(const double* spins, double* mag_out) {
    // Use shared memory for reduction
    __shared__ double shared_mag[N_SU3];
    
    int tid = threadIdx.x;
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (tid < N_SU3) {
        shared_mag[tid] = 0.0;
    }
    __syncthreads();
    
    // Each thread accumulates its contribution
    if (site < lattice_size_SU3) {
        for (size_t j = 0; j < N_SU3; ++j) {
            atomicAddDouble(&shared_mag[j], spins[site * N_SU3 + j]);
        }
    }
    __syncthreads();
    
    // Thread 0 writes block result to global memory
    if (tid == 0) {
        for (size_t j = 0; j < N_SU3; ++j) {
            atomicAddDouble(&mag_out[j], shared_mag[j]);
        }
    }
}

// Kernel to compute global magnetization for SU2 (with sublattice frames)
template <size_t N_SU2, size_t N_ATOMS_SU2, size_t lattice_size_SU2, size_t dim1, size_t dim2, size_t dim>
__global__
void compute_magnetization_global_SU2_kernel(
    const double* spins,
    const double* sublattice_frames,  // [N_ATOMS_SU2 * N_SU2 * N_SU2]
    double* mag_out)
{
    __shared__ double shared_mag[N_SU2];
    
    int tid = threadIdx.x;
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (tid < N_SU2) {
        shared_mag[tid] = 0.0;
    }
    __syncthreads();
    
    if (site < lattice_size_SU2) {
        // Determine sublattice index
        size_t sublattice = site % N_ATOMS_SU2;
        
        // Transform spin to global frame
        double spin_global[N_SU2];
        for (size_t mu = 0; mu < N_SU2; ++mu) {
            spin_global[mu] = 0.0;
            for (size_t nu = 0; nu < N_SU2; ++nu) {
                spin_global[mu] += sublattice_frames[sublattice * N_SU2 * N_SU2 + nu * N_SU2 + mu] 
                                   * spins[site * N_SU2 + nu];
            }
        }
        
        // Add to shared memory
        for (size_t j = 0; j < N_SU2; ++j) {
            atomicAddDouble(&shared_mag[j], spin_global[j]);
        }
    }
    __syncthreads();
    
    // Thread 0 writes block result to global memory
    if (tid == 0) {
        for (size_t j = 0; j < N_SU2; ++j) {
            atomicAddDouble(&mag_out[j], shared_mag[j]);
        }
    }
}

// Kernel to compute global magnetization for SU3 (with sublattice frames)
template <size_t N_SU3, size_t N_ATOMS_SU3, size_t lattice_size_SU3, size_t dim1, size_t dim2, size_t dim>
__global__
void compute_magnetization_global_SU3_kernel(
    const double* spins,
    const double* sublattice_frames,  // [N_ATOMS_SU3 * N_SU3 * N_SU3]
    double* mag_out)
{
    __shared__ double shared_mag[N_SU3];
    
    int tid = threadIdx.x;
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    if (tid < N_SU3) {
        shared_mag[tid] = 0.0;
    }
    __syncthreads();
    
    if (site < lattice_size_SU3) {
        // Determine sublattice index
        size_t sublattice = site % N_ATOMS_SU3;
        
        // Transform spin to global frame
        double spin_global[N_SU3];
        for (size_t mu = 0; mu < N_SU3; ++mu) {
            spin_global[mu] = 0.0;
            for (size_t nu = 0; nu < N_SU3; ++nu) {
                spin_global[mu] += sublattice_frames[sublattice * N_SU3 * N_SU3 + nu * N_SU3 + mu] 
                                   * spins[site * N_SU3 + nu];
            }
        }
        
        // Add to shared memory
        for (size_t j = 0; j < N_SU3; ++j) {
            atomicAddDouble(&shared_mag[j], spin_global[j]);
        }
    }
    __syncthreads();
    
    // Thread 0 writes block result to global memory
    if (tid == 0) {
        for (size_t j = 0; j < N_SU3; ++j) {
            atomicAddDouble(&mag_out[j], shared_mag[j]);
        }
    }
}

// ==================== GPU Magnetization Method Implementations ====================

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
__host__
void mixed_lattice_cuda<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>::compute_magnetization_local_SU2_cuda(std::array<double, N_SU2>& mag_out) {
    cudaMemset(d_mag_local_SU2, 0, N_SU2 * sizeof(double));
    
    const int block_size = 256;
    const int num_blocks = (lattice_size_SU2 + block_size - 1) / block_size;
    
    compute_magnetization_local_SU2_kernel<N_SU2, lattice_size_SU2>
        <<<num_blocks, block_size>>>(d_spins.spins_SU2, d_mag_local_SU2);
    
    std::vector<double> temp_mag(N_SU2);
    cudaMemcpy(temp_mag.data(), d_mag_local_SU2, N_SU2 * sizeof(double), cudaMemcpyDeviceToHost);
    
    const double inv_size = 1.0 / double(lattice_size_SU2);
    for (size_t j = 0; j < N_SU2; ++j) {
        mag_out[j] = temp_mag[j] * inv_size;
    }
}

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
__host__
void mixed_lattice_cuda<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>::compute_magnetization_local_SU3_cuda(std::array<double, N_SU3>& mag_out) {
    cudaMemset(d_mag_local_SU3, 0, N_SU3 * sizeof(double));
    
    const int block_size = 256;
    const int num_blocks = (lattice_size_SU3 + block_size - 1) / block_size;
    
    compute_magnetization_local_SU3_kernel<N_SU3, lattice_size_SU3>
        <<<num_blocks, block_size>>>(d_spins.spins_SU3, d_mag_local_SU3);
    
    std::vector<double> temp_mag(N_SU3);
    cudaMemcpy(temp_mag.data(), d_mag_local_SU3, N_SU3 * sizeof(double), cudaMemcpyDeviceToHost);
    
    const double inv_size = 1.0 / double(lattice_size_SU3);
    for (size_t j = 0; j < N_SU3; ++j) {
        mag_out[j] = temp_mag[j] * inv_size;
    }
}

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
__host__
void mixed_lattice_cuda<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>::compute_magnetization_global_SU2_cuda(std::array<double, N_SU2>& mag_out) {
    cudaMemset(d_mag_global_SU2, 0, N_SU2 * sizeof(double));
    
    std::vector<double> sublattice_frames_flat(N_ATOMS_SU2 * N_SU2 * N_SU2);
    for (size_t l = 0; l < N_ATOMS_SU2; ++l) {
        for (size_t i = 0; i < N_SU2; ++i) {
            for (size_t j = 0; j < N_SU2; ++j) {
                sublattice_frames_flat[l * N_SU2 * N_SU2 + i * N_SU2 + j] = 
                    this->sublattice_frames_SU2[l][i][j];
            }
        }
    }
    
    double* d_sublattice_frames_SU2;
    cudaMalloc(&d_sublattice_frames_SU2, N_ATOMS_SU2 * N_SU2 * N_SU2 * sizeof(double));
    cudaMemcpy(d_sublattice_frames_SU2, sublattice_frames_flat.data(), 
               N_ATOMS_SU2 * N_SU2 * N_SU2 * sizeof(double), cudaMemcpyHostToDevice);
    
    const int block_size = 256;
    const int num_blocks = (lattice_size_SU2 + block_size - 1) / block_size;
    
    compute_magnetization_global_SU2_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, dim1, dim2, dim>
        <<<num_blocks, block_size>>>(d_spins.spins_SU2, d_sublattice_frames_SU2, d_mag_global_SU2);
    
    std::vector<double> temp_mag(N_SU2);
    cudaMemcpy(temp_mag.data(), d_mag_global_SU2, N_SU2 * sizeof(double), cudaMemcpyDeviceToHost);
    
    const double inv_size = 1.0 / double(lattice_size_SU2);
    for (size_t j = 0; j < N_SU2; ++j) {
        mag_out[j] = temp_mag[j] * inv_size;
    }
    
    cudaFree(d_sublattice_frames_SU2);
}

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
__host__
void mixed_lattice_cuda<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>::compute_magnetization_global_SU3_cuda(std::array<double, N_SU3>& mag_out) {
    cudaMemset(d_mag_global_SU3, 0, N_SU3 * sizeof(double));
    
    std::vector<double> sublattice_frames_flat(N_ATOMS_SU3 * N_SU3 * N_SU3);
    for (size_t l = 0; l < N_ATOMS_SU3; ++l) {
        for (size_t i = 0; i < N_SU3; ++i) {
            for (size_t j = 0; j < N_SU3; ++j) {
                sublattice_frames_flat[l * N_SU3 * N_SU3 + i * N_SU3 + j] = 
                    this->sublattice_frames_SU3[l][i][j];
            }
        }
    }
    
    double* d_sublattice_frames_SU3;
    cudaMalloc(&d_sublattice_frames_SU3, N_ATOMS_SU3 * N_SU3 * N_SU3 * sizeof(double));
    cudaMemcpy(d_sublattice_frames_SU3, sublattice_frames_flat.data(), 
               N_ATOMS_SU3 * N_SU3 * N_SU3 * sizeof(double), cudaMemcpyHostToDevice);
    
    const int block_size = 256;
    const int num_blocks = (lattice_size_SU3 + block_size - 1) / block_size;
    
    compute_magnetization_global_SU3_kernel<N_SU3, N_ATOMS_SU3, lattice_size_SU3, dim1, dim2, dim>
        <<<num_blocks, block_size>>>(d_spins.spins_SU3, d_sublattice_frames_SU3, d_mag_global_SU3);
    
    std::vector<double> temp_mag(N_SU3);
    cudaMemcpy(temp_mag.data(), d_mag_global_SU3, N_SU3 * sizeof(double), cudaMemcpyDeviceToHost);
    
    const double inv_size = 1.0 / double(lattice_size_SU3);
    for (size_t j = 0; j < N_SU3; ++j) {
        mag_out[j] = temp_mag[j] * inv_size;
    }
    
    cudaFree(d_sublattice_frames_SU3);
}

//Explicitly declare template specializations for the mixed lattice class
template class mixed_lattice_cuda<3, 4, 8, 4, 8, 8, 8>;
template __global__ void LLG_kernel<3, 4, 2048, 8, 4, 2048>(
    RKWorkArrays<3, 8>,
    SU2_DeviceParams<3, 2048>,
    SU3_DeviceParams<8, 2048>,
    MixedInteractionParams<3, 8>,
    DriveFieldParams);

template class mixed_lattice_cuda<3, 4, 8, 4, 4, 4, 4>;
template __global__ void LLG_kernel<3, 4, 256, 8, 4, 256>(
    RKWorkArrays<3, 8>,
    SU2_DeviceParams<3, 256>,
    SU3_DeviceParams<8, 256>,
    MixedInteractionParams<3, 8>,
    DriveFieldParams);

template class mixed_lattice_cuda<3, 4, 8, 4, 1, 1, 1>;
template __global__ void LLG_kernel<3, 4, 4, 8, 4, 4>(
    RKWorkArrays<3, 8>,
    SU2_DeviceParams<3, 4>,
    SU3_DeviceParams<8, 4>,
    MixedInteractionParams<3, 8>,
    DriveFieldParams);

template class mixed_lattice_cuda<3, 4, 8, 4, 2, 2, 2>;
template __global__ void LLG_kernel<3, 4, 32, 8, 4, 32>(
    RKWorkArrays<3, 8>,
    SU2_DeviceParams<3, 32>,
    SU3_DeviceParams<8, 32>,
    MixedInteractionParams<3, 8>,
    DriveFieldParams);

template class mixed_lattice_cuda<3, 4, 8, 4, 3, 3, 3>;
template __global__ void LLG_kernel<3, 4, 108, 8, 4, 108>(
    RKWorkArrays<3, 8>,
    SU2_DeviceParams<3, 108>,
    SU3_DeviceParams<8, 108>,
    MixedInteractionParams<3, 8>,
    DriveFieldParams);