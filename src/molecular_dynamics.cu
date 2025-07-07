#include "molecular_dynamics.cuh"

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
    const double sqrt3_2 = 0.86602540378; // sqrt(3)/2
    
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
    double* k_SU2, double* k_SU3,
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
    double curr_time, double dt)
{
    int site_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (site_index > lattice_size_SU2 + lattice_size_SU3) return;

    if (site_index < lattice_size_SU2) {
        compute_local_field_SU2<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3>(
        d_local_field_SU2, site_index, d_spins_SU2, d_field_SU2, d_onsite_interaction_SU2,
        d_bilinear_interaction_SU2, d_bilinear_partners_SU2,
        d_trilinear_interaction_SU2, d_trilinear_partners_SU2,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2,
        d_spins_SU3, num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2);

        // Add drive field if applicable
        drive_field_T_SU2<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3>(
            d_local_field_SU2, curr_time, site_index, d_field_drive_1_SU2, d_field_drive_2_SU2, 
            d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
            max_mixed_tri_neighbors_SU2, d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2, d_spins_SU3);
        
            // Compute derivatives (dS/dt) using Landau-Lifshitz equation
        landau_Lifshitz_SU2<N_SU2, lattice_size_SU2>(
            k_SU2, site_index, d_local_field_SU2, d_spins_SU2);
    
    }else{
        size_t site_index_SU3 = site_index - lattice_size_SU2;        
        compute_local_field_SU3<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3>(
            d_local_field_SU3, site_index_SU3, d_spins_SU3, d_field_SU3, d_onsite_interaction_SU3,
            d_bilinear_interaction_SU3, d_bilinear_partners_SU3,
            d_trilinear_interaction_SU3, d_trilinear_partners_SU3,
            d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3,
            d_spins_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
            max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3);

        drive_field_T_SU3<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3>(
            d_local_field_SU3, curr_time, site_index_SU3, d_field_drive_1_SU3, d_field_drive_2_SU3, 
            d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
            max_mixed_tri_neighbors_SU3, d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3, d_spins_SU2);
            
        landau_Lifshitz_SU3<N_SU3, lattice_size_SU3>(
            k_SU3, site_index_SU3, d_local_field_SU3, d_spins_SU3);
    }
}


template<size_t N_SU2, size_t N_ATOMS_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t lattice_size_SU3>
__global__
void get_local_field(
    double* k_SU2, double* k_SU3,
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
    double* d_field_drive_1_SU2, double* d_field_drive_2_SU2, 
    double d_field_drive_amp_SU2, double d_field_drive_width_SU2, double d_field_drive_freq_SU2, double d_t_B_1_SU2, double d_t_B_2_SU2,
    double curr_time, double dt)
{
    int site_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (site_index > lattice_size_SU2 + lattice_size_SU3) return;

    if (site_index < lattice_size_SU2) {
        compute_local_field_SU2<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3>(
        k_SU2, site_index, d_spins_SU2, d_field_SU2, d_onsite_interaction_SU2,
        d_bilinear_interaction_SU2, d_bilinear_partners_SU2,
        d_trilinear_interaction_SU2, d_trilinear_partners_SU2,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2,
        d_spins_SU3, num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2);

        // Add drive field if applicable
        drive_field_T_SU2<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3>(
            k_SU2, curr_time, site_index, d_field_drive_1_SU2, d_field_drive_2_SU2, 
            d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
            max_mixed_tri_neighbors_SU2, d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2, d_spins_SU3);
    
    }else{        
        compute_local_field_SU3<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3>(
            k_SU3, site_index - lattice_size_SU2, d_spins_SU3, d_field_SU3, d_onsite_interaction_SU3,
            d_bilinear_interaction_SU3, d_bilinear_partners_SU3,
            d_trilinear_interaction_SU3, d_trilinear_partners_SU3,
            d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3,
            d_spins_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
            max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3);

        drive_field_T_SU3<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3>(
            k_SU3, curr_time, site_index - lattice_size_SU2, d_field_drive_1_SU2, d_field_drive_2_SU2, 
            d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
            max_mixed_tri_neighbors_SU3, d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3, d_spins_SU2);
    }
}

template <size_t N_SU2, size_t N_ATOMS_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t lattice_size_SU3>
__host__
void get_local_field_kernel(
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
    double* d_field_drive_1_SU2, double* d_field_drive_2_SU2, 
    double d_field_drive_amp_SU2, double d_field_drive_width_SU2, double d_field_drive_freq_SU2, double d_t_B_1_SU2, double d_t_B_2_SU2,
    double curr_time, double dt, double spin_length_SU2, double spin_length_SU3)
{
    // Optimized kernel launch configuration
    // Use device properties to determine optimal block size
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    
    // Calculate optimal block size based on register usage and shared memory
    const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
    const int max_threads_per_block = deviceProp.maxThreadsPerBlock;
    const int warp_size = deviceProp.warpSize;
    
    // Choose block size that's a multiple of warp size and maximizes occupancy
    int block_size = min(max_threads_per_block, 
                        ((total_sites < 1024) ? 128 : 256)); // Adaptive block size
    block_size = ((block_size + warp_size - 1) / warp_size) * warp_size; // Round to warp boundary
    
    dim3 block_dim(block_size);
    dim3 grid_dim((total_sites + block_dim.x - 1) / block_dim.x);

    double* k_SU2 = nullptr;
    double* k_SU3 = nullptr;

    // Allocate device memory for local fields
    cudaMalloc(&k_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
    cudaMalloc(&k_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));

    // Compute local fields for SU(2) and SU(3) spins
    get_local_field<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_dim, block_dim>>>(
        k_SU2, k_SU3,
        d_spins_SU2, d_spins_SU3,
        d_local_field_SU2, d_local_field_SU3,
        d_field_SU2, d_field_SU3,
        d_onsite_interaction_SU2, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU2, d_bilinear_interaction_SU3,
        d_bilinear_partners_SU2, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU2, d_trilinear_interaction_SU3,
        d_trilinear_partners_SU2, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_interaction_SU3,
        d_mixed_trilinear_partners_SU2, d_mixed_trilinear_partners_SU3,
        num_bi_SU2, num_tri_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3,
        d_field_drive_1_SU2, d_field_drive_2_SU2,
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
        curr_time, dt);
    

    cudaMemcpy(d_spins_SU2, k_SU2, lattice_size_SU2 * N_SU2 * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_spins_SU3, k_SU3, lattice_size_SU3 * N_SU3 * sizeof(double), cudaMemcpyDeviceToDevice);
}


template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
__host__
void mixed_lattice_cuda<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>::get_local_field_cuda(double step_size, double curr_time, double tol) {
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

    // Execute get_local_field_kernel
    get_local_field_kernel<N_SU2, N_ATOMS_SU2, N_ATOMS_SU2*dim*dim1*dim2, N_SU3, N_ATOMS_SU3, N_ATOMS_SU3*dim*dim1*dim2>(
        d_spins.spins_SU2, d_spins.spins_SU3,
        d_local_fields_SU2, d_local_fields_SU3,
        d_field_SU2, d_field_SU3,
        d_onsite_interaction_SU2, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU2, d_bilinear_interaction_SU3,
        d_bilinear_partners_SU2, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU2, d_trilinear_interaction_SU3,
        d_trilinear_partners_SU2, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_interaction_SU3,
        d_mixed_trilinear_partners_SU2, d_mixed_trilinear_partners_SU3,
        d_num_bi_SU2, d_num_tri_SU2, d_num_bi_SU3, d_num_tri_SU3, d_num_tri_SU2_SU3,
        max_bilinear_neighbors_SU2, max_trilinear_neighbors_SU2, max_mixed_trilinear_neighbors_SU2,
        max_bilinear_neighbors_SU3, max_trilinear_neighbors_SU3, max_mixed_trilinear_neighbors_SU3,
        d_field_drive_1_SU2, d_field_drive_2_SU2, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
        curr_time, step_size, d_spin_length_SU2, d_spin_length_SU3);    
    
    // Check for errors in kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to launch SSPRK53_step_kernel (Error code: %s)\n", cudaGetErrorString(err));
    }
    
    // Make sure everything is synchronized
    cudaDeviceSynchronize();
    
    // Cleanup temporary arrays
    cudaFree(d_local_fields_SU2);
    cudaFree(d_local_fields_SU3);
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
        
        size_t size_SU2 = lattice_size_SU2 * N_SU2 * sizeof(double);
        size_t size_SU3 = lattice_size_SU3 * N_SU3 * sizeof(double);

        cudaMalloc(&work_SU2_1, size_SU2);
        cudaMalloc(&work_SU2_2, size_SU2);
        cudaMalloc(&work_SU2_3, size_SU2);
        cudaMalloc(&work_SU3_1, size_SU3);
        cudaMalloc(&work_SU3_2, size_SU3);
        cudaMalloc(&work_SU3_3, size_SU3);

        // Set allocated memory to zero
        cudaMemset(work_SU2_1, 0, size_SU2);
        cudaMemset(work_SU2_2, 0, size_SU2);
        cudaMemset(work_SU2_3, 0, size_SU2);
        cudaMemset(work_SU3_1, 0, size_SU3);
        cudaMemset(work_SU3_2, 0, size_SU3);
        cudaMemset(work_SU3_3, 0, size_SU3);
        
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
    double* d_field_drive_1_SU2, double* d_field_drive_2_SU2,  double* d_field_drive_1_SU3, double* d_field_drive_2_SU3, 
    double d_field_drive_amp_SU2, double d_field_drive_width_SU2, double d_field_drive_freq_SU2, double d_t_B_1_SU2, double d_t_B_2_SU2,
    double curr_time, double dt, double spin_length_SU2, double spin_length_SU3,
    // Pre-allocated working arrays passed from caller
    double* work_SU2_1, double* work_SU2_2, double* work_SU2_3,
    double* work_SU3_1, double* work_SU3_2, double* work_SU3_3)
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
    const double b10_h = b10 * dt;
    const double b21_h = b21 * dt;
    const double b32_h = b32 * dt;
    const double b43_h = b43 * dt;
    const double b54_h = b54 * dt;
    const double c1_h = c1 * dt;
    const double c2_h = c2 * dt;
    const double c3_h = c3 * dt;
    const double c4_h = c4 * dt;

    // Use pre-allocated working arrays instead of allocating
    double* k_SU2 = work_SU2_1;
    double* tmp_SU2 = work_SU2_2;
    double* u_SU2 = work_SU2_3;
    double* k_SU3 = work_SU3_1;
    double* tmp_SU3 = work_SU3_2;
    double* u_SU3 = work_SU3_3;

    // Optimize grid configuration - compute once
    constexpr dim3 block_size(256);
    constexpr dim3 grid_size((lattice_size_SU2 + lattice_size_SU3 + 255) / 256);


    
    // Stage 1: Compute k1 = f(t, y)
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        k_SU2, k_SU3,
        d_spins_SU2, d_spins_SU3,
        d_local_field_SU2, d_local_field_SU3,
        d_field_SU2, d_field_SU3,
        d_onsite_interaction_SU2, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU2, d_bilinear_interaction_SU3,
        d_bilinear_partners_SU2, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU2, d_trilinear_interaction_SU3,
        d_trilinear_partners_SU2, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_interaction_SU3,
        d_mixed_trilinear_partners_SU2, d_mixed_trilinear_partners_SU3,
        num_bi_SU2, num_tri_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3,
        d_field_drive_1_SU2, d_field_drive_2_SU2,  d_field_drive_1_SU3, d_field_drive_2_SU3, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
        curr_time, dt);

    // Stage 1 update: tmp = y + b10*h*k1
    update_arrays_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size, block_size>>>
    (tmp_SU2, d_spins_SU2, 1.0, k_SU2, b10_h, 
     tmp_SU3, d_spins_SU3, 1.0, k_SU3, b10_h);
    
    // Stage 2: Compute k2 = f(t + c1*h, tmp)
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        k_SU2, k_SU3,
        tmp_SU2, tmp_SU3,
        d_local_field_SU2, d_local_field_SU3,
        d_field_SU2, d_field_SU3,
        d_onsite_interaction_SU2, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU2, d_bilinear_interaction_SU3,
        d_bilinear_partners_SU2, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU2, d_trilinear_interaction_SU3,
        d_trilinear_partners_SU2, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_interaction_SU3,
        d_mixed_trilinear_partners_SU2, d_mixed_trilinear_partners_SU3,
        num_bi_SU2, num_tri_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3,
        d_field_drive_1_SU2, d_field_drive_2_SU2,  d_field_drive_1_SU3, d_field_drive_2_SU3, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
        curr_time + c1_h, dt);
        
    // Stage 2 update: u = tmp + b21*h*k2
    update_arrays_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size, block_size>>>
    (u_SU2, tmp_SU2, 1.0, k_SU2, b21_h, 
     u_SU3, tmp_SU3, 1.0, k_SU3, b21_h);
    
    // Stage 3: Compute k3 = f(t + c2*h, u)
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        k_SU2, k_SU3,
        u_SU2, u_SU3,
        d_local_field_SU2, d_local_field_SU3,
        d_field_SU2, d_field_SU3,
        d_onsite_interaction_SU2, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU2, d_bilinear_interaction_SU3,
        d_bilinear_partners_SU2, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU2, d_trilinear_interaction_SU3,
        d_trilinear_partners_SU2, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_interaction_SU3,
        d_mixed_trilinear_partners_SU2, d_mixed_trilinear_partners_SU3,
        num_bi_SU2, num_tri_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3,
        d_field_drive_1_SU2, d_field_drive_2_SU2,  d_field_drive_1_SU3, d_field_drive_2_SU3, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
        curr_time + c2_h, dt);

    // Stage 3 update: tmp = a30*y + a32*u + b32*h*k3
    update_arrays_three_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size, block_size>>>
    (tmp_SU2, d_spins_SU2, a30, u_SU2, a32, k_SU2, b32_h,
     tmp_SU3, d_spins_SU3, a30, u_SU3, a32, k_SU3, b32_h);

    // Stage 4: Compute k4 = f(t + c3*h, tmp)
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        k_SU2, k_SU3,
        tmp_SU2, tmp_SU3,
        d_local_field_SU2, d_local_field_SU3,
        d_field_SU2, d_field_SU3,
        d_onsite_interaction_SU2, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU2, d_bilinear_interaction_SU3,
        d_bilinear_partners_SU2, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU2, d_trilinear_interaction_SU3,
        d_trilinear_partners_SU2, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_interaction_SU3,
        d_mixed_trilinear_partners_SU2, d_mixed_trilinear_partners_SU3,
        num_bi_SU2, num_tri_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3,
        d_field_drive_1_SU2, d_field_drive_2_SU2,  d_field_drive_1_SU3, d_field_drive_2_SU3, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
        curr_time + c3_h, dt);

    // Stage 4 update: Reuse tmp arrays for efficiency (tmp = a40*y + a43*tmp + b43*h*k4)
    update_arrays_three_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> <<<grid_size, block_size>>>
    (tmp_SU2, d_spins_SU2, a40, tmp_SU2, a43, k_SU2, b43_h,
     tmp_SU3, d_spins_SU3, a40, tmp_SU3, a43, k_SU3, b43_h);

    // Stage 5: Compute k5 = f(t + c4*h, tmp)
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        k_SU2, k_SU3,
        tmp_SU2, tmp_SU3,
        d_local_field_SU2, d_local_field_SU3,
        d_field_SU2, d_field_SU3,
        d_onsite_interaction_SU2, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU2, d_bilinear_interaction_SU3,
        d_bilinear_partners_SU2, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU2, d_trilinear_interaction_SU3,
        d_trilinear_partners_SU2, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_interaction_SU3,
        d_mixed_trilinear_partners_SU2, d_mixed_trilinear_partners_SU3,
        num_bi_SU2, num_tri_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3,
        d_field_drive_1_SU2, d_field_drive_2_SU2,  d_field_drive_1_SU3, d_field_drive_2_SU3, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
        curr_time + c4_h, dt);

    // Final update: y_new = a52*u + a54*tmp + b54*h*k5
    update_arrays_three_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size, block_size>>>
    (d_spins_SU2, u_SU2, a52, tmp_SU2, a54, k_SU2, b54_h,
     d_spins_SU3, u_SU3, a52, tmp_SU3, a54, k_SU3, b54_h);

    // Single final synchronization only - no intermediate sync needed
    cudaDeviceSynchronize();
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

    // Execute optimized SSPRK53 step with pre-allocated working arrays
    SSPRK53_step_kernel<N_SU2, N_ATOMS_SU2, N_ATOMS_SU2*dim*dim1*dim2, N_SU3, N_ATOMS_SU3, N_ATOMS_SU3*dim*dim1*dim2>(
    d_spins.spins_SU2, d_spins.spins_SU3,
    d_local_fields_SU2, d_local_fields_SU3,
    d_field_SU2, d_field_SU3,
    d_onsite_interaction_SU2, d_onsite_interaction_SU3,
    d_bilinear_interaction_SU2, d_bilinear_interaction_SU3,
    d_bilinear_partners_SU2, d_bilinear_partners_SU3,
    d_trilinear_interaction_SU2, d_trilinear_interaction_SU3,
    d_trilinear_partners_SU2, d_trilinear_partners_SU3,
    d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_interaction_SU3,
    d_mixed_trilinear_partners_SU2, d_mixed_trilinear_partners_SU3,
    d_num_bi_SU2, d_num_tri_SU2, d_num_bi_SU3, d_num_tri_SU3, d_num_tri_SU2_SU3,
    max_bilinear_neighbors_SU2, max_trilinear_neighbors_SU2, max_mixed_trilinear_neighbors_SU2,
    max_bilinear_neighbors_SU3, max_trilinear_neighbors_SU3, max_mixed_trilinear_neighbors_SU3,
    d_field_drive_1_SU2, d_field_drive_2_SU2,  d_field_drive_1_SU3, d_field_drive_2_SU3, 
    d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
    curr_time, step_size, d_spin_length_SU2, d_spin_length_SU3,
    // Pass pre-allocated working arrays from memory pool
    memory_pool.get_work_SU2_1(), memory_pool.get_work_SU2_2(), memory_pool.get_work_SU2_3(),
    memory_pool.get_work_SU3_1(), memory_pool.get_work_SU3_2(), memory_pool.get_work_SU3_3());    
    
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
    double* d_field_drive_1_SU2, double* d_field_drive_2_SU2,  double* d_field_drive_1_SU3, double* d_field_drive_2_SU3, 
    double d_field_drive_amp_SU2, double d_field_drive_width_SU2, double d_field_drive_freq_SU2, double d_t_B_1_SU2, double d_t_B_2_SU2,
    double curr_time, double dt, double spin_length_SU2, double spin_length_SU3)
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
    
    // Synchronize before starting the computation
    cudaDeviceSynchronize();

    // Compute local fields
    LLG_kernel<N_SU2, N_ATOMS_SU2, lattice_size_SU2, N_SU3, N_ATOMS_SU3, lattice_size_SU3><<<grid_size, block_size>>>(
        k_SU2, k_SU3,
        d_spins_SU2, d_spins_SU3,
        d_local_field_SU2, d_local_field_SU3,
        d_field_SU2, d_field_SU3,
        d_onsite_interaction_SU2, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU2, d_bilinear_interaction_SU3,
        d_bilinear_partners_SU2, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU2, d_trilinear_interaction_SU3,
        d_trilinear_partners_SU2, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_interaction_SU3,
        d_mixed_trilinear_partners_SU2, d_mixed_trilinear_partners_SU3,
        num_bi_SU2, num_tri_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3,
        d_field_drive_1_SU2, d_field_drive_2_SU2,  d_field_drive_1_SU3, d_field_drive_2_SU3, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
        curr_time, dt);

    
    // Synchronize before updating arrays
    cudaDeviceSynchronize();
    
    double *temp2_SU2, *temp2_SU3;
    cudaMalloc(&temp2_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
    cudaMalloc(&temp2_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
    
    // Update spins using Euler step: S(t+dt) = S(t) + dt * (dS/dt)
    update_arrays_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size, block_size>>>
    (temp2_SU2, d_spins_SU2, 1.0, k_SU2, dt, 
     temp2_SU3, d_spins_SU3, 1.0, k_SU3, dt);

    cudaMemcpy(d_spins_SU2, temp2_SU2, lattice_size_SU2 * N_SU2 * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_spins_SU3, temp2_SU3, lattice_size_SU3 * N_SU3 * sizeof(double), cudaMemcpyDeviceToDevice);

    // Final synchronization
    cudaDeviceSynchronize();
    
    // Free temporary arrays
    cudaFree(temp2_SU2);
    cudaFree(temp2_SU3);
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
    
    // Execute Euler step
    euler_step_kernel<N_SU2, N_ATOMS_SU2, N_ATOMS_SU2*dim*dim1*dim2, N_SU3, N_ATOMS_SU3, N_ATOMS_SU3*dim*dim1*dim2>(
        d_spins.spins_SU2, d_spins.spins_SU3,
        d_local_fields_SU2, d_local_fields_SU3,
        d_field_SU2, d_field_SU3,
        d_onsite_interaction_SU2, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU2, d_bilinear_interaction_SU3,
        d_bilinear_partners_SU2, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU2, d_trilinear_interaction_SU3,
        d_trilinear_partners_SU2, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_interaction_SU3,
        d_mixed_trilinear_partners_SU2, d_mixed_trilinear_partners_SU3,
        d_num_bi_SU2, d_num_tri_SU2, d_num_bi_SU3, d_num_tri_SU3, d_num_tri_SU2_SU3,
        max_bilinear_neighbors_SU2, max_trilinear_neighbors_SU2, max_mixed_trilinear_neighbors_SU2,
        max_bilinear_neighbors_SU3, max_trilinear_neighbors_SU3, max_mixed_trilinear_neighbors_SU3,
        d_field_drive_1_SU2, d_field_drive_2_SU2, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2,
        curr_time, step_size, d_spin_length_SU2, d_spin_length_SU3);
    
    // Check for errors in kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to launch euler_step_kernel (Error code: %s)\n", cudaGetErrorString(err));
    }
    
    // Make sure everything is synchronized
    cudaDeviceSynchronize();
    
    // Cleanup temporary arrays
    cudaFree(d_local_fields_SU2);
    cudaFree(d_local_fields_SU3);
}


//Explicitly declare template specializations for the mixed lattice class
template class mixed_lattice_cuda<3, 4, 8, 4, 8, 8, 8>;
template void __global__ LLG_kernel<3, 4, 2048, 8, 4, 2048>(
    double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*,
    unsigned long*, unsigned long*, double*, double*, unsigned long*, unsigned long*, double*, double*, unsigned long*, unsigned long*,
    unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long,
    double*, double*, double*, double*, double, double, double, double, double, double, double
);

template class mixed_lattice_cuda<3, 4, 8, 4, 4, 4, 4>;
template void __global__ LLG_kernel<3, 4, 256, 8, 4, 256>(
    double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*,
    unsigned long*, unsigned long*, double*, double*, unsigned long*, unsigned long*, double*, double*, unsigned long*, unsigned long*,
    unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long,
    double*, double*, double*, double*, double, double, double, double, double, double, double
);

template class mixed_lattice_cuda<3, 4, 8, 4, 1, 1, 1>;
template void __global__ LLG_kernel<3, 4, 4, 8, 4, 4>(
    double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*,
    unsigned long*, unsigned long*, double*, double*, unsigned long*, unsigned long*, double*, double*, unsigned long*, unsigned long*,
    unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long,
    double*, double*, double*, double*, double, double, double, double, double, double, double
);


template class mixed_lattice_cuda<3, 4, 8, 4, 2, 2, 2>;
template void __global__ LLG_kernel<3, 4, 32, 8, 4, 32>(
    double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*,
    unsigned long*, unsigned long*, double*, double*, unsigned long*, unsigned long*, double*, double*, unsigned long*, unsigned long*,
    unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long,
    double*, double*, double*, double*, double, double, double, double, double, double, double
);

template class mixed_lattice_cuda<3, 4, 8, 4, 3, 3, 3>;
template void __global__ LLG_kernel<3, 4, 108, 8, 4, 108>(
    double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*,
    unsigned long*, unsigned long*, double*, double*, unsigned long*, unsigned long*, double*, double*, unsigned long*, unsigned long*,
    unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long, unsigned long,
    double*, double*, double*, double*, double, double, double, double, double, double, double
);