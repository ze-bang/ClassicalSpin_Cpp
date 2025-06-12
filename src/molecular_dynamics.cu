#include "molecular_dynamics.cuh"

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
    const double sqrt3 = 1.73205080756887729;

    result[0] = a[1]*b[2] - a[2]*b[1] + 0.5*(a[4]*b[5] - a[5]*b[4]) + 0.5*(a[6]*b[7] - a[7]*b[6]);
    result[1] = a[2]*b[0] - a[0]*b[2] + 0.5*(a[3]*b[5] - a[5]*b[3]) - 0.5*(a[6]*b[7] - a[7]*b[6]);
    result[2] = a[0]*b[1] - a[1]*b[0] + 0.5*(a[3]*b[4] - a[4]*b[3]) + 0.5*(a[5]*b[6] - a[6]*b[5]);
    result[3] = 0.5*(a[1]*b[5] - a[5]*b[1] - a[2]*b[4] + a[4]*b[2]) + 0.5*sqrt3*(a[6]*b[7] - a[7]*b[6]);
    result[4] = 0.5*(a[2]*b[3] - a[3]*b[2] + a[5]*b[0] - a[0]*b[5]) + 0.5*sqrt3*(a[7]*b[6] - a[6]*b[7]);
    result[5] = 0.5*(a[0]*b[4] - a[4]*b[0] - a[1]*b[3] + a[3]*b[1]) + 0.5*sqrt3*(a[6]*b[6] + a[7]*b[7]);
    result[6] = 0.5*(a[0]*b[7] - a[7]*b[0] + a[1]*b[6] - a[6]*b[1] + a[4]*b[7] - a[7]*b[4] + a[5]*b[6] - a[6]*b[5]) + 0.5*sqrt3*(a[3]*b[7] - a[7]*b[3] - a[4]*b[6] + a[6]*b[4]);
    result[7] = 0.5*(a[0]*b[6] - a[6]*b[0] - a[1]*b[7] + a[7]*b[1] - a[4]*b[6] + a[6]*b[4] + a[5]*b[7] - a[7]*b[5]) + 0.5*sqrt3*(a[3]*b[6] - a[6]*b[3] + a[4]*b[7] - a[7]*b[4]);
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
__global__ void update_arrays_kernel(double* out, const double* in1, double a1, 
                                    const double* in2, double a2,
                                    size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a1 * in1[idx] + a2 * in2[idx];
    }
}

// Kernel to update arrays with linear combination of three arrays
__global__ void update_arrays_three_kernel(double* out, 
                                          const double* in1, double a1,
                                          const double* in2, double a2,
                                          const double* in3, double a3,
                                          size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a1 * in1[idx] + a2 * in2[idx] + a3 * in3[idx];
    }
}

// Kernel to normalize spins for SU2
template<size_t N>
__global__ void normalize_spins_SU2_kernel(double* spins, double spin_length, size_t size) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site < size) {
        double* spin = &spins[site * N];
        double norm = 0.0;
        for (int i = 0; i < N; i++) {
            norm += spin[i] * spin[i];
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (int i = 0; i < N; i++) {
                spin[i] = spin[i] * spin_length / norm;
            }
        }
    }
}

// Kernel to normalize spins for SU3
template<size_t N>
__global__ void normalize_spins_SU3_kernel(double* spins, double spin_length, size_t size) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site < size) {
        double* spin = &spins[site * N];
        double norm = 0.0;
        for (int i = 0; i < N; i++) {
            norm += spin[i] * spin[i];
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (int i = 0; i < N; i++) {
                spin[i] = spin[i] * spin_length / norm;
            }
        }
    }
}


template <size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
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
    double* d_field_drive_1_SU2, double* d_field_drive_2_SU2, 
    double d_field_drive_amp_SU2, double d_field_drive_width_SU2, double d_field_drive_freq_SU2, double d_t_B_1_SU2, double d_t_B_2_SU2,
    double curr_time, double dt)
{
    // SSPRK53 coefficients from mixed_lattice.h
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

    // Pre-compute step size multiplications
    const double b10_h = b10 * dt;
    const double b21_h = b21 * dt;
    const double b32_h = b32 * dt;
    const double b43_h = b43 * dt;
    const double b54_h = b54 * dt;
    const double c1_h = c1 * dt;
    const double c2_h = c2 * dt;
    const double c3_h = c3 * dt;
    const double c4_h = c4 * dt;

    // Allocate all working arrays upfront
    double* tmp_SU2 = nullptr, *k_SU2 = nullptr, *u_SU2 = nullptr;
    double* tmp_SU3 = nullptr, *k_SU3 = nullptr, *u_SU3 = nullptr;
    double* d_drive_field_SU2 = nullptr, *d_drive_field_SU3 = nullptr;

    // Use cudaError_t to check for errors
    cudaError_t err;
    
    err = cudaMalloc(&d_drive_field_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate d_drive_field_SU2 (Error code: %s)\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc(&d_drive_field_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate d_drive_field_SU3 (Error code: %s)\n", cudaGetErrorString(err));
        cudaFree(d_drive_field_SU2);
        return;
    }

    err = cudaMalloc(&tmp_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate tmp_SU2 (Error code: %s)\n", cudaGetErrorString(err));
        cudaFree(d_drive_field_SU2);
        cudaFree(d_drive_field_SU3);
        return;
    }
    
    err = cudaMalloc(&k_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate k_SU2 (Error code: %s)\n", cudaGetErrorString(err));
        cudaFree(d_drive_field_SU2);
        cudaFree(d_drive_field_SU3);
        cudaFree(tmp_SU2);
        return;
    }
    
    err = cudaMalloc(&u_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate u_SU2 (Error code: %s)\n", cudaGetErrorString(err));
        cudaFree(d_drive_field_SU2);
        cudaFree(d_drive_field_SU3);
        cudaFree(tmp_SU2);
        cudaFree(k_SU2);
        return;
    }
    
    err = cudaMalloc(&tmp_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate tmp_SU3 (Error code: %s)\n", cudaGetErrorString(err));
        cudaFree(d_drive_field_SU2);
        cudaFree(d_drive_field_SU3);
        cudaFree(tmp_SU2);
        cudaFree(k_SU2);
        cudaFree(u_SU2);
        return;
    }
    
    err = cudaMalloc(&k_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate k_SU3 (Error code: %s)\n", cudaGetErrorString(err));
        cudaFree(d_drive_field_SU2);
        cudaFree(d_drive_field_SU3);
        cudaFree(tmp_SU2);
        cudaFree(k_SU2);
        cudaFree(u_SU2);
        cudaFree(tmp_SU3);
        return;
    }
    
    err = cudaMalloc(&u_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: Failed to allocate u_SU3 (Error code: %s)\n", cudaGetErrorString(err));
        cudaFree(d_drive_field_SU2);
        cudaFree(d_drive_field_SU3);
        cudaFree(tmp_SU2);
        cudaFree(k_SU2);
        cudaFree(u_SU2);
        cudaFree(tmp_SU3);
        cudaFree(k_SU3);
        return;
    }



    dim3 block_size(256);
    dim3 grid_size_SU2((lattice_size_SU2 + block_size.x - 1) / block_size.x);
    dim3 grid_size_SU3((lattice_size_SU3 + block_size.x - 1) / block_size.x);
    // Stage 1: 
    compute_local_field_SU2_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size_SU2, block_size>>>(
        d_local_field_SU2, d_spins_SU2, d_field_SU2, d_onsite_interaction_SU2,
        d_bilinear_interaction_SU2, d_bilinear_partners_SU2,
        d_trilinear_interaction_SU2, d_trilinear_partners_SU2,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2,
        d_spins_SU3, num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2);
    
    compute_local_field_SU3_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size_SU3, block_size>>>(
        d_local_field_SU3, d_spins_SU3, d_field_SU3, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU3, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU3, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3,
        d_spins_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3);
    
    drive_field_T_SU2_kernel<N_SU2, lattice_size_SU2><<<grid_size_SU2, block_size>>>(
        curr_time, d_local_field_SU2, d_field_drive_1_SU2, d_field_drive_2_SU2, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2);

    landau_Lifshitz_SU2_kernel<N_SU2, lattice_size_SU2> <<<grid_size_SU2, block_size>>>(
        k_SU2, d_spins_SU2, d_local_field_SU2);

    landau_Lifshitz_SU3_kernel<N_SU3, lattice_size_SU3> <<<grid_size_SU3, block_size>>>(
        k_SU3, d_spins_SU3, d_local_field_SU3);

    update_arrays_kernel<<<grid_size_SU2, block_size>>>(tmp_SU2, d_spins_SU2, 1.0, k_SU2, b10_h, lattice_size_SU2 * N_SU2);
    update_arrays_kernel<<<grid_size_SU3, block_size>>>(tmp_SU3, d_spins_SU3, 1.0, k_SU3, b10_h, lattice_size_SU3 * N_SU3);
    // Stage 2
    compute_local_field_SU2_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size_SU2, block_size>>>(
        d_local_field_SU2, tmp_SU2, d_field_SU2, d_onsite_interaction_SU2,
        d_bilinear_interaction_SU2, d_bilinear_partners_SU2,
        d_trilinear_interaction_SU2, d_trilinear_partners_SU2,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2,
        tmp_SU3, num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2);
        
    compute_local_field_SU3_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size_SU3, block_size>>>(
        d_local_field_SU3, tmp_SU3, d_field_SU3, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU3, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU3, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3,
        tmp_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3);
    
    drive_field_T_SU2_kernel<N_SU2, lattice_size_SU2><<<grid_size_SU2, block_size>>>(
        curr_time + c1_h, d_local_field_SU2, d_field_drive_1_SU2, d_field_drive_2_SU2, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2);

    landau_Lifshitz_SU2_kernel<N_SU2, lattice_size_SU2> <<<grid_size_SU2, block_size>>>(
        k_SU2, tmp_SU2, d_local_field_SU2);

    landau_Lifshitz_SU3_kernel<N_SU3, lattice_size_SU3> <<<grid_size_SU3, block_size>>>(
        k_SU3, tmp_SU3, d_local_field_SU3);

    update_arrays_kernel<<<grid_size_SU2, block_size>>>(u_SU2, tmp_SU2, 1.0, k_SU2, b21_h, lattice_size_SU2 * N_SU2);
    update_arrays_kernel<<<grid_size_SU3, block_size>>>(u_SU3, tmp_SU3, 1.0, k_SU3, b21_h, lattice_size_SU3 * N_SU3);
    
    // Stage 3
    compute_local_field_SU2_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size_SU2, block_size>>>(
        d_local_field_SU2, u_SU2, d_field_SU2, d_onsite_interaction_SU2,
        d_bilinear_interaction_SU2, d_bilinear_partners_SU2,
        d_trilinear_interaction_SU2, d_trilinear_partners_SU2,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2,
        u_SU3, num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2);
        
    compute_local_field_SU3_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size_SU3, block_size>>>(
        d_local_field_SU3, u_SU3, d_field_SU3, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU3, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU3, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3,
        u_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3);
    
    drive_field_T_SU2_kernel<N_SU2, lattice_size_SU2><<<grid_size_SU2, block_size>>>(
        curr_time + c2_h, d_local_field_SU2, d_field_drive_1_SU2, d_field_drive_2_SU2, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2);

    landau_Lifshitz_SU2_kernel<N_SU2, lattice_size_SU2> <<<grid_size_SU2, block_size>>>(
        k_SU2, u_SU2, d_local_field_SU2);

    landau_Lifshitz_SU3_kernel<N_SU3, lattice_size_SU3> <<<grid_size_SU3, block_size>>>(
        k_SU3, u_SU3, d_local_field_SU3);

    update_arrays_three_kernel<<<grid_size_SU2, block_size>>>(tmp_SU2, d_spins_SU2, a30, u_SU2, a32, k_SU2, b32_h, lattice_size_SU2 * N_SU2);
    update_arrays_three_kernel<<<grid_size_SU3, block_size>>>(tmp_SU3, d_spins_SU3, a30, u_SU3, a32, k_SU3, b32_h, lattice_size_SU3 * N_SU3);

    // Stage 4
    compute_local_field_SU2_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size_SU2, block_size>>>(
        d_local_field_SU2, tmp_SU2, d_field_SU2, d_onsite_interaction_SU2,
        d_bilinear_interaction_SU2, d_bilinear_partners_SU2,
        d_trilinear_interaction_SU2, d_trilinear_partners_SU2,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2,
        tmp_SU3, num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2);
        
    compute_local_field_SU3_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size_SU3, block_size>>>(
        d_local_field_SU3, tmp_SU3, d_field_SU3, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU3, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU3, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3,
        tmp_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3);
    
    drive_field_T_SU2_kernel<N_SU2, lattice_size_SU2><<<grid_size_SU2, block_size>>>(
        curr_time + c3_h, d_local_field_SU2, d_field_drive_1_SU2, d_field_drive_2_SU2, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2);

    landau_Lifshitz_SU2_kernel<N_SU2, lattice_size_SU2> <<<grid_size_SU2, block_size>>>(
        k_SU2, tmp_SU2, d_local_field_SU2);

    landau_Lifshitz_SU3_kernel<N_SU3, lattice_size_SU3> <<<grid_size_SU3, block_size>>>(
        k_SU3, tmp_SU3, d_local_field_SU3);

    // Create temporary arrays to hold the results
    double *temp2_SU2, *temp2_SU3;
    cudaMalloc(&temp2_SU2, lattice_size_SU2 * N_SU2 * sizeof(double));
    cudaMalloc(&temp2_SU3, lattice_size_SU3 * N_SU3 * sizeof(double));
    
    update_arrays_three_kernel<<<grid_size_SU2, block_size>>>(temp2_SU2, d_spins_SU2, a40, tmp_SU2, a43, k_SU2, b43_h, lattice_size_SU2 * N_SU2);
    update_arrays_three_kernel<<<grid_size_SU3, block_size>>>(temp2_SU3, d_spins_SU3, a40, tmp_SU3, a43, k_SU3, b43_h, lattice_size_SU3 * N_SU3);
    
    // Copy results back to tmp arrays
    cudaMemcpy(tmp_SU2, temp2_SU2, lattice_size_SU2 * N_SU2 * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(tmp_SU3, temp2_SU3, lattice_size_SU3 * N_SU3 * sizeof(double), cudaMemcpyDeviceToDevice);
    
    // Free temporary arrays
    cudaFree(temp2_SU2);
    cudaFree(temp2_SU3);


    // Stage 5
    compute_local_field_SU2_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size_SU2, block_size>>>(
        d_local_field_SU2, tmp_SU2, d_field_SU2, d_onsite_interaction_SU2,
        d_bilinear_interaction_SU2, d_bilinear_partners_SU2,
        d_trilinear_interaction_SU2, d_trilinear_partners_SU2,
        d_mixed_trilinear_interaction_SU2, d_mixed_trilinear_partners_SU2,
        tmp_SU3, num_bi_SU2, num_tri_SU2, num_tri_SU2_SU3,
        max_bi_neighbors_SU2, max_tri_neighbors_SU2, max_mixed_tri_neighbors_SU2);
        
    compute_local_field_SU3_kernel<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3><<<grid_size_SU3, block_size>>>(
        d_local_field_SU3, tmp_SU3, d_field_SU3, d_onsite_interaction_SU3,
        d_bilinear_interaction_SU3, d_bilinear_partners_SU3,
        d_trilinear_interaction_SU3, d_trilinear_partners_SU3,
        d_mixed_trilinear_interaction_SU3, d_mixed_trilinear_partners_SU3,
        tmp_SU2, num_bi_SU3, num_tri_SU3, num_tri_SU2_SU3,
        max_bi_neighbors_SU3, max_tri_neighbors_SU3, max_mixed_tri_neighbors_SU3);
    
    drive_field_T_SU2_kernel<N_SU2, lattice_size_SU2><<<grid_size_SU2, block_size>>>(
        curr_time + c3_h, d_local_field_SU2, d_field_drive_1_SU2, d_field_drive_2_SU2, 
        d_field_drive_amp_SU2, d_field_drive_width_SU2, d_field_drive_freq_SU2, d_t_B_1_SU2, d_t_B_2_SU2);

    landau_Lifshitz_SU2_kernel<N_SU2, lattice_size_SU2> <<<grid_size_SU2, block_size>>>(
        k_SU2, tmp_SU2, d_local_field_SU2);

    landau_Lifshitz_SU3_kernel<N_SU3, lattice_size_SU3> <<<grid_size_SU3, block_size>>>(
        k_SU3, tmp_SU3, d_local_field_SU3);

    update_arrays_three_kernel<<<grid_size_SU2, block_size>>>(d_spins_SU2, u_SU2, a52, tmp_SU2, a54, k_SU2, b54_h, lattice_size_SU2 * N_SU2);
    update_arrays_three_kernel<<<grid_size_SU3, block_size>>>(d_spins_SU3, u_SU3, a52, tmp_SU3, a54, k_SU3, b54_h, lattice_size_SU3 * N_SU3);

    // Cleanup temporary arrays - check if pointers are valid before freeing
    if (tmp_SU2) cudaFree(tmp_SU2);
    if (k_SU2) cudaFree(k_SU2);
    if (u_SU2) cudaFree(u_SU2);
    if (tmp_SU3) cudaFree(tmp_SU3);
    if (k_SU3) cudaFree(k_SU3);
    if (u_SU3) cudaFree(u_SU3);
    if (d_drive_field_SU2) cudaFree(d_drive_field_SU2);
    if (d_drive_field_SU3) cudaFree(d_drive_field_SU3);
}


template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
__host__
void mixed_lattice_cuda<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>::SSPRK53_step_cuda(double step_size, double curr_time, double tol) {
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
    // Print some diagnostic info
    fprintf(stdout, "SSPRK53_step_cuda info: lattice_size_SU2=%zu, lattice_size_SU3=%zu, curr_time=%f, step_size=%f\n", 
            (size_t)(N_ATOMS_SU2*dim*dim1*dim2), (size_t)(N_ATOMS_SU3*dim*dim1*dim2), curr_time, step_size);
    
    // Check if any CUDA arrays are NULL before passing to the kernel
    if (d_spins.spins_SU2 == nullptr || d_spins.spins_SU3 == nullptr) {
        fprintf(stderr, "CUDA error: NULL pointer in d_spins detected\n");
        cudaFree(d_local_fields_SU2);
        cudaFree(d_local_fields_SU3);
        return;
    }

    // Execute SSPRK53 step
    SSPRK53_step_kernel<N_SU2, N_ATOMS_SU2*dim*dim1*dim2, N_SU3, N_ATOMS_SU3*dim*dim1*dim2>(
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
    curr_time, step_size);    
    
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


//Explicitly declare template specializations for the mixed lattice class
template class mixed_lattice_cuda<3, 4, 8, 4, 8, 8, 8>;