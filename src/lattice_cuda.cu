#include "lattice_cuda.cuh"

// CUDA device functions for vector operations
static __device__
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

static __device__
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

static __device__
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

static __device__
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

static __device__
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

static __device__
void cross_product_device(double* result, const double* a, const double* b, size_t n) {
    if (n == 3) {
        // Standard 3D cross product for SU(2)
        result[0] = a[1] * b[2] - a[2] * b[1];
        result[1] = a[2] * b[0] - a[0] * b[2];
        result[2] = a[0] * b[1] - a[1] * b[0];
    } else if (n == 8) {
        // SU(3) Lie algebra cross product
        constexpr double sqrt3_2 = 0.86602540378443864676372317075294; // sqrt(3)/2
        
        result[0] = a[1]*b[2] - a[2]*b[1] + 0.5*(a[3]*b[6] - a[6]*b[3]) - 0.5*(a[4]*b[5] - a[5]*b[4]);
        result[1] = a[2]*b[0] - a[0]*b[2] + 0.5*(a[3]*b[5] - a[5]*b[3]) + 0.5*(a[4]*b[6] - a[6]*b[4]);
        result[2] = a[0]*b[1] - a[1]*b[0] + 0.5*(a[3]*b[4] - a[4]*b[3]) - 0.5*(a[5]*b[6] - a[6]*b[5]);
        result[3] = 0.5*(a[6]*b[0] - a[0]*b[6]) + 0.5*(a[5]*b[1] - a[1]*b[5]) + 0.5*(a[4]*b[2] - a[2]*b[4]) + sqrt3_2*(a[4]*b[7] - a[7]*b[4]);
        result[4] = -0.5*(a[5]*b[0] - a[0]*b[5]) + 0.5*(a[6]*b[1] - a[1]*b[6]) + 0.5*(a[2]*b[3] - a[3]*b[2]) + sqrt3_2*(a[7]*b[3] - a[3]*b[7]);
        result[5] = -0.5*(a[0]*b[4] - a[4]*b[0]) + 0.5*(a[1]*b[3] - a[3]*b[1]) - 0.5*(a[6]*b[2] - a[2]*b[6]) + sqrt3_2*(a[6]*b[7] - a[7]*b[6]);
        result[6] = 0.5*(a[0]*b[3] - a[3]*b[0]) + 0.5*(a[1]*b[4] - a[4]*b[1]) - 0.5*(a[2]*b[5] - a[5]*b[2]) + sqrt3_2*(a[7]*b[5] - a[5]*b[7]);
        result[7] = sqrt3_2*(a[3]*b[4] - a[4]*b[3]) + sqrt3_2*(a[5]*b[6] - a[6]*b[5]);
    }
}

// Device function to compute local field
template <size_t N, size_t lattice_size>
__device__
void compute_local_field(
    double* out,
    int site_index,
    const double* spins,
    const InteractionData<N>& interactions,
    const NeighborCounts& neighbors
) {
    double local_field[N] = {0.0};

    // Onsite contribution
    const size_t onsite_base = site_index * N * N;
    const double* spin_site = &spins[site_index * N];
    
    #pragma unroll
    for (size_t i = 0; i < N; ++i) {
        #pragma unroll
        for (size_t j = 0; j < N; ++j) {
            local_field[i] += interactions.onsite_interaction[onsite_base + i * N + j] * spin_site[j];
        }
    }
    
    // Bilinear contributions
    if (neighbors.num_bi > 0) {
        const size_t bi_base = site_index * neighbors.max_bi_neighbors;
        const size_t bi_interaction_base = bi_base * N * N;
        
        for (size_t idx = 0; idx < neighbors.num_bi && idx < neighbors.max_bi_neighbors; ++idx) {
            size_t partner = interactions.bilinear_partners[bi_base + idx];
            if (partner < lattice_size) {
                const double* partner_spin = &spins[partner * N];
                const double* J_matrix = &interactions.bilinear_interaction[bi_interaction_base + idx * N * N];
                
                #pragma unroll
                for (size_t i = 0; i < N; ++i) {
                    #pragma unroll
                    for (size_t j = 0; j < N; ++j) {
                        local_field[i] += J_matrix[i * N + j] * partner_spin[j];
                    }
                }
            }
        }
    }
    
    // Trilinear contributions
    if (neighbors.num_tri > 0) {
        const size_t tri_base = site_index * neighbors.max_tri_neighbors;
        const size_t tri_partner_base = tri_base * 2;
        const size_t tri_interaction_base = tri_base * N * N * N;
        
        for (size_t idx = 0; idx < neighbors.num_tri && idx < neighbors.max_tri_neighbors; ++idx) {
            size_t partner1 = interactions.trilinear_partners[tri_partner_base + idx * 2];
            size_t partner2 = interactions.trilinear_partners[tri_partner_base + idx * 2 + 1];
            
            if (partner1 < lattice_size && partner2 < lattice_size) {
                const double* partner_spin1 = &spins[partner1 * N];
                const double* partner_spin2 = &spins[partner2 * N];
                const double* K_tensor = &interactions.trilinear_interaction[tri_interaction_base + idx * N * N * N];
                
                // Contract tensor: result[i] += sum_{j,k} K[i,j,k] * S1[j] * S2[k]
                double temp_field[N];
                contract_trilinear_field_device(temp_field, K_tensor, partner_spin1, partner_spin2, N);
                
                #pragma unroll
                for (size_t i = 0; i < N; ++i) {
                    local_field[i] += temp_field[i];
                }
            }
        }
    }
    
    // Subtract external field
    const double* field_ptr = &interactions.field[site_index * N];
    #pragma unroll
    for (size_t i = 0; i < N; ++i) {
        out[site_index * N + i] = local_field[i] - field_ptr[i];
    }
}

// Device function for Landau-Lifshitz dynamics
template <size_t N, size_t lattice_size>
__device__
void landau_Lifshitz(double* out, int site_index, double* spins, const double* local_field) {
    double spin[N], local_field_here[N];
    
    #pragma unroll
    for (size_t i = 0; i < N; ++i) {
        spin[i] = spins[site_index * N + i];
        local_field_here[i] = local_field[site_index * N + i];
    }

    // Compute the cross product with the local field
    double cross[N];
    cross_product_device(cross, spin, local_field_here, N);

    // Update the spins using the Landau-Lifshitz equation
    #pragma unroll
    for (size_t i = 0; i < N; ++i) {
        out[site_index * N + i] = cross[i];
    }
}

// Device function for drive field
template <size_t N, size_t N_ATOMS, size_t lattice_size>
__device__
void drive_field_T(
    double* out, double currT, int site_index,
    double* d_field_drive_1, double* d_field_drive_2, 
    double d_field_drive_amp, double d_field_drive_width, 
    double d_field_drive_freq, double d_t_B_1, double d_t_B_2, 
    size_t max_tri_neighbors, double* trilinear_interaction, 
    size_t* trilinear_partners, double* d_spins)
{
    // Pre-compute common exponential terms
    const double dt1 = currT - d_t_B_1;
    const double dt2 = currT - d_t_B_2;
    const double inv_2width_sq = 1.0 / (4.0 * d_field_drive_width * d_field_drive_width);
    const double exp1 = exp(-dt1 * dt1 * inv_2width_sq);
    const double exp2 = exp(-dt2 * dt2 * inv_2width_sq);
    const double omega = 2.0 * M_PI * d_field_drive_freq;
    
    // Compute factors once
    const double factor1 = d_field_drive_amp * exp1 * cos(omega * dt1);
    const double factor2 = d_field_drive_amp * exp2 * cos(omega * dt2);
    
    // Cache sublattice index for site
    const int site_sublattice = site_index % N_ATOMS;
    const size_t site_sublattice_base = site_sublattice * N;

    if (factor1 < 1e-14 && factor2 < 1e-14) return;
    
    // Add drive field contribution
    #pragma unroll
    for (size_t i = 0; i < N; ++i) {
        out[site_index * N + i] += factor1 * d_field_drive_1[site_sublattice_base + i] +
                                   factor2 * d_field_drive_2[site_sublattice_base + i];
    }
}

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
) {
    int site_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (site_index >= lattice_size) return;
    
    double energy = 0.0;
    const double* spin_here = &spins[site_index * N];
    
    // Field energy
    #pragma unroll
    for (size_t i = 0; i < N; ++i) {
        energy -= spin_here[i] * field[site_index * N + i];
    }
    
    // Onsite energy
    energy += contract_device(spin_here, &onsite_interaction[site_index * N * N], spin_here, N) / 2.0;
    
    // Bilinear interactions
    for (size_t i = 0; i < num_bi && i < max_bi_neighbors; ++i) {
        size_t partner = bilinear_partners[site_index * max_bi_neighbors + i];
        if (partner < lattice_size) {
            energy += contract_device(spin_here, 
                &bilinear_interaction[site_index * max_bi_neighbors * N * N + i * N * N],
                &spins[partner * N], N) / 2.0;
        }
    }
    
    // Trilinear interactions
    for (size_t i = 0; i < num_tri && i < max_tri_neighbors; ++i) {
        size_t partner1 = trilinear_partners[site_index * max_tri_neighbors * 2 + i * 2];
        size_t partner2 = trilinear_partners[site_index * max_tri_neighbors * 2 + i * 2 + 1];
        if (partner1 < lattice_size && partner2 < lattice_size) {
            energy += contract_trilinear_device(
                &trilinear_interaction[site_index * max_tri_neighbors * N * N * N + i * N * N * N],
                spin_here, &spins[partner1 * N], &spins[partner2 * N], N) / 3.0;
        }
    }
    
    d_energies[site_index] = energy;
}

// CUDA kernel for Landau-Lifshitz-Gilbert equation
template<size_t N, size_t N_ATOMS, size_t lattice_size>
__global__
void LLG_kernel(
    double* k, double* d_spins, double* d_local_field,
    InteractionData<N> interactions,
    NeighborCounts neighbors,
    FieldDriveParams<N, N_ATOMS> field_drive,
    TimeStepParams time_params)
{
    int site_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (site_index >= lattice_size) return;

    // Compute local field
    compute_local_field<N, lattice_size>(
        d_local_field, site_index, d_spins, interactions, neighbors);

    // Add drive field if applicable
    drive_field_T<N, N_ATOMS, lattice_size>(
        d_local_field, time_params.curr_time, site_index, 
        field_drive.field_drive_1, field_drive.field_drive_2, 
        field_drive.field_drive_amp, field_drive.field_drive_width, 
        field_drive.field_drive_freq, field_drive.t_B_1, field_drive.t_B_2,
        neighbors.max_tri_neighbors, interactions.trilinear_interaction, 
        interactions.trilinear_partners, d_spins);
    
    // Compute derivatives (dS/dt) using Landau-Lifshitz equation
    landau_Lifshitz<N, lattice_size>(k, site_index, d_spins, d_local_field);
}

// Kernel to update arrays with linear combination
template <size_t N, size_t lattice_size>
__global__ 
void update_arrays_kernel(double* out, const double* in1, double a1, 
                         const double* in2, double a2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lattice_size) return;
    
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        out[idx * N + i] = a1 * in1[idx * N + i] + a2 * in2[idx * N + i];
    }
}

// Kernel to update arrays with linear combination of three arrays
template <size_t N, size_t lattice_size>
__global__ 
void update_arrays_three_kernel(double* out, 
                               const double* in1, double a1,
                               const double* in2, double a2,
                               const double* in3, double a3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lattice_size) return;
    
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        out[idx * N + i] = a1 * in1[idx * N + i] + 
                          a2 * in2[idx * N + i] + 
                          a3 * in3[idx * N + i];
    }
}

// Kernel to normalize spins
template <size_t N, size_t lattice_size>
__global__ 
void normalize_spins_kernel(double* spins, double spin_length) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    double* spin = &spins[site * N];
    double norm = 0.0;
    
    #pragma unroll
    for (int i = 0; i < N; i++) {
        norm += spin[i] * spin[i];
    }
    norm = sqrt(norm);
    
    if (norm > 1e-10) {
        #pragma unroll
        for (int i = 0; i < N; i++) {
            spin[i] = spin[i] * spin_length / norm;
        }
    }
}

// Memory pool class for SSPRK53 working arrays
template<size_t N, size_t lattice_size>
class SSPRK53_MemoryPool {
private:
    double* work_1;
    double* work_2;
    double* work_3;
    bool allocated;

public:
    SSPRK53_MemoryPool() : work_1(nullptr), work_2(nullptr), work_3(nullptr), allocated(false) {
        allocate();
    }
    
    ~SSPRK53_MemoryPool() {
        deallocate();
    }
    
    void allocate() {
        if (allocated) return;
        
        cudaMalloc(&work_1, lattice_size * N * sizeof(double));
        cudaMalloc(&work_2, lattice_size * N * sizeof(double));
        cudaMalloc(&work_3, lattice_size * N * sizeof(double));
        allocated = true;
    }
    
    void deallocate() {
        if (!allocated) return;
        
        if (work_1) cudaFree(work_1);
        if (work_2) cudaFree(work_2);
        if (work_3) cudaFree(work_3);
        
        work_1 = work_2 = work_3 = nullptr;
        allocated = false;
    }
    
    double* get_work_1() const { return work_1; }
    double* get_work_2() const { return work_2; }
    double* get_work_3() const { return work_3; }
    bool is_allocated() const { return allocated; }
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
    WorkingArrays work_arrays)
{
    // SSPRK53 coefficients
    constexpr double a30 = 0.355909775063327;
    constexpr double a32 = 0.644090224936674;
    constexpr double a40 = 0.367933791638137;
    constexpr double a43 = 0.632066208361863;
    constexpr double a50 = 0.621590188814956;
    constexpr double a53 = 0.173645142993633;
    constexpr double a54 = 0.204764668191411;
    
    dim3 block(256);
    dim3 grid((lattice_size + block.x - 1) / block.x);
    
    double dt = time_params.dt;
    double curr_time = time_params.curr_time;
    
    // Stage 1: k1 = f(t, S)
    time_params.curr_time = curr_time;
    LLG_kernel<N, N_ATOMS, lattice_size><<<grid, block>>>(
        work_arrays.work_1, d_spins, d_local_field, interactions, neighbors, field_drive, time_params);
    
    // Stage 2: S2 = S + dt * k1, k2 = f(t + dt, S2)
    update_arrays_kernel<N, lattice_size><<<grid, block>>>(
        work_arrays.work_2, d_spins, 1.0, work_arrays.work_1, dt);
    normalize_spins_kernel<N, lattice_size><<<grid, block>>>(
        work_arrays.work_2, time_params.spin_length);
    
    time_params.curr_time = curr_time + dt;
    LLG_kernel<N, N_ATOMS, lattice_size><<<grid, block>>>(
        work_arrays.work_3, work_arrays.work_2, d_local_field, interactions, neighbors, field_drive, time_params);
    
    // Stage 3: S3 = a30*S + a32*(S2 + dt*k2)
    update_arrays_kernel<N, lattice_size><<<grid, block>>>(
        work_arrays.work_2, work_arrays.work_2, 1.0, work_arrays.work_3, dt);
    update_arrays_kernel<N, lattice_size><<<grid, block>>>(
        work_arrays.work_2, d_spins, a30, work_arrays.work_2, a32);
    normalize_spins_kernel<N, lattice_size><<<grid, block>>>(
        work_arrays.work_2, time_params.spin_length);
    
    // k3 = f(t + dt, S3)
    LLG_kernel<N, N_ATOMS, lattice_size><<<grid, block>>>(
        work_arrays.work_3, work_arrays.work_2, d_local_field, interactions, neighbors, field_drive, time_params);
    
    // Stage 4: S4 = a40*S + a43*(S3 + dt*k3)
    update_arrays_kernel<N, lattice_size><<<grid, block>>>(
        work_arrays.work_2, work_arrays.work_2, 1.0, work_arrays.work_3, dt);
    update_arrays_kernel<N, lattice_size><<<grid, block>>>(
        work_arrays.work_2, d_spins, a40, work_arrays.work_2, a43);
    normalize_spins_kernel<N, lattice_size><<<grid, block>>>(
        work_arrays.work_2, time_params.spin_length);
    
    // k4 = f(t + dt, S4)
    LLG_kernel<N, N_ATOMS, lattice_size><<<grid, block>>>(
        work_arrays.work_3, work_arrays.work_2, d_local_field, interactions, neighbors, field_drive, time_params);
    
    // Stage 5: S_new = a50*S + a53*(S3 + dt*k3) + a54*(S4 + dt*k4)
    // First compute S3_updated = S3 + dt*k3 (reuse work_arrays.work_1 for k3 storage)
    time_params.curr_time = curr_time + dt;
    LLG_kernel<N, N_ATOMS, lattice_size><<<grid, block>>>(
        work_arrays.work_1, work_arrays.work_2, d_local_field, interactions, neighbors, field_drive, time_params);
    
    // Compute S4_updated = S4 + dt*k4 (in work_arrays.work_2)
    update_arrays_kernel<N, lattice_size><<<grid, block>>>(
        work_arrays.work_2, work_arrays.work_2, 1.0, work_arrays.work_3, dt);
    
    // Now compute S_new = a50*S + a53*S3_updated + a54*S4_updated
    update_arrays_three_kernel<N, lattice_size><<<grid, block>>>(
        d_spins, d_spins, a50, work_arrays.work_1, a53, work_arrays.work_2, a54);
    normalize_spins_kernel<N, lattice_size><<<grid, block>>>(
        d_spins, time_params.spin_length);
    
    cudaDeviceSynchronize();
}

// Euler step kernel
template<size_t N, size_t N_ATOMS, size_t lattice_size>
__host__
void euler_step_kernel(
    double* d_spins, double* d_local_field,
    InteractionData<N> interactions,
    NeighborCounts neighbors,
    FieldDriveParams<N, N_ATOMS> field_drive,
    TimeStepParams time_params)
{
    dim3 block(256);
    dim3 grid((lattice_size + block.x - 1) / block.x);
    
    double* d_k;
    cudaMalloc(&d_k, lattice_size * N * sizeof(double));
    
    // Compute k = f(t, S)
    LLG_kernel<N, N_ATOMS, lattice_size><<<grid, block>>>(
        d_k, d_spins, d_local_field, interactions, neighbors, field_drive, time_params);
    
    // Update: S_new = S + dt * k
    update_arrays_kernel<N, lattice_size><<<grid, block>>>(
        d_spins, d_spins, 1.0, d_k, time_params.dt);
    normalize_spins_kernel<N, lattice_size><<<grid, block>>>(
        d_spins, time_params.spin_length);
    
    cudaDeviceSynchronize();
    cudaFree(d_k);
}

// Template implementations for lattice_cuda class
template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
void lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::calculate_max_neighbors() {
    max_bilinear_neighbors = 0;
    max_trilinear_neighbors = 0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        size_t bi_count = this->bilinear_partners[i].size();
        size_t tri_count = this->trilinear_partners[i].size();
        
        if (bi_count > max_bilinear_neighbors) max_bilinear_neighbors = bi_count;
        if (tri_count > max_trilinear_neighbors) max_trilinear_neighbors = tri_count;
    }
    
    d_num_bi = this->num_bi;
    d_num_tri = this->num_tri;
}

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
void lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::allocate_cuda_memory() {
    // Allocate spin and position arrays
    d_spins.allocate();
    d_site_pos.allocate();
    
    // Allocate field arrays
    cudaMalloc(&d_field, lattice_size * N * sizeof(double));
    cudaMalloc(&d_field_drive_1, N_ATOMS * N * sizeof(double));
    cudaMalloc(&d_field_drive_2, N_ATOMS * N * sizeof(double));
    
    // Allocate onsite interaction
    cudaMalloc(&d_onsite_interaction, lattice_size * N * N * sizeof(double));
    
    // Allocate bilinear interactions (ensure at least size 1 to avoid zero-size allocations)
    size_t bi_alloc_size = max_bilinear_neighbors > 0 ? max_bilinear_neighbors : 1;
    cudaMalloc(&d_bilinear_interaction, lattice_size * bi_alloc_size * N * N * sizeof(double));
    cudaMalloc(&d_bilinear_partners, lattice_size * bi_alloc_size * sizeof(size_t));
    
    // Allocate trilinear interactions (ensure at least size 1 to avoid zero-size allocations)
    size_t tri_alloc_size = max_trilinear_neighbors > 0 ? max_trilinear_neighbors : 1;
    cudaMalloc(&d_trilinear_interaction, lattice_size * tri_alloc_size * N * N * N * sizeof(double));
    cudaMalloc(&d_trilinear_partners, lattice_size * tri_alloc_size * 2 * sizeof(size_t));
    
    // Allocate random states
    cudaMalloc(&d_rng_states, lattice_size * sizeof(curandState));
}

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
void lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::copy_data_to_device() {
    // Copy spins
    std::vector<double> spins_flat(lattice_size * N);
    for (size_t i = 0; i < lattice_size; ++i) {
        for (size_t j = 0; j < N; ++j) {
            spins_flat[i * N + j] = this->spins[i][j];
        }
    }
    cudaMemcpy(d_spins.spins, spins_flat.data(), lattice_size * N * sizeof(double), cudaMemcpyHostToDevice);
    
    // Copy positions
    std::vector<double> pos_flat(lattice_size * 3);
    for (size_t i = 0; i < lattice_size; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            pos_flat[i * 3 + j] = this->site_pos[i][j];
        }
    }
    cudaMemcpy(d_site_pos.pos, pos_flat.data(), lattice_size * 3 * sizeof(double), cudaMemcpyHostToDevice);
    
    // Copy fields
    std::vector<double> field_flat(lattice_size * N);
    for (size_t i = 0; i < lattice_size; ++i) {
        for (size_t j = 0; j < N; ++j) {
            field_flat[i * N + j] = this->field[i][j];
        }
    }
    cudaMemcpy(d_field, field_flat.data(), lattice_size * N * sizeof(double), cudaMemcpyHostToDevice);
    
    // Copy onsite interactions
    std::vector<double> onsite_flat(lattice_size * N * N);
    for (size_t i = 0; i < lattice_size; ++i) {
        for (size_t j = 0; j < N * N; ++j) {
            onsite_flat[i * N * N + j] = this->onsite_interaction[i][j];
        }
    }
    cudaMemcpy(d_onsite_interaction, onsite_flat.data(), lattice_size * N * N * sizeof(double), cudaMemcpyHostToDevice);
    
    // Initialize field drive arrays to zero (not used unless explicitly set)
    std::vector<double> field_drive_flat(N_ATOMS * N, 0.0);
    cudaMemcpy(d_field_drive_1, field_drive_flat.data(), N_ATOMS * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field_drive_2, field_drive_flat.data(), N_ATOMS * N * sizeof(double), cudaMemcpyHostToDevice);
    
    // Copy interaction arrays
    copy_interaction_arrays_to_device();
}

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
void lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::copy_interaction_arrays_to_device() {
    // Copy bilinear interactions
    size_t bi_alloc_size = max_bilinear_neighbors > 0 ? max_bilinear_neighbors : 1;
    std::vector<double> bi_flat(lattice_size * bi_alloc_size * N * N, 0.0);
    std::vector<size_t> bi_partners_flat(lattice_size * bi_alloc_size, lattice_size);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        size_t num_neighbors = this->bilinear_partners[i].size();
        for (size_t j = 0; j < num_neighbors && j < max_bilinear_neighbors; ++j) {
            bi_partners_flat[i * max_bilinear_neighbors + j] = this->bilinear_partners[i][j];
            for (size_t k = 0; k < N * N; ++k) {
                bi_flat[(i * max_bilinear_neighbors + j) * N * N + k] = this->bilinear_interaction[i][j][k];
            }
        }
    }
    
    cudaMemcpy(d_bilinear_interaction, bi_flat.data(), 
               lattice_size * bi_alloc_size * N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bilinear_partners, bi_partners_flat.data(), 
               lattice_size * bi_alloc_size * sizeof(size_t), cudaMemcpyHostToDevice);
    
    // Copy trilinear interactions
    size_t tri_alloc_size = max_trilinear_neighbors > 0 ? max_trilinear_neighbors : 1;
    std::vector<double> tri_flat(lattice_size * tri_alloc_size * N * N * N, 0.0);
    std::vector<size_t> tri_partners_flat(lattice_size * tri_alloc_size * 2, lattice_size);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        size_t num_neighbors = this->trilinear_partners[i].size();
        for (size_t j = 0; j < num_neighbors && j < max_trilinear_neighbors; ++j) {
            tri_partners_flat[i * max_trilinear_neighbors * 2 + j * 2] = this->trilinear_partners[i][j][0];
            tri_partners_flat[i * max_trilinear_neighbors * 2 + j * 2 + 1] = this->trilinear_partners[i][j][1];
            for (size_t k = 0; k < N * N * N; ++k) {
                tri_flat[(i * max_trilinear_neighbors + j) * N * N * N + k] = this->trilinear_interaction[i][j][k];
            }
        }
    }
    
    cudaMemcpy(d_trilinear_interaction, tri_flat.data(), 
               lattice_size * tri_alloc_size * N * N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trilinear_partners, tri_partners_flat.data(), 
               lattice_size * tri_alloc_size * 2 * sizeof(size_t), cudaMemcpyHostToDevice);
}

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
void lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::cleanup_cuda_memory() {
    d_spins.deallocate();
    d_site_pos.deallocate();
    
    if (d_field) cudaFree(d_field);
    if (d_field_drive_1) cudaFree(d_field_drive_1);
    if (d_field_drive_2) cudaFree(d_field_drive_2);
    if (d_onsite_interaction) cudaFree(d_onsite_interaction);
    if (d_bilinear_interaction) cudaFree(d_bilinear_interaction);
    if (d_bilinear_partners) cudaFree(d_bilinear_partners);
    if (d_trilinear_interaction) cudaFree(d_trilinear_interaction);
    if (d_trilinear_partners) cudaFree(d_trilinear_partners);
    if (d_rng_states) cudaFree(d_rng_states);
}

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
double lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::total_energy_cuda() {
    double* d_energies;
    cudaMalloc(&d_energies, lattice_size * sizeof(double));
    
    dim3 block(256);
    dim3 grid((lattice_size + block.x - 1) / block.x);
    
    compute_site_energy_kernel<N, lattice_size><<<grid, block>>>(
        d_energies, d_spins.spins, d_field, d_onsite_interaction,
        d_bilinear_interaction, d_bilinear_partners,
        d_trilinear_interaction, d_trilinear_partners,
        d_num_bi, d_num_tri, max_bilinear_neighbors, max_trilinear_neighbors
    );
    
    thrust::device_ptr<double> d_ptr(d_energies);
    double total_energy = thrust::reduce(d_ptr, d_ptr + lattice_size);
    
    cudaFree(d_energies);
    return total_energy;
}

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
void lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::SSPRK53_step_cuda(double step_size, double curr_time, double tol) {
    static SSPRK53_MemoryPool<N, lattice_size> pool;
    
    InteractionData<N> interactions = {
        d_field, d_onsite_interaction, d_bilinear_interaction,
        d_bilinear_partners, d_trilinear_interaction, d_trilinear_partners
    };
    
    NeighborCounts neighbors = {
        d_num_bi, d_num_tri, max_bilinear_neighbors, max_trilinear_neighbors
    };
    
    FieldDriveParams<N, N_ATOMS> field_drive = {
        d_field_drive_1, d_field_drive_2, d_field_drive_amp,
        d_field_drive_width, d_field_drive_freq, d_t_B_1, d_t_B_2
    };
    
    TimeStepParams time_params = {curr_time, step_size, d_spin_length};
    
    double* d_local_field;
    cudaMalloc(&d_local_field, lattice_size * N * sizeof(double));
    
    WorkingArrays work = {pool.get_work_1(), pool.get_work_2(), pool.get_work_3()};
    
    SSPRK53_step_kernel<N, N_ATOMS, lattice_size>(
        d_spins.spins, d_local_field, interactions, neighbors, field_drive, time_params, work);
    
    cudaFree(d_local_field);
}

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
void lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::euler_step_cuda(double step_size, double curr_time, double tol) {
    InteractionData<N> interactions = {
        d_field, d_onsite_interaction, d_bilinear_interaction,
        d_bilinear_partners, d_trilinear_interaction, d_trilinear_partners
    };
    
    NeighborCounts neighbors = {
        d_num_bi, d_num_tri, max_bilinear_neighbors, max_trilinear_neighbors
    };
    
    FieldDriveParams<N, N_ATOMS> field_drive = {
        d_field_drive_1, d_field_drive_2, d_field_drive_amp,
        d_field_drive_width, d_field_drive_freq, d_t_B_1, d_t_B_2
    };
    
    TimeStepParams time_params = {curr_time, step_size, d_spin_length};
    
    double* d_local_field;
    cudaMalloc(&d_local_field, lattice_size * N * sizeof(double));
    
    euler_step_kernel<N, N_ATOMS, lattice_size>(
        d_spins.spins, d_local_field, interactions, neighbors, field_drive, time_params);
    
    cudaFree(d_local_field);
}

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
void lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::molecular_dynamics_cuda(
    double T_start, double T_end, double step_size, std::string dir_name, 
    bool save_all, size_t save_interval)
{
    std::cout << "Starting GPU molecular dynamics simulation..." << std::endl;
    std::cout << "  Time range: " << T_start << " to " << T_end << std::endl;
    std::cout << "  Step size: " << step_size << std::endl;
    
    // Create output directory
    std::filesystem::create_directories(dir_name);
    
    size_t num_steps = static_cast<size_t>((T_end - T_start) / step_size);
    double curr_time = T_start;
    
    // Pre-allocate magnetization arrays
    std::vector<array<double, N>> mag_local_history;
    std::vector<double> energy_history;
    
    auto start_wall = std::chrono::high_resolution_clock::now();
    
    for (size_t step = 0; step < num_steps; ++step) {
        // Perform time evolution
        SSPRK53_step_cuda(step_size, curr_time, 1e-6);
        curr_time += step_size;
        
        // Save data at intervals
        if (step % save_interval == 0) {
            // Copy spins back to host for measurement
            copy_spins_to_host();
            
            // Compute magnetization on CPU
            array<double, N> mag_local = this->magnetization_global(this->spins);
            mag_local_history.push_back(mag_local);
            
            // Compute energy on GPU
            double energy = total_energy_cuda();
            energy_history.push_back(energy);
            
            if (step % (save_interval * 10) == 0) {
                std::cout << "Step " << step << "/" << num_steps 
                         << ", Time = " << curr_time 
                         << ", Energy = " << energy << std::endl;
            }
        }
    }
    
    auto end_wall = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_wall - start_wall;
    
    std::cout << "Simulation completed in " << elapsed.count() << " seconds" << std::endl;
    
    // Save final results
    copy_spins_to_host();
    this->write_to_file(dir_name + "/final_spins.txt", this->spins);
    
    // Save magnetization history
    this->write_to_file_2d_vector_array(dir_name + "/magnetization.txt", mag_local_history);
    
    // Save energy history
    this->write_column_vector(dir_name + "/energy.txt", energy_history);
}

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
void lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::copy_spins_to_host() {
    std::vector<double> spins_flat(lattice_size * N);
    cudaMemcpy(spins_flat.data(), d_spins.spins, lattice_size * N * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        for (size_t j = 0; j < N; ++j) {
            this->spins[i][j] = spins_flat[i * N + j];
        }
    }
}

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
void lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::copy_spins_to_device() {
    std::vector<double> spins_flat(lattice_size * N);
    for (size_t i = 0; i < lattice_size; ++i) {
        for (size_t j = 0; j < N; ++j) {
            spins_flat[i * N + j] = this->spins[i][j];
        }
    }
    cudaMemcpy(d_spins.spins, spins_flat.data(), lattice_size * N * sizeof(double), cudaMemcpyHostToDevice);
}

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
void lattice_cuda<N, N_ATOMS, dim1, dim2, dim>::print_lattice_info_cuda(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    file << "========================================\n";
    file << "CUDA Lattice Information\n";
    file << "========================================\n\n";
    file << "Lattice dimensions: " << dim1 << " x " << dim2 << " x " << dim << "\n";
    file << "Number of atoms per unit cell: " << N_ATOMS << "\n";
    file << "Total lattice sites: " << lattice_size << "\n";
    file << "Spin dimension (N): " << N << "\n";
    file << "Spin length: " << d_spin_length << "\n\n";
    
    file << "Interaction information:\n";
    file << "  Max bilinear neighbors: " << max_bilinear_neighbors << "\n";
    file << "  Max trilinear neighbors: " << max_trilinear_neighbors << "\n";
    file << "  Number of bilinear interactions: " << d_num_bi << "\n";
    file << "  Number of trilinear interactions: " << d_num_tri << "\n\n";
    
    file << "Memory allocation:\n";
    file << "  Spins: " << (lattice_size * N * sizeof(double)) / (1024.0 * 1024.0) << " MB\n";
    file << "  Fields: " << (lattice_size * N * sizeof(double)) / (1024.0 * 1024.0) << " MB\n";
    file << "  Bilinear interactions: " << (lattice_size * max_bilinear_neighbors * N * N * sizeof(double)) / (1024.0 * 1024.0) << " MB\n";
    file << "  Trilinear interactions: " << (lattice_size * max_trilinear_neighbors * N * N * N * sizeof(double)) / (1024.0 * 1024.0) << " MB\n";
    
    file.close();
    std::cout << "Lattice information saved to " << filename << std::endl;
}

// Explicit template instantiations for commonly used lattice configurations
template class lattice_cuda<3, 2, 24, 24, 1>;
template class lattice_cuda<3, 2, 20, 20, 1>;
template class lattice_cuda<3, 2, 12, 12, 1>;
template class lattice_cuda<3, 1, 16, 16, 16>;
template class lattice_cuda<8, 1, 8, 8, 8>;
template class lattice_cuda<3, 2, 4, 4, 1>;
