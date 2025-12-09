#include "classical_spin/gpu/gpu_common_helpers.cuh"

// ======================= Common GPU Kernels =======================
// 
// Device helper functions are defined inline in the header file (gpu_common_helpers.cuh)
// to allow cross-translation-unit usage with CUDA.
// Only global kernels are defined here.

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

// ======================= Fused RK Kernels for Performance =======================
// These kernels combine multiple array updates to reduce kernel launch overhead

__global__
void rk4_final_update_kernel(
    double* state,
    const double* k1, const double* k2,
    const double* k3, const double* k4,
    double dt_over_6,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // state += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    state[idx] += dt_over_6 * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]);
}

__global__
void dopri5_final_update_kernel(
    double* state,
    const double* k1, const double* k3,
    const double* k4, const double* k5, const double* k6,
    double dt, double b1, double b3, double b4, double b5, double b6,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // state += dt * (b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
    state[idx] += dt * (b1 * k1[idx] + b3 * k3[idx] + b4 * k4[idx] + 
                        b5 * k5[idx] + b6 * k6[idx]);
}

__global__
void rk_stage_update_2_kernel(
    double* out,
    const double* state,
    const double* k1, double a1,
    const double* k2, double a2,
    double dt,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    out[idx] = state[idx] + dt * (a1 * k1[idx] + a2 * k2[idx]);
}

__global__
void rk_stage_update_3_kernel(
    double* out,
    const double* state,
    const double* k1, double a1,
    const double* k2, double a2,
    const double* k3, double a3,
    double dt,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    out[idx] = state[idx] + dt * (a1 * k1[idx] + a2 * k2[idx] + a3 * k3[idx]);
}

__global__
void rk_stage_update_4_kernel(
    double* out,
    const double* state,
    const double* k1, double a1,
    const double* k2, double a2,
    const double* k3, double a3,
    const double* k4, double a4,
    double dt,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    out[idx] = state[idx] + dt * (a1 * k1[idx] + a2 * k2[idx] + 
                                   a3 * k3[idx] + a4 * k4[idx]);
}

__global__
void rk_stage_update_5_kernel(
    double* out,
    const double* state,
    const double* k1, double a1,
    const double* k2, double a2,
    const double* k3, double a3,
    const double* k4, double a4,
    const double* k5, double a5,
    double dt,
    size_t size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    out[idx] = state[idx] + dt * (a1 * k1[idx] + a2 * k2[idx] + a3 * k3[idx] + 
                                   a4 * k4[idx] + a5 * k5[idx]);
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

