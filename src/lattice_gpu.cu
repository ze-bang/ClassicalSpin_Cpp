/**
 * @file lattice_gpu.cu
 * @brief GPU implementations for Lattice class molecular dynamics
 * 
 * This file contains all CUDA/Thrust implementations for GPU-accelerated
 * molecular dynamics simulations. Separating GPU code into .cu files provides:
 * - Cleaner compilation (nvcc handles .cu, g++ handles .cpp/.h)
 * - Better maintainability and organization
 * - Easier debugging of GPU-specific code
 * - No need for __CUDACC__ guards in headers
 */

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <vector>
#include <array>
#include <string>
#include <memory>
#include <iostream>
#include <cmath>

#include "simple_linear_alg.h"

#ifdef HDF5_ENABLED
#include "hdf5_io.h"
#endif

// Custom atomicAdd for double on older CUDA architectures (sm < 60)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ inline double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ inline double atomicAddDouble(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

// Forward declaration of Lattice class
class Lattice;

namespace LatticeGPU {

// ============================================================================
// GPU DATA STRUCTURES
// ============================================================================

/**
 * GPU data structure for Lattice
 * Contains flattened arrays suitable for GPU memory
 */
struct GPULatticeData {
    thrust::device_vector<double> d_field;
    thrust::device_vector<double> d_onsite_interaction;
    thrust::device_vector<double> d_bilinear_interaction;
    thrust::device_vector<size_t> d_bilinear_partners;
    thrust::device_vector<int8_t> d_bilinear_wrap_dir;
    thrust::device_vector<double> d_trilinear_interaction;
    thrust::device_vector<size_t> d_trilinear_partners;
    thrust::device_vector<double> d_field_drive;
    thrust::device_vector<double> d_twist_matrices;
    
    // Lattice parameters
    size_t lattice_size;
    size_t spin_dim;
    size_t N_atoms;
    size_t num_bi;
    size_t num_tri;
    
    // Pulse parameters
    double field_drive_amp;
    double field_drive_freq;
    double field_drive_width;
    double t_pulse_0;
    double t_pulse_1;
};

// ============================================================================
// CUDA KERNELS
// ============================================================================

/**
 * CUDA kernel for computing local field contribution from bilinear interactions
 */
__global__ void compute_bilinear_field_kernel(
    const double* __restrict__ spins,
    const double* __restrict__ field,
    const double* __restrict__ onsite_interaction,
    const double* __restrict__ bilinear_interaction,
    const size_t* __restrict__ bilinear_partners,
    double* __restrict__ local_field,
    size_t lattice_size,
    size_t spin_dim,
    size_t num_bi
) {
    size_t site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    size_t spin_offset = site * spin_dim;
    
    // Initialize local field with external field and onsite contribution
    for (size_t mu = 0; mu < spin_dim; ++mu) {
        double H_mu = -field[spin_offset + mu];
        
        // Add onsite anisotropy contribution: H_mu -= sum_nu A_mu,nu * S_nu
        for (size_t nu = 0; nu < spin_dim; ++nu) {
            H_mu -= onsite_interaction[site * spin_dim * spin_dim + mu * spin_dim + nu] 
                    * spins[spin_offset + nu];
        }
        
        local_field[spin_offset + mu] = H_mu;
    }
    
    // Add bilinear interaction contributions
    for (size_t n = 0; n < num_bi; ++n) {
        size_t partner = bilinear_partners[site * num_bi + n];
        size_t partner_offset = partner * spin_dim;
        size_t mat_offset = (site * num_bi + n) * spin_dim * spin_dim;
        
        for (size_t mu = 0; mu < spin_dim; ++mu) {
            double contrib = 0.0;
            for (size_t nu = 0; nu < spin_dim; ++nu) {
                contrib += bilinear_interaction[mat_offset + mu * spin_dim + nu] 
                          * spins[partner_offset + nu];
            }
            local_field[spin_offset + mu] -= contrib;
        }
    }
}

/**
 * CUDA kernel for Landau-Lifshitz equation: dS/dt = S × H
 * For SU(2) (spin_dim=3), uses standard cross product
 */
__global__ void landau_lifshitz_kernel_su2(
    const double* __restrict__ spins,
    const double* __restrict__ local_field,
    double* __restrict__ dsdt,
    size_t lattice_size
) {
    size_t site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    size_t offset = site * 3;
    
    double Sx = spins[offset];
    double Sy = spins[offset + 1];
    double Sz = spins[offset + 2];
    
    double Hx = local_field[offset];
    double Hy = local_field[offset + 1];
    double Hz = local_field[offset + 2];
    
    // Cross product: S × H
    dsdt[offset]     = Sy * Hz - Sz * Hy;
    dsdt[offset + 1] = Sz * Hx - Sx * Hz;
    dsdt[offset + 2] = Sx * Hy - Sy * Hx;
}

/**
 * CUDA kernel for computing magnetization components
 */
__global__ void compute_magnetization_kernel(
    const double* __restrict__ spins,
    double* __restrict__ magnetization,
    size_t lattice_size,
    size_t spin_dim
) {
    extern __shared__ double sdata[];
    
    size_t tid = threadIdx.x;
    size_t site = blockIdx.x * blockDim.x + threadIdx.x;
    size_t component = blockIdx.y;
    
    // Load data into shared memory
    if (site < lattice_size) {
        sdata[tid] = spins[site * spin_dim + component];
    } else {
        sdata[tid] = 0.0;
    }
    __syncthreads();
    
    // Parallel reduction
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && site + s < lattice_size) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicAddDouble(&magnetization[component], sdata[0]);
    }
}

/**
 * CUDA kernel for adding time-dependent drive field
 */
__global__ void add_drive_field_kernel(
    double* __restrict__ local_field,
    const double* __restrict__ field_drive_0,
    const double* __restrict__ field_drive_1,
    double t,
    double t_pulse_0,
    double t_pulse_1,
    double amp,
    double freq,
    double width,
    size_t N_atoms,
    size_t spin_dim,
    size_t lattice_size
) {
    size_t site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    size_t atom = site % N_atoms;
    size_t spin_offset = site * spin_dim;
    size_t drive_offset = atom * spin_dim;
    
    // Compute pulse envelope for both pulses
    double dt0 = t - t_pulse_0;
    double dt1 = t - t_pulse_1;
    double envelope0 = amp * exp(-0.5 * dt0 * dt0 / (width * width)) * cos(freq * dt0);
    double envelope1 = amp * exp(-0.5 * dt1 * dt1 / (width * width)) * cos(freq * dt1);
    
    // Add drive field contribution
    for (size_t mu = 0; mu < spin_dim; ++mu) {
        local_field[spin_offset + mu] -= envelope0 * field_drive_0[drive_offset + mu];
        local_field[spin_offset + mu] -= envelope1 * field_drive_1[drive_offset + mu];
    }
}

// ============================================================================
// GPU DATA TRANSFER FUNCTIONS
// ============================================================================

/**
 * Transfer Lattice data to GPU
 * 
 * @param lattice_size Total number of spin sites
 * @param spin_dim Dimension of each spin vector
 * @param N_atoms Number of atoms per unit cell
 * @param num_bi Number of bilinear neighbors per site
 * @param num_tri Number of trilinear interactions per site
 * @param field External field vectors (flattened)
 * @param onsite Onsite interaction matrices (flattened)
 * @param bilinear Bilinear interaction matrices (flattened)
 * @param partners Bilinear partner indices (flattened)
 * @param wrap_dir Wrap direction for twisted boundaries (flattened)
 * @param field_drive_0 First pulse direction (per atom)
 * @param field_drive_1 Second pulse direction (per atom)
 * @param t_pulse Times for pulse centers
 * @param amp, freq, width Pulse parameters
 */
GPULatticeData transfer_to_gpu(
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms,
    size_t num_bi,
    size_t num_tri,
    const std::vector<double>& field,
    const std::vector<double>& onsite,
    const std::vector<double>& bilinear,
    const std::vector<size_t>& partners,
    const std::vector<int8_t>& wrap_dir,
    const std::vector<double>& field_drive_0,
    const std::vector<double>& field_drive_1,
    double t_pulse_0,
    double t_pulse_1,
    double amp,
    double freq,
    double width
) {
    GPULatticeData gpu_data;
    
    gpu_data.lattice_size = lattice_size;
    gpu_data.spin_dim = spin_dim;
    gpu_data.N_atoms = N_atoms;
    gpu_data.num_bi = num_bi;
    gpu_data.num_tri = num_tri;
    
    gpu_data.d_field = thrust::device_vector<double>(field.begin(), field.end());
    gpu_data.d_onsite_interaction = thrust::device_vector<double>(onsite.begin(), onsite.end());
    gpu_data.d_bilinear_interaction = thrust::device_vector<double>(bilinear.begin(), bilinear.end());
    gpu_data.d_bilinear_partners = thrust::device_vector<size_t>(partners.begin(), partners.end());
    gpu_data.d_bilinear_wrap_dir = thrust::device_vector<int8_t>(wrap_dir.begin(), wrap_dir.end());
    
    // Combine field drives into single vector for transfer
    std::vector<double> combined_drive;
    combined_drive.insert(combined_drive.end(), field_drive_0.begin(), field_drive_0.end());
    combined_drive.insert(combined_drive.end(), field_drive_1.begin(), field_drive_1.end());
    gpu_data.d_field_drive = thrust::device_vector<double>(combined_drive.begin(), combined_drive.end());
    
    gpu_data.field_drive_amp = amp;
    gpu_data.field_drive_freq = freq;
    gpu_data.field_drive_width = width;
    gpu_data.t_pulse_0 = t_pulse_0;
    gpu_data.t_pulse_1 = t_pulse_1;
    
    return gpu_data;
}

// ============================================================================
// GPU COMPUTATION FUNCTIONS
// ============================================================================

/**
 * Compute Landau-Lifshitz equation on GPU
 * 
 * @param d_state Current spin state (device)
 * @param d_dsdt Output: time derivatives (device)
 * @param t Current time
 * @param gpu_data GPU lattice data
 */
void compute_landau_lifshitz_gpu(
    const thrust::device_vector<double>& d_state,
    thrust::device_vector<double>& d_dsdt,
    double t,
    const GPULatticeData& gpu_data
) {
    size_t lattice_size = gpu_data.lattice_size;
    size_t spin_dim = gpu_data.spin_dim;
    
    // Temporary storage for local field
    thrust::device_vector<double> d_local_field(lattice_size * spin_dim);
    
    // Compute local field from interactions
    int block_size = 256;
    int num_blocks = (lattice_size + block_size - 1) / block_size;
    
    compute_bilinear_field_kernel<<<num_blocks, block_size>>>(
        thrust::raw_pointer_cast(d_state.data()),
        thrust::raw_pointer_cast(gpu_data.d_field.data()),
        thrust::raw_pointer_cast(gpu_data.d_onsite_interaction.data()),
        thrust::raw_pointer_cast(gpu_data.d_bilinear_interaction.data()),
        thrust::raw_pointer_cast(gpu_data.d_bilinear_partners.data()),
        thrust::raw_pointer_cast(d_local_field.data()),
        lattice_size,
        spin_dim,
        gpu_data.num_bi
    );
    
    // Add time-dependent drive field if applicable
    if (gpu_data.field_drive_amp > 0) {
        size_t drive_size = gpu_data.N_atoms * spin_dim;
        add_drive_field_kernel<<<num_blocks, block_size>>>(
            thrust::raw_pointer_cast(d_local_field.data()),
            thrust::raw_pointer_cast(gpu_data.d_field_drive.data()),
            thrust::raw_pointer_cast(gpu_data.d_field_drive.data()) + drive_size,
            t,
            gpu_data.t_pulse_0,
            gpu_data.t_pulse_1,
            gpu_data.field_drive_amp,
            gpu_data.field_drive_freq,
            gpu_data.field_drive_width,
            gpu_data.N_atoms,
            spin_dim,
            lattice_size
        );
    }
    
    // Compute dS/dt = S × H
    if (spin_dim == 3) {
        landau_lifshitz_kernel_su2<<<num_blocks, block_size>>>(
            thrust::raw_pointer_cast(d_state.data()),
            thrust::raw_pointer_cast(d_local_field.data()),
            thrust::raw_pointer_cast(d_dsdt.data()),
            lattice_size
        );
    } else {
        // For higher spin dimensions, fall back to host computation
        // This would require implementing SU(N) cross product kernels
        thrust::host_vector<double> h_state = d_state;
        thrust::host_vector<double> h_field = d_local_field;
        thrust::host_vector<double> h_dsdt(d_dsdt.size());
        
        // SU(N) cross product on host (temporary fallback)
        for (size_t site = 0; site < lattice_size; ++site) {
            SpinVector S = Eigen::Map<const SpinVector>(
                h_state.data() + site * spin_dim, spin_dim);
            SpinVector H = Eigen::Map<const SpinVector>(
                h_field.data() + site * spin_dim, spin_dim);
            SpinVector dS = cross_product(S, H);
            for (size_t mu = 0; mu < spin_dim; ++mu) {
                h_dsdt[site * spin_dim + mu] = dS(mu);
            }
        }
        
        d_dsdt = h_dsdt;
    }
    
    cudaDeviceSynchronize();
}

/**
 * Compute magnetization on GPU
 * 
 * @param d_state Current spin state
 * @param gpu_data GPU lattice data
 * @return Magnetization vector (on host)
 */
SpinVector compute_magnetization_gpu(
    const thrust::device_vector<double>& d_state,
    const GPULatticeData& gpu_data
) {
    size_t lattice_size = gpu_data.lattice_size;
    size_t spin_dim = gpu_data.spin_dim;
    
    thrust::device_vector<double> d_magnetization(spin_dim, 0.0);
    
    int block_size = 256;
    int num_blocks = (lattice_size + block_size - 1) / block_size;
    size_t shared_mem_size = block_size * sizeof(double);
    
    // Launch kernel for each component
    dim3 grid(num_blocks, spin_dim);
    compute_magnetization_kernel<<<grid, block_size, shared_mem_size>>>(
        thrust::raw_pointer_cast(d_state.data()),
        thrust::raw_pointer_cast(d_magnetization.data()),
        lattice_size,
        spin_dim
    );
    
    cudaDeviceSynchronize();
    
    thrust::host_vector<double> h_magnetization = d_magnetization;
    SpinVector M(spin_dim);
    for (size_t mu = 0; mu < spin_dim; ++mu) {
        M(mu) = h_magnetization[mu] / static_cast<double>(lattice_size);
    }
    
    return M;
}

/**
 * Compute total energy on GPU
 * 
 * @param d_state Current spin state
 * @param gpu_data GPU lattice data
 * @return Total energy
 */
double compute_energy_gpu(
    const thrust::device_vector<double>& d_state,
    const GPULatticeData& gpu_data
) {
    // For now, fall back to CPU for energy computation
    // Full GPU implementation would require additional kernels
    thrust::host_vector<double> h_state = d_state;
    
    size_t lattice_size = gpu_data.lattice_size;
    size_t spin_dim = gpu_data.spin_dim;
    
    thrust::host_vector<double> h_field = gpu_data.d_field;
    thrust::host_vector<double> h_onsite = gpu_data.d_onsite_interaction;
    thrust::host_vector<double> h_bilinear = gpu_data.d_bilinear_interaction;
    thrust::host_vector<size_t> h_partners = gpu_data.d_bilinear_partners;
    
    double E = 0.0;
    
    for (size_t site = 0; site < lattice_size; ++site) {
        size_t spin_offset = site * spin_dim;
        
        // Field energy
        for (size_t mu = 0; mu < spin_dim; ++mu) {
            E -= h_field[spin_offset + mu] * h_state[spin_offset + mu];
        }
        
        // Onsite energy
        for (size_t mu = 0; mu < spin_dim; ++mu) {
            for (size_t nu = 0; nu < spin_dim; ++nu) {
                E += 0.5 * h_onsite[site * spin_dim * spin_dim + mu * spin_dim + nu]
                     * h_state[spin_offset + mu] * h_state[spin_offset + nu];
            }
        }
        
        // Bilinear energy (count each bond once)
        for (size_t n = 0; n < gpu_data.num_bi; ++n) {
            size_t partner = h_partners[site * gpu_data.num_bi + n];
            if (partner > site) {  // Avoid double counting
                size_t mat_offset = (site * gpu_data.num_bi + n) * spin_dim * spin_dim;
                for (size_t mu = 0; mu < spin_dim; ++mu) {
                    for (size_t nu = 0; nu < spin_dim; ++nu) {
                        E += h_bilinear[mat_offset + mu * spin_dim + nu]
                             * h_state[spin_offset + mu] 
                             * h_state[partner * spin_dim + nu];
                    }
                }
            }
        }
    }
    
    return E;
}

// ============================================================================
// DEVICE FUNCTORS FOR RK4 INTEGRATION
// ============================================================================

/**
 * Functor for RK4 half-step: y + 0.5 * dt * k
 */
struct RK4HalfStep {
    double dt;
    __host__ __device__ RK4HalfStep(double dt_) : dt(dt_) {}
    __host__ __device__ double operator()(double y, double k) const {
        return y + 0.5 * dt * k;
    }
};

/**
 * Functor for RK4 full-step: y + dt * k
 */
struct RK4FullStep {
    double dt;
    __host__ __device__ RK4FullStep(double dt_) : dt(dt_) {}
    __host__ __device__ double operator()(double y, double k) const {
        return y + dt * k;
    }
};

/**
 * Functor for RK4 combine: y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
 */
struct RK4Combine {
    double dt;
    __host__ __device__ RK4Combine(double dt_) : dt(dt_) {}
    __host__ __device__ double operator()(
        thrust::tuple<double, double, double, double, double> t) const {
        double y = thrust::get<0>(t);
        double k1_val = thrust::get<1>(t);
        double k2_val = thrust::get<2>(t);
        double k3_val = thrust::get<3>(t);
        double k4_val = thrust::get<4>(t);
        return y + dt / 6.0 * (k1_val + 2.0 * k2_val + 2.0 * k3_val + k4_val);
    }
};

// ============================================================================
// INTEGRATION WRAPPERS
// ============================================================================

/**
 * Simple RK4 integrator on GPU
 * 
 * @param d_state State vector (modified in place)
 * @param gpu_data GPU lattice data
 * @param t_start Start time
 * @param t_end End time
 * @param dt Time step
 * @param observer Callback for each step
 */
template<typename Observer>
void integrate_rk4_gpu(
    thrust::device_vector<double>& d_state,
    const GPULatticeData& gpu_data,
    double t_start,
    double t_end,
    double dt,
    Observer observer
) {
    size_t state_size = d_state.size();
    
    thrust::device_vector<double> k1(state_size);
    thrust::device_vector<double> k2(state_size);
    thrust::device_vector<double> k3(state_size);
    thrust::device_vector<double> k4(state_size);
    thrust::device_vector<double> temp(state_size);
    
    double t = t_start;
    while (t < t_end) {
        // Adjust dt for last step
        double dt_step = dt;
        if (t + dt_step > t_end) {
            dt_step = t_end - t;
        }
        
        // k1 = f(t, y)
        compute_landau_lifshitz_gpu(d_state, k1, t, gpu_data);
        
        // k2 = f(t + dt/2, y + dt/2 * k1)
        thrust::transform(d_state.begin(), d_state.end(), k1.begin(), 
                         temp.begin(), 
                         RK4HalfStep(dt_step));
        compute_landau_lifshitz_gpu(temp, k2, t + 0.5 * dt_step, gpu_data);
        
        // k3 = f(t + dt/2, y + dt/2 * k2)
        thrust::transform(d_state.begin(), d_state.end(), k2.begin(),
                         temp.begin(),
                         RK4HalfStep(dt_step));
        compute_landau_lifshitz_gpu(temp, k3, t + 0.5 * dt_step, gpu_data);
        
        // k4 = f(t + dt, y + dt * k3)
        thrust::transform(d_state.begin(), d_state.end(), k3.begin(),
                         temp.begin(),
                         RK4FullStep(dt_step));
        compute_landau_lifshitz_gpu(temp, k4, t + dt_step, gpu_data);
        
        // y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(d_state.begin(), k1.begin(), k2.begin(), k3.begin(), k4.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(d_state.end(), k1.end(), k2.end(), k3.end(), k4.end())),
            d_state.begin(),
            RK4Combine(dt_step)
        );
        
        t += dt_step;
        
        // Call observer
        observer(d_state, t);
    }
}

} // namespace LatticeGPU

// ============================================================================
// C-STYLE INTERFACE FOR LATTICE CLASS
// ============================================================================

extern "C" {

/**
 * Run GPU molecular dynamics for Lattice class
 * This function is called from the Lattice class when use_gpu=true
 */
void lattice_molecular_dynamics_gpu_impl(
    // State and lattice parameters
    double* state_data,
    size_t state_size,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms,
    size_t num_bi,
    size_t num_tri,
    
    // Flattened interaction data
    const double* field_data,
    size_t field_size,
    const double* onsite_data,
    size_t onsite_size,
    const double* bilinear_data,
    size_t bilinear_size,
    const size_t* partners_data,
    size_t partners_size,
    const int8_t* wrap_dir_data,
    size_t wrap_dir_size,
    
    // Field drive parameters
    const double* field_drive_0,
    const double* field_drive_1,
    size_t field_drive_size,
    double t_pulse_0,
    double t_pulse_1,
    double amp,
    double freq,
    double width,
    
    // Integration parameters
    double t_start,
    double t_end,
    double dt,
    size_t save_interval,
    
    // Output callback (for HDF5 writing)
    void (*save_callback)(double t, const double* state, size_t size, void* user_data),
    void* user_data
) {
    using namespace LatticeGPU;
    
    // Create host vectors from raw arrays
    std::vector<double> field(field_data, field_data + field_size);
    std::vector<double> onsite(onsite_data, onsite_data + onsite_size);
    std::vector<double> bilinear(bilinear_data, bilinear_data + bilinear_size);
    std::vector<size_t> partners(partners_data, partners_data + partners_size);
    std::vector<int8_t> wrap_dir(wrap_dir_data, wrap_dir_data + wrap_dir_size);
    std::vector<double> fd0(field_drive_0, field_drive_0 + field_drive_size);
    std::vector<double> fd1(field_drive_1, field_drive_1 + field_drive_size);
    
    // Transfer to GPU
    GPULatticeData gpu_data = transfer_to_gpu(
        lattice_size, spin_dim, N_atoms, num_bi, num_tri,
        field, onsite, bilinear, partners, wrap_dir,
        fd0, fd1, t_pulse_0, t_pulse_1, amp, freq, width
    );
    
    // Transfer state to GPU
    thrust::device_vector<double> d_state(state_data, state_data + state_size);
    
    // Integration with observer
    size_t step_count = 0;
    thrust::host_vector<double> h_state_out;
    
    auto observer = [&](const thrust::device_vector<double>& d_x, double t) {
        if (step_count % save_interval == 0 && save_callback != nullptr) {
            h_state_out = d_x;
            save_callback(t, h_state_out.data(), h_state_out.size(), user_data);
        }
        ++step_count;
    };
    
    // Run integration
    integrate_rk4_gpu(d_state, gpu_data, t_start, t_end, dt, observer);
    
    // Copy final state back
    thrust::copy(d_state.begin(), d_state.end(), state_data);
}

/**
 * Check if CUDA is available
 */
int cuda_available() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0) ? 1 : 0;
}

/**
 * Get CUDA device properties
 */
void get_cuda_device_info(char* name, size_t name_size, size_t* total_memory, int* compute_major, int* compute_minor) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    strncpy(name, prop.name, name_size - 1);
    name[name_size - 1] = '\0';
    *total_memory = prop.totalGlobalMem;
    *compute_major = prop.major;
    *compute_minor = prop.minor;
}

} // extern "C"
