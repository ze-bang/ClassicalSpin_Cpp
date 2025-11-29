/**
 * @file mixed_lattice_gpu.cu
 * @brief GPU implementations for MixedLattice class molecular dynamics
 * 
 * This file contains all CUDA/Thrust implementations for GPU-accelerated
 * molecular dynamics simulations of mixed SU(2)/SU(3) systems.
 * 
 * Key features:
 * - Handles both SU(2) and SU(3) sublattices on GPU
 * - Supports mixed bilinear interactions between sublattices
 * - Optimized memory layout for coalesced access
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

// Forward declaration
class MixedLattice;

namespace MixedLatticeGPU {

// ============================================================================
// GPU DATA STRUCTURES
// ============================================================================

/**
 * GPU data structure for MixedLattice
 * Contains flattened arrays for both SU(2) and SU(3) sublattices
 */
struct GPUMixedLatticeData {
    // SU(2) sublattice data
    thrust::device_vector<double> d_field_SU2;
    thrust::device_vector<double> d_onsite_interaction_SU2;
    thrust::device_vector<double> d_bilinear_interaction_SU2;
    thrust::device_vector<size_t> d_bilinear_partners_SU2;
    thrust::device_vector<double> d_field_drive_SU2;
    
    // SU(3) sublattice data
    thrust::device_vector<double> d_field_SU3;
    thrust::device_vector<double> d_onsite_interaction_SU3;
    thrust::device_vector<double> d_bilinear_interaction_SU3;
    thrust::device_vector<size_t> d_bilinear_partners_SU3;
    thrust::device_vector<double> d_field_drive_SU3;
    
    // Mixed interaction data (SU(2)-SU(3) coupling)
    thrust::device_vector<double> d_mixed_bilinear_interaction;
    thrust::device_vector<size_t> d_mixed_bilinear_partners_SU2;  // Which SU(2) sites
    thrust::device_vector<size_t> d_mixed_bilinear_partners_SU3;  // Which SU(3) partners
    
    // Sublattice parameters
    size_t lattice_size_SU2;
    size_t lattice_size_SU3;
    size_t spin_dim_SU2;
    size_t spin_dim_SU3;
    size_t N_atoms_SU2;
    size_t N_atoms_SU3;
    size_t num_bi_SU2;
    size_t num_bi_SU3;
    size_t num_mixed_bi;
    
    // Pulse parameters for SU(2)
    double field_drive_amp_SU2;
    double field_drive_freq_SU2;
    double field_drive_width_SU2;
    double t_pulse_0_SU2;
    double t_pulse_1_SU2;
    
    // Pulse parameters for SU(3)
    double field_drive_amp_SU3;
    double field_drive_freq_SU3;
    double field_drive_width_SU3;
    double t_pulse_0_SU3;
    double t_pulse_1_SU3;
};

// ============================================================================
// CUDA KERNELS FOR SU(2) SUBLATTICE
// ============================================================================

/**
 * Compute local field for SU(2) sites (3-component spins)
 * Includes contributions from:
 * - External field
 * - Onsite anisotropy
 * - SU(2)-SU(2) bilinear interactions
 */
__global__ void compute_local_field_SU2_kernel(
    const double* __restrict__ spins_SU2,
    const double* __restrict__ field_SU2,
    const double* __restrict__ onsite_SU2,
    const double* __restrict__ bilinear_SU2,
    const size_t* __restrict__ partners_SU2,
    double* __restrict__ local_field_SU2,
    size_t lattice_size_SU2,
    size_t spin_dim_SU2,
    size_t num_bi_SU2
) {
    size_t site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size_SU2) return;
    
    size_t offset = site * spin_dim_SU2;
    
    // Initialize with external field
    for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
        double H_mu = -field_SU2[offset + mu];
        
        // Onsite contribution
        for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
            H_mu -= onsite_SU2[site * spin_dim_SU2 * spin_dim_SU2 + mu * spin_dim_SU2 + nu] 
                    * spins_SU2[offset + nu];
        }
        
        local_field_SU2[offset + mu] = H_mu;
    }
    
    // Bilinear contributions
    for (size_t n = 0; n < num_bi_SU2; ++n) {
        size_t partner = partners_SU2[site * num_bi_SU2 + n];
        size_t partner_offset = partner * spin_dim_SU2;
        size_t mat_offset = (site * num_bi_SU2 + n) * spin_dim_SU2 * spin_dim_SU2;
        
        for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
            double contrib = 0.0;
            for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
                contrib += bilinear_SU2[mat_offset + mu * spin_dim_SU2 + nu]
                          * spins_SU2[partner_offset + nu];
            }
            local_field_SU2[offset + mu] -= contrib;
        }
    }
}

/**
 * Compute local field for SU(3) sites (8-component spins)
 */
__global__ void compute_local_field_SU3_kernel(
    const double* __restrict__ spins_SU3,
    const double* __restrict__ field_SU3,
    const double* __restrict__ onsite_SU3,
    const double* __restrict__ bilinear_SU3,
    const size_t* __restrict__ partners_SU3,
    double* __restrict__ local_field_SU3,
    size_t lattice_size_SU3,
    size_t spin_dim_SU3,
    size_t num_bi_SU3
) {
    size_t site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size_SU3) return;
    
    size_t offset = site * spin_dim_SU3;
    
    // Initialize with external field
    for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
        double H_mu = -field_SU3[offset + mu];
        
        // Onsite contribution
        for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
            H_mu -= onsite_SU3[site * spin_dim_SU3 * spin_dim_SU3 + mu * spin_dim_SU3 + nu]
                    * spins_SU3[offset + nu];
        }
        
        local_field_SU3[offset + mu] = H_mu;
    }
    
    // Bilinear contributions
    for (size_t n = 0; n < num_bi_SU3; ++n) {
        size_t partner = partners_SU3[site * num_bi_SU3 + n];
        size_t partner_offset = partner * spin_dim_SU3;
        size_t mat_offset = (site * num_bi_SU3 + n) * spin_dim_SU3 * spin_dim_SU3;
        
        for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
            double contrib = 0.0;
            for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
                contrib += bilinear_SU3[mat_offset + mu * spin_dim_SU3 + nu]
                          * spins_SU3[partner_offset + nu];
            }
            local_field_SU3[offset + mu] -= contrib;
        }
    }
}

/**
 * Add mixed bilinear contributions to SU(2) local field
 * H_SU2[i] += sum_j J_ij * S_SU3[j]
 */
__global__ void add_mixed_bilinear_to_SU2_kernel(
    const double* __restrict__ spins_SU3,
    const double* __restrict__ mixed_bilinear,
    const size_t* __restrict__ su2_sites,
    const size_t* __restrict__ su3_partners,
    double* __restrict__ local_field_SU2,
    size_t num_interactions,
    size_t spin_dim_SU2,
    size_t spin_dim_SU3
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_interactions) return;
    
    size_t su2_site = su2_sites[idx];
    size_t su3_partner = su3_partners[idx];
    
    size_t mat_offset = idx * spin_dim_SU2 * spin_dim_SU3;
    
    for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
        double contrib = 0.0;
        for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
            contrib += mixed_bilinear[mat_offset + mu * spin_dim_SU3 + nu]
                      * spins_SU3[su3_partner * spin_dim_SU3 + nu];
        }
        atomicAddDouble(&local_field_SU2[su2_site * spin_dim_SU2 + mu], -contrib);
    }
}

/**
 * Add mixed bilinear contributions to SU(3) local field
 * H_SU3[j] += sum_i J_ij^T * S_SU2[i]
 */
__global__ void add_mixed_bilinear_to_SU3_kernel(
    const double* __restrict__ spins_SU2,
    const double* __restrict__ mixed_bilinear,
    const size_t* __restrict__ su2_sites,
    const size_t* __restrict__ su3_partners,
    double* __restrict__ local_field_SU3,
    size_t num_interactions,
    size_t spin_dim_SU2,
    size_t spin_dim_SU3
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_interactions) return;
    
    size_t su2_site = su2_sites[idx];
    size_t su3_partner = su3_partners[idx];
    
    size_t mat_offset = idx * spin_dim_SU2 * spin_dim_SU3;
    
    // Transpose contribution: J^T applied
    for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
        double contrib = 0.0;
        for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
            contrib += mixed_bilinear[mat_offset + mu * spin_dim_SU3 + nu]
                      * spins_SU2[su2_site * spin_dim_SU2 + mu];
        }
        atomicAddDouble(&local_field_SU3[su3_partner * spin_dim_SU3 + nu], -contrib);
    }
}

/**
 * Landau-Lifshitz for SU(2) spins: dS/dt = S × H
 */
__global__ void landau_lifshitz_SU2_kernel(
    const double* __restrict__ spins_SU2,
    const double* __restrict__ local_field_SU2,
    double* __restrict__ dsdt_SU2,
    size_t lattice_size_SU2
) {
    size_t site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size_SU2) return;
    
    size_t offset = site * 3;
    
    double Sx = spins_SU2[offset];
    double Sy = spins_SU2[offset + 1];
    double Sz = spins_SU2[offset + 2];
    
    double Hx = local_field_SU2[offset];
    double Hy = local_field_SU2[offset + 1];
    double Hz = local_field_SU2[offset + 2];
    
    // Cross product: S × H
    dsdt_SU2[offset]     = Sy * Hz - Sz * Hy;
    dsdt_SU2[offset + 1] = Sz * Hx - Sx * Hz;
    dsdt_SU2[offset + 2] = Sx * Hy - Sy * Hx;
}

/**
 * Add time-dependent drive field for SU(2) sublattice
 */
__global__ void add_drive_field_SU2_kernel(
    double* __restrict__ local_field_SU2,
    const double* __restrict__ field_drive_0,
    const double* __restrict__ field_drive_1,
    double t,
    double t_pulse_0,
    double t_pulse_1,
    double amp,
    double freq,
    double width,
    size_t N_atoms_SU2,
    size_t spin_dim_SU2,
    size_t lattice_size_SU2
) {
    size_t site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size_SU2) return;
    
    size_t atom = site % N_atoms_SU2;
    size_t spin_offset = site * spin_dim_SU2;
    size_t drive_offset = atom * spin_dim_SU2;
    
    double dt0 = t - t_pulse_0;
    double dt1 = t - t_pulse_1;
    double envelope0 = amp * exp(-0.5 * dt0 * dt0 / (width * width)) * cos(freq * dt0);
    double envelope1 = amp * exp(-0.5 * dt1 * dt1 / (width * width)) * cos(freq * dt1);
    
    for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
        local_field_SU2[spin_offset + mu] -= envelope0 * field_drive_0[drive_offset + mu];
        local_field_SU2[spin_offset + mu] -= envelope1 * field_drive_1[drive_offset + mu];
    }
}

/**
 * Add time-dependent drive field for SU(3) sublattice
 */
__global__ void add_drive_field_SU3_kernel(
    double* __restrict__ local_field_SU3,
    const double* __restrict__ field_drive_0,
    const double* __restrict__ field_drive_1,
    double t,
    double t_pulse_0,
    double t_pulse_1,
    double amp,
    double freq,
    double width,
    size_t N_atoms_SU3,
    size_t spin_dim_SU3,
    size_t lattice_size_SU3
) {
    size_t site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size_SU3) return;
    
    size_t atom = site % N_atoms_SU3;
    size_t spin_offset = site * spin_dim_SU3;
    size_t drive_offset = atom * spin_dim_SU3;
    
    double dt0 = t - t_pulse_0;
    double dt1 = t - t_pulse_1;
    double envelope0 = amp * exp(-0.5 * dt0 * dt0 / (width * width)) * cos(freq * dt0);
    double envelope1 = amp * exp(-0.5 * dt1 * dt1 / (width * width)) * cos(freq * dt1);
    
    for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
        local_field_SU3[spin_offset + mu] -= envelope0 * field_drive_0[drive_offset + mu];
        local_field_SU3[spin_offset + mu] -= envelope1 * field_drive_1[drive_offset + mu];
    }
}

// ============================================================================
// GPU DATA TRANSFER FUNCTIONS
// ============================================================================

/**
 * Transfer MixedLattice data to GPU
 */
GPUMixedLatticeData transfer_to_gpu(
    // SU(2) parameters
    size_t lattice_size_SU2,
    size_t spin_dim_SU2,
    size_t N_atoms_SU2,
    size_t num_bi_SU2,
    const std::vector<double>& field_SU2,
    const std::vector<double>& onsite_SU2,
    const std::vector<double>& bilinear_SU2,
    const std::vector<size_t>& partners_SU2,
    const std::vector<double>& field_drive_0_SU2,
    const std::vector<double>& field_drive_1_SU2,
    double t_pulse_0_SU2,
    double t_pulse_1_SU2,
    double amp_SU2,
    double freq_SU2,
    double width_SU2,
    
    // SU(3) parameters
    size_t lattice_size_SU3,
    size_t spin_dim_SU3,
    size_t N_atoms_SU3,
    size_t num_bi_SU3,
    const std::vector<double>& field_SU3,
    const std::vector<double>& onsite_SU3,
    const std::vector<double>& bilinear_SU3,
    const std::vector<size_t>& partners_SU3,
    const std::vector<double>& field_drive_0_SU3,
    const std::vector<double>& field_drive_1_SU3,
    double t_pulse_0_SU3,
    double t_pulse_1_SU3,
    double amp_SU3,
    double freq_SU3,
    double width_SU3,
    
    // Mixed interaction parameters
    const std::vector<double>& mixed_bilinear,
    const std::vector<size_t>& mixed_su2_sites,
    const std::vector<size_t>& mixed_su3_partners,
    size_t num_mixed_bi
) {
    GPUMixedLatticeData gpu_data;
    
    // SU(2) sublattice
    gpu_data.lattice_size_SU2 = lattice_size_SU2;
    gpu_data.spin_dim_SU2 = spin_dim_SU2;
    gpu_data.N_atoms_SU2 = N_atoms_SU2;
    gpu_data.num_bi_SU2 = num_bi_SU2;
    
    gpu_data.d_field_SU2 = thrust::device_vector<double>(field_SU2.begin(), field_SU2.end());
    gpu_data.d_onsite_interaction_SU2 = thrust::device_vector<double>(onsite_SU2.begin(), onsite_SU2.end());
    gpu_data.d_bilinear_interaction_SU2 = thrust::device_vector<double>(bilinear_SU2.begin(), bilinear_SU2.end());
    gpu_data.d_bilinear_partners_SU2 = thrust::device_vector<size_t>(partners_SU2.begin(), partners_SU2.end());
    
    std::vector<double> combined_drive_SU2;
    combined_drive_SU2.insert(combined_drive_SU2.end(), field_drive_0_SU2.begin(), field_drive_0_SU2.end());
    combined_drive_SU2.insert(combined_drive_SU2.end(), field_drive_1_SU2.begin(), field_drive_1_SU2.end());
    gpu_data.d_field_drive_SU2 = thrust::device_vector<double>(combined_drive_SU2.begin(), combined_drive_SU2.end());
    
    gpu_data.field_drive_amp_SU2 = amp_SU2;
    gpu_data.field_drive_freq_SU2 = freq_SU2;
    gpu_data.field_drive_width_SU2 = width_SU2;
    gpu_data.t_pulse_0_SU2 = t_pulse_0_SU2;
    gpu_data.t_pulse_1_SU2 = t_pulse_1_SU2;
    
    // SU(3) sublattice
    gpu_data.lattice_size_SU3 = lattice_size_SU3;
    gpu_data.spin_dim_SU3 = spin_dim_SU3;
    gpu_data.N_atoms_SU3 = N_atoms_SU3;
    gpu_data.num_bi_SU3 = num_bi_SU3;
    
    gpu_data.d_field_SU3 = thrust::device_vector<double>(field_SU3.begin(), field_SU3.end());
    gpu_data.d_onsite_interaction_SU3 = thrust::device_vector<double>(onsite_SU3.begin(), onsite_SU3.end());
    gpu_data.d_bilinear_interaction_SU3 = thrust::device_vector<double>(bilinear_SU3.begin(), bilinear_SU3.end());
    gpu_data.d_bilinear_partners_SU3 = thrust::device_vector<size_t>(partners_SU3.begin(), partners_SU3.end());
    
    std::vector<double> combined_drive_SU3;
    combined_drive_SU3.insert(combined_drive_SU3.end(), field_drive_0_SU3.begin(), field_drive_0_SU3.end());
    combined_drive_SU3.insert(combined_drive_SU3.end(), field_drive_1_SU3.begin(), field_drive_1_SU3.end());
    gpu_data.d_field_drive_SU3 = thrust::device_vector<double>(combined_drive_SU3.begin(), combined_drive_SU3.end());
    
    gpu_data.field_drive_amp_SU3 = amp_SU3;
    gpu_data.field_drive_freq_SU3 = freq_SU3;
    gpu_data.field_drive_width_SU3 = width_SU3;
    gpu_data.t_pulse_0_SU3 = t_pulse_0_SU3;
    gpu_data.t_pulse_1_SU3 = t_pulse_1_SU3;
    
    // Mixed interactions
    gpu_data.num_mixed_bi = num_mixed_bi;
    gpu_data.d_mixed_bilinear_interaction = thrust::device_vector<double>(mixed_bilinear.begin(), mixed_bilinear.end());
    gpu_data.d_mixed_bilinear_partners_SU2 = thrust::device_vector<size_t>(mixed_su2_sites.begin(), mixed_su2_sites.end());
    gpu_data.d_mixed_bilinear_partners_SU3 = thrust::device_vector<size_t>(mixed_su3_partners.begin(), mixed_su3_partners.end());
    
    return gpu_data;
}

// ============================================================================
// GPU COMPUTATION FUNCTIONS
// ============================================================================

/**
 * Compute Landau-Lifshitz equation for mixed lattice on GPU
 * State layout: [SU2 spins (lattice_size_SU2 * spin_dim_SU2), SU3 spins (lattice_size_SU3 * spin_dim_SU3)]
 */
void compute_landau_lifshitz_mixed_gpu(
    const thrust::device_vector<double>& d_state,
    thrust::device_vector<double>& d_dsdt,
    double t,
    const GPUMixedLatticeData& gpu_data
) {
    size_t lattice_size_SU2 = gpu_data.lattice_size_SU2;
    size_t lattice_size_SU3 = gpu_data.lattice_size_SU3;
    size_t spin_dim_SU2 = gpu_data.spin_dim_SU2;
    size_t spin_dim_SU3 = gpu_data.spin_dim_SU3;
    size_t SU2_size = lattice_size_SU2 * spin_dim_SU2;
    
    // Temporary storage for local fields
    thrust::device_vector<double> d_local_field_SU2(SU2_size);
    thrust::device_vector<double> d_local_field_SU3(lattice_size_SU3 * spin_dim_SU3);
    
    int block_size = 256;
    int num_blocks_SU2 = (lattice_size_SU2 + block_size - 1) / block_size;
    int num_blocks_SU3 = (lattice_size_SU3 + block_size - 1) / block_size;
    
    // Get pointers to SU2 and SU3 parts of state
    const double* spins_SU2 = thrust::raw_pointer_cast(d_state.data());
    const double* spins_SU3 = thrust::raw_pointer_cast(d_state.data()) + SU2_size;
    
    // Step 1: Compute SU(2) local fields
    compute_local_field_SU2_kernel<<<num_blocks_SU2, block_size>>>(
        spins_SU2,
        thrust::raw_pointer_cast(gpu_data.d_field_SU2.data()),
        thrust::raw_pointer_cast(gpu_data.d_onsite_interaction_SU2.data()),
        thrust::raw_pointer_cast(gpu_data.d_bilinear_interaction_SU2.data()),
        thrust::raw_pointer_cast(gpu_data.d_bilinear_partners_SU2.data()),
        thrust::raw_pointer_cast(d_local_field_SU2.data()),
        lattice_size_SU2,
        spin_dim_SU2,
        gpu_data.num_bi_SU2
    );
    
    // Step 2: Compute SU(3) local fields
    compute_local_field_SU3_kernel<<<num_blocks_SU3, block_size>>>(
        spins_SU3,
        thrust::raw_pointer_cast(gpu_data.d_field_SU3.data()),
        thrust::raw_pointer_cast(gpu_data.d_onsite_interaction_SU3.data()),
        thrust::raw_pointer_cast(gpu_data.d_bilinear_interaction_SU3.data()),
        thrust::raw_pointer_cast(gpu_data.d_bilinear_partners_SU3.data()),
        thrust::raw_pointer_cast(d_local_field_SU3.data()),
        lattice_size_SU3,
        spin_dim_SU3,
        gpu_data.num_bi_SU3
    );
    
    // Step 3: Add mixed interaction contributions
    if (gpu_data.num_mixed_bi > 0) {
        int num_blocks_mixed = (gpu_data.num_mixed_bi + block_size - 1) / block_size;
        
        add_mixed_bilinear_to_SU2_kernel<<<num_blocks_mixed, block_size>>>(
            spins_SU3,
            thrust::raw_pointer_cast(gpu_data.d_mixed_bilinear_interaction.data()),
            thrust::raw_pointer_cast(gpu_data.d_mixed_bilinear_partners_SU2.data()),
            thrust::raw_pointer_cast(gpu_data.d_mixed_bilinear_partners_SU3.data()),
            thrust::raw_pointer_cast(d_local_field_SU2.data()),
            gpu_data.num_mixed_bi,
            spin_dim_SU2,
            spin_dim_SU3
        );
        
        add_mixed_bilinear_to_SU3_kernel<<<num_blocks_mixed, block_size>>>(
            spins_SU2,
            thrust::raw_pointer_cast(gpu_data.d_mixed_bilinear_interaction.data()),
            thrust::raw_pointer_cast(gpu_data.d_mixed_bilinear_partners_SU2.data()),
            thrust::raw_pointer_cast(gpu_data.d_mixed_bilinear_partners_SU3.data()),
            thrust::raw_pointer_cast(d_local_field_SU3.data()),
            gpu_data.num_mixed_bi,
            spin_dim_SU2,
            spin_dim_SU3
        );
    }
    
    // Step 4: Add time-dependent drive fields
    if (gpu_data.field_drive_amp_SU2 > 0) {
        size_t drive_size_SU2 = gpu_data.N_atoms_SU2 * spin_dim_SU2;
        add_drive_field_SU2_kernel<<<num_blocks_SU2, block_size>>>(
            thrust::raw_pointer_cast(d_local_field_SU2.data()),
            thrust::raw_pointer_cast(gpu_data.d_field_drive_SU2.data()),
            thrust::raw_pointer_cast(gpu_data.d_field_drive_SU2.data()) + drive_size_SU2,
            t,
            gpu_data.t_pulse_0_SU2,
            gpu_data.t_pulse_1_SU2,
            gpu_data.field_drive_amp_SU2,
            gpu_data.field_drive_freq_SU2,
            gpu_data.field_drive_width_SU2,
            gpu_data.N_atoms_SU2,
            spin_dim_SU2,
            lattice_size_SU2
        );
    }
    
    if (gpu_data.field_drive_amp_SU3 > 0) {
        size_t drive_size_SU3 = gpu_data.N_atoms_SU3 * spin_dim_SU3;
        add_drive_field_SU3_kernel<<<num_blocks_SU3, block_size>>>(
            thrust::raw_pointer_cast(d_local_field_SU3.data()),
            thrust::raw_pointer_cast(gpu_data.d_field_drive_SU3.data()),
            thrust::raw_pointer_cast(gpu_data.d_field_drive_SU3.data()) + drive_size_SU3,
            t,
            gpu_data.t_pulse_0_SU3,
            gpu_data.t_pulse_1_SU3,
            gpu_data.field_drive_amp_SU3,
            gpu_data.field_drive_freq_SU3,
            gpu_data.field_drive_width_SU3,
            gpu_data.N_atoms_SU3,
            spin_dim_SU3,
            lattice_size_SU3
        );
    }
    
    // Step 5: Compute dS/dt = S × H for SU(2)
    double* dsdt_SU2 = thrust::raw_pointer_cast(d_dsdt.data());
    landau_lifshitz_SU2_kernel<<<num_blocks_SU2, block_size>>>(
        spins_SU2,
        thrust::raw_pointer_cast(d_local_field_SU2.data()),
        dsdt_SU2,
        lattice_size_SU2
    );
    
    // Step 6: Compute dS/dt = S × H for SU(3)
    // For SU(3), need to use generalized cross product on host (GPU version would need SU(3) structure constants)
    if (spin_dim_SU3 == 8) {
        // Fall back to host for SU(3) cross product
        thrust::host_vector<double> h_spins_SU3(spins_SU3, spins_SU3 + lattice_size_SU3 * spin_dim_SU3);
        thrust::host_vector<double> h_field_SU3 = d_local_field_SU3;
        thrust::host_vector<double> h_dsdt_SU3(lattice_size_SU3 * spin_dim_SU3);
        
        for (size_t site = 0; site < lattice_size_SU3; ++site) {
            SpinVector S = Eigen::Map<const SpinVector>(
                h_spins_SU3.data() + site * spin_dim_SU3, spin_dim_SU3);
            SpinVector H = Eigen::Map<const SpinVector>(
                h_field_SU3.data() + site * spin_dim_SU3, spin_dim_SU3);
            SpinVector dS = cross_product(S, H);
            for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
                h_dsdt_SU3[site * spin_dim_SU3 + mu] = dS(mu);
            }
        }
        
        thrust::copy(h_dsdt_SU3.begin(), h_dsdt_SU3.end(), 
                    d_dsdt.begin() + SU2_size);
    } else if (spin_dim_SU3 == 3) {
        // If SU(3) is actually SU(2) (3-component), use the SU(2) kernel
        double* dsdt_SU3 = thrust::raw_pointer_cast(d_dsdt.data()) + SU2_size;
        landau_lifshitz_SU2_kernel<<<num_blocks_SU3, block_size>>>(
            spins_SU3,
            thrust::raw_pointer_cast(d_local_field_SU3.data()),
            dsdt_SU3,
            lattice_size_SU3
        );
    }
    
    cudaDeviceSynchronize();
}

/**
 * Compute magnetization for both sublattices
 */
void compute_magnetizations_gpu(
    const thrust::device_vector<double>& d_state,
    const GPUMixedLatticeData& gpu_data,
    SpinVector& M_SU2,
    SpinVector& M_SU3
) {
    size_t SU2_size = gpu_data.lattice_size_SU2 * gpu_data.spin_dim_SU2;
    
    thrust::host_vector<double> h_state = d_state;
    
    // SU(2) magnetization
    M_SU2 = SpinVector::Zero(gpu_data.spin_dim_SU2);
    for (size_t i = 0; i < gpu_data.lattice_size_SU2; ++i) {
        for (size_t mu = 0; mu < gpu_data.spin_dim_SU2; ++mu) {
            M_SU2(mu) += h_state[i * gpu_data.spin_dim_SU2 + mu];
        }
    }
    M_SU2 /= static_cast<double>(gpu_data.lattice_size_SU2);
    
    // SU(3) magnetization
    M_SU3 = SpinVector::Zero(gpu_data.spin_dim_SU3);
    for (size_t i = 0; i < gpu_data.lattice_size_SU3; ++i) {
        for (size_t mu = 0; mu < gpu_data.spin_dim_SU3; ++mu) {
            M_SU3(mu) += h_state[SU2_size + i * gpu_data.spin_dim_SU3 + mu];
        }
    }
    M_SU3 /= static_cast<double>(gpu_data.lattice_size_SU3);
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
 * RK4 integrator for mixed lattice on GPU
 */
template<typename Observer>
void integrate_rk4_mixed_gpu(
    thrust::device_vector<double>& d_state,
    const GPUMixedLatticeData& gpu_data,
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
        double dt_step = dt;
        if (t + dt_step > t_end) {
            dt_step = t_end - t;
        }
        
        // k1 = f(t, y)
        compute_landau_lifshitz_mixed_gpu(d_state, k1, t, gpu_data);
        
        // k2 = f(t + dt/2, y + dt/2 * k1)
        thrust::transform(d_state.begin(), d_state.end(), k1.begin(),
                         temp.begin(),
                         RK4HalfStep(dt_step));
        compute_landau_lifshitz_mixed_gpu(temp, k2, t + 0.5 * dt_step, gpu_data);
        
        // k3 = f(t + dt/2, y + dt/2 * k2)
        thrust::transform(d_state.begin(), d_state.end(), k2.begin(),
                         temp.begin(),
                         RK4HalfStep(dt_step));
        compute_landau_lifshitz_mixed_gpu(temp, k3, t + 0.5 * dt_step, gpu_data);
        
        // k4 = f(t + dt, y + dt * k3)
        thrust::transform(d_state.begin(), d_state.end(), k3.begin(),
                         temp.begin(),
                         RK4FullStep(dt_step));
        compute_landau_lifshitz_mixed_gpu(temp, k4, t + dt_step, gpu_data);
        
        // y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(d_state.begin(), k1.begin(), k2.begin(), k3.begin(), k4.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(d_state.end(), k1.end(), k2.end(), k3.end(), k4.end())),
            d_state.begin(),
            RK4Combine(dt_step)
        );
        
        t += dt_step;
        observer(d_state, t);
    }
}

} // namespace MixedLatticeGPU

// ============================================================================
// C-STYLE INTERFACE FOR MIXED_LATTICE CLASS
// ============================================================================

extern "C" {

/**
 * Run GPU molecular dynamics for MixedLattice class
 */
void mixed_lattice_molecular_dynamics_gpu_impl(
    // State data
    double* state_data,
    size_t state_size,
    
    // SU(2) lattice parameters
    size_t lattice_size_SU2,
    size_t spin_dim_SU2,
    size_t N_atoms_SU2,
    size_t num_bi_SU2,
    const double* field_SU2,
    size_t field_size_SU2,
    const double* onsite_SU2,
    size_t onsite_size_SU2,
    const double* bilinear_SU2,
    size_t bilinear_size_SU2,
    const size_t* partners_SU2,
    size_t partners_size_SU2,
    const double* field_drive_0_SU2,
    const double* field_drive_1_SU2,
    size_t field_drive_size_SU2,
    double t_pulse_0_SU2,
    double t_pulse_1_SU2,
    double amp_SU2,
    double freq_SU2,
    double width_SU2,
    
    // SU(3) lattice parameters
    size_t lattice_size_SU3,
    size_t spin_dim_SU3,
    size_t N_atoms_SU3,
    size_t num_bi_SU3,
    const double* field_SU3,
    size_t field_size_SU3,
    const double* onsite_SU3,
    size_t onsite_size_SU3,
    const double* bilinear_SU3,
    size_t bilinear_size_SU3,
    const size_t* partners_SU3,
    size_t partners_size_SU3,
    const double* field_drive_0_SU3,
    const double* field_drive_1_SU3,
    size_t field_drive_size_SU3,
    double t_pulse_0_SU3,
    double t_pulse_1_SU3,
    double amp_SU3,
    double freq_SU3,
    double width_SU3,
    
    // Mixed interaction parameters
    const double* mixed_bilinear,
    size_t mixed_bilinear_size,
    const size_t* mixed_su2_sites,
    const size_t* mixed_su3_partners,
    size_t num_mixed_bi,
    
    // Integration parameters
    double t_start,
    double t_end,
    double dt,
    size_t save_interval,
    
    // Output callback
    void (*save_callback)(double t, const double* state, size_t size, void* user_data),
    void* user_data
) {
    using namespace MixedLatticeGPU;
    
    // Create host vectors
    std::vector<double> v_field_SU2(field_SU2, field_SU2 + field_size_SU2);
    std::vector<double> v_onsite_SU2(onsite_SU2, onsite_SU2 + onsite_size_SU2);
    std::vector<double> v_bilinear_SU2(bilinear_SU2, bilinear_SU2 + bilinear_size_SU2);
    std::vector<size_t> v_partners_SU2(partners_SU2, partners_SU2 + partners_size_SU2);
    std::vector<double> v_fd0_SU2(field_drive_0_SU2, field_drive_0_SU2 + field_drive_size_SU2);
    std::vector<double> v_fd1_SU2(field_drive_1_SU2, field_drive_1_SU2 + field_drive_size_SU2);
    
    std::vector<double> v_field_SU3(field_SU3, field_SU3 + field_size_SU3);
    std::vector<double> v_onsite_SU3(onsite_SU3, onsite_SU3 + onsite_size_SU3);
    std::vector<double> v_bilinear_SU3(bilinear_SU3, bilinear_SU3 + bilinear_size_SU3);
    std::vector<size_t> v_partners_SU3(partners_SU3, partners_SU3 + partners_size_SU3);
    std::vector<double> v_fd0_SU3(field_drive_0_SU3, field_drive_0_SU3 + field_drive_size_SU3);
    std::vector<double> v_fd1_SU3(field_drive_1_SU3, field_drive_1_SU3 + field_drive_size_SU3);
    
    std::vector<double> v_mixed_bilinear(mixed_bilinear, mixed_bilinear + mixed_bilinear_size);
    std::vector<size_t> v_mixed_su2(mixed_su2_sites, mixed_su2_sites + num_mixed_bi);
    std::vector<size_t> v_mixed_su3(mixed_su3_partners, mixed_su3_partners + num_mixed_bi);
    
    // Transfer to GPU
    GPUMixedLatticeData gpu_data = transfer_to_gpu(
        lattice_size_SU2, spin_dim_SU2, N_atoms_SU2, num_bi_SU2,
        v_field_SU2, v_onsite_SU2, v_bilinear_SU2, v_partners_SU2,
        v_fd0_SU2, v_fd1_SU2, t_pulse_0_SU2, t_pulse_1_SU2, amp_SU2, freq_SU2, width_SU2,
        
        lattice_size_SU3, spin_dim_SU3, N_atoms_SU3, num_bi_SU3,
        v_field_SU3, v_onsite_SU3, v_bilinear_SU3, v_partners_SU3,
        v_fd0_SU3, v_fd1_SU3, t_pulse_0_SU3, t_pulse_1_SU3, amp_SU3, freq_SU3, width_SU3,
        
        v_mixed_bilinear, v_mixed_su2, v_mixed_su3, num_mixed_bi
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
    integrate_rk4_mixed_gpu(d_state, gpu_data, t_start, t_end, dt, observer);
    
    // Copy final state back
    thrust::copy(d_state.begin(), d_state.end(), state_data);
}

} // extern "C"
