#ifndef MIXED_LATTICE_GPU_API_H
#define MIXED_LATTICE_GPU_API_H

/**
 * Host-only API header for mixed GPU lattice operations (SU(2) + SU(3))
 * 
 * This header can be included by pure C++ code (compiled with g++/clang++)
 * while the actual GPU implementations live in mixed_lattice_gpu.cu (compiled with nvcc).
 * 
 * Architecture:
 * - mixed_lattice_gpu_api.h: Host-callable declarations (this file) - for C++ TUs
 * - mixed_lattice_gpu.cuh: Device declarations + kernel signatures - for CUDA TUs only
 * - mixed_lattice_gpu.cu: Kernel implementations
 * 
 * The mixed_gpu:: namespace provides the interface between CPU MixedLattice and GPU kernels.
 */

#include <vector>
#include <string>
#include <utility>
#include <cstddef>
#include <array>

// Forward declaration - actual GPU data lives in .cu files
namespace mixed_gpu {

/**
 * Opaque handle to GPU mixed lattice data
 * The actual GPUMixedLatticeData struct is defined in mixed_lattice_gpu.cuh and only
 * accessible to CUDA translation units.
 */
struct GPUMixedLatticeDataHandle;

/**
 * Create GPU mixed lattice data from host arrays
 * 
 * @param lattice_size_SU2 Total number of SU(2) sites
 * @param spin_dim_SU2 Dimension of SU(2) spins (typically 3)
 * @param N_atoms_SU2 Number of SU(2) atoms per unit cell
 * @param lattice_size_SU3 Total number of SU(3) sites
 * @param spin_dim_SU3 Dimension of SU(3) spins (typically 8)
 * @param N_atoms_SU3 Number of SU(3) atoms per unit cell
 * @param max_bilinear_SU2 Maximum bilinear neighbors for SU(2)
 * @param max_bilinear_SU3 Maximum bilinear neighbors for SU(3)
 * @param max_mixed_bilinear Maximum mixed bilinear neighbors
 * @param flat_field_SU2 Flattened SU(2) external field
 * @param flat_onsite_SU2 Flattened SU(2) onsite matrices
 * @param flat_bilinear_SU2 Flattened SU(2)-SU(2) bilinear matrices
 * @param flat_partners_SU2 Flattened SU(2) bilinear partner indices
 * @param num_bilinear_per_site_SU2 Number of SU(2) bilinear neighbors per site
 * @param flat_field_SU3 Flattened SU(3) external field
 * @param flat_onsite_SU3 Flattened SU(3) onsite matrices
 * @param flat_bilinear_SU3 Flattened SU(3)-SU(3) bilinear matrices
 * @param flat_partners_SU3 Flattened SU(3) bilinear partner indices
 * @param num_bilinear_per_site_SU3 Number of SU(3) bilinear neighbors per site
 * @param flat_mixed_bilinear Flattened SU(2)-SU(3) mixed bilinear matrices
 * @param flat_mixed_partners_SU2 Mixed partner indices (SU(2) side)
 * @param flat_mixed_partners_SU3 Mixed partner indices (SU(3) side)
 * @param num_mixed_per_site_SU2 Number of mixed neighbors per SU(2) site
 * @return Opaque handle to GPU data (managed internally)
 */
GPUMixedLatticeDataHandle* create_gpu_mixed_lattice_data(
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
);

/**
 * Destroy GPU mixed lattice data and free GPU memory
 */
void destroy_gpu_mixed_lattice_data(GPUMixedLatticeDataHandle* handle);

/**
 * Set pulse parameters for SU(2) sublattice
 */
void set_gpu_pulse_SU2(
    GPUMixedLatticeDataHandle* handle,
    const std::vector<double>& flat_field_drive,
    double pulse_amp, double pulse_width, double pulse_freq,
    double t_pulse_1, double t_pulse_2
);

/**
 * Set pulse parameters for SU(3) sublattice
 */
void set_gpu_pulse_SU3(
    GPUMixedLatticeDataHandle* handle,
    const std::vector<double>& flat_field_drive,
    double pulse_amp, double pulse_width, double pulse_freq,
    double t_pulse_1, double t_pulse_2
);

/**
 * Set initial spin state on GPU (combined SU(2) + SU(3))
 * 
 * @param handle GPU data handle
 * @param flat_spins Flattened spin configuration [lattice_size_SU2 * spin_dim_SU2 + lattice_size_SU3 * spin_dim_SU3]
 */
void set_gpu_mixed_spins(
    GPUMixedLatticeDataHandle* handle,
    const std::vector<double>& flat_spins
);

/**
 * Get current spin state from GPU
 * 
 * @param handle GPU data handle
 * @param flat_spins Output: flattened spin configuration
 */
void get_gpu_mixed_spins(
    GPUMixedLatticeDataHandle* handle,
    std::vector<double>& flat_spins
);

/**
 * Perform GPU integration using specified method
 * 
 * Available methods:
 * - "euler": Explicit Euler (1st order)
 * - "rk2" or "midpoint": Modified midpoint (2nd order)
 * - "rk4": Classic Runge-Kutta (4th order)
 * - "dopri5": Dormand-Prince 5(4) - recommended
 * - "ssprk53": SSP RK 5-stage 3rd order (default, optimized for spin dynamics)
 * 
 * @param handle GPU data handle
 * @param T_start Start time
 * @param T_end End time
 * @param dt Step size
 * @param save_interval Steps between trajectory saves
 * @param trajectory Output: (time, state) pairs saved at intervals
 * @param method Integration method (default: ssprk53)
 */
void integrate_mixed_gpu(
    GPUMixedLatticeDataHandle* handle,
    double T_start,
    double T_end,
    double dt,
    size_t save_interval,
    std::vector<std::pair<double, std::vector<double>>>& trajectory,
    const std::string& method = "ssprk53"
);

/**
 * Single integration step on GPU
 * 
 * @param handle GPU data handle
 * @param t Current time
 * @param dt Step size
 * @param method Integration method (default: ssprk53)
 */
void step_mixed_gpu(
    GPUMixedLatticeDataHandle* handle,
    double t,
    double dt,
    const std::string& method = "ssprk53"
);

/**
 * Compute magnetization on GPU (for both sublattices)
 * 
 * @param handle GPU data handle
 * @param mag_SU2 Output: SU(2) local magnetization [spin_dim_SU2]
 * @param mag_staggered_SU2 Output: SU(2) staggered magnetization [spin_dim_SU2]
 * @param mag_SU3 Output: SU(3) local magnetization [spin_dim_SU3]
 * @param mag_staggered_SU3 Output: SU(3) staggered magnetization [spin_dim_SU3]
 */
void compute_magnetization_mixed_gpu(
    GPUMixedLatticeDataHandle* handle,
    std::vector<double>& mag_SU2,
    std::vector<double>& mag_staggered_SU2,
    std::vector<double>& mag_SU3,
    std::vector<double>& mag_staggered_SU3
);

/**
 * Normalize spins on GPU (both sublattices)
 */
void normalize_spins_mixed_gpu(
    GPUMixedLatticeDataHandle* handle,
    double spin_length_SU2,
    double spin_length_SU3
);

/**
 * Get lattice dimensions from handle
 */
void get_mixed_lattice_dims(
    GPUMixedLatticeDataHandle* handle,
    size_t& lattice_size_SU2, size_t& spin_dim_SU2, size_t& N_atoms_SU2,
    size_t& lattice_size_SU3, size_t& spin_dim_SU3, size_t& N_atoms_SU3
);

} // namespace mixed_gpu

#endif // MIXED_LATTICE_GPU_API_H
