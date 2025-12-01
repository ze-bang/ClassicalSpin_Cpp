#ifndef LATTICE_GPU_API_H
#define LATTICE_GPU_API_H

/**
 * Host-only API header for GPU lattice operations
 * 
 * This header can be included by pure C++ code (compiled with g++/clang++)
 * while the actual GPU implementations live in lattice_gpu.cu (compiled with nvcc).
 * 
 * Architecture:
 * - lattice_gpu_api.h: Host-callable declarations (this file) - for C++ TUs
 * - lattice_gpu.cuh: Device declarations + kernel signatures - for CUDA TUs only
 * - lattice_gpu.cu: Kernel implementations
 * 
 * The gpu:: namespace provides the interface between CPU Lattice and GPU kernels.
 */

#include <vector>
#include <string>
#include <utility>
#include <cstddef>

// Forward declaration - actual GPU data lives in .cu files
namespace gpu {

/**
 * Opaque handle to GPU lattice data
 * The actual GPULatticeData struct is defined in lattice_gpu.cuh and only
 * accessible to CUDA translation units.
 */
struct GPULatticeDataHandle;

/**
 * Create GPU lattice data from host arrays
 * 
 * @param lattice_size Total number of sites
 * @param spin_dim Dimension of each spin
 * @param N_atoms Number of atoms per unit cell
 * @param max_bilinear Maximum number of bilinear neighbors per site
 * @param flat_field Flattened external field [lattice_size * spin_dim]
 * @param flat_onsite Flattened onsite matrices [lattice_size * spin_dim * spin_dim]
 * @param flat_bilinear Flattened bilinear matrices [lattice_size * max_bilinear * spin_dim * spin_dim]
 * @param flat_partners Flattened partner indices [lattice_size * max_bilinear]
 * @param num_bilinear_per_site Number of bilinear neighbors per site [lattice_size]
 * @return Opaque handle to GPU data (managed internally)
 */
GPULatticeDataHandle* create_gpu_lattice_data(
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms,
    size_t max_bilinear,
    const std::vector<double>& flat_field,
    const std::vector<double>& flat_onsite,
    const std::vector<double>& flat_bilinear,
    const std::vector<size_t>& flat_partners,
    const std::vector<size_t>& num_bilinear_per_site
);

/**
 * Destroy GPU lattice data and free GPU memory
 */
void destroy_gpu_lattice_data(GPULatticeDataHandle* handle);

/**
 * Set pulse parameters on GPU
 */
void set_gpu_pulse(
    GPULatticeDataHandle* handle,
    const std::vector<double>& flat_field_drive,
    double pulse_amp,
    double pulse_width,
    double pulse_freq,
    double t_pulse_1,
    double t_pulse_2
);

/**
 * Set initial spin state on GPU
 * 
 * @param handle GPU data handle
 * @param flat_spins Flattened spin configuration [lattice_size * spin_dim]
 */
void set_gpu_spins(
    GPULatticeDataHandle* handle,
    const std::vector<double>& flat_spins
);

/**
 * Get current spin state from GPU
 * 
 * @param handle GPU data handle
 * @param flat_spins Output: flattened spin configuration [lattice_size * spin_dim]
 */
void get_gpu_spins(
    GPULatticeDataHandle* handle,
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
void integrate_gpu(
    GPULatticeDataHandle* handle,
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
void step_gpu(
    GPULatticeDataHandle* handle,
    double t,
    double dt,
    const std::string& method = "ssprk53"
);

/**
 * Compute total energy on GPU
 */
double compute_energy_gpu(GPULatticeDataHandle* handle);

/**
 * Normalize spins on GPU
 */
void normalize_spins_gpu(GPULatticeDataHandle* handle, double spin_length);

/**
 * Compute magnetization on GPU (local and staggered)
 */
void compute_magnetization_gpu(
    GPULatticeDataHandle* handle,
    std::vector<double>& mag_local,
    std::vector<double>& mag_staggered
);

} // namespace gpu

#endif // LATTICE_GPU_API_H
