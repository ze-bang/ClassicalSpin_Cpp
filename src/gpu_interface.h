/**
 * @file gpu_interface.h
 * @brief C-style interface declarations for GPU implementations
 * 
 * This header provides forward declarations for GPU functions implemented
 * in lattice_gpu.cu and mixed_lattice_gpu.cu. Include this header in
 * CPU code to call GPU implementations without requiring CUDA headers.
 * 
 * Benefits of this separation:
 * - CPU code can be compiled with regular C++ compilers
 * - GPU code is isolated in .cu files compiled by nvcc
 * - Clean interface between CPU and GPU code
 * - No __CUDACC__ guards needed in header files
 */

#ifndef GPU_INTERFACE_H
#define GPU_INTERFACE_H

#include <cstddef>  // for size_t
#include <cstdint>  // for int8_t

// ============================================================================
// GPU AVAILABILITY CHECKING
// ============================================================================

extern "C" {

/**
 * Check if CUDA GPU is available
 * @return 1 if CUDA is available, 0 otherwise
 */
int cuda_available();

/**
 * Get CUDA device information
 * @param name Buffer to store device name
 * @param name_size Size of name buffer
 * @param total_memory Pointer to store total GPU memory
 * @param compute_major Pointer to store compute capability major version
 * @param compute_minor Pointer to store compute capability minor version
 */
void get_cuda_device_info(char* name, size_t name_size, 
                          size_t* total_memory, 
                          int* compute_major, int* compute_minor);

// ============================================================================
// LATTICE GPU INTERFACE
// ============================================================================

/**
 * GPU implementation of molecular dynamics for Lattice class
 * 
 * @param state_data Initial state (modified in place to final state)
 * @param state_size Size of state array
 * @param lattice_size Number of lattice sites
 * @param spin_dim Dimension of spin vectors
 * @param N_atoms Atoms per unit cell
 * @param num_bi Number of bilinear neighbors per site
 * @param num_tri Number of trilinear interactions per site
 * @param field_data Flattened external field vectors
 * @param field_size Size of field array
 * @param onsite_data Flattened onsite interaction matrices
 * @param onsite_size Size of onsite array
 * @param bilinear_data Flattened bilinear interaction matrices
 * @param bilinear_size Size of bilinear array
 * @param partners_data Bilinear partner indices
 * @param partners_size Size of partners array
 * @param wrap_dir_data Wrap direction for twisted boundaries
 * @param wrap_dir_size Size of wrap_dir array
 * @param field_drive_0 First pulse direction
 * @param field_drive_1 Second pulse direction
 * @param field_drive_size Size of field_drive arrays
 * @param t_pulse_0 Time of first pulse center
 * @param t_pulse_1 Time of second pulse center
 * @param amp Pulse amplitude
 * @param freq Pulse frequency
 * @param width Pulse width (Gaussian sigma)
 * @param t_start Integration start time
 * @param t_end Integration end time
 * @param dt Time step
 * @param save_interval Steps between saves
 * @param save_callback Callback for saving data (can be NULL)
 * @param user_data User data passed to callback
 */
void lattice_molecular_dynamics_gpu_impl(
    double* state_data,
    size_t state_size,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms,
    size_t num_bi,
    size_t num_tri,
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
    const double* field_drive_0,
    const double* field_drive_1,
    size_t field_drive_size,
    double t_pulse_0,
    double t_pulse_1,
    double amp,
    double freq,
    double width,
    double t_start,
    double t_end,
    double dt,
    size_t save_interval,
    void (*save_callback)(double t, const double* state, size_t size, void* user_data),
    void* user_data
);

// ============================================================================
// MIXED LATTICE GPU INTERFACE
// ============================================================================

/**
 * GPU implementation of molecular dynamics for MixedLattice class
 * Handles coupled SU(2) and SU(3) sublattices
 * 
 * State layout: [SU2 spins][SU3 spins]
 * 
 * @param state_data Initial state (modified in place)
 * @param state_size Total state size
 * 
 * SU(2) sublattice parameters:
 * @param lattice_size_SU2 Number of SU(2) sites
 * @param spin_dim_SU2 Dimension of SU(2) spins (typically 3)
 * @param N_atoms_SU2 SU(2) atoms per unit cell
 * @param num_bi_SU2 Bilinear neighbors per SU(2) site
 * @param field_SU2 SU(2) external field
 * @param field_size_SU2 Size of field_SU2
 * @param onsite_SU2 SU(2) onsite interactions
 * @param onsite_size_SU2 Size of onsite_SU2
 * @param bilinear_SU2 SU(2)-SU(2) bilinear interactions
 * @param bilinear_size_SU2 Size of bilinear_SU2
 * @param partners_SU2 SU(2) bilinear partners
 * @param partners_size_SU2 Size of partners_SU2
 * @param field_drive_0_SU2 First pulse for SU(2)
 * @param field_drive_1_SU2 Second pulse for SU(2)
 * @param field_drive_size_SU2 Size of field drive arrays
 * @param t_pulse_0_SU2 First pulse time for SU(2)
 * @param t_pulse_1_SU2 Second pulse time for SU(2)
 * @param amp_SU2, freq_SU2, width_SU2 Pulse parameters for SU(2)
 * 
 * SU(3) sublattice parameters (analogous to SU(2)):
 * @param lattice_size_SU3, spin_dim_SU3, N_atoms_SU3, num_bi_SU3
 * @param field_SU3, field_size_SU3
 * @param onsite_SU3, onsite_size_SU3
 * @param bilinear_SU3, bilinear_size_SU3
 * @param partners_SU3, partners_size_SU3
 * @param field_drive_0_SU3, field_drive_1_SU3, field_drive_size_SU3
 * @param t_pulse_0_SU3, t_pulse_1_SU3
 * @param amp_SU3, freq_SU3, width_SU3
 * 
 * Mixed interaction parameters:
 * @param mixed_bilinear Mixed SU(2)-SU(3) interaction matrices
 * @param mixed_bilinear_size Size of mixed_bilinear
 * @param mixed_su2_sites Which SU(2) sites have mixed interactions
 * @param mixed_su3_partners Which SU(3) sites are partners
 * @param num_mixed_bi Number of mixed bilinear interactions
 * 
 * Integration parameters:
 * @param t_start, t_end, dt Integration time range and step
 * @param save_interval Steps between saves
 * @param save_callback Callback for saving data
 * @param user_data User data for callback
 */
void mixed_lattice_molecular_dynamics_gpu_impl(
    double* state_data,
    size_t state_size,
    
    // SU(2) parameters
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
    
    // SU(3) parameters
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
    
    // Mixed interactions
    const double* mixed_bilinear,
    size_t mixed_bilinear_size,
    const size_t* mixed_su2_sites,
    const size_t* mixed_su3_partners,
    size_t num_mixed_bi,
    
    // Integration
    double t_start,
    double t_end,
    double dt,
    size_t save_interval,
    void (*save_callback)(double t, const double* state, size_t size, void* user_data),
    void* user_data
);

} // extern "C"

// ============================================================================
// C++ WRAPPER HELPERS
// ============================================================================

#include <string>
#include <iostream>

namespace gpu {

/**
 * Check if GPU is available and print device info
 */
inline bool check_and_print_gpu_info() {
    if (!cuda_available()) {
        std::cerr << "Warning: No CUDA-capable GPU available. GPU methods will fall back to CPU." << std::endl;
        return false;
    }
    
    char name[256];
    size_t total_memory;
    int compute_major, compute_minor;
    get_cuda_device_info(name, sizeof(name), &total_memory, &compute_major, &compute_minor);
    
    std::cout << "GPU Device: " << name << std::endl;
    std::cout << "  Total Memory: " << (total_memory / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Compute Capability: " << compute_major << "." << compute_minor << std::endl;
    
    return true;
}

/**
 * Helper to check GPU availability without printing
 */
inline bool is_gpu_available() {
    return cuda_available() != 0;
}

} // namespace gpu

#endif // GPU_INTERFACE_H
