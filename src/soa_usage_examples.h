#ifndef SOA_USAGE_EXAMPLES_H
#define SOA_USAGE_EXAMPLES_H

#include "molecular_dynamics.cuh"

// Example usage of SoA optimization for classical spin simulations
// This file demonstrates how to use the Structure-of-Arrays optimization

// Example 1: Basic SoA enablement
template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
void example_enable_soa_optimization(mixed_lattice_cuda<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3, dim1, dim2, dim>& lattice) {
    std::cout << "=== SoA Optimization Example ===" << std::endl;
    
    // Enable SoA optimization
    lattice.enable_soa_optimization(true);
    
    // The lattice now uses SoA layout for optimal memory access
    // Kernels can use either legacy AoS or new SoA access patterns
    
    std::cout << "SoA optimization enabled. Ready for optimized kernels." << std::endl;
}

// Example 2: Performance comparison between AoS and SoA access patterns
template<size_t N_SU, size_t lattice_size>
__global__
void compare_access_patterns_kernel(
    // AoS layout
    const double* aos_spins,
    // SoA layout
    const double* const* soa_spins,
    double* aos_result,
    double* soa_result,
    size_t num_iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lattice_size) return;
    
    double aos_sum = 0.0, soa_sum = 0.0;
    
    // AoS access pattern (strided)
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        for (size_t comp = 0; comp < N_SU; ++comp) {
            aos_sum += aos_spins[idx * N_SU + comp];  // Stride = N_SU
        }
    }
    
    // SoA access pattern (coalesced)
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        for (size_t comp = 0; comp < N_SU; ++comp) {
            soa_sum += soa_spins[comp][idx];  // Stride = 1 (optimal)
        }
    }
    
    aos_result[idx] = aos_sum;
    soa_result[idx] = soa_sum;
}

// Example 3: SoA-optimized kernel for computing spin magnitudes
template<size_t N_SU, size_t lattice_size>
__global__
void compute_spin_magnitudes_soa(
    const soa_spin_arrays<N_SU, lattice_size> spins_soa,
    double* magnitudes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lattice_size) return;
    
    double magnitude_squared = 0.0;
    
    // Optimal memory access pattern
    for (size_t comp = 0; comp < N_SU; ++comp) {
        double component = spins_soa.component(comp, idx);
        magnitude_squared += component * component;
    }
    
    magnitudes[idx] = sqrt(magnitude_squared);
}

// Example 4: Component-wise operations (SoA advantage)
template<size_t N_SU, size_t lattice_size>
__global__
void normalize_component_soa(
    soa_spin_arrays<N_SU, lattice_size> spins_soa,
    size_t component_index,
    double normalization_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lattice_size) return;
    
    // Direct access to component array - highly efficient
    spins_soa.components[component_index][idx] *= normalization_factor;
}

// Example 5: Migration from AoS to SoA kernel
template<size_t N_SU2, size_t lattice_size_SU2>
__device__
void migrate_kernel_to_soa(
    // Old AoS version
    /*
    double* spin_updates,
    const double* aos_spins,
    int site_index
    ) {
        for (size_t i = 0; i < N_SU2; ++i) {
            spin_updates[site_index * N_SU2 + i] = aos_spins[site_index * N_SU2 + i] * 2.0;
        }
    }
    */
    
    // New SoA version
    soa_spin_arrays<N_SU2, lattice_size_SU2>& spin_updates_soa,
    const soa_spin_arrays<N_SU2, lattice_size_SU2>& spins_soa,
    int site_index
) {
    // More efficient - each component access is coalesced
    for (size_t i = 0; i < N_SU2; ++i) {
        spin_updates_soa.components[i][site_index] = spins_soa.component(i, site_index) * 2.0;
    }
}

// Example 6: Compile-time optimization selection
#ifdef USE_SOA_OPTIMIZATION
template<size_t N_SU2, size_t lattice_size_SU2>
__device__
void optimized_spin_operation(
    soa_spin_arrays<N_SU2, lattice_size_SU2>& spins,
    int site_index,
    double factor
) {
    // SoA optimized version
    for (size_t comp = 0; comp < N_SU2; ++comp) {
        spins.components[comp][site_index] *= factor;
    }
}
#else
template<size_t N_SU2>
__device__
void optimized_spin_operation(
    double* spins,
    int site_index,
    double factor
) {
    // Legacy AoS version
    for (size_t comp = 0; comp < N_SU2; ++comp) {
        spins[site_index * N_SU2 + comp] *= factor;
    }
}
#endif

// Example 7: Host function to demonstrate usage
void demonstrate_soa_benefits() {
    // These would be actual lattice parameters in real usage
    constexpr size_t N_SU2 = 3;      // SU(2) has 3 components (x,y,z)
    constexpr size_t N_ATOMS_SU2 = 2; // 2 atoms per unit cell
    constexpr size_t N_SU3 = 8;      // SU(3) has 8 components
    constexpr size_t N_ATOMS_SU3 = 1; // 1 atom per unit cell
    constexpr size_t dim1 = 10, dim2 = 10, dim = 10; // 10x10x10 lattice
    
    std::cout << "\n=== SoA Benefits Demonstration ===" << std::endl;
    std::cout << "Lattice size SU2: " << dim1 * dim2 * dim * N_ATOMS_SU2 << " sites" << std::endl;
    std::cout << "Lattice size SU3: " << dim1 * dim2 * dim * N_ATOMS_SU3 << " sites" << std::endl;
    
    // Memory layout comparison
    std::cout << "\nMemory Layout Comparison:" << std::endl;
    std::cout << "AoS stride for SU2: " << N_SU2 << " elements" << std::endl;
    std::cout << "AoS stride for SU3: " << N_SU3 << " elements" << std::endl;
    std::cout << "SoA stride: 1 element (optimal for coalescing)" << std::endl;
    
    // Performance expectations
    double expected_speedup_su2 = static_cast<double>(N_SU2);
    double expected_speedup_su3 = static_cast<double>(N_SU3);
    std::cout << "\nExpected Performance Improvements:" << std::endl;
    std::cout << "SU2 memory bandwidth: up to " << expected_speedup_su2 << "x improvement" << std::endl;
    std::cout << "SU3 memory bandwidth: up to " << expected_speedup_su3 << "x improvement" << std::endl;
    
    std::cout << "\nBest Use Cases for SoA:" << std::endl;
    std::cout << "- Component-wise operations (normalize x-components)" << std::endl;
    std::cout << "- Memory-bound kernels (field calculations)" << std::endl;
    std::cout << "- Large lattice simulations" << std::endl;
    std::cout << "- High-throughput GPU architectures" << std::endl;
    std::cout << "================================\n" << std::endl;
}

#endif // SOA_USAGE_EXAMPLES_H
