#ifndef MIXED_LATTICE_REFACTORED_H
#define MIXED_LATTICE_REFACTORED_H

#include "unitcell.h"
#include "simple_linear_alg.h"
#include "classical_spin/core/spin_config.h"  // For should_rank_write
#include "classical_spin/mc/mc_common.h"      // Common MC structs & templates
#include "classical_spin/lattice/pulse_chunking.h"  // W3: pulse-window chunking helper
#include <vector>
#include <functional>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <mpi.h>
#include <boost/numeric/odeint.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

// Include Boost uBLAS for implicit solvers (rosenbrock4, implicit_euler)
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_controller.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_dense_output.hpp>
#include <boost/numeric/odeint/stepper/implicit_euler.hpp>

#ifdef HDF5_ENABLED
#include "hdf5_io.h"
#endif

// GPU support: API header for all C++ TUs, full .cuh only for CUDA TUs
#ifdef CUDA_ENABLED
#include "mixed_lattice_gpu_api.h"
#endif

#if defined(CUDA_ENABLED) && defined(__CUDACC__)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include "mixed_lattice_gpu.cuh"
#endif

// Optional profiling instrumentation
#ifdef ENABLE_PROFILING
    #define PROFILE_START(name) auto __profile_start_##name = std::chrono::high_resolution_clock::now()
    #define PROFILE_END(name) do { \
        auto __profile_end_##name = std::chrono::high_resolution_clock::now(); \
        auto __profile_duration_##name = std::chrono::duration_cast<std::chrono::microseconds>(__profile_end_##name - __profile_start_##name).count(); \
        std::cout << "[PROFILE] " << #name << ": " << __profile_duration_##name << " us" << std::endl; \
    } while(0)
#else
    #define PROFILE_START(name)
    #define PROFILE_END(name)
#endif

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::function;
using std::array;

// Common MC structs (from mc_common.h)
using mc::BinningResult;
using mc::Observable;
using mc::VectorObservable;
using mc::AutocorrelationResult;
using mc::OptimizedTempGridResult;

// Legacy type aliases for backward compatibility
using MixedBinningResult = mc::BinningResult;
using MixedObservable = mc::Observable;
using MixedVectorObservable = mc::VectorObservable;

// Complete set of thermodynamic observables for mixed lattice with uncertainties
// (Extended version with SU2/SU3 split — MixedLattice-specific)
struct MixedThermodynamicObservables {
    double temperature;
    
    // Energy observables
    Observable energy_total;         // <E>/N_total
    Observable energy_SU2;           // <E_SU2>/N_SU2
    Observable energy_SU3;           // <E_SU3>/N_SU3
    Observable specific_heat;        // C_V = (<E²> - <E>²) / (T² N)
    
    // SU(2) sublattice observables
    vector<VectorObservable> sublattice_magnetization_SU2;  // <S_α> for each SU(2) sublattice
    vector<VectorObservable> energy_sublattice_cross_SU2;   // <E * S_α> - <E><S_α>
    
    // SU(3) sublattice observables
    vector<VectorObservable> sublattice_magnetization_SU3;  // <S_α> for each SU(3) sublattice
    vector<VectorObservable> energy_sublattice_cross_SU3;   // <E * S_α> - <E><S_α>
};

/**
 * MixedLattice class: Template-free implementation for coupled SU(2) and SU(3) systems
 * 
 * This class manages a periodic lattice with two types of spins:
 * - SU(2) spins (typically spin-1/2 with 3 components)
 * - SU(3) spins (typically with 8 components - Gell-Mann generators)
 * 
 * Supports:
 * - Bilinear and trilinear interactions within each sublattice
 * - Mixed bilinear and trilinear interactions between sublattices
 * - Monte Carlo sampling (Metropolis, overrelaxation)
 * - Parallel tempering
 * - Molecular dynamics (Landau-Lifshitz equations)
 * - Time-dependent external fields
 */
class MixedLattice {
public:
    // Type aliases for clarity
    using SpinConfigSU2 = vector<SpinVector>;  // SU(2) spin configuration
    using SpinConfigSU3 = vector<SpinVector>;  // SU(3) spin configuration
    using ODEState = vector<double>;            // Flat state vector for Boost.Odeint

    // Core lattice properties
    size_t spin_dim_SU2;         // Dimension of SU(2) spins (typically 3)
    size_t spin_dim_SU3;         // Dimension of SU(3) spins (typically 8)
    size_t N_atoms_SU2;          // Number of SU(2) atoms per unit cell
    size_t N_atoms_SU3;          // Number of SU(3) atoms per unit cell
    size_t dim1, dim2, dim3;     // Lattice dimensions
    size_t lattice_size_SU2;     // Total SU(2) sites = N_atoms_SU2 * dim1 * dim2 * dim3
    size_t lattice_size_SU3;     // Total SU(3) sites = N_atoms_SU3 * dim1 * dim2 * dim3
    float spin_length_SU2;       // Magnitude of SU(2) spin vectors
    float spin_length_SU3;       // Magnitude of SU(3) spin vectors

    // Spin configurations and positions
    SpinConfigSU2 spins_SU2;                    // Current SU(2) spins
    SpinConfigSU3 spins_SU3;                    // Current SU(3) spins
    vector<Eigen::Vector3d> site_positions_SU2; // Real-space positions for SU(2) sites
    vector<Eigen::Vector3d> site_positions_SU3; // Real-space positions for SU(3) sites

    // SU(2) interaction lookup tables
    vector<SpinVector> field_SU2;                               // External field at each SU(2) site
    vector<SpinMatrix> onsite_interaction_SU2;                  // On-site anisotropy for SU(2)
    vector<vector<SpinMatrix>> bilinear_interaction_SU2;        // SU(2)-SU(2) bilinear coupling
    vector<vector<SpinTensor3>> trilinear_interaction_SU2;      // SU(2)-SU(2)-SU(2) trilinear coupling
    vector<vector<size_t>> bilinear_partners_SU2;               // SU(2) bilinear partner indices
    vector<vector<array<size_t, 2>>> trilinear_partners_SU2;   // SU(2) trilinear partner pairs

    // SU(3) interaction lookup tables
    vector<SpinVector> field_SU3;                               // External field at each SU(3) site
    vector<SpinMatrix> onsite_interaction_SU3;                  // On-site anisotropy for SU(3)
    vector<vector<SpinMatrix>> bilinear_interaction_SU3;        // SU(3)-SU(3) bilinear coupling
    vector<vector<SpinTensor3>> trilinear_interaction_SU3;      // SU(3)-SU(3)-SU(3) trilinear coupling
    vector<vector<size_t>> bilinear_partners_SU3;               // SU(3) bilinear partner indices
    vector<vector<array<size_t, 2>>> trilinear_partners_SU3;   // SU(3) trilinear partner pairs

    // Mixed SU(2)-SU(3) interaction lookup tables
    vector<vector<Eigen::MatrixXd>> mixed_bilinear_interaction_SU2;  // SU(2)-SU(3) bilinear (from SU(2) side)
    vector<vector<Eigen::MatrixXd>> mixed_bilinear_interaction_SU3;  // SU(3)-SU(2) bilinear (from SU(3) side)
    vector<vector<size_t>> mixed_bilinear_partners_SU2;              // SU(3) partner indices for SU(2)
    vector<vector<size_t>> mixed_bilinear_partners_SU3;              // SU(2) partner indices for SU(3)

    vector<vector<SpinTensor3>> mixed_trilinear_interaction_SU2;  // SU(2)-SU(2)-SU(3) trilinear (vector of matrices)
    vector<vector<SpinTensor3>> mixed_trilinear_interaction_SU3;  // SU(3)-SU(2)-SU(2) trilinear (vector of matrices)
    vector<vector<array<size_t, 2>>> mixed_trilinear_partners_SU2;             // (SU(2), SU(3)) partner pairs
    vector<vector<array<size_t, 2>>> mixed_trilinear_partners_SU3;             // (SU(2), SU(2)) partner pairs

    // ============================================================
    // PACKED INTERACTION BUFFERS (SoA, row-major, double*)
    // ============================================================
    // Mirror data already in {bilinear,trilinear,mixed_*}_interaction_*
    // but laid out as one contiguous double[] per site, row-major in
    // (a,b) for bilinear and (a,b,c) for trilinear. This:
    //   - removes the vector<MatrixXd> pointer-chase per bond,
    //   - flips the inner-loop stride from column-major MatrixXd
    //     (stride = rows) to unit-stride row-major (stride = 1),
    //   - lets the compiler vectorize and unroll fixed-size kernels
    //     (3x3, 3x8, 8x8, 3x3x3, 3x3x8, 8x3x3, 8x8x8) used by the
    //     SU(2)/SU(3) MD RHS.
    // Built once in build_packed_interaction_buffers() at the end of the
    // constructor; the original SpinMatrix / SpinTensor3 storage is kept
    // for the MC code path and any rebuild operations.
    //
    // Layout per site (n indexes bonds at this site):
    //   bilinear_packed_*[site][n*da*db + a*db + b]
    //   trilinear_packed_*[site][n*da*db*dc + (a*db + b)*dc + c]
    vector<vector<double>> bilinear_packed_SU2;             // da=db=spin_dim_SU2
    vector<vector<double>> bilinear_packed_SU3;             // da=db=spin_dim_SU3
    vector<vector<double>> mixed_bilinear_packed_SU2;       // da=spin_dim_SU2, db=spin_dim_SU3
    vector<vector<double>> mixed_bilinear_packed_SU3;       // da=spin_dim_SU3, db=spin_dim_SU2
    vector<vector<double>> trilinear_packed_SU2;            // da=db=dc=spin_dim_SU2
    vector<vector<double>> trilinear_packed_SU3;            // da=db=dc=spin_dim_SU3
    vector<vector<double>> mixed_trilinear_packed_SU2;      // da=db=spin_dim_SU2, dc=spin_dim_SU3
    vector<vector<double>> mixed_trilinear_packed_SU3;      // da=spin_dim_SU3, db=dc=spin_dim_SU2

    // Sublattice frame transformations
    vector<SpinMatrix> sublattice_frames_SU2;  // Frame transformations for SU(2) sublattices
    vector<SpinMatrix> sublattice_frames_SU3;  // Frame transformations for SU(3) sublattices
    vector<double> afm_sublattice_signs_SU2;   // AFM sublattice signs for SU(2) (Bertaut modes)

    // Interaction counts per site
    size_t num_bi_SU2;       // Number of SU(2)-SU(2) bilinear neighbors per site
    size_t num_tri_SU2;      // Number of SU(2)-SU(2)-SU(2) trilinear interactions per site
    size_t num_bi_SU3;       // Number of SU(3)-SU(3) bilinear neighbors per site
    size_t num_tri_SU3;      // Number of SU(3)-SU(3)-SU(3) trilinear interactions per site
    size_t num_bi_SU2_SU3;   // Number of mixed bilinear interactions per site
    size_t num_tri_SU2_SU3;  // Number of mixed trilinear interactions per site

    // ------------------------------------------------------------------
    // Sublattice colouring of the bond graph for the parallel coloured
    // Metropolis / over-relaxation sweeps. Built once at init() time.
    //
    // SU(2) and SU(3) sublattices are coloured **independently** because the
    // parallel sweep updates each species in its own pass: while updating
    // SU(2) sites, the SU(3) configuration is frozen, so only intra-SU(2)
    // interactions create write-vs-read races. The relevant edges for the
    // SU(2) graph are: SU(2)-SU(2) bilinear, SU(2)-SU(2)-SU(2) trilinear,
    // and the SU(2)-SU(2) pair inside any SU(2)-SU(2)-SU(3) mixed trilinear.
    // Symmetric story for SU(3).
    //
    // Mixed bilinear (SU(2)-SU(3)) does NOT add intra-sublattice edges.
    //
    // Layout: same CSR pattern as Lattice (cf. lattice.h::color_of_site).
    // ------------------------------------------------------------------
    vector<uint16_t>           color_of_site_SU2;
    vector<size_t>             sites_by_color_csr_off_SU2;   // size n_colors_SU2 + 1
    vector<size_t>             sites_by_color_csr_SU2;       // size lattice_size_SU2
    size_t                     n_colors_SU2 = 0;

    vector<uint16_t>           color_of_site_SU3;
    vector<size_t>             sites_by_color_csr_off_SU3;   // size n_colors_SU3 + 1
    vector<size_t>             sites_by_color_csr_SU3;       // size lattice_size_SU3
    size_t                     n_colors_SU3 = 0;

    // Time-dependent fields for molecular dynamics
    array<SpinVector, 2> field_drive_SU2;     // Two pulse components for SU(2)
    array<SpinVector, 2> field_drive_SU3;     // Two pulse components for SU(3)
    array<double, 2> t_pulse_SU2;             // Pulse center times for SU(2)
    array<double, 2> t_pulse_SU3;             // Pulse center times for SU(3)
    double field_drive_amp_SU2;               // Pulse amplitude for SU(2)
    double field_drive_freq_SU2;              // Pulse frequency for SU(2)
    double field_drive_width_SU2;             // Pulse width (Gaussian) for SU(2)
    double field_drive_amp_SU3;               // Pulse amplitude for SU(3)
    double field_drive_freq_SU3;              // Pulse frequency for SU(3)
    double field_drive_width_SU3;             // Pulse width (Gaussian) for SU(3)

    // SU(2) Gilbert damping: dS/dt += (alpha/|S|) * S × (S × H)
    double alpha_gilbert = 0.0;

    // SU(3) Bloch damping/relaxation (tmfeo3_notes.tex Eq. blochdampedfull)
    // dn^a/dt = (1/ℏ) f_{abc} h^b n^c  −  Γ_a (n^a − n^a_eq)
    // damping_rates_SU3[a]: phenomenological relaxation rates Γ_a for each Gell-Mann channel
    // equilibrium_SU3[site]: thermal equilibrium Bloch vector n^a_eq for each SU(3) site
    SpinVector damping_rates_SU3;               // 8-component, one Γ per Gell-Mann channel
    SpinConfigSU3 equilibrium_SU3;              // Per-site equilibrium Bloch vector

    // ============================================================
    // LOCAL FIELD CACHING FOR OPTIMIZED MONTE CARLO
    // ============================================================
    // Cached local fields for each site (used in interleaved sweeps)
    mutable vector<SpinVector> cached_local_field_SU2;
    mutable vector<SpinVector> cached_local_field_SU3;
    mutable vector<bool> field_valid_SU2;  // Whether cached field is valid
    mutable vector<bool> field_valid_SU3;  // Whether cached field is valid
    mutable bool use_field_caching;        // Enable/disable caching mode

    // Reverse lookup: which SU3 sites are affected by changes to each SU2 site
    vector<vector<size_t>> mixed_bilinear_reverse_SU2;  // SU2[i] -> list of SU3 sites coupled to it
    // Reverse lookup: which SU2 sites are affected by changes to each SU3 site  
    vector<vector<size_t>> mixed_bilinear_reverse_SU3;  // SU3[i] -> list of SU2 sites coupled to it

    /**
     * Constructor: Build a mixed lattice from two unit cells
     * 
     * @param uc_SU2         Unit cell defining the SU(2) sublattice structure
     * @param uc_SU3         Unit cell defining the SU(3) sublattice structure
     * @param dim1           Lattice size in first dimension
     * @param dim2           Lattice size in second dimension
     * @param dim3           Lattice size in third dimension
     * @param spin_l_SU2     Magnitude of SU(2) spin vectors
     * @param spin_l_SU3     Magnitude of SU(3) spin vectors
     */
    MixedLattice(const UnitCell& uc_SU2, const UnitCell& uc_SU3,
                 size_t dim1, size_t dim2, size_t dim3,
                 float spin_l_SU2 = 1.0, float spin_l_SU3 = 1.0)
        : MixedLattice(MixedUnitCell(uc_SU2, uc_SU3), dim1, dim2, dim3, spin_l_SU2, spin_l_SU3)
    {
        // Delegating constructor - mixed interactions will be empty
        cout << "Note: Using separate unit cells - mixed SU(2)-SU(3) interactions not set." << endl;
    }

    /**
     * Constructor: Build a mixed lattice from a MixedUnitCell
     * 
     * @param mixed_uc       Mixed unit cell defining both sublattices and mixed interactions
     * @param dim1           Lattice size in first dimension
     * @param dim2           Lattice size in second dimension
     * @param dim3           Lattice size in third dimension
     * @param spin_l_SU2     Magnitude of SU(2) spin vectors
     * @param spin_l_SU3     Magnitude of SU(3) spin vectors
     */
    MixedLattice(const MixedUnitCell& mixed_uc, size_t dim1, size_t dim2, size_t dim3,
                 float spin_l_SU2 = 1.0, float spin_l_SU3 = 1.0)
        : spin_dim_SU2(mixed_uc.SU2_cell.N),
          spin_dim_SU3(mixed_uc.SU3_cell.N),
          N_atoms_SU2(mixed_uc.SU2_cell.N_atoms),
          N_atoms_SU3(mixed_uc.SU3_cell.N_atoms),
          dim1(dim1), dim2(dim2), dim3(dim3),
          spin_length_SU2(spin_l_SU2),
          spin_length_SU3(spin_l_SU3)
    {
        lattice_size_SU2 = N_atoms_SU2 * dim1 * dim2 * dim3;
        lattice_size_SU3 = N_atoms_SU3 * dim1 * dim2 * dim3;
        
        cout << "Initializing mixed lattice with dimensions: " << dim1 << " x " << dim2 << " x " << dim3 << endl;
        cout << "SU(2): " << lattice_size_SU2 << " sites (" << N_atoms_SU2 << " atoms/cell, spin_dim=" << spin_dim_SU2 << ", spin_length=" << spin_length_SU2 << ")" << endl;
        cout << "SU(3): " << lattice_size_SU3 << " sites (" << N_atoms_SU3 << " atoms/cell, spin_dim=" << spin_dim_SU3 << ", spin_length=" << spin_length_SU3 << ")" << endl;

        // Initialize arrays
        spins_SU2.resize(lattice_size_SU2);
        spins_SU3.resize(lattice_size_SU3);
        site_positions_SU2.resize(lattice_size_SU2);
        site_positions_SU3.resize(lattice_size_SU3);
        
        field_SU2.resize(lattice_size_SU2);
        field_SU3.resize(lattice_size_SU3);
        onsite_interaction_SU2.resize(lattice_size_SU2);
        onsite_interaction_SU3.resize(lattice_size_SU3);
        
        bilinear_interaction_SU2.resize(lattice_size_SU2);
        bilinear_interaction_SU3.resize(lattice_size_SU3);
        trilinear_interaction_SU2.resize(lattice_size_SU2);
        trilinear_interaction_SU3.resize(lattice_size_SU3);
        
        bilinear_partners_SU2.resize(lattice_size_SU2);
        bilinear_partners_SU3.resize(lattice_size_SU3);
        trilinear_partners_SU2.resize(lattice_size_SU2);
        trilinear_partners_SU3.resize(lattice_size_SU3);
        
        mixed_bilinear_interaction_SU2.resize(lattice_size_SU2);
        mixed_bilinear_interaction_SU3.resize(lattice_size_SU3);
        mixed_bilinear_partners_SU2.resize(lattice_size_SU2);
        mixed_bilinear_partners_SU3.resize(lattice_size_SU3);
        
        mixed_trilinear_interaction_SU2.resize(lattice_size_SU2);
        mixed_trilinear_interaction_SU3.resize(lattice_size_SU3);
        mixed_trilinear_partners_SU2.resize(lattice_size_SU2);
        mixed_trilinear_partners_SU3.resize(lattice_size_SU3);
        
        sublattice_frames_SU2.resize(N_atoms_SU2);
        sublattice_frames_SU3.resize(N_atoms_SU3);
        
        // Copy sublattice frames
        for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
            sublattice_frames_SU2[atom] = mixed_uc.SU2_cell.sublattice_frames[atom];
        }
        for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
            sublattice_frames_SU3[atom] = mixed_uc.SU3_cell.sublattice_frames[atom];
        }
        afm_sublattice_signs_SU2 = mixed_uc.SU2_cell.afm_sublattice_signs;

        // Initialize time-dependent fields
        field_drive_SU2[0] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_SU2[1] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_SU3[0] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_SU3[1] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        t_pulse_SU2[0] = 0.0;
        t_pulse_SU2[1] = 0.0;
        t_pulse_SU3[0] = 0.0;
        t_pulse_SU3[1] = 0.0;
        field_drive_amp_SU2 = 0.0;
        field_drive_freq_SU2 = 0.0;
        field_drive_width_SU2 = 1.0;
        field_drive_amp_SU3 = 0.0;
        field_drive_freq_SU3 = 0.0;
        field_drive_width_SU3 = 1.0;

        // SU(3) Bloch damping: default zero rates (no damping)
        damping_rates_SU3 = SpinVector::Zero(spin_dim_SU3);
        // equilibrium_SU3 will be sized after lattice is built (see below)

        // Initialize random seed
        seed_lehman(chrono::system_clock::now().time_since_epoch().count() * 2 + 1);

        // Build SU(2) sublattice
        build_sublattice(mixed_uc.SU2_cell, spins_SU2, site_positions_SU2, field_SU2,
                        onsite_interaction_SU2, bilinear_interaction_SU2,
                        trilinear_interaction_SU2, bilinear_partners_SU2,
                        trilinear_partners_SU2, num_bi_SU2, num_tri_SU2,
                        spin_length_SU2, spin_dim_SU2, N_atoms_SU2);

        // Build SU(3) sublattice
        build_sublattice(mixed_uc.SU3_cell, spins_SU3, site_positions_SU3, field_SU3,
                        onsite_interaction_SU3, bilinear_interaction_SU3,
                        trilinear_interaction_SU3, bilinear_partners_SU3,
                        trilinear_partners_SU3, num_bi_SU3, num_tri_SU3,
                        spin_length_SU3, spin_dim_SU3, N_atoms_SU3);

        // Build mixed SU(2)-SU(3) interactions
        build_mixed_interactions(mixed_uc, num_bi_SU2_SU3, num_tri_SU2_SU3);

        // Initialize local field caching infrastructure
        cached_local_field_SU2.resize(lattice_size_SU2);
        cached_local_field_SU3.resize(lattice_size_SU3);
        field_valid_SU2.resize(lattice_size_SU2, false);
        field_valid_SU3.resize(lattice_size_SU3, false);
        use_field_caching = false;  // Disabled by default

        // Initialize SU(3) Bloch damping equilibrium (default: zero = infinite temperature)
        equilibrium_SU3.resize(lattice_size_SU3, SpinVector::Zero(spin_dim_SU3));

        // Build reverse lookup tables for mixed interactions
        build_reverse_lookup_tables();

        // Build per-sublattice colour partition for the parallel coloured
        // Metropolis / over-relaxation sweeps. See header doc on
        // color_of_site_SU{2,3} for the edge-set we use.
        build_color_partition();

        // Pack {bi,tri}linear interaction tensors into row-major double[]
        // buffers used by the MD hot path (see field declarations).
        build_packed_interaction_buffers();

        cout << "Mixed lattice initialization complete!" << endl;
        cout << "SU(2) - Max bilinear: " << num_bi_SU2 << ", Max trilinear: " << num_tri_SU2 << endl;
        cout << "SU(3) - Max bilinear: " << num_bi_SU3 << ", Max trilinear: " << num_tri_SU3 << endl;
        cout << "Mixed - Bilinear: " << num_bi_SU2_SU3 << ", Trilinear: " << num_tri_SU2_SU3 << endl;
        cout << "SU(2) sublattice colouring: " << n_colors_SU2 << " colour(s)" << endl;
        cout << "SU(3) sublattice colouring: " << n_colors_SU3 << " colour(s)" << endl;
    }

    // ============================================================
    // UTILITY METHODS
    // ============================================================

    /**
     * Flatten multi-index to linear site index
     */
    size_t flatten_index(size_t i, size_t j, size_t k, size_t atom, size_t N_atoms) const {
        return ((i * dim2 + j) * dim3 + k) * N_atoms + atom;
    }

    /**
     * Apply periodic boundary condition
     */
    size_t periodic_boundary(int coord, size_t dim_size) const {
        if (coord < 0) {
            return coord + dim_size;
        } else if (coord >= (int)dim_size) {
            return coord - dim_size;
        }
        return coord;
    }

    /**
     * Flatten with periodic boundaries
     */
    size_t flatten_index_periodic(int i, int j, int k, size_t atom, size_t N_atoms) const {
        return flatten_index(periodic_boundary(i, dim1),
                           periodic_boundary(j, dim2),
                           periodic_boundary(k, dim3),
                           atom, N_atoms);
    }

    /**
     * Generate random spin on n-sphere
     */
    SpinVector gen_random_spin(float spin_l, size_t spin_dim) {
        SpinVector spin(spin_dim);

        if (spin_dim == 3) {
            // Marsaglia (1972) method for uniform sampling on the 2-sphere:
            // rejection-sample u1, u2 uniformly in the unit disk, then map
            // to the sphere algebraically.
            //
            //   x = 2*u1 * sqrt(1 - s)
            //   y = 2*u2 * sqrt(1 - s)
            //   z = 1 - 2*s,    where s = u1^2 + u2^2 < 1
            //
            // Acceptance probability is pi/4 ~ 0.785, so on average ~2.55
            // RNG draws per accepted spin. Replaces the cos+sin+sqrt
            // formulation: same distribution, no transcendental calls.
            // For SU(2) Metropolis this is a few ns per proposed move and
            // is shared between both species (called every accept/reject).
            double u1, u2, s;
            do {
                u1 = 2.0 * random_double_lehman(0.0, 1.0) - 1.0;
                u2 = 2.0 * random_double_lehman(0.0, 1.0) - 1.0;
                s  = u1 * u1 + u2 * u2;
            } while (s >= 1.0);
            const double factor = 2.0 * std::sqrt(1.0 - s);
            spin(0) = factor * u1;
            spin(1) = factor * u2;
            spin(2) = 1.0 - 2.0 * s;
        } else {
            // General n-sphere sampling. For spin_dim==8 (SU(3)) the
            // hypercube + reject-near-zero approach is fine; transcendental
            // savings here would be negligible since the inner loop is d
            // multiplies + 1 sqrt regardless.
            double norm_sq = 0.0;
            do {
                norm_sq = 0.0;
                for (size_t i = 0; i < spin_dim; ++i) {
                    spin(i) = random_double_lehman(-1, 1);
                    norm_sq += spin(i) * spin(i);
                }
            } while (norm_sq < 1e-10);
            spin /= std::sqrt(norm_sq);
        }

        return spin * spin_l;
    }

    /**
     * Build a sublattice from a unit cell
     */
    void build_sublattice(const UnitCell& uc,
                         SpinConfigSU2& spins,
                         vector<Eigen::Vector3d>& positions,
                         vector<SpinVector>& field,
                         vector<SpinMatrix>& onsite,
                         vector<vector<SpinMatrix>>& bilinear,
                         vector<vector<SpinTensor3>>& trilinear,
                         vector<vector<size_t>>& bi_partners,
                         vector<vector<array<size_t, 2>>>& tri_partners,
                         size_t& num_bi, size_t& num_tri,
                         float spin_length, size_t spin_dim, size_t N_atoms)
    {
        const size_t lattice_size = N_atoms * dim1 * dim2 * dim3;

        // Phase 1: Count interactions per site
        vector<size_t> bi_count(lattice_size, 0);
        vector<size_t> tri_count(lattice_size, 0);
        
        size_t site_idx = 0;
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t l = 0; l < N_atoms; ++l) {
                        // Calculate position
                        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
                        for (int d = 0; d < 3; d++) {
                            pos(d) = uc.lattice_vectors[0](d) * int(i) + 
                                    uc.lattice_vectors[1](d) * int(j) + 
                                    uc.lattice_vectors[2](d) * int(k) + 
                                    uc.lattice_pos[l](d);
                        }
                        positions[site_idx] = pos;
                        
                        // Generate random spin
                        spins[site_idx] = gen_random_spin(spin_length, spin_dim);
                        
                        // Copy field and onsite interaction
                        field[site_idx] = uc.field[l];
                        onsite[site_idx] = uc.onsite_interaction[l];
                        
                        // Count bilinear interactions
                        auto bi_range = uc.bilinear_interaction.equal_range(l);
                        for (auto it = bi_range.first; it != bi_range.second; ++it) {
                            const auto& J = it->second;
                            bi_count[site_idx]++;
                            size_t partner = flatten_index_periodic(
                                int(i) + J.offset[0], int(j) + J.offset[1], int(k) + J.offset[2], J.partner, N_atoms);
                            bi_count[partner]++;
                        }
                        
                        // Count trilinear interactions
                        auto tri_range = uc.trilinear_interaction.equal_range(l);
                        for (auto it = tri_range.first; it != tri_range.second; ++it) {
                            const auto& J = it->second;
                            size_t partner1 = flatten_index_periodic(
                                int(i) + J.offset1[0], int(j) + J.offset1[1], int(k) + J.offset1[2], J.partner1, N_atoms);
                            size_t partner2 = flatten_index_periodic(
                                int(i) + J.offset2[0], int(j) + J.offset2[1], int(k) + J.offset2[2], J.partner2, N_atoms);
                            tri_count[site_idx]++;
                            tri_count[partner1]++;
                            tri_count[partner2]++;
                        }
                        
                        site_idx++;
                    }
                }
            }
        }

        // Phase 2: Allocate storage
        for (size_t idx = 0; idx < lattice_size; ++idx) {
            bilinear[idx].reserve(bi_count[idx]);
            bi_partners[idx].reserve(bi_count[idx]);
            trilinear[idx].reserve(tri_count[idx]);
            tri_partners[idx].reserve(tri_count[idx]);
        }

        // Phase 3: Build interactions
        site_idx = 0;
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t l = 0; l < N_atoms; ++l) {
                        // Bilinear interactions
                        auto bi_range = uc.bilinear_interaction.equal_range(l);
                        for (auto it = bi_range.first; it != bi_range.second; ++it) {
                            const auto& J = it->second;
                            size_t partner = flatten_index_periodic(
                                int(i) + J.offset[0], int(j) + J.offset[1], int(k) + J.offset[2], J.partner, N_atoms);
                            
                            bilinear[site_idx].push_back(J.interaction);
                            bi_partners[site_idx].push_back(partner);
                            
                            // Symmetric interaction
                            bilinear[partner].push_back(J.interaction.transpose());
                            bi_partners[partner].push_back(site_idx);
                        }
                        
                        // Trilinear interactions
                        auto tri_range = uc.trilinear_interaction.equal_range(l);
                        for (auto it = tri_range.first; it != tri_range.second; ++it) {
                            const auto& J = it->second;
                            size_t partner1 = flatten_index_periodic(
                                int(i) + J.offset1[0], int(j) + J.offset1[1], int(k) + J.offset1[2], J.partner1, N_atoms);
                            size_t partner2 = flatten_index_periodic(
                                int(i) + J.offset2[0], int(j) + J.offset2[1], int(k) + J.offset2[2], J.partner2, N_atoms);
                            
                            // Add interaction T_abc S_a^(0) S_b^(1) S_c^(2)
                            trilinear[site_idx].push_back(J.interaction);
                            tri_partners[site_idx].push_back({partner1, partner2});
                            
                            // Add symmetric contributions for energy conservation
                            // For partner1: T_bac S_b^(1) S_a^(0) S_c^(2) (swap first two indices)
                            // T_permuted[b](a,c) = T_original[a](b,c)
                            SpinTensor3 tensor_p1(spin_dim);
                            for (size_t b = 0; b < spin_dim; ++b) {
                                tensor_p1[b] = SpinMatrix(spin_dim, spin_dim);
                                for (size_t a = 0; a < spin_dim; ++a) {
                                    for (size_t c = 0; c < spin_dim; ++c) {
                                        tensor_p1[b](a, c) = J.interaction[a](b, c);
                                    }
                                }
                            }
                            trilinear[partner1].push_back(tensor_p1);
                            tri_partners[partner1].push_back({site_idx, partner2});
                            
                            // For partner2: T_cab S_c^(2) S_a^(0) S_b^(1) (cyclic permutation)
                            // T_permuted[c](a,b) = T_original[a](b,c)
                            SpinTensor3 tensor_p2(spin_dim);
                            for (size_t c = 0; c < spin_dim; ++c) {
                                tensor_p2[c] = SpinMatrix(spin_dim, spin_dim);
                                for (size_t a = 0; a < spin_dim; ++a) {
                                    for (size_t b = 0; b < spin_dim; ++b) {
                                        tensor_p2[c](a, b) = J.interaction[a](b, c);
                                    }
                                }
                            }
                            trilinear[partner2].push_back(tensor_p2);
                            tri_partners[partner2].push_back({site_idx, partner1});
                        }
                        
                        site_idx++;
                    }
                }
            }
        }

        num_bi = *std::max_element(bi_count.begin(), bi_count.end());
        num_tri = *std::max_element(tri_count.begin(), tri_count.end());
    }

    /**
     * Build mixed SU(2)-SU(3) interactions from MixedUnitCell
     */
    void build_mixed_interactions(const MixedUnitCell& mixed_uc, 
                                  size_t& num_bi_mixed, size_t& num_tri_mixed)
    {
        // Compute max mixed interaction counts
        num_bi_mixed = 0;
        num_tri_mixed = 0;
        
        // Count bilinear interactions
        for (auto it = mixed_uc.bilinear_SU2_SU3.begin(); 
             it != mixed_uc.bilinear_SU2_SU3.end(); ) {
            int source = it->first;
            auto range = mixed_uc.bilinear_SU2_SU3.equal_range(source);
            num_bi_mixed = std::max(num_bi_mixed, static_cast<size_t>(std::distance(range.first, range.second)));
            it = range.second;
        }
        
        // Count trilinear interactions
        for (auto it = mixed_uc.trilinear_SU2_SU3.begin();
             it != mixed_uc.trilinear_SU2_SU3.end(); ) {
            int source = it->first;
            auto range = mixed_uc.trilinear_SU2_SU3.equal_range(source);
            num_tri_mixed = std::max(num_tri_mixed, static_cast<size_t>(std::distance(range.first, range.second)));
            it = range.second;
        }

        cout << "Building mixed interactions: max " << num_bi_mixed 
             << " bilinear, " << num_tri_mixed << " trilinear per site" << endl;

        // Build mixed bilinear interactions
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    // Process SU(2) sites as sources
                    for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
                        size_t site_idx = flatten_index(i, j, k, atom, N_atoms_SU2);
                        
                        auto bi_range = mixed_uc.bilinear_SU2_SU3.equal_range(atom);
                        for (auto it = bi_range.first; it != bi_range.second; ++it) {
                            const auto& bi = it->second;
                            // Partner is in SU(3) sublattice
                            int pi = static_cast<int>(i) + bi.offset(0);
                            int pj = static_cast<int>(j) + bi.offset(1);
                            int pk = static_cast<int>(k) + bi.offset(2);
                            size_t partner_idx = flatten_index_periodic(pi, pj, pk, bi.partner, N_atoms_SU3);
                            
                            // bi.interaction is N_SU2 x N_SU3 (3x8)
                            // For SU2 energy: S2.dot(J * S3), need J to be N_SU2 x N_SU3 (3x8)
                            // For SU3 energy: S3.dot(J^T * S2), need J^T to be N_SU3 x N_SU2 (8x3)
                            mixed_bilinear_interaction_SU2[site_idx].push_back(bi.interaction);
                            mixed_bilinear_partners_SU2[site_idx].push_back(partner_idx);
                            
                            // Add transposed contribution to SU(3) side
                            mixed_bilinear_interaction_SU3[partner_idx].push_back(bi.interaction.transpose());
                            mixed_bilinear_partners_SU3[partner_idx].push_back(site_idx);
                        }
                    }
                }
            }
        }

        // Build mixed trilinear interactions
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    // Process SU(2) sites as sources
                    for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
                        size_t site_idx = flatten_index(i, j, k, atom, N_atoms_SU2);
                        
                        auto tri_range = mixed_uc.trilinear_SU2_SU3.equal_range(atom);
                        for (auto it = tri_range.first; it != tri_range.second; ++it) {
                            const auto& tri = it->second;
                            
                            // First partner (SU2)
                            int p1i = static_cast<int>(i) + tri.offset1(0);
                            int p1j = static_cast<int>(j) + tri.offset1(1);
                            int p1k = static_cast<int>(k) + tri.offset1(2);
                            size_t partner1_idx = flatten_index_periodic(p1i, p1j, p1k, tri.partner1, N_atoms_SU2);
                            
                            // Second partner (SU3)
                            int p2i = static_cast<int>(i) + tri.offset2(0);
                            int p2j = static_cast<int>(j) + tri.offset2(1);
                            int p2k = static_cast<int>(k) + tri.offset2(2);
                            size_t partner2_idx = flatten_index_periodic(p2i, p2j, p2k, tri.partner2, N_atoms_SU3);
                            
                            // Original: K[a](b,c) for site[a] with SU2[b] and SU3[c]
                            // H = Σ_abc K[a](b,c) S_source^a S_partner1^b λ_partner2^c
                            mixed_trilinear_interaction_SU2[site_idx].push_back(tri.interaction);
                            mixed_trilinear_partners_SU2[site_idx].push_back({partner1_idx, partner2_idx});
                            
                            // Symmetric contribution to SU2 partner1: K_bac[b](a,c) = K[a](b,c)
                            // ∂H/∂S_partner1^b = Σ_ac K[a](b,c) S_source^a λ^c
                            // For partner1: T_p1[b] is (spin_dim_SU2 × spin_dim_SU3)
                            //   with T_p1[b](a,c) = K[a](b,c)
                            SpinTensor3 K_bac(spin_dim_SU2);
                            for (size_t b = 0; b < spin_dim_SU2; ++b) {
                                K_bac[b] = Eigen::MatrixXd(spin_dim_SU2, spin_dim_SU3);
                                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                                        K_bac[b](a, c) = tri.interaction[a](b, c);
                                    }
                                }
                            }
                            mixed_trilinear_interaction_SU2[partner1_idx].push_back(K_bac);
                            mixed_trilinear_partners_SU2[partner1_idx].push_back({site_idx, partner2_idx});
                            
                            // Symmetric contribution to SU3 site: K_cab[c](a,b) = K[a](b,c)
                            // ∂H/∂λ^c = Σ_ab K[a](b,c) S_source^a S_partner1^b
                            SpinTensor3 K_cab(spin_dim_SU3);
                            for (size_t c = 0; c < spin_dim_SU3; ++c) {
                                K_cab[c] = Eigen::MatrixXd(spin_dim_SU2, spin_dim_SU2);
                                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                                        K_cab[c](a, b) = tri.interaction[a](b, c);
                                    }
                                }
                            }
                            mixed_trilinear_interaction_SU3[partner2_idx].push_back(K_cab);
                            mixed_trilinear_partners_SU3[partner2_idx].push_back({site_idx, partner1_idx});
                        }
                    }
                }
            }
        }

        cout << "Mixed interactions built successfully!" << endl;
    }

    // ============================================================
    // PACKED INTERACTION BUFFER BUILDER
    // ============================================================
    /**
     * Pack the bilinear / trilinear / mixed interaction tensors into
     * contiguous row-major double[] buffers used by the MD hot path
     * (`get_local_field_*_flat_into`).
     *
     * Layout (per site, n indexes the bond):
     *   bilinear_packed_*[site][n*da*db + a*db + b]              = J^n(a,b)
     *   trilinear_packed_*[site][n*da*db*dc + (a*db + b)*dc + c] = T^n[a](b,c)
     *
     * The original `bilinear_interaction_*`, `trilinear_interaction_*`,
     * `mixed_*_interaction_*` storage is preserved (used by MC code path
     * and for any rebuilds). Only one rebuild call is required after the
     * full interaction graph is set; in our setup that is at the end of
     * the `MixedLattice` constructor (after `build_color_partition()`).
     *
     * Idempotent and safe to call again after the user mutates the
     * interaction tables (e.g. via `set_*` helpers); cost is O(total
     * coupling tensor entries), trivially ~ms even for large lattices.
     */
    void build_packed_interaction_buffers() {
        const size_t d2 = spin_dim_SU2;
        const size_t d3 = spin_dim_SU3;
        const size_t d22  = d2 * d2;
        const size_t d23  = d2 * d3;
        const size_t d33  = d3 * d3;
        const size_t d222 = d22 * d2;
        const size_t d223 = d22 * d3;   // SU2-SU2-SU3 mixed trilinear (from SU2 side)
        const size_t d322 = d3 * d22;   // SU3-SU2-SU2 mixed trilinear (from SU3 side)
        const size_t d333 = d33 * d3;

        bilinear_packed_SU2.assign(lattice_size_SU2, {});
        mixed_bilinear_packed_SU2.assign(lattice_size_SU2, {});
        trilinear_packed_SU2.assign(lattice_size_SU2, {});
        mixed_trilinear_packed_SU2.assign(lattice_size_SU2, {});

        bilinear_packed_SU3.assign(lattice_size_SU3, {});
        mixed_bilinear_packed_SU3.assign(lattice_size_SU3, {});
        trilinear_packed_SU3.assign(lattice_size_SU3, {});
        mixed_trilinear_packed_SU3.assign(lattice_size_SU3, {});

        // ---- SU(2) sublattice ----
        for (size_t site = 0; site < lattice_size_SU2; ++site) {
            // Bilinear SU(2)-SU(2): J(a,b)
            const size_t n_bi = bilinear_interaction_SU2[site].size();
            bilinear_packed_SU2[site].assign(n_bi * d22, 0.0);
            for (size_t n = 0; n < n_bi; ++n) {
                const auto& J = bilinear_interaction_SU2[site][n];
                double* p = bilinear_packed_SU2[site].data() + n * d22;
                for (size_t a = 0; a < d2; ++a)
                    for (size_t b = 0; b < d2; ++b)
                        p[a * d2 + b] = J(a, b);
            }
            // Mixed bilinear SU(2)-SU(3): J(a,c)
            const size_t n_mb = mixed_bilinear_interaction_SU2[site].size();
            mixed_bilinear_packed_SU2[site].assign(n_mb * d23, 0.0);
            for (size_t n = 0; n < n_mb; ++n) {
                const auto& J = mixed_bilinear_interaction_SU2[site][n];
                double* p = mixed_bilinear_packed_SU2[site].data() + n * d23;
                for (size_t a = 0; a < d2; ++a)
                    for (size_t c = 0; c < d3; ++c)
                        p[a * d3 + c] = J(a, c);
            }
            // Trilinear SU(2)-SU(2)-SU(2): T[a](b,c)
            const size_t n_tri = trilinear_interaction_SU2[site].size();
            trilinear_packed_SU2[site].assign(n_tri * d222, 0.0);
            for (size_t n = 0; n < n_tri; ++n) {
                const auto& T = trilinear_interaction_SU2[site][n];
                double* p = trilinear_packed_SU2[site].data() + n * d222;
                for (size_t a = 0; a < d2; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < d2; ++b)
                        for (size_t c = 0; c < d2; ++c)
                            p[(a * d2 + b) * d2 + c] = Ta(b, c);
                }
            }
            // Mixed trilinear SU(2)-SU(2)-SU(3): T[a](b,c)
            const size_t n_mtri = mixed_trilinear_interaction_SU2[site].size();
            mixed_trilinear_packed_SU2[site].assign(n_mtri * d223, 0.0);
            for (size_t n = 0; n < n_mtri; ++n) {
                const auto& T = mixed_trilinear_interaction_SU2[site][n];
                double* p = mixed_trilinear_packed_SU2[site].data() + n * d223;
                for (size_t a = 0; a < d2; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < d2; ++b)
                        for (size_t c = 0; c < d3; ++c)
                            p[(a * d2 + b) * d3 + c] = Ta(b, c);
                }
            }
        }

        // ---- SU(3) sublattice ----
        for (size_t site = 0; site < lattice_size_SU3; ++site) {
            // Bilinear SU(3)-SU(3): J(a,b)
            const size_t n_bi = bilinear_interaction_SU3[site].size();
            bilinear_packed_SU3[site].assign(n_bi * d33, 0.0);
            for (size_t n = 0; n < n_bi; ++n) {
                const auto& J = bilinear_interaction_SU3[site][n];
                double* p = bilinear_packed_SU3[site].data() + n * d33;
                for (size_t a = 0; a < d3; ++a)
                    for (size_t b = 0; b < d3; ++b)
                        p[a * d3 + b] = J(a, b);
            }
            // Mixed bilinear SU(3)-SU(2): J(a,b) with a in SU3, b in SU2
            const size_t n_mb = mixed_bilinear_interaction_SU3[site].size();
            mixed_bilinear_packed_SU3[site].assign(n_mb * d3 * d2, 0.0);
            for (size_t n = 0; n < n_mb; ++n) {
                const auto& J = mixed_bilinear_interaction_SU3[site][n];
                double* p = mixed_bilinear_packed_SU3[site].data() + n * d3 * d2;
                for (size_t a = 0; a < d3; ++a)
                    for (size_t b = 0; b < d2; ++b)
                        p[a * d2 + b] = J(a, b);
            }
            // Trilinear SU(3)-SU(3)-SU(3): T[a](b,c)
            const size_t n_tri = trilinear_interaction_SU3[site].size();
            trilinear_packed_SU3[site].assign(n_tri * d333, 0.0);
            for (size_t n = 0; n < n_tri; ++n) {
                const auto& T = trilinear_interaction_SU3[site][n];
                double* p = trilinear_packed_SU3[site].data() + n * d333;
                for (size_t a = 0; a < d3; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < d3; ++b)
                        for (size_t c = 0; c < d3; ++c)
                            p[(a * d3 + b) * d3 + c] = Ta(b, c);
                }
            }
            // Mixed trilinear SU(3)-SU(2)-SU(2): T[a](b,c), a in SU3, b,c in SU2
            const size_t n_mtri = mixed_trilinear_interaction_SU3[site].size();
            mixed_trilinear_packed_SU3[site].assign(n_mtri * d322, 0.0);
            for (size_t n = 0; n < n_mtri; ++n) {
                const auto& T = mixed_trilinear_interaction_SU3[site][n];
                double* p = mixed_trilinear_packed_SU3[site].data() + n * d322;
                for (size_t a = 0; a < d3; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < d2; ++b)
                        for (size_t c = 0; c < d2; ++c)
                            p[(a * d2 + b) * d2 + c] = Ta(b, c);
                }
            }
        }
    }

    // ============================================================
    // LOCAL FIELD CACHING INFRASTRUCTURE
    // ============================================================

    /**
     * Build reverse lookup tables for mixed bilinear interactions
     * 
     * These tables enable efficient cache invalidation:
     * - mixed_bilinear_reverse_SU2[i] = list of SU(3) sites whose fields depend on SU(2) site i
     * - mixed_bilinear_reverse_SU3[i] = list of SU(2) sites whose fields depend on SU(3) site i
     */
    void build_reverse_lookup_tables() {
        // Initialize reverse lookup tables
        mixed_bilinear_reverse_SU2.resize(lattice_size_SU2);
        mixed_bilinear_reverse_SU3.resize(lattice_size_SU3);
        
        // Build reverse lookup from SU(2) -> SU(3)
        // When SU(2) site i changes, we need to invalidate SU(3) sites that couple to it
        for (size_t su3_site = 0; su3_site < lattice_size_SU3; ++su3_site) {
            for (size_t n = 0; n < mixed_bilinear_partners_SU3[su3_site].size(); ++n) {
                size_t su2_partner = mixed_bilinear_partners_SU3[su3_site][n];
                mixed_bilinear_reverse_SU2[su2_partner].push_back(su3_site);
            }
        }
        
        // Build reverse lookup from SU(3) -> SU(2)
        // When SU(3) site i changes, we need to invalidate SU(2) sites that couple to it
        for (size_t su2_site = 0; su2_site < lattice_size_SU2; ++su2_site) {
            for (size_t n = 0; n < mixed_bilinear_partners_SU2[su2_site].size(); ++n) {
                size_t su3_partner = mixed_bilinear_partners_SU2[su2_site][n];
                mixed_bilinear_reverse_SU3[su3_partner].push_back(su2_site);
            }
        }
        
        // Remove duplicates in reverse lookup tables
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            std::sort(mixed_bilinear_reverse_SU2[i].begin(), mixed_bilinear_reverse_SU2[i].end());
            mixed_bilinear_reverse_SU2[i].erase(
                std::unique(mixed_bilinear_reverse_SU2[i].begin(), mixed_bilinear_reverse_SU2[i].end()),
                mixed_bilinear_reverse_SU2[i].end());
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            std::sort(mixed_bilinear_reverse_SU3[i].begin(), mixed_bilinear_reverse_SU3[i].end());
            mixed_bilinear_reverse_SU3[i].erase(
                std::unique(mixed_bilinear_reverse_SU3[i].begin(), mixed_bilinear_reverse_SU3[i].end()),
                mixed_bilinear_reverse_SU3[i].end());
        }
        
        // Report statistics
        size_t max_reverse_SU2 = 0, max_reverse_SU3 = 0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            max_reverse_SU2 = std::max(max_reverse_SU2, mixed_bilinear_reverse_SU2[i].size());
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            max_reverse_SU3 = std::max(max_reverse_SU3, mixed_bilinear_reverse_SU3[i].size());
        }
        if (max_reverse_SU2 > 0 || max_reverse_SU3 > 0) {
            cout << "Reverse lookup tables built: max SU2->SU3=" << max_reverse_SU2 
                 << ", max SU3->SU2=" << max_reverse_SU3 << endl;
        }
    }

    /**
     * Build the per-sublattice colour partition used by metropolis_parallel /
     * overrelaxation_parallel.
     *
     * SU(2) edges considered:
     *   - SU(2)-SU(2) bilinear: bilinear_partners_SU2
     *   - SU(2)-SU(2)-SU(2) trilinear: both partners in trilinear_partners_SU2
     *   - intra-SU(2) pair inside SU(2)-SU(2)-SU(3) mixed trilinear:
     *     mixed_trilinear_partners_SU2[i][.][0] is the other SU(2) site
     * (mixed *bilinear* SU(2)-SU(3) does NOT add intra-SU(2) edges; the SU(3)
     * is read but not written during the SU(2) parallel pass.)
     *
     * SU(3) edges considered: mirror of the above.
     *
     * Algorithm: greedy first-fit on natural site order. Builds the CSR
     * sites_by_color_csr_off_SU{2,3} / sites_by_color_csr_SU{2,3} tables.
     */
    void build_color_partition() {
        auto greedy = [&](size_t N,
                          const std::vector<std::vector<size_t>>& edges,
                          std::vector<uint16_t>& color_of_site,
                          std::vector<size_t>& csr_off,
                          std::vector<size_t>& csr,
                          size_t& n_colors_out) {
            if (N == 0) {
                n_colors_out = 0;
                color_of_site.clear();
                csr_off.clear();
                csr.clear();
                return;
            }
            color_of_site.assign(N, std::numeric_limits<uint16_t>::max());
            std::vector<uint8_t> forbidden(64, 0);
            size_t max_color = 0;
            for (size_t i = 0; i < N; ++i) {
                std::fill(forbidden.begin(), forbidden.end(), uint8_t(0));
                for (size_t j : edges[i]) {
                    if (j >= N) continue;
                    const uint16_t c = color_of_site[j];
                    if (c == std::numeric_limits<uint16_t>::max()) continue;
                    if (size_t(c) >= forbidden.size()) forbidden.resize(size_t(c) + 1, 0);
                    forbidden[c] = 1;
                }
                uint16_t chosen = 0;
                while (size_t(chosen) < forbidden.size() && forbidden[chosen]) ++chosen;
                color_of_site[i] = chosen;
                if (size_t(chosen) > max_color) max_color = chosen;
            }
            n_colors_out = max_color + 1;
            csr_off.assign(n_colors_out + 1, 0);
            for (size_t i = 0; i < N; ++i) csr_off[color_of_site[i] + 1] += 1;
            for (size_t c = 0; c < n_colors_out; ++c) csr_off[c + 1] += csr_off[c];
            csr.assign(N, 0);
            std::vector<size_t> cursor(n_colors_out, 0);
            for (size_t i = 0; i < N; ++i) {
                const uint16_t c = color_of_site[i];
                csr[csr_off[c] + cursor[c]] = i;
                ++cursor[c];
            }
        };

        // Build undirected adjacency for SU(2) sublattice.
        std::vector<std::vector<size_t>> adj_SU2(lattice_size_SU2);
        auto add_edge_SU2 = [&](size_t a, size_t b) {
            if (a == b || a >= lattice_size_SU2 || b >= lattice_size_SU2) return;
            adj_SU2[a].push_back(b);
            adj_SU2[b].push_back(a);
        };
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t j : bilinear_partners_SU2[i]) add_edge_SU2(i, j);
            for (const auto& pr : trilinear_partners_SU2[i]) {
                add_edge_SU2(i, pr[0]);
                add_edge_SU2(i, pr[1]);
                add_edge_SU2(pr[0], pr[1]);
            }
            for (const auto& pr : mixed_trilinear_partners_SU2[i]) {
                add_edge_SU2(i, pr[0]);  // pr[0] is another SU(2); pr[1] is SU(3)
            }
        }
        greedy(lattice_size_SU2, adj_SU2,
               color_of_site_SU2, sites_by_color_csr_off_SU2,
               sites_by_color_csr_SU2, n_colors_SU2);

        // Build undirected adjacency for SU(3) sublattice.
        std::vector<std::vector<size_t>> adj_SU3(lattice_size_SU3);
        auto add_edge_SU3 = [&](size_t a, size_t b) {
            if (a == b || a >= lattice_size_SU3 || b >= lattice_size_SU3) return;
            adj_SU3[a].push_back(b);
            adj_SU3[b].push_back(a);
        };
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t j : bilinear_partners_SU3[i]) add_edge_SU3(i, j);
            for (const auto& pr : trilinear_partners_SU3[i]) {
                add_edge_SU3(i, pr[0]);
                add_edge_SU3(i, pr[1]);
                add_edge_SU3(pr[0], pr[1]);
            }
            // mixed_trilinear_partners_SU3 stores (SU(2), SU(2)) pairs from the
            // SU(3) site's perspective, so they create no intra-SU(3) edges.
        }
        greedy(lattice_size_SU3, adj_SU3,
               color_of_site_SU3, sites_by_color_csr_off_SU3,
               sites_by_color_csr_SU3, n_colors_SU3);
    }

    /**
     * Enable or disable local field caching mode
     * 
     * When enabled, local fields are cached and only invalidated when
     * neighboring spins change. This is beneficial for interleaved sweeps
     * with mixed interactions.
     */
    void enable_field_caching(bool enable = true) {
        use_field_caching = enable;
        if (enable) {
            invalidate_all_fields();
        }
    }

    /**
     * Initialize field cache by computing all local fields
     */
    void init_field_cache() const {
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            if (!field_valid_SU2[i]) {
                cached_local_field_SU2[i] = get_local_field_SU2(i);
                field_valid_SU2[i] = true;
            }
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            if (!field_valid_SU3[i]) {
                cached_local_field_SU3[i] = get_local_field_SU3(i);
                field_valid_SU3[i] = true;
            }
        }
    }

    /**
     * Invalidate all cached fields
     */
    void invalidate_all_fields() const {
        std::fill(field_valid_SU2.begin(), field_valid_SU2.end(), false);
        std::fill(field_valid_SU3.begin(), field_valid_SU3.end(), false);
    }

    /**
     * Invalidate fields affected by an SU(2) spin update
     * 
     * When SU(2) site i is updated:
     * - All SU(2) sites coupled to i via bilinear/trilinear interactions
     * - All SU(3) sites coupled to i via mixed interactions
     */
    void invalidate_fields_from_SU2_update(size_t su2_site) const {
        // Invalidate the updated site itself
        field_valid_SU2[su2_site] = false;
        
        // Invalidate SU(2) neighbors (bilinear partners)
        for (size_t partner : bilinear_partners_SU2[su2_site]) {
            field_valid_SU2[partner] = false;
        }
        
        // Invalidate SU(2) trilinear partners
        for (const auto& partners : trilinear_partners_SU2[su2_site]) {
            field_valid_SU2[partners[0]] = false;
            field_valid_SU2[partners[1]] = false;
        }
        
        // Invalidate SU(3) sites coupled via mixed bilinear
        for (size_t su3_site : mixed_bilinear_reverse_SU2[su2_site]) {
            field_valid_SU3[su3_site] = false;
        }
        
        // Invalidate SU(2) and SU(3) sites coupled via mixed trilinear
        for (const auto& partners : mixed_trilinear_partners_SU2[su2_site]) {
            field_valid_SU2[partners[0]] = false;  // SU(2) partner
            field_valid_SU3[partners[1]] = false;  // SU(3) partner
        }
    }

    /**
     * Invalidate fields affected by an SU(3) spin update
     * 
     * When SU(3) site i is updated:
     * - All SU(3) sites coupled to i via bilinear/trilinear interactions
     * - All SU(2) sites coupled to i via mixed interactions
     */
    void invalidate_fields_from_SU3_update(size_t su3_site) const {
        // Invalidate the updated site itself
        field_valid_SU3[su3_site] = false;
        
        // Invalidate SU(3) neighbors (bilinear partners)
        for (size_t partner : bilinear_partners_SU3[su3_site]) {
            field_valid_SU3[partner] = false;
        }
        
        // Invalidate SU(3) trilinear partners
        for (const auto& partners : trilinear_partners_SU3[su3_site]) {
            field_valid_SU3[partners[0]] = false;
            field_valid_SU3[partners[1]] = false;
        }
        
        // Invalidate SU(2) sites coupled via mixed bilinear
        for (size_t su2_site : mixed_bilinear_reverse_SU3[su3_site]) {
            field_valid_SU2[su2_site] = false;
        }
        
        // Invalidate SU(2) and SU(3) sites coupled via mixed trilinear (SU3-SU2-SU2)
        for (const auto& partners : mixed_trilinear_partners_SU3[su3_site]) {
            field_valid_SU2[partners[0]] = false;  // SU(2) partner 1
            field_valid_SU2[partners[1]] = false;  // SU(2) partner 2
        }
    }

    /**
     * Get cached local field for SU(2) site (computes if invalid)
     */
    SpinVector get_cached_local_field_SU2(size_t site_index) const {
        if (!field_valid_SU2[site_index]) {
            cached_local_field_SU2[site_index] = get_local_field_SU2(site_index);
            field_valid_SU2[site_index] = true;
        }
        return cached_local_field_SU2[site_index];
    }

    /**
     * Get cached local field for SU(3) site (computes if invalid)
     */
    SpinVector get_cached_local_field_SU3(size_t site_index) const {
        if (!field_valid_SU3[site_index]) {
            cached_local_field_SU3[site_index] = get_local_field_SU3(site_index);
            field_valid_SU3[site_index] = true;
        }
        return cached_local_field_SU3[site_index];
    }

    // ============================================================
    // ENERGY CALCULATIONS
    // ============================================================

    /**
     * Compute energy difference for an SU(2) spin flip
     */
    double site_energy_SU2_diff(const SpinVector& new_spin, const SpinVector& old_spin, size_t site_index) const {
        const SpinVector spin_diff = new_spin - old_spin;
        
        // Field energy
        double field_energy = -spin_diff.dot(field_SU2[site_index]);
        
        // Onsite energy
        double onsite_energy = (new_spin + old_spin).dot(onsite_interaction_SU2[site_index] * spin_diff);
        
        // Bilinear SU(2)-SU(2) interactions
        double bilinear_energy = 0.0;
        for (size_t i = 0; i < bilinear_partners_SU2[site_index].size(); ++i) {
            const size_t partner_idx = bilinear_partners_SU2[site_index][i];
            bilinear_energy += spin_diff.dot(bilinear_interaction_SU2[site_index][i] * spins_SU2[partner_idx]);
        }
        
        // Mixed bilinear SU(2)-SU(3) interactions
        double mixed_bilinear_energy = 0.0;
        for (size_t i = 0; i < mixed_bilinear_partners_SU2[site_index].size(); ++i) {
            const size_t partner_idx = mixed_bilinear_partners_SU2[site_index][i];
            mixed_bilinear_energy += spin_diff.dot(mixed_bilinear_interaction_SU2[site_index][i] * spins_SU3[partner_idx]);
        }
        
        // Trilinear SU(2)-SU(2)-SU(2) interactions.
        //
        // For a single-spin Metropolis move at site i, the change in any
        // trilinear term T_{abc} S^i_a S^j_b S^k_c is
        //     dE = (S_new - S_old) . V,  V[a] = sum_{bc} T[a,b,c] S^j_b S^k_c
        // when neither partner is site i. This collapses two O(d^3)
        // monomial evaluations (old and new) into a single O(d^3)
        // contraction plus an O(d) dot product -- the canonical
        // tensor-network "contract first, project later" optimization.
        // The rare self-coupling case (p1==i or p2==i) still needs the
        // explicit old/new path because S^i appears in two or three slots.
        double trilinear_energy = 0.0;
        for (size_t i = 0; i < trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU2[site_index][i][1];
            const auto& T = trilinear_interaction_SU2[site_index][i];
            const bool p1_self = (p1_idx == site_index);
            const bool p2_self = (p2_idx == site_index);

            if (!p1_self && !p2_self) {
                // Fast path (the common case): pre-contract partners.
                const SpinVector& p1 = spins_SU2[p1_idx];
                const SpinVector& p2 = spins_SU2[p2_idx];
                double dE_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double Va = 0.0;
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        const double p1b = p1(b);
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            Va += Ta(b, c) * p1b * p2(c);
                        }
                    }
                    dE_term += spin_diff(a) * Va;
                }
                trilinear_energy += dE_term;  // multiplicity == 1
            } else {
                // Slow path: a single-spin flip changes more than one slot
                // of the trilinear monomial, so the full old/new evaluation
                // is required and a multiplicity correction restores the
                // unique-term counting expected by total_energy().
                const SpinVector& p1_old = p1_self ? old_spin : spins_SU2[p1_idx];
                const SpinVector& p1_new = p1_self ? new_spin : spins_SU2[p1_idx];
                const SpinVector& p2_old = p2_self ? old_spin : spins_SU2[p2_idx];
                const SpinVector& p2_new = p2_self ? new_spin : spins_SU2[p2_idx];

                double old_term = 0.0;
                double new_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            const double coeff = Ta(b, c);
                            old_term += coeff * old_spin(a) * p1_old(b) * p2_old(c);
                            new_term += coeff * new_spin(a) * p1_new(b) * p2_new(c);
                        }
                    }
                }
                const double multiplicity = 1.0 +
                    (p1_self ? 1.0 : 0.0) + (p2_self ? 1.0 : 0.0);
                trilinear_energy += (new_term - old_term) / multiplicity;
            }
        }

        // Mixed trilinear SU(2)-SU(2)-SU(3) interactions.
        // The SU(3) partner can never collide with an SU(2) site (different
        // species), so only the SU(2) partner p1 may collide.
        double mixed_trilinear_energy = 0.0;
        for (size_t i = 0; i < mixed_trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU2[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU2[site_index][i];
            const bool p1_self = (p1_idx == site_index);

            if (!p1_self) {
                // Fast path: pre-contract V[a] = sum_{bc} T[a,b,c] p1(b) p2(c).
                const SpinVector& p1 = spins_SU2[p1_idx];
                const SpinVector& p2 = spins_SU3[p2_idx];
                double dE_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double Va = 0.0;
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        const double p1b = p1(b);
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            Va += Ta(b, c) * p1b * p2(c);
                        }
                    }
                    dE_term += spin_diff(a) * Va;
                }
                mixed_trilinear_energy += dE_term;  // multiplicity == 1
            } else {
                const SpinVector& p1_old = old_spin;
                const SpinVector& p1_new = new_spin;
                const SpinVector& p2 = spins_SU3[p2_idx];
                double old_term = 0.0;
                double new_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            const double coeff = Ta(b, c);
                            old_term += coeff * old_spin(a) * p1_old(b) * p2(c);
                            new_term += coeff * new_spin(a) * p1_new(b) * p2(c);
                        }
                    }
                }
                mixed_trilinear_energy += (new_term - old_term) / 2.0;
            }
        }
        
        return field_energy + onsite_energy + bilinear_energy + mixed_bilinear_energy + 
               trilinear_energy + mixed_trilinear_energy;
    }

    /**
     * Compute energy difference for an SU(3) spin flip
     */
    double site_energy_SU3_diff(const SpinVector& new_spin, const SpinVector& old_spin, size_t site_index) const {
        const SpinVector spin_diff = new_spin - old_spin;
        
        // Field energy
        double field_energy = -spin_diff.dot(field_SU3[site_index]);
        
        // Onsite energy
        double onsite_energy = (new_spin + old_spin).dot(onsite_interaction_SU3[site_index] * spin_diff);
        
        // Bilinear SU(3)-SU(3) interactions
        double bilinear_energy = 0.0;
        for (size_t i = 0; i < bilinear_partners_SU3[site_index].size(); ++i) {
            const size_t partner_idx = bilinear_partners_SU3[site_index][i];
            bilinear_energy += spin_diff.dot(bilinear_interaction_SU3[site_index][i] * spins_SU3[partner_idx]);
        }
        
        // Mixed bilinear SU(3)-SU(2) interactions
        double mixed_bilinear_energy = 0.0;
        for (size_t i = 0; i < mixed_bilinear_partners_SU3[site_index].size(); ++i) {
            const size_t partner_idx = mixed_bilinear_partners_SU3[site_index][i];
            mixed_bilinear_energy += spin_diff.dot(mixed_bilinear_interaction_SU3[site_index][i] * spins_SU2[partner_idx]);
        }
        
        // Trilinear SU(3)-SU(3)-SU(3) interactions.
        // See site_energy_SU2_diff for the algebra; the savings here are
        // proportionally larger because spin_dim_SU3 = 8 (so a triple loop
        // is 512 multiply-adds vs 27 for SU(2)).
        double trilinear_energy = 0.0;
        for (size_t i = 0; i < trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU3[site_index][i][1];
            const auto& T = trilinear_interaction_SU3[site_index][i];
            const bool p1_self = (p1_idx == site_index);
            const bool p2_self = (p2_idx == site_index);

            if (!p1_self && !p2_self) {
                const SpinVector& p1 = spins_SU3[p1_idx];
                const SpinVector& p2 = spins_SU3[p2_idx];
                double dE_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    double Va = 0.0;
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        const double p1b = p1(b);
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            Va += Ta(b, c) * p1b * p2(c);
                        }
                    }
                    dE_term += spin_diff(a) * Va;
                }
                trilinear_energy += dE_term;  // multiplicity == 1
            } else {
                const SpinVector& p1_old = p1_self ? old_spin : spins_SU3[p1_idx];
                const SpinVector& p1_new = p1_self ? new_spin : spins_SU3[p1_idx];
                const SpinVector& p2_old = p2_self ? old_spin : spins_SU3[p2_idx];
                const SpinVector& p2_new = p2_self ? new_spin : spins_SU3[p2_idx];

                double old_term = 0.0;
                double new_term = 0.0;
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    const auto& Ta = T[a];
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            const double coeff = Ta(b, c);
                            old_term += coeff * old_spin(a) * p1_old(b) * p2_old(c);
                            new_term += coeff * new_spin(a) * p1_new(b) * p2_new(c);
                        }
                    }
                }
                const double multiplicity = 1.0 +
                    (p1_self ? 1.0 : 0.0) + (p2_self ? 1.0 : 0.0);
                trilinear_energy += (new_term - old_term) / multiplicity;
            }
        }

        // Mixed trilinear SU(3)-SU(2)-SU(2) interactions.
        // SU(2) partners are on a different sublattice from the SU(3) site,
        // so collisions are impossible; we always take the fast path.
        double mixed_trilinear_energy = 0.0;
        for (size_t i = 0; i < mixed_trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU3[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU3[site_index][i];
            const SpinVector& p1 = spins_SU2[p1_idx];
            const SpinVector& p2 = spins_SU2[p2_idx];

            double dE_term = 0.0;
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double Va = 0.0;
                const auto& Ta = T[a];
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    const double p1b = p1(b);
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        Va += Ta(b, c) * p1b * p2(c);
                    }
                }
                dE_term += spin_diff(a) * Va;
            }
            mixed_trilinear_energy += dE_term;
        }
        
        return field_energy + onsite_energy + bilinear_energy + mixed_bilinear_energy + 
               trilinear_energy + mixed_trilinear_energy;
    }

    /**
     * Compute total energy of the system as sum of SU2 and SU3 contributions
     */
    double total_energy() const {
        return total_energy_SU2() + total_energy_SU3();
    }

    /**
     * Compute energy density (energy per site)
     */
    double energy_density() const {
        return total_energy() / (lattice_size_SU2 + lattice_size_SU3);
    }

    /**
     * Compute energy density for SU(2) sector (energy per SU2 site)
     */
    double energy_density_SU2() const {
        return total_energy_SU2() / lattice_size_SU2;
    }

    /**
     * Compute energy density for SU(3) sector (energy per SU3 site)
     */
    double energy_density_SU3() const {
        return total_energy_SU3() / lattice_size_SU3;
    }

    /**
     * Compute total energy of the SU(2) sublattice only
     * Includes SU2-SU2 interactions, SU2 field/onsite, and half of mixed interactions
     */
    double total_energy_SU2() const {
        double energy = 0.0;
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            const auto& spin = spins_SU2[i];
            
            // Field and onsite
            energy -= spin.dot(field_SU2[i]);
            energy += spin.dot(onsite_interaction_SU2[i] * spin);
            
            // Bilinear SU2-SU2
            for (size_t j = 0; j < bilinear_partners_SU2[i].size(); ++j) {
                const size_t partner = bilinear_partners_SU2[i][j];
                energy += 0.5 * spin.dot(bilinear_interaction_SU2[i][j] * spins_SU2[partner]);
            }
            
            // Mixed bilinear SU2-SU3 (count half for SU2)
            for (size_t j = 0; j < mixed_bilinear_partners_SU2[i].size(); ++j) {
                const size_t partner = mixed_bilinear_partners_SU2[i][j];
                energy += 0.5 * spin.dot(mixed_bilinear_interaction_SU2[i][j] * spins_SU3[partner]);
            }

            // Mixed trilinear SU2-SU2-SU3
            for (size_t j = 0; j < mixed_trilinear_partners_SU2[i].size(); ++j) {
                const size_t p1 = mixed_trilinear_partners_SU2[i][j][0];
                const size_t p2 = mixed_trilinear_partners_SU2[i][j][1];
                const auto& T = mixed_trilinear_interaction_SU2[i][j];

                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            temp += T[a](b, c) * spins_SU2[p1](b) * spins_SU3[p2](c);
                        }
                    }
                    energy += (1.0 / 3.0) * spin(a) * temp;
                }
            }
            
            // Trilinear SU2-SU2-SU2
            for (size_t j = 0; j < trilinear_partners_SU2[i].size(); ++j) {
                const size_t p1 = trilinear_partners_SU2[i][j][0];
                const size_t p2 = trilinear_partners_SU2[i][j][1];
                const auto& T = trilinear_interaction_SU2[i][j];
                
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            temp += T[a](b, c) * spins_SU2[p1](b) * spins_SU2[p2](c);
                        }
                    }
                    energy += (1.0/3.0) * spin(a) * temp;
                }
            }
        }
        
        return energy;
    }

    /**
     * Compute total energy of the SU(3) sublattice only
     * Includes SU3-SU3 interactions, SU3 field/onsite, and half of mixed interactions
     */
    double total_energy_SU3() const {
        double energy = 0.0;
        
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            const auto& spin = spins_SU3[i];
            
            // Field and onsite
            energy -= spin.dot(field_SU3[i]);
            energy += spin.dot(onsite_interaction_SU3[i] * spin);
            
            // Bilinear SU3-SU3
            for (size_t j = 0; j < bilinear_partners_SU3[i].size(); ++j) {
                const size_t partner = bilinear_partners_SU3[i][j];
                energy += 0.5 * spin.dot(bilinear_interaction_SU3[i][j] * spins_SU3[partner]);
            }
            
            // Mixed bilinear SU3-SU2 (count half for SU3)
            for (size_t j = 0; j < mixed_bilinear_partners_SU3[i].size(); ++j) {
                const size_t partner = mixed_bilinear_partners_SU3[i][j];
                energy += 0.5 * spin.dot(mixed_bilinear_interaction_SU3[i][j] * spins_SU2[partner]);
            }

            // Mixed trilinear SU3-SU2-SU2
            for (size_t j = 0; j < mixed_trilinear_partners_SU3[i].size(); ++j) {
                const size_t p1 = mixed_trilinear_partners_SU3[i][j][0];
                const size_t p2 = mixed_trilinear_partners_SU3[i][j][1];
                const auto& T = mixed_trilinear_interaction_SU3[i][j];

                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            temp += T[a](b, c) * spins_SU2[p1](b) * spins_SU2[p2](c);
                        }
                    }
                    energy += (1.0 / 3.0) * spin(a) * temp;
                }
            }
            
            // Trilinear SU3-SU3-SU3
            for (size_t j = 0; j < trilinear_partners_SU3[i].size(); ++j) {
                const size_t p1 = trilinear_partners_SU3[i][j][0];
                const size_t p2 = trilinear_partners_SU3[i][j][1];
                const auto& T = trilinear_interaction_SU3[i][j];
                
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            temp += T[a](b, c) * spins_SU3[p1](b) * spins_SU3[p2](c);
                        }
                    }
                    energy += (1.0/3.0) * spin(a) * temp;
                }
            }
        }
        
        return energy;
    }

    /**
     * Compute total energy directly from flat state array (zero-allocation version)
     * State layout: [SU2_site0_components... SU2_siteN... SU3_site0_components... SU3_siteM...]
     * Includes all interaction terms with proper double-counting avoidance
     */
    double total_energy_flat(const double* state_flat) const {
        double energy = 0.0;
        const size_t offset_SU3 = lattice_size_SU2 * spin_dim_SU2;
        
        // SU(2) contributions
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            const double* spin = &state_flat[i * spin_dim_SU2];
            
            // Field
            for (size_t d = 0; d < spin_dim_SU2; ++d) {
                energy -= spin[d] * field_SU2[i](d);
            }
            
            // Onsite
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    energy += spin[a] * onsite_interaction_SU2[i](a, b) * spin[b];
                }
            }
            
            // Bilinear (half-counted)
            for (size_t j = 0; j < bilinear_partners_SU2[i].size(); ++j) {
                const size_t partner = bilinear_partners_SU2[i][j];
                const double* partner_spin = &state_flat[partner * spin_dim_SU2];
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        energy += 0.5 * spin[a] * bilinear_interaction_SU2[i][j](a, b) * partner_spin[b];
                    }
                }
            }
            
            // Mixed bilinear (half-counted)
            for (size_t j = 0; j < mixed_bilinear_partners_SU2[i].size(); ++j) {
                const size_t partner = mixed_bilinear_partners_SU2[i][j];
                const double* partner_spin = &state_flat[offset_SU3 + partner * spin_dim_SU3];
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        energy += 0.5 * spin[a] * mixed_bilinear_interaction_SU2[i][j](a, b) * partner_spin[b];
                    }
                }
            }

            // Mixed trilinear
            for (size_t j = 0; j < mixed_trilinear_partners_SU2[i].size(); ++j) {
                const size_t p1 = mixed_trilinear_partners_SU2[i][j][0];
                const size_t p2 = mixed_trilinear_partners_SU2[i][j][1];
                const double* spin1 = &state_flat[p1 * spin_dim_SU2];
                const double* spin2 = &state_flat[offset_SU3 + p2 * spin_dim_SU3];
                const auto& T = mixed_trilinear_interaction_SU2[i][j];

                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            temp += T[a](b, c) * spin1[b] * spin2[c];
                        }
                    }
                    energy += (1.0 / 3.0) * spin[a] * temp;
                }
            }
            
            // Trilinear
            for (size_t j = 0; j < trilinear_partners_SU2[i].size(); ++j) {
                const size_t p1 = trilinear_partners_SU2[i][j][0];
                const size_t p2 = trilinear_partners_SU2[i][j][1];
                const double* spin1 = &state_flat[p1 * spin_dim_SU2];
                const double* spin2 = &state_flat[p2 * spin_dim_SU2];
                const auto& T = trilinear_interaction_SU2[i][j];
                
                for (size_t a = 0; a < spin_dim_SU2; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            temp += T[a](b, c) * spin1[b] * spin2[c];
                        }
                    }
                    energy += (1.0/3.0) * spin[a] * temp;
                }
            }
        }
        
        // SU(3) contributions
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            const double* spin = &state_flat[offset_SU3 + i * spin_dim_SU3];
            
            // Field
            for (size_t d = 0; d < spin_dim_SU3; ++d) {
                energy -= spin[d] * field_SU3[i](d);
            }
            
            // Onsite
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                for (size_t b = 0; b < spin_dim_SU3; ++b) {
                    energy += spin[a] * onsite_interaction_SU3[i](a, b) * spin[b];
                }
            }
            
            // Bilinear (half-counted)
            for (size_t j = 0; j < bilinear_partners_SU3[i].size(); ++j) {
                const size_t partner = bilinear_partners_SU3[i][j];
                const double* partner_spin = &state_flat[offset_SU3 + partner * spin_dim_SU3];
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        energy += 0.5 * spin[a] * bilinear_interaction_SU3[i][j](a, b) * partner_spin[b];
                    }
                }
            }
            
            // Mixed bilinear (half-counted)
            for (size_t j = 0; j < mixed_bilinear_partners_SU3[i].size(); ++j) {
                const size_t partner = mixed_bilinear_partners_SU3[i][j];
                const double* partner_spin = &state_flat[partner * spin_dim_SU2];
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        energy += 0.5 * spin[a] * mixed_bilinear_interaction_SU3[i][j](a, b) * partner_spin[b];
                    }
                }
            }

            // Mixed trilinear
            for (size_t j = 0; j < mixed_trilinear_partners_SU3[i].size(); ++j) {
                const size_t p1 = mixed_trilinear_partners_SU3[i][j][0];
                const size_t p2 = mixed_trilinear_partners_SU3[i][j][1];
                const double* spin1 = &state_flat[p1 * spin_dim_SU2];
                const double* spin2 = &state_flat[p2 * spin_dim_SU2];
                const auto& T = mixed_trilinear_interaction_SU3[i][j];

                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU2; ++b) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            temp += T[a](b, c) * spin1[b] * spin2[c];
                        }
                    }
                    energy += (1.0 / 3.0) * spin[a] * temp;
                }
            }
            
            // Trilinear
            for (size_t j = 0; j < trilinear_partners_SU3[i].size(); ++j) {
                const size_t p1 = trilinear_partners_SU3[i][j][0];
                const size_t p2 = trilinear_partners_SU3[i][j][1];
                const double* spin1 = &state_flat[offset_SU3 + p1 * spin_dim_SU3];
                const double* spin2 = &state_flat[offset_SU3 + p2 * spin_dim_SU3];
                const auto& T = trilinear_interaction_SU3[i][j];
                
                for (size_t a = 0; a < spin_dim_SU3; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim_SU3; ++b) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            temp += T[a](b, c) * spin1[b] * spin2[c];
                        }
                    }
                    energy += (1.0/3.0) * spin[a] * temp;
                }
            }
        }
        
        return energy;
    }

    // ============================================================
    // MONTE CARLO METHODS
    // ============================================================

    /**
     * Single Metropolis sweep over both sublattices (sequential: SU2 then SU3)
     * 
     * Optimized with:
     * - Precomputed inverse temperature
     * - Batched random number generation
     * - Branchless acceptance criterion
     */
    double metropolis(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (T <= 0) return 0.0;
        
        size_t accepted = 0;
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        const double inv_T = 1.0 / T;  // Precompute inverse temperature
        
        // Batch size for random number pre-generation
        constexpr size_t BATCH_SIZE = 64;
        vector<size_t> random_sites(BATCH_SIZE);
        vector<double> random_uniforms(BATCH_SIZE);
        
        // Sweep SU(2) sublattice
        for (size_t batch_start = 0; batch_start < lattice_size_SU2; batch_start += BATCH_SIZE) {
            const size_t batch_end = std::min(batch_start + BATCH_SIZE, lattice_size_SU2);
            const size_t current_batch_size = batch_end - batch_start;
            
            // Pre-generate random numbers for this batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                random_sites[j] = random_int_lehman(lattice_size_SU2);
                random_uniforms[j] = random_double_lehman(0, 1);
            }
            
            // Process batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                const size_t i = random_sites[j];
                const double rand_uniform = random_uniforms[j];
                
                SpinVector new_spin;
                if (gaussian_move) {
                    new_spin = spins_SU2[i] + gen_random_spin(sigma, spin_dim_SU2);
                    double norm = new_spin.norm();
                    if (norm > 1e-12) new_spin *= spin_length_SU2 / norm;
                    else new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                } else {
                    new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                }
                
                const double dE = site_energy_SU2_diff(new_spin, spins_SU2[i], i);

                // Acceptance: short-circuit for downhill moves (was bitwise `|`,
                // which wastefully evaluated exp() even when dE <= 0).
                const bool accept = (dE <= 0) || (rand_uniform < exp(-dE * inv_T));
                if (accept) {
                    spins_SU2[i] = new_spin;
                    accepted++;
                }
            }
        }

        // Sweep SU(3) sublattice
        for (size_t batch_start = 0; batch_start < lattice_size_SU3; batch_start += BATCH_SIZE) {
            const size_t batch_end = std::min(batch_start + BATCH_SIZE, lattice_size_SU3);
            const size_t current_batch_size = batch_end - batch_start;
            
            // Pre-generate random numbers for this batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                random_sites[j] = random_int_lehman(lattice_size_SU3);
                random_uniforms[j] = random_double_lehman(0, 1);
            }
            
            // Process batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                const size_t i = random_sites[j];
                const double rand_uniform = random_uniforms[j];
                
                SpinVector new_spin;
                if (gaussian_move) {
                    new_spin = spins_SU3[i] + gen_random_spin(sigma, spin_dim_SU3);
                    double norm = new_spin.norm();
                    if (norm > 1e-12) new_spin *= spin_length_SU3 / norm;
                    else new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                } else {
                    new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                }
                
                const double dE = site_energy_SU3_diff(new_spin, spins_SU3[i], i);

                // Acceptance: logical OR for short-circuit (was bitwise).
                const bool accept = (dE <= 0) || (rand_uniform < exp(-dE * inv_T));
                if (accept) {
                    spins_SU3[i] = new_spin;
                    accepted++;
                }
            }
        }

        return double(accepted) / double(total_sites);
    }

    /**
     * Coloured Metropolis sweep — parallel over SU(2) sites within each
     * SU(2) colour, then parallel over SU(3) sites within each SU(3) colour.
     *
     * Differs from `metropolis()` (random-with-replacement, coupon-collector
     * style) in that this version visits every SU(2) and every SU(3) site
     * exactly once per call, in a colour-stratified deterministic order
     * within each colour. The Markov chain is still detailed-balance
     * correct (each colour pass is a valid Metropolis sub-sweep over a
     * subset of independent single-spin moves), and the per-site sampling
     * is *better* (no missed sites).
     *
     * Race-free guarantee: within an SU(2) colour, no two sites share any
     * SU(2)-SU(2) bilinear, SU(2)-SU(2)-SU(2) trilinear, or
     * SU(2)-SU(2)-SU(3) trilinear interaction (the SU(3) read partners are
     * frozen during the SU(2) pass, hence not racy). Symmetric story for
     * SU(3). Mixed bilinear couples SU(2) ↔ SU(3) directly across the two
     * passes, and is correct because each pass treats the other species as
     * a frozen background.
     *
     * Falls back to serial `metropolis()` if no colour partition is built
     * or only one OpenMP thread is available.
     */
    double metropolis_parallel(double T, bool gaussian_move = false,
                               double sigma = 60.0) {
        if (n_colors_SU2 == 0 && n_colors_SU3 == 0)
            return metropolis(T, gaussian_move, sigma);
#ifdef _OPENMP
        if (omp_get_max_threads() <= 1) return metropolis(T, gaussian_move, sigma);
#else
        return metropolis(T, gaussian_move, sigma);
#endif
        if (T <= 0) return 0.0;

        const double inv_T = 1.0 / T;
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;

#ifdef _OPENMP
        const int n_threads = omp_get_max_threads();
#else
        const int n_threads = 1;
#endif
        // Cache-line padded per-thread counters: avoid false sharing of
        // adjacent counter slots in the final reduction.
        struct alignas(64) PaddedAccept { size_t v = 0; char pad[64 - sizeof(size_t)]; };
        std::vector<PaddedAccept> per_thread_accepted(n_threads);

        // PERSISTENT OpenMP region wrapping BOTH the SU(2) and SU(3) colour
        // passes — one fork/join per sweep, with #pragma omp barrier
        // between every colour boundary. For a TmFeO3 mixed lattice this
        // collapses (n_colors_SU2 + n_colors_SU3) team-creation costs (each
        // ~few µs) into one. At small L this was the dominant per-sweep
        // overhead for the parallel kernel.
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            size_t local_accepted = 0;

            // -------------------------- SU(2) pass --------------------------
            for (size_t c = 0; c < n_colors_SU2; ++c) {
                const size_t off_lo = sites_by_color_csr_off_SU2[c];
                const size_t off_hi = sites_by_color_csr_off_SU2[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (size_t off = off_lo; off < off_hi; ++off) {
                    const size_t i = sites_by_color_csr_SU2[off];

                    SpinVector new_spin;
                    if (gaussian_move) {
                        new_spin = spins_SU2[i] + gen_random_spin(sigma, spin_dim_SU2);
                        const double norm = new_spin.norm();
                        if (norm > 1e-12) new_spin *= spin_length_SU2 / norm;
                        else new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                    } else {
                        new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                    }

                    const double dE = site_energy_SU2_diff(new_spin, spins_SU2[i], i);
                    const double rand_uniform = random_double_lehman(0.0, 1.0);
                    const bool accept = (dE <= 0.0) ||
                                        (rand_uniform < std::exp(-dE * inv_T));
                    if (accept) {
                        spins_SU2[i] = new_spin;
                        ++local_accepted;
                    }
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }

            // -------------------------- SU(3) pass --------------------------
            for (size_t c = 0; c < n_colors_SU3; ++c) {
                const size_t off_lo = sites_by_color_csr_off_SU3[c];
                const size_t off_hi = sites_by_color_csr_off_SU3[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (size_t off = off_lo; off < off_hi; ++off) {
                    const size_t i = sites_by_color_csr_SU3[off];

                    SpinVector new_spin;
                    if (gaussian_move) {
                        new_spin = spins_SU3[i] + gen_random_spin(sigma, spin_dim_SU3);
                        const double norm = new_spin.norm();
                        if (norm > 1e-12) new_spin *= spin_length_SU3 / norm;
                        else new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                    } else {
                        new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                    }

                    const double dE = site_energy_SU3_diff(new_spin, spins_SU3[i], i);
                    const double rand_uniform = random_double_lehman(0.0, 1.0);
                    const bool accept = (dE <= 0.0) ||
                                        (rand_uniform < std::exp(-dE * inv_T));
                    if (accept) {
                        spins_SU3[i] = new_spin;
                        ++local_accepted;
                    }
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }

            per_thread_accepted[tid].v += local_accepted;
        } // end omp parallel

        size_t accepted = 0;
        for (auto& a : per_thread_accepted) accepted += a.v;
        return double(accepted) / double(total_sites);
    }

    /**
     * Coloured over-relaxation sweep. Same race-free / parallelism story as
     * `metropolis_parallel`. Falls back to serial `overrelaxation()` if the
     * colour partition is empty or only one thread is available.
     *
     * NOTE: unlike serial `overrelaxation()`, this version does *not* call
     * `get_cached_local_field_*` even if `use_field_caching` is on. The
     * field cache uses shared `field_valid_*` / `cached_local_field_*`
     * arrays that are not safe to mutate from multiple threads. The
     * cache is much less useful in a coloured sweep anyway because each
     * site's local field would typically be invalidated by neighbour
     * updates in the previous colour pass.
     */
    void overrelaxation_parallel() {
        if (n_colors_SU2 == 0 && n_colors_SU3 == 0) { overrelaxation(); return; }
#ifdef _OPENMP
        if (omp_get_max_threads() <= 1) { overrelaxation(); return; }
#else
        overrelaxation(); return;
#endif

        // PERSISTENT OpenMP region wrapping both sublattices.
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            // -------------------------- SU(2) pass --------------------------
            for (size_t c = 0; c < n_colors_SU2; ++c) {
                const size_t off_lo = sites_by_color_csr_off_SU2[c];
                const size_t off_hi = sites_by_color_csr_off_SU2[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (size_t off = off_lo; off < off_hi; ++off) {
                    const size_t i = sites_by_color_csr_SU2[off];
                    SpinVector local_field = get_local_field_SU2(i);
                    const double norm = local_field.squaredNorm();
                    if (norm > 1e-12) {
                        const double proj = 2.0 * spins_SU2[i].dot(local_field) / norm;
                        spins_SU2[i] = local_field * proj - spins_SU2[i];
                    }
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }

            // -------------------------- SU(3) pass --------------------------
            for (size_t c = 0; c < n_colors_SU3; ++c) {
                const size_t off_lo = sites_by_color_csr_off_SU3[c];
                const size_t off_hi = sites_by_color_csr_off_SU3[c + 1];
#ifdef _OPENMP
                #pragma omp for schedule(static) nowait
#endif
                for (size_t off = off_lo; off < off_hi; ++off) {
                    const size_t i = sites_by_color_csr_SU3[off];
                    SpinVector local_field = get_local_field_SU3(i);
                    const double norm = local_field.squaredNorm();
                    if (norm > 1e-12) {
                        const double proj = 2.0 * spins_SU3[i].dot(local_field) / norm;
                        spins_SU3[i] = local_field * proj - spins_SU3[i];
                    }
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }
        } // end omp parallel
    }

    /**
     * Interleaved Metropolis sweep: alternates between SU(2) and SU(3) updates
     * 
     * This improves equilibration when mixed bilinear interactions are non-zero,
     * as changes in one sublattice immediately affect the other sublattice's
     * energy landscape during the same sweep.
     * 
     * Uses local field caching with lazy invalidation for efficiency.
     * 
     * Optimized with:
     * - Precomputed inverse temperature
     * - Batched random number generation
     * - Branchless acceptance criterion
     * 
     * @param T           Temperature
     * @param gaussian_move Use Gaussian moves (true) or uniform random (false)
     * @param sigma       Width of Gaussian moves
     * @return            Acceptance rate
     */
    double metropolis_interleaved(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (T <= 0) return 0.0;
        
        size_t accepted = 0;
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        const double inv_T = 1.0 / T;  // Precompute inverse temperature
        
        // Determine whether to use caching (beneficial when mixed interactions exist)
        const bool has_mixed = (num_bi_SU2_SU3 > 0 || num_tri_SU2_SU3 > 0);
        if (has_mixed && use_field_caching) {
            // Initialize all cached fields
            init_field_cache();
        }
        
        // Batch size for random number pre-generation
        constexpr size_t BATCH_SIZE = 64;
        vector<size_t> random_sublattice(BATCH_SIZE);  // Which sublattice to update
        vector<size_t> random_sites(BATCH_SIZE);        // Site within sublattice
        vector<double> random_uniforms(BATCH_SIZE);     // For acceptance
        
        for (size_t batch_start = 0; batch_start < total_sites; batch_start += BATCH_SIZE) {
            const size_t batch_end = std::min(batch_start + BATCH_SIZE, total_sites);
            const size_t current_batch_size = batch_end - batch_start;
            
            // Pre-generate random numbers for this batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                random_sublattice[j] = random_int_lehman(total_sites);
                random_uniforms[j] = random_double_lehman(0, 1);
            }
            
            // Process batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                // Probabilistically choose which sublattice to update
                const bool update_SU2 = (random_sublattice[j] < lattice_size_SU2);
                const double rand_uniform = random_uniforms[j];
                
                if (update_SU2) {
                    const size_t i = random_int_lehman(lattice_size_SU2);
                    
                    SpinVector new_spin;
                    if (gaussian_move) {
                        new_spin = spins_SU2[i] + gen_random_spin(sigma, spin_dim_SU2);
                        double norm = new_spin.norm();
                        if (norm > 1e-12) new_spin *= spin_length_SU2 / norm;
                        else new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                    } else {
                        new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
                    }
                    
                    const double dE = site_energy_SU2_diff(new_spin, spins_SU2[i], i);

                    // Acceptance: logical OR (was bitwise).
                    const bool accept = (dE <= 0) || (rand_uniform < exp(-dE * inv_T));
                    if (accept) {
                        spins_SU2[i] = new_spin;
                        accepted++;

                        // Invalidate cached fields for affected sites
                        if (has_mixed && use_field_caching) {
                            invalidate_fields_from_SU2_update(i);
                        }
                    }
                } else {
                    const size_t i = random_int_lehman(lattice_size_SU3);
                    
                    SpinVector new_spin;
                    if (gaussian_move) {
                        new_spin = spins_SU3[i] + gen_random_spin(sigma, spin_dim_SU3);
                        double norm = new_spin.norm();
                        if (norm > 1e-12) new_spin *= spin_length_SU3 / norm;
                        else new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                    } else {
                        new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
                    }
                    
                    const double dE = site_energy_SU3_diff(new_spin, spins_SU3[i], i);

                    // Acceptance: logical OR (was bitwise).
                    const bool accept = (dE <= 0) || (rand_uniform < exp(-dE * inv_T));
                    if (accept) {
                        spins_SU3[i] = new_spin;
                        accepted++;

                        // Invalidate cached fields for affected sites
                        if (has_mixed && use_field_caching) {
                            invalidate_fields_from_SU3_update(i);
                        }
                    }
                }
            }
        }
        
        return double(accepted) / double(total_sites);
    }

    /**
     * Over-relaxation sweep (microcanonical, zero acceptance rate)
     * Reflects spins across their local field direction
     */
    void overrelaxation() {
        // Over-relaxation for SU(2) spins
        for (size_t count = 0; count < lattice_size_SU2; ++count) {
            size_t i = random_int_lehman(lattice_size_SU2);
            SpinVector local_field = get_local_field_SU2(i);
            double norm = local_field.squaredNorm();
            
            if (norm > 1e-12) {
                double proj = 2.0 * spins_SU2[i].dot(local_field) / norm;
                spins_SU2[i] = local_field * proj - spins_SU2[i];
            }
        }
        
        // Over-relaxation for SU(3) spins
        for (size_t count = 0; count < lattice_size_SU3; ++count) {
            size_t i = random_int_lehman(lattice_size_SU3);
            SpinVector local_field = get_local_field_SU3(i);
            double norm = local_field.squaredNorm();
            
            if (norm > 1e-12) {
                double proj = 2.0 * spins_SU3[i].dot(local_field) / norm;
                spins_SU3[i] = local_field * proj - spins_SU3[i];
            }
        }
    }

    /**
     * Interleaved over-relaxation sweep (microcanonical)
     * 
     * Alternates between SU(2) and SU(3) updates, ensuring that changes
     * in one sublattice are immediately reflected in the local field
     * computation of the other sublattice during the same sweep.
     * 
     * Uses local field caching with lazy invalidation for efficiency
     * when mixed interactions are present.
     */
    void overrelaxation_interleaved() {
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        const bool has_mixed = (num_bi_SU2_SU3 > 0 || num_tri_SU2_SU3 > 0);
        
        // Initialize field cache if using caching mode
        if (has_mixed && use_field_caching) {
            init_field_cache();
        }
        
        for (size_t n = 0; n < total_sites; ++n) {
            // Probabilistically choose which sublattice to update
            bool update_SU2 = (random_int_lehman(total_sites) < lattice_size_SU2);
            
            if (update_SU2) {
                size_t i = random_int_lehman(lattice_size_SU2);
                SpinVector local_field = (has_mixed && use_field_caching) ? 
                    get_cached_local_field_SU2(i) : get_local_field_SU2(i);
                double norm = local_field.squaredNorm();
                
                if (norm > 1e-12) {
                    double proj = 2.0 * spins_SU2[i].dot(local_field) / norm;
                    spins_SU2[i] = local_field * proj - spins_SU2[i];
                    
                    // Invalidate affected fields
                    if (has_mixed && use_field_caching) {
                        invalidate_fields_from_SU2_update(i);
                    }
                }
            } else {
                size_t i = random_int_lehman(lattice_size_SU3);
                SpinVector local_field = (has_mixed && use_field_caching) ?
                    get_cached_local_field_SU3(i) : get_local_field_SU3(i);
                double norm = local_field.squaredNorm();
                
                if (norm > 1e-12) {
                    double proj = 2.0 * spins_SU3[i].dot(local_field) / norm;
                    spins_SU3[i] = local_field * proj - spins_SU3[i];
                    
                    // Invalidate affected fields
                    if (has_mixed && use_field_caching) {
                        invalidate_fields_from_SU3_update(i);
                    }
                }
            }
        }
    }

    /**
     * Interleaved deterministic sweep with caching
     * 
     * Zero-temperature relaxation that alternates between sublattices
     * and uses local field caching for efficiency.
     */
    void deterministic_sweep_interleaved() {
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        const bool has_mixed = (num_bi_SU2_SU3 > 0 || num_tri_SU2_SU3 > 0);
        
        // Initialize field cache if using caching mode
        if (has_mixed && use_field_caching) {
            init_field_cache();
        }
        
        for (size_t n = 0; n < total_sites; ++n) {
            // Probabilistically choose which sublattice to update
            bool update_SU2 = (random_int_lehman(total_sites) < lattice_size_SU2);
            
            if (update_SU2) {
                size_t i = random_int_lehman(lattice_size_SU2);
                SpinVector local_field = (has_mixed && use_field_caching) ? 
                    get_cached_local_field_SU2(i) : get_local_field_SU2(i);
                double norm = local_field.norm();
                
                if (norm > 1e-12) {
                    spins_SU2[i] = -local_field / norm * spin_length_SU2;
                    
                    // Invalidate affected fields
                    if (has_mixed && use_field_caching) {
                        invalidate_fields_from_SU2_update(i);
                    }
                }
            } else {
                size_t i = random_int_lehman(lattice_size_SU3);
                SpinVector local_field = (has_mixed && use_field_caching) ?
                    get_cached_local_field_SU3(i) : get_local_field_SU3(i);
                double norm = local_field.norm();
                
                if (norm > 1e-12) {
                    spins_SU3[i] = -local_field / norm * spin_length_SU3;
                    
                    // Invalidate affected fields
                    if (has_mixed && use_field_caching) {
                        invalidate_fields_from_SU3_update(i);
                    }
                }
            }
        }
    }

    /**
     * Deterministic sweep: align each spin antiparallel to its local field
     * This is a zero-temperature relaxation step that randomly selects sites
     */
    void deterministic_sweep() {
        // Deterministic update for SU(2) spins
        for (size_t count = 0; count < lattice_size_SU2; ++count) {
            size_t i = random_int_lehman(lattice_size_SU2);
            SpinVector local_field = get_local_field_SU2(i);
            double norm = local_field.norm();
            
            if (norm > 1e-12) {
                spins_SU2[i] = -local_field / norm * spin_length_SU2;
            }
        }
        
        // Deterministic update for SU(3) spins
        for (size_t count = 0; count < lattice_size_SU3; ++count) {
            size_t i = random_int_lehman(lattice_size_SU3);
            SpinVector local_field = get_local_field_SU3(i);
            double norm = local_field.norm();
            
            if (norm > 1e-12) {
                spins_SU3[i] = -local_field / norm * spin_length_SU3;
            }
        }
    }

    /**
     * Zero-temperature greedy quench with convergence check
     */
    void greedy_quench(double rel_tol = 1e-12, size_t max_sweeps = 10000);

    /**
     * Main simulated annealing routine
     * Matches structure and features from Lattice::simulated_annealing
     */
    void simulated_annealing(double T_start, double T_end, size_t n_anneal,
                            bool gaussian_move = false,
                            double cooling_rate = 0.9,
                            string out_dir = "",
                            bool save_observables = false,
                            bool T_zero = false,
                            size_t n_deterministics = 1000,
                            size_t twist_sweep_count = 100);

    /**
     * Perform detailed measurements at final temperature
     * Computes: energy, specific heat, sublattice magnetizations (SU2 and SU3), 
     * and cross-correlations. All with binning analysis for error estimation.
     */
    void perform_final_measurements(double T_final, double sigma, bool gaussian_move,
                                   const string& out_dir);

    /** Compute autocorrelation — delegates to mc::compute_autocorrelation */
    AutocorrelationResult compute_autocorrelation(const vector<double>& energies, 
                                                   size_t base_interval = 10);

    // ============================================================
    // BINNING ANALYSIS (delegated to mc::* functions)
    // ============================================================

    /** Binning analysis — delegates to mc::binning_analysis */
    static BinningResult binning_analysis(const vector<double>& data) {
        return mc::binning_analysis(data);
    }

    /** Component-wise binning analysis for vector observable — delegates to mc::binning_analysis_vector */
    static vector<BinningResult> binning_analysis_vector(const vector<SpinVector>& data) {
        return mc::binning_analysis_vector<SpinVector>(data);
    }

    // ============================================================
    // SUBLATTICE MAGNETIZATION
    // ============================================================

    /**
     * Compute magnetization for each SU(2) sublattice separately
     * 
     * @return Vector of SpinVectors, one per SU(2) sublattice (N_atoms_SU2 sublattices)
     */
    vector<SpinVector> magnetization_sublattice_SU2() const {
        vector<SpinVector> M_sub(N_atoms_SU2);
        size_t n_cells = dim1 * dim2 * dim3;
        
        for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
            M_sub[atom] = SpinVector::Zero(spin_dim_SU2);
        }
        
        // Sum over all unit cells for each sublattice
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
                        size_t site_idx = flatten_index(i, j, k, atom, N_atoms_SU2);
                        
                        // Transform to global frame using sublattice frame
                        SpinVector spin_global = SpinVector::Zero(spin_dim_SU2);
                        for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
                            for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
                                spin_global(mu) += sublattice_frames_SU2[atom](nu, mu) * spins_SU2[site_idx](nu);
                            }
                        }
                        M_sub[atom] += spin_global;
                    }
                }
            }
        }
        
        // Normalize by number of unit cells
        for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
            M_sub[atom] /= double(n_cells);
        }
        
        return M_sub;
    }

    /**
     * Compute magnetization for each SU(3) sublattice separately
     * 
     * @return Vector of SpinVectors, one per SU(3) sublattice (N_atoms_SU3 sublattices)
     */
    vector<SpinVector> magnetization_sublattice_SU3() const {
        vector<SpinVector> M_sub(N_atoms_SU3);
        size_t n_cells = dim1 * dim2 * dim3;
        
        for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
            M_sub[atom] = SpinVector::Zero(spin_dim_SU3);
        }
        
        // Sum over all unit cells for each sublattice
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
                        size_t site_idx = flatten_index(i, j, k, atom, N_atoms_SU3);
                        
                        // Transform to global frame using sublattice frame
                        SpinVector spin_global = SpinVector::Zero(spin_dim_SU3);
                        for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
                            for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
                                spin_global(mu) += sublattice_frames_SU3[atom](nu, mu) * spins_SU3[site_idx](nu);
                            }
                        }
                        M_sub[atom] += spin_global;
                    }
                }
            }
        }
        
        // Normalize by number of unit cells
        for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
            M_sub[atom] /= double(n_cells);
        }
        
        return M_sub;
    }

    // ============================================================
    // COMPREHENSIVE OBSERVABLE COLLECTION
    // ============================================================

    /**
     * Collect a single measurement of all thermodynamic observables
     * Returns: (energy, energy_SU2, energy_SU3, sublattice_mags_SU2, sublattice_mags_SU3)
     */
    struct MixedMeasurement {
        double energy;
        double energy_SU2;
        double energy_SU3;
        vector<SpinVector> sublattice_mags_SU2;
        vector<SpinVector> sublattice_mags_SU3;
    };

    MixedMeasurement measure_all_observables() const {
        MixedMeasurement m;
        m.energy = total_energy();
        m.energy_SU2 = total_energy_SU2();
        m.energy_SU3 = total_energy_SU3();
        m.sublattice_mags_SU2 = magnetization_sublattice_SU2();
        m.sublattice_mags_SU3 = magnetization_sublattice_SU3();
        return m;
    }

    /**
     * Compute comprehensive thermodynamic observables with binning error analysis
     * 
     * @param measurements Vector of MixedMeasurement from MC sampling
     * @param T Temperature
     * @return MixedThermodynamicObservables struct with all observables and uncertainties
     */
    MixedThermodynamicObservables compute_thermodynamic_observables(
        const vector<MixedMeasurement>& measurements,
        double T) const;

    /**
     * Save comprehensive thermodynamic observables to files
     */
    void save_thermodynamic_observables(const string& out_dir,
                                         const MixedThermodynamicObservables& obs) const;

    /**
     * Print thermodynamic observables summary to stdout
     */
    void print_thermodynamic_observables(const MixedThermodynamicObservables& obs) const;

    /**
     * Save thermodynamic observables to HDF5 format for mixed lattice
     * Single file per rank with all data organized in groups
     */
    void save_thermodynamic_observables_hdf5(const string& out_dir,
                                              const MixedThermodynamicObservables& obs,
                                              const vector<double>& energies,
                                              const vector<pair<SpinVector, SpinVector>>& magnetizations,
                                              const vector<MixedMeasurement>& measurements,
                                              size_t n_anneal,
                                              size_t n_measure,
                                              size_t probe_rate,
                                              size_t swap_rate,
                                              size_t overrelaxation_rate,
                                              double acceptance_rate,
                                              double swap_acceptance_rate) const;

    /**
     * Save aggregated heat capacity data from all temperatures to HDF5 format
     * Called by rank 0 to save temperature-dependent thermodynamic data
     */
    void save_heat_capacity_hdf5(const string& out_dir,
                                  const vector<double>& temperatures,
                                  const vector<double>& heat_capacity,
                                  const vector<double>& dHeat) const;

    /**
     * Save sublattice magnetization time series to files
     */
    void save_sublattice_magnetization_timeseries(const string& out_dir,
                                                   const vector<MixedMeasurement>& measurements) const;

    /**
     * Compute and save thermodynamic observables for mixed lattice
     */
    void compute_and_save_observables(const vector<double>& energies,
                                     const vector<pair<SpinVector, SpinVector>>& magnetizations,
                                     double T, const string& out_dir);

    /**
     * Save observables for mixed lattice
     */
    void save_observables(const string& dir_path,
                         const vector<double>& energies,
                         const vector<pair<SpinVector, SpinVector>>& magnetizations);

    /**
     * Save autocorrelation results
     */
    void save_autocorrelation_results(const string& out_dir, 
                                     const AutocorrelationResult& acf);

    // ============================================================
    // TEMPERATURE LADDER OPTIMIZATION
    // Based on:
    //   Katzgraber et al., Phys. Rev. E 73, 056702 (2006) - feedback temperature placement
    //   Bittner et al., Phys. Rev. Lett. 101, 130603 (2008) - adaptive sweep schedule
    // ============================================================

    /**
     * Generate optimized temperature grid for parallel tempering (MixedLattice version)
     * 
     * Phase 1 (Katzgraber): Feedback-optimized temperature placement with
     * rule d_beta_i proportional to A_i (current fraction). This achieves uniform
     * acceptance rates (~50%) across all temperature pairs.
     * 
     * Phase 2 (Bittner): Measures autocorrelation time tau_int(T) at each
     * temperature and sets n_sweeps(T_i) = n_base * tau_int(T_i) / min(tau_int).
     * This minimizes round-trip time by decorrelating at bottleneck temperatures.
     * 
     * @param Tmin              Minimum (coldest) temperature
     * @param Tmax              Maximum (hottest) temperature  
     * @param R                 Number of replicas (temperatures)
     * @param warmup_sweeps     MC sweeps for initial equilibration per replica
     * @param sweeps_per_iter   MC sweeps per feedback iteration
     * @param feedback_iters    Number of feedback optimization iterations
     * @param gaussian_move     Use Gaussian moves (true) or uniform (false)
     * @param overrelaxation_rate  Apply overrelaxation every N sweeps (0 = disabled)
     * @param target_acceptance Target acceptance rate (default: 0.5 per Katzgraber)
     * @param convergence_tol   Convergence tolerance for acceptance rate uniformity
     * @return OptimizedTempGridResult containing temperatures, diagnostics, and adaptive sweep schedule
     */
    OptimizedTempGridResult generate_optimized_temperature_grid(
        double Tmin, double Tmax, size_t R,
        size_t warmup_sweeps = 500,
        size_t sweeps_per_iter = 500,
        size_t feedback_iters = 20,
        bool gaussian_move = false,
        size_t overrelaxation_rate = 0,
        double target_acceptance = 0.5,
        double convergence_tol = 0.05) {
        
        OptimizedTempGridResult result;
        result.converged = false;
        result.feedback_iterations_used = 0;
        
        if (R < 2) {
            result.temperatures = {Tmin};
            if (R == 1) return result;
        }
        if (R == 2) {
            result.temperatures = {Tmin, Tmax};
            result.acceptance_rates = {0.5};
            result.converged = true;
            return result;
        }
        
        cout << "=== Feedback-Optimized Temperature Grid (MixedLattice) ===" << endl;
        cout << "References: Katzgraber et al. PRE 73, 056702 (2006)" << endl;
        cout << "            Bittner et al. PRL 101, 130603 (2008)" << endl;
        cout << "T_min = " << Tmin << ", T_max = " << Tmax << ", R = " << R << endl;
        cout << "Target acceptance rate: " << target_acceptance * 100 << "%" << endl;
        
        // Helper: linear spacing
        auto linspace = [](double a, double b, size_t n) {
            vector<double> result(n);
            for (size_t i = 0; i < n; ++i) {
                result[i] = a + (b - a) * double(i) / double(n - 1);
            }
            return result;
        };
        
        // Helper: convert beta to temperature
        auto temps_from_beta = [](const vector<double>& b) {
            vector<double> T(b.size());
            for (size_t i = 0; i < b.size(); ++i) {
                T[i] = 1.0 / b[i];
            }
            return T;
        };
        
        // Initialize with geometric spacing in temperature
        double beta_min = 1.0 / Tmax;  // Hottest = smallest beta
        double beta_max = 1.0 / Tmin;  // Coldest = largest beta
        vector<double> beta = linspace(beta_min, beta_max, R);
        
        // Store original spins
        SpinConfigSU2 original_spins_SU2 = spins_SU2;
        SpinConfigSU3 original_spins_SU3 = spins_SU3;
        
        // Initialize replicas at each temperature
        vector<SpinConfigSU2> reps_SU2(R, spins_SU2);
        vector<SpinConfigSU3> reps_SU3(R, spins_SU3);
        double sigma = 1000.0;
        
        // OPTIMIZATION 1: Cache energies to avoid redundant calculations
        vector<double> cached_energies(R);
        
        // Warmup phase: equilibrate each replica at its temperature
        cout << "Warming up " << R << " replicas..." << endl;
        for (size_t k = 0; k < R; ++k) {
            spins_SU2 = reps_SU2[k];
            spins_SU3 = reps_SU3[k];
            double T_k = 1.0 / beta[k];
            for (size_t i = 0; i < warmup_sweeps; ++i) {
                metropolis_interleaved(T_k, gaussian_move, sigma);
                if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                    overrelaxation();
                }
            }
            reps_SU2[k] = spins_SU2;
            reps_SU3[k] = spins_SU3;
            cached_energies[k] = total_energy();
        }
        
        // Feedback optimization loop
        vector<double> acceptance_rates(R - 1, 0.0);
        
        // OPTIMIZATION 2: Adaptive damping - start aggressive, become conservative
        double base_damping = 0.3;
        
        for (size_t iter = 0; iter < feedback_iters; ++iter) {
            // Adaptive damping
            double damping = base_damping + 0.4 * (double(iter) / double(feedback_iters));
            
            vector<size_t> attempts(R - 1, 0);
            vector<size_t> accepts(R - 1, 0);
            
            // OPTIMIZATION 3: Reduced sweeps for early iterations
            size_t effective_sweeps = sweeps_per_iter;
            if (iter < 3) {
                effective_sweeps = std::max(size_t(50), sweeps_per_iter / 4);
            } else if (iter < 6) {
                effective_sweeps = std::max(size_t(100), sweeps_per_iter / 2);
            }
            
            // Run MC sweeps and measure acceptance rates
            for (size_t sweep = 0; sweep < effective_sweeps; ++sweep) {
                // Update each replica
                for (size_t k = 0; k < R; ++k) {
                    spins_SU2 = reps_SU2[k];
                    spins_SU3 = reps_SU3[k];
                    double T_k = 1.0 / beta[k];
                    metropolis_interleaved(T_k, gaussian_move, sigma);
                    if (overrelaxation_rate > 0 && sweep % overrelaxation_rate == 0) {
                        overrelaxation();
                    }
                    reps_SU2[k] = spins_SU2;
                    reps_SU3[k] = spins_SU3;
                    cached_energies[k] = total_energy();
                }
                
                // Attempt replica exchanges for ALL adjacent pairs
                for (int parity = 0; parity <= 1; ++parity) {
                    for (size_t e = parity; e < R - 1; e += 2) {
                        // Use cached energies
                        // beta array is sorted: beta[0] = beta_min (hottest), beta[R-1] = beta_max (coldest)
                        // So beta[e] < beta[e+1] (e is hotter, e+1 is colder)
                        double E_hot = cached_energies[e];
                        double E_cold = cached_energies[e + 1];
                        
                        // Metropolis criterion for replica exchange:
                        // P_swap = min(1, exp(Δ)) where Δ = (β_cold - β_hot)(E_cold - E_hot)
                        // When energies are properly ordered (E_cold < E_hot), Δ < 0
                        double dBeta = beta[e + 1] - beta[e];  // > 0 (β_cold - β_hot)
                        double dE = E_cold - E_hot;            // typically < 0 (E_cold - E_hot)
                        double delta = dBeta * dE;             // typically < 0
                        
                        ++attempts[e];
                        if (delta >= 0 || random_double_lehman(0.0, 1.0) < std::exp(delta)) {
                            ++accepts[e];
                            std::swap(reps_SU2[e], reps_SU2[e + 1]);
                            std::swap(reps_SU3[e], reps_SU3[e + 1]);
                            std::swap(cached_energies[e], cached_energies[e + 1]);
                        }
                    }
                }
            }
            
            // Compute acceptance rates
            for (size_t e = 0; e < R - 1; ++e) {
                acceptance_rates[e] = double(accepts[e]) / double(attempts[e]);
            }
            
            // Check convergence
            double max_deviation = 0.0;
            double mean_rate = 0.0;
            for (size_t e = 0; e < R - 1; ++e) {
                max_deviation = std::max(max_deviation, std::abs(acceptance_rates[e] - target_acceptance));
                mean_rate += acceptance_rates[e];
            }
            mean_rate /= (R - 1);
            
            cout << "Iter " << iter + 1 << "/" << feedback_iters 
                 << ": mean A = " << std::fixed << std::setprecision(3) << mean_rate
                 << ", max dev = " << max_deviation 
                 << " (sweeps=" << effective_sweeps << ", damp=" << std::setprecision(2) << damping << ")" << endl;
            
            result.feedback_iterations_used = iter + 1;
            
            if (max_deviation < convergence_tol) {
                result.converged = true;
                cout << "Converged at iteration " << iter + 1 << endl;
                break;
            }
            
            // Katzgraber feedback: d_beta_i proportional to A_i (current fraction)
            // Construct new beta positions from cumulative weights
            vector<double> weights(R - 1);
            double total_weight = 0.0;
            for (size_t e = 0; e < R - 1; ++e) {
                double A_e = acceptance_rates[e];
                if (A_e < 0.01) A_e = 0.01;
                if (A_e > 0.99) A_e = 0.99;
                weights[e] = A_e;  // Weight proportional to A
                total_weight += weights[e];
            }
            for (size_t e = 0; e < R - 1; ++e) weights[e] /= total_weight;
            
            vector<double> new_beta(R);
            new_beta[0] = beta_min;
            double cumulative = 0.0;
            for (size_t e = 0; e < R - 1; ++e) {
                cumulative += weights[e];
                new_beta[e + 1] = beta_min + cumulative * (beta_max - beta_min);
            }
            new_beta[R - 1] = beta_max;
            
            // Apply damping
            for (size_t k = 1; k < R - 1; ++k) {
                new_beta[k] = (1.0 - damping) * beta[k] + damping * new_beta[k];
            }
            beta = new_beta;
        }
        
        // ================================================================
        // PHASE 2: Bittner et al. adaptive sweep schedule
        // Measure tau_int(T) at each temperature and set
        // n_sweeps(T_i) = n_base * tau_int(T_i) / min(tau_int)
        // ================================================================
        cout << "\nMeasuring autocorrelation times for adaptive sweep schedule..." << endl;
        
        size_t tau_samples = std::max(size_t(500), sweeps_per_iter);
        result.autocorrelation_times.resize(R);
        result.sweeps_per_temp.resize(R);
        
        for (size_t k = 0; k < R; ++k) {
            spins_SU2 = reps_SU2[k];
            spins_SU3 = reps_SU3[k];
            double T_k = 1.0 / beta[k];
            
            vector<double> energy_series;
            energy_series.reserve(tau_samples);
            for (size_t i = 0; i < tau_samples; ++i) {
                metropolis_interleaved(T_k, gaussian_move, sigma);
                if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                    overrelaxation();
                }
                energy_series.push_back(total_energy());
            }
            
            AutocorrelationResult acf = compute_autocorrelation(energy_series, 1);
            result.autocorrelation_times[k] = std::max(1.0, acf.tau_int);
        }
        
        double tau_min_val = *std::min_element(result.autocorrelation_times.begin(), 
                                                result.autocorrelation_times.end());
        size_t n_base = 10;
        for (size_t k = 0; k < R; ++k) {
            result.sweeps_per_temp[k] = std::max(size_t(1),
                static_cast<size_t>(std::ceil(n_base * result.autocorrelation_times[k] / tau_min_val)));
        }
        
        cout << "Autocorrelation times and sweep schedule:" << endl;
        for (size_t k = 0; k < std::min(R, size_t(15)); ++k) {
            cout << "  T[" << k << "] = " << std::scientific << std::setprecision(4) 
                 << 1.0 / beta[k] << "  tau_int = " << std::fixed << std::setprecision(1)
                 << result.autocorrelation_times[k] << "  n_sweeps = " << result.sweeps_per_temp[k] << endl;
        }
        if (R > 15) cout << "  ... (" << R - 15 << " more)" << endl;
        
        // Restore original spins
        spins_SU2 = original_spins_SU2;
        spins_SU3 = original_spins_SU3;
        
        // Build result
        result.temperatures = temps_from_beta(beta);
        std::sort(result.temperatures.begin(), result.temperatures.end());
        result.acceptance_rates = acceptance_rates;
        
        result.local_diffusivities.resize(R - 1);
        for (size_t e = 0; e < R - 1; ++e) {
            double A = acceptance_rates[e];
            result.local_diffusivities[e] = A * (1.0 - A);
        }
        
        result.mean_acceptance_rate = 0.0;
        for (double A : acceptance_rates) result.mean_acceptance_rate += A;
        result.mean_acceptance_rate /= (R - 1);
        
        // Compute round-trip time with Bittner sweep weighting:
        // tau_rt ~ sum_i n_avg_i / f_i  where f_i = A_i * d_beta_i / sum(A_j * d_beta_j)
        double total_current = 0.0;
        for (size_t e = 0; e < R - 1; ++e) {
            double d_beta = std::abs(beta[e + 1] - beta[e]);
            double A = std::max(acceptance_rates[e], 1e-6);
            total_current += A * d_beta;
        }
        double sum_inv_f = 0.0;
        for (size_t e = 0; e < R - 1; ++e) {
            double d_beta = std::abs(beta[e + 1] - beta[e]);
            double A = std::max(acceptance_rates[e], 1e-6);
            double f_i = A * d_beta / total_current;
            double n_avg = 0.5 * (result.sweeps_per_temp[e] + result.sweeps_per_temp[e + 1]);
            sum_inv_f += n_avg / f_i;
        }
        result.round_trip_estimate = sum_inv_f;
        
        // Print summary
        cout << "\n=== Optimized Temperature Grid Summary ===" << endl;
        cout << "Temperatures (ascending):" << endl;
        for (size_t k = 0; k < std::min(R, size_t(15)); ++k) {
            cout << "  T[" << k << "] = " << std::scientific << std::setprecision(6) 
                 << result.temperatures[k];
            if (k < R - 1) {
                cout << "  (A = " << std::fixed << std::setprecision(3) 
                     << acceptance_rates[k] << ")";
            }
            cout << endl;
        }
        if (R > 15) cout << "  ... (" << R - 15 << " more)" << endl;
        
        cout << "\nMean acceptance rate: " << std::fixed << std::setprecision(3) 
             << result.mean_acceptance_rate * 100 << "%" << endl;
        cout << "Estimated round-trip time scale: " << std::scientific 
             << result.round_trip_estimate << endl;
        cout << "Converged: " << (result.converged ? "YES" : "NO") << endl;
        
        return result;
    }

    // ============================================================
    // PARALLEL TEMPERING

    /**
     * MPI-distributed temperature grid optimization for MixedLattice (FAST VERSION)
     * 
     * Each MPI rank handles one replica. Much faster than serial version.
     * References:
     *   Katzgraber et al., Phys. Rev. E 73, 056702 (2006) - feedback temperature placement
     *   Bittner et al., Phys. Rev. Lett. 101, 130603 (2008) - adaptive sweep schedule
     */
    OptimizedTempGridResult generate_optimized_temperature_grid_mpi(
        double Tmin, double Tmax,
        size_t warmup_sweeps = 500,
        size_t sweeps_per_iter = 500,
        size_t feedback_iters = 20,
        bool gaussian_move = false,
        size_t overrelaxation_rate = 0,
        double target_acceptance = 0.45,
        double convergence_tol = 0.05,
        MPI_Comm comm = MPI_COMM_WORLD,
        bool use_gradient = true) {
        
        int rank, R;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &R);
        
        OptimizedTempGridResult result;
        result.converged = false;
        result.feedback_iterations_used = 0;
        
        if (R < 2) {
            result.temperatures = {Tmin};
            result.converged = true;
            return result;
        }
        if (R == 2) {
            result.temperatures = {Tmin, Tmax};
            result.acceptance_rates = {0.5};
            result.converged = true;
            return result;
        }
        // MixedLattice has a different interface (SU2+SU3, no single spins/lattice_size),
        // so we always use the inlined Katzgraber feedback; gradient optimizer not applied here.
        (void)use_gradient;
        // Deterministic per-rank seeding (replaces wall-clock).
        seed_lehman_from_rank(static_cast<unsigned long long>(rank) + 1ULL);
        
        if (rank == 0) {
            cout << "=== Feedback-Optimized Temperature Grid (MixedLattice MPI) ===" << endl;
            cout << "References: Katzgraber et al. PRE 73, 056702 (2006)" << endl;
            cout << "            Bittner et al. PRL 101, 130603 (2008)" << endl;
            cout << "Target acceptance rate: " << target_acceptance * 100 << "%" << endl;
        }
        
        // Initialize beta array
        double beta_min = 1.0 / Tmax;
        double beta_max = 1.0 / Tmin;
        vector<double> beta(R);
        for (int i = 0; i < R; ++i) {
            beta[i] = beta_min + (beta_max - beta_min) * double(i) / double(R - 1);
        }
        
        double my_beta = beta[rank];
        double my_T = 1.0 / my_beta;
        double sigma = 1000.0;
        
        // Warmup
        if (rank == 0) cout << "Warming up replicas..." << endl;
        for (size_t i = 0; i < warmup_sweeps; ++i) {
            metropolis_interleaved(my_T, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                overrelaxation();
            }
        }
        MPI_Barrier(comm);
        
        vector<double> acceptance_rates(R - 1, 0.0);
        double base_damping = 0.5;  // Less aggressive for faster convergence
        
        for (size_t iter = 0; iter < feedback_iters; ++iter) {
            double damping = base_damping + 0.3 * (double(iter) / double(feedback_iters));
            
            int local_attempts = 0;
            int local_accepts = 0;
            
            // Use full sweeps for better statistics
            size_t effective_sweeps = sweeps_per_iter;
            
            for (size_t sweep = 0; sweep < effective_sweeps; ++sweep) {
                metropolis_interleaved(my_T, gaussian_move, sigma);
                if (overrelaxation_rate > 0 && sweep % overrelaxation_rate == 0) {
                    overrelaxation();
                }
                
                // Replica exchanges
                for (int parity = 0; parity <= 1; ++parity) {
                    int partner_rank;
                    if (parity == 0) {
                        partner_rank = (rank % 2 == 0) ? rank + 1 : rank - 1;
                    } else {
                        partner_rank = (rank % 2 == 1) ? rank + 1 : rank - 1;
                    }
                    
                    if (partner_rank < 0 || partner_rank >= R) continue;
                    
                    double my_E = total_energy();
                    double partner_E;
                    MPI_Sendrecv(&my_E, 1, MPI_DOUBLE, partner_rank, 0,
                                &partner_E, 1, MPI_DOUBLE, partner_rank, 0,
                                comm, MPI_STATUS_IGNORE);
                    
                    int accept_int = 0;
                    if (rank < partner_rank) {
                        double beta_hot = my_beta;
                        double beta_cold = beta[partner_rank];
                        double E_hot = my_E;
                        double E_cold = partner_E;
                        double delta = (beta_cold - beta_hot) * (E_hot - E_cold);
                        bool accept = (delta >= 0) || (random_double_lehman(0.0, 1.0) < std::exp(delta));
                        accept_int = accept ? 1 : 0;
                        ++local_attempts;
                        if (accept) ++local_accepts;
                    }
                    
                    int recv_accept_int = 0;
                    MPI_Sendrecv(&accept_int, 1, MPI_INT, partner_rank, 1,
                                &recv_accept_int, 1, MPI_INT, partner_rank, 1,
                                comm, MPI_STATUS_IGNORE);
                    
                    bool accept = (rank < partner_rank) ? (accept_int == 1) : (recv_accept_int == 1);
                    
                    if (accept) {
                        // Exchange SU2 spins (persistent buffer + Sendrecv_replace)
                        size_t su2_size = spins_SU2.size() * 3;
                        thread_local vector<double> su2_buf;
                        if (su2_buf.size() < su2_size) su2_buf.resize(su2_size);
                        for (size_t i = 0; i < spins_SU2.size(); ++i) {
                            for (size_t j = 0; j < 3; ++j) {
                                su2_buf[i * 3 + j] = spins_SU2[i](j);
                            }
                        }
                        MPI_Sendrecv_replace(su2_buf.data(), su2_size, MPI_DOUBLE,
                                             partner_rank, 2, partner_rank, 2,
                                             comm, MPI_STATUS_IGNORE);
                        for (size_t i = 0; i < spins_SU2.size(); ++i) {
                            for (size_t j = 0; j < 3; ++j) {
                                spins_SU2[i](j) = su2_buf[i * 3 + j];
                            }
                        }

                        // Exchange SU3 spins (persistent buffer + Sendrecv_replace)
                        size_t su3_size = spins_SU3.size() * 8;
                        thread_local vector<double> su3_buf;
                        if (su3_buf.size() < su3_size) su3_buf.resize(su3_size);
                        for (size_t i = 0; i < spins_SU3.size(); ++i) {
                            for (size_t j = 0; j < 8; ++j) {
                                su3_buf[i * 8 + j] = spins_SU3[i](j);
                            }
                        }
                        MPI_Sendrecv_replace(su3_buf.data(), su3_size, MPI_DOUBLE,
                                             partner_rank, 3, partner_rank, 3,
                                             comm, MPI_STATUS_IGNORE);
                        for (size_t i = 0; i < spins_SU3.size(); ++i) {
                            for (size_t j = 0; j < 8; ++j) {
                                spins_SU3[i](j) = su3_buf[i * 8 + j];
                            }
                        }
                    }
                }
            }
            
            // Gather statistics using MPI_Gather (cleaner and safer than Send/Recv)
            // Each rank k sends stats for edge k (if k < R-1)
            int my_attempts = (rank < R - 1) ? local_attempts : 0;
            int my_accepts = (rank < R - 1) ? local_accepts : 0;
            
            vector<int> recv_attempts(R);
            vector<int> recv_accepts(R);
            MPI_Gather(&my_attempts, 1, MPI_INT, recv_attempts.data(), 1, MPI_INT, 0, comm);
            MPI_Gather(&my_accepts, 1, MPI_INT, recv_accepts.data(), 1, MPI_INT, 0, comm);
            
            bool converged = false;
            if (rank == 0) {
                for (int e = 0; e < R - 1; ++e) {
                    if (recv_attempts[e] > 0) {
                        acceptance_rates[e] = double(recv_accepts[e]) / double(recv_attempts[e]);
                    }
                }
                
                // Check convergence using mean deviation (more stable than max)
                double max_deviation = 0.0;
                double mean_deviation = 0.0;
                double mean_rate = 0.0;
                double min_rate = 1.0, max_rate = 0.0;
                for (int e = 0; e < R - 1; ++e) {
                    double dev = std::abs(acceptance_rates[e] - target_acceptance);
                    max_deviation = std::max(max_deviation, dev);
                    mean_deviation += dev;
                    mean_rate += acceptance_rates[e];
                    min_rate = std::min(min_rate, acceptance_rates[e]);
                    max_rate = std::max(max_rate, acceptance_rates[e]);
                }
                mean_rate /= (R - 1);
                mean_deviation /= (R - 1);
                
                cout << "Iter " << iter + 1 << "/" << feedback_iters 
                     << ": mean A = " << std::fixed << std::setprecision(3) << mean_rate
                     << " [" << min_rate << ", " << max_rate << "]"
                     << ", mean dev = " << mean_deviation << endl;
                
                // Warn if acceptance is uniformly low - need more replicas
                if (mean_rate < 0.1 && iter == 0) {
                    cout << "WARNING: Mean acceptance rate is very low (" << mean_rate << ").\n"
                         << "         Consider using more replicas or a smaller temperature range." << endl;
                }
                
                result.feedback_iterations_used = iter + 1;
                
                // Converge based on mean deviation (less sensitive to noise)
                if (mean_deviation < convergence_tol) {
                    converged = true;
                    cout << "Converged at iteration " << iter + 1 << endl;
                }
                
                if (!converged) {
                    // Bittner et al. feedback: Δβ_i ∝ A_i
                    // High A → more spacing, Low A → less spacing (closer temps)
                    
                    vector<double> weights(R - 1);
                    double total_weight = 0.0;
                    
                    for (int e = 0; e < R - 1; ++e) {
                        double A_e = acceptance_rates[e];
                        if (A_e < 0.01) A_e = 0.01;
                        if (A_e > 0.99) A_e = 0.99;
                        weights[e] = A_e;  // Weight proportional to A
                        total_weight += weights[e];
                    }
                    
                    for (int e = 0; e < R - 1; ++e) {
                        weights[e] /= total_weight;
                    }
                    
                    vector<double> new_beta(R);
                    new_beta[0] = beta_min;
                    double cumulative = 0.0;
                    for (int e = 0; e < R - 1; ++e) {
                        cumulative += weights[e];
                        new_beta[e + 1] = beta_min + cumulative * (beta_max - beta_min);
                    }
                    new_beta[R - 1] = beta_max;
                    
                    // Damping
                    for (int k = 1; k < R - 1; ++k) {
                        new_beta[k] = (1.0 - damping) * beta[k] + damping * new_beta[k];
                    }
                    
                    beta = new_beta;
                }
            }
            
            int conv_int = converged ? 1 : 0;
            MPI_Bcast(&conv_int, 1, MPI_INT, 0, comm);
            MPI_Bcast(beta.data(), R, MPI_DOUBLE, 0, comm);
            
            my_beta = beta[rank];
            my_T = 1.0 / my_beta;
            
            if (conv_int == 1) {
                result.converged = true;
                break;
            }
        }
        
        MPI_Bcast(acceptance_rates.data(), R - 1, MPI_DOUBLE, 0, comm);
        
        // ================================================================
        // PHASE 2: Bittner et al. adaptive sweep schedule
        // Each rank measures tau_int(T) at its temperature, then we gather
        // to build the temperature-dependent sweep schedule.
        // ================================================================
        if (rank == 0) {
            cout << "\nMeasuring autocorrelation times for adaptive sweep schedule..." << endl;
        }
        
        size_t tau_samples = std::max(size_t(500), sweeps_per_iter);
        vector<double> energy_series;
        energy_series.reserve(tau_samples);
        
        for (size_t i = 0; i < tau_samples; ++i) {
            metropolis_interleaved(my_T, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                overrelaxation();
            }
            energy_series.push_back(total_energy());
        }
        
        AutocorrelationResult acf = compute_autocorrelation(energy_series, 1);
        double my_tau_int = std::max(1.0, acf.tau_int);
        
        vector<double> all_tau_int(R);
        MPI_Allgather(&my_tau_int, 1, MPI_DOUBLE, all_tau_int.data(), 1, MPI_DOUBLE, comm);
        
        double tau_min_val = *std::min_element(all_tau_int.begin(), all_tau_int.end());
        size_t n_base = 10;
        
        result.autocorrelation_times = all_tau_int;
        result.sweeps_per_temp.resize(R);
        for (int k = 0; k < R; ++k) {
            result.sweeps_per_temp[k] = std::max(size_t(1),
                static_cast<size_t>(std::ceil(n_base * all_tau_int[k] / tau_min_val)));
        }
        
        if (rank == 0) {
            cout << "Autocorrelation times and sweep schedule:" << endl;
            for (int k = 0; k < std::min(R, 15); ++k) {
                cout << "  T[" << k << "] = " << std::scientific << std::setprecision(4) 
                     << 1.0 / beta[k] << "  tau_int = " << std::fixed << std::setprecision(1)
                     << all_tau_int[k] << "  n_sweeps = " << result.sweeps_per_temp[k] << endl;
            }
            if (R > 15) cout << "  ... (" << R - 15 << " more)" << endl;
        }
        
        // Build result (on all ranks)
        result.temperatures.resize(R);
        for (int i = 0; i < R; ++i) {
            result.temperatures[i] = 1.0 / beta[i];
        }
        std::sort(result.temperatures.begin(), result.temperatures.end());
        result.acceptance_rates = acceptance_rates;
        
        result.local_diffusivities.resize(R - 1);
        for (int e = 0; e < R - 1; ++e) {
            double A = acceptance_rates[e];
            result.local_diffusivities[e] = A * (1.0 - A);
        }
        
        result.mean_acceptance_rate = 0.0;
        for (double A : acceptance_rates) {
            result.mean_acceptance_rate += A;
        }
        result.mean_acceptance_rate /= (R - 1);
        
        // Compute round-trip time with Bittner sweep weighting:
        // tau_rt ~ sum_i n_avg_i / f_i  where f_i = A_i * d_beta_i / sum(A_j * d_beta_j)
        double sum_inv_f = 0.0;
        double total_current = 0.0;
        for (int e = 0; e < R - 1; ++e) {
            double d_beta = std::abs(beta[e + 1] - beta[e]);
            double A = std::max(acceptance_rates[e], 1e-6);
            total_current += A * d_beta;
        }
        for (int e = 0; e < R - 1; ++e) {
            double d_beta = std::abs(beta[e + 1] - beta[e]);
            double A = std::max(acceptance_rates[e], 1e-6);
            double f_i = A * d_beta / total_current;
            double n_avg = 0.5 * (result.sweeps_per_temp[e] + result.sweeps_per_temp[e + 1]);
            sum_inv_f += n_avg / f_i;
        }
        result.round_trip_estimate = sum_inv_f;
        
        if (rank == 0) {
            cout << "\n=== Optimized Temperature Grid Summary ===" << endl;
            cout << "Mean acceptance rate: " << std::fixed << std::setprecision(3) 
                 << result.mean_acceptance_rate * 100 << "%" << endl;
            cout << "Estimated round-trip time scale: " << std::scientific 
                 << result.round_trip_estimate << endl;
            cout << "Converged: " << (result.converged ? "YES" : "NO") << endl;
        }
        
        // CRITICAL: Synchronize all ranks before returning
        // This ensures no stray MPI messages remain and all ranks exit together
        MPI_Barrier(comm);
        
        return result;
    }
    // ============================================================

    /**
     * Parallel tempering with MPI for mixed lattice
     * Collects: energy, specific heat, sublattice magnetizations (SU2 and SU3), 
     * and cross-correlations. All with binning analysis for error estimation.
     * 
     * Automatically uses interleaved sweeps when mixed interactions are present.
     * @param comm MPI communicator to use (default: MPI_COMM_WORLD)
     * @param sweeps_per_temp   Bittner adaptive sweep schedule: if non-empty, overrides swap_rate
     *                          with max(sweeps_per_temp) to ensure all replicas decorrelate.
     */
    void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move = true, bool use_interleaved = true,
                           MPI_Comm comm = MPI_COMM_WORLD, bool verbose = false,
                           const vector<size_t>& sweeps_per_temp = {});

private:
    /**
     * Attempt replica exchange between neighboring temperatures
     * Returns 1 if exchange successful, 0 otherwise
     * @param comm MPI communicator to use (default: MPI_COMM_WORLD)
     */
    int attempt_replica_exchange(int rank, int size, const vector<double>& temp,
                                double curr_Temp, size_t swap_parity, MPI_Comm comm = MPI_COMM_WORLD);

public:
    // ============================================================
    // MOLECULAR DYNAMICS
    // ============================================================

    /**
     * Get local field for SU(2) site
     */
    SpinVector get_local_field_SU2(size_t site_index) const {
        SpinVector H = -field_SU2[site_index];
        
        // Onsite
        H += 2.0 * onsite_interaction_SU2[site_index] * spins_SU2[site_index];
        
        // Bilinear
        for (size_t i = 0; i < bilinear_partners_SU2[site_index].size(); ++i) {
            H += bilinear_interaction_SU2[site_index][i] * spins_SU2[bilinear_partners_SU2[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < mixed_bilinear_partners_SU2[site_index].size(); ++i) {
            H += mixed_bilinear_interaction_SU2[site_index][i] * spins_SU3[mixed_bilinear_partners_SU2[site_index][i]];
        }
        
        // Trilinear SU(2)-SU(2)-SU(2) contributions
        for (size_t i = 0; i < trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU2[site_index][i][1];
            const auto& T = trilinear_interaction_SU2[site_index][i];
            
            // Contract tensor with partner spins: H[a] = sum_bc T[a](b,c) * S1[b] * S2[c]
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * spins_SU2[p1_idx](b) * spins_SU2[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        // Mixed trilinear SU(2)-SU(2)-SU(3) contributions
        for (size_t i = 0; i < mixed_trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU2[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU2[site_index][i];
            
            // Contract: H[a] = sum_bc T[a](b,c) * SU2[b] * SU3[c]
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * spins_SU2[p1_idx](b) * spins_SU3[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        return H;
    }

    /**
     * Get local field for SU(3) site
     */
    SpinVector get_local_field_SU3(size_t site_index) const {
        SpinVector H = -field_SU3[site_index];
        
        // Onsite
        H += 2.0 * onsite_interaction_SU3[site_index] * spins_SU3[site_index];
        
        // Bilinear
        for (size_t i = 0; i < bilinear_partners_SU3[site_index].size(); ++i) {
            H += bilinear_interaction_SU3[site_index][i] * spins_SU3[bilinear_partners_SU3[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < mixed_bilinear_partners_SU3[site_index].size(); ++i) {
            H += mixed_bilinear_interaction_SU3[site_index][i] * spins_SU2[mixed_bilinear_partners_SU3[site_index][i]];
        }
        
        // Trilinear SU(3)-SU(3)-SU(3) contributions
        for (size_t i = 0; i < trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU3[site_index][i][1];
            const auto& T = trilinear_interaction_SU3[site_index][i];
            
            // Contract tensor with partner spins: H[a] = sum_bc T[a](b,c) * S1[b] * S2[c]
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU3; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * spins_SU3[p1_idx](b) * spins_SU3[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        // Mixed trilinear SU(3)-SU(2)-SU(2) contributions
        for (size_t i = 0; i < mixed_trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU3[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU3[site_index][i];
            
            // Contract: H[a] = sum_bc T[a](b,c) * SU2_1[b] * SU2_2[c]
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * spins_SU2[p1_idx](b) * spins_SU2[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        return H;
    }

    /**
     * Set pulse parameters for SU(2) (drive field is transformed to local frame)
     */
    void set_pulse_SU2(const vector<SpinVector>& field_in1, double t_B1,
                      const vector<SpinVector>& field_in2, double t_B2,
                      double amp, double width, double freq);

    /**
     * Set pulse parameters for SU(3) (drive field is transformed to local frame)
     */
    void set_pulse_SU3(const vector<SpinVector>& field_in1, double t_B1,
                      const vector<SpinVector>& field_in2, double t_B2,
                      double amp, double width, double freq);

    /**
     * Reset pulse fields
     */
    void reset_pulse() {
        field_drive_SU2[0].setZero();
        field_drive_SU2[1].setZero();
        field_drive_SU3[0].setZero();
        field_drive_SU3[1].setZero();
        field_drive_amp_SU2 = 0.0;
        field_drive_freq_SU2 = 0.0;
        field_drive_amp_SU3 = 0.0;
        field_drive_freq_SU3 = 0.0;
    }

    /**
     * Set SU(3) Bloch damping rates Γ_a for each Gell-Mann channel.
     * See tmfeo3_notes.tex Eq. blochdampedfull:
     *   dn^a/dt += −Γ_a (n^a − n^a_eq)
     * @param rates  8-component vector of damping rates (one per Gell-Mann index)
     */
    void set_damping_SU3(const SpinVector& rates) {
        assert(rates.size() == (int)spin_dim_SU3);
        damping_rates_SU3 = rates;
    }

    /**
     * Set per-site SU(3) equilibrium Bloch vectors for Bloch damping.
     * @param eq  Per-site equilibrium vectors (size = lattice_size_SU3)
     */
    void set_equilibrium_SU3(const SpinConfigSU3& eq) {
        assert(eq.size() == lattice_size_SU3);
        equilibrium_SU3 = eq;
    }

    /**
     * Set uniform equilibrium Bloch vector for all SU(3) sites.
     * Transforms to local sublattice frame via F^T.
     * @param eq_global  Single equilibrium vector in global (Bertaut) frame
     */
    void set_equilibrium_SU3_uniform(const SpinVector& eq_global) {
        for (size_t site = 0; site < lattice_size_SU3; ++site) {
            size_t atom = site % N_atoms_SU3;
            equilibrium_SU3[site] = sublattice_frames_SU3[atom].transpose() * eq_global;
        }
    }

    /**
     * Compute time-dependent drive field for SU(2) site (pre-transformed to local frame)
     */
    SpinVector drive_field_SU2_at_time(double t, size_t site_index) const;

    /**
     * Compute time-dependent drive field for SU(3) site (pre-transformed to local frame)
     */
    SpinVector drive_field_SU3_at_time(double t, size_t site_index) const;

    /**
     * Drive-field envelope helpers — return the two pulse factors at time
     * `t` without touching any per-site data. Used by `landau_lifshitz` to
     * hoist the two `exp + cos` calls per pulse out of the per-site loop;
     * total cost goes from O(2 * lattice_size) transcendentals per RHS to
     * O(2) per RHS.
     */
    void drive_envelopes_SU2(double t, double& factor1, double& factor2) const;
    void drive_envelopes_SU3(double t, double& factor1, double& factor2) const;

    /**
     * Convert spin configurations to flat state vector
     */
    ODEState spins_to_state() const {
        ODEState state(lattice_size_SU2 * spin_dim_SU2 + lattice_size_SU3 * spin_dim_SU3);
        
        size_t idx = 0;
        // Pack SU(2) spins
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t j = 0; j < spin_dim_SU2; ++j) {
                state[idx++] = spins_SU2[i](j);
            }
        }
        // Pack SU(3) spins
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t j = 0; j < spin_dim_SU3; ++j) {
                state[idx++] = spins_SU3[i](j);
            }
        }
        
        return state;
    }

    /**
     * Convert flat state vector to spin configurations
     */
    void state_to_spins(const ODEState& state, SpinConfigSU2& spins2, SpinConfigSU3& spins3) const {
        // Resize output vectors
        spins2.resize(lattice_size_SU2);
        spins3.resize(lattice_size_SU3);
        
        size_t idx = 0;
        // Unpack SU(2) spins
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            spins2[i] = SpinVector(spin_dim_SU2);
            for (size_t j = 0; j < spin_dim_SU2; ++j) {
                spins2[i](j) = state[idx++];
            }
        }
        // Unpack SU(3) spins
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            spins3[i] = SpinVector(spin_dim_SU3);
            for (size_t j = 0; j < spin_dim_SU3; ++j) {
                spins3[i](j) = state[idx++];
            }
        }
    }

    /**
     * Generic ODE integration wrapper supporting multiple methods
     * Mirrors the integrator selection from lattice.h
     * 
     * @param system_func ODE system function (x, dxdt, t)
     * @param state Current state (will be modified in-place)
     * @param T_start Start time
     * @param T_end End time
     * @param dt_step Time step (initial for adaptive methods)
     * @param observer Observer function called at intervals
     * @param method Integration method (see list below)
     * @param use_adaptive Use adaptive stepping (if method supports it)
     * @param abs_tol Absolute tolerance for adaptive methods
     * @param rel_tol Relative tolerance for adaptive methods
     * 
     * Available methods:
     * 
     * EXPLICIT METHODS (recommended for non-stiff problems):
     * - "euler": Explicit Euler (1st order, simple, inaccurate)
     * - "rk2" or "midpoint": Runge-Kutta 2nd order
     * - "rk4": Classic Runge-Kutta 4th order (good balance, fixed step)
     * - "rk5" or "rkck54": Cash-Karp 5(4) (adaptive, good for smooth problems)
     * - "rk54" or "rkf54": Runge-Kutta-Fehlberg 5(4) (adaptive)
     * - "dopri5": Dormand-Prince 5(4) (default, recommended for general use)
     * - "rk78" or "rkf78": Runge-Kutta-Fehlberg 7(8) (high accuracy, expensive)
     * - "bulirsch_stoer" or "bs": Bulirsch-Stoer (very high accuracy, expensive)
     * - "adams_bashforth" or "ab": Adams-Bashforth 5-step multistep (efficient for smooth problems)
     * - "adams_moulton" or "am": Adams-Bashforth-Moulton predictor-corrector (more accurate)
     * 
     * IMPLICIT METHODS (recommended for stiff problems):
     * - "rosenbrock4" or "rb4": Rosenbrock 4th order (stiff systems, uses numerical Jacobian)
     * - "implicit_euler" or "ie": Implicit Euler (1st order, very stable for stiff systems)
     * 
     * Note: Implicit methods use numerical Jacobian approximation via finite differences.
     * They are more stable for stiff problems but computationally more expensive.
     */
    template<typename System, typename Observer>
    void integrate_ode_system(System system_func, ODEState& state,
                             double T_start, double T_end, double dt_step,
                             Observer observer, const string& method,
                             bool use_adaptive = false,
                             double abs_tol = 1e-6, double rel_tol = 1e-6) {
        namespace odeint = boost::numeric::odeint;
        
        if (method == "euler") {
            odeint::integrate_const(
                odeint::euler<ODEState>(),
                system_func, state, T_start, T_end, dt_step, observer
            );
        } else if (method == "rk2" || method == "midpoint") {
            odeint::integrate_const(
                odeint::modified_midpoint<ODEState>(),
                system_func, state, T_start, T_end, dt_step, observer
            );
        } else if (method == "rk4") {
            odeint::integrate_const(
                odeint::runge_kutta4<ODEState>(),
                system_func, state, T_start, T_end, dt_step, observer
            );
        } else if (method == "rk5" || method == "rkck54") {
            if (use_adaptive) {
                odeint::integrate_adaptive(
                    odeint::make_controlled<odeint::runge_kutta_cash_karp54<ODEState>>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            } else {
                odeint::integrate_const(
                    odeint::make_controlled<odeint::runge_kutta_cash_karp54<ODEState>>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            }
        } else if (method == "rk54" || method == "rkf54") {
            // Aliased to Cash-Karp 5(4): see Lattice::integrate_ode_system
            // for the rationale (Boost has no fehlberg54 stepper; the
            // previous fehlberg78 wiring was a copy-paste bug that
            // silently doubled the per-step cost).
            if (use_adaptive) {
                odeint::integrate_adaptive(
                    odeint::make_controlled<odeint::runge_kutta_cash_karp54<ODEState>>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            } else {
                odeint::integrate_const(
                    odeint::make_controlled<odeint::runge_kutta_cash_karp54<ODEState>>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            }
        } else if (method == "dopri5") {
            if (use_adaptive) {
                odeint::integrate_adaptive(
                    odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            } else {
                odeint::integrate_const(
                    odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            }
        } else if (method == "rk78" || method == "rkf78") {
            if (use_adaptive) {
                odeint::integrate_adaptive(
                    odeint::make_controlled<odeint::runge_kutta_fehlberg78<ODEState>>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            } else {
                odeint::integrate_const(
                    odeint::make_controlled<odeint::runge_kutta_fehlberg78<ODEState>>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            }
        } else if (method == "bulirsch_stoer" || method == "bs") {
            if (use_adaptive) {
                odeint::integrate_adaptive(
                    odeint::bulirsch_stoer<ODEState>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            } else {
                odeint::integrate_const(
                    odeint::bulirsch_stoer<ODEState>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            }
        } else if (method == "adams_bashforth" || method == "ab") {
            // Adams-Bashforth 5-step multistep method (efficient for smooth problems)
            odeint::adams_bashforth<5, ODEState> stepper;
            odeint::integrate_const(stepper, system_func, state, T_start, T_end, dt_step, observer);
        } else if (method == "adams_moulton" || method == "am") {
            // Adams-Bashforth-Moulton predictor-corrector (higher accuracy multistep)
            odeint::adams_bashforth_moulton<5, ODEState> stepper;
            odeint::integrate_const(stepper, system_func, state, T_start, T_end, dt_step, observer);
        } else if (method == "rosenbrock4" || method == "rb4") {
            // Rosenbrock 4th order implicit method (good for stiff systems)
            // Uses numerical Jacobian approximation via finite differences
            using ublas_state = boost::numeric::ublas::vector<double>;
            using ublas_matrix = boost::numeric::ublas::matrix<double>;
            
            const size_t N = state.size();
            const double eps_jac = 1e-8;  // Finite difference step for Jacobian
            
            // Convert std::vector state to ublas::vector
            ublas_state ublas_x(N);
            for (size_t i = 0; i < N; ++i) {
                ublas_x(i) = state[i];
            }
            
            // Create wrapper for system function that works with ublas types
            auto ublas_system = [&system_func, N](const ublas_state& x, ublas_state& dxdt, double t) {
                ODEState x_vec(N), dxdt_vec(N);
                for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
                system_func(x_vec, dxdt_vec, t);
                for (size_t i = 0; i < N; ++i) dxdt(i) = dxdt_vec[i];
            };
            
            // Create numerical Jacobian function
            auto ublas_jacobian = [&system_func, N, eps_jac](const ublas_state& x, ublas_matrix& J, double t, ublas_state& dfdt) {
                ODEState x_vec(N), dxdt_base(N), dxdt_pert(N);
                for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
                
                // Compute base derivative
                system_func(x_vec, dxdt_base, t);
                
                // Compute Jacobian columns by finite differences
                J.resize(N, N);
                for (size_t j = 0; j < N; ++j) {
                    double x_orig = x_vec[j];
                    double h = eps_jac * std::max(1.0, std::abs(x_orig));
                    x_vec[j] = x_orig + h;
                    system_func(x_vec, dxdt_pert, t);
                    x_vec[j] = x_orig;
                    
                    for (size_t i = 0; i < N; ++i) {
                        J(i, j) = (dxdt_pert[i] - dxdt_base[i]) / h;
                    }
                }
                
                // Compute df/dt by finite differences in time
                double h_t = eps_jac * std::max(1.0, std::abs(t));
                for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
                system_func(x_vec, dxdt_pert, t + h_t);
                for (size_t i = 0; i < N; ++i) {
                    dfdt(i) = (dxdt_pert[i] - dxdt_base[i]) / h_t;
                }
            };
            
            // Create implicit system as pair of (system, jacobian)
            auto implicit_system = std::make_pair(ublas_system, ublas_jacobian);
            
            // Create ublas observer wrapper
            auto ublas_observer = [&observer, N](const ublas_state& x, double t) {
                ODEState x_vec(N);
                for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
                observer(x_vec, t);
            };
            
            // Use rosenbrock4 with dense output for adaptive stepping
            // Note: rosenbrock4<double> means double is the value_type (scalar type)
            //       The state type is automatically ublas::vector<double>
            if (use_adaptive) {
                odeint::integrate_adaptive(
                    odeint::make_dense_output<odeint::rosenbrock4<double>>(abs_tol, rel_tol),
                    implicit_system, ublas_x, T_start, T_end, dt_step, ublas_observer);
            } else {
                odeint::integrate_const(
                    odeint::make_dense_output<odeint::rosenbrock4<double>>(abs_tol, rel_tol),
                    implicit_system, ublas_x, T_start, T_end, dt_step, ublas_observer);
            }
            
            // Copy result back to std::vector state
            for (size_t i = 0; i < N; ++i) {
                state[i] = ublas_x(i);
            }
        } else if (method == "implicit_euler" || method == "ie") {
            // Implicit Euler method (1st order, very stable for stiff systems)
            // Uses numerical Jacobian approximation via finite differences
            using ublas_state = boost::numeric::ublas::vector<double>;
            using ublas_matrix = boost::numeric::ublas::matrix<double>;
            
            const size_t N = state.size();
            const double eps_jac = 1e-8;
            
            // Convert to ublas state
            ublas_state ublas_x(N);
            for (size_t i = 0; i < N; ++i) {
                ublas_x(i) = state[i];
            }
            
            // Create wrapper for system function
            auto ublas_system = [&system_func, N](const ublas_state& x, ublas_state& dxdt, double t) {
                ODEState x_vec(N), dxdt_vec(N);
                for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
                system_func(x_vec, dxdt_vec, t);
                for (size_t i = 0; i < N; ++i) dxdt(i) = dxdt_vec[i];
            };
            
            // Create numerical Jacobian function for implicit_euler
            // Note: implicit_euler uses 3-argument Jacobian: (x, J, t) without dfdt
            auto ublas_jacobian = [&system_func, N, eps_jac](const ublas_state& x, ublas_matrix& J, double t) {
                ODEState x_vec(N), dxdt_base(N), dxdt_pert(N);
                for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
                
                system_func(x_vec, dxdt_base, t);
                
                J.resize(N, N);
                for (size_t j = 0; j < N; ++j) {
                    double x_orig = x_vec[j];
                    double h = eps_jac * std::max(1.0, std::abs(x_orig));
                    x_vec[j] = x_orig + h;
                    system_func(x_vec, dxdt_pert, t);
                    x_vec[j] = x_orig;
                    
                    for (size_t i = 0; i < N; ++i) {
                        J(i, j) = (dxdt_pert[i] - dxdt_base[i]) / h;
                    }
                }
            };
            
            auto implicit_system = std::make_pair(ublas_system, ublas_jacobian);
            
            // Create ublas observer wrapper
            auto ublas_observer = [&observer, N](const ublas_state& x, double t) {
                ODEState x_vec(N);
                for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
                observer(x_vec, t);
            };
            
            // Implicit Euler integration with manual stepping
            // Note: implicit_euler<double> uses ublas::vector<double> as state
            odeint::implicit_euler<double> stepper;
            double t = T_start;
            while (t < T_end) {
                stepper.do_step(implicit_system, ublas_x, t, dt_step);
                t += dt_step;
                ublas_observer(ublas_x, t);
            }
            
            // Copy result back
            for (size_t i = 0; i < N; ++i) {
                state[i] = ublas_x(i);
            }
        } else {
            cout << "Warning: Unknown method '" << method << "', using dopri5" << endl;
            cout << "Available explicit methods: euler, rk2/midpoint, rk4, rk5/rkck54, rk54/rkf54, dopri5, " << endl;
            cout << "                            rk78/rkf78, bulirsch_stoer/bs, adams_bashforth/ab, adams_moulton/am" << endl;
            cout << "Available implicit methods: rosenbrock4/rb4, implicit_euler/ie" << endl;
            if (use_adaptive) {
                odeint::integrate_adaptive(
                    odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            } else {
                odeint::integrate_const(
                    odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                    system_func, state, T_start, T_end, dt_step, observer
                );
            }
        }
    }

    /**
     * ODE system function for Boost.Odeint
     */
    void ode_system(const ODEState& x, ODEState& dxdt, double t);

    /**
     * Compute Landau-Lifshitz derivatives using structure constants
     * Note: For SU(2), use cross product. For SU(3), use Gell-Mann structure constants.
     * 
     * Direct ODEState implementation - no intermediate SpinConfig conversions for efficiency.
     * State layout: [SU2_site0_components... SU2_siteN... SU3_site0_components... SU3_siteM...]
     */
    void landau_lifshitz(const ODEState& state, ODEState& dsdt, double t);

private:
    /**
     * Compute local field for SU(2) site directly from flat state vector
     * More efficient than get_local_field_SU2() which uses SpinConfig
     * 
     * @param site Site index in SU(2) sublattice
     * @param state Flat ODE state vector
     * @param offset_SU3 Starting index of SU(3) spins in state vector
     * @param t Current time (for time-dependent drive)
     */
    /**
     * Heap-free, drive-hoisted variants of `get_local_field_SU{2,3}_flat`.
     * The full local field is written directly into the caller-supplied
     * `H_out[0..spin_dim_SU{2,3}-1]` and the time-dependent envelope
     * factors are passed in by the caller (typically computed once per RHS
     * evaluation in `landau_lifshitz`). These are the form used by the LLG
     * hot loop; together they eliminate one `Eigen::VectorXd` heap
     * allocation per site per RHS call and the redundant `exp + cos` calls
     * per site that the legacy `..._flat(t, site)` interface incurs.
     */
    void get_local_field_SU2_flat_into(size_t site, const ODEState& state,
                                       size_t offset_SU3,
                                       double drive_factor1, double drive_factor2,
                                       double* H_out) const;
    void get_local_field_SU3_flat_into(size_t site, const ODEState& state,
                                       size_t offset_SU3,
                                       double drive_factor1, double drive_factor2,
                                       double* H_out) const;

    SpinVector get_local_field_SU2_flat(size_t site, const ODEState& state, 
                                         size_t offset_SU3, double t) const;
    
    /**
     * Compute local field for SU(3) site directly from flat state vector
     * More efficient than get_local_field_SU3() which uses SpinConfig
     * 
     * @param site Site index in SU(3) sublattice
     * @param state Flat ODE state vector
     * @param offset_SU3 Starting index of SU(3) spins in state vector
     * @param t Current time (for time-dependent drive)
     */
    SpinVector get_local_field_SU3_flat(size_t site, const ODEState& state, 
                                         size_t offset_SU3, double t) const;

    /**
     * Helper: Get integration tolerances based on method
     */
    static std::pair<double, double> get_integration_tolerances(const string& method) {
        if (method == "bulirsch_stoer") {
            return {1e-8, 1e-8};  // abs_tol, rel_tol
        }
        return {1e-6, 1e-6};
    }

    /**
     * Helper: Safely create directories if path is non-empty
     */
    static void ensure_directory_exists(const string& dir_path) {
        if (!dir_path.empty()) {
            ensure_directory_exists(dir_path);
        }
    }

    /**
     * Helper: Compute local and antiferromagnetic magnetization from flat state for a sublattice
     * @param x Flat state array
     * @param offset Starting index in flat array
     * @param lattice_size Number of sites
     * @param spin_dim Spin dimension
     * @param M_local_arr Output array for local magnetization
     * @param M_antiferro_arr Output array for antiferromagnetic magnetization
     */
    static void compute_sublattice_magnetizations_from_flat(const double* x, size_t offset,
                                                            size_t lattice_size, size_t spin_dim,
                                                            double* M_local_arr, double* M_antiferro_arr) {
        std::fill(M_local_arr, M_local_arr + spin_dim, 0.0);
        std::fill(M_antiferro_arr, M_antiferro_arr + spin_dim, 0.0);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            double sign = (i % 2 == 0) ? 1.0 : -1.0;
            size_t idx = offset + i * spin_dim;
            for (size_t d = 0; d < spin_dim; ++d) {
                M_local_arr[d] += x[idx + d];
                M_antiferro_arr[d] += x[idx + d] * sign;
            }
        }
    }

    /**
     * Helper: Perform MC sweeps with optional overrelaxation
     * Returns sum of acceptance rates from metropolis calls
     * 
     * @param n_sweeps Number of sweeps to perform
     * @param T Temperature
     * @param gaussian_move Use Gaussian moves
     * @param sigma Gaussian move width
     * @param overrelaxation_rate Perform overrelaxation every N sweeps (0 = disabled)
     * @param interleaved Use interleaved sweeps (better for mixed interactions)
     */
    double perform_mc_sweeps(size_t n_sweeps, double T, bool gaussian_move, 
                            double& sigma, size_t overrelaxation_rate = 0,
                            bool interleaved = true);

    /**
     * Get local field for SU(2) from temporary state
     */
    SpinVector get_local_field_SU2_state(size_t site_index, 
                                         const SpinConfigSU2& curr_spins2,
                                         const SpinConfigSU3& curr_spins3) const {
        SpinVector H = -field_SU2[site_index];
        
        // Onsite
        H += 2.0 * onsite_interaction_SU2[site_index] * curr_spins2[site_index];
        
        // Bilinear
        for (size_t i = 0; i < bilinear_partners_SU2[site_index].size(); ++i) {
            H += bilinear_interaction_SU2[site_index][i] * curr_spins2[bilinear_partners_SU2[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < mixed_bilinear_partners_SU2[site_index].size(); ++i) {
            H += mixed_bilinear_interaction_SU2[site_index][i] * curr_spins3[mixed_bilinear_partners_SU2[site_index][i]];
        }
        
        // Trilinear SU(2)-SU(2)-SU(2) contributions
        for (size_t i = 0; i < trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU2[site_index][i][1];
            const auto& T = trilinear_interaction_SU2[site_index][i];
            
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * curr_spins2[p1_idx](b) * curr_spins2[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        // Mixed trilinear contributions
        for (size_t i = 0; i < mixed_trilinear_partners_SU2[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU2[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU2[site_index][i];
            
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * curr_spins2[p1_idx](b) * curr_spins3[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        return H;
    }

    /**
     * Get local field for SU(3) from temporary state
     */
    SpinVector get_local_field_SU3_state(size_t site_index,
                                         const SpinConfigSU2& curr_spins2,
                                         const SpinConfigSU3& curr_spins3) const {
        SpinVector H = -field_SU3[site_index];
        
        // Onsite
        H += 2.0 * onsite_interaction_SU3[site_index] * curr_spins3[site_index];
        
        // Bilinear
        for (size_t i = 0; i < bilinear_partners_SU3[site_index].size(); ++i) {
            H += bilinear_interaction_SU3[site_index][i] * curr_spins3[bilinear_partners_SU3[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < mixed_bilinear_partners_SU3[site_index].size(); ++i) {
            H += mixed_bilinear_interaction_SU3[site_index][i] * curr_spins2[mixed_bilinear_partners_SU3[site_index][i]];
        }
        
        // Trilinear SU(3)-SU(3)-SU(3) contributions
        for (size_t i = 0; i < trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU3[site_index][i][1];
            const auto& T = trilinear_interaction_SU3[site_index][i];
            
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU3; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * curr_spins3[p1_idx](b) * curr_spins3[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        // Mixed trilinear contributions
        for (size_t i = 0; i < mixed_trilinear_partners_SU3[site_index].size(); ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU3[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU3[site_index][i];
            
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * curr_spins2[p1_idx](b) * curr_spins2[p2_idx](c);
                    }
                }
                H(a) += temp;
            }
        }
        
        return H;
    }

public:

    // ============================================================
    // OBSERVABLES
    // ============================================================

    /**
     * Compute SU(2) magnetization
     */
    SpinVector magnetization_SU2() const {
        SpinVector mag = SpinVector::Zero(spin_dim_SU2);
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            mag += spins_SU2[i];
        }
        return mag / double(lattice_size_SU2);
    }

    /**
     * Compute SU(3) magnetization
     */
    SpinVector magnetization_SU3() const {
        SpinVector mag = SpinVector::Zero(spin_dim_SU3);
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            mag += spins_SU3[i];
        }
        return mag / double(lattice_size_SU3);
    }

    /**
     * Helper function to compute SU(2) global magnetization from flat state
     */
    void compute_magnetization_global_SU2_from_flat(const double* x, double* M_global_arr) const {
        for (size_t d = 0; d < spin_dim_SU2; ++d) M_global_arr[d] = 0.0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            size_t atom = i % N_atoms_SU2;
            size_t idx = i * spin_dim_SU2;
            for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
                for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
                    M_global_arr[mu] += sublattice_frames_SU2[atom](nu, mu) * x[idx + nu];
                }
            }
        }
        for (size_t d = 0; d < spin_dim_SU2; ++d) M_global_arr[d] /= double(lattice_size_SU2);
    }

    /**
     * Helper function to compute SU(3) global magnetization from flat state
     */
    void compute_magnetization_global_SU3_from_flat(const double* x, double* M_global_arr) const {
        for (size_t d = 0; d < spin_dim_SU3; ++d) M_global_arr[d] = 0.0;
        size_t offset = lattice_size_SU2 * spin_dim_SU2;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            size_t atom = i % N_atoms_SU3;
            size_t idx = offset + i * spin_dim_SU3;
            for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
                for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
                    M_global_arr[mu] += sublattice_frames_SU3[atom](nu, mu) * x[idx + nu];
                }
            }
        }
        for (size_t d = 0; d < spin_dim_SU3; ++d) M_global_arr[d] /= double(lattice_size_SU3);
    }

    /**
     * Helper function to compute SU(2) staggered magnetization from flat state
     * Uses sublattice frames and AFM signs (Bertaut modes)
     */
    void compute_magnetization_staggered_SU2_from_flat(const double* x, double* M_stag_arr) const {
        for (size_t d = 0; d < spin_dim_SU2; ++d) M_stag_arr[d] = 0.0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            size_t atom = i % N_atoms_SU2;
            double sign = afm_sublattice_signs_SU2[atom];
            size_t idx = i * spin_dim_SU2;
            for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
                for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
                    M_stag_arr[mu] += sign * sublattice_frames_SU2[atom](nu, mu) * x[idx + nu];
                }
            }
        }
        for (size_t d = 0; d < spin_dim_SU2; ++d) M_stag_arr[d] /= double(lattice_size_SU2);
    }

    // ============================================================
    // FILE I/O
    // ============================================================

    /**
     * Save spin configuration
     */
    void save_spin_config(const string& filename) const {
        // SU(2) spins
        {
            ofstream file(filename + "_SU2.txt");
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                for (size_t j = 0; j < spin_dim_SU2; ++j) {
                    file << spins_SU2[i](j) << " ";
                }
                file << "\n";
            }
        }
        
        // SU(3) spins
        {
            ofstream file(filename + "_SU3.txt");
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                for (size_t j = 0; j < spin_dim_SU3; ++j) {
                    file << spins_SU3[i](j) << " ";
                }
                file << "\n";
            }
        }
    }

    /**
     * Save spin configuration to a directory with clean naming
     * Creates: spins_SU2.txt and spins_SU3.txt in the directory
     */
    void save_spin_config_to_dir(const string& dir, const string& prefix = "spins") const {
        // SU(2) spins
        {
            ofstream file(dir + "/" + prefix + "_SU2.txt");
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                for (size_t j = 0; j < spin_dim_SU2; ++j) {
                    file << spins_SU2[i](j) << " ";
                }
                file << "\n";
            }
        }
        
        // SU(3) spins
        {
            ofstream file(dir + "/" + prefix + "_SU3.txt");
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                for (size_t j = 0; j < spin_dim_SU3; ++j) {
                    file << spins_SU3[i](j) << " ";
                }
                file << "\n";
            }
        }
    }

    /**
     * Save energy information to a directory
     * Creates: energy.txt with total, SU2, and SU3 energies (both total and per-site)
     */
    void save_energy_to_dir(const string& dir, const string& prefix = "energy") const {
        double E_total = total_energy();
        double E_SU2 = total_energy_SU2();
        double E_SU3 = total_energy_SU3();
        size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        
        ofstream file(dir + "/" + prefix + ".txt");
        file << std::setprecision(15);
        file << "# Energy summary" << endl;
        file << "# N_SU2 = " << lattice_size_SU2 << endl;
        file << "# N_SU3 = " << lattice_size_SU3 << endl;
        file << "# N_total = " << total_sites << endl;
        file << "#" << endl;
        file << "# Total energies:" << endl;
        file << "E_total = " << E_total << endl;
        file << "E_SU2 = " << E_SU2 << endl;
        file << "E_SU3 = " << E_SU3 << endl;
        file << "#" << endl;
        file << "# Energy per site:" << endl;
        file << "E_total/N = " << E_total / total_sites << endl;
        file << "E_SU2/N_SU2 = " << E_SU2 / lattice_size_SU2 << endl;
        file << "E_SU3/N_SU3 = " << E_SU3 / lattice_size_SU3 << endl;
        file.close();
    }

    /**
     * Load spin configuration
     */
    void load_spin_config(const string& filename) {
        // Load SU(2) spins
        {
            ifstream file(filename + "_SU2.txt");
            if (!file) {
                cerr << "Error: Cannot open " << filename << "_SU2.txt" << endl;
                return;
            }
            
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                for (size_t j = 0; j < spin_dim_SU2; ++j) {
                    file >> spins_SU2[i](j);
                }
            }
        }
        
        // Load SU(3) spins
        {
            ifstream file(filename + "_SU3.txt");
            if (!file) {
                cerr << "Error: Cannot open " << filename << "_SU3.txt" << endl;
                return;
            }
            
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                for (size_t j = 0; j < spin_dim_SU3; ++j) {
                    file >> spins_SU3[i](j);
                }
            }
        }
    }

    /**
     * Save site positions (legacy naming - appends _SU2.txt/_SU3.txt to filename)
     */
    void save_positions(const string& filename) const {
        // SU(2) positions
        {
            ofstream file(filename + "_SU2.txt");
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                file << site_positions_SU2[i](0) << " "
                     << site_positions_SU2[i](1) << " "
                     << site_positions_SU2[i](2) << "\n";
            }
        }
        
        // SU(3) positions
        {
            ofstream file(filename + "_SU3.txt");
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                file << site_positions_SU3[i](0) << " "
                     << site_positions_SU3[i](1) << " "
                     << site_positions_SU3[i](2) << "\n";
            }
        }
    }

    /**
     * Save site positions to a directory with clean naming
     * Creates: positions_SU2.txt and positions_SU3.txt in the directory
     */
    void save_positions_to_dir(const string& dir) const {
        // SU(2) positions
        {
            ofstream file(dir + "/positions_SU2.txt");
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                file << site_positions_SU2[i](0) << " "
                     << site_positions_SU2[i](1) << " "
                     << site_positions_SU2[i](2) << "\n";
            }
        }
        
        // SU(3) positions
        {
            ofstream file(dir + "/positions_SU3.txt");
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                file << site_positions_SU3[i](0) << " "
                     << site_positions_SU3[i](1) << " "
                     << site_positions_SU3[i](2) << "\n";
            }
        }
    }

    /**
     * Initialize with ferromagnetic state
     */
    void init_ferromagnetic(const SpinVector& direction_SU2, const SpinVector& direction_SU3) {
        const SpinVector dir_SU2 = direction_SU2.normalized() * spin_length_SU2;
        const SpinVector dir_SU3 = direction_SU3.normalized() * spin_length_SU3;
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            spins_SU2[i] = dir_SU2;
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            spins_SU3[i] = dir_SU3;
        }
    }

    /**
     * Initialize with random state
     */
    void init_random() {
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            spins_SU2[i] = gen_random_spin(spin_length_SU2, spin_dim_SU2);
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            spins_SU3[i] = gen_random_spin(spin_length_SU3, spin_dim_SU3);
        }
    }

    /**
     * Molecular dynamics with single pulse field for SU(2) sublattice
     * Returns magnetization trajectory without I/O
     */
    vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>> single_pulse_drive(
               const vector<SpinVector>& field_in_SU2, const vector<SpinVector>& field_in_SU3,
               double t_B,
               double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
               double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
               double T_start, double T_end, double step_size,
               const string& method = "dopri5", bool use_gpu = false,
               vector<vector<double>>* spin_state_out = nullptr,
               // W3: pulse-window-aware chunked integration. The pulse
               // window is built from max(σ_SU2, σ_SU3) so we never
               // under-resolve either drive envelope.
               bool pulse_window_chunking = true,
               // Ingredient XVIII: pump-probe ODE tolerances (default 1e-8;
               // previously hard-coded 1e-10).
               double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
               double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol) {
        
        if (use_gpu) {
#ifdef CUDA_ENABLED
            return single_pulse_drive_gpu(field_in_SU2, field_in_SU3, t_B,
                            pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                            pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                            T_start, T_end, step_size, method);
#else
            std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED)." << endl;
            std::cerr << "Falling back to CPU implementation." << endl;
            // Fall through to CPU implementation
#endif
        }
        
        // Set up pulse
        set_pulse_SU2(field_in_SU2, t_B, vector<SpinVector>(N_atoms_SU2, SpinVector::Zero(spin_dim_SU2)),
                     0.0, pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2);
        set_pulse_SU3(field_in_SU3, t_B, vector<SpinVector>(N_atoms_SU3, SpinVector::Zero(spin_dim_SU3)),
                     0.0, pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3);
        
        // Storage for trajectory: (time, ([M_SU2_antiferro, M_SU2_local, M_SU2_global], [M_SU3_antiferro, M_SU3_local, M_SU3_global]))
        vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>> trajectory;
        
        // Start from current spins configuration
        ODEState state = spins_to_state();
        
        // Create ODE system wrapper
        auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
            this->ode_system(x, dxdt, t);
        };
        
        // Observer to collect magnetization at regular intervals
        double last_save_time = T_start;
        auto observer = [&](const ODEState& x, double t) {
            if (t - last_save_time >= step_size - 1e-10 || t >= T_end - 1e-10) {
                // Compute SU(2) magnetizations using helper functions
                double M_SU2_local_arr[8] = {0};
                double M_SU2_antiferro_arr[8] = {0};
                double M_SU2_global_arr[8] = {0};
                
                compute_sublattice_magnetizations_from_flat(x.data(), 0, 
                    lattice_size_SU2, spin_dim_SU2, M_SU2_local_arr, M_SU2_antiferro_arr);
                compute_magnetization_global_SU2_from_flat(x.data(), M_SU2_global_arr);
                compute_magnetization_staggered_SU2_from_flat(x.data(), M_SU2_antiferro_arr);
                
                // Compute SU(3) magnetizations using helper functions
                double M_SU3_local_arr[8] = {0};
                double M_SU3_antiferro_arr[8] = {0};
                double M_SU3_global_arr[8] = {0};
                
                compute_sublattice_magnetizations_from_flat(x.data(), lattice_size_SU2 * spin_dim_SU2, 
                    lattice_size_SU3, spin_dim_SU3, M_SU3_local_arr, M_SU3_antiferro_arr);
                compute_magnetization_global_SU3_from_flat(x.data(), M_SU3_global_arr);
                
                SpinVector M_SU2_local = Eigen::Map<Eigen::VectorXd>(M_SU2_local_arr, spin_dim_SU2) / double(lattice_size_SU2);
                SpinVector M_SU2_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU2_antiferro_arr, spin_dim_SU2);
                SpinVector M_SU2_global = Eigen::Map<Eigen::VectorXd>(M_SU2_global_arr, spin_dim_SU2);
                SpinVector M_SU3_local = Eigen::Map<Eigen::VectorXd>(M_SU3_local_arr, spin_dim_SU3) / double(lattice_size_SU3);
                SpinVector M_SU3_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU3_antiferro_arr, spin_dim_SU3) / double(lattice_size_SU3);
                SpinVector M_SU3_global = Eigen::Map<Eigen::VectorXd>(M_SU3_global_arr, spin_dim_SU3);
                
                trajectory.push_back({t, {{M_SU2_antiferro, M_SU2_local, M_SU2_global}, {M_SU3_antiferro, M_SU3_local, M_SU3_global}}});
                last_save_time = t;
                if (spin_state_out != nullptr) {
                    spin_state_out->push_back(vector<double>(x.begin(), x.end()));
                }
            }
        };

        // ---- W3: pulse-window-aware chunked integration ------------------
        // Ensure we observe at the end of every chunk so the trajectory
        // grid stays exactly the same as the unchunked path.
        if (pulse_window_chunking) {
            namespace ck = classical_spin_pulse_chunking;
            const double sigma = std::max(pulse_width_SU2, pulse_width_SU3);
            const auto segments = ck::build_pulse_segments(
                T_start, T_end,
                /*pulse_centers=*/ {t_B},
                /*window=*/ ck::kPulseWindowSigmas * sigma,
                /*T_step=*/ step_size,
                /*free_dt_factor=*/ ck::kFreeDtFactor);
            for (const auto& seg : segments) {
                integrate_ode_system(system_func, state,
                                     seg.t0, seg.t1, seg.dt_init,
                                     observer, method, false, abs_tol, rel_tol);
            }
        } else {
            integrate_ode_system(system_func, state, T_start, T_end, step_size,
                                observer, method, false, abs_tol, rel_tol);
        }

        // Reset pulse
        reset_pulse();
        
        return trajectory;
    }

    /**
     * Molecular dynamics with two-pulse field
     * Returns magnetization trajectory without I/O
     */
    vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>> double_pulse_drive(
               const vector<SpinVector>& field_in_1_SU2, const vector<SpinVector>& field_in_1_SU3,
               double t_B_1,
               const vector<SpinVector>& field_in_2_SU2, const vector<SpinVector>& field_in_2_SU3,
               double t_B_2,
               double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
               double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
               double T_start, double T_end, double step_size,
               const string& method = "dopri5", bool use_gpu = false,
               vector<vector<double>>* spin_state_out = nullptr,
               bool pulse_window_chunking = true,
               double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
               double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol) {
        
        if (use_gpu) {
#ifdef CUDA_ENABLED
            return double_pulse_drive_gpu(field_in_1_SU2, field_in_1_SU3, t_B_1,
                                field_in_2_SU2, field_in_2_SU3, t_B_2,
                                pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                T_start, T_end, step_size, method);
#else
            std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED)." << endl;
            std::cerr << "Falling back to CPU implementation." << endl;
            // Fall through to CPU implementation
#endif
        }
        
        // Set up two-pulse configuration
        set_pulse_SU2(field_in_1_SU2, t_B_1, field_in_2_SU2, t_B_2,
                     pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2);
        set_pulse_SU3(field_in_1_SU3, t_B_1, field_in_2_SU3, t_B_2,
                     pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3);
        
        // Storage for trajectory
        vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>> trajectory;
        
        // Start from current spins configuration
        ODEState state = spins_to_state();
        
        // Create ODE system wrapper
        auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
            this->ode_system(x, dxdt, t);
        };
        
        // Observer to collect magnetization at regular intervals
        double last_save_time = T_start;
        auto observer = [&](const ODEState& x, double t) {
            if (t - last_save_time >= step_size - 1e-10 || t >= T_end - 1e-10) {
                // Compute SU(2) magnetizations using helper functions
                double M_SU2_local_arr[8] = {0};
                double M_SU2_antiferro_arr[8] = {0};
                double M_SU2_global_arr[8] = {0};
                
                compute_sublattice_magnetizations_from_flat(x.data(), 0, 
                    lattice_size_SU2, spin_dim_SU2, M_SU2_local_arr, M_SU2_antiferro_arr);
                compute_magnetization_global_SU2_from_flat(x.data(), M_SU2_global_arr);
                compute_magnetization_staggered_SU2_from_flat(x.data(), M_SU2_antiferro_arr);
                
                // Compute SU(3) magnetizations using helper functions
                double M_SU3_local_arr[8] = {0};
                double M_SU3_antiferro_arr[8] = {0};
                double M_SU3_global_arr[8] = {0};
                
                compute_sublattice_magnetizations_from_flat(x.data(), lattice_size_SU2 * spin_dim_SU2, 
                    lattice_size_SU3, spin_dim_SU3, M_SU3_local_arr, M_SU3_antiferro_arr);
                compute_magnetization_global_SU3_from_flat(x.data(), M_SU3_global_arr);
                
                SpinVector M_SU2_local = Eigen::Map<Eigen::VectorXd>(M_SU2_local_arr, spin_dim_SU2) / double(lattice_size_SU2);
                SpinVector M_SU2_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU2_antiferro_arr, spin_dim_SU2);
                SpinVector M_SU2_global = Eigen::Map<Eigen::VectorXd>(M_SU2_global_arr, spin_dim_SU2);
                SpinVector M_SU3_local = Eigen::Map<Eigen::VectorXd>(M_SU3_local_arr, spin_dim_SU3) / double(lattice_size_SU3);
                SpinVector M_SU3_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU3_antiferro_arr, spin_dim_SU3) / double(lattice_size_SU3);
                SpinVector M_SU3_global = Eigen::Map<Eigen::VectorXd>(M_SU3_global_arr, spin_dim_SU3);
                
                trajectory.push_back({t, {{M_SU2_antiferro, M_SU2_local, M_SU2_global}, {M_SU3_antiferro, M_SU3_local, M_SU3_global}}});
                last_save_time = t;
                if (spin_state_out != nullptr) {
                    spin_state_out->push_back(vector<double>(x.begin(), x.end()));
                }
            }
        };

        // ---- W3: pulse-window-aware chunked integration ------------------
        if (pulse_window_chunking) {
            namespace ck = classical_spin_pulse_chunking;
            const double sigma = std::max(pulse_width_SU2, pulse_width_SU3);
            const auto segments = ck::build_pulse_segments(
                T_start, T_end,
                /*pulse_centers=*/ {t_B_1, t_B_2},
                /*window=*/ ck::kPulseWindowSigmas * sigma,
                /*T_step=*/ step_size,
                /*free_dt_factor=*/ ck::kFreeDtFactor);
            for (const auto& seg : segments) {
                integrate_ode_system(system_func, state,
                                     seg.t0, seg.t1, seg.dt_init,
                                     observer, method, false, abs_tol, rel_tol);
            }
        } else {
            integrate_ode_system(system_func, state, T_start, T_end, step_size,
                                observer, method, false, abs_tol, rel_tol);
        }

        // Reset pulse
        reset_pulse();
        
        return trajectory;
    }

    /**
     * Molecular dynamics using Boost.Odeint
     */
    void molecular_dynamics(double T_start, double T_end, double dt_initial,
                           const string& out_dir = "", size_t save_interval = 100,
                           const string& method = "dopri5", bool use_gpu = false,
                           // Ingredient XVIII: MD tolerance overrides. Negative
                           // values fall back to get_integration_tolerances(method)
                           // so legacy callers keep their existing 1e-6 (or 1e-8 for BS).
                           double abs_tol = -1.0, double rel_tol = -1.0);

    /**
     * CPU implementation of molecular dynamics
     * Requires HDF5 for output - all non-HDF5 I/O has been retired.
     */
    void molecular_dynamics_cpu(double T_start, double T_end, double dt_initial,
                           const string& out_dir = "", size_t save_interval = 100,
                           const string& method = "dopri5",
                           double abs_tol = -1.0, double rel_tol = -1.0);

    /**
     * Print lattice information
     */
    void print_info() const {
        cout << "=== Mixed Lattice Information ===" << endl;
        cout << "Dimensions: " << dim1 << " x " << dim2 << " x " << dim3 << endl;
        cout << "\nSU(2) Sublattice:" << endl;
        cout << "  Sites: " << lattice_size_SU2 << endl;
        cout << "  Spin dimension: " << spin_dim_SU2 << endl;
        cout << "  Atoms per cell: " << N_atoms_SU2 << endl;
        cout << "  Max bilinear: " << num_bi_SU2 << endl;
        cout << "  Max trilinear: " << num_tri_SU2 << endl;
        cout << "\nSU(3) Sublattice:" << endl;
        cout << "  Sites: " << lattice_size_SU3 << endl;
        cout << "  Spin dimension: " << spin_dim_SU3 << endl;
        cout << "  Atoms per cell: " << N_atoms_SU3 << endl;
        cout << "  Max bilinear: " << num_bi_SU3 << endl;
        cout << "  Max trilinear: " << num_tri_SU3 << endl;
        cout << "\nMixed Interactions:" << endl;
        cout << "  Bilinear: " << num_bi_SU2_SU3 << endl;
        cout << "  Trilinear: " << num_tri_SU2_SU3 << endl;
        cout << "=================================" << endl;
    }

    // ------------------------------------------------------------------
    // W1 (time-translation) helpers — see Ingredient XV in
    // docs/optimization_notes.tex.
    //
    // The single-pulse drive trajectory M_1(τ; t) of a deterministic
    // LLG flow that starts from a stationary initial state is just the
    // time-shifted single-pulse-applied-at-t=0 trajectory:
    //
    //     M_1(τ; t) = M_pulse(t − τ)
    //
    // (where M_pulse is the trajectory produced by `single_pulse_drive`
    // with t_B = 0). The same identity reduces M_0(τ; t) to the *no-drive*
    // trajectory of a stationary configuration, which is itself constant
    // in time. We exploit this by computing only M_pulse once per (M_0,
    // first pulse) pair and synthesising every M_1(τ) from it.
    //
    // The synthesis is only safe when:
    //   1. langevin_temperature == 0  (deterministic LLG).
    //   2. The initial state is stationary, i.e. max_t |dS/dt|_∞ < tol
    //      with the drive disabled. We expose `max_dSdt_norm_no_drive()`
    //      so the driver can verify this at runtime.
    // ------------------------------------------------------------------
    using PumpProbeTrajectory =
        vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>>;

    /**
     * Sample max ||dS/dt||_∞ across both sublattices over the
     * current configuration with the pulse drive disabled. Returns a
     * bound that is < tol iff the configuration is a (numerical)
     * stationary point of the deterministic LLG flow.
     *
     * This is W1's runtime safety guard: the spectroscopy driver
     * skips the M_1 → time-shift synthesis whenever the bound exceeds
     * `stationarity_tol`, falling back to the explicit single-pulse
     * integration. We never silently produce a wrong trajectory.
     */
    double max_dSdt_norm_no_drive() const;

    /**
     * Build M_1(τ; t) by time-shifting an M_pulse trajectory captured
     * with t_B = 0:  M_1(τ; t)[i] = M_pulse((t_i − τ)). Indices outside
     * the M_pulse window are filled with the stationary ground-state
     * magnetisation `M_ground` (M_pulse asymptotes to M_ground far
     * from the pulse window).
     *
     * @param M_pulse_trajectory  Output of single_pulse_drive(t_B = 0)
     *                            covering [T_start − τ_max, T_end].
     * @param M_ground            Stationary magnetisation.
     * @param tau                 Delay τ (the pulse fires at t = τ).
     * @param T_start, T_end, T_step  Output grid (same as M_pulse cadence).
     */
    PumpProbeTrajectory synthesize_M1_from_M0(
        const PumpProbeTrajectory& M_pulse_trajectory,
        const pair<array<SpinVector, 3>, array<SpinVector, 3>>& M_ground,
        double tau,
        double T_start, double T_end, double T_step) const;

    /**
     * Complete pump-probe nonlinear spectroscopy workflow for mixed lattice
     * Handles both SU(2) and SU(3) sublattices with consistent nomenclature
     * 
     * NOTE: Ground state should be prepared beforehand via simulated_annealing()
     *       or loaded from file before calling this method.
     *
     * Optional W1/W2/W3 controls:
     *   reuse_m0_for_m1       — opt in to time-translation synthesis of
     *                           M_1 trajectories. Auto-disabled if the
     *                           runtime stationarity guard fails.
     *   stationarity_tol      — guard threshold on max ||dS/dt||_∞.
     *   outer_omp_threads     — OpenMP team size for the τ loop. <= 0
     *                           leaves it to the runtime / env.
     *   pulse_window_chunking — pass-through to single_/double_pulse_drive.
     */
    void pump_probe_spectroscopy(const vector<SpinVector>& field_in_SU2,
                                 const vector<SpinVector>& field_in_SU3,
                                 double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                                 double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                                 double tau_start, double tau_end, double tau_step,
                                 double T_start, double T_end, double T_step,
                                 double Temp_start = 5.0, double Temp_end = 1e-3,
                                 size_t n_anneal = 1000,
                                 bool T_zero_quench = false, size_t quench_sweeps = 1000,
                                 string dir_name = "spectroscopy_mixed",
                                 string method = "dopri5", bool use_gpu = false,
                                 bool reuse_m0_for_m1 = true,
                                 double stationarity_tol = 1e-6,
                                 int outer_omp_threads = 0,
                                 bool pulse_window_chunking = true,
                                 double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                                 double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);

    /**
     * MPI-parallelized pump-probe spectroscopy for mixed lattice
     * 
     * Distributes tau delay values across MPI ranks for parallel computation.
     * Each rank computes a subset of tau values, then rank 0 gathers and writes results.
     *
     * W2 (inter-τ parallelism) is provided here by MPI; the new W1/W3
     * controls behave like the serial driver above.
     */
    void pump_probe_spectroscopy_mpi(const vector<SpinVector>& field_in_SU2,
                                     const vector<SpinVector>& field_in_SU3,
                                     double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                                     double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                                     double tau_start, double tau_end, double tau_step,
                                     double T_start, double T_end, double T_step,
                                     double Temp_start = 5.0, double Temp_end = 1e-3,
                                     size_t n_anneal = 1000,
                                     bool T_zero_quench = false, size_t quench_sweeps = 1000,
                                     string dir_name = "spectroscopy_mixed",
                                     string method = "dopri5", bool use_gpu = false,
                                     bool save_spin_trajectories = false,
                                     bool reuse_m0_for_m1 = true,
                                     double stationarity_tol = 1e-6,
                                     bool pulse_window_chunking = true,
                                     double abs_tol = classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
                                     double rel_tol = classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);

// ============================================================
// GPU Implementation Section
// ============================================================
#if defined(CUDA_ENABLED) && defined(__CUDACC__)
private:
    /**
     * GPU data structure for mixed lattice
     * Contains both SU(2) and SU(3) lattice data
     */
    struct GPUMixedLatticeData {
        // SU(2) sublattice data
        thrust::device_vector<double> d_field_SU2;
        thrust::device_vector<double> d_onsite_interaction_SU2;
        thrust::device_vector<double> d_bilinear_interaction_SU2;
        thrust::device_vector<size_t> d_bilinear_partners_SU2;
        thrust::device_vector<int8_t> d_bilinear_wrap_dir_SU2;
        thrust::device_vector<double> d_trilinear_interaction_SU2;
        thrust::device_vector<size_t> d_trilinear_partners_SU2;
        thrust::device_vector<double> d_field_drive_SU2;
        thrust::device_vector<double> d_twist_matrices_SU2;
        
        // SU(3) sublattice data
        thrust::device_vector<double> d_field_SU3;
        thrust::device_vector<double> d_onsite_interaction_SU3;
        thrust::device_vector<double> d_bilinear_interaction_SU3;
        thrust::device_vector<size_t> d_bilinear_partners_SU3;
        thrust::device_vector<int8_t> d_bilinear_wrap_dir_SU3;
        thrust::device_vector<double> d_trilinear_interaction_SU3;
        thrust::device_vector<size_t> d_trilinear_partners_SU3;
        thrust::device_vector<double> d_field_drive_SU3;
        thrust::device_vector<double> d_twist_matrices_SU3;
        
        // Mixed interaction data
        thrust::device_vector<double> d_mixed_bilinear_interaction;
        thrust::device_vector<size_t> d_mixed_bilinear_partners_SU2;
        thrust::device_vector<size_t> d_mixed_bilinear_partners_SU3;
        thrust::device_vector<int8_t> d_mixed_bilinear_wrap_dir;
        
        thrust::device_vector<double> d_mixed_trilinear_interaction;
        thrust::device_vector<size_t> d_mixed_trilinear_partners_SU2;
        thrust::device_vector<size_t> d_mixed_trilinear_partners_SU3;
        
        // Sizes
        size_t lattice_size_SU2;
        size_t lattice_size_SU3;
        size_t num_bi_SU2;
        size_t num_tri_SU2;
        size_t num_bi_SU3;
        size_t num_tri_SU3;
        size_t num_mixed_bi;
        size_t num_mixed_tri;
        
        // Pulse parameters
        double field_drive_amp_SU2;
        double field_drive_freq_SU2;
        double field_drive_width_SU2;
        double t_pulse_0_SU2;
        double t_pulse_1_SU2;
        
        double field_drive_amp_SU3;
        double field_drive_freq_SU3;
        double field_drive_width_SU3;
        double t_pulse_0_SU3;
        double t_pulse_1_SU3;
    };
    
    // GPU data cache for avoiding repeated transfers (analogous to Lattice::gpu_data_cache_)
    mutable mixed_gpu::GPUMixedLatticeData gpu_mixed_data_cache_;
    mutable bool gpu_mixed_data_initialized_ = false;
    
    /**
     * Ensure GPU mixed lattice data is initialized (lazy initialization)
     * Uses the modular mixed_gpu:: implementation from mixed_lattice_gpu.cuh/cu
     */
    void ensure_gpu_mixed_data_initialized() const {
        if (gpu_mixed_data_initialized_) return;
        gpu_mixed_data_cache_ = create_gpu_mixed_data();
        gpu_mixed_data_initialized_ = true;
    }
    
    /**
     * Update GPU pulse parameters for SU(2) sublattice
     */
    void update_gpu_pulse_SU2() const {
        std::vector<double> flat_field_drive;
        flat_field_drive.reserve(2 * N_atoms_SU2 * spin_dim_SU2);
        for (size_t p = 0; p < 2; ++p) {
            for (size_t d = 0; d < field_drive_SU2[p].size(); ++d) {
                flat_field_drive.push_back(field_drive_SU2[p](d));
            }
        }
        
        mixed_gpu::set_gpu_pulse_SU2(
            gpu_mixed_data_cache_,
            flat_field_drive,
            field_drive_amp_SU2,
            field_drive_width_SU2,
            field_drive_freq_SU2,
            t_pulse_SU2[0],
            t_pulse_SU2[1]
        );
    }
    
    /**
     * Update GPU pulse parameters for SU(3) sublattice
     */
    void update_gpu_pulse_SU3() const {
        std::vector<double> flat_field_drive;
        flat_field_drive.reserve(2 * N_atoms_SU3 * spin_dim_SU3);
        for (size_t p = 0; p < 2; ++p) {
            for (size_t d = 0; d < field_drive_SU3[p].size(); ++d) {
                flat_field_drive.push_back(field_drive_SU3[p](d));
            }
        }
        
        mixed_gpu::set_gpu_pulse_SU3(
            gpu_mixed_data_cache_,
            flat_field_drive,
            field_drive_amp_SU3,
            field_drive_width_SU3,
            field_drive_freq_SU3,
            t_pulse_SU3[0],
            t_pulse_SU3[1]
        );
    }
    
    /**
     * Transfer mixed lattice data to GPU (LEGACY - use ensure_gpu_mixed_data_initialized instead)
     */
    GPUMixedLatticeData transfer_mixed_lattice_data_to_gpu() const {
        GPUMixedLatticeData gpu_data;
        
        // Set sizes
        gpu_data.lattice_size_SU2 = lattice_size_SU2;
        gpu_data.lattice_size_SU3 = lattice_size_SU3;
        gpu_data.num_bi_SU2 = num_bi_SU2;
        gpu_data.num_tri_SU2 = num_tri_SU2;
        gpu_data.num_bi_SU3 = num_bi_SU3;
        gpu_data.num_tri_SU3 = num_tri_SU3;
        
        // Transfer SU(2) field data
        vector<double> flat_field_SU2;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t d = 0; d < spin_dim_SU2; ++d) {
                flat_field_SU2.push_back(field_SU2[i](d));
            }
        }
        gpu_data.d_field_SU2 = thrust::device_vector<double>(flat_field_SU2.begin(), flat_field_SU2.end());
        
        // Transfer SU(3) field data
        vector<double> flat_field_SU3;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t d = 0; d < spin_dim_SU3; ++d) {
                flat_field_SU3.push_back(field_SU3[i](d));
            }
        }
        gpu_data.d_field_SU3 = thrust::device_vector<double>(flat_field_SU3.begin(), flat_field_SU3.end());
        
        // Transfer SU(2) onsite interactions
        vector<double> flat_onsite_SU2;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t r = 0; r < spin_dim_SU2; ++r) {
                for (size_t c = 0; c < spin_dim_SU2; ++c) {
                    flat_onsite_SU2.push_back(onsite_interaction_SU2[i](r, c));
                }
            }
        }
        gpu_data.d_onsite_interaction_SU2 = thrust::device_vector<double>(flat_onsite_SU2.begin(), flat_onsite_SU2.end());
        
        // Transfer SU(3) onsite interactions
        vector<double> flat_onsite_SU3;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t r = 0; r < spin_dim_SU3; ++r) {
                for (size_t c = 0; c < spin_dim_SU3; ++c) {
                    flat_onsite_SU3.push_back(onsite_interaction_SU3[i](r, c));
                }
            }
        }
        gpu_data.d_onsite_interaction_SU3 = thrust::device_vector<double>(flat_onsite_SU3.begin(), flat_onsite_SU3.end());
        
        // Transfer SU(2) bilinear interactions
        vector<double> flat_bilinear_SU2;
        vector<size_t> flat_partners_SU2;
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t n = 0; n < num_bi_SU2; ++n) {
                if (n < bilinear_partners_SU2[i].size()) {
                    flat_partners_SU2.push_back(bilinear_partners_SU2[i][n]);
                    for (size_t r = 0; r < spin_dim_SU2; ++r) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            flat_bilinear_SU2.push_back(bilinear_interaction_SU2[i][n](r, c));
                        }
                    }
                }
            }
        }
        gpu_data.d_bilinear_interaction_SU2 = thrust::device_vector<double>(flat_bilinear_SU2.begin(), flat_bilinear_SU2.end());
        gpu_data.d_bilinear_partners_SU2 = thrust::device_vector<size_t>(flat_partners_SU2.begin(), flat_partners_SU2.end());
        
        // Transfer SU(3) bilinear interactions
        vector<double> flat_bilinear_SU3;
        vector<size_t> flat_partners_SU3;
        
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t n = 0; n < num_bi_SU3; ++n) {
                if (n < bilinear_partners_SU3[i].size()) {
                    flat_partners_SU3.push_back(bilinear_partners_SU3[i][n]);
                    for (size_t r = 0; r < spin_dim_SU3; ++r) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            flat_bilinear_SU3.push_back(bilinear_interaction_SU3[i][n](r, c));
                        }
                    }
                }
            }
        }
        gpu_data.d_bilinear_interaction_SU3 = thrust::device_vector<double>(flat_bilinear_SU3.begin(), flat_bilinear_SU3.end());
        gpu_data.d_bilinear_partners_SU3 = thrust::device_vector<size_t>(flat_partners_SU3.begin(), flat_partners_SU3.end());
        
        // Transfer mixed bilinear interactions
        vector<double> flat_mixed_bilinear;
        vector<size_t> flat_mixed_partners_SU2_list;
        vector<size_t> flat_mixed_partners_SU3_list;
        
        gpu_data.num_mixed_bi = num_bi_SU2_SU3;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t n = 0; n < mixed_bilinear_partners_SU2[i].size(); ++n) {
                flat_mixed_partners_SU2_list.push_back(i);
                flat_mixed_partners_SU3_list.push_back(mixed_bilinear_partners_SU2[i][n]);
                
                for (size_t r = 0; r < spin_dim_SU2; ++r) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        flat_mixed_bilinear.push_back(mixed_bilinear_interaction_SU2[i][n](r, c));
                    }
                }
            }
        }
        gpu_data.d_mixed_bilinear_interaction = thrust::device_vector<double>(flat_mixed_bilinear.begin(), flat_mixed_bilinear.end());
        gpu_data.d_mixed_bilinear_partners_SU2 = thrust::device_vector<size_t>(flat_mixed_partners_SU2_list.begin(), flat_mixed_partners_SU2_list.end());
        gpu_data.d_mixed_bilinear_partners_SU3 = thrust::device_vector<size_t>(flat_mixed_partners_SU3_list.begin(), flat_mixed_partners_SU3_list.end());
        
        // Store pulse parameters
        gpu_data.field_drive_amp_SU2 = field_drive_amp_SU2;
        gpu_data.field_drive_freq_SU2 = field_drive_freq_SU2;
        gpu_data.field_drive_width_SU2 = field_drive_width_SU2;
        gpu_data.t_pulse_0_SU2 = t_pulse_SU2[0];
        gpu_data.t_pulse_1_SU2 = t_pulse_SU2[1];
        
        gpu_data.field_drive_amp_SU3 = field_drive_amp_SU3;
        gpu_data.field_drive_freq_SU3 = field_drive_freq_SU3;
        gpu_data.field_drive_width_SU3 = field_drive_width_SU3;
        gpu_data.t_pulse_0_SU3 = t_pulse_SU3[0];
        gpu_data.t_pulse_1_SU3 = t_pulse_SU3[1];
        
        return gpu_data;
    }
    
    /**
     * Convert mixed lattice data to the new GPU format (mixed_gpu::GPUMixedLatticeData)
     * This provides a clean interface for the modular GPU implementation
     */
    mixed_gpu::GPUMixedLatticeData create_gpu_mixed_data() const {
        // Flatten SU(2) field
        std::vector<double> flat_field_SU2;
        flat_field_SU2.reserve(lattice_size_SU2 * spin_dim_SU2);
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t d = 0; d < spin_dim_SU2; ++d) {
                flat_field_SU2.push_back(field_SU2[i](d));
            }
        }
        
        // Flatten SU(2) onsite
        std::vector<double> flat_onsite_SU2;
        flat_onsite_SU2.reserve(lattice_size_SU2 * spin_dim_SU2 * spin_dim_SU2);
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t r = 0; r < spin_dim_SU2; ++r) {
                for (size_t c = 0; c < spin_dim_SU2; ++c) {
                    flat_onsite_SU2.push_back(onsite_interaction_SU2[i](r, c));
                }
            }
        }
        
        // Compute max bilinear neighbors for SU(2)
        size_t max_bi_SU2 = 0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            max_bi_SU2 = std::max(max_bi_SU2, bilinear_partners_SU2[i].size());
        }
        
        // Flatten SU(2) bilinear (padded to max_bi_SU2)
        std::vector<double> flat_bilinear_SU2;
        std::vector<size_t> flat_partners_SU2;
        std::vector<size_t> num_bi_per_site_SU2;
        flat_bilinear_SU2.reserve(lattice_size_SU2 * max_bi_SU2 * spin_dim_SU2 * spin_dim_SU2);
        flat_partners_SU2.reserve(lattice_size_SU2 * max_bi_SU2);
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            num_bi_per_site_SU2.push_back(bilinear_partners_SU2[i].size());
            for (size_t n = 0; n < max_bi_SU2; ++n) {
                if (n < bilinear_partners_SU2[i].size()) {
                    flat_partners_SU2.push_back(bilinear_partners_SU2[i][n]);
                    for (size_t r = 0; r < spin_dim_SU2; ++r) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            flat_bilinear_SU2.push_back(bilinear_interaction_SU2[i][n](r, c));
                        }
                    }
                } else {
                    flat_partners_SU2.push_back(SIZE_MAX);  // Invalid partner
                    for (size_t j = 0; j < spin_dim_SU2 * spin_dim_SU2; ++j) {
                        flat_bilinear_SU2.push_back(0.0);
                    }
                }
            }
        }
        
        // Flatten SU(3) field
        std::vector<double> flat_field_SU3;
        flat_field_SU3.reserve(lattice_size_SU3 * spin_dim_SU3);
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t d = 0; d < spin_dim_SU3; ++d) {
                flat_field_SU3.push_back(field_SU3[i](d));
            }
        }
        
        // Flatten SU(3) onsite
        std::vector<double> flat_onsite_SU3;
        flat_onsite_SU3.reserve(lattice_size_SU3 * spin_dim_SU3 * spin_dim_SU3);
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t r = 0; r < spin_dim_SU3; ++r) {
                for (size_t c = 0; c < spin_dim_SU3; ++c) {
                    flat_onsite_SU3.push_back(onsite_interaction_SU3[i](r, c));
                }
            }
        }
        
        // Compute max bilinear neighbors for SU(3)
        size_t max_bi_SU3 = 0;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            max_bi_SU3 = std::max(max_bi_SU3, bilinear_partners_SU3[i].size());
        }
        
        // Flatten SU(3) bilinear
        std::vector<double> flat_bilinear_SU3;
        std::vector<size_t> flat_partners_SU3;
        std::vector<size_t> num_bi_per_site_SU3;
        flat_bilinear_SU3.reserve(lattice_size_SU3 * max_bi_SU3 * spin_dim_SU3 * spin_dim_SU3);
        flat_partners_SU3.reserve(lattice_size_SU3 * max_bi_SU3);
        
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            num_bi_per_site_SU3.push_back(bilinear_partners_SU3[i].size());
            for (size_t n = 0; n < max_bi_SU3; ++n) {
                if (n < bilinear_partners_SU3[i].size()) {
                    flat_partners_SU3.push_back(bilinear_partners_SU3[i][n]);
                    for (size_t r = 0; r < spin_dim_SU3; ++r) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            flat_bilinear_SU3.push_back(bilinear_interaction_SU3[i][n](r, c));
                        }
                    }
                } else {
                    flat_partners_SU3.push_back(SIZE_MAX);
                    for (size_t j = 0; j < spin_dim_SU3 * spin_dim_SU3; ++j) {
                        flat_bilinear_SU3.push_back(0.0);
                    }
                }
            }
        }
        
        // Compute max mixed bilinear neighbors from SU2 perspective
        size_t max_mixed_bi = 0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            max_mixed_bi = std::max(max_mixed_bi, mixed_bilinear_partners_SU2[i].size());
        }
        
        // Compute max mixed bilinear neighbors from SU3 perspective
        size_t max_mixed_bi_SU3 = 0;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            max_mixed_bi_SU3 = std::max(max_mixed_bi_SU3, mixed_bilinear_partners_SU3[i].size());
        }
        
        // Flatten mixed bilinear from SU2 perspective (3x8 matrices)
        std::vector<double> flat_mixed_bilinear;
        std::vector<size_t> flat_mixed_partners_SU2;
        std::vector<size_t> flat_mixed_partners_SU3;
        std::vector<size_t> num_mixed_per_site_SU2;
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            num_mixed_per_site_SU2.push_back(mixed_bilinear_partners_SU2[i].size());
            for (size_t n = 0; n < max_mixed_bi; ++n) {
                if (n < mixed_bilinear_partners_SU2[i].size()) {
                    flat_mixed_partners_SU2.push_back(i);
                    flat_mixed_partners_SU3.push_back(mixed_bilinear_partners_SU2[i][n]);
                    for (size_t r = 0; r < spin_dim_SU2; ++r) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            flat_mixed_bilinear.push_back(mixed_bilinear_interaction_SU2[i][n](r, c));
                        }
                    }
                } else {
                    flat_mixed_partners_SU2.push_back(SIZE_MAX);
                    flat_mixed_partners_SU3.push_back(SIZE_MAX);
                    for (size_t j = 0; j < spin_dim_SU2 * spin_dim_SU3; ++j) {
                        flat_mixed_bilinear.push_back(0.0);
                    }
                }
            }
        }
        
        // Flatten mixed bilinear from SU3 perspective (8x3 matrices)
        std::vector<double> flat_mixed_bilinear_SU3;
        std::vector<size_t> flat_mixed_partners_SU2_from_SU3;
        std::vector<size_t> num_mixed_per_site_SU3;
        
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            num_mixed_per_site_SU3.push_back(mixed_bilinear_partners_SU3[i].size());
            for (size_t n = 0; n < max_mixed_bi_SU3; ++n) {
                if (n < mixed_bilinear_partners_SU3[i].size()) {
                    flat_mixed_partners_SU2_from_SU3.push_back(mixed_bilinear_partners_SU3[i][n]);
                    // mixed_bilinear_interaction_SU3[i][n] is 8x3 (spin_dim_SU3 x spin_dim_SU2) [transposed from storage]
                    for (size_t r = 0; r < spin_dim_SU3; ++r) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            flat_mixed_bilinear_SU3.push_back(mixed_bilinear_interaction_SU3[i][n](r, c));
                        }
                    }
                } else {
                    flat_mixed_partners_SU2_from_SU3.push_back(SIZE_MAX);
                    for (size_t j = 0; j < spin_dim_SU3 * spin_dim_SU2; ++j) {
                        flat_mixed_bilinear_SU3.push_back(0.0);
                    }
                }
            }
        }
        
        return mixed_gpu::create_gpu_mixed_lattice_data(
            lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
            lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
            max_bi_SU2, max_bi_SU3, max_mixed_bi, max_mixed_bi_SU3,
            flat_field_SU2, flat_onsite_SU2, flat_bilinear_SU2, flat_partners_SU2, num_bi_per_site_SU2,
            flat_field_SU3, flat_onsite_SU3, flat_bilinear_SU3, flat_partners_SU3, num_bi_per_site_SU3,
            flat_mixed_bilinear, flat_mixed_partners_SU2, flat_mixed_partners_SU3, num_mixed_per_site_SU2,
            flat_mixed_bilinear_SU3, flat_mixed_partners_SU2_from_SU3, num_mixed_per_site_SU3
        );
    }
    
    /**
     * GPU ODE system function - DEPRECATED, use mixed_gpu::GPUMixedODESystem instead
     * Kept for backward compatibility
     */
    void ode_system_gpu(const thrust::device_vector<double>& x, 
                       thrust::device_vector<double>& dxdt, 
                       double t,
                       const GPUMixedLatticeData& d_data) const {
        // This is now a legacy wrapper - the real implementation is in mixed_lattice_gpu.cu
        // For backward compatibility, we still support the old interface
        thrust::host_vector<double> h_x = x;
        thrust::host_vector<double> h_dxdt(x.size());
        
        ODEState x_state(h_x.begin(), h_x.end());
        ODEState dxdt_state(h_dxdt.size());
        
        const_cast<MixedLattice*>(this)->landau_lifshitz(x_state, dxdt_state, t);
        
        std::copy(dxdt_state.begin(), dxdt_state.end(), h_dxdt.begin());
        dxdt = h_dxdt;
    }
    
    /**
     * GPU integration - pure GPU implementation without host-device ping-pong
     * 
     * Uses the modular mixed_gpu:: implementation from mixed_lattice_gpu.cuh/cu
     * This keeps all state on GPU during integration, only copying back at save intervals.
     * 
     * @param state Device state vector (modified in-place)
     * @param T_start Start time
     * @param T_end End time
     * @param dt_step Time step
     * @param observer Observer function called at save intervals
     * @param method Integration method: "euler", "rk2", "rk4", "dopri5", "ssprk53", etc.
     */
    template<typename Observer>
    void integrate_pure_gpu(mixed_gpu::GPUMixedLatticeData& gpu_data,
                            mixed_gpu::GPUState& state,
                            double T_start, double T_end, double dt_step,
                            Observer observer, const std::string& method) {
        mixed_gpu::GPUMixedODESystem system(gpu_data);
        
        double t = T_start;
        size_t step = 0;
        
        while (t < T_end) {
            // Call observer at each step
            observer(state, t);
            
            // Perform one integration step
            mixed_gpu::step_mixed_gpu(system, state, t, dt_step, method);
            
            t += dt_step;
            step++;
        }
        
        // Final observer call
        observer(state, t);
    }
    
    /**
     * GPU integration wrapper - DEPRECATED hybrid CPU/GPU approach
     * 
     * Use integrate_pure_gpu() for better performance.
     * This method is kept for backward compatibility.
     */
    template<typename System, typename Observer>
    void integrate_ode_system_gpu(System system_func, thrust::device_vector<double>& state,
                                  double T_start, double T_end, double dt_step,
                                  Observer observer, const string& method,
                                  bool use_adaptive = false,
                                  double abs_tol = 1e-6, double rel_tol = 1e-6) {
        // Copy initial state to host once
        thrust::host_vector<double> h_state = state;
        ODEState cpu_state(h_state.begin(), h_state.end());
        
        // Pre-allocate device vectors to avoid repeated allocations
        thrust::device_vector<double> d_x(cpu_state.size());
        thrust::device_vector<double> d_dxdt(cpu_state.size());
        
        // System wrapper: transfers state, evaluates on GPU, transfers derivatives
        auto cpu_system = [&](const ODEState& x, ODEState& dxdt, double t) {
            // Transfer current state to GPU
            thrust::copy(x.begin(), x.end(), d_x.begin());
            // Evaluate system function on GPU
            system_func(d_x, d_dxdt, t);
            // Transfer derivatives back to CPU
            thrust::copy(d_dxdt.begin(), d_dxdt.end(), dxdt.begin());
        };
        
        // Observer wrapper
        auto cpu_observer = [&](const ODEState& x, double t) {
            thrust::copy(x.begin(), x.end(), d_x.begin());
            observer(d_x, t);
        };
        
        // Perform integration on CPU with GPU-evaluated derivatives
        integrate_ode_system(cpu_system, cpu_state, T_start, T_end, dt_step,
                            cpu_observer, method, use_adaptive, abs_tol, rel_tol);
        
        // Copy final state back to device
        thrust::copy(cpu_state.begin(), cpu_state.end(), state.begin());
    }
    
    /**
     * GPU version of molecular_dynamics - uses pure GPU integration
     */
    void molecular_dynamics_gpu(double T_start, double T_end, double dt_initial,
                               const string& out_dir = "", size_t save_interval = 100,
                               const string& method = "dopri5") {
#ifndef HDF5_ENABLED
        std::cerr << "Error: HDF5 support is required for molecular dynamics output." << endl;
        std::cerr << "Please rebuild with -DHDF5_ENABLED flag and HDF5 libraries." << endl;
        return;
#else
        ensure_directory_exists(out_dir);
        
        cout << "Running mixed lattice molecular dynamics with pure GPU acceleration: t=" << T_start << " → " << T_end << endl;
        cout << "Integration method: " << method << endl;
        cout << "Initial step size: " << dt_initial << endl;
        
        // Create GPU mixed lattice data using the new modular implementation
        auto gpu_data = create_gpu_mixed_data();
        
        // Transfer initial state to GPU
        ODEState state = spins_to_state();
        mixed_gpu::GPUState d_state(state.begin(), state.end());
        
        // Create HDF5 writer with comprehensive metadata
        std::unique_ptr<HDF5MixedMDWriter> hdf5_writer;
        if (!out_dir.empty()) {
            string hdf5_file = out_dir + "/trajectory.h5";
            cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
            hdf5_writer = std::make_unique<HDF5MixedMDWriter>(
                hdf5_file, 
                lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
                lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
                dim1, dim2, dim3, method + "_gpu", 
                dt_initial, T_start, T_end, save_interval, 
                spin_length_SU2, spin_length_SU3,
                &site_positions_SU2, &site_positions_SU3, 10000);
        }
        
        // Observer for saving data (called at each step)
        size_t step_count = 0;
        size_t save_count = 0;
        thrust::host_vector<double> h_state;
        
        auto observer = [&](const mixed_gpu::GPUState& d_x, double t) {
            if (step_count % save_interval == 0) {
                // Copy state back to host for I/O (only at save intervals)
                h_state = d_x;
                
                // Compute magnetizations directly from flat state
                SpinVector M_SU2 = SpinVector::Zero(spin_dim_SU2);
                SpinVector M_SU3 = SpinVector::Zero(spin_dim_SU3);
                SpinVector M_SU2_antiferro = SpinVector::Zero(spin_dim_SU2);
                SpinVector M_SU3_antiferro = SpinVector::Zero(spin_dim_SU3);
                
                double M_SU2_arr[8] = {0};
                double M_SU2_antiferro_arr[8] = {0};
                compute_sublattice_magnetizations_from_flat(thrust::raw_pointer_cast(h_state.data()), 0, 
                    lattice_size_SU2, spin_dim_SU2, M_SU2_arr, M_SU2_antiferro_arr);
                compute_magnetization_staggered_SU2_from_flat(thrust::raw_pointer_cast(h_state.data()), M_SU2_antiferro_arr);
                M_SU2 = Eigen::Map<Eigen::VectorXd>(M_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
                M_SU2_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU2_antiferro_arr, spin_dim_SU2);
                
                double M_SU3_arr[8] = {0};
                double M_SU3_antiferro_arr[8] = {0};
                size_t SU3_offset = lattice_size_SU2 * spin_dim_SU2;
                compute_sublattice_magnetizations_from_flat(thrust::raw_pointer_cast(h_state.data()), SU3_offset, 
                    lattice_size_SU3, spin_dim_SU3, M_SU3_arr, M_SU3_antiferro_arr);
                M_SU3 = Eigen::Map<Eigen::VectorXd>(M_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
                M_SU3_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU3_antiferro_arr, spin_dim_SU3) / double(lattice_size_SU3);
                
                // Compute energy density
                double E = total_energy_flat(thrust::raw_pointer_cast(h_state.data())) / (lattice_size_SU2 + lattice_size_SU3);
                
                // Write to HDF5
                if (hdf5_writer) {
                    hdf5_writer->write_flat_step(t, 
                                                M_SU2_antiferro, M_SU2, 
                                                M_SU3_antiferro, M_SU3,
                                                thrust::raw_pointer_cast(h_state.data()));
                    save_count++;
                }
                
                // Progress output
                if (step_count % (save_interval * 10) == 0) {
                    cout << "t=" << t << ", E/N=" << E 
                         << ", |M_SU2|=" << M_SU2.norm() 
                         << ", |M_SU3|=" << M_SU3.norm() << endl;
                }
            }
            ++step_count;
        };
        
        // Use the new pure GPU integration (no host-device ping-pong)
        integrate_pure_gpu(gpu_data, d_state, T_start, T_end, dt_initial, observer, method);
        
        // Note: MixedLattice::spins_SU2 and spins_SU3 remain unchanged (initial configuration preserved)
        // The evolved state is stored in the device vector 'd_state'
        
        // Close HDF5 file
        if (hdf5_writer) {
            hdf5_writer->close();
            cout << "HDF5 trajectory saved with " << save_count << " snapshots" << endl;
        }
        
        cout << "GPU molecular dynamics complete! (" << step_count << " steps)" << endl;
#endif // HDF5_ENABLED
    }
    
    /**
     * GPU version of single_pulse_drive for mixed lattice
     * Returns nested pair: outer=(SU2_results, SU3_results), 
     * inner for each = (M_antiferro, M_local)
     */
    vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>>
    single_pulse_drive_gpu(const vector<SpinVector>& field_in_SU2,
              const vector<SpinVector>& field_in_SU3,
              double t_B,
              double pulse_amp_SU2_in, double pulse_width_SU2_in, double pulse_freq_SU2_in,
              double pulse_amp_SU3_in, double pulse_width_SU3_in, double pulse_freq_SU3_in,
              double T_start, double T_end, double step_size,
              string method = "dopri5") {
        
        // Set up pulses for both sublattices (on CPU side first)
        set_pulse_SU2(field_in_SU2, t_B, 
                     vector<SpinVector>(N_atoms_SU2, SpinVector::Zero(spin_dim_SU2)), 0.0,
                     pulse_amp_SU2_in, pulse_width_SU2_in, pulse_freq_SU2_in);
        
        set_pulse_SU3(field_in_SU3, t_B,
                     vector<SpinVector>(N_atoms_SU3, SpinVector::Zero(spin_dim_SU3)), 0.0,
                     pulse_amp_SU3_in, pulse_width_SU3_in, pulse_freq_SU3_in);
        
        // Ensure GPU data is initialized and update pulse parameters
        ensure_gpu_mixed_data_initialized();
        update_gpu_pulse_SU2();
        update_gpu_pulse_SU3();
        
        // Storage for trajectory
        vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>> trajectory;
        
        // Initial state on GPU
        ODEState state = spins_to_state();
        mixed_gpu::GPUState d_state(state.begin(), state.end());
        
        // Pure GPU integration (all computation on device, only copy back at save intervals)
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        mixed_gpu::GPUMixedODESystem gpu_system(gpu_mixed_data_cache_);
        
        // Calculate save interval (save every step for this function)
        size_t save_interval = 1;
        mixed_gpu::integrate_mixed_gpu(gpu_system, d_state, T_start, T_end, step_size,
                                       save_interval, raw_trajectory, method);
        
        // Convert raw trajectory to magnetization trajectory (post-processing on CPU)
        for (const auto& [t, state_vec] : raw_trajectory) {
            size_t total_SU2 = lattice_size_SU2 * spin_dim_SU2;
            
            // Compute SU(2) magnetizations
            double M_local_SU2_arr[8] = {0};
            double M_antiferro_SU2_arr[8] = {0};
            double M_global_SU2_arr[8] = {0};
            
            compute_sublattice_magnetizations_from_flat(state_vec.data(), 0, 
                lattice_size_SU2, spin_dim_SU2, M_local_SU2_arr, M_antiferro_SU2_arr);
            
            // Transform to global frame using sublattice frame
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                size_t atom = i % N_atoms_SU2;
                for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
                    for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
                        M_global_SU2_arr[mu] += sublattice_frames_SU2[atom](nu, mu) * state_vec[i * spin_dim_SU2 + nu];
                    }
                }
            }
            
            SpinVector M_local_SU2 = Eigen::Map<Eigen::VectorXd>(M_local_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            SpinVector M_antiferro_SU2 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            SpinVector M_global_SU2 = Eigen::Map<Eigen::VectorXd>(M_global_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            
            // Compute SU(3) magnetizations
            double M_local_SU3_arr[8] = {0};
            double M_antiferro_SU3_arr[8] = {0};
            double M_global_SU3_arr[8] = {0};
            
            compute_sublattice_magnetizations_from_flat(state_vec.data(), total_SU2, 
                lattice_size_SU3, spin_dim_SU3, M_local_SU3_arr, M_antiferro_SU3_arr);
                
            // Transform to global frame using sublattice frame
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                size_t atom = i % N_atoms_SU3;
                for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
                    for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
                        M_global_SU3_arr[mu] += sublattice_frames_SU3[atom](nu, mu) * state_vec[total_SU2 + i * spin_dim_SU3 + nu];
                    }
                }
            }
            
            SpinVector M_local_SU3 = Eigen::Map<Eigen::VectorXd>(M_local_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            SpinVector M_antiferro_SU3 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            SpinVector M_global_SU3 = Eigen::Map<Eigen::VectorXd>(M_global_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            
            trajectory.push_back({t, {{M_antiferro_SU2, M_local_SU2, M_global_SU2}, {M_antiferro_SU3, M_local_SU3, M_global_SU3}}});
        }
        
        // Reset pulses
        field_drive_SU2[0] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_SU2[1] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_amp_SU2 = 0.0;
        
        field_drive_SU3[0] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_SU3[1] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_amp_SU3 = 0.0;
        
        return trajectory;
    }
    
    /**
     * GPU version of double_pulse_drive for mixed lattice
     * Uses pure GPU integration (all computation on device, only copy back for post-processing)
     */
    vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>>
    double_pulse_drive_gpu(const vector<SpinVector>& field_in_1_SU2,
                  const vector<SpinVector>& field_in_1_SU3,
                  double t_B_1,
                  const vector<SpinVector>& field_in_2_SU2,
                  const vector<SpinVector>& field_in_2_SU3,
                  double t_B_2,
                  double pulse_amp_SU2_in, double pulse_width_SU2_in, double pulse_freq_SU2_in,
                  double pulse_amp_SU3_in, double pulse_width_SU3_in, double pulse_freq_SU3_in,
                  double T_start, double T_end, double step_size,
                  string method = "dopri5") {
        
        // Set up two-pulse configuration for both sublattices (on CPU side first)
        set_pulse_SU2(field_in_1_SU2, t_B_1, field_in_2_SU2, t_B_2,
                     pulse_amp_SU2_in, pulse_width_SU2_in, pulse_freq_SU2_in);
        
        set_pulse_SU3(field_in_1_SU3, t_B_1, field_in_2_SU3, t_B_2,
                     pulse_amp_SU3_in, pulse_width_SU3_in, pulse_freq_SU3_in);
        
        // Ensure GPU data is initialized and update pulse parameters
        ensure_gpu_mixed_data_initialized();
        update_gpu_pulse_SU2();
        update_gpu_pulse_SU3();
        
        // Storage for trajectory
        vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>> trajectory;
        
        // Initial state on GPU
        ODEState state = spins_to_state();
        mixed_gpu::GPUState d_state(state.begin(), state.end());
        
        // Pure GPU integration (all computation on device)
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        mixed_gpu::GPUMixedODESystem gpu_system(gpu_mixed_data_cache_);
        
        // Calculate save interval (save every step for this function)
        size_t save_interval = 1;
        mixed_gpu::integrate_mixed_gpu(gpu_system, d_state, T_start, T_end, step_size,
                                       save_interval, raw_trajectory, method);
        
        // Convert raw trajectory to magnetization trajectory (post-processing on CPU)
        for (const auto& [t, state_vec] : raw_trajectory) {
            size_t total_SU2 = lattice_size_SU2 * spin_dim_SU2;
            
            // Compute SU(2) magnetizations
            double M_local_SU2_arr[8] = {0};
            double M_antiferro_SU2_arr[8] = {0};
            double M_global_SU2_arr[8] = {0};
            
            compute_sublattice_magnetizations_from_flat(state_vec.data(), 0, 
                lattice_size_SU2, spin_dim_SU2, M_local_SU2_arr, M_antiferro_SU2_arr);
            
            // Transform to global frame using sublattice frame
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                size_t atom = i % N_atoms_SU2;
                for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
                    for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
                        M_global_SU2_arr[mu] += sublattice_frames_SU2[atom](nu, mu) * state_vec[i * spin_dim_SU2 + nu];
                    }
                }
            }
            
            SpinVector M_local_SU2 = Eigen::Map<Eigen::VectorXd>(M_local_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            SpinVector M_antiferro_SU2 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            SpinVector M_global_SU2 = Eigen::Map<Eigen::VectorXd>(M_global_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            
            // Compute SU(3) magnetizations
            double M_local_SU3_arr[8] = {0};
            double M_antiferro_SU3_arr[8] = {0};
            double M_global_SU3_arr[8] = {0};
            
            compute_sublattice_magnetizations_from_flat(state_vec.data(), total_SU2, 
                lattice_size_SU3, spin_dim_SU3, M_local_SU3_arr, M_antiferro_SU3_arr);
                
            // Transform to global frame using sublattice frame
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                size_t atom = i % N_atoms_SU3;
                for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
                    for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
                        M_global_SU3_arr[mu] += sublattice_frames_SU3[atom](nu, mu) * state_vec[total_SU2 + i * spin_dim_SU3 + nu];
                    }
                }
            }
            
            SpinVector M_local_SU3 = Eigen::Map<Eigen::VectorXd>(M_local_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            SpinVector M_antiferro_SU3 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            SpinVector M_global_SU3 = Eigen::Map<Eigen::VectorXd>(M_global_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            
            trajectory.push_back({t, {{M_antiferro_SU2, M_local_SU2, M_global_SU2}, {M_antiferro_SU3, M_local_SU3, M_global_SU3}}});
        }
        
        // Reset pulses
        field_drive_SU2[0] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_SU2[1] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_amp_SU2 = 0.0;
        
        field_drive_SU3[0] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_SU3[1] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_amp_SU3 = 0.0;
        
        return trajectory;
    }
#endif // defined(CUDA_ENABLED) && defined(__CUDACC__)

// =============================================================================
// GPU Implementation using opaque API (for C++ TUs compiled with g++)
// This section is used when CUDA_ENABLED but not compiling with NVCC
// =============================================================================
#if defined(CUDA_ENABLED) && !defined(__CUDACC__)
private:
    // GPU data handle (opaque pointer managed by CUDA library)
    mutable mixed_gpu::GPUMixedLatticeDataHandle* gpu_mixed_handle_ = nullptr;
    mutable bool gpu_mixed_data_initialized_ = false;
    
    /**
     * Flatten SU(2) sublattice data for GPU transfer
     */
    void flatten_SU2_data(
        vector<double>& flat_field,
        vector<double>& flat_onsite,
        vector<double>& flat_bilinear,
        vector<size_t>& flat_partners,
        vector<size_t>& num_bilinear_per_site
    ) const {
        flat_field.clear();
        flat_field.reserve(lattice_size_SU2 * spin_dim_SU2);
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t d = 0; d < spin_dim_SU2; ++d) {
                flat_field.push_back(field_SU2[i](d));
            }
        }
        
        flat_onsite.clear();
        flat_onsite.reserve(lattice_size_SU2 * spin_dim_SU2 * spin_dim_SU2);
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t r = 0; r < spin_dim_SU2; ++r) {
                for (size_t c = 0; c < spin_dim_SU2; ++c) {
                    flat_onsite.push_back(onsite_interaction_SU2[i](r, c));
                }
            }
        }
        
        flat_bilinear.clear();
        flat_partners.clear();
        num_bilinear_per_site.clear();
        flat_bilinear.reserve(lattice_size_SU2 * num_bi_SU2 * spin_dim_SU2 * spin_dim_SU2);
        flat_partners.reserve(lattice_size_SU2 * num_bi_SU2);
        num_bilinear_per_site.reserve(lattice_size_SU2);
        
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            num_bilinear_per_site.push_back(bilinear_partners_SU2[i].size());
            for (size_t n = 0; n < num_bi_SU2; ++n) {
                if (n < bilinear_partners_SU2[i].size()) {
                    flat_partners.push_back(bilinear_partners_SU2[i][n]);
                    for (size_t r = 0; r < spin_dim_SU2; ++r) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            flat_bilinear.push_back(bilinear_interaction_SU2[i][n](r, c));
                        }
                    }
                } else {
                    flat_partners.push_back(0);
                    for (size_t j = 0; j < spin_dim_SU2 * spin_dim_SU2; ++j) {
                        flat_bilinear.push_back(0.0);
                    }
                }
            }
        }
    }
    
    /**
     * Flatten SU(3) sublattice data for GPU transfer
     */
    void flatten_SU3_data(
        vector<double>& flat_field,
        vector<double>& flat_onsite,
        vector<double>& flat_bilinear,
        vector<size_t>& flat_partners,
        vector<size_t>& num_bilinear_per_site
    ) const {
        flat_field.clear();
        flat_field.reserve(lattice_size_SU3 * spin_dim_SU3);
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t d = 0; d < spin_dim_SU3; ++d) {
                flat_field.push_back(field_SU3[i](d));
            }
        }
        
        flat_onsite.clear();
        flat_onsite.reserve(lattice_size_SU3 * spin_dim_SU3 * spin_dim_SU3);
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            for (size_t r = 0; r < spin_dim_SU3; ++r) {
                for (size_t c = 0; c < spin_dim_SU3; ++c) {
                    flat_onsite.push_back(onsite_interaction_SU3[i](r, c));
                }
            }
        }
        
        flat_bilinear.clear();
        flat_partners.clear();
        num_bilinear_per_site.clear();
        flat_bilinear.reserve(lattice_size_SU3 * num_bi_SU3 * spin_dim_SU3 * spin_dim_SU3);
        flat_partners.reserve(lattice_size_SU3 * num_bi_SU3);
        num_bilinear_per_site.reserve(lattice_size_SU3);
        
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            num_bilinear_per_site.push_back(bilinear_partners_SU3[i].size());
            for (size_t n = 0; n < num_bi_SU3; ++n) {
                if (n < bilinear_partners_SU3[i].size()) {
                    flat_partners.push_back(bilinear_partners_SU3[i][n]);
                    for (size_t r = 0; r < spin_dim_SU3; ++r) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            flat_bilinear.push_back(bilinear_interaction_SU3[i][n](r, c));
                        }
                    }
                } else {
                    flat_partners.push_back(0);
                    for (size_t j = 0; j < spin_dim_SU3 * spin_dim_SU3; ++j) {
                        flat_bilinear.push_back(0.0);
                    }
                }
            }
        }
    }
    
    /**
     * Ensure GPU mixed lattice data is initialized (lazy initialization)
     */
    void ensure_gpu_mixed_data_initialized() const {
        if (gpu_mixed_data_initialized_) return;
        
        // Flatten SU(2) data
        vector<double> flat_field_SU2, flat_onsite_SU2, flat_bilinear_SU2;
        vector<size_t> flat_partners_SU2, num_bi_per_site_SU2;
        flatten_SU2_data(flat_field_SU2, flat_onsite_SU2, flat_bilinear_SU2,
                        flat_partners_SU2, num_bi_per_site_SU2);
        
        // Flatten SU(3) data
        vector<double> flat_field_SU3, flat_onsite_SU3, flat_bilinear_SU3;
        vector<size_t> flat_partners_SU3, num_bi_per_site_SU3;
        flatten_SU3_data(flat_field_SU3, flat_onsite_SU3, flat_bilinear_SU3,
                        flat_partners_SU3, num_bi_per_site_SU3);
        
        // Compute max mixed bilinear neighbors
        size_t max_mixed_bi = 0;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            max_mixed_bi = std::max(max_mixed_bi, mixed_bilinear_partners_SU2[i].size());
        }
        size_t max_mixed_bi_SU3 = 0;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            max_mixed_bi_SU3 = std::max(max_mixed_bi_SU3, mixed_bilinear_partners_SU3[i].size());
        }
        
        // Flatten mixed bilinear from SU2 perspective (3x8 matrices)
        vector<double> flat_mixed_bilinear;
        vector<size_t> flat_mixed_partners_SU2, flat_mixed_partners_SU3, num_mixed_per_site_SU2;
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            num_mixed_per_site_SU2.push_back(mixed_bilinear_partners_SU2[i].size());
            for (size_t n = 0; n < max_mixed_bi; ++n) {
                if (n < mixed_bilinear_partners_SU2[i].size()) {
                    flat_mixed_partners_SU2.push_back(i);
                    flat_mixed_partners_SU3.push_back(mixed_bilinear_partners_SU2[i][n]);
                    for (size_t r = 0; r < spin_dim_SU2; ++r) {
                        for (size_t c = 0; c < spin_dim_SU3; ++c) {
                            flat_mixed_bilinear.push_back(mixed_bilinear_interaction_SU2[i][n](r, c));
                        }
                    }
                } else {
                    flat_mixed_partners_SU2.push_back(SIZE_MAX);
                    flat_mixed_partners_SU3.push_back(SIZE_MAX);
                    for (size_t j = 0; j < spin_dim_SU2 * spin_dim_SU3; ++j) {
                        flat_mixed_bilinear.push_back(0.0);
                    }
                }
            }
        }
        
        // Flatten mixed bilinear from SU3 perspective (8x3 matrices)
        vector<double> flat_mixed_bilinear_SU3;
        vector<size_t> flat_mixed_partners_SU2_from_SU3, num_mixed_per_site_SU3;
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            num_mixed_per_site_SU3.push_back(mixed_bilinear_partners_SU3[i].size());
            for (size_t n = 0; n < max_mixed_bi_SU3; ++n) {
                if (n < mixed_bilinear_partners_SU3[i].size()) {
                    flat_mixed_partners_SU2_from_SU3.push_back(mixed_bilinear_partners_SU3[i][n]);
                    for (size_t r = 0; r < spin_dim_SU3; ++r) {
                        for (size_t c = 0; c < spin_dim_SU2; ++c) {
                            flat_mixed_bilinear_SU3.push_back(mixed_bilinear_interaction_SU3[i][n](r, c));
                        }
                    }
                } else {
                    flat_mixed_partners_SU2_from_SU3.push_back(SIZE_MAX);
                    for (size_t j = 0; j < spin_dim_SU3 * spin_dim_SU2; ++j) {
                        flat_mixed_bilinear_SU3.push_back(0.0);
                    }
                }
            }
        }
        
        // Create GPU handle
        gpu_mixed_handle_ = mixed_gpu::create_gpu_mixed_lattice_data(
            lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
            lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
            num_bi_SU2, num_bi_SU3, max_mixed_bi, max_mixed_bi_SU3,
            flat_field_SU2, flat_onsite_SU2, flat_bilinear_SU2,
            flat_partners_SU2, num_bi_per_site_SU2,
            flat_field_SU3, flat_onsite_SU3, flat_bilinear_SU3,
            flat_partners_SU3, num_bi_per_site_SU3,
            flat_mixed_bilinear, flat_mixed_partners_SU2,
            flat_mixed_partners_SU3, num_mixed_per_site_SU2,
            flat_mixed_bilinear_SU3, flat_mixed_partners_SU2_from_SU3, num_mixed_per_site_SU3
        );
        
        gpu_mixed_data_initialized_ = true;
    }
    
    /**
     * Update GPU pulse parameters for SU(2)
     */
    void update_gpu_pulse_SU2() const {
        if (!gpu_mixed_handle_) return;
        
        vector<double> flat_field_drive;
        flat_field_drive.reserve(2 * N_atoms_SU2 * spin_dim_SU2);
        for (size_t p = 0; p < 2; ++p) {
            for (size_t d = 0; d < field_drive_SU2[p].size(); ++d) {
                flat_field_drive.push_back(field_drive_SU2[p](d));
            }
        }
        
        mixed_gpu::set_gpu_pulse_SU2(
            gpu_mixed_handle_,
            flat_field_drive,
            field_drive_amp_SU2,
            field_drive_width_SU2,
            field_drive_freq_SU2,
            t_pulse_SU2[0],
            t_pulse_SU2[1]
        );
    }
    
    /**
     * Update GPU pulse parameters for SU(3)
     */
    void update_gpu_pulse_SU3() const {
        if (!gpu_mixed_handle_) return;
        
        vector<double> flat_field_drive;
        flat_field_drive.reserve(2 * N_atoms_SU3 * spin_dim_SU3);
        for (size_t p = 0; p < 2; ++p) {
            for (size_t d = 0; d < field_drive_SU3[p].size(); ++d) {
                flat_field_drive.push_back(field_drive_SU3[p](d));
            }
        }
        
        mixed_gpu::set_gpu_pulse_SU3(
            gpu_mixed_handle_,
            flat_field_drive,
            field_drive_amp_SU3,
            field_drive_width_SU3,
            field_drive_freq_SU3,
            t_pulse_SU3[0],
            t_pulse_SU3[1]
        );
    }
    
    /**
     * GPU version of molecular_dynamics using opaque API
     */
    void molecular_dynamics_gpu(double T_start, double T_end, double dt_initial,
                               const string& out_dir = "", size_t save_interval = 100,
                               const string& method = "dopri5") {
#ifndef HDF5_ENABLED
        std::cerr << "Error: HDF5 support is required for molecular dynamics output." << endl;
        return;
#else
        ensure_directory_exists(out_dir);
        
        cout << "Running mixed lattice molecular dynamics with GPU acceleration (API): t=" << T_start << " → " << T_end << endl;
        cout << "Integration method: " << method << endl;
        cout << "Step size: " << dt_initial << endl;
        
        // Ensure GPU data is initialized
        ensure_gpu_mixed_data_initialized();
        
        // Transfer initial state to GPU
        ODEState h_state = spins_to_state();
        mixed_gpu::set_gpu_mixed_spins(gpu_mixed_handle_, h_state);
        
        // Create HDF5 writer
        std::unique_ptr<HDF5MixedMDWriter> hdf5_writer;
        if (!out_dir.empty()) {
            string hdf5_file = out_dir + "/trajectory.h5";
            cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
            hdf5_writer = std::make_unique<HDF5MixedMDWriter>(
                hdf5_file, 
                lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
                lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
                dim1, dim2, dim3, method + "_gpu_api", 
                dt_initial, T_start, T_end, save_interval, 
                spin_length_SU2, spin_length_SU3,
                &site_positions_SU2, &site_positions_SU3, 10000);
        }
        
        // Integrate on GPU
        std::vector<std::pair<double, std::vector<double>>> trajectory;
        mixed_gpu::integrate_mixed_gpu(gpu_mixed_handle_, T_start, T_end, dt_initial, 
                                       save_interval, trajectory, method);
        
        // Write trajectory to HDF5
        size_t save_count = 0;
        for (const auto& [t, state_vec] : trajectory) {
            double M_SU2_arr[8] = {0}, M_SU2_antiferro_arr[8] = {0}, M_SU2_global_arr[8] = {0};
            double M_SU3_arr[8] = {0}, M_SU3_antiferro_arr[8] = {0}, M_SU3_global_arr[8] = {0};
            
            compute_sublattice_magnetizations_from_flat(state_vec.data(), 0, 
                lattice_size_SU2, spin_dim_SU2, M_SU2_arr, M_SU2_antiferro_arr);
            compute_magnetization_global_SU2_from_flat(state_vec.data(), M_SU2_global_arr);
            compute_magnetization_staggered_SU2_from_flat(state_vec.data(), M_SU2_antiferro_arr);
            
            size_t SU3_offset = lattice_size_SU2 * spin_dim_SU2;
            compute_sublattice_magnetizations_from_flat(state_vec.data(), SU3_offset, 
                lattice_size_SU3, spin_dim_SU3, M_SU3_arr, M_SU3_antiferro_arr);
            compute_magnetization_global_SU3_from_flat(state_vec.data(), M_SU3_global_arr);
            
            SpinVector M_SU2 = Eigen::Map<Eigen::VectorXd>(M_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            SpinVector M_SU2_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU2_antiferro_arr, spin_dim_SU2);
            SpinVector M_SU2_global = Eigen::Map<Eigen::VectorXd>(M_SU2_global_arr, spin_dim_SU2);
            SpinVector M_SU3 = Eigen::Map<Eigen::VectorXd>(M_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            SpinVector M_SU3_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU3_antiferro_arr, spin_dim_SU3) / double(lattice_size_SU3);
            SpinVector M_SU3_global = Eigen::Map<Eigen::VectorXd>(M_SU3_global_arr, spin_dim_SU3);
            
            if (hdf5_writer) {
                hdf5_writer->write_flat_step(t, M_SU2_antiferro, M_SU2, M_SU2_global, 
                                            M_SU3_antiferro, M_SU3, M_SU3_global, state_vec.data());
                save_count++;
            }
            
            if (save_count % 10 == 0) {
                cout << "t=" << t << ", |M_SU2|=" << M_SU2.norm() << ", |M_SU3|=" << M_SU3.norm() << endl;
            }
        }
        
        if (hdf5_writer) {
            hdf5_writer->close();
            cout << "HDF5 trajectory saved with " << save_count << " snapshots" << endl;
        }
        
        cout << "GPU molecular dynamics complete!" << endl;
#endif
    }
    
    /**
     * GPU version of single_pulse_drive using opaque API
     */
    vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>>
    single_pulse_drive_gpu(const vector<SpinVector>& field_in_SU2,
                           const vector<SpinVector>& field_in_SU3,
                           double t_B,
                           double pulse_amp_SU2_in, double pulse_width_SU2_in, double pulse_freq_SU2_in,
                           double pulse_amp_SU3_in, double pulse_width_SU3_in, double pulse_freq_SU3_in,
                           double T_start, double T_end, double step_size,
                           const string& method = "dopri5") {
        
        // Set up pulses
        set_pulse_SU2(field_in_SU2, t_B, 
                     vector<SpinVector>(N_atoms_SU2, SpinVector::Zero(spin_dim_SU2)), 0.0,
                     pulse_amp_SU2_in, pulse_width_SU2_in, pulse_freq_SU2_in);
        set_pulse_SU3(field_in_SU3, t_B,
                     vector<SpinVector>(N_atoms_SU3, SpinVector::Zero(spin_dim_SU3)), 0.0,
                     pulse_amp_SU3_in, pulse_width_SU3_in, pulse_freq_SU3_in);
        
        ensure_gpu_mixed_data_initialized();
        update_gpu_pulse_SU2();
        update_gpu_pulse_SU3();
        
        // Transfer initial state
        ODEState h_state = spins_to_state();
        mixed_gpu::set_gpu_mixed_spins(gpu_mixed_handle_, h_state);
        
        // Integrate
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        mixed_gpu::integrate_mixed_gpu(gpu_mixed_handle_, T_start, T_end, step_size,
                                       1, raw_trajectory, method);
        
        // Convert to magnetization trajectory
        vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>> trajectory;
        size_t total_SU2 = lattice_size_SU2 * spin_dim_SU2;
        
        for (const auto& [t, state_vec] : raw_trajectory) {
            double M_local_SU2_arr[8] = {0}, M_antiferro_SU2_arr[8] = {0}, M_global_SU2_arr[8] = {0};
            double M_local_SU3_arr[8] = {0}, M_antiferro_SU3_arr[8] = {0}, M_global_SU3_arr[8] = {0};
            
            compute_sublattice_magnetizations_from_flat(state_vec.data(), 0, 
                lattice_size_SU2, spin_dim_SU2, M_local_SU2_arr, M_antiferro_SU2_arr);
            compute_sublattice_magnetizations_from_flat(state_vec.data(), total_SU2, 
                lattice_size_SU3, spin_dim_SU3, M_local_SU3_arr, M_antiferro_SU3_arr);
            
            // Global frame transformation (simplified)
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                size_t atom = i % N_atoms_SU2;
                for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
                    for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
                        M_global_SU2_arr[mu] += sublattice_frames_SU2[atom](nu, mu) * state_vec[i * spin_dim_SU2 + nu];
                    }
                }
            }
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                size_t atom = i % N_atoms_SU3;
                for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
                    for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
                        M_global_SU3_arr[mu] += sublattice_frames_SU3[atom](nu, mu) * state_vec[total_SU2 + i * spin_dim_SU3 + nu];
                    }
                }
            }
            
            SpinVector M_local_SU2 = Eigen::Map<Eigen::VectorXd>(M_local_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            SpinVector M_antiferro_SU2 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            SpinVector M_global_SU2 = Eigen::Map<Eigen::VectorXd>(M_global_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            SpinVector M_local_SU3 = Eigen::Map<Eigen::VectorXd>(M_local_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            SpinVector M_antiferro_SU3 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            SpinVector M_global_SU3 = Eigen::Map<Eigen::VectorXd>(M_global_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            
            trajectory.push_back({t, {{M_antiferro_SU2, M_local_SU2, M_global_SU2}, 
                                      {M_antiferro_SU3, M_local_SU3, M_global_SU3}}});
        }
        
        // Reset pulses
        field_drive_SU2[0] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_SU2[1] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_amp_SU2 = 0.0;
        field_drive_SU3[0] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_SU3[1] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_amp_SU3 = 0.0;
        
        return trajectory;
    }
    
    /**
     * GPU version of double_pulse_drive using opaque API
     */
    vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>>
    double_pulse_drive_gpu(const vector<SpinVector>& field_in_1_SU2,
                           const vector<SpinVector>& field_in_1_SU3,
                           double t_B_1,
                           const vector<SpinVector>& field_in_2_SU2,
                           const vector<SpinVector>& field_in_2_SU3,
                           double t_B_2,
                           double pulse_amp_SU2_in, double pulse_width_SU2_in, double pulse_freq_SU2_in,
                           double pulse_amp_SU3_in, double pulse_width_SU3_in, double pulse_freq_SU3_in,
                           double T_start, double T_end, double step_size,
                           const string& method = "dopri5") {
        
        // Set up two-pulse configuration
        set_pulse_SU2(field_in_1_SU2, t_B_1, field_in_2_SU2, t_B_2,
                     pulse_amp_SU2_in, pulse_width_SU2_in, pulse_freq_SU2_in);
        set_pulse_SU3(field_in_1_SU3, t_B_1, field_in_2_SU3, t_B_2,
                     pulse_amp_SU3_in, pulse_width_SU3_in, pulse_freq_SU3_in);
        
        ensure_gpu_mixed_data_initialized();
        update_gpu_pulse_SU2();
        update_gpu_pulse_SU3();
        
        // Transfer initial state
        ODEState h_state = spins_to_state();
        mixed_gpu::set_gpu_mixed_spins(gpu_mixed_handle_, h_state);
        
        // Integrate
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        mixed_gpu::integrate_mixed_gpu(gpu_mixed_handle_, T_start, T_end, step_size,
                                       1, raw_trajectory, method);
        
        // Convert to magnetization trajectory (same as single_pulse_drive_gpu)
        vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>> trajectory;
        size_t total_SU2 = lattice_size_SU2 * spin_dim_SU2;
        
        for (const auto& [t, state_vec] : raw_trajectory) {
            double M_local_SU2_arr[8] = {0}, M_antiferro_SU2_arr[8] = {0}, M_global_SU2_arr[8] = {0};
            double M_local_SU3_arr[8] = {0}, M_antiferro_SU3_arr[8] = {0}, M_global_SU3_arr[8] = {0};
            
            compute_sublattice_magnetizations_from_flat(state_vec.data(), 0, 
                lattice_size_SU2, spin_dim_SU2, M_local_SU2_arr, M_antiferro_SU2_arr);
            compute_sublattice_magnetizations_from_flat(state_vec.data(), total_SU2, 
                lattice_size_SU3, spin_dim_SU3, M_local_SU3_arr, M_antiferro_SU3_arr);
            
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                size_t atom = i % N_atoms_SU2;
                for (size_t mu = 0; mu < spin_dim_SU2; ++mu) {
                    for (size_t nu = 0; nu < spin_dim_SU2; ++nu) {
                        M_global_SU2_arr[mu] += sublattice_frames_SU2[atom](nu, mu) * state_vec[i * spin_dim_SU2 + nu];
                    }
                }
            }
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                size_t atom = i % N_atoms_SU3;
                for (size_t mu = 0; mu < spin_dim_SU3; ++mu) {
                    for (size_t nu = 0; nu < spin_dim_SU3; ++nu) {
                        M_global_SU3_arr[mu] += sublattice_frames_SU3[atom](nu, mu) * state_vec[total_SU2 + i * spin_dim_SU3 + nu];
                    }
                }
            }
            
            SpinVector M_local_SU2 = Eigen::Map<Eigen::VectorXd>(M_local_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            SpinVector M_antiferro_SU2 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            SpinVector M_global_SU2 = Eigen::Map<Eigen::VectorXd>(M_global_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
            SpinVector M_local_SU3 = Eigen::Map<Eigen::VectorXd>(M_local_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            SpinVector M_antiferro_SU3 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            SpinVector M_global_SU3 = Eigen::Map<Eigen::VectorXd>(M_global_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
            
            trajectory.push_back({t, {{M_antiferro_SU2, M_local_SU2, M_global_SU2}, 
                                      {M_antiferro_SU3, M_local_SU3, M_global_SU3}}});
        }
        
        // Reset pulses
        field_drive_SU2[0] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_SU2[1] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_amp_SU2 = 0.0;
        field_drive_SU3[0] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_SU3[1] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_amp_SU3 = 0.0;
        
        return trajectory;
    }
#endif // defined(CUDA_ENABLED) && !defined(__CUDACC__)

};

#endif // MIXED_LATTICE_REFACTORED_H
