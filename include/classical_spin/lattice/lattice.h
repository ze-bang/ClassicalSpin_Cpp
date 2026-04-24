#ifndef LATTICE_REFACTORED_H
#define LATTICE_REFACTORED_H

#include "unitcell.h"
#include "simple_linear_alg.h"
#include "hdf5_io.h"
#include "classical_spin/mc/mc_common.h"      // Common MC structs & templates
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
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <mpi.h>
#include <boost/numeric/odeint.hpp>

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
#include "lattice_gpu_api.h"
#endif

#if defined(CUDA_ENABLED) && defined(__CUDACC__)
#include "lattice_gpu.cuh"
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
using mc::AutocorrelationResult;
using mc::BinningResult;
using mc::Observable;
using mc::VectorObservable;
using mc::ThermodynamicObservables;
using mc::OptimizedTempGridResult;

// Simulated annealing parameters (Lattice-specific)
struct SAParams {
    double T_start = 1.0;
    double T_end = 1e-3;
    double cooling_rate = 0.9;
    size_t sweeps_per_temp = 100;
    vector<double> probe_T;
    vector<double> probe_acc;
    vector<double> probe_tau;
};

/**
 * Real-space correlation accumulator for efficient finite-T structure factor calculation
 * 
 * Stores spin-spin and dimer-dimer correlations binned by displacement, enabling:
 * - Post-hoc Fourier transform to any q-point: S(q) = Σ_Δr C(Δr) exp(-i q·Δr)
 * - Fixed storage regardless of number of MC samples
 * - Online error estimation via binning analysis
 * 
 * Storage: O(N_cells × N_sub² × 9) for spin correlations
 *        + O(N_cells × N_bond × N_bond²) for dimer correlations
 * vs O(N_sites × N_samples) for full snapshots
 */
struct RealSpaceCorrelationAccumulator {
    // ========== LATTICE GEOMETRY ==========
    size_t dim1, dim2, dim3;           // Lattice dimensions
    size_t n_sites;                     // Total sites
    size_t n_sublattices;               // Sites per unit cell (N_atoms)
    size_t spin_dim;                    // Dimension of spin vectors
    
    // ========== INDEXING WITH SYMMETRY ==========
    // Cell displacements: (Δn1, Δn2, Δn3) → linear index
    // For PBC: Δn1 ∈ [0, dim1), Δn2 ∈ [0, dim2), Δn3 ∈ [0, dim3)
    size_t n_cell_displacements;        // dim1 * dim2 * dim3
    
    // Sublattice pairs with symmetry: (sub_i, sub_j) with sub_i ≤ sub_j
    // For n_sub=4: (0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
    size_t n_sublattice_pairs;          // n_sublattices * (n_sublattices + 1) / 2
    
    // Spin components with symmetry: (α, β) with α ≤ β
    // (x,x), (x,y), (x,z), (y,y), (y,z), (z,z) = 6 components
    static constexpr size_t n_spin_components = 6;
    
    // ========== SPIN-SPIN CORRELATIONS ==========
    // C^{αβ}_{sub_pair}(Δcell) = <S_i^α S_j^β + S_i^β S_j^α> / 2  for α≠β
    //                          = <S_i^α S_j^α>                     for α=β
    // Shape: [n_cell_displacements][n_sublattice_pairs][n_spin_components]
    vector<double> spin_corr_sum;       // Σ samples
    vector<double> spin_corr_sq_sum;    // Σ samples² (for error)
    
    // Sublattice-resolved single-site averages (for connected correlator)
    vector<Eigen::Vector3d> spin_mean_sum;      // [n_sublattices] Σ <S_α>
    vector<Eigen::Vector3d> spin_mean_sq_sum;   // For error estimation
    
    // ========== DIMER-DIMER CORRELATIONS ==========
    // Bond type = sublattice pair (i,j) with i ≤ j that the bond connects
    // For pyrochlore (n_sub=4): 10 bond types
    size_t n_bond_types;                // = n_sublattice_pairs
    
    // Dimer components: D^α = S_i^α S_j^α for α ∈ {x, y, z}
    static constexpr size_t n_dimer_components = 3;  // x, y, z
    
    // Dimer correlation: <D^α_μ(0) D^α_ν(ΔR)> for same spin component α
    // Shape: [n_cell_displacements][n_bond_types][n_bond_types][n_dimer_components]
    // Flattened: [cell_disp * n_bond_types * n_bond_types * 3]
    vector<double> dimer_corr_sum;      // Σ samples
    vector<double> dimer_corr_sq_sum;   // For error
    
    // Bond-type and component resolved means: <D^α_μ>
    // Shape: [n_bond_types][n_dimer_components]
    vector<double> dimer_mean_sum;      // <D^α_μ> for each bond type and component
    vector<double> dimer_mean_sq_sum;   // For error
    
    // ========== DISPLACEMENT GEOMETRY ==========
    vector<Eigen::Vector3d> cell_displacement_vectors;  // Real-space ΔR for each cell offset
    vector<array<int, 3>> cell_displacement_indices;    // (Δn1, Δn2, Δn3)
    vector<Eigen::Vector3d> sublattice_positions_;      // Sublattice positions within unit cell

    // ========== CACHED LOOKUP TABLES (filled once in initialize()) ==========
    //
    // Pre-computed displaced_cell_idx[disp_idx * n_cells + cell_i] = cell_j,
    // where cell_j is the cell index obtained by translating cell_i by the
    // displacement (dn1, dn2, dn3) corresponding to disp_idx, modulo PBC.
    //
    // This lookup is purely geometric (depends only on dim1*dim2*dim3) and was
    // previously rebuilt on **every** call to
    // accumulate_spin_correlations / accumulate_dimer_correlations — a 24 MB
    // allocation per measurement on a 12³ unit-cell grid. Cached here so the
    // hot accumulation loop becomes a flat array lookup with no allocation.
    vector<size_t> displaced_cell_idx_cache;            // size n_cells * n_cells

    // Reusable scratch buffer for the per-call sublattice-resolved spin
    // layout used by accumulate_spin_correlations. Allocated once on first
    // use, sized to n_sublattices * n_cells. This replaces a fresh
    // vector<vector<Eigen::Vector3d>> on every call.
    mutable vector<Eigen::Vector3d> spins_by_sub_buf;   // size n_sublattices * n_cells
    
    // ========== BOOKKEEPING ==========
    size_t n_samples;                   // Number of accumulated samples
    bool initialized;
    
    // ========== CONSTRUCTORS ==========
    RealSpaceCorrelationAccumulator() : n_samples(0), initialized(false) {}
    
    /**
     * Initialize for given lattice geometry
     * 
     * @param d1, d2, d3    Lattice dimensions
     * @param n_sub         Number of sublattices (atoms per unit cell)
     * @param n_bonds       Number of distinct bond types (ignored, will be set to n_sublattice_pairs)
     * @param sdim          Spin dimension (typically 3)
     * @param lattice_vectors  The 3 lattice vectors (a1, a2, a3)
     * @param sublattice_positions  Positions within unit cell for each sublattice
     */
    void initialize(size_t d1, size_t d2, size_t d3, 
                   size_t n_sub, size_t /* n_bonds */, size_t sdim,
                   const array<Eigen::Vector3d, 3>& lattice_vectors,
                   const vector<Eigen::Vector3d>& sublattice_positions);
    
    /**
     * Get cell displacement index (no sublattice info)
     */
    size_t cell_displacement_index(size_t dn1, size_t dn2, size_t dn3) const {
        return (dn1 * dim2 + dn2) * dim3 + dn3;
    }
    
    /**
     * Get sublattice pair index with symmetry: (i,j) → index, requires i ≤ j
     * For n_sub=4: (0,0)→0, (0,1)→1, (0,2)→2, (0,3)→3, (1,1)→4, (1,2)→5, ...
     * This is also the bond type index for bonds connecting sublattices i and j.
     */
    size_t sublattice_pair_index(size_t sub_i, size_t sub_j) const {
        size_t s_min = std::min(sub_i, sub_j);
        size_t s_max = std::max(sub_i, sub_j);
        return s_min * (2 * n_sublattices - s_min - 1) / 2 + s_max;
    }
    
    /**
     * Inverse of sublattice_pair_index: index → (sub_i, sub_j) with i ≤ j
     * Also gives the sublattices that a bond type connects.
     */
    pair<size_t, size_t> bond_type_to_sublattices(size_t bond_type) const {
        // Triangular number inversion
        size_t sub_i = 0;
        size_t cumsum = n_sublattices;
        while (bond_type >= cumsum) {
            sub_i++;
            cumsum += (n_sublattices - sub_i);
        }
        size_t sub_j = bond_type - (sub_i == 0 ? 0 : sub_i * n_sublattices - sub_i * (sub_i + 1) / 2);
        return {sub_i, sub_j};
    }
    
    /**
     * Get bond center position for a given bond type (within unit cell)
     * Bond center = (r_sub_i + r_sub_j) / 2
     */
    Eigen::Vector3d bond_center(size_t bond_type) const;
    
    /**
     * Get spin component index with symmetry: (α,β) → index, requires α ≤ β
     * (0,0)→0, (0,1)→1, (0,2)→2, (1,1)→3, (1,2)→4, (2,2)→5
     */
    static size_t spin_component_index(size_t alpha, size_t beta) {
        size_t a_min = std::min(alpha, beta);
        size_t a_max = std::max(alpha, beta);
        return a_min * (2 * 3 - a_min - 1) / 2 + a_max;
    }
    
    /**
     * Get full spin correlation index
     * @return Index into spin_corr_sum array
     */
    size_t spin_corr_index(size_t cell_disp_idx, size_t sub_pair_idx, size_t spin_comp_idx) const {
        return (cell_disp_idx * n_sublattice_pairs + sub_pair_idx) * n_spin_components + spin_comp_idx;
    }
    
    /**
     * Get dimer correlation index
     * Shape: [n_cell_displacements][n_bond_types][n_bond_types][n_dimer_components]
     * @param cell_disp_idx  Cell displacement index
     * @param type_mu        Bond type at origin
     * @param type_nu        Bond type at displaced cell
     * @param comp           Dimer component (0=x, 1=y, 2=z)
     */
    size_t dimer_corr_index(size_t cell_disp_idx, size_t type_mu, size_t type_nu, size_t comp) const {
        return ((cell_disp_idx * n_bond_types + type_mu) * n_bond_types + type_nu) * n_dimer_components + comp;
    }
    
    /**
     * Get dimer mean index
     * Shape: [n_bond_types][n_dimer_components]
     */
    size_t dimer_mean_index(size_t bond_type, size_t comp) const {
        return bond_type * n_dimer_components + comp;
    }
    
    /**
     * Accumulate one sample of spin-spin correlations
     * Call this every probe_rate MC sweeps
     * 
     * Uses symmetry: C^{αβ}_{ij} = C^{βα}_{ji}, so we store symmetrized form
     * 
     * OPTIMIZED: Pre-compute spin arrays organized by sublattice for cache efficiency,
     * then use direct indexing instead of per-cell modular arithmetic.
     * 
     * @param spins         Current spin configuration [n_sites]
     * @param site_to_sub   Mapping from site index to sublattice index (unused, kept for API)
     * @param site_to_cell  Mapping from site index to (n1, n2, n3) cell indices (unused, kept for API)
     */
    void accumulate_spin_correlations(
        const vector<Eigen::VectorXd>& spins,
        const function<size_t(size_t)>& /* site_to_sublattice */,
        const function<array<size_t, 3>(size_t)>& /* site_to_cell */) 
;
    
    /**
     * Accumulate dimer-dimer correlations
     * 
     * Dimer operator: D^α_b = S_i^α S_j^α for α ∈ {x, y, z}
     * Correlator: <D^α_μ(0) D^α_ν(ΔR)> for each component α
     * 
     * OPTIMIZED: Pre-compute cell displacement lookup, avoid repeated modular arithmetic.
     * 
     * @param spins         Current spin configuration
     * @param bonds         List of (site_i, site_j) pairs defining bonds
     * @param bond_types    Bond type index for each bond (= sublattice pair index)
     * @param bond_cells    Unit cell indices (n1, n2, n3) for each bond's "home" cell
     */
    void accumulate_dimer_correlations(
        const vector<Eigen::VectorXd>& spins,
        const vector<array<size_t, 2>>& bonds,
        const vector<size_t>& bond_types,
        const vector<array<size_t, 3>>& bond_cells)
;
    
    /**
     * Compute spin structure factor S^{αβ}(q) at arbitrary q-point
     * Sublattice-resolved: S(q) = Σ_{ΔR,s,s'} C_{ss'}(ΔR) exp(-i q·(ΔR + r_s' - r_s))
     * 
     * @param q           Wavevector in Cartesian coordinates
     * @param connected   If true, subtract <S_i><S_j> (use for susceptibility)
     * @return 3x3 matrix S^{αβ}(q) (symmetric form)
     */
    Eigen::Matrix3d compute_Sq(const Eigen::Vector3d& q, bool connected = true) const;
    
    /**
     * Compute dimer structure factor at arbitrary q
     * S_D^{αμν}(q) = Σ_ΔR χ^α_D(ΔR,μ,ν) exp(-i q·Δr_bond)
     * 
     * where Δr_bond = ΔR + center(ν) - center(μ) is the displacement between bond centers
     * and center(μ) = (r_{sub_i} + r_{sub_j})/2 for a bond connecting sublattices i,j
     * 
     * Returns array of 3 matrices [n_bond_types × n_bond_types], one per component (x,y,z)
     * 
     * @param q           Wavevector in Cartesian coordinates  
     * @param connected   If true, subtract <D^α_μ><D^α_ν>
     * @return array of 3 matrices for x, y, z dimer components
     */
    array<Eigen::MatrixXd, 3> compute_Sq_dimer(const Eigen::Vector3d& q, bool connected = true) const;
    
    /**
     * Compute total dimer structure factor (sum over components)
     * S_D^{μν}(q) = Σ_α S_D^{αμν}(q) = Σ_α Σ_ΔR <D^α_μ(0) D^α_ν(ΔR)> exp(-i q·ΔR)
     */
    Eigen::MatrixXd compute_Sq_dimer_total(const Eigen::Vector3d& q, bool connected = true) const;
    
    /**
     * Compute S(q) with error estimate using variance
     * Returns (mean, error) pair for each matrix element
     */
    pair<Eigen::Matrix3d, Eigen::Matrix3d> compute_Sq_with_error(
        const Eigen::Vector3d& q, bool connected = true) const 
    {
        if (n_samples < 2) {
            return {compute_Sq(q, connected), Eigen::Matrix3d::Zero()};
        }
        
        // For proper error estimation, return zero error for now
        // (would need jackknife/bootstrap for correlated samples)
        return {compute_Sq(q, connected), Eigen::Matrix3d::Zero()};
    }
    
    /**
     * Reset accumulator (keep geometry, clear statistics)
     */
    void reset() {
        std::fill(spin_corr_sum.begin(), spin_corr_sum.end(), 0.0);
        std::fill(spin_corr_sq_sum.begin(), spin_corr_sq_sum.end(), 0.0);
        for (auto& m : spin_mean_sum) m.setZero();
        for (auto& m : spin_mean_sq_sum) m.setZero();
        std::fill(dimer_corr_sum.begin(), dimer_corr_sum.end(), 0.0);
        std::fill(dimer_corr_sq_sum.begin(), dimer_corr_sq_sum.end(), 0.0);
        std::fill(dimer_mean_sum.begin(), dimer_mean_sum.end(), 0.0);
        std::fill(dimer_mean_sq_sum.begin(), dimer_mean_sq_sum.end(), 0.0);
        n_samples = 0;
    }
    
    /**
     * Merge another accumulator into this one (for MPI reduction)
     */
    void merge(const RealSpaceCorrelationAccumulator& other);
    
    /**
     * Get storage size in bytes
     */
    size_t storage_bytes() const {
        size_t spin_storage = spin_corr_sum.size() * 2 * sizeof(double);
        spin_storage += n_sublattices * 2 * sizeof(Eigen::Vector3d);
        size_t dimer_storage = dimer_corr_sum.size() * 2 * sizeof(double);
        dimer_storage += n_bond_types * 2 * sizeof(double);
        size_t geometry_storage = cell_displacement_vectors.size() * sizeof(Eigen::Vector3d);
        geometry_storage += cell_displacement_indices.size() * sizeof(array<int, 3>);
        geometry_storage += sublattice_positions_.size() * sizeof(Eigen::Vector3d);
        return spin_storage + dimer_storage + geometry_storage;
    }
    
#ifdef HDF5_ENABLED
    /**
     * Save correlation data to HDF5 file
     * 
     * Data layout:
     * - spin_corr_sum: [n_cell_displacements × n_sublattice_pairs × 6] flattened
     * - dimer_corr_sum: [n_cell_displacements × n_bond_type_pairs] flattened
     * 
     * Sublattice pair (i,j) with i≤j maps to index: i*(2*n_sub - i - 1)/2 + j
     * Spin component (α,β) with α≤β: xx=0, xy=1, xz=2, yy=3, yz=4, zz=5
     */
    void save_hdf5(const string& filename, const string& group_name = "/correlations") const;
#endif
    
    /**
     * Save spin structure factor to text file for a grid of q-points
     * 
     * @param filename      Output file path
     * @param q1_range      Range in reciprocal lattice units for q1 (min, max)
     * @param q2_range      Range in reciprocal lattice units for q2 (min, max)
     * @param q3_range      Range in reciprocal lattice units for q3 (min, max)
     * @param n_q           Number of q-points per dimension
     * @param b1, b2, b3    Reciprocal lattice vectors
     * @param connected     Whether to compute connected correlator
     */
    void save_structure_factor_grid(
        const string& filename,
        pair<double, double> q1_range,
        pair<double, double> q2_range,
        pair<double, double> q3_range,
        size_t n_q1, size_t n_q2, size_t n_q3,
        const Eigen::Vector3d& b1,
        const Eigen::Vector3d& b2,
        const Eigen::Vector3d& b3,
        bool connected = true) const 
;
    
    /**
     * MPI reduce: gather accumulators from all ranks and merge
     * After this call, rank 0 has the combined accumulator
     * 
     * @param comm  MPI communicator
     * @return Combined accumulator (valid on rank 0 only)
     */
    void mpi_reduce(MPI_Comm comm = MPI_COMM_WORLD);
};

/**
 * Lattice class: Template-free implementation using Eigen3 and std::vector
 * 
 * This class manages a periodic lattice of classical spins with:
 * - Runtime-configurable dimensions (spin_dim, N_atoms, dim1, dim2, dim3)
 * - Bilinear and trilinear interactions
 * - Twisted boundary conditions
 * - Monte Carlo sampling (Metropolis, Wolff, Swendsen-Wang)
 * - Simulated annealing with auto-tuning
 * - Parallel tempering with twist boundary conditions
 * - Molecular dynamics (Landau-Lifshitz equations)
 */
class Lattice {
public:
    // Type aliases for clarity
    using SpinConfig = vector<SpinVector>;
    using CrossProductMethod = function<SpinVector(const SpinVector&, const SpinVector&)>;
    using ODEState = vector<double>;  // Flat state vector for Boost.Odeint

    // Core lattice properties
    UnitCell unit_cell;
    std::string lattice_type;  // Lattice type identifier (e.g., "pyrochlore", "pyrochlore_non_kramer")
    size_t spin_dim;         // Dimension of spin vectors (e.g., 3 for SU(2), 8 for SU(3))
    size_t N_atoms;          // Number of atoms per unit cell
    size_t dim1, dim2, dim3; // Lattice dimensions
    size_t lattice_size;     // Total number of sites = N_atoms * dim1 * dim2 * dim3
    float spin_length;       // Magnitude of spin vectors

    // Spin configuration and positions
    SpinConfig spins;                // Current spin configuration
    vector<Eigen::Vector3d> site_positions; // Real-space positions

    // Interaction lookup tables
    SpinConfig field;                                    // External field at each site
    vector<SpinMatrix> onsite_interaction;               // On-site anisotropy
    vector<vector<SpinMatrix>> bilinear_interaction;     // Bilinear coupling
    vector<vector<SpinTensor3>> trilinear_interaction;   // Trilinear coupling
    vector<vector<size_t>> bilinear_partners;            // Partner site indices
    vector<vector<array<size_t, 2>>> trilinear_partners; // Trilinear partner pairs
    vector<SpinMatrix> sublattice_frames;                // Sublattice frame transformations
    vector<double> afm_sublattice_signs;                  // Signs for staggered magnetization per sublattice

    size_t num_bi;  // Number of bilinear neighbors per site
    size_t num_tri; // Number of trilinear interactions per site

    // ------------------------------------------------------------------
    // Flat / SoA mirror of the bilinear interaction table.
    //
    // Built once at the end of initialize() (and rebuilt by the copy
    // constructor and assignment operator). Used by the hot-loop kernels
    // -- site_energy_diff_flat, overrelaxation, get_local_field_flat --
    // to eliminate the two pointer indirections of
    // vector<vector<Eigen::Matrix3d>> and to put each site's J matrices
    // on consecutive cache lines.
    //
    // Layout (CSR-style): for site i, neighbour n in
    //   [bi_flat_offset[i], bi_flat_offset[i+1]):
    //     partner index = bi_flat_partner[n]
    //     J matrix      = &bi_flat_J[n * spin_dim * spin_dim]    (row-major)
    //     wrap          = bi_flat_wrap[n]
    //     needs_twist   = bi_flat_needs_twist[n]   (precomputed wrap != 0)
    //
    // The nested vector<vector<>> tables (bilinear_interaction,
    // bilinear_partners, bilinear_wrap_dir) are still maintained as the
    // source of truth; the SoA tables are a derived cache used only by
    // the per-site hot kernels.
    // ------------------------------------------------------------------
    vector<size_t>             bi_flat_offset;
    vector<size_t>             bi_flat_partner;
    vector<double>             bi_flat_J;
    vector<uint8_t>            bi_flat_needs_twist;
    vector<array<int8_t, 3>>   bi_flat_wrap;
    size_t                     bi_flat_D2 = 0;   // = spin_dim * spin_dim, cached

    // Twist boundary conditions
    array<SpinMatrix, 3> twist_matrices;                 // Rotation matrices per dimension
    array<SpinVector, 3> rotation_axis;                  // Rotation axes
    array<double, 3> twist_angles;                       // Current twist angles (radians)
    vector<vector<array<int8_t, 3>>> bilinear_wrap_dir;  // Wrap direction per neighbor
    array<vector<size_t>, 3> boundary_sites_per_dim;     // Sites near boundaries
    array<size_t, 3> boundary_thickness;                 // Layers affected by twist

    // Time-dependent field for molecular dynamics
    array<SpinVector, 2> field_drive; // Two pulse components
    array<double, 2> t_pulse;         // Pulse center times
    double field_drive_amp;           // Pulse amplitude
    double field_drive_freq;          // Pulse frequency
    double field_drive_width;         // Pulse width (Gaussian)

    // Gilbert damping parameter for LLG dynamics
    double alpha_gilbert = 0.0;       // 0 = undamped (pure LL)

    // ------------------------------------------------------------------
    // Persistent scratch buffers for cluster-MC sweeps.
    //
    // wolff_update / swendsen_wang_sweep used to allocate fresh
    // vector<double>(N) / vector<int>(N) on every call; on a 4×L³ pyrochlore
    // with L=12 (27 648 sites) this is ~hundreds of kB of malloc/free per
    // sweep. Keeping the buffers as members reuses the same allocation
    // across sweeps (and across replicas in PT, since PT swaps `spins`
    // pointers, not Lattice objects). They are `mutable` so const-qualified
    // helpers can use them when needed.
    // ------------------------------------------------------------------
    mutable vector<double>  cluster_proj_buf;     // size lattice_size
    mutable vector<uint8_t> cluster_in_cluster;   // size lattice_size
    mutable vector<size_t>  cluster_stack_buf;    // BFS stack (Wolff)
    mutable vector<int>     uf_parent;            // Union-Find parent
    mutable vector<int>     uf_size;              // Union-Find subtree size
    mutable vector<uint8_t> uf_forbid_flip;       // SW: ghost-bonded clusters
    mutable vector<uint8_t> uf_flip_root;         // SW: per-root flip flag

    /**
     * Check if lattice is a pyrochlore type (pyrochlore or pyrochlore_non_kramer)
     * Used to validate pyrochlore-specific order parameters.
     */
    bool is_pyrochlore() const {
        return lattice_type == "pyrochlore" || lattice_type == "pyrochlore_non_kramer";
    }

    /**
     * Constructor: Build a lattice from a unit cell
     * 
     * @param uc        Unit cell defining the lattice structure
     * @param dim1      Lattice size in first dimension
     * @param dim2      Lattice size in second dimension
     * @param dim3      Lattice size in third dimension
     * @param spin_l    Magnitude of spin vectors
     */
    Lattice(const UnitCell& uc, size_t dim1, size_t dim2, size_t dim3, float spin_l = 1.0)
        : unit_cell(uc), 
          spin_dim(uc.N), 
          N_atoms(uc.N_atoms),
          dim1(dim1), 
          dim2(dim2), 
          dim3(dim3),
          spin_length(spin_l)
    {
        lattice_size = N_atoms * dim1 * dim2 * dim3;
        
        // Initialize arrays
        spins.resize(lattice_size);
        site_positions.resize(lattice_size);
        field.resize(lattice_size);
        onsite_interaction.resize(lattice_size);
        bilinear_interaction.resize(lattice_size);
        trilinear_interaction.resize(lattice_size);
        bilinear_partners.resize(lattice_size);
        trilinear_partners.resize(lattice_size);
        bilinear_wrap_dir.resize(lattice_size);
        sublattice_frames.resize(N_atoms);
        
        // Copy sublattice frames from unit cell
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            sublattice_frames[atom] = uc.sublattice_frames[atom];
        }
        
        // Copy AFM sublattice signs from unit cell
        afm_sublattice_signs = uc.afm_sublattice_signs;

        // Initialize twist matrices to identity
        for (size_t d = 0; d < 3; ++d) {
            twist_matrices[d] = SpinMatrix::Identity(spin_dim, spin_dim);
            rotation_axis[d] = SpinVector::Zero(spin_dim);
            if (spin_dim >= 3) rotation_axis[d](2) = 1.0; // Default: z-axis
        }

        // Initialize time-dependent field
        field_drive[0] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive[1] = SpinVector::Zero(N_atoms * spin_dim);
        t_pulse[0] = 0.0;
        t_pulse[1] = 0.0;
        field_drive_amp = 0.0;
        field_drive_freq = 0.0;
        field_drive_width = 1.0;

        // Initialize random seed
        seed_lehman(chrono::system_clock::now().time_since_epoch().count() * 2 + 1);

        // Compute boundary thickness (max neighbor offset in each dimension)
        boundary_thickness = {0, 0, 0};
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            auto range = unit_cell.bilinear_interaction.equal_range(atom);
            for (auto it = range.first; it != range.second; ++it) {
                const auto& bi = it->second;
                boundary_thickness[0] = std::max(boundary_thickness[0], size_t(std::abs(bi.offset[0])));
                boundary_thickness[1] = std::max(boundary_thickness[1], size_t(std::abs(bi.offset[1])));
                boundary_thickness[2] = std::max(boundary_thickness[2], size_t(std::abs(bi.offset[2])));
            }
        }

        // Build the lattice
        cout << "Initializing lattice with dimensions: " << dim1 << " x " << dim2 << " x " << dim3 << endl;
        cout << "Total sites: " << lattice_size << endl;
        cout << "Spin dimension: " << spin_dim << ", Atoms per cell: " << N_atoms << endl;

        // First pass: count interactions per site for proper sizing
        vector<size_t> bi_count(lattice_size, 0);
        vector<size_t> tri_count(lattice_size, 0);
        
        size_t site_idx = 0;
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms; ++atom) {
                        // Count forward bilinear interactions
                        auto bi_range = unit_cell.bilinear_interaction.equal_range(atom);
                        bi_count[site_idx] = std::distance(bi_range.first, bi_range.second);
                        
                        // Count forward trilinear interactions
                        auto tri_range = unit_cell.trilinear_interaction.equal_range(atom);
                        tri_count[site_idx] = std::distance(tri_range.first, tri_range.second);
                        
                        ++site_idx;
                    }
                }
            }
        }

        // Second pass: add reverse interaction counts
        site_idx = 0;
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms; ++atom) {
                        // Add reverse bilinear counts
                        auto bi_range = unit_cell.bilinear_interaction.equal_range(atom);
                        for (auto it = bi_range.first; it != bi_range.second; ++it) {
                            const auto& bi = it->second;
                            int pi = i + bi.offset[0];
                            int pj = j + bi.offset[1];
                            int pk = k + bi.offset[2];
                            
                            if (pi < 0) pi += dim1;
                            else if (pi >= (int)dim1) pi -= dim1;
                            if (pj < 0) pj += dim2;
                            else if (pj >= (int)dim2) pj -= dim2;
                            if (pk < 0) pk += dim3;
                            else if (pk >= (int)dim3) pk -= dim3;
                            
                            size_t partner_idx = flatten_index(pi, pj, pk, bi.partner);
                            bi_count[partner_idx]++;
                        }
                        
                        // Add reverse trilinear counts (2 permutations per interaction)
                        auto tri_range = unit_cell.trilinear_interaction.equal_range(atom);
                        for (auto it = tri_range.first; it != tri_range.second; ++it) {
                            const auto& tri = it->second;
                            size_t p1 = flatten_index_periodic(i + tri.offset1[0], 
                                                               j + tri.offset1[1], 
                                                               k + tri.offset1[2], 
                                                               tri.partner1);
                            size_t p2 = flatten_index_periodic(i + tri.offset2[0], 
                                                               j + tri.offset2[1], 
                                                               k + tri.offset2[2], 
                                                               tri.partner2);
                            tri_count[p1]++;
                            tri_count[p2]++;
                        }
                        
                        ++site_idx;
                    }
                }
            }
        }

        // Allocate storage with correct sizes
        for (size_t idx = 0; idx < lattice_size; ++idx) {
            bilinear_interaction[idx].reserve(bi_count[idx]);
            bilinear_partners[idx].reserve(bi_count[idx]);
            bilinear_wrap_dir[idx].reserve(bi_count[idx]);
            trilinear_interaction[idx].reserve(tri_count[idx]);
            trilinear_partners[idx].reserve(tri_count[idx]);
        }

        // Third pass: build interactions
        site_idx = 0;
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms; ++atom) {
                        // Compute real-space position
                        Eigen::Vector3d pos = unit_cell.lattice_pos[atom];
                        pos += i * unit_cell.lattice_vectors[0];
                        pos += j * unit_cell.lattice_vectors[1];
                        pos += k * unit_cell.lattice_vectors[2];
                        site_positions[site_idx] = pos;

                        // Initialize spin randomly on sphere
                        spins[site_idx] = gen_random_spin(spin_length);

                        // Copy field from unit cell
                        field[site_idx] = unit_cell.field[atom];

                        // Copy on-site interaction
                        onsite_interaction[site_idx] = unit_cell.onsite_interaction[atom];

                        // Build bilinear interactions (forward and reverse)
                        auto bi_range = unit_cell.bilinear_interaction.equal_range(atom);
                        for (auto it = bi_range.first; it != bi_range.second; ++it) {
                            const auto& bi = it->second;
                            
                            // Compute partner site with periodic boundaries
                            int pi = i + bi.offset[0];
                            int pj = j + bi.offset[1];
                            int pk = k + bi.offset[2];
                            
                            // Track wrapping for twist boundaries
                            array<int8_t, 3> wrap = {0, 0, 0};
                            if (pi < 0) { pi += dim1; wrap[0] = -1; }
                            else if (pi >= (int)dim1) { pi -= dim1; wrap[0] = +1; }
                            if (pj < 0) { pj += dim2; wrap[1] = -1; }
                            else if (pj >= (int)dim2) { pj -= dim2; wrap[1] = +1; }
                            if (pk < 0) { pk += dim3; wrap[2] = -1; }
                            else if (pk >= (int)dim3) { pk -= dim3; wrap[2] = +1; }
                            
                            size_t partner_idx = flatten_index(pi, pj, pk, bi.partner);
                            
                            // Add forward interaction: site_idx -> partner_idx with J
                            bilinear_interaction[site_idx].push_back(bi.interaction);
                            bilinear_partners[site_idx].push_back(partner_idx);
                            bilinear_wrap_dir[site_idx].push_back(wrap);
                            
                            // Add reverse interaction: partner_idx -> site_idx with J^T
                            array<int8_t, 3> wrap_reverse = {
                                static_cast<int8_t>(-wrap[0]),
                                static_cast<int8_t>(-wrap[1]),
                                static_cast<int8_t>(-wrap[2])
                            };
                            bilinear_interaction[partner_idx].push_back(bi.interaction.transpose());
                            bilinear_partners[partner_idx].push_back(site_idx);
                            bilinear_wrap_dir[partner_idx].push_back(wrap_reverse);
                        }

                        // Build trilinear interactions (forward and two permutations)
                        auto tri_range = unit_cell.trilinear_interaction.equal_range(atom);
                        for (auto it = tri_range.first; it != tri_range.second; ++it) {
                            const auto& tri = it->second;
                            
                            size_t p1 = flatten_index_periodic(i + tri.offset1[0], 
                                                               j + tri.offset1[1], 
                                                               k + tri.offset1[2], 
                                                               tri.partner1);
                            size_t p2 = flatten_index_periodic(i + tri.offset2[0], 
                                                               j + tri.offset2[1], 
                                                               k + tri.offset2[2], 
                                                               tri.partner2);
                            
                            // Add forward interaction: T[i,j,k] with (site_idx, p1, p2)
                            trilinear_interaction[site_idx].push_back(tri.interaction);
                            trilinear_partners[site_idx].push_back({p1, p2});
                            
                            // Add permutation 1: T[j,k,i] with (p1, p2, site_idx)
                            // transpose3D(T, N, N, N) performs cyclic permutation: T[i](j,k) -> T_new[j](k,i)
                            SpinTensor3 perm1 = transpose3D(tri.interaction, spin_dim, spin_dim, spin_dim);
                            trilinear_interaction[p1].push_back(perm1);
                            trilinear_partners[p1].push_back({p2, site_idx});
                            
                            // Add permutation 2: T[k,i,j] with (p2, site_idx, p1)
                            // Apply transpose3D twice for double cyclic permutation
                            SpinTensor3 perm2 = transpose3D(perm1, spin_dim, spin_dim, spin_dim);
                            trilinear_interaction[p2].push_back(perm2);
                            trilinear_partners[p2].push_back({site_idx, p1});
                        }

                        ++site_idx;
                    }
                }
            }
        }

        // Set num_bi and num_tri to maximum (for compatibility)
        num_bi = *std::max_element(bi_count.begin(), bi_count.end());
        num_tri = *std::max_element(tri_count.begin(), tri_count.end());

        build_boundary_sites();
        build_flat_bilinear_tables();
        cout << "Lattice initialization complete!" << endl;
        cout << "Max bilinear interactions per site: " << num_bi << endl;
        cout << "Max trilinear interactions per site: " << num_tri << endl;
    }

    /**
     * Build the SoA / flat mirror of the bilinear interaction table.
     *
     * Idempotent. Call after the nested `bilinear_interaction`,
     * `bilinear_partners`, and `bilinear_wrap_dir` tables are populated
     * (i.e. at the end of initialize() and after the copy constructor /
     * assignment op runs).
     *
     * Costs O(total_bonds * spin_dim^2) memory and time; ~480 kB for
     * a 32x32x1 honeycomb with spin_dim=3 (negligible).
     */
    void build_flat_bilinear_tables() {
        bi_flat_D2 = spin_dim * spin_dim;

        bi_flat_offset.assign(lattice_size + 1, 0);
        for (size_t i = 0; i < lattice_size; ++i) {
            bi_flat_offset[i + 1] = bi_flat_offset[i] + bilinear_partners[i].size();
        }
        const size_t total_bonds = bi_flat_offset[lattice_size];

        bi_flat_partner.assign(total_bonds, 0);
        bi_flat_J.assign(total_bonds * bi_flat_D2, 0.0);
        bi_flat_needs_twist.assign(total_bonds, 0);
        bi_flat_wrap.assign(total_bonds, std::array<int8_t, 3>{0, 0, 0});

        for (size_t i = 0; i < lattice_size; ++i) {
            const size_t base = bi_flat_offset[i];
            const size_t n_bi = bilinear_partners[i].size();
            for (size_t n = 0; n < n_bi; ++n) {
                const size_t k = base + n;
                bi_flat_partner[k] = bilinear_partners[i][n];

                const auto& J = bilinear_interaction[i][n];
                double* dst = &bi_flat_J[k * bi_flat_D2];
                for (size_t a = 0; a < spin_dim; ++a) {
                    for (size_t b = 0; b < spin_dim; ++b) {
                        dst[a * spin_dim + b] = J(a, b);
                    }
                }

                const auto& wrap = bilinear_wrap_dir[i][n];
                bi_flat_wrap[k] = wrap;
                bi_flat_needs_twist[k] =
                    (wrap[0] != 0 || wrap[1] != 0 || wrap[2] != 0) ? uint8_t(1) : uint8_t(0);
            }
        }
    }

    /**
     * Copy constructor
     */
    Lattice(const Lattice& other)
        : unit_cell(other.unit_cell),
          spin_dim(other.spin_dim),
          N_atoms(other.N_atoms),
          dim1(other.dim1),
          dim2(other.dim2),
          dim3(other.dim3),
          lattice_size(other.lattice_size),
          spin_length(other.spin_length),
          spins(other.spins),
          site_positions(other.site_positions),
          field(other.field),
          onsite_interaction(other.onsite_interaction),
          bilinear_interaction(other.bilinear_interaction),
          trilinear_interaction(other.trilinear_interaction),
          bilinear_partners(other.bilinear_partners),
          trilinear_partners(other.trilinear_partners),
          num_bi(other.num_bi),
          num_tri(other.num_tri),
          bi_flat_offset(other.bi_flat_offset),
          bi_flat_partner(other.bi_flat_partner),
          bi_flat_J(other.bi_flat_J),
          bi_flat_needs_twist(other.bi_flat_needs_twist),
          bi_flat_wrap(other.bi_flat_wrap),
          bi_flat_D2(other.bi_flat_D2),
          twist_matrices(other.twist_matrices),
          rotation_axis(other.rotation_axis),
          bilinear_wrap_dir(other.bilinear_wrap_dir),
          boundary_sites_per_dim(other.boundary_sites_per_dim),
          boundary_thickness(other.boundary_thickness),
          sublattice_frames(other.sublattice_frames)
    {}

    // ============================================================
    // UTILITY METHODS
    // ============================================================

    /**
     * Flatten multi-index to linear site index
     */
    size_t flatten_index(size_t i, size_t j, size_t k, size_t atom) const {
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
    size_t flatten_index_periodic(int i, int j, int k, size_t atom) const {
        return flatten_index(periodic_boundary(i, dim1),
                           periodic_boundary(j, dim2),
                           periodic_boundary(k, dim3),
                           atom);
    }

    /**
     * Generate random spin on n-sphere
     */
    SpinVector gen_random_spin(float spin_l) {
        SpinVector spin(spin_dim);
        gen_random_spin_into(spin.data(), spin_l);
        return spin;
    }

    /**
     * Zero-allocation variant: write a uniformly random length-spin_l vector
     * into the caller-provided buffer (must have at least spin_dim doubles).
     *
     * Used by the hot Metropolis loop to avoid allocating an Eigen::VectorXd
     * per proposed move.
     */
    void gen_random_spin_into(double* out, float spin_l) const {
        if (spin_dim == 3) {
            // Marsaglia (1972) method for uniform sampling on the
            // 2-sphere: rejection-sample u1, u2 uniformly in the
            // unit disk, then map to the sphere algebraically.
            //
            //   x = 2*u1 * sqrt(1 - s)
            //   y = 2*u2 * sqrt(1 - s)
            //   z = 1 - 2*s,    where s = u1^2 + u2^2 < 1
            //
            // Acceptance probability is pi/4 ~ 0.785, so on average
            // 1.27 rejection iterations per call. This replaces the
            // cos+sin+sqrt formulation: same distribution, same
            // 2-RNG-calls-per-accept budget, but no transcendental
            // calls. On the Metropolis hot loop this is a few ns
            // per proposed move (~ 5-10% of the per-site work).
            double u1, u2, s;
            do {
                u1 = 2.0 * random_double_lehman(0.0, 1.0) - 1.0;
                u2 = 2.0 * random_double_lehman(0.0, 1.0) - 1.0;
                s  = u1 * u1 + u2 * u2;
            } while (s >= 1.0);
            const double factor = 2.0 * std::sqrt(1.0 - s);
            out[0] = double(spin_l) * factor * u1;
            out[1] = double(spin_l) * factor * u2;
            out[2] = double(spin_l) * (1.0 - 2.0 * s);
            return;
        }
        // General n-sphere via gaussian-on-sphere fallback (rejection of
        // near-zero norms to avoid biasing). spin_dim should be small.
        double sum_sq = 0.0;
        do {
            sum_sq = 0.0;
            for (size_t i = 0; i < spin_dim; ++i) {
                out[i] = random_double_lehman(-1.0, 1.0);
                sum_sq += out[i] * out[i];
            }
        } while (sum_sq < 1e-20);
        const double inv_norm = double(spin_l) / std::sqrt(sum_sq);
        for (size_t i = 0; i < spin_dim; ++i) out[i] *= inv_norm;
    }

    /**
     * Build boundary site lists for twist updates
     */
    void build_boundary_sites() {
        for (size_t d = 0; d < 3; ++d) {
            boundary_sites_per_dim[d].clear();
        }

        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms; ++atom) {
                        size_t idx = flatten_index(i, j, k, atom);
                        
                        // Check if site is near boundary in each dimension
                        if (i < boundary_thickness[0] || i >= dim1 - boundary_thickness[0]) {
                            boundary_sites_per_dim[0].push_back(idx);
                        }
                        if (j < boundary_thickness[1] || j >= dim2 - boundary_thickness[1]) {
                            boundary_sites_per_dim[1].push_back(idx);
                        }
                        if (k < boundary_thickness[2] || k >= dim3 - boundary_thickness[2]) {
                            boundary_sites_per_dim[2].push_back(idx);
                        }
                    }
                }
            }
        }
    }

    /**
     * Apply twist matrix to partner spin crossing boundary
     */
    inline SpinVector apply_twist_to_partner_spin(const SpinVector& partner_spin,
                                                   const array<int8_t, 3>& wrap) const {
        SpinVector result = partner_spin;
        for (size_t d = 0; d < 3; ++d) {
            if (wrap[d] == +1) {
                result = twist_matrices[d] * result;
            } else if (wrap[d] == -1) {
                result = twist_matrices[d].transpose() * result;
            }
        }
        return result;
    }

    /**
     * Set twist rotation axes
     */
    void set_twist_axes(const array<SpinVector, 3>& axes) {
        rotation_axis = axes;
        // Initialize twist matrices and angles to zero rotation
        for (size_t d = 0; d < 3; ++d) {
            twist_angles[d] = 0.0;
            twist_matrices[d] = rotation_from_axis_angle(rotation_axis[d], 0.0);
        }
    }

    /**
     * Create rotation matrix from axis-angle representation (Rodrigues' formula)
     */
    static SpinMatrix rotation_from_axis_angle(const SpinVector& axis, double angle) {
        size_t N = axis.size();
        if (N != 3) {
            return SpinMatrix::Identity(N, N); // Only defined for 3D spins
        }
        
        SpinVector n = axis.normalized();
        double c = std::cos(angle);
        double s = std::sin(angle);
        
        SpinMatrix K = SpinMatrix::Zero(3, 3);
        K(0, 1) = -n(2);
        K(0, 2) =  n(1);
        K(1, 0) =  n(2);
        K(1, 2) = -n(0);
        K(2, 0) = -n(1);
        K(2, 1) =  n(0);
        
        return SpinMatrix::Identity(3, 3) + s * K + (1.0 - c) * K * K;
    }

    // ============================================================
    // ENERGY CALCULATIONS
    // ============================================================

    /**
     * Compute energy of a single site
     */
    double site_energy(const SpinVector& spin_here, size_t site_index) const {
        double E = 0.0;
        
        // Zeeman energy: -B · S
        E -= spin_here.dot(field[site_index]);
        
        // On-site anisotropy: S^T A S
        E += spin_here.dot(onsite_interaction[site_index] * spin_here);
        
        // Bilinear interactions: S_i^T J S_j
        size_t n_bi = bilinear_partners[site_index].size();
        for (size_t n = 0; n < n_bi; ++n) {
            size_t partner = bilinear_partners[site_index][n];
            SpinVector partner_spin = apply_twist_to_partner_spin(
                spins[partner], bilinear_wrap_dir[site_index][n]);
            E += spin_here.dot(bilinear_interaction[site_index][n] * partner_spin);
        }
        
        // Trilinear interactions: contract(T, S_i, S_j, S_k)
        size_t n_tri = trilinear_partners[site_index].size();
        for (size_t n = 0; n < n_tri; ++n) {
            size_t p1 = trilinear_partners[site_index][n][0];
            size_t p2 = trilinear_partners[site_index][n][1];
            E += contract_trilinear(trilinear_interaction[site_index][n],
                                   spin_here, 
                                   spins[p1], 
                                   spins[p2]);
        }
        
        return E;
    }

    /**
     * Compute energy difference for spin flip (optimized for Metropolis)
     */
    double site_energy_diff(const SpinVector& new_spin, const SpinVector& old_spin, 
                           size_t site_index) const {
        SpinVector delta = new_spin - old_spin;
        double dE = 0.0;
        
        // Zeeman
        dE -= delta.dot(field[site_index]);
        
        // On-site
        dE += new_spin.dot(onsite_interaction[site_index] * new_spin) 
            - old_spin.dot(onsite_interaction[site_index] * old_spin);
        
        // Bilinear
        size_t n_bi = bilinear_partners[site_index].size();
        for (size_t n = 0; n < n_bi; ++n) {
            size_t partner = bilinear_partners[site_index][n];
            SpinVector partner_spin = apply_twist_to_partner_spin(
                spins[partner], bilinear_wrap_dir[site_index][n]);
            dE += delta.dot(bilinear_interaction[site_index][n] * partner_spin);
        }
        
        // Trilinear
        size_t n_tri = trilinear_partners[site_index].size();
        for (size_t n = 0; n < n_tri; ++n) {
            size_t p1 = trilinear_partners[site_index][n][0];
            size_t p2 = trilinear_partners[site_index][n][1];
            dE += contract_trilinear(trilinear_interaction[site_index][n],
                                    delta,
                                    spins[p1],
                                    spins[p2]);
        }
        
        return dE;
    }

    /**
     * Zero-allocation Δenergy for a proposed Metropolis move.
     *
     * `new_spin_buf` and `old_spin_buf` are raw double pointers (typically
     * stack buffers in the Metropolis hot loop and `spins[site].data()`).
     * Avoids constructing two Eigen::VectorXd temporaries (heap allocation +
     * Eigen expression-template overhead) per proposed move, which is the
     * single biggest CPU win for a Metropolis sweep on small spin_dim.
     *
     * Equivalent to site_energy_diff(new, old, site).
     */
    double site_energy_diff_flat(const double* __restrict new_spin_buf,
                                 const double* __restrict old_spin_buf,
                                 size_t site_index) const {
        // Stack scratch — spin_dim is at most 8 for SU(3); 16 leaves headroom.
        constexpr size_t MAX_SPIN_DIM = 16;
        double delta_buf[MAX_SPIN_DIM];
        for (size_t d = 0; d < spin_dim; ++d) {
            delta_buf[d] = new_spin_buf[d] - old_spin_buf[d];
        }

        double dE = 0.0;

        // Zeeman: -delta · B
        const double* B = field[site_index].data();
        for (size_t d = 0; d < spin_dim; ++d) dE -= delta_buf[d] * B[d];

        // On-site: new^T A new - old^T A old
        const auto& A = onsite_interaction[site_index];
        for (size_t a = 0; a < spin_dim; ++a) {
            double row_new = 0.0, row_old = 0.0;
            for (size_t b = 0; b < spin_dim; ++b) {
                row_new += A(a, b) * new_spin_buf[b];
                row_old += A(a, b) * old_spin_buf[b];
            }
            dE += new_spin_buf[a] * row_new - old_spin_buf[a] * row_old;
        }

        // Bilinear: delta^T J S_partner.
        //
        // Hot path: read partners and J matrices straight from the flat
        // SoA tables (built once in initialize()). This eliminates the
        // two pointer indirections of vector<vector<Eigen::Matrix3d>>
        // and lets each site's J matrices stream from L1 cache.
        //
        // The twist-BC branch is taken only at the lattice boundary AND
        // only when twist angles are nonzero, so __builtin_expect drives
        // the branch predictor toward the no-twist path.
        const size_t bi_base = bi_flat_offset[site_index];
        const size_t bi_end  = bi_flat_offset[site_index + 1];
        const size_t D2      = bi_flat_D2;
        for (size_t k = bi_base; k < bi_end; ++k) {
            const size_t partner = bi_flat_partner[k];
            const double* __restrict P = spins[partner].data();
            const double* __restrict J = &bi_flat_J[k * D2];

            if (__builtin_expect(bi_flat_needs_twist[k] != 0, 0)) {
                // Cold path: apply twist transform to the partner spin.
                double twist_buf[3] = {P[0], P[1], P[2]};
                const auto& wrap = bi_flat_wrap[k];
                for (size_t dim = 0; dim < 3; ++dim) {
                    if (wrap[dim] == 0) continue;
                    double tmp[3] = {0.0, 0.0, 0.0};
                    if (wrap[dim] > 0) {
                        for (size_t d = 0; d < 3; ++d) {
                            tmp[d] = twist_matrices[dim](d, 0) * twist_buf[0]
                                   + twist_matrices[dim](d, 1) * twist_buf[1]
                                   + twist_matrices[dim](d, 2) * twist_buf[2];
                        }
                    } else {
                        for (size_t d = 0; d < 3; ++d) {
                            tmp[d] = twist_matrices[dim](0, d) * twist_buf[0]
                                   + twist_matrices[dim](1, d) * twist_buf[1]
                                   + twist_matrices[dim](2, d) * twist_buf[2];
                        }
                    }
                    twist_buf[0] = tmp[0]; twist_buf[1] = tmp[1]; twist_buf[2] = tmp[2];
                }
                for (size_t a = 0; a < spin_dim; ++a) {
                    double row = 0.0;
                    for (size_t b = 0; b < spin_dim; ++b)
                        row += J[a * spin_dim + b] * twist_buf[b];
                    dE += delta_buf[a] * row;
                }
            } else {
                // Common case: tight matrix-vector multiply, all loads
                // from contiguous memory. Auto-vectorizable.
                for (size_t a = 0; a < spin_dim; ++a) {
                    double row = 0.0;
                    for (size_t b = 0; b < spin_dim; ++b)
                        row += J[a * spin_dim + b] * P[b];
                    dE += delta_buf[a] * row;
                }
            }
        }

        // Trilinear: delta . (T : S_j ⊗ S_k)
        const size_t n_tri = trilinear_partners[site_index].size();
        for (size_t n = 0; n < n_tri; ++n) {
            const size_t p1 = trilinear_partners[site_index][n][0];
            const size_t p2 = trilinear_partners[site_index][n][1];
            const double* S_j = spins[p1].data();
            const double* S_k = spins[p2].data();
            const auto& T = trilinear_interaction[site_index][n];

            // Cache S_j[b] * S_k[c] outer product (small spin_dim)
            double S_jk[64]; // up to spin_dim=8
            for (size_t b = 0; b < spin_dim; ++b) {
                for (size_t c = 0; c < spin_dim; ++c) {
                    S_jk[b * spin_dim + c] = S_j[b] * S_k[c];
                }
            }
            for (size_t a = 0; a < spin_dim; ++a) {
                double sum_a = 0.0;
                for (size_t b = 0; b < spin_dim; ++b) {
                    for (size_t c = 0; c < spin_dim; ++c) {
                        sum_a += T[a](b, c) * S_jk[b * spin_dim + c];
                    }
                }
                dE += delta_buf[a] * sum_a;
            }
        }

        return dE;
    }

    /**
     * Compute total energy of configuration
     */
    double total_energy(const SpinConfig& config) const {
        double E = 0.0;
        
        for (size_t i = 0; i < lattice_size; ++i) {
            // Zeeman
            E -= config[i].dot(field[i]);
            
            // On-site
            E += config[i].dot(onsite_interaction[i] * config[i]);
            
            // Bilinear (count each pair once)
            size_t n_bi = bilinear_partners[i].size();
            for (size_t n = 0; n < n_bi; ++n) {
                size_t partner = bilinear_partners[i][n];
                if (partner > i) { // Avoid double counting
                    SpinVector partner_spin = apply_twist_to_partner_spin(
                        config[partner], bilinear_wrap_dir[i][n]);
                    E += config[i].dot(bilinear_interaction[i][n] * partner_spin);
                }
            }
            
            // Trilinear (count each triple once)
            size_t n_tri = trilinear_partners[i].size();
            for (size_t n = 0; n < n_tri; ++n) {
                size_t p1 = trilinear_partners[i][n][0];
                size_t p2 = trilinear_partners[i][n][1];
                if (p1 > i && p2 > i) { // Avoid double counting
                    E += contract_trilinear(trilinear_interaction[i][n],
                                           config[i],
                                           config[p1],
                                           config[p2]);
                }
            }
        }
        
        return E;
    }

    /**
     * Total energy of current spin configuration (no-arg wrapper)
     */
    double total_energy() const {
        return total_energy(spins);
    }

    /**
     * Energy per site
     */
    double energy_density() const {
        return total_energy(spins) / lattice_size;
    }

    /**
     * Compute total energy directly from flat state array (zero-allocation version)
     * Includes all interaction terms with proper double-counting avoidance
     */
    double total_energy_flat(const double* state_flat) const {
        double E = 0.0;
        
        for (size_t i = 0; i < lattice_size; ++i) {
            const double* S_i = &state_flat[i * spin_dim];
            
            // Zeeman: -B·S
            for (size_t d = 0; d < spin_dim; ++d) {
                E -= field[i](d) * S_i[d];
            }
            
            // On-site: S^T A S
            for (size_t d = 0; d < spin_dim; ++d) {
                for (size_t d2 = 0; d2 < spin_dim; ++d2) {
                    E += S_i[d] * onsite_interaction[i](d, d2) * S_i[d2];
                }
            }
            
            // Bilinear: S_i^T J S_j (count each pair once)
            size_t n_bi = bilinear_partners[i].size();
            for (size_t n = 0; n < n_bi; ++n) {
                size_t partner = bilinear_partners[i][n];
                if (partner > i) { // Avoid double counting
                    const double* S_j = &state_flat[partner * spin_dim];
                    
                    // Apply twist if needed
                    const auto& wrap = bilinear_wrap_dir[i][n];
                    if (spin_dim == 3 && (wrap[0] != 0 || wrap[1] != 0 || wrap[2] != 0)) {
                        double S_j_twisted[3];
                        std::copy(S_j, S_j + 3, S_j_twisted);
                        
                        for (size_t d = 0; d < 3; ++d) {
                            if (wrap[d] != 0) {
                                double twisted[3] = {0, 0, 0};
                                // For positive wrap, apply twist_matrices[d]
                                // For negative wrap, apply transpose (inverse for rotations)
                                if (wrap[d] > 0) {
                                    for (size_t d2 = 0; d2 < 3; ++d2) {
                                        twisted[d2] += twist_matrices[d](d2, 0) * S_j_twisted[0];
                                        twisted[d2] += twist_matrices[d](d2, 1) * S_j_twisted[1];
                                        twisted[d2] += twist_matrices[d](d2, 2) * S_j_twisted[2];
                                    }
                                } else {
                                    // Apply transpose: R^T[d2, d3] = R[d3, d2]
                                    for (size_t d2 = 0; d2 < 3; ++d2) {
                                        twisted[d2] += twist_matrices[d](0, d2) * S_j_twisted[0];
                                        twisted[d2] += twist_matrices[d](1, d2) * S_j_twisted[1];
                                        twisted[d2] += twist_matrices[d](2, d2) * S_j_twisted[2];
                                    }
                                }
                                std::copy(twisted, twisted + 3, S_j_twisted);
                            }
                        }
                        
                        // S_i^T * J * S_j_twisted
                        for (size_t d = 0; d < spin_dim; ++d) {
                            for (size_t d2 = 0; d2 < spin_dim; ++d2) {
                                E += S_i[d] * bilinear_interaction[i][n](d, d2) * S_j_twisted[d2];
                            }
                        }
                    } else {
                        // No twist needed
                        for (size_t d = 0; d < spin_dim; ++d) {
                            for (size_t d2 = 0; d2 < spin_dim; ++d2) {
                                E += S_i[d] * bilinear_interaction[i][n](d, d2) * S_j[d2];
                            }
                        }
                    }
                }
            }
            
            // Trilinear: contract(T, S_i, S_j, S_k) (count each triple once)
            size_t n_tri = trilinear_partners[i].size();
            for (size_t n = 0; n < n_tri; ++n) {
                size_t p1 = trilinear_partners[i][n][0];
                size_t p2 = trilinear_partners[i][n][1];
                
                if (p1 > i && p2 > i) { // Avoid double counting
                    const double* S_j = &state_flat[p1 * spin_dim];
                    const double* S_k = &state_flat[p2 * spin_dim];
                    const auto& T = trilinear_interaction[i][n];
                    
                    // Contract tensor: E += sum_{abc} T[a][b,c] * S_i[a] * S_j[b] * S_k[c]
                    // Optimize by caching S_j[b] * S_k[c] products
                    if (spin_dim <= 3) {
                        // Small dimension: unroll and cache products
                        double S_jk[9]; // max 3x3 = 9 products
                        for (size_t b = 0; b < spin_dim; ++b) {
                            for (size_t c = 0; c < spin_dim; ++c) {
                                S_jk[b * spin_dim + c] = S_j[b] * S_k[c];
                            }
                        }
                        
                        for (size_t a = 0; a < spin_dim; ++a) {
                            double temp = 0.0;
                            for (size_t b = 0; b < spin_dim; ++b) {
                                for (size_t c = 0; c < spin_dim; ++c) {
                                    temp += T[a](b, c) * S_jk[b * spin_dim + c];
                                }
                            }
                            E += S_i[a] * temp;
                        }
                    } else {
                        // Larger dimension: optimize for cache locality
                        // Strategy: compute matrix-vector products incrementally
                        for (size_t a = 0; a < spin_dim; ++a) {
                            double sum_a = 0.0;
                            const auto& T_a = T[a]; // T[a] is a spin_dim x spin_dim matrix
                            
                            // Compute T_a * outer(S_j, S_k) contracted with basis vectors
                            // This is: sum_{b,c} T_a(b,c) * S_j[b] * S_k[c]
                            for (size_t b = 0; b < spin_dim; ++b) {
                                double temp = 0.0;
                                // Inner loop: accumulate over c with S_j[b] factored out
                                for (size_t c = 0; c < spin_dim; ++c) {
                                    temp += T_a(b, c) * S_k[c];
                                }
                                sum_a += S_j[b] * temp;
                            }
                            
                            E += S_i[a] * sum_a;
                        }
                    }
                }
            }
        }
        
        return E;
    }

    /**
     * Compute local field at a site: H_eff = -dE/dS
     */
    SpinVector get_local_field(size_t site_index) const {
        SpinVector H = -field[site_index];
        
        // On-site contribution
        H += 2.0 * onsite_interaction[site_index] * spins[site_index];
        
        // Bilinear
        size_t n_bi = bilinear_partners[site_index].size();
        for (size_t n = 0; n < n_bi; ++n) {
            size_t partner = bilinear_partners[site_index][n];
            SpinVector partner_spin = apply_twist_to_partner_spin(
                spins[partner], bilinear_wrap_dir[site_index][n]);
            H += bilinear_interaction[site_index][n] * partner_spin;
        }
        
        // Trilinear (simplified - approximate field contribution)
        size_t n_tri = trilinear_partners[site_index].size();
        for (size_t n = 0; n < n_tri; ++n) {
            size_t p1 = trilinear_partners[site_index][n][0];
            size_t p2 = trilinear_partners[site_index][n][1];
            // Approximate: treat as coupling between spin products
            double coupling = contract_trilinear(trilinear_interaction[site_index][n],
                                                SpinVector::Ones(spin_dim),
                                                spins[p1],
                                                spins[p2]);
            H += coupling * spins[site_index];
        }
        
        return H;
    }


    /**
     * Compute local field from flat state array (zero-allocation version for ODE integration)
     */
    void get_local_field_flat(const double* state_flat, size_t site_index, double* H_out) const {
        // Initialize H = -B
        for (size_t d = 0; d < spin_dim; ++d) {
            H_out[d] = -field[site_index](d);
        }
        
        // On-site: H += 2*A*S
        const double* S_i = &state_flat[site_index * spin_dim];
        for (size_t d = 0; d < spin_dim; ++d) {
            for (size_t d2 = 0; d2 < spin_dim; ++d2) {
                H_out[d] += 2.0 * onsite_interaction[site_index](d, d2) * S_i[d2];
            }
        }
        
        // Bilinear: H += J * S_partner.
        // Hot path uses the SoA bilinear table built in initialize().
        const size_t bi_base = bi_flat_offset[site_index];
        const size_t bi_end  = bi_flat_offset[site_index + 1];
        const size_t D2      = bi_flat_D2;
        for (size_t k = bi_base; k < bi_end; ++k) {
            const size_t partner = bi_flat_partner[k];
            const double* __restrict S_partner = &state_flat[partner * spin_dim];
            const double* __restrict J         = &bi_flat_J[k * D2];

            if (__builtin_expect(bi_flat_needs_twist[k] != 0, 0)) {
                double S_twisted[8] = {0.0};
                for (size_t d = 0; d < 3; ++d) S_twisted[d] = S_partner[d];
                const auto& wrap = bi_flat_wrap[k];
                for (size_t dim = 0; dim < 3; ++dim) {
                    if (wrap[dim] == 0) continue;
                    double temp[3] = {0.0, 0.0, 0.0};
                    if (wrap[dim] > 0) {
                        for (size_t d = 0; d < 3; ++d) {
                            for (size_t d2 = 0; d2 < 3; ++d2)
                                temp[d] += twist_matrices[dim](d, d2) * S_twisted[d2];
                        }
                    } else {
                        for (size_t d = 0; d < 3; ++d) {
                            for (size_t d2 = 0; d2 < 3; ++d2)
                                temp[d] += twist_matrices[dim](d2, d) * S_twisted[d2];
                        }
                    }
                    for (size_t d = 0; d < 3; ++d) S_twisted[d] = temp[d];
                }
                for (size_t d = 0; d < spin_dim; ++d) {
                    double row = 0.0;
                    for (size_t d2 = 0; d2 < spin_dim; ++d2)
                        row += J[d * spin_dim + d2] * S_twisted[d2];
                    H_out[d] += row;
                }
            } else {
                for (size_t d = 0; d < spin_dim; ++d) {
                    double row = 0.0;
                    for (size_t d2 = 0; d2 < spin_dim; ++d2)
                        row += J[d * spin_dim + d2] * S_partner[d2];
                    H_out[d] += row;
                }
            }
        }
        
        // Trilinear: H += dE/dS_i from trilinear terms
        // For E = sum_{abc} T[a](b,c) * S_i[a] * S_j[b] * S_k[c]
        // dE/dS_i[a] = sum_{bc} T[a](b,c) * S_j[b] * S_k[c]
        size_t n_tri = trilinear_partners[site_index].size();
        for (size_t n = 0; n < n_tri; ++n) {
            size_t p1 = trilinear_partners[site_index][n][0];
            size_t p2 = trilinear_partners[site_index][n][1];
            
            const double* S_j = &state_flat[p1 * spin_dim];
            const double* S_k = &state_flat[p2 * spin_dim];
            const auto& T = trilinear_interaction[site_index][n];
            
            // Compute field contribution: H[a] += sum_{bc} T[a](b,c) * S_j[b] * S_k[c]
            if (spin_dim <= 3) {
                // Small dimension: cache S_j[b] * S_k[c] products
                double S_jk[9];
                for (size_t b = 0; b < spin_dim; ++b) {
                    for (size_t c = 0; c < spin_dim; ++c) {
                        S_jk[b * spin_dim + c] = S_j[b] * S_k[c];
                    }
                }
                
                for (size_t a = 0; a < spin_dim; ++a) {
                    double temp = 0.0;
                    for (size_t b = 0; b < spin_dim; ++b) {
                        for (size_t c = 0; c < spin_dim; ++c) {
                            temp += T[a](b, c) * S_jk[b * spin_dim + c];
                        }
                    }
                    H_out[a] += temp;
                }
            } else {
                // Larger dimension: cache-friendly pattern
                for (size_t a = 0; a < spin_dim; ++a) {
                    double sum_a = 0.0;
                    const auto& T_a = T[a];
                    
                    for (size_t b = 0; b < spin_dim; ++b) {
                        double temp = 0.0;
                        for (size_t c = 0; c < spin_dim; ++c) {
                            temp += T_a(b, c) * S_k[c];
                        }
                        sum_a += S_j[b] * temp;
                    }
                    
                    H_out[a] += sum_a;
                }
            }
        }
    }
    
    /**
     * Compute local field at a site: H_eff = -dE/dS
     */
    SpinVector get_local_field_lattice(SpinConfig curr_spins, size_t site_index) const {
        SpinVector H = -field[site_index];
        
        // On-site contribution
        H += 2.0 * onsite_interaction[site_index] * curr_spins[site_index];
        
        // Bilinear
        size_t n_bi = bilinear_partners[site_index].size();
        for (size_t n = 0; n < n_bi; ++n) {
            size_t partner = bilinear_partners[site_index][n];
            SpinVector partner_spin = apply_twist_to_partner_spin(
                curr_spins[partner], bilinear_wrap_dir[site_index][n]);
            H += bilinear_interaction[site_index][n] * partner_spin;
        }
        
        // Trilinear (simplified - approximate field contribution)
        size_t n_tri = trilinear_partners[site_index].size();
        for (size_t n = 0; n < n_tri; ++n) {
            size_t p1 = trilinear_partners[site_index][n][0];
            size_t p2 = trilinear_partners[site_index][n][1];
            // Approximate: treat as coupling between spin products
            double coupling = contract_trilinear(trilinear_interaction[site_index][n],
                                                SpinVector::Ones(spin_dim),
                                                curr_spins[p1],
                                                curr_spins[p2]);
            H += coupling * curr_spins[site_index];
        }
        
        return H;
    }


    // ============================================================
    // AUTOCORRELATION ANALYSIS
    // ============================================================

    /**
     * Compute integrated autocorrelation time from energy time series
     */
    AutocorrelationResult compute_autocorrelation(const vector<double>& energies, 
                                                   size_t base_interval = 10);

    // ============================================================
    // BINNING ANALYSIS FOR ERROR ESTIMATION
    // ============================================================

    /**
     * Binning analysis for error estimation of a scalar observable
     * 
     * Performs recursive blocking to estimate statistical error with autocorrelation.
     * The error plateaus when bin size exceeds the correlation length.
     * 
     * @param data Vector of observable measurements
     * @return BinningResult containing mean, error, and binning information
     */
    static BinningResult binning_analysis(const vector<double>& data) {
        BinningResult result;
        
        if (data.empty()) {
            result.mean = 0.0;
            result.error = 0.0;
            result.tau_int = 1.0;
            result.optimal_bin_level = 0;
            return result;
        }
        
        size_t n = data.size();
        
        // Compute mean
        result.mean = std::accumulate(data.begin(), data.end(), 0.0) / double(n);
        
        if (n < 4) {
            // Too few samples for meaningful analysis
            double var = 0.0;
            for (double x : data) var += (x - result.mean) * (x - result.mean);
            result.error = std::sqrt(var / (n * (n - 1)));
            result.tau_int = 1.0;
            result.optimal_bin_level = 0;
            return result;
        }
        
        // Recursive blocking: progressively halve the data by averaging pairs
        vector<double> binned_data = data;
        size_t level = 0;
        size_t max_levels = static_cast<size_t>(std::log2(n)) - 1;
        
        result.errors_by_level.reserve(max_levels);
        
        while (binned_data.size() >= 4) {
            size_t m = binned_data.size();
            
            // Compute variance at this level
            double sum = 0.0, sum2 = 0.0;
            for (double x : binned_data) {
                sum += x;
                sum2 += x * x;
            }
            double mean_level = sum / m;
            double var_level = (sum2 / m - mean_level * mean_level);
            double error_level = std::sqrt(var_level / (m - 1));
            
            result.errors_by_level.push_back(error_level);
            
            // Block the data: average consecutive pairs
            vector<double> new_binned;
            new_binned.reserve(m / 2);
            for (size_t i = 0; i + 1 < m; i += 2) {
                new_binned.push_back(0.5 * (binned_data[i] + binned_data[i + 1]));
            }
            binned_data = std::move(new_binned);
            ++level;
        }
        
        // Find optimal level where error has plateaued
        // Look for where error stops growing significantly
        result.optimal_bin_level = 0;
        if (result.errors_by_level.size() > 2) {
            double max_error = 0.0;
            for (size_t l = 0; l < result.errors_by_level.size(); ++l) {
                if (result.errors_by_level[l] > max_error) {
                    max_error = result.errors_by_level[l];
                    result.optimal_bin_level = l;
                }
            }
        }
        
        // Use error from optimal level, or last level with at least 4 samples
        if (!result.errors_by_level.empty()) {
            // Use error from level where error has approximately plateaued
            size_t use_level = std::min(result.optimal_bin_level + 1, result.errors_by_level.size() - 1);
            result.error = result.errors_by_level[use_level];
            
            // Estimate tau_int from ratio of blocked variance to naive variance
            if (result.errors_by_level[0] > 1e-20) {
                double ratio = result.error / result.errors_by_level[0];
                result.tau_int = 0.5 * ratio * ratio;
            } else {
                result.tau_int = 1.0;
            }
        } else {
            result.error = 0.0;
            result.tau_int = 1.0;
        }
        
        return result;
    }

    /**
     * Binning analysis for vector observable (component-wise)
     * 
     * @param data Vector of vector measurements
     * @return Vector of BinningResult, one per component
     */
    static vector<BinningResult> binning_analysis_vector(const vector<SpinVector>& data) {
        if (data.empty()) return {};
        
        size_t dim = data[0].size();
        vector<BinningResult> results(dim);
        
        // Extract each component and analyze separately
        for (size_t d = 0; d < dim; ++d) {
            vector<double> component(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                component[i] = data[i](d);
            }
            results[d] = binning_analysis(component);
        }
        
        return results;
    }

    // ============================================================
    // SUBLATTICE MAGNETIZATION
    // ============================================================

    /**
     * Compute magnetization for each sublattice separately
     * 
     * @return Vector of SpinVectors, one per sublattice (N_atoms sublattices)
     *         Each vector is averaged over all unit cells in the local sublattice frame
     */
    vector<SpinVector> magnetization_sublattice() const {
        vector<SpinVector> M_sub(N_atoms);
        size_t n_cells = dim1 * dim2 * dim3;
        double inv_n_cells = 1.0 / double(n_cells);
        
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            M_sub[atom] = SpinVector::Zero(spin_dim);
        }
        
        // Flat loop over all sites - more cache-friendly
        for (size_t site = 0; site < lattice_size; ++site) {
            size_t atom = site % N_atoms;
            M_sub[atom] += spins[site];
        }
        
        // Normalize by number of unit cells
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            M_sub[atom] *= inv_n_cells;
        }
        
        return M_sub;
    }

    /**
     * Compute magnetization for each sublattice from flat state array
     * 
     * @param state_flat Flat spin state array [lattice_size * spin_dim]
     * @param M_sub_out Output: N_atoms SpinVectors for sublattice magnetizations (in local frame)
     */
    void magnetization_sublattice_from_flat(const double* state_flat, 
                                             vector<SpinVector>& M_sub_out) const {
        size_t n_cells = dim1 * dim2 * dim3;
        
        M_sub_out.resize(N_atoms);
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            M_sub_out[atom] = SpinVector::Zero(spin_dim);
        }
        
        // Sum over all sites (in local frame)
        for (size_t i = 0; i < lattice_size; ++i) {
            size_t atom = i % N_atoms;
            const double* spin_ptr = state_flat + i * spin_dim;
            
            for (size_t mu = 0; mu < spin_dim; ++mu) {
                M_sub_out[atom](mu) += spin_ptr[mu];
            }
        }
        
        // Normalize
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            M_sub_out[atom] /= double(n_cells);
        }
    }

    // ============================================================
    // KAGOME PLANE ORDER PARAMETERS (PYROCHLORE NON-KRAMERS)
    // For pyrochlore: sublattices 0=apex, 1,2,3=kagome base
    // 
    // From unitcell_builders.cpp build_pyrochlore_non_kramer():
    // 
    // KAGOME NN BONDS (sublattices 1,2,3):
    // ┌─────────┬─────────────────┬─────────────────┬────────┐
    // │ Bond    │ Intra-cell      │ Inter-cell      │ J type │
    // ├─────────┼─────────────────┼─────────────────┼────────┤
    // │ (1,2)   │ (0, 0, 0)       │ (-1,+1, 0)      │ Jy     │
    // │ (1,3)   │ (0, 0, 0)       │ (-1, 0,+1)      │ Jx     │
    // │ (2,3)   │ (0, 0, 0)       │ ( 0,+1,-1)      │ Jz     │
    // └─────────┴─────────────────┴─────────────────┴────────┘
    // 
    // TRIANGLES (chirality):
    //   Per unit cell: 1 "up" triangle = {1(i,j,k), 2(i,j,k), 3(i,j,k)}
    //   Uses intra-cell bonds only. The inter-cell bonds connect to
    //   different triangles (no closed "down" kagome triangles on 1,2,3).
    // 
    // DIMERS (nematic):
    //   Per unit cell: 6 bonds = 3 types × 2 (intra + inter)
    //   Each bond type has 2N_cells total bonds in the lattice.
    // 
    // Spins are stored in LOCAL FRAME (x,y,z per sublattice).
    // ============================================================

    /**
     * Pyrochlore local frame: columns of R_sub give x_hat, y_hat, z_hat in global coords.
     * S_global = S_local_x * x_hat[sub] + S_local_y * y_hat[sub] + S_local_z * z_hat[sub]
     *
     * x_hat[4][3], y_hat[4][3], z_hat[4][3]  (sublattice, global xyz)
     */
    struct PyrochloreLocalFrame {
        // z_hat = local Ising axis
        static constexpr double z_hat[4][3] = {
            { 1.0/sqrt(3.0),  1.0/sqrt(3.0),  1.0/sqrt(3.0)},
            { 1.0/sqrt(3.0), -1.0/sqrt(3.0), -1.0/sqrt(3.0)},
            {-1.0/sqrt(3.0),  1.0/sqrt(3.0), -1.0/sqrt(3.0)},
            {-1.0/sqrt(3.0), -1.0/sqrt(3.0),  1.0/sqrt(3.0)}
        };
        // y_hat
        static constexpr double y_hat[4][3] = {
            { 0.0,            -1.0/sqrt(2.0),  1.0/sqrt(2.0)},
            { 0.0,             1.0/sqrt(2.0), -1.0/sqrt(2.0)},
            { 0.0,            -1.0/sqrt(2.0), -1.0/sqrt(2.0)},
            { 0.0,             1.0/sqrt(2.0),  1.0/sqrt(2.0)}
        };
        // x_hat
        static constexpr double x_hat[4][3] = {
            {-2.0/sqrt(6.0),  1.0/sqrt(6.0),  1.0/sqrt(6.0)},
            {-2.0/sqrt(6.0), -1.0/sqrt(6.0), -1.0/sqrt(6.0)},
            { 2.0/sqrt(6.0),  1.0/sqrt(6.0), -1.0/sqrt(6.0)},
            { 2.0/sqrt(6.0), -1.0/sqrt(6.0),  1.0/sqrt(6.0)}
        };

        /** Transform a local-frame spin (Sx,Sy,Sz) on sublattice sub to global frame */
        static void to_global(int sub, double Sx_l, double Sy_l, double Sz_l,
                              double& Sx_g, double& Sy_g, double& Sz_g) {
            Sx_g = Sx_l * x_hat[sub][0] + Sy_l * y_hat[sub][0] + Sz_l * z_hat[sub][0];
            Sy_g = Sx_l * x_hat[sub][1] + Sy_l * y_hat[sub][1] + Sz_l * z_hat[sub][1];
            Sz_g = Sx_l * x_hat[sub][2] + Sy_l * y_hat[sub][2] + Sz_l * z_hat[sub][2];
        }
    };

    /**
     * Structure to hold all pyrochlore order parameters
     * Computed in a single pass for efficiency
     */
    struct PyrochloreOrderParameters {
        // --- Local-frame order parameters ---
        double scalar_chirality;           // χ = <S1·(S2×S3)>  (local frame)
        Eigen::Vector3d vector_chirality;  // κ = <S1×S2 + S2×S3 + S3×S1>  (local frame)
        Eigen::Matrix3d nematic_order;     // Q[bond_type][component] = <Si·Sj>  (local frame)
        double monopole_density;           // <Q> per tetrahedron (signed)
        Eigen::Vector4d monopole_by_sublattice;  // Monopole density by minority sublattice

        // --- Global-frame order parameters ---
        double scalar_chirality_global;          // χ_global = <S1_g·(S2_g×S3_g)>
        Eigen::Vector3d vector_chirality_global;  // κ_global (3-component, global xyz)
        Eigen::Matrix3d nematic_order_global;     // Q_global[bond_type][global_component]
        
        PyrochloreOrderParameters() 
            : scalar_chirality(0.0), 
              vector_chirality(Eigen::Vector3d::Zero()),
              nematic_order(Eigen::Matrix3d::Zero()),
              monopole_density(0.0),
              monopole_by_sublattice(Eigen::Vector4d::Zero()),
              scalar_chirality_global(0.0),
              vector_chirality_global(Eigen::Vector3d::Zero()),
              nematic_order_global(Eigen::Matrix3d::Zero()) {}
    };
    
    /**
     * Compute ALL pyrochlore order parameters in a single pass
     * This is the optimized version that avoids redundant loops over cells.
     * 
     * Combines:
     * - Scalar chirality (kagome triangles)
     * - Vector chirality (kagome triangles)  
     * - Monopole density (tetrahedra)
     * - Signed monopole density (tetrahedra)
     * - Monopole density by sublattice (tetrahedra)
     * 
     * Note: Nematic order still uses bilinear_partners for bond enumeration
     * and is computed separately (different loop structure).
     * 
     * @return PyrochloreOrderParameters struct with all computed values
     */
    PyrochloreOrderParameters compute_pyrochlore_order_parameters_fast() const {
        PyrochloreOrderParameters result;
        
        // Only valid for pyrochlore lattices
        if (!is_pyrochlore()) {
            std::cerr << "Warning: compute_pyrochlore_order_parameters_fast() is only valid for pyrochlore lattices" << std::endl;
            return result;
        }
        if (N_atoms < 4 || spin_dim < 3) {
            return result;
        }
        
        const size_t n_cells = dim1 * dim2 * dim3;
        const double inv_n_cells = 1.0 / double(n_cells);
        
        // Accumulators — local frame
        double chi_sum = 0.0;                              // Scalar chirality
        Eigen::Vector3d kappa_sum = Eigen::Vector3d::Zero();  // Vector chirality
        double Q_sum = 0.0;                                // Q monopole density (signed)
        Eigen::Vector4d monopole_sub = Eigen::Vector4d::Zero();  // By sublattice

        // Accumulators — global frame
        double chi_sum_g = 0.0;
        Eigen::Vector3d kappa_sum_g = Eigen::Vector3d::Zero();
        
        // Single pass over all unit cells
        for (size_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
            // Extract i, j, k from cell_idx
            size_t k = cell_idx % dim3;
            size_t j = (cell_idx / dim3) % dim2;
            size_t i = cell_idx / (dim2 * dim3);
            
            // Get all 4 sublattice spins for this cell
            size_t idx0 = flatten_index(i, j, k, 0);
            size_t idx1 = flatten_index(i, j, k, 1);
            size_t idx2 = flatten_index(i, j, k, 2);
            size_t idx3 = flatten_index(i, j, k, 3);
            
            // Cache spin vectors (avoiding repeated VectorXd allocation)
            const double S0z = spins[idx0](2);
            const double S1x = spins[idx1](0), S1y = spins[idx1](1), S1z = spins[idx1](2);
            const double S2x = spins[idx2](0), S2y = spins[idx2](1), S2z = spins[idx2](2);
            const double S3x = spins[idx3](0), S3y = spins[idx3](1), S3z = spins[idx3](2);
            
            // ===== CHIRALITY in LOCAL FRAME (kagome triangle 1,2,3) =====
            // S2 × S3
            double cross_x = S2y * S3z - S2z * S3y;
            double cross_y = S2z * S3x - S2x * S3z;
            double cross_z = S2x * S3y - S2y * S3x;
            
            // Scalar chirality: χ = S1 · (S2 × S3)
            chi_sum += S1x * cross_x + S1y * cross_y + S1z * cross_z;
            
            // Vector chirality: κ = S1 × S2 + S2 × S3 + S3 × S1
            // S1 × S2
            double s1xs2_x = S1y * S2z - S1z * S2y;
            double s1xs2_y = S1z * S2x - S1x * S2z;
            double s1xs2_z = S1x * S2y - S1y * S2x;
            
            // S3 × S1
            double s3xs1_x = S3y * S1z - S3z * S1y;
            double s3xs1_y = S3z * S1x - S3x * S1z;
            double s3xs1_z = S3x * S1y - S3y * S1x;
            
            kappa_sum(0) += s1xs2_x + cross_x + s3xs1_x;  // Note: S2×S3 = cross
            kappa_sum(1) += s1xs2_y + cross_y + s3xs1_y;
            kappa_sum(2) += s1xs2_z + cross_z + s3xs1_z;

            // ===== CHIRALITY in GLOBAL FRAME =====
            // Transform sublattice spins 1,2,3 to global frame
            double G1x, G1y, G1z, G2x, G2y, G2z, G3x, G3y, G3z;
            PyrochloreLocalFrame::to_global(1, S1x, S1y, S1z, G1x, G1y, G1z);
            PyrochloreLocalFrame::to_global(2, S2x, S2y, S2z, G2x, G2y, G2z);
            PyrochloreLocalFrame::to_global(3, S3x, S3y, S3z, G3x, G3y, G3z);

            // G2 × G3
            double gcross_x = G2y * G3z - G2z * G3y;
            double gcross_y = G2z * G3x - G2x * G3z;
            double gcross_z = G2x * G3y - G2y * G3x;

            // Scalar chirality: χ_g = G1 · (G2 × G3)
            chi_sum_g += G1x * gcross_x + G1y * gcross_y + G1z * gcross_z;

            // Vector chirality: κ_g = G1×G2 + G2×G3 + G3×G1
            double g1xg2_x = G1y * G2z - G1z * G2y;
            double g1xg2_y = G1z * G2x - G1x * G2z;
            double g1xg2_z = G1x * G2y - G1y * G2x;

            double g3xg1_x = G3y * G1z - G3z * G1y;
            double g3xg1_y = G3z * G1x - G3x * G1z;
            double g3xg1_z = G3x * G1y - G3y * G1x;

            kappa_sum_g(0) += g1xg2_x + gcross_x + g3xg1_x;
            kappa_sum_g(1) += g1xg2_y + gcross_y + g3xg1_y;
            kappa_sum_g(2) += g1xg2_z + gcross_z + g3xg1_z;
            
            // ===== MONOPOLE (tetrahedron 0,1,2,3) =====
            // Q = Σ_μ S^z_μ (signed monopole charge)
            double Q_tet = S0z + S1z + S2z + S3z;
            Q_sum += Q_tet;
            
            // Monopole by sublattice (3-1 split only)
            std::array<double, 4> Sz = {S0z, S1z, S2z, S3z};
            int n_pos = 0, n_neg = 0;
            for (size_t mu = 0; mu < 4; ++mu) {
                if (Sz[mu] > 0) n_pos++;
                else n_neg++;
            }
            
            if (n_pos == 3 && n_neg == 1) {
                // Find the minority (negative) sublattice
                for (size_t mu = 0; mu < 4; ++mu) {
                    if (Sz[mu] <= 0) {
                        monopole_sub(mu) += 1.0;
                        break;
                    }
                }
            } else if (n_pos == 1 && n_neg == 3) {
                // Find the minority (positive) sublattice
                for (size_t mu = 0; mu < 4; ++mu) {
                    if (Sz[mu] > 0) {
                        monopole_sub(mu) += 1.0;
                        break;
                    }
                }
            }
        }
        
        // Normalize
        result.scalar_chirality = chi_sum * inv_n_cells;
        result.vector_chirality = kappa_sum * inv_n_cells;
        result.monopole_density = Q_sum * inv_n_cells;
        result.monopole_by_sublattice = monopole_sub * inv_n_cells;

        result.scalar_chirality_global = chi_sum_g * inv_n_cells;
        result.vector_chirality_global = kappa_sum_g * inv_n_cells;
        
        // Nematic order requires bond enumeration - compute separately
        result.nematic_order = compute_kagome_nematic_order();
        result.nematic_order_global = compute_kagome_nematic_order_global();
        
        return result;
    }

    /**
     * Compute scalar chirality on kagome triangles
     * χ = S1 · (S2 × S3) per intra-cell triangle
     * 
     * Triangle vertices: 1(i,j,k), 2(i,j,k), 3(i,j,k)
     * Edges: (1-2) Jy, (2-3) Jz, (3-1) Jx — all intra-cell
     * 
     * @return Average scalar chirality per triangle (1 triangle per unit cell)
     */
    double compute_kagome_scalar_chirality() const {
        return compute_pyrochlore_order_parameters_fast().scalar_chirality;
    }
    
    /**
     * Compute vector chirality on kagome triangles
     * κ = S1 × S2 + S2 × S3 + S3 × S1 per intra-cell triangle
     * 
     * Triangle vertices: 1(i,j,k), 2(i,j,k), 3(i,j,k)
     * 
     * @return Average vector chirality (3-component) per triangle
     */
    Eigen::Vector3d compute_kagome_vector_chirality() const {
        return compute_pyrochlore_order_parameters_fast().vector_chirality;
    }
    
    /**
     * Compute component-resolved nematic bond order on kagome NN bonds
     * 
     * Uses bilinear_partners to enumerate all NN bonds automatically.
     * Bond types for kagome sublattices (1,2,3):
     *   Type 0: (1-2) bonds — Jy type
     *   Type 1: (2-3) bonds — Jz type  
     *   Type 2: (1-3) bonds — Jx type
     * 
     * Returns 3×3 matrix: Q[bond_type][local_component]
     * - Rows: bond types (0, 1, 2)
     * - Cols: local spin components (x=0, y=1, z=2)
     * 
     * Q^α_μ = <S_i^α S_j^α> averaged over all bonds of type μ
     */
    Eigen::Matrix3d compute_kagome_nematic_order() const {
        // Only valid for pyrochlore lattices
        if (!is_pyrochlore()) {
            std::cerr << "Warning: compute_kagome_nematic_order() is only valid for pyrochlore lattices" << std::endl;
            return Eigen::Matrix3d::Zero();
        }
        if (N_atoms < 4 || spin_dim < 3) {
            return Eigen::Matrix3d::Zero();
        }
        
        Eigen::Matrix3d Q_sum = Eigen::Matrix3d::Zero();
        Eigen::Vector3i bond_counts = Eigen::Vector3i::Zero();  // Count bonds per type
        
        // Loop over all kagome sites (sublattices 1, 2, 3)
        for (size_t site_i = 0; site_i < lattice_size; ++site_i) {
            size_t sub_i = site_i % N_atoms;
            if (sub_i == 0) continue;  // Skip apex (sublattice 0)
            
            // Loop over NN partners from bilinear_partners
            for (size_t partner_idx = 0; partner_idx < bilinear_partners[site_i].size(); ++partner_idx) {
                size_t site_j = bilinear_partners[site_i][partner_idx];
                size_t sub_j = site_j % N_atoms;
                
                if (sub_j == 0) continue;  // Skip apex bonds
                if (site_j <= site_i) continue;  // Avoid double counting (only count i < j)
                
                // Determine bond type from sublattice pair
                int bond_type = -1;
                if ((sub_i == 1 && sub_j == 2) || (sub_i == 2 && sub_j == 1)) {
                    bond_type = 0;  // (1-2) bond
                } else if ((sub_i == 2 && sub_j == 3) || (sub_i == 3 && sub_j == 2)) {
                    bond_type = 1;  // (2-3) bond
                } else if ((sub_i == 1 && sub_j == 3) || (sub_i == 3 && sub_j == 1)) {
                    bond_type = 2;  // (1-3) bond
                }
                
                if (bond_type >= 0) {
                    // Accumulate S_i^α * S_j^α for each component
                    for (int alpha = 0; alpha < 3; ++alpha) {
                        Q_sum(bond_type, alpha) += spins[site_i](alpha) * spins[site_j](alpha);
                    }
                    bond_counts(bond_type)++;
                }
            }
        }
        
        // Normalize by number of bonds per type
        for (int bond_type = 0; bond_type < 3; ++bond_type) {
            if (bond_counts(bond_type) > 0) {
                Q_sum.row(bond_type) /= bond_counts(bond_type);
            }
        }
        
        return Q_sum;
    }

    /**
     * Compute component-resolved nematic bond order on kagome NN bonds
     * in the GLOBAL Cartesian frame.
     * 
     * Each local-frame spin is transformed to the global frame using the
     * pyrochlore local frame (x_hat, y_hat, z_hat per sublattice) before
     * computing the product S_i^α_global * S_j^α_global.
     * 
     * Returns 3×3 matrix: Q_global[bond_type][global_component]
     * - Rows: bond types (0=1-2, 1=2-3, 2=1-3)
     * - Cols: global Cartesian components (X=0, Y=1, Z=2)
     */
    Eigen::Matrix3d compute_kagome_nematic_order_global() const {
        if (!is_pyrochlore()) {
            std::cerr << "Warning: compute_kagome_nematic_order_global() is only valid for pyrochlore lattices" << std::endl;
            return Eigen::Matrix3d::Zero();
        }
        if (N_atoms < 4 || spin_dim < 3) {
            return Eigen::Matrix3d::Zero();
        }
        
        Eigen::Matrix3d Q_sum = Eigen::Matrix3d::Zero();
        Eigen::Vector3i bond_counts = Eigen::Vector3i::Zero();
        
        for (size_t site_i = 0; site_i < lattice_size; ++site_i) {
            size_t sub_i = site_i % N_atoms;
            if (sub_i == 0) continue;
            
            for (size_t partner_idx = 0; partner_idx < bilinear_partners[site_i].size(); ++partner_idx) {
                size_t site_j = bilinear_partners[site_i][partner_idx];
                size_t sub_j = site_j % N_atoms;
                
                if (sub_j == 0) continue;
                if (site_j <= site_i) continue;
                
                int bond_type = -1;
                if ((sub_i == 1 && sub_j == 2) || (sub_i == 2 && sub_j == 1)) {
                    bond_type = 0;
                } else if ((sub_i == 2 && sub_j == 3) || (sub_i == 3 && sub_j == 2)) {
                    bond_type = 1;
                } else if ((sub_i == 1 && sub_j == 3) || (sub_i == 3 && sub_j == 1)) {
                    bond_type = 2;
                }
                
                if (bond_type >= 0) {
                    // Transform both spins to global frame
                    double Gi_x, Gi_y, Gi_z, Gj_x, Gj_y, Gj_z;
                    PyrochloreLocalFrame::to_global(static_cast<int>(sub_i),
                        spins[site_i](0), spins[site_i](1), spins[site_i](2),
                        Gi_x, Gi_y, Gi_z);
                    PyrochloreLocalFrame::to_global(static_cast<int>(sub_j),
                        spins[site_j](0), spins[site_j](1), spins[site_j](2),
                        Gj_x, Gj_y, Gj_z);
                    
                    // Accumulate Si_global^α * Sj_global^α
                    Q_sum(bond_type, 0) += Gi_x * Gj_x;
                    Q_sum(bond_type, 1) += Gi_y * Gj_y;
                    Q_sum(bond_type, 2) += Gi_z * Gj_z;
                    bond_counts(bond_type)++;
                }
            }
        }
        
        for (int bond_type = 0; bond_type < 3; ++bond_type) {
            if (bond_counts(bond_type) > 0) {
                Q_sum.row(bond_type) /= bond_counts(bond_type);
            }
        }
        
        return Q_sum;
    }
    
    /**
     * Compute monopole density decomposed by sublattice type
     * 
     * For a 3-in-1-out monopole, the "type" is determined by which sublattice μ
     * has the minority spin (the 1-out). Similarly for 1-in-3-out.
     * 
     * Returns a 4-component vector:
     *   density[μ] = fraction of tetrahedra where sublattice μ is the minority
     * 
     * For ice-rule states (2-in-2-out) or double monopoles (4-in or 4-out),
     * no sublattice is counted as minority.
     * 
     * @return Eigen::Vector4d with monopole density per sublattice type
     */
    Eigen::Vector4d compute_monopole_density_by_sublattice() const {
        return compute_pyrochlore_order_parameters_fast().monopole_by_sublattice;
    }
    
    /**
     * Compute all kagome order parameters at once
     * Uses the fast single-pass implementation internally.
     * 
     * @return Tuple of (scalar_chirality, vector_chirality, nematic_order_matrix)
     */
    std::tuple<double, Eigen::Vector3d, Eigen::Matrix3d> compute_kagome_order_parameters() const {
        auto params = compute_pyrochlore_order_parameters_fast();
        return {params.scalar_chirality, params.vector_chirality, params.nematic_order};
    }
    
    /**
     * Compute all monopole diagnostics at once
     * Uses the fast single-pass implementation internally.
     * 
     * @return Tuple of (monopole_density (signed), monopole_by_sublattice)
     */
    std::tuple<double, Eigen::Vector4d> compute_monopole_diagnostics() const {
        auto params = compute_pyrochlore_order_parameters_fast();
        return {params.monopole_density, params.monopole_by_sublattice};
    }

    // ============================================================
    // COMPREHENSIVE OBSERVABLE COLLECTION
    // ============================================================

    /**
     * Collect a single measurement of all thermodynamic observables
     * Returns: (energy, sublattice_magnetizations)
     */
    std::pair<double, vector<SpinVector>> measure_observables() const {
        double E = total_energy(spins);
        vector<SpinVector> M_sub = magnetization_sublattice();
        return {E, M_sub};
    }

    /**
     * Compute comprehensive thermodynamic observables with binning error analysis
     * 
     * @param energies Vector of energy measurements
     * @param sublattice_mags Vector of sublattice magnetization measurements
     *                        Each element is a vector of N_atoms SpinVectors
     * @param T Temperature
     * @return ThermodynamicObservables struct with all observables and uncertainties
     */
    ThermodynamicObservables compute_thermodynamic_observables(
        const vector<double>& energies,
        const vector<vector<SpinVector>>& sublattice_mags,
        double T) const;

    /**
     * Save comprehensive thermodynamic observables to files
     */
    void save_thermodynamic_observables(const string& out_dir,
                                         const ThermodynamicObservables& obs) const;

    /**
     * Print thermodynamic observables summary to stdout
     */
    void print_thermodynamic_observables(const ThermodynamicObservables& obs) const {
        cout << "\n=== Thermodynamic Observables at T = " << obs.temperature << " ===" << endl;
        cout << std::scientific << std::setprecision(6);
        
        cout << "<E>/N = " << obs.energy.value << " ± " << obs.energy.error << endl;
        cout << "C_V   = " << obs.specific_heat.value << " ± " << obs.specific_heat.error << endl;
        
        cout << "\nTotal magnetization <M>/N:" << endl;
        for (size_t d = 0; d < obs.magnetization.values.size(); ++d) {
            cout << "  M_" << d << " = " << obs.magnetization.values[d] 
                 << " ± " << obs.magnetization.errors[d] << endl;
        }
        
        cout << "\nSublattice magnetizations:" << endl;
        for (size_t alpha = 0; alpha < obs.sublattice_magnetization.size(); ++alpha) {
            cout << "  Sublattice " << alpha << ": (";
            const auto& M = obs.sublattice_magnetization[alpha];
            for (size_t d = 0; d < M.values.size(); ++d) {
                if (d > 0) cout << ", ";
                cout << M.values[d] << "±" << M.errors[d];
            }
            cout << ")" << endl;
        }
        
        cout << "\nEnergy-sublattice cross correlations:" << endl;
        for (size_t alpha = 0; alpha < obs.energy_sublattice_cross.size(); ++alpha) {
            cout << "  Sublattice " << alpha << ": (";
            const auto& cross = obs.energy_sublattice_cross[alpha];
            for (size_t d = 0; d < cross.values.size(); ++d) {
                if (d > 0) cout << ", ";
                cout << cross.values[d] << "±" << cross.errors[d];
            }
            cout << ")" << endl;
        }
    }

    /**
     * Save thermodynamic observables to HDF5 format
     * Single file per rank with all data organized in groups
     */
    void save_thermodynamic_observables_hdf5(const string& out_dir,
                                              const ThermodynamicObservables& obs,
                                              const vector<double>& energies,
                                              const vector<SpinVector>& magnetizations,
                                              const vector<vector<SpinVector>>& sublattice_mags,
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

    // ============================================================
    // MONTE CARLO METHODS
    // ============================================================

    /**
     * Metropolis sweep with local spin updates. Returns acceptance rate.
     *
     * Implementation notes (post-audit refactor):
     *  - **Sequential** site order, not random-with-replacement.  Sequential
     *    sweeps are cache-friendly (spins, partners and J-matrices stream in
     *    order), still ergodic, and standard practice in modern MC packages
     *    (ALPS/looper, ALF, SpinW). This changes the exact MC trajectory
     *    but not equilibrium averages.
     *  - **Zero heap allocations** per proposed move: the new spin is built
     *    in a stack buffer, energy diff is computed via `site_energy_diff_flat`
     *    which uses raw `double*` access into `spins[site].data()`, and on
     *    accept we `memcpy` into the existing Eigen storage.
     *  - **Logical OR** acceptance (was bitwise `|`), with `dE <= 0` short
     *    circuit to skip exp() entirely on downhill moves.
     */
    double metropolis(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (T <= 0) return 0.0;

        const double beta = 1.0 / T;
        size_t accepted = 0;

        constexpr size_t MAX_SPIN_DIM = 16;
        alignas(32) double new_spin_buf[MAX_SPIN_DIM];

        // Batch random uniforms: amortises the lehman_next() function-call
        // overhead and keeps the inner loop tight.
        constexpr size_t BATCH_SIZE = 64;
        double rand_uniforms[BATCH_SIZE];

        for (size_t batch_start = 0; batch_start < lattice_size; batch_start += BATCH_SIZE) {
            const size_t batch_end = std::min(batch_start + BATCH_SIZE, lattice_size);
            const size_t n_in_batch = batch_end - batch_start;

            for (size_t j = 0; j < n_in_batch; ++j) {
                rand_uniforms[j] = random_double_lehman(0.0, 1.0);
            }

            for (size_t j = 0; j < n_in_batch; ++j) {
                const size_t site      = batch_start + j;
                const double rand_uni  = rand_uniforms[j];
                double*      old_spin  = spins[site].data();

                // Build the proposed spin in `new_spin_buf` with no
                // allocation. For Gaussian moves we add a length-spin_length
                // offset and renormalise (rejection if it underflows).
                if (gaussian_move) {
                    gen_random_spin_into(new_spin_buf, spin_length);
                    double sum_sq = 0.0;
                    for (size_t d = 0; d < spin_dim; ++d) {
                        new_spin_buf[d] = old_spin[d] + sigma * new_spin_buf[d];
                        sum_sq += new_spin_buf[d] * new_spin_buf[d];
                    }
                    if (sum_sq < 1e-20) continue;  // pathological: skip
                    const double inv_norm = double(spin_length) / std::sqrt(sum_sq);
                    for (size_t d = 0; d < spin_dim; ++d) new_spin_buf[d] *= inv_norm;
                } else {
                    gen_random_spin_into(new_spin_buf, spin_length);
                }

                const double dE = site_energy_diff_flat(new_spin_buf, old_spin, site);

                // Logical OR + downhill short-circuit (was bitwise `|` which
                // forced unnecessary exp() evaluation on downhill moves).
                const bool accept = (dE <= 0.0) ||
                                    (rand_uni < std::exp(-beta * dE));
                if (accept) {
                    std::memcpy(old_spin, new_spin_buf, spin_dim * sizeof(double));
                    ++accepted;
                }
            }
        }

        return double(accepted) / double(lattice_size);
    }

    /**
     * Gaussian move around current spin
     *
     * Backwards-compatible API. Internally allocates a temporary, so prefer
     * `metropolis()` (which has the inlined zero-allocation path) for hot
     * loops. Kept for callers in tests / external utilities.
     */
    SpinVector gaussian_spin_move(const SpinVector& current_spin, double sigma) {
        SpinVector new_spin = current_spin + gen_random_spin(spin_length) * sigma;
        double norm = new_spin.norm();
        if (norm < 1e-10) return current_spin;
        return new_spin * (spin_length / norm);
    }

    /**
     * Over-relaxation sweep (microcanonical, zero acceptance rate).
     * Reflects each spin across its local field direction.
     *
     * Now sequential (was random with replacement → ~37% sites missed per
     * sweep due to coupon-collector). Ergodicity unchanged; thermodynamic
     * averages unaffected.
     */
    void overrelaxation() {
        constexpr size_t MAX_SPIN_DIM = 16;
        double H_buf[MAX_SPIN_DIM];
        for (size_t site = 0; site < lattice_size; ++site) {
            const double* B = field[site].data();
            for (size_t d = 0; d < spin_dim; ++d) H_buf[d] = -B[d];

            const auto& A = onsite_interaction[site];
            const double* S = spins[site].data();
            for (size_t a = 0; a < spin_dim; ++a) {
                double acc = 0.0;
                for (size_t b = 0; b < spin_dim; ++b) acc += A(a, b) * S[b];
                H_buf[a] += 2.0 * acc;
            }

            // Bilinear contribution — SoA hot path, twist BC is cold.
            const size_t bi_base = bi_flat_offset[site];
            const size_t bi_end  = bi_flat_offset[site + 1];
            const size_t D2      = bi_flat_D2;
            for (size_t k = bi_base; k < bi_end; ++k) {
                const size_t partner = bi_flat_partner[k];
                const double* __restrict P = spins[partner].data();
                const double* __restrict J = &bi_flat_J[k * D2];

                if (__builtin_expect(bi_flat_needs_twist[k] != 0, 0)) {
                    double tw[3] = {P[0], P[1], P[2]};
                    const auto& wrap = bi_flat_wrap[k];
                    for (size_t dim = 0; dim < 3; ++dim) {
                        if (wrap[dim] == 0) continue;
                        double tmp[3] = {0, 0, 0};
                        if (wrap[dim] > 0) {
                            for (size_t d = 0; d < 3; ++d)
                                tmp[d] = twist_matrices[dim](d, 0) * tw[0]
                                       + twist_matrices[dim](d, 1) * tw[1]
                                       + twist_matrices[dim](d, 2) * tw[2];
                        } else {
                            for (size_t d = 0; d < 3; ++d)
                                tmp[d] = twist_matrices[dim](0, d) * tw[0]
                                       + twist_matrices[dim](1, d) * tw[1]
                                       + twist_matrices[dim](2, d) * tw[2];
                        }
                        tw[0] = tmp[0]; tw[1] = tmp[1]; tw[2] = tmp[2];
                    }
                    for (size_t a = 0; a < spin_dim; ++a) {
                        double acc = 0.0;
                        for (size_t b = 0; b < spin_dim; ++b)
                            acc += J[a * spin_dim + b] * tw[b];
                        H_buf[a] += acc;
                    }
                } else {
                    for (size_t a = 0; a < spin_dim; ++a) {
                        double acc = 0.0;
                        for (size_t b = 0; b < spin_dim; ++b)
                            acc += J[a * spin_dim + b] * P[b];
                        H_buf[a] += acc;
                    }
                }
            }

            const size_t n_tri = trilinear_partners[site].size();
            for (size_t n = 0; n < n_tri; ++n) {
                const size_t p1 = trilinear_partners[site][n][0];
                const size_t p2 = trilinear_partners[site][n][1];
                const double* S_j = spins[p1].data();
                const double* S_k = spins[p2].data();
                const auto& T = trilinear_interaction[site][n];
                double S_jk[64];
                for (size_t b = 0; b < spin_dim; ++b)
                    for (size_t c = 0; c < spin_dim; ++c)
                        S_jk[b * spin_dim + c] = S_j[b] * S_k[c];
                for (size_t a = 0; a < spin_dim; ++a) {
                    double acc = 0.0;
                    for (size_t b = 0; b < spin_dim; ++b)
                        for (size_t c = 0; c < spin_dim; ++c)
                            acc += T[a](b, c) * S_jk[b * spin_dim + c];
                    H_buf[a] += acc;
                }
            }

            // Reflect: S' = 2 (S·H) H / |H|^2 - S
            double norm_sq = 0.0;
            for (size_t d = 0; d < spin_dim; ++d) norm_sq += H_buf[d] * H_buf[d];
            if (norm_sq <= 0.0) continue;

            double S_dot_H = 0.0;
            double* Sw = spins[site].data();
            for (size_t d = 0; d < spin_dim; ++d) S_dot_H += Sw[d] * H_buf[d];
            const double k = 2.0 * S_dot_H / norm_sq;
            for (size_t d = 0; d < spin_dim; ++d) Sw[d] = k * H_buf[d] - Sw[d];
        }
    }

    /**
     * Wolff cluster update - single cluster flip. Returns cluster size.
     *
     * Performance refactor (post-audit):
     *  - Uses **persistent member buffers** (cluster_proj_buf,
     *    cluster_in_cluster, cluster_stack_buf) instead of allocating
     *    vector<double>(N) / vector<uint8_t>(N) on every call. For a
     *    27 648-site pyrochlore at ~10⁵ Wolff updates this saves several GB
     *    of cumulative malloc traffic and the corresponding TLB/page-fault
     *    overhead.
     *  - Inlined J·r·r as a 3×3×3 unrolled dot for spin_dim==3 to dodge
     *    Eigen's dynamic matrix-vector dispatch on every neighbour visit.
     *  - Spin reflection is in-place via raw double pointers.
     */
    size_t wolff_update(double T, bool use_ghost_field = false) {
        if (T <= 0) return 0;

        const double beta = 1.0 / T;

        const size_t seed = random_int_lehman(lattice_size);
        SpinVector r = random_unit_vector();
        const double* r_data = r.data();

        // Reuse persistent buffers (allocate-on-grow semantics).
        if (cluster_proj_buf.size()   < lattice_size) cluster_proj_buf.resize(lattice_size);
        if (cluster_in_cluster.size() < lattice_size) cluster_in_cluster.assign(lattice_size, 0);
        else std::fill(cluster_in_cluster.begin(), cluster_in_cluster.begin() + lattice_size, 0);
        cluster_stack_buf.clear();
        if (cluster_stack_buf.capacity() < lattice_size / 8 + 16)
            cluster_stack_buf.reserve(lattice_size / 8 + 16);

        // Precompute projections S_i · r (sequential for cache locality).
        for (size_t i = 0; i < lattice_size; ++i) {
            const double* S = spins[i].data();
            double acc = 0.0;
            for (size_t d = 0; d < spin_dim; ++d) acc += S[d] * r_data[d];
            cluster_proj_buf[i] = acc;
        }

        bool attached_to_ghost = false;
        cluster_in_cluster[seed] = 1;
        cluster_stack_buf.push_back(seed);

        while (!cluster_stack_buf.empty()) {
            const size_t i = cluster_stack_buf.back();
            cluster_stack_buf.pop_back();

            const size_t n_bi = bilinear_partners[i].size();
            for (size_t n = 0; n < n_bi; ++n) {
                const size_t j = bilinear_partners[i][n];
                if (cluster_in_cluster[j]) continue;

                // r · J · r  (scalar coupling along the reflection plane normal)
                const auto& J = bilinear_interaction[i][n];
                double K_r = 0.0;
                for (size_t a = 0; a < spin_dim; ++a) {
                    double row = 0.0;
                    for (size_t b = 0; b < spin_dim; ++b) row += J(a, b) * r_data[b];
                    K_r += r_data[a] * row;
                }
                if (K_r <= 0.0) continue;

                // Partner projection along r, including twist BC.
                double proj_j;
                const auto& wrap = bilinear_wrap_dir[i][n];
                if (spin_dim == 3 && (wrap[0] != 0 || wrap[1] != 0 || wrap[2] != 0)) {
                    double tw[3] = { spins[j].data()[0], spins[j].data()[1], spins[j].data()[2] };
                    for (size_t dim = 0; dim < 3; ++dim) {
                        if (wrap[dim] == 0) continue;
                        double tmp[3] = {0,0,0};
                        if (wrap[dim] > 0) {
                            for (size_t d = 0; d < 3; ++d)
                                tmp[d] = twist_matrices[dim](d,0)*tw[0]
                                       + twist_matrices[dim](d,1)*tw[1]
                                       + twist_matrices[dim](d,2)*tw[2];
                        } else {
                            for (size_t d = 0; d < 3; ++d)
                                tmp[d] = twist_matrices[dim](0,d)*tw[0]
                                       + twist_matrices[dim](1,d)*tw[1]
                                       + twist_matrices[dim](2,d)*tw[2];
                        }
                        tw[0]=tmp[0]; tw[1]=tmp[1]; tw[2]=tmp[2];
                    }
                    proj_j = tw[0]*r_data[0] + tw[1]*r_data[1] + tw[2]*r_data[2];
                } else {
                    proj_j = cluster_proj_buf[j];
                }

                const double prod = cluster_proj_buf[i] * proj_j;
                if (prod <= 0.0) continue;
                const double P_add = 1.0 - std::exp(-2.0 * beta * K_r * prod);
                if (random_double_lehman(0.0, 1.0) < P_add) {
                    cluster_in_cluster[j] = 1;
                    cluster_stack_buf.push_back(j);
                }
            }

            if (use_ghost_field) {
                // Use field·field via raw pointer to skip Eigen .norm() temp.
                const double* B = field[i].data();
                double Bnorm_sq = 0.0;
                for (size_t d = 0; d < spin_dim; ++d) Bnorm_sq += B[d] * B[d];
                if (Bnorm_sq > 1e-20) {
                    double K_field = 0.0;
                    for (size_t d = 0; d < spin_dim; ++d) K_field += r_data[d] * B[d];
                    const double prod = K_field * cluster_proj_buf[i];
                    if (prod > 0.0) {
                        const double P_ghost = 1.0 - std::exp(-beta * std::abs(prod));
                        if (random_double_lehman(0.0, 1.0) < P_ghost) {
                            attached_to_ghost = true;
                        }
                    }
                }
            }
        }

        size_t cluster_size = 0;
        if (!attached_to_ghost) {
            for (size_t i = 0; i < lattice_size; ++i) {
                if (cluster_in_cluster[i]) {
                    double* S = spins[i].data();
                    const double two_proj = 2.0 * cluster_proj_buf[i];
                    for (size_t d = 0; d < spin_dim; ++d) S[d] -= two_proj * r_data[d];
                    ++cluster_size;
                }
            }
        }

        return cluster_size;
    }

    /**
     * Generate random unit vector
     */
    SpinVector random_unit_vector() const {
        SpinVector v = const_cast<Lattice*>(this)->gen_random_spin(1.0);
        double norm = v.norm();
        if (norm < 1e-10) {
            v = SpinVector::Zero(spin_dim);
            v(0) = 1.0;
            return v;
        }
        return v / norm;
    }

    /**
     * Swendsen-Wang sweep - build and flip all clusters.
     * Returns number of clusters flipped.
     *
     * Performance refactor (post-audit):
     *  - Persistent Union-Find buffers (uf_parent, uf_size, uf_forbid_flip,
     *    uf_flip_root) reused across calls — was four `vector<...>(N)` per
     *    sweep.
     *  - **Iterative** path-compression `find()` (was a recursive
     *    `std::function<int(int)>` lambda → up to N stack frames + virtual
     *    indirection per call; clang/gcc can't inline through std::function).
     *    Two-pass iterative compression is the textbook implementation.
     *  - Inlined r·J·r and r·field via raw pointers.
     *  - In-place spin reflection (no Eigen temporaries).
     */
    size_t swendsen_wang_sweep(double T, bool use_ghost_field = false) {
        if (T <= 0) return 0;

        const double beta = 1.0 / T;
        SpinVector r = random_unit_vector();
        const double* r_data = r.data();

        // Persistent buffers
        if (cluster_proj_buf.size() < lattice_size) cluster_proj_buf.resize(lattice_size);
        if (uf_parent.size()        < lattice_size) uf_parent.resize(lattice_size);
        if (uf_size.size()          < lattice_size) uf_size.resize(lattice_size);
        if (uf_forbid_flip.size()   < lattice_size) uf_forbid_flip.assign(lattice_size, 0);
        else std::fill(uf_forbid_flip.begin(), uf_forbid_flip.begin() + lattice_size, 0);
        if (uf_flip_root.size()     < lattice_size) uf_flip_root.assign(lattice_size, 0);
        else std::fill(uf_flip_root.begin(), uf_flip_root.begin() + lattice_size, 0);

        std::iota(uf_parent.begin(), uf_parent.begin() + lattice_size, 0);
        std::fill(uf_size.begin(), uf_size.begin() + lattice_size, 1);

        // Precompute projections
        for (size_t i = 0; i < lattice_size; ++i) {
            const double* S = spins[i].data();
            double acc = 0.0;
            for (size_t d = 0; d < spin_dim; ++d) acc += S[d] * r_data[d];
            cluster_proj_buf[i] = acc;
        }

        // Iterative path-compression find (two-pass).
        auto find_root = [&](int x) noexcept -> int {
            int root = x;
            while (uf_parent[root] != root) root = uf_parent[root];
            // Path compression
            while (uf_parent[x] != root) {
                int next = uf_parent[x];
                uf_parent[x] = root;
                x = next;
            }
            return root;
        };

        auto unite = [&](int a, int b) noexcept {
            a = find_root(a);
            b = find_root(b);
            if (a == b) return;
            if (uf_size[a] < uf_size[b]) std::swap(a, b);
            uf_parent[b] = a;
            uf_size[a]  += uf_size[b];
        };

        // Build clusters via bond percolation.
        for (size_t i = 0; i < lattice_size; ++i) {
            const size_t n_bi = bilinear_partners[i].size();
            for (size_t n = 0; n < n_bi; ++n) {
                const size_t j = bilinear_partners[i][n];
                if (j <= i) continue;  // each bond once

                const auto& J = bilinear_interaction[i][n];
                double K_r = 0.0;
                for (size_t a = 0; a < spin_dim; ++a) {
                    double row = 0.0;
                    for (size_t b = 0; b < spin_dim; ++b) row += J(a, b) * r_data[b];
                    K_r += r_data[a] * row;
                }
                if (K_r <= 0.0) continue;

                double proj_j;
                const auto& wrap = bilinear_wrap_dir[i][n];
                if (spin_dim == 3 && (wrap[0] != 0 || wrap[1] != 0 || wrap[2] != 0)) {
                    double tw[3] = { spins[j].data()[0], spins[j].data()[1], spins[j].data()[2] };
                    for (size_t dim = 0; dim < 3; ++dim) {
                        if (wrap[dim] == 0) continue;
                        double tmp[3] = {0,0,0};
                        if (wrap[dim] > 0) {
                            for (size_t d = 0; d < 3; ++d)
                                tmp[d] = twist_matrices[dim](d,0)*tw[0]
                                       + twist_matrices[dim](d,1)*tw[1]
                                       + twist_matrices[dim](d,2)*tw[2];
                        } else {
                            for (size_t d = 0; d < 3; ++d)
                                tmp[d] = twist_matrices[dim](0,d)*tw[0]
                                       + twist_matrices[dim](1,d)*tw[1]
                                       + twist_matrices[dim](2,d)*tw[2];
                        }
                        tw[0]=tmp[0]; tw[1]=tmp[1]; tw[2]=tmp[2];
                    }
                    proj_j = tw[0]*r_data[0] + tw[1]*r_data[1] + tw[2]*r_data[2];
                } else {
                    proj_j = cluster_proj_buf[j];
                }

                const double prod = cluster_proj_buf[i] * proj_j;
                if (prod <= 0.0) continue;
                const double P_bond = 1.0 - std::exp(-2.0 * beta * K_r * prod);
                if (random_double_lehman(0.0, 1.0) < P_bond) {
                    unite(static_cast<int>(i), static_cast<int>(j));
                }
            }
        }

        // Ghost bonds (prevent flipping clusters with strong field overlap).
        if (use_ghost_field) {
            for (size_t i = 0; i < lattice_size; ++i) {
                const double* B = field[i].data();
                double Bnorm_sq = 0.0;
                for (size_t d = 0; d < spin_dim; ++d) Bnorm_sq += B[d] * B[d];
                if (Bnorm_sq <= 1e-20) continue;
                double K_field = 0.0;
                for (size_t d = 0; d < spin_dim; ++d) K_field += r_data[d] * B[d];
                const double prod = K_field * cluster_proj_buf[i];
                if (prod <= 0.0) continue;
                const double P_ghost = 1.0 - std::exp(-beta * std::abs(prod));
                if (random_double_lehman(0.0, 1.0) < P_ghost) {
                    uf_forbid_flip[find_root(static_cast<int>(i))] = 1;
                }
            }
        }

        // Decide per-root flip with probability 1/2.
        for (size_t i = 0; i < lattice_size; ++i) {
            const int root = find_root(static_cast<int>(i));
            if (static_cast<int>(i) == root && !uf_forbid_flip[root]) {
                uf_flip_root[root] = (random_double_lehman(0.0, 1.0) < 0.5) ? 1 : 0;
            }
        }

        // Apply flips and count flipped roots.
        size_t flipped_clusters = 0;
        for (size_t i = 0; i < lattice_size; ++i) {
            const int root = find_root(static_cast<int>(i));
            if (uf_flip_root[root]) {
                double* S = spins[i].data();
                const double two_proj = 2.0 * cluster_proj_buf[i];
                for (size_t d = 0; d < spin_dim; ++d) S[d] -= two_proj * r_data[d];
                if (static_cast<int>(i) == root) ++flipped_clusters;
            }
        }

        return flipped_clusters;
    }

    /**
     * Convenience: multiple Wolff updates
     */
    size_t wolff_sweep(double T, size_t k = 1, bool use_ghost_field = false) {
        size_t total = 0;
        for (size_t c = 0; c < k; ++c) {
            total += wolff_update(T, use_ghost_field);
        }
        return total;
    }

    /**
     * Deterministic sweep: align each spin antiparallel to its local field
     * This is a zero-temperature relaxation step that randomly selects sites
     */
    void deterministic_sweep(size_t num_sweeps) {
        for (size_t sweep = 0; sweep < num_sweeps; ++sweep) {
            size_t count = 0;
            while (count < lattice_size) {
                size_t i = random_int_lehman(lattice_size);
                SpinVector local_field = get_local_field(i);
                double norm = local_field.norm();
                
                if (norm < 1e-15) {
                    continue;
                } else {
                    spins[i] = -local_field / norm * spin_length;
                }
                count++;
            }
        }
    }

    /**
     * Zero-temperature greedy quench with convergence check
     */
    void greedy_quench(double rel_tol = 1e-12, size_t max_sweeps = 10000) {
        double E_prev = total_energy(spins);
        
        for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
            deterministic_sweep(1);
            
            // Check convergence
            double E_curr = total_energy(spins);
            if (std::abs(E_curr - E_prev) <= rel_tol * (std::abs(E_prev) + 1e-18)) {
                break;
            }
            E_prev = E_curr;
        }
    }

    /**
     * Metropolis update for twist boundary matrices
     * Returns acceptance count (number of accepted moves)
     * 
     * Optimizes the rotation angle around a fixed z-axis.
     * Uses hybrid strategy:
     *   - At high T: larger steps + occasional global moves for exploration
     *   - At low T: small incremental steps for fine-tuning
     *   - Occasional random global moves (10% probability) to escape local minima
     */
    size_t metropolis_twist_sweep(double T);
    
    /**
     * Extract axis-angle representation from rotation matrix
     * Uses the formula: angle = arccos((trace(R) - 1) / 2)
     * Axis is extracted from the antisymmetric part of R
     */
    static void extract_axis_angle_from_rotation(const SpinMatrix& R, SpinVector& axis, double& angle) {
        if (R.rows() != 3 || R.cols() != 3) {
            axis = SpinVector::Zero(R.rows());
            if (axis.size() >= 3) axis(2) = 1.0;
            angle = 0.0;
            return;
        }
        
        double trace = R(0, 0) + R(1, 1) + R(2, 2);
        double cos_angle = (trace - 1.0) / 2.0;
        cos_angle = std::clamp(cos_angle, -1.0, 1.0);
        angle = std::acos(cos_angle);
        
        // Handle special cases
        if (std::abs(angle) < 1e-10) {
            // Identity rotation
            axis = SpinVector::Zero(3);
            axis(2) = 1.0;
            angle = 0.0;
            return;
        }
        
        if (std::abs(angle - M_PI) < 1e-10) {
            // 180 degree rotation - extract axis from diagonal
            axis = SpinVector::Zero(3);
            axis(0) = std::sqrt(std::max(0.0, (R(0, 0) + 1.0) / 2.0));
            axis(1) = std::sqrt(std::max(0.0, (R(1, 1) + 1.0) / 2.0));
            axis(2) = std::sqrt(std::max(0.0, (R(2, 2) + 1.0) / 2.0));
            // Determine signs from off-diagonal elements
            if (R(0, 1) < 0) axis(1) = -axis(1);
            if (R(0, 2) < 0) axis(2) = -axis(2);
            return;
        }
        
        // General case: extract axis from antisymmetric part
        double sin_angle = std::sin(angle);
        axis = SpinVector::Zero(3);
        axis(0) = (R(2, 1) - R(1, 2)) / (2.0 * sin_angle);
        axis(1) = (R(0, 2) - R(2, 0)) / (2.0 * sin_angle);
        axis(2) = (R(1, 0) - R(0, 1)) / (2.0 * sin_angle);
        
        // Normalize (should already be unit, but ensure numerical stability)
        double norm = axis.norm();
        if (norm > 1e-10) {
            axis /= norm;
        } else {
            axis(2) = 1.0;
        }
    }

    
    // ============================================================
    // SIMULATED ANNEALING AUTO-TUNING
    // ============================================================

    /**
     * Auto-tune simulated annealing parameters
     */
    SAParams tune_simulated_annealing(double Tmin_guess = 0.0,
                                      double Tmax_guess = 0.0,
                                      bool gaussian_move = false,
                                      size_t overrelaxation_rate = 0,
                                      size_t pilot_sweeps = 300,
                                      double acc_hi_target = 0.7,
                                      double acc_lo_target = 0.02) {
        SAParams params;
        
        // Backup current state
        SpinConfig spins_backup = spins;
        
        // Lambda: probe acceptance and autocorrelation at temperature T
        auto probe_once = [&](double T, size_t sweeps, double base_interval,
                             double& acc_out, double& tau_out) {
            double sigma = 1000.0;
            double acc_sum = 0.0;
            vector<double> energies;
            energies.reserve(sweeps / size_t(base_interval) + 1);
            
            for (size_t i = 0; i < sweeps; ++i) {
                acc_sum += metropolis(T, gaussian_move, sigma);
                
                if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                    overrelaxation();
                }
                
                if (i % size_t(base_interval) == 0) {
                    energies.push_back(total_energy(spins));
                }
            }
            
            acc_out = acc_sum / double(sweeps);
            
            if (energies.size() > 10) {
                auto acf = compute_autocorrelation(energies, size_t(base_interval));
                tau_out = acf.tau_int;
            } else {
                tau_out = 1.0;
            }
        };
        
        // Calibrate T_start (high acceptance)
        {
            double T = (Tmax_guess > 0.0) ? Tmax_guess : 1.0;
            double acc = 0.0, tau = 1.0;
            
            // Expand up if needed
            for (size_t iter = 0; iter < 25; ++iter) {
                probe_once(T, pilot_sweeps, 10, acc, tau);
                params.probe_T.push_back(T);
                params.probe_acc.push_back(acc);
                params.probe_tau.push_back(tau);
                
                if (acc >= acc_hi_target) break;
                T *= 2.0;
            }
            
            // Binary search to center in target band
            double Thigh = params.probe_T.back();
            double Tlow = Thigh / 100.0;
            for (size_t k = 0; k < 20; ++k) {
                double Tmid = 0.5 * (Tlow + Thigh);
                probe_once(Tmid, pilot_sweeps, 10, acc, tau);
                params.probe_T.push_back(Tmid);
                params.probe_acc.push_back(acc);
                params.probe_tau.push_back(tau);
                
                if (acc > acc_hi_target) {
                    Thigh = Tmid;
                } else {
                    Tlow = Tmid;
                }
            }
            
            params.T_start = Thigh;
        }
        
        // Calibrate T_end (low acceptance, energy converged)
        {
            double T = (Tmin_guess > 0.0 && Tmin_guess < params.T_start) ? 
                       Tmin_guess : params.T_start * 1e-3;
            T = max(T, params.T_start * 1e-6);
            
            double acc = 1.0, tau = 1.0;
            double cur = params.T_start;
            
            for (size_t iter = 0; iter < 40 && cur > T; ++iter) {
                cur *= 0.5;
                probe_once(cur, pilot_sweeps, 10, acc, tau);
                params.probe_T.push_back(cur);
                params.probe_acc.push_back(acc);
                params.probe_tau.push_back(tau);
                
                if (acc < acc_lo_target) break;
            }
            
            params.T_end = max(1e-12, min(cur, params.T_start / 1e3));
        }
        
        // Choose sweeps_per_temp from autocorrelation time
        double tau_max = 1.0;
        for (double t : params.probe_tau) {
            tau_max = std::max(tau_max, t);
        }
        params.sweeps_per_temp = std::max<size_t>(100, size_t(10.0 * tau_max));
        
        // Number of temperature steps
        size_t K = std::max<size_t>(50, size_t(10.0 * std::sqrt(tau_max)));
        K = std::min<size_t>(2000, K);
        params.cooling_rate = std::pow(params.T_end / params.T_start, 1.0 / double(K));
        params.cooling_rate = std::clamp(params.cooling_rate, 0.85, 0.995);
        
        // Restore original spins
        spins = spins_backup;
        
        cout << "Auto-tuned SA parameters:" << endl;
        cout << "  T_start = " << params.T_start << endl;
        cout << "  T_end = " << params.T_end << endl;
        cout << "  cooling_rate = " << params.cooling_rate << endl;
        cout << "  sweeps_per_temp = " << params.sweeps_per_temp << endl;
        
        return params;
    }

    // ============================================================
    // SIMULATED ANNEALING
    // ============================================================

    /**
     * Main simulated annealing routine
     * 
     * @param T_start              Starting temperature
     * @param T_end                Final temperature (cooling stops here)
     * @param n_anneal             Number of MC sweeps per temperature step
     * @param overrelaxation_rate  Overrelaxation frequency (0 = disabled)
     * @param boundary_update      Enable twist boundary condition updates
     * @param gaussian_move        Use Gaussian moves instead of uniform
     * @param cooling_rate         Temperature reduction factor (default: 0.9)
     * @param out_dir              Output directory for configurations
     * @param save_observables     Save energy/magnetization trajectories
     * @param T_zero               Perform zero-temperature deterministic sweeps
     * @param n_deterministics     Number of T=0 sweeps (if T_zero=true)
     * @param twist_sweep_count    Twist BC sweeps per MC sweep (default: 100)
     */
    void simulated_annealing(double T_start, double T_end, size_t n_anneal,
                            size_t overrelaxation_rate = 0,
                            bool boundary_update = false,
                            bool gaussian_move = false,
                            double cooling_rate = 0.9,
                            string out_dir = "",
                            bool save_observables = false,
                            bool T_zero = false,
                            size_t n_deterministics = 1000,
                            size_t twist_sweep_count = 100);

    /**
     * Perform detailed measurements at final temperature
     * Computes: energy, specific heat, sublattice magnetizations, and cross-correlations
     * All with binning analysis for error estimation
     */
    void perform_final_measurements(double T_final, double sigma, bool gaussian_move,
                                   size_t overrelaxation_rate, const string& out_dir);

    /**
     * Save sublattice magnetization time series to file
     */
    void save_sublattice_magnetization_timeseries(const string& out_dir,
                                                   const vector<vector<SpinVector>>& sublattice_mags) const;

    /**
     * Compute and save thermodynamic observables
     */
    void compute_and_save_observables(const vector<double>& energies,
                                     const vector<SpinVector>& magnetizations,
                                     double T, const string& out_dir);

    /**
     * Save autocorrelation results
     */
    void save_autocorrelation_results(const string& out_dir, 
                                     const AutocorrelationResult& acf);

    /**
     * Cluster-based annealing (Wolff/SW)
     */
    void cluster_annealing(double T_start, double T_end, size_t n_anneal,
                          size_t wolff_per_temp, bool use_sw = false,
                          bool use_ghost_field = false, double cooling_rate = 0.9,
                          string out_dir = "");

    // ============================================================
    // PARALLEL TEMPERING
    // ============================================================

    /**
     * Parallel tempering with MPI
     * Collects: energy, specific heat, sublattice magnetizations, and cross-correlations
     * All with binning analysis for error estimation
     * 
     * @param temp              Temperature ladder (one per MPI rank)
     * @param n_anneal          Number of equilibration sweeps
     * @param n_measure         Number of measurement sweeps
     * @param overrelaxation_rate Apply overrelaxation every N sweeps (0 = disabled)
     * @param swap_rate         Attempt replica exchange every N sweeps (used when sweeps_per_temp is empty)
     * @param probe_rate        Record observables every N sweeps
     * @param dir_name          Output directory
     * @param rank_to_write     List of ranks that should write output (-1 = all)
     * @param gaussian_move     Use Gaussian moves (true) or uniform (false)
     * @param comm              MPI communicator (default: MPI_COMM_WORLD)
     * @param verbose           If true, save spin configurations
     * @param accumulate_correlations  If true, accumulate real-space correlations for S(q)
     * @param n_bond_types      Number of bond types for dimer correlations (default: 3)
     * @param sweeps_per_temp   Bittner adaptive sweep schedule: sweeps between exchanges per temperature.
     *                          If non-empty, overrides swap_rate. Each rank does sweeps_per_temp[rank] MC
     *                          sweeps between exchange attempts, adapting to local autocorrelation time.
     *                          Computed by generate_optimized_temperature_grid[_mpi].
     */
    void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move = true, MPI_Comm comm = MPI_COMM_WORLD,
                           bool verbose = false, bool accumulate_correlations = false,
                           size_t n_bond_types = 3,
                           const vector<size_t>& sweeps_per_temp = {});
    
    /**
     * Save kagome order parameters to HDF5 file (pyrochlore patch)
     * Nematic order is now component-resolved: [bond_type x spin_component] 3x3 matrix
     * Includes monopole density (signed) and monopole by sublattice
     * Includes global-frame chirality and nematic order parameters
     */
    void save_kagome_order_parameters(const string& rank_dir, double temperature,
                                       const vector<double>& scalar_chi,
                                       const vector<Eigen::Vector3d>& vector_chi,
                                       const vector<Eigen::Matrix3d>& nematic,
                                       const vector<double>& monopole,
                                       const vector<Eigen::Vector4d>& monopole_sub,
                                       const vector<double>& scalar_chi_global,
                                       const vector<Eigen::Vector3d>& vector_chi_global,
                                       const vector<Eigen::Matrix3d>& nematic_global) const;
    
    /**
     * Internal helper to accumulate correlations (avoids name conflict with parameter)
     */
    void accumulate_correlations_internal(RealSpaceCorrelationAccumulator& acc) const {
        // Define site-to-sublattice mapping
        auto site_to_sublattice = [this](size_t site) -> size_t {
            return site % N_atoms;
        };
        
        // Define site-to-cell mapping
        auto site_to_cell = [this](size_t site) -> array<size_t, 3> {
            size_t cell_idx = site / N_atoms;
            size_t n3 = cell_idx % dim3;
            size_t n2 = (cell_idx / dim3) % dim2;
            size_t n1 = cell_idx / (dim2 * dim3);
            return {n1, n2, n3};
        };
        
        // Accumulate spin-spin correlations
        acc.accumulate_spin_correlations(spins, site_to_sublattice, site_to_cell);
        
        // Accumulate dimer-dimer correlations (extracts bonds from bilinear_partners)
        accumulate_dimer_correlations(acc);
    }

    /**
     * Attempt replica exchange between neighboring temperatures
     * @param comm MPI communicator to use (default: MPI_COMM_WORLD)
     */
    int attempt_replica_exchange(int rank, int size, const vector<double>& temp,
                                double curr_Temp, size_t swap_parity, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * Estimate sampling interval using autocorrelation
     */
    size_t estimate_sampling_interval(double curr_Temp, bool gaussian_move, double& sigma,
                                     size_t overrelaxation_rate, size_t n_measure,
                                     size_t probe_rate, int rank);

    /**
     * Gather and save statistics (root process)
     */
    void gather_and_save_statistics(int rank, int size, double curr_Temp,
                                   const vector<double>& energies,
                                   const vector<SpinVector>& magnetizations,
                                   vector<double>& heat_capacity, vector<double>& dHeat,
                                   const vector<double>& temp, const string& dir_name,
                                   const vector<int>& rank_to_write,
                                   size_t n_anneal, size_t n_measure,
                                   double curr_accept, int swap_accept,
                                   size_t swap_rate, size_t overrelaxation_rate,
                                   size_t probe_rate, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * Gather and save comprehensive statistics with binning analysis (MPI version)
     * Includes: energy, specific heat, sublattice magnetizations, cross-correlations
     * @param comm MPI communicator to use (default: MPI_COMM_WORLD)
     */
    void gather_and_save_statistics_comprehensive(int rank, int size, double curr_Temp,
                                   const vector<double>& energies,
                                   const vector<SpinVector>& magnetizations,
                                   const vector<vector<SpinVector>>& sublattice_mags,
                                   vector<double>& heat_capacity, vector<double>& dHeat,
                                   const vector<double>& temp, const string& dir_name,
                                   const vector<int>& rank_to_write,
                                   size_t n_anneal, size_t n_measure,
                                   double curr_accept, int swap_accept,
                                   size_t swap_rate, size_t overrelaxation_rate,
                                   size_t probe_rate, MPI_Comm comm = MPI_COMM_WORLD,
                                   bool verbose = false);

    // ============================================================
    // TEMPERATURE LADDER OPTIMIZATION
    // Based on:
    //   Katzgraber et al., J. Stat. Mech. P03018 (2006) [arXiv:cond-mat/0602085]
    //     - Feedback-optimized temperature placement (Δβ_i ∝ 1/f_i)
    //   Bittner et al., Phys. Rev. Lett. 101, 130603 (2008) [arXiv:0809.0571]
    //     - Temperature-dependent sweep schedule (n_sweeps_i ∝ τ_int(T_i))
    // ============================================================

    /**
     * Generate optimized temperature grid for parallel tempering
     * 
     * Combines two key algorithms:
     * 
     * 1. Katzgraber et al. (2006) feedback-optimized temperature placement:
     *    - Measure the "current fraction" f_i at each edge (fraction of replicas
     *      that make full round trips passing through edge i).
     *    - In practice, f_i ∝ A_i (acceptance rate at edge i) for uniform swap rates.
     *    - The feedback rule adjusts Δβ_i ∝ A_i: edges with HIGH acceptance get
     *      MORE β-spacing, edges with LOW acceptance get LESS spacing (denser temps).
     *    - This concentrates temperatures at bottlenecks (e.g., phase transitions).
     *    - Minimizes round-trip time τ_rt = (Σ_i 1/f_i)².
     * 
     * 2. Bittner et al. (2008) temperature-dependent sweep schedule:
     *    - After optimizing the temperature grid, measure the canonical
     *      autocorrelation time τ_int(T) at each temperature.
     *    - Set the number of MC sweeps between exchange attempts proportional
     *      to τ_int(T_i): n_sweeps_i = n_base × τ_int(T_i) / min(τ_int).
     *    - This ensures each replica is decorrelated before attempting an exchange,
     *      dramatically reducing round-trip time at critical temperatures.
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
     * @return OptimizedTempGridResult containing temperatures, sweep schedule, diagnostics
     */
    OptimizedTempGridResult generate_optimized_temperature_grid(
        double Tmin, double Tmax, size_t R,
        size_t warmup_sweeps = 500,
        size_t sweeps_per_iter = 500,
        size_t feedback_iters = 20,
        bool gaussian_move = false,
        size_t overrelaxation_rate = 0,
        double target_acceptance = 0.5,
        double convergence_tol = 0.05);

    /**
     * Legacy wrapper for backward compatibility
     * Calls the Katzgraber/Bittner-optimized algorithm with default 50% acceptance target
     */
    vector<double> optimize_temperature_ladder_roundtrip(
        double Tmin, double Tmax, size_t R,
        size_t warmup_sweeps = 200,
        size_t sweeps_per_iter = 200,
        size_t feedback_iters = 10,
        bool gaussian_move = false,
        size_t overrelaxation_rate = 0) {
        
        OptimizedTempGridResult result = generate_optimized_temperature_grid(
            Tmin, Tmax, R,
            warmup_sweeps, sweeps_per_iter, feedback_iters,
            gaussian_move, overrelaxation_rate,
            0.5,   // Katzgraber optimal for uniform acceptance
            0.05   // 5% convergence tolerance
        );
        return result.temperatures;
    }

    /**
     * MPI-distributed temperature grid optimization
     * 
     * This version distributes replicas across MPI ranks, same as the main PT simulation.
     * Each rank handles exactly one replica, making it R times faster than the serial version.
     * 
     * Combines:
     *   Katzgraber et al. (2006) - feedback-optimized temperature placement (Δβ_i ∝ A_i)
     *   Bittner et al. (2008) - temperature-dependent sweep schedule (n_i ∝ τ_int(T_i))
     * 
     * @param Tmin              Minimum (coldest) temperature
     * @param Tmax              Maximum (hottest) temperature  
     * @param warmup_sweeps     MC sweeps for initial equilibration
     * @param sweeps_per_iter   MC sweeps per feedback iteration
     * @param feedback_iters    Number of feedback optimization iterations
     * @param gaussian_move     Use Gaussian moves (true) or uniform (false)
     * @param overrelaxation_rate  Apply overrelaxation every N sweeps (0 = disabled)
     * @param target_acceptance Target acceptance rate (default: 0.45, Denschlag et al. 2009)
     * @param convergence_tol   Convergence tolerance for acceptance rate uniformity
     * @param comm              MPI communicator (default: MPI_COMM_WORLD)
     * @param use_gradient      If true, use gradient-based optimizer (Miyata et al. 2024); else Katzgraber
     * @return OptimizedTempGridResult with temperatures, sweep schedule, diagnostics
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
        bool use_gradient = true);

    /**
     * Generate geometric temperature ladder (simple, no optimization)
     * 
     * Uses logarithmic spacing: T_i = T_min * (T_max/T_min)^(i/(R-1))
     * This is optimal for systems with roughly constant specific heat.
     * 
     * @param Tmin  Minimum temperature
     * @param Tmax  Maximum temperature
     * @param R     Number of temperatures
     * @return Vector of temperatures in ascending order
     */
    static vector<double> generate_geometric_temperature_ladder(
        double Tmin, double Tmax, size_t R) {
        
        vector<double> temps(R);
        if (R == 1) {
            temps[0] = Tmin;
            return temps;
        }
        
        for (size_t i = 0; i < R; ++i) {
            double frac = double(i) / double(R - 1);
            temps[i] = Tmin * std::pow(Tmax / Tmin, frac);
        }
        return temps;
    }

    /**
     * Landau-Lifshitz equations (zero-allocation flat array version for ODE integrator)
     * dS/dt = (H_eff - B_drive) × S
     * 
     * Pure flat implementation without Eigen conversions for maximum performance.
     * For SU(2): standard 3D cross product
     * For SU(3): structure constant contraction (a × b)_i = f_{ijk} a_j b_k
     */
    void landau_lifshitz_flat(const double* state_flat, double* dsdt_flat, double t) const;
    
    /**
     * Subtract time-dependent drive field from H array (in-place, flat version)
     * Drive field is pre-transformed to local frame during set_pulse()
     */
    void drive_field_at_time_flat(double t, size_t site_index, double* H) const;

    /**
     * Compute time-dependent drive field (pre-transformed to local frame during set_pulse)
     */
    SpinVector drive_field_at_time(double t, size_t site_index) const;

    /**
     * Set time-dependent pulse (drive field is transformed to local frame)
     */
    void set_pulse(const vector<SpinVector>& field_in1, double t_B1,
                  const vector<SpinVector>& field_in2, double t_B2,
                  double pulse_amp, double pulse_width, double pulse_freq);

    /**
     * Convert SpinConfig to flat state vector for Boost.Odeint
     */
    ODEState spins_to_state(const SpinConfig& spins_vec) const {
        ODEState state(lattice_size * spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t j = 0; j < spin_dim; ++j) {
                state[i * spin_dim + j] = spins_vec[i](j);
            }
        }
        return state;
    }

    /**
     * Convert flat state vector back to SpinConfig
     */
    SpinConfig state_to_spins(const ODEState& state) const {
        SpinConfig spins_vec(lattice_size);
        for (size_t i = 0; i < lattice_size; ++i) {
            spins_vec[i] = SpinVector(spin_dim);
            for (size_t j = 0; j < spin_dim; ++j) {
                spins_vec[i](j) = state[i * spin_dim + j];
            }
        }
        return spins_vec;
    }

private:
    /**
     * Helper: Execute ODE integration with selected method
     * Centralizes integrator selection logic to reduce code duplication
     * 
     * @param system_func   ODE system function (dx/dt = f(x, t))
     * @param state         Initial state vector (modified in-place)
     * @param T_start       Integration start time
     * @param T_end         Integration end time
     * @param dt_step       Time step (fixed for const methods, initial for adaptive)
     * @param observer      Observer function called at each step
     * @param method        Integration method (see list below)
     * @param use_adaptive  If true, use integrate_adaptive; if false, use integrate_const
     * @param abs_tol       Absolute tolerance for adaptive methods
     * @param rel_tol       Relative tolerance for adaptive methods
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
            // Explicit Euler method (1st order, simple but inaccurate)
            odeint::integrate_const(
                odeint::euler<ODEState>(),
                system_func, state, T_start, T_end, dt_step, observer
            );
        } else if (method == "rk2" || method == "midpoint") {
            // Modified midpoint method (2nd order)
            odeint::integrate_const(
                odeint::modified_midpoint<ODEState>(),
                system_func, state, T_start, T_end, dt_step, observer
            );
        } else if (method == "rk4") {
            // Classic fixed-step RK4 (4th order, good balance)
            odeint::integrate_const(
                odeint::runge_kutta4<ODEState>(),
                system_func, state, T_start, T_end, dt_step, observer
            );
        } else if (method == "rk5" || method == "rkck54") {
            // Cash-Karp 5(4) adaptive method
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
            // Runge-Kutta-Fehlberg 5(4) adaptive method
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
        } else if (method == "dopri5") {
            // Dormand-Prince 5(4) adaptive method (default, recommended)
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
            // Runge-Kutta-Fehlberg 7(8) (very high accuracy)
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
            // Bulirsch-Stoer method (very high accuracy, expensive)
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
            // Default to dopri5 if unknown method specified
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

public:
    /**
     * ODE system function for Boost.Odeint: dx/dt = f(x, t)
     * Zero-allocation version working directly with flat arrays
     */
    void ode_system(const ODEState& x, ODEState& dxdt, double t);

    /**
     * Run molecular dynamics simulation using Boost.Odeint with optional GPU acceleration
     * 
     * @param T_start       Start time
     * @param T_end         End time
     * @param dt_initial    Initial step size (adaptive methods will adjust)
     * @param out_dir       Output directory for trajectories
     * @param save_interval Number of steps between saves
     * @param method        Integration method: euler, rk2, rk4, rk5/rkck54, rk54/rkf54, dopri5 (default),
     *                      rk78/rkf78, bulirsch_stoer/bs, adams_bashforth/ab, adams_moulton/am
     * @param use_gpu       Enable GPU acceleration with Thrust (requires CUDA)
     */
    void molecular_dynamics(double T_start, double T_end, double dt_initial,
                           string out_dir = "", size_t save_interval = 100,
                           string method = "dopri5", bool use_gpu = false);

    /**
     * Run molecular dynamics simulation using Boost.Odeint (CPU implementation)
     * Requires HDF5 for output - all non-HDF5 I/O has been retired.
     *
     * @param renorm_interval  If > 0, renormalize spins to |S| = spin_length
     *                         every `renorm_interval` integration steps.
     *                         Landau-Lifshitz dynamics conserves |S| exactly,
     *                         but explicit integrators accumulate drift over
     *                         long runs.  Periodic projection back onto the
     *                         |S|=spin_length sphere preserves physical
     *                         observables (especially energy and entropy)
     *                         without forcing a smaller dt.  Set to 0 to
     *                         disable; recommended value: 100–1000.
     *                         Only applied for SU(2) (spin_dim == 3).
     */
    void molecular_dynamics_cpu(double T_start, double T_end, double dt_initial,
                           string out_dir = "", size_t save_interval = 100,
                           string method = "dopri5",
                           size_t renorm_interval = 0);

#if defined(CUDA_ENABLED) && defined(__CUDACC__)
    /**
     * Run molecular dynamics simulation with GPU acceleration (CUDA/Thrust)
     * Uses true GPU integration - all computation stays on device
     * Only transfers data to host for HDF5 I/O at save intervals
     */
    void molecular_dynamics_gpu(double T_start, double T_end, double dt_initial,
                           string out_dir = "", size_t save_interval = 100,
                           string method = "dopri5") {
#ifndef HDF5_ENABLED
        std::cerr << "Error: HDF5 support is required for molecular dynamics output." << endl;
        std::cerr << "Please rebuild with -DHDF5_ENABLED flag and HDF5 libraries." << endl;
        return;
#endif
        
        if (!out_dir.empty()) {
            std::filesystem::create_directories(out_dir);
        }
        
        cout << "Running molecular dynamics with GPU acceleration: t=" << T_start << " → " << T_end << endl;
        cout << "Integration method: " << method << " (GPU-native)" << endl;
        cout << "Step size: " << dt_initial << endl;
        
        // Ensure GPU data is initialized
        ensure_gpu_data_initialized();
        
        // Create GPU ODE system
        gpu::GPUODESystem gpu_system(gpu_data_cache_);
        
        // Transfer initial state to GPU
        ODEState h_state = spins_to_state(spins);
        gpu::GPUState d_state(h_state.begin(), h_state.end());
        
        // Create HDF5 writer
        std::unique_ptr<HDF5MDWriter> hdf5_writer;
        if (!out_dir.empty()) {
            string hdf5_file = out_dir + "/trajectory.h5";
            cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
            hdf5_writer = std::make_unique<HDF5MDWriter>(
                hdf5_file, lattice_size, spin_dim, N_atoms, 
                dim1, dim2, dim3, method + "_gpu_native", 
                dt_initial, T_start, T_end, save_interval, spin_length, 
                &site_positions, 10000);
        }
        
        // Integrate on GPU - all computation on device, only transfer for I/O
        std::vector<std::pair<double, std::vector<double>>> trajectory;
        gpu::integrate_gpu(gpu_system, d_state, T_start, T_end, dt_initial, 
                          save_interval, trajectory);
        
        // Write trajectory to HDF5 (post-processing on CPU)
        size_t save_count = 0;
        for (const auto& [t, state_vec] : trajectory) {
            double M_local_arr[8] = {0};
            double M_antiferro_arr[8] = {0};
            double M_global_arr[8] = {0};
            
            compute_magnetizations_from_flat(state_vec.data(), 
                lattice_size, spin_dim, M_local_arr, M_antiferro_arr);
            compute_magnetization_global_from_flat(state_vec.data(), M_global_arr);
            compute_magnetization_staggered_from_flat(state_vec.data(), M_antiferro_arr);
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim);
            SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
            
            if (hdf5_writer) {
                hdf5_writer->write_flat_step(t, M_antiferro, M_local, M_global, state_vec.data());
                save_count++;
            }
            
            // Progress output
            if (save_count % 10 == 0) {
                double E = total_energy_flat(state_vec.data()) / lattice_size;
                cout << "t=" << t << ", E/N=" << E << ", |M|=" << M_local.norm() << endl;
            }
        }
        
        // Close HDF5 file
        if (hdf5_writer) {
            hdf5_writer->close();
            cout << "HDF5 trajectory saved with " << save_count << " snapshots" << endl;
        }
        
        cout << "GPU molecular dynamics complete! (" << trajectory.size() << " saved states)" << endl;
    }
#endif // CUDA_ENABLED

    // ============================================================
    // OBSERVABLES
    // ============================================================

    /**
     * Compute global magnetization: M = Σ S_i / N (transformed to global frame)
     */
    SpinVector magnetization_global() const {
        SpinVector M = SpinVector::Zero(spin_dim);
        
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t l = 0; l < N_atoms; ++l) {
                        size_t current_site_index = flatten_index(i, j, k, l);
                        
                        // Transform spin to global frame: spin_global = R * spin_local
                        // where R = sublattice_frames[l] has columns [x_local | y_local | z_local]
                        SpinVector spin_global = SpinVector::Zero(spin_dim);
                        for (size_t mu = 0; mu < spin_dim; ++mu) {
                            for (size_t nu = 0; nu < spin_dim; ++nu) {
                                spin_global(mu) += sublattice_frames[l](mu, nu) * spins[current_site_index](nu);
                            }
                        }
                        M += spin_global;
                    }
                }
            }
        }
        
        return M / double(lattice_size);
    }

    /**
     * Helper function: Compute magnetization_global from flat state array
     * @param x Flat state array [lattice_size * spin_dim]
     * @param M_global_arr Output array to write results [spin_dim]
     */
    void compute_magnetization_global_from_flat(const double* x, double* M_global_arr) const {
        // Initialize output array
        for (size_t d = 0; d < spin_dim; ++d) {
            M_global_arr[d] = 0.0;
        }
        
        // Accumulate global magnetization
        for (size_t i = 0; i < lattice_size; ++i) {
            size_t atom = i % N_atoms;
            size_t idx = i * spin_dim;
            
            // Transform to global frame: spin_global = R * spin_local
            // where R = sublattice_frames[atom] has columns [x_local | y_local | z_local]
            for (size_t mu = 0; mu < spin_dim; ++mu) {
                for (size_t nu = 0; nu < spin_dim; ++nu) {
                    M_global_arr[mu] += sublattice_frames[atom](mu, nu) * x[idx + nu];
                }
            }
        }
        
        // Normalize by lattice size
        for (size_t d = 0; d < spin_dim; ++d) {
            M_global_arr[d] /= double(lattice_size);
        }
    }

    /**
     * Compute staggered (antiferromagnetic) magnetization in global frame.
     * Uses sublattice_frames for local-to-global transformation and
     * afm_sublattice_signs for the staggered pattern (e.g., G-mode for orthoferrites).
     */
    void compute_magnetization_staggered_from_flat(const double* x, double* M_stag_arr) const {
        for (size_t d = 0; d < spin_dim; ++d) {
            M_stag_arr[d] = 0.0;
        }
        
        for (size_t i = 0; i < lattice_size; ++i) {
            size_t atom = i % N_atoms;
            size_t idx = i * spin_dim;
            double sign = afm_sublattice_signs[atom];
            
            for (size_t mu = 0; mu < spin_dim; ++mu) {
                for (size_t nu = 0; nu < spin_dim; ++nu) {
                    M_stag_arr[mu] += sign * sublattice_frames[atom](mu, nu) * x[idx + nu];
                }
            }
        }
        
        for (size_t d = 0; d < spin_dim; ++d) {
            M_stag_arr[d] /= double(lattice_size);
        }
    }

    /**
     * Compute structure factor S(q)
     */
    double structure_factor(const Eigen::Vector3d& q) const {
        std::complex<double> S_q(0, 0);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            double phase = q.dot(site_positions[i]);
            std::complex<double> exp_iqr(std::cos(phase), std::sin(phase));
            
            // Project spin onto first component (generalize for vectorial S(q))
            S_q += spins[i](0) * exp_iqr;
        }
        
        return std::norm(S_q) / double(lattice_size);
    }

    // ============================================================
    // REAL-SPACE CORRELATION ACCUMULATOR
    // ============================================================
    
    /**
     * Create a RealSpaceCorrelationAccumulator initialized for this lattice
     * 
     * @param n_bond_types  Number of distinct bond types (default: N*(N+1)/2 for N sublattices)
     * @return Initialized accumulator ready to accumulate samples
     */
    RealSpaceCorrelationAccumulator create_correlation_accumulator(size_t n_bond_types = 0) const {
        RealSpaceCorrelationAccumulator acc;
        
        if (n_bond_types == 0) {
            // Default: number of undirected sublattice pairs = N*(N+1)/2
            // This covers all possible bond types (0,0), (0,1), (1,1), etc.
            n_bond_types = N_atoms * (N_atoms + 1) / 2;
        }
        
        // Get lattice vectors from unit cell
        array<Eigen::Vector3d, 3> lattice_vectors = {
            unit_cell.lattice_vectors[0],
            unit_cell.lattice_vectors[1],
            unit_cell.lattice_vectors[2]
        };
        
        // Get sublattice positions
        vector<Eigen::Vector3d> sublattice_positions(N_atoms);
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            sublattice_positions[atom] = unit_cell.lattice_pos[atom];
        }
        
        acc.initialize(dim1, dim2, dim3, N_atoms, n_bond_types, spin_dim,
                      lattice_vectors, sublattice_positions);
        
        return acc;
    }
    
    /**
     * Accumulate current spin configuration into the correlation accumulator
     * Call this every probe_rate MC sweeps during measurement phase
     * 
     * @param acc  Reference to the accumulator to update
     */
    void accumulate_correlations(RealSpaceCorrelationAccumulator& acc) const {
        // Define site-to-sublattice mapping
        auto site_to_sublattice = [this](size_t site) -> size_t {
            return site % N_atoms;
        };
        
        // Define site-to-cell mapping
        auto site_to_cell = [this](size_t site) -> array<size_t, 3> {
            size_t cell_idx = site / N_atoms;
            size_t n3 = cell_idx % dim3;
            size_t n2 = (cell_idx / dim3) % dim2;
            size_t n1 = cell_idx / (dim2 * dim3);
            return {n1, n2, n3};
        };
        
        acc.accumulate_spin_correlations(spins, site_to_sublattice, site_to_cell);
    }
    
    /**
     * Accumulate current dimer correlations into the accumulator
     * 
     * This extracts bond information from the bilinear_partners structure
     * and classifies bonds by type based on sublattice pair.
     * 
     * Bond type = sorted sublattice pair (sub_i, sub_j) mapped to linear index.
     * For N sublattices, there are N*(N+1)/2 undirected bond types:
     *   (0,0), (0,1), (1,1), (0,2), (1,2), (2,2), ...
     * 
     * @param acc  Reference to the accumulator to update
     */
    void accumulate_dimer_correlations(RealSpaceCorrelationAccumulator& acc) const {
        // Build bond list from bilinear_partners (only forward bonds to avoid double counting)
        vector<array<size_t, 2>> bonds;
        vector<size_t> bond_types;
        vector<array<size_t, 3>> bond_cells;
        
        // Helper: compute bond type from sorted sublattice pair
        // Maps (min(sub_i, sub_j), max(sub_i, sub_j)) to linear index
        // Using triangular number indexing: type = max*(max+1)/2 + min
        auto sublattice_pair_to_bond_type = [](size_t sub_i, size_t sub_j) -> size_t {
            size_t s_min = std::min(sub_i, sub_j);
            size_t s_max = std::max(sub_i, sub_j);
            return s_max * (s_max + 1) / 2 + s_min;
        };
        
        for (size_t site_i = 0; site_i < lattice_size; ++site_i) {
            size_t sub_i = site_i % N_atoms;
            size_t cell_i = site_i / N_atoms;
            size_t n1 = cell_i / (dim2 * dim3);
            size_t n2 = (cell_i / dim3) % dim2;
            size_t n3 = cell_i % dim3;
            
            for (size_t nb = 0; nb < bilinear_partners[site_i].size(); ++nb) {
                size_t site_j = bilinear_partners[site_i][nb];
                
                // Only count forward bonds (site_i < site_j) to avoid double counting
                if (site_j > site_i) {
                    size_t sub_j = site_j % N_atoms;
                    
                    // Classify bond type by sorted sublattice pair
                    size_t bond_type = sublattice_pair_to_bond_type(sub_i, sub_j);
                    
                    // Clamp to n_bond_types in case accumulator was initialized with fewer
                    if (bond_type >= acc.n_bond_types) {
                        bond_type = bond_type % acc.n_bond_types;
                    }
                    
                    bonds.push_back({site_i, site_j});
                    bond_types.push_back(bond_type);
                    bond_cells.push_back({n1, n2, n3});
                }
            }
        }
        
        acc.accumulate_dimer_correlations(spins, bonds, bond_types, bond_cells);
    }
    
    /**
     * Compute full S(q) tensor from current spin configuration
     * Returns S^{αβ}(q) = (1/N) Σ_{ij} S_i^α S_j^β exp(-i q·(r_i - r_j))
     * 
     * @param q  Wavevector in Cartesian coordinates
     * @return 3x3 structure factor tensor (or spin_dim x spin_dim)
     */
    Eigen::Matrix3d structure_factor_tensor(const Eigen::Vector3d& q) const {
        Eigen::Matrix3d Sq = Eigen::Matrix3d::Zero();
        
        // Compute Fourier components
        Eigen::Vector3cd S_q = Eigen::Vector3cd::Zero();
        for (size_t i = 0; i < lattice_size; ++i) {
            double phase = q.dot(site_positions[i]);
            std::complex<double> exp_iqr(std::cos(phase), std::sin(phase));
            S_q += spins[i].head<3>().cast<std::complex<double>>() * exp_iqr;
        }
        
        // S^{αβ}(q) = (1/N) S_q^α S_{-q}^β = (1/N) S_q^α conj(S_q^β)
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                Sq(a, b) = std::real(S_q(a) * std::conj(S_q(b))) / double(lattice_size);
            }
        }
        
        return Sq;
    }

    // ============================================================
    // FILE I/O
    // ============================================================

    /**
     * Save spin configuration to file
     */
    void save_spin_config(const string& filename) const {
        ofstream file(filename);
        if (!file) {
            std::cerr << "Error: Cannot open file " << filename << endl;
            return;
        }
        
        file << std::scientific << std::setprecision(16);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            for (int j = 0; j < spins[i].size(); ++j) {
                file << spins[i](j);
                if (j < spins[i].size() - 1) file << " ";
            }
            file << "\n";
        }
        
        file.close();
    }

    /**
     * Load spin configuration from file
     */
    void load_spin_config(const string& filename) {
        ifstream file(filename);
        if (!file) {
            std::cerr << "Error: Cannot open file " << filename << endl;
            return;
        }
        
        size_t idx = 0;
        string line;
        
        while (std::getline(file, line) && idx < lattice_size) {
            std::istringstream iss(line);
            for (int j = 0; j < spin_dim; ++j) {
                double val;
                if (!(iss >> val)) {
                    std::cerr << "Error: Incomplete spin data at line " << idx << endl;
                    file.close();
                    return;
                }
                spins[idx](j) = val;
            }
            ++idx;
        }
        
        file.close();
        
        if (idx != lattice_size) {
            std::cerr << "Warning: File contained " << idx << " spins, expected " << lattice_size << endl;
        }
    }

    /**
     * Save twist boundary condition data to file
     * Includes both axis-angle representation and full SO(3) matrices
     */
    void save_twist_angles(const string& filename) const {
        ofstream file(filename);
        if (!file) {
            std::cerr << "Error: Cannot open file " << filename << endl;
            return;
        }
        
        file << std::scientific << std::setprecision(16);
        file << "# Twist boundary condition data for each dimension\n";
        file << "# Section 1: Axis-angle representation\n";
        file << "# Format: dimension axis_x axis_y axis_z angle(rad)\n";
        
        for (size_t d = 0; d < 3; ++d) {
            file << d << " ";
            for (size_t i = 0; i < rotation_axis[d].size(); ++i) {
                file << rotation_axis[d](i) << " ";
            }
            file << twist_angles[d] << "\n";
        }
        
        file << "\n# Section 2: Full SO(3) rotation matrices\n";
        file << "# Format: dimension followed by 3x3 matrix (row-major)\n";
        
        for (size_t d = 0; d < 3; ++d) {
            file << "# Dimension " << d << " twist matrix:\n";
            file << d << "\n";
            for (size_t row = 0; row < twist_matrices[d].rows(); ++row) {
                for (size_t col = 0; col < twist_matrices[d].cols(); ++col) {
                    file << twist_matrices[d](row, col);
                    if (col < twist_matrices[d].cols() - 1) file << " ";
                }
                file << "\n";
            }
        }
        
        file.close();
    }

    /**
     * Save site positions to file
     */
    void save_positions(const string& filename) const {
        ofstream file(filename);
        if (!file) {
            std::cerr << "Error: Cannot open file " << filename << endl;
            return;
        }
        
        file << std::scientific << std::setprecision(16);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            file << site_positions[i](0) << " " 
                 << site_positions[i](1) << " "
                 << site_positions[i](2) << "\n";
        }
        
        file.close();
    }

    /**
     * Initialize spins from file
     */
    void read_spins_from_file(const string& filename) {
        load_spin_config(filename);
    }

    /**
     * Set a specific spin
     */
    void set_spin(size_t site_index, const SpinVector& spin_in) {
        if (site_index < lattice_size) {
            spins[site_index] = spin_in;
        }
    }

    /**
     * Get a specific spin
     */
    const SpinVector& get_spin(size_t site_index) const {
        return spins[site_index];
    }

    /**
     * Initialize with ferromagnetic configuration
     */
    void init_ferromagnetic(const SpinVector& direction) {
        SpinVector spin_aligned = direction.normalized() * spin_length;
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = spin_aligned;
        }
    }

    /**
     * Initialize with Néel (antiferromagnetic) configuration
     */
    void init_neel(const SpinVector& direction) {
        SpinVector spin_up = direction.normalized() * spin_length;
        SpinVector spin_down = -spin_up;
        
        for (size_t idx = 0; idx < lattice_size; ++idx) {
            size_t i = idx / (N_atoms * dim2 * dim3);
            size_t j = (idx / (N_atoms * dim3)) % dim2;
            size_t k = (idx / N_atoms) % dim3;
            
            spins[idx] = ((i + j + k) % 2 == 0) ? spin_up : spin_down;
        }
    }

    /**
     * Initialize with random spins
     */
    void init_random() {
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = gen_random_spin(spin_length);
        }
    }

    /**
     * Print lattice information
     */
    void print_info() const {
        cout << "=== Lattice Information ===" << endl;
        cout << "Dimensions: " << dim1 << " × " << dim2 << " × " << dim3 << endl;
        cout << "Atoms per cell: " << N_atoms << endl;
        cout << "Total sites: " << lattice_size << endl;
        cout << "Spin dimension: " << spin_dim << endl;
        cout << "Spin length: " << spin_length << endl;
        cout << "Max bilinear neighbors per site: " << num_bi << endl;
        cout << "Max trilinear interactions per site: " << num_tri << endl;
        cout << "Current energy density: " << energy_density() << endl;
        cout << "Current magnetization: |M| = " << magnetization_global().norm() << endl;
    }

    // ============================================================
    // ADDITIONAL MAGNETIZATION OBSERVABLES
    // ============================================================

    /**
     * Compute local magnetization (simple average, no frame transformation)
     */
    SpinVector magnetization_local() const {
        SpinVector M = SpinVector::Zero(spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            M += spins[i];
        }
        return M / double(lattice_size);
    }

    /**
     * Compute antiferromagnetic magnetization with alternating signs
     */
    SpinVector magnetization_local_antiferro() const {
        SpinVector M = SpinVector::Zero(spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            M += spins[i] * std::pow(-1.0, static_cast<double>(i));
        }
        return M / double(lattice_size);
    }

    // ============================================================
    // TIME-DEPENDENT FIELD MOLECULAR DYNAMICS
    // ============================================================

private:
    /**
     * Helper: Perform MC sweeps with optional overrelaxation
     * Returns sum of acceptance rates from metropolis calls
     * 
     * @param n_sweeps             Number of MC sweeps
     * @param T                    Temperature
     * @param gaussian_move        Use Gaussian moves
     * @param sigma                Gaussian width (modified in-place)
     * @param overrelaxation_rate  Overrelaxation frequency (0 = disabled)
     * @param boundary_update      Enable twist boundary updates
     * @param twist_sweep_count    Number of twist sweeps per MC sweep (default: 100)
     * @param twist_acc_ptr        Optional pointer to store cumulative twist acceptance count
     */
    double perform_mc_sweeps(size_t n_sweeps, double T, bool gaussian_move, 
                            double& sigma, size_t overrelaxation_rate = 0,
                            bool boundary_update = false,
                            size_t twist_sweep_count = 100,
                            size_t* twist_acc_ptr = nullptr);

    /**
     * Helper: Collect energy samples with regular MC sweeps
     */
    vector<double> collect_energy_samples(size_t n_samples, size_t interval,
                                         double T, bool gaussian_move, double& sigma,
                                         size_t overrelaxation_rate = 0);

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
            std::filesystem::create_directories(dir_path);
        }
    }

    /**
     * Helper: Compute local and antiferromagnetic magnetization from flat state
     * @param x Flat state array
     * @param lattice_size Number of sites
     * @param spin_dim Spin dimension
     * @param M_local_arr Output array for local magnetization
     * @param M_antiferro_arr Output array for antiferromagnetic magnetization
     */
    static void compute_magnetizations_from_flat(const double* x, size_t lattice_size, 
                                                 size_t spin_dim, double* M_local_arr, 
                                                 double* M_antiferro_arr) {
        std::fill(M_local_arr, M_local_arr + spin_dim, 0.0);
        std::fill(M_antiferro_arr, M_antiferro_arr + spin_dim, 0.0);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            double sign = (i % 2 == 0) ? 1.0 : -1.0;
            for (size_t d = 0; d < spin_dim; ++d) {
                M_local_arr[d] += x[i * spin_dim + d];
                M_antiferro_arr[d] += x[i * spin_dim + d] * sign;
            }
        }
    }

    /**
     * Helper: Save energy and magnetization time series to files
     */
    void save_observables(const string& dir_path,
                         const vector<double>& energies,
                         const vector<SpinVector>& magnetizations);


public:
    /**
     * Molecular dynamics with single pulse field
     * Returns magnetization trajectory without I/O
     * @param use_gpu Enable GPU acceleration
     */
    vector<pair<double, array<SpinVector, 3>>> single_pulse_drive(
               const vector<SpinVector>& field_in, double t_B, 
               double pulse_amp, double pulse_width, double pulse_freq,
               double T_start, double T_end, double step_size,
               string method = "dopri5", bool use_gpu = false);

    /**
     * Molecular dynamics with two-pulse field
     * Returns magnetization trajectory without I/O
     * @param use_gpu Enable GPU acceleration
     */
    vector<pair<double, array<SpinVector, 3>>> double_pulse_drive(
                   const vector<SpinVector>& field_in_1, double t_B_1,
                   const vector<SpinVector>& field_in_2, double t_B_2,
                   double pulse_amp, double pulse_width, double pulse_freq,
                   double T_start, double T_end, double step_size,
                   string method = "dopri5", bool use_gpu = false);

    /**
     * Complete pump-probe nonlinear spectroscopy workflow
     * 
     * This method performs a typical 2D coherent spectroscopy experiment:
     * 1. Uses current spin configuration as ground state (assumed pre-loaded)
     * 2. Runs reference single-pulse dynamics (pump at t=0)
     * 3. Scans delay times (tau) to measure pump-probe response
     * 
     * For each tau value, computes:
     * - M1(t, tau): Response to probe pulse at time tau
     * - M01(t, tau): Response to pump (t=0) + probe (t=tau)
     * 
     * This enables extraction of nonlinear response via:
     * M_nonlinear = M01 - M0 - M1
     * 
     * NOTE: Ground state should be prepared beforehand via simulated_annealing()
     *       or loaded from file before calling this method.
     * 
     * @param field_in          Pulse field direction (one per sublattice)
     * @param pulse_amp         Pulse amplitude
     * @param pulse_width       Gaussian pulse width
     * @param pulse_freq        Pulse oscillation frequency
     * @param tau_start         Initial delay time
     * @param tau_end           Final delay time
     * @param tau_step          Delay time step
     * @param T_start           Integration start time
     * @param T_end             Integration end time
     * @param T_step            Integration time step
     * @param Temp_start        Annealing start temperature (for metadata only)
     * @param Temp_end          Annealing end temperature (for metadata only)
     * @param n_anneal          Sweeps per temperature (for metadata only)
     * @param T_zero_quench     Was T=0 quench used? (for metadata only)
     * @param quench_sweeps     Number of deterministic sweeps (for metadata only)
     * @param dir_name          Output directory
     * @param method            ODE integration method
     */
    void pump_probe_spectroscopy(const vector<SpinVector>& field_in,
                                 double pulse_amp, double pulse_width, double pulse_freq,
                                 double tau_start, double tau_end, double tau_step,
                                 double T_start, double T_end, double T_step,
                                 double Temp_start = 5.0, double Temp_end = 1e-3,
                                 size_t n_anneal = 1000,
                                 bool T_zero_quench = false, size_t quench_sweeps = 1000,
                                 string dir_name = "spectroscopy", string method = "dopri5",
                                 bool use_gpu = false);

    /**
     * MPI-parallelized pump-probe spectroscopy
     * 
     * Distributes tau delay values across MPI ranks for parallel computation.
     * Each rank computes a subset of tau values, then rank 0 gathers and writes results.
     * 
     * This is more efficient than trial-based parallelization when num_trials == 1
     * since each tau delay is independent and can be computed in parallel.
     * 
     * @param field_in        Pulse direction for each sublattice
     * @param pulse_amp       Pulse amplitude
     * @param pulse_width     Pulse width (Gaussian)
     * @param pulse_freq      Pulse frequency
     * @param tau_start       Starting delay time
     * @param tau_end         Ending delay time
     * @param tau_step        Delay time step
     * @param T_start         Integration start time
     * @param T_end           Integration end time
     * @param T_step          Integration time step
     * @param Temp_start      Annealing start temperature (for equilibration info)
     * @param Temp_end        Annealing end temperature
     * @param n_anneal        Number of annealing steps
     * @param T_zero_quench   Whether T=0 quench was used
     * @param quench_sweeps   Number of quench sweeps
     * @param dir_name        Output directory
     * @param method          ODE integration method
     * @param use_gpu         Use GPU acceleration
     */
    void pump_probe_spectroscopy_mpi(const vector<SpinVector>& field_in,
                                     double pulse_amp, double pulse_width, double pulse_freq,
                                     double tau_start, double tau_end, double tau_step,
                                     double T_start, double T_end, double T_step,
                                     double Temp_start = 5.0, double Temp_end = 1e-3,
                                     size_t n_anneal = 1000,
                                     bool T_zero_quench = false, size_t quench_sweeps = 1000,
                                     string dir_name = "spectroscopy", string method = "dopri5",
                                     bool use_gpu = false);

    // Note: GPU-accelerated methods use the modular GPU implementation in lattice_gpu.cuh/cu
    // For C++ compilation, use_gpu parameter will automatically fallback to CPU implementation.

    /**
     * Return true if any site has a trilinear coupling configured.
     *
     * The GPU code path in `src/gpu/lattice_gpu.cu::compute_local_field_device`
     * currently does not include a trilinear term, and
     * `create_gpu_lattice_data_internal` never uploads the trilinear tables.
     * Callers therefore must refuse to run on GPU when any trilinear coupling
     * is present, otherwise the simulation silently drops real physics.
     */
    bool has_trilinear_interactions() const {
        for (size_t i = 0; i < trilinear_partners.size(); ++i) {
            if (!trilinear_partners[i].empty()) return true;
        }
        return false;
    }

#if defined(CUDA_ENABLED) && defined(__CUDACC__)
private:
    // GPU data cache for avoiding repeated transfers
    mutable gpu::GPULatticeData gpu_data_cache_;
    mutable bool gpu_data_initialized_ = false;
    
    /**
     * Ensure GPU lattice data is initialized (lazy initialization)
     * Uses the modular gpu:: implementation from lattice_gpu.cuh/cu
     */
    void ensure_gpu_data_initialized() const {
        if (gpu_data_initialized_) return;

        if (has_trilinear_interactions()) {
            throw std::runtime_error(
                "Lattice::ensure_gpu_data_initialized: this lattice has "
                "trilinear couplings, but the GPU code path in "
                "src/gpu/lattice_gpu.cu does not implement them yet. "
                "Set use_gpu=false, or run on a Hamiltonian without "
                "trilinear terms. (See audit item T3.)");
        }

        // Flatten field data
        vector<double> flat_field;
        flat_field.reserve(lattice_size * spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t d = 0; d < spin_dim; ++d) {
                flat_field.push_back(field[i](d));
            }
        }
        
        // Flatten onsite interaction matrices
        vector<double> flat_onsite;
        flat_onsite.reserve(lattice_size * spin_dim * spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t r = 0; r < spin_dim; ++r) {
                for (size_t c = 0; c < spin_dim; ++c) {
                    flat_onsite.push_back(onsite_interaction[i](r, c));
                }
            }
        }
        
        // Flatten bilinear interaction data
        vector<double> flat_bilinear;
        vector<size_t> flat_partners;
        vector<size_t> num_bilinear_per_site;
        
        flat_bilinear.reserve(lattice_size * num_bi * spin_dim * spin_dim);
        flat_partners.reserve(lattice_size * num_bi);
        num_bilinear_per_site.reserve(lattice_size);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            num_bilinear_per_site.push_back(bilinear_partners[i].size());
            for (size_t n = 0; n < num_bi; ++n) {
                if (n < bilinear_partners[i].size()) {
                    flat_partners.push_back(bilinear_partners[i][n]);
                    for (size_t r = 0; r < spin_dim; ++r) {
                        for (size_t c = 0; c < spin_dim; ++c) {
                            flat_bilinear.push_back(bilinear_interaction[i][n](r, c));
                        }
                    }
                } else {
                    flat_partners.push_back(0);
                    for (size_t j = 0; j < spin_dim * spin_dim; ++j) {
                        flat_bilinear.push_back(0.0);
                    }
                }
            }
        }
        
        // Create GPU data using the modular implementation
        gpu_data_cache_ = gpu::create_gpu_lattice_data(
            lattice_size, spin_dim, N_atoms, num_bi,
            flat_field, flat_onsite, flat_bilinear, 
            flat_partners, num_bilinear_per_site
        );
        
        gpu_data_initialized_ = true;
    }
    
    /**
     * Update GPU pulse parameters
     */
    void update_gpu_pulse() const {
        vector<double> flat_field_drive;
        flat_field_drive.reserve(2 * N_atoms * spin_dim);
        for (size_t p = 0; p < 2; ++p) {
            for (size_t d = 0; d < field_drive[p].size(); ++d) {
                flat_field_drive.push_back(field_drive[p](d));
            }
        }
        
        gpu::set_gpu_pulse(
            gpu_data_cache_,
            flat_field_drive,
            field_drive_amp,
            field_drive_width,
            field_drive_freq,
            t_pulse[0],
            t_pulse[1]
        );
    }
    
    /**
     * GPU version of single_pulse_drive using true GPU integration
     * Uses gpu::integrate_gpu for pure GPU execution without per-step host transfers
     */
    vector<pair<double, array<SpinVector, 3>>> single_pulse_drive_gpu(
               const vector<SpinVector>& field_in, double t_B,
               double pulse_amp, double pulse_width, double pulse_freq,
               double T_start, double T_end, double step_size,
               string method = "dopri5") {
        
        // Set up pulse on CPU side first
        set_pulse(field_in, t_B, vector<SpinVector>(N_atoms, SpinVector::Zero(spin_dim)), 
                 0.0, pulse_amp, pulse_width, pulse_freq);
        
        // Ensure GPU data is initialized and update pulse
        ensure_gpu_data_initialized();
        update_gpu_pulse();
        
        // Create GPU ODE system
        gpu::GPUODESystem gpu_system(gpu_data_cache_);
        
        // Transfer initial state to GPU
        ODEState h_state = spins_to_state(spins);
        gpu::GPUState d_state(h_state.begin(), h_state.end());
        
        // Calculate save interval from step size
        double total_time = T_end - T_start;
        size_t total_steps = static_cast<size_t>(total_time / step_size) + 1;
        size_t save_interval = 1;  // Save every step for trajectory output
        
        // Integrate on GPU - all computation stays on device
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        gpu::integrate_gpu(gpu_system, d_state, T_start, T_end, step_size, 
                          save_interval, raw_trajectory);
        
        // Convert raw trajectory to magnetization trajectory
        vector<pair<double, array<SpinVector, 3>>> trajectory;
        trajectory.reserve(raw_trajectory.size());
        
        for (const auto& [t, state_vec] : raw_trajectory) {
            double M_local_arr[8] = {0};
            double M_antiferro_arr[8] = {0};
            double M_global_arr[8] = {0};
            
            compute_magnetizations_from_flat(state_vec.data(), 
                lattice_size, spin_dim, M_local_arr, M_antiferro_arr);
            compute_magnetization_global_from_flat(state_vec.data(), M_global_arr);
            compute_magnetization_staggered_from_flat(state_vec.data(), M_antiferro_arr);
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim);
            SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
            
            trajectory.push_back({t, {M_antiferro, M_local, M_global}});
        }
        
        // Reset pulse
        field_drive[0] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive[1] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive_amp = 0.0;
        
        return trajectory;
    }
    
    /**
     * GPU version of double_pulse_drive using true GPU integration
     */
    vector<pair<double, array<SpinVector, 3>>> double_pulse_drive_gpu(
               const vector<SpinVector>& field_in_1, double t_B_1,
               const vector<SpinVector>& field_in_2, double t_B_2,
               double pulse_amp, double pulse_width, double pulse_freq,
               double T_start, double T_end, double step_size,
               string method = "dopri5") {
        
        // Set up two-pulse configuration
        set_pulse(field_in_1, t_B_1, field_in_2, t_B_2, 
                 pulse_amp, pulse_width, pulse_freq);
        
        // Ensure GPU data is initialized and update pulse
        ensure_gpu_data_initialized();
        update_gpu_pulse();
        
        // Create GPU ODE system
        gpu::GPUODESystem gpu_system(gpu_data_cache_);
        
        // Transfer initial state to GPU
        ODEState h_state = spins_to_state(spins);
        gpu::GPUState d_state(h_state.begin(), h_state.end());
        
        // Integrate on GPU
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        gpu::integrate_gpu(gpu_system, d_state, T_start, T_end, step_size, 
                          1, raw_trajectory);
        
        // Convert raw trajectory to magnetization trajectory
        vector<pair<double, array<SpinVector, 3>>> trajectory;
        trajectory.reserve(raw_trajectory.size());
        
        for (const auto& [t, state_vec] : raw_trajectory) {
            double M_local_arr[8] = {0};
            double M_antiferro_arr[8] = {0};
            double M_global_arr[8] = {0};
            
            compute_magnetizations_from_flat(state_vec.data(), 
                lattice_size, spin_dim, M_local_arr, M_antiferro_arr);
            compute_magnetization_global_from_flat(state_vec.data(), M_global_arr);
            compute_magnetization_staggered_from_flat(state_vec.data(), M_antiferro_arr);
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim);
            SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
            
            trajectory.push_back({t, {M_antiferro, M_local, M_global}});
        }
        
        // Reset pulse
        field_drive[0] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive[1] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive_amp = 0.0;
        
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
    mutable gpu::GPULatticeDataHandle* gpu_handle_ = nullptr;
    mutable bool gpu_data_initialized_ = false;
    
    /**
     * Ensure GPU lattice data is initialized (lazy initialization)
     * Uses the opaque API from lattice_gpu_api.h
     */
    void ensure_gpu_data_initialized() const {
        if (gpu_data_initialized_) return;

        if (has_trilinear_interactions()) {
            throw std::runtime_error(
                "Lattice::ensure_gpu_data_initialized: this lattice has "
                "trilinear couplings, but the GPU code path in "
                "src/gpu/lattice_gpu.cu does not implement them yet. "
                "Set use_gpu=false, or run on a Hamiltonian without "
                "trilinear terms. (See audit item T3.)");
        }

        // Flatten field data
        vector<double> flat_field;
        flat_field.reserve(lattice_size * spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t d = 0; d < spin_dim; ++d) {
                flat_field.push_back(field[i](d));
            }
        }
        
        // Flatten onsite interaction matrices
        vector<double> flat_onsite;
        flat_onsite.reserve(lattice_size * spin_dim * spin_dim);
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t r = 0; r < spin_dim; ++r) {
                for (size_t c = 0; c < spin_dim; ++c) {
                    flat_onsite.push_back(onsite_interaction[i](r, c));
                }
            }
        }
        
        // Flatten bilinear interaction data
        vector<double> flat_bilinear;
        vector<size_t> flat_partners;
        vector<size_t> num_bilinear_per_site;
        
        flat_bilinear.reserve(lattice_size * num_bi * spin_dim * spin_dim);
        flat_partners.reserve(lattice_size * num_bi);
        num_bilinear_per_site.reserve(lattice_size);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            num_bilinear_per_site.push_back(bilinear_partners[i].size());
            for (size_t n = 0; n < num_bi; ++n) {
                if (n < bilinear_partners[i].size()) {
                    flat_partners.push_back(bilinear_partners[i][n]);
                    for (size_t r = 0; r < spin_dim; ++r) {
                        for (size_t c = 0; c < spin_dim; ++c) {
                            flat_bilinear.push_back(bilinear_interaction[i][n](r, c));
                        }
                    }
                } else {
                    flat_partners.push_back(0);
                    for (size_t j = 0; j < spin_dim * spin_dim; ++j) {
                        flat_bilinear.push_back(0.0);
                    }
                }
            }
        }
        
        // Create GPU data using opaque API
        gpu_handle_ = gpu::create_gpu_lattice_data(
            lattice_size, spin_dim, N_atoms, num_bi,
            flat_field, flat_onsite, flat_bilinear, 
            flat_partners, num_bilinear_per_site
        );
        
        gpu_data_initialized_ = true;
    }
    
    /**
     * Update GPU pulse parameters
     */
    void update_gpu_pulse() const {
        if (!gpu_handle_) return;
        
        vector<double> flat_field_drive;
        flat_field_drive.reserve(2 * N_atoms * spin_dim);
        for (size_t p = 0; p < 2; ++p) {
            for (size_t d = 0; d < field_drive[p].size(); ++d) {
                flat_field_drive.push_back(field_drive[p](d));
            }
        }
        
        gpu::set_gpu_pulse(
            gpu_handle_,
            flat_field_drive,
            field_drive_amp,
            field_drive_width,
            field_drive_freq,
            t_pulse[0],
            t_pulse[1]
        );
    }
    
    /**
     * GPU version of molecular_dynamics using opaque API
     */
    void molecular_dynamics_gpu(double T_start, double T_end, double dt_initial,
                           string out_dir = "", size_t save_interval = 100,
                           string method = "dopri5") {
#ifndef HDF5_ENABLED
        std::cerr << "Error: HDF5 support is required for molecular dynamics output." << endl;
        return;
#else
        if (!out_dir.empty()) {
            std::filesystem::create_directories(out_dir);
        }
        
        cout << "Running molecular dynamics with GPU acceleration: t=" << T_start << " → " << T_end << endl;
        cout << "Integration method: " << method << " (GPU via API)" << endl;
        cout << "Step size: " << dt_initial << endl;
        
        // Ensure GPU data is initialized
        ensure_gpu_data_initialized();
        
        // Transfer initial state to GPU
        ODEState h_state = spins_to_state(spins);
        gpu::set_gpu_spins(gpu_handle_, h_state);
        
        // Create HDF5 writer
        std::unique_ptr<HDF5MDWriter> hdf5_writer;
        if (!out_dir.empty()) {
            string hdf5_file = out_dir + "/trajectory.h5";
            cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
            hdf5_writer = std::make_unique<HDF5MDWriter>(
                hdf5_file, lattice_size, spin_dim, N_atoms, 
                dim1, dim2, dim3, method + "_gpu_api", 
                dt_initial, T_start, T_end, save_interval, spin_length, 
                &site_positions, 10000);
        }
        
        // Integrate on GPU
        std::vector<std::pair<double, std::vector<double>>> trajectory;
        gpu::integrate_gpu(gpu_handle_, T_start, T_end, dt_initial, 
                          save_interval, trajectory, method);
        
        // Write trajectory to HDF5 (post-processing on CPU)
        size_t save_count = 0;
        for (const auto& [t, state_vec] : trajectory) {
            double M_local_arr[8] = {0};
            double M_antiferro_arr[8] = {0};
            double M_global_arr[8] = {0};
            
            compute_magnetizations_from_flat(state_vec.data(), 
                lattice_size, spin_dim, M_local_arr, M_antiferro_arr);
            compute_magnetization_global_from_flat(state_vec.data(), M_global_arr);
            compute_magnetization_staggered_from_flat(state_vec.data(), M_antiferro_arr);
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim);
            SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
            
            if (hdf5_writer) {
                hdf5_writer->write_flat_step(t, M_antiferro, M_local, M_global, state_vec.data());
                save_count++;
            }
            
            // Progress output
            if (save_count % 10 == 0) {
                double E = total_energy_flat(state_vec.data()) / lattice_size;
                cout << "t=" << t << ", E/N=" << E << ", |M|=" << M_local.norm() << endl;
            }
        }
        
        // Close HDF5 file
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
    vector<pair<double, array<SpinVector, 3>>> single_pulse_drive_gpu(
               const vector<SpinVector>& field_in, double t_B,
               double pulse_amp, double pulse_width, double pulse_freq,
               double T_start, double T_end, double step_size,
               string method = "dopri5") {
        
        // Set up pulse
        set_pulse(field_in, t_B, vector<SpinVector>(N_atoms, SpinVector::Zero(spin_dim)), 
                 0.0, pulse_amp, pulse_width, pulse_freq);
        
        // Ensure GPU data is initialized
        ensure_gpu_data_initialized();
        update_gpu_pulse();
        
        // Transfer initial state to GPU
        ODEState h_state = spins_to_state(spins);
        gpu::set_gpu_spins(gpu_handle_, h_state);
        
        // Integrate on GPU
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        gpu::integrate_gpu(gpu_handle_, T_start, T_end, step_size, 
                          1, raw_trajectory, method);
        
        // Convert raw trajectory to magnetization trajectory
        vector<pair<double, array<SpinVector, 3>>> trajectory;
        trajectory.reserve(raw_trajectory.size());
        
        for (const auto& [t, state_vec] : raw_trajectory) {
            double M_local_arr[8] = {0};
            double M_antiferro_arr[8] = {0};
            double M_global_arr[8] = {0};
            
            compute_magnetizations_from_flat(state_vec.data(), 
                lattice_size, spin_dim, M_local_arr, M_antiferro_arr);
            compute_magnetization_global_from_flat(state_vec.data(), M_global_arr);
            compute_magnetization_staggered_from_flat(state_vec.data(), M_antiferro_arr);
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim);
            SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
            
            trajectory.push_back({t, {M_antiferro, M_local, M_global}});
        }
        
        // Reset pulse
        field_drive[0] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive[1] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive_amp = 0.0;
        
        return trajectory;
    }
    
    /**
     * GPU version of double_pulse_drive using opaque API
     */
    vector<pair<double, array<SpinVector, 3>>> double_pulse_drive_gpu(
                   const vector<SpinVector>& field_in_1, double t_B_1,
                   const vector<SpinVector>& field_in_2, double t_B_2,
                   double pulse_amp, double pulse_width, double pulse_freq,
                   double T_start, double T_end, double step_size,
                   string method = "dopri5") {
        
        // Set up two-pulse configuration
        set_pulse(field_in_1, t_B_1, field_in_2, t_B_2, 
                 pulse_amp, pulse_width, pulse_freq);
        
        // Ensure GPU data is initialized
        ensure_gpu_data_initialized();
        update_gpu_pulse();
        
        // Transfer initial state to GPU
        ODEState h_state = spins_to_state(spins);
        gpu::set_gpu_spins(gpu_handle_, h_state);
        
        // Integrate on GPU
        std::vector<std::pair<double, std::vector<double>>> raw_trajectory;
        gpu::integrate_gpu(gpu_handle_, T_start, T_end, step_size, 
                          1, raw_trajectory, method);
        
        // Convert raw trajectory to magnetization trajectory
        vector<pair<double, array<SpinVector, 3>>> trajectory;
        trajectory.reserve(raw_trajectory.size());
        
        for (const auto& [t, state_vec] : raw_trajectory) {
            double M_local_arr[8] = {0};
            double M_antiferro_arr[8] = {0};
            double M_global_arr[8] = {0};
            
            compute_magnetizations_from_flat(state_vec.data(), 
                lattice_size, spin_dim, M_local_arr, M_antiferro_arr);
            compute_magnetization_global_from_flat(state_vec.data(), M_global_arr);
            compute_magnetization_staggered_from_flat(state_vec.data(), M_antiferro_arr);
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim);
            SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
            
            trajectory.push_back({t, {M_antiferro, M_local, M_global}});
        }
        
        // Reset pulse
        field_drive[0] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive[1] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive_amp = 0.0;
        
        return trajectory;
    }
#endif // defined(CUDA_ENABLED) && !defined(__CUDACC__)

};

#endif // LATTICE_REFACTORED_H
