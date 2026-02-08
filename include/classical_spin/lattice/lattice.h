#ifndef LATTICE_REFACTORED_H
#define LATTICE_REFACTORED_H

#include "unitcell.h"
#include "simple_linear_alg.h"
#include "hdf5_io.h"
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

// Helper struct for autocorrelation analysis
struct AutocorrelationResult {
    double tau_int;
    size_t sampling_interval;
    vector<double> correlation_function;
};

// Simulated annealing parameters
struct SAParams {
    double T_start = 1.0;
    double T_end = 1e-3;
    double cooling_rate = 0.9;
    size_t sweeps_per_temp = 100;
    vector<double> probe_T;
    vector<double> probe_acc;
    vector<double> probe_tau;
};

// Binning analysis result for error estimation
struct BinningResult {
    double mean;
    double error;
    double tau_int;  // Integrated autocorrelation time estimate from binning
    size_t optimal_bin_level;
    vector<double> errors_by_level;  // Errors at each binning level
};

// Observable with uncertainty (mean ± error)
struct Observable {
    double value;
    double error;
    
    Observable(double v = 0.0, double e = 0.0) : value(v), error(e) {}
};

// Vector observable with uncertainty for each component
struct VectorObservable {
    vector<double> values;
    vector<double> errors;
    
    VectorObservable() = default;
    VectorObservable(size_t dim) : values(dim, 0.0), errors(dim, 0.0) {}
};

// Complete set of thermodynamic observables with uncertainties
struct ThermodynamicObservables {
    double temperature;
    Observable energy;                      // <E>/N
    Observable specific_heat;               // C_V = (<E²> - <E>²) / (T² N)
    VectorObservable magnetization;         // <M> = <Σ_i S_i> / N (total magnetization per site)
    vector<VectorObservable> sublattice_magnetization;  // <S_α> for each sublattice α
    vector<VectorObservable> energy_sublattice_cross;   // <E * S_α> - <E><S_α> for each sublattice
};

// Result from optimized temperature grid generation
// Based on Bittner et al., Phys. Rev. Lett. 101, 130603 (2008) [arXiv:0809.0571]
struct OptimizedTempGridResult {
    vector<double> temperatures;              // Optimized temperature ladder
    vector<double> acceptance_rates;          // Final acceptance rates between adjacent pairs
    vector<double> local_diffusivities;       // Local diffusivity D(T) ∝ A(1-A) at each T
    double mean_acceptance_rate;              // Average acceptance rate across all pairs
    double round_trip_estimate;               // Estimated round-trip time in sweeps
    size_t feedback_iterations_used;          // Number of feedback iterations performed
    bool converged;                           // Whether the algorithm converged
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
                   const vector<Eigen::Vector3d>& sublattice_positions) {
        dim1 = d1;
        dim2 = d2;
        dim3 = d3;
        n_sublattices = n_sub;
        spin_dim = sdim;
        n_sites = d1 * d2 * d3 * n_sub;
        
        // Cell displacements (no sublattice info)
        n_cell_displacements = d1 * d2 * d3;
        
        // Sublattice pairs with symmetry: (i,j) with i ≤ j
        n_sublattice_pairs = n_sub * (n_sub + 1) / 2;
        
        // Bond types = sublattice pairs (bond classified by which sublattices it connects)
        n_bond_types = n_sublattice_pairs;
        
        // Total spin correlation storage: [cell_disp][sub_pair][spin_comp]
        size_t spin_corr_size = n_cell_displacements * n_sublattice_pairs * n_spin_components;
        spin_corr_sum.resize(spin_corr_size, 0.0);
        spin_corr_sq_sum.resize(spin_corr_size, 0.0);
        spin_mean_sum.resize(n_sublattices, Eigen::Vector3d::Zero());
        spin_mean_sq_sum.resize(n_sublattices, Eigen::Vector3d::Zero());
        
        // Dimer correlation storage: [cell_disp][bond_type_mu][bond_type_nu][dimer_comp]
        // = [n_cell_displacements * n_bond_types * n_bond_types * 3]
        size_t dimer_corr_size = n_cell_displacements * n_bond_types * n_bond_types * n_dimer_components;
        dimer_corr_sum.resize(dimer_corr_size, 0.0);
        dimer_corr_sq_sum.resize(dimer_corr_size, 0.0);
        
        // Dimer means: [n_bond_types * n_dimer_components]
        dimer_mean_sum.resize(n_bond_types * n_dimer_components, 0.0);
        dimer_mean_sq_sum.resize(n_bond_types * n_dimer_components, 0.0);
        
        // Build cell displacement vectors and indices
        cell_displacement_vectors.resize(n_cell_displacements);
        cell_displacement_indices.resize(n_cell_displacements);
        
        for (size_t dn1 = 0; dn1 < d1; ++dn1) {
            for (size_t dn2 = 0; dn2 < d2; ++dn2) {
                for (size_t dn3 = 0; dn3 < d3; ++dn3) {
                    size_t idx = cell_displacement_index(dn1, dn2, dn3);
                    cell_displacement_vectors[idx] = 
                        double(dn1) * lattice_vectors[0] +
                        double(dn2) * lattice_vectors[1] +
                        double(dn3) * lattice_vectors[2];
                    cell_displacement_indices[idx] = {int(dn1), int(dn2), int(dn3)};
                }
            }
        }
        
        // Store sublattice positions for computing full displacement vectors
        sublattice_positions_ = sublattice_positions;
        
        n_samples = 0;
        initialized = true;
    }
    
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
    Eigen::Vector3d bond_center(size_t bond_type) const {
        auto [sub_i, sub_j] = bond_type_to_sublattices(bond_type);
        return 0.5 * (sublattice_positions_[sub_i] + sublattice_positions_[sub_j]);
    }
    
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
     * @param spins         Current spin configuration [n_sites]
     * @param site_to_sub   Mapping from site index to sublattice index
     * @param site_to_cell  Mapping from site index to (n1, n2, n3) cell indices
     */
    void accumulate_spin_correlations(
        const vector<Eigen::VectorXd>& spins,
        const function<size_t(size_t)>& site_to_sublattice,
        const function<array<size_t, 3>(size_t)>& site_to_cell) 
    {
        if (!initialized) {
            throw std::runtime_error("RealSpaceCorrelationAccumulator not initialized");
        }
        
        size_t n_cells = dim1 * dim2 * dim3;
        
        // Compute sublattice magnetizations for this sample
        vector<Eigen::Vector3d> M_sub(n_sublattices, Eigen::Vector3d::Zero());
        for (size_t site = 0; site < n_sites; ++site) {
            size_t sub = site_to_sublattice(site);
            M_sub[sub] += spins[site].head<3>();
        }
        for (size_t sub = 0; sub < n_sublattices; ++sub) {
            M_sub[sub] /= double(n_cells);
            spin_mean_sum[sub] += M_sub[sub];
            spin_mean_sq_sum[sub] += M_sub[sub].cwiseProduct(M_sub[sub]);
        }
        
        // Accumulate correlations for each (cell_displacement, sublattice_pair, spin_component)
        // Loop over cell displacements
        for (size_t dn1 = 0; dn1 < dim1; ++dn1) {
            for (size_t dn2 = 0; dn2 < dim2; ++dn2) {
                for (size_t dn3 = 0; dn3 < dim3; ++dn3) {
                    size_t cell_disp_idx = cell_displacement_index(dn1, dn2, dn3);
                    
                    // Loop over sublattice pairs with symmetry (sub_i ≤ sub_j)
                    for (size_t sub_i = 0; sub_i < n_sublattices; ++sub_i) {
                        for (size_t sub_j = sub_i; sub_j < n_sublattices; ++sub_j) {
                            size_t sub_pair_idx = sublattice_pair_index(sub_i, sub_j);
                            
                            // Accumulate 6 symmetric spin components
                            array<double, 6> corr_components = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                            
                            // Average over all unit cells
                            for (size_t n1 = 0; n1 < dim1; ++n1) {
                                for (size_t n2 = 0; n2 < dim2; ++n2) {
                                    for (size_t n3 = 0; n3 < dim3; ++n3) {
                                        // Site i at (n1, n2, n3, sub_i)
                                        size_t site_i = ((n1 * dim2 + n2) * dim3 + n3) * n_sublattices + sub_i;
                                        
                                        // Site j at ((n1+dn1)%dim1, (n2+dn2)%dim2, (n3+dn3)%dim3, sub_j)
                                        size_t m1 = (n1 + dn1) % dim1;
                                        size_t m2 = (n2 + dn2) % dim2;
                                        size_t m3 = (n3 + dn3) % dim3;
                                        size_t site_j = ((m1 * dim2 + m2) * dim3 + m3) * n_sublattices + sub_j;
                                        
                                        Eigen::Vector3d Si = spins[site_i].head<3>();
                                        Eigen::Vector3d Sj = spins[site_j].head<3>();
                                        
                                        // Symmetric components: xx, xy, xz, yy, yz, zz
                                        // For off-diagonal: symmetrize (S_i^α S_j^β + S_i^β S_j^α)/2
                                        corr_components[0] += Si[0] * Sj[0];  // xx
                                        corr_components[1] += 0.5 * (Si[0] * Sj[1] + Si[1] * Sj[0]);  // xy
                                        corr_components[2] += 0.5 * (Si[0] * Sj[2] + Si[2] * Sj[0]);  // xz
                                        corr_components[3] += Si[1] * Sj[1];  // yy
                                        corr_components[4] += 0.5 * (Si[1] * Sj[2] + Si[2] * Sj[1]);  // yz
                                        corr_components[5] += Si[2] * Sj[2];  // zz
                                    }
                                }
                            }
                            
                            // Normalize by number of cells
                            double inv_n_cells = 1.0 / double(n_cells);
                            for (size_t c = 0; c < 6; ++c) {
                                corr_components[c] *= inv_n_cells;
                                size_t idx = spin_corr_index(cell_disp_idx, sub_pair_idx, c);
                                spin_corr_sum[idx] += corr_components[c];
                                spin_corr_sq_sum[idx] += corr_components[c] * corr_components[c];
                            }
                        }
                    }
                }
            }
        }
        
        n_samples++;
    }
    
    /**
     * Accumulate dimer-dimer correlations
     * 
     * Dimer operator: D^α_b = S_i^α S_j^α for α ∈ {x, y, z}
     * Correlator: <D^α_μ(0) D^α_ν(ΔR)> for each component α
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
    {
        if (!initialized) {
            throw std::runtime_error("RealSpaceCorrelationAccumulator not initialized");
        }
        
        size_t n_bonds = bonds.size();
        size_t n_cells = dim1 * dim2 * dim3;
        
        // Compute all dimer operators D^α_b = S_i^α S_j^α for each component
        // Shape: [n_bonds][3]
        vector<array<double, 3>> D(n_bonds);
        for (size_t b = 0; b < n_bonds; ++b) {
            size_t i = bonds[b][0];
            size_t j = bonds[b][1];
            Eigen::Vector3d Si = spins[i].head<3>();
            Eigen::Vector3d Sj = spins[j].head<3>();
            D[b][0] = Si[0] * Sj[0];  // D^x
            D[b][1] = Si[1] * Sj[1];  // D^y
            D[b][2] = Si[2] * Sj[2];  // D^z
        }
        
        // Compute mean dimer per bond type and component
        // Shape: [n_bond_types][3]
        vector<array<double, 3>> D_mean(n_bond_types, {0.0, 0.0, 0.0});
        vector<size_t> D_count(n_bond_types, 0);
        for (size_t b = 0; b < n_bonds; ++b) {
            size_t type = bond_types[b];
            for (size_t c = 0; c < 3; ++c) {
                D_mean[type][c] += D[b][c];
            }
            D_count[type]++;
        }
        for (size_t t = 0; t < n_bond_types; ++t) {
            if (D_count[t] > 0) {
                for (size_t c = 0; c < 3; ++c) {
                    D_mean[t][c] /= double(D_count[t]);
                    size_t idx = dimer_mean_index(t, c);
                    dimer_mean_sum[idx] += D_mean[t][c];
                    dimer_mean_sq_sum[idx] += D_mean[t][c] * D_mean[t][c];
                }
            }
        }
        
        // Group bonds by cell and type for efficient pairing
        // Structure: bonds_by_cell_type[cell_idx][type] = list of bond indices
        vector<vector<vector<size_t>>> bonds_by_cell_type(
            n_cells, vector<vector<size_t>>(n_bond_types));
        
        for (size_t b = 0; b < n_bonds; ++b) {
            auto& [n1, n2, n3] = bond_cells[b];
            size_t cell_idx = (n1 * dim2 + n2) * dim3 + n3;
            size_t type = bond_types[b];
            bonds_by_cell_type[cell_idx][type].push_back(b);
        }
        
        // Accumulate dimer-dimer correlations for each cell offset, bond type pair, and component
        for (size_t dn1 = 0; dn1 < dim1; ++dn1) {
            for (size_t dn2 = 0; dn2 < dim2; ++dn2) {
                for (size_t dn3 = 0; dn3 < dim3; ++dn3) {
                    size_t cell_disp_idx = cell_displacement_index(dn1, dn2, dn3);
                    
                    for (size_t type_mu = 0; type_mu < n_bond_types; ++type_mu) {
                        for (size_t type_nu = 0; type_nu < n_bond_types; ++type_nu) {
                            // Correlations for each dimer component
                            array<double, 3> corr = {0.0, 0.0, 0.0};
                            size_t count = 0;
                            
                            // Sum over all cell pairs with this offset
                            for (size_t n1 = 0; n1 < dim1; ++n1) {
                                for (size_t n2 = 0; n2 < dim2; ++n2) {
                                    for (size_t n3 = 0; n3 < dim3; ++n3) {
                                        size_t cell_i = (n1 * dim2 + n2) * dim3 + n3;
                                        size_t m1 = (n1 + dn1) % dim1;
                                        size_t m2 = (n2 + dn2) % dim2;
                                        size_t m3 = (n3 + dn3) % dim3;
                                        size_t cell_j = (m1 * dim2 + m2) * dim3 + m3;
                                        
                                        // All bond pairs of these types
                                        for (size_t bi : bonds_by_cell_type[cell_i][type_mu]) {
                                            for (size_t bj : bonds_by_cell_type[cell_j][type_nu]) {
                                                for (size_t c = 0; c < 3; ++c) {
                                                    corr[c] += D[bi][c] * D[bj][c];
                                                }
                                                count++;
                                            }
                                        }
                                    }
                                }
                            }
                            
                            if (count > 0) {
                                double inv_count = 1.0 / double(count);
                                for (size_t c = 0; c < 3; ++c) {
                                    corr[c] *= inv_count;
                                    size_t idx = dimer_corr_index(cell_disp_idx, type_mu, type_nu, c);
                                    dimer_corr_sum[idx] += corr[c];
                                    dimer_corr_sq_sum[idx] += corr[c] * corr[c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Compute spin structure factor S^{αβ}(q) at arbitrary q-point
     * Sublattice-resolved: S(q) = Σ_{ΔR,s,s'} C_{ss'}(ΔR) exp(-i q·(ΔR + r_s' - r_s))
     * 
     * @param q           Wavevector in Cartesian coordinates
     * @param connected   If true, subtract <S_i><S_j> (use for susceptibility)
     * @return 3x3 matrix S^{αβ}(q) (symmetric form)
     */
    Eigen::Matrix3d compute_Sq(const Eigen::Vector3d& q, bool connected = true) const {
        if (n_samples == 0) {
            return Eigen::Matrix3d::Zero();
        }
        
        // Use 6 symmetric components, then expand to 3x3
        array<double, 6> Sq_sym = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        
        for (size_t cell_disp_idx = 0; cell_disp_idx < n_cell_displacements; ++cell_disp_idx) {
            for (size_t sub_i = 0; sub_i < n_sublattices; ++sub_i) {
                for (size_t sub_j = sub_i; sub_j < n_sublattices; ++sub_j) {
                    size_t sub_pair_idx = sublattice_pair_index(sub_i, sub_j);
                    
                    // Compute full displacement including sublattice positions
                    Eigen::Vector3d dr = cell_displacement_vectors[cell_disp_idx] 
                                        + sublattice_positions_[sub_j] - sublattice_positions_[sub_i];
                    double phase = q.dot(dr);
                    double cos_phase = std::cos(phase);
                    
                    // Multiplicity: 1 for diagonal (i=j), 2 for off-diagonal (symmetry)
                    double mult = (sub_i == sub_j) ? 1.0 : 2.0;
                    
                    for (size_t c = 0; c < 6; ++c) {
                        size_t idx = spin_corr_index(cell_disp_idx, sub_pair_idx, c);
                        double C = spin_corr_sum[idx] / double(n_samples);
                        
                        // Subtract disconnected part if requested
                        if (connected) {
                            // For symmetric components, need to match symmetrization
                            // Component c corresponds to (α,β) pair
                            // 0=xx, 1=xy, 2=xz, 3=yy, 4=yz, 5=zz
                            Eigen::Vector3d mi = spin_mean_sum[sub_i] / double(n_samples);
                            Eigen::Vector3d mj = spin_mean_sum[sub_j] / double(n_samples);
                            
                            double disc = 0.0;
                            switch(c) {
                                case 0: disc = mi[0] * mj[0]; break;  // xx
                                case 1: disc = 0.5 * (mi[0] * mj[1] + mi[1] * mj[0]); break;  // xy
                                case 2: disc = 0.5 * (mi[0] * mj[2] + mi[2] * mj[0]); break;  // xz
                                case 3: disc = mi[1] * mj[1]; break;  // yy
                                case 4: disc = 0.5 * (mi[1] * mj[2] + mi[2] * mj[1]); break;  // yz
                                case 5: disc = mi[2] * mj[2]; break;  // zz
                            }
                            C -= disc;
                        }
                        
                        Sq_sym[c] += mult * C * cos_phase;
                    }
                }
            }
        }
        
        // Expand to 3x3 symmetric matrix
        Eigen::Matrix3d Sq;
        Sq(0,0) = Sq_sym[0];  // xx
        Sq(0,1) = Sq_sym[1];  Sq(1,0) = Sq_sym[1];  // xy
        Sq(0,2) = Sq_sym[2];  Sq(2,0) = Sq_sym[2];  // xz
        Sq(1,1) = Sq_sym[3];  // yy
        Sq(1,2) = Sq_sym[4];  Sq(2,1) = Sq_sym[4];  // yz
        Sq(2,2) = Sq_sym[5];  // zz
        
        return Sq;
    }
    
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
    array<Eigen::MatrixXd, 3> compute_Sq_dimer(const Eigen::Vector3d& q, bool connected = true) const {
        array<Eigen::MatrixXd, 3> Sq_D;
        for (size_t c = 0; c < 3; ++c) {
            Sq_D[c] = Eigen::MatrixXd::Zero(n_bond_types, n_bond_types);
        }
        
        if (n_samples == 0) {
            return Sq_D;
        }
        
        // Precompute bond centers for each bond type
        vector<Eigen::Vector3d> bond_centers(n_bond_types);
        for (size_t mu = 0; mu < n_bond_types; ++mu) {
            bond_centers[mu] = bond_center(mu);
        }
        
        for (size_t cell_disp_idx = 0; cell_disp_idx < n_cell_displacements; ++cell_disp_idx) {
            for (size_t mu = 0; mu < n_bond_types; ++mu) {
                for (size_t nu = 0; nu < n_bond_types; ++nu) {
                    // Displacement between bond centers: ΔR + center(ν) - center(μ)
                    Eigen::Vector3d dr = cell_displacement_vectors[cell_disp_idx] 
                                        + bond_centers[nu] - bond_centers[mu];
                    double phase = q.dot(dr);
                    double cos_phase = std::cos(phase);
                    
                    for (size_t c = 0; c < 3; ++c) {
                        size_t idx = dimer_corr_index(cell_disp_idx, mu, nu, c);
                        double chi = dimer_corr_sum[idx] / double(n_samples);
                        
                        if (connected) {
                            double D_mu = dimer_mean_sum[dimer_mean_index(mu, c)] / double(n_samples);
                            double D_nu = dimer_mean_sum[dimer_mean_index(nu, c)] / double(n_samples);
                            chi -= D_mu * D_nu;
                        }
                        
                        Sq_D[c](mu, nu) += chi * cos_phase;
                    }
                }
            }
        }
        
        return Sq_D;
    }
    
    /**
     * Compute total dimer structure factor (sum over components)
     * S_D^{μν}(q) = Σ_α S_D^{αμν}(q) = Σ_α Σ_ΔR <D^α_μ(0) D^α_ν(ΔR)> exp(-i q·ΔR)
     */
    Eigen::MatrixXd compute_Sq_dimer_total(const Eigen::Vector3d& q, bool connected = true) const {
        auto Sq_components = compute_Sq_dimer(q, connected);
        Eigen::MatrixXd Sq_total = Eigen::MatrixXd::Zero(n_bond_types, n_bond_types);
        for (size_t c = 0; c < 3; ++c) {
            Sq_total += Sq_components[c];
        }
        return Sq_total;
    }
    
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
    void merge(const RealSpaceCorrelationAccumulator& other) {
        if (!initialized || !other.initialized) return;
        if (n_cell_displacements != other.n_cell_displacements) {
            throw std::runtime_error("Cannot merge accumulators with different geometry");
        }
        
        for (size_t i = 0; i < spin_corr_sum.size(); ++i) {
            spin_corr_sum[i] += other.spin_corr_sum[i];
            spin_corr_sq_sum[i] += other.spin_corr_sq_sum[i];
        }
        for (size_t i = 0; i < n_sublattices; ++i) {
            spin_mean_sum[i] += other.spin_mean_sum[i];
            spin_mean_sq_sum[i] += other.spin_mean_sq_sum[i];
        }
        for (size_t i = 0; i < dimer_corr_sum.size(); ++i) {
            dimer_corr_sum[i] += other.dimer_corr_sum[i];
            dimer_corr_sq_sum[i] += other.dimer_corr_sq_sum[i];
        }
        for (size_t i = 0; i < n_bond_types; ++i) {
            dimer_mean_sum[i] += other.dimer_mean_sum[i];
            dimer_mean_sq_sum[i] += other.dimer_mean_sq_sum[i];
        }
        n_samples += other.n_samples;
    }
    
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
    void save_hdf5(const string& filename, const string& group_name = "/correlations") const {
        try {
            H5::H5File file(filename, H5F_ACC_TRUNC);
            H5::Group group = file.createGroup(group_name);
            
            // Helper lambda for writing scalar attributes
            auto write_attr = [](H5::Group& grp, const string& name, hsize_t val) {
                H5::DataSpace scalar_space(H5S_SCALAR);
                H5::Attribute attr = grp.createAttribute(name, H5::PredType::NATIVE_HSIZE, scalar_space);
                attr.write(H5::PredType::NATIVE_HSIZE, &val);
            };
            
            // Helper lambda for writing 1D double vectors
            auto write_vec = [](H5::Group& grp, const string& name, const vector<double>& vec) {
                hsize_t dims[1] = {vec.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
                dataset.write(vec.data(), H5::PredType::NATIVE_DOUBLE);
            };
            
            // Helper lambda for writing 1D int vectors
            auto write_int_vec = [](H5::Group& grp, const string& name, const vector<int>& vec) {
                hsize_t dims[1] = {vec.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = grp.createDataSet(name, H5::PredType::NATIVE_INT, dataspace);
                dataset.write(vec.data(), H5::PredType::NATIVE_INT);
            };
            
            // Save metadata
            write_attr(group, "n_samples", n_samples);
            write_attr(group, "dim1", dim1);
            write_attr(group, "dim2", dim2);
            write_attr(group, "dim3", dim3);
            write_attr(group, "n_sublattices", n_sublattices);
            write_attr(group, "n_bond_types", n_bond_types);
            write_attr(group, "n_cell_displacements", n_cell_displacements);
            write_attr(group, "n_sublattice_pairs", n_sublattice_pairs);
            write_attr(group, "n_spin_components", n_spin_components);
            write_attr(group, "n_dimer_components", n_dimer_components);
            
            // ========== CELL DISPLACEMENT METADATA ==========
            // Save cell displacement indices: (dn1, dn2, dn3) for each cell offset
            vector<int> cell_disp_indices_flat(n_cell_displacements * 3);
            for (size_t i = 0; i < n_cell_displacements; ++i) {
                cell_disp_indices_flat[i * 3 + 0] = cell_displacement_indices[i][0];
                cell_disp_indices_flat[i * 3 + 1] = cell_displacement_indices[i][1];
                cell_disp_indices_flat[i * 3 + 2] = cell_displacement_indices[i][2];
            }
            write_int_vec(group, "cell_displacement_indices", cell_disp_indices_flat);
            
            // Save cell displacement vectors
            vector<double> cell_disp_flat(n_cell_displacements * 3);
            for (size_t i = 0; i < n_cell_displacements; ++i) {
                cell_disp_flat[i * 3 + 0] = cell_displacement_vectors[i](0);
                cell_disp_flat[i * 3 + 1] = cell_displacement_vectors[i](1);
                cell_disp_flat[i * 3 + 2] = cell_displacement_vectors[i](2);
            }
            write_vec(group, "cell_displacement_vectors", cell_disp_flat);
            
            // ========== SUBLATTICE PAIR MAPPING ==========
            // For each sublattice pair index (also = bond type index), store (sub_i, sub_j)
            // This maps bond type → which sublattices the bond connects
            vector<int> sublattice_pairs(n_sublattice_pairs * 2);
            size_t pair_idx = 0;
            for (size_t sub_i = 0; sub_i < n_sublattices; ++sub_i) {
                for (size_t sub_j = sub_i; sub_j < n_sublattices; ++sub_j) {
                    sublattice_pairs[pair_idx * 2 + 0] = static_cast<int>(sub_i);
                    sublattice_pairs[pair_idx * 2 + 1] = static_cast<int>(sub_j);
                    pair_idx++;
                }
            }
            write_int_vec(group, "sublattice_pairs", sublattice_pairs);
            // Note: bond_types array is same as sublattice_pairs since bond type = sublattice pair
            write_int_vec(group, "bond_types", sublattice_pairs);
            
            // Save sublattice positions
            vector<double> sub_pos_flat(n_sublattices * 3);
            for (size_t i = 0; i < n_sublattices; ++i) {
                sub_pos_flat[i * 3 + 0] = sublattice_positions_[i](0);
                sub_pos_flat[i * 3 + 1] = sublattice_positions_[i](1);
                sub_pos_flat[i * 3 + 2] = sublattice_positions_[i](2);
            }
            write_vec(group, "sublattice_positions", sub_pos_flat);
            
            // Save bond centers: center(μ) = (r_{sub_i} + r_{sub_j})/2
            // Shape: [n_bond_types × 3]
            vector<double> bond_centers_flat(n_bond_types * 3);
            for (size_t mu = 0; mu < n_bond_types; ++mu) {
                Eigen::Vector3d center = bond_center(mu);
                bond_centers_flat[mu * 3 + 0] = center(0);
                bond_centers_flat[mu * 3 + 1] = center(1);
                bond_centers_flat[mu * 3 + 2] = center(2);
            }
            write_vec(group, "bond_centers", bond_centers_flat);
            
            // ========== SPIN CORRELATIONS ==========
            // Shape: [n_cell_displacements][n_sublattice_pairs][6]
            // Component order: xx=0, xy=1, xz=2, yy=3, yz=4, zz=5
            write_vec(group, "spin_corr_sum", spin_corr_sum);
            
            // Save spin means per sublattice
            vector<double> spin_mean_flat(n_sublattices * 3);
            for (size_t i = 0; i < n_sublattices; ++i) {
                spin_mean_flat[i * 3 + 0] = spin_mean_sum[i](0);
                spin_mean_flat[i * 3 + 1] = spin_mean_sum[i](1);
                spin_mean_flat[i * 3 + 2] = spin_mean_sum[i](2);
            }
            write_vec(group, "spin_mean_sum", spin_mean_flat);
            
            // ========== DIMER CORRELATIONS ==========
            // Shape: [n_cell_displacements][n_bond_types][n_bond_types][3]
            // Component order: x=0, y=1, z=2
            write_vec(group, "dimer_corr_sum", dimer_corr_sum);
            
            // Dimer means: [n_bond_types][3]
            write_vec(group, "dimer_mean_sum", dimer_mean_sum);
            
            file.close();
        } catch (const H5::Exception& e) {
            std::cerr << "HDF5 error saving correlations: " << e.getDetailMsg() << std::endl;
        }
    }
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
    {
        ofstream file(filename);
        if (!file) {
            std::cerr << "Error: Cannot open file " << filename << endl;
            return;
        }
        
        file << "# Spin structure factor S(q) from real-space correlation accumulator\n";
        file << "# n_samples = " << n_samples << "\n";
        file << "# Lattice: " << dim1 << " x " << dim2 << " x " << dim3 
             << " x " << n_sublattices << " sublattices\n";
        file << "# Connected correlator: " << (connected ? "yes" : "no") << "\n";
        file << "# Columns: q1(rlu) q2(rlu) q3(rlu) qx qy qz S_total S_xx S_yy S_zz S_xy S_xz S_yz\n";
        file << std::scientific << std::setprecision(8);
        
        for (size_t i1 = 0; i1 < n_q1; ++i1) {
            double q1 = (n_q1 > 1) ? q1_range.first + (q1_range.second - q1_range.first) * i1 / (n_q1 - 1)
                                   : 0.5 * (q1_range.first + q1_range.second);
            for (size_t i2 = 0; i2 < n_q2; ++i2) {
                double q2 = (n_q2 > 1) ? q2_range.first + (q2_range.second - q2_range.first) * i2 / (n_q2 - 1)
                                       : 0.5 * (q2_range.first + q2_range.second);
                for (size_t i3 = 0; i3 < n_q3; ++i3) {
                    double q3 = (n_q3 > 1) ? q3_range.first + (q3_range.second - q3_range.first) * i3 / (n_q3 - 1)
                                           : 0.5 * (q3_range.first + q3_range.second);
                    
                    // Convert to Cartesian q-vector
                    Eigen::Vector3d q = q1 * b1 + q2 * b2 + q3 * b3;
                    
                    // Compute S(q)
                    Eigen::Matrix3d Sq = compute_Sq(q, connected);
                    double S_total = Sq.trace();
                    
                    file << q1 << " " << q2 << " " << q3 << " "
                         << q(0) << " " << q(1) << " " << q(2) << " "
                         << S_total << " "
                         << Sq(0,0) << " " << Sq(1,1) << " " << Sq(2,2) << " "
                         << Sq(0,1) << " " << Sq(0,2) << " " << Sq(1,2) << "\n";
                }
            }
        }
        
        file.close();
    }
    
    /**
     * MPI reduce: gather accumulators from all ranks and merge
     * After this call, rank 0 has the combined accumulator
     * 
     * @param comm  MPI communicator
     * @return Combined accumulator (valid on rank 0 only)
     */
    void mpi_reduce(MPI_Comm comm = MPI_COMM_WORLD) {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        
        if (size == 1) return;  // Nothing to reduce
        
        // Flatten spin means for MPI
        vector<double> spin_mean_flat(n_sublattices * 3);
        vector<double> spin_mean_sq_flat(n_sublattices * 3);
        for (size_t i = 0; i < n_sublattices; ++i) {
            for (size_t d = 0; d < 3; ++d) {
                spin_mean_flat[i * 3 + d] = spin_mean_sum[i](d);
                spin_mean_sq_flat[i * 3 + d] = spin_mean_sq_sum[i](d);
            }
        }
        
        // Reduce to rank 0
        if (rank == 0) {
            vector<double> recv_spin(spin_corr_sum.size());
            vector<double> recv_spin_sq(spin_corr_sq_sum.size());
            vector<double> recv_mean(n_sublattices * 3);
            vector<double> recv_mean_sq(n_sublattices * 3);
            vector<double> recv_dimer(dimer_corr_sum.size());
            vector<double> recv_dimer_sq(dimer_corr_sq_sum.size());
            vector<double> recv_dimer_mean(dimer_mean_sum.size());
            vector<double> recv_dimer_mean_sq(dimer_mean_sq_sum.size());
            size_t recv_samples;
            
            for (int src = 1; src < size; ++src) {
                MPI_Recv(&recv_samples, 1, MPI_UNSIGNED_LONG, src, 0, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_spin.data(), spin_corr_sum.size(), MPI_DOUBLE, src, 1, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_spin_sq.data(), spin_corr_sq_sum.size(), MPI_DOUBLE, src, 2, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_mean.data(), n_sublattices * 3, MPI_DOUBLE, src, 3, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_mean_sq.data(), n_sublattices * 3, MPI_DOUBLE, src, 4, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_dimer.data(), dimer_corr_sum.size(), MPI_DOUBLE, src, 5, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_dimer_sq.data(), dimer_corr_sq_sum.size(), MPI_DOUBLE, src, 6, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_dimer_mean.data(), dimer_mean_sum.size(), MPI_DOUBLE, src, 7, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_dimer_mean_sq.data(), dimer_mean_sq_sum.size(), MPI_DOUBLE, src, 8, comm, MPI_STATUS_IGNORE);
                
                // Add to local
                n_samples += recv_samples;
                for (size_t i = 0; i < spin_corr_sum.size(); ++i) {
                    spin_corr_sum[i] += recv_spin[i];
                    spin_corr_sq_sum[i] += recv_spin_sq[i];
                }
                for (size_t i = 0; i < n_sublattices * 3; ++i) {
                    spin_mean_flat[i] += recv_mean[i];
                    spin_mean_sq_flat[i] += recv_mean_sq[i];
                }
                for (size_t i = 0; i < dimer_corr_sum.size(); ++i) {
                    dimer_corr_sum[i] += recv_dimer[i];
                    dimer_corr_sq_sum[i] += recv_dimer_sq[i];
                }
                for (size_t i = 0; i < dimer_mean_sum.size(); ++i) {
                    dimer_mean_sum[i] += recv_dimer_mean[i];
                    dimer_mean_sq_sum[i] += recv_dimer_mean_sq[i];
                }
            }
            
            // Unflatten spin means back
            for (size_t i = 0; i < n_sublattices; ++i) {
                for (size_t d = 0; d < 3; ++d) {
                    spin_mean_sum[i](d) = spin_mean_flat[i * 3 + d];
                    spin_mean_sq_sum[i](d) = spin_mean_sq_flat[i * 3 + d];
                }
            }
        } else {
            // Send to rank 0
            MPI_Send(&n_samples, 1, MPI_UNSIGNED_LONG, 0, 0, comm);
            MPI_Send(spin_corr_sum.data(), spin_corr_sum.size(), MPI_DOUBLE, 0, 1, comm);
            MPI_Send(spin_corr_sq_sum.data(), spin_corr_sq_sum.size(), MPI_DOUBLE, 0, 2, comm);
            MPI_Send(spin_mean_flat.data(), n_sublattices * 3, MPI_DOUBLE, 0, 3, comm);
            MPI_Send(spin_mean_sq_flat.data(), n_sublattices * 3, MPI_DOUBLE, 0, 4, comm);
            MPI_Send(dimer_corr_sum.data(), dimer_corr_sum.size(), MPI_DOUBLE, 0, 5, comm);
            MPI_Send(dimer_corr_sq_sum.data(), dimer_corr_sq_sum.size(), MPI_DOUBLE, 0, 6, comm);
            MPI_Send(dimer_mean_sum.data(), dimer_mean_sum.size(), MPI_DOUBLE, 0, 7, comm);
            MPI_Send(dimer_mean_sq_sum.data(), dimer_mean_sq_sum.size(), MPI_DOUBLE, 0, 8, comm);
        }
    }
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

    size_t num_bi;  // Number of bilinear neighbors per site
    size_t num_tri; // Number of trilinear interactions per site

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
        cout << "Lattice initialization complete!" << endl;
        cout << "Max bilinear interactions per site: " << num_bi << endl;
        cout << "Max trilinear interactions per site: " << num_tri << endl;
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
        
        if (spin_dim == 3) {
            // Efficient sphere sampling for 3D
            double z = random_double_lehman(-1.0, 1.0);
            double phi = random_double_lehman(0.0, 2.0 * M_PI);
            double r = std::sqrt(1.0 - z * z);
            spin(0) = r * std::cos(phi);
            spin(1) = r * std::sin(phi);
            spin(2) = z;
        } else {
            // General n-sphere sampling (Marsaglia method)
            for (size_t i = 0; i < spin_dim; ++i) {
                spin(i) = random_double_lehman(-1.0, 1.0);
            }
            double norm = spin.norm();
            while (norm < 1e-10) {
                for (size_t i = 0; i < spin_dim; ++i) {
                    spin(i) = random_double_lehman(-1.0, 1.0);
                }
                norm = spin.norm();
            }
            spin /= norm;
        }
        
        return spin * spin_l;
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
        
        // Bilinear: H += J*S_j
        size_t n_bi = bilinear_partners[site_index].size();
        for (size_t n = 0; n < n_bi; ++n) {
            size_t partner = bilinear_partners[site_index][n];
            const double* S_partner = &state_flat[partner * spin_dim];
            
            // Apply twist if needed (optimized for 3D)
            double S_twisted[8];  // Stack buffer
            const double* S_use = S_partner;
            
            const auto& wrap = bilinear_wrap_dir[site_index][n];
            if (spin_dim == 3 && (wrap[0] != 0 || wrap[1] != 0 || wrap[2] != 0)) {
                for (size_t d = 0; d < 3; ++d) S_twisted[d] = S_partner[d];
                for (size_t dim = 0; dim < 3; ++dim) {
                    if (wrap[dim] != 0) {
                        double temp[3];
                        // For positive wrap, apply twist_matrices[dim]
                        // For negative wrap, apply transpose (inverse for rotations)
                        if (wrap[dim] > 0) {
                            for (size_t d = 0; d < 3; ++d) {
                                temp[d] = 0.0;
                                for (size_t d2 = 0; d2 < 3; ++d2) {
                                    temp[d] += twist_matrices[dim](d, d2) * S_twisted[d2];
                                }
                            }
                        } else {
                            // Apply transpose: R^T[d, d2] = R[d2, d]
                            for (size_t d = 0; d < 3; ++d) {
                                temp[d] = 0.0;
                                for (size_t d2 = 0; d2 < 3; ++d2) {
                                    temp[d] += twist_matrices[dim](d2, d) * S_twisted[d2];
                                }
                            }
                        }
                        for (size_t d = 0; d < 3; ++d) S_twisted[d] = temp[d];
                    }
                }
                S_use = S_twisted;
            }
            
            // H += J * S_partner
            for (size_t d = 0; d < spin_dim; ++d) {
                for (size_t d2 = 0; d2 < spin_dim; ++d2) {
                    H_out[d] += bilinear_interaction[site_index][n](d, d2) * S_use[d2];
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
                                                   size_t base_interval = 10) {
        AutocorrelationResult result;
        
        if (energies.size() < 10) {
            result.tau_int = 1.0;
            result.sampling_interval = base_interval;
            return result;
        }
        
        // Calculate mean and variance
        double mean = std::accumulate(energies.begin(), energies.end(), 0.0) / energies.size();
        double variance = 0.0;
        for (double e : energies) {
            variance += (e - mean) * (e - mean);
        }
        variance /= energies.size();
        
        if (variance < 1e-20) {
            result.tau_int = 1.0;
            result.sampling_interval = base_interval;
            return result;
        }
        
        // Compute autocorrelation function
        size_t max_lag = std::min(energies.size() / 4, size_t(1000));
        result.correlation_function.resize(max_lag);
        
        for (size_t lag = 0; lag < max_lag; ++lag) {
            double corr = 0.0;
            size_t count = energies.size() - lag;
            for (size_t i = 0; i < count; ++i) {
                corr += (energies[i] - mean) * (energies[i + lag] - mean);
            }
            result.correlation_function[lag] = corr / (count * variance);
        }
        
        // Calculate integrated autocorrelation time
        result.tau_int = 0.5;
        for (size_t lag = 1; lag < max_lag; ++lag) {
            if (result.correlation_function[lag] < 0.1) break; // Stop when decorrelated
            result.tau_int += result.correlation_function[lag];
        }
        
        // Determine sampling interval
        result.sampling_interval = std::max(size_t(2 * result.tau_int * base_interval), size_t(100));
        
        return result;
    }

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
        
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            M_sub[atom] = SpinVector::Zero(spin_dim);
        }
        
        // Sum over all unit cells for each sublattice (in local frame)
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    for (size_t atom = 0; atom < N_atoms; ++atom) {
                        size_t site_idx = flatten_index(i, j, k, atom);
                        M_sub[atom] += spins[site_idx];
                    }
                }
            }
        }
        
        // Normalize by number of unit cells
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            M_sub[atom] /= double(n_cells);
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
        double T) const {
        
        ThermodynamicObservables obs;
        obs.temperature = T;
        size_t n_samples = energies.size();
        
        if (n_samples == 0) return obs;
        
        // 1. Energy per site with binning analysis
        vector<double> energy_per_site(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            energy_per_site[i] = energies[i] / double(lattice_size);
        }
        BinningResult E_result = binning_analysis(energy_per_site);
        obs.energy.value = E_result.mean;
        obs.energy.error = E_result.error;
        
        // 1b. Total magnetization per site with binning analysis
        //     M = (1/N) Σ_α Σ_i∈α S_i = (1/N_atoms) Σ_α M_α
        if (!sublattice_mags.empty() && !sublattice_mags[0].empty()) {
            size_t sdim = sublattice_mags[0][0].size();
            size_t n_sublattices = sublattice_mags[0].size();
            obs.magnetization = VectorObservable(sdim);
            
            for (size_t d = 0; d < sdim; ++d) {
                vector<double> M_total_d(n_samples);
                for (size_t i = 0; i < n_samples; ++i) {
                    double M_sum = 0.0;
                    for (size_t alpha = 0; alpha < n_sublattices; ++alpha) {
                        M_sum += sublattice_mags[i][alpha](d);
                    }
                    M_total_d[i] = M_sum / double(n_sublattices);
                }
                BinningResult M_result = binning_analysis(M_total_d);
                obs.magnetization.values[d] = M_result.mean;
                obs.magnetization.errors[d] = M_result.error;
            }
        }
        
        // 2. Specific heat per site: c_V = Var(E) / (T² N²) = Var(E/N) / T²
        //    Since E is extensive (E ~ N), Var(E) ~ N², so c_V ~ O(1)
        //    Error propagation via jackknife on binned data
        {
            double N2 = double(lattice_size) * double(lattice_size);
            
            // Handle edge case: need at least 2 samples for variance
            if (n_samples < 2) {
                obs.specific_heat.value = 0.0;
                obs.specific_heat.error = 0.0;
            } else {
                // Compute mean first for numerical stability (two-pass algorithm)
                double E_mean = 0.0;
                for (size_t i = 0; i < n_samples; ++i) {
                    E_mean += energies[i];
                }
                E_mean /= n_samples;
                
                // Compute variance using shifted data for numerical stability
                // Var(E) = <(E - E_mean)²> which avoids catastrophic cancellation
                double var_E = 0.0;
                for (size_t i = 0; i < n_samples; ++i) {
                    double delta = energies[i] - E_mean;
                    var_E += delta * delta;
                }
                var_E /= n_samples;  // Biased estimator (for heat capacity)
                
                // Ensure non-negative variance (numerical protection)
                var_E = std::max(0.0, var_E);
                obs.specific_heat.value = var_E / (T * T * N2);
                
                // Jackknife error estimation for specific heat
                // Use at most 100 jackknife blocks, at least 2
                size_t n_jack = std::min(n_samples, size_t(100));
                n_jack = std::max(n_jack, size_t(2));
                size_t block_size = std::max(size_t(1), n_samples / n_jack);
                // Recalculate n_jack based on actual block_size to handle remainders
                n_jack = (n_samples + block_size - 1) / block_size;
                
                vector<double> C_jack(n_jack);
                
                for (size_t j = 0; j < n_jack; ++j) {
                    // Leave out block j: indices [j*block_size, min((j+1)*block_size, n_samples))
                    size_t block_start = j * block_size;
                    size_t block_end = std::min((j + 1) * block_size, n_samples);
                    
                    // Compute jackknife mean (excluding block j)
                    double E_sum = 0.0;
                    size_t count = 0;
                    for (size_t i = 0; i < n_samples; ++i) {
                        if (i < block_start || i >= block_end) {
                            E_sum += energies[i];
                            ++count;
                        }
                    }
                    
                    if (count < 2) {
                        C_jack[j] = obs.specific_heat.value;  // Fallback
                        continue;
                    }
                    
                    double E_j = E_sum / count;
                    
                    // Compute jackknife variance (excluding block j)
                    double var_j = 0.0;
                    for (size_t i = 0; i < n_samples; ++i) {
                        if (i < block_start || i >= block_end) {
                            double delta = energies[i] - E_j;
                            var_j += delta * delta;
                        }
                    }
                    var_j /= count;
                    var_j = std::max(0.0, var_j);  // Numerical protection
                    
                    C_jack[j] = var_j / (T * T * N2);
                }
                
                // Compute jackknife error estimate
                double C_mean = 0.0;
                for (double c : C_jack) C_mean += c;
                C_mean /= n_jack;
                
                double C_var = 0.0;
                for (double c : C_jack) C_var += (c - C_mean) * (c - C_mean);
                C_var *= double(n_jack - 1) / double(n_jack);  // Jackknife variance factor
                obs.specific_heat.error = std::sqrt(std::max(0.0, C_var));
            }
        }
        
        // 3. Sublattice magnetizations with binning analysis
        if (!sublattice_mags.empty() && !sublattice_mags[0].empty()) {
            size_t n_sublattices = sublattice_mags[0].size();
            size_t sdim = sublattice_mags[0][0].size();
            
            obs.sublattice_magnetization.resize(n_sublattices);
            
            for (size_t alpha = 0; alpha < n_sublattices; ++alpha) {
                obs.sublattice_magnetization[alpha] = VectorObservable(sdim);
                
                // Extract time series for each component
                for (size_t d = 0; d < sdim; ++d) {
                    vector<double> M_alpha_d(n_samples);
                    for (size_t i = 0; i < n_samples; ++i) {
                        M_alpha_d[i] = sublattice_mags[i][alpha](d);
                    }
                    BinningResult M_result = binning_analysis(M_alpha_d);
                    obs.sublattice_magnetization[alpha].values[d] = M_result.mean;
                    obs.sublattice_magnetization[alpha].errors[d] = M_result.error;
                }
            }
            
            // 4. Cross term <E * S_α> - <E><S_α> for each sublattice
            obs.energy_sublattice_cross.resize(n_sublattices);
            
            for (size_t alpha = 0; alpha < n_sublattices; ++alpha) {
                obs.energy_sublattice_cross[alpha] = VectorObservable(sdim);
                
                for (size_t d = 0; d < sdim; ++d) {
                    // Compute <E * S_α,d>
                    vector<double> ES_alpha_d(n_samples);
                    for (size_t i = 0; i < n_samples; ++i) {
                        ES_alpha_d[i] = energies[i] * sublattice_mags[i][alpha](d);
                    }
                    
                    BinningResult ES_result = binning_analysis(ES_alpha_d);
                    
                    // Cross correlation = <ES> - <E><S>
                    double E_mean = obs.energy.value * double(lattice_size);
                    double S_mean = obs.sublattice_magnetization[alpha].values[d];
                    double cross_val = ES_result.mean - E_mean * S_mean;
                    
                    // Error propagation: δ(AB - CD) ≈ sqrt(δ(AB)² + (B δA)² + (A δB)²)
                    // Simplified: use jackknife for proper covariance handling
                    size_t n_jack = std::min(n_samples, size_t(100));
                    size_t block_size = n_samples / n_jack;
                    vector<double> cross_jack(n_jack);
                    
                    for (size_t j = 0; j < n_jack; ++j) {
                        double E_sum = 0.0, S_sum = 0.0, ES_sum = 0.0;
                        size_t count = 0;
                        for (size_t i = 0; i < n_samples; ++i) {
                            if (i / block_size != j) {
                                E_sum += energies[i];
                                S_sum += sublattice_mags[i][alpha](d);
                                ES_sum += energies[i] * sublattice_mags[i][alpha](d);
                                ++count;
                            }
                        }
                        double E_j = E_sum / count;
                        double S_j = S_sum / count;
                        double ES_j = ES_sum / count;
                        cross_jack[j] = ES_j - E_j * S_j;
                    }
                    
                    double cross_mean = 0.0;
                    for (double c : cross_jack) cross_mean += c;
                    cross_mean /= n_jack;
                    
                    double cross_var = 0.0;
                    for (double c : cross_jack) cross_var += (c - cross_mean) * (c - cross_mean);
                    cross_var *= double(n_jack - 1) / n_jack;
                    
                    obs.energy_sublattice_cross[alpha].values[d] = cross_val;
                    obs.energy_sublattice_cross[alpha].errors[d] = std::sqrt(cross_var);
                }
            }
        }
        
        return obs;
    }

    /**
     * Save comprehensive thermodynamic observables to files
     */
    void save_thermodynamic_observables(const string& out_dir,
                                         const ThermodynamicObservables& obs) const {
        ensure_directory_exists(out_dir);
        
        // Save main observables summary
        ofstream summary_file(out_dir + "/observables_summary.txt");
        summary_file << "# Thermodynamic Observables at T = " << obs.temperature << endl;
        summary_file << "# Format: observable mean error" << endl;
        summary_file << std::scientific << std::setprecision(12);
        
        summary_file << "energy_per_site " << obs.energy.value << " " << obs.energy.error << endl;
        summary_file << "specific_heat " << obs.specific_heat.value << " " << obs.specific_heat.error << endl;
        
        // Save total magnetization
        for (size_t d = 0; d < obs.magnetization.values.size(); ++d) {
            summary_file << "magnetization_" << d << " " 
                        << obs.magnetization.values[d] << " " 
                        << obs.magnetization.errors[d] << endl;
        }
        summary_file.close();
        
        // Save sublattice magnetizations
        ofstream sub_mag_file(out_dir + "/sublattice_magnetization.txt");
        sub_mag_file << "# Sublattice magnetizations <S_α>" << endl;
        sub_mag_file << "# Format: sublattice component mean error" << endl;
        sub_mag_file << std::scientific << std::setprecision(12);
        
        for (size_t alpha = 0; alpha < obs.sublattice_magnetization.size(); ++alpha) {
            const auto& M = obs.sublattice_magnetization[alpha];
            for (size_t d = 0; d < M.values.size(); ++d) {
                sub_mag_file << alpha << " " << d << " " 
                            << M.values[d] << " " << M.errors[d] << endl;
            }
        }
        sub_mag_file.close();
        
        // Save energy-sublattice cross terms
        ofstream cross_file(out_dir + "/energy_sublattice_cross.txt");
        cross_file << "# Energy-sublattice cross correlations <E*S_α> - <E><S_α>" << endl;
        cross_file << "# Format: sublattice component mean error" << endl;
        cross_file << std::scientific << std::setprecision(12);
        
        for (size_t alpha = 0; alpha < obs.energy_sublattice_cross.size(); ++alpha) {
            const auto& cross = obs.energy_sublattice_cross[alpha];
            for (size_t d = 0; d < cross.values.size(); ++d) {
                cross_file << alpha << " " << d << " " 
                          << cross.values[d] << " " << cross.errors[d] << endl;
            }
        }
        cross_file.close();
    }

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
                                              double swap_acceptance_rate) const {
#ifdef HDF5_ENABLED
        ensure_directory_exists(out_dir);
        
        string filename = out_dir + "/parallel_tempering_data.h5";
        size_t n_samples = energies.size();
        
        // Create HDF5 writer
        HDF5PTWriter writer(filename, obs.temperature, lattice_size, spin_dim, N_atoms,
                           n_samples, n_anneal, n_measure, probe_rate, swap_rate,
                           overrelaxation_rate, acceptance_rate, swap_acceptance_rate);
        
        // Write time series data
        writer.write_timeseries(energies, magnetizations, sublattice_mags);
        
        // Prepare observable data in format expected by writer
        vector<vector<double>> sublattice_mag_means(N_atoms);
        vector<vector<double>> sublattice_mag_errors(N_atoms);
        vector<vector<double>> energy_cross_means(N_atoms);
        vector<vector<double>> energy_cross_errors(N_atoms);
        
        for (size_t alpha = 0; alpha < N_atoms; ++alpha) {
            sublattice_mag_means[alpha] = obs.sublattice_magnetization[alpha].values;
            sublattice_mag_errors[alpha] = obs.sublattice_magnetization[alpha].errors;
            energy_cross_means[alpha] = obs.energy_sublattice_cross[alpha].values;
            energy_cross_errors[alpha] = obs.energy_sublattice_cross[alpha].errors;
        }
        
        // Write observables (including total magnetization)
        writer.write_observables(obs.energy.value, obs.energy.error,
                                obs.specific_heat.value, obs.specific_heat.error,
                                obs.magnetization.values, obs.magnetization.errors,
                                sublattice_mag_means, sublattice_mag_errors,
                                energy_cross_means, energy_cross_errors);
        
        writer.close();
#else
        std::cerr << "Warning: HDF5 support not enabled. Cannot save HDF5 output." << std::endl;
        std::cerr << "Compile with -DHDF5_ENABLED to enable HDF5 output." << std::endl;
#endif
    }

    /**
     * Save aggregated heat capacity data from all temperatures to HDF5 format
     * Called by rank 0 to save temperature-dependent thermodynamic data
     */
    void save_heat_capacity_hdf5(const string& out_dir,
                                  const vector<double>& temperatures,
                                  const vector<double>& heat_capacity,
                                  const vector<double>& dHeat) const {
#ifdef HDF5_ENABLED
        ensure_directory_exists(out_dir);
        
        string filename = out_dir + "/parallel_tempering_aggregated.h5";
        size_t n_temps = temperatures.size();
        
        // Create HDF5 file
        H5::H5File file(filename, H5F_ACC_TRUNC);
        
        // Create main data group
        H5::Group data_group = file.createGroup("/temperature_scan");
        H5::Group metadata_group = file.createGroup("/metadata");
        
        // Write metadata
        std::time_t now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        
        H5::DataSpace scalar_space(H5S_SCALAR);
        
        // Number of temperatures
        H5::Attribute n_temps_attr = metadata_group.createAttribute(
            "n_temperatures", H5::PredType::NATIVE_HSIZE, scalar_space);
        n_temps_attr.write(H5::PredType::NATIVE_HSIZE, &n_temps);
        
        // Timestamp
        H5::StrType str_type(H5::PredType::C_S1, strlen(time_str) + 1);
        H5::Attribute time_attr = metadata_group.createAttribute(
            "creation_time", str_type, scalar_space);
        time_attr.write(str_type, time_str);
        
        // Version info
        std::string version = "ClassicalSpin_Cpp v1.0";
        H5::StrType version_type(H5::PredType::C_S1, version.size() + 1);
        H5::Attribute version_attr = metadata_group.createAttribute(
            "code_version", version_type, scalar_space);
        version_attr.write(version_type, version.c_str());
        
        std::string format = "HDF5_PT_Aggregated_v1.0";
        H5::StrType format_type(H5::PredType::C_S1, format.size() + 1);
        H5::Attribute format_attr = metadata_group.createAttribute(
            "file_format", format_type, scalar_space);
        format_attr.write(format_type, format.c_str());
        
        // Write temperature array
        hsize_t dims[1] = {n_temps};
        H5::DataSpace dataspace(1, dims);
        
        H5::DataSet temp_dataset = data_group.createDataSet(
            "temperature", H5::PredType::NATIVE_DOUBLE, dataspace);
        temp_dataset.write(temperatures.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write heat capacity array
        H5::DataSet heat_dataset = data_group.createDataSet(
            "specific_heat", H5::PredType::NATIVE_DOUBLE, dataspace);
        heat_dataset.write(heat_capacity.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write heat capacity error array
        H5::DataSet dheat_dataset = data_group.createDataSet(
            "specific_heat_error", H5::PredType::NATIVE_DOUBLE, dataspace);
        dheat_dataset.write(dHeat.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Close everything
        temp_dataset.close();
        heat_dataset.close();
        dheat_dataset.close();
        data_group.close();
        metadata_group.close();
        file.close();
#else
        std::cerr << "Warning: HDF5 support not enabled. Cannot save HDF5 output." << std::endl;
        std::cerr << "Compile with -DHDF5_ENABLED to enable HDF5 output." << std::endl;
#endif
    }

    // ============================================================
    // MONTE CARLO METHODS
    // ============================================================

    /**
     * Metropolis sweep with local spin updates
     * Returns acceptance rate
     * 
     * Optimized with:
     * - Precomputed inverse temperature (beta)
     * - Batched random number generation
     * - Branchless acceptance criterion
     */
    double metropolis(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (T <= 0) return 0.0;
        
        const double beta = 1.0 / T;
        size_t accepted = 0;
        
        // Batch size for random number pre-generation
        constexpr size_t BATCH_SIZE = 64;
        vector<size_t> random_sites(BATCH_SIZE);
        vector<double> random_uniforms(BATCH_SIZE);
        
        for (size_t batch_start = 0; batch_start < lattice_size; batch_start += BATCH_SIZE) {
            const size_t batch_end = std::min(batch_start + BATCH_SIZE, lattice_size);
            const size_t current_batch_size = batch_end - batch_start;
            
            // Pre-generate random numbers for this batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                random_sites[j] = random_int_lehman(lattice_size);
                random_uniforms[j] = random_double_lehman(0.0, 1.0);
            }
            
            // Process batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                const size_t site = random_sites[j];
                const double rand_uniform = random_uniforms[j];
                
                // Generate new spin
                SpinVector new_spin = gaussian_move ? 
                    gaussian_spin_move(spins[site], sigma) :
                    gen_random_spin(spin_length);
                
                // Compute energy change
                const double dE = site_energy_diff(new_spin, spins[site], site);
                
                // Branchless acceptance: avoid branch misprediction
                const bool accept = (dE < 0.0) | (rand_uniform < std::exp(-beta * dE));
                if (accept) {
                    spins[site] = new_spin;
                    ++accepted;
                }
            }
        }
        
        return double(accepted) / double(lattice_size);
    }

    /**
     * Gaussian move around current spin
     */
    SpinVector gaussian_spin_move(const SpinVector& current_spin, double sigma) {
        SpinVector new_spin = current_spin + gen_random_spin(spin_length) * sigma;
        double norm = new_spin.norm();
        if (norm < 1e-10) return current_spin;
        return new_spin * (spin_length / norm);
    }

    /**
     * Over-relaxation sweep (microcanonical, zero acceptance rate)
     */
    void overrelaxation() {
        size_t count = 0;
        while (count < lattice_size) {
            size_t site = random_int_lehman(lattice_size);
            
            // Get local field
            SpinVector local_field = get_local_field(site);
            double norm_sq = local_field.dot(local_field);
            
            if (norm_sq == 0) {
                continue;
            } else {
                // Reflect spin: S' = 2(S·H)H/|H|^2 - S
                double proj = 2.0 * spins[site].dot(local_field) / norm_sq;
                spins[site] = local_field * proj - spins[site];
            }
            count++;
        }
    }

    /**
     * Wolff cluster update - single cluster flip
     * Returns cluster size
     */
    size_t wolff_update(double T, bool use_ghost_field = false) {
        if (T <= 0) return 0;
        
        const double beta = 1.0 / T;
        
        // Random seed site and reflection plane
        size_t seed = random_int_lehman(lattice_size);
        SpinVector r = random_unit_vector();
        
        // Precompute projections
        vector<double> proj(lattice_size);
        for (size_t i = 0; i < lattice_size; ++i) {
            proj[i] = spins[i].dot(r);
        }
        
        // BFS cluster growth
        vector<uint8_t> in_cluster(lattice_size, 0);
        vector<size_t> stack;
        stack.reserve(lattice_size / 10);
        bool attached_to_ghost = false;
        
        in_cluster[seed] = 1;
        stack.push_back(seed);
        
        while (!stack.empty()) {
            size_t i = stack.back();
            stack.pop_back();
            
            // Try to add neighbors
            size_t n_bi = bilinear_partners[i].size();
            for (size_t n = 0; n < n_bi; ++n) {
                size_t j = bilinear_partners[i][n];
                if (in_cluster[j]) continue;
                
                // Compute projected coupling and twisted partner spin projection
                SpinVector partner_spin = apply_twist_to_partner_spin(
                    spins[j], bilinear_wrap_dir[i][n]);
                double proj_j_twisted = partner_spin.dot(r);
                double K_r = r.dot(bilinear_interaction[i][n] * r);
                
                // Add bond with probability 1 - exp(-2 beta K_r s_i s_j)
                // Note: use twisted projection for partner
                if (K_r > 0 && proj[i] * proj_j_twisted > 0) {
                    double P_add = 1.0 - std::exp(-2.0 * beta * K_r * proj[i] * proj_j_twisted);
                    if (random_double_lehman(0.0, 1.0) < P_add) {
                        in_cluster[j] = 1;
                        stack.push_back(j);
                    }
                }
            }
            
            // Ghost field bonds (heuristic)
            if (use_ghost_field && field[i].norm() > 1e-10) {
                double K_field = r.dot(field[i]);
                if (K_field * proj[i] > 0) {
                    double P_ghost = 1.0 - std::exp(-beta * std::abs(K_field * proj[i]));
                    if (random_double_lehman(0.0, 1.0) < P_ghost) {
                        attached_to_ghost = true;
                    }
                }
            }
        }
        
        // Flip cluster if not attached to ghost
        size_t cluster_size = 0;
        if (!attached_to_ghost) {
            for (size_t i = 0; i < lattice_size; ++i) {
                if (in_cluster[i]) {
                    // Reflect across plane perpendicular to r
                    spins[i] = spins[i] - 2.0 * proj[i] * r;
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
     * Swendsen-Wang sweep - build and flip all clusters
     * Returns number of clusters flipped
     */
    size_t swendsen_wang_sweep(double T, bool use_ghost_field = false) {
        if (T <= 0) return 0;
        
        const double beta = 1.0 / T;
        SpinVector r = random_unit_vector();
        
        // Precompute projections
        vector<double> proj(lattice_size);
        for (size_t i = 0; i < lattice_size; ++i) {
            proj[i] = spins[i].dot(r);
        }
        
        // Union-Find structure
        vector<int> parent(lattice_size);
        vector<int> size(lattice_size, 1);
        std::iota(parent.begin(), parent.end(), 0);
        
        function<int(int)> find = [&](int x) {
            return (parent[x] == x) ? x : (parent[x] = find(parent[x]));
        };
        
        auto unite = [&](int a, int b) {
            a = find(a);
            b = find(b);
            if (a == b) return;
            if (size[a] < size[b]) std::swap(a, b);
            parent[b] = a;
            size[a] += size[b];
        };
        
        // Build clusters via bond percolation
        for (size_t i = 0; i < lattice_size; ++i) {
            size_t n_bi = bilinear_partners[i].size();
            for (size_t n = 0; n < n_bi; ++n) {
                size_t j = bilinear_partners[i][n];
                if (j <= i) continue; // Avoid double counting
                
                SpinVector partner_spin = apply_twist_to_partner_spin(
                    spins[j], bilinear_wrap_dir[i][n]);
                double proj_j_twisted = partner_spin.dot(r);
                double K_r = r.dot(bilinear_interaction[i][n] * r);
                
                // Use twisted projection for partner
                if (K_r > 0 && proj[i] * proj_j_twisted > 0) {
                    double P_bond = 1.0 - std::exp(-2.0 * beta * K_r * proj[i] * proj_j_twisted);
                    if (random_double_lehman(0.0, 1.0) < P_bond) {
                        unite(i, j);
                    }
                }
            }
        }
        
        // Ghost bonds (prevent flipping clusters aligned with field)
        vector<uint8_t> forbid_flip(lattice_size, 0);
        if (use_ghost_field) {
            for (size_t i = 0; i < lattice_size; ++i) {
                if (field[i].norm() > 1e-10) {
                    double K_field = r.dot(field[i]);
                    if (K_field * proj[i] > 0) {
                        double P_ghost = 1.0 - std::exp(-beta * std::abs(K_field * proj[i]));
                        if (random_double_lehman(0.0, 1.0) < P_ghost) {
                            forbid_flip[find(i)] = 1;
                        }
                    }
                }
            }
        }
        
        // Flip each cluster with probability 1/2
        vector<uint8_t> flip_root(lattice_size, 0);
        for (size_t i = 0; i < lattice_size; ++i) {
            int root = find(i);
            if (i == root && !forbid_flip[root]) {
                flip_root[root] = (random_double_lehman(0.0, 1.0) < 0.5) ? 1 : 0;
            }
        }
        
        // Apply flips
        size_t flipped_clusters = 0;
        for (size_t i = 0; i < lattice_size; ++i) {
            if (flip_root[find(i)]) {
                spins[i] = spins[i] - 2.0 * proj[i] * r;
                if (i == find(i)) ++flipped_clusters;
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
    size_t metropolis_twist_sweep(double T) {
        if (T <= 0) return 0;
        if (spin_dim != 3) return 0;  // Only defined for 3D spins
        
        size_t accepted = 0;
        
        // Probability of attempting a global (random) move vs incremental
        const double global_move_prob = 0.1;
        
        // For each dimension that has length > 1, attempt one rotation update
        for (size_t d = 0; d < 3; ++d) {
            size_t Ld = (d == 0) ? dim1 : (d == 1) ? dim2 : dim3;
            if (Ld <= 1) continue;
            
            // Energy on boundary sites before the move
            double E_before = 0.0;
            for (size_t idx : boundary_sites_per_dim[d]) {
                E_before += site_energy(spins[idx], idx);
            }
            
            double angle_new;
            bool is_global_move = (random_double_lehman(0, 1) < global_move_prob);
            
            if (is_global_move) {
                // Global move: propose completely random angle
                angle_new = random_double_lehman(0, 2 * M_PI);
            } else {
                // Incremental move: small perturbation to current angle
                // Step size scales with temperature (larger at high T)
                double max_step = std::min(0.3, std::max(0.02, T * 0.5));
                double delta = random_double_lehman(-max_step, max_step);
                angle_new = twist_angles[d] + delta;
            }
            
            // Generate rotation matrix around z-axis
            SpinMatrix R_new = rotation_from_axis_angle(rotation_axis[d], angle_new);
            
            // Save old state and apply proposed move
            SpinMatrix saved_R = twist_matrices[d];
            double saved_angle = twist_angles[d];
            twist_matrices[d] = R_new;
            twist_angles[d] = angle_new;
            
            // Energy after the move
            double E_after = 0.0;
            for (size_t idx : boundary_sites_per_dim[d]) {
                E_after += site_energy(spins[idx], idx);
            }
            
            double dE = E_after - E_before;
            bool accept = (dE < 0) || (random_double_lehman(0, 1) < std::exp(-dE / T));
            if (!accept) {
                twist_matrices[d] = saved_R;
                twist_angles[d] = saved_angle;
            } else {
                accepted++;
            }
        }
        
        return accepted;
    }
    
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
                            size_t twist_sweep_count = 100) {
        
        // Setup output directory
        if (!out_dir.empty()) {
            std::filesystem::create_directories(out_dir);
        }
        
        // Initialize random seed
        seed_lehman(chrono::system_clock::now().time_since_epoch().count() * 2 + 1);
        
        double T = T_start;
        double sigma = 1000.0;
        
        cout << "Starting simulated annealing: T=" << T_start << " → " << T_end << endl;
        if (T_zero) {
            cout << "T=0 mode enabled: will perform " << n_deterministics << " deterministic sweeps at T=0" << endl;
        }
        if (boundary_update) {
            cout << "Twist boundary updates enabled: " << twist_sweep_count << " twist sweeps per MC sweep" << endl;
        }
        
        size_t temp_step = 0;
        while (T > T_end) {
            // Perform sweeps at this temperature
            double acc_sum = perform_mc_sweeps(n_anneal, T, gaussian_move, sigma, 
                                              overrelaxation_rate, boundary_update, twist_sweep_count);
            
            // Calculate acceptance rate (normalize differently if overrelaxation is used)
            double acceptance = (overrelaxation_rate > 0) ? 
                acc_sum / double(n_anneal) * overrelaxation_rate : 
                acc_sum / double(n_anneal);
            
            // Progress report
            if (temp_step % 10 == 0 || T <= T_end * 1.5) {
                double E = energy_density();
                cout << "T=" << std::scientific << T << ", E/N=" << E 
                     << ", acc=" << std::fixed << acceptance;
                if (gaussian_move) cout << ", σ=" << sigma;
                cout << endl;
            }
            
            // Adaptive sigma adjustment for gaussian moves (match original logic)
            if (gaussian_move && acceptance < 0.5) {
                sigma = sigma * 0.5 / (1.0 - acceptance);
                if (temp_step % 10 == 0 || T <= T_end * 1.5) {
                    cout << "Sigma adjusted to " << sigma << endl;
                }
            }
            
            // Cool down
            T *= cooling_rate;
            ++temp_step;
        }
        
        cout << "Final energy density: " << energy_density() << endl;
        
        // Save spin config after annealing (before deterministic sweeps)
        if (!out_dir.empty()) {
            save_spin_config(out_dir + "/spins_T=" + std::to_string(T) + ".txt");
            save_positions(out_dir + "/positions.txt");
            if (boundary_update) {
                save_twist_angles(out_dir + "/twist_angles_T=" + std::to_string(T) + ".txt");
            }
        }
        
        // Final measurements if requested (before deterministic sweeps)
        if (save_observables && !out_dir.empty()) {
            perform_final_measurements(T_end, sigma, gaussian_move, 
                                      overrelaxation_rate, out_dir);
        }
        
        // T=0 deterministic sweeps if requested
        if (T_zero && n_deterministics > 0) {
            cout << "\nPerforming " << n_deterministics << " deterministic sweeps at T=0..." << endl;
            for (size_t sweep = 0; sweep < n_deterministics; ++sweep) {
                deterministic_sweep(1);
                
                if (sweep % 100 == 0 || sweep == n_deterministics - 1) {
                    double E = energy_density();
                    cout << "Deterministic sweep " << sweep << "/" << n_deterministics 
                         << ", E/N=" << E << endl;
                }
            }
            cout << "Deterministic sweeps completed. Final energy: " << energy_density() << endl;
            // Save final configuration
            if (!out_dir.empty()) {
                save_spin_config(out_dir + "/spins_T=0.txt");
                if (boundary_update) {
                    save_twist_angles(out_dir + "/twist_angles_T=0.txt");
                }
            }
        }
    }

    /**
     * Perform detailed measurements at final temperature
     * Computes: energy, specific heat, sublattice magnetizations, and cross-correlations
     * All with binning analysis for error estimation
     */
    void perform_final_measurements(double T_final, double sigma, bool gaussian_move,
                                   size_t overrelaxation_rate, const string& out_dir) {
        cout << "\n=== Final measurements at T=" << T_final << " ===" << endl;
        
        // Step 1: Estimate autocorrelation time
        cout << "Estimating autocorrelation time..." << endl;
        vector<double> prelim_energies;
        size_t prelim_samples = 10000;
        size_t prelim_interval = 10;
        prelim_energies.reserve(prelim_samples / prelim_interval);
        
        for (size_t i = 0; i < prelim_samples; ++i) {
            metropolis(T_final, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                overrelaxation();
            }
            if (i % prelim_interval == 0) {
                prelim_energies.push_back(total_energy(spins));
            }
        }
        
        AutocorrelationResult acf = compute_autocorrelation(prelim_energies, prelim_interval);
        cout << "  τ_int = " << acf.tau_int << endl;
        cout << "  Sampling interval = " << acf.sampling_interval << " sweeps" << endl;
        
        // Step 2: Equilibrate
        size_t equilibration = 10 * acf.sampling_interval;
        cout << "Equilibrating for " << equilibration << " sweeps..." << endl;
        perform_mc_sweeps(equilibration, T_final, gaussian_move, sigma, overrelaxation_rate, 
                         false, 100);  // boundary_update=false for measurements
        
        // Step 3: Collect samples - now including sublattice magnetizations
        size_t n_samples = 1000;
        size_t n_measure = n_samples * acf.sampling_interval;
        cout << "Collecting " << n_samples << " independent samples..." << endl;
        
        vector<double> energies;
        vector<SpinVector> magnetizations;
        vector<vector<SpinVector>> sublattice_mags;  // NEW: sublattice magnetizations
        energies.reserve(n_samples);
        magnetizations.reserve(n_samples);
        sublattice_mags.reserve(n_samples);
        
        for (size_t i = 0; i < n_measure; ++i) {
            metropolis(T_final, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                overrelaxation();
            }
            
            if (i % acf.sampling_interval == 0) {
                energies.push_back(total_energy(spins));
                magnetizations.push_back(magnetization_global());
                sublattice_mags.push_back(magnetization_sublattice());  // NEW
            }
        }
        
        cout << "Collected " << energies.size() << " samples" << endl;
        
        // Step 4: Compute comprehensive observables with binning analysis
        ThermodynamicObservables obs = compute_thermodynamic_observables(
            energies, sublattice_mags, T_final);
        
        // Print and save results
        print_thermodynamic_observables(obs);
        save_thermodynamic_observables(out_dir, obs);
        
        // Also save raw time series for further analysis
        save_observables(out_dir, energies, magnetizations);
        
        // Save sublattice magnetization time series
        save_sublattice_magnetization_timeseries(out_dir, sublattice_mags);
        
        // Save autocorrelation function
        save_autocorrelation_results(out_dir, acf);
    }

    /**
     * Save sublattice magnetization time series to file
     */
    void save_sublattice_magnetization_timeseries(const string& out_dir,
                                                   const vector<vector<SpinVector>>& sublattice_mags) const {
        ensure_directory_exists(out_dir);
        
        if (sublattice_mags.empty()) return;
        
        size_t n_sublattices = sublattice_mags[0].size();
        
        for (size_t alpha = 0; alpha < n_sublattices; ++alpha) {
            ofstream file(out_dir + "/sublattice_" + std::to_string(alpha) + "_timeseries.txt");
            file << std::scientific << std::setprecision(12);
            file << "# Sublattice " << alpha << " magnetization time series" << endl;
            file << "# Each row: M_0 M_1 ... M_{spin_dim-1}" << endl;
            
            for (const auto& sample : sublattice_mags) {
                const auto& M = sample[alpha];
                for (size_t d = 0; d < static_cast<size_t>(M.size()); ++d) {
                    if (d > 0) file << " ";
                    file << M(d);
                }
                file << "\n";
            }
            file.close();
        }
    }

    /**
     * Compute and save thermodynamic observables
     */
    void compute_and_save_observables(const vector<double>& energies,
                                     const vector<SpinVector>& magnetizations,
                                     double T, const string& out_dir) {
        // Mean energy
        double E_mean = std::accumulate(energies.begin(), energies.end(), 0.0) / energies.size();
        
        // Energy variance
        double E2_mean = 0.0;
        for (double E : energies) {
            E2_mean += E * E;
        }
        E2_mean /= energies.size();
        double var_E = E2_mean - E_mean * E_mean;
        
        // Specific heat
        double C_V = var_E / (T * T * lattice_size);
        
        // Mean magnetization
        SpinVector M_mean = SpinVector::Zero(spin_dim);
        for (const auto& M : magnetizations) {
            M_mean += M;
        }
        M_mean /= magnetizations.size();
        
        cout << "Observables:" << endl;
        cout << "  <E>/N = " << E_mean / lattice_size << endl;
        cout << "  C_V = " << C_V << endl;
        cout << "  |<M>| = " << M_mean.norm() << endl;
        
        // Save to files
        ofstream heat_file(out_dir + "/specific_heat.txt", std::ios::app);
        heat_file << T << " " << C_V << " " << std::sqrt(var_E) / (T * T * lattice_size) << endl;
        heat_file.close();
        
        save_observables(out_dir, energies, magnetizations);
    }

    /**
     * Save autocorrelation results
     */
    void save_autocorrelation_results(const string& out_dir, 
                                     const AutocorrelationResult& acf) {
        ofstream acf_file(out_dir + "/autocorrelation.txt");
        acf_file << "# lag autocorrelation" << endl;
        acf_file << "# tau_int = " << acf.tau_int << endl;
        acf_file << "# sampling_interval = " << acf.sampling_interval << endl;
        
        size_t max_output = std::min(size_t(100), acf.correlation_function.size());
        for (size_t lag = 0; lag < max_output; ++lag) {
            acf_file << lag << " " << acf.correlation_function[lag] << "\n";
        }
        acf_file.close();
    }

    /**
     * Cluster-based annealing (Wolff/SW)
     */
    void cluster_annealing(double T_start, double T_end, size_t n_anneal,
                          size_t wolff_per_temp, bool use_sw = false,
                          bool use_ghost_field = false, double cooling_rate = 0.9,
                          string out_dir = "") {
        if (!out_dir.empty()) {
            std::filesystem::create_directories(out_dir);
        }
        
        double T = T_start;
        cout << "Cluster annealing: T=" << T_start << " → " << T_end << endl;
        
        while (T > T_end) {
            for (size_t i = 0; i < n_anneal; ++i) {
                if (use_sw) {
                    swendsen_wang_sweep(T, use_ghost_field);
                } else {
                    wolff_sweep(T, wolff_per_temp, use_ghost_field);
                }
            }
            
            double E = energy_density();
            cout << "T=" << T << ", E/N=" << E << endl;
            
            T *= cooling_rate;
        }
        
        if (!out_dir.empty()) {
            save_spin_config(out_dir + "/spins_T=" + std::to_string(T) + ".txt");
        }
    }

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
     * @param swap_rate         Attempt replica exchange every N sweeps
     * @param probe_rate        Record observables every N sweeps
     * @param dir_name          Output directory
     * @param rank_to_write     List of ranks that should write output (-1 = all)
     * @param gaussian_move     Use Gaussian moves (true) or uniform (false)
     * @param comm              MPI communicator (default: MPI_COMM_WORLD)
     * @param verbose           If true, save spin configurations
     * @param accumulate_correlations  If true, accumulate real-space correlations for S(q)
     * @param n_bond_types      Number of bond types for dimer correlations (default: 3)
     */
    void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move = true, MPI_Comm comm = MPI_COMM_WORLD,
                           bool verbose = false, bool accumulate_correlations = false,
                           size_t n_bond_types = 3) {
        // Initialize MPI
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(nullptr, nullptr);
        }
        
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        
        // Set random seed
        seed_lehman((std::chrono::system_clock::now().time_since_epoch().count() + rank * 1000) * 2 + 1);
        
        double curr_Temp = temp[rank];
        double sigma = 1000.0;
        int swap_accept = 0;
        double curr_accept = 0;
        
        vector<double> heat_capacity, dHeat;
        if (rank == 0) {
            heat_capacity.resize(size);
            dHeat.resize(size);
        }
        
        vector<double> energies;
        vector<SpinVector> magnetizations;
        vector<vector<SpinVector>> sublattice_mags;  // sublattice magnetizations
        size_t expected_samples = n_measure / probe_rate + 100;
        energies.reserve(expected_samples);
        magnetizations.reserve(expected_samples);
        sublattice_mags.reserve(expected_samples);
        
        // Initialize correlation accumulator if requested
        RealSpaceCorrelationAccumulator corr_acc;
        if (accumulate_correlations) {
            corr_acc = create_correlation_accumulator(n_bond_types);
            cout << "Rank " << rank << ": Correlation accumulator initialized ("
                 << corr_acc.storage_bytes() / 1024.0 << " KB)" << endl;
        }
        
        cout << "Rank " << rank << ": T=" << curr_Temp << endl;
        
        // Equilibration
        cout << "Rank " << rank << ": Equilibrating..." << endl;
        for (size_t i = 0; i < n_anneal; ++i) {
            if (overrelaxation_rate > 0) {
                overrelaxation();
                if (i % overrelaxation_rate == 0) {
                    curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
                }
            } else {
                curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
            }
            
            // Attempt replica exchange
            if (swap_rate > 0 && i % swap_rate == 0) {
                swap_accept += attempt_replica_exchange(rank, size, temp, curr_Temp, i / swap_rate, comm);
            }
        }
        
        // Main measurement phase
        cout << "Rank " << rank << ": Measuring..." << endl;
        for (size_t i = 0; i < n_measure; ++i) {
            if (overrelaxation_rate > 0) {
                overrelaxation();
                if (i % overrelaxation_rate == 0) {
                    curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
                }
            } else {
                curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
            }
            
            if (swap_rate > 0 && i % swap_rate == 0) {
                swap_accept += attempt_replica_exchange(rank, size, temp, curr_Temp, i / swap_rate, comm);
            }
            
            if (i % probe_rate == 0) {
                energies.push_back(total_energy(spins));
                magnetizations.push_back(magnetization_global());
                sublattice_mags.push_back(magnetization_sublattice());
                
                // Accumulate real-space correlations for S(q)
                if (accumulate_correlations) {
                    accumulate_correlations_internal(corr_acc);
                }
            }
        }
        
        cout << "Rank " << rank << ": Collected " << energies.size() << " samples" << endl;
        
        // Save correlation data before MPI operations
        if (accumulate_correlations) {
            // Each rank saves its own correlation data to rank_#/correlations_T*.h5
            bool should_write = (std::find(rank_to_write.begin(), rank_to_write.end(), rank) != rank_to_write.end())
                               || (std::find(rank_to_write.begin(), rank_to_write.end(), -1) != rank_to_write.end());
            
            if (should_write) {
                string corr_dir = dir_name + "/rank_" + std::to_string(rank);
                std::filesystem::create_directories(corr_dir);
                
#ifdef HDF5_ENABLED
                // Save raw correlations to HDF5
                string h5_filename = corr_dir + "/correlations_T" + std::to_string(curr_Temp) + ".h5";
                corr_acc.save_hdf5(h5_filename);
                cout << "Rank " << rank << ": Saved correlations to " << h5_filename 
                     << " (" << corr_acc.n_samples << " samples)" << endl;
#endif
            }
        }
        
        // Gather and save statistics with comprehensive observables
        gather_and_save_statistics_comprehensive(rank, size, curr_Temp, energies, 
                                                  magnetizations, sublattice_mags,
                                                  heat_capacity, dHeat, temp, dir_name, 
                                                  rank_to_write, n_anneal, n_measure, 
                                                  curr_accept, swap_accept,
                                                  swap_rate, overrelaxation_rate, probe_rate, comm, verbose);
    }
    
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
                                double curr_Temp, size_t swap_parity, MPI_Comm comm = MPI_COMM_WORLD) {
        // Determine partner based on checkerboard pattern
        int partner_rank;
        if (swap_parity % 2 == 0) {
            partner_rank = (rank % 2 == 0) ? rank + 1 : rank - 1;
        } else {
            partner_rank = (rank % 2 == 1) ? rank + 1 : rank - 1;
        }
        
        if (partner_rank < 0 || partner_rank >= size) {
            return 0;
        }
        
        // Exchange energies
        double E = total_energy(spins);
        double E_partner, T_partner = temp[partner_rank];
        
        MPI_Sendrecv(&E, 1, MPI_DOUBLE, partner_rank, 0,
                    &E_partner, 1, MPI_DOUBLE, partner_rank, 0,
                    comm, MPI_STATUS_IGNORE);
        
        // Decide acceptance using parallel tempering Metropolis criterion:
        // P_swap = min(1, exp[Δ]) where Δ = (β_cold - β_hot)(E_cold - E_hot)
        // With ordering: rank 0 = coldest (β large), highest rank = hottest (β small)
        // rank < partner_rank means: curr is COLDER, partner is HOTTER
        int accept_int = 0;
        if (rank < partner_rank) {
            // Only the lower rank computes the acceptance decision
            // curr = cold (i), partner = hot (j)
            // β_curr > β_partner, typically E_curr < E_partner (cold has lower energy)
            // Δ = (β_curr - β_partner)(E_curr - E_partner) = (positive)(negative) = negative typically
            double beta_curr = 1.0 / curr_Temp;
            double beta_partner = 1.0 / T_partner;
            double delta = (beta_curr - beta_partner) * (E - E_partner);
            bool accept = (delta >= 0) || (random_double_lehman(0.0, 1.0) < std::exp(delta));
            accept_int = accept ? 1 : 0;
            
            // // DEBUG: Print exchange details (only first few times to avoid spam)
            // static int debug_count = 0;
            // if (debug_count < 20) {
            //     cout << "[PT DEBUG] rank=" << rank << "->" << partner_rank
            //          << " T_cold=" << curr_Temp << " T_hot=" << T_partner
            //          << " E_cold=" << E << " E_hot=" << E_partner
            //          << " delta=" << delta << " accept=" << accept << endl;
            //     ++debug_count;
            // }
        }
        
        // Communicate decision between partners using point-to-point (NOT collective MPI_Bcast!)
        // Lower rank sends decision to higher rank
        int recv_accept_int = 0;
        MPI_Sendrecv(&accept_int, 1, MPI_INT, partner_rank, 2,
                     &recv_accept_int, 1, MPI_INT, partner_rank, 2,
                     comm, MPI_STATUS_IGNORE);
        
        // Lower rank uses its own decision, higher rank uses received decision
        bool accept = (rank < partner_rank) ? (accept_int == 1) : (recv_accept_int == 1);
        
        // Exchange configurations if accepted
        if (accept) {
            // Serialize spins
            vector<double> send_buf(lattice_size * spin_dim);
            vector<double> recv_buf(lattice_size * spin_dim);
            
            for (size_t i = 0; i < lattice_size; ++i) {
                for (size_t j = 0; j < spin_dim; ++j) {
                    send_buf[i * spin_dim + j] = spins[i](j);
                }
            }
            
            MPI_Sendrecv(send_buf.data(), send_buf.size(), MPI_DOUBLE, partner_rank, 1,
                        recv_buf.data(), recv_buf.size(), MPI_DOUBLE, partner_rank, 1,
                        comm, MPI_STATUS_IGNORE);
            
            // Deserialize
            for (size_t i = 0; i < lattice_size; ++i) {
                for (size_t j = 0; j < spin_dim; ++j) {
                    spins[i](j) = recv_buf[i * spin_dim + j];
                }
            }
        }
        
        return accept ? 1 : 0;
    }

    /**
     * Estimate sampling interval using autocorrelation
     */
    size_t estimate_sampling_interval(double curr_Temp, bool gaussian_move, double& sigma,
                                     size_t overrelaxation_rate, size_t n_measure,
                                     size_t probe_rate, int rank) {
        size_t prelim_samples = std::min(size_t(10000), n_measure / 10);
        size_t prelim_interval = std::max(size_t(1), overrelaxation_rate > 0 ? overrelaxation_rate : 1);
        
        vector<double> prelim_energies = collect_energy_samples(prelim_samples, prelim_interval,
                                                                curr_Temp, gaussian_move, sigma,
                                                                overrelaxation_rate);
        
        AutocorrelationResult acf = compute_autocorrelation(prelim_energies, prelim_interval);
        size_t effective_interval = std::max(acf.sampling_interval, probe_rate);
        
        cout << "Rank " << rank << ": τ_int=" << acf.tau_int 
             << ", interval=" << effective_interval << endl;
        
        return effective_interval;
    }

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
                                   size_t probe_rate) {
        // Compute local heat capacity per site using binning analysis
        // c_V = Var(E) / (T² N²) = Var(E/N) / T²
        double E_mean = std::accumulate(energies.begin(), energies.end(), 0.0) / energies.size();
        double E2_mean = 0.0;
        for (double E : energies) {
            E2_mean += E * E;
        }
        E2_mean /= energies.size();
        double var_E = E2_mean - E_mean * E_mean;
        
        double N2 = double(lattice_size) * double(lattice_size);
        double curr_heat_capacity = var_E / (curr_Temp * curr_Temp * N2);
        double curr_dHeat = std::sqrt(var_E) / (curr_Temp * curr_Temp * N2);
        
        // Gather to root
        MPI_Gather(&curr_heat_capacity, 1, MPI_DOUBLE, heat_capacity.data(), 
                   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&curr_dHeat, 1, MPI_DOUBLE, dHeat.data(), 
                   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Report
        double total_steps = n_anneal + n_measure;
        double acc_rate = curr_accept / total_steps;
        double swap_rate_actual = (swap_rate > 0) ? double(swap_accept) / (total_steps / swap_rate) : 0.0;
        
        cout << "Rank " << rank << ": T=" << curr_Temp 
             << ", acc=" << acc_rate 
             << ", swap_acc=" << swap_rate_actual << endl;
        
        // Save results with proper MPI synchronization
        if (!dir_name.empty()) {
            // Rank 0 creates the main output directory first
            if (rank == 0) {
                filesystem::create_directories(dir_name);
            }
            // Ensure directory exists before other ranks proceed
            MPI_Barrier(MPI_COMM_WORLD);
            
            // Check if this rank should write (supports FULL mode with sentinel -1)
            bool should_write = should_rank_write(rank, rank_to_write);
            
            if (should_write) {
                string rank_dir = dir_name + "/rank_" + std::to_string(rank);
                // Each rank creates its own subdirectory (no race condition)
                filesystem::create_directories(rank_dir);
                save_observables(rank_dir, energies, magnetizations);
                save_spin_config(rank_dir + "/spins_T=" + std::to_string(curr_Temp) + ".txt");
            }
            
            // Wait for all ranks to finish writing before rank 0 writes aggregated results
            MPI_Barrier(MPI_COMM_WORLD);
            
            // Root process saves heat capacity
            if (rank == 0) {
                ofstream heat_file(dir_name + "/heat_capacity.txt");
                heat_file << "# T C_V dC_V\n";
                for (int r = 0; r < size; ++r) {
                    heat_file << temp[r] << " " << heat_capacity[r] << " " << dHeat[r] << "\n";
                }
                heat_file.close();
            }
        }
    }

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
                                   bool verbose = false) {
        
        // Compute comprehensive thermodynamic observables with binning analysis
        ThermodynamicObservables obs = compute_thermodynamic_observables(
            energies, sublattice_mags, curr_Temp);
        
        double curr_heat_capacity = obs.specific_heat.value;
        double curr_dHeat = obs.specific_heat.error;
        
        // Gather heat capacity to root
        MPI_Gather(&curr_heat_capacity, 1, MPI_DOUBLE, heat_capacity.data(), 
                   1, MPI_DOUBLE, 0, comm);
        MPI_Gather(&curr_dHeat, 1, MPI_DOUBLE, dHeat.data(), 
                   1, MPI_DOUBLE, 0, comm);
        
        // Report acceptance rates
        double total_steps = n_anneal + n_measure;
        // Account for overrelaxation: Metropolis is only called every overrelaxation_rate steps
        double metro_steps = (overrelaxation_rate > 0) ? total_steps / overrelaxation_rate : total_steps;
        double acc_rate = curr_accept / metro_steps;
        double swap_rate_actual = (swap_rate > 0) ? double(swap_accept) / (total_steps / swap_rate) : 0.0;
        
        cout << "Rank " << rank << ": T=" << curr_Temp 
             << ", acc=" << acc_rate 
             << ", swap_acc=" << swap_rate_actual 
             << ", <E>/N=" << obs.energy.value << "±" << obs.energy.error
             << ", C_V=" << obs.specific_heat.value << "±" << obs.specific_heat.error
             << endl;
        
        // Save results with proper MPI synchronization
        if (!dir_name.empty()) {
            // Rank 0 creates the main output directory first
            if (rank == 0) {
                filesystem::create_directories(dir_name);
            }
            // Ensure directory exists before other ranks proceed
            MPI_Barrier(comm);
            
            // Check if this rank should write (supports FULL mode with sentinel -1)
            bool should_write = should_rank_write(rank, rank_to_write);
            
            if (should_write) {
                string rank_dir = dir_name + "/rank_" + std::to_string(rank);
                // Each rank creates its own subdirectory (no race condition)
                filesystem::create_directories(rank_dir);
                
                // Save to HDF5 format (single file with all data)
#ifdef HDF5_ENABLED
                save_thermodynamic_observables_hdf5(rank_dir, obs, energies, magnetizations,
                                                   sublattice_mags, n_anneal, n_measure,
                                                   probe_rate, swap_rate, overrelaxation_rate,
                                                   acc_rate, swap_rate_actual);
#else
                if (rank == 0) {
                    cerr << "Warning: HDF5 not enabled. Compile with -DHDF5_ENABLED=ON to enable output." << endl;
                }
#endif
                
                // Save spin configuration only if verbose mode is enabled
                save_spin_config(rank_dir + "/spins_T=" + std::to_string(curr_Temp) + ".txt");
            }
            
            // Wait for all ranks to finish writing before rank 0 writes aggregated results
            MPI_Barrier(comm);
            
            // Root process saves aggregated results across all temperatures
            if (rank == 0) {
#ifdef HDF5_ENABLED
                // Save heat capacity to HDF5
                save_heat_capacity_hdf5(dir_name, temp, heat_capacity, dHeat);
#endif
            }
        }
        
        // Synchronize before finishing
        MPI_Barrier(comm);
    }

    // ============================================================
    // TEMPERATURE LADDER OPTIMIZATION
    // Based on Bittner et al., Phys. Rev. Lett. 101, 130603 (2008)
    // arXiv:0809.0571 - "Make life simple: unleash the full power 
    // of the parallel tempering algorithm"
    // ============================================================

    /**
     * Generate optimized temperature grid for parallel tempering
     * 
     * Implements the feedback-optimized algorithm from Bittner et al.
     * Key insight: Optimal round-trip time is achieved when acceptance rate
     * is uniform across all temperature pairs, with target A ≈ 0.5 (50%).
     * 
     * The algorithm:
     * 1. Start with geometric temperature spacing
     * 2. Run short simulations to measure acceptance rates A_i between pairs
     * 3. Adjust spacing: Δβ_{i+1} = Δβ_i * f(A_i) where f aims for uniform A
     * 4. The optimal acceptance rate that minimizes round-trip time is 50%
     * 
     * The local diffusivity D(β) ∝ A(1-A) is maximized at A = 0.5.
     * Round-trip time τ_rt ∝ ∫ dβ / D(β), minimized when A = const = 0.5.
     * 
     * @param Tmin              Minimum (coldest) temperature
     * @param Tmax              Maximum (hottest) temperature  
     * @param R                 Number of replicas (temperatures)
     * @param warmup_sweeps     MC sweeps for initial equilibration per replica
     * @param sweeps_per_iter   MC sweeps per feedback iteration
     * @param feedback_iters    Number of feedback optimization iterations
     * @param gaussian_move     Use Gaussian moves (true) or uniform (false)
     * @param overrelaxation_rate  Apply overrelaxation every N sweeps (0 = disabled)
     * @param target_acceptance Target acceptance rate (default: 0.5 per Bittner)
     * @param convergence_tol   Convergence tolerance for acceptance rate uniformity
     * @return OptimizedTempGridResult containing temperatures and diagnostics
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
        
        cout << "=== Bittner et al. Optimized Temperature Grid (Fast) ===" << endl;
        cout << "Reference: Phys. Rev. Lett. 101, 130603 (2008)" << endl;
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
        
        // Initialize with geometric spacing in temperature (linear in log T)
        // This is equivalent to linear spacing in β for a specific heat ~ const
        double beta_min = 1.0 / Tmax;  // Hottest = smallest beta
        double beta_max = 1.0 / Tmin;  // Coldest = largest beta
        vector<double> beta = linspace(beta_min, beta_max, R);
        
        // Store original spins
        SpinConfig original_spins = spins;
        
        // OPTIMIZATION 1: Use independent Lattice copies for parallel warmup
        // Each replica gets its own Lattice object to avoid state conflicts
        vector<Lattice> replica_lattices(R, *this);
        double sigma = 1000.0;  // For Gaussian moves
        
        // Warmup phase: equilibrate each replica at its temperature
        // OPTIMIZATION 2: Parallelize warmup with OpenMP
        cout << "Warming up " << R << " replicas";
        #ifdef _OPENMP
        cout << " (parallel)";
        #endif
        cout << "..." << endl;
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t k = 0; k < R; ++k) {
            double T_k = 1.0 / beta[k];
            double local_sigma = sigma;
            // Seed each replica differently
            #pragma omp critical
            {
                seed_lehman((std::chrono::system_clock::now().time_since_epoch().count() + k * 12345) * 2 + 1);
            }
            for (size_t i = 0; i < warmup_sweeps; ++i) {
                replica_lattices[k].metropolis(T_k, gaussian_move, local_sigma);
                if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                    replica_lattices[k].overrelaxation();
                }
            }
        }
        
        // OPTIMIZATION 3: Cache energies to avoid redundant calculations
        vector<double> cached_energies(R);
        for (size_t k = 0; k < R; ++k) {
            cached_energies[k] = replica_lattices[k].total_energy(replica_lattices[k].spins);
        }
        
        // Feedback optimization loop
        // Based on Eq. (4) in Bittner et al.: adjust Δβ to make A uniform
        vector<double> acceptance_rates(R - 1, 0.0);
        
        // OPTIMIZATION 4: Adaptive parameters - start aggressive, become conservative
        double base_damping = 0.3;  // Start more aggressive
        
        for (size_t iter = 0; iter < feedback_iters; ++iter) {
            // Adaptive damping: start aggressive, become more conservative
            double damping = base_damping + 0.4 * (double(iter) / double(feedback_iters));
            
            // Statistics accumulators for this iteration
            vector<size_t> attempts(R - 1, 0);
            vector<size_t> accepts(R - 1, 0);
            
            // OPTIMIZATION 5: Reduced sweeps for early iterations when far from convergence
            size_t effective_sweeps = sweeps_per_iter;
            if (iter < 3) {
                effective_sweeps = std::max(size_t(50), sweeps_per_iter / 4);
            } else if (iter < 6) {
                effective_sweeps = std::max(size_t(100), sweeps_per_iter / 2);
            }
            
            // Run MC sweeps and measure acceptance rates
            for (size_t sweep = 0; sweep < effective_sweeps; ++sweep) {
                // OPTIMIZATION 6: Parallel replica updates with OpenMP
                #pragma omp parallel for schedule(static)
                for (size_t k = 0; k < R; ++k) {
                    double T_k = 1.0 / beta[k];
                    double local_sigma = sigma;
                    replica_lattices[k].metropolis(T_k, gaussian_move, local_sigma);
                    if (overrelaxation_rate > 0 && sweep % overrelaxation_rate == 0) {
                        replica_lattices[k].overrelaxation();
                    }
                }
                
                // Update cached energies
                #pragma omp parallel for schedule(static)
                for (size_t k = 0; k < R; ++k) {
                    cached_energies[k] = replica_lattices[k].total_energy(replica_lattices[k].spins);
                }
                
                // Attempt replica exchanges for ALL adjacent pairs
                // Use checkerboard pattern to avoid conflicts
                for (int parity = 0; parity <= 1; ++parity) {
                    // OPTIMIZATION 7: Parallel exchange attempts for same parity
                    #pragma omp parallel for schedule(static)
                    for (size_t e = parity; e < R - 1; e += 2) {
                        // Use cached energies
                        // beta array is sorted: beta[0] = beta_min (hottest), beta[R-1] = beta_max (coldest)
                        // So beta[e] < beta[e+1] (e is hotter, e+1 is colder)
                        double E_hot = cached_energies[e];       // Hotter (smaller β)
                        double E_cold = cached_energies[e + 1];  // Colder (larger β)
                        
                        // Metropolis criterion for replica exchange:
                        // P_swap = min(1, exp(Δ)) where Δ = (β_cold - β_hot)(E_cold - E_hot)
                        // When energies are properly ordered (E_cold < E_hot), Δ < 0
                        double dBeta = beta[e + 1] - beta[e];  // > 0 (β_cold - β_hot)
                        double dE = E_cold - E_hot;            // typically < 0 (E_cold - E_hot)
                        double delta = dBeta * dE;             // typically < 0
                        
                        #pragma omp atomic
                        ++attempts[e];
                        
                        if (delta >= 0 || random_double_lehman(0.0, 1.0) < std::exp(delta)) {
                            #pragma omp atomic
                            ++accepts[e];
                            
                            // Swap configurations and cached energies
                            #pragma omp critical
                            {
                                std::swap(replica_lattices[e].spins, replica_lattices[e + 1].spins);
                                std::swap(cached_energies[e], cached_energies[e + 1]);
                            }
                        }
                    }
                }
            }
            
            // Compute acceptance rates
            for (size_t e = 0; e < R - 1; ++e) {
                acceptance_rates[e] = double(accepts[e]) / double(attempts[e]);
            }
            
            // Check convergence: all rates within tolerance of target
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
            
            // Adjust β spacing using the Bittner feedback rule
            // The key insight is: to achieve uniform A, adjust Δβ proportionally
            // If A_e < target: reduce Δβ_e (bring temperatures closer)
            // If A_e > target: increase Δβ_e (spread temperatures apart)
            // 
            // Update rule: Δβ_new = Δβ_old * sqrt(A_e / target) with damping
            // This is derived from the fact that A ≈ erfc(Δβ * σ_E) where σ_E
            // is the energy fluctuation, so Δβ ∝ -log(A) / σ_E approximately.
            
            vector<double> new_beta(R);
            new_beta[0] = beta[0];  // Keep minimum β (maximum T) fixed
            
            for (size_t e = 0; e < R - 1; ++e) {
                double A_e = acceptance_rates[e];
                
                // Bittner-style adjustment: use complementary error function approximation
                // For Gaussian energy distributions: A ≈ erfc(Δβ * σ_E / sqrt(2))
                // Inversion: Δβ_new / Δβ_old ≈ sqrt(log(1/A_old) / log(1/target))
                // 
                // Simplified practical rule with damping:
                double ratio;
                if (A_e < 0.01) A_e = 0.01;  // Prevent division issues
                if (A_e > 0.99) A_e = 0.99;
                
                // Use the relationship: A ≈ exp(-c * Δβ²) approximately
                // So: Δβ_new = Δβ_old * sqrt(log(A_old) / log(target))
                // With smoothing factor for stability
                double log_ratio = std::log(target_acceptance) / std::log(A_e);
                ratio = std::sqrt(std::abs(log_ratio));
                
                // Apply adaptive damping to prevent oscillations
                ratio = 1.0 + damping * (ratio - 1.0);
                
                // Clamp adjustment factor (more aggressive early, conservative late)
                double clamp_min = (iter < 3) ? 0.5 : 0.7;
                double clamp_max = (iter < 3) ? 2.0 : 1.5;
                ratio = std::clamp(ratio, clamp_min, clamp_max);
                
                double d_beta = (beta[e + 1] - beta[e]) * ratio;
                new_beta[e + 1] = new_beta[e] + d_beta;
            }
            
            // Rescale to preserve endpoints exactly
            double scale = (beta_max - beta_min) / (new_beta.back() - new_beta.front());
            for (size_t k = 1; k < R; ++k) {
                new_beta[k] = beta_min + scale * (new_beta[k] - new_beta.front());
            }
            
            beta = new_beta;
        }
        
        // Restore original spins
        spins = original_spins;
        
        // Build result
        result.temperatures = temps_from_beta(beta);
        std::sort(result.temperatures.begin(), result.temperatures.end());  // Ascending order
        
        result.acceptance_rates = acceptance_rates;
        
        // Compute local diffusivities D(T) = A(1-A)
        result.local_diffusivities.resize(R - 1);
        for (size_t e = 0; e < R - 1; ++e) {
            double A = acceptance_rates[e];
            result.local_diffusivities[e] = A * (1.0 - A);
        }
        
        // Mean acceptance rate
        result.mean_acceptance_rate = 0.0;
        for (double A : acceptance_rates) {
            result.mean_acceptance_rate += A;
        }
        result.mean_acceptance_rate /= (R - 1);
        
        // Estimate round-trip time
        // τ_rt ∝ ∫ dβ / D(β) ≈ Σ Δβ_i / D_i
        double tau_rt = 0.0;
        for (size_t e = 0; e < R - 1; ++e) {
            double d_beta = std::abs(beta[e + 1] - beta[e]);
            double D = result.local_diffusivities[e];
            if (D > 1e-6) {
                tau_rt += d_beta / D;
            } else {
                tau_rt += d_beta / 1e-6;  // Prevent divergence
            }
        }
        result.round_trip_estimate = tau_rt;
        
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
        cout << "Target acceptance rate: " << target_acceptance * 100 << "%" << endl;
        cout << "Estimated round-trip time scale: " << std::scientific 
             << result.round_trip_estimate << endl;
        cout << "Converged: " << (result.converged ? "YES" : "NO") << endl;
        
        return result;
    }

    /**
     * Legacy wrapper for backward compatibility
     * Calls the new Bittner-optimized algorithm with default 50% acceptance target
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
            0.5,   // Bittner optimal: 50% acceptance
            0.05   // 5% convergence tolerance
        );
        return result.temperatures;
    }

    /**
     * MPI-distributed temperature grid optimization (FAST VERSION)
     * 
     * This version distributes replicas across MPI ranks, same as the main PT simulation.
     * Each rank handles exactly one replica, making it R times faster than the serial version.
     * 
     * Based on Bittner et al., Phys. Rev. Lett. 101, 130603 (2008)
     * 
     * @param Tmin              Minimum (coldest) temperature
     * @param Tmax              Maximum (hottest) temperature  
     * @param warmup_sweeps     MC sweeps for initial equilibration
     * @param sweeps_per_iter   MC sweeps per feedback iteration
     * @param feedback_iters    Number of feedback optimization iterations
     * @param gaussian_move     Use Gaussian moves (true) or uniform (false)
     * @param overrelaxation_rate  Apply overrelaxation every N sweeps (0 = disabled)
     * @param target_acceptance Target acceptance rate (default: 0.5 per Bittner)
     * @param convergence_tol   Convergence tolerance for acceptance rate uniformity
     * @param comm              MPI communicator (default: MPI_COMM_WORLD)
     * @return OptimizedTempGridResult containing temperatures and diagnostics (valid on all ranks)
     */
    OptimizedTempGridResult generate_optimized_temperature_grid_mpi(
        double Tmin, double Tmax,
        size_t warmup_sweeps = 500,
        size_t sweeps_per_iter = 500,
        size_t feedback_iters = 20,
        bool gaussian_move = false,
        size_t overrelaxation_rate = 0,
        double target_acceptance = 0.5,
        double convergence_tol = 0.05,
        MPI_Comm comm = MPI_COMM_WORLD) {
        
        // Get MPI info - number of ranks = number of replicas
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
        
        // Set random seed unique to each rank
        seed_lehman((std::chrono::system_clock::now().time_since_epoch().count() + rank * 12345) * 2 + 1);
        
        if (rank == 0) {
            cout << "=== Bittner et al. Optimized Temperature Grid (MPI) ===" << endl;
            cout << "Reference: Phys. Rev. Lett. 101, 130603 (2008)" << endl;
            cout << "T_min = " << Tmin << ", T_max = " << Tmax << ", R = " << R << " (MPI ranks)" << endl;
            cout << "Target acceptance rate: " << target_acceptance * 100 << "%" << endl;
        }
        
        // Initialize beta array on all ranks (linear spacing)
        // beta[0] = beta_min (hottest), beta[R-1] = beta_max (coldest)
        double beta_min = 1.0 / Tmax;
        double beta_max = 1.0 / Tmin;
        vector<double> beta(R);
        for (int i = 0; i < R; ++i) {
            beta[i] = beta_min + (beta_max - beta_min) * double(i) / double(R - 1);
        }
        
        // Each rank's current temperature
        double my_beta = beta[rank];
        double my_T = 1.0 / my_beta;
        double sigma = 1000.0;
        
        // Warmup phase
        if (rank == 0) cout << "Warming up replicas..." << endl;
        for (size_t i = 0; i < warmup_sweeps; ++i) {
            metropolis(my_T, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                overrelaxation();
            }
        }
        MPI_Barrier(comm);
        
        // Acceptance rate tracking
        vector<double> acceptance_rates(R - 1, 0.0);
        double base_damping = 0.5;  // Less aggressive damping for faster convergence
        
        // Feedback optimization loop
        for (size_t iter = 0; iter < feedback_iters; ++iter) {
            // Damping increases over iterations for stability
            double damping = base_damping + 0.3 * (double(iter) / double(feedback_iters));
            
            // Local acceptance counters for this rank's edge
            // rank k tracks exchanges with rank k+1
            int local_attempts = 0;
            int local_accepts = 0;
            
            // Use full sweeps for better statistics (reduced noise)
            size_t effective_sweeps = sweeps_per_iter;
            
            // MC sweeps with replica exchanges
            for (size_t sweep = 0; sweep < effective_sweeps; ++sweep) {
                // Local MC update
                metropolis(my_T, gaussian_move, sigma);
                if (overrelaxation_rate > 0 && sweep % overrelaxation_rate == 0) {
                    overrelaxation();
                }
                
                // Attempt replica exchanges using checkerboard pattern
                for (int parity = 0; parity <= 1; ++parity) {
                    int partner_rank;
                    if (parity == 0) {
                        partner_rank = (rank % 2 == 0) ? rank + 1 : rank - 1;
                    } else {
                        partner_rank = (rank % 2 == 1) ? rank + 1 : rank - 1;
                    }
                    
                    if (partner_rank < 0 || partner_rank >= R) continue;
                    
                    // Exchange energies with partner
                    double my_E = total_energy(spins);
                    double partner_E;
                    MPI_Sendrecv(&my_E, 1, MPI_DOUBLE, partner_rank, 0,
                                &partner_E, 1, MPI_DOUBLE, partner_rank, 0,
                                comm, MPI_STATUS_IGNORE);
                    
                    // Lower rank computes acceptance
                    int accept_int = 0;
                    if (rank < partner_rank) {
                        // rank is hotter (smaller β), partner is colder (larger β)
                        double beta_hot = my_beta;
                        double beta_cold = beta[partner_rank];
                        double E_hot = my_E;
                        double E_cold = partner_E;
                        
                        // Δ = -(β_cold - β_hot)(E_hot - E_cold)
                        double delta = -(beta_cold - beta_hot) * (E_hot - E_cold);
                        bool accept = (delta >= 0) || (random_double_lehman(0.0, 1.0) < std::exp(delta));
                        accept_int = accept ? 1 : 0;
                        
                        // Track statistics for edge (rank, rank+1)
                        ++local_attempts;
                        if (accept) ++local_accepts;
                    }
                    
                    // Communicate decision
                    int recv_accept_int = 0;
                    MPI_Sendrecv(&accept_int, 1, MPI_INT, partner_rank, 1,
                                &recv_accept_int, 1, MPI_INT, partner_rank, 1,
                                comm, MPI_STATUS_IGNORE);
                    
                    bool accept = (rank < partner_rank) ? (accept_int == 1) : (recv_accept_int == 1);
                    
                    // Exchange configurations if accepted
                    if (accept) {
                        vector<double> send_buf(lattice_size * spin_dim);
                        vector<double> recv_buf(lattice_size * spin_dim);
                        
                        for (size_t i = 0; i < lattice_size; ++i) {
                            for (size_t j = 0; j < spin_dim; ++j) {
                                send_buf[i * spin_dim + j] = spins[i](j);
                            }
                        }
                        
                        MPI_Sendrecv(send_buf.data(), send_buf.size(), MPI_DOUBLE, partner_rank, 2,
                                    recv_buf.data(), recv_buf.size(), MPI_DOUBLE, partner_rank, 2,
                                    comm, MPI_STATUS_IGNORE);
                        
                        for (size_t i = 0; i < lattice_size; ++i) {
                            for (size_t j = 0; j < spin_dim; ++j) {
                                spins[i](j) = recv_buf[i * spin_dim + j];
                            }
                        }
                    }
                }
            }
            
            // Gather acceptance statistics to rank 0 using MPI_Gather
            // Each rank k (for k < R-1) tracks statistics for edge k→k+1
            // Only the lower rank in each pair computes the acceptance
            
            // Use MPI_Gather for cleaner collection
            // Note: rank R-1 doesn't have an edge to track, but we still need it to participate
            
            // Create send buffers - rank k sends stats for edge k (if k < R-1)
            int my_attempts = (rank < R - 1) ? local_attempts : 0;
            int my_accepts = (rank < R - 1) ? local_accepts : 0;
            
            // Gather from all ranks to rank 0
            vector<int> recv_attempts(R);
            vector<int> recv_accepts(R);
            MPI_Gather(&my_attempts, 1, MPI_INT, recv_attempts.data(), 1, MPI_INT, 0, comm);
            MPI_Gather(&my_accepts, 1, MPI_INT, recv_accepts.data(), 1, MPI_INT, 0, comm);
            
            // Rank 0 computes new beta array
            bool converged = false;
            if (rank == 0) {
                // Compute acceptance rates from gathered data
                // recv_attempts[k] contains attempts for edge k→k+1
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
                
                // Converge based on mean deviation (less sensitive to statistical noise)
                if (mean_deviation < convergence_tol) {
                    converged = true;
                    cout << "Converged at iteration " << iter + 1 << endl;
                }
                
                if (!converged) {
                    // Bittner et al. feedback optimization
                    // 
                    // Key insight: to minimize round-trip time, we minimize ∫ dβ / A(β)
                    // With discrete intervals: minimize Σ Δβ_i / A_i
                    // Subject to: Σ Δβ_i = β_max - β_min
                    // 
                    // Optimal solution: Δβ_i ∝ A_i
                    // i.e., edges with HIGH acceptance get MORE spacing,
                    // edges with LOW acceptance get LESS spacing (closer temps → higher A)
                    
                    // Compute weights proportional to acceptance rate
                    vector<double> weights(R - 1);
                    double total_weight = 0.0;
                    
                    for (int e = 0; e < R - 1; ++e) {
                        double A_e = acceptance_rates[e];
                        if (A_e < 0.01) A_e = 0.01;
                        if (A_e > 0.99) A_e = 0.99;
                        
                        // Weight proportional to A: high A → more spacing
                        weights[e] = A_e;
                        total_weight += weights[e];
                    }
                    
                    // Normalize weights to sum to 1
                    for (int e = 0; e < R - 1; ++e) {
                        weights[e] /= total_weight;
                    }
                    
                    // Compute new beta positions
                    vector<double> new_beta(R);
                    new_beta[0] = beta_min;
                    
                    double cumulative = 0.0;
                    for (int e = 0; e < R - 1; ++e) {
                        cumulative += weights[e];
                        new_beta[e + 1] = beta_min + cumulative * (beta_max - beta_min);
                    }
                    new_beta[R - 1] = beta_max;  // Ensure exact endpoint
                    
                    // Apply damping: blend old and new positions
                    for (int k = 1; k < R - 1; ++k) {
                        new_beta[k] = (1.0 - damping) * beta[k] + damping * new_beta[k];
                    }
                    
                    beta = new_beta;
                }
            }
            
            // Broadcast convergence flag and new beta array to all ranks
            int conv_int = converged ? 1 : 0;
            MPI_Bcast(&conv_int, 1, MPI_INT, 0, comm);
            MPI_Bcast(beta.data(), R, MPI_DOUBLE, 0, comm);
            
            // Update local temperature
            my_beta = beta[rank];
            my_T = 1.0 / my_beta;
            
            if (conv_int == 1) {
                result.converged = true;
                break;
            }
        }
        
        // Broadcast final acceptance rates
        MPI_Bcast(acceptance_rates.data(), R - 1, MPI_DOUBLE, 0, comm);
        
        // Build result (on all ranks)
        result.temperatures.resize(R);
        for (int i = 0; i < R; ++i) {
            result.temperatures[i] = 1.0 / beta[i];
        }
        std::sort(result.temperatures.begin(), result.temperatures.end());
        
        result.acceptance_rates = acceptance_rates;
        
        // Compute diagnostics
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
        
        double tau_rt = 0.0;
        for (int e = 0; e < R - 1; ++e) {
            double d_beta = std::abs(beta[e + 1] - beta[e]);
            double D = result.local_diffusivities[e];
            tau_rt += d_beta / std::max(D, 1e-6);
        }
        result.round_trip_estimate = tau_rt;
        
        // Print summary (rank 0 only)
        if (rank == 0) {
            cout << "\n=== Optimized Temperature Grid Summary ===" << endl;
            cout << "Temperatures (ascending):" << endl;
            for (int k = 0; k < std::min(R, 15); ++k) {
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
            cout << "Target acceptance rate: " << target_acceptance * 100 << "%" << endl;
            cout << "Converged: " << (result.converged ? "YES" : "NO") << endl;
        }
        
        // CRITICAL: Synchronize all ranks before returning
        // This ensures no stray MPI messages remain in flight and all ranks
        // exit the optimization at the same time, ready for the main PT simulation
        MPI_Barrier(comm);
        
        // NOTE: Spin configurations have been modified by replica exchanges during optimization.
        // The caller (e.g., parallel_tempering) will re-equilibrate at the assigned temperature,
        // so this is not a problem. However, if fresh random starts are desired, call init_random()
        // after this function returns.
        
        return result;
    }

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
    void landau_lifshitz_flat(const double* state_flat, double* dsdt_flat, double t) const {
        if (spin_dim == 3) {
            // SU(2): Standard cross product dS/dt = H × S
            for (size_t i = 0; i < lattice_size; ++i) {
                const size_t idx = i * 3;
                
                // Get local field (H_eff)
                double H[3];
                get_local_field_flat(state_flat, i, H);
                
                // Subtract drive field: H = H_eff - B_drive
                drive_field_at_time_flat(t, i, H);  // Subtracts in-place
                
                // Get spin components
                const double Sx = state_flat[idx + 0];
                const double Sy = state_flat[idx + 1];
                const double Sz = state_flat[idx + 2];
                
                // Cross product: dS/dt = H × S
                dsdt_flat[idx + 0] = H[1] * Sz - H[2] * Sy;
                dsdt_flat[idx + 1] = H[2] * Sx - H[0] * Sz;
                dsdt_flat[idx + 2] = H[0] * Sy - H[1] * Sx;
            }
        } else if (spin_dim == 8) {
            // SU(3): Structure constant contraction dS_i/dt = f_{ijk} H_j S_k
            const auto& f = get_SU3_structure();
            
            for (size_t site = 0; site < lattice_size; ++site) {
                const size_t idx = site * 8;
                
                // Get local field (H_eff)
                double H[8];
                get_local_field_flat(state_flat, site, H);
                
                // Subtract drive field: H = H_eff - B_drive
                drive_field_at_time_flat(t, site, H);  // Subtracts in-place
                
                // Get spin pointer
                const double* S = &state_flat[idx];
                
                // Structure constant contraction: dS_i/dt = sum_{jk} f[i](j,k) * H[j] * S[k]
                for (size_t i = 0; i < 8; ++i) {
                    double dSdt_i = 0.0;
                    for (size_t j = 0; j < 8; ++j) {
                        for (size_t k = 0; k < 8; ++k) {
                            dSdt_i += f[i](j, k) * H[j] * S[k];
                        }
                    }
                    dsdt_flat[idx + i] = dSdt_i;
                }
            }
        } else {
            // General case: use cross_product function (fallback)
            for (size_t i = 0; i < lattice_size; ++i) {
                const size_t idx = i * spin_dim;
                
                double H_arr[16];  // Max reasonable spin_dim
                get_local_field_flat(state_flat, i, H_arr);
                
                SpinVector H_eff = Eigen::Map<const Eigen::VectorXd>(H_arr, spin_dim);
                SpinVector B_drive = drive_field_at_time(t, i);
                SpinVector S_i = Eigen::Map<const Eigen::VectorXd>(&state_flat[idx], spin_dim);
                
                SpinVector dS_dt = cross_product(H_eff - B_drive, S_i);
                
                for (size_t d = 0; d < spin_dim; ++d) {
                    dsdt_flat[idx + d] = dS_dt(d);
                }
            }
        }
    }
    
    /**
     * Subtract time-dependent drive field from H array (in-place, flat version)
     * Drive field is pre-transformed to local frame during set_pulse()
     */
    void drive_field_at_time_flat(double t, size_t site_index, double* H) const {
        const size_t atom = site_index % N_atoms;
        
        const double dt1 = t - t_pulse[0];
        const double dt2 = t - t_pulse[1];
        
        const double factor1 = field_drive_amp * 
                        std::exp(-std::pow(dt1 / (2.0 * field_drive_width), 2)) *
                        std::cos( field_drive_freq * dt1);
        
        const double factor2 = field_drive_amp * 
                        std::exp(-std::pow(dt2 / (2.0 * field_drive_width), 2)) *
                        std::cos( field_drive_freq * dt2);
        
        for (size_t d = 0; d < spin_dim; ++d) {
            H[d] -= field_drive[0](atom * spin_dim + d) * factor1 +
                    field_drive[1](atom * spin_dim + d) * factor2;
        }
    }

    /**
     * Compute time-dependent drive field (pre-transformed to local frame during set_pulse)
     */
    SpinVector drive_field_at_time(double t, size_t site_index) const {
        size_t atom = site_index % N_atoms;
        
        double factor1 = field_drive_amp * 
                        std::exp(-std::pow((t - t_pulse[0]) / (2.0 * field_drive_width), 2)) *
                        std::cos( field_drive_freq * (t - t_pulse[0]));
        
        double factor2 = field_drive_amp * 
                        std::exp(-std::pow((t - t_pulse[1]) / (2.0 * field_drive_width), 2)) *
                        std::cos( field_drive_freq * (t - t_pulse[1]));
        
        return field_drive[0].segment(atom * spin_dim, spin_dim) * factor1 +
               field_drive[1].segment(atom * spin_dim, spin_dim) * factor2;
    }

    /**
     * Set time-dependent pulse (drive field is transformed to local frame)
     */
    void set_pulse(const vector<SpinVector>& field_in1, double t_B1,
                  const vector<SpinVector>& field_in2, double t_B2,
                  double pulse_amp, double pulse_width, double pulse_freq) {
        // Pack field components, transforming to local frame: B_local = R * B_global
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            field_drive[0].segment(atom * spin_dim, spin_dim) = sublattice_frames[atom] * field_in1[atom];
            field_drive[1].segment(atom * spin_dim, spin_dim) = sublattice_frames[atom] * field_in2[atom];
        }
        
        t_pulse[0] = t_B1;
        t_pulse[1] = t_B2;
        field_drive_amp = pulse_amp;
        field_drive_width = pulse_width;
        field_drive_freq = pulse_freq;
    }

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
    void ode_system(const ODEState& x, ODEState& dxdt, double t) {
        landau_lifshitz_flat(x.data(), dxdt.data(), t);
    }

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
                           string method = "dopri5", bool use_gpu = false) {
        if (use_gpu) {
#ifdef CUDA_ENABLED
            molecular_dynamics_gpu(T_start, T_end, dt_initial, out_dir, save_interval, method);
#else
            std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED)." << endl;
            std::cerr << "Falling back to CPU implementation." << endl;
            molecular_dynamics_cpu(T_start, T_end, dt_initial, out_dir, save_interval, method);
#endif
        } else {
            molecular_dynamics_cpu(T_start, T_end, dt_initial, out_dir, save_interval, method);
        }
    }

    /**
     * Run molecular dynamics simulation using Boost.Odeint (CPU implementation)
     * Requires HDF5 for output - all non-HDF5 I/O has been retired.
     */
    void molecular_dynamics_cpu(double T_start, double T_end, double dt_initial,
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
        
        cout << "Running molecular dynamics with Boost.Odeint: t=" << T_start << " → " << T_end << endl;
        cout << "Integration method: " << method << endl;
        cout << "Initial step size: " << dt_initial << endl;
        
        // Convert current spins to flat state vector
        ODEState state = spins_to_state(spins);
        
        // Create HDF5 writer with comprehensive metadata
        std::unique_ptr<HDF5MDWriter> hdf5_writer;
        if (!out_dir.empty()) {
            string hdf5_file = out_dir + "/trajectory.h5";
            cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
            hdf5_writer = std::make_unique<HDF5MDWriter>(
                hdf5_file, lattice_size, spin_dim, N_atoms, 
                dim1, dim2, dim3, method, 
                dt_initial, T_start, T_end, save_interval, spin_length, 
                &site_positions, 10000);
        }
        
        // Observer to save data at specified intervals
        size_t step_count = 0;
        size_t save_count = 0;
        auto observer = [&](const ODEState& x, double t) {
            if (step_count % save_interval == 0) {
                // Compute magnetizations directly from flat state (zero allocation)
                double M_local_arr[8] = {0};  // Max spin_dim
                double M_antiferro_arr[8] = {0};
                double M_global_arr[8] = {0};
                
                for (size_t i = 0; i < lattice_size; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    for (size_t d = 0; d < spin_dim; ++d) {
                        M_local_arr[d] += x[i * spin_dim + d];
                        M_antiferro_arr[d] += x[i * spin_dim + d] * sign;
                    }
                }
                
                // Compute global magnetization (sublattice-frame transformed)
                compute_magnetization_global_from_flat(x.data(), M_global_arr);
                
                SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
                SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim) / double(lattice_size);
                SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
                
                // Compute accurate energy density directly from flat state (includes bilinear)
                double E = total_energy_flat(x.data()) / lattice_size;
                
                // Write to HDF5 directly from flat state (no conversion needed)
                if (hdf5_writer) {
                    hdf5_writer->write_flat_step(t, M_antiferro, M_local, M_global, x.data());
                    save_count++;
                }
                
                // Progress output
                if (step_count % (save_interval * 10) == 0) {
                    cout << "t=" << t << ", E/N=" << E << ", |M|=" << M_local.norm() << endl;
                }
            }
            ++step_count;
        };
        
        // Create ODE system wrapper for Boost.Odeint
        auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
            this->ode_system(x, dxdt, t);
        };
        
        // Integrate using selected method
        double abs_tol = (method == "bulirsch_stoer") ? 1e-8 : 1e-6;
        double rel_tol = (method == "bulirsch_stoer") ? 1e-8 : 1e-6;
        integrate_ode_system(system_func, state, T_start, T_end, dt_initial,
                            observer, method, true, abs_tol, rel_tol);
        
        // Note: Lattice::spins remains unchanged (initial configuration preserved)
        // The evolved state is stored in the ODEState 'state' variable
        
        // Close HDF5 file
        if (hdf5_writer) {
            hdf5_writer->close();
            cout << "HDF5 trajectory saved with " << save_count << " snapshots" << endl;
        }
        
        cout << "Molecular dynamics complete! (" << step_count << " steps)" << endl;
    }

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
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim) / double(lattice_size);
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
                            size_t* twist_acc_ptr = nullptr) {
        double acc_sum = 0.0;
        size_t total_twist_accepted = 0;
        size_t total_twist_attempted = 0;
        
        // Perform MC sweeps
        for (size_t i = 0; i < n_sweeps; ++i) {
            if (overrelaxation_rate > 0) {
                overrelaxation();
                if (i % overrelaxation_rate == 0) {
                    acc_sum += metropolis(T, gaussian_move, sigma);
                }
            } else {
                acc_sum += metropolis(T, gaussian_move, sigma);
            }
        }
        
        // Perform twist boundary updates after MC sweeps are complete
        if (boundary_update) {
            for (size_t j = 0; j < twist_sweep_count; ++j) {
                size_t accepted = metropolis_twist_sweep(T);
                total_twist_accepted += accepted;
                // Count number of dimensions with Ld > 1 as attempts
                size_t n_dims = ((dim1 > 1) ? 1 : 0) + ((dim2 > 1) ? 1 : 0) + ((dim3 > 1) ? 1 : 0);
                total_twist_attempted += n_dims;
            }
            
            // Report twist boundary diagnostics
            if (total_twist_attempted > 0) {
                double twist_acc_rate = double(total_twist_accepted) / double(total_twist_attempted);
                cout << "  [Twist BC: " << total_twist_accepted << "/" << total_twist_attempted 
                     << " accepted (" << std::fixed << std::setprecision(3) << twist_acc_rate * 100.0 
                     << "%)]" << endl;
            }
        }
        
        if (twist_acc_ptr) {
            *twist_acc_ptr = total_twist_accepted;
        }
        
        return acc_sum;
    }

    /**
     * Helper: Collect energy samples with regular MC sweeps
     */
    vector<double> collect_energy_samples(size_t n_samples, size_t interval,
                                         double T, bool gaussian_move, double& sigma,
                                         size_t overrelaxation_rate = 0) {
        vector<double> energies;
        energies.reserve(n_samples / interval + 1);
        
        for (size_t i = 0; i < n_samples; ++i) {
            if (overrelaxation_rate > 0) {
                overrelaxation();
                if (i % overrelaxation_rate == 0) {
                    metropolis(T, gaussian_move, sigma);
                }
            } else {
                metropolis(T, gaussian_move, sigma);
            }
            
            if (i % interval == 0) {
                energies.push_back(total_energy(spins));
            }
        }
        
        return energies;
    }

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
                         const vector<SpinVector>& magnetizations) {
        ensure_directory_exists(dir_path);
        
        // Save energy time series
        ofstream energy_file(dir_path + "/energy.txt");
        for (double E : energies) {
            energy_file << E / lattice_size << "\n";
        }
        energy_file.close();
        
        // Save magnetization time series
        ofstream mag_file(dir_path + "/magnetization.txt");
        for (const auto& M : magnetizations) {
            for (int i = 0; i < M.size(); ++i) {
                mag_file << M(i);
                if (i < M.size() - 1) mag_file << " ";
            }
            mag_file << "\n";
        }
        mag_file.close();
    }


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
               string method = "dopri5", bool use_gpu = false) {
        
        if (use_gpu) {
#ifdef CUDA_ENABLED
            return single_pulse_drive_gpu(field_in, t_B, pulse_amp, pulse_width, pulse_freq, 
                            T_start, T_end, step_size, method);
#else
            std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED)." << endl;
            std::cerr << "Falling back to CPU implementation." << endl;
            // Fall through to CPU implementation
#endif
        }
        
        // Set up pulse
        set_pulse(field_in, t_B, vector<SpinVector>(N_atoms, SpinVector::Zero(spin_dim)), 
                 0.0, pulse_amp, pulse_width, pulse_freq);
        
        // Storage for trajectory: (time, [M_antiferro, M_local, M_global])
        vector<pair<double, array<SpinVector, 3>>> trajectory;
        
        // Start from initial spins configuration (always use Lattice::spins as starting point)
        ODEState state = spins_to_state(spins);
        
        // Create ODE system wrapper
        auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
            this->ode_system(x, dxdt, t);
        };
        
        // Observer to collect magnetization at regular intervals
        double last_save_time = T_start;
        auto observer = [&](const ODEState& x, double t) {
            if (t - last_save_time >= step_size - 1e-10 || t >= T_end - 1e-10) {
                // Compute magnetizations directly from flat state
                double M_local_arr[8] = {0};
                double M_antiferro_arr[8] = {0};
                double M_global_arr[8] = {0};
                
                for (size_t i = 0; i < lattice_size; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    
                    for (size_t d = 0; d < spin_dim; ++d) {
                        M_local_arr[d] += x[i * spin_dim + d];
                        M_antiferro_arr[d] += x[i * spin_dim + d] * sign;
                    }
                }
                
                // Use helper function for global magnetization
                compute_magnetization_global_from_flat(x.data(), M_global_arr);
                
                SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
                SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim) / double(lattice_size);
                SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
                
                trajectory.push_back({t, {M_antiferro, M_local, M_global}});
                last_save_time = t;
            }
        };
        
        // Integrate (state was initialized from spins earlier)
        integrate_ode_system(system_func, state, T_start, T_end, step_size,
                            observer, method, false, 1e-10, 1e-10);
        
        // Note: Lattice::spins remains unchanged - only ODEState evolved
        
        // Reset pulse
        field_drive[0] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive[1] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive_amp = 0.0;
        
        return trajectory;
    }

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
                   string method = "dopri5", bool use_gpu = false) {
        
        if (use_gpu) {
#ifdef CUDA_ENABLED
            return double_pulse_drive_gpu(field_in_1, t_B_1, field_in_2, t_B_2, 
                                pulse_amp, pulse_width, pulse_freq,
                                T_start, T_end, step_size, method);
#else
            std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED)." << endl;
            std::cerr << "Falling back to CPU implementation." << endl;
            // Fall through to CPU implementation
#endif
        }
        
        // Set up two-pulse configuration
        set_pulse(field_in_1, t_B_1, field_in_2, t_B_2, 
                 pulse_amp, pulse_width, pulse_freq);
        
        // Storage for trajectory: (time, [M_antiferro, M_local, M_global])
        vector<pair<double, array<SpinVector, 3>>> trajectory;
        
        // Start from initial spins configuration (always use Lattice::spins as starting point)
        ODEState state = spins_to_state(spins);
        
        // Create ODE system wrapper
        auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
            this->ode_system(x, dxdt, t);
        };
        
        // Observer to collect magnetization at regular intervals
        double last_save_time = T_start;
        auto observer = [&](const ODEState& x, double t) {
            if (t - last_save_time >= step_size - 1e-10 || t >= T_end - 1e-10) {
                // Compute magnetizations directly from flat state
                double M_local_arr[8] = {0};
                double M_antiferro_arr[8] = {0};
                double M_global_arr[8] = {0};
                
                compute_magnetizations_from_flat(x.data(), lattice_size, spin_dim, 
                    M_local_arr, M_antiferro_arr);
                
                // Use helper function for global magnetization
                compute_magnetization_global_from_flat(x.data(), M_global_arr);
                
                SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
                SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim) / double(lattice_size);
                SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
                
                trajectory.push_back({t, {M_antiferro, M_local, M_global}});
                last_save_time = t;
            }
        };
        
        // Integrate (state was initialized from spins earlier)
        integrate_ode_system(system_func, state, T_start, T_end, step_size,
                            observer, method, false, 1e-10, 1e-10);
        
        // Note: Lattice::spins remains unchanged - only ODEState evolved
        
        // Reset pulse
        field_drive[0] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive[1] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive_amp = 0.0;
        
        return trajectory;
    }

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
                                 bool use_gpu = false) {
        
        std::filesystem::create_directories(dir_name);
        
        cout << "\n==========================================" << endl;
        cout << "Pump-Probe Spectroscopy Workflow" << endl;
        cout << "==========================================" << endl;
        cout << "Pulse parameters:" << endl;
        cout << "  Amplitude: " << pulse_amp << endl;
        cout << "  Width: " << pulse_width << endl;
        cout << "  Frequency: " << pulse_freq << endl;
        cout << "Delay scan: " << tau_start << " → " << tau_end << " (step: " << tau_step << ")" << endl;
        cout << "Integration time: " << T_start << " → " << T_end << " (step: " << T_step << ")" << endl;
        
        // Use current spin configuration as ground state (assumed pre-loaded)
        cout << "\n[1/3] Using current configuration as ground state..." << endl;
        double E_ground = energy_density();
        SpinVector M_ground = magnetization_local();
        cout << "  Ground state: E/N = " << E_ground << ", |M| = " << M_ground.norm() << endl;
        
        // Save initial configuration
        save_positions(dir_name + "/positions.txt");
        save_spin_config(dir_name + "/initial_spins.txt");
        
        // Backup ground state
        SpinConfig ground_state = spins;
        
        // Step 2: Reference single-pulse dynamics (pump at t=0)
        cout << "\n[2/3] Running reference single-pulse dynamics (M0)..." << endl;
        if (use_gpu) cout << "  Using GPU acceleration" << endl;
        auto M0_trajectory = single_pulse_drive(field_in, 0.0, pulse_amp, pulse_width, pulse_freq,
                                   T_start, T_end, T_step, method, use_gpu);
        
        // Step 3: Delay time scan
        int tau_steps = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
        cout << "\n[3/3] Scanning delay times (" << tau_steps << " steps)..." << endl;
        
        // Store all trajectories in memory first
        vector<vector<pair<double, array<SpinVector, 3>>>> M1_trajectories;
        vector<vector<pair<double, array<SpinVector, 3>>>> M01_trajectories;
        vector<double> tau_values;
        
        M1_trajectories.reserve(tau_steps);
        M01_trajectories.reserve(tau_steps);
        tau_values.reserve(tau_steps);
        
        double current_tau = tau_start;
        for (int i = 0; i < tau_steps; ++i) {
            cout << "\n--- Delay time " << (i+1) << "/" << tau_steps << ": tau = " << current_tau << " ---" << endl;
            
            tau_values.push_back(current_tau);
            
            // Restore ground state for each scan point
            spins = ground_state;
            
            // M1: Probe pulse only at time tau
            cout << "  Computing M1 (probe at tau=" << current_tau << ")..." << endl;
            auto M1_trajectory = single_pulse_drive(field_in, current_tau, pulse_amp, pulse_width, pulse_freq,
                                       T_start, T_end, T_step, method, use_gpu);
            M1_trajectories.push_back(M1_trajectory);
            
            // Restore ground state again
            spins = ground_state;
            
            // M01: Pump at t=0 + Probe at t=tau
            cout << "  Computing M01 (pump at 0 + probe at tau=" << current_tau << ")..." << endl;
            auto M01_trajectory = double_pulse_drive(field_in, 0.0, field_in, current_tau,
                                            pulse_amp, pulse_width, pulse_freq,
                                            T_start, T_end, T_step, method, use_gpu);
            M01_trajectories.push_back(M01_trajectory);
            
            current_tau += tau_step;
        }
        
        // Write everything to a single HDF5 file with comprehensive metadata
        string hdf5_file = dir_name + "/pump_probe_spectroscopy.h5";
        cout << "\nWriting all data to single HDF5 file: " << hdf5_file << endl;
        
#ifdef HDF5_ENABLED
        try {
            // Create HDF5 writer with comprehensive metadata
            HDF5PumpProbeWriter writer(
                hdf5_file,
                // Lattice parameters
                lattice_size, spin_dim, N_atoms, dim1, dim2, dim3, spin_length,
                // Pulse parameters
                pulse_amp, pulse_width, pulse_freq,
                // Time evolution
                T_start, T_end, T_step, method,
                // Delay scan
                tau_start, tau_end, tau_step,
                // Ground state info
                E_ground, M_ground, Temp_start, Temp_end, n_anneal,
                T_zero_quench, quench_sweeps,
                // Optional data
                &field_in, &site_positions
            );
            
            // Write reference trajectory
            writer.write_reference_trajectory(M0_trajectory);
            
            // Write delay-dependent trajectories
            for (int i = 0; i < tau_steps; ++i) {
                writer.write_tau_trajectory(i, tau_values[i], M1_trajectories[i], M01_trajectories[i]);
            }
            
            writer.close();
            cout << "Successfully wrote all data to single HDF5 file" << endl;
            
        } catch (H5::Exception& e) {
            std::cerr << "HDF5 Error: " << e.getDetailMsg() << endl;
        }
#else
        cout << "Warning: HDF5 support not enabled. Data not saved to HDF5 file." << endl;
        cout << "  Rebuild with -DHDF5_ENABLED to enable HDF5 output." << endl;
#endif
        
        // Restore ground state at end
        spins = ground_state;
        
        cout << "\n==========================================" << endl;
        cout << "Pump-Probe Spectroscopy Complete!" << endl;
        cout << "Output directory: " << dir_name << endl;
        cout << "Total delay points: " << tau_steps << endl;
        cout << "==========================================" << endl;
    }

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
                                     bool use_gpu = false) {
        
        int rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        std::filesystem::create_directories(dir_name);
        
        // Calculate total tau steps
        int tau_steps = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
        
        if (rank == 0) {
            cout << "\n==========================================" << endl;
            cout << "Pump-Probe Spectroscopy (MPI Parallel)" << endl;
            cout << "==========================================" << endl;
            cout << "MPI ranks: " << mpi_size << endl;
            cout << "Pulse parameters:" << endl;
            cout << "  Amplitude: " << pulse_amp << endl;
            cout << "  Width: " << pulse_width << endl;
            cout << "  Frequency: " << pulse_freq << endl;
            cout << "Delay scan: " << tau_start << " → " << tau_end << " (step: " << tau_step << ")" << endl;
            cout << "Total delay points: " << tau_steps << endl;
            cout << "Integration time: " << T_start << " → " << T_end << " (step: " << T_step << ")" << endl;
            cout << "Tau points per rank: ~" << (tau_steps + mpi_size - 1) / mpi_size << endl;
            if (use_gpu) {
                cout << "GPU acceleration: ENABLED (each rank uses assigned GPU)" << endl;
            }
        }
        
        // Use current spin configuration as ground state (assumed pre-loaded)
        if (rank == 0) {
            cout << "\n[1/4] Using current configuration as ground state..." << endl;
        }
        double E_ground = energy_density();
        SpinVector M_ground = magnetization_local();
        if (rank == 0) {
            cout << "  Ground state: E/N = " << E_ground << ", |M| = " << M_ground.norm() << endl;
        }
        
        // Save initial configuration (rank 0 only)
        if (rank == 0) {
            save_positions(dir_name + "/positions.txt");
            save_spin_config(dir_name + "/initial_spins.txt");
        }
        
        // Backup ground state
        SpinConfig ground_state = spins;
        
        // Step 2: Reference single-pulse dynamics (all ranks compute, but only rank 0 keeps result)
        // Actually all ranks need the same M0, so we can compute once and broadcast
        if (rank == 0) {
            cout << "\n[2/4] Running reference single-pulse dynamics (M0)..." << endl;
            if (use_gpu) cout << "  Using GPU acceleration" << endl;
        }
        
        // All ranks compute M0 (they have same ground state)
        auto M0_trajectory = single_pulse_drive(field_in, 0.0, pulse_amp, pulse_width, pulse_freq,
                                   T_start, T_end, T_step, method, use_gpu);
        
        // Restore ground state
        spins = ground_state;
        
        // Step 3: Distribute tau values across ranks
        if (rank == 0) {
            cout << "\n[3/4] Distributing tau delays across " << mpi_size << " ranks..." << endl;
        }
        
        // Calculate which tau indices this rank handles
        vector<int> my_tau_indices;
        vector<double> my_tau_values;
        for (int i = rank; i < tau_steps; i += mpi_size) {
            my_tau_indices.push_back(i);
            my_tau_values.push_back(tau_start + i * tau_step);
        }
        
        if (rank == 0) {
            cout << "  Each rank processing " << my_tau_indices.size() << " tau points" << endl;
        }
        
        // Local storage for this rank's trajectories
        vector<vector<pair<double, array<SpinVector, 3>>>> local_M1_trajectories;
        vector<vector<pair<double, array<SpinVector, 3>>>> local_M01_trajectories;
        
        local_M1_trajectories.reserve(my_tau_indices.size());
        local_M01_trajectories.reserve(my_tau_indices.size());
        
        // Compute trajectories for assigned tau values
        for (size_t idx = 0; idx < my_tau_indices.size(); ++idx) {
            double current_tau = my_tau_values[idx];
            int global_idx = my_tau_indices[idx];
            
            cout << "[Rank " << rank << "] Computing tau[" << global_idx << "] = " << current_tau 
                 << " (" << (idx+1) << "/" << my_tau_indices.size() << ")" << endl;
            
            // Restore ground state
            spins = ground_state;
            
            // M1: Probe pulse only at time tau
            auto M1_trajectory = single_pulse_drive(field_in, current_tau, pulse_amp, pulse_width, pulse_freq,
                                       T_start, T_end, T_step, method, use_gpu);
            local_M1_trajectories.push_back(M1_trajectory);
            
            // Restore ground state again
            spins = ground_state;
            
            // M01: Pump at t=0 + Probe at t=tau
            auto M01_trajectory = double_pulse_drive(field_in, 0.0, field_in, current_tau,
                                            pulse_amp, pulse_width, pulse_freq,
                                            T_start, T_end, T_step, method, use_gpu);
            local_M01_trajectories.push_back(M01_trajectory);
        }
        
        // Synchronize before gathering
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            cout << "\n[4/4] Gathering results from all ranks..." << endl;
        }
        
        // Compute sizes for serialization
        size_t time_points = M0_trajectory.size();
        size_t data_per_point = 1 + 3 * spin_dim;  // time + 3 SpinVectors
        size_t traj_size = time_points * data_per_point;
        
        // Compute tau values (needed for HDF5)
        vector<double> tau_values(tau_steps);
        for (int i = 0; i < tau_steps; ++i) {
            tau_values[i] = tau_start + i * tau_step;
        }

#ifdef HDF5_ENABLED
        // ===== STREAMING APPROACH: Write to HDF5 as we receive data =====
        // This avoids storing all trajectories in memory at once
        
        // Type alias for trajectory
        typedef vector<pair<double, array<SpinVector, 3>>> TrajectoryType;
        
        // Rank 0 opens HDF5 file and prepares for streaming writes
        H5::H5File* file_ptr = nullptr;
        H5::Group metadata_group, reference_group, tau_scan_group;
        
        if (rank == 0) {
            string hdf5_file = dir_name + "/pump_probe_spectroscopy.h5";
            cout << "\nWriting data to HDF5 file (streaming): " << hdf5_file << endl;
            
            try {
                file_ptr = new H5::H5File(hdf5_file, H5F_ACC_TRUNC);
                
                // Create groups
                metadata_group = file_ptr->createGroup("/metadata");
                reference_group = file_ptr->createGroup("/reference");
                tau_scan_group = file_ptr->createGroup("/tau_scan");
                
                // Write metadata (simplified - essential params only)
                H5::DataSpace attr_space(H5S_SCALAR);
                {
                    auto write_attr = [&](const char* name, double val) {
                        H5::Attribute attr = metadata_group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, attr_space);
                        attr.write(H5::PredType::NATIVE_DOUBLE, &val);
                    };
                    auto write_attr_int = [&](const char* name, size_t val) {
                        H5::Attribute attr = metadata_group.createAttribute(name, H5::PredType::NATIVE_HSIZE, attr_space);
                        attr.write(H5::PredType::NATIVE_HSIZE, &val);
                    };
                    
                    write_attr_int("lattice_size", lattice_size);
                    write_attr_int("spin_dim", spin_dim);
                    write_attr_int("N_atoms", N_atoms);
                    write_attr("pulse_amp", pulse_amp);
                    write_attr("pulse_width", pulse_width);
                    write_attr("pulse_freq", pulse_freq);
                    write_attr("T_start", T_start);
                    write_attr("T_end", T_end);
                    write_attr("T_step", T_step);
                    write_attr("tau_start", tau_start);
                    write_attr("tau_end", tau_end);
                    write_attr("tau_step", tau_step);
                    write_attr_int("tau_steps", static_cast<size_t>(tau_steps));
                    write_attr("ground_state_energy", E_ground);
                }
                
                // Write tau values array
                hsize_t tau_dims[1] = {static_cast<hsize_t>(tau_steps)};
                H5::DataSpace tau_space(1, tau_dims);
                H5::DataSet tau_dataset = tau_scan_group.createDataSet("tau_values", H5::PredType::NATIVE_DOUBLE, tau_space);
                tau_dataset.write(tau_values.data(), H5::PredType::NATIVE_DOUBLE);
                
                // Write reference trajectory M0
                hsize_t time_dims[1] = {time_points};
                H5::DataSpace time_space(1, time_dims);
                
                vector<double> times(time_points);
                for (size_t i = 0; i < time_points; ++i) times[i] = M0_trajectory[i].first;
                H5::DataSet time_ds = reference_group.createDataSet("times", H5::PredType::NATIVE_DOUBLE, time_space);
                time_ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);
                
                // Write M0 magnetization data
                auto write_mag_dataset = [&](H5::Group& grp, const char* name, int mag_idx) {
                    hsize_t dims[2] = {time_points, spin_dim};
                    H5::DataSpace dspace(2, dims);
                    vector<double> data(time_points * spin_dim);
                    for (size_t t = 0; t < time_points; ++t) {
                        for (size_t d = 0; d < spin_dim; ++d) {
                            data[t * spin_dim + d] = M0_trajectory[t].second[mag_idx](d);
                        }
                    }
                    H5::DataSet ds = grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dspace);
                    ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
                };
                
                write_mag_dataset(reference_group, "M_antiferro", 0);
                write_mag_dataset(reference_group, "M_local", 1);
                write_mag_dataset(reference_group, "M_global", 2);
                
                cout << "  Reference trajectory (M0) written." << endl;
                
            } catch (H5::Exception& e) {
                std::cerr << "HDF5 Error opening file: " << e.getDetailMsg() << endl;
                if (file_ptr) delete file_ptr;
                file_ptr = nullptr;
            }
        }
        
        // Helper lambda to write a trajectory to HDF5 (rank 0 only)
        auto write_tau_to_hdf5 = [&](int tau_idx, const TrajectoryType& M1_traj, const TrajectoryType& M01_traj) {
            if (!file_ptr) return;
            
            std::string grp_name = "/tau_scan/tau_" + std::to_string(tau_idx);
            H5::Group tau_grp = file_ptr->createGroup(grp_name);
            
            // Write tau value as attribute
            H5::DataSpace attr_space(H5S_SCALAR);
            double tau_val = tau_values[tau_idx];
            H5::Attribute tau_attr = tau_grp.createAttribute("tau_value", H5::PredType::NATIVE_DOUBLE, attr_space);
            tau_attr.write(H5::PredType::NATIVE_DOUBLE, &tau_val);
            
            size_t n_times = M1_traj.size();
            
            auto write_mag = [&](const char* name, const TrajectoryType& traj, int mag_idx) {
                hsize_t dims[2] = {n_times, spin_dim};
                H5::DataSpace dspace(2, dims);
                vector<double> data(n_times * spin_dim);
                for (size_t t = 0; t < n_times; ++t) {
                    for (size_t d = 0; d < spin_dim; ++d) {
                        data[t * spin_dim + d] = traj[t].second[mag_idx](d);
                    }
                }
                H5::DataSet ds = tau_grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dspace);
                ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            };
            
            write_mag("M1_antiferro", M1_traj, 0);
            write_mag("M1_local", M1_traj, 1);
            write_mag("M1_global", M1_traj, 2);
            
            write_mag("M01_antiferro", M01_traj, 0);
            write_mag("M01_local", M01_traj, 1);
            write_mag("M01_global", M01_traj, 2);
            
            tau_grp.close();
        };
        
        // Helper lambda to deserialize buffer to trajectory
        auto deserialize_trajectory = [&](const vector<double>& buffer) -> TrajectoryType {
            TrajectoryType traj(time_points);
            for (size_t t = 0; t < time_points; ++t) {
                size_t offset = t * data_per_point;
                traj[t].first = buffer[offset];
                for (int m = 0; m < 3; ++m) {
                    traj[t].second[m] = SpinVector::Zero(spin_dim);
                    for (size_t d = 0; d < spin_dim; ++d) {
                        traj[t].second[m](d) = buffer[offset + 1 + m * spin_dim + d];
                    }
                }
            }
            return traj;
        };
        
        // First: rank 0 writes its own local results immediately
        if (rank == 0) {
            for (size_t idx = 0; idx < my_tau_indices.size(); ++idx) {
                int tau_idx = my_tau_indices[idx];
                write_tau_to_hdf5(tau_idx, local_M1_trajectories[idx], local_M01_trajectories[idx]);
            }
            cout << "  Rank 0 local trajectories written (" << my_tau_indices.size() << " tau points)." << endl;
            
            // Free local memory on rank 0 after writing
            local_M1_trajectories.clear();
            local_M1_trajectories.shrink_to_fit();
            local_M01_trajectories.clear();
            local_M01_trajectories.shrink_to_fit();
        }
        
        // Now receive from other ranks and write immediately (streaming)
        vector<double> M1_buffer(traj_size);
        vector<double> M01_buffer(traj_size);
        
        int progress_interval = std::max(1, tau_steps / 20);  // Report every 5%
        int received_count = 0;
        
        for (int tau_idx = 0; tau_idx < tau_steps; ++tau_idx) {
            int owner_rank = tau_idx % mpi_size;
            
            if (owner_rank == 0) continue;  // Already written above
            
            if (rank == 0) {
                // Receive from owner and write immediately
                MPI_Recv(M1_buffer.data(), traj_size, MPI_DOUBLE, owner_rank, 
                        2 * tau_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(M01_buffer.data(), traj_size, MPI_DOUBLE, owner_rank, 
                        2 * tau_idx + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Deserialize and write to HDF5 immediately (no storage)
                TrajectoryType M1_traj = deserialize_trajectory(M1_buffer);
                TrajectoryType M01_traj = deserialize_trajectory(M01_buffer);
                
                write_tau_to_hdf5(tau_idx, M1_traj, M01_traj);
                
                received_count++;
                if (received_count % progress_interval == 0) {
                    int local_tau_count = static_cast<int>(my_tau_indices.size());
                    cout << "  Progress: " << received_count << "/" << (tau_steps - local_tau_count) 
                         << " remote tau points received and written." << endl;
                }
                
            } else if (rank == owner_rank) {
                // Find local index for this tau
                size_t local_idx = 0;
                for (size_t i = 0; i < my_tau_indices.size(); ++i) {
                    if (my_tau_indices[i] == tau_idx) {
                        local_idx = i;
                        break;
                    }
                }
                
                // Serialize M1
                for (size_t t = 0; t < time_points; ++t) {
                    size_t offset = t * data_per_point;
                    M1_buffer[offset] = local_M1_trajectories[local_idx][t].first;
                    for (int m = 0; m < 3; ++m) {
                        for (size_t d = 0; d < spin_dim; ++d) {
                            M1_buffer[offset + 1 + m * spin_dim + d] = local_M1_trajectories[local_idx][t].second[m](d);
                        }
                    }
                }
                
                // Serialize M01
                for (size_t t = 0; t < time_points; ++t) {
                    size_t offset = t * data_per_point;
                    M01_buffer[offset] = local_M01_trajectories[local_idx][t].first;
                    for (int m = 0; m < 3; ++m) {
                        for (size_t d = 0; d < spin_dim; ++d) {
                            M01_buffer[offset + 1 + m * spin_dim + d] = local_M01_trajectories[local_idx][t].second[m](d);
                        }
                    }
                }
                
                MPI_Send(M1_buffer.data(), traj_size, MPI_DOUBLE, 0, 2 * tau_idx, MPI_COMM_WORLD);
                MPI_Send(M01_buffer.data(), traj_size, MPI_DOUBLE, 0, 2 * tau_idx + 1, MPI_COMM_WORLD);
            }
        }
        
        // Close HDF5 file
        if (rank == 0 && file_ptr) {
            metadata_group.close();
            reference_group.close();
            tau_scan_group.close();
            file_ptr->close();
            delete file_ptr;
            cout << "Successfully wrote all data to HDF5 file (streaming mode)" << endl;
        }
        
#else
        // No HDF5 - skip the communication and output
        if (rank == 0) {
            cout << "Warning: HDF5 support not enabled. Data not saved to HDF5 file." << endl;
        }
#endif
        
        if (rank == 0) {
            cout << "\n==========================================" << endl;
            cout << "Pump-Probe Spectroscopy (MPI) Complete!" << endl;
            cout << "Output directory: " << dir_name << endl;
            cout << "Total delay points: " << tau_steps << endl;
            cout << "==========================================" << endl;
        }
        
        // Restore ground state at end
        spins = ground_state;
        
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Note: GPU-accelerated methods use the modular GPU implementation in lattice_gpu.cuh/cu
    // For C++ compilation, use_gpu parameter will automatically fallback to CPU implementation.

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
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim) / double(lattice_size);
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
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim) / double(lattice_size);
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
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim) / double(lattice_size);
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
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim) / double(lattice_size);
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
            
            SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
            SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim) / double(lattice_size);
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
