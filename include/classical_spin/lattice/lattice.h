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
        // Recompute twist matrices from axes (will be updated during simulation)
        for (size_t d = 0; d < 3; ++d) {
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
                    if (spin_dim == 3 && (bilinear_wrap_dir[i][n][0] != 0 || 
                                          bilinear_wrap_dir[i][n][1] != 0 || 
                                          bilinear_wrap_dir[i][n][2] != 0)) {
                        double S_j_twisted[3];
                        std::copy(S_j, S_j + 3, S_j_twisted);
                        
                        for (size_t d = 0; d < 3; ++d) {
                            if (bilinear_wrap_dir[i][n][d] != 0) {
                                double twisted[3] = {0, 0, 0};
                                for (size_t d2 = 0; d2 < 3; ++d2) {
                                    twisted[d2] += twist_matrices[d](d2, 0) * S_j_twisted[0];
                                    twisted[d2] += twist_matrices[d](d2, 1) * S_j_twisted[1];
                                    twisted[d2] += twist_matrices[d](d2, 2) * S_j_twisted[2];
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
                        for (size_t d = 0; d < 3; ++d) {
                            temp[d] = 0.0;
                            for (size_t d2 = 0; d2 < 3; ++d2) {
                                temp[d] += twist_matrices[dim](d, d2) * S_twisted[d2];
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
    // MONTE CARLO METHODS
    // ============================================================

    /**
     * Metropolis sweep with local spin updates
     * Returns acceptance rate
     */
    double metropolis(double T, bool gaussian_move = false, double sigma = 60.0) {
        if (T <= 0) return 0.0;
        
        const double beta = 1.0 / T;
        size_t accepted = 0;
        
        for (size_t count = 0; count < lattice_size; ++count) {
            size_t site = random_int_lehman(lattice_size);
            
            // Generate new spin
            SpinVector new_spin = gaussian_move ? 
                gaussian_spin_move(spins[site], sigma) :
                gen_random_spin(spin_length);
            
            // Compute energy change
            double dE = site_energy_diff(new_spin, spins[site], site);
            
            // Metropolis criterion
            if (dE < 0.0 || random_double_lehman(0.0, 1.0) < std::exp(-beta * dE)) {
                spins[site] = new_spin;
                ++accepted;
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
                
                // Compute projected coupling
                SpinVector partner_spin = apply_twist_to_partner_spin(
                    spins[j], bilinear_wrap_dir[i][n]);
                double K_r = r.dot(bilinear_interaction[i][n] * r);
                
                // Add bond with probability 1 - exp(-2 beta K_r s_i s_j)
                if (K_r > 0 && proj[i] * proj[j] > 0) {
                    double P_add = 1.0 - std::exp(-2.0 * beta * K_r * proj[i] * proj[j]);
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
                double K_r = r.dot(bilinear_interaction[i][n] * r);
                
                if (K_r > 0 && proj[i] * proj[j] > 0) {
                    double P_bond = 1.0 - std::exp(-2.0 * beta * K_r * proj[i] * proj[j]);
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
     */
    void metropolis_twist_sweep(double T) {
        if (T <= 0) return;
        
        const double beta = 1.0 / T;
        const double angle_step = 0.1; // radians
        
        for (size_t d = 0; d < 3; ++d) {
            // Skip dimensions with size 1
            size_t dim_size = (d == 0) ? dim1 : (d == 1) ? dim2 : dim3;
            if (dim_size <= 1) continue;
            
            // Compute energy of boundary sites
            double E_old = 0.0;
            for (size_t site : boundary_sites_per_dim[d]) {
                E_old += site_energy(spins[site], site);
            }
            
            // Propose new twist angle
            double delta_angle = random_double_lehman(-angle_step, angle_step);
            SpinMatrix twist_old = twist_matrices[d];
            
            // Get current angle from matrix (simplified for small angles)
            // Full implementation would extract angle from rotation matrix
            twist_matrices[d] = rotation_from_axis_angle(rotation_axis[d], delta_angle) * twist_old;
            
            // Compute new energy
            double E_new = 0.0;
            for (size_t site : boundary_sites_per_dim[d]) {
                E_new += site_energy(spins[site], site);
            }
            
            // Metropolis acceptance
            double dE = E_new - E_old;
            if (dE > 0 && random_double_lehman(0.0, 1.0) >= std::exp(-beta * dE)) {
                // Reject: restore old twist matrix
                twist_matrices[d] = twist_old;
            }
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
     */
    void simulated_annealing(double T_start, double T_end, size_t n_anneal,
                            size_t overrelaxation_rate = 0,
                            bool boundary_update = false,
                            bool gaussian_move = false,
                            double cooling_rate = 0.9,
                            string out_dir = "",
                            bool save_observables = false,
                            bool T_zero = false,
                            size_t n_deterministics = 1000) {
        
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
        
        size_t temp_step = 0;
        while (T > T_end) {
            // Perform sweeps at this temperature
            double acc_sum = perform_mc_sweeps(n_anneal, T, gaussian_move, sigma, 
                                              overrelaxation_rate, boundary_update);
            
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
            
            // Save intermediate configuration
            if (!out_dir.empty() && temp_step % 50 == 0) {
                save_spin_config(out_dir + "/spins_T=" + std::to_string(T) + ".dat");
            }
            
            // Cool down
            T *= cooling_rate;
            ++temp_step;
        }
        
        cout << "Final energy density: " << energy_density() << endl;
        
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
        }
        
        // Final measurements if requested
        if (save_observables && !out_dir.empty()) {
            perform_final_measurements(T_end, sigma, gaussian_move, 
                                      overrelaxation_rate, out_dir);
        }
        
        // Save final configuration
        if (!out_dir.empty()) {
            save_spin_config(out_dir + "/spins_final.dat");
        }
    }

    /**
     * Perform detailed measurements at final temperature
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
        perform_mc_sweeps(equilibration, T_final, gaussian_move, sigma, overrelaxation_rate);
        
        // Step 3: Collect samples
        size_t n_samples = 1000;
        size_t n_measure = n_samples * acf.sampling_interval;
        cout << "Collecting " << n_samples << " independent samples..." << endl;
        
        vector<double> energies;
        vector<SpinVector> magnetizations;
        energies.reserve(n_samples);
        magnetizations.reserve(n_samples);
        
        for (size_t i = 0; i < n_measure; ++i) {
            metropolis(T_final, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                overrelaxation();
            }
            
            if (i % acf.sampling_interval == 0) {
                energies.push_back(total_energy(spins));
                magnetizations.push_back(magnetization_global());
            }
        }
        
        cout << "Collected " << energies.size() << " samples" << endl;
        
        // Step 4: Compute observables
        compute_and_save_observables(energies, magnetizations, T_final, out_dir);
        
        // Save autocorrelation function
        save_autocorrelation_results(out_dir, acf);
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
            save_spin_config(out_dir + "/spins_final.dat");
        }
    }

    // ============================================================
    // PARALLEL TEMPERING
    // ============================================================

    /**
     * Parallel tempering with MPI
     */
    void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move = true) {
        // Initialize MPI
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(nullptr, nullptr);
        }
        
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
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
        size_t expected_samples = n_measure / probe_rate + 100;
        energies.reserve(expected_samples);
        magnetizations.reserve(expected_samples);
        
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
                swap_accept += attempt_replica_exchange(rank, size, temp, curr_Temp, i / swap_rate);
            }
        }
        
        // Estimate sampling interval
        cout << "Rank " << rank << ": Computing autocorrelation..." << endl;
        probe_rate = estimate_sampling_interval(curr_Temp, gaussian_move, sigma,
                                               overrelaxation_rate, n_measure, probe_rate, rank);
        
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
                swap_accept += attempt_replica_exchange(rank, size, temp, curr_Temp, i / swap_rate);
            }
            
            if (i % probe_rate == 0) {
                energies.push_back(total_energy(spins));
                magnetizations.push_back(magnetization_global());
            }
        }
        
        cout << "Rank " << rank << ": Collected " << energies.size() << " samples" << endl;
        
        // Gather and save statistics
        gather_and_save_statistics(rank, size, curr_Temp, energies, magnetizations,
                                  heat_capacity, dHeat, temp, dir_name, rank_to_write,
                                  n_anneal, n_measure, curr_accept, swap_accept,
                                  swap_rate, overrelaxation_rate, probe_rate);
    }

    /**
     * Attempt replica exchange between neighboring temperatures
     */
    int attempt_replica_exchange(int rank, int size, const vector<double>& temp,
                                double curr_Temp, size_t swap_parity) {
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
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Decide acceptance
        bool accept = false;
        if (rank < partner_rank) {
            double delta_beta = (1.0 / curr_Temp) - (1.0 / T_partner);
            double delta_E = E_partner - E;
            double P_swap = std::exp(delta_beta * delta_E);
            accept = (random_double_lehman(0.0, 1.0) < P_swap);
        }
        
        // Broadcast decision
        int accept_int = accept ? 1 : 0;
        MPI_Bcast(&accept_int, 1, MPI_INT, std::min(rank, partner_rank), MPI_COMM_WORLD);
        accept = (accept_int == 1);
        
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
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
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
        // Compute local heat capacity using binning analysis
        // (Simplified - full version would use binning_analysis from original code)
        double E_mean = std::accumulate(energies.begin(), energies.end(), 0.0) / energies.size();
        double E2_mean = 0.0;
        for (double E : energies) {
            E2_mean += E * E;
        }
        E2_mean /= energies.size();
        double var_E = E2_mean - E_mean * E_mean;
        
        double curr_heat_capacity = var_E / (curr_Temp * curr_Temp * lattice_size);
        double curr_dHeat = std::sqrt(var_E) / (curr_Temp * curr_Temp * lattice_size);
        
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
        
        // Save results
        if (!dir_name.empty()) {
            filesystem::create_directories(dir_name);
            
            // Check if this rank should write
            bool should_write = std::find(rank_to_write.begin(), rank_to_write.end(), rank) != rank_to_write.end();
            
            if (should_write) {
                string rank_dir = dir_name + "/rank_" + std::to_string(rank);
                save_observables(rank_dir, energies, magnetizations);
                save_spin_config(rank_dir + "/spins_final.dat");
            }
            
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

    // ============================================================
    // TEMPERATURE LADDER OPTIMIZATION
    // ============================================================

    /**
     * Optimize temperature ladder for parallel tempering
     */
    vector<double> optimize_temperature_ladder_roundtrip(
        double Tmin, double Tmax, size_t R,
        size_t warmup_sweeps = 200,
        size_t sweeps_per_iter = 200,
        size_t feedback_iters = 10,
        bool gaussian_move = false,
        size_t overrelaxation_rate = 0) {
        
        if (R < 3) {
            return vector<double>{Tmin, Tmax};
        }
        
        // Linear spacing in beta
        auto linspace = [](double a, double b, size_t n) {
            vector<double> result(n);
            for (size_t i = 0; i < n; ++i) {
                result[i] = a + (b - a) * double(i) / double(n - 1);
            }
            return result;
        };
        
        vector<double> beta = linspace(1.0 / Tmax, 1.0 / Tmin, R);
        
        auto temps_from_beta = [](const vector<double>& b) {
            vector<double> T(b.size());
            for (size_t i = 0; i < b.size(); ++i) {
                T[i] = 1.0 / b[i];
            }
            return T;
        };
        
        // Initialize replicas
        vector<SpinConfig> reps(R, spins);
        vector<size_t> rep_at(R);
        std::iota(rep_at.begin(), rep_at.end(), 0);
        
        // Warmup
        double sigma = 1000.0;
        for (size_t k = 0; k < R; ++k) {
            spins = reps[k];
            vector<double> T = temps_from_beta(beta);
            for (size_t i = 0; i < warmup_sweeps; ++i) {
                metropolis(T[k], gaussian_move, sigma);
                if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                    overrelaxation();
                }
            }
            reps[k] = spins;
        }
        
        // Feedback iterations
        vector<int8_t> dir_rep(R, +1);
        vector<size_t> rt_count(R, 0);
        
        for (size_t it = 0; it < feedback_iters; ++it) {
            vector<size_t> att(R - 1, 0), acc(R - 1, 0);
            
            // Diffusion sweeps
            for (size_t sweep = 0; sweep < sweeps_per_iter; ++sweep) {
                // Update each replica
                for (size_t k = 0; k < R; ++k) {
                    spins = reps[k];
                    vector<double> T = temps_from_beta(beta);
                    metropolis(T[k], gaussian_move, sigma);
                    if (overrelaxation_rate > 0) overrelaxation();
                    reps[k] = spins;
                }
                
                // Attempt swaps
                for (size_t e = 0; e < R - 1; ++e) {
                    size_t k1 = rep_at[e];
                    size_t k2 = rep_at[e + 1];
                    
                    double E1 = total_energy(reps[k1]);
                    double E2 = total_energy(reps[k2]);
                    double dBeta = beta[e + 1] - beta[e];
                    double dE = E2 - E1;
                    
                    bool swap = (random_double_lehman(0.0, 1.0) < std::exp(dBeta * dE));
                    
                    ++att[e];
                    if (swap) {
                        ++acc[e];
                        rep_at[e] = k2;
                        rep_at[e + 1] = k1;
                    }
                }
                
                // Track round trips
                for (size_t r = 0; r < R; ++r) {
                    size_t slot = rep_at[r];
                    if (slot == 0 && dir_rep[r] == -1) {
                        dir_rep[r] = +1;
                        ++rt_count[r];
                    } else if (slot == R - 1 && dir_rep[r] == +1) {
                        dir_rep[r] = -1;
                    }
                }
            }
            
            // Adjust beta spacing based on acceptance rates
            vector<double> new_beta(R);
            new_beta[0] = beta[0];
            for (size_t e = 0; e < R - 1; ++e) {
                double a_e = double(acc[e]) / double(att[e]);
                double target = 0.3;
                double factor = std::sqrt(target / (a_e + 0.01));
                factor = std::clamp(factor, 0.8, 1.25);
                double d_beta = (beta[e + 1] - beta[e]) * factor;
                new_beta[e + 1] = new_beta[e] + d_beta;
            }
            
            // Rescale to match endpoints
            double scale = (beta.back() - beta.front()) / (new_beta.back() - new_beta.front());
            for (size_t k = 1; k < R; ++k) {
                new_beta[k] = beta.front() + scale * (new_beta[k] - new_beta.front());
            }
            beta = new_beta;
        }
        
        vector<double> T = temps_from_beta(beta);
        std::sort(T.begin(), T.end());
        
        cout << "Optimized temperature ladder:" << endl;
        for (size_t k = 0; k < std::min(R, size_t(10)); ++k) {
            cout << "  T[" << k << "] = " << T[k] << endl;
        }
        if (R > 10) cout << "  ..." << endl;
        
        return T;
    }

    /**
     * Landau-Lifshitz equations (zero-allocation flat array version for ODE integrator)
     * dS/dt = (H_eff - B_drive) × S
     */
    void landau_lifshitz_flat(const double* state_flat, double* dsdt_flat, double t) const {
        if (spin_dim == 3) {
            // Optimized 3D cross product
            for (size_t i = 0; i < lattice_size; ++i) {
                double H_eff[3];
                get_local_field_flat(state_flat, i, H_eff);
                
                // Subtract drive field
                size_t atom = i % N_atoms;
                double factor1 = field_drive_amp * 
                                std::exp(-std::pow((t - t_pulse[0]) / (2.0 * field_drive_width), 2)) *
                                std::cos(2.0 * M_PI * field_drive_freq * (t - t_pulse[0]));
                double factor2 = field_drive_amp * 
                                std::exp(-std::pow((t - t_pulse[1]) / (2.0 * field_drive_width), 2)) *
                                std::cos(2.0 * M_PI * field_drive_freq * (t - t_pulse[1]));
                
                for (size_t d = 0; d < 3; ++d) {
                    H_eff[d] -= field_drive[0](atom * 3 + d) * factor1 + 
                                field_drive[1](atom * 3 + d) * factor2;
                }
                
                // Cross product: H_eff × S
                const double* S = &state_flat[i * 3];
                double* dS_dt = &dsdt_flat[i * 3];
                
                dS_dt[0] = H_eff[1] * S[2] - H_eff[2] * S[1];
                dS_dt[1] = H_eff[2] * S[0] - H_eff[0] * S[2];
                dS_dt[2] = H_eff[0] * S[1] - H_eff[1] * S[0];
            }
        } else {
            // General case
            for (size_t i = 0; i < lattice_size; ++i) {
                double H_eff_arr[8];
                get_local_field_flat(state_flat, i, H_eff_arr);
                
                SpinVector H_eff = Eigen::Map<const Eigen::VectorXd>(H_eff_arr, spin_dim);
                SpinVector B_drive = drive_field_at_time(t, i);
                SpinVector S_i = Eigen::Map<const Eigen::VectorXd>(&state_flat[i * spin_dim], spin_dim);
                
                SpinVector dS_dt = cross_product(H_eff - B_drive, S_i);
                
                for (size_t d = 0; d < spin_dim; ++d) {
                    dsdt_flat[i * spin_dim + d] = dS_dt(d);
                }
            }
        }
    }
    
    /**
     * Landau-Lifshitz equations: dS/dt = (H_eff - B_drive) × S
     * Compute dS/dt from current spin configuration and time
     * 
     * Note: Sign convention matches original implementation (H × S, not S × H)
     */
    SpinConfig landau_lifshitz(const SpinConfig& current_spins, double curr_time,
                               CrossProductMethod cross_prod) {
        SpinConfig dsdt(lattice_size);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            SpinVector H_eff = get_local_field_lattice(current_spins, i);
            SpinVector B_drive = drive_field_at_time(curr_time, i);
            // dS/dt = (H_eff - B_drive) × S
            dsdt[i] = cross_prod(H_eff - B_drive, current_spins[i]);
        }
        
        return dsdt;
    }

    /**
     * Compute time-dependent drive field
     */
    SpinVector drive_field_at_time(double t, size_t site_index) const {
        size_t atom = site_index % N_atoms;
        
        double factor1 = field_drive_amp * 
                        std::exp(-std::pow((t - t_pulse[0]) / (2.0 * field_drive_width), 2)) *
                        std::cos(2.0 * M_PI * field_drive_freq * (t - t_pulse[0]));
        
        double factor2 = field_drive_amp * 
                        std::exp(-std::pow((t - t_pulse[1]) / (2.0 * field_drive_width), 2)) *
                        std::cos(2.0 * M_PI * field_drive_freq * (t - t_pulse[1]));
        
        return field_drive[0].segment(atom * spin_dim, spin_dim) * factor1 +
               field_drive[1].segment(atom * spin_dim, spin_dim) * factor2;
    }

    /**
     * Set time-dependent pulse
     */
    void set_pulse(const vector<SpinVector>& field_in1, double t_B1,
                  const vector<SpinVector>& field_in2, double t_B2,
                  double pulse_amp, double pulse_width, double pulse_freq) {
        // Pack field components
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            field_drive[0].segment(atom * spin_dim, spin_dim) = field_in1[atom];
            field_drive[1].segment(atom * spin_dim, spin_dim) = field_in2[atom];
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
     * - "euler": Explicit Euler (1st order, simple, inaccurate)
     * - "rk2" or "midpoint": Runge-Kutta 2nd order
     * - "rk4": Classic Runge-Kutta 4th order (good balance, fixed step)
     * - "rk5" or "rkck54": Cash-Karp 5(4) (adaptive, good for smooth problems)
     * - "rk54" or "rkf54": Runge-Kutta-Fehlberg 5(4) (adaptive)
     * - "dopri5": Dormand-Prince 5(4) (default, recommended for general use)
     * - "rk78" or "rkf78": Runge-Kutta-Fehlberg 7(8) (high accuracy, expensive)
     * - "bulirsch_stoer" or "bs": Bulirsch-Stoer (very high accuracy, expensive)
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
        } else {
            // Default to dopri5 if unknown method specified
            cout << "Warning: Unknown method '" << method << "', using dopri5" << endl;
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
     * @param method        Integration method: "euler", "rk2", "rk4", "rk5", "dopri5" (default),
     *                      "rk78", "bulirsch_stoer", "adams_bashforth", etc. (see integrate_ode_system)
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
                        
                        // Transform spin to global frame using sublattice frame
                        SpinVector spin_global = SpinVector::Zero(spin_dim);
                        for (size_t mu = 0; mu < spin_dim; ++mu) {
                            for (size_t nu = 0; nu < spin_dim; ++nu) {
                                spin_global(mu) += sublattice_frames[l](nu, mu) * spins[current_site_index](nu);
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
            
            // Transform to global frame using sublattice frame
            for (size_t mu = 0; mu < spin_dim; ++mu) {
                for (size_t nu = 0; nu < spin_dim; ++nu) {
                    M_global_arr[mu] += sublattice_frames[atom](nu, mu) * x[idx + nu];
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
     */
    double perform_mc_sweeps(size_t n_sweeps, double T, bool gaussian_move, 
                            double& sigma, size_t overrelaxation_rate = 0,
                            bool boundary_update = false) {
        double acc_sum = 0.0;
        for (size_t i = 0; i < n_sweeps; ++i) {
            if (overrelaxation_rate > 0) {
                overrelaxation();
                if (i % overrelaxation_rate == 0) {
                    acc_sum += metropolis(T, gaussian_move, sigma);
                }
            } else {
                acc_sum += metropolis(T, gaussian_move, sigma);
            }
            
            if (boundary_update && i % 10 == 0) {
                metropolis_twist_sweep(T);
            }
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
        save_spin_config(dir_name + "/spins_initial.txt");
        
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
            save_spin_config(dir_name + "/spins_initial.txt");
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
        
        // Prepare full arrays for gathering (rank 0 only needs full storage)
        vector<vector<pair<double, array<SpinVector, 3>>>> M1_trajectories(tau_steps);
        vector<vector<pair<double, array<SpinVector, 3>>>> M01_trajectories(tau_steps);
        vector<double> tau_values(tau_steps);
        
        // Fill in tau values
        for (int i = 0; i < tau_steps; ++i) {
            tau_values[i] = tau_start + i * tau_step;
        }
        
        // Place local results in correct positions
        for (size_t idx = 0; idx < my_tau_indices.size(); ++idx) {
            int global_idx = my_tau_indices[idx];
            M1_trajectories[global_idx] = local_M1_trajectories[idx];
            M01_trajectories[global_idx] = local_M01_trajectories[idx];
        }
        
        // Now gather trajectories from all ranks to rank 0
        // We need to serialize trajectories for MPI communication
        // Each trajectory point: (time, [M_total, M_staggered, M_local]) with spin_dim components each
        
        size_t time_points = M0_trajectory.size();
        size_t data_per_point = 1 + 3 * spin_dim;  // time + 3 SpinVectors
        size_t traj_size = time_points * data_per_point;
        
        // For each tau index not owned by rank 0, receive from owner
        for (int tau_idx = 0; tau_idx < tau_steps; ++tau_idx) {
            int owner_rank = tau_idx % mpi_size;
            
            if (owner_rank == 0) {
                // Rank 0 already has this data
                continue;
            }
            
            if (rank == 0) {
                // Rank 0 receives data from owner
                vector<double> M1_buffer(traj_size);
                vector<double> M01_buffer(traj_size);
                
                MPI_Recv(M1_buffer.data(), traj_size, MPI_DOUBLE, owner_rank, 
                        2 * tau_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(M01_buffer.data(), traj_size, MPI_DOUBLE, owner_rank, 
                        2 * tau_idx + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Deserialize M1 trajectory
                M1_trajectories[tau_idx].resize(time_points);
                for (size_t t = 0; t < time_points; ++t) {
                    size_t offset = t * data_per_point;
                    M1_trajectories[tau_idx][t].first = M1_buffer[offset];
                    for (int m = 0; m < 3; ++m) {
                        M1_trajectories[tau_idx][t].second[m] = SpinVector::Zero(spin_dim);
                        for (size_t d = 0; d < spin_dim; ++d) {
                            M1_trajectories[tau_idx][t].second[m](d) = M1_buffer[offset + 1 + m * spin_dim + d];
                        }
                    }
                }
                
                // Deserialize M01 trajectory
                M01_trajectories[tau_idx].resize(time_points);
                for (size_t t = 0; t < time_points; ++t) {
                    size_t offset = t * data_per_point;
                    M01_trajectories[tau_idx][t].first = M01_buffer[offset];
                    for (int m = 0; m < 3; ++m) {
                        M01_trajectories[tau_idx][t].second[m] = SpinVector::Zero(spin_dim);
                        for (size_t d = 0; d < spin_dim; ++d) {
                            M01_trajectories[tau_idx][t].second[m](d) = M01_buffer[offset + 1 + m * spin_dim + d];
                        }
                    }
                }
            } else if (rank == owner_rank) {
                // This rank sends data to rank 0
                // Find local index for this tau
                size_t local_idx = 0;
                for (size_t i = 0; i < my_tau_indices.size(); ++i) {
                    if (my_tau_indices[i] == tau_idx) {
                        local_idx = i;
                        break;
                    }
                }
                
                // Serialize M1 trajectory
                vector<double> M1_buffer(traj_size);
                for (size_t t = 0; t < time_points; ++t) {
                    size_t offset = t * data_per_point;
                    M1_buffer[offset] = local_M1_trajectories[local_idx][t].first;
                    for (int m = 0; m < 3; ++m) {
                        for (size_t d = 0; d < spin_dim; ++d) {
                            M1_buffer[offset + 1 + m * spin_dim + d] = local_M1_trajectories[local_idx][t].second[m](d);
                        }
                    }
                }
                
                // Serialize M01 trajectory
                vector<double> M01_buffer(traj_size);
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
        
        // Only rank 0 writes output
        if (rank == 0) {
            string hdf5_file = dir_name + "/pump_probe_spectroscopy.h5";
            cout << "\nWriting all data to single HDF5 file: " << hdf5_file << endl;
            
#ifdef HDF5_ENABLED
            try {
                HDF5PumpProbeWriter writer(
                    hdf5_file,
                    lattice_size, spin_dim, N_atoms, dim1, dim2, dim3, spin_length,
                    pulse_amp, pulse_width, pulse_freq,
                    T_start, T_end, T_step, method,
                    tau_start, tau_end, tau_step,
                    E_ground, M_ground, Temp_start, Temp_end, n_anneal,
                    T_zero_quench, quench_sweeps,
                    &field_in, &site_positions
                );
                
                writer.write_reference_trajectory(M0_trajectory);
                
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
#endif
            
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
