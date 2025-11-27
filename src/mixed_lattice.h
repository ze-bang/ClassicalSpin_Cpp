#ifndef MIXED_LATTICE_REFACTORED_H
#define MIXED_LATTICE_REFACTORED_H

#include "unitcell.h"
#include "simple_linear_alg.h"
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

#ifdef HDF5_ENABLED
#include "hdf5_io.h"
#endif

#if defined(CUDA_ENABLED) && defined(__CUDACC__)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
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

    // Sublattice frame transformations
    vector<SpinMatrix> sublattice_frames_SU2;  // Frame transformations for SU(2) sublattices
    vector<SpinMatrix> sublattice_frames_SU3;  // Frame transformations for SU(3) sublattices

    // Interaction counts per site
    size_t num_bi_SU2;       // Number of SU(2)-SU(2) bilinear neighbors per site
    size_t num_tri_SU2;      // Number of SU(2)-SU(2)-SU(2) trilinear interactions per site
    size_t num_bi_SU3;       // Number of SU(3)-SU(3) bilinear neighbors per site
    size_t num_tri_SU3;      // Number of SU(3)-SU(3)-SU(3) trilinear interactions per site
    size_t num_bi_SU2_SU3;   // Number of mixed bilinear interactions per site
    size_t num_tri_SU2_SU3;  // Number of mixed trilinear interactions per site

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
        : spin_dim_SU2(mixed_uc.SU2_cell.spin_dim),
          spin_dim_SU3(mixed_uc.SU3_cell.spin_dim),
          N_atoms_SU2(mixed_uc.SU2_cell.N_atoms),
          N_atoms_SU3(mixed_uc.SU3_cell.N_atoms),
          dim1(dim1), dim2(dim2), dim3(dim3),
          spin_length_SU2(spin_l_SU2),
          spin_length_SU3(spin_l_SU3)
    {
        lattice_size_SU2 = N_atoms_SU2 * dim1 * dim2 * dim3;
        lattice_size_SU3 = N_atoms_SU3 * dim1 * dim2 * dim3;
        
        cout << "Initializing mixed lattice with dimensions: " << dim1 << " x " << dim2 << " x " << dim3 << endl;
        cout << "SU(2): " << lattice_size_SU2 << " sites (" << N_atoms_SU2 << " atoms/cell, spin_dim=" << spin_dim_SU2 << ")" << endl;
        cout << "SU(3): " << lattice_size_SU3 << " sites (" << N_atoms_SU3 << " atoms/cell, spin_dim=" << spin_dim_SU3 << ")" << endl;

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
            sublattice_frames_SU2[atom] = uc_SU2.sublattice_frames[atom];
        }
        for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
            sublattice_frames_SU3[atom] = uc_SU3.sublattice_frames[atom];
        }

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

        cout << "Mixed lattice initialization complete!" << endl;
        cout << "SU(2) - Max bilinear: " << num_bi_SU2 << ", Max trilinear: " << num_tri_SU2 << endl;
        cout << "SU(3) - Max bilinear: " << num_bi_SU3 << ", Max trilinear: " << num_tri_SU3 << endl;
        cout << "Mixed - Bilinear: " << num_bi_SU2_SU3 << ", Trilinear: " << num_tri_SU2_SU3 << endl;
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
            double z = random_double_lehman(-1, 1);
            double r = sqrt(1.0 - z*z);
            double phi = random_double_lehman(0, 2*M_PI);
            spin(0) = r * cos(phi);
            spin(1) = r * sin(phi);
            spin(2) = z;
        } else {
            // General n-sphere sampling
            double norm = 0.0;
            for (size_t i = 0; i < spin_dim; ++i) {
                spin(i) = random_double_lehman(-1, 1);
                norm += spin(i) * spin(i);
            }
            spin /= sqrt(norm);
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
                        for (const auto& J : uc.bilinear_interaction[l]) {
                            bi_count[site_idx]++;
                            size_t partner = flatten_index_periodic(
                                int(i) + J.offset[0], int(j) + J.offset[1], int(k) + J.offset[2], J.partner, N_atoms);
                            bi_count[partner]++;
                        }
                        
                        // Count trilinear interactions
                        for (const auto& J : uc.trilinear_interaction[l]) {
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
                        for (const auto& J : uc.bilinear_interaction[l]) {
                            size_t partner = flatten_index_periodic(
                                int(i) + J.offset[0], int(j) + J.offset[1], int(k) + J.offset[2], J.partner, N_atoms);
                            
                            bilinear[site_idx].push_back(J.bilinear_interaction);
                            bi_partners[site_idx].push_back(partner);
                            
                            // Symmetric interaction
                            bilinear[partner].push_back(J.bilinear_interaction.transpose());
                            bi_partners[partner].push_back(site_idx);
                        }
                        
                        // Trilinear interactions
                        for (const auto& J : uc.trilinear_interaction[l]) {
                            size_t partner1 = flatten_index_periodic(
                                int(i) + J.offset1[0], int(j) + J.offset1[1], int(k) + J.offset1[2], J.partner1, N_atoms);
                            size_t partner2 = flatten_index_periodic(
                                int(i) + J.offset2[0], int(j) + J.offset2[1], int(k) + J.offset2[2], J.partner2, N_atoms);
                            
                            // Add interaction T_abc S_a^(0) S_b^(1) S_c^(2)
                            trilinear[site_idx].push_back(J.trilinear_interaction);
                            tri_partners[site_idx].push_back({partner1, partner2});
                            
                            // Add symmetric contributions for energy conservation
                            // For partner1: T_bac S_b^(1) S_a^(0) S_c^(2) (swap first two indices)
                            // T_permuted[b](a,c) = T_original[a](b,c)
                            SpinTensor3 tensor_p1(spin_dim);
                            for (size_t b = 0; b < spin_dim; ++b) {
                                tensor_p1[b] = SpinMatrix(spin_dim, spin_dim);
                                for (size_t a = 0; a < spin_dim; ++a) {
                                    for (size_t c = 0; c < spin_dim; ++c) {
                                        tensor_p1[b](a, c) = J.trilinear_interaction[a](b, c);
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
                                        tensor_p2[c](a, b) = J.trilinear_interaction[a](b, c);
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
            num_bi_mixed = std::max(num_bi_mixed, std::distance(range.first, range.second));
            it = range.second;
        }
        
        // Count trilinear interactions
        for (auto it = mixed_uc.trilinear_SU2_SU3.begin();
             it != mixed_uc.trilinear_SU2_SU3.end(); ) {
            int source = it->first;
            auto range = mixed_uc.trilinear_SU2_SU3.equal_range(source);
            num_tri_mixed = std::max(num_tri_mixed, std::distance(range.first, range.second));
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
                            
                            mixed_bilinear_interaction_SU2[site_idx].push_back(bi.interaction);
                            mixed_bilinear_partners_SU2[site_idx].push_back(partner_idx);
                            
                            // Add symmetric contribution to SU(3) side
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
                            mixed_trilinear_interaction_SU2[site_idx].push_back(tri.interaction);
                            mixed_trilinear_partners_SU2[site_idx].push_back({partner1_idx, partner2_idx});
                            
                            // Symmetric contribution to SU3 site: K[c](a,b)
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
        for (size_t i = 0; i < num_bi_SU2; ++i) {
            const size_t partner_idx = bilinear_partners_SU2[site_index][i];
            bilinear_energy += spin_diff.dot(bilinear_interaction_SU2[site_index][i] * spins_SU2[partner_idx]);
        }
        
        // Mixed bilinear SU(2)-SU(3) interactions
        double mixed_bilinear_energy = 0.0;
        for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
            const size_t partner_idx = mixed_bilinear_partners_SU2[site_index][i];
            mixed_bilinear_energy += spin_diff.dot(mixed_bilinear_interaction_SU2[site_index][i] * spins_SU3[partner_idx]);
        }
        
        // Trilinear SU(2)-SU(2)-SU(2) interactions
        double trilinear_energy = 0.0;
        for (size_t i = 0; i < num_tri_SU2; ++i) {
            const size_t p1_idx = trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU2[site_index][i][1];
            const auto& T = trilinear_interaction_SU2[site_index][i];
            
            // Contract: sum_abc T[a](b,c) * spin_diff[a] * S1[b] * S2[c]
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * spins_SU2[p1_idx](b) * spins_SU2[p2_idx](c);
                    }
                }
                trilinear_energy += spin_diff(a) * temp;
            }
        }
        
        // Mixed trilinear SU(2)-SU(2)-SU(3) interactions
        double mixed_trilinear_energy = 0.0;
        for (size_t i = 0; i < num_tri_SU2_SU3; ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU2[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU2[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU2[site_index][i];
            
            // Contract: sum_abc T[a](b,c) * spin_diff[a] * SU2[b] * SU3[c]
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * spins_SU2[p1_idx](b) * spins_SU3[p2_idx](c);
                    }
                }
                mixed_trilinear_energy += spin_diff(a) * temp;
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
        for (size_t i = 0; i < num_bi_SU3; ++i) {
            const size_t partner_idx = bilinear_partners_SU3[site_index][i];
            bilinear_energy += spin_diff.dot(bilinear_interaction_SU3[site_index][i] * spins_SU3[partner_idx]);
        }
        
        // Mixed bilinear SU(3)-SU(2) interactions
        double mixed_bilinear_energy = 0.0;
        for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
            const size_t partner_idx = mixed_bilinear_partners_SU3[site_index][i];
            mixed_bilinear_energy += spin_diff.dot(mixed_bilinear_interaction_SU3[site_index][i] * spins_SU2[partner_idx]);
        }
        
        // Trilinear SU(3)-SU(3)-SU(3) interactions
        double trilinear_energy = 0.0;
        for (size_t i = 0; i < num_tri_SU3; ++i) {
            const size_t p1_idx = trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = trilinear_partners_SU3[site_index][i][1];
            const auto& T = trilinear_interaction_SU3[site_index][i];
            
            // Contract: sum_abc T[a](b,c) * spin_diff[a] * S1[b] * S2[c]
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU3; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * spins_SU3[p1_idx](b) * spins_SU3[p2_idx](c);
                    }
                }
                trilinear_energy += spin_diff(a) * temp;
            }
        }
        
        // Mixed trilinear SU(3)-SU(2)-SU(2) interactions
        double mixed_trilinear_energy = 0.0;
        for (size_t i = 0; i < num_tri_SU2_SU3; ++i) {
            const size_t p1_idx = mixed_trilinear_partners_SU3[site_index][i][0];
            const size_t p2_idx = mixed_trilinear_partners_SU3[site_index][i][1];
            const auto& T = mixed_trilinear_interaction_SU3[site_index][i];
            
            // Contract: sum_abc T[a](b,c) * spin_diff[a] * SU2_1[b] * SU2_2[c]
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * spins_SU2[p1_idx](b) * spins_SU2[p2_idx](c);
                    }
                }
                mixed_trilinear_energy += spin_diff(a) * temp;
            }
        }
        
        return field_energy + onsite_energy + bilinear_energy + mixed_bilinear_energy + 
               trilinear_energy + mixed_trilinear_energy;
    }

    /**
     * Compute total energy of the system
     */
    double total_energy() const {
        double energy = 0.0;
        
        // SU(2) contributions
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            const auto& spin = spins_SU2[i];
            
            // Field and onsite
            energy -= spin.dot(field_SU2[i]);
            energy += spin.dot(onsite_interaction_SU2[i] * spin);
            
            // Bilinear
            for (size_t j = 0; j < num_bi_SU2; ++j) {
                const size_t partner = bilinear_partners_SU2[i][j];
                energy += 0.5 * spin.dot(bilinear_interaction_SU2[i][j] * spins_SU2[partner]);
            }
            
            // Mixed bilinear
            for (size_t j = 0; j < num_bi_SU2_SU3; ++j) {
                const size_t partner = mixed_bilinear_partners_SU2[i][j];
                energy += 0.5 * spin.dot(mixed_bilinear_interaction_SU2[i][j] * spins_SU3[partner]);
            }
            
            // Trilinear
            for (size_t j = 0; j < num_tri_SU2; ++j) {
                const size_t p1 = trilinear_partners_SU2[i][j][0];
                const size_t p2 = trilinear_partners_SU2[i][j][1];
                const auto& T = trilinear_interaction_SU2[i][j];
                
                // Proper tensor contraction: sum_abc T[a](b,c) * S_i[a] * S1[b] * S2[c]
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
        
        // SU(3) contributions
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            const auto& spin = spins_SU3[i];
            
            // Field and onsite
            energy -= spin.dot(field_SU3[i]);
            energy += spin.dot(onsite_interaction_SU3[i] * spin);
            
            // Bilinear
            for (size_t j = 0; j < num_bi_SU3; ++j) {
                const size_t partner = bilinear_partners_SU3[i][j];
                energy += 0.5 * spin.dot(bilinear_interaction_SU3[i][j] * spins_SU3[partner]);
            }
            
            // Mixed bilinear
            for (size_t j = 0; j < num_bi_SU2_SU3; ++j) {
                const size_t partner = mixed_bilinear_partners_SU3[i][j];
                energy += 0.5 * spin.dot(mixed_bilinear_interaction_SU3[i][j] * spins_SU2[partner]);
            }
            
            // Trilinear
            for (size_t j = 0; j < num_tri_SU3; ++j) {
                const size_t p1 = trilinear_partners_SU3[i][j][0];
                const size_t p2 = trilinear_partners_SU3[i][j][1];
                const auto& T = trilinear_interaction_SU3[i][j];
                
                // Proper tensor contraction: sum_abc T[a](b,c) * S_i[a] * S1[b] * S2[c]
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
     * Compute energy density
     */
    double energy_density() const {
        return total_energy() / (lattice_size_SU2 + lattice_size_SU3);
    }

    // ============================================================
    // MONTE CARLO METHODS
    // ============================================================

    /**
     * Single Metropolis sweep over both sublattices
     */
    double metropolis(double T, bool gaussian_move = false, double sigma = 60.0) {
        size_t accepted = 0;
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        
        // Sweep SU(2) sublattice
        for (size_t n = 0; n < lattice_size_SU2; ++n) {
            size_t i = random_int_lehman(lattice_size_SU2);
            
            SpinVector new_spin;
            if (gaussian_move) {
                new_spin = spins_SU2[i] + gen_random_spin(sigma, spin_dim_SU2);
                new_spin *= spin_length_SU2 / new_spin.norm();
            } else {
                new_spin = gen_random_spin(spin_length_SU2, spin_dim_SU2);
            }
            
            double dE = site_energy_SU2_diff(new_spin, spins_SU2[i], i);
            
            if (dE <= 0 || random_double_lehman(0, 1) < exp(-dE / T)) {
                spins_SU2[i] = new_spin;
                accepted++;
            }
        }
        
        // Sweep SU(3) sublattice
        for (size_t n = 0; n < lattice_size_SU3; ++n) {
            size_t i = random_int_lehman(lattice_size_SU3);
            
            SpinVector new_spin;
            if (gaussian_move) {
                new_spin = spins_SU3[i] + gen_random_spin(sigma, spin_dim_SU3);
                new_spin *= spin_length_SU3 / new_spin.norm();
            } else {
                new_spin = gen_random_spin(spin_length_SU3, spin_dim_SU3);
            }
            
            double dE = site_energy_SU3_diff(new_spin, spins_SU3[i], i);
            
            if (dE <= 0 || random_double_lehman(0, 1) < exp(-dE / T)) {
                spins_SU3[i] = new_spin;
                accepted++;
            }
        }
        
        return double(accepted) / double(total_sites);
    }

    /**
     * Simulated annealing
     */
    void simulated_annealing(double T_start, double T_end, size_t n_anneal,
                            bool gaussian_move = true, const string& out_dir = "") {
        if (!out_dir.empty()) {
            std::filesystem::create_directory(out_dir);
        }
        
        double T = T_start;
        double sigma = 1000.0;
        
        cout << "Starting simulated annealing: " << T_start << " → " << T_end << endl;
        
        while (T > T_end) {
            double acceptance = 0.0;
            for (size_t i = 0; i < n_anneal; ++i) {
                acceptance += metropolis(T, gaussian_move, sigma);
            }
            acceptance /= n_anneal;
            
            cout << "T = " << T << ", acceptance = " << acceptance << endl;
            
            if (gaussian_move && acceptance < 0.5) {
                sigma *= 0.5 / (1 - acceptance);
                cout << "  Adjusted sigma = " << sigma << endl;
            }
            
            T *= 0.9;
        }
        
        if (!out_dir.empty()) {
            save_spin_config(out_dir + "/spin_final");
            save_positions(out_dir + "/pos");
        }
    }

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
        for (size_t i = 0; i < num_bi_SU2; ++i) {
            H += bilinear_interaction_SU2[site_index][i] * spins_SU2[bilinear_partners_SU2[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
            H += mixed_bilinear_interaction_SU2[site_index][i] * spins_SU3[mixed_bilinear_partners_SU2[site_index][i]];
        }
        
        // Trilinear SU(2)-SU(2)-SU(2) contributions
        for (size_t i = 0; i < num_tri_SU2; ++i) {
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
        for (size_t i = 0; i < num_tri_SU2_SU3; ++i) {
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
        for (size_t i = 0; i < num_bi_SU3; ++i) {
            H += bilinear_interaction_SU3[site_index][i] * spins_SU3[bilinear_partners_SU3[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
            H += mixed_bilinear_interaction_SU3[site_index][i] * spins_SU2[mixed_bilinear_partners_SU3[site_index][i]];
        }
        
        // Trilinear SU(3)-SU(3)-SU(3) contributions
        for (size_t i = 0; i < num_tri_SU3; ++i) {
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
        for (size_t i = 0; i < num_tri_SU2_SU3; ++i) {
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
     * Set pulse parameters for SU(2)
     */
    void set_pulse_SU2(const vector<SpinVector>& field_in1, double t_B1,
                      const vector<SpinVector>& field_in2, double t_B2,
                      double amp, double width, double freq) {
        // Pack field vectors
        field_drive_SU2[0] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_SU2[1] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        
        for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
            field_drive_SU2[0].segment(atom * spin_dim_SU2, spin_dim_SU2) = field_in1[atom];
            field_drive_SU2[1].segment(atom * spin_dim_SU2, spin_dim_SU2) = field_in2[atom];
        }
        
        t_pulse_SU2[0] = t_B1;
        t_pulse_SU2[1] = t_B2;
        field_drive_amp_SU2 = amp;
        field_drive_width_SU2 = width;
        field_drive_freq_SU2 = freq;
    }

    /**
     * Set pulse parameters for SU(3)
     */
    void set_pulse_SU3(const vector<SpinVector>& field_in1, double t_B1,
                      const vector<SpinVector>& field_in2, double t_B2,
                      double amp, double width, double freq) {
        // Pack field vectors
        field_drive_SU3[0] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_SU3[1] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        
        for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
            field_drive_SU3[0].segment(atom * spin_dim_SU3, spin_dim_SU3) = field_in1[atom];
            field_drive_SU3[1].segment(atom * spin_dim_SU3, spin_dim_SU3) = field_in2[atom];
        }
        
        t_pulse_SU3[0] = t_B1;
        t_pulse_SU3[1] = t_B2;
        field_drive_amp_SU3 = amp;
        field_drive_width_SU3 = width;
        field_drive_freq_SU3 = freq;
    }

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
     * Compute time-dependent drive field for SU(2) site
     */
    SpinVector drive_field_SU2_at_time(double t, size_t site_index) const {
        const size_t atom = site_index % N_atoms_SU2;
        SpinVector result = SpinVector::Zero(spin_dim_SU2);
        
        // Two Gaussian pulses
        for (int pulse = 0; pulse < 2; ++pulse) {
            double dt = t - t_pulse_SU2[pulse];
            double envelope = exp(-pow(dt / (2 * field_drive_width_SU2), 2));
            double oscillation = cos(2 * M_PI * field_drive_freq_SU2 * dt);
            double factor = field_drive_amp_SU2 * envelope * oscillation;
            
            result += factor * field_drive_SU2[pulse].segment(atom * spin_dim_SU2, spin_dim_SU2);
        }
        
        return result;
    }

    /**
     * Compute time-dependent drive field for SU(3) site
     */
    SpinVector drive_field_SU3_at_time(double t, size_t site_index) const {
        const size_t atom = site_index % N_atoms_SU3;
        SpinVector result = SpinVector::Zero(spin_dim_SU3);
        
        // Two Gaussian pulses
        for (int pulse = 0; pulse < 2; ++pulse) {
            double dt = t - t_pulse_SU3[pulse];
            double envelope = exp(-pow(dt / (2 * field_drive_width_SU3), 2));
            double oscillation = cos(2 * M_PI * field_drive_freq_SU3 * dt);
            double factor = field_drive_amp_SU3 * envelope * oscillation;
            
            result += factor * field_drive_SU3[pulse].segment(atom * spin_dim_SU3, spin_dim_SU3);
        }
        
        return result;
    }

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
        size_t idx = 0;
        // Unpack SU(2) spins
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            for (size_t j = 0; j < spin_dim_SU2; ++j) {
                spins2[i](j) = state[idx++];
            }
        }
        // Unpack SU(3) spins
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
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
     * @param method Integration method: "euler", "rk2", "rk4", "rk5", "dopri5" (default), "rk78", "bulirsch_stoer"
     * @param use_adaptive Use adaptive stepping (if method supports it)
     * @param abs_tol Absolute tolerance for adaptive methods
     * @param rel_tol Relative tolerance for adaptive methods
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
        } else {
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

    /**
     * ODE system function for Boost.Odeint
     */
    void ode_system(const ODEState& x, ODEState& dxdt, double t) {
        landau_lifshitz(x, dxdt, t);
    }

    /**
     * Compute Landau-Lifshitz derivatives using structure constants
     * Note: For SU(2), use cross product. For SU(3), use Gell-Mann structure constants.
     */
    void landau_lifshitz(const ODEState& state, ODEState& dsdt, double t) {
        // Unpack state to spin configurations
        SpinConfigSU2 curr_spins_SU2(lattice_size_SU2);
        SpinConfigSU3 curr_spins_SU3(lattice_size_SU3);
        for (size_t i = 0; i < lattice_size_SU2; ++i) curr_spins_SU2[i] = SpinVector::Zero(spin_dim_SU2);
        for (size_t i = 0; i < lattice_size_SU3; ++i) curr_spins_SU3[i] = SpinVector::Zero(spin_dim_SU3);
        state_to_spins(state, curr_spins_SU2, curr_spins_SU3);
        
        size_t idx = 0;
        
        // SU(2) dynamics: dS/dt = H × S (cross product for spin-1/2)
        for (size_t i = 0; i < lattice_size_SU2; ++i) {
            SpinVector H = get_local_field_SU2_state(i, curr_spins_SU2, curr_spins_SU3);
            H -= drive_field_SU2_at_time(t, i);
            
            // Cross product for SU(2): (H × S)
            if (spin_dim_SU2 == 3) {
                SpinVector dSdt = H.cross(curr_spins_SU2[i]);
                for (size_t j = 0; j < 3; ++j) {
                    dsdt[idx++] = dSdt(j);
                }
            } else {
                // General case: would need structure constants
                for (size_t j = 0; j < spin_dim_SU2; ++j) {
                    dsdt[idx++] = 0.0;
                }
            }
        }
        
        // SU(3) dynamics: dS/dt = f_ijk H_j S_k (structure constant contraction)
        // For SU(3), f_ijk are the Gell-Mann structure constants
        for (size_t i = 0; i < lattice_size_SU3; ++i) {
            SpinVector H = get_local_field_SU3_state(i, curr_spins_SU2, curr_spins_SU3);
            H -= drive_field_SU3_at_time(t, i);
            
            // Structure constant contraction: (f_ijk H_j S_k)
            SpinVector dSdt = SpinVector::Zero(spin_dim_SU3);
            for (size_t ii = 0; ii < spin_dim_SU3; ++ii) {
                for (size_t jj = 0; jj < spin_dim_SU3; ++jj) {
                    for (size_t kk = 0; kk < spin_dim_SU3; ++kk) {
                        dSdt(ii) += SU3_structure[ii][jj][kk] * H(jj) * curr_spins_SU3[i](kk);
                    }
                }
            }
            
            for (size_t j = 0; j < spin_dim_SU3; ++j) {
                dsdt[idx++] = dSdt(j);
            }
        }
    }

private:
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
        for (size_t i = 0; i < num_bi_SU2; ++i) {
            H += bilinear_interaction_SU2[site_index][i] * curr_spins2[bilinear_partners_SU2[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
            H += mixed_bilinear_interaction_SU2[site_index][i] * curr_spins3[mixed_bilinear_partners_SU2[site_index][i]];
        }
        
        // Trilinear SU(2)-SU(2)-SU(2) contributions
        for (size_t i = 0; i < num_tri_SU2; ++i) {
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
        for (size_t i = 0; i < num_tri_SU2_SU3; ++i) {
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
        for (size_t i = 0; i < num_bi_SU3; ++i) {
            H += bilinear_interaction_SU3[site_index][i] * curr_spins3[bilinear_partners_SU3[site_index][i]];
        }
        
        // Mixed bilinear
        for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
            H += mixed_bilinear_interaction_SU3[site_index][i] * curr_spins2[mixed_bilinear_partners_SU3[site_index][i]];
        }
        
        // Trilinear SU(3)-SU(3)-SU(3) contributions
        for (size_t i = 0; i < num_tri_SU3; ++i) {
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
        for (size_t i = 0; i < num_tri_SU2_SU3; ++i) {
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
     * Save site positions
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
    vector<pair<double, pair<pair<SpinVector, SpinVector>, pair<SpinVector, SpinVector>>>> M_B_t(
               const vector<SpinVector>& field_in_SU2, const vector<SpinVector>& field_in_SU3,
               double t_B,
               double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
               double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
               double T_start, double T_end, double step_size,
               const string& method = "dopri5", bool use_gpu = false) {
        
        if (use_gpu) {
#if defined(CUDA_ENABLED) && defined(__CUDACC__)
            return M_B_t_gpu(field_in_SU2, field_in_SU3, t_B,
                            pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                            pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                            T_start, T_end, step_size, method);
#else
            std::cerr << "Warning: GPU support not available. Falling back to CPU." << endl;
            // Fall through to CPU implementation
#endif
        }
        
        // Set up pulse
        set_pulse_SU2(field_in_SU2, t_B, vector<SpinVector>(N_atoms_SU2, SpinVector::Zero(spin_dim_SU2)),
                     0.0, pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2);
        set_pulse_SU3(field_in_SU3, t_B, vector<SpinVector>(N_atoms_SU3, SpinVector::Zero(spin_dim_SU3)),
                     0.0, pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3);
        
        // Storage for trajectory: (time, ((M_SU2_antiferro, M_SU2_local), (M_SU3_antiferro, M_SU3_local)))
        vector<pair<double, pair<pair<SpinVector, SpinVector>, pair<SpinVector, SpinVector>>>> trajectory;
        
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
                // Compute SU(2) magnetizations directly from flat state
                double M_SU2_local_arr[8] = {0};
                double M_SU2_antiferro_arr[8] = {0};
                size_t idx = 0;
                for (size_t i = 0; i < lattice_size_SU2; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    for (size_t d = 0; d < spin_dim_SU2; ++d) {
                        M_SU2_local_arr[d] += x[idx + d];
                        M_SU2_antiferro_arr[d] += x[idx + d] * sign;
                    }
                    idx += spin_dim_SU2;
                }
                
                // Compute SU(3) magnetizations
                double M_SU3_local_arr[8] = {0};
                double M_SU3_antiferro_arr[8] = {0};
                for (size_t i = 0; i < lattice_size_SU3; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    for (size_t d = 0; d < spin_dim_SU3; ++d) {
                        M_SU3_local_arr[d] += x[idx + d];
                        M_SU3_antiferro_arr[d] += x[idx + d] * sign;
                    }
                    idx += spin_dim_SU3;
                }
                
                SpinVector M_SU2_local = Eigen::Map<Eigen::VectorXd>(M_SU2_local_arr, spin_dim_SU2) / double(lattice_size_SU2);
                SpinVector M_SU2_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU2_antiferro_arr, spin_dim_SU2) / double(lattice_size_SU2);
                SpinVector M_SU3_local = Eigen::Map<Eigen::VectorXd>(M_SU3_local_arr, spin_dim_SU3) / double(lattice_size_SU3);
                SpinVector M_SU3_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU3_antiferro_arr, spin_dim_SU3) / double(lattice_size_SU3);
                
                trajectory.push_back({t, {{M_SU2_antiferro, M_SU2_local}, {M_SU3_antiferro, M_SU3_local}}});
                last_save_time = t;
            }
        };
        
        // Integrate
        integrate_ode_system(system_func, state, T_start, T_end, step_size,
                            observer, method, false, 1e-10, 1e-10);
        
        // Reset pulse
        reset_pulse();
        
        return trajectory;
    }

    /**
     * Molecular dynamics with two-pulse field
     * Returns magnetization trajectory without I/O
     */
    vector<pair<double, pair<pair<SpinVector, SpinVector>, pair<SpinVector, SpinVector>>>> M_BA_BB_t(
               const vector<SpinVector>& field_in_1_SU2, const vector<SpinVector>& field_in_1_SU3,
               double t_B_1,
               const vector<SpinVector>& field_in_2_SU2, const vector<SpinVector>& field_in_2_SU3,
               double t_B_2,
               double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
               double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
               double T_start, double T_end, double step_size,
               const string& method = "dopri5", bool use_gpu = false) {
        
        if (use_gpu) {
#if defined(CUDA_ENABLED) && defined(__CUDACC__)
            return M_BA_BB_t_gpu(field_in_1_SU2, field_in_1_SU3, t_B_1,
                                field_in_2_SU2, field_in_2_SU3, t_B_2,
                                pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                T_start, T_end, step_size, method);
#else
            std::cerr << "Warning: GPU support not available. Falling back to CPU." << endl;
            // Fall through to CPU implementation
#endif
        }
        
        // Set up two-pulse configuration
        set_pulse_SU2(field_in_1_SU2, t_B_1, field_in_2_SU2, t_B_2,
                     pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2);
        set_pulse_SU3(field_in_1_SU3, t_B_1, field_in_2_SU3, t_B_2,
                     pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3);
        
        // Storage for trajectory
        vector<pair<double, pair<pair<SpinVector, SpinVector>, pair<SpinVector, SpinVector>>>> trajectory;
        
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
                // Compute SU(2) magnetizations
                double M_SU2_local_arr[8] = {0};
                double M_SU2_antiferro_arr[8] = {0};
                size_t idx = 0;
                for (size_t i = 0; i < lattice_size_SU2; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    for (size_t d = 0; d < spin_dim_SU2; ++d) {
                        M_SU2_local_arr[d] += x[idx + d];
                        M_SU2_antiferro_arr[d] += x[idx + d] * sign;
                    }
                    idx += spin_dim_SU2;
                }
                
                // Compute SU(3) magnetizations
                double M_SU3_local_arr[8] = {0};
                double M_SU3_antiferro_arr[8] = {0};
                for (size_t i = 0; i < lattice_size_SU3; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    for (size_t d = 0; d < spin_dim_SU3; ++d) {
                        M_SU3_local_arr[d] += x[idx + d];
                        M_SU3_antiferro_arr[d] += x[idx + d] * sign;
                    }
                    idx += spin_dim_SU3;
                }
                
                SpinVector M_SU2_local = Eigen::Map<Eigen::VectorXd>(M_SU2_local_arr, spin_dim_SU2) / double(lattice_size_SU2);
                SpinVector M_SU2_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU2_antiferro_arr, spin_dim_SU2) / double(lattice_size_SU2);
                SpinVector M_SU3_local = Eigen::Map<Eigen::VectorXd>(M_SU3_local_arr, spin_dim_SU3) / double(lattice_size_SU3);
                SpinVector M_SU3_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU3_antiferro_arr, spin_dim_SU3) / double(lattice_size_SU3);
                
                trajectory.push_back({t, {{M_SU2_antiferro, M_SU2_local}, {M_SU3_antiferro, M_SU3_local}}});
                last_save_time = t;
            }
        };
        
        // Integrate
        integrate_ode_system(system_func, state, T_start, T_end, step_size,
                            observer, method, false, 1e-10, 1e-10);
        
        // Reset pulse
        reset_pulse();
        
        return trajectory;
    }

    /**
     * Molecular dynamics using Boost.Odeint
     */
    void molecular_dynamics(double T_start, double T_end, double dt_initial,
                           const string& out_dir = "", size_t save_interval = 100,
                           const string& method = "dopri5", bool use_gpu = false) {
        if (use_gpu) {
#if defined(CUDA_ENABLED) && defined(__CUDACC__)
            molecular_dynamics_gpu(T_start, T_end, dt_initial, out_dir, save_interval, method);
#else
            std::cerr << "Warning: GPU support not available in this compilation unit." << endl;
            std::cerr << "GPU methods require CUDA compilation (.cu files)" << endl;
            std::cerr << "Falling back to CPU implementation." << endl;
            molecular_dynamics_cpu(T_start, T_end, dt_initial, out_dir, save_interval, method);
#endif
        } else {
            molecular_dynamics_cpu(T_start, T_end, dt_initial, out_dir, save_interval, method);
        }
    }

    /**
     * CPU implementation of molecular dynamics
     */
    void molecular_dynamics_cpu(double T_start, double T_end, double dt_initial,
                           const string& out_dir = "", size_t save_interval = 100,
                           const string& method = "dopri5") {
#ifndef HDF5_ENABLED
        std::cerr << "Error: HDF5 support is required for molecular dynamics output." << endl;
        std::cerr << "Please rebuild with -DHDF5_ENABLED flag and HDF5 libraries." << endl;
        return;
#else
        if (!out_dir.empty()) {
            std::filesystem::create_directory(out_dir);
        }
        
        cout << "Running mixed lattice molecular dynamics: t=" << T_start << " → " << T_end << endl;
        cout << "Integration method: " << method << endl;
        cout << "Initial step size: " << dt_initial << endl;
        
        // Convert current spins to flat state vector
        ODEState state = spins_to_state();
        
        // Create HDF5 writer with separate SU2 and SU3 groups
        std::unique_ptr<HDF5MixedMDWriter> hdf5_writer;
        if (!out_dir.empty()) {
            string hdf5_file = out_dir + "/trajectory.h5";
            cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
            hdf5_writer = std::make_unique<HDF5MixedMDWriter>(
                hdf5_file, 
                lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
                lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
                dim1, dim2, dim3, method, 
                dt_initial, T_start, T_end, save_interval, 
                spin_length_SU2, spin_length_SU3,
                &site_positions_SU2, &site_positions_SU3, 10000);
        }
        
        // Observer to save data at specified intervals
        size_t step_count = 0;
        size_t save_count = 0;
        auto observer = [&](const ODEState& x, double t) {
            if (step_count % save_interval == 0) {
                // Compute magnetizations directly from flat state
                SpinVector M_SU2 = SpinVector::Zero(spin_dim_SU2);
                SpinVector M_SU3 = SpinVector::Zero(spin_dim_SU3);
                SpinVector M_SU2_antiferro = SpinVector::Zero(spin_dim_SU2);
                SpinVector M_SU3_antiferro = SpinVector::Zero(spin_dim_SU3);
                
                size_t idx = 0;
                for (size_t i = 0; i < lattice_size_SU2; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    for (size_t d = 0; d < spin_dim_SU2; ++d) {
                        M_SU2(d) += x[idx + d];
                        M_SU2_antiferro(d) += x[idx + d] * sign;
                    }
                    idx += spin_dim_SU2;
                }
                M_SU2 /= double(lattice_size_SU2);
                M_SU2_antiferro /= double(lattice_size_SU2);
                
                for (size_t i = 0; i < lattice_size_SU3; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    for (size_t d = 0; d < spin_dim_SU3; ++d) {
                        M_SU3(d) += x[idx + d];
                        M_SU3_antiferro(d) += x[idx + d] * sign;
                    }
                    idx += spin_dim_SU3;
                }
                M_SU3 /= double(lattice_size_SU3);
                M_SU3_antiferro /= double(lattice_size_SU3);
                
                // Write to HDF5 directly from flat state (no conversion needed)
                if (hdf5_writer) {
                    hdf5_writer->write_flat_step(t, 
                                                M_SU2_antiferro, M_SU2, 
                                                M_SU3_antiferro, M_SU3,
                                                x.data());
                    save_count++;
                }
                
                // Progress output
                if (step_count % (save_interval * 10) == 0) {
                    cout << "t=" << t << ", |M_SU2|=" << M_SU2.norm() 
                         << ", |M_SU3|=" << M_SU3.norm() << endl;
                }
            }
            ++step_count;
        };
        
        // Create ODE system wrapper
        auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
            this->ode_system(x, dxdt, t);
        };
        
        // Integrate using selected method
        double abs_tol = (method == "bulirsch_stoer") ? 1e-8 : 1e-6;
        double rel_tol = (method == "bulirsch_stoer") ? 1e-8 : 1e-6;
        integrate_ode_system(system_func, state, T_start, T_end, dt_initial,
                            observer, method, true, abs_tol, rel_tol);
        
        // Close HDF5 file
        if (hdf5_writer) {
            hdf5_writer->close();
            cout << "HDF5 trajectory saved with " << save_count << " snapshots" << endl;
        }
        
        cout << "Molecular dynamics complete! (" << step_count << " steps)" << endl;
#endif // HDF5_ENABLED
    }

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

    /**
     * Complete pump-probe nonlinear spectroscopy workflow for mixed lattice
     * Handles both SU(2) and SU(3) sublattices with consistent nomenclature
     * 
     * NOTE: Ground state should be prepared beforehand via simulated_annealing()
     *       or loaded from file before calling this method.
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
                                 string method = "dopri5", bool use_gpu = false) {
        
        std::filesystem::create_directories(dir_name);
        
        cout << "\n==========================================" << endl;
        cout << "Mixed Lattice Pump-Probe Spectroscopy" << endl;
        cout << "==========================================" << endl;
        cout << "SU(2) Pulse: amp=" << pulse_amp_SU2 << ", width=" << pulse_width_SU2 
             << ", freq=" << pulse_freq_SU2 << endl;
        cout << "SU(3) Pulse: amp=" << pulse_amp_SU3 << ", width=" << pulse_width_SU3 
             << ", freq=" << pulse_freq_SU3 << endl;
        cout << "Delay scan: " << tau_start << " → " << tau_end << " (step: " << tau_step << ")" << endl;
        cout << "Integration: " << T_start << " → " << T_end << " (step: " << T_step << ")" << endl;
        
        // Use current spin configuration as ground state (assumed pre-loaded)
        cout << "\n[1/3] Using current configuration as ground state..." << endl;
        double E_ground = energy_density();
        SpinVector M_ground_SU2 = magnetization_SU2();
        SpinVector M_ground_SU3 = magnetization_SU3();
        cout << "  Ground state: E/N = " << E_ground << endl;
        cout << "    |M_SU2| = " << M_ground_SU2.norm() << endl;
        cout << "    |M_SU3| = " << M_ground_SU3.norm() << endl;
        
        // Save initial configuration
        save_positions(dir_name + "/positions.txt");
        save_spin_config(dir_name + "/spins_initial.txt");
        
        // Backup ground state
        SpinConfigSU2 ground_state_SU2 = spins_SU2;
        SpinConfigSU3 ground_state_SU3 = spins_SU3;
        
        // Step 2: Reference single-pulse dynamics (pump at t=0)
        cout << "\n[2/3] Running reference single-pulse dynamics (M0)..." << endl;
        if (use_gpu) cout << "  Using GPU acceleration" << endl;
        auto M0_trajectory = M_B_t(field_in_SU2, field_in_SU3, 0.0,
                                   pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                   pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                   T_start, T_end, T_step, method, use_gpu);
        
        // Step 3: Delay time scan
        int tau_steps = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
        cout << "\n[3/3] Scanning delay times (" << tau_steps << " steps)..." << endl;
        
        // Store trajectories
        typedef vector<pair<double, pair<pair<SpinVector, SpinVector>, pair<SpinVector, SpinVector>>>> TrajectoryType;
        vector<TrajectoryType> M1_trajectories;
        vector<TrajectoryType> M01_trajectories;
        vector<double> tau_values;
        
        M1_trajectories.reserve(tau_steps);
        M01_trajectories.reserve(tau_steps);
        tau_values.reserve(tau_steps);
        
        double current_tau = tau_start;
        for (int i = 0; i < tau_steps; ++i) {
            cout << "\n--- Delay " << (i+1) << "/" << tau_steps << ": tau = " << current_tau << " ---" << endl;
            
            tau_values.push_back(current_tau);
            
            // Restore ground state
            spins_SU2 = ground_state_SU2;
            spins_SU3 = ground_state_SU3;
            
            // M1: Probe pulse only at tau
            cout << "  Computing M1 (probe at tau)..." << endl;
            auto M1_traj = M_B_t(field_in_SU2, field_in_SU3, current_tau,
                                pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                T_start, T_end, T_step, method, use_gpu);
            M1_trajectories.push_back(M1_traj);
            
            // Restore ground state
            spins_SU2 = ground_state_SU2;
            spins_SU3 = ground_state_SU3;
            
            // M01: Pump at 0 + Probe at tau
            cout << "  Computing M01 (pump + probe)..." << endl;
            auto M01_traj = M_BA_BB_t(field_in_SU2, field_in_SU3, 0.0,
                                     field_in_SU2, field_in_SU3, current_tau,
                                     pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                     pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                     T_start, T_end, T_step, method, use_gpu);
            M01_trajectories.push_back(M01_traj);
            
            current_tau += tau_step;
        }
        
        // Write everything to a single HDF5 file with comprehensive metadata
        string hdf5_file = dir_name + "/pump_probe_spectroscopy.h5";
        cout << "\n[Complete] Writing all data to HDF5 file: " << hdf5_file << endl;
        
#ifdef HDF5_ENABLED
        try {
            // Create HDF5 writer with comprehensive metadata
            HDF5MixedPumpProbeWriter writer(
                hdf5_file,
                // Lattice parameters
                lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
                lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
                dim1, dim2, dim3,
                spin_length_SU2, spin_length_SU3,
                // Pulse parameters
                pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                // Time evolution
                T_start, T_end, T_step, method,
                // Delay scan
                tau_start, tau_end, tau_step,
                // Ground state info
                E_ground, M_ground_SU2, M_ground_SU3,
                Temp_start, Temp_end, n_anneal,
                T_zero_quench, quench_sweeps,
                // Optional data
                &field_in_SU2, &field_in_SU3,
                &site_positions_SU2, &site_positions_SU3
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
        cout << "Note: HDF5 support not enabled. Rebuild with -DHDF5_ENABLED flag." << endl;
        cout << "Trajectories stored in memory: " << tau_steps << " delay points" << endl;
        cout << "  M0:  " << M0_trajectory.size() << " time steps" << endl;
        cout << "  M1:  " << tau_steps << " trajectories" << endl;
        cout << "  M01: " << tau_steps << " trajectories" << endl;
#endif
        
        // Restore ground state at end
        spins_SU2 = ground_state_SU2;
        spins_SU3 = ground_state_SU3;
        
        cout << "\n==========================================" << endl;
        cout << "Pump-Probe Spectroscopy Complete!" << endl;
        cout << "Output directory: " << dir_name << endl;
        cout << "Total delay points: " << tau_steps << endl;
        cout << "==========================================" << endl;
    }

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
    
    /**
     * Transfer mixed lattice data to GPU
     */
    GPUMixedLatticeData transfer_mixed_lattice_data_to_gpu() const {
        GPUMixedLatticeData gpu_data;
        
        // Set sizes
        gpu_data.lattice_size_SU2 = lattice_SU2.lattice_size;
        gpu_data.lattice_size_SU3 = lattice_SU3.lattice_size;
        gpu_data.num_bi_SU2 = lattice_SU2.num_bi;
        gpu_data.num_tri_SU2 = lattice_SU2.num_tri;
        gpu_data.num_bi_SU3 = lattice_SU3.num_bi;
        gpu_data.num_tri_SU3 = lattice_SU3.num_tri;
        
        // Transfer SU(2) field data
        vector<double> flat_field_SU2;
        for (size_t i = 0; i < lattice_SU2.lattice_size; ++i) {
            for (size_t d = 0; d < 3; ++d) {
                flat_field_SU2.push_back(lattice_SU2.field[i](d));
            }
        }
        gpu_data.d_field_SU2 = thrust::device_vector<double>(flat_field_SU2.begin(), flat_field_SU2.end());
        
        // Transfer SU(3) field data
        vector<double> flat_field_SU3;
        for (size_t i = 0; i < lattice_SU3.lattice_size; ++i) {
            for (size_t d = 0; d < 8; ++d) {
                flat_field_SU3.push_back(lattice_SU3.field[i](d));
            }
        }
        gpu_data.d_field_SU3 = thrust::device_vector<double>(flat_field_SU3.begin(), flat_field_SU3.end());
        
        // Transfer SU(2) onsite interactions
        vector<double> flat_onsite_SU2;
        for (size_t i = 0; i < lattice_SU2.lattice_size; ++i) {
            for (size_t r = 0; r < 3; ++r) {
                for (size_t c = 0; c < 3; ++c) {
                    flat_onsite_SU2.push_back(lattice_SU2.onsite_interaction[i](r, c));
                }
            }
        }
        gpu_data.d_onsite_interaction_SU2 = thrust::device_vector<double>(flat_onsite_SU2.begin(), flat_onsite_SU2.end());
        
        // Transfer SU(3) onsite interactions
        vector<double> flat_onsite_SU3;
        for (size_t i = 0; i < lattice_SU3.lattice_size; ++i) {
            for (size_t r = 0; r < 8; ++r) {
                for (size_t c = 0; c < 8; ++c) {
                    flat_onsite_SU3.push_back(lattice_SU3.onsite_interaction[i](r, c));
                }
            }
        }
        gpu_data.d_onsite_interaction_SU3 = thrust::device_vector<double>(flat_onsite_SU3.begin(), flat_onsite_SU3.end());
        
        // Transfer SU(2) bilinear interactions
        vector<double> flat_bilinear_SU2;
        vector<size_t> flat_partners_SU2;
        vector<int8_t> flat_wrap_SU2;
        
        for (size_t i = 0; i < lattice_SU2.lattice_size; ++i) {
            for (size_t n = 0; n < lattice_SU2.num_bi; ++n) {
                if (n < lattice_SU2.bilinear_partners[i].size()) {
                    flat_partners_SU2.push_back(lattice_SU2.bilinear_partners[i][n]);
                    for (size_t r = 0; r < 3; ++r) {
                        for (size_t c = 0; c < 3; ++c) {
                            flat_bilinear_SU2.push_back(lattice_SU2.bilinear_interaction[i][n](r, c));
                        }
                    }
                    for (size_t d = 0; d < 3; ++d) {
                        flat_wrap_SU2.push_back(lattice_SU2.bilinear_wrap_dir[i][n][d]);
                    }
                }
            }
        }
        gpu_data.d_bilinear_interaction_SU2 = thrust::device_vector<double>(flat_bilinear_SU2.begin(), flat_bilinear_SU2.end());
        gpu_data.d_bilinear_partners_SU2 = thrust::device_vector<size_t>(flat_partners_SU2.begin(), flat_partners_SU2.end());
        gpu_data.d_bilinear_wrap_dir_SU2 = thrust::device_vector<int8_t>(flat_wrap_SU2.begin(), flat_wrap_SU2.end());
        
        // Transfer SU(3) bilinear interactions
        vector<double> flat_bilinear_SU3;
        vector<size_t> flat_partners_SU3;
        vector<int8_t> flat_wrap_SU3;
        
        for (size_t i = 0; i < lattice_SU3.lattice_size; ++i) {
            for (size_t n = 0; n < lattice_SU3.num_bi; ++n) {
                if (n < lattice_SU3.bilinear_partners[i].size()) {
                    flat_partners_SU3.push_back(lattice_SU3.bilinear_partners[i][n]);
                    for (size_t r = 0; r < 8; ++r) {
                        for (size_t c = 0; c < 8; ++c) {
                            flat_bilinear_SU3.push_back(lattice_SU3.bilinear_interaction[i][n](r, c));
                        }
                    }
                    for (size_t d = 0; d < 3; ++d) {
                        flat_wrap_SU3.push_back(lattice_SU3.bilinear_wrap_dir[i][n][d]);
                    }
                }
            }
        }
        gpu_data.d_bilinear_interaction_SU3 = thrust::device_vector<double>(flat_bilinear_SU3.begin(), flat_bilinear_SU3.end());
        gpu_data.d_bilinear_partners_SU3 = thrust::device_vector<size_t>(flat_partners_SU3.begin(), flat_partners_SU3.end());
        gpu_data.d_bilinear_wrap_dir_SU3 = thrust::device_vector<int8_t>(flat_wrap_SU3.begin(), flat_wrap_SU3.end());
        
        // Transfer mixed bilinear interactions
        vector<double> flat_mixed_bilinear;
        vector<size_t> flat_mixed_partners_SU2;
        vector<size_t> flat_mixed_partners_SU3;
        vector<int8_t> flat_mixed_wrap;
        
        gpu_data.num_mixed_bi = 0;
        if (!bilinear_SU2_SU3.empty() && !bilinear_SU2_SU3[0].empty()) {
            gpu_data.num_mixed_bi = bilinear_SU2_SU3[0].size();
            
            for (size_t i = 0; i < lattice_SU2.lattice_size; ++i) {
                for (size_t n = 0; n < bilinear_SU2_SU3[i].size(); ++n) {
                    flat_mixed_partners_SU2.push_back(i);
                    flat_mixed_partners_SU3.push_back(bilinear_partners_SU2_SU3[i][n]);
                    
                    for (size_t r = 0; r < 3; ++r) {
                        for (size_t c = 0; c < 8; ++c) {
                            flat_mixed_bilinear.push_back(bilinear_SU2_SU3[i][n](r, c));
                        }
                    }
                    
                    for (size_t d = 0; d < 3; ++d) {
                        flat_mixed_wrap.push_back(bilinear_wrap_dir_SU2_SU3[i][n][d]);
                    }
                }
            }
        }
        gpu_data.d_mixed_bilinear_interaction = thrust::device_vector<double>(flat_mixed_bilinear.begin(), flat_mixed_bilinear.end());
        gpu_data.d_mixed_bilinear_partners_SU2 = thrust::device_vector<size_t>(flat_mixed_partners_SU2.begin(), flat_mixed_partners_SU2.end());
        gpu_data.d_mixed_bilinear_partners_SU3 = thrust::device_vector<size_t>(flat_mixed_partners_SU3.begin(), flat_mixed_partners_SU3.end());
        gpu_data.d_mixed_bilinear_wrap_dir = thrust::device_vector<int8_t>(flat_mixed_wrap.begin(), flat_mixed_wrap.end());
        
        // Store pulse parameters
        gpu_data.field_drive_amp_SU2 = lattice_SU2.field_drive_amp;
        gpu_data.field_drive_freq_SU2 = lattice_SU2.field_drive_freq;
        gpu_data.field_drive_width_SU2 = lattice_SU2.field_drive_width;
        gpu_data.t_pulse_0_SU2 = lattice_SU2.t_pulse_0;
        gpu_data.t_pulse_1_SU2 = lattice_SU2.t_pulse_1;
        
        gpu_data.field_drive_amp_SU3 = lattice_SU3.field_drive_amp;
        gpu_data.field_drive_freq_SU3 = lattice_SU3.field_drive_freq;
        gpu_data.field_drive_width_SU3 = lattice_SU3.field_drive_width;
        gpu_data.t_pulse_0_SU3 = lattice_SU3.t_pulse_0;
        gpu_data.t_pulse_1_SU3 = lattice_SU3.t_pulse_1;
        
        return gpu_data;
    }
    
    /**
     * GPU ODE system function
     * For mixed lattice, this needs to handle both SU(2) and SU(3) spins
     */
    void ode_system_gpu(const thrust::device_vector<double>& x, 
                       thrust::device_vector<double>& dxdt, 
                       double t,
                       const GPUMixedLatticeData& d_data) const {
        // Placeholder: In full implementation, this would call CUDA kernels
        // For now, fall back to CPU computation
        thrust::host_vector<double> h_x = x;
        thrust::host_vector<double> h_dxdt(x.size());
        
        landau_lifshitz_flat(thrust::raw_pointer_cast(h_x.data()), 
                            thrust::raw_pointer_cast(h_dxdt.data()), t);
        
        dxdt = h_dxdt;
    }
    
    /**
     * GPU integration wrapper for mixed lattice
     */
    template<typename System, typename Observer>
    void integrate_ode_system_gpu(System system_func, thrust::device_vector<double>& state,
                                  double T_start, double T_end, double dt_step,
                                  Observer observer, const string& method,
                                  bool use_adaptive = false,
                                  double abs_tol = 1e-6, double rel_tol = 1e-6) {
        // For GPU integration, copy to host, integrate, and copy back
        thrust::host_vector<double> h_state = state;
        ODEState cpu_state(h_state.begin(), h_state.end());
        
        auto cpu_system = [&](const ODEState& x, ODEState& dxdt, double t) {
            thrust::device_vector<double> d_x(x.begin(), x.end());
            thrust::device_vector<double> d_dxdt(x.size());
            system_func(d_x, d_dxdt, t);
            thrust::host_vector<double> h_dxdt = d_dxdt;
            std::copy(h_dxdt.begin(), h_dxdt.end(), dxdt.begin());
        };
        
        auto cpu_observer = [&](const ODEState& x, double t) {
            thrust::device_vector<double> d_x(x.begin(), x.end());
            observer(d_x, t);
        };
        
        integrate_ode_system(cpu_system, cpu_state, T_start, T_end, dt_step,
                            cpu_observer, method, use_adaptive, abs_tol, rel_tol);
        
        h_state = thrust::host_vector<double>(cpu_state.begin(), cpu_state.end());
        state = h_state;
    }
    
    /**
     * GPU version of molecular_dynamics
     */
    void molecular_dynamics_gpu(double T_start, double T_end, double dt_initial,
                               const string& out_dir, size_t save_interval,
                               const string& method) {
        // Transfer data to GPU
        auto d_lattice_data = transfer_mixed_lattice_data_to_gpu();
        
        // Initial state on GPU
        ODEState state = spins_to_state(spins_SU2, spins_SU3);
        thrust::device_vector<double> d_state(state.begin(), state.end());
        
        // Setup HDF5 writer if directory specified
        HDF5MixedMDWriter* writer = nullptr;
        if (!out_dir.empty()) {
            string filename = out_dir + "/molecular_dynamics.h5";
            writer = new HDF5MixedMDWriter(filename, lattice_SU2.N_atoms, lattice_SU3.N_atoms);
        }
        
        // Observer
        size_t step_count = 0;
        auto observer = [&](const thrust::device_vector<double>& d_x, double t) {
            if (step_count % save_interval == 0) {
                thrust::host_vector<double> x = d_x;
                
                // Extract SU(2) and SU(3) spins
                size_t total_SU2 = lattice_SU2.lattice_size * 3;
                
                vector<SpinVector> current_spins_SU2(lattice_SU2.lattice_size);
                vector<SpinVector> current_spins_SU3(lattice_SU3.lattice_size);
                
                for (size_t i = 0; i < lattice_SU2.lattice_size; ++i) {
                    SpinVector spin(3);
                    for (size_t d = 0; d < 3; ++d) {
                        spin(d) = x[i * 3 + d];
                    }
                    current_spins_SU2[i] = spin;
                }
                
                for (size_t i = 0; i < lattice_SU3.lattice_size; ++i) {
                    SpinVector spin(8);
                    for (size_t d = 0; d < 8; ++d) {
                        spin(d) = x[total_SU2 + i * 8 + d];
                    }
                    current_spins_SU3[i] = spin;
                }
                
                if (writer) {
                    writer->write_trajectory_step(t, current_spins_SU2, current_spins_SU3);
                }
                
                cout << "t = " << t << endl;
            }
            step_count++;
        };
        
        // System function
        auto gpu_system_func = [this, &d_lattice_data](const thrust::device_vector<double>& x, 
                                                        thrust::device_vector<double>& dxdt, 
                                                        double t) {
            this->ode_system_gpu(x, dxdt, t, d_lattice_data);
        };
        
        // Integrate on GPU
        integrate_ode_system_gpu(gpu_system_func, d_state, T_start, T_end, dt_initial,
                                observer, method, false, 1e-10, 1e-10);
        
        // Cleanup
        if (writer) {
            delete writer;
        }
    }
    
    /**
     * GPU version of M_B_t for mixed lattice
     * Returns nested pair: outer=(SU2_results, SU3_results), 
     * inner for each = (M_antiferro, M_local)
     */
    vector<pair<double, pair<pair<SpinVector, SpinVector>, pair<SpinVector, SpinVector>>>>
    M_B_t_gpu(const vector<SpinVector>& field_in_SU2,
              const vector<SpinVector>& field_in_SU3,
              double t_B,
              double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
              double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
              double T_start, double T_end, double step_size,
              string method = "dopri5") {
        
        // Set up pulses for both sublattices
        lattice_SU2.set_pulse(field_in_SU2, t_B, 
                             vector<SpinVector>(lattice_SU2.N_atoms, SpinVector::Zero(3)), 0.0,
                             pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2);
        
        lattice_SU3.set_pulse(field_in_SU3, t_B,
                             vector<SpinVector>(lattice_SU3.N_atoms, SpinVector::Zero(8)), 0.0,
                             pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3);
        
        // Transfer data to GPU
        auto d_lattice_data = transfer_mixed_lattice_data_to_gpu();
        
        // Storage for trajectory
        vector<pair<double, pair<pair<SpinVector, SpinVector>, pair<SpinVector, SpinVector>>>> trajectory;
        
        // Initial state on GPU
        ODEState state = spins_to_state(spins_SU2, spins_SU3);
        thrust::device_vector<double> d_state(state.begin(), state.end());
        
        // Observer
        double last_save_time = T_start;
        auto observer = [&](const thrust::device_vector<double>& d_x, double t) {
            if (t - last_save_time >= step_size - 1e-10 || t >= T_end - 1e-10) {
                thrust::host_vector<double> x = d_x;
                
                size_t total_SU2 = lattice_SU2.lattice_size * 3;
                
                // Compute SU(2) magnetizations
                double M_local_SU2_arr[3] = {0};
                double M_antiferro_SU2_arr[3] = {0};
                
                for (size_t i = 0; i < lattice_SU2.lattice_size; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    for (size_t d = 0; d < 3; ++d) {
                        M_local_SU2_arr[d] += x[i * 3 + d];
                        M_antiferro_SU2_arr[d] += x[i * 3 + d] * sign;
                    }
                }
                
                SpinVector M_local_SU2 = Eigen::Map<Eigen::VectorXd>(M_local_SU2_arr, 3) / double(lattice_SU2.lattice_size);
                SpinVector M_antiferro_SU2 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU2_arr, 3) / double(lattice_SU2.lattice_size);
                
                // Compute SU(3) magnetizations
                double M_local_SU3_arr[8] = {0};
                double M_antiferro_SU3_arr[8] = {0};
                
                for (size_t i = 0; i < lattice_SU3.lattice_size; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    for (size_t d = 0; d < 8; ++d) {
                        M_local_SU3_arr[d] += x[total_SU2 + i * 8 + d];
                        M_antiferro_SU3_arr[d] += x[total_SU2 + i * 8 + d] * sign;
                    }
                }
                
                SpinVector M_local_SU3 = Eigen::Map<Eigen::VectorXd>(M_local_SU3_arr, 8) / double(lattice_SU3.lattice_size);
                SpinVector M_antiferro_SU3 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU3_arr, 8) / double(lattice_SU3.lattice_size);
                
                trajectory.push_back({t, {{M_antiferro_SU2, M_local_SU2}, {M_antiferro_SU3, M_local_SU3}}});
                last_save_time = t;
            }
        };
        
        // System function
        auto gpu_system_func = [this, &d_lattice_data](const thrust::device_vector<double>& x, 
                                                        thrust::device_vector<double>& dxdt, 
                                                        double t) {
            this->ode_system_gpu(x, dxdt, t, d_lattice_data);
        };
        
        // Integrate on GPU
        integrate_ode_system_gpu(gpu_system_func, d_state, T_start, T_end, step_size,
                                observer, method, false, 1e-10, 1e-10);
        
        // Reset pulses
        lattice_SU2.field_drive[0] = SpinVector::Zero(lattice_SU2.N_atoms * 3);
        lattice_SU2.field_drive[1] = SpinVector::Zero(lattice_SU2.N_atoms * 3);
        lattice_SU2.field_drive_amp = 0.0;
        
        lattice_SU3.field_drive[0] = SpinVector::Zero(lattice_SU3.N_atoms * 8);
        lattice_SU3.field_drive[1] = SpinVector::Zero(lattice_SU3.N_atoms * 8);
        lattice_SU3.field_drive_amp = 0.0;
        
        return trajectory;
    }
    
    /**
     * GPU version of M_BA_BB_t for mixed lattice
     */
    vector<pair<double, pair<pair<SpinVector, SpinVector>, pair<SpinVector, SpinVector>>>>
    M_BA_BB_t_gpu(const vector<SpinVector>& field_in_1_SU2,
                  const vector<SpinVector>& field_in_1_SU3,
                  double t_B_1,
                  const vector<SpinVector>& field_in_2_SU2,
                  const vector<SpinVector>& field_in_2_SU3,
                  double t_B_2,
                  double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                  double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                  double T_start, double T_end, double step_size,
                  string method = "dopri5") {
        
        // Set up two-pulse configuration for both sublattices
        lattice_SU2.set_pulse(field_in_1_SU2, t_B_1, field_in_2_SU2, t_B_2,
                             pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2);
        
        lattice_SU3.set_pulse(field_in_1_SU3, t_B_1, field_in_2_SU3, t_B_2,
                             pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3);
        
        // Transfer data to GPU
        auto d_lattice_data = transfer_mixed_lattice_data_to_gpu();
        
        // Storage for trajectory
        vector<pair<double, pair<pair<SpinVector, SpinVector>, pair<SpinVector, SpinVector>>>> trajectory;
        
        // Initial state on GPU
        ODEState state = spins_to_state(spins_SU2, spins_SU3);
        thrust::device_vector<double> d_state(state.begin(), state.end());
        
        // Observer
        double last_save_time = T_start;
        auto observer = [&](const thrust::device_vector<double>& d_x, double t) {
            if (t - last_save_time >= step_size - 1e-10 || t >= T_end - 1e-10) {
                thrust::host_vector<double> x = d_x;
                
                size_t total_SU2 = lattice_SU2.lattice_size * 3;
                
                // Compute SU(2) magnetizations
                double M_local_SU2_arr[3] = {0};
                double M_antiferro_SU2_arr[3] = {0};
                
                for (size_t i = 0; i < lattice_SU2.lattice_size; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    for (size_t d = 0; d < 3; ++d) {
                        M_local_SU2_arr[d] += x[i * 3 + d];
                        M_antiferro_SU2_arr[d] += x[i * 3 + d] * sign;
                    }
                }
                
                SpinVector M_local_SU2 = Eigen::Map<Eigen::VectorXd>(M_local_SU2_arr, 3) / double(lattice_SU2.lattice_size);
                SpinVector M_antiferro_SU2 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU2_arr, 3) / double(lattice_SU2.lattice_size);
                
                // Compute SU(3) magnetizations
                double M_local_SU3_arr[8] = {0};
                double M_antiferro_SU3_arr[8] = {0};
                
                for (size_t i = 0; i < lattice_SU3.lattice_size; ++i) {
                    double sign = (i % 2 == 0) ? 1.0 : -1.0;
                    for (size_t d = 0; d < 8; ++d) {
                        M_local_SU3_arr[d] += x[total_SU2 + i * 8 + d];
                        M_antiferro_SU3_arr[d] += x[total_SU2 + i * 8 + d] * sign;
                    }
                }
                
                SpinVector M_local_SU3 = Eigen::Map<Eigen::VectorXd>(M_local_SU3_arr, 8) / double(lattice_SU3.lattice_size);
                SpinVector M_antiferro_SU3 = Eigen::Map<Eigen::VectorXd>(M_antiferro_SU3_arr, 8) / double(lattice_SU3.lattice_size);
                
                trajectory.push_back({t, {{M_antiferro_SU2, M_local_SU2}, {M_antiferro_SU3, M_local_SU3}}});
                last_save_time = t;
            }
        };
        
        // System function
        auto gpu_system_func = [this, &d_lattice_data](const thrust::device_vector<double>& x, 
                                                        thrust::device_vector<double>& dxdt, 
                                                        double t) {
            this->ode_system_gpu(x, dxdt, t, d_lattice_data);
        };
        
        // Integrate on GPU
        integrate_ode_system_gpu(gpu_system_func, d_state, T_start, T_end, step_size,
                                observer, method, false, 1e-10, 1e-10);
        
        // Reset pulses
        lattice_SU2.field_drive[0] = SpinVector::Zero(lattice_SU2.N_atoms * 3);
        lattice_SU2.field_drive[1] = SpinVector::Zero(lattice_SU2.N_atoms * 3);
        lattice_SU2.field_drive_amp = 0.0;
        
        lattice_SU3.field_drive[0] = SpinVector::Zero(lattice_SU3.N_atoms * 8);
        lattice_SU3.field_drive[1] = SpinVector::Zero(lattice_SU3.N_atoms * 8);
        lattice_SU3.field_drive_amp = 0.0;
        
        return trajectory;
    }
#endif // defined(CUDA_ENABLED) && defined(__CUDACC__)
};

#endif // MIXED_LATTICE_REFACTORED_H
