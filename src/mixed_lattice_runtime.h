#ifndef MIXED_LATTICE_RUNTIME_H
#define MIXED_LATTICE_RUNTIME_H

#define _USE_MATH_DEFINES
#include "unitcell_runtime.h"
#include "lattice_config.h"
#include "simple_linear_alg.h"
#include <cmath>
#include <cstring>
#include <numbers>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <random>
#include <chrono>
#include <math.h>
#include <tuple>
#include <mpi.h>
#include <atomic>
#include <mutex>
#include "cuda_contractions_wrapper.cuh"
#include "convergence_monitor.h"

// Performance monitoring utilities
struct PerformanceTimer {
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;
    bool verbose;
    
    PerformanceTimer(const std::string& timer_name, bool verbose_output = true) 
        : name(timer_name), verbose(verbose_output) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    ~PerformanceTimer() {
        if (verbose) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "[PERF] " << name << " took " << duration.count() << " ms" << std::endl;
        }
    }
    
    double elapsed_ms() const {
        auto current_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
    }
};

// Runtime version of mixed lattice spin configuration
struct MixedLatticeSpin {
    vector<vector<double>> spins_SU2;  // lattice_size_SU2 x N_SU2
    vector<vector<double>> spins_SU3;  // lattice_size_SU3 x N_SU3
    
    MixedLatticeSpin(size_t lattice_size_su2, size_t n_su2, 
                     size_t lattice_size_su3, size_t n_su3) {
        spins_SU2.resize(lattice_size_su2, vector<double>(n_su2, 0.0));
        spins_SU3.resize(lattice_size_su3, vector<double>(n_su3, 0.0));
    }
    
    MixedLatticeSpin(const vector<vector<double>>& spins_su2_in, 
                     const vector<vector<double>>& spins_su3_in) 
        : spins_SU2(spins_su2_in), spins_SU3(spins_su3_in) {
    }
    
    MixedLatticeSpin(const MixedLatticeSpin& spins_in) {
        spins_SU2 = spins_in.spins_SU2;
        spins_SU3 = spins_in.spins_SU3;
    }

    void set(MixedLatticeSpin& spins_in) noexcept {
        using std::swap;
        swap(spins_SU2, spins_in.spins_SU2);
        swap(spins_SU3, spins_in.spins_SU3);
    }

    double length() const {
        double result = 0;
        for (size_t i = 0; i < spins_SU2.size(); ++i) {
            for (size_t j = 0; j < spins_SU2[i].size(); ++j) {
                result += spins_SU2[i][j] * spins_SU2[i][j];
            }
        }
        for (size_t i = 0; i < spins_SU3.size(); ++i) {
            for (size_t j = 0; j < spins_SU3[i].size(); ++j) {
                result += spins_SU3[i][j] * spins_SU3[i][j];
            }
        }
        return sqrt(result);
    }
};

// Runtime version of mixed lattice positions
struct MixedLatticePos {
    vector<array<double,3>> pos_SU2;
    vector<array<double,3>> pos_SU3;
    
    MixedLatticePos(size_t lattice_size_su2, size_t lattice_size_su3) {
        pos_SU2.resize(lattice_size_su2, {0.0, 0.0, 0.0});
        pos_SU3.resize(lattice_size_su3, {0.0, 0.0, 0.0});
    }
    
    MixedLatticePos(const vector<array<double,3>>& pos_su2_in, 
                    const vector<array<double,3>>& pos_su3_in) 
        : pos_SU2(pos_su2_in), pos_SU3(pos_su3_in) {
    }
    
    MixedLatticePos(const MixedLatticePos& pos_in) {
        pos_SU2 = pos_in.pos_SU2;
        pos_SU3 = pos_in.pos_SU3;
    }
};

// Runtime version of mixed lattice class
class MixedLattice {
public:
    typedef vector<vector<double>> spin_config_SU2;
    typedef vector<vector<double>> spin_config_SU3;

    MixedLatticeConfig config;
    MixedUnitCell UC;
    size_t lattice_size_SU2;
    size_t lattice_size_SU3;
    double spin_length_SU2, spin_length_SU3;
    MixedLatticeSpin spins;
    MixedLatticePos site_pos;

    double field_drive_freq_SU2;
    double field_drive_amp_SU2;
    double field_drive_width_SU2;
    double t_B_1_SU2;
    double t_B_2_SU2;
    double field_drive_freq_SU3;
    double field_drive_amp_SU3;
    double field_drive_width_SU3;
    double t_B_1_SU3;
    double t_B_2_SU3;

    double field_drive_freq_SU2_SU3;
    double field_drive_amp_SU2_SU3;
    double field_drive_width_SU2_SU3;
    double t_B_1_SU2_SU3;
    double t_B_2_SU2_SU3;

    // Look up table for SU2
    vector<vector<double>> field_SU2;  // lattice_size_SU2 x N_SU2
    vector<vector<double>> field_drive_1_SU2;  // N_ATOMS_SU2 x N_SU2
    vector<vector<double>> field_drive_2_SU2;  // N_ATOMS_SU2 x N_SU2

    vector<vector<double>> onsite_interaction_SU2;  // lattice_size_SU2 x (N_SU2*N_SU2)
    vector<vector<vector<double>>> bilinear_interaction_SU2;  // lattice_size_SU2 x num_bi x (N_SU2*N_SU2)
    vector<vector<vector<double>>> trilinear_interaction_SU2;  // lattice_size_SU2 x num_tri x (N_SU2*N_SU2*N_SU2)

    vector<vector<size_t>> bilinear_partners_SU2;  // lattice_size_SU2 x num_bi
    vector<vector<array<size_t, 2>>> trilinear_partners_SU2;  // lattice_size_SU2 x num_tri

    // Look up table for SU3
    vector<vector<double>> field_SU3;  // lattice_size_SU3 x N_SU3
    vector<vector<double>> field_drive_1_SU3;  // N_ATOMS_SU3 x N_SU3
    vector<vector<double>> field_drive_2_SU3;  // N_ATOMS_SU3 x N_SU3

    vector<vector<double>> onsite_interaction_SU3;  // lattice_size_SU3 x (N_SU3*N_SU3)
    vector<vector<vector<double>>> bilinear_interaction_SU3;  // lattice_size_SU3 x num_bi x (N_SU3*N_SU3)
    vector<vector<vector<double>>> trilinear_interaction_SU3;  // lattice_size_SU3 x num_tri x (N_SU3*N_SU3*N_SU3)

    vector<vector<size_t>> bilinear_partners_SU3;  // lattice_size_SU3 x num_bi
    vector<vector<array<size_t, 2>>> trilinear_partners_SU3;  // lattice_size_SU3 x num_tri

    // Look up table for SU2 and SU3 mix
    vector<vector<vector<double>>> mixed_bilinear_interaction_SU2;  // lattice_size_SU2 x num_bi_mixed x (N_SU2*N_SU3)
    vector<vector<vector<double>>> mixed_bilinear_interaction_SU3;  // lattice_size_SU3 x num_bi_mixed x (N_SU2*N_SU3)
    vector<vector<size_t>> mixed_bilinear_partners_SU2;  // lattice_size_SU2 x num_bi_mixed
    vector<vector<size_t>> mixed_bilinear_partners_SU3;  // lattice_size_SU3 x num_bi_mixed

    vector<vector<double>> field_drive_1_SU2_SU3;  // N_ATOMS_SU2 x N_SU2
    vector<vector<double>> field_drive_2_SU2_SU3;  // N_ATOMS_SU2 x N_SU2

    vector<vector<vector<double>>> mixed_trilinear_interaction_SU2;  // lattice_size_SU2 x num_tri_mixed x (N_SU2*N_SU2*N_SU3)
    vector<vector<vector<double>>> mixed_trilinear_interaction_SU3;  // lattice_size_SU3 x num_tri_mixed x (N_SU2*N_SU2*N_SU3)

    vector<vector<array<size_t, 2>>> mixed_trilinear_partners_SU2;  // lattice_size_SU2 x num_tri_mixed
    vector<vector<array<size_t, 2>>> mixed_trilinear_partners_SU3;  // lattice_size_SU3 x num_tri_mixed

    vector<double> SU2_structure_tensor;
    vector<double> SU3_structure_tensor;

    size_t num_bi_SU2;
    size_t num_tri_SU2;
    size_t num_bi_SU3;
    size_t num_tri_SU3;
    size_t num_bi_SU2_SU3;
    size_t num_tri_SU2_SU3;

    DeviceStructureTensorManager device_structure_tensor_manager;

    vector<vector<vector<double>>> sublattice_frames_SU2;  // N_ATOMS_SU2 x N_SU2 x N_SU2
    vector<vector<vector<double>>> sublattice_frames_SU3;  // N_ATOMS_SU3 x N_SU3 x N_SU3

    void gen_random_spin(vector<double>& temp_spin, double spin_l, size_t N) {
        temp_spin.resize(N);
        vector<double> euler_angles(N - 2);
        double z = random_double_lehman(-1, 1);
        double r = sqrt(1.0 - z*z);

        for(size_t i = 0; i < N - 2; ++i) {
            euler_angles[i] = random_double_lehman(0, 2*M_PI);
            temp_spin[i] = r;
            for(size_t j = 0; j < i; ++j) {
                temp_spin[i] *= sin(euler_angles[j]);
            }
            if (i == N - 3) {
                temp_spin[i+1] = temp_spin[i] * sin(euler_angles[i]);
            }
            temp_spin[i] *= cos(euler_angles[i]);
        }
        temp_spin[N-1] = z;
        for(size_t i = 0; i < N; ++i) {
            temp_spin[i] *= spin_l;
        }
    }

    vector<double> gen_random_spin_SU2() {
        vector<double> temp_spin;
        gen_random_spin(temp_spin, spin_length_SU2, config.N_SU2);
        return temp_spin;
    }

    vector<double> gen_random_spin_SU3() {
        vector<double> temp_spin;
        gen_random_spin(temp_spin, spin_length_SU3, config.N_SU3);
        return temp_spin;
    }

    size_t flatten_index(size_t i, size_t j, size_t k, size_t l, size_t N_ATOMS) {
        return i * config.dim2 * config.dim * N_ATOMS + 
               j * config.dim * N_ATOMS + 
               k * N_ATOMS + l;
    }
    
    size_t periodic_boundary(int i, size_t D) {
        if(i < 0) {
            return size_t((D + i) % D);
        }
        else {
            return size_t(i % D);
        }
    }

    size_t flatten_index_periodic_boundary(int i, int j, int k, int l, size_t N_ATOMS) {
        return periodic_boundary(i, config.dim1) * config.dim2 * config.dim * N_ATOMS + 
               periodic_boundary(j, config.dim2) * config.dim * N_ATOMS + 
               periodic_boundary(k, config.dim) * N_ATOMS + l;
    }

    void set_up_sublattice(double spin_length, vector<vector<double>>& spins_sub, 
                           vector<array<double,3>>& site_pos_sub,
                           vector<vector<double>>& field_sub, 
                           vector<vector<double>>& onsite_interaction_sub,
                           vector<vector<vector<double>>>& bilinear_interaction_sub,
                           vector<vector<vector<double>>>& trilinear_interaction_sub,
                           vector<vector<size_t>>& bilinear_partners_sub,
                           vector<vector<array<size_t, 2>>>& trilinear_partners_sub,
                           UnitCell* atoms, size_t& num_bi, size_t& num_tri, size_t N, size_t N_ATOMS) {
        
        size_t lattice_size = config.dim1 * config.dim2 * config.dim * N_ATOMS;
        
        // Initialize arrays
        spins_sub.resize(lattice_size, vector<double>(N));
        site_pos_sub.resize(lattice_size);
        field_sub.resize(lattice_size, vector<double>(N));
        onsite_interaction_sub.resize(lattice_size, vector<double>(N * N));
        bilinear_interaction_sub.resize(lattice_size);
        trilinear_interaction_sub.resize(lattice_size);
        bilinear_partners_sub.resize(lattice_size);
        trilinear_partners_sub.resize(lattice_size);

        const vector<array<double,3>>& basis = atoms->lattice_pos;
        const array<array<double,3>, 3>& unit_vector = atoms->lattice_vectors;

        // Phase 1: Initialize basic site properties
        #pragma omp parallel for collapse(4) schedule(static)
        for (size_t i = 0; i < config.dim1; ++i) {
            for (size_t j = 0; j < config.dim2; ++j) {
                for (size_t k = 0; k < config.dim; ++k) {
                    for (size_t l = 0; l < N_ATOMS; ++l) {
                        size_t current_site_index = flatten_index(i, j, k, l, N_ATOMS);
                        
                        // Calculate position
                        for (int d = 0; d < 3; d++) {
                            site_pos_sub[current_site_index][d] = 
                                unit_vector[0][d] * int(i) + 
                                unit_vector[1][d] * int(j) + 
                                unit_vector[2][d] * int(k) + 
                                basis[l][d];
                        }
                        
                        // Generate random spin
                        gen_random_spin(spins_sub[current_site_index], spin_length, N);
                        
                        // Copy field and onsite interaction
                        field_sub[current_site_index] = atoms->field[l];
                        onsite_interaction_sub[current_site_index] = atoms->onsite_interaction[l];
                    }
                }
            }
        }

        // Phase 2: Build interaction lists with mutex protection
        vector<std::mutex> site_mutexes(lattice_size);
        
        #pragma omp parallel for collapse(4) schedule(static)
        for (size_t i = 0; i < config.dim1; ++i) {
            for (size_t j = 0; j < config.dim2; ++j) {
                for (size_t k = 0; k < config.dim; ++k) {
                    for (size_t l = 0; l < N_ATOMS; ++l) {
                        size_t current_site_index = flatten_index(i, j, k, l, N_ATOMS);
                        
                        // Process bilinear interactions
                        auto bilinear_matched = atoms->bilinear_interaction.equal_range(l);
                        for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m) {
                            const Bilinear& J = m->second;
                            size_t partner = flatten_index_periodic_boundary(
                                int(i) + J.offset[0], 
                                int(j) + J.offset[1], 
                                int(k) + J.offset[2], 
                                J.partner, N_ATOMS);
                            
                            {
                                std::lock_guard<std::mutex> lock(site_mutexes[current_site_index]);
                                bilinear_interaction_sub[current_site_index].push_back(J.bilinear_interaction);
                                bilinear_partners_sub[current_site_index].push_back(partner);
                            }
                            
                            {
                                std::lock_guard<std::mutex> lock(site_mutexes[partner]);
                                bilinear_interaction_sub[partner].push_back(J.bilinear_interaction);
                                bilinear_partners_sub[partner].push_back(current_site_index);
                            }
                        }
                        
                        // Process trilinear interactions
                        auto trilinear_matched = atoms->trilinear_interaction.equal_range(l);
                        for (auto m = trilinear_matched.first; m != trilinear_matched.second; ++m) {
                            const Trilinear& J = m->second;
                            size_t partner1 = flatten_index_periodic_boundary(
                                i + J.offset1[0], 
                                j + J.offset1[1], 
                                k + J.offset1[2], 
                                J.partner1, N_ATOMS);
                            size_t partner2 = flatten_index_periodic_boundary(
                                i + J.offset2[0], 
                                j + J.offset2[1], 
                                k + J.offset2[2], 
                                J.partner2, N_ATOMS);
                            
                            {
                                std::lock_guard<std::mutex> lock(site_mutexes[current_site_index]);
                                trilinear_interaction_sub[current_site_index].push_back(J.trilinear_interaction);
                                trilinear_partners_sub[current_site_index].push_back({partner1, partner2});
                            }
                            
                            vector<double> J_transposed1 = transpose3D_vec(J.trilinear_interaction, N, N, N);
                            {
                                std::lock_guard<std::mutex> lock(site_mutexes[partner1]);
                                trilinear_interaction_sub[partner1].push_back(J_transposed1);
                                trilinear_partners_sub[partner1].push_back({partner2, current_site_index});
                            }
                            
                            vector<double> J_transposed2 = transpose3D_vec(J_transposed1, N, N, N);
                            {
                                std::lock_guard<std::mutex> lock(site_mutexes[partner2]);
                                trilinear_interaction_sub[partner2].push_back(J_transposed2);
                                trilinear_partners_sub[partner2].push_back({current_site_index, partner1});
                            }
                        }
                    }
                }
            }
        }

        num_bi = bilinear_partners_sub.size() > 0 ? bilinear_partners_sub[0].size() : 0;
        num_tri = trilinear_partners_sub.size() > 0 ? trilinear_partners_sub[0].size() : 0;
    }

    static vector<double> transpose3D_vec(const vector<double>& tensor, size_t d1, size_t d2, size_t d3) {
        vector<double> result(d1 * d2 * d3);
        for (size_t i = 0; i < d1; ++i) {
            for (size_t j = 0; j < d2; ++j) {
                for (size_t k = 0; k < d3; ++k) {
                    result[j * d1 * d3 + k * d1 + i] = tensor[i * d2 * d3 + j * d3 + k];
                }
            }
        }
        return result;
    }
    
    MixedLattice(const MixedLatticeConfig& cfg, MixedUnitCell* atoms, 
                 double spin_length_SU2_in, double spin_length_SU3_in)
        : config(cfg), UC(*atoms), 
          spins(cfg.lattice_size_SU2(), cfg.N_SU2, cfg.lattice_size_SU3(), cfg.N_SU3),
          site_pos(cfg.lattice_size_SU2(), cfg.lattice_size_SU3()) {
        
        PerformanceTimer total_timer("Mixed Lattice Construction", true);
        
        // Initialize random seed
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, std::numeric_limits<int>::max());
        seed_lehman(dis(gen));
        lehman_next();

        // Initialize member variables
        spin_length_SU2 = spin_length_SU2_in;
        spin_length_SU3 = spin_length_SU3_in;
        field_drive_freq_SU2 = 0.0;
        field_drive_amp_SU2 = 0.0;
        field_drive_width_SU2 = 1.0;
        field_drive_freq_SU3 = 0.0;
        field_drive_amp_SU3 = 0.0;
        field_drive_width_SU3 = 1.0;
        t_B_1_SU2 = 0.0;
        t_B_2_SU2 = 0.0;
        t_B_1_SU3 = 0.0;
        t_B_2_SU3 = 0.0;
        field_drive_freq_SU2_SU3 = 0.0;
        field_drive_amp_SU2_SU3 = 0.0;
        field_drive_width_SU2_SU3 = 1.0;
        t_B_1_SU2_SU3 = 0.0;
        t_B_2_SU2_SU3 = 0.0;

        sublattice_frames_SU2 = atoms->SU2.sublattice_frames;
        sublattice_frames_SU3 = atoms->SU3.sublattice_frames;

        // Set up sublattices in parallel
        {
            PerformanceTimer sublattice_timer("Sublattice Setup (SU2 & SU3)", true);
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    set_up_sublattice(spin_length_SU2, spins.spins_SU2, site_pos.pos_SU2, 
                                    field_SU2, onsite_interaction_SU2, bilinear_interaction_SU2, 
                                    trilinear_interaction_SU2, bilinear_partners_SU2, 
                                    trilinear_partners_SU2, &(atoms->SU2), num_bi_SU2, num_tri_SU2,
                                    config.N_SU2, config.N_ATOMS_SU2);
                }
                
                #pragma omp section
                {
                    set_up_sublattice(spin_length_SU3, spins.spins_SU3, site_pos.pos_SU3, 
                                    field_SU3, onsite_interaction_SU3, bilinear_interaction_SU3, 
                                    trilinear_interaction_SU3, bilinear_partners_SU3, 
                                    trilinear_partners_SU3, &(atoms->SU3), num_bi_SU3, num_tri_SU3,
                                    config.N_SU3, config.N_ATOMS_SU3);
                }
            }
        }
        
        // Pre-allocate tensor storage
        size_t SU2_struct_const_size = config.N_SU2 * config.N_SU2 * config.N_SU2;
        size_t SU3_struct_const_size = config.N_SU3 * config.N_SU3 * config.N_SU3;
        SU2_structure_tensor.resize(config.N_ATOMS_SU2 * config.dim1 * config.dim2 * config.dim * SU2_struct_const_size);
        SU3_structure_tensor.resize(config.N_ATOMS_SU3 * config.dim1 * config.dim2 * config.dim * SU3_struct_const_size);
        
        lattice_size_SU2 = config.lattice_size_SU2();
        lattice_size_SU3 = config.lattice_size_SU3();

        cout << "Mixed lattice initialized with SU2 size: " << lattice_size_SU2 
             << ", SU3 size: " << lattice_size_SU3 << endl;

        // Initialize mixed interactions (simplified for now - would need full implementation)
        num_bi_SU2_SU3 = 0;
        num_tri_SU2_SU3 = 0;

        cout << "Finished setting up lattice" << endl;
    }

    double total_energy(MixedLatticeSpin& curr_spins) {
        // Simplified energy calculation
        double total = 0.0;
        
        // Add SU2 contributions
        for(size_t i = 0; i < lattice_size_SU2; ++i) {
            total -= dot_vec(curr_spins.spins_SU2[i], field_SU2[i]);
            total += contract_vec(curr_spins.spins_SU2[i], onsite_interaction_SU2[i], 
                                 curr_spins.spins_SU2[i], config.N_SU2);
        }
        
        // Add SU3 contributions
        for(size_t i = 0; i < lattice_size_SU3; ++i) {
            total -= dot_vec(curr_spins.spins_SU3[i], field_SU3[i]);
            total += contract_vec(curr_spins.spins_SU3[i], onsite_interaction_SU3[i], 
                                 curr_spins.spins_SU3[i], config.N_SU3);
        }
        
        return total;
    }

    static double dot_vec(const vector<double>& a, const vector<double>& b) {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    static double contract_vec(const vector<double>& spin, const vector<double>& interaction, 
                               const vector<double>& partner_spin, size_t N) {
        double result = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                result += spin[i] * interaction[i * N + j] * partner_spin[j];
            }
        }
        return result;
    }

    // Additional methods would be implemented similarly...
};

#endif // MIXED_LATTICE_RUNTIME_H
