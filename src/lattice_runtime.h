#ifndef LATTICE_RUNTIME_H
#define LATTICE_RUNTIME_H

#define _USE_MATH_DEFINES
#include "unitcell_runtime.h"
#include "lattice_config.h"
#include "simple_linear_alg.h"
#include "file_io.h"
#include <cmath>
#include <numbers>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <random>
#include <chrono>
#include <math.h>
#include <functional>
#include <mpi.h>
#include "binning_analysis.h"
#include <sstream>
#include <cstdint>
#include <deque>

// Optional profiling instrumentation (compile with -DENABLE_PROFILING to enable)
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


// Helper struct to store autocorrelation analysis results
struct AutocorrelationResult {
    double tau_int;
    size_t sampling_interval;
    vector<double> correlation_function;
};

// Pilot evaluation of a given ladder T: returns edge acceptance and round-trip rate
struct LadderEval {
    double mean_accept = 0.0;
    double min_accept = 0.0;
    double roundtrip_rate = 0.0; // per sweep per replica
    vector<double> edge_accept;  // size R-1
    size_t sweeps = 0;
};

// Auto-tuning for simulated annealing schedule
struct SAParams {
    double T_start = 1.0;
    double T_end = 1e-3;
    double cooling_rate = 0.9;
    size_t sweeps_per_temp = 100;
    vector<double> probe_T;
    vector<double> probe_acc;
    vector<double> probe_tau;
};

// Runtime lattice class (no templates)
class Lattice {
public:
    typedef vector<vector<double>> spin_config;  // lattice_size x N
    typedef function<vector<double>(const vector<double>&, const vector<double>&)> cross_product_method;
    typedef function<spin_config(double&, const spin_config&, const double, cross_product_method)> ODE_method;

    LatticeConfig config;
    UnitCell UC;
    size_t lattice_size;
    spin_config spins;
    vector<array<double,3>> site_pos;
    vector<vector<double>> twist_matrices;  // 3 x (N*N)
    vector<vector<double>> rotation_axis;   // 3 x N

    // Lookup table for the lattice
    spin_config field;
    vector<vector<double>> field_drive_1;   // N_ATOMS x N
    vector<vector<double>> field_drive_2;   // N_ATOMS x N
    vector<vector<vector<double>>> sublattice_frames;  // N_ATOMS x N x N
    double field_drive_freq;
    double field_drive_amp;
    double field_drive_width;
    double t_B_1;
    double t_B_2;

    vector<vector<double>> onsite_interaction;  // lattice_size x (N*N)

    vector<vector<vector<double>>> bilinear_interaction;  // lattice_size x num_bi x (N*N)
    vector<vector<vector<double>>> trilinear_interaction;  // lattice_size x num_tri x (N*N*N)

    vector<vector<size_t>> bilinear_partners;  // lattice_size x num_bi
    vector<vector<array<size_t, 2>>> trilinear_partners;  // lattice_size x num_tri x 2

    size_t num_bi;
    size_t num_tri;
    size_t num_gen;
    float spin_length;

private:
    // Pre-allocated temporary arrays to avoid allocations in hot loops
    mutable vector<double> temp_spin_array;
    mutable vector<double> temp_local_field;

protected:
    // Twist-boundary Monte Carlo support
    vector<vector<array<int8_t,3>>> bilinear_wrap_dir;  // lattice_size x num_bi x 3
    
    // Boundary sites per lattice dimension
    array<vector<size_t>, 3> boundary_sites_per_dim;
    array<size_t, 3> boundary_thickness;

public:
    // Configure twist axes per dimension
    void set_twist_axes(const vector<vector<double>>& axes) {
        rotation_axis = axes;
    }
    
    const vector<vector<double>>& get_twist_axes() const { return rotation_axis; }
    const vector<vector<double>>& get_twist_matrices() const { return twist_matrices; }

    vector<double> gen_random_spin(float spin_l) {
        vector<double> temp_spin(config.N);
        vector<double> euler_angles(config.N - 2);
        double z = random_double_lehman(-1, 1);
        double r = sqrt(1.0 - z*z);

        for(size_t i = 0; i < config.N - 2; ++i) {
            euler_angles[i] = random_double_lehman(0, 2*M_PI);
            temp_spin[i] = r;
            for(size_t j = 0; j < i; ++j) {
                temp_spin[i] *= sin(euler_angles[j]);
            }
            if (i == config.N - 3) {
                temp_spin[i+1] = temp_spin[i] * sin(euler_angles[i]);
            }
            temp_spin[i] *= cos(euler_angles[i]);
        }
        temp_spin[config.N - 1] = z;
        
        for(size_t i = 0; i < config.N; ++i) {
            temp_spin[i] *= spin_l;
        }
        return temp_spin;
    }

    size_t flatten_index(size_t i, size_t j, size_t k, size_t l) {
        return i * config.dim2 * config.dim * config.N_ATOMS + 
               j * config.dim * config.N_ATOMS + 
               k * config.N_ATOMS + l;
    }
    
    size_t periodic_boundary(int i, size_t D) {
        if(i < 0) {
            return size_t((D + i) % D);
        }
        else {
            return size_t(i % D);
        }
    }

    size_t flatten_index_periodic_boundary(int i, int j, int k, int l) {
        return periodic_boundary(i, config.dim1) * config.dim2 * config.dim * config.N_ATOMS + 
               periodic_boundary(j, config.dim2) * config.dim * config.N_ATOMS + 
               periodic_boundary(k, config.dim) * config.N_ATOMS + l;
    }

    vector<vector<double>> default_twist_matrix() {
        vector<vector<double>> twist_matrix(3, vector<double>(config.N * config.N, 0.0));
        for (size_t d = 0; d < 3; ++d) {
            for (size_t i = 0; i < config.N; ++i) {
                twist_matrix[d][i * config.N + i] = 1.0;
            }
        }
        return twist_matrix;
    }

    Lattice(const LatticeConfig& cfg, UnitCell* atoms, float spin_l = 1, bool periodic = true)
        : config(cfg), UC(*atoms)
    {
        lattice_size = config.lattice_size();
        spin_length = spin_l;

        // Initialize all data structures with proper sizes
        spins.resize(lattice_size, vector<double>(config.N, 0.0));
        site_pos.resize(lattice_size);
        field.resize(lattice_size, vector<double>(config.N, 0.0));
        onsite_interaction.resize(lattice_size, vector<double>(config.N * config.N, 0.0));
        
        field_drive_1.resize(config.N_ATOMS, vector<double>(config.N, 0.0));
        field_drive_2.resize(config.N_ATOMS, vector<double>(config.N, 0.0));
        field_drive_freq = 0.0;
        field_drive_amp = 0.0;
        field_drive_width = 1.0;
        t_B_1 = 0.0;
        t_B_2 = 0.0;

        srand(time(NULL));
        seed_lehman(rand() * 2 + 1);

        boundary_thickness = {0, 0, 0};
        sublattice_frames = UC.sublattice_frames;

        // Precompute maximum neighbor offset
        for (size_t l = 0; l < config.N_ATOMS; ++l) {
            auto bilinear_matched = UC.bilinear_interaction.equal_range(l);
            for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m) {
                const Bilinear& J = m->second;
                boundary_thickness[0] = std::max(boundary_thickness[0], size_t(std::abs(J.offset[0])));
                boundary_thickness[1] = std::max(boundary_thickness[1], size_t(std::abs(J.offset[1])));
                boundary_thickness[2] = std::max(boundary_thickness[2], size_t(std::abs(J.offset[2])));
            }
            auto trilinear_matched = UC.trilinear_interaction.equal_range(l);
            for (auto m = trilinear_matched.first; m != trilinear_matched.second; ++m) {
                const Trilinear& J = m->second;
                boundary_thickness[0] = std::max(boundary_thickness[0], 
                    size_t(std::max(std::abs(J.offset1[0]), std::abs(J.offset2[0]))));
                boundary_thickness[1] = std::max(boundary_thickness[1], 
                    size_t(std::max(std::abs(J.offset1[1]), std::abs(J.offset2[1]))));
                boundary_thickness[2] = std::max(boundary_thickness[2], 
                    size_t(std::max(std::abs(J.offset1[2]), std::abs(J.offset2[2]))));
            }
        }

        // Initialize lattice sites
        cout << "Initializing lattice sites..." << endl;
        size_t processed_sites = 0;
        size_t total_sites = lattice_size;

        for (size_t i = 0; i < config.dim1; ++i) {
            for (size_t j = 0; j < config.dim2; ++j) {
                for(size_t k = 0; k < config.dim; ++k) {
                    for (size_t l = 0; l < config.N_ATOMS; ++l) {
                        size_t current_site_index = flatten_index(i, j, k, l);

                        // Calculate position
                        for(size_t d = 0; d < 3; ++d) {
                            site_pos[current_site_index][d] = 
                                UC.lattice_vectors[0][d] * int(i) +
                                UC.lattice_vectors[1][d] * int(j) +
                                UC.lattice_vectors[2][d] * int(k) +
                                UC.lattice_pos[l][d];
                        }

                        spins[current_site_index] = gen_random_spin(spin_length);
                        field[current_site_index] = UC.field[l];
                        onsite_interaction[current_site_index] = UC.onsite_interaction[l];

                        // Process bilinear interactions
                        auto bilinear_matched = UC.bilinear_interaction.equal_range(l);
                        for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m) {
                            const Bilinear& J = m->second;
                            int partner_i = int(i) + J.offset[0];
                            int partner_j = int(j) + J.offset[1];
                            int partner_k = int(k) + J.offset[2];
                            size_t partner = flatten_index_periodic_boundary(partner_i, partner_j, partner_k, J.partner);
                            
                            if (periodic || (partner_i < int(config.dim1) && partner_i >= 0 && 
                                           partner_j < int(config.dim2) && partner_j >= 0 && 
                                           partner_k < int(config.dim) && partner_k >= 0)) {
                                array<int8_t,3> wrap_dir = {0, 0, 0};
                                if (partner_i < 0) wrap_dir[0] = -1; 
                                else if (partner_i >= int(config.dim1)) wrap_dir[0] = +1;
                                if (partner_j < 0) wrap_dir[1] = -1; 
                                else if (partner_j >= int(config.dim2)) wrap_dir[1] = +1;
                                if (partner_k < 0) wrap_dir[2] = -1; 
                                else if (partner_k >= int(config.dim)) wrap_dir[2] = +1;
                                
                                bilinear_interaction[current_site_index].push_back(J.bilinear_interaction);
                                bilinear_partners[current_site_index].push_back(partner);
                                bilinear_wrap_dir[current_site_index].push_back(wrap_dir);
                                
                                // Add symmetric interaction
                                vector<double> transposed = transpose2D_vec(J.bilinear_interaction, config.N, config.N);
                                bilinear_interaction[partner].push_back(transposed);
                                bilinear_partners[partner].push_back(current_site_index);
                                array<int8_t,3> wrap_dir_partner = {int8_t(-wrap_dir[0]), int8_t(-wrap_dir[1]), int8_t(-wrap_dir[2])};
                                bilinear_wrap_dir[partner].push_back(wrap_dir_partner);
                            }
                        }

                        // Process trilinear interactions
                        auto trilinear_matched = UC.trilinear_interaction.equal_range(l);
                        for (auto m = trilinear_matched.first; m != trilinear_matched.second; ++m) {
                            const Trilinear& J = m->second;
                            int partner1_i = int(i) + J.offset1[0];
                            int partner1_j = int(j) + J.offset1[1];
                            int partner1_k = int(k) + J.offset1[2];
                            int partner2_i = int(i) + J.offset2[0];
                            int partner2_j = int(j) + J.offset2[1];
                            int partner2_k = int(k) + J.offset2[2];

                            size_t partner1 = flatten_index_periodic_boundary(partner1_i, partner1_j, partner1_k, J.partner1);
                            size_t partner2 = flatten_index_periodic_boundary(partner2_i, partner2_j, partner2_k, J.partner2);
                            
                            if (periodic || ((partner1_i < int(config.dim1) && partner1_i >= 0 && 
                                            partner1_j < int(config.dim2) && partner1_j >= 0 && 
                                            partner1_k < int(config.dim) && partner1_k >= 0) &&
                                           (partner2_i < int(config.dim1) && partner2_i >= 0 && 
                                            partner2_j < int(config.dim2) && partner2_j >= 0 && 
                                            partner2_k < int(config.dim) && partner2_k >= 0))) {
                                
                                trilinear_interaction[current_site_index].push_back(J.trilinear_interaction);
                                trilinear_partners[current_site_index].push_back({partner1, partner2});

                                vector<double> transposed1 = transpose3D_vec(J.trilinear_interaction, config.N, config.N, config.N);
                                trilinear_interaction[partner1].push_back(transposed1);
                                trilinear_partners[partner1].push_back({partner2, current_site_index});

                                vector<double> transposed2 = transpose3D_vec(transposed1, config.N, config.N, config.N);
                                trilinear_interaction[partner2].push_back(transposed2);
                                trilinear_partners[partner2].push_back({current_site_index, partner1});
                            }
                        }

                        processed_sites++;
                    }
                }
            }
        }

        twist_matrices = default_twist_matrix();
        rotation_axis.resize(3, vector<double>(config.N, 0.0));
        for (size_t d = 0; d < 3; ++d) {
            if (config.N >= 3) rotation_axis[d][2] = 1.0;
        }

        num_bi = bilinear_partners.size() > 0 ? bilinear_partners[0].size() : 0;
        num_tri = trilinear_partners.size() > 0 ? trilinear_partners[0].size() : 0;
        num_gen = config.N;
        
        cout << "\nLattice initialization complete!" << endl;
        cout << "Total sites: " << lattice_size << endl;
        cout << "Bilinear interactions: " << num_bi << endl;
        cout << "Trilinear interactions: " << num_tri << endl;
        cout << "Spin dimension: " << num_gen << endl;
        
        build_boundary_sites();
    }

    void build_boundary_sites() {
        boundary_sites_per_dim[0].clear();
        boundary_sites_per_dim[1].clear();
        boundary_sites_per_dim[2].clear();
        
        for (size_t i = 0; i < config.dim1; ++i) {
            bool is_boundary_x = (config.dim1 > 1) && (i < boundary_thickness[0] || i >= config.dim1 - boundary_thickness[0]);
            for (size_t j = 0; j < config.dim2; ++j) {
                bool is_boundary_y = (config.dim2 > 1) && (j < boundary_thickness[1] || j >= config.dim2 - boundary_thickness[1]);
                for (size_t k = 0; k < config.dim; ++k) {
                    bool is_boundary_z = (config.dim > 1) && (k < boundary_thickness[2] || k >= config.dim - boundary_thickness[2]);
                    for (size_t l = 0; l < config.N_ATOMS; ++l) {
                        size_t idx = flatten_index(i, j, k, l);
                        if (is_boundary_x) boundary_sites_per_dim[0].push_back(idx);
                        if (is_boundary_y) boundary_sites_per_dim[1].push_back(idx);
                        if (is_boundary_z) boundary_sites_per_dim[2].push_back(idx);
                    }
                }
            }
        }
    }

    // Add methods for energy calculations, Monte Carlo, etc.
    // These would follow similar patterns to the template versions
    // but using runtime-sized vectors instead of compile-time arrays
    
    double site_energy(const vector<double>& spin_here, size_t site_index) {
        double single_site_energy = 0, double_site_energy = 0, triple_site_energy = 0;
        
        single_site_energy -= dot(spin_here, field[site_index]);
        single_site_energy += contract(spin_here, onsite_interaction[site_index], spin_here, config.N);
        
        for (size_t i = 0; i < num_bi; ++i) {
            vector<double> partner_spin_eff = apply_twist_to_partner_spin(
                spins[bilinear_partners[site_index][i]], bilinear_wrap_dir[site_index][i]);
            double_site_energy += contract(spin_here, bilinear_interaction[site_index][i], 
                                          partner_spin_eff, config.N);
        }
        
        for (size_t i = 0; i < num_tri; ++i) {
            size_t p1 = trilinear_partners[site_index][i][0];
            size_t p2 = trilinear_partners[site_index][i][1];
            triple_site_energy += contract_trilinear(trilinear_interaction[site_index][i], 
                                                     spin_here, spins[p1], spins[p2], config.N);
        }
        
        return single_site_energy + double_site_energy + triple_site_energy;
    }

    vector<double> apply_twist_to_partner_spin(const vector<double>& partner_spin,
                                               const array<int8_t,3>& wrap) const {
        vector<double> s = partner_spin;
        for (size_t d = 0; d < 3; ++d) {
            int8_t w = wrap[d];
            if (w == 0) continue;
            if (w > 0) {
                s = multiply_vec(twist_matrices[d], s, config.N, config.N);
            } else {
                vector<double> transposed = transpose2D_vec(twist_matrices[d], config.N, config.N);
                s = multiply_vec(transposed, s, config.N, config.N);
            }
        }
        return s;
    }

    double total_energy(spin_config& curr_spins) {
        double field_energy = 0.0;
        double onsite_energy = 0.0;
        double bilinear_energy = 0.0;
        double trilinear_energy = 0.0;
        
        for(size_t i = 0; i < lattice_size; ++i) {
            field_energy -= dot(curr_spins[i], field[i]);
            onsite_energy += contract(curr_spins[i], onsite_interaction[i], curr_spins[i], config.N);
            
            for (size_t j = 0; j < num_bi; ++j) {
                size_t partner_idx = bilinear_partners[i][j];
                vector<double> partner_spin_eff = apply_twist_to_partner_spin(
                    curr_spins[partner_idx], bilinear_wrap_dir[i][j]);
                bilinear_energy += contract(curr_spins[i], bilinear_interaction[i][j], 
                                           partner_spin_eff, config.N);
            }

            for (size_t j = 0; j < num_tri; ++j) {
                trilinear_energy += contract_trilinear(trilinear_interaction[i][j], 
                                                       curr_spins[i], 
                                                       curr_spins[trilinear_partners[i][j][0]], 
                                                       curr_spins[trilinear_partners[i][j][1]], 
                                                       config.N);
            }
        }
        
        return field_energy + onsite_energy + bilinear_energy/2 + trilinear_energy/3;
    }

    // Helper functions for linear algebra with runtime dimensions
    static vector<double> transpose2D_vec(const vector<double>& matrix, size_t rows, size_t cols) {
        vector<double> result(rows * cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[j * rows + i] = matrix[i * cols + j];
            }
        }
        return result;
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

    static vector<double> multiply_vec(const vector<double>& matrix, const vector<double>& vec, 
                                       size_t rows, size_t cols) {
        vector<double> result(rows, 0.0);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i] += matrix[i * cols + j] * vec[j];
            }
        }
        return result;
    }

    static double dot(const vector<double>& a, const vector<double>& b) {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    static double contract(const vector<double>& spin, const vector<double>& interaction, 
                          const vector<double>& partner_spin, size_t N) {
        double result = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                result += spin[i] * interaction[i * N + j] * partner_spin[j];
            }
        }
        return result;
    }

    static double contract_trilinear(const vector<double>& interaction, 
                                     const vector<double>& spin1,
                                     const vector<double>& spin2, 
                                     const vector<double>& spin3, 
                                     size_t N) {
        double result = 0.0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < N; ++k) {
                    result += interaction[i * N * N + j * N + k] * spin1[i] * spin2[j] * spin3[k];
                }
            }
        }
        return result;
    }

    // Placeholder for additional methods that would be implemented similarly
    // (metropolis, simulated_annealing, parallel_tempering, etc.)
};

#endif // LATTICE_RUNTIME_H
