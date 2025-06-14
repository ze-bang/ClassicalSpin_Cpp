#ifndef MIXED_LATTICE_H
#define MIXED_LATTICE_H

#define _USE_MATH_DEFINES
#include "unitcell.h"
#include "simple_linear_alg.h"
#include <cmath>
#include <numbers>
#include <iostream>
#include <fstream>
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
// #include <boost>

template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
struct mixed_lattice_spin{
    array<array<double,N_SU2>, lattice_size_SU2> spins_SU2;
    array<array<double,N_SU3>, lattice_size_SU3> spins_SU3;
    mixed_lattice_spin(){
        for (size_t i=0; i< lattice_size_SU2; ++i){
            for (size_t j=0; j< N_SU2; ++j){
                spins_SU2[i][j] = 0;
            }
        }
        for (size_t i=0; i< lattice_size_SU3; ++i){
            for (size_t j=0; j< N_SU3; ++j){
                spins_SU3[i][j] = 0;
            }
        }
    }
    mixed_lattice_spin(const array<array<double,N_SU2>, lattice_size_SU2> spins_SU2_in, const array<array<double,N_SU3>, lattice_size_SU3> spins_SU3_in) : spins_SU2(spins_SU2_in), spins_SU3(spins_SU3_in) {
    };
    mixed_lattice_spin(const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> &spins_in){
        spins_SU2 = spins_in.spins_SU2;
        spins_SU3 = spins_in.spins_SU3;
    };

    void set(mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3>& spins_in) noexcept {
        using std::swap;
        swap(spins_SU2, spins_in.spins_SU2);
        swap(spins_SU3, spins_in.spins_SU3);
    }


    double length(){
        double result = 0;
        for (size_t i = 0; i < lattice_size_SU2; ++i){
            result += dot(spins_SU2[i], spins_SU2[i]);
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i){
            result += dot(spins_SU3[i], spins_SU3[i]);
        }
        return sqrt(result);
    }
};

template<size_t lattice_size_SU2, size_t lattice_size_SU3>
struct mixed_lattice_pos{
    array<array<double,3>, lattice_size_SU2> pos_SU2;
    array<array<double,3>, lattice_size_SU3> pos_SU3;
    mixed_lattice_pos(){
        for (size_t i=0; i< lattice_size_SU2; ++i){
            for (size_t j=0; j< 3; ++j){
                pos_SU2[i][j] = 0;
            }
        }
        for (size_t i=0; i< lattice_size_SU3; ++i){
            for (size_t j=0; j< 3; ++j){
                pos_SU3[i][j] = 0;
            }
        }
    }
    mixed_lattice_pos(const array<array<double,3>, lattice_size_SU2> spins_SU2_in, const array<array<double,3>, lattice_size_SU3> spins_SU3_in) : pos_SU2(spins_SU2_in), pos_SU3(spins_SU3_in) {
    };
    mixed_lattice_pos(const mixed_lattice_pos<lattice_size_SU2, lattice_size_SU3> &pos_in){
        pos_SU2 = pos_in.pos_SU2;
        pos_SU3 = pos_in.pos_SU3;
    };
};

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim>
class mixed_lattice
{   
    public:

    typedef array<array<double,N_SU2>,N_ATOMS_SU2*dim1*dim2*dim> spin_config_SU2;
    typedef array<array<double,N_SU3>,N_ATOMS_SU3*dim1*dim2*dim> spin_config_SU3;

    mixed_UnitCell<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3> UC;
    size_t lattice_size_SU2;
    size_t lattice_size_SU3;
    double spin_length_SU2, spin_length_SU3;
    mixed_lattice_spin<N_SU2, dim1*dim2*dim*N_ATOMS_SU2, N_SU3, dim1*dim2*dim*N_ATOMS_SU3> spins;
    mixed_lattice_pos<dim1*dim2*dim*N_ATOMS_SU2, dim1*dim2*dim*N_ATOMS_SU3>  site_pos;

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
    //Look up table for SU2
    array<array<double,N_SU2>, N_ATOMS_SU2*dim1*dim2*dim> field_SU2;
    array<array<double,N_SU2>, N_ATOMS_SU2> field_drive_1_SU2;
    array<array<double,N_SU2>, N_ATOMS_SU2> field_drive_2_SU2;

    array<array<double, N_SU2 * N_SU2>, N_ATOMS_SU2*dim1*dim2*dim> onsite_interaction_SU2;
    array<vector<array<double, N_SU2 * N_SU2>>, N_ATOMS_SU2*dim1*dim2*dim> bilinear_interaction_SU2;    
    array<vector<array<double, N_SU2 * N_SU2 * N_SU2>>, N_ATOMS_SU2*dim1*dim2*dim> trilinear_interaction_SU2;

    array<vector<size_t>, N_ATOMS_SU2*dim1*dim2*dim> bilinear_partners_SU2;
    array<vector<array<size_t, 2>>, N_ATOMS_SU2*dim1*dim2*dim> trilinear_partners_SU2;

    //Look up table for SU3
    array<array<double,N_SU3>, N_ATOMS_SU3*dim1*dim2*dim> field_SU3;
    array<array<double,N_SU3>, N_ATOMS_SU3> field_drive_1_SU3;
    array<array<double,N_SU3>, N_ATOMS_SU3> field_drive_2_SU3;

    array<array<double, N_SU3 * N_SU3>, N_ATOMS_SU2*dim1*dim2*dim> onsite_interaction_SU3;
    array<vector<array<double, N_SU3 * N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim> bilinear_interaction_SU3;
    array<vector<array<double, N_SU3 * N_SU3 * N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim> trilinear_interaction_SU3;

    array<vector<size_t>, N_ATOMS_SU3*dim1*dim2*dim> bilinear_partners_SU3;
    array<vector<array<size_t, 2>>, N_ATOMS_SU3*dim1*dim2*dim> trilinear_partners_SU3;

    //Look up table for SU2 and SU3 mix
    array<vector<array<double, N_SU2 * N_SU3>>, N_ATOMS_SU2*dim1*dim2*dim> mixed_bilinear_interaction_SU2;
    array<vector<size_t>, N_ATOMS_SU2*dim1*dim2*dim> mixed_bilinear_partners_SU2;
    array<vector<size_t>, N_ATOMS_SU3*dim1*dim2*dim> mixed_bilinear_partners_SU3;

    array<vector<array<double, N_SU2 * N_SU2 * N_SU3>>, N_ATOMS_SU2*dim1*dim2*dim> mixed_trilinear_interaction_SU2;
    array<vector<array<double, N_SU2 * N_SU2 * N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim> mixed_trilinear_interaction_SU3;

    array<vector<array<size_t, 2>>, N_ATOMS_SU2*dim1*dim2*dim> mixed_trilinear_partners_SU2;
    array<vector<array<size_t, 2>>, N_ATOMS_SU3*dim1*dim2*dim> mixed_trilinear_partners_SU3;

    vector<double> SU2_structure_tensor;
    vector<double> SU3_structure_tensor;

    size_t num_bi_SU2;
    size_t num_tri_SU2;
    size_t num_bi_SU3;
    size_t num_tri_SU3;
    size_t num_tri_SU2_SU3;

    DeviceStructureTensorManager device_structure_tensor_manager;

    template<size_t N>
    void gen_random_spin(array<double, N> &temp_spin, const double spin_l){
        // array<double,N> temp_spin;
        array<double,N-2> euler_angles;
        // double z = random_double(-1,1, gen);
        double z = random_double_lehman(-1,1);
        double r = sqrt(1.0 - z*z);

        for(size_t i = 0; i < N-2; ++i){
            euler_angles[i] = random_double_lehman(0, 2*M_PI);
            temp_spin[i] = r;
            for(size_t j = 0; j < i; ++j){
                temp_spin[i] *= sin(euler_angles[j]);
            }
            if (i == N-3){
                temp_spin[i+1] = temp_spin[i]*sin(euler_angles[i]);
            }
            temp_spin[i] *= cos(euler_angles[i]);

        }
        temp_spin[N-1] = z;
        temp_spin *= spin_l;
    }

    
    array<double,N_SU2> gen_random_spin_SU2(){
        array<double,N_SU2> temp_spin;
        array<double,N_SU2-2> euler_angles;
        // double z = random_double(-1,1, gen);
        double z = random_double_lehman(-1,1);
        double r = sqrt(1.0 - z*z);

        for(size_t i = 0; i < N_SU2-2; ++i){
            // euler_angles[i] = random_double(0, 2*M_PI, gen);
            euler_angles[i] = random_double_lehman(0, 2*M_PI);
            temp_spin[i] = r;
            for(size_t j = 0; j < i; ++j){
                temp_spin[i] *= sin(euler_angles[j]);
            }
            if (i == N_SU2-3){
                temp_spin[i+1] = temp_spin[i]*sin(euler_angles[i]);
            }
            temp_spin[i] *= cos(euler_angles[i]);

        }
        temp_spin[N_SU2-1] = z;
        return temp_spin*spin_length_SU2;
    }

    array<double,N_SU3> gen_random_spin_SU3(){
        array<double,N_SU3> temp_spin;
        array<double,N_SU3-2> euler_angles;
        // double z = random_double(-1,1, gen);
        double z = random_double_lehman(-1,1);
        double r = sqrt(1.0 - z*z);

        for(size_t i = 0; i < N_SU3-2; ++i){
            // euler_angles[i] = random_double(0, 2*M_PI, gen);
            euler_angles[i] = random_double_lehman(0, 2*M_PI);
            temp_spin[i] = r;
            for(size_t j = 0; j < i; ++j){
                temp_spin[i] *= sin(euler_angles[j]);
            }
            if (i == N_SU3-3){
                temp_spin[i+1] = temp_spin[i]*sin(euler_angles[i]);
            }
            temp_spin[i] *= cos(euler_angles[i]);

        }
        temp_spin[N_SU3-1] = z;
        return temp_spin*spin_length_SU3;
    }

    size_t flatten_index(size_t i, size_t j, size_t k, size_t l, size_t N_ATOMS){
        return i*dim2*dim*N_ATOMS+ j*dim*N_ATOMS+ k*N_ATOMS + l;
    }
    
    size_t periodic_boundary(int i, size_t D){
        if(i < 0){
            return size_t((D+i) % D);
        }
        else{
            return size_t(i % D);
        }
    }

    size_t flatten_index_periodic_boundary(int i, int j, int k, int l, size_t N_ATOMS){
        return periodic_boundary(i, dim1)*dim2*dim*N_ATOMS+ periodic_boundary(j, dim2)*dim*N_ATOMS+ periodic_boundary(k, dim)*N_ATOMS + l;
    }

    template<size_t N, size_t lattice_size, size_t N_ATOMS>
    void set_up_sublattice(const double spin_length, array<array<double,N>, lattice_size> &spins, 
                            array<array<double,3>, lattice_size> &site_pos,
                            array<array<double,N>, lattice_size> &field, 
                            array<array<double, N * N>, lattice_size> &onsite_interaction,
                            array<vector<array<double, N * N>>, lattice_size> &bilinear_interaction,
                            array<vector<array<double, N*N*N>>, lattice_size> &trilinear_interaction,
                            array<vector<size_t>, lattice_size> &bilinear_partners,
                            array<vector<array<size_t, 2>>, lattice_size> &trilinear_partners,
                            UnitCell<N, N_ATOMS> *atoms, size_t &num_bi, size_t &num_tri) {
        
        // Pre-compute and cache frequently used values
        const array<array<double,3>, N_ATOMS> &basis = atoms->lattice_pos;
        const array<array<double,3>, 3> &unit_vector = atoms->lattice_vectors;

        // Main lattice setup with parallel processing across lattice sites
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim; ++k) {
                    // Thread-local temporary vectors to reduce mutex contention
                    vector<vector<array<double, N * N>>> thread_bilinear_interaction(N_ATOMS);
                    vector<vector<size_t>> thread_bilinear_partners(N_ATOMS);
                    vector<vector<array<double, N*N*N>>> thread_trilinear_interaction(N_ATOMS);
                    vector<vector<array<size_t, 2>>> thread_trilinear_partners(N_ATOMS);
                    
                    for (size_t l = 0; l < N_ATOMS; ++l) {
                        size_t current_site_index = flatten_index(i, j, k, l, N_ATOMS);
                        
                        // Calculate position (vectorizable)
                        #pragma omp simd
                        for (int d = 0; d < 3; d++) {
                            site_pos[current_site_index][d] = 
                                unit_vector[0][d] * int(i) + 
                                unit_vector[1][d] * int(j) + 
                                unit_vector[2][d] * int(k) + 
                                basis[l][d];
                        }
                        
                        // Generate random spin
                        gen_random_spin(spins[current_site_index], spin_length);
                        
                        // Copy field and onsite interaction (use std::copy for better optimization)
                        field[current_site_index] = atoms->field[l];
                        onsite_interaction[current_site_index] = atoms->onsite_interaction[l];
                        // Handle bilinear interactions
                        auto bilinear_matched = atoms->bilinear_interaction.equal_range(l);
                        for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m) {
                            const bilinear<N> &J = m->second;
                            size_t partner = flatten_index_periodic_boundary(
                                int(i) + J.offset[0], 
                                int(j) + J.offset[1], 
                                int(k) + J.offset[2], 
                                J.partner, N_ATOMS);
                            
                            // Store in thread-local vectors
                            thread_bilinear_interaction[l].push_back(J.bilinear_interaction);
                            thread_bilinear_partners[l].push_back(partner);
                        }
                        
                        // Handle trilinear interactions
                        auto trilinear_matched = atoms->trilinear_interaction.equal_range(l);
                        for (auto m = trilinear_matched.first; m != trilinear_matched.second; ++m) {
                            const trilinear<N> &J = m->second;
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
                            
                            // Store in thread-local vectors
                            thread_trilinear_interaction[l].push_back(J.trilinear_interaction);
                            thread_trilinear_partners[l].push_back({partner1, partner2});
                        }
                    }
                    
                    // Now merge thread-local data into global arrays using critical sections
                    #pragma omp critical
                    {
                        for (size_t l = 0; l < N_ATOMS; ++l) {
                            size_t current_site_index = flatten_index(i, j, k, l, N_ATOMS);
                            
                            // Add bilinear interactions from this thread
                            for (size_t idx = 0; idx < thread_bilinear_interaction[l].size(); ++idx) {
                                const auto &J = thread_bilinear_interaction[l][idx];
                                size_t partner = thread_bilinear_partners[l][idx];
                                
                                bilinear_interaction[current_site_index].push_back(J);
                                bilinear_partners[current_site_index].push_back(partner);
                                
                                // Add symmetric interaction
                                bilinear_interaction[partner].push_back(J);
                                bilinear_partners[partner].push_back(current_site_index);
                            }
                            
                            // Add trilinear interactions from this thread
                            for (size_t idx = 0; idx < thread_trilinear_interaction[l].size(); ++idx) {
                                const auto &J = thread_trilinear_interaction[l][idx];
                                const auto &partners = thread_trilinear_partners[l][idx];
                                size_t partner1 = partners[0];
                                size_t partner2 = partners[1];
                                
                                trilinear_interaction[current_site_index].push_back(J);
                                trilinear_partners[current_site_index].push_back({partner1, partner2});
                                
                                // Add symmetric interactions with proper tensor transposition
                                auto J_transposed1 = transpose3D(J, N, N, N);
                                trilinear_interaction[partner1].push_back(J_transposed1);
                                trilinear_partners[partner1].push_back({partner2, current_site_index});
                                
                                auto J_transposed2 = transpose3D(transpose3D(J, N, N, N), N, N, N);
                                trilinear_interaction[partner2].push_back(J_transposed2);
                                trilinear_partners[partner2].push_back({current_site_index, partner1});
                            }
                        }
                    }
                }
            }
        }

        // Only need to check one site to get interaction counts since all sites have the same structure
        num_bi = bilinear_partners[0].size();
        num_tri = trilinear_partners[0].size();
    }
    
    mixed_lattice(mixed_UnitCell<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3> *atoms, double spin_length_SU2_in, double spin_length_SU3_in): UC(*atoms) {
        // Initialize random seed with better entropy source
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, std::numeric_limits<int>::max());
        seed_lehman(dis(gen));
        lehman_next();

        // Initialize member variables with direct initialization
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

        // Set up sublattices in parallel
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                set_up_sublattice(spin_length_SU2, spins.spins_SU2, site_pos.pos_SU2, 
                                field_SU2, onsite_interaction_SU2, bilinear_interaction_SU2, 
                                trilinear_interaction_SU2, bilinear_partners_SU2, 
                                trilinear_partners_SU2, &(atoms->SU2), num_bi_SU2, num_tri_SU2);
            }
            
            #pragma omp section
            {
                set_up_sublattice(spin_length_SU3, spins.spins_SU3, site_pos.pos_SU3, 
                                field_SU3, onsite_interaction_SU3, bilinear_interaction_SU3, 
                                trilinear_interaction_SU3, bilinear_partners_SU3, 
                                trilinear_partners_SU3, &(atoms->SU3), num_bi_SU3, num_tri_SU3);
            }
        }
        
        // Pre-allocate tensor storage with exact sizes
        constexpr size_t SU2_struct_const_size = N_SU2 * N_SU2 * N_SU2;
        constexpr size_t SU3_struct_const_size = N_SU3 * N_SU3 * N_SU3;
        SU2_structure_tensor.resize(N_ATOMS_SU2 * dim1 * dim2 * dim * SU2_struct_const_size);
        SU3_structure_tensor.resize(N_ATOMS_SU3 * dim1 * dim2 * dim * SU3_struct_const_size);
        
        // Pre-compute lattice sizes
        lattice_size_SU2 = dim1 * dim2 * dim * N_ATOMS_SU2;
        lattice_size_SU3 = dim1 * dim2 * dim * N_ATOMS_SU3;

        // Process trilinear mixed interactions in parallel across outer loop dimensions
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim; ++k) {
                    // Thread-local vectors to avoid synchronization overhead
                    vector<vector<array<double, N_SU2 * N_SU2 * N_SU3>>> local_mixed_trilinear_interaction_SU2(N_ATOMS_SU2);
                    vector<vector<array<size_t, 2>>> local_mixed_trilinear_partners_SU2(N_ATOMS_SU2);
                    
                    for (size_t l = 0; l < N_ATOMS_SU3; ++l) {
                        const size_t current_site_index = flatten_index(i, j, k, l, N_ATOMS_SU3);
                        
                        // Process mixed trilinear interactions
                        auto trilinear_matched = atoms->trilinear_SU2_SU3.equal_range(l);
                        for (auto m = trilinear_matched.first; m != trilinear_matched.second; ++m) {
                            const mixed_trilinear<N_SU2, N_SU3>& J = m->second;
                            const size_t partner1 = flatten_index_periodic_boundary(
                                i + J.offset1[0], j + J.offset1[1], k + J.offset1[2], J.partner1, N_ATOMS_SU2);
                            const size_t partner2 = flatten_index_periodic_boundary(
                                i + J.offset2[0], j + J.offset2[1], k + J.offset2[2], J.partner2, N_ATOMS_SU2);
                            
                            // Add directly to site-specific array for SU3
                            #pragma omp critical(SU3_update)
                            {
                                mixed_trilinear_interaction_SU3[current_site_index].push_back(J.trilinear_interaction);
                                mixed_trilinear_partners_SU3[current_site_index].push_back({partner1, partner2});
                            }
                            
                            // Precompute transposed tensors once
                            const auto transposed = transpose3D(J.trilinear_interaction, N_SU3, N_SU2, N_SU2);
                            const auto swapped = swap_axis_3D(transposed, N_SU2, N_SU2, N_SU3);
                            
                            // Store in thread-local vectors to avoid synchronization
                            if (local_mixed_trilinear_interaction_SU2.size() <= partner1) {
                                local_mixed_trilinear_interaction_SU2.resize(partner1 + 1);
                                local_mixed_trilinear_partners_SU2.resize(partner1 + 1);
                            }
                            if (local_mixed_trilinear_interaction_SU2.size() <= partner2) {
                                local_mixed_trilinear_interaction_SU2.resize(partner2 + 1);
                                local_mixed_trilinear_partners_SU2.resize(partner2 + 1);
                            }
                            
                            local_mixed_trilinear_interaction_SU2[partner1].push_back(transposed);
                            local_mixed_trilinear_partners_SU2[partner1].push_back({partner2, current_site_index});
                            
                            local_mixed_trilinear_interaction_SU2[partner2].push_back(swapped);
                            local_mixed_trilinear_partners_SU2[partner2].push_back({partner1, current_site_index});
                        }
                    }
                    
                    // Merge thread-local data into global arrays
                    #pragma omp critical(SU2_update)
                    {
                        for (size_t idx = 0; idx < local_mixed_trilinear_interaction_SU2.size(); ++idx) {
                            for (size_t v = 0; v < local_mixed_trilinear_interaction_SU2[idx].size(); ++v) {
                                mixed_trilinear_interaction_SU2[idx].push_back(
                                    local_mixed_trilinear_interaction_SU2[idx][v]);
                                mixed_trilinear_partners_SU2[idx].push_back(
                                    local_mixed_trilinear_partners_SU2[idx][v]);
                            }
                        }
                    }
                }
            }
        }
        
        // Initialize structure tensors in parallel
        // For SU2 structure tensor - these are constant across sites, so calculate once
        array<array<array<double, N_SU2>, N_SU2>, N_SU2> SU2_structure_cache;
        bool SU2_structure_initialized = false;
        
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t site_idx = 0; site_idx < lattice_size_SU2; ++site_idx) {
            for (size_t a = 0; a < N_SU2; ++a) {
                // Lazily initialize the structure tensor cache once per thread
                if (!SU2_structure_initialized) {
                    #pragma omp critical(SU2_struct_init)
                    {
                        if (!SU2_structure_initialized) {
                            for (size_t i = 0; i < N_SU2; ++i)
                                for (size_t j = 0; j < N_SU2; ++j)
                                    for (size_t k = 0; k < N_SU2; ++k)
                                        SU2_structure_cache[i][j][k] = SU2_structure[i][j][k];
                            SU2_structure_initialized = true;
                        }
                    }
                }
                
                const size_t base_idx = site_idx * SU2_struct_const_size + a * N_SU2 * N_SU2;
                #pragma omp simd collapse(2)
                for (size_t b = 0; b < N_SU2; ++b) {
                    for (size_t c = 0; c < N_SU2; ++c) {
                        SU2_structure_tensor[base_idx + b * N_SU2 + c] = SU2_structure_cache[a][b][c];
                    }
                }
            }
        }
        
        // For SU3 structure tensor
        array<array<array<double, N_SU3>, N_SU3>, N_SU3> SU3_structure_cache;
        bool SU3_structure_initialized = false;
        
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t site_idx = 0; site_idx < lattice_size_SU3; ++site_idx) {
            for (size_t a = 0; a < N_SU3; ++a) {
                // Lazily initialize the structure tensor cache once per thread
                if (!SU3_structure_initialized) {
                    #pragma omp critical(SU3_struct_init)
                    {
                        if (!SU3_structure_initialized) {
                            for (size_t i = 0; i < N_SU3; ++i)
                                for (size_t j = 0; j < N_SU3; ++j)
                                    for (size_t k = 0; k < N_SU3; ++k)
                                        SU3_structure_cache[i][j][k] = SU3_structure[i][j][k];
                            SU3_structure_initialized = true;
                        }
                    }
                }
                
                const size_t base_idx = site_idx * SU3_struct_const_size + a * N_SU3 * N_SU3;
                #pragma omp simd collapse(2)
                for (size_t b = 0; b < N_SU3; ++b) {
                    for (size_t c = 0; c < N_SU3; ++c) {
                        SU3_structure_tensor[base_idx + b * N_SU3 + c] = SU3_structure_cache[a][b][c];
                    }
                }
            }
        }
        
        // Get the number of trilinear SU2-SU3 interactions (should be same for all sites)
        num_tri_SU2_SU3 = mixed_trilinear_partners_SU3[0].size();
        
        // Initialize device structure tensor manager - can be done in parallel with above work
        #pragma omp parallel sections
        {
            #pragma omp section
            device_structure_tensor_manager.initSU2(SU2_structure_tensor, N_SU2, N_ATOMS_SU2, dim1, dim2, dim);
            
            #pragma omp section
            device_structure_tensor_manager.initSU3(SU3_structure_tensor, N_SU3, N_ATOMS_SU3, dim1, dim2, dim);
        }

        cout << "Finished setting up lattice" << endl;
        cout << num_bi_SU2 << " " << num_tri_SU2 << " " << num_bi_SU3 << " " << num_tri_SU3 << " " << num_tri_SU2_SU3 << endl;
    }

    void print_lattice_info(const string& filename = "lattice_info_CPU.txt") const {
        ofstream outfile(filename);
        if (!outfile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << endl;
            return;
        }

        outfile << "Lattice size SU2: " << lattice_size_SU2 << endl;
        outfile << "Lattice size SU3: " << lattice_size_SU3 << endl;
        outfile << "Number of bilinear interactions SU2: " << num_bi_SU2 << endl;
        outfile << "Number of trilinear interactions SU2: " << num_tri_SU2 << endl;
        outfile << "Number of bilinear interactions SU3: " << num_bi_SU3 << endl;
        outfile << "Number of trilinear interactions SU3: " << num_tri_SU3 << endl;
        outfile << "Number of mixed trilinear interactions SU2-SU3: " << num_tri_SU2_SU3 << endl;

        // Print SU2 interactions
        outfile << "\nSU2 Interactions:" << endl;
        outfile << "-----------------" << endl;

        // Print onsite interactions for first few sites
        outfile << "Onsite interactions (first 3 sites):" << endl;
        for (size_t i = 0; i < min(size_t(3), lattice_size_SU2); ++i) {
            outfile << "Site " << i << ": ";
            for (size_t j = 0; j < N_SU2 * N_SU2; ++j) {
                outfile << onsite_interaction_SU2[i][j] << " ";
            }
            outfile << endl;
        }

        // Print bilinear interactions and partners for first few sites
        outfile << "\nBilinear interactions (first 3 sites):" << endl;
        for (size_t i = 0; i < min(size_t(3), lattice_size_SU2); ++i) {
            outfile << "Site " << i << " has " << bilinear_partners_SU2[i].size() << " bilinear partners:" << endl;
            for (size_t j = 0; j < bilinear_partners_SU2[i].size(); ++j) {
                outfile << "  Partner: " << bilinear_partners_SU2[i][j] << ", Matrix: ";
                for (size_t k = 0; k < N_SU2 * N_SU2; ++k) {
                    outfile << bilinear_interaction_SU2[i][j][k] << " ";
                }
                outfile << endl;
            }
        }

        // Print trilinear interactions and partners for first few sites
        outfile << "\nTrilinear interactions (first 3 sites):" << endl;
        for (size_t i = 0; i < min(size_t(3), lattice_size_SU2); ++i) {
            outfile << "Site " << i << " has " << trilinear_partners_SU2[i].size() << " trilinear partners:" << endl;
            for (size_t j = 0; j < trilinear_partners_SU2[i].size(); ++j) {
                outfile << "  Partners: (" << trilinear_partners_SU2[i][j][0] << ", " 
                     << trilinear_partners_SU2[i][j][1] << ")" << endl;
            }
        }

        // Print SU3 interactions
        outfile << "\nSU3 Interactions:" << endl;
        outfile << "-----------------" << endl;

        // Print onsite interactions for first few sites
        outfile << "Onsite interactions (first 3 sites):" << endl;
        for (size_t i = 0; i < min(size_t(3), lattice_size_SU3); ++i) {
            outfile << "Site " << i << ": ";
            for (size_t j = 0; j < N_SU3 * N_SU3; ++j) {
                outfile << onsite_interaction_SU3[i][j] << " ";
            }
            outfile << endl;
        }

        // Print bilinear interactions and partners for first few sites
        outfile << "\nBilinear interactions (first 3 sites):" << endl;
        for (size_t i = 0; i < min(size_t(3), lattice_size_SU3); ++i) {
            outfile << "Site " << i << " has " << bilinear_partners_SU3[i].size() << " bilinear partners:" << endl;
            for (size_t j = 0; j < bilinear_partners_SU3[i].size(); ++j) {
                outfile << "  Partner: " << bilinear_partners_SU3[i][j] << ", Matrix: ";
                for (size_t k = 0; k < N_SU3 * N_SU3; ++k) {
                    outfile << bilinear_interaction_SU3[i][j][k] << " ";
                }
                outfile << endl;
            }
        }

        // Print trilinear interactions and partners for first few sites
        outfile << "\nTrilinear interactions (first 3 sites):" << endl;
        for (size_t i = 0; i < min(size_t(3), lattice_size_SU3); ++i) {
            outfile << "Site " << i << " has " << trilinear_partners_SU3[i].size() << " trilinear partners:" << endl;
            for (size_t j = 0; j < trilinear_partners_SU3[i].size(); ++j) {
                outfile << "  Partners: (" << trilinear_partners_SU3[i][j][0] << ", " 
                     << trilinear_partners_SU3[i][j][1] << ")" << endl;
            }
        }

        // Print mixed SU2-SU3 interactions
        outfile << "\nMixed SU2-SU3 Interactions:" << endl;
        outfile << "---------------------------" << endl;

        // Print mixed trilinear interactions from SU2 side
        outfile << "Mixed trilinear interactions from SU2 sites (first 3 sites):" << endl;
        for (size_t i = 0; i < min(size_t(3), lattice_size_SU2); ++i) {
            outfile << "SU2 Site " << i << " has " << mixed_trilinear_partners_SU2[i].size() << " mixed partners:" << endl;
            for (size_t j = 0; j < mixed_trilinear_partners_SU2[i].size(); ++j) {
                outfile << "  Partners: (SU2 site " << mixed_trilinear_partners_SU2[i][j][0] 
                     << ", SU3 site " << mixed_trilinear_partners_SU2[i][j][1] << "), Tensor: ";
                // Print the trilinear interaction tensor (N_SU2 * N_SU2 * N_SU3 elements)
                for (size_t k = 0; k < N_SU2 * N_SU2 * N_SU3; ++k) {
                    outfile << mixed_trilinear_interaction_SU2[i][j][k] << " ";
                }
                outfile << endl;
            }
        }

        // Print mixed trilinear interactions from SU3 side
        outfile << "\nMixed trilinear interactions from SU3 sites (first 3 sites):" << endl;
        for (size_t i = 0; i < min(size_t(3), lattice_size_SU3); ++i) {
            outfile << "SU3 Site " << i << " has " << mixed_trilinear_partners_SU3[i].size() << " mixed partners:" << endl;
            for (size_t j = 0; j < mixed_trilinear_partners_SU3[i].size(); ++j) {
                outfile << "  Partners: (SU2 site " << mixed_trilinear_partners_SU3[i][j][0] 
                     << ", SU2 site " << mixed_trilinear_partners_SU3[i][j][1] << "), Tensor: ";
                // Print the trilinear interaction tensor (N_SU3 * N_SU2 * N_SU2 elements)
                for (size_t k = 0; k < N_SU3 * N_SU2 * N_SU2; ++k) {
                    outfile << mixed_trilinear_interaction_SU3[i][j][k] << " ";
                }
                outfile << endl;
            }
        }

        outfile.close();
        cout << "Lattice information written to " << filename << endl;
    }

    void read_spin_from_file(const string &filename){
        ifstream file;
        file.open(filename+"_SU2.txt");
        if (!file){
            cout << "Unable to open file " + filename +"_SU2.txt";
            exit(1);
        }
        string line;
        size_t count = 0;
        while(getline(file, line)){
            istringstream iss(line);
            array<double, N_SU2> spin;
            for(size_t i = 0; i < N_SU2; ++i){
                iss >> spin[i];
            }
            spins.spins_SU2[count] = spin;
            count++;
        }
        file.close();

        file.open(filename+"_SU3.txt");
        if (!file){
            cout << "Unable to open file"+filename+"_SU3.txt";
            exit(1);
        }
        count = 0;
        while(getline(file, line)){
            istringstream iss(line);
            array<double, N_SU3> spin;
            for(size_t i = 0; i < N_SU3; ++i){
                iss >> spin[i];
            }
            spins.spins_SU3[count] = spin;
            count++;
        }
        file.close();
    }
    double site_energy_SU2(array<double, N_SU2> &spin_here, size_t site_index){
        double energy = 0.0;
        energy -= dot(spin_here, field_SU2[site_index]);
        energy += contract(spin_here, onsite_interaction_SU2[site_index], spin_here);

        #pragma omp simd
        for (size_t i=0; i<num_bi_SU2; ++i) {
            energy += contract(spin_here, bilinear_interaction_SU2[site_index][i], spins.spins_SU2[bilinear_partners_SU2[site_index][i]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU2; ++i){
            energy += contract_trilinear(trilinear_interaction_SU2[site_index][i], spin_here, spins.spins_SU2[trilinear_partners_SU2[site_index][i][0]], spins.spins_SU2[trilinear_partners_SU2[site_index][i][1]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            energy += contract_trilinear(mixed_trilinear_interaction_SU2[site_index][i], spin_here, spins.spins_SU2[mixed_trilinear_partners_SU2[site_index][i][0]], spins.spins_SU3[mixed_trilinear_partners_SU2[site_index][i][1]]);
        }
        return energy;
    }

    double site_energy_SU3(array<double, N_SU3> &spin_here, size_t site_index){
        double energy = 0.0;
        energy -= dot(spin_here, field_SU3[site_index]);
        energy += contract(spin_here, onsite_interaction_SU3[site_index], spin_here);

        #pragma omp simd
        for (size_t i=0; i<num_bi_SU3; ++i) {
            energy += contract(spin_here, bilinear_interaction_SU3[site_index][i], spins.spins_SU3[bilinear_partners_SU3[site_index][i]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU3; ++i){
            energy += contract_trilinear(trilinear_interaction_SU3[site_index][i], spin_here, spins.spins_SU3[trilinear_partners_SU3[site_index][i][0]], spins.spins_SU3[trilinear_partners_SU3[site_index][i][1]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            energy += contract_trilinear(mixed_trilinear_interaction_SU3[site_index][i], spin_here, spins.spins_SU2[mixed_trilinear_partners_SU3[site_index][i][0]], spins.spins_SU2[mixed_trilinear_partners_SU3[site_index][i][1]]);
        }
        return energy;
    }

    double total_energy(mixed_lattice_spin<N_SU2, dim1*dim2*dim*N_ATOMS_SU2, N_SU3, dim1*dim2*dim*N_ATOMS_SU3> &curr_spins){
        double field_energy = 0.0;
        double onsite_energy = 0.0;
        double bilinear_energy = 0.0;
        double trilinear_energy = 0.0;

        size_t site_index, i;
        #pragma omp simd
        for(site_index = 0; site_index < lattice_size_SU2; ++site_index){
            field_energy -= dot(curr_spins.spins_SU2[site_index], field_SU2[site_index]);
            onsite_energy += contract(curr_spins.spins_SU2[site_index], onsite_interaction_SU2[site_index], curr_spins.spins_SU2[site_index]);

            // #pragma omp simd reduction(+:bilinear_energy)
            for (i=0; i<num_bi_SU2; ++i) {
                bilinear_energy += contract(curr_spins.spins_SU2[site_index], bilinear_interaction_SU2[site_index][i], spins.spins_SU2[bilinear_partners_SU2[site_index][i]]);
            }
            // #pragma omp simd reduction(+:trilinear_energy)
            for (i=0; i < num_tri_SU3; ++i){
                trilinear_energy += contract_trilinear(trilinear_interaction_SU2[site_index][i], curr_spins.spins_SU2[site_index], spins.spins_SU2[trilinear_partners_SU2[site_index][i][0]], spins.spins_SU2[trilinear_partners_SU2[site_index][i][1]]);
            }
            // #pragma omp simd reduction(+:trilinear_energy)
            for (i=0; i < num_tri_SU2_SU3; ++i){
                trilinear_energy += contract_trilinear(mixed_trilinear_interaction_SU2[site_index][i], curr_spins.spins_SU2[site_index], spins.spins_SU2[mixed_trilinear_partners_SU2[site_index][i][0]], spins.spins_SU3[mixed_trilinear_partners_SU2[site_index][i][1]]);
            }
        }

        #pragma omp simd
        for(size_t site_index = 0; site_index < lattice_size_SU3; ++site_index){
            field_energy -= dot(curr_spins.spins_SU3[site_index], field_SU3[site_index]);
            onsite_energy += contract(curr_spins.spins_SU3[site_index], onsite_interaction_SU3[site_index], curr_spins.spins_SU3[site_index]);

            // #pragma omp simd reduction(+:bilinear_energy)
            for (size_t i=0; i<num_bi_SU3; ++i) {
                bilinear_energy += contract(curr_spins.spins_SU3[site_index], bilinear_interaction_SU3[site_index][i], spins.spins_SU3[bilinear_partners_SU3[site_index][i]]);
            }
            // #pragma omp simd reduction(+:trilinear_energy)
            for (size_t i=0; i < num_tri_SU3; ++i){
                trilinear_energy += contract_trilinear(trilinear_interaction_SU3[site_index][i], curr_spins.spins_SU3[site_index], spins.spins_SU3[trilinear_partners_SU3[site_index][i][0]], spins.spins_SU3[trilinear_partners_SU3[site_index][i][1]]);
            }
            // #pragma omp simd reduction(+:trilinear_energy)
            for (size_t i=0; i < num_tri_SU2_SU3; ++i){
                trilinear_energy += contract_trilinear(mixed_trilinear_interaction_SU3[site_index][i], curr_spins.spins_SU3[site_index], spins.spins_SU2[mixed_trilinear_partners_SU3[site_index][i][0]], spins.spins_SU2[mixed_trilinear_partners_SU3[site_index][i][1]]);
            }
        }
        return field_energy + onsite_energy/2 + bilinear_energy/2 + trilinear_energy/3;
    }

    double energy_density(mixed_lattice_spin<N_SU2, dim1*dim2*dim*N_ATOMS_SU2, N_SU3, dim1*dim2*dim*N_ATOMS_SU3>  &curr_spins){
        return total_energy(curr_spins)/(lattice_size_SU2+lattice_size_SU3);
    }

    void set_pulse_SU2(const array<array<double,N_SU2>, N_ATOMS_SU2> &field_in_SU2, double t_B, const array<array<double,N_SU2>, N_ATOMS_SU2> &field_in_2_SU2, double t_B_2, double pulse_amp, double pulse_width, double pulse_freq){
        field_drive_1_SU2 = field_in_SU2;
        field_drive_2_SU2 = field_in_2_SU2;
        field_drive_amp_SU2 = pulse_amp;
        field_drive_freq_SU2 = pulse_freq;
        field_drive_width_SU2 = pulse_width;
        t_B_1_SU2 = t_B;
        t_B_2_SU2 = t_B_2;
    }

    void set_pulse_SU3(const array<array<double,N_SU3>, N_ATOMS_SU3> &field_in_SU3, double t_B, const array<array<double,N_SU3>, N_ATOMS_SU3> &field_in_2_SU3, double t_B_2, double pulse_amp, double pulse_width, double pulse_freq){
        field_drive_1_SU3 = field_in_SU3;
        field_drive_2_SU3 = field_in_2_SU3;
        field_drive_amp_SU3 = pulse_amp;
        field_drive_freq_SU3 = pulse_freq;
        field_drive_width_SU3 = pulse_width;
        t_B_1_SU3 = t_B;
        t_B_2_SU3 = t_B_2;
    }
    
    void reset_pulse(){
        field_drive_1_SU2 = {{0}};
        field_drive_2_SU2 = {{0}};
        field_drive_amp_SU2 = 0;
        field_drive_freq_SU2 = 0;
        field_drive_width_SU2 = 1;
        t_B_1_SU2 = 0;
        t_B_2_SU2 = 0;

        field_drive_1_SU3 = {{0}};
        field_drive_2_SU3 = {{0}};
        field_drive_amp_SU3 = 0;
        field_drive_freq_SU3 = 0;
        field_drive_width_SU3 = 1;
        t_B_1_SU3 = 0;
        t_B_2_SU3 = 0;
    }

    array<double, N_SU2>  get_local_field_SU2(size_t site_index){
        array<double,N_SU2> local_field = multiply(onsite_interaction_SU2[site_index], spins.spins_SU2[site_index]);

        #pragma omp simd
        for (size_t i=0; i< num_bi_SU2; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU2[site_index][i], spins.spins_SU2[bilinear_partners_SU2[site_index][i]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU2; ++i){
            size_t partner1 = trilinear_partners_SU2[site_index][i][0];
            size_t partner2 = trilinear_partners_SU2[site_index][i][1];
            array<double, N_SU2> current_spin_SU2_partner1 = spins.spins_SU2[partner1];
            array<double, N_SU2> current_spin_SU2_partner2 = spins.spins_SU2[partner2];
            local_field = local_field + contract_trilinear_field<N_SU2, N_SU2, N_SU2>(trilinear_interaction_SU2[site_index][i], current_spin_SU2_partner1, current_spin_SU2_partner2);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            size_t partner1 = mixed_trilinear_partners_SU2[site_index][i][0];
            size_t partner2 = mixed_trilinear_partners_SU2[site_index][i][1];
            array<double, N_SU2> current_spin_SU2_partner1 = spins.spins_SU2[partner1];
            array<double, N_SU3> current_spin_SU3_partner2 = spins.spins_SU3[partner2];
            local_field = local_field + contract_trilinear_field<N_SU2, N_SU2, N_SU3>(mixed_trilinear_interaction_SU2[site_index][i], current_spin_SU2_partner1, current_spin_SU3_partner2);
        }
        return local_field-field_SU2[site_index];
    }

    array<double, N_SU3>  get_local_field_SU3(size_t site_index){
        array<double,N_SU3> local_field = multiply(onsite_interaction_SU3[site_index], spins.spins_SU3[site_index]);
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU3; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU3[site_index][i], spins.spins_SU3[bilinear_partners_SU3[site_index][i]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU3; ++i){
            size_t partner1 = trilinear_partners_SU3[site_index][i][0];
            size_t partner2 = trilinear_partners_SU3[site_index][i][1];
            array<double, N_SU3> current_spin_SU3_partner1 = spins.spins_SU3[partner1];
            array<double, N_SU3> current_spin_SU3_partner2 = spins.spins_SU3[partner2];
            local_field = local_field + contract_trilinear_field<N_SU3, N_SU3, N_SU3>(trilinear_interaction_SU3[site_index][i], current_spin_SU3_partner1, current_spin_SU3_partner2);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            size_t partner1 = mixed_trilinear_partners_SU3[site_index][i][0];
            size_t partner2 = mixed_trilinear_partners_SU3[site_index][i][1];
            array<double, N_SU2> current_spin_SU2_partner1 = spins.spins_SU2[partner1];
            array<double, N_SU2> current_spin_SU2_partner2 = spins.spins_SU2[partner2];
            local_field = local_field + contract_trilinear_field<N_SU3, N_SU2, N_SU2>(mixed_trilinear_interaction_SU3[site_index][i], current_spin_SU2_partner1, current_spin_SU2_partner2);
        }
        return local_field-field_SU3[site_index];
    }

    array<double, N_SU2>  get_local_field_SU2_lattice(size_t site_index, const spin_config_SU2 &current_spin_SU2, const spin_config_SU3 &current_spin_SU3){
        array<double,N_SU2> local_field = {{0.0}};
        local_field += multiply(onsite_interaction_SU2[site_index], current_spin_SU2[site_index]);
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU2; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU2[site_index][i], current_spin_SU2[bilinear_partners_SU2[site_index][i]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU2; ++i){
            size_t partner1 = trilinear_partners_SU2[site_index][i][0];
            size_t partner2 = trilinear_partners_SU2[site_index][i][1];
            array<double, N_SU2> current_spin_SU2_partner1 = current_spin_SU2[partner1];
            array<double, N_SU2> current_spin_SU2_partner2 = current_spin_SU2[partner2];
            local_field = local_field + contract_trilinear_field<N_SU2, N_SU2, N_SU2>(trilinear_interaction_SU2[site_index][i], current_spin_SU2_partner1, current_spin_SU2_partner2);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            size_t partner1 = mixed_trilinear_partners_SU2[site_index][i][0];
            size_t partner2 = mixed_trilinear_partners_SU2[site_index][i][1];
            array<double, N_SU2> current_spin_SU2_partner1 = current_spin_SU2[partner1];
            array<double, N_SU3> current_spin_SU3_partner2 = current_spin_SU3[partner2];
            local_field = local_field + contract_trilinear_field<N_SU2, N_SU2, N_SU3>(mixed_trilinear_interaction_SU2[site_index][i], current_spin_SU2_partner1, current_spin_SU3_partner2);
        }
        return local_field-field_SU2[site_index];
    }

    array<double, N_SU3>  get_local_field_SU3_lattice(size_t site_index, const spin_config_SU2 &current_spin_SU2, const spin_config_SU3 &current_spin_SU3){
        array<double,N_SU3> local_field = {{0.0}};
        local_field += multiply(onsite_interaction_SU3[site_index], current_spin_SU3[site_index]);
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU3; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU3[site_index][i], current_spin_SU3[bilinear_partners_SU3[site_index][i]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU3; ++i){
            size_t partner1 = trilinear_partners_SU3[site_index][i][0];
            size_t partner2 = trilinear_partners_SU3[site_index][i][1];
            array<double, N_SU3> current_spin_SU3_partner1 = current_spin_SU3[partner1];
            array<double, N_SU3> current_spin_SU3_partner2 = current_spin_SU3[partner2];
            local_field = local_field + contract_trilinear_field<N_SU3, N_SU3, N_SU3>(trilinear_interaction_SU3[site_index][i], current_spin_SU3_partner1, current_spin_SU3_partner2);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            size_t partner1 = mixed_trilinear_partners_SU3[site_index][i][0];
            size_t partner2 = mixed_trilinear_partners_SU3[site_index][i][1];
            array<double, N_SU2> current_spin_SU2_partner1 = current_spin_SU2[partner1];
            array<double, N_SU2> current_spin_SU2_partner2 = current_spin_SU2[partner2];
            local_field = local_field + contract_trilinear_field<N_SU3, N_SU2, N_SU2>(mixed_trilinear_interaction_SU3[site_index][i], current_spin_SU2_partner1, current_spin_SU2_partner2);
        }
        return local_field-field_SU3[site_index];
    }

    array<double,N_SU2> gaussian_move_SU2(const array<double,N_SU2> &current_spin, double sigma=60){
        array<double,N_SU2> new_spin;
        new_spin = current_spin + gen_random_spin_SU2()*sigma;
        return new_spin/sqrt(dot(new_spin, new_spin)) * spin_length_SU2;
    }

    array<double,N_SU3> gaussian_move_SU3(const array<double,N_SU3> &current_spin, double sigma=60){
        array<double,N_SU3> new_spin;
        new_spin = current_spin + gen_random_spin_SU3()*sigma;
        return new_spin/sqrt(dot(new_spin, new_spin)) * spin_length_SU3;
    }

    double metropolis(mixed_lattice_spin<N_SU2, dim1*dim2*dim*N_ATOMS_SU2, N_SU3, dim1*dim2*dim*N_ATOMS_SU3>& curr_spin, double T, bool gaussian=false, double sigma=60) {
        int accept = 0;
        const size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        const double inv_T = 1.0 / T;  // Precompute inverse temperature
        
        // Process sites in a batch for better cache locality
        #pragma omp parallel for reduction(+:accept)
        for (size_t count = 0; count < total_sites; ++count) {
            size_t i = random_int_lehman(total_sites);

            if (i < lattice_size_SU2) {
                // SU2 case
                array<double,N_SU2> new_spin_SU2;
                const double E_old = site_energy_SU2(curr_spin.spins_SU2[i], i);
                
                // Generate new spin configuration
                if (gaussian) {
                    new_spin_SU2 = gaussian_move_SU2(curr_spin.spins_SU2[i], sigma);
                } else {
                    new_spin_SU2 = gen_random_spin_SU2();
                }
                
                const double E_new = site_energy_SU2(new_spin_SU2, i);
                const double dE = E_new - E_old;
                
                // Fast path for acceptance when dE <= 0
                bool accepted = (dE <= 0);
                if (!accepted) {
                    // Only compute exp when needed
                    accepted = (random_double_lehman(0,1) < exp(-dE * inv_T));
                }
                
                if (accepted) {
                    curr_spin.spins_SU2[i] = new_spin_SU2;
                    accept++;
                }
            } else {
                // SU3 case
                const size_t i_SU3 = i - lattice_size_SU2;
                array<double,N_SU3> new_spin_SU3;
                const double E_old = site_energy_SU3(curr_spin.spins_SU3[i_SU3], i_SU3);
                
                // Generate new spin configuration
                if (gaussian) {
                    new_spin_SU3 = gaussian_move_SU3(curr_spin.spins_SU3[i_SU3], sigma);
                } else {
                    new_spin_SU3 = gen_random_spin_SU3();
                }
                
                const double E_new = site_energy_SU3(new_spin_SU3, i_SU3);
                const double dE = E_new - E_old;
                
                // Fast path for acceptance when dE <= 0
                bool accepted = (dE <= 0);
                if (!accepted) {
                    // Only compute exp when needed
                    accepted = (random_double_lehman(0,1) < exp(-dE * inv_T));
                }
                
                if (accepted) {
                    curr_spin.spins_SU3[i_SU3] = new_spin_SU3;
                    accept++;
                }
            }
        }
        return static_cast<double>(accept) / double(total_sites);
    }
    void overrelaxation(){
        array<double,N_SU2> local_field_SU2;
        array<double,N_SU3> local_field_SU3;
        int i;
        double proj;
        for(size_t count = 0; count < lattice_size_SU2; ++count){
            // i = random_int(0, lattice_size-1, gen);
            i = random_int_lehman(lattice_size_SU2);
            local_field_SU2 = get_local_field_SU2(i);
            double norm = dot(local_field_SU2, local_field_SU2);
            if(norm == 0){
                continue;
            }
            else{
                proj = 2* dot(spins.spins_SU2[i], local_field_SU2)/norm;
                spins.spins_SU2[i] = local_field_SU2*proj - spins.spins_SU2[i];
            }
            count++;
        }
        for(size_t count = 0; count < lattice_size_SU3; ++count){
            // i = random_int(0, lattice_size-1, gen);
            i = random_int_lehman(lattice_size_SU3);
            local_field_SU3 = get_local_field_SU3(i);
            double norm = dot(local_field_SU3, local_field_SU3);
            if(norm == 0){
                continue;
            }
            else{
                proj = 2* dot(spins.spins_SU3[i], local_field_SU3)/norm;
                spins.spins_SU3[i] = local_field_SU3*proj - spins.spins_SU3[i];
            }
            count++;
        }
    }
    void deterministic_sweep(){

        #pragma omp parallel for simd
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            array<double,N_SU2> local_field = get_local_field_SU2(i);
            double norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                spins.spins_SU2[i] = local_field/(-norm)*spin_length_SU2;
            }
        }

        #pragma omp parallel for simd
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            array<double,N_SU3> local_field = get_local_field_SU3(i);
            double norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                spins.spins_SU3[i] = local_field/(-norm)*spin_length_SU3;
            }
        }
    }
    
    void write_to_file_spin(string filename){
        ofstream myfile;
        myfile.open(filename+"_SU2.txt");
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            for(size_t j = 0; j<3; ++j){
                myfile << spins.spins_SU2[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
        myfile.open(filename+"_SU3.txt");
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            for(size_t j = 0; j<8; ++j){
                myfile << spins.spins_SU3[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }


    void write_to_file_spin_t(string filename){
        ofstream myfile;
        myfile.open(filename+"_SU2.txt", ios::app);
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            for(size_t j = 0; j<3; ++j){
                myfile << spins.spins_SU2[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
        myfile.open(filename+"_SU3.txt", ios::app);
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            for(size_t j = 0; j<8; ++j){
                myfile << spins.spins_SU3[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_to_file_pos(string filename){
        ofstream myfile;
        myfile.open(filename+"_SU2.txt");
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            for(size_t j = 0; j<3; ++j){
                myfile << site_pos.pos_SU2[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
        myfile.open(filename+"_SU3.txt");
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            for(size_t j = 0; j<3; ++j){
                myfile << site_pos.pos_SU3[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_to_file(string filename, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> towrite){
        ofstream myfile;
        myfile.open(filename+"_SU2.txt", ios::app);
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            for(size_t j = 0; j<3; ++j){
                myfile << towrite.spins_SU2[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
        myfile.open(filename+"_SU3.txt", ios::app);
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            for(size_t j = 0; j<8; ++j){
                myfile << towrite.spins_SU3[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void print_mixed_spin(mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> toprint){
        for (size_t i = 0; i < lattice_size_SU2; ++i){
            cout << "SU2: ";
            for (size_t j = 0; j < 3; ++j){
                cout << toprint.spins_SU2[i][j] << " ";
            }
            cout << endl;
        }
        for (size_t i = 0; i < lattice_size_SU3; ++i){
            cout << "SU3: ";
            for (size_t j = 0; j < 8; ++j){
                cout << toprint.spins_SU3[i][j] << " ";
            }
            cout << endl;
        }
    }


    void simulated_annealing(double T_start, double T_end, size_t n_anneal, size_t n_deterministics, size_t overrelaxation_rate, bool gaussian_move=true, string dir_name=""){
        srand (time(NULL));
        seed_lehman(rand()*2+1);
        double curr_accept;
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        double T = T_start;
        double sigma = 1000;
        double acceptance_rate = 0;
        std::cout << "Starting simulated annealing with T_start: " << T_start << " and T_end: " << T_end << std::endl;
        std::cout << "Number of anneal steps: " << n_anneal << std::endl;
        std::cout << "Number of deterministic steps: " << n_deterministics << std::endl;
        std::cout << "Overrelaxation rate: " << overrelaxation_rate << std::endl;
        std::cout << "Gaussian move: " << gaussian_move << std::endl;
        while(T > T_end){
            curr_accept = 0;
            for(size_t i = 0; i<n_anneal; ++i){
                if(overrelaxation_rate > 0){
                    overrelaxation();
                    if (i%overrelaxation_rate == 0){
                        curr_accept += metropolis(spins, T, gaussian_move,sigma);
                    }
                }
                else{
                    curr_accept += metropolis(spins, T, gaussian_move,sigma);
                }
            }
            if (overrelaxation_rate > 0){
                acceptance_rate = curr_accept/n_anneal*overrelaxation_rate;
                cout << "Temperature: " << T << " Acceptance rate: " << acceptance_rate << endl;
            }else{
                acceptance_rate = curr_accept/n_anneal;
                cout << "Temperature: " << T << " Acceptance rate: " << acceptance_rate << endl;
            }
            if (gaussian_move && acceptance_rate < 0.5){
                sigma = sigma * 0.5 / (1-acceptance_rate); 
                cout << "Sigma is adjusted to: " << sigma << endl;   
            }
            T *= 0.9;
        }

        for (size_t i = 0; i < n_deterministics; ++i){
            deterministic_sweep();
        }
        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            write_to_file_spin(dir_name + "/spin");
            write_to_file_pos(dir_name + "/pos");
        }
    }


    void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure, size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate, string dir_name, const vector<int> rank_to_write, bool gaussian_move = true){

        int initialized;
        int swap_accept = 0;
        double curr_accept = 0;
        int overrelaxation_flag = overrelaxation_rate > 0 ? overrelaxation_rate : 1;
        MPI_Initialized(&initialized);
        if (!initialized){
            MPI_Init(NULL, NULL);
        }
        int rank, size, partner_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        srand (time(NULL));
        seed_lehman(rand()*2+1);
        double E, T_partner, E_partner;
        bool accept;        
        spin_config_SU2 newspins_SU2;
        spin_config_SU3 newspins_SU3;
        double curr_Temp = temp[rank];
        // vector<double> heat_capacity, dHeat;
        // if (rank == 0){
        //     heat_capacity.resize(size);
        //     dHeat.resize(size);
        // }   
        // vector<double> energies;
        // vector<array<double,N>> magnetizations;
        // vector<spin_config> spin_configs_at_temp;

        cout << "Initialized Process on rank: " << rank << " with temperature: " << curr_Temp << endl;

        for(size_t i=0; i < n_anneal+n_measure; ++i){

            // Metropolisfh
            if(overrelaxation_rate > 0){
                overrelaxation();
                if (i%overrelaxation_rate == 0){
                    curr_accept += metropolis(spins, curr_Temp, gaussian_move);
                }
            }
            else{
                curr_accept += metropolis(spins, curr_Temp, gaussian_move);
            }
            E = total_energy(spins);

            if ((i % swap_rate == 0) && (i % overrelaxation_flag == 0)){
                accept = false;
                if ((i / swap_rate) % 2 ==0){
                    partner_rank = rank % 2 == 0 ? rank + 1 : rank - 1;
                }else{
                    partner_rank = rank % 2 == 0 ? rank - 1 : rank + 1;
                }
                if ((partner_rank >= 0) && (partner_rank < size)){
                    T_partner = temp[partner_rank];
                    if (partner_rank % 2 == 0){
                        MPI_Send(&E, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD);
                        MPI_Recv(&E_partner, 1, MPI_DOUBLE, partner_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    } else{
                        MPI_Recv(&E_partner, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Send(&E, 1, MPI_DOUBLE, partner_rank, 1, MPI_COMM_WORLD);
                    }
                    if (partner_rank % 2 == 0){
                        accept = min(double(1.0), exp((1/curr_Temp-1/T_partner)*(E - E_partner))) > random_double_lehman(0,1);
                        MPI_Send(&accept, 1, MPI_C_BOOL, partner_rank, 2, MPI_COMM_WORLD);
                    } else{
                        MPI_Recv(&accept, 1, MPI_C_BOOL, partner_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    if (accept){
                        if (partner_rank % 2 == 0){
                            MPI_Send(&(spins.spins_SU2), N_SU2*lattice_size_SU2, MPI_DOUBLE, partner_rank, 4, MPI_COMM_WORLD);
                            MPI_Recv(&newspins_SU2, N_SU2*lattice_size_SU2, MPI_DOUBLE, partner_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            MPI_Send(&(spins.spins_SU3), N_SU3*lattice_size_SU3, MPI_DOUBLE, partner_rank, 6, MPI_COMM_WORLD);
                            MPI_Recv(&newspins_SU3, N_SU3*lattice_size_SU3, MPI_DOUBLE, partner_rank, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        } else{
                            MPI_Recv(&newspins_SU2, N_SU2*lattice_size_SU2, MPI_DOUBLE, partner_rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            MPI_Send(&(spins.spins_SU2), N_SU2*lattice_size_SU2, MPI_DOUBLE, partner_rank, 3, MPI_COMM_WORLD);
                            MPI_Recv(&newspins_SU3, N_SU3*lattice_size_SU3, MPI_DOUBLE, partner_rank, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            MPI_Send(&(spins.spins_SU3), N_SU3*lattice_size_SU3, MPI_DOUBLE, partner_rank, 5, MPI_COMM_WORLD);
                        }
                        copy(newspins_SU2.begin(), newspins_SU2.end(), spins.spins_SU2.begin());
                        copy(newspins_SU3.begin(), newspins_SU3.end(), spins.spins_SU3.begin());
                        E = E_partner;
                        swap_accept++;
                    }
                }
            }
            // if (i >= n_anneal){
            //     if (i % probe_rate == 0){
            //         if(dir_name != ""){
            //             magnetizations.push_back(magnetization_local(spins));
            //             spin_configs_at_temp.push_back(spins);
            //             energies.push_back(E);
            //         }
            //     }
            // }
            if (i % 10000 == 0){
                std::cout << "Percentage sweep done: " << double(i)/double(n_anneal+n_measure) * 100 << " % on rank: " << rank << endl;
            }
        }
        
        // std::tuple<double,double> varE = binning_analysis(energies, int(energies.size()/10));
        // double curr_heat_capacity = 1/(curr_Temp*curr_Temp)*get<0>(varE)/lattice_size;
        // double curr_dHeat = 1/(curr_Temp*curr_Temp)*get<1>(varE)/lattice_size;
        // MPI_Gather(&curr_heat_capacity, 1, MPI_DOUBLE, heat_capacity.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Gather(&curr_dHeat, 1, MPI_DOUBLE, dHeat.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        cout << "Process finished on rank: " << rank << " with temperature: " << curr_Temp << " with local acceptance rate: " << double(curr_accept)/double(n_anneal+n_measure)*overrelaxation_flag << " Swap Acceptance rate: " << double(swap_accept)/double(n_anneal+n_measure)*swap_rate*overrelaxation_flag << endl;
        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            for(size_t i=0; i<rank_to_write.size(); ++i){
                if (rank == rank_to_write[i]){
                    write_to_file_spin(dir_name + "/spin" + to_string(rank));
                    // write_to_file_2d_vector_array(dir_name + "/magnetization" + to_string(rank) + ".txt", magnetizations);
                    // write_column_vector(dir_name + "/energy" + to_string(rank) + ".txt", energies);
                    // for(size_t a=0; a<spin_configs_at_temp.size(); ++a){
                    //     write_to_file_spin(dir_name + "/spin" + to_string(rank) + "_T" + to_string(temp[a]) + ".txt", spin_configs_at_temp[a]);
                    // }
                }
            }
            // if (rank == 0){
            //     write_to_file_pos(dir_name + "/pos.txt");
            //     ofstream myfile;
            //     myfile.open(dir_name + "/heat_capacity.txt", ios::app);
            //     for(size_t j = 0; j<size; ++j){
            //         myfile << temp[j] << " " << heat_capacity[j] << " " << dHeat[j] << endl;
            //     }
            //     myfile.close();
            // }
        }
        int finalized;
        if (!MPI_Finalized(&finalized)){
            MPI_Finalize();
        }
        // measurement("spin0.txt", temp[0], n_measure, probe_rate, overrelaxation_rate, gaussian_move, rank_to_write, dir_name);
    }


    const array<double, N_SU2> drive_field_T_SU2(double currT, size_t ind){
        double factor1_SU2 = double(field_drive_amp_SU2*exp(-pow((currT-t_B_1_SU2)/(2*field_drive_width_SU2),2))*cos(2*M_PI*field_drive_freq_SU2*(currT-t_B_1_SU2)));
        double factor2_SU2 = double(field_drive_amp_SU2*exp(-pow((currT-t_B_2_SU2)/(2*field_drive_width_SU2),2))*cos(2*M_PI*field_drive_freq_SU2*(currT-t_B_2_SU2)));
        return field_drive_1_SU2[ind]*factor1_SU2 + field_drive_2_SU2[ind]*factor2_SU2;
    }

    const array<double, N_SU3> drive_field_T_SU3(double currT, size_t ind){
        double factor1_SU3 = double(field_drive_amp_SU3*exp(-pow((currT-t_B_1_SU3)/(2*field_drive_width_SU3),2))*cos(2*M_PI*field_drive_freq_SU3*(currT-t_B_1_SU3)));
        double factor2_SU3 = double(field_drive_amp_SU3*exp(-pow((currT-t_B_2_SU3)/(2*field_drive_width_SU3),2))*cos(2*M_PI*field_drive_freq_SU3*(currT-t_B_2_SU3)));
        return field_drive_1_SU3[ind]*factor1_SU3 + field_drive_2_SU3[ind]*factor2_SU3;
    }

    spin_config_SU2 landau_lifshitz_SU2(const spin_config_SU2 &current_spin_SU2, const spin_config_SU3 &current_spin_SU3, const double &curr_time) {
        // Use thread-local static vectors to avoid repeated allocations
        std::vector<double> fields_flat;
        std::vector<double> spins_flat;
        std::vector<double> result_flat;
        
        const size_t total_elements = lattice_size_SU2 * N_SU2;
        
        // Resize only if needed
        if (fields_flat.size() != total_elements) {
            fields_flat.resize(total_elements);
            spins_flat.resize(total_elements);
            result_flat.resize(total_elements);
        }

        // Combine field computation and flattening in a single pass
        #pragma omp parallel for schedule(static) collapse(2)
        for(size_t i = 0; i < lattice_size_SU2; ++i) {
            for(size_t comp = 0; comp < N_SU2; ++comp) {
                const size_t flat_idx = i * N_SU2 + comp;
                spins_flat[flat_idx] = current_spin_SU2[i][comp];
            }
        }
        
        // Compute local fields and flatten in one pass
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < lattice_size_SU2; ++i) {
            array<double, N_SU2> local_field = get_local_field_SU2_lattice(i, current_spin_SU2, current_spin_SU3);
            array<double, N_SU2> drive_field = drive_field_T_SU2(curr_time, i % N_ATOMS_SU2);
            
            const size_t base_idx = i * N_SU2;
            #pragma omp simd
            for(size_t comp = 0; comp < N_SU2; ++comp) {
                fields_flat[base_idx + comp] = local_field[comp] - drive_field[comp];
            }
        }
        
        // Call CUDA wrapper
        device_structure_tensor_manager.contractSU2<double, N_ATOMS_SU2*dim1*dim2*dim, N_SU2>(
            fields_flat,
            spins_flat,
            result_flat
        );
        
        // Convert back to spin_config_SU2
        spin_config_SU2 dS;
        #pragma omp parallel for schedule(static) collapse(2)
        for(size_t site = 0; site < lattice_size_SU2; ++site) {
            for(size_t comp = 0; comp < N_SU2; ++comp) {
                dS[site][comp] = result_flat[site * N_SU2 + comp];
            }
        }
        return dS;
    }

    spin_config_SU3 landau_lifshitz_SU3(const spin_config_SU2 &current_spin_SU2, const spin_config_SU3 &current_spin_SU3, const double &curr_time){
        // Use thread-local static vectors to avoid repeated allocations
        std::vector<double> fields_flat;
        std::vector<double> spins_flat;
        std::vector<double> result_flat;
        
        const size_t total_elements = lattice_size_SU3 * N_SU3;
        
        // Resize only if needed
        if (fields_flat.size() != total_elements) {
            fields_flat.resize(total_elements);
            spins_flat.resize(total_elements);
            result_flat.resize(total_elements);
        }

        // Combine field computation and flattening in a single pass
        #pragma omp parallel for schedule(static) collapse(2)
        for(size_t i = 0; i < lattice_size_SU3; ++i) {
            for(size_t comp = 0; comp < N_SU3; ++comp) {
                const size_t flat_idx = i * N_SU3 + comp;
                spins_flat[flat_idx] = current_spin_SU3[i][comp];
            }
        }
        
        // Compute local fields and flatten in one pass
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < lattice_size_SU3; ++i) {
            array<double, N_SU3> local_field = get_local_field_SU3_lattice(i, current_spin_SU2, current_spin_SU3);
            array<double, N_SU3> drive_field = drive_field_T_SU3(curr_time, i % N_ATOMS_SU3);
            
            const size_t base_idx = i * N_SU3;
            #pragma omp simd
            for(size_t comp = 0; comp < N_SU3; ++comp) {
                fields_flat[base_idx + comp] = local_field[comp] - drive_field[comp];
            }
        }
        
        // Call CUDA wrapper
        device_structure_tensor_manager.contractSU3<double, N_ATOMS_SU3*dim1*dim2*dim, N_SU3>(
            fields_flat,
            spins_flat,
            result_flat
        );
        
        // Convert back to spin_config_SU3
        spin_config_SU3 dS;
        #pragma omp parallel for schedule(static) collapse(2)
        for(size_t site = 0; site < lattice_size_SU3; ++site) {
            for(size_t comp = 0; comp < N_SU3; ++comp) {
                dS[site][comp] = result_flat[site * N_SU3 + comp];
            }
        }
        return dS;
    }

    // Previous CPU implementation of landau_lifshitz equations
    // spin_config_SU2 landau_lifshitz_SU2(const spin_config_SU2 &current_spin_SU2, const spin_config_SU3 &current_spin_SU3, const double &curr_time){
    //     spin_config_SU2 dS;
    //     // Pre-compute drive field factors once
    //     const double factor1_SU2 = field_drive_amp_SU2 * exp(-pow((curr_time - t_B_1_SU2) / (2 * field_drive_width_SU2), 2)) * cos(2 * M_PI * field_drive_freq_SU2 * (curr_time - t_B_1_SU2));
    //     const double factor2_SU2 = field_drive_amp_SU2 * exp(-pow((curr_time - t_B_2_SU2) / (2 * field_drive_width_SU2), 2)) * cos(2 * M_PI * field_drive_freq_SU2 * (curr_time - t_B_2_SU2));
    //     #pragma omp parallel for schedule(static)
    //     for(size_t i = 0; i < lattice_size_SU2; ++i){
    //         const size_t atom_index = i % N_ATOMS_SU2;  
    //         // Compute drive field inline to avoid function call overhead
    //         array<double, N_SU2> drive_field = field_drive_1_SU2[atom_index] * factor1_SU2 + field_drive_2_SU2[atom_index] * factor2_SU2;
    //         // Get local field
    //         array<double, N_SU2> local_field = get_local_field_SU2_lattice(i, current_spin_SU2, current_spin_SU3);
    //         // Compute effective field
    //         local_field = local_field - drive_field;
    //         // Compute cross product
    //         dS[i] = cross_prod_SU2(local_field, current_spin_SU2[i]);
    //     }
    //     return dS;
    // }
    // spin_config_SU3 landau_lifshitz_SU3(const spin_config_SU2 &current_spin_SU2, const spin_config_SU3 &current_spin_SU3, const double &curr_time){
    //     spin_config_SU3 dS;
    //     // Pre-compute drive field factors once
    //     const double factor1_SU3 = field_drive_amp_SU3 * exp(-pow((curr_time - t_B_1_SU3) / (2 * field_drive_width_SU3), 2)) * cos(2 * M_PI * field_drive_freq_SU3 * (curr_time - t_B_1_SU3));
    //     const double factor2_SU3 = field_drive_amp_SU3 * exp(-pow((curr_time - t_B_2_SU3) / (2 * field_drive_width_SU3), 2)) * cos(2 * M_PI * field_drive_freq_SU3 * (curr_time - t_B_2_SU3));
    //     #pragma omp parallel for schedule(static)
    //     for(size_t i = 0; i < lattice_size_SU3; ++i){
    //         const size_t atom_index = i % N_ATOMS_SU3;
    //         // Compute drive field inline to avoid function call overhead
    //         array<double, N_SU3> drive_field = field_drive_1_SU3[atom_index] * factor1_SU3 + field_drive_2_SU3[atom_index] * factor2_SU3;
    //         // Get local field
    //         array<double, N_SU3> local_field = get_local_field_SU3_lattice(i, current_spin_SU2, current_spin_SU3);    
    //         // Compute effective field
    //         local_field = local_field - drive_field;
    //         // Compute cross product
    //         dS[i] = cross_prod_SU3(local_field, current_spin_SU3[i]);
    //     }
    //     return dS;
    // }

    void RK4_step(double &step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> &curr_spins, const double &curr_time, const double tol){
        spin_config_SU2 k1_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time);
        spin_config_SU3 k1_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time);

        spin_config_SU2 lval_k2_SU2 = curr_spins.spins_SU2 + k1_SU2*(0.5*step_size);
        spin_config_SU3 lval_k2_SU3 = curr_spins.spins_SU3 + k1_SU3*(0.5*step_size);

        spin_config_SU2 k2_SU2 = landau_lifshitz_SU2(lval_k2_SU2, lval_k2_SU3, curr_time + step_size*0.5);
        spin_config_SU3 k2_SU3 = landau_lifshitz_SU3(lval_k2_SU2, lval_k2_SU3, curr_time + step_size*0.5);

        spin_config_SU2 lval_k3_SU2 = curr_spins.spins_SU2 + k2_SU2*(0.5*step_size);
        spin_config_SU3 lval_k3_SU3 = curr_spins.spins_SU3 + k2_SU3*(0.5*step_size);

        spin_config_SU2 k3_SU2 = landau_lifshitz_SU2(lval_k3_SU2, lval_k3_SU3, curr_time + step_size*0.5);
        spin_config_SU3 k3_SU3 = landau_lifshitz_SU3(lval_k3_SU2, lval_k3_SU3, curr_time + step_size*0.5);

        spin_config_SU2 lval_k4_SU2 = curr_spins.spins_SU2 + k3_SU2*step_size;
        spin_config_SU3 lval_k4_SU3 = curr_spins.spins_SU3 + k3_SU3*step_size;

        spin_config_SU2 k4_SU2 = landau_lifshitz_SU2(lval_k4_SU2, lval_k4_SU3, curr_time + step_size);
        spin_config_SU3 k4_SU3 = landau_lifshitz_SU3(lval_k4_SU2, lval_k4_SU3, curr_time + step_size);

        spin_config_SU2 new_spins_SU2 = curr_spins.spins_SU2 + (k1_SU2+ k2_SU2 * 2 + k3_SU2 * 2 + k4_SU2)*(step_size/6);
        spin_config_SU3 new_spins_SU3 = curr_spins.spins_SU3 + (k1_SU3+ k2_SU3 * 2 + k3_SU3 * 2 + k4_SU3)*(step_size/6);

        curr_spins.spins_SU2 = std::move(new_spins_SU2);
        curr_spins.spins_SU3 = std::move(new_spins_SU3);

    }

    void RK45_step(double &step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> &curr_spins, const double &curr_time, const double tol){
        spin_config_SU2 k1_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time)*step_size;
        spin_config_SU3 k1_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time)*step_size;

        spin_config_SU2 k2_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(1.0/4.0), curr_spins.spins_SU3 + k1_SU3*(1.0/4.0), curr_time + step_size*(1/4))*step_size;
        spin_config_SU3 k2_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2 + k1_SU2*(1.0/4.0), curr_spins.spins_SU3 + k1_SU3*(1.0/4.0), curr_time + step_size*(1/4))*step_size;

        spin_config_SU2 k3_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(3.0/32.0)+ k2_SU2*(9.0/32.0), curr_spins.spins_SU3 + k1_SU3*(3.0/32.0)+ k2_SU3*(9.0/32.0), curr_time + step_size*(3/8))*step_size;
        spin_config_SU3 k3_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2 + k1_SU2*(3.0/32.0)+ k2_SU2*(9.0/32.0), curr_spins.spins_SU3 + k1_SU3*(3.0/32.0)+ k2_SU3*(9.0/32.0), curr_time + step_size*(3/8))*step_size;

        spin_config_SU2 k4_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(1932.0/2197.0)+ k2_SU2*(-7200.0/2197.0) + k3_SU2*(7296.0/2197.0), curr_spins.spins_SU3 + k1_SU3*(1932.0/2197.0) + k2_SU3*(-7200.0/2197.0) + k3_SU3*(7296.0/2197.0), curr_time + step_size*(12/13))*step_size;
        spin_config_SU3 k4_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2 + k1_SU2*(1932.0/2197.0)+ k2_SU2*(-7200.0/2197.0) + k3_SU2*(7296.0/2197.0), curr_spins.spins_SU3 + k1_SU3*(1932.0/2197.0) + k2_SU3*(-7200.0/2197.0) + k3_SU3*(7296.0/2197.0), curr_time + step_size*(12/13))*step_size;

        spin_config_SU2 k5_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(439.0/216.0) + k2_SU2*(-8.0) + k3_SU2*(3680.0/513.0) + k4_SU2*(-845.0/4104.0), curr_spins.spins_SU3 + k1_SU3*(439.0/216.0) + k2_SU3*(-8.0) + k3_SU3*(3680.0/513.0) + k4_SU3*(-845.0/4104.0), curr_time + step_size)*step_size;
        spin_config_SU3 k5_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2 + k1_SU2*(439.0/216.0) + k2_SU2*(-8.0) + k3_SU2*(3680.0/513.0) + k4_SU2*(-845.0/4104.0), curr_spins.spins_SU3 + k1_SU3*(439.0/216.0) + k2_SU3*(-8.0) + k3_SU3*(3680.0/513.0) + k4_SU3*(-845.0/4104.0), curr_time + step_size)*step_size;

        spin_config_SU2 k6_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(-8.0/27.0)+ k2_SU2*(2.0) + k3_SU2*(-3544.0/2565.0)+ k4_SU2*(1859.0/4104.0)+ k5_SU2*(-11.0/40.0), curr_spins.spins_SU3 + k1_SU3*(-8.0/27.0) + k2_SU3*(2.0) + k3_SU3*(-3544.0/2565.0)+ k4_SU3*(1859.0/4104.0)+ k5_SU3*(-11.0/40.0), curr_time + step_size/2)*step_size;
        spin_config_SU3 k6_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2 + k1_SU2*(-8.0/27.0)+ k2_SU2*(2.0) + k3_SU2*(-3544.0/2565.0)+ k4_SU2*(1859.0/4104.0)+ k5_SU2*(-11.0/40.0), curr_spins.spins_SU3 + k1_SU3*(-8.0/27.0) + k2_SU3*(2.0) + k3_SU3*(-3544.0/2565.0)+ k4_SU3*(1859.0/4104.0)+ k5_SU3*(-11.0/40.0), curr_time + step_size/2)*step_size;

        spin_config_SU2 y_SU2 = curr_spins.spins_SU2 + k1_SU2*(25.0/216.0) + k3_SU2*(1408.0/2565.0) + k4_SU2*(2197.0/4101.0) - k5_SU2*(1.0/5.0);
        spin_config_SU3 y_SU3 = curr_spins.spins_SU3 + k1_SU3*(25.0/216.0) + k3_SU3*(1408.0/2565.0) + k4_SU3*(2197.0/4101.0) - k5_SU3*(1.0/5.0);

        spin_config_SU2 z_SU2 = curr_spins.spins_SU2 + k1_SU2*(16.0/135.0) + k3_SU2*(6656.0/12825.0) + k4_SU2*(28561.0/56430.0) - k5_SU2*(9.0/50.0) + k6_SU2*(2.0/55.0);
        spin_config_SU3 z_SU3 = curr_spins.spins_SU3 + k1_SU3*(16.0/135.0) + k3_SU3*(6656.0/12825.0) + k4_SU3*(28561.0/56430.0) - k5_SU3*(9.0/50.0) + k6_SU3*(2.0/55.0);

        double error_SU2 = norm_average_2D(y_SU2-z_SU2);
        double error_SU3 = norm_average_2D(y_SU3-z_SU3);

        double error = max(error_SU2, error_SU3);
        step_size *= 0.9*pow(tol/error, 0.2);

        if (error < tol){
            curr_spins.spins_SU2 = z_SU2;
            curr_spins.spins_SU3 = z_SU3;
        }
        else{
            RK45_step(step_size, curr_spins, curr_time, tol);
        }
    }
    
    void RK45_step_fixed(double &step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> &curr_spins, const double &curr_time, const double tol){
        spin_config_SU2 k1_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time)*step_size;
        spin_config_SU3 k1_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time)*step_size;

        spin_config_SU2 k2_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(1.0/4.0), curr_spins.spins_SU3 + k1_SU3*(1.0/4.0), curr_time + step_size*(1/4))*step_size;
        spin_config_SU3 k2_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2 + k1_SU2*(1.0/4.0), curr_spins.spins_SU3 + k1_SU3*(1.0/4.0), curr_time + step_size*(1/4))*step_size;

        spin_config_SU2 k3_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(3.0/32.0)+ k2_SU2*(9.0/32.0), curr_spins.spins_SU3 + k1_SU3*(3.0/32.0)+ k2_SU3*(9.0/32.0), curr_time + step_size*(3/8))*step_size;
        spin_config_SU3 k3_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2 + k1_SU2*(3.0/32.0)+ k2_SU2*(9.0/32.0), curr_spins.spins_SU3 + k1_SU3*(3.0/32.0)+ k2_SU3*(9.0/32.0), curr_time + step_size*(3/8))*step_size;

        spin_config_SU2 k4_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(1932.0/2197.0)+ k2_SU2*(-7200.0/2197.0) + k3_SU2*(7296.0/2197.0), curr_spins.spins_SU3 + k1_SU3*(1932.0/2197.0) + k2_SU3*(-7200.0/2197.0) + k3_SU3*(7296.0/2197.0), curr_time + step_size*(12/13))*step_size;
        spin_config_SU3 k4_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2 + k1_SU2*(1932.0/2197.0)+ k2_SU2*(-7200.0/2197.0) + k3_SU2*(7296.0/2197.0), curr_spins.spins_SU3 + k1_SU3*(1932.0/2197.0) + k2_SU3*(-7200.0/2197.0) + k3_SU3*(7296.0/2197.0), curr_time + step_size*(12/13))*step_size;

        spin_config_SU2 k5_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(439.0/216.0) + k2_SU2*(-8.0) + k3_SU2*(3680.0/513.0) + k4_SU2*(-845.0/4104.0), curr_spins.spins_SU3 + k1_SU3*(439.0/216.0) + k2_SU3*(-8.0) + k3_SU3*(3680.0/513.0) + k4_SU3*(-845.0/4104.0), curr_time + step_size)*step_size;
        spin_config_SU3 k5_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2 + k1_SU2*(439.0/216.0) + k2_SU2*(-8.0) + k3_SU2*(3680.0/513.0) + k4_SU2*(-845.0/4104.0), curr_spins.spins_SU3 + k1_SU3*(439.0/216.0) + k2_SU3*(-8.0) + k3_SU3*(3680.0/513.0) + k4_SU3*(-845.0/4104.0), curr_time + step_size)*step_size;

        spin_config_SU2 k6_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(-8.0/27.0)+ k2_SU2*(2.0) + k3_SU2*(-3544.0/2565.0)+ k4_SU2*(1859.0/4104.0)+ k5_SU2*(-11.0/40.0), curr_spins.spins_SU3 + k1_SU3*(-8.0/27.0) + k2_SU3*(2.0) + k3_SU3*(-3544.0/2565.0)+ k4_SU3*(1859.0/4104.0)+ k5_SU3*(-11.0/40.0), curr_time + step_size/2)*step_size;
        spin_config_SU3 k6_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2 + k1_SU2*(-8.0/27.0)+ k2_SU2*(2.0) + k3_SU2*(-3544.0/2565.0)+ k4_SU2*(1859.0/4104.0)+ k5_SU2*(-11.0/40.0), curr_spins.spins_SU3 + k1_SU3*(-8.0/27.0) + k2_SU3*(2.0) + k3_SU3*(-3544.0/2565.0)+ k4_SU3*(1859.0/4104.0)+ k5_SU3*(-11.0/40.0), curr_time + step_size/2)*step_size;

        spin_config_SU2 y_SU2 = curr_spins.spins_SU2 + k1_SU2*(25.0/216.0) + k3_SU2*(1408.0/2565.0) + k4_SU2*(2197.0/4101.0) - k5_SU2*(1.0/5.0);
        spin_config_SU3 y_SU3 = curr_spins.spins_SU3 + k1_SU3*(25.0/216.0) + k3_SU3*(1408.0/2565.0) + k4_SU3*(2197.0/4101.0) - k5_SU3*(1.0/5.0);

        spin_config_SU2 z_SU2 = curr_spins.spins_SU2 + k1_SU2*(16.0/135.0) + k3_SU2*(6656.0/12825.0) + k4_SU2*(28561.0/56430.0) - k5_SU2*(9.0/50.0) + k6_SU2*(2.0/55.0);
        spin_config_SU3 z_SU3 = curr_spins.spins_SU3 + k1_SU3*(16.0/135.0) + k3_SU3*(6656.0/12825.0) + k4_SU3*(28561.0/56430.0) - k5_SU3*(9.0/50.0) + k6_SU3*(2.0/55.0);

        double error_SU2 = norm_average_2D(y_SU2-z_SU2);
        double error_SU3 = norm_average_2D(y_SU3-z_SU3);

        double error = max(error_SU2, error_SU3);
        step_size *= 0.9*pow(tol/error, 0.2);

        curr_spins.spins_SU2 = std::move(z_SU2);
        curr_spins.spins_SU3 = std::move(z_SU3);
    }
    
    void euler_step(const double step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> &curr_spins, const double &curr_time, const double tol) {
        // First compute both derivatives
        spin_config_SU2 dS_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time);
        spin_config_SU3 dS_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time);
        
        // Then update the spins in parallel using OpenMP
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // Update SU2 spins
                #pragma omp parallel for simd collapse(2)
                for(size_t i = 0; i < lattice_size_SU2; ++i) {
                    for(size_t j = 0; j < N_SU2; ++j) {
                        curr_spins.spins_SU2[i][j] += dS_SU2[i][j] * step_size;
                    }
                }
            }
            
            #pragma omp section
            {
                // Update SU3 spins
                #pragma omp parallel for simd collapse(2)
                for(size_t i = 0; i < lattice_size_SU3; ++i) {
                    for(size_t j = 0; j < N_SU3; ++j) {
                        curr_spins.spins_SU3[i][j] += dS_SU3[i][j] * step_size;
                    }
                }
            }
        }
    }

    void SSPRK53_step(const double step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> &curr_spins, const double &curr_time, const double tol){
        // Pre-compute all step size multiplications
        constexpr double a30 = 0.355909775063327;
        constexpr double a32 = 0.644090224936674;
        constexpr double a40 = 0.367933791638137;
        constexpr double a43 = 0.632066208361863;
        constexpr double a52 = 0.237593836598569;
        constexpr double a54 = 0.762406163401431;
        constexpr double b10 = 0.377268915331368;
        constexpr double b21 = 0.377268915331368;
        constexpr double b32 = 0.242995220537396;
        constexpr double b43 = 0.238458932846290;
        constexpr double b54 = 0.287632146308408;
        constexpr double c1 = 0.377268915331368;
        constexpr double c2 = 0.754537830662736;
        constexpr double c3 = 0.728985661612188;
        constexpr double c4 = 0.699226135931670;

        // Pre-compute step size multiplications
        const double b10_h = b10 * step_size;
        const double b21_h = b21 * step_size;
        const double b32_h = b32 * step_size;
        const double b43_h = b43 * step_size;
        const double b54_h = b54 * step_size;
        const double c1_h = c1 * step_size;
        const double c2_h = c2 * step_size;
        const double c3_h = c3 * step_size;
        const double c4_h = c4 * step_size;

        // Allocate all working arrays upfront
        spin_config_SU2 tmp_SU2, k_SU2, u_SU2;
        spin_config_SU3 tmp_SU3, k_SU3, u_SU3;

        // Stage 1: Compute initial derivatives in parallel
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                k_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time);
                tmp_SU2 = curr_spins.spins_SU2 + k_SU2 * b10_h;
            }
            #pragma omp section
            {
                k_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time);
                tmp_SU3 = curr_spins.spins_SU3 + k_SU3 * b10_h;
            }
        }

        // Stage 2
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                k_SU2 = landau_lifshitz_SU2(tmp_SU2, tmp_SU3, curr_time + c1_h);
                u_SU2 = tmp_SU2 + k_SU2 * b21_h;
            }
            #pragma omp section
            {
                k_SU3 = landau_lifshitz_SU3(tmp_SU2, tmp_SU3, curr_time + c1_h);
                u_SU3 = tmp_SU3 + k_SU3 * b21_h;
            }
        }

        // Stage 3
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                k_SU2 = landau_lifshitz_SU2(u_SU2, u_SU3, curr_time + c2_h);
                tmp_SU2 = curr_spins.spins_SU2 * a30 + u_SU2 * a32 + k_SU2 * b32_h;
            }
            #pragma omp section
            {
                k_SU3 = landau_lifshitz_SU3(u_SU2, u_SU3, curr_time + c2_h);
                tmp_SU3 = curr_spins.spins_SU3 * a30 + u_SU3 * a32 + k_SU3 * b32_h;
            }
        }

        // Stage 4
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                k_SU2 = landau_lifshitz_SU2(tmp_SU2, tmp_SU3, curr_time + c3_h);
                // Fuse operations to reduce memory access
                tmp_SU2 = curr_spins.spins_SU2 * a40 + tmp_SU2 * a43 + k_SU2 * b43_h;
            }
            #pragma omp section
            {
                k_SU3 = landau_lifshitz_SU3(tmp_SU2, tmp_SU3, curr_time + c3_h);
                tmp_SU3 = curr_spins.spins_SU3 * a40 + tmp_SU3 * a43 + k_SU3 * b43_h;
            }
        }

        // Stage 5 and final result
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> result;
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                k_SU2 = landau_lifshitz_SU2(tmp_SU2, tmp_SU3, curr_time + c4_h);
                result.spins_SU2 = u_SU2 * a52 + tmp_SU2 * a54 + k_SU2 * b54_h;
            }
            #pragma omp section
            {
                k_SU3 = landau_lifshitz_SU3(tmp_SU2, tmp_SU3, curr_time + c4_h);
                result.spins_SU3 = u_SU3 * a52 + tmp_SU3 * a54 + k_SU3 * b54_h;
            }
        }

        // Update the current spins with the result using the set method
        curr_spins.spins_SU2 = std::move(result.spins_SU2);
        curr_spins.spins_SU3 = std::move(result.spins_SU3);
    }


    void test_step(const double step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> &curr_spins, const double &curr_time, const double tol){
        // Pre-compute all step size multiplications
        // Update the current spins with the result using the set method
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> result;

        for(size_t i = 0; i < lattice_size_SU2; ++i) {
            array<double, N_SU2> local_field = get_local_field_SU2_lattice(i, curr_spins.spins_SU2, curr_spins.spins_SU3);
            array<double, N_SU2> drive_field = drive_field_T_SU2(curr_time, i % N_ATOMS_SU2);
            result.spins_SU2[i] = local_field - drive_field;
        }
        for(size_t i = 0; i < lattice_size_SU3; ++i) {
            array<double, N_SU3> local_field = get_local_field_SU3_lattice(i, curr_spins.spins_SU2, curr_spins.spins_SU3);
            result.spins_SU3[i] = local_field;
        }

        curr_spins.spins_SU2 = std::move(result.spins_SU2);
        curr_spins.spins_SU3 = std::move(result.spins_SU3);
    }


    void write_to_file_magnetization_local_SU2(string filename, array<double, N_SU2> towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        for(size_t j = 0; j<N_SU2; ++j){
            myfile << towrite[j] << " ";
        }
        myfile << endl;
        myfile.close();
    }

    void write_to_file_magnetization_local_SU3(string filename, array<double, N_SU3> towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        for(size_t j = 0; j<N_SU3; ++j){
            myfile << towrite[j] << " ";
        }
        myfile << endl;
        myfile.close();
    }

    array<double,N_SU2> magnetization_local(mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> &spin_t) {
        array<double,N_SU2> mag = {{0}};
        
        #pragma omp parallel
        {
            array<double,N_SU2> local_mag = {{0}};
            
            #pragma omp for nowait
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                for (size_t j = 0; j < N_SU2; ++j) {
                    local_mag[j] += spin_t.spins_SU2[i][j];
                }
            }
            
            #pragma omp critical
            {
                for (size_t j = 0; j < N_SU2; ++j) {
                    mag[j] += local_mag[j];
                }
            }
        }
        
        const double inv_size = 1.0 / double(lattice_size_SU2);
        for (size_t j = 0; j < N_SU2; ++j) {
            mag[j] *= inv_size;
        }
        
        return mag;
    }

    array<double,N_SU2> magnetization_local_antiferromagnetic(mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> &spin_t) {
        array<double,N_SU2> mag = {{0}};
        
        #pragma omp parallel
        {
            array<double,N_SU2> local_mag = {{0}};
            
            #pragma omp for nowait
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                const int sign = 1 - 2 * (i & 1); // Efficient (-1)^i calculation
                for (size_t j = 0; j < N_SU2; ++j) {
                    local_mag[j] += spin_t.spins_SU2[i][j] * sign;
                }
            }
            
            #pragma omp critical
            {
                for (size_t j = 0; j < N_SU2; ++j) {
                    mag[j] += local_mag[j];
                }
            }
        }
        
        const double inv_size = 1.0 / double(lattice_size_SU2);
        for (size_t j = 0; j < N_SU2; ++j) {
            mag[j] *= inv_size;
        }
        
        return mag;
    }

    array<double,N_SU3> magnetization_local_SU3(mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> &spin_t) {
        array<double,N_SU3> mag = {{0}};
        
        #pragma omp parallel
        {
            array<double,N_SU3> local_mag = {{0}};
            
            #pragma omp for nowait
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                for (size_t j = 0; j < N_SU3; ++j) {
                    local_mag[j] += spin_t.spins_SU3[i][j];
                }
            }
            
            #pragma omp critical
            {
                for (size_t j = 0; j < N_SU3; ++j) {
                    mag[j] += local_mag[j];
                }
            }
        }
        
        const double inv_size = 1.0 / double(lattice_size_SU3);
        for (size_t j = 0; j < N_SU3; ++j) {
            mag[j] *= inv_size;
        }
        
        return mag;
    }

    array<double,N_SU3> magnetization_local_antiferromagnetic_SU3(mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> &spin_t) {
        array<double,N_SU3> mag = {{0}};
        
        #pragma omp parallel
        {
            array<double,N_SU3> local_mag = {{0}};
            
            #pragma omp for nowait
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                const int sign = 1 - 2 * (i & 1); // Efficient (-1)^i calculation
                for (size_t j = 0; j < N_SU3; ++j) {
                    local_mag[j] += spin_t.spins_SU3[i][j] * sign;
                }
            }
            
            #pragma omp critical
            {
                for (size_t j = 0; j < N_SU3; ++j) {
                    mag[j] += local_mag[j];
                }
            }
        }
        
        const double inv_size = 1.0 / double(lattice_size_SU3);
        for (size_t j = 0; j < N_SU3; ++j) {
            mag[j] *= inv_size;
        }
        
        return mag;
    }


    void molecular_dynamics(double T_start, double T_end, double step_size, string dir_name, bool verbose= false) {
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        
        // Pre-allocate time vector with estimated size
        const size_t estimated_steps = static_cast<size_t>((T_end - T_start) / step_size) + 1;
        vector<double> time;
        time.reserve(estimated_steps);
        
        // Write initial state once
        write_to_file_pos(dir_name + "/pos");
        write_to_file_spin(dir_name + "/spin");
        
        // Open file handles once instead of reopening for each write
        ofstream spin_file_SU2(dir_name + "/spin_t_CPU_SU2.txt", ios::out | ios::trunc);
        ofstream spin_file_SU3(dir_name + "/spin_t_CPU_SU3.txt", ios::out | ios::trunc);
        
        if (!spin_file_SU2.is_open() || !spin_file_SU3.is_open()) {
            cerr << "Error opening output files" << endl;
            return;
        }
        
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> spin_t(spins);
        
        double tol = 1e-6;
        double currT = T_start;
        size_t count = 0;
        
        // Write initial state
        time.push_back(currT);
        
        // Pre-allocate buffers for file writing
        constexpr size_t BUFFER_SIZE = 8192;
        spin_file_SU2.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
        spin_file_SU3.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
        
        // Main time evolution loop with buffered output
        const int output_frequency = 1; // Write every N steps instead of every step
        
        while(currT < T_end){             
            SSPRK53_step(step_size, spin_t, currT, tol);
            
            // Buffer writes - only write every output_frequency steps
            if (count % output_frequency == 0) {
                // Write SU2 spins
                for(size_t i = 0; i < lattice_size_SU2; ++i){
                    for(size_t j = 0; j < 3; ++j){
                        spin_file_SU2 << spin_t.spins_SU2[i][j] << " ";
                    }
                    spin_file_SU2 << "\n";
                }
                
                // Write SU3 spins
                for(size_t i = 0; i < lattice_size_SU3; ++i){
                    for(size_t j = 0; j < 8; ++j){
                        spin_file_SU3 << spin_t.spins_SU3[i][j] << " ";
                    }
                    spin_file_SU3 << "\n";
                }
            }
            
            currT += step_size;
            time.push_back(currT);
            count++;
            
            // Progress reporting with less frequency
            if (count % 1000 == 0) {
                cout << "Time: " << currT << " (" << (100.0 * (currT - T_start) / (T_end - T_start)) << "%)" << endl;
            }
        }
        
        // Close file handles
        spin_file_SU2.close();
        spin_file_SU3.close();
        
        // Write time steps
        ofstream time_sections(dir_name + "/Time_steps.txt");
        for(const auto& t : time){
            time_sections << t << "\n";
        }
        time_sections.close();
    }

    void M_B_t(array<array<double,N_SU2>, N_ATOMS_SU2> &field_in, double t_B, double pulse_amp, double pulse_width, double pulse_freq, double T_start, double T_end, double step_size, string dir_name){
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> spin_t(spins);
        
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        
        const double tol = 1e-6;
        const size_t estimated_steps = static_cast<size_t>((T_end - T_start) / step_size) + 1;
        
        // Pre-allocate vectors
        vector<double> time;
        time.reserve(estimated_steps);
        
        // Open file handles once
        ofstream mag_file_f(dir_name + "/M_t_f.txt", ios::out | ios::trunc);
        ofstream mag_file(dir_name + "/M_t.txt", ios::out | ios::trunc);
        
        if (!mag_file_f.is_open() || !mag_file.is_open()) {
            cerr << "Error opening magnetization files" << endl;
            return;
        }
        
        // Set up buffering for better I/O performance
        constexpr size_t BUFFER_SIZE = 8192;
        mag_file_f.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
        mag_file.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
        
        double currT = T_start;
        time.push_back(currT);
        
        // Write initial magnetization
        auto mag_f = magnetization_local(spin_t);
        auto mag = magnetization_local_antiferromagnetic(spin_t);
        
        for(size_t j = 0; j < N_SU2; ++j){
            mag_file_f << mag_f[j] << " ";
            mag_file << mag[j] << " ";
        }
        mag_file_f << "\n";
        mag_file << "\n";
        
        set_pulse_SU2(field_in, t_B, {{0}}, 0, pulse_amp, pulse_width, pulse_freq);
        
        // Main time evolution loop
        while(currT < T_end){
            SSPRK53_step(step_size, spin_t, currT, tol);
            
            // Calculate magnetizations in parallel
            #pragma omp parallel sections
            {
                #pragma omp section
                mag_f = magnetization_local(spin_t);
                
                #pragma omp section
                mag = magnetization_local_antiferromagnetic(spin_t);
            }
            
            // Write magnetizations
            for(size_t j = 0; j < N_SU2; ++j){
                mag_file_f << mag_f[j] << " ";
                mag_file << mag[j] << " ";
            }
            mag_file_f << "\n";
            mag_file << "\n";
            
            currT += step_size;
            time.push_back(currT);
        }
        
        reset_pulse();
        
        // Close magnetization files
        mag_file_f.close();
        mag_file.close();
        
        // Write time steps
        ofstream time_sections(dir_name + "/Time_steps.txt", ios::out | ios::trunc);
        for(const auto& t : time){
            time_sections << t << "\n";
        }
        time_sections.close();      
    }

    void M_BA_BB_t(array<array<double,N_SU2>, N_ATOMS_SU2> &field_in_1, double t_B_1, array<array<double,N_SU2>, N_ATOMS_SU2> &field_in_2, double t_B_2, double pulse_amp, double pulse_width, double pulse_freq, double T_start, double T_end, double step_size, string dir_name){
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim, N_SU3, N_ATOMS_SU3*dim1*dim2*dim> spin_t(spins);
        
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        
        const double tol = 1e-6;
        const size_t estimated_steps = static_cast<size_t>((T_end - T_start) / step_size) + 1;
        
        // Pre-allocate vectors
        vector<double> time;
        time.reserve(estimated_steps);
        
        // Open file handles once
        ofstream mag_file_f(dir_name + "/M_t_f.txt", ios::out | ios::trunc);
        ofstream mag_file(dir_name + "/M_t.txt", ios::out | ios::trunc);
        
        if (!mag_file_f.is_open() || !mag_file.is_open()) {
            cerr << "Error opening magnetization files" << endl;
            return;
        }
        
        // Set up buffering for better I/O performance
        constexpr size_t BUFFER_SIZE = 8192;
        mag_file_f.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
        mag_file.rdbuf()->pubsetbuf(nullptr, BUFFER_SIZE);
        
        double currT = T_start;
        time.push_back(currT);
        
        // Write initial magnetization
        auto mag_f = magnetization_local(spin_t);
        auto mag = magnetization_local_antiferromagnetic(spin_t);
        
        for(size_t j = 0; j < N_SU2; ++j){
            mag_file_f << mag_f[j] << " ";
            mag_file << mag[j] << " ";
        }
        mag_file_f << "\n";
        mag_file << "\n";
        
        set_pulse_SU2(field_in_1, t_B_1, field_in_2, t_B_2, pulse_amp, pulse_width, pulse_freq);
        
        // Main time evolution loop
        while(currT < T_end){
            SSPRK53_step(step_size, spin_t, currT, tol);
            
            // Calculate magnetizations in parallel
            #pragma omp parallel sections
            {
                #pragma omp section
                mag_f = magnetization_local(spin_t);
                
                #pragma omp section
                mag = magnetization_local_antiferromagnetic(spin_t);
            }
            
            // Write magnetizations
            for(size_t j = 0; j < N_SU2; ++j){
                mag_file_f << mag_f[j] << " ";
                mag_file << mag[j] << " ";
            }
            mag_file_f << "\n";
            mag_file << "\n";
            
            currT += step_size;
            time.push_back(currT);
        }
        
        reset_pulse();
        
        // Close magnetization files
        mag_file_f.close();
        mag_file.close();
        
        // Write time steps
        ofstream time_sections(dir_name + "/Time_steps.txt", ios::out | ios::trunc);
        for(const auto& t : time){
            time_sections << t << "\n";
        }
        time_sections.close();
    }
    
};
#endif // MIXED_LATTICE_H