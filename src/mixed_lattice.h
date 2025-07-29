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
#include "convergence_monitor.h"
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

    double field_drive_freq_SU2_SU3;
    double field_drive_amp_SU2_SU3;
    double field_drive_width_SU2_SU3;
    double t_B_1_SU2_SU3;
    double t_B_2_SU2_SU3;
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
    array<vector<array<double, N_SU2 * N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim> mixed_bilinear_interaction_SU3;
    array<vector<size_t>, N_ATOMS_SU2*dim1*dim2*dim> mixed_bilinear_partners_SU2;
    array<vector<size_t>, N_ATOMS_SU3*dim1*dim2*dim> mixed_bilinear_partners_SU3;

    array<array<double,N_SU2>, N_ATOMS_SU2> field_drive_1_SU2_SU3;
    array<array<double,N_SU2>, N_ATOMS_SU2> field_drive_2_SU2_SU3;

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
    size_t num_bi_SU2_SU3;
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
        field_drive_freq_SU2_SU3 = 0.0;
        field_drive_amp_SU2_SU3 = 0.0;
        field_drive_width_SU2_SU3 = 1.0;
        t_B_1_SU2_SU3 = 0.0;
        t_B_2_SU2_SU3 = 0.0;

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

        cout << "Mixed lattice initialized with SU2 size: " << lattice_size_SU2 
             << ", SU3 size: " << lattice_size_SU3 << endl;

        // Process bilinear mixed interactions in parallel across outer loop dimensions
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim; ++k) {
                    for (size_t l = 0; l < N_ATOMS_SU3; ++l) {
                        const size_t current_site_index = flatten_index(i, j, k, l, N_ATOMS_SU3);

                        // Process mixed bilinear interactions
                        auto bilinear_matched = atoms->bilinear_SU2_SU3.equal_range(l);
                        for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m) {
                            const mixed_bilinear<N_SU3, N_SU2>& J = m->second;
                            const size_t partner = flatten_index_periodic_boundary(
                                i + J.offset[0], j + J.offset[1], k + J.offset[2], J.partner, N_ATOMS_SU2);

                            mixed_bilinear_interaction_SU3[current_site_index].push_back(J.bilinear_interaction);
                            mixed_bilinear_partners_SU3[current_site_index].push_back(partner);
                            
                            // Precompute transposed tensors once
                            const auto transposed = transpose2D<N_SU3, N_SU2>(J.bilinear_interaction);

                            #pragma omp critical(bilinear_SU2_update)
                            {
                                mixed_bilinear_interaction_SU2[partner].push_back(transposed);
                                mixed_bilinear_partners_SU2[partner].push_back(current_site_index);
                            }

                        }
                    }
                }
            }
        }
        

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
        num_bi_SU2_SU3 = mixed_bilinear_partners_SU2[0].size();
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

    double site_energy_SU2_diff(const array<double, N_SU2> &new_spin, const array<double, N_SU2> &old_spin, const size_t site_index) const {
        // Calculate field energy directly - no need for temporary spin_diff array
        double field_energy = 0.0;
        #pragma omp simd reduction(+:field_energy)
        for (size_t j = 0; j < N_SU2; ++j) {
            field_energy -= (new_spin[j] - old_spin[j]) * field_SU2[site_index][j];
        }
        
        // Compute onsite energy more efficiently by exploiting the quadratic form
        // For quadratic forms: new^T·A·new - old^T·A·old = (new+old)^T·A·(new-old)
        double onsite_energy = 0.0;
        const auto& interaction = onsite_interaction_SU2[site_index];
        
        #pragma omp simd collapse(2) reduction(+:onsite_energy)
        for (size_t i = 0; i < N_SU2; ++i) {
            for (size_t j = 0; j < N_SU2; ++j) {
                onsite_energy += (new_spin[i] + old_spin[i]) * interaction[i*N_SU2 + j] * (new_spin[j] - old_spin[j]);
            }
        }
        
        // Pre-fetch critical data for bilinear interactions
        double bilinear_energy = 0.0;
        double bilinear_energy_mixed = 0.0;
        #pragma omp simd reduction(+:bilinear_energy)
        for (size_t i = 0; i < num_bi_SU2; ++i) {
            const size_t partner_idx = bilinear_partners_SU2[site_index][i];
            // Prefetch next iteration's data
            if (i < num_bi_SU2 - 1) {
                __builtin_prefetch(&bilinear_partners_SU2[site_index][i+1], 0, 3);
                __builtin_prefetch(&bilinear_interaction_SU2[site_index][i+1], 0, 3);
                __builtin_prefetch(&spins.spins_SU2[bilinear_partners_SU2[site_index][i+1]], 0, 3);
            }
            
            const auto& partner_spin = spins.spins_SU2[partner_idx]; 
            const auto& interaction = bilinear_interaction_SU2[site_index][i];
            
            // Inline the contract operation for bilinear interactions
            for (size_t a = 0; a < N_SU2; ++a) {
                const double diff_a = new_spin[a] - old_spin[a];
                if (std::abs(diff_a) < 1e-10) continue; 
                
                for (size_t b = 0; b < N_SU2; ++b) {
                    bilinear_energy += diff_a * interaction[a*N_SU2 + b] * partner_spin[b];
                }
            }
        }
        // Optimized mixed bilinear energy calculation
        #pragma omp simd reduction(+:bilinear_energy_mixed)
        for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
            const size_t partner_idx = mixed_bilinear_partners_SU2[site_index][i];
            // Prefetch next iteration's data to avoid cache misses
            if (i < num_bi_SU2_SU3 - 1) {
                __builtin_prefetch(&mixed_bilinear_partners_SU2[site_index][i+1], 0, 3);
                __builtin_prefetch(&mixed_bilinear_interaction_SU2[site_index][i+1], 0, 3);
                __builtin_prefetch(&spins.spins_SU3[mixed_bilinear_partners_SU2[site_index][i+1]], 0, 3);
            }
            
            const auto& partner_spin = spins.spins_SU3[partner_idx]; 
            const auto& interaction = mixed_bilinear_interaction_SU2[site_index][i];
            
            // Process all non-zero differences in one pass to reduce branch mispredictions
            for (size_t a = 0; a < N_SU2; ++a) {
                const double diff_a = new_spin[a] - old_spin[a];
                if (std::abs(diff_a) < 1e-10) continue; // Use absolute comparison for stability
                
                // Compute dot product directly with better memory access pattern
                // This helps the compiler better vectorize the inner loop
                double dot_result = 0.0;
                #pragma omp simd reduction(+:dot_result)
                for (size_t b = 0; b < N_SU3; ++b) {
                    dot_result += interaction[a*N_SU3 + b] * partner_spin[b];
                }
                
                bilinear_energy_mixed += diff_a * dot_result;
            }
        }

        // Specialized trilinear computation to reduce redundant calculations
        double trilinear_energy_SU2 = 0.0;
        double trilinear_energy_mixed = 0.0;
        
        // Process trilinear interactions in blocks for better cache utilization
        constexpr size_t BLOCK_SIZE = 4; // Tune this parameter based on cache size
        
        for (size_t i_block = 0; i_block < num_tri_SU2; i_block += BLOCK_SIZE) {
            const size_t end_idx = std::min(i_block + BLOCK_SIZE, num_tri_SU2);
            
            for (size_t i = i_block; i < end_idx; ++i) {
                const size_t partner1_idx = trilinear_partners_SU2[site_index][i][0];
                const size_t partner2_idx = trilinear_partners_SU2[site_index][i][1];
                const auto& partner1_spin = spins.spins_SU2[partner1_idx];
                const auto& partner2_spin = spins.spins_SU2[partner2_idx];
                const auto& interaction = trilinear_interaction_SU2[site_index][i];
                
                // Process only non-zero differences to avoid unnecessary computations
                for (size_t a = 0; a < N_SU2; ++a) {
                    const double diff_a = new_spin[a] - old_spin[a];
                    if (std::abs(diff_a) < 1e-10) continue;
                    
                    for (size_t b = 0; b < N_SU2; ++b) {
                        for (size_t c = 0; c < N_SU2; ++c) {
                            trilinear_energy_SU2 += diff_a * interaction[(a*N_SU2 + b)*N_SU2 + c] * 
                                                   partner1_spin[b] * partner2_spin[c];
                        }
                    }
                }
            }
        }
        
        // Mixed trilinear interactions - optimized similarly
        for (size_t i_block = 0; i_block < num_tri_SU2_SU3; i_block += BLOCK_SIZE) {
            const size_t end_idx = std::min(i_block + BLOCK_SIZE, num_tri_SU2_SU3);
            
            for (size_t i = i_block; i < end_idx; ++i) {
                const size_t partner1_idx = mixed_trilinear_partners_SU2[site_index][i][0];
                const size_t partner2_idx = mixed_trilinear_partners_SU2[site_index][i][1];
                const auto& partner1_spin = spins.spins_SU2[partner1_idx];
                const auto& partner2_spin = spins.spins_SU3[partner2_idx];
                const auto& interaction = mixed_trilinear_interaction_SU2[site_index][i];
                
                // Process only non-zero differences to avoid unnecessary calculations
                for (size_t a = 0; a < N_SU2; ++a) {
                    const double diff_a = new_spin[a] - old_spin[a];
                    if (std::abs(diff_a) < 1e-10) continue;
                    
                    for (size_t b = 0; b < N_SU2; ++b) {
                        for (size_t c = 0; c < N_SU3; ++c) {
                            trilinear_energy_mixed += diff_a * interaction[(a*N_SU2 + b)*N_SU3 + c] * 
                                                    partner1_spin[b] * partner2_spin[c];
                        }
                    }
                }
            }
        }
        
        return field_energy + onsite_energy + bilinear_energy/2 + bilinear_energy_mixed/2 + trilinear_energy_SU2/3 + trilinear_energy_mixed/3;
    }

    double site_energy_SU3_diff(const array<double, N_SU3> &new_spin, const array<double, N_SU3> &old_spin, const size_t site_index) const {
        // Direct field energy calculation
        double field_energy = 0.0;
        #pragma omp simd reduction(+:field_energy)
        for (size_t j = 0; j < N_SU3; ++j) {
            field_energy -= (new_spin[j] - old_spin[j]) * field_SU3[site_index][j];
        }
        
        // Optimize onsite energy using the quadratic form property
        double onsite_energy = 0.0;
        const auto& interaction = onsite_interaction_SU3[site_index];
        
        #pragma omp simd collapse(2) reduction(+:onsite_energy)
        for (size_t i = 0; i < N_SU3; ++i) {
            for (size_t j = 0; j < N_SU3; ++j) {
                onsite_energy += (new_spin[i] + old_spin[i]) * interaction[i*N_SU3 + j] * (new_spin[j] - old_spin[j]);
            }
        }
        
        // Bilinear interactions with prefetching and early skipping
        double bilinear_energy = 0.0;
        double bilinear_energy_mixed = 0.0;
        for (size_t i = 0; i < num_bi_SU3; ++i) {
            const size_t partner_idx = bilinear_partners_SU3[site_index][i];
            // Prefetch next iteration's data
            if (i < num_bi_SU3 - 1) {
                __builtin_prefetch(&bilinear_partners_SU3[site_index][i+1], 0, 3);
                __builtin_prefetch(&bilinear_interaction_SU3[site_index][i+1], 0, 3);
                __builtin_prefetch(&spins.spins_SU3[bilinear_partners_SU3[site_index][i+1]], 0, 3);
            }
            
            const auto& partner_spin = spins.spins_SU3[partner_idx];
            const auto& interaction = bilinear_interaction_SU3[site_index][i];
            
            // Inline the contract operation with early skipping of zero components
            for (size_t a = 0; a < N_SU3; ++a) {
                const double diff_a = new_spin[a] - old_spin[a];
                if (std::abs(diff_a) < 1e-10) continue;
                
                for (size_t b = 0; b < N_SU3; ++b) {
                    bilinear_energy += diff_a * interaction[a*N_SU3 + b] * partner_spin[b];
                }
            }
        }

                // Optimized mixed bilinear energy calculation
        #pragma omp simd reduction(+:bilinear_energy_mixed)
        for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
            const size_t partner_idx = mixed_bilinear_partners_SU3[site_index][i];
            // Prefetch next iteration's data to avoid cache misses
            if (i < num_bi_SU2_SU3 - 1) {
                __builtin_prefetch(&mixed_bilinear_partners_SU3[site_index][i+1], 0, 3);
                __builtin_prefetch(&mixed_bilinear_interaction_SU3[site_index][i+1], 0, 3);
                __builtin_prefetch(&spins.spins_SU2[mixed_bilinear_partners_SU3[site_index][i+1]], 0, 3);
            }
            
            const auto& partner_spin = spins.spins_SU2[partner_idx]; 
            const auto& interaction = mixed_bilinear_interaction_SU3[site_index][i];
            
            // Process all non-zero differences in one pass to reduce branch mispredictions
            for (size_t a = 0; a < N_SU3; ++a) {
                const double diff_a = new_spin[a] - old_spin[a];
                if (std::abs(diff_a) < 1e-10) continue; // Use absolute comparison for stability
                
                // Compute dot product directly with better memory access pattern
                // This helps the compiler better vectorize the inner loop
                double dot_result = 0.0;
                #pragma omp simd reduction(+:dot_result)
                for (size_t b = 0; b < N_SU2; ++b) {
                    dot_result += interaction[a*N_SU2 + b] * partner_spin[b];
                }
                
                bilinear_energy_mixed += diff_a * dot_result;
            }
        }

        
        // Block processing for trilinear interactions to improve cache utilization
        double trilinear_energy_SU3 = 0.0;
        double trilinear_energy_mixed = 0.0;
        
        constexpr size_t BLOCK_SIZE = 4; // Tune based on cache size

        // Optimized SU3 trilinear interactions with blocking and early skipping
        for (size_t i_block = 0; i_block < num_tri_SU3; i_block += BLOCK_SIZE) {
            const size_t end_idx = std::min(i_block + BLOCK_SIZE, num_tri_SU3);
            
            for (size_t i = i_block; i < end_idx; ++i) {
                const size_t partner1_idx = trilinear_partners_SU3[site_index][i][0];
                const size_t partner2_idx = trilinear_partners_SU3[site_index][i][1];
                const auto& partner1_spin = spins.spins_SU3[partner1_idx];
                const auto& partner2_spin = spins.spins_SU3[partner2_idx];
                const auto& interaction = trilinear_interaction_SU3[site_index][i];
                
                // Skip computation for zero differences
                for (size_t a = 0; a < N_SU3; ++a) {
                    const double diff_a = new_spin[a] - old_spin[a];
                    if (std::abs(diff_a) < 1e-10) continue;
                    
                    for (size_t b = 0; b < N_SU3; ++b) {
                        for (size_t c = 0; c < N_SU3; ++c) {
                            trilinear_energy_SU3 += diff_a * interaction[(a*N_SU3 + b)*N_SU3 + c] * 
                                                   partner1_spin[b] * partner2_spin[c];
                        }
                    }
                }
            }
        }
        
        // Mixed trilinear interactions with similar optimizations
        for (size_t i_block = 0; i_block < num_tri_SU2_SU3; i_block += BLOCK_SIZE) {
            const size_t end_idx = std::min(i_block + BLOCK_SIZE, num_tri_SU2_SU3);
            
            for (size_t i = i_block; i < end_idx; ++i) {
                const size_t partner1_idx = mixed_trilinear_partners_SU3[site_index][i][0];
                const size_t partner2_idx = mixed_trilinear_partners_SU3[site_index][i][1];
                const auto& partner1_spin = spins.spins_SU2[partner1_idx]; 
                const auto& partner2_spin = spins.spins_SU2[partner2_idx];
                const auto& interaction = mixed_trilinear_interaction_SU3[site_index][i];
                
                for (size_t a = 0; a < N_SU3; ++a) {
                    const double diff_a = new_spin[a] - old_spin[a];
                    if (std::abs(diff_a) < 1e-10) continue;
                    
                    for (size_t b = 0; b < N_SU2; ++b) {
                        for (size_t c = 0; c < N_SU2; ++c) {
                            trilinear_energy_mixed += diff_a * interaction[(a*N_SU2 + b)*N_SU2 + c] * 
                                                    partner1_spin[b] * partner2_spin[c];
                        }
                    }
                }
            }
        }
        
        return field_energy + onsite_energy + bilinear_energy/2 + bilinear_energy_mixed/2 + trilinear_energy_SU3/3 + trilinear_energy_mixed/3;
    }

    double total_energy(mixed_lattice_spin<N_SU2, dim1*dim2*dim*N_ATOMS_SU2, N_SU3, dim1*dim2*dim*N_ATOMS_SU3> &curr_spins) {
        // Pre-calculate scaling factors for better numerical stability
        constexpr double onsite_factor = 0.5;
        constexpr double bilinear_factor = 0.5;
        constexpr double trilinear_factor = 1.0/3.0;
        
        // Use OpenMP reductions to safely accumulate energy across threads
        double field_energy_SU2 = 0.0;
        double onsite_energy_SU2 = 0.0;
        double bilinear_energy_SU2 = 0.0;
        double trilinear_energy_SU2 = 0.0;
        double mixed_bilinear_energy_SU2 = 0.0;
        double mixed_trilinear_energy_SU2 = 0.0;
        
        double field_energy_SU3 = 0.0;
        double onsite_energy_SU3 = 0.0;
        double bilinear_energy_SU3 = 0.0;
        double trilinear_energy_SU3 = 0.0;
        double mixed_bilinear_energy_SU3 = 0.0;
        double mixed_trilinear_energy_SU3 = 0.0;
        
        // Process SU2 and SU3 lattices in parallel sections
        #pragma omp parallel sections reduction(+:field_energy_SU2,onsite_energy_SU2,bilinear_energy_SU2,trilinear_energy_SU2,mixed_trilinear_energy_SU2,field_energy_SU3,onsite_energy_SU3,bilinear_energy_SU3,trilinear_energy_SU3,mixed_trilinear_energy_SU3)
        {
            #pragma omp section
            {
                // Process SU2 lattice with better vectorization opportunities
                #pragma omp parallel for reduction(+:field_energy_SU2,onsite_energy_SU2,bilinear_energy_SU2,trilinear_energy_SU2,mixed_trilinear_energy_SU2) schedule(static)
                for(size_t site_index = 0; site_index < lattice_size_SU2; ++site_index) {
                    // Fetch current spin once to avoid repeated memory access
                    const auto& current_spin = curr_spins.spins_SU2[site_index];
                    
                    // Field energy computation
                    field_energy_SU2 -= dot(current_spin, field_SU2[site_index]);
                    
                    // Onsite energy computation
                    onsite_energy_SU2 += contract(current_spin, onsite_interaction_SU2[site_index], current_spin);
                    
                    // Bilinear energy - use SIMD for auto-vectorization of hot loop
                    #pragma omp simd reduction(+:bilinear_energy_SU2)
                    for (size_t i = 0; i < num_bi_SU2; ++i) {
                        bilinear_energy_SU2 += contract(
                            current_spin,
                            bilinear_interaction_SU2[site_index][i],
                            curr_spins.spins_SU2[bilinear_partners_SU2[site_index][i]]
                        );
                    }

                    // Mixed Bilinear energy - SU2-SU3 interactions
                    #pragma omp simd reduction(+:mixed_bilinear_energy_SU2)
                    for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
                        mixed_bilinear_energy_SU2 += contract_mixed_bilinear(
                            current_spin,
                            mixed_bilinear_interaction_SU2[site_index][i],
                            curr_spins.spins_SU3[mixed_bilinear_partners_SU2[site_index][i]]
                        );
                    }
                    
                    // Trilinear energy - SU2 components
                    #pragma omp simd reduction(+:trilinear_energy_SU2)
                    for (size_t i = 0; i < num_tri_SU2; ++i) {
                        trilinear_energy_SU2 += contract_trilinear(
                            trilinear_interaction_SU2[site_index][i],
                            current_spin,
                            curr_spins.spins_SU2[trilinear_partners_SU2[site_index][i][0]],
                            curr_spins.spins_SU2[trilinear_partners_SU2[site_index][i][1]]
                        );
                    }
                    
                    // Mixed trilinear energy - SU2-SU3 interactions
                    #pragma omp simd reduction(+:mixed_trilinear_energy_SU2)
                    for (size_t i = 0; i < num_tri_SU2_SU3; ++i) {
                        mixed_trilinear_energy_SU2 += contract_trilinear(
                            mixed_trilinear_interaction_SU2[site_index][i],
                            current_spin,
                            curr_spins.spins_SU2[mixed_trilinear_partners_SU2[site_index][i][0]],
                            curr_spins.spins_SU3[mixed_trilinear_partners_SU2[site_index][i][1]]
                        );
                    }
                }
            }
            
            #pragma omp section
            {
                // Process SU3 lattice
                #pragma omp parallel for reduction(+:field_energy_SU3,onsite_energy_SU3,bilinear_energy_SU3,trilinear_energy_SU3,mixed_trilinear_energy_SU3) schedule(static)
                for(size_t site_index = 0; site_index < lattice_size_SU3; ++site_index) {
                    // Fetch current spin once to avoid repeated memory access
                    const auto& current_spin = curr_spins.spins_SU3[site_index];
                    
                    // Field energy computation
                    field_energy_SU3 -= dot(current_spin, field_SU3[site_index]);
                    
                    // Onsite energy computation
                    onsite_energy_SU3 += contract(current_spin, onsite_interaction_SU3[site_index], current_spin);
                    
                    // Bilinear energy - use SIMD for auto-vectorization of hot loop
                    #pragma omp simd reduction(+:bilinear_energy_SU3)
                    for (size_t i = 0; i < num_bi_SU3; ++i) {
                        bilinear_energy_SU3 += contract(
                            current_spin,
                            bilinear_interaction_SU3[site_index][i],
                            curr_spins.spins_SU3[bilinear_partners_SU3[site_index][i]]
                        );
                    }

                    // Mixed Bilinear energy - SU3-SU2 interactions
                    #pragma omp simd reduction(+:mixed_bilinear_energy_SU3)
                    for (size_t i = 0; i < num_bi_SU2_SU3; ++i) {
                        mixed_bilinear_energy_SU3 += contract_mixed_bilinear(  
                            current_spin,
                            mixed_bilinear_interaction_SU3[site_index][i],
                            curr_spins.spins_SU2[mixed_bilinear_partners_SU3[site_index][i]]
                        );
                    }
                    
                    // Trilinear energy - SU3 components
                    #pragma omp simd reduction(+:trilinear_energy_SU3)
                    for (size_t i = 0; i < num_tri_SU3; ++i) {
                        trilinear_energy_SU3 += contract_trilinear(
                            trilinear_interaction_SU3[site_index][i],
                            current_spin,
                            curr_spins.spins_SU3[trilinear_partners_SU3[site_index][i][0]],
                            curr_spins.spins_SU3[trilinear_partners_SU3[site_index][i][1]]
                        );
                    }
                    
                    // Mixed trilinear energy - SU3-SU2 interactions
                    #pragma omp simd reduction(+:mixed_trilinear_energy_SU3)
                    for (size_t i = 0; i < num_tri_SU2_SU3; ++i) {
                        mixed_trilinear_energy_SU3 += contract_trilinear(
                            mixed_trilinear_interaction_SU3[site_index][i],
                            current_spin,
                            curr_spins.spins_SU2[mixed_trilinear_partners_SU3[site_index][i][0]],
                            curr_spins.spins_SU2[mixed_trilinear_partners_SU3[site_index][i][1]]
                        );
                    }
                }
            }
        }
        
        // Combine all energy components with appropriate scaling factors
        const double field_energy = field_energy_SU2 + field_energy_SU3;
        const double onsite_energy = (onsite_energy_SU2 + onsite_energy_SU3) * onsite_factor;
        const double bilinear_energy = (bilinear_energy_SU2 + bilinear_energy_SU3 
                                        + mixed_bilinear_energy_SU2 + mixed_bilinear_energy_SU3) * bilinear_factor;
        const double trilinear_energy = (trilinear_energy_SU2 + trilinear_energy_SU3 + 
                                        mixed_trilinear_energy_SU2 + mixed_trilinear_energy_SU3) * trilinear_factor;
        
        // Return the total energy
        return field_energy + onsite_energy + bilinear_energy + trilinear_energy;
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

        const array<double, N_SU3> drive_field_basis_x_SU3 = {{0,0,0,0,2.3915/5,0,0.9128/5,0}};
        const array<double, N_SU3> drive_field_basis_y_SU3 = {{0,0,0,0,2.7866/5,0,-0.4655/5,0}};
        const array<double, N_SU3> drive_field_basis_z_SU3 = {{0,0,0,0,0,0,0,0}};

        for (size_t atom_idx = 0; atom_idx < N_ATOMS_SU3; ++atom_idx) {
            // Convert SU2 drive fields to SU3 basis
            field_drive_1_SU3[atom_idx] = drive_field_basis_x_SU3 * field_drive_1_SU2[0][0] +
                                          drive_field_basis_y_SU3 * field_drive_1_SU2[0][1] +
                                          drive_field_basis_z_SU3 * field_drive_1_SU2[0][2];
            field_drive_2_SU3[atom_idx] = drive_field_basis_x_SU3 * field_drive_2_SU2[0][0] +
                                          drive_field_basis_y_SU3 * field_drive_2_SU2[0][1] +
                                          drive_field_basis_z_SU3 * field_drive_2_SU2[0][2];
        }
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

    void set_pulse_SU2_SU3(const array<array<double,N_SU2>, N_ATOMS_SU2> &field_in_SU2_SU3, double t_B, const array<array<double,N_SU2>, N_ATOMS_SU2> &field_in_2_SU2_SU3, double t_B_2, double pulse_amp, double pulse_width, double pulse_freq){
        field_drive_1_SU2_SU3 = field_in_SU2_SU3;
        field_drive_2_SU2_SU3 = field_in_2_SU2_SU3;
        field_drive_amp_SU2_SU3 = pulse_amp;
        field_drive_freq_SU2_SU3 = pulse_freq;
        field_drive_width_SU2_SU3 = pulse_width;
        t_B_1_SU2_SU3 = t_B;
        t_B_2_SU2_SU3 = t_B_2;
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

        field_drive_1_SU2_SU3 = {{0}};
        field_drive_2_SU2_SU3 = {{0}};
        field_drive_amp_SU2_SU3 = 0;
        field_drive_freq_SU2_SU3 = 0;
        field_drive_width_SU2_SU3 = 1;
        t_B_1_SU2_SU3 = 0;
        t_B_2_SU2_SU3 = 0;
    }

    array<double, N_SU2>  get_local_field_SU2(size_t site_index){
        array<double,N_SU2> local_field = multiply<N_SU2, N_SU2>(onsite_interaction_SU2[site_index], spins.spins_SU2[site_index]) * 2;

        #pragma omp simd
        for (size_t i=0; i< num_bi_SU2; ++i) {
            local_field = local_field + multiply<N_SU2, N_SU2>(bilinear_interaction_SU2[site_index][i], spins.spins_SU2[bilinear_partners_SU2[site_index][i]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_bi_SU2_SU3; ++i) {
            local_field = local_field + multiply<N_SU2, N_SU3>(mixed_bilinear_interaction_SU2[site_index][i], spins.spins_SU3[mixed_bilinear_partners_SU2[site_index][i]]);
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
        array<double,N_SU3> local_field = multiply<N_SU3, N_SU3>(onsite_interaction_SU3[site_index], spins.spins_SU3[site_index]) * 2;
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU3; ++i) {
            local_field = local_field + multiply<N_SU3, N_SU3>(bilinear_interaction_SU3[site_index][i], spins.spins_SU3[bilinear_partners_SU3[site_index][i]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_bi_SU2_SU3; ++i){
            local_field = local_field + multiply<N_SU3, N_SU2>(mixed_bilinear_interaction_SU3[site_index][i], spins.spins_SU2[mixed_bilinear_partners_SU3[site_index][i]]);
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
        local_field += multiply<N_SU2, N_SU2>(onsite_interaction_SU2[site_index], current_spin_SU2[site_index]);
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU2; ++i) {
            local_field = local_field + multiply<N_SU2, N_SU2>(bilinear_interaction_SU2[site_index][i], current_spin_SU2[bilinear_partners_SU2[site_index][i]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_bi_SU2_SU3; ++i) {
            local_field = local_field + multiply<N_SU2, N_SU3>(mixed_bilinear_interaction_SU2[site_index][i], current_spin_SU3[mixed_bilinear_partners_SU2[site_index][i]]);
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
        local_field += multiply<N_SU3, N_SU3>(onsite_interaction_SU3[site_index], current_spin_SU3[site_index]);
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU3; ++i) {
            local_field = local_field + multiply<N_SU3, N_SU3>(bilinear_interaction_SU3[site_index][i], current_spin_SU3[bilinear_partners_SU3[site_index][i]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_bi_SU2_SU3; ++i) {
            local_field = local_field + multiply<N_SU3, N_SU2>(mixed_bilinear_interaction_SU3[site_index][i], current_spin_SU2[mixed_bilinear_partners_SU3[site_index][i]]);
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
        
        // Pre-allocate arrays to avoid repeated allocations
        array<double, N_SU2> new_spin_SU2;
        array<double, N_SU3> new_spin_SU3;
        
        // Batch generate random numbers for better performance
        constexpr size_t BATCH_SIZE = 64;
        std::vector<size_t> random_sites(BATCH_SIZE);
        std::vector<double> random_uniforms(BATCH_SIZE);
        
        // Process sites in batches for better cache locality
        for (size_t batch_start = 0; batch_start < total_sites; batch_start += BATCH_SIZE) {
            const size_t batch_end = std::min(batch_start + BATCH_SIZE, total_sites);
            const size_t current_batch_size = batch_end - batch_start;
            
            // Generate random sites and uniform numbers in batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                random_sites[j] = random_int_lehman(total_sites);
                random_uniforms[j] = random_double_lehman(0, 1);
            }
            
            // Process batch
            for (size_t j = 0; j < current_batch_size; ++j) {
                const size_t i = random_sites[j];
                const double rand_uniform = random_uniforms[j];
                
                if (i < lattice_size_SU2) {
                    // SU2 case - inline spin generation to avoid function call overhead
                    if (gaussian) {
                        // Inline gaussian move computation
                        const auto& current_spin_ref = curr_spin.spins_SU2[i];
                        gen_random_spin(new_spin_SU2, spin_length_SU2);
                        
                        // Compute new_spin = current + random * sigma, then normalize
                        double norm_sq = 0.0;
                        #pragma omp simd reduction(+:norm_sq)
                        for (size_t k = 0; k < N_SU2; ++k) {
                            new_spin_SU2[k] = current_spin_ref[k] + new_spin_SU2[k] * sigma;
                            norm_sq += new_spin_SU2[k] * new_spin_SU2[k];
                        }
                        
                        const double inv_norm = spin_length_SU2 / sqrt(norm_sq);
                        #pragma omp simd
                        for (size_t k = 0; k < N_SU2; ++k) {
                            new_spin_SU2[k] *= inv_norm;
                        }
                    } else {
                        gen_random_spin(new_spin_SU2, spin_length_SU2);
                    }
                    
                    const double dE = site_energy_SU2_diff(new_spin_SU2, curr_spin.spins_SU2[i], i);
                    
                    // Branchless acceptance check using conditional move
                    const bool accepted = (dE <= 0) | (rand_uniform < exp(-dE * inv_T));
                    
                    if (accepted) {
                        // Use std::copy for potentially better optimization
                        std::copy(new_spin_SU2.begin(), new_spin_SU2.end(), curr_spin.spins_SU2[i].begin());
                        accept++;
                    }
                } else {
                    // SU3 case
                    const size_t i_SU3 = i - lattice_size_SU2;
                    
                    if (gaussian) {
                        // Inline gaussian move computation
                        const auto& current_spin_ref = curr_spin.spins_SU3[i_SU3];
                        gen_random_spin(new_spin_SU3, spin_length_SU3);
                        
                        // Compute new_spin = current + random * sigma, then normalize
                        double norm_sq = 0.0;
                        #pragma omp simd reduction(+:norm_sq)
                        for (size_t k = 0; k < N_SU3; ++k) {
                            new_spin_SU3[k] = current_spin_ref[k] + new_spin_SU3[k] * sigma;
                            norm_sq += new_spin_SU3[k] * new_spin_SU3[k];
                        }
                        
                        const double inv_norm = spin_length_SU3 / sqrt(norm_sq);
                        #pragma omp simd
                        for (size_t k = 0; k < N_SU3; ++k) {
                            new_spin_SU3[k] *= inv_norm;
                        }
                    } else {
                        gen_random_spin(new_spin_SU3, spin_length_SU3);
                    }
                    
                    const double dE = site_energy_SU3_diff(new_spin_SU3, curr_spin.spins_SU3[i_SU3], i_SU3);
                    
                    // Branchless acceptance check
                    const bool accepted = (dE <= 0) | (rand_uniform < exp(-dE * inv_T));
                    
                    if (accepted) {
                        std::copy(new_spin_SU3.begin(), new_spin_SU3.end(), curr_spin.spins_SU3[i_SU3].begin());
                        accept++;
                    }
                }
            }
        }
        
        return static_cast<double>(accept) / static_cast<double>(total_sites);
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



    // Enhanced simulated annealing with convergence monitoring
    void simulated_annealing_with_convergence(double T_start, double T_end, 
                                             size_t max_steps_per_temp = 10000,
                                             size_t n_deterministics = 1000,
                                             size_t overrelaxation_rate = 0,
                                             bool gaussian_move = true, 
                                             string dir_name = "",
                                             double energy_tol = 1e-8,
                                             double acceptance_tol = 1e-4,
                                             double config_tol = 1e-6,
                                             bool early_stop = true) {
        
        srand(time(NULL));
        seed_lehman(rand() * 2 + 1);
        
        // Setup convergence monitoring
        ConvergenceMonitor monitor;
        monitor.set_tolerances(energy_tol, acceptance_tol, config_tol);
        
        // Create output directory
        if (dir_name != "") {
            filesystem::create_directory(dir_name);
        }
        
        double T = T_start;
        double sigma = 1000;
        size_t total_steps = 0;
        
        // Store initial configuration for change tracking
        auto prev_config = spins;
        
        std::cout << "Starting enhanced simulated annealing with convergence monitoring:" << std::endl;
        std::cout << "T_start: " << T_start << ", T_end: " << T_end << std::endl;
        std::cout << "Max steps per temp: " << max_steps_per_temp << std::endl;
        std::cout << "Tolerances - Energy: " << energy_tol 
                  << ", Acceptance: " << acceptance_tol 
                  << ", Config: " << config_tol << std::endl;
        
        bool globally_converged = false;
        
        while (T > T_end && !globally_converged) {
            size_t steps_at_temp = 0;
            double cumulative_accept = 0;
            bool temp_converged = false;
            
            std::cout << "\n--- Temperature: " << T << " ---" << std::endl;
            
            while (steps_at_temp < max_steps_per_temp && !temp_converged) {
                if (overrelaxation_rate > 0) {
                    overrelaxation();
                    if (steps_at_temp % overrelaxation_rate == 0) {
                        cumulative_accept += metropolis(spins, T, gaussian_move, sigma);
                    }
                } else {
                    cumulative_accept += metropolis(spins, T, gaussian_move, sigma);
                }
                
                // Record metrics every few steps
                if (steps_at_temp % 10 == 0) {
                    double E = energy_density(spins);
                    monitor.record_energy(E);
                    
                    double avg_accept = cumulative_accept / (steps_at_temp / overrelaxation_rate);
                    monitor.record_acceptance(avg_accept);
                    
                    // Calculate configuration change
                    double config_change = calculate_configuration_change(prev_config, spins, spin_length_SU2, spin_length_SU3);
                    monitor.record_config_change(config_change);
                    prev_config = std::move(spins);
                }
                
                // Check convergence at this temperature
                if (steps_at_temp > 1000 && steps_at_temp % 500 == 0) {
                    auto metrics = monitor.get_metrics();
                    // Check if we've converged at this temperature
                    temp_converged = metrics.energy_converged && metrics.acceptance_converged;
                    
                    if (temp_converged) {
                        std::cout << "*** TEMPERATURE CONVERGED at T = " << T << " after " << steps_at_temp << " steps ***" << std::endl;
                        std::cout << "Moving to next temperature..." << std::endl;
                        break; // Exit the inner loop and proceed to next temperature
                    }
                    
                    if (steps_at_temp % int(max_steps_per_temp / 10) == 0) {
                        monitor.print_status(T, total_steps + steps_at_temp);
                    }
                }
                
                // More frequent convergence check after longer equilibration
                else if (steps_at_temp > 3000 && steps_at_temp % 100 == 0) {
                    auto metrics = monitor.get_metrics();
                    if (metrics.energy_converged && metrics.acceptance_converged) {
                        temp_converged = true;
                        std::cout << "*** RAPID CONVERGENCE DETECTED at T = " << T << " after " << steps_at_temp << " steps ***" << std::endl;
                        std::cout << "Moving to next temperature..." << std::endl;
                        break; // Exit immediately when converged
                    }
                }
                
                steps_at_temp++;
            }

            // Adjust sigma based on acceptance rate
            auto metrics = monitor.get_metrics();
            if (gaussian_move && metrics.acceptance_rate > 0 && metrics.acceptance_rate < 0.5) {
                sigma = sigma * 0.5 / (1 - metrics.acceptance_rate);
                std::cout << "Sigma adjusted to: " << sigma << std::endl;
            }
            
            // Final metrics for this temperature
            auto final_metrics = monitor.get_metrics();
            std::cout << "Temperature " << T << " completed:" << std::endl;
            std::cout << "  Steps taken: " << steps_at_temp << " / " << max_steps_per_temp << std::endl;
            std::cout << "  Final energy: " << final_metrics.energy_mean << std::endl;
            std::cout << "  Final acceptance: " << final_metrics.acceptance_rate << std::endl;
            std::cout << "  Energy variance: " << final_metrics.energy_variance << std::endl;
            std::cout << "  Config change: " << final_metrics.config_change << std::endl;
            std::cout << "  Converged: " << (temp_converged ? "Yes (early exit)" : "No (max steps reached)") << std::endl;
            
            monitor.reset_for_new_temperature();
            total_steps += steps_at_temp;
            
            // Temperature reduction
            T *= 0.9;
            
            // Check for global convergence at low temperatures
            if (early_stop && T < T_end * 2) {
                globally_converged = final_metrics.is_fully_converged();
                if (globally_converged) {
                    std::cout << "\n*** GLOBAL CONVERGENCE ACHIEVED ***" << std::endl;
                    std::cout << "Stopping early at T = " << T << std::endl;
                }
            }
        }
        
        // Final deterministic optimization
        std::cout << "\nPerforming final deterministic optimization..." << std::endl;
        for (size_t i = 0; i < n_deterministics; ++i) {
            deterministic_sweep();
        }
        
        // Output final results
        double final_energy = total_energy(spins);
        std::cout << "\nSimulated annealing completed:" << std::endl;
        std::cout << "  Total steps: " << total_steps << std::endl;
        std::cout << "  Final temperature: " << T << std::endl;
        std::cout << "  Final energy per site: : " << final_energy << std::endl;
        std::cout << "  Globally converged: " << (globally_converged ? "Yes" : "No") << std::endl;
        
        // Save results
        if (dir_name != "") {
            filesystem::create_directory(dir_name);
            write_to_file_spin(dir_name + "/spin_final");
            write_to_file_pos(dir_name + "/pos");
            monitor.save_convergence_data(dir_name + "/convergence_data.txt");
            
            // Save final energy
            std::ofstream energy_file(dir_name + "/final_energy.txt");
            energy_file << final_energy << std::endl;
            energy_file.close();
        }
    }

    void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure, size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate, string dir_name, const vector<int> rank_to_write, bool gaussian_move = true){

        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized){
            MPI_Init(NULL, NULL);
        }

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        srand(time(NULL) + rank); // Ensure different seeds per rank
        seed_lehman(rand() * 2 + 1);

        double curr_Temp = temp[rank];
        double E = total_energy(spins);

        // Use heap allocation for large temporary buffers to prevent stack overflow
        std::vector<double> recv_buffer_SU2(N_SU2 * lattice_size_SU2);
        std::vector<double> send_buffer_SU2(N_SU2 * lattice_size_SU2);
        std::vector<double> recv_buffer_SU3(N_SU3 * lattice_size_SU3);
        std::vector<double> send_buffer_SU3(N_SU3 * lattice_size_SU3);

        long long swap_attempts = 0;
        long long swap_accepts = 0;
        double total_accept = 0;
        size_t metropolis_steps = 0;

        cout << "Initialized Process on rank: " << rank << " with temperature: " << curr_Temp << endl;

        for(size_t i = 0; i < n_anneal + n_measure; ++i){
            // Metropolis step
            if (overrelaxation_rate > 0 && i % overrelaxation_rate != 0) {
                overrelaxation();
            } else {
                total_accept += metropolis(spins, curr_Temp, gaussian_move);
                metropolis_steps++;
            }

            // Parallel Tempering Swap Attempt
            if (i > 0 && i % swap_rate == 0) {
                E = total_energy(spins);
                swap_attempts++;

                // Determine swap partner
                int partner_rank;
                if ((i / swap_rate) % 2 == 0) { // Pair (0,1), (2,3), ...
                    partner_rank = (rank % 2 == 0) ? rank + 1 : rank - 1;
                } else { // Pair (1,2), (3,4), ...
                    partner_rank = (rank % 2 != 0) ? rank + 1 : rank - 1;
                }

                if (partner_rank >= 0 && partner_rank < size) {
                    double E_partner;
                    // Exchange energies with partner using a single, safer call
                    MPI_Sendrecv(&E, 1, MPI_DOUBLE, partner_rank, 0,
                                 &E_partner, 1, MPI_DOUBLE, partner_rank, 0,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    bool accept_swap = false;
                    if (rank < partner_rank) { // Lower rank decides
                        double delta_beta = (1.0 / temp[partner_rank]) - (1.0 / curr_Temp);
                        double delta_E = E - E_partner;
                        if (delta_beta * delta_E <= 0 || random_double_lehman(0, 1) < exp(delta_beta * delta_E)) {
                            accept_swap = true;
                        }
                        MPI_Send(&accept_swap, 1, MPI_C_BOOL, partner_rank, 1, MPI_COMM_WORLD);
                    } else {
                        MPI_Recv(&accept_swap, 1, MPI_C_BOOL, partner_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }

                    if (accept_swap) {
                        swap_accepts++;
                        // Print swap information
                        cout << "Rank " << rank << " accepted swap with rank " << partner_rank
                                << " at step " << i << ": E = " << E << ", E_partner = " << E_partner << endl;
                        E = E_partner; // Energy is swapped regardless of who sends/receives first

                        // Prepare send buffers
                        std::copy(spins.spins_SU2.begin(), spins.spins_SU2.end(), reinterpret_cast<array<double, N_SU2>*>(send_buffer_SU2.data()));
                        std::copy(spins.spins_SU3.begin(), spins.spins_SU3.end(), reinterpret_cast<array<double, N_SU3>*>(send_buffer_SU3.data()));

                        // Exchange SU2 configurations
                        MPI_Sendrecv(send_buffer_SU2.data(), send_buffer_SU2.size(), MPI_DOUBLE, partner_rank, 2,
                                     recv_buffer_SU2.data(), recv_buffer_SU2.size(), MPI_DOUBLE, partner_rank, 2,
                                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        
                        // Exchange SU3 configurations
                        MPI_Sendrecv(send_buffer_SU3.data(), send_buffer_SU3.size(), MPI_DOUBLE, partner_rank, 3,
                                     recv_buffer_SU3.data(), recv_buffer_SU3.size(), MPI_DOUBLE, partner_rank, 3,
                                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        // Copy received data back to spins
                        std::copy(recv_buffer_SU2.begin(), recv_buffer_SU2.end(), reinterpret_cast<double*>(spins.spins_SU2.data()));
                        std::copy(recv_buffer_SU3.begin(), recv_buffer_SU3.end(), reinterpret_cast<double*>(spins.spins_SU3.data()));
                    }
                }
            }

            if (i > 0 && i % 10000 == 0){
                double metro_rate = (metropolis_steps > 0) ? total_accept / metropolis_steps : 0.0;
                double swap_rate_val = (swap_attempts > 0) ? static_cast<double>(swap_accepts) / swap_attempts : 0.0;
                std::cout << "Rank " << rank << " (" << double(i) * 100.0 / (n_anneal + n_measure) << "%): "
                          << "Metro Accept Rate: " << metro_rate << ", Swap Accept Rate: " << swap_rate_val << std::endl;
            }
        }
        
        cout << "Process finished on rank: " << rank << " with temperature: " << curr_Temp << endl;

        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            for(int rank_to_write_val : rank_to_write){
                if (rank == rank_to_write_val){
                    write_to_file_spin(dir_name + "/spin" + to_string(rank));
                    if (rank == 0) {
                        write_to_file_pos(dir_name + "/pos");
                    }
                }
            }
        }

        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized){
            MPI_Finalize();
        }
    }

    const array<double, N_SU2> drive_field_T_SU2(double currT, size_t ind, const spin_config_SU3 &spins_SU3) {
        // Pre-compute expensive transcendental functions once
        const double exp_factor1 = exp(-pow((currT - t_B_1_SU2) / (2 * field_drive_width_SU2), 2));
        const double exp_factor2 = exp(-pow((currT - t_B_2_SU2) / (2 * field_drive_width_SU2), 2));
        const double cos_factor1 = cos(2 * M_PI * field_drive_freq_SU2 * (currT - t_B_1_SU2));
        const double cos_factor2 = cos(2 * M_PI * field_drive_freq_SU2 * (currT - t_B_2_SU2));
        
        const double factor1_SU2 = field_drive_amp_SU2 * exp_factor1 * cos_factor1;
        const double factor2_SU2 = field_drive_amp_SU2 * exp_factor2 * cos_factor2;
        
        // Early exit if both factors are negligible
        constexpr double EPSILON = 1e-15;
        if (std::abs(factor1_SU2) < EPSILON && std::abs(factor2_SU2) < EPSILON) {
            return {{0}};
        }
        
        // Get the atom index once
        const size_t atom_idx = ind % N_ATOMS_SU2;
        
        // Initialize result with the direct field contribution
        array<double, N_SU2> result;
        #pragma omp simd
        for (size_t j = 0; j < N_SU2; ++j) {
            result[j] = field_drive_1_SU2[atom_idx][j] * factor1_SU2 + 
                        field_drive_2_SU2[atom_idx][j] * factor2_SU2;
        }
        return result;
    }

    const array<double, N_SU3> drive_field_T_SU3(double currT, size_t ind, const spin_config_SU2 &spins_SU2) {
        // Pre-compute expensive transcendental functions once
        const double exp_factor1 = exp(-pow((currT - t_B_1_SU2) / (2 * field_drive_width_SU2), 2));
        const double exp_factor2 = exp(-pow((currT - t_B_2_SU2) / (2 * field_drive_width_SU2), 2));
        const double cos_factor1 = cos(2 * M_PI * field_drive_freq_SU2 * (currT - t_B_1_SU2));
        const double cos_factor2 = cos(2 * M_PI * field_drive_freq_SU2 * (currT - t_B_2_SU2));
        
        const double factor1_SU3 = field_drive_amp_SU2 * exp_factor1 * cos_factor1;
        const double factor2_SU3 = field_drive_amp_SU2 * exp_factor2 * cos_factor2;
        
        // Early exit if both factors are negligible
        constexpr double EPSILON = 1e-15;
        if (std::abs(factor1_SU3) < EPSILON && std::abs(factor2_SU3) < EPSILON) {
            return {{0}};
        }
        
        // Get the atom index once
        const size_t atom_idx = ind % N_ATOMS_SU3;
    
        // Initialize result with the direct field contribution
        array<double, N_SU3> result;
        #pragma omp simd
        for (size_t j = 0; j < N_SU3; ++j) {
            result[j] = field_drive_1_SU3[atom_idx][j] * factor1_SU3 + 
                        field_drive_2_SU3[atom_idx][j] * factor2_SU3;
        }
        return result;
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
            array<double, N_SU2> drive_field = drive_field_T_SU2(curr_time, i, current_spin_SU3);
            
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
            array<double, N_SU3> drive_field = drive_field_T_SU3(curr_time, i, current_spin_SU2);
            
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