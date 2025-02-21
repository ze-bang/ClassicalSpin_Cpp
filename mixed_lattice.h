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

    int set(const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> &spins_in){
        spins_SU2 = spins_in.spins_SU2;
        spins_SU3 = spins_in.spins_SU3;
        return 0;
    };
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

template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> operator+(const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> &a, const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> &b) {
    mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> result;
    for (size_t i = 0; i < lattice_size_SU2; ++i) {
        for (size_t j = 0; j < N_SU2; ++j) {
            result.spins_SU2[i][j] = a.spins_SU2[i][j] + b.spins_SU2[i][j];
        }
    }
    for (size_t i = 0; i < lattice_size_SU3; ++i) {
        for (size_t j = 0; j < N_SU3; ++j) {
            result.spins_SU3[i][j] = a.spins_SU3[i][j] + b.spins_SU3[i][j];
        }
    }
    return result;
}

template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> operator-(const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> &a, const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> &b) {
    mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> result;
    for (size_t i = 0; i < lattice_size_SU2; ++i) {
        for (size_t j = 0; j < N_SU2; ++j) {
            result.spins_SU2[i][j] = a.spins_SU2[i][j] + b.spins_SU2[i][j];
        }
    }
    for (size_t i = 0; i < lattice_size_SU3; ++i) {
        for (size_t j = 0; j < N_SU3; ++j) {
            result.spins_SU3[i][j] = a.spins_SU3[i][j] + b.spins_SU3[i][j];
        }
    }
    return result;
}

template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> operator*(const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> &a, const double &b) {
    mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> result;
    for (size_t i = 0; i < lattice_size_SU2; ++i) {
        for (size_t j = 0; j < N_SU2; ++j) {
            result.spins_SU2[i][j] = a.spins_SU2[i][j]*b;
        }
    }
    for (size_t i = 0; i < lattice_size_SU3; ++i) {
        for (size_t j = 0; j < N_SU3; ++j) {
            result.spins_SU3[i][j] = a.spins_SU3[i][j]*b;
        }
    }
    return result;
}

template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> operator/(const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> &a, const double &b) {
    mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> result;
    for (size_t i = 0; i < lattice_size_SU2; ++i) {
        for (size_t j = 0; j < N_SU2; ++j) {
            result.spins_SU2[i][j] = a.spins_SU2[i][j]/b;
        }
    }
    for (size_t i = 0; i < lattice_size_SU3; ++i) {
        for (size_t j = 0; j < N_SU3; ++j) {
            result.spins_SU3[i][j] = a.spins_SU3[i][j]/b;
        }
    }
    return result;
}

template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
double dot(const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> &a, const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3> &b) {
    double result = 0;
    for (size_t i = 0; i < lattice_size_SU2; ++i) {
        for (size_t j = 0; j < N_SU2; ++j) {
            result += a.spins_SU2[i][j] * b.spins_SU2[i][j];
        }
    }
    for (size_t i = 0; i < lattice_size_SU3; ++i) {
        for (size_t j = 0; j < N_SU3; ++j) {
            result += a.spins_SU3[i][j] * b.spins_SU3[i][j];
        }
    }
    return result;
}

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim3>
class mixed_lattice
{   
    public:

    typedef array<array<double,N_SU2>,N_ATOMS_SU2*dim1*dim2*dim3> spin_config_SU2;
    typedef array<array<double,N_SU3>,N_ATOMS_SU3*dim1*dim2*dim3> spin_config_SU3;

    mixed_UnitCell<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3> UC;
    size_t lattice_size_SU2;
    size_t lattice_size_SU3;
    double spin_length_SU2, spin_length_SU3;
    mixed_lattice_spin<N_SU2, dim1*dim2*dim3*N_ATOMS_SU2, N_SU3, dim1*dim2*dim3*N_ATOMS_SU3> spins;
    mixed_lattice_pos<dim1*dim2*dim3*N_ATOMS_SU2, dim1*dim2*dim3*N_ATOMS_SU3>  site_pos;

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
    array<array<double,N_SU2>, N_ATOMS_SU2*dim1*dim2*dim3> field_SU2;
    array<array<double,N_SU2>, N_ATOMS_SU2> field_drive_1_SU2;
    array<array<double,N_SU2>, N_ATOMS_SU2> field_drive_2_SU2;

    array<array<double, N_SU2 * N_SU2>, N_ATOMS_SU2*dim1*dim2*dim3> onsite_interaction_SU2;
    array<vector<array<double, N_SU2 * N_SU2>>, N_ATOMS_SU2*dim1*dim2*dim3> bilinear_interaction_SU2;    
    array<vector<array<array<array<double, N_SU2>, N_SU2>, N_SU2>>, N_ATOMS_SU2*dim1*dim2*dim3> trilinear_interaction_SU2;

    array<vector<size_t>, N_ATOMS_SU2*dim1*dim2*dim3> bilinear_partners_SU2;
    array<vector<array<size_t, 2>>, N_ATOMS_SU2*dim1*dim2*dim3> trilinear_partners_SU2;

    //Look up table for SU3
    array<array<double,N_SU3>, N_ATOMS_SU3*dim1*dim2*dim3> field_SU3;
    array<array<double,N_SU3>, N_ATOMS_SU3> field_drive_1_SU3;
    array<array<double,N_SU3>, N_ATOMS_SU3> field_drive_2_SU3;

    array<array<double, N_SU3 * N_SU3>, N_ATOMS_SU2*dim1*dim2*dim3> onsite_interaction_SU3;
    array<vector<array<double, N_SU3* N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim3> bilinear_interaction_SU3;
    array<vector<array<array<array<double, N_SU3>, N_SU3>, N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim3> trilinear_interaction_SU3;

    array<vector<size_t>, N_ATOMS_SU3*dim1*dim2*dim3> bilinear_partners_SU3;
    array<vector<array<size_t, 2>>, N_ATOMS_SU3*dim1*dim2*dim3> trilinear_partners_SU3;

    //Look up table for SU2 and SU3 mix
    array<vector<array<double, N_SU2 * N_SU3>>, N_ATOMS_SU2*dim1*dim2*dim3> mixed_bilinear_interaction_SU2;
    array<vector<array<size_t, 2>>, N_ATOMS_SU2*dim1*dim2*dim3> mixed_bilinear_partners_SU2;
    array<vector<array<size_t, 2>>, N_ATOMS_SU3*dim1*dim2*dim3> mixed_bilinear_partners_SU3;

    array<vector<array<array<array<double, N_SU3>, N_SU2>, N_SU2>>, N_ATOMS_SU2*dim1*dim2*dim3> mixed_trilinear_interaction_SU2;
    array<vector<array<array<array<double, N_SU2>, N_SU2>, N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim3> mixed_trilinear_interaction_SU3;

    array<vector<array<size_t, 2>>, N_ATOMS_SU2*dim1*dim2*dim3> mixed_trilinear_partners_SU2;
    array<vector<array<size_t, 2>>, N_ATOMS_SU3*dim1*dim2*dim3> mixed_trilinear_partners_SU3;

    size_t num_bi_SU2;
    size_t num_tri_SU2;
    size_t num_bi_SU3;
    size_t num_tri_SU3;
    size_t num_tri_SU2_SU3;

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
        return i*dim2*dim3*N_ATOMS+ j*dim3*N_ATOMS+ k*N_ATOMS + l;
    }
    
    size_t periodic_boundary(int i, size_t dim){
        if(i < 0){
            return size_t((dim+i) % dim);
        }
        else{
            return size_t(i % dim);
        }
    }

    size_t flatten_index_periodic_boundary(int i, int j, int k, int l, size_t N_ATOMS){
        return periodic_boundary(i, dim1)*dim2*dim3*N_ATOMS+ periodic_boundary(j, dim2)*dim3*N_ATOMS+ periodic_boundary(k, dim3)*N_ATOMS + l;
    }

    template<size_t N, size_t lattice_size, size_t N_ATOMS>
    void set_up_sublattice(const double spin_length, array<array<double,N>, lattice_size>  &spins, array<array<double,3>, lattice_size> &site_pos, array<array<double,N>, lattice_size> &field, array<array<double, N * N>, lattice_size> &onsite_interaction, array<vector<array<double, N * N>>, lattice_size> &bilinear_interaction, array<vector<array<array<array<double, N>, N>, N>>, lattice_size> &trilinear_interaction, array<vector<size_t>, lattice_size> &bilinear_partners, array<vector<array<size_t, 2>>, lattice_size> &trilinear_partners, UnitCell<N, N_ATOMS> *atoms, size_t &num_bi, size_t &num_tri){
        array<array<double,3>, N_ATOMS> basis = atoms->lattice_pos;
        array<array<double,3>, 3> unit_vector = atoms->lattice_vectors;

        for (size_t i=0; i<dim1; ++i){
            for (size_t j=0; j< dim2; ++j){
                for(size_t k=0; k<dim3;++k){
                    for (size_t l=0; l<N_ATOMS;++l){
                        size_t current_site_index = flatten_index(i,j,k,l,N_ATOMS);

                        site_pos[current_site_index]  = unit_vector[0]*int(i) + unit_vector[1]*int(j) + unit_vector[2]*int(k) + basis[l];

                        gen_random_spin(spins[current_site_index], spin_length);
                        field[current_site_index] = atoms->field[l];
                        onsite_interaction[current_site_index] = atoms->onsite_interaction[l];

                        auto bilinear_matched = atoms->bilinear_interaction.equal_range(l);
                        for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m){
                            bilinear<N> J = m->second;
                            size_t partner = flatten_index_periodic_boundary(int(i)+J.offset[0], int(j)+J.offset[1], int(k)+J.offset[2], J.partner,N_ATOMS);
                            bilinear_interaction[current_site_index].push_back(J.bilinear_interaction);
                            bilinear_partners[current_site_index].push_back(partner);
                            bilinear_interaction[partner].push_back(J.bilinear_interaction);
                            bilinear_partners[partner].push_back(current_site_index);
                        }

                        auto trilinear_matched = atoms->trilinear_interaction.equal_range(l);
                        for (auto m = trilinear_matched.first; m != trilinear_matched.second; ++m){
                            trilinear<N> J = m->second;
                            size_t partner1 = flatten_index_periodic_boundary(i+J.offset1[0], j+J.offset1[1], k+J.offset1[2], J.partner1,N_ATOMS);
                            size_t partner2 = flatten_index_periodic_boundary(i+J.offset2[0], j+J.offset2[1], k+J.offset2[2], J.partner2,N_ATOMS);
                            
                            trilinear_interaction[current_site_index].push_back(J.trilinear_interaction);
                            trilinear_partners[current_site_index].push_back({partner1, partner2});

                            trilinear_interaction[partner1].push_back(transpose3D(J.trilinear_interaction));
                            trilinear_partners[partner1].push_back({partner2, current_site_index});

                            trilinear_interaction[partner2].push_back(transpose3D(transpose3D(J.trilinear_interaction)));
                            trilinear_partners[partner2].push_back({current_site_index, partner1});
                        }
                    }
                }
            }
        }

        num_bi = bilinear_partners[0].size();
        num_tri = trilinear_partners[0].size();
    }
    

    mixed_lattice(mixed_UnitCell<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3> *atoms, double spin_length_SU2_in, double spin_length_SU3_in): UC(*atoms){

        srand(time(NULL));
        seed_lehman(rand()*2+1);
        lehman_next();

        field_drive_freq_SU2 = 0;
        field_drive_amp_SU2 = 0;
        field_drive_width_SU2 = 1;
        field_drive_freq_SU3 = 0;
        field_drive_amp_SU3 = 0;
        field_drive_width_SU3 = 1;
        t_B_1_SU2 = 0;
        t_B_2_SU2 = 0;
        t_B_1_SU3 = 0;
        t_B_2_SU3 = 0;
        spin_length_SU2 = spin_length_SU2_in;
        spin_length_SU3 = spin_length_SU3_in;

        set_up_sublattice(spin_length_SU2, spins.spins_SU2, site_pos.pos_SU2, field_SU2, onsite_interaction_SU2, bilinear_interaction_SU2, trilinear_interaction_SU2, bilinear_partners_SU2, trilinear_partners_SU2, &(atoms->SU2), num_bi_SU2, num_tri_SU2);
        set_up_sublattice(spin_length_SU3, spins.spins_SU3, site_pos.pos_SU3, field_SU3, onsite_interaction_SU3, bilinear_interaction_SU3, trilinear_interaction_SU3, bilinear_partners_SU3, trilinear_partners_SU3, &(atoms->SU3), num_bi_SU3, num_tri_SU3);
        
        for (size_t i=0; i < dim1; ++i){
            for (size_t j=0; j < dim2; ++j){
                for(size_t k=0; k < dim3;++k){
                    for (size_t l=0; l < N_ATOMS_SU3;++l){
                        size_t current_site_index = flatten_index(i,j,k,l,N_ATOMS_SU3);         

                        auto trilinear_matched = atoms->trilinear_SU2_SU3.equal_range(l);
                        for (auto m = trilinear_matched.first; m != trilinear_matched.second; ++m){
                            mixed_trilinear<N_SU2, N_SU3> J = m->second;
                            size_t partner1 = flatten_index_periodic_boundary(i+J.offset1[0], j+J.offset1[1], k+J.offset1[2], J.partner1, N_ATOMS_SU2);
                            size_t partner2 = flatten_index_periodic_boundary(i+J.offset2[0], j+J.offset2[1], k+J.offset2[2], J.partner2, N_ATOMS_SU2);
                            
                            mixed_trilinear_interaction_SU3[current_site_index].push_back(J.trilinear_interaction);
                            mixed_trilinear_partners_SU3[current_site_index].push_back({partner1, partner2});

                            mixed_trilinear_interaction_SU2[partner1].push_back(transpose3D(J.trilinear_interaction));
                            mixed_trilinear_partners_SU2[partner1].push_back({partner2, current_site_index});

                            mixed_trilinear_interaction_SU2[partner2].push_back(swap_axis_3D(transpose3D(J.trilinear_interaction)));
                            mixed_trilinear_partners_SU2[partner2].push_back({partner1, current_site_index});
                        }
                    }
                }
            }
        }

        num_tri_SU2_SU3 = mixed_trilinear_partners_SU3[0].size();
        lattice_size_SU2 = dim1*dim2*dim3*N_ATOMS_SU2;
        lattice_size_SU3 = dim1*dim2*dim3*N_ATOMS_SU3;
        cout << "Finished setting up lattice" << endl;  
        // cout << num_bi_SU2 << " " << num_tri_SU2 << " " << num_bi_SU3 << " " << num_tri_SU3 << " " << num_tri_SU2_SU3 << endl;
    };

    void read_spin_from_file(const string &filename){
        ifstream file;
        file.open(filename+"_SU2.txt");
        if (!file){
            cout << "Unable to open file";
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
            cout << "Unable to open file";
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
        for (size_t i=0; i < num_tri_SU3; ++i){
            energy += contract_trilinear(trilinear_interaction_SU2[site_index][i], spin_here, spins.spins_SU2[trilinear_partners_SU2[site_index][i][0]], spins.spins_SU2[trilinear_partners_SU2[site_index][i][1]]);
        }
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
        for (size_t i=0; i < num_tri_SU3; ++i){
            energy += contract_trilinear(trilinear_interaction_SU3[site_index][i], spin_here, spins.spins_SU3[trilinear_partners_SU3[site_index][i][0]], spins.spins_SU3[trilinear_partners_SU3[site_index][i][1]]);
        }
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            energy += contract_trilinear(mixed_trilinear_interaction_SU3[site_index][i], spin_here, spins.spins_SU2[mixed_trilinear_partners_SU3[site_index][i][0]], spins.spins_SU2[mixed_trilinear_partners_SU3[site_index][i][1]]);
        }
        return energy;
    }

    template<size_t N>
    double site_energy(array<double,N> &spins, size_t site_index){
        if (N == N_SU2){
            return site_energy_SU2(spins, site_index);
        }else{
            return site_energy_SU3(spins, site_index);
        }
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
        array<double,N_SU2> local_field = {0};
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU2; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU2[site_index][i], spins.spins_SU2[bilinear_partners_SU2[site_index][i]]);
        }
        for (size_t i=0; i < num_tri_SU2; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction_SU2[site_index][i], spins.spins_SU2[trilinear_partners_SU2[site_index][i][0]], spins.spins_SU2[trilinear_partners_SU2[site_index][i][1]]);
        }
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            local_field = local_field + contract_trilinear_field(mixed_trilinear_interaction_SU2[site_index][i], spins.spins_SU2[mixed_trilinear_partners_SU2[site_index][i][0]], spins.spins_SU3[mixed_trilinear_partners_SU2[site_index][i][1]]);
        }
        return local_field-field_SU2[site_index];
    }

    array<double, N_SU3>  get_local_field_SU3(size_t site_index){
        array<double,N_SU3> local_field = {0};
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU3; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU3[site_index][i], spins.spins_SU3[bilinear_partners_SU3[site_index][i]]);
        }
        for (size_t i=0; i < num_tri_SU3; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction_SU3[site_index][i], spins.spins_SU3[trilinear_partners_SU3[site_index][i][0]], spins.spins_SU3[trilinear_partners_SU3[site_index][i][1]]);
        }
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            local_field = local_field + contract_trilinear_field(mixed_trilinear_interaction_SU3[site_index][i], spins.spins_SU2[mixed_trilinear_partners_SU3[site_index][i][0]], spins.spins_SU2[mixed_trilinear_partners_SU3[site_index][i][1]]);
        }
        return local_field-field_SU3[site_index];
    }

    array<double, N_SU2>  get_local_field_SU2_lattice(size_t site_index, const spin_config_SU2 &current_spin_SU2, const spin_config_SU3 &current_spin_SU3){
        array<double,N_SU2> local_field = {0};
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU2; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU2[site_index][i], current_spin_SU2[bilinear_partners_SU2[site_index][i]]);
        }
        for (size_t i=0; i < num_tri_SU2; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction_SU2[site_index][i], current_spin_SU2[trilinear_partners_SU2[site_index][i][0]], current_spin_SU2[trilinear_partners_SU2[site_index][i][1]]);
        }
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            local_field = local_field + contract_trilinear_field(mixed_trilinear_interaction_SU2[site_index][i], current_spin_SU2[mixed_trilinear_partners_SU2[site_index][i][0]], current_spin_SU3[mixed_trilinear_partners_SU2[site_index][i][1]]);
        }
        return local_field-field_SU2[site_index];
    }

    array<double, N_SU3>  get_local_field_SU3_lattice(size_t site_index, const spin_config_SU2 &current_spin_SU2, const spin_config_SU3 &current_spin_SU3){
        array<double,N_SU3> local_field = {0};
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU3; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU3[site_index][i], current_spin_SU3[bilinear_partners_SU3[site_index][i]]);
        }
        for (size_t i=0; i < num_tri_SU3; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction_SU3[site_index][i], current_spin_SU3[trilinear_partners_SU3[site_index][i][0]], current_spin_SU3[trilinear_partners_SU3[site_index][i][1]]);
        }
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            local_field = local_field + contract_trilinear_field(mixed_trilinear_interaction_SU3[site_index][i], current_spin_SU2[mixed_trilinear_partners_SU3[site_index][i][0]], current_spin_SU2[mixed_trilinear_partners_SU3[site_index][i][1]]);
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

    double metropolis(double T, bool gaussian=false, double sigma=60){
        double E, E_new, dE, r;
        float spinl;
        int i;
        int accept = 0;
        size_t count = 0;

        while(count < lattice_size_SU2+lattice_size_SU3){
            int SU2_or_SU3 = random_int_lehman(2);
            if (SU2_or_SU3 == 0){
                i = random_int_lehman(lattice_size_SU2);
                auto temp_spin = spins.spins_SU2[i];
                E = site_energy_SU2(temp_spin, i);
                if (gaussian){
                    spins.spins_SU2[i] = gaussian_move_SU2(spins.spins_SU2[i], sigma);
                }else{
                    gen_random_spin(spins.spins_SU2[i], spin_length_SU2);
                }
                E_new = site_energy_SU2(spins.spins_SU2[i], i);
                dE = E_new - E;
            
                if(dE < 0){
                    accept++;
                }
                else{
                    r = random_double_lehman(0,1);
                    if(r < exp(-dE/T)){
                        accept++;
                    }
                    else{
                        spins.spins_SU2[i] = temp_spin;
                    }
                }
                count++; 
            }else{
                i = random_int_lehman(lattice_size_SU3);
                // cout << i << " " << lattice_size_SU3 << endl;
                auto temp_spin = spins.spins_SU3[i];
                E = site_energy_SU3(temp_spin, i);
                if (gaussian){
                    spins.spins_SU3[i] = gaussian_move_SU3(spins.spins_SU3[i], sigma);
                }else{
                    gen_random_spin(spins.spins_SU3[i], spin_length_SU3);
                }                
                E_new = site_energy_SU3(spins.spins_SU3[i], i);
                dE = E_new - E;
            
                if(dE < 0){
                    accept++;
                }
                else{
                    r = random_double_lehman(0,1);
                    if(r < exp(-dE/T)){
                        accept++;
                    }
                    else{
                        spins.spins_SU3[i] = temp_spin;
                    }
                }
                count++; 
            }

        }

        double acceptance_rate = double(accept)/double(lattice_size_SU2+lattice_size_SU3);
        return acceptance_rate;
    }

    void overrelaxation(){
        array<double,N_SU2> local_field_SU2;
        array<double,N_SU3> local_field_SU3;
        int i;
        double proj;
        size_t count = 0;
        while(count < lattice_size_SU2){
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
        count = 0;
        while(count < lattice_size_SU3){
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
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            array<double,N_SU2> local_field = get_local_field_SU2(i);
            double norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                spins.spins_SU2[i] = local_field/(-norm);
            }
        }
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            array<double,N_SU3> local_field = get_local_field_SU3(i);
            double norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                spins.spins_SU3[i] = local_field/(-norm);
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

    void write_to_file(string filename, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> towrite){
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

    void print_mixed_spin(mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> toprint){
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


    void simulated_annealing(double T_start, double T_end, size_t n_anneal, size_t n_deterministics, size_t overrelaxation_rate, bool gaussian_move=false, string dir_name=""){
        srand (time(NULL));
        seed_lehman(rand()*2+1);
        double curr_accept;
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        double T = T_start;
        double sigma = 40;
        double acceptance_rate = 0;

        while(T > T_end){
            curr_accept = 0;
            for(size_t i = 0; i<n_anneal; ++i){
                if(overrelaxation_rate > 0){
                    overrelaxation();
                    if (i%overrelaxation_rate == 0){
                        curr_accept += metropolis(T, gaussian_move,sigma);
                    }
                }
                else{
                    curr_accept += metropolis(T, gaussian_move,sigma);
                }
            }
            if (overrelaxation_rate > 0){
                acceptance_rate = curr_accept/n_anneal*overrelaxation_rate;
                cout << "Temperature: " << T << " Acceptance rate: " << acceptance_rate << endl;
            }else{
                acceptance_rate = curr_accept/n_anneal;
                cout << "Temperature: " << T << " Acceptance rate: " << acceptance_rate << endl;
            }
            if (gaussian_move){
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

    mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> landau_lifshitz(const mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> &current_spin, const double &curr_time){
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> dS;
        #pragma omp simd
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            dS.spins_SU2[i] = cross_prod_SU2(get_local_field_SU2_lattice(i, current_spin) - drive_field_T_SU2(curr_time, i % N_ATOMS_SU2), current_spin.spins_SU2[i]);
        }
        #pragma omp simd
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            dS.spins_SU3[i] = cross_prod_SU3(get_local_field_SU3_lattice(i, current_spin)- drive_field_T_SU3(curr_time, i % N_ATOMS_SU3), current_spin.spins_SU3[i]);
            cout << dS.spins_SU3[i][0] << " " << dS.spins_SU3[i][1] << " " << dS.spins_SU3[i][2] << " " << dS.spins_SU3[i][3] << " " << dS.spins_SU3[i][4] << " " << dS.spins_SU3[i][5] << " " << dS.spins_SU3[i][6] << " " << dS.spins_SU3[i][7] << endl;
        }
        return dS;
    }


    spin_config_SU2 landau_lifshitz_SU2(const spin_config_SU2 &current_spin_SU2, const spin_config_SU3 &current_spin_SU3, const double &curr_time){
        spin_config_SU2 dS;
        #pragma omp simd
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            dS[i] = cross_prod_SU2(get_local_field_SU2_lattice(i, current_spin_SU2, current_spin_SU3)- drive_field_T_SU2(curr_time, i % N_ATOMS_SU2), current_spin_SU2[i]);
        }
        return dS;
    }

    spin_config_SU3 landau_lifshitz_SU3(const spin_config_SU2 &current_spin_SU2, const spin_config_SU3 &current_spin_SU3, const double &curr_time){
        spin_config_SU3 dS;
        #pragma omp simd
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            dS[i] = cross_prod_SU3(get_local_field_SU3_lattice(i, current_spin_SU2, current_spin_SU3)- drive_field_T_SU3(curr_time, i % N_ATOMS_SU3), current_spin_SU3[i]);
        }
        return dS;
    }


    mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> RK4_step(double &step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> &curr_spins, const double &curr_time, const double tol){
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

        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> new_spins;
        new_spins.spins_SU2 = curr_spins.spins_SU2+ (k1_SU2+ k2_SU2 * 2 + k3_SU2 * 2 + k4_SU2)*(step_size/6);
        new_spins.spins_SU3 = curr_spins.spins_SU3+ (k1_SU3+ k2_SU3 * 2 + k3_SU3 * 2 + k4_SU3)*(step_size/6);

        return new_spins;
    }


    // mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> RK45_step(double &step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> &curr_spins, const double &curr_time, const double tol){
        
    //     spin_config_SU2 k1_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time)*step_size;
    //     spin_config_SU2 k2_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(1.0/4.0), curr_spins.spins_SU3, curr_time + step_size*(1/4))*step_size;
    //     spin_config_SU2 k3_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(3.0/32.0)+ k2_SU2*(9.0/32.0), curr_spins.spins_SU3 , curr_time + step_size*(3/8))*step_size;
    //     spin_config_SU2 k4_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(1932.0/2197.0)+ k2_SU2*(-7200.0/2197.0) + k3_SU2*(7296.0/2197.0), curr_spins.spins_SU3 , curr_time + step_size*(12/13))*step_size;
    //     spin_config_SU2 k5_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(439.0/216.0) + k2_SU2*(-8.0) + k3_SU2*(3680.0/513.0) + k4_SU2*(-845.0/4104.0), curr_spins.spins_SU3, curr_time + step_size)*step_size;
    //     spin_config_SU2 k6_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2 + k1_SU2*(-8.0/27.0)+ k2_SU2*(2.0) + k3_SU2*(-3544.0/2565.0)+ k4_SU2*(1859.0/4104.0)+ k5_SU2*(-11.0/40.0), curr_spins.spins_SU3 , curr_time + step_size/2)*step_size;

    //     spin_config_SU2 y_SU2 = curr_spins.spins_SU2 + k1_SU2*(25.0/216.0) + k3_SU2*(1408.0/2565.0) + k4_SU2*(2197.0/4101.0) - k5_SU2*(1.0/5.0);
    //     spin_config_SU2 z_SU2 = curr_spins.spins_SU2 + k1_SU2*(16.0/135.0) + k3_SU2*(6656.0/12825.0) + k4_SU2*(28561.0/56430.0) - k5_SU2*(9.0/50.0) + k6_SU2*(2.0/55.0);

    //     double error_SU2 = norm_average_2D(y_SU2-z_SU2);
        
    //     spin_config_SU3 k1_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time)*step_size;
    //     spin_config_SU3 k2_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3 + k1_SU3*(1.0/4.0), curr_time + step_size*(1/4))*step_size;
    //     spin_config_SU3 k3_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3 + k1_SU3*(3.0/32.0) + k2_SU3*(9.0/32.0), curr_time + step_size*(3/8))*step_size;
    //     spin_config_SU3 k4_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3 + k1_SU3*(1932.0/2197.0) + k2_SU3*(-7200.0/2197.0) + k3_SU3*(7296.0/2197.0), curr_time + step_size*(12/13))*step_size;
    //     spin_config_SU3 k5_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3 + k1_SU3*(439.0/216.0) + k2_SU3*(-8.0) + k3_SU3*(3680.0/513.0) + k4_SU3*(-845.0/4104.0), curr_time + step_size)*step_size;
    //     spin_config_SU3 k6_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3 + k1_SU3*(-8.0/27.0) + k2_SU3*(2.0) + k3_SU3*(-3544.0/2565.0)+ k4_SU3*(1859.0/4104.0)+ k5_SU3*(-11.0/40.0), curr_time + step_size/2)*step_size;

    //     spin_config_SU3 y_SU3 = curr_spins.spins_SU3 + k1_SU3*(25.0/216.0) + k3_SU3*(1408.0/2565.0) + k4_SU3*(2197.0/4101.0) - k5_SU3*(1.0/5.0);
    //     spin_config_SU3 z_SU3 = curr_spins.spins_SU3 + k1_SU3*(16.0/135.0) + k3_SU3*(6656.0/12825.0) + k4_SU3*(28561.0/56430.0) - k5_SU3*(9.0/50.0) + k6_SU3*(2.0/55.0);

    //     double error_SU3 = norm_average_2D(y_SU3-z_SU3);
    //     double error = max(error_SU2, error_SU3);
    //     step_size *= 0.9*pow(tol/error, 0.2);

    //     mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> z;
    //     z.spins_SU2 = z_SU2;
    //     z.spins_SU3 = z_SU3;
    //     if (error < tol){
    //         return z;
    //     }
    //     else{
    //         return RK45_step(step_size, curr_spins, curr_time, tol);
    //     }
    //     return z;
    // }
    
    mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> RK45_step(double &step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> &curr_spins, const double &curr_time, const double tol){
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

        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> z;
        z.spins_SU2 = z_SU2;
        z.spins_SU3 = z_SU3;
        if (error < tol){
            return z;
        }
        else{
            return RK45_step(step_size, curr_spins, curr_time, tol);
        }
        return z;
    }
    

    mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> RK45_step_fixed(double &step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> &curr_spins, const double &curr_time, const double tol){
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

        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> z;
        z.spins_SU2 = z_SU2;
        z.spins_SU3 = z_SU3;
        return z;
    }
    mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> euler_step(const double step_size, const mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> &curr_spins, const double &curr_time, const double tol){
        spin_config_SU2 dS_SU2 = landau_lifshitz_SU2(curr_spins.spins_SU2,curr_spins.spins_SU3, curr_time);
        spin_config_SU3 dS_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time);
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> new_spins;
        new_spins.spins_SU2 = curr_spins.spins_SU2 + dS_SU2*step_size;
        new_spins.spins_SU3 = curr_spins.spins_SU3 + dS_SU3*step_size;
        return new_spins;
    }

    // mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> SSPRK53_step(const double step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> &curr_spins, const double &curr_time, const double tol){
    //     double a30 = 0.355909775063327;
    //     double a32 = 0.644090224936674;
    //     double a40 = 0.367933791638137;
    //     double a43 = 0.632066208361863;
    //     double a52 = 0.237593836598569;
    //     double a54 = 0.762406163401431;
    //     double b10 = 0.377268915331368;
    //     double b21 = 0.377268915331368;
    //     double b32 = 0.242995220537396;
    //     double b43 = 0.238458932846290;
    //     double b54 =0.287632146308408;
    //     double c1 = 0.377268915331368;
    //     double c2 = 0.754537830662736;
    //     double c3 = 0.728985661612188;
    //     double c4 = 0.699226135931670;
    //     spin_config_SU2 tmp_SU2 = curr_spins.spins_SU2 + landau_lifshitz_SU2(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time) * b10 * step_size;
    //     spin_config_SU2 k_SU2 = landau_lifshitz_SU2(tmp_SU2, curr_spins.spins_SU3, curr_time + c1*step_size);
    //     spin_config_SU2 u_SU2 = tmp_SU2 + k_SU2 * step_size * b21;
    //     //u3
    //     k_SU2 = landau_lifshitz_SU2(u_SU2, curr_spins.spins_SU3, curr_time + c2*step_size);
    //     tmp_SU2 = curr_spins.spins_SU2 * a30 + u_SU2 * a32 + k_SU2 * step_size * b32;
    //     k_SU2 = landau_lifshitz_SU2(tmp_SU2, curr_spins.spins_SU3, curr_time + c3*step_size);
    //     //u4
    //     tmp_SU2 = curr_spins.spins_SU2 * a40 + tmp_SU2 * a43 + k_SU2 * step_size * b43;
    //     k_SU2 = landau_lifshitz_SU2(tmp_SU2, curr_spins.spins_SU3, curr_time + c4*step_size);
    //     u_SU2 = u_SU2 * a52 + tmp_SU2 * a54 + k_SU2 * step_size * b54;
    //     spin_config_SU3 tmp_SU3 = curr_spins.spins_SU3 + landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time) * b10 * step_size;
    //     spin_config_SU3 k_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, tmp_SU3, curr_time + c1*step_size);
    //     spin_config_SU3 u_SU3 = tmp_SU3 + k_SU3 * step_size * b21;
    //     //u3
    //     k_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, u_SU3, curr_time + c2*step_size);
    //     tmp_SU3 = curr_spins.spins_SU3 * a30 + u_SU3 * a32 + k_SU3 * step_size * b32;
    //     k_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, tmp_SU3, curr_time + c3*step_size);
    //     //u4
    //     tmp_SU3 = curr_spins.spins_SU3 * a40 + tmp_SU3 * a43 + k_SU3 * step_size * b43;
    //     k_SU3 = landau_lifshitz_SU3(curr_spins.spins_SU2, tmp_SU3, curr_time + c4*step_size);
    //     u_SU3 = u_SU3 * a52 + tmp_SU3 * a54 + k_SU3 * step_size * b54;
    //     mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> u;
    //     u.spins_SU2 = u_SU2;
    //     u.spins_SU3 = u_SU3;
    //     return u;
    // }

    mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> SSPRK53_step(const double step_size, mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> &curr_spins, const double &curr_time, const double tol){
        double a30 = 0.355909775063327;
        double a32 = 0.644090224936674;
        double a40 = 0.367933791638137;
        double a43 = 0.632066208361863;
        double a52 = 0.237593836598569;
        double a54 = 0.762406163401431;
        double b10 = 0.377268915331368;
        double b21 = 0.377268915331368;
        double b32 = 0.242995220537396;
        double b43 = 0.238458932846290;
        double b54 =0.287632146308408;
        double c1 = 0.377268915331368;
        double c2 = 0.754537830662736;
        double c3 = 0.728985661612188;
        double c4 = 0.699226135931670;


        spin_config_SU2 tmp_SU2 = curr_spins.spins_SU2 + landau_lifshitz_SU2(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time) * b10 * step_size;
        spin_config_SU3 tmp_SU3 = curr_spins.spins_SU3 + landau_lifshitz_SU3(curr_spins.spins_SU2, curr_spins.spins_SU3, curr_time) * b10 * step_size;

        spin_config_SU2 k_SU2 = landau_lifshitz_SU2(tmp_SU2, tmp_SU3, curr_time + c1*step_size);
        spin_config_SU3 k_SU3 = landau_lifshitz_SU3(tmp_SU2, tmp_SU3, curr_time + c1*step_size);

        spin_config_SU2 u_SU2 = tmp_SU2 + k_SU2 * step_size * b21;
        spin_config_SU3 u_SU3 = tmp_SU3 + k_SU3 * step_size * b21;
        //u3
        k_SU2 = landau_lifshitz_SU2(u_SU2, u_SU3, curr_time + c2*step_size);
        k_SU3 = landau_lifshitz_SU3(u_SU2, u_SU3, curr_time + c2*step_size);

        tmp_SU2 = curr_spins.spins_SU2 * a30 + u_SU2 * a32 + k_SU2 * step_size * b32;
        tmp_SU3 = curr_spins.spins_SU3 * a30 + u_SU3 * a32 + k_SU3 * step_size * b32;

        k_SU2 = landau_lifshitz_SU2(tmp_SU2, tmp_SU3, curr_time + c3*step_size);
        k_SU3 = landau_lifshitz_SU3(tmp_SU2, tmp_SU3, curr_time + c3*step_size);
        //u4
        tmp_SU2 = curr_spins.spins_SU2 * a40 + tmp_SU2 * a43 + k_SU2 * step_size * b43;
        tmp_SU3 = curr_spins.spins_SU3 * a40 + tmp_SU3 * a43 + k_SU3 * step_size * b43;

        k_SU2 = landau_lifshitz_SU2(tmp_SU2, tmp_SU3, curr_time + c4*step_size);
        k_SU3 = landau_lifshitz_SU3(tmp_SU2, tmp_SU3, curr_time + c4*step_size);

        u_SU2 = u_SU2 * a52 + tmp_SU2 * a54 + k_SU2 * step_size * b54;
        u_SU3 = u_SU3 * a52 + tmp_SU3 * a54 + k_SU3 * step_size * b54;

        //u3
        //u4

        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> u;
        u.spins_SU2 = u_SU2;
        u.spins_SU3 = u_SU3;
        return u;
    }


    void molecular_dynamics(double T_start, double T_end, double step_size, string dir_name){
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        write_to_file_pos(dir_name + "/pos");
        write_to_file_spin(dir_name + "/spin");
        write_to_file_spin(dir_name + "/spin_t");
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> spin_t(spins);

        double tol = 1e-6;

        int check_frequency = 10;
        double currT = T_start;
        size_t count = 1;
        vector<double> time;

        time.push_back(currT);

        while(currT < T_end){
            spin_t.set(SSPRK53_step(step_size, spin_t, currT, tol));
            write_to_file(dir_name + "/spin_t", spin_t);
            currT = currT + step_size;
            cout << "Time: " << currT << endl;
            time.push_back(currT);
            count++;
        }

        ofstream time_sections;
        time_sections.open(dir_name + "/Time_steps.txt");
        for(size_t i = 0; i<count; ++i){
            time_sections << time[i] << endl;
        }
        time_sections.close();
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

    array<double,N_SU2>  magnetization_local(mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> &spin_t){
        array<double,N_SU2> mag = {{0}};
        for (size_t i=0; i< lattice_size_SU2; ++i){
            mag = mag + spin_t.spins_SU2[i];
        }
        return mag/double(lattice_size_SU2);
    }


    
    void M_B_t(array<array<double,N_SU2>, N_ATOMS_SU2> &field_in, double t_B, double pulse_amp, double pulse_width, double pulse_freq, double T_start, double T_end, double step_size, string dir_name){
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> spin_t(spins);
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        double tol = 1e-6;

        double currT = T_start;
        size_t count = 1;
        vector<double> time;
        time.push_back(currT);
        write_to_file_magnetization_local_SU2(dir_name + "/M_t.txt", magnetization_local(spin_t));

        set_pulse_SU2(field_in, t_B, {{0}}, 0, pulse_amp, pulse_width, pulse_freq);
        while(currT < T_end){
            // double factor = double(pulse_amp*exp(-pow((currT+t_B)/(2*pulse_width),2))*cos(2*M_PI*pulse_freq*(currT+t_B)));
            // pulse_info << "Current Time: " << currT << " Pulse Time: " << t_B << " Factor: " << factor << " Field: " endl;
            spin_t.set(SSPRK53_step(step_size, spin_t, currT, tol));
            write_to_file_magnetization_local_SU2(dir_name + "/M_t.txt", magnetization_local(spin_t));
            // write_to_file(dir_name + "/spin_t.txt", spin_t);
            currT = currT + step_size;
            time.push_back(currT);
            count++;
        }
        reset_pulse();
        // pulse_info.close();dfsbf
        ofstream time_sections;
        time_sections.open(dir_name + "/Time_steps.txt");
        for(size_t i = 0; i<count; ++i){
            time_sections << time[i] << endl;
        }
        time_sections.close();      
    };

    void M_BA_BB_t(array<array<double,N_SU2>, N_ATOMS_SU2> &field_in_1, double t_B_1, array<array<double,N_SU2>, N_ATOMS_SU2> &field_in_2, double t_B_2, double pulse_amp, double pulse_width, double pulse_freq, double T_start, double T_end, double step_size, string dir_name){
        // simulated_annealing(Temp_start, Temp_end, n_anneal, overrelaxation_rate);
        // write_to_file_pos(dir_name + "/pos.txt");
        // write_to_file_spin(dir_name + "/spin_t.txt", spins);
        mixed_lattice_spin<N_SU2, N_ATOMS_SU2*dim1*dim2*dim3, N_SU3, N_ATOMS_SU3*dim1*dim2*dim3> spin_t(spins);
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        double tol = 1e-6;
        double currT = T_start;
        size_t count = 1;
        vector<double> time;

        time.push_back(currT);
        write_to_file_magnetization_local_SU2(dir_name + "/M_t.txt", magnetization_local(spin_t));
        set_pulse_SU2(field_in_1, t_B_1, field_in_2, t_B_2, pulse_amp, pulse_width, pulse_freq);
        while(currT < T_end){
            spin_t.set(SSPRK53_step(step_size, spin_t, currT, tol));
            write_to_file_magnetization_local_SU2(dir_name + "/M_t.txt", magnetization_local(spin_t));
            currT = currT + step_size;
            time.push_back(currT);
            count++;
        }  
        reset_pulse();
        ofstream time_sections;
        time_sections.open(dir_name + "/Time_steps.txt");
        for(size_t i = 0; i<count; ++i){
            time_sections << time[i] << endl;
        }
        time_sections.close();     
    }
};
#endif // MIXED_LATTICE_H