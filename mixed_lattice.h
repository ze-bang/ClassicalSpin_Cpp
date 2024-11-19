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

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3, size_t dim1, size_t dim2, size_t dim3>
class mixed_lattice
{   
    public:

    mixed_UnitCell<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3> UC;
    size_t lattice_size_SU2;
    size_t lattice_size_SU3;

    //Spin and lattice position of SU2 and SU3
    // array<array<float,N_SU2>, N_ATOMS_SU2*dim1*dim2*dim3>  spins_SU2;
    // array<array<float,3>, N_ATOMS_SU2*dim1*dim2*dim3> get<0>(site_pos);
    // array<array<float,N_SU3>, N_ATOMS_SU3*dim1*dim2*dim3>  spins_SU3;
    // array<array<float,3>, N_ATOMS_SU3*dim1*dim2*dim3> get<1>(site_pos);

    typedef tuple<array<array<float,N_SU2>, N_ATOMS_SU2*dim1*dim2*dim3>, array<array<float,N_SU3>, N_ATOMS_SU3*dim1*dim2*dim3>> mixed_lattice_spin;
    typedef tuple<array<array<float,3>, N_ATOMS_SU2*dim1*dim2*dim3>, array<array<float,3>, N_ATOMS_SU3*dim1*dim2*dim3>> mixed_lattice_pos;

    mixed_lattice_spin spins;
    mixed_lattice_pos site_pos;

    //Look up table for SU2
    array<array<float,N_SU2>, N_ATOMS_SU2*dim1*dim2*dim3> field_SU2;
    array<vector<array<array<float, N_SU2>, N_SU2>>, N_ATOMS_SU2*dim1*dim2*dim3> bilinear_interaction_SU2;
    array<vector<array<array<array<float, N_SU2>, N_SU2>, N_SU2>>, N_ATOMS_SU2*dim1*dim2*dim3> trilinear_interaction_SU2;

    array<vector<size_t>, N_ATOMS_SU2*dim1*dim2*dim3> bilinear_partners_SU2;
    array<vector<array<size_t, 2>>, N_ATOMS_SU2*dim1*dim2*dim3> trilinear_partners_SU2;

    //Look up table for SU3
    array<array<float,N_SU3>, N_ATOMS_SU3*dim1*dim2*dim3> field_SU3;
    array<vector<array<array<float, N_SU3>, N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim3> bilinear_interaction_SU3;
    array<vector<array<array<array<float, N_SU3>, N_SU3>, N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim3> trilinear_interaction_SU3;

    array<vector<size_t>, N_ATOMS_SU3*dim1*dim2*dim3> bilinear_partners_SU3;
    array<vector<array<size_t, 2>>, N_ATOMS_SU3*dim1*dim2*dim3> trilinear_partners_SU3;

    //Look up table for SU2 and SU3 mix
    array<vector<array<array<array<float, N_SU3>, N_SU2>, N_SU2>>, N_ATOMS_SU2*dim1*dim2*dim3> mixed_trilinear_interaction_SU2;
    array<vector<array<array<array<float, N_SU2>, N_SU2>, N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim3> mixed_trilinear_interaction_SU3;

    array<vector<array<size_t, 2>>, N_ATOMS_SU2*dim1*dim2*dim3> mixed_trilinear_partners_SU2;
    array<vector<array<size_t, 2>>, N_ATOMS_SU3*dim1*dim2*dim3> mixed_trilinear_partners_SU3;

    size_t num_bi_SU2;
    size_t num_tri_SU2;
    size_t num_bi_SU3;
    size_t num_tri_SU3;
    size_t num_tri_SU2_SU3;


    template<size_t N>
    void gen_random_spin(std::mt19937 &gen, array<float,N> &temp_spin){
        array<float,N-2> euler_angles;
        float z = random_float(-1,1, gen);
        float r = sqrt(1.0 - z*z);

        for(size_t i = 0; i < N-2; ++i){
            euler_angles[i] = random_float(0, 2*M_PI, gen);
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
    void set_up_sublattice(array<array<float,N>, lattice_size>  &spins, array<array<float,3>, lattice_size> &site_pos, array<array<float,N>, lattice_size> &field, array<vector<array<array<float, N>, N>>, lattice_size> &bilinear_interaction, array<vector<array<array<array<float, N>, N>, N>>, lattice_size> &trilinear_interaction, array<vector<size_t>, lattice_size> &bilinear_partners, array<vector<array<size_t, 2>>, lattice_size> &trilinear_partners, UnitCell<N, N_ATOMS> *atoms, size_t &num_bi, size_t &num_tri){
        std::random_device rd;
        std::mt19937 gen(rd());

        array<array<float,3>, N_ATOMS> basis = atoms->lattice_pos;
        array<array<float,3>, 3> unit_vector = atoms->lattice_vectors;

        for (size_t i=0; i<dim1; ++i){
            for (size_t j=0; j< dim2; ++j){
                for(size_t k=0; k<dim3;++k){
                    for (size_t l=0; l<N_ATOMS;++l){
                        size_t current_site_index = flatten_index(i,j,k,l,N_ATOMS);

                        site_pos[current_site_index]  = unit_vector[0]*int(i) + unit_vector[1]*int(j) + unit_vector[2]*int(k) + basis[l];

                        gen_random_spin(gen, spins[current_site_index]);
                        field[current_site_index] = atoms->field[l];
                        
                        auto bilinear_matched = atoms->bilinear_interaction.equal_range(l);

                        size_t count = 0;
                        for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m){
                            bilinear<N> J = m->second;
                            size_t partner = flatten_index_periodic_boundary(int(i)+J.offset[0], int(j)+J.offset[1], int(k)+J.offset[2], J.partner,N_ATOMS);
                            bilinear_interaction[current_site_index].push_back(J.bilinear_interaction);
                            bilinear_partners[current_site_index].push_back(partner);
                            bilinear_interaction[partner].push_back(J.bilinear_interaction);
                            bilinear_partners[partner].push_back(current_site_index);
                            count++;
                        }

                        auto trilinear_matched = atoms->trilinear_interaction.equal_range(l);
                        count = 0;
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
                            count++;
                        }
                    }
                }
            }
        }

        num_bi = bilinear_partners[0].size();
        num_tri = trilinear_partners[0].size();
    }
    

    mixed_lattice(mixed_UnitCell<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3> *atoms): UC(*atoms){

        std::random_device rd;
        std::mt19937 gen(rd());

        set_up_sublattice(get<0>(spins), get<0>(site_pos), field_SU2, bilinear_interaction_SU2, trilinear_interaction_SU2, bilinear_partners_SU2, trilinear_partners_SU2, &(atoms->SU2), num_bi_SU2, num_tri_SU2);
        set_up_sublattice(get<1>(spins), get<1>(site_pos), field_SU3, bilinear_interaction_SU3, trilinear_interaction_SU3, bilinear_partners_SU3, trilinear_partners_SU3, &(atoms->SU3), num_tri_SU2, num_tri_SU3);
        
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
    };

    float site_energy_SU2(array<float, N_SU2> &spin_here, size_t site_index){
        float energy = 0.0;
        energy += dot(spin_here, field_SU2[site_index]);
        #pragma omp simd
        for (size_t i=0; i<num_bi_SU2; ++i) {
            energy += contract(spin_here, bilinear_interaction_SU2[site_index][i], get<0>(spins)[bilinear_partners_SU2[site_index][i]]);
        }
        for (size_t i=0; i < num_tri_SU3; ++i){
            energy += contract_trilinear(trilinear_interaction_SU2[site_index][i], spin_here, get<0>(spins)[trilinear_partners_SU2[site_index][i][0]], get<0>(spins)[trilinear_partners_SU2[site_index][i][1]]);
        }
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            energy += contract_trilinear(mixed_trilinear_interaction_SU2[site_index][i], spin_here, get<0>(spins)[mixed_trilinear_partners_SU2[site_index][i][0]], get<1>(spins)[mixed_trilinear_partners_SU2[site_index][i][1]]);
        }
        return energy;
    }

    float site_energy_SU3(array<float, N_SU3> &spin_here, size_t site_index){
        float energy = 0.0;
        energy += dot(spin_here, field_SU3[site_index]);
        #pragma omp simd
        for (size_t i=0; i<num_bi_SU3; ++i) {
            energy += contract(spin_here, bilinear_interaction_SU3[site_index][i], get<1>(spins)[bilinear_partners_SU3[site_index][i]]);
        }
        for (size_t i=0; i < num_tri_SU3; ++i){
            energy += contract_trilinear(trilinear_interaction_SU3[site_index][i], spin_here, get<1>(spins)[trilinear_partners_SU3[site_index][i][0]], get<1>(spins)[trilinear_partners_SU3[site_index][i][1]]);
        }
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            energy += contract_trilinear(mixed_trilinear_interaction_SU3[site_index][i], spin_here, get<0>(spins)[mixed_trilinear_partners_SU3[site_index][i][0]], get<0>(spins)[mixed_trilinear_partners_SU3[site_index][i][1]]);
        }
        return energy;
    }

    template<size_t N>
    float site_energy(array<float,N> &spins, size_t site_index){
        if constexpr (N == N_SU2){
            return site_energy_SU2(spins, site_index);
        }else{
            return site_energy_SU3(spins, site_index);
        }
    }
    
    array<float, N_SU2>  get_local_field_SU2(size_t site_index){
        array<float,N_SU2> local_field = {0};
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU2; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU2[site_index][i], get<0>(spins)[bilinear_partners_SU2[site_index][i]]);
        }
        for (size_t i=0; i < num_tri_SU2; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction_SU2[site_index][i], get<0>(spins)[trilinear_partners_SU2[site_index][i][0]], get<0>(spins)[trilinear_partners_SU2[site_index][i][1]]);
        }
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            local_field = local_field + contract_trilinear_field(mixed_trilinear_interaction_SU2[site_index][i], get<0>(spins)[mixed_trilinear_partners_SU2[site_index][i][0]], get<1>(spins)[mixed_trilinear_partners_SU2[site_index][i][1]]);
        }
        return local_field+field_SU2[site_index];
    }

    array<float, N_SU3>  get_local_field_SU3(size_t site_index){
        array<float,N_SU3> local_field = {0};
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU3; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU3[site_index][i], get<0>(spins)[bilinear_partners_SU3[site_index][i]]);
        }
        for (size_t i=0; i < num_tri_SU3; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction_SU3[site_index][i], get<1>(spins)[trilinear_partners_SU3[site_index][i][0]], get<1>(spins)[trilinear_partners_SU3[site_index][i][1]]);
        }
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            local_field = local_field + contract_trilinear_field(mixed_trilinear_interaction_SU3[site_index][i], get<0>(spins)[mixed_trilinear_partners_SU3[site_index][i][0]], get<0>(spins)[mixed_trilinear_partners_SU3[site_index][i][1]]);
        }
        return local_field+field_SU3[site_index];
    }

    array<float, N_SU2>  get_local_field_SU2_lattice(size_t site_index, mixed_lattice_spin &current_spin){
        array<float,N_SU2> local_field = {0};
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU2; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU2[site_index][i], get<0>(current_spin)[bilinear_partners_SU2[site_index][i]]);
        }
        for (size_t i=0; i < num_tri_SU2; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction_SU2[site_index][i], get<0>(current_spin)[trilinear_partners_SU2[site_index][i][0]], get<0>(current_spin)[trilinear_partners_SU2[site_index][i][1]]);
        }
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            local_field = local_field + contract_trilinear_field(mixed_trilinear_interaction_SU2[site_index][i], get<0>(current_spin)[mixed_trilinear_partners_SU2[site_index][i][0]], get<1>(current_spin)[mixed_trilinear_partners_SU2[site_index][i][1]]);
        }
        return local_field+field_SU2[site_index];
    }

    array<float, N_SU3>  get_local_field_SU3_lattice(size_t site_index, mixed_lattice_spin &current_spin){
        array<float,N_SU3> local_field = {0};
        #pragma omp simd
        for (size_t i=0; i< num_bi_SU3; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU3[site_index][i], get<0>(current_spin)[bilinear_partners_SU3[site_index][i]]);
        }
        for (size_t i=0; i < num_tri_SU3; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction_SU3[site_index][i], get<1>(current_spin)[trilinear_partners_SU3[site_index][i][0]], get<1>(current_spin)[trilinear_partners_SU3[site_index][i][1]]);
        }
        for (size_t i=0; i < num_tri_SU2_SU3; ++i){
            local_field = local_field + contract_trilinear_field(mixed_trilinear_interaction_SU3[site_index][i], get<0>(current_spin)[mixed_trilinear_partners_SU3[site_index][i][0]], get<0>(current_spin)[mixed_trilinear_partners_SU3[site_index][i][1]]);
        }
        return local_field+field_SU3[site_index];
    }

    
    float metropolis(float T, std::mt19937 &gen){
        float E, E_new, dE, r, i;
        int accept = 0;
        size_t count = 0;
        while(count < lattice_size_SU2+lattice_size_SU3){
            int SU2_or_SU3 = random_int(0,1, gen);
            size_t lattice_size;
            if (SU2_or_SU3 == 0){
                lattice_size = lattice_size_SU2;
            }else{
                lattice_size = lattice_size_SU3;
            }
            i = random_int(0, lattice_size-1, gen);
            E = site_energy(spins[SU2_or_SU3][i], i);

            constexpr auto new_spin = gen_random_spin(gen, spins[SU2_or_SU3][i]);

            E_new = site_energy(new_spin, i);
            dE = E_new - E;
            
            if(dE < 0){
                spins[SU2_or_SU3][i] = new_spin;
                accept++;
            }
            else{
                r = random_float(0,1, gen);
                if(r < exp(-dE/T)){
                    get<0>(spins)[i] = new_spin;
                    accept++;
                }
            }
            count++;
        }

        float acceptance_rate = float(accept)/float(lattice_size_SU2+lattice_size_SU3);
        return acceptance_rate;
    }

    
    void deterministic_sweep(){
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            array<float,N_SU2> local_field = get_local_field_SU2(i);
            float norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                get<0>(spins)[i] = -local_field/norm;
            }
        }
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            array<float,N_SU3> local_field = get_local_field_SU3(i);
            float norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                get<1>(spins)[i] = -local_field/norm;
            }
        }
    }
    
    void write_to_file_spin(string filename){
        ofstream myfile;
        myfile.open(filename);
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            for(size_t j = 0; j<3; ++j){
                myfile << get<0>(spins)[i][j] << " ";
            }
            myfile << endl;
        }
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            for(size_t j = 0; j<8; ++j){
                myfile << get<1>(spins)[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_to_file_pos(string filename){
        ofstream myfile;
        myfile.open(filename);
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            for(size_t j = 0; j<3; ++j){
                myfile << get<0>(site_pos)[i][j] << " ";
            }
            myfile << endl;
        }
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            for(size_t j = 0; j<3; ++j){
                myfile << get<1>(site_pos)[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_to_file(string filename, mixed_lattice_spin towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            for(size_t j = 0; j<3; ++j){
                myfile << get<0>(spins)[i][j] << " ";
            }
            myfile << endl;
        }
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            for(size_t j = 0; j<8; ++j){
                myfile << get<1>(spins)[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }


    void simulated_annealing(float T_start, float T_end, size_t n_therm, size_t n_anneal, size_t n_deterministics, size_t overrelaxation_rate, string dir_name){
        std::random_device rd;
        std::mt19937 gen(rd());
        float T = T_start;
        while(T > T_end){
            float curr_accept = 0;
            for(size_t i = 0; i<n_anneal; ++i){
                curr_accept += metropolis(T, gen);
            }
            cout << "Temperature: " << T << " Acceptance rate: " << curr_accept/n_anneal << endl;            
            T *= 0.9;
            if(dir_name != ""){
                filesystem::create_directory(dir_name);
                write_to_file_spin(dir_name + "/spin.txt");
                write_to_file_pos(dir_name + "/pos.txt");
            }
        }

        for(size_t i = 0; i < n_deterministics; ++i){
            deterministic_sweep();
        }   

        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            write_to_file_spin(dir_name + "/spin.txt");
            write_to_file_pos(dir_name + "/pos.txt");
        }
    }


    mixed_lattice_spin landau_lifshitz(mixed_lattice_spin &current_spin){
        mixed_lattice_spin dS;
        #pragma omp simd
        for(size_t i = 0; i<lattice_size_SU2; ++i){
            get<0>(dS)[i] = cross_prod_SU2(get_local_field_SU2_lattice(i, current_spin), get<0>(current_spin)[i]);
        }
        #pragma omp simd
        for(size_t i = 0; i<lattice_size_SU3; ++i){
            get<1>(dS)[i] = cross_prod_SU3(get_local_field_SU3_lattice(i, current_spin), get<1>(current_spin)[i]);
        }
        return dS;
    }

    mixed_lattice_spin RK45_step(float &step_size, const mixed_lattice_spin &curr_spins, const double tol){
        mixed_lattice_spin k1 = landau_lifshitz(curr_spins)*step_size;
        mixed_lattice_spin k2 = landau_lifshitz(curr_spins + k1*(1.0/4.0))*step_size;
        mixed_lattice_spin k3 = landau_lifshitz(curr_spins + k1*(3.0/32.0) + k2*(9.0/32.0))*step_size;
        mixed_lattice_spin k4 = landau_lifshitz(curr_spins + k1*(1932.0/2197.0) + k2*(-7200.0/2197.0) + k3*(7296.0/2197.0))*step_size;
        mixed_lattice_spin k5 = landau_lifshitz(curr_spins + k1*(439.0/216.0) + k2*(-8.0) + k3*(3680.0/513.0) + k4*(-845.0/4104.0))*step_size;
        mixed_lattice_spin k6 = landau_lifshitz(curr_spins + k1*(-8.0/27.0) + k2*(2.0) + k3*(-3544.0/2565.0)+ k4*(1859.0/4104.0)+ k5*(-11.0/40.0))*step_size;

        mixed_lattice_spin y = curr_spins + k1*(25.0/216.0) + k3*(1408.0/2565.0) + k4*(2197.0/4101.0) - k5*(1.0/5.0);
        mixed_lattice_spin z = curr_spins + k1*(16.0/135.0) + k3*(6656.0/12825.0) + k4*(28561.0/56430.0) - k5*(9.0/50.0) + k6*(2.0/55.0);

        double error = norm_average_2D(z-y);
        step_size *= 0.9*pow(tol/error, 0.2);
        cout << "Step size: " << step_size << " Error: " << error << endl;
        if (error < tol){
            return z;
        }
        else{
            return RK45_step(step_size, curr_spins, tol);
        }
        return z;
    }


    void molecular_dynamics(float Temp_start, float Temp_end, size_t n_therm, size_t n_anneal, size_t overrelaxation_rate, float T_end, float step_size, string dir_name){
        std::random_device rd;
        std::mt19937 gen(rd());
        float curr_accept;
        filesystem::create_directory(dir_name);
        float T = Temp_start;

        while(T > Temp_end){
            curr_accept = 0;
            for(size_t i = 0; i<n_anneal; ++i){
                if(overrelaxation_rate > 0){
                    // overrelaxation();
                    if (i%overrelaxation_rate == 0){
                        curr_accept += metropolis(T, gen);
                    }
                }
                else{
                    curr_accept += metropolis(T, gen);
                }
            }
            if (overrelaxation_rate > 0){
                cout << "Temperature: " << T << " Acceptance rate: " << curr_accept/n_anneal*overrelaxation_rate << endl;
            }else{
                cout << "Temperature: " << T << " Acceptance rate: " << curr_accept/n_anneal << endl;
            }
            T *= 0.9;
        }
        write_to_file_pos(dir_name + "/pos.txt");
        write_to_file_spin(dir_name + "/spin_t.txt", spins);
        mixed_lattice_spin spin_t = spins;

        double tol = 1e-8;

        int check_frequency = 10;
        float currT = 0;
        size_t count = 1;
        vector<float> time;

        time.push_back(currT);

        while(currT < T_end){
            spin_t = RK45_step(step_size, spin_t, tol);
            write_to_file(dir_name + "/spin_t.txt", spin_t);
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

};
#endif // MIXED_LATTICE_H