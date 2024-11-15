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
    int lattice_size_SU2;
    int lattice_size_SU3;

    //Spin and lattice position of SU2 and SU3
    array<array<float,N_SU2>, N_ATOMS_SU2*dim1*dim2*dim3>  spins_SU2;
    array<array<float,3>, N_ATOMS_SU2*dim1*dim2*dim3> site_pos_SU2;
    array<array<float,N_SU3>, N_ATOMS_SU3*dim1*dim2*dim3>  spins_SU3;
    array<array<float,3>, N_ATOMS_SU3*dim1*dim2*dim3> site_pos_SU3;

    //Look up table for SU2
    array<array<float,N_SU2>, N_ATOMS_SU2*dim1*dim2*dim3> field_SU2;
    array<vector<array<array<float, N_SU2>, N_SU2>>, N_ATOMS_SU2*dim1*dim2*dim3> bilinear_interaction_SU2;
    array<vector<array<array<array<float, N_SU2>, N_SU2>, N_SU2>>, N_ATOMS_SU2*dim1*dim2*dim3> trilinear_interaction_SU2;

    array<vector<int>, N_ATOMS_SU2*dim1*dim2*dim3> bilinear_partners_SU2;
    array<vector<array<int, 2>>, N_ATOMS_SU2*dim1*dim2*dim3> trilinear_partners_SU2;

    //Look up table for SU3
    array<array<float,N_SU3>, N_ATOMS_SU3*dim1*dim2*dim3> field_SU3;
    array<vector<array<array<float, N_SU3>, N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim3> bilinear_interaction_SU3;
    array<vector<array<array<array<float, N_SU3>, N_SU3>, N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim3> trilinear_interaction_SU3;

    array<vector<int>, N_ATOMS_SU3*dim1*dim2*dim3> bilinear_partners_SU3;
    array<vector<array<int, 2>>, N_ATOMS_SU3*dim1*dim2*dim3> trilinear_partners_SU3;

    //Look up table for SU2 and SU3 mix
    array<vector<array<array<array<float, N_SU3>, N_SU2>, N_SU2>>, N_ATOMS_SU2*dim1*dim2*dim3> mixed_trilinear_interaction_SU2;
    array<vector<array<array<array<float, N_SU2>, N_SU2>, N_SU3>>, N_ATOMS_SU3*dim1*dim2*dim3> mixed_trilinear_interaction_SU3;

    array<vector<array<int, 2>>, N_ATOMS_SU2*dim1*dim2*dim3> mixed_trilinear_partners_SU2;
    array<vector<array<int, 2>>, N_ATOMS_SU3*dim1*dim2*dim3> mixed_trilinear_partners_SU3;


    int num_bi_SU2;
    int num_tri_SU2;
    int num_bi_SU3;
    int num_tri_SU3;
    int num_tri_SU2_SU3;


    template<size_t N>
    void gen_random_spin(std::mt19937 &gen, array<float,N> &temp_spin){
        array<float,N-2> euler_angles;
        float z = random_float(-1,1, gen);
        float r = sqrt(1.0 - z*z);

        for(int i = 0; i < N-2; ++i){
            euler_angles[i] = random_float(0, 2*M_PI, gen);
            temp_spin[i] = r;
            for(int j = 0; j < i; ++j){
                temp_spin[i] *= sin(euler_angles[j]);
            }
            if (i == N-3){
                temp_spin[i+1] = temp_spin[i]*sin(euler_angles[i]);
            }
            temp_spin[i] *= cos(euler_angles[i]);

        }
        temp_spin[N-1] = z;
    }

    int flatten_index(int i, int j, int k, int l, int N_ATOMS){
        return i*dim2*dim3*N_ATOMS+ j*dim3*N_ATOMS+ k*N_ATOMS + l;
    }
    
    int periodic_boundary(int i, int dim){
        if(i < 0){
            return (dim+i) % dim;
        }
        else{
            return i % dim;
        }
    }

    int flatten_index_periodic_boundary(int i, int j, int k, int l, int N_ATOMS){
        return periodic_boundary(i, dim1)*dim2*dim3*N_ATOMS+ periodic_boundary(j, dim2)*dim3*N_ATOMS+ periodic_boundary(k, dim3)*N_ATOMS + l;
    }

    template<size_t N, size_t lattice_size, size_t N_ATOMS>
    void set_up_sublattice(array<array<float,N>, lattice_size>  &spins, array<array<float,3>, lattice_size> &site_pos, array<array<float,N>, lattice_size> &field, array<vector<array<array<float, N>, N>>, lattice_size> &bilinear_interaction, array<vector<array<array<array<float, N>, N>, N>>, lattice_size> &trilinear_interaction, array<vector<int>, lattice_size> &bilinear_partners, array<vector<array<int, 2>>, lattice_size> &trilinear_partners, UnitCell<N, N_ATOMS> *atoms, int &num_bi, int &num_tri){
        std::random_device rd;
        std::mt19937 gen(rd());

        array<array<float,3>, N_ATOMS> basis = atoms->lattice_pos;
        array<array<float,3>, 3> unit_vector = atoms->lattice_vectors;

        for (int i=0; i<dim1; ++i){
            for (int j=0; j< dim2; ++j){
                for(int k=0; k<dim3;++k){
                    for (int l=0; l<N_ATOMS;++l){
                        int current_site_index = flatten_index(i,j,k,l,N_ATOMS);

                        site_pos[current_site_index]  = unit_vector[0]*i + unit_vector[1]*j + unit_vector[2]*k + basis[l];

                        gen_random_spin(gen, spins[current_site_index]);
                        field[current_site_index] = atoms->field[l];
                        
                        auto bilinear_matched = atoms->bilinear_interaction.equal_range(l);

                        int count = 0;
                        for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m){
                            bilinear<N> J = m->second;
                            int partner = flatten_index_periodic_boundary(i+J.offset[0], j+J.offset[1], k+J.offset[2], J.partner,N_ATOMS);
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
                            int partner1 = flatten_index_periodic_boundary(i+J.offset1[0], j+J.offset1[1], k+J.offset1[2], J.partner1,N_ATOMS);
                            int partner2 = flatten_index_periodic_boundary(i+J.offset2[0], j+J.offset2[1], k+J.offset2[2], J.partner2,N_ATOMS);
                            
                            trilinear_interaction[current_site_index].push_back(J.trilinear_interaction);
                            trilinear_partners[current_site_index].push_back({partner1, partner2});

                            trilinear_interaction[partner1].push_back(transpose3D(J.trilinear_interaction));
                            trilinear_partners[partner1].push_back({partner2, current_site_index});

                            trilinear_interaction[partner2].push_back(transpose3D(J.trilinear_interaction));
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

        set_up_sublattice(spins_SU2, site_pos_SU2, field_SU2, bilinear_interaction_SU2, trilinear_interaction_SU2, bilinear_partners_SU2, trilinear_partners_SU2, &(atoms->SU2), num_bi_SU2, num_tri_SU2);
        set_up_sublattice(spins_SU3, site_pos_SU3, field_SU3, bilinear_interaction_SU3, trilinear_interaction_SU3, bilinear_partners_SU3, trilinear_partners_SU3, &(atoms->SU3), num_tri_SU2, num_tri_SU3);
        
        for (int i=0; i < dim1; ++i){
            for (int j=0; j < dim2; ++j){
                for(int k=0; k < dim3;++k){
                    for (int l=0; l < N_ATOMS_SU3;++l){
                        int current_site_index = flatten_index(i,j,k,l,N_ATOMS_SU3);

                        int count = 0;
                        
                        auto trilinear_matched = atoms->trilinear_SU2_SU3.equal_range(l);
                        for (auto m = trilinear_matched.first; m != trilinear_matched.second; ++m){
                            mixed_trilinear<N_SU2, N_SU3> J = m->second;
                            int partner1 = flatten_index_periodic_boundary(i+J.offset1[0], j+J.offset1[1], k+J.offset1[2], J.partner1, N_ATOMS_SU2);
                            int partner2 = flatten_index_periodic_boundary(i+J.offset2[0], j+J.offset2[1], k+J.offset2[2], J.partner2, N_ATOMS_SU2);
                            
                            mixed_trilinear_interaction_SU3[current_site_index].push_back(J.trilinear_interaction);
                            mixed_trilinear_partners_SU3[current_site_index].push_back({partner1, partner2});

                            mixed_trilinear_interaction_SU2[partner1].push_back(transpose3D(J.trilinear_interaction));
                            mixed_trilinear_partners_SU2[partner1].push_back({partner2, current_site_index});

                            mixed_trilinear_interaction_SU2[partner2].push_back(swap_axis_3D(transpose3D(J.trilinear_interaction)));
                            mixed_trilinear_partners_SU2[partner2].push_back({partner1, current_site_index});
                            count++;
                        }
                    }
                }
            }
        }

        num_tri_SU2_SU3 = mixed_trilinear_partners_SU3[0].size();
        lattice_size_SU2 = dim1*dim2*dim3*N_ATOMS_SU2;
        lattice_size_SU3 = dim1*dim2*dim3*N_ATOMS_SU3;
    };

    float site_energy_SU2(array<float, N_SU2> &spin_here, int site_index){
        float energy = 0.0;
        energy += dot(spin_here, field_SU2[site_index]);
        #pragma omp simd
        for (int i=0; i<num_bi_SU2; ++i) {
            energy += contract(spin_here, bilinear_interaction_SU2[site_index][i], spins_SU2[bilinear_partners_SU2[site_index][i]]);
        }
        for (int i=0; i < num_tri_SU3; ++i){
            energy += contract_trilinear(trilinear_interaction_SU2[site_index][i], spin_here, spins_SU2[trilinear_partners_SU2[site_index][i][0]], spins_SU2[trilinear_partners_SU2[site_index][i][1]]);
        }
        for (int i=0; i < num_tri_SU2_SU3; ++i){
            energy += contract_trilinear(mixed_trilinear_interaction_SU2[site_index][i], spin_here, spins_SU2[mixed_trilinear_partners_SU2[site_index][i][0]], spins_SU3[mixed_trilinear_partners_SU2[site_index][i][1]]);
        }
        return energy;
    }

    float site_energy_SU3(array<float, N_SU3> &spin_here, int site_index){
        float energy = 0.0;
        energy += dot(spin_here, field_SU3[site_index]);
        #pragma omp simd
        for (int i=0; i<num_bi_SU3; ++i) {
            energy += contract(spin_here, bilinear_interaction_SU3[site_index][i], spins_SU3[bilinear_partners_SU3[site_index][i]]);
        }
        for (int i=0; i < num_tri_SU3; ++i){
            energy += contract_trilinear(trilinear_interaction_SU3[site_index][i], spin_here, spins_SU3[trilinear_partners_SU3[site_index][i][0]], spins_SU3[trilinear_partners_SU3[site_index][i][1]]);
        }
        for (int i=0; i < num_tri_SU2_SU3; ++i){
            energy += contract_trilinear(mixed_trilinear_interaction_SU3[site_index][i], spin_here, spins_SU2[mixed_trilinear_partners_SU3[site_index][i][0]], spins_SU2[mixed_trilinear_partners_SU3[site_index][i][1]]);
        }
        return energy;
    }
    
    array<float, N_SU2>  get_local_field_SU2(int site_index){
        array<float,N_SU2> local_field = {0};
        #pragma omp simd
        for (int i=0; i< num_bi_SU2; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU2[site_index][i], spins_SU2[bilinear_partners_SU2[site_index][i]]);
        }
        for (int i=0; i < num_tri_SU2; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction_SU2[site_index][i], spins_SU2[trilinear_partners_SU2[site_index][i][0]], spins_SU2[trilinear_partners_SU2[site_index][i][1]]);
        }
        for (int i=0; i < num_tri_SU2_SU3; ++i){
            local_field = local_field + contract_trilinear_field(mixed_trilinear_interaction_SU2[site_index][i], spins_SU2[mixed_trilinear_partners_SU2[site_index][i][0]], spins_SU3[mixed_trilinear_partners_SU2[site_index][i][1]]);
        }
        return local_field+field_SU2[site_index];
    }

    array<float, N_SU3>  get_local_field_SU3(int site_index){
        array<float,N_SU3> local_field = {0};
        #pragma omp simd
        for (int i=0; i< num_bi_SU3; ++i) {
            local_field = local_field + multiply(bilinear_interaction_SU3[site_index][i], spins_SU2[bilinear_partners_SU3[site_index][i]]);
        }
        for (int i=0; i < num_tri_SU3; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction_SU3[site_index][i], spins_SU3[trilinear_partners_SU3[site_index][i][0]], spins_SU3[trilinear_partners_SU3[site_index][i][1]]);
        }
        for (int i=0; i < num_tri_SU2_SU3; ++i){
            local_field = local_field + contract_trilinear_field(mixed_trilinear_interaction_SU3[site_index][i], spins_SU2[mixed_trilinear_partners_SU3[site_index][i][0]], spins_SU2[mixed_trilinear_partners_SU3[site_index][i][1]]);
        }
        return local_field+field_SU3[site_index];
    }

    
    float metropolis(float T, std::mt19937 &gen){
        float E, E_new, dE, r, i;
        int accept = 0;
        int count = 0;
        while(count < lattice_size_SU2+lattice_size_SU3){
            int SU2_or_SU3 = random_int(0,1, gen);
            if (SU2_or_SU3 == 0){
                i = random_int(0, lattice_size_SU2-1, gen);
                E = site_energy_SU2(spins_SU2[i], i);
                array<float,N_SU2> new_spin = gen_random_spin(gen, spins_SU2[i]);
                E_new = site_energy_SU2(new_spin, i);
                dE = E_new - E;
                
                if(dE < 0){
                    spins_SU2[i] = new_spin;
                    accept++;
                }
                else{
                    r = random_float(0,1, gen);
                    if(r < exp(-dE/T)){
                        spins_SU2[i] = new_spin;
                        accept++;
                    }
                }
            }else{
                i = random_int(0, lattice_size_SU3-1, gen);
                E = site_energy_SU3(spins_SU3[i], i);
                array<float,N_SU3> new_spin = gen_random_spin(gen, spins_SU3[i]);
                E_new = site_energy_SU3(new_spin, i);
                dE = E_new - E;
                
                if(dE < 0){
                    spins_SU3[i] = new_spin;
                    accept++;
                }
                else{
                    r = random_float(0,1, gen);
                    if(r < exp(-dE/T)){
                        spins_SU3[i] = new_spin;
                        accept++;
                    }
                }
            }
            count++;
        }

        float acceptance_rate = float(accept)/float(lattice_size_SU2+lattice_size_SU3);
        return acceptance_rate;
    }

    
    void deterministic_sweep(){
        for(int i = 0; i<lattice_size_SU2; ++i){
            array<float,N_SU2> local_field = get_local_field_SU2(i);
            float norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                spins_SU2[i] = -local_field/norm;
            }
        }
        for(int i = 0; i<lattice_size_SU3; ++i){
            array<float,N_SU3> local_field = get_local_field_SU3(i);
            float norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                spins_SU3[i] = -local_field/norm;
            }
        }
    }
    
    void write_to_file_spin(string filename){
        ofstream myfile;
        myfile.open(filename);
        for(int i = 0; i<lattice_size_SU2; ++i){
            for(int j = 0; j<3; ++j){
                myfile << spins_SU2[i][j] << " ";
            }
            myfile << endl;
        }
        for(int i = 0; i<lattice_size_SU3; ++i){
            for(int j = 0; j<8; ++j){
                myfile << spins_SU3[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_to_file_pos(string filename){
        ofstream myfile;
        myfile.open(filename);
        for(int i = 0; i<lattice_size_SU2; ++i){
            for(int j = 0; j<3; ++j){
                myfile << site_pos_SU2[i][j] << " ";
            }
            myfile << endl;
        }
        for(int i = 0; i<lattice_size_SU3; ++i){
            for(int j = 0; j<3; ++j){
                myfile << site_pos_SU3[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void simulated_annealing(float T_start, float T_end, int n_therm, int n_anneal, int n_deterministics, int overrelaxation_rate, string dir_name){
        std::random_device rd;
        std::mt19937 gen(rd());
        float T = T_start;
        while(T > T_end){
            float curr_accept = 0;
            for(int i = 0; i<n_anneal; ++i){
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

        for(int i = 0; i < n_deterministics; ++i){
            deterministic_sweep();
        }   

        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            write_to_file_spin(dir_name + "/spin.txt");
            write_to_file_pos(dir_name + "/pos.txt");
        }
    }


};
#endif // MIXED_LATTICE_H