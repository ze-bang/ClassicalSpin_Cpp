#ifndef LATTICE_H
#define LATTICE_H

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

# define M_PI           3.14159265358979323846  /* pi */


template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim3, size_t num_bi, size_t num_tri>
class lattice
{   
    public:
    UnitCell<N, N_ATOMS> UC;
    int lattice_size;
    array<array<float,N>, N_ATOMS*dim1*dim2*dim3>  spins;
    array<array<float,3>, N_ATOMS*dim1*dim2*dim3> site_pos;
    //Lookup table for the lattice
    array<array<float,N>, N_ATOMS*dim1*dim2*dim3> field;

    bilinear<N> bilinear_interaction[N_ATOMS*dim1*dim2*dim3][num_bi];
    trilinear<N> trilinear_interaction[N_ATOMS*dim1*dim2*dim3][num_tri];

    array<float,N> gen_random_spin(std::mt19937 &gen){
        array<float,N> temp_spin;
        array<float,N-1> euler_angles;
        float z = random_float(-1,1, gen);

        for(int i = 0; i<N-2; i++){
            euler_angles[i] = random_float(0, 2*M_PI, gen);
        }

        euler_angles[N-2] = z;

        if(N==3){
            float r = sqrt(1.0 - z*z);
            temp_spin = {{r*cos(euler_angles[N-3]), r*sin(euler_angles[N-3]), z}};
        }

        return temp_spin;
    }

    lattice(UnitCell<N, N_ATOMS> *atoms): UC(*atoms){
        array<array<float,3>, N_ATOMS> basis;
        array<array<float,3>, 3> unit_vector;

        std::random_device rd;
        std::mt19937 gen(rd());

        lattice_size = dim1*dim2*dim3*N_ATOMS;
        basis = UC.lattice_pos;
        unit_vector = UC.lattice_vectors;


        for (int i=0; i<dim1; i++){
            for (int j=0; j< dim2; j++){
                for(int k=0; k<dim3;k++){
                    for (int l=0; l<N_ATOMS;l++){

                        int current_site_index = i*dim2*dim3*N_ATOMS+ j*dim3*N_ATOMS+ k*N_ATOMS + l;
                        site_pos[current_site_index]  = unit_vector[0]*i + unit_vector[1]*j + unit_vector[2]*k + basis[l];
                        
                        array<float,3> temp = unit_vector[0]*i;
                        spins[current_site_index] = gen_random_spin(gen);
                        field[current_site_index] = UC.field[l];
                        auto bilinear_matched = UC.bilinear_interaction.equal_range(l);
                        
                        int count = 0;
                        for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m){
                            bilinear<N> J = m->second;
                            int partner = ((i+J.offset[0])%dim1)*dim2*dim3*N_ATOMS+ ((j+J.offset[1])%dim2)*dim3*N_ATOMS+ ((k+J.offset[2])%dim3)*N_ATOMS + J.partner;
                            bilinear<N> J_flatten = bilinear<N>(J.bilinear_interaction, partner);
                            bilinear_interaction[current_site_index][count] = J_flatten;
                            count++;
                        }
                        auto trilinear_matched = UC.trilinear_interaction.equal_range(l);
                        count = 0;
                        for (auto m = trilinear_matched.first; m != trilinear_matched.second; ++m){
                            trilinear<N> J = m->second;
                            int partner1 = ((i+J.offset1[0])%dim1)*dim2*dim3*N_ATOMS+ ((j+J.offset1[1])%dim2)*dim3*N_ATOMS+ ((k+J.offset1[2])%dim3)*N_ATOMS + J.partner1;
                            int partner2 = ((i+J.offset2[0])%dim1)*dim2*dim3*N_ATOMS+ ((j+J.offset2[1])%dim2)*dim3*N_ATOMS+ ((k+J.offset2[2])%dim3)*N_ATOMS + J.partner2;
                            trilinear<N> J_flatten = trilinear<N>(J.trilinear_interaction, partner1, partner2);
                            trilinear_interaction[current_site_index][count] = J_flatten;
                            count++;
                        }
                    }
                }
            }
        }
    };

    void set_random_spin(int site_index){
        array<float, N> rand_spin = gen_random_spin();
        spins[site_index] = rand_spin;
    }

    void set_spin(int site_index, array<float, N> &spin_in){
        spins[site_index] = spin_in;
    }

    float site_energy(int site_index) {
        float energy = 0.0;
        int partner;
        array<float,N> partner_spin;
        array<float,N> curr_spin = spins[site_index];

        energy -= dot(curr_spin, field[site_index]);

        bilinear<N> *J = bilinear_interaction[site_index];
        for (int i=0; i<num_bi; i++) {
            partner = J[i].partner;
            partner_spin = spins[partner];
            energy += dot(curr_spin, multiply(J[i].bilinear_interaction, partner_spin));
        }

        return energy;
    }

    float what_if_energy(array<float, N> &spin_here, int site_index){
        float energy = 0.0;
        int partner;
        array<float,N> partner_spin;

        energy -= dot(spin_here, field[site_index]);

        bilinear<N> *J = bilinear_interaction[site_index];
        for (int i=0; i<num_bi; i++) {
            partner = J[i].partner;
            partner_spin = spins[partner];
            energy += dot(spin_here, multiply(J[i].bilinear_interaction, partner_spin));
        }
        return energy;
    }
    
    array<float, N>  get_local_field(int site_index){
        array<float,N> local_field;
        int partner;
        array<float,N> partner_spin;

        local_field = {0};

        bilinear<N> *J = bilinear_interaction[site_index];
        for (int i=0; i<num_bi; i++) {
            partner = J[i].partner;
            partner_spin = spins[partner];
            local_field = local_field + multiply(J[i].bilinear_interaction, partner_spin);
        }
        return local_field-field[site_index];
    }

    void deterministic_sweep(){
        for(int i = 0; i<lattice_size; i++){
            array<float,N> local_field = get_local_field(i);
            float norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                for(int j=0; j < N; j++){
                    spins[i][j] = local_field[j]/norm;
                }
            }
        }
    }
    
    void overrelaxation(){
        array<float,N> local_field;
        float proj;
        for(int i = 0; i<lattice_size; i++){
            local_field = get_local_field(i);
            float norm = dot(local_field, local_field);
            if(norm == 0){
                continue;
            }
            else{
                proj = 2* dot(spins[i], local_field)/norm;
                spins[i] = local_field*proj - spins[i];
            }
        }
    }

    float metropolis(float T, std::mt19937 &gen){
        float E, E_new, dE, r, i;
        array<float,N> new_spin;
        int accept = 0;
        int count = 0;
        // auto start = chrono::high_resolution_clock::now();
        // auto end = chrono::high_resolution_clock::now();
        // auto duration = duration_cast<chrono::microseconds>(end - start);

        while(count < lattice_size){
            i = random_int(0, lattice_size-1, gen);
            E = site_energy(i);
            new_spin = gen_random_spin(gen);
            E_new = what_if_energy(new_spin, i);
            dE = E_new - E;
            
            // cout << "Energy: " << E << " New Energy: " << E_new << " dE: " << dE << endl;

            if(dE < 0){
                spins[i] = new_spin;
                accept++;
            }

            else{
                r = random_float(0,1, gen);
                if(r < exp(-dE/T)){
                    spins[i] = new_spin;
                    accept++;
                }
            }
            count++;
        }

        float acceptance_rate = float(accept)/float(lattice_size);
        return acceptance_rate;
    }
    
    void write_to_file_spin(string filename){
        ofstream myfile;
        myfile.open(filename);
        for(int i = 0; i<lattice_size; i++){
            for(int j = 0; j<N; j++){
                myfile << spins[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_to_file_pos(string filename){
        ofstream myfile;
        myfile.open(filename);
        for(int i = 0; i<lattice_size; i++){
            for(int j = 0; j<3; j++){
                myfile << site_pos[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_T_param(float T_end, int num_steps, string dir_name){
        ofstream myfile;
        myfile.open(dir_name);
        myfile << T_end << " " << num_steps << endl;
        myfile.close();
    }

    void simulated_annealing(float T_start, float T_end, int n_therm, int n_anneal, int n_deterministics, int overrelaxation_rate, string dir_name){
        std::random_device rd;
        std::mt19937 gen(rd());
        float T = T_start;
        for(int i = 0; i<n_therm; i++){
            metropolis(T, gen);
        }
        while(T > T_end){
            float curr_accept = 0;
            for(int i = 0; i<n_anneal; i++){
                if(overrelaxation_rate > 0){
                    overrelaxation();
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
        for(int i = 0; i<n_deterministics; i++){
            deterministic_sweep();
        }   

        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            write_to_file_spin(dir_name + "/spin.txt");
            write_to_file_pos(dir_name + "/pos.txt");
        }
    }

    array<array<float,N>,N_ATOMS*dim1*dim2*dim3> landau_lifshitz(array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &current_spin){
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> dS;
        for(int i = 0; i<lattice_size; i++){
            array<float,N> local_field = get_local_field(i);
            array<float,N> local_spin = current_spin[i];
            dS[i] = cross_prod(local_field, local_spin);
        }
        return dS;
    }

    array<array<float,N>,N_ATOMS*dim1*dim2*dim3> RK4_step(float step_size, array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &curr_spins){
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k1 = landau_lifshitz(curr_spins);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> lval_k2 = array_add_2d(curr_spins, s_prod_2D(k1, 0.5*step_size));
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k2 = landau_lifshitz(lval_k2);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> lval_k3 = array_add_2d(curr_spins, s_prod_2D(k2, 0.5*step_size));
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k3 = landau_lifshitz(lval_k3);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> lval_k4 = array_add_2d(curr_spins, s_prod_2D(k3, step_size));
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k4 = landau_lifshitz(lval_k4);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> new_spins  = array_add_2d(curr_spins, s_prod_2D(array_add_2d_mult(k1, s_prod_2D(k2, 2), s_prod_2D(k3, 2), k4), step_size/6));
        return new_spins;
    }



    void molecular_dynamics_SU2(float Temp_start, float Temp_end, int n_therm, int n_anneal, int overrelaxation_rate, float T_end, int n_steps, string dir_name){
        std::random_device rd;
        std::mt19937 gen(rd());
        float curr_accept;
        filesystem::create_directory(dir_name);
        float T = Temp_start;
        for(int i = 0; i<n_therm; i++){
            metropolis(T, gen);
        }
        while(T > Temp_end){
            curr_accept = 0;
            for(int i = 0; i<n_anneal; i++){
                if(overrelaxation_rate > 0){
                    overrelaxation();
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
        float step_size = T_end/float(n_steps);
        write_to_file_pos(dir_name + "/pos.txt");
        write_to_file_spin(dir_name + "/spin_0.txt");
        write_T_param(T_end, n_steps, dir_name + "/T.txt");
        for(int i = 0; i<n_steps; i++){
            spins = RK4_step(step_size, spins);
            write_to_file_spin(dir_name + "/spin_" + to_string(i+1) + ".txt");
        }
    }
};




#endif // LATTICE_H