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

    array<float,N> gen_random_spin(){
        array<float,N> temp_spin;
        array<float,N> euler_angles;
        float theta = acos(2 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 1);
        for(int i = 0; i<N-1; i++){
            euler_angles[i] = 2 * M_PI *static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }
        euler_angles[N-1] = theta;

        for(int i = 0; i<N-1;i++){
            temp_spin[i] = sin(euler_angles[N-1]);
            for(int j = 0; j<i; j++){
                temp_spin[i] *= cos(euler_angles[j]);
            }
            if(i<N-2){
                temp_spin[i] *= cos(euler_angles[i]);
            }
            else{
                temp_spin[i] *= cos(euler_angles[i]);
            }
        }
        
        temp_spin[N-1] = cos(euler_angles[N-1]);
        return temp_spin;
    }

    lattice(UnitCell<N, N_ATOMS> *atoms): UC(*atoms){
        array<array<float,3>, N_ATOMS> basis;
        array<array<float,3>, 3> unit_vector;

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
                        spins[current_site_index] = gen_random_spin();
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
        for(int i = 0; i<N; i++){
            spins[site_index][i] = rand_spin[i];
        }   
    }

    void set_spin(int site_index, array<float, N> &spin_in){
        for(int i = 0; i<N; i++){
            spins[site_index][i] = spin_in[i];
        }
    }

    float site_energy(int site_index) {
        float energy = 0.0;

        energy += dot(spins[site_index], field[site_index]);

        bilinear<N> *J = bilinear_interaction[site_index];
        for (int i=0; i<num_bi; i++) {
            int partner = J[i].partner;
            array<float,N> spin = spins[site_index];
            array<float,N> partner_spin = spins[partner];
            energy += dot(spin, multiply(J[i].bilinear_interaction, partner_spin));
        }

        return energy;
    }

    float what_if_energy(array<float, N> &spin_here, int site_index){
        float energy = 0.0;

        energy += dot(spin_here, field[site_index]);

        bilinear<N> *J = bilinear_interaction[site_index];
        for (int i=0; i<num_bi; i++) {
            int partner = J[i].partner;
            array<float,N> partner_spin = spins[partner];
            energy += dot(spin_here, multiply(J[i].bilinear_interaction, partner_spin));
        }
        return energy;
    }
    
    array<float, N> get_local_field(int site_index){
        array<float,N> local_field;
        
        local_field = field[site_index];

        bilinear<N> *J = bilinear_interaction[site_index];
        for (int i=0; i<num_bi; i++) {
            int partner = J[i].partner;
            array<float,N> partner_spin = spins[partner];
            for(int j = 0; j<N; j++){
                local_field[j] += dot(J[i].bilinear_interaction[j], partner_spin);
            }
        }
        return local_field;
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
    
    float metropolis(float T){
        int accept = 0;
        for(int i; i < lattice_size; i++){
            set_random_spin(i);
            float E = site_energy(i);
            array<float,N> new_spin = gen_random_spin();
            float E_new = what_if_energy(new_spin, i);
            float dE = E_new - E;
            if(dE < 0){
                for(int j = 0; j<N; j++){
                    spins[i][j] = new_spin[j];
                }
                accept++;
            }
            else{
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                if(r < exp(-dE/T)){
                    for(int j = 0; j<N; j++){
                        spins[i][j] = new_spin[j];
                    }
                    accept++;
                }
            }
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

    void simulated_annealing(float T_start, float T_end, int n_therm, int n_anneal, int n_deterministics, string dir_name){
        float T = T_start;
        for(int i = 0; i<n_therm; i++){
            metropolis(T);
        }
        while(T > T_end){
            float curr_accept = metropolis(T);
            cout << "Temperature: " << T << " Acceptance rate: " << curr_accept << endl;
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
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> new_spins = array_add_2d(curr_spins, s_prod_2D(array_add_2d_mult(k1, s_prod_2D(k2, 2), s_prod_2D(k3, 2), k4), step_size/6));
        return new_spins;
    }

    void molecular_dynamics_SU2(float Temp_start, float Temp_end, int n_therm, int n_anneal, float T_end, int n_steps, string dir_name){
        filesystem::create_directory(dir_name);
        float T = Temp_start;
        for(int i = 0; i<n_therm; i++){
            metropolis(T);
        }
        while(T > Temp_end){
            float curr_accept = metropolis(T);
            cout << "Temperature: " << T << " Acceptance rate: " << curr_accept << endl;
            T *= 0.9;
        }
        float step_size = T_end/float(n_steps);
        write_to_file_pos(dir_name + "/pos.txt");
        write_to_file_spin(dir_name + "/spin_0.txt");
        for(int i = 0; i<n_steps; i++){
            spins = RK4_step(step_size, spins);
            write_to_file_spin(dir_name + "/spin_" + to_string((i+1)*step_size) + ".txt");
        }
    }
};




#endif // LATTICE_H