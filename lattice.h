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

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim3, size_t num_bi, size_t num_tri>
class lattice
{   
    public:
    UnitCell<N, N_ATOMS, num_bi, num_tri> UC;
    int lattice_size;
    array<array<float,N>, N_ATOMS*dim1*dim2*dim3>  spins;
    array<array<float,3>, N_ATOMS*dim1*dim2*dim3> site_pos;
    //Lookup table for the lattice
    array<array<float,N>, N_ATOMS*dim1*dim2*dim3> field;

    array<array<array<float, N>, N>, N_ATOMS*dim1*dim2*dim3> bilinear_interaction;
    array<array<array<array<float, N>, N>, N>, N_ATOMS*dim1*dim2*dim3> trilinear_interaction;

    array<int, N_ATOMS*dim1*dim2*dim3> bilinear_partners;
    array<array<int, 2>, N_ATOMS*dim1*dim2*dim3> trilinear_partners;

    array<float,N> gen_random_spin(std::mt19937 &gen){
        array<float,N> temp_spin;
        array<float,N-1> euler_angles;
        float z = random_float(-1,1, gen);
        float r = sqrt(1.0 - z*z);

        for(int i = 0; i < N-1; ++i){
            float phi = random_float(0, 2*M_PI, gen);
            euler_angles[i] = r;
            for(int j = 0; j < i; ++j){
                euler_angles[i] *= sin(euler_angles[j]);
            }
            if (i == N-2){
                euler_angles[i] *= sin(euler_angles[i]);
            }
            else{
                euler_angles[i] *= cos(euler_angles[i]);
            }

        }
        temp_spin[N-1] = z;

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
                        int count = 0;
                        for (size_t m = 0; m < num_bi; ++m){
                            bilinear<N> J = UC.bilinear_interaction[l];
                            int partner = ((i+J.offset[0])%dim1)*dim2*dim3*N_ATOMS+ ((j+J.offset[1])%dim2)*dim3*N_ATOMS+ ((k+J.offset[2])%dim3)*N_ATOMS + J.partner;
                            bilinear_interaction[current_site_index][count] = J.bilinear_interaction;
                            bilinear_partners[current_site_index][count] = partner;
                            count++;
                        }
                        count = 0;
                        for (size_t m = 0; m < num_tri; ++m){
                            trilinear<N> J = UC.trilinear_interaction[l];
                            int partner1 = ((i+J.offset1[0])%dim1)*dim2*dim3*N_ATOMS+ ((j+J.offset1[1])%dim2)*dim3*N_ATOMS+ ((k+J.offset1[2])%dim3)*N_ATOMS + J.partner1;
                            int partner2 = ((i+J.offset2[0])%dim1)*dim2*dim3*N_ATOMS+ ((j+J.offset2[1])%dim2)*dim3*N_ATOMS+ ((k+J.offset2[2])%dim3)*N_ATOMS + J.partner2;
                            trilinear_interaction[current_site_index][count] = J.trilinear_interaction;
                            trilinear_partners[current_site_index][count] = {partner1, partner2};
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
        
        energy += dot(curr_spin, field[site_index]);

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


        energy += dot(spin_here, field[site_index]);
        for (int i=0; i<num_bi; i++) {
            energy += dot(spin_here, multiply(bilinear_interaction[site_index][i], spins[bilinear_partners[site_index][i]]));
        }
        return energy;
    }
    
    float site_energy(array<float, N> &spin_here, int site_index, const array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &current_spins){
        float energy = 0.0;
        energy += dot(spin_here, field[site_index]);
        for (int i=0; i<num_bi; i++) {
            energy += dot(spin_here, multiply(bilinear_interaction[site_index][i], current_spins[bilinear_partners[site_index][i]]));
        }
        return energy;
    }

    array<float, N>  get_local_field(int site_index){
        array<float,N> local_field;
        int partner;
        array<float,N> partner_spin;

        local_field = {0};
        for (int i=0; i<num_bi; i++) {
            local_field = local_field + multiply(bilinear_interaction[site_index][i], spins[bilinear_partners[site_index][i]]);
        }

        return local_field+field[site_index];
    }


    array<float, N>  get_local_field_lattice(int site_index,const array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &current_spin){
        array<float,N> local_field;
        int partner;
        array<float,N> partner_spin;

        local_field = {0};
        for (int i=0; i<num_bi; i++) {
            local_field = local_field + multiply(bilinear_interaction[site_index][i], current_spin[bilinear_partners[site_index][i]]);
        }

        return local_field+field[site_index];
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
                    spins[i][j] = -local_field[j]/norm;
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
        while(count < lattice_size){
            i = random_int(0, lattice_size-1, gen);
            E = site_energy(i);
            new_spin = gen_random_spin(gen);
            E_new = what_if_energy(new_spin, i);
            dE = E_new - E;
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
    
    void write_to_file_spin(string filename, array<array<float,N>, N_ATOMS*dim1*dim2*dim3> towrite){
        ofstream myfile;
        myfile.open(filename);
        for(int i = 0; i<lattice_size; i++){
            for(int j = 0; j<N; j++){
                myfile << towrite[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_to_file(string filename, array<array<float,N>, N_ATOMS*dim1*dim2*dim3> towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        for(int i = 0; i<lattice_size; i++){
            for(int j = 0; j<N; j++){
                myfile << towrite[i][j] << " ";
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
            if(dir_name != ""){
                filesystem::create_directory(dir_name);
                write_to_file_spin(dir_name + "/spin.txt", spins);
                write_to_file_pos(dir_name + "/pos.txt");
            }
        }
        for(int i = 0; i<n_deterministics; i++){
            deterministic_sweep();
        }   

        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            write_to_file_spin(dir_name + "/spin.txt", spins);
            write_to_file_pos(dir_name + "/pos.txt");
        }
    }

    void landau_lifshitz_ode_int(array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &current_spin, array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &dS, const double /* t */){
        for(int i = 0; i<lattice_size; i++){
            dS[i] = cross_prod(get_local_field_lattice(i, current_spin), current_spin[i]);
        }
    }

    array<array<float,N>,N_ATOMS*dim1*dim2*dim3> landau_lifshitz(array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &current_spin){
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> dS;
        for(int i = 0; i<lattice_size; i++){
            dS[i] = cross_prod(get_local_field_lattice(i, current_spin), current_spin[i]);
        }
        return dS;
    }

    array<array<float,N>,N_ATOMS*dim1*dim2*dim3> RK4_step(const float step_size, array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &curr_spins){
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

    array<array<float,N>,N_ATOMS*dim1*dim2*dim3> euler_step(const float step_size, array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &curr_spins){
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> dS = landau_lifshitz(curr_spins);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> new_spins = array_add_2d(curr_spins, s_prod_2D(dS, step_size));
        return new_spins;
    }

    void euler_step_ode_int(const float step_size, array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &curr_spins, array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &dS){
        landau_lifshitz_ode_int(curr_spins, dS, 0);
        curr_spins = array_add_2d(curr_spins, s_prod_2D(dS, step_size));
    }


    void molecular_dynamics_SU2(float Temp_start, float Temp_end, int n_therm, int n_anneal, int overrelaxation_rate, float T_end, int n_steps, string dir_name){
        std::random_device rd;
        std::mt19937 gen(rd());
        float curr_accept;
        filesystem::create_directory(dir_name);
        float T = Temp_start;
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
        int n_steps = int(T_end/step_size);
        write_to_file_pos(dir_name + "/pos.txt");
        write_T_param(T_end, n_steps+1, dir_name + "/MD_param.txt");
        write_to_file_spin(dir_name + "/spin_t.txt", spins);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> spin_t = spins;
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> dS = {0};
        for(int i = 0; i<n_steps; i++){
            spin_t = RK4_step(step_size, spin_t);
            write_to_file(dir_name + "/spin_t.txt", spin_t);
        }
    }
};
#endif // LATTICE_H