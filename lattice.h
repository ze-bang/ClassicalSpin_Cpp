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
#include <functional>
#include <mpi.h>
#include "binning_analysis.h"
template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim3>
class lattice
{   
    public:

    typedef array<array<double,N>,N_ATOMS*dim1*dim2*dim3> spin_config;
    typedef function<array<double,N>(const array<double,N> &, const array<double, N> &)> cross_product_method;
    typedef function<spin_config(double &, const spin_config &, const double, cross_product_method)> ODE_method;

    UnitCell<N, N_ATOMS> UC;
    size_t lattice_size;
    spin_config  spins;
    array<array<double,3>, N_ATOMS*dim1*dim2*dim3> site_pos;
    //Lookup table for the lattice
    spin_config field;
    spin_config driving_field;
    array<array<array<double, N>, N>, N_ATOMS*dim1*dim2*dim3> onsite_interaction;

    array<vector<array<array<double, N>, N>>, N_ATOMS*dim1*dim2*dim3> bilinear_interaction;
    array<vector<array<array<array<double, N>, N>, N>>, N_ATOMS*dim1*dim2*dim3> trilinear_interaction;

    array<vector<size_t>, N_ATOMS*dim1*dim2*dim3> bilinear_partners;
    array<vector<array<size_t, 2>>, N_ATOMS*dim1*dim2*dim3> trilinear_partners;

    size_t num_bi;
    size_t num_tri;
    size_t num_gen;
    float spin_length;

    array<double,N> gen_random_spin(std::mt19937 &gen, float spin_l){
        array<double,N> temp_spin;
        array<double,N-2> euler_angles;
        double z = random_double(-1,1, gen);
        double r = sqrt(1.0 - z*z);

        for(size_t i = 0; i < N-2; ++i){
            euler_angles[i] = random_double(0, 2*M_PI, gen);
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
        return temp_spin*spin_l;
    }


    size_t flatten_index(size_t i, size_t j, size_t k, size_t l){
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

    size_t flatten_index_periodic_boundary(int i, int j, int k, int l){
        return periodic_boundary(i, dim1)*dim2*dim3*N_ATOMS+ periodic_boundary(j, dim2)*dim3*N_ATOMS+ periodic_boundary(k, dim3)*N_ATOMS + l;
    }

    lattice(const UnitCell<N, N_ATOMS> *atoms, float spin_l=1): UC(*atoms){
        array<array<double,3>, N_ATOMS> basis;
        array<array<double,3>, 3> unit_vector;

        std::random_device rd;
        std::mt19937 gen(rd());

        lattice_size = dim1*dim2*dim3*N_ATOMS;
        basis = UC.lattice_pos;
        unit_vector = UC.lattice_vectors;
        spin_length = spin_l;

        for (size_t i=0; i< dim1; ++i){
            for (size_t j=0; j< dim2; ++j){
                for(size_t k=0; k< dim3;++k){
                    for (size_t l=0; l< N_ATOMS;++l){

                        size_t current_site_index = flatten_index(i,j,k,l);

                        site_pos[current_site_index]  = unit_vector[0]*int(i) + unit_vector[1]*int(j)  + unit_vector[2]*int(k)  + basis[l];
                        spins[current_site_index] = gen_random_spin(gen, spin_length);
                        field[current_site_index] = UC.field[l];
                        driving_field[current_site_index] = {0};
                        onsite_interaction[current_site_index] = UC.onsite_interaction[l];
                        auto bilinear_matched = UC.bilinear_interaction.equal_range(l);
                        int count = 0;
                        for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m){
                            bilinear<N> J = m->second;
                            size_t partner = flatten_index_periodic_boundary(int(i)+J.offset[0], int(j)+J.offset[1], int(k)+J.offset[2], J.partner);
                            bilinear_interaction[current_site_index].push_back(J.bilinear_interaction);
                            bilinear_partners[current_site_index].push_back(partner);
                            bilinear_interaction[partner].push_back(transpose2D(J.bilinear_interaction));
                            bilinear_partners[partner].push_back(current_site_index);
                            count++;
                        }
                        auto trilinear_matched = UC.trilinear_interaction.equal_range(l);
                        count = 0;
                        for (auto m = trilinear_matched.first; m != trilinear_matched.second; ++m){
                            trilinear<N> J = m->second;
                            size_t partner1 = flatten_index_periodic_boundary(i+J.offset1[0], j+J.offset1[1], k+J.offset1[2], J.partner1);
                            size_t partner2 = flatten_index_periodic_boundary(i+J.offset2[0], j+J.offset2[1], k+J.offset2[2], J.partner2);
                            
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
        num_gen = spins[0].size();
        // for(size_t i = 0 ; i < lattice_size; ++i){
        //     for(size_t j =0; j < num_bi; ++j){
        //         int partner = bilinear_partners[i][j];
        //         array<array<double,N>, N> J = bilinear_interaction[i][j];
        //         cout << "Bilinear: " << i << " at site " << site_pos[i][0] << " " << site_pos[i][1] << " " << site_pos[i][2] << " " << partner  << " on site " << site_pos[partner][0] << " " << site_pos[partner][1] << " " << site_pos[partner][2] << " with interaction " << J[0][0] << " " << J[1][1] << " " << J[2][2] << endl;
        //     }
        // }

    };

    lattice(const lattice<N, N_ATOMS, dim1, dim2, dim3> *lattice_in){
        UC = lattice_in->UC;
        lattice_size = lattice_in->lattice_size;
        spins = lattice_in->spins;
        site_pos = lattice_in->site_pos;
        field = lattice_in->field;
        onsite_interaction = lattice_in->onsite_interaction;
        bilinear_interaction = lattice_in->bilinear_interaction;
        trilinear_interaction = lattice_in->trilinear_interaction;
        bilinear_partners = lattice_in->bilinear_partners;
        trilinear_partners = lattice_in->trilinear_partners;
        num_bi = lattice_in->num_bi;
        num_tri = lattice_in->num_tri;
        num_gen = lattice_in->num_gen;
    };

    void reset_lattice(const lattice<N, N_ATOMS, dim1, dim2, dim3> *lattice_in){
        UC = lattice_in->UC;
        lattice_size = lattice_in->lattice_size;
        spins = lattice_in->spins;
        site_pos = lattice_in->site_pos;
        field = lattice_in->field;
        onsite_interaction = lattice_in->onsite_interaction;
        driving_field = lattice_in->driving_field;
        bilinear_interaction = lattice_in->bilinear_interaction;
        trilinear_interaction = lattice_in->trilinear_interaction;
        bilinear_partners = lattice_in->bilinear_partners;
        trilinear_partners = lattice_in->trilinear_partners;
        num_bi = lattice_in->num_bi;
        num_tri = lattice_in->num_tri;
        num_gen = lattice_in->num_gen;
    };


    void set_spin(size_t site_index, array<double, N> &spin_in){
        spins[site_index] = spin_in;
    }

    double site_energy(array<double, N> &spin_here, size_t site_index){
        double energy = 0.0;
        energy -= dot(spin_here, field[site_index]);
        energy -= dot(spin_here, driving_field[site_index]);
        energy += contract(spin_here, onsite_interaction[site_index], spin_here);
        #pragma omp simd
        for (size_t i=0; i< num_bi; ++i) {
            energy += contract(spin_here, bilinear_interaction[site_index][i], spins[bilinear_partners[site_index][i]]);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri; ++i){
            energy += contract_trilinear(trilinear_interaction[site_index][i], spin_here, spins[trilinear_partners[site_index][i][0]], spins[trilinear_partners[site_index][i][1]]);
        }
        return energy;
    }

    double total_energy(spin_config &curr_spins){
        double field_energy = 0.0;
        double drive_energy = 0.0;
        double onsite_energy = 0.0;
        double bilinear_energy = 0.0;
        double trilinear_energy = 0.0;

        #pragma omp simd
        for(size_t i = 0; i < lattice_size; ++i){
            field_energy -= dot(curr_spins[i], field[i]);
            drive_energy -= dot(curr_spins[i], driving_field[i]);
            onsite_energy += contract(curr_spins[i], onsite_interaction[i], curr_spins[i]);
            #pragma omp simd
            for (size_t j=0; j< num_bi; ++j) {
                bilinear_energy += contract(curr_spins[i], bilinear_interaction[i][j], curr_spins[bilinear_partners[i][j]]);
            }
            #pragma omp simd
            for (size_t j=0; j < num_tri; ++j){
                trilinear_energy += contract_trilinear(trilinear_interaction[i][j], curr_spins[i], curr_spins[trilinear_partners[i][j][0]], curr_spins[trilinear_partners[i][j][1]]);
            }
        }
        return field_energy + drive_energy + onsite_energy + bilinear_energy/2 + trilinear_energy/3;
    }

    double energy_density(spin_config &curr_spins){
        return total_energy(curr_spins)/lattice_size;
    }
    
    array<double, N>  get_local_field(size_t site_index){
        array<double,N> local_field;
        local_field = multiply(onsite_interaction[site_index], spins[site_index]);
        #pragma omp simd
        for (size_t i=0; i< num_bi; ++i) {
            local_field = local_field + multiply(bilinear_interaction[site_index][i], spins[bilinear_partners[site_index][i]]);
        }
        for (size_t i=0; i < num_tri; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction[site_index][i], spins[trilinear_partners[site_index][i][0]], spins[trilinear_partners[site_index][i][1]]);
        }
        return local_field-field[site_index]-driving_field[site_index];
    }


    array<double, N>  get_local_field_lattice(size_t site_index, const spin_config &current_spin){
        array<double,N> local_field;
        local_field =  multiply(onsite_interaction[site_index], spins[site_index]);
        #pragma omp simd
        for (size_t i=0; i< num_bi; ++i) {
            local_field = local_field + multiply(bilinear_interaction[site_index][i], current_spin[bilinear_partners[site_index][i]]);
        }
        for (size_t i=0; i < num_tri; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction[site_index][i], current_spin[trilinear_partners[site_index][i][0]], current_spin[trilinear_partners[site_index][i][1]]);
        }
        return local_field-field[site_index]-driving_field[site_index];
    }


    void deterministic_sweep(std::mt19937 &gen){
        size_t count = 0;
        int i;
        while(count < lattice_size){
            i = random_int(0, lattice_size-1, gen);
            array<double,N> local_field = get_local_field(i);
            double norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                for(size_t j=0; j < N; ++j){
                    spins[i][j] = -local_field[j]/norm;
                }
            }
            count++;
        }
    }
    
    array<double,N> gaussian_move(const array<double,N> &current_spin, std::mt19937 &gen, double sigma=60){
        array<double,N> new_spin;
        new_spin = current_spin + gen_random_spin(gen,spin_length)*sigma;
        return new_spin/sqrt(dot(new_spin, new_spin)) * spin_length;
    }

    void overrelaxation(std::mt19937 &gen){
        array<double,N> local_field;
        int i;
        double proj;
        size_t count = 0;
        while(count < lattice_size){
            i = random_int(0, lattice_size-1, gen);
            local_field = get_local_field(i);
            double norm = dot(local_field, local_field);
            if(norm == 0){
                continue;
            }
            else{
                proj = 2* dot(spins[i], local_field)/norm;
                spins[i] = local_field*proj - spins[i];
            }
            count++;
        }
    }

    double metropolis(spin_config &curr_spin, double T, std::mt19937 &gen, bool gaussian=false, double sigma=60){
        double E, E_new, dE, r;
        int i;
        array<double,N> new_spin;
        int accept = 0;
        size_t count = 0;
        while(count < lattice_size){
            i = random_int(0, lattice_size-1, gen);
            E = site_energy(curr_spin[i], i);
            if (gaussian){
                new_spin = gaussian_move(curr_spin[i], gen, sigma);
            }
            else{
                new_spin = gen_random_spin(gen,spin_length);
            }
            E_new = site_energy(new_spin, i);
            dE = E_new - E;
            
            if(dE < 0){
                curr_spin[i] = new_spin;
                accept++;
            }
            else{
                r = random_double(0,1, gen);
                if(r < exp(-dE/T)){
                    curr_spin[i] = new_spin;
                    accept++;
                }
            }
            count++;
        }

        double acceptance_rate = double(accept)/double(lattice_size);
        return acceptance_rate;
    }

    
    void write_to_file_spin(string filename, spin_config towrite){
        ofstream myfile;
        myfile.open(filename);
        for(size_t i = 0; i<lattice_size; ++i){
            for(size_t j = 0; j<N; ++j){
                myfile << towrite[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_to_file(string filename, spin_config towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        for(size_t i = 0; i<lattice_size; ++i){
            for(size_t j = 0; j<N; ++j){
                myfile << towrite[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }
    void write_to_file_magnetization_init(string filename, double towrite){
        ofstream myfile;
        myfile.open(filename);
        myfile << towrite << " ";
        myfile << endl;
        myfile.close();
    }
    void write_to_file_magnetization(string filename, double towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        myfile << towrite << " ";
        myfile << endl;
        myfile.close();
    }

    void write_to_file_magnetization_local(string filename, array<double, N> towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        for(size_t j = 0; j<N; ++j){
            myfile << towrite[j] << " ";
        }
        myfile << endl;
        myfile.close();
    }

    void write_to_file_2d_vector_array(string filename, vector<array<double, N>> towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        for(size_t i = 0; i<towrite.size(); ++i){
            for(size_t j = 0; j<N; ++j){
                myfile << towrite[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_column_vector(string filename, vector<double> towrite){
        ofstream myfile;
        myfile.open(filename);
        for(size_t j = 0; j<towrite.size(); ++j){
            myfile << towrite[j] << endl;
        }
        myfile.close();
    }

    void write_to_file_pos(string filename){
        ofstream myfile;
        myfile.open(filename);
        for(size_t i = 0; i<lattice_size; ++i){
            for(size_t j = 0; j<3; ++j){
                myfile << site_pos[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_T_param(double T_end, size_t num_steps, string dir_name){
        ofstream myfile;
        myfile.open(dir_name);
        myfile << T_end << " " << num_steps << " " << lattice_size << endl;
        myfile.close();
    }

    void simulated_annealing(double T_start, double T_end, size_t n_anneal, size_t overrelaxation_rate, bool gaussian_move = false, string out_dir = ""){
        if (out_dir != ""){
            filesystem::create_directory(out_dir);
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        double T = T_start;
        double acceptance_rate = 0;
        double sigma = 40;
        cout << "Gaussian Move: " << gaussian_move << endl;
        while(T > T_end){
            double curr_accept = 0;
            for(size_t i = 0; i<n_anneal; ++i){
                if(overrelaxation_rate > 0){
                    overrelaxation(gen);
                    if (i%overrelaxation_rate == 0){
                        curr_accept += metropolis(spins, T, gen, gaussian_move, sigma);
                    }
                }
                else{
                    curr_accept += metropolis(spins, T, gen, gaussian_move, sigma);
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
            if(out_dir != ""){
                vector<double> energies;
                for(size_t i = 0; i<10000; ++i){
                    metropolis(spins, T, gen, gaussian_move, sigma);
                    if (i % 100 == 0){
                        energies.push_back(total_energy(spins));
                    }
                }
                std::tuple<double,double> varE = binning_analysis(energies, int(energies.size()/10));
                double curr_heat_capacity = 1/(T*T)*get<0>(varE)/lattice_size;
                double curr_dHeat = 1/(T*T)*get<1>(varE)/lattice_size;
                ofstream myfile;
                myfile.open(out_dir + "/specific_heat.txt", ios::app);
                myfile << T << " " << curr_heat_capacity << " " << curr_dHeat;
                myfile << endl;
                myfile.close();
            }
            T *= 0.9;
        }
        if(out_dir != ""){
            write_to_file_spin(out_dir + "/spin.txt", spins);
            write_to_file_pos(out_dir + "/pos.txt");
        }
    }

    void simulated_annealing_deterministic(double T_start, double T_end, size_t n_anneal, size_t n_deterministics, size_t overrelaxation_rate, string dir_name, bool gaussian_move = false){
        std::random_device rd;
        std::mt19937 gen(rd());
        simulated_annealing(T_start, T_end, n_anneal, overrelaxation_rate, gaussian_move);
        for(size_t i = 0; i<n_deterministics; ++i){
            deterministic_sweep(gen);
        }   

        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            write_to_file_spin(dir_name + "/spin.txt", spins);
            write_to_file_pos(dir_name + "/pos.txt");
        }
    }

    void parallel_tempering(vector<double> temp, size_t n_therm, size_t n_anneal, size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate, string dir_name, const vector<int> rank_to_write, bool gaussian_move = false){
        std::random_device rd;
        std::mt19937 gen(rd());

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

        double E, T_partner, E_partner;
        bool accept;        
        spin_config new_spins;
        double curr_Temp = temp[rank];
        vector<double> heat_capacity, dHeat;
        if (rank == 0){
            heat_capacity.resize(size);
            dHeat.resize(size);
        }   
        vector<double> energies;
        vector<array<double,N>> magnetizations;

        cout << "Initialized Process on rank: " << rank << " with temperature: " << curr_Temp << endl;

        for(size_t i=0; i < n_anneal+n_therm; ++i){

            // Metropolis
            if(overrelaxation_rate > 0){
                overrelaxation(gen);
                if (i%overrelaxation_rate == 0){
                    curr_accept += metropolis(spins, curr_Temp, gen, gaussian_move);
                }
            }
            else{
                curr_accept += metropolis(spins, curr_Temp, gen, gaussian_move);
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
                        accept = min(double(1.0), exp((1/curr_Temp-1/T_partner)*(E - E_partner))) > random_double(0,1, gen);
                        MPI_Send(&accept, 1, MPI_C_BOOL, partner_rank, 2, MPI_COMM_WORLD);
                    } else{
                        MPI_Recv(&accept, 1, MPI_C_BOOL, partner_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    if (accept){
                        if (partner_rank % 2 == 0){
                            MPI_Send(&spins, N*lattice_size, MPI_DOUBLE, partner_rank, 4, MPI_COMM_WORLD);
                            MPI_Recv(&new_spins, N*lattice_size, MPI_DOUBLE, partner_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        } else{
                            MPI_Recv(&new_spins, N*lattice_size, MPI_DOUBLE, partner_rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            MPI_Send(&spins, N*lattice_size, MPI_DOUBLE, partner_rank, 3, MPI_COMM_WORLD);
                        }
                        copy(new_spins.begin(), new_spins.end(), spins.begin());
                        E = E_partner;
                        swap_accept++;
                    }
                }
            }

            if (i >= n_therm){
                if (i % probe_rate == 0){
                    if(dir_name != ""){
                        magnetizations.push_back(magnetization_local(spins));
                        energies.push_back(E);
                    }
                }
            }
        }
        
        // double curr_heat_capacity = 1/(curr_Temp*curr_Temp)*variance(energies)/lattice_size;
        std::tuple<double,double> varE = binning_analysis(energies, int(energies.size()/10));
        double curr_heat_capacity = 1/(curr_Temp*curr_Temp)*get<0>(varE)/lattice_size;
        double curr_dHeat = 1/(curr_Temp*curr_Temp)*get<1>(varE)/lattice_size;
        MPI_Gather(&curr_heat_capacity, 1, MPI_DOUBLE, heat_capacity.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&curr_dHeat, 1, MPI_DOUBLE, dHeat.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        cout << "Process finished on rank: " << rank << " with temperature: " << curr_Temp << " with local acceptance rate: " << double(curr_accept)/double(n_anneal+n_therm)*overrelaxation_flag << " Swap Acceptance rate: " << double(swap_accept)/double(n_anneal+n_therm)*swap_rate*overrelaxation_flag << endl;
        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            for(size_t i=0; i<rank_to_write.size(); ++i){
                if (rank == rank_to_write[i]){
                    write_to_file_spin(dir_name + "/spin" + to_string(rank) + ".txt", spins);
                    write_to_file_2d_vector_array(dir_name + "/magnetization" + to_string(rank) + ".txt", magnetizations);
                    write_column_vector(dir_name + "/energy" + to_string(rank) + ".txt", energies);
                }
            }
            if (rank == 0){
                write_to_file_pos(dir_name + "/pos.txt");
                ofstream myfile;
                myfile.open(dir_name + "/heat_capacity.txt", ios::app);
                for(size_t j = 0; j<size; ++j){
                    myfile << temp[j] << " " << heat_capacity[j] << " " << dHeat[j] << endl;
                }
                myfile.close();
            }
        }
    }

    void print_2D(const spin_config &a){
        for(size_t i = 0; i<N_ATOMS*dim1*dim2*dim3; ++i){
            for(size_t j = 0; j<N; ++j){
                cout << a[i][j] << " ";
            }
            cout << endl;
        }
    }

    spin_config landau_lifshitz(const spin_config &current_spin, cross_product_method cross_prod){
        spin_config dS;
        #pragma omp simd
        for(size_t i = 0; i<lattice_size; ++i){
            dS[i] = cross_prod(get_local_field_lattice(i, current_spin), current_spin[i]);
        }
        return dS;
    }

    spin_config RK4_step(double &step_size, const spin_config &curr_spins, const double tol, cross_product_method cross_prod){
        spin_config k1 = landau_lifshitz(curr_spins,cross_prod);
        spin_config lval_k2 = curr_spins + k1*(0.5*step_size);
        spin_config k2 = landau_lifshitz(lval_k2,cross_prod);
        spin_config lval_k3 = curr_spins + k2*(0.5*step_size);
        spin_config k3 = landau_lifshitz(lval_k3,cross_prod);
        spin_config lval_k4 = curr_spins + k3*step_size;
        spin_config k4 = landau_lifshitz(lval_k4,cross_prod);
        spin_config new_spins  = curr_spins + (k1+ k2 * 2 + k3 * 2 + k4)*(step_size/6);
        return new_spins;
    }

    spin_config RK45_step(double &step_size, const spin_config &curr_spins, const double tol, cross_product_method cross_prod){
        spin_config k1 = landau_lifshitz(curr_spins,cross_prod)*step_size;
        spin_config k2 = landau_lifshitz(curr_spins + k1*(1.0/4.0),cross_prod)*step_size;
        spin_config k3 = landau_lifshitz(curr_spins + k1*(3.0/32.0) + k2*(9.0/32.0),cross_prod)*step_size;
        spin_config k4 = landau_lifshitz(curr_spins + k1*(1932.0/2197.0) + k2*(-7200.0/2197.0) + k3*(7296.0/2197.0),cross_prod)*step_size;
        spin_config k5 = landau_lifshitz(curr_spins + k1*(439.0/216.0) + k2*(-8.0) + k3*(3680.0/513.0) + k4*(-845.0/4104.0),cross_prod)*step_size;
        spin_config k6 = landau_lifshitz(curr_spins + k1*(-8.0/27.0) + k2*(2.0) + k3*(-3544.0/2565.0)+ k4*(1859.0/4104.0)+ k5*(-11.0/40.0),cross_prod)*step_size;

        spin_config y = curr_spins + k1*(25.0/216.0) + k3*(1408.0/2565.0) + k4*(2197.0/4101.0) - k5*(1.0/5.0);
        spin_config z = curr_spins + k1*(16.0/135.0) + k3*(6656.0/12825.0) + k4*(28561.0/56430.0) - k5*(9.0/50.0) + k6*(2.0/55.0);

        double error = norm_average_2D(z-y);
        step_size *= 0.9*pow(tol/error, 0.2);
        if (error < tol){
            return z;
        }
        else{
            return RK45_step(step_size, curr_spins, tol, cross_prod);
        }
        return z;
    }
    spin_config RK45_step_fixed(double &step_size, const spin_config &curr_spins, const double tol, cross_product_method cross_prod){
        spin_config k1 = landau_lifshitz(curr_spins,cross_prod)*step_size;
        spin_config k2 = landau_lifshitz(curr_spins + k1*(1.0/4.0),cross_prod)*step_size;
        spin_config k3 = landau_lifshitz(curr_spins + k1*(3.0/32.0) + k2*(9.0/32.0),cross_prod)*step_size;
        spin_config k4 = landau_lifshitz(curr_spins + k1*(1932.0/2197.0) + k2*(-7200.0/2197.0) + k3*(7296.0/2197.0),cross_prod)*step_size;
        spin_config k5 = landau_lifshitz(curr_spins + k1*(439.0/216.0) + k2*(-8.0) + k3*(3680.0/513.0) + k4*(-845.0/4104.0),cross_prod)*step_size;
        spin_config k6 = landau_lifshitz(curr_spins + k1*(-8.0/27.0) + k2*(2.0) + k3*(-3544.0/2565.0)+ k4*(1859.0/4104.0)+ k5*(-11.0/40.0),cross_prod)*step_size;
        spin_config z = curr_spins + k1*(16.0/135.0) + k3*(6656.0/12825.0) + k4*(28561.0/56430.0) - k5*(9.0/50.0) + k6*(2.0/55.0);
        return z;
    }
    spin_config euler_step(const double step_size, const spin_config &curr_spins, const double tol, cross_product_method cross_prod){
        spin_config dS = landau_lifshitz(curr_spins, cross_prod);
        spin_config new_spins = curr_spins + dS*step_size;
        return new_spins;
    }

    spin_config SSPRK53_step(const double step_size, const spin_config &curr_spins, const double tol, cross_product_method cross_prod){
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

        spin_config tmp = curr_spins + landau_lifshitz(curr_spins, cross_prod) * b10 * step_size;
        spin_config k = landau_lifshitz(tmp, cross_prod);
        spin_config u = tmp + k * step_size * b21;
        //u3
        k = landau_lifshitz(u, cross_prod);
        tmp = curr_spins * a30 + u * a32 + k * step_size * b32;
        k = landau_lifshitz(tmp, cross_prod);
        //u4
        tmp = curr_spins * a40 + tmp * a43 + k * step_size * b43;
        k = landau_lifshitz(tmp, cross_prod);
        u = u * a52 + tmp * a54 + k * step_size * b54;
        return u;
    }


    void molecular_dynamics(double Temp_start, double Temp_end, size_t n_anneal, size_t overrelaxation_rate, double T_start, double T_end, double step_size, string dir_name, bool gaussian_move = false){
        // simulated_annealing(Temp_start, Temp_end, n_anneal, overrelaxation_rate, gaussian_move);
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        write_to_file_pos(dir_name + "/pos.txt");
        write_to_file_spin(dir_name + "/spin_t.txt", spins);
        spin_config spin_t = spins;
        cross_product_method cross_prod;
        if constexpr(N==3){
            cross_prod = cross_prod_SU2;
        }else if constexpr(N==8){
            cross_prod = cross_prod_SU3;
        }

        double tol = 1e-4;

        int check_frequency = 10;
        double currT = T_start;
        size_t count = 1;
        vector<double> time;

        time.push_back(currT);

        while(currT < T_end){
            spin_t = SSPRK53_step(step_size, spin_t, tol, cross_prod);
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

    void set_pulse(double currT, const array<array<double,N>, N_ATOMS> &field_in, double t_B, double pulse_amp, double pulse_width, double pulse_freq){
        double factor = double(pulse_amp*exp(-pow((currT-t_B)/(2*pulse_width),2))*cos(2*M_PI*pulse_freq*(currT-t_B)));
        for (size_t i=0; i< dim1; ++i){
            for (size_t j=0; j< dim2; ++j){
                for(size_t k=0; k< dim3;++k){
                    for (size_t l=0; l< N_ATOMS;++l){
                        size_t current_site_index = flatten_index(i,j,k,l);
                        driving_field[current_site_index] = field_in[l]*factor;
                    }
                }
            }
        }
    }

    void reset_pulse(){
        for (size_t i=0; i< lattice_size; ++i){
            driving_field[i] = {0};
        }
    }

    void set_two_pulse(double currT, const array<array<double,N>, N_ATOMS> &field_in_1, double t_B_1, const array<array<double,N>, N_ATOMS> &field_in_2, double t_B_2, double pulse_amp, double pulse_width, double pulse_freq){
        double factor1 = double(pulse_amp*exp(-pow((currT-t_B_1)/(2*pulse_width),2))*cos(2*M_PI*pulse_freq*(currT-t_B_1)));
        double factor2 = double(pulse_amp*exp(-pow((currT-t_B_2)/(2*pulse_width),2))*cos(2*M_PI*pulse_freq*(currT-t_B_2)));
        // cout << "Current Time: " << currT << " Pulse Time 1: " << t_B_1 << " Pulse Time 2: " << t_B_2 << " Factor 1: " << factor1 << " Factor 2: " << factor2 << endl;
        for (size_t i=0; i< dim1; ++i){
            for (size_t j=0; j< dim2; ++j){
                for(size_t k=0; k< dim3;++k){
                    for (size_t l=0; l< N_ATOMS;++l){
                        size_t current_site_index = flatten_index(i,j,k,l);
                        driving_field[current_site_index] = field_in_1[l]*factor1 + field_in_2[l]*factor2;
                    }
                }
            }
        }
        ofstream pulse_info;
        pulse_info.open("pulse_t.txt", ios::app);
        pulse_info << "Current Time: " << currT << " Pulse Time 1: " << t_B_1 << " Pulse Time 2: " << t_B_2 << " " << driving_field[0][0] << " " << driving_field[0][1] << " " << driving_field[0][2] << endl;
        pulse_info.close();
    }

    double magnetization(const spin_config &current_spins, array<array<double, N>, N_ATOMS> &field_current){
        double mag = 0;
        for (size_t i=0; i< dim1; ++i){
            for (size_t j=0; j< dim2; ++j){
                for(size_t k=0; k< dim3;++k){
                    for (size_t l=0; l< N_ATOMS;++l){
                        size_t current_site_index = flatten_index(i,j,k,l);
                        mag += dot(current_spins[current_site_index], field_current[l]);
                    }
                }
            }
        }
        return mag/double(lattice_size);
    }

    array<double,N> magnetization_local(const spin_config &current_spins){
        array<double,N> mag = {{0}};
        for (size_t i=0; i< lattice_size; ++i){
            mag = mag + current_spins[i];
        }
        return mag/double(lattice_size);
    }

    void M_B_t(array<array<double,N>, N_ATOMS> &field_in, double t_B, double pulse_amp, double pulse_width, double pulse_freq, double T_start, double T_end, double step_size, string dir_name){
        spin_config spin_t = spins;
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        double tol = 1e-8;

        double currT = T_start;
        size_t count = 1;
        vector<double> time;
        time.push_back(currT);
        write_to_file_magnetization_local(dir_name + "/M_t.txt", magnetization_local(spin_t));
        // write_to_file_spin(dir_name + "/spin_t.txt", spin_t);
        // ofstream pulse_info;
        // pulse_info.open(dir_name + "/pulse_t.txt");
        cross_product_method cross_prod;
        if constexpr(N==3){
            cross_prod = cross_prod_SU2;
        }else if constexpr(N==8){
            cross_prod = cross_prod_SU3;
        }


        while(currT < T_end){
            set_pulse(currT, field_in, t_B, pulse_amp, pulse_width, pulse_freq);
            // double factor = double(pulse_amp*exp(-pow((currT+t_B)/(2*pulse_width),2))*cos(2*M_PI*pulse_freq*(currT+t_B)));
            // pulse_info << "Current Time: " << currT << " Pulse Time: " << t_B << " Factor: " << factor << " Field: " endl;
            spin_t = SSPRK53_step(step_size, spin_t, tol, cross_prod);
            write_to_file_magnetization_local(dir_name + "/M_t.txt", magnetization_local(spin_t));
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

    void M_BA_BB_t(array<array<double,N>, N_ATOMS> &field_in_1, double t_B_1, array<array<double,N>, N_ATOMS> &field_in_2, double t_B_2, double pulse_amp, double pulse_width, double pulse_freq, double T_start, double T_end, double step_size, string dir_name){
        // simulated_annealing(Temp_start, Temp_end, n_anneal, overrelaxation_rate);
        // write_to_file_pos(dir_name + "/pos.txt");
        // write_to_file_spin(dir_name + "/spin_t.txt", spins);
        cross_product_method cross_prod;
        if constexpr(N==3){
            cross_prod = cross_prod_SU2;
        }else if constexpr(N==8){
            cross_prod = cross_prod_SU3;
        }
        spin_config spin_t = spins;
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        double tol = 1e-12;
        double currT = T_start;
        size_t count = 1;
        vector<double> time;

        time.push_back(currT);
        write_to_file_magnetization_local(dir_name + "/M_t.txt", magnetization_local(spin_t));

        while(currT < T_end){

            set_two_pulse(currT, field_in_1, t_B_1, field_in_2, t_B_2, pulse_amp, pulse_width, pulse_freq);
            spin_t = SSPRK53_step(step_size, spin_t, tol, cross_prod);
            write_to_file_magnetization_local(dir_name + "/M_t.txt", magnetization_local(spin_t));
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
#endif // LATTICE_H