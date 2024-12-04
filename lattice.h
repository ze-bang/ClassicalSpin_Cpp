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

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim3>
class lattice
{   
    public:

    typedef array<array<float,N>,N_ATOMS*dim1*dim2*dim3> spin_config;

    UnitCell<N, N_ATOMS> UC;
    size_t lattice_size;
    spin_config  spins;
    array<array<float,3>, N_ATOMS*dim1*dim2*dim3> site_pos;
    //Lookup table for the lattice
    spin_config field;
    spin_config driving_field;
    array<array<array<float, N>, N>, N_ATOMS*dim1*dim2*dim3> onsite_interaction;

    array<vector<array<array<float, N>, N>>, N_ATOMS*dim1*dim2*dim3> bilinear_interaction;
    array<vector<array<array<array<float, N>, N>, N>>, N_ATOMS*dim1*dim2*dim3> trilinear_interaction;

    array<vector<size_t>, N_ATOMS*dim1*dim2*dim3> bilinear_partners;
    array<vector<array<size_t, 2>>, N_ATOMS*dim1*dim2*dim3> trilinear_partners;

    size_t num_bi;
    size_t num_tri;
    size_t num_gen;

    array<float,N> gen_random_spin(std::mt19937 &gen){
        array<float,N> temp_spin;
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
        return temp_spin;
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

    lattice(const UnitCell<N, N_ATOMS> *atoms): UC(*atoms){
        array<array<float,3>, N_ATOMS> basis;
        array<array<float,3>, 3> unit_vector;

        std::random_device rd;
        std::mt19937 gen(rd());

        lattice_size = dim1*dim2*dim3*N_ATOMS;
        basis = UC.lattice_pos;
        unit_vector = UC.lattice_vectors;


        for (size_t i=0; i< dim1; ++i){
            for (size_t j=0; j< dim2; ++j){
                for(size_t k=0; k< dim3;++k){
                    for (size_t l=0; l< N_ATOMS;++l){

                        size_t current_site_index = flatten_index(i,j,k,l);

                        site_pos[current_site_index]  = unit_vector[0]*int(i) + unit_vector[1]*int(j)  + unit_vector[2]*int(k)  + basis[l];
                        spins[current_site_index] = gen_random_spin(gen);
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
                            bilinear_interaction[partner].push_back(J.bilinear_interaction);
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

    void set_random_spin(size_t site_index){
        array<float, N> rand_spin = gen_random_spin();
        spins[site_index] = rand_spin;
    }

    void set_spin(size_t site_index, array<float, N> &spin_in){
        spins[site_index] = spin_in;
    }

    float site_energy(array<float, N> &spin_here, size_t site_index){
        float energy = 0.0;
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
    
    array<float, N>  get_local_field(size_t site_index){
        array<float,N> local_field;
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


    array<float, N>  get_local_field_lattice(size_t site_index, const spin_config &current_spin){
        array<float,N> local_field;
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
            array<float,N> local_field = get_local_field(i);
            float norm = sqrt(dot(local_field, local_field));
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
    
    array<float,N> gaussian_move(const array<float,N> &current_spin, std::mt19937 &gen, float sigma=60){
        array<float,N> new_spin;
        new_spin = current_spin + gen_random_spin(gen)*sigma;
        return new_spin/sqrt(dot(new_spin, new_spin));
    }

    void overrelaxation(std::mt19937 &gen){
        array<float,N> local_field;
        int i;
        float proj;
        size_t count = 0;
        while(count < lattice_size){
            i = random_int(0, lattice_size-1, gen);
            local_field = get_local_field(i);
            float norm = dot(local_field, local_field);
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

    float metropolis(float T, std::mt19937 &gen, bool gaussian=false, float sigma=60){
        float E, E_new, dE, r;
        int i;
        array<float,N> new_spin;
        int accept = 0;
        size_t count = 0;
        while(count < lattice_size){
            i = random_int(0, lattice_size-1, gen);
            E = site_energy(spins[i], i);
            if (gaussian){
                new_spin = gaussian_move(spins[i], gen, sigma);
            }
            else{
                new_spin = gen_random_spin(gen);
            }
            E_new = site_energy(new_spin, i);
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
    void write_to_file_magnetization_init(string filename, float towrite){
        ofstream myfile;
        myfile.open(filename);
        myfile << towrite << " ";
        myfile << endl;
        myfile.close();
    }
    void write_to_file_magnetization(string filename, float towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        myfile << towrite << " ";
        myfile << endl;
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

    void write_T_param(float T_end, size_t num_steps, string dir_name){
        ofstream myfile;
        myfile.open(dir_name);
        myfile << T_end << " " << num_steps << " " << lattice_size << endl;
        myfile.close();
    }

    void simulated_annealing(float T_start, float T_end, size_t n_therm, size_t n_anneal, size_t overrelaxation_rate, bool gaussian_move = false){
        std::random_device rd;
        std::mt19937 gen(rd());
        float T = T_start;
        float acceptance_rate = 0;
        float sigma = 40;
        cout << "Gaussian Move: " << gaussian_move << endl;
        while(T > T_end){
            float curr_accept = 0;
            for(size_t i = 0; i<n_anneal; ++i){
                if(overrelaxation_rate > 0){
                    overrelaxation(gen);
                    if (i%overrelaxation_rate == 0){
                        curr_accept += metropolis(T, gen, gaussian_move, sigma);
                    }
                }
                else{
                    curr_accept += metropolis(T, gen, gaussian_move, sigma);
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
    }

    void simulated_annealing_deterministic(float T_start, float T_end, size_t n_therm, size_t n_anneal, size_t n_deterministics, size_t overrelaxation_rate, string dir_name, bool gaussian_move = false){
        std::random_device rd;
        std::mt19937 gen(rd());
        simulated_annealing(T_start, T_end, n_therm, n_anneal, overrelaxation_rate, gaussian_move);
        for(size_t i = 0; i<n_deterministics; ++i){
            deterministic_sweep(gen);
        }   

        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            write_to_file_spin(dir_name + "/spin.txt", spins);
            write_to_file_pos(dir_name + "/pos.txt");
        }
    }

    void landau_lifshitz_ode_int(spin_config &current_spin, spin_config &dS, array<float,N> (*cross_prod)(const array<float, N>, const array<float, N>), const double /* t */){
        for(size_t i = 0; i<lattice_size; ++i){
            dS[i] = cross_prod(get_local_field_lattice(i, current_spin), current_spin[i]);
        }
    }

    spin_config landau_lifshitz(const spin_config &current_spin, array<float,N> (*cross_prod)(const array<float, N>, const array<float, N>)){
        spin_config dS;
        #pragma omp simd
        for(size_t i = 0; i<lattice_size; ++i){
            dS[i] = cross_prod(get_local_field_lattice(i, current_spin), current_spin[i]);
        }
        return dS;
    }

    spin_config RK4_step(const float step_size, const spin_config &curr_spins, const double tol, array<float,N> (*cross_prod)(const array<float, N>, const array<float, N>)){
        spin_config k1 = landau_lifshitz(curr_spins,cross_prod);
        spin_config lval_k2 = curr_spins + k1*(0.5*step_size);
        spin_config k2 = landau_lifshitz(lval_k2,cross_prod));
        spin_config lval_k3 = curr_spins + k2*(0.5*step_size);
        spin_config k3 = landau_lifshitz(lval_k3,cross_prod));
        spin_config lval_k4 = curr_spins + k3*step_size;
        spin_config k4 = landau_lifshitz(lval_k4,cross_prod));
        spin_config new_spins  = curr_spins + (k1+ k2 * 2 + k3 * 2 + k4)*(step_size/6);
        return new_spins;
    }

    void print_2D(const spin_config &a){
        for(size_t i = 0; i<N_ATOMS*dim1*dim2*dim3; ++i){
            for(size_t j = 0; j<N; ++j){
                cout << a[i][j] << " ";
            }
            cout << endl;
        }
    }

    spin_config RK45_step(float &step_size, const spin_config &curr_spins, const double tol, array<float,N> (*cross_prod)(const array<float, N>, const array<float, N>)){
        spin_config k1 = landau_lifshitz(curr_spins)*step_size;
        spin_config k2 = landau_lifshitz(curr_spins + k1*(1.0/4.0))*step_size;
        spin_config k3 = landau_lifshitz(curr_spins + k1*(3.0/32.0) + k2*(9.0/32.0))*step_size;
        spin_config k4 = landau_lifshitz(curr_spins + k1*(1932.0/2197.0) + k2*(-7200.0/2197.0) + k3*(7296.0/2197.0))*step_size;
        spin_config k5 = landau_lifshitz(curr_spins + k1*(439.0/216.0) + k2*(-8.0) + k3*(3680.0/513.0) + k4*(-845.0/4104.0))*step_size;
        spin_config k6 = landau_lifshitz(curr_spins + k1*(-8.0/27.0) + k2*(2.0) + k3*(-3544.0/2565.0)+ k4*(1859.0/4104.0)+ k5*(-11.0/40.0))*step_size;

        spin_config y = curr_spins + k1*(25.0/216.0) + k3*(1408.0/2565.0) + k4*(2197.0/4101.0) - k5*(1.0/5.0);
        spin_config z = curr_spins + k1*(16.0/135.0) + k3*(6656.0/12825.0) + k4*(28561.0/56430.0) - k5*(9.0/50.0) + k6*(2.0/55.0);

        double error = norm_average_2D(z-y);
        step_size *= 0.9*pow(tol/error, 0.2);
        if (error < tol){
            return z;
        }
        else{
            return RK45_step(step_size, curr_spins, tol);
        }
        return z;
    }

    spin_config RK45_step_fixed(const float &step_size, const spin_config &curr_spins, const double tol, array<float,N> (*cross_prod)(const array<float, N>, const array<float, N>)){
        spin_config k1 = landau_lifshitz(curr_spins,cross_prod)*step_size;
        spin_config k2 = landau_lifshitz(curr_spins + k1*(1.0/4.0),cross_prod)*step_size;
        spin_config k3 = landau_lifshitz(curr_spins + k1*(3.0/32.0) + k2*(9.0/32.0),cross_prod)*step_size;
        spin_config k4 = landau_lifshitz(curr_spins + k1*(1932.0/2197.0) + k2*(-7200.0/2197.0) + k3*(7296.0/2197.0),cross_prod)*step_size;
        spin_config k5 = landau_lifshitz(curr_spins + k1*(439.0/216.0) + k2*(-8.0) + k3*(3680.0/513.0) + k4*(-845.0/4104.0),cross_prod)*step_size;
        spin_config k6 = landau_lifshitz(curr_spins + k1*(-8.0/27.0) + k2*(2.0) + k3*(-3544.0/2565.0)+ k4*(1859.0/4104.0)+ k5*(-11.0/40.0),cross_prod)*step_size;
        spin_config z = curr_spins + k1*(16.0/135.0) + k3*(6656.0/12825.0) + k4*(28561.0/56430.0) - k5*(9.0/50.0) + k6*(2.0/55.0);
        return z;
    }

    spin_config euler_step(const float step_size, spin_config &curr_spins, const double tol, array<float,N> (*cross_prod)(const array<float, N>, const array<float, N>)){
        spin_config dS = landau_lifshitz(curr_spins, cross_prod);
        spin_config new_spins = curr_spins + dS*step_size;
        return new_spins;
    }


    void molecular_dynamics(float Temp_start, float Temp_end, size_t n_therm, size_t n_anneal, size_t overrelaxation_rate, float T_start, float T_end, float step_size, string dir_name, , bool gaussian_move = false){
        // simulated_annealing(Temp_start, Temp_end, n_therm, n_anneal, overrelaxation_rate, gaussian_move);
        if(dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        write_to_file_pos(dir_name + "/pos.txt");
        write_to_file_spin(dir_name + "/spin_t.txt", spins);
        spin_config spin_t = spins;

        double tol = 1e-12;

        int check_frequency = 10;
        float currT = T_start;
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

    void set_pulse(float currT, const array<array<float,N>, N_ATOMS> &field_in, float t_B, float pulse_amp, float pulse_width, float pulse_freq){
        float factor = float(pulse_amp*exp(-pow((currT-t_B)/(2*pulse_width),2))*cos(2*M_PI*pulse_freq*(currT-t_B)));
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
        for (size_t i=0; i< dim1; ++i){
            for (size_t j=0; j< dim2; ++j){
                for(size_t k=0; k< dim3;++k){
                    for (size_t l=0; l< N_ATOMS;++l){
                        size_t current_site_index = flatten_index(i,j,k,l);
                        driving_field[current_site_index] = {0};
                    }
                }
            }
        }
    }

    void set_two_pulse(float currT, const array<array<float,N>, N_ATOMS> &field_in_1, float t_B_1, const array<array<float,N>, N_ATOMS> &field_in_2, float t_B_2, float pulse_amp, float pulse_width, float pulse_freq){
        float factor1 = float(pulse_amp*exp(-pow((currT-t_B_1)/(2*pulse_width),2))*cos(2*M_PI*pulse_freq*(currT-t_B_1)));
        float factor2 = float(pulse_amp*exp(-pow((currT-t_B_2)/(2*pulse_width),2))*cos(2*M_PI*pulse_freq*(currT-t_B_2)));
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
    }

    float magnetization(const spin_config &current_spins, array<array<float, N>, N_ATOMS> &field_current){
        float mag = 0;
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
        return mag/float(lattice_size);
    }

    void M_B_t(array<array<float,N>, N_ATOMS> &field_in, float t_B, float pulse_amp, float pulse_width, float pulse_freq, float T_start, float T_end, float step_size, string dir_name){
        spin_config spin_t = spins;
        if(dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        double tol = 1e-8;

        float currT = T_start;
        size_t count = 1;
        vector<float> time;
        time.push_back(currT);
        write_to_file_magnetization_init(dir_name + "/M_t.txt", magnetization(spin_t-spins, field_in));
        // write_to_file_spin(dir_name + "/spin_t.txt", spin_t);
        // ofstream pulse_info;
        // pulse_info.open(dir_name + "/pulse_t.txt");
        

        while(currT < T_end){
            set_pulse(currT, field_in, t_B, pulse_amp, pulse_width, pulse_freq);
            // float factor = float(pulse_amp*exp(-pow((currT+t_B)/(2*pulse_width),2))*cos(2*M_PI*pulse_freq*(currT+t_B)));
            // pulse_info << "Current Time: " << currT << " Pulse Time: " << t_B << " Factor: " << factor << " Field: " endl;
            spin_t = RK45_step_fixed(step_size, spin_t, tol);
            write_to_file_magnetization(dir_name + "/M_t.txt", magnetization(spin_t-spins, field_in));
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

    void M_BA_BB_t(array<array<float,N>, N_ATOMS> &field_in_1, float t_B_1, array<array<float,N>, N_ATOMS> &field_in_2, float t_B_2, float pulse_amp, float pulse_width, float pulse_freq, float T_start, float T_end, float step_size, string dir_name){
        // simulated_annealing(Temp_start, Temp_end, n_therm, n_anneal, overrelaxation_rate);
        // write_to_file_pos(dir_name + "/pos.txt");
        // write_to_file_spin(dir_name + "/spin_t.txt", spins);
        spin_config spin_t = spins;
        if(dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        double tol = 1e-12;

        float currT = T_start;
        size_t count = 1;
        vector<float> time;

        time.push_back(currT);
        write_to_file_magnetization_init(dir_name + "/M_t.txt", magnetization(spin_t-spins, field_in_1));

        while(currT < T_end){

            set_two_pulse(currT, field_in_1, t_B_1, field_in_2, t_B_2, pulse_amp, pulse_width, pulse_freq);
            spin_t = RK45_step_fixed(step_size, spin_t, tol);
            write_to_file_magnetization(dir_name + "/M_t.txt", magnetization(spin_t-spins, field_in_1));
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