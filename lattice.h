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
    UnitCell<N, N_ATOMS> UC;
    int lattice_size;
    array<array<float,N>, N_ATOMS*dim1*dim2*dim3>  spins;
    array<array<float,3>, N_ATOMS*dim1*dim2*dim3> site_pos;
    //Lookup table for the lattice
    array<array<float,N>, N_ATOMS*dim1*dim2*dim3> field;



    array<vector<array<array<float, N>, N>>, N_ATOMS*dim1*dim2*dim3> bilinear_interaction;
    array<vector<array<array<array<float, N>, N>, N>>, N_ATOMS*dim1*dim2*dim3> trilinear_interaction;

    array<vector<int>, N_ATOMS*dim1*dim2*dim3> bilinear_partners;
    array<vector<array<int, 2>>, N_ATOMS*dim1*dim2*dim3> trilinear_partners;

    int num_bi;
    int num_tri;

    array<float,N> gen_random_spin(std::mt19937 &gen){
        array<float,N> temp_spin;
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

        return temp_spin;
    }

    int flatten_index(int i, int j, int k, int l){
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

    int flatten_index_periodic_boundary(int i, int j, int k, int l){
        return periodic_boundary(i, dim1)*dim2*dim3*N_ATOMS+ periodic_boundary(j, dim2)*dim3*N_ATOMS+ periodic_boundary(k, dim3)*N_ATOMS + l;
    }

    lattice(UnitCell<N, N_ATOMS> *atoms): UC(*atoms){
        array<array<float,3>, N_ATOMS> basis;
        array<array<float,3>, 3> unit_vector;

        std::random_device rd;
        std::mt19937 gen(rd());

        lattice_size = dim1*dim2*dim3*N_ATOMS;
        basis = UC.lattice_pos;
        unit_vector = UC.lattice_vectors;


        for (int i=0; i<dim1; ++i){
            for (int j=0; j< dim2; ++j){
                for(int k=0; k<dim3;++k){
                    for (int l=0; l<N_ATOMS;++l){

                        int current_site_index = flatten_index(i,j,k,l);

                        site_pos[current_site_index]  = unit_vector[0]*i + unit_vector[1]*j + unit_vector[2]*k + basis[l];
                        spins[current_site_index] = gen_random_spin(gen);
                        field[current_site_index] = UC.field[l];
                        
                        auto bilinear_matched = UC.bilinear_interaction.equal_range(l);
                        int count = 0;
                        for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m){
                            bilinear<N> J = m->second;
                            int partner = flatten_index_periodic_boundary(i+J.offset[0], j+J.offset[1], k+J.offset[2], J.partner);
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
                            int partner1 = flatten_index_periodic_boundary(i+J.offset1[0], j+J.offset1[1], k+J.offset1[2], J.partner1);
                            int partner2 = flatten_index_periodic_boundary(i+J.offset2[0], j+J.offset2[1], k+J.offset2[2], J.partner2);
                            
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
    };

    void set_random_spin(int site_index){
        array<float, N> rand_spin = gen_random_spin();
        spins[site_index] = rand_spin;
    }

    void set_spin(int site_index, array<float, N> &spin_in){
        spins[site_index] = spin_in;
    }

    float site_energy(array<float, N> &spin_here, int site_index){
        float energy = 0.0;
        energy += dot(spin_here, field[site_index]);
        #pragma omp simd
        for (int i=0; i<num_bi; ++i) {
            energy += contract(spin_here, bilinear_interaction[site_index][i], spins[bilinear_partners[site_index][i]]);
        }
        for (int i=0; i < num_tri; ++i){
            energy += contract_trilinear(trilinear_interaction[site_index][i], spin_here, spins[trilinear_partners[site_index][i][0]], spins[trilinear_partners[site_index][i][1]]);
        }
        return energy;
    }
    
    array<float, N>  get_local_field(int site_index){
        array<float,N> local_field;
        local_field = {0};
        #pragma omp simd
        for (int i=0; i<num_bi; ++i) {
            local_field = local_field + multiply(bilinear_interaction[site_index][i], spins[bilinear_partners[site_index][i]]);
        }
        for (int i=0; i < num_tri; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction[site_index][i], spins[trilinear_partners[site_index][i][0]], spins[trilinear_partners[site_index][i][1]]);
        }
        return local_field+field[site_index];
    }


    array<float, N>  get_local_field_lattice(int site_index,const array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &current_spin){
        array<float,N> local_field;
        local_field = {0};
        #pragma omp simd
        for (int i=0; i<num_bi; ++i) {
            local_field = local_field + multiply(bilinear_interaction[site_index][i], current_spin[bilinear_partners[site_index][i]]);
        }
        for (int i=0; i < num_tri; ++i){
            local_field = local_field + contract_trilinear_field(trilinear_interaction[site_index][i], current_spin[trilinear_partners[site_index][i][0]], current_spin[trilinear_partners[site_index][i][1]]);
        }
        return local_field+field[site_index];
    }


    void deterministic_sweep(){
        for(int i = 0; i<lattice_size; ++i){
            array<float,N> local_field = get_local_field(i);
            float norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                for(int j=0; j < N; ++j){
                    spins[i][j] = -local_field[j]/norm;
                }
            }
        }
    }
    
    void overrelaxation(){
        array<float,N> local_field;
        float proj;
        for(int i = 0; i<lattice_size; ++i){
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
            E = site_energy(spins[i], i);
            new_spin = gen_random_spin(gen);
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
    
    void write_to_file_spin(string filename, array<array<float,N>, N_ATOMS*dim1*dim2*dim3> towrite){
        ofstream myfile;
        myfile.open(filename);
        for(int i = 0; i<lattice_size; ++i){
            for(int j = 0; j<N; ++j){
                myfile << towrite[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_to_file(string filename, array<array<float,N>, N_ATOMS*dim1*dim2*dim3> towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        for(int i = 0; i<lattice_size; ++i){
            for(int j = 0; j<N; ++j){
                myfile << towrite[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }


    void write_to_file_pos(string filename){
        ofstream myfile;
        myfile.open(filename);
        for(int i = 0; i<lattice_size; ++i){
            for(int j = 0; j<3; ++j){
                myfile << site_pos[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_T_param(float T_end, int num_steps, string dir_name){
        ofstream myfile;
        myfile.open(dir_name);
        myfile << T_end << " " << num_steps << " " << lattice_size << endl;
        myfile.close();
    }

    void simulated_annealing(float T_start, float T_end, int n_therm, int n_anneal, int n_deterministics, int overrelaxation_rate, string dir_name){
        std::random_device rd;
        std::mt19937 gen(rd());
        float T = T_start;
        while(T > T_end){
            float curr_accept = 0;
            for(int i = 0; i<n_anneal; ++i){
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
        for(int i = 0; i<n_deterministics; ++i){
            deterministic_sweep();
        }   

        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            write_to_file_spin(dir_name + "/spin.txt", spins);
            write_to_file_pos(dir_name + "/pos.txt");
        }
    }

    void landau_lifshitz_ode_int(array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &current_spin, array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &dS, const double /* t */){
        for(int i = 0; i<lattice_size; ++i){
            dS[i] = cross_prod_SU2(get_local_field_lattice(i, current_spin), current_spin[i]);
        }
    }

    array<array<float,N>,N_ATOMS*dim1*dim2*dim3> landau_lifshitz(const array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &current_spin){
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> dS;
        #pragma omp simd
        for(int i = 0; i<lattice_size; ++i){
            dS[i] = cross_prod_SU2(get_local_field_lattice(i, current_spin), current_spin[i]);
        }
        return dS;
    }

    array<array<float,N>,N_ATOMS*dim1*dim2*dim3> RK4_step(const float step_size, const array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &curr_spins){
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k1 = landau_lifshitz(curr_spins);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> lval_k2 = curr_spins + k1*(0.5*step_size);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k2 = landau_lifshitz(lval_k2);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> lval_k3 = curr_spins + k2*(0.5*step_size);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k3 = landau_lifshitz(lval_k3);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> lval_k4 = curr_spins + k3*step_size;
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k4 = landau_lifshitz(lval_k4);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> new_spins  = curr_spins + (k1+ k2 * 2 + k3 * 2 + k4)*(step_size/6);
        return new_spins;
    }

    void print_2D(const array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &a){
        for(int i = 0; i<N_ATOMS*dim1*dim2*dim3; ++i){
            for(int j = 0; j<N; ++j){
                cout << a[i][j] << " ";
            }
            cout << endl;
        }
    }

    array<array<float,N>,N_ATOMS*dim1*dim2*dim3> RK45_step(float &step_size, const array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &curr_spins, const double tol){
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k1 = landau_lifshitz(curr_spins)*step_size;
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k2 = landau_lifshitz(curr_spins + k1*(1.0/4.0))*step_size;
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k3 = landau_lifshitz(curr_spins + k1*(3.0/32.0) + k2*(9.0/32.0))*step_size;
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k4 = landau_lifshitz(curr_spins + k1*(1932.0/2197.0) + k2*(-7200.0/2197.0) + k3*(7296.0/2197.0))*step_size;
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k5 = landau_lifshitz(curr_spins + k1*(439.0/216.0) + k2*(-8.0) + k3*(3680.0/513.0) + k4*(-845.0/4104.0))*step_size;
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> k6 = landau_lifshitz(curr_spins + k1*(-8.0/27.0) + k2*(2.0) + k3*(-3544.0/2565.0)+ k4*(1859.0/4104.0)+ k5*(-11.0/40.0))*step_size;

        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> y = curr_spins + k1*(25.0/216.0) + k3*(1408.0/2565.0) + k4*(2197.0/4101.0) - k5*(1.0/5.0);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> z = curr_spins + k1*(16.0/135.0) + k3*(6656.0/12825.0) + k4*(28561.0/56430.0) - k5*(9.0/50.0) + k6*(2.0/55.0);

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


    array<array<float,N>,N_ATOMS*dim1*dim2*dim3> euler_step(const float step_size, array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &curr_spins){
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> dS = landau_lifshitz(curr_spins);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> new_spins = curr_spins + dS*step_size;
        return new_spins;
    }

    void euler_step_ode_int(const float step_size, array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &curr_spins, array<array<float,N>,N_ATOMS*dim1*dim2*dim3> &dS){
        landau_lifshitz_ode_int(curr_spins, dS, 0);
        curr_spins = curr_spins + dS*step_size;
    }


    void molecular_dynamics_SU2(float Temp_start, float Temp_end, int n_therm, int n_anneal, int overrelaxation_rate, float T_end, float step_size, string dir_name){
        std::random_device rd;
        std::mt19937 gen(rd());
        float curr_accept;
        filesystem::create_directory(dir_name);
        float T = Temp_start;

        while(T > Temp_end){
            curr_accept = 0;
            for(int i = 0; i<n_anneal; ++i){
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
        write_to_file_pos(dir_name + "/pos.txt");
        write_to_file_spin(dir_name + "/spin_t.txt", spins);
        array<array<float,N>,N_ATOMS*dim1*dim2*dim3> spin_t = spins;

        double tol = 1e-8;

        int check_frequency = 10;
        float currT = 0;
        int count = 1;
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
        for(int i = 0; i<count; ++i){
            time_sections << time[i] << endl;
        }
        time_sections.close();
    }
};
#endif // LATTICE_H