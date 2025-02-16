#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include "unitcell.h"
#include "lattice.h"
#include "mixed_lattice.h"
// #include "lattice_strain_field.h"
#include <iostream>
#include <mpi.h>
#include "simple_linear_alg.h"

void simulated_annealing_honeycomb(double T_start, double T_end, double K, double Gamma, double Gammap, double h, string dir="", bool deterministic=false){
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;
    array<array<double,3>, 3> Jx = {{{K,Gammap,Gammap},{Gammap,0,Gamma},{Gammap,Gamma,0}}};
    array<array<double,3>, 3> Jy = {{{0,Gammap,Gamma},{Gammap,K,Gammap},{Gamma,Gammap,0}}};
    array<array<double,3>, 3> Jz = {{{0,Gamma,Gammap},{Gamma,0,Gammap},{Gammap,Gammap,K}}};

    array<double, 3> field = {h/double(sqrt(3)),h/double(sqrt(3)),h/double(sqrt(3))};
    
    cout << "Setting up honeycomb lattice with parameters: " << endl;
    cout << "Jx: " << Jx[0][0] << " " << Jx[0][1] << " " << Jx[0][2] << endl;
    cout << "Jy: " << Jy[1][0] << " " << Jy[1][1] << " " << Jy[1][2] << endl;
    cout << "Jz: " << Jz[2][0] << " " << Jz[2][1] << " " << Jz[2][2] << endl;
    cout << "Field: " << field[0] << " " << field[1] << " " << field[2] << endl;

    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    lattice<3, 2, 20, 20, 1> MC(&atoms);
    if (deterministic == false){
        MC.simulated_annealing(T_start, T_end, 10000, 0, false, dir);
    }
    else{
        MC.simulated_annealing_deterministic(T_start, T_end, 10000, 10000, 0, dir);
    }
}

void parallel_tempering_honeycomb(double T_start, double T_end, double K, double Gamma, double Gammap, double h, string dir, const vector<int> &rank_to_write){
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;
    array<array<double,3>, 3> Jx = {{{K,Gammap,Gammap},{Gammap,0,Gamma},{Gammap,Gamma,0}}};
    array<array<double,3>, 3> Jy = {{{0,Gammap,Gamma},{Gammap,K,Gammap},{Gamma,Gammap,0}}};
    array<array<double,3>, 3> Jz = {{{0,Gamma,Gammap},{Gamma,0,Gammap},{Gammap,Gammap,K}}};


    array<double, 3> field = {h/double(sqrt(3)),h/double(sqrt(3)),h/double(sqrt(3))};
    
    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<double> temps = logspace(log10(T_start), log10(T_end), size);


    lattice<3, 2, 20, 20, 1> MC(&atoms);
    MC.parallel_tempering(temps, 1e6, 1e6, 10, 50, 2e3, dir, rank_to_write);

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

void MD_kitaev_honeycomb(size_t num_trials, double K, double Gamma, double Gammap, double h, string dir){
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;
    array<array<double,3>, 3> Jx = {{{K,Gammap,Gammap},{Gammap,0,Gamma},{Gammap,Gamma,0}}};
    array<array<double,3>, 3> Jy = {{{0,Gammap,Gamma},{Gammap,K,Gammap},{Gamma,Gammap,0}}};
    array<array<double,3>, 3> Jz = {{{0,Gamma,Gammap},{Gamma,0,Gammap},{Gammap,Gammap,K}}};


    array<double, 3> field = {h/double(sqrt(3)),h/double(sqrt(3)),h/double(sqrt(3))};
    
    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    for(size_t i=0; i<num_trials;++i){

        lattice<3, 2, 12, 12, 1> MC(&atoms);
        MC.simulated_annealing(1, 1e-4, 100000, 0, true);
        // std::random_device rd;
        // std::mt19937 gen(rd());
        // for (size_t i = 0; i<100000; ++i){
        //     MC.deterministic_sweep(gen);
        // }
        MC.molecular_dynamics(1,1e-3, 1000000, 100, 0, 600, 0.25, dir+"/"+std::to_string(i));
    }
}

// kitaev local xyz {1,-1,-1} / {-1, 1, -1} / {-1, -1, 1}
void nonlinearspectroscopy_kitaev_honeycomb(double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double K, double Gamma, double Gammap, double h, string dir, bool T_zero){
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;
    array<array<double,3>, 3> Jx = {{{K,Gammap,Gammap},{Gammap,0,Gamma},{Gammap,Gamma,0}}};
    array<array<double,3>, 3> Jy = {{{0,Gammap,Gamma},{Gammap,K,Gammap},{Gamma,Gammap,0}}};
    array<array<double,3>, 3> Jz = {{{0,Gamma,Gammap},{Gamma,0,Gammap},{Gammap,Gammap,K}}};

    array<double, 3> field = {h/double(sqrt(3)),h/double(sqrt(3)),h/double(sqrt(3))};


    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    // array<array<double, 3>,2> field_drive = {{{-1/sqrt(3), -1/sqrt(3), 1/sqrt(3)},{-1/sqrt(3), -1/sqrt(3), 1/sqrt(3)}}};
    array<array<double, 3>,2> field_drive = {{{0,0,1},{0,0,1}}};

    double pulse_amp = 0.1;
    double pulse_width = 0.38;
    double pulse_freq = 0.33;

    int T_steps = abs(int((T_end-T_start)/T_step_size))+1;
    int tau_steps = abs(int((tau_end-tau_start)/tau_step_size))+1;
    tau_step_size = tau_end - tau_start < 0 ? - abs(tau_step_size) : abs(tau_step_size);
    T_step_size = T_end - T_start < 0 ? - abs(T_step_size) : abs(T_step_size);
    lattice<3, 2, 20, 20, 1> MC(&atoms);
    MC.simulated_annealing(Temp_start, Temp_end, 10000, 0, true);

    if (T_zero){
        for (size_t i = 0; i<100000; ++i){
            MC.deterministic_sweep();
        }
    }
    MC.write_to_file_pos(dir+"/pos.txt");
    MC.write_to_file_spin(dir+"/spin_0.txt", MC.spins);

    filesystem::create_directory(dir+"/M_time_0");
    MC.M_B_t(field_drive, 0.0, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_0/M0");

    ofstream run_param;
    run_param.open(dir + "/param.txt");
    run_param << tau_start << " " << tau_end << " " << tau_steps  << " " << T_start << " " << T_end << " " << T_steps << " " << K << " " << h << endl;
    run_param.close();

    double current_tau = tau_start;

    for(int i=0; i< tau_steps;++i){
        filesystem::create_directory(dir+"/M_time_"+ std::to_string(i));
        cout << "Time: " << current_tau << endl;
        MC.M_B_t(field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i) + "/M1");
        MC.M_BA_BB_t(field_drive, 0.0, field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i)+ "/M01");
        current_tau += tau_step_size;
    }
}


void full_nonlinearspectroscopy_kitaev_honeycomb(size_t num_trials, double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double K, double Gamma, double Gammap, double h, string dir, bool T_zero){
    int initialized;

    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    filesystem::create_directory(dir);

    for(size_t i = 0; i < num_trials; ++i){
        nonlinearspectroscopy_kitaev_honeycomb(Temp_start, Temp_end, tau_start, tau_end, tau_step_size, T_start, T_end, T_step_size, K, Gamma, Gammap, h, dir+std::to_string(i), T_zero);
    }
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

void simulated_annealing_TmFeO3_Fe(int num_trials, double T_start, double T_end, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double h, const array<double,3> &fielddir, string dir, bool T_zero=false){
    filesystem::create_directory(dir);
    TmFeO3_Fe<3> Fe_atoms;
    TmFeO3_Tm<8> Tm_atoms;

    array<array<double, 3>, 3> Ja = {{{Jai, 0, 0}, {0, Jai, 0}, {0, 0, Jai}}};
    array<array<double, 3>, 3> Jb = {{{Jbi, 0, 0}, {0, Jbi, 0}, {0, 0, Jbi}}};
    array<array<double, 3>, 3> Jc = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};

    array<array<double, 3>, 3> J2a = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    array<array<double, 3>, 3> J2b = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    array<array<double, 3>, 3> J2c = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};

    array<double, 9> K = {{Ka, 0, 0, 0, 0, 0, 0, 0, Kc}};

    array<array<double, 3>,3> D = {{{0, D2, -D1}, {-D2, 0, 0}, {D1, 0, 0}}};
    //In plane interactions
    //Nearest Neighbours
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {1,-1,0});
    //Next Nearest Neighbours
    Fe_atoms.set_bilinear_interaction(J2a, 0, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 0, 0, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 1, 1, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 1, 1, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 2, 2, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 2, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 3, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 3, 3, {0,1,0});
    //Out of plane interaction
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, {0,0,1});

    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,1,0});

    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,1,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,1,1});

    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,-1,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,-1,1});

    //single ion anisotropy
    Fe_atoms.set_onsite_interaction(K, 0);
    Fe_atoms.set_onsite_interaction(K, 1);
    Fe_atoms.set_onsite_interaction(K, 2);
    Fe_atoms.set_onsite_interaction(K, 3);

    //Dzyaloshinskii-Moriya interaction
    Fe_atoms.set_bilinear_interaction(D, 0, 0, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 0, 0, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 1, 1, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 1, 1, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 2, 2, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 2, 2, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 3, 3, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 3, 3, {1,-1,0});

    Fe_atoms.set_field(fielddir*h, 0);
    Fe_atoms.set_field(fielddir*h, 1);
    Fe_atoms.set_field(fielddir*h, 2);
    Fe_atoms.set_field(fielddir*h, 3);

    for(size_t i = 0; i < num_trials; ++i){
        lattice<3, 4, 12, 12, 12> MC_FE(&Fe_atoms, 2.5);
        MC_FE.simulated_annealing(T_start, T_end, 10000, 0, false);
        if (T_zero){
            // std::random_device rd;
            // std::mt19937 gen(rd());
            for(size_t i = 0; i<100000; ++i){
                MC_FE.deterministic_sweep();
            }   
        }
        MC_FE.write_to_file_pos(dir+"/pos.txt");
        MC_FE.write_to_file_spin(dir+"/spin"+ to_string(i) +".txt", MC_FE.spins);
    }
}


void MD_TmFeO3_Fe(int num_trials, double T_start, double T_end, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double h, const array<double,3> &fielddir, string dir){
    filesystem::create_directory(dir);
    TmFeO3_Fe<3> Fe_atoms;
    TmFeO3_Tm<8> Tm_atoms;

    array<array<double, 3>, 3> Ja = {{{Jai, 0, 0}, {0, Jai, 0}, {0, 0, Jai}}};
    array<array<double, 3>, 3> Jb = {{{Jbi, 0, 0}, {0, Jbi, 0}, {0, 0, Jbi}}};
    array<array<double, 3>, 3> Jc = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};

    array<array<double, 3>, 3> J2a = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    array<array<double, 3>, 3> J2b = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    array<array<double, 3>, 3> J2c = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};

    array<double, 9> K = {{Ka, 0, 0, 0, 0, 0, 0, 0, Kc}};

    array<array<double, 3>,3> D = {{{0, D2, -D1}, {-D2, 0, 0}, {D1, 0, 0}}};
    //In plane interactions
    //Nearest Neighbours
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {1,-1,0});
    //Next Nearest Neighbours
    Fe_atoms.set_bilinear_interaction(J2a, 0, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 0, 0, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 1, 1, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 1, 1, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 2, 2, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 2, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 3, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 3, 3, {0,1,0});
    //Out of plane interaction
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, {0,0,1});

    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,1,0});

    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,1,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,1,1});

    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,-1,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,-1,1});

    //single ion anisotropy
    Fe_atoms.set_onsite_interaction(K, 0);
    Fe_atoms.set_onsite_interaction(K, 1);
    Fe_atoms.set_onsite_interaction(K, 2);
    Fe_atoms.set_onsite_interaction(K, 3);

    //Dzyaloshinskii-Moriya interaction
    Fe_atoms.set_bilinear_interaction(D, 0, 0, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 0, 0, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 1, 1, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 1, 1, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 2, 2, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 2, 2, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 3, 3, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 3, 3, {1,-1,0});

    Fe_atoms.set_field(fielddir*h, 0);
    Fe_atoms.set_field(fielddir*h, 1);
    Fe_atoms.set_field(fielddir*h, 2);
    Fe_atoms.set_field(fielddir*h, 3);

    for(size_t i = 0; i < num_trials; ++i){
        lattice<3, 4, 8, 8, 8> MC_FE(&Fe_atoms, 2.5);
        MC_FE.simulated_annealing(T_start, T_end, 10000, 0, false);
        MC_FE.molecular_dynamics(T_start, T_end, 10000, 0, 0, 200/Jai, 5e-2/Jai, dir+"/"+std::to_string(i));
    }
}

void MD_TmFeO3(int num_trials, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double xii, double h, const array<double,3> &fielddir, double e1, double e2, string dir){
    filesystem::create_directory(dir);
    TmFeO3_Fe<3> Fe_atoms;
    TmFeO3_Tm<8> Tm_atoms;

    array<array<double, 3>, 3> Ja = {{{Jai, 0, 0}, {0, Jai, 0}, {0, 0, Jai}}};
    array<array<double, 3>, 3> Jb = {{{Jbi, 0, 0}, {0, Jbi, 0}, {0, 0, Jbi}}};
    array<array<double, 3>, 3> Jc = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};

    array<array<double, 3>, 3> J2a = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    array<array<double, 3>, 3> J2b = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    array<array<double, 3>, 3> J2c = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};

    array<double, 9> K = {{Ka, 0, 0, 0, 0, 0, 0, 0, Kc}};

    array<array<double, 3>,3> D = {{{0, D2, -D1}, {-D2, 0, 0}, {D1, 0, 0}}};
    //In plane interactions

    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {1,-1,0});
    //Next Nearest Neighbour
    Fe_atoms.set_bilinear_interaction(J2a, 0, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 0, 0, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 1, 1, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 1, 1, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 2, 2, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 2, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 3, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 3, 3, {0,1,0});
    //Out of plane interaction
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, {0,0,1});

    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,1,0});

    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,1,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,1,1});

    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,-1,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,-1,1});

    //single ion anisotropy
    Fe_atoms.set_onsite_interaction(K, 0);
    Fe_atoms.set_onsite_interaction(K, 1);
    Fe_atoms.set_onsite_interaction(K, 2);
    Fe_atoms.set_onsite_interaction(K, 3);

    //Dzyaloshinskii-Moriya interaction
    Fe_atoms.set_bilinear_interaction(D, 0, 0, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 0, 0, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 1, 1, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 1, 1, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 2, 2, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 2, 2, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 3, 3, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 3, 3, {1,-1,0});

    Fe_atoms.set_field(fielddir*h, 0);
    Fe_atoms.set_field(fielddir*h, 1);
    Fe_atoms.set_field(fielddir*h, 2);
    Fe_atoms.set_field(fielddir*h, 3);

    //Tm atoms

    Tm_atoms.set_field({0,0,e1,0,0,0,0,e2}, 0);
    Tm_atoms.set_field({0,0,e1,0,0,0,0,e2}, 1);
    Tm_atoms.set_field({0,0,e1,0,0,0,0,e2}, 2);
    Tm_atoms.set_field({0,0,e1,0,0,0,0,e2}, 3);


    TmFeO3<3, 8> TFO(&Fe_atoms, &Tm_atoms);

    array<array<array<double,3>,3>,8> xi = {{{0}}};

    xi[0] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
    xi[1] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};

    ///////////////////
    TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {1,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,1,0});
    TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {1,0,0}, {0,1,0});

    TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {0,0,1}, {0,0,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {1,0,1}, {0,0,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {0,0,1}, {0,1,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {1,0,1}, {0,1,1});

    TFO.set_mix_trilinear_interaction(xi, 2, 0, 3, {0,0,0}, {0,0,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 0, 3, {1,0,0}, {1,0,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 1, 2, {0,0,0}, {0,0,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 1, 2, {0,1,0}, {0,1,1});
    //////////////////
    TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,0,0}, {0,1,0});
    TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,1,0}, {0,1,0});
    TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,0,0}, {-1,1,0});
    TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,1,0}, {-1,1,0});

    TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,0,1}, {0,1,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,1,1}, {0,1,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,0,1}, {-1,1,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,1,1}, {-1,1,1});

    TFO.set_mix_trilinear_interaction(xi, 0, 0, 3, {0,0,0}, {0,0,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 0, 3, {0,1,0}, {0,1,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 1, 2, {0,1,0}, {0,1,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 1, 2, {-1,1,0}, {-1,1,1});
    //////////////////
    TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {1,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {0,0,0}, {0,1,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {1,0,0}, {0,1,0});

    TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {1,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {0,0,0}, {0,1,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {1,0,0}, {0,1,0});

    TFO.set_mix_trilinear_interaction(xi, 1, 0, 3, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 0, 3, {1,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 2, 1, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 2, 1, {0,1,0}, {0,1,0});
    //////////////////
    TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {1,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,-1,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {1,0,0}, {1,-1,0});

    TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {1,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,-1,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {1,0,0}, {1,-1,0});

    TFO.set_mix_trilinear_interaction(xi, 3, 1, 2, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 1, 2, {1,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 0, 3, {1,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 0, 3, {1,-1,0}, {1,-1,0});

    // for(size_t i = 0; i < num_trials; ++i){
    //     lattice<3, 4, 8, 8, 8> MC_FE(&Fe_atoms, 2.5);
    //     MC_FE.simulated_annealing(15, 1e-3, 10000, 0, true);
    //     MC_FE.molecular_dynamics(15, 1e-3, 10000, 0, 0, 1000, 1e-1, dir+"/"+std::to_string(i));
    // }
}


void MD_TmFeO3_2DCS(double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double h, const array<double,3> &fielddir, string dir, bool T_zero=false, string spin_config=""){
    filesystem::create_directory(dir);
    TmFeO3_Fe<3> Fe_atoms;
    TmFeO3_Tm<8> Tm_atoms;

    array<array<double, 3>, 3> Ja = {{{Jai, 0, 0}, {0, Jai, 0}, {0, 0, Jai}}};
    array<array<double, 3>, 3> Jb = {{{Jbi, 0, 0}, {0, Jbi, 0}, {0, 0, Jbi}}};
    array<array<double, 3>, 3> Jc = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};

    array<array<double, 3>, 3> J2a = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    array<array<double, 3>, 3> J2b = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    array<array<double, 3>, 3> J2c = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};

    array<double, 9> K = {{Ka, 0, 0, 0, 0, 0, 0, 0, Kc}};

    array<array<double, 3>,3> D = {{{0, D2, -D1}, {-D2, 0, 0}, {D1, 0, 0}}};
    //In plane interactions
    //Nearest Neighbours
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {1,-1,0});
    //Next Nearest Neighbours
    Fe_atoms.set_bilinear_interaction(J2a, 0, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 0, 0, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 1, 1, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 1, 1, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 2, 2, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 2, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 3, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 3, 3, {0,1,0});
    //Out of plane interaction
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, {0,0,1});

    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,1,0});

    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,1,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,1,1});

    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,-1,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,-1,1});

    //single ion anisotropy
    Fe_atoms.set_onsite_interaction(K, 0);
    Fe_atoms.set_onsite_interaction(K, 1);
    Fe_atoms.set_onsite_interaction(K, 2);
    Fe_atoms.set_onsite_interaction(K, 3);

    //Dzyaloshinskii-Moriya interaction
    Fe_atoms.set_bilinear_interaction(D, 0, 0, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 0, 0, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 1, 1, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 1, 1, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 2, 2, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 2, 2, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 3, 3, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 3, 3, {1,-1,0});

    Fe_atoms.set_field(fielddir*h, 0);
    Fe_atoms.set_field(fielddir*h, 1);
    Fe_atoms.set_field(fielddir*h, 2);
    Fe_atoms.set_field(fielddir*h, 3);

    array<array<double, 3>,4> field_drive = {{{1,0,0},{1,0,0},{1,0,0},{1,0,0}}};

    double pulse_amp = 0.05;
    double pulse_width = 0.38;
    double pulse_freq = 0.33;

    int T_steps = abs(int((T_end-T_start)/T_step_size))+1;
    int tau_steps = abs(int((tau_end-tau_start)/tau_step_size))+1;
    tau_step_size = tau_end - tau_start < 0 ? - abs(tau_step_size) : abs(tau_step_size);
    T_step_size = T_end - T_start < 0 ? - abs(T_step_size) : abs(T_step_size);
    lattice<3, 4, 8, 8, 8> MC(&Fe_atoms, 2.5);
    if (spin_config != ""){
        MC.read_spin_from_file(spin_config);
    }else{
        MC.simulated_annealing(Temp_start, Temp_end, 10000, 0, true);
        std::random_device rd;
        std::mt19937 gen(rd());
        if (T_zero){
            for (size_t i = 0; i<100000; ++i){
                MC.deterministic_sweep();
            }
        }
    }
    MC.write_to_file_pos(dir+"/pos.txt");
    MC.write_to_file_spin(dir+"/spin_0.txt", MC.spins);

    filesystem::create_directory(dir+"/M_time_0");
    MC.M_B_t(field_drive, 0.0, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_0/M0");

    ofstream run_param;
    run_param.open(dir + "/param.txt");
    run_param << tau_start << " " << tau_end << " " << tau_steps  << " " << T_start << " " << T_end << " " << T_steps << endl;
    run_param.close();

    double current_tau = tau_start;

    for(int i=0; i< tau_steps;++i){
        filesystem::create_directory(dir+"/M_time_"+ std::to_string(i));
        cout << "Time: " << current_tau << endl;
        MC.M_B_t(field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i) + "/M1");
        MC.M_BA_BB_t(field_drive, 0.0, field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i)+ "/M01");
        current_tau += tau_step_size;
    }
}

void TmFeO3_2DCS(size_t num_trials, double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double h, const array<double,3> &fielddir, string dir, bool T_zero=false, string spin_config=""){
    int initialized;

    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    filesystem::create_directory(dir);

    for(size_t i = 0; i < num_trials; ++i){
        MD_TmFeO3_2DCS(Temp_start, Temp_end, tau_start, tau_end, tau_step_size, T_start, T_end, T_step_size, Jai, Jbi, Jci, J2ai, J2bi, J2ci, Ka, Kc, D1,  D2, h,fielddir, dir+"/"+std::to_string(i), T_zero, spin_config);
    }
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

//Pyrochlore stuff


void MD_pyrochlore(size_t num_trials, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, double theta=0, bool theta_or_Jxz=false){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};

    z1 /= double(sqrt(3));
    z2 /= double(sqrt(3));
    z3 /= double(sqrt(3));
    z4 /= double(sqrt(3));


    array<double, 3> y1 = {0,1,-1};
    array<double, 3> y2 = {0,-1,1};
    array<double, 3> y3 = {0,-1,-1};
    array<double, 3> y4 = {0,1,1};
    y1 /= sqrt(2);
    y2 /= sqrt(2);
    y3 /= sqrt(2);
    y4 /= sqrt(2);

    array<double, 3> x1 = {-2,1,1};
    array<double, 3> x2 = {-2,-1,-1};
    array<double, 3> x3 = {2,1,-1};
    array<double, 3> x4 = {2,-1,1};
    x1 /= sqrt(6);
    x2 /= sqrt(6);
    x3 /= sqrt(6);
    x4 /= sqrt(6);

    double Jx, Jy, Jz, theta_in;
    if (theta_or_Jxz){
        Jx = Jxx;
        Jy = Jyy;
        Jz = Jzz;
        theta_in = theta;
        cout << "Hi" << endl;
    }
    else{
        double Jz_Jx = Jzz-Jxx;
        double Jz_Jz_sign = (Jzz-Jxx < 0) ? -1 : 1;
        Jx = (Jxx+Jzz)/2 - Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*theta*theta)/2; 
        Jy = Jyy;
        Jz = (Jxx+Jzz)/2 + Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*theta*theta)/2; 
        theta_in = atan(2*theta/(Jzz-Jxx))/2;
        double maxJ = max(Jx, max(Jy, Jz));
        Jx /= maxJ;
        Jy /= maxJ;
        Jz /= maxJ;
    }
    cout << Jx << " " << Jy << " " << Jz << " " << theta << endl;

    array<array<double,3>, 3> J = {{{Jxx,0,0},{0,Jyy,0},{0,0,Jzz}}};
    array<double, 3> field = field_dir*h;


    atoms.set_bilinear_interaction(J, 0, 1, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 0, 0}); 

    atoms.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 1, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 1}); 
    atoms.set_bilinear_interaction(J, 1, 2, {-1, 1, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {-1, 0, 1}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 1, -1}); 

    array<double, 3> rot_field = {gzz*sin(theta_in)+gxx*cos(theta_in),0,gzz*cos(theta_in)-gxx*sin(theta_in)};
    array<double, 3> By1 = {0, gyy*(pow(dot(field,y1),3) - 3*pow(dot(field,x1),2)*dot(field,y1)),0};
    array<double, 3> By2 = {0, gyy*(pow(dot(field,y2),3) - 3*pow(dot(field,x2),2)*dot(field,y2)),0};
    array<double, 3> By3 = {0, gyy*(pow(dot(field,y3),3) - 3*pow(dot(field,x3),2)*dot(field,y3)),0};
    array<double, 3> By4 = {0, gyy*(pow(dot(field,y4),3) - 3*pow(dot(field,x4),2)*dot(field,y4)),0};

    // array<double, 3> temp = rot_field*dot(field, z1)+ By1;
    // cout << gxx << " " << gyy << " " << gzz << " " << temp[0] << " " << temp[1] << " " << temp[2] << endl;
    // array<double, 3> temp1 = rot_field*dot(field, z2)+ By2;
    // cout << gxx << " " << gyy << " " << gzz << " " << temp1[0] << " " << temp1[1] << " " << temp1[2] << endl;
    // array<double, 3> temp2 = rot_field*dot(field, z3)+ By3;
    // cout << gxx << " " << gyy << " " << gzz << " " << temp2[0] << " " << temp2[1] << " " << temp2[2] << endl;
    // array<double, 3> temp3 = rot_field*dot(field, z4)+ By4;
    // cout << gxx << " " << gyy << " " << gzz << " " << temp3[0] << " " << temp3[1] << " " << temp3[2] << endl;

    atoms.set_field(rot_field*dot(field, z1)+ By1, 0);
    atoms.set_field(rot_field*dot(field, z2)+ By2, 1);
    atoms.set_field(rot_field*dot(field, z3)+ By3, 2);
    atoms.set_field(rot_field*dot(field, z4)+ By4, 3);

    int initialized;

    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start = rank*num_trials/size;
    int end = (rank+1)*num_trials/size;
    double k_B = 0.08620689655;
    for(int i=start; i<end;++i){
        lattice<3, 4, 16, 16, 16> MC(&atoms, 0.5);
        MC.simulated_annealing(5, 1e-3, 1e4, 10, true);
        MC.molecular_dynamics(5, 1e-3, 1e4, 0, 0, 600, 0.25, dir+"/"+std::to_string(i));
        for(int i=0; i<1e6; ++i){
            MC.deterministic_sweep();
        }
        if(dir != ""){
            filesystem::create_directory(dir);
            ofstream myfile;
            myfile.open(dir+"/"+std::to_string(i)+"/spin_0.txt");
            for(size_t i = 0; i<MC.lattice_size; ++i){
                for(size_t j = 0; j<3; ++j){
                    myfile << MC.spins[i][j] << " ";
                }
                myfile << endl;
            }
            myfile.close();
        }
    }
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

void pyrochlore_2DCS(size_t num_trials, bool T_zero, double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, array<double, 3> field_extern, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, double Jxz=0, string spin_config=""){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;
    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};

    z1 /= double(sqrt(3));
    z2 /= double(sqrt(3));
    z3 /= double(sqrt(3));
    z4 /= double(sqrt(3));


    array<double, 3> y1 = {0,1,-1};
    array<double, 3> y2 = {0,-1,1};
    array<double, 3> y3 = {0,-1,-1};
    array<double, 3> y4 = {0,1,1};
    y1 /= sqrt(2);
    y2 /= sqrt(2);
    y3 /= sqrt(2);
    y4 /= sqrt(2);

    array<double, 3> x1 = {-2,1,1};
    array<double, 3> x2 = {-2,-1,-1};
    array<double, 3> x3 = {2,1,-1};
    array<double, 3> x4 = {2,-1,1};
    x1 /= sqrt(6);
    x2 /= sqrt(6);
    x3 /= sqrt(6);
    x4 /= sqrt(6);

    double Jz_Jx = Jzz-Jxx;
    double Jz_Jz_sign = (Jzz-Jxx < 0) ? -1 : 1;
    double Jx = (Jxx+Jzz)/2 - Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*Jxz*Jxz)/2; 
    double Jy = Jyy;
    double Jz = (Jxx+Jzz)/2 + Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*Jxz*Jxz)/2; 
    double theta = atan(2*Jxz/(Jzz-Jxx));

    array<array<double,3>, 3> J = {{{Jx,0,0},{0,Jy,0},{0,0,Jz}}};
    array<double, 3> field = field_dir*h;


    atoms.set_bilinear_interaction(J, 0, 1, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 0, 0}); 

    atoms.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 1, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 1}); 
    atoms.set_bilinear_interaction(J, 1, 2, {-1, 1, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {-1, 0, 1}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 1, -1}); 

    array<double, 3> rot_field = {gzz*sin(theta)+gxx*cos(theta),0,gzz*cos(theta)-gxx*sin(theta)};
    array<double, 3> By1 = {0, gyy*(pow(dot(field,y1),3) - 3*pow(dot(field,x1),2)*dot(field,y1)),0};
    array<double, 3> By2 = {0, gyy*(pow(dot(field,y2),3) - 3*pow(dot(field,x2),2)*dot(field,y2)),0};
    array<double, 3> By3 = {0, gyy*(pow(dot(field,y3),3) - 3*pow(dot(field,x3),2)*dot(field,y3)),0};
    array<double, 3> By4 = {0, gyy*(pow(dot(field,y4),3) - 3*pow(dot(field,x4),2)*dot(field,y4)),0};


    atoms.set_field(rot_field*dot(field, z1)+ By1, 0);
    atoms.set_field(rot_field*dot(field, z2)+ By2, 1);
    atoms.set_field(rot_field*dot(field, z3)+ By3, 2);
    atoms.set_field(rot_field*dot(field, z4)+ By4, 3);

    array<array<double, 3>,4> field_drive = {{{0,0,dot(field, z1)},{0,0,dot(field, z2)},{0,0,dot(field, z3)},{0,0,dot(field, z4)}}};

    double pulse_amp = 0.1;
    double pulse_width = 0.38;
    double pulse_freq = 0.33;

    int T_steps = abs(int((T_end-T_start)/T_step_size))+1;
    int tau_steps = abs(int((tau_end-tau_start)/tau_step_size))+1;
    tau_step_size = tau_end - tau_start < 0 ? - abs(tau_step_size) : abs(tau_step_size);
    T_step_size = T_end - T_start < 0 ? - abs(T_step_size) : abs(T_step_size);
    lattice<3, 4, 12, 12, 12> MC(&atoms, 0.5);
    if (spin_config != ""){
        MC.read_spin_from_file(spin_config);
    }else{
        MC.simulated_annealing(Temp_start, Temp_end, 10000, 0, true);
        if (T_zero){
            for (size_t i = 0; i<100000; ++i){
                MC.deterministic_sweep();
            }
        }
    }
    MC.write_to_file_pos(dir+"/pos.txt");
    MC.write_to_file_spin(dir+"/spin_0.txt", MC.spins);

    filesystem::create_directory(dir+"/M_time_0");
    MC.M_B_t(field_drive, 0.0, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_0/M0");

    ofstream run_param;
    run_param.open(dir + "/param.txt");
    run_param << tau_start << " " << tau_end << " " << tau_steps  << " " << T_start << " " << T_end << " " << T_steps << endl;
    run_param.close();

    double current_tau = tau_start;

    for(int i=0; i< tau_steps;++i){
        filesystem::create_directory(dir+"/M_time_"+ std::to_string(i));
        cout << "Time: " << current_tau << endl;
        MC.M_B_t(field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i) + "/M1");
        MC.M_BA_BB_t(field_drive, 0.0, field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i)+ "/M01");
        current_tau += tau_step_size;
    }

}
void  simulated_annealing_pyrochlore(double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, double theta=0, bool theta_or_Jxz=true, bool save=false){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};

    z1 /= double(sqrt(3));
    z2 /= double(sqrt(3));
    z3 /= double(sqrt(3));
    z4 /= double(sqrt(3));


    array<double, 3> y1 = {0,1,-1};
    array<double, 3> y2 = {0,-1,1};
    array<double, 3> y3 = {0,-1,-1};
    array<double, 3> y4 = {0,1,1};
    y1 /= sqrt(2);
    y2 /= sqrt(2);
    y3 /= sqrt(2);
    y4 /= sqrt(2);

    array<double, 3> x1 = {-2,1,1};
    array<double, 3> x2 = {-2,-1,-1};
    array<double, 3> x3 = {2,1,-1};
    array<double, 3> x4 = {2,-1,1};
    x1 /= sqrt(6);
    x2 /= sqrt(6);
    x3 /= sqrt(6);
    x4 /= sqrt(6);
    double Jx, Jy, Jz, theta_in;
    cout << "Begin simulated annealing with parameters: " << Jxx << " " << Jyy << " " << Jzz << " " << theta_in << endl;
    if (theta_or_Jxz){

        Jx = Jxx;
        Jy = Jyy;
        Jz = Jzz;
        theta_in = theta;
    }
    else{
        Jy = Jyy;
        theta_in = atan(2*theta/(Jxx-Jzz))/2;
        Jx = cos(theta_in)*cos(theta_in)*Jxx + sin(theta_in)*sin(theta_in)*Jzz + sin(2*theta_in)*theta;
        Jz = sin(theta_in)*sin(theta_in)*Jxx + cos(theta_in)*cos(theta_in)*Jzz - sin(2*theta_in)*theta;
        // cout << "Begin simulated annealing with parameters: " << Jx << " " << Jy << " " << Jz << " " << theta_in << endl;
        // Jx = (Jxx + Jzz)/2 - sqrt((Jxx-Jzz)*(Jxx-Jzz) + 4*theta*theta)/2;
        // Jz = (Jxx + Jzz)/2 + sqrt((Jxx-Jzz)*(Jxx-Jzz) + 4*theta*theta)/2;
        // cout << "Begin simulated annealing with parameters: " << Jx << " " << Jy << " " << Jz << " " << theta_in << endl;
        double maxJ = max(Jx, max(Jy, Jz));
        Jx /= maxJ;
        Jy /= maxJ;
        Jz /= maxJ;
    }
    cout << "Renormalized parameters: " << Jx << " " << Jy << " " << Jz << " " << theta_in << endl;
    array<array<double,3>, 3> J = {{{Jx,0,0},{0,Jy,0},{0,0,Jz}}};
    array<double, 3> field = field_dir*h;


    atoms.set_bilinear_interaction(J, 0, 1, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 0, 0}); 

    atoms.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 1, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 1}); 
    atoms.set_bilinear_interaction(J, 1, 2, {-1, 1, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {-1, 0, 1}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 1, -1}); 

    array<double, 3> rot_field = {gzz*sin(theta_in)+gxx*cos(theta_in),0,gzz*cos(theta_in)-gxx*sin(theta_in)};
    array<double, 3> By1 = {0, gyy*(pow(dot(field,y1),3) - 3*pow(dot(field,x1),2)*dot(field,y1)),0};
    array<double, 3> By2 = {0, gyy*(pow(dot(field,y2),3) - 3*pow(dot(field,x2),2)*dot(field,y2)),0};
    array<double, 3> By3 = {0, gyy*(pow(dot(field,y3),3) - 3*pow(dot(field,x3),2)*dot(field,y3)),0};
    array<double, 3> By4 = {0, gyy*(pow(dot(field,y4),3) - 3*pow(dot(field,x4),2)*dot(field,y4)),0};


    atoms.set_field(rot_field*dot(field, z1)+ By1, 0);
    atoms.set_field(rot_field*dot(field, z2)+ By2, 1);
    atoms.set_field(rot_field*dot(field, z3)+ By3, 2);
    atoms.set_field(rot_field*dot(field, z4)+ By4, 3);

    lattice<3, 4, 8, 8, 8> MC(&atoms, 0.5);
    // MC.simulated_annealing_deterministic(5, 1e-7, 10000, 10000, 0, dir);
    MC.simulated_annealing(5, 1e-4, 1e4, 0, true, dir, save);
}

// void  magnetostriction_pyrochlore(double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir){
//     filesystem::create_directory(dir);
//     Pyrochlore<3> atoms;

//     array<double,3> z1 = {1, 1, 1};
//     array<double,3> z2 = {1,-1,-1};
//     array<double,3> z3 = {-1,1,-1};
//     array<double,3> z4 = {-1,-1,1};

//     z1 = z1/double(sqrt(3));
//     z2 = z2/double(sqrt(3));
//     z3 = z3/double(sqrt(3));
//     z4 = z4/double(sqrt(3));

//     array<array<double,3>, 3> J = {{{Jxx,0,0},{0,Jyy,0},{0,0,Jzz}}};
//     array<double, 3> g = {gxx, gyy, gzz};
//     array<double, 3> field = field_dir*h;


//     atoms.set_bilinear_interaction(J, 0, 1, {0, 0, 0}); 
//     atoms.set_bilinear_interaction(J, 0, 2, {0, 0, 0}); 
//     atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 0}); 
//     atoms.set_bilinear_interaction(J, 1, 2, {0, 0, 0}); 
//     atoms.set_bilinear_interaction(J, 1, 3, {0, 0, 0}); 
//     atoms.set_bilinear_interaction(J, 2, 3, {0, 0, 0}); 

//     atoms.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); 
//     atoms.set_bilinear_interaction(J, 0, 2, {0, 1, 0}); 
//     atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 1}); 
//     atoms.set_bilinear_interaction(J, 1, 2, {-1, 1, 0}); 
//     atoms.set_bilinear_interaction(J, 1, 3, {-1, 0, 1}); 
//     atoms.set_bilinear_interaction(J, 2, 3, {0, 1, -1}); 

//     atoms.set_field(g*dot(field, z1), 0);
//     atoms.set_field(g*dot(field, z2), 1);
//     atoms.set_field(g*dot(field, z3), 2);
//     atoms.set_field(g*dot(field, z4), 3);

//     array<double, 10> strain = {4e-7, -8e-7, 12e-7, -2.6e-7, 0.27e-7, -0.8e-7, 0.5e-7, -0.7e-7, 0.43e-7, 0.51e-7};

//     lattice_strain_field<3, 4, 8, 8, 8> MC(&atoms, 0.5, strain);
//     // MC.simulated_annealing_deterministic(5, 1e-7, 10000, 10000, 0, dir);
//     MC.simulated_annealing(5, 1e-4, 10000, 0, true, dir);
// }

void parallel_tempering_pyrochlore(double T_start, double T_end, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, const vector<int> &rank_to_write, double Jxz=0){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};

    z1 /= double(sqrt(3));
    z2 /= double(sqrt(3));
    z3 /= double(sqrt(3));
    z4 /= double(sqrt(3));


    array<double, 3> y1 = {0,1,-1};
    array<double, 3> y2 = {0,-1,1};
    array<double, 3> y3 = {0,-1,-1};
    array<double, 3> y4 = {0,1,1};
    y1 /= sqrt(2);
    y2 /= sqrt(2);
    y3 /= sqrt(2);
    y4 /= sqrt(2);

    array<double, 3> x1 = {-2,1,1};
    array<double, 3> x2 = {-2,-1,-1};
    array<double, 3> x3 = {2,1,-1};
    array<double, 3> x4 = {2,-1,1};
    x1 /= sqrt(6);
    x2 /= sqrt(6);
    x3 /= sqrt(6);
    x4 /= sqrt(6);


    double Jz_Jx = Jzz-Jxx;
    double Jz_Jz_sign = (Jzz-Jxx < 0) ? -1 : 1;
    double Jx = (Jxx+Jzz)/2 - Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*Jxz*Jxz)/2; 
    double Jy = Jyy;
    double Jz = (Jxx+Jzz)/2 + Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*Jxz*Jxz)/2; 
    double theta = atan(2*Jxz/(Jzz-Jxx));

    array<array<double,3>, 3> J = {{{Jx,0,0},{0,Jy,0},{0,0,Jz}}};
    array<double, 3> field = field_dir*h;


    atoms.set_bilinear_interaction(J, 0, 1, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 0, 0}); 

    atoms.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 1, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 1}); 
    atoms.set_bilinear_interaction(J, 1, 2, {-1, 1, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {-1, 0, 1}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 1, -1}); 

    array<double, 3> rot_field = {gzz*sin(theta)+gxx*cos(theta),0,gzz*cos(theta)-gxx*sin(theta)};
    array<double, 3> By1 = {0, gyy*(pow(dot(field,y1),3) - 3*pow(dot(field,x1),2)*dot(field,y1)),0};
    array<double, 3> By2 = {0, gyy*(pow(dot(field,y2),3) - 3*pow(dot(field,x2),2)*dot(field,y2)),0};
    array<double, 3> By3 = {0, gyy*(pow(dot(field,y3),3) - 3*pow(dot(field,x3),2)*dot(field,y3)),0};
    array<double, 3> By4 = {0, gyy*(pow(dot(field,y4),3) - 3*pow(dot(field,x4),2)*dot(field,y4)),0};


    atoms.set_field(rot_field*dot(field, z1)+ By1, 0);
    atoms.set_field(rot_field*dot(field, z2)+ By2, 1);
    atoms.set_field(rot_field*dot(field, z3)+ By3, 2);
    atoms.set_field(rot_field*dot(field, z4)+ By4, 3);

    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    lattice<3, 4, 16, 16, 16> MC(&atoms, 0.5);

    vector<double> temps = logspace(log10(T_start), log10(T_end), size);

    MC.parallel_tempering(temps, 1e6, 1e6, 10, 50, 2e3, dir, rank_to_write, true);

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

void phase_diagram_pyrochlore(double Jpm_min, double Jpm_max, int num_Jpm, double h_min, double h_max, int num_h, double Jpmpm, array<double, 3> field_dir, string dir){
    filesystem::create_directory(dir);
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int totaljob_num = num_Jpm*num_h;

    int start = rank*totaljob_num/size;
    int end = (rank+1)*totaljob_num/size;

    for(int i=start; i<end; ++i){
        int Jpm_ind = i % num_Jpm;
        int h_ind = i / num_Jpm;
        double Jpm = Jpm_min + Jpm_ind*(Jpm_max-Jpm_min)/num_Jpm;
        double h = h_min + h_ind*(h_max-h_min)/num_h;
        cout << "Jpm: " << Jpm << " h: " << h << "i: " << i << endl;
        string subdir = dir + "/Jpm_" + std::to_string(Jpm) + "_h_" + std::to_string(h) + "_index_" + std::to_string(Jpm_ind) + "_" + std::to_string(h_ind);
        simulated_annealing_pyrochlore(-2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0.01, 4e-4, 1, h, field_dir, subdir);
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }

}

void pyrochlore_line_scan(double Jxx, double Jyy, double Jzz, double h_min, double h_max, int num_h, array<double, 3> field_dir, string dir, double theta, bool theta_or_Jxz, bool save){
    filesystem::create_directory(dir);
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int totaljob_num = num_h;

    int start = rank*totaljob_num/size;
    int end = (rank+1)*totaljob_num/size;

    for(int i=start; i<end; ++i){
        double h = h_min + i*(h_max-h_min)/num_h;
        cout << "h: " << h << "i: " << i << endl;
        string subdir = dir + "/h_" + std::to_string(h) + "_index_" + std::to_string(i);
        simulated_annealing_pyrochlore(Jxx, Jyy, Jzz, 0.01, 4e-4, 1, h, field_dir, subdir, theta, theta_or_Jxz, save);
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }

}


void phase_diagram_pyrochlore_0_field(int num_Jpm, string dir){
    filesystem::create_directory(dir);
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int totaljob_num = num_Jpm*num_Jpm;

    int start = rank*totaljob_num/size;
    int end = (rank+1)*totaljob_num/size;

    for(int i=start; i<end; ++i){
        int Jpm_ind = i % num_Jpm;
        int h_ind = i / num_Jpm;
        double Jxx = -1 + double(Jpm_ind*2)/double(num_Jpm);
        double Jzz = -1 + double(h_ind*2)/double(num_Jpm);
        cout << "Jxx: " << Jxx << " Jzz: " << Jzz << "i: " << i << endl;
        string subdir = dir + "/Jxx_" + std::to_string(Jxx) + "_Jzz_" + std::to_string(Jzz) + "_index_" + std::to_string(Jpm_ind) + "_" + std::to_string(h_ind);
        simulated_annealing_pyrochlore(Jxx, 1, Jzz, 0.01, 4e-4, 1, 0, {0,0,1}, subdir);
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

#endif