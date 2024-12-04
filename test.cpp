#include "unitcell.h"
#include "lattice.h"
#include "mixed_lattice.h"
#include <iostream>
#include <mpi.h>
#include "simple_linear_alg.h"

void MD_kitaev_honeycomb(size_t num_trials, float K, float Gamma, float Gammap, float h, string dir){
    srand(time(NULL));
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;
    array<array<float,3>, 3> Jx = {{{K,Gammap,Gammap},{Gammap,0,Gamma},{Gammap,Gamma,0}}};
    array<array<float,3>, 3> Jy = {{{0,Gammap,Gamma},{Gammap,K,Gammap},{Gamma,Gammap,0}}};
    array<array<float,3>, 3> Jz = {{{0,Gamma,Gammap},{Gamma,0,Gammap},{Gammap,Gammap,K}}};


    array<float, 3> field = {h/float(sqrt(3)),h/float(sqrt(3)),h/float(sqrt(3))};
    
    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    for(size_t i=0; i<num_trials;++i){

        lattice<3, 2, 20, 20, 1> MC(&atoms);
        MC.simulated_annealing(1, 1e-7, 1000, 10000, 0, true);
        std::random_device rd;
        std::mt19937 gen(rd());
        for (size_t i = 0; i<100000; ++i){
            MC.deterministic_sweep(gen);
        }
        MC.molecular_dynamics(1,1e-7, 1000, 1000000, 100, -600, 600, 1e-1, dir+"/"+std::to_string(i));
    }
}

// kitaev local xyz {1,-1,-1} / {-1, 1, -1} / {-1, -1, 1}
void nonlinearspectroscopy_kitaev_honeycomb(float Temp, float tau_start, float tau_end, float tau_step_size, float T_start, float T_end, float T_step_size, float K, float Gamma, float Gammap, float h, string dir, bool T_zero){
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;
    array<array<float,3>, 3> Jx = {{{K,Gammap,Gammap},{Gammap,0,Gamma},{Gammap,Gamma,0}}};
    array<array<float,3>, 3> Jy = {{{0,Gammap,Gamma},{Gammap,K,Gammap},{Gamma,Gammap,0}}};
    array<array<float,3>, 3> Jz = {{{0,Gamma,Gammap},{Gamma,0,Gammap},{Gammap,Gammap,K}}};

    array<float, 3> field = {h/float(sqrt(3)),h/float(sqrt(3)),h/float(sqrt(3))};


    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    // array<array<float, 3>,2> field_drive = {{{-1/sqrt(3), -1/sqrt(3), 1/sqrt(3)},{-1/sqrt(3), -1/sqrt(3), 1/sqrt(3)}}};
    array<array<float, 3>,2> field_drive = {{{0,0,1},{0,0,1}}};

    float pulse_amp = 0.1;
    float pulse_width = 0.38;
    float pulse_freq = 0.33;

    int T_steps = int((T_end-T_start)/T_step_size)+1;
    int tau_steps = int((tau_end-tau_start)/tau_step_size)+1;

    lattice<3, 2, 20, 20, 1> MC(&atoms);
    MC.simulated_annealing(2.0, Temp, 1000, 10000, 0, true);
    std::random_device rd;
    std::mt19937 gen(rd());
    if (T_zero){
        for (size_t i = 0; i<100000; ++i){
            MC.deterministic_sweep(gen);
        }
    }
    MC.write_to_file_pos(dir+"/pos.txt");
    MC.write_to_file_spin(dir+"/spin_0.txt", MC.spins);

    // lattice<3, 2, 20, 20, 1> temp_lattice(&MC);
    filesystem::create_directory(dir+"/M_time_0");
    MC.M_B_t(field_drive, 0.0, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_0/M0");

    ofstream run_param;
    run_param.open(dir + "/param.txt");
    run_param << tau_start << " " << tau_end << " " << tau_steps  << " " << T_start << " " << T_end << " " << T_steps << " " << K << " " << h << endl;
    run_param.close();

    float current_tau = tau_start;

    for(int i=0; i< tau_steps;++i){
        filesystem::create_directory(dir+"/M_time_"+ std::to_string(i));
        cout << "Time: " << current_tau << endl;
        // temp_lattice.reset_lattice(&MC);
        MC.M_B_t(field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i) + "/M1");
        // temp_lattice.reset_lattice(&MC);
        MC.M_BA_BB_t(field_drive, 0.0, field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i)+ "/M01");
        current_tau += tau_step_size;
    }
}


void full_nonlinearspectroscopy_kitaev_honeycomb(size_t num_trials, float Temp, float tau_start, float tau_end, float tau_step_size, float T_start, float T_end, float T_step_size, float K, float Gamma, float Gammap, float h, string dir, bool T_zero){
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
        nonlinearspectroscopy_kitaev_honeycomb(Temp, tau_start, tau_end, tau_step_size, T_start, T_end, T_step_size, K, Gamma, Gammap, h, dir+std::to_string(i), T_zero);
    }
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

void MD_TmFeO3(int num_trials, float J, float xi, string dir){
    HoneyComb<3> atoms_SU2;
    HoneyComb<8> atoms_SU3;
    
    mixed_UnitCell<3, 2, 8, 2> atoms(&atoms_SU2, &atoms_SU3);
    mixed_lattice<3, 2, 8, 2, 16, 16, 1> MC(&atoms);
}

void MD_pyrochlore(size_t num_trials, float Jxx, float Jyy, float Jzz, float gxx, float gyy, float gzz, float h, array<float, 3> field_dir, string dir){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<float,3> z1 = {1, 1, 1};
    array<float,3> z2 = {1,-1,-1};
    array<float,3> z3 = {-1,1,-1};
    array<float,3> z4 = {-1,-1,1};

    z1 = z1/float(sqrt(3));
    z2 = z2/float(sqrt(3));
    z3 = z3/float(sqrt(3));
    z4 = z4/float(sqrt(3));

    array<array<float,3>, 3> J = {{{Jxx,0,0},{0,Jyy,0},{0,0,Jzz}}};
    array<float, 3> g = {gxx, gyy, gzz};
    array<float, 3> field = field_dir*h;


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

    atoms.set_field(g*dot(field, z1), 0);
    atoms.set_field(g*dot(field, z2), 1);
    atoms.set_field(g*dot(field, z3), 2);
    atoms.set_field(g*dot(field, z4), 3);
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

    float k_B = 0.08620689655;
    float hbar = 4.135667696e-12;
    for(int i=start; i<end;++i){

        lattice<3, 4, 12, 12, 12> MC(&atoms);
        // MC.simulated_annealing(1, 0.001, 1000, 10000, 1000000, 0, dir+"/"+std::to_string(i));
        MC.molecular_dynamics(14,0.09, 1000, 10000, 0, 0, 1000, 1e-1, dir+"/"+std::to_string(i));
    }
}


void simulated_annealing_pyrochlore(float Jxx, float Jyy, float Jzz, float gxx, float gyy, float gzz, float h, array<float, 3> field_dir, string dir){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<float,3> z1 = {1, 1, 1};
    array<float,3> z2 = {1,-1,-1};
    array<float,3> z3 = {-1,1,-1};
    array<float,3> z4 = {-1,-1,1};

    z1 = z1/float(sqrt(3));
    z2 = z2/float(sqrt(3));
    z3 = z3/float(sqrt(3));
    z4 = z4/float(sqrt(3));

    array<array<float,3>, 3> J = {{{Jxx,0,0},{0,Jyy,0},{0,0,Jzz}}};
    array<float, 3> g = {gxx, gyy, gzz};
    array<float, 3> field = field_dir*h;


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

    atoms.set_field(g*dot(field, z1), 0);
    atoms.set_field(g*dot(field, z2), 1);
    atoms.set_field(g*dot(field, z3), 2);
    atoms.set_field(g*dot(field, z4), 3);

    lattice<3, 4, 8, 8, 8> MC(&atoms);
    MC.simulated_annealing_deterministic(14, 0.06, 1000, 10000, 10000, 0, dir);
}

void phase_diagram_pyrochlore(float Jpm_min, float Jpm_max, int num_Jpm, float h_min, float h_max, int num_h, float Jpmpm, array<float, 3> field_dir, string dir){
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
        float Jpm = Jpm_min + Jpm_ind*(Jpm_max-Jpm_min)/num_Jpm;
        float h = h_min + h_ind*(h_max-h_min)/num_h;
        cout << "Jpm: " << Jpm << " h: " << h << "i: " << i << endl;
        string subdir = dir + "/Jpm_" + std::to_string(Jpm) + "_h_" + std::to_string(h);
        simulated_annealing_pyrochlore(-2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0, 0, 1, h, field_dir, subdir);
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }

}
int main(int argc, char** argv) {
    float k_B = 0.08620689655;
    float mu_B = 5.7883818012e-2;
    // MD_TmFeO3(1, -1.0, -0.06, "test_L=12");
    // MD_kitaev_honeycomb(1, -1.0, 0.25, -0.02, 0.7, "integrity_test");
    string dir = "test_short_K=-1/";
    full_nonlinearspectroscopy_kitaev_honeycomb(10, 1e-7, 0, 600, 0.25, 0, 600, 0.25, -1.0, 0.0, -0.0, 0.0, dir, true);
    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.2, {0,0,1}, "MC_phase_diagram_CZO_001");
    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.0, {1,1,1}, "MC_phase_diagram_CZO_111");
    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.0, {1,1,0}, "MC_phase_diagram_CZO_110");

    // MD_pyrochlore(1, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0, 0, 2.24, 1.5*mu_B/0.063, {1/sqrt(2), 1/sqrt(2), 0}, "pyrochlore_test_110");
    // MD_pyrochlore(20, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0, 0, 2.24, 1.5*mu_B/0.063, {0,0,1}, "pyrochlore_test_001");
    // std::cout << "finished" << std::endl;   
    return 0;
}