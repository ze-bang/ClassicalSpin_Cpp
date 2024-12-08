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

        lattice<3, 2, 24, 24, 1> MC(&atoms);
        // MC.simulated_annealing(1, 0.001, 1000, 10000, 1000000, 0, dir+"/"+std::to_string(i));
        MC.molecular_dynamics(1,0.001, 1000, 10000, 0, 100, 1e-1, dir+"/"+std::to_string(i));
    }
}


void nonlinearspectroscopy_kitaev_honeycomb(float tau_end, size_t tau_steps, float K, float Gamma, float Gammap, float h, string dir, bool T_zero){
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

    array<array<float, 3>,2> field_drive = {{{0,0,1},{0,0,1}}};

    float pulse_width = 0.01;
    
    float T_end = 250;
    float T_step_size = 0.1;
    int T_steps = int(T_end/T_step_size)+1;

    lattice<3, 2, 24, 24, 1> MC(&atoms);
    MC.simulated_annealing(1.0, 0.001, 1000, 10000, 0);
    if (T_zero){
        for (size_t i = 0; i<10000; ++i){
            MC.deterministic_sweep();
        }
    }
    MC.write_to_file_pos(dir+"/pos.txt");
    MC.write_to_file_spin(dir+"/spin_0.txt", MC.spins);

    lattice<3, 2, 24, 24, 1> temp_lattice(&MC);
    filesystem::create_directory(dir+"/M_time_0");
    temp_lattice.M_B_t(field_drive, 0.0, pulse_width, T_end, T_step_size, dir+"/M_time_0/M0");

    ofstream run_param;
    run_param.open(dir + "/param.txt");
    run_param << tau_end << " " << tau_steps + 1 << " " << T_end << " " << T_steps << " " << K << " " << h << endl;
    run_param.close();

    for(int i=0; i< tau_steps+1;++i){
        float current_time = i*tau_end/tau_steps;
        filesystem::create_directory(dir+"/M_time_"+ std::to_string(i));
        temp_lattice.reset_lattice(&MC);
        MC.M_B_t(field_drive, current_time, pulse_width, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i) + "/M1");
        temp_lattice.reset_lattice(&MC);
        MC.M_BA_BB_t(field_drive, current_time, field_drive, pulse_width, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i)+ "/M01");
    }
}


void full_nonlinearspectroscopy_kitaev_honeycomb(size_t num_trials, float tau_end, size_t tau_steps, float K, float Gamma, float Gammap, float h, string dir, bool T_zero){
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
        nonlinearspectroscopy_kitaev_honeycomb(tau_end, tau_steps, K, Gamma, Gammap, h, dir+std::to_string(i), T_zero);
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
        MC.molecular_dynamics(14,0.09, 1000, 10000, 0, 1000, 1e-1, dir+"/"+std::to_string(i));
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
    MC.simulated_annealing_deterministic(5, 1e-7, 1000, 10000, 10000, 0, dir);
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
        string subdir = dir + "/Jpm_" + std::to_string(Jpm) + "_h_" + std::to_string(h) + "_index_" + std::to_string(Jpm_ind) + "_" + std::to_string(h_ind);
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
    // string dir = "kitaev_honeycomb_nonlinear_Gamma=0.25_Gammap=-0.02_h=0.7/";
    // full_nonlinearspectroscopy_kitaev_honeycomb(50, 250, 2500, -1.0, 0.25, -0.02, 0.7, dir, true);
    // phase_diagram_pyrochlore(-0.3, 0.3, 200, 0.0, 8.0, 100, 0.2, {0,0,1}, "MC_phase_diagram_CZO_001_XAIAO_T_zero");
    // phase_diagram_pyrochlore(-0.3, 0.3, 200, 0.0, 8.0, 100, 0.2, {1/sqrt(3),1/sqrt(3),1/sqrt(3)}, "MC_phase_diagram_CZO_111_XAIAO_T_zero");
    // phase_diagram_pyrochlore(-0.3, 0.3, 200, 0.0, 8.0, 100, 0.2, {1/sqrt(2),1/sqrt(2),0}, "MC_phase_diagram_CZO_110_XAIAO_T_zero");
    // phase_diagram_pyrochlore(-0.3, 0.3, 200, 0.0, 8.0, 100, -0.2, {0,0,1}, "MC_phase_diagram_CZO_001_ZAIAO_T_zero");
    // phase_diagram_pyrochlore(-0.3, 0.3, 200, 0.0, 8.0, 100, -0.2, {1/sqrt(3),1/sqrt(3),1/sqrt(3)}, "MC_phase_diagram_CZO_111_ZAIAO_T_zero");
    phase_diagram_pyrochlore(-0.3, 0.3, 200, 0.0, 8.0, 100, -0.2, {1/sqrt(2),1/sqrt(2),0}, "MC_phase_diagram_CZO_110_ZAIAO_T_zero");
    // MD_pyrochlore(1, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0, 0, 2.24, 1.5*mu_B/0.063, {1/sqrt(2), 1/sqrt(2), 0}, "pyrochlore_test_110");
    // MD_pyrochlore(20, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0, 0, 2.24, 1.5*mu_B/0.063, {0,0,1}, "pyrochlore_test_001");
    // std::cout << "finished" << std::endl;   
    return 0;
}