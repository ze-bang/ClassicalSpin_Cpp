#include "unitcell.h"
#include "lattice.h"
#include "mixed_lattice.h"
#include <iostream>
#include <mpi.h>
#include "simple_linear_alg.h"

void simulated_annealing_honeycomb(double T_start, double T_end, double K, double Gamma, double Gammap, double h, string dir=""){
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

    lattice<3, 2, 20, 20, 1> MC(&atoms);
    MC.simulated_annealing(T_start, T_end, 10000, 0, false, dir);
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

        lattice<3, 2, 20, 20, 1> MC(&atoms);
        MC.simulated_annealing(1, 1e-7, 10000, 0, true);
        std::random_device rd;
        std::mt19937 gen(rd());
        for (size_t i = 0; i<100000; ++i){
            MC.deterministic_sweep(gen);
        }
        MC.molecular_dynamics(1,1e-7, 1000000, 100, -600, 600, 1e-1, dir+"/"+std::to_string(i));
    }
}

// kitaev local xyz {1,-1,-1} / {-1, 1, -1} / {-1, -1, 1}
void nonlinearspectroscopy_kitaev_honeycomb(double Temp, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double K, double Gamma, double Gammap, double h, string dir, bool T_zero){
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

    int T_steps = int((T_end-T_start)/T_step_size)+1;
    int tau_steps = int((tau_end-tau_start)/tau_step_size)+1;

    lattice<3, 2, 20, 20, 1> MC(&atoms);
    MC.simulated_annealing(2.0, Temp, 10000, 0, true);
    std::random_device rd;
    std::mt19937 gen(rd());
    if (T_zero){
        for (size_t i = 0; i<100000; ++i){
            MC.deterministic_sweep(gen);
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


void full_nonlinearspectroscopy_kitaev_honeycomb(size_t num_trials, double Temp, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double K, double Gamma, double Gammap, double h, string dir, bool T_zero){
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

void MD_TmFeO3(int num_trials, double J, double xi, string dir){
    HoneyComb<3> atoms_SU2;
    HoneyComb<8> atoms_SU3;
    
    mixed_UnitCell<3, 2, 8, 2> atoms(&atoms_SU2, &atoms_SU3);
    mixed_lattice<3, 2, 8, 2, 16, 16, 1> MC(&atoms);
}

void MD_pyrochlore(size_t num_trials, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};

    z1 = z1/double(sqrt(3));
    z2 = z2/double(sqrt(3));
    z3 = z3/double(sqrt(3));
    z4 = z4/double(sqrt(3));

    array<array<double,3>, 3> J = {{{Jxx,0,0},{0,Jyy,0},{0,0,Jzz}}};
    array<double, 3> g = {gxx, gyy, gzz};
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

    for(int i=start; i<end;++i){

        lattice<3, 4, 12, 12, 12> MC(&atoms);
        // MC.simulated_annealing(1, 0.001, 1000, 10000, 1000000, 0, dir+"/"+std::to_string(i));
        MC.molecular_dynamics(14,0.09, 10000, 0, 0, 1000, 1e-1, dir+"/"+std::to_string(i));
    }
}


void  simulated_annealing_pyrochlore(double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};

    z1 = z1/double(sqrt(3));
    z2 = z2/double(sqrt(3));
    z3 = z3/double(sqrt(3));
    z4 = z4/double(sqrt(3));

    array<array<double,3>, 3> J = {{{Jxx,0,0},{0,Jyy,0},{0,0,Jzz}}};
    array<double, 3> g = {gxx, gyy, gzz};
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

    atoms.set_field(g*dot(field, z1), 0);
    atoms.set_field(g*dot(field, z2), 1);
    atoms.set_field(g*dot(field, z3), 2);
    atoms.set_field(g*dot(field, z4), 3);

    lattice<3, 4, 4, 4, 4> MC(&atoms, 0.5);
    // MC.simulated_annealing_deterministic(5, 1e-7, 10000, 10000, 0, dir);
    MC.simulated_annealing(10, 1e-3, 10000, 0, true, dir);
}

void parallel_tempering_pyrochlore(double T_start, double T_end, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, const vector<int> &rank_to_write){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};

    z1 = z1/double(sqrt(3));
    z2 = z2/double(sqrt(3));
    z3 = z3/double(sqrt(3));
    z4 = z4/double(sqrt(3));

    array<array<double,3>, 3> J = {{{Jxx,0,0},{0,Jyy,0},{0,0,Jzz}}};
    array<double, 3> g = {gxx, gyy, gzz};
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

    atoms.set_field(g*dot(field, z1), 0);
    atoms.set_field(g*dot(field, z2), 1);
    atoms.set_field(g*dot(field, z3), 2);
    atoms.set_field(g*dot(field, z4), 3);

    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    lattice<3, 4, 4, 4, 4> MC(&atoms, 0.5);

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
        simulated_annealing_pyrochlore(-2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0, 0, 1, h, field_dir, subdir);
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }

}
int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    // MD_TmFeO3(1, -1.0, -0.06, "test_L=12");
    // MD_kitaev_honeycomb(1, -1.0, 0.25, -0.02, 0.7, "integrity_test");
    // string dir = "test_long_h=0.7/";
    // full_nonlinearspectroscopy_kitaev_honeycomb(1, 1e-7, 0, 200, 0.05, 0, 200, 0.05, -1.0, 0.25, -0.02, 0.7, dir, true);
    // simulated_annealing_honeycomb(1, 1e-6, -1, 0.25, -0.02, 0.7, "test_simulated_annealing");
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<int> rank_to_write = {size-1};
    double Jpm_start = argv[1] ? atof(argv[1]) : 0.0;
    double Jpm_end = argv[2] ? atof(argv[2]) : 0.0;
    int num_Jpm = argv[3] ? atoi(argv[3]) : 0;
    double Jpmpm = argv[4] ? atof(argv[4]) : 0.0;
    double h_min = argv[5] ? atof(argv[5]) : 0.0;
    double h_max = argv[6] ? atof(argv[6]) : 0.0;
    double num_H = argv[7] ? atoi(argv[7]) : 0;
    string dir_string = argv[8] ? argv[8] : "001";
    array<double, 3> field_dir;
    if (dir_string == "001"){
        field_dir = {0,0,1};
    }else if(dir_string == "110"){
        field_dir = {1/sqrt(2), 1/sqrt(2), 0};
    }else{
        field_dir = {1/sqrt(3),1/sqrt(3),1/sqrt(3)};
    }
    int SLURM_TASK_ID = argv[9] ? atoi(argv[9]) : 0;
    int Jpm_ind = SLURM_TASK_ID % num_Jpm;
    int h_ind = SLURM_TASK_ID / num_Jpm;
    double Jpm = Jpm_start + Jpm_ind*(Jpm_end-Jpm_start)/num_Jpm;
    double h = h_min + h_ind*(h_max-h_min)/num_H;
    string dir_name = argv[10] ? argv[10] : "";
    filesystem::create_directory(dir_name);
    string sub_dir = dir_name + "/Jpm_" + std::to_string(Jpm) + "_h_" + std::to_string(h) + "_" + std::to_string(Jpm_ind) + "_" + std::to_string(h_ind);
    int MPI_n_tasks = argv[11] ? atoi(argv[11]) : 1;
    parallel_tempering_pyrochlore(5, 1e-4, -2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0, 0, 1, h, field_dir, sub_dir, {MPI_n_tasks-1});

    // simulated_annealing_pyrochlore(-0.4, 1, 0.4, 0, 0, 1, 0, {0,0,1}, "test");

    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.2, {0,0,1}, "MC_phase_diagram_CZO_001");
    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.0, {1,1,1}, "MC_phase_diagram_CZO_111");
    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.0, {1,1,0}, "MC_phase_diagram_CZO_110");

    // MD_pyrochlore(1, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0, 0, 2.24, 1.5*mu_B/0.063, {1/sqrt(2), 1/sqrt(2), 0}, "pyrochlore_test_110");
    // MD_pyrochlore(20, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0, 0, 2.24, 1.5*mu_B/0.063, {0,0,1}, "pyrochlore_test_001");
    // std::cout << "finished" << std::endl;   
    return 0;
}