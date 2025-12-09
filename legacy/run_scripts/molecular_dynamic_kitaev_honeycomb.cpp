#include "experiments.h"


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

    double pulse_amp = 0.5;
    double pulse_width = 0.38;
    double pulse_freq = 0.33;

    int T_steps = abs(int((T_end-T_start)/T_step_size))+1;
    int tau_steps = abs(int((tau_end-tau_start)/tau_step_size))+1;
    tau_step_size = tau_end - tau_start < 0 ? - abs(tau_step_size) : abs(tau_step_size);
    T_step_size = T_end - T_start < 0 ? - abs(T_step_size) : abs(T_step_size);
    lattice<3, 2, 6, 6, 1> MC(&atoms);
    MC.simulated_annealing(Temp_start, Temp_end, 1000, 1);

    if (T_zero){
        for (size_t i = 0; i<1000; ++i){
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

    // Print all simulation parameters
    if (rank == 0) {
        cout << "=====================================" << endl;
        cout << "Simulation Parameters:" << endl;
        cout << "=====================================" << endl;
        cout << "Number of trials: " << num_trials << endl;
        cout << "Temperature annealing: " << Temp_start << " -> " << Temp_end << endl;
        cout << "Tau range: " << tau_start << " -> " << tau_end << endl;
        cout << "Tau step size: " << tau_step_size << endl;
        cout << "Time range: " << T_start << " -> " << T_end << endl;
        cout << "Time step size: " << T_step_size << endl;
        cout << "K (Kitaev): " << K << endl;
        cout << "Gamma: " << Gamma << endl;
        cout << "Gamma': " << Gammap << endl;
        cout << "Magnetic field h: " << h << endl;
        cout << "Field components: {" << h/sqrt(3) << ", " << h/sqrt(3) << ", " << h/sqrt(3) << "}" << endl;
        cout << "T_zero mode: " << (T_zero ? "true" : "false") << endl;
        cout << "Output directory: " << dir << endl;
        cout << "MPI size: " << size << endl;
        cout << "=====================================" << endl;

        // Save parameters to file
        ofstream param_file;
        param_file.open(dir + "/simulation_params.txt");
        param_file << "Simulation Parameters" << endl;
        param_file << "=====================================" << endl;
        param_file << "num_trials: " << num_trials << endl;
        param_file << "Temp_start: " << Temp_start << endl;
        param_file << "Temp_end: " << Temp_end << endl;
        param_file << "tau_start: " << tau_start << endl;
        param_file << "tau_end: " << tau_end << endl;
        param_file << "tau_step_size: " << tau_step_size << endl;
        param_file << "T_start: " << T_start << endl;
        param_file << "T_end: " << T_end << endl;
        param_file << "T_step_size: " << T_step_size << endl;
        param_file << "K: " << K << endl;
        param_file << "Gamma: " << Gamma << endl;
        param_file << "Gammap: " << Gammap << endl;
        param_file << "h: " << h << endl;
        param_file << "field_x: " << h/sqrt(3) << endl;
        param_file << "field_y: " << h/sqrt(3) << endl;
        param_file << "field_z: " << h/sqrt(3) << endl;
        param_file << "T_zero: " << (T_zero ? "true" : "false") << endl;
        param_file << "MPI_size: " << size << endl;
        param_file.close();
    }

    for(size_t i = 0; i < num_trials; ++i){
        nonlinearspectroscopy_kitaev_honeycomb(Temp_start, Temp_end, tau_start, tau_end, tau_step_size, T_start, T_end, T_step_size, K, Gamma, Gammap, h, dir+"/"+std::to_string(i), T_zero);
    }
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

void MD_kitaev_honeycomb_real(size_t num_trials, string dir, double J=0, double K=-1, double Gamma=0.25, double Gammap=-0.02, double h=0.7){
    filesystem::create_directory(dir);
    HoneyComb_standarx<3> atoms;
    array<array<double,3>, 3> Jx = {{{J+K,Gammap,Gammap},{Gammap,J,Gamma},{Gammap,Gamma,J}}};
    array<array<double,3>, 3> Jy = {{{J,Gammap,Gamma},{Gammap,J+K,Gammap},{Gamma,Gammap,J}}};
    array<array<double,3>, 3> Jz = {{{J,Gamma,Gammap},{Gamma,J,Gammap},{Gammap,Gammap,J+K}}};


    array<double, 3> field = {h/double(sqrt(3)),h/double(sqrt(3)),h/double(sqrt(3))};
    
    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    double k_B = 0.08620689655;

    for(size_t i=0; i<num_trials;++i){

        lattice<3, 2, 24, 24, 1> MC(&atoms);
        MC.simulated_annealing(5, 1e-2, 1000, 1);
        MC.write_to_file_spin(dir+"/"+std::to_string(i)+"/spin_initial.txt", MC.spins);
        MC.molecular_dynamics(0, 100, 1e-2, dir+"/"+std::to_string(i));
    }
}


#include <math.h>

void MD_honeycomb_J1_J3(string dir, size_t num_trials=1){
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;
    double J1, K1, eta1, Gamma1, Gammap11, Gammap21;
    double J3, K3, eta3, Gamma3, Gammap13, Gammap23;
    double h = 0;

    J1 = -5.5;
    K1 = 0.1;
    eta1 = 0.06;
    Gamma1 = 2.2;
    Gammap11 = 2.0;
    Gammap21 = 2.2;
    J3 = 1.38;
    K3 = 0.0;
    eta3 = 0.0;
    Gamma3 = -1.2;
    Gammap13 = -1.2;
    Gammap23 = -1.2;

    array<array<double,3>, 3> J1x_ = {{{J1+K1,Gammap11,Gammap21},{Gammap11,J1+eta1,Gamma1},{Gammap21,Gamma1,J1-eta1}}};
    array<array<double,3>, 3> J1y_ = {{{J1-eta1,Gammap21,Gamma1},{Gammap21,J1+K1,Gammap11},{Gamma1,Gammap11,J1+eta1}}};
    array<array<double,3>, 3> J1z_ = {{{J1+eta1,Gamma1,Gammap11},{Gamma1,J1-eta1,Gammap21},{Gammap11,Gammap21,J1+K1}}};

    array<array<double,3>, 3> J3x_ = {{{J3+K3,Gammap13,Gammap23},{Gammap13,J3+eta3,Gamma3},{Gammap23,Gamma3,J3-eta3}}};
    array<array<double,3>, 3> J3y_ = {{{J3-eta3,Gammap23,Gamma3},{Gammap23,J3+K3,Gammap13},{Gamma3,Gammap13,J3+eta3}}};
    array<array<double,3>, 3> J3z_ = {{{J3,Gamma3+eta3,Gammap13},{Gamma3,J3-eta3,Gammap23},{Gammap13,Gammap23,J3+K3}}};


    array<double, 3> field = {h/double(sqrt(3)),h/double(sqrt(3)),h/double(sqrt(3))};
    
    atoms.set_bilinear_interaction(J1x_, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J1y_, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});

    atoms.set_bilinear_interaction(J3x_, 0, 1, {1,0,0});
    atoms.set_bilinear_interaction(J3y_, 0, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J3z_, 0, 1, {1,-2,0});

    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    double k_B = 0.08620689655;

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int start = rank*num_trials/size;
    int end = (rank+1)*num_trials/size;

    for(size_t i=start; i<end;++i){

        lattice<3, 2, 24, 24, 1> MC(&atoms);
        MC.simulated_annealing(150*k_B, 2*k_B, 100000, 100, true);
        MC.molecular_dynamics(0, 100, 1e-2, dir+"/"+std::to_string(i));
    }
}


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<int> rank_to_write = {size-1};
    double K = argv[1] ? atof(argv[1]) : 0.0;
    double Gamma = argv[2] ? atof(argv[2]) : 0.0;
    double Gammap = argv[3] ? atof(argv[3]) : 0.0;
    double h = argv[4] ? atof(argv[4]) : 0.0;
    string dir_name = argv[5] ? argv[5] : "";
    filesystem::create_directory(dir_name);
    int num_trials = argv[6] ? atoi(argv[6]) : 1;
    double J = argv[7] ? atof(argv[7]) : 0.0;
    MD_kitaev_honeycomb_real(1, "KITAEV");
    // MD_honeycomb_J1_J3("BCAO_J1J3",20);
    // full_nonlinearspectroscopy_kitaev_honeycomb(1, 5, 0.01, -200, 0, 0.2, -200, 200, 0.2, -1, -0.25, 0.02, 0.7, "KITAEV", true);
    return 0;
}