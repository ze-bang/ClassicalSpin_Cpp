#include "experiments.h"
#include "../src/molecular_dynamics.cuh"

void MD_TmFeO3_Fe_2DCS(double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double h, const array<double,3> &fielddir, string dir, bool T_zero=false, string spin_config=""){
    int initialized;

    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    filesystem::create_directories(dir);
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

    array<array<double, 3>,4> field_drive = {{{0,1,0},{0,1,0},{0,1,0},{0,1,0}}};

    double pulse_amp = 0.9;
    double pulse_width = 0.38;
    double pulse_freq = 0.33;

    int T_steps = abs(int((T_end-T_start)/T_step_size))+1;
    int tau_steps = abs(int((tau_end-tau_start)/tau_step_size))+1;
    tau_step_size = tau_end - tau_start < 0 ? - abs(tau_step_size) : abs(tau_step_size);
    T_step_size = T_end - T_start < 0 ? - abs(T_step_size) : abs(T_step_size);
    lattice<3, 4, 8, 8, 8> MC(&Fe_atoms, 2.5);
    if (spin_config != ""){
        // Check if the spin configuration file exists
        if (filesystem::exists(spin_config+"_SU2.txt")) {
            MC.read_spin_from_file(spin_config+"_SU2.txt");
        } else {
            cout << "Warning: Spin configuration file " << spin_config+"_SU2.txt" << " does not exist. Using simulated annealing instead." << endl;
            MC.simulated_annealing(Temp_start, Temp_end, 100000, 1000, true);
            if (T_zero){
                for (size_t i = 0; i<100000; ++i){
                    MC.deterministic_sweep();
                }
            }
        }
    }else{
        MC.simulated_annealing(Temp_start, Temp_end, 100000, 1000, true);
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

    if (rank==0){
        filesystem::create_directories(dir+"/M_time_0");
        MC.M_B_t(field_drive, 0.0, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_0"+ "/M0");
    }

    ofstream run_param;
    run_param.open(dir + "/param.txt");
    run_param << tau_start << " " << tau_end << " " << tau_steps  << " " << T_start << " " << T_end << " " << T_steps << endl;
    run_param.close();


    int tau_length = int(tau_steps/size);

    double current_tau = tau_start+tau_steps*rank/size*tau_step_size;


    for(int i=0; i< tau_length;++i){
        filesystem::create_directories(dir+"/M_time_"+std::to_string(current_tau));
        cout << "Time: " << current_tau << endl;
        MC.M_B_t(field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+std::to_string(current_tau)+"/M1");
        MC.M_BA_BB_t(field_drive, 0.0, field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+std::to_string(current_tau)+"/M01");
        current_tau += tau_step_size;
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

void MD_TmFeO3_2DCS(double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double e1, double e2, double xii, double h, const array<double,3> &fielddir, string dir, bool T_zero=false, string spin_config=""){
    int initialized;

    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    filesystem::create_directories(dir);
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

    if (xii != 0.0){

        array<array<array<double,3>,3>,8> xi = {{{0}}};

        xi[0] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
        xi[1] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};

        ///////////////////
        TFO.set_mix_trilinear_interaction(xi, 1, 0, 3, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 1, 2, {0,1,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 2, 3, {0,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 1, 0, {0,0,0}, {1,0,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 1, 0, {0,1,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 2, 3, {0,1,0}, {1,0,0});
        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 2, 2, 3, {0,0,1}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 2, 2, 3, {0,1,1}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 2, 1, 2, {0,0,0}, {0,0,1});
        TFO.set_mix_trilinear_interaction(xi, 2, 0, 3, {1,0,0}, {1,0,1});
        //////////////////

        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,0,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,0,1}, {0,1,1});

        TFO.set_mix_trilinear_interaction(xi, 0, 1, 2, {-1,1,0}, {-1,1,1});
        TFO.set_mix_trilinear_interaction(xi, 0, 0, 3, {0,0,0}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,1,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,1,1}, {0,1,1});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,0,0});
        
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,-1,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,-1,0});

        TFO.set_mix_trilinear_interaction(xi, 3, 0, 3, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 2, {1,0,0}, {1,0,0});

    
    }
    array<array<double, 3>,4> field_drive = {{{1,0,0},{1,0,0},{1,0,0},{1,0,0}}};

    double pulse_amp = 4;
    double pulse_width = 0.38;
    double pulse_freq = 0.33;

    int T_steps = abs(int((T_end-T_start)/T_step_size))+1;
    int tau_steps = abs(int((tau_end-tau_start)/tau_step_size))+1;
    tau_step_size = tau_end - tau_start < 0 ? - abs(tau_step_size) : abs(tau_step_size);
    T_step_size = T_end - T_start < 0 ? - abs(T_step_size) : abs(T_step_size);

    mixed_lattice<3, 4, 8, 4, 8, 8, 8> MC(&TFO, 2.5, 1.0);

    if (spin_config != ""){
        // Check if the spin configuration file exists
        try {
            MC.read_spin_from_file(spin_config);
            cout << "Successfully loaded spin configuration from " << spin_config << endl;
        } catch (const std::exception& e) {
            cout << "Error loading spin configuration: " << e.what() << endl;
            cout << "Falling back to simulated annealing." << endl;
            MC.simulated_annealing(Temp_start, Temp_end, 100000, 0, 1000, true);
            if (T_zero) {
                for (size_t i = 0; i < 100000; ++i) {
                    MC.deterministic_sweep();
                }
            }
        }
    } else {
        cout << "Warning: Spin configuration file " << spin_config << " does not exist. Using simulated annealing instead." << endl;
        MC.simulated_annealing(Temp_start, Temp_end, 100000, 0, 1000, true);
        if (T_zero) {
            for (size_t i = 0; i < 100000; ++i) {
                MC.deterministic_sweep();
                }
            }
        }

    MC.write_to_file_pos(dir+"/pos.txt");
    MC.write_to_file_spin(dir+"/spin_0.txt");

    if (rank==0){
        filesystem::create_directories(dir+"/M_time_0");
        MC.M_B_t(field_drive, 0.0, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_0/M0");
        ofstream run_param;
        run_param.open(dir + "/param.txt");
        run_param << tau_start << " " << tau_end << " " << tau_steps  << " " << T_start << " " << T_end << " " << T_steps << endl;
        run_param.close();
    }


    int tau_length = int(tau_steps/size);

    double current_tau = tau_start+tau_steps*rank/size*tau_step_size;

    for(int i=0; i< tau_length;++i){
        filesystem::create_directories(dir+"/M_time_"+ std::to_string(current_tau));
        cout << "Time: " << current_tau << endl;
        MC.M_B_t(field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+std::to_string(current_tau)+"/M1");
        MC.M_BA_BB_t(field_drive, 0.0, field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+std::to_string(current_tau)+"/M01");
        current_tau += tau_step_size;
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

void MD_TmFeO3_2DCS_cuda(double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double e1, double e2, double chii, double xii, double h, const array<double,3> &fielddir, array<array<double, 3>,4> &field_drive, string dir, bool T_zero=false, string spin_config="", bool if_zero_is_in_T_range=false){
    int initialized;

    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check for valid MPI size to prevent division by zero
    if (size <= 0) {
        cout << "Error: Invalid MPI size (" << size << "). Setting size to 1." << endl;
        size = 1;
    }
    
    // Get CUDA device count and assign GPUs to MPI ranks
    int device_count;
    cudaGetDeviceCount(&device_count);
    int device_id = rank % device_count;  // Distribute ranks across available GPUs
    cudaSetDevice(device_id);
    
    if (rank == 0) {
        cout << "Total MPI processes: " << size << ", Available GPUs: " << device_count << endl;
        cout << "GPU assignment - Each GPU handles ~" << (size + device_count - 1) / device_count << " MPI ranks" << endl;
    }
    cout << "Rank " << rank << " using GPU " << device_id << endl;
    
    // Synchronize all ranks before starting computations
    MPI_Barrier(MPI_COMM_WORLD);
    
    filesystem::create_directories(dir);
    TmFeO3_Fe<3> Fe_atoms;
    TmFeO3_Tm<8> Tm_atoms;
    // Define eta vectors
    array<array<double, 3>, 4> eta = {{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}};

    // Original exchange matrices
    array<array<double, 3>, 3> Ja_orig = {{{Jai, D2, -D1}, {-D2, Jai, 0}, {D1, 0, Jai}}};
    array<array<double, 3>, 3> Jb_orig = {{{Jbi, D2, -D1}, {-D2, Jbi, 0}, {D1, 0, Jbi}}};
    array<array<double, 3>, 3> Jc_orig = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};

    array<array<double, 3>, 3> J2a_orig = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    array<array<double, 3>, 3> J2b_orig = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    array<array<double, 3>, 3> J2c_orig = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};
    // Create 4x4 sublattice versions for each exchange matrix
    // First two indices denote sublattices i,j; last two are matrix indices
    array<array<array<array<double, 3>, 3>, 4>, 4> Ja;
    array<array<array<array<double, 3>, 3>, 4>, 4> Jb;
    array<array<array<array<double, 3>, 3>, 4>, 4> Jc;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2a;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2b;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2c;
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    Ja[i][j][a][b] = Ja_orig[a][b] * eta[i][a] * eta[j][b];
                    Jb[i][j][a][b] = Jb_orig[a][b] * eta[i][a] * eta[j][b];
                    Jc[i][j][a][b] = Jc_orig[a][b] * eta[i][a] * eta[j][b];
                    J2a[i][j][a][b] = J2a_orig[a][b] * eta[i][a] * eta[j][b];
                    J2b[i][j][a][b] = J2b_orig[a][b] * eta[i][a] * eta[j][b];
                    J2c[i][j][a][b] = J2c_orig[a][b] * eta[i][a] * eta[j][b];
                }
            }
        }
    }

    array<double, 9> K = {{Ka, 0, 0, 0, 0, 0, 0, 0, Kc}};
    //In plane interactions
    Fe_atoms.set_bilinear_interaction(Ja[1][0], 1, 0, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb[1][0], 1, 0, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb[1][0], 1, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja[1][0], 1, 0, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(Ja[2][3], 2, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb[2][3], 2, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb[2][3], 2, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja[2][3], 2, 3, {1,-1,0});

    //Next Nearest Neighbour
    Fe_atoms.set_bilinear_interaction(J2a[0][0], 0, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[0][0], 0, 0, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a[1][1], 1, 1, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[1][1], 1, 1, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a[2][2], 2, 2, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[2][2], 2, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a[3][3], 3, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[3][3], 3, 3, {0,1,0});

    //Out of plane interaction
    Fe_atoms.set_bilinear_interaction(Jc[0][3], 0, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc[0][3], 0, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(Jc[1][2], 1, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc[1][2], 1, 2, {0,0,1});

    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,1,0});

    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,1,1});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,1,1});

    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,-1,1});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,-1,1});

    //single ion anisotropy
    Fe_atoms.set_onsite_interaction(K, 0);
    Fe_atoms.set_onsite_interaction(K, 1);
    Fe_atoms.set_onsite_interaction(K, 2);
    Fe_atoms.set_onsite_interaction(K, 3);

    Fe_atoms.set_field(fielddir*h, 0);
    Fe_atoms.set_field(fielddir*h, 1);
    Fe_atoms.set_field(fielddir*h, 2);
    Fe_atoms.set_field(fielddir*h, 3);

    //Tm atoms
    //Set energy splitting for Tm atoms
    //\alpha\lambda3 + \beta\lambda8 + \gamma\identity
    double alpha = e1/2;
    double beta = sqrt(3)/6*(2*e2-e1);
    double gamma = -(e1+e2)/3 *3/16;

    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 0);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 1);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 2);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 3);



    TmFeO3<3, 8> TFO(&Fe_atoms, &Tm_atoms);
    // I have finally cracked the correct model...
    // Here we go!

    if (chii != 0.0){
        array<array<double,3>,8> chi = {{{0}}};
        chi[1] = {{0, 0, 5.264*chii}};
        chi[4] = {{2.3915*chii,2.7866*chii,0}};
        chi[6] = {{0.9128*chii,-0.4655*chii,0}};

        array<array<double,3>,8> chi_inv = {{{0}}};
        chi_inv[1] = {{0, 0, 5.264*chii}};
        chi_inv[4] = {{-2.3915*chii,-2.7866*chii,0}};
        chi_inv[6] = {{-0.9128*chii,0.4655*chii,0}};

        TFO.set_mix_bilinear_interaction(chi, 1, 0, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 1, 3, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 1, 1, {0,1,0});
        TFO.set_mix_bilinear_interaction(chi, 1, 2, {0,1,0});


        TFO.set_mix_bilinear_interaction(chi_inv, 1, 2, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 3, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 1, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 0, {1,0,0});

        ///////////////
        TFO.set_mix_bilinear_interaction(chi, 0, 0, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 0, 3, {0,0,1});
        TFO.set_mix_bilinear_interaction(chi, 0, 1, {0,1,0});
        TFO.set_mix_bilinear_interaction(chi, 0, 2, {0,1,1});

        TFO.set_mix_bilinear_interaction(chi_inv, 0, 2, {-1,1,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 0, 3, {0,1,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 0, 1, {-1,1,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 0, 0, {0,1,0});

        ///////////////
        TFO.set_mix_bilinear_interaction(chi, 2, 0, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 2, 3, {0,0,1});
        TFO.set_mix_bilinear_interaction(chi, 2, 1, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 2, 2, {0,0,1});

        TFO.set_mix_bilinear_interaction(chi_inv, 2, 2, {0,1,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 3, {1,0,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 1, {0,1,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 0, {1,0,0});

        ///////////////
        TFO.set_mix_bilinear_interaction(chi, 3, 0, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi, 3, 3, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi, 3, 1, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 3, 2, {0,0,0});

        TFO.set_mix_bilinear_interaction(chi_inv, 3, 2, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 3, {1,-1,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 1, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 0, {1,-1,0});
    }

    if (xii != 0.0){

        array<array<array<double,3>,3>,8> xi = {{{0}}};
        xi[0] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
        // xi[2] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
        // xi[7] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};

        ////////// Trilinear coupling/Oxygen path way
        TFO.set_mix_trilinear_interaction(xi, 1, 0, 3, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 1, 2, {0,1,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 2, 3, {0,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 1, 0, {0,0,0}, {1,0,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 1, 0, {0,1,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 2, 3, {0,1,0}, {1,0,0});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 2, 2, 3, {0,0,1}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 2, 2, 3, {0,1,1}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 2, 1, 2, {0,0,0}, {0,0,1});
        TFO.set_mix_trilinear_interaction(xi, 2, 0, 3, {1,0,0}, {1,0,1});
        //////////////////

        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,0,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,0,1}, {0,1,1});

        TFO.set_mix_trilinear_interaction(xi, 0, 1, 2, {-1,1,0}, {-1,1,1});
        TFO.set_mix_trilinear_interaction(xi, 0, 0, 3, {0,0,0}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,1,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,1,1}, {0,1,1});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,0,0});
        
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,-1,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,-1,0});

        TFO.set_mix_trilinear_interaction(xi, 3, 0, 3, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 2, {1,0,0}, {1,0,0});

        ///////////// Trilinear Interaction - Nearest neighbours

        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {1,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {1,0,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {1,0,1}, {0,0,1});
        TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {1,0,1}, {0,1,1});

        TFO.set_mix_trilinear_interaction(xi, 2, 0, 3, {0,0,0}, {0,0,1});
        TFO.set_mix_trilinear_interaction(xi, 2, 1, 2, {0,1,0}, {0,1,1});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,0,0}, {-1,1,0});
        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,1,0}, {-1,1,0});

        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,0,1}, {-1,1,1});
        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,1,1}, {-1,1,1});

        TFO.set_mix_trilinear_interaction(xi, 0, 0, 3, {0,1,0}, {0,1,1});
        TFO.set_mix_trilinear_interaction(xi, 0, 1, 2, {0,1,0}, {0,1,1});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {0,0,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {0,0,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 0, 3, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 2, 1, {0,0,0}, {0,0,0});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {1,0,0}, {1,-1,0});

        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {1,0,0}, {1,-1,0});

        TFO.set_mix_trilinear_interaction(xi, 3, 1, 2, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 0, 3, {1,-1,0}, {1,-1,0});
    }
    
    //
    cout << "Finished setting up TmFeO3 model." << endl;
    cout << "Starting calculations..." << endl;

    // Use the field_drive parameter passed to the function
    // array<array<double, 3>,4> field_drive = {{{1,0,0},{1,0,0},{1,0,0},{1,0,0}}};
    // array<array<double, 3>,4> field_drive = {{{0,0,1},{0,0,1},{0,0,1},{0,0,1}}};

    double pulse_amp = 4;
    double pulse_width = 0.38;
    double pulse_freq = 0.33;

    int T_steps = abs(int((T_end-T_start)/T_step_size))+1;
    int tau_steps = abs(int((tau_end-tau_start)/tau_step_size))+1;
    tau_step_size = tau_end - tau_start < 0 ? - abs(tau_step_size) : abs(tau_step_size);
    T_step_size = T_end - T_start < 0 ? - abs(T_step_size) : abs(T_step_size);

    mixed_lattice_cuda<3, 4, 8, 4, 4, 4, 4> MC(&TFO, 2.5, 1.0);

    cout << "Initialized mixed lattice with CUDA support." << endl;
    // Continue with the rest of the initialization code...
    if (spin_config != ""){
        // Check if the spin configuration file exists
        try {
            MC.read_spin_from_file(spin_config);
            cout << "Successfully loaded spin configuration from " << spin_config << endl;
        } catch (const std::exception& e) {
            cout << "Error loading spin configuration: " << e.what() << endl;
            cout << "Falling back to simulated annealing." << endl;
            MC.simulated_annealing(Temp_start, Temp_end, 100000, 0, 1000, true);
        }
    } else {
        cout << "No spin configuration specified. Using simulated annealing." << endl;
        MC.simulated_annealing(Temp_start, Temp_end, 100000, 0, 1000, true);
        MC.write_to_file_spin(dir+"/spin");
        spin_config = dir+"/spin";
    }

    if (T_zero) {
        for (size_t i = 0; i < 100000; ++i) {
            MC.deterministic_sweep();
        }
    }
    MC.write_to_file_spin(dir+"/spin_zero");
    spin_config = dir+"/spin_zero";

    MC.write_to_file_pos(dir+"/pos.txt");


    cout << "Starting calculations..." << endl;
    cout << "Drive field direction: " << fielddir[0] << ", " << fielddir[1] << ", " << fielddir[2] << endl;
    // MC.molecular_dynamics_cuda(0, 600, 1e-2, dir+"/spin_t.txt", 1);

    if (rank==0 && if_zero_is_in_T_range){
        ofstream run_param;
        run_param.open(dir + "/param.txt");
        run_param << tau_start << " " << tau_end << " " << tau_steps  << " " << T_start << " " << T_end << " " << T_steps << endl;
        run_param.close();
    }

    int tau_length = int(tau_steps/size);
    double current_tau = tau_start+tau_steps*rank/size*tau_step_size;

    for(int i=0; i< tau_length; ++i){
        filesystem::create_directories(dir+"/M_time_"+ std::to_string(current_tau));
        cout << "Time: " << current_tau << endl;
        // Use the CUDA versions of the methods
        MC.read_spin_from_file(spin_config);
        MC.M_B_t_cuda(field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+std::to_string(current_tau)+"/M1");
        MC.read_spin_from_file(spin_config);
        MC.M_BA_BB_t_cuda(field_drive, 0.0, field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+std::to_string(current_tau)+"/M01");
        current_tau += tau_step_size;
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

void TmFeO3_2DCS(size_t num_trials, double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double e1, double e2, double xii, double h, const array<double,3> &fielddir, string dir, bool T_zero=false, string spin_config=""){
    int initialized;

    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    filesystem::create_directories(dir);

    for(size_t i = 0; i < num_trials; ++i){
        MD_TmFeO3_Fe_2DCS(Temp_start, Temp_end, tau_start, tau_end, tau_step_size, T_start, T_end, T_step_size, Jai, Jbi, Jci, J2ai, J2bi, J2ci, Ka, Kc, D1,  D2, h,fielddir, dir+"/"+std::to_string(i), T_zero, spin_config);
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
    bool T_zero = (argc > 1) ? atoi(argv[1]) : 0;
    double Temp_start = (argc > 2) ? atof(argv[2]) : 20;
    double Temp_end = (argc > 3) ? atof(argv[3]) : 0.01;
    double tau_start = (argc > 4) ? atof(argv[4]) : 0;
    double tau_end = (argc > 5) ? atof(argv[5]) : -20;
    double tau_step_size = (argc > 6) ? atof(argv[6]) : 0.01;
    double T_start = (argc > 7) ? atof(argv[7]) : -20.0;
    double T_end = (argc > 8) ? atof(argv[8]) : 20.0;
    double T_step_size = (argc > 9) ? atof(argv[9]) : 0.01;


    double J1ab = (argc > 10) ? atof(argv[10]) : 4.92;
    double J1c = (argc > 11) ? atof(argv[11]) : 4.92;
    double J2ab = (argc > 12) ? atof(argv[12]) : 0.29;
    double J2c = (argc > 13) ? atof(argv[13]) : 0.29;
    double Ka = (argc > 14) ? atof(argv[14]) : 0.0;
    double Kc = (argc > 15) ? atof(argv[15]) : -0.09;
    double D1 = (argc > 16) ? atof(argv[16]) : 0.0;
    double D2 = (argc > 17) ? atof(argv[17]) : 0.0;
    double e1 = (argc > 18) ? atof(argv[18]) : 4.0;
    double e2 = (argc > 19) ? atof(argv[19]) : 0.0;
    double chii = (argc > 20) ? atof(argv[20]) : 0.05; // TmFeO bilinear coupling parameter
    double xii = (argc > 21) ? atof(argv[21]) : 0.0;
    double h = (argc > 22) ? atof(argv[22]) : 0.0;
    
    // Field drive components - default to x-direction [1,0,0] for all atoms
    double field_drive_x = (argc > 23) ? atof(argv[23]) : 1.0;
    double field_drive_y = (argc > 24) ? atof(argv[24]) : 0.0;
    double field_drive_z = (argc > 25) ? atof(argv[25]) : 0.0;

    J1c /= J1ab;
    J2ab /= J1ab;
    J2c /= J1ab;
    Ka /= J1ab;
    Kc /= J1ab;
    D1 /= J1ab;
    D2 /= J1ab;
    e1 /= J1ab;
    e2 /= J1ab;
    h /= J1ab;
    J1ab = 1;
    string dir_name = (argc > 26) ? argv[26] : "TmFeO3_2DCS_xii=0.05";
    filesystem::create_directories(dir_name);
    int slurm_ID = (argc > 27) ? atoi(argv[27]) : 1;
    int total_jobs = (argc > 28) ? atoi(argv[28]) : 1;
    string spin_config_file = (argc > 29) ? argv[29] : "TFO_4_0_xii=0.05/spin_zero.txt";

    cout << "Slurm ID: " << slurm_ID << ", Total Jobs: " << total_jobs << endl;

    double tau_length = (tau_end - tau_start);
    double tau_section = tau_length/total_jobs;
    double tau_start_here = tau_start + (slurm_ID-1)*tau_section;
    double tau_end_here = tau_start + tau_section;

    cout << "Initializing TmFeO3 2DCS calculation with parameters: J1ab: " << J1ab << " J1c: " << J1c << " J2ab: " << J2ab << " J2c: " << J2c << " Ka: " << Ka << " Kc: " << Kc << " D1: " << D1 << " D2: " << D2 << " H: " << h << " xi::" << xii << " saving to: " << dir_name << endl;
    cout << "Field drive: [" << field_drive_x << ", " << field_drive_y << ", " << field_drive_z << "]" << endl;
    cout << "Reading from " << spin_config_file << endl;
    string output_dir = dir_name;
    filesystem::create_directories(output_dir);
    bool if_zero_is_in_T_range = slurm_ID == 1;
    
    // Define field_drive using command line parameters
    array<array<double, 3>,4> field_drive = {{{field_drive_x, field_drive_y, field_drive_z},
                                              {field_drive_x, -field_drive_y, -field_drive_z},
                                              {-field_drive_x, field_drive_y, -field_drive_z},
                                              {-field_drive_x, -field_drive_y, field_drive_z}}};
    
    MD_TmFeO3_2DCS_cuda(Temp_start, Temp_end, tau_start_here, tau_end_here, tau_step_size, T_start, T_end, T_step_size, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, e1, e2, chii, xii, h, {0.0, 0.0, 1.0}, field_drive, output_dir, T_zero, spin_config_file, if_zero_is_in_T_range);
    return 0;
}

//J1ab=J1c=4.92meV J2ab=J2c=0.29meV Ka=0meV Kc=-0.09meV D1=D2=0 xii is the modeling param E1 = -0.97meV E2=-3.89134081434meV