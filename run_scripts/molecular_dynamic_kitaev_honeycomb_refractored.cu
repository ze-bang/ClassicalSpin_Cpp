#include "../src/unitcell.h"
#include "../src/lattice.h"
#include <iostream>
#include <mpi.h>
#include "../src/simple_linear_alg.h"
#include <omp.h>



// void parallel_tempering_honeycomb(double T_start, double T_end, double K, double Gamma, double Gammap, double h, string dir, const vector<int> &rank_to_write){
//     filesystem::create_directory(dir);
//     HoneyComb<3> atoms;
//     array<array<double,3>, 3> Jx = {{{K,Gammap,Gammap},{Gammap,0,Gamma},{Gammap,Gamma,0}}};
//     array<array<double,3>, 3> Jy = {{{0,Gammap,Gamma},{Gammap,K,Gammap},{Gamma,Gammap,0}}};
//     array<array<double,3>, 3> Jz = {{{0,Gamma,Gammap},{Gamma,0,Gammap},{Gammap,Gammap,K}}};


//     array<double, 3> field = {h/double(sqrt(3)),h/double(sqrt(3)),h/double(sqrt(3))};
    
//     atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
//     atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
//     atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
//     atoms.set_field(field, 0);
//     atoms.set_field(field, 1);

//     int initialized;
//     MPI_Initialized(&initialized);
//     if (!initialized){
//         MPI_Init(NULL, NULL);
//     }
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     vector<double> temps = logspace(log10(T_start), log10(T_end), size);


//     lattice MC(&atoms, 12, 12, 1);
//     MC.parallel_tempering(temps, 1e6, 1e6, 10, 50, 2e3, dir, rank_to_write);

//     int finalized;
//     MPI_Finalized(&finalized);
//     if (!finalized){
//         MPI_Finalize();
//     }
// }


// // kitaev local xyz {1,-1,-1} / {-1, 1, -1} / {-1, -1, 1}
// void nonlinearspectroscopy_kitaev_honeycomb(double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double K, double Gamma, double Gammap, double h, string dir, bool T_zero){
//     filesystem::create_directory(dir);
//     HoneyComb<3> atoms;
//     array<array<double,3>, 3> Jx = {{{K,Gammap,Gammap},{Gammap,0,Gamma},{Gammap,Gamma,0}}};
//     array<array<double,3>, 3> Jy = {{{0,Gammap,Gamma},{Gammap,K,Gammap},{Gamma,Gammap,0}}};
//     array<array<double,3>, 3> Jz = {{{0,Gamma,Gammap},{Gamma,0,Gammap},{Gammap,Gammap,K}}};

//     array<double, 3> field = {h/double(sqrt(3)),h/double(sqrt(3)),h/double(sqrt(3))};


//     atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
//     atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
//     atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
//     atoms.set_field(field, 0);
//     atoms.set_field(field, 1);

//     // array<array<double, 3>,2> field_drive = {{{-1/sqrt(3), -1/sqrt(3), 1/sqrt(3)},{-1/sqrt(3), -1/sqrt(3), 1/sqrt(3)}}};
//     array<array<double, 3>,2> field_drive = {{{0,0,1},{0,0,1}}};

//     double pulse_amp = 0.5;
//     double pulse_width = 0.38;
//     double pulse_freq = 0.33;

//     int T_steps = abs(int((T_end-T_start)/T_step_size))+1;
//     int tau_steps = abs(int((tau_end-tau_start)/tau_step_size))+1;
//     tau_step_size = tau_end - tau_start < 0 ? - abs(tau_step_size) : abs(tau_step_size);
//     T_step_size = T_end - T_start < 0 ? - abs(T_step_size) : abs(T_step_size);
//     lattice<3, 2, 6, 6, 1> MC(&atoms);
//     MC.simulated_annealing(Temp_start, Temp_end, 1000, 1);

//     if (T_zero){
//         for (size_t i = 0; i<1000; ++i){
//             MC.deterministic_sweep();
//         }
//     }
//     MC.write_to_file_pos(dir+"/pos.txt");
//     MC.write_to_file_spin(dir+"/spin_0.txt", MC.spins);

//     filesystem::create_directory(dir+"/M_time_0");
//     MC.M_B_t(field_drive, 0.0, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_0/M0");

//     ofstream run_param;
//     run_param.open(dir + "/param.txt");
//     run_param << tau_start << " " << tau_end << " " << tau_steps  << " " << T_start << " " << T_end << " " << T_steps << " " << K << " " << h << endl;
//     run_param.close();

//     double current_tau = tau_start;

//     for(int i=0; i< tau_steps;++i){
//         filesystem::create_directory(dir+"/M_time_"+ std::to_string(i));
//         cout << "Time: " << current_tau << endl;
//         MC.M_B_t(field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i) + "/M1");
//         MC.M_BA_BB_t(field_drive, 0.0, field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i)+ "/M01");
//         current_tau += tau_step_size;
//     }
// }


// void full_nonlinearspectroscopy_kitaev_honeycomb(size_t num_trials, double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, double K, double Gamma, double Gammap, double h, string dir, bool T_zero){
//     int initialized;

//     MPI_Initialized(&initialized);
//     if (!initialized){
//         MPI_Init(NULL, NULL);
//     }
//     int rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     int size;
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     filesystem::create_directory(dir);

//     // Print all simulation parameters
//     if (rank == 0) {
//         cout << "=====================================" << endl;
//         cout << "Simulation Parameters:" << endl;
//         cout << "=====================================" << endl;
//         cout << "Number of trials: " << num_trials << endl;
//         cout << "Temperature annealing: " << Temp_start << " -> " << Temp_end << endl;
//         cout << "Tau range: " << tau_start << " -> " << tau_end << endl;
//         cout << "Tau step size: " << tau_step_size << endl;
//         cout << "Time range: " << T_start << " -> " << T_end << endl;
//         cout << "Time step size: " << T_step_size << endl;
//         cout << "K (Kitaev): " << K << endl;
//         cout << "Gamma: " << Gamma << endl;
//         cout << "Gamma': " << Gammap << endl;
//         cout << "Magnetic field h: " << h << endl;
//         cout << "Field components: {" << h/sqrt(3) << ", " << h/sqrt(3) << ", " << h/sqrt(3) << "}" << endl;
//         cout << "T_zero mode: " << (T_zero ? "true" : "false") << endl;
//         cout << "Output directory: " << dir << endl;
//         cout << "MPI size: " << size << endl;
//         cout << "=====================================" << endl;

//         // Save parameters to file
//         ofstream param_file;
//         param_file.open(dir + "/simulation_params.txt");
//         param_file << "Simulation Parameters" << endl;
//         param_file << "=====================================" << endl;
//         param_file << "num_trials: " << num_trials << endl;
//         param_file << "Temp_start: " << Temp_start << endl;
//         param_file << "Temp_end: " << Temp_end << endl;
//         param_file << "tau_start: " << tau_start << endl;
//         param_file << "tau_end: " << tau_end << endl;
//         param_file << "tau_step_size: " << tau_step_size << endl;
//         param_file << "T_start: " << T_start << endl;
//         param_file << "T_end: " << T_end << endl;
//         param_file << "T_step_size: " << T_step_size << endl;
//         param_file << "K: " << K << endl;
//         param_file << "Gamma: " << Gamma << endl;
//         param_file << "Gammap: " << Gammap << endl;
//         param_file << "h: " << h << endl;
//         param_file << "field_x: " << h/sqrt(3) << endl;
//         param_file << "field_y: " << h/sqrt(3) << endl;
//         param_file << "field_z: " << h/sqrt(3) << endl;
//         param_file << "T_zero: " << (T_zero ? "true" : "false") << endl;
//         param_file << "MPI_size: " << size << endl;
//         param_file.close();
//     }

//     for(size_t i = 0; i < num_trials; ++i){
//         nonlinearspectroscopy_kitaev_honeycomb(Temp_start, Temp_end, tau_start, tau_end, tau_step_size, T_start, T_end, T_step_size, K, Gamma, Gammap, h, dir+"/"+std::to_string(i), T_zero);
//     }
//     int finalized;
//     MPI_Finalized(&finalized);
//     if (!finalized){
//         MPI_Finalize();
//     }
// }

void MD_kitaev_honeycomb_real(size_t num_trials, string dir, double J=0, double K=-1, double Gamma=0.25, double Gammap=-0.02, double h=0.7){
    filesystem::create_directory(dir);
    HoneyComb_alt atoms(3);
    
    // Create Eigen matrices for interactions
    Eigen::Matrix3d Jx, Jy, Jz;
    Jx << J+K, Gammap, Gammap,
          Gammap, J, Gamma,
          Gammap, Gamma, J;
    Jy << J, Gammap, Gamma,
          Gammap, J+K, Gammap,
          Gamma, Gammap, J;
    Jz << J, Gamma, Gammap,
          Gamma, J, Gammap,
          Gammap, Gammap, J+K;

    // Create field vector
    Eigen::Vector3d field(h/sqrt(3), h/sqrt(3), h/sqrt(3));
    
    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    for(size_t i=0; i<num_trials;++i){
        Lattice MC(atoms, 24, 24, 1);
        MC.simulated_annealing(5, 1e-3, 1000, 1);
        MC.save_spin_config(dir+"/spins_initial_"+std::to_string(i)+".txt");
        MC.molecular_dynamics(0, 600, 0.25, dir+"/"+std::to_string(i), 1, "rk54", true);
    }
}

void dump_new_lattice_info(double J=0, double K=-1, double Gamma=0.25, double Gammap=-0.02, double h=0.7){
    std::cout << "========================================" << std::endl;
    std::cout << "New Lattice Implementation Info" << std::endl;
    std::cout << "========================================" << std::endl;
    
    HoneyComb_alt atoms(3);
    Eigen::Matrix3d Jx, Jy, Jz;
    Jx << J+K, Gammap, Gammap,
          Gammap, J, Gamma,
          Gammap, Gamma, J;
    Jy << J, Gammap, Gamma,
          Gammap, J+K, Gammap,
          Gamma, Gammap, J;
    Jz << J, Gamma, Gammap,
          Gamma, J, Gammap,
          Gammap, Gammap, J+K;
    Eigen::Vector3d field(h/sqrt(3), h/sqrt(3), h/sqrt(3));
    
    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    
    Lattice MC(atoms, 24, 24, 1);
    
    // Output basic properties
    std::cout << "Lattice size: " << MC.lattice_size << std::endl;
    std::cout << "num_bi: " << MC.num_bi << std::endl;
    std::cout << "num_tri: " << MC.num_tri << std::endl;
    std::cout << "spin_length: " << MC.spin_length << std::endl;
    
    // Save detailed info to file
    filesystem::create_directory("lattice_test_output");
    ofstream out("lattice_test_output/new_lattice_info.txt");
    
    out << "lattice_size: " << MC.lattice_size << "\n";
    out << "num_bi: " << MC.num_bi << "\n";
    out << "num_tri: " << MC.num_tri << "\n\n";
    
    // Sample first 10 sites
    for(size_t site = 0; site < min(MC.lattice_size, size_t(10)); ++site) {
        out << "Site " << site << ":\n";
        out << "  Position: " << MC.site_positions[site].transpose() << "\n";
        out << "  Field: " << MC.field[site].transpose() << "\n";
        out << "  Spin: " << MC.spins[site].transpose() << "\n";
        out << "  Num bilinear partners: " << MC.bilinear_partners[site].size() << "\n";
        out << "  Partners: ";
        for(size_t p = 0; p < MC.bilinear_partners[site].size(); ++p) {
            out << MC.bilinear_partners[site][p] << " ";
        }
        out << "\n";
        
        // Output interaction matrices
        for(size_t i = 0; i < min(MC.bilinear_interaction[site].size(), size_t(3)); ++i) {
            out << "  Interaction " << i << " matrix:\n";
            out << MC.bilinear_interaction[site][i] << "\n";
        }
        
        // Compute site energy
        double site_energy = MC.site_energy(MC.spins[site], site);
        out << "  Site energy: " << fixed << setprecision(12) << site_energy << "\n\n";
    }
    
    // Total energy
    double total_energy = MC.total_energy(MC.spins);
    out << "Total energy: " << fixed << setprecision(12) << total_energy << "\n";
    out << "Energy density: " << fixed << setprecision(12) << MC.energy_density() << "\n";
    
    // Save spin configuration for comparison (set to simple test config)
    // Set all spins to [0, 0, 1] for deterministic comparison
    for(size_t site = 0; site < MC.lattice_size; ++site) {
        MC.spins[site](0) = 0.0;
        MC.spins[site](1) = 0.0;
        MC.spins[site](2) = 1.0;
    }
    
    // Recompute energy with deterministic spins
    total_energy = MC.total_energy(MC.spins);
    std::cout << "Total energy (all spins=[0,0,1]): " << fixed << setprecision(12) << total_energy << std::endl;
    
    ofstream spin_out("lattice_test_output/new_spins.txt");
    for(size_t site = 0; site < MC.lattice_size; ++site) {
        spin_out << MC.spins[site].transpose() << "\n";
    }
    spin_out.close();
    
    out.close();
    
    std::cout << "Detailed info saved to lattice_test_output/new_lattice_info.txt" << std::endl;
    std::cout << "Spin configuration saved to lattice_test_output/new_spins.txt" << std::endl;
    std::cout << "Total energy: " << fixed << setprecision(12) << total_energy << std::endl;
    std::cout << "========================================" << std::endl;
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
    double K = (argc > 1) ? atof(argv[1]) : -1.0;
    double Gamma = (argc > 2) ? atof(argv[2]) : 0.25;
    double Gammap = (argc > 3) ? atof(argv[3]) : -0.02;
    double h = (argc > 4) ? atof(argv[4]) : 0.7;
    string dir_name = (argc > 5) ? argv[5] : "";
    if (!dir_name.empty()) {
        filesystem::create_directory(dir_name);
    }
    int num_trials = (argc > 6) ? atoi(argv[6]) : 1;
    double J = (argc > 7) ? atof(argv[7]) : 0.0;
    
    // Run info dump for new implementation
    // dump_new_lattice_info(J, K, Gamma, Gammap, h);
    
    // Optionally run MD simulation
    MD_kitaev_honeycomb_real(1, "KITAEV");
    
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}