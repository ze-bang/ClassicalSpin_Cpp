#include "experiments.h"
#include <math.h>
#include <mpi.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

using namespace std;

// Structure to hold all simulation parameters
struct SimulationParams {
    // Magnetic field and output directory
    double h = 0.0;
    array<double, 3> field_dir = {0, 1, 0};
    string dir = "BCAO_PT_simulation";

    // Exchange interaction parameters from Emily's model
    double J1xy = -7.6, J1z = -1.2, D = 0.1, E = -0.1, F = 0, G = 0;
    double J3xy = 2.5, J3z = -0.85;

    // Parallel Tempering parameters
    double T_start = 1e-2;
    double T_end = 30.0;
    size_t thermalization_sweeps = 1000000;
    size_t measurement_sweeps = 1000000;
    size_t overrelaxation_rate = 10; // Overrelaxation rate for the simulation
    size_t swap_interval = 50;
    size_t probe_rate = 2000;

    size_t num_trials = 5; // Number of trials for the simulation
};

// Function to read parameters from a file
SimulationParams read_parameters(const string& filename) {
    SimulationParams params;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cout << "Warning: Could not open parameter file '" << filename << "'. Using default parameters.\n";
        return params;
    }
    
    string line;
    while (getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#' || line[0] == '/') continue;
        
        istringstream iss(line);
        string key, value;
        if (getline(iss, key, '=') && getline(iss, value)) {
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            // Common parameters
            if (key == "h") params.h = stod(value);
            else if (key == "field_dir") {
                stringstream ss(value);
                string item;
                int i = 0;
                while (getline(ss, item, ',') && i < 3) {
                    params.field_dir[i++] = stod(item);
                }
            }
            else if (key == "num_trials") params.num_trials = stoi(value);
            else if (key == "dir") params.dir = value;
            // Model parameters
            else if (key == "J1xy") params.J1xy = stod(value);
            else if (key == "J1z") params.J1z = stod(value);
            else if (key == "D") params.D = stod(value);
            else if (key == "E") params.E = stod(value);
            else if (key == "F") params.F = stod(value);
            else if (key == "G") params.G = stod(value);
            else if (key == "J3xy") params.J3xy = stod(value);
            else if (key == "J3z") params.J3z = stod(value);
            // PT parameters
            else if (key == "T_start") params.T_start = stod(value);
            else if (key == "T_end") params.T_end = stod(value);
            else if (key == "thermalization_sweeps") params.thermalization_sweeps = stoul(value);
            else if (key == "measurement_sweeps") params.measurement_sweeps = stoul(value);
            else if (key == "overrelaxation_rate") params.overrelaxation_rate = stoul(value);
            else if (key == "swap_interval") params.swap_interval = stoul(value);
            else if (key == "probe_rate") params.probe_rate = stoul(value);
        }
    }
    
    file.close();
    return params;
}

// Function to create a default parameter file
void create_default_parameter_file(const string& filename) {
    ofstream file(filename);
    file << "# BCAO Honeycomb Parallel Tempering Parameters\n";
    file << "# Lines starting with # are comments\n";
    file << "# Format: parameter_name = value\n\n";
    
    file << "# Magnetic field strength in mu_B units\n";
    file << "h = 0.0\n\n";
    file << "# Field direction (x,y,z)\n";
    file << "field_dir = 0,1,0\n\n";
    file << "# Output directory\n";
    file << "dir = BCAO_PT_simulation\n\n";
    
    file << "# Exchange interaction parameters\n";
    file << "J1xy = -7.6\n";
    file << "J1z = -1.2\n";
    file << "D = 0.1\n";
    file << "E = -0.1\n";
    file << "F = 0\n";
    file << "G = 0\n\n";
    file << "# Third nearest neighbor parameters\n";
    file << "J3xy = 2.5\n";
    file << "J3z = -0.85\n\n";

    file << "# Parallel Tempering parameters\n";
    file << "T_start = 1.0\n";
    file << "T_end = 25.0\n";
    file << "thermalization_sweeps = 1000000\n";
    file << "measurement_sweeps = 1000000\n";
    file << "swap_interval = 10\n";
    file << "measurement_interval = 50\n";
    file << "num_bins = 2000\n";

    file.close();
}

// Main simulation function for Parallel Tempering
void PT_BCAO_honeycomb(const SimulationParams& params){
    filesystem::create_directory(params.dir);
    HoneyComb_standarx<3> atoms;

    // Define interaction matrices based on Emily's model
    array<array<double,3>, 3> J1z_ = {{{params.J1xy+params.D, params.E, params.F},{params.E, params.J1xy-params.D, params.G},{params.F, params.G, params.J1z}}};
    array<array<double,3>, 3> U_2pi_3 = {{{cos(2*M_PI/3), sin(2*M_PI/3), 0},{-sin(2*M_PI/3), cos(2*M_PI/3), 0},{0, 0, 1}}};

    auto transpose = [](const array<array<double,3>, 3>& m) {
        array<array<double,3>, 3> res;
        for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) res[i][j] = m[j][i];
        return res;
    };

    auto multiply = [](const array<array<double,3>, 3>& A, const array<array<double,3>, 3>& B) {
        array<array<double,3>, 3> C;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                C[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    };

    array<array<double,3>, 3> J1x_ = multiply(multiply(U_2pi_3, J1z_), transpose(U_2pi_3));
    array<array<double,3>, 3> J1y_ = multiply(multiply(transpose(U_2pi_3), J1z_), U_2pi_3);
    array<array<double,3>, 3> J3_ = {{{params.J3xy,0,0},{0,params.J3xy,0},{0,0,params.J3z}}};

    array<double, 3> field = {4.8*params.h*params.field_dir[0], 4.85*params.h*params.field_dir[1], 2.5*params.h*params.field_dir[2]};
    
    // Set interactions
    atoms.set_bilinear_interaction(J1x_, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J1y_, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    // MPI setup
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Temperature schedule
    vector<double> temps = logspace(log10(params.T_start), log10(params.T_end), size);

    // Lattice and simulation
    lattice<3, 2, 36, 36, 1> MC(&atoms, 1);
    MC.parallel_tempering(temps, params.thermalization_sweeps, params.measurement_sweeps, params.overrelaxation_rate, params.swap_interval, params.probe_rate, params.dir, {0});
    if (rank == 0) {
        cout << "Parallel Tempering simulation completed. Results saved in: " << params.dir << "\n";
        MC.write_to_file_spin(params.dir + "/spin.txt", MC.spins);
        for (size_t i = 0; i < 1e4; ++i) {
            MC.deterministic_sweep();
        }
        MC.write_to_file_spin(params.dir + "/spin_zero.txt", MC.spins);
    }
}

int main(int argc, char** argv) {
    string param_file = "bcao_pt_parameters.txt";

    if (argc > 1) {
        string arg = argv[1];
        if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [parameter_file]\n";
            cout << "  parameter_file    Path to parameter file (default: bcao_pt_parameters.txt)\n";
            cout << "  --help, -h        Show this help message\n";
            cout << "  --create-params   Create default parameter file\n\n";
            cout << "If parameter file doesn't exist, default parameters will be used.\n";
            return 0;
        } else if (arg == "--create-params") {
            create_default_parameter_file("bcao_pt_parameters.txt");
            cout << "Created default parameter file: bcao_pt_parameters.txt\n";
            cout << "Edit this file and run the simulation again.\n";
            return 0;
        } else {
            param_file = arg;
        }
    }
    
    SimulationParams params = read_parameters(param_file);
    
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(&argc, &argv);
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        cout << "Running PT_BCAO_honeycomb with parameters from: " << param_file << "\n";
        cout << "Field: " << params.h << " mu_B, Direction: [" << params.field_dir[0] << "," << params.field_dir[1] << "," << params.field_dir[2] << "]\n";
        cout << "Temperature Range: " << params.T_start << "K to " << params.T_end << "K\n";
        cout << "Output directory: " << params.dir << "\n";
        filesystem::create_directory(params.dir);
    }
    
    for (size_t i = 0; i < params.num_trials; ++i) {
        SimulationParams trial_params = params;
        trial_params.dir = params.dir + "/trial_" + to_string(i);

        if (rank == 0) {
            cout << "Starting trial " << i + 1 << " of " << params.num_trials << "\n";
        }
        
        // Synchronize all processes before starting the next trial
        MPI_Barrier(MPI_COMM_WORLD);

        // Run the Parallel Tempering simulation
        PT_BCAO_honeycomb(trial_params);
    }
    
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}