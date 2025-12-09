#include "experiments.h"
#include <math.h>
#include <mpi.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

struct SimulationParams {
    size_t num_trials = 1;
    double h = 0.0;
    array<double, 3> field_dir = {0, 1, 0};
    string dir = "BCAO_simulation";
    double J1xy = -7.65, J1z = -1.2, D = 0.1, E = -0.1, F = 0, G = 0;
    double J3xy = 2.64, J3z = -0.81;
    double h_start = 0.0, h_end = 1.0;
    int num_steps = 50;
    // Twist matrices (3 blocks of 3x3, each flattened to 9 doubles), optional
    bool has_twist_matrix = false;
    array<array<double, 9>, 3> twist_matrix = {{{
        1,0,0, 0,1,0, 0,0,1
    },{
        1,0,0, 0,1,0, 0,0,1
    },{
        1,0,0, 0,1,0, 0,0,1
    }}};
    bool tbc = false; // Whether to apply twist boundary conditions
    
    // Phase diagram parameters (legacy 2D support)
    string param1_name = "";  // e.g., "h", "J1xy", "D", "E", etc.
    double param1_start = 0.0;
    double param1_end = 1.0;
    int param1_steps = 10;
    
    string param2_name = "";  // e.g., "h", "J1xy", "D", "E", etc.
    double param2_start = 0.0;
    double param2_end = 1.0;
    int param2_steps = 10;
    
    // N-dimensional phase diagram parameters
    vector<string> scan_params;      // List of parameter names to scan
    vector<double> scan_starts;      // Starting values for each parameter
    vector<double> scan_ends;        // Ending values for each parameter
    vector<int> scan_steps;          // Number of steps for each parameter
};

SimulationParams read_parameters(const string& filename) {
    SimulationParams params;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cout << "Warning: Could not open parameter file '" << filename << "'. Using default parameters.\n";
        return params;
    }
    
    string line;
    array<bool, 3> twist_row_found = {false, false, false};
    const double k_B = 0.08620689655; // meV/K
    const double mu_B = 0.05788; // meV/T
    const double mu_B_k_B = mu_B / k_B; // K/T
    while (getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#' || line[0] == '/') continue;
        
        istringstream iss(line);
        string key, value;
        if (getline(iss, key, '=') && getline(iss, value)) {
            // Remove whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (key == "num_trials") params.num_trials = stoi(value);
            else if (key == "h") params.h = stod(value);
            else if (key == "field_dir") {
                stringstream ss(value);
                string item;
                int i = 0;
                while (getline(ss, item, ',') && i < 3) {
                    params.field_dir[i++] = stod(item);
                }
            }
            else if (key == "dir") params.dir = value;
            else if (key == "J1xy") params.J1xy = stod(value);
            else if (key == "J1z") params.J1z = stod(value);
            else if (key == "D") params.D = stod(value);
            else if (key == "E") params.E = stod(value);
            else if (key == "F") params.F = stod(value);
            else if (key == "G") params.G = stod(value);
            else if (key == "J3xy") params.J3xy = stod(value);
            else if (key == "J3z") params.J3z = stod(value);
            else if (key == "h_start") params.h_start = stod(value);
            else if (key == "h_end") params.h_end = stod(value);
            else if (key == "num_steps") params.num_steps = stoi(value);
            else if (key == "param1_name") params.param1_name = value;
            else if (key == "param1_start") params.param1_start = stod(value);
            else if (key == "param1_end") params.param1_end = stod(value);
            else if (key == "param1_steps") params.param1_steps = stoi(value);
            else if (key == "param2_name") params.param2_name = value;
            else if (key == "param2_start") params.param2_start = stod(value);
            else if (key == "param2_end") params.param2_end = stod(value);
            else if (key == "param2_steps") params.param2_steps = stoi(value);
            else if (key == "scan_params") {
                stringstream ss(value);
                string item;
                while (getline(ss, item, ',')) {
                    item.erase(0, item.find_first_not_of(" \t"));
                    item.erase(item.find_last_not_of(" \t") + 1);
                    if (!item.empty()) params.scan_params.push_back(item);
                }
            }
            else if (key == "scan_starts") {
                stringstream ss(value);
                string item;
                while (getline(ss, item, ',')) {
                    params.scan_starts.push_back(stod(item));
                }
            }
            else if (key == "scan_ends") {
                stringstream ss(value);
                string item;
                while (getline(ss, item, ',')) {
                    params.scan_ends.push_back(stod(item));
                }
            }
            else if (key == "scan_steps") {
                stringstream ss(value);
                string item;
                while (getline(ss, item, ',')) {
                    params.scan_steps.push_back(stoi(item));
                }
            }
            else if (key.rfind("twist_matrix_", 0) == 0) {
                int idx = -1;
                try {
                    idx = stoi(key.substr(13));
                } catch (...) {
                    idx = -1;
                }
                if (idx >= 0 && idx < 3) {
                    string v = value;
                    for (char &ch : v) if (ch == ',') ch = ' ';
                    stringstream vss(v);
                    for (int j = 0; j < 9; ++j) {
                        double val; 
                        if (!(vss >> val)) break; 
                        params.twist_matrix[idx][j] = val;
                    }
                    twist_row_found[idx] = true;
                }
            }
            else if (key == "tbc") {
                if (value == "1" || value == "true" || value == "True") {
                    params.tbc = true;
                } else if (value == "0" || value == "false" || value == "False") {
                    params.tbc = false;
                } else {
                    cout << "Warning: Invalid value for tbc; expected 0/1 or true/false. Keeping default (" << (params.tbc ? "true" : "false") << ").\n";
                }
            }
        }
    }
    
    if (twist_row_found[0] && twist_row_found[1] && twist_row_found[2]) {
        params.has_twist_matrix = true;
    }
    
    file.close();
    return params;
}

void create_default_parameter_file(const string& filename) {
    ofstream file(filename);
    file << "# BCAO Honeycomb Molecular Dynamics Parameters\n";
    file << "# Lines starting with # are comments\n";
    file << "# Format: parameter_name = value\n\n";
    file << "# Number of simulation trials\n";
    file << "num_trials = 5\n\n";
    file << "# Magnetic field strength in mu_B units\n";
    file << "h = 0.0\n\n";
    file << "# Field direction (x,y,z)\n";
    file << "field_dir = 0,1,0\n\n";
    file << "# Output directory\n";
    file << "dir = BCAO_simulation\n\n";
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
    file << "# Field scan parameters (for --field-scan mode)\n";
    file << "h_start = 0.0\n";
    file << "h_end = 1.0\n";
    file << "num_steps = 50\n\n";
    file << "# Generic phase diagram parameters (for --phase-diagram mode)\n";
    file << "# Available parameter names: h, J1xy, J1z, D, E, F, G, J3xy, J3z\n";
    file << "param1_name = h\n";
    file << "param1_start = 0.0\n";
    file << "param1_end = 1.0\n";
    file << "param1_steps = 10\n\n";
    file << "param2_name = J1xy\n";
    file << "param2_start = -8.0\n";
    file << "param2_end = -7.0\n";
    file << "param2_steps = 10\n\n";
    file << "# N-dimensional phase diagram (for --ndim-scan mode)\n";
    file << "# Comma-separated lists - all must have same length\n";
    file << "# Examples:\n";
    file << "#   1D: scan_params = h\n";
    file << "#   2D: scan_params = h,D\n";
    file << "#   3D: scan_params = h,D,J1xy\n";
    file << "#   4D: scan_params = h,D,E,J3xy\n";
    file << "# scan_params = h,D,E\n";
    file << "# scan_starts = 0.0,-0.2,-0.3\n";
    file << "# scan_ends = 1.0,0.5,0.2\n";
    file << "# scan_steps = 10,10,10\n\n";
    file << "# Twist boundary conditions (0 or 1)\n";
    file << "tbc = 0\n\n";
    file << "# Twist matrices (each 3x3 flattened, comma-separated)\n";
    file << "twist_matrix_0 = 1,0,0, 0,1,0, 0,0,1\n";
    file << "twist_matrix_1 = 1,0,0, 0,1,0, 0,0,1\n";
    file << "twist_matrix_2 = 1,0,0, 0,1,0, 0,0,1\n";
    file.close();
}


void sim_BCAO_honeycomb(size_t num_trials, double h, array<double, 3> field_dir, string dir, double J1xy=-7.6, double J1z=-1.2, double D=0.1, double E=-0.1, double F=0, double G=0, double J3xy=2.5, double J3z = -0.85, const array<array<double, 9>, 3>* custom_twist = nullptr, bool field_scan = false, bool tbc = false, const vector<size_t>* trial_indices = nullptr, vector<pair<size_t, double>>* trial_results = nullptr) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const double k_B = 1;
    const double mu_B = 0.05788; // meV/T
    filesystem::create_directories(dir);
    HoneyComb<3> atoms;

    J1xy *= k_B;
    J1z *= k_B;
    D *= k_B;
    E *= k_B;
    F *= k_B;
    G *= k_B;
    J3xy *= k_B;
    J3z *= k_B;

    array<array<double,3>, 3> J1z_ = {{{J1xy+D, E, F},
                                        {E, J1xy-D, G},
                                        {F, G, J1z}}};
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

    array<array<double,3>, 3> J3_ = {{{J3xy,0,0},{0,J3xy,0},{0,0,J3z}}};

    if (rank == 0) {
        std::cout << field_dir[0] << " " << field_dir[1] << " " << field_dir[2] << std::endl;
    }
    array<double, 3> field = {5*h*mu_B*field_dir[0],5*h*mu_B*field_dir[1],2.5*h*mu_B*field_dir[2]};

    //nearest neighbour
    atoms.set_bilinear_interaction(J1x_, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J1y_, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});

    //third nearest neighbour
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});

    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    vector<size_t> generated_indices;
    const vector<size_t>* indices_ptr = trial_indices;
    if (!indices_ptr || indices_ptr->empty()) {
        generated_indices.resize(num_trials);
        std::iota(generated_indices.begin(), generated_indices.end(), 0);
        indices_ptr = &generated_indices;
    }
    const vector<size_t>& trials_to_run = *indices_ptr;
    const size_t actual_trial_count = trials_to_run.size();

    if (trial_results) {
        trial_results->clear();
        trial_results->reserve(actual_trial_count);
    }

    // Save simulation parameters (only rank 0)
    if (rank == 0 || field_scan) {
        ofstream param_file(dir + "/simulation_parameters.txt");
        param_file << "Simulation Parameters for BCAO Honeycomb MD\n";
        param_file << "==========================================\n";
        param_file << "Number of trials requested: " << num_trials << "\n";
        param_file << "Trials executed in this call: " << actual_trial_count << "\n";
        param_file << "Magnetic field strength (h): " << h << "\n";
        param_file << "Field direction: [" << field_dir[0] << ", " << field_dir[1] << ", " << field_dir[2] << "]\n";
        param_file << "Applied field: [" << field[0] << ", " << field[1] << ", " << field[2] << "]\n";
        param_file << "J1xy: " << J1xy << "\n";
        param_file << "J1z: " << J1z << "\n";
        param_file << "D: " << D << "\n";
        param_file << "E: " << E << "\n";
        param_file << "F: " << F << "\n";
        param_file << "G: " << G << "\n";
        param_file << "J3xy: " << J3xy << "\n";
        param_file << "J3z: " << J3z << "\n";
        param_file << "Lattice size: 36x36x1\n";
        param_file << "Temperature for SA: 5K to 0.001K\n";
        param_file << "SA steps: 100000\n";
        param_file << "MD steps: 100\n";
        param_file << "MD timestep: 0.01\n";
        param_file << "MPI processes: " << size << "\n";
        param_file.close();
    }

    // Variables for storing local minimum energy
    double local_min_energy = std::numeric_limits<double>::max();
    int local_min_index = -1;

    // Run requested trials
    for (size_t trial_id : trials_to_run) {
        filesystem::create_directories(dir + "/" + std::to_string(trial_id));
        lattice<3, 2, 18, 18, 1> MC(&atoms, 0.5, true);
        // auto SA_params = MC.tune_simulated_annealing(0.5, 10.0, false, 20, 1000, 0.7, 0.05);
        // {
        //     std::ostringstream oss;
        //     oss.setf(std::ios::fixed);
        //     oss.precision(10);
        //     oss << "Process " << rank << " Trial " << trial_id
        //     << " SA params: T_start=" << SA_params.T_start
        //     << ", T_end=" << SA_params.T_end
        //     << ", sweeps_per_temp=" << SA_params.sweeps_per_temp
        //     << ", cooling_rate=" << SA_params.cooling_rate << "\n";
        //     cout << oss.str();
        // }
        // MC.simulated_annealing(SA_params.T_start, SA_params.T_end, SA_params.sweeps_per_temp*100, 20, tbc, false, SA_params.cooling_rate, dir +"/"+std::to_string(trial_id), true);
        MC.simulated_annealing(10*k_B, 0.1*k_B, 1e6, 10, tbc, false, 0.9, dir +"/"+std::to_string(trial_id), true);
        double energy_density = MC.energy_density(MC.spins);
        ofstream energy_file(dir +"/"+std::to_string(trial_id)+ "/energy_density.txt");
        energy_file << "Energy Density: " << energy_density << "\n";
        energy_file.close();      
        ofstream twist_file(dir +"/"+std::to_string(trial_id)+ "/twist_matrix.txt");
        twist_file << "Twist Matrix:\n";
        for (const auto& row : MC.twist_matrices) {
            for (const auto& val : row) {
                twist_file << val << " ";
            }
            twist_file << "\n";
        }
        twist_file.close();

        const auto energy_langscape = MC.local_energy_densities(MC.spins);
        ofstream landscape_file(dir +"/"+std::to_string(trial_id)+ "/energy_landscape.txt");
        landscape_file << "Energy Landscape:\n";
        for (size_t j = 0; j < energy_langscape.size(); ++j) {
            landscape_file << j << " " << energy_langscape[j] << "\n";
        }
        landscape_file.close();

        if (trial_results) {
            trial_results->emplace_back(trial_id, energy_density);
        }

        // Update local minimum
        if (energy_density < local_min_energy) {
            local_min_energy = energy_density;
            local_min_index = static_cast<int>(trial_id);
        }
    }
    // Only perform MPI collective operations if not in field_scan mode
    // or if there are multiple trials to aggregate
    if (!field_scan && actual_trial_count > 1) {
        // Gather all minimum energies to rank 0
        struct {
            double energy;
            int index;
            int rank;
        } local_min = {local_min_energy, local_min_index, rank}, global_min;

        MPI_Reduce(&local_min.energy, &global_min.energy, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        
        // Find which process has the global minimum
        if (rank == 0) {
            double *all_energies = new double[size];
            int *all_indices = new int[size];
            
            MPI_Gather(&local_min_energy, 1, MPI_DOUBLE, all_energies, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(&local_min_index, 1, MPI_INT, all_indices, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            double min_energy = all_energies[0];
            int min_index = all_indices[0];
            
            for (int i = 1; i < size; ++i) {
                if (all_energies[i] < min_energy) {
                    min_energy = all_energies[i];
                    min_index = all_indices[i];
                }
            }
            
            // Output the information about the best configuration to a file
            ofstream best_config_file(dir + "/best_configuration.txt");
            best_config_file << "Best Configuration Found:\n";
            best_config_file << "Trial Index: " << min_index << "\n";
            best_config_file << "Minimum Energy Density: " << min_energy << "\n";
            best_config_file.close();
            
            delete[] all_energies;
            delete[] all_indices;
        } else {
            MPI_Gather(&local_min_energy, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(&local_min_index, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
        }
    } else if (!field_scan && local_min_index >= 0) {
        // For single trial, just output the local result if we have one
        if (rank == 0) {
            ofstream best_config_file(dir + "/best_configuration.txt");
            best_config_file << "Best Configuration Found:\n";
            best_config_file << "Trial Index: " << local_min_index << "\n";
            best_config_file << "Minimum Energy Density: " << local_min_energy << "\n";
            best_config_file.close();
        }
    }
    
    // Only call barrier if not in field_scan mode
    // In field_scan mode, the barrier should be called in magnetic_field_scan function
    if (!field_scan) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void sim_BCAO_honeycomb_restarted(size_t num_trials, double h, array<double, 3> field_dir, string dir, double J1xy=-7.6, double J1z=-1.2, double D=0.1, double E=-0.1, double F=0, double G=0, double J3xy=2.5, double J3z = -0.85, const array<array<double, 9>, 3>* custom_twist = nullptr){
    filesystem::create_directories(dir);
    HoneyComb<3> atoms;


    array<array<double,3>, 3> J1z_ = {{{J1xy+D, E, F},
                                        {E, J1xy-D, G},
                                        {F, G, J1z}}};
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

    array<array<double,3>, 3> J3_ = {{{J3xy,0,0},{0,J3xy,0},{0,0,J3z}}};

    std::cout << field_dir[0] << " " << field_dir[1] << " " << field_dir[2] << std::endl;
    array<double, 3> field = {h*field_dir[0],h*field_dir[1],h*field_dir[2]};
    

    //nearest neighbour
    atoms.set_bilinear_interaction(J1x_, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J1y_, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});

    //third nearest neighbour
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});

    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    double k_B = 0.08620689655;

    // Save simulation parameters
    ofstream param_file(dir + "/simulation_parameters.txt");
    param_file << "Simulation Parameters for BCAO Honeycomb MD\n";
    param_file << "==========================================\n";
    param_file << "Number of trials: " << num_trials << "\n";
    param_file << "Magnetic field strength (h): " << h << "\n";
    param_file << "Field direction: [" << field_dir[0] << ", " << field_dir[1] << ", " << field_dir[2] << "]\n";
    param_file << "Applied field: [" << field[0] << ", " << field[1] << ", " << field[2] << "]\n";
    param_file << "J1xy: " << J1xy << "\n";
    param_file << "J1z: " << J1z << "\n";
    param_file << "D: " << D << "\n";
    param_file << "E: " << E << "\n";
    param_file << "F: " << F << "\n";
    param_file << "G: " << G << "\n";
    param_file << "J3xy: " << J3xy << "\n";
    param_file << "J3z: " << J3z << "\n";
    param_file << "Lattice size: 24x24x1\n";
    param_file << "Temperature for SA: 10K to 0.001K\n";
    param_file << "SA steps: 100000\n";
    param_file << "MD steps: 100\n";
    param_file << "MD timestep: 0.01\n";
    param_file.close();
    double min_energy;
    int min_index = 0;

    for(size_t i=0; i<num_trials;++i){
        filesystem::create_directories(dir + "/" + std::to_string(i));
        lattice<3, 2, 36, 36, 1> MC(&atoms, 1, true);
        MC.adaptive_restarted_simulated_annealing(20, 1e-3, 1e6, 10, num_trials, num_trials, true);
        MC.write_to_file_spin(dir +"/"+std::to_string(i)+ "/spin_0.001T.txt", MC.spins);        
        // Additional sweeps for convergence
        // for (size_t k = 0; k < 1e7; ++k) {
        //     MC.deterministic_sweep();
        // }
        MC.write_to_file_pos(dir +"/"+std::to_string(i)+ "/pos.txt");
        // Save the final configuration
        MC.write_to_file_spin(dir +"/"+std::to_string(i)+ "/spin_zero.txt", MC.spins);
        // Calculate and save the energy density
        double energy_density = MC.energy_density(MC.spins);
        ofstream energy_file(dir +"/"+std::to_string(i)+ "/energy_density.txt");
        energy_file << "Energy Density: " << energy_density << "\n";
        energy_file.close();      
        if (i == 0) {
            min_energy = MC.energy_density(MC.spins);
            min_index = i;
        } else {
            if (MC.energy_density(MC.spins) < min_energy) {
                min_energy = MC.energy_density(MC.spins);
                min_index = i;
            }
        }
    }

    // Output the information about the best configuration to a file
    ofstream best_config_file(dir + "/best_configuration.txt");
    best_config_file << "Best Configuration Found:\n";
    best_config_file << "Trial Index: " << min_index << "\n";
    best_config_file << "Minimum Energy Density: " << min_energy << "\n";
    best_config_file.close();
}

void F_scan(size_t num_steps, double h_start, double h_end, double F_start, double F_end, array<double, 3> field_dir, string dir, 
                        double J1xy=-7.6, double J1z=-1.2, double D=0.1, double E=-0.1, double G=0,
                        double J3xy=2.5, double J3z = -0.85, const array<array<double, 9>, 3>* custom_twist = nullptr) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<double> h_values;
    if (num_steps > 0) {
        double step = (h_end - h_start) / num_steps;
        for (size_t i = 0; i <= num_steps; ++i) {
            h_values.push_back(h_start + i * step);
        }
    } else {
        h_values.push_back(h_start);
    }

    for (size_t i = rank; i < h_values.size(); i += size) {
        for (size_t j = 0; j < num_steps; ++j) {
            double F = F_start + j * (F_end - F_start) / num_steps;
            filesystem::create_directories(dir + "/F_" + to_string(F));
            string subdir = dir + "/F_" + to_string(F) + "/h_" + to_string(h_values[i]);
            // Each process runs the simulation for its assigned 'h' value, with one trial.
            std::cout << "Running simulation for h = " << h_values[i] << ", F = " << F << " on process " << rank << std::endl;
            sim_BCAO_honeycomb(3, h_values[i], field_dir, subdir, J1xy, J1z, D, E, F, G, J3xy, J3z, custom_twist, true);
        }
    }
}
void J1xy_mag_field_phase_diagram(size_t J1_steps, double J1_start, double J1_end,
                        size_t num_steps, double h_start, double h_end, array<double, 3> field_dir, string dir, 
                        double J1z=-1.2, double D=0.1, double E=-0.1, double F=0, double G=0,
                        double J3xy=2.5, double J3z = -0.85, const array<array<double, 9>, 3>* custom_twist = nullptr, bool tbc = false, size_t num_trials=5) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<double> J1xy_values;
    if (J1_steps > 0) {
        double step = (J1_end - J1_start) / J1_steps;
        for (size_t i = 0; i <= J1_steps; ++i) {
            J1xy_values.push_back(J1_start + i * step);
        }
    } else {
        J1xy_values.push_back(J1_start);
    }
    vector<double> h_values;
    if (num_steps > 0) {
        double step = (h_end - h_start) / num_steps;
        for (size_t i = 0; i <= num_steps; ++i) {
            h_values.push_back(h_start + i * step);
        }
    } else {
        h_values.push_back(h_start);
    }
    size_t total_tasks = J1xy_values.size() * h_values.size() * num_trials;
    
    // Custom hash function for tuple<size_t, size_t>
    struct TupleHash {
        size_t operator()(const tuple<size_t, size_t>& t) const {
            auto h1 = std::hash<size_t>{}(get<0>(t));
            auto h2 = std::hash<size_t>{}(get<1>(t));
            return h1 ^ (h2 << 1);
        }
    };
    
    unordered_map<tuple<size_t, size_t>, vector<size_t>, TupleHash> assignments;
    
    for (size_t task = rank; task < total_tasks; task += static_cast<size_t>(size)) {
        size_t J1_idx = task / (h_values.size() * num_trials);
        size_t h_idx = (task / num_trials) % h_values.size();
        size_t trial_idx = task % num_trials;
        assignments[make_tuple(J1_idx, h_idx)].push_back(trial_idx);
    }
    for (auto &entry : assignments) {
        size_t J1_idx = get<0>(entry.first);
        size_t h_idx = get<1>(entry.first);
        auto &trial_list = entry.second;
        std::sort(trial_list.begin(), trial_list.end());
        double h = h_values[h_idx];
        double J1xy = J1xy_values[J1_idx];
        string subdir = dir + "/J1_" + to_string(J1xy) + "/h_" + to_string(h);

        for (size_t trial_id : trial_list) {
            std::cout << "Running simulation for h = " << h << ", trial = " << trial_id
                      << " on process " << rank << std::endl;
        }

        vector<pair<size_t, double>> trial_outcomes;
        sim_BCAO_honeycomb(num_trials, h, field_dir, subdir, J1xy, J1z, D, E, F, G,
                           J3xy, J3z, custom_twist, true, tbc, &trial_list, &trial_outcomes);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void magnetic_field_scan(size_t num_steps, double h_start, double h_end, array<double, 3> field_dir, string dir, 
                        double J1xy=-7.6, double J1z=-1.2, double D=0.1, double E=-0.1, double F=0, double G=0,
                        double J3xy=2.5, double J3z = -0.85, const array<array<double, 9>, 3>* custom_twist = nullptr, bool tbc = false, size_t num_trials=5) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    num_trials = 5; // Fixed number of trials for field scan
    vector<double> h_values;
    if (num_steps > 0) {
        double step = (h_end - h_start) / num_steps;
        for (size_t i = 0; i <= num_steps; ++i) {
            h_values.push_back(h_start + i * step);
        }
    } else {
        h_values.push_back(h_start);
    }

    size_t total_tasks = h_values.size() * num_trials;
    unordered_map<size_t, vector<size_t>> assignments;
    for (size_t task = rank; task < total_tasks; task += static_cast<size_t>(size)) {
        size_t h_idx = task / num_trials;
        size_t trial_idx = task % num_trials;
        assignments[h_idx].push_back(trial_idx);
    }

    vector<double> local_best_energy(h_values.size(), std::numeric_limits<double>::max());
    vector<int> local_best_trial(h_values.size(), -1);

    for (auto &entry : assignments) {
        size_t h_idx = entry.first;
        auto &trial_list = entry.second;
        std::sort(trial_list.begin(), trial_list.end());
        double h = h_values[h_idx];
        string subdir = dir + "/h_" + to_string(h);

        for (size_t trial_id : trial_list) {
            std::cout << "Running simulation for h = " << h << ", trial = " << trial_id
                      << " on process " << rank << std::endl;
        }

        vector<pair<size_t, double>> trial_outcomes;
        sim_BCAO_honeycomb(num_trials, h, field_dir, subdir, J1xy, J1z, D, E, F, G,
                           J3xy, J3z, custom_twist, true, tbc, &trial_list, &trial_outcomes);

        for (const auto &outcome : trial_outcomes) {
            if (outcome.second < local_best_energy[h_idx]) {
                local_best_energy[h_idx] = outcome.second;
                local_best_trial[h_idx] = static_cast<int>(outcome.first);
            }
        }
    }

    struct MinResult {
        double energy;
        int trial;
    };

    vector<MinResult> local_minima(h_values.size());
    for (size_t idx = 0; idx < h_values.size(); ++idx) {
        local_minima[idx].energy = local_best_energy[idx];
        local_minima[idx].trial = local_best_trial[idx];
    }

    vector<MinResult> global_minima;
    if (rank == 0) {
        global_minima.resize(h_values.size());
    }

    MPI_Reduce(local_minima.data(), rank == 0 ? global_minima.data() : nullptr,
               static_cast<int>(h_values.size()), MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (size_t idx = 0; idx < h_values.size(); ++idx) {
            double h = h_values[idx];
            string subdir = dir + "/h_" + to_string(h);
            filesystem::create_directories(subdir);

            ofstream best_config_file(subdir + "/best_configuration.txt");
            best_config_file << "Best Configuration Found:\n";
            if (global_minima[idx].trial >= 0) {
                best_config_file << "Trial Index: " << global_minima[idx].trial << "\n";
                best_config_file << "Minimum Energy Density: " << global_minima[idx].energy << "\n";
            } else {
                best_config_file << "Trial Index: N/A\n";
                best_config_file << "Minimum Energy Density: N/A\n";
            }
            best_config_file.close();
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// Generic 2D phase diagram scanner for any two parameters
void generic_phase_diagram(const string& param1_name, double param1_start, double param1_end, int param1_steps,
                          const string& param2_name, double param2_start, double param2_end, int param2_steps,
                          size_t num_trials, array<double, 3> field_dir, string dir,
                          double h, double J1xy, double J1z, double D, double E, double F, double G,
                          double J3xy, double J3z, const array<array<double, 9>, 3>* custom_twist = nullptr, bool tbc = false) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Generate parameter grids
    vector<double> param1_values, param2_values;
    if (param1_steps > 0) {
        double step = (param1_end - param1_start) / param1_steps;
        for (int i = 0; i <= param1_steps; ++i) {
            param1_values.push_back(param1_start + i * step);
        }
    } else {
        param1_values.push_back(param1_start);
    }
    
    if (param2_steps > 0) {
        double step = (param2_end - param2_start) / param2_steps;
        for (int i = 0; i <= param2_steps; ++i) {
            param2_values.push_back(param2_start + i * step);
        }
    } else {
        param2_values.push_back(param2_start);
    }
    
    size_t total_tasks = param1_values.size() * param2_values.size() * num_trials;
    
    // Custom hash function for tuple<size_t, size_t>
    struct TupleHash {
        size_t operator()(const tuple<size_t, size_t>& t) const {
            auto h1 = std::hash<size_t>{}(get<0>(t));
            auto h2 = std::hash<size_t>{}(get<1>(t));
            return h1 ^ (h2 << 1);
        }
    };
    
    unordered_map<tuple<size_t, size_t>, vector<size_t>, TupleHash> assignments;
    
    // Distribute tasks across MPI processes
    for (size_t task = rank; task < total_tasks; task += static_cast<size_t>(size)) {
        size_t p1_idx = task / (param2_values.size() * num_trials);
        size_t p2_idx = (task / num_trials) % param2_values.size();
        size_t trial_idx = task % num_trials;
        assignments[make_tuple(p1_idx, p2_idx)].push_back(trial_idx);
    }
    
    // Lambda to get pointer to parameter by name
    auto get_param_ptr = [&](const string& name, double& h_ref, double& J1xy_ref, double& J1z_ref,
                             double& D_ref, double& E_ref, double& F_ref, double& G_ref,
                             double& J3xy_ref, double& J3z_ref) -> double* {
        if (name == "h") return &h_ref;
        else if (name == "J1xy") return &J1xy_ref;
        else if (name == "J1z") return &J1z_ref;
        else if (name == "D") return &D_ref;
        else if (name == "E") return &E_ref;
        else if (name == "F") return &F_ref;
        else if (name == "G") return &G_ref;
        else if (name == "J3xy") return &J3xy_ref;
        else if (name == "J3z") return &J3z_ref;
        return nullptr;
    };
    
    // Run simulations for assigned parameter combinations
    for (auto &entry : assignments) {
        size_t p1_idx = get<0>(entry.first);
        size_t p2_idx = get<1>(entry.first);
        auto &trial_list = entry.second;
        std::sort(trial_list.begin(), trial_list.end());
        
        double p1_val = param1_values[p1_idx];
        double p2_val = param2_values[p2_idx];
        
        // Create local copies of all parameters
        double h_local = h, J1xy_local = J1xy, J1z_local = J1z;
        double D_local = D, E_local = E, F_local = F, G_local = G;
        double J3xy_local = J3xy, J3z_local = J3z;
        
        // Update the relevant parameters
        double* param1_ptr = get_param_ptr(param1_name, h_local, J1xy_local, J1z_local,
                                           D_local, E_local, F_local, G_local, J3xy_local, J3z_local);
        double* param2_ptr = get_param_ptr(param2_name, h_local, J1xy_local, J1z_local,
                                           D_local, E_local, F_local, G_local, J3xy_local, J3z_local);
        
        if (param1_ptr) *param1_ptr = p1_val;
        if (param2_ptr) *param2_ptr = p2_val;
        
        // Create subdirectory with parameter names
        string subdir = dir + "/" + param1_name + "_" + to_string(p1_val) + "_" + param2_name + "_" + to_string(p2_val);
        
        if (rank == 0 || trial_list.size() > 0) {
            std::cout << "Running simulation for " << param1_name << " = " << p1_val 
                      << ", " << param2_name << " = " << p2_val
                      << " with " << trial_list.size() << " trials on process " << rank << std::endl;
        }
        
        vector<pair<size_t, double>> trial_outcomes;
        sim_BCAO_honeycomb(num_trials, h_local, field_dir, subdir, 
                           J1xy_local, J1z_local, D_local, E_local, F_local, G_local,
                           J3xy_local, J3z_local, custom_twist, true, tbc, &trial_list, &trial_outcomes);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Save phase diagram metadata
        ofstream meta_file(dir + "/phase_diagram_info.txt");
        meta_file << "Generic Phase Diagram Scan\n";
        meta_file << "==========================\n";
        meta_file << "Parameter 1: " << param1_name << "\n";
        meta_file << "  Range: " << param1_start << " to " << param1_end << "\n";
        meta_file << "  Steps: " << param1_steps << "\n";
        meta_file << "\nParameter 2: " << param2_name << "\n";
        meta_file << "  Range: " << param2_start << " to " << param2_end << "\n";
        meta_file << "  Steps: " << param2_steps << "\n";
        meta_file << "\nTrials per point: " << num_trials << "\n";
        meta_file << "Total points: " << param1_values.size() * param2_values.size() << "\n";
        meta_file.close();
    }
}

// N-dimensional phase diagram scanner for arbitrary number of parameters
void ndim_phase_diagram(const vector<string>& param_names, 
                        const vector<double>& param_starts,
                        const vector<double>& param_ends, 
                        const vector<int>& param_steps,
                        size_t num_trials, array<double, 3> field_dir, string dir,
                        double h, double J1xy, double J1z, double D, double E, double F, double G,
                        double J3xy, double J3z, const array<array<double, 9>, 3>* custom_twist = nullptr, bool tbc = false) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    size_t ndim = param_names.size();
    
    if (ndim == 0) {
        if (rank == 0) {
            cerr << "Error: No parameters specified for N-dimensional scan\n";
        }
        return;
    }
    
    // Generate grids for each parameter
    vector<vector<double>> param_grids(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        if (param_steps[i] > 0) {
            double step = (param_ends[i] - param_starts[i]) / param_steps[i];
            for (int j = 0; j <= param_steps[i]; ++j) {
                param_grids[i].push_back(param_starts[i] + j * step);
            }
        } else {
            param_grids[i].push_back(param_starts[i]);
        }
    }
    
    // Calculate total number of grid points
    size_t total_points = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_points *= param_grids[i].size();
    }
    
    size_t total_tasks = total_points * num_trials;
    
    if (rank == 0) {
        cout << "N-dimensional phase diagram scan:\n";
        cout << "  Dimensions: " << ndim << "\n";
        cout << "  Total grid points: " << total_points << "\n";
        cout << "  Trials per point: " << num_trials << "\n";
        cout << "  Total simulations: " << total_tasks << "\n";
        for (size_t i = 0; i < ndim; ++i) {
            cout << "  " << param_names[i] << ": " << param_starts[i] << " to " 
                 << param_ends[i] << " (" << param_grids[i].size() << " values)\n";
        }
    }
    
    // Lambda to convert flat index to multi-dimensional indices
    auto index_to_coords = [&](size_t flat_idx) -> vector<size_t> {
        vector<size_t> coords(ndim);
        for (int i = ndim - 1; i >= 0; --i) {
            coords[i] = flat_idx % param_grids[i].size();
            flat_idx /= param_grids[i].size();
        }
        return coords;
    };
    
    // Lambda to get pointer to parameter by name
    auto get_param_ptr = [&](const string& name, double& h_ref, double& J1xy_ref, double& J1z_ref,
                             double& D_ref, double& E_ref, double& F_ref, double& G_ref,
                             double& J3xy_ref, double& J3z_ref) -> double* {
        if (name == "h") return &h_ref;
        else if (name == "J1xy") return &J1xy_ref;
        else if (name == "J1z") return &J1z_ref;
        else if (name == "D") return &D_ref;
        else if (name == "E") return &E_ref;
        else if (name == "F") return &F_ref;
        else if (name == "G") return &G_ref;
        else if (name == "J3xy") return &J3xy_ref;
        else if (name == "J3z") return &J3z_ref;
        return nullptr;
    };
    
    // Hash function for vector<size_t>
    struct VectorHash {
        size_t operator()(const vector<size_t>& v) const {
            size_t seed = v.size();
            for (auto& i : v) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
    
    // Group tasks by parameter combination
    unordered_map<vector<size_t>, vector<size_t>, VectorHash> assignments;
    
    for (size_t task = rank; task < total_tasks; task += static_cast<size_t>(size)) {
        size_t point_idx = task / num_trials;
        size_t trial_idx = task % num_trials;
        vector<size_t> coords = index_to_coords(point_idx);
        assignments[coords].push_back(trial_idx);
    }
    
    // Run simulations for assigned parameter combinations
    for (auto &entry : assignments) {
        const vector<size_t>& coords = entry.first;
        auto &trial_list = entry.second;
        std::sort(trial_list.begin(), trial_list.end());
        
        // Create local copies of all parameters
        double h_local = h, J1xy_local = J1xy, J1z_local = J1z;
        double D_local = D, E_local = E, F_local = F, G_local = G;
        double J3xy_local = J3xy, J3z_local = J3z;
        
        // Build directory name and update parameters
        stringstream subdir_ss;
        subdir_ss << dir;
        
        for (size_t i = 0; i < ndim; ++i) {
            double param_val = param_grids[i][coords[i]];
            subdir_ss << "/" << param_names[i] << "_" << param_val;
            
            double* param_ptr = get_param_ptr(param_names[i], h_local, J1xy_local, J1z_local,
                                             D_local, E_local, F_local, G_local, J3xy_local, J3z_local);
            if (param_ptr) *param_ptr = param_val;
        }
        
        string subdir = subdir_ss.str();
        
        if (rank == 0 || trial_list.size() > 0) {
            cout << "Process " << rank << " running: ";
            for (size_t i = 0; i < ndim; ++i) {
                cout << param_names[i] << "=" << param_grids[i][coords[i]];
                if (i < ndim - 1) cout << ", ";
            }
            cout << " (" << trial_list.size() << " trials)\n";
        }
        
        vector<pair<size_t, double>> trial_outcomes;
        sim_BCAO_honeycomb(num_trials, h_local, field_dir, subdir, 
                           J1xy_local, J1z_local, D_local, E_local, F_local, G_local,
                           J3xy_local, J3z_local, custom_twist, true, tbc, &trial_list, &trial_outcomes);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Save phase diagram metadata
        ofstream meta_file(dir + "/phase_diagram_info.txt");
        meta_file << "N-Dimensional Phase Diagram Scan\n";
        meta_file << "=================================\n";
        meta_file << "Dimensions: " << ndim << "\n";
        meta_file << "Total grid points: " << total_points << "\n";
        meta_file << "Trials per point: " << num_trials << "\n";
        meta_file << "Total simulations: " << total_tasks << "\n\n";
        
        for (size_t i = 0; i < ndim; ++i) {
            meta_file << "Parameter " << (i+1) << ": " << param_names[i] << "\n";
            meta_file << "  Range: " << param_starts[i] << " to " << param_ends[i] << "\n";
            meta_file << "  Steps: " << param_steps[i] << "\n";
            meta_file << "  Values: " << param_grids[i].size() << "\n\n";
        }
        meta_file.close();
    }
}

int main(int argc, char** argv) {
    double mu_B = 5.7883818012e-2;
    
    // Default parameter file name
    string param_file = "bcao_parameters.txt";
    bool use_restart = false;
    bool do_field_scan = false;
    bool do_f_scan = false;
    bool do_J1xy_scan = false;
    bool do_phase_diagram = false;
    bool do_ndim_scan = false;
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [options] [parameter_file]\n\n";
            cout << "Options:\n";
            cout << "  --help, -h          Show this help message\n";
            cout << "  --create-params     Create default parameter file (bcao_parameters.txt)\n";
            cout << "  --restart           Use the adaptive restarted simulated annealing method\n";
            cout << "  --field-scan        Perform a magnetic field scan instead of a single run\n";
            cout << "  --J1xy-scan         Perform a J1xy vs Magnetic field phase diagram scan\n";
            cout << "  --phase-diagram     Perform a generic 2D phase diagram scan\n";
            cout << "                      Uses param1_name, param1_start, param1_end, param1_steps\n";
            cout << "                      and param2_name, param2_start, param2_end, param2_steps\n";
            cout << "                      from parameter file\n";
            cout << "  --ndim-scan         Perform an N-dimensional phase diagram scan\n";
            cout << "                      Uses scan_params, scan_starts, scan_ends, scan_steps\n";
            cout << "                      from parameter file (comma-separated lists)\n\n";
            cout << "Arguments:\n";
            cout << "  parameter_file      Path to parameter file (default: bcao_parameters.txt)\n\n";
            cout << "Available parameters for phase diagram:\n";
            cout << "  h, J1xy, J1z, D, E, F, G, J3xy, J3z\n\n";
            cout << "Examples:\n";
            cout << "  1D scan: scan_params = h\n";
            cout << "  2D scan: scan_params = h,D\n";
            cout << "  3D scan: scan_params = h,D,J1xy\n";
            cout << "  4D scan: scan_params = h,D,E,J3xy\n\n";
            cout << "If parameter file doesn't exist, default parameters will be used.\n";
            return 0;
        } else if (arg == "--create-params") {
            create_default_parameter_file("bcao_parameters.txt");
            cout << "Created default parameter file: bcao_parameters.txt\n";
            cout << "Edit this file and run the simulation again.\n";
            return 0;
        } else if (arg == "--restart") {
            use_restart = true;
        } else if (arg == "--field-scan") {
            do_field_scan = true;
        } else if (arg == "--f-scan"){
            do_f_scan = true;
        } else if (arg == "--J1xy-scan"){
            do_J1xy_scan = true;
        } else if (arg == "--phase-diagram"){
            do_phase_diagram = true;
        } else if (arg == "--ndim-scan"){
            do_ndim_scan = true;
        }
        else {
            // Assume the last non-flag argument is the parameter file
            param_file = arg;
        }
    }
    
    // Read parameters from file
    SimulationParams params = read_parameters(param_file);
    
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(&argc, &argv);
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        cout << "Running simulation with parameters from: " << param_file << "\n";
        if (do_ndim_scan) {
            cout << "Mode: N-Dimensional Phase Diagram\n";
            cout << "Scanning " << params.scan_params.size() << " parameters:\n";
            for (size_t i = 0; i < params.scan_params.size(); ++i) {
                cout << "  " << params.scan_params[i] << ": " << params.scan_starts[i] 
                     << " to " << params.scan_ends[i] << " (" << params.scan_steps[i] << " steps)\n";
            }
        } else if (do_phase_diagram) {
            cout << "Mode: Generic Phase Diagram (2D)\n";
            cout << "Parameter 1: " << params.param1_name << " from " << params.param1_start 
                 << " to " << params.param1_end << " (" << params.param1_steps << " steps)\n";
            cout << "Parameter 2: " << params.param2_name << " from " << params.param2_start 
                 << " to " << params.param2_end << " (" << params.param2_steps << " steps)\n";
        } else if (do_field_scan) {
            cout << "Mode: Magnetic Field Scan\n";
        } else if (do_J1xy_scan) {
            cout << "Mode: J1xy vs Magnetic Field Phase Diagram\n";
        } else {
            cout << "Mode: Single Simulation\n";
            cout << "Restarted Annealing: " << (use_restart ? "Yes" : "No") << "\n";
            cout << "Trials: " << params.num_trials << ", Field: " << params.h << " mu_B\n";
        }
        cout << "Field direction: [" << params.field_dir[0] << "," << params.field_dir[1] << "," << params.field_dir[2] << "]\n";
        cout << "Output directory: " << params.dir << "\n";
        cout << "Running on " << size << " MPI processes.\n";
    }

    const array<array<double, 9>, 3>* custom_twist_ptr = params.has_twist_matrix ? &params.twist_matrix : nullptr;

    if (do_ndim_scan) {
        // Validate N-dimensional scan parameters
        vector<string> valid_params = {"h", "J1xy", "J1z", "D", "E", "F", "G", "J3xy", "J3z"};
        
        if (params.scan_params.empty()) {
            if (rank == 0) {
                cerr << "Error: No parameters specified for N-dimensional scan.\n";
                cerr << "Set scan_params in parameter file (e.g., scan_params = h,D,E)\n";
            }
            MPI_Finalize();
            return 1;
        }
        
        if (params.scan_params.size() != params.scan_starts.size() ||
            params.scan_params.size() != params.scan_ends.size() ||
            params.scan_params.size() != params.scan_steps.size()) {
            if (rank == 0) {
                cerr << "Error: Mismatched array sizes for N-dimensional scan.\n";
                cerr << "scan_params size: " << params.scan_params.size() << "\n";
                cerr << "scan_starts size: " << params.scan_starts.size() << "\n";
                cerr << "scan_ends size: " << params.scan_ends.size() << "\n";
                cerr << "scan_steps size: " << params.scan_steps.size() << "\n";
            }
            MPI_Finalize();
            return 1;
        }
        
        for (const auto& param : params.scan_params) {
            if (find(valid_params.begin(), valid_params.end(), param) == valid_params.end()) {
                if (rank == 0) {
                    cerr << "Error: Invalid parameter name '" << param << "' in scan_params\n";
                    cerr << "Valid parameters: h, J1xy, J1z, D, E, F, G, J3xy, J3z\n";
                }
                MPI_Finalize();
                return 1;
            }
        }
        
        ndim_phase_diagram(params.scan_params, params.scan_starts, params.scan_ends, params.scan_steps,
                          params.num_trials, params.field_dir, params.dir,
                          params.h, params.J1xy, params.J1z, params.D, params.E, params.F, params.G,
                          params.J3xy, params.J3z, custom_twist_ptr, params.tbc);
    } else if (do_phase_diagram) {
        // Validate parameter names
        vector<string> valid_params = {"h", "J1xy", "J1z", "D", "E", "F", "G", "J3xy", "J3z"};
        bool param1_valid = find(valid_params.begin(), valid_params.end(), params.param1_name) != valid_params.end();
        bool param2_valid = find(valid_params.begin(), valid_params.end(), params.param2_name) != valid_params.end();
        
        if (!param1_valid || !param2_valid || params.param1_name.empty() || params.param2_name.empty()) {
            if (rank == 0) {
                cerr << "Error: Invalid parameter names for phase diagram.\n";
                cerr << "param1_name = '" << params.param1_name << "', param2_name = '" << params.param2_name << "'\n";
                cerr << "Valid parameters: h, J1xy, J1z, D, E, F, G, J3xy, J3z\n";
            }
            MPI_Finalize();
            return 1;
        }
        
        generic_phase_diagram(params.param1_name, params.param1_start, params.param1_end, params.param1_steps,
                             params.param2_name, params.param2_start, params.param2_end, params.param2_steps,
                             params.num_trials, params.field_dir, params.dir,
                             params.h, params.J1xy, params.J1z, params.D, params.E, params.F, params.G,
                             params.J3xy, params.J3z, custom_twist_ptr, params.tbc);
    } else if (do_field_scan) {
        magnetic_field_scan(params.num_steps, params.h_start, params.h_end, params.field_dir, params.dir,
                            params.J1xy, params.J1z, params.D, params.E, params.F, params.G,
                            params.J3xy, params.J3z, custom_twist_ptr, params.tbc);
    } else if (do_f_scan){
        F_scan(params.num_steps, params.h_start, params.h_end, 0, -1, params.field_dir, params.dir,
                            params.J1xy, params.J1z, params.D, params.E, params.G,
                            params.J3xy, params.J3z, custom_twist_ptr);
    } else if (do_J1xy_scan){
        J1xy_mag_field_phase_diagram(5, params.J1xy,  params.J1xy-1,
                        params.num_steps, params.h_start, params.h_end, params.field_dir, params.dir,
                        params.J1z, params.D, params.E, params.F, params.G,
                        params.J3xy, params.J3z, custom_twist_ptr, params.tbc);
    }
    else {
        if (use_restart) {
            sim_BCAO_honeycomb_restarted(params.num_trials, params.h, params.field_dir, params.dir, 
                                         params.J1xy, params.J1z, params.D, params.E, params.F, params.G, 
                                         params.J3xy, params.J3z, custom_twist_ptr);
        } else {
            sim_BCAO_honeycomb(params.num_trials, params.h, params.field_dir, params.dir, 
                               params.J1xy, params.J1z, params.D, params.E, params.F, params.G, 
                               params.J3xy, params.J3z, custom_twist_ptr);
        }
    }
    
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}