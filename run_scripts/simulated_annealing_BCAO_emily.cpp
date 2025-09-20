#include "experiments.h"
#include <math.h>
#include <mpi.h>
#include <filesystem>
#include <fstream>
#include <sstream>

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
    file << "# Twist matrices (each 3x3 flattened, comma-separated)\n";
    file << "twist_matrix_0 = 1,0,0, 0,1,0, 0,0,1\n";
    file << "twist_matrix_1 = 1,0,0, 0,1,0, 0,0,1\n";
    file << "twist_matrix_2 = 1,0,0, 0,1,0, 0,0,1\n";
    file.close();
}


void sim_BCAO_honeycomb(size_t num_trials, double h, array<double, 3> field_dir, string dir, double J1xy=-7.6, double J1z=-1.2, double D=0.1, double E=-0.1, double F=0, double G=0, double J3xy=2.5, double J3z = -0.85, const array<array<double, 9>, 3>* custom_twist = nullptr, bool field_scan = false, bool tbc = false) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
    const double mu_B = 0.05788; // meV/T
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
    double k_B = 0.08620689655;

    // Save simulation parameters (only rank 0)
    if (rank == 0 || field_scan) {
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

    // Each process handles a subset of trials
    for(size_t i = 0; i < num_trials; ++i){
        filesystem::create_directories(dir + "/" + std::to_string(i));
        lattice<3, 2, 24, 24, 1> MC(&atoms, 1, true);
        MC.simulated_annealing(10, 0.5, 2e4, 20, tbc, false, 0.9, dir +"/"+std::to_string(i), true);
        double energy_density = MC.energy_density(MC.spins);
        ofstream energy_file(dir +"/"+std::to_string(i)+ "/energy_density.txt");
        energy_file << "Energy Density: " << energy_density << "\n";
        energy_file.close();      
        ofstream twist_file(dir +"/"+std::to_string(i)+ "/twist_matrix.txt");
        twist_file << "Twist Matrix:\n";
        for (const auto& row : MC.twist_matrices) {
            for (const auto& val : row) {
                twist_file << val << " ";
            }
            twist_file << "\n";
        }
        twist_file.close();

        const auto energy_langscape = MC.local_energy_densities(MC.spins);
        ofstream landscape_file(dir +"/"+std::to_string(i)+ "/energy_landscape.txt");
        landscape_file << "Energy Landscape:\n";
        for (size_t j = 0; j < energy_langscape.size(); ++j) {
            landscape_file << j << " " << energy_langscape[j] << "\n";
        }
        landscape_file.close();

        // Update local minimum
        if (energy_density < local_min_energy) {
            local_min_energy = energy_density;
            local_min_index = i;
        }
    }
    // Only perform MPI collective operations if not in field_scan mode
    // or if there are multiple trials to aggregate
    if (num_trials > 1 && !field_scan) {
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
    } else if (local_min_index >= 0) {
        // For single trial or field_scan mode, just output the local result if we have one
        if (rank == 0 || field_scan) {
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


void magnetic_field_scan(size_t num_steps, double h_start, double h_end, array<double, 3> field_dir, string dir, 
                        double J1xy=-7.6, double J1z=-1.2, double D=0.1, double E=-0.1, double F=0, double G=0,
                        double J3xy=2.5, double J3z = -0.85, const array<array<double, 9>, 3>* custom_twist = nullptr, bool tbc = false) {
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
        double h = h_values[i];
        string subdir = dir + "/h_" + to_string(h);
        // Each process runs the simulation for its assigned 'h' value, with one trial.
        std::cout << "Running simulation for h = " << h << " on process " << rank << std::endl;
        sim_BCAO_honeycomb(20, h, field_dir, subdir, J1xy, J1z, D, E, F, G, J3xy, J3z, custom_twist, true, tbc);
    }
    
    // Synchronize all processes after field scan is complete
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    double mu_B = 5.7883818012e-2;
    
    // Default parameter file name
    string param_file = "bcao_parameters.txt";
    bool use_restart = false;
    bool do_field_scan = false;
    bool do_f_scan = false;
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [options] [parameter_file]\n\n";
            cout << "Options:\n";
            cout << "  --help, -h          Show this help message\n";
            cout << "  --create-params     Create default parameter file (bcao_parameters.txt)\n";
            cout << "  --restart           Use the adaptive restarted simulated annealing method\n";
            cout << "  --field-scan        Perform a magnetic field scan instead of a single run\n\n";
            cout << "Arguments:\n";
            cout << "  parameter_file      Path to parameter file (default: bcao_parameters.txt)\n\n";
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
        }else {
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
        if (do_field_scan) {
            cout << "Mode: Magnetic Field Scan\n";
            // Assuming h_start, h_end, num_steps are part of SimulationParams and read_parameters
            // If not, they need to be added. For now, we'll use hardcoded or default values.
            // Let's assume they are in params.
            // cout << "Field scan from " << params.h_start << " to " << params.h_end << " in " << params.num_steps << " steps.\n";
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

    if (do_field_scan) {
        magnetic_field_scan(params.num_steps, params.h_start, params.h_end, params.field_dir, params.dir,
                            params.J1xy, params.J1z, params.D, params.E, params.F, params.G,
                            params.J3xy, params.J3z, custom_twist_ptr, params.tbc);
    } else if (do_f_scan){
        F_scan(params.num_steps, params.h_start, params.h_end, 0, -1, params.field_dir, params.dir,
                            params.J1xy, params.J1z, params.D, params.E, params.G,
                            params.J3xy, params.J3z, custom_twist_ptr);
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