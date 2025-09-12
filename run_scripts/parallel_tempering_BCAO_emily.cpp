#include "experiments.h"
#include <math.h>
#include <mpi.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <array>
#include <iomanip>

using namespace std;

namespace timing_helpers {
    inline void log_timing(const string& filepath, const string& label, double seconds, int rank) {
        if (rank != 0) return;
        // Ensure parent directory exists
        try {
            filesystem::path p(filepath);
            if (p.has_parent_path()) filesystem::create_directories(p.parent_path());
        } catch (...) {
            // best-effort; ignore directory creation errors
        }
        ofstream ofs(filepath, ios::app);
        if (ofs) {
            ofs.setf(ios::fixed);
            ofs << label << ": " << setprecision(6) << seconds << " s" << '\n';
        }
        // Also echo to stdout for quick visibility
        cout.setf(ios::fixed);
        cout << label << ": " << setprecision(6) << seconds << " s" << '\n';
    }
}

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
    size_t thermalization_sweeps = 5000000;
    size_t measurement_sweeps = 1000000;
    size_t overrelaxation_rate = 4000; // Overrelaxation rate for the simulation
    size_t swap_interval = 50;
    size_t probe_rate = 2000;

    size_t num_trials = 5; // Number of trials for the simulation

    // Field scan parameters (set num_steps > 0 to enable field scan mode)
    double h_start = 0.0;
    double h_end = 0.0;
    size_t num_steps = 0;

    array<array<double, 9>, 3> twist_matrix = {{
        {1, 0, 0, 
         0, 1, 0, 
         0, 0, 1},
        {1, 0, 0, 
         0, 1, 0, 
         0, 0, 1},
        {1, 0, 0, 
         0, 1, 0, 
         0, 0, 1}
    }};

};

// Function to read parameters from a file
SimulationParams read_parameters(const string& filename) {
    SimulationParams params;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cout << "Warning: Could not open parameter file '" << filename << "'. Using default parameters.\n";
        return params;
    }
    
    auto sanitize_and_split = [](string s) -> vector<string> {
        // Replace semicolons and whitespace with commas, remove brackets/parentheses
        for (char& c : s) {
            if (c == ';' || c == '\t' || c == ' ') c = ',';
            if (c == '[' || c == ']' || c == '(' || c == ')') c = ' ';
        }
        // collapse repeats of commas
        string tmp;
        bool last_comma = false;
        for (char c : s) {
            if (c == ',') {
                if (!last_comma) tmp.push_back(c);
                last_comma = true;
            } else {
                tmp.push_back(c);
                last_comma = false;
            }
        }
        // split by comma
        vector<string> out;
        string token;
        stringstream ss(tmp);
        while (getline(ss, token, ',')) {
            // trim
            size_t start = token.find_first_not_of(" \t\n\r");
            size_t end = token.find_last_not_of(" \t\n\r");
            if (start != string::npos && end != string::npos) {
                out.emplace_back(token.substr(start, end - start + 1));
            }
        }
        return out;
    };

    auto parse_matrix9 = [&](const string& value, array<double, 9>& out) -> bool {
        vector<string> toks = sanitize_and_split(value);
        if (toks.size() < 9) return false;
        try {
            for (size_t i = 0; i < 9; ++i) out[i] = stod(toks[i]);
        } catch (...) { return false; }
        return true;
    };

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
            else if (key == "h_start") params.h_start = stod(value);
            else if (key == "h_end") params.h_end = stod(value);
            else if (key == "num_steps") params.num_steps = stoul(value);
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
            // Twist matrix (accept multiple formats)
            else if (key == "twist_matrix" || key == "twist") {
                // Expect 27 numbers; allow commas/semicolons/spaces/brackets
                vector<string> toks = sanitize_and_split(value);
                if (toks.size() >= 27) {
                    try {
                        size_t idx = 0;
                        for (int b = 0; b < 3; ++b) {
                            for (int i = 0; i < 9; ++i) {
                                params.twist_matrix[b][i] = stod(toks[idx++]);
                            }
                        }
                    } catch (...) {
                        cout << "Warning: Failed to parse twist_matrix; keeping defaults.\n";
                    }
                } else {
                    cout << "Warning: twist_matrix expects 27 numbers; got " << toks.size() << ". Keeping defaults.\n";
                }
            }
            else if (key == "twist_matrix_0" || key == "twist_x") {
                array<double,9> tmp{};
                if (parse_matrix9(value, tmp)) params.twist_matrix[0] = tmp;
                else cout << "Warning: Failed to parse twist_matrix_0; keeping default.\n";
            }
            else if (key == "twist_matrix_1" || key == "twist_y") {
                array<double,9> tmp{};
                if (parse_matrix9(value, tmp)) params.twist_matrix[1] = tmp;
                else cout << "Warning: Failed to parse twist_matrix_1; keeping default.\n";
            }
            else if (key == "twist_matrix_2" || key == "twist_z") {
                array<double,9> tmp{};
                if (parse_matrix9(value, tmp)) params.twist_matrix[2] = tmp;
                else cout << "Warning: Failed to parse twist_matrix_2; keeping default.\n";
            }
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

    file << "# Field scan (set num_steps>0 to enable field-scan mode)\n";
    file << "h_start = 0.0\n";
    file << "h_end = 0.0\n";
    file << "num_steps = 0\n\n";
    
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

    file << "\n# Twist boundary matrices (three 3x3 blocks, row-major).\n";
    file << "# Option A: one line with 27 numbers (x-block, then y-block, then z-block):\n";
    file << "# twist_matrix = 1,0,0, 0,1,0, 0,0,1,   1,0,0, 0,1,0, 0,0,1,   1,0,0, 0,1,0, 0,0,1\n";
    file << "# Option B: three lines with 9 numbers each (aliases: twist_x/y/z):\n";
    file << "twist_matrix_0 = 1,0,0, 0,1,0, 0,0,1\n";
    file << "twist_matrix_1 = 1,0,0, 0,1,0, 0,0,1\n";
    file << "twist_matrix_2 = 1,0,0, 0,1,0, 0,0,1\n";

    file.close();
}

// Main simulation function for Parallel Tempering
void PT_BCAO_honeycomb(const SimulationParams& params, bool boundary_update){
    filesystem::create_directories(params.dir);
    HoneyComb<3> atoms;
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const string timing_file = params.dir + "/timing.log";
    double t_total = MPI_Wtime();
    double t_step = MPI_Wtime();

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

    array<double, 3> field = {params.h*params.field_dir[0], params.h*params.field_dir[1], params.h*params.field_dir[2]};
    
    // Set interactions
    atoms.set_bilinear_interaction(J1x_, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J1y_, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    MPI_Barrier(MPI_COMM_WORLD);
    timing_helpers::log_timing(timing_file, "step_1_setup_model_and_fields", MPI_Wtime() - t_step, rank);

    // Reset step timer
    t_step = MPI_Wtime();

    // Temperature schedule
    vector<double> temps = logspace(log10(params.T_start), log10(params.T_end), size);


    
    // Lattice and simulation
    lattice<3, 2, 60, 60, 1> MC(&atoms, 1, true);
    MPI_Barrier(MPI_COMM_WORLD);
    timing_helpers::log_timing(timing_file, "step_2_setup_temps_and_lattice", MPI_Wtime() - t_step, rank);

    // Parallel tempering run
    t_step = MPI_Wtime();
    MC.parallel_tempering(temps, params.thermalization_sweeps, params.measurement_sweeps, params.overrelaxation_rate, params.swap_interval, params.probe_rate, params.dir, {0}, boundary_update);
    MPI_Barrier(MPI_COMM_WORLD);
    timing_helpers::log_timing(timing_file, "step_3_parallel_tempering", MPI_Wtime() - t_step, rank);
    if (rank == 0) {
        t_step = MPI_Wtime();
        cout << "Parallel Tempering simulation completed. Results saved in: " << params.dir << "\n";
        MC.write_to_file_spin(params.dir + "/spin.txt", MC.spins);
        // for (size_t i = 0; i < 1e7; ++i) {
        //     MC.deterministic_sweep();
        // }
        // MC.write_to_file_spin(params.dir + "/spin_zero.txt", MC.spins);
        ofstream param_file(params.dir + "/simulation_parameters.txt");
        param_file << "Simulation Parameters for BCAO Honeycomb MD\n";
        param_file << "==========================================\n";
        param_file << "Field: " << params.h << " mu_B, Direction: [" << params.field_dir[0] << "," << params.field_dir[1] << "," << params.field_dir[2] << "]\n";
        param_file << "Temperature Range: " << params.T_start << "K to " << params.T_end << "K\n";
        param_file << "Output directory: " << params.dir << "\n";
        param_file << "J1xy: " << params.J1xy << "\n";
        param_file << "J1z: " << params.J1z << "\n";
        param_file << "D: " << params.D << "\n";
        param_file << "E: " << params.E << "\n";
        param_file << "F: " << params.F << "\n";
        param_file << "G: " << params.G << "\n";
        param_file << "J3xy: " << params.J3xy << "\n";
        param_file << "J3z: " << params.J3z << "\n";
        param_file << "Lattice size: 100x100x1\n";
        param_file.close();
        cout << "Simulation parameters saved to: " << params.dir + "/simulation_parameters.txt" << "\n";
        
        const auto energy_landscape = MC.local_energy_densities(MC.spins);
        ofstream energy_file(params.dir + "/energy_landscape.txt");
        energy_file << "# Energy landscape for BCAO Honeycomb\n";
        for (size_t i = 0; i < energy_landscape.size(); ++i) {
            energy_file << i << " " << energy_landscape[i] << "\n";
        }
        energy_file.close();
        cout << "Energy landscape saved to: " << params.dir + "/energy_landscape.txt" << "\n";

        ofstream twist_file(params.dir + "/twist_matrix.txt");
        twist_file << "Twist Matrix:\n";
        for (const auto& row : MC.twist_matrices) {
            for (const auto& val : row) {
                twist_file << val << " ";
            }
            twist_file << "\n";
        }
        twist_file.close();
        cout << "Twist matrix saved to: " << params.dir + "/twist_matrix.txt" << "\n";

        timing_helpers::log_timing(timing_file, "step_4_postprocessing_and_output", MPI_Wtime() - t_step, rank);
        timing_helpers::log_timing(timing_file, "total_PT_BCAO_honeycomb", MPI_Wtime() - t_total, rank);
    }
    else{
        // Other ranks can perform their own tasks or just wait
        cout << "Rank " << rank << " completed its part of the simulation.\n";
    }
}

const array<double, 3> operator*(const array<double, 3>& vec, double scalar) {
    return {vec[0] * scalar, vec[1] * scalar, vec[2] * scalar};
}

const array<double, 3> operator+(const array<double, 3>& a, const array<double, 3>& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

int main(int argc, char** argv) {
    string param_file = "bcao_pt_parameters.txt";

    if (argc > 1) {
        string arg = argv[1];
        if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [parameter_file] [--field_scan]\n";
            cout << "  parameter_file    Path to parameter file (default: bcao_pt_parameters.txt)\n";
            cout << "  --help, -h        Show this help message\n";
            cout << "  --create-params   Create default parameter file\n";
            cout << "  --field_scan      Enable field-scan mode (must be passed as argv[2])\n\n";
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

    // Enable field-scan mode only when argv[2] is --field_scan
    bool field_scan = (argc > 2 && string(argv[2]) == "--field_scan");
    bool magnetotropic = (argc > 2 && string(argv[2]) == "--magnetotropic");

    SimulationParams params = read_parameters(param_file);
    
    int slurm_job_id = argc > 3 ? stoi(argv[3]) : 0;
    int slurm_job_range = argc > 4 ? stoi(argv[4]) : 0;

    // int slurm_chunk = params.num_steps / slurm_job_range;


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
        if (field_scan) {
            cout << "Field scan enabled: h from " << params.h_start << " to " << params.h_end << " in " << params.num_steps << " steps\n";
        }
        filesystem::create_directories(params.dir);
    }
    const string overview_timing = params.dir + "/timing_overview.log";
    double t_total_program = MPI_Wtime();
    const array<double, 3> c_axis = {{0,0,1}};
    
    if (field_scan || magnetotropic) {
        params.num_trials = 1; // Force single trial in field-scan mode
    }

    for (size_t i = 0; i < params.num_trials; ++i) {
        SimulationParams trial_params = params;
        double t_trial = MPI_Wtime();

        if (rank == 0) {
            cout << "Starting trial " << i + 1 << " of " << params.num_trials << "\n";
        }
        // Synchronize all processes before starting the next trial
        MPI_Barrier(MPI_COMM_WORLD);

        // If field scan is enabled, sweep over h; otherwise, run a single h
        if (field_scan) {
            size_t steps = params.num_steps;
            if (steps == 0) {
                if (rank == 0) cout << "Warning: field_scan enabled but num_steps==0; nothing to do.\n";
                continue;
            }
            std::cout << "SLURM job id: " << slurm_job_id << ", range: " << slurm_job_range << "\n";
            for (size_t s = slurm_job_id; s < steps; ++slurm_job_range) {
                double hval = (steps > 1)
                    ? (params.h_start + (double)s * (params.h_end - params.h_start) / (double)(steps - 1))
                    : params.h_start;
                trial_params.h = hval;
                trial_params.num_trials = 1;
                ostringstream hs; hs.setf(ios::fixed); hs << setprecision(6) << hval;
                trial_params.dir = params.dir + "/trial_" + to_string(i) + "/h_" + hs.str();

                if (rank == 0) {
                    cout << "  Field step " << (s + 1) << "/" << steps << ": h = " << hval << "\n";
                }
                // Barrier before each field step
                MPI_Barrier(MPI_COMM_WORLD);
                double t_field_step = MPI_Wtime();
                PT_BCAO_honeycomb(trial_params, true);
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0) {
                    timing_helpers::log_timing(overview_timing, "trial_" + to_string(i) + "_field_step_" + to_string(s) + "_elapsed", MPI_Wtime() - t_field_step, rank);
                }
            }
        } else if (magnetotropic) {
            size_t steps = params.num_steps;
            for (size_t s = slurm_job_id; s < steps; ++slurm_job_range) {
                trial_params.h = params.h_end;
                trial_params.num_trials = 1;
                ostringstream hs; hs.setf(ios::fixed); hs << setprecision(6) << trial_params.h;
                trial_params.dir = params.dir + "/trial_" + to_string(i) + "/h_" + hs.str();
                trial_params.field_dir = c_axis * sin((double)s * M_PI / (double)(steps - 1) - M_PI/2) + array<double,3>{1,0,0} * cos((double)s * M_PI / (double)(steps - 1) - M_PI/2);
                if (rank == 0) {
                    cout << "  Field step " << (s + 1) << "/" << steps << ": h = " << trial_params.h << "\n";
                    cout << "    Field direction: [" << trial_params.field_dir[0] << "," << trial_params.field_dir[1] << "," << trial_params.field_dir[2] << "]\n";
                }
                // Barrier before each field step
                MPI_Barrier(MPI_COMM_WORLD);
                double t_field_step = MPI_Wtime();
                PT_BCAO_honeycomb(trial_params, true);
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0) {
                    timing_helpers::log_timing(overview_timing, "trial_" + to_string(i) + "_field_step_" + to_string(s) + "_elapsed", MPI_Wtime() - t_field_step, rank);
                }
            }
        } else {
            trial_params.dir = params.dir + "/trial_" + to_string(i);
            // Barrier before the fixed-h run
            MPI_Barrier(MPI_COMM_WORLD);
            // Run the Parallel Tempering simulation
            PT_BCAO_honeycomb(trial_params, true);
        }
        // End of trial
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            timing_helpers::log_timing(overview_timing, "trial_" + to_string(i) + "_elapsed", MPI_Wtime() - t_trial, rank);
        }
    }
    // Total program time
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        timing_helpers::log_timing(overview_timing, "program_total_elapsed", MPI_Wtime() - t_total_program, rank);
    }
    
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}