#ifndef SIMULATION_CONFIG_H
#define SIMULATION_CONFIG_H

#include <string>
#include <array>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>

using namespace std;

// Enum for geometry types
enum class GeometryType {
    HONEYCOMB,
    PYROCHLORE,
    KAGOME,
    TRIANGULAR,
    SQUARE
};

// Enum for simulation methods
enum class SimulationMethod {
    SIMULATED_ANNEALING,
    MOLECULAR_DYNAMICS,
    PARALLEL_TEMPERING,
    FIELD_SCAN,
    PARAMETER_SCAN
};

// Configuration structure for simulations
struct SimulationConfig {
    // Geometry
    GeometryType geometry = GeometryType::HONEYCOMB;
    array<int, 3> lattice_size = {24, 24, 1};
    int num_sublattices = 2;
    
    // Simulation method
    SimulationMethod method = SimulationMethod::SIMULATED_ANNEALING;
    
    // General simulation parameters
    int num_trials = 1;
    string output_dir = "output";
    double initial_step_size = 0.5;
    bool use_twist_boundary = false;
    
    // Temperature parameters
    double T_start = 10.0;
    double T_end = 0.001;
    int annealing_steps = 100000;
    int equilibration_steps = 1000;
    double cooling_rate = 0.9;
    
    // Molecular dynamics parameters
    double md_time = 100.0;
    double md_timestep = 0.01;
    
    // Parallel tempering parameters
    int num_replicas = 8;
    int pt_sweeps_per_exchange = 10;
    int pt_exchange_frequency = 50;
    
    // Field parameters
    double field_strength = 0.0;
    array<double, 3> field_direction = {0, 1, 0};
    array<double, 3> field_scaling = {1.0, 1.0, 1.0};  // g-factors or anisotropy
    
    // Field scan parameters
    bool do_field_scan = false;
    double field_start = 0.0;
    double field_end = 1.0;
    int field_steps = 10;
    
    // Parameter scan parameters
    bool do_parameter_scan = false;
    string scan_parameter = "";
    double scan_start = 0.0;
    double scan_end = 1.0;
    int scan_steps = 10;
    
    // Hamiltonian parameters (generic storage)
    map<string, double> hamiltonian_params;
    
    // Advanced: custom interaction matrices
    bool use_custom_interactions = false;
    vector<array<array<double, 3>, 3>> custom_J_matrices;
    
    // MPI parameters
    bool use_mpi = true;
    
    // Parse configuration from file
    static SimulationConfig from_file(const string& filename);
    
    // Write configuration to file
    void to_file(const string& filename) const;
    
    // Get specific Hamiltonian parameters with defaults
    double get_param(const string& key, double default_val = 0.0) const {
        auto it = hamiltonian_params.find(key);
        return (it != hamiltonian_params.end()) ? it->second : default_val;
    }
    
    // Set Hamiltonian parameter
    void set_param(const string& key, double value) {
        hamiltonian_params[key] = value;
    }
    
    // Validate configuration
    bool validate() const;
    
    // Print configuration
    void print() const;
};

// Utility functions for parsing
inline string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

inline GeometryType parse_geometry(const string& str) {
    string s = trim(str);
    if (s == "honeycomb" || s == "HONEYCOMB") return GeometryType::HONEYCOMB;
    if (s == "pyrochlore" || s == "PYROCHLORE") return GeometryType::PYROCHLORE;
    if (s == "kagome" || s == "KAGOME") return GeometryType::KAGOME;
    if (s == "triangular" || s == "TRIANGULAR") return GeometryType::TRIANGULAR;
    if (s == "square" || s == "SQUARE") return GeometryType::SQUARE;
    throw runtime_error("Unknown geometry type: " + str);
}

inline SimulationMethod parse_method(const string& str) {
    string s = trim(str);
    if (s == "simulated_annealing" || s == "SA") return SimulationMethod::SIMULATED_ANNEALING;
    if (s == "molecular_dynamics" || s == "MD") return SimulationMethod::MOLECULAR_DYNAMICS;
    if (s == "parallel_tempering" || s == "PT") return SimulationMethod::PARALLEL_TEMPERING;
    if (s == "field_scan") return SimulationMethod::FIELD_SCAN;
    if (s == "parameter_scan") return SimulationMethod::PARAMETER_SCAN;
    throw runtime_error("Unknown simulation method: " + str);
}

inline array<double, 3> parse_vector3(const string& str) {
    array<double, 3> vec = {0, 0, 0};
    stringstream ss(str);
    string item;
    int i = 0;
    while (getline(ss, item, ',') && i < 3) {
        vec[i++] = stod(trim(item));
    }
    return vec;
}

inline array<int, 3> parse_int_vector3(const string& str) {
    array<int, 3> vec = {0, 0, 0};
    stringstream ss(str);
    string item;
    int i = 0;
    while (getline(ss, item, ',') && i < 3) {
        vec[i++] = stoi(trim(item));
    }
    return vec;
}

inline bool parse_bool(const string& str) {
    string s = trim(str);
    return (s == "true" || s == "True" || s == "1" || s == "yes" || s == "Yes");
}

// Implementation of config file parsing
inline SimulationConfig SimulationConfig::from_file(const string& filename) {
    SimulationConfig config;
    ifstream file(filename);
    
    if (!file.is_open()) {
        throw runtime_error("Cannot open config file: " + filename);
    }
    
    string line;
    int line_num = 0;
    
    while (getline(file, line)) {
        line_num++;
        
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#' || line[0] == '/') continue;
        
        // Find the equals sign
        size_t eq_pos = line.find('=');
        if (eq_pos == string::npos) continue;
        
        string key = trim(line.substr(0, eq_pos));
        string value = trim(line.substr(eq_pos + 1));
        
        try {
            // Parse different configuration options
            if (key == "geometry") {
                config.geometry = parse_geometry(value);
            }
            else if (key == "lattice_size") {
                config.lattice_size = parse_int_vector3(value);
            }
            else if (key == "num_sublattices") {
                config.num_sublattices = stoi(value);
            }
            else if (key == "method") {
                config.method = parse_method(value);
            }
            else if (key == "num_trials") {
                config.num_trials = stoi(value);
            }
            else if (key == "output_dir") {
                config.output_dir = value;
            }
            else if (key == "initial_step_size") {
                config.initial_step_size = stod(value);
            }
            else if (key == "use_twist_boundary" || key == "tbc") {
                config.use_twist_boundary = parse_bool(value);
            }
            else if (key == "T_start") {
                config.T_start = stod(value);
            }
            else if (key == "T_end") {
                config.T_end = stod(value);
            }
            else if (key == "annealing_steps") {
                config.annealing_steps = stoi(value);
            }
            else if (key == "equilibration_steps") {
                config.equilibration_steps = stoi(value);
            }
            else if (key == "cooling_rate") {
                config.cooling_rate = stod(value);
            }
            else if (key == "md_time") {
                config.md_time = stod(value);
            }
            else if (key == "md_timestep") {
                config.md_timestep = stod(value);
            }
            else if (key == "num_replicas") {
                config.num_replicas = stoi(value);
            }
            else if (key == "pt_sweeps_per_exchange") {
                config.pt_sweeps_per_exchange = stoi(value);
            }
            else if (key == "pt_exchange_frequency") {
                config.pt_exchange_frequency = stoi(value);
            }
            else if (key == "field_strength" || key == "h") {
                config.field_strength = stod(value);
            }
            else if (key == "field_direction") {
                config.field_direction = parse_vector3(value);
            }
            else if (key == "field_scaling") {
                config.field_scaling = parse_vector3(value);
            }
            else if (key == "do_field_scan") {
                config.do_field_scan = parse_bool(value);
            }
            else if (key == "field_start" || key == "h_start") {
                config.field_start = stod(value);
            }
            else if (key == "field_end" || key == "h_end") {
                config.field_end = stod(value);
            }
            else if (key == "field_steps") {
                config.field_steps = stoi(value);
            }
            else if (key == "do_parameter_scan") {
                config.do_parameter_scan = parse_bool(value);
            }
            else if (key == "scan_parameter") {
                config.scan_parameter = value;
            }
            else if (key == "scan_start") {
                config.scan_start = stod(value);
            }
            else if (key == "scan_end") {
                config.scan_end = stod(value);
            }
            else if (key == "scan_steps") {
                config.scan_steps = stoi(value);
            }
            else if (key == "use_mpi") {
                config.use_mpi = parse_bool(value);
            }
            else {
                // Treat as Hamiltonian parameter
                config.hamiltonian_params[key] = stod(value);
            }
        } catch (const exception& e) {
            cerr << "Error parsing line " << line_num << " in " << filename << ": " << e.what() << endl;
            cerr << "Line content: " << line << endl;
            throw;
        }
    }
    
    file.close();
    return config;
}

inline void SimulationConfig::to_file(const string& filename) const {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot write config file: " + filename);
    }
    
    file << "# Simulation Configuration File\n";
    file << "# Generated automatically\n\n";
    
    file << "# Geometry\n";
    file << "geometry = ";
    switch (geometry) {
        case GeometryType::HONEYCOMB: file << "honeycomb"; break;
        case GeometryType::PYROCHLORE: file << "pyrochlore"; break;
        case GeometryType::KAGOME: file << "kagome"; break;
        case GeometryType::TRIANGULAR: file << "triangular"; break;
        case GeometryType::SQUARE: file << "square"; break;
    }
    file << "\n";
    file << "lattice_size = " << lattice_size[0] << "," << lattice_size[1] << "," << lattice_size[2] << "\n";
    file << "num_sublattices = " << num_sublattices << "\n\n";
    
    file << "# Simulation Method\n";
    file << "method = ";
    switch (method) {
        case SimulationMethod::SIMULATED_ANNEALING: file << "simulated_annealing"; break;
        case SimulationMethod::MOLECULAR_DYNAMICS: file << "molecular_dynamics"; break;
        case SimulationMethod::PARALLEL_TEMPERING: file << "parallel_tempering"; break;
        case SimulationMethod::FIELD_SCAN: file << "field_scan"; break;
        case SimulationMethod::PARAMETER_SCAN: file << "parameter_scan"; break;
    }
    file << "\n";
    file << "num_trials = " << num_trials << "\n";
    file << "output_dir = " << output_dir << "\n\n";
    
    file << "# Temperature Parameters\n";
    file << "T_start = " << T_start << "\n";
    file << "T_end = " << T_end << "\n";
    file << "annealing_steps = " << annealing_steps << "\n";
    file << "cooling_rate = " << cooling_rate << "\n\n";
    
    file << "# Field Parameters\n";
    file << "field_strength = " << field_strength << "\n";
    file << "field_direction = " << field_direction[0] << "," << field_direction[1] << "," << field_direction[2] << "\n\n";
    
    file << "# Hamiltonian Parameters\n";
    for (const auto& [key, value] : hamiltonian_params) {
        file << key << " = " << value << "\n";
    }
    
    file.close();
}

inline bool SimulationConfig::validate() const {
    bool valid = true;
    
    if (num_trials < 1) {
        cerr << "Error: num_trials must be >= 1\n";
        valid = false;
    }
    
    if (T_start < T_end) {
        cerr << "Error: T_start must be >= T_end\n";
        valid = false;
    }
    
    if (do_field_scan && field_steps < 1) {
        cerr << "Error: field_steps must be >= 1 for field scan\n";
        valid = false;
    }
    
    return valid;
}

inline void SimulationConfig::print() const {
    cout << "==================== Simulation Configuration ====================\n";
    cout << "Geometry: ";
    switch (geometry) {
        case GeometryType::HONEYCOMB: cout << "Honeycomb"; break;
        case GeometryType::PYROCHLORE: cout << "Pyrochlore"; break;
        case GeometryType::KAGOME: cout << "Kagome"; break;
        case GeometryType::TRIANGULAR: cout << "Triangular"; break;
        case GeometryType::SQUARE: cout << "Square"; break;
    }
    cout << "\n";
    cout << "Lattice size: " << lattice_size[0] << " x " << lattice_size[1] << " x " << lattice_size[2] << "\n";
    cout << "Method: ";
    switch (method) {
        case SimulationMethod::SIMULATED_ANNEALING: cout << "Simulated Annealing"; break;
        case SimulationMethod::MOLECULAR_DYNAMICS: cout << "Molecular Dynamics"; break;
        case SimulationMethod::PARALLEL_TEMPERING: cout << "Parallel Tempering"; break;
        case SimulationMethod::FIELD_SCAN: cout << "Field Scan"; break;
        case SimulationMethod::PARAMETER_SCAN: cout << "Parameter Scan"; break;
    }
    cout << "\n";
    cout << "Trials: " << num_trials << "\n";
    cout << "Output: " << output_dir << "\n";
    cout << "Temperature: " << T_start << " -> " << T_end << "\n";
    cout << "Field: " << field_strength << " along [" 
         << field_direction[0] << "," << field_direction[1] << "," << field_direction[2] << "]\n";
    
    if (!hamiltonian_params.empty()) {
        cout << "\nHamiltonian Parameters:\n";
        for (const auto& [key, value] : hamiltonian_params) {
            cout << "  " << key << " = " << value << "\n";
        }
    }
    cout << "================================================================\n";
}

#endif // SIMULATION_CONFIG_H
