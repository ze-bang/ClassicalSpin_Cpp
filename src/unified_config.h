#ifndef UNIFIED_CONFIG_H
#define UNIFIED_CONFIG_H

#include <string>
#include <array>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <algorithm>

using namespace std;

// Enum for lattice/system types
enum class SystemType {
    HONEYCOMB_BCAO,           // Ba3CoAl2O9 honeycomb
    HONEYCOMB_KITAEV,         // Generic Kitaev honeycomb
    PYROCHLORE,               // Pyrochlore lattice
    TMFEO3,                   // TmFeO3 (mixed SU2+SU3)
    CUSTOM                    // Custom from JSON
};

// Enum for simulation methods
enum class SimulationType {
    SIMULATED_ANNEALING,
    PARALLEL_TEMPERING,
    MOLECULAR_DYNAMICS,
    PUMP_PROBE,
    TWOD_COHERENT_SPECTROSCOPY,  // 2DCS / pump-probe spectroscopy
    CUSTOM
};

// Configuration structure for unified simulations
struct UnifiedConfig {
    // System specification
    SystemType system = SystemType::HONEYCOMB_BCAO;
    array<size_t, 3> lattice_size = {24, 24, 1};
    float spin_length = 1.0;
    
    // Simulation method
    SimulationType simulation = SimulationType::SIMULATED_ANNEALING;
    
    // General simulation parameters
    int num_trials = 1;
    string output_dir = "output";
    double initial_step_size = 0.5;
    bool use_twist_boundary = false;
    
    // Temperature parameters
    double T_start = 10.0;
    double T_end = 0.001;
    size_t annealing_steps = 100000;
    size_t equilibration_steps = 1000;
    double cooling_rate = 0.9;
    
    // Molecular dynamics parameters
    double md_time_start = 0.0;
    double md_time_end = 100.0;
    double md_timestep = 0.01;
    string md_integrator = "dopri5";  // euler, rk2, rk4, rk5, dopri5, rk78, bulirsch_stoer
    bool use_gpu = false;
    
    // Parallel tempering parameters
    size_t num_replicas = 8;
    size_t pt_sweeps_per_exchange = 10;
    size_t pt_exchange_frequency = 50;
    size_t overrelaxation_rate = 0;
    size_t probe_rate = 2000;
    vector<int> ranks_to_write = {0};
    
    // Pump-probe parameters
    double pump_amplitude = 1.0;
    double pump_width = 10.0;
    double pump_frequency = 0.0;
    double pump_time = 0.0;
    array<double, 3> pump_direction = {0, 1, 0};
    
    double probe_amplitude = 0.1;
    double probe_width = 10.0;
    double probe_frequency = 0.0;
    double probe_time = 50.0;
    array<double, 3> probe_direction = {0, 1, 0};
    
    // 2DCS spectroscopy parameters (delay time scan)
    double tau_start = -200.0;
    double tau_end = 200.0;
    double tau_step = 1.0;
    
    // Field parameters
    double field_strength = 0.0;
    array<double, 3> field_direction = {0, 1, 0};
    array<double, 3> g_factor = {1.0, 1.0, 1.0};  // g-factors or anisotropy
    
    // Hamiltonian parameters (generic storage for system-specific parameters)
    map<string, double> hamiltonian_params;
    
    // Initial configuration
    string initial_spin_config = "";  // Empty means random
    bool use_ferromagnetic_init = false;
    array<double, 3> ferromagnetic_direction = {0, 0, 1};
    
    // MPI parameters
    bool use_mpi = true;
    
    // Parse configuration from file
    static UnifiedConfig from_file(const string& filename);
    
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

inline SystemType parse_system(const string& str) {
    string s = trim(str);
    if (s == "honeycomb_bcao" || s == "HONEYCOMB_BCAO" || s == "BCAO") return SystemType::HONEYCOMB_BCAO;
    if (s == "honeycomb_kitaev" || s == "HONEYCOMB_KITAEV" || s == "KITAEV") return SystemType::HONEYCOMB_KITAEV;
    if (s == "pyrochlore" || s == "PYROCHLORE") return SystemType::PYROCHLORE;
    if (s == "tmfeo3" || s == "TMFEO3" || s == "TmFeO3") return SystemType::TMFEO3;
    if (s == "custom" || s == "CUSTOM") return SystemType::CUSTOM;
    throw runtime_error("Unknown system type: " + str);
}

inline SimulationType parse_simulation(const string& str) {
    string s = trim(str);
    if (s == "simulated_annealing" || s == "SA" || s == "annealing") return SimulationType::SIMULATED_ANNEALING;
    if (s == "parallel_tempering" || s == "PT" || s == "tempering") return SimulationType::PARALLEL_TEMPERING;
    if (s == "molecular_dynamics" || s == "MD" || s == "dynamics") return SimulationType::MOLECULAR_DYNAMICS;
    if (s == "pump_probe" || s == "PUMP_PROBE" || s == "pump-probe") return SimulationType::PUMP_PROBE;
    if (s == "2dcs" || s == "2DCS" || s == "spectroscopy" || s == "pump_probe_spectroscopy") return SimulationType::TWOD_COHERENT_SPECTROSCOPY;
    if (s == "custom" || s == "CUSTOM") return SimulationType::CUSTOM;
    throw runtime_error("Unknown simulation type: " + str);
}

inline array<double, 3> parse_vector3(const string& str) {
    array<double, 3> vec = {0, 0, 0};
    string s = trim(str);
    // Remove parentheses and brackets
    s.erase(std::remove(s.begin(), s.end(), '('), s.end());
    s.erase(std::remove(s.begin(), s.end(), ')'), s.end());
    s.erase(std::remove(s.begin(), s.end(), '['), s.end());
    s.erase(std::remove(s.begin(), s.end(), ']'), s.end());
    
    stringstream ss(s);
    string item;
    int i = 0;
    while (getline(ss, item, ',') && i < 3) {
        vec[i++] = stod(trim(item));
    }
    return vec;
}

inline array<size_t, 3> parse_size_vector3(const string& str) {
    array<size_t, 3> vec = {0, 0, 0};
    string s = trim(str);
    s.erase(std::remove(s.begin(), s.end(), '('), s.end());
    s.erase(std::remove(s.begin(), s.end(), ')'), s.end());
    s.erase(std::remove(s.begin(), s.end(), '['), s.end());
    s.erase(std::remove(s.begin(), s.end(), ']'), s.end());
    
    stringstream ss(s);
    string item;
    int i = 0;
    while (getline(ss, item, ',') && i < 3) {
        vec[i++] = stoull(trim(item));
    }
    return vec;
}

inline vector<int> parse_int_list(const string& str) {
    vector<int> result;
    string s = trim(str);
    s.erase(std::remove(s.begin(), s.end(), '['), s.end());
    s.erase(std::remove(s.begin(), s.end(), ']'), s.end());
    
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')) {
        result.push_back(stoi(trim(item)));
    }
    return result;
}

inline bool parse_bool(const string& str) {
    string s = trim(str);
    return (s == "true" || s == "True" || s == "TRUE" || s == "1" || s == "yes" || s == "Yes" || s == "YES");
}

// Implementation of config file parsing
inline UnifiedConfig UnifiedConfig::from_file(const string& filename) {
    UnifiedConfig config;
    ifstream file(filename);
    
    if (!file.is_open()) {
        throw runtime_error("Cannot open config file: " + filename);
    }
    
    string line;
    int line_num = 0;
    
    while (getline(file, line)) {
        line_num++;
        
        // Skip comments and empty lines
        size_t comment_pos = line.find('#');
        if (comment_pos != string::npos) {
            line = line.substr(0, comment_pos);
        }
        line = trim(line);
        if (line.empty() || line[0] == '/' || line[0] == ';') continue;
        
        // Find the delimiter (= or :)
        size_t eq_pos = line.find('=');
        size_t colon_pos = line.find(':');
        size_t delim_pos = (eq_pos != string::npos) ? eq_pos : colon_pos;
        
        if (delim_pos == string::npos) continue;
        
        string key = trim(line.substr(0, delim_pos));
        string value = trim(line.substr(delim_pos + 1));
        
        try {
            // Parse different configuration options
            if (key == "system" || key == "system_type") {
                config.system = parse_system(value);
            }
            else if (key == "lattice_size") {
                config.lattice_size = parse_size_vector3(value);
            }
            else if (key == "spin_length") {
                config.spin_length = stof(value);
            }
            else if (key == "simulation" || key == "simulation_type") {
                config.simulation = parse_simulation(value);
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
            else if (key == "T_start" || key == "temperature_start") {
                config.T_start = stod(value);
            }
            else if (key == "T_end" || key == "temperature_end") {
                config.T_end = stod(value);
            }
            else if (key == "annealing_steps") {
                config.annealing_steps = stoull(value);
            }
            else if (key == "equilibration_steps") {
                config.equilibration_steps = stoull(value);
            }
            else if (key == "cooling_rate") {
                config.cooling_rate = stod(value);
            }
            else if (key == "md_time_start") {
                config.md_time_start = stod(value);
            }
            else if (key == "md_time_end") {
                config.md_time_end = stod(value);
            }
            else if (key == "md_timestep" || key == "dt") {
                config.md_timestep = stod(value);
            }
            else if (key == "md_integrator" || key == "integrator") {
                config.md_integrator = value;
            }
            else if (key == "use_gpu" || key == "gpu") {
                config.use_gpu = parse_bool(value);
            }
            else if (key == "num_replicas") {
                config.num_replicas = stoull(value);
            }
            else if (key == "pt_sweeps_per_exchange") {
                config.pt_sweeps_per_exchange = stoull(value);
            }
            else if (key == "pt_exchange_frequency") {
                config.pt_exchange_frequency = stoull(value);
            }
            else if (key == "overrelaxation_rate") {
                config.overrelaxation_rate = stoull(value);
            }
            else if (key == "probe_rate") {
                config.probe_rate = stoull(value);
            }
            else if (key == "ranks_to_write") {
                config.ranks_to_write = parse_int_list(value);
            }
            else if (key == "pump_amplitude") {
                config.pump_amplitude = stod(value);
            }
            else if (key == "pump_width") {
                config.pump_width = stod(value);
            }
            else if (key == "pump_frequency") {
                config.pump_frequency = stod(value);
            }
            else if (key == "pump_time") {
                config.pump_time = stod(value);
            }
            else if (key == "pump_direction") {
                config.pump_direction = parse_vector3(value);
            }
            else if (key == "probe_amplitude") {
                config.probe_amplitude = stod(value);
            }
            else if (key == "probe_width") {
                config.probe_width = stod(value);
            }
            else if (key == "probe_frequency") {
                config.probe_frequency = stod(value);
            }
            else if (key == "probe_time") {
                config.probe_time = stod(value);
            }
            else if (key == "probe_direction") {
                config.probe_direction = parse_vector3(value);
            }
            else if (key == "tau_start") {
                config.tau_start = stod(value);
            }
            else if (key == "tau_end") {
                config.tau_end = stod(value);
            }
            else if (key == "tau_step") {
                config.tau_step = stod(value);
            }
            else if (key == "field_strength" || key == "h") {
                config.field_strength = stod(value);
            }
            else if (key == "field_direction") {
                config.field_direction = parse_vector3(value);
            }
            else if (key == "g_factor") {
                config.g_factor = parse_vector3(value);
            }
            else if (key == "initial_spin_config") {
                config.initial_spin_config = value;
            }
            else if (key == "use_ferromagnetic_init") {
                config.use_ferromagnetic_init = parse_bool(value);
            }
            else if (key == "ferromagnetic_direction") {
                config.ferromagnetic_direction = parse_vector3(value);
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

inline void UnifiedConfig::to_file(const string& filename) const {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot write config file: " + filename);
    }
    
    file << "# Unified Simulation Configuration File\n";
    file << "# Generated automatically\n\n";
    
    file << "# System\n";
    file << "system = ";
    switch (system) {
        case SystemType::HONEYCOMB_BCAO: file << "honeycomb_bcao"; break;
        case SystemType::HONEYCOMB_KITAEV: file << "honeycomb_kitaev"; break;
        case SystemType::PYROCHLORE: file << "pyrochlore"; break;
        case SystemType::TMFEO3: file << "tmfeo3"; break;
        case SystemType::CUSTOM: file << "custom"; break;
    }
    file << "\n";
    file << "lattice_size = " << lattice_size[0] << "," << lattice_size[1] << "," << lattice_size[2] << "\n";
    file << "spin_length = " << spin_length << "\n\n";
    
    file << "# Simulation Type\n";
    file << "simulation = ";
    switch (simulation) {
        case SimulationType::SIMULATED_ANNEALING: file << "simulated_annealing"; break;
        case SimulationType::PARALLEL_TEMPERING: file << "parallel_tempering"; break;
        case SimulationType::MOLECULAR_DYNAMICS: file << "molecular_dynamics"; break;
        case SimulationType::PUMP_PROBE: file << "pump_probe"; break;
        case SimulationType::CUSTOM: file << "custom"; break;
    }
    file << "\n";
    file << "num_trials = " << num_trials << "\n";
    file << "output_dir = " << output_dir << "\n\n";
    
    file << "# Temperature Parameters\n";
    file << "T_start = " << T_start << "\n";
    file << "T_end = " << T_end << "\n";
    file << "annealing_steps = " << annealing_steps << "\n";
    file << "cooling_rate = " << cooling_rate << "\n\n";
    
    file << "# Molecular Dynamics Parameters\n";
    file << "md_time_start = " << md_time_start << "\n";
    file << "md_time_end = " << md_time_end << "\n";
    file << "md_timestep = " << md_timestep << "\n";
    file << "md_integrator = " << md_integrator << "\n";
    file << "use_gpu = " << (use_gpu ? "true" : "false") << "\n\n";
    
    file << "# Field Parameters\n";
    file << "field_strength = " << field_strength << "\n";
    file << "field_direction = " << field_direction[0] << "," << field_direction[1] << "," << field_direction[2] << "\n\n";
    
    file << "# Hamiltonian Parameters\n";
    for (const auto& [key, value] : hamiltonian_params) {
        file << key << " = " << value << "\n";
    }
    
    file.close();
}

inline bool UnifiedConfig::validate() const {
    bool valid = true;
    
    if (num_trials < 1) {
        cerr << "Error: num_trials must be >= 1\n";
        valid = false;
    }
    
    if (T_start < T_end && simulation == SimulationType::SIMULATED_ANNEALING) {
        cerr << "Error: T_start must be >= T_end for annealing\n";
        valid = false;
    }
    
    if (lattice_size[0] == 0 || lattice_size[1] == 0 || lattice_size[2] == 0) {
        cerr << "Error: lattice_size dimensions must be > 0\n";
        valid = false;
    }
    
    return valid;
}

inline void UnifiedConfig::print() const {
    cout << "==================== Unified Simulation Configuration ====================\n";
    cout << "System: ";
    switch (system) {
        case SystemType::HONEYCOMB_BCAO: cout << "BCAO Honeycomb"; break;
        case SystemType::HONEYCOMB_KITAEV: cout << "Kitaev Honeycomb"; break;
        case SystemType::PYROCHLORE: cout << "Pyrochlore"; break;
        case SystemType::TMFEO3: cout << "TmFeO3"; break;
        case SystemType::CUSTOM: cout << "Custom"; break;
    }
    cout << "\n";
    cout << "Lattice size: " << lattice_size[0] << " x " << lattice_size[1] << " x " << lattice_size[2] << "\n";
    cout << "Simulation: ";
    switch (simulation) {
        case SimulationType::SIMULATED_ANNEALING: cout << "Simulated Annealing"; break;
        case SimulationType::PARALLEL_TEMPERING: cout << "Parallel Tempering"; break;
        case SimulationType::MOLECULAR_DYNAMICS: cout << "Molecular Dynamics"; break;
        case SimulationType::PUMP_PROBE: cout << "Pump-Probe"; break;
        case SimulationType::CUSTOM: cout << "Custom"; break;
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
    cout << "=========================================================================\n";
}

#endif // UNIFIED_CONFIG_H
