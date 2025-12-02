#ifndef SPIN_CONFIG_H
#define SPIN_CONFIG_H

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
#include <cmath>

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
    PARAMETER_SWEEP,             // Sweep over any Hamiltonian parameter
    CUSTOM
};

// Configuration structure for spin simulations
struct SpinConfig {
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
    bool gaussian_move = false;
    bool save_observables = true;
    bool deterministic = false;
    bool T_zero = false;
    size_t n_deterministics = 1000;
    
    // Molecular dynamics parameters
    double md_time_start = 0.0;
    double md_time_end = 100.0;
    double md_timestep = 0.01;
    size_t md_save_interval = 1;  // Save every N steps
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
    bool parallel_tau = true;  // If true, parallelize tau loop across MPI ranks when num_trials == 1
    
    // Parameter sweep parameters
    string sweep_parameter = "";      // Name of parameter to sweep (deprecated, use sweep_parameters)
    double sweep_start = 0.0;          // Starting value (deprecated)
    double sweep_end = 1.0;            // Ending value (deprecated)
    double sweep_step = 0.1;           // Step size (deprecated)
    
    // N-dimensional parameter sweep
    vector<string> sweep_parameters;   // Names of parameters to sweep
    vector<double> sweep_starts;       // Starting values for each parameter
    vector<double> sweep_ends;         // Ending values for each parameter
    vector<double> sweep_steps;        // Step sizes for each parameter
    
    SimulationType sweep_base_simulation = SimulationType::SIMULATED_ANNEALING;  // Simulation to run at each sweep point
    
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
    static SpinConfig from_file(const string& filename);
    
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
    if (s == "parameter_sweep" || s == "PARAMETER_SWEEP" || s == "sweep") return SimulationType::PARAMETER_SWEEP;
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

inline vector<double> parse_double_list(const string& str) {
    vector<double> result;
    string s = trim(str);
    s.erase(std::remove(s.begin(), s.end(), '['), s.end());
    s.erase(std::remove(s.begin(), s.end(), ']'), s.end());
    
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')) {
        result.push_back(stod(trim(item)));
    }
    return result;
}

inline vector<string> parse_string_list(const string& str) {
    vector<string> result;
    string s = trim(str);
    s.erase(std::remove(s.begin(), s.end(), '['), s.end());
    s.erase(std::remove(s.begin(), s.end(), ']'), s.end());
    
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')) {
        result.push_back(trim(item));
    }
    return result;
}

inline bool parse_bool(const string& str) {
    string s = trim(str);
    return (s == "true" || s == "True" || s == "TRUE" || s == "1" || s == "yes" || s == "Yes" || s == "YES");
}

// Implementation of config file parsing



#endif // SPIN_CONFIG_H
