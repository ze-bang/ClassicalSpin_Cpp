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

using std::string;
using std::array;
using std::vector;
using std::map;
using std::cout;
using std::endl;

// Enum for geometry types
enum class GeometryType {
    HONEYCOMB_SU2,      // Standard honeycomb with SU(2) spins
    PYROCHLORE_SU2,     // Pyrochlore with SU(2) spins
    KAGOME_SU2,         // Kagome with SU(2) spins
    TRIANGULAR_SU2,     // Triangular with SU(2) spins
    SQUARE_SU2,         // Square with SU(2) spins
    CUSTOM_SU2,         // Custom unit cell with SU(2) spins
    TMFEO3_MIXED        // TmFeO3 mixed SU(2)/SU(3) system
};

// Enum for simulation methods
enum class SimulationMethod {
    SIMULATED_ANNEALING,
    MOLECULAR_DYNAMICS,
    PARALLEL_TEMPERING,
    PUMP_PROBE,         // Single pulse time-resolved
    TWO_PULSE_2DCS,     // Two-pulse 2D coherent spectroscopy
    FIELD_SCAN,
    PARAMETER_SCAN
};

// Configuration structure for simulations
struct SimulationConfig {
    // ============================================================
    // GEOMETRY AND LATTICE
    // ============================================================
    GeometryType geometry = GeometryType::HONEYCOMB_SU2;
    array<size_t, 3> lattice_size = {24, 24, 1};
    size_t spin_dim = 3;            // 3 for SU(2), 8 for SU(3)
    float spin_length = 1.0;
    
    // ============================================================
    // SIMULATION METHOD
    // ============================================================
    SimulationMethod method = SimulationMethod::SIMULATED_ANNEALING;
    
    // ============================================================
    // GENERAL PARAMETERS
    // ============================================================
    size_t num_trials = 1;
    string output_dir = "output";
    bool use_twist_boundary = false;
    bool use_gpu = false;
    string integration_method = "dopri5";  // ODE integration method
    
    // ============================================================
    // TEMPERATURE PARAMETERS
    // ============================================================
    double T_start = 10.0;
    double T_end = 0.001;
    size_t annealing_steps = 100000;
    size_t equilibration_steps = 1000;
    double cooling_rate = 0.9;
    bool auto_tune_annealing = true;
    
    // ============================================================
    // MONTE CARLO PARAMETERS
    // ============================================================
    bool use_gaussian_move = false;
    double gaussian_sigma = 60.0;
    size_t overrelaxation_rate = 0;  // 0 = no overrelaxation
    bool save_observables = true;
    
    // ============================================================
    // MOLECULAR DYNAMICS PARAMETERS
    // ============================================================
    double md_time_start = 0.0;
    double md_time_end = 100.0;
    double md_timestep = 0.01;
    double md_save_interval = 0.1;
    
    // ============================================================
    // PARALLEL TEMPERING PARAMETERS
    // ============================================================
    size_t num_replicas = 8;
    size_t pt_sweeps_per_exchange = 10;
    size_t pt_exchange_interval = 50;
    size_t pt_probe_rate = 2000;
    bool pt_boundary_update = false;
    vector<int> pt_ranks_to_write = {0};
    
    // ============================================================
    // PUMP-PROBE PARAMETERS
    // ============================================================
    double pulse_amplitude = 0.1;
    double pulse_width = 1.0;
    double pulse_frequency = 0.0;
    double pump_time = 0.0;
    double probe_time = 0.0;
    vector<double> delay_times;      // For 2DCS
    size_t num_delay_steps = 20;
    double delay_start = 0.0;
    double delay_end = 10.0;
    array<double, 3> pulse_direction = {1.0, 0.0, 0.0};
    
    // ============================================================
    // FIELD PARAMETERS
    // ============================================================
    double field_strength = 0.0;
    array<double, 3> field_direction = {0.0, 1.0, 0.0};
    array<double, 3> field_scaling = {1.0, 1.0, 1.0};  // g-factors or anisotropy
    
    // Field scan parameters
    double field_start = 0.0;
    double field_end = 1.0;
    size_t field_steps = 10;
    
    // ============================================================
    // HAMILTONIAN PARAMETERS
    // ============================================================
    // Generic storage for material-specific parameters
    map<string, double> params;
    
    // Honeycomb/Kitaev parameters
    double J1xy = -7.6;
    double J1z = -1.2;
    double D = 0.1;
    double E = -0.1;
    double F = 0.0;
    double G = 0.0;
    double J3xy = 2.5;
    double J3z = -0.85;
    
    // Pyrochlore parameters
    double Jxx = 1.0;
    double Jyy = 1.0;
    double Jzz = 1.0;
    double Jpm = 0.0;
    double Jpmpm = 0.0;
    double Jxz = 0.0;
    double gxx = 0.01;
    double gyy = 4e-4;
    double gzz = 1.0;
    double theta_rot = 0.0;
    
    // TmFeO3 parameters (mixed system)
    double chi_Tm = 0.05;
    
    // ============================================================
    // INITIALIZATION OPTIONS
    // ============================================================
    string initial_config = "random";  // "random", "ferromagnetic", "from_file"
    string initial_config_file = "";
    array<double, 3> ferro_direction = {0.0, 0.0, 1.0};
    
    // ============================================================
    // METHODS
    // ============================================================
    
    /**
     * Parse configuration from file
     */
    static SimulationConfig from_file(const string& filename);
    
    /**
     * Write configuration to file
     */
    void to_file(const string& filename) const;
    
    /**
     * Get specific parameter with default
     */
    double get_param(const string& key, double default_val = 0.0) const {
        auto it = params.find(key);
        return (it != params.end()) ? it->second : default_val;
    }
    
    /**
     * Set parameter
     */
    void set_param(const string& key, double value) {
        params[key] = value;
    }
    
    /**
     * Validate configuration
     */
    bool validate() const;
    
    /**
     * Print configuration
     */
    void print() const;
};

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

inline string trim(const string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

inline GeometryType parse_geometry(const string& str) {
    string s = trim(str);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    
    if (s == "honeycomb" || s == "honeycomb_su2") return GeometryType::HONEYCOMB_SU2;
    if (s == "pyrochlore" || s == "pyrochlore_su2") return GeometryType::PYROCHLORE_SU2;
    if (s == "kagome" || s == "kagome_su2") return GeometryType::KAGOME_SU2;
    if (s == "triangular" || s == "triangular_su2") return GeometryType::TRIANGULAR_SU2;
    if (s == "square" || s == "square_su2") return GeometryType::SQUARE_SU2;
    if (s == "custom" || s == "custom_su2") return GeometryType::CUSTOM_SU2;
    if (s == "tmfeo3" || s == "tmfeo3_mixed") return GeometryType::TMFEO3_MIXED;
    
    throw std::runtime_error("Unknown geometry type: " + str);
}

inline SimulationMethod parse_method(const string& str) {
    string s = trim(str);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    
    if (s == "simulated_annealing" || s == "sa") return SimulationMethod::SIMULATED_ANNEALING;
    if (s == "molecular_dynamics" || s == "md") return SimulationMethod::MOLECULAR_DYNAMICS;
    if (s == "parallel_tempering" || s == "pt") return SimulationMethod::PARALLEL_TEMPERING;
    if (s == "pump_probe" || s == "pp") return SimulationMethod::PUMP_PROBE;
    if (s == "two_pulse_2dcs" || s == "2dcs") return SimulationMethod::TWO_PULSE_2DCS;
    if (s == "field_scan") return SimulationMethod::FIELD_SCAN;
    if (s == "parameter_scan") return SimulationMethod::PARAMETER_SCAN;
    
    throw std::runtime_error("Unknown simulation method: " + str);
}

inline array<double, 3> parse_vector3(const string& str) {
    array<double, 3> vec = {0, 0, 0};
    std::stringstream ss(str);
    string item;
    int i = 0;
    while (std::getline(ss, item, ',') && i < 3) {
        vec[i++] = std::stod(trim(item));
    }
    return vec;
}

inline array<size_t, 3> parse_size_vector3(const string& str) {
    array<size_t, 3> vec = {0, 0, 0};
    std::stringstream ss(str);
    string item;
    int i = 0;
    while (std::getline(ss, item, ',') && i < 3) {
        vec[i++] = std::stoul(trim(item));
    }
    return vec;
}

inline vector<double> parse_double_list(const string& str) {
    vector<double> result;
    std::stringstream ss(str);
    string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stod(trim(item)));
    }
    return result;
}

inline vector<int> parse_int_list(const string& str) {
    vector<int> result;
    std::stringstream ss(str);
    string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stoi(trim(item)));
    }
    return result;
}

inline bool parse_bool(const string& str) {
    string s = trim(str);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s == "true" || s == "1" || s == "yes");
}

// ============================================================
// CONFIGURATION FILE PARSING
// ============================================================

inline SimulationConfig SimulationConfig::from_file(const string& filename) {
    SimulationConfig config;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    
    string line;
    int line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        
        // Remove comments
        size_t comment_pos = line.find('#');
        if (comment_pos != string::npos) {
            line = line.substr(0, comment_pos);
        }
        
        line = trim(line);
        if (line.empty()) continue;
        
        // Find the equals sign
        size_t eq_pos = line.find('=');
        if (eq_pos == string::npos) continue;
        
        string key = trim(line.substr(0, eq_pos));
        string value = trim(line.substr(eq_pos + 1));
        
        try {
            // Geometry
            if (key == "geometry") config.geometry = parse_geometry(value);
            else if (key == "lattice_size") config.lattice_size = parse_size_vector3(value);
            else if (key == "spin_dim") config.spin_dim = std::stoul(value);
            else if (key == "spin_length") config.spin_length = std::stof(value);
            
            // Method
            else if (key == "method") config.method = parse_method(value);
            
            // General
            else if (key == "num_trials") config.num_trials = std::stoul(value);
            else if (key == "output_dir") config.output_dir = value;
            else if (key == "use_twist_boundary" || key == "tbc") config.use_twist_boundary = parse_bool(value);
            else if (key == "use_gpu") config.use_gpu = parse_bool(value);
            else if (key == "integration_method") config.integration_method = value;
            
            // Temperature
            else if (key == "T_start") config.T_start = std::stod(value);
            else if (key == "T_end") config.T_end = std::stod(value);
            else if (key == "annealing_steps") config.annealing_steps = std::stoul(value);
            else if (key == "equilibration_steps") config.equilibration_steps = std::stoul(value);
            else if (key == "cooling_rate") config.cooling_rate = std::stod(value);
            else if (key == "auto_tune_annealing") config.auto_tune_annealing = parse_bool(value);
            
            // Monte Carlo
            else if (key == "use_gaussian_move") config.use_gaussian_move = parse_bool(value);
            else if (key == "gaussian_sigma") config.gaussian_sigma = std::stod(value);
            else if (key == "overrelaxation_rate") config.overrelaxation_rate = std::stoul(value);
            else if (key == "save_observables") config.save_observables = parse_bool(value);
            
            // Molecular Dynamics
            else if (key == "md_time_start") config.md_time_start = std::stod(value);
            else if (key == "md_time_end") config.md_time_end = std::stod(value);
            else if (key == "md_timestep") config.md_timestep = std::stod(value);
            else if (key == "md_save_interval") config.md_save_interval = std::stod(value);
            
            // Parallel Tempering
            else if (key == "num_replicas") config.num_replicas = std::stoul(value);
            else if (key == "pt_sweeps_per_exchange") config.pt_sweeps_per_exchange = std::stoul(value);
            else if (key == "pt_exchange_interval") config.pt_exchange_interval = std::stoul(value);
            else if (key == "pt_probe_rate") config.pt_probe_rate = std::stoul(value);
            else if (key == "pt_boundary_update") config.pt_boundary_update = parse_bool(value);
            else if (key == "pt_ranks_to_write") config.pt_ranks_to_write = parse_int_list(value);
            
            // Pump-Probe
            else if (key == "pulse_amplitude") config.pulse_amplitude = std::stod(value);
            else if (key == "pulse_width") config.pulse_width = std::stod(value);
            else if (key == "pulse_frequency") config.pulse_frequency = std::stod(value);
            else if (key == "pump_time") config.pump_time = std::stod(value);
            else if (key == "probe_time") config.probe_time = std::stod(value);
            else if (key == "delay_times") config.delay_times = parse_double_list(value);
            else if (key == "num_delay_steps") config.num_delay_steps = std::stoul(value);
            else if (key == "delay_start") config.delay_start = std::stod(value);
            else if (key == "delay_end") config.delay_end = std::stod(value);
            else if (key == "pulse_direction") config.pulse_direction = parse_vector3(value);
            
            // Field
            else if (key == "field_strength" || key == "h") config.field_strength = std::stod(value);
            else if (key == "field_direction") config.field_direction = parse_vector3(value);
            else if (key == "field_scaling") config.field_scaling = parse_vector3(value);
            else if (key == "field_start") config.field_start = std::stod(value);
            else if (key == "field_end") config.field_end = std::stod(value);
            else if (key == "field_steps") config.field_steps = std::stoul(value);
            
            // Honeycomb/Kitaev Hamiltonian
            else if (key == "J1xy") config.J1xy = std::stod(value);
            else if (key == "J1z") config.J1z = std::stod(value);
            else if (key == "D") config.D = std::stod(value);
            else if (key == "E") config.E = std::stod(value);
            else if (key == "F") config.F = std::stod(value);
            else if (key == "G") config.G = std::stod(value);
            else if (key == "J3xy") config.J3xy = std::stod(value);
            else if (key == "J3z") config.J3z = std::stod(value);
            
            // Pyrochlore Hamiltonian
            else if (key == "Jxx") config.Jxx = std::stod(value);
            else if (key == "Jyy") config.Jyy = std::stod(value);
            else if (key == "Jzz") config.Jzz = std::stod(value);
            else if (key == "Jpm") config.Jpm = std::stod(value);
            else if (key == "Jpmpm") config.Jpmpm = std::stod(value);
            else if (key == "Jxz") config.Jxz = std::stod(value);
            else if (key == "gxx") config.gxx = std::stod(value);
            else if (key == "gyy") config.gyy = std::stod(value);
            else if (key == "gzz") config.gzz = std::stod(value);
            else if (key == "theta_rot") config.theta_rot = std::stod(value);
            
            // TmFeO3
            else if (key == "chi_Tm") config.chi_Tm = std::stod(value);
            
            // Initialization
            else if (key == "initial_config") config.initial_config = value;
            else if (key == "initial_config_file") config.initial_config_file = value;
            else if (key == "ferro_direction") config.ferro_direction = parse_vector3(value);
            
            // Unknown parameters go to generic map
            else {
                config.params[key] = std::stod(value);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line " << line_num << " in " << filename << ": " << e.what() << endl;
            std::cerr << "Line content: " << line << endl;
            throw;
        }
    }
    
    file.close();
    return config;
}

inline void SimulationConfig::to_file(const string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot write config file: " + filename);
    }
    
    file << "# Simulation Configuration File\n";
    file << "# Generated automatically\n\n";
    
    file << "# Geometry\n";
    file << "geometry = ";
    switch (geometry) {
        case GeometryType::HONEYCOMB_SU2: file << "honeycomb_su2"; break;
        case GeometryType::PYROCHLORE_SU2: file << "pyrochlore_su2"; break;
        case GeometryType::KAGOME_SU2: file << "kagome_su2"; break;
        case GeometryType::TRIANGULAR_SU2: file << "triangular_su2"; break;
        case GeometryType::SQUARE_SU2: file << "square_su2"; break;
        case GeometryType::CUSTOM_SU2: file << "custom_su2"; break;
        case GeometryType::TMFEO3_MIXED: file << "tmfeo3_mixed"; break;
    }
    file << "\n";
    file << "lattice_size = " << lattice_size[0] << "," << lattice_size[1] << "," << lattice_size[2] << "\n";
    file << "spin_dim = " << spin_dim << "\n";
    file << "spin_length = " << spin_length << "\n\n";
    
    file << "# Simulation Method\n";
    file << "method = ";
    switch (method) {
        case SimulationMethod::SIMULATED_ANNEALING: file << "simulated_annealing"; break;
        case SimulationMethod::MOLECULAR_DYNAMICS: file << "molecular_dynamics"; break;
        case SimulationMethod::PARALLEL_TEMPERING: file << "parallel_tempering"; break;
        case SimulationMethod::PUMP_PROBE: file << "pump_probe"; break;
        case SimulationMethod::TWO_PULSE_2DCS: file << "two_pulse_2dcs"; break;
        case SimulationMethod::FIELD_SCAN: file << "field_scan"; break;
        case SimulationMethod::PARAMETER_SCAN: file << "parameter_scan"; break;
    }
    file << "\n";
    file << "output_dir = " << output_dir << "\n\n";
    
    file << "# Temperature Parameters\n";
    file << "T_start = " << T_start << "\n";
    file << "T_end = " << T_end << "\n";
    file << "annealing_steps = " << annealing_steps << "\n";
    file << "cooling_rate = " << cooling_rate << "\n\n";
    
    file << "# Field Parameters\n";
    file << "field_strength = " << field_strength << "\n";
    file << "field_direction = " << field_direction[0] << "," << field_direction[1] << "," << field_direction[2] << "\n\n";
    
    if (!params.empty()) {
        file << "# Additional Parameters\n";
        for (const auto& [key, value] : params) {
            file << key << " = " << value << "\n";
        }
    }
    
    file.close();
}

inline bool SimulationConfig::validate() const {
    bool valid = true;
    
    if (num_trials < 1) {
        std::cerr << "Error: num_trials must be >= 1\n";
        valid = false;
    }
    
    if (T_start < T_end) {
        std::cerr << "Error: T_start must be >= T_end\n";
        valid = false;
    }
    
    if (lattice_size[0] == 0 || lattice_size[1] == 0 || lattice_size[2] == 0) {
        std::cerr << "Error: lattice_size dimensions must be > 0\n";
        valid = false;
    }
    
    return valid;
}

inline void SimulationConfig::print() const {
    cout << "==================== Simulation Configuration ====================\n";
    cout << "Geometry: ";
    switch (geometry) {
        case GeometryType::HONEYCOMB_SU2: cout << "Honeycomb (SU2)"; break;
        case GeometryType::PYROCHLORE_SU2: cout << "Pyrochlore (SU2)"; break;
        case GeometryType::KAGOME_SU2: cout << "Kagome (SU2)"; break;
        case GeometryType::TRIANGULAR_SU2: cout << "Triangular (SU2)"; break;
        case GeometryType::SQUARE_SU2: cout << "Square (SU2)"; break;
        case GeometryType::CUSTOM_SU2: cout << "Custom (SU2)"; break;
        case GeometryType::TMFEO3_MIXED: cout << "TmFeO3 (Mixed)"; break;
    }
    cout << "\n";
    cout << "Lattice size: " << lattice_size[0] << " x " << lattice_size[1] << " x " << lattice_size[2] << "\n";
    cout << "Spin dimension: " << spin_dim << ", Length: " << spin_length << "\n";
    cout << "Method: ";
    switch (method) {
        case SimulationMethod::SIMULATED_ANNEALING: cout << "Simulated Annealing"; break;
        case SimulationMethod::MOLECULAR_DYNAMICS: cout << "Molecular Dynamics"; break;
        case SimulationMethod::PARALLEL_TEMPERING: cout << "Parallel Tempering"; break;
        case SimulationMethod::PUMP_PROBE: cout << "Pump-Probe"; break;
        case SimulationMethod::TWO_PULSE_2DCS: cout << "Two-Pulse 2DCS"; break;
        case SimulationMethod::FIELD_SCAN: cout << "Field Scan"; break;
        case SimulationMethod::PARAMETER_SCAN: cout << "Parameter Scan"; break;
    }
    cout << "\n";
    cout << "Output: " << output_dir << "\n";
    cout << "Temperature: " << T_start << " -> " << T_end << "\n";
    cout << "Field: " << field_strength << " along [" 
         << field_direction[0] << "," << field_direction[1] << "," << field_direction[2] << "]\n";
    cout << "================================================================\n";
}

#endif // SIMULATION_CONFIG_H
