/**
 * unified_config.cpp - Implementation of configuration parsing
 * 
 * This file contains the implementation of the UnifiedConfig class methods
 * that were previously inline in the header file. Separating them reduces
 * compile times by avoiding recompilation when only the config changes.
 */

#include "unified_config.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

UnifiedConfig UnifiedConfig::from_file(const string& filename) {
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
            else if (key == "simulation_mode") {
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
            else if (key == "gaussian_move") {
                config.gaussian_move = parse_bool(value);
            }
            else if (key == "save_observables") {
                config.save_observables = parse_bool(value);
            }
            else if (key == "deterministic") {
                config.deterministic = parse_bool(value);
            }
            else if (key == "T_zero") {
                config.T_zero = parse_bool(value);
            }
            else if (key == "n_deterministics") {
                config.n_deterministics = stoull(value);
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
            else if (key == "md_save_interval") {
                config.md_save_interval = stoull(value);
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
            else if (key == "sweep_parameter") {
                config.sweep_parameter = value;
                // For backward compatibility, also populate new arrays
                if (!value.empty() && config.sweep_parameters.empty()) {
                    config.sweep_parameters.push_back(value);
                }
            }
            else if (key == "sweep_start") {
                config.sweep_start = stod(value);
                if (config.sweep_starts.empty()) {
                    config.sweep_starts.push_back(stod(value));
                }
            }
            else if (key == "sweep_end") {
                config.sweep_end = stod(value);
                if (config.sweep_ends.empty()) {
                    config.sweep_ends.push_back(stod(value));
                }
            }
            else if (key == "sweep_step") {
                config.sweep_step = stod(value);
                if (config.sweep_steps.empty()) {
                    config.sweep_steps.push_back(stod(value));
                }
            }
            else if (key == "sweep_parameters") {
                config.sweep_parameters = parse_string_list(value);
            }
            else if (key == "sweep_starts") {
                config.sweep_starts = parse_double_list(value);
            }
            else if (key == "sweep_ends") {
                config.sweep_ends = parse_double_list(value);
            }
            else if (key == "sweep_steps") {
                config.sweep_steps = parse_double_list(value);
            }
            else if (key == "sweep_base_simulation") {
                config.sweep_base_simulation = parse_simulation(value);
            }
            else if (key == "field_strength" || key == "h") {
                config.field_strength = stod(value);
            }
            else if (key == "field_direction") {
                config.field_direction = parse_vector3(value);
                double norm = std::sqrt(
                    config.field_direction[0]*config.field_direction[0] +
                    config.field_direction[1]*config.field_direction[1] +
                    config.field_direction[2]*config.field_direction[2]
                );
                for (auto& comp : config.field_direction) {
                    comp /= norm;
                }
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

void UnifiedConfig::to_file(const string& filename) const {
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
        case SimulationType::TWOD_COHERENT_SPECTROSCOPY: file << "2dcs"; break;
        case SimulationType::PARAMETER_SWEEP: file << "parameter_sweep"; break;
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
    
    file << "# Parameter Sweep Parameters\n";
    file << "sweep_parameter = " << sweep_parameter << "\n";
    file << "sweep_start = " << sweep_start << "\n";
    file << "sweep_end = " << sweep_end << "\n";
    file << "sweep_step = " << sweep_step << "\n\n";
    
    file << "# Field Parameters\n";
    file << "field_strength = " << field_strength << "\n";
    file << "field_direction = " << field_direction[0] << "," << field_direction[1] << "," << field_direction[2] << "\n\n";
    
    file << "# Hamiltonian Parameters\n";
    for (const auto& [key, value] : hamiltonian_params) {
        file << key << " = " << value << "\n";
    }
    
    file.close();
}

bool UnifiedConfig::validate() const {
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

void UnifiedConfig::print() const {
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
        case SimulationType::TWOD_COHERENT_SPECTROSCOPY: cout << "2DCS Spectroscopy"; break;
        case SimulationType::PARAMETER_SWEEP: cout << "Parameter Sweep"; break;
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
