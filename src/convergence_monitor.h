#ifndef CONVERGENCE_MONITOR_H
#define CONVERGENCE_MONITOR_H

#include <vector>
#include <deque>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

// Forward declaration instead of including mixed_lattice.h
template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
struct mixed_lattice_spin;


/**
 * Convergence monitoring utilities for simulated annealing in mixed lattice systems
 */

struct ConvergenceMetrics {
    double energy_variance;
    double energy_mean;
    double config_change;
    double acceptance_rate;
    bool energy_converged;
    bool acceptance_converged;
    bool config_converged;
    
    bool is_fully_converged() const {
        return energy_converged && acceptance_converged && config_converged;
    }
};

class ConvergenceMonitor {
private:
    std::vector<double> energy_history;
    std::deque<double> acceptance_history;
    std::vector<double> config_change_history;
    
    // Parameters
    size_t energy_window = 1000;
    size_t acceptance_window = 500;
    size_t config_window = 100;
    
    double energy_variance_tol = 1e-8;
    double acceptance_target = 0.44;
    double acceptance_tolerance = 0.05;
    double config_change_tol = 1e-6;
    
public:
    
    void set_tolerances(double energy_tol, double accept_tol, double config_tol) {
        energy_variance_tol = energy_tol;
        acceptance_tolerance = accept_tol;
        config_change_tol = config_tol;
    }
    
    void record_energy(double energy) {
        energy_history.push_back(energy);
        // Keep only recent history to avoid memory issues
        if (energy_history.size() > 10000) {
            energy_history.erase(energy_history.begin(), energy_history.begin() + 1000);
        }
    }
    
    void record_acceptance(double rate) {
        acceptance_history.push_back(rate);
        if (acceptance_history.size() > acceptance_window) {
            acceptance_history.pop_front();
        }
    }
    
    void record_config_change(double change) {
        config_change_history.push_back(change);
        if (config_change_history.size() > config_window) {
            config_change_history.erase(config_change_history.begin());
        }
    }
    
    bool check_energy_convergence() const {
        if (energy_history.size() < energy_window) return false;
        
        auto recent_start = energy_history.end() - energy_window;
        double mean = std::accumulate(recent_start, energy_history.end(), 0.0) / energy_window;
        
        double variance = 0.0;
        for (auto it = recent_start; it != energy_history.end(); ++it) {
            variance += (*it - mean) * (*it - mean);
        }
        variance /= (energy_window - 1);
        
        return variance < energy_variance_tol;
    }
    
    bool check_acceptance_convergence() const {
        if (acceptance_history.size() < acceptance_window) return false;
        
        double mean = std::accumulate(acceptance_history.begin(), acceptance_history.end(), 0.0) 
                     / acceptance_history.size();
        
        return std::abs(mean - acceptance_target) < acceptance_tolerance;
    }
    
    bool check_config_convergence() const {
        if (config_change_history.size() < config_window) return false;
        
        double mean_change = std::accumulate(config_change_history.begin(), 
                                           config_change_history.end(), 0.0) 
                           / config_change_history.size();
        
        return mean_change < config_change_tol;
    }
    
    ConvergenceMetrics get_metrics() const {
        ConvergenceMetrics metrics{};
        
        // Energy metrics
        if (energy_history.size() >= energy_window) {
            auto recent_start = energy_history.end() - energy_window;
            metrics.energy_mean = std::accumulate(recent_start, energy_history.end(), 0.0) / energy_window;
            
            double variance = 0.0;
            for (auto it = recent_start; it != energy_history.end(); ++it) {
                variance += (*it - metrics.energy_mean) * (*it - metrics.energy_mean);
            }
            metrics.energy_variance = variance / (energy_window - 1);
        }
        
        // Acceptance metrics
        if (!acceptance_history.empty()) {
            metrics.acceptance_rate = std::accumulate(acceptance_history.begin(), 
                                                    acceptance_history.end(), 0.0) 
                                    / acceptance_history.size();
        }
        
        // Configuration change metrics
        if (!config_change_history.empty()) {
            metrics.config_change = std::accumulate(config_change_history.begin(), 
                                                   config_change_history.end(), 0.0) 
                                  / config_change_history.size();
        }
        
        // Convergence flags
        metrics.energy_converged = check_energy_convergence();
        metrics.acceptance_converged = check_acceptance_convergence();
        metrics.config_converged = check_config_convergence();
        
        return metrics;
    }
    
    void print_status(double temperature, size_t step) const {
        auto metrics = get_metrics();
        
        std::cout << "Step: " << step << " T: " << temperature << std::endl;
        std::cout << "  Energy: " << metrics.energy_mean 
                  << " (var: " << metrics.energy_variance << ")" << std::endl;
        std::cout << "  Acceptance: " << metrics.acceptance_rate << std::endl;
        std::cout << "  Config change: " << metrics.config_change << std::endl;
        std::cout << "  Converged: E=" << (metrics.energy_converged ? "Y" : "N")
                  << " A=" << (metrics.acceptance_converged ? "Y" : "N")
                  << " C=" << (metrics.config_converged ? "Y" : "N") << std::endl;
    }
    
    void clear_history() {
        energy_history.clear();
        acceptance_history.clear();
        config_change_history.clear();
    }
    
    void reset_for_new_temperature() {
        // Keep only recent history to maintain continuity but reset convergence state
        if (energy_history.size() > 200) {
            std::vector<double> recent_energy(energy_history.end() - 200, energy_history.end());
            energy_history = std::move(recent_energy);
        }
        
        if (acceptance_history.size() > 100) {
            std::deque<double> recent_acceptance(acceptance_history.end() - 100, acceptance_history.end());
            acceptance_history = std::move(recent_acceptance);
        }
        
        if (config_change_history.size() > 50) {
            std::vector<double> recent_config(config_change_history.end() - 50, config_change_history.end());
            config_change_history = std::move(recent_config);
        }
    }
    
    // Save convergence data to file
    void save_convergence_data(const std::string& filename) const {
        std::ofstream file(filename);
        file << "# Step Energy Acceptance ConfigChange\n";
        
        size_t min_size = std::min({energy_history.size(), 
                                   acceptance_history.size(), 
                                   config_change_history.size()});
        
        for (size_t i = 0; i < min_size; ++i) {
            file << i << " " 
                 << energy_history[energy_history.size() - min_size + i] << " "
                 << acceptance_history[acceptance_history.size() - min_size + i] << " "
                 << config_change_history[config_change_history.size() - min_size + i] << "\n";
        }
    }
};

// Utility function to calculate configuration change between two mixed lattice spins
template<size_t N_SU2, size_t lattice_size_SU2, size_t N_SU3, size_t lattice_size_SU3>
double calculate_configuration_change(
    const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3>& old_config,
    const mixed_lattice_spin<N_SU2, lattice_size_SU2, N_SU3, lattice_size_SU3>& new_config,
    double spin_length_SU2 = 1.0,
    double spin_length_SU3 = 1.0) {
    
    double total_squared_change = 0.0;
    size_t total_spins = 0;
    
    // SU2 spins
    for (size_t i = 0; i < lattice_size_SU2; ++i) {
        double squared_diff = 0.0;
        
        for (size_t j = 0; j < N_SU2; ++j) {
            double diff = new_config.spins_SU2[i][j] - old_config.spins_SU2[i][j];
            squared_diff += diff * diff;
        }

        total_squared_change += squared_diff / (spin_length_SU2 * spin_length_SU2);
        total_spins++;
    }
    
    // SU3 spins
    for (size_t i = 0; i < lattice_size_SU3; ++i) {
        double squared_diff = 0.0;
        
        for (size_t j = 0; j < N_SU3; ++j) {
            double diff = new_config.spins_SU3[i][j] - old_config.spins_SU3[i][j];
            squared_diff += diff * diff;
        }
        total_squared_change += squared_diff / (spin_length_SU3 * spin_length_SU3);
        total_spins++;
    }
    
    // Return RMS of normalized changes
    return total_spins > 0 ? std::sqrt(total_squared_change / total_spins) : 0.0;
}

#endif // CONVERGENCE_MONITOR_H
