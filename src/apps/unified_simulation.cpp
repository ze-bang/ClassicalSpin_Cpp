#include "classical_spin/core/unified_config.h"
#include "classical_spin/core/unitcell.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/lattice.h"
#include "classical_spin/lattice/mixed_lattice.h"
#include <mpi.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <filesystem>
#include <fstream>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using namespace std;

// ============================================================================
// SIMULATION RUNNERS
// ============================================================================

/**
 * Run simulated annealing
 */
void run_simulated_annealing(Lattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running simulated annealing..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize spins for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        lattice.simulated_annealing(
            config.T_start,
            config.T_end,
            config.annealing_steps,
            config.overrelaxation_rate,
            config.use_twist_boundary,
            config.gaussian_move,
            config.cooling_rate,
            trial_dir,
            config.save_observables,
            config.T_zero,
            config.n_deterministics
        );
        
        // Save final configuration
        lattice.save_positions(trial_dir + "/positions.txt");
        lattice.save_spin_config(trial_dir + "/spins.txt");
        
        if (rank == 0) {
            ofstream energy_file(trial_dir + "/final_energy.txt");
            energy_file << "Energy Density: " << lattice.energy_density() << "\n";
            energy_file.close();
            
            cout << "Trial " << trial << " completed. Final energy: " << lattice.energy_density() << endl;
        }
    }
    
    if (rank == 0) {
        cout << "Simulated annealing completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run parallel tempering
 */
void run_parallel_tempering(Lattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running parallel tempering with " << size << " replicas..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
    }
    
    // Generate temperature ladder
    vector<double> temps(size);
    for (int i = 0; i < size; ++i) {
        double log_T = log10(config.T_start) + 
                      (log10(config.T_end) - log10(config.T_start)) * i / (size - 1);
        temps[i] = pow(10, log_T);
    }
    
    for (int trial = 0; trial < config.num_trials; ++trial) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (rank == 0 && config.num_trials > 1) {
            cout << "\n=== Trial " << trial << " / " << config.num_trials << " ===" << endl;
        }
        
        // Re-initialize spins for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        lattice.parallel_tempering(
            temps,
            config.annealing_steps,
            config.annealing_steps,
            config.overrelaxation_rate,
            config.pt_exchange_frequency,
            config.probe_rate,
            trial_dir,
            config.ranks_to_write,
            config.gaussian_move
        );
        
        if (rank == 0) {
            cout << "Trial " << trial << " completed." << endl;
        }
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Parallel tempering completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run molecular dynamics
 */
void run_molecular_dynamics(Lattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running molecular dynamics..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        if (config.use_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    if (config.use_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            // Log GPU assignment for all ranks (synchronized output)
            for (int r = 0; r < size; ++r) {
                if (rank == r) {
                    cout << "[Rank " << rank << "] Assigned to GPU " << device_id 
                         << " (" << device_count << " GPU(s) available)" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (rank == 0 && config.num_trials > 1) {
            cout << "\n=== Trial " << trial << " / " << config.num_trials << " ===" << endl;
        }
        
        // Re-initialize spins for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        // First equilibrate at low temperature (skip if spins loaded from file)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Equilibrating system..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.overrelaxation_rate,
                config.use_twist_boundary,
                config.gaussian_move,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics
            );
        } else if (rank == 0) {
            cout << "Skipping equilibration (using loaded spin configuration)" << endl;
        }
        
        // Run MD
        if (rank == 0) {
            cout << "Starting MD integration..." << endl;
            cout << "Time range: " << config.md_time_start << " -> " << config.md_time_end << endl;
            cout << "Timestep: " << config.md_timestep << endl;
            cout << "Integration method: " << config.md_integrator << endl;
        }
        
        lattice.molecular_dynamics(
            config.md_time_start,
            config.md_time_end,
            config.md_timestep,
            trial_dir,
            config.md_save_interval,
            config.md_integrator,
            config.use_gpu
        );
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
        cout << "[Rank " << rank << "] Results saved to: " << trial_dir << "/trajectory.h5" << endl;
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Molecular dynamics completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run pump-probe experiment
 */
void run_pump_probe(Lattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running pump-probe simulation..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        if (config.use_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    if (config.use_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            // Log GPU assignment for all ranks (synchronized output)
            for (int r = 0; r < size; ++r) {
                if (rank == r) {
                    cout << "[Rank " << rank << "] Assigned to GPU " << device_id 
                         << " (" << device_count << " GPU(s) available)" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif
    
    // Setup pulse field directions (one per sublattice)
    auto pump_dir_norm = config.pump_direction;
    double norm = sqrt(pump_dir_norm[0]*pump_dir_norm[0] + 
                      pump_dir_norm[1]*pump_dir_norm[1] + 
                      pump_dir_norm[2]*pump_dir_norm[2]);
    if (norm > 1e-10) {
        pump_dir_norm[0] /= norm;
        pump_dir_norm[1] /= norm;
        pump_dir_norm[2] /= norm;
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize spins for each trial (except first on this rank)
        if (trial != rank) {
            lattice.init_random();
        }
        
        // First equilibrate (skip if spins loaded from file)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Equilibrating system..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.overrelaxation_rate,
                config.use_twist_boundary,
                config.gaussian_move,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics
            );
        } else if (rank == 0) {
            cout << "Skipping equilibration (using loaded spin configuration)" << endl;
        }
        
        // Setup pump pulse
        if (rank == 0) {
            cout << "Setting up pump-probe pulses..." << endl;
            cout << "Pump: t=" << config.pump_time << ", A=" << config.pump_amplitude 
                 << ", w=" << config.pump_width << ", f=" << config.pump_frequency << endl;
            cout << "Probe: t=" << config.probe_time << ", A=" << config.probe_amplitude << endl;
        }
        
        vector<Eigen::VectorXd> field_dirs;
        for (size_t i = 0; i < lattice.N_atoms; ++i) {
            Eigen::VectorXd field_dir(3);
            field_dir << pump_dir_norm[0], pump_dir_norm[1], pump_dir_norm[2];
            field_dirs.push_back(field_dir);
        }
        
        if (rank == 0) {
            cout << "Pulse direction: [" << pump_dir_norm[0] << ", " 
                 << pump_dir_norm[1] << ", " << pump_dir_norm[2] << "]" << endl;
        }
        
        // Run single pulse magnetization dynamics
        auto trajectory = lattice.single_pulse_drive(
            field_dirs,
            config.pump_time,
            config.pump_amplitude,
            config.pump_width,
            config.pump_frequency,
            config.md_time_start,
            config.md_time_end,
            config.md_timestep,
            config.md_integrator,
            config.use_gpu
        );
        
        // Save trajectory
        if (rank == 0) {
            ofstream traj_file(trial_dir + "/pump_probe_trajectory.txt");
            for (const auto& [t, mag_data] : trajectory) {
                traj_file << t << " "
                         << mag_data[0].transpose() << " "  // mag antiferro
                         << mag_data[1].transpose() << " "  // mag local
                         << mag_data[2].transpose() << "\n"; // mag global
            }
            traj_file.close();
            cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
            cout << "[Rank " << rank << "] Results saved to: " << trial_dir << "/trajectory.h5" << endl;
        }
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Pump-probe simulation completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run 2D coherent spectroscopy (2DCS) / pump-probe spectroscopy
 * This is equivalent to pump-probe with a delay time (tau) scan
 */
void run_2dcs_spectroscopy(Lattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running 2D coherent spectroscopy (2DCS)..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "Delay scan: tau = " << config.tau_start << " to " << config.tau_end 
             << " (step: " << config.tau_step << ")" << endl;
        if (config.use_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    if (config.use_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            // Log GPU assignment for all ranks (synchronized output)
            for (int r = 0; r < size; ++r) {
                if (rank == r) {
                    cout << "[Rank " << rank << "] Assigned to GPU " << device_id 
                         << " (" << device_count << " GPU(s) available)" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif
    
    // Setup pulse field directions normalization
    auto pump_dir_norm = config.pump_direction;
    double norm = sqrt(pump_dir_norm[0]*pump_dir_norm[0] + 
                      pump_dir_norm[1]*pump_dir_norm[1] + 
                      pump_dir_norm[2]*pump_dir_norm[2]);
    if (norm > 1e-10) {
        pump_dir_norm[0] /= norm;
        pump_dir_norm[1] /= norm;
        pump_dir_norm[2] /= norm;
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize spins for each trial (except first on this rank)
        if (trial != rank) {
            lattice.init_random();
        }
        
        // First equilibrate to ground state (skip if spins loaded from file)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "\n[1/3] Equilibrating to ground state..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.overrelaxation_rate,
                config.use_twist_boundary,
                config.gaussian_move,
                config.cooling_rate,
                "",
                config.save_observables,
                config.T_zero,
                config.n_deterministics
            );
        } else if (rank == 0) {
            cout << "\n[1/3] Skipping equilibration (using loaded spin configuration)" << endl;
        }
        
        // Create field directions for all sublattices
        vector<Eigen::VectorXd> field_dirs;
        for (size_t i = 0; i < lattice.N_atoms; ++i) {
            Eigen::VectorXd field_dir(3);
            field_dir << pump_dir_norm[0], pump_dir_norm[1], pump_dir_norm[2];
            field_dirs.push_back(field_dir);
        }
        
        if (rank == 0) {
            cout << "\n[2/3] Pulse configuration:" << endl;
            cout << "  Amplitude: " << config.pump_amplitude << endl;
            cout << "  Width: " << config.pump_width << endl;
            cout << "  Frequency: " << config.pump_frequency << endl;
            cout << "  Direction: [" << pump_dir_norm[0] << ", " 
                 << pump_dir_norm[1] << ", " << pump_dir_norm[2] << "]" << endl;
            cout << "\n[3/3] Running pump-probe spectroscopy scan..." << endl;
        }
        
        // Run the full 2DCS scan using lattice method
        lattice.pump_probe_spectroscopy(
            field_dirs,
            config.pump_amplitude,
            config.pump_width,
            config.pump_frequency,
            config.tau_start,
            config.tau_end,
            config.tau_step,
            config.md_time_start,
            config.md_time_end,
            config.md_timestep,
            config.T_start,
            config.T_end,
            config.annealing_steps,
            false,  // T_zero_quench
            config.overrelaxation_rate,
            trial_dir,
            config.md_integrator,
            config.use_gpu
        );
        
        cout << "[Rank " << rank << "] Trial " << trial << " 2DCS spectroscopy completed!" << endl;
        cout << "[Rank " << rank << "] Results saved to: " << trial_dir << "/pump_probe_spectroscopy.h5" << endl;
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\n2DCS spectroscopy completed (" << config.num_trials << " trials)!" << endl;
        cout << "\nTo analyze:" << endl;
        cout << "  - M0(t): Reference single-pulse response" << endl;
        cout << "  - M1(t,tau): Probe-only response at delay tau" << endl;
        cout << "  - M01(t,tau): Two-pulse response (pump + probe)" << endl;
        cout << "  - Nonlinear signal: M01 - M0 - M1" << endl;
    }
}

/**
 * Run simulated annealing for mixed lattice
 */
void run_simulated_annealing_mixed(MixedLattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running simulated annealing on mixed lattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize spins for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        lattice.simulated_annealing(
            config.T_start,
            config.T_end,
            config.annealing_steps,
            config.gaussian_move,
            config.cooling_rate,
            trial_dir,
            config.save_observables,
            config.T_zero,
            config.n_deterministics
        );
        
        // Save final configuration
        lattice.save_positions(trial_dir + "/positions.txt");
        lattice.save_spin_config(trial_dir + "/spins.txt");
        
        if (rank == 0) {
            ofstream energy_file(trial_dir + "/final_energy.txt");
            energy_file << "Energy Density: " << lattice.energy_density() << "\n";
            energy_file.close();
            
            cout << "Trial " << trial << " completed. Final energy: " << lattice.energy_density() << endl;
        }
    }
    
    if (rank == 0) {
        cout << "Simulated annealing completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run parallel tempering for mixed lattice
 */
void run_parallel_tempering_mixed(MixedLattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running parallel tempering on mixed lattice with " << size << " replicas..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
    }
    
    // Generate temperature ladder
    vector<double> temps(size);
    for (int i = 0; i < size; ++i) {
        double log_T = log10(config.T_start) + 
                      (log10(config.T_end) - log10(config.T_start)) * i / (size - 1);
        temps[i] = pow(10, log_T);
    }
    
    for (int trial = 0; trial < config.num_trials; ++trial) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (rank == 0 && config.num_trials > 1) {
            cout << "\n=== Trial " << trial << " / " << config.num_trials << " ===" << endl;
        }
        
        // Re-initialize spins for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        lattice.parallel_tempering(
            temps,
            config.annealing_steps,
            config.annealing_steps,
            config.overrelaxation_rate,
            config.pt_exchange_frequency,
            config.probe_rate,
            trial_dir,
            config.ranks_to_write,
            config.gaussian_move
        );
        
        if (rank == 0) {
            cout << "Trial " << trial << " completed." << endl;
        }
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Parallel tempering completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run molecular dynamics for mixed lattice
 */
void run_molecular_dynamics_mixed(MixedLattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running molecular dynamics on mixed lattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        if (config.use_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    if (config.use_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            // Log GPU assignment for all ranks (synchronized output)
            for (int r = 0; r < size; ++r) {
                if (rank == r) {
                    cout << "[Rank " << rank << "] Assigned to GPU " << device_id 
                         << " (" << device_count << " GPU(s) available)" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize spins for each trial (except first on this rank)
        if (trial != rank) {
            lattice.init_random();
        }
        
        // Equilibrate (skip if spins loaded from file)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Equilibrating system..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.gaussian_move,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics
            );
        } else if (rank == 0) {
            cout << "Skipping equilibration (using loaded spin configuration)" << endl;
        }
        
        // Run MD
        if (rank == 0) {
            cout << "Starting MD integration..." << endl;
            cout << "Time range: " << config.md_time_start << " -> " << config.md_time_end << endl;
            cout << "Timestep: " << config.md_timestep << endl;
            cout << "Integration method: " << config.md_integrator << endl;
        }
        
        lattice.molecular_dynamics(
            config.md_time_start,
            config.md_time_end,
            config.md_timestep,
            trial_dir,
            config.md_save_interval,
            config.md_integrator,
            config.use_gpu
        );
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
        cout << "[Rank " << rank << "] Results saved to: " << trial_dir << "/trajectory_mixed.h5" << endl;
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Molecular dynamics completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run pump-probe experiment for mixed lattice
 */
void run_pump_probe_mixed(MixedLattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running pump-probe simulation on mixed lattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        if (config.use_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    if (config.use_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            // Log GPU assignment for all ranks (synchronized output)
            for (int r = 0; r < size; ++r) {
                if (rank == r) {
                    cout << "[Rank " << rank << "] Assigned to GPU " << device_id 
                         << " (" << device_count << " GPU(s) available)" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif
    
    // Prepare pulse directions
    SpinVector pump_dir_su2(3);
    pump_dir_su2 << config.pump_direction[0], config.pump_direction[1], config.pump_direction[2];
    pump_dir_su2.normalize();
    
    // For SU3, pump direction in Gell-Mann basis (default: λ3)
    const int su3_pump_component = static_cast<int>(config.get_param("su3_pump_component", 2.0));
    SpinVector pump_dir_su3 = SpinVector::Zero(8);
    if (su3_pump_component >= 0 && su3_pump_component < 8) {
        pump_dir_su3(su3_pump_component) = 1.0;
    } else {
        pump_dir_su3(2) = 1.0;  // Default to λ3
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize spins for each trial (except first on this rank)
        if (trial != rank) {
            lattice.init_random();
        }
        
        // Equilibrate (skip if spins loaded from file)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Equilibrating system..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.gaussian_move,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics
            );
        } else if (rank == 0) {
            cout << "Skipping equilibration (using loaded spin configuration)" << endl;
        }
        
        // Setup pump field directions
        if (rank == 0) {
            cout << "Setting up pump-probe pulses..." << endl;
        }
        
        // Create field directions for SU2 (Fe) and SU3 (Tm)
        vector<SpinVector> field_dirs_su2(lattice.lattice_size_SU2);
        vector<SpinVector> field_dirs_su3(lattice.lattice_size_SU3);
        
        for (size_t i = 0; i < lattice.lattice_size_SU2; ++i) {
            field_dirs_su2[i] = pump_dir_su2;
        }
        for (size_t i = 0; i < lattice.lattice_size_SU3; ++i) {
            field_dirs_su3[i] = pump_dir_su3;
        }
        
        // Run single pulse magnetization dynamics
        if (rank == 0) {
            cout << "Running pump-probe dynamics..." << endl;
        }
        
        auto trajectory = lattice.single_pulse_drive(
            field_dirs_su2, field_dirs_su3, config.pump_time,
            config.pump_amplitude, config.pump_width, config.pump_frequency,
            config.pump_amplitude, config.pump_width, config.pump_frequency,
            config.md_time_start, config.md_time_end, config.md_timestep,
            config.md_integrator, config.use_gpu
        );
        
        // Save trajectory
        if (rank == 0) {
            ofstream traj_file(trial_dir + "/pump_probe_trajectory.txt");
            for (const auto& [t, mag_data] : trajectory) {
                traj_file << t << " "
                         << mag_data.first[0].transpose() << " "  // SU2 mag antiferro
                         << mag_data.first[1].transpose() << " "  // SU2 mag local
                         << mag_data.first[2].transpose() << " "  // SU2 mag global
                         << mag_data.second[0].transpose() << " " // SU3 mag antiferro
                         << mag_data.second[1].transpose() << " " // SU3 mag local
                         << mag_data.second[2].transpose() << "\n"; // SU3 mag global
            }
            traj_file.close();
            cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
        }
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Pump-probe simulation completed (" << config.num_trials << " trials)." << endl;
    }
}   

/**
 * Run 2D coherent spectroscopy (2DCS) for mixed lattice
 */
void run_2dcs_spectroscopy_mixed(MixedLattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running 2D coherent spectroscopy (2DCS) on mixed lattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "Delay scan: tau = " << config.tau_start << " to " << config.tau_end 
             << " (step: " << config.tau_step << ")" << endl;
        if (config.use_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    if (config.use_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            // Log GPU assignment for all ranks (synchronized output)
            for (int r = 0; r < size; ++r) {
                if (rank == r) {
                    cout << "[Rank " << rank << "] Assigned to GPU " << device_id 
                         << " (" << device_count << " GPU(s) available)" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif
    
    // Setup pulse field directions
    SpinVector pump_dir_su2(3);
    pump_dir_su2 << config.pump_direction[0], config.pump_direction[1], config.pump_direction[2];
    pump_dir_su2.normalize();
    
    // For SU3, pump direction in Gell-Mann basis (default: λ3)
    const int su3_pump_component = static_cast<int>(config.get_param("su3_pump_component", 2.0));
    SpinVector pump_dir_su3 = SpinVector::Zero(8);
    if (su3_pump_component >= 0 && su3_pump_component < 8) {
        pump_dir_su3(su3_pump_component) = 1.0;
    } else {
        pump_dir_su3(2) = 1.0;  // Default to λ3
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize spins for each trial (except first on this rank)
        if (trial != rank) {
            lattice.init_random();
        }
        
        // First equilibrate to ground state (skip if spins loaded from file)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "\n[1/3] Equilibrating to ground state..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.gaussian_move,
                config.cooling_rate,
                trial_dir,
                config.save_observables,
                config.T_zero,
                config.n_deterministics
            );
        } else if (rank == 0) {
            cout << "\n[1/3] Skipping equilibration (using loaded spin configuration)" << endl;
        }
        
        vector<SpinVector> field_dirs_su2(lattice.lattice_size_SU2);
        vector<SpinVector> field_dirs_su3(lattice.lattice_size_SU3);
        
        for (size_t i = 0; i < lattice.lattice_size_SU2; ++i) {
            field_dirs_su2[i] = pump_dir_su2;
        }
        for (size_t i = 0; i < lattice.lattice_size_SU3; ++i) {
            field_dirs_su3[i] = pump_dir_su3;
        }
        
        if (rank == 0) {
            cout << "\n[2/3] Pulse configuration:" << endl;
            cout << "  SU2 Amplitude: " << config.pump_amplitude << endl;
            cout << "  SU2 Width: " << config.pump_width << endl;
            cout << "  SU2 Frequency: " << config.pump_frequency << endl;
            cout << "  SU2 Direction: [" << pump_dir_su2.transpose() << "]" << endl;
            cout << "  SU3 Component: λ" << su3_pump_component << endl;
            cout << "\n[3/3] Running pump-probe spectroscopy scan..." << endl;
        }
        
        // Run the full 2DCS scan using mixed lattice method
        lattice.pump_probe_spectroscopy(
            field_dirs_su2,
            field_dirs_su3,
            config.pump_amplitude,
            config.pump_width,
            config.pump_frequency,
            config.pump_amplitude,
            config.pump_width,
            config.pump_frequency,
            config.tau_start,
            config.tau_end,
            config.tau_step,
            config.md_time_start,
            config.md_time_end,
            config.md_timestep,
            config.T_start,
            config.T_end,
            config.annealing_steps,
            false,  // T_zero_quench
            config.overrelaxation_rate,
            trial_dir,
            config.md_integrator,
            config.use_gpu
        );
        
        cout << "[Rank " << rank << "] Trial " << trial << " 2DCS spectroscopy completed!" << endl;
        cout << "[Rank " << rank << "] Results saved to: " << trial_dir << "/pump_probe_spectroscopy_mixed.h5" << endl;
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\n2DCS spectroscopy completed (" << config.num_trials << " trials)!" << endl;
        cout << "\nTo analyze:" << endl;
        cout << "  - M0(t): Reference single-pulse response" << endl;
        cout << "  - M1(t,tau): Probe-only response at delay tau" << endl;
        cout << "  - M01(t,tau): Two-pulse response (pump + probe)" << endl;
        cout << "  - Nonlinear signal: M01 - M0 - M1" << endl;
    }
}

/**
 * Run parameter sweep for any Hamiltonian parameter
 * Sweeps over specified parameter and runs the base simulation at each point
 */
void run_parameter_sweep(const UnifiedConfig& base_config, int rank, int size) {
    // Determine which sweep mode to use
    vector<string> params;
    vector<vector<double>> param_grids;
    
    if (!base_config.sweep_parameters.empty() && base_config.sweep_parameters.size() == base_config.sweep_starts.size() &&
        base_config.sweep_parameters.size() == base_config.sweep_ends.size() && 
        base_config.sweep_parameters.size() == base_config.sweep_steps.size()) {
        // N-dimensional sweep mode
        params = base_config.sweep_parameters;
        for (size_t p = 0; p < params.size(); ++p) {
            vector<double> grid;
            for (double val = base_config.sweep_starts[p];
                 (base_config.sweep_steps[p] > 0 ? val <= base_config.sweep_ends[p] : val >= base_config.sweep_ends[p]);
                 val += base_config.sweep_steps[p]) {
                grid.push_back(val);
            }
            param_grids.push_back(grid);
        }
    } else if (!base_config.sweep_parameter.empty()) {
        // Legacy 1D sweep mode
        params.push_back(base_config.sweep_parameter);
        vector<double> grid;
        for (double val = base_config.sweep_start;
             (base_config.sweep_step > 0 ? val <= base_config.sweep_end : val >= base_config.sweep_end);
             val += base_config.sweep_step) {
            grid.push_back(val);
        }
        param_grids.push_back(grid);
    } else {
        if (rank == 0) {
            cerr << "Error: No sweep parameters specified!" << endl;
        }
        return;
    }
    
    // Check if GPU is needed for the base simulation
    bool needs_gpu = base_config.use_gpu && (
        base_config.sweep_base_simulation == SimulationType::MOLECULAR_DYNAMICS ||
        base_config.sweep_base_simulation == SimulationType::PUMP_PROBE ||
        base_config.sweep_base_simulation == SimulationType::TWOD_COHERENT_SPECTROSCOPY
    );
    
    if (rank == 0) {
        cout << "Running " << params.size() << "D parameter sweep..." << endl;
        for (size_t p = 0; p < params.size(); ++p) {
            cout << "  Parameter " << (p+1) << ": " << params[p] 
                 << " (" << param_grids[p].size() << " points)" << endl;
        }
        cout << "Base simulation: ";
        switch (base_config.sweep_base_simulation) {
            case SimulationType::SIMULATED_ANNEALING: cout << "Simulated Annealing"; break;
            case SimulationType::PARALLEL_TEMPERING: cout << "Parallel Tempering"; break;
            case SimulationType::MOLECULAR_DYNAMICS: cout << "Molecular Dynamics"; break;
            case SimulationType::PUMP_PROBE: cout << "Pump-Probe"; break;
            case SimulationType::TWOD_COHERENT_SPECTROSCOPY: cout << "2DCS Spectroscopy"; break;
            default: cout << "Unknown"; break;
        }
        cout << endl;
        cout << "MPI ranks: " << size << endl;
        if (needs_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else if (base_config.use_gpu) {
            cout << "GPU acceleration: Not used by base simulation type" << endl;
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    // Do this ONCE before the sweep loop to avoid repeated setup
    if (needs_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            // Log GPU assignment for all ranks (synchronized output)
            for (int r = 0; r < size; ++r) {
                if (rank == r) {
                    cout << "[Rank " << rank << "] Assigned to GPU " << device_id 
                         << " (parameter sweep, " << device_count << " GPU(s) available)" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif
    
    // Generate all combinations of parameter values (Cartesian product)
    vector<vector<double>> all_sweep_points;
    function<void(size_t, vector<double>&)> generate_combinations;
    generate_combinations = [&](size_t depth, vector<double>& current) {
        if (depth == params.size()) {
            all_sweep_points.push_back(current);
            return;
        }
        for (double val : param_grids[depth]) {
            current.push_back(val);
            generate_combinations(depth + 1, current);
            current.pop_back();
        }
    };
    vector<double> current;
    generate_combinations(0, current);
    
    if (rank == 0) {
        cout << "Total sweep points: " << all_sweep_points.size() << endl;
    }
    
    // Distribute sweep points across MPI ranks
    for (size_t i = rank; i < all_sweep_points.size(); i += size) {
        const auto& param_values = all_sweep_points[i];
        
        // Print progress
        stringstream progress;
        progress << "[Rank " << rank << "] Processing point " << i+1 << "/" << all_sweep_points.size() << ": ";
        for (size_t p = 0; p < params.size(); ++p) {
            if (p > 0) progress << ", ";
            progress << params[p] << "=" << param_values[p];
        }
        if (rank == 0 || all_sweep_points.size() > 1) {
            cout << progress.str() << endl;
        }
        
        // Create modified config for this sweep point
        UnifiedConfig sweep_config = base_config;
        sweep_config.simulation = base_config.sweep_base_simulation;
        
        // Apply all parameter values
        for (size_t p = 0; p < params.size(); ++p) {
            const string& param_name = params[p];
            double param_value = param_values[p];
            
            sweep_config.set_param(param_name, param_value);
            
            // Also check if it's a special field parameter
            if (param_name == "field_strength" || param_name == "h") {
                sweep_config.field_strength = param_value;
            }
        }
        
        // Create output directory for this sweep point
        stringstream ss;
        ss << base_config.output_dir;
        for (size_t p = 0; p < params.size(); ++p) {
            ss << "/" << params[p] << "_" << scientific << param_values[p];
        }
        sweep_config.output_dir = ss.str();
        filesystem::create_directories(sweep_config.output_dir);
        
        // Build unit cell with updated parameters
        if (sweep_config.system == SystemType::TMFEO3) {
            MixedUnitCell mixed_uc = build_tmfeo3(sweep_config);
            MixedLattice mixed_lattice(mixed_uc, sweep_config.lattice_size[0], 
                                      sweep_config.lattice_size[1], 
                                      sweep_config.lattice_size[2],
                                      sweep_config.use_twist_boundary);
            
            // Initialize spins
            if (sweep_config.use_ferromagnetic_init) {
                SpinVector dir_su2(3);
                dir_su2 << sweep_config.ferromagnetic_direction[0],
                          sweep_config.ferromagnetic_direction[1],
                          sweep_config.ferromagnetic_direction[2];
                SpinVector dir_su3 = SpinVector::Zero(8);
                const int su3_init_component = static_cast<int>(sweep_config.get_param("su3_init_component", 2.0));
                if (su3_init_component >= 0 && su3_init_component < 8) {
                    dir_su3(su3_init_component) = 1.0;
                } else {
                    dir_su3(2) = 1.0;  // Default to λ3
                }
                mixed_lattice.init_ferromagnetic(dir_su2, dir_su3);
            } else if (!sweep_config.initial_spin_config.empty()) {
                mixed_lattice.load_spin_config(sweep_config.initial_spin_config);
            }
            // else: spins already initialized randomly in constructor
            
            // Run appropriate simulation
            // Pass 0, 1 for rank/size since each MPI rank works independently on different sweep points
            // GPU is already set up at the beginning of run_parameter_sweep
            switch (sweep_config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_mixed(mixed_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering_mixed(mixed_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_mixed(mixed_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_mixed(mixed_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_spectroscopy_mixed(mixed_lattice, sweep_config, 0, 1);
                    break;
                default:
                    cerr << "[Rank " << rank << "] Error: Unsupported base simulation for parameter sweep with mixed lattice" << endl;
                    break;
            }
        } else {
            // Standard lattice systems - build unit cell based on system type
            UnitCell* uc_ptr = nullptr;
            switch (sweep_config.system) {
                case SystemType::HONEYCOMB_BCAO:
                    uc_ptr = new UnitCell(build_bcao_honeycomb(sweep_config));
                    break;
                case SystemType::HONEYCOMB_KITAEV:
                    uc_ptr = new UnitCell(build_kitaev_honeycomb(sweep_config));
                    break;
                case SystemType::PYROCHLORE:
                    uc_ptr = new UnitCell(build_pyrochlore(sweep_config));
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Error: Unknown system type for parameter sweep" << endl;
                    }
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return;
            }
            
            Lattice lattice(*uc_ptr, sweep_config.lattice_size[0], 
                          sweep_config.lattice_size[1], 
                          sweep_config.lattice_size[2],
                          sweep_config.use_twist_boundary);
            
            // Initialize spins
            if (sweep_config.use_ferromagnetic_init) {
                SpinVector dir(3);
                dir << sweep_config.ferromagnetic_direction[0],
                      sweep_config.ferromagnetic_direction[1],
                      sweep_config.ferromagnetic_direction[2];
                lattice.init_ferromagnetic(dir);
            } else if (!sweep_config.initial_spin_config.empty()) {
                lattice.load_spin_config(sweep_config.initial_spin_config);
            }
            // else: spins already initialized randomly in constructor
            
            // Run appropriate simulation
            // Pass 0, 1 for rank/size since each MPI rank works independently on different sweep points
            // GPU is already set up at the beginning of run_parameter_sweep
            switch (sweep_config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing(lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering(lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics(lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe(lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_spectroscopy(lattice, sweep_config, 0, 1);
                    break;
                default:
                    cerr << "[Rank " << rank << "] Error: Unsupported base simulation for parameter sweep" << endl;
                    break;
            }
            
            // Clean up unit cell pointer
            delete uc_ptr;
        }
        
        // Print completion message
        stringstream completion;
        completion << "[Rank " << rank << "] Completed point " << i+1 << ": ";
        for (size_t p = 0; p < params.size(); ++p) {
            if (p > 0) completion << ", ";
            completion << params[p] << "=" << param_values[p];
        }
        cout << completion.str() << endl;
    }
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Parameter sweep completed (" << all_sweep_points.size() << " points)." << endl;
    }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char** argv) {
    // Initialize MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    string config_file = "simulation.param";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    // Load configuration
    UnifiedConfig config;
    try {
        config = UnifiedConfig::from_file(config_file);
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Error loading configuration: " << e.what() << endl;
            cerr << "Usage: " << argv[0] << " [config_file]\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Validate configuration
    if (!config.validate()) {
        if (rank == 0) {
            cerr << "Configuration validation failed\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Print configuration on rank 0
    if (rank == 0) {
        config.print();
    }
    
    // Build system and run simulation
    try {
        if (config.system == SystemType::TMFEO3) {
            // Mixed SU(2)+SU(3) system
            if (rank == 0) {
                cout << "\nBuilding TmFeO3 system..." << endl;
            }
            
            auto mixed_uc = build_tmfeo3(config);
            MixedLattice mixed_lattice(mixed_uc, 
                                       config.lattice_size[0],
                                       config.lattice_size[1],
                                       config.lattice_size[2],
                                       config.spin_length,
                                       config.spin_length);
            
            // Initialize spins
            if (config.use_ferromagnetic_init) {
                SpinVector dir_su2(3);
                dir_su2 << config.ferromagnetic_direction[0],
                          config.ferromagnetic_direction[1],
                          config.ferromagnetic_direction[2];
                SpinVector dir_su3 = SpinVector::Zero(8);
                const int su3_init_component = static_cast<int>(config.get_param("su3_init_component", 2.0));
                if (su3_init_component >= 0 && su3_init_component < 8) {
                    dir_su3(su3_init_component) = 1.0;
                } else {
                    dir_su3(2) = 1.0;  // Default to λ3
                }
                mixed_lattice.init_ferromagnetic(dir_su2, dir_su3);
            } else if (!config.initial_spin_config.empty()) {
                mixed_lattice.load_spin_config(config.initial_spin_config);
            } else {
                mixed_lattice.init_random();
            }
            
            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_spectroscopy_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::PARAMETER_SWEEP:
                    run_parameter_sweep(config, rank, size);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not supported for mixed lattice\n";
                    }
                    break;
            }
        } else {
            // Regular SU(2) system
            if (rank == 0) {
                cout << "\nBuilding unit cell..." << endl;
            }
            
            UnitCell* uc_ptr = nullptr;
            switch (config.system) {
                case SystemType::HONEYCOMB_BCAO:
                    uc_ptr = new UnitCell(build_bcao_honeycomb(config));
                    break;
                case SystemType::HONEYCOMB_KITAEV:
                    uc_ptr = new UnitCell(build_kitaev_honeycomb(config));
                    break;
                case SystemType::PYROCHLORE:
                    uc_ptr = new UnitCell(build_pyrochlore(config));
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Error: Unknown system type for parameter sweep" << endl;
                    }
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
            
            Lattice lattice(*uc_ptr, 
                          config.lattice_size[0],
                          config.lattice_size[1],
                          config.lattice_size[2],
                          config.spin_length);
            
            // Initialize spins
            if (config.use_ferromagnetic_init) {
                // Initialize all spins in same direction
                SpinVector dir(3);
                dir << config.ferromagnetic_direction[0],
                      config.ferromagnetic_direction[1],
                      config.ferromagnetic_direction[2];
                dir.normalize();
                dir *= config.spin_length;
                for (size_t i = 0; i < lattice.lattice_size; ++i) {
                    lattice.spins[i] = dir;
                }
            } else if (!config.initial_spin_config.empty()) {
                lattice.load_spin_config(config.initial_spin_config);
            }
            // else: spins already initialized randomly in constructor
            
            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing(lattice, config, rank, size);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering(lattice, config, rank, size);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics(lattice, config, rank, size);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe(lattice, config, rank, size);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_spectroscopy(lattice, config, rank, size);
                    break;
                case SimulationType::PARAMETER_SWEEP:
                    run_parameter_sweep(config, rank, size);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not implemented\n";
                    }
                    break;
            }
        }
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Error during simulation: " << e.what() << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Finalize MPI
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
    
    if (rank == 0) {
        cout << "\n=== Simulation completed successfully ===" << endl;
    }
    
    return 0;
}
