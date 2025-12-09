#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/lattice.h"
#include "classical_spin/lattice/mixed_lattice.h"
#include "classical_spin/lattice/phonon_lattice.h"
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
void run_simulated_annealing(Lattice& lattice, const SpinConfig& config, int rank, int size) {
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
void run_parallel_tempering(Lattice& lattice, const SpinConfig& config, int rank, int size) {
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
void run_molecular_dynamics(Lattice& lattice, const SpinConfig& config, int rank, int size) {
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
        
        // Save initial spin configuration before time evolution
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        
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
void run_pump_probe(Lattice& lattice, const SpinConfig& config, int rank, int size) {
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
    // Normalize all pump directions
    vector<vector<double>> pump_dirs_norm = config.pump_directions;
    for (auto& dir : pump_dirs_norm) {
        double norm = 0.0;
        for (const auto& comp : dir) {
            norm += comp * comp;
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (auto& comp : dir) {
                comp /= norm;
            }
        }
    }
    
    // Validate pump direction count: must be 1 (broadcast to all) or match N_atoms
    if (pump_dirs_norm.size() != 1 && pump_dirs_norm.size() != lattice.N_atoms) {
        if (rank == 0) {
            cerr << "Error: pump_direction must have either 1 direction (broadcast to all sublattices) "
                 << "or exactly " << lattice.N_atoms << " directions (one per sublattice). "
                 << "Got " << pump_dirs_norm.size() << " directions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    // Validate pump direction dimension matches lattice spin_dim
    for (const auto& dir : pump_dirs_norm) {
        if (dir.size() != lattice.spin_dim) {
            if (rank == 0) {
                cerr << "Error: pump_direction dimension (" << dir.size() 
                     << ") does not match lattice spin_dim (" << lattice.spin_dim << ")" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
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
        
        // Save initial spin configuration before time evolution
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        
        // Setup pump pulse
        if (rank == 0) {
            cout << "Setting up pump-probe pulses..." << endl;
            cout << "Pump: t=" << config.pump_time << ", A=" << config.pump_amplitude 
                 << ", w=" << config.pump_width << ", f=" << config.pump_frequency << endl;
            cout << "Probe: t=" << config.probe_time << ", A=" << config.probe_amplitude << endl;
        }
        
        // Create field directions for each sublattice using general spin_dim
        vector<Eigen::VectorXd> field_dirs;
        for (size_t i = 0; i < lattice.N_atoms; ++i) {
            Eigen::VectorXd field_dir(lattice.spin_dim);
            // Use per-sublattice direction if provided, otherwise broadcast first direction to all
            size_t dir_idx = (pump_dirs_norm.size() == 1) ? 0 : i;
            for (size_t d = 0; d < lattice.spin_dim; ++d) {
                field_dir(d) = pump_dirs_norm[dir_idx][d];
            }
            field_dirs.push_back(field_dir);
        }
        
        if (rank == 0) {
            if (pump_dirs_norm.size() == 1) {
                cout << "Pulse direction (all sublattices): [";
                for (size_t d = 0; d < pump_dirs_norm[0].size(); ++d) {
                    if (d > 0) cout << ", ";
                    cout << pump_dirs_norm[0][d];
                }
                cout << "]" << endl;
            } else {
                cout << "Pulse directions per sublattice:" << endl;
                for (size_t i = 0; i < pump_dirs_norm.size(); ++i) {
                    cout << "  Sublattice " << i << ": [";
                    for (size_t d = 0; d < pump_dirs_norm[i].size(); ++d) {
                        if (d > 0) cout << ", ";
                        cout << pump_dirs_norm[i][d];
                    }
                    cout << "]" << endl;
                }
            }
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
void run_2dcs_spectroscopy(Lattice& lattice, const SpinConfig& config, int rank, int size) {
    // Determine parallelization strategy:
    // - If num_trials == 1 and parallel_tau is enabled: parallelize over tau (use MPI version)
    // - If num_trials > 1: parallelize over trials (original behavior)
    bool use_tau_parallel = (config.num_trials == 1) && config.parallel_tau && (size > 1);
    
    if (rank == 0) {
        cout << "Running 2D coherent spectroscopy (2DCS)..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "Delay scan: tau = " << config.tau_start << " to " << config.tau_end 
             << " (step: " << config.tau_step << ")" << endl;
        if (use_tau_parallel) {
            cout << "Parallelization mode: tau-parallel (distributing delay points across ranks)" << endl;
        } else if (config.num_trials > 1) {
            cout << "Parallelization mode: trial-parallel (distributing trials across ranks)" << endl;
        } else {
            cout << "Parallelization mode: single rank" << endl;
        }
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
    
    // Setup pulse field directions normalization (per-sublattice)
    vector<vector<double>> pump_dirs_norm = config.pump_directions;
    for (auto& dir : pump_dirs_norm) {
        double norm = 0.0;
        for (const auto& comp : dir) {
            norm += comp * comp;
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (auto& comp : dir) {
                comp /= norm;
            }
        }
    }
    
    // Validate pump direction count: must be 1 (broadcast to all) or match N_atoms
    if (pump_dirs_norm.size() != 1 && pump_dirs_norm.size() != lattice.N_atoms) {
        if (rank == 0) {
            cerr << "Error: pump_direction must have either 1 direction (broadcast to all sublattices) "
                 << "or exactly " << lattice.N_atoms << " directions (one per sublattice). "
                 << "Got " << pump_dirs_norm.size() << " directions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    // Validate pump direction dimension matches lattice spin_dim
    for (const auto& dir : pump_dirs_norm) {
        if (dir.size() != lattice.spin_dim) {
            if (rank == 0) {
                cerr << "Error: pump_direction dimension (" << dir.size() 
                     << ") does not match lattice spin_dim (" << lattice.spin_dim << ")" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    }
    
    // Create field directions for all sublattices using general spin_dim
    vector<Eigen::VectorXd> field_dirs;
    for (size_t i = 0; i < lattice.N_atoms; ++i) {
        Eigen::VectorXd field_dir(lattice.spin_dim);
        // Use per-sublattice direction if provided, otherwise broadcast first direction to all
        size_t dir_idx = (pump_dirs_norm.size() == 1) ? 0 : i;
        for (size_t d = 0; d < lattice.spin_dim; ++d) {
            field_dir(d) = pump_dirs_norm[dir_idx][d];
        }
        field_dirs.push_back(field_dir);
    }
    
    if (use_tau_parallel) {
        // Single trial, parallelize over tau values
        string trial_dir = config.output_dir + "/sample_0";
        filesystem::create_directories(trial_dir);
        
        // Equilibrate to ground state (only rank 0)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "\n[1/2] Equilibrating to ground state..." << endl;
                lattice.simulated_annealing(
                    config.T_start,
                    config.T_end,
                    config.annealing_steps,
                    config.overrelaxation_rate,
                    config.use_twist_boundary,
                    config.gaussian_move,
                    config.cooling_rate,
                    "",
                    false,  // Don't save observables during equilibration
                    config.T_zero,
                    config.n_deterministics
                );
            }
        } else if (rank == 0) {
            cout << "\n[1/2] Skipping equilibration (using loaded spin configuration)" << endl;
        }
        
        // Wait for rank 0 to finish annealing
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Synchronize spins across all ranks to ensure consistent ground state
        // Broadcast spin configuration from rank 0 to all ranks
        vector<double> spin_buffer(lattice.lattice_size * lattice.spin_dim);
        if (rank == 0) {
            for (size_t i = 0; i < lattice.lattice_size; ++i) {
                for (size_t d = 0; d < lattice.spin_dim; ++d) {
                    spin_buffer[i * lattice.spin_dim + d] = lattice.spins[i](d);
                }
            }
        }
        MPI_Bcast(spin_buffer.data(), spin_buffer.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            for (size_t i = 0; i < lattice.lattice_size; ++i) {
                for (size_t d = 0; d < lattice.spin_dim; ++d) {
                    lattice.spins[i](d) = spin_buffer[i * lattice.spin_dim + d];
                }
            }
        }
        
        // Save initial spin configuration before time evolution (rank 0 only)
        if (rank == 0) {
            lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        }
        
        if (rank == 0) {
            cout << "\n[2/2] Running MPI-parallel pump-probe spectroscopy..." << endl;
            cout << "  Pulse amplitude: " << config.pump_amplitude << endl;
            cout << "  Pulse width: " << config.pump_width << endl;
            cout << "  Pulse frequency: " << config.pump_frequency << endl;
            if (pump_dirs_norm.size() == 1) {
                cout << "  Direction: [" << pump_dirs_norm[0][0] << ", " 
                     << pump_dirs_norm[0][1] << ", " << pump_dirs_norm[0][2] << "]" << endl;
            } else {
                cout << "  Directions per sublattice:" << endl;
                for (size_t i = 0; i < pump_dirs_norm.size(); ++i) {
                    cout << "    Sublattice " << i << ": [" << pump_dirs_norm[i][0] << ", " 
                         << pump_dirs_norm[i][1] << ", " << pump_dirs_norm[i][2] << "]" << endl;
                }
            }
        }
        
        // Run MPI-parallelized version
        lattice.pump_probe_spectroscopy_mpi(
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
        
    } else {
        // Original behavior: distribute trials across MPI ranks
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
                if (rank == 0 || config.num_trials == 1) {
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
            
            // Save initial spin configuration before time evolution
            lattice.save_spin_config(trial_dir + "/initial_spins.txt");
            
            if (rank == 0 || config.num_trials == 1) {
                cout << "\n[2/3] Pulse configuration:" << endl;
                cout << "  Amplitude: " << config.pump_amplitude << endl;
                cout << "  Width: " << config.pump_width << endl;
                cout << "  Frequency: " << config.pump_frequency << endl;
                if (pump_dirs_norm.size() == 1) {
                    cout << "  Direction: [" << pump_dirs_norm[0][0] << ", " 
                         << pump_dirs_norm[0][1] << ", " << pump_dirs_norm[0][2] << "]" << endl;
                } else {
                    cout << "  Directions per sublattice:" << endl;
                    for (size_t i = 0; i < pump_dirs_norm.size(); ++i) {
                        cout << "    Sublattice " << i << ": [" << pump_dirs_norm[i][0] << ", " 
                             << pump_dirs_norm[i][1] << ", " << pump_dirs_norm[i][2] << "]" << endl;
                    }
                }
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

// ============================================================================
// PhononLattice (SPIN-PHONON) SIMULATION RUNNERS
// ============================================================================

/**
 * Build PhononLattice parameters from SpinConfig
 * Uses Kitaev-Heisenberg-Γ-Γ' model for spin interactions
 */
void build_phonon_params(const SpinConfig& config, 
                         SpinPhononCouplingParams& sp_params,
                         PhononParams& ph_params,
                         DriveParams& dr_params) {
    // Kitaev-Heisenberg-Γ-Γ' spin interaction parameters (Songvilay defaults)
    sp_params.J = config.get_param("J", -0.1);
    sp_params.K = config.get_param("K", -9.0);
    sp_params.Gamma = config.get_param("Gamma", 1.8);
    sp_params.Gammap = config.get_param("Gammap", 0.3);
    
    // 2nd neighbor (J2) - isotropic Heisenberg, sublattice-dependent
    sp_params.J2_A = config.get_param("J2_A", 0.3);
    sp_params.J2_B = config.get_param("J2_B", 0.3);
    
    // 3rd neighbor (J3) - isotropic Heisenberg
    sp_params.J3 = config.get_param("J3", 0.9);
    
    // Spin-phonon coupling strengths
    sp_params.lambda_E1 = config.get_param("lambda_E1", config.get_param("lambda_xy", 0.0));
    sp_params.lambda_E2 = config.get_param("lambda_E2", 0.0);
    sp_params.lambda_A1 = config.get_param("lambda_A1", config.get_param("lambda_R", 0.0));
    
    // E1 Phonon parameters
    ph_params.omega_E1 = config.get_param("omega_E1", config.get_param("omega_E", 1.0));
    ph_params.gamma_E1 = config.get_param("gamma_E1", config.get_param("gamma_E", 0.1));
    ph_params.lambda_E1 = config.get_param("lambda_E1_quartic", config.get_param("lambda_E", 0.0));
    
    // E2 Phonon parameters (Raman active, not directly THz driven)
    ph_params.omega_E2 = config.get_param("omega_E2", 0.8);
    ph_params.gamma_E2 = config.get_param("gamma_E2", 0.1);
    ph_params.lambda_E2 = config.get_param("lambda_E2_quartic", 0.0);
    
    // A1 Phonon parameters
    ph_params.omega_A1 = config.get_param("omega_A1", config.get_param("omega_A", 0.5));
    ph_params.gamma_A1 = config.get_param("gamma_A1", config.get_param("gamma_A", 0.05));
    ph_params.lambda_A1 = config.get_param("lambda_A1_quartic", config.get_param("lambda_A", 0.0));
    
    // Three-phonon coupling
    ph_params.g3_E1A1 = config.get_param("g3_E1A1", config.get_param("g3", 0.0));
    ph_params.g3_E2A1 = config.get_param("g3_E2A1", 0.0);
    ph_params.g3_E1E2 = config.get_param("g3_E1E2", 0.0);  // E1-E2 bilinear coupling
    ph_params.Z_star = config.get_param("Z_star", 1.0);  // Effective charge (E1 is IR active)
    
    // Drive parameters (pulse 1 - pump) - only drives E1 mode
    dr_params.E0_1 = config.pump_amplitude;
    dr_params.omega_1 = config.pump_frequency > 0 ? config.pump_frequency : ph_params.omega_E1;
    dr_params.t_1 = config.pump_time;
    dr_params.sigma_1 = config.pump_width;
    dr_params.phi_1 = config.get_param("pump_phase", 0.0);
    dr_params.theta_1 = config.get_param("pump_polarization", 0.0);
    
    // Drive parameters (pulse 2 - probe) - only drives E1 mode
    dr_params.E0_2 = config.probe_amplitude;
    dr_params.omega_2 = config.probe_frequency > 0 ? config.probe_frequency : ph_params.omega_E1;
    dr_params.t_2 = config.probe_time;
    dr_params.sigma_2 = config.probe_width;
    dr_params.phi_2 = config.get_param("probe_phase", 0.0);
    dr_params.theta_2 = config.get_param("probe_polarization", 0.0);
}

/**
 * Run simulated annealing for PhononLattice (spin subsystem only)
 */
void run_simulated_annealing_phonon(PhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running simulated annealing on PhononLattice (spin subsystem)..." << endl;
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
            config.cooling_rate,
            trial_dir,
            config.save_observables,
            config.T_zero,
            config.n_deterministics,
            config.adiabatic_phonons
        );
        
        // Save final configuration
        lattice.save_positions(trial_dir + "/positions.txt");
        lattice.save_spin_config(trial_dir + "/spins.txt");
        lattice.print_state();
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "PhononLattice simulated annealing completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run molecular dynamics for PhononLattice (full spin-phonon dynamics)
 */
void run_molecular_dynamics_phonon(PhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running spin-phonon molecular dynamics on PhononLattice..." << endl;
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
        
        // First equilibrate at low temperature (spin subsystem only)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Equilibrating spin subsystem..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.overrelaxation_rate,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics,
                config.adiabatic_phonons
            );
        } else {
            lattice.load_spin_config(config.initial_spin_config);
        }
        
        // Relax phonons to equilibrium for the current spin configuration
        // This finds the joint spin-phonon equilibrium before time evolution
        // Skip if adiabatic_phonons was used (phonons already relaxed during SA) and relax_phonons is false
        if (config.relax_phonons || config.adiabatic_phonons) {
            if (rank == 0) {
                if (config.phonon_only_relax) {
                    cout << "Relaxing phonons only (spins fixed)..." << endl;
                } else {
                    cout << "Relaxing spins and phonons to joint equilibrium..." << endl;
                }
            }
            lattice.relax_joint(1e-6, 100, 10, config.phonon_only_relax);
        } else {
            if (rank == 0) {
                cout << "Skipping phonon relaxation (relax_phonons = false)" << endl;
            }
        }
        
        // Save initial spin configuration before time evolution
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        
        // Run spin-phonon MD
        if (rank == 0) {
            cout << "Starting spin-phonon dynamics..." << endl;
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
            config.md_integrator
        );
        
        lattice.print_state();
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "PhononLattice spin-phonon dynamics completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run pump-probe for PhononLattice (THz driving IR phonon)
 */
void run_pump_probe_phonon(PhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running THz pump-probe on PhononLattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "\nDrive parameters:" << endl;
        cout << "  Pump amplitude: " << config.pump_amplitude << endl;
        cout << "  Pump frequency: " << config.pump_frequency << endl;
        cout << "  Pump time: " << config.pump_time << endl;
        cout << "  Pump width: " << config.pump_width << endl;
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        // Equilibrate spin subsystem
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Equilibrating spin subsystem..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.overrelaxation_rate,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics,
                config.adiabatic_phonons
            );
        } else {
            lattice.load_spin_config(config.initial_spin_config);
        }
        lattice.set_ordering_pattern();
        // Relax spins and phonons to joint equilibrium
        // Skip if adiabatic_phonons was used (phonons already relaxed during SA) and relax_phonons is false
        if (config.relax_phonons || config.adiabatic_phonons) {
            if (rank == 0) {
                if (config.phonon_only_relax) {
                    cout << "Relaxing phonons only (spins fixed)..." << endl;
                } else {
                    cout << "Relaxing spins and phonons to joint equilibrium..." << endl;
                }
            }
            lattice.relax_joint(1e-6, 100, 10, config.phonon_only_relax);
        } else {
            if (rank == 0) {
                cout << "Skipping phonon relaxation (relax_phonons = false)" << endl;
            }
        }
        
        // Save initial configuration
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        
        // Run spin-phonon dynamics with THz drive
        if (rank == 0) {
            cout << "Starting THz-driven spin-phonon dynamics..." << endl;
        }
        
        lattice.molecular_dynamics(
            config.md_time_start,
            config.md_time_end,
            config.md_timestep,
            trial_dir,
            config.md_save_interval,
            config.md_integrator
        );
        
        lattice.print_state();
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "PhononLattice THz pump-probe completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run 2D coherent spectroscopy for PhononLattice
 * Full pump-probe delay scan with nonlinear signal extraction
 */
void run_2dcs_phonon(PhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    // Get 2DCS-specific parameters
    double pulse_amp = config.pump_amplitude;
    double pulse_width = config.pump_width;
    double pulse_freq = config.pump_frequency > 0 ? config.pump_frequency 
                        : config.get_param("omega_E", 1.0);
    double pulse_theta = config.get_param("pulse_polarization", 0.0);
    
    double tau_start = config.get_param("tau_start", 0.0);
    double tau_end = config.get_param("tau_end", 100.0);
    double tau_step = config.get_param("tau_step", 5.0);
    
    if (rank == 0) {
        cout << "\n===========================================" << endl;
        cout << "2D Coherent Spectroscopy on PhononLattice" << endl;
        cout << "===========================================" << endl;
        cout << "Pulse parameters:" << endl;
        cout << "  Amplitude: " << pulse_amp << endl;
        cout << "  Width: " << pulse_width << endl;
        cout << "  Frequency: " << pulse_freq << endl;
        cout << "  Polarization: " << pulse_theta << " rad" << endl;
        cout << "Delay scan: " << tau_start << " → " << tau_end 
             << " (step: " << tau_step << ")" << endl;
        cout << "Time evolution: " << config.md_time_start << " → " 
             << config.md_time_end << " (step: " << config.md_timestep << ")" << endl;
    }
    
    // Distribute trials across MPI ranks (typically 1 trial for 2DCS)
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // Re-initialize for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        // Equilibrate spin subsystem if no initial config provided
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "\nEquilibrating spin subsystem via simulated annealing..." << endl;
            }
            lattice.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.overrelaxation_rate,
                config.cooling_rate,
                "",
                false,
                config.T_zero,
                config.n_deterministics,
                config.adiabatic_phonons
            );
        } else {
            lattice.load_spin_config(config.initial_spin_config);
        }
        
        // Relax spins and phonons to joint equilibrium before 2DCS
        // Skip if adiabatic_phonons was used (phonons already relaxed during SA) and relax_phonons is false
        if (config.relax_phonons || config.adiabatic_phonons) {
            if (rank == 0) {
                if (config.phonon_only_relax) {
                    cout << "Relaxing phonons only (spins fixed)..." << endl;
                } else {
                    cout << "Relaxing spins and phonons to joint equilibrium..." << endl;
                }
            }
            lattice.relax_joint(1e-6, 100, 10, config.phonon_only_relax);
        } else {
            if (rank == 0) {
                cout << "Skipping phonon relaxation (relax_phonons = false)" << endl;
            }
        }
        
        // Run 2DCS workflow
        if (size > 1) {
            // Use MPI-parallel version
            lattice.pump_probe_spectroscopy_mpi(
                pulse_theta,
                pulse_amp, pulse_width, pulse_freq,
                tau_start, tau_end, tau_step,
                config.md_time_start, config.md_time_end, config.md_timestep,
                trial_dir, config.md_integrator
            );
        } else {
            // Single-rank version
            lattice.pump_probe_spectroscopy(
                pulse_theta,
                pulse_amp, pulse_width, pulse_freq,
                tau_start, tau_end, tau_step,
                config.md_time_start, config.md_time_end, config.md_timestep,
                trial_dir, config.md_integrator
            );
        }
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\n===========================================" << endl;
        cout << "PhononLattice 2DCS completed!" << endl;
        cout << "===========================================" << endl;
    }
}

/**
 * Run simulated annealing for mixed lattice
 */
void run_simulated_annealing_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size) {
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
void run_parallel_tempering_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size) {
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
void run_molecular_dynamics_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size) {
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
        
        // Save initial spin configuration before time evolution
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        
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
void run_pump_probe_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size) {
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
    
    // Prepare per-sublattice pulse directions for SU2
    // Normalize all pump directions
    vector<vector<double>> pump_dirs_norm = config.pump_directions;
    for (auto& dir : pump_dirs_norm) {
        double norm = 0.0;
        for (const auto& comp : dir) {
            norm += comp * comp;
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (auto& comp : dir) {
                comp /= norm;
            }
        }
    }
    
    // Validate pump direction count: must be 1 (broadcast to all) or match N_atoms_SU2
    if (pump_dirs_norm.size() != 1 && pump_dirs_norm.size() != lattice.N_atoms_SU2) {
        if (rank == 0) {
            cerr << "Error: pump_direction must have either 1 direction (broadcast to all SU2 sublattices) "
                 << "or exactly " << lattice.N_atoms_SU2 << " directions (one per SU2 sublattice). "
                 << "Got " << pump_dirs_norm.size() << " directions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    // Validate pump direction dimension matches lattice spin_dim_SU2
    for (const auto& dir : pump_dirs_norm) {
        if (dir.size() != lattice.spin_dim_SU2) {
            if (rank == 0) {
                cerr << "Error: pump_direction dimension (" << dir.size() 
                     << ") does not match lattice spin_dim_SU2 (" << lattice.spin_dim_SU2 << ")" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    }
    
    // Prepare per-sublattice pulse directions for SU3 (Gell-Mann basis)
    // Normalize all SU3 pump directions
    vector<vector<double>> pump_dirs_su3_norm = config.pump_directions_su3;
    for (auto& dir : pump_dirs_su3_norm) {
        double norm = 0.0;
        for (const auto& comp : dir) {
            norm += comp * comp;
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (auto& comp : dir) {
                comp /= norm;
            }
        }
    }
    
    // Validate SU3 pump direction count: must be 1 (broadcast to all) or match N_atoms_SU3
    if (pump_dirs_su3_norm.size() != 1 && pump_dirs_su3_norm.size() != lattice.N_atoms_SU3) {
        if (rank == 0) {
            cerr << "Error: pump_direction_su3 must have either 1 direction (broadcast to all SU3 sublattices) "
                 << "or exactly " << lattice.N_atoms_SU3 << " directions (one per SU3 sublattice). "
                 << "Got " << pump_dirs_su3_norm.size() << " directions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    // Validate SU3 pump direction dimension matches lattice spin_dim_SU3
    for (const auto& dir : pump_dirs_su3_norm) {
        if (dir.size() != lattice.spin_dim_SU3) {
            if (rank == 0) {
                cerr << "Error: pump_direction_su3 dimension (" << dir.size() 
                     << ") does not match lattice spin_dim_SU3 (" << lattice.spin_dim_SU3 << ")" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
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
        
        // Save initial spin configuration before time evolution
        lattice.save_spin_config_to_dir(trial_dir, "initial_spins");
        
        // Setup pump field directions
        if (rank == 0) {
            cout << "Setting up pump-probe pulses..." << endl;
        }
        
        // Create field directions for SU2 (Fe) - using per-sublattice directions and general spin_dim
        vector<SpinVector> field_dirs_su2(lattice.lattice_size_SU2);
        vector<SpinVector> field_dirs_su3(lattice.lattice_size_SU3);
        
        for (size_t i = 0; i < lattice.lattice_size_SU2; ++i) {
            size_t atom = i % lattice.N_atoms_SU2;
            // Use per-sublattice direction if provided, otherwise broadcast first direction to all
            size_t dir_idx = (pump_dirs_norm.size() == 1) ? 0 : atom;
            SpinVector pump_dir_su2(lattice.spin_dim_SU2);
            for (size_t d = 0; d < lattice.spin_dim_SU2; ++d) {
                pump_dir_su2(d) = pump_dirs_norm[dir_idx][d];
            }
            field_dirs_su2[i] = pump_dir_su2;
        }
        for (size_t i = 0; i < lattice.lattice_size_SU3; ++i) {
            size_t atom = i % lattice.N_atoms_SU3;
            // Use per-sublattice direction if provided, otherwise broadcast first direction to all
            size_t dir_idx = (pump_dirs_su3_norm.size() == 1) ? 0 : atom;
            SpinVector pump_dir_su3(lattice.spin_dim_SU3);
            for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
                pump_dir_su3(d) = pump_dirs_su3_norm[dir_idx][d];
            }
            field_dirs_su3[i] = pump_dir_su3;
        }
        
        // Run single pulse magnetization dynamics
        if (rank == 0) {
            cout << "Running pump-probe dynamics..." << endl;
        }
        
        auto trajectory = lattice.single_pulse_drive(
            field_dirs_su2, field_dirs_su3, config.pump_time,
            config.pump_amplitude, config.pump_width, config.pump_frequency,
            config.pump_amplitude_su3, config.pump_width_su3, config.pump_frequency_su3,
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
void run_2dcs_spectroscopy_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size) {
    // Determine parallelization strategy
    bool use_tau_parallel = (config.num_trials == 1) && config.parallel_tau && (size > 1);
    
    if (rank == 0) {
        cout << "Running 2D coherent spectroscopy (2DCS) on mixed lattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "Delay scan: tau = " << config.tau_start << " to " << config.tau_end 
             << " (step: " << config.tau_step << ")" << endl;
        if (use_tau_parallel) {
            cout << "Parallelization mode: tau-parallel (distributing delay points across ranks)" << endl;
        } else if (config.num_trials > 1) {
            cout << "Parallelization mode: trial-parallel (distributing trials across ranks)" << endl;
        } else {
            cout << "Parallelization mode: single rank" << endl;
        }
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
    
    // Setup per-sublattice pulse field directions for SU2
    vector<vector<double>> pump_dirs_norm = config.pump_directions;
    for (auto& dir : pump_dirs_norm) {
        double norm = 0.0;
        for (const auto& comp : dir) {
            norm += comp * comp;
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (auto& comp : dir) {
                comp /= norm;
            }
        }
    }
    
    // Validate pump direction count: must be 1 (broadcast to all) or match N_atoms_SU2
    if (pump_dirs_norm.size() != 1 && pump_dirs_norm.size() != lattice.N_atoms_SU2) {
        if (rank == 0) {
            cerr << "Error: pump_direction must have either 1 direction (broadcast to all SU2 sublattices) "
                 << "or exactly " << lattice.N_atoms_SU2 << " directions (one per SU2 sublattice). "
                 << "Got " << pump_dirs_norm.size() << " directions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    // Validate pump direction dimension matches lattice spin_dim_SU2
    for (const auto& dir : pump_dirs_norm) {
        if (dir.size() != lattice.spin_dim_SU2) {
            if (rank == 0) {
                cerr << "Error: pump_direction dimension (" << dir.size() 
                     << ") does not match lattice spin_dim_SU2 (" << lattice.spin_dim_SU2 << ")" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    }
    
    // Prepare per-sublattice pulse directions for SU3 (Gell-Mann basis)
    // Normalize all SU3 pump directions
    vector<vector<double>> pump_dirs_su3_norm = config.pump_directions_su3;
    for (auto& dir : pump_dirs_su3_norm) {
        double norm = 0.0;
        for (const auto& comp : dir) {
            norm += comp * comp;
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (auto& comp : dir) {
                comp /= norm;
            }
        }
    }
    
    // Validate SU3 pump direction count: must be 1 (broadcast to all) or match N_atoms_SU3
    if (pump_dirs_su3_norm.size() != 1 && pump_dirs_su3_norm.size() != lattice.N_atoms_SU3) {
        if (rank == 0) {
            cerr << "Error: pump_direction_su3 must have either 1 direction (broadcast to all SU3 sublattices) "
                 << "or exactly " << lattice.N_atoms_SU3 << " directions (one per SU3 sublattice). "
                 << "Got " << pump_dirs_su3_norm.size() << " directions." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }
    
    // Validate SU3 pump direction dimension matches lattice spin_dim_SU3
    for (const auto& dir : pump_dirs_su3_norm) {
        if (dir.size() != lattice.spin_dim_SU3) {
            if (rank == 0) {
                cerr << "Error: pump_direction_su3 dimension (" << dir.size() 
                     << ") does not match lattice spin_dim_SU3 (" << lattice.spin_dim_SU3 << ")" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    }
    
    // Create field directions for all sublattices using general spin_dim
    vector<SpinVector> field_dirs_su2(lattice.lattice_size_SU2);
    vector<SpinVector> field_dirs_su3(lattice.lattice_size_SU3);
    
    for (size_t i = 0; i < lattice.lattice_size_SU2; ++i) {
        size_t atom = i % lattice.N_atoms_SU2;
        // Use per-sublattice direction if provided, otherwise broadcast first direction to all
        size_t dir_idx = (pump_dirs_norm.size() == 1) ? 0 : atom;
        SpinVector pump_dir_su2(lattice.spin_dim_SU2);
        for (size_t d = 0; d < lattice.spin_dim_SU2; ++d) {
            pump_dir_su2(d) = pump_dirs_norm[dir_idx][d];
        }
        field_dirs_su2[i] = pump_dir_su2;
    }
    for (size_t i = 0; i < lattice.lattice_size_SU3; ++i) {
        size_t atom = i % lattice.N_atoms_SU3;
        // Use per-sublattice direction if provided, otherwise broadcast first direction to all
        size_t dir_idx = (pump_dirs_su3_norm.size() == 1) ? 0 : atom;
        SpinVector pump_dir_su3(lattice.spin_dim_SU3);
        for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
            pump_dir_su3(d) = pump_dirs_su3_norm[dir_idx][d];
        }
        field_dirs_su3[i] = pump_dir_su3;
    }
    
    if (use_tau_parallel) {
        // Single trial, parallelize over tau values
        string trial_dir = config.output_dir + "/sample_0";
        filesystem::create_directories(trial_dir);
        
        // Equilibrate to ground state (only rank 0)
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "\n[1/2] Equilibrating to ground state..." << endl;
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
            }
        } else if (rank == 0) {
            cout << "\n[1/2] Skipping equilibration (using loaded spin configuration)" << endl;
        }
        
        // Wait for rank 0 to finish annealing
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Synchronize SU2 spins across all ranks
        vector<double> spin_buffer_su2(lattice.lattice_size_SU2 * lattice.spin_dim_SU2);
        if (rank == 0) {
            for (size_t i = 0; i < lattice.lattice_size_SU2; ++i) {
                for (size_t d = 0; d < lattice.spin_dim_SU2; ++d) {
                    spin_buffer_su2[i * lattice.spin_dim_SU2 + d] = lattice.spins_SU2[i](d);
                }
            }
        }
        MPI_Bcast(spin_buffer_su2.data(), spin_buffer_su2.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            for (size_t i = 0; i < lattice.lattice_size_SU2; ++i) {
                for (size_t d = 0; d < lattice.spin_dim_SU2; ++d) {
                    lattice.spins_SU2[i](d) = spin_buffer_su2[i * lattice.spin_dim_SU2 + d];
                }
            }
        }
        
        // Synchronize SU3 spins across all ranks
        vector<double> spin_buffer_su3(lattice.lattice_size_SU3 * lattice.spin_dim_SU3);
        if (rank == 0) {
            for (size_t i = 0; i < lattice.lattice_size_SU3; ++i) {
                for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
                    spin_buffer_su3[i * lattice.spin_dim_SU3 + d] = lattice.spins_SU3[i](d);
                }
            }
        }
        MPI_Bcast(spin_buffer_su3.data(), spin_buffer_su3.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            for (size_t i = 0; i < lattice.lattice_size_SU3; ++i) {
                for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
                    lattice.spins_SU3[i](d) = spin_buffer_su3[i * lattice.spin_dim_SU3 + d];
                }
            }
        }
        
        // Save initial spin configuration before time evolution (rank 0 only)
        if (rank == 0) {
            lattice.save_spin_config_to_dir(trial_dir, "initial_spins");
        }
        
        if (rank == 0) {
            cout << "\n[2/2] Running MPI-parallel pump-probe spectroscopy..." << endl;
            cout << "  SU2 Pulse: amplitude=" << config.pump_amplitude 
                 << ", width=" << config.pump_width 
                 << ", frequency=" << config.pump_frequency << endl;
            if (pump_dirs_norm.size() == 1) {
                cout << "  SU2 Direction: [" << pump_dirs_norm[0][0] << ", " 
                     << pump_dirs_norm[0][1] << ", " << pump_dirs_norm[0][2] << "]" << endl;
            } else {
                cout << "  SU2 Directions per sublattice:" << endl;
                for (size_t i = 0; i < pump_dirs_norm.size(); ++i) {
                    cout << "    Sublattice " << i << ": [" << pump_dirs_norm[i][0] << ", " 
                         << pump_dirs_norm[i][1] << ", " << pump_dirs_norm[i][2] << "]" << endl;
                }
            }
            cout << "  SU3 Pulse: amplitude=" << config.pump_amplitude_su3 
                 << ", width=" << config.pump_width_su3 
                 << ", frequency=" << config.pump_frequency_su3 << endl;
            if (pump_dirs_su3_norm.size() == 1) {
                cout << "  SU3 Direction (Gell-Mann): [";
                for (int k = 0; k < 8; ++k) {
                    if (k > 0) cout << ", ";
                    cout << pump_dirs_su3_norm[0][k];
                }
                cout << "]" << endl;
            } else {
                cout << "  SU3 Directions per sublattice (Gell-Mann):" << endl;
                for (size_t i = 0; i < pump_dirs_su3_norm.size(); ++i) {
                    cout << "    Sublattice " << i << ": [";
                    for (int k = 0; k < 8; ++k) {
                        if (k > 0) cout << ", ";
                        cout << pump_dirs_su3_norm[i][k];
                    }
                    cout << "]" << endl;
                }
            }
        }
        
        // Run MPI-parallelized version
        lattice.pump_probe_spectroscopy_mpi(
            field_dirs_su2,
            field_dirs_su3,
            config.pump_amplitude,
            config.pump_width,
            config.pump_frequency,
            config.pump_amplitude_su3,
            config.pump_width_su3,
            config.pump_frequency_su3,
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
        
    } else {
        // Original behavior: distribute trials across MPI ranks
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
                if (rank == 0 || config.num_trials == 1) {
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
            
            // Save initial spin configuration before time evolution
            lattice.save_spin_config_to_dir(trial_dir, "initial_spins");
            
            if (rank == 0 || config.num_trials == 1) {
                cout << "\n[2/3] Pulse configuration:" << endl;
                cout << "  SU2 Pulse: amplitude=" << config.pump_amplitude 
                     << ", width=" << config.pump_width 
                     << ", frequency=" << config.pump_frequency << endl;
                if (pump_dirs_norm.size() == 1) {
                    cout << "  SU2 Direction: [" << pump_dirs_norm[0][0] << ", " 
                         << pump_dirs_norm[0][1] << ", " << pump_dirs_norm[0][2] << "]" << endl;
                } else {
                    cout << "  SU2 Directions per sublattice:" << endl;
                    for (size_t i = 0; i < pump_dirs_norm.size(); ++i) {
                        cout << "    Sublattice " << i << ": [" << pump_dirs_norm[i][0] << ", " 
                             << pump_dirs_norm[i][1] << ", " << pump_dirs_norm[i][2] << "]" << endl;
                    }
                }
                cout << "  SU3 Pulse: amplitude=" << config.pump_amplitude_su3 
                     << ", width=" << config.pump_width_su3 
                     << ", frequency=" << config.pump_frequency_su3 << endl;
                if (pump_dirs_su3_norm.size() == 1) {
                    cout << "  SU3 Direction (Gell-Mann): [";
                    for (int k = 0; k < 8; ++k) {
                        if (k > 0) cout << ", ";
                        cout << pump_dirs_su3_norm[0][k];
                    }
                    cout << "]" << endl;
                } else {
                    cout << "  SU3 Directions per sublattice (Gell-Mann):" << endl;
                    for (size_t i = 0; i < pump_dirs_su3_norm.size(); ++i) {
                        cout << "    Sublattice " << i << ": [";
                        for (int k = 0; k < 8; ++k) {
                            if (k > 0) cout << ", ";
                            cout << pump_dirs_su3_norm[i][k];
                        }
                        cout << "]" << endl;
                    }
                }
                cout << "\n[3/3] Running pump-probe spectroscopy scan..." << endl;
            }
            
            // Run the full 2DCS scan using mixed lattice method
            lattice.pump_probe_spectroscopy(
                field_dirs_su2,
                field_dirs_su3,
                config.pump_amplitude,
                config.pump_width,
                config.pump_frequency,
                config.pump_amplitude_su3,
                config.pump_width_su3,
                config.pump_frequency_su3,
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
void run_parameter_sweep(const SpinConfig& base_config, int rank, int size) {
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
        SpinConfig sweep_config = base_config;
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
                // Create SU2 direction from config (use general dimension)
                SpinVector dir_su2(mixed_lattice.spin_dim_SU2);
                for (size_t d = 0; d < mixed_lattice.spin_dim_SU2; ++d) {
                    dir_su2(d) = (d < sweep_config.ferromagnetic_direction.size()) ? 
                                 sweep_config.ferromagnetic_direction[d] : 0.0;
                }
                // Create SU3 direction
                SpinVector dir_su3 = SpinVector::Zero(mixed_lattice.spin_dim_SU3);
                const int su3_init_component = static_cast<int>(sweep_config.get_param("su3_init_component", 2.0));
                if (su3_init_component >= 0 && su3_init_component < static_cast<int>(mixed_lattice.spin_dim_SU3)) {
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
                case SystemType::TMFEO3_FE:
                    uc_ptr = new UnitCell(build_tmfeo3_fe(sweep_config));
                    break;
                case SystemType::TMFEO3_TM:
                    uc_ptr = new UnitCell(build_tmfeo3_tm(sweep_config));
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
                // Initialize all spins in same direction (use general spin_dim)
                SpinVector dir(lattice.spin_dim);
                for (size_t d = 0; d < lattice.spin_dim; ++d) {
                    dir(d) = (d < sweep_config.ferromagnetic_direction.size()) ? 
                             sweep_config.ferromagnetic_direction[d] : 0.0;
                }
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
    SpinConfig config;
    try {
        config = SpinConfig::from_file(config_file);
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
        if (config.system == SystemType::NCTO) {
            // PhononLattice spin-phonon coupled system (honeycomb)
            if (rank == 0) {
                cout << "\nBuilding PhononLattice spin-phonon lattice..." << endl;
            }
            
            PhononLattice phonon_lattice(config.lattice_size[0],
                                         config.lattice_size[1],
                                         config.lattice_size[2],
                                         config.spin_length);
            
            // Build parameters from config
            SpinPhononCouplingParams sp_params;
            PhononParams ph_params;
            DriveParams dr_params;
            build_phonon_params(config, sp_params, ph_params, dr_params);
            
            // Set parameters (this builds the interaction matrices)
            phonon_lattice.set_parameters(sp_params, ph_params, dr_params);
            
            // Set Gilbert damping if specified
            phonon_lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.0);
            
            // Set magnetic field
            Eigen::Vector3d B;
            B << config.field_strength * config.field_direction[0],
                 config.field_strength * config.field_direction[1],
                 config.field_strength * config.field_direction[2];
            phonon_lattice.set_field(B);
            
            // Initialize spins
            if (config.use_ferromagnetic_init) {
                Eigen::Vector3d dir;
                dir << config.ferromagnetic_direction[0],
                       config.ferromagnetic_direction[1],
                       config.ferromagnetic_direction[2];
                phonon_lattice.init_ferromagnetic(dir);
            } else if (!config.initial_spin_config.empty()) {
                phonon_lattice.load_spin_config(config.initial_spin_config);
            } else {
                phonon_lattice.init_random();
            }
            
            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_phonon(phonon_lattice, config, rank, size);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_phonon(phonon_lattice, config, rank, size);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_phonon(phonon_lattice, config, rank, size);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_phonon(phonon_lattice, config, rank, size);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not supported for PhononLattice. "
                             << "Supported: SA, MD, pump_probe, 2dcs" << endl;
                    }
                    break;
            }
        } else if (config.system == SystemType::TMFEO3) {
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
                                       config.spin_length_su3);
            
            // Initialize spins
            if (config.use_ferromagnetic_init) {
                // Create SU2 direction from config (use general dimension)
                SpinVector dir_su2(mixed_lattice.spin_dim_SU2);
                for (size_t d = 0; d < mixed_lattice.spin_dim_SU2; ++d) {
                    dir_su2(d) = (d < config.ferromagnetic_direction.size()) ? 
                                 config.ferromagnetic_direction[d] : 0.0;
                }
                // Create SU3 direction
                SpinVector dir_su3 = SpinVector::Zero(mixed_lattice.spin_dim_SU3);
                const int su3_init_component = static_cast<int>(config.get_param("su3_init_component", 2.0));
                if (su3_init_component >= 0 && su3_init_component < static_cast<int>(mixed_lattice.spin_dim_SU3)) {
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
                case SystemType::TMFEO3_FE:
                    uc_ptr = new UnitCell(build_tmfeo3_fe(config));
                    break;
                case SystemType::TMFEO3_TM:
                    uc_ptr = new UnitCell(build_tmfeo3_tm(config));
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
                // Initialize all spins in same direction (use general spin_dim)
                SpinVector dir(lattice.spin_dim);
                for (size_t d = 0; d < lattice.spin_dim; ++d) {
                    dir(d) = (d < config.ferromagnetic_direction.size()) ? 
                             config.ferromagnetic_direction[d] : 0.0;
                }
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
