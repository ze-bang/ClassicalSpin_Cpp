#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/core/gneb.h"
#include "classical_spin/core/gneb_strain.h"
#include "classical_spin/lattice/lattice.h"
#include "classical_spin/lattice/mixed_lattice.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "classical_spin/lattice/strain_phonon_lattice.h"
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
            config.n_deterministics,
            config.twist_sweep_count
        );
        
        // Save final configuration
        lattice.save_positions(trial_dir + "/positions.txt");
        // lattice.save_spin_config(trial_dir + "/spins.txt");
        
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
 * @param comm MPI communicator to use (default: MPI_COMM_WORLD)
 */
void run_parallel_tempering(Lattice& lattice, const SpinConfig& config, int rank, int size, MPI_Comm comm = MPI_COMM_WORLD) {
    if (rank == 0) {
        cout << "Running parallel tempering with " << size << " replicas..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
    }
    
    // Generate temperature ladder
    vector<double> temps(size);
    
    if (config.pt_optimize_temperatures) {
        // Use MPI-distributed feedback-optimized temperature grid (Bittner et al.)
        // All ranks participate - much faster than single-rank optimization
        if (rank == 0) {
            cout << "Generating optimized temperature grid (Bittner et al., MPI-distributed)..." << endl;
        }
        
        OptimizedTempGridResult opt_result = lattice.generate_optimized_temperature_grid_mpi(
            config.T_end,    // Tmin (coldest)
            config.T_start,  // Tmax (hottest)
            config.pt_optimization_warmup,
            config.pt_optimization_sweeps,
            config.pt_optimization_iterations,
            config.gaussian_move,
            config.overrelaxation_rate,
            config.pt_target_acceptance,
            0.05,  // convergence tolerance
            comm
        );
        temps = opt_result.temperatures;
        
        // Save optimized temperature grid info to file (rank 0 only)
        if (rank == 0 && !config.output_dir.empty()) {
            filesystem::create_directories(config.output_dir);
            ofstream opt_file(config.output_dir + "/optimized_temperatures.txt");
            opt_file << "# Optimized temperature grid (Bittner et al., Phys. Rev. Lett. 101, 130603)\n";
            opt_file << "# Target acceptance rate: " << config.pt_target_acceptance << "\n";
            opt_file << "# Mean acceptance rate: " << opt_result.mean_acceptance_rate << "\n";
            opt_file << "# Converged: " << (opt_result.converged ? "yes" : "no") << "\n";
            opt_file << "# Feedback iterations: " << opt_result.feedback_iterations_used << "\n";
            opt_file << "# Round-trip estimate: " << opt_result.round_trip_estimate << "\n";
            opt_file << "#\n";
            opt_file << "# rank  temperature  acceptance_rate  diffusivity\n";
            for (int i = 0; i < size; ++i) {
                opt_file << i << "  " << scientific << setprecision(12) << temps[i];
                if (i < size - 1) {
                    opt_file << "  " << fixed << setprecision(4) << opt_result.acceptance_rates[i]
                             << "  " << scientific << setprecision(6) << opt_result.local_diffusivities[i];
                }
                opt_file << "\n";
            }
            opt_file.close();
        }
    } else {
        // Use geometric (logarithmic) temperature spacing
        if (rank == 0) {
            cout << "Using geometric temperature grid..." << endl;
            temps = Lattice::generate_geometric_temperature_ladder(config.T_end, config.T_start, size);
        }
        // Broadcast temperatures from rank 0 to all ranks
        MPI_Bcast(temps.data(), size, MPI_DOUBLE, 0, comm);
    }
    
    // Re-initialize spins after temperature optimization (or geometric grid setup)
    // This ensures each rank starts with fresh random spins - the optimization
    // phase leaves spins in a "mixed" state from many replica exchanges
    lattice.init_random();
    MPI_Barrier(comm);
    
    for (int trial = 0; trial < config.num_trials; ++trial) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        if (rank == 0) {
            filesystem::create_directories(trial_dir);
        }
        MPI_Barrier(comm);  // Ensure directory is created before others proceed
        
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
            config.gaussian_move,
            comm,
            false,  // verbose
            config.pt_accumulate_correlations,
            config.pt_n_bond_types
        );
        
        // T=0 deterministic quench for coldest replica (rank 0)
        if (config.T_zero && rank == 0 && config.n_deterministics > 0) {
            cout << "Rank 0: Performing " << config.n_deterministics << " deterministic sweeps at T=0..." << endl;
            for (size_t sweep = 0; sweep < config.n_deterministics; ++sweep) {
                lattice.deterministic_sweep(1);
                if (sweep % 100 == 0 || sweep == config.n_deterministics - 1) {
                    cout << "Deterministic sweep " << sweep << "/" << config.n_deterministics 
                         << ", E/N = " << lattice.energy_density() << endl;
                }
            }
            cout << "Deterministic sweeps completed. Final energy: " << lattice.energy_density() << endl;
            // Save the T=0 quenched configuration
            lattice.save_spin_config(trial_dir + "/rank_0/spins_T0_quench.txt");
        }
        MPI_Barrier(comm);
        
        if (rank == 0) {
            cout << "Trial " << trial << " completed." << endl;
        }
    }
    
    // Synchronize all ranks
    MPI_Barrier(comm);
    
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
                config.n_deterministics,
            config.twist_sweep_count
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
                config.n_deterministics,
            config.twist_sweep_count
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
                    config.n_deterministics,
                config.twist_sweep_count
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
                    config.n_deterministics,
                config.twist_sweep_count
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
                         DriveParams& dr_params,
                         TimeDependentSpinPhononParams& td_sp_params) {
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
    
    // Six-spin ring exchange on hexagonal plaquettes
    sp_params.J7 = config.get_param("J7", 0.0);
    
    // Spin-phonon coupling strengths
    sp_params.lambda_E1 = config.get_param("lambda_E1", config.get_param("lambda_xy", 0.0));
    sp_params.lambda_E2 = config.get_param("lambda_E2", 0.0);
    sp_params.lambda_A1 = config.get_param("lambda_A1", config.get_param("lambda_R", 0.0));
    
    // Time-dependent spin-phonon coupling parameters
    // Mode: 0 = constant (default), 1 = window
    double time_mode = config.get_param("lambda_time_mode", 0.0);
    td_sp_params.mode = (time_mode > 0.5) ? "window" : "constant";
    
    // Window function parameters for E1 mode
    td_sp_params.t_start_E1 = config.get_param("lambda_E1_t_start", 0.0);
    td_sp_params.t_end_E1 = config.get_param("lambda_E1_t_end", 1e30);
    td_sp_params.lambda_E1_target = config.get_param("lambda_E1_target", sp_params.lambda_E1);
    
    // Window function parameters for E2 mode
    td_sp_params.t_start_E2 = config.get_param("lambda_E2_t_start", 0.0);
    td_sp_params.t_end_E2 = config.get_param("lambda_E2_t_end", 1e30);
    td_sp_params.lambda_E2_target = config.get_param("lambda_E2_target", sp_params.lambda_E2);
    
    // Window function parameters for A1 mode
    td_sp_params.t_start_A1 = config.get_param("lambda_A1_t_start", 0.0);
    td_sp_params.t_end_A1 = config.get_param("lambda_A1_t_end", 1e30);
    td_sp_params.lambda_A1_target = config.get_param("lambda_A1_target", sp_params.lambda_A1);
    
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
    
    // Drive strength per bond type for E1 (IR active): relative scaling of the THz field
    // Default: 1.0 (full strength). Set to 0 to disable, or any value to scale.
    dr_params.drive_strength_E1[0] = config.get_param("drive_strength_0", 1.0);  // x-bond
    dr_params.drive_strength_E1[1] = config.get_param("drive_strength_1", 1.0);  // y-bond
    dr_params.drive_strength_E1[2] = config.get_param("drive_strength_2", 1.0);  // z-bond
    
    // Drive strength per bond type for E2 (Raman active): allows artificial E2 driving
    // Default: 0.0 (E2 is not IR active). Set to non-zero to drive E2 phonon.
    dr_params.drive_strength_E2[0] = config.get_param("drive_strength_E2_0", 0.0);  // x-bond
    dr_params.drive_strength_E2[1] = config.get_param("drive_strength_E2_1", 0.0);  // y-bond
    dr_params.drive_strength_E2[2] = config.get_param("drive_strength_E2_2", 0.0);  // z-bond
}

/**
 * Build StrainPhononLattice parameters from SpinConfig
 * Uses Kitaev-Heisenberg-Γ-Γ' model with magnetoelastic (spin-strain) coupling
 * 
 * Magnetoelastic coupling (D3d point group):
 * H_c^{A1g} = λ_{A1g} Σ_r (ε_xx + ε_yy) {(J+K)f_K^{A1g} + J f_J^{A1g} + Γ f_Γ^{A1g}}
 * H_c^{Eg} = λ_{Eg} Σ_r {(ε_xx - ε_yy)[(J+K)f_K^{Eg,1} + J f_J^{Eg,1} + Γ f_Γ^{Eg,1}]
 *                       + 2ε_xy[(J+K)f_K^{Eg,2} + J f_J^{Eg,2} + Γ f_Γ^{Eg,2}]}
 */
void build_strain_params(const SpinConfig& config,
                         MagnetoelasticParams& me_params,
                         ElasticParams& el_params,
                         StrainDriveParams& dr_params) {
    // Kitaev-Heisenberg-Γ-Γ' spin interaction parameters
    me_params.J = config.get_param("J", -0.1);
    me_params.K = config.get_param("K", -9.0);
    me_params.Gamma = config.get_param("Gamma", 1.8);
    me_params.Gammap = config.get_param("Gammap", 0.3);
    
    // 2nd neighbor (J2) - isotropic Heisenberg, sublattice-dependent
    me_params.J2_A = config.get_param("J2_A", 0.3);
    me_params.J2_B = config.get_param("J2_B", 0.3);
    
    // 3rd neighbor (J3) - isotropic Heisenberg
    me_params.J3 = config.get_param("J3", 0.9);
    
    // Six-spin ring exchange on hexagonal plaquettes
    me_params.J7 = config.get_param("J7", 0.0);
    
    // Gamma parameter for time-dependent J7 modulation via E1 phonon
    // J7(t) = J7 * (1 - γ*f(t)*λ_Eg/4)^4 * (1 + γ*f(t)*λ_Eg/2)^2
    me_params.gamma_J7 = config.get_param("gamma_J7", 0.0);
    
    // Magnetoelastic coupling strengths (A1g and Eg channels)
    me_params.lambda_A1g = config.get_param("lambda_A1g", 0.0);
    me_params.lambda_Eg = config.get_param("lambda_Eg", 0.0);
    
    // Elastic constants (Voigt notation)
    // For isotropic solid: C44 = (C11 - C12) / 2
    el_params.C11 = config.get_param("C11", 1.0);
    el_params.C12 = config.get_param("C12", 0.3);
    el_params.C44 = config.get_param("C44", 0.35);
    
    // Effective mass for strain dynamics
    el_params.M = config.get_param("strain_mass", config.get_param("M", 1.0));
    
    // Damping coefficients
    el_params.gamma_A1g = config.get_param("gamma_A1g", 0.1);
    el_params.gamma_Eg = config.get_param("gamma_Eg", 0.1);
    
    // Quartic anharmonicity (optional)
    el_params.lambda_A1g = config.get_param("lambda_A1g_quartic", 0.0);
    el_params.lambda_Eg = config.get_param("lambda_Eg_quartic", 0.0);
    
    // Drive parameters (pulse 1 - pump)
    dr_params.E0_1 = config.pump_amplitude;
    dr_params.omega_1 = config.pump_frequency > 0 ? config.pump_frequency : el_params.omega_A1g();
    dr_params.t_1 = config.pump_time;
    dr_params.sigma_1 = config.pump_width;
    dr_params.phi_1 = config.get_param("pump_phase", 0.0);
    
    // Drive parameters (pulse 2 - probe)
    dr_params.E0_2 = config.probe_amplitude;
    dr_params.omega_2 = config.probe_frequency > 0 ? config.probe_frequency : el_params.omega_A1g();
    dr_params.t_2 = config.probe_time;
    dr_params.sigma_2 = config.probe_width;
    dr_params.phi_2 = config.get_param("probe_phase", 0.0);
    
    // Drive strength for different strain modes
    dr_params.drive_strength_A1g = config.get_param("drive_strength_A1g", 1.0);
    dr_params.drive_strength_Eg1 = config.get_param("drive_strength_Eg1", 0.0);
    dr_params.drive_strength_Eg2 = config.get_param("drive_strength_Eg2", 0.0);
}

/**
 * Run simulated annealing for StrainPhononLattice (spin subsystem only)
 */
void run_simulated_annealing_strain(StrainPhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running simulated annealing on StrainPhononLattice (spin subsystem)..." << endl;
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
        
        lattice.anneal(config.T_start, config.T_end, config.annealing_steps,
                       config.cooling_rate, config.overrelaxation_rate,
                       config.gaussian_move, trial_dir,
                       config.T_zero, config.n_deterministics);
        
        // Save final configuration
        lattice.save_spin_config(trial_dir + "/spins.txt");
        lattice.save_strain_state(trial_dir + "/strain_state.txt");
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "StrainPhononLattice simulated annealing completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run molecular dynamics for StrainPhononLattice (full spin-strain dynamics)
 */
void run_molecular_dynamics_strain(StrainPhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running spin-strain molecular dynamics on StrainPhononLattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        // Re-initialize spins for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        lattice.integrate_rk4(config.md_timestep, config.md_time_start, config.md_time_end, 
                              config.md_save_interval, trial_dir);
        
        // Save final configuration
        lattice.save_spin_config(trial_dir + "/spins_final.txt");
        lattice.save_strain_state(trial_dir + "/strain_state_final.txt");
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "StrainPhononLattice molecular dynamics completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run kinetic barrier analysis for StrainPhononLattice using GNEB with strain
 * 
 * This finds the minimum energy path in the COMBINED (spin, strain) configuration
 * space. The strain is a dynamical degree of freedom that relaxes along with
 * spins to find the TRUE transition pathway including lattice distortion.
 * 
 * Key physics:
 * - Configuration space is (S_1, ..., S_N, ε_Eg1, ε_Eg2)
 * - Spins live on S^2 (geodesic manifold), strain is Euclidean
 * - The MEP shows how the lattice distorts during the magnetic transition
 * - No external parameter sweep needed - strain follows the optimal path
 */
void run_kinetic_barrier_analysis_strain(StrainPhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "\n" << string(70, '=') << endl;
        cout << "KINETIC BARRIER ANALYSIS (GNEB with STRAIN)" << endl;
        cout << string(70, '=') << endl;
        cout << "Configuration space: (spins, ε_Eg1, ε_Eg2)" << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "\nGNEB parameters:" << endl;
        cout << "  N images:         " << config.gneb_n_images << endl;
        cout << "  Spring constant:  " << config.gneb_spring_constant << endl;
        cout << "  Max iterations:   " << config.gneb_max_iterations << endl;
        cout << "  Force tolerance:  " << config.gneb_force_tolerance << endl;
        cout << "  Climbing image:   " << (config.gneb_use_climbing_image ? "yes" : "no") << endl;
        cout << "  Strain weight:    1.0 (equal to spin weight)" << endl;
        cout << "\nMagnetoelastic coupling:" << endl;
        cout << "  lambda_Eg: " << lattice.magnetoelastic_params.lambda_Eg << endl;
        cout << string(70, '-') << endl;
    }
    
    // Get lattice dimensions
    const size_t n_sites = lattice.lattice_size;
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        filesystem::create_directories(trial_dir + "/gneb");
        
        if (rank == 0 || config.num_trials > 1) {
            cout << "\n[Rank " << rank << "] === Trial " << trial << " ===" << endl;
        }
        
        // ================================================================
        // STEP 1: Get initial spin state and relax strain
        // ================================================================
        SpinStrainConfig initial_state(n_sites);
        
        if (!config.gneb_initial_state_file.empty()) {
            // Load combined (spins, strain) from file
            cout << "[Rank " << rank << "] Loading initial spin+strain config from file..." << endl;
            lattice.load_spin_strain_config(config.gneb_initial_state_file);
            cout << "  File: " << config.gneb_initial_state_file << endl;
            
            // Extract to SpinStrainConfig
            for (size_t i = 0; i < n_sites; ++i) {
                initial_state.spins[i] = lattice.spins[i];
            }
            // Get Eg strain from lattice (averaging over bond types)
            double Eg1 = 0.0, Eg2 = 0.0;
            for (size_t b = 0; b < 3; ++b) {
                Eg1 += (lattice.strain.epsilon_xx[b] - lattice.strain.epsilon_yy[b]) / 2.0;
                Eg2 += lattice.strain.epsilon_xy[b];
            }
            initial_state.strain = StrainEg(Eg1 / 3.0, Eg2 / 3.0);
        } else {
            // Find via annealing (strain is relaxed during annealing)
            cout << "[Rank " << rank << "] Finding initial (triple-Q) ground state..." << endl;
            
            // Zero out strain initially
            for (size_t b = 0; b < 3; ++b) {
                lattice.strain.epsilon_xx[b] = 0.0;
                lattice.strain.epsilon_yy[b] = 0.0;
                lattice.strain.epsilon_xy[b] = 0.0;
            }
            
            lattice.init_random();
            lattice.anneal(config.T_start, config.T_end, config.annealing_steps,
                           config.cooling_rate, config.overrelaxation_rate,
                           config.gaussian_move, trial_dir + "/initial_anneal",
                           config.T_zero, config.n_deterministics);
            
            // Extract spin configuration
            for (size_t i = 0; i < n_sites; ++i) {
                initial_state.spins[i] = lattice.spins[i];
            }
            
            // Relax strain at fixed initial spins to find equilibrium strain
            cout << "[Rank " << rank << "] Relaxing strain for initial state..." << endl;
            auto [init_Eg1, init_Eg2] = lattice.relax_strain_at_fixed_spins(initial_state.spins);
            initial_state.strain = StrainEg(init_Eg1, init_Eg2);
        }
        
        // Update lattice strain to match initial_state
        for (size_t b = 0; b < 3; ++b) {
            lattice.strain.epsilon_xx[b] = initial_state.strain.Eg1;
            lattice.strain.epsilon_yy[b] = -initial_state.strain.Eg1;
            lattice.strain.epsilon_xy[b] = initial_state.strain.Eg2;
        }
        
        // Analyze initial state
        auto initial_cv = lattice.compute_collective_variables();
        cout << "[Rank " << rank << "] Initial state:" << endl;
        cout << "  m_3Q       = " << initial_cv.m_3Q << endl;
        cout << "  m_zigzag   = " << initial_cv.m_zigzag << endl;
        cout << "  f_Eg_amp   = " << initial_cv.f_Eg_amplitude << endl;
        cout << "  ε_Eg       = (" << initial_state.strain.Eg1 << ", " << initial_state.strain.Eg2 << ")" << endl;
        
        lattice.save_spin_strain_config(trial_dir + "/triple_q_state.txt");
        
        // ================================================================
        // STEP 2: Get final spin state and relax strain
        // ================================================================
        SpinStrainConfig final_state(n_sites);
        
        if (!config.gneb_final_state_file.empty()) {
            // Load combined (spins, strain) from file
            cout << "[Rank " << rank << "] Loading final spin+strain config from file..." << endl;
            lattice.load_spin_strain_config(config.gneb_final_state_file);
            cout << "  File: " << config.gneb_final_state_file << endl;
            
            // Extract to SpinStrainConfig
            for (size_t i = 0; i < n_sites; ++i) {
                final_state.spins[i] = lattice.spins[i];
            }
            // Get Eg strain from lattice (averaging over bond types)
            double Eg1 = 0.0, Eg2 = 0.0;
            for (size_t b = 0; b < 3; ++b) {
                Eg1 += (lattice.strain.epsilon_xx[b] - lattice.strain.epsilon_yy[b]) / 2.0;
                Eg2 += lattice.strain.epsilon_xy[b];
            }
            final_state.strain = StrainEg(Eg1 / 3.0, Eg2 / 3.0);
        } else {
            // Find via annealing with bias strain
            cout << "[Rank " << rank << "] Finding zigzag state with applied strain..." << endl;
            
            // Apply strong Eg strain to induce zigzag
            const double bias_strain = 3.0;
            for (size_t b = 0; b < 3; ++b) {
                lattice.strain.epsilon_xx[b] = bias_strain / 2.0;
                lattice.strain.epsilon_yy[b] = -bias_strain / 2.0;
                lattice.strain.epsilon_xy[b] = 0.0;
            }
            
            lattice.init_random();
            lattice.anneal(config.T_start, config.T_end, config.annealing_steps,
                           config.cooling_rate, config.overrelaxation_rate,
                           config.gaussian_move, trial_dir + "/zigzag_anneal",
                           config.T_zero, config.n_deterministics);
            
            // Extract spin configuration - strain will be relaxed below
            for (size_t i = 0; i < n_sites; ++i) {
                final_state.spins[i] = lattice.spins[i];
            }
            
            // Relax strain at fixed final spins (find equilibrium without bias)
            cout << "[Rank " << rank << "] Relaxing strain for final state..." << endl;
            auto [final_Eg1, final_Eg2] = lattice.relax_strain_at_fixed_spins(final_state.spins);
            final_state.strain = StrainEg(final_Eg1, final_Eg2);
        }
        
        // Update lattice strain to match final_state
        for (size_t b = 0; b < 3; ++b) {
            lattice.strain.epsilon_xx[b] = final_state.strain.Eg1;
            lattice.strain.epsilon_yy[b] = -final_state.strain.Eg1;
            lattice.strain.epsilon_xy[b] = final_state.strain.Eg2;
        }
        
        // Analyze final state
        auto final_cv = lattice.compute_collective_variables();
        cout << "[Rank " << rank << "] Final state:" << endl;
        cout << "  m_3Q       = " << final_cv.m_3Q << endl;
        cout << "  m_zigzag   = " << final_cv.m_zigzag << endl;
        cout << "  f_Eg_amp   = " << final_cv.f_Eg_amplitude << endl;
        cout << "  ε_Eg       = (" << final_state.strain.Eg1 << ", " << final_state.strain.Eg2 << ")" << endl;
        
        lattice.save_spin_strain_config(trial_dir + "/zigzag_state.txt");
        
        // ================================================================
        // STEP 3: Set up GNEB optimizer with strain
        // ================================================================
        cout << "[Rank " << rank << "] Setting up GNEB with strain optimizer..." << endl;
        
        // Energy function: E(spins, strain)
        auto energy_func = [&lattice](const SpinStrainConfig& cfg) -> double {
            return lattice.energy_for_gneb_with_strain(
                cfg.spins, cfg.strain.Eg1, cfg.strain.Eg2);
        };
        
        // Gradient function: (∂E/∂S, ∂E/∂ε)
        auto gradient_func = [&lattice](const SpinStrainConfig& cfg) -> SpinStrainGradient {
            auto [grad_spins, dE_dEg1, dE_dEg2] = lattice.gradient_for_gneb_with_strain(
                cfg.spins, cfg.strain.Eg1, cfg.strain.Eg2);
            
            SpinStrainGradient grad(cfg.spins.size());
            grad.d_spins = grad_spins;
            grad.d_strain = StrainEg(dE_dEg1, dE_dEg2);
            return grad;
        };
        
        GNEBStrainOptimizer gneb(energy_func, gradient_func, n_sites);
        
        // GNEB parameters from config
        GNEBStrainParams gneb_params;
        gneb_params.n_images = config.gneb_n_images;
        gneb_params.spring_constant = config.gneb_spring_constant;
        gneb_params.max_iterations = config.gneb_max_iterations;
        gneb_params.force_tolerance = config.gneb_force_tolerance;
        gneb_params.climbing_image = config.gneb_use_climbing_image;
        gneb_params.weight_strain = 1.0;  // Equal weight for strain and spins
        gneb_params.max_strain_amplitude = 10.0;  // Safety limit
        gneb_params.verbosity = (rank == 0) ? 2 : 0;
        
        // ================================================================
        // STEP 4: Find MEP in combined (spin, strain) space
        // ================================================================
        cout << "[Rank " << rank << "] Finding minimum energy path in (spin, strain) space..." << endl;
        
        auto mep_result = gneb.find_mep(initial_state, final_state, gneb_params);
        
        cout << "[Rank " << rank << "] MEP converged in " << mep_result.iterations_used << " iterations" << endl;
        cout << "  Barrier height: " << mep_result.barrier << endl;
        cout << "  Saddle image:   " << mep_result.saddle_index << endl;
        cout << "  Saddle strain:  ε_Eg = (" << mep_result.saddle_strain.Eg1 
             << ", " << mep_result.saddle_strain.Eg2 << ")" << endl;
        cout << "  Max strain amp: " << mep_result.max_strain_amplitude << endl;
        cout << "  Final max force:" << mep_result.max_force << endl;
        
        // ================================================================
        // STEP 5: Save MEP with strain information
        // ================================================================
        {
            ofstream mep_file(trial_dir + "/mep_with_strain.txt");
            mep_file << "# Minimum energy path in (spin, strain) space\n";
            mep_file << "# Strain coordinates: ε_Eg1 = (ε_xx - ε_yy)/2, ε_Eg2 = ε_xy\n";
            mep_file << "# image  reaction_coord  energy  strain_Eg1  strain_Eg2  strain_amp  m_3Q  m_zigzag  f_Eg_amp\n";
            
            for (size_t i = 0; i < mep_result.energies.size(); ++i) {
                // Set lattice to this configuration to compute CVs
                for (size_t j = 0; j < n_sites; ++j) {
                    lattice.spins[j] = mep_result.images[i].spins[j];
                }
                auto cv = lattice.compute_collective_variables();
                
                const auto& img = mep_result.images[i];
                mep_file << i << "  " << mep_result.arc_lengths[i] 
                         << "  " << mep_result.energies[i]
                         << "  " << img.strain.Eg1 
                         << "  " << img.strain.Eg2
                         << "  " << img.strain.amplitude()
                         << "  " << cv.m_3Q << "  " << cv.m_zigzag 
                         << "  " << cv.f_Eg_amplitude << "\n";
            }
        }
        
        // ================================================================
        // STEP 6: Save summary
        // ================================================================
        {
            ofstream summary(trial_dir + "/barrier_summary.txt");
            summary << "# Kinetic Barrier Analysis Summary (GNEB with Strain)\n";
            summary << "# Trial: " << trial << "\n";
            summary << "# Configuration space: (spins, ε_Eg1, ε_Eg2)\n";
            summary << "#\n";
            summary << "barrier = " << mep_result.barrier << "\n";
            summary << "gneb_iterations = " << mep_result.iterations_used << "\n";
            summary << "gneb_converged = " << (mep_result.converged ? "true" : "false") << "\n";
            summary << "saddle_image = " << mep_result.saddle_index << "\n";
            summary << "max_force = " << mep_result.max_force << "\n";
            summary << "#\n";
            summary << "# Strain at key points:\n";
            summary << "initial_strain_Eg1 = " << mep_result.initial_strain.Eg1 << "\n";
            summary << "initial_strain_Eg2 = " << mep_result.initial_strain.Eg2 << "\n";
            summary << "saddle_strain_Eg1 = " << mep_result.saddle_strain.Eg1 << "\n";
            summary << "saddle_strain_Eg2 = " << mep_result.saddle_strain.Eg2 << "\n";
            summary << "final_strain_Eg1 = " << mep_result.final_strain.Eg1 << "\n";
            summary << "final_strain_Eg2 = " << mep_result.final_strain.Eg2 << "\n";
            summary << "max_strain_amplitude = " << mep_result.max_strain_amplitude << "\n";
            summary << "#\n";
            summary << "# Initial (triple-Q) state:\n";
            summary << "initial_m_3Q = " << initial_cv.m_3Q << "\n";
            summary << "initial_m_zigzag = " << initial_cv.m_zigzag << "\n";
            summary << "initial_f_Eg_amp = " << initial_cv.f_Eg_amplitude << "\n";
            summary << "#\n";
            summary << "# Final (zigzag) state:\n";
            summary << "final_m_3Q = " << final_cv.m_3Q << "\n";
            summary << "final_m_zigzag = " << final_cv.m_zigzag << "\n";
            summary << "final_f_Eg_amp = " << final_cv.f_Eg_amplitude << "\n";
        }
        
        // Save MEP images if requested
        if (config.gneb_save_path_evolution) {
            gneb.save_path(trial_dir + "/gneb", "mep");
            
            // Also save in our custom format with more info
            for (size_t img = 0; img < mep_result.images.size(); ++img) {
                string img_file = trial_dir + "/gneb/image_" + to_string(img) + ".txt";
                ofstream out(img_file);
                out << "# MEP image " << img << "\n";
                out << "# strain_Eg1 = " << mep_result.images[img].strain.Eg1 << "\n";
                out << "# strain_Eg2 = " << mep_result.images[img].strain.Eg2 << "\n";
                out << "# energy = " << mep_result.energies[img] << "\n";
                out << "# site  Sx  Sy  Sz\n";
                for (size_t i = 0; i < n_sites; ++i) {
                    out << i << "  " << mep_result.images[img].spins[i].x() 
                        << "  " << mep_result.images[img].spins[i].y()
                        << "  " << mep_result.images[img].spins[i].z() << "\n";
                }
            }
        }
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
        cout << "  Output saved to: " << trial_dir << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\n" << string(70, '=') << endl;
        cout << "Kinetic barrier analysis with strain completed (" << config.num_trials << " trials)" << endl;
        cout << string(70, '=') << endl;
    }
}

/**
 * Run pump-probe for StrainPhononLattice (strain-driven spin dynamics)
 * 
 * This simulates a THz/acoustic pump-probe experiment on the magnetoelastic system:
 * 1. Equilibrate spins via simulated annealing
 * 2. Run spin-strain dynamics with acoustic drive
 * 3. Record magnetization trajectories
 */
void run_pump_probe_strain(StrainPhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running acoustic pump-probe on StrainPhononLattice..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "\nDrive parameters:" << endl;
        cout << "  Pump amplitude: " << config.pump_amplitude << endl;
        cout << "  Pump frequency: " << config.pump_frequency << endl;
        cout << "  Pump time: " << config.pump_time << endl;
        cout << "  Pump width: " << config.pump_width << endl;
        cout << "  Drive strength A1g: " << lattice.drive_params.drive_strength_A1g << endl;
        cout << "  Drive strength Eg1: " << lattice.drive_params.drive_strength_Eg1 << endl;
        cout << "  Drive strength Eg2: " << lattice.drive_params.drive_strength_Eg2 << endl;
        cout << "\nTime-dependent J7 modulation:" << endl;
        cout << "  J7 = " << lattice.magnetoelastic_params.J7 << endl;
        cout << "  gamma_J7 = " << lattice.magnetoelastic_params.gamma_J7 << endl;
        cout << "  lambda_Eg = " << lattice.magnetoelastic_params.lambda_Eg << endl;
        cout << "\nEquilibration:" << endl;
        cout << "  T_zero = " << (config.T_zero ? "true" : "false") << endl;
        cout << "  n_deterministics = " << config.n_deterministics << endl;
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
        
        // Equilibrate spin subsystem via simulated annealing
        if (config.initial_spin_config.empty()) {
            if (rank == 0) {
                cout << "Equilibrating spin subsystem via simulated annealing..." << endl;
            }
            lattice.anneal(config.T_start, config.T_end, config.annealing_steps,
                           config.cooling_rate, config.overrelaxation_rate,
                           config.gaussian_move, trial_dir,
                           config.T_zero, config.n_deterministics);
        } else {
            // Load combined spin+strain config (same format as GNEB endpoints)
            cout << "[Rank " << rank << "] Loading initial spin+strain config from: " 
                 << config.initial_spin_config << endl;
            lattice.load_spin_strain_config(config.initial_spin_config);
        }
        
        // Relax strain to equilibrium given the spin configuration
        if (rank == 0) {
            cout << "Relaxing strain to adiabatic equilibrium..." << endl;
        }
        lattice.relax_strain();
        
        // Save initial configuration (combined spin+strain format)
        lattice.save_spin_strain_config(trial_dir + "/initial_spin_strain.txt");
        lattice.save_spin_config(trial_dir + "/initial_spins.txt");
        lattice.save_strain_state(trial_dir + "/initial_strain.txt");
        
        // Run spin-strain dynamics with acoustic drive
        if (rank == 0) {
            cout << "Starting strain-driven spin dynamics..." << endl;
            cout << "  Time range: " << config.md_time_start << " to " << config.md_time_end << endl;
            cout << "  Timestep: " << config.md_timestep << endl;
        }
        
        lattice.integrate_rk4(config.md_timestep, config.md_time_start, config.md_time_end, 
                              config.md_save_interval, trial_dir);
        
        // Save final configuration (combined spin+strain format)
        lattice.save_spin_strain_config(trial_dir + "/final_spin_strain.txt");
        lattice.save_spin_config(trial_dir + "/final_spins.txt");
        lattice.save_strain_state(trial_dir + "/final_strain.txt");
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "StrainPhononLattice pump-probe completed (" << config.num_trials << " trials)." << endl;
    }
}

/**
 * Run parallel tempering for StrainPhononLattice
 * Uses MPI for replica exchange between temperatures
 */
void run_parallel_tempering_strain(StrainPhononLattice& lattice, const SpinConfig& config, int rank, int size, MPI_Comm comm = MPI_COMM_WORLD) {
    if (rank == 0) {
        cout << "Running parallel tempering on StrainPhononLattice with " << size << " replicas..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
    }
    
    // Generate temperature ladder
    vector<double> temps(size);
    
    if (config.pt_optimize_temperatures) {
        // Use MPI-distributed feedback-optimized temperature grid (Bittner et al.)
        if (rank == 0) {
            cout << "Generating optimized temperature grid (Bittner et al., MPI-distributed)..." << endl;
        }
        
        SPL_OptimizedTempGridResult opt_result = lattice.generate_optimized_temperature_grid_mpi(
            config.T_end,    // Tmin (coldest)
            config.T_start,  // Tmax (hottest)
            config.pt_optimization_warmup,
            config.pt_optimization_sweeps,
            config.pt_optimization_iterations,
            config.gaussian_move,
            config.overrelaxation_rate,
            config.pt_target_acceptance,
            0.05,  // convergence tolerance
            comm
        );
        temps = opt_result.temperatures;
        
        // Save optimized temperature grid info to file (rank 0 only)
        if (rank == 0 && !config.output_dir.empty()) {
            filesystem::create_directories(config.output_dir);
            ofstream opt_file(config.output_dir + "/optimized_temperatures.txt");
            opt_file << "# Optimized temperature grid (Bittner et al., Phys. Rev. Lett. 101, 130603)\n";
            opt_file << "# Target acceptance rate: " << config.pt_target_acceptance << "\n";
            opt_file << "# Mean acceptance rate: " << opt_result.mean_acceptance_rate << "\n";
            opt_file << "# Converged: " << (opt_result.converged ? "yes" : "no") << "\n";
            opt_file << "# Feedback iterations: " << opt_result.feedback_iterations_used << "\n";
            opt_file << "# Round-trip estimate: " << opt_result.round_trip_estimate << "\n";
            opt_file << "#\n";
            opt_file << "# rank  temperature  acceptance_rate  diffusivity\n";
            for (int i = 0; i < size; ++i) {
                opt_file << i << "  " << scientific << setprecision(12) << temps[i];
                if (i < size - 1) {
                    opt_file << "  " << fixed << setprecision(4) << opt_result.acceptance_rates[i]
                             << "  " << scientific << setprecision(6) << opt_result.local_diffusivities[i];
                }
                opt_file << "\n";
            }
            opt_file.close();
        }
    } else {
        // Use geometric (logarithmic) temperature spacing
        if (rank == 0) {
            cout << "Using geometric temperature grid..." << endl;
            temps = StrainPhononLattice::generate_geometric_temperature_ladder(config.T_end, config.T_start, size);
        }
        // Broadcast temperatures from rank 0 to all ranks
        MPI_Bcast(temps.data(), size, MPI_DOUBLE, 0, comm);
    }
    
    // Re-initialize spins after temperature optimization
    lattice.init_random();
    MPI_Barrier(comm);
    
    for (int trial = 0; trial < config.num_trials; ++trial) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        if (rank == 0) {
            filesystem::create_directories(trial_dir);
        }
        MPI_Barrier(comm);  // Ensure directory is created before others proceed
        
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
            config.gaussian_move,
            comm,
            true  // verbose: save spin configurations
        );
        
        // T=0 deterministic quench for coldest replica (rank 0)
        if (config.T_zero && rank == 0 && config.n_deterministics > 0) {
            cout << "Rank 0: Performing " << config.n_deterministics << " deterministic sweeps at T=0..." << endl;
            for (size_t sweep = 0; sweep < config.n_deterministics; ++sweep) {
                lattice.deterministic_sweep(1);
                lattice.relax_strain(false);  // Relax strain after each deterministic sweep
                if (sweep % 100 == 0 || sweep == config.n_deterministics - 1) {
                    cout << "Deterministic sweep " << sweep << "/" << config.n_deterministics 
                         << ", E/N = " << lattice.spin_energy() / lattice.lattice_size << endl;
                }
            }
            cout << "Deterministic sweeps completed. Final energy: " << lattice.spin_energy() / lattice.lattice_size << endl;
            // Save the T=0 quenched configuration
            lattice.save_spin_config(trial_dir + "/rank_0/spins_T0_quench.txt");
        }
        MPI_Barrier(comm);
        
        if (rank == 0) {
            cout << "Trial " << trial << " completed." << endl;
        }
    }
    
    // Synchronize all ranks
    MPI_Barrier(comm);
    
    if (rank == 0) {
        cout << "StrainPhononLattice parallel tempering completed (" << config.num_trials << " trials)." << endl;
    }
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
        // lattice.save_spin_config(trial_dir + "/spins.txt");
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
        cout << "E1 drive strength per bond type: x=" << lattice.drive_params.drive_strength_E1[0]
             << ", y=" << lattice.drive_params.drive_strength_E1[1]
             << ", z=" << lattice.drive_params.drive_strength_E1[2] << endl;
        cout << "E2 drive strength per bond type: x=" << lattice.drive_params.drive_strength_E2[0]
             << ", y=" << lattice.drive_params.drive_strength_E2[1]
             << ", z=" << lattice.drive_params.drive_strength_E2[2] << endl;
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
        cout << "  E1 drive strength per bond type: x=" << lattice.drive_params.drive_strength_E1[0]
             << ", y=" << lattice.drive_params.drive_strength_E1[1]
             << ", z=" << lattice.drive_params.drive_strength_E1[2] << endl;
        cout << "  E2 drive strength per bond type: x=" << lattice.drive_params.drive_strength_E2[0]
             << ", y=" << lattice.drive_params.drive_strength_E2[1]
             << ", z=" << lattice.drive_params.drive_strength_E2[2] << endl;
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
        
        // Set ordering pattern AFTER all equilibration is complete
        // This ensures O_custom = 1 at t=0 (spins match the ordering pattern)
        lattice.set_ordering_pattern();
        
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
            config.n_deterministics,
        config.twist_sweep_count
        );
        

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
 * @param comm MPI communicator to use (default: MPI_COMM_WORLD)
 */
void run_parallel_tempering_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size, MPI_Comm comm = MPI_COMM_WORLD) {
    if (rank == 0) {
        cout << "Running parallel tempering on mixed lattice with " << size << " replicas..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
    }
    
    // Generate temperature ladder
    vector<double> temps(size);
    
    if (config.pt_optimize_temperatures) {
        // Use MPI-distributed feedback-optimized temperature grid (Bittner et al.)
        // All ranks participate - much faster than single-rank optimization
        if (rank == 0) {
            cout << "Generating optimized temperature grid (Bittner et al., MPI-distributed) for MixedLattice..." << endl;
        }
        
        OptimizedTempGridResult opt_result = lattice.generate_optimized_temperature_grid_mpi(
            config.T_end,    // Tmin (coldest)
            config.T_start,  // Tmax (hottest)
            config.pt_optimization_warmup,
            config.pt_optimization_sweeps,
            config.pt_optimization_iterations,
            config.gaussian_move,
            config.overrelaxation_rate,
            config.pt_target_acceptance,
            0.05,  // convergence tolerance
            comm
        );
        temps = opt_result.temperatures;
        
        // Save optimized temperature grid info to file (rank 0 only)
        if (rank == 0 && !config.output_dir.empty()) {
            filesystem::create_directories(config.output_dir);
            ofstream opt_file(config.output_dir + "/optimized_temperatures.txt");
            opt_file << "# Optimized temperature grid (Bittner et al., Phys. Rev. Lett. 101, 130603)\n";
            opt_file << "# Target acceptance rate: " << config.pt_target_acceptance << "\n";
            opt_file << "# Mean acceptance rate: " << opt_result.mean_acceptance_rate << "\n";
            opt_file << "# Converged: " << (opt_result.converged ? "yes" : "no") << "\n";
            opt_file << "# Feedback iterations: " << opt_result.feedback_iterations_used << "\n";
            opt_file << "# Round-trip estimate: " << opt_result.round_trip_estimate << "\n";
            opt_file << "#\n";
            opt_file << "# rank  temperature  acceptance_rate  diffusivity\n";
            for (int i = 0; i < size; ++i) {
                opt_file << i << "  " << scientific << setprecision(12) << temps[i];
                if (i < size - 1) {
                    opt_file << "  " << fixed << setprecision(4) << opt_result.acceptance_rates[i]
                             << "  " << scientific << setprecision(6) << opt_result.local_diffusivities[i];
                }
                opt_file << "\n";
            }
            opt_file.close();
        }
    } else {
        // Use geometric (logarithmic) temperature spacing
        if (rank == 0) {
            cout << "Using geometric temperature grid..." << endl;
            temps = Lattice::generate_geometric_temperature_ladder(config.T_end, config.T_start, size);
        }
        // Broadcast temperatures from rank 0 to all ranks
        MPI_Bcast(temps.data(), size, MPI_DOUBLE, 0, comm);
    }
    
    // Re-initialize spins after temperature optimization (or geometric grid setup)
    // This ensures each rank starts with fresh random spins - the optimization
    // phase leaves spins in a "mixed" state from many replica exchanges
    lattice.init_random();
    MPI_Barrier(comm);
    
    for (int trial = 0; trial < config.num_trials; ++trial) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        if (rank == 0) {
            filesystem::create_directories(trial_dir);
        }
        MPI_Barrier(comm);  // Ensure directory is created before others proceed
        
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
            config.gaussian_move,
            true,  // use_interleaved
            comm
        );
        
        // T=0 deterministic quench for coldest replica (rank 0)
        if (config.T_zero && rank == 0 && config.n_deterministics > 0) {
            size_t total_sites = lattice.lattice_size_SU2 + lattice.lattice_size_SU3;
            cout << "Rank 0: Performing " << config.n_deterministics << " deterministic sweeps at T=0..." << endl;
            for (size_t sweep = 0; sweep < config.n_deterministics; ++sweep) {
                lattice.deterministic_sweep();
                if (sweep % 100 == 0 || sweep == config.n_deterministics - 1) {
                    cout << "Deterministic sweep " << sweep << "/" << config.n_deterministics 
                         << ", E/N = " << lattice.total_energy() / total_sites << endl;
                }
            }
            cout << "Deterministic sweeps completed. Final energy: " << lattice.total_energy() / total_sites << endl;
            // Save the T=0 quenched configuration
            lattice.save_spin_config_to_dir(trial_dir + "/rank_0", "spins_T0_quench");
        }
        MPI_Barrier(comm);
        
        if (rank == 0) {
            cout << "Trial " << trial << " completed." << endl;
        }
    }
    
    // Synchronize all ranks
    MPI_Barrier(comm);
    
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
                config.n_deterministics,
            config.twist_sweep_count
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
                config.n_deterministics,
            config.twist_sweep_count
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
                    config.n_deterministics,
                config.twist_sweep_count
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
                    config.n_deterministics,
                config.twist_sweep_count
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
    
    // ========================================================================
    // PARALLEL TEMPERING SPECIAL HANDLING
    // For parallel tempering, we need to split MPI_COMM_WORLD into sub-communicators
    // so each sweep point can run proper PT with multiple temperature replicas
    // ========================================================================
    bool is_parallel_tempering = (base_config.sweep_base_simulation == SimulationType::PARALLEL_TEMPERING);
    
    MPI_Comm sweep_comm = MPI_COMM_WORLD;  // Communicator for this rank's sweep point
    int sweep_rank = rank;                  // Rank within sweep_comm
    int sweep_size = size;                  // Size of sweep_comm
    int sweep_point_idx = -1;               // Which sweep point this rank is assigned to
    
    if (is_parallel_tempering) {
        // Determine ranks per sweep point
        int ranks_per_point = base_config.pt_ranks_per_point;
        if (ranks_per_point <= 0) {
            // Auto-detect: divide total ranks among sweep points, minimum 2 for PT
            ranks_per_point = max(2, size / static_cast<int>(all_sweep_points.size()));
        }
        
        // Calculate number of concurrent sweep points we can run
        int num_concurrent_points = size / ranks_per_point;
        if (num_concurrent_points < 1) {
            num_concurrent_points = 1;
            ranks_per_point = size;
        }
        
        // Assign this rank to a sweep point group (color) and compute local rank
        int color = rank / ranks_per_point;  // Which group (sweep point) this rank belongs to
        if (color >= num_concurrent_points) {
            // Extra ranks that don't fit into groups - assign to last group
            color = num_concurrent_points - 1;
        }
        
        // Create sub-communicator for this group
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sweep_comm);
        MPI_Comm_rank(sweep_comm, &sweep_rank);
        MPI_Comm_size(sweep_comm, &sweep_size);
        
        if (rank == 0) {
            cout << "\nParallel Tempering in Parameter Sweep Mode:" << endl;
            cout << "  Ranks per sweep point: " << ranks_per_point << endl;
            cout << "  Concurrent sweep points: " << num_concurrent_points << endl;
            cout << "  Total sweep points: " << all_sweep_points.size() << endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Distribute sweep points across groups (not individual ranks)
        // Each group processes sweep points in round-robin fashion
        for (size_t i = color; i < all_sweep_points.size(); i += num_concurrent_points) {
            const auto& param_values = all_sweep_points[i];
            
            // Print progress (only rank 0 of each group)
            if (sweep_rank == 0) {
                stringstream progress;
                progress << "[Group " << color << "] Processing point " << i+1 << "/" << all_sweep_points.size() << ": ";
                for (size_t p = 0; p < params.size(); ++p) {
                    if (p > 0) progress << ", ";
                    progress << params[p] << "=" << param_values[p];
                }
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
            if (sweep_rank == 0) {
                filesystem::create_directories(sweep_config.output_dir);
            }
            MPI_Barrier(sweep_comm);  // Ensure directory is created before others proceed
            
            // Run parallel tempering with the sub-communicator's rank/size
            if (sweep_config.system == SystemType::TMFEO3) {
                MixedUnitCell mixed_uc = build_tmfeo3(sweep_config);
                MixedLattice mixed_lattice(mixed_uc, sweep_config.lattice_size[0], 
                                          sweep_config.lattice_size[1], 
                                          sweep_config.lattice_size[2],
                                          sweep_config.use_twist_boundary);
                
                // Initialize spins
                if (sweep_config.use_ferromagnetic_init) {
                    SpinVector dir_su2(mixed_lattice.spin_dim_SU2);
                    for (size_t d = 0; d < mixed_lattice.spin_dim_SU2; ++d) {
                        dir_su2(d) = (d < sweep_config.ferromagnetic_direction.size()) ? 
                                     sweep_config.ferromagnetic_direction[d] : 0.0;
                    }
                    SpinVector dir_su3 = SpinVector::Zero(mixed_lattice.spin_dim_SU3);
                    const int su3_init_component = static_cast<int>(sweep_config.get_param("su3_init_component", 2.0));
                    if (su3_init_component >= 0 && su3_init_component < static_cast<int>(mixed_lattice.spin_dim_SU3)) {
                        dir_su3(su3_init_component) = 1.0;
                    } else {
                        dir_su3(2) = 1.0;
                    }
                    mixed_lattice.init_ferromagnetic(dir_su2, dir_su3);
                } else if (!sweep_config.initial_spin_config.empty()) {
                    mixed_lattice.load_spin_config(sweep_config.initial_spin_config);
                }
                
                // Run PT with proper sub-communicator rank/size
                run_parallel_tempering_mixed(mixed_lattice, sweep_config, sweep_rank, sweep_size, sweep_comm);
            } else if (sweep_config.system == SystemType::NCTO_STRAIN) {
                // StrainPhononLattice magnetoelastic system
                StrainPhononLattice strain_lattice(sweep_config.lattice_size[0],
                                                   sweep_config.lattice_size[1],
                                                   sweep_config.lattice_size[2],
                                                   sweep_config.spin_length);
                
                // Build parameters from config
                MagnetoelasticParams me_params;
                ElasticParams el_params;
                StrainDriveParams dr_params;
                build_strain_params(sweep_config, me_params, el_params, dr_params);
                
                // Set parameters
                strain_lattice.set_parameters(me_params, el_params, dr_params);
                strain_lattice.alpha_gilbert = sweep_config.get_param("alpha_gilbert", 0.0);
                
                // Set magnetic field
                Eigen::Vector3d B;
                B << sweep_config.field_strength * sweep_config.field_direction[0],
                     sweep_config.field_strength * sweep_config.field_direction[1],
                     sweep_config.field_strength * sweep_config.field_direction[2];
                strain_lattice.set_uniform_field(B);
                
                // Initialize spins
                if (!sweep_config.initial_spin_config.empty()) {
                    strain_lattice.load_spin_config(sweep_config.initial_spin_config);
                } else {
                    strain_lattice.init_random();
                }
                
                // Run PT with proper sub-communicator rank/size
                run_parallel_tempering_strain(strain_lattice, sweep_config, sweep_rank, sweep_size, sweep_comm);
            } else {
                // Standard lattice systems
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
                    case SystemType::PYROCHLORE_NON_KRAMER:
                        uc_ptr = new UnitCell(build_pyrochlore_non_kramer(sweep_config));
                        break;
                    case SystemType::TMFEO3_FE:
                        uc_ptr = new UnitCell(build_tmfeo3_fe(sweep_config));
                        break;
                    case SystemType::TMFEO3_TM:
                        uc_ptr = new UnitCell(build_tmfeo3_tm(sweep_config));
                        break;
                    default:
                        if (sweep_rank == 0) {
                            cerr << "Error: Unknown system type for parameter sweep with parallel tempering" << endl;
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
                    SpinVector dir(lattice.spin_dim);
                    for (size_t d = 0; d < lattice.spin_dim; ++d) {
                        dir(d) = (d < sweep_config.ferromagnetic_direction.size()) ? 
                                 sweep_config.ferromagnetic_direction[d] : 0.0;
                    }
                    lattice.init_ferromagnetic(dir);
                } else if (!sweep_config.initial_spin_config.empty()) {
                    lattice.load_spin_config(sweep_config.initial_spin_config);
                }
                
                // Run PT with proper sub-communicator rank/size
                run_parallel_tempering(lattice, sweep_config, sweep_rank, sweep_size, sweep_comm);
                
                delete uc_ptr;
            }
            
            // Print completion message (only rank 0 of each group)
            if (sweep_rank == 0) {
                stringstream completion;
                completion << "[Group " << color << "] Completed point " << i+1 << ": ";
                for (size_t p = 0; p < params.size(); ++p) {
                    if (p > 0) completion << ", ";
                    completion << params[p] << "=" << param_values[p];
                }
                cout << completion.str() << endl;
            }
            
            // Synchronize within the group before moving to next sweep point
            MPI_Barrier(sweep_comm);
        }
        
        // Free the sub-communicator
        MPI_Comm_free(&sweep_comm);
        
    } else {
        // ====================================================================
        // NON-PARALLEL-TEMPERING: Original behavior
        // Distribute sweep points across MPI ranks (one rank per point)
        // ====================================================================
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
        if (sweep_config.system == SystemType::NCTO) {
            // PhononLattice spin-phonon coupled system (honeycomb)
            PhononLattice phonon_lattice(sweep_config.lattice_size[0],
                                         sweep_config.lattice_size[1],
                                         sweep_config.lattice_size[2],
                                         sweep_config.spin_length);
            
            // Build parameters from config
            SpinPhononCouplingParams sp_params;
            PhononParams ph_params;
            DriveParams dr_params;
            TimeDependentSpinPhononParams td_sp_params;
            build_phonon_params(sweep_config, sp_params, ph_params, dr_params, td_sp_params);
            
            // Set parameters
            phonon_lattice.set_parameters(sp_params, ph_params, dr_params);
            phonon_lattice.set_time_dependent_spin_phonon(td_sp_params);
            phonon_lattice.alpha_gilbert = sweep_config.get_param("alpha_gilbert", 0.0);
            
            // Set magnetic field
            Eigen::Vector3d B;
            B << sweep_config.field_strength * sweep_config.field_direction[0],
                 sweep_config.field_strength * sweep_config.field_direction[1],
                 sweep_config.field_strength * sweep_config.field_direction[2];
            phonon_lattice.set_field(B);
            
            // Initialize spins
            if (sweep_config.use_ferromagnetic_init) {
                Eigen::Vector3d dir;
                dir << sweep_config.ferromagnetic_direction[0],
                       sweep_config.ferromagnetic_direction[1],
                       sweep_config.ferromagnetic_direction[2];
                phonon_lattice.init_ferromagnetic(dir);
            } else if (!sweep_config.initial_spin_config.empty()) {
                phonon_lattice.load_spin_config(sweep_config.initial_spin_config);
            } else {
                phonon_lattice.init_random();
            }
            
            // Run appropriate simulation
            switch (sweep_config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_phonon(phonon_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_phonon(phonon_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_phonon(phonon_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_phonon(phonon_lattice, sweep_config, 0, 1);
                    break;
                default:
                    cerr << "[Rank " << rank << "] Error: Unsupported base simulation for parameter sweep with PhononLattice" << endl;
                    break;
            }
        } else if (sweep_config.system == SystemType::NCTO_STRAIN) {
            // StrainPhononLattice magnetoelastic (spin-strain) coupled system (honeycomb)
            StrainPhononLattice strain_lattice(sweep_config.lattice_size[0],
                                               sweep_config.lattice_size[1],
                                               sweep_config.lattice_size[2],
                                               sweep_config.spin_length);
            
            // Build parameters from config
            MagnetoelasticParams me_params;
            ElasticParams el_params;
            StrainDriveParams dr_params;
            build_strain_params(sweep_config, me_params, el_params, dr_params);
            
            // Set parameters
            strain_lattice.set_parameters(me_params, el_params, dr_params);
            strain_lattice.alpha_gilbert = sweep_config.get_param("alpha_gilbert", 0.0);
            
            // Set magnetic field
            Eigen::Vector3d B;
            B << sweep_config.field_strength * sweep_config.field_direction[0],
                 sweep_config.field_strength * sweep_config.field_direction[1],
                 sweep_config.field_strength * sweep_config.field_direction[2];
            strain_lattice.set_uniform_field(B);
            
            // Initialize spins
            if (!sweep_config.initial_spin_config.empty()) {
                strain_lattice.load_spin_config(sweep_config.initial_spin_config);
            } else {
                strain_lattice.init_random();
            }
            
            // Run appropriate simulation
            switch (sweep_config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_strain(strain_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_strain(strain_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_strain(strain_lattice, sweep_config, 0, 1);
                    break;
                default:
                    cerr << "[Rank " << rank << "] Error: Unsupported base simulation for parameter sweep with StrainPhononLattice" << endl;
                    break;
            }
        } else if (sweep_config.system == SystemType::TMFEO3) {
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
                case SystemType::PYROCHLORE_NON_KRAMER:
                    uc_ptr = new UnitCell(build_pyrochlore_non_kramer(sweep_config));
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
        }  // End of for loop over sweep points
    }  // End of else (non-PT)
    
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
            TimeDependentSpinPhononParams td_sp_params;
            build_phonon_params(config, sp_params, ph_params, dr_params, td_sp_params);
            
            // Set parameters (this builds the interaction matrices)
            phonon_lattice.set_parameters(sp_params, ph_params, dr_params);
            
            // Set time-dependent spin-phonon coupling parameters
            phonon_lattice.set_time_dependent_spin_phonon(td_sp_params);
            
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
                case SimulationType::PARAMETER_SWEEP:
                    run_parameter_sweep(config, rank, size);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not supported for PhononLattice. "
                             << "Supported: SA, MD, pump_probe, 2dcs, parameter_sweep" << endl;
                    }
                    break;
            }
        } else if (config.system == SystemType::NCTO_STRAIN) {
            // StrainPhononLattice magnetoelastic (spin-strain) coupled system (honeycomb)
            if (rank == 0) {
                cout << "\nBuilding StrainPhononLattice magnetoelastic lattice..." << endl;
            }
            
            StrainPhononLattice strain_lattice(config.lattice_size[0],
                                               config.lattice_size[1],
                                               config.lattice_size[2],
                                               config.spin_length);
            
            // Build parameters from config
            MagnetoelasticParams me_params;
            ElasticParams el_params;
            StrainDriveParams dr_params;
            build_strain_params(config, me_params, el_params, dr_params);
            
            // Set parameters (this builds the interaction matrices)
            strain_lattice.set_parameters(me_params, el_params, dr_params);
            
            // Set Gilbert damping if specified
            strain_lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.0);
            
            // Set magnetic field
            Eigen::Vector3d B;
            B << config.field_strength * config.field_direction[0],
                 config.field_strength * config.field_direction[1],
                 config.field_strength * config.field_direction[2];
            strain_lattice.set_uniform_field(B);
            
            // Initialize spins
            if (!config.initial_spin_config.empty()) {
                strain_lattice.load_spin_config(config.initial_spin_config);
            } else {
                strain_lattice.init_random();
            }
            
            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_strain(strain_lattice, config, rank, size);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering_strain(strain_lattice, config, rank, size, MPI_COMM_WORLD);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_strain(strain_lattice, config, rank, size);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_strain(strain_lattice, config, rank, size);
                    break;
                case SimulationType::KINETIC_BARRIER_ANALYSIS:
                    run_kinetic_barrier_analysis_strain(strain_lattice, config, rank, size);
                    break;
                case SimulationType::PARAMETER_SWEEP:
                    run_parameter_sweep(config, rank, size);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not supported for StrainPhononLattice. "
                             << "Supported: SA, PT, MD, pump_probe, kinetic_barrier, parameter_sweep" << endl;
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
                case SystemType::PYROCHLORE_NON_KRAMER:
                    uc_ptr = new UnitCell(build_pyrochlore_non_kramer(config));
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
