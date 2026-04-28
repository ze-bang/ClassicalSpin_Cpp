/**
 * runners_lattice.cpp — spin_solver runners for the regular SU(2) Lattice.
 *
 * This TU was split out of `spin_solver.cpp` to cut its size and allow
 * parallel compilation. See `src/apps/spin_solver_runners.h` for the full
 * list of runners and `docs/refactor_backlog.md` for the broader plan.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/lattice/lattice.h"

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <cmath>

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
void run_parallel_tempering(Lattice& lattice, const SpinConfig& config, int rank, int size, MPI_Comm comm) {
    if (rank == 0) {
        cout << "Running parallel tempering with " << size << " replicas..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
    }
    
    // Generate temperature ladder
    vector<double> temps(size);
    vector<size_t> sweeps_per_temp;  // Bittner adaptive sweep schedule (empty = use fixed swap_rate)
    
    if (config.pt_optimize_temperatures) {
        // Use MPI-distributed feedback-optimized temperature grid
        // Phase 1: Katzgraber et al. feedback for uniform acceptance rates
        // Phase 2: Bittner et al. adaptive sweep schedule for minimal round-trip time
        if (rank == 0) {
            bool use_grad = (config.pt_temperature_optimizer == "gradient");
            cout << "Generating optimized temperature grid ("
                 << (use_grad ? "gradient-based, Miyata et al. 2024" : "Katzgraber+Bittner")
                 << ", MPI-distributed)..." << endl;
        }
        bool use_gradient = (config.pt_temperature_optimizer == "gradient");
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
            comm,
            use_gradient
        );
        temps = opt_result.temperatures;
        sweeps_per_temp = opt_result.sweeps_per_temp;
        
        // Save optimized temperature grid info to file (rank 0 only)
        if (rank == 0 && !config.output_dir.empty()) {
            filesystem::create_directories(config.output_dir);
            ofstream opt_file(config.output_dir + "/optimized_temperatures.txt");
            opt_file << "# Optimized temperature grid\n";
            opt_file << "# References: Katzgraber et al., PRE 73, 056702 (2006)\n";
            opt_file << "#             Bittner et al., PRL 101, 130603 (2008)\n";
            opt_file << "# Target acceptance rate: " << config.pt_target_acceptance << "\n";
            opt_file << "# Mean acceptance rate: " << opt_result.mean_acceptance_rate << "\n";
            opt_file << "# Converged: " << (opt_result.converged ? "yes" : "no") << "\n";
            opt_file << "# Feedback iterations: " << opt_result.feedback_iterations_used << "\n";
            opt_file << "# Round-trip estimate: " << opt_result.round_trip_estimate << "\n";
            opt_file << "#\n";
            opt_file << "# rank  temperature  acceptance_rate  diffusivity  tau_int  n_sweeps\n";
            for (int i = 0; i < size; ++i) {
                opt_file << i << "  " << scientific << setprecision(12) << temps[i];
                if (i < size - 1) {
                    opt_file << "  " << fixed << setprecision(4) << opt_result.acceptance_rates[i]
                             << "  " << scientific << setprecision(6) << opt_result.local_diffusivities[i];
                } else {
                    opt_file << "  " << fixed << setprecision(4) << 0.0
                             << "  " << scientific << setprecision(6) << 0.0;
                }
                if (!opt_result.autocorrelation_times.empty()) {
                    opt_file << "  " << fixed << setprecision(2) << opt_result.autocorrelation_times[i];
                }
                if (!opt_result.sweeps_per_temp.empty()) {
                    opt_file << "  " << opt_result.sweeps_per_temp[i];
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
            config.pt_n_bond_types,
            sweeps_per_temp  // Bittner adaptive sweep schedule
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
            config.use_gpu,
            config.md_abs_tol,
            config.md_rel_tol
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
    // Set Gilbert damping if specified
    lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.0);
    
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
        if (lattice.alpha_gilbert > 0.0) {
            cout << "Gilbert damping: alpha = " << lattice.alpha_gilbert << endl;
        }
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
            config.use_gpu,
            // Ingredient XV (W1/W3 — W2 replaced by MPI here):
            config.reuse_m0_for_m1,
            config.stationarity_tol,
            config.pulse_window_chunking,
            // Ingredient XVIII: pump-probe ODE tolerances (default 1e-8).
            config.pump_probe_abs_tol,
            config.pump_probe_rel_tol
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
            
            // Always equilibrate for 2DCS (need true ground state even if seed loaded)
            if (rank == 0 || config.num_trials == 1) {
                if (!config.initial_spin_config.empty()) {
                    cout << "\n[1/3] Equilibrating from loaded seed to true ground state..." << endl;
                } else {
                    cout << "\n[1/3] Equilibrating to ground state..." << endl;
                }
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
                config.use_gpu,
                // Ingredient XV (W1/W2/W3):
                config.reuse_m0_for_m1,
                config.stationarity_tol,
                config.pump_probe_omp_threads,
                config.pulse_window_chunking,
                config.pump_probe_abs_tol,
                config.pump_probe_rel_tol
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
