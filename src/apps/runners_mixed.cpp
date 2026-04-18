/**
 * runners_mixed.cpp — spin_solver runners for MixedLattice (TmFeO3 SU(2)+SU(3)).
 *
 * Split out of `spin_solver.cpp`; see `src/apps/spin_solver_runners.h`.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/lattice/lattice.h"       // Lattice::generate_geometric_temperature_ladder
#include "classical_spin/lattice/mixed_lattice.h"

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

/**
 * Run simulated annealing on MixedLattice (SU(2)+SU(3)).
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
void run_parallel_tempering_mixed(MixedLattice& lattice, const SpinConfig& config, int rank, int size, MPI_Comm comm) {
    if (rank == 0) {
        cout << "Running parallel tempering on mixed lattice with " << size << " replicas..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
    }
    
    // Generate temperature ladder
    vector<double> temps(size);
    vector<size_t> sweeps_per_temp;  // Bittner adaptive sweep schedule
    
    if (config.pt_optimize_temperatures) {
        // Use MPI-distributed feedback-optimized temperature grid
        if (rank == 0) {
            bool use_grad_mix = (config.pt_temperature_optimizer == "gradient");
            cout << "Generating optimized temperature grid ("
                 << (use_grad_mix ? "gradient-based, Miyata et al. 2024" : "Katzgraber+Bittner")
                 << ", MPI-distributed) for MixedLattice..." << endl;
        }
        bool use_gradient_mix = (config.pt_temperature_optimizer == "gradient");
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
            use_gradient_mix
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
            true,  // use_interleaved
            comm,
            false,  // verbose
            sweeps_per_temp  // Bittner adaptive sweep schedule
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
    double local_pump_amplitude_su3 = config.pump_amplitude_su3;
    double local_pump_width_su3 = config.pump_width_su3;
    double local_pump_frequency_su3 = config.pump_frequency_su3;
    
    // Auto-compute SU3 pulse from physical 3D direction using mu_act projection
    vector<vector<double>> pump_dirs_su3_norm = config.pump_directions_su3;
    if (config.auto_su3_pump) {
        const double mu_2x = config.get_param("mu_2x", 0.0);
        const double mu_2y = config.get_param("mu_2y", 0.0);
        const double mu_2z = config.get_param("mu_2z", 5.264);
        const double mu_5x = config.get_param("mu_5x", 2.3915);
        const double mu_5y = config.get_param("mu_5y", -2.7866);
        const double mu_5z = config.get_param("mu_5z", 0.0);
        const double mu_7x = config.get_param("mu_7x", 0.9128);
        const double mu_7y = config.get_param("mu_7y", 0.4655);
        const double mu_7z = config.get_param("mu_7z", 0.0);
        const double g_ratio = config.get_param("g_ratio_tm", 7.0/12.0);
        
        double mu_act[3][3] = {
            {mu_2x, mu_5x, mu_7x},
            {mu_2y, mu_5y, mu_7y},
            {mu_2z, mu_5z, mu_7z}
        };
        const int active_idx[3] = {1, 4, 6};
        
        // Project each physical 3D pump direction to 8D Gell-Mann space
        // B_a = Σ_α μ_{αa} n̂_α  (sublattice-0 reference, frames handle the rest)
        pump_dirs_su3_norm.clear();
        for (const auto& dir3d : pump_dirs_norm) {
            vector<double> su3_dir(lattice.spin_dim_SU3, 0.0);
            for (int a = 0; a < 3; ++a) {
                for (int al = 0; al < 3; ++al) {
                    su3_dir[active_idx[a]] += mu_act[al][a] * dir3d[al];
                }
            }
            pump_dirs_su3_norm.push_back(su3_dir);
        }
        
        double su3_norm = 0.0;
        for (double v : pump_dirs_su3_norm[0]) su3_norm += v * v;
        su3_norm = sqrt(su3_norm);
        
        // SU3 amplitude = physical_amplitude * g_ratio * |μ^T n̂|
        local_pump_amplitude_su3 = config.pump_amplitude * g_ratio * su3_norm;
        local_pump_width_su3 = config.pump_width;
        local_pump_frequency_su3 = config.pump_frequency;
        
        if (rank == 0) {
            cout << "Auto-computing SU3 pulse from physical direction:" << endl;
            cout << "  g_ratio_tm = " << g_ratio << endl;
            cout << "  |mu^T * n| = " << su3_norm << endl;
            cout << "  SU3 amplitude = " << local_pump_amplitude_su3 
                 << " (Fe amplitude = " << config.pump_amplitude << ")" << endl;
        }
    }
    
    // Normalize all SU3 pump directions
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
            local_pump_amplitude_su3, local_pump_width_su3, local_pump_frequency_su3,
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
    double local_pump_amplitude_su3 = config.pump_amplitude_su3;
    double local_pump_width_su3 = config.pump_width_su3;
    double local_pump_frequency_su3 = config.pump_frequency_su3;
    
    // Auto-compute SU3 pulse from physical 3D direction using mu_act projection
    vector<vector<double>> pump_dirs_su3_norm = config.pump_directions_su3;
    if (config.auto_su3_pump) {
        const double mu_2x = config.get_param("mu_2x", 0.0);
        const double mu_2y = config.get_param("mu_2y", 0.0);
        const double mu_2z = config.get_param("mu_2z", 5.264);
        const double mu_5x = config.get_param("mu_5x", 2.3915);
        const double mu_5y = config.get_param("mu_5y", -2.7866);
        const double mu_5z = config.get_param("mu_5z", 0.0);
        const double mu_7x = config.get_param("mu_7x", 0.9128);
        const double mu_7y = config.get_param("mu_7y", 0.4655);
        const double mu_7z = config.get_param("mu_7z", 0.0);
        const double g_ratio = config.get_param("g_ratio_tm", 7.0/12.0);
        
        double mu_act[3][3] = {
            {mu_2x, mu_5x, mu_7x},
            {mu_2y, mu_5y, mu_7y},
            {mu_2z, mu_5z, mu_7z}
        };
        const int active_idx[3] = {1, 4, 6};
        
        pump_dirs_su3_norm.clear();
        for (const auto& dir3d : pump_dirs_norm) {
            vector<double> su3_dir(lattice.spin_dim_SU3, 0.0);
            for (int a = 0; a < 3; ++a) {
                for (int al = 0; al < 3; ++al) {
                    su3_dir[active_idx[a]] += mu_act[al][a] * dir3d[al];
                }
            }
            pump_dirs_su3_norm.push_back(su3_dir);
        }
        
        double su3_norm = 0.0;
        for (double v : pump_dirs_su3_norm[0]) su3_norm += v * v;
        su3_norm = sqrt(su3_norm);
        
        local_pump_amplitude_su3 = config.pump_amplitude * g_ratio * su3_norm;
        local_pump_width_su3 = config.pump_width;
        local_pump_frequency_su3 = config.pump_frequency;
        
        if (rank == 0) {
            cout << "Auto-computing SU3 pulse from physical direction:" << endl;
            cout << "  g_ratio_tm = " << g_ratio << endl;
            cout << "  |mu^T * n| = " << su3_norm << endl;
            cout << "  SU3 amplitude = " << local_pump_amplitude_su3 
                 << " (Fe amplitude = " << config.pump_amplitude << ")" << endl;
        }
    }
    
    // Normalize all SU3 pump directions
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
        
        // Always equilibrate for 2DCS (only rank 0)
        if (rank == 0) {
            if (!config.initial_spin_config.empty()) {
                cout << "\n[1/2] Equilibrating from loaded seed to true ground state..." << endl;
            } else {
                cout << "\n[1/2] Equilibrating to ground state..." << endl;
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
        const bool save_spin_traj = (config.get_param("save_spin_trajectories", 0.0) > 0.5);
        lattice.pump_probe_spectroscopy_mpi(
            field_dirs_su2,
            field_dirs_su3,
            config.pump_amplitude,
            config.pump_width,
            config.pump_frequency,
            local_pump_amplitude_su3,
            local_pump_width_su3,
            local_pump_frequency_su3,
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
            save_spin_traj
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
                config.gaussian_move,
                config.cooling_rate,
                trial_dir,
                config.save_observables,
                config.T_zero,
                config.n_deterministics,
                config.twist_sweep_count
            );
            
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
                local_pump_amplitude_su3,
                local_pump_width_su3,
                local_pump_frequency_su3,
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
