/**
 * runners_phonon.cpp — spin_solver runners for PhononLattice (discrete phonon modes).
 *
 * Split out of `spin_solver.cpp` so each lattice family compiles in its
 * own TU. See `src/apps/spin_solver_runners.h`.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/lattice/phonon_lattice.h"

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
 * Build spin-phonon / phonon / drive / time-dependent parameters from a SpinConfig.
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
    // Scalar quadratic E1 modulation of ring exchange:
    //   J7_eff = J7 + lambda_E1_J7_0 * |epsilon|^2.
    // Negative lambda_E1_J7_0 makes J7 decrease under E1 driving.
    sp_params.lambda_E1_J7_0 = config.get_param("lambda_E1_J7_0",
                                      config.get_param("lambda_J7_0", 0.0));
    
    // E1 magnetoelastic couplings: one isotropic (λ_X,0) and one
    // anisotropic (λ_X,2) coefficient per exchange channel X ∈ {J, K, Γ, Γ'}.
    // The leading symmetry-allowed E1 coupling is quadratic in ε; see
    // δX_γ(ε) = λ_X,0 (ε_x²+ε_y²) + λ_X,2 [(ε_x²-ε_y²) cos2θ_γ + 2ε_x ε_y sin2θ_γ].
    sp_params.lambda_E1_J_0      = config.get_param("lambda_E1_J_0",      config.get_param("lambda_J_0",      0.0));
    sp_params.lambda_E1_J_2      = config.get_param("lambda_E1_J_2",      config.get_param("lambda_J_2",      0.0));
    sp_params.lambda_E1_K_0      = config.get_param("lambda_E1_K_0",      config.get_param("lambda_K_0",      0.0));
    sp_params.lambda_E1_K_2      = config.get_param("lambda_E1_K_2",      config.get_param("lambda_K_2",      0.0));
    sp_params.lambda_E1_Gamma_0  = config.get_param("lambda_E1_Gamma_0",  config.get_param("lambda_Gamma_0",  0.0));
    sp_params.lambda_E1_Gamma_2  = config.get_param("lambda_E1_Gamma_2",  config.get_param("lambda_Gamma_2",  0.0));
    sp_params.lambda_E1_Gammap_0 = config.get_param("lambda_E1_Gammap_0", config.get_param("lambda_Gammap_0", 0.0));
    sp_params.lambda_E1_Gammap_2 = config.get_param("lambda_E1_Gammap_2", config.get_param("lambda_Gammap_2", 0.0));

    // Time-dependent E1 magnetoelastic scaling (single multiplicative
    // factor on all 8 quadratic coefficients).
    const double time_mode = config.get_param("lambda_time_mode", 0.0);
    td_sp_params.mode = (time_mode > 0.5) ? "window" : "constant";
    td_sp_params.t_start_E1 = config.get_param("lambda_E1_t_start", 0.0);
    td_sp_params.t_end_E1   = config.get_param("lambda_E1_t_end",   1e30);
    td_sp_params.e1_coupling_scale_target =
        config.get_param("lambda_E1_target", 1.0);

    // Zone-center E1 phonon parameters: ω_E1, γ_E1, optional quartic
    // self-coupling λ (ε²)²/4, and effective charge Z_E1*.
    ph_params.omega_E1          = config.get_param("omega_E1",          config.get_param("omega_E", 1.0));
    ph_params.gamma_E1          = config.get_param("gamma_E1",          config.get_param("gamma_E", 0.1));
    ph_params.lambda_E1_quartic = config.get_param("lambda_E1_quartic", config.get_param("lambda_E", 0.0));
    ph_params.Z_star            = config.get_param("Z_star", 1.0);

    // Drive parameters (pulse 1 - pump) - couples linearly to ε via -Z*·E·ε
    dr_params.E0_1    = config.pump_amplitude;
    dr_params.omega_1 = config.pump_frequency > 0 ? config.pump_frequency : ph_params.omega_E1;
    dr_params.t_1     = config.pump_time;
    dr_params.sigma_1 = config.pump_width;
    dr_params.phi_1   = config.get_param("pump_phase", 0.0);
    dr_params.theta_1 = config.get_param("pump_polarization", 0.0);

    // Drive parameters (pulse 2 - probe)
    dr_params.E0_2    = config.probe_amplitude;
    dr_params.omega_2 = config.probe_frequency > 0 ? config.probe_frequency : ph_params.omega_E1;
    dr_params.t_2     = config.probe_time;
    dr_params.sigma_2 = config.probe_width;
    dr_params.phi_2   = config.get_param("probe_phase", 0.0);
    dr_params.theta_2 = config.get_param("probe_polarization", 0.0);
}

/**
 * Run simulated annealing on PhononLattice (spin subsystem only).
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
            config.adiabatic_phonons,
            config.gaussian_move,
            config.get_param("preserve_initial_phonons", 0.0) > 0.5
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
        cout << "Zone-center E1 effective charge Z* = " << lattice.phonon_params.Z_star << endl;
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
                config.adiabatic_phonons,
                config.gaussian_move
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

        // Decide between deterministic MD and Langevin based on T.
        const double langevin_T_md = config.get_param("langevin_temperature", 0.0);
        const bool use_langevin = (langevin_T_md > 0.0);

        if (use_langevin) {
            lattice.langevin_temperature = langevin_T_md;
            if (lattice.alpha_gilbert <= 0.0)
                lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.01);
            if (rank == 0) {
                cout << "Starting Langevin dynamics (qualitative spin thermostat):" << endl;
                cout << "  T (k_B T) = " << langevin_T_md
                     << ", alpha_gilbert = " << lattice.alpha_gilbert << endl;
                cout << "  Time range: " << config.md_time_start
                     << " -> " << config.md_time_end << endl;
                cout << "  Fixed timestep: " << config.md_timestep << endl;
            }
            const uint64_t seed = static_cast<uint64_t>(
                config.get_param("langevin_seed", 0.0));
            lattice.integrate_langevin(
                config.md_time_start,
                config.md_time_end,
                config.md_timestep,
                trial_dir,
                config.md_save_interval,
                seed
            );
        } else {
            // Deterministic spin-phonon MD
            if (rank == 0) {
                cout << "Starting spin-phonon dynamics..." << endl;
                cout << "Time range: " << config.md_time_start
                     << " -> " << config.md_time_end << endl;
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
                config.md_abs_tol,
                config.md_rel_tol
            );
        }

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
        cout << "  Zone-center E1 effective charge Z* = " << lattice.phonon_params.Z_star << endl;
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
                config.adiabatic_phonons,
                config.gaussian_move
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
            config.md_integrator,
            config.md_abs_tol,
            config.md_rel_tol
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
                config.adiabatic_phonons,
                config.gaussian_move
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
                trial_dir, config.md_integrator,
                // Ingredient XV (W1/W3 — W2 replaced by MPI here):
                config.reuse_m0_for_m1,
                config.stationarity_tol,
                config.pulse_window_chunking,
                // Ingredient XVIII: pump-probe ODE tolerances (default 1e-8).
                config.pump_probe_abs_tol,
                config.pump_probe_rel_tol
            );
        } else {
            // Single-rank version
            lattice.pump_probe_spectroscopy(
                pulse_theta,
                pulse_amp, pulse_width, pulse_freq,
                tau_start, tau_end, tau_step,
                config.md_time_start, config.md_time_end, config.md_timestep,
                trial_dir, config.md_integrator,
                // Ingredient XV (W1/W2/W3):
                config.reuse_m0_for_m1,
                config.stationarity_tol,
                config.pump_probe_omp_threads,
                config.pulse_window_chunking,
                config.pump_probe_abs_tol,
                config.pump_probe_rel_tol
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
