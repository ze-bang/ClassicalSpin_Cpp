/**
 * runners_strain.cpp — spin_solver runners for StrainPhononLattice
 * (spin + magnetoelastic strain field).
 *
 * Also hosts the adiabatic-gradient helpers used by the kinetic-barrier
 * GNEB analysis (originally file-local statics in `spin_solver.cpp`).
 * Split out of `spin_solver.cpp`; see `src/apps/spin_solver_runners.h`.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/gneb.h"
#include "classical_spin/core/gneb_strain.h"
#include "classical_spin/lattice/strain_phonon_lattice.h"

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
 * Build strain / magnetoelastic parameters from a SpinConfig.
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
    
    // Quartic anharmonicity (optional, prevents ME runaway)
    el_params.kappa_A1g = config.get_param("kappa_A1g", config.get_param("lambda_A1g_quartic", 0.0));
    el_params.kappa_Eg = config.get_param("kappa_Eg", config.get_param("lambda_Eg_quartic", 0.0));
    
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
 * Run simulated annealing on StrainPhononLattice (spin subsystem only).
 */
void run_simulated_annealing_strain(StrainPhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running simulated annealing on StrainPhononLattice (spin subsystem)..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
    }
    
    // Fixed-strain (Born-Oppenheimer) mode: pin strain at external value
    if (config.fix_strain) {
        lattice.fix_strain_ = true;
        lattice.set_strain_Eg(config.external_strain_Eg1, config.external_strain_Eg2);
        if (rank == 0) {
            cout << "FIXED STRAIN mode: Eg1 = " << config.external_strain_Eg1 
                 << ", Eg2 = " << config.external_strain_Eg2 << endl;
        }
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
            if (!config.initial_spin_config.empty()) {
                lattice.load_spin_config(config.initial_spin_config);
            } else {
                lattice.init_random();
            }
        } else if (!config.initial_spin_config.empty()) {
            lattice.load_spin_config(config.initial_spin_config);
            if (rank == 0) {
                cout << "Loaded initial spin config from: " << config.initial_spin_config << endl;
            }
        }
        
        // Re-apply fixed strain after init_random (which may reset strain)
        if (config.fix_strain) {
            lattice.set_strain_Eg(config.external_strain_Eg1, config.external_strain_Eg2);
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
    
    // Check for Langevin mode
    double langevin_T = config.get_param("langevin_temperature", 0.0);
    bool use_langevin = (langevin_T > 0.0);
    if (use_langevin) {
        lattice.langevin_temperature = langevin_T;
        if (lattice.alpha_gilbert <= 0.0) {
            // Langevin requires damping; use default if not set
            lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.01);
        }
        if (rank == 0) {
            cout << "Langevin dynamics enabled: k_B T = " << langevin_T
                 << ", α = " << lattice.alpha_gilbert << endl;
        }
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        // Re-initialize spins for each trial (except first)
        if (trial > 0) {
            lattice.init_random();
        }
        
        if (use_langevin) {
            lattice.integrate_langevin(config.md_timestep, config.md_time_start, config.md_time_end,
                                       config.md_save_interval, trial_dir);
        } else {
            lattice.integrate_rk4(config.md_timestep, config.md_time_start, config.md_time_end, 
                                  config.md_save_interval, trial_dir);
        }
        
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
 * Compute the tangent-space projected gradient norm on the adiabatic PES
 * (strain relaxed to BO equilibrium) at a given spin configuration. This is
 * the quantity GNEB's convergence criterion acts on: |∇E_⊥| = max_i |∇E_i - (∇E_i·S_i)S_i|.
 */
static double adiabatic_projected_gradient_norm(StrainPhononLattice& lattice,
                                                const vector<Eigen::Vector3d>& spins) {
    auto [Eg1_eq, Eg2_eq] = lattice.relax_strain_at_fixed_spins(spins);
    auto [grad_spins, dE_dEg1, dE_dEg2] = lattice.gradient_for_gneb_with_strain(spins, Eg1_eq, Eg2_eq);
    double max_f = 0.0;
    for (size_t i = 0; i < spins.size(); ++i) {
        Eigen::Vector3d g_perp = grad_spins[i] - grad_spins[i].dot(spins[i]) * spins[i];
        max_f = std::max(max_f, g_perp.norm());
    }
    return max_f;
}

/**
 * Polish a spin configuration to the nearest local minimum on the adiabatic PES
 * using FIRE on spins only (strain relaxed to BO at every step). Returns the
 * final max projected gradient norm, and optionally writes iteration progress
 * to cout when verbose.
 *
 * This is used to ensure that endpoints loaded from files (e.g. pump-probe
 * output) are true stationary points before feeding them to GNEB — otherwise
 * the "barrier" measured by GNEB is contaminated by residual endpoint torque.
 */
static double polish_endpoint_adiabatic(StrainPhononLattice& lattice,
                                        vector<Eigen::Vector3d>& spins,
                                        const string& label,
                                        size_t max_iter,
                                        double force_tol,
                                        bool verbose) {
    // Defensive: unit-normalize the input spins.
    for (auto& S : spins) {
        double n = S.norm();
        if (n > 1e-12) S /= n;
    }

    double f0 = adiabatic_projected_gradient_norm(lattice, spins);
    if (verbose) {
        cout << "  [" << label << "] initial |∇E_⊥|_max = " << f0 << endl;
    }
    if (f0 < force_tol) {
        if (verbose) cout << "  [" << label << "] already at local min — no polish needed" << endl;
        return f0;
    }

    // FIRE on the tangent-projected gradient, strain adiabatically relaxed.
    size_t n_sites = spins.size();
    vector<Eigen::Vector3d> velocity(n_sites, Eigen::Vector3d::Zero());
    double dt = 0.05;
    const double dt_max = 0.5;
    const double dt_min = 1e-3;
    const double f_inc = 1.1, f_dec = 0.5;
    const double alpha_start = 0.1, alpha_decrease = 0.99;
    const size_t N_min = 5;
    double alpha = alpha_start;
    size_t n_positive = 0;

    double f_max = f0;
    size_t it = 0;
    for (; it < max_iter; ++it) {
        auto [Eg1_eq, Eg2_eq] = lattice.relax_strain_at_fixed_spins(spins);
        auto [grad_spins, dE_dEg1, dE_dEg2] = lattice.gradient_for_gneb_with_strain(spins, Eg1_eq, Eg2_eq);

        // Projected force F_i = -∇E_i + (∇E_i·S_i) S_i
        vector<Eigen::Vector3d> force(n_sites);
        f_max = 0.0;
        for (size_t i = 0; i < n_sites; ++i) {
            Eigen::Vector3d g_perp = grad_spins[i] - grad_spins[i].dot(spins[i]) * spins[i];
            force[i] = -g_perp;
            f_max = std::max(f_max, g_perp.norm());
        }
        if (f_max < force_tol) break;

        // FIRE bookkeeping
        double P = 0.0, F_sq = 0.0, V_sq = 0.0;
        for (size_t i = 0; i < n_sites; ++i) {
            P    += force[i].dot(velocity[i]);
            F_sq += force[i].squaredNorm();
            V_sq += velocity[i].squaredNorm();
        }
        double F_norm = std::sqrt(F_sq), V_norm = std::sqrt(V_sq);

        if (F_norm > 1e-12 && V_norm > 1e-12) {
            for (size_t i = 0; i < n_sites; ++i) {
                velocity[i] = (1.0 - alpha) * velocity[i]
                              + alpha * (V_norm / F_norm) * force[i];
            }
        }
        if (P > 0) {
            ++n_positive;
            if (n_positive > N_min) {
                dt = std::min(dt * f_inc, dt_max);
                alpha *= alpha_decrease;
            }
        } else {
            n_positive = 0;
            dt = std::max(dt * f_dec, dt_min);
            alpha = alpha_start;
            for (auto& v : velocity) v.setZero();
        }

        for (size_t i = 0; i < n_sites; ++i) {
            velocity[i] += dt * force[i];
            spins[i]    += dt * velocity[i];
            double n = spins[i].norm();
            if (n > 1e-12) spins[i] /= n;
            // Re-project velocity onto tangent plane so it doesn't accumulate radial drift
            velocity[i] -= velocity[i].dot(spins[i]) * spins[i];
        }

        if (verbose && it % 200 == 0) {
            cout << "  [" << label << "] iter " << std::setw(5) << it
                 << "  |∇E_⊥| = " << std::scientific << std::setprecision(3) << f_max
                 << "  dt = " << std::fixed << std::setprecision(3) << dt << endl;
        }
    }

    if (verbose) {
        cout << "  [" << label << "] polish done after " << it << " iter, "
             << "|∇E_⊥|_max = " << f_max
             << (f_max < force_tol ? "  [converged]" : "  [did not converge]") << endl;
    }
    return f_max;
}
void run_kinetic_barrier_analysis_strain(StrainPhononLattice& lattice, const SpinConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "\n" << string(70, '=') << endl;
        if (config.gneb_dynamic_strain) {
            cout << "KINETIC BARRIER ANALYSIS (GNEB with DYNAMIC STRAIN)" << endl;
        } else if (config.gneb_fixed_strain) {
            cout << "KINETIC BARRIER ANALYSIS (GNEB with FIXED STRAIN)" << endl;
        } else {
            cout << "KINETIC BARRIER ANALYSIS (GNEB with ADIABATIC STRAIN)" << endl;
        }
        cout << string(70, '=') << endl;
        if (config.gneb_dynamic_strain) {
            cout << "Configuration space: spins + strain (ε_Eg1, ε_Eg2)" << endl;
            cout << "  -> Strain is a dynamic GNEB degree of freedom" << endl;
            cout << "  -> Strain weight: " << config.gneb_weight_strain << endl;
        } else if (config.gneb_fixed_strain) {
            cout << "Configuration space: spins only (strain fixed externally)" << endl;
            cout << "  -> Appropriate for driven phonon experiments" << endl;
        } else {
            cout << "Configuration space: spins only (strain relaxed adiabatically)" << endl;
            cout << "  -> Appropriate for quasi-static strain" << endl;
        }
        cout << "Number of trials: " << config.num_trials << endl;
        cout << "MPI ranks: " << size << endl;
        cout << "\nGNEB parameters:" << endl;
        cout << "  N images:         " << config.gneb_n_images << endl;
        cout << "  Spring constant:  " << config.gneb_spring_constant << endl;
        cout << "  Max iterations:   " << config.gneb_max_iterations << endl;
        cout << "  Force tolerance:  " << config.gneb_force_tolerance << endl;
        cout << "  Climbing image:   " << (config.gneb_use_climbing_image ? "yes" : "no") << endl;
        cout << "\nMagnetoelastic coupling:" << endl;
        cout << "  lambda_Eg: " << lattice.magnetoelastic_params.lambda_Eg << endl;
        cout << "  Strain mode: " << (config.gneb_dynamic_strain ? "DYNAMIC" : (config.gneb_fixed_strain ? "FIXED" : "adiabatic")) << endl;
        if (config.gneb_strain_sweep) {
            cout << "\nStrain sweep: ENABLED" << endl;
            cout << "  N strain steps:   " << config.gneb_n_strain_steps << endl;
            cout << "  Strain max:       " << (config.gneb_strain_max < 0 ? "auto (from zigzag)" : to_string(config.gneb_strain_max)) << endl;
            cout << "  Strain direction: " << config.gneb_strain_direction << " rad" << endl;
            cout << "  Zigzag domain:    " << config.gneb_zigzag_domain << endl;
        }
        cout << string(70, '-') << endl;
    }
    
    // Get lattice dimensions
    const size_t n_sites = lattice.lattice_size;
    
    // Quenched exchange disorder (per-trial realization). Endpoint polishing
    // downstream will re-minimize initial / final states on the disordered
    // landscape, so we apply disorder BEFORE any strain relax / polish step.
    double kb_disorder_strength = config.get_param("disorder_strength", 0.0);
    unsigned int kb_disorder_seed_base =
        static_cast<unsigned int>(config.get_param("disorder_seed", 0.0));
    bool kb_has_disorder = (kb_disorder_strength > 0.0);
    if (kb_has_disorder) {
        lattice.store_clean_interactions();
        if (rank == 0) {
            cout << "\nQuenched disorder enabled (GNEB ensemble):" << endl;
            cout << "  Exchange disorder σ = " << kb_disorder_strength << endl;
            cout << "  Base seed = " << kb_disorder_seed_base << endl;
            cout << "  Each trial gets a unique realization." << endl;
        }
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        filesystem::create_directories(trial_dir + "/gneb");
        
        if (rank == 0 || config.num_trials > 1) {
            cout << "\n[Rank " << rank << "] === Trial " << trial << " ===" << endl;
        }
        
        if (kb_has_disorder) {
            lattice.restore_clean_interactions();
            unsigned int trial_seed =
                kb_disorder_seed_base + static_cast<unsigned int>(trial);
            lattice.apply_exchange_disorder(kb_disorder_strength, trial_seed);
        }
        
        // ================================================================
        // STEP 1: Get initial spin state and relax strain
        // ================================================================
        GNEBSpinConfig initial_spins(n_sites);
        StrainEg initial_strain;
        
        if (!config.gneb_initial_state_file.empty()) {
            // Load combined (spins, strain) from file
            cout << "[Rank " << rank << "] Loading initial spin+strain config from file..." << endl;
            lattice.load_spin_strain_config(config.gneb_initial_state_file);
            cout << "  File: " << config.gneb_initial_state_file << endl;
            
            // Extract spins
            for (size_t i = 0; i < n_sites; ++i) {
                initial_spins[i] = lattice.spins[i];
            }
            // Relax strain to equilibrium for these spins
            auto [init_Eg1, init_Eg2] = lattice.relax_strain_at_fixed_spins(initial_spins);
            initial_strain = StrainEg(init_Eg1, config.gneb_pin_Eg2_zero ? 0.0 : init_Eg2);
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
                initial_spins[i] = lattice.spins[i];
            }
            
            // Relax strain at fixed initial spins to find equilibrium strain
            cout << "[Rank " << rank << "] Relaxing strain for initial state..." << endl;
            auto [init_Eg1, init_Eg2] = lattice.relax_strain_at_fixed_spins(initial_spins);
            initial_strain = StrainEg(init_Eg1, config.gneb_pin_Eg2_zero ? 0.0 : init_Eg2);
        }

        // Endpoint diagnostics + optional polish on the adiabatic PES.
        // A nonzero |∇E_⊥| at the endpoint means it's not a true local minimum,
        // which contaminates the measured barrier. We always report it; we
        // optionally FIRE-polish to the nearest local min (default on).
        {
            double f_init = adiabatic_projected_gradient_norm(lattice, initial_spins);
            cout << "[Rank " << rank << "] Initial endpoint |∇E_⊥|_max (adiabatic PES) = "
                 << std::scientific << std::setprecision(6) << f_init << std::fixed << endl;
            if (config.gneb_polish_endpoints && f_init >= config.gneb_polish_force_tol) {
                cout << "[Rank " << rank << "] Polishing initial endpoint to local minimum..." << endl;
                polish_endpoint_adiabatic(lattice, initial_spins, "initial",
                                          config.gneb_polish_max_iter,
                                          config.gneb_polish_force_tol,
                                          /*verbose=*/rank == 0);
                // Re-relax strain for the polished spins
                auto [init_Eg1, init_Eg2] = lattice.relax_strain_at_fixed_spins(initial_spins);
                initial_strain = StrainEg(init_Eg1, config.gneb_pin_Eg2_zero ? 0.0 : init_Eg2);
            }
        }

        // Update lattice strain to match initial_state
        for (size_t b = 0; b < 3; ++b) {
            lattice.strain.epsilon_xx[b] = initial_strain.Eg1;
            lattice.strain.epsilon_yy[b] = -initial_strain.Eg1;
            lattice.strain.epsilon_xy[b] = initial_strain.Eg2;
        }
        
        // Analyze initial state
        for (size_t i = 0; i < n_sites; ++i) {
            lattice.spins[i] = initial_spins[i];
        }
        auto initial_cv = lattice.compute_collective_variables();
        cout << "[Rank " << rank << "] Initial state:" << endl;
        cout << "  m_3Q       = " << initial_cv.m_3Q << endl;
        cout << "  m_zigzag   = " << initial_cv.m_zigzag << endl;
        cout << "  f_Eg_amp   = " << initial_cv.f_Eg_amplitude << endl;
        cout << "  ε_Eg       = (" << initial_strain.Eg1 << ", " << initial_strain.Eg2 << ")" << endl;
        
        lattice.save_spin_strain_config(trial_dir + "/triple_q_state.txt");
        
        // ================================================================
        // STEP 2: Get final spin state and relax strain
        // ================================================================
        GNEBSpinConfig final_spins(n_sites);
        StrainEg final_strain;
        
        if (!config.gneb_final_state_file.empty()) {
            // Load combined (spins, strain) from file
            cout << "[Rank " << rank << "] Loading final spin+strain config from file..." << endl;
            lattice.load_spin_strain_config(config.gneb_final_state_file);
            cout << "  File: " << config.gneb_final_state_file << endl;
            
            // Extract spins
            for (size_t i = 0; i < n_sites; ++i) {
                final_spins[i] = lattice.spins[i];
            }
            // Relax strain to equilibrium for these spins
            auto [final_Eg1, final_Eg2] = lattice.relax_strain_at_fixed_spins(final_spins);
            final_strain = StrainEg(final_Eg1, config.gneb_pin_Eg2_zero ? 0.0 : final_Eg2);
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
                final_spins[i] = lattice.spins[i];
            }
            
            // Relax strain at fixed final spins (find equilibrium without bias)
            cout << "[Rank " << rank << "] Relaxing strain for final state..." << endl;
            auto [final_Eg1, final_Eg2] = lattice.relax_strain_at_fixed_spins(final_spins);
            final_strain = StrainEg(final_Eg1, config.gneb_pin_Eg2_zero ? 0.0 : final_Eg2);
        }

        // Endpoint diagnostics + optional polish for the final state.
        {
            double f_fin = adiabatic_projected_gradient_norm(lattice, final_spins);
            cout << "[Rank " << rank << "] Final endpoint |∇E_⊥|_max (adiabatic PES) = "
                 << std::scientific << std::setprecision(6) << f_fin << std::fixed << endl;
            if (config.gneb_polish_endpoints && f_fin >= config.gneb_polish_force_tol) {
                cout << "[Rank " << rank << "] Polishing final endpoint to local minimum..." << endl;
                polish_endpoint_adiabatic(lattice, final_spins, "final",
                                          config.gneb_polish_max_iter,
                                          config.gneb_polish_force_tol,
                                          /*verbose=*/rank == 0);
                auto [final_Eg1, final_Eg2] = lattice.relax_strain_at_fixed_spins(final_spins);
                final_strain = StrainEg(final_Eg1, config.gneb_pin_Eg2_zero ? 0.0 : final_Eg2);
            }
        }

        // Update lattice strain to match final_state
        for (size_t b = 0; b < 3; ++b) {
            lattice.strain.epsilon_xx[b] = final_strain.Eg1;
            lattice.strain.epsilon_yy[b] = -final_strain.Eg1;
            lattice.strain.epsilon_xy[b] = final_strain.Eg2;
        }
        
        // Analyze final state
        for (size_t i = 0; i < n_sites; ++i) {
            lattice.spins[i] = final_spins[i];
        }
        auto final_cv = lattice.compute_collective_variables();
        cout << "[Rank " << rank << "] Final state:" << endl;
        cout << "  m_3Q       = " << final_cv.m_3Q << endl;
        cout << "  m_zigzag   = " << final_cv.m_zigzag << endl;
        cout << "  f_Eg_amp   = " << final_cv.f_Eg_amplitude << endl;
        cout << "  ε_Eg       = (" << final_strain.Eg1 << ", " << final_strain.Eg2 << ")" << endl;
        
        lattice.save_spin_strain_config(trial_dir + "/zigzag_state.txt");
        
        // ================================================================
        // DYNAMIC STRAIN GNEB: strain as a DOF in the elastic band
        // ================================================================
        if (config.gneb_dynamic_strain) {
            cout << "[Rank " << rank << "] Using GNEBStrainOptimizer (strain as dynamic DOF)" << endl;
            
            // Build SpinStrainConfig endpoints
            SpinStrainConfig ss_initial(n_sites);
            for (size_t i = 0; i < n_sites; ++i) {
                ss_initial.spins[i] = initial_spins[i];
            }
            ss_initial.strain = initial_strain;
            
            SpinStrainConfig ss_final(n_sites);
            for (size_t i = 0; i < n_sites; ++i) {
                ss_final.spins[i] = final_spins[i];
            }
            ss_final.strain = final_strain;
            
            // Energy function: E(spins, strain)
            auto ss_energy_func = [&lattice](const SpinStrainConfig& cfg) -> double {
                return lattice.energy_for_gneb_with_strain(cfg.spins, cfg.strain.Eg1, cfg.strain.Eg2);
            };
            
            // Gradient function: (∂E/∂S_i, ∂E/∂ε_Eg1, ∂E/∂ε_Eg2)
            auto ss_gradient_func = [&lattice, n_sites](const SpinStrainConfig& cfg) -> SpinStrainGradient {
                auto [grad_spins, dE_dEg1, dE_dEg2] = lattice.gradient_for_gneb_with_strain(
                    cfg.spins, cfg.strain.Eg1, cfg.strain.Eg2);
                SpinStrainGradient grad(n_sites);
                grad.d_spins = grad_spins;
                grad.d_strain = StrainEg(dE_dEg1, dE_dEg2);
                return grad;
            };
            
            GNEBStrainOptimizer gneb_strain(ss_energy_func, ss_gradient_func, n_sites);
            
            // Set parameters
            GNEBStrainParams ss_params;
            ss_params.n_images = config.gneb_n_images;
            ss_params.spring_constant = config.gneb_spring_constant;
            ss_params.max_iterations = config.gneb_max_iterations;
            ss_params.force_tolerance = config.gneb_force_tolerance;
            ss_params.step_size = config.gneb_step_size;
            ss_params.fire_dtmax = config.gneb_fire_dtmax;
            ss_params.max_strain_amplitude = config.gneb_max_strain_amplitude;
            ss_params.climbing_start = config.gneb_climbing_start;
            ss_params.redistribution_freq = config.gneb_redistribution_freq;
            ss_params.climbing_image = config.gneb_use_climbing_image;
            ss_params.weight_strain = config.gneb_weight_strain;
            ss_params.adiabatic_strain = config.gneb_adiabatic_strain;
            ss_params.verbosity = (rank == 0) ? 2 : 0;
            
            // Born-Oppenheimer strain relaxation function
            if (config.gneb_adiabatic_strain) {
                bool pin_Eg2 = config.gneb_pin_Eg2_zero;
                auto relax_func = [&lattice, pin_Eg2](const SpinStrainConfig& cfg) -> StrainEg {
                    // Warm-start from current strain (avoids reconverging from zero)
                    auto [Eg1, Eg2] = lattice.relax_strain_at_fixed_spins(
                        cfg.spins, cfg.strain.Eg1, cfg.strain.Eg2);
                    // Pin Eg2=0 to constrain MEP to a single C₃ strain channel
                    if (pin_Eg2) Eg2 = 0.0;
                    return StrainEg(Eg1, Eg2);
                };
                gneb_strain.set_strain_relaxation(relax_func);
                
                cout << "[Rank " << rank << "] Using Born-Oppenheimer strain (adiabatic relaxation)";
                if (pin_Eg2) cout << " [Eg2 pinned to 0]";
                cout << endl;
            }
            
            // Find MEP in combined (spin, strain) space
            GNEBStrainResult ss_result;
            
            if (!config.gneb_initial_path_dir.empty()) {
                // Load initial path from directory with path_00.txt, path_01.txt, ...
                cout << "[Rank " << rank << "] Loading initial path from: " 
                     << config.gneb_initial_path_dir << endl;
                
                vector<SpinStrainConfig> initial_path;
                for (int idx = 0; ; ++idx) {
                    string padded = (idx < 10) ? "0" + to_string(idx) : to_string(idx);
                    string path_file = config.gneb_initial_path_dir + "/path_" + padded + ".txt";
                    
                    if (!filesystem::exists(path_file)) break;
                    
                    // Load spin+strain config
                    lattice.load_spin_strain_config(path_file);
                    
                    SpinStrainConfig cfg(n_sites);
                    for (size_t i = 0; i < n_sites; ++i) {
                        cfg.spins[i] = lattice.spins[i];
                    }
                    // Extract Eg strain from lattice (bond 0)
                    cfg.strain = StrainEg(lattice.strain.epsilon_xx[0], 
                                          lattice.strain.epsilon_xy[0]);
                    
                    initial_path.push_back(cfg);
                    cout << "  Loaded image " << idx << ": ε = (" 
                         << cfg.strain.Eg1 << ", " << cfg.strain.Eg2 
                         << "), |ε| = " << cfg.strain.amplitude() << endl;
                }
                
                cout << "[Rank " << rank << "] Loaded " << initial_path.size() 
                     << " images for initial path" << endl;
                
                // Override n_images with actual path size
                ss_params.n_images = initial_path.size();
                
                cout << "[Rank " << rank << "] Finding MEP from initial path..." << endl;
                ss_result = gneb_strain.find_mep_from_path(initial_path, ss_params);
            } else {
                cout << "[Rank " << rank << "] Finding MEP (from geodesic interpolation)..." << endl;
                ss_result = gneb_strain.find_mep(ss_initial, ss_final, ss_params);
            }
            
            cout << "[Rank " << rank << "] Dynamic strain GNEB result:" << endl;
            cout << "  Converged:    " << (ss_result.converged ? "yes" : "no") << endl;
            cout << "  Iterations:   " << ss_result.iterations_used << endl;
            cout << "  Barrier:      " << ss_result.barrier << endl;
            cout << "  ΔE:           " << ss_result.delta_E << endl;
            cout << "  Saddle image: " << ss_result.saddle_index << endl;
            cout << "  Saddle strain: (" << ss_result.saddle_strain.Eg1 
                 << ", " << ss_result.saddle_strain.Eg2 << ")" << endl;
            
            // Save path using built-in method
            string gneb_dir = trial_dir + "/gneb";
            filesystem::create_directories(gneb_dir);
            gneb_strain.save_path(gneb_dir, "mep");
            
            // Also save in mep_with_strain.txt format for compatibility
            {
                ofstream mep_file(trial_dir + "/mep_with_strain.txt");
                mep_file << "# Minimum energy path with DYNAMIC strain (spin + strain GNEB)\n";
                mep_file << "# gneb_dynamic_strain = true\n";
                mep_file << "# weight_strain = " << config.gneb_weight_strain << "\n";
                mep_file << "# image  reaction_coord  energy  strain_Eg1  strain_Eg2  strain_amp  m_3Q  m_zigzag  f_Eg_amp\n";
                
                for (size_t i = 0; i < ss_result.images.size(); ++i) {
                    // Set lattice spins to compute collective variables
                    for (size_t j = 0; j < n_sites; ++j) {
                        lattice.spins[j] = ss_result.images[i].spins[j];
                    }
                    auto cv = lattice.compute_collective_variables();
                    
                    mep_file << i << "  " << ss_result.arc_lengths[i] 
                             << "  " << ss_result.energies[i]
                             << "  " << ss_result.images[i].strain.Eg1 
                             << "  " << ss_result.images[i].strain.Eg2
                             << "  " << ss_result.images[i].strain.amplitude()
                             << "  " << cv.m_3Q << "  " << cv.m_zigzag 
                             << "  " << cv.f_Eg_amplitude << "\n";
                }
            }
            
            // Save barrier summary
            {
                ofstream summary(trial_dir + "/barrier_summary.txt");
                summary << "# Kinetic Barrier Summary (Dynamic Strain GNEB)\n";
                summary << "# Trial: " << trial << "\n";
                summary << "# Configuration space: (S_1,...,S_N, ε_Eg1, ε_Eg2)\n";
                summary << "# weight_strain = " << config.gneb_weight_strain << "\n";
                summary << "#\n";
                summary << "barrier = " << ss_result.barrier << "\n";
                summary << "gneb_iterations = " << ss_result.iterations_used << "\n";
                summary << "gneb_converged = " << (ss_result.converged ? "true" : "false") << "\n";
                summary << "saddle_image = " << ss_result.saddle_index << "\n";
                summary << "max_force = " << ss_result.max_force << "\n";
                summary << "#\n";
                summary << "initial_strain_Eg1 = " << ss_result.initial_strain.Eg1 << "\n";
                summary << "initial_strain_Eg2 = " << ss_result.initial_strain.Eg2 << "\n";
                summary << "saddle_strain_Eg1 = " << ss_result.saddle_strain.Eg1 << "\n";
                summary << "saddle_strain_Eg2 = " << ss_result.saddle_strain.Eg2 << "\n";
                summary << "final_strain_Eg1 = " << ss_result.final_strain.Eg1 << "\n";
                summary << "final_strain_Eg2 = " << ss_result.final_strain.Eg2 << "\n";
                summary << "max_strain_amplitude = " << ss_result.max_strain_amplitude << "\n";
                summary << "#\n";
                summary << "# Initial energy: " << ss_result.energies.front() << "\n";
                summary << "# Saddle energy: " << ss_result.energies[ss_result.saddle_index] << "\n";
                summary << "# Final energy: " << ss_result.energies.back() << "\n";
            }
            
            cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
            cout << "  Output saved to: " << trial_dir << endl;
            continue;  // Skip the spin-only GNEB path below
        }
        
        // ================================================================
        // STEP 3: Determine strain sweep range
        // ================================================================
        // The zigzag equilibrium strain sets the natural scale for the sweep.
        // At ε_ext = 0, the barrier is the unbiased triple-Q → zigzag barrier.
        // At ε_ext = ε_zigzag^eq, the zigzag is maximally stabilized.
        
        double strain_sweep_max = 0.0;
        double strain_dir_Eg1 = 1.0;  // Unit vector for strain direction
        double strain_dir_Eg2 = 0.0;
        size_t n_strain_pts = 1;      // Default: single point (no sweep)
        
        if (config.gneb_strain_sweep) {
            // Determine max strain from zigzag equilibrium if not specified
            if (config.gneb_strain_max < 0) {
                // Use the zigzag equilibrium strain amplitude
                strain_sweep_max = final_strain.amplitude();
                cout << "[Rank " << rank << "] Auto strain max from zigzag: " << strain_sweep_max << endl;
                
                // Use the direction of the zigzag equilibrium strain
                if (strain_sweep_max > 1e-10) {
                    strain_dir_Eg1 = final_strain.Eg1 / strain_sweep_max;
                    strain_dir_Eg2 = final_strain.Eg2 / strain_sweep_max;
                }
            } else {
                strain_sweep_max = config.gneb_strain_max;
                // Use specified direction angle: Eg1 = cos(θ), Eg2 = sin(θ)
                strain_dir_Eg1 = std::cos(config.gneb_strain_direction);
                strain_dir_Eg2 = std::sin(config.gneb_strain_direction);
            }
            n_strain_pts = config.gneb_n_strain_steps + 1;  // Include 0
            
            cout << "[Rank " << rank << "] Strain sweep: " << n_strain_pts << " points from 0 to " 
                 << strain_sweep_max << endl;
            cout << "  Direction: (" << strain_dir_Eg1 << ", " << strain_dir_Eg2 << ")" << endl;
        }
        
        // Storage for barrier vs strain results
        vector<double> sweep_strains(n_strain_pts);
        vector<double> sweep_barriers(n_strain_pts);
        vector<double> sweep_E_initial(n_strain_pts);
        vector<double> sweep_E_final(n_strain_pts);
        vector<double> sweep_E_saddle(n_strain_pts);
        vector<size_t> sweep_saddle_idx(n_strain_pts);
        vector<bool>   sweep_converged(n_strain_pts);
        
        // Store converged path to use as initial guess for next strain point
        vector<GNEBSpinConfig> previous_path;
        
        // ================================================================
        // STEP 4: GNEB loop (single point or strain sweep)
        // ================================================================
        for (size_t s_idx = 0; s_idx < n_strain_pts; ++s_idx) {
            // External strain amplitude for this sweep point
            double ext_amp = (n_strain_pts > 1) ? 
                strain_sweep_max * static_cast<double>(s_idx) / static_cast<double>(n_strain_pts - 1) : 0.0;
            double ext_Eg1 = ext_amp * strain_dir_Eg1;
            double ext_Eg2 = ext_amp * strain_dir_Eg2;
            sweep_strains[s_idx] = ext_amp;
            
            if (config.gneb_strain_sweep) {
                cout << "\n[Rank " << rank << "] --- Strain point " << s_idx 
                     << "/" << (n_strain_pts-1) << ": ε_ext = " << ext_amp 
                     << " (" << ext_Eg1 << ", " << ext_Eg2 << ") ---" << endl;
            }
            
            // DEBUG: Print energy decomposition for initial and final states
            if (s_idx == 0 || s_idx == n_strain_pts - 1) {
                cout << "[DEBUG] Energy decomposition at ε = " << ext_amp << ":" << endl;
                
                // For initial state
                for (size_t i = 0; i < n_sites; ++i) lattice.spins[i] = initial_spins[i];
                for (size_t b = 0; b < 3; ++b) {
                    lattice.strain.epsilon_xx[b] = ext_Eg1;
                    lattice.strain.epsilon_yy[b] = -ext_Eg1;
                    lattice.strain.epsilon_xy[b] = ext_Eg2;
                }
                double E_spin_init = lattice.spin_energy();
                double E_strain_init = lattice.strain_energy();
                double E_me_init = lattice.magnetoelastic_energy();
                double E_gneb_init = lattice.energy_for_gneb_with_strain(initial_spins, ext_Eg1, ext_Eg2);
                
                // Also print the spin bilinears
                double fK_Eg1 = lattice.f_K_Eg1();
                double fJ_Eg1 = lattice.f_J_Eg1();
                double fG_Eg1 = lattice.f_Gamma_Eg1();
                double fGp_Eg1 = lattice.f_Gammap_Eg1();
                double J = lattice.magnetoelastic_params.J;
                double K = lattice.magnetoelastic_params.K;
                double Gamma = lattice.magnetoelastic_params.Gamma;
                double Gammap = lattice.magnetoelastic_params.Gammap;
                double lambda_Eg = lattice.magnetoelastic_params.lambda_Eg;
                double Sigma_Eg1 = (J+K)*fK_Eg1 + J*fJ_Eg1 + Gamma*fG_Eg1 + Gammap*fGp_Eg1;
                
                cout << "  Initial (triple-Q):" << endl;
                cout << "    spin_energy = " << E_spin_init << endl;
                cout << "    strain_energy = " << E_strain_init << endl;
                cout << "    magnetoelastic_energy = " << E_me_init << endl;
                cout << "    energy_for_gneb = " << E_gneb_init << endl;
                cout << "    --- Spin bilinears ---" << endl;
                cout << "    f_K_Eg1 = " << fK_Eg1 << ", f_J_Eg1 = " << fJ_Eg1 << endl;
                cout << "    f_Gamma_Eg1 = " << fG_Eg1 << ", f_Gammap_Eg1 = " << fGp_Eg1 << endl;
                cout << "    Sigma_Eg1 = (J+K)*fK + J*fJ + Γ*fΓ + Γ'*fΓ' = " << Sigma_Eg1 << endl;
                cout << "    J=" << J << ", K=" << K << ", Γ=" << Gamma << ", Γ'=" << Gammap << ", λ_Eg=" << lambda_Eg << endl;
                
                // For final state
                for (size_t i = 0; i < n_sites; ++i) lattice.spins[i] = final_spins[i];
                double E_spin_fin = lattice.spin_energy();
                double E_me_fin = lattice.magnetoelastic_energy();
                double E_gneb_fin = lattice.energy_for_gneb_with_strain(final_spins, ext_Eg1, ext_Eg2);
                double fK_Eg1_f = lattice.f_K_Eg1();
                double fJ_Eg1_f = lattice.f_J_Eg1();
                double fG_Eg1_f = lattice.f_Gamma_Eg1();
                double fGp_Eg1_f = lattice.f_Gammap_Eg1();
                double Sigma_Eg1_f = (J+K)*fK_Eg1_f + J*fJ_Eg1_f + Gamma*fG_Eg1_f + Gammap*fGp_Eg1_f;
                
                cout << "  Final (zigzag):" << endl;
                cout << "    spin_energy = " << E_spin_fin << endl;
                cout << "    magnetoelastic_energy = " << E_me_fin << endl;
                cout << "    energy_for_gneb = " << E_gneb_fin << endl;
                cout << "    --- Spin bilinears ---" << endl;
                cout << "    f_K_Eg1 = " << fK_Eg1_f << ", f_J_Eg1 = " << fJ_Eg1_f << endl;
                cout << "    f_Gamma_Eg1 = " << fG_Eg1_f << ", f_Gammap_Eg1 = " << fGp_Eg1_f << endl;
                cout << "    Sigma_Eg1 = " << Sigma_Eg1_f << endl;
                cout << "  Delta_Sigma_Eg1 = " << (Sigma_Eg1_f - Sigma_Eg1) << endl;
            }
            
            // Energy and gradient functions depend on strain mode
            std::function<double(const GNEBSpinConfig&)> energy_func;
            std::function<GNEBSpinConfig(const GNEBSpinConfig&)> gradient_func;
            
            if (config.gneb_fixed_strain) {
                // FIXED STRAIN MODE: strain = ext_strain (no relaxation)
                // This is appropriate for driven phonon experiments
                energy_func = [&lattice, ext_Eg1, ext_Eg2](const GNEBSpinConfig& spins) -> double {
                    return lattice.energy_for_gneb_with_strain(spins, ext_Eg1, ext_Eg2);
                };
                
                gradient_func = [&lattice, ext_Eg1, ext_Eg2](const GNEBSpinConfig& spins) -> GNEBSpinConfig {
                    auto [grad_spins, dE_dEg1, dE_dEg2] = lattice.gradient_for_gneb_with_strain(
                        spins, ext_Eg1, ext_Eg2);
                    return grad_spins;
                };
            } else {
                // ADIABATIC MODE: strain relaxes to equilibrium at each spin config
                // E_eff(spins) = E(spins, ε_ext + ε_int^eq(spins))
                energy_func = [&lattice, ext_Eg1, ext_Eg2](const GNEBSpinConfig& spins) -> double {
                    auto [int_Eg1, int_Eg2] = lattice.relax_strain_with_external(spins, ext_Eg1, ext_Eg2);
                    double total_Eg1 = ext_Eg1 + int_Eg1;
                    double total_Eg2 = ext_Eg2 + int_Eg2;
                    return lattice.energy_for_gneb_with_strain(spins, total_Eg1, total_Eg2);
                };
                
                gradient_func = [&lattice, ext_Eg1, ext_Eg2](const GNEBSpinConfig& spins) -> GNEBSpinConfig {
                    auto [int_Eg1, int_Eg2] = lattice.relax_strain_with_external(spins, ext_Eg1, ext_Eg2);
                    double total_Eg1 = ext_Eg1 + int_Eg1;
                    double total_Eg2 = ext_Eg2 + int_Eg2;
                    auto [grad_spins, dE_dEg1, dE_dEg2] = lattice.gradient_for_gneb_with_strain(
                        spins, total_Eg1, total_Eg2);
                    return grad_spins;
                };
            }
            
            GNEBOptimizer gneb(energy_func, gradient_func, n_sites);
            
            // GNEB parameters from config
            GNEBParams gneb_params;
            gneb_params.n_images = config.gneb_n_images;
            gneb_params.spring_constant = config.gneb_spring_constant;
            gneb_params.max_iterations = config.gneb_max_iterations;
            gneb_params.force_tolerance = config.gneb_force_tolerance;
            gneb_params.climbing_image = config.gneb_use_climbing_image;
            gneb_params.verbosity = (rank == 0 && !config.gneb_strain_sweep) ? 2 : 
                                    (rank == 0 ? 1 : 0);
            
            // Find MEP - use previous path as initial guess if available
            GNEBResult mep_result;
            if (previous_path.empty()) {
                // First strain point: start from geodesic interpolation
                cout << "[Rank " << rank << "] Finding MEP (from geodesic interpolation)..." << endl;
                mep_result = gneb.find_mep(initial_spins, final_spins, gneb_params);
            } else {
                // Subsequent strain points: use previous converged path as initial guess
                cout << "[Rank " << rank << "] Refining MEP (using previous path as initial guess)..." << endl;
                mep_result = gneb.find_mep_from_path(previous_path, gneb_params);
            }
            
            // Store converged path for next strain point
            previous_path = mep_result.images;
            
            cout << "[Rank " << rank << "] MEP: iter=" << mep_result.iterations_used 
                 << " barrier=" << mep_result.barrier 
                 << " saddle=" << mep_result.saddle_index 
                 << " converged=" << (mep_result.converged ? "yes" : "no") << endl;
            
            sweep_barriers[s_idx] = mep_result.barrier;
            sweep_E_initial[s_idx] = mep_result.energies.front();
            sweep_E_final[s_idx] = mep_result.energies.back();
            sweep_E_saddle[s_idx] = mep_result.energies[mep_result.saddle_index];
            sweep_saddle_idx[s_idx] = mep_result.saddle_index;
            sweep_converged[s_idx] = mep_result.converged;
            
            // Compute adiabatic strain at each image along MEP
            vector<StrainEg> mep_strains(mep_result.images.size());
            double max_strain_amplitude = 0.0;
            for (size_t i = 0; i < mep_result.images.size(); ++i) {
                auto [int_Eg1, int_Eg2] = lattice.relax_strain_with_external(
                    mep_result.images[i], ext_Eg1, ext_Eg2);
                // Store total strain
                mep_strains[i] = StrainEg(ext_Eg1 + int_Eg1, ext_Eg2 + int_Eg2);
                max_strain_amplitude = max(max_strain_amplitude, mep_strains[i].amplitude());
            }

            // ──────────────────────────────────────────────────────────────
            // BO-invariant check: |∂E/∂ε| at every image must vanish to BO
            // tolerance after adiabatic relaxation. If it doesn't, the
            // adiabatic strain relaxation under-converged at that image and
            // the reported barrier is biased.
            //
            // Theoretical statement (for this Hamiltonian, E quadratic and
            // strictly convex in ε at fixed spins): critical points of the
            // MEP on the BO surface satisfy ∂E/∂ε = 0 exactly, so the BO
            // saddle is identical to a saddle of the full E(S, ε). This
            // check makes that theoretical identity a numerical invariant.
            // ──────────────────────────────────────────────────────────────
            double max_dE_deps_all = 0.0;
            double max_dE_deps_saddle = 0.0;
            {
                for (size_t i = 0; i < mep_result.images.size(); ++i) {
                    auto [gs, dE_dEg1, dE_dEg2] = lattice.gradient_for_gneb_with_strain(
                        mep_result.images[i], mep_strains[i].Eg1, mep_strains[i].Eg2);
                    double g = std::sqrt(dE_dEg1 * dE_dEg1 + dE_dEg2 * dE_dEg2);
                    max_dE_deps_all = std::max(max_dE_deps_all, g);
                    if (i == mep_result.saddle_index) max_dE_deps_saddle = g;
                }
                cout << "[Rank " << rank << "] BO invariant check: "
                     << "max |∂E/∂ε| over path = " << std::scientific << std::setprecision(3)
                     << max_dE_deps_all
                     << "  (saddle: " << max_dE_deps_saddle << ")" << std::fixed << endl;
                // Warn if the adiabatic relaxation didn't deliver BO-quality strain.
                const double bo_warn_tol = 1e-3;
                if (max_dE_deps_all > bo_warn_tol) {
                    cout << "[Rank " << rank << "] WARNING: |∂E/∂ε| = "
                         << max_dE_deps_all << " exceeds BO tolerance "
                         << bo_warn_tol << " — adiabatic strain relaxation "
                         << "is under-converged on the MEP; barrier is biased." << endl;
                }
            }
            
            // Save MEP for this strain point
            string strain_label = config.gneb_strain_sweep ? 
                "/strain_" + to_string(s_idx) : "";
            string mep_dir = trial_dir + strain_label;
            if (config.gneb_strain_sweep) {
                filesystem::create_directories(mep_dir);
            }
            
            // Save MEP with strain info
            {
                ofstream mep_file(mep_dir + "/mep_with_strain.txt");
                mep_file << "# Minimum energy path with adiabatic strain relaxation\n";
                mep_file << "# External strain: ε_Eg1 = " << ext_Eg1 << ", ε_Eg2 = " << ext_Eg2 << "\n";
                mep_file << "# External strain amplitude: " << ext_amp << "\n";
                mep_file << "# image  reaction_coord  energy  strain_Eg1  strain_Eg2  strain_amp  m_3Q  m_zigzag  f_Eg_amp\n";
                
                for (size_t i = 0; i < mep_result.energies.size(); ++i) {
                    for (size_t j = 0; j < n_sites; ++j) {
                        lattice.spins[j] = mep_result.images[i][j];
                    }
                    auto cv = lattice.compute_collective_variables();
                    
                    mep_file << i << "  " << mep_result.arc_lengths[i] 
                             << "  " << mep_result.energies[i]
                             << "  " << mep_strains[i].Eg1 
                             << "  " << mep_strains[i].Eg2
                             << "  " << mep_strains[i].amplitude()
                             << "  " << cv.m_3Q << "  " << cv.m_zigzag 
                             << "  " << cv.f_Eg_amplitude << "\n";
                }
            }
            
            // Save per-strain summary
            {
                ofstream summary(mep_dir + "/barrier_summary.txt");
                summary << "# Kinetic Barrier Summary (Spin-only GNEB + Adiabatic Strain)\n";
                summary << "# Trial: " << trial << "\n";
                summary << "# External strain: ε_Eg1 = " << ext_Eg1 << ", ε_Eg2 = " << ext_Eg2 << "\n";
                summary << "# External strain amplitude: " << ext_amp << "\n";
                summary << "#\n";
                summary << "barrier = " << mep_result.barrier << "\n";
                summary << "gneb_iterations = " << mep_result.iterations_used << "\n";
                summary << "gneb_converged = " << (mep_result.converged ? "true" : "false") << "\n";
                summary << "saddle_image = " << mep_result.saddle_index << "\n";
                summary << "saddle_curvature = " << mep_result.saddle_curvature << "\n";
                summary << "max_force = " << mep_result.max_force << "\n";
                summary << "#\n";
                summary << "external_strain_Eg1 = " << ext_Eg1 << "\n";
                summary << "external_strain_Eg2 = " << ext_Eg2 << "\n";
                summary << "external_strain_amplitude = " << ext_amp << "\n";
                summary << "#\n";
                summary << "# Adiabatic (total) strain at key points:\n";
                summary << "initial_strain_Eg1 = " << mep_strains[0].Eg1 << "\n";
                summary << "initial_strain_Eg2 = " << mep_strains[0].Eg2 << "\n";
                summary << "saddle_strain_Eg1 = " << mep_strains[mep_result.saddle_index].Eg1 << "\n";
                summary << "saddle_strain_Eg2 = " << mep_strains[mep_result.saddle_index].Eg2 << "\n";
                summary << "final_strain_Eg1 = " << mep_strains.back().Eg1 << "\n";
                summary << "final_strain_Eg2 = " << mep_strains.back().Eg2 << "\n";
                summary << "max_strain_amplitude = " << max_strain_amplitude << "\n";
                summary << "#\n";
                summary << "# BO invariant: |∂E/∂ε| should vanish at every image (convex elastic PES)\n";
                summary << "bo_dE_deps_max_path   = " << max_dE_deps_all << "\n";
                summary << "bo_dE_deps_at_saddle  = " << max_dE_deps_saddle << "\n";
                summary << "#\n";
                summary << "# Initial (triple-Q) energy: " << mep_result.energies.front() << "\n";
                summary << "# Saddle energy: " << mep_result.energies[mep_result.saddle_index] << "\n";
                summary << "# Final (zigzag) energy: " << mep_result.energies.back() << "\n";
            }
            
            // Save MEP images if requested (only for non-sweep or if sweep evolution saving is on)
            if (config.gneb_save_path_evolution && (!config.gneb_strain_sweep || s_idx == 0 || s_idx == n_strain_pts-1)) {
                string gneb_dir = mep_dir + "/gneb";
                filesystem::create_directories(gneb_dir);
                gneb.save_path(gneb_dir, "mep");

                // Overwrite mep_image_*.txt with strain-annotated versions so each
                // image file carries its own (ε_Eg1, ε_Eg2, ε_ext, E) metadata.
                for (size_t img = 0; img < mep_result.images.size(); ++img) {
                    string img_file = gneb_dir + "/mep_image_" + to_string(img) + ".txt";
                    ofstream out(img_file);
                    out << "# MEP image " << img << " (spin-only GNEB + adiabatic strain)\n";
                    out << "# external_strain_Eg1 = " << ext_Eg1 << "\n";
                    out << "# external_strain_Eg2 = " << ext_Eg2 << "\n";
                    out << "# total_strain_Eg1 = " << mep_strains[img].Eg1 << "\n";
                    out << "# total_strain_Eg2 = " << mep_strains[img].Eg2 << "\n";
                    out << "# total_strain_amplitude = " << mep_strains[img].amplitude() << "\n";
                    out << "# arc_length = " << mep_result.arc_lengths[img] << "\n";
                    out << "# energy = " << mep_result.energies[img] << "\n";
                    out << "# site  Sx  Sy  Sz\n";
                    for (size_t i = 0; i < n_sites; ++i) {
                        out << i << "  " << mep_result.images[img][i].x()
                            << "  " << mep_result.images[img][i].y()
                            << "  " << mep_result.images[img][i].z() << "\n";
                    }
                }

                // Tidy summary of strain along the MEP (matches mep_energies.txt layout)
                {
                    ofstream sfile(gneb_dir + "/mep_strains.txt");
                    sfile << "# image  arc_length  strain_Eg1  strain_Eg2  strain_amplitude  energy\n";
                    for (size_t img = 0; img < mep_result.images.size(); ++img) {
                        sfile << img
                              << "  " << mep_result.arc_lengths[img]
                              << "  " << mep_strains[img].Eg1
                              << "  " << mep_strains[img].Eg2
                              << "  " << mep_strains[img].amplitude()
                              << "  " << mep_result.energies[img] << "\n";
                    }
                }
            }
        }  // End strain sweep loop
        
        // ================================================================
        // STEP 5: Save strain sweep summary (barrier vs strain)
        // ================================================================
        if (config.gneb_strain_sweep) {
            ofstream sweep_file(trial_dir + "/barrier_vs_strain.txt");
            sweep_file << "# Kinetic barrier vs external strain\n";
            sweep_file << "# Triple-Q → Zigzag transition\n";
            sweep_file << "# Strain direction: (" << strain_dir_Eg1 << ", " << strain_dir_Eg2 << ")\n";
            sweep_file << "# Zigzag equilibrium strain: (" << final_strain.Eg1 << ", " << final_strain.Eg2 
                       << ") amplitude=" << final_strain.amplitude() << "\n";
            sweep_file << "# strain_amplitude  barrier  E_initial  E_saddle  E_final  saddle_idx  converged\n";
            
            for (size_t s_idx = 0; s_idx < n_strain_pts; ++s_idx) {
                sweep_file << sweep_strains[s_idx] << "  " 
                           << sweep_barriers[s_idx] << "  "
                           << sweep_E_initial[s_idx] << "  "
                           << sweep_E_saddle[s_idx] << "  "
                           << sweep_E_final[s_idx] << "  "
                           << sweep_saddle_idx[s_idx] << "  "
                           << (sweep_converged[s_idx] ? 1 : 0) << "\n";
            }
            sweep_file.close();
            
            cout << "\n[Rank " << rank << "] === Barrier vs Strain Summary ===" << endl;
            cout << "  Strain        Barrier       ΔE(3Q-ZZ)     Converged" << endl;
            for (size_t s_idx = 0; s_idx < n_strain_pts; ++s_idx) {
                double dE = sweep_E_final[s_idx] - sweep_E_initial[s_idx];
                cout << "  " << sweep_strains[s_idx] 
                     << "        " << sweep_barriers[s_idx]
                     << "        " << dE
                     << "        " << (sweep_converged[s_idx] ? "yes" : "no") << endl;
            }
            
            // Find spinodal point (where barrier vanishes)
            double spinodal_strain = -1.0;
            for (size_t s_idx = 1; s_idx < n_strain_pts; ++s_idx) {
                if (sweep_barriers[s_idx] <= 0.0 && sweep_barriers[s_idx-1] > 0.0) {
                    // Linear interpolation
                    double f = sweep_barriers[s_idx-1] / (sweep_barriers[s_idx-1] - sweep_barriers[s_idx]);
                    spinodal_strain = sweep_strains[s_idx-1] + f * (sweep_strains[s_idx] - sweep_strains[s_idx-1]);
                    break;
                }
            }
            
            if (spinodal_strain > 0) {
                cout << "  Spinodal strain (barrier=0): ε_c ≈ " << spinodal_strain << endl;
                
                ofstream spinodal_file(trial_dir + "/spinodal.txt");
                spinodal_file << "# Spinodal strain where kinetic barrier vanishes\n";
                spinodal_file << "spinodal_strain = " << spinodal_strain << "\n";
                spinodal_file << "strain_direction = " << config.gneb_strain_direction << "\n";
                spinodal_file << "strain_dir_Eg1 = " << strain_dir_Eg1 << "\n";
                spinodal_file << "strain_dir_Eg2 = " << strain_dir_Eg2 << "\n";
            } else {
                cout << "  Barrier remains positive across entire strain range." << endl;
            }
        }
        
        cout << "[Rank " << rank << "] Trial " << trial << " completed." << endl;
        cout << "  Output saved to: " << trial_dir << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\n" << string(70, '=') << endl;
        cout << "Kinetic barrier analysis (adiabatic strain) completed (" << config.num_trials << " trials)" << endl;
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
    
    // ── Quenched disorder parameters ──
    double disorder_strength = config.get_param("disorder_strength", 0.0);
    double dilution_fraction = config.get_param("dilution_fraction", 0.0);
    unsigned int disorder_seed_base = static_cast<unsigned int>(config.get_param("disorder_seed", 0.0));
    bool has_disorder = (disorder_strength > 0.0 || dilution_fraction > 0.0);
    
    if (has_disorder) {
        // Store clean interactions before any disorder is applied
        lattice.store_clean_interactions();
        if (rank == 0) {
            cout << "\nQuenched disorder enabled:" << endl;
            if (disorder_strength > 0.0)
                cout << "  Exchange disorder σ = " << disorder_strength << endl;
            if (dilution_fraction > 0.0)
                cout << "  Site dilution p = " << dilution_fraction << endl;
            cout << "  Base seed = " << disorder_seed_base << endl;
            cout << "  Each trial gets a unique disorder realization." << endl;
        }
    }
    
    // ── Local strain (per-unit-cell DOF) ──
    bool use_local_strain = static_cast<bool>(config.get_param("local_strain", 0.0));
    double K_gradient = config.get_param("K_gradient", 0.0);
    if (use_local_strain) {
        lattice.elastic_params.K_gradient = K_gradient;
        lattice.init_local_strain();
        if (rank == 0) {
            cout << "\nLocal strain (per-cell DOF) enabled:" << endl;
            cout << "  K_gradient = " << K_gradient << endl;
            cout << "  N_cells = " << lattice.get_N_cells() << endl;
        }
    }
    
    // Distribute trials across MPI ranks
    for (int trial = rank; trial < config.num_trials; trial += size) {
        string trial_dir = config.output_dir + "/sample_" + to_string(trial);
        filesystem::create_directories(trial_dir);
        
        if (config.num_trials > 1) {
            cout << "[Rank " << rank << "] Trial " << trial << " / " << config.num_trials << endl;
        }
        
        // ── Apply fresh disorder realization for this trial ──
        if (has_disorder) {
            lattice.restore_clean_interactions();
            unsigned int trial_seed = disorder_seed_base + static_cast<unsigned int>(trial);
            if (disorder_strength > 0.0)
                lattice.apply_exchange_disorder(disorder_strength, trial_seed);
            if (dilution_fraction > 0.0)
                lattice.apply_site_dilution(dilution_fraction, trial_seed + 1000000);
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
        
        // Check for Langevin mode
        double langevin_T_pp = config.get_param("langevin_temperature", 0.0);
        if (langevin_T_pp > 0.0) {
            lattice.langevin_temperature = langevin_T_pp;
            if (lattice.alpha_gilbert <= 0.0)
                lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.01);
            if (rank == 0)
                cout << "  Langevin dynamics: k_B T = " << langevin_T_pp
                     << ", α = " << lattice.alpha_gilbert << endl;
            lattice.integrate_langevin(config.md_timestep, config.md_time_start, config.md_time_end,
                                       config.md_save_interval, trial_dir);
        } else {
            lattice.integrate_rk4(config.md_timestep, config.md_time_start, config.md_time_end, 
                                  config.md_save_interval, trial_dir);
        }
        
        // Save final configuration (combined spin+strain format)
        lattice.save_spin_strain_config(trial_dir + "/final_spin_strain.txt");
        lattice.save_spin_config(trial_dir + "/final_spins.txt");
        lattice.save_strain_state(trial_dir + "/final_strain.txt");
        if (use_local_strain) {
            lattice.save_local_strain_map(trial_dir + "/final_local_strain_map.txt");
        }
        
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
void run_parallel_tempering_strain(StrainPhononLattice& lattice, const SpinConfig& config, int rank, int size, MPI_Comm comm) {
    if (rank == 0) {
        cout << "Running parallel tempering on StrainPhononLattice with " << size << " replicas..." << endl;
        cout << "Number of trials: " << config.num_trials << endl;
    }
    
    // Generate temperature ladder
    vector<double> temps(size);
    vector<size_t> sweeps_per_temp;  // Bittner adaptive sweep schedule
    
    if (config.pt_optimize_temperatures) {
        // Use MPI-distributed feedback-optimized temperature grid
        if (rank == 0) {
            bool use_grad_spl = (config.pt_temperature_optimizer == "gradient");
            cout << "Generating optimized temperature grid ("
                 << (use_grad_spl ? "gradient-based, Miyata et al. 2024" : "Katzgraber+Bittner")
                 << ", MPI-distributed) for StrainPhononLattice..." << endl;
        }
        bool use_gradient_spl = (config.pt_temperature_optimizer == "gradient");
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
            comm,
            use_gradient_spl
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
            true,  // verbose: save spin configurations
            sweeps_per_temp  // Bittner adaptive sweep schedule
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
