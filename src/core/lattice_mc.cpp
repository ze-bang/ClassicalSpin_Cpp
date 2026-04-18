/**
 * @file lattice_mc.cpp
 * @brief Lattice Monte Carlo driver and sweep logic.
 *
 * Holds the heavier MC-driver methods that were previously inline in
 * `lattice.h`: simulated/cluster annealing, perform_final_measurements,
 * perform_mc_sweeps, the Metropolis twist-boundary sweep, and the
 * MC-side observable output routines.
 *
 * The class declaration and small inline helpers (metropolis(),
 * overrelaxation(), wolff_update(), deterministic_sweep(), ...) stay in
 * the header to keep hot inner loops inlineable.
 */

#include "classical_spin/lattice/lattice.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

// ---- Lattice::metropolis_twist_sweep ----
    size_t Lattice::metropolis_twist_sweep(double T) {
        if (T <= 0) return 0;
        if (spin_dim != 3) return 0;  // Only defined for 3D spins
        
        size_t accepted = 0;
        
        // Probability of attempting a global (random) move vs incremental
        const double global_move_prob = 0.1;
        
        // For each dimension that has length > 1, attempt one rotation update
        for (size_t d = 0; d < 3; ++d) {
            size_t Ld = (d == 0) ? dim1 : (d == 1) ? dim2 : dim3;
            if (Ld <= 1) continue;
            
            // Energy on boundary sites before the move
            double E_before = 0.0;
            for (size_t idx : boundary_sites_per_dim[d]) {
                E_before += site_energy(spins[idx], idx);
            }
            
            double angle_new;
            bool is_global_move = (random_double_lehman(0, 1) < global_move_prob);
            
            if (is_global_move) {
                // Global move: propose completely random angle
                angle_new = random_double_lehman(0, 2 * M_PI);
            } else {
                // Incremental move: small perturbation to current angle
                // Step size scales with temperature (larger at high T)
                double max_step = std::min(0.3, std::max(0.02, T * 0.5));
                double delta = random_double_lehman(-max_step, max_step);
                angle_new = twist_angles[d] + delta;
            }
            
            // Generate rotation matrix around z-axis
            SpinMatrix R_new = rotation_from_axis_angle(rotation_axis[d], angle_new);
            
            // Save old state and apply proposed move
            SpinMatrix saved_R = twist_matrices[d];
            double saved_angle = twist_angles[d];
            twist_matrices[d] = R_new;
            twist_angles[d] = angle_new;
            
            // Energy after the move
            double E_after = 0.0;
            for (size_t idx : boundary_sites_per_dim[d]) {
                E_after += site_energy(spins[idx], idx);
            }
            
            double dE = E_after - E_before;
            bool accept = (dE < 0) || (random_double_lehman(0, 1) < std::exp(-dE / T));
            if (!accept) {
                twist_matrices[d] = saved_R;
                twist_angles[d] = saved_angle;
            } else {
                accepted++;
            }
        }
        
        return accepted;
    }

// ---- Lattice::simulated_annealing ----
    void Lattice::simulated_annealing(double T_start, double T_end, size_t n_anneal,
                            size_t overrelaxation_rate,
                            bool boundary_update,
                            bool gaussian_move,
                            double cooling_rate,
                            string out_dir,
                            bool save_observables,
                            bool T_zero,
                            size_t n_deterministics,
                            size_t twist_sweep_count) {
        
        // Setup output directory
        if (!out_dir.empty()) {
            std::filesystem::create_directories(out_dir);
        }
        
        // Initialize random seed
        seed_lehman(chrono::system_clock::now().time_since_epoch().count() * 2 + 1);
        
        double T = T_start;
        double sigma = 1000.0;
        
        cout << "Starting simulated annealing: T=" << T_start << " → " << T_end << endl;
        if (T_zero) {
            cout << "T=0 mode enabled: will perform " << n_deterministics << " deterministic sweeps at T=0" << endl;
        }
        if (boundary_update) {
            cout << "Twist boundary updates enabled: " << twist_sweep_count << " twist sweeps per MC sweep" << endl;
        }
        
        size_t temp_step = 0;
        while (T > T_end) {
            // Perform sweeps at this temperature
            double acc_sum = perform_mc_sweeps(n_anneal, T, gaussian_move, sigma, 
                                              overrelaxation_rate, boundary_update, twist_sweep_count);
            
            // Calculate acceptance rate (normalize differently if overrelaxation is used)
            double acceptance = (overrelaxation_rate > 0) ? 
                acc_sum / double(n_anneal) * overrelaxation_rate : 
                acc_sum / double(n_anneal);
            
            // Progress report
            if (temp_step % 10 == 0 || T <= T_end * 1.5) {
                double E = energy_density();
                cout << "T=" << std::scientific << T << ", E/N=" << E 
                     << ", acc=" << std::fixed << acceptance;
                if (gaussian_move) cout << ", σ=" << sigma;
                cout << endl;
            }
            
            // Adaptive sigma adjustment for gaussian moves (match original logic)
            if (gaussian_move && acceptance < 0.5) {
                sigma = sigma * 0.5 / (1.0 - acceptance);
                if (temp_step % 10 == 0 || T <= T_end * 1.5) {
                    cout << "Sigma adjusted to " << sigma << endl;
                }
            }
            
            // Cool down
            T *= cooling_rate;
            ++temp_step;
        }
        
        cout << "Final energy density: " << energy_density() << endl;
        
        // Save spin config after annealing (before deterministic sweeps)
        if (!out_dir.empty()) {
            save_spin_config(out_dir + "/spins_T=" + std::to_string(T) + ".txt");
            save_positions(out_dir + "/positions.txt");
            if (boundary_update) {
                save_twist_angles(out_dir + "/twist_angles_T=" + std::to_string(T) + ".txt");
            }
        }
        
        // Final measurements if requested (before deterministic sweeps)
        if (save_observables && !out_dir.empty()) {
            perform_final_measurements(T_end, sigma, gaussian_move, 
                                      overrelaxation_rate, out_dir);
        }
        
        // T=0 deterministic sweeps if requested
        if (T_zero && n_deterministics > 0) {
            cout << "\nPerforming " << n_deterministics << " deterministic sweeps at T=0..." << endl;
            for (size_t sweep = 0; sweep < n_deterministics; ++sweep) {
                deterministic_sweep(1);
                
                if (sweep % 100 == 0 || sweep == n_deterministics - 1) {
                    double E = energy_density();
                    cout << "Deterministic sweep " << sweep << "/" << n_deterministics 
                         << ", E/N=" << E << endl;
                }
            }
            cout << "Deterministic sweeps completed. Final energy: " << energy_density() << endl;
            // Save final configuration
            if (!out_dir.empty()) {
                save_spin_config(out_dir + "/spins_T=0.txt");
                if (boundary_update) {
                    save_twist_angles(out_dir + "/twist_angles_T=0.txt");
                }
            }
        }
    }

// ---- Lattice::perform_final_measurements ----
    void Lattice::perform_final_measurements(double T_final, double sigma, bool gaussian_move,
                                   size_t overrelaxation_rate, const string& out_dir) {
        cout << "\n=== Final measurements at T=" << T_final << " ===" << endl;
        
        // Step 1: Estimate autocorrelation time
        cout << "Estimating autocorrelation time..." << endl;
        vector<double> prelim_energies;
        size_t prelim_samples = 10000;
        size_t prelim_interval = 10;
        prelim_energies.reserve(prelim_samples / prelim_interval);
        
        for (size_t i = 0; i < prelim_samples; ++i) {
            metropolis(T_final, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                overrelaxation();
            }
            if (i % prelim_interval == 0) {
                prelim_energies.push_back(total_energy(spins));
            }
        }
        
        AutocorrelationResult acf = compute_autocorrelation(prelim_energies, prelim_interval);
        cout << "  τ_int = " << acf.tau_int << endl;
        cout << "  Sampling interval = " << acf.sampling_interval << " sweeps" << endl;
        
        // Step 2: Equilibrate
        size_t equilibration = 10 * acf.sampling_interval;
        cout << "Equilibrating for " << equilibration << " sweeps..." << endl;
        perform_mc_sweeps(equilibration, T_final, gaussian_move, sigma, overrelaxation_rate, 
                         false, 100);  // boundary_update=false for measurements
        
        // Step 3: Collect samples - now including sublattice magnetizations
        size_t n_samples = 1000;
        size_t n_measure = n_samples * acf.sampling_interval;
        cout << "Collecting " << n_samples << " independent samples..." << endl;
        
        vector<double> energies;
        vector<SpinVector> magnetizations;
        vector<vector<SpinVector>> sublattice_mags;  // NEW: sublattice magnetizations
        energies.reserve(n_samples);
        magnetizations.reserve(n_samples);
        sublattice_mags.reserve(n_samples);
        
        for (size_t i = 0; i < n_measure; ++i) {
            metropolis(T_final, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                overrelaxation();
            }
            
            if (i % acf.sampling_interval == 0) {
                energies.push_back(total_energy(spins));
                magnetizations.push_back(magnetization_global());
                sublattice_mags.push_back(magnetization_sublattice());  // NEW
            }
        }
        
        cout << "Collected " << energies.size() << " samples" << endl;
        
        // Step 4: Compute comprehensive observables with binning analysis
        ThermodynamicObservables obs = compute_thermodynamic_observables(
            energies, sublattice_mags, T_final);
        
        // Print and save results
        print_thermodynamic_observables(obs);
        save_thermodynamic_observables(out_dir, obs);
        
        // Also save raw time series for further analysis
        save_observables(out_dir, energies, magnetizations);
        
        // Save sublattice magnetization time series
        save_sublattice_magnetization_timeseries(out_dir, sublattice_mags);
        
        // Save autocorrelation function
        save_autocorrelation_results(out_dir, acf);
    }

// ---- Lattice::save_sublattice_magnetization_timeseries ----
    void Lattice::save_sublattice_magnetization_timeseries(const string& out_dir,
                                                   const vector<vector<SpinVector>>& sublattice_mags) const {
        ensure_directory_exists(out_dir);
        
        if (sublattice_mags.empty()) return;
        
        size_t n_sublattices = sublattice_mags[0].size();
        
        for (size_t alpha = 0; alpha < n_sublattices; ++alpha) {
            ofstream file(out_dir + "/sublattice_" + std::to_string(alpha) + "_timeseries.txt");
            file << std::scientific << std::setprecision(12);
            file << "# Sublattice " << alpha << " magnetization time series" << endl;
            file << "# Each row: M_0 M_1 ... M_{spin_dim-1}" << endl;
            
            for (const auto& sample : sublattice_mags) {
                const auto& M = sample[alpha];
                for (size_t d = 0; d < static_cast<size_t>(M.size()); ++d) {
                    if (d > 0) file << " ";
                    file << M(d);
                }
                file << "\n";
            }
            file.close();
        }
    }

// ---- Lattice::compute_and_save_observables ----
    void Lattice::compute_and_save_observables(const vector<double>& energies,
                                     const vector<SpinVector>& magnetizations,
                                     double T, const string& out_dir) {
        // Mean energy
        double E_mean = std::accumulate(energies.begin(), energies.end(), 0.0) / energies.size();
        
        // Energy variance
        double E2_mean = 0.0;
        for (double E : energies) {
            E2_mean += E * E;
        }
        E2_mean /= energies.size();
        double var_E = E2_mean - E_mean * E_mean;
        
        // Specific heat
        double C_V = var_E / (T * T * lattice_size);
        
        // Mean magnetization
        SpinVector M_mean = SpinVector::Zero(spin_dim);
        for (const auto& M : magnetizations) {
            M_mean += M;
        }
        M_mean /= magnetizations.size();
        
        cout << "Observables:" << endl;
        cout << "  <E>/N = " << E_mean / lattice_size << endl;
        cout << "  C_V = " << C_V << endl;
        cout << "  |<M>| = " << M_mean.norm() << endl;
        
        // Save to files
        ofstream heat_file(out_dir + "/specific_heat.txt", std::ios::app);
        heat_file << T << " " << C_V << " " << std::sqrt(var_E) / (T * T * lattice_size) << endl;
        heat_file.close();
        
        save_observables(out_dir, energies, magnetizations);
    }

// ---- Lattice::save_autocorrelation_results ----
    void Lattice::save_autocorrelation_results(const string& out_dir, 
                                     const AutocorrelationResult& acf) {
        ofstream acf_file(out_dir + "/autocorrelation.txt");
        acf_file << "# lag autocorrelation" << endl;
        acf_file << "# tau_int = " << acf.tau_int << endl;
        acf_file << "# sampling_interval = " << acf.sampling_interval << endl;
        
        size_t max_output = std::min(size_t(100), acf.correlation_function.size());
        for (size_t lag = 0; lag < max_output; ++lag) {
            acf_file << lag << " " << acf.correlation_function[lag] << "\n";
        }
        acf_file.close();
    }

// ---- Lattice::cluster_annealing ----
    void Lattice::cluster_annealing(double T_start, double T_end, size_t n_anneal,
                          size_t wolff_per_temp, bool use_sw,
                          bool use_ghost_field, double cooling_rate,
                          string out_dir) {
        if (!out_dir.empty()) {
            std::filesystem::create_directories(out_dir);
        }
        
        double T = T_start;
        cout << "Cluster annealing: T=" << T_start << " → " << T_end << endl;
        
        while (T > T_end) {
            for (size_t i = 0; i < n_anneal; ++i) {
                if (use_sw) {
                    swendsen_wang_sweep(T, use_ghost_field);
                } else {
                    wolff_sweep(T, wolff_per_temp, use_ghost_field);
                }
            }
            
            double E = energy_density();
            cout << "T=" << T << ", E/N=" << E << endl;
            
            T *= cooling_rate;
        }
        
        if (!out_dir.empty()) {
            save_spin_config(out_dir + "/spins_T=" + std::to_string(T) + ".txt");
        }
    }

// ---- Lattice::perform_mc_sweeps ----
    double Lattice::perform_mc_sweeps(size_t n_sweeps, double T, bool gaussian_move, 
                            double& sigma, size_t overrelaxation_rate,
                            bool boundary_update,
                            size_t twist_sweep_count,
                            size_t* twist_acc_ptr) {
        double acc_sum = 0.0;
        size_t total_twist_accepted = 0;
        size_t total_twist_attempted = 0;
        
        // Perform MC sweeps
        for (size_t i = 0; i < n_sweeps; ++i) {
            if (overrelaxation_rate > 0) {
                overrelaxation();
                if (i % overrelaxation_rate == 0) {
                    acc_sum += metropolis(T, gaussian_move, sigma);
                }
            } else {
                acc_sum += metropolis(T, gaussian_move, sigma);
            }
        }
        
        // Perform twist boundary updates after MC sweeps are complete
        if (boundary_update) {
            for (size_t j = 0; j < twist_sweep_count; ++j) {
                size_t accepted = metropolis_twist_sweep(T);
                total_twist_accepted += accepted;
                // Count number of dimensions with Ld > 1 as attempts
                size_t n_dims = ((dim1 > 1) ? 1 : 0) + ((dim2 > 1) ? 1 : 0) + ((dim3 > 1) ? 1 : 0);
                total_twist_attempted += n_dims;
            }
            
            // Report twist boundary diagnostics
            if (total_twist_attempted > 0) {
                double twist_acc_rate = double(total_twist_accepted) / double(total_twist_attempted);
                cout << "  [Twist BC: " << total_twist_accepted << "/" << total_twist_attempted 
                     << " accepted (" << std::fixed << std::setprecision(3) << twist_acc_rate * 100.0 
                     << "%)]" << endl;
            }
        }
        
        if (twist_acc_ptr) {
            *twist_acc_ptr = total_twist_accepted;
        }
        
        return acc_sum;
    }

// ---- Lattice::save_observables ----
    void Lattice::save_observables(const string& dir_path,
                         const vector<double>& energies,
                         const vector<SpinVector>& magnetizations) {
        ensure_directory_exists(dir_path);
        
        // Save energy time series
        ofstream energy_file(dir_path + "/energy.txt");
        for (double E : energies) {
            energy_file << E / lattice_size << "\n";
        }
        energy_file.close();
        
        // Save magnetization time series
        ofstream mag_file(dir_path + "/magnetization.txt");
        for (const auto& M : magnetizations) {
            for (int i = 0; i < M.size(); ++i) {
                mag_file << M(i);
                if (i < M.size() - 1) mag_file << " ";
            }
            mag_file << "\n";
        }
        mag_file.close();
    }

