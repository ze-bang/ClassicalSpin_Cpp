/**
 * @file mixed_lattice_mc.cpp
 * @brief MixedLattice Monte Carlo driver methods.
 *
 * Hosts the heavy MC drivers previously inline in `mixed_lattice.h`:
 * simulated_annealing, perform_final_measurements, greedy_quench,
 * perform_mc_sweeps. The hot inner loops — metropolis(),
 * overrelaxation(), deterministic_sweep() and their interleaved
 * variants — remain in the header so the optimizer can keep inlining
 * across site loops.
 */

#include "classical_spin/lattice/mixed_lattice.h"

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

// ---- MixedLattice::greedy_quench ----
    void MixedLattice::greedy_quench(double rel_tol, size_t max_sweeps) {
        double E_prev = total_energy();
        for (size_t s = 0; s < max_sweeps; ++s) {
            deterministic_sweep();
            double E = total_energy();
            if (fabs(E - E_prev) <= rel_tol * (fabs(E_prev) + 1e-18)) {
                cout << "Greedy quench converged after " << s + 1 << " sweeps" << endl;
                break;
            }
            E_prev = E;
        }
    }

// ---- MixedLattice::simulated_annealing ----
    void MixedLattice::simulated_annealing(double T_start, double T_end, size_t n_anneal,
                            bool gaussian_move,
                            double cooling_rate,
                            string out_dir,
                            bool save_observables,
                            bool T_zero,
                            size_t n_deterministics,
                            size_t twist_sweep_count) {
        
        // Setup output directory
        ensure_directory_exists(out_dir);
        
        double T = T_start;
        double sigma = 1000.0;
        
        cout << "Starting mixed lattice simulated annealing: T=" << T_start << " → " << T_end << endl;
        if (T_zero) {
            cout << "T=0 mode enabled: will perform " << n_deterministics << " deterministic sweeps at T=0" << endl;
        }
        
        size_t temp_step = 0;
        while (T > T_end) {
            // Perform sweeps at this temperature
            double acc_sum = perform_mc_sweeps(n_anneal, T, gaussian_move, sigma);
            
            // Calculate acceptance rate
            double acceptance = acc_sum / double(n_anneal);
            
            // Progress report
            if (temp_step % 10 == 0 || T <= T_end * 1.5) {
                double E = energy_density();
                SpinVector M_SU2 = magnetization_SU2();
                SpinVector M_SU3 = magnetization_SU3();
                cout << "T=" << std::scientific << T << ", E/N=" << E 
                     << ", |M_SU2|=" << M_SU2.norm()
                     << ", |M_SU3|=" << M_SU3.norm()
                     << ", acc=" << std::fixed << acceptance;
                if (gaussian_move) cout << ", σ=" << sigma;
                cout << endl;
            }
            
            // Adaptive sigma adjustment for gaussian moves
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
        
        double E_total = total_energy();
        double E_SU2 = total_energy_SU2();
        double E_SU3 = total_energy_SU3();
        size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        cout << "Final energy density: " << E_total / total_sites << endl;
        cout << "  Total Energy:     " << E_total << endl;
        cout << "  SU2 Energy:       " << E_SU2 << " (E/N_SU2 = " << E_SU2 / lattice_size_SU2 << ")" << endl;
        cout << "  SU3 Energy:       " << E_SU3 << " (E/N_SU3 = " << E_SU3 / lattice_size_SU3 << ")" << endl;
        
        // Save spin config after annealing (before deterministic sweeps)
        if (!out_dir.empty()) {
            save_spin_config_to_dir(out_dir, "spins_T=" + std::to_string(T_end));
            save_energy_to_dir(out_dir, "energy_T=" + std::to_string(T_end));
            save_positions_to_dir(out_dir);
        }
        
        // Final measurements if requested (before deterministic sweeps)
        if (save_observables && !out_dir.empty()) {
            perform_final_measurements(T_end, sigma, gaussian_move, out_dir);
        }
        
        // T=0 deterministic sweeps if requested
        if (T_zero && n_deterministics > 0) {
            cout << "\nPerforming " << n_deterministics << " deterministic sweeps at T=0..." << endl;
            for (size_t sweep = 0; sweep < n_deterministics; ++sweep) {
                deterministic_sweep();
                
                if (sweep % 100 == 0 || sweep == n_deterministics - 1) {
                    double E = energy_density();
                    SpinVector M_SU2 = magnetization_SU2();
                    SpinVector M_SU3 = magnetization_SU3();
                    cout << "Deterministic sweep " << sweep << "/" << n_deterministics 
                         << ", E/N=" << E 
                         << ", |M_SU2|=" << M_SU2.norm()
                         << ", |M_SU3|=" << M_SU3.norm() << endl;
                }
            }
            double E_total_final = total_energy();
            double E_SU2_final = total_energy_SU2();
            double E_SU3_final = total_energy_SU3();
            cout << "Deterministic sweeps completed. Final energy: " << E_total_final / (lattice_size_SU2 + lattice_size_SU3) << endl;
            cout << "  Total Energy:     " << E_total_final << endl;
            cout << "  SU2 Energy:       " << E_SU2_final << " (E/N_SU2 = " << E_SU2_final / lattice_size_SU2 << ")" << endl;
            cout << "  SU3 Energy:       " << E_SU3_final << " (E/N_SU3 = " << E_SU3_final / lattice_size_SU3 << ")" << endl;
            // Save final configuration
            if (!out_dir.empty()) {
                save_spin_config_to_dir(out_dir, "spins_T=0");
                save_energy_to_dir(out_dir, "energy_T=0");
            }
        }
    }

// ---- MixedLattice::perform_final_measurements ----
    void MixedLattice::perform_final_measurements(double T_final, double sigma, bool gaussian_move,
                                   const string& out_dir) {
        cout << "\n=== Final measurements at T=" << T_final << " ===" << endl;
        
        // Step 1: Estimate autocorrelation time
        cout << "Estimating autocorrelation time..." << endl;
        vector<double> prelim_energies;
        size_t prelim_samples = 10000;
        size_t prelim_interval = 10;
        prelim_energies.reserve(prelim_samples / prelim_interval);
        
        for (size_t i = 0; i < prelim_samples; ++i) {
            metropolis(T_final, gaussian_move, sigma);
            if (i % prelim_interval == 0) {
                prelim_energies.push_back(total_energy());
            }
        }
        
        AutocorrelationResult acf = compute_autocorrelation(prelim_energies, prelim_interval);
        cout << "  τ_int = " << acf.tau_int << endl;
        cout << "  Sampling interval = " << acf.sampling_interval << " sweeps" << endl;
        
        // Step 2: Equilibrate
        size_t equilibration = 10 * acf.sampling_interval;
        cout << "Equilibrating for " << equilibration << " sweeps..." << endl;
        perform_mc_sweeps(equilibration, T_final, gaussian_move, sigma);
        
        // Step 3: Collect samples with comprehensive observables
        size_t n_samples = 1000;
        size_t n_measure = n_samples * acf.sampling_interval;
        cout << "Collecting " << n_samples << " independent samples..." << endl;
        
        vector<double> energies;
        vector<pair<SpinVector, SpinVector>> magnetizations;
        vector<MixedMeasurement> measurements;  // NEW: comprehensive measurements
        energies.reserve(n_samples);
        magnetizations.reserve(n_samples);
        measurements.reserve(n_samples);
        
        for (size_t i = 0; i < n_measure; ++i) {
            metropolis(T_final, gaussian_move, sigma);
            
            if (i % acf.sampling_interval == 0) {
                energies.push_back(total_energy());
                magnetizations.push_back({magnetization_SU2(), magnetization_SU3()});
                measurements.push_back(measure_all_observables());  // NEW
            }
        }
        
        cout << "Collected " << energies.size() << " samples" << endl;
        
        // Step 4: Compute comprehensive observables with binning analysis
        MixedThermodynamicObservables obs = compute_thermodynamic_observables(measurements, T_final);
        
        // Print and save results
        print_thermodynamic_observables(obs);
        save_thermodynamic_observables(out_dir, obs);
        
        // Also save raw time series for further analysis
        save_observables(out_dir, energies, magnetizations);
        
        // Save sublattice magnetization time series
        save_sublattice_magnetization_timeseries(out_dir, measurements);
        
        // Save autocorrelation function
        save_autocorrelation_results(out_dir, acf);
    }

// ---- MixedLattice::perform_mc_sweeps ----
    double MixedLattice::perform_mc_sweeps(size_t n_sweeps, double T, bool gaussian_move, 
                            double& sigma, size_t overrelaxation_rate,
                            bool interleaved) {
        double acc_sum = 0.0;
        
        // Check if we have mixed interactions - if so, prefer interleaved mode
        const bool has_mixed = (num_bi_SU2_SU3 > 0 || num_tri_SU2_SU3 > 0);
        const bool use_interleaved = interleaved && has_mixed;
        
        for (size_t i = 0; i < n_sweeps; ++i) {
            if (overrelaxation_rate > 0) {
                if (use_interleaved) {
                    overrelaxation_interleaved();
                } else {
                    overrelaxation();
                }
                if (i % overrelaxation_rate == 0) {
                    if (use_interleaved) {
                        acc_sum += metropolis_interleaved(T, gaussian_move, sigma);
                    } else {
                        acc_sum += metropolis(T, gaussian_move, sigma);
                    }
                }
            } else {
                if (use_interleaved) {
                    acc_sum += metropolis_interleaved(T, gaussian_move, sigma);
                } else {
                    acc_sum += metropolis(T, gaussian_move, sigma);
                }
            }
        }
        
        return acc_sum;
    }

