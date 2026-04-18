/**
 * @file mixed_lattice_observables.cpp
 * @brief MixedLattice thermodynamic observables & I/O.
 *
 * Contains the statistical-analysis and output routines that were
 * previously inline in `mixed_lattice.h`: autocorrelation analysis,
 * thermodynamic observable aggregation, per-rank HDF5 writers, plain-
 * text output, sublattice magnetization time series, and the various
 * save_* drivers.
 */

#include "classical_spin/lattice/mixed_lattice.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <cstring>

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

// ---- MixedLattice::compute_autocorrelation ----
    AutocorrelationResult MixedLattice::compute_autocorrelation(const vector<double>& energies, 
                                                   size_t base_interval) {
        return mc::compute_autocorrelation(energies, base_interval);
    }

// ---- MixedLattice::compute_thermodynamic_observables ----
    MixedThermodynamicObservables MixedLattice::compute_thermodynamic_observables(
        const vector<MixedMeasurement>& measurements,
        double T) const {
        
        MixedThermodynamicObservables obs;
        obs.temperature = T;
        size_t n_samples = measurements.size();
        
        if (n_samples == 0) return obs;
        
        size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        
        // 1. Energy observables with binning analysis
        vector<double> energy_per_site(n_samples);
        vector<double> energy_SU2_per_site(n_samples);
        vector<double> energy_SU3_per_site(n_samples);
        
        for (size_t i = 0; i < n_samples; ++i) {
            energy_per_site[i] = measurements[i].energy / double(total_sites);
            energy_SU2_per_site[i] = measurements[i].energy_SU2 / double(lattice_size_SU2);
            energy_SU3_per_site[i] = measurements[i].energy_SU3 / double(lattice_size_SU3);
        }
        
        MixedBinningResult E_result = binning_analysis(energy_per_site);
        MixedBinningResult E_SU2_result = binning_analysis(energy_SU2_per_site);
        MixedBinningResult E_SU3_result = binning_analysis(energy_SU3_per_site);
        
        obs.energy_total.value = E_result.mean;
        obs.energy_total.error = E_result.error;
        obs.energy_SU2.value = E_SU2_result.mean;
        obs.energy_SU2.error = E_SU2_result.error;
        obs.energy_SU3.value = E_SU3_result.mean;
        obs.energy_SU3.error = E_SU3_result.error;
        
        // 2. Specific heat per site with jackknife error estimation
        //    c_V = Var(E) / (T² N²) = Var(E/N) / T²
        {
            double N2 = double(total_sites) * double(total_sites);
            
            // Extract energies for easier access
            vector<double> E_total(n_samples);
            for (size_t i = 0; i < n_samples; ++i) {
                E_total[i] = measurements[i].energy;
            }
            
            // Handle edge case: need at least 2 samples for variance
            if (n_samples < 2) {
                obs.specific_heat.value = 0.0;
                obs.specific_heat.error = 0.0;
            } else {
                // Compute mean first for numerical stability (two-pass algorithm)
                double E_mean = 0.0;
                for (size_t i = 0; i < n_samples; ++i) {
                    E_mean += E_total[i];
                }
                E_mean /= n_samples;
                
                // Compute variance using shifted data for numerical stability
                // Var(E) = <(E - E_mean)²> which avoids catastrophic cancellation
                double var_E = 0.0;
                for (size_t i = 0; i < n_samples; ++i) {
                    double delta = E_total[i] - E_mean;
                    var_E += delta * delta;
                }
                var_E /= n_samples;  // Biased estimator (for heat capacity)
                
                // Ensure non-negative variance (numerical protection)
                var_E = std::max(0.0, var_E);
                obs.specific_heat.value = var_E / (T * T * N2);
                
                // Jackknife error estimation
                // Use at most 100 jackknife blocks, at least 2
                size_t n_jack = std::min(n_samples, size_t(100));
                n_jack = std::max(n_jack, size_t(2));
                size_t block_size = std::max(size_t(1), n_samples / n_jack);
                // Recalculate n_jack based on actual block_size to handle remainders
                n_jack = (n_samples + block_size - 1) / block_size;
                
                vector<double> C_jack(n_jack);
                
                for (size_t j = 0; j < n_jack; ++j) {
                    // Leave out block j: indices [j*block_size, min((j+1)*block_size, n_samples))
                    size_t block_start = j * block_size;
                    size_t block_end = std::min((j + 1) * block_size, n_samples);
                    
                    // Compute jackknife mean (excluding block j)
                    double E_sum = 0.0;
                    size_t count = 0;
                    for (size_t i = 0; i < n_samples; ++i) {
                        if (i < block_start || i >= block_end) {
                            E_sum += E_total[i];
                            ++count;
                        }
                    }
                    
                    if (count < 2) {
                        C_jack[j] = obs.specific_heat.value;  // Fallback
                        continue;
                    }
                    
                    double E_j = E_sum / count;
                    
                    // Compute jackknife variance (excluding block j)
                    double var_j = 0.0;
                    for (size_t i = 0; i < n_samples; ++i) {
                        if (i < block_start || i >= block_end) {
                            double delta = E_total[i] - E_j;
                            var_j += delta * delta;
                        }
                    }
                    var_j /= count;
                    var_j = std::max(0.0, var_j);  // Numerical protection
                    
                    C_jack[j] = var_j / (T * T * N2);
                }
                
                // Compute jackknife error estimate
                double C_mean = 0.0;
                for (double c : C_jack) C_mean += c;
                C_mean /= n_jack;
                
                double C_var = 0.0;
                for (double c : C_jack) C_var += (c - C_mean) * (c - C_mean);
                C_var *= double(n_jack - 1) / double(n_jack);  // Jackknife variance factor
                obs.specific_heat.error = std::sqrt(std::max(0.0, C_var));
            }
        }
        
        // 3. SU(2) sublattice magnetizations with binning analysis
        if (!measurements[0].sublattice_mags_SU2.empty()) {
            size_t n_sublattices_SU2 = measurements[0].sublattice_mags_SU2.size();
            
            obs.sublattice_magnetization_SU2.resize(n_sublattices_SU2);
            obs.energy_sublattice_cross_SU2.resize(n_sublattices_SU2);
            
            for (size_t alpha = 0; alpha < n_sublattices_SU2; ++alpha) {
                obs.sublattice_magnetization_SU2[alpha] = MixedVectorObservable(spin_dim_SU2);
                obs.energy_sublattice_cross_SU2[alpha] = MixedVectorObservable(spin_dim_SU2);
                
                for (size_t d = 0; d < spin_dim_SU2; ++d) {
                    // Extract time series for magnetization component
                    vector<double> M_alpha_d(n_samples);
                    for (size_t i = 0; i < n_samples; ++i) {
                        M_alpha_d[i] = measurements[i].sublattice_mags_SU2[alpha](d);
                    }
                    MixedBinningResult M_result = binning_analysis(M_alpha_d);
                    obs.sublattice_magnetization_SU2[alpha].values[d] = M_result.mean;
                    obs.sublattice_magnetization_SU2[alpha].errors[d] = M_result.error;
                    
                    // Cross term <E * S_α,d> - <E><S_α,d>
                    vector<double> ES_alpha_d(n_samples);
                    for (size_t i = 0; i < n_samples; ++i) {
                        ES_alpha_d[i] = measurements[i].energy * measurements[i].sublattice_mags_SU2[alpha](d);
                    }
                    
                    MixedBinningResult ES_result = binning_analysis(ES_alpha_d);
                    double E_mean = obs.energy_total.value * double(total_sites);
                    double S_mean = M_result.mean;
                    double cross_val = ES_result.mean - E_mean * S_mean;
                    
                    // Jackknife for cross-correlation error
                    size_t n_jack = std::min(n_samples, size_t(100));
                    size_t block_size = n_samples / n_jack;
                    vector<double> cross_jack(n_jack);
                    
                    for (size_t j = 0; j < n_jack; ++j) {
                        double E_sum = 0.0, S_sum = 0.0, ES_sum = 0.0;
                        size_t count = 0;
                        for (size_t i = 0; i < n_samples; ++i) {
                            if (i / block_size != j) {
                                E_sum += measurements[i].energy;
                                S_sum += measurements[i].sublattice_mags_SU2[alpha](d);
                                ES_sum += measurements[i].energy * measurements[i].sublattice_mags_SU2[alpha](d);
                                ++count;
                            }
                        }
                        double E_j = E_sum / count;
                        double S_j = S_sum / count;
                        double ES_j = ES_sum / count;
                        cross_jack[j] = ES_j - E_j * S_j;
                    }
                    
                    double cross_mean = 0.0;
                    for (double c : cross_jack) cross_mean += c;
                    cross_mean /= n_jack;
                    
                    double cross_var = 0.0;
                    for (double c : cross_jack) cross_var += (c - cross_mean) * (c - cross_mean);
                    cross_var *= double(n_jack - 1) / n_jack;
                    
                    obs.energy_sublattice_cross_SU2[alpha].values[d] = cross_val;
                    obs.energy_sublattice_cross_SU2[alpha].errors[d] = std::sqrt(cross_var);
                }
            }
        }
        
        // 4. SU(3) sublattice magnetizations with binning analysis
        if (!measurements[0].sublattice_mags_SU3.empty()) {
            size_t n_sublattices_SU3 = measurements[0].sublattice_mags_SU3.size();
            
            obs.sublattice_magnetization_SU3.resize(n_sublattices_SU3);
            obs.energy_sublattice_cross_SU3.resize(n_sublattices_SU3);
            
            for (size_t alpha = 0; alpha < n_sublattices_SU3; ++alpha) {
                obs.sublattice_magnetization_SU3[alpha] = MixedVectorObservable(spin_dim_SU3);
                obs.energy_sublattice_cross_SU3[alpha] = MixedVectorObservable(spin_dim_SU3);
                
                for (size_t d = 0; d < spin_dim_SU3; ++d) {
                    // Extract time series for magnetization component
                    vector<double> M_alpha_d(n_samples);
                    for (size_t i = 0; i < n_samples; ++i) {
                        M_alpha_d[i] = measurements[i].sublattice_mags_SU3[alpha](d);
                    }
                    MixedBinningResult M_result = binning_analysis(M_alpha_d);
                    obs.sublattice_magnetization_SU3[alpha].values[d] = M_result.mean;
                    obs.sublattice_magnetization_SU3[alpha].errors[d] = M_result.error;
                    
                    // Cross term <E * S_α,d> - <E><S_α,d>
                    vector<double> ES_alpha_d(n_samples);
                    for (size_t i = 0; i < n_samples; ++i) {
                        ES_alpha_d[i] = measurements[i].energy * measurements[i].sublattice_mags_SU3[alpha](d);
                    }
                    
                    MixedBinningResult ES_result = binning_analysis(ES_alpha_d);
                    double E_mean = obs.energy_total.value * double(total_sites);
                    double S_mean = M_result.mean;
                    double cross_val = ES_result.mean - E_mean * S_mean;
                    
                    // Jackknife for cross-correlation error
                    size_t n_jack = std::min(n_samples, size_t(100));
                    size_t block_size = n_samples / n_jack;
                    vector<double> cross_jack(n_jack);
                    
                    for (size_t j = 0; j < n_jack; ++j) {
                        double E_sum = 0.0, S_sum = 0.0, ES_sum = 0.0;
                        size_t count = 0;
                        for (size_t i = 0; i < n_samples; ++i) {
                            if (i / block_size != j) {
                                E_sum += measurements[i].energy;
                                S_sum += measurements[i].sublattice_mags_SU3[alpha](d);
                                ES_sum += measurements[i].energy * measurements[i].sublattice_mags_SU3[alpha](d);
                                ++count;
                            }
                        }
                        double E_j = E_sum / count;
                        double S_j = S_sum / count;
                        double ES_j = ES_sum / count;
                        cross_jack[j] = ES_j - E_j * S_j;
                    }
                    
                    double cross_mean = 0.0;
                    for (double c : cross_jack) cross_mean += c;
                    cross_mean /= n_jack;
                    
                    double cross_var = 0.0;
                    for (double c : cross_jack) cross_var += (c - cross_mean) * (c - cross_mean);
                    cross_var *= double(n_jack - 1) / n_jack;
                    
                    obs.energy_sublattice_cross_SU3[alpha].values[d] = cross_val;
                    obs.energy_sublattice_cross_SU3[alpha].errors[d] = std::sqrt(cross_var);
                }
            }
        }
        
        return obs;
    }

// ---- MixedLattice::save_thermodynamic_observables ----
    void MixedLattice::save_thermodynamic_observables(const string& out_dir,
                                         const MixedThermodynamicObservables& obs) const {
        ensure_directory_exists(out_dir);
        
        // Save main observables summary
        ofstream summary_file(out_dir + "/observables_summary.txt");
        summary_file << "# Mixed Lattice Thermodynamic Observables at T = " << obs.temperature << endl;
        summary_file << "# Format: observable mean error" << endl;
        summary_file << std::scientific << std::setprecision(12);
        
        summary_file << "energy_per_site_total " << obs.energy_total.value << " " << obs.energy_total.error << endl;
        summary_file << "energy_per_site_SU2 " << obs.energy_SU2.value << " " << obs.energy_SU2.error << endl;
        summary_file << "energy_per_site_SU3 " << obs.energy_SU3.value << " " << obs.energy_SU3.error << endl;
        summary_file << "specific_heat " << obs.specific_heat.value << " " << obs.specific_heat.error << endl;
        summary_file.close();
        
        // Save SU(2) sublattice magnetizations
        ofstream sub_mag_SU2_file(out_dir + "/sublattice_magnetization_SU2.txt");
        sub_mag_SU2_file << "# SU(2) sublattice magnetizations <S_α>" << endl;
        sub_mag_SU2_file << "# Format: sublattice component mean error" << endl;
        sub_mag_SU2_file << std::scientific << std::setprecision(12);
        
        for (size_t alpha = 0; alpha < obs.sublattice_magnetization_SU2.size(); ++alpha) {
            const auto& M = obs.sublattice_magnetization_SU2[alpha];
            for (size_t d = 0; d < M.values.size(); ++d) {
                sub_mag_SU2_file << alpha << " " << d << " " 
                                << M.values[d] << " " << M.errors[d] << endl;
            }
        }
        sub_mag_SU2_file.close();
        
        // Save SU(3) sublattice magnetizations
        ofstream sub_mag_SU3_file(out_dir + "/sublattice_magnetization_SU3.txt");
        sub_mag_SU3_file << "# SU(3) sublattice magnetizations <S_α>" << endl;
        sub_mag_SU3_file << "# Format: sublattice component mean error" << endl;
        sub_mag_SU3_file << std::scientific << std::setprecision(12);
        
        for (size_t alpha = 0; alpha < obs.sublattice_magnetization_SU3.size(); ++alpha) {
            const auto& M = obs.sublattice_magnetization_SU3[alpha];
            for (size_t d = 0; d < M.values.size(); ++d) {
                sub_mag_SU3_file << alpha << " " << d << " " 
                                << M.values[d] << " " << M.errors[d] << endl;
            }
        }
        sub_mag_SU3_file.close();
        
        // Save SU(2) energy-sublattice cross terms
        ofstream cross_SU2_file(out_dir + "/energy_sublattice_cross_SU2.txt");
        cross_SU2_file << "# SU(2) energy-sublattice cross correlations <E*S_α> - <E><S_α>" << endl;
        cross_SU2_file << "# Format: sublattice component mean error" << endl;
        cross_SU2_file << std::scientific << std::setprecision(12);
        
        for (size_t alpha = 0; alpha < obs.energy_sublattice_cross_SU2.size(); ++alpha) {
            const auto& cross = obs.energy_sublattice_cross_SU2[alpha];
            for (size_t d = 0; d < cross.values.size(); ++d) {
                cross_SU2_file << alpha << " " << d << " " 
                              << cross.values[d] << " " << cross.errors[d] << endl;
            }
        }
        cross_SU2_file.close();
        
        // Save SU(3) energy-sublattice cross terms
        ofstream cross_SU3_file(out_dir + "/energy_sublattice_cross_SU3.txt");
        cross_SU3_file << "# SU(3) energy-sublattice cross correlations <E*S_α> - <E><S_α>" << endl;
        cross_SU3_file << "# Format: sublattice component mean error" << endl;
        cross_SU3_file << std::scientific << std::setprecision(12);
        
        for (size_t alpha = 0; alpha < obs.energy_sublattice_cross_SU3.size(); ++alpha) {
            const auto& cross = obs.energy_sublattice_cross_SU3[alpha];
            for (size_t d = 0; d < cross.values.size(); ++d) {
                cross_SU3_file << alpha << " " << d << " " 
                              << cross.values[d] << " " << cross.errors[d] << endl;
            }
        }
        cross_SU3_file.close();
    }

// ---- MixedLattice::print_thermodynamic_observables ----
    void MixedLattice::print_thermodynamic_observables(const MixedThermodynamicObservables& obs) const {
        cout << "\n=== Mixed Lattice Thermodynamic Observables at T = " << obs.temperature << " ===" << endl;
        cout << std::scientific << std::setprecision(6);
        
        cout << "<E>/N_total = " << obs.energy_total.value << " ± " << obs.energy_total.error << endl;
        cout << "<E>/N_SU2   = " << obs.energy_SU2.value << " ± " << obs.energy_SU2.error << endl;
        cout << "<E>/N_SU3   = " << obs.energy_SU3.value << " ± " << obs.energy_SU3.error << endl;
        cout << "C_V         = " << obs.specific_heat.value << " ± " << obs.specific_heat.error << endl;
        
        cout << "\nSU(2) Sublattice magnetizations:" << endl;
        for (size_t alpha = 0; alpha < obs.sublattice_magnetization_SU2.size(); ++alpha) {
            cout << "  Sublattice " << alpha << ": (";
            const auto& M = obs.sublattice_magnetization_SU2[alpha];
            for (size_t d = 0; d < M.values.size(); ++d) {
                if (d > 0) cout << ", ";
                cout << M.values[d] << "±" << M.errors[d];
            }
            cout << ")" << endl;
        }
        
        cout << "\nSU(3) Sublattice magnetizations:" << endl;
        for (size_t alpha = 0; alpha < obs.sublattice_magnetization_SU3.size(); ++alpha) {
            cout << "  Sublattice " << alpha << ": (";
            const auto& M = obs.sublattice_magnetization_SU3[alpha];
            for (size_t d = 0; d < M.values.size(); ++d) {
                if (d > 0) cout << ", ";
                cout << M.values[d] << "±" << M.errors[d];
            }
            cout << ")" << endl;
        }
        
        cout << "\nSU(2) Energy-sublattice cross correlations:" << endl;
        for (size_t alpha = 0; alpha < obs.energy_sublattice_cross_SU2.size(); ++alpha) {
            cout << "  Sublattice " << alpha << ": (";
            const auto& cross = obs.energy_sublattice_cross_SU2[alpha];
            for (size_t d = 0; d < cross.values.size(); ++d) {
                if (d > 0) cout << ", ";
                cout << cross.values[d] << "±" << cross.errors[d];
            }
            cout << ")" << endl;
        }
        
        cout << "\nSU(3) Energy-sublattice cross correlations:" << endl;
        for (size_t alpha = 0; alpha < obs.energy_sublattice_cross_SU3.size(); ++alpha) {
            cout << "  Sublattice " << alpha << ": (";
            const auto& cross = obs.energy_sublattice_cross_SU3[alpha];
            for (size_t d = 0; d < cross.values.size(); ++d) {
                if (d > 0) cout << ", ";
                cout << cross.values[d] << "±" << cross.errors[d];
            }
            cout << ")" << endl;
        }
    }

// ---- MixedLattice::save_thermodynamic_observables_hdf5 ----
    void MixedLattice::save_thermodynamic_observables_hdf5(const string& out_dir,
                                              const MixedThermodynamicObservables& obs,
                                              const vector<double>& energies,
                                              const vector<pair<SpinVector, SpinVector>>& magnetizations,
                                              const vector<MixedMeasurement>& measurements,
                                              size_t n_anneal,
                                              size_t n_measure,
                                              size_t probe_rate,
                                              size_t swap_rate,
                                              size_t overrelaxation_rate,
                                              double acceptance_rate,
                                              double swap_acceptance_rate) const {
#ifdef HDF5_ENABLED
        ensure_directory_exists(out_dir);
        
        string filename = out_dir + "/parallel_tempering_data.h5";
        size_t n_samples = energies.size();
        
        // Extract sublattice magnetizations from measurements
        vector<vector<SpinVector>> sublattice_mags_SU2(n_samples);
        vector<vector<SpinVector>> sublattice_mags_SU3(n_samples);
        
        for (size_t i = 0; i < n_samples; ++i) {
            sublattice_mags_SU2[i] = measurements[i].sublattice_mags_SU2;
            sublattice_mags_SU3[i] = measurements[i].sublattice_mags_SU3;
        }
        
        // Create HDF5 writer
        HDF5MixedPTWriter writer(filename, obs.temperature, 
                                 lattice_size_SU2, lattice_size_SU3,
                                 spin_dim_SU2, spin_dim_SU3,
                                 N_atoms_SU2, N_atoms_SU3,
                                 n_samples, n_anneal, n_measure, probe_rate, swap_rate,
                                 overrelaxation_rate, acceptance_rate, swap_acceptance_rate);
        
        // Write time series data
        writer.write_timeseries(energies, magnetizations, sublattice_mags_SU2, sublattice_mags_SU3);
        
        // Prepare observable data in format expected by writer
        vector<vector<double>> sublattice_mag_SU2_means(N_atoms_SU2);
        vector<vector<double>> sublattice_mag_SU2_errors(N_atoms_SU2);
        vector<vector<double>> sublattice_mag_SU3_means(N_atoms_SU3);
        vector<vector<double>> sublattice_mag_SU3_errors(N_atoms_SU3);
        vector<vector<double>> energy_cross_SU2_means(N_atoms_SU2);
        vector<vector<double>> energy_cross_SU2_errors(N_atoms_SU2);
        vector<vector<double>> energy_cross_SU3_means(N_atoms_SU3);
        vector<vector<double>> energy_cross_SU3_errors(N_atoms_SU3);
        
        for (size_t alpha = 0; alpha < N_atoms_SU2; ++alpha) {
            sublattice_mag_SU2_means[alpha] = obs.sublattice_magnetization_SU2[alpha].values;
            sublattice_mag_SU2_errors[alpha] = obs.sublattice_magnetization_SU2[alpha].errors;
            energy_cross_SU2_means[alpha] = obs.energy_sublattice_cross_SU2[alpha].values;
            energy_cross_SU2_errors[alpha] = obs.energy_sublattice_cross_SU2[alpha].errors;
        }
        
        for (size_t alpha = 0; alpha < N_atoms_SU3; ++alpha) {
            sublattice_mag_SU3_means[alpha] = obs.sublattice_magnetization_SU3[alpha].values;
            sublattice_mag_SU3_errors[alpha] = obs.sublattice_magnetization_SU3[alpha].errors;
            energy_cross_SU3_means[alpha] = obs.energy_sublattice_cross_SU3[alpha].values;
            energy_cross_SU3_errors[alpha] = obs.energy_sublattice_cross_SU3[alpha].errors;
        }
        
        // Write observables
        writer.write_observables(obs.energy_total.value, obs.energy_total.error,
                                obs.energy_SU2.value, obs.energy_SU2.error,
                                obs.energy_SU3.value, obs.energy_SU3.error,
                                obs.specific_heat.value, obs.specific_heat.error,
                                sublattice_mag_SU2_means, sublattice_mag_SU2_errors,
                                sublattice_mag_SU3_means, sublattice_mag_SU3_errors,
                                energy_cross_SU2_means, energy_cross_SU2_errors,
                                energy_cross_SU3_means, energy_cross_SU3_errors);
        
        writer.close();
#else
        std::cerr << "Warning: HDF5 support not enabled. Cannot save HDF5 output." << std::endl;
        std::cerr << "Compile with -DHDF5_ENABLED to enable HDF5 output." << std::endl;
#endif
    }

// ---- MixedLattice::save_heat_capacity_hdf5 ----
    void MixedLattice::save_heat_capacity_hdf5(const string& out_dir,
                                  const vector<double>& temperatures,
                                  const vector<double>& heat_capacity,
                                  const vector<double>& dHeat) const {
#ifdef HDF5_ENABLED
        ensure_directory_exists(out_dir);
        
        string filename = out_dir + "/parallel_tempering_aggregated.h5";
        size_t n_temps = temperatures.size();
        
        // Create HDF5 file
        H5::H5File file(filename, H5F_ACC_TRUNC);
        
        // Create main data group
        H5::Group data_group = file.createGroup("/temperature_scan");
        H5::Group metadata_group = file.createGroup("/metadata");
        
        // Write metadata
        std::time_t now = std::time(nullptr);
        char time_str[100];
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        
        H5::DataSpace scalar_space(H5S_SCALAR);
        
        // Number of temperatures
        H5::Attribute n_temps_attr = metadata_group.createAttribute(
            "n_temperatures", H5::PredType::NATIVE_HSIZE, scalar_space);
        n_temps_attr.write(H5::PredType::NATIVE_HSIZE, &n_temps);
        
        // Timestamp
        H5::StrType str_type(H5::PredType::C_S1, strlen(time_str) + 1);
        H5::Attribute time_attr = metadata_group.createAttribute(
            "creation_time", str_type, scalar_space);
        time_attr.write(str_type, time_str);
        
        // Version info
        std::string version = "ClassicalSpin_Cpp v1.0";
        H5::StrType version_type(H5::PredType::C_S1, version.size() + 1);
        H5::Attribute version_attr = metadata_group.createAttribute(
            "code_version", version_type, scalar_space);
        version_attr.write(version_type, version.c_str());
        
        std::string format = "HDF5_MixedPT_Aggregated_v1.0";
        H5::StrType format_type(H5::PredType::C_S1, format.size() + 1);
        H5::Attribute format_attr = metadata_group.createAttribute(
            "file_format", format_type, scalar_space);
        format_attr.write(format_type, format.c_str());
        
        // Write temperature array
        hsize_t dims[1] = {n_temps};
        H5::DataSpace dataspace(1, dims);
        
        H5::DataSet temp_dataset = data_group.createDataSet(
            "temperature", H5::PredType::NATIVE_DOUBLE, dataspace);
        temp_dataset.write(temperatures.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write heat capacity array
        H5::DataSet heat_dataset = data_group.createDataSet(
            "specific_heat", H5::PredType::NATIVE_DOUBLE, dataspace);
        heat_dataset.write(heat_capacity.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write heat capacity error array
        H5::DataSet dheat_dataset = data_group.createDataSet(
            "specific_heat_error", H5::PredType::NATIVE_DOUBLE, dataspace);
        dheat_dataset.write(dHeat.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Close everything
        temp_dataset.close();
        heat_dataset.close();
        dheat_dataset.close();
        data_group.close();
        metadata_group.close();
        file.close();
#else
        std::cerr << "Warning: HDF5 support not enabled. Cannot save HDF5 output." << std::endl;
        std::cerr << "Compile with -DHDF5_ENABLED to enable HDF5 output." << std::endl;
#endif
    }

// ---- MixedLattice::save_sublattice_magnetization_timeseries ----
    void MixedLattice::save_sublattice_magnetization_timeseries(const string& out_dir,
                                                   const vector<MixedMeasurement>& measurements) const {
        ensure_directory_exists(out_dir);
        
        if (measurements.empty()) return;
        
        // SU(2) sublattice time series
        size_t n_sublattices_SU2 = measurements[0].sublattice_mags_SU2.size();
        for (size_t alpha = 0; alpha < n_sublattices_SU2; ++alpha) {
            ofstream file(out_dir + "/sublattice_SU2_" + std::to_string(alpha) + "_timeseries.txt");
            file << std::scientific << std::setprecision(12);
            file << "# SU(2) Sublattice " << alpha << " magnetization time series" << endl;
            
            for (const auto& m : measurements) {
                const auto& M = m.sublattice_mags_SU2[alpha];
                for (size_t d = 0; d < static_cast<size_t>(M.size()); ++d) {
                    if (d > 0) file << " ";
                    file << M(d);
                }
                file << "\n";
            }
            file.close();
        }
        
        // SU(3) sublattice time series
        size_t n_sublattices_SU3 = measurements[0].sublattice_mags_SU3.size();
        for (size_t alpha = 0; alpha < n_sublattices_SU3; ++alpha) {
            ofstream file(out_dir + "/sublattice_SU3_" + std::to_string(alpha) + "_timeseries.txt");
            file << std::scientific << std::setprecision(12);
            file << "# SU(3) Sublattice " << alpha << " magnetization time series" << endl;
            
            for (const auto& m : measurements) {
                const auto& M = m.sublattice_mags_SU3[alpha];
                for (size_t d = 0; d < static_cast<size_t>(M.size()); ++d) {
                    if (d > 0) file << " ";
                    file << M(d);
                }
                file << "\n";
            }
            file.close();
        }
    }

// ---- MixedLattice::compute_and_save_observables ----
    void MixedLattice::compute_and_save_observables(const vector<double>& energies,
                                     const vector<pair<SpinVector, SpinVector>>& magnetizations,
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
        
        // Specific heat (per total site)
        size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        double C_V = var_E / (T * T * total_sites);
        
        // Mean magnetizations
        SpinVector M_mean_SU2 = SpinVector::Zero(spin_dim_SU2);
        SpinVector M_mean_SU3 = SpinVector::Zero(spin_dim_SU3);
        for (const auto& M_pair : magnetizations) {
            M_mean_SU2 += M_pair.first;
            M_mean_SU3 += M_pair.second;
        }
        M_mean_SU2 /= magnetizations.size();
        M_mean_SU3 /= magnetizations.size();
        
        cout << "Observables:" << endl;
        cout << "  <E>/N = " << E_mean / total_sites << endl;
        cout << "  C_V = " << C_V << endl;
        cout << "  |<M_SU2>| = " << M_mean_SU2.norm() << endl;
        cout << "  |<M_SU3>| = " << M_mean_SU3.norm() << endl;
        
        // Save to files
        ofstream heat_file(out_dir + "/specific_heat.txt", std::ios::app);
        heat_file << T << " " << C_V << " " << std::sqrt(var_E) / (T * T * total_sites) << endl;
        heat_file.close();
        
        save_observables(out_dir, energies, magnetizations);
    }

// ---- MixedLattice::save_observables ----
    void MixedLattice::save_observables(const string& dir_path,
                         const vector<double>& energies,
                         const vector<pair<SpinVector, SpinVector>>& magnetizations) {
        ensure_directory_exists(dir_path);
        
        size_t total_sites = lattice_size_SU2 + lattice_size_SU3;
        
        // Save energy time series
        ofstream energy_file(dir_path + "/energy.txt");
        for (double E : energies) {
            energy_file << E / total_sites << "\n";
        }
        energy_file.close();
        
        // Save magnetization time series
        ofstream mag_su2_file(dir_path + "/magnetization_SU2.txt");
        ofstream mag_su3_file(dir_path + "/magnetization_SU3.txt");
        for (const auto& M_pair : magnetizations) {
            const auto& M_SU2 = M_pair.first;
            const auto& M_SU3 = M_pair.second;
            
            for (int i = 0; i < M_SU2.size(); ++i) {
                mag_su2_file << M_SU2(i);
                if (i < M_SU2.size() - 1) mag_su2_file << " ";
            }
            mag_su2_file << "\n";
            
            for (int i = 0; i < M_SU3.size(); ++i) {
                mag_su3_file << M_SU3(i);
                if (i < M_SU3.size() - 1) mag_su3_file << " ";
            }
            mag_su3_file << "\n";
        }
        mag_su2_file.close();
        mag_su3_file.close();
    }

// ---- MixedLattice::save_autocorrelation_results ----
    void MixedLattice::save_autocorrelation_results(const string& out_dir, 
                                     const AutocorrelationResult& acf) {
        ensure_directory_exists(out_dir);
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

