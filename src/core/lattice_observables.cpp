/**
 * @file lattice_observables.cpp
 * @brief Statistical analysis and observable-output routines.
 *
 * Hosts the autocorrelation / binning / thermodynamic-observable
 * machinery and the matching HDF5 + text writers that were inline in
 * `lattice.h`. These methods pull in HDF5, `<algorithm>`, and a lot of
 * formatting code; keeping them out of the header significantly
 * reduces the include cost for every TU that just needs the simulation
 * driver types.
 */

#include "classical_spin/lattice/lattice.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

// ---- Lattice::compute_autocorrelation ----
    AutocorrelationResult Lattice::compute_autocorrelation(const vector<double>& energies, 
                                                   size_t base_interval) {
        AutocorrelationResult result;
        
        if (energies.size() < 10) {
            result.tau_int = 1.0;
            result.sampling_interval = base_interval;
            return result;
        }
        
        size_t N = energies.size();
        
        // Calculate mean and variance
        double mean = std::accumulate(energies.begin(), energies.end(), 0.0) / N;
        double variance = 0.0;
        for (double e : energies) {
            variance += (e - mean) * (e - mean);
        }
        variance /= N;
        
        if (variance < 1e-20) {
            result.tau_int = 1.0;
            result.sampling_interval = base_interval;
            return result;
        }
        
        // Compute autocorrelation function
        size_t max_lag = std::min(N / 4, size_t(1000));
        result.correlation_function.resize(max_lag);
        
        for (size_t lag = 0; lag < max_lag; ++lag) {
            double corr = 0.0;
            size_t count = N - lag;
            for (size_t i = 0; i < count; ++i) {
                corr += (energies[i] - mean) * (energies[i + lag] - mean);
            }
            result.correlation_function[lag] = corr / (count * variance);
        }
        
        // Calculate integrated autocorrelation time using Sokal's self-consistent
        // window method: sum until lag >= C * tau_int (C ~ 6 is standard).
        // This avoids the bias from a hard cutoff and adapts to slow decorrelation.
        // Reference: A.D. Sokal, "Monte Carlo Methods in Statistical Mechanics"
        //            Lecture Notes, Cours de Troisième Cycle de la Physique en Suisse Romande (1989)
        constexpr double sokal_C = 6.0;
        result.tau_int = 0.5;
        for (size_t lag = 1; lag < max_lag; ++lag) {
            // Also stop if ACF goes negative (noise-dominated regime)
            if (result.correlation_function[lag] < 0.0) break;
            result.tau_int += result.correlation_function[lag];
            // Sokal self-consistent window: stop when lag >= C * tau_int
            if (static_cast<double>(lag) >= sokal_C * result.tau_int) break;
        }
        
        // Warn if tau_int is large relative to time series length
        // Reliable estimation requires N >> 2*tau_int; if not, tau_int may be underestimated
        if (2.0 * result.tau_int > 0.1 * N) {
            cout << "[WARNING] Autocorrelation time τ_int=" << result.tau_int 
                 << " samples is large relative to time series length N=" << N
                 << ". Estimate may be unreliable — consider longer preliminary runs." << endl;
        }
        
        // Determine sampling interval in MC sweeps:
        // Each sample in the time series is separated by base_interval sweeps,
        // so tau_int (in sample units) corresponds to tau_int * base_interval sweeps.
        // We require at least 2*tau_int spacing for approximately independent samples.
        size_t tau_int_sweeps = static_cast<size_t>(std::ceil(result.tau_int)) * base_interval;
        result.sampling_interval = std::max(size_t(2) * tau_int_sweeps, size_t(100));
        
        return result;
    }

// ---- Lattice::compute_thermodynamic_observables ----
    ThermodynamicObservables Lattice::compute_thermodynamic_observables(
        const vector<double>& energies,
        const vector<vector<SpinVector>>& sublattice_mags,
        double T) const {
        
        ThermodynamicObservables obs;
        obs.temperature = T;
        size_t n_samples = energies.size();
        
        if (n_samples == 0) return obs;
        
        // 1. Energy per site with binning analysis
        vector<double> energy_per_site(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            energy_per_site[i] = energies[i] / double(lattice_size);
        }
        BinningResult E_result = binning_analysis(energy_per_site);
        obs.energy.value = E_result.mean;
        obs.energy.error = E_result.error;
        
        // 1b. Total magnetization per site with binning analysis
        //     M = (1/N) Σ_α Σ_i∈α S_i = (1/N_atoms) Σ_α M_α
        if (!sublattice_mags.empty() && !sublattice_mags[0].empty()) {
            size_t sdim = sublattice_mags[0][0].size();
            size_t n_sublattices = sublattice_mags[0].size();
            obs.magnetization = VectorObservable(sdim);
            
            for (size_t d = 0; d < sdim; ++d) {
                vector<double> M_total_d(n_samples);
                for (size_t i = 0; i < n_samples; ++i) {
                    double M_sum = 0.0;
                    for (size_t alpha = 0; alpha < n_sublattices; ++alpha) {
                        M_sum += sublattice_mags[i][alpha](d);
                    }
                    M_total_d[i] = M_sum / double(n_sublattices);
                }
                BinningResult M_result = binning_analysis(M_total_d);
                obs.magnetization.values[d] = M_result.mean;
                obs.magnetization.errors[d] = M_result.error;
            }
        }
        
        // 2. Specific heat per site: c_V = Var(E) / (T² N²) = Var(E/N) / T²
        //    Since E is extensive (E ~ N), Var(E) ~ N², so c_V ~ O(1)
        //    Error propagation via jackknife on binned data
        {
            double N2 = double(lattice_size) * double(lattice_size);
            
            // Handle edge case: need at least 2 samples for variance
            if (n_samples < 2) {
                obs.specific_heat.value = 0.0;
                obs.specific_heat.error = 0.0;
            } else {
                // Compute mean first for numerical stability (two-pass algorithm)
                double E_mean = 0.0;
                for (size_t i = 0; i < n_samples; ++i) {
                    E_mean += energies[i];
                }
                E_mean /= n_samples;
                
                // Compute variance using shifted data for numerical stability
                // Var(E) = <(E - E_mean)²> which avoids catastrophic cancellation
                double var_E = 0.0;
                for (size_t i = 0; i < n_samples; ++i) {
                    double delta = energies[i] - E_mean;
                    var_E += delta * delta;
                }
                var_E /= n_samples;  // Biased estimator (for heat capacity)
                
                // Ensure non-negative variance (numerical protection)
                var_E = std::max(0.0, var_E);
                obs.specific_heat.value = var_E / (T * T * N2);
                
                // Jackknife error estimation for specific heat
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
                            E_sum += energies[i];
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
                            double delta = energies[i] - E_j;
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
        
        // 3. Sublattice magnetizations with binning analysis
        if (!sublattice_mags.empty() && !sublattice_mags[0].empty()) {
            size_t n_sublattices = sublattice_mags[0].size();
            size_t sdim = sublattice_mags[0][0].size();
            
            obs.sublattice_magnetization.resize(n_sublattices);
            
            for (size_t alpha = 0; alpha < n_sublattices; ++alpha) {
                obs.sublattice_magnetization[alpha] = VectorObservable(sdim);
                
                // Extract time series for each component
                for (size_t d = 0; d < sdim; ++d) {
                    vector<double> M_alpha_d(n_samples);
                    for (size_t i = 0; i < n_samples; ++i) {
                        M_alpha_d[i] = sublattice_mags[i][alpha](d);
                    }
                    BinningResult M_result = binning_analysis(M_alpha_d);
                    obs.sublattice_magnetization[alpha].values[d] = M_result.mean;
                    obs.sublattice_magnetization[alpha].errors[d] = M_result.error;
                }
            }
            
            // 4. Cross term <E * S_α> - <E><S_α> for each sublattice
            obs.energy_sublattice_cross.resize(n_sublattices);
            
            for (size_t alpha = 0; alpha < n_sublattices; ++alpha) {
                obs.energy_sublattice_cross[alpha] = VectorObservable(sdim);
                
                for (size_t d = 0; d < sdim; ++d) {
                    // Compute <E * S_α,d>
                    vector<double> ES_alpha_d(n_samples);
                    for (size_t i = 0; i < n_samples; ++i) {
                        ES_alpha_d[i] = energies[i] * sublattice_mags[i][alpha](d);
                    }
                    
                    BinningResult ES_result = binning_analysis(ES_alpha_d);
                    
                    // Cross correlation = <ES> - <E><S>
                    double E_mean = obs.energy.value * double(lattice_size);
                    double S_mean = obs.sublattice_magnetization[alpha].values[d];
                    double cross_val = ES_result.mean - E_mean * S_mean;
                    
                    // Error propagation: δ(AB - CD) ≈ sqrt(δ(AB)² + (B δA)² + (A δB)²)
                    // Simplified: use jackknife for proper covariance handling
                    size_t n_jack = std::min(n_samples, size_t(100));
                    size_t block_size = n_samples / n_jack;
                    vector<double> cross_jack(n_jack);
                    
                    for (size_t j = 0; j < n_jack; ++j) {
                        double E_sum = 0.0, S_sum = 0.0, ES_sum = 0.0;
                        size_t count = 0;
                        for (size_t i = 0; i < n_samples; ++i) {
                            if (i / block_size != j) {
                                E_sum += energies[i];
                                S_sum += sublattice_mags[i][alpha](d);
                                ES_sum += energies[i] * sublattice_mags[i][alpha](d);
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
                    
                    obs.energy_sublattice_cross[alpha].values[d] = cross_val;
                    obs.energy_sublattice_cross[alpha].errors[d] = std::sqrt(cross_var);
                }
            }
        }
        
        return obs;
    }

// ---- Lattice::save_thermodynamic_observables ----
    void Lattice::save_thermodynamic_observables(const string& out_dir,
                                         const ThermodynamicObservables& obs) const {
        ensure_directory_exists(out_dir);
        
        // Save main observables summary
        ofstream summary_file(out_dir + "/observables_summary.txt");
        summary_file << "# Thermodynamic Observables at T = " << obs.temperature << endl;
        summary_file << "# Format: observable mean error" << endl;
        summary_file << std::scientific << std::setprecision(12);
        
        summary_file << "energy_per_site " << obs.energy.value << " " << obs.energy.error << endl;
        summary_file << "specific_heat " << obs.specific_heat.value << " " << obs.specific_heat.error << endl;
        
        // Save total magnetization
        for (size_t d = 0; d < obs.magnetization.values.size(); ++d) {
            summary_file << "magnetization_" << d << " " 
                        << obs.magnetization.values[d] << " " 
                        << obs.magnetization.errors[d] << endl;
        }
        summary_file.close();
        
        // Save sublattice magnetizations
        ofstream sub_mag_file(out_dir + "/sublattice_magnetization.txt");
        sub_mag_file << "# Sublattice magnetizations <S_α>" << endl;
        sub_mag_file << "# Format: sublattice component mean error" << endl;
        sub_mag_file << std::scientific << std::setprecision(12);
        
        for (size_t alpha = 0; alpha < obs.sublattice_magnetization.size(); ++alpha) {
            const auto& M = obs.sublattice_magnetization[alpha];
            for (size_t d = 0; d < M.values.size(); ++d) {
                sub_mag_file << alpha << " " << d << " " 
                            << M.values[d] << " " << M.errors[d] << endl;
            }
        }
        sub_mag_file.close();
        
        // Save energy-sublattice cross terms
        ofstream cross_file(out_dir + "/energy_sublattice_cross.txt");
        cross_file << "# Energy-sublattice cross correlations <E*S_α> - <E><S_α>" << endl;
        cross_file << "# Format: sublattice component mean error" << endl;
        cross_file << std::scientific << std::setprecision(12);
        
        for (size_t alpha = 0; alpha < obs.energy_sublattice_cross.size(); ++alpha) {
            const auto& cross = obs.energy_sublattice_cross[alpha];
            for (size_t d = 0; d < cross.values.size(); ++d) {
                cross_file << alpha << " " << d << " " 
                          << cross.values[d] << " " << cross.errors[d] << endl;
            }
        }
        cross_file.close();
    }

// ---- Lattice::save_thermodynamic_observables_hdf5 ----
    void Lattice::save_thermodynamic_observables_hdf5(const string& out_dir,
                                              const ThermodynamicObservables& obs,
                                              const vector<double>& energies,
                                              const vector<SpinVector>& magnetizations,
                                              const vector<vector<SpinVector>>& sublattice_mags,
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
        
        // Create HDF5 writer
        HDF5PTWriter writer(filename, obs.temperature, lattice_size, spin_dim, N_atoms,
                           n_samples, n_anneal, n_measure, probe_rate, swap_rate,
                           overrelaxation_rate, acceptance_rate, swap_acceptance_rate);
        
        // Write time series data
        writer.write_timeseries(energies, magnetizations, sublattice_mags);
        
        // Prepare observable data in format expected by writer
        vector<vector<double>> sublattice_mag_means(N_atoms);
        vector<vector<double>> sublattice_mag_errors(N_atoms);
        vector<vector<double>> energy_cross_means(N_atoms);
        vector<vector<double>> energy_cross_errors(N_atoms);
        
        for (size_t alpha = 0; alpha < N_atoms; ++alpha) {
            sublattice_mag_means[alpha] = obs.sublattice_magnetization[alpha].values;
            sublattice_mag_errors[alpha] = obs.sublattice_magnetization[alpha].errors;
            energy_cross_means[alpha] = obs.energy_sublattice_cross[alpha].values;
            energy_cross_errors[alpha] = obs.energy_sublattice_cross[alpha].errors;
        }
        
        // Write observables (including total magnetization)
        writer.write_observables(obs.energy.value, obs.energy.error,
                                obs.specific_heat.value, obs.specific_heat.error,
                                obs.magnetization.values, obs.magnetization.errors,
                                sublattice_mag_means, sublattice_mag_errors,
                                energy_cross_means, energy_cross_errors);
        
        writer.close();
#else
        std::cerr << "Warning: HDF5 support not enabled. Cannot save HDF5 output." << std::endl;
        std::cerr << "Compile with -DHDF5_ENABLED to enable HDF5 output." << std::endl;
#endif
    }

// ---- Lattice::save_heat_capacity_hdf5 ----
    void Lattice::save_heat_capacity_hdf5(const string& out_dir,
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
        
        std::string format = "HDF5_PT_Aggregated_v1.0";
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

// ---- Lattice::save_kagome_order_parameters ----
    void Lattice::save_kagome_order_parameters(const string& rank_dir, double temperature,
                                       const vector<double>& scalar_chi,
                                       const vector<Eigen::Vector3d>& vector_chi,
                                       const vector<Eigen::Matrix3d>& nematic,
                                       const vector<double>& monopole,
                                       const vector<Eigen::Vector4d>& monopole_sub,
                                       const vector<double>& scalar_chi_global,
                                       const vector<Eigen::Vector3d>& vector_chi_global,
                                       const vector<Eigen::Matrix3d>& nematic_global) const {
#ifdef HDF5_ENABLED
        string filename = rank_dir + "/kagome_order_T" + std::to_string(temperature) + ".h5";
        
        try {
            H5::H5File file(filename, H5F_ACC_TRUNC);
            
            size_t n_samples = scalar_chi.size();
            
            // Create metadata group
            H5::Group metadata = file.createGroup("/metadata");
            H5::DataSpace scalar_space(H5S_SCALAR);
            
            H5::Attribute temp_attr = metadata.createAttribute("temperature", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            temp_attr.write(H5::PredType::NATIVE_DOUBLE, &temperature);
            
            H5::Attribute n_attr = metadata.createAttribute("n_samples", 
                H5::PredType::NATIVE_HSIZE, scalar_space);
            n_attr.write(H5::PredType::NATIVE_HSIZE, &n_samples);
            
            std::string desc = "Kagome order parameters for pyrochlore (sublattices 1,2,3)";
            H5::StrType str_type(H5::PredType::C_S1, desc.size() + 1);
            H5::Attribute desc_attr = metadata.createAttribute("description", str_type, scalar_space);
            desc_attr.write(str_type, desc.c_str());
            
            // Create data group
            H5::Group data = file.createGroup("/timeseries");
            
            // Write scalar chirality
            hsize_t dims1d[1] = {n_samples};
            H5::DataSpace space1d(1, dims1d);
            
            H5::DataSet chi_scalar = data.createDataSet("scalar_chirality", 
                H5::PredType::NATIVE_DOUBLE, space1d);
            chi_scalar.write(scalar_chi.data(), H5::PredType::NATIVE_DOUBLE);
            
            // Write vector chirality [n_samples x 3]
            hsize_t dims2d[2] = {n_samples, 3};
            H5::DataSpace space2d(2, dims2d);
            
            vector<double> vec_chi_flat(n_samples * 3);
            for (size_t i = 0; i < n_samples; ++i) {
                vec_chi_flat[i * 3 + 0] = vector_chi[i](0);
                vec_chi_flat[i * 3 + 1] = vector_chi[i](1);
                vec_chi_flat[i * 3 + 2] = vector_chi[i](2);
            }
            
            H5::DataSet chi_vec = data.createDataSet("vector_chirality", 
                H5::PredType::NATIVE_DOUBLE, space2d);
            chi_vec.write(vec_chi_flat.data(), H5::PredType::NATIVE_DOUBLE);
            
            // Write nematic order [n_samples x 3 bonds x 3 components]
            // Q[bond_type][component] where bond_type = {12, 23, 31}, component = {x, y, z}
            hsize_t dims3d[3] = {n_samples, 3, 3};
            H5::DataSpace space3d(3, dims3d);
            
            vector<double> nem_flat(n_samples * 9);
            for (size_t i = 0; i < n_samples; ++i) {
                for (int b = 0; b < 3; ++b) {      // bond type
                    for (int c = 0; c < 3; ++c) {  // spin component
                        nem_flat[i * 9 + b * 3 + c] = nematic[i](b, c);
                    }
                }
            }
            
            H5::DataSet nem_ds = data.createDataSet("nematic_bond_order", 
                H5::PredType::NATIVE_DOUBLE, space3d);
            nem_ds.write(nem_flat.data(), H5::PredType::NATIVE_DOUBLE);
            
            // Add attribute describing dimensions
            std::string nem_desc = "Shape: [n_samples, bond_type, spin_component]. "
                                   "bond_type: 0=1-2, 1=2-3, 2=3-1. "
                                   "spin_component: 0=x, 1=y, 2=z. "
                                   "Q_ij^alpha = <S_i^alpha * S_j^alpha>";
            H5::StrType nem_str_type(H5::PredType::C_S1, nem_desc.size() + 1);
            H5::Attribute nem_attr = nem_ds.createAttribute("description", nem_str_type, scalar_space);
            nem_attr.write(nem_str_type, nem_desc.c_str());
            
            // Write monopole density (signed) [n_samples]
            H5::DataSet mono_ds = data.createDataSet("monopole_density", 
                H5::PredType::NATIVE_DOUBLE, space1d);
            mono_ds.write(monopole.data(), H5::PredType::NATIVE_DOUBLE);
            
            std::string mono_desc = "Signed monopole density: sum of S_0 . (S_1 x S_2 + S_2 x S_3 + S_3 x S_1) "
                                    "over all tetrahedra, normalized by N_cells. "
                                    "Positive = 'in' monopole, negative = 'out' monopole.";
            H5::StrType mono_str_type(H5::PredType::C_S1, mono_desc.size() + 1);
            H5::Attribute mono_attr = mono_ds.createAttribute("description", mono_str_type, scalar_space);
            mono_attr.write(mono_str_type, mono_desc.c_str());
            
            // Write monopole by sublattice [n_samples x 4]
            hsize_t dims_mono_sub[2] = {n_samples, 4};
            H5::DataSpace space_mono_sub(2, dims_mono_sub);
            
            vector<double> mono_sub_flat(n_samples * 4);
            for (size_t i = 0; i < n_samples; ++i) {
                for (int s = 0; s < 4; ++s) {
                    mono_sub_flat[i * 4 + s] = monopole_sub[i](s);
                }
            }
            
            H5::DataSet mono_sub_ds = data.createDataSet("monopole_by_sublattice", 
                H5::PredType::NATIVE_DOUBLE, space_mono_sub);
            mono_sub_ds.write(mono_sub_flat.data(), H5::PredType::NATIVE_DOUBLE);
            
            std::string mono_sub_desc = "Shape: [n_samples, sublattice]. "
                                        "Sublattice 0=apex, 1,2,3=kagome. "
                                        "Sum of apex contribution for each tetrahedron type.";
            H5::StrType mono_sub_str_type(H5::PredType::C_S1, mono_sub_desc.size() + 1);
            H5::Attribute mono_sub_attr = mono_sub_ds.createAttribute("description", mono_sub_str_type, scalar_space);
            mono_sub_attr.write(mono_sub_str_type, mono_sub_desc.c_str());
            
            // ===== GLOBAL-FRAME TIMESERIES =====
            
            // Write global scalar chirality [n_samples]
            H5::DataSet chi_scalar_g = data.createDataSet("scalar_chirality_global", 
                H5::PredType::NATIVE_DOUBLE, space1d);
            chi_scalar_g.write(scalar_chi_global.data(), H5::PredType::NATIVE_DOUBLE);
            
            // Write global vector chirality [n_samples x 3]
            vector<double> vec_chi_g_flat(n_samples * 3);
            for (size_t i = 0; i < n_samples; ++i) {
                vec_chi_g_flat[i * 3 + 0] = vector_chi_global[i](0);
                vec_chi_g_flat[i * 3 + 1] = vector_chi_global[i](1);
                vec_chi_g_flat[i * 3 + 2] = vector_chi_global[i](2);
            }
            H5::DataSet chi_vec_g = data.createDataSet("vector_chirality_global", 
                H5::PredType::NATIVE_DOUBLE, space2d);
            chi_vec_g.write(vec_chi_g_flat.data(), H5::PredType::NATIVE_DOUBLE);
            
            // Write global nematic order [n_samples x 3 bonds x 3 global components]
            vector<double> nem_g_flat(n_samples * 9);
            for (size_t i = 0; i < n_samples; ++i) {
                for (int b = 0; b < 3; ++b) {
                    for (int c = 0; c < 3; ++c) {
                        nem_g_flat[i * 9 + b * 3 + c] = nematic_global[i](b, c);
                    }
                }
            }
            H5::DataSet nem_g_ds = data.createDataSet("nematic_bond_order_global", 
                H5::PredType::NATIVE_DOUBLE, space3d);
            nem_g_ds.write(nem_g_flat.data(), H5::PredType::NATIVE_DOUBLE);
            
            std::string nem_g_desc = "Shape: [n_samples, bond_type, global_component]. "
                                     "bond_type: 0=1-2, 1=2-3, 2=3-1. "
                                     "global_component: 0=X, 1=Y, 2=Z (global Cartesian). "
                                     "Q_ij^alpha = <S_i^alpha_global * S_j^alpha_global>";
            H5::StrType nem_g_str_type(H5::PredType::C_S1, nem_g_desc.size() + 1);
            H5::Attribute nem_g_attr = nem_g_ds.createAttribute("description", nem_g_str_type, scalar_space);
            nem_g_attr.write(nem_g_str_type, nem_g_desc.c_str());
            
            // Write summary statistics
            H5::Group stats = file.createGroup("/statistics");
            
            // Scalar chirality stats - save as datasets for easier access
            double chi_mean = std::accumulate(scalar_chi.begin(), scalar_chi.end(), 0.0) / n_samples;
            double chi2_mean = 0.0;
            for (double c : scalar_chi) chi2_mean += c * c;
            chi2_mean /= n_samples;
            double chi_std = std::sqrt(std::max(0.0, chi2_mean - chi_mean * chi_mean));
            
            // Save as both attributes (backward compat) and datasets
            H5::Attribute chi_mean_attr = stats.createAttribute("scalar_chirality_mean", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            chi_mean_attr.write(H5::PredType::NATIVE_DOUBLE, &chi_mean);
            
            H5::Attribute chi_std_attr = stats.createAttribute("scalar_chirality_std", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            chi_std_attr.write(H5::PredType::NATIVE_DOUBLE, &chi_std);
            
            // Also save chi^2 mean for susceptibility: chi_susc = (<chi^2> - <chi>^2) / T
            H5::Attribute chi2_attr = stats.createAttribute("scalar_chirality_squared_mean", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            chi2_attr.write(H5::PredType::NATIVE_DOUBLE, &chi2_mean);
            
            // Vector chirality component-resolved stats
            Eigen::Vector3d kappa_mean = Eigen::Vector3d::Zero();
            for (const auto& k : vector_chi) kappa_mean += k;
            kappa_mean /= n_samples;
            
            hsize_t dims_kappa[1] = {3};
            H5::DataSpace space_kappa(1, dims_kappa);
            double kappa_arr[3] = {kappa_mean(0), kappa_mean(1), kappa_mean(2)};
            H5::DataSet kappa_mean_ds = stats.createDataSet("vector_chirality_mean", 
                H5::PredType::NATIVE_DOUBLE, space_kappa);
            kappa_mean_ds.write(kappa_arr, H5::PredType::NATIVE_DOUBLE);
            
            // Also save magnitude for convenience
            double kappa_mag_mean = 0.0;
            for (const auto& k : vector_chi) kappa_mag_mean += k.norm();
            kappa_mag_mean /= n_samples;
            
            H5::Attribute kappa_mag_attr = stats.createAttribute("vector_chirality_magnitude_mean", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            kappa_mag_attr.write(H5::PredType::NATIVE_DOUBLE, &kappa_mag_mean);
            
            // Nematic order stats: mean over samples [3 bonds x 3 components]
            Eigen::Matrix3d Q_mean = Eigen::Matrix3d::Zero();
            for (const auto& Q : nematic) Q_mean += Q;
            Q_mean /= n_samples;
            
            hsize_t dims_q[2] = {3, 3};
            H5::DataSpace space_q(2, dims_q);
            double Q_arr[9];
            for (int b = 0; b < 3; ++b) {
                for (int c = 0; c < 3; ++c) {
                    Q_arr[b * 3 + c] = Q_mean(b, c);
                }
            }
            H5::DataSet Q_mean_ds = stats.createDataSet("nematic_bond_order_mean", 
                H5::PredType::NATIVE_DOUBLE, space_q);
            Q_mean_ds.write(Q_arr, H5::PredType::NATIVE_DOUBLE);
            
            // Monopole density stats
            double mono_mean = std::accumulate(monopole.begin(), monopole.end(), 0.0) / n_samples;
            double mono2_mean = 0.0;
            for (double m : monopole) mono2_mean += m * m;
            mono2_mean /= n_samples;
            double mono_std = std::sqrt(std::max(0.0, mono2_mean - mono_mean * mono_mean));
            
            H5::Attribute mono_mean_attr = stats.createAttribute("monopole_density_mean", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            mono_mean_attr.write(H5::PredType::NATIVE_DOUBLE, &mono_mean);
            
            H5::Attribute mono_std_attr = stats.createAttribute("monopole_density_std", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            mono_std_attr.write(H5::PredType::NATIVE_DOUBLE, &mono_std);
            
            // Monopole by sublattice mean
            Eigen::Vector4d mono_sub_mean = Eigen::Vector4d::Zero();
            for (const auto& ms : monopole_sub) mono_sub_mean += ms;
            mono_sub_mean /= n_samples;
            
            hsize_t dims_mono_stats[1] = {4};
            H5::DataSpace space_mono_stats(1, dims_mono_stats);
            double mono_sub_arr[4] = {mono_sub_mean(0), mono_sub_mean(1), 
                                       mono_sub_mean(2), mono_sub_mean(3)};
            H5::DataSet mono_sub_mean_ds = stats.createDataSet("monopole_by_sublattice_mean", 
                H5::PredType::NATIVE_DOUBLE, space_mono_stats);
            mono_sub_mean_ds.write(mono_sub_arr, H5::PredType::NATIVE_DOUBLE);
            
            // ===== GLOBAL-FRAME STATISTICS =====
            
            // Scalar chirality global stats
            double chi_g_mean = std::accumulate(scalar_chi_global.begin(), scalar_chi_global.end(), 0.0) / n_samples;
            double chi_g2_mean = 0.0;
            for (double c : scalar_chi_global) chi_g2_mean += c * c;
            chi_g2_mean /= n_samples;
            double chi_g_std = std::sqrt(std::max(0.0, chi_g2_mean - chi_g_mean * chi_g_mean));
            
            H5::Attribute chi_g_mean_attr = stats.createAttribute("scalar_chirality_global_mean", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            chi_g_mean_attr.write(H5::PredType::NATIVE_DOUBLE, &chi_g_mean);
            
            H5::Attribute chi_g_std_attr = stats.createAttribute("scalar_chirality_global_std", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            chi_g_std_attr.write(H5::PredType::NATIVE_DOUBLE, &chi_g_std);
            
            H5::Attribute chi_g2_attr = stats.createAttribute("scalar_chirality_global_squared_mean", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            chi_g2_attr.write(H5::PredType::NATIVE_DOUBLE, &chi_g2_mean);
            
            // Vector chirality global stats
            Eigen::Vector3d kappa_g_mean = Eigen::Vector3d::Zero();
            for (const auto& k : vector_chi_global) kappa_g_mean += k;
            kappa_g_mean /= n_samples;
            
            double kappa_g_arr[3] = {kappa_g_mean(0), kappa_g_mean(1), kappa_g_mean(2)};
            H5::DataSet kappa_g_mean_ds = stats.createDataSet("vector_chirality_global_mean", 
                H5::PredType::NATIVE_DOUBLE, space_kappa);
            kappa_g_mean_ds.write(kappa_g_arr, H5::PredType::NATIVE_DOUBLE);
            
            double kappa_g_mag_mean = 0.0;
            for (const auto& k : vector_chi_global) kappa_g_mag_mean += k.norm();
            kappa_g_mag_mean /= n_samples;
            
            H5::Attribute kappa_g_mag_attr = stats.createAttribute("vector_chirality_global_magnitude_mean", 
                H5::PredType::NATIVE_DOUBLE, scalar_space);
            kappa_g_mag_attr.write(H5::PredType::NATIVE_DOUBLE, &kappa_g_mag_mean);
            
            // Nematic global stats: mean [3 bonds x 3 global components]
            Eigen::Matrix3d Q_g_mean = Eigen::Matrix3d::Zero();
            for (const auto& Q : nematic_global) Q_g_mean += Q;
            Q_g_mean /= n_samples;
            
            double Q_g_arr[9];
            for (int b = 0; b < 3; ++b) {
                for (int c = 0; c < 3; ++c) {
                    Q_g_arr[b * 3 + c] = Q_g_mean(b, c);
                }
            }
            H5::DataSet Q_g_mean_ds = stats.createDataSet("nematic_bond_order_global_mean", 
                H5::PredType::NATIVE_DOUBLE, space_q);
            Q_g_mean_ds.write(Q_g_arr, H5::PredType::NATIVE_DOUBLE);
            
            file.close();
            
        } catch (const H5::Exception& e) {
            cerr << "HDF5 error saving kagome order parameters: " << e.getDetailMsg() << endl;
        }
#else
        // Plain text fallback
        string filename = rank_dir + "/kagome_order_T" + std::to_string(temperature) + ".txt";
        ofstream file(filename);
        file << "# Kagome order parameters (sublattices 1,2,3)\n";
        file << "# Component-resolved nematic: Q_ij^alpha = <S_i^alpha * S_j^alpha>\n";
        file << "# Local-frame and global-frame order parameters\n";
        file << "# sample scalar_chi kappa_x kappa_y kappa_z "
             << "Q12_x Q12_y Q12_z Q23_x Q23_y Q23_z Q31_x Q31_y Q31_z "
             << "monopole mono_sub0 mono_sub1 mono_sub2 mono_sub3 "
             << "scalar_chi_g kappa_gx kappa_gy kappa_gz "
             << "Q12_gX Q12_gY Q12_gZ Q23_gX Q23_gY Q23_gZ Q31_gX Q31_gY Q31_gZ\n";
        for (size_t i = 0; i < scalar_chi.size(); ++i) {
            file << i << " " << scalar_chi[i] << " "
                 << vector_chi[i](0) << " " << vector_chi[i](1) << " " << vector_chi[i](2);
            for (int b = 0; b < 3; ++b) {
                for (int c = 0; c < 3; ++c) {
                    file << " " << nematic[i](b, c);
                }
            }
            file << " " << monopole[i];
            for (int s = 0; s < 4; ++s) {
                file << " " << monopole_sub[i](s);
            }
            // Global-frame quantities
            file << " " << scalar_chi_global[i] << " "
                 << vector_chi_global[i](0) << " " << vector_chi_global[i](1) << " " << vector_chi_global[i](2);
            for (int b = 0; b < 3; ++b) {
                for (int c = 0; c < 3; ++c) {
                    file << " " << nematic_global[i](b, c);
                }
            }
            file << "\n";
        }
        file.close();
#endif
    }

