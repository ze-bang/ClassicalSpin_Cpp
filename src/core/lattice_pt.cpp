/**
 * @file lattice_pt.cpp
 * @brief Lattice parallel-tempering driver + temperature-grid optimizers.
 *
 * The parallel-tempering code is the largest single slice of `Lattice`
 * and the most MPI-heavy. Carving it out of the header also lets the
 * Bittner/Katzgraber and gradient-based (Miyata) temperature-grid
 * optimizers live next to their driver instead of polluting every TU
 * that only wants `simulated_annealing` or `molecular_dynamics`.
 */

#include "classical_spin/lattice/lattice.h"

#include <mpi.h>
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

// ---- Lattice::parallel_tempering ----
    void Lattice::parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move, MPI_Comm comm,
                           bool verbose, bool accumulate_correlations,
                           size_t n_bond_types,
                           const vector<size_t>& sweeps_per_temp) {
        // Initialize MPI
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(nullptr, nullptr);
        }
        
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        
        // Set random seed
        seed_lehman((std::chrono::system_clock::now().time_since_epoch().count() + rank * 1000) * 2 + 1);
        
        double curr_Temp = temp[rank];
        double sigma = 1000.0;
        int swap_accept = 0;
        double curr_accept = 0;
        
        // Determine exchange frequency: Bittner adaptive or fixed
        // When using adaptive sweeps, the exchange frequency is set to the maximum
        // sweeps_per_temp across all replicas, ensuring all replicas are decorrelated
        // before exchanges (especially at bottleneck temperatures near phase transitions).
        bool use_adaptive_sweeps = !sweeps_per_temp.empty() && sweeps_per_temp.size() >= static_cast<size_t>(size);
        size_t effective_swap_rate = swap_rate;
        if (use_adaptive_sweeps) {
            // Use the max sweep count as the global exchange interval
            // This ensures the slowest-decorrelating replica is adequately sampled
            effective_swap_rate = *std::max_element(sweeps_per_temp.begin(), sweeps_per_temp.end());
            if (rank == 0) {
                cout << "Using Bittner adaptive sweep schedule:" << endl;
                cout << "  Exchange frequency set to max(sweeps_per_temp) = " << effective_swap_rate << endl;
                cout << "  (was swap_rate = " << swap_rate << ")" << endl;
                size_t min_spt = *std::min_element(sweeps_per_temp.begin(), sweeps_per_temp.end());
                cout << "  Sweep range: [" << min_spt << ", " << effective_swap_rate << "]" << endl;
            }
        }
        
        vector<double> heat_capacity, dHeat;
        if (rank == 0) {
            heat_capacity.resize(size);
            dHeat.resize(size);
        }
        
        vector<double> energies;
        vector<SpinVector> magnetizations;
        vector<vector<SpinVector>> sublattice_mags;  // sublattice magnetizations
        
        // Kagome order parameters (pyrochlore patch)
        vector<double> scalar_chiralities;
        vector<Eigen::Vector3d> vector_chiralities;
        vector<Eigen::Matrix3d> nematic_orders;  // [bond_type x spin_component]
        vector<double> monopole_densities;
        vector<Eigen::Vector4d> monopole_by_sublattices;

        // Global-frame order parameters
        vector<double> scalar_chiralities_global;
        vector<Eigen::Vector3d> vector_chiralities_global;
        vector<Eigen::Matrix3d> nematic_orders_global;  // [bond_type x global_component]
        
        size_t expected_samples = n_measure / probe_rate + 100;
        energies.reserve(expected_samples);
        magnetizations.reserve(expected_samples);
        sublattice_mags.reserve(expected_samples);
        scalar_chiralities.reserve(expected_samples);
        vector_chiralities.reserve(expected_samples);
        nematic_orders.reserve(expected_samples);
        monopole_densities.reserve(expected_samples);
        monopole_by_sublattices.reserve(expected_samples);
        scalar_chiralities_global.reserve(expected_samples);
        vector_chiralities_global.reserve(expected_samples);
        nematic_orders_global.reserve(expected_samples);
        
        // Initialize correlation accumulator if requested
        RealSpaceCorrelationAccumulator corr_acc;
        if (accumulate_correlations) {
            corr_acc = create_correlation_accumulator(n_bond_types);
            cout << "Rank " << rank << ": Correlation accumulator initialized ("
                 << corr_acc.storage_bytes() / 1024.0 << " KB)" << endl;
        }
        
        cout << "Rank " << rank << ": T=" << curr_Temp << endl;
        
        // Equilibration
        cout << "Rank " << rank << ": Equilibrating..." << endl;
        for (size_t i = 0; i < n_anneal; ++i) {
            if (overrelaxation_rate > 0) {
                overrelaxation();
                if (i % overrelaxation_rate == 0) {
                    curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
                }
            } else {
                curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
            }
            
            // Attempt replica exchange (use adaptive or fixed rate)
            if (effective_swap_rate > 0 && i % effective_swap_rate == 0) {
                swap_accept += attempt_replica_exchange(rank, size, temp, curr_Temp, i / effective_swap_rate, comm);
            }
        }
        
        // Main measurement phase
        // First: estimate autocorrelation to validate probe_rate
        {
            size_t pilot_samples = std::min(size_t(5000), n_measure / 5);
            size_t pilot_interval = std::max(size_t(1), std::min(probe_rate, size_t(10)));
            vector<double> pilot_energies;
            pilot_energies.reserve(pilot_samples / pilot_interval + 1);
            for (size_t i = 0; i < pilot_samples; ++i) {
                if (overrelaxation_rate > 0) {
                    overrelaxation();
                    if (i % overrelaxation_rate == 0) {
                        metropolis(curr_Temp, gaussian_move, sigma);
                    }
                } else {
                    metropolis(curr_Temp, gaussian_move, sigma);
                }
                if (effective_swap_rate > 0 && i % effective_swap_rate == 0) {
                    attempt_replica_exchange(rank, size, temp, curr_Temp, i / effective_swap_rate, comm);
                }
                if (i % pilot_interval == 0) {
                    pilot_energies.push_back(total_energy(spins));
                }
            }
            AutocorrelationResult pilot_acf = compute_autocorrelation(pilot_energies, pilot_interval);
            size_t tau_int_sweeps = static_cast<size_t>(std::ceil(pilot_acf.tau_int)) * pilot_interval;
            size_t min_probe_rate = 2 * tau_int_sweeps;
            size_t n_indep = (probe_rate >= min_probe_rate) ? (n_measure / probe_rate) : (n_measure / min_probe_rate);
            
            cout << "Rank " << rank << ": τ_int=" << pilot_acf.tau_int 
                 << " samples (=" << tau_int_sweeps << " sweeps)"
                 << ", recommended probe_rate ≥ " << min_probe_rate << " sweeps" << endl;
            
            if (probe_rate < min_probe_rate) {
                cout << "[WARNING] Rank " << rank << ": probe_rate=" << probe_rate
                     << " < 2·τ_int=" << min_probe_rate 
                     << " sweeps. Samples will be correlated! "
                     << "Effective independent samples ≈ " << n_indep 
                     << " (vs " << n_measure / probe_rate << " total samples). "
                     << "Consider increasing probe_rate to " << min_probe_rate << "." << endl;
            } else {
                cout << "Rank " << rank << ": probe_rate=" << probe_rate 
                     << " ≥ 2·τ_int=" << min_probe_rate 
                     << " — samples are approximately independent ("
                     << n_measure / probe_rate << " samples)." << endl;
            }
        }
        
        cout << "Rank " << rank << ": Measuring..." << endl;
        for (size_t i = 0; i < n_measure; ++i) {
            if (overrelaxation_rate > 0) {
                overrelaxation();
                if (i % overrelaxation_rate == 0) {
                    curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
                }
            } else {
                curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
            }
            
            if (effective_swap_rate > 0 && i % effective_swap_rate == 0) {
                swap_accept += attempt_replica_exchange(rank, size, temp, curr_Temp, i / effective_swap_rate, comm);
            }
            
            if (i % probe_rate == 0) {
                energies.push_back(total_energy(spins));
                magnetizations.push_back(magnetization_global());
                sublattice_mags.push_back(magnetization_sublattice());
                
                // Kagome order parameters (pyrochlore) - use fast single-pass version
                if (is_pyrochlore() && N_atoms >= 4) {
                    auto params = compute_pyrochlore_order_parameters_fast();
                    scalar_chiralities.push_back(params.scalar_chirality);
                    vector_chiralities.push_back(params.vector_chirality);
                    nematic_orders.push_back(params.nematic_order);
                    monopole_densities.push_back(params.monopole_density);
                    monopole_by_sublattices.push_back(params.monopole_by_sublattice);
                    scalar_chiralities_global.push_back(params.scalar_chirality_global);
                    vector_chiralities_global.push_back(params.vector_chirality_global);
                    nematic_orders_global.push_back(params.nematic_order_global);
                }
                
                // Accumulate real-space correlations for S(q)
                if (accumulate_correlations) {
                    accumulate_correlations_internal(corr_acc);
                }
            }
        }
        
        cout << "Rank " << rank << ": Collected " << energies.size() << " samples" << endl;
        
        // Save correlation data before MPI operations
        if (accumulate_correlations) {
            // Each rank saves its own correlation data to rank_#/correlations_T*.h5
            bool should_write = (std::find(rank_to_write.begin(), rank_to_write.end(), rank) != rank_to_write.end())
                               || (std::find(rank_to_write.begin(), rank_to_write.end(), -1) != rank_to_write.end());
            
            if (should_write) {
                string corr_dir = dir_name + "/rank_" + std::to_string(rank);
                std::filesystem::create_directories(corr_dir);
                
#ifdef HDF5_ENABLED
                // Save raw correlations to HDF5
                string h5_filename = corr_dir + "/correlations_T" + std::to_string(curr_Temp) + ".h5";
                corr_acc.save_hdf5(h5_filename);
                cout << "Rank " << rank << ": Saved correlations to " << h5_filename 
                     << " (" << corr_acc.n_samples << " samples)" << endl;
#endif
            }
        }
        
        // Save kagome order parameters (pyrochlore only)
        if (is_pyrochlore() && N_atoms >= 4 && !scalar_chiralities.empty()) {
            bool should_write = (std::find(rank_to_write.begin(), rank_to_write.end(), rank) != rank_to_write.end())
                               || (std::find(rank_to_write.begin(), rank_to_write.end(), -1) != rank_to_write.end());
            
            if (should_write) {
                string rank_dir = dir_name + "/rank_" + std::to_string(rank);
                std::filesystem::create_directories(rank_dir);
                save_kagome_order_parameters(rank_dir, curr_Temp, 
                                             scalar_chiralities, vector_chiralities, nematic_orders,
                                             monopole_densities, monopole_by_sublattices,
                                             scalar_chiralities_global, vector_chiralities_global,
                                             nematic_orders_global);
            }
        }
        
        // Gather and save statistics with comprehensive observables
        gather_and_save_statistics_comprehensive(rank, size, curr_Temp, energies, 
                                                  magnetizations, sublattice_mags,
                                                  heat_capacity, dHeat, temp, dir_name, 
                                                  rank_to_write, n_anneal, n_measure, 
                                                  curr_accept, swap_accept,
                                                  swap_rate, overrelaxation_rate, probe_rate, comm, verbose);
    }

// ---- Lattice::attempt_replica_exchange ----
    int Lattice::attempt_replica_exchange(int rank, int size, const vector<double>& temp,
                                double curr_Temp, size_t swap_parity, MPI_Comm comm) {
        // Determine partner based on checkerboard pattern
        int partner_rank;
        if (swap_parity % 2 == 0) {
            partner_rank = (rank % 2 == 0) ? rank + 1 : rank - 1;
        } else {
            partner_rank = (rank % 2 == 1) ? rank + 1 : rank - 1;
        }
        
        if (partner_rank < 0 || partner_rank >= size) {
            return 0;
        }
        
        // Exchange energies
        double E = total_energy(spins);
        double E_partner, T_partner = temp[partner_rank];
        
        MPI_Sendrecv(&E, 1, MPI_DOUBLE, partner_rank, 0,
                    &E_partner, 1, MPI_DOUBLE, partner_rank, 0,
                    comm, MPI_STATUS_IGNORE);
        
        // Decide acceptance using parallel tempering Metropolis criterion:
        // P_swap = min(1, exp[Δ]) where Δ = (β_cold - β_hot)(E_cold - E_hot)
        // With ordering: rank 0 = coldest (β large), highest rank = hottest (β small)
        // rank < partner_rank means: curr is COLDER, partner is HOTTER
        int accept_int = 0;
        if (rank < partner_rank) {
            // Only the lower rank computes the acceptance decision
            // curr = cold (i), partner = hot (j)
            // β_curr > β_partner, typically E_curr < E_partner (cold has lower energy)
            // Δ = (β_curr - β_partner)(E_curr - E_partner) = (positive)(negative) = negative typically
            double beta_curr = 1.0 / curr_Temp;
            double beta_partner = 1.0 / T_partner;
            double delta = (beta_curr - beta_partner) * (E - E_partner);
            bool accept = (delta >= 0) || (random_double_lehman(0.0, 1.0) < std::exp(delta));
            accept_int = accept ? 1 : 0;
            
            // // DEBUG: Print exchange details (only first few times to avoid spam)
            // static int debug_count = 0;
            // if (debug_count < 20) {
            //     cout << "[PT DEBUG] rank=" << rank << "->" << partner_rank
            //          << " T_cold=" << curr_Temp << " T_hot=" << T_partner
            //          << " E_cold=" << E << " E_hot=" << E_partner
            //          << " delta=" << delta << " accept=" << accept << endl;
            //     ++debug_count;
            // }
        }
        
        // Communicate decision between partners using point-to-point (NOT collective MPI_Bcast!)
        // Lower rank sends decision to higher rank
        int recv_accept_int = 0;
        MPI_Sendrecv(&accept_int, 1, MPI_INT, partner_rank, 2,
                     &recv_accept_int, 1, MPI_INT, partner_rank, 2,
                     comm, MPI_STATUS_IGNORE);
        
        // Lower rank uses its own decision, higher rank uses received decision
        bool accept = (rank < partner_rank) ? (accept_int == 1) : (recv_accept_int == 1);
        
        // Exchange configurations if accepted
        if (accept) {
            // Serialize spins
            vector<double> send_buf(lattice_size * spin_dim);
            vector<double> recv_buf(lattice_size * spin_dim);
            
            for (size_t i = 0; i < lattice_size; ++i) {
                for (size_t j = 0; j < spin_dim; ++j) {
                    send_buf[i * spin_dim + j] = spins[i](j);
                }
            }
            
            MPI_Sendrecv(send_buf.data(), send_buf.size(), MPI_DOUBLE, partner_rank, 1,
                        recv_buf.data(), recv_buf.size(), MPI_DOUBLE, partner_rank, 1,
                        comm, MPI_STATUS_IGNORE);
            
            // Deserialize
            for (size_t i = 0; i < lattice_size; ++i) {
                for (size_t j = 0; j < spin_dim; ++j) {
                    spins[i](j) = recv_buf[i * spin_dim + j];
                }
            }
        }
        
        return accept ? 1 : 0;
    }

// ---- Lattice::estimate_sampling_interval ----
    size_t Lattice::estimate_sampling_interval(double curr_Temp, bool gaussian_move, double& sigma,
                                     size_t overrelaxation_rate, size_t n_measure,
                                     size_t probe_rate, int rank) {
        size_t prelim_samples = std::min(size_t(10000), n_measure / 10);
        size_t prelim_interval = std::max(size_t(1), overrelaxation_rate > 0 ? overrelaxation_rate : 1);
        
        vector<double> prelim_energies = collect_energy_samples(prelim_samples, prelim_interval,
                                                                curr_Temp, gaussian_move, sigma,
                                                                overrelaxation_rate);
        
        AutocorrelationResult acf = compute_autocorrelation(prelim_energies, prelim_interval);
        size_t effective_interval = std::max(acf.sampling_interval, probe_rate);
        
        cout << "Rank " << rank << ": τ_int=" << acf.tau_int 
             << ", interval=" << effective_interval << endl;
        
        return effective_interval;
    }

// ---- Lattice::gather_and_save_statistics ----
    void Lattice::gather_and_save_statistics(int rank, int size, double curr_Temp,
                                   const vector<double>& energies,
                                   const vector<SpinVector>& magnetizations,
                                   vector<double>& heat_capacity, vector<double>& dHeat,
                                   const vector<double>& temp, const string& dir_name,
                                   const vector<int>& rank_to_write,
                                   size_t n_anneal, size_t n_measure,
                                   double curr_accept, int swap_accept,
                                   size_t swap_rate, size_t overrelaxation_rate,
                                   size_t probe_rate) {
        // Compute local heat capacity per site using binning analysis
        // c_V = Var(E) / (T² N²) = Var(E/N) / T²
        double E_mean = std::accumulate(energies.begin(), energies.end(), 0.0) / energies.size();
        double E2_mean = 0.0;
        for (double E : energies) {
            E2_mean += E * E;
        }
        E2_mean /= energies.size();
        double var_E = E2_mean - E_mean * E_mean;
        
        double N2 = double(lattice_size) * double(lattice_size);
        double curr_heat_capacity = var_E / (curr_Temp * curr_Temp * N2);
        double curr_dHeat = std::sqrt(var_E) / (curr_Temp * curr_Temp * N2);
        
        // Gather to root
        MPI_Gather(&curr_heat_capacity, 1, MPI_DOUBLE, heat_capacity.data(), 
                   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&curr_dHeat, 1, MPI_DOUBLE, dHeat.data(), 
                   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Report
        double total_steps = n_anneal + n_measure;
        double acc_rate = curr_accept / total_steps;
        double swap_rate_actual = (swap_rate > 0) ? double(swap_accept) / (total_steps / swap_rate) : 0.0;
        
        cout << "Rank " << rank << ": T=" << curr_Temp 
             << ", acc=" << acc_rate 
             << ", swap_acc=" << swap_rate_actual << endl;
        
        // Save results with proper MPI synchronization
        if (!dir_name.empty()) {
            // Rank 0 creates the main output directory first
            if (rank == 0) {
                filesystem::create_directories(dir_name);
            }
            // Ensure directory exists before other ranks proceed
            MPI_Barrier(MPI_COMM_WORLD);
            
            // Check if this rank should write (supports FULL mode with sentinel -1)
            bool should_write = should_rank_write(rank, rank_to_write);
            
            if (should_write) {
                string rank_dir = dir_name + "/rank_" + std::to_string(rank);
                // Each rank creates its own subdirectory (no race condition)
                filesystem::create_directories(rank_dir);
                save_observables(rank_dir, energies, magnetizations);
                save_spin_config(rank_dir + "/spins_T=" + std::to_string(curr_Temp) + ".txt");
            }
            
            // Wait for all ranks to finish writing before rank 0 writes aggregated results
            MPI_Barrier(MPI_COMM_WORLD);
            
            // Root process saves heat capacity
            if (rank == 0) {
                ofstream heat_file(dir_name + "/heat_capacity.txt");
                heat_file << "# T C_V dC_V\n";
                for (int r = 0; r < size; ++r) {
                    heat_file << temp[r] << " " << heat_capacity[r] << " " << dHeat[r] << "\n";
                }
                heat_file.close();
            }
        }
    }

// ---- Lattice::gather_and_save_statistics_comprehensive ----
    void Lattice::gather_and_save_statistics_comprehensive(int rank, int size, double curr_Temp,
                                   const vector<double>& energies,
                                   const vector<SpinVector>& magnetizations,
                                   const vector<vector<SpinVector>>& sublattice_mags,
                                   vector<double>& heat_capacity, vector<double>& dHeat,
                                   const vector<double>& temp, const string& dir_name,
                                   const vector<int>& rank_to_write,
                                   size_t n_anneal, size_t n_measure,
                                   double curr_accept, int swap_accept,
                                   size_t swap_rate, size_t overrelaxation_rate,
                                   size_t probe_rate, MPI_Comm comm,
                                   bool verbose) {
        
        // Compute comprehensive thermodynamic observables with binning analysis
        ThermodynamicObservables obs = compute_thermodynamic_observables(
            energies, sublattice_mags, curr_Temp);
        
        double curr_heat_capacity = obs.specific_heat.value;
        double curr_dHeat = obs.specific_heat.error;
        
        // Gather heat capacity to root
        MPI_Gather(&curr_heat_capacity, 1, MPI_DOUBLE, heat_capacity.data(), 
                   1, MPI_DOUBLE, 0, comm);
        MPI_Gather(&curr_dHeat, 1, MPI_DOUBLE, dHeat.data(), 
                   1, MPI_DOUBLE, 0, comm);
        
        // Report acceptance rates
        double total_steps = n_anneal + n_measure;
        // Account for overrelaxation: Metropolis is only called every overrelaxation_rate steps
        double metro_steps = (overrelaxation_rate > 0) ? total_steps / overrelaxation_rate : total_steps;
        double acc_rate = curr_accept / metro_steps;
        double swap_rate_actual = (swap_rate > 0) ? double(swap_accept) / (total_steps / swap_rate) : 0.0;
        
        cout << "Rank " << rank << ": T=" << curr_Temp 
             << ", acc=" << acc_rate 
             << ", swap_acc=" << swap_rate_actual 
             << ", <E>/N=" << obs.energy.value << "±" << obs.energy.error
             << ", C_V=" << obs.specific_heat.value << "±" << obs.specific_heat.error
             << endl;
        
        // Save results with proper MPI synchronization
        if (!dir_name.empty()) {
            // Rank 0 creates the main output directory first
            if (rank == 0) {
                filesystem::create_directories(dir_name);
            }
            // Ensure directory exists before other ranks proceed
            MPI_Barrier(comm);
            
            // Check if this rank should write (supports FULL mode with sentinel -1)
            bool should_write = should_rank_write(rank, rank_to_write);
            
            if (should_write) {
                string rank_dir = dir_name + "/rank_" + std::to_string(rank);
                // Each rank creates its own subdirectory (no race condition)
                filesystem::create_directories(rank_dir);
                
                // Save to HDF5 format (single file with all data)
#ifdef HDF5_ENABLED
                save_thermodynamic_observables_hdf5(rank_dir, obs, energies, magnetizations,
                                                   sublattice_mags, n_anneal, n_measure,
                                                   probe_rate, swap_rate, overrelaxation_rate,
                                                   acc_rate, swap_rate_actual);
#else
                if (rank == 0) {
                    cerr << "Warning: HDF5 not enabled. Compile with -DHDF5_ENABLED=ON to enable output." << endl;
                }
#endif
                
                // Save spin configuration only if verbose mode is enabled
                save_spin_config(rank_dir + "/spins_T=" + std::to_string(curr_Temp) + ".txt");
            }
            
            // Wait for all ranks to finish writing before rank 0 writes aggregated results
            MPI_Barrier(comm);
            
            // Root process saves aggregated results across all temperatures
            if (rank == 0) {
#ifdef HDF5_ENABLED
                // Save heat capacity to HDF5
                save_heat_capacity_hdf5(dir_name, temp, heat_capacity, dHeat);
#endif
            }
        }
        
        // Synchronize before finishing
        MPI_Barrier(comm);
    }

// ---- Lattice::generate_optimized_temperature_grid ----
    OptimizedTempGridResult Lattice::generate_optimized_temperature_grid(
        double Tmin, double Tmax, size_t R,
        size_t warmup_sweeps,
        size_t sweeps_per_iter,
        size_t feedback_iters,
        bool gaussian_move,
        size_t overrelaxation_rate,
        double target_acceptance,
        double convergence_tol) {
        
        OptimizedTempGridResult result;
        result.converged = false;
        result.feedback_iterations_used = 0;
        
        if (R < 2) {
            result.temperatures = {Tmin};
            if (R == 1) return result;
        }
        if (R == 2) {
            result.temperatures = {Tmin, Tmax};
            result.acceptance_rates = {0.5};
            result.converged = true;
            return result;
        }
        
        cout << "=== Feedback-Optimized Temperature Grid ===" << endl;
        cout << "References:" << endl;
        cout << "  Katzgraber et al., J. Stat. Mech. P03018 (2006) - temperature placement" << endl;
        cout << "  Bittner et al., Phys. Rev. Lett. 101, 130603 (2008) - adaptive sweeps" << endl;
        cout << "T_min = " << Tmin << ", T_max = " << Tmax << ", R = " << R << endl;
        
        // Helper: convert beta to temperature
        auto temps_from_beta = [](const vector<double>& b) {
            vector<double> T(b.size());
            for (size_t i = 0; i < b.size(); ++i) {
                T[i] = 1.0 / b[i];
            }
            return T;
        };
        
        // Initialize with linear spacing in β
        double beta_min = 1.0 / Tmax;  // Hottest = smallest beta
        double beta_max = 1.0 / Tmin;  // Coldest = largest beta
        vector<double> beta(R);
        for (size_t i = 0; i < R; ++i) {
            beta[i] = beta_min + (beta_max - beta_min) * double(i) / double(R - 1);
        }
        
        // Store original spins
        SpinConfig original_spins = spins;
        
        // Use independent Lattice copies for parallel warmup
        vector<Lattice> replica_lattices(R, *this);
        double sigma = 1000.0;  // For Gaussian moves
        
        // Warmup phase: equilibrate each replica at its temperature
        cout << "Warming up " << R << " replicas";
        #ifdef _OPENMP
        cout << " (parallel)";
        #endif
        cout << "..." << endl;
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t k = 0; k < R; ++k) {
            double T_k = 1.0 / beta[k];
            double local_sigma = sigma;
            #pragma omp critical
            {
                seed_lehman((std::chrono::system_clock::now().time_since_epoch().count() + k * 12345) * 2 + 1);
            }
            for (size_t i = 0; i < warmup_sweeps; ++i) {
                replica_lattices[k].metropolis(T_k, gaussian_move, local_sigma);
                if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                    replica_lattices[k].overrelaxation();
                }
            }
        }
        
        // Cache energies
        vector<double> cached_energies(R);
        for (size_t k = 0; k < R; ++k) {
            cached_energies[k] = replica_lattices[k].total_energy(replica_lattices[k].spins);
        }
        
        // ================================================================
        // PHASE 1: Katzgraber feedback-optimized temperature placement
        // Feedback rule: Δβ_i(new) ∝ A_i (acceptance rate at edge i)
        // This concentrates temperatures where acceptance is low (bottlenecks)
        // and spreads them where acceptance is high (easy regions).
        // ================================================================
        vector<double> acceptance_rates(R - 1, 0.0);
        double base_damping = 0.5;
        
        for (size_t iter = 0; iter < feedback_iters; ++iter) {
            // Damping increases over iterations for stability
            double damping = base_damping + 0.3 * (double(iter) / double(feedback_iters));
            
            vector<size_t> attempts(R - 1, 0);
            vector<size_t> accepts(R - 1, 0);
            
            // Reduced sweeps for early iterations when far from convergence
            size_t effective_sweeps = sweeps_per_iter;
            if (iter < 3) {
                effective_sweeps = std::max(size_t(50), sweeps_per_iter / 4);
            } else if (iter < 6) {
                effective_sweeps = std::max(size_t(100), sweeps_per_iter / 2);
            }
            
            // Run MC sweeps and measure acceptance rates
            for (size_t sweep = 0; sweep < effective_sweeps; ++sweep) {
                #pragma omp parallel for schedule(static)
                for (size_t k = 0; k < R; ++k) {
                    double T_k = 1.0 / beta[k];
                    double local_sigma = sigma;
                    replica_lattices[k].metropolis(T_k, gaussian_move, local_sigma);
                    if (overrelaxation_rate > 0 && sweep % overrelaxation_rate == 0) {
                        replica_lattices[k].overrelaxation();
                    }
                }
                
                #pragma omp parallel for schedule(static)
                for (size_t k = 0; k < R; ++k) {
                    cached_energies[k] = replica_lattices[k].total_energy(replica_lattices[k].spins);
                }
                
                // Attempt replica exchanges (checkerboard pattern)
                for (int parity = 0; parity <= 1; ++parity) {
                    #pragma omp parallel for schedule(static)
                    for (size_t e = parity; e < R - 1; e += 2) {
                        double E_hot = cached_energies[e];
                        double E_cold = cached_energies[e + 1];
                        double dBeta = beta[e + 1] - beta[e];
                        double dE = E_cold - E_hot;
                        double delta = dBeta * dE;
                        
                        #pragma omp atomic
                        ++attempts[e];
                        
                        if (delta >= 0 || random_double_lehman(0.0, 1.0) < std::exp(delta)) {
                            #pragma omp atomic
                            ++accepts[e];
                            
                            #pragma omp critical
                            {
                                std::swap(replica_lattices[e].spins, replica_lattices[e + 1].spins);
                                std::swap(cached_energies[e], cached_energies[e + 1]);
                            }
                        }
                    }
                }
            }
            
            // Compute acceptance rates
            for (size_t e = 0; e < R - 1; ++e) {
                acceptance_rates[e] = double(accepts[e]) / double(attempts[e]);
            }
            
            // Check convergence
            double max_deviation = 0.0;
            double mean_rate = 0.0;
            for (size_t e = 0; e < R - 1; ++e) {
                max_deviation = std::max(max_deviation, std::abs(acceptance_rates[e] - target_acceptance));
                mean_rate += acceptance_rates[e];
            }
            mean_rate /= (R - 1);
            
            cout << "Iter " << iter + 1 << "/" << feedback_iters 
                 << ": mean A = " << std::fixed << std::setprecision(3) << mean_rate
                 << ", max dev = " << max_deviation 
                 << " (sweeps=" << effective_sweeps << ", damp=" << std::setprecision(2) << damping << ")" << endl;
            
            result.feedback_iterations_used = iter + 1;
            
            if (max_deviation < convergence_tol) {
                result.converged = true;
                cout << "Converged at iteration " << iter + 1 << endl;
                break;
            }
            
            // Katzgraber et al. feedback rule:
            // The optimal temperature set minimizes the round-trip time τ_rt = (Σ 1/f_i)².
            // The current fraction f_i at edge i is proportional to the acceptance rate A_i
            // times the local β-spacing. To minimize τ_rt, we need Δβ_i ∝ A_i:
            //   - Low A_i (bottleneck) → small Δβ_i (dense temperatures)
            //   - High A_i (easy region) → large Δβ_i (sparse temperatures)
            
            vector<double> weights(R - 1);
            double total_weight = 0.0;
            
            for (size_t e = 0; e < R - 1; ++e) {
                double A_e = std::clamp(acceptance_rates[e], 0.01, 0.99);
                weights[e] = A_e;
                total_weight += weights[e];
            }
            
            // Normalize weights to sum to 1
            for (size_t e = 0; e < R - 1; ++e) {
                weights[e] /= total_weight;
            }
            
            // Compute new beta positions from cumulative weights
            vector<double> new_beta(R);
            new_beta[0] = beta_min;
            double cumulative = 0.0;
            for (size_t e = 0; e < R - 1; ++e) {
                cumulative += weights[e];
                new_beta[e + 1] = beta_min + cumulative * (beta_max - beta_min);
            }
            new_beta[R - 1] = beta_max;  // Ensure exact endpoint
            
            // Apply damping: blend old and new positions
            for (size_t k = 1; k < R - 1; ++k) {
                new_beta[k] = (1.0 - damping) * beta[k] + damping * new_beta[k];
            }
            
            beta = new_beta;
        }
        
        // ================================================================
        // PHASE 2: Bittner et al. adaptive sweep schedule
        // Measure τ_int(T) at each temperature and set n_sweeps ∝ τ_int(T)
        // ================================================================
        cout << "\nMeasuring autocorrelation times for adaptive sweep schedule..." << endl;
        
        // Collect energy time series at each temperature to estimate τ_int
        size_t tau_samples = std::max(size_t(500), sweeps_per_iter);
        vector<double> tau_int_values(R, 1.0);
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t k = 0; k < R; ++k) {
            double T_k = 1.0 / beta[k];
            double local_sigma = sigma;
            vector<double> energy_series;
            energy_series.reserve(tau_samples);
            
            for (size_t i = 0; i < tau_samples; ++i) {
                replica_lattices[k].metropolis(T_k, gaussian_move, local_sigma);
                if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                    replica_lattices[k].overrelaxation();
                }
                energy_series.push_back(
                    replica_lattices[k].total_energy(replica_lattices[k].spins));
            }
            
            // Compute autocorrelation time using Sokal's self-consistent window
            AutocorrelationResult acf = compute_autocorrelation(energy_series, 1);
            tau_int_values[k] = std::max(1.0, acf.tau_int);
        }
        
        // Compute sweep schedule: n_sweeps_i = n_base × τ_int(T_i) / min(τ_int)
        double tau_min = *std::min_element(tau_int_values.begin(), tau_int_values.end());
        size_t n_base = 10;  // Minimum sweeps between exchanges
        
        result.autocorrelation_times = tau_int_values;
        result.sweeps_per_temp.resize(R);
        for (size_t k = 0; k < R; ++k) {
            result.sweeps_per_temp[k] = std::max(size_t(1), 
                static_cast<size_t>(std::ceil(n_base * tau_int_values[k] / tau_min)));
        }
        
        cout << "Autocorrelation times and sweep schedule:" << endl;
        for (size_t k = 0; k < std::min(R, size_t(15)); ++k) {
            cout << "  T[" << k << "] = " << std::scientific << std::setprecision(4) 
                 << 1.0 / beta[k] << "  τ_int = " << std::fixed << std::setprecision(1)
                 << tau_int_values[k] << "  n_sweeps = " << result.sweeps_per_temp[k] << endl;
        }
        if (R > 15) cout << "  ... (" << R - 15 << " more)" << endl;
        
        // Restore original spins
        spins = original_spins;
        
        // Build result
        result.temperatures = temps_from_beta(beta);
        std::sort(result.temperatures.begin(), result.temperatures.end());  // Ascending order
        
        result.acceptance_rates = acceptance_rates;
        
        // Compute local diffusivities D_i = A_i(1 - A_i)
        result.local_diffusivities.resize(R - 1);
        for (size_t e = 0; e < R - 1; ++e) {
            double A = acceptance_rates[e];
            result.local_diffusivities[e] = A * (1.0 - A);
        }
        
        // Mean acceptance rate
        result.mean_acceptance_rate = 0.0;
        for (double A : acceptance_rates) {
            result.mean_acceptance_rate += A;
        }
        result.mean_acceptance_rate /= (R - 1);
        
        // Estimate round-trip time using Katzgraber formula:
        // τ_rt ∝ (Σ_i 1/f_i)² where f_i is the current fraction at edge i
        // f_i ∝ A_i × Δβ_i / Σ_j(A_j × Δβ_j) (normalized current)
        // With Bittner adaptive sweeps factored in: each edge contributes
        // n_sweeps_avg(edge) / f_i to the round-trip time
        double sum_inv_f = 0.0;
        double total_current = 0.0;
        for (size_t e = 0; e < R - 1; ++e) {
            double d_beta = std::abs(beta[e + 1] - beta[e]);
            double A = std::max(acceptance_rates[e], 1e-6);
            total_current += A * d_beta;
        }
        for (size_t e = 0; e < R - 1; ++e) {
            double d_beta = std::abs(beta[e + 1] - beta[e]);
            double A = std::max(acceptance_rates[e], 1e-6);
            double f_i = A * d_beta / total_current;
            // Weight by average sweeps at this edge (Bittner factor)
            double n_avg = 0.5 * (result.sweeps_per_temp[e] + result.sweeps_per_temp[e + 1]);
            sum_inv_f += n_avg / f_i;
        }
        result.round_trip_estimate = sum_inv_f;
        
        // Print summary
        cout << "\n=== Optimized Temperature Grid Summary ===" << endl;
        cout << "Temperatures (ascending):" << endl;
        for (size_t k = 0; k < std::min(R, size_t(15)); ++k) {
            cout << "  T[" << k << "] = " << std::scientific << std::setprecision(6) 
                 << result.temperatures[k];
            if (k < R - 1) {
                cout << "  (A = " << std::fixed << std::setprecision(3) 
                     << acceptance_rates[k] << ")";
            }
            cout << endl;
        }
        if (R > 15) cout << "  ... (" << R - 15 << " more)" << endl;
        
        cout << "\nMean acceptance rate: " << std::fixed << std::setprecision(3) 
             << result.mean_acceptance_rate * 100 << "%" << endl;
        cout << "Estimated round-trip time scale: " << std::scientific 
             << result.round_trip_estimate << endl;
        cout << "Converged: " << (result.converged ? "YES" : "NO") << endl;
        
        return result;
    }

// ---- Lattice::generate_optimized_temperature_grid_mpi ----
    OptimizedTempGridResult Lattice::generate_optimized_temperature_grid_mpi(
        double Tmin, double Tmax,
        size_t warmup_sweeps,
        size_t sweeps_per_iter,
        size_t feedback_iters,
        bool gaussian_move,
        size_t overrelaxation_rate,
        double target_acceptance,
        double convergence_tol,
        MPI_Comm comm,
        bool use_gradient) {
        
        // Get MPI info - number of ranks = number of replicas
        int rank, R;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &R);
        
        OptimizedTempGridResult result;
        result.converged = false;
        result.feedback_iterations_used = 0;
        
        if (R < 2) {
            result.temperatures = {Tmin};
            result.converged = true;
            return result;
        }
        if (R == 2) {
            result.temperatures = {Tmin, Tmax};
            result.acceptance_rates = {0.5};
            result.converged = true;
            return result;
        }
        
        if (use_gradient) {
            return mc::generate_optimized_temperature_grid_mpi(*this, Tmin, Tmax,
                warmup_sweeps, sweeps_per_iter, feedback_iters, gaussian_move,
                overrelaxation_rate, target_acceptance, convergence_tol, comm, true);
        }
        
        // Set random seed unique to each rank
        seed_lehman((std::chrono::system_clock::now().time_since_epoch().count() + rank * 12345) * 2 + 1);
        
        if (rank == 0) {
            cout << "=== Feedback-Optimized Temperature Grid (MPI) ===" << endl;
            cout << "References:" << endl;
            cout << "  Katzgraber et al., J. Stat. Mech. P03018 (2006) - temperature placement" << endl;
            cout << "  Bittner et al., Phys. Rev. Lett. 101, 130603 (2008) - adaptive sweeps" << endl;
            cout << "T_min = " << Tmin << ", T_max = " << Tmax << ", R = " << R << " (MPI ranks)" << endl;
        }
        
        // Initialize beta array on all ranks (linear spacing)
        // beta[0] = beta_min (hottest), beta[R-1] = beta_max (coldest)
        double beta_min = 1.0 / Tmax;
        double beta_max = 1.0 / Tmin;
        vector<double> beta(R);
        for (int i = 0; i < R; ++i) {
            beta[i] = beta_min + (beta_max - beta_min) * double(i) / double(R - 1);
        }
        
        // Each rank's current temperature
        double my_beta = beta[rank];
        double my_T = 1.0 / my_beta;
        double sigma = 1000.0;
        
        // Warmup phase
        if (rank == 0) cout << "Warming up replicas..." << endl;
        for (size_t i = 0; i < warmup_sweeps; ++i) {
            metropolis(my_T, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                overrelaxation();
            }
        }
        MPI_Barrier(comm);
        
        // Acceptance rate tracking
        vector<double> acceptance_rates(R - 1, 0.0);
        double base_damping = 0.5;  // Less aggressive damping for faster convergence
        
        // Feedback optimization loop
        for (size_t iter = 0; iter < feedback_iters; ++iter) {
            // Damping increases over iterations for stability
            double damping = base_damping + 0.3 * (double(iter) / double(feedback_iters));
            
            // Local acceptance counters for this rank's edge
            // rank k tracks exchanges with rank k+1
            int local_attempts = 0;
            int local_accepts = 0;
            
            // Use full sweeps for better statistics (reduced noise)
            size_t effective_sweeps = sweeps_per_iter;
            
            // MC sweeps with replica exchanges
            for (size_t sweep = 0; sweep < effective_sweeps; ++sweep) {
                // Local MC update
                metropolis(my_T, gaussian_move, sigma);
                if (overrelaxation_rate > 0 && sweep % overrelaxation_rate == 0) {
                    overrelaxation();
                }
                
                // Attempt replica exchanges using checkerboard pattern
                for (int parity = 0; parity <= 1; ++parity) {
                    int partner_rank;
                    if (parity == 0) {
                        partner_rank = (rank % 2 == 0) ? rank + 1 : rank - 1;
                    } else {
                        partner_rank = (rank % 2 == 1) ? rank + 1 : rank - 1;
                    }
                    
                    if (partner_rank < 0 || partner_rank >= R) continue;
                    
                    // Exchange energies with partner
                    double my_E = total_energy(spins);
                    double partner_E;
                    MPI_Sendrecv(&my_E, 1, MPI_DOUBLE, partner_rank, 0,
                                &partner_E, 1, MPI_DOUBLE, partner_rank, 0,
                                comm, MPI_STATUS_IGNORE);
                    
                    // Lower rank computes acceptance
                    int accept_int = 0;
                    if (rank < partner_rank) {
                        // rank is hotter (smaller β), partner is colder (larger β)
                        double beta_hot = my_beta;
                        double beta_cold = beta[partner_rank];
                        double E_hot = my_E;
                        double E_cold = partner_E;
                        
                        // Δ = -(β_cold - β_hot)(E_hot - E_cold)
                        double delta = -(beta_cold - beta_hot) * (E_hot - E_cold);
                        bool accept = (delta >= 0) || (random_double_lehman(0.0, 1.0) < std::exp(delta));
                        accept_int = accept ? 1 : 0;
                        
                        // Track statistics for edge (rank, rank+1)
                        ++local_attempts;
                        if (accept) ++local_accepts;
                    }
                    
                    // Communicate decision
                    int recv_accept_int = 0;
                    MPI_Sendrecv(&accept_int, 1, MPI_INT, partner_rank, 1,
                                &recv_accept_int, 1, MPI_INT, partner_rank, 1,
                                comm, MPI_STATUS_IGNORE);
                    
                    bool accept = (rank < partner_rank) ? (accept_int == 1) : (recv_accept_int == 1);
                    
                    // Exchange configurations if accepted
                    if (accept) {
                        vector<double> send_buf(lattice_size * spin_dim);
                        vector<double> recv_buf(lattice_size * spin_dim);
                        
                        for (size_t i = 0; i < lattice_size; ++i) {
                            for (size_t j = 0; j < spin_dim; ++j) {
                                send_buf[i * spin_dim + j] = spins[i](j);
                            }
                        }
                        
                        MPI_Sendrecv(send_buf.data(), send_buf.size(), MPI_DOUBLE, partner_rank, 2,
                                    recv_buf.data(), recv_buf.size(), MPI_DOUBLE, partner_rank, 2,
                                    comm, MPI_STATUS_IGNORE);
                        
                        for (size_t i = 0; i < lattice_size; ++i) {
                            for (size_t j = 0; j < spin_dim; ++j) {
                                spins[i](j) = recv_buf[i * spin_dim + j];
                            }
                        }
                    }
                }
            }
            
            // Gather acceptance statistics to rank 0 using MPI_Gather
            // Each rank k (for k < R-1) tracks statistics for edge k→k+1
            // Only the lower rank in each pair computes the acceptance
            
            // Use MPI_Gather for cleaner collection
            // Note: rank R-1 doesn't have an edge to track, but we still need it to participate
            
            // Create send buffers - rank k sends stats for edge k (if k < R-1)
            int my_attempts = (rank < R - 1) ? local_attempts : 0;
            int my_accepts = (rank < R - 1) ? local_accepts : 0;
            
            // Gather from all ranks to rank 0
            vector<int> recv_attempts(R);
            vector<int> recv_accepts(R);
            MPI_Gather(&my_attempts, 1, MPI_INT, recv_attempts.data(), 1, MPI_INT, 0, comm);
            MPI_Gather(&my_accepts, 1, MPI_INT, recv_accepts.data(), 1, MPI_INT, 0, comm);
            
            // Rank 0 computes new beta array
            bool converged = false;
            if (rank == 0) {
                // Compute acceptance rates from gathered data
                // recv_attempts[k] contains attempts for edge k→k+1
                for (int e = 0; e < R - 1; ++e) {
                    if (recv_attempts[e] > 0) {
                        acceptance_rates[e] = double(recv_accepts[e]) / double(recv_attempts[e]);
                    }
                }
                
                // Check convergence using mean deviation (more stable than max)
                double max_deviation = 0.0;
                double mean_deviation = 0.0;
                double mean_rate = 0.0;
                double min_rate = 1.0, max_rate = 0.0;
                for (int e = 0; e < R - 1; ++e) {
                    double dev = std::abs(acceptance_rates[e] - target_acceptance);
                    max_deviation = std::max(max_deviation, dev);
                    mean_deviation += dev;
                    mean_rate += acceptance_rates[e];
                    min_rate = std::min(min_rate, acceptance_rates[e]);
                    max_rate = std::max(max_rate, acceptance_rates[e]);
                }
                mean_rate /= (R - 1);
                mean_deviation /= (R - 1);
                
                cout << "Iter " << iter + 1 << "/" << feedback_iters 
                     << ": mean A = " << std::fixed << std::setprecision(3) << mean_rate
                     << " [" << min_rate << ", " << max_rate << "]"
                     << ", mean dev = " << mean_deviation << endl;
                
                // Warn if acceptance is uniformly low - need more replicas
                if (mean_rate < 0.1 && iter == 0) {
                    cout << "WARNING: Mean acceptance rate is very low (" << mean_rate << ").\n"
                         << "         Consider using more replicas or a smaller temperature range." << endl;
                }
                
                result.feedback_iterations_used = iter + 1;
                
                // Converge based on mean deviation (less sensitive to statistical noise)
                if (mean_deviation < convergence_tol) {
                    converged = true;
                    cout << "Converged at iteration " << iter + 1 << endl;
                }
                
                if (!converged) {
                    // Katzgraber et al. feedback-optimized temperature placement:
                    // 
                    // Minimize round-trip time τ_rt = (Σ_i 1/f_i)² where f_i is the
                    // current fraction at edge i, proportional to A_i × Δβ_i.
                    // Optimal solution: Δβ_i ∝ A_i
                    //   - Low A_i (bottleneck) → small Δβ_i (dense temperatures)  
                    //   - High A_i (easy region) → large Δβ_i (sparse temperatures)
                    
                    // Compute weights proportional to acceptance rate
                    vector<double> weights(R - 1);
                    double total_weight = 0.0;
                    
                    for (int e = 0; e < R - 1; ++e) {
                        double A_e = acceptance_rates[e];
                        if (A_e < 0.01) A_e = 0.01;
                        if (A_e > 0.99) A_e = 0.99;
                        
                        // Weight proportional to A: high A → more spacing
                        weights[e] = A_e;
                        total_weight += weights[e];
                    }
                    
                    // Normalize weights to sum to 1
                    for (int e = 0; e < R - 1; ++e) {
                        weights[e] /= total_weight;
                    }
                    
                    // Compute new beta positions
                    vector<double> new_beta(R);
                    new_beta[0] = beta_min;
                    
                    double cumulative = 0.0;
                    for (int e = 0; e < R - 1; ++e) {
                        cumulative += weights[e];
                        new_beta[e + 1] = beta_min + cumulative * (beta_max - beta_min);
                    }
                    new_beta[R - 1] = beta_max;  // Ensure exact endpoint
                    
                    // Apply damping: blend old and new positions
                    for (int k = 1; k < R - 1; ++k) {
                        new_beta[k] = (1.0 - damping) * beta[k] + damping * new_beta[k];
                    }
                    
                    beta = new_beta;
                }
            }
            
            // Broadcast convergence flag and new beta array to all ranks
            int conv_int = converged ? 1 : 0;
            MPI_Bcast(&conv_int, 1, MPI_INT, 0, comm);
            MPI_Bcast(beta.data(), R, MPI_DOUBLE, 0, comm);
            
            // Update local temperature
            my_beta = beta[rank];
            my_T = 1.0 / my_beta;
            
            if (conv_int == 1) {
                result.converged = true;
                break;
            }
        }
        
        // Broadcast final acceptance rates
        MPI_Bcast(acceptance_rates.data(), R - 1, MPI_DOUBLE, 0, comm);
        
        // ================================================================
        // PHASE 2: Bittner et al. adaptive sweep schedule
        // Each rank measures τ_int(T) at its temperature, then we gather
        // to build the temperature-dependent sweep schedule.
        // ================================================================
        if (rank == 0) {
            cout << "\nMeasuring autocorrelation times for adaptive sweep schedule..." << endl;
        }
        
        // Collect energy time series at this rank's temperature
        size_t tau_samples = std::max(size_t(500), sweeps_per_iter);
        vector<double> energy_series;
        energy_series.reserve(tau_samples);
        
        for (size_t i = 0; i < tau_samples; ++i) {
            metropolis(my_T, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                overrelaxation();
            }
            energy_series.push_back(total_energy(spins));
        }
        
        // Compute local autocorrelation time
        AutocorrelationResult acf = compute_autocorrelation(energy_series, 1);
        double my_tau_int = std::max(1.0, acf.tau_int);
        
        // Gather τ_int from all ranks
        vector<double> all_tau_int(R);
        MPI_Allgather(&my_tau_int, 1, MPI_DOUBLE, all_tau_int.data(), 1, MPI_DOUBLE, comm);
        
        // Compute sweep schedule: n_sweeps_i = n_base × τ_int(T_i) / min(τ_int)
        double tau_min_val = *std::min_element(all_tau_int.begin(), all_tau_int.end());
        size_t n_base = 10;  // Minimum sweeps between exchanges
        
        result.autocorrelation_times = all_tau_int;
        result.sweeps_per_temp.resize(R);
        for (int k = 0; k < R; ++k) {
            result.sweeps_per_temp[k] = std::max(size_t(1),
                static_cast<size_t>(std::ceil(n_base * all_tau_int[k] / tau_min_val)));
        }
        
        if (rank == 0) {
            cout << "Autocorrelation times and sweep schedule:" << endl;
            for (int k = 0; k < std::min(R, 15); ++k) {
                cout << "  T[" << k << "] = " << std::scientific << std::setprecision(4) 
                     << 1.0 / beta[k] << "  τ_int = " << std::fixed << std::setprecision(1)
                     << all_tau_int[k] << "  n_sweeps = " << result.sweeps_per_temp[k] << endl;
            }
            if (R > 15) cout << "  ... (" << R - 15 << " more)" << endl;
        }
        
        // Build result (on all ranks)
        result.temperatures.resize(R);
        for (int i = 0; i < R; ++i) {
            result.temperatures[i] = 1.0 / beta[i];
        }
        std::sort(result.temperatures.begin(), result.temperatures.end());
        
        result.acceptance_rates = acceptance_rates;
        
        // Compute diagnostics
        result.local_diffusivities.resize(R - 1);
        for (int e = 0; e < R - 1; ++e) {
            double A = acceptance_rates[e];
            result.local_diffusivities[e] = A * (1.0 - A);
        }
        
        result.mean_acceptance_rate = 0.0;
        for (double A : acceptance_rates) {
            result.mean_acceptance_rate += A;
        }
        result.mean_acceptance_rate /= (R - 1);
        
        // Compute round-trip time using Katzgraber formula with Bittner sweeps:
        // τ_rt ∝ Σ_i n_avg_i / f_i where f_i ∝ A_i × Δβ_i (current fraction)
        double sum_inv_f = 0.0;
        double total_current = 0.0;
        for (int e = 0; e < R - 1; ++e) {
            double d_beta = std::abs(beta[e + 1] - beta[e]);
            double A = std::max(acceptance_rates[e], 1e-6);
            total_current += A * d_beta;
        }
        for (int e = 0; e < R - 1; ++e) {
            double d_beta = std::abs(beta[e + 1] - beta[e]);
            double A = std::max(acceptance_rates[e], 1e-6);
            double f_i = A * d_beta / total_current;
            double n_avg = 0.5 * (result.sweeps_per_temp[e] + result.sweeps_per_temp[e + 1]);
            sum_inv_f += n_avg / f_i;
        }
        result.round_trip_estimate = sum_inv_f;
        
        // Print summary (rank 0 only)
        if (rank == 0) {
            cout << "\n=== Optimized Temperature Grid Summary ===" << endl;
            cout << "Temperatures (ascending):" << endl;
            for (int k = 0; k < std::min(R, 15); ++k) {
                cout << "  T[" << k << "] = " << std::scientific << std::setprecision(6) 
                     << result.temperatures[k];
                if (k < R - 1) {
                    cout << "  (A = " << std::fixed << std::setprecision(3) 
                         << acceptance_rates[k] << ")";
                }
                cout << endl;
            }
            if (R > 15) cout << "  ... (" << R - 15 << " more)" << endl;
            
            cout << "\nMean acceptance rate: " << std::fixed << std::setprecision(3) 
                 << result.mean_acceptance_rate * 100 << "%" << endl;
            cout << "Estimated round-trip time scale: " << std::scientific 
                 << result.round_trip_estimate << endl;
            cout << "Converged: " << (result.converged ? "YES" : "NO") << endl;
        }
        
        // CRITICAL: Synchronize all ranks before returning
        // This ensures no stray MPI messages remain in flight and all ranks
        // exit the optimization at the same time, ready for the main PT simulation
        MPI_Barrier(comm);
        
        // NOTE: Spin configurations have been modified by replica exchanges during optimization.
        // The caller (e.g., parallel_tempering) will re-equilibrate at the assigned temperature,
        // so this is not a problem. However, if fresh random starts are desired, call init_random()
        // after this function returns.
        
        return result;
    }

