/**
 * @file mixed_lattice_pt.cpp
 * @brief MixedLattice parallel-tempering driver + replica-exchange.
 *
 * The PT driver and replica-exchange attempt for MixedLattice were the
 * two single biggest inline methods in `mixed_lattice.h`. They drag in
 * `<mpi.h>`, a lot of HDF5-guarded output, and heavy container
 * bookkeeping — none of which other translation units need.
 */

#include "classical_spin/lattice/mixed_lattice.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#include <mpi.h>

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

// ---- MixedLattice::parallel_tempering ----
    void MixedLattice::parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move, bool use_interleaved,
                           MPI_Comm comm, bool verbose,
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
        
        if (size != static_cast<int>(temp.size())) {
            if (rank == 0) {
                cout << "Error: Number of MPI ranks (" << size 
                     << ") must equal number of temperatures (" << temp.size() << ")" << endl;
            }
            return;
        }
        
        double curr_Temp = temp[rank];
        double sigma = 1000.0;
        int swap_accept = 0;
        double curr_accept = 0;
        
        // Determine exchange frequency: Bittner adaptive or fixed
        bool use_adaptive_sweeps = !sweeps_per_temp.empty() && sweeps_per_temp.size() >= static_cast<size_t>(size);
        size_t effective_swap_rate = swap_rate;
        if (use_adaptive_sweeps) {
            effective_swap_rate = *std::max_element(sweeps_per_temp.begin(), sweeps_per_temp.end());
            if (rank == 0) {
                cout << "Using Bittner adaptive sweep schedule:" << endl;
                cout << "  Exchange frequency set to max(sweeps_per_temp) = " << effective_swap_rate << endl;
            }
        }
        
        // Determine if we should use interleaved mode (beneficial with mixed interactions)
        const bool has_mixed = (num_bi_SU2_SU3 > 0 || num_tri_SU2_SU3 > 0);
        const bool do_interleaved = use_interleaved && has_mixed;
        
        if (rank == 0 && do_interleaved) {
            cout << "Using interleaved sweeps (mixed interactions detected)" << endl;
        }
        
        vector<double> energies;
        vector<pair<SpinVector, SpinVector>> magnetizations; // (SU2, SU3)
        vector<MixedMeasurement> measurements;  // NEW: comprehensive measurements
        size_t expected_samples = n_measure / probe_rate + 100;
        energies.reserve(expected_samples);
        magnetizations.reserve(expected_samples);
        measurements.reserve(expected_samples);
        
        cout << "Rank " << rank << ": T=" << curr_Temp << endl;
        
        // Equilibration phase
        cout << "Rank " << rank << ": Equilibrating..." << endl;
        for (size_t i = 0; i < n_anneal; ++i) {
            if (overrelaxation_rate > 0) {
                if (do_interleaved) {
                    overrelaxation_interleaved();
                } else {
                    overrelaxation();
                }
                if (i % overrelaxation_rate == 0) {
                    if (do_interleaved) {
                        curr_accept += metropolis_interleaved(curr_Temp, gaussian_move, sigma);
                    } else {
                        curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
                    }
                }
            } else {
                if (do_interleaved) {
                    curr_accept += metropolis_interleaved(curr_Temp, gaussian_move, sigma);
                } else {
                    curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
                }
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
                    if (do_interleaved) {
                        overrelaxation_interleaved();
                    } else {
                        overrelaxation();
                    }
                    if (i % overrelaxation_rate == 0) {
                        if (do_interleaved) {
                            metropolis_interleaved(curr_Temp, gaussian_move, sigma);
                        } else {
                            metropolis(curr_Temp, gaussian_move, sigma);
                        }
                    }
                } else {
                    if (do_interleaved) {
                        metropolis_interleaved(curr_Temp, gaussian_move, sigma);
                    } else {
                        metropolis(curr_Temp, gaussian_move, sigma);
                    }
                }
                if (effective_swap_rate > 0 && i % effective_swap_rate == 0) {
                    attempt_replica_exchange(rank, size, temp, curr_Temp, i / effective_swap_rate, comm);
                }
                if (i % pilot_interval == 0) {
                    pilot_energies.push_back(total_energy());
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
                if (do_interleaved) {
                    overrelaxation_interleaved();
                } else {
                    overrelaxation();
                }
                if (i % overrelaxation_rate == 0) {
                    if (do_interleaved) {
                        curr_accept += metropolis_interleaved(curr_Temp, gaussian_move, sigma);
                    } else {
                        curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
                    }
                }
            } else {
                if (do_interleaved) {
                    curr_accept += metropolis_interleaved(curr_Temp, gaussian_move, sigma);
                } else {
                    curr_accept += metropolis(curr_Temp, gaussian_move, sigma);
                }
            }
            
            if (effective_swap_rate > 0 && i % effective_swap_rate == 0) {
                swap_accept += attempt_replica_exchange(rank, size, temp, curr_Temp, i / effective_swap_rate, comm);
            }
            
            if (i % probe_rate == 0) {
                energies.push_back(total_energy());
                magnetizations.push_back({magnetization_SU2(), magnetization_SU3()});
                measurements.push_back(measure_all_observables());  // NEW
            }
        }
        
        cout << "Rank " << rank << ": Collected " << energies.size() << " samples" << endl;
        
        // Compute comprehensive thermodynamic observables with binning analysis
        MixedThermodynamicObservables obs = compute_thermodynamic_observables(measurements, curr_Temp);
        
        double curr_heat_capacity = obs.specific_heat.value;
        double curr_dHeat = obs.specific_heat.error;
        
        // Gather heat capacity to root
        vector<double> heat_capacity, dHeat;
        if (rank == 0) {
            heat_capacity.resize(size);
            dHeat.resize(size);
        }
        
        MPI_Gather(&curr_heat_capacity, 1, MPI_DOUBLE, heat_capacity.data(), 
                   1, MPI_DOUBLE, 0, comm);
        MPI_Gather(&curr_dHeat, 1, MPI_DOUBLE, dHeat.data(), 
                   1, MPI_DOUBLE, 0, comm);
        
        // Report statistics
        double total_steps = n_anneal + n_measure;
        // Account for overrelaxation: Metropolis is only called every overrelaxation_rate steps
        double metro_steps = (overrelaxation_rate > 0) ? total_steps / overrelaxation_rate : total_steps;
        double acc_rate = curr_accept / metro_steps;
        double swap_rate_actual = (effective_swap_rate > 0) ? double(swap_accept) / (total_steps / effective_swap_rate) : 0.0;
        
        cout << "Rank " << rank << ": T=" << curr_Temp 
             << ", acc=" << acc_rate 
             << ", swap_acc=" << swap_rate_actual 
             << ", <E>/N=" << obs.energy_total.value << "±" << obs.energy_total.error
             << ", C_V=" << obs.specific_heat.value << "±" << obs.specific_heat.error
             << endl;
        
        // Save results with proper MPI synchronization
        if (!dir_name.empty()) {
            // Rank 0 creates the main output directory first
            if (rank == 0) {
                ensure_directory_exists(dir_name);
            }
            // Ensure directory exists before other ranks proceed
            MPI_Barrier(comm);
            
            // Check if this rank should write (supports FULL mode with sentinel -1)
            bool should_write = should_rank_write(rank, rank_to_write);
            
            if (should_write) {
                string rank_dir = dir_name + "/rank_" + std::to_string(rank);
                // Each rank creates its own subdirectory (no race condition)
                ensure_directory_exists(rank_dir);
                
                // Save to HDF5 format (single file with all data)
#ifdef HDF5_ENABLED
                save_thermodynamic_observables_hdf5(rank_dir, obs, energies, magnetizations,
                                                   measurements, n_anneal, n_measure,
                                                   probe_rate, swap_rate, overrelaxation_rate,
                                                   acc_rate, swap_rate_actual);
#else
                if (rank == 0) {
                    cerr << "Warning: HDF5 not enabled. Compile with -DHDF5_ENABLED=ON to enable output." << endl;
                }
#endif
                
                // Save spin configuration only if verbose mode is enabled
                if (verbose) {
                    save_spin_config_to_dir(rank_dir, "spins_T=" + std::to_string(curr_Temp));
                }
            }
            
            // Wait for all ranks to finish writing before rank 0 writes aggregated results
            MPI_Barrier(comm);
            
            // Root process saves aggregated heat capacity
            if (rank == 0) {
#ifdef HDF5_ENABLED
                // Save heat capacity to HDF5 (using the same function from lattice.h)
                save_heat_capacity_hdf5(dir_name, temp, heat_capacity, dHeat);
#endif
            }
        }
        
        MPI_Barrier(comm);
        
        if (rank == 0) {
            cout << "Parallel tempering completed" << endl;
        }
    }

// ---- MixedLattice::attempt_replica_exchange ----
    int MixedLattice::attempt_replica_exchange(int rank, int size, const vector<double>& temp,
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
        double E = total_energy();
        double E_partner, T_partner = temp[partner_rank];
        
        MPI_Sendrecv(&E, 1, MPI_DOUBLE, partner_rank, 0,
                    &E_partner, 1, MPI_DOUBLE, partner_rank, 0,
                    comm, MPI_STATUS_IGNORE);
        
        // Decide acceptance using parallel tempering Metropolis criterion:
        // P_swap = min(1, exp[Δ]) where Δ = (β_cold - β_hot)(E_cold - E_hot)
        // With ordering: rank 0 = coldest (β large), highest rank = hottest (β small)
        // rank < partner_rank means: curr is COLDER, partner is HOTTER
        bool accept = false;
        if (rank < partner_rank) {
            // Only the lower rank computes the acceptance decision
            // curr = cold (i), partner = hot (j)
            // β_curr > β_partner, typically E_curr < E_partner (cold has lower energy)
            // Δ = (β_curr - β_partner)(E_curr - E_partner) = (positive)(negative) = negative typically
            double beta_curr = 1.0 / curr_Temp;
            double beta_partner = 1.0 / T_partner;
            double delta = (beta_curr - beta_partner) * (E - E_partner);
            accept = (delta >= 0) || (random_double_lehman(0.0, 1.0) < std::exp(delta));
            
            // DEBUG: Print exchange details (only first few times to avoid spam)
            static int debug_count = 0;
            if (debug_count < 20) {
                cout << "[PT DEBUG] rank=" << rank << "->" << partner_rank
                     << " T_cold=" << curr_Temp << " T_hot=" << T_partner
                     << " E_cold=" << E << " E_hot=" << E_partner
                     << " delta=" << delta << " accept=" << accept << endl;
                ++debug_count;
            }
        }
        
        // Communicate decision between partners using point-to-point
        // (MPI_Bcast is collective and cannot be used pairwise)
        int accept_int = accept ? 1 : 0;
        int recv_accept_int = 0;
        MPI_Sendrecv(&accept_int, 1, MPI_INT, partner_rank, 3,
                     &recv_accept_int, 1, MPI_INT, partner_rank, 3,
                     comm, MPI_STATUS_IGNORE);
        // The lower-ranked partner made the decision; both use that result
        accept = (rank < partner_rank) ? (accept_int == 1) : (recv_accept_int == 1);
        
        // Exchange configurations if accepted
        if (accept) {
            // Serialize SU(2) spins
            vector<double> send_buf_SU2(lattice_size_SU2 * spin_dim_SU2);
            vector<double> recv_buf_SU2(lattice_size_SU2 * spin_dim_SU2);
            
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                for (size_t j = 0; j < spin_dim_SU2; ++j) {
                    send_buf_SU2[i * spin_dim_SU2 + j] = spins_SU2[i](j);
                }
            }
            
            MPI_Sendrecv(send_buf_SU2.data(), send_buf_SU2.size(), MPI_DOUBLE, partner_rank, 1,
                        recv_buf_SU2.data(), recv_buf_SU2.size(), MPI_DOUBLE, partner_rank, 1,
                        comm, MPI_STATUS_IGNORE);
            
            // Deserialize SU(2)
            for (size_t i = 0; i < lattice_size_SU2; ++i) {
                for (size_t j = 0; j < spin_dim_SU2; ++j) {
                    spins_SU2[i](j) = recv_buf_SU2[i * spin_dim_SU2 + j];
                }
            }
            
            // Serialize SU(3) spins
            vector<double> send_buf_SU3(lattice_size_SU3 * spin_dim_SU3);
            vector<double> recv_buf_SU3(lattice_size_SU3 * spin_dim_SU3);
            
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                for (size_t j = 0; j < spin_dim_SU3; ++j) {
                    send_buf_SU3[i * spin_dim_SU3 + j] = spins_SU3[i](j);
                }
            }
            
            MPI_Sendrecv(send_buf_SU3.data(), send_buf_SU3.size(), MPI_DOUBLE, partner_rank, 2,
                        recv_buf_SU3.data(), recv_buf_SU3.size(), MPI_DOUBLE, partner_rank, 2,
                        comm, MPI_STATUS_IGNORE);
            
            // Deserialize SU(3)
            for (size_t i = 0; i < lattice_size_SU3; ++i) {
                for (size_t j = 0; j < spin_dim_SU3; ++j) {
                    spins_SU3[i](j) = recv_buf_SU3[i * spin_dim_SU3 + j];
                }
            }
        }
        
        return accept ? 1 : 0;
    }

