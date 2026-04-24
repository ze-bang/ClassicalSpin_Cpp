/**
 * @file lattice_md.cpp
 * @brief Lattice molecular dynamics + pulse / pump-probe drivers.
 *
 * Hosts the Landau–Lifshitz RHS, ODE stepper glue, pulse-field logic,
 * and the full pump-probe spectroscopy drivers (single-process and
 * MPI-parallel). These methods drag in `boost::odeint`, fmt-heavy
 * trajectory I/O, and a lot of drive-field conditionals — all of which
 * the header previously forced on every TU that included
 * `classical_spin/lattice/lattice.h`.
 *
 * Kept in the header (and therefore NOT moved here):
 *   - `integrate_ode_system<System, Observer>` — templated, must be
 *     visible at the point of instantiation.
 *   - All CUDA-guarded `*_gpu` stubs / overloads — they live in paired
 *     `#ifdef CUDA_ENABLED` / `#else` blocks that are awkward to split.
 */

#include "classical_spin/lattice/lattice.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <boost/numeric/odeint.hpp>

// ---- Lattice::landau_lifshitz_flat ----
    void Lattice::landau_lifshitz_flat(const double* state_flat, double* dsdt_flat, double t) const {
        // Drive-field envelope: factor1, factor2 depend only on `t`, not on
        // the site, so we hoist the two `exp + cos` calls out of the per-site
        // loop. When the drive amplitude is exactly zero (the common case
        // during MC + free LLG evolution), we also skip the drive call
        // entirely via an outer branch instead of per-site.
        const bool drive_active = (field_drive_amp != 0.0);
        double drive_factor1 = 0.0, drive_factor2 = 0.0;
        if (drive_active) {
            const double dt1 = t - t_pulse[0];
            const double dt2 = t - t_pulse[1];
            drive_factor1 = field_drive_amp *
                            std::exp(-std::pow(dt1 / (2.0 * field_drive_width), 2)) *
                            std::cos(field_drive_freq * dt1);
            drive_factor2 = field_drive_amp *
                            std::exp(-std::pow(dt2 / (2.0 * field_drive_width), 2)) *
                            std::cos(field_drive_freq * dt2);
        }

        if (spin_dim == 3) {
            // SU(2): Standard cross product dS/dt = H × S
            for (size_t i = 0; i < lattice_size; ++i) {
                const size_t idx = i * 3;

                double H[3];
                get_local_field_flat(state_flat, i, H);
                if (drive_active) {
                    apply_drive_field_flat(i, drive_factor1, drive_factor2, H);
                }

                const double Sx = state_flat[idx + 0];
                const double Sy = state_flat[idx + 1];
                const double Sz = state_flat[idx + 2];

                // Cross product: dS/dt = H × S
                double dSx = H[1] * Sz - H[2] * Sy;
                double dSy = H[2] * Sx - H[0] * Sz;
                double dSz = H[0] * Sy - H[1] * Sx;

                if (alpha_gilbert > 0.0) {
                    // Gilbert damping: -α/|S| S × (S × H)
                    // S × H = (dSx, dSy, dSz), so S × (S × H):
                    double SxdS_x = Sy * dSz - Sz * dSy;
                    double SxdS_y = Sz * dSx - Sx * dSz;
                    double SxdS_z = Sx * dSy - Sy * dSx;
                    double inv_S = 1.0 / std::sqrt(Sx*Sx + Sy*Sy + Sz*Sz);
                    dSx -= alpha_gilbert * inv_S * SxdS_x;
                    dSy -= alpha_gilbert * inv_S * SxdS_y;
                    dSz -= alpha_gilbert * inv_S * SxdS_z;
                }

                dsdt_flat[idx + 0] = dSx;
                dsdt_flat[idx + 1] = dSy;
                dsdt_flat[idx + 2] = dSz;
            }
        } else if (spin_dim == 8) {
            // SU(3): structure-constant contraction dS_i/dt = sum_{jk} f_{ijk} H_j S_k
            //
            // The structure constants f_{ijk} are the antisymmetric Gell-Mann
            // constants — extremely sparse, with only 9 unique non-zero
            // (a<b<c) triples. The full 8×8×8 textbook contraction does 512
            // mul-adds per site, of which 458 are 0 * something. The
            // hand-rolled `cross_prod_SU3_flat` evaluates only the 54 non-zero
            // entries directly, fully inlined, with no Eigen alloc; on the
            // SU(3) RHS this is roughly an order of magnitude faster.
            for (size_t site = 0; site < lattice_size; ++site) {
                const size_t idx = site * 8;

                double H[8];
                get_local_field_flat(state_flat, site, H);
                if (drive_active) {
                    apply_drive_field_flat(site, drive_factor1, drive_factor2, H);
                }

                const double* S = &state_flat[idx];
                cross_prod_SU3_flat(H, S, &dsdt_flat[idx], /*accumulate=*/false);
            }
        } else {
            // General case: use cross_product function (fallback)
            for (size_t i = 0; i < lattice_size; ++i) {
                const size_t idx = i * spin_dim;

                double H_arr[16];  // Max reasonable spin_dim
                get_local_field_flat(state_flat, i, H_arr);
                if (drive_active) {
                    apply_drive_field_flat(i, drive_factor1, drive_factor2, H_arr);
                }

                SpinVector H_eff = Eigen::Map<const Eigen::VectorXd>(H_arr, spin_dim);
                SpinVector S_i = Eigen::Map<const Eigen::VectorXd>(&state_flat[idx], spin_dim);

                SpinVector dS_dt = cross_product(H_eff, S_i);

                for (size_t d = 0; d < spin_dim; ++d) {
                    dsdt_flat[idx + d] = dS_dt(d);
                }
            }
        }
    }

// ---- Lattice::drive_field_at_time_flat ----
    void Lattice::drive_field_at_time_flat(double t, size_t site_index, double* H) const {
        // Public/legacy entry point: keep the original semantics (compute the
        // envelopes from `t` on each call). The hot LLG path uses the
        // hoisted helper `apply_drive_field_flat` directly so that the two
        // exp+cos calls are not repeated per site.
        if (field_drive_amp == 0.0) return;

        const double dt1 = t - t_pulse[0];
        const double dt2 = t - t_pulse[1];

        const double factor1 = field_drive_amp *
                        std::exp(-std::pow(dt1 / (2.0 * field_drive_width), 2)) *
                        std::cos(field_drive_freq * dt1);

        const double factor2 = field_drive_amp *
                        std::exp(-std::pow(dt2 / (2.0 * field_drive_width), 2)) *
                        std::cos(field_drive_freq * dt2);

        apply_drive_field_flat(site_index, factor1, factor2, H);
    }

// ---- Lattice::apply_drive_field_flat ----
    void Lattice::apply_drive_field_flat(size_t site_index,
                                         double factor1, double factor2,
                                         double* H) const {
        // Subtracts the two-pulse drive contribution from H in place.
        // factor1, factor2 are the time-dependent envelopes (already
        // multiplied by amplitude) — by accepting them as arguments we
        // avoid the per-site exp/cos that would dominate the LLG hot loop.
        const size_t atom = site_index % N_atoms;
        const double* fd0 = field_drive[0].data() + atom * spin_dim;
        const double* fd1 = field_drive[1].data() + atom * spin_dim;
        for (size_t d = 0; d < spin_dim; ++d) {
            H[d] -= fd0[d] * factor1 + fd1[d] * factor2;
        }
    }

// ---- Lattice::drive_field_at_time ----
    SpinVector Lattice::drive_field_at_time(double t, size_t site_index) const {
        size_t atom = site_index % N_atoms;
        
        double factor1 = field_drive_amp * 
                        std::exp(-std::pow((t - t_pulse[0]) / (2.0 * field_drive_width), 2)) *
                        std::cos( field_drive_freq * (t - t_pulse[0]));
        
        double factor2 = field_drive_amp * 
                        std::exp(-std::pow((t - t_pulse[1]) / (2.0 * field_drive_width), 2)) *
                        std::cos( field_drive_freq * (t - t_pulse[1]));
        
        return field_drive[0].segment(atom * spin_dim, spin_dim) * factor1 +
               field_drive[1].segment(atom * spin_dim, spin_dim) * factor2;
    }

// ---- Lattice::set_pulse ----
    void Lattice::set_pulse(const vector<SpinVector>& field_in1, double t_B1,
                  const vector<SpinVector>& field_in2, double t_B2,
                  double pulse_amp, double pulse_width, double pulse_freq) {
        // Pack field components, transforming to local frame: B_local = R * B_global
        for (size_t atom = 0; atom < N_atoms; ++atom) {
            field_drive[0].segment(atom * spin_dim, spin_dim) = sublattice_frames[atom] * field_in1[atom];
            field_drive[1].segment(atom * spin_dim, spin_dim) = sublattice_frames[atom] * field_in2[atom];
        }
        
        t_pulse[0] = t_B1;
        t_pulse[1] = t_B2;
        field_drive_amp = pulse_amp;
        field_drive_width = pulse_width;
        field_drive_freq = pulse_freq;
    }

// ---- Lattice::ode_system ----
    void Lattice::ode_system(const ODEState& x, ODEState& dxdt, double t) {
        landau_lifshitz_flat(x.data(), dxdt.data(), t);
    }

// ---- Lattice::molecular_dynamics ----
    void Lattice::molecular_dynamics(double T_start, double T_end, double dt_initial,
                           string out_dir, size_t save_interval,
                           string method, bool use_gpu) {
        if (use_gpu) {
#ifdef CUDA_ENABLED
            molecular_dynamics_gpu(T_start, T_end, dt_initial, out_dir, save_interval, method);
#else
            std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED)." << endl;
            std::cerr << "Falling back to CPU implementation." << endl;
            molecular_dynamics_cpu(T_start, T_end, dt_initial, out_dir, save_interval, method);
#endif
        } else {
            molecular_dynamics_cpu(T_start, T_end, dt_initial, out_dir, save_interval, method);
        }
    }

// ---- Lattice::molecular_dynamics_cpu ----
    void Lattice::molecular_dynamics_cpu(double T_start, double T_end, double dt_initial,
                           string out_dir, size_t save_interval,
                           string method,
                           size_t renorm_interval) {
#ifndef HDF5_ENABLED
        std::cerr << "Error: HDF5 support is required for molecular dynamics output." << endl;
        std::cerr << "Please rebuild with -DHDF5_ENABLED flag and HDF5 libraries." << endl;
        return;
#endif
        
        if (!out_dir.empty()) {
            std::filesystem::create_directories(out_dir);
        }
        
        cout << "Running molecular dynamics with Boost.Odeint: t=" << T_start << " → " << T_end << endl;
        cout << "Integration method: " << method << endl;
        cout << "Initial step size: " << dt_initial << endl;
        
        // Convert current spins to flat state vector
        ODEState state = spins_to_state(spins);
        
        // Create HDF5 writer with comprehensive metadata
        std::unique_ptr<HDF5MDWriter> hdf5_writer;
        if (!out_dir.empty()) {
            string hdf5_file = out_dir + "/trajectory.h5";
            cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
            hdf5_writer = std::make_unique<HDF5MDWriter>(
                hdf5_file, lattice_size, spin_dim, N_atoms, 
                dim1, dim2, dim3, method, 
                dt_initial, T_start, T_end, save_interval, spin_length, 
                &site_positions, 10000);
        }
        
        // Observer to save data + (optionally) project spins back to |S|=spin_length.
        // The Landau-Lifshitz equation conserves |S| analytically, but explicit
        // integrators (dopri5, rk4, ...) accumulate small norm-drift; projecting
        // every N steps keeps the trajectory on the physical Bloch sphere with
        // negligible cost (one sqrt per site per call).
        size_t step_count = 0;
        size_t save_count = 0;
        const bool do_renorm = (renorm_interval > 0) && (spin_dim == 3);
        auto observer = [&](const ODEState& x, double t) {
            if (do_renorm && (step_count > 0) && (step_count % renorm_interval == 0)) {
                // Project to unit sphere of radius spin_length.
                // Observer receives const&; the underlying buffer is owned by the
                // outer `state` ODEState, which we know is the same memory while
                // integrate_const/adaptive holds the internal stepper.
                double* sx = const_cast<double*>(x.data());
                const double sl = static_cast<double>(spin_length);
                for (size_t i = 0; i < lattice_size; ++i) {
                    double* p = sx + i * spin_dim;
                    double n2 = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
                    if (n2 > 0.0) {
                        double s = sl / std::sqrt(n2);
                        p[0] *= s; p[1] *= s; p[2] *= s;
                    }
                }
            }
            if (step_count % save_interval == 0) {
                // Compute magnetizations directly from flat state (zero allocation)
                double M_local_arr[8] = {0};  // Max spin_dim
                double M_antiferro_arr[8] = {0};
                double M_global_arr[8] = {0};
                
                for (size_t i = 0; i < lattice_size; ++i) {
                    for (size_t d = 0; d < spin_dim; ++d) {
                        M_local_arr[d] += x[i * spin_dim + d];
                    }
                }
                
                // Compute global magnetization (sublattice-frame transformed)
                compute_magnetization_global_from_flat(x.data(), M_global_arr);
                // Compute staggered magnetization (sublattice-frame + AFM signs)
                compute_magnetization_staggered_from_flat(x.data(), M_antiferro_arr);
                
                SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
                SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim);
                SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
                
                // Compute accurate energy density directly from flat state (includes bilinear)
                double E = total_energy_flat(x.data()) / lattice_size;
                
                // Write to HDF5 directly from flat state (no conversion needed)
                if (hdf5_writer) {
                    hdf5_writer->write_flat_step(t, M_antiferro, M_local, M_global, x.data());
                    save_count++;
                }
                
                // Progress output
                if (step_count % (save_interval * 10) == 0) {
                    cout << "t=" << t << ", E/N=" << E << ", |M|=" << M_local.norm() << endl;
                }
            }
            ++step_count;
        };
        
        // Create ODE system wrapper for Boost.Odeint
        auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
            this->ode_system(x, dxdt, t);
        };
        
        // Integrate using selected method
        double abs_tol = (method == "bulirsch_stoer") ? 1e-8 : 1e-6;
        double rel_tol = (method == "bulirsch_stoer") ? 1e-8 : 1e-6;
        integrate_ode_system(system_func, state, T_start, T_end, dt_initial,
                            observer, method, true, abs_tol, rel_tol);
        
        // Note: Lattice::spins remains unchanged (initial configuration preserved)
        // The evolved state is stored in the ODEState 'state' variable
        
        // Close HDF5 file
        if (hdf5_writer) {
            hdf5_writer->close();
            cout << "HDF5 trajectory saved with " << save_count << " snapshots" << endl;
        }
        
        cout << "Molecular dynamics complete! (" << step_count << " steps)" << endl;
    }

// ---- Lattice::collect_energy_samples ----
    vector<double> Lattice::collect_energy_samples(size_t n_samples, size_t interval,
                                         double T, bool gaussian_move, double& sigma,
                                         size_t overrelaxation_rate) {
        vector<double> energies;
        energies.reserve(n_samples / interval + 1);
        
        for (size_t i = 0; i < n_samples; ++i) {
            if (overrelaxation_rate > 0) {
                overrelaxation();
                if (i % overrelaxation_rate == 0) {
                    metropolis(T, gaussian_move, sigma);
                }
            } else {
                metropolis(T, gaussian_move, sigma);
            }
            
            if (i % interval == 0) {
                energies.push_back(total_energy(spins));
            }
        }
        
        return energies;
    }

// ---- Lattice::single_pulse_drive ----
    vector<pair<double, array<SpinVector, 3>>> Lattice::single_pulse_drive(
               const vector<SpinVector>& field_in, double t_B, 
               double pulse_amp, double pulse_width, double pulse_freq,
               double T_start, double T_end, double step_size,
               string method, bool use_gpu) {
        
        if (use_gpu) {
#ifdef CUDA_ENABLED
            return single_pulse_drive_gpu(field_in, t_B, pulse_amp, pulse_width, pulse_freq, 
                            T_start, T_end, step_size, method);
#else
            std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED)." << endl;
            std::cerr << "Falling back to CPU implementation." << endl;
            // Fall through to CPU implementation
#endif
        }
        
        // Set up pulse
        set_pulse(field_in, t_B, vector<SpinVector>(N_atoms, SpinVector::Zero(spin_dim)), 
                 0.0, pulse_amp, pulse_width, pulse_freq);
        
        // Storage for trajectory: (time, [M_antiferro, M_local, M_global])
        vector<pair<double, array<SpinVector, 3>>> trajectory;
        
        // Start from initial spins configuration (always use Lattice::spins as starting point)
        ODEState state = spins_to_state(spins);
        
        // Create ODE system wrapper
        auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
            this->ode_system(x, dxdt, t);
        };
        
        // Observer to collect magnetization at regular intervals
        double last_save_time = T_start;
        auto observer = [&](const ODEState& x, double t) {
            if (t - last_save_time >= step_size - 1e-10 || t >= T_end - 1e-10) {
                // Compute magnetizations directly from flat state
                double M_local_arr[8] = {0};
                double M_antiferro_arr[8] = {0};
                double M_global_arr[8] = {0};
                
                compute_magnetizations_from_flat(x.data(), lattice_size, spin_dim, 
                    M_local_arr, M_antiferro_arr);
                
                // Use helper function for global magnetization
                compute_magnetization_global_from_flat(x.data(), M_global_arr);
                // Compute staggered magnetization (sublattice-frame + AFM signs)
                compute_magnetization_staggered_from_flat(x.data(), M_antiferro_arr);
                
                SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
                SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim);
                SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
                
                trajectory.push_back({t, {M_antiferro, M_local, M_global}});
                last_save_time = t;
            }
        };
        
        // Integrate (state was initialized from spins earlier)
        integrate_ode_system(system_func, state, T_start, T_end, step_size,
                            observer, method, false, 1e-10, 1e-10);
        
        // Note: Lattice::spins remains unchanged - only ODEState evolved
        
        // Reset pulse
        field_drive[0] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive[1] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive_amp = 0.0;
        
        return trajectory;
    }

// ---- Lattice::double_pulse_drive ----
    vector<pair<double, array<SpinVector, 3>>> Lattice::double_pulse_drive(
                   const vector<SpinVector>& field_in_1, double t_B_1,
                   const vector<SpinVector>& field_in_2, double t_B_2,
                   double pulse_amp, double pulse_width, double pulse_freq,
                   double T_start, double T_end, double step_size,
                   string method, bool use_gpu) {
        
        if (use_gpu) {
#ifdef CUDA_ENABLED
            return double_pulse_drive_gpu(field_in_1, t_B_1, field_in_2, t_B_2, 
                                pulse_amp, pulse_width, pulse_freq,
                                T_start, T_end, step_size, method);
#else
            std::cerr << "Warning: GPU support not available (compiled without CUDA_ENABLED)." << endl;
            std::cerr << "Falling back to CPU implementation." << endl;
            // Fall through to CPU implementation
#endif
        }
        
        // Set up two-pulse configuration
        set_pulse(field_in_1, t_B_1, field_in_2, t_B_2, 
                 pulse_amp, pulse_width, pulse_freq);
        
        // Storage for trajectory: (time, [M_antiferro, M_local, M_global])
        vector<pair<double, array<SpinVector, 3>>> trajectory;
        
        // Start from initial spins configuration (always use Lattice::spins as starting point)
        ODEState state = spins_to_state(spins);
        
        // Create ODE system wrapper
        auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
            this->ode_system(x, dxdt, t);
        };
        
        // Observer to collect magnetization at regular intervals
        double last_save_time = T_start;
        auto observer = [&](const ODEState& x, double t) {
            if (t - last_save_time >= step_size - 1e-10 || t >= T_end - 1e-10) {
                // Compute magnetizations directly from flat state
                double M_local_arr[8] = {0};
                double M_antiferro_arr[8] = {0};
                double M_global_arr[8] = {0};
                
                compute_magnetizations_from_flat(x.data(), lattice_size, spin_dim, 
                    M_local_arr, M_antiferro_arr);
                
                // Use helper function for global magnetization
                compute_magnetization_global_from_flat(x.data(), M_global_arr);
                // Compute staggered magnetization (sublattice-frame + AFM signs)
                compute_magnetization_staggered_from_flat(x.data(), M_antiferro_arr);
                
                SpinVector M_local = Eigen::Map<Eigen::VectorXd>(M_local_arr, spin_dim) / double(lattice_size);
                SpinVector M_antiferro = Eigen::Map<Eigen::VectorXd>(M_antiferro_arr, spin_dim);
                SpinVector M_global = Eigen::Map<Eigen::VectorXd>(M_global_arr, spin_dim);
                
                trajectory.push_back({t, {M_antiferro, M_local, M_global}});
                last_save_time = t;
            }
        };
        
        // Integrate (state was initialized from spins earlier)
        integrate_ode_system(system_func, state, T_start, T_end, step_size,
                            observer, method, false, 1e-10, 1e-10);
        
        // Note: Lattice::spins remains unchanged - only ODEState evolved
        
        // Reset pulse
        field_drive[0] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive[1] = SpinVector::Zero(N_atoms * spin_dim);
        field_drive_amp = 0.0;
        
        return trajectory;
    }

// ---- Lattice::pump_probe_spectroscopy ----
    void Lattice::pump_probe_spectroscopy(const vector<SpinVector>& field_in,
                                 double pulse_amp, double pulse_width, double pulse_freq,
                                 double tau_start, double tau_end, double tau_step,
                                 double T_start, double T_end, double T_step,
                                 double Temp_start, double Temp_end,
                                 size_t n_anneal,
                                 bool T_zero_quench, size_t quench_sweeps,
                                 string dir_name, string method,
                                 bool use_gpu) {
        
        std::filesystem::create_directories(dir_name);
        
        cout << "\n==========================================" << endl;
        cout << "Pump-Probe Spectroscopy Workflow" << endl;
        cout << "==========================================" << endl;
        cout << "Pulse parameters:" << endl;
        cout << "  Amplitude: " << pulse_amp << endl;
        cout << "  Width: " << pulse_width << endl;
        cout << "  Frequency: " << pulse_freq << endl;
        cout << "Delay scan: " << tau_start << " → " << tau_end << " (step: " << tau_step << ")" << endl;
        cout << "Integration time: " << T_start << " → " << T_end << " (step: " << T_step << ")" << endl;
        
        // Use current spin configuration as ground state (assumed pre-loaded)
        cout << "\n[1/3] Using current configuration as ground state..." << endl;
        double E_ground = energy_density();
        SpinVector M_ground = magnetization_local();
        cout << "  Ground state: E/N = " << E_ground << ", |M| = " << M_ground.norm() << endl;
        
        // Save initial configuration
        save_positions(dir_name + "/positions.txt");
        save_spin_config(dir_name + "/initial_spins.txt");
        
        // Backup ground state
        SpinConfig ground_state = spins;
        
        // Step 2: Reference single-pulse dynamics (pump at t=0)
        cout << "\n[2/3] Running reference single-pulse dynamics (M0)..." << endl;
        if (use_gpu) cout << "  Using GPU acceleration" << endl;
        auto M0_trajectory = single_pulse_drive(field_in, 0.0, pulse_amp, pulse_width, pulse_freq,
                                   T_start, T_end, T_step, method, use_gpu);
        
        // Step 3: Delay time scan
        int tau_steps = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
        cout << "\n[3/3] Scanning delay times (" << tau_steps << " steps)..." << endl;
        
        // Open HDF5 writer BEFORE the loop to write incrementally (avoids OOM)
        string hdf5_file = dir_name + "/pump_probe_spectroscopy.h5";
        
#ifdef HDF5_ENABLED
        HDF5PumpProbeWriter writer(
            hdf5_file,
            lattice_size, spin_dim, N_atoms, dim1, dim2, dim3, spin_length,
            pulse_amp, pulse_width, pulse_freq,
            T_start, T_end, T_step, method,
            tau_start, tau_end, tau_step,
            E_ground, M_ground, Temp_start, Temp_end, n_anneal,
            T_zero_quench, quench_sweeps,
            &field_in, &site_positions
        );
        
        // Write reference trajectory
        writer.write_reference_trajectory(M0_trajectory);
#endif
        
        double current_tau = tau_start;
        for (int i = 0; i < tau_steps; ++i) {
            cout << "\n--- Delay time " << (i+1) << "/" << tau_steps << ": tau = " << current_tau << " ---" << endl;
            
            // Restore ground state for each scan point
            spins = ground_state;
            
            // M1: Probe pulse only at time tau
            cout << "  Computing M1 (probe at tau=" << current_tau << ")..." << endl;
            auto M1_trajectory = single_pulse_drive(field_in, current_tau, pulse_amp, pulse_width, pulse_freq,
                                       T_start, T_end, T_step, method, use_gpu);
            
            // Restore ground state again
            spins = ground_state;
            
            // M01: Pump at t=0 + Probe at t=tau
            cout << "  Computing M01 (pump at 0 + probe at tau=" << current_tau << ")..." << endl;
            auto M01_trajectory = double_pulse_drive(field_in, 0.0, field_in, current_tau,
                                            pulse_amp, pulse_width, pulse_freq,
                                            T_start, T_end, T_step, method, use_gpu);
            
            // Write this tau step immediately and release memory
#ifdef HDF5_ENABLED
            writer.write_tau_trajectory(i, current_tau, M1_trajectory, M01_trajectory);
#endif
            
            current_tau += tau_step;
        }
        
#ifdef HDF5_ENABLED
        writer.close();
        cout << "\n[Complete] All data written incrementally to: " << hdf5_file << endl;
#else
        cout << "Warning: HDF5 support not enabled. Data not saved to HDF5 file." << endl;
        cout << "  Rebuild with -DHDF5_ENABLED to enable HDF5 output." << endl;
#endif
        
        // Restore ground state at end
        spins = ground_state;
        
        cout << "\n==========================================" << endl;
        cout << "Pump-Probe Spectroscopy Complete!" << endl;
        cout << "Output directory: " << dir_name << endl;
        cout << "Total delay points: " << tau_steps << endl;
        cout << "==========================================" << endl;
    }

// ---- Lattice::pump_probe_spectroscopy_mpi ----
    void Lattice::pump_probe_spectroscopy_mpi(const vector<SpinVector>& field_in,
                                     double pulse_amp, double pulse_width, double pulse_freq,
                                     double tau_start, double tau_end, double tau_step,
                                     double T_start, double T_end, double T_step,
                                     double Temp_start, double Temp_end,
                                     size_t n_anneal,
                                     bool T_zero_quench, size_t quench_sweeps,
                                     string dir_name, string method,
                                     bool use_gpu) {
        
        int rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        std::filesystem::create_directories(dir_name);
        
        // Calculate total tau steps
        int tau_steps = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
        const bool scheduler_writer_only = (mpi_size > 1);
        const int worker_count = scheduler_writer_only ? (mpi_size - 1) : 1;
        
        if (rank == 0) {
            cout << "\n==========================================" << endl;
            cout << "Pump-Probe Spectroscopy (MPI Parallel)" << endl;
            cout << "==========================================" << endl;
            cout << "MPI ranks: " << mpi_size << endl;
            cout << "Pulse parameters:" << endl;
            cout << "  Amplitude: " << pulse_amp << endl;
            cout << "  Width: " << pulse_width << endl;
            cout << "  Frequency: " << pulse_freq << endl;
            cout << "Delay scan: " << tau_start << " → " << tau_end << " (step: " << tau_step << ")" << endl;
            cout << "Total delay points: " << tau_steps << endl;
            cout << "Integration time: " << T_start << " → " << T_end << " (step: " << T_step << ")" << endl;
            if (scheduler_writer_only) {
                cout << "Rank 0 role: scheduler/writer only" << endl;
                cout << "Tau points per worker rank: ~" << (tau_steps + worker_count - 1) / worker_count << endl;
            } else {
                cout << "Tau points per rank: ~" << (tau_steps + mpi_size - 1) / mpi_size << endl;
            }
            if (use_gpu) {
                cout << "GPU acceleration: ENABLED (each rank uses assigned GPU)" << endl;
            }
        }
        
        // Use current spin configuration as ground state (assumed pre-loaded)
        if (rank == 0) {
            cout << "\n[1/4] Using current configuration as ground state..." << endl;
        }
        double E_ground = energy_density();
        SpinVector M_ground = magnetization_local();
        if (rank == 0) {
            cout << "  Ground state: E/N = " << E_ground << ", |M| = " << M_ground.norm() << endl;
        }
        
        // Save initial configuration (rank 0 only)
        if (rank == 0) {
            save_positions(dir_name + "/positions.txt");
            save_spin_config(dir_name + "/initial_spins.txt");
        }
        
        // Backup ground state
        SpinConfig ground_state = spins;
        
        // Step 2: Reference single-pulse dynamics (all ranks compute, but only rank 0 keeps result)
        // Actually all ranks need the same M0, so we can compute once and broadcast
        if (rank == 0) {
            cout << "\n[2/4] Running reference single-pulse dynamics (M0)..." << endl;
            if (use_gpu) cout << "  Using GPU acceleration" << endl;
        }
        
        vector<pair<double, array<SpinVector, 3>>> M0_trajectory;
        if (rank == 0) {
            M0_trajectory = single_pulse_drive(field_in, 0.0, pulse_amp, pulse_width, pulse_freq,
                                               T_start, T_end, T_step, method, use_gpu);
        }
        
        // Restore ground state
        spins = ground_state;
        
        // Step 3: Distribute tau values across ranks
        if (rank == 0) {
            cout << "\n[3/4] Distributing tau delays across " << mpi_size << " ranks..." << endl;
        }
        
        // Calculate which tau indices this rank handles
        vector<int> my_tau_indices;
        vector<double> my_tau_values;
        if (!scheduler_writer_only) {
            for (int i = rank; i < tau_steps; i += mpi_size) {
                my_tau_indices.push_back(i);
                my_tau_values.push_back(tau_start + i * tau_step);
            }
        } else if (rank > 0) {
            for (int i = rank - 1; i < tau_steps; i += worker_count) {
                my_tau_indices.push_back(i);
                my_tau_values.push_back(tau_start + i * tau_step);
            }
        }
        
        if (rank == 0) {
            if (scheduler_writer_only) {
                cout << "  Rank 0 processes 0 tau points (scheduler/writer only)" << endl;
            } else {
                cout << "  Each rank processing " << my_tau_indices.size() << " tau points" << endl;
            }
        }
        
        // Local storage for this rank's trajectories
        vector<vector<pair<double, array<SpinVector, 3>>>> local_M1_trajectories;
        vector<vector<pair<double, array<SpinVector, 3>>>> local_M01_trajectories;
        
        local_M1_trajectories.reserve(my_tau_indices.size());
        local_M01_trajectories.reserve(my_tau_indices.size());
        
        // Compute trajectories for assigned tau values
        for (size_t idx = 0; idx < my_tau_indices.size(); ++idx) {
            double current_tau = my_tau_values[idx];
            int global_idx = my_tau_indices[idx];
            
            cout << "[Rank " << rank << "] Computing tau[" << global_idx << "] = " << current_tau 
                 << " (" << (idx+1) << "/" << my_tau_indices.size() << ")" << endl;
            
            // Restore ground state
            spins = ground_state;
            
            // M1: Probe pulse only at time tau
            auto M1_trajectory = single_pulse_drive(field_in, current_tau, pulse_amp, pulse_width, pulse_freq,
                                       T_start, T_end, T_step, method, use_gpu);
            local_M1_trajectories.push_back(M1_trajectory);
            
            // Restore ground state again
            spins = ground_state;
            
            // M01: Pump at t=0 + Probe at t=tau
            auto M01_trajectory = double_pulse_drive(field_in, 0.0, field_in, current_tau,
                                            pulse_amp, pulse_width, pulse_freq,
                                            T_start, T_end, T_step, method, use_gpu);
            local_M01_trajectories.push_back(M01_trajectory);
        }
        
        // Synchronize before gathering
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            cout << "\n[4/4] Gathering results from all ranks..." << endl;
        }
        
        // Compute sizes for serialization
        unsigned long long time_points_ull = 0;
        if (rank == 0) {
            time_points_ull = static_cast<unsigned long long>(M0_trajectory.size());
        } else if (!local_M1_trajectories.empty()) {
            time_points_ull = static_cast<unsigned long long>(local_M1_trajectories.front().size());
        }
        MPI_Bcast(&time_points_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        size_t time_points = static_cast<size_t>(time_points_ull);
        size_t data_per_point = 1 + 3 * spin_dim;  // time + 3 SpinVectors
        size_t traj_size = time_points * data_per_point;
        
        // Compute tau values (needed for HDF5)
        vector<double> tau_values(tau_steps);
        for (int i = 0; i < tau_steps; ++i) {
            tau_values[i] = tau_start + i * tau_step;
        }

#ifdef HDF5_ENABLED
        // ===== STREAMING APPROACH: Write to HDF5 as we receive data =====
        // This avoids storing all trajectories in memory at once
        
        // Type alias for trajectory
        typedef vector<pair<double, array<SpinVector, 3>>> TrajectoryType;
        
        // Rank 0 opens HDF5 file and prepares for streaming writes
        H5::H5File* file_ptr = nullptr;
        H5::Group metadata_group, reference_group, tau_scan_group;
        
        if (rank == 0) {
            string hdf5_file = dir_name + "/pump_probe_spectroscopy.h5";
            cout << "\nWriting data to HDF5 file (streaming): " << hdf5_file << endl;
            
            try {
                file_ptr = new H5::H5File(hdf5_file, H5F_ACC_TRUNC);
                
                // Create groups
                metadata_group = file_ptr->createGroup("/metadata");
                reference_group = file_ptr->createGroup("/reference");
                tau_scan_group = file_ptr->createGroup("/tau_scan");
                
                // Write metadata (simplified - essential params only)
                H5::DataSpace attr_space(H5S_SCALAR);
                {
                    auto write_attr = [&](const char* name, double val) {
                        H5::Attribute attr = metadata_group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, attr_space);
                        attr.write(H5::PredType::NATIVE_DOUBLE, &val);
                    };
                    auto write_attr_int = [&](const char* name, size_t val) {
                        H5::Attribute attr = metadata_group.createAttribute(name, H5::PredType::NATIVE_HSIZE, attr_space);
                        attr.write(H5::PredType::NATIVE_HSIZE, &val);
                    };
                    
                    write_attr_int("lattice_size", lattice_size);
                    write_attr_int("spin_dim", spin_dim);
                    write_attr_int("N_atoms", N_atoms);
                    write_attr("pulse_amp", pulse_amp);
                    write_attr("pulse_width", pulse_width);
                    write_attr("pulse_freq", pulse_freq);
                    write_attr("T_start", T_start);
                    write_attr("T_end", T_end);
                    write_attr("T_step", T_step);
                    write_attr("tau_start", tau_start);
                    write_attr("tau_end", tau_end);
                    write_attr("tau_step", tau_step);
                    write_attr_int("tau_steps", static_cast<size_t>(tau_steps));
                    write_attr("ground_state_energy", E_ground);
                }
                
                // Write tau values array
                hsize_t tau_dims[1] = {static_cast<hsize_t>(tau_steps)};
                H5::DataSpace tau_space(1, tau_dims);
                H5::DataSet tau_dataset = tau_scan_group.createDataSet("tau_values", H5::PredType::NATIVE_DOUBLE, tau_space);
                tau_dataset.write(tau_values.data(), H5::PredType::NATIVE_DOUBLE);
                
                // Write reference trajectory M0
                hsize_t time_dims[1] = {time_points};
                H5::DataSpace time_space(1, time_dims);
                
                vector<double> times(time_points);
                for (size_t i = 0; i < time_points; ++i) times[i] = M0_trajectory[i].first;
                H5::DataSet time_ds = reference_group.createDataSet("times", H5::PredType::NATIVE_DOUBLE, time_space);
                time_ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);
                
                // Write M0 magnetization data
                auto write_mag_dataset = [&](H5::Group& grp, const char* name, int mag_idx) {
                    hsize_t dims[2] = {time_points, spin_dim};
                    H5::DataSpace dspace(2, dims);
                    vector<double> data(time_points * spin_dim);
                    for (size_t t = 0; t < time_points; ++t) {
                        for (size_t d = 0; d < spin_dim; ++d) {
                            data[t * spin_dim + d] = M0_trajectory[t].second[mag_idx](d);
                        }
                    }
                    H5::DataSet ds = grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dspace);
                    ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
                };
                
                write_mag_dataset(reference_group, "M_antiferro", 0);
                write_mag_dataset(reference_group, "M_local", 1);
                write_mag_dataset(reference_group, "M_global", 2);
                
                cout << "  Reference trajectory (M0) written." << endl;
                
            } catch (H5::Exception& e) {
                std::cerr << "HDF5 Error opening file: " << e.getDetailMsg() << endl;
                if (file_ptr) delete file_ptr;
                file_ptr = nullptr;
            }
        }
        
        // Helper lambda to write a trajectory to HDF5 (rank 0 only)
        auto write_tau_to_hdf5 = [&](int tau_idx, const TrajectoryType& M1_traj, const TrajectoryType& M01_traj) {
            if (!file_ptr) return;
            
            std::string grp_name = "/tau_scan/tau_" + std::to_string(tau_idx);
            H5::Group tau_grp = file_ptr->createGroup(grp_name);
            
            // Write tau value as attribute
            H5::DataSpace attr_space(H5S_SCALAR);
            double tau_val = tau_values[tau_idx];
            H5::Attribute tau_attr = tau_grp.createAttribute("tau_value", H5::PredType::NATIVE_DOUBLE, attr_space);
            tau_attr.write(H5::PredType::NATIVE_DOUBLE, &tau_val);
            
            size_t n_times = M1_traj.size();
            
            auto write_mag = [&](const char* name, const TrajectoryType& traj, int mag_idx) {
                hsize_t dims[2] = {n_times, spin_dim};
                H5::DataSpace dspace(2, dims);
                vector<double> data(n_times * spin_dim);
                for (size_t t = 0; t < n_times; ++t) {
                    for (size_t d = 0; d < spin_dim; ++d) {
                        data[t * spin_dim + d] = traj[t].second[mag_idx](d);
                    }
                }
                H5::DataSet ds = tau_grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dspace);
                ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            };
            
            write_mag("M1_antiferro", M1_traj, 0);
            write_mag("M1_local", M1_traj, 1);
            write_mag("M1_global", M1_traj, 2);
            
            write_mag("M01_antiferro", M01_traj, 0);
            write_mag("M01_local", M01_traj, 1);
            write_mag("M01_global", M01_traj, 2);
            
            tau_grp.close();
        };
        
        // Helper lambda to deserialize buffer to trajectory
        auto deserialize_trajectory = [&](const vector<double>& buffer) -> TrajectoryType {
            TrajectoryType traj(time_points);
            for (size_t t = 0; t < time_points; ++t) {
                size_t offset = t * data_per_point;
                traj[t].first = buffer[offset];
                for (int m = 0; m < 3; ++m) {
                    traj[t].second[m] = SpinVector::Zero(spin_dim);
                    for (size_t d = 0; d < spin_dim; ++d) {
                        traj[t].second[m](d) = buffer[offset + 1 + m * spin_dim + d];
                    }
                }
            }
            return traj;
        };
        
        // First: rank 0 writes its own local results immediately
        if (rank == 0) {
            for (size_t idx = 0; idx < my_tau_indices.size(); ++idx) {
                int tau_idx = my_tau_indices[idx];
                write_tau_to_hdf5(tau_idx, local_M1_trajectories[idx], local_M01_trajectories[idx]);
            }
            cout << "  Rank 0 local trajectories written (" << my_tau_indices.size() << " tau points)." << endl;
            
            // Free local memory on rank 0 after writing
            local_M1_trajectories.clear();
            local_M1_trajectories.shrink_to_fit();
            local_M01_trajectories.clear();
            local_M01_trajectories.shrink_to_fit();
        }
        
        // Now receive from other ranks and write immediately (streaming)
        vector<double> M1_buffer(traj_size);
        vector<double> M01_buffer(traj_size);
        
        int progress_interval = std::max(1, tau_steps / 20);  // Report every 5%
        int received_count = 0;
        
        for (int tau_idx = 0; tau_idx < tau_steps; ++tau_idx) {
            int owner_rank = scheduler_writer_only ? (1 + (tau_idx % worker_count)) : 0;
            
            if (owner_rank == 0) continue;  // Already written above
            
            if (rank == 0) {
                // Receive from owner and write immediately
                MPI_Recv(M1_buffer.data(), traj_size, MPI_DOUBLE, owner_rank, 
                        2 * tau_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(M01_buffer.data(), traj_size, MPI_DOUBLE, owner_rank, 
                        2 * tau_idx + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Deserialize and write to HDF5 immediately (no storage)
                TrajectoryType M1_traj = deserialize_trajectory(M1_buffer);
                TrajectoryType M01_traj = deserialize_trajectory(M01_buffer);
                
                write_tau_to_hdf5(tau_idx, M1_traj, M01_traj);
                
                received_count++;
                if (received_count % progress_interval == 0) {
                    int local_tau_count = static_cast<int>(my_tau_indices.size());
                    cout << "  Progress: " << received_count << "/" << (tau_steps - local_tau_count) 
                         << " remote tau points received and written." << endl;
                }
                
            } else if (rank == owner_rank) {
                // Find local index for this tau
                size_t local_idx = 0;
                for (size_t i = 0; i < my_tau_indices.size(); ++i) {
                    if (my_tau_indices[i] == tau_idx) {
                        local_idx = i;
                        break;
                    }
                }
                
                // Serialize M1
                for (size_t t = 0; t < time_points; ++t) {
                    size_t offset = t * data_per_point;
                    M1_buffer[offset] = local_M1_trajectories[local_idx][t].first;
                    for (int m = 0; m < 3; ++m) {
                        for (size_t d = 0; d < spin_dim; ++d) {
                            M1_buffer[offset + 1 + m * spin_dim + d] = local_M1_trajectories[local_idx][t].second[m](d);
                        }
                    }
                }
                
                // Serialize M01
                for (size_t t = 0; t < time_points; ++t) {
                    size_t offset = t * data_per_point;
                    M01_buffer[offset] = local_M01_trajectories[local_idx][t].first;
                    for (int m = 0; m < 3; ++m) {
                        for (size_t d = 0; d < spin_dim; ++d) {
                            M01_buffer[offset + 1 + m * spin_dim + d] = local_M01_trajectories[local_idx][t].second[m](d);
                        }
                    }
                }
                
                MPI_Send(M1_buffer.data(), traj_size, MPI_DOUBLE, 0, 2 * tau_idx, MPI_COMM_WORLD);
                MPI_Send(M01_buffer.data(), traj_size, MPI_DOUBLE, 0, 2 * tau_idx + 1, MPI_COMM_WORLD);
            }
        }
        
        // Close HDF5 file
        if (rank == 0 && file_ptr) {
            metadata_group.close();
            reference_group.close();
            tau_scan_group.close();
            file_ptr->close();
            delete file_ptr;
            cout << "Successfully wrote all data to HDF5 file (streaming mode)" << endl;
        }
        
#else
        // No HDF5 - skip the communication and output
        if (rank == 0) {
            cout << "Warning: HDF5 support not enabled. Data not saved to HDF5 file." << endl;
        }
#endif
        
        if (rank == 0) {
            cout << "\n==========================================" << endl;
            cout << "Pump-Probe Spectroscopy (MPI) Complete!" << endl;
            cout << "Output directory: " << dir_name << endl;
            cout << "Total delay points: " << tau_steps << endl;
            cout << "==========================================" << endl;
        }
        
        // Restore ground state at end
        spins = ground_state;
        
        MPI_Barrier(MPI_COMM_WORLD);
    }

