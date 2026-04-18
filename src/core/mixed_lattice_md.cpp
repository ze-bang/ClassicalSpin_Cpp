/**
 * @file mixed_lattice_md.cpp
 * @brief MixedLattice molecular dynamics + pulse / pump-probe drivers.
 *
 * Hosts the Landau–Lifshitz / Gell-Mann RHS for the SU(2)+SU(3) system,
 * the ODE system glue, pulse field logic, and the full pump-probe
 * spectroscopy drivers (single-process and MPI-parallel).
 *
 * Kept in the header (NOT moved here):
 *   - `integrate_ode_system<System, Observer>` — templated, must be
 *     visible at instantiation points.
 *   - All CUDA-guarded `*_gpu` stubs — they live in paired
 *     `#ifdef CUDA_ENABLED` / `#else` blocks that are awkward to split.
 *   - Small inline helpers (`set_damping_SU3`, `set_equilibrium_SU3`,
 *     `reset_pulse`, …) so hot setup paths stay inlineable.
 */

#include "classical_spin/lattice/mixed_lattice.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include <boost/numeric/odeint.hpp>

// ---- MixedLattice::set_pulse_SU2 ----
    void MixedLattice::set_pulse_SU2(const vector<SpinVector>& field_in1, double t_B1,
                      const vector<SpinVector>& field_in2, double t_B2,
                      double amp, double width, double freq) {
        // Pack field vectors, transforming to local frame: B_local = R * B_global
        field_drive_SU2[0] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        field_drive_SU2[1] = SpinVector::Zero(N_atoms_SU2 * spin_dim_SU2);
        
        int mpi_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        
        for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
            // B_local = F^T * B_global (field transforms covariantly: B^(i) = F_i^T B^(0))
            // For symmetric frames (e.g. diagonal sign matrices), F^T = F
            SpinVector local_field1 = sublattice_frames_SU2[atom].transpose() * field_in1[atom];
            SpinVector local_field2 = sublattice_frames_SU2[atom].transpose() * field_in2[atom];
            field_drive_SU2[0].segment(atom * spin_dim_SU2, spin_dim_SU2) = local_field1;
            field_drive_SU2[1].segment(atom * spin_dim_SU2, spin_dim_SU2) = local_field2;
        }
        
        if (mpi_rank == 0) {
            cout << "\n========== SU(2) Pulse Configuration ==========" << endl;
            cout << "Pulse 1: t_center = " << t_B1 << ", Pulse 2: t_center = " << t_B2 << endl;
            cout << "Amplitude = " << amp << ", Width = " << width << ", Frequency = " << freq << endl;
            cout << "------------------------------------------------" << endl;
            
            for (size_t atom = 0; atom < N_atoms_SU2; ++atom) {
                SpinVector local_field1 = field_drive_SU2[0].segment(atom * spin_dim_SU2, spin_dim_SU2);
                SpinVector local_field2 = field_drive_SU2[1].segment(atom * spin_dim_SU2, spin_dim_SU2);
                
                cout << "SU(2) Atom " << atom << ":" << endl;
                cout << "  Pulse 1 - Global: [";
                for (size_t d = 0; d < spin_dim_SU2; ++d) {
                    cout << std::setw(10) << std::setprecision(6) << field_in1[atom](d);
                    if (d < spin_dim_SU2 - 1) cout << ", ";
                }
                cout << "]" << endl;
                cout << "           Local:  [";
                for (size_t d = 0; d < spin_dim_SU2; ++d) {
                    cout << std::setw(10) << std::setprecision(6) << local_field1(d);
                    if (d < spin_dim_SU2 - 1) cout << ", ";
                }
                cout << "]" << endl;
                cout << "  Pulse 2 - Global: [";
                for (size_t d = 0; d < spin_dim_SU2; ++d) {
                    cout << std::setw(10) << std::setprecision(6) << field_in2[atom](d);
                    if (d < spin_dim_SU2 - 1) cout << ", ";
                }
                cout << "]" << endl;
                cout << "           Local:  [";
                for (size_t d = 0; d < spin_dim_SU2; ++d) {
                    cout << std::setw(10) << std::setprecision(6) << local_field2(d);
                    if (d < spin_dim_SU2 - 1) cout << ", ";
                }
                cout << "]" << endl;
            }
            cout << "================================================\n" << endl;
        }
        
        t_pulse_SU2[0] = t_B1;
        t_pulse_SU2[1] = t_B2;
        field_drive_amp_SU2 = amp;
        field_drive_width_SU2 = width;
        field_drive_freq_SU2 = freq;
    }

// ---- MixedLattice::set_pulse_SU3 ----
    void MixedLattice::set_pulse_SU3(const vector<SpinVector>& field_in1, double t_B1,
                      const vector<SpinVector>& field_in2, double t_B2,
                      double amp, double width, double freq) {
        // Pack field vectors, transforming to local frame: B_local = R * B_global
        field_drive_SU3[0] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        field_drive_SU3[1] = SpinVector::Zero(N_atoms_SU3 * spin_dim_SU3);
        
        int mpi_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        
        for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
            // B_local = F^T * B_global (field transforms covariantly: B^(i) = F_i^T B^(0))
            // Derivation: B^(i) = χ^T D_i h = (χ^{-1} D_i χ)^T χ^T h = F_i^T B^(0)
            SpinVector local_field1 = sublattice_frames_SU3[atom].transpose() * field_in1[atom];
            SpinVector local_field2 = sublattice_frames_SU3[atom].transpose() * field_in2[atom];
            field_drive_SU3[0].segment(atom * spin_dim_SU3, spin_dim_SU3) = local_field1;
            field_drive_SU3[1].segment(atom * spin_dim_SU3, spin_dim_SU3) = local_field2;
        }
        
        if (mpi_rank == 0) {
            cout << "\n========== SU(3) Pulse Configuration ==========" << endl;
            cout << "Pulse 1: t_center = " << t_B1 << ", Pulse 2: t_center = " << t_B2 << endl;
            cout << "Amplitude = " << amp << ", Width = " << width << ", Frequency = " << freq << endl;
            cout << "------------------------------------------------" << endl;
            
            for (size_t atom = 0; atom < N_atoms_SU3; ++atom) {
                SpinVector local_field1 = field_drive_SU3[0].segment(atom * spin_dim_SU3, spin_dim_SU3);
                SpinVector local_field2 = field_drive_SU3[1].segment(atom * spin_dim_SU3, spin_dim_SU3);
                
                cout << "SU(3) Atom " << atom << ":" << endl;
                cout << "  Pulse 1 - Global: [";
                for (size_t d = 0; d < spin_dim_SU3; ++d) {
                    cout << std::setw(10) << std::setprecision(6) << field_in1[atom](d);
                    if (d < spin_dim_SU3 - 1) cout << ", ";
                }
                cout << "]" << endl;
                cout << "           Local:  [";
                for (size_t d = 0; d < spin_dim_SU3; ++d) {
                    cout << std::setw(10) << std::setprecision(6) << local_field1(d);
                    if (d < spin_dim_SU3 - 1) cout << ", ";
                }
                cout << "]" << endl;
                cout << "  Pulse 2 - Global: [";
                for (size_t d = 0; d < spin_dim_SU3; ++d) {
                    cout << std::setw(10) << std::setprecision(6) << field_in2[atom](d);
                    if (d < spin_dim_SU3 - 1) cout << ", ";
                }
                cout << "]" << endl;
                cout << "           Local:  [";
                for (size_t d = 0; d < spin_dim_SU3; ++d) {
                    cout << std::setw(10) << std::setprecision(6) << local_field2(d);
                    if (d < spin_dim_SU3 - 1) cout << ", ";
                }
                cout << "]" << endl;
            }
            cout << "================================================\n" << endl;
        }
        
        t_pulse_SU3[0] = t_B1;
        t_pulse_SU3[1] = t_B2;
        field_drive_amp_SU3 = amp;
        field_drive_width_SU3 = width;
        field_drive_freq_SU3 = freq;
    }

// ---- MixedLattice::drive_field_SU2_at_time ----
    SpinVector MixedLattice::drive_field_SU2_at_time(double t, size_t site_index) const {
        const size_t atom = site_index % N_atoms_SU2;
        SpinVector result = SpinVector::Zero(spin_dim_SU2);
        
        // Two Gaussian pulses
        for (int pulse = 0; pulse < 2; ++pulse) {
            double dt = t - t_pulse_SU2[pulse];
            double envelope = exp(-pow(dt / (2 * field_drive_width_SU2), 2));
            double oscillation = cos(field_drive_freq_SU2 * dt);
            double factor = field_drive_amp_SU2 * envelope * oscillation;
            
            result += factor * field_drive_SU2[pulse].segment(atom * spin_dim_SU2, spin_dim_SU2);
        }
        
        return result;
    }

// ---- MixedLattice::drive_field_SU3_at_time ----
    SpinVector MixedLattice::drive_field_SU3_at_time(double t, size_t site_index) const {
        const size_t atom = site_index % N_atoms_SU3;
        SpinVector result = SpinVector::Zero(spin_dim_SU3);
        
        // Two Gaussian pulses
        for (int pulse = 0; pulse < 2; ++pulse) {
            double dt = t - t_pulse_SU3[pulse];
            double envelope = exp(-pow(dt / (2 * field_drive_width_SU3), 2));
            double oscillation = cos(field_drive_freq_SU3 * dt);
            double factor = field_drive_amp_SU3 * envelope * oscillation;
            
            result += factor * field_drive_SU3[pulse].segment(atom * spin_dim_SU3, spin_dim_SU3);
        }
        
        return result;
    }

// ---- MixedLattice::ode_system ----
    void MixedLattice::ode_system(const ODEState& x, ODEState& dxdt, double t) {
        landau_lifshitz(x, dxdt, t);
    }

// ---- MixedLattice::landau_lifshitz ----
    void MixedLattice::landau_lifshitz(const ODEState& state, ODEState& dsdt, double t) {
        const size_t offset_SU3 = lattice_size_SU2 * spin_dim_SU2;  // Start index for SU(3) spins
        
        // SU(2) dynamics: dS/dt = H × S (cross product for spin-1/2)
        for (size_t site = 0; site < lattice_size_SU2; ++site) {
            const size_t idx = site * spin_dim_SU2;
            
            // Compute local field using helper function
            SpinVector H = get_local_field_SU2_flat(site, state, offset_SU3, t);
            
            // Compute dS/dt = H × S for SU(2), plus LLG Gilbert damping
            if (spin_dim_SU2 == 3) {
                // Standard cross product for 3D vectors
                dsdt[idx + 0] = H(1) * state[idx + 2] - H(2) * state[idx + 1];
                dsdt[idx + 1] = H(2) * state[idx + 0] - H(0) * state[idx + 2];
                dsdt[idx + 2] = H(0) * state[idx + 1] - H(1) * state[idx + 0];
                // Gilbert damping: dS/dt += (alpha/|S|) * S × (S × H)
                // S × (S × H) = S(S·H) - H|S|^2  (BAC-CAB identity)
                if (alpha_gilbert != 0.0) {
                    const double Sx = state[idx + 0], Sy = state[idx + 1], Sz = state[idx + 2];
                    const double S2 = Sx*Sx + Sy*Sy + Sz*Sz;
                    const double SdotH = Sx*H(0) + Sy*H(1) + Sz*H(2);
                    const double inv_S = (S2 > 0.0) ? alpha_gilbert / std::sqrt(S2) : 0.0;
                    dsdt[idx + 0] += inv_S * (Sx * SdotH - H(0) * S2);
                    dsdt[idx + 1] += inv_S * (Sy * SdotH - H(1) * S2);
                    dsdt[idx + 2] += inv_S * (Sz * SdotH - H(2) * S2);
                }
            } else {
                // General case: would need proper SU(N) structure constants
                for (size_t j = 0; j < spin_dim_SU2; ++j) {
                    dsdt[idx + j] = 0.0;
                }
            }
        }
        
        // SU(3) dynamics: dS/dt = f_ijk H_j S_k − Γ_i (S_i − S_i^eq)
        const auto& f = get_SU3_structure();
        for (size_t site = 0; site < lattice_size_SU3; ++site) {
            const size_t idx = offset_SU3 + site * spin_dim_SU3;
            
            // Compute local field using helper function
            SpinVector H = get_local_field_SU3_flat(site, state, offset_SU3, t);
            
            // Compute dS/dt = f_ijk H_j S_k for SU(3) using Gell-Mann structure constants
            for (size_t i = 0; i < spin_dim_SU3; ++i) {
                double dSdt_i = 0.0;
                for (size_t j = 0; j < spin_dim_SU3; ++j) {
                    for (size_t k = 0; k < spin_dim_SU3; ++k) {
                        dSdt_i += f[i](j, k) * H(j) * state[idx + k];
                    }
                }
                // Bloch damping: −Γ_i (n^i − n^i_eq)
                if (damping_rates_SU3(i) != 0.0) {
                    dSdt_i -= damping_rates_SU3(i) * (state[idx + i] - equilibrium_SU3[site](i));
                }
                dsdt[idx + i] = dSdt_i;
            }
        }
    }

// ---- MixedLattice::get_local_field_SU2_flat ----
    SpinVector MixedLattice::get_local_field_SU2_flat(size_t site, const ODEState& state, 
                                         size_t offset_SU3, double t) const {
        const size_t idx = site * spin_dim_SU2;
        SpinVector H = -field_SU2[site];
        
        // Onsite: 2*A*S
        for (size_t a = 0; a < spin_dim_SU2; ++a) {
            for (size_t b = 0; b < spin_dim_SU2; ++b) {
                H(a) += 2.0 * onsite_interaction_SU2[site](a, b) * state[idx + b];
            }
        }
        
        // Bilinear SU(2)-SU(2): J*S_partner
        for (size_t n = 0; n < bilinear_partners_SU2[site].size(); ++n) {
            const size_t partner = bilinear_partners_SU2[site][n];
            const size_t partner_idx = partner * spin_dim_SU2;
            const auto& J = bilinear_interaction_SU2[site][n];
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    H(a) += J(a, b) * state[partner_idx + b];
                }
            }
        }
        
        // Mixed bilinear SU(2)-SU(3)
        for (size_t n = 0; n < mixed_bilinear_partners_SU2[site].size(); ++n) {
            const size_t partner = mixed_bilinear_partners_SU2[site][n];
            const size_t partner_idx = offset_SU3 + partner * spin_dim_SU3;
            const auto& J = mixed_bilinear_interaction_SU2[site][n];
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                for (size_t c = 0; c < spin_dim_SU3; ++c) {
                    H(a) += J(a, c) * state[partner_idx + c];
                }
            }
        }
        
        // Trilinear SU(2)-SU(2)-SU(2): T[a](b,c) * S1[b] * S2[c]
        for (size_t n = 0; n < trilinear_partners_SU2[site].size(); ++n) {
            const size_t p1 = trilinear_partners_SU2[site][n][0];
            const size_t p2 = trilinear_partners_SU2[site][n][1];
            const size_t p1_idx = p1 * spin_dim_SU2;
            const size_t p2_idx = p2 * spin_dim_SU2;
            const auto& T = trilinear_interaction_SU2[site][n];
            
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * state[p1_idx + b] * state[p2_idx + c];
                    }
                }
                H(a) += temp;
            }
        }
        
        // Mixed trilinear SU(2)-SU(2)-SU(3): T[a](b,c) * S_SU2[b] * S_SU3[c]
        for (size_t n = 0; n < mixed_trilinear_partners_SU2[site].size(); ++n) {
            const size_t p1 = mixed_trilinear_partners_SU2[site][n][0];
            const size_t p2 = mixed_trilinear_partners_SU2[site][n][1];
            const size_t p1_idx = p1 * spin_dim_SU2;
            const size_t p2_idx = offset_SU3 + p2 * spin_dim_SU3;
            const auto& T = mixed_trilinear_interaction_SU2[site][n];
            
            for (size_t a = 0; a < spin_dim_SU2; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * state[p1_idx + b] * state[p2_idx + c];
                    }
                }
                H(a) += temp;
            }
        }
        
        // Add time-dependent drive
        H -= drive_field_SU2_at_time(t, site);
        
        return H;
    }

// ---- MixedLattice::get_local_field_SU3_flat ----
    SpinVector MixedLattice::get_local_field_SU3_flat(size_t site, const ODEState& state, 
                                         size_t offset_SU3, double t) const {
        const size_t idx = offset_SU3 + site * spin_dim_SU3;
        SpinVector H = -field_SU3[site];
        
        // Onsite: 2*A*S
        for (size_t a = 0; a < spin_dim_SU3; ++a) {
            for (size_t b = 0; b < spin_dim_SU3; ++b) {
                H(a) += 2.0 * onsite_interaction_SU3[site](a, b) * state[idx + b];
            }
        }
        
        // Bilinear SU(3)-SU(3): J*S_partner
        for (size_t n = 0; n < bilinear_partners_SU3[site].size(); ++n) {
            const size_t partner = bilinear_partners_SU3[site][n];
            const size_t partner_idx = offset_SU3 + partner * spin_dim_SU3;
            const auto& J = bilinear_interaction_SU3[site][n];
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                for (size_t b = 0; b < spin_dim_SU3; ++b) {
                    H(a) += J(a, b) * state[partner_idx + b];
                }
            }
        }
        
        // Mixed bilinear SU(3)-SU(2)
        for (size_t n = 0; n < mixed_bilinear_partners_SU3[site].size(); ++n) {
            const size_t partner = mixed_bilinear_partners_SU3[site][n];
            const size_t partner_idx = partner * spin_dim_SU2;
            const auto& J = mixed_bilinear_interaction_SU3[site][n];
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    H(a) += J(a, b) * state[partner_idx + b];
                }
            }
        }
        
        // Trilinear SU(3)-SU(3)-SU(3): T[a](b,c) * S1[b] * S2[c]
        for (size_t n = 0; n < trilinear_partners_SU3[site].size(); ++n) {
            const size_t p1 = trilinear_partners_SU3[site][n][0];
            const size_t p2 = trilinear_partners_SU3[site][n][1];
            const size_t p1_idx = offset_SU3 + p1 * spin_dim_SU3;
            const size_t p2_idx = offset_SU3 + p2 * spin_dim_SU3;
            const auto& T = trilinear_interaction_SU3[site][n];
            
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU3; ++b) {
                    for (size_t c = 0; c < spin_dim_SU3; ++c) {
                        temp += T[a](b, c) * state[p1_idx + b] * state[p2_idx + c];
                    }
                }
                H(a) += temp;
            }
        }
        
        // Mixed trilinear SU(3)-SU(2)-SU(2): T[a](b,c) * S_SU2_1[b] * S_SU2_2[c]
        for (size_t n = 0; n < mixed_trilinear_partners_SU3[site].size(); ++n) {
            const size_t p1 = mixed_trilinear_partners_SU3[site][n][0];
            const size_t p2 = mixed_trilinear_partners_SU3[site][n][1];
            const size_t p1_idx = p1 * spin_dim_SU2;
            const size_t p2_idx = p2 * spin_dim_SU2;
            const auto& T = mixed_trilinear_interaction_SU3[site][n];
            
            for (size_t a = 0; a < spin_dim_SU3; ++a) {
                double temp = 0.0;
                for (size_t b = 0; b < spin_dim_SU2; ++b) {
                    for (size_t c = 0; c < spin_dim_SU2; ++c) {
                        temp += T[a](b, c) * state[p1_idx + b] * state[p2_idx + c];
                    }
                }
                H(a) += temp;
            }
        }
        
        // Add time-dependent drive
        H -= drive_field_SU3_at_time(t, site);
        
        return H;
    }

// ---- MixedLattice::molecular_dynamics ----
    void MixedLattice::molecular_dynamics(double T_start, double T_end, double dt_initial,
                           const string& out_dir, size_t save_interval,
                           const string& method, bool use_gpu) {
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

// ---- MixedLattice::molecular_dynamics_cpu ----
    void MixedLattice::molecular_dynamics_cpu(double T_start, double T_end, double dt_initial,
                           const string& out_dir, size_t save_interval,
                           const string& method) {
#ifndef HDF5_ENABLED
        std::cerr << "Error: HDF5 support is required for molecular dynamics output." << endl;
        std::cerr << "Please rebuild with -DHDF5_ENABLED flag and HDF5 libraries." << endl;
        return;
#else
        ensure_directory_exists(out_dir);
        
        cout << "Running mixed lattice molecular dynamics with Boost.Odeint: t=" << T_start << " → " << T_end << endl;
        cout << "Integration method: " << method << endl;
        cout << "Initial step size: " << dt_initial << endl;
        
        // Convert current spins to flat state vector
        ODEState state = spins_to_state();
        
        // Create HDF5 writer with comprehensive metadata
        std::unique_ptr<HDF5MixedMDWriter> hdf5_writer;
        if (!out_dir.empty()) {
            string hdf5_file = out_dir + "/trajectory.h5";
            cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
            hdf5_writer = std::make_unique<HDF5MixedMDWriter>(
                hdf5_file, 
                lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
                lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
                dim1, dim2, dim3, method, 
                dt_initial, T_start, T_end, save_interval, 
                spin_length_SU2, spin_length_SU3,
                &site_positions_SU2, &site_positions_SU3, 10000);
        }
        
        // Observer to save data at specified intervals
        size_t step_count = 0;
        size_t save_count = 0;
        auto observer = [&](const ODEState& x, double t) {
            if (step_count % save_interval == 0) {
                // Compute magnetizations directly from flat state (zero allocation)
                SpinVector M_SU2 = SpinVector::Zero(spin_dim_SU2);
                SpinVector M_SU3 = SpinVector::Zero(spin_dim_SU3);
                SpinVector M_SU2_antiferro = SpinVector::Zero(spin_dim_SU2);
                SpinVector M_SU3_antiferro = SpinVector::Zero(spin_dim_SU3);
                SpinVector M_SU2_global = SpinVector::Zero(spin_dim_SU2);
                SpinVector M_SU3_global = SpinVector::Zero(spin_dim_SU3);
                
                double M_SU2_arr[8] = {0};
                double M_SU2_antiferro_arr[8] = {0};
                double M_SU2_global_arr[8] = {0};
                compute_sublattice_magnetizations_from_flat(x.data(), 0, 
                    lattice_size_SU2, spin_dim_SU2, M_SU2_arr, M_SU2_antiferro_arr);
                compute_magnetization_global_SU2_from_flat(x.data(), M_SU2_global_arr);
                compute_magnetization_staggered_SU2_from_flat(x.data(), M_SU2_antiferro_arr);
                M_SU2 = Eigen::Map<Eigen::VectorXd>(M_SU2_arr, spin_dim_SU2) / double(lattice_size_SU2);
                M_SU2_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU2_antiferro_arr, spin_dim_SU2);
                M_SU2_global = Eigen::Map<Eigen::VectorXd>(M_SU2_global_arr, spin_dim_SU2);
                
                double M_SU3_arr[8] = {0};
                double M_SU3_antiferro_arr[8] = {0};
                double M_SU3_global_arr[8] = {0};
                size_t SU3_offset = lattice_size_SU2 * spin_dim_SU2;
                compute_sublattice_magnetizations_from_flat(x.data(), SU3_offset, 
                    lattice_size_SU3, spin_dim_SU3, M_SU3_arr, M_SU3_antiferro_arr);
                compute_magnetization_global_SU3_from_flat(x.data(), M_SU3_global_arr);
                M_SU3 = Eigen::Map<Eigen::VectorXd>(M_SU3_arr, spin_dim_SU3) / double(lattice_size_SU3);
                M_SU3_antiferro = Eigen::Map<Eigen::VectorXd>(M_SU3_antiferro_arr, spin_dim_SU3) / double(lattice_size_SU3);
                M_SU3_global = Eigen::Map<Eigen::VectorXd>(M_SU3_global_arr, spin_dim_SU3);
                
                // Compute accurate energy density directly from flat state (includes all interactions)
                double E = total_energy_flat(x.data()) / (lattice_size_SU2 + lattice_size_SU3);
                
                // Write to HDF5 directly from flat state (no conversion needed)
                if (hdf5_writer) {
                    hdf5_writer->write_flat_step(t, 
                                                M_SU2_antiferro, M_SU2, M_SU2_global,
                                                M_SU3_antiferro, M_SU3, M_SU3_global,
                                                x.data());
                    save_count++;
                }
                
                // Progress output
                if (step_count % (save_interval * 10) == 0) {
                    cout << "t=" << t << ", E/N=" << E 
                         << ", |M_SU2|=" << M_SU2.norm() 
                         << ", |M_SU3|=" << M_SU3.norm() << endl;
                }
            }
            ++step_count;
        };
        
        // Create ODE system wrapper for Boost.Odeint
        auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
            this->ode_system(x, dxdt, t);
        };
        
        // Integrate using selected method
        auto [abs_tol, rel_tol] = get_integration_tolerances(method);
        integrate_ode_system(system_func, state, T_start, T_end, dt_initial,
                            observer, method, true, abs_tol, rel_tol);
        
        // Note: MixedLattice::spins_SU2 and spins_SU3 remain unchanged (initial configuration preserved)
        // The evolved state is stored in the ODEState 'state' variable
        
        // Close HDF5 file
        if (hdf5_writer) {
            hdf5_writer->close();
            cout << "HDF5 trajectory saved with " << save_count << " snapshots" << endl;
        }
        
        cout << "Molecular dynamics complete! (" << step_count << " steps)" << endl;
#endif // HDF5_ENABLED
    }

// ---- MixedLattice::pump_probe_spectroscopy ----
    void MixedLattice::pump_probe_spectroscopy(const vector<SpinVector>& field_in_SU2,
                                 const vector<SpinVector>& field_in_SU3,
                                 double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                                 double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                                 double tau_start, double tau_end, double tau_step,
                                 double T_start, double T_end, double T_step,
                                 double Temp_start, double Temp_end,
                                 size_t n_anneal,
                                 bool T_zero_quench, size_t quench_sweeps,
                                 string dir_name,
                                 string method, bool use_gpu) {
        
        std::filesystem::create_directories(dir_name);
        
        cout << "\n==========================================" << endl;
        cout << "Mixed Lattice Pump-Probe Spectroscopy" << endl;
        cout << "==========================================" << endl;
        cout << "SU(2) Pulse: amp=" << pulse_amp_SU2 << ", width=" << pulse_width_SU2 
             << ", freq=" << pulse_freq_SU2 << endl;
        cout << "SU(3) Pulse: amp=" << pulse_amp_SU3 << ", width=" << pulse_width_SU3 
             << ", freq=" << pulse_freq_SU3 << endl;
        cout << "Delay scan: " << tau_start << " → " << tau_end << " (step: " << tau_step << ")" << endl;
        cout << "Integration: " << T_start << " → " << T_end << " (step: " << T_step << ")" << endl;
        
        // Use current spin configuration as ground state (assumed pre-loaded)
        cout << "\n[1/3] Using current configuration as ground state..." << endl;
        double E_ground = energy_density();
        double E_ground_SU2 = total_energy_SU2();
        double E_ground_SU3 = total_energy_SU3();
        SpinVector M_ground_SU2 = magnetization_SU2();
        SpinVector M_ground_SU3 = magnetization_SU3();
        cout << "  Ground state: E/N = " << E_ground << endl;
        cout << "    Total Energy:     " << total_energy() << endl;
        cout << "    SU2 Energy:       " << E_ground_SU2 << " (E/N_SU2 = " << E_ground_SU2 / lattice_size_SU2 << ")" << endl;
        cout << "    SU3 Energy:       " << E_ground_SU3 << " (E/N_SU3 = " << E_ground_SU3 / lattice_size_SU3 << ")" << endl;
        cout << "    |M_SU2| = " << M_ground_SU2.norm() << endl;
        cout << "    |M_SU3| = " << M_ground_SU3.norm() << endl;
        
        // Save initial configuration
        save_positions_to_dir(dir_name);
        save_spin_config_to_dir(dir_name, "initial_spins");
        
        // Backup ground state
        SpinConfigSU2 ground_state_SU2 = spins_SU2;
        SpinConfigSU3 ground_state_SU3 = spins_SU3;
        
        // Step 2: Reference single-pulse dynamics (pump at t=0)
        cout << "\n[2/3] Running reference single-pulse dynamics (M0)..." << endl;
        if (use_gpu) cout << "  Using GPU acceleration" << endl;
        auto M0_trajectory = single_pulse_drive(field_in_SU2, field_in_SU3, 0.0,
                                   pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                   pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                   T_start, T_end, T_step, method, use_gpu);
        
        // Step 3: Delay time scan
        int tau_steps = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
        cout << "\n[3/3] Scanning delay times (" << tau_steps << " steps)..." << endl;
        
        typedef vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>> TrajectoryType;
        
        // Open HDF5 writer BEFORE the loop to write incrementally (avoids OOM)
        string hdf5_file = dir_name + "/pump_probe_spectroscopy.h5";
        
#ifdef HDF5_ENABLED
        HDF5MixedPumpProbeWriter writer(
            hdf5_file,
            lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
            lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
            dim1, dim2, dim3,
            spin_length_SU2, spin_length_SU3,
            pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
            pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
            T_start, T_end, T_step, method,
            tau_start, tau_end, tau_step,
            E_ground, M_ground_SU2, M_ground_SU3,
            Temp_start, Temp_end, n_anneal,
            T_zero_quench, quench_sweeps,
            &field_in_SU2, &field_in_SU3,
            &site_positions_SU2, &site_positions_SU3
        );
        
        // Write reference trajectory
        writer.write_reference_trajectory(M0_trajectory);
#endif
        
        double current_tau = tau_start;
        for (int i = 0; i < tau_steps; ++i) {
            cout << "\n--- Delay " << (i+1) << "/" << tau_steps << ": tau = " << current_tau << " ---" << endl;
            
            // Restore ground state
            spins_SU2 = ground_state_SU2;
            spins_SU3 = ground_state_SU3;
            
            // M1: Probe pulse only at tau
            cout << "  Computing M1 (probe at tau)..." << endl;
            auto M1_traj = single_pulse_drive(field_in_SU2, field_in_SU3, current_tau,
                                pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                T_start, T_end, T_step, method, use_gpu);
            
            // Restore ground state
            spins_SU2 = ground_state_SU2;
            spins_SU3 = ground_state_SU3;
            
            // M01: Pump at 0 + Probe at tau
            cout << "  Computing M01 (pump + probe)..." << endl;
            auto M01_traj = double_pulse_drive(field_in_SU2, field_in_SU3, 0.0,
                                     field_in_SU2, field_in_SU3, current_tau,
                                     pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                     pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                     T_start, T_end, T_step, method, use_gpu);
            
            // Write this tau step immediately and release memory
#ifdef HDF5_ENABLED
            writer.write_tau_trajectory(i, current_tau, M1_traj, M01_traj);
#endif
            
            current_tau += tau_step;
        }
        
#ifdef HDF5_ENABLED
        writer.close();
        cout << "\n[Complete] All data written incrementally to: " << hdf5_file << endl;
#else
        cout << "Note: HDF5 support not enabled. Rebuild with -DHDF5_ENABLED flag." << endl;
#endif
        
        // Restore ground state at end
        spins_SU2 = ground_state_SU2;
        spins_SU3 = ground_state_SU3;
        
        cout << "\n==========================================" << endl;
        cout << "Pump-Probe Spectroscopy Complete!" << endl;
        cout << "Output directory: " << dir_name << endl;
        cout << "Total delay points: " << tau_steps << endl;
        cout << "==========================================" << endl;
    }

// ---- MixedLattice::pump_probe_spectroscopy_mpi ----
    void MixedLattice::pump_probe_spectroscopy_mpi(const vector<SpinVector>& field_in_SU2,
                                     const vector<SpinVector>& field_in_SU3,
                                     double pulse_amp_SU2, double pulse_width_SU2, double pulse_freq_SU2,
                                     double pulse_amp_SU3, double pulse_width_SU3, double pulse_freq_SU3,
                                     double tau_start, double tau_end, double tau_step,
                                     double T_start, double T_end, double T_step,
                                     double Temp_start, double Temp_end,
                                     size_t n_anneal,
                                     bool T_zero_quench, size_t quench_sweeps,
                                     string dir_name,
                                     string method, bool use_gpu,
                                     bool save_spin_trajectories) {
        
        int rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        std::filesystem::create_directories(dir_name);
        
        int tau_steps = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
        const bool scheduler_writer_only = (mpi_size > 1);
        const int worker_count = scheduler_writer_only ? (mpi_size - 1) : 1;
        const size_t state_dim = lattice_size_SU2 * spin_dim_SU2 + lattice_size_SU3 * spin_dim_SU3;
        
        if (rank == 0) {
            cout << "\n==========================================" << endl;
            cout << "Mixed Lattice Pump-Probe (MPI Parallel)" << endl;
            cout << "==========================================" << endl;
            cout << "MPI ranks: " << mpi_size << endl;
            cout << "SU(2) Pulse: amp=" << pulse_amp_SU2 << ", width=" << pulse_width_SU2 
                 << ", freq=" << pulse_freq_SU2 << endl;
            cout << "SU(3) Pulse: amp=" << pulse_amp_SU3 << ", width=" << pulse_width_SU3 
                 << ", freq=" << pulse_freq_SU3 << endl;
            cout << "Delay scan: " << tau_start << " → " << tau_end << " (step: " << tau_step << ")" << endl;
            cout << "Total delay points: " << tau_steps << endl;
            if (scheduler_writer_only) {
                cout << "Rank 0 role: scheduler/writer only" << endl;
                cout << "Tau points per worker rank: ~" << (tau_steps + worker_count - 1) / worker_count << endl;
            } else {
                cout << "Tau points per rank: ~" << (tau_steps + mpi_size - 1) / mpi_size << endl;
            }
            if (use_gpu) {
                cout << "GPU acceleration: ENABLED (each rank uses assigned GPU)" << endl;
            }
            if (save_spin_trajectories) {
                cout << "Spin trajectory saving: ENABLED (state_dim=" << state_dim << ")" << endl;
                cout << "  WARNING: This will significantly increase memory usage and file size." << endl;
            }
        }
        
        // Ground state info
        if (rank == 0) {
            cout << "\n[1/4] Using current configuration as ground state..." << endl;
        }
        double E_ground = energy_density();
        double E_ground_SU2 = total_energy_SU2();
        double E_ground_SU3 = total_energy_SU3();
        SpinVector M_ground_SU2 = magnetization_SU2();
        SpinVector M_ground_SU3 = magnetization_SU3();
        if (rank == 0) {
            cout << "  Ground state: E/N = " << E_ground << endl;
            cout << "    Total Energy:     " << total_energy() << endl;
            cout << "    SU2 Energy:       " << E_ground_SU2 << " (E/N_SU2 = " << E_ground_SU2 / lattice_size_SU2 << ")" << endl;
            cout << "    SU3 Energy:       " << E_ground_SU3 << " (E/N_SU3 = " << E_ground_SU3 / lattice_size_SU3 << ")" << endl;
            cout << "    |M_SU2| = " << M_ground_SU2.norm() << endl;
            cout << "    |M_SU3| = " << M_ground_SU3.norm() << endl;
        }
        
        // Save initial configuration (rank 0 only)
        if (rank == 0) {
            save_positions_to_dir(dir_name);
            save_spin_config_to_dir(dir_name, "initial_spins");
            save_energy_to_dir(dir_name, "energy_initial");
        }
        
        // Backup ground state
        SpinConfigSU2 ground_state_SU2 = spins_SU2;
        SpinConfigSU3 ground_state_SU3 = spins_SU3;
        
        // Reference trajectory M0
        if (rank == 0) {
            cout << "\n[2/4] Running reference single-pulse dynamics (M0)..." << endl;
        }
        
        typedef vector<pair<double, pair<array<SpinVector, 3>, array<SpinVector, 3>>>> TrajectoryType;
        
        TrajectoryType M0_trajectory;
        vector<double> M0_spin_flat;  // flat (n_t * state_dim) spin state for M0, rank 0 only
        if (rank == 0) {
            vector<vector<double>> M0_spin_states;
            M0_trajectory = single_pulse_drive(field_in_SU2, field_in_SU3, 0.0,
                                               pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                               pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                               T_start, T_end, T_step, method, use_gpu,
                                               save_spin_trajectories ? &M0_spin_states : nullptr);
            if (save_spin_trajectories && !M0_spin_states.empty()) {
                size_t n_t = M0_spin_states.size();
                M0_spin_flat.resize(n_t * state_dim);
                for (size_t t = 0; t < n_t; ++t) {
                    std::copy(M0_spin_states[t].begin(), M0_spin_states[t].end(),
                              M0_spin_flat.data() + t * state_dim);
                }
            }
            // Note: M0_spin_states[0] (if save_spin_traj) holds flat (n_t * state_dim) buffer
            // It is written to HDF5 after the file is opened below
        }
        
        // Restore ground state
        spins_SU2 = ground_state_SU2;
        spins_SU3 = ground_state_SU3;
        
        // Distribute tau values
        if (rank == 0) {
            cout << "\n[3/4] Distributing tau delays across " << mpi_size << " ranks..." << endl;
        }
        
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
        
        // Local trajectories
        vector<TrajectoryType> local_M1_trajectories;
        vector<TrajectoryType> local_M01_trajectories;
        // Local spin state flat buffers (n_t * state_dim each), only populated if save_spin_trajectories
        vector<vector<double>> local_spin_flat_M1;
        vector<vector<double>> local_spin_flat_M01;
        
        local_M1_trajectories.reserve(my_tau_indices.size());
        local_M01_trajectories.reserve(my_tau_indices.size());
        if (save_spin_trajectories) {
            local_spin_flat_M1.reserve(my_tau_indices.size());
            local_spin_flat_M01.reserve(my_tau_indices.size());
        }
        
        for (size_t idx = 0; idx < my_tau_indices.size(); ++idx) {
            double current_tau = my_tau_values[idx];
            int global_idx = my_tau_indices[idx];
            
            cout << "[Rank " << rank << "] Computing tau[" << global_idx << "] = " << current_tau 
                 << " (" << (idx+1) << "/" << my_tau_indices.size() << ")" << endl;
            
            // Restore ground state
            spins_SU2 = ground_state_SU2;
            spins_SU3 = ground_state_SU3;
            
            // M1: Probe at tau
            {
                vector<vector<double>> M1_spin_states;
                auto M1_traj = single_pulse_drive(field_in_SU2, field_in_SU3, current_tau,
                                    pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                    pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                    T_start, T_end, T_step, method, use_gpu,
                                    save_spin_trajectories ? &M1_spin_states : nullptr);
                local_M1_trajectories.push_back(M1_traj);
                if (save_spin_trajectories && !M1_spin_states.empty()) {
                    size_t n_t = M1_spin_states.size();
                    vector<double> flat(n_t * state_dim);
                    for (size_t t = 0; t < n_t; ++t)
                        std::copy(M1_spin_states[t].begin(), M1_spin_states[t].end(), flat.data() + t * state_dim);
                    local_spin_flat_M1.push_back(std::move(flat));
                } else if (save_spin_trajectories) {
                    local_spin_flat_M1.emplace_back();
                }
            }
            
            // Restore ground state
            spins_SU2 = ground_state_SU2;
            spins_SU3 = ground_state_SU3;
            
            // M01: Pump at 0 + Probe at tau
            {
                vector<vector<double>> M01_spin_states;
                auto M01_traj = double_pulse_drive(field_in_SU2, field_in_SU3, 0.0,
                                         field_in_SU2, field_in_SU3, current_tau,
                                         pulse_amp_SU2, pulse_width_SU2, pulse_freq_SU2,
                                         pulse_amp_SU3, pulse_width_SU3, pulse_freq_SU3,
                                         T_start, T_end, T_step, method, use_gpu,
                                         save_spin_trajectories ? &M01_spin_states : nullptr);
                local_M01_trajectories.push_back(M01_traj);
                if (save_spin_trajectories && !M01_spin_states.empty()) {
                    size_t n_t = M01_spin_states.size();
                    vector<double> flat(n_t * state_dim);
                    for (size_t t = 0; t < n_t; ++t)
                        std::copy(M01_spin_states[t].begin(), M01_spin_states[t].end(), flat.data() + t * state_dim);
                    local_spin_flat_M01.push_back(std::move(flat));
                } else if (save_spin_trajectories) {
                    local_spin_flat_M01.emplace_back();
                }
            }
        }
        
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
        // Data per point: time + 3 SU2 vectors + 3 SU3 vectors
        size_t data_per_point = 1 + 3 * spin_dim_SU2 + 3 * spin_dim_SU3;
        size_t traj_size = time_points * data_per_point;
        // Size of flat spin state per trajectory (when save_spin_trajectories)
        size_t state_traj_size_spins = time_points * state_dim;
        
        // Compute tau values (needed for HDF5)
        vector<double> tau_values(tau_steps);
        for (int i = 0; i < tau_steps; ++i) {
            tau_values[i] = tau_start + i * tau_step;
        }

#ifdef HDF5_ENABLED
        // ===== STREAMING APPROACH: Write to HDF5 as we receive data =====
        // This avoids storing all trajectories in memory at once
        
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
                    
                    write_attr_int("lattice_size_SU2", lattice_size_SU2);
                    write_attr_int("lattice_size_SU3", lattice_size_SU3);
                    write_attr_int("spin_dim_SU2", spin_dim_SU2);
                    write_attr_int("spin_dim_SU3", spin_dim_SU3);
                    write_attr_int("N_atoms_SU2", N_atoms_SU2);
                    write_attr_int("N_atoms_SU3", N_atoms_SU3);
                    write_attr("pulse_amp_SU2", pulse_amp_SU2);
                    write_attr("pulse_width_SU2", pulse_width_SU2);
                    write_attr("pulse_freq_SU2", pulse_freq_SU2);
                    write_attr("pulse_amp_SU3", pulse_amp_SU3);
                    write_attr("pulse_width_SU3", pulse_width_SU3);
                    write_attr("pulse_freq_SU3", pulse_freq_SU3);
                    write_attr("T_start", T_start);
                    write_attr("T_end", T_end);
                    write_attr("T_step", T_step);
                    write_attr("tau_start", tau_start);
                    write_attr("tau_end", tau_end);
                    write_attr("tau_step", tau_step);
                    write_attr_int("tau_steps", static_cast<size_t>(tau_steps));
                    write_attr("ground_state_energy", E_ground);
                    write_attr_int("save_spin_trajectories", save_spin_trajectories ? 1u : 0u);
                    write_attr_int("state_dim_SU2", lattice_size_SU2 * spin_dim_SU2);
                    write_attr_int("state_dim_SU3", lattice_size_SU3 * spin_dim_SU3);
                    write_attr_int("state_dim_total", state_dim);
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
                
                // Write M0 magnetization data (SU2 and SU3)
                auto write_mag_dataset = [&](H5::Group& grp, const char* name, size_t sdim, int mag_idx, bool is_su2) {
                    hsize_t dims[2] = {time_points, sdim};
                    H5::DataSpace dspace(2, dims);
                    vector<double> data(time_points * sdim);
                    for (size_t t = 0; t < time_points; ++t) {
                        const SpinVector& mag = is_su2 ? M0_trajectory[t].second.first[mag_idx] 
                                                       : M0_trajectory[t].second.second[mag_idx];
                        for (size_t d = 0; d < sdim; ++d) {
                            data[t * sdim + d] = mag(d);
                        }
                    }
                    H5::DataSet ds = grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dspace);
                    ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
                };
                
                write_mag_dataset(reference_group, "M_antiferro_SU2", spin_dim_SU2, 0, true);
                write_mag_dataset(reference_group, "M_local_SU2", spin_dim_SU2, 1, true);
                write_mag_dataset(reference_group, "M_global_SU2", spin_dim_SU2, 2, true);
                write_mag_dataset(reference_group, "M_antiferro_SU3", spin_dim_SU3, 0, false);
                write_mag_dataset(reference_group, "M_local_SU3", spin_dim_SU3, 1, false);
                write_mag_dataset(reference_group, "M_global_SU3", spin_dim_SU3, 2, false);
                
                // Write M0 spin state if enabled
                if (save_spin_trajectories && !M0_spin_flat.empty()) {
                    hsize_t n_t = static_cast<hsize_t>(time_points);
                    hsize_t sd = static_cast<hsize_t>(state_dim);
                    hsize_t dims[2] = {n_t, sd};
                    hsize_t chunk[2] = {std::min((hsize_t)128, n_t), sd};
                    H5::DSetCreatPropList plist;
                    plist.setChunk(2, chunk);
                    plist.setDeflate(4);
                    H5::DataSpace dsp(2, dims);
                    H5::DataSet ds = reference_group.createDataSet("M0_spin_state",
                                        H5::PredType::NATIVE_DOUBLE, dsp, plist);
                    ds.write(M0_spin_flat.data(), H5::PredType::NATIVE_DOUBLE);
                }
                
                cout << "  Reference trajectory (M0) written." << endl;
                
            } catch (H5::Exception& e) {
                std::cerr << "HDF5 Error opening file: " << e.getDetailMsg() << endl;
                if (file_ptr) delete file_ptr;
                file_ptr = nullptr;
            }
        }
        
        // Helper lambda to write a trajectory to HDF5 (rank 0 only)
        auto write_tau_to_hdf5 = [&](int tau_idx, const TrajectoryType& M1_traj, const TrajectoryType& M01_traj,
                                     const vector<double>* spin_M1_flat = nullptr,
                                     const vector<double>* spin_M01_flat = nullptr) {
            if (!file_ptr) return;
            
            std::string grp_name = "/tau_scan/tau_" + std::to_string(tau_idx);
            H5::Group tau_grp = file_ptr->createGroup(grp_name);
            
            // Write tau value as attribute
            H5::DataSpace attr_space(H5S_SCALAR);
            double tau_val = tau_values[tau_idx];
            H5::Attribute tau_attr = tau_grp.createAttribute("tau_value", H5::PredType::NATIVE_DOUBLE, attr_space);
            tau_attr.write(H5::PredType::NATIVE_DOUBLE, &tau_val);
            
            size_t n_times = M1_traj.size();
            
            auto write_mag = [&](const char* name, const TrajectoryType& traj, size_t sdim, int mag_idx, bool is_su2) {
                hsize_t dims[2] = {n_times, sdim};
                H5::DataSpace dspace(2, dims);
                vector<double> data(n_times * sdim);
                for (size_t t = 0; t < n_times; ++t) {
                    const SpinVector& mag = is_su2 ? traj[t].second.first[mag_idx] 
                                                   : traj[t].second.second[mag_idx];
                    for (size_t d = 0; d < sdim; ++d) {
                        data[t * sdim + d] = mag(d);
                    }
                }
                H5::DataSet ds = tau_grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dspace);
                ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            };
            
            write_mag("M1_antiferro_SU2", M1_traj, spin_dim_SU2, 0, true);
            write_mag("M1_local_SU2", M1_traj, spin_dim_SU2, 1, true);
            write_mag("M1_global_SU2", M1_traj, spin_dim_SU2, 2, true);
            write_mag("M1_antiferro_SU3", M1_traj, spin_dim_SU3, 0, false);
            write_mag("M1_local_SU3", M1_traj, spin_dim_SU3, 1, false);
            write_mag("M1_global_SU3", M1_traj, spin_dim_SU3, 2, false);
            
            write_mag("M01_antiferro_SU2", M01_traj, spin_dim_SU2, 0, true);
            write_mag("M01_local_SU2", M01_traj, spin_dim_SU2, 1, true);
            write_mag("M01_global_SU2", M01_traj, spin_dim_SU2, 2, true);
            write_mag("M01_antiferro_SU3", M01_traj, spin_dim_SU3, 0, false);
            write_mag("M01_local_SU3", M01_traj, spin_dim_SU3, 1, false);
            write_mag("M01_global_SU3", M01_traj, spin_dim_SU3, 2, false);
            
            // Write full spin state trajectories with gzip compression if provided
            auto write_spin_state = [&](const char* name, const vector<double>* flat_buf) {
                if (flat_buf == nullptr || flat_buf->empty()) return;
                hsize_t n_t = static_cast<hsize_t>(n_times);
                hsize_t sd = static_cast<hsize_t>(state_dim);
                hsize_t dims[2] = {n_t, sd};
                hsize_t chunk[2] = {std::min((hsize_t)128, n_t), sd};
                H5::DSetCreatPropList plist;
                plist.setChunk(2, chunk);
                plist.setDeflate(4);
                H5::DataSpace dsp(2, dims);
                H5::DataSet ds = tau_grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dsp, plist);
                ds.write(flat_buf->data(), H5::PredType::NATIVE_DOUBLE);
            };
            write_spin_state("M1_spin_state", spin_M1_flat);
            write_spin_state("M01_spin_state", spin_M01_flat);
            
            tau_grp.close();
        };
        
        // Helper lambda to deserialize buffer to trajectory
        auto deserialize_trajectory = [&](const vector<double>& buffer) -> TrajectoryType {
            TrajectoryType traj(time_points);
            for (size_t t = 0; t < time_points; ++t) {
                size_t offset = t * data_per_point;
                traj[t].first = buffer[offset];
                // SU2 magnetizations
                for (int m = 0; m < 3; ++m) {
                    traj[t].second.first[m] = SpinVector::Zero(spin_dim_SU2);
                    for (size_t d = 0; d < spin_dim_SU2; ++d) {
                        traj[t].second.first[m](d) = buffer[offset + 1 + m * spin_dim_SU2 + d];
                    }
                }
                // SU3 magnetizations
                size_t su3_offset = offset + 1 + 3 * spin_dim_SU2;
                for (int m = 0; m < 3; ++m) {
                    traj[t].second.second[m] = SpinVector::Zero(spin_dim_SU3);
                    for (size_t d = 0; d < spin_dim_SU3; ++d) {
                        traj[t].second.second[m](d) = buffer[su3_offset + m * spin_dim_SU3 + d];
                    }
                }
            }
            return traj;
        };
        
        // First: rank 0 writes its own local results immediately
        if (rank == 0) {
            for (size_t idx = 0; idx < my_tau_indices.size(); ++idx) {
                int tau_idx = my_tau_indices[idx];
                const vector<double>* sp_m1 = save_spin_trajectories && idx < local_spin_flat_M1.size()
                                              ? &local_spin_flat_M1[idx] : nullptr;
                const vector<double>* sp_m01 = save_spin_trajectories && idx < local_spin_flat_M01.size()
                                               ? &local_spin_flat_M01[idx] : nullptr;
                write_tau_to_hdf5(tau_idx, local_M1_trajectories[idx], local_M01_trajectories[idx], sp_m1, sp_m01);
            }
            cout << "  Rank 0 local trajectories written (" << my_tau_indices.size() << " tau points)." << endl;
            
            // Free local memory on rank 0 after writing
            local_M1_trajectories.clear();
            local_M1_trajectories.shrink_to_fit();
            local_M01_trajectories.clear();
            local_M01_trajectories.shrink_to_fit();
            local_spin_flat_M1.clear();
            local_spin_flat_M1.shrink_to_fit();
            local_spin_flat_M01.clear();
            local_spin_flat_M01.shrink_to_fit();
        }
        
        // Now receive from other ranks and write immediately (streaming)
        vector<double> M1_buffer(traj_size);
        vector<double> M01_buffer(traj_size);
        vector<double> spin_M1_buf, spin_M01_buf;
        if (save_spin_trajectories) {
            spin_M1_buf.resize(state_traj_size_spins);
            spin_M01_buf.resize(state_traj_size_spins);
        }
        
        int progress_interval = std::max(1, tau_steps / 20);  // Report every 5%
        int received_count = 0;
        
        for (int tau_idx = 0; tau_idx < tau_steps; ++tau_idx) {
            int owner_rank = scheduler_writer_only ? (1 + (tau_idx % worker_count)) : 0;
            
            if (owner_rank == 0) continue;  // Already written above
            
            if (rank == 0) {
                // Receive obs trajectories from owner
                MPI_Recv(M1_buffer.data(), traj_size, MPI_DOUBLE, owner_rank, 
                        2 * tau_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(M01_buffer.data(), traj_size, MPI_DOUBLE, owner_rank, 
                        2 * tau_idx + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Receive spin state trajectories if enabled
                if (save_spin_trajectories) {
                    MPI_Recv(spin_M1_buf.data(), state_traj_size_spins, MPI_DOUBLE, owner_rank,
                             2 * tau_steps + 2 * tau_idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(spin_M01_buf.data(), state_traj_size_spins, MPI_DOUBLE, owner_rank,
                             2 * tau_steps + 2 * tau_idx + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                
                // Deserialize and write to HDF5 immediately (no storage)
                TrajectoryType M1_traj = deserialize_trajectory(M1_buffer);
                TrajectoryType M01_traj = deserialize_trajectory(M01_buffer);
                
                write_tau_to_hdf5(tau_idx, M1_traj, M01_traj,
                                  save_spin_trajectories ? &spin_M1_buf : nullptr,
                                  save_spin_trajectories ? &spin_M01_buf : nullptr);
                
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
                        for (size_t d = 0; d < spin_dim_SU2; ++d) {
                            M1_buffer[offset + 1 + m * spin_dim_SU2 + d] = 
                                local_M1_trajectories[local_idx][t].second.first[m](d);
                        }
                    }
                    size_t su3_offset = offset + 1 + 3 * spin_dim_SU2;
                    for (int m = 0; m < 3; ++m) {
                        for (size_t d = 0; d < spin_dim_SU3; ++d) {
                            M1_buffer[su3_offset + m * spin_dim_SU3 + d] = 
                                local_M1_trajectories[local_idx][t].second.second[m](d);
                        }
                    }
                }
                
                // Serialize M01
                for (size_t t = 0; t < time_points; ++t) {
                    size_t offset = t * data_per_point;
                    M01_buffer[offset] = local_M01_trajectories[local_idx][t].first;
                    for (int m = 0; m < 3; ++m) {
                        for (size_t d = 0; d < spin_dim_SU2; ++d) {
                            M01_buffer[offset + 1 + m * spin_dim_SU2 + d] = 
                                local_M01_trajectories[local_idx][t].second.first[m](d);
                        }
                    }
                    size_t su3_offset = offset + 1 + 3 * spin_dim_SU2;
                    for (int m = 0; m < 3; ++m) {
                        for (size_t d = 0; d < spin_dim_SU3; ++d) {
                            M01_buffer[su3_offset + m * spin_dim_SU3 + d] = 
                                local_M01_trajectories[local_idx][t].second.second[m](d);
                        }
                    }
                }
                
                MPI_Send(M1_buffer.data(), traj_size, MPI_DOUBLE, 0, 2 * tau_idx, MPI_COMM_WORLD);
                MPI_Send(M01_buffer.data(), traj_size, MPI_DOUBLE, 0, 2 * tau_idx + 1, MPI_COMM_WORLD);
                
                // Send spin state trajectories if enabled
                if (save_spin_trajectories) {
                    const vector<double>& sf_m1 = local_spin_flat_M1[local_idx];
                    const vector<double>& sf_m01 = local_spin_flat_M01[local_idx];
                    MPI_Send(sf_m1.data(), static_cast<int>(sf_m1.size()), MPI_DOUBLE, 0,
                             2 * tau_steps + 2 * tau_idx, MPI_COMM_WORLD);
                    MPI_Send(sf_m01.data(), static_cast<int>(sf_m01.size()), MPI_DOUBLE, 0,
                             2 * tau_steps + 2 * tau_idx + 1, MPI_COMM_WORLD);
                }
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
            cout << "Note: HDF5 support not enabled. Skipping file output." << endl;
        }
#endif
        
        if (rank == 0) {
            cout << "\n==========================================" << endl;
            cout << "Pump-Probe Spectroscopy (MPI) Complete!" << endl;
            cout << "Output directory: " << dir_name << endl;
            cout << "Total delay points: " << tau_steps << endl;
            cout << "==========================================" << endl;
        }
        
        // Restore ground state
        spins_SU2 = ground_state_SU2;
        spins_SU3 = ground_state_SU3;
        
        MPI_Barrier(MPI_COMM_WORLD);
    }

