/**
 * ncto_lattice.cpp - Implementation of NCTO spin-phonon lattice
 * 
 * NCTO = Na2Co2TeO6 (or similar honeycomb Kitaev material with phonons)
 * 
 * This implements:
 * - Honeycomb lattice structure with Kitaev-type bonds
 * - Global IR (E1) and Raman (A1) phonon modes
 * - Spin-phonon coupling (Q_R-dependent exchanges)
 * - Nonlinear phononics (cubic g*Q_R*(Qx^2+Qy^2) coupling)
 * - THz drive coupling to IR mode
 */

#include "classical_spin/lattice/ncto_lattice.h"
#include <fstream>
#include <iomanip>

namespace odeint = boost::numeric::odeint;

// ============================================================
// HONEYCOMB LATTICE CONSTRUCTION
// ============================================================

void NCTOLattice::build_honeycomb() {
    // Honeycomb lattice vectors
    Eigen::Vector3d a1(1.0, 0.0, 0.0);
    Eigen::Vector3d a2(0.5, std::sqrt(3.0)/2.0, 0.0);
    Eigen::Vector3d a3(0.0, 0.0, 1.0);
    
    // Sublattice positions within unit cell
    Eigen::Vector3d pos0(0.0, 0.0, 0.0);
    Eigen::Vector3d pos1(0.0, 1.0/std::sqrt(3.0), 0.0);
    
    // Build lattice sites
    size_t site_idx = 0;
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                // Sublattice 0
                site_positions[site_idx] = pos0 + double(i)*a1 + double(j)*a2 + double(k)*a3;
                field[site_idx] = Eigen::Vector3d::Zero();
                ++site_idx;
                
                // Sublattice 1
                site_positions[site_idx] = pos1 + double(i)*a1 + double(j)*a2 + double(k)*a3;
                field[site_idx] = Eigen::Vector3d::Zero();
                ++site_idx;
            }
        }
    }
    
    cout << "Built honeycomb lattice with " << lattice_size << " sites" << endl;
}

// ============================================================
// PARAMETER SETTING AND INTERACTION BUILDING
// ============================================================

void NCTOLattice::set_parameters(const NCTOSpinPhononParams& sp_params,
                                 const NCTOPhononParams& ph_params,
                                 const NCTODriveParams& dr_params) {
    spin_phonon_params = sp_params;
    phonon_params = ph_params;
    drive_params = dr_params;
    
    // Build Kitaev interaction matrices for each bond type
    // Bond types: x-bond (0), y-bond (1), z-bond (2)
    
    // At Q_R = 0, the effective exchange parameters are the base values
    double J = sp_params.J1_0;
    double K = sp_params.K_0;
    double G = sp_params.Gamma_0;
    double Gp = sp_params.Gammap_0;
    
    // Coupling strengths for spin-phonon
    double lJ = sp_params.lambda_J;
    double lK = sp_params.lambda_K;
    double lG = sp_params.lambda_Gamma;
    double lGp = sp_params.lambda_Gammap;
    
    // Build base and coupling matrices for each bond type
    // x-bond: Kitaev on S^x S^x, Gamma on yz+zy
    Eigen::Matrix3d Jx_base, Jx_coup;
    Jx_base << J + K, Gp, Gp,
               Gp, J, G,
               Gp, G, J;
    Jx_coup << lJ + lK, lGp, lGp,
               lGp, lJ, lG,
               lGp, lG, lJ;
    
    // y-bond: Kitaev on S^y S^y, Gamma on xz+zx
    Eigen::Matrix3d Jy_base, Jy_coup;
    Jy_base << J, Gp, G,
               Gp, J + K, Gp,
               G, Gp, J;
    Jy_coup << lJ, lGp, lG,
               lGp, lJ + lK, lGp,
               lG, lGp, lJ;
    
    // z-bond: Kitaev on S^z S^z, Gamma on xy+yx
    Eigen::Matrix3d Jz_base, Jz_coup;
    Jz_base << J, G, Gp,
               G, J, Gp,
               Gp, Gp, J + K;
    Jz_coup << lJ, lG, lGp,
               lG, lJ, lGp,
               lGp, lGp, lJ + lK;
    
    // J3 matrix (isotropic Heisenberg)
    Eigen::Matrix3d J3_mat = sp_params.J3 * Eigen::Matrix3d::Identity();
    
    // Clear existing interactions
    for (size_t i = 0; i < lattice_size; ++i) {
        bilinear_base[i].clear();
        bilinear_coupling[i].clear();
        bilinear_partners[i].clear();
        bilinear_bond_types[i].clear();
        j3_interaction[i].clear();
        j3_partners[i].clear();
    }
    
    // Build nearest-neighbor interactions
    // For each unit cell, atom 0 connects to atom 1 via three bond types
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                size_t site0 = flatten_index(i, j, k, 0);  // Sublattice A
                size_t site1 = flatten_index(i, j, k, 1);  // Sublattice B in same cell
                
                // x-bond: (i,j,k,0) -> (i,j-1,k,1)
                size_t partner_x = flatten_index_periodic(i, j-1, k, 1);
                bilinear_base[site0].push_back(Jx_base);
                bilinear_coupling[site0].push_back(Jx_coup);
                bilinear_partners[site0].push_back(partner_x);
                bilinear_bond_types[site0].push_back(0);
                // Reverse bond
                bilinear_base[partner_x].push_back(Jx_base.transpose());
                bilinear_coupling[partner_x].push_back(Jx_coup.transpose());
                bilinear_partners[partner_x].push_back(site0);
                bilinear_bond_types[partner_x].push_back(0);
                
                // y-bond: (i,j,k,0) -> (i+1,j-1,k,1)
                size_t partner_y = flatten_index_periodic(i+1, j-1, k, 1);
                bilinear_base[site0].push_back(Jy_base);
                bilinear_coupling[site0].push_back(Jy_coup);
                bilinear_partners[site0].push_back(partner_y);
                bilinear_bond_types[site0].push_back(1);
                // Reverse bond
                bilinear_base[partner_y].push_back(Jy_base.transpose());
                bilinear_coupling[partner_y].push_back(Jy_coup.transpose());
                bilinear_partners[partner_y].push_back(site0);
                bilinear_bond_types[partner_y].push_back(1);
                
                // z-bond: (i,j,k,0) -> (i,j,k,1) (same unit cell)
                bilinear_base[site0].push_back(Jz_base);
                bilinear_coupling[site0].push_back(Jz_coup);
                bilinear_partners[site0].push_back(site1);
                bilinear_bond_types[site0].push_back(2);
                // Reverse bond
                bilinear_base[site1].push_back(Jz_base.transpose());
                bilinear_coupling[site1].push_back(Jz_coup.transpose());
                bilinear_partners[site1].push_back(site0);
                bilinear_bond_types[site1].push_back(2);
                
                // Third-neighbor interactions (J3)
                // Third neighbors on honeycomb: same sublattice, connected through intermediate sites
                if (std::abs(sp_params.J3) > 1e-12) {
                    // From sublattice A (site0):
                    // Third neighbors at offsets: (±1,0,0), (-1,±1,0), (+1,-2,0) in unit cell coords
                    // Actually for honeycomb, third neighbors are those at distance 2*a
                    
                    // J3 bonds for sublattice A (atom 0)
                    vector<Eigen::Vector3i> j3_offsets_A = {
                        {1, 0, 0}, {-1, 0, 0}, {1, -2, 0}
                    };
                    for (const auto& off : j3_offsets_A) {
                        size_t partner_j3 = flatten_index_periodic(i+off(0), j+off(1), k+off(2), 1);
                        j3_interaction[site0].push_back(J3_mat);
                        j3_partners[site0].push_back(partner_j3);
                        // Reverse
                        j3_interaction[partner_j3].push_back(J3_mat);
                        j3_partners[partner_j3].push_back(site0);
                    }
                }
            }
        }
    }
    
    cout << "Set NCTO parameters:" << endl;
    cout << "  Spin-phonon: J1=" << sp_params.J1_0 << ", K=" << sp_params.K_0 
         << ", Γ=" << sp_params.Gamma_0 << ", J3=" << sp_params.J3 << endl;
    cout << "  Couplings: λ_J=" << sp_params.lambda_J << ", λ_K=" << sp_params.lambda_K
         << ", λ_Γ=" << sp_params.lambda_Gamma << endl;
    cout << "  Phonon: ω_IR=" << ph_params.omega_IR << ", ω_R=" << ph_params.omega_R
         << ", g=" << ph_params.g << ", β=" << ph_params.beta << endl;
    cout << "  Drive: E0_1=" << dr_params.E0_1 << ", ω_1=" << dr_params.omega_1
         << ", t_1=" << dr_params.t_1 << endl;
}

// ============================================================
// ENERGY CALCULATIONS
// ============================================================

double NCTOLattice::spin_energy() const {
    double E = 0.0;
    double Q_R = phonons.Q_R;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        // Zeeman energy: -B · S
        E -= Si.dot(field[i]);
        
        // Bilinear interactions with Q_R dependence
        // J_eff = J_base + Q_R * J_coupling
        for (size_t n = 0; n < bilinear_partners[i].size(); ++n) {
            size_t j = bilinear_partners[i][n];
            if (j > i) {  // Avoid double counting
                const Eigen::Vector3d& Sj = spins[j];
                Eigen::Matrix3d J_eff = bilinear_base[i][n] + Q_R * bilinear_coupling[i][n];
                E += Si.dot(J_eff * Sj);
            }
        }
        
        // J3 interactions (not Q_R dependent)
        for (size_t n = 0; n < j3_partners[i].size(); ++n) {
            size_t j = j3_partners[i][n];
            if (j > i) {  // Avoid double counting
                const Eigen::Vector3d& Sj = spins[j];
                E += Si.dot(j3_interaction[i][n] * Sj);
            }
        }
    }
    
    return E;
}

double NCTOLattice::spin_phonon_energy() const {
    // The spin-phonon coupling energy is the Q_R-dependent part of H_spin
    // H_sp-ph = Q_R * Σ [λJ Si·Sj + λK Si^γ Sj^γ + λΓ(Si^α Sj^β + Si^β Sj^α)]
    // This is already included in spin_energy(), but we can extract just the coupling part
    
    double E = 0.0;
    double Q_R = phonons.Q_R;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < bilinear_partners[i].size(); ++n) {
            size_t j = bilinear_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                E += Q_R * Si.dot(bilinear_coupling[i][n] * Sj);
            }
        }
    }
    
    return E;
}

double NCTOLattice::dH_spin_dQR() const {
    // Derivative of spin-phonon energy with respect to Q_R
    // dH_sp-ph/dQ_R = Σ [λJ Si·Sj + λK Si^γ Sj^γ + λΓ(Si^α Sj^β + Si^β Sj^α)]
    
    double dE = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < bilinear_partners[i].size(); ++n) {
            size_t j = bilinear_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                dE += Si.dot(bilinear_coupling[i][n] * Sj);
            }
        }
    }
    
    return dE;
}

// ============================================================
// LOCAL FIELD CALCULATION
// ============================================================

SpinVector NCTOLattice::get_local_field(size_t site, double Q_R) const {
    // H_eff = -dH/dS = B - 2*A*S - Σ_j J_eff * S_j
    // For now we ignore on-site anisotropy
    
    Eigen::Vector3d H = field[site];  // External field
    
    // Bilinear contributions
    for (size_t n = 0; n < bilinear_partners[site].size(); ++n) {
        size_t j = bilinear_partners[site][n];
        const Eigen::Vector3d& Sj = spins[j];
        Eigen::Matrix3d J_eff = bilinear_base[site][n] + Q_R * bilinear_coupling[site][n];
        H -= J_eff * Sj;
    }
    
    // J3 contributions
    for (size_t n = 0; n < j3_partners[site].size(); ++n) {
        size_t j = j3_partners[site][n];
        const Eigen::Vector3d& Sj = spins[j];
        H -= j3_interaction[site][n] * Sj;
    }
    
    return H;
}

// ============================================================
// EQUATIONS OF MOTION
// ============================================================

void NCTOLattice::phonon_derivatives(const PhononState& ph, double t, double dH_dQR,
                                     double& dQx, double& dQy, double& dQR,
                                     double& dVx, double& dVy, double& dVR) const {
    // Get THz drive field
    double Ex, Ey;
    drive_params.E_field(t, Ex, Ey);
    
    // =============================================
    // IR mode equations (driven damped oscillator)
    // =============================================
    // dQx/dt = Vx
    dQx = ph.V_x;
    // d²Qx/dt² = -ω_IR² Qx - 2g*Q_R*Qx - γ_IR*Vx + Z*Ex(t)
    dVx = -phonon_params.omega_IR * phonon_params.omega_IR * ph.Q_x
          - 2.0 * phonon_params.g * ph.Q_R * ph.Q_x
          - phonon_params.gamma_IR * ph.V_x
          + phonon_params.Z_star * Ex;
    
    // dQy/dt = Vy
    dQy = ph.V_y;
    // d²Qy/dt² = -ω_IR² Qy - 2g*Q_R*Qy - γ_IR*Vy + Z*Ey(t)
    dVy = -phonon_params.omega_IR * phonon_params.omega_IR * ph.Q_y
          - 2.0 * phonon_params.g * ph.Q_R * ph.Q_y
          - phonon_params.gamma_IR * ph.V_y
          + phonon_params.Z_star * Ey;
    
    // =============================================
    // Raman mode equation
    // =============================================
    // dQ_R/dt = V_R
    dQR = ph.V_R;
    // d²Q_R/dt² = -ω_R² Q_R - β*Q_R³ - g*(Qx²+Qy²) - γ_R*V_R - dH_sp-ph/dQ_R
    double Q_IR_sq = ph.Q_x * ph.Q_x + ph.Q_y * ph.Q_y;
    dVR = -phonon_params.omega_R * phonon_params.omega_R * ph.Q_R
          - phonon_params.beta * ph.Q_R * ph.Q_R * ph.Q_R
          - phonon_params.g * Q_IR_sq
          - phonon_params.gamma_R * ph.V_R
          - dH_dQR;
}

void NCTOLattice::ode_system(const ODEState& x, ODEState& dxdt, double t) {
    // State layout: [S0_x, S0_y, S0_z, S1_x, ..., SN_z, Qx, Qy, Q_R, Vx, Vy, V_R]
    
    const size_t spin_offset = spin_dim * lattice_size;
    
    // Extract phonon state
    PhononState ph;
    ph.from_array(&x[spin_offset]);
    double Q_R = ph.Q_R;
    
    // Compute dH/dQ_R for Raman equation
    // We need to use the spins from state x, not from this->spins
    // dH_sp-ph/dQ_R = Σ [λJ Si·Sj + λK Si^γ Sj^γ + λΓ(Si^α Sj^β + Si^β Sj^α)]
    double dH_dQR_val = 0.0;
    for (size_t i = 0; i < lattice_size; ++i) {
        const size_t idx = i * spin_dim;
        Eigen::Vector3d Si(x[idx], x[idx+1], x[idx+2]);
        
        for (size_t n = 0; n < bilinear_partners[i].size(); ++n) {
            size_t j = bilinear_partners[i][n];
            if (j > i) {
                const size_t jdx = j * spin_dim;
                Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
                dH_dQR_val += Si.dot(bilinear_coupling[i][n] * Sj);
            }
        }
    }
    
    // =============================================
    // Spin equations: dS/dt = S × H_eff + α S × (S × H_eff)
    // =============================================
    for (size_t i = 0; i < lattice_size; ++i) {
        const size_t idx = i * spin_dim;
        Eigen::Vector3d Si(x[idx], x[idx+1], x[idx+2]);
        
        // Compute local field: H_eff = B - Σ_j J_eff(Q_R) · S_j
        Eigen::Vector3d H = field[i];
        for (size_t n = 0; n < bilinear_partners[i].size(); ++n) {
            size_t j = bilinear_partners[i][n];
            const size_t jdx = j * spin_dim;
            Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
            Eigen::Matrix3d J_eff = bilinear_base[i][n] + Q_R * bilinear_coupling[i][n];
            H -= J_eff * Sj;
        }
        for (size_t n = 0; n < j3_partners[i].size(); ++n) {
            size_t j = j3_partners[i][n];
            const size_t jdx = j * spin_dim;
            Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
            H -= j3_interaction[i][n] * Sj;
        }
        
        // LLG: dS/dt = S × H + α S × (S × H)
        Eigen::Vector3d dSdt = Si.cross(H);
        if (alpha_gilbert > 0) {
            dSdt += alpha_gilbert * Si.cross(Si.cross(H)) / spin_length;
        }
        
        dxdt[idx] = dSdt(0);
        dxdt[idx+1] = dSdt(1);
        dxdt[idx+2] = dSdt(2);
    }
    
    // =============================================
    // Phonon equations: Newton + damping
    // =============================================
    double dQx, dQy, dQR, dVx, dVy, dVR;
    phonon_derivatives(ph, t, dH_dQR_val, dQx, dQy, dQR, dVx, dVy, dVR);
    
    // State layout: [... spins ..., Qx, Qy, Q_R, Vx, Vy, V_R]
    dxdt[spin_offset]   = dQx;  // dQx/dt = Vx
    dxdt[spin_offset+1] = dQy;  // dQy/dt = Vy
    dxdt[spin_offset+2] = dQR;  // dQ_R/dt = V_R
    dxdt[spin_offset+3] = dVx;  // dVx/dt = acceleration
    dxdt[spin_offset+4] = dVy;  // dVy/dt = acceleration
    dxdt[spin_offset+5] = dVR;  // dV_R/dt = acceleration
}

// ============================================================
// ODE INTEGRATION
// ============================================================

template<typename System, typename Observer>
void NCTOLattice::integrate_ode_system(System system_func, ODEState& state,
                                       double T_start, double T_end, double dt_step,
                                       Observer observer, const string& method,
                                       bool use_adaptive,
                                       double abs_tol, double rel_tol) {
    namespace odeint = boost::numeric::odeint;
    
    if (method == "euler") {
        odeint::integrate_const(
            odeint::euler<ODEState>(),
            system_func, state, T_start, T_end, dt_step, observer
        );
    } else if (method == "rk2" || method == "midpoint") {
        odeint::integrate_const(
            odeint::modified_midpoint<ODEState>(),
            system_func, state, T_start, T_end, dt_step, observer
        );
    } else if (method == "rk4") {
        odeint::integrate_const(
            odeint::runge_kutta4<ODEState>(),
            system_func, state, T_start, T_end, dt_step, observer
        );
    } else if (method == "dopri5") {
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer
            );
        } else {
            odeint::integrate_const(
                odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer
            );
        }
    } else if (method == "rk78" || method == "rkf78") {
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_controlled<odeint::runge_kutta_fehlberg78<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer
            );
        } else {
            odeint::integrate_const(
                odeint::make_controlled<odeint::runge_kutta_fehlberg78<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer
            );
        }
    } else if (method == "bulirsch_stoer" || method == "bs") {
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::bulirsch_stoer<ODEState>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer
            );
        } else {
            odeint::integrate_const(
                odeint::bulirsch_stoer<ODEState>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer
            );
        }
    } else {
        cout << "Warning: Unknown method '" << method << "', using dopri5" << endl;
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer
            );
        } else {
            odeint::integrate_const(
                odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer
            );
        }
    }
}

// ============================================================
// MOLECULAR DYNAMICS
// ============================================================

void NCTOLattice::molecular_dynamics(double T_start, double T_end, double dt_initial,
                                     string out_dir, size_t save_interval,
                                     string method) {
    if (!out_dir.empty()) {
        std::filesystem::create_directories(out_dir);
    }
    
    cout << "Running NCTO spin-phonon dynamics: t=" << T_start << " → " << T_end << endl;
    cout << "Integration method: " << method << endl;
    cout << "Initial step size: " << dt_initial << endl;
    
    // Convert current state to flat ODE state vector
    ODEState state = to_state();
    
    // Open output file for trajectory
    std::ofstream traj_file;
    if (!out_dir.empty()) {
        traj_file.open(out_dir + "/trajectory.txt");
        traj_file << "# time Mx My Mz Mstag_x Mstag_y Mstag_z Qx Qy Q_R E\n";
        traj_file << std::scientific << std::setprecision(8);
    }
    
    // Observer to save data at specified intervals
    size_t step_count = 0;
    size_t save_count = 0;
    auto observer = [&](const ODEState& x, double t) {
        if (step_count % save_interval == 0) {
            // Update internal state for observable computation
            const_cast<NCTOLattice*>(this)->from_state(x);
            
            Eigen::Vector3d M = magnetization();
            Eigen::Vector3d Ms = staggered_magnetization();
            double E = energy_density();
            
            if (traj_file.is_open()) {
                traj_file << t << " "
                         << M(0) << " " << M(1) << " " << M(2) << " "
                         << Ms(0) << " " << Ms(1) << " " << Ms(2) << " "
                         << phonons.Q_x << " " << phonons.Q_y << " " << phonons.Q_R << " "
                         << E << "\n";
            }
            
            if (step_count % (save_interval * 10) == 0) {
                cout << "t=" << t << ", E=" << E 
                     << ", |M|=" << M.norm() 
                     << ", |Q_IR|=" << IR_amplitude()
                     << ", Q_R=" << phonons.Q_R << endl;
            }
            
            save_count++;
        }
        step_count++;
    };
    
    // Create ODE system wrapper
    auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
        this->ode_system(x, dxdt, t);
    };
    
    // Integrate
    double abs_tol = 1e-6, rel_tol = 1e-6;
    integrate_ode_system(system_func, state, T_start, T_end, dt_initial,
                        observer, method, true, abs_tol, rel_tol);
    
    // Update internal state with final configuration
    from_state(state);
    
    if (traj_file.is_open()) {
        traj_file.close();
        cout << "Trajectory saved to " << out_dir << "/trajectory.txt" << endl;
    }
    
    cout << "NCTO dynamics complete! (" << step_count << " steps, " << save_count << " saved)" << endl;
}

// ============================================================
// SIMULATED ANNEALING
// ============================================================

void NCTOLattice::simulated_annealing(double T_start, double T_end, size_t n_steps,
                                      size_t overrelax_rate,
                                      double cooling_rate,
                                      string out_dir,
                                      bool save_observables) {
    cout << "Running NCTO simulated annealing (spin subsystem only)..." << endl;
    cout << "T: " << T_start << " → " << T_end << ", steps: " << n_steps << endl;
    
    // Keep phonons at equilibrium (Q_R = 0) during thermal equilibration
    phonons = PhononState();
    double Q_R = phonons.Q_R;
    
    if (!out_dir.empty()) {
        std::filesystem::create_directories(out_dir);
    }
    
    std::ofstream obs_file;
    if (save_observables && !out_dir.empty()) {
        obs_file.open(out_dir + "/annealing.txt");
        obs_file << "# step T E acc_rate\n";
    }
    
    double T = T_start;
    size_t accepted = 0;
    size_t total_moves = 0;
    
    for (size_t step = 0; step < n_steps; ++step) {
        // Single Metropolis sweep
        for (size_t i = 0; i < lattice_size; ++i) {
            // Save old spin and energy
            Eigen::Vector3d old_spin = spins[i];
            
            // Compute old local energy
            double E_old = -old_spin.dot(field[i]);
            for (size_t n = 0; n < bilinear_partners[i].size(); ++n) {
                size_t j = bilinear_partners[i][n];
                Eigen::Matrix3d J_eff = bilinear_base[i][n] + Q_R * bilinear_coupling[i][n];
                E_old += old_spin.dot(J_eff * spins[j]);
            }
            for (size_t n = 0; n < j3_partners[i].size(); ++n) {
                size_t j = j3_partners[i][n];
                E_old += old_spin.dot(j3_interaction[i][n] * spins[j]);
            }
            
            // Propose new spin
            Eigen::Vector3d new_spin = gen_random_spin();
            
            // Compute new local energy
            double E_new = -new_spin.dot(field[i]);
            for (size_t n = 0; n < bilinear_partners[i].size(); ++n) {
                size_t j = bilinear_partners[i][n];
                Eigen::Matrix3d J_eff = bilinear_base[i][n] + Q_R * bilinear_coupling[i][n];
                E_new += new_spin.dot(J_eff * spins[j]);
            }
            for (size_t n = 0; n < j3_partners[i].size(); ++n) {
                size_t j = j3_partners[i][n];
                E_new += new_spin.dot(j3_interaction[i][n] * spins[j]);
            }
            
            // Metropolis accept/reject
            double dE = E_new - E_old;
            if (dE < 0 || random_double_lehman(0, 1) < std::exp(-dE / T)) {
                spins[i] = new_spin;
                accepted++;
            }
            total_moves++;
        }
        
        // Overrelaxation
        if (overrelax_rate > 0 && step % overrelax_rate == 0) {
            for (size_t i = 0; i < lattice_size; ++i) {
                Eigen::Vector3d H = get_local_field(i, Q_R);
                if (H.norm() > 1e-10) {
                    H.normalize();
                    spins[i] = 2.0 * H.dot(spins[i]) * H - spins[i];
                    spins[i] = spins[i].normalized() * spin_length;
                }
            }
        }
        
        // Save observables
        if (save_observables && step % 100 == 0 && obs_file.is_open()) {
            double acc_rate = double(accepted) / total_moves;
            obs_file << step << " " << T << " " << energy_density() << " " << acc_rate << "\n";
        }
        
        // Cool down
        T *= cooling_rate;
        if (T < T_end) T = T_end;
    }
    
    if (obs_file.is_open()) {
        obs_file.close();
    }
    
    cout << "Simulated annealing complete! Final E=" << energy_density() << endl;
}

// ============================================================
// I/O
// ============================================================

void NCTOLattice::save_spin_config(const string& filename) const {
    std::ofstream file(filename);
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        file << spins[i](0) << " " << spins[i](1) << " " << spins[i](2) << "\n";
    }
    
    file.close();
}

void NCTOLattice::load_spin_config(const string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    for (size_t i = 0; i < lattice_size; ++i) {
        file >> spins[i](0) >> spins[i](1) >> spins[i](2);
        spins[i] = spins[i].normalized() * spin_length;
    }
    
    file.close();
}

void NCTOLattice::save_positions(const string& filename) const {
    std::ofstream file(filename);
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        file << site_positions[i](0) << " " 
             << site_positions[i](1) << " " 
             << site_positions[i](2) << "\n";
    }
    
    file.close();
}

// Explicit template instantiation for common observers
template void NCTOLattice::integrate_ode_system(
    std::function<void(const NCTOLattice::ODEState&, NCTOLattice::ODEState&, double)>,
    NCTOLattice::ODEState&, double, double, double,
    std::function<void(const NCTOLattice::ODEState&, double)>,
    const std::string&, bool, double, double);
