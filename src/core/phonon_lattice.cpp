/**
 * @file phonon_lattice.cpp
 * @brief Implementation of spin-phonon coupled honeycomb lattice
 * 
 * Implements honeycomb lattice with:
 * - Generic NN and 3rd NN spin interactions
 * - E1 (Qx, Qy) and A1 (Q_R) phonon modes
 * - Three-phonon coupling: g3*(Qx² + Qy²)*Q_R
 * - Spin-phonon: Qx*(SxSz+SzSx) + Qy*(SySz+SzSy) + Q_R*(SxSx+SySy+SzSz)
 */

#include "classical_spin/lattice/phonon_lattice.h"
#include <fstream>
#include <iomanip>
#include <memory>

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

namespace odeint = boost::numeric::odeint;

// ============================================================
// CONSTRUCTOR
// ============================================================

PhononLattice::PhononLattice(size_t d1, size_t d2, size_t d3, float spin_l)
    : dim1(d1), dim2(d2), dim3(d3), spin_length(spin_l)
{
    lattice_size = N_atoms * dim1 * dim2 * dim3;
    state_size = spin_dim * lattice_size + PhononState::N_DOF;
    
    // Initialize arrays
    spins.resize(lattice_size);
    site_positions.resize(lattice_size);
    field.resize(lattice_size);
    
    nn_interaction.resize(lattice_size);
    nn_partners.resize(lattice_size);
    nn_bond_types.resize(lattice_size);
    j3_interaction.resize(lattice_size);
    j3_partners.resize(lattice_size);
    
    // Initialize RNG
    seed_lehman(std::chrono::system_clock::now().time_since_epoch().count() * 2 + 1);
    
    cout << "Initializing PhononLattice with dimensions: " 
         << dim1 << " x " << dim2 << " x " << dim3 << endl;
    cout << "Total spin sites: " << lattice_size << endl;
    cout << "Phonon DOF: " << PhononState::N_DOF << " (Qx, Qy, Q_R, Vx, Vy, V_R)" << endl;
    cout << "Total ODE state size: " << state_size << endl;
    
    // Build lattice
    build_honeycomb();
    
    // Initialize random spins
    init_random();
    
    cout << "PhononLattice initialization complete!" << endl;
}

// ============================================================
// HONEYCOMB LATTICE CONSTRUCTION
// ============================================================

void PhononLattice::build_honeycomb() {
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
                // Sublattice A
                site_positions[site_idx] = pos0 + double(i)*a1 + double(j)*a2 + double(k)*a3;
                field[site_idx] = Eigen::Vector3d::Zero();
                ++site_idx;
                
                // Sublattice B
                site_positions[site_idx] = pos1 + double(i)*a1 + double(j)*a2 + double(k)*a3;
                field[site_idx] = Eigen::Vector3d::Zero();
                ++site_idx;
            }
        }
    }
    
    cout << "Built honeycomb lattice with " << lattice_size << " sites" << endl;
}

// ============================================================
// PARAMETER SETTING
// ============================================================

void PhononLattice::set_parameters(const SpinPhononCouplingParams& sp_params,
                                   const PhononParams& ph_params,
                                   const DriveParams& dr_params) {
    spin_phonon_params = sp_params;
    phonon_params = ph_params;
    drive_params = dr_params;
    
    // Clear existing interactions
    for (size_t i = 0; i < lattice_size; ++i) {
        nn_interaction[i].clear();
        nn_partners[i].clear();
        nn_bond_types[i].clear();
        j3_interaction[i].clear();
        j3_partners[i].clear();
    }
    
    // Build NN interactions on honeycomb
    // For honeycomb lattice with 2 atoms per unit cell:
    // - x-bond (type 0): connects (i,j,k,0) to (i,j-1,k,1)
    // - y-bond (type 1): connects (i,j,k,0) to (i+1,j-1,k,1)  
    // - z-bond (type 2): connects (i,j,k,0) to (i,j,k,1)
    
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                size_t site0 = flatten_index(i, j, k, 0);  // Sublattice A
                size_t site1 = flatten_index(i, j, k, 1);  // Sublattice B
                
                // x-bond
                size_t partner_x = flatten_index_periodic(i, j-1, k, 1);
                nn_interaction[site0].push_back(sp_params.J1);
                nn_partners[site0].push_back(partner_x);
                nn_bond_types[site0].push_back(0);
                // Reverse bond
                nn_interaction[partner_x].push_back(sp_params.J1.transpose());
                nn_partners[partner_x].push_back(site0);
                nn_bond_types[partner_x].push_back(0);
                
                // y-bond
                size_t partner_y = flatten_index_periodic(i+1, j-1, k, 1);
                nn_interaction[site0].push_back(sp_params.J1);
                nn_partners[site0].push_back(partner_y);
                nn_bond_types[site0].push_back(1);
                // Reverse bond
                nn_interaction[partner_y].push_back(sp_params.J1.transpose());
                nn_partners[partner_y].push_back(site0);
                nn_bond_types[partner_y].push_back(1);
                
                // z-bond (same unit cell)
                nn_interaction[site0].push_back(sp_params.J1);
                nn_partners[site0].push_back(site1);
                nn_bond_types[site0].push_back(2);
                // Reverse bond
                nn_interaction[site1].push_back(sp_params.J1.transpose());
                nn_partners[site1].push_back(site0);
                nn_bond_types[site1].push_back(2);
                
                // 3rd NN interactions
                // On honeycomb, 3rd NN are at distance 2a, connecting same sublattice
                if (sp_params.J3.norm() > 1e-12) {
                    // 3rd NN offsets for sublattice A
                    vector<std::tuple<int,int,int,size_t>> j3_offsets = {
                        {1, 0, 0, 0}, {-1, 0, 0, 0}, {0, 1, 0, 0}, {0, -1, 0, 0}, {1, -1, 0, 0}, {-1, 1, 0, 0}
                    };
                    
                    for (const auto& [di, dj, dk, atom] : j3_offsets) {
                        size_t partner_j3 = flatten_index_periodic(i+di, j+dj, k+dk, atom);
                        // Only add if partner > site0 to avoid double counting
                        if (partner_j3 > site0) {
                            j3_interaction[site0].push_back(sp_params.J3);
                            j3_partners[site0].push_back(partner_j3);
                            j3_interaction[partner_j3].push_back(sp_params.J3.transpose());
                            j3_partners[partner_j3].push_back(site0);
                        }
                    }
                    
                    // 3rd NN for sublattice B (site1)
                    for (const auto& [di, dj, dk, atom_offset] : j3_offsets) {
                        size_t partner_j3 = flatten_index_periodic(i+di, j+dj, k+dk, 1);
                        if (partner_j3 > site1) {
                            j3_interaction[site1].push_back(sp_params.J3);
                            j3_partners[site1].push_back(partner_j3);
                            j3_interaction[partner_j3].push_back(sp_params.J3.transpose());
                            j3_partners[partner_j3].push_back(site1);
                        }
                    }
                }
            }
        }
    }
    
    cout << "Set PhononLattice parameters:" << endl;
    cout << "  J1 matrix:" << endl << sp_params.J1 << endl;
    cout << "  J3 matrix:" << endl << sp_params.J3 << endl;
    cout << "  λ_xy=" << sp_params.lambda_xy << ", λ_R=" << sp_params.lambda_R << endl;
    cout << "  Phonon: ω_E=" << ph_params.omega_E << ", ω_A=" << ph_params.omega_A
         << ", g3=" << ph_params.g3 << endl;
    cout << "  Drive: E0_1=" << dr_params.E0_1 << ", ω_1=" << dr_params.omega_1 << endl;
}

// ============================================================
// ENERGY CALCULATIONS
// ============================================================

double PhononLattice::spin_energy() const {
    double E = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        // Zeeman: -B · S
        E -= Si.dot(field[i]);
        
        // NN interactions
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {  // Avoid double counting
                const Eigen::Vector3d& Sj = spins[j];
                E += Si.dot(nn_interaction[i][n] * Sj);
            }
        }
        
        // 3rd NN interactions
        for (size_t n = 0; n < j3_partners[i].size(); ++n) {
            size_t j = j3_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                E += Si.dot(j3_interaction[i][n] * Sj);
            }
        }
    }
    
    return E;
}

double PhononLattice::phonon_energy() const {
    // Kinetic: (1/2)(Vx² + Vy² + V_R²)
    double T = phonons.kinetic_energy();
    
    // E1 potential: (1/2)ω_E²(Qx² + Qy²)
    double Q_E_sq = phonons.Q_x * phonons.Q_x + phonons.Q_y * phonons.Q_y;
    double V_E = 0.5 * phonon_params.omega_E * phonon_params.omega_E * Q_E_sq;
    
    // A1 potential: (1/2)ω_A² Q_R²
    double V_A = 0.5 * phonon_params.omega_A * phonon_params.omega_A * phonons.Q_R * phonons.Q_R;
    
    // Three-phonon coupling: g3*(Qx² + Qy²)*Q_R
    double V_3ph = phonon_params.g3 * Q_E_sq * phonons.Q_R;
    
    return T + V_E + V_A + V_3ph;
}

double PhononLattice::spin_phonon_energy() const {
    // H_sp-ph = Σ_<ij> [λ_xy * Qx * (Si_x*Sj_z + Si_z*Sj_x)
    //                 + λ_xy * Qy * (Si_y*Sj_z + Si_z*Sj_y)
    //                 + λ_R  * Q_R * (Si · Sj)]
    
    double E = 0.0;
    double Qx = phonons.Q_x;
    double Qy = phonons.Q_y;
    double QR = phonons.Q_R;
    double lxy = spin_phonon_params.lambda_xy;
    double lR = spin_phonon_params.lambda_R;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {  // Avoid double counting
                const Eigen::Vector3d& Sj = spins[j];
                
                // Qx * (Si_x*Sj_z + Si_z*Sj_x)
                double xz_term = Si(0)*Sj(2) + Si(2)*Sj(0);
                E += lxy * Qx * xz_term;
                
                // Qy * (Si_y*Sj_z + Si_z*Sj_y)
                double yz_term = Si(1)*Sj(2) + Si(2)*Sj(1);
                E += lxy * Qy * yz_term;
                
                // Q_R * (Si · Sj)
                E += lR * QR * Si.dot(Sj);
            }
        }
    }
    
    return E;
}

// ============================================================
// DERIVATIVES
// ============================================================

double PhononLattice::dH_dQx() const {
    // ∂H_sp-ph/∂Qx = Σ_<ij> λ_xy * (Si_x*Sj_z + Si_z*Sj_x)
    double deriv = 0.0;
    double lxy = spin_phonon_params.lambda_xy;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                deriv += lxy * (Si(0)*Sj(2) + Si(2)*Sj(0));
            }
        }
    }
    
    return deriv;
}

double PhononLattice::dH_dQy() const {
    // ∂H_sp-ph/∂Qy = Σ_<ij> λ_xy * (Si_y*Sj_z + Si_z*Sj_y)
    double deriv = 0.0;
    double lxy = spin_phonon_params.lambda_xy;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                deriv += lxy * (Si(1)*Sj(2) + Si(2)*Sj(1));
            }
        }
    }
    
    return deriv;
}

double PhononLattice::dH_dQR() const {
    // ∂H_sp-ph/∂Q_R = Σ_<ij> λ_R * (Si · Sj)
    double deriv = 0.0;
    double lR = spin_phonon_params.lambda_R;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                deriv += lR * Si.dot(Sj);
            }
        }
    }
    
    return deriv;
}

SpinVector PhononLattice::get_local_field(size_t site) const {
    // H_eff = -∂H/∂Si = B - ∂H_spin/∂Si - ∂H_sp-ph/∂Si
    
    Eigen::Vector3d H = field[site];  // External field
    
    double Qx = phonons.Q_x;
    double Qy = phonons.Q_y;
    double QR = phonons.Q_R;
    double lxy = spin_phonon_params.lambda_xy;
    double lR = spin_phonon_params.lambda_R;
    
    // NN contributions
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        const Eigen::Vector3d& Sj = spins[j];
        
        // Pure spin contribution: -J · Sj
        H -= nn_interaction[site][n] * Sj;
        
        // Spin-phonon contribution from Qx*(Si_x*Sj_z + Si_z*Sj_x)
        // ∂/∂Si_x: λ_xy * Qx * Sj_z
        // ∂/∂Si_z: λ_xy * Qx * Sj_x
        H(0) -= lxy * Qx * Sj(2);
        H(2) -= lxy * Qx * Sj(0);
        
        // Spin-phonon contribution from Qy*(Si_y*Sj_z + Si_z*Sj_y)
        H(1) -= lxy * Qy * Sj(2);
        H(2) -= lxy * Qy * Sj(1);
        
        // Spin-phonon contribution from Q_R*(Si · Sj)
        // ∂/∂Si: λ_R * Q_R * Sj
        H -= lR * QR * Sj;
    }
    
    // 3rd NN contributions
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

void PhononLattice::phonon_derivatives(
    const PhononState& ph, double t,
    double dH_dQx_val, double dH_dQy_val, double dH_dQR_val,
    double& dQx, double& dQy, double& dQR,
    double& dVx, double& dVy, double& dVR) const 
{
    // Get THz drive field
    double Ex, Ey;
    drive_params.E_field(t, Ex, Ey);
    
    // E1 mode (Qx): d²Qx/dt² = -ω_E²Qx - 2g3*Qx*Q_R - γ_E*Vx - ∂H_sp-ph/∂Qx + Z*Ex
    dQx = ph.V_x;
    dVx = -phonon_params.omega_E * phonon_params.omega_E * ph.Q_x
          - 2.0 * phonon_params.g3 * ph.Q_x * ph.Q_R
          - phonon_params.gamma_E * ph.V_x
          - dH_dQx_val
          + phonon_params.Z_star * Ex;
    
    // E1 mode (Qy): d²Qy/dt² = -ω_E²Qy - 2g3*Qy*Q_R - γ_E*Vy - ∂H_sp-ph/∂Qy + Z*Ey
    dQy = ph.V_y;
    dVy = -phonon_params.omega_E * phonon_params.omega_E * ph.Q_y
          - 2.0 * phonon_params.g3 * ph.Q_y * ph.Q_R
          - phonon_params.gamma_E * ph.V_y
          - dH_dQy_val
          + phonon_params.Z_star * Ey;
    
    // A1 mode (Q_R): d²Q_R/dt² = -ω_A²Q_R - g3*(Qx²+Qy²) - γ_A*V_R - ∂H_sp-ph/∂Q_R
    double Q_E_sq = ph.Q_x * ph.Q_x + ph.Q_y * ph.Q_y;
    dQR = ph.V_R;
    dVR = -phonon_params.omega_A * phonon_params.omega_A * ph.Q_R
          - phonon_params.g3 * Q_E_sq
          - phonon_params.gamma_A * ph.V_R
          - dH_dQR_val;
}

void PhononLattice::ode_system(const ODEState& x, ODEState& dxdt, double t) {
    // State: [S0_x, S0_y, S0_z, ..., SN_z, Qx, Qy, Q_R, Vx, Vy, V_R]
    
    const size_t spin_offset = spin_dim * lattice_size;
    
    // Extract phonon state
    PhononState ph;
    ph.from_array(&x[spin_offset]);
    
    double Qx = ph.Q_x;
    double Qy = ph.Q_y;
    double QR = ph.Q_R;
    double lxy = spin_phonon_params.lambda_xy;
    double lR = spin_phonon_params.lambda_R;
    
    // Compute spin-phonon derivatives for phonon EOM
    // Note: These are ONLY the spin-dependent parts
    double dHsp_dQx = 0.0;
    double dHsp_dQy = 0.0;
    double dHsp_dQR = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const size_t idx = i * spin_dim;
        Eigen::Vector3d Si(x[idx], x[idx+1], x[idx+2]);
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const size_t jdx = j * spin_dim;
                Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
                
                dHsp_dQx += lxy * (Si(0)*Sj(2) + Si(2)*Sj(0));
                dHsp_dQy += lxy * (Si(1)*Sj(2) + Si(2)*Sj(1));
                dHsp_dQR += lR * Si.dot(Sj);
            }
        }
    }
    
    // =============================================
    // Spin equations: dS/dt = S × H_eff + α S × (S × H_eff)
    // =============================================
    for (size_t i = 0; i < lattice_size; ++i) {
        const size_t idx = i * spin_dim;
        Eigen::Vector3d Si(x[idx], x[idx+1], x[idx+2]);
        
        // Compute local effective field
        Eigen::Vector3d H = field[i];
        
        // NN interactions + spin-phonon
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            const size_t jdx = j * spin_dim;
            Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
            
            // Pure spin: -J · Sj
            H -= nn_interaction[i][n] * Sj;
            
            // Spin-phonon coupling
            H(0) -= lxy * Qx * Sj(2);
            H(2) -= lxy * Qx * Sj(0);
            H(1) -= lxy * Qy * Sj(2);
            H(2) -= lxy * Qy * Sj(1);
            H -= lR * QR * Sj;
        }
        
        // 3rd NN interactions
        for (size_t n = 0; n < j3_partners[i].size(); ++n) {
            size_t j = j3_partners[i][n];
            const size_t jdx = j * spin_dim;
            Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
            H -= j3_interaction[i][n] * Sj;
        }
        
        // LLG equation
        Eigen::Vector3d dSdt = Si.cross(H);
        if (alpha_gilbert > 0) {
            dSdt += alpha_gilbert * Si.cross(Si.cross(H)) / spin_length;
        }
        
        dxdt[idx] = dSdt(0);
        dxdt[idx+1] = dSdt(1);
        dxdt[idx+2] = dSdt(2);
    }
    
    // =============================================
    // Phonon equations
    // =============================================
    double dQx, dQy, dQR, dVx, dVy, dVR;
    phonon_derivatives(ph, t, dHsp_dQx, dHsp_dQy, dHsp_dQR,
                       dQx, dQy, dQR, dVx, dVy, dVR);
    
    dxdt[spin_offset]   = dQx;
    dxdt[spin_offset+1] = dQy;
    dxdt[spin_offset+2] = dQR;
    dxdt[spin_offset+3] = dVx;
    dxdt[spin_offset+4] = dVy;
    dxdt[spin_offset+5] = dVR;
}

// ============================================================
// ODE INTEGRATION
// ============================================================

template<typename System, typename Observer>
void PhononLattice::integrate_ode_system(
    System system_func, ODEState& state,
    double T_start, double T_end, double dt_step,
    Observer observer, const string& method,
    bool use_adaptive, double abs_tol, double rel_tol) 
{
    namespace odeint = boost::numeric::odeint;
    
    if (method == "euler") {
        odeint::integrate_const(
            odeint::euler<ODEState>(),
            system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "rk2" || method == "midpoint") {
        odeint::integrate_const(
            odeint::modified_midpoint<ODEState>(),
            system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "rk4") {
        odeint::integrate_const(
            odeint::runge_kutta4<ODEState>(),
            system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "dopri5") {
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        } else {
            odeint::integrate_const(
                odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        }
    } else if (method == "rk78" || method == "rkf78") {
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_controlled<odeint::runge_kutta_fehlberg78<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        } else {
            odeint::integrate_const(
                odeint::make_controlled<odeint::runge_kutta_fehlberg78<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        }
    } else if (method == "bulirsch_stoer" || method == "bs") {
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::bulirsch_stoer<ODEState>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        } else {
            odeint::integrate_const(
                odeint::bulirsch_stoer<ODEState>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        }
    } else {
        cout << "Warning: Unknown method '" << method << "', using dopri5" << endl;
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        } else {
            odeint::integrate_const(
                odeint::make_controlled<odeint::runge_kutta_dopri5<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        }
    }
}

// ============================================================
// MOLECULAR DYNAMICS
// ============================================================

void PhononLattice::molecular_dynamics(
    double T_start, double T_end, double dt_initial,
    string out_dir, size_t save_interval, string method) 
{
    if (!out_dir.empty()) {
        std::filesystem::create_directories(out_dir);
    }
    
    cout << "Running PhononLattice spin-phonon dynamics: t=" << T_start << " → " << T_end << endl;
    cout << "Integration method: " << method << endl;
    cout << "Initial step size: " << dt_initial << endl;
    
    // Convert to flat state
    ODEState state = to_state();
    
#ifdef HDF5_ENABLED
    std::unique_ptr<H5::H5File> h5file;
    std::unique_ptr<H5::Group> traj_group;
    std::unique_ptr<H5::Group> meta_group;
    
    vector<double> times, energies;
    vector<double> Mx, My, Mz;
    vector<double> Qx_traj, Qy_traj, QR_traj;
    
    if (!out_dir.empty()) {
        string hdf5_file = out_dir + "/trajectory.h5";
        cout << "Writing trajectory to: " << hdf5_file << endl;
        h5file = std::make_unique<H5::H5File>(hdf5_file, H5F_ACC_TRUNC);
        traj_group = std::make_unique<H5::Group>(h5file->createGroup("/trajectory"));
        meta_group = std::make_unique<H5::Group>(h5file->createGroup("/metadata"));
        
        // Write metadata
        H5::DataSpace scalar_space(H5S_SCALAR);
        auto write_scalar = [&](const string& name, double val) {
            H5::Attribute attr = meta_group->createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar_space);
            attr.write(H5::PredType::NATIVE_DOUBLE, &val);
        };
        write_scalar("omega_E", phonon_params.omega_E);
        write_scalar("omega_A", phonon_params.omega_A);
        write_scalar("g3", phonon_params.g3);
        write_scalar("lambda_xy", spin_phonon_params.lambda_xy);
        write_scalar("lambda_R", spin_phonon_params.lambda_R);
        write_scalar("dt", dt_initial);
    }
#endif
    
    size_t step_count = 0;
    size_t save_count = 0;
    
    auto observer = [&](const ODEState& x, double t) {
        if (step_count % save_interval == 0) {
            const_cast<PhononLattice*>(this)->from_state(x);
            
            Eigen::Vector3d M = magnetization();
            double E = energy_density();
            
#ifdef HDF5_ENABLED
            if (h5file) {
                times.push_back(t);
                energies.push_back(E);
                Mx.push_back(M(0)); My.push_back(M(1)); Mz.push_back(M(2));
                Qx_traj.push_back(phonons.Q_x);
                Qy_traj.push_back(phonons.Q_y);
                QR_traj.push_back(phonons.Q_R);
            }
#endif
            
            if (step_count % (save_interval * 10) == 0) {
                cout << "t=" << t << ", E=" << E 
                     << ", |M|=" << M.norm()
                     << ", |Q_E|=" << E1_amplitude()
                     << ", Q_R=" << phonons.Q_R << endl;
            }
            
            save_count++;
        }
        step_count++;
    };
    
    auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
        this->ode_system(x, dxdt, t);
    };
    
    integrate_ode_system(system_func, state, T_start, T_end, dt_initial,
                        observer, method, true, 1e-6, 1e-6);
    
    from_state(state);
    
#ifdef HDF5_ENABLED
    if (h5file && !times.empty()) {
        hsize_t dims[1] = {times.size()};
        H5::DataSpace dataspace(1, dims);
        
        auto write_dataset = [&](const string& name, const vector<double>& data) {
            H5::DataSet ds = traj_group->createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
            ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };
        
        write_dataset("time", times);
        write_dataset("energy", energies);
        write_dataset("Mx", Mx);
        write_dataset("My", My);
        write_dataset("Mz", Mz);
        write_dataset("Qx", Qx_traj);
        write_dataset("Qy", Qy_traj);
        write_dataset("QR", QR_traj);
        
        traj_group->close();
        meta_group->close();
        h5file->close();
        cout << "HDF5 trajectory saved with " << save_count << " snapshots" << endl;
    }
#endif
    
    cout << "Dynamics complete! (" << step_count << " steps, " << save_count << " saved)" << endl;
}

// ============================================================
// SIMULATED ANNEALING
// ============================================================

void PhononLattice::simulated_annealing(
    double T_start, double T_end, size_t n_steps,
    size_t overrelax_rate, double cooling_rate,
    string out_dir, bool save_observables) 
{
    cout << "Running PhononLattice simulated annealing (spin subsystem only)..." << endl;
    cout << "T: " << T_start << " → " << T_end << ", steps: " << n_steps << endl;
    
    // Keep phonons at equilibrium
    phonons = PhononState();
    
    if (!out_dir.empty()) {
        std::filesystem::create_directories(out_dir);
    }
    
#ifdef HDF5_ENABLED
    std::unique_ptr<H5::H5File> h5file;
    vector<double> steps_data, temps_data, energies_data, acc_rates_data;
    
    if (save_observables && !out_dir.empty()) {
        h5file = std::make_unique<H5::H5File>(out_dir + "/annealing.h5", H5F_ACC_TRUNC);
    }
#endif
    
    double T = T_start;
    size_t accepted = 0;
    size_t total_moves = 0;
    
    for (size_t step = 0; step < n_steps; ++step) {
        // Metropolis sweep
        for (size_t i = 0; i < lattice_size; ++i) {
            Eigen::Vector3d old_spin = spins[i];
            
            // Compute old local energy
            double E_old = -old_spin.dot(field[i]);
            for (size_t n = 0; n < nn_partners[i].size(); ++n) {
                size_t j = nn_partners[i][n];
                E_old += old_spin.dot(nn_interaction[i][n] * spins[j]);
            }
            for (size_t n = 0; n < j3_partners[i].size(); ++n) {
                size_t j = j3_partners[i][n];
                E_old += old_spin.dot(j3_interaction[i][n] * spins[j]);
            }
            
            // Propose new spin
            Eigen::Vector3d new_spin = gen_random_spin();
            
            // Compute new local energy
            double E_new = -new_spin.dot(field[i]);
            for (size_t n = 0; n < nn_partners[i].size(); ++n) {
                size_t j = nn_partners[i][n];
                E_new += new_spin.dot(nn_interaction[i][n] * spins[j]);
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
                Eigen::Vector3d H = get_local_field(i);
                if (H.norm() > 1e-10) {
                    H.normalize();
                    spins[i] = 2.0 * H.dot(spins[i]) * H - spins[i];
                    spins[i] = spins[i].normalized() * spin_length;
                }
            }
        }
        
#ifdef HDF5_ENABLED
        if (save_observables && step % 100 == 0 && h5file) {
            steps_data.push_back(static_cast<double>(step));
            temps_data.push_back(T);
            energies_data.push_back(energy_density());
            acc_rates_data.push_back(double(accepted) / total_moves);
        }
#endif
        
        // Cool
        T *= cooling_rate;
        if (T < T_end) T = T_end;
    }
    
#ifdef HDF5_ENABLED
    if (h5file && !steps_data.empty()) {
        H5::Group ann_group = h5file->createGroup("/annealing");
        hsize_t dims[1] = {steps_data.size()};
        H5::DataSpace dataspace(1, dims);
        
        H5::DataSet ds = ann_group.createDataSet("steps", H5::PredType::NATIVE_DOUBLE, dataspace);
        ds.write(steps_data.data(), H5::PredType::NATIVE_DOUBLE);
        ds = ann_group.createDataSet("temperature", H5::PredType::NATIVE_DOUBLE, dataspace);
        ds.write(temps_data.data(), H5::PredType::NATIVE_DOUBLE);
        ds = ann_group.createDataSet("energy", H5::PredType::NATIVE_DOUBLE, dataspace);
        ds.write(energies_data.data(), H5::PredType::NATIVE_DOUBLE);
        ds = ann_group.createDataSet("acceptance_rate", H5::PredType::NATIVE_DOUBLE, dataspace);
        ds.write(acc_rates_data.data(), H5::PredType::NATIVE_DOUBLE);
        
        h5file->close();
        cout << "Annealing data saved to " << out_dir << "/annealing.h5" << endl;
    }
#endif
    
    cout << "Simulated annealing complete! Final E=" << energy_density() << endl;
}

// ============================================================
// I/O
// ============================================================

#ifdef HDF5_ENABLED
void PhononLattice::save_spin_config_hdf5(const string& filename) const {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    
    hsize_t dims[2] = {lattice_size, 3};
    H5::DataSpace dataspace(2, dims);
    H5::DataSet dataset = file.createDataSet("spins", H5::PredType::NATIVE_DOUBLE, dataspace);
    
    vector<double> spin_data(lattice_size * 3);
    for (size_t i = 0; i < lattice_size; ++i) {
        spin_data[i*3 + 0] = spins[i](0);
        spin_data[i*3 + 1] = spins[i](1);
        spin_data[i*3 + 2] = spins[i](2);
    }
    dataset.write(spin_data.data(), H5::PredType::NATIVE_DOUBLE);
    
    file.close();
}

void PhononLattice::load_spin_config_hdf5(const string& filename) {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet("spins");
    
    vector<double> spin_data(lattice_size * 3);
    dataset.read(spin_data.data(), H5::PredType::NATIVE_DOUBLE);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        spins[i](0) = spin_data[i*3 + 0];
        spins[i](1) = spin_data[i*3 + 1];
        spins[i](2) = spin_data[i*3 + 2];
        spins[i] = spins[i].normalized() * spin_length;
    }
    
    file.close();
}

void PhononLattice::save_state_hdf5(const string& filename) const {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    
    H5::Group spin_group = file.createGroup("/spins");
    H5::Group phonon_group = file.createGroup("/phonons");
    
    // Save spins
    {
        hsize_t dims[2] = {lattice_size, 3};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = spin_group.createDataSet("configuration", H5::PredType::NATIVE_DOUBLE, dataspace);
        
        vector<double> spin_data(lattice_size * 3);
        for (size_t i = 0; i < lattice_size; ++i) {
            spin_data[i*3 + 0] = spins[i](0);
            spin_data[i*3 + 1] = spins[i](1);
            spin_data[i*3 + 2] = spins[i](2);
        }
        dataset.write(spin_data.data(), H5::PredType::NATIVE_DOUBLE);
    }
    
    // Save phonon state
    {
        hsize_t dims[1] = {6};
        H5::DataSpace dataspace(1, dims);
        H5::DataSet dataset = phonon_group.createDataSet("state", H5::PredType::NATIVE_DOUBLE, dataspace);
        
        double ph_data[6];
        phonons.to_array(ph_data);
        dataset.write(ph_data, H5::PredType::NATIVE_DOUBLE);
    }
    
    file.close();
}

void PhononLattice::load_state_hdf5(const string& filename) {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    
    // Load spins
    {
        H5::DataSet dataset = file.openDataSet("/spins/configuration");
        vector<double> spin_data(lattice_size * 3);
        dataset.read(spin_data.data(), H5::PredType::NATIVE_DOUBLE);
        
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i](0) = spin_data[i*3 + 0];
            spins[i](1) = spin_data[i*3 + 1];
            spins[i](2) = spin_data[i*3 + 2];
            spins[i] = spins[i].normalized() * spin_length;
        }
    }
    
    // Load phonon state
    {
        H5::DataSet dataset = file.openDataSet("/phonons/state");
        double ph_data[6];
        dataset.read(ph_data, H5::PredType::NATIVE_DOUBLE);
        phonons.from_array(ph_data);
    }
    
    file.close();
}
#endif

void PhononLattice::save_spin_config(const string& filename) const {
    std::ofstream file(filename);
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        file << spins[i](0) << " " << spins[i](1) << " " << spins[i](2) << "\n";
    }
    
    file.close();
}

void PhononLattice::load_spin_config(const string& filename) {
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

void PhononLattice::save_positions(const string& filename) const {
    std::ofstream file(filename);
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        file << site_positions[i](0) << " " 
             << site_positions[i](1) << " " 
             << site_positions[i](2) << "\n";
    }
    
    file.close();
}

// Explicit template instantiation
template void PhononLattice::integrate_ode_system(
    std::function<void(const PhononLattice::ODEState&, PhononLattice::ODEState&, double)>,
    PhononLattice::ODEState&, double, double, double,
    std::function<void(const PhononLattice::ODEState&, double)>,
    const std::string&, bool, double, double);
