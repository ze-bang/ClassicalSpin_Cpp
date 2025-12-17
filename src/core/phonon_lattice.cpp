/**
 * @file phonon_lattice.cpp
 * @brief Implementation of spin-phonon coupled honeycomb lattice
 * 
 * Implements honeycomb lattice with:
 * - Generic NN, 2nd NN, and 3rd NN spin interactions
 * - Three phonon modes: E1 (Qx_E1, Qy_E1), E2 (Qx_E2, Qy_E2), A1 (Q_A1)
 * - Three-phonon coupling: g3_E1A1*(Qx_E1² + Qy_E1²)*Q_A1 + g3_E2A1*(Qx_E2² + Qy_E2²)*Q_A1
 * - E1-E2 bilinear coupling: g3_E1E2*(Qx_E1*Qx_E2 + Qy_E1*Qy_E2)
 * - Spin-phonon E1 (bond-dependent, like Γ'):
 *     x-bond: λ_E1 * [Qx_E1*(SxSy+SySx) + Qy_E1*(SxSz+SzSx)]
 *     y-bond: λ_E1 * [Qx_E1*(SySz+SzSy) + Qy_E1*(SxSy+SySx)]
 *     z-bond: λ_E1 * [Qx_E1*(SxSz+SzSx) + Qy_E1*(SySz+SzSy)]
 * - Spin-phonon E2 (bond-dependent, like Γ+η):
 *     x-bond: λ_E2 * [Qx_E2*(SySy-SzSz) + Qy_E2*(SySz+SzSy)]
 *     y-bond: λ_E2 * [Qx_E2*(SzSz-SxSx) + Qy_E2*(SxSz+SzSx)]
 *     z-bond: λ_E2 * [Qx_E2*(SxSx-SySy) + Qy_E2*(SxSy+SySx)]
 * - Spin-phonon A1:  λ_A1 * Q_A1*(Si·Sj)
 * 
 * COORDINATE FRAME:
 * The spin Hamiltonian (Kitaev-Heisenberg-Γ-Γ') exchange matrices are defined in
 * the local Kitaev frame and then transformed to the global Cartesian frame using:
 *   J_global = R * J_local * R^T
 * 
 * where R is the Kitaev local-to-global rotation matrix with columns:
 *   x' = (1, 1, -2)/√6,  y' = (-1, 1, 0)/√2,  z' = (1, 1, 1)/√3
 * 
 * Spins are stored and evolved in the GLOBAL Cartesian frame.
 */

#include "classical_spin/lattice/phonon_lattice.h"
#include <fstream>
#include <iomanip>
#include <memory>
#include <filesystem>
#include <mpi.h>

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
    j2_interaction.resize(lattice_size);
    j2_partners.resize(lattice_size);
    j3_interaction.resize(lattice_size);
    j3_partners.resize(lattice_size);
    
    // Initialize Kitaev local frame for honeycomb lattice
    // The Kitaev local frame transforms from local coordinates to the global cubic frame
    // Local basis: x' = (1,1,-2)/√6, y' = (-1,1,0)/√2, z' = (1,1,1)/√3
    // This is the same for both sublattices in the honeycomb lattice
    // S_global = R * S_local where columns of R are the local basis vectors
    SpinMatrix kitaev_frame(3, 3);
    kitaev_frame << 1.0/std::sqrt(6.0), -1.0/std::sqrt(2.0), 1.0/std::sqrt(3.0),
                    1.0/std::sqrt(6.0),  1.0/std::sqrt(2.0), 1.0/std::sqrt(3.0),
                   -2.0/std::sqrt(6.0),  0.0,                1.0/std::sqrt(3.0);
    
    // Both sublattices use the same local frame
    sublattice_frames[0] = kitaev_frame;
    sublattice_frames[1] = kitaev_frame;
    
    // Initialize RNG
    seed_lehman(std::chrono::system_clock::now().time_since_epoch().count() * 2 + 1);
    
    cout << "Initializing PhononLattice with dimensions: " 
         << dim1 << " x " << dim2 << " x " << dim3 << endl;
    cout << "Total spin sites: " << lattice_size << endl;
    cout << "Phonon DOF: " << PhononState::N_DOF << " (Qx, Qy, Q_R, Vx, Vy, V_R)" << endl;
    cout << "Total ODE state size: " << state_size << endl;
    cout << "Kitaev local frame initialized (same for both sublattices)" << endl;
    
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
        j2_interaction[i].clear();
        j2_partners[i].clear();
        j3_interaction[i].clear();
        j3_partners[i].clear();
    }
    
    // Build bond-dependent Kitaev-Heisenberg-Γ-Γ' exchange matrices
    SpinMatrix Jx = sp_params.get_Jx();
    SpinMatrix Jy = sp_params.get_Jy();
    SpinMatrix Jz = sp_params.get_Jz();
    SpinMatrix J2_A_mat = sp_params.get_J2_A_matrix();
    SpinMatrix J2_B_mat = sp_params.get_J2_B_matrix();
    SpinMatrix J3_mat = sp_params.get_J3_matrix();
    
    // Build NN interactions on honeycomb
    // Honeycomb lattice structure:
    //   - Lattice vectors: a1 = (1, 0, 0), a2 = (0.5, √3/2, 0)
    //   - Sublattice A at (0, 0, 0), Sublattice B at (0, 1/√3, 0)
    //   - NN distance: 1/√3 ≈ 0.577  (3 neighbors, A↔B)
    //   - 2nd NN distance: 1.0        (6 neighbors, A↔A, B↔B)  
    //   - 3rd NN distance: 2/√3 ≈ 1.155 (3 neighbors, A↔B)
    //
    // NN bonds (Kitaev bond types):
    //   - z-bond (type 2): A(i,j,k) → B(i,j,k)     [same unit cell]
    //   - x-bond (type 0): A(i,j,k) → B(i,j-1,k)   [offset (0,-1,0)]
    //   - y-bond (type 1): A(i,j,k) → B(i+1,j-1,k) [offset (1,-1,0)]
    
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                size_t site0 = flatten_index(i, j, k, 0);  // Sublattice A
                size_t site1 = flatten_index(i, j, k, 1);  // Sublattice B
                
                // x-bond (use Jx matrix)
                size_t partner_x = flatten_index_periodic(i, j-1, k, 1);
                nn_interaction[site0].push_back(Jx);
                nn_partners[site0].push_back(partner_x);
                nn_bond_types[site0].push_back(0);
                // Reverse bond
                nn_interaction[partner_x].push_back(Jx.transpose());
                nn_partners[partner_x].push_back(site0);
                nn_bond_types[partner_x].push_back(0);
                
                // y-bond (use Jy matrix)
                size_t partner_y = flatten_index_periodic(i+1, j-1, k, 1);
                nn_interaction[site0].push_back(Jy);
                nn_partners[site0].push_back(partner_y);
                nn_bond_types[site0].push_back(1);
                // Reverse bond
                nn_interaction[partner_y].push_back(Jy.transpose());
                nn_partners[partner_y].push_back(site0);
                nn_bond_types[partner_y].push_back(1);
                
                // z-bond (use Jz matrix, same unit cell)
                nn_interaction[site0].push_back(Jz);
                nn_partners[site0].push_back(site1);
                nn_bond_types[site0].push_back(2);
                // Reverse bond
                nn_interaction[site1].push_back(Jz.transpose());
                nn_partners[site1].push_back(site0);
                nn_bond_types[site1].push_back(2);
                
                // 2nd NN interactions (isotropic Heisenberg, sublattice-dependent)
                // On honeycomb, 2nd NN connect same sublattice at distance sqrt(3)*a
                // 2nd NN offsets: (±1, 0), (0, ±1), (±1, ∓1) in lattice coordinates
                // These connect A-A and B-B sites with different couplings J2_A and J2_B
                if (std::abs(sp_params.J2_A) > 1e-12 || std::abs(sp_params.J2_B) > 1e-12) {
                    // 2nd NN offset vectors (same for both sublattices in lattice coords)
                    vector<std::tuple<int,int,int>> j2_offsets = {
                        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {1, -1, 0}, {-1, 1, 0}
                    };
                    
                    // 2nd NN for sublattice A (site0) with coupling J2_A
                    for (const auto& [di, dj, dk] : j2_offsets) {
                        size_t partner_j2 = flatten_index_periodic(i+di, j+dj, k+dk, 0);
                        // Only add if partner > site0 to avoid double counting
                        if (partner_j2 > site0) {
                            j2_interaction[site0].push_back(J2_A_mat);
                            j2_partners[site0].push_back(partner_j2);
                            j2_interaction[partner_j2].push_back(J2_A_mat.transpose());
                            j2_partners[partner_j2].push_back(site0);
                        }
                    }
                    
                    // 2nd NN for sublattice B (site1) with coupling J2_B
                    for (const auto& [di, dj, dk] : j2_offsets) {
                        size_t partner_j2 = flatten_index_periodic(i+di, j+dj, k+dk, 1);
                        if (partner_j2 > site1) {
                            j2_interaction[site1].push_back(J2_B_mat);
                            j2_partners[site1].push_back(partner_j2);
                            j2_interaction[partner_j2].push_back(J2_B_mat.transpose());
                            j2_partners[partner_j2].push_back(site1);
                        }
                    }
                }
                
                // 3rd NN interactions (isotropic Heisenberg J3)
                // On honeycomb, 3rd NN are at distance 2/sqrt(3), connecting OPPOSITE sublattices (A↔B)
                // 3rd NN offsets from A(i,j,k,0) to B: (+1,-2,1), (-1,0,1), (+1,0,1)
                // 3rd NN offsets from B(i,j,k,1) to A: (-1,+2,0), (-1,0,0), (+1,0,0)
                if (std::abs(sp_params.J3) > 1e-12) {
                    // 3rd NN from sublattice A (site0) to sublattice B
                    vector<std::tuple<int,int,int>> j3_A_to_B_offsets = {
                        {1, -2, 0}, {-1, 0, 0}, {1, 0, 0}
                    };
                    
                    for (const auto& [di, dj, dk] : j3_A_to_B_offsets) {
                        size_t partner_j3 = flatten_index_periodic(i+di, j+dj, k+dk, 1);  // Connect to sublattice B
                        // Only add if partner > site0 to avoid double counting
                        if (partner_j3 > site0) {
                            j3_interaction[site0].push_back(J3_mat);
                            j3_partners[site0].push_back(partner_j3);
                            j3_interaction[partner_j3].push_back(J3_mat.transpose());
                            j3_partners[partner_j3].push_back(site0);
                        }
                    }
                    
                    // 3rd NN from sublattice B (site1) to sublattice A
                    vector<std::tuple<int,int,int>> j3_B_to_A_offsets = {
                        {-1, 2, 0}, {-1, 0, 0}, {1, 0, 0}
                    };
                    
                    for (const auto& [di, dj, dk] : j3_B_to_A_offsets) {
                        size_t partner_j3 = flatten_index_periodic(i+di, j+dj, k+dk, 0);  // Connect to sublattice A
                        if (partner_j3 > site1) {
                            j3_interaction[site1].push_back(J3_mat);
                            j3_partners[site1].push_back(partner_j3);
                            j3_interaction[partner_j3].push_back(J3_mat.transpose());
                            j3_partners[partner_j3].push_back(site1);
                        }
                    }
                }
            }
        }
    }
    
    cout << "Set PhononLattice parameters (Kitaev-Heisenberg-Γ-Γ' in GLOBAL Cartesian frame):" << endl;
    cout << "  Exchange matrices transformed: J_global = R * J_local * R^T" << endl;
    cout << "  Local frame basis: x'=(1,1,-2)/√6, y'=(-1,1,0)/√2, z'=(1,1,1)/√3" << endl;
    cout << "  J=" << sp_params.J << ", K=" << sp_params.K 
         << ", Γ=" << sp_params.Gamma << ", Γ'=" << sp_params.Gammap << endl;
    cout << "  J2_A=" << sp_params.J2_A << ", J2_B=" << sp_params.J2_B << endl;
    cout << "  J3=" << sp_params.J3 << endl;
    cout << "  Spin-phonon: λ_E1=" << sp_params.lambda_E1 
         << ", λ_E2=" << sp_params.lambda_E2 
         << ", λ_A1=" << sp_params.lambda_A1 << endl;
    cout << "  E1 mode: ω_E1=" << ph_params.omega_E1 << ", γ_E1=" << ph_params.gamma_E1
         << ", λ_E1(quartic)=" << ph_params.lambda_E1 << endl;
    cout << "  E2 mode: ω_E2=" << ph_params.omega_E2 << ", γ_E2=" << ph_params.gamma_E2
         << ", λ_E2(quartic)=" << ph_params.lambda_E2 << endl;
    cout << "  A1 mode: ω_A1=" << ph_params.omega_A1 << ", γ_A1=" << ph_params.gamma_A1
         << ", λ_A1(quartic)=" << ph_params.lambda_A1 << endl;
    cout << "  3-phonon: g3_E1A1=" << ph_params.g3_E1A1 << ", g3_E2A1=" << ph_params.g3_E2A1 << endl;
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
        
        // 2nd NN interactions
        for (size_t n = 0; n < j2_partners[i].size(); ++n) {
            size_t j = j2_partners[i][n];
            if (j > i) {  // Avoid double counting
                const Eigen::Vector3d& Sj = spins[j];
                E += Si.dot(j2_interaction[i][n] * Sj);
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
    // Kinetic energy for all modes
    double T = phonons.kinetic_energy();
    
    // E1 mode: (1/2)ω_E1²(Qx_E1² + Qy_E1²) + (λ_E1/4)(Qx_E1² + Qy_E1²)²
    double Q_E1_sq = phonons.Q_x_E1 * phonons.Q_x_E1 + phonons.Q_y_E1 * phonons.Q_y_E1;
    double V_E1 = 0.5 * phonon_params.omega_E1 * phonon_params.omega_E1 * Q_E1_sq;
    double V_4E1 = 0.25 * phonon_params.lambda_E1 * Q_E1_sq * Q_E1_sq;
    
    // E2 mode: (1/2)ω_E2²(Qx_E2² + Qy_E2²) + (λ_E2/4)(Qx_E2² + Qy_E2²)²
    double Q_E2_sq = phonons.Q_x_E2 * phonons.Q_x_E2 + phonons.Q_y_E2 * phonons.Q_y_E2;
    double V_E2 = 0.5 * phonon_params.omega_E2 * phonon_params.omega_E2 * Q_E2_sq;
    double V_4E2 = 0.25 * phonon_params.lambda_E2 * Q_E2_sq * Q_E2_sq;
    
    // A1 mode: (1/2)ω_A1² Q_A1² + (λ_A1/4)*Q_A1⁴
    double Q_A1_sq = phonons.Q_A1 * phonons.Q_A1;
    double V_A1 = 0.5 * phonon_params.omega_A1 * phonon_params.omega_A1 * Q_A1_sq;
    double V_4A1 = 0.25 * phonon_params.lambda_A1 * Q_A1_sq * Q_A1_sq;
    
    // Three-phonon coupling: g3_E1A1*(Qx_E1² + Qy_E1²)*Q_A1 + g3_E2A1*(Qx_E2² + Qy_E2²)*Q_A1
    double V_3ph = phonon_params.g3_E1A1 * Q_E1_sq * phonons.Q_A1
                 + phonon_params.g3_E2A1 * Q_E2_sq * phonons.Q_A1;
    
    // E1-E2 bilinear coupling: g3_E1E2 * (Qx_E1*Qx_E2 + Qy_E1*Qy_E2)
    double V_E1E2 = phonon_params.g3_E1E2 * (phonons.Q_x_E1 * phonons.Q_x_E2 
                                            + phonons.Q_y_E1 * phonons.Q_y_E2);
    
    return T + V_E1 + V_4E1 + V_E2 + V_4E2 + V_A1 + V_4A1 + V_3ph + V_E1E2;
}

double PhononLattice::spin_phonon_energy() const {
    // H_sp-ph = Σ_<ij> bond-dependent coupling:
    // E1 (like Γ'):
    //   x-bond: λ_E1 * [Qx_E1*(SxSy+SySx) + Qy_E1*(SxSz+SzSx)]
    //   y-bond: λ_E1 * [Qx_E1*(SySz+SzSy) + Qy_E1*(SxSy+SySx)]
    //   z-bond: λ_E1 * [Qx_E1*(SxSz+SzSx) + Qy_E1*(SySz+SzSy)]
    // E2 (like K+Γ):
    //   x-bond: λ_E2 * [Qx_E2*(SySy-SzSz) + Qy_E2*(SySz+SzSy)]
    //   y-bond: λ_E2 * [Qx_E2*(SzSz-SxSx) + Qy_E2*(SxSz+SzSx)]
    //   z-bond: λ_E2 * [Qx_E2*(SxSx-SySy) + Qy_E2*(SxSy+SySx)]
    // A1: λ_A1 * Q_A1 * (Si · Sj)
    
    double E = 0.0;
    double Qx_E1 = phonons.Q_x_E1;
    double Qy_E1 = phonons.Q_y_E1;
    double Qx_E2 = phonons.Q_x_E2;
    double Qy_E2 = phonons.Q_y_E2;
    double Q_A1 = phonons.Q_A1;
    double l_E1 = spin_phonon_params.lambda_E1;
    double l_E2 = spin_phonon_params.lambda_E2;
    double l_A1 = spin_phonon_params.lambda_A1;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {  // Avoid double counting
                const Eigen::Vector3d& Sj = spins[j];
                int bond_type = nn_bond_types[i][n];  // 0=x, 1=y, 2=z
                
                // E1 coupling (bond-dependent, like Γ')
                if (bond_type == 0) {  // x-bond: Qx_E1*(SxSy+SySx) + Qy_E1*(SxSz+SzSx)
                    E += l_E1 * Qx_E1 * (Si(0)*Sj(1) + Si(1)*Sj(0));
                    E += l_E1 * Qy_E1 * (Si(0)*Sj(2) + Si(2)*Sj(0));
                } else if (bond_type == 1) {  // y-bond: Qx_E1*(SySz+SzSy) + Qy_E1*(SxSy+SySx)
                    E += l_E1 * Qx_E1 * (Si(1)*Sj(2) + Si(2)*Sj(1));
                    E += l_E1 * Qy_E1 * (Si(0)*Sj(1) + Si(1)*Sj(0));
                } else {  // z-bond: Qx_E1*(SxSz+SzSx) + Qy_E1*(SySz+SzSy)
                    E += l_E1 * Qx_E1 * (Si(0)*Sj(2) + Si(2)*Sj(0));
                    E += l_E1 * Qy_E1 * (Si(1)*Sj(2) + Si(2)*Sj(1));
                }
                
                // E2 coupling (bond-dependent, like Γ+η)
                if (bond_type == 0) {  // x-bond: Qx_E2*(SySy-SzSz) + Qy_E2*(SySz+SzSy)
                    E += l_E2 * Qx_E2 * (Si(1)*Sj(1) - Si(2)*Sj(2));
                    E += l_E2 * Qy_E2 * (Si(1)*Sj(2) + Si(2)*Sj(1));
                } else if (bond_type == 1) {  // y-bond: Qx_E2*(SzSz-SxSx) + Qy_E2*(SxSz+SzSx)
                    E += l_E2 * Qx_E2 * (Si(2)*Sj(2) - Si(0)*Sj(0));
                    E += l_E2 * Qy_E2 * (Si(0)*Sj(2) + Si(2)*Sj(0));
                } else {  // z-bond: Qx_E2*(SxSx-SySy) + Qy_E2*(SxSy+SySx)
                    E += l_E2 * Qx_E2 * (Si(0)*Sj(0) - Si(1)*Sj(1));
                    E += l_E2 * Qy_E2 * (Si(0)*Sj(1) + Si(1)*Sj(0));
                }
                
                // A1: Q_A1 * (Si · Sj)
                E += l_A1 * Q_A1 * Si.dot(Sj);
            }
        }
    }
    
    return E;
}

// ============================================================
// DERIVATIVES
// ============================================================

double PhononLattice::dH_dQx_E1() const {
    // ∂H_sp-ph/∂Qx_E1 with bond-dependent coupling:
    //   x-bond: λ_E1 * (SxSy+SySx)
    //   y-bond: λ_E1 * (SySz+SzSy)
    //   z-bond: λ_E1 * (SxSz+SzSx)
    double deriv = 0.0;
    double l_E1 = spin_phonon_params.lambda_E1;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int bond_type = nn_bond_types[i][n];
                
                if (bond_type == 0) {  // x-bond: (SxSy+SySx)
                    deriv += l_E1 * (Si(0)*Sj(1) + Si(1)*Sj(0));
                } else if (bond_type == 1) {  // y-bond: (SySz+SzSy)
                    deriv += l_E1 * (Si(1)*Sj(2) + Si(2)*Sj(1));
                } else {  // z-bond: (SxSz+SzSx)
                    deriv += l_E1 * (Si(0)*Sj(2) + Si(2)*Sj(0));
                }
            }
        }
    }
    
    return deriv;
}

double PhononLattice::dH_dQy_E1() const {
    // ∂H_sp-ph/∂Qy_E1 with bond-dependent coupling (orthogonal to Qx_E1):
    //   x-bond: λ_E1 * (SxSz+SzSx)
    //   y-bond: λ_E1 * (SxSy+SySx)
    //   z-bond: λ_E1 * (SySz+SzSy)
    double deriv = 0.0;
    double l_E1 = spin_phonon_params.lambda_E1;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int bond_type = nn_bond_types[i][n];
                
                if (bond_type == 0) {  // x-bond: (SxSz+SzSx)
                    deriv += l_E1 * (Si(0)*Sj(2) + Si(2)*Sj(0));
                } else if (bond_type == 1) {  // y-bond: (SxSy+SySx)
                    deriv += l_E1 * (Si(0)*Sj(1) + Si(1)*Sj(0));
                } else {  // z-bond: (SySz+SzSy)
                    deriv += l_E1 * (Si(1)*Sj(2) + Si(2)*Sj(1));
                }
            }
        }
    }
    
    return deriv;
}

double PhononLattice::dH_dQx_E2() const {
    // ∂H_sp-ph/∂Qx_E2 with bond-dependent η-like coupling:
    //   x-bond: λ_E2 * (SySy-SzSz)
    //   y-bond: λ_E2 * (SzSz-SxSx)
    //   z-bond: λ_E2 * (SxSx-SySy)
    double deriv = 0.0;
    double l_E2 = spin_phonon_params.lambda_E2;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int bond_type = nn_bond_types[i][n];
                
                if (bond_type == 0) {  // x-bond: (SySy-SzSz)
                    deriv += l_E2 * (Si(1)*Sj(1) - Si(2)*Sj(2));
                } else if (bond_type == 1) {  // y-bond: (SzSz-SxSx)
                    deriv += l_E2 * (Si(2)*Sj(2) - Si(0)*Sj(0));
                } else {  // z-bond: (SxSx-SySy)
                    deriv += l_E2 * (Si(0)*Sj(0) - Si(1)*Sj(1));
                }
            }
        }
    }
    
    return deriv;
}

double PhononLattice::dH_dQy_E2() const {
    // ∂H_sp-ph/∂Qy_E2 with bond-dependent coupling:
    //   x-bond: λ_E2 * (SySz+SzSy)
    //   y-bond: λ_E2 * (SxSz+SzSx)
    //   z-bond: λ_E2 * (SxSy+SySx)
    double deriv = 0.0;
    double l_E2 = spin_phonon_params.lambda_E2;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int bond_type = nn_bond_types[i][n];
                
                if (bond_type == 0) {  // x-bond: (SySz+SzSy)
                    deriv += l_E2 * (Si(1)*Sj(2) + Si(2)*Sj(1));
                } else if (bond_type == 1) {  // y-bond: (SxSz+SzSx)
                    deriv += l_E2 * (Si(0)*Sj(2) + Si(2)*Sj(0));
                } else {  // z-bond: (SxSy+SySx)
                    deriv += l_E2 * (Si(0)*Sj(1) + Si(1)*Sj(0));
                }
            }
        }
    }
    
    return deriv;
}

double PhononLattice::dH_dQ_A1() const {
    // ∂H_sp-ph/∂Q_A1 = Σ_<ij> λ_A1 * (Si · Sj)
    double deriv = 0.0;
    double l_A1 = spin_phonon_params.lambda_A1;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                deriv += l_A1 * Si.dot(Sj);
            }
        }
    }
    
    return deriv;
}

SpinVector PhononLattice::get_local_field(size_t site) const {
    // H_eff = -∂H/∂Si = B - ∂H_spin/∂Si - ∂H_sp-ph/∂Si
    
    Eigen::Vector3d H = field[site];  // External field
    
    double Qx_E1 = phonons.Q_x_E1;
    double Qy_E1 = phonons.Q_y_E1;
    double Qx_E2 = phonons.Q_x_E2;
    double Qy_E2 = phonons.Q_y_E2;
    double Q_A1 = phonons.Q_A1;
    double l_E1 = spin_phonon_params.lambda_E1;
    double l_E2 = spin_phonon_params.lambda_E2;
    double l_A1 = spin_phonon_params.lambda_A1;
    
    // NN contributions
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        const Eigen::Vector3d& Sj = spins[j];
        int bond_type = nn_bond_types[site][n];
        
        // Pure spin contribution: -J · Sj
        H -= nn_interaction[site][n] * Sj;
        
        // E1 spin-phonon coupling (bond-dependent, like Γ')
        if (bond_type == 0) {  // x-bond: Qx_E1*(SxSy+SySx) + Qy_E1*(SxSz+SzSx)
            H(0) -= l_E1 * Qx_E1 * Sj(1);  // ∂/∂Si_x of Si_x*Sj_y
            H(1) -= l_E1 * Qx_E1 * Sj(0);  // ∂/∂Si_y of Si_y*Sj_x
            H(0) -= l_E1 * Qy_E1 * Sj(2);  // ∂/∂Si_x of Si_x*Sj_z
            H(2) -= l_E1 * Qy_E1 * Sj(0);  // ∂/∂Si_z of Si_z*Sj_x
        } else if (bond_type == 1) {  // y-bond: Qx_E1*(SySz+SzSy) + Qy_E1*(SxSy+SySx)
            H(1) -= l_E1 * Qx_E1 * Sj(2);  // ∂/∂Si_y of Si_y*Sj_z
            H(2) -= l_E1 * Qx_E1 * Sj(1);  // ∂/∂Si_z of Si_z*Sj_y
            H(0) -= l_E1 * Qy_E1 * Sj(1);  // ∂/∂Si_x of Si_x*Sj_y
            H(1) -= l_E1 * Qy_E1 * Sj(0);  // ∂/∂Si_y of Si_y*Sj_x
        } else {  // z-bond: Qx_E1*(SxSz+SzSx) + Qy_E1*(SySz+SzSy)
            H(0) -= l_E1 * Qx_E1 * Sj(2);  // ∂/∂Si_x of Si_x*Sj_z
            H(2) -= l_E1 * Qx_E1 * Sj(0);  // ∂/∂Si_z of Si_z*Sj_x
            H(1) -= l_E1 * Qy_E1 * Sj(2);  // ∂/∂Si_y of Si_y*Sj_z
            H(2) -= l_E1 * Qy_E1 * Sj(1);  // ∂/∂Si_z of Si_z*Sj_y
        }
        
        // E2 spin-phonon coupling (bond-dependent, like Γ+η)
        if (bond_type == 0) {  // x-bond: Qx_E2*(SySy-SzSz) + Qy_E2*(SySz+SzSy)
            H(1) -= l_E2 * Qx_E2 * Sj(1);   // ∂/∂Si_y of Si_y*Sj_y
            H(2) -= -l_E2 * Qx_E2 * Sj(2);  // ∂/∂Si_z of -Si_z*Sj_z
            H(1) -= l_E2 * Qy_E2 * Sj(2);   // ∂/∂Si_y of Si_y*Sj_z
            H(2) -= l_E2 * Qy_E2 * Sj(1);   // ∂/∂Si_z of Si_z*Sj_y
        } else if (bond_type == 1) {  // y-bond: Qx_E2*(SzSz-SxSx) + Qy_E2*(SxSz+SzSx)
            H(2) -= l_E2 * Qx_E2 * Sj(2);   // ∂/∂Si_z of Si_z*Sj_z
            H(0) -= -l_E2 * Qx_E2 * Sj(0);  // ∂/∂Si_x of -Si_x*Sj_x
            H(0) -= l_E2 * Qy_E2 * Sj(2);   // ∂/∂Si_x of Si_x*Sj_z
            H(2) -= l_E2 * Qy_E2 * Sj(0);   // ∂/∂Si_z of Si_z*Sj_x
        } else {  // z-bond: Qx_E2*(SxSx-SySy) + Qy_E2*(SxSy+SySx)
            H(0) -= l_E2 * Qx_E2 * Sj(0);   // ∂/∂Si_x of Si_x*Sj_x
            H(1) -= -l_E2 * Qx_E2 * Sj(1);  // ∂/∂Si_y of -Si_y*Sj_y
            H(0) -= l_E2 * Qy_E2 * Sj(1);   // ∂/∂Si_x of Si_x*Sj_y
            H(1) -= l_E2 * Qy_E2 * Sj(0);   // ∂/∂Si_y of Si_y*Sj_x
        }
        
        // A1 coupling: Q_A1*(Si · Sj)
        // ∂/∂Si: λ_A1 * Q_A1 * Sj
        H -= l_A1 * Q_A1 * Sj;
    }
    
    // 2nd NN contributions
    for (size_t n = 0; n < j2_partners[site].size(); ++n) {
        size_t j = j2_partners[site][n];
        const Eigen::Vector3d& Sj = spins[j];
        H -= j2_interaction[site][n] * Sj;
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
    double dH_dQx_E1_val, double dH_dQy_E1_val,
    double dH_dQx_E2_val, double dH_dQy_E2_val,
    double dH_dQ_A1_val,
    PhononState& dph_dt) const 
{
    // Get THz drive field (only E1 is IR active)
    double Ex, Ey;
    drive_params.E_field(t, Ex, Ey);
    
    // Precompute common terms
    double Q_E1_sq = ph.Q_x_E1 * ph.Q_x_E1 + ph.Q_y_E1 * ph.Q_y_E1;
    double Q_E2_sq = ph.Q_x_E2 * ph.Q_x_E2 + ph.Q_y_E2 * ph.Q_y_E2;
    double Q_A1 = ph.Q_A1;
    double Q_A1_sq = Q_A1 * Q_A1;
    
    // E1 mode (Qx_E1): THz driven
    // d²Qx_E1/dt² = -ω_E1²Qx_E1 - λ_E1*(Qx_E1²+Qy_E1²)*Qx_E1 - 2g3_E1A1*Qx_E1*Q_A1 
    //              - g3_E1E2*Qx_E2 - γ_E1*Vx_E1 - ∂H_sp-ph/∂Qx_E1 + Z*Ex
    dph_dt.Q_x_E1 = ph.V_x_E1;
    dph_dt.V_x_E1 = -phonon_params.omega_E1 * phonon_params.omega_E1 * ph.Q_x_E1
          - phonon_params.lambda_E1 * Q_E1_sq * ph.Q_x_E1
          - 2.0 * phonon_params.g3_E1A1 * ph.Q_x_E1 * Q_A1
          - phonon_params.g3_E1E2 * ph.Q_x_E2
          - phonon_params.gamma_E1 * ph.V_x_E1
          - dH_dQx_E1_val
          + phonon_params.Z_star * Ex;
    
    // E1 mode (Qy_E1): THz driven
    dph_dt.Q_y_E1 = ph.V_y_E1;
    dph_dt.V_y_E1 = -phonon_params.omega_E1 * phonon_params.omega_E1 * ph.Q_y_E1
          - phonon_params.lambda_E1 * Q_E1_sq * ph.Q_y_E1
          - 2.0 * phonon_params.g3_E1A1 * ph.Q_y_E1 * Q_A1
          - phonon_params.g3_E1E2 * ph.Q_y_E2
          - phonon_params.gamma_E1 * ph.V_y_E1
          - dH_dQy_E1_val
          + phonon_params.Z_star * Ey;
    
    // E2 mode (Qx_E2): Raman active, not THz driven
    // d²Qx_E2/dt² = -ω_E2²Qx_E2 - λ_E2*(Qx_E2²+Qy_E2²)*Qx_E2 - 2g3_E2A1*Qx_E2*Q_A1 
    //              - g3_E1E2*Qx_E1 - γ_E2*Vx_E2 - ∂H_sp-ph/∂Qx_E2
    dph_dt.Q_x_E2 = ph.V_x_E2;
    dph_dt.V_x_E2 = -phonon_params.omega_E2 * phonon_params.omega_E2 * ph.Q_x_E2
          - phonon_params.lambda_E2 * Q_E2_sq * ph.Q_x_E2
          - 2.0 * phonon_params.g3_E2A1 * ph.Q_x_E2 * Q_A1
          - phonon_params.g3_E1E2 * ph.Q_x_E1
          - phonon_params.gamma_E2 * ph.V_x_E2
          - dH_dQx_E2_val;
    
    // E2 mode (Qy_E2): Raman active, not THz driven
    dph_dt.Q_y_E2 = ph.V_y_E2;
    dph_dt.V_y_E2 = -phonon_params.omega_E2 * phonon_params.omega_E2 * ph.Q_y_E2
          - phonon_params.lambda_E2 * Q_E2_sq * ph.Q_y_E2
          - 2.0 * phonon_params.g3_E2A1 * ph.Q_y_E2 * Q_A1
          - phonon_params.g3_E1E2 * ph.Q_y_E1
          - phonon_params.gamma_E2 * ph.V_y_E2
          - dH_dQy_E2_val;
    
    // A1 mode: Raman active, not THz driven
    // d²Q_A1/dt² = -ω_A1²Q_A1 - λ_A1*Q_A1³ - g3_E1A1*(Qx_E1²+Qy_E1²) - g3_E2A1*(Qx_E2²+Qy_E2²)
    //             - γ_A1*V_A1 - ∂H_sp-ph/∂Q_A1
    dph_dt.Q_A1 = ph.V_A1;
    dph_dt.V_A1 = -phonon_params.omega_A1 * phonon_params.omega_A1 * Q_A1
          - phonon_params.lambda_A1 * Q_A1_sq * Q_A1
          - phonon_params.g3_E1A1 * Q_E1_sq
          - phonon_params.g3_E2A1 * Q_E2_sq
          - phonon_params.gamma_A1 * ph.V_A1
          - dH_dQ_A1_val;
}

void PhononLattice::ode_system(const ODEState& x, ODEState& dxdt, double t) {
    // State: [S0_x, S0_y, S0_z, ..., SN_z, Qx_E1, Qy_E1, Qx_E2, Qy_E2, Q_A1, Vx_E1, Vy_E1, Vx_E2, Vy_E2, V_A1]
    
    const size_t spin_offset = spin_dim * lattice_size;
    
    // Extract phonon state
    PhononState ph;
    ph.from_array(&x[spin_offset]);
    
    double Qx_E1 = ph.Q_x_E1;
    double Qy_E1 = ph.Q_y_E1;
    double Qx_E2 = ph.Q_x_E2;
    double Qy_E2 = ph.Q_y_E2;
    double Q_A1 = ph.Q_A1;
    
    // Get time-dependent spin-phonon coupling strengths
    double l_E1 = get_lambda_E1(t);
    double l_E2 = get_lambda_E2(t);
    double l_A1 = get_lambda_A1(t);
    
    // Compute spin-phonon derivatives for phonon EOM
    // Note: These are ONLY the spin-dependent parts
    double dHsp_dQx_E1 = 0.0;
    double dHsp_dQy_E1 = 0.0;
    double dHsp_dQx_E2 = 0.0;
    double dHsp_dQy_E2 = 0.0;
    double dHsp_dQ_A1 = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const size_t idx = i * spin_dim;
        Eigen::Vector3d Si(x[idx], x[idx+1], x[idx+2]);
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const size_t jdx = j * spin_dim;
                Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
                int bond_type = nn_bond_types[i][n];
                
                // E1 terms (bond-dependent)
                if (bond_type == 0) {  // x-bond
                    dHsp_dQx_E1 += l_E1 * (Si(0)*Sj(1) + Si(1)*Sj(0));  // (SxSy+SySx)
                    dHsp_dQy_E1 += l_E1 * (Si(0)*Sj(2) + Si(2)*Sj(0));  // (SxSz+SzSx)
                } else if (bond_type == 1) {  // y-bond
                    dHsp_dQx_E1 += l_E1 * (Si(1)*Sj(2) + Si(2)*Sj(1));  // (SySz+SzSy)
                    dHsp_dQy_E1 += l_E1 * (Si(0)*Sj(1) + Si(1)*Sj(0));  // (SxSy+SySx)
                } else {  // z-bond
                    dHsp_dQx_E1 += l_E1 * (Si(0)*Sj(2) + Si(2)*Sj(0));  // (SxSz+SzSx)
                    dHsp_dQy_E1 += l_E1 * (Si(1)*Sj(2) + Si(2)*Sj(1));  // (SySz+SzSy)
                }
                
                // E2 terms (bond-dependent)
                if (bond_type == 0) {  // x-bond
                    dHsp_dQx_E2 += l_E2 * (Si(1)*Sj(1) - Si(2)*Sj(2));  // (SySy-SzSz)
                    dHsp_dQy_E2 += l_E2 * (Si(1)*Sj(2) + Si(2)*Sj(1));  // (SySz+SzSy)
                } else if (bond_type == 1) {  // y-bond
                    dHsp_dQx_E2 += l_E2 * (Si(2)*Sj(2) - Si(0)*Sj(0));  // (SzSz-SxSx)
                    dHsp_dQy_E2 += l_E2 * (Si(0)*Sj(2) + Si(2)*Sj(0));  // (SxSz+SzSx)
                } else {  // z-bond
                    dHsp_dQx_E2 += l_E2 * (Si(0)*Sj(0) - Si(1)*Sj(1));  // (SxSx-SySy)
                    dHsp_dQy_E2 += l_E2 * (Si(0)*Sj(1) + Si(1)*Sj(0));  // (SxSy+SySx)
                }
                
                // A1 term
                dHsp_dQ_A1 += l_A1 * Si.dot(Sj);
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
            int bond_type = nn_bond_types[i][n];
            
            // Pure spin: -J · Sj
            H -= nn_interaction[i][n] * Sj;
            
            // E1 spin-phonon coupling (bond-dependent, like Γ')
            if (bond_type == 0) {  // x-bond: Qx_E1*(SxSy+SySx) + Qy_E1*(SxSz+SzSx)
                H(0) -= l_E1 * Qx_E1 * Sj(1);
                H(1) -= l_E1 * Qx_E1 * Sj(0);
                H(0) -= l_E1 * Qy_E1 * Sj(2);
                H(2) -= l_E1 * Qy_E1 * Sj(0);
            } else if (bond_type == 1) {  // y-bond: Qx_E1*(SySz+SzSy) + Qy_E1*(SxSy+SySx)
                H(1) -= l_E1 * Qx_E1 * Sj(2);
                H(2) -= l_E1 * Qx_E1 * Sj(1);
                H(0) -= l_E1 * Qy_E1 * Sj(1);
                H(1) -= l_E1 * Qy_E1 * Sj(0);
            } else {  // z-bond: Qx_E1*(SxSz+SzSx) + Qy_E1*(SySz+SzSy)
                H(0) -= l_E1 * Qx_E1 * Sj(2);
                H(2) -= l_E1 * Qx_E1 * Sj(0);
                H(1) -= l_E1 * Qy_E1 * Sj(2);
                H(2) -= l_E1 * Qy_E1 * Sj(1);
            }
            
            // E2 spin-phonon coupling (bond-dependent, like Γ+η)
            if (bond_type == 0) {  // x-bond: Qx_E2*(SySy-SzSz) + Qy_E2*(SySz+SzSy)
                H(1) -= l_E2 * Qx_E2 * Sj(1);
                H(2) -= -l_E2 * Qx_E2 * Sj(2);
                H(1) -= l_E2 * Qy_E2 * Sj(2);
                H(2) -= l_E2 * Qy_E2 * Sj(1);
            } else if (bond_type == 1) {  // y-bond: Qx_E2*(SzSz-SxSx) + Qy_E2*(SxSz+SzSx)
                H(2) -= l_E2 * Qx_E2 * Sj(2);
                H(0) -= -l_E2 * Qx_E2 * Sj(0);
                H(0) -= l_E2 * Qy_E2 * Sj(2);
                H(2) -= l_E2 * Qy_E2 * Sj(0);
            } else {  // z-bond: Qx_E2*(SxSx-SySy) + Qy_E2*(SxSy+SySx)
                H(0) -= l_E2 * Qx_E2 * Sj(0);
                H(1) -= -l_E2 * Qx_E2 * Sj(1);
                H(0) -= l_E2 * Qy_E2 * Sj(1);
                H(1) -= l_E2 * Qy_E2 * Sj(0);
            }
            
            // Spin-phonon coupling: A1 term
            H -= l_A1 * Q_A1 * Sj;
        }
        
        // 2nd NN interactions
        for (size_t n = 0; n < j2_partners[i].size(); ++n) {
            size_t j = j2_partners[i][n];
            const size_t jdx = j * spin_dim;
            Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
            H -= j2_interaction[i][n] * Sj;
        }
        
        // 3rd NN interactions
        for (size_t n = 0; n < j3_partners[i].size(); ++n) {
            size_t j = j3_partners[i][n];
            const size_t jdx = j * spin_dim;
            Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
            H -= j3_interaction[i][n] * Sj;
        }
        
        // LLG equation: dS/dt = S × H_eff + α S × (S × H_eff)
        Eigen::Vector3d dSdt = spin_derivative(Si, H);
        dxdt[idx] = dSdt(0);
        dxdt[idx+1] = dSdt(1);
        dxdt[idx+2] = dSdt(2);
    }
    
    // =============================================
    // Phonon equations
    // =============================================
    PhononState dph_dt;
    phonon_derivatives(ph, t, dHsp_dQx_E1, dHsp_dQy_E1, dHsp_dQx_E2, dHsp_dQy_E2, dHsp_dQ_A1,
                       dph_dt);
    
    dph_dt.to_array(&dxdt[spin_offset]);
}

// ============================================================
// ODE INTEGRATION
// ============================================================

/**
 * Generic ODE integrator with support for multiple methods
 * 
 * Available methods:
 * - "euler": Explicit Euler (1st order, simple, inaccurate)
 * - "rk2" or "midpoint": Runge-Kutta 2nd order / modified midpoint
 * - "rk4": Classic Runge-Kutta 4th order (good balance, fixed step)
 * - "rk5" or "rkck54": Cash-Karp 5(4) (adaptive, good for smooth problems)
 * - "rk54" or "rkf54": Runge-Kutta-Fehlberg 5(4) (adaptive, equivalent to rkck54)
 * - "dopri5": Dormand-Prince 5(4) (default, recommended for general use)
 * - "rk78" or "rkf78": Runge-Kutta-Fehlberg 7(8) (high accuracy, expensive)
 * - "bulirsch_stoer" or "bs": Bulirsch-Stoer (very high accuracy, expensive)
 * - "adams_bashforth" or "ab": Adams-Bashforth 5-step multistep (efficient for smooth problems)
 * - "adams_moulton" or "am": Adams-Bashforth-Moulton 5-step predictor-corrector (more accurate)
 * - "velocity_verlet", "verlet", "symplectic": Falls back to rk4 (symplectic methods need pair<q,p> state)
 * 
 * For spin-phonon dynamics, recommended methods are:
 * - dopri5: Good default for most problems (adaptive 5th order)
 * - rk78: When high accuracy is needed (adaptive 7th-8th order)
 * - bulirsch_stoer: For very high accuracy requirements
 * - rk4: When fixed step is preferred (simple, robust)
 * - adams_moulton: Efficient for smooth long-time evolution (multistep)
 */
template<typename System, typename Observer>
void PhononLattice::integrate_ode_system(
    System system_func, ODEState& state,
    double T_start, double T_end, double dt_step,
    Observer observer, const string& method,
    bool use_adaptive, double abs_tol, double rel_tol) 
{
    namespace odeint = boost::numeric::odeint;
    
    if (method == "euler") {
        // Explicit Euler method (1st order, simple but inaccurate)
        odeint::integrate_const(
            odeint::euler<ODEState>(),
            system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "rk2" || method == "midpoint") {
        // Modified midpoint method (2nd order)
        odeint::integrate_const(
            odeint::modified_midpoint<ODEState>(),
            system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "rk4") {
        // Classic fixed-step RK4 (4th order, good balance)
        odeint::integrate_const(
            odeint::runge_kutta4<ODEState>(),
            system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "rk5" || method == "rkck54") {
        // Cash-Karp 5(4) adaptive method
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_controlled<odeint::runge_kutta_cash_karp54<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        } else {
            odeint::integrate_const(
                odeint::make_controlled<odeint::runge_kutta_cash_karp54<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        }
    } else if (method == "rk54" || method == "rkf54") {
        // Runge-Kutta-Fehlberg 5(4) adaptive method
        // Note: boost::odeint doesn't have a separate rkf54, use cash_karp54 as equivalent
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_controlled<odeint::runge_kutta_cash_karp54<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        } else {
            odeint::integrate_const(
                odeint::make_controlled<odeint::runge_kutta_cash_karp54<ODEState>>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        }
    } else if (method == "dopri5") {
        // Dormand-Prince 5(4) adaptive method (default, recommended)
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
        // Runge-Kutta-Fehlberg 7(8) (very high accuracy)
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
        // Bulirsch-Stoer method (very high accuracy, expensive)
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::bulirsch_stoer<ODEState>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        } else {
            odeint::integrate_const(
                odeint::bulirsch_stoer<ODEState>(abs_tol, rel_tol),
                system_func, state, T_start, T_end, dt_step, observer);
        }
    } else if (method == "adams_bashforth" || method == "ab") {
        // Adams-Bashforth 5-step multistep method (efficient for smooth problems)
        // Uses rk4 for initial steps, then switches to multistep
        odeint::adams_bashforth<5, ODEState> stepper;
        odeint::integrate_const(stepper, system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "adams_moulton" || method == "am") {
        // Adams-Bashforth-Moulton predictor-corrector (higher accuracy multistep)
        odeint::adams_bashforth_moulton<5, ODEState> stepper;
        odeint::integrate_const(stepper, system_func, state, T_start, T_end, dt_step, observer);
    } else if (method == "rosenbrock4" || method == "rb4") {
        // Rosenbrock 4th order implicit method (good for stiff systems)
        // Uses numerical Jacobian approximation via finite differences
        using ublas_state = boost::numeric::ublas::vector<double>;
        using ublas_matrix = boost::numeric::ublas::matrix<double>;
        
        const size_t N = state.size();
        const double eps_jac = 1e-8;  // Finite difference step for Jacobian
        
        // Convert std::vector state to ublas::vector
        ublas_state ublas_x(N);
        for (size_t i = 0; i < N; ++i) {
            ublas_x(i) = state[i];
        }
        
        // Create wrapper for system function that works with ublas types
        // Must be templated to work with both ublas and std::vector types
        auto ublas_system = [&system_func, N](const ublas_state& x, ublas_state& dxdt, double t) {
            ODEState x_vec(N), dxdt_vec(N);
            for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
            system_func(x_vec, dxdt_vec, t);
            for (size_t i = 0; i < N; ++i) dxdt(i) = dxdt_vec[i];
        };
        
        // Create numerical Jacobian function
        auto ublas_jacobian = [&system_func, N, eps_jac](const ublas_state& x, ublas_matrix& J, double t, ublas_state& dfdt) {
            ODEState x_vec(N), dxdt_base(N), dxdt_pert(N);
            for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
            
            // Compute base derivative
            system_func(x_vec, dxdt_base, t);
            
            // Compute Jacobian columns by finite differences
            J.resize(N, N);
            for (size_t j = 0; j < N; ++j) {
                double x_orig = x_vec[j];
                double h = eps_jac * std::max(1.0, std::abs(x_orig));
                x_vec[j] = x_orig + h;
                system_func(x_vec, dxdt_pert, t);
                x_vec[j] = x_orig;
                
                for (size_t i = 0; i < N; ++i) {
                    J(i, j) = (dxdt_pert[i] - dxdt_base[i]) / h;
                }
            }
            
            // Compute df/dt by finite differences in time
            double h_t = eps_jac * std::max(1.0, std::abs(t));
            for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
            system_func(x_vec, dxdt_pert, t + h_t);
            for (size_t i = 0; i < N; ++i) {
                dfdt(i) = (dxdt_pert[i] - dxdt_base[i]) / h_t;
            }
        };
        
        // Create implicit system as pair of (system, jacobian)
        auto implicit_system = std::make_pair(ublas_system, ublas_jacobian);
        
        // Create ublas observer wrapper
        auto ublas_observer = [&observer, N](const ublas_state& x, double t) {
            ODEState x_vec(N);
            for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
            observer(x_vec, t);
        };
        
        // Use rosenbrock4 with dense output for adaptive stepping
        // Note: rosenbrock4<double> means double is the value_type (scalar type)
        //       The state type is automatically ublas::vector<double>
        if (use_adaptive) {
            odeint::integrate_adaptive(
                odeint::make_dense_output<odeint::rosenbrock4<double>>(abs_tol, rel_tol),
                implicit_system, ublas_x, T_start, T_end, dt_step, ublas_observer);
        } else {
            odeint::integrate_const(
                odeint::make_dense_output<odeint::rosenbrock4<double>>(abs_tol, rel_tol),
                implicit_system, ublas_x, T_start, T_end, dt_step, ublas_observer);
        }
        
        // Copy result back to std::vector state
        for (size_t i = 0; i < N; ++i) {
            state[i] = ublas_x(i);
        }
    } else if (method == "implicit_euler" || method == "ie") {
        // Implicit Euler method (1st order, very stable for stiff systems)
        // Uses numerical Jacobian approximation via finite differences
        using ublas_state = boost::numeric::ublas::vector<double>;
        using ublas_matrix = boost::numeric::ublas::matrix<double>;
        
        const size_t N = state.size();
        const double eps_jac = 1e-8;
        
        // Convert to ublas state
        ublas_state ublas_x(N);
        for (size_t i = 0; i < N; ++i) {
            ublas_x(i) = state[i];
        }
        
        // Create wrapper for system function
        auto ublas_system = [&system_func, N](const ublas_state& x, ublas_state& dxdt, double t) {
            ODEState x_vec(N), dxdt_vec(N);
            for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
            system_func(x_vec, dxdt_vec, t);
            for (size_t i = 0; i < N; ++i) dxdt(i) = dxdt_vec[i];
        };
        
        // Create numerical Jacobian function for implicit_euler
        // Note: implicit_euler uses 3-argument Jacobian: (x, J, t) without dfdt
        auto ublas_jacobian = [&system_func, N, eps_jac](const ublas_state& x, ublas_matrix& J, double t) {
            ODEState x_vec(N), dxdt_base(N), dxdt_pert(N);
            for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
            
            system_func(x_vec, dxdt_base, t);
            
            J.resize(N, N);
            for (size_t j = 0; j < N; ++j) {
                double x_orig = x_vec[j];
                double h = eps_jac * std::max(1.0, std::abs(x_orig));
                x_vec[j] = x_orig + h;
                system_func(x_vec, dxdt_pert, t);
                x_vec[j] = x_orig;
                
                for (size_t i = 0; i < N; ++i) {
                    J(i, j) = (dxdt_pert[i] - dxdt_base[i]) / h;
                }
            }
        };
        
        auto implicit_system = std::make_pair(ublas_system, ublas_jacobian);
        
        // Create ublas observer wrapper
        auto ublas_observer = [&observer, N](const ublas_state& x, double t) {
            ODEState x_vec(N);
            for (size_t i = 0; i < N; ++i) x_vec[i] = x(i);
            observer(x_vec, t);
        };
        
        // Implicit Euler integration with manual stepping
        // Note: implicit_euler<double> uses ublas::vector<double> as state
        odeint::implicit_euler<double> stepper;
        double t = T_start;
        while (t < T_end) {
            stepper.do_step(implicit_system, ublas_x, t, dt_step);
            t += dt_step;
            ublas_observer(ublas_x, t);
        }
        
        // Copy result back
        for (size_t i = 0; i < N; ++i) {
            state[i] = ublas_x(i);
        }
    } else if (method == "velocity_verlet" || method == "verlet" || 
               method == "symplectic_rkn" || method == "symrkn" ||
               method == "symplectic" || method == "sym") {
        // Symplectic methods require special pair<q,p> state type
        // For flat vector state, use high-order method with small timestep for energy conservation
        cout << "Note: symplectic methods require pair<q,p> state; using rk4 (fixed step) as alternative" << endl;
        cout << "      For better energy conservation, reduce timestep or use bulirsch_stoer" << endl;
        odeint::integrate_const(
            odeint::runge_kutta4<ODEState>(),
            system_func, state, T_start, T_end, dt_step, observer);
    } else {
        // Default to dopri5 if unknown method specified
        cout << "Warning: Unknown method '" << method << "', using dopri5" << endl;
        cout << "Available explicit methods: euler, rk2/midpoint, rk4, rk5/rkck54, rk54/rkf54, dopri5, " << endl;
        cout << "                            rk78/rkf78, bulirsch_stoer/bs, adams_bashforth/ab, adams_moulton/am" << endl;
        cout << "Available implicit methods: rosenbrock4/rb4, implicit_euler/ie" << endl;
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
#ifndef HDF5_ENABLED
    std::cerr << "Error: HDF5 support is required for molecular dynamics output." << endl;
    std::cerr << "Please rebuild with -DHDF5_ENABLED flag and HDF5 libraries." << endl;
    return;
#endif

    if (!out_dir.empty()) {
        std::filesystem::create_directories(out_dir);
    }
    
    cout << "Running PhononLattice spin-phonon dynamics: t=" << T_start << " → " << T_end << endl;
    cout << "Integration method: " << method << endl;
    cout << "Initial step size: " << dt_initial << endl;
    
    // Convert to flat state
    ODEState state = to_state();
    
#ifdef HDF5_ENABLED
    // Create HDF5 writer with comprehensive metadata (like Lattice class)
    std::unique_ptr<HDF5MDWriter> hdf5_writer;
    
    // Storage for phonon trajectory (appended separately since HDF5MDWriter doesn't handle phonons)
    // E1 mode (2-component), E2 mode (2-component), A1 mode (1-component)
    vector<double> times_phonon;
    vector<double> Qx_E1_traj, Qy_E1_traj;
    vector<double> Qx_E2_traj, Qy_E2_traj;
    vector<double> Q_A1_traj;
    vector<double> Vx_E1_traj, Vy_E1_traj;
    vector<double> Vx_E2_traj, Vy_E2_traj;
    vector<double> V_A1_traj;
    vector<double> energy_traj;
    
    if (!out_dir.empty()) {
        string hdf5_file = out_dir + "/trajectory.h5";
        cout << "Writing trajectory to HDF5 file: " << hdf5_file << endl;
        hdf5_writer = std::make_unique<HDF5MDWriter>(
            hdf5_file, lattice_size, spin_dim, N_atoms, 
            dim1, dim2, dim3, method, 
            dt_initial, T_start, T_end, save_interval, spin_length, 
            &site_positions, 10000);
    }
#endif
    
    size_t step_count = 0;
    size_t save_count = 0;
    
    auto observer = [&](const ODEState& x, double t) {
        if (step_count % save_interval == 0) {
            // Extract spin part of state (excluding phonon DOF)
            const size_t spin_state_size = spin_dim * lattice_size;
            
            // Compute magnetizations directly from flat state (zero allocation)
            Eigen::Vector3d M_local = Eigen::Vector3d::Zero();
            Eigen::Vector3d M_staggered = Eigen::Vector3d::Zero();
            
            for (size_t i = 0; i < lattice_size; ++i) {
                double sign = (i % N_atoms == 0) ? 1.0 : -1.0;  // Sublattice alternation
                for (size_t d = 0; d < spin_dim; ++d) {
                    M_local(d) += x[i * spin_dim + d];
                    M_staggered(d) += sign * x[i * spin_dim + d];
                }
            }
            M_local /= lattice_size;
            M_staggered /= lattice_size;
            
            // M_global is same as M_local for PhononLattice (no frame transformation)
            Eigen::Vector3d M_global = M_local;
            
            // Extract phonon state from end of ODE state vector
            // Layout: [Qx_E1, Qy_E1, Qx_E2, Qy_E2, Q_A1, Vx_E1, Vy_E1, Vx_E2, Vy_E2, V_A1]
            double Qx_E1 = x[spin_state_size + 0];
            double Qy_E1 = x[spin_state_size + 1];
            double Qx_E2 = x[spin_state_size + 2];
            double Qy_E2 = x[spin_state_size + 3];
            double Q_A1 = x[spin_state_size + 4];
            double Vx_E1 = x[spin_state_size + 5];
            double Vy_E1 = x[spin_state_size + 6];
            double Vx_E2 = x[spin_state_size + 7];
            double Vy_E2 = x[spin_state_size + 8];
            double V_A1 = x[spin_state_size + 9];
            
#ifdef HDF5_ENABLED
            // Write full spin configuration to HDF5 (like Lattice class)
            if (hdf5_writer) {
                hdf5_writer->write_flat_step(t, M_staggered, M_local, M_global, x.data());
                
                // Store phonon data for later writing
                times_phonon.push_back(t);
                Qx_E1_traj.push_back(Qx_E1);
                Qy_E1_traj.push_back(Qy_E1);
                Qx_E2_traj.push_back(Qx_E2);
                Qy_E2_traj.push_back(Qy_E2);
                Q_A1_traj.push_back(Q_A1);
                Vx_E1_traj.push_back(Vx_E1);
                Vy_E1_traj.push_back(Vy_E1);
                Vx_E2_traj.push_back(Vx_E2);
                Vy_E2_traj.push_back(Vy_E2);
                V_A1_traj.push_back(V_A1);
                
                // Compute energy for monitoring
                const_cast<PhononLattice*>(this)->from_state(x);
                energy_traj.push_back(energy_density());
            }
#endif
            
            // Progress output
            if (step_count % (save_interval * 10) == 0) {
                double E1_amp = std::sqrt(Qx_E1*Qx_E1 + Qy_E1*Qy_E1);
                double E2_amp = std::sqrt(Qx_E2*Qx_E2 + Qy_E2*Qy_E2);
                cout << "t=" << t << ", |M|=" << M_local.norm()
                     << ", |M_stag|=" << M_staggered.norm()
                     << ", |Q_E1|=" << E1_amp
                     << ", |Q_E2|=" << E2_amp
                     << ", Q_A1=" << Q_A1 << endl;
            }
            
            save_count++;
        }
        step_count++;
    };
    
    auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
        this->ode_system(x, dxdt, t);
    };
    
    // Integrate using selected method
    double abs_tol = (method == "bulirsch_stoer") ? 1e-8 : 1e-6;
    double rel_tol = (method == "bulirsch_stoer") ? 1e-8 : 1e-6;
    integrate_ode_system(system_func, state, T_start, T_end, dt_initial,
                        observer, method, true, abs_tol, rel_tol);
    
    from_state(state);
    
#ifdef HDF5_ENABLED
    // Write phonon trajectory data to HDF5 file
    if (hdf5_writer && !times_phonon.empty()) {
        // Get the underlying HDF5 file handle to add phonon data
        // We need to close the main writer first, then reopen to add phonon group
        hdf5_writer->close();
        
        // Reopen file in append mode to add phonon trajectory
        string hdf5_file = out_dir + "/trajectory.h5";
        H5::H5File h5file(hdf5_file, H5F_ACC_RDWR);
        
        // Create phonon trajectory group
        H5::Group phonon_group = h5file.createGroup("/phonon_trajectory");
        
        hsize_t dims[1] = {times_phonon.size()};
        H5::DataSpace dataspace(1, dims);
        
        auto write_dataset = [&](const string& name, const vector<double>& data) {
            H5::DataSet ds = phonon_group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
            ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };
        
        // E1 mode (IR active, 2-component)
        write_dataset("Qx_E1", Qx_E1_traj);
        write_dataset("Qy_E1", Qy_E1_traj);
        write_dataset("Vx_E1", Vx_E1_traj);
        write_dataset("Vy_E1", Vy_E1_traj);
        
        // E2 mode (Raman active, 2-component)
        write_dataset("Qx_E2", Qx_E2_traj);
        write_dataset("Qy_E2", Qy_E2_traj);
        write_dataset("Vx_E2", Vx_E2_traj);
        write_dataset("Vy_E2", Vy_E2_traj);
        
        // A1 mode (Raman active, 1-component)
        write_dataset("Q_A1", Q_A1_traj);
        write_dataset("V_A1", V_A1_traj);
        
        // Energy
        write_dataset("energy", energy_traj);
        
        // Write phonon parameters as metadata
        H5::Group meta_group = h5file.openGroup("/metadata");
        H5::DataSpace scalar_space(H5S_SCALAR);
        
        auto write_scalar = [&](const string& name, double val) {
            H5::Attribute attr = meta_group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar_space);
            attr.write(H5::PredType::NATIVE_DOUBLE, &val);
        };
        
        write_scalar("omega_E1", phonon_params.omega_E1);
        write_scalar("omega_E2", phonon_params.omega_E2);
        write_scalar("omega_A1", phonon_params.omega_A1);
        write_scalar("g3_E1A1", phonon_params.g3_E1A1);
        write_scalar("g3_E2A1", phonon_params.g3_E2A1);
        write_scalar("g3_E1E2", phonon_params.g3_E1E2);
        write_scalar("gamma_E1", phonon_params.gamma_E1);
        write_scalar("gamma_E2", phonon_params.gamma_E2);
        write_scalar("gamma_A1", phonon_params.gamma_A1);
        write_scalar("lambda_E1_quartic", phonon_params.lambda_E1);
        write_scalar("lambda_E2_quartic", phonon_params.lambda_E2);
        write_scalar("lambda_A1_quartic", phonon_params.lambda_A1);
        write_scalar("lambda_E1", spin_phonon_params.lambda_E1);
        write_scalar("lambda_E2", spin_phonon_params.lambda_E2);
        write_scalar("lambda_A1", spin_phonon_params.lambda_A1);
        
        phonon_group.close();
        meta_group.close();
        h5file.close();
        
        cout << "HDF5 trajectory saved with " << save_count << " snapshots (full spin + phonon)" << endl;
    }
#endif
    
    cout << "Dynamics complete! (" << step_count << " steps, " << save_count << " saved)" << endl;
}

// ============================================================
// MONTE CARLO METHODS
// ============================================================

size_t PhononLattice::metropolis_sweep(double T) {
    size_t accepted = 0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        Eigen::Vector3d old_spin = spins[i];
        Eigen::Vector3d new_spin = gen_random_spin();
        
        double dE = site_energy_diff(new_spin, old_spin, i);
        
        if (dE < 0 || random_double_lehman(0, 1) < std::exp(-dE / T)) {
            spins[i] = new_spin;
            accepted++;
        }
    }
    return accepted;
}

void PhononLattice::overrelaxation_sweep() {
    for (size_t i = 0; i < lattice_size; ++i) {
        Eigen::Vector3d H = get_local_field(i);
        double norm = H.norm();
        if (norm > 1e-10) {
            H /= norm;
            // Reflect spin about local field direction
            spins[i] = 2.0 * H.dot(spins[i]) * H - spins[i];
            spins[i] = spins[i].normalized() * spin_length;
        }
    }
}

// ============================================================
// SIMULATED ANNEALING
// ============================================================

void PhononLattice::simulated_annealing(
    double T_start, double T_end, size_t n_steps,
    size_t overrelax_rate, double cooling_rate,
    string out_dir, bool save_observables,
    bool T_zero, size_t n_deterministics,
    bool adiabatic_phonons) 
{
    cout << "Starting PhononLattice simulated annealing..." << endl;
    cout << "T: " << T_start << " → " << T_end << ", sweeps per temp: " << n_steps << endl;
    if (adiabatic_phonons) {
        cout << "Adiabatic phonons ENABLED: phonons will be relaxed at each temperature step" << endl;
    } else {
        cout << "Adiabatic phonons DISABLED: phonons will be kept at Q=0 during MC" << endl;
    }
    
    // Initialize phonons to zero (will be updated if adiabatic_phonons is true)
    phonons = PhononState();
    
    if (!out_dir.empty()) {
        std::filesystem::create_directories(out_dir);
    }
    
#ifdef HDF5_ENABLED
    std::unique_ptr<H5::H5File> h5file;
    vector<double> steps_data, temps_data, energies_data, acc_rates_data;
    vector<double> Qx_data, Qy_data, QR_data;  // Track phonons if adiabatic
    
    if (save_observables && !out_dir.empty()) {
        h5file = std::make_unique<H5::H5File>(out_dir + "/annealing.h5", H5F_ACC_TRUNC);
    }
#endif
    
    double T = T_start;
    size_t temp_step = 0;
    
    while (T > T_end) {
        size_t accepted = 0;
        
        // Perform n_steps sweeps at this temperature
        for (size_t step = 0; step < n_steps; ++step) {
            accepted += metropolis_sweep(T);
            
            // Overrelaxation
            if (overrelax_rate > 0 && step % overrelax_rate == 0) {
                overrelaxation_sweep();
            }
        }
        
        // If using adiabatic phonons, relax phonons to equilibrium for current spin configuration
        if (adiabatic_phonons) {
            relax_phonons(1e-10, 1000, 1.0);
        }
        
        // Calculate acceptance rate (only counts Metropolis moves, not overrelaxation)
        double acceptance = double(accepted) / double(n_steps * lattice_size);
        
        // Progress report every 10 temperature steps or near the end
        if (temp_step % 10 == 0 || T <= T_end * 1.5) {
            double E = energy_density();
            Eigen::Vector3d M = magnetization();
            Eigen::Vector3d M_stag = staggered_magnetization();
            cout << "T=" << std::scientific << std::setprecision(4) << T 
                 << ", E/N=" << std::fixed << std::setprecision(6) << E 
                 << ", acc=" << std::fixed << std::setprecision(4) << acceptance
                 << ", |M|=" << std::fixed << std::setprecision(4) << M.norm()
                 << ", |M_stag|=" << std::fixed << std::setprecision(4) << M_stag.norm();
            if (adiabatic_phonons) {
                cout << ", |Q_E1|=" << std::fixed << std::setprecision(4) << E1_amplitude()
                     << ", |Q_E2|=" << std::fixed << std::setprecision(4) << E2_amplitude()
                     << ", Q_A1=" << std::fixed << std::setprecision(4) << phonons.Q_A1;
            }
            cout << endl;
        }
        
#ifdef HDF5_ENABLED
        if (save_observables && h5file) {
            steps_data.push_back(static_cast<double>(temp_step));
            temps_data.push_back(T);
            energies_data.push_back(energy_density());
            acc_rates_data.push_back(acceptance);
            if (adiabatic_phonons) {
                Qx_data.push_back(phonons.Q_x_E1);
                Qy_data.push_back(phonons.Q_y_E1);
                QR_data.push_back(phonons.Q_A1);
            }
        }
#endif
        
        // Cool down
        T *= cooling_rate;
        ++temp_step;
    }
    
    // Final report
    double E_final = energy_density();
    Eigen::Vector3d M_final = magnetization();
    Eigen::Vector3d M_stag_final = staggered_magnetization();
    cout << "\n=== Simulated Annealing Complete ===" << endl;
    cout << "Temperature steps: " << temp_step << endl;
    cout << "Final energy density: " << E_final << endl;
    cout << "Final magnetization: [" << M_final.transpose() << "], |M|=" << M_final.norm() << endl;
    cout << "Final staggered M: [" << M_stag_final.transpose() << "], |M_stag|=" << M_stag_final.norm() << endl;
    if (adiabatic_phonons) {
        cout << "E1: Qx=" << phonons.Q_x_E1 << ", Qy=" << phonons.Q_y_E1 << ", |Q_E1|=" << E1_amplitude() << endl;
        cout << "E2: Qx=" << phonons.Q_x_E2 << ", Qy=" << phonons.Q_y_E2 << ", |Q_E2|=" << E2_amplitude() << endl;
        cout << "A1: Q=" << phonons.Q_A1 << endl;
    }
    cout << "====================================" << endl;
    
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
        
        if (adiabatic_phonons && !Qx_data.empty()) {
            ds = ann_group.createDataSet("Qx", H5::PredType::NATIVE_DOUBLE, dataspace);
            ds.write(Qx_data.data(), H5::PredType::NATIVE_DOUBLE);
            ds = ann_group.createDataSet("Qy", H5::PredType::NATIVE_DOUBLE, dataspace);
            ds.write(Qy_data.data(), H5::PredType::NATIVE_DOUBLE);
            ds = ann_group.createDataSet("QR", H5::PredType::NATIVE_DOUBLE, dataspace);
            ds.write(QR_data.data(), H5::PredType::NATIVE_DOUBLE);
        }
        
        h5file->close();
        cout << "Annealing data saved to " << out_dir << "/annealing.h5" << endl;
    }
#endif
    
    // Save spin config after annealing (before deterministic sweeps)
    if (!out_dir.empty()) {
        save_spin_config(out_dir + "/spins_T=" + std::to_string(T_end) + ".txt");
    }
    
    // T=0 deterministic sweeps if requested
    if (T_zero && n_deterministics > 0) {
        // Energy breakdown BEFORE deterministic sweeps
        cout << "\n=== Energy BEFORE deterministic sweeps ===" << endl;
        cout << "  Spin energy:       " << spin_energy() << " (" << spin_energy()/lattice_size << " per site)" << endl;
        cout << "  Phonon energy:     " << phonon_energy() << endl;
        cout << "  Spin-phonon energy: " << spin_phonon_energy() << endl;
        cout << "  Total energy:      " << total_energy() << " (" << energy_density() << " per site)" << endl;
        cout << "  |Q_E1| = " << E1_amplitude() << ", |Q_E2| = " << E2_amplitude() << ", Q_A1 = " << phonons.Q_A1 << endl;
        
        cout << "\nPerforming " << n_deterministics << " deterministic sweeps at T=0..." << endl;
        deterministic_sweep(n_deterministics);
        
        // If using adiabatic phonons, relax phonons again after deterministic sweeps
        if (adiabatic_phonons) {
            relax_phonons(1e-10, 1000, 1.0);
        }
        
        // Energy breakdown AFTER deterministic sweeps
        cout << "\n=== Energy AFTER deterministic sweeps ===" << endl;
        cout << "  Spin energy:       " << spin_energy() << " (" << spin_energy()/lattice_size << " per site)" << endl;
        cout << "  Phonon energy:     " << phonon_energy() << endl;
        cout << "  Spin-phonon energy: " << spin_phonon_energy() << endl;
        cout << "  Total energy:      " << total_energy() << " (" << energy_density() << " per site)" << endl;
        cout << "  |Q_E1| = " << E1_amplitude() << ", |Q_E2| = " << E2_amplitude() << ", Q_A1 = " << phonons.Q_A1 << endl;
        
        // Save final configuration after T=0 sweeps
        if (!out_dir.empty()) {
            save_spin_config(out_dir + "/spins_T=0.txt");
            cout << "Final spin config saved to " << out_dir << "/spins_T=0.txt" << endl;
        }
    } else if (!out_dir.empty()) {
        // If no T=0 sweeps, just save the final config
        save_spin_config(out_dir + "/spins_final.txt");
        cout << "Final spin config saved to " << out_dir << "/spins_final.txt" << endl;
    }
}

// ============================================================
// PHONON RELAXATION
// ============================================================

bool PhononLattice::relax_phonons(double tol, size_t max_iter, double damping) {
    cout << "Relaxing phonons to equilibrium for current spin configuration..." << endl;
    
    // Compute spin-phonon coupling derivatives (fixed for current spin config)
    double dH_dQx_E1_spin = dH_dQx_E1();
    double dH_dQy_E1_spin = dH_dQy_E1();
    double dH_dQx_E2_spin = dH_dQx_E2();
    double dH_dQy_E2_spin = dH_dQy_E2();
    double dH_dQ_A1_spin = dH_dQ_A1();
    
    cout << "  Spin-phonon forces:" << endl;
    cout << "    E1: dH/dQx_E1=" << dH_dQx_E1_spin << ", dH/dQy_E1=" << dH_dQy_E1_spin << endl;
    cout << "    E2: dH/dQx_E2=" << dH_dQx_E2_spin << ", dH/dQy_E2=" << dH_dQy_E2_spin << endl;
    cout << "    A1: dH/dQ_A1=" << dH_dQ_A1_spin << endl;
    
    double omega_E1_sq = phonon_params.omega_E1 * phonon_params.omega_E1;
    double omega_E2_sq = phonon_params.omega_E2 * phonon_params.omega_E2;
    double omega_A1_sq = phonon_params.omega_A1 * phonon_params.omega_A1;
    double lambda_E1 = phonon_params.lambda_E1;
    double lambda_E2 = phonon_params.lambda_E2;
    double lambda_A1 = phonon_params.lambda_A1;
    double g3_E1A1 = phonon_params.g3_E1A1;
    double g3_E2A1 = phonon_params.g3_E2A1;
    double g3_E1E2 = phonon_params.g3_E1E2;
    
    // Start with the linear approximation (ignoring quartic and g3 coupling)
    double Qx_E1 = -dH_dQx_E1_spin / omega_E1_sq;
    double Qy_E1 = -dH_dQy_E1_spin / omega_E1_sq;
    double Qx_E2 = -dH_dQx_E2_spin / omega_E2_sq;
    double Qy_E2 = -dH_dQy_E2_spin / omega_E2_sq;
    double Q_A1 = -dH_dQ_A1_spin / omega_A1_sq;
    
    // Use Newton-Raphson iteration for nonlinear terms
    bool has_nonlinear = lambda_E1 > 1e-10 || lambda_E2 > 1e-10 || lambda_A1 > 1e-10 
                       || std::abs(g3_E1A1) > 1e-10 || std::abs(g3_E2A1) > 1e-10
                       || std::abs(g3_E1E2) > 1e-10;
    
    if (has_nonlinear) {
        cout << "  Using iterative solver for nonlinear terms..." << endl;
        
        for (size_t iter = 0; iter < max_iter; ++iter) {
            double Q_E1_sq = Qx_E1*Qx_E1 + Qy_E1*Qy_E1;
            double Q_E2_sq = Qx_E2*Qx_E2 + Qy_E2*Qy_E2;
            
            // Gradient (residual): should be zero at equilibrium
            // E1: F_x = ω_E1² Qx_E1 + λ_E1 Q_E1² Qx_E1 + 2 g3_E1A1 Qx_E1 Q_A1 + g3_E1E2 Qx_E2 + dH_dQx_E1
            double F_x_E1 = omega_E1_sq * Qx_E1 + lambda_E1 * Q_E1_sq * Qx_E1 
                          + 2.0 * g3_E1A1 * Qx_E1 * Q_A1 + g3_E1E2 * Qx_E2 + dH_dQx_E1_spin;
            double F_y_E1 = omega_E1_sq * Qy_E1 + lambda_E1 * Q_E1_sq * Qy_E1 
                          + 2.0 * g3_E1A1 * Qy_E1 * Q_A1 + g3_E1E2 * Qy_E2 + dH_dQy_E1_spin;
            
            // E2: F_x = ω_E2² Qx_E2 + λ_E2 Q_E2² Qx_E2 + 2 g3_E2A1 Qx_E2 Q_A1 + g3_E1E2 Qx_E1 + dH_dQx_E2
            double F_x_E2 = omega_E2_sq * Qx_E2 + lambda_E2 * Q_E2_sq * Qx_E2 
                          + 2.0 * g3_E2A1 * Qx_E2 * Q_A1 + g3_E1E2 * Qx_E1 + dH_dQx_E2_spin;
            double F_y_E2 = omega_E2_sq * Qy_E2 + lambda_E2 * Q_E2_sq * Qy_E2 
                          + 2.0 * g3_E2A1 * Qy_E2 * Q_A1 + g3_E1E2 * Qy_E1 + dH_dQy_E2_spin;
            
            // A1: F = ω_A1² Q_A1 + λ_A1 Q_A1³ + g3_E1A1 Q_E1² + g3_E2A1 Q_E2² + dH_dQ_A1
            double F_A1 = omega_A1_sq * Q_A1 + lambda_A1 * Q_A1 * Q_A1 * Q_A1 
                        + g3_E1A1 * Q_E1_sq + g3_E2A1 * Q_E2_sq + dH_dQ_A1_spin;
            
            double residual = std::sqrt(F_x_E1*F_x_E1 + F_y_E1*F_y_E1 
                                      + F_x_E2*F_x_E2 + F_y_E2*F_y_E2 
                                      + F_A1*F_A1);
            
            if (residual < tol) {
                phonons.Q_x_E1 = Qx_E1;
                phonons.Q_y_E1 = Qy_E1;
                phonons.Q_x_E2 = Qx_E2;
                phonons.Q_y_E2 = Qy_E2;
                phonons.Q_A1 = Q_A1;
                phonons.V_x_E1 = 0.0;
                phonons.V_y_E1 = 0.0;
                phonons.V_x_E2 = 0.0;
                phonons.V_y_E2 = 0.0;
                phonons.V_A1 = 0.0;
                
                cout << "  Phonon relaxation converged in " << iter << " iterations!" << endl;
                cout << "  Equilibrium E1: Qx=" << Qx_E1 << ", Qy=" << Qy_E1 
                     << ", |Q_E1|=" << std::sqrt(Q_E1_sq) << endl;
                cout << "  Equilibrium E2: Qx=" << Qx_E2 << ", Qy=" << Qy_E2 
                     << ", |Q_E2|=" << std::sqrt(Q_E2_sq) << endl;
                cout << "  Equilibrium A1: Q=" << Q_A1 << endl;
                return true;
            }
            
            // Jacobian diagonal elements (approximate, ignoring off-diagonal)
            double J_xx_E1 = omega_E1_sq + lambda_E1 * (3.0*Qx_E1*Qx_E1 + Qy_E1*Qy_E1) + 2.0*g3_E1A1*Q_A1;
            double J_yy_E1 = omega_E1_sq + lambda_E1 * (Qx_E1*Qx_E1 + 3.0*Qy_E1*Qy_E1) + 2.0*g3_E1A1*Q_A1;
            double J_xx_E2 = omega_E2_sq + lambda_E2 * (3.0*Qx_E2*Qx_E2 + Qy_E2*Qy_E2) + 2.0*g3_E2A1*Q_A1;
            double J_yy_E2 = omega_E2_sq + lambda_E2 * (Qx_E2*Qx_E2 + 3.0*Qy_E2*Qy_E2) + 2.0*g3_E2A1*Q_A1;
            double J_A1 = omega_A1_sq + 3.0*lambda_A1*Q_A1*Q_A1;
            
            // Damped Newton step (diagonal approximation)
            double step_size = damping;
            if (J_xx_E1 > 1e-10) Qx_E1 -= step_size * F_x_E1 / J_xx_E1;
            if (J_yy_E1 > 1e-10) Qy_E1 -= step_size * F_y_E1 / J_yy_E1;
            if (J_xx_E2 > 1e-10) Qx_E2 -= step_size * F_x_E2 / J_xx_E2;
            if (J_yy_E2 > 1e-10) Qy_E2 -= step_size * F_y_E2 / J_yy_E2;
            if (J_A1 > 1e-10) Q_A1 -= step_size * F_A1 / J_A1;
            
            if (iter % 100 == 0 && iter > 0) {
                cout << "  iter=" << iter << ", |F|=" << residual << endl;
            }
        }
        
        cout << "  WARNING: Phonon relaxation did not fully converge after " << max_iter << " iterations!" << endl;
    }
    
    phonons.Q_x_E1 = Qx_E1;
    phonons.Q_y_E1 = Qy_E1;
    phonons.Q_x_E2 = Qx_E2;
    phonons.Q_y_E2 = Qy_E2;
    phonons.Q_A1 = Q_A1;
    phonons.V_x_E1 = 0.0;
    phonons.V_y_E1 = 0.0;
    phonons.V_x_E2 = 0.0;
    phonons.V_y_E2 = 0.0;
    phonons.V_A1 = 0.0;
    
    cout << "  Equilibrium E1: Qx=" << Qx_E1 << ", Qy=" << Qy_E1 
         << ", |Q_E1|=" << std::sqrt(Qx_E1*Qx_E1 + Qy_E1*Qy_E1) << endl;
    cout << "  Equilibrium E2: Qx=" << Qx_E2 << ", Qy=" << Qy_E2 
         << ", |Q_E2|=" << std::sqrt(Qx_E2*Qx_E2 + Qy_E2*Qy_E2) << endl;
    cout << "  Equilibrium A1: Q=" << Q_A1 << endl;
    
    // Energy breakdown AFTER phonon relaxation
    cout << "  --- Energy after phonon relaxation ---" << endl;
    cout << "    Spin energy:       " << spin_energy() << " (" << spin_energy()/lattice_size << " per site)" << endl;
    cout << "    Phonon energy:     " << phonon_energy() << endl;
    cout << "    Spin-phonon energy: " << spin_phonon_energy() << endl;
    cout << "    Total energy:      " << total_energy() << " (" << energy_density() << " per site)" << endl;
    
    return true;
}

bool PhononLattice::relax_joint(double tol, size_t max_iter, size_t spin_sweeps_per_iter, bool phonon_only) {
    if (phonon_only) {
        cout << "Phonon-only relaxation (spins fixed)..." << endl;
    } else {
        cout << "Joint spin-phonon relaxation to find true steady state..." << endl;
    }
    
    double prev_energy = total_energy();
    double prev_Q_E = E1_amplitude();
    
    for (size_t iter = 0; iter < max_iter; ++iter) {
        // Step 1: Relax phonons for current spin configuration
        relax_phonons(1e-10, 1000, 1.0);
        
        // Step 2: Relax spins for current phonon configuration (skip if phonon_only)
        if (!phonon_only) {
            deterministic_sweep(spin_sweeps_per_iter);
        }
        
        // Check convergence
        double curr_energy = total_energy();
        double curr_Q_E = E1_amplitude();
        
        double dE = std::abs(curr_energy - prev_energy);
        double dQ = std::abs(curr_Q_E - prev_Q_E);
        
        if (iter % 10 == 0 || (dE < tol && dQ < tol)) {
            cout << "  " << (phonon_only ? "Phonon" : "Joint") << " relax iter " << iter 
                 << ": E=" << curr_energy 
                 << ", |Q_E|=" << curr_Q_E
                 << ", dE=" << dE 
                 << ", dQ=" << dQ << endl;
        }
        
        if (dE < tol && dQ < tol) {
            cout << "  " << (phonon_only ? "Phonon-only" : "Joint") << " relaxation converged in " << iter << " iterations!" << endl;
            cout << "\n=== Energy AFTER " << (phonon_only ? "phonon-only" : "joint") << " relaxation (equilibrium state) ===" << endl;
            cout << "  Spin energy:       " << spin_energy() << " (" << spin_energy()/lattice_size << " per site)" << endl;
            cout << "  Phonon energy:     " << phonon_energy() << endl;
            cout << "  Spin-phonon energy: " << spin_phonon_energy() << endl;
            cout << "  Total energy:      " << total_energy() << " (" << energy_density() << " per site)" << endl;
            cout << "  |Q_E1| = " << E1_amplitude() << ", |Q_E2| = " << E2_amplitude() << ", Q_A1 = " << phonons.Q_A1 << endl;
            return true;
        }
        
        prev_energy = curr_energy;
        prev_Q_E = curr_Q_E;
    }
    
    cout << "  WARNING: " << (phonon_only ? "Phonon-only" : "Joint") << " relaxation did not fully converge after " << max_iter << " iterations" << endl;
    cout << "\n=== Energy after " << max_iter << " iterations ===" << endl;
    cout << "  Spin energy:       " << spin_energy() << " (" << spin_energy()/lattice_size << " per site)" << endl;
    cout << "  Phonon energy:     " << phonon_energy() << endl;
    cout << "  Spin-phonon energy: " << spin_phonon_energy() << endl;
    cout << "  Total energy:      " << total_energy() << " (" << energy_density() << " per site)" << endl;
    cout << "  |Q_E1| = " << E1_amplitude() << ", |Q_E2| = " << E2_amplitude() << ", Q_A1 = " << phonons.Q_A1 << endl;
    return false;
}

// ============================================================
// DETERMINISTIC SWEEP
// ============================================================

void PhononLattice::deterministic_sweep(size_t num_sweeps) {
    for (size_t sweep = 0; sweep < num_sweeps; ++sweep) {
        size_t count = 0;
        while (count < lattice_size) {
            size_t i = random_int_lehman(lattice_size);
            Eigen::Vector3d local_field = get_local_field(i);
            double norm = local_field.norm();
            
            if (norm < 1e-15) {
                continue;
            } else {
                // Align spin PARALLEL to local field (minimizes energy)
                // H_eff = -∂H/∂Si, so Si should point along H_eff to minimize E = -Si·H_eff
                spins[i] = local_field / norm * spin_length;
            }
            count++;
        }
    }
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

// ============================================================
// 2DCS SPECTROSCOPY
// ============================================================

PhononLattice::MagTrajectory PhononLattice::single_pulse_drive(
    double polarization, double t_B,
    double pulse_amp, double pulse_width, double pulse_freq,
    double T_start, double T_end, double step_size, const string& method) {
    
    // Set up single pulse
    drive_params.E0_1 = pulse_amp;
    drive_params.omega_1 = pulse_freq;
    drive_params.t_1 = t_B;
    drive_params.sigma_1 = pulse_width;
    drive_params.phi_1 = 0.0;
    drive_params.theta_1 = polarization;
    
    // Disable second pulse
    drive_params.E0_2 = 0.0;
    
    // Storage for trajectory
    MagTrajectory trajectory;
    
    // Build initial state from spins + phonons
    ODEState state(state_size);
    for (size_t i = 0; i < lattice_size; ++i) {
        state[i*spin_dim + 0] = spins[i](0);
        state[i*spin_dim + 1] = spins[i](1);
        state[i*spin_dim + 2] = spins[i](2);
    }
    phonons.to_array(&state[spin_dim * lattice_size]);
    
    // Create ODE system wrapper
    auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
        this->ode_system(x, dxdt, t);
    };
    
    // Observer to collect magnetization at regular intervals
    double last_save_time = T_start;
    auto observer = [&](const ODEState& x, double t) {
        if (t - last_save_time >= step_size - 1e-10 || t >= T_end - 1e-10) {
            // Compute magnetizations directly from flat state
            Eigen::Vector3d M_local = Eigen::Vector3d::Zero();
            Eigen::Vector3d M_antiferro = Eigen::Vector3d::Zero();
            Eigen::Vector3d M_global = Eigen::Vector3d::Zero();
            double O_custom = 0.0;
            
            for (size_t i = 0; i < lattice_size; ++i) {
                double sign = (i % 2 == 0) ? 1.0 : -1.0;
                size_t atom = i % N_atoms;
                Eigen::Vector3d S(x[i*spin_dim], x[i*spin_dim+1], x[i*spin_dim+2]);
                M_local += S;
                M_antiferro += sign * S;
                // Transform to global frame using sublattice frame
                M_global += sublattice_frames[atom] * S;
                // Custom order parameter projection
                if (has_ordering_pattern) {
                    O_custom += S.dot(ordering_pattern[i]);
                }
            }
            M_local /= double(lattice_size);
            M_antiferro /= double(lattice_size);
            M_global /= double(lattice_size);
            O_custom /= double(lattice_size);
            
            // Store custom order parameter in 4th element (x component)
            Eigen::Vector3d O_vec(O_custom, 0.0, 0.0);
            trajectory.push_back({t, {M_antiferro, M_local, M_global, O_vec}});
            last_save_time = t;
        }
    };
    
    // Integrate
    integrate_ode_system(system_func, state, T_start, T_end, step_size,
                        observer, method, false, 1e-10, 1e-10);
    
    // Reset drive
    drive_params.E0_1 = 0.0;
    
    return trajectory;
}

PhononLattice::MagTrajectory PhononLattice::double_pulse_drive(
    double polarization_1, double t_B_1,
    double polarization_2, double t_B_2,
    double pulse_amp, double pulse_width, double pulse_freq,
    double T_start, double T_end, double step_size, const string& method) {
    
    // Set up pump (pulse 1)
    drive_params.E0_1 = pulse_amp;
    drive_params.omega_1 = pulse_freq;
    drive_params.t_1 = t_B_1;
    drive_params.sigma_1 = pulse_width;
    drive_params.phi_1 = 0.0;
    drive_params.theta_1 = polarization_1;
    
    // Set up probe (pulse 2)
    drive_params.E0_2 = pulse_amp;
    drive_params.omega_2 = pulse_freq;
    drive_params.t_2 = t_B_2;
    drive_params.sigma_2 = pulse_width;
    drive_params.phi_2 = 0.0;
    drive_params.theta_2 = polarization_2;
    
    // Storage for trajectory
    MagTrajectory trajectory;
    
    // Build initial state from spins + phonons
    ODEState state(state_size);
    for (size_t i = 0; i < lattice_size; ++i) {
        state[i*spin_dim + 0] = spins[i](0);
        state[i*spin_dim + 1] = spins[i](1);
        state[i*spin_dim + 2] = spins[i](2);
    }
    phonons.to_array(&state[spin_dim * lattice_size]);
    
    // Create ODE system wrapper
    auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
        this->ode_system(x, dxdt, t);
    };
    
    // Observer to collect magnetization at regular intervals
    double last_save_time = T_start;
    auto observer = [&](const ODEState& x, double t) {
        if (t - last_save_time >= step_size - 1e-10 || t >= T_end - 1e-10) {
            // Compute magnetizations directly from flat state
            Eigen::Vector3d M_local = Eigen::Vector3d::Zero();
            Eigen::Vector3d M_antiferro = Eigen::Vector3d::Zero();
            Eigen::Vector3d M_global = Eigen::Vector3d::Zero();
            double O_custom = 0.0;
            
            for (size_t i = 0; i < lattice_size; ++i) {
                double sign = (i % 2 == 0) ? 1.0 : -1.0;
                size_t atom = i % N_atoms;
                Eigen::Vector3d S(x[i*spin_dim], x[i*spin_dim+1], x[i*spin_dim+2]);
                M_local += S;
                M_antiferro += sign * S;
                // Transform to global frame using sublattice frame
                M_global += sublattice_frames[atom] * S;
                // Custom order parameter projection
                if (has_ordering_pattern) {
                    O_custom += S.dot(ordering_pattern[i]);
                }
            }
            M_local /= double(lattice_size);
            M_antiferro /= double(lattice_size);
            M_global /= double(lattice_size);
            O_custom /= double(lattice_size);
            
            // Store custom order parameter in 4th element (x component)
            Eigen::Vector3d O_vec(O_custom, 0.0, 0.0);
            trajectory.push_back({t, {M_antiferro, M_local, M_global, O_vec}});
            last_save_time = t;
        }
    };
    
    // Integrate
    integrate_ode_system(system_func, state, T_start, T_end, step_size,
                        observer, method, false, 1e-10, 1e-10);
    
    // Reset drive
    drive_params.E0_1 = 0.0;
    drive_params.E0_2 = 0.0;
    
    return trajectory;
}

void PhononLattice::pump_probe_spectroscopy(
    double polarization,
    double pulse_amp, double pulse_width, double pulse_freq,
    double tau_start, double tau_end, double tau_step,
    double T_start, double T_end, double T_step,
    const string& dir_name, const string& method) {
    
    std::filesystem::create_directories(dir_name);
    
    cout << "\n==========================================" << endl;
    cout << "Pump-Probe Spectroscopy (PhononLattice)" << endl;
    cout << "==========================================" << endl;
    cout << "Pulse parameters:" << endl;
    cout << "  Amplitude: " << pulse_amp << endl;
    cout << "  Width: " << pulse_width << endl;
    cout << "  Frequency: " << pulse_freq << endl;
    cout << "  Polarization: " << polarization << " rad" << endl;
    cout << "Delay scan: " << tau_start << " → " << tau_end << " (step: " << tau_step << ")" << endl;
    cout << "Integration time: " << T_start << " → " << T_end << " (step: " << T_step << ")" << endl;
    
    // Use current configuration as ground state
    cout << "\n[1/3] Using current configuration as ground state..." << endl;
    double E_ground = energy_density();
    SpinVector M_ground = magnetization();
    SpinVector M_ground_global = magnetization_global();
    cout << "  Ground state: E/N = " << E_ground << ", |M| = " << M_ground.norm() << endl;
    cout << "  Global magnetization: " << M_ground_global.transpose() << endl;
    
    // Set the ordering pattern from current (equilibrated) configuration
    set_ordering_pattern();
    cout << "  Ordering pattern captured from ground state" << endl;
    
    // Save initial configuration
    save_positions(dir_name + "/positions.txt");
    save_spin_config(dir_name + "/spins_initial.txt");
    
    // Backup ground state
    SpinConfig ground_state = spins;
    PhononState ground_phonons = phonons;
    
    // Step 2: Reference single-pulse dynamics (pump at t=0)
    cout << "\n[2/3] Running reference single-pulse dynamics (M0)..." << endl;
    auto M0_trajectory = single_pulse_drive(polarization, 0.0, 
                                            pulse_amp, pulse_width, pulse_freq,
                                            T_start, T_end, T_step, method);
    
    // Restore ground state
    spins = ground_state;
    phonons = ground_phonons;
    
    // Step 3: Delay time scan
    int tau_steps = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
    cout << "\n[3/3] Scanning delay times (" << tau_steps << " steps)..." << endl;
    
    // Store all trajectories
    vector<MagTrajectory> M1_trajectories;
    vector<MagTrajectory> M01_trajectories;
    vector<double> tau_values;
    
    M1_trajectories.reserve(tau_steps);
    M01_trajectories.reserve(tau_steps);
    tau_values.reserve(tau_steps);
    
    double current_tau = tau_start;
    for (int i = 0; i < tau_steps; ++i) {
        cout << "\n--- Delay time " << (i+1) << "/" << tau_steps << ": tau = " << current_tau << " ---" << endl;
        
        tau_values.push_back(current_tau);
        
        // M1: Probe pulse only at time tau
        spins = ground_state;
        phonons = ground_phonons;
        cout << "  Computing M1 (probe at tau=" << current_tau << ")..." << endl;
        auto M1_trajectory = single_pulse_drive(polarization, current_tau,
                                                pulse_amp, pulse_width, pulse_freq,
                                                T_start, T_end, T_step, method);
        M1_trajectories.push_back(M1_trajectory);
        
        // M01: Pump at t=0 + Probe at t=tau
        spins = ground_state;
        phonons = ground_phonons;
        cout << "  Computing M01 (pump at 0 + probe at tau=" << current_tau << ")..." << endl;
        auto M01_trajectory = double_pulse_drive(polarization, 0.0, polarization, current_tau,
                                                 pulse_amp, pulse_width, pulse_freq,
                                                 T_start, T_end, T_step, method);
        M01_trajectories.push_back(M01_trajectory);
        
        current_tau += tau_step;
    }
    
#ifdef HDF5_ENABLED
    // Write to HDF5
    string hdf5_file = dir_name + "/pump_probe_spectroscopy.h5";
    cout << "\nWriting all data to HDF5 file: " << hdf5_file << endl;
    
    try {
        H5::H5File file(hdf5_file, H5F_ACC_TRUNC);
        
        // Create groups
        H5::Group params_group = file.createGroup("/parameters");
        H5::Group ref_group = file.createGroup("/reference");
        H5::Group scan_group = file.createGroup("/delay_scan");
        
        // Write parameters
        {
            hsize_t dims[1] = {1};
            H5::DataSpace scalar_space(1, dims);
            
            H5::DataSet ds = params_group.createDataSet("pulse_amp", H5::PredType::NATIVE_DOUBLE, scalar_space);
            ds.write(&pulse_amp, H5::PredType::NATIVE_DOUBLE);
            
            ds = params_group.createDataSet("pulse_width", H5::PredType::NATIVE_DOUBLE, scalar_space);
            ds.write(&pulse_width, H5::PredType::NATIVE_DOUBLE);
            
            ds = params_group.createDataSet("pulse_freq", H5::PredType::NATIVE_DOUBLE, scalar_space);
            ds.write(&pulse_freq, H5::PredType::NATIVE_DOUBLE);
            
            ds = params_group.createDataSet("polarization", H5::PredType::NATIVE_DOUBLE, scalar_space);
            ds.write(&polarization, H5::PredType::NATIVE_DOUBLE);
            
            ds = params_group.createDataSet("E_ground", H5::PredType::NATIVE_DOUBLE, scalar_space);
            ds.write(&E_ground, H5::PredType::NATIVE_DOUBLE);
        }
        
        // Write tau values
        {
            hsize_t dims[1] = {static_cast<hsize_t>(tau_values.size())};
            H5::DataSpace space(1, dims);
            H5::DataSet ds = scan_group.createDataSet("tau_values", H5::PredType::NATIVE_DOUBLE, space);
            ds.write(tau_values.data(), H5::PredType::NATIVE_DOUBLE);
        }
        
        // Write M0 reference trajectory
        {
            size_t n_times = M0_trajectory.size();
            vector<double> times(n_times);
            vector<double> M_antiferro(n_times * 3);
            vector<double> M_local(n_times * 3);
            vector<double> M_global(n_times * 3);
            vector<double> O_custom(n_times);
            
            for (size_t t = 0; t < n_times; ++t) {
                times[t] = M0_trajectory[t].first;
                for (int d = 0; d < 3; ++d) {
                    M_antiferro[t*3 + d] = M0_trajectory[t].second[0](d);
                    M_local[t*3 + d] = M0_trajectory[t].second[1](d);
                    M_global[t*3 + d] = M0_trajectory[t].second[2](d);
                }
                O_custom[t] = M0_trajectory[t].second[3](0);  // Custom order stored in x-component
            }
            
            hsize_t dims1[1] = {n_times};
            H5::DataSpace space1(1, dims1);
            H5::DataSet ds = ref_group.createDataSet("time", H5::PredType::NATIVE_DOUBLE, space1);
            ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);
            
            hsize_t dims2[2] = {n_times, 3};
            H5::DataSpace space2(2, dims2);
            ds = ref_group.createDataSet("M_antiferro", H5::PredType::NATIVE_DOUBLE, space2);
            ds.write(M_antiferro.data(), H5::PredType::NATIVE_DOUBLE);
            
            ds = ref_group.createDataSet("M_local", H5::PredType::NATIVE_DOUBLE, space2);
            ds.write(M_local.data(), H5::PredType::NATIVE_DOUBLE);
            
            ds = ref_group.createDataSet("M_global", H5::PredType::NATIVE_DOUBLE, space2);
            ds.write(M_global.data(), H5::PredType::NATIVE_DOUBLE);
            
            ds = ref_group.createDataSet("O_custom", H5::PredType::NATIVE_DOUBLE, space1);
            ds.write(O_custom.data(), H5::PredType::NATIVE_DOUBLE);
        }
        
        // Write delay-dependent trajectories
        for (size_t i = 0; i < tau_values.size(); ++i) {
            string group_name = "/delay_scan/tau_" + std::to_string(i);
            H5::Group tau_group = file.createGroup(group_name);
            
            // Write tau value
            {
                hsize_t dims[1] = {1};
                H5::DataSpace space(1, dims);
                H5::DataSet ds = tau_group.createDataSet("tau", H5::PredType::NATIVE_DOUBLE, space);
                ds.write(&tau_values[i], H5::PredType::NATIVE_DOUBLE);
            }
            
            // Write M1 trajectory
            {
                const auto& traj = M1_trajectories[i];
                size_t n_times = traj.size();
                vector<double> times(n_times);
                vector<double> M_antiferro(n_times * 3);
                vector<double> M_global(n_times * 3);
                vector<double> O_custom(n_times);
                
                for (size_t t = 0; t < n_times; ++t) {
                    times[t] = traj[t].first;
                    for (int d = 0; d < 3; ++d) {
                        M_antiferro[t*3 + d] = traj[t].second[0](d);
                        M_global[t*3 + d] = traj[t].second[2](d);
                    }
                    O_custom[t] = traj[t].second[3](0);
                }
                
                H5::Group m1_group = tau_group.createGroup("M1");
                hsize_t dims1[1] = {n_times};
                H5::DataSpace space1(1, dims1);
                H5::DataSet ds = m1_group.createDataSet("time", H5::PredType::NATIVE_DOUBLE, space1);
                ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);
                
                hsize_t dims2[2] = {n_times, 3};
                H5::DataSpace space2(2, dims2);
                ds = m1_group.createDataSet("M_antiferro", H5::PredType::NATIVE_DOUBLE, space2);
                ds.write(M_antiferro.data(), H5::PredType::NATIVE_DOUBLE);
                
                ds = m1_group.createDataSet("M_global", H5::PredType::NATIVE_DOUBLE, space2);
                ds.write(M_global.data(), H5::PredType::NATIVE_DOUBLE);
                
                ds = m1_group.createDataSet("O_custom", H5::PredType::NATIVE_DOUBLE, space1);
                ds.write(O_custom.data(), H5::PredType::NATIVE_DOUBLE);
            }
            
            // Write M01 trajectory
            {
                const auto& traj = M01_trajectories[i];
                size_t n_times = traj.size();
                vector<double> times(n_times);
                vector<double> M_antiferro(n_times * 3);
                vector<double> M_global(n_times * 3);
                vector<double> O_custom(n_times);
                
                for (size_t t = 0; t < n_times; ++t) {
                    times[t] = traj[t].first;
                    for (int d = 0; d < 3; ++d) {
                        M_antiferro[t*3 + d] = traj[t].second[0](d);
                        M_global[t*3 + d] = traj[t].second[2](d);
                    }
                    O_custom[t] = traj[t].second[3](0);
                }
                
                H5::Group m01_group = tau_group.createGroup("M01");
                hsize_t dims1[1] = {n_times};
                H5::DataSpace space1(1, dims1);
                H5::DataSet ds = m01_group.createDataSet("time", H5::PredType::NATIVE_DOUBLE, space1);
                ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);
                
                hsize_t dims2[2] = {n_times, 3};
                H5::DataSpace space2(2, dims2);
                ds = m01_group.createDataSet("M_antiferro", H5::PredType::NATIVE_DOUBLE, space2);
                ds.write(M_antiferro.data(), H5::PredType::NATIVE_DOUBLE);
                
                ds = m01_group.createDataSet("M_global", H5::PredType::NATIVE_DOUBLE, space2);
                ds.write(M_global.data(), H5::PredType::NATIVE_DOUBLE);
                
                ds = m01_group.createDataSet("O_custom", H5::PredType::NATIVE_DOUBLE, space1);
                ds.write(O_custom.data(), H5::PredType::NATIVE_DOUBLE);
            }
        }
        
        file.close();
        cout << "Successfully wrote all data to HDF5 file" << endl;
        
    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getDetailMsg() << endl;
    }
#else
    cout << "Warning: HDF5 support not enabled. Data not saved to HDF5 file." << endl;
    cout << "  Rebuild with -DHDF5_ENABLED to enable HDF5 output." << endl;
#endif
    
    // Restore ground state
    spins = ground_state;
    phonons = ground_phonons;
    
    cout << "\n==========================================" << endl;
    cout << "Pump-Probe Spectroscopy Complete!" << endl;
    cout << "Output directory: " << dir_name << endl;
    cout << "Total delay points: " << tau_steps << endl;
    cout << "==========================================" << endl;
}

void PhononLattice::pump_probe_spectroscopy_mpi(
    double polarization,
    double pulse_amp, double pulse_width, double pulse_freq,
    double tau_start, double tau_end, double tau_step,
    double T_start, double T_end, double T_step,
    const string& dir_name, const string& method) {
    
    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    std::filesystem::create_directories(dir_name);
    
    // Calculate total tau steps
    int tau_steps = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
    
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
        cout << "Tau points per rank: ~" << (tau_steps + mpi_size - 1) / mpi_size << endl;
    }
    
    // Use current configuration as ground state
    double E_ground = energy_density();
    if (rank == 0) {
        cout << "\n[1/4] Using current configuration as ground state..." << endl;
        cout << "  Ground state: E/N = " << E_ground << endl;
        save_positions(dir_name + "/positions.txt");
        save_spin_config(dir_name + "/spins_initial.txt");
    }
    
    // Backup ground state
    SpinConfig ground_state = spins;
    PhononState ground_phonons = phonons;
    
    // Compute M0 on rank 0
    MagTrajectory M0_trajectory;
    if (rank == 0) {
        cout << "\n[2/4] Computing reference trajectory (M0)..." << endl;
        M0_trajectory = single_pulse_drive(polarization, 0.0,
                                           pulse_amp, pulse_width, pulse_freq,
                                           T_start, T_end, T_step, method);
        spins = ground_state;
        phonons = ground_phonons;
    }
    
    // Distribute tau values across ranks
    vector<int> tau_counts(mpi_size), tau_offsets(mpi_size);
    int base_count = tau_steps / mpi_size;
    int remainder = tau_steps % mpi_size;
    
    for (int r = 0; r < mpi_size; ++r) {
        tau_counts[r] = base_count + (r < remainder ? 1 : 0);
        tau_offsets[r] = (r == 0) ? 0 : tau_offsets[r-1] + tau_counts[r-1];
    }
    
    int my_start = tau_offsets[rank];
    int my_count = tau_counts[rank];
    
    if (rank == 0) {
        cout << "\n[3/4] Parallel delay scan..." << endl;
    }
    
    // Each rank computes its subset of tau values
    vector<double> my_tau_values(my_count);
    vector<MagTrajectory> my_M1_trajectories(my_count);
    vector<MagTrajectory> my_M01_trajectories(my_count);
    
    for (int i = 0; i < my_count; ++i) {
        int global_idx = my_start + i;
        double current_tau = tau_start + global_idx * tau_step;
        my_tau_values[i] = current_tau;
        
        cout << "[Rank " << rank << "] tau = " << current_tau 
             << " (" << (i+1) << "/" << my_count << ")" << endl;
        
        // M1: Probe only
        spins = ground_state;
        phonons = ground_phonons;
        my_M1_trajectories[i] = single_pulse_drive(polarization, current_tau,
                                                   pulse_amp, pulse_width, pulse_freq,
                                                   T_start, T_end, T_step, method);
        
        // M01: Pump + Probe
        spins = ground_state;
        phonons = ground_phonons;
        my_M01_trajectories[i] = double_pulse_drive(polarization, 0.0, polarization, current_tau,
                                                    pulse_amp, pulse_width, pulse_freq,
                                                    T_start, T_end, T_step, method);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Gather results to rank 0
    if (rank == 0) {
        cout << "\n[4/4] Gathering results and writing output..." << endl;
        
        // Collect all trajectories
        vector<double> all_tau_values(tau_steps);
        vector<MagTrajectory> all_M1_trajectories(tau_steps);
        vector<MagTrajectory> all_M01_trajectories(tau_steps);
        
        // Copy rank 0 data
        for (int i = 0; i < my_count; ++i) {
            all_tau_values[i] = my_tau_values[i];
            all_M1_trajectories[i] = my_M1_trajectories[i];
            all_M01_trajectories[i] = my_M01_trajectories[i];
        }
        
        // Receive from other ranks
        for (int r = 1; r < mpi_size; ++r) {
            for (int i = 0; i < tau_counts[r]; ++i) {
                int global_idx = tau_offsets[r] + i;
                double tau_val;
                MPI_Recv(&tau_val, 1, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                all_tau_values[global_idx] = tau_val;
                
                // Receive M1 trajectory size and data
                int traj_size;
                MPI_Recv(&traj_size, 1, MPI_INT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                vector<double> m1_buffer(traj_size * 7);  // t, M_antiferro(3), M_local(3)
                MPI_Recv(m1_buffer.data(), traj_size * 7, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                all_M1_trajectories[global_idx].resize(traj_size);
                for (int t = 0; t < traj_size; ++t) {
                    all_M1_trajectories[global_idx][t].first = m1_buffer[t*7];
                    all_M1_trajectories[global_idx][t].second[0] = Eigen::Vector3d(
                        m1_buffer[t*7+1], m1_buffer[t*7+2], m1_buffer[t*7+3]);
                    all_M1_trajectories[global_idx][t].second[1] = Eigen::Vector3d(
                        m1_buffer[t*7+4], m1_buffer[t*7+5], m1_buffer[t*7+6]);
                    all_M1_trajectories[global_idx][t].second[2] = Eigen::Vector3d::Zero();
                }
                
                // Receive M01 trajectory
                MPI_Recv(&traj_size, 1, MPI_INT, r, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                vector<double> m01_buffer(traj_size * 7);
                MPI_Recv(m01_buffer.data(), traj_size * 7, MPI_DOUBLE, r, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                all_M01_trajectories[global_idx].resize(traj_size);
                for (int t = 0; t < traj_size; ++t) {
                    all_M01_trajectories[global_idx][t].first = m01_buffer[t*7];
                    all_M01_trajectories[global_idx][t].second[0] = Eigen::Vector3d(
                        m01_buffer[t*7+1], m01_buffer[t*7+2], m01_buffer[t*7+3]);
                    all_M01_trajectories[global_idx][t].second[1] = Eigen::Vector3d(
                        m01_buffer[t*7+4], m01_buffer[t*7+5], m01_buffer[t*7+6]);
                    all_M01_trajectories[global_idx][t].second[2] = Eigen::Vector3d::Zero();
                }
            }
        }
        
#ifdef HDF5_ENABLED
        // Write HDF5 (same as serial version)
        string hdf5_file = dir_name + "/pump_probe_spectroscopy.h5";
        cout << "Writing to: " << hdf5_file << endl;
        
        try {
            H5::H5File file(hdf5_file, H5F_ACC_TRUNC);
            
            H5::Group params_group = file.createGroup("/parameters");
            H5::Group ref_group = file.createGroup("/reference");
            H5::Group scan_group = file.createGroup("/delay_scan");
            
            // Write parameters
            {
                hsize_t dims[1] = {1};
                H5::DataSpace scalar_space(1, dims);
                
                H5::DataSet ds = params_group.createDataSet("pulse_amp", H5::PredType::NATIVE_DOUBLE, scalar_space);
                ds.write(&pulse_amp, H5::PredType::NATIVE_DOUBLE);
                
                ds = params_group.createDataSet("pulse_width", H5::PredType::NATIVE_DOUBLE, scalar_space);
                ds.write(&pulse_width, H5::PredType::NATIVE_DOUBLE);
                
                ds = params_group.createDataSet("pulse_freq", H5::PredType::NATIVE_DOUBLE, scalar_space);
                ds.write(&pulse_freq, H5::PredType::NATIVE_DOUBLE);
            }
            
            // Write tau values
            {
                hsize_t dims[1] = {static_cast<hsize_t>(all_tau_values.size())};
                H5::DataSpace space(1, dims);
                H5::DataSet ds = scan_group.createDataSet("tau_values", H5::PredType::NATIVE_DOUBLE, space);
                ds.write(all_tau_values.data(), H5::PredType::NATIVE_DOUBLE);
            }
            
            // Write M0 reference
            {
                size_t n_times = M0_trajectory.size();
                vector<double> times(n_times);
                vector<double> M_antiferro(n_times * 3);
                vector<double> M_global(n_times * 3);
                vector<double> O_custom(n_times);
                
                for (size_t t = 0; t < n_times; ++t) {
                    times[t] = M0_trajectory[t].first;
                    for (int d = 0; d < 3; ++d) {
                        M_antiferro[t*3 + d] = M0_trajectory[t].second[0](d);
                        M_global[t*3 + d] = M0_trajectory[t].second[2](d);
                    }
                    O_custom[t] = M0_trajectory[t].second[3](0);
                }
                
                hsize_t dims1[1] = {n_times};
                H5::DataSpace space1(1, dims1);
                H5::DataSet ds = ref_group.createDataSet("time", H5::PredType::NATIVE_DOUBLE, space1);
                ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);
                
                hsize_t dims2[2] = {n_times, 3};
                H5::DataSpace space2(2, dims2);
                ds = ref_group.createDataSet("M_antiferro", H5::PredType::NATIVE_DOUBLE, space2);
                ds.write(M_antiferro.data(), H5::PredType::NATIVE_DOUBLE);
                
                ds = ref_group.createDataSet("M_global", H5::PredType::NATIVE_DOUBLE, space2);
                ds.write(M_global.data(), H5::PredType::NATIVE_DOUBLE);
                
                ds = ref_group.createDataSet("O_custom", H5::PredType::NATIVE_DOUBLE, space1);
                ds.write(O_custom.data(), H5::PredType::NATIVE_DOUBLE);
            }
            
            // Write delay-dependent data
            for (size_t i = 0; i < all_tau_values.size(); ++i) {
                string group_name = "/delay_scan/tau_" + std::to_string(i);
                H5::Group tau_group = file.createGroup(group_name);
                
                {
                    hsize_t dims[1] = {1};
                    H5::DataSpace space(1, dims);
                    H5::DataSet ds = tau_group.createDataSet("tau", H5::PredType::NATIVE_DOUBLE, space);
                    ds.write(&all_tau_values[i], H5::PredType::NATIVE_DOUBLE);
                }
                
                // M1
                {
                    const auto& traj = all_M1_trajectories[i];
                    size_t n_times = traj.size();
                    vector<double> times(n_times);
                    vector<double> M_af(n_times * 3);
                    vector<double> M_global(n_times * 3);
                    vector<double> O_custom(n_times);
                    
                    for (size_t t = 0; t < n_times; ++t) {
                        times[t] = traj[t].first;
                        for (int d = 0; d < 3; ++d) {
                            M_af[t*3 + d] = traj[t].second[0](d);
                            M_global[t*3 + d] = traj[t].second[2](d);
                        }
                        O_custom[t] = traj[t].second[3](0);
                    }
                    
                    H5::Group m1_group = tau_group.createGroup("M1");
                    hsize_t dims1[1] = {n_times};
                    H5::DataSpace space1(1, dims1);
                    H5::DataSet ds = m1_group.createDataSet("time", H5::PredType::NATIVE_DOUBLE, space1);
                    ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);
                    
                    hsize_t dims2[2] = {n_times, 3};
                    H5::DataSpace space2(2, dims2);
                    ds = m1_group.createDataSet("M_antiferro", H5::PredType::NATIVE_DOUBLE, space2);
                    ds.write(M_af.data(), H5::PredType::NATIVE_DOUBLE);
                    
                    ds = m1_group.createDataSet("M_global", H5::PredType::NATIVE_DOUBLE, space2);
                    ds.write(M_global.data(), H5::PredType::NATIVE_DOUBLE);
                    
                    ds = m1_group.createDataSet("O_custom", H5::PredType::NATIVE_DOUBLE, space1);
                    ds.write(O_custom.data(), H5::PredType::NATIVE_DOUBLE);
                }
                
                // M01
                {
                    const auto& traj = all_M01_trajectories[i];
                    size_t n_times = traj.size();
                    vector<double> times(n_times);
                    vector<double> M_af(n_times * 3);
                    vector<double> M_global(n_times * 3);
                    vector<double> O_custom(n_times);
                    
                    for (size_t t = 0; t < n_times; ++t) {
                        times[t] = traj[t].first;
                        for (int d = 0; d < 3; ++d) {
                            M_af[t*3 + d] = traj[t].second[0](d);
                            M_global[t*3 + d] = traj[t].second[2](d);
                        }
                        O_custom[t] = traj[t].second[3](0);
                    }
                    
                    H5::Group m01_group = tau_group.createGroup("M01");
                    hsize_t dims1[1] = {n_times};
                    H5::DataSpace space1(1, dims1);
                    H5::DataSet ds = m01_group.createDataSet("time", H5::PredType::NATIVE_DOUBLE, space1);
                    ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);
                    
                    hsize_t dims2[2] = {n_times, 3};
                    H5::DataSpace space2(2, dims2);
                    ds = m01_group.createDataSet("M_antiferro", H5::PredType::NATIVE_DOUBLE, space2);
                    ds.write(M_af.data(), H5::PredType::NATIVE_DOUBLE);
                    
                    ds = m01_group.createDataSet("M_global", H5::PredType::NATIVE_DOUBLE, space2);
                    ds.write(M_global.data(), H5::PredType::NATIVE_DOUBLE);
                    
                    ds = m01_group.createDataSet("O_custom", H5::PredType::NATIVE_DOUBLE, space1);
                    ds.write(O_custom.data(), H5::PredType::NATIVE_DOUBLE);
                }
            }
            
            file.close();
            cout << "Successfully wrote HDF5 file" << endl;
            
        } catch (H5::Exception& e) {
            std::cerr << "HDF5 Error: " << e.getDetailMsg() << endl;
        }
#else
        cout << "Warning: HDF5 not enabled" << endl;
#endif
        
        cout << "\n==========================================" << endl;
        cout << "Pump-Probe Spectroscopy Complete!" << endl;
        cout << "==========================================" << endl;
        
    } else {
        // Send data to rank 0
        for (int i = 0; i < my_count; ++i) {
            MPI_Send(&my_tau_values[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            
            // Send M1 trajectory
            int traj_size = static_cast<int>(my_M1_trajectories[i].size());
            MPI_Send(&traj_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            
            vector<double> m1_buffer(traj_size * 7);
            for (int t = 0; t < traj_size; ++t) {
                m1_buffer[t*7] = my_M1_trajectories[i][t].first;
                m1_buffer[t*7+1] = my_M1_trajectories[i][t].second[0](0);
                m1_buffer[t*7+2] = my_M1_trajectories[i][t].second[0](1);
                m1_buffer[t*7+3] = my_M1_trajectories[i][t].second[0](2);
                m1_buffer[t*7+4] = my_M1_trajectories[i][t].second[1](0);
                m1_buffer[t*7+5] = my_M1_trajectories[i][t].second[1](1);
                m1_buffer[t*7+6] = my_M1_trajectories[i][t].second[1](2);
            }
            MPI_Send(m1_buffer.data(), traj_size * 7, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            
            // Send M01 trajectory
            traj_size = static_cast<int>(my_M01_trajectories[i].size());
            MPI_Send(&traj_size, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
            
            vector<double> m01_buffer(traj_size * 7);
            for (int t = 0; t < traj_size; ++t) {
                m01_buffer[t*7] = my_M01_trajectories[i][t].first;
                m01_buffer[t*7+1] = my_M01_trajectories[i][t].second[0](0);
                m01_buffer[t*7+2] = my_M01_trajectories[i][t].second[0](1);
                m01_buffer[t*7+3] = my_M01_trajectories[i][t].second[0](2);
                m01_buffer[t*7+4] = my_M01_trajectories[i][t].second[1](0);
                m01_buffer[t*7+5] = my_M01_trajectories[i][t].second[1](1);
                m01_buffer[t*7+6] = my_M01_trajectories[i][t].second[1](2);
            }
            MPI_Send(m01_buffer.data(), traj_size * 7, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        }
    }
    
    // Restore ground state
    spins = ground_state;
    phonons = ground_phonons;
}

// Explicit template instantiation
template void PhononLattice::integrate_ode_system(
    std::function<void(const PhononLattice::ODEState&, PhononLattice::ODEState&, double)>,
    PhononLattice::ODEState&, double, double, double,
    std::function<void(const PhononLattice::ODEState&, double)>,
    const std::string&, bool, double, double);
