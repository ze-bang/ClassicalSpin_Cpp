/**
 * @file strain_phonon_lattice.cpp
 * @brief Implementation of spin-strain coupled honeycomb lattice with magnetoelastic coupling
 * 
 * Implements honeycomb lattice with:
 * - Generic NN, 2nd NN, and 3rd NN spin interactions (Kitaev-Heisenberg-Γ-Γ')
 * - Strain field phonons with elastic Hamiltonian
 * - Magnetoelastic coupling through A1g and Eg symmetry channels of D3d
 * 
 * Magnetoelastic coupling (following D3d point group basis functions):
 * H_c^{A1g} = λ_{A1g} Σ_r (ε_xx + ε_yy) {(J+K)f_K^{A1g} + J f_J^{A1g} + Γ f_Γ^{A1g}}
 * H_c^{Eg} = λ_{Eg} Σ_r {(ε_xx - ε_yy)[(J+K)f_K^{Eg,1} + J f_J^{Eg,1} + Γ f_Γ^{Eg,1}]
 *                       + 2ε_xy[(J+K)f_K^{Eg,2} + J f_J^{Eg,2} + Γ f_Γ^{Eg,2}]}
 * 
 * Spin basis functions (D3d irreps, M_x/y/z = x/y/z-type Kitaev bonds):
 * 
 * A1g irrep:
 *   f_K^{A1g} = S^x·S^x_{M_x} + S^y·S^y_{M_y} + S^z·S^z_{M_z}
 *   f_J^{A1g} = S^x·S^x_{M_{y,z}} + S^y·S^y_{M_{x,z}} + S^z·S^z_{M_{x,y}}
 *              (non-Kitaev components: α≠γ on γ-bond)
 *   f_Γ^{A1g} = (S^y·S^z + S^z·S^y)_{M_x} + (S^x·S^z + S^z·S^x)_{M_y} + (S^x·S^y + S^y·S^x)_{M_z}
 * 
 * Eg irrep:
 *   f_K^{Eg,1} = S^x·S^x_{M_x} + S^y·S^y_{M_y} - 2·S^z·S^z_{M_z}
 *   f_K^{Eg,2} = √3·(S^x·S^x_{M_x} - S^y·S^y_{M_y})
 *   f_J^{Eg,1} = S^x·S^x_{M_{y,z}} + S^y·S^y_{M_{x,z}} - 2·S^z·S^z_{M_{x,y}}
 *   f_J^{Eg,2} = √3·(S^x·S^x_{M_{y,z}} - S^y·S^y_{M_{x,z}})
 *   f_Γ^{Eg,1} = (S^y·S^z + S^z·S^y)_{M_x} + (S^x·S^z + S^z·S^x)_{M_y} - 2·(S^x·S^y + S^y·S^x)_{M_z}
 *   f_Γ^{Eg,2} = √3·[(S^y·S^z + S^z·S^y)_{M_x} - (S^x·S^z + S^z·S^x)_{M_y}]
 */

#include "classical_spin/lattice/strain_phonon_lattice.h"
#include <boost/numeric/odeint.hpp>
#include <fstream>
#include <iomanip>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <mpi.h>

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

namespace odeint = boost::numeric::odeint;

// ============================================================
// CONSTRUCTOR
// ============================================================

StrainPhononLattice::StrainPhononLattice(const UnitCell& uc, size_t d1, size_t d2, size_t d3, float spin_l)
    : unit_cell(uc), N_atoms(uc.N_atoms), dim1(d1), dim2(d2), dim3(d3), spin_length(spin_l)
{
    lattice_size = N_atoms * dim1 * dim2 * dim3;
    state_size = spin_dim * lattice_size + StrainState::N_DOF;
    
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
    site_hexagons.resize(lattice_size);
    
    // Extract sublattice frames from UnitCell
    sublattice_frames.resize(N_atoms);
    for (size_t a = 0; a < N_atoms; ++a) {
        sublattice_frames[a] = uc.sublattice_frames[a];
    }
    
    // Initialize random number generator
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(static_cast<unsigned int>(seed));
    uniform_dist = std::uniform_real_distribution<double>(0.0, 1.0);
    
    cout << "StrainPhononLattice created with dimensions: "
         << dim1 << " x " << dim2 << " x " << dim3 << endl;
    cout << "Total spin sites: " << lattice_size << endl;
    cout << "Strain DOF: " << StrainState::N_DOF << " (ε_xx, ε_yy, ε_xy, V_xx, V_yy, V_xy) × 3 bonds" << endl;
    cout << "Total ODE state size: " << state_size << endl;
    
    // Build lattice positions and interactions from UnitCell
    const auto& a1 = uc.lattice_vectors[0];
    const auto& a2 = uc.lattice_vectors[1];
    const auto& a3 = uc.lattice_vectors[2];
    
    // Build site positions and fields
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                for (size_t a = 0; a < N_atoms; ++a) {
                    size_t site = flatten_index(i, j, k, a);
                    site_positions[site] = uc.lattice_pos[a] + double(i)*a1 + double(j)*a2 + double(k)*a3;
                    
                    // Set field from UnitCell (field is VectorXd, extract 3D)
                    field[site] = uc.field[a].head<3>();
                }
            }
        }
    }
    
    // Build interactions from UnitCell bilinear interactions
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                for (size_t a = 0; a < N_atoms; ++a) {
                    size_t site = flatten_index(i, j, k, a);
                    
                    auto range = uc.bilinear_interaction.equal_range(a);
                    for (auto it = range.first; it != range.second; ++it) {
                        const auto& bi = it->second;
                        
                        int pi = static_cast<int>(i) + bi.offset[0];
                        int pj = static_cast<int>(j) + bi.offset[1];
                        int pk = static_cast<int>(k) + bi.offset[2];
                        size_t partner = flatten_periodic(pi, pj, pk, bi.partner);
                        
                        if (bi.bond_type >= 0) {
                            // NN interaction with bond type
                            nn_interaction[site].push_back(bi.interaction);
                            nn_partners[site].push_back(partner);
                            nn_bond_types[site].push_back(bi.bond_type);
                            
                            nn_interaction[partner].push_back(bi.interaction.transpose());
                            nn_partners[partner].push_back(site);
                            nn_bond_types[partner].push_back(bi.bond_type);
                        } else {
                            // J2 or J3 (classify by sublattice)
                            if (a == static_cast<size_t>(bi.partner)) {
                                // Same sublattice -> J2
                                if (partner > site) {
                                    j2_interaction[site].push_back(bi.interaction);
                                    j2_partners[site].push_back(partner);
                                    j2_interaction[partner].push_back(bi.interaction.transpose());
                                    j2_partners[partner].push_back(site);
                                }
                            } else {
                                // Different sublattice -> J3
                                if (partner > site) {
                                    j3_interaction[site].push_back(bi.interaction);
                                    j3_partners[site].push_back(partner);
                                    j3_interaction[partner].push_back(bi.interaction.transpose());
                                    j3_partners[partner].push_back(site);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Initialize random spins
    init_random();
    
    cout << "StrainPhononLattice initialization complete!" << endl;
}

// ============================================================
// PARAMETER SETTING
// ============================================================

void StrainPhononLattice::set_parameters(const MagnetoelasticParams& me_params,
                                         const ElasticParams& el_params,
                                         const StrainDriveParams& dr_params) {
    magnetoelastic_params = me_params;
    elastic_params = el_params;
    drive_params = dr_params;
    
    // Clear hexagons (interactions are already built from UnitCell in constructor)
    for (size_t i = 0; i < lattice_size; ++i) {
        site_hexagons[i].clear();
    }
    hexagons.clear();
    
    // Build hexagonal plaquettes for ring exchange (matching PhononLattice exactly)
    if (std::abs(magnetoelastic_params.J7) > 1e-12) {
        // Hexagon definition (matching PhononLattice CODE exactly):
        //   0: A(i,j,k)      - central A site
        //   1: B(i,j,k)      - z-bond neighbor (same unit cell)
        //   2: A(i,j+1,k)    - next unit cell in j direction
        //   3: B(i+1,j,k)    - B site in (i+1,j,k) unit cell
        //   4: A(i+1,j,k)    - A site in (i+1,j,k) unit cell
        //   5: B(i+1,j-1,k)  - B site in (i+1,j-1,k) unit cell
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    std::array<size_t, 6> hex;
                    hex[0] = flatten_index(i, j, k, 0);           // A(i,j,k)
                    hex[1] = flatten_index(i, j, k, 1);           // B(i,j,k)
                    hex[2] = flatten_periodic(i, j+1, k, 0);      // A(i,j+1,k)
                    hex[3] = flatten_periodic(i+1, j, k, 1);      // B(i+1,j,k)
                    hex[4] = flatten_periodic(i+1, j, k, 0);      // A(i+1,j,k)
                    hex[5] = flatten_periodic(i+1, j-1, k, 1);    // B(i+1,j-1,k)
                    
                    size_t hex_idx = hexagons.size();
                    hexagons.push_back(hex);
                    
                    // Register this hexagon for each site
                    for (size_t pos = 0; pos < 6; ++pos) {
                        site_hexagons[hex[pos]].push_back({hex_idx, pos});
                    }
                }
            }
        }
        cout << "  Built " << hexagons.size() << " hexagonal plaquettes for ring exchange" << endl;
    }
    
    cout << "Set StrainPhononLattice parameters:" << endl;
    cout << "  J=" << me_params.J << ", K=" << me_params.K 
         << ", Γ=" << me_params.Gamma << ", Γ'=" << me_params.Gammap << endl;
    cout << "  J2_A=" << me_params.J2_A << ", J2_B=" << me_params.J2_B << endl;
    cout << "  J3=" << me_params.J3 << ", J7=" << me_params.J7 << endl;
    cout << "  λ_A1g=" << me_params.lambda_A1g << ", λ_Eg=" << me_params.lambda_Eg << endl;
    cout << "  C11=" << el_params.C11 << ", C12=" << el_params.C12 << ", C44=" << el_params.C44 << endl;
}

void StrainPhononLattice::set_nn_coupling(size_t site1, size_t site2, 
                                          const SpinMatrix& J, int bond_type) {
    // Find the neighbor index
    for (size_t n = 0; n < nn_partners[site1].size(); ++n) {
        if (nn_partners[site1][n] == site2) {
            nn_interaction[site1][n] = J;
            nn_bond_types[site1][n] = bond_type;
            return;
        }
    }
}

void StrainPhononLattice::set_j2_coupling(size_t site, size_t partner, const SpinMatrix& J) {
    j2_partners[site].push_back(partner);
    j2_interaction[site].push_back(J);
}

void StrainPhononLattice::set_j3_coupling(size_t site, size_t partner, const SpinMatrix& J) {
    j3_partners[site].push_back(partner);
    j3_interaction[site].push_back(J);
}

void StrainPhononLattice::set_field(size_t site, const SpinVector& B) {
    field[site] = B;
}

void StrainPhononLattice::set_uniform_field(const SpinVector& B) {
    for (size_t i = 0; i < lattice_size; ++i) {
        field[i] = B;
    }
}

// ============================================================
// INITIALIZATION
// ============================================================

void StrainPhononLattice::init_random() {
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        double x = normal_dist(rng);
        double y = normal_dist(rng);
        double z = normal_dist(rng);
        double norm = std::sqrt(x*x + y*y + z*z);
        spins[i] = Eigen::Vector3d(x/norm, y/norm, z/norm) * spin_length;
    }
}

void StrainPhononLattice::init_neel() {
    for (size_t i = 0; i < lattice_size; ++i) {
        size_t atom = i % N_atoms;
        if (atom == 0) {
            spins[i] = Eigen::Vector3d(0, 0, spin_length);
        } else {
            spins[i] = Eigen::Vector3d(0, 0, -spin_length);
        }
    }
}

void StrainPhononLattice::init_zigzag() {
    // Zigzag order: ferromagnetic chains along one direction, antiferromagnetic between chains
    for (size_t k = 0; k < dim3; ++k) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t i = 0; i < dim1; ++i) {
                double sign = (j % 2 == 0) ? 1.0 : -1.0;
                size_t idx_A = flatten_index(i, j, k, 0);
                size_t idx_B = flatten_index(i, j, k, 1);
                spins[idx_A] = Eigen::Vector3d(0, 0, sign * spin_length);
                spins[idx_B] = Eigen::Vector3d(0, 0, sign * spin_length);
            }
        }
    }
}

void StrainPhononLattice::set_spin(size_t site, const SpinVector& s) {
    spins[site] = s;
}

// ============================================================
// TIME-DEPENDENT PARAMETERS
// ============================================================

double StrainPhononLattice::get_effective_J7(double t) const {
    // J7 modulated by the DEVIATION of Eg strain from equilibrium
    // 
    // The ring exchange J7 depends on bond lengths, which are modulated by strain.
    // For Eg symmetry strain (shear modes), the bond length changes as:
    //   δr/r ~ δε_Eg = |ε_Eg(t) - ε_Eg^eq|
    //
    // The exchange integral has exponential distance dependence:
    //   J(r) ~ exp(-r/ξ)
    // So J7 ~ J7_0 * exp(-6*δr/ξ) for 6-site ring (approximately)
    //
    // For small strain deviation, we can expand:
    //   J7(δε) ≈ J7_0 * (1 - γ * |δε_Eg|)^n
    // where γ is a coupling constant and n depends on the geometry.
    //
    // Using deviation from equilibrium:
    //   J7(t) = J7_0 * (1 - γ*|δε_Eg|/4)^4 * (1 + γ*|δε_Eg|/2)^2
    // This ensures J7 = J7_0 at equilibrium (when strain = strain_equilibrium)
    
    double J7_0 = magnetoelastic_params.J7;
    double gamma = magnetoelastic_params.gamma_J7;
    
    // If no coupling, return bare J7
    if (std::abs(gamma) < 1e-12) {
        return J7_0;
    }
    
    // Get the Eg strain DEVIATION from equilibrium
    // δε_Eg = |ε_Eg(t) - ε_Eg^eq| computed component-wise then averaged
    double delta_eps_Eg_norm = 0.0;
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        // Current Eg strain components
        double Eg1 = strain.epsilon_xx[b] - strain.epsilon_yy[b];
        double Eg2 = 2.0 * strain.epsilon_xy[b];
        
        // Equilibrium Eg strain components
        double Eg1_eq = strain_equilibrium.epsilon_xx[b] - strain_equilibrium.epsilon_yy[b];
        double Eg2_eq = 2.0 * strain_equilibrium.epsilon_xy[b];
        
        // Deviation from equilibrium
        double dEg1 = Eg1 - Eg1_eq;
        double dEg2 = Eg2 - Eg2_eq;
        
        delta_eps_Eg_norm += std::sqrt(dEg1 * dEg1 + dEg2 * dEg2);
    }
    delta_eps_Eg_norm /= StrainState::N_BONDS;
    
    // Compute the modulation factors using strain deviation
    double x = gamma * delta_eps_Eg_norm;
    double factor1 = 1.0 - x / 4.0;  // (1 - γ*|δε_Eg|/4)
    double factor2 = 1.0 + x / 2.0;  // (1 + γ*|δε_Eg|/2)
    
    // Ensure factors don't go negative (would give unphysical results)
    if (factor1 < 0.0) factor1 = 0.0;
    if (factor2 < 0.0) factor2 = 0.0;
    
    // J7(t) = J7_0 * factor1^4 * factor2^2
    double J7_t = J7_0 * std::pow(factor1, 4) * std::pow(factor2, 2);
    
    return J7_t;
}

double StrainPhononLattice::get_ring_exchange_normalized() const {
    // Compute the ring exchange energy without the J7 prefactor
    // E_ring = (1/6) Σ_{hex} Σ_{cyclic perms} [...]
    // This is needed for computing ∂H_J7/∂ε = (dJ7/dε) * E_ring
    
    if (hexagons.empty()) {
        return 0.0;
    }
    
    double E = 0.0;
    double prefactor = 1.0 / 6.0;  // No J7 factor
    
    for (const auto& hex : hexagons) {
        const Eigen::Vector3d& S0 = spins[hex[0]];
        const Eigen::Vector3d& S1 = spins[hex[1]];
        const Eigen::Vector3d& S2 = spins[hex[2]];
        const Eigen::Vector3d& S3 = spins[hex[3]];
        const Eigen::Vector3d& S4 = spins[hex[4]];
        const Eigen::Vector3d& S5 = spins[hex[5]];
        
        double d01 = S0.dot(S1), d02 = S0.dot(S2), d03 = S0.dot(S3);
        double d04 = S0.dot(S4), d05 = S0.dot(S5);
        double d12 = S1.dot(S2), d13 = S1.dot(S3), d14 = S1.dot(S4), d15 = S1.dot(S5);
        double d23 = S2.dot(S3), d24 = S2.dot(S4), d25 = S2.dot(S5);
        double d34 = S3.dot(S4), d35 = S3.dot(S5);
        double d45 = S4.dot(S5);
        
        double term0 = 2.0*d01*d23*d45 - 6.0*d02*d13*d45 + 3.0*d03*d12*d45 
                     + 3.0*d02*d14*d35 - d03*d14*d25;
        double term1 = 2.0*d12*d34*d05 - 6.0*d13*d24*d05 + 3.0*d14*d23*d05 
                     + 3.0*d13*d25*d04 - d14*d25*d03;
        double term2 = 2.0*d23*d45*d01 - 6.0*d24*d35*d01 + 3.0*d25*d34*d01 
                     + 3.0*d24*d03*d15 - d25*d03*d14;
        double term3 = 2.0*d34*d05*d12 - 6.0*d35*d04*d12 + 3.0*d03*d45*d12 
                     + 3.0*d35*d14*d02 - d03*d14*d25;
        double term4 = 2.0*d45*d01*d23 - 6.0*d04*d15*d23 + 3.0*d14*d05*d23 
                     + 3.0*d04*d25*d13 - d14*d25*d03;
        double term5 = 2.0*d05*d12*d34 - 6.0*d15*d02*d34 + 3.0*d25*d01*d34 
                     + 3.0*d15*d03*d24 - d25*d03*d14;
        
        E += prefactor * (term0 + term1 + term2 + term3 + term4 + term5);
    }
    
    return E;
}

void StrainPhononLattice::get_dJ7_deps(double* dJ7_deps_xx, double* dJ7_deps_yy, 
                                       double* dJ7_deps_xy) const {
    // Compute ∂J7/∂ε_xx, ∂J7/∂ε_yy, ∂J7/∂ε_xy for each bond type
    //
    // J7 = J7_0 * f1^4 * f2^2
    // where f1 = 1 - γ*r/4, f2 = 1 + γ*r/2
    // and r = |δε_Eg| = (1/N_bonds) Σ_b sqrt((Eg1_b - Eg1_eq)² + (Eg2_b - Eg2_eq)²)
    //
    // Chain rule:
    // ∂J7/∂ε_xx = (dJ7/dr) * (∂r/∂ε_xx)
    //
    // dJ7/dr = J7_0 * [4*f1^3*(-γ/4)*f2^2 + f1^4*2*f2*(γ/2)]
    //        = J7_0 * f1^3 * f2 * [-γ*f2 + γ*f1]
    //        = J7_0 * γ * f1^3 * f2 * (f1 - f2)
    //        = J7_0 * γ * f1^3 * f2 * (-3γr/4)
    //
    // ∂r/∂ε_xx = (1/N_bonds) * (Eg1 - Eg1_eq) / |δε_Eg|  (where Eg1 = ε_xx - ε_yy)
    
    double J7_0 = magnetoelastic_params.J7;
    double gamma = magnetoelastic_params.gamma_J7;
    
    // Initialize to zero
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        dJ7_deps_xx[b] = 0.0;
        dJ7_deps_yy[b] = 0.0;
        dJ7_deps_xy[b] = 0.0;
    }
    
    // If no J7-strain coupling, return zeros
    if (std::abs(gamma) < 1e-12 || std::abs(J7_0) < 1e-12) {
        return;
    }
    
    // Compute Eg strain deviation for each bond and the total norm
    double dEg1[StrainState::N_BONDS], dEg2[StrainState::N_BONDS];
    double norm_b[StrainState::N_BONDS];
    double r = 0.0;  // Average norm
    
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        double Eg1 = strain.epsilon_xx[b] - strain.epsilon_yy[b];
        double Eg2 = 2.0 * strain.epsilon_xy[b];
        double Eg1_eq = strain_equilibrium.epsilon_xx[b] - strain_equilibrium.epsilon_yy[b];
        double Eg2_eq = 2.0 * strain_equilibrium.epsilon_xy[b];
        
        dEg1[b] = Eg1 - Eg1_eq;
        dEg2[b] = Eg2 - Eg2_eq;
        norm_b[b] = std::sqrt(dEg1[b] * dEg1[b] + dEg2[b] * dEg2[b]);
        r += norm_b[b];
    }
    r /= StrainState::N_BONDS;
    
    // If at equilibrium, derivative is zero (to avoid division by zero)
    if (r < 1e-12) {
        return;
    }
    
    // Compute dJ7/dr
    double x = gamma * r;
    double f1 = 1.0 - x / 4.0;
    double f2 = 1.0 + x / 2.0;
    
    // Clamp factors
    if (f1 < 0.0) f1 = 0.0;
    if (f2 < 0.0) f2 = 0.0;
    
    // dJ7/dr = J7_0 * [4*f1^3*(-γ/4)*f2^2 + f1^4*2*f2*(γ/2)]
    //        = J7_0 * γ * f1^3 * f2 * (f1 - f2)
    double dJ7_dr = J7_0 * gamma * std::pow(f1, 3) * f2 * (f1 - f2);
    
    // For each bond type, compute ∂r/∂ε and then ∂J7/∂ε
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        if (norm_b[b] < 1e-12) continue;  // Skip if this bond is at equilibrium
        
        // ∂r/∂ε_xx = (1/N_bonds) * (∂|δε_Eg|_b/∂ε_xx)
        //          = (1/N_bonds) * (dEg1_b / |δε_Eg|_b) * (∂Eg1/∂ε_xx)
        //          = (1/N_bonds) * (dEg1_b / norm_b) * 1
        double dr_deps_xx = (1.0 / StrainState::N_BONDS) * (dEg1[b] / norm_b[b]);
        
        // ∂r/∂ε_yy = (1/N_bonds) * (dEg1_b / norm_b) * (-1)
        double dr_deps_yy = (1.0 / StrainState::N_BONDS) * (dEg1[b] / norm_b[b]) * (-1.0);
        
        // ∂r/∂ε_xy = (1/N_bonds) * (dEg2_b / norm_b) * 2
        double dr_deps_xy = (1.0 / StrainState::N_BONDS) * (dEg2[b] / norm_b[b]) * 2.0;
        
        dJ7_deps_xx[b] = dJ7_dr * dr_deps_xx;
        dJ7_deps_yy[b] = dJ7_dr * dr_deps_yy;
        dJ7_deps_xy[b] = dJ7_dr * dr_deps_xy;
    }
}

// ============================================================
// ENERGY CALCULATIONS
// ============================================================

double StrainPhononLattice::spin_energy() const {
    double E = 0.0;
    
    // NN exchange
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        // Zeeman term
        E -= Si.dot(field[i]);
        
        // NN interactions (count once by j > i)
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                E += Si.dot(nn_interaction[i][n] * Sj);
            }
        }
        
        // 2nd NN
        for (size_t n = 0; n < j2_partners[i].size(); ++n) {
            size_t j = j2_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                E += Si.dot(j2_interaction[i][n] * Sj);
            }
        }
        
        // 3rd NN
        for (size_t n = 0; n < j3_partners[i].size(); ++n) {
            size_t j = j3_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                E += Si.dot(j3_interaction[i][n] * Sj);
            }
        }
    }
    
    // Ring exchange
    E += ring_exchange_energy();
    
    return E;
}

double StrainPhononLattice::ring_exchange_energy() const {
    // Six-spin ring exchange on hexagonal plaquettes
    // H_7 = (J_7/6) Σ_{hex} [2(S_i·S_j)(S_k·S_l)(S_m·S_n)
    //                       -6(S_i·S_k)(S_j·S_l)(S_m·S_n)
    //                       +3(S_i·S_l)(S_j·S_k)(S_m·S_n)
    //                       +3(S_i·S_k)(S_j·S_m)(S_l·S_n)
    //                       -(S_i·S_l)(S_j·S_m)(S_k·S_n)
    //                       + cyclic permutations of (i,j,k,l,m,n)]
    
    // Use strain-dependent J7 if gamma_J7 is nonzero
    // This makes the static energy consistent with the dynamics
    double J7;
    if (std::abs(magnetoelastic_params.gamma_J7) > 1e-12) {
        J7 = get_effective_J7(0.0);  // t=0 is unused; uses current strain state
    } else {
        J7 = magnetoelastic_params.J7;
    }
    if (std::abs(J7) < 1e-12 || hexagons.empty()) {
        return 0.0;
    }
    
    double E = 0.0;
    double prefactor = J7 / 6.0;
    
    for (const auto& hex : hexagons) {
        // Get the 6 spins around the hexagon
        const Eigen::Vector3d& S0 = spins[hex[0]];
        const Eigen::Vector3d& S1 = spins[hex[1]];
        const Eigen::Vector3d& S2 = spins[hex[2]];
        const Eigen::Vector3d& S3 = spins[hex[3]];
        const Eigen::Vector3d& S4 = spins[hex[4]];
        const Eigen::Vector3d& S5 = spins[hex[5]];
        
        // Precompute all pairwise dot products (15 unique pairs)
        double d01 = S0.dot(S1), d02 = S0.dot(S2), d03 = S0.dot(S3);
        double d04 = S0.dot(S4), d05 = S0.dot(S5);
        double d12 = S1.dot(S2), d13 = S1.dot(S3), d14 = S1.dot(S4), d15 = S1.dot(S5);
        double d23 = S2.dot(S3), d24 = S2.dot(S4), d25 = S2.dot(S5);
        double d34 = S3.dot(S4), d35 = S3.dot(S5);
        double d45 = S4.dot(S5);
        
        // Sum over 6 cyclic permutations
        // For each permutation (i,j,k,l,m,n), compute:
        //   2*(i·j)*(k·l)*(m·n) - 6*(i·k)*(j·l)*(m·n) + 3*(i·l)*(j·k)*(m·n)
        //   + 3*(i·k)*(j·m)*(l·n) - (i·l)*(j·m)*(k·n)
        
        // Permutation 0: (0,1,2,3,4,5)
        double term0 = 2.0*d01*d23*d45 - 6.0*d02*d13*d45 + 3.0*d03*d12*d45 
                     + 3.0*d02*d14*d35 - d03*d14*d25;
        
        // Permutation 1: (1,2,3,4,5,0)
        double term1 = 2.0*d12*d34*d05 - 6.0*d13*d24*d05 + 3.0*d14*d23*d05 
                     + 3.0*d13*d25*d04 - d14*d25*d03;
        
        // Permutation 2: (2,3,4,5,0,1)
        double term2 = 2.0*d23*d45*d01 - 6.0*d24*d35*d01 + 3.0*d25*d34*d01 
                     + 3.0*d24*d03*d15 - d25*d03*d14;
        
        // Permutation 3: (3,4,5,0,1,2)
        double term3 = 2.0*d34*d05*d12 - 6.0*d35*d04*d12 + 3.0*d03*d45*d12 
                     + 3.0*d35*d14*d02 - d03*d14*d25;
        
        // Permutation 4: (4,5,0,1,2,3)
        double term4 = 2.0*d45*d01*d23 - 6.0*d04*d15*d23 + 3.0*d14*d05*d23 
                     + 3.0*d04*d25*d13 - d14*d25*d03;
        
        // Permutation 5: (5,0,1,2,3,4)
        double term5 = 2.0*d05*d12*d34 - 6.0*d15*d02*d34 + 3.0*d25*d01*d34 
                     + 3.0*d15*d03*d24 - d25*d03*d14;
        
        E += prefactor * (term0 + term1 + term2 + term3 + term4 + term5);
    }
    
    return E;
}

double StrainPhononLattice::spin_energy(double t) const {
    // Time-dependent version that uses effective J7(t)
    double E = 0.0;
    
    // NN exchange
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        // Zeeman term
        E -= Si.dot(field[i]);
        
        // NN interactions (count once by j > i)
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                E += Si.dot(nn_interaction[i][n] * Sj);
            }
        }
        
        // 2nd NN
        for (size_t n = 0; n < j2_partners[i].size(); ++n) {
            size_t j = j2_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                E += Si.dot(j2_interaction[i][n] * Sj);
            }
        }
        
        // 3rd NN
        for (size_t n = 0; n < j3_partners[i].size(); ++n) {
            size_t j = j3_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                E += Si.dot(j3_interaction[i][n] * Sj);
            }
        }
    }
    
    // Time-dependent ring exchange
    E += ring_exchange_energy(t);
    
    return E;
}

double StrainPhononLattice::ring_exchange_energy(double t) const {
    // Time-dependent version using effective J7(t)
    // J7(t) = J7 * (1 - γ*|δε_Eg|/4)^4 * (1 + γ*|δε_Eg|/2)^2
    // where |δε_Eg| is the Eg strain deviation from equilibrium
    
    double J7_eff = get_effective_J7(t);
    if (std::abs(J7_eff) < 1e-12 || hexagons.empty()) {
        return 0.0;
    }
    
    double E = 0.0;
    double prefactor = J7_eff / 6.0;
    
    for (const auto& hex : hexagons) {
        const Eigen::Vector3d& S0 = spins[hex[0]];
        const Eigen::Vector3d& S1 = spins[hex[1]];
        const Eigen::Vector3d& S2 = spins[hex[2]];
        const Eigen::Vector3d& S3 = spins[hex[3]];
        const Eigen::Vector3d& S4 = spins[hex[4]];
        const Eigen::Vector3d& S5 = spins[hex[5]];
        
        double d01 = S0.dot(S1), d02 = S0.dot(S2), d03 = S0.dot(S3);
        double d04 = S0.dot(S4), d05 = S0.dot(S5);
        double d12 = S1.dot(S2), d13 = S1.dot(S3), d14 = S1.dot(S4), d15 = S1.dot(S5);
        double d23 = S2.dot(S3), d24 = S2.dot(S4), d25 = S2.dot(S5);
        double d34 = S3.dot(S4), d35 = S3.dot(S5);
        double d45 = S4.dot(S5);
        
        double term0 = 2.0*d01*d23*d45 - 6.0*d02*d13*d45 + 3.0*d03*d12*d45 
                     + 3.0*d02*d14*d35 - d03*d14*d25;
        double term1 = 2.0*d12*d34*d05 - 6.0*d13*d24*d05 + 3.0*d14*d23*d05 
                     + 3.0*d13*d25*d04 - d14*d25*d03;
        double term2 = 2.0*d23*d45*d01 - 6.0*d24*d35*d01 + 3.0*d25*d34*d01 
                     + 3.0*d24*d03*d15 - d25*d03*d14;
        double term3 = 2.0*d34*d05*d12 - 6.0*d35*d04*d12 + 3.0*d03*d45*d12 
                     + 3.0*d35*d14*d02 - d03*d14*d25;
        double term4 = 2.0*d45*d01*d23 - 6.0*d04*d15*d23 + 3.0*d14*d05*d23 
                     + 3.0*d04*d25*d13 - d14*d25*d03;
        double term5 = 2.0*d05*d12*d34 - 6.0*d15*d02*d34 + 3.0*d25*d01*d34 
                     + 3.0*d15*d03*d24 - d25*d03*d14;
        
        E += prefactor * (term0 + term1 + term2 + term3 + term4 + term5);
    }
    
    return E;
}

double StrainPhononLattice::strain_energy() const {
    // H_strain = (1/2) Σ_b [C11(ε_xx² + ε_yy²) + 2C12 ε_xx ε_yy + 4C44 ε_xy²]
    //          + kinetic energy
    
    double V = 0.0;  // Potential energy
    double T = strain.kinetic_energy() * elastic_params.M;  // Kinetic energy
    
    double C11 = elastic_params.C11;
    double C12 = elastic_params.C12;
    double C44 = elastic_params.C44;
    
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        double exx = strain.epsilon_xx[b];
        double eyy = strain.epsilon_yy[b];
        double exy = strain.epsilon_xy[b];
        
        // Elastic potential energy
        V += 0.5 * C11 * (exx * exx + eyy * eyy);
        V += C12 * exx * eyy;
        V += 2.0 * C44 * exy * exy;  // Factor of 4 for (2ε_xy)² term
        
        // Optional quartic anharmonicity
        double A1g = exx + eyy;
        double Eg_sq = (exx - eyy) * (exx - eyy) + 4.0 * exy * exy;
        V += 0.25 * elastic_params.lambda_A1g * A1g * A1g * A1g * A1g;
        V += 0.25 * elastic_params.lambda_Eg * Eg_sq * Eg_sq;
    }
    
    return T + V;
}

// ============================================================
// SITE ENERGY FOR METROPOLIS
// ============================================================

double StrainPhononLattice::site_energy(const SpinVector& spin_here, size_t site) const {
    double E = 0.0;
    
    // Zeeman term: -B · S
    E -= spin_here.dot(field[site]);
    
    // NN exchange: S_i · J · S_j
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        E += spin_here.dot(nn_interaction[site][n] * spins[j]);
    }
    
    // 2nd NN exchange
    for (size_t n = 0; n < j2_partners[site].size(); ++n) {
        size_t j = j2_partners[site][n];
        E += spin_here.dot(j2_interaction[site][n] * spins[j]);
    }
    
    // 3rd NN exchange
    for (size_t n = 0; n < j3_partners[site].size(); ++n) {
        size_t j = j3_partners[site][n];
        E += spin_here.dot(j3_interaction[site][n] * spins[j]);
    }
    
    // Magnetoelastic coupling contribution
    // This is more complex - we need the derivative w.r.t. spin dotted with spin
    // H_c = λ_{A1g} (ε_xx + ε_yy) [(J+K)f_K^{A1g} + J f_J^{A1g} + Γ f_Γ^{A1g}] + ...
    // For simplicity, we use the effective field approach here
    SpinVector H_me = get_magnetoelastic_field(site);
    E -= spin_here.dot(H_me);  // H_me = -∂H_c/∂S, so E contribution is -S · H_me
    
    // Ring exchange contribution from hexagons containing this site
    E += site_ring_exchange_energy(spin_here, site);
    
    return E;
}

double StrainPhononLattice::site_energy_diff(const SpinVector& new_spin, 
                                              const SpinVector& old_spin,
                                              size_t site) const {
    SpinVector delta = new_spin - old_spin;
    double dE = 0.0;
    
    // Zeeman: -B · δS
    dE -= delta.dot(field[site]);
    
    // NN exchange: δS · J · S_j
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        dE += delta.dot(nn_interaction[site][n] * spins[j]);
    }
    
    // 2nd NN exchange
    for (size_t n = 0; n < j2_partners[site].size(); ++n) {
        size_t j = j2_partners[site][n];
        dE += delta.dot(j2_interaction[site][n] * spins[j]);
    }
    
    // 3rd NN exchange
    for (size_t n = 0; n < j3_partners[site].size(); ++n) {
        size_t j = j3_partners[site][n];
        dE += delta.dot(j3_interaction[site][n] * spins[j]);
    }
    
    // Magnetoelastic contribution: -δS · H_me
    SpinVector H_me = get_magnetoelastic_field(site);
    dE -= delta.dot(H_me);
    
    // Ring exchange: need to compute E_ring(new) - E_ring(old) for hexagons containing this site
    // Since ring exchange is linear in each spin, we can compute this efficiently
    dE += site_ring_exchange_energy(new_spin, site) - site_ring_exchange_energy(old_spin, site);
    
    return dE;
}

double StrainPhononLattice::site_ring_exchange_energy(const SpinVector& spin_here, 
                                                       size_t site) const {
    // Compute the ring exchange energy contribution from hexagons containing this site
    // when the spin at 'site' is 'spin_here'
    //
    // H_7 = (J_7/6) Σ_{hex} Σ_{cyclic perms} [terms]
    // Each hexagon is counted once, but we sum over all hexagons containing this site
    
    double J7;
    if (std::abs(magnetoelastic_params.gamma_J7) > 1e-12) {
        J7 = get_effective_J7(0.0);
    } else {
        J7 = magnetoelastic_params.J7;
    }
    if (std::abs(J7) < 1e-12 || site_hexagons[site].empty()) {
        return 0.0;
    }
    
    double E = 0.0;
    const double prefactor = J7 / 6.0;  // J7/6 * 1/6 for per-site contribution
    
    // Loop over all hexagons containing this site
    for (const auto& [hex_idx, pos] : site_hexagons[site]) {
        const auto& hex = hexagons[hex_idx];
        
        // Load all 6 spins into local array for better cache locality
        // Substitute spin_here at the appropriate position
        const SpinVector* S[6];
        SpinVector S_local;
        for (size_t i = 0; i < 6; ++i) {
            if (hex[i] == site) {
                S_local = spin_here;
                S[i] = &S_local;
            } else {
                S[i] = &spins[hex[i]];
            }
        }
        
        // Precompute only the 15 unique pairwise dot products in a flat array
        // Indexing: d[i,j] where i<j is stored at index (i*(11-i))/2 + j - 1
        // But for simplicity and performance, we use explicit variables
        // This is more cache-friendly than a 2D array
        const double d01 = S[0]->dot(*S[1]), d02 = S[0]->dot(*S[2]), d03 = S[0]->dot(*S[3]);
        const double d04 = S[0]->dot(*S[4]), d05 = S[0]->dot(*S[5]);
        const double d12 = S[1]->dot(*S[2]), d13 = S[1]->dot(*S[3]), d14 = S[1]->dot(*S[4]);
        const double d15 = S[1]->dot(*S[5]);
        const double d23 = S[2]->dot(*S[3]), d24 = S[2]->dot(*S[4]), d25 = S[2]->dot(*S[5]);
        const double d34 = S[3]->dot(*S[4]), d35 = S[3]->dot(*S[5]);
        const double d45 = S[4]->dot(*S[5]);
        
        // Sum over 6 cyclic permutations using explicit expressions
        // This unrolling improves instruction-level parallelism
        // Permutation 0: (0,1,2,3,4,5)
        double hex_E = 2.0*d01*d23*d45 - 6.0*d02*d13*d45 + 3.0*d03*d12*d45 
                     + 3.0*d02*d14*d35 - d03*d14*d25;
        
        // Permutation 1: (1,2,3,4,5,0)
        hex_E += 2.0*d12*d34*d05 - 6.0*d13*d24*d05 + 3.0*d14*d23*d05 
               + 3.0*d13*d25*d04 - d14*d25*d03;
        
        // Permutation 2: (2,3,4,5,0,1)
        hex_E += 2.0*d23*d45*d01 - 6.0*d24*d35*d01 + 3.0*d25*d34*d01 
               + 3.0*d24*d03*d15 - d25*d03*d14;
        
        // Permutation 3: (3,4,5,0,1,2)
        hex_E += 2.0*d34*d05*d12 - 6.0*d35*d04*d12 + 3.0*d03*d45*d12 
               + 3.0*d35*d14*d02 - d03*d14*d25;
        
        // Permutation 4: (4,5,0,1,2,3)
        hex_E += 2.0*d45*d01*d23 - 6.0*d04*d15*d23 + 3.0*d14*d05*d23 
               + 3.0*d04*d25*d13 - d14*d25*d03;
        
        // Permutation 5: (5,0,1,2,3,4)
        hex_E += 2.0*d05*d12*d34 - 6.0*d15*d02*d34 + 3.0*d25*d01*d34 
               + 3.0*d15*d03*d24 - d25*d03*d14;
        
        E += prefactor * hex_E;
    }
    
    return E;
}

// ============================================================
// SPIN BASIS FUNCTIONS FOR D3d IRREPS
// ============================================================

double StrainPhononLattice::f_K_Eg1() const {
    // f_K^{Eg,1} = S^x_r S^x_{M_x} + S^y_r S^y_{M_y} - 2 S^z_r S^z_{M_z}
    // This is the Eg,1 basis for Kitaev terms under D3d
    double f_x = 0.0, f_y = 0.0, f_z = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int bond_type = nn_bond_types[i][n];
                
                double kitaev_term = Si(bond_type) * Sj(bond_type);
                if (bond_type == 0) f_x += kitaev_term;
                else if (bond_type == 1) f_y += kitaev_term;
                else f_z += kitaev_term;
            }
        }
    }
    
    // f_K^{Eg,1} = f_x + f_y - 2*f_z (from paper Table 2)
    return f_x + f_y - 2.0 * f_z;
}

double StrainPhononLattice::f_K_Eg2() const {
    // f_K^{Eg,2} = √3 (S^x_r S^x_{M_x} - S^y_r S^y_{M_y})
    // This is the Eg,2 basis for Kitaev terms under D3d
    double f_x = 0.0, f_y = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int bond_type = nn_bond_types[i][n];
                
                double kitaev_term = Si(bond_type) * Sj(bond_type);
                if (bond_type == 0) f_x += kitaev_term;
                else if (bond_type == 1) f_y += kitaev_term;
            }
        }
    }
    
    // f_K^{Eg,2} = √3 * (f_x - f_y) (from paper Table 2)
    return std::sqrt(3.0) * (f_x - f_y);
}

double StrainPhononLattice::f_J_Eg1() const {
    // f_J^{Eg,1} = S^x_r S^x_{M_{y,z}} + S^y_r S^y_{M_{x,z}} - 2 S^z_r S^z_{M_{x,y}}
    // On x-bond (γ=0): sum S^y·S^y + S^z·S^z (non-Kitaev components)
    // On y-bond (γ=1): sum S^x·S^x + S^z·S^z
    // On z-bond (γ=2): sum S^x·S^x + S^y·S^y
    // Then combine: contributions from x,y bonds (coeff +1) and z bonds (coeff -2)
    double f_x = 0.0, f_y = 0.0, f_z = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int gamma = nn_bond_types[i][n];
                
                // Sum over non-Kitaev components α ≠ γ
                int alpha = (gamma + 1) % 3;
                int beta = (gamma + 2) % 3;
                double non_kitaev_term = Si(alpha) * Sj(alpha) + Si(beta) * Sj(beta);
                
                if (gamma == 0) f_x += non_kitaev_term;  // x-bond contribution
                else if (gamma == 1) f_y += non_kitaev_term;  // y-bond contribution
                else f_z += non_kitaev_term;  // z-bond contribution
            }
        }
    }
    
    // f_J^{Eg,1} = f_x + f_y - 2*f_z (from paper Table 2)
    return f_x + f_y - 2.0 * f_z;
}

double StrainPhononLattice::f_J_Eg2() const {
    // f_J^{Eg,2} = √3 (S^x_r S^x_{M_{y,z}} - S^y_r S^y_{M_{x,z}})
    // On x-bond (γ=0): contributes S^y·S^y + S^z·S^z (but only x-component matters for Eg2)
    // On y-bond (γ=1): contributes S^x·S^x + S^z·S^z
    // The asymmetry is between x-bonds and y-bonds
    double f_x = 0.0, f_y = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int gamma = nn_bond_types[i][n];
                
                // Sum over non-Kitaev components
                int alpha = (gamma + 1) % 3;
                int beta = (gamma + 2) % 3;
                double non_kitaev_term = Si(alpha) * Sj(alpha) + Si(beta) * Sj(beta);
                
                if (gamma == 0) f_x += non_kitaev_term;
                else if (gamma == 1) f_y += non_kitaev_term;
            }
        }
    }
    
    // f_J^{Eg,2} = √3 * (f_x - f_y) (from paper Table 2)
    return std::sqrt(3.0) * (f_x - f_y);
}

double StrainPhononLattice::f_Gamma_Eg1() const {
    // f_Γ^{Eg,1} = (S^y_r S^z_{M_x} + S^z_r S^y_{M_x}) + (S^x_r S^z_{M_y} + S^z_r S^x_{M_y})
    //            - 2(S^x_r S^y_{M_z} + S^y_r S^x_{M_z})
    // This is the off-diagonal symmetric exchange in Eg,1 basis
    double f_x = 0.0, f_y = 0.0, f_z = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int gamma = nn_bond_types[i][n];
                int alpha = (gamma + 1) % 3;
                int beta = (gamma + 2) % 3;
                
                // Γ term: S^α S^β + S^β S^α where α,β ≠ γ
                double gamma_term = Si(alpha) * Sj(beta) + Si(beta) * Sj(alpha);
                if (gamma == 0) f_x += gamma_term;
                else if (gamma == 1) f_y += gamma_term;
                else f_z += gamma_term;
            }
        }
    }
    
    // f_Γ^{Eg,1} = f_x + f_y - 2*f_z (from paper Table 2)
    return f_x + f_y - 2.0 * f_z;
}

double StrainPhononLattice::f_Gamma_Eg2() const {
    // f_Γ^{Eg,2} = √3 [(S^y_r S^z_{M_x} + S^z_r S^y_{M_x}) - (S^x_r S^z_{M_y} + S^z_r S^x_{M_y})]
    // This is the off-diagonal symmetric exchange in Eg,2 basis
    double f_x = 0.0, f_y = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int gamma = nn_bond_types[i][n];
                int alpha = (gamma + 1) % 3;
                int beta = (gamma + 2) % 3;
                
                // Γ term: S^α S^β + S^β S^α where α,β ≠ γ
                double gamma_term = Si(alpha) * Sj(beta) + Si(beta) * Sj(alpha);
                if (gamma == 0) f_x += gamma_term;
                else if (gamma == 1) f_y += gamma_term;
            }
        }
    }
    
    // f_Γ^{Eg,2} = √3 * (f_x - f_y) (from paper Table 2)
    return std::sqrt(3.0) * (f_x - f_y);
}

double StrainPhononLattice::f_Gammap_Eg1() const {
    // f_Γ'^{Eg,1} = O_x + O_y - 2*O_z where O_γ is the Γ' operator on γ-bonds
    // Γ' operator on γ-bond: S_i^γ (S_j^α + S_j^β) + (S_i^α + S_i^β) S_j^γ where α,β ≠ γ
    double f_x = 0.0, f_y = 0.0, f_z = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int gamma = nn_bond_types[i][n];
                int alpha = (gamma + 1) % 3;
                int beta = (gamma + 2) % 3;
                
                // Γ' term: S_i^γ (S_j^α + S_j^β) + (S_i^α + S_i^β) S_j^γ
                double gammap_term = Si(gamma) * (Sj(alpha) + Sj(beta)) 
                                   + (Si(alpha) + Si(beta)) * Sj(gamma);
                if (gamma == 0) f_x += gammap_term;
                else if (gamma == 1) f_y += gammap_term;
                else f_z += gammap_term;
            }
        }
    }
    
    // f_Γ'^{Eg,1} = f_x + f_y - 2*f_z
    return f_x + f_y - 2.0 * f_z;
}

double StrainPhononLattice::f_Gammap_Eg2() const {
    // f_Γ'^{Eg,2} = √3 (O_x - O_y) where O_γ is the Γ' operator on γ-bonds
    double f_x = 0.0, f_y = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int gamma = nn_bond_types[i][n];
                int alpha = (gamma + 1) % 3;
                int beta = (gamma + 2) % 3;
                
                // Γ' term: S_i^γ (S_j^α + S_j^β) + (S_i^α + S_i^β) S_j^γ
                double gammap_term = Si(gamma) * (Sj(alpha) + Sj(beta)) 
                                   + (Si(alpha) + Si(beta)) * Sj(gamma);
                if (gamma == 0) f_x += gammap_term;
                else if (gamma == 1) f_y += gammap_term;
            }
        }
    }
    
    // f_Γ'^{Eg,2} = √3 * (f_x - f_y)
    return std::sqrt(3.0) * (f_x - f_y);
}

// ============================================================
// MAGNETOELASTIC ENERGY
// ============================================================

double StrainPhononLattice::magnetoelastic_energy() const {
    // H_c = H_c^{Eg} (A1g terms removed)
    //
    // H_c^{Eg} = λ_{Eg} Σ_b {(ε_xx - ε_yy)_b[(J+K)f_K^{Eg,1} + J f_J^{Eg,1} + Γ f_Γ^{Eg,1} + Γ' f_Γ'^{Eg,1}]
    //                      + 2(ε_xy)_b[(J+K)f_K^{Eg,2} + J f_J^{Eg,2} + Γ f_Γ^{Eg,2} + Γ' f_Γ'^{Eg,2}]}
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double Gammap = magnetoelastic_params.Gammap;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    // Compute Eg spin basis functions
    double fK_Eg1 = f_K_Eg1();
    double fK_Eg2 = f_K_Eg2();
    double fJ_Eg1 = f_J_Eg1();
    double fJ_Eg2 = f_J_Eg2();
    double fG_Eg1 = f_Gamma_Eg1();
    double fG_Eg2 = f_Gamma_Eg2();
    double fGp_Eg1 = f_Gammap_Eg1();
    double fGp_Eg2 = f_Gammap_Eg2();
    
    // Eg channel contribution (includes Kitaev, Heisenberg, Gamma, and Gamma' terms)
    double Eg1_spin_factor = (J + K) * fK_Eg1 + J * fJ_Eg1 + Gamma * fG_Eg1 + Gammap * fGp_Eg1;
    double Eg2_spin_factor = (J + K) * fK_Eg2 + J * fJ_Eg2 + Gamma * fG_Eg2 + Gammap * fGp_Eg2;
    
    double E_Eg = 0.0;
    
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        double exx = strain.epsilon_xx[b];
        double eyy = strain.epsilon_yy[b];
        double exy = strain.epsilon_xy[b];
        
        // Eg: (ε_xx - ε_yy) * Eg1 + 2ε_xy * Eg2
        E_Eg += lambda_Eg * ((exx - eyy) * Eg1_spin_factor + 2.0 * exy * Eg2_spin_factor);
    }
    
    return E_Eg;
}

// ============================================================
// DERIVATIVES FOR STRAIN EOM
// ============================================================

double StrainPhononLattice::dH_deps_xx(size_t bond_type) const {
    // ∂H_c/∂ε_xx = λ_{Eg} * Eg1_spin_factor (A1g terms removed)
    //            + elastic terms: C11 * ε_xx + C12 * ε_yy
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double Gammap = magnetoelastic_params.Gammap;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    // Spin factor (includes Kitaev, Heisenberg, Gamma, and Gamma' terms)
    double Eg1_spin_factor = (J + K) * f_K_Eg1() + J * f_J_Eg1() 
                           + Gamma * f_Gamma_Eg1() + Gammap * f_Gammap_Eg1();
    
    // Magnetoelastic contribution: ∂/∂ε_xx of (ε_xx - ε_yy) = +1
    double dH_me = lambda_Eg * Eg1_spin_factor;
    
    return dH_me;
}

double StrainPhononLattice::dH_deps_yy(size_t bond_type) const {
    // ∂H_c/∂ε_yy = -λ_{Eg} * Eg1_spin_factor (A1g terms removed)
    //            + elastic terms: C11 * ε_yy + C12 * ε_xx
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double Gammap = magnetoelastic_params.Gammap;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    // Spin factor (includes Kitaev, Heisenberg, Gamma, and Gamma' terms)
    double Eg1_spin_factor = (J + K) * f_K_Eg1() + J * f_J_Eg1() 
                           + Gamma * f_Gamma_Eg1() + Gammap * f_Gammap_Eg1();
    
    // Magnetoelastic contribution: ∂/∂ε_yy of (ε_xx - ε_yy) = -1
    double dH_me = -lambda_Eg * Eg1_spin_factor;
    
    return dH_me;
}

double StrainPhononLattice::dH_deps_xy(size_t bond_type) const {
    // ∂H_c/∂ε_xy = 2 * λ_{Eg} * Eg2_spin_factor
    //            + elastic terms: 4 * C44 * ε_xy
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double Gammap = magnetoelastic_params.Gammap;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    // Spin factor (includes Kitaev, Heisenberg, Gamma, and Gamma' terms)
    double Eg2_spin_factor = (J + K) * f_K_Eg2() + J * f_J_Eg2() 
                           + Gamma * f_Gamma_Eg2() + Gammap * f_Gammap_Eg2();
    
    // Magnetoelastic contribution: ∂/∂ε_xy of 2ε_xy = 2
    double dH_me = 2.0 * lambda_Eg * Eg2_spin_factor;
    
    return dH_me;
}

// ============================================================
// DERIVATIVE OF SPIN BASIS FUNCTIONS W.R.T. SPINS
// ============================================================

SpinVector StrainPhononLattice::df_K_Eg1_dS(size_t site) const {
    // ∂f_K^{Eg,1}/∂S_site where f_K^{Eg,1} = f_x + f_y - 2*f_z
    // Each bond contributes S_j^γ to component γ
    SpinVector df_x = Eigen::Vector3d::Zero();
    SpinVector df_y = Eigen::Vector3d::Zero();
    SpinVector df_z = Eigen::Vector3d::Zero();
    
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        int gamma = nn_bond_types[site][n];
        
        SpinVector contrib = Eigen::Vector3d::Zero();
        contrib(gamma) = spins[j](gamma);
        
        if (gamma == 0) df_x += contrib;
        else if (gamma == 1) df_y += contrib;
        else df_z += contrib;
    }
    
    // f_K^{Eg,1} = f_x + f_y - 2*f_z
    return df_x + df_y - 2.0 * df_z;
}

SpinVector StrainPhononLattice::df_K_Eg2_dS(size_t site) const {
    // ∂f_K^{Eg,2}/∂S_site where f_K^{Eg,2} = √3*(f_x - f_y)
    SpinVector df_x = Eigen::Vector3d::Zero();
    SpinVector df_y = Eigen::Vector3d::Zero();
    
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        int gamma = nn_bond_types[site][n];
        
        SpinVector contrib = Eigen::Vector3d::Zero();
        contrib(gamma) = spins[j](gamma);
        
        if (gamma == 0) df_x += contrib;
        else if (gamma == 1) df_y += contrib;
    }
    
    // f_K^{Eg,2} = √3 * (f_x - f_y)
    return std::sqrt(3.0) * (df_x - df_y);
}

SpinVector StrainPhononLattice::df_J_Eg1_dS(size_t site) const {
    // ∂f_J^{Eg,1}/∂S_site where f_J^{Eg,1} uses non-Kitaev components
    // f_J^{Eg,1} = f_x + f_y - 2*f_z where f_γ = Σ (S^α·S^α + S^β·S^β) on γ-bonds
    SpinVector df_x = Eigen::Vector3d::Zero();
    SpinVector df_y = Eigen::Vector3d::Zero();
    SpinVector df_z = Eigen::Vector3d::Zero();
    
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        int gamma = nn_bond_types[site][n];
        int alpha = (gamma + 1) % 3;
        int beta = (gamma + 2) % 3;
        
        // Non-Kitaev contribution: S^α_j in component α, S^β_j in component β
        SpinVector contrib = Eigen::Vector3d::Zero();
        contrib(alpha) = spins[j](alpha);
        contrib(beta) = spins[j](beta);
        
        if (gamma == 0) df_x += contrib;
        else if (gamma == 1) df_y += contrib;
        else df_z += contrib;
    }
    
    // f_J^{Eg,1} = f_x + f_y - 2*f_z
    return df_x + df_y - 2.0 * df_z;
}

SpinVector StrainPhononLattice::df_J_Eg2_dS(size_t site) const {
    // ∂f_J^{Eg,2}/∂S_site where f_J^{Eg,2} = √3*(f_x - f_y)
    // with non-Kitaev components
    SpinVector df_x = Eigen::Vector3d::Zero();
    SpinVector df_y = Eigen::Vector3d::Zero();
    
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        int gamma = nn_bond_types[site][n];
        int alpha = (gamma + 1) % 3;
        int beta = (gamma + 2) % 3;
        
        // Non-Kitaev contribution
        SpinVector contrib = Eigen::Vector3d::Zero();
        contrib(alpha) = spins[j](alpha);
        contrib(beta) = spins[j](beta);
        
        if (gamma == 0) df_x += contrib;
        else if (gamma == 1) df_y += contrib;
    }
    
    // f_J^{Eg,2} = √3 * (f_x - f_y)
    return std::sqrt(3.0) * (df_x - df_y);
}

SpinVector StrainPhononLattice::df_Gamma_Eg1_dS(size_t site) const {
    // ∂f_Γ^{Eg,1}/∂S_site where f_Γ^{Eg,1} = f_x + f_y - 2*f_z
    SpinVector df_x = Eigen::Vector3d::Zero();
    SpinVector df_y = Eigen::Vector3d::Zero();
    SpinVector df_z = Eigen::Vector3d::Zero();
    
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        int gamma = nn_bond_types[site][n];
        int alpha = (gamma + 1) % 3;
        int beta = (gamma + 2) % 3;
        
        SpinVector contrib = Eigen::Vector3d::Zero();
        contrib(alpha) = spins[j](beta);
        contrib(beta) = spins[j](alpha);
        
        if (gamma == 0) df_x += contrib;
        else if (gamma == 1) df_y += contrib;
        else df_z += contrib;
    }
    
    // f_Γ^{Eg,1} = f_x + f_y - 2*f_z
    return df_x + df_y - 2.0 * df_z;
}

SpinVector StrainPhononLattice::df_Gamma_Eg2_dS(size_t site) const {
    // ∂f_Γ^{Eg,2}/∂S_site where f_Γ^{Eg,2} = √3*(f_x - f_y)
    SpinVector df_x = Eigen::Vector3d::Zero();
    SpinVector df_y = Eigen::Vector3d::Zero();
    
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        int gamma = nn_bond_types[site][n];
        int alpha = (gamma + 1) % 3;
        int beta = (gamma + 2) % 3;
        
        SpinVector contrib = Eigen::Vector3d::Zero();
        contrib(alpha) = spins[j](beta);
        contrib(beta) = spins[j](alpha);
        
        if (gamma == 0) df_x += contrib;
        else if (gamma == 1) df_y += contrib;
    }
    
    // f_Γ^{Eg,2} = √3 * (f_x - f_y)
    return std::sqrt(3.0) * (df_x - df_y);
}

SpinVector StrainPhononLattice::df_Gammap_Eg1_dS(size_t site) const {
    // ∂f_Γ'^{Eg,1}/∂S_site where f_Γ'^{Eg,1} = f_x + f_y - 2*f_z
    // The Γ' operator on γ-bond: S_i^γ (S_j^α + S_j^β) + (S_i^α + S_i^β) S_j^γ
    // Derivative w.r.t. S_i:
    //   ∂/∂S_i^γ: S_j^α + S_j^β
    //   ∂/∂S_i^α: S_j^γ
    //   ∂/∂S_i^β: S_j^γ
    SpinVector df_x = Eigen::Vector3d::Zero();
    SpinVector df_y = Eigen::Vector3d::Zero();
    SpinVector df_z = Eigen::Vector3d::Zero();
    
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        int gamma = nn_bond_types[site][n];
        int alpha = (gamma + 1) % 3;
        int beta = (gamma + 2) % 3;
        
        SpinVector contrib = Eigen::Vector3d::Zero();
        // ∂/∂S_i^γ contributes S_j^α + S_j^β
        contrib(gamma) = spins[j](alpha) + spins[j](beta);
        // ∂/∂S_i^α and ∂/∂S_i^β both contribute S_j^γ
        contrib(alpha) = spins[j](gamma);
        contrib(beta) = spins[j](gamma);
        
        if (gamma == 0) df_x += contrib;
        else if (gamma == 1) df_y += contrib;
        else df_z += contrib;
    }
    
    // f_Γ'^{Eg,1} = f_x + f_y - 2*f_z
    return df_x + df_y - 2.0 * df_z;
}

SpinVector StrainPhononLattice::df_Gammap_Eg2_dS(size_t site) const {
    // ∂f_Γ'^{Eg,2}/∂S_site where f_Γ'^{Eg,2} = √3*(f_x - f_y)
    SpinVector df_x = Eigen::Vector3d::Zero();
    SpinVector df_y = Eigen::Vector3d::Zero();
    
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        int gamma = nn_bond_types[site][n];
        int alpha = (gamma + 1) % 3;
        int beta = (gamma + 2) % 3;
        
        SpinVector contrib = Eigen::Vector3d::Zero();
        // ∂/∂S_i^γ contributes S_j^α + S_j^β
        contrib(gamma) = spins[j](alpha) + spins[j](beta);
        // ∂/∂S_i^α and ∂/∂S_i^β both contribute S_j^γ
        contrib(alpha) = spins[j](gamma);
        contrib(beta) = spins[j](gamma);
        
        if (gamma == 0) df_x += contrib;
        else if (gamma == 1) df_y += contrib;
    }
    
    // f_Γ'^{Eg,2} = √3 * (f_x - f_y)
    return std::sqrt(3.0) * (df_x - df_y);
}

// ============================================================
// EFFECTIVE FIELD
// ============================================================

SpinVector StrainPhononLattice::get_magnetoelastic_field(size_t site) const {
    // H_me = -∂H_c/∂S_i (A1g terms removed)
    // 
    // H_c = λ_{Eg} Σ_b {(ε_xx - ε_yy)_b[(J+K)f_K^{Eg,1} + J f_J^{Eg,1} + Γ f_Γ^{Eg,1} + Γ' f_Γ'^{Eg,1}]
    //                  + 2(ε_xy)_b[(J+K)f_K^{Eg,2} + J f_J^{Eg,2} + Γ f_Γ^{Eg,2} + Γ' f_Γ'^{Eg,2}]}
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double Gammap = magnetoelastic_params.Gammap;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    // Compute Eg strain factors (summed over bond types)
    double Eg1_strain = 0.0;
    double Eg2_strain = 0.0;
    
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        Eg1_strain += strain.epsilon_xx[b] - strain.epsilon_yy[b];
        Eg2_strain += 2.0 * strain.epsilon_xy[b];
    }
    
    // Compute spin derivatives (Eg only, includes Gammap)
    SpinVector df_K_Eg1 = df_K_Eg1_dS(site);
    SpinVector df_J_Eg1 = df_J_Eg1_dS(site);
    SpinVector df_G_Eg1 = df_Gamma_Eg1_dS(site);
    SpinVector df_Gp_Eg1 = df_Gammap_Eg1_dS(site);
    
    SpinVector df_K_Eg2 = df_K_Eg2_dS(site);
    SpinVector df_J_Eg2 = df_J_Eg2_dS(site);
    SpinVector df_G_Eg2 = df_Gamma_Eg2_dS(site);
    SpinVector df_Gp_Eg2 = df_Gammap_Eg2_dS(site);
    
    // Eg contribution (includes Kitaev, Heisenberg, Gamma, and Gamma' terms)
    SpinVector H_Eg1 = lambda_Eg * Eg1_strain *
                       ((J + K) * df_K_Eg1 + J * df_J_Eg1 + Gamma * df_G_Eg1 + Gammap * df_Gp_Eg1);
    SpinVector H_Eg2 = lambda_Eg * Eg2_strain *
                       ((J + K) * df_K_Eg2 + J * df_J_Eg2 + Gamma * df_G_Eg2 + Gammap * df_Gp_Eg2);
    
    // Total magnetoelastic field (negative because H_eff = -∂H/∂S)
    return -(H_Eg1 + H_Eg2);
}

SpinVector StrainPhononLattice::get_local_field(size_t site) const {
    // H_eff = B - ∂H_spin/∂S - ∂H_c/∂S
    
    SpinVector H = field[site];
    
    // NN exchange contribution
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        H -= nn_interaction[site][n] * spins[j];
    }
    
    // 2nd NN
    for (size_t n = 0; n < j2_partners[site].size(); ++n) {
        size_t j = j2_partners[site][n];
        H -= j2_interaction[site][n] * spins[j];
    }
    
    // 3rd NN
    for (size_t n = 0; n < j3_partners[site].size(); ++n) {
        size_t j = j3_partners[site][n];
        H -= j3_interaction[site][n] * spins[j];
    }
    
    // Ring exchange
    H += get_ring_exchange_field(site);
    
    // Magnetoelastic coupling
    H += get_magnetoelastic_field(site);
    
    return H;
}

SpinVector StrainPhononLattice::get_ring_exchange_field(size_t site) const {
    // Compute H_eff_ring = -∂H_7/∂S_site
    // 
    // Ring exchange energy: H_7 = (J7/6) Σ_hex Σ_{perms} [2*d_ij*d_kl*d_mn - 6*d_ik*d_jl*d_mn 
    //                                                    + 3*d_il*d_jk*d_mn + 3*d_ik*d_jm*d_ln - d_il*d_jm*d_kn]
    //
    // For a site at position p in the hexagon, we need to differentiate ALL 6 permutations
    // with respect to S_p. The derivative of d_ab w.r.t. S_p is S_b if a=p, or S_a if b=p, else 0.
    
    Eigen::Vector3d H = Eigen::Vector3d::Zero();
    double J7;
    if (std::abs(magnetoelastic_params.gamma_J7) > 1e-12) {
        J7 = get_effective_J7(0.0);
    } else {
        J7 = magnetoelastic_params.J7;
    }
    
    if (std::abs(J7) < 1e-12 || site_hexagons[site].empty()) {
        return H;
    }
    
    const double prefactor = -J7 / 6.0;  // Negative because H_eff = -∂H/∂S
    
    // Loop over all hexagons containing this site
    for (const auto& [hex_idx, pos] : site_hexagons[site]) {
        const auto& hex = hexagons[hex_idx];
        
        // Pre-load spin pointers for better cache locality
        const SpinVector* S[6] = {
            &spins[hex[0]], &spins[hex[1]], &spins[hex[2]],
            &spins[hex[3]], &spins[hex[4]], &spins[hex[5]]
        };
        
        // Precompute all 15 unique pairwise dot products
        const double d01 = S[0]->dot(*S[1]), d02 = S[0]->dot(*S[2]), d03 = S[0]->dot(*S[3]);
        const double d04 = S[0]->dot(*S[4]), d05 = S[0]->dot(*S[5]);
        const double d12 = S[1]->dot(*S[2]), d13 = S[1]->dot(*S[3]), d14 = S[1]->dot(*S[4]);
        const double d15 = S[1]->dot(*S[5]);
        const double d23 = S[2]->dot(*S[3]), d24 = S[2]->dot(*S[4]), d25 = S[2]->dot(*S[5]);
        const double d34 = S[3]->dot(*S[4]), d35 = S[3]->dot(*S[5]);
        const double d45 = S[4]->dot(*S[5]);
        
        Eigen::Vector3d dH = Eigen::Vector3d::Zero();
        
        // Compute ∂/∂S_p of ALL 6 permutations
        // Permutation k: uses indices (k, k+1, k+2, k+3, k+4, k+5) mod 6 as (i,j,k,l,m,n)
        // Energy term: 2*d_ij*d_kl*d_mn - 6*d_ik*d_jl*d_mn + 3*d_il*d_jk*d_mn + 3*d_ik*d_jm*d_ln - d_il*d_jm*d_kn
        //
        // For each permutation, the site at position 'pos' can appear as i,j,k,l,m, or n
        // We need to differentiate each term that contains a dot product involving 'pos'
        
        // For clarity, we compute the full derivative for each position explicitly
        // This is verbose but correct and verifiable
        
        if (pos == 0) {
            // ∂/∂S_0 of all 6 permutations
            // Perm 0 (0,1,2,3,4,5): 2*d01*d23*d45 - 6*d02*d13*d45 + 3*d03*d12*d45 + 3*d02*d14*d35 - d03*d14*d25
            //   d01 → S1, d02 → S2, d03 → S3, d04 → S4, d05 → S5
            dH += 2.0 * (*S[1]) * d23 * d45;   // ∂d01
            dH += -6.0 * (*S[2]) * d13 * d45;  // ∂d02
            dH += 3.0 * (*S[3]) * d12 * d45;   // ∂d03
            dH += 3.0 * (*S[2]) * d14 * d35;   // ∂d02
            dH += -(*S[3]) * d14 * d25;        // ∂d03
            
            // Perm 1 (1,2,3,4,5,0): 2*d12*d34*d50 - 6*d13*d24*d50 + 3*d14*d23*d50 + 3*d13*d25*d40 - d14*d25*d30
            //   d50=d05 → S5, d40=d04 → S4, d30=d03 → S3
            dH += 2.0 * (*S[5]) * d12 * d34;   // ∂d05
            dH += -6.0 * (*S[5]) * d13 * d24;  // ∂d05
            dH += 3.0 * (*S[5]) * d14 * d23;   // ∂d05
            dH += 3.0 * (*S[4]) * d13 * d25;   // ∂d04
            dH += -(*S[3]) * d14 * d25;        // ∂d03
            
            // Perm 2 (2,3,4,5,0,1): 2*d23*d45*d01 - 6*d24*d35*d01 + 3*d25*d34*d01 + 3*d24*d30*d15 - d25*d30*d14
            //   d01 → S1, d30=d03 → S3
            dH += 2.0 * (*S[1]) * d23 * d45;   // ∂d01
            dH += -6.0 * (*S[1]) * d24 * d35;  // ∂d01
            dH += 3.0 * (*S[1]) * d25 * d34;   // ∂d01
            dH += 3.0 * (*S[3]) * d24 * d15;   // ∂d03
            dH += -(*S[3]) * d25 * d14;        // ∂d03
            
            // Perm 3 (3,4,5,0,1,2): 2*d34*d50*d12 - 6*d35*d40*d12 + 3*d30*d45*d12 + 3*d35*d01*d42 - d30*d01*d52
            //   d50=d05 → S5, d40=d04 → S4, d30=d03 → S3, d01 → S1
            dH += 2.0 * (*S[5]) * d34 * d12;   // ∂d05
            dH += -6.0 * (*S[4]) * d35 * d12;  // ∂d04
            dH += 3.0 * (*S[3]) * d45 * d12;   // ∂d03
            dH += 3.0 * (*S[1]) * d35 * d24;   // ∂d01
            dH += -(*S[3]) * d01 * d25;        // ∂d03
            dH += -(*S[1]) * d03 * d25;        // ∂d01
            
            // Perm 4 (4,5,0,1,2,3): 2*d45*d01*d23 - 6*d40*d15*d23 + 3*d41*d05*d23 + 3*d40*d12*d53 - d41*d12*d03
            //   d01 → S1, d40=d04 → S4, d05 → S5
            dH += 2.0 * (*S[1]) * d45 * d23;   // ∂d01
            dH += -6.0 * (*S[4]) * d15 * d23;  // ∂d04
            dH += 3.0 * (*S[5]) * d14 * d23;   // ∂d05
            dH += 3.0 * (*S[4]) * d12 * d35;   // ∂d04
            dH += -(*S[3]) * d14 * d12;        // ∂d03
            
            // Perm 5 (5,0,1,2,3,4): 2*d50*d12*d34 - 6*d51*d02*d34 + 3*d52*d01*d34 + 3*d51*d03*d24 - d52*d03*d14
            //   d50=d05 → S5, d02 → S2, d01 → S1
            dH += 2.0 * (*S[5]) * d12 * d34;   // ∂d05
            dH += -6.0 * (*S[2]) * d15 * d34;  // ∂d02
            dH += 3.0 * (*S[1]) * d25 * d34;   // ∂d01
            dH += 3.0 * (*S[2]) * d15 * d24;   // ∂d02  (wait, there's also ∂d03)
            dH += 3.0 * (*S[3]) * d15 * d24;   // ∂d03
            dH += -(*S[3]) * d25 * d14;        // ∂d03
        }
        else if (pos == 1) {
            // ∂/∂S_1 of all 6 permutations
            // Perm 0: 2*d01*d23*d45 - 6*d02*d13*d45 + 3*d03*d12*d45 + 3*d02*d14*d35 - d03*d14*d25
            dH += 2.0 * (*S[0]) * d23 * d45;   // ∂d01
            dH += -6.0 * (*S[3]) * d02 * d45;  // ∂d13
            dH += 3.0 * (*S[2]) * d03 * d45;   // ∂d12
            dH += 3.0 * (*S[4]) * d02 * d35;   // ∂d14
            dH += -(*S[4]) * d03 * d25;        // ∂d14
            
            // Perm 1: 2*d12*d34*d05 - 6*d13*d24*d05 + 3*d14*d23*d05 + 3*d13*d25*d04 - d14*d25*d03
            dH += 2.0 * (*S[2]) * d34 * d05;   // ∂d12
            dH += -6.0 * (*S[3]) * d24 * d05;  // ∂d13
            dH += 3.0 * (*S[4]) * d23 * d05;   // ∂d14
            dH += 3.0 * (*S[3]) * d25 * d04;   // ∂d13
            dH += -(*S[4]) * d25 * d03;        // ∂d14
            
            // Perm 2: 2*d23*d45*d01 - 6*d24*d35*d01 + 3*d25*d34*d01 + 3*d24*d03*d15 - d25*d03*d14
            dH += 2.0 * (*S[0]) * d23 * d45;   // ∂d01
            dH += -6.0 * (*S[0]) * d24 * d35;  // ∂d01
            dH += 3.0 * (*S[0]) * d25 * d34;   // ∂d01
            dH += 3.0 * (*S[5]) * d24 * d03;   // ∂d15
            dH += -(*S[4]) * d25 * d03;        // ∂d14
            
            // Perm 3: 2*d34*d05*d12 - 6*d35*d04*d12 + 3*d03*d45*d12 + 3*d35*d01*d24 - d03*d01*d25
            dH += 2.0 * (*S[2]) * d34 * d05;   // ∂d12
            dH += -6.0 * (*S[2]) * d35 * d04;  // ∂d12
            dH += 3.0 * (*S[2]) * d03 * d45;   // ∂d12
            dH += 3.0 * (*S[0]) * d35 * d24;   // ∂d01
            dH += -(*S[0]) * d03 * d25;        // ∂d01
            
            // Perm 4: 2*d45*d01*d23 - 6*d04*d15*d23 + 3*d14*d05*d23 + 3*d04*d12*d35 - d14*d12*d03  (typo fixed: last term uses d03)
            dH += 2.0 * (*S[0]) * d45 * d23;   // ∂d01
            dH += -6.0 * (*S[5]) * d04 * d23;  // ∂d15
            dH += 3.0 * (*S[4]) * d05 * d23;   // ∂d14
            dH += 3.0 * (*S[2]) * d04 * d35;   // ∂d12
            dH += -(*S[4]) * d12 * d03;        // ∂d14
            dH += -(*S[2]) * d14 * d03;        // ∂d12
            
            // Perm 5: 2*d05*d12*d34 - 6*d15*d02*d34 + 3*d25*d01*d34 + 3*d15*d03*d24 - d25*d03*d14
            dH += 2.0 * (*S[2]) * d05 * d34;   // ∂d12
            dH += -6.0 * (*S[5]) * d02 * d34;  // ∂d15
            dH += 3.0 * (*S[0]) * d25 * d34;   // ∂d01
            dH += 3.0 * (*S[5]) * d03 * d24;   // ∂d15
            dH += -(*S[4]) * d25 * d03;        // ∂d14
        }
        else if (pos == 2) {
            // ∂/∂S_2 of all 6 permutations
            // Perm 0: 2*d01*d23*d45 - 6*d02*d13*d45 + 3*d03*d12*d45 + 3*d02*d14*d35 - d03*d14*d25
            dH += 2.0 * (*S[3]) * d01 * d45;   // ∂d23
            dH += -6.0 * (*S[0]) * d13 * d45;  // ∂d02
            dH += 3.0 * (*S[1]) * d03 * d45;   // ∂d12
            dH += 3.0 * (*S[0]) * d14 * d35;   // ∂d02
            dH += -(*S[5]) * d03 * d14;        // ∂d25
            
            // Perm 1: 2*d12*d34*d05 - 6*d13*d24*d05 + 3*d14*d23*d05 + 3*d13*d25*d04 - d14*d25*d03
            dH += 2.0 * (*S[1]) * d34 * d05;   // ∂d12
            dH += -6.0 * (*S[4]) * d13 * d05;  // ∂d24
            dH += 3.0 * (*S[3]) * d14 * d05;   // ∂d23
            dH += 3.0 * (*S[5]) * d13 * d04;   // ∂d25
            dH += -(*S[5]) * d14 * d03;        // ∂d25
            
            // Perm 2: 2*d23*d45*d01 - 6*d24*d35*d01 + 3*d25*d34*d01 + 3*d24*d03*d15 - d25*d03*d14
            dH += 2.0 * (*S[3]) * d45 * d01;   // ∂d23
            dH += -6.0 * (*S[4]) * d35 * d01;  // ∂d24
            dH += 3.0 * (*S[5]) * d34 * d01;   // ∂d25
            dH += 3.0 * (*S[4]) * d03 * d15;   // ∂d24
            dH += -(*S[5]) * d03 * d14;        // ∂d25
            
            // Perm 3: 2*d34*d05*d12 - 6*d35*d04*d12 + 3*d03*d45*d12 + 3*d35*d01*d24 - d03*d01*d25
            dH += 2.0 * (*S[1]) * d34 * d05;   // ∂d12
            dH += -6.0 * (*S[1]) * d35 * d04;  // ∂d12
            dH += 3.0 * (*S[1]) * d03 * d45;   // ∂d12
            dH += 3.0 * (*S[4]) * d35 * d01;   // ∂d24
            dH += -(*S[5]) * d03 * d01;        // ∂d25
            
            // Perm 4: 2*d45*d01*d23 - 6*d04*d15*d23 + 3*d14*d05*d23 + 3*d04*d12*d35 - d14*d12*d03
            dH += 2.0 * (*S[3]) * d45 * d01;   // ∂d23
            dH += -6.0 * (*S[3]) * d04 * d15;  // ∂d23
            dH += 3.0 * (*S[3]) * d14 * d05;   // ∂d23
            dH += 3.0 * (*S[1]) * d04 * d35;   // ∂d12
            dH += -(*S[1]) * d14 * d03;        // ∂d12
            
            // Perm 5: 2*d05*d12*d34 - 6*d15*d02*d34 + 3*d25*d01*d34 + 3*d15*d03*d24 - d25*d03*d14
            dH += 2.0 * (*S[1]) * d05 * d34;   // ∂d12
            dH += -6.0 * (*S[0]) * d15 * d34;  // ∂d02
            dH += 3.0 * (*S[5]) * d01 * d34;   // ∂d25
            dH += 3.0 * (*S[4]) * d15 * d03;   // ∂d24
            dH += -(*S[5]) * d03 * d14;        // ∂d25
        }
        else if (pos == 3) {
            // ∂/∂S_3 of all 6 permutations
            // Perm 0: 2*d01*d23*d45 - 6*d02*d13*d45 + 3*d03*d12*d45 + 3*d02*d14*d35 - d03*d14*d25
            dH += 2.0 * (*S[2]) * d01 * d45;   // ∂d23
            dH += -6.0 * (*S[1]) * d02 * d45;  // ∂d13
            dH += 3.0 * (*S[0]) * d12 * d45;   // ∂d03
            dH += 3.0 * (*S[5]) * d02 * d14;   // ∂d35
            dH += -(*S[0]) * d14 * d25;        // ∂d03
            
            // Perm 1: 2*d12*d34*d05 - 6*d13*d24*d05 + 3*d14*d23*d05 + 3*d13*d25*d04 - d14*d25*d03
            dH += 2.0 * (*S[4]) * d12 * d05;   // ∂d34
            dH += -6.0 * (*S[1]) * d24 * d05;  // ∂d13
            dH += 3.0 * (*S[2]) * d14 * d05;   // ∂d23
            dH += 3.0 * (*S[1]) * d25 * d04;   // ∂d13
            dH += -(*S[0]) * d14 * d25;        // ∂d03
            
            // Perm 2: 2*d23*d45*d01 - 6*d24*d35*d01 + 3*d25*d34*d01 + 3*d24*d03*d15 - d25*d03*d14
            dH += 2.0 * (*S[2]) * d45 * d01;   // ∂d23
            dH += -6.0 * (*S[5]) * d24 * d01;  // ∂d35
            dH += 3.0 * (*S[4]) * d25 * d01;   // ∂d34
            dH += 3.0 * (*S[0]) * d24 * d15;   // ∂d03
            dH += -(*S[0]) * d25 * d14;        // ∂d03
            
            // Perm 3: 2*d34*d05*d12 - 6*d35*d04*d12 + 3*d03*d45*d12 + 3*d35*d01*d24 - d03*d01*d25
            dH += 2.0 * (*S[4]) * d05 * d12;   // ∂d34
            dH += -6.0 * (*S[5]) * d04 * d12;  // ∂d35
            dH += 3.0 * (*S[0]) * d45 * d12;   // ∂d03
            dH += 3.0 * (*S[5]) * d01 * d24;   // ∂d35
            dH += -(*S[0]) * d01 * d25;        // ∂d03
            
            // Perm 4: 2*d45*d01*d23 - 6*d04*d15*d23 + 3*d14*d05*d23 + 3*d04*d12*d35 - d14*d12*d03
            dH += 2.0 * (*S[2]) * d45 * d01;   // ∂d23
            dH += -6.0 * (*S[2]) * d04 * d15;  // ∂d23
            dH += 3.0 * (*S[2]) * d14 * d05;   // ∂d23
            dH += 3.0 * (*S[5]) * d04 * d12;   // ∂d35
            dH += -(*S[0]) * d14 * d12;        // ∂d03
            
            // Perm 5: 2*d05*d12*d34 - 6*d15*d02*d34 + 3*d25*d01*d34 + 3*d15*d03*d24 - d25*d03*d14
            dH += 2.0 * (*S[4]) * d05 * d12;   // ∂d34
            dH += -6.0 * (*S[4]) * d15 * d02;  // ∂d34
            dH += 3.0 * (*S[4]) * d25 * d01;   // ∂d34
            dH += 3.0 * (*S[0]) * d15 * d24;   // ∂d03
            dH += -(*S[0]) * d25 * d14;        // ∂d03
        }
        else if (pos == 4) {
            // ∂/∂S_4 of all 6 permutations
            // Perm 0: 2*d01*d23*d45 - 6*d02*d13*d45 + 3*d03*d12*d45 + 3*d02*d14*d35 - d03*d14*d25
            dH += 2.0 * (*S[5]) * d01 * d23;   // ∂d45
            dH += -6.0 * (*S[5]) * d02 * d13;  // ∂d45
            dH += 3.0 * (*S[5]) * d03 * d12;   // ∂d45
            dH += 3.0 * (*S[1]) * d02 * d35;   // ∂d14
            dH += -(*S[1]) * d03 * d25;        // ∂d14
            
            // Perm 1: 2*d12*d34*d05 - 6*d13*d24*d05 + 3*d14*d23*d05 + 3*d13*d25*d04 - d14*d25*d03
            dH += 2.0 * (*S[3]) * d12 * d05;   // ∂d34
            dH += -6.0 * (*S[2]) * d13 * d05;  // ∂d24
            dH += 3.0 * (*S[1]) * d23 * d05;   // ∂d14
            dH += 3.0 * (*S[0]) * d13 * d25;   // ∂d04
            dH += -(*S[1]) * d25 * d03;        // ∂d14
            
            // Perm 2: 2*d23*d45*d01 - 6*d24*d35*d01 + 3*d25*d34*d01 + 3*d24*d03*d15 - d25*d03*d14
            dH += 2.0 * (*S[5]) * d23 * d01;   // ∂d45
            dH += -6.0 * (*S[2]) * d35 * d01;  // ∂d24
            dH += 3.0 * (*S[3]) * d25 * d01;   // ∂d34
            dH += 3.0 * (*S[2]) * d03 * d15;   // ∂d24
            dH += -(*S[1]) * d25 * d03;        // ∂d14
            
            // Perm 3: 2*d34*d05*d12 - 6*d35*d04*d12 + 3*d03*d45*d12 + 3*d35*d01*d24 - d03*d01*d25
            dH += 2.0 * (*S[3]) * d05 * d12;   // ∂d34
            dH += -6.0 * (*S[0]) * d35 * d12;  // ∂d04
            dH += 3.0 * (*S[5]) * d03 * d12;   // ∂d45
            dH += 3.0 * (*S[2]) * d35 * d01;   // ∂d24
            // no d04, d14, d24, d34, d45 in last term
            
            // Perm 4: 2*d45*d01*d23 - 6*d04*d15*d23 + 3*d14*d05*d23 + 3*d04*d12*d35 - d14*d12*d03
            dH += 2.0 * (*S[5]) * d01 * d23;   // ∂d45
            dH += -6.0 * (*S[0]) * d15 * d23;  // ∂d04
            dH += 3.0 * (*S[1]) * d05 * d23;   // ∂d14
            dH += 3.0 * (*S[0]) * d12 * d35;   // ∂d04
            dH += -(*S[1]) * d12 * d03;        // ∂d14
            
            // Perm 5: 2*d05*d12*d34 - 6*d15*d02*d34 + 3*d25*d01*d34 + 3*d15*d03*d24 - d25*d03*d14
            dH += 2.0 * (*S[3]) * d05 * d12;   // ∂d34
            dH += -6.0 * (*S[3]) * d15 * d02;  // ∂d34
            dH += 3.0 * (*S[3]) * d25 * d01;   // ∂d34
            dH += 3.0 * (*S[2]) * d15 * d03;   // ∂d24
            dH += -(*S[1]) * d25 * d03;        // ∂d14
        }
        else { // pos == 5
            // ∂/∂S_5 of all 6 permutations
            // Perm 0: 2*d01*d23*d45 - 6*d02*d13*d45 + 3*d03*d12*d45 + 3*d02*d14*d35 - d03*d14*d25
            dH += 2.0 * (*S[4]) * d01 * d23;   // ∂d45
            dH += -6.0 * (*S[4]) * d02 * d13;  // ∂d45
            dH += 3.0 * (*S[4]) * d03 * d12;   // ∂d45
            dH += 3.0 * (*S[3]) * d02 * d14;   // ∂d35
            dH += -(*S[2]) * d03 * d14;        // ∂d25
            
            // Perm 1: 2*d12*d34*d05 - 6*d13*d24*d05 + 3*d14*d23*d05 + 3*d13*d25*d04 - d14*d25*d03
            dH += 2.0 * (*S[0]) * d12 * d34;   // ∂d05
            dH += -6.0 * (*S[0]) * d13 * d24;  // ∂d05
            dH += 3.0 * (*S[0]) * d14 * d23;   // ∂d05
            dH += 3.0 * (*S[2]) * d13 * d04;   // ∂d25
            dH += -(*S[2]) * d14 * d03;        // ∂d25
            
            // Perm 2: 2*d23*d45*d01 - 6*d24*d35*d01 + 3*d25*d34*d01 + 3*d24*d03*d15 - d25*d03*d14
            dH += 2.0 * (*S[4]) * d23 * d01;   // ∂d45
            dH += -6.0 * (*S[3]) * d24 * d01;  // ∂d35
            dH += 3.0 * (*S[2]) * d34 * d01;   // ∂d25
            dH += 3.0 * (*S[1]) * d24 * d03;   // ∂d15
            dH += -(*S[2]) * d03 * d14;        // ∂d25
            
            // Perm 3: 2*d34*d05*d12 - 6*d35*d04*d12 + 3*d03*d45*d12 + 3*d35*d01*d24 - d03*d01*d25
            dH += 2.0 * (*S[0]) * d34 * d12;   // ∂d05
            dH += -6.0 * (*S[3]) * d04 * d12;  // ∂d35
            dH += 3.0 * (*S[4]) * d03 * d12;   // ∂d45
            dH += 3.0 * (*S[3]) * d01 * d24;   // ∂d35
            dH += -(*S[2]) * d03 * d01;        // ∂d25
            
            // Perm 4: 2*d45*d01*d23 - 6*d04*d15*d23 + 3*d14*d05*d23 + 3*d04*d12*d35 - d14*d12*d03
            dH += 2.0 * (*S[4]) * d01 * d23;   // ∂d45
            dH += -6.0 * (*S[1]) * d04 * d23;  // ∂d15
            dH += 3.0 * (*S[0]) * d14 * d23;   // ∂d05
            dH += 3.0 * (*S[3]) * d04 * d12;   // ∂d35
            // no d*5 in last term
            
            // Perm 5: 2*d05*d12*d34 - 6*d15*d02*d34 + 3*d25*d01*d34 + 3*d15*d03*d24 - d25*d03*d14
            dH += 2.0 * (*S[0]) * d12 * d34;   // ∂d05
            dH += -6.0 * (*S[1]) * d02 * d34;  // ∂d15
            dH += 3.0 * (*S[2]) * d01 * d34;   // ∂d25
            dH += 3.0 * (*S[1]) * d03 * d24;   // ∂d15
            dH += -(*S[2]) * d03 * d14;        // ∂d25
        }
        
        H += prefactor * dH;
    }
    
    return H;
}

SpinVector StrainPhononLattice::get_ring_exchange_field(size_t site, double t) const {
    // Time-dependent version using effective J7(t)
    // J7(t) = J7 * (1 - γ*|δε_Eg|/4)^4 * (1 + γ*|δε_Eg|/2)^2
    // where |δε_Eg| is the Eg strain deviation from equilibrium
    
    Eigen::Vector3d H = Eigen::Vector3d::Zero();
    double J7_eff = get_effective_J7(t);
    
    if (std::abs(J7_eff) < 1e-12 || site_hexagons[site].empty()) {
        return H;
    }
    
    double prefactor = -J7_eff / 6.0;  // Negative because H_eff = -∂H/∂S
    
    // Loop over all hexagons containing this site
    for (const auto& [hex_idx, pos] : site_hexagons[site]) {
        const auto& hex = hexagons[hex_idx];
        
        const Eigen::Vector3d& S0 = spins[hex[0]];
        const Eigen::Vector3d& S1 = spins[hex[1]];
        const Eigen::Vector3d& S2 = spins[hex[2]];
        const Eigen::Vector3d& S3 = spins[hex[3]];
        const Eigen::Vector3d& S4 = spins[hex[4]];
        const Eigen::Vector3d& S5 = spins[hex[5]];
        
        double d01 = S0.dot(S1), d02 = S0.dot(S2), d03 = S0.dot(S3);
        double d04 = S0.dot(S4), d05 = S0.dot(S5);
        double d12 = S1.dot(S2), d13 = S1.dot(S3), d14 = S1.dot(S4), d15 = S1.dot(S5);
        double d23 = S2.dot(S3), d24 = S2.dot(S4), d25 = S2.dot(S5);
        double d34 = S3.dot(S4), d35 = S3.dot(S5);
        double d45 = S4.dot(S5);
        
        Eigen::Vector3d dH = Eigen::Vector3d::Zero();
        
        for (int shift = 0; shift < 6; ++shift) {
            size_t pi = shift % 6;
            size_t pj = (shift + 1) % 6;
            size_t pk = (shift + 2) % 6;
            size_t pl = (shift + 3) % 6;
            size_t pm = (shift + 4) % 6;
            size_t pn = (shift + 5) % 6;
            
            const Eigen::Vector3d& Si = spins[hex[pi]];
            const Eigen::Vector3d& Sj_perm = spins[hex[pj]];
            const Eigen::Vector3d& Sk = spins[hex[pk]];
            const Eigen::Vector3d& Sl = spins[hex[pl]];
            const Eigen::Vector3d& Sm = spins[hex[pm]];
            const Eigen::Vector3d& Sn = spins[hex[pn]];
            
            double dij = Si.dot(Sj_perm);
            double dik = Si.dot(Sk);
            double dil = Si.dot(Sl);
            double djk = Sj_perm.dot(Sk);
            double djl = Sj_perm.dot(Sl);
            double djm = Sj_perm.dot(Sm);
            double dkl = Sk.dot(Sl);
            double dkn = Sk.dot(Sn);
            double dln = Sl.dot(Sn);
            double dmn = Sm.dot(Sn);
            
            int role = -1;
            if (pos == pi) role = 0;
            else if (pos == pj) role = 1;
            else if (pos == pk) role = 2;
            else if (pos == pl) role = 3;
            else if (pos == pm) role = 4;
            else if (pos == pn) role = 5;
            
            if (role == 0) {
                dH += 2.0 * Sj_perm * dkl * dmn;
                dH += -6.0 * Sk * djl * dmn;
                dH += 3.0 * Sl * djk * dmn;
                dH += 3.0 * Sk * djm * dln;
                dH += -Sl * djm * dkn;
            }
            else if (role == 1) {
                dH += 2.0 * Si * dkl * dmn;
                dH += -6.0 * Sl * dik * dmn;
                dH += 3.0 * Sk * dil * dmn;
                dH += 3.0 * Sm * dik * dln;
                dH += -Sm * dil * dkn;
            }
            else if (role == 2) {
                dH += 2.0 * Sl * dij * dmn;
                dH += -6.0 * Si * djl * dmn;
                dH += 3.0 * Sj_perm * dil * dmn;
                dH += 3.0 * Si * djm * dln;
                dH += -Sn * dil * djm;
            }
            else if (role == 3) {
                dH += 2.0 * Sk * dij * dmn;
                dH += -6.0 * Sj_perm * dik * dmn;
                dH += 3.0 * Si * djk * dmn;
                dH += 3.0 * Sn * dik * djm;
                dH += -Si * djm * dkn;
            }
            else if (role == 4) {
                dH += 2.0 * Sn * dij * dkl;
                dH += -6.0 * Sn * dik * djl;
                dH += 3.0 * Sn * dil * djk;
                dH += 3.0 * Sj_perm * dik * dln;
                dH += -Sj_perm * dil * dkn;
            }
            else if (role == 5) {
                dH += 2.0 * Sm * dij * dkl;
                dH += -6.0 * Sm * dik * djl;
                dH += 3.0 * Sm * dil * djk;
                dH += 3.0 * Sl * dik * djm;
                dH += -Sk * dil * djm;
            }
        }
        
        H += prefactor * dH;
    }
    
    return H;
}

SpinVector StrainPhononLattice::get_local_field(size_t site, double t) const {
    // Time-dependent version that uses effective J7(t)
    SpinVector H = Eigen::Vector3d::Zero();
    
    // External field
    H += field[site];
    
    // NN exchange
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        H -= nn_interaction[site][n] * spins[j];
    }
    
    // 2nd NN
    for (size_t n = 0; n < j2_partners[site].size(); ++n) {
        size_t j = j2_partners[site][n];
        H -= j2_interaction[site][n] * spins[j];
    }
    
    // 3rd NN
    for (size_t n = 0; n < j3_partners[site].size(); ++n) {
        size_t j = j3_partners[site][n];
        H -= j3_interaction[site][n] * spins[j];
    }
    
    // Time-dependent ring exchange
    H += get_ring_exchange_field(site, t);
    
    // Magnetoelastic coupling
    H += get_magnetoelastic_field(site);
    
    return H;
}

// ============================================================
// EQUATIONS OF MOTION
// ============================================================

void StrainPhononLattice::strain_derivatives(
    const StrainState& eps, double t,
    const double* dH_deps_xx_arr,
    const double* dH_deps_yy_arr,
    const double* dH_deps_xy_arr,
    StrainState& deps_dt) const
{
    // Strain equations of motion:
    // M d²ε_xx/dt² = -(C11 ε_xx + C12 ε_yy) - ∂H_c/∂ε_xx - γ V_xx + F_A1g + F_Eg1
    // M d²ε_yy/dt² = -(C11 ε_yy + C12 ε_xx) - ∂H_c/∂ε_yy - γ V_yy + F_A1g - F_Eg1
    // M d²ε_xy/dt² = -4C44 ε_xy - ∂H_c/∂ε_xy - γ V_xy + F_Eg2
    
    double C11 = elastic_params.C11;
    double C12 = elastic_params.C12;
    double C44 = elastic_params.C44;
    double M = elastic_params.M;
    double gamma_A1g = elastic_params.gamma_A1g;
    double gamma_Eg = elastic_params.gamma_Eg;
    
    // Get drive forces
    double F_A1g = drive_params.A1g_force(t);
    double F_Eg1 = drive_params.Eg1_force(t);
    double F_Eg2 = drive_params.Eg2_force(t);
    
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        double exx = eps.epsilon_xx[b];
        double eyy = eps.epsilon_yy[b];
        double exy = eps.epsilon_xy[b];
        
        // Position derivatives = velocities
        deps_dt.epsilon_xx[b] = eps.V_xx[b];
        deps_dt.epsilon_yy[b] = eps.V_yy[b];
        deps_dt.epsilon_xy[b] = eps.V_xy[b];
        
        // Velocity derivatives (accelerations)
        // For A1g mode (ε_xx + ε_yy): use gamma_A1g
        // For Eg mode (ε_xx - ε_yy, ε_xy): use gamma_Eg
        
        // ε_xx has both A1g and Eg character
        double elastic_force_xx = -(C11 * exx + C12 * eyy);
        deps_dt.V_xx[b] = (elastic_force_xx - dH_deps_xx_arr[b] 
                         - 0.5 * (gamma_A1g + gamma_Eg) * eps.V_xx[b]
                         + F_A1g + F_Eg1) / M;
        
        // ε_yy has both A1g and Eg character  
        double elastic_force_yy = -(C11 * eyy + C12 * exx);
        deps_dt.V_yy[b] = (elastic_force_yy - dH_deps_yy_arr[b]
                         - 0.5 * (gamma_A1g + gamma_Eg) * eps.V_yy[b]
                         + F_A1g - F_Eg1) / M;
        
        // ε_xy is pure Eg
        double elastic_force_xy = -4.0 * C44 * exy;
        deps_dt.V_xy[b] = (elastic_force_xy - dH_deps_xy_arr[b]
                         - gamma_Eg * eps.V_xy[b]
                         + F_Eg2) / M;
    }
}

void StrainPhononLattice::ode_system(const ODEState& x, ODEState& dxdt, double t) {
    // State layout: [S0_x, S0_y, S0_z, ..., SN_z, ε_xx_0, ε_xx_1, ε_xx_2, ...]
    
    const size_t spin_offset = spin_dim * lattice_size;
    
    // Extract strain state
    StrainState eps;
    eps.from_array(&x[spin_offset]);
    
    // Temporarily update strain for field calculations
    // (Note: this modifies class state during ODE evaluation)
    StrainState saved_strain = strain;
    const_cast<StrainPhononLattice*>(this)->strain = eps;
    
    // Compute magnetoelastic coupling derivatives
    double dH_deps_xx_arr[StrainState::N_BONDS];
    double dH_deps_yy_arr[StrainState::N_BONDS];
    double dH_deps_xy_arr[StrainState::N_BONDS];
    
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        dH_deps_xx_arr[b] = dH_deps_xx(b);
        dH_deps_yy_arr[b] = dH_deps_yy(b);
        dH_deps_xy_arr[b] = dH_deps_xy(b);
    }
    
    // Update spins from state vector
    for (size_t i = 0; i < lattice_size; ++i) {
        const size_t idx = i * spin_dim;
        const_cast<StrainPhononLattice*>(this)->spins[i] = 
            Eigen::Vector3d(x[idx], x[idx+1], x[idx+2]);
    }
    
    // Add J7-ring exchange force: ∂H_J7/∂ε = (dJ7/dε) * E_ring
    // This term was previously missing and caused energy non-conservation
    double dJ7_deps_xx[StrainState::N_BONDS];
    double dJ7_deps_yy[StrainState::N_BONDS];
    double dJ7_deps_xy[StrainState::N_BONDS];
    get_dJ7_deps(dJ7_deps_xx, dJ7_deps_yy, dJ7_deps_xy);
    
    double E_ring = get_ring_exchange_normalized();
    
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        dH_deps_xx_arr[b] += dJ7_deps_xx[b] * E_ring;
        dH_deps_yy_arr[b] += dJ7_deps_yy[b] * E_ring;
        dH_deps_xy_arr[b] += dJ7_deps_xy[b] * E_ring;
    }
    
    // Update current time for time-dependent parameters
    const_cast<StrainPhononLattice*>(this)->current_time = t;
    
    // Spin equations of motion (use time-dependent local field for J7 modulation)
    for (size_t i = 0; i < lattice_size; ++i) {
        const size_t idx = i * spin_dim;
        Eigen::Vector3d Si(x[idx], x[idx+1], x[idx+2]);
        
        SpinVector H = get_local_field(i, t);  // Time-dependent version
        Eigen::Vector3d dSdt = spin_derivative(Si, H);
        
        dxdt[idx] = dSdt(0);
        dxdt[idx+1] = dSdt(1);
        dxdt[idx+2] = dSdt(2);
    }
    
    // Strain equations of motion
    StrainState deps_dt;
    strain_derivatives(eps, t, dH_deps_xx_arr, dH_deps_yy_arr, dH_deps_xy_arr, deps_dt);
    deps_dt.to_array(&dxdt[spin_offset]);
    
    // Restore original strain
    const_cast<StrainPhononLattice*>(this)->strain = saved_strain;
}

// ============================================================
// INTEGRATION
// ============================================================

void StrainPhononLattice::integrate_rk4(double dt, double t_start, double t_final, 
                                        size_t output_every,
                                        const string& output_dir) {
    // Create output directory
    std::filesystem::create_directories(output_dir);
    
    cout << "Running StrainPhononLattice spin-strain dynamics: t=" << t_start << " → " << t_final << endl;
    cout << "Integration method: rk4" << endl;
    cout << "Step size: " << dt << endl;
    cout << "Save every " << output_every << " steps" << endl;
    
    // Initialize ODE state
    ODEState state(state_size);
    
    // Pack initial state
    for (size_t i = 0; i < lattice_size; ++i) {
        const size_t idx = i * spin_dim;
        state[idx] = spins[i](0);
        state[idx+1] = spins[i](1);
        state[idx+2] = spins[i](2);
    }
    strain.to_array(&state[spin_dim * lattice_size]);
    
    // RK4 integrator
    odeint::runge_kutta4<ODEState> stepper;
    
    // Trajectory storage - observables
    vector<double> times;
    vector<Eigen::Vector3d> M_traj;
    vector<Eigen::Vector3d> M_stag_traj;
    vector<double> energy_traj;
    
    // Strain trajectories - per bond type (like PhononLattice)
    vector<double> eps_xx_0_traj, eps_xx_1_traj, eps_xx_2_traj;
    vector<double> eps_yy_0_traj, eps_yy_1_traj, eps_yy_2_traj;
    vector<double> eps_xy_0_traj, eps_xy_1_traj, eps_xy_2_traj;
    vector<double> V_xx_0_traj, V_xx_1_traj, V_xx_2_traj;
    vector<double> V_yy_0_traj, V_yy_1_traj, V_yy_2_traj;
    vector<double> V_xy_0_traj, V_xy_1_traj, V_xy_2_traj;
    
    // Mode amplitude trajectories
    vector<double> eps_A1g_traj;
    vector<double> eps_Eg1_traj;
    vector<double> eps_Eg2_traj;
    vector<double> J7_eff_traj;
    
    // Full spin configuration trajectory (like PhononLattice /trajectory/spins)
    vector<vector<double>> spin_traj;  // (n_times, lattice_size * 3)
    
    double t = t_start;
    size_t step = 0;
    size_t save_count = 0;
    
    while (t < t_final) {
        if (step % output_every == 0) {
            // Compute observables
            Eigen::Vector3d M = magnetization_local();
            Eigen::Vector3d M_stag = magnetization_local_antiferro();
            double E = total_energy();
            double eps_A1g = A1g_amplitude();
            double eps_Eg1 = Eg1_amplitude();
            double eps_Eg2 = Eg2_amplitude();
            double J7_eff = get_effective_J7(t);
            
            // Store trajectory
            times.push_back(t);
            M_traj.push_back(M);
            M_stag_traj.push_back(M_stag);
            energy_traj.push_back(E);
            eps_A1g_traj.push_back(eps_A1g);
            eps_Eg1_traj.push_back(eps_Eg1);
            eps_Eg2_traj.push_back(eps_Eg2);
            J7_eff_traj.push_back(J7_eff);
            
            // Store per-bond-type strain (like PhononLattice per-bond-type phonon)
            eps_xx_0_traj.push_back(strain.epsilon_xx[0]);
            eps_xx_1_traj.push_back(strain.epsilon_xx[1]);
            eps_xx_2_traj.push_back(strain.epsilon_xx[2]);
            eps_yy_0_traj.push_back(strain.epsilon_yy[0]);
            eps_yy_1_traj.push_back(strain.epsilon_yy[1]);
            eps_yy_2_traj.push_back(strain.epsilon_yy[2]);
            eps_xy_0_traj.push_back(strain.epsilon_xy[0]);
            eps_xy_1_traj.push_back(strain.epsilon_xy[1]);
            eps_xy_2_traj.push_back(strain.epsilon_xy[2]);
            V_xx_0_traj.push_back(strain.V_xx[0]);
            V_xx_1_traj.push_back(strain.V_xx[1]);
            V_xx_2_traj.push_back(strain.V_xx[2]);
            V_yy_0_traj.push_back(strain.V_yy[0]);
            V_yy_1_traj.push_back(strain.V_yy[1]);
            V_yy_2_traj.push_back(strain.V_yy[2]);
            V_xy_0_traj.push_back(strain.V_xy[0]);
            V_xy_1_traj.push_back(strain.V_xy[1]);
            V_xy_2_traj.push_back(strain.V_xy[2]);
            
            // Store full spin configuration (like PhononLattice /trajectory/spins)
            vector<double> spin_snapshot(lattice_size * 3);
            for (size_t i = 0; i < lattice_size; ++i) {
                spin_snapshot[i*3 + 0] = spins[i](0);
                spin_snapshot[i*3 + 1] = spins[i](1);
                spin_snapshot[i*3 + 2] = spins[i](2);
            }
            spin_traj.push_back(std::move(spin_snapshot));
            
            // Progress output
            if (step % (output_every * 10) == 0) {
                cout << "t = " << t << ", E = " << E 
                     << ", |M| = " << M.norm()
                     << ", |M_stag| = " << M_stag.norm()
                     << ", ε_A1g = " << eps_A1g 
                     << ", |ε_Eg| = " << std::sqrt(eps_Eg1*eps_Eg1 + eps_Eg2*eps_Eg2) << endl;
            }
            
            save_count++;
        }
        
        stepper.do_step([this](const ODEState& x, ODEState& dxdt, double t) {
            this->ode_system(x, dxdt, t);
        }, state, t, dt);
        
        t += dt;
        ++step;
        
        // Unpack state
        for (size_t i = 0; i < lattice_size; ++i) {
            const size_t idx = i * spin_dim;
            spins[i] = Eigen::Vector3d(state[idx], state[idx+1], state[idx+2]);
            // Renormalize
            spins[i] = spins[i].normalized() * spin_length;
            state[idx] = spins[i](0);
            state[idx+1] = spins[i](1);
            state[idx+2] = spins[i](2);
        }
        strain.from_array(&state[spin_dim * lattice_size]);
    }
    
    // Save text output (always)
    save_trajectory_txt(output_dir, times, M_traj, M_stag_traj, energy_traj,
                        eps_A1g_traj, eps_Eg1_traj, eps_Eg2_traj, J7_eff_traj);
    
    // Save per-bond-type strain trajectory to separate file
    {
        ofstream strain_file(output_dir + "/strain_per_bond_trajectory.txt");
        strain_file << "# t eps_xx_0 eps_xx_1 eps_xx_2 eps_yy_0 eps_yy_1 eps_yy_2 "
                    << "eps_xy_0 eps_xy_1 eps_xy_2 V_xx_0 V_xx_1 V_xx_2 "
                    << "V_yy_0 V_yy_1 V_yy_2 V_xy_0 V_xy_1 V_xy_2\n";
        strain_file << std::scientific << std::setprecision(12);
        for (size_t i = 0; i < times.size(); ++i) {
            strain_file << times[i] << " "
                        << eps_xx_0_traj[i] << " " << eps_xx_1_traj[i] << " " << eps_xx_2_traj[i] << " "
                        << eps_yy_0_traj[i] << " " << eps_yy_1_traj[i] << " " << eps_yy_2_traj[i] << " "
                        << eps_xy_0_traj[i] << " " << eps_xy_1_traj[i] << " " << eps_xy_2_traj[i] << " "
                        << V_xx_0_traj[i] << " " << V_xx_1_traj[i] << " " << V_xx_2_traj[i] << " "
                        << V_yy_0_traj[i] << " " << V_yy_1_traj[i] << " " << V_yy_2_traj[i] << " "
                        << V_xy_0_traj[i] << " " << V_xy_1_traj[i] << " " << V_xy_2_traj[i] << "\n";
        }
        strain_file.close();
        cout << "  strain_per_bond_trajectory.txt: " << times.size() << " timesteps" << endl;
    }
    
#ifdef HDF5_ENABLED
    // Save HDF5 output (like PhononLattice)
    {
        string hdf5_file = output_dir + "/trajectory.h5";
        cout << "Writing HDF5 trajectory to: " << hdf5_file << endl;
        
        H5::H5File h5file(hdf5_file, H5F_ACC_TRUNC);
        
        // Create trajectory group
        H5::Group traj_group = h5file.createGroup("/trajectory");
        
        // Write times
        hsize_t n_times = times.size();
        H5::DataSpace times_space(1, &n_times);
        H5::DataSet times_ds = traj_group.createDataSet("times", H5::PredType::NATIVE_DOUBLE, times_space);
        times_ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write magnetization trajectories (n_times, 3)
        hsize_t mag_dims[2] = {n_times, 3};
        H5::DataSpace mag_space(2, mag_dims);
        vector<double> mag_flat(n_times * 3);
        for (size_t i = 0; i < n_times; ++i) {
            mag_flat[i*3 + 0] = M_traj[i](0);
            mag_flat[i*3 + 1] = M_traj[i](1);
            mag_flat[i*3 + 2] = M_traj[i](2);
        }
        H5::DataSet mag_ds = traj_group.createDataSet("magnetization_local", H5::PredType::NATIVE_DOUBLE, mag_space);
        mag_ds.write(mag_flat.data(), H5::PredType::NATIVE_DOUBLE);
        
        vector<double> mag_stag_flat(n_times * 3);
        for (size_t i = 0; i < n_times; ++i) {
            mag_stag_flat[i*3 + 0] = M_stag_traj[i](0);
            mag_stag_flat[i*3 + 1] = M_stag_traj[i](1);
            mag_stag_flat[i*3 + 2] = M_stag_traj[i](2);
        }
        H5::DataSet mag_stag_ds = traj_group.createDataSet("magnetization_antiferro", H5::PredType::NATIVE_DOUBLE, mag_space);
        mag_stag_ds.write(mag_stag_flat.data(), H5::PredType::NATIVE_DOUBLE);
        
        // Write full spin configuration trajectory (n_times, n_sites, 3) - like PhononLattice
        hsize_t spin_dims[3] = {n_times, lattice_size, 3};
        H5::DataSpace spin_space(3, spin_dims);
        vector<double> spin_flat(n_times * lattice_size * 3);
        for (size_t t_idx = 0; t_idx < n_times; ++t_idx) {
            for (size_t i = 0; i < lattice_size; ++i) {
                spin_flat[t_idx * lattice_size * 3 + i * 3 + 0] = spin_traj[t_idx][i*3 + 0];
                spin_flat[t_idx * lattice_size * 3 + i * 3 + 1] = spin_traj[t_idx][i*3 + 1];
                spin_flat[t_idx * lattice_size * 3 + i * 3 + 2] = spin_traj[t_idx][i*3 + 2];
            }
        }
        H5::DataSet spin_ds = traj_group.createDataSet("spins", H5::PredType::NATIVE_DOUBLE, spin_space);
        spin_ds.write(spin_flat.data(), H5::PredType::NATIVE_DOUBLE);
        
        traj_group.close();
        
        // Create strain trajectory group
        H5::Group strain_group = h5file.createGroup("/strain_trajectory");
        H5::DataSpace scalar_traj_space(1, &n_times);
        
        auto write_dataset = [&](const string& name, const vector<double>& data) {
            H5::DataSet ds = strain_group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, scalar_traj_space);
            ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };
        
        // Strain tensor components - per bond type (like PhononLattice per-bond phonon)
        write_dataset("eps_xx_0", eps_xx_0_traj);
        write_dataset("eps_xx_1", eps_xx_1_traj);
        write_dataset("eps_xx_2", eps_xx_2_traj);
        write_dataset("eps_yy_0", eps_yy_0_traj);
        write_dataset("eps_yy_1", eps_yy_1_traj);
        write_dataset("eps_yy_2", eps_yy_2_traj);
        write_dataset("eps_xy_0", eps_xy_0_traj);
        write_dataset("eps_xy_1", eps_xy_1_traj);
        write_dataset("eps_xy_2", eps_xy_2_traj);
        
        // Velocity components - per bond type
        write_dataset("V_xx_0", V_xx_0_traj);
        write_dataset("V_xx_1", V_xx_1_traj);
        write_dataset("V_xx_2", V_xx_2_traj);
        write_dataset("V_yy_0", V_yy_0_traj);
        write_dataset("V_yy_1", V_yy_1_traj);
        write_dataset("V_yy_2", V_yy_2_traj);
        write_dataset("V_xy_0", V_xy_0_traj);
        write_dataset("V_xy_1", V_xy_1_traj);
        write_dataset("V_xy_2", V_xy_2_traj);
        
        // Mode amplitudes
        write_dataset("eps_A1g", eps_A1g_traj);
        write_dataset("eps_Eg1", eps_Eg1_traj);
        write_dataset("eps_Eg2", eps_Eg2_traj);
        write_dataset("J7_eff", J7_eff_traj);
        write_dataset("energy", energy_traj);
        
        strain_group.close();
        
        // Create metadata group
        H5::Group meta_group = h5file.createGroup("/metadata");
        H5::DataSpace scalar_space(H5S_SCALAR);
        
        auto write_attr_double = [&](const string& name, double val) {
            H5::Attribute attr = meta_group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar_space);
            attr.write(H5::PredType::NATIVE_DOUBLE, &val);
        };
        auto write_attr_int = [&](const string& name, int val) {
            H5::Attribute attr = meta_group.createAttribute(name, H5::PredType::NATIVE_INT, scalar_space);
            attr.write(H5::PredType::NATIVE_INT, &val);
        };
        
        // Lattice parameters
        write_attr_int("dim1", dim1);
        write_attr_int("dim2", dim2);
        write_attr_int("dim3", dim3);
        write_attr_int("lattice_size", lattice_size);
        write_attr_double("spin_length", spin_length);
        
        // Integration parameters
        write_attr_double("dt", dt);
        write_attr_double("t_start", t_start);
        write_attr_double("t_final", t_final);
        write_attr_int("save_interval", output_every);
        
        // Elastic parameters
        write_attr_double("C11", elastic_params.C11);
        write_attr_double("C12", elastic_params.C12);
        write_attr_double("C44", elastic_params.C44);
        write_attr_double("M", elastic_params.M);
        write_attr_double("gamma_A1g", elastic_params.gamma_A1g);
        write_attr_double("gamma_Eg", elastic_params.gamma_Eg);
        
        // Magnetoelastic parameters
        write_attr_double("lambda_A1g", magnetoelastic_params.lambda_A1g);
        write_attr_double("lambda_Eg", magnetoelastic_params.lambda_Eg);
        write_attr_double("J", magnetoelastic_params.J);
        write_attr_double("K", magnetoelastic_params.K);
        write_attr_double("Gamma", magnetoelastic_params.Gamma);
        write_attr_double("Gammap", magnetoelastic_params.Gammap);
        write_attr_double("J7", magnetoelastic_params.J7);
        write_attr_double("gamma_J7", magnetoelastic_params.gamma_J7);
        
        // Drive parameters
        write_attr_double("pump_amplitude", drive_params.E0_1);
        write_attr_double("pump_frequency", drive_params.omega_1);
        write_attr_double("pump_time", drive_params.t_1);
        write_attr_double("pump_width", drive_params.sigma_1);
        write_attr_double("drive_strength_A1g", drive_params.drive_strength_A1g);
        write_attr_double("drive_strength_Eg1", drive_params.drive_strength_Eg1);
        write_attr_double("drive_strength_Eg2", drive_params.drive_strength_Eg2);
        
        // Write positions
        hsize_t pos_dims[2] = {lattice_size, 3};
        H5::DataSpace pos_space(2, pos_dims);
        vector<double> pos_flat(lattice_size * 3);
        for (size_t i = 0; i < lattice_size; ++i) {
            pos_flat[i*3 + 0] = site_positions[i](0);
            pos_flat[i*3 + 1] = site_positions[i](1);
            pos_flat[i*3 + 2] = site_positions[i](2);
        }
        H5::DataSet pos_ds = meta_group.createDataSet("positions", H5::PredType::NATIVE_DOUBLE, pos_space);
        pos_ds.write(pos_flat.data(), H5::PredType::NATIVE_DOUBLE);
        
        meta_group.close();
        h5file.close();
        
        cout << "HDF5 trajectory saved with " << save_count << " snapshots" << endl;
    }
#endif
    
    cout << "Integration complete! Saved " << times.size() << " snapshots to " << output_dir << endl;
}

void StrainPhononLattice::integrate_adaptive(double dt_init, double t_start, double t_final,
                                             double abs_tol, double rel_tol,
                                             size_t output_every,
                                             const string& output_dir) {
    // Similar to integrate_rk4 but with adaptive stepping
    // ... (implementation similar to PhononLattice)
}

// ============================================================
// LANGEVIN DYNAMICS  (stochastic Heun integrator)
// ============================================================

void StrainPhononLattice::integrate_langevin(double dt, double t_start, double t_final,
                                             size_t output_every,
                                             const string& output_dir) {
    // ----------------------------------------------------------
    // Validate inputs
    // ----------------------------------------------------------
    if (alpha_gilbert <= 0.0) {
        cerr << "WARNING: alpha_gilbert = " << alpha_gilbert
             << " ≤ 0 — Langevin dynamics requires positive damping for thermalization.\n"
             << "         Setting alpha_gilbert = 0.01 as fallback." << endl;
        alpha_gilbert = 0.01;
    }
    if (langevin_temperature < 0.0) {
        cerr << "ERROR: langevin_temperature < 0 — nonsensical. Aborting." << endl;
        return;
    }

    std::filesystem::create_directories(output_dir);

    cout << "Running StrainPhononLattice Langevin dynamics: t=" << t_start
         << " → " << t_final << endl;
    cout << "Integration method: stochastic Heun (predictor-corrector)" << endl;
    cout << "Step size: " << dt << endl;
    cout << "Temperature (k_B T): " << langevin_temperature << endl;
    cout << "Gilbert damping α: " << alpha_gilbert << endl;
    cout << "Save every " << output_every << " steps" << endl;

    // ----------------------------------------------------------
    // Noise amplitude prefactors
    //   Spin noise:   σ_spin  = sqrt(2 α k_B T / (|S| dt))
    //                 (each Cartesian component gets N(0, σ_spin) per step)
    //   Strain noise: σ_strain = sqrt(2 γ k_B T / dt)
    //                 (one random force per strain velocity DOF per step)
    // ----------------------------------------------------------
    const double kBT = langevin_temperature;
    const double spin_noise_sigma =
        (kBT > 0.0) ? std::sqrt(2.0 * alpha_gilbert * kBT / (spin_length * dt)) : 0.0;

    const double strain_noise_sigma_A1g =
        (kBT > 0.0) ? std::sqrt(2.0 * elastic_params.gamma_A1g * kBT / dt) : 0.0;
    const double strain_noise_sigma_Eg =
        (kBT > 0.0) ? std::sqrt(2.0 * elastic_params.gamma_Eg * kBT / dt) : 0.0;

    // ----------------------------------------------------------
    // Trajectory storage (same as integrate_rk4)
    // ----------------------------------------------------------
    vector<double> times;
    vector<Eigen::Vector3d> M_traj, M_stag_traj;
    vector<double> energy_traj;
    vector<double> eps_xx_0_traj, eps_xx_1_traj, eps_xx_2_traj;
    vector<double> eps_yy_0_traj, eps_yy_1_traj, eps_yy_2_traj;
    vector<double> eps_xy_0_traj, eps_xy_1_traj, eps_xy_2_traj;
    vector<double> V_xx_0_traj, V_xx_1_traj, V_xx_2_traj;
    vector<double> V_yy_0_traj, V_yy_1_traj, V_yy_2_traj;
    vector<double> V_xy_0_traj, V_xy_1_traj, V_xy_2_traj;
    vector<double> eps_A1g_traj, eps_Eg1_traj, eps_Eg2_traj, J7_eff_traj;
    vector<vector<double>> spin_traj;

    double t = t_start;
    size_t step = 0;
    size_t save_count = 0;

    // ----------------------------------------------------------
    // Helper lambdas for one-site spin RHS
    // ----------------------------------------------------------
    // deterministic + noise part of LLG for a single spin
    auto spin_rhs = [&](const Eigen::Vector3d& S, const Eigen::Vector3d& H_eff,
                        const Eigen::Vector3d& xi) -> Eigen::Vector3d {
        Eigen::Vector3d H_total = H_eff + xi;
        Eigen::Vector3d dSdt = S.cross(H_total);
        dSdt -= (alpha_gilbert / spin_length) * S.cross(S.cross(H_total));
        return dSdt;
    };

    // ----------------------------------------------------------
    // Helper: compute full RHS vector for spins + strain,
    //   given the SAME noise realization (ξ for spins, η for strain).
    //   Operates on the member arrays spins[] and strain directly.
    // ----------------------------------------------------------
    // Noise vectors (allocated once, reused every step)
    vector<Eigen::Vector3d> xi_noise(lattice_size);       // spin noise per site
    vector<double> eta_xx(StrainState::N_BONDS, 0.0);     // strain noise
    vector<double> eta_yy(StrainState::N_BONDS, 0.0);
    vector<double> eta_xy(StrainState::N_BONDS, 0.0);

    // RHS buffers
    vector<Eigen::Vector3d> spin_deriv(lattice_size);     // dS/dt per site
    StrainState strain_deriv;                              // dε/dt

    auto compute_full_rhs = [&](double time,
                                vector<Eigen::Vector3d>& out_spin_deriv,
                                StrainState& out_strain_deriv) {
        current_time = time;

        // -- magnetoelastic strain derivatives --
        double dH_deps_xx_arr[StrainState::N_BONDS];
        double dH_deps_yy_arr[StrainState::N_BONDS];
        double dH_deps_xy_arr[StrainState::N_BONDS];
        for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
            dH_deps_xx_arr[b] = dH_deps_xx(b);
            dH_deps_yy_arr[b] = dH_deps_yy(b);
            dH_deps_xy_arr[b] = dH_deps_xy(b);
        }

        // J7-ring exchange strain force
        double dJ7_deps_xx[StrainState::N_BONDS];
        double dJ7_deps_yy[StrainState::N_BONDS];
        double dJ7_deps_xy[StrainState::N_BONDS];
        get_dJ7_deps(dJ7_deps_xx, dJ7_deps_yy, dJ7_deps_xy);
        double E_ring = get_ring_exchange_normalized();
        for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
            dH_deps_xx_arr[b] += dJ7_deps_xx[b] * E_ring;
            dH_deps_yy_arr[b] += dJ7_deps_yy[b] * E_ring;
            dH_deps_xy_arr[b] += dJ7_deps_xy[b] * E_ring;
        }

        // -- Strain EOM (deterministic part from strain_derivatives) --
        strain_derivatives(strain, time, dH_deps_xx_arr, dH_deps_yy_arr, dH_deps_xy_arr,
                           out_strain_deriv);

        // Add thermal noise to strain velocity derivatives: η / M
        for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
            out_strain_deriv.V_xx[b] += eta_xx[b] / elastic_params.M;
            out_strain_deriv.V_yy[b] += eta_yy[b] / elastic_params.M;
            out_strain_deriv.V_xy[b] += eta_xy[b] / elastic_params.M;
        }

        // -- Spin sLLG --
        for (size_t i = 0; i < lattice_size; ++i) {
            SpinVector H = get_local_field(i, time);
            out_spin_deriv[i] = spin_rhs(spins[i], H, xi_noise[i]);
        }
    };

    // ----------------------------------------------------------
    // Main integration loop — stochastic Heun
    // ----------------------------------------------------------
    // Saved copies for predictor-corrector
    vector<Eigen::Vector3d> spins_saved(lattice_size);
    StrainState strain_saved;
    vector<Eigen::Vector3d> spin_deriv_pred(lattice_size);
    StrainState strain_deriv_pred;

    while (t < t_final) {
        // ---- record observables ----
        if (step % output_every == 0) {
            Eigen::Vector3d M = magnetization_local();
            Eigen::Vector3d M_stag = magnetization_local_antiferro();
            double E = total_energy();
            double eA1g = A1g_amplitude();
            double eEg1 = Eg1_amplitude();
            double eEg2 = Eg2_amplitude();
            double J7e = get_effective_J7(t);

            times.push_back(t);
            M_traj.push_back(M);
            M_stag_traj.push_back(M_stag);
            energy_traj.push_back(E);
            eps_A1g_traj.push_back(eA1g);
            eps_Eg1_traj.push_back(eEg1);
            eps_Eg2_traj.push_back(eEg2);
            J7_eff_traj.push_back(J7e);

            eps_xx_0_traj.push_back(strain.epsilon_xx[0]);
            eps_xx_1_traj.push_back(strain.epsilon_xx[1]);
            eps_xx_2_traj.push_back(strain.epsilon_xx[2]);
            eps_yy_0_traj.push_back(strain.epsilon_yy[0]);
            eps_yy_1_traj.push_back(strain.epsilon_yy[1]);
            eps_yy_2_traj.push_back(strain.epsilon_yy[2]);
            eps_xy_0_traj.push_back(strain.epsilon_xy[0]);
            eps_xy_1_traj.push_back(strain.epsilon_xy[1]);
            eps_xy_2_traj.push_back(strain.epsilon_xy[2]);
            V_xx_0_traj.push_back(strain.V_xx[0]);
            V_xx_1_traj.push_back(strain.V_xx[1]);
            V_xx_2_traj.push_back(strain.V_xx[2]);
            V_yy_0_traj.push_back(strain.V_yy[0]);
            V_yy_1_traj.push_back(strain.V_yy[1]);
            V_yy_2_traj.push_back(strain.V_yy[2]);
            V_xy_0_traj.push_back(strain.V_xy[0]);
            V_xy_1_traj.push_back(strain.V_xy[1]);
            V_xy_2_traj.push_back(strain.V_xy[2]);

            vector<double> snap(lattice_size * 3);
            for (size_t i = 0; i < lattice_size; ++i) {
                snap[i*3+0] = spins[i](0);
                snap[i*3+1] = spins[i](1);
                snap[i*3+2] = spins[i](2);
            }
            spin_traj.push_back(std::move(snap));

            if (step % (output_every * 10) == 0) {
                cout << "t = " << t << ", E = " << E
                     << ", |M| = " << M.norm()
                     << ", |M_stag| = " << M_stag.norm()
                     << ", ε_A1g = " << eA1g
                     << ", |ε_Eg| = " << std::sqrt(eEg1*eEg1 + eEg2*eEg2) << endl;
            }
            save_count++;
        }

        // ======================================================
        // 1. Draw noise (held constant across predictor & corrector)
        // ======================================================
        for (size_t i = 0; i < lattice_size; ++i) {
            xi_noise[i] = Eigen::Vector3d(normal_dist(rng), normal_dist(rng), normal_dist(rng))
                          * spin_noise_sigma;
        }
        for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
            // ε_xx and ε_yy have mixed A1g/Eg character → average damping
            double sigma_mixed = 0.5 * (strain_noise_sigma_A1g + strain_noise_sigma_Eg);
            eta_xx[b] = normal_dist(rng) * sigma_mixed;
            eta_yy[b] = normal_dist(rng) * sigma_mixed;
            eta_xy[b] = normal_dist(rng) * strain_noise_sigma_Eg;  // pure Eg
        }

        // ======================================================
        // 2. Save current state
        // ======================================================
        for (size_t i = 0; i < lattice_size; ++i) spins_saved[i] = spins[i];
        strain_saved = strain;

        // ======================================================
        // 3. Predictor: evaluate RHS at current state
        // ======================================================
        compute_full_rhs(t, spin_deriv, strain_deriv);

        // Euler step → predicted state
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = spins_saved[i] + dt * spin_deriv[i];
            spins[i] = spins[i].normalized() * spin_length;
        }
        for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
            strain.epsilon_xx[b] = strain_saved.epsilon_xx[b] + dt * strain_deriv.epsilon_xx[b];
            strain.epsilon_yy[b] = strain_saved.epsilon_yy[b] + dt * strain_deriv.epsilon_yy[b];
            strain.epsilon_xy[b] = strain_saved.epsilon_xy[b] + dt * strain_deriv.epsilon_xy[b];
            strain.V_xx[b] = strain_saved.V_xx[b] + dt * strain_deriv.V_xx[b];
            strain.V_yy[b] = strain_saved.V_yy[b] + dt * strain_deriv.V_yy[b];
            strain.V_xy[b] = strain_saved.V_xy[b] + dt * strain_deriv.V_xy[b];
        }

        // ======================================================
        // 4. Corrector: evaluate RHS at predicted state (same noise)
        // ======================================================
        compute_full_rhs(t + dt, spin_deriv_pred, strain_deriv_pred);

        // ======================================================
        // 5. Heun update: average predictor & corrector
        // ======================================================
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = spins_saved[i]
                + 0.5 * dt * (spin_deriv[i] + spin_deriv_pred[i]);
            spins[i] = spins[i].normalized() * spin_length;
        }
        for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
            strain.epsilon_xx[b] = strain_saved.epsilon_xx[b]
                + 0.5 * dt * (strain_deriv.epsilon_xx[b] + strain_deriv_pred.epsilon_xx[b]);
            strain.epsilon_yy[b] = strain_saved.epsilon_yy[b]
                + 0.5 * dt * (strain_deriv.epsilon_yy[b] + strain_deriv_pred.epsilon_yy[b]);
            strain.epsilon_xy[b] = strain_saved.epsilon_xy[b]
                + 0.5 * dt * (strain_deriv.epsilon_xy[b] + strain_deriv_pred.epsilon_xy[b]);
            strain.V_xx[b] = strain_saved.V_xx[b]
                + 0.5 * dt * (strain_deriv.V_xx[b] + strain_deriv_pred.V_xx[b]);
            strain.V_yy[b] = strain_saved.V_yy[b]
                + 0.5 * dt * (strain_deriv.V_yy[b] + strain_deriv_pred.V_yy[b]);
            strain.V_xy[b] = strain_saved.V_xy[b]
                + 0.5 * dt * (strain_deriv.V_xy[b] + strain_deriv_pred.V_xy[b]);
        }

        t += dt;
        ++step;
    }

    // ----------------------------------------------------------
    // Save output (text)
    // ----------------------------------------------------------
    save_trajectory_txt(output_dir, times, M_traj, M_stag_traj, energy_traj,
                        eps_A1g_traj, eps_Eg1_traj, eps_Eg2_traj, J7_eff_traj);

    {
        ofstream strain_file(output_dir + "/strain_per_bond_trajectory.txt");
        strain_file << "# t eps_xx_0 eps_xx_1 eps_xx_2 eps_yy_0 eps_yy_1 eps_yy_2 "
                    << "eps_xy_0 eps_xy_1 eps_xy_2 V_xx_0 V_xx_1 V_xx_2 "
                    << "V_yy_0 V_yy_1 V_yy_2 V_xy_0 V_xy_1 V_xy_2\n";
        strain_file << std::scientific << std::setprecision(12);
        for (size_t i = 0; i < times.size(); ++i) {
            strain_file << times[i] << " "
                        << eps_xx_0_traj[i] << " " << eps_xx_1_traj[i] << " " << eps_xx_2_traj[i] << " "
                        << eps_yy_0_traj[i] << " " << eps_yy_1_traj[i] << " " << eps_yy_2_traj[i] << " "
                        << eps_xy_0_traj[i] << " " << eps_xy_1_traj[i] << " " << eps_xy_2_traj[i] << " "
                        << V_xx_0_traj[i] << " " << V_xx_1_traj[i] << " " << V_xx_2_traj[i] << " "
                        << V_yy_0_traj[i] << " " << V_yy_1_traj[i] << " " << V_yy_2_traj[i] << " "
                        << V_xy_0_traj[i] << " " << V_xy_1_traj[i] << " " << V_xy_2_traj[i] << "\n";
        }
        strain_file.close();
        cout << "  strain_per_bond_trajectory.txt: " << times.size() << " timesteps" << endl;
    }

#ifdef HDF5_ENABLED
    {
        string hdf5_file = output_dir + "/trajectory.h5";
        cout << "Writing Langevin HDF5 trajectory to: " << hdf5_file << endl;

        H5::H5File h5file(hdf5_file, H5F_ACC_TRUNC);
        H5::Group traj_group = h5file.createGroup("/trajectory");

        hsize_t n_times = times.size();
        H5::DataSpace times_space(1, &n_times);
        H5::DataSet times_ds = traj_group.createDataSet("times", H5::PredType::NATIVE_DOUBLE, times_space);
        times_ds.write(times.data(), H5::PredType::NATIVE_DOUBLE);

        hsize_t mag_dims[2] = {n_times, 3};
        H5::DataSpace mag_space(2, mag_dims);
        vector<double> mag_flat(n_times * 3);
        for (size_t i = 0; i < n_times; ++i) {
            mag_flat[i*3+0] = M_traj[i](0);
            mag_flat[i*3+1] = M_traj[i](1);
            mag_flat[i*3+2] = M_traj[i](2);
        }
        traj_group.createDataSet("magnetization_local", H5::PredType::NATIVE_DOUBLE, mag_space)
            .write(mag_flat.data(), H5::PredType::NATIVE_DOUBLE);

        vector<double> mag_stag_flat(n_times * 3);
        for (size_t i = 0; i < n_times; ++i) {
            mag_stag_flat[i*3+0] = M_stag_traj[i](0);
            mag_stag_flat[i*3+1] = M_stag_traj[i](1);
            mag_stag_flat[i*3+2] = M_stag_traj[i](2);
        }
        traj_group.createDataSet("magnetization_antiferro", H5::PredType::NATIVE_DOUBLE, mag_space)
            .write(mag_stag_flat.data(), H5::PredType::NATIVE_DOUBLE);

        hsize_t spin_dims[3] = {n_times, lattice_size, 3};
        H5::DataSpace spin_space(3, spin_dims);
        vector<double> spin_flat(n_times * lattice_size * 3);
        for (size_t t_idx = 0; t_idx < n_times; ++t_idx) {
            for (size_t i = 0; i < lattice_size; ++i) {
                spin_flat[t_idx * lattice_size * 3 + i*3+0] = spin_traj[t_idx][i*3+0];
                spin_flat[t_idx * lattice_size * 3 + i*3+1] = spin_traj[t_idx][i*3+1];
                spin_flat[t_idx * lattice_size * 3 + i*3+2] = spin_traj[t_idx][i*3+2];
            }
        }
        traj_group.createDataSet("spins", H5::PredType::NATIVE_DOUBLE, spin_space)
            .write(spin_flat.data(), H5::PredType::NATIVE_DOUBLE);
        traj_group.close();

        H5::Group strain_group = h5file.createGroup("/strain_trajectory");
        H5::DataSpace scalar_traj_space(1, &n_times);
        auto write_ds = [&](const string& name, const vector<double>& data) {
            strain_group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, scalar_traj_space)
                .write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };
        write_ds("eps_xx_0", eps_xx_0_traj); write_ds("eps_xx_1", eps_xx_1_traj); write_ds("eps_xx_2", eps_xx_2_traj);
        write_ds("eps_yy_0", eps_yy_0_traj); write_ds("eps_yy_1", eps_yy_1_traj); write_ds("eps_yy_2", eps_yy_2_traj);
        write_ds("eps_xy_0", eps_xy_0_traj); write_ds("eps_xy_1", eps_xy_1_traj); write_ds("eps_xy_2", eps_xy_2_traj);
        write_ds("V_xx_0", V_xx_0_traj); write_ds("V_xx_1", V_xx_1_traj); write_ds("V_xx_2", V_xx_2_traj);
        write_ds("V_yy_0", V_yy_0_traj); write_ds("V_yy_1", V_yy_1_traj); write_ds("V_yy_2", V_yy_2_traj);
        write_ds("V_xy_0", V_xy_0_traj); write_ds("V_xy_1", V_xy_1_traj); write_ds("V_xy_2", V_xy_2_traj);
        write_ds("eps_A1g", eps_A1g_traj); write_ds("eps_Eg1", eps_Eg1_traj); write_ds("eps_Eg2", eps_Eg2_traj);
        write_ds("J7_eff", J7_eff_traj); write_ds("energy", energy_traj);
        strain_group.close();

        H5::Group meta_group = h5file.createGroup("/metadata");
        H5::DataSpace scalar_space(H5S_SCALAR);
        auto write_attr_d = [&](const string& n, double v) {
            meta_group.createAttribute(n, H5::PredType::NATIVE_DOUBLE, scalar_space)
                .write(H5::PredType::NATIVE_DOUBLE, &v);
        };
        auto write_attr_i = [&](const string& n, int v) {
            meta_group.createAttribute(n, H5::PredType::NATIVE_INT, scalar_space)
                .write(H5::PredType::NATIVE_INT, &v);
        };
        write_attr_i("dim1", dim1); write_attr_i("dim2", dim2); write_attr_i("dim3", dim3);
        write_attr_i("lattice_size", lattice_size);
        write_attr_d("spin_length", spin_length);
        write_attr_d("dt", dt); write_attr_d("t_start", t_start); write_attr_d("t_final", t_final);
        write_attr_i("save_interval", output_every);
        write_attr_d("C11", elastic_params.C11); write_attr_d("C12", elastic_params.C12);
        write_attr_d("C44", elastic_params.C44); write_attr_d("M", elastic_params.M);
        write_attr_d("gamma_A1g", elastic_params.gamma_A1g); write_attr_d("gamma_Eg", elastic_params.gamma_Eg);
        write_attr_d("lambda_A1g", magnetoelastic_params.lambda_A1g);
        write_attr_d("lambda_Eg", magnetoelastic_params.lambda_Eg);
        write_attr_d("J", magnetoelastic_params.J); write_attr_d("K", magnetoelastic_params.K);
        write_attr_d("Gamma", magnetoelastic_params.Gamma); write_attr_d("Gammap", magnetoelastic_params.Gammap);
        write_attr_d("J7", magnetoelastic_params.J7); write_attr_d("gamma_J7", magnetoelastic_params.gamma_J7);
        write_attr_d("alpha_gilbert", alpha_gilbert);
        write_attr_d("langevin_temperature", langevin_temperature);

        // Write positions
        hsize_t pos_dims[2] = {lattice_size, 3};
        H5::DataSpace pos_space(2, pos_dims);
        vector<double> pos_flat(lattice_size * 3);
        for (size_t i = 0; i < lattice_size; ++i) {
            pos_flat[i*3+0] = site_positions[i](0);
            pos_flat[i*3+1] = site_positions[i](1);
            pos_flat[i*3+2] = site_positions[i](2);
        }
        meta_group.createDataSet("positions", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(2, pos_dims))
            .write(pos_flat.data(), H5::PredType::NATIVE_DOUBLE);

        meta_group.close();
        h5file.close();
        cout << "HDF5 Langevin trajectory saved with " << save_count << " snapshots" << endl;
    }
#endif

    cout << "Langevin integration complete! Saved " << times.size()
         << " snapshots to " << output_dir << endl;
}

void StrainPhononLattice::relax_strain(bool verbose) {
    // If strain is pinned (Born-Oppenheimer scan), skip relaxation
    if (fix_strain_) {
        if (verbose) {
            cout << "Strain relaxation SKIPPED (fix_strain = true)" << endl;
            cout << "  Current Eg1 = " << strain.epsilon_xx[0] 
                 << ", Eg2 = " << strain.epsilon_xy[0] << endl;
        }
        return;
    }
    
    // Relax strain to equilibrium given current spin configuration.
    // We ignore A1g channel - only Eg magnetoelastic coupling is active.
    // 
    // The total Hamiltonian (elastic + Eg magnetoelastic) is:
    // H = (1/2)[C11(ε_xx² + ε_yy²) + 2C12 ε_xx ε_yy + 4C44 ε_xy²] 
    //   + λ_Eg [(ε_xx - ε_yy) f_Eg1 + 2ε_xy f_Eg2]
    //
    // Setting ∂H/∂ε = 0:
    // C11 ε_xx + C12 ε_yy + λ_Eg f_Eg1 = 0
    // C12 ε_xx + C11 ε_yy - λ_Eg f_Eg1 = 0
    // 4C44 ε_xy + 2λ_Eg f_Eg2 = 0
    //
    // From the first two equations (2x2 linear system):
    // [C11  C12] [ε_xx]   [-λ_Eg f_Eg1]
    // [C12  C11] [ε_yy] = [+λ_Eg f_Eg1]
    //
    // Using Cramer's rule with det = C11² - C12²:
    // ε_xx = [-λ_Eg f_Eg1 * C11 - λ_Eg f_Eg1 * C12] / det = -λ_Eg f_Eg1 (C11 + C12) / det
    // ε_yy = [C11 * λ_Eg f_Eg1 + C12 * λ_Eg f_Eg1] / det = +λ_Eg f_Eg1 (C11 + C12) / det
    //
    // ε_xy = -λ_Eg f_Eg2 / (2C44)
    
    double C11 = elastic_params.C11;
    double C12 = elastic_params.C12;
    double C44 = elastic_params.C44;
    
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    // Determinant of the 2x2 elastic matrix for ε_xx, ε_yy
    double det = C11 * C11 - C12 * C12;
    
    // Avoid division by zero
    if (std::abs(det) < 1e-12) {
        std::cerr << "Warning: Elastic matrix is singular, cannot relax strain analytically." << std::endl;
        return;
    }
    
    // Compute Eg spin basis function factors (includes Kitaev, Heisenberg, Gamma, Gamma' terms)
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double Gammap = magnetoelastic_params.Gammap;
    
    double Eg1_spin_factor = (J + K) * f_K_Eg1() + J * f_J_Eg1() 
                           + Gamma * f_Gamma_Eg1() + Gammap * f_Gammap_Eg1();
    double Eg2_spin_factor = (J + K) * f_K_Eg2() + J * f_J_Eg2() 
                           + Gamma * f_Gamma_Eg2() + Gammap * f_Gammap_Eg2();
    
    // Solve for equilibrium strain (Eg-only, same for all bond types)
    // With drive H_drive = -Σ_b [F1*(ε_xx-ε_yy)/2 + F2*ε_xy]:
    //   ∂E/∂ε_xx = C11 ε_xx + C12 ε_yy + λ_Eg Σ_Eg1 - F1/2 = 0
    //   ∂E/∂ε_yy = C12 ε_xx + C11 ε_yy - λ_Eg Σ_Eg1 + F1/2 = 0
    //   ∂E/∂ε_xy = 4C44 ε_xy + 2λ_Eg Σ_Eg2 - F2 = 0
    // Solution (traceless: ε_yy = -ε_xx):
    //   ε_xx = (F1/2 - λ_Eg Σ_Eg1)(C11+C12) / det
    //   ε_yy = -ε_xx
    //   ε_xy = (F2 - 2λ_Eg Σ_Eg2) / (4C44)
    double eps_xx_eq = (drive_F_Eg1_ / 2.0 - lambda_Eg * Eg1_spin_factor) * (C11 + C12) / det;
    double eps_yy_eq = -eps_xx_eq;
    double eps_xy_eq = (drive_F_Eg2_ - 2.0 * lambda_Eg * Eg2_spin_factor) / (4.0 * C44);
    
    // Set all bond types to equilibrium values
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        strain.epsilon_xx[b] = eps_xx_eq;
        strain.epsilon_yy[b] = eps_yy_eq;
        strain.epsilon_xy[b] = eps_xy_eq;
        
        // Zero out velocities
        strain.V_xx[b] = 0.0;
        strain.V_yy[b] = 0.0;
        strain.V_xy[b] = 0.0;
        
        // Store equilibrium values for J7 modulation calculation
        strain_equilibrium.epsilon_xx[b] = eps_xx_eq;
        strain_equilibrium.epsilon_yy[b] = eps_yy_eq;
        strain_equilibrium.epsilon_xy[b] = eps_xy_eq;
        strain_equilibrium.V_xx[b] = 0.0;
        strain_equilibrium.V_yy[b] = 0.0;
        strain_equilibrium.V_xy[b] = 0.0;
    }
    
    if (verbose) {
        std::cout << "Strain relaxed to equilibrium:" << std::endl;
        std::cout << "  ε_xx = " << eps_xx_eq << std::endl;
        std::cout << "  ε_yy = " << eps_yy_eq << std::endl;
        std::cout << "  ε_xy = " << eps_xy_eq << std::endl;
        std::cout << "  A1g = " << eps_xx_eq + eps_yy_eq << std::endl;
        std::cout << "  Eg1 = " << eps_xx_eq - eps_yy_eq << std::endl;
        
        // Diagnostic: print the ∂H/∂ε = 0 analysis
        std::cout << "  --- ∂H/∂ε = 0 diagnostic ---" << std::endl;
        
        // Individual Eg spin basis functions
        double fK1 = f_K_Eg1(), fK2 = f_K_Eg2();
        double fJ1 = f_J_Eg1(), fJ2 = f_J_Eg2();
        double fG1 = f_Gamma_Eg1(), fG2 = f_Gamma_Eg2();
        double fGp1 = f_Gammap_Eg1(), fGp2 = f_Gammap_Eg2();
        
        std::cout << "  Spin basis functions (Eg irrep):" << std::endl;
        std::cout << "    f_K^{Eg,1}  = " << fK1 << ",  f_K^{Eg,2}  = " << fK2 << std::endl;
        std::cout << "    f_J^{Eg,1}  = " << fJ1 << ",  f_J^{Eg,2}  = " << fJ2 << std::endl;
        std::cout << "    f_Γ^{Eg,1}  = " << fG1 << ",  f_Γ^{Eg,2}  = " << fG2 << std::endl;
        std::cout << "    f_Γ'^{Eg,1} = " << fGp1 << ", f_Γ'^{Eg,2} = " << fGp2 << std::endl;
        
        std::cout << "  Composite Eg spin factors:" << std::endl;
        std::cout << "    Σ_Eg1 = (J+K)f_K1 + J·f_J1 + Γ·f_Γ1 + Γ'·f_Γ'1 = " << Eg1_spin_factor << std::endl;
        std::cout << "    Σ_Eg2 = (J+K)f_K2 + J·f_J2 + Γ·f_Γ2 + Γ'·f_Γ'2 = " << Eg2_spin_factor << std::endl;
        
        // Show the ∂H/∂ε = 0 equations and their solutions
        std::cout << "  Stationarity conditions ∂H/∂ε = 0:" << std::endl;
        std::cout << "    C11·ε_xx + C12·ε_yy + λ_Eg·Σ_Eg1 = 0" << std::endl;
        std::cout << "    C12·ε_xx + C11·ε_yy - λ_Eg·Σ_Eg1 = 0" << std::endl;
        std::cout << "    4·C44·ε_xy + 2·λ_Eg·Σ_Eg2 = 0" << std::endl;
        std::cout << "  Solution (det = C11²-C12² = " << det << "):" << std::endl;
        std::cout << "    ε_xx* = -λ_Eg·Σ_Eg1·(C11+C12)/det = " << eps_xx_eq << std::endl;
        std::cout << "    ε_yy* = +λ_Eg·Σ_Eg1·(C11+C12)/det = " << eps_yy_eq << std::endl;
        std::cout << "    ε_xy* = -λ_Eg·Σ_Eg2/(2·C44)       = " << eps_xy_eq << std::endl;
        
        // Verify: evaluate ∂H/∂ε at the solved equilibrium
        double residual_xx = C11 * eps_xx_eq + C12 * eps_yy_eq + lambda_Eg * Eg1_spin_factor;
        double residual_yy = C12 * eps_xx_eq + C11 * eps_yy_eq - lambda_Eg * Eg1_spin_factor;
        double residual_xy = 4.0 * C44 * eps_xy_eq + 2.0 * lambda_Eg * Eg2_spin_factor;
        std::cout << "  Verification (∂H/∂ε at ε*):" << std::endl;
        std::cout << "    ∂H/∂ε_xx = " << residual_xx << std::endl;
        std::cout << "    ∂H/∂ε_yy = " << residual_yy << std::endl;
        std::cout << "    ∂H/∂ε_xy = " << residual_xy << std::endl;
        std::cout << "  ---------------------------------" << std::endl;
    }
}

// ============================================================
// MONTE CARLO
// ============================================================

SpinVector StrainPhononLattice::gen_random_spin(float spin_l) {
    // Efficient sphere sampling for 3D using cylindrical projection
    // (same method as Lattice::gen_random_spin in lattice.h)
    SpinVector spin(spin_dim);  // spin_dim = 3 for this class
    
    double z = uniform_dist(rng) * 2.0 - 1.0;  // uniform in [-1, 1]
    double phi = uniform_dist(rng) * 2.0 * M_PI;  // uniform in [0, 2π]
    double r = std::sqrt(1.0 - z * z);
    
    spin(0) = r * std::cos(phi);
    spin(1) = r * std::sin(phi);
    spin(2) = z;
    
    return spin * spin_l;
}

SpinVector StrainPhononLattice::gaussian_spin_move(const SpinVector& current_spin, double sigma) {
    // Perturb current spin by adding a random direction scaled by sigma
    SpinVector perturbation = gen_random_spin(1.0) * sigma;
    
    SpinVector new_spin = current_spin + perturbation;
    double norm = new_spin.norm();
    if (norm < 1e-10) return current_spin;
    return new_spin * (spin_length / norm);
}

double StrainPhononLattice::metropolis(double temperature, bool gaussian_move, double sigma) {
    if (temperature <= 0) return 0.0;
    
    const double beta = 1.0 / temperature;
    size_t accepted = 0;
    
    // Use random site selection like in lattice.h for better sampling
    std::uniform_int_distribution<size_t> site_dist(0, lattice_size - 1);
    
    for (size_t sweep_step = 0; sweep_step < lattice_size; ++sweep_step) {
        // Random site selection
        size_t site = site_dist(rng);
        
        // Generate new spin
        SpinVector new_spin;
        if (gaussian_move) {
            new_spin = gaussian_spin_move(spins[site], sigma);
        } else {
            // Propose a completely random spin on the sphere
            new_spin = gen_random_spin(spin_length);
        }
        
        // Compute energy difference using explicit site_energy_diff
        // This computes E(new_spin) - E(old_spin) for all terms:
        // exchange (NN, 2nd NN, 3rd NN) + magnetoelastic + ring exchange
        SpinVector old_spin = spins[site];
        double dE = site_energy_diff(new_spin, old_spin, site);
        
        // Metropolis acceptance
        double rand_uniform = uniform_dist(rng);
        const bool accept = (dE < 0.0) || (rand_uniform < std::exp(-beta * dE));
        if (accept) {
            spins[site] = new_spin;
            ++accepted;
        }
    }
    
    // Relax strain to equilibrium for the updated spin configuration.
    // This ensures the magnetoelastic field (used in site_energy_diff) and
    // total_energy() (used in replica exchange criteria) remain self-consistent
    // under the Born-Oppenheimer approximation.
    relax_strain(false);
    
    return double(accepted) / double(lattice_size);
}

void StrainPhononLattice::overrelaxation() {
    // Over-relaxation sweep: reflect each spin about its local field
    // S' = 2(S·H)H/|H|² - S
    // This is a microcanonical update (preserves energy) that accelerates decorrelation
    
    std::uniform_int_distribution<size_t> site_dist(0, lattice_size - 1);
    
    size_t count = 0;
    while (count < lattice_size) {
        size_t site = site_dist(rng);
        
        // Get local field
        SpinVector local_field = get_local_field(site);
        double norm_sq = local_field.dot(local_field);
        
        if (norm_sq < 1e-20) {
            // Zero field, skip this site
            continue;
        }
        
        // Reflect spin: S' = 2(S·H)H/|H|² - S
        double proj = 2.0 * spins[site].dot(local_field) / norm_sq;
        spins[site] = local_field * proj - spins[site];
        
        count++;
    }
    
    // Relax strain after spin updates (Born-Oppenheimer consistency)
    relax_strain(false);
}

// greedy_quench is now an inline wrapper in strain_phonon_lattice.h
// delegating to mc::greedy_quench() from mc_common.h

void StrainPhononLattice::simulated_annealing(double T_start, double T_end, 
                                 size_t n_sweeps,
                                 double cooling_rate,
                                 size_t overrelaxation_rate,
                                 bool gaussian_move,
                                 const string& out_dir,
                                 bool T_zero,
                                 size_t n_deterministics) {
    // Setup output directory
    if (!out_dir.empty()) {
        std::filesystem::create_directories(out_dir);
    }
    
    double T = T_start;
    double sigma = 1000.0;  // Initial Gaussian width (will be adapted)
    
    cout << "Starting simulated annealing: T=" << T_start << " → " << T_end << endl;
    cout << "Cooling rate: " << cooling_rate << ", sweeps per temperature: " << n_sweeps << endl;
    if (overrelaxation_rate > 0) {
        cout << "Overrelaxation enabled: every " << overrelaxation_rate << " sweeps" << endl;
    }
    if (gaussian_move) {
        cout << "Using Gaussian spin moves with adaptive sigma" << endl;
    }
    if (T_zero) {
        cout << "T=0 mode enabled: will perform " << n_deterministics << " deterministic sweeps at T=0" << endl;
    }
    
    // How often to re-relax strain during MC sweeps
    const size_t strain_relax_interval = 1;  // Every sweep
    
    size_t temp_step = 0;
    while (T > T_end) {
        // Perform sweeps at this temperature
        double acc_sum = 0.0;
        for (size_t s = 0; s < n_sweeps; ++s) {
            // Overrelaxation before Metropolis (if enabled)
            if (overrelaxation_rate > 0) {
                overrelaxation();
                if (s % overrelaxation_rate == 0) {
                    acc_sum += metropolis(T, gaussian_move, sigma);
                }
            } else {
                acc_sum += metropolis(T, gaussian_move, sigma);
            }
        }
        
        // Calculate acceptance rate (normalize differently if overrelaxation is used)
        double acceptance = (overrelaxation_rate > 0) ? 
            acc_sum / double(n_sweeps) * overrelaxation_rate : 
            acc_sum / double(n_sweeps);
        
        // Progress report every 10 steps or near T_end
        if (temp_step % 10 == 0 || T <= T_end * 1.5) {
            double E = spin_energy() / lattice_size;  // Energy per spin
            cout << "T=" << std::scientific << T << ", E/N=" << E 
                 << ", acc=" << std::fixed << std::setprecision(3) << acceptance;
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
    
    // Final strain relaxation (verbose)
    relax_strain(true);
    
    // Save spin config after annealing (before deterministic sweeps)
    if (!out_dir.empty()) {
        save_spin_config(out_dir + "/spins_T=" + std::to_string(T_end) + ".txt");
    }
    
    // T=0 deterministic sweeps if requested
    if (T_zero && n_deterministics > 0) {
        cout << "\nPerforming " << n_deterministics << " deterministic sweeps at T=0..." << endl;
        deterministic_sweep(n_deterministics, out_dir);
        relax_strain(true);  // Final verbose strain relaxation
        cout << "Deterministic sweeps completed. Final energy: " << spin_energy() / lattice_size << endl;
        
        // Save final configuration
        if (!out_dir.empty()) {
            save_spin_config(out_dir + "/spins_T=0.txt");
        }
    }
    
    // Save final spin config
    if (!out_dir.empty()) {
        save_spin_config(out_dir + "/spins.txt");
        save_spin_config_global(out_dir + "/spins_global.txt");
        save_strain_state(out_dir + "/strain.txt");
        save_spin_strain_config(out_dir + "/spin_strain_config.txt");  // Combined for GNEB
        save_positions(out_dir + "/positions.txt");
    }
    
    // Output final energies and C3 order breaking parameters
    cout << "\n========== Final State Summary ==========" << endl;
    cout << "Energies:" << endl;
    double E_spin = spin_energy();  // This already includes ring exchange
    double E_strain = strain_energy();
    double E_magnetoelastic = magnetoelastic_energy();
    double E_ring = ring_exchange_energy();  // For display only
    double E_drive = drive_energy();
    cout << std::setprecision(10);
    cout << "  Spin energy (incl. ring): " << std::scientific << E_spin << endl;
    cout << "  Strain elastic energy:    " << std::scientific << E_strain << endl;
    cout << "  Magnetoelastic energy:    " << std::scientific << E_magnetoelastic << endl;
    cout << "  (Ring exchange contrib):  " << std::scientific << E_ring << endl;
    if (std::abs(drive_F_Eg1_) > 1e-15 || std::abs(drive_F_Eg2_) > 1e-15) {
        cout << "  Drive energy:             " << std::scientific << E_drive << endl;
    }
    double total_E = E_spin + E_strain + E_magnetoelastic + E_drive;
    cout << "  Total energy:             " << std::scientific << total_E << endl;
    cout << "  Total energy/site:        " << std::scientific << total_E / lattice_size << endl;
    
    cout << "\nC3 Order Breaking Parameters (D3d Irrep Spin Basis Functions):" << endl;
    cout << "  Eg channel:" << endl;
    cout << "    f_K^{Eg,1}    = " << std::scientific << f_K_Eg1() << endl;
    cout << "    f_K^{Eg,2}    = " << std::scientific << f_K_Eg2() << endl;
    cout << "    f_J^{Eg,1}    = " << std::scientific << f_J_Eg1() << endl;
    cout << "    f_J^{Eg,2}    = " << std::scientific << f_J_Eg2() << endl;
    cout << "    f_Γ^{Eg,1}    = " << std::scientific << f_Gamma_Eg1() << endl;
    cout << "    f_Γ^{Eg,2}    = " << std::scientific << f_Gamma_Eg2() << endl;
    cout << "    f_Γ'^{Eg,1}   = " << std::scientific << f_Gammap_Eg1() << endl;
    cout << "    f_Γ'^{Eg,2}   = " << std::scientific << f_Gammap_Eg2() << endl;
    
    cout << "\nStrain state:" << endl;
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        cout << "  Bond " << b << ": ε_xx=" << std::scientific << strain.epsilon_xx[b]
             << ", ε_yy=" << strain.epsilon_yy[b]
             << ", ε_xy=" << strain.epsilon_xy[b] << endl;
    }
    cout << "==========================================" << endl;
    
    cout << "Annealing complete!" << endl;
}

void StrainPhononLattice::deterministic_sweep(size_t num_sweeps, const string& output_dir) {
    // Deterministic update: align each spin parallel to its local field.
    // This ensures S × H_eff = 0, eliminating precession in dynamics.
    // Uses sequential site visits so every site is updated exactly once per sweep.
    
    // Create a shuffled index array for each sweep (visits each site exactly once)
    std::vector<size_t> site_order(lattice_size);
    std::iota(site_order.begin(), site_order.end(), 0);
    
    const double torque_tol = 1e-14;  // Convergence threshold
    double max_torque = 0.0;
    double mean_torque = 0.0;
    
    // Track convergence history for file output
    std::vector<size_t> sweep_history;
    std::vector<double> max_torque_history;
    std::vector<double> mean_torque_history;
    std::vector<double> energy_history;
    std::vector<size_t> worst_site_history;
    size_t converged_sweep = num_sweeps;
    
    for (size_t sweep = 0; sweep < num_sweeps; ++sweep) {
        // Shuffle site order each sweep to avoid systematic bias
        std::shuffle(site_order.begin(), site_order.end(), rng);
        
        for (size_t idx = 0; idx < lattice_size; ++idx) {
            size_t i = site_order[idx];
            
            // Get local field (includes exchange, magnetoelastic, ring exchange)
            Eigen::Vector3d local_field = get_local_field(i);
            double norm = local_field.norm();
            
            if (norm < 1e-15) {
                // Field is essentially zero, skip this site
                continue;
            }
            
            // Align spin PARALLEL to local field (minimizes energy)
            // H_eff = -∂H/∂Si, so Si should point along H_eff to minimize E = -Si·H_eff
            spins[i] = local_field / norm * spin_length;
        }
        
        // Re-relax strain after each sweep to maintain adiabatic equilibrium
        relax_strain(false);
        
        // Check convergence periodically (every 100 sweeps or at end)
        if ((sweep + 1) % 100 == 0 || sweep == num_sweeps - 1) {
            max_torque = 0.0;
            mean_torque = 0.0;
            size_t worst_site = 0;
            for (size_t i = 0; i < lattice_size; ++i) {
                Eigen::Vector3d H = get_local_field(i);
                Eigen::Vector3d S_i(spins[i](0), spins[i](1), spins[i](2));
                Eigen::Vector3d torque = S_i.cross(H);
                double torque_mag = torque.norm();
                if (torque_mag > max_torque) {
                    max_torque = torque_mag;
                    worst_site = i;
                }
                mean_torque += torque_mag;
            }
            mean_torque /= lattice_size;
            
            // Record history
            sweep_history.push_back(sweep + 1);
            max_torque_history.push_back(max_torque);
            mean_torque_history.push_back(mean_torque);
            energy_history.push_back(spin_energy() / lattice_size);
            worst_site_history.push_back(worst_site);
            
            cout << "  Sweep " << (sweep + 1) << "/" << num_sweeps 
                 << ": max_torque = " << scientific << setprecision(3) << max_torque
                 << " (site " << worst_site << ")"
                 << ", mean_torque = " << mean_torque 
                 << ", E/N = " << fixed << setprecision(6) << spin_energy() / lattice_size
                 << endl;
            
            if (max_torque < torque_tol) {
                cout << "  Converged after " << (sweep + 1) << " sweeps!" << endl;
                converged_sweep = sweep + 1;
                break;
            }
        }
    }
    
    // Final strain relaxation
    relax_strain(true);
    
    // Final detailed torque diagnostic
    max_torque = 0.0;
    mean_torque = 0.0;
    size_t worst_site = 0;
    Eigen::Vector3d worst_torque_vec = Eigen::Vector3d::Zero();
    Eigen::Vector3d worst_H = Eigen::Vector3d::Zero();
    Eigen::Vector3d worst_S = Eigen::Vector3d::Zero();
    
    for (size_t i = 0; i < lattice_size; ++i) {
        Eigen::Vector3d H = get_local_field(i);
        Eigen::Vector3d S_i(spins[i](0), spins[i](1), spins[i](2));
        Eigen::Vector3d torque = S_i.cross(H);
        double torque_mag = torque.norm();
        if (torque_mag > max_torque) {
            max_torque = torque_mag;
            worst_site = i;
            worst_torque_vec = torque;
            worst_H = H;
            worst_S = S_i;
        }
        mean_torque += torque_mag;
    }
    mean_torque /= lattice_size;
    
    cout << "Completed " << num_sweeps << " deterministic sweeps" << endl;
    cout << "  Residual torque: max = " << scientific << setprecision(3) << max_torque 
         << ", mean = " << mean_torque << endl;
    
    if (max_torque > 1e-12) {
        cout << "  Worst site " << worst_site << ":" << endl;
        cout << "    S  = (" << fixed << setprecision(8) << worst_S(0) << ", " << worst_S(1) << ", " << worst_S(2) << ")" << endl;
        cout << "    H  = (" << worst_H(0) << ", " << worst_H(1) << ", " << worst_H(2) << ")" << endl;
        cout << "    τ  = (" << scientific << setprecision(3) << worst_torque_vec(0) << ", " << worst_torque_vec(1) << ", " << worst_torque_vec(2) << ")" << endl;
        cout << "    |S| = " << worst_S.norm() << ", |H| = " << worst_H.norm() << endl;
        cout << "    S·H/|S||H| = " << fixed << setprecision(12) << worst_S.dot(worst_H) / (worst_S.norm() * worst_H.norm()) << endl;
        
        // Check: re-align this site and see if torque drops
        Eigen::Vector3d H_fresh = get_local_field(worst_site);
        spins[worst_site] = H_fresh / H_fresh.norm() * spin_length;
        Eigen::Vector3d H_after = get_local_field(worst_site);
        Eigen::Vector3d S_after(spins[worst_site](0), spins[worst_site](1), spins[worst_site](2));
        Eigen::Vector3d torque_after = S_after.cross(H_after);
        cout << "    After re-align: |τ| = " << scientific << setprecision(3) << torque_after.norm() << endl;
        cout << "    H changed? |ΔH| = " << (H_after - H_fresh).norm() << endl;
    }
    
    if (max_torque > 0.01) {
        cout << "  WARNING: Large residual torque detected! Spins not fully equilibrated." << endl;
        cout << "           Consider increasing n_deterministics or adding Gilbert damping." << endl;
    }
    
    // Save diagnostics to file if output directory is specified
    if (!output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
        string filename = output_dir + "/deterministic_sweep_diagnostics.txt";
        ofstream diag_file(filename);
        if (diag_file.is_open()) {
            diag_file << "# Deterministic Sweep Convergence Diagnostics\n";
            diag_file << "# Lattice size: " << dim1 << " x " << dim2 << " x " << dim3 << " (" << lattice_size << " sites)\n";
            diag_file << "# Tolerance: " << scientific << setprecision(3) << torque_tol << "\n";
            diag_file << "# Converged: " << (max_torque < torque_tol ? "yes" : "no") << "\n";
            diag_file << "# Converged at sweep: " << converged_sweep << "\n";
            diag_file << "#\n";
            diag_file << "# sweep  max_torque  mean_torque  energy_per_spin  worst_site\n";
            for (size_t k = 0; k < sweep_history.size(); ++k) {
                diag_file << fixed << setprecision(0) << sweep_history[k] << "  "
                          << scientific << setprecision(6) << max_torque_history[k] << "  "
                          << mean_torque_history[k] << "  "
                          << fixed << setprecision(8) << energy_history[k] << "  "
                          << worst_site_history[k] << "\n";
            }
            diag_file << "#\n";
            diag_file << "# Final state:\n";
            diag_file << "#   max_torque = " << scientific << setprecision(6) << max_torque << "\n";
            diag_file << "#   mean_torque = " << mean_torque << "\n";
            if (max_torque > 1e-12) {
                diag_file << "#   worst_site = " << worst_site << "\n";
                diag_file << "#   S = (" << fixed << setprecision(8) << worst_S(0) << ", " << worst_S(1) << ", " << worst_S(2) << ")\n";
                diag_file << "#   H = (" << worst_H(0) << ", " << worst_H(1) << ", " << worst_H(2) << ")\n";
                diag_file << "#   tau = (" << scientific << setprecision(6) << worst_torque_vec(0) << ", " << worst_torque_vec(1) << ", " << worst_torque_vec(2) << ")\n";
                diag_file << "#   |S| = " << worst_S.norm() << ", |H| = " << worst_H.norm() << "\n";
                diag_file << "#   S.H/(|S||H|) = " << fixed << setprecision(12) << worst_S.dot(worst_H) / (worst_S.norm() * worst_H.norm()) << "\n";
            }
            diag_file.close();
            cout << "  Diagnostics saved to: " << filename << endl;
        }
    }
}

// ============================================================
// I/O
// ============================================================

void StrainPhononLattice::save_spin_config(const string& filename) const {
    ofstream file(filename);
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        file << spins[i](0) << " " << spins[i](1) << " " << spins[i](2) << "\n";
    }
}

void StrainPhononLattice::load_spin_config(const string& filename) {
    ifstream file(filename);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        file >> spins[i](0) >> spins[i](1) >> spins[i](2);
    }
}

void StrainPhononLattice::save_spin_strain_config(const string& filename) const {
    ofstream file(filename);
    file << std::scientific << std::setprecision(12);
    
    // Header with strain information
    // Compute uniform Eg strain components (averaged over bond types)
    double Eg1 = 0.0, Eg2 = 0.0;
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        Eg1 += (strain.epsilon_xx[b] - strain.epsilon_yy[b]) / 2.0;
        Eg2 += strain.epsilon_xy[b];
    }
    Eg1 /= StrainState::N_BONDS;
    Eg2 /= StrainState::N_BONDS;
    
    file << "# Combined spin-strain configuration for GNEB\n";
    file << "# strain_Eg1 = " << Eg1 << "\n";
    file << "# strain_Eg2 = " << Eg2 << "\n";
    file << "# n_sites = " << lattice_size << "\n";
    file << "# Sx Sy Sz\n";
    
    for (size_t i = 0; i < lattice_size; ++i) {
        file << spins[i](0) << " " << spins[i](1) << " " << spins[i](2) << "\n";
    }
}

void StrainPhononLattice::load_spin_strain_config(const string& filename) {
    ifstream file(filename);
    string line;
    
    double Eg1 = 0.0, Eg2 = 0.0;
    bool found_Eg1 = false, found_Eg2 = false;
    
    // Parse header for strain values
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        if (line[0] == '#') {
            // Look for strain values in comments
            if (line.find("strain_Eg1") != string::npos) {
                size_t eq_pos = line.find('=');
                if (eq_pos != string::npos) {
                    Eg1 = std::stod(line.substr(eq_pos + 1));
                    found_Eg1 = true;
                }
            } else if (line.find("strain_Eg2") != string::npos) {
                size_t eq_pos = line.find('=');
                if (eq_pos != string::npos) {
                    Eg2 = std::stod(line.substr(eq_pos + 1));
                    found_Eg2 = true;
                }
            }
        } else {
            // First non-comment line is spin data - rewind and read
            break;
        }
    }
    
    // Apply uniform Eg strain if found
    if (found_Eg1 || found_Eg2) {
        for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
            strain.epsilon_xx[b] = Eg1;
            strain.epsilon_yy[b] = -Eg1;
            strain.epsilon_xy[b] = Eg2;
        }
        // Note: strain values are used directly in energy/field calculations,
        // no separate update function needed
    }
    
    // Parse spin data starting from current position
    // Support both 3-column (Sx Sy Sz) and 4-column (index Sx Sy Sz) formats
    // Detect format from first non-comment line
    std::istringstream first_iss(line);
    std::vector<double> values;
    double val;
    while (first_iss >> val) {
        values.push_back(val);
    }
    
    bool has_index = (values.size() >= 4);
    if (has_index) {
        // 4+ columns: skip first (site index), use next 3
        spins[0] = Eigen::Vector3d(values[1], values[2], values[3]);
    } else if (values.size() >= 3) {
        // 3 columns: Sx Sy Sz directly
        spins[0] = Eigen::Vector3d(values[0], values[1], values[2]);
    }
    
    // Read remaining spins
    for (size_t i = 1; i < lattice_size; ++i) {
        if (has_index) {
            int idx;
            file >> idx >> spins[i](0) >> spins[i](1) >> spins[i](2);
        } else {
            file >> spins[i](0) >> spins[i](1) >> spins[i](2);
        }
    }
}

void StrainPhononLattice::save_strain_state(const string& filename) const {
    ofstream file(filename);
    file << std::scientific << std::setprecision(12);
    
    file << "# Strain state (per bond type)\n";
    file << "# bond_type epsilon_xx epsilon_yy epsilon_xy V_xx V_yy V_xy\n";
    
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        file << b << " "
             << strain.epsilon_xx[b] << " "
             << strain.epsilon_yy[b] << " "
             << strain.epsilon_xy[b] << " "
             << strain.V_xx[b] << " "
             << strain.V_yy[b] << " "
             << strain.V_xy[b] << "\n";
    }
}

void StrainPhononLattice::save_positions(const string& filename) const {
    ofstream file(filename);
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        file << site_positions[i](0) << " " 
             << site_positions[i](1) << " " 
             << site_positions[i](2) << "\n";
    }
}

void StrainPhononLattice::save_spin_config_global(const string& filename) const {
    ofstream file(filename);
    file << std::scientific << std::setprecision(12);
    
    // Transform each spin from local Kitaev frame to global Cartesian frame
    // S_global = R * S_local where R is the sublattice rotation matrix
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                for (size_t atom = 0; atom < N_atoms; ++atom) {
                    size_t site_idx = flatten_index(i, j, k, atom);
                    SpinVector spin_global = sublattice_frames[atom] * spins[site_idx];
                    file << spin_global(0) << " " << spin_global(1) << " " << spin_global(2) << "\n";
                }
            }
        }
    }
}

// ============================================================
// OBSERVABLES
// ============================================================

Eigen::Vector3d StrainPhononLattice::magnetization_local() const {
    Eigen::Vector3d M = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < lattice_size; ++i) {
        M += spins[i];
    }
    return M / lattice_size;
}

Eigen::Vector3d StrainPhononLattice::magnetization_local_antiferro() const {
    Eigen::Vector3d M_stag = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < lattice_size; ++i) {
        // Sublattice alternation: sites with atom index 0 are A, others are B
        double sign = (i % N_atoms == 0) ? 1.0 : -1.0;
        M_stag += sign * spins[i];
    }
    return M_stag / lattice_size;
}

double StrainPhononLattice::A1g_amplitude() const {
    // A1g mode: (ε_xx + ε_yy) averaged over bond types
    double A1g = 0.0;
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        A1g += strain.epsilon_xx[b] + strain.epsilon_yy[b];
    }
    return A1g / StrainState::N_BONDS;
}

double StrainPhononLattice::Eg1_amplitude() const {
    // Eg1 mode: (ε_xx - ε_yy) averaged over bond types
    double Eg1 = 0.0;
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        Eg1 += strain.epsilon_xx[b] - strain.epsilon_yy[b];
    }
    return Eg1 / StrainState::N_BONDS;
}

double StrainPhononLattice::Eg2_amplitude() const {
    // Eg2 mode: 2*ε_xy averaged over bond types
    double Eg2 = 0.0;
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        Eg2 += 2.0 * strain.epsilon_xy[b];
    }
    return Eg2 / StrainState::N_BONDS;
}

double StrainPhononLattice::order_parameter() const {
    if (!has_ordering_pattern || ordering_pattern.size() != lattice_size) {
        return magnetization_local().norm();
    }
    
    double op = 0.0;
    for (size_t i = 0; i < lattice_size; ++i) {
        op += spins[i].dot(ordering_pattern[i]);
    }
    return std::abs(op) / lattice_size;
}

void StrainPhononLattice::save_trajectory_txt(const string& output_dir,
                                              const vector<double>& times,
                                              const vector<Eigen::Vector3d>& M_traj,
                                              const vector<Eigen::Vector3d>& M_stag_traj,
                                              const vector<double>& energy_traj,
                                              const vector<double>& eps_A1g_traj,
                                              const vector<double>& eps_Eg1_traj,
                                              const vector<double>& eps_Eg2_traj,
                                              const vector<double>& J7_eff_traj) const {
    // Save magnetization trajectory
    ofstream mag_file(output_dir + "/magnetization.txt");
    mag_file << "# t Mx My Mz |M| Mx_stag My_stag Mz_stag |M_stag|\n";
    mag_file << std::scientific << std::setprecision(12);
    for (size_t i = 0; i < times.size(); ++i) {
        mag_file << times[i] << " "
                 << M_traj[i](0) << " " << M_traj[i](1) << " " << M_traj[i](2) << " "
                 << M_traj[i].norm() << " "
                 << M_stag_traj[i](0) << " " << M_stag_traj[i](1) << " " << M_stag_traj[i](2) << " "
                 << M_stag_traj[i].norm() << "\n";
    }
    mag_file.close();
    
    // Save energy trajectory
    ofstream energy_file(output_dir + "/energy.txt");
    energy_file << "# t E\n";
    energy_file << std::scientific << std::setprecision(12);
    for (size_t i = 0; i < times.size(); ++i) {
        energy_file << times[i] << " " << energy_traj[i] << "\n";
    }
    energy_file.close();
    
    // Save strain trajectory
    ofstream strain_file(output_dir + "/strain_trajectory.txt");
    strain_file << "# t eps_A1g eps_Eg1 eps_Eg2 |eps_Eg| J7_eff\n";
    strain_file << std::scientific << std::setprecision(12);
    for (size_t i = 0; i < times.size(); ++i) {
        double Eg_amp = std::sqrt(eps_Eg1_traj[i]*eps_Eg1_traj[i] + eps_Eg2_traj[i]*eps_Eg2_traj[i]);
        strain_file << times[i] << " "
                    << eps_A1g_traj[i] << " "
                    << eps_Eg1_traj[i] << " "
                    << eps_Eg2_traj[i] << " "
                    << Eg_amp << " "
                    << J7_eff_traj[i] << "\n";
    }
    strain_file.close();
    
    // Save per-bond-type strain data
    ofstream strain_bonds_file(output_dir + "/strain_per_bond.txt");
    strain_bonds_file << "# Current strain state per bond type\n";
    strain_bonds_file << "# bond eps_xx eps_yy eps_xy V_xx V_yy V_xy\n";
    strain_bonds_file << std::scientific << std::setprecision(12);
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        strain_bonds_file << b << " "
                          << strain.epsilon_xx[b] << " "
                          << strain.epsilon_yy[b] << " "
                          << strain.epsilon_xy[b] << " "
                          << strain.V_xx[b] << " "
                          << strain.V_yy[b] << " "
                          << strain.V_xy[b] << "\n";
    }
    strain_bonds_file.close();
    
    cout << "Trajectory saved to " << output_dir << endl;
    cout << "  magnetization.txt: " << times.size() << " timesteps" << endl;
    cout << "  energy.txt: " << times.size() << " timesteps" << endl;
    cout << "  strain_trajectory.txt: " << times.size() << " timesteps" << endl;
}

// ============================================================
// MAGNETIZATION HELPERS
// ============================================================

SpinVector StrainPhononLattice::magnetization_global() const {
    SpinVector M = SpinVector::Zero(spin_dim);
    for (size_t i = 0; i < lattice_size; ++i) {
        size_t atom = i % N_atoms;
        M += sublattice_frames[atom] * spins[i];
    }
    return M / static_cast<double>(lattice_size);
}

vector<SpinVector> StrainPhononLattice::magnetization_sublattice() const {
    vector<SpinVector> M(N_atoms, SpinVector::Zero(spin_dim));
    vector<size_t> counts(N_atoms, 0);
    for (size_t i = 0; i < lattice_size; ++i) {
        size_t atom = i % N_atoms;
        M[atom] += sublattice_frames[atom] * spins[i];
        counts[atom]++;
    }
    for (size_t a = 0; a < N_atoms; ++a) {
        if (counts[a] > 0) M[a] /= static_cast<double>(counts[a]);
    }
    return M;
}

// ============================================================
// MC algorithms (parallel_tempering, attempt_replica_exchange, binning_analysis,
// estimate_autocorrelation_time, compute_thermodynamic_observables,
// generate_optimized_temperature_grid_mpi, generate_geometric_temperature_ladder,
// gather_and_save_statistics_comprehensive, save_thermodynamic_observables_hdf5,
// save_heat_capacity_hdf5) are now provided by mc::* template functions
// in mc_common.h and called via inline wrappers in strain_phonon_lattice.h.
// ============================================================


// ============================================================
// GNEB AND TRANSITION PATH ANALYSIS
// ============================================================

double StrainPhononLattice::energy_for_gneb(const vector<Eigen::Vector3d>& config) const {
    // Temporarily set spins and compute energy
    // This is a bit ugly but necessary for the GNEB interface
    auto* self = const_cast<StrainPhononLattice*>(this);
    SpinConfig original_spins = self->spins;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        self->spins[i] = config[i];
    }
    
    double E = spin_energy();
    
    // Restore original spins
    self->spins = original_spins;
    
    return E;
}

vector<Eigen::Vector3d> StrainPhononLattice::gradient_for_gneb(
    const vector<Eigen::Vector3d>& config) const {
    // Temporarily set spins to compute gradient
    auto* self = const_cast<StrainPhononLattice*>(this);
    SpinConfig original_spins = self->spins;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        self->spins[i] = config[i];
    }
    
    // Compute gradient: ∂E/∂S_i = -H_eff_i (effective field is -gradient)
    vector<Eigen::Vector3d> grad(lattice_size);
    for (size_t i = 0; i < lattice_size; ++i) {
        Eigen::Vector3d H_eff = get_local_field(i);
        grad[i] = -H_eff;  // gradient is negative of effective field
    }
    
    // Restore original spins
    self->spins = original_spins;
    
    return grad;
}

vector<Eigen::Vector3d> StrainPhononLattice::compute_Eg_phonon_force(
    const vector<Eigen::Vector3d>& config, double Q_Eg) const {
    // Compute the force from Eg phonon coupling:
    // F_i = +g Q_Eg · ∂f_Eg/∂S_i
    //
    // The key insight: even if f_Eg = 0 at the triple-Q state (A1g symmetry),
    // ∂f_Eg/∂S_i ≠ 0 generically, so the drive produces a linear-in-Q
    // restoring force that pushes along the E1 symmetry direction.
    
    // Temporarily set configuration
    auto* self = const_cast<StrainPhononLattice*>(this);
    SpinConfig original_spins = self->spins;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        self->spins[i] = config[i];
    }
    
    // Get the Eg derivatives
    auto [df_Eg1, df_Eg2] = compute_Eg_derivatives(config);
    
    // Coupling strength (combines K, J, Gamma contributions)
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    // The Eg phonon couples via:
    // H_sp-ph = -λ_Eg * [(ε_xx - ε_yy) * f_Eg1 + 2ε_xy * f_Eg2]
    //
    // For a driven phonon Q_Eg in the Eg1 channel:
    // F_i = λ_Eg * Q_Eg * ∂f_Eg1/∂S_i
    
    vector<Eigen::Vector3d> force(lattice_size);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        // Force from Eg1 channel (assuming drive is in Eg1 direction)
        force[i] = lambda_Eg * Q_Eg * df_Eg1[i];
        
        // Could add Eg2 component if needed:
        // force[i] += lambda_Eg * Q_Eg2 * df_Eg2[i];
    }
    
    // Restore original spins
    self->spins = original_spins;
    
    return force;
}

std::pair<vector<Eigen::Vector3d>, vector<Eigen::Vector3d>> 
StrainPhononLattice::compute_Eg_derivatives(const vector<Eigen::Vector3d>& config) const {
    // Compute ∂f_Eg1/∂S_i and ∂f_Eg2/∂S_i for each site
    // These are the directions in spin space that the Eg phonon "pushes"
    
    // Temporarily set configuration (already done in caller, but be safe)
    auto* self = const_cast<StrainPhononLattice*>(this);
    SpinConfig original_spins = self->spins;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        self->spins[i] = config[i];
    }
    
    vector<Eigen::Vector3d> df_Eg1(lattice_size);
    vector<Eigen::Vector3d> df_Eg2(lattice_size);
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double Gammap = magnetoelastic_params.Gammap;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        // Total Eg1 derivative combines K, J, Γ, and Γ' contributions
        // f_Eg1 = (J+K) f_K_Eg1 + J f_J_Eg1 + Γ f_Γ_Eg1 + Γ' f_Γ'_Eg1
        // So ∂f_Eg1/∂S = (J+K) ∂f_K_Eg1/∂S + J ∂f_J_Eg1/∂S + Γ ∂f_Γ_Eg1/∂S + Γ' ∂f_Γ'_Eg1/∂S
        
        SpinVector dfK_Eg1 = df_K_Eg1_dS(i);
        SpinVector dfJ_Eg1 = df_J_Eg1_dS(i);
        SpinVector dfG_Eg1 = df_Gamma_Eg1_dS(i);
        SpinVector dfGp_Eg1 = df_Gammap_Eg1_dS(i);
        
        df_Eg1[i] = (J + K) * dfK_Eg1 + J * dfJ_Eg1 + Gamma * dfG_Eg1 + Gammap * dfGp_Eg1;
        
        // Same for Eg2
        SpinVector dfK_Eg2 = df_K_Eg2_dS(i);
        SpinVector dfJ_Eg2 = df_J_Eg2_dS(i);
        SpinVector dfG_Eg2 = df_Gamma_Eg2_dS(i);
        SpinVector dfGp_Eg2 = df_Gammap_Eg2_dS(i);
        
        df_Eg2[i] = (J + K) * dfK_Eg2 + J * dfJ_Eg2 + Gamma * dfG_Eg2 + Gammap * dfGp_Eg2;
    }
    
    // Restore original spins
    self->spins = original_spins;
    
    return {df_Eg1, df_Eg2};
}

vector<Eigen::Vector3d> StrainPhononLattice::get_spin_config() const {
    vector<Eigen::Vector3d> config(lattice_size);
    for (size_t i = 0; i < lattice_size; ++i) {
        config[i] = spins[i];
    }
    return config;
}

void StrainPhononLattice::set_spin_config(const vector<Eigen::Vector3d>& config) {
    if (config.size() != lattice_size) {
        throw std::runtime_error("Config size mismatch in set_spin_config");
    }
    for (size_t i = 0; i < lattice_size; ++i) {
        spins[i] = config[i];
        spins[i].normalize();
    }
}

// ============================================================
// GNEB WITH STRAIN: Combined spin + strain configuration space
// ============================================================

double StrainPhononLattice::energy_for_gneb_with_strain(
    const vector<Eigen::Vector3d>& spins_in,
    double strain_Eg1, double strain_Eg2) const {
    
    // Save current state
    auto* self = const_cast<StrainPhononLattice*>(this);
    SpinConfig original_spins = self->spins;
    StrainState original_strain = self->strain;
    
    // Set new configuration
    for (size_t i = 0; i < lattice_size; ++i) {
        self->spins[i] = spins_in[i];
    }
    
    // Apply uniform Eg strain to all bond types
    // Eg1 = (ε_xx - ε_yy)/2, Eg2 = ε_xy
    // So: ε_xx = ε_Eg1, ε_yy = -ε_Eg1, ε_xy = ε_Eg2
    for (size_t b = 0; b < 3; ++b) {
        self->strain.epsilon_xx[b] = strain_Eg1;
        self->strain.epsilon_yy[b] = -strain_Eg1;
        self->strain.epsilon_xy[b] = strain_Eg2;
    }
    
    // Compute total energy (spin + elastic + magnetoelastic)
    double E = total_energy();
    
    // Restore original state
    self->spins = original_spins;
    self->strain = original_strain;
    
    return E;
}

std::tuple<vector<Eigen::Vector3d>, double, double>
StrainPhononLattice::gradient_for_gneb_with_strain(
    const vector<Eigen::Vector3d>& spins_in,
    double strain_Eg1, double strain_Eg2) const {
    
    // Save current state
    auto* self = const_cast<StrainPhononLattice*>(this);
    SpinConfig original_spins = self->spins;
    StrainState original_strain = self->strain;
    
    // Set new configuration
    for (size_t i = 0; i < lattice_size; ++i) {
        self->spins[i] = spins_in[i];
    }
    
    // Apply strain
    for (size_t b = 0; b < 3; ++b) {
        self->strain.epsilon_xx[b] = strain_Eg1;
        self->strain.epsilon_yy[b] = -strain_Eg1;
        self->strain.epsilon_xy[b] = strain_Eg2;
    }
    
    // Compute spin gradients: ∂E/∂S_i = -H_eff_i
    vector<Eigen::Vector3d> grad_spins(lattice_size);
    for (size_t i = 0; i < lattice_size; ++i) {
        Eigen::Vector3d H_eff = get_local_field(i);
        grad_spins[i] = -H_eff;
    }
    
    // Compute strain gradients using finite differences
    // (more robust than analytic for complex magnetoelastic coupling)
    const double delta = 1e-5;
    
    // ∂E/∂ε_Eg1
    double E_plus_Eg1 = energy_for_gneb_with_strain(spins_in, strain_Eg1 + delta, strain_Eg2);
    double E_minus_Eg1 = energy_for_gneb_with_strain(spins_in, strain_Eg1 - delta, strain_Eg2);
    double dE_dEg1 = (E_plus_Eg1 - E_minus_Eg1) / (2.0 * delta);
    
    // ∂E/∂ε_Eg2
    double E_plus_Eg2 = energy_for_gneb_with_strain(spins_in, strain_Eg1, strain_Eg2 + delta);
    double E_minus_Eg2 = energy_for_gneb_with_strain(spins_in, strain_Eg1, strain_Eg2 - delta);
    double dE_dEg2 = (E_plus_Eg2 - E_minus_Eg2) / (2.0 * delta);
    
    // Restore original state
    self->spins = original_spins;
    self->strain = original_strain;
    
    return {grad_spins, dE_dEg1, dE_dEg2};
}

std::pair<double, double> StrainPhononLattice::relax_strain_at_fixed_spins(
    const vector<Eigen::Vector3d>& spins_in,
    size_t /*max_iter*/,
    double /*tolerance*/) const {
    
    // Analytic solution: energy is exactly quadratic in strain, so
    // ∂E/∂ε = 0 is a linear system with a closed-form solution.
    // This is the same formula as relax_strain(), but operates on
    // externally-provided spins rather than the lattice's internal state.
    
    // Temporarily load input spins to evaluate spin basis functions
    auto* self = const_cast<StrainPhononLattice*>(this);
    SpinConfig original_spins = self->spins;
    for (size_t i = 0; i < lattice_size; ++i) {
        self->spins[i] = spins_in[i];
    }
    
    double C11 = elastic_params.C11;
    double C12 = elastic_params.C12;
    double C44 = elastic_params.C44;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    double det = C11 * C11 - C12 * C12;
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double Gammap = magnetoelastic_params.Gammap;
    
    double Eg1_spin_factor = (J + K) * f_K_Eg1() + J * f_J_Eg1()
                           + Gamma * f_Gamma_Eg1() + Gammap * f_Gammap_Eg1();
    double Eg2_spin_factor = (J + K) * f_K_Eg2() + J * f_J_Eg2()
                           + Gamma * f_Gamma_Eg2() + Gammap * f_Gammap_Eg2();
    
    // Restore original spins
    self->spins = original_spins;
    
    // Analytic equilibrium: ε_xx = (F1/2 - λ_Eg Σ_Eg1)(C11+C12) / det
    //                       ε_xy = (F2 - 2λ_Eg Σ_Eg2) / (4C44)
    double Eg1 = (drive_F_Eg1_ / 2.0 - lambda_Eg * Eg1_spin_factor) * (C11 + C12) / det;
    double Eg2 = (drive_F_Eg2_ - 2.0 * lambda_Eg * Eg2_spin_factor) / (4.0 * C44);
    
    return {Eg1, Eg2};
}

std::pair<double, double> StrainPhononLattice::relax_strain_at_fixed_spins(
    const vector<Eigen::Vector3d>& spins_in,
    double /*init_Eg1*/, double /*init_Eg2*/,
    size_t /*max_iter*/,
    double /*tolerance*/) const {
    
    // Analytic solution — initial guess is irrelevant since the energy is
    // exactly quadratic in strain. The closed-form minimum is exact.
    return relax_strain_at_fixed_spins(spins_in);
}

// ============================================================
// GNEB WITH EXTERNAL STRAIN: Fixed external + relaxable internal strain
// ============================================================

double StrainPhononLattice::energy_for_gneb_with_external_strain(
    const vector<Eigen::Vector3d>& spins_in,
    double internal_Eg1, double internal_Eg2,
    double external_Eg1, double external_Eg2) const {
    
    // Total strain = external + internal
    double total_Eg1 = external_Eg1 + internal_Eg1;
    double total_Eg2 = external_Eg2 + internal_Eg2;
    
    return energy_for_gneb_with_strain(spins_in, total_Eg1, total_Eg2);
}

std::tuple<vector<Eigen::Vector3d>, double, double>
StrainPhononLattice::gradient_for_gneb_with_external_strain(
    const vector<Eigen::Vector3d>& spins_in,
    double internal_Eg1, double internal_Eg2,
    double external_Eg1, double external_Eg2) const {
    
    // Total strain = external + internal
    double total_Eg1 = external_Eg1 + internal_Eg1;
    double total_Eg2 = external_Eg2 + internal_Eg2;
    
    // Gradients w.r.t. total strain are the same as w.r.t. internal strain
    // since ∂E/∂ε_int = ∂E/∂ε_total * ∂ε_total/∂ε_int = ∂E/∂ε_total
    return gradient_for_gneb_with_strain(spins_in, total_Eg1, total_Eg2);
}

std::pair<double, double> StrainPhononLattice::relax_strain_with_external(
    const vector<Eigen::Vector3d>& spins_in,
    double external_Eg1, double external_Eg2,
    size_t /*max_iter*/,
    double /*tolerance*/) const {
    
    // Analytic solution: total equilibrium strain is determined by spins + drive.
    // Internal strain = total equilibrium - external.
    auto [total_Eg1, total_Eg2] = relax_strain_at_fixed_spins(spins_in);
    
    return {total_Eg1 - external_Eg1, total_Eg2 - external_Eg2};
}

void StrainPhononLattice::init_zigzag_pattern(int direction) {
    // Zigzag pattern: FM chains along one direction, AFM perpendicular
    // The "direction" specifies which bond type has AFM coupling
    
    // For honeycomb, zigzag has ordering wavevector at M point
    // Depending on direction, the pattern differs
    
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                size_t siteA = flatten_index(i, j, k, 0);
                size_t siteB = flatten_index(i, j, k, 1);
                
                int sign;
                switch (direction) {
                    case 0:  // x-bond zigzag
                        sign = ((i + j) % 2 == 0) ? 1 : -1;
                        break;
                    case 1:  // y-bond zigzag
                        sign = (i % 2 == 0) ? 1 : -1;
                        break;
                    case 2:  // z-bond zigzag
                    default:
                        sign = (j % 2 == 0) ? 1 : -1;
                        break;
                }
                
                // Spins point along z (can be generalized)
                spins[siteA] = sign * Eigen::Vector3d(0, 0, 1);
                spins[siteB] = sign * Eigen::Vector3d(0, 0, 1);  // Same sign within unit cell
            }
        }
    }
    
    // Normalize (already done but be safe)
    for (size_t i = 0; i < lattice_size; ++i) {
        spins[i].normalize();
    }
    
    cout << "Initialized zigzag pattern (direction = " << direction << ")" << endl;
}

void StrainPhononLattice::init_triple_q() {
    // Triple-Q pattern: superposition of three M-point ordering vectors
    // This is a 120° coplanar or non-coplanar structure depending on parameters
    
    // For simplicity, we initialize with a 120° structure
    // Real triple-Q would require solving for the ground state
    
    // Triple-Q wavevectors at M points of honeycomb BZ:
    // M1 = (π, π/√3), M2 = (0, 2π/√3), M3 = (-π, π/√3)
    
    Eigen::Vector3d Q1(M_PI, M_PI / std::sqrt(3.0), 0);
    Eigen::Vector3d Q2(0, 2.0 * M_PI / std::sqrt(3.0), 0);
    Eigen::Vector3d Q3(-M_PI, M_PI / std::sqrt(3.0), 0);
    
    // Three ordering directions (120° apart in spin space)
    Eigen::Vector3d n1(1, 0, 0);
    Eigen::Vector3d n2(-0.5, std::sqrt(3.0)/2.0, 0);
    Eigen::Vector3d n3(-0.5, -std::sqrt(3.0)/2.0, 0);
    
    for (size_t idx = 0; idx < lattice_size; ++idx) {
        Eigen::Vector3d r = site_positions[idx];
        
        // Superposition of three M-point waves
        double phase1 = Q1.dot(r);
        double phase2 = Q2.dot(r);
        double phase3 = Q3.dot(r);
        
        Eigen::Vector3d S = std::cos(phase1) * n1 
                         + std::cos(phase2) * n2 
                         + std::cos(phase3) * n3;
        
        if (S.norm() > 1e-10) {
            spins[idx] = S.normalized();
        } else {
            spins[idx] = Eigen::Vector3d(0, 0, 1);
        }
    }
    
    cout << "Initialized triple-Q pattern (3M superposition)" << endl;
}

double StrainPhononLattice::structure_factor(const Eigen::Vector3d& q) const {
    // S(q) = |Σ_i S_i exp(-i q·r_i)|² / N
    std::complex<double> Sq_x(0, 0), Sq_y(0, 0), Sq_z(0, 0);
    
    for (size_t i = 0; i < lattice_size; ++i) {
        double phase = q.dot(site_positions[i]);
        std::complex<double> exp_factor(std::cos(phase), -std::sin(phase));
        
        Sq_x += spins[i](0) * exp_factor;
        Sq_y += spins[i](1) * exp_factor;
        Sq_z += spins[i](2) * exp_factor;
    }
    
    double S_total = (std::norm(Sq_x) + std::norm(Sq_y) + std::norm(Sq_z)) / lattice_size;
    
    return S_total;
}

StrainPhononLattice::CollectiveVars StrainPhononLattice::compute_collective_variables() const {
    CollectiveVars cv;
    
    // Triple-Q order: sum of structure factors at three M points
    Eigen::Vector3d M1(M_PI, M_PI / std::sqrt(3.0), 0);
    Eigen::Vector3d M2(0, 2.0 * M_PI / std::sqrt(3.0), 0);
    Eigen::Vector3d M3(-M_PI, M_PI / std::sqrt(3.0), 0);
    
    cv.m_3Q = structure_factor(M1) + structure_factor(M2) + structure_factor(M3);
    cv.m_3Q /= 3.0;  // Average
    
    // Zigzag order: structure factor at the zigzag wavevector
    // For honeycomb, zigzag has q at the M point (same as triple-Q components)
    // But zigzag picks one dominant M point, while triple-Q has equal weight at all three
    // We use the max of the three
    cv.m_zigzag = std::max({structure_factor(M1), structure_factor(M2), structure_factor(M3)});
    
    // Eg symmetry breaking: |f_Eg| = sqrt(f_Eg1² + f_Eg2²)
    // where f_Eg = (J+K)f_K_Eg + J f_J_Eg + Γ f_Γ_Eg + Γ' f_Γ'_Eg
    // This must match magnetoelastic_energy() formula exactly!
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double Gammap = magnetoelastic_params.Gammap;
    
    double fEg1 = (J + K) * f_K_Eg1() + J * f_J_Eg1() + Gamma * f_Gamma_Eg1() + Gammap * f_Gammap_Eg1();
    double fEg2 = (J + K) * f_K_Eg2() + J * f_J_Eg2() + Gamma * f_Gamma_Eg2() + Gammap * f_Gammap_Eg2();
    
    cv.f_Eg_amplitude = std::sqrt(fEg1 * fEg1 + fEg2 * fEg2);
    
    // Total energy
    cv.E_total = spin_energy();
    
    return cv;
}

// ============================================================
// STATIC STRUCTURE FACTOR
// ============================================================

StrainPhononLattice::StructureFactorResult StrainPhononLattice::compute_static_structure_factor(
    size_t n_q1, size_t n_q2,
    double q1_min, double q1_max,
    double q2_min, double q2_max) const {
    
    StructureFactorResult result;
    
    // Reciprocal lattice vectors for honeycomb lattice
    // Real space: a1 = (1, 0, 0), a2 = (1/2, sqrt(3)/2, 0)
    // Reciprocal: b1* = 2π(1, -1/sqrt(3), 0), b2* = 2π(0, 2/sqrt(3), 0)
    const double sqrt3 = std::sqrt(3.0);
    Eigen::Vector3d b1(2.0 * M_PI, -2.0 * M_PI / sqrt3, 0.0);
    Eigen::Vector3d b2(0.0, 4.0 * M_PI / sqrt3, 0.0);
    
    // Generate q-grid
    result.q1_vals.resize(n_q1);
    result.q2_vals.resize(n_q2);
    
    for (size_t i = 0; i < n_q1; ++i) {
        result.q1_vals[i] = q1_min + (q1_max - q1_min) * double(i) / double(n_q1 - 1);
    }
    for (size_t j = 0; j < n_q2; ++j) {
        result.q2_vals[j] = q2_min + (q2_max - q2_min) * double(j) / double(n_q2 - 1);
    }
    
    // Initialize S(q) arrays
    result.S_total.resize(n_q1, vector<double>(n_q2, 0.0));
    result.S_xx.resize(n_q1, vector<double>(n_q2, 0.0));
    result.S_yy.resize(n_q1, vector<double>(n_q2, 0.0));
    result.S_zz.resize(n_q1, vector<double>(n_q2, 0.0));
    
    // Compute S(q) for each q-point
    for (size_t i = 0; i < n_q1; ++i) {
        for (size_t j = 0; j < n_q2; ++j) {
            // q = q1 * b1* + q2 * b2*
            Eigen::Vector3d q = result.q1_vals[i] * b1 + result.q2_vals[j] * b2;
            
            // Compute Fourier transform: Σ_k S_k exp(-i q·r_k)
            std::complex<double> Sq_x(0, 0), Sq_y(0, 0), Sq_z(0, 0);
            
            for (size_t k = 0; k < lattice_size; ++k) {
                double phase = q.dot(site_positions[k]);
                std::complex<double> exp_factor(std::cos(phase), -std::sin(phase));
                
                Sq_x += spins[k](0) * exp_factor;
                Sq_y += spins[k](1) * exp_factor;
                Sq_z += spins[k](2) * exp_factor;
            }
            
            // Structure factor components
            result.S_xx[i][j] = std::norm(Sq_x) / lattice_size;
            result.S_yy[i][j] = std::norm(Sq_y) / lattice_size;
            result.S_zz[i][j] = std::norm(Sq_z) / lattice_size;
            result.S_total[i][j] = result.S_xx[i][j] + result.S_yy[i][j] + result.S_zz[i][j];
        }
    }
    
    return result;
}

void StrainPhononLattice::save_structure_factor(const string& filename, 
                                                 const StructureFactorResult& sf) const {
    ofstream file(filename);
    file << std::scientific << std::setprecision(12);
    
    // Header
    file << "# Static spin structure factor S(q)\n";
    file << "# Columns: q1 q2 S_total S_xx S_yy S_zz\n";
    file << "# q1, q2 are in units of reciprocal lattice vectors (b1*, b2*)\n";
    file << "# Grid: " << sf.q1_vals.size() << " x " << sf.q2_vals.size() << "\n";
    
    // Data
    for (size_t i = 0; i < sf.q1_vals.size(); ++i) {
        for (size_t j = 0; j < sf.q2_vals.size(); ++j) {
            file << sf.q1_vals[i] << " " << sf.q2_vals[j] << " "
                 << sf.S_total[i][j] << " "
                 << sf.S_xx[i][j] << " "
                 << sf.S_yy[i][j] << " "
                 << sf.S_zz[i][j] << "\n";
        }
        file << "\n";  // Blank line between q1 slices for gnuplot
    }
}