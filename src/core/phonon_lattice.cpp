/**
 * @file phonon_lattice.cpp
 * @brief Spin–phonon coupled honeycomb lattice (NCTO E1 magnetoelastic model).
 *
 * Implements the model described in docs/tmfeo3_notes.tex:
 *   - Standard J–K–Γ–Γ' nearest-neighbour spin Hamiltonian (LOCAL Kitaev frame
 *     rotated to the GLOBAL Cartesian frame for spin storage), plus optional
 *     sublattice-dependent J2, isotropic J3, six-spin ring exchange J7, and
 *     uniform Zeeman field.
 *   - Single zone-center two-component E1 optical strain coordinate
 *         ε = (Q_x, Q_y),  with conjugate velocities (V_x, V_y).
 *   - Quadratic-in-ε E1 magnetoelastic coupling
 *         δX_γ(ε) = λ_{X,0}(ε_x²+ε_y²)
 *                 + λ_{X,2}[(ε_x²-ε_y²) cos(2θ_γ)
 *                           + 2 ε_x ε_y sin(2θ_γ)]
 *     for X ∈ {J, K, Γ, Γ'} on bonds γ ∈ {x, y, z} with bond-axis angles
 *     θ_x = 0, θ_y = 2π/3, θ_z = 4π/3.
 *   - Single global polar THz drive  H_drive = -Z*[E_x(t) Q_x + E_y(t) Q_y].
 *
 * The current implementation strictly follows the leading-order C6 invariant
 * spin-bond-dependent magnetoelastic Hamiltonian. Linear-in-ε exchange-striction
 * is symmetry-forbidden once the bond bilinears are summed over the three bond
 * types, so it is intentionally absent here.
 */

#include "classical_spin/lattice/phonon_lattice.h"
#include "classical_spin/lattice/pulse_chunking.h"
#include <fstream>
#include <iomanip>
#include <memory>
#include <filesystem>
#include <random>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

namespace odeint = boost::numeric::odeint;

namespace {

constexpr double SQRT3 = 1.7320508075688772935;

/// Quadratic E1 modulation of one exchange channel X on a single bond:
///   δX_γ(ε) = scale · [λ0 (ε_x²+ε_y²) + λ2 ((ε_x²-ε_y²) cos2θ + 2 ε_xε_y sin2θ)]
struct E1ExchangeCoefficients {
    double J = 0.0;
    double K = 0.0;
    double Gamma = 0.0;
    double Gammap = 0.0;
};

/// (cos 2θ_γ, sin 2θ_γ) for the three bond axes, with θ_x=0, θ_y=2π/3, θ_z=4π/3.
std::pair<double, double> e1_bond_form_factor(int bond_type) {
    switch (bond_type) {
        case 0: return {1.0, 0.0};                  // x: 2θ = 0
        case 1: return {-0.5, -0.5 * SQRT3};        // y: 2θ = 4π/3
        default: return {-0.5, 0.5 * SQRT3};        // z: 2θ = 8π/3 ≡ 2π/3
    }
}

/// δX_γ(ε) for one (lambda0, lambda2) channel on bond γ.
double e1_delta(double lambda0, double lambda2, double qx, double qy,
                int bond_type, double scale = 1.0) {
    const auto [c2, s2] = e1_bond_form_factor(bond_type);
    const double q0 = qx * qx + qy * qy;
    const double qc = qx * qx - qy * qy;
    const double qs = 2.0 * qx * qy;
    return scale * (lambda0 * q0 + lambda2 * (qc * c2 + qs * s2));
}

/// ∂δX_γ/∂ε_x.
double e1_delta_dqx(double lambda0, double lambda2, double qx, double qy,
                    int bond_type, double scale = 1.0) {
    const auto [c2, s2] = e1_bond_form_factor(bond_type);
    return scale * (2.0 * lambda0 * qx + 2.0 * lambda2 * (qx * c2 + qy * s2));
}

/// ∂δX_γ/∂ε_y.
double e1_delta_dqy(double lambda0, double lambda2, double qx, double qy,
                    int bond_type, double scale = 1.0) {
    const auto [c2, s2] = e1_bond_form_factor(bond_type);
    return scale * (2.0 * lambda0 * qy + 2.0 * lambda2 * (-qy * c2 + qx * s2));
}

E1ExchangeCoefficients e1_exchange_coefficients(const SpinPhononCouplingParams& params,
                                                double qx, double qy,
                                                int bond_type, double scale = 1.0) {
    return {
        e1_delta(params.lambda_E1_J_0,      params.lambda_E1_J_2,      qx, qy, bond_type, scale),
        e1_delta(params.lambda_E1_K_0,      params.lambda_E1_K_2,      qx, qy, bond_type, scale),
        e1_delta(params.lambda_E1_Gamma_0,  params.lambda_E1_Gamma_2,  qx, qy, bond_type, scale),
        e1_delta(params.lambda_E1_Gammap_0, params.lambda_E1_Gammap_2, qx, qy, bond_type, scale),
    };
}

E1ExchangeCoefficients e1_exchange_dqx(const SpinPhononCouplingParams& params,
                                       double qx, double qy,
                                       int bond_type, double scale = 1.0) {
    return {
        e1_delta_dqx(params.lambda_E1_J_0,      params.lambda_E1_J_2,      qx, qy, bond_type, scale),
        e1_delta_dqx(params.lambda_E1_K_0,      params.lambda_E1_K_2,      qx, qy, bond_type, scale),
        e1_delta_dqx(params.lambda_E1_Gamma_0,  params.lambda_E1_Gamma_2,  qx, qy, bond_type, scale),
        e1_delta_dqx(params.lambda_E1_Gammap_0, params.lambda_E1_Gammap_2, qx, qy, bond_type, scale),
    };
}

E1ExchangeCoefficients e1_exchange_dqy(const SpinPhononCouplingParams& params,
                                       double qx, double qy,
                                       int bond_type, double scale = 1.0) {
    return {
        e1_delta_dqy(params.lambda_E1_J_0,      params.lambda_E1_J_2,      qx, qy, bond_type, scale),
        e1_delta_dqy(params.lambda_E1_K_0,      params.lambda_E1_K_2,      qx, qy, bond_type, scale),
        e1_delta_dqy(params.lambda_E1_Gamma_0,  params.lambda_E1_Gamma_2,  qx, qy, bond_type, scale),
        e1_delta_dqy(params.lambda_E1_Gammap_0, params.lambda_E1_Gammap_2, qx, qy, bond_type, scale),
    };
}

/// Build the bond-γ exchange increment (in the local Kitaev frame) from
/// channel-resolved coefficients. Mirrors the static form of J^{(γ)} in
/// kitaev_bonds.h but with possibly modulated coefficients.
Eigen::Matrix3d e1_exchange_matrix_local(const E1ExchangeCoefficients& coeffs,
                                         int bond_type) {
    Eigen::Matrix3d M = coeffs.J * Eigen::Matrix3d::Identity();
    const int gamma = bond_type;
    const int alpha = (gamma == 0) ? 1 : 0;
    const int beta  = 3 - gamma - alpha;

    M(gamma, gamma) += coeffs.K;
    M(alpha, beta)  += coeffs.Gamma;
    M(beta, alpha)  += coeffs.Gamma;
    M(gamma, alpha) += coeffs.Gammap;
    M(alpha, gamma) += coeffs.Gammap;
    M(gamma, beta)  += coeffs.Gammap;
    M(beta, gamma)  += coeffs.Gammap;
    return M;
}

/// One-bond E1 magnetoelastic energy contribution
///   δH_γ = Σ_X δX_γ(ε) O_{ij,γ}^{(X)} = (R^T S_i)·M·(R^T S_j),
/// where the spins are passed in the global frame.
double e1_energy_local(const Eigen::Vector3d& Si_global, const Eigen::Vector3d& Sj_global,
                       const SpinPhononCouplingParams& params,
                       double qx, double qy, int bond_type, double scale = 1.0) {
    const Eigen::Matrix3d R = SpinPhononCouplingParams::get_kitaev_rotation();
    const Eigen::Vector3d Si = R.transpose() * Si_global;
    const Eigen::Vector3d Sj = R.transpose() * Sj_global;
    const Eigen::Matrix3d M = e1_exchange_matrix_local(
        e1_exchange_coefficients(params, qx, qy, bond_type, scale), bond_type);
    return Si.dot(M * Sj);
}

/// One-bond contribution to the GLOBAL-frame effective field on site i,
/// H_eff(i) -= ∂(δH_γ)/∂S_i = R · (M · R^T S_j) (sign flip already included).
Eigen::Vector3d e1_field_global(const Eigen::Vector3d& Sj_global,
                                const SpinPhononCouplingParams& params,
                                double qx, double qy, int bond_type, double scale = 1.0) {
    const Eigen::Matrix3d R = SpinPhononCouplingParams::get_kitaev_rotation();
    const Eigen::Vector3d Sj = R.transpose() * Sj_global;
    const Eigen::Matrix3d M = e1_exchange_matrix_local(
        e1_exchange_coefficients(params, qx, qy, bond_type, scale), bond_type);
    return -R * (M * Sj);
}

/// One-bond contribution to ∂(δH_γ)/∂ε_a, given precomputed channel
/// derivatives @c deriv_coeffs (one of the e1_exchange_dq{x,y} returns).
double e1_dH_dQ_local(const Eigen::Vector3d& Si_global, const Eigen::Vector3d& Sj_global,
                      const E1ExchangeCoefficients& deriv_coeffs, int bond_type) {
    const Eigen::Matrix3d R = SpinPhononCouplingParams::get_kitaev_rotation();
    const Eigen::Vector3d Si = R.transpose() * Si_global;
    const Eigen::Vector3d Sj = R.transpose() * Sj_global;
    const Eigen::Matrix3d dM = e1_exchange_matrix_local(deriv_coeffs, bond_type);
    return Si.dot(dM * Sj);
}

}  // namespace

// ============================================================
// CONSTRUCTOR
// ============================================================

PhononLattice::PhononLattice(const UnitCell& uc, size_t d1, size_t d2, size_t d3, float spin_l)
    : unit_cell(uc), N_atoms(uc.N_atoms), dim1(d1), dim2(d2), dim3(d3), spin_length(spin_l)
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
    site_hexagons.resize(lattice_size);
    
    // Copy sublattice frames from UnitCell
    sublattice_frames.resize(N_atoms);
    for (size_t atom = 0; atom < N_atoms; ++atom) {
        sublattice_frames[atom] = uc.sublattice_frames[atom];
    }
    afm_sublattice_signs = uc.afm_sublattice_signs;
    
    // Initialize RNG
    auto seed_val = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(static_cast<unsigned int>(seed_val));
    seed_lehman(seed_val * 2 + 1);
    
    cout << "Initializing PhononLattice with dimensions: "
         << dim1 << " x " << dim2 << " x " << dim3 << endl;
    cout << "Atoms per unit cell: " << N_atoms << endl;
    cout << "Total spin sites: " << lattice_size << endl;
    cout << "Phonon DOF: " << PhononState::N_DOF << " (zone-center E1)" << endl;
    cout << "Total ODE state size: " << state_size << endl;
    
    // Build lattice site positions from UnitCell
    size_t site_idx = 0;
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                for (size_t atom = 0; atom < N_atoms; ++atom) {
                    Eigen::Vector3d pos = uc.lattice_pos[atom];
                    pos += double(i) * uc.lattice_vectors[0];
                    pos += double(j) * uc.lattice_vectors[1];
                    pos += double(k) * uc.lattice_vectors[2];
                    site_positions[site_idx] = pos;
                    
                    // Copy field from unit cell
                    field[site_idx] = uc.field[atom].head<3>();
                    
                    ++site_idx;
                }
            }
        }
    }
    
    // Build interaction topology from UnitCell
    // Iterate over all bilinear interactions and classify by bond_type
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                for (size_t atom = 0; atom < N_atoms; ++atom) {
                    size_t site = flatten_index(i, j, k, atom);
                    
                    auto range = uc.bilinear_interaction.equal_range(atom);
                    for (auto it = range.first; it != range.second; ++it) {
                        const auto& bi = it->second;
                        
                        // Compute partner site with periodic boundaries
                        size_t partner = flatten_index_periodic(
                            (int)i + bi.offset[0],
                            (int)j + bi.offset[1],
                            (int)k + bi.offset[2],
                            bi.partner);
                        
                        if (bi.bond_type >= 0) {
                            // NN interaction with bond_type info -> nn_interaction
                            nn_interaction[site].push_back(bi.interaction);
                            nn_partners[site].push_back(partner);
                            nn_bond_types[site].push_back(bi.bond_type);
                            
                            // Reverse bond
                            nn_interaction[partner].push_back(bi.interaction.transpose());
                            nn_partners[partner].push_back(site);
                            nn_bond_types[partner].push_back(bi.bond_type);
                        } else {
                            // J2/J3 interaction (no phonon coupling)
                            // Use j2 for same-sublattice, j3 for different sublattice
                            size_t partner_sub = bi.partner;
                            if (atom == partner_sub) {
                                // Same sublattice -> J2
                                // Only add if partner > site to avoid double counting
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
    
    cout << "PhononLattice initialization complete!" << endl;
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
    
    // Clear hexagons (interactions are already built from UnitCell in constructor)
    for (size_t i = 0; i < lattice_size; ++i) {
        site_hexagons[i].clear();
    }
    hexagons.clear();
    
    // Build hexagonal plaquettes for ring exchange
    // On honeycomb, each hexagon consists of alternating A and B sites
    // For each unit cell at (i,j,k), we define one hexagon with sites going counterclockwise:
    //   0: A(i,j,k)     - central A site
    //   1: B(i,j,k)     - z-bond neighbor (same unit cell)
    //   2: A(i,j+1,k)   - from B via y-bond to next A
    //   3: B(i-1,j+1,k) - z-bond from A(i,j+1,k)
    //   4: A(i-1,j,k)   - from B(i-1,j+1,k) via x-bond
    //   5: B(i-1,j,k)   - z-bond from A(i-1,j,k)
    if (std::abs(sp_params.J7) > 1e-12 || std::abs(sp_params.lambda_E1_J7_0) > 1e-12) {
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    std::array<size_t, 6> hex;
                    hex[0] = flatten_index(i, j, k, 0);                          // A(i,j,k)
                    hex[1] = flatten_index(i, j, k, 1);                          // B(i,j,k)
                    hex[2] = flatten_index_periodic(i, j+1, k, 0);               // A(i,j+1,k)
                    hex[3] = flatten_index_periodic(i+1, j, k, 1);             // B(i-1,j+1,k)
                    hex[4] = flatten_index_periodic(i+1, j, k, 0);               // A(i-1,j,k)
                    hex[5] = flatten_index_periodic(i+1, j-1, k, 1);               // B(i-1,j,k)
                    
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
    
    cout << "Set PhononLattice parameters (J–K–Γ–Γ' in GLOBAL Cartesian frame):" << endl;
    cout << "  Exchange matrices transformed: J_global = R · J_local · Rᵀ" << endl;
    cout << "  Local frame basis: x'=(1,1,-2)/√6, y'=(-1,1,0)/√2, z'=(1,1,1)/√3" << endl;
    cout << "  J=" << sp_params.J << ", K=" << sp_params.K
         << ", Γ=" << sp_params.Gamma << ", Γ'=" << sp_params.Gammap << endl;
    cout << "  J2_A=" << sp_params.J2_A << ", J2_B=" << sp_params.J2_B << endl;
    cout << "  J3=" << sp_params.J3 << ", J7=" << sp_params.J7 << endl;
    cout << "  Spin-phonon: leading C6-allowed quadratic δX_γ(ε) modulation" << endl;
    cout << "    E1 λ0(J,K,Γ,Γ')=(" << sp_params.lambda_E1_J_0 << ", "
         << sp_params.lambda_E1_K_0 << ", " << sp_params.lambda_E1_Gamma_0
         << ", " << sp_params.lambda_E1_Gammap_0 << ")" << endl;
    cout << "    E1 λ2(J,K,Γ,Γ')=(" << sp_params.lambda_E1_J_2 << ", "
         << sp_params.lambda_E1_K_2 << ", " << sp_params.lambda_E1_Gamma_2
         << ", " << sp_params.lambda_E1_Gammap_2 << ")" << endl;
    cout << "    E1 λ(J7,0)=" << sp_params.lambda_E1_J7_0
         << " so J7_eff=J7+λ(J7,0)|ε|²" << endl;
    cout << "  E1 mode: ω_E1=" << ph_params.omega_E1 << ", γ_E1=" << ph_params.gamma_E1
         << ", λ_E1(quartic)=" << ph_params.lambda_E1_quartic
         << ", Z*=" << ph_params.Z_star << endl;
    cout << "  Drive: E0_1=" << dr_params.E0_1 << ", ω_1=" << dr_params.omega_1
         << ", E0_2=" << dr_params.E0_2 << ", ω_2=" << dr_params.omega_2 << endl;
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
    
    // Add ring exchange contribution
    E += ring_exchange_energy();
    
    return E;
}

double PhononLattice::ring_exchange_energy() const {
    const double qx = phonons.Q_x_E1;
    const double qy = phonons.Q_y_E1;
    const double J7 = effective_J7(qx, qy);
    return J7 * ring_exchange_normalized();
}

double PhononLattice::ring_exchange_normalized() const {
    // Six-spin ring exchange on hexagonal plaquettes
    // H_7 = (J_7/6) Σ_{hex} [2(S_i·S_j)(S_k·S_l)(S_m·S_n)
    //                       -6(S_i·S_k)(S_j·S_l)(S_m·S_n)
    //                       +3(S_i·S_l)(S_j·S_k)(S_m·S_n)
    //                       +3(S_i·S_k)(S_j·S_m)(S_l·S_n)
    //                       -(S_i·S_l)(S_j·S_m)(S_k·S_n)
    //                       + cyclic permutations of (i,j,k,l,m,n)]
    //
    // The cyclic permutations are: (0,1,2,3,4,5), (1,2,3,4,5,0), (2,3,4,5,0,1),
    //                              (3,4,5,0,1,2), (4,5,0,1,2,3), (5,0,1,2,3,4)
    
    if (hexagons.empty()) {
        return 0.0;
    }
    
    double E = 0.0;
    constexpr double prefactor = 1.0 / 6.0;
    
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

double PhononLattice::phonon_energy() const {
    // E1 mode (zone-center, single 2-component coordinate):
    //   E_ph = (1/2)(V_x²+V_y²) + (1/2) ω_E1² (Q_x²+Q_y²) + (λ_E1_quartic/4)(Q_x²+Q_y²)²
    const double T = phonons.kinetic_energy();
    const double Q_sq = phonons.Q_x_E1 * phonons.Q_x_E1 + phonons.Q_y_E1 * phonons.Q_y_E1;
    const double V_harm = 0.5 * phonon_params.omega_E1 * phonon_params.omega_E1 * Q_sq;
    const double V_quartic = 0.25 * phonon_params.lambda_E1_quartic * Q_sq * Q_sq;
    return T + V_harm + V_quartic;
}

double PhononLattice::spin_phonon_energy() const {
    // H_sp-ph = Σ_<ij>γ Σ_X δX_γ(ε) O_{ij,γ}^{(X)}, X ∈ {J, K, Γ, Γ'}
    // Computed via the e1_energy_local helper which folds δX_γ(ε) into the
    // local-frame exchange matrix M and contracts with R^T S.
    const double qx = phonons.Q_x_E1;
    const double qy = phonons.Q_y_E1;

    double E = 0.0;
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            const size_t j = nn_partners[i][n];
            if (j > i) {  // avoid double counting
                const int bond_type = nn_bond_types[i][n];
                E += e1_energy_local(Si, spins[j], spin_phonon_params,
                                     qx, qy, bond_type);
            }
        }
    }
    return E;
}

double PhononLattice::site_energy(const Eigen::Vector3d& spin_here, size_t site) const {
    // Local-frame energy contribution of one site against its neighbours
    // (Zeeman + NN J^{(γ)} with E1 modulation + 2nd/3rd NN). Ring exchange
    // is handled separately in site_energy_diff and total_energy.
    double E = -spin_here.dot(field[site]);

    const double qx = phonons.Q_x_E1;
    const double qy = phonons.Q_y_E1;

    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        const size_t j = nn_partners[site][n];
        const Eigen::Vector3d& Sj = spins[j];
        const int bond_type = nn_bond_types[site][n];

        // Static spin-spin contribution (full bond energy for this neighbour).
        E += spin_here.dot(nn_interaction[site][n] * Sj);

        // Add the E1 quadratic exchange-striction contribution.
        E += e1_energy_local(spin_here, Sj, spin_phonon_params,
                             qx, qy, bond_type);
    }
    for (size_t n = 0; n < j2_partners[site].size(); ++n) {
        E += spin_here.dot(j2_interaction[site][n] * spins[j2_partners[site][n]]);
    }
    for (size_t n = 0; n < j3_partners[site].size(); ++n) {
        E += spin_here.dot(j3_interaction[site][n] * spins[j3_partners[site][n]]);
    }
    return E;
}

double PhononLattice::site_energy_diff(const Eigen::Vector3d& new_spin,
                                       const Eigen::Vector3d& old_spin,
                                       size_t site) const {
    // dE = E(new_spin) - E(old_spin) for a proposed Metropolis move.
    Eigen::Vector3d delta = new_spin - old_spin;
    double dE = -delta.dot(field[site]);

    const double qx = phonons.Q_x_E1;
    const double qy = phonons.Q_y_E1;

    // NN interactions (static + E1 quadratic exchange modulation)
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        const size_t j = nn_partners[site][n];
        const Eigen::Vector3d& Sj = spins[j];
        const int bond_type = nn_bond_types[site][n];

        // Pure spin-spin interaction.
        dE += delta.dot(nn_interaction[site][n] * Sj);

        // E1 magnetoelastic contribution. e1_field_global returns
        // the H_eff(i) contribution = -∂(δH_γ)/∂S_i, so the
        // bond-energy increment is dE = -δS · H_eff.
        dE -= delta.dot(e1_field_global(Sj, spin_phonon_params, qx, qy, bond_type));
    }
    // 2nd NN interactions (no spin-phonon coupling on 2nd NN)
    for (size_t n = 0; n < j2_partners[site].size(); ++n) {
        size_t j = j2_partners[site][n];
        dE += delta.dot(j2_interaction[site][n] * spins[j]);
    }
    // 3rd NN interactions (no spin-phonon coupling on 3rd NN)
    for (size_t n = 0; n < j3_partners[site].size(); ++n) {
        size_t j = j3_partners[site][n];
        dE += delta.dot(j3_interaction[site][n] * spins[j]);
    }
    
    // Ring exchange contribution
    // For ring exchange, we compute the energy difference by calculating
    // the energy change for each hexagon containing this site
    double J7 = effective_J7(qx, qy);
    if (std::abs(J7) > 1e-12 && !site_hexagons[site].empty()) {
        double prefactor = J7 / 6.0;
        
        for (const auto& [hex_idx, pos] : site_hexagons[site]) {
            const auto& hex = hexagons[hex_idx];
            
            // Temporarily modify spin for energy calculation
            // Get spins with old value for site
            std::array<Eigen::Vector3d, 6> S_old, S_new;
            for (size_t p = 0; p < 6; ++p) {
                S_old[p] = spins[hex[p]];
                S_new[p] = spins[hex[p]];
            }
            S_new[pos] = new_spin;
            
            // Compute energy for old and new configurations
            auto compute_hex_energy = [&](const std::array<Eigen::Vector3d, 6>& S) {
                // Precompute dot products
                double d01 = S[0].dot(S[1]), d02 = S[0].dot(S[2]), d03 = S[0].dot(S[3]);
                double d04 = S[0].dot(S[4]), d05 = S[0].dot(S[5]);
                double d12 = S[1].dot(S[2]), d13 = S[1].dot(S[3]), d14 = S[1].dot(S[4]), d15 = S[1].dot(S[5]);
                double d23 = S[2].dot(S[3]), d24 = S[2].dot(S[4]), d25 = S[2].dot(S[5]);
                double d34 = S[3].dot(S[4]), d35 = S[3].dot(S[5]);
                double d45 = S[4].dot(S[5]);
                
                // Sum over 6 cyclic permutations
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
                
                return prefactor * (term0 + term1 + term2 + term3 + term4 + term5);
            };
            
            dE += compute_hex_energy(S_new) - compute_hex_energy(S_old);
        }
    }
    
    return dE;
}

// ============================================================
// DERIVATIVES
// ============================================================

double PhononLattice::dH_dQx_E1() const {
    // ∂H_sp-ph/∂ε_x for the zone-center E1 coordinate, summed over all NN bonds.
    const double qx = phonons.Q_x_E1;
    const double qy = phonons.Q_y_E1;

    // Channel-resolved derivative coefficients per bond type are bond-independent
    // (they don't depend on the bond at all because ε is uniform); the e1_exchange_dqx
    // helper still wants the bond_type to know cos(2θ_γ) and sin(2θ_γ).
    E1ExchangeCoefficients dcoeffs[3];
    for (int b = 0; b < 3; ++b) {
        dcoeffs[b] = e1_exchange_dqx(spin_phonon_params, qx, qy, b);
    }

    double deriv = 0.0;
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            const size_t j = nn_partners[i][n];
            if (j > i) {
                const int bond_type = nn_bond_types[i][n];
                deriv += e1_dH_dQ_local(Si, spins[j], dcoeffs[bond_type], bond_type);
            }
        }
    }
    // Ring-exchange contribution:
    // H_7(ε) = [J7 + λ_J7 |ε|²] R_7(spins)
    // so ∂H_7/∂Q_x = 2 λ_J7 Q_x R_7.
    deriv += dJ7_dQx_E1(qx, qy) * ring_exchange_normalized();
    return deriv;
}

double PhononLattice::dH_dQy_E1() const {
    const double qx = phonons.Q_x_E1;
    const double qy = phonons.Q_y_E1;

    E1ExchangeCoefficients dcoeffs[3];
    for (int b = 0; b < 3; ++b) {
        dcoeffs[b] = e1_exchange_dqy(spin_phonon_params, qx, qy, b);
    }

    double deriv = 0.0;
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            const size_t j = nn_partners[i][n];
            if (j > i) {
                const int bond_type = nn_bond_types[i][n];
                deriv += e1_dH_dQ_local(Si, spins[j], dcoeffs[bond_type], bond_type);
            }
        }
    }
    deriv += dJ7_dQy_E1(qx, qy) * ring_exchange_normalized();
    return deriv;
}

SpinVector PhononLattice::get_local_field(size_t site) const {
    // H_eff = -∂H/∂S_i = B - ∂H_spin/∂S_i - ∂H_sp-ph/∂S_i + (ring-exchange field)
    Eigen::Vector3d H = field[site];

    const double qx = phonons.Q_x_E1;
    const double qy = phonons.Q_y_E1;

    // NN contributions (static + E1 quadratic exchange modulation)
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        const size_t j = nn_partners[site][n];
        const Eigen::Vector3d& Sj = spins[j];
        const int bond_type = nn_bond_types[site][n];

        H -= nn_interaction[site][n] * Sj;
        H += e1_field_global(Sj, spin_phonon_params, qx, qy, bond_type);
    }

    // 2nd NN
    for (size_t n = 0; n < j2_partners[site].size(); ++n) {
        H -= j2_interaction[site][n] * spins[j2_partners[site][n]];
    }
    // 3rd NN
    for (size_t n = 0; n < j3_partners[site].size(); ++n) {
        H -= j3_interaction[site][n] * spins[j3_partners[site][n]];
    }

    // Ring exchange
    H += get_ring_exchange_field(site);

    return H;
}

SpinVector PhononLattice::get_ring_exchange_field(size_t site) const {
    return get_ring_exchange_field(site, phonons.Q_x_E1, phonons.Q_y_E1);
}

SpinVector PhononLattice::get_ring_exchange_field(size_t site, double qx, double qy) const {
    // Compute H_eff_ring = -∂H_7/∂S_site
    //
    // H_7 = (J_7/6) Σ_{hex} Σ_{cyclic perms} [
    //   2*(S_i·S_j)*(S_k·S_l)*(S_m·S_n) - 6*(S_i·S_k)*(S_j·S_l)*(S_m·S_n)
    //   + 3*(S_i·S_l)*(S_j·S_k)*(S_m·S_n) + 3*(S_i·S_k)*(S_j·S_m)*(S_l·S_n)
    //   - (S_i·S_l)*(S_j·S_m)*(S_k·S_n)
    // ]
    //
    // Taking ∂/∂S_α for site α within hexagon, we get contributions whenever α appears
    // in one of the dot products. Using ∂(S_α·S_β)/∂S_α = S_β
    
    Eigen::Vector3d H = Eigen::Vector3d::Zero();
    double J7 = effective_J7(qx, qy);
    
    if (std::abs(J7) < 1e-12 || site_hexagons[site].empty()) {
        return H;
    }
    
    double prefactor = -J7 / 6.0;  // Negative because H_eff = -∂H/∂S
    
    // Loop over all hexagons containing this site
    for (const auto& [hex_idx, pos] : site_hexagons[site]) {
        const auto& hex = hexagons[hex_idx];
        
        // Get all 6 spins
        const Eigen::Vector3d& S0 = spins[hex[0]];
        const Eigen::Vector3d& S1 = spins[hex[1]];
        const Eigen::Vector3d& S2 = spins[hex[2]];
        const Eigen::Vector3d& S3 = spins[hex[3]];
        const Eigen::Vector3d& S4 = spins[hex[4]];
        const Eigen::Vector3d& S5 = spins[hex[5]];
        
        // Precompute all pairwise dot products
        double d01 = S0.dot(S1), d02 = S0.dot(S2), d03 = S0.dot(S3);
        double d04 = S0.dot(S4), d05 = S0.dot(S5);
        double d12 = S1.dot(S2), d13 = S1.dot(S3), d14 = S1.dot(S4), d15 = S1.dot(S5);
        double d23 = S2.dot(S3), d24 = S2.dot(S4), d25 = S2.dot(S5);
        double d34 = S3.dot(S4), d35 = S3.dot(S5);
        double d45 = S4.dot(S5);
        
        // We need to compute the derivative of the sum over 6 cyclic permutations
        // with respect to S_{pos}
        // For each permutation (i,j,k,l,m,n), we have 5 terms:
        //   T1 = 2*(i·j)*(k·l)*(m·n)
        //   T2 = -6*(i·k)*(j·l)*(m·n)
        //   T3 = 3*(i·l)*(j·k)*(m·n)
        //   T4 = 3*(i·k)*(j·m)*(l·n)
        //   T5 = -(i·l)*(j·m)*(k·n)
        
        Eigen::Vector3d dH = Eigen::Vector3d::Zero();
        
        // We'll compute contributions from all 6 cyclic permutations
        // The permutations map position p to index (p+shift) mod 6
        for (int shift = 0; shift < 6; ++shift) {
            // Map: i=shift, j=shift+1, k=shift+2, l=shift+3, m=shift+4, n=shift+5 (mod 6)
            size_t pi = shift % 6;
            size_t pj = (shift + 1) % 6;
            size_t pk = (shift + 2) % 6;
            size_t pl = (shift + 3) % 6;
            size_t pm = (shift + 4) % 6;
            size_t pn = (shift + 5) % 6;
            
            // Get the spins for this permutation
            const Eigen::Vector3d& Si = spins[hex[pi]];
            const Eigen::Vector3d& Sj_perm = spins[hex[pj]];
            const Eigen::Vector3d& Sk = spins[hex[pk]];
            const Eigen::Vector3d& Sl = spins[hex[pl]];
            const Eigen::Vector3d& Sm = spins[hex[pm]];
            const Eigen::Vector3d& Sn = spins[hex[pn]];
            
            // Compute dot products for this permutation
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
            
            // Determine which role our site plays in this permutation
            // pos is the position in the hexagon (0-5)
            // We need to find which of i,j,k,l,m,n corresponds to pos
            int role = -1;
            if (pos == pi) role = 0;       // site is i
            else if (pos == pj) role = 1;  // site is j
            else if (pos == pk) role = 2;  // site is k
            else if (pos == pl) role = 3;  // site is l
            else if (pos == pm) role = 4;  // site is m
            else if (pos == pn) role = 5;  // site is n
            
            // Compute derivative based on which role our site plays
            // T1 = 2*(i·j)*(k·l)*(m·n)
            // T2 = -6*(i·k)*(j·l)*(m·n)
            // T3 = 3*(i·l)*(j·k)*(m·n)
            // T4 = 3*(i·k)*(j·m)*(l·n)
            // T5 = -(i·l)*(j·m)*(k·n)
            
            if (role == 0) {  // site is i
                // ∂T1/∂Si = 2 * Sj * (k·l) * (m·n)
                dH += 2.0 * Sj_perm * dkl * dmn;
                // ∂T2/∂Si = -6 * Sk * (j·l) * (m·n)
                dH += -6.0 * Sk * djl * dmn;
                // ∂T3/∂Si = 3 * Sl * (j·k) * (m·n)
                dH += 3.0 * Sl * djk * dmn;
                // ∂T4/∂Si = 3 * Sk * (j·m) * (l·n)
                dH += 3.0 * Sk * djm * dln;
                // ∂T5/∂Si = -Sl * (j·m) * (k·n)
                dH += -Sl * djm * dkn;
            }
            else if (role == 1) {  // site is j
                // ∂T1/∂Sj = 2 * Si * (k·l) * (m·n)
                dH += 2.0 * Si * dkl * dmn;
                // ∂T2/∂Sj = -6 * Sl * (i·k) * (m·n)
                dH += -6.0 * Sl * dik * dmn;
                // ∂T3/∂Sj = 3 * Sk * (i·l) * (m·n)
                dH += 3.0 * Sk * dil * dmn;
                // ∂T4/∂Sj = 3 * Sm * (i·k) * (l·n)
                dH += 3.0 * Sm * dik * dln;
                // ∂T5/∂Sj = -Sm * (i·l) * (k·n)
                dH += -Sm * dil * dkn;
            }
            else if (role == 2) {  // site is k
                // ∂T1/∂Sk = 2 * Sl * (i·j) * (m·n)
                dH += 2.0 * Sl * dij * dmn;
                // ∂T2/∂Sk = -6 * Si * (j·l) * (m·n)
                dH += -6.0 * Si * djl * dmn;
                // ∂T3/∂Sk = 3 * Sj_perm * (i·l) * (m·n)
                dH += 3.0 * Sj_perm * dil * dmn;
                // ∂T4/∂Sk = 3 * Si * (j·m) * (l·n)
                dH += 3.0 * Si * djm * dln;
                // ∂T5/∂Sk = -Sn * (i·l) * (j·m)
                dH += -Sn * dil * djm;
            }
            else if (role == 3) {  // site is l
                // ∂T1/∂Sl = 2 * Sk * (i·j) * (m·n)
                dH += 2.0 * Sk * dij * dmn;
                // ∂T2/∂Sl = -6 * Sj_perm * (i·k) * (m·n)
                dH += -6.0 * Sj_perm * dik * dmn;
                // ∂T3/∂Sl = 3 * Si * (j·k) * (m·n)
                dH += 3.0 * Si * djk * dmn;
                // ∂T4/∂Sl = 3 * Sn * (i·k) * (j·m)
                dH += 3.0 * Sn * dik * djm;
                // ∂T5/∂Sl = -Si * (j·m) * (k·n)
                dH += -Si * djm * dkn;
            }
            else if (role == 4) {  // site is m
                // ∂T1/∂Sm = 2 * Sn * (i·j) * (k·l)
                dH += 2.0 * Sn * dij * dkl;
                // ∂T2/∂Sm = -6 * Sn * (i·k) * (j·l)
                dH += -6.0 * Sn * dik * djl;
                // ∂T3/∂Sm = 3 * Sn * (i·l) * (j·k)
                dH += 3.0 * Sn * dil * djk;
                // ∂T4/∂Sm = 3 * Sj_perm * (i·k) * (l·n)
                dH += 3.0 * Sj_perm * dik * dln;
                // ∂T5/∂Sm = -Sj_perm * (i·l) * (k·n)
                dH += -Sj_perm * dil * dkn;
            }
            else if (role == 5) {  // site is n
                // ∂T1/∂Sn = 2 * Sm * (i·j) * (k·l)
                dH += 2.0 * Sm * dij * dkl;
                // ∂T2/∂Sn = -6 * Sm * (i·k) * (j·l)
                dH += -6.0 * Sm * dik * djl;
                // ∂T3/∂Sn = 3 * Sm * (i·l) * (j·k)
                dH += 3.0 * Sm * dil * djk;
                // ∂T4/∂Sn = 3 * Sl * (i·k) * (j·m)
                dH += 3.0 * Sl * dik * djm;
                // ∂T5/∂Sn = -Sk * (i·l) * (j·m)
                dH += -Sk * dil * djm;
            }
        }
        
        H += prefactor * dH;
    }
    
    return H;
}

// ============================================================
// EQUATIONS OF MOTION
// ============================================================

void PhononLattice::phonon_derivatives(
    const PhononState& ph, double t,
    double dHsp_dQx, double dHsp_dQy,
    PhononState& dph_dt) const
{
    // Polar THz drive
    double Ex, Ey;
    drive_params.E_field(t, Ex, Ey);

    const double omega_sq = phonon_params.omega_E1 * phonon_params.omega_E1;
    const double Q_sq = ph.Q_x_E1 * ph.Q_x_E1 + ph.Q_y_E1 * ph.Q_y_E1;
    const double l4 = phonon_params.lambda_E1_quartic;
    const double gamma = phonon_params.gamma_E1;
    const double Z = phonon_params.Z_star;

    dph_dt.Q_x_E1 = ph.V_x_E1;
    dph_dt.V_x_E1 = -omega_sq * ph.Q_x_E1
                    - l4 * Q_sq * ph.Q_x_E1
                    - gamma * ph.V_x_E1
                    - dHsp_dQx
                    + Z * Ex;

    dph_dt.Q_y_E1 = ph.V_y_E1;
    dph_dt.V_y_E1 = -omega_sq * ph.Q_y_E1
                    - l4 * Q_sq * ph.Q_y_E1
                    - gamma * ph.V_y_E1
                    - dHsp_dQy
                    + Z * Ey;
}

void PhononLattice::ode_system(const ODEState& x, ODEState& dxdt, double t) {
    // Flat layout: [S0_x, S0_y, S0_z, ..., S_{N-1}_z, Q_x, Q_y, V_x, V_y].
    const size_t spin_offset = spin_dim * lattice_size;

    // Sync the spin cache from the ODE state if ring-exchange is active
    // (get_ring_exchange_field reads spins[]).
    if (std::abs(spin_phonon_params.J7) > 1e-12 ||
        std::abs(spin_phonon_params.lambda_E1_J7_0) > 1e-12) {
        for (size_t i = 0; i < lattice_size; ++i) {
            const size_t idx = i * spin_dim;
            spins[i](0) = x[idx];
            spins[i](1) = x[idx+1];
            spins[i](2) = x[idx+2];
        }
    }

    PhononState ph;
    ph.from_array(&x[spin_offset]);

    // Time-dependent multiplicative scale on the 8 quadratic E1 coefficients.
    const double e1_scale = get_e1_coupling_scale(t);

    // Precompute the channel-resolved ∂δX_γ/∂ε_a coefficients once per bond
    // type (they depend only on ε and bond_type, not on the spins).
    const double qx = ph.Q_x_E1;
    const double qy = ph.Q_y_E1;
    E1ExchangeCoefficients dqx_coeffs[3];
    E1ExchangeCoefficients dqy_coeffs[3];
    for (int b = 0; b < 3; ++b) {
        dqx_coeffs[b] = e1_exchange_dqx(spin_phonon_params, qx, qy, b, e1_scale);
        dqy_coeffs[b] = e1_exchange_dqy(spin_phonon_params, qx, qy, b, e1_scale);
    }

    // Accumulate ∂H_sp-ph/∂ε_x and ∂H_sp-ph/∂ε_y over all NN bonds while we
    // also build the spin equations of motion below.
    double dHsp_dQx = 0.0;
    double dHsp_dQy = 0.0;

    // Spin equations:  dS/dt = S × H_eff + α S × (S × H_eff)
    for (size_t i = 0; i < lattice_size; ++i) {
        const size_t idx = i * spin_dim;
        Eigen::Vector3d Si(x[idx], x[idx+1], x[idx+2]);

        Eigen::Vector3d H = field[i];

        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            const size_t j = nn_partners[i][n];
            const size_t jdx = j * spin_dim;
            Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
            const int bond_type = nn_bond_types[i][n];

            // Static spin-spin contribution to H_eff(i).
            H -= nn_interaction[i][n] * Sj;

            // E1 quadratic exchange modulation contribution to H_eff(i).
            H += e1_field_global(Sj, spin_phonon_params, qx, qy, bond_type, e1_scale);

            // Accumulate phonon-side derivatives (each bond is counted once
            // by the j > i guard).
            if (j > i) {
                dHsp_dQx += e1_dH_dQ_local(Si, Sj, dqx_coeffs[bond_type], bond_type);
                dHsp_dQy += e1_dH_dQ_local(Si, Sj, dqy_coeffs[bond_type], bond_type);
            }
        }

        for (size_t n = 0; n < j2_partners[i].size(); ++n) {
            const size_t j = j2_partners[i][n];
            const size_t jdx = j * spin_dim;
            Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
            H -= j2_interaction[i][n] * Sj;
        }

        for (size_t n = 0; n < j3_partners[i].size(); ++n) {
            const size_t j = j3_partners[i][n];
            const size_t jdx = j * spin_dim;
            Eigen::Vector3d Sj(x[jdx], x[jdx+1], x[jdx+2]);
            H -= j3_interaction[i][n] * Sj;
        }

        H += get_ring_exchange_field(i, qx, qy);

        // Langevin thermostat: inject pre-generated Gaussian noise field
        // (held constant across the RK4 sub-stages of one macro step).
        if (use_langevin_noise) {
            H += langevin_noise[i];
        }

        const Eigen::Vector3d dSdt = spin_derivative(Si, H);
        dxdt[idx]   = dSdt(0);
        dxdt[idx+1] = dSdt(1);
        dxdt[idx+2] = dSdt(2);
    }

    if (std::abs(spin_phonon_params.lambda_E1_J7_0) > 1e-12) {
        const double R7 = ring_exchange_normalized();
        dHsp_dQx += dJ7_dQx_E1(qx, qy) * R7;
        dHsp_dQy += dJ7_dQy_E1(qx, qy) * R7;
    }

    // Phonon equations
    PhononState dph_dt;
    phonon_derivatives(ph, t, dHsp_dQx, dHsp_dQy, dph_dt);
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
    string out_dir, size_t save_interval, string method,
    double abs_tol_in, double rel_tol_in) 
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
    ODEState state = spins_to_state();
    
#ifdef HDF5_ENABLED
    // Create HDF5 writer with comprehensive metadata (like Lattice class)
    std::unique_ptr<HDF5MDWriter> hdf5_writer;

    // Storage for the zone-center E1 phonon trajectory.
    // The phonon has 4 DOF: (Q_x, Q_y, V_x, V_y).
    vector<double> times_phonon;
    vector<double> Qx_E1_traj, Qy_E1_traj;
    vector<double> Vx_E1_traj, Vy_E1_traj;
    vector<double> Ex_drive_traj, Ey_drive_traj;
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
            
            // Extract zone-center E1 phonon state from the tail of the ODE
            // state vector. Flat layout: [..spins.., Q_x, Q_y, V_x, V_y].
            const size_t p_idx = spin_state_size;
            const double Qx_E1 = x[p_idx + 0];
            const double Qy_E1 = x[p_idx + 1];
            const double Vx_E1 = x[p_idx + 2];
            const double Vy_E1 = x[p_idx + 3];

            // Sample the THz drive at this snapshot for diagnostics.
            double Ex_t, Ey_t;
            drive_params.E_field(t, Ex_t, Ey_t);

#ifdef HDF5_ENABLED
            // Write full spin configuration to HDF5 (like Lattice class)
            if (hdf5_writer) {
                hdf5_writer->write_flat_step(t, M_staggered, M_local, M_global, x.data());

                times_phonon.push_back(t);
                Qx_E1_traj.push_back(Qx_E1); Qy_E1_traj.push_back(Qy_E1);
                Vx_E1_traj.push_back(Vx_E1); Vy_E1_traj.push_back(Vy_E1);
                Ex_drive_traj.push_back(Ex_t); Ey_drive_traj.push_back(Ey_t);

                // Compute energy for monitoring
                const_cast<PhononLattice*>(this)->state_to_spins(x);
                energy_traj.push_back(energy_density());
            }
#endif

            if (step_count % (save_interval * 10) == 0) {
                const double Qmag = std::sqrt(Qx_E1 * Qx_E1 + Qy_E1 * Qy_E1);
                cout << "t=" << t << ", |M|=" << M_local.norm()
                     << ", |M_stag|=" << M_staggered.norm()
                     << ", ε=(" << Qx_E1 << ", " << Qy_E1 << "), |ε|=" << Qmag
                     << ", E_drive=(" << Ex_t << ", " << Ey_t << ")" << endl;
            }

            save_count++;
        }
        step_count++;
    };
    
    auto system_func = [this](const ODEState& x, ODEState& dxdt, double t) {
        this->ode_system(x, dxdt, t);
    };
    
    // Integrate using selected method.
    // User overrides win when positive; otherwise fall back to the
    // method-aware defaults (1e-6, or 1e-8 for Bulirsch-Stoer).
    double abs_tol = (abs_tol_in > 0.0)
        ? abs_tol_in
        : ((method == "bulirsch_stoer") ? 1e-8 : 1e-6);
    double rel_tol = (rel_tol_in > 0.0)
        ? rel_tol_in
        : ((method == "bulirsch_stoer") ? 1e-8 : 1e-6);
    integrate_ode_system(system_func, state, T_start, T_end, dt_initial,
                        observer, method, true, abs_tol, rel_tol);
    
    state_to_spins(state);
    
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
        
        // Zone-center E1 mode (single 2-component field).
        write_dataset("Qx_E1", Qx_E1_traj);
        write_dataset("Qy_E1", Qy_E1_traj);
        write_dataset("Vx_E1", Vx_E1_traj);
        write_dataset("Vy_E1", Vy_E1_traj);

        // THz drive samples on the saved grid.
        write_dataset("Ex_drive", Ex_drive_traj);
        write_dataset("Ey_drive", Ey_drive_traj);

        // Energy
        write_dataset("energy", energy_traj);

        // Write phonon parameters as metadata
        H5::Group meta_group = h5file.openGroup("/metadata");
        H5::DataSpace scalar_space(H5S_SCALAR);

        auto write_scalar = [&](const string& name, double val) {
            H5::Attribute attr = meta_group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar_space);
            attr.write(H5::PredType::NATIVE_DOUBLE, &val);
        };

        // E1 phonon parameters
        write_scalar("omega_E1", phonon_params.omega_E1);
        write_scalar("gamma_E1", phonon_params.gamma_E1);
        write_scalar("lambda_E1_quartic", phonon_params.lambda_E1_quartic);
        write_scalar("Z_star", phonon_params.Z_star);

        // E1 magnetoelastic couplings (one isotropic + one anisotropic per channel)
        write_scalar("lambda_E1_J_0", spin_phonon_params.lambda_E1_J_0);
        write_scalar("lambda_E1_J_2", spin_phonon_params.lambda_E1_J_2);
        write_scalar("lambda_E1_K_0", spin_phonon_params.lambda_E1_K_0);
        write_scalar("lambda_E1_K_2", spin_phonon_params.lambda_E1_K_2);
        write_scalar("lambda_E1_Gamma_0", spin_phonon_params.lambda_E1_Gamma_0);
        write_scalar("lambda_E1_Gamma_2", spin_phonon_params.lambda_E1_Gamma_2);
        write_scalar("lambda_E1_Gammap_0", spin_phonon_params.lambda_E1_Gammap_0);
        write_scalar("lambda_E1_Gammap_2", spin_phonon_params.lambda_E1_Gammap_2);
        write_scalar("lambda_E1_J7_0", spin_phonon_params.lambda_E1_J7_0);

        // Pump pulse 1 parameters
        write_scalar("pump_amplitude", drive_params.E0_1);
        write_scalar("pump_frequency", drive_params.omega_1);
        write_scalar("pump_time", drive_params.t_1);
        write_scalar("pump_width", drive_params.sigma_1);
        write_scalar("pump_phase", drive_params.phi_1);
        write_scalar("pump_polarization", drive_params.theta_1);
        
        // Pump pulse 2 parameters (probe)
        write_scalar("probe_amplitude", drive_params.E0_2);
        write_scalar("probe_frequency", drive_params.omega_2);
        write_scalar("probe_time", drive_params.t_2);
        write_scalar("probe_width", drive_params.sigma_2);
        write_scalar("probe_phase", drive_params.phi_2);
        write_scalar("probe_polarization", drive_params.theta_2);
        
        phonon_group.close();
        meta_group.close();
        h5file.close();
        
        cout << "HDF5 trajectory saved with " << save_count << " snapshots (full spin + phonon)" << endl;
    }
#endif
    
    cout << "Dynamics complete! (" << step_count << " steps, " << save_count << " saved)" << endl;
}

// ============================================================
// LANGEVIN DYNAMICS (qualitative, fixed-step RK4 + per-step noise)
// ============================================================

void PhononLattice::integrate_langevin(double t_start, double t_end, double dt,
                                       const string& output_dir,
                                       size_t save_every,
                                       uint64_t seed) {
    if (langevin_temperature <= 0.0) {
        std::cerr << "ERROR: integrate_langevin called with langevin_temperature = "
                  << langevin_temperature << " (must be > 0). Aborting." << std::endl;
        return;
    }
    if (alpha_gilbert <= 0.0) {
        std::cerr << "WARNING: alpha_gilbert = " << alpha_gilbert
                  << " ≤ 0 — Langevin dynamics requires positive damping. "
                  << "Defaulting to alpha_gilbert = 0.01." << std::endl;
        alpha_gilbert = 0.01;
    }
    if (dt <= 0.0) {
        std::cerr << "ERROR: dt must be > 0. Aborting." << std::endl;
        return;
    }
    if (t_end <= t_start) {
        std::cerr << "ERROR: t_end (" << t_end << ") <= t_start (" << t_start
                  << "). Aborting." << std::endl;
        return;
    }

    if (!output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
    }

    // Stochastic LLG with Strang-like splitting (Step A: deterministic RK4
    // on H_eff; Step B: stochastic Euler-Maruyama on the noise force).
    // Per-step Wiener increment: dW_α ~ N(0, dt) per Cartesian α, per site.
    // Spin update from noise: dS = -|γ| S × (σ_W dW) where σ_W is calibrated
    //     by FDT  σ_W² = 2 α k_B T / (|S|).
    // We sample η ~ N(0,1) and inject H_noise = sigma_eff η, where
    //     sigma_eff = sqrt(2 α k_B T / (|S| dt))
    // and apply ONE Euler half-step S_i ← S_i - dt · S_i × H_noise per macro
    // step (no double counting across the RK4 sub-stages).
    const double sigma_eff =
        std::sqrt(2.0 * alpha_gilbert * langevin_temperature / (spin_length * dt));

    // RNG setup
    if (seed == 0) {
        std::random_device rd;
        seed = ((uint64_t)rd() << 32) | rd();
    }
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Allocate noise buffer (used as a temporary force inside the stochastic
    // half-step; kept zero during the deterministic RK4 sub-stages).
    langevin_noise.assign(lattice_size, Eigen::Vector3d::Zero());
    use_langevin_noise = false;  // off during deterministic RK4

    std::cout << "PhononLattice Langevin dynamics (Strang split: RK4 + Euler-Maruyama)"
              << std::endl;
    std::cout << "  t = " << t_start << " → " << t_end << ", dt = " << dt << std::endl;
    std::cout << "  T (k_B T)            = " << langevin_temperature << std::endl;
    std::cout << "  Gilbert damping α    = " << alpha_gilbert << std::endl;
    std::cout << "  Noise sigma          = " << sigma_eff << std::endl;
    std::cout << "  RNG seed             = " << seed << std::endl;
    std::cout << "  Save every           = " << save_every << " steps" << std::endl;

    // Pack current (spins, phonons) into the flat ODE state vector.
    ODEState state(state_size);
    for (size_t i = 0; i < lattice_size; ++i) {
        state[i * spin_dim + 0] = spins[i](0);
        state[i * spin_dim + 1] = spins[i](1);
        state[i * spin_dim + 2] = spins[i](2);
    }
    phonons.to_array(&state[spin_dim * lattice_size]);

    // Helper to copy state vector back into spins[]/phonons (for observables).
    auto sync_back = [&]() {
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i](0) = state[i * spin_dim + 0];
            spins[i](1) = state[i * spin_dim + 1];
            spins[i](2) = state[i * spin_dim + 2];
        }
        phonons.from_array(&state[spin_dim * lattice_size]);
    };

    // Trajectory storage (text output for portability)
    struct Frame {
        double t;
        Eigen::Vector3d M, M_stag;
        double E_total;
        double Qx, Qy, Vx, Vy;
        SpinConfig spin_snapshot;
    };
    vector<Frame> traj;
    traj.reserve(static_cast<size_t>((t_end - t_start) / (dt * save_every)) + 1);

    // RK4 integration with per-step Gaussian noise
    using Stepper = boost::numeric::odeint::runge_kutta4<ODEState>;
    Stepper stepper;
    auto rhs = [this](const ODEState& x, ODEState& dxdt, double t) {
        ode_system(x, dxdt, t);
    };

    double t = t_start;
    size_t step = 0;
    const size_t spin_offset = spin_dim * lattice_size;

    while (t < t_end) {
        // Save observables BEFORE stepping
        if (step % save_every == 0) {
            sync_back();
            Frame f;
            f.t = t;
            f.M = magnetization_local();
            f.M_stag = magnetization_local_antiferro();
            f.E_total = total_energy();
            f.Qx = state[spin_offset + 0];
            f.Qy = state[spin_offset + 1];
            f.Vx = state[spin_offset + 2];
            f.Vy = state[spin_offset + 3];
            f.spin_snapshot = spins;
            traj.push_back(f);
        }

        // ── Step A: deterministic RK4 on H_eff (no noise) ──
        use_langevin_noise = false;
        stepper.do_step(rhs, state, t, dt);

        // ── Step B: stochastic Euler-Maruyama on the noise force ──
        //
        //   H_noise_i  = sigma_eff * η_i,    η_i ~ N(0, I_3)
        //   ΔS_i       = +dt · S_i × H_noise_i  − dt α / |S| · S_i × (S_i × H_noise_i)
        //
        // Signs match the deterministic LLG convention used by spin_derivative
        // (precession +S×H, dissipation −α S×(S×H)/|S|). Drawing η ~ N(0,1)
        // with σ² = 2 α k_B T / (|S| dt) realizes the Itô discretization of
        // the Wiener increment dW = sqrt(dt) η consistent with the FDT.
        const double inv_S = (spin_length > 0) ? alpha_gilbert / spin_length : 0.0;
        for (size_t i = 0; i < lattice_size; ++i) {
            const size_t idx = i * spin_dim;
            Eigen::Vector3d S(state[idx], state[idx + 1], state[idx + 2]);
            Eigen::Vector3d H_noise(sigma_eff * normal(rng),
                                    sigma_eff * normal(rng),
                                    sigma_eff * normal(rng));
            Eigen::Vector3d dS = dt * S.cross(H_noise);
            if (alpha_gilbert > 0.0) {
                dS -= dt * inv_S * S.cross(S.cross(H_noise));
            }
            state[idx]     += dS(0);
            state[idx + 1] += dS(1);
            state[idx + 2] += dS(2);

            // Renormalize to enforce |S| = spin_length
            const double sx = state[idx];
            const double sy = state[idx + 1];
            const double sz = state[idx + 2];
            const double n = std::sqrt(sx * sx + sy * sy + sz * sz);
            if (n > 0.0) {
                const double s = spin_length / n;
                state[idx]     = sx * s;
                state[idx + 1] = sy * s;
                state[idx + 2] = sz * s;
            }
        }

        t += dt;
        ++step;
    }

    // Final sync
    sync_back();

    // Disable noise injection so subsequent calls (if any) are deterministic
    use_langevin_noise = false;

    // Write trajectory to text file
    if (!output_dir.empty() && !traj.empty()) {
        std::ofstream f(output_dir + "/langevin_trajectory.txt");
        f << "# t  Mx My Mz |M|  Mstag_x Mstag_y Mstag_z |M_stag|  "
          << "E_total  Qx_E1 Qy_E1 Vx_E1 Vy_E1\n";
        f << std::scientific << std::setprecision(10);
        for (const auto& fr : traj) {
            const double Mn = fr.M.norm();
            const double Msn = fr.M_stag.norm();
            f << fr.t << ' '
              << fr.M(0) << ' ' << fr.M(1) << ' ' << fr.M(2) << ' ' << Mn << ' '
              << fr.M_stag(0) << ' ' << fr.M_stag(1) << ' ' << fr.M_stag(2)
              << ' ' << Msn << ' '
              << fr.E_total << ' '
              << fr.Qx << ' ' << fr.Qy << ' ' << fr.Vx << ' ' << fr.Vy << '\n';
        }
        std::cout << "Langevin trajectory written to " << output_dir
                  << "/langevin_trajectory.txt (" << traj.size() << " snapshots)" << std::endl;

        std::ofstream sf(output_dir + "/langevin_spins.txt");
        sf << "# Langevin spin snapshots\n";
        sf << "# frame t site Sx Sy Sz\n";
        sf << std::scientific << std::setprecision(12);
        for (size_t frame_idx = 0; frame_idx < traj.size(); ++frame_idx) {
            const auto& fr = traj[frame_idx];
            for (size_t site = 0; site < fr.spin_snapshot.size(); ++site) {
                const auto& S = fr.spin_snapshot[site];
                sf << frame_idx << ' ' << fr.t << ' ' << site << ' '
                   << S(0) << ' ' << S(1) << ' ' << S(2) << '\n';
            }
        }
        std::cout << "Langevin spin snapshots written to " << output_dir
                  << "/langevin_spins.txt" << std::endl;

        // Also save final state
        save_spin_config(output_dir + "/final_spins.txt");
        save_state_hdf5(output_dir + "/final_state.h5");
    }

    std::cout << "Langevin dynamics complete (" << step << " steps, "
              << traj.size() << " snapshots saved)." << std::endl;
}

// ============================================================
// MONTE CARLO METHODS
// ============================================================

double PhononLattice::metropolis(double T, bool gaussian_move, double sigma) {
    if (T <= 0) return 0.0;
    
    const double beta = 1.0 / T;
    size_t accepted = 0;
    
    std::uniform_int_distribution<size_t> site_dist(0, lattice_size - 1);
    
    for (size_t sweep_step = 0; sweep_step < lattice_size; ++sweep_step) {
        size_t i = site_dist(rng);
        
        Eigen::Vector3d old_spin = spins[i];
        Eigen::Vector3d new_spin;
        if (gaussian_move) {
            new_spin = gaussian_spin_move(old_spin, sigma);
        } else {
            new_spin = gen_random_spin();
        }
        
        double dE = site_energy_diff(new_spin, old_spin, i);
        
        double rand_val = uniform_dist(rng);
        const bool accept = (dE < 0.0) || (rand_val < std::exp(-beta * dE));
        if (accept) {
            spins[i] = new_spin;
            accepted++;
        }
    }
    return static_cast<double>(accepted) / lattice_size;
}

void PhononLattice::overrelaxation() {
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
    bool adiabatic_phonons, bool gaussian_move) 
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
    vector<double> Qx_data, Qy_data;  ///< zone-center E1 components (if adiabatic)

    if (save_observables && !out_dir.empty()) {
        h5file = std::make_unique<H5::H5File>(out_dir + "/annealing.h5", H5F_ACC_TRUNC);
    }
#endif
    
    double T = T_start;
    size_t temp_step = 0;
    
    while (T > T_end) {
        double accepted_rate = 0;
        
        // Perform n_steps sweeps at this temperature
        for (size_t step = 0; step < n_steps; ++step) {
            accepted_rate += metropolis(T, gaussian_move);
            
            // Overrelaxation
            if (overrelax_rate > 0 && step % overrelax_rate == 0) {
                overrelaxation();
            }
        }
        
        // If using adiabatic phonons, relax phonons to equilibrium for current spin configuration
        if (adiabatic_phonons) {
            relax_phonons(1e-10, 1000, 1.0);
        }
        
        // Calculate acceptance rate (only counts Metropolis moves, not overrelaxation)
        double acceptance = accepted_rate / double(n_steps);
        
        // Progress report every 10 temperature steps or near the end
        if (temp_step % 10 == 0 || T <= T_end * 1.5) {
            double E = energy_density();
            Eigen::Vector3d M = magnetization_local();
            Eigen::Vector3d M_stag = magnetization_local_antiferro();
            cout << "T=" << std::scientific << std::setprecision(4) << T 
                 << ", E/N=" << std::fixed << std::setprecision(6) << E 
                 << ", acc=" << std::fixed << std::setprecision(4) << acceptance
                 << ", |M|=" << std::fixed << std::setprecision(4) << M.norm()
                 << ", |M_stag|=" << std::fixed << std::setprecision(4) << M_stag.norm();
            if (adiabatic_phonons) {
                cout << ", ε=(" << std::fixed << std::setprecision(4) << phonons.Q_x_E1
                     << ", " << phonons.Q_y_E1 << ")"
                     << ", |ε|=" << std::fixed << std::setprecision(4) << E1_amplitude();
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
            }
        }
#endif
        
        // Cool down
        T *= cooling_rate;
        ++temp_step;
    }
    
    // Final report
    double E_final = energy_density();
    Eigen::Vector3d M_final = magnetization_local();
    Eigen::Vector3d M_stag_final = magnetization_local_antiferro();
    cout << "\n=== Simulated Annealing Complete ===" << endl;
    cout << "Temperature steps: " << temp_step << endl;
    cout << "Final energy density: " << E_final << endl;
    cout << "Final magnetization: [" << M_final.transpose() << "], |M|=" << M_final.norm() << endl;
    cout << "Final staggered M: [" << M_stag_final.transpose() << "], |M_stag|=" << M_stag_final.norm() << endl;
    if (adiabatic_phonons) {
        cout << "Zone-center E1 phonon equilibrium:" << endl;
        cout << "  ε = (" << phonons.Q_x_E1 << ", " << phonons.Q_y_E1 << ")"
             << ", |ε| = " << E1_amplitude() << endl;
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
            ds = ann_group.createDataSet("Qx_E1", H5::PredType::NATIVE_DOUBLE, dataspace);
            ds.write(Qx_data.data(), H5::PredType::NATIVE_DOUBLE);
            ds = ann_group.createDataSet("Qy_E1", H5::PredType::NATIVE_DOUBLE, dataspace);
            ds.write(Qy_data.data(), H5::PredType::NATIVE_DOUBLE);
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
        cout << "  |ε_E1| = " << E1_amplitude() << endl;

        cout << "\nPerforming " << n_deterministics << " deterministic sweeps at T=0..." << endl;
        deterministic_sweep(n_deterministics);
        
        // If using adiabatic phonons, relax phonons again after deterministic sweeps
        if (adiabatic_phonons) {
            relax_phonons(1e-10, 1000, 1.0);

            // Enforce a joint spin-phonon equilibrium so subsequent dynamics start from a stationary state
            cout << "Running joint spin-phonon relaxation to enforce equilibrium before dynamics..." << endl;
            relax_joint(1e-10, 100, 1, false);
        }
        
        // Energy breakdown AFTER deterministic sweeps
        cout << "\n=== Energy AFTER deterministic sweeps ===" << endl;
        cout << "  Spin energy:       " << spin_energy() << " (" << spin_energy()/lattice_size << " per site)" << endl;
        cout << "  Phonon energy:     " << phonon_energy() << endl;
        cout << "  Spin-phonon energy: " << spin_phonon_energy() << endl;
        cout << "  Total energy:      " << total_energy() << " (" << energy_density() << " per site)" << endl;
        cout << "  |ε_E1| = " << E1_amplitude() << endl;
        
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
    cout << "Relaxing zone-center E1 phonon to equilibrium for current spin configuration..." << endl;

    const double omega_sq = phonon_params.omega_E1 * phonon_params.omega_E1;
    const double l4 = phonon_params.lambda_E1_quartic;

    // Newton iteration on the equilibrium condition
    //   F_a(ε) = ω² ε_a + λ4 (ε_x²+ε_y²) ε_a + ∂H_sp-ph/∂ε_a = 0,  a = x, y.
    // The spin-phonon force depends on ε itself (since δX_γ ∝ ε²), so we need
    // a true Newton iteration; the diagonal Jacobian below is an approximation
    // and we damp the step to maintain stability.
    auto dHdQ_at = [this](double qx, double qy, double& dHdqx, double& dHdqy) {
        // Re-evaluate ∂H_sp-ph/∂ε_a at the candidate ε using the channel-resolved
        // derivative coefficients.
        E1ExchangeCoefficients dqx_coeffs[3];
        E1ExchangeCoefficients dqy_coeffs[3];
        for (int b = 0; b < 3; ++b) {
            dqx_coeffs[b] = e1_exchange_dqx(spin_phonon_params, qx, qy, b);
            dqy_coeffs[b] = e1_exchange_dqy(spin_phonon_params, qx, qy, b);
        }
        double Dx = 0.0, Dy = 0.0;
        for (size_t i = 0; i < lattice_size; ++i) {
            const Eigen::Vector3d& Si = spins[i];
            for (size_t n = 0; n < nn_partners[i].size(); ++n) {
                const size_t j = nn_partners[i][n];
                if (j > i) {
                    const int bond_type = nn_bond_types[i][n];
                    Dx += e1_dH_dQ_local(Si, spins[j], dqx_coeffs[bond_type], bond_type);
                    Dy += e1_dH_dQ_local(Si, spins[j], dqy_coeffs[bond_type], bond_type);
                }
            }
        }
        if (std::abs(spin_phonon_params.lambda_E1_J7_0) > 1e-12) {
            const double R7 = ring_exchange_normalized();
            Dx += dJ7_dQx_E1(qx, qy) * R7;
            Dy += dJ7_dQy_E1(qx, qy) * R7;
        }
        dHdqx = Dx;
        dHdqy = Dy;
    };

    // Initial guess: linear approximation about ε = 0.
    // At ε = 0 the spin-phonon force is identically zero (δX_γ is quadratic).
    // So we start at ε = 0 and let Newton + small random kick (handled by
    // upstream callers) drive the iteration.
    double qx = phonons.Q_x_E1;
    double qy = phonons.Q_y_E1;

    bool converged = false;
    for (size_t iter = 0; iter < max_iter; ++iter) {
        double dHdqx = 0.0, dHdqy = 0.0;
        dHdQ_at(qx, qy, dHdqx, dHdqy);

        const double Q_sq = qx * qx + qy * qy;
        const double Fx = omega_sq * qx + l4 * Q_sq * qx + dHdqx;
        const double Fy = omega_sq * qy + l4 * Q_sq * qy + dHdqy;

        const double residual = std::sqrt(Fx * Fx + Fy * Fy);
        if (residual < tol) {
            converged = true;
            cout << "  Phonon relaxation converged in " << iter << " iterations." << endl;
            break;
        }

        // Diagonal Jacobian approximation (ignoring spin-side ε dependence).
        // For small ε this is dominated by ω², which is already a good preconditioner.
        const double Jxx = omega_sq + l4 * (3.0 * qx * qx + qy * qy);
        const double Jyy = omega_sq + l4 * (qx * qx + 3.0 * qy * qy);

        if (Jxx > 1e-12) qx -= damping * Fx / Jxx;
        if (Jyy > 1e-12) qy -= damping * Fy / Jyy;

        if (iter > 0 && iter % 200 == 0) {
            cout << "  iter=" << iter << ", |F|=" << residual
                 << ", ε=(" << qx << ", " << qy << ")" << endl;
        }
    }

    phonons.Q_x_E1 = qx;
    phonons.Q_y_E1 = qy;
    phonons.V_x_E1 = 0.0;
    phonons.V_y_E1 = 0.0;

    cout << "  E1 equilibrium: ε=(" << qx << ", " << qy
         << "), |ε|=" << std::sqrt(qx * qx + qy * qy) << endl;
    cout << "  --- Energy after phonon relaxation ---" << endl;
    cout << "    Spin energy:        " << spin_energy()
         << " (" << spin_energy() / lattice_size << " per site)" << endl;
    cout << "    Phonon energy:      " << phonon_energy() << endl;
    cout << "    Spin-phonon energy: " << spin_phonon_energy() << endl;
    cout << "    Total energy:       " << total_energy()
         << " (" << energy_density() << " per site)" << endl;

    if (!converged) {
        cout << "  WARNING: phonon relaxation did not converge in " << max_iter << " iterations." << endl;
    }
    return converged;
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
            // Final phonon relaxation to ensure phonons are at equilibrium for final spin config
            // This is needed because deterministic_sweep() moved spins after the last relax_phonons()
            if (!phonon_only) {
                relax_phonons(1e-10, 1000, 1.0);
            }
            
            cout << "  " << (phonon_only ? "Phonon-only" : "Joint") << " relaxation converged in " << iter << " iterations!" << endl;
            cout << "\n=== Energy AFTER " << (phonon_only ? "phonon-only" : "joint") << " relaxation (equilibrium state) ===" << endl;
            cout << "  Spin energy:       " << spin_energy() << " (" << spin_energy()/lattice_size << " per site)" << endl;
            cout << "  Phonon energy:     " << phonon_energy() << endl;
            cout << "  Spin-phonon energy: " << spin_phonon_energy() << endl;
            cout << "  Total energy:      " << total_energy() << " (" << energy_density() << " per site)" << endl;
            cout << "  |Q_E1| = " << E1_amplitude() << endl;
            return true;
        }
        
        prev_energy = curr_energy;
        prev_Q_E = curr_Q_E;
    }
    
    // Final phonon relaxation even if not converged, to ensure V=0 and phonons at equilibrium
    if (!phonon_only) {
        relax_phonons(1e-10, 1000, 1.0);
    }
    
    cout << "  WARNING: " << (phonon_only ? "Phonon-only" : "Joint") << " relaxation did not fully converge after " << max_iter << " iterations" << endl;
    cout << "\n=== Energy after " << max_iter << " iterations ===" << endl;
    cout << "  Spin energy:       " << spin_energy() << " (" << spin_energy()/lattice_size << " per site)" << endl;
    cout << "  Phonon energy:     " << phonon_energy() << endl;
    cout << "  Spin-phonon energy: " << spin_phonon_energy() << endl;
    cout << "  Total energy:      " << total_energy() << " (" << energy_density() << " per site)" << endl;
    cout << "  |Q_E1| = " << E1_amplitude() << endl;
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
    
    // Save zone-center E1 phonon state: [Q_x, Q_y, V_x, V_y]
    {
        hsize_t dims[1] = {PhononState::N_DOF};
        H5::DataSpace dataspace(1, dims);
        H5::DataSet dataset = phonon_group.createDataSet("state", H5::PredType::NATIVE_DOUBLE, dataspace);

        double ph_data[PhononState::N_DOF];
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
    
    // Load zone-center E1 phonon state
    {
        H5::DataSet dataset = file.openDataSet("/phonons/state");
        double ph_data[PhononState::N_DOF];
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
    double T_start, double T_end, double step_size, const string& method,
    bool pulse_window_chunking,
    double abs_tol, double rel_tol) {
    
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
                size_t atom = i % N_atoms;
                double sign = afm_sublattice_signs[atom];
                Eigen::Vector3d S(x[i*spin_dim], x[i*spin_dim+1], x[i*spin_dim+2]);
                M_local += S;
                // Transform to global frame using sublattice frame
                Eigen::Vector3d S_global = sublattice_frames[atom] * S;
                M_antiferro += sign * S_global;
                M_global += S_global;
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
    
    // ---- W3: pulse-window-aware chunked integration ------------------
    // Pass `step_size` (not seg.dt_init) so observer cadence is uniform
    // across all segments — required for MPI buffer-size correctness in
    // pump_probe_spectroscopy_mpi.  See mixed_lattice.h for full notes.
    if (pulse_window_chunking) {
        namespace ck = classical_spin_pulse_chunking;
        const auto segments = ck::build_pulse_segments(
            T_start, T_end,
            /*pulse_centers=*/ {t_B},
            /*window=*/ ck::kPulseWindowSigmas * pulse_width,
            /*T_step=*/ step_size,
            /*free_dt_factor=*/ ck::kFreeDtFactor);
        for (const auto& seg : segments) {
            integrate_ode_system(system_func, state,
                                 seg.t0, seg.t1, step_size,
                                 observer, method, false, abs_tol, rel_tol);
        }
    } else {
        integrate_ode_system(system_func, state, T_start, T_end, step_size,
                            observer, method, false, abs_tol, rel_tol);
    }

    // Reset drive
    drive_params.E0_1 = 0.0;
    
    return trajectory;
}

PhononLattice::MagTrajectory PhononLattice::double_pulse_drive(
    double polarization_1, double t_B_1,
    double polarization_2, double t_B_2,
    double pulse_amp, double pulse_width, double pulse_freq,
    double T_start, double T_end, double step_size, const string& method,
    bool pulse_window_chunking,
    double abs_tol, double rel_tol) {
    
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
                size_t atom = i % N_atoms;
                double sign = afm_sublattice_signs[atom];
                Eigen::Vector3d S(x[i*spin_dim], x[i*spin_dim+1], x[i*spin_dim+2]);
                M_local += S;
                // Transform to global frame using sublattice frame
                Eigen::Vector3d S_global = sublattice_frames[atom] * S;
                M_antiferro += sign * S_global;
                M_global += S_global;
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
    
    // ---- W3: pulse-window-aware chunked integration ------------------
    // See single_pulse_drive() — uniform observer cadence required.
    if (pulse_window_chunking) {
        namespace ck = classical_spin_pulse_chunking;
        const auto segments = ck::build_pulse_segments(
            T_start, T_end,
            /*pulse_centers=*/ {t_B_1, t_B_2},
            /*window=*/ ck::kPulseWindowSigmas * pulse_width,
            /*T_step=*/ step_size,
            /*free_dt_factor=*/ ck::kFreeDtFactor);
        for (const auto& seg : segments) {
            integrate_ode_system(system_func, state,
                                 seg.t0, seg.t1, step_size,
                                 observer, method, false, abs_tol, rel_tol);
        }
    } else {
        integrate_ode_system(system_func, state, T_start, T_end, step_size,
                            observer, method, false, abs_tol, rel_tol);
    }

    // Reset drive
    drive_params.E0_1 = 0.0;
    drive_params.E0_2 = 0.0;
    
    return trajectory;
}

// ============================================================
// 2DCS / pump-probe optimisation helpers (Ingredient XV).
// Mirror of the Lattice / MixedLattice helpers; see lattice_md.cpp
// for the full prose.
// ============================================================

double PhononLattice::max_dSdt_norm_no_drive() const {
    // Build the current state (spins + phonons), evaluate ode_system
    // with the drive disabled, then return ‖dS/dt‖_∞ over the spin
    // subset only. The spin degrees of freedom are stored in the first
    // `lattice_size * spin_dim` slots of the flat state buffer; the
    // remaining `PhononState::N_DOF` entries are the zone-center E1
    // phonon DOFs (Q_x, Q_y, V_x, V_y), which we intentionally exclude
    // from the bound (see header doc for why).
    ODEState state = spins_to_state();
    ODEState dsdt(state_size, 0.0);

    PhononLattice* self = const_cast<PhononLattice*>(this);
    const double saved_E0_1 = self->drive_params.E0_1;
    const double saved_E0_2 = self->drive_params.E0_2;
    self->drive_params.E0_1 = 0.0;
    self->drive_params.E0_2 = 0.0;
    try {
        self->ode_system(state, dsdt, 0.0);
    } catch (...) {
        self->drive_params.E0_1 = saved_E0_1;
        self->drive_params.E0_2 = saved_E0_2;
        throw;
    }
    self->drive_params.E0_1 = saved_E0_1;
    self->drive_params.E0_2 = saved_E0_2;

    const size_t spin_count = lattice_size * spin_dim;
    double max_norm = 0.0;
    for (size_t i = 0; i < spin_count; ++i) {
        const double a = std::abs(dsdt[i]);
        if (a > max_norm) max_norm = a;
    }
    return max_norm;
}

PhononLattice::PumpProbeTrajectory PhononLattice::synthesize_M1_from_M0(
    const PumpProbeTrajectory& M_pulse_trajectory,
    const std::array<Eigen::Vector3d, 4>& M_ground,
    double tau, double T_step) const {
    PumpProbeTrajectory M1;
    M1.reserve(M_pulse_trajectory.size());
    if (M_pulse_trajectory.empty()) return M1;

    const double T_start_M0 = M_pulse_trajectory.front().first;
    const double tau_threshold = tau + T_start_M0;
    const ptrdiff_t n = static_cast<ptrdiff_t>(M_pulse_trajectory.size());

    for (const auto& [t_i, mag_i] : M_pulse_trajectory) {
        (void) mag_i;
        if (t_i < tau_threshold) {
            M1.push_back({t_i, M_ground});
        } else {
            const double rel = (t_i - tau - T_start_M0) / T_step;
            ptrdiff_t idx = static_cast<ptrdiff_t>(std::lround(rel));
            if (idx < 0) idx = 0;
            if (idx >= n) idx = n - 1;
            M1.push_back({t_i, M_pulse_trajectory[static_cast<size_t>(idx)].second});
        }
    }
    return M1;
}

void PhononLattice::pump_probe_spectroscopy(
    double polarization,
    double pulse_amp, double pulse_width, double pulse_freq,
    double tau_start, double tau_end, double tau_step,
    double T_start, double T_end, double T_step,
    const string& dir_name, const string& method,
    bool reuse_m0_for_m1,
    double stationarity_tol,
    int outer_omp_threads,
    bool pulse_window_chunking,
    double abs_tol, double rel_tol) {
    
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
    cout << "Optimisations: W1(reuse_m0_for_m1=" << (reuse_m0_for_m1 ? "on" : "off")
         << ", tol=" << stationarity_tol << "), "
         << "W2(outer_omp_threads=" << outer_omp_threads << "), "
         << "W3(pulse_window_chunking=" << (pulse_window_chunking ? "on" : "off") << ")" << endl;
    
    cout << "\n[1/3] Using current configuration as ground state..." << endl;
    double E_ground = energy_density();
    SpinVector M_ground = magnetization_local();
    SpinVector M_ground_global = magnetization_global();
    cout << "  Ground state: E/N = " << E_ground << ", |M| = " << M_ground.norm() << endl;
    cout << "  Global magnetization: " << M_ground_global.transpose() << endl;
    
    set_ordering_pattern();
    cout << "  Ordering pattern captured from ground state" << endl;
    
    save_positions(dir_name + "/positions.txt");
    save_spin_config(dir_name + "/spins_initial.txt");
    
    SpinConfig ground_state = spins;
    PhononState ground_phonons = phonons;
    
    cout << "\n[2/3] Running reference single-pulse dynamics (M0)..." << endl;
    auto M0_trajectory = single_pulse_drive(polarization, 0.0,
                                            pulse_amp, pulse_width, pulse_freq,
                                            T_start, T_end, T_step, method,
                                            pulse_window_chunking, abs_tol, rel_tol);
    
    spins = ground_state;
    phonons = ground_phonons;

    // ----- W1: capture ground-state magnetisation observables in the
    //       same layout the observer in single_pulse_drive uses:
    //       slot 0 = M_antiferro, 1 = M_local, 2 = M_global,
    //       3 = (O_custom, 0, 0).
    std::array<Eigen::Vector3d, 4> M_ground_arr = {
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()
    };
    {
        Eigen::Vector3d Ml = Eigen::Vector3d::Zero();
        Eigen::Vector3d Ma = Eigen::Vector3d::Zero();
        Eigen::Vector3d Mg = Eigen::Vector3d::Zero();
        double Oc = 0.0;
        for (size_t i = 0; i < lattice_size; ++i) {
            const size_t atom = i % N_atoms;
            const double sign = afm_sublattice_signs[atom];
            const Eigen::Vector3d S = spins[i];
            Ml += S;
            const Eigen::Vector3d S_global = sublattice_frames[atom] * S;
            Ma += sign * S_global;
            Mg += S_global;
            if (has_ordering_pattern) Oc += S.dot(ordering_pattern[i]);
        }
        Ml /= double(lattice_size);
        Ma /= double(lattice_size);
        Mg /= double(lattice_size);
        Oc /= double(lattice_size);
        M_ground_arr = { Ma, Ml, Mg, Eigen::Vector3d(Oc, 0.0, 0.0) };
    }

    bool can_reuse_m0 = false;
    if (reuse_m0_for_m1) {
        const double max_dS = max_dSdt_norm_no_drive();
        cout << "  [W1 guard] max |dS/dt|_inf at ground state = "
             << max_dS << " (tol = " << stationarity_tol << ")" << endl;
        if (max_dS <= stationarity_tol) {
            can_reuse_m0 = true;
            cout << "  [W1] Spin sector is stationary — synthesising M1(τ) from M0 by time-shift." << endl;
        } else {
            cout << "  [W1] Spin sector NOT stationary — falling back to fresh M1 integration each τ." << endl;
        }
    }

    int tau_steps = static_cast<int>(std::abs((tau_end - tau_start) / tau_step)) + 1;
    cout << "\n[3/3] Scanning delay times (" << tau_steps << " steps)..." << endl;
    
    vector<MagTrajectory> M1_trajectories(tau_steps);
    vector<MagTrajectory> M01_trajectories(tau_steps);
    vector<double> tau_values(tau_steps);
    for (int i = 0; i < tau_steps; ++i) tau_values[i] = tau_start + i * tau_step;

    // ----- W2: outer OpenMP parallelism over τ -----
    int n_outer = 1;
#ifdef _OPENMP
    if (outer_omp_threads <= 0) {
        n_outer = std::max(1, omp_get_max_threads());
    } else {
        n_outer = outer_omp_threads;
    }
    n_outer = std::min(n_outer, std::max(1, tau_steps));
    const int saved_max_active = omp_get_max_active_levels();
    omp_set_max_active_levels(1);
#else
    (void) outer_omp_threads;
#endif

    if (n_outer <= 1) {
        for (int i = 0; i < tau_steps; ++i) {
            const double current_tau = tau_values[i];
            cout << "\n--- Delay time " << (i+1) << "/" << tau_steps
                 << ": tau = " << current_tau << " ---" << endl;

            if (can_reuse_m0) {
                M1_trajectories[i] = synthesize_M1_from_M0(M0_trajectory, M_ground_arr, current_tau, T_step);
            } else {
                spins = ground_state;
                phonons = ground_phonons;
                cout << "  Computing M1 (probe at tau=" << current_tau << ")..." << endl;
                M1_trajectories[i] = single_pulse_drive(polarization, current_tau,
                                                       pulse_amp, pulse_width, pulse_freq,
                                                       T_start, T_end, T_step, method,
                                                       pulse_window_chunking, abs_tol, rel_tol);
            }

            spins = ground_state;
            phonons = ground_phonons;
            cout << "  Computing M01 (pump at 0 + probe at tau=" << current_tau << ")..." << endl;
            M01_trajectories[i] = double_pulse_drive(polarization, 0.0, polarization, current_tau,
                                                     pulse_amp, pulse_width, pulse_freq,
                                                     T_start, T_end, T_step, method,
                                                     pulse_window_chunking, abs_tol, rel_tol);
        }
    } else {
#ifdef _OPENMP
        cout << "  [W2] Distributing " << tau_steps << " τ points across "
             << n_outer << " OpenMP threads..." << endl;
        #pragma omp parallel num_threads(n_outer)
        {
            // PhononLattice has only value-typed members → the implicit
            // copy ctor is a full deep clone. Each thread owns its own
            // spins, phonons, drive_params, ordering_pattern, RNG, etc.
            PhononLattice local_lat(*this);
            local_lat.spins = ground_state;
            local_lat.phonons = ground_phonons;

            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < tau_steps; ++i) {
                const double current_tau = tau_values[i];
                if (can_reuse_m0) {
                    M1_trajectories[i] = synthesize_M1_from_M0(M0_trajectory, M_ground_arr, current_tau, T_step);
                } else {
                    local_lat.spins = ground_state;
                    local_lat.phonons = ground_phonons;
                    M1_trajectories[i] = local_lat.single_pulse_drive(
                        polarization, current_tau,
                        pulse_amp, pulse_width, pulse_freq,
                        T_start, T_end, T_step, method,
                        pulse_window_chunking, abs_tol, rel_tol);
                }

                local_lat.spins = ground_state;
                local_lat.phonons = ground_phonons;
                M01_trajectories[i] = local_lat.double_pulse_drive(
                    polarization, 0.0, polarization, current_tau,
                    pulse_amp, pulse_width, pulse_freq,
                    T_start, T_end, T_step, method,
                    pulse_window_chunking, abs_tol, rel_tol);
            }
        }
#endif
    }

#ifdef _OPENMP
    omp_set_max_active_levels(saved_max_active);
#endif

    
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
    const string& dir_name, const string& method,
    bool reuse_m0_for_m1,
    double stationarity_tol,
    bool pulse_window_chunking,
    double abs_tol, double rel_tol) {
    
    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    std::filesystem::create_directories(dir_name);
    
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
        cout << "Optimisations: W1(reuse_m0_for_m1=" << (reuse_m0_for_m1 ? "on" : "off")
             << ", tol=" << stationarity_tol << "), "
             << "W3(pulse_window_chunking=" << (pulse_window_chunking ? "on" : "off") << ")" << endl;
    }
    
    double E_ground = energy_density();
    if (rank == 0) {
        cout << "\n[1/4] Using current configuration as ground state..." << endl;
        cout << "  Ground state: E/N = " << E_ground << endl;
        save_positions(dir_name + "/positions.txt");
        save_spin_config(dir_name + "/spins_initial.txt");
    }
    
    // Set ordering pattern from current config on every rank so that
    // local M0 / M1 / M01 observers compute identical custom O.
    set_ordering_pattern();

    SpinConfig ground_state = spins;
    PhononState ground_phonons = phonons;

    // ----- W1: capture ground-state magnetisation observables on
    //       every rank (cheap and identical), then have rank 0 evaluate
    //       the stationarity guard and broadcast the verdict. -----
    std::array<Eigen::Vector3d, 4> M_ground_arr = {
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()
    };
    {
        Eigen::Vector3d Ml = Eigen::Vector3d::Zero();
        Eigen::Vector3d Ma = Eigen::Vector3d::Zero();
        Eigen::Vector3d Mg = Eigen::Vector3d::Zero();
        double Oc = 0.0;
        for (size_t i = 0; i < lattice_size; ++i) {
            const size_t atom = i % N_atoms;
            const double sign = afm_sublattice_signs[atom];
            const Eigen::Vector3d S = spins[i];
            Ml += S;
            const Eigen::Vector3d S_global = sublattice_frames[atom] * S;
            Ma += sign * S_global;
            Mg += S_global;
            if (has_ordering_pattern) Oc += S.dot(ordering_pattern[i]);
        }
        Ml /= double(lattice_size);
        Ma /= double(lattice_size);
        Mg /= double(lattice_size);
        Oc /= double(lattice_size);
        M_ground_arr = { Ma, Ml, Mg, Eigen::Vector3d(Oc, 0.0, 0.0) };
    }

    int can_reuse_m0 = 0;
    if (reuse_m0_for_m1) {
        if (rank == 0) {
            const double max_dS = max_dSdt_norm_no_drive();
            cout << "  [W1 guard] max |dS/dt|_inf at ground state = "
                 << max_dS << " (tol = " << stationarity_tol << ")" << endl;
            can_reuse_m0 = (max_dS <= stationarity_tol) ? 1 : 0;
        }
        MPI_Bcast(&can_reuse_m0, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (can_reuse_m0) {
                cout << "  [W1] Spin sector is stationary — M1(τ) will be synthesised from M0." << endl;
            } else {
                cout << "  [W1] Spin sector NOT stationary — every rank will integrate fresh M1." << endl;
            }
        }
    }
    
    MagTrajectory M0_trajectory;
    if (rank == 0) {
        cout << "\n[2/4] Computing reference trajectory (M0)..." << endl;
        M0_trajectory = single_pulse_drive(polarization, 0.0,
                                           pulse_amp, pulse_width, pulse_freq,
                                           T_start, T_end, T_step, method,
                                           pulse_window_chunking, abs_tol, rel_tol);
        spins = ground_state;
        phonons = ground_phonons;
    }

    // If we will synthesise M1 from M0 on remote ranks, broadcast M0.
    // The trajectory is a flat (n_times) × (1 time + 4·3 doubles) packed
    // buffer; cheap compared to one full integration.
    if (can_reuse_m0) {
        int n_times = (rank == 0) ? static_cast<int>(M0_trajectory.size()) : 0;
        MPI_Bcast(&n_times, 1, MPI_INT, 0, MPI_COMM_WORLD);
        const int stride = 1 + 4 * 3;  // t + 4 Vector3d
        vector<double> buf(static_cast<size_t>(n_times) * stride, 0.0);
        if (rank == 0) {
            for (int t = 0; t < n_times; ++t) {
                buf[t * stride + 0] = M0_trajectory[t].first;
                for (int k = 0; k < 4; ++k) {
                    for (int d = 0; d < 3; ++d) {
                        buf[t * stride + 1 + k * 3 + d] = M0_trajectory[t].second[k](d);
                    }
                }
            }
        }
        MPI_Bcast(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            M0_trajectory.assign(static_cast<size_t>(n_times),
                                 {0.0, {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
                                        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}});
            for (int t = 0; t < n_times; ++t) {
                M0_trajectory[t].first = buf[t * stride + 0];
                for (int k = 0; k < 4; ++k) {
                    M0_trajectory[t].second[k] = Eigen::Vector3d(
                        buf[t * stride + 1 + k * 3 + 0],
                        buf[t * stride + 1 + k * 3 + 1],
                        buf[t * stride + 1 + k * 3 + 2]);
                }
            }
        }
    }
    
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
    
    vector<double> my_tau_values(my_count);
    vector<MagTrajectory> my_M1_trajectories(my_count);
    vector<MagTrajectory> my_M01_trajectories(my_count);
    
    for (int i = 0; i < my_count; ++i) {
        int global_idx = my_start + i;
        double current_tau = tau_start + global_idx * tau_step;
        my_tau_values[i] = current_tau;
        
        cout << "[Rank " << rank << "] tau = " << current_tau 
             << " (" << (i+1) << "/" << my_count << ")" << endl;
        
        if (can_reuse_m0) {
            my_M1_trajectories[i] = synthesize_M1_from_M0(M0_trajectory, M_ground_arr,
                                                          current_tau, T_step);
        } else {
            spins = ground_state;
            phonons = ground_phonons;
            my_M1_trajectories[i] = single_pulse_drive(polarization, current_tau,
                                                       pulse_amp, pulse_width, pulse_freq,
                                                       T_start, T_end, T_step, method,
                                                       pulse_window_chunking, abs_tol, rel_tol);
        }
        
        spins = ground_state;
        phonons = ground_phonons;
        my_M01_trajectories[i] = double_pulse_drive(polarization, 0.0, polarization, current_tau,
                                                    pulse_amp, pulse_width, pulse_freq,
                                                    T_start, T_end, T_step, method,
                                                    pulse_window_chunking, abs_tol, rel_tol);
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

// ============================================================
// MC algorithms (greedy_quench, parallel_tempering, binning_analysis,
// estimate_autocorrelation_time, compute_thermodynamic_observables,
// generate_optimized_temperature_grid_mpi, generate_geometric_temperature_ladder,
// attempt_replica_exchange, gather_and_save_statistics_comprehensive,
// save_thermodynamic_observables_hdf5, save_heat_capacity_hdf5)
// are now provided by mc::* template functions in mc_common.h
// and called via inline wrappers in phonon_lattice.h.
// ============================================================

// Explicit template instantiation
template void PhononLattice::integrate_ode_system(
    std::function<void(const PhononLattice::ODEState&, PhononLattice::ODEState&, double)>,
    PhononLattice::ODEState&, double, double, double,
    std::function<void(const PhononLattice::ODEState&, double)>,
    const std::string&, bool, double, double);

