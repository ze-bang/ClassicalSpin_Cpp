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
#include <mpi.h>

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

namespace odeint = boost::numeric::odeint;

// ============================================================
// CONSTRUCTOR
// ============================================================

StrainPhononLattice::StrainPhononLattice(size_t d1, size_t d2, size_t d3, float spin_l)
    : dim1(d1), dim2(d2), dim3(d3), spin_length(spin_l)
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
    
    // Initialize Kitaev local frame (same for both sublattices)
    SpinMatrix R = MagnetoelasticParams::get_kitaev_rotation();
    sublattice_frames[0] = R;
    sublattice_frames[1] = R;
    
    // Initialize random number generator
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(static_cast<unsigned int>(seed));
    uniform_dist = std::uniform_real_distribution<double>(0.0, 1.0);
    
    cout << "StrainPhononLattice created with dimensions: "
         << dim1 << " x " << dim2 << " x " << dim3 << endl;
    cout << "Total spin sites: " << lattice_size << endl;
    cout << "Strain DOF: " << StrainState::N_DOF << " (ε_xx, ε_yy, ε_xy, V_xx, V_yy, V_xy) × 3 bonds" << endl;
    cout << "Total ODE state size: " << state_size << endl;
    
    // Build lattice
    build_honeycomb();
    
    // Initialize random spins
    init_random();
    
    cout << "StrainPhononLattice initialization complete!" << endl;
}

// ============================================================
// HONEYCOMB LATTICE CONSTRUCTION
// ============================================================

void StrainPhononLattice::build_honeycomb() {
    // Honeycomb lattice vectors (matching PhononLattice exactly)
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
    
    // Build NN connectivity (3 NN per site on honeycomb)
    // Matching PhononLattice bond assignment exactly:
    //   - z-bond (type 2): A(i,j,k) → B(i,j,k)     [same unit cell]
    //   - x-bond (type 0): A(i,j,k) → B(i,j-1,k)   [offset (0,-1,0)]
    //   - y-bond (type 1): A(i,j,k) → B(i+1,j-1,k) [offset (1,-1,0)]
    SpinMatrix Jx = magnetoelastic_params.get_Jx();
    SpinMatrix Jy = magnetoelastic_params.get_Jy();
    SpinMatrix Jz = magnetoelastic_params.get_Jz();
    
    for (size_t i = 0; i < dim1; ++i) {
        for (size_t j = 0; j < dim2; ++j) {
            for (size_t k = 0; k < dim3; ++k) {
                size_t site0 = flatten_index(i, j, k, 0);  // Sublattice A
                size_t site1 = flatten_index(i, j, k, 1);  // Sublattice B
                
                // x-bond (use Jx matrix)
                size_t partner_x = flatten_periodic(i, j-1, k, 1);
                nn_interaction[site0].push_back(Jx);
                nn_partners[site0].push_back(partner_x);
                nn_bond_types[site0].push_back(0);
                // Reverse bond (MUST use transpose for anisotropic exchange!)
                nn_interaction[partner_x].push_back(Jx.transpose());
                nn_partners[partner_x].push_back(site0);
                nn_bond_types[partner_x].push_back(0);
                
                // y-bond (use Jy matrix)
                size_t partner_y = flatten_periodic(i+1, j-1, k, 1);
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
                // 2nd NN offsets: (±1, 0), (0, ±1), (±1, ∓1) in lattice coordinates
                if (std::abs(magnetoelastic_params.J2_A) > 1e-12 || std::abs(magnetoelastic_params.J2_B) > 1e-12) {
                    SpinMatrix J2_A_mat = magnetoelastic_params.get_J2_A_matrix();
                    SpinMatrix J2_B_mat = magnetoelastic_params.get_J2_B_matrix();
                    
                    vector<std::tuple<int,int,int>> j2_offsets = {
                        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {1, -1, 0}, {-1, 1, 0}
                    };
                    
                    // 2nd NN for sublattice A (site0) with coupling J2_A
                    for (const auto& [di, dj, dk] : j2_offsets) {
                        size_t partner_j2 = flatten_periodic(i+di, j+dj, k+dk, 0);
                        if (partner_j2 > site0) {
                            j2_interaction[site0].push_back(J2_A_mat);
                            j2_partners[site0].push_back(partner_j2);
                            j2_interaction[partner_j2].push_back(J2_A_mat.transpose());
                            j2_partners[partner_j2].push_back(site0);
                        }
                    }
                    
                    // 2nd NN for sublattice B (site1) with coupling J2_B
                    for (const auto& [di, dj, dk] : j2_offsets) {
                        size_t partner_j2 = flatten_periodic(i+di, j+dj, k+dk, 1);
                        if (partner_j2 > site1) {
                            j2_interaction[site1].push_back(J2_B_mat);
                            j2_partners[site1].push_back(partner_j2);
                            j2_interaction[partner_j2].push_back(J2_B_mat.transpose());
                            j2_partners[partner_j2].push_back(site1);
                        }
                    }
                }
                
                // 3rd NN interactions (isotropic Heisenberg J3)
                // 3rd NN connect OPPOSITE sublattices (A↔B)
                if (std::abs(magnetoelastic_params.J3) > 1e-12) {
                    SpinMatrix J3_mat = magnetoelastic_params.get_J3_matrix();
                    
                    // 3rd NN from sublattice A (site0) to sublattice B
                    vector<std::tuple<int,int,int>> j3_A_to_B_offsets = {
                        {1, -2, 0}, {-1, 0, 0}, {1, 0, 0}
                    };
                    
                    for (const auto& [di, dj, dk] : j3_A_to_B_offsets) {
                        size_t partner_j3 = flatten_periodic(i+di, j+dj, k+dk, 1);
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
                        size_t partner_j3 = flatten_periodic(i+di, j+dj, k+dk, 0);
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
    
    // Build hexagonal plaquettes for ring exchange (if J7 is non-zero)
    if (std::abs(magnetoelastic_params.J7) > 1e-12) {
        // Matching PhononLattice hexagon definition (actual code, not comments):
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
    }
    
    cout << "Built honeycomb lattice with " << lattice_size << " sites" << endl;
    if (!hexagons.empty()) {
        cout << "Built " << hexagons.size() << " hexagonal plaquettes for ring exchange" << endl;
    }
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
    
    // Clear existing interactions (matching PhononLattice approach)
    for (size_t i = 0; i < lattice_size; ++i) {
        nn_interaction[i].clear();
        nn_partners[i].clear();
        nn_bond_types[i].clear();
        j2_interaction[i].clear();
        j2_partners[i].clear();
        j3_interaction[i].clear();
        j3_partners[i].clear();
        site_hexagons[i].clear();
    }
    hexagons.clear();
    
    // Build bond-dependent Kitaev-Heisenberg-Γ-Γ' exchange matrices
    SpinMatrix Jx = magnetoelastic_params.get_Jx();
    SpinMatrix Jy = magnetoelastic_params.get_Jy();
    SpinMatrix Jz = magnetoelastic_params.get_Jz();
    SpinMatrix J2_A_mat = magnetoelastic_params.get_J2_A_matrix();
    SpinMatrix J2_B_mat = magnetoelastic_params.get_J2_B_matrix();
    SpinMatrix J3_mat = magnetoelastic_params.get_J3_matrix();
    
    // Build NN interactions on honeycomb (matching PhononLattice exactly)
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
                size_t partner_x = flatten_periodic(i, j-1, k, 1);
                nn_interaction[site0].push_back(Jx);
                nn_partners[site0].push_back(partner_x);
                nn_bond_types[site0].push_back(0);
                // Reverse bond (MUST use transpose for anisotropic exchange!)
                nn_interaction[partner_x].push_back(Jx.transpose());
                nn_partners[partner_x].push_back(site0);
                nn_bond_types[partner_x].push_back(0);
                
                // y-bond (use Jy matrix)
                size_t partner_y = flatten_periodic(i+1, j-1, k, 1);
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
                // 2nd NN connect same sublattice (A-A, B-B)
                if (std::abs(magnetoelastic_params.J2_A) > 1e-12 || std::abs(magnetoelastic_params.J2_B) > 1e-12) {
                    vector<std::tuple<int,int,int>> j2_offsets = {
                        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {1, -1, 0}, {-1, 1, 0}
                    };
                    
                    // 2nd NN for sublattice A with J2_A
                    for (const auto& [di, dj, dk] : j2_offsets) {
                        size_t partner_j2 = flatten_periodic(i+di, j+dj, k+dk, 0);
                        if (partner_j2 > site0) {
                            j2_interaction[site0].push_back(J2_A_mat);
                            j2_partners[site0].push_back(partner_j2);
                            j2_interaction[partner_j2].push_back(J2_A_mat.transpose());
                            j2_partners[partner_j2].push_back(site0);
                        }
                    }
                    
                    // 2nd NN for sublattice B with J2_B
                    for (const auto& [di, dj, dk] : j2_offsets) {
                        size_t partner_j2 = flatten_periodic(i+di, j+dj, k+dk, 1);
                        if (partner_j2 > site1) {
                            j2_interaction[site1].push_back(J2_B_mat);
                            j2_partners[site1].push_back(partner_j2);
                            j2_interaction[partner_j2].push_back(J2_B_mat.transpose());
                            j2_partners[partner_j2].push_back(site1);
                        }
                    }
                }
                
                // 3rd NN interactions (A↔B across longer distance)
                if (std::abs(magnetoelastic_params.J3) > 1e-12) {
                    // 3rd NN from sublattice A (site0) to sublattice B
                    vector<std::tuple<int,int,int>> j3_A_to_B_offsets = {
                        {1, -2, 0}, {-1, 0, 0}, {1, 0, 0}
                    };
                    
                    for (const auto& [di, dj, dk] : j3_A_to_B_offsets) {
                        size_t partner_j3 = flatten_periodic(i+di, j+dj, k+dk, 1);
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
                        size_t partner_j3 = flatten_periodic(i+di, j+dj, k+dk, 0);
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
    
    double J7 = magnetoelastic_params.J7;
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
// SPIN BASIS FUNCTIONS FOR D3d IRREPS
// ============================================================

double StrainPhononLattice::f_K_A1g() const {
    // f_K^{A1g} = Σ_<ij>_γ S_i^γ S_j^γ
    // Sum of Kitaev terms over all NN bonds
    double f = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int bond_type = nn_bond_types[i][n];
                
                // Kitaev term: S_i^γ S_j^γ where γ = bond type
                f += Si(bond_type) * Sj(bond_type);
            }
        }
    }
    
    return f;
}

double StrainPhononLattice::f_J_A1g() const {
    // f_J^{A1g} = Σ_r (S^x_r S^x_{M_{y,z}} + S^y_r S^y_{M_{x,z}} + S^z_r S^z_{M_{x,y}})
    // where M_{y,z} means y and z bonds (not x), etc.
    // On each bond type γ, we sum the spin components α ≠ γ: S^α_i S^α_j
    double f = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int gamma = nn_bond_types[i][n];  // bond type (0=x, 1=y, 2=z)
                
                // Sum over components α ≠ γ
                int alpha = (gamma + 1) % 3;
                int beta = (gamma + 2) % 3;
                f += Si(alpha) * Sj(alpha) + Si(beta) * Sj(beta);
            }
        }
    }
    
    return f;
}

double StrainPhononLattice::f_Gamma_A1g() const {
    // f_Γ^{A1g} = Σ_<ij>_γ (S_i^α S_j^β + S_i^β S_j^α) where (α,β) ≠ γ
    double f = 0.0;
    
    for (size_t i = 0; i < lattice_size; ++i) {
        const Eigen::Vector3d& Si = spins[i];
        
        for (size_t n = 0; n < nn_partners[i].size(); ++n) {
            size_t j = nn_partners[i][n];
            if (j > i) {
                const Eigen::Vector3d& Sj = spins[j];
                int gamma = nn_bond_types[i][n];
                
                // Get α, β ≠ γ
                int alpha = (gamma + 1) % 3;
                int beta = (gamma + 2) % 3;
                
                f += Si(alpha) * Sj(beta) + Si(beta) * Sj(alpha);
            }
        }
    }
    
    return f;
}

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

// ============================================================
// MAGNETOELASTIC ENERGY
// ============================================================

double StrainPhononLattice::magnetoelastic_energy() const {
    // H_c = H_c^{A1g} + H_c^{Eg}
    //
    // H_c^{A1g} = λ_{A1g} Σ_b (ε_xx + ε_yy)_b {(J+K)f_K^{A1g} + J f_J^{A1g} + Γ f_Γ^{A1g}}
    //
    // H_c^{Eg} = λ_{Eg} Σ_b {(ε_xx - ε_yy)_b[(J+K)f_K^{Eg,1} + J f_J^{Eg,1} + Γ f_Γ^{Eg,1}]
    //                      + 2(ε_xy)_b[(J+K)f_K^{Eg,2} + J f_J^{Eg,2} + Γ f_Γ^{Eg,2}]}
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double lambda_A1g = magnetoelastic_params.lambda_A1g;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    // Compute spin basis functions
    double fK_A1g = f_K_A1g();
    double fJ_A1g = f_J_A1g();
    double fG_A1g = f_Gamma_A1g();
    
    double fK_Eg1 = f_K_Eg1();
    double fK_Eg2 = f_K_Eg2();
    double fJ_Eg1 = f_J_Eg1();
    double fJ_Eg2 = f_J_Eg2();
    double fG_Eg1 = f_Gamma_Eg1();
    double fG_Eg2 = f_Gamma_Eg2();
    
    // A1g channel contribution
    double A1g_spin_factor = (J + K) * fK_A1g + J * fJ_A1g + Gamma * fG_A1g;
    
    // Eg channel contribution
    double Eg1_spin_factor = (J + K) * fK_Eg1 + J * fJ_Eg1 + Gamma * fG_Eg1;
    double Eg2_spin_factor = (J + K) * fK_Eg2 + J * fJ_Eg2 + Gamma * fG_Eg2;
    
    double E_A1g = 0.0;
    double E_Eg = 0.0;
    
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        double exx = strain.epsilon_xx[b];
        double eyy = strain.epsilon_yy[b];
        double exy = strain.epsilon_xy[b];
        
        // A1g: (ε_xx + ε_yy)
        E_A1g += lambda_A1g * (exx + eyy) * A1g_spin_factor;
        
        // Eg: (ε_xx - ε_yy) * Eg1 + 2ε_xy * Eg2
        E_Eg += lambda_Eg * ((exx - eyy) * Eg1_spin_factor + 2.0 * exy * Eg2_spin_factor);
    }
    
    return E_A1g + E_Eg;
}

// ============================================================
// DERIVATIVES FOR STRAIN EOM
// ============================================================

double StrainPhononLattice::dH_deps_xx(size_t bond_type) const {
    // ∂H_c/∂ε_xx = λ_{A1g} * A1g_spin_factor + λ_{Eg} * Eg1_spin_factor
    //            + elastic terms: C11 * ε_xx + C12 * ε_yy
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double lambda_A1g = magnetoelastic_params.lambda_A1g;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    // Spin factors
    double A1g_spin_factor = (J + K) * f_K_A1g() + J * f_J_A1g() + Gamma * f_Gamma_A1g();
    double Eg1_spin_factor = (J + K) * f_K_Eg1() + J * f_J_Eg1() + Gamma * f_Gamma_Eg1();
    
    // Magnetoelastic contribution
    double dH_me = lambda_A1g * A1g_spin_factor + lambda_Eg * Eg1_spin_factor;
    
    return dH_me;
}

double StrainPhononLattice::dH_deps_yy(size_t bond_type) const {
    // ∂H_c/∂ε_yy = λ_{A1g} * A1g_spin_factor - λ_{Eg} * Eg1_spin_factor
    //            + elastic terms: C11 * ε_yy + C12 * ε_xx
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double lambda_A1g = magnetoelastic_params.lambda_A1g;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    double A1g_spin_factor = (J + K) * f_K_A1g() + J * f_J_A1g() + Gamma * f_Gamma_A1g();
    double Eg1_spin_factor = (J + K) * f_K_Eg1() + J * f_J_Eg1() + Gamma * f_Gamma_Eg1();
    
    // Note: ε_yy contributes +1 to A1g and -1 to Eg1
    double dH_me = lambda_A1g * A1g_spin_factor - lambda_Eg * Eg1_spin_factor;
    
    return dH_me;
}

double StrainPhononLattice::dH_deps_xy(size_t bond_type) const {
    // ∂H_c/∂ε_xy = 2 * λ_{Eg} * Eg2_spin_factor
    //            + elastic terms: 4 * C44 * ε_xy
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    double Eg2_spin_factor = (J + K) * f_K_Eg2() + J * f_J_Eg2() + Gamma * f_Gamma_Eg2();
    
    double dH_me = 2.0 * lambda_Eg * Eg2_spin_factor;
    
    return dH_me;
}

// ============================================================
// DERIVATIVE OF SPIN BASIS FUNCTIONS W.R.T. SPINS
// ============================================================

SpinVector StrainPhononLattice::df_K_A1g_dS(size_t site) const {
    // ∂f_K^{A1g}/∂S_site = Σ_n S_n^γ δ_γ,bond_type(site,n)
    // where n runs over neighbors of site
    SpinVector df = Eigen::Vector3d::Zero();
    
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        int gamma = nn_bond_types[site][n];
        df(gamma) += spins[j](gamma);
    }
    
    return df;
}

SpinVector StrainPhononLattice::df_J_A1g_dS(size_t site) const {
    // ∂f_J^{A1g}/∂S_site for f_J^{A1g} = Σ (S^α_i S^α_j) where α ≠ γ (bond type)
    // On a γ-bond connecting site to neighbor j:
    //   contribution to derivative is S_j^α in component α, and S_j^β in component β
    //   where α, β ≠ γ
    SpinVector df = Eigen::Vector3d::Zero();
    
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        int gamma = nn_bond_types[site][n];
        int alpha = (gamma + 1) % 3;
        int beta = (gamma + 2) % 3;
        
        // ∂/∂S^α_i (S^α_i S^α_j) = S^α_j
        // ∂/∂S^β_i (S^β_i S^β_j) = S^β_j
        df(alpha) += spins[j](alpha);
        df(beta) += spins[j](beta);
    }
    
    return df;
}

SpinVector StrainPhononLattice::df_Gamma_A1g_dS(size_t site) const {
    // ∂f_Γ^{A1g}/∂S_site for Γ term: S_i^α S_j^β + S_i^β S_j^α
    SpinVector df = Eigen::Vector3d::Zero();
    
    for (size_t n = 0; n < nn_partners[site].size(); ++n) {
        size_t j = nn_partners[site][n];
        int gamma = nn_bond_types[site][n];
        int alpha = (gamma + 1) % 3;
        int beta = (gamma + 2) % 3;
        
        // ∂/∂S_i^α (S_i^α S_j^β) = S_j^β
        // ∂/∂S_i^β (S_i^β S_j^α) = S_j^α
        df(alpha) += spins[j](beta);
        df(beta) += spins[j](alpha);
    }
    
    return df;
}

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

// ============================================================
// EFFECTIVE FIELD
// ============================================================

SpinVector StrainPhononLattice::get_magnetoelastic_field(size_t site) const {
    // H_me = -∂H_c/∂S_i
    // 
    // H_c = λ_{A1g} Σ_b (ε_xx + ε_yy)_b [(J+K)f_K^{A1g} + J f_J^{A1g} + Γ f_Γ^{A1g}]
    //     + λ_{Eg} Σ_b {(ε_xx - ε_yy)_b[(J+K)f_K^{Eg,1} + J f_J^{Eg,1} + Γ f_Γ^{Eg,1}]
    //                  + 2(ε_xy)_b[(J+K)f_K^{Eg,2} + J f_J^{Eg,2} + Γ f_Γ^{Eg,2}]}
    
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    double lambda_A1g = magnetoelastic_params.lambda_A1g;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    // Compute strain factors (summed over bond types)
    double A1g_strain = 0.0;
    double Eg1_strain = 0.0;
    double Eg2_strain = 0.0;
    
    for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
        A1g_strain += strain.epsilon_xx[b] + strain.epsilon_yy[b];
        Eg1_strain += strain.epsilon_xx[b] - strain.epsilon_yy[b];
        Eg2_strain += 2.0 * strain.epsilon_xy[b];
    }
    
    // Compute spin derivatives
    SpinVector df_K_A1g = df_K_A1g_dS(site);
    SpinVector df_J_A1g = df_J_A1g_dS(site);
    SpinVector df_G_A1g = df_Gamma_A1g_dS(site);
    
    SpinVector df_K_Eg1 = df_K_Eg1_dS(site);
    SpinVector df_J_Eg1 = df_J_Eg1_dS(site);
    SpinVector df_G_Eg1 = df_Gamma_Eg1_dS(site);
    
    SpinVector df_K_Eg2 = df_K_Eg2_dS(site);
    SpinVector df_J_Eg2 = df_J_Eg2_dS(site);
    SpinVector df_G_Eg2 = df_Gamma_Eg2_dS(site);
    
    // A1g contribution
    SpinVector H_A1g = lambda_A1g * A1g_strain * 
                       ((J + K) * df_K_A1g + J * df_J_A1g + Gamma * df_G_A1g);
    
    // Eg contribution
    SpinVector H_Eg1 = lambda_Eg * Eg1_strain *
                       ((J + K) * df_K_Eg1 + J * df_J_Eg1 + Gamma * df_G_Eg1);
    SpinVector H_Eg2 = lambda_Eg * Eg2_strain *
                       ((J + K) * df_K_Eg2 + J * df_J_Eg2 + Gamma * df_G_Eg2);
    
    // Total magnetoelastic field (negative because H_eff = -∂H/∂S)
    return -(H_A1g + H_Eg1 + H_Eg2);
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
    // H_7 = (J_7/6) Σ_{hex} Σ_{cyclic perms} [
    //   2*(S_i·S_j)*(S_k·S_l)*(S_m·S_n) - 6*(S_i·S_k)*(S_j·S_l)*(S_m·S_n)
    //   + 3*(S_i·S_l)*(S_j·S_k)*(S_m·S_n) + 3*(S_i·S_k)*(S_j·S_m)*(S_l·S_n)
    //   - (S_i·S_l)*(S_j·S_m)*(S_k·S_n)
    // ]
    
    Eigen::Vector3d H = Eigen::Vector3d::Zero();
    double J7 = magnetoelastic_params.J7;
    
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
        
        Eigen::Vector3d dH = Eigen::Vector3d::Zero();
        
        // Compute contributions from all 6 cyclic permutations
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
            int role = -1;
            if (pos == pi) role = 0;       // site is i
            else if (pos == pj) role = 1;  // site is j
            else if (pos == pk) role = 2;  // site is k
            else if (pos == pl) role = 3;  // site is l
            else if (pos == pm) role = 4;  // site is m
            else if (pos == pn) role = 5;  // site is n
            
            // Compute derivative based on which role our site plays
            // T1 = 2*(i·j)*(k·l)*(m·n), T2 = -6*(i·k)*(j·l)*(m·n)
            // T3 = 3*(i·l)*(j·k)*(m·n), T4 = 3*(i·k)*(j·m)*(l·n)
            // T5 = -(i·l)*(j·m)*(k·n)
            
            if (role == 0) {  // site is i
                dH += 2.0 * Sj_perm * dkl * dmn;
                dH += -6.0 * Sk * djl * dmn;
                dH += 3.0 * Sl * djk * dmn;
                dH += 3.0 * Sk * djm * dln;
                dH += -Sl * djm * dkn;
            }
            else if (role == 1) {  // site is j
                dH += 2.0 * Si * dkl * dmn;
                dH += -6.0 * Sl * dik * dmn;
                dH += 3.0 * Sk * dil * dmn;
                dH += 3.0 * Sm * dik * dln;
                dH += -Sm * dil * dkn;
            }
            else if (role == 2) {  // site is k
                dH += 2.0 * Sl * dij * dmn;
                dH += -6.0 * Si * djl * dmn;
                dH += 3.0 * Sj_perm * dil * dmn;
                dH += 3.0 * Si * djm * dln;
                dH += -Sn * dil * djm;
            }
            else if (role == 3) {  // site is l
                dH += 2.0 * Sk * dij * dmn;
                dH += -6.0 * Sj_perm * dik * dmn;
                dH += 3.0 * Si * djk * dmn;
                dH += 3.0 * Sn * dik * djm;
                dH += -Si * djm * dkn;
            }
            else if (role == 4) {  // site is m
                dH += 2.0 * Sn * dij * dkl;
                dH += -6.0 * Sn * dik * djl;
                dH += 3.0 * Sn * dil * djk;
                dH += 3.0 * Sj_perm * dik * dln;
                dH += -Sj_perm * dil * dkn;
            }
            else if (role == 5) {  // site is n
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
            Eigen::Vector3d M = total_magnetization();
            Eigen::Vector3d M_stag = staggered_magnetization();
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

void StrainPhononLattice::relax_strain(bool verbose) {
    // Relax strain to equilibrium given current spin configuration.
    // 
    // The total Hamiltonian (elastic + magnetoelastic) is:
    // H = (1/2)[C11(ε_xx² + ε_yy²) + 2C12 ε_xx ε_yy + 4C44 ε_xy²] 
    //   + λ_A1g (ε_xx + ε_yy) f_A1g + λ_Eg [(ε_xx - ε_yy) f_Eg1 + 2ε_xy f_Eg2]
    //
    // Setting ∂H/∂ε = 0:
    // C11 ε_xx + C12 ε_yy = -λ_A1g f_A1g - λ_Eg f_Eg1
    // C12 ε_xx + C11 ε_yy = -λ_A1g f_A1g + λ_Eg f_Eg1
    // 4C44 ε_xy = -2λ_Eg f_Eg2
    //
    // Solving for ε_xx, ε_yy:
    // From the first two equations (2x2 linear system):
    // [C11  C12] [ε_xx]   [-λ_A1g f_A1g - λ_Eg f_Eg1]
    // [C12  C11] [ε_yy] = [-λ_A1g f_A1g + λ_Eg f_Eg1]
    //
    // Using Cramer's rule with det = C11² - C12²:
    // ε_xx = [(-λ_A1g f_A1g - λ_Eg f_Eg1) C11 - (-λ_A1g f_A1g + λ_Eg f_Eg1) C12] / det
    //      = [-λ_A1g f_A1g (C11 - C12) - λ_Eg f_Eg1 (C11 + C12)] / det
    //
    // ε_yy = [C11 (-λ_A1g f_A1g + λ_Eg f_Eg1) - C12 (-λ_A1g f_A1g - λ_Eg f_Eg1)] / det
    //      = [-λ_A1g f_A1g (C11 - C12) + λ_Eg f_Eg1 (C11 + C12)] / det
    //
    // ε_xy = -λ_Eg f_Eg2 / (2C44)
    
    double C11 = elastic_params.C11;
    double C12 = elastic_params.C12;
    double C44 = elastic_params.C44;
    
    double lambda_A1g = magnetoelastic_params.lambda_A1g;
    double lambda_Eg = magnetoelastic_params.lambda_Eg;
    
    // Determinant of the 2x2 elastic matrix for ε_xx, ε_yy
    double det = C11 * C11 - C12 * C12;
    
    // Avoid division by zero
    if (std::abs(det) < 1e-12) {
        std::cerr << "Warning: Elastic matrix is singular, cannot relax strain analytically." << std::endl;
        return;
    }
    
    // Compute spin basis function factors
    double J = magnetoelastic_params.J;
    double K = magnetoelastic_params.K;
    double Gamma = magnetoelastic_params.Gamma;
    
    double A1g_spin_factor = (J + K) * f_K_A1g() + J * f_J_A1g() + Gamma * f_Gamma_A1g();
    double Eg1_spin_factor = (J + K) * f_K_Eg1() + J * f_J_Eg1() + Gamma * f_Gamma_Eg1();
    double Eg2_spin_factor = (J + K) * f_K_Eg2() + J * f_J_Eg2() + Gamma * f_Gamma_Eg2();
    
    // RHS of the linear system
    double b_xx = -lambda_A1g * A1g_spin_factor - lambda_Eg * Eg1_spin_factor;
    double b_yy = -lambda_A1g * A1g_spin_factor + lambda_Eg * Eg1_spin_factor;
    
    // Solve for equilibrium strain (same for all bond types in this approximation)
    double eps_xx_eq = (b_xx * C11 - b_yy * C12) / det;
    double eps_yy_eq = (-b_xx * C12 + b_yy * C11) / det;
    double eps_xy_eq = -lambda_Eg * Eg2_spin_factor / (2.0 * C44);
    
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

double StrainPhononLattice::mc_sweep(double temperature, bool gaussian_move, double sigma) {
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
        
        // Compute energy difference using full local field
        // H_eff includes exchange + magnetoelastic + ring exchange
        SpinVector old_spin = spins[site];
        SpinVector delta = new_spin - old_spin;
        
        // dE = H_spin(new) - H_spin(old) = -delta · H_eff
        // Note: get_local_field returns H_eff = -∂H/∂S, so dE = -delta · H_eff
        SpinVector H_eff = get_local_field(site);
        double dE = -delta.dot(H_eff);
        
        // Metropolis acceptance (branchless style)
        double rand_uniform = uniform_dist(rng);
        const bool accept = (dE < 0.0) || (rand_uniform < std::exp(-beta * dE));
        if (accept) {
            spins[site] = new_spin;
            ++accepted;
        }
    }
    
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
}

void StrainPhononLattice::anneal(double T_start, double T_end, 
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
    
    // Relax strain at the start to get initial magnetoelastic field
    relax_strain(false);
    
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
                    acc_sum += mc_sweep(T, gaussian_move, sigma);
                }
            } else {
                acc_sum += mc_sweep(T, gaussian_move, sigma);
            }
            
            // Periodically relax strain to maintain adiabatic equilibrium
            if ((s + 1) % strain_relax_interval == 0) {
                relax_strain(false);
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
    
    // Additional sweeps at T_end for equilibration
    cout << "Equilibrating at T=" << T_end << "..." << endl;
    double final_acc_sum = 0.0;
    for (size_t s = 0; s < n_sweeps; ++s) {
        final_acc_sum += mc_sweep(T_end, gaussian_move, sigma);
        relax_strain(false);
    }
    double final_acceptance = final_acc_sum / double(n_sweeps);
    double final_E = spin_energy() / lattice_size;
    cout << "Final T=" << T_end << ", E/N=" << final_E 
         << ", acc=" << std::fixed << std::setprecision(3) << final_acceptance << endl;
    
    // Final strain relaxation (verbose)
    relax_strain(true);
    
    // Save spin config after annealing (before deterministic sweeps)
    if (!out_dir.empty()) {
        save_spin_config(out_dir + "/spins_T=" + std::to_string(T_end) + ".txt");
    }
    
    // T=0 deterministic sweeps if requested
    if (T_zero && n_deterministics > 0) {
        cout << "\nPerforming " << n_deterministics << " deterministic sweeps at T=0..." << endl;
        for (size_t sweep = 0; sweep < n_deterministics; ++sweep) {
            deterministic_sweep(1);
            relax_strain(false);  // Relax strain after each deterministic sweep
            
            if (sweep % 100 == 0 || sweep == n_deterministics - 1) {
                double E = spin_energy() / lattice_size;
                cout << "Deterministic sweep " << sweep << "/" << n_deterministics 
                     << ", E/N=" << E << endl;
            }
        }
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
        save_strain_state(out_dir + "/strain.txt");
    }
    
    cout << "Annealing complete!" << endl;
}

void StrainPhononLattice::deterministic_sweep(size_t num_sweeps) {
    // Deterministic update: align each spin parallel to its local field.
    // This ensures S × H_eff = 0, eliminating precession in dynamics.
    // Uses random site selection within each sweep for better convergence.
    
    std::uniform_int_distribution<size_t> site_dist(0, lattice_size - 1);
    
    for (size_t sweep = 0; sweep < num_sweeps; ++sweep) {
        size_t count = 0;
        while (count < lattice_size) {
            size_t i = site_dist(rng);
            
            // Get local field (includes exchange, magnetoelastic, ring exchange)
            Eigen::Vector3d local_field = get_local_field(i);
            double norm = local_field.norm();
            
            if (norm < 1e-15) {
                // Field is essentially zero, skip this site
                count++;
                continue;
            }
            
            // Align spin PARALLEL to local field (minimizes energy)
            // H_eff = -∂H/∂Si, so Si should point along H_eff to minimize E = -Si·H_eff
            spins[i] = local_field / norm * spin_length;
            count++;
        }
        
        // Re-relax strain after each sweep to maintain adiabatic equilibrium
        relax_strain(false);
    }
    
    // Final strain relaxation
    relax_strain(true);
    
    cout << "Completed " << num_sweeps << " deterministic sweeps" << endl;
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

// ============================================================
// OBSERVABLES
// ============================================================

Eigen::Vector3d StrainPhononLattice::total_magnetization() const {
    Eigen::Vector3d M = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < lattice_size; ++i) {
        M += spins[i];
    }
    return M / lattice_size;
}

Eigen::Vector3d StrainPhononLattice::staggered_magnetization() const {
    Eigen::Vector3d M_stag = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < lattice_size; ++i) {
        // Sublattice alternation for honeycomb: sites 0,2,4,... are A, 1,3,5,... are B
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
        return total_magnetization().norm();
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
