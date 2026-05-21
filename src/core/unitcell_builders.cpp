/**
 * unitcell_builders.cpp - Unit cell builder functions
 * 
 * This file contains the implementations of the unit cell builder functions
 * that create specific lattice types (BCAO, Kitaev, Pyrochlore, TmFeO3).
 * These are separated from the main simulation code to reduce compile time.
 *
 * TmFeO3 organisation (see docs/tmfeo3_notes.tex for full theory):
 *
 *   * `build_tmfeo3`     - mixed Fe(SU2) + Tm(SU3) lattice (the production builder).
 *   * `build_tmfeo3_fe`  - Fe-only standalone lattice (Fe magnon physics, no Tm).
 *   * `build_tmfeo3_tm`  - Tm-only standalone lattice (Tm CEF dynamics, no Fe).
 *
 * All three share the same Fe and Tm sectors, so their bodies are thin
 * wrappers around three file-local helpers in the anonymous namespace below:
 *
 *   * `apply_tmfeo3_fe_sector(Fe_atoms, config)`               - all Fe-Fe
 *       bilinear/onsite/Zeeman terms in a single direct storage convention.
 *
 *   * `apply_tmfeo3_tm_sector(Tm_atoms, config)`              - Tm CEF
 *       splittings, Tm-Tm bilinears, sublattice CEF frames, sublattice-
 *       projected Tm Zeeman.  Tm always lives in its CEF local basis
 *       in a single direct storage convention.
 *
 *   * `apply_tmfeo3_fe_tm_couplings(mixed_uc, config)`         - the on-site
 *       mixed Fe-Fe-Tm trilinear W only. The Fe-Tm bonds are stored as 16
 *       explicit even/odd inversion pairs so the A2+ channels on the odd
 *       bonds are guaranteed to be related to their even partners by the
 *       required inversion sign flip.
 */

#include "classical_spin/core/unitcell_builders.h"

#include <array>
#include <cmath>
#include <string>

using namespace std;

UnitCell build_bcao_honeycomb(const SpinConfig& config) {
    const double J1xy = config.get_param("J1xy", -7.6);
    const double J1z = config.get_param("J1z", -1.2);
    const double D = config.get_param("D", 0.1);
    const double E = config.get_param("E", -0.1);
    const double F = config.get_param("F", 0.0);
    const double G = config.get_param("G", 0.0);
    const double J3xy = config.get_param("J3xy", 2.5);
    const double J3z = config.get_param("J3z", -0.85);
    
    // Use HoneyComb class from unitcell.h (already has lattice vectors and positions)
    HoneyComb atoms(3);
    
    // Build interaction matrices following molecular_dynamic_BCAO_emily.cpp pattern
    Eigen::Matrix3d J1z_mat;
    J1z_mat << J1xy + D, E, F,
               E, J1xy - D, G,
               F, G, J1z;
    
    // Rotation matrices for 120-degree rotations
    Eigen::Matrix3d U_2pi_3;
    double c = cos(2*M_PI/3);
    double s = sin(2*M_PI/3);
    U_2pi_3 << c, s, 0,
               -s, c, 0,
               0, 0, 1;
    
    Eigen::Matrix3d J1x_mat = U_2pi_3 * J1z_mat * U_2pi_3.transpose();
    Eigen::Matrix3d J1y_mat = U_2pi_3.transpose() * J1z_mat * U_2pi_3;
    
    Eigen::Matrix3d J3_mat = Eigen::Matrix3d::Zero();
    J3_mat(0, 0) = J3xy;
    J3_mat(1, 1) = J3xy;
    J3_mat(2, 2) = J3z;
    
    // Set nearest neighbor interactions
    atoms.set_bilinear_interaction(J1x_mat, 0, 1, Eigen::Vector3i(0, -1, 0));
    atoms.set_bilinear_interaction(J1y_mat, 0, 1, Eigen::Vector3i(1, -1, 0));
    atoms.set_bilinear_interaction(J1z_mat, 0, 1, Eigen::Vector3i(0, 0, 0));
    
    // Set third nearest neighbor interactions
    atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(1, 0, 0));
    atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(-1, 0, 0));
    atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(1, -2, 0));
    
    // Set magnetic field (with anisotropic g-factors from g_factor)
    Eigen::Vector3d field;
    field << config.g_factor[0] * config.field_strength * config.field_direction[0],
             config.g_factor[1] * config.field_strength * config.field_direction[1],
             config.g_factor[2] * config.field_strength * config.field_direction[2];
    
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    
    return atoms;
}

UnitCell build_kitaev_honeycomb(const SpinConfig& config) {
    const double K = config.get_param("K", -1.0);
    const double Gamma = config.get_param("Gamma", 0.25);
    const double Gammap = config.get_param("Gammap", -0.02);
    const double J = config.get_param("J", 0.0);
    
    // Extended interactions for KCTO-type models
    const double J2_A = config.get_param("J2_A", 0.0);
    const double J2_B = config.get_param("J2_B", 0.0);
    const double J3 = config.get_param("J3", 0.0);
    const double J_perp = config.get_param("J_perp", 0.0);
    
    // Single-ion anisotropy (easy-plane: Ka=Kb>0, Kc<0 or vice versa)
    const double Ka = config.get_param("Ka", 0.0);
    const double Kb = config.get_param("Kb", 0.0);
    const double Kc = config.get_param("Kc", 0.0);
    
    // Use HoneyComb class from unitcell.h
    HoneyComb atoms(3);
    
    // Kitaev interactions following molecular_dynamic_kitaev_honeycomb.cpp pattern
    Eigen::Matrix3d Jx;
    Jx << J + K, Gammap, Gammap,
          Gammap, J, Gamma,
          Gammap, Gamma, J;
    
    Eigen::Matrix3d Jy;
    Jy << J, Gammap, Gamma,
          Gammap, J + K, Gammap,
          Gamma, Gammap, J;
    
    Eigen::Matrix3d Jz;
    Jz << J, Gamma, Gammap,
          Gamma, J, Gammap,
          Gammap, Gammap, J + K;
    
    // Set nearest neighbor bonds (following exact pattern from legacy code)
    atoms.set_bilinear_interaction(Jx, 0, 1, Eigen::Vector3i(0, -1, 0));
    atoms.set_bilinear_interaction(Jy, 0, 1, Eigen::Vector3i(1, -1, 0));
    atoms.set_bilinear_interaction(Jz, 0, 1, Eigen::Vector3i(0, 0, 0));
    
    // 2nd NN interactions (isotropic Heisenberg, sublattice-dependent)
    if (std::abs(J2_A) > 1e-12 || std::abs(J2_B) > 1e-12) {
        Eigen::Matrix3d J2A_mat = J2_A * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d J2B_mat = J2_B * Eigen::Matrix3d::Identity();
        
        // A-sublattice 2nd NN
        atoms.set_bilinear_interaction(J2A_mat, 0, 0, Eigen::Vector3i(1, 0, 0));
        atoms.set_bilinear_interaction(J2A_mat, 0, 0, Eigen::Vector3i(0, 1, 0));
        atoms.set_bilinear_interaction(J2A_mat, 0, 0, Eigen::Vector3i(1, -1, 0));
        
        // B-sublattice 2nd NN
        atoms.set_bilinear_interaction(J2B_mat, 1, 1, Eigen::Vector3i(1, 0, 0));
        atoms.set_bilinear_interaction(J2B_mat, 1, 1, Eigen::Vector3i(0, 1, 0));
        atoms.set_bilinear_interaction(J2B_mat, 1, 1, Eigen::Vector3i(1, -1, 0));
    }
    
    // 3rd NN interactions (isotropic Heisenberg, A↔B)
    if (std::abs(J3) > 1e-12) {
        Eigen::Matrix3d J3_mat = J3 * Eigen::Matrix3d::Identity();
        atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(1, -2, 0));
        atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(-1, 0, 0));
        atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(1, 0, 0));
    }
    
    // Interlayer coupling (isotropic Heisenberg between same sublattice sites)
    if (std::abs(J_perp) > 1e-12) {
        Eigen::Matrix3d Jp_mat = J_perp * Eigen::Matrix3d::Identity();
        atoms.set_bilinear_interaction(Jp_mat, 0, 0, Eigen::Vector3i(0, 0, 1));
        atoms.set_bilinear_interaction(Jp_mat, 1, 1, Eigen::Vector3i(0, 0, 1));
    }
    
    // Single-ion anisotropy
    if (std::abs(Ka) > 1e-12 || std::abs(Kb) > 1e-12 || std::abs(Kc) > 1e-12) {
        Eigen::MatrixXd K_mat = Eigen::MatrixXd::Zero(3, 3);
        K_mat(0, 0) = Ka;
        K_mat(1, 1) = Kb;
        K_mat(2, 2) = Kc;
        atoms.set_onsite_interaction(K_mat, 0);
        atoms.set_onsite_interaction(K_mat, 1);
    }
    
    // Set Kitaev local frame for honeycomb sublattices
    // Transforms from local Kitaev basis to global cubic frame:
    // Local basis: x' = (1,1,-2)/√6, y' = (-1,1,0)/√2, z' = (1,1,1)/√3
    // S_global = R * S_local where columns of R are the local basis vectors
    Eigen::Matrix3d kitaev_frame;
    kitaev_frame << 1.0/std::sqrt(6.0), -1.0/std::sqrt(2.0), 1.0/std::sqrt(3.0),
                    1.0/std::sqrt(6.0),  1.0/std::sqrt(2.0), 1.0/std::sqrt(3.0),
                   -2.0/std::sqrt(6.0),  0.0,                1.0/std::sqrt(3.0);
    
    // Both honeycomb sublattices use the same local frame
    atoms.set_sublattice_frame(kitaev_frame, 0);
    atoms.set_sublattice_frame(kitaev_frame, 1);
    
    // Set magnetic field (with g-factor anisotropy)
    Eigen::Vector3d field;
    field << config.g_factor[0] * config.field_strength * config.field_direction[0],
             config.g_factor[1] * config.field_strength * config.field_direction[1],
             config.g_factor[2] * config.field_strength * config.field_direction[2];
    
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    
    return atoms;
}

UnitCell build_pyrochlore(const SpinConfig& config) {
    const double Jxx = config.get_param("Jxx", 1.0);
    const double Jyy = config.get_param("Jyy", 1.0);
    const double Jzz = config.get_param("Jzz", 1.0);
    const double gxx = config.get_param("gxx", 0.01);
    const double gyy = config.get_param("gyy", 4e-4);
    const double gzz = config.get_param("gzz", 1.0);
    const double theta = config.get_param("theta", 0.0);
    const double h = config.field_strength;
    
    // Use Pyrochlore class from unitcell.h (already has structure defined)
    Pyrochlore atoms(3);
    
    // Local axes for each sublattice (following molecular_dynamic_pyrochlore.cpp)
    Eigen::Vector3d z1(1, 1, 1);  z1 /= sqrt(3.0);
    Eigen::Vector3d z2(1, -1, -1); z2 /= sqrt(3.0);
    Eigen::Vector3d z3(-1, 1, -1); z3 /= sqrt(3.0);
    Eigen::Vector3d z4(-1, -1, 1); z4 /= sqrt(3.0);
    
    Eigen::Vector3d y1(0, 1, -1);  y1 /= sqrt(2.0);
    Eigen::Vector3d y2(0, -1, 1);  y2 /= sqrt(2.0);
    Eigen::Vector3d y3(0, -1, -1); y3 /= sqrt(2.0);
    Eigen::Vector3d y4(0, 1, 1);   y4 /= sqrt(2.0);
    
    Eigen::Vector3d x1(-2, 1, 1);  x1 /= sqrt(6.0);
    Eigen::Vector3d x2(-2, -1, -1); x2 /= sqrt(6.0);
    Eigen::Vector3d x3(2, 1, -1);  x3 /= sqrt(6.0);
    Eigen::Vector3d x4(2, -1, 1);  x4 /= sqrt(6.0);
    
    // Exchange matrix
    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    J(0, 0) = Jxx;
    J(1, 1) = Jyy;
    J(2, 2) = Jzz;
    
    // Set nearest neighbor interactions (following exact pattern from legacy code)
    atoms.set_bilinear_interaction(J, 0, 1, Eigen::Vector3i(0, 0, 0));
    atoms.set_bilinear_interaction(J, 0, 2, Eigen::Vector3i(0, 0, 0));
    atoms.set_bilinear_interaction(J, 0, 3, Eigen::Vector3i(0, 0, 0));
    atoms.set_bilinear_interaction(J, 1, 2, Eigen::Vector3i(0, 0, 0));
    atoms.set_bilinear_interaction(J, 1, 3, Eigen::Vector3i(0, 0, 0));
    atoms.set_bilinear_interaction(J, 2, 3, Eigen::Vector3i(0, 0, 0));
    
    // Inter-tetrahedron interactions
    atoms.set_bilinear_interaction(J, 0, 1, Eigen::Vector3i(1, 0, 0));
    atoms.set_bilinear_interaction(J, 0, 2, Eigen::Vector3i(0, 1, 0));
    atoms.set_bilinear_interaction(J, 0, 3, Eigen::Vector3i(0, 0, 1));
    atoms.set_bilinear_interaction(J, 1, 2, Eigen::Vector3i(-1, 1, 0));
    atoms.set_bilinear_interaction(J, 1, 3, Eigen::Vector3i(-1, 0, 1));
    atoms.set_bilinear_interaction(J, 2, 3, Eigen::Vector3i(0, 1, -1));
    
    // Build field vector
    Eigen::Vector3d field_global;
    field_global << config.field_direction[0] * h,
                    config.field_direction[1] * h,
                    config.field_direction[2] * h;
    
    // Rotated field components with theta rotation
    Eigen::Vector3d rot_field;
    rot_field << gzz * sin(theta) + gxx * cos(theta),
                 0,
                 gzz * cos(theta) - gxx * sin(theta);
    
    // Y-component fields with gyy factor
    Eigen::Vector3d By1, By2, By3, By4;
    By1 << 0, gyy * (pow(field_global.dot(y1), 3) - 3 * pow(field_global.dot(x1), 2) * field_global.dot(y1)), 0;
    By2 << 0, gyy * (pow(field_global.dot(y2), 3) - 3 * pow(field_global.dot(x2), 2) * field_global.dot(y2)), 0;
    By3 << 0, gyy * (pow(field_global.dot(y3), 3) - 3 * pow(field_global.dot(x3), 2) * field_global.dot(y3)), 0;
    By4 << 0, gyy * (pow(field_global.dot(y4), 3) - 3 * pow(field_global.dot(x4), 2) * field_global.dot(y4)), 0;
    
    // Set fields for each sublattice
    Eigen::Vector3d field1, field2, field3, field4;
    field1 = rot_field * field_global.dot(z1) + By1;
    field2 = rot_field * field_global.dot(z2) + By2;
    field3 = rot_field * field_global.dot(z3) + By3;
    field4 = rot_field * field_global.dot(z4) + By4;
    
    atoms.set_field(field1, 0);
    atoms.set_field(field2, 1);
    atoms.set_field(field3, 2);
    atoms.set_field(field4, 3);
    
    return atoms;
}

UnitCell build_pyrochlore_non_kramer(const SpinConfig& config) {
    // Non-Kramers pyrochlore with Jpm, Jzz, Jpmpm exchange
    // Following legacy/run_scripts/experiments.h: simulated_annealing_pyrochlore_non_kramer
    const double Jpm = config.get_param("Jpm", 0.0);
    const double Jzz = config.get_param("Jzz", 1.0);
    const double Jpmpm = config.get_param("Jpmpm", 0.0);
    const double J2 = config.get_param("J2", 0.0);  // Second nearest neighbor Heisenberg
    const double h = config.field_strength;
    
    // Non-Kramers field response parameters (delta1 and delta2)
    const double delta1 = config.get_param("delta1", 0.0);
    const double delta2 = config.get_param("delta2", 0.0);
    
    // Use Pyrochlore class from unitcell.h
    Pyrochlore atoms(3);
    
    // Local axes for each sublattice
    Eigen::Vector3d z1(1, 1, 1);   z1 /= sqrt(3.0);
    Eigen::Vector3d z2(1, -1, -1); z2 /= sqrt(3.0);
    Eigen::Vector3d z3(-1, 1, -1); z3 /= sqrt(3.0);
    Eigen::Vector3d z4(-1, -1, 1); z4 /= sqrt(3.0);
    
    Eigen::Vector3d y1(0, -1, 1);  y1 /= sqrt(2.0);
    Eigen::Vector3d y2(0, 1, -1);  y2 /= sqrt(2.0);
    Eigen::Vector3d y3(0, -1, -1); y3 /= sqrt(2.0);
    Eigen::Vector3d y4(0, 1, 1);   y4 /= sqrt(2.0);
    
    Eigen::Vector3d x1(-2, 1, 1);  x1 /= sqrt(6.0);
    Eigen::Vector3d x2(-2, -1, -1); x2 /= sqrt(6.0);
    Eigen::Vector3d x3(2, 1, -1);  x3 /= sqrt(6.0);
    Eigen::Vector3d x4(2, -1, 1);  x4 /= sqrt(6.0);
    
    // Exchange matrix as function of angle theta
    // J(theta) = [[-2*Jpm + 2*Jpmpm*cos(theta), -2*Jpmpm*sin(theta), 0],
    //             [-2*Jpmpm*sin(theta), -2*Jpm - 2*Jpmpm*cos(theta), 0],
    //             [0, 0, Jzz]]
    auto build_exchange = [&](double theta) -> Eigen::Matrix3d {
        Eigen::Matrix3d J;
        J << -2*Jpm + 2*Jpmpm*cos(theta), -2*Jpmpm*sin(theta), 0,
             -2*Jpmpm*sin(theta), -2*Jpm - 2*Jpmpm*cos(theta), 0,
             0, 0, Jzz;
        return J;
    };
    
    // Three types of bonds with different angles
    Eigen::Matrix3d Jx = build_exchange(2*M_PI/3);
    Eigen::Matrix3d Jy = build_exchange(4*M_PI/3);
    Eigen::Matrix3d Jz = build_exchange(0);
    
    // Intra-tetrahedron interactions (following legacy pattern)
    atoms.set_bilinear_interaction(Jz, 0, 1, Eigen::Vector3i(0, 0, 0));
    atoms.set_bilinear_interaction(Jx, 0, 2, Eigen::Vector3i(0, 0, 0));
    atoms.set_bilinear_interaction(Jy, 0, 3, Eigen::Vector3i(0, 0, 0));
    atoms.set_bilinear_interaction(Jy, 1, 2, Eigen::Vector3i(0, 0, 0));
    atoms.set_bilinear_interaction(Jx, 1, 3, Eigen::Vector3i(0, 0, 0));
    atoms.set_bilinear_interaction(Jz, 2, 3, Eigen::Vector3i(0, 0, 0));
    
    // Inter-tetrahedron interactions
    atoms.set_bilinear_interaction(Jz, 0, 1, Eigen::Vector3i(1, 0, 0));
    atoms.set_bilinear_interaction(Jx, 0, 2, Eigen::Vector3i(0, 1, 0));
    atoms.set_bilinear_interaction(Jy, 0, 3, Eigen::Vector3i(0, 0, 1));
    atoms.set_bilinear_interaction(Jy, 1, 2, Eigen::Vector3i(-1, 1, 0));
    atoms.set_bilinear_interaction(Jx, 1, 3, Eigen::Vector3i(-1, 0, 1));
    atoms.set_bilinear_interaction(Jz, 2, 3, Eigen::Vector3i(0, 1, -1));
    
    // Second nearest neighbor J2 Heisenberg interaction (same sublattice)
    Eigen::Matrix3d J2_mat = Eigen::Matrix3d::Identity() * J2;
    
    for (int s = 0; s < 4; ++s) {
        atoms.set_bilinear_interaction(J2_mat, s, s, Eigen::Vector3i(1, 0, 0));
        atoms.set_bilinear_interaction(J2_mat, s, s, Eigen::Vector3i(0, 1, 0));
        atoms.set_bilinear_interaction(J2_mat, s, s, Eigen::Vector3i(0, 0, 1));
    }
    
    // Build field vector
    Eigen::Vector3d field_global;
    field_global << config.field_direction[0] * h,
                    config.field_direction[1] * h,
                    config.field_direction[2] * h;
    
    // Compute local field components for each sublattice
    std::array<Eigen::Vector3d, 4> x_arr = {x1, x2, x3, x4};
    std::array<Eigen::Vector3d, 4> y_arr = {y1, y2, y3, y4};
    std::array<Eigen::Vector3d, 4> z_arr = {z1, z2, z3, z4};
    
    // Non-Kramers field response: 
    // field_local = {delta1*hx*hz + delta2*(hy^2 - hx^2), delta1*hy*hz + 2*delta2*hx*hy, hz}
    for (int i = 0; i < 4; ++i) {
        double hx = field_global.dot(x_arr[i]);
        double hy = field_global.dot(y_arr[i]);
        double hz = field_global.dot(z_arr[i]);
        
        Eigen::Vector3d field_local;
        field_local << delta1*hx*hz + delta2*(hy*hy - hx*hx),
                       delta1*hy*hz + 2*delta2*hx*hy,
                       hz;
        
        atoms.set_field(field_local, i);
    }
    
    return atoms;
}

// =============================================================================
// TmFeO3 helpers (file-local).
// Shared by build_tmfeo3, build_tmfeo3_fe, build_tmfeo3_tm.
// =============================================================================

namespace {

void warn_if_tmfeo3_legacy_frame_flags_ignored(const SpinConfig& config,
                                               const char* builder_name) {
    const bool has_use_global = config.has_param("use_global_frame");
    const bool has_use_local = config.has_param("use_local_frame");
    if (!has_use_global && !has_use_local) {
        return;
    }

    static bool warned = false;
    if (warned) {
        return;
    }
    warned = true;

    std::cerr
        << "[TmFeO3] Ignoring legacy frame-selection flag"
        << ((has_use_global && has_use_local) ? "s" : "")
        << " in " << builder_name << ": current tmfeo3 builders always store "
        << "Fe spins in lab Cartesian and Tm SU(3) states in the canonical "
        << "local CEF basis.\n";
}

std::array<Eigen::Matrix3d, 4> tmfeo3_tm_local_frames_xyz() {
    // Tm local dipole axes follow the Klein-four eta pattern:
    //   sublattice 0: R_0 = diag(+,+,+)
    //   sublattice 1: R_1 = diag(+,-,-)
    //   sublattice 2: R_2 = diag(-,+,-)
    //   sublattice 3: R_3 = diag(-,-,+)
    // For a local dipole vector J_loc = (J_x, J_y, J_z), the global dipole is
    // J_glob^(mu) = R_mu J_loc^(mu).
    std::array<Eigen::Matrix3d, 4> frames;
    frames[0] = Eigen::Vector3d(+1.0, +1.0, +1.0).asDiagonal();
    frames[1] = Eigen::Vector3d(+1.0, -1.0, -1.0).asDiagonal();
    frames[2] = Eigen::Vector3d(-1.0, +1.0, -1.0).asDiagonal();
    frames[3] = Eigen::Vector3d(-1.0, -1.0, +1.0).asDiagonal();
    return frames;
}

SpinMatrix tmfeo3_induced_su3_frame(const Eigen::Matrix3d& mu_act,
                                    const Eigen::Matrix3d& R_xyz) {
    // The SU(3) state is stored in the lambda basis, while the physical Tm
    // magnetic dipole lives in the projected local Cartesian basis:
    //   J_local = mu_act * lambda_active,
    // with lambda_active = (lambda_2, lambda_5, lambda_7).
    // The local->global induced action on the active SU(3) triplet is
    //   lambda_global = F_mu lambda_local,
    //   F_mu = mu_act^{-1} R_mu mu_act.
    // The coherence partners (lambda_1, lambda_4, lambda_6) must carry the
    // same 3x3 action so each |E_a><E_b| pair has one bulk character, while
    // the diagonal population channels (lambda_3, lambda_8) remain invariant.
    constexpr int active_im[3] = {1, 4, 6};
    constexpr int active_re[3] = {0, 3, 5};

    SpinMatrix frame = SpinMatrix::Identity(8, 8);
    Eigen::FullPivLU<Eigen::Matrix3d> lu(mu_act);
    if (lu.rank() != 3) {
        return frame;
    }

    const Eigen::Matrix3d active_frame = lu.solve(R_xyz * mu_act);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            frame(active_im[row], active_im[col]) = active_frame(row, col);
            frame(active_re[row], active_re[col]) = active_frame(row, col);
        }
    }
    return frame;
}



// -----------------------------------------------------------------------------
// Fe sector
// -----------------------------------------------------------------------------
//
// All Fe-Fe couplings are written and stored directly in lab Cartesian.
//
// DM convention (notes section "DM vector symmetry" + appendix bond table):
//   * z = 1/2 plane, Fe1 -> Fe0 in-plane bonds:  D = (0, +D_1, +D_2)
//   * z = 0   plane, Fe2 -> Fe3 in-plane bonds:  D = (0, -D_1, +D_2)
// (S_1 conjugates the exchange by R_z(pi), which flips D_y but preserves D_z.)
//
// Out-of-plane Fe0-Fe3 / Fe1-Fe2 c-axis bonds and all J_2 bonds carry pure
// isotropic exchange.
// Default Hamiltonian values: the locked Geom II + Geom III baseline
// (see tmfeo3_2dcs_geomII_geomIII/README.md).  Picked so that:
//   * the Gamma_2 (F_x, G_z) phase is the unique global minimum,
//   * single-ion |Kc - Ka| ~ 0.0034 meV gives qFM ~ 0.38 THz,
//   * D1 = 0.049 meV gives a weak-FM canting consistent with the
//     experimental F_x / G_z ratio,
//   * D2 = 0 forbids any C_y component (Gamma_2 purity).
void apply_tmfeo3_fe_sector(TmFeO3_Fe& Fe_atoms, const SpinConfig& config) {
    const double Jai  = config.get_param("J1ab",  4.74);
    const double Jbi  = Jai;
    const double Jci  = config.get_param("J1c",   5.15);
    const double J2ai = config.get_param("J2ab",  0.15);
    const double J2bi = J2ai;
    const double J2ci = config.get_param("J2c",   0.30);
    const double Ka   = config.get_param("Ka",   -0.0153);
    const double Kb   = config.get_param("Kb",    0.0);
    const double Kc   = config.get_param("Kc",   -0.0187);
    const double D1   = config.get_param("D1",    0.049);
    const double D2   = config.get_param("D2",    0.0);
    const double h    = config.field_strength;    
    // Lab-Cartesian DM exchange matrix builder:
    //   J_{ab} = Jiso * delta_{ab} + epsilon_{abc} D_c
    auto J_with_DM = [](double Jiso, double Dy, double Dz) {
        SpinMatrix J = SpinMatrix::Zero(3, 3);
        J(0, 0) = Jiso;  J(0, 1) = Dz;    J(0, 2) = -Dy;
        J(1, 0) = -Dz;   J(1, 1) = Jiso;
        J(2, 0) = Dy;                      J(2, 2) = Jiso;
        return J;
    };
    auto J_iso = [](double Jiso) {
        SpinMatrix J = SpinMatrix::Zero(3, 3);
        J(0, 0) = Jiso;
        J(1, 1) = Jiso;
        J(2, 2) = Jiso;
        return J;
    };

    const auto Ja_orig   = J_with_DM(Jai, +D1, +D2);  // z=1/2 plane
    const auto Jb_orig   = J_with_DM(Jbi, +D1, +D2);  // z=1/2 plane
    const auto Ja23_orig = J_with_DM(Jai, -D1, +D2);  // z=0   plane (D_y flipped)
    const auto Jb23_orig = J_with_DM(Jbi, -D1, +D2);  // z=0   plane
    const auto Jc_orig   = J_iso(Jci);
    const auto J2a_orig  = J_iso(J2ai);
    const auto J2b_orig  = J_iso(J2bi);
    const auto J2c_orig  = J_iso(J2ci);

    // In-plane J_1 bonds in z = 1/2 plane (Fe1 -> Fe0).
    Fe_atoms.set_bilinear_interaction(Ja_orig, 1, 0, Eigen::Vector3i(0,  0, 0));
    Fe_atoms.set_bilinear_interaction(Ja_orig, 1, 0, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(Jb_orig, 1, 0, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(Jb_orig, 1, 0, Eigen::Vector3i(1,  0, 0));

    // In-plane J_1 bonds in z = 0 plane (Fe2 -> Fe3); D_y flipped vs Fe1->Fe0.
    Fe_atoms.set_bilinear_interaction(Ja23_orig, 2, 3, Eigen::Vector3i(0,  0, 0));
    Fe_atoms.set_bilinear_interaction(Ja23_orig, 2, 3, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(Jb23_orig, 2, 3, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(Jb23_orig, 2, 3, Eigen::Vector3i(1,  0, 0));

    // Out-of-plane J_1c (Fe0 <-> Fe3 and Fe1 <-> Fe2 along c).
    Fe_atoms.set_bilinear_interaction(Jc_orig, 0, 3, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(Jc_orig, 0, 3, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(Jc_orig, 1, 2, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(Jc_orig, 1, 2, Eigen::Vector3i(0, 0, 1));

    // Intra-sublattice J_2 along a, b, c.
    for (int i = 0; i < 4; ++i) {
        Fe_atoms.set_bilinear_interaction(J2a_orig, i, i, Eigen::Vector3i(1, 0, 0));
        Fe_atoms.set_bilinear_interaction(J2b_orig, i, i, Eigen::Vector3i(0, 1, 0));
        Fe_atoms.set_bilinear_interaction(J2c_orig, i, i, Eigen::Vector3i(0, 0, 1));
    }

    // Cross-sublattice J_2c bonds: Fe0<->Fe2 and Fe1<->Fe3 (8 offsets each).
    {
        const Eigen::Vector3i offs02[] = {
            { 0, 0, 0}, { 0, 1, 0}, {-1, 0, 0}, {-1, 1, 0},
            { 0, 0, 1}, { 0, 1, 1}, {-1, 0, 1}, {-1, 1, 1}
        };
        for (const auto& off : offs02)
            Fe_atoms.set_bilinear_interaction(J2c_orig, 0, 2, off);

        const Eigen::Vector3i offs13[] = {
            { 0,  0, 0}, { 0, -1, 0}, { 1,  0, 0}, { 1, -1, 0},
            { 0,  0, 1}, { 0, -1, 1}, { 1,  0, 1}, { 1, -1, 1}
        };
        for (const auto& off : offs13)
            Fe_atoms.set_bilinear_interaction(J2c_orig, 1, 3, off);
    }

    // Single-ion anisotropy.
    Eigen::MatrixXd K_mat = Eigen::MatrixXd::Zero(3, 3);
    K_mat(0, 0) = Ka;
    K_mat(1, 1) = Kb;
    K_mat(2, 2) = Kc;
    for (int i = 0; i < 4; ++i) Fe_atoms.set_onsite_interaction(K_mat, i);

    // External Zeeman field in lab Cartesian.
        Eigen::Vector3d h_lab;
        h_lab << config.field_direction[0] * h,
                 config.field_direction[1] * h,
                 config.field_direction[2] * h;
        for (int i = 0; i < 4; ++i) {
        Fe_atoms.set_field(h_lab, i);
    }
}

// -----------------------------------------------------------------------------
// Tm sector
// -----------------------------------------------------------------------------
//
// Tm always lives in its CEF local basis.  Per notes Sec.~"Action on the
// projected qutrit subspace", R_mu acts at the 3-vector level on the
// projected dipoles J^a = mu_act^{a,b} <lambda_b>, not as a unitary on the
// qutrit Hilbert space, so the Gell-Mann basis is sublattice-independent.
//
// The Tm Zeeman field and the CEF-splitting field are added to `Tm_atoms.field`
// directly (in the local basis).
void apply_tmfeo3_tm_sector(TmFeO3_Tm& Tm_atoms, const SpinConfig& config) {
    // Locked Geom II + Geom III baseline (see tmfeo3_2dcs_geomII_geomIII/
    // README.md).  Code convention: per-site CEF Hamiltonian = diag(0, e1, e2),
    // so e1 = epsilon_2 and e2 = epsilon_3 directly (epsilon_1 = 0).
    //   e1 = 2.067834 meV  ->  omega_{12} = 0.500 THz   (E1 -> E2)
    //   e2 = 4.9628   meV  ->  omega_{13} = 1.200 THz   (E1 -> E3)
    //   omega_{23} = e2 - e1 = 2.8950 meV = 0.700 THz   (E2 -> E3)
    const double e1 = config.get_param("e1", 2.067834);
    const double e2 = config.get_param("e2", 4.9628);

    // Projected magnetic-moment matrix mu_act_{alpha,a}: J_alpha = mu_{alpha a} lambda_a
    // for the three time-odd Gell-Mann channels {lambda_2, lambda_5, lambda_7}.
    // Defaults from a TmFeO3 J=6 CEF calculation.
    const double mu_2x = config.get_param("mu_2x", 0.0);
    const double mu_2y = config.get_param("mu_2y", 0.0);
    const double mu_2z = config.get_param("mu_2z", 5.264);
    const double mu_5x = config.get_param("mu_5x",  2.3915);
    const double mu_5y = config.get_param("mu_5y", -2.7866);
    const double mu_5z = config.get_param("mu_5z",  0.0);
    const double mu_7x = config.get_param("mu_7x",  0.9128);
    const double mu_7y = config.get_param("mu_7y",  0.4655);
    const double mu_7z = config.get_param("mu_7z",  0.0);

    // g_Tm/g_Fe = (7/6)/2 = 7/12.  Scales the Tm Zeeman against the Fe
    // Zeeman so that both ions respond to the same physical lab field h.
    const double g_ratio_tm = config.get_param("g_ratio_tm", 7.0 / 12.0);

    const double tm_alpha_scale = config.get_param("tm_alpha_scale", 1.0);
    const double tm_beta_scale  = config.get_param("tm_beta_scale",  1.0);

    // Tm-Tm diagonal SU(3) bilinears (J_a * lambda_a x lambda_a).
    const double Jtm_1 = config.get_param("Jtm_1", 0.0);
    const double Jtm_2 = config.get_param("Jtm_2", 0.0);
    const double Jtm_3 = config.get_param("Jtm_3", 0.0);
    const double Jtm_4 = config.get_param("Jtm_4", 0.0);
    const double Jtm_5 = config.get_param("Jtm_5", 0.0);
    const double Jtm_6 = config.get_param("Jtm_6", 0.0);
    const double Jtm_7 = config.get_param("Jtm_7", 0.0);
    const double Jtm_8 = config.get_param("Jtm_8", 0.0);

    const double h = config.field_strength;

    // CEF energy splittings as a constant SU(3) field on every Tm site.
    // For diag(0, e1, e2) the (lambda_3, lambda_8) projections are
    //   B_3 = e1, B_8 = (2 e2 - e1) / sqrt(3).
    const double alpha = e1 * tm_alpha_scale;
    const double beta  = (2.0 * e2 - e1) / std::sqrt(3.0) * tm_beta_scale;
    Eigen::VectorXd tm_field(8);
    tm_field << 0, 0, alpha, 0, 0, 0, 0, beta;
    for (int i = 0; i < 4; ++i) Tm_atoms.set_field(tm_field, i);

    Eigen::Matrix3d mu_act;
    mu_act << mu_2x, mu_5x, mu_7x,
                mu_2y, mu_5y, mu_7y,
                mu_2z, mu_5z, mu_7z;
    const int active_im[3] = {1, 4, 6};   // lambda_2, lambda_5, lambda_7

    // Register the Tm sublattice frame on the SU(3) unit cell.  This is the
    // only extra ingredient needed for the existing mixed-lattice machinery to
    // do the right thing for dynamic readout/drive:
    //   * saved global SU(3) magnetization applies lambda_global = F_mu lambda_local
    //   * set_pulse_SU3() applies the covariant field transform field_local = F_mu^T field_global
    // The static onsite field below remains in the local qutrit basis by
    // construction; it is only used to set the local CEF splitting / Zeeman.
    const auto R_tm_xyz = tmfeo3_tm_local_frames_xyz();
    for (int sub = 0; sub < 4; ++sub) {
        Tm_atoms.set_sublattice_frame(tmfeo3_induced_su3_frame(mu_act, R_tm_xyz[sub]), sub);
    }

    if (h != 0.0 && config.field_direction.size() >= 3) {
        Eigen::Vector3d h_vec;
        h_vec << config.field_direction[0] * h,
                 config.field_direction[1] * h,
                 config.field_direction[2] * h;
        for (int sub = 0; sub < 4; ++sub) {
            for (int a = 0; a < 3; ++a) {
                double B_a = 0.0;
                for (int al = 0; al < 3; ++al) {
                    B_a += mu_act(al, a) * h_vec(al);
                }
                Tm_atoms.field[sub](active_im[a]) += g_ratio_tm * B_a;
            }
        }
    }

    // Tm-Tm diagonal bilinears.  Bond geometry follows from the 4c Wyckoff
    // positions; see notes section "Tm-Tm bond structure".
    const bool any_jtm = (Jtm_1 || Jtm_2 || Jtm_3 || Jtm_4
                       || Jtm_5 || Jtm_6 || Jtm_7 || Jtm_8);
    if (any_jtm) {
        Eigen::MatrixXd J_tm = Eigen::MatrixXd::Zero(8, 8);
        J_tm(0, 0) = Jtm_1; J_tm(1, 1) = Jtm_2; J_tm(2, 2) = Jtm_3;
        J_tm(3, 3) = Jtm_4; J_tm(4, 4) = Jtm_5; J_tm(5, 5) = Jtm_6;
        J_tm(6, 6) = Jtm_7; J_tm(7, 7) = Jtm_8;

        // In-plane: Tm0 <-> Tm2 in z = 0.75 plane, Tm1 <-> Tm3 in z = 0.25 plane.
        const Eigen::Vector3i in_plane_offs[] = {
            {0, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {-1, 1, 0}
        };
        for (const auto& off : in_plane_offs) {
            Tm_atoms.set_bilinear_interaction(J_tm, 0, 2, off);
            Tm_atoms.set_bilinear_interaction(J_tm, 1, 3, off);
        }

        // Out-of-plane c-axis NN (2 bonds per pair).
        Tm_atoms.set_bilinear_interaction(J_tm, 0, 3, Eigen::Vector3i(-1, 1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm, 0, 3, Eigen::Vector3i(-1, 1, 1));
        Tm_atoms.set_bilinear_interaction(J_tm, 2, 1, Eigen::Vector3i( 0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm, 2, 1, Eigen::Vector3i( 0, 0, 1));
    }
}

// -----------------------------------------------------------------------------
// Fe-Tm couplings (only used by build_tmfeo3, not by the standalone builders)
// -----------------------------------------------------------------------------

// 16 Fe-Tm inversion-related bond pairs = 4 orbits * 4 pairs/orbit.
// For each pair, the odd bond is the inversion-related partner of the even
// bond. The on-site W A2+ channels (lambda_4, lambda_6) therefore enter with
// opposite signs on the two bonds, while the A1+ channels (lambda_1, lambda_3,
// lambda_8) are identical.
struct FeTmBond {
    int fe;               // Fe sublattice index
    int tm;               // Tm sublattice index
    Eigen::Vector3i off;  // cell offset (Tm cell - Fe cell)
};

struct FeTmBondPair {
    int orbit;                 // 1..4 (selects W_orbit scale)
    FeTmBond even;             // E/S1/S2/S1S2 coset representative
    FeTmBond odd;              // inversion-related partner in I/S1I/S2I/S1S2I coset
};

const std::array<FeTmBondPair, 16>& fe_tm_w_bond_pairs() {
    static const std::array<FeTmBondPair, 16> kPairs = {{
        // Fe0 (z = 1/2)
        {1, {0, 3, { -1,  0,  0}}, {0, 0, {  0,  0,  0}}},
        {2, {0, 2, {  0,  0,  0}}, {0, 1, { -1,  0,  0}}},
        {3, {0, 1, {  0,  0,  0}}, {0, 2, { -1,  0,  0}}},
        {4, {0, 0, {  0, -1,  0}}, {0, 3, { -1,  1,  0}}},
        // Fe1 (z = 1/2)
        {1, {1, 2, {  0,  0,  0}}, {1, 1, {  0, -1,  0}}},
        {2, {1, 3, {  0,  0,  0}}, {1, 0, {  0, -1,  0}}},
        {3, {1, 0, {  1, -1,  0}}, {1, 3, { -1,  0,  0}}},
        {4, {1, 1, {  0,  0,  0}}, {1, 2, {  0, -1,  0}}},
        // Fe2 (z = 0)
        {1, {2, 1, {  0, -1,  0}}, {2, 2, {  0,  0, -1}}},
        {2, {2, 0, {  0, -1, -1}}, {2, 3, {  0,  0,  0}}},
        {3, {2, 3, { -1,  0,  0}}, {2, 0, {  1, -1, -1}}},
        {4, {2, 2, {  0, -1, -1}}, {2, 1, {  0,  0,  0}}},
        // Fe3 (z = 0)
        {1, {3, 0, {  0,  0, -1}}, {3, 3, { -1,  0,  0}}},
        {2, {3, 1, { -1,  0,  0}}, {3, 2, {  0,  0, -1}}},
        {3, {3, 2, { -1,  0, -1}}, {3, 1, {  0,  0,  0}}},
        {4, {3, 3, { -1,  1,  0}}, {3, 0, {  0, -1, -1}}}
    }};
    return kPairs;
}

void apply_tmfeo3_fe_tm_couplings(MixedUnitCell& mixed_uc, const SpinConfig& config) {
    struct TrilinearChannel {
        double xx, yy, zz, xy, xz, yz;
    };

    // Legacy shorthand used in tracked TmFeO3 configs:
    //   u_{1,3,8} -> W_{1,3,8}^{zz}=+u, W_{1,3,8}^{xx}=-u
    //   v_{4,6}   -> W_{4,6}^{xz}=+v
    const double u1 = config.get_param("u1", 0.0);
    const double u3 = config.get_param("u3", 0.0);
    const double u8 = config.get_param("u8", 0.0);
    const double v4 = config.get_param("v4", 0.0);
    const double v6 = config.get_param("v6", 0.0);

    auto read_tri_ch = [&](const std::string& pfx,
                           double zz_minus_xx = 0.0,
                           double xz = 0.0) -> TrilinearChannel {
        return {
            config.get_param(pfx + "_xx", 0.0) - zz_minus_xx,
            config.get_param(pfx + "_yy", 0.0),
            config.get_param(pfx + "_zz", 0.0) + zz_minus_xx,
            config.get_param(pfx + "_xy", 0.0),
            config.get_param(pfx + "_xz", 0.0) + xz,
            config.get_param(pfx + "_yz", 0.0)
        };
    };
    auto tri_ch_nonzero = [](const TrilinearChannel& channel) {
        return channel.xx != 0.0 || channel.yy != 0.0 || channel.zz != 0.0
            || channel.xy != 0.0 || channel.xz != 0.0 || channel.yz != 0.0;
    };

    const TrilinearChannel W1_ch = read_tri_ch("W1", u1, 0.0);
    const TrilinearChannel W3_ch = read_tri_ch("W3", u3, 0.0);
    const TrilinearChannel W4_ch = read_tri_ch("W4", 0.0, v4);
    const TrilinearChannel W6_ch = read_tri_ch("W6", 0.0, v6);
    const TrilinearChannel W8_ch = read_tri_ch("W8", u8, 0.0);

    const double W_orbit_scale[4] = {
        config.get_param("W_orbit1_scale", 1.0),
        config.get_param("W_orbit2_scale", 1.0),
        config.get_param("W_orbit3_scale", 1.0),
        config.get_param("W_orbit4_scale", 1.0)
    };

    auto fill_channel = [](SpinTensor3& tensor, int lambda_index,
                           const TrilinearChannel& channel, double sign) {
        tensor[0](0, lambda_index) = sign * channel.xx;
        tensor[1](1, lambda_index) = sign * channel.yy;
        tensor[2](2, lambda_index) = sign * channel.zz;
        tensor[0](1, lambda_index) = sign * channel.xy;
        tensor[1](0, lambda_index) = sign * channel.xy;
        tensor[0](2, lambda_index) = sign * channel.xz;
        tensor[2](0, lambda_index) = sign * channel.xz;
        tensor[1](2, lambda_index) = sign * channel.yz;
        tensor[2](1, lambda_index) = sign * channel.yz;
    };
    auto scale_tensor3 = [](const SpinTensor3& tensor, double scale) {
        SpinTensor3 out(tensor.size());
        for (size_t i = 0; i < tensor.size(); ++i) {
            out[i] = scale * tensor[i];
        }
        return out;
    };

    const auto& bond_pairs = fe_tm_w_bond_pairs();

    // On-site Fe-Fe-Tm term:
    //   H_W = sum_{(i,mu)} lambda_mu^(n) W_n^{ab} S_i^a S_i^b.
    // Each odd bond is inserted only through its explicit inversion-related
    // pair so the mirror-odd A2+ channels (lambda_4, lambda_6) are guaranteed
    // to carry the opposite sign relative to the even bond.
    const bool any_W = tri_ch_nonzero(W1_ch) || tri_ch_nonzero(W3_ch)
                    || tri_ch_nonzero(W4_ch) || tri_ch_nonzero(W6_ch)
                    || tri_ch_nonzero(W8_ch);
    if (any_W) {
        auto build_W = [&](double sign_A2) {
            SpinTensor3 tensor(3);
            for (auto& block : tensor) {
                block = Eigen::MatrixXd::Zero(3, 8);
            }
            fill_channel(tensor, 0, W1_ch, 1.0);
            fill_channel(tensor, 2, W3_ch, 1.0);
            fill_channel(tensor, 7, W8_ch, 1.0);
            fill_channel(tensor, 3, W4_ch, sign_A2);
            fill_channel(tensor, 5, W6_ch, sign_A2);
            return tensor;
        };
        const SpinTensor3 W_even_base = build_W(+1.0);
        const SpinTensor3 W_odd_base = build_W(-1.0);
        const Eigen::Vector3i zero = Eigen::Vector3i::Zero();
        for (const auto& pair : bond_pairs) {
            const double orbit_scale = W_orbit_scale[pair.orbit - 1];
            mixed_uc.set_mixed_trilinear(
                scale_tensor3(W_even_base, orbit_scale),
                pair.even.fe, pair.even.fe, pair.even.tm, zero, pair.even.off);
            mixed_uc.set_mixed_trilinear(
                scale_tensor3(W_odd_base, orbit_scale),
                pair.odd.fe, pair.odd.fe, pair.odd.tm, zero, pair.odd.off);
        }
    }

}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// TmFeO3 builder entry points (thin wrappers around the helpers above).
// -----------------------------------------------------------------------------

MixedUnitCell build_tmfeo3(const SpinConfig& config) {
    warn_if_tmfeo3_legacy_frame_flags_ignored(config, "build_tmfeo3()");

    TmFeO3_Fe Fe_atoms(3);
    TmFeO3_Tm Tm_atoms(8);

    apply_tmfeo3_fe_sector(Fe_atoms, config);
    apply_tmfeo3_tm_sector(Tm_atoms, config);

    MixedUnitCell mixed_uc(Fe_atoms, Tm_atoms);
    apply_tmfeo3_fe_tm_couplings(mixed_uc, config);

    return mixed_uc;
}

UnitCell build_tmfeo3_fe(const SpinConfig& config) {
    warn_if_tmfeo3_legacy_frame_flags_ignored(config, "build_tmfeo3_fe()");

    TmFeO3_Fe Fe_atoms(3);
    apply_tmfeo3_fe_sector(Fe_atoms, config);
    return Fe_atoms;
}

UnitCell build_tmfeo3_tm(const SpinConfig& config) {
    warn_if_tmfeo3_legacy_frame_flags_ignored(config, "build_tmfeo3_tm()");

    TmFeO3_Tm Tm_atoms(8);
    apply_tmfeo3_tm_sector(Tm_atoms, config);
    return Tm_atoms;
}


// ============================================================================
// PHONON HONEYCOMB BUILDER
// ============================================================================

UnitCell build_phonon_honeycomb(const SpinConfig& config) {
    // Kitaev-Heisenberg-Γ-Γ' parameters
    const double K = config.get_param("K", -9.0);
    const double Gamma = config.get_param("Gamma", 1.8);
    const double Gammap = config.get_param("Gammap", 0.3);
    const double J = config.get_param("J", -0.1);
    
    // 2nd NN (sublattice-dependent isotropic Heisenberg)
    const double J2_A = config.get_param("J2_A", 0.3);
    const double J2_B = config.get_param("J2_B", 0.3);
    
    // 3rd NN (isotropic Heisenberg)
    const double J3 = config.get_param("J3", 0.9);
    
    // Use HoneyComb class from unitcell.h
    HoneyComb atoms(3);
    
    // Bond-dependent Kitaev-Heisenberg-Γ-Γ' exchange matrices in LOCAL Kitaev frame
    Eigen::Matrix3d Jx;
    Jx << J + K, Gammap, Gammap,
          Gammap, J, Gamma,
          Gammap, Gamma, J;
    
    Eigen::Matrix3d Jy;
    Jy << J, Gammap, Gamma,
          Gammap, J + K, Gammap,
          Gamma, Gammap, J;
    
    Eigen::Matrix3d Jz;
    Jz << J, Gamma, Gammap,
          Gamma, J, Gammap,
          Gammap, Gammap, J + K;
    
    // Transform to global Cartesian frame: J_global = R * J_local * R^T
    Eigen::Matrix3d R;
    R << 1.0/std::sqrt(6.0), -1.0/std::sqrt(2.0), 1.0/std::sqrt(3.0),
         1.0/std::sqrt(6.0),  1.0/std::sqrt(2.0), 1.0/std::sqrt(3.0),
        -2.0/std::sqrt(6.0),  0.0,                1.0/std::sqrt(3.0);
    
    Eigen::Matrix3d Jx_global = R * Jx * R.transpose();
    Eigen::Matrix3d Jy_global = R * Jy * R.transpose();
    Eigen::Matrix3d Jz_global = R * Jz * R.transpose();
    
    // Set NN bonds with bond_type metadata (x=0, y=1, z=2)
    // x-bond: A(i,j) -> B(i, j-1)
    atoms.set_bilinear_interaction(Jx_global, 0, 1, Eigen::Vector3i(0, -1, 0), 0);
    // y-bond: A(i,j) -> B(i+1, j-1)
    atoms.set_bilinear_interaction(Jy_global, 0, 1, Eigen::Vector3i(1, -1, 0), 1);
    // z-bond: A(i,j) -> B(i, j) (same unit cell)
    atoms.set_bilinear_interaction(Jz_global, 0, 1, Eigen::Vector3i(0, 0, 0), 2);
    
    // 2nd NN interactions (isotropic Heisenberg, sublattice-dependent)
    if (std::abs(J2_A) > 1e-12 || std::abs(J2_B) > 1e-12) {
        Eigen::Matrix3d J2A_mat = J2_A * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d J2B_mat = J2_B * Eigen::Matrix3d::Identity();
        
        // A-sublattice 2nd NN (bond_type = -1, no phonon coupling)
        atoms.set_bilinear_interaction(J2A_mat, 0, 0, Eigen::Vector3i(1, 0, 0));
        atoms.set_bilinear_interaction(J2A_mat, 0, 0, Eigen::Vector3i(0, 1, 0));
        atoms.set_bilinear_interaction(J2A_mat, 0, 0, Eigen::Vector3i(1, -1, 0));
        
        // B-sublattice 2nd NN
        atoms.set_bilinear_interaction(J2B_mat, 1, 1, Eigen::Vector3i(1, 0, 0));
        atoms.set_bilinear_interaction(J2B_mat, 1, 1, Eigen::Vector3i(0, 1, 0));
        atoms.set_bilinear_interaction(J2B_mat, 1, 1, Eigen::Vector3i(1, -1, 0));
    }
    
    // 3rd NN interactions (isotropic Heisenberg, A↔B)
    if (std::abs(J3) > 1e-12) {
        Eigen::Matrix3d J3_mat = J3 * Eigen::Matrix3d::Identity();
        
        // 3rd NN from A to B
        atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(1, -2, 0));
        atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(-1, 0, 0));
        atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(1, 0, 0));
    }
    
    // Set Kitaev local frame for both sublattices
    atoms.set_sublattice_frame(R, 0);
    atoms.set_sublattice_frame(R, 1);
    
    // Set magnetic field
    Eigen::Vector3d field;
    field << config.field_strength * config.field_direction[0],
             config.field_strength * config.field_direction[1],
             config.field_strength * config.field_direction[2];
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    
    return atoms;
}


// ============================================================================
// STRAIN HONEYCOMB BUILDER
// ============================================================================

UnitCell build_strain_honeycomb(const SpinConfig& config) {
    // Kitaev-Heisenberg-Γ-Γ' parameters (same as phonon but used in local frame)
    const double K = config.get_param("K", -9.0);
    const double Gamma = config.get_param("Gamma", 1.8);
    const double Gammap = config.get_param("Gammap", 0.3);
    const double J = config.get_param("J", -0.1);
    
    // 2nd NN (sublattice-dependent isotropic Heisenberg)
    const double J2_A = config.get_param("J2_A", 0.3);
    const double J2_B = config.get_param("J2_B", 0.3);
    
    // 3rd NN (isotropic Heisenberg)
    const double J3 = config.get_param("J3", 0.9);
    
    // Use HoneyComb class from unitcell.h
    HoneyComb atoms(3);
    
    // Bond-dependent exchange matrices in LOCAL Kitaev frame
    // NOTE: StrainPhononLattice works in LOCAL frame (unlike PhononLattice which transforms to global)
    Eigen::Matrix3d Jx;
    Jx << J + K, Gammap, Gammap,
          Gammap, J, Gamma,
          Gammap, Gamma, J;
    
    Eigen::Matrix3d Jy;
    Jy << J, Gammap, Gamma,
          Gammap, J + K, Gammap,
          Gamma, Gammap, J;
    
    Eigen::Matrix3d Jz;
    Jz << J, Gamma, Gammap,
          Gamma, J, Gammap,
          Gammap, Gammap, J + K;
    
    // Set NN bonds with bond_type metadata (x=0, y=1, z=2)
    // Stored in LOCAL Kitaev frame (StrainPhononLattice uses local frame)
    atoms.set_bilinear_interaction(Jx, 0, 1, Eigen::Vector3i(0, -1, 0), 0);
    atoms.set_bilinear_interaction(Jy, 0, 1, Eigen::Vector3i(1, -1, 0), 1);
    atoms.set_bilinear_interaction(Jz, 0, 1, Eigen::Vector3i(0, 0, 0), 2);
    
    // 2nd NN interactions
    if (std::abs(J2_A) > 1e-12 || std::abs(J2_B) > 1e-12) {
        Eigen::Matrix3d J2A_mat = J2_A * Eigen::Matrix3d::Identity();
        Eigen::Matrix3d J2B_mat = J2_B * Eigen::Matrix3d::Identity();
        
        atoms.set_bilinear_interaction(J2A_mat, 0, 0, Eigen::Vector3i(1, 0, 0));
        atoms.set_bilinear_interaction(J2A_mat, 0, 0, Eigen::Vector3i(0, 1, 0));
        atoms.set_bilinear_interaction(J2A_mat, 0, 0, Eigen::Vector3i(1, -1, 0));
        
        atoms.set_bilinear_interaction(J2B_mat, 1, 1, Eigen::Vector3i(1, 0, 0));
        atoms.set_bilinear_interaction(J2B_mat, 1, 1, Eigen::Vector3i(0, 1, 0));
        atoms.set_bilinear_interaction(J2B_mat, 1, 1, Eigen::Vector3i(1, -1, 0));
    }
    
    // 3rd NN interactions
    if (std::abs(J3) > 1e-12) {
        Eigen::Matrix3d J3_mat = J3 * Eigen::Matrix3d::Identity();
        
        atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(1, -2, 0));
        atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(-1, 0, 0));
        atoms.set_bilinear_interaction(J3_mat, 0, 1, Eigen::Vector3i(1, 0, 0));
    }
    
    // Set Kitaev local frame
    Eigen::Matrix3d R;
    R << 1.0/std::sqrt(6.0), -1.0/std::sqrt(2.0), 1.0/std::sqrt(3.0),
         1.0/std::sqrt(6.0),  1.0/std::sqrt(2.0), 1.0/std::sqrt(3.0),
        -2.0/std::sqrt(6.0),  0.0,                1.0/std::sqrt(3.0);
    atoms.set_sublattice_frame(R, 0);
    atoms.set_sublattice_frame(R, 1);
    
    // Set magnetic field
    Eigen::Vector3d field;
    field << config.field_strength * config.field_direction[0],
             config.field_strength * config.field_direction[1],
             config.field_strength * config.field_direction[2];
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    
    return atoms;
}
