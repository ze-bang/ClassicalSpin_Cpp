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
    //   F_mu = mu_act^{-1} R_xyz mu_act.
    // The coherence partners (lambda_1, lambda_4, lambda_6) must carry the
    // same 3x3 action so each |E_a><E_b| pair has one bulk character, while
    // the diagonal population channels (lambda_3, lambda_8) remain invariant.
    //
    // Physical justification (non-Kramers derivation):
    //   Tm^3+ is non-Kramers (J=6 integer), so time reversal squares to +1
    //   and the three CEF eigenstates |1>,|2>,|3> can be chosen real.
    //   The four Tm sublattices are related by proper C_2 rotations (Pbnm
    //   coset reps), which act as real orthogonal matrices V_{g_i} on the
    //   3-level subspace via rho -> V rho V^T (real V).
    //   Under this action, V_{jj'} V_{kk'} multiplies both Re(rho_{j'k'})
    //   and Im(rho_{j'k'}) identically, so:
    //     F_i^E = F_i^M = mu_act^{-1} D_i mu_act
    //   for all four sublattices.  The A1+/A2+ inversion-parity distinction
    //   (lambda_1 vs lambda_{4,6}) affects external-coupling signs (the W
    //   inv flag) but NOT the proper-rotation frame, which sees all
    //   off-diagonal pairs identically.
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

    const auto Ja   = J_with_DM(Jai, +D1, +D2);   // z=1/2 plane
    const auto Jb   = J_with_DM(Jbi, +D1, +D2);   // z=1/2 plane
    const auto Ja23 = J_with_DM(Jai, -D1, +D2);   // z=0   plane (D_y flipped)
    const auto Jb23 = J_with_DM(Jbi, -D1, +D2);   // z=0   plane
    const auto Jc   = J_iso(Jci);
    const auto J2a  = J_iso(J2ai);
    const auto J2b  = J_iso(J2bi);
    const auto J2c  = J_iso(J2ci);

    // In-plane J_1 bonds in z = 1/2 plane (Fe1 -> Fe0).
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, Eigen::Vector3i(0,  0, 0));
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, Eigen::Vector3i(1,  0, 0));

    // In-plane J_1 bonds in z = 0 plane (Fe2 -> Fe3); D_y flipped vs Fe1->Fe0.
    Fe_atoms.set_bilinear_interaction(Ja23, 2, 3, Eigen::Vector3i(0,  0, 0));
    Fe_atoms.set_bilinear_interaction(Ja23, 2, 3, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(Jb23, 2, 3, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(Jb23, 2, 3, Eigen::Vector3i(1,  0, 0));

    // Out-of-plane J_1c (Fe0 <-> Fe3 and Fe1 <-> Fe2 along c).
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, Eigen::Vector3i(0, 0, 1));

    // Intra-sublattice J_2 along a, b, c.
    for (int i = 0; i < 4; ++i) {
        Fe_atoms.set_bilinear_interaction(J2a, i, i, Eigen::Vector3i(1, 0, 0));
        Fe_atoms.set_bilinear_interaction(J2b, i, i, Eigen::Vector3i(0, 1, 0));
        Fe_atoms.set_bilinear_interaction(J2c, i, i, Eigen::Vector3i(0, 0, 1));
    }

    // Cross-sublattice J_2c bonds: Fe0<->Fe2 and Fe1<->Fe3 (8 offsets each).
    {
        const Eigen::Vector3i offs02[] = {
            { 0, 0, 0}, { 0, 1, 0}, {-1, 0, 0}, {-1, 1, 0},
            { 0, 0, 1}, { 0, 1, 1}, {-1, 0, 1}, {-1, 1, 1}
        };
        for (const auto& off : offs02)
            Fe_atoms.set_bilinear_interaction(J2c, 0, 2, off);

        const Eigen::Vector3i offs13[] = {
            { 0,  0, 0}, { 0, -1, 0}, { 1,  0, 0}, { 1, -1, 0},
            { 0,  0, 1}, { 0, -1, 1}, { 1,  0, 1}, { 1, -1, 1}
        };
        for (const auto& off : offs13)
            Fe_atoms.set_bilinear_interaction(J2c, 1, 3, off);
    }

    // Single-ion anisotropy.
    Eigen::MatrixXd K_mat = Eigen::MatrixXd::Zero(3, 3);
    K_mat(0, 0) = Ka;
    K_mat(1, 1) = Kb;
    K_mat(2, 2) = Kc;
    for (int i = 0; i < 4; ++i) Fe_atoms.set_onsite_interaction(K_mat, i);

    // Lab-frame Γ_2 (Fx, Gz): G_z is the staggered (+,-,+,-) pattern.
    Fe_atoms.set_afm_sublattice_signs({+1.0, -1.0, +1.0, -1.0});

    // Direct Fe Zeeman drive: uniform lab field on every Fe site.
    Eigen::Vector3d h_lab;
    h_lab << config.field_direction[0] * h,
             config.field_direction[1] * h,
             config.field_direction[2] * h;
    for (int i = 0; i < 4; ++i)
        Fe_atoms.set_field(h_lab, i);
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
    // tm_identity_frame=1: set all Tm sublattice frames to identity so that
    // M_global_SU3 and all drives act directly in the local CEF basis at every
    // site without staggered projection.  Useful for testing the chi2 couplings
    // when the physical observable is the net uniform (sigma_F) Tm moment.
    const bool tm_identity_frame = config.get_param("tm_identity_frame", 0.0) != 0.0;
    const auto R_tm_xyz = tmfeo3_tm_local_frames_xyz();
    for (int sub = 0; sub < 4; ++sub) {
        if (tm_identity_frame) {
            Tm_atoms.set_sublattice_frame(SpinMatrix::Identity(8, 8), sub);
        } else {
            Tm_atoms.set_sublattice_frame(tmfeo3_induced_su3_frame(mu_act, R_tm_xyz[sub]), sub);
        }
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
// The same Fe-Tm topology is used for both the linear chi term and the on-site
// trilinear W term. For each pair, the odd bond is the inversion-related
// partner of the even bond. The mirror-odd Tm channels therefore pick up the
// opposite sign on the odd bond, while the mirror-even channels do not.
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
    struct BilinearChannel {
        double x, y, z;
    };

    struct TrilinearChannel {
        double xx, yy, zz, xy, xz, yz;
    };

    const BilinearChannel chi2_ch = {
        config.get_param("chi2x", 0.0),
        config.get_param("chi2y", 0.0),
        config.get_param("chi2z", 0.0)
    };
    const BilinearChannel chi5_ch = {
        config.get_param("chi5x", 0.0),
        config.get_param("chi5y", 0.0),
        config.get_param("chi5z", 0.0)
    };
    const BilinearChannel chi7_ch = {
        config.get_param("chi7x", 0.0),
        config.get_param("chi7y", 0.0),
        config.get_param("chi7z", 0.0)
    };
    const double chi_orbit_scale[4] = {
        config.get_param("chi_orbit1_scale", 1.0),
        config.get_param("chi_orbit2_scale", 1.0),
        config.get_param("chi_orbit3_scale", 1.0),
        config.get_param("chi_orbit4_scale", 1.0)
    };

    // ---- chiJ mode: coupling in physical Tm dipole (Jx,Jy,Jz) space ----
    // Parameterises H = sum_{alpha,beta} chiJ[beta][alpha] * S^alpha_Fe * J^beta_Tm
    // where J^beta_Tm = sum_a mu_act(beta,a) * lambda_active_a  is the physical
    // projected magnetic moment operator and S^alpha_Fe is in the Fe local frame.
    // Naming: chiJ_<Tm-moment><Fe-spin>, e.g. chiJ_zSz = coupling of J_z^Tm to S_z^Fe.
    // Defaults all zero so existing configs are unaffected.
    // mu_act defaults must match apply_tmfeo3_tm_sector.
    Eigen::Matrix3d mu_J;   // mu_J(beta, a): beta=xyz moment, a=active index {lam2,lam5,lam7}
    mu_J << config.get_param("mu_2x", 0.0),    config.get_param("mu_5x",  2.3915), config.get_param("mu_7x", 0.9128),
            config.get_param("mu_2y", 0.0),    config.get_param("mu_5y", -2.7866), config.get_param("mu_7y", 0.4655),
            config.get_param("mu_2z", 5.264),  config.get_param("mu_5z",  0.0),    config.get_param("mu_7z", 0.0);

    Eigen::Matrix3d chiJ = Eigen::Matrix3d::Zero(); // chiJ(beta, alpha): Tm-moment beta, Fe-component alpha
    chiJ(0, 0) = config.get_param("chiJ_xSx", 0.0);
    chiJ(0, 1) = config.get_param("chiJ_xSy", 0.0);
    chiJ(0, 2) = config.get_param("chiJ_xSz", 0.0);
    chiJ(1, 0) = config.get_param("chiJ_ySx", 0.0);
    chiJ(1, 1) = config.get_param("chiJ_ySy", 0.0);
    chiJ(1, 2) = config.get_param("chiJ_ySz", 0.0);
    chiJ(2, 0) = config.get_param("chiJ_zSx", 0.0);
    chiJ(2, 1) = config.get_param("chiJ_zSy", 0.0);
    chiJ(2, 2) = config.get_param("chiJ_zSz", 0.0);

    // chiJ_lam(alpha, a) = sum_beta chiJ(beta,alpha) * mu_J(beta,a)
    // Fe-component alpha drives active Gell-Mann generator a via the moment matrix.
    const Eigen::Matrix3d chiJ_lam = chiJ.transpose() * mu_J;
    const bool any_chiJ = chiJ_lam.cwiseAbs().maxCoeff() > 0.0;


    auto read_tri_ch = [&](const std::string& pfx) -> TrilinearChannel {
        return {
            config.get_param(pfx + "_xx", 0.0),
            config.get_param(pfx + "_yy", 0.0),
            config.get_param(pfx + "_zz", 0.0),
            config.get_param(pfx + "_xy", 0.0),
            config.get_param(pfx + "_xz", 0.0),
            config.get_param(pfx + "_yz", 0.0)
        };
    };
    auto tri_ch_nonzero = [](const TrilinearChannel& channel) {
        return channel.xx != 0.0 || channel.yy != 0.0 || channel.zz != 0.0
            || channel.xy != 0.0 || channel.xz != 0.0 || channel.yz != 0.0;
    };
    auto bilinear_ch_nonzero = [](const BilinearChannel& channel) {
        return channel.x != 0.0 || channel.y != 0.0 || channel.z != 0.0;
    };

    const TrilinearChannel W1_ch = read_tri_ch("W1");
    const TrilinearChannel W3_ch = read_tri_ch("W3");
    const TrilinearChannel W4_ch = read_tri_ch("W4");
    const TrilinearChannel W6_ch = read_tri_ch("W6");
    const TrilinearChannel W8_ch = read_tri_ch("W8");

    const double W_orbit_scale[4] = {
        config.get_param("W_orbit1_scale", 1.0),
        config.get_param("W_orbit2_scale", 1.0),
        config.get_param("W_orbit3_scale", 1.0),
        config.get_param("W_orbit4_scale", 1.0)
    };

    // chi coupling tensor for a specific bond: chi^{ab} couples the LOCAL-frame
    // Fe spin component S^a_loc to the Tm CEF-basis Gell-Mann operator lambda^b.
    // Fe spins are stored in the global lab Cartesian frame.
    // The chi/W/kappa parameters are defined and interpreted in the global Fe frame.
    //
    // Pbnm even/odd bond sign analysis:
    //   Under inversion I: Fe sites permute as [0123] with no spin rotation in global frame.
    //   The sign pattern comes only from: E/B field parity x Tm lambda inversion parity.
    //   Proper cosets: S_glob just permutes (no rotation), so the bond network closure
    //   automatically satisfies them — no additional constraints.
    // Pbnm bond-topology invariance is handled through the even/odd bond pairs and
    // the Tm sublattice frames.
    auto build_chi_bond = [&](double sign_57) {
        SpinMatrix tensor = SpinMatrix::Zero(3, 8);
        // chi{2,5,7}{x,y,z} channels (direct Gell-Mann coupling, global Fe frame)
        tensor(0, 1) = chi2_ch.x;
        tensor(1, 1) = chi2_ch.y;
        tensor(2, 1) = chi2_ch.z;
        tensor(0, 4) = sign_57 * chi5_ch.x;
        tensor(1, 4) = sign_57 * chi5_ch.y;
        tensor(2, 4) = sign_57 * chi5_ch.z;
        tensor(0, 6) = sign_57 * chi7_ch.x;
        tensor(1, 6) = sign_57 * chi7_ch.y;
        tensor(2, 6) = sign_57 * chi7_ch.z;
        // chiJ mode: coupling in physical (Jx,Jy,Jz) Tm moment space.
        // chiJ_lam(alpha,a) encodes sum_beta chiJ(beta,alpha)*mu_J(beta,a).
        if (any_chiJ) {
            for (int alpha = 0; alpha < 3; ++alpha) {
                tensor(alpha, 1) += chiJ_lam(alpha, 0);          // lam2, inversion-even
                tensor(alpha, 4) += sign_57 * chiJ_lam(alpha, 1); // lam5, inversion-odd
                tensor(alpha, 6) += sign_57 * chiJ_lam(alpha, 2); // lam7, inversion-odd
            }
        }
        return tensor;
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
    auto scale_matrix = [](const SpinMatrix& matrix, double scale) {
        return scale * matrix;
    };

    const auto& bond_pairs = fe_tm_w_bond_pairs();

    // Linear Fe-Tm term:
    //   H_chi = sum_{(i,mu)} S_i^a chi_{ab} lambda_mu^b.
    // The live TmFeO3 parameterization keeps only the time-odd Tm channels
    // {lambda_2, lambda_5, lambda_7}. lambda_2 is inversion-even, while
    // lambda_5/lambda_7 flip sign on the inversion-related partner (sign_57).
    const bool any_chi = bilinear_ch_nonzero(chi2_ch)
                      || bilinear_ch_nonzero(chi5_ch)
                      || bilinear_ch_nonzero(chi7_ch)
                      || any_chiJ;
    if (any_chi) {
        for (const auto& pair : bond_pairs) {
            const double orbit_scale = chi_orbit_scale[pair.orbit - 1];
            mixed_uc.set_mixed_bilinear(
                scale_matrix(build_chi_bond(+1.0), orbit_scale),
                pair.even.fe, pair.even.tm, pair.even.off);
            mixed_uc.set_mixed_bilinear(
                scale_matrix(build_chi_bond(-1.0), orbit_scale),
                pair.odd.fe, pair.odd.tm, pair.odd.off);
        }
    }

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
        // W channels use LOCAL-frame Fe spin components (same convention as Fe-Fe H).
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

    // -------------------------------------------------------------------------
    // Field-assisted Fe-Tm exchange: H_{E chi} and H_{B chi}
    //   (tmfeo3_foundation.tex, Eqs. H_E_chi_reorg / H_B_chi_reorg).
    //
    // These are pulse-envelope-modulated mixed bilinears.  At runtime the
    // coupling is scaled by the SU(3) pulse envelope (electric/THz field E(t),
    // H_{E chi}) or the SU(2) pulse envelope (magnetic field B(t), H_{B chi}),
    // so they vanish outside the pulse window and are absent from the static
    // energy.  The kappaE_*/kappaB_* parameters are the coupling coefficients
    // already contracted onto the configured pulse polarization (matching the
    // tex specialization, e.g. kappaE_5y = kappa^E_{c;5y}).
    //
    // Inversion-coset signs (Eqs. Ipa / IEpa):
    //   * H_{E chi}: lambda^2 is A1 (p=+1) -> the E-prefactor flips it on the
    //     odd bond (sign -1); lambda^{5,7} are A2 (p=-1) -> the E-prefactor
    //     leaves them with sign +1 on BOTH bonds (no q=0 cancellation).
    //   * H_{B chi}: B is axial (no extra sign).  lambda^1 (p=+1) keeps +1 on
    //     both bonds; lambda^{4,6} (p=-1) flip sign on the odd bond.
    // -------------------------------------------------------------------------
    {
        const double kappaE_2x = config.get_param("kappaE_2x", 0.0);
        const double kappaE_2y = config.get_param("kappaE_2y", 0.0);
        const double kappaE_2z = config.get_param("kappaE_2z", 0.0);
        const double kappaE_5x = config.get_param("kappaE_5x", 0.0);
        const double kappaE_5y = config.get_param("kappaE_5y", 0.0);
        const double kappaE_5z = config.get_param("kappaE_5z", 0.0);
        const double kappaE_7x = config.get_param("kappaE_7x", 0.0);
        const double kappaE_7y = config.get_param("kappaE_7y", 0.0);
        const double kappaE_7z = config.get_param("kappaE_7z", 0.0);

        const double kappaB_1x = config.get_param("kappaB_1x", 0.0);
        const double kappaB_1y = config.get_param("kappaB_1y", 0.0);
        const double kappaB_1z = config.get_param("kappaB_1z", 0.0);
        const double kappaB_4x = config.get_param("kappaB_4x", 0.0);
        const double kappaB_4y = config.get_param("kappaB_4y", 0.0);
        const double kappaB_4z = config.get_param("kappaB_4z", 0.0);
        const double kappaB_6x = config.get_param("kappaB_6x", 0.0);
        const double kappaB_6y = config.get_param("kappaB_6y", 0.0);
        const double kappaB_6z = config.get_param("kappaB_6z", 0.0);

        const bool any_assistE = kappaE_2x || kappaE_2y || kappaE_2z
                              || kappaE_5x || kappaE_5y || kappaE_5z
                              || kappaE_7x || kappaE_7y || kappaE_7z;
        const bool any_assistB = kappaB_1x || kappaB_1y || kappaB_1z
                              || kappaB_4x || kappaB_4y || kappaB_4z
                              || kappaB_6x || kappaB_6y || kappaB_6z;

        // Tm Gell-Mann column indices for the active channels.
        constexpr int LAM1 = 0, LAM2 = 1, LAM4 = 3, LAM5 = 4, LAM6 = 5, LAM7 = 6;

        // E-assisted exchange tensor for one bond (3 Fe components x 8 Tm channels).
        // Fe spin components are in the LOCAL sigma_C frame.
        // Odd-bond signs: E is polar (flips under I), S_loc is invariant under I.
        //   lambda^2  (A1, inv-even): odd sign = (-E)(+S)(+lam) -> -1.
        //   lambda^5,7 (A2, inv-odd): odd sign = (-E)(+S)(-lam) -> +1.
        auto build_assistE = [&](bool odd) {
            SpinMatrix T = SpinMatrix::Zero(3, 8);
            const double s2  = odd ? -1.0 : 1.0;  // lambda^2: flips on odd bond
            const double s57 = 1.0;               // lambda^{5,7}: +1 on both bonds
            T(0, LAM2) = s2 * kappaE_2x;
            T(1, LAM2) = s2 * kappaE_2y;
            T(2, LAM2) = s2 * kappaE_2z;
            T(0, LAM5) = s57 * kappaE_5x;
            T(1, LAM5) = s57 * kappaE_5y;
            T(2, LAM5) = s57 * kappaE_5z;
            T(0, LAM7) = s57 * kappaE_7x;
            T(1, LAM7) = s57 * kappaE_7y;
            T(2, LAM7) = s57 * kappaE_7z;
            return T;
        };

        // B-assisted exchange tensor for one bond.
        // Fe spin components are in the LOCAL sigma_C frame.
        // Odd-bond signs: B is axial (invariant under I), S_loc invariant under I.
        //   lambda^1  (A1, inv-even): odd sign = (+B)(+S)(+lam) -> +1.
        //   lambda^4,6 (A2, inv-odd): odd sign = (+B)(+S)(-lam) -> -1.
        auto build_assistB = [&](bool odd) {
            SpinMatrix T = SpinMatrix::Zero(3, 8);
            const double s1  = 1.0;               // lambda^1: +1 on both bonds
            const double s46 = odd ? -1.0 : 1.0;  // lambda^{4,6}: flip on odd bond
            T(0, LAM1) = s1 * kappaB_1x;
            T(1, LAM1) = s1 * kappaB_1y;
            T(2, LAM1) = s1 * kappaB_1z;
            T(0, LAM4) = s46 * kappaB_4x;
            T(1, LAM4) = s46 * kappaB_4y;
            T(2, LAM4) = s46 * kappaB_4z;
            T(0, LAM6) = s46 * kappaB_6x;
            T(1, LAM6) = s46 * kappaB_6y;
            T(2, LAM6) = s46 * kappaB_6z;
            return T;
        };

        if (any_assistE || any_assistB) {
            for (const auto& pair : bond_pairs) {
                if (any_assistE) {
                    mixed_uc.set_mixed_bilinear_drive(
                        build_assistE(false),
                        pair.even.fe, pair.even.tm, pair.even.off, /*envelope=*/0);
                    mixed_uc.set_mixed_bilinear_drive(
                        build_assistE(true),
                        pair.odd.fe, pair.odd.tm, pair.odd.off, /*envelope=*/0);
                }
                if (any_assistB) {
                    mixed_uc.set_mixed_bilinear_drive(
                        build_assistB(false),
                        pair.even.fe, pair.even.tm, pair.even.off, /*envelope=*/1);
                    mixed_uc.set_mixed_bilinear_drive(
                        build_assistB(true),
                        pair.odd.fe, pair.odd.tm, pair.odd.off, /*envelope=*/1);
                }
            }
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
