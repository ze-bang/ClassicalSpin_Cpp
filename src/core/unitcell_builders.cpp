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

// -----------------------------------------------------------------------------
// Pbnm sublattice "baking" matrices.
//
// The whole TmFeO3 model is assembled in ONE global lab/Cartesian + canonical
// Gell-Mann basis.  There is no separate local frame and no global/local frame
// switch: every UnitCell sublattice frame is left at identity, and the
// per-sublattice Pbnm rotations are absorbed (baked) directly into the stored
// fields and coupling tensors.
//
//   * Klein-four proper rotations R_mu = diag(+/-1) relate the four Fe (4b)
//     and the four Tm (4c) sublattices:
//         R_0 = diag(+,+,+)   R_1 = diag(+,-,-)
//         R_2 = diag(-,+,-)   R_3 = diag(-,-,+)
//   * The Tm magnetic dipole is axial, mu_j = R_j mu_0, so a uniform field
//     couples to each Tm sublattice through its own rotated dipole.
//   * The same rotation is realised on the 8-dim Gell-Mann index by the induced
//     SU(3) baking matrix
//         F_mu = mu_act^{-1} R_mu mu_act      (active (2,5,7)/(1,4,6) blocks),
//     the unique map that makes J = mu_act . lambda rotate as an axial vector
//     while the populations (lambda^3, lambda^8) stay invariant.  F_mu^2 = I.
std::array<Eigen::Matrix3d, 4> tmfeo3_klein_frames() {
    std::array<Eigen::Matrix3d, 4> R;
    R[0] = Eigen::Vector3d(+1.0, +1.0, +1.0).asDiagonal();
    R[1] = Eigen::Vector3d(+1.0, -1.0, -1.0).asDiagonal();
    R[2] = Eigen::Vector3d(-1.0, +1.0, -1.0).asDiagonal();
    R[3] = Eigen::Vector3d(-1.0, -1.0, +1.0).asDiagonal();
    return R;
}

SpinMatrix tmfeo3_su3_bake(const Eigen::Matrix3d& mu_act,
                           const Eigen::Matrix3d& R_xyz) {
    // 8x8 Gell-Mann baking matrix F_mu = mu_act^{-1} R_xyz mu_act induced by the
    // 3x3 axial rotation R_xyz.  It acts with the same 3x3 block on the
    // imaginary triplet (lambda^2,5,7 -> idx 1,4,6) and the real triplet
    // (lambda^1,4,6 -> idx 0,3,5), and leaves (lambda^3, lambda^8) untouched.
    constexpr int active_im[3] = {1, 4, 6};
    constexpr int active_re[3] = {0, 3, 5};

    SpinMatrix F = SpinMatrix::Identity(8, 8);
    Eigen::FullPivLU<Eigen::Matrix3d> lu(mu_act);
    if (lu.rank() != 3) {
        return F;
    }

    const Eigen::Matrix3d blk = lu.solve(R_xyz * mu_act);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            F(active_im[row], active_im[col]) = blk(row, col);
            F(active_re[row], active_re[col]) = blk(row, col);
        }
    }
    return F;
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

    // H_Fe^(0) = (1/2) sum_<ik> S_i^T J_ik S_k + sum_i S_i^T A_i S_i, with
    //   J_{ik,ab} = J_ik delta_ab + Gamma_{ik,ab} + eps_{abc} D_{ik,c}.
    // Everything is written directly in the lab Cartesian frame; the Pbnm
    // covariance J_{g(ik)} = R_g^T J_ik R_g is realised by the explicit per-bond
    // matrices below (the z=1/2 and z=0 planes differ by the S_1 = R_z(pi)
    // image, which flips D_y but preserves D_z).
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

    // Single-ion anisotropy A_i = diag(Ka, Kb, Kc), the same on every Fe site
    // (R_g^T A R_g = A for the diagonal Klein-four rotations).
    Eigen::MatrixXd K_mat = Eigen::MatrixXd::Zero(3, 3);
    K_mat(0, 0) = Ka;
    K_mat(1, 1) = Kb;
    K_mat(2, 2) = Kc;
    for (int i = 0; i < 4; ++i) Fe_atoms.set_onsite_interaction(K_mat, i);

    // Lab-frame Gamma_2 (Fx, Gz): G_z is the staggered (+,-,+,-) pattern.
    Fe_atoms.set_afm_sublattice_signs({+1.0, -1.0, +1.0, -1.0});

    // Direct Fe Zeeman drive H_Fe,B = -g_Fe mu_B B . S_i: a uniform lab field on
    // every Fe site (sublattice frames stay identity).
    Eigen::Vector3d h_lab;
    h_lab << config.field_direction[0] * h,
             config.field_direction[1] * h,
             config.field_direction[2] * h;
    for (int i = 0; i < 4; ++i) Fe_atoms.set_field(h_lab, i);
}



UnitCell build_tmfeo3_fe(const SpinConfig& config) {
    TmFeO3_Fe Fe_atoms(3);
    apply_tmfeo3_fe_sector(Fe_atoms, config);
    return Fe_atoms;
}

UnitCell build_tmfeo3_tm(const SpinConfig& config) {
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
