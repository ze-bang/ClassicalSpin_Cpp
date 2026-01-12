/**
 * unitcell_builders.cpp - Unit cell builder functions
 * 
 * This file contains the implementations of the unit cell builder functions
 * that create specific lattice types (BCAO, Kitaev, Pyrochlore, TmFeO3).
 * These are separated from the main simulation code to reduce compile time.
 */

#include "classical_spin/core/unitcell_builders.h"
#include <cmath>

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
    
    // Set magnetic field
    Eigen::Vector3d field;
    field << config.field_strength * config.field_direction[0],
             config.field_strength * config.field_direction[1],
             config.field_strength * config.field_direction[2];
    
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

MixedUnitCell build_tmfeo3(const SpinConfig& config) {
    const double Jai = config.get_param("J1ab", 4.92);
    const double Jbi = Jai;
    const double Jci = config.get_param("J1c", 4.92);
    const double J2ai = config.get_param("J2ab", 0.29);
    const double J2bi = J2ai;
    const double J2ci = config.get_param("J2c", 0.29);
    const double Ka = config.get_param("Ka", 0.0);
    const double Kc = config.get_param("Kc", -0.09);
    const double D1 = config.get_param("D1", 0.0);
    const double D2 = config.get_param("D2", 0.0);
    const double chi2x = config.get_param("chi2x", 0.0);
    const double chi2y = config.get_param("chi2y", 0.0);
    const double chi2z = config.get_param("chi2z", 0.0);
    const double chi5x = config.get_param("chi5x", 0.0);
    const double chi5y = config.get_param("chi5y", 0.0);
    const double chi5z = config.get_param("chi5z", 0.0);
    const double chi7x = config.get_param("chi7x", 0.0);
    const double chi7y = config.get_param("chi7y", 0.0);
    const double chi7z = config.get_param("chi7z", 0.0);
    const double e1 = config.get_param("e1", 0.97);
    const double e2 = config.get_param("e2", 3.97);
    // Tm-Tm diagonal bilinear coupling (nearest neighbor) - each component couples λ_a ⊗ λ_a only
    const double Jtm_1 = config.get_param("Jtm_1", 0.0);
    const double Jtm_2 = config.get_param("Jtm_2", 0.0);
    const double Jtm_3 = config.get_param("Jtm_3", 0.0);
    const double Jtm_4 = config.get_param("Jtm_4", 0.0);
    const double Jtm_5 = config.get_param("Jtm_5", 0.0);
    const double Jtm_6 = config.get_param("Jtm_6", 0.0);
    const double Jtm_7 = config.get_param("Jtm_7", 0.0);
    const double Jtm_8 = config.get_param("Jtm_8", 0.0);
    const double h = config.field_strength;
    
    // Use TmFeO3_Fe and TmFeO3_Tm classes from unitcell.h (already have structure)
    TmFeO3_Fe Fe_atoms(3);
    TmFeO3_Tm Tm_atoms(8);
    
    // Local frame transformation (following molecular_dynamic_TmFeO3.cpp exactly)
    std::array<std::array<double, 3>, 4> eta = {{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}};
    
    // Original exchange matrices in global frame
    std::array<std::array<double, 3>, 3> Ja_orig = {{{Jai, D2, -D1}, {-D2, Jai, 0}, {D1, 0, Jai}}};
    std::array<std::array<double, 3>, 3> Jb_orig = {{{Jbi, D2, -D1}, {-D2, Jbi, 0}, {D1, 0, Jbi}}};
    std::array<std::array<double, 3>, 3> Jc_orig = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};
    std::array<std::array<double, 3>, 3> J2a_orig = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    std::array<std::array<double, 3>, 3> J2b_orig = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    std::array<std::array<double, 3>, 3> J2c_orig = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};
    
    // Transform to local frames: J_local[i][j][a][b] = J_orig[a][b] * eta[i][a] * eta[j][b]
    std::array<std::array<std::array<std::array<double, 3>, 3>, 4>, 4> Ja, Jb, Jc, J2a, J2b, J2c;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    Ja[i][j][a][b] = Ja_orig[a][b] * eta[i][a] * eta[j][b];
                    Jb[i][j][a][b] = Jb_orig[a][b] * eta[i][a] * eta[j][b];
                    Jc[i][j][a][b] = Jc_orig[a][b] * eta[i][a] * eta[j][b];
                    J2a[i][j][a][b] = J2a_orig[a][b] * eta[i][a] * eta[j][b];
                    J2b[i][j][a][b] = J2b_orig[a][b] * eta[i][a] * eta[j][b];
                    J2c[i][j][a][b] = J2c_orig[a][b] * eta[i][a] * eta[j][b];
                }
            }
        }
    }
    
    // Convert to Eigen matrices for setting interactions
    auto to_eigen = [](const std::array<std::array<double, 3>, 3>& arr) {
        Eigen::Matrix3d mat;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mat(i, j) = arr[i][j];
            }
        }
        return mat;
    };
    
    // Set Fe-Fe interactions (following exact bond pattern from legacy code)
    // In-plane interactions (J1 type)
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja[1][0]), 1, 0, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja[1][0]), 1, 0, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb[1][0]), 1, 0, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb[1][0]), 1, 0, Eigen::Vector3i(1, 0, 0));
    
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja[2][3]), 2, 3, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja[2][3]), 2, 3, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb[2][3]), 2, 3, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb[2][3]), 2, 3, Eigen::Vector3i(1, 0, 0));
    
    // Next nearest neighbor (J2 type, along a, b, and c axes - same sublattice)
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[0][0]), 0, 0, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[0][0]), 0, 0, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][0]), 0, 0, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[1][1]), 1, 1, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[1][1]), 1, 1, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][1]), 1, 1, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[2][2]), 2, 2, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[2][2]), 2, 2, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[2][2]), 2, 2, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[3][3]), 3, 3, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[3][3]), 3, 3, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[3][3]), 3, 3, Eigen::Vector3i(0, 0, 1));
    
    // Out of plane interactions (J1 type along c-axis)
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[0][3]), 0, 3, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[0][3]), 0, 3, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[1][2]), 1, 2, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[1][2]), 1, 2, Eigen::Vector3i(0, 0, 1));
    
    // J2 out-of-plane interactions (cross-sublattice diagonal)
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(-1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(-1, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(0, 1, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(-1, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(-1, 1, 1));
    
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(0, -1, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(1, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(1, -1, 1));
    
    // Single ion anisotropy (same in all local frames)
    Eigen::MatrixXd K_mat = Eigen::MatrixXd::Zero(3, 3);
    K_mat(0, 0) = Ka;
    K_mat(2, 2) = Kc;
    Fe_atoms.set_onsite_interaction(K_mat, 0);
    Fe_atoms.set_onsite_interaction(K_mat, 1);
    Fe_atoms.set_onsite_interaction(K_mat, 2);
    Fe_atoms.set_onsite_interaction(K_mat, 3);
    
    // External magnetic field
    Eigen::Vector3d field;
    field << config.field_direction[0] * h,
             config.field_direction[1] * h,
             config.field_direction[2] * h;
    Fe_atoms.set_field(field, 0);
    Fe_atoms.set_field(field, 1);
    Fe_atoms.set_field(field, 2);
    Fe_atoms.set_field(field, 3);
    
    const double tm_alpha_scale = config.get_param("tm_alpha_scale", 1.0);
    const double tm_beta_scale = config.get_param("tm_beta_scale", 1.0);
    double alpha = e1 * tm_alpha_scale;
    double beta = (2.0 * e2 - e1) / sqrt(3.0) * tm_beta_scale;
    Eigen::VectorXd tm_field(8);
    tm_field << 0, 0, alpha, 0, 0, 0, 0, beta;
    
    Tm_atoms.set_field(tm_field, 0);
    Tm_atoms.set_field(tm_field, 1);
    Tm_atoms.set_field(tm_field, 2);
    Tm_atoms.set_field(tm_field, 3);
    
    // Tm-Tm diagonal bilinear interactions (nearest neighbor)
    // Diagonal in Gell-Mann space: J_a * λ_a ⊗ λ_a (no cross terms)
    // Nearest neighbors based on real-space positions:
    //   Tm0 (0.02111, 0.92839, 0.75) ↔ Tm2 (0.47889, 0.42839, 0.75): d ≈ 0.68
    //   Tm1 (0.52111, 0.57161, 0.25) ↔ Tm3 (0.97889, 0.07161, 0.25): d ≈ 0.68
    if (Jtm_1 != 0.0 || Jtm_2 != 0.0 || Jtm_3 != 0.0 || Jtm_4 != 0.0 ||
        Jtm_5 != 0.0 || Jtm_6 != 0.0 || Jtm_7 != 0.0 || Jtm_8 != 0.0) {
        Eigen::MatrixXd J_tm_mat = Eigen::MatrixXd::Zero(8, 8);
        J_tm_mat(0, 0) = Jtm_1;
        J_tm_mat(1, 1) = Jtm_2;
        J_tm_mat(2, 2) = Jtm_3;
        J_tm_mat(3, 3) = Jtm_4;
        J_tm_mat(4, 4) = Jtm_5;
        J_tm_mat(5, 5) = Jtm_6;
        J_tm_mat(6, 6) = Jtm_7;
        J_tm_mat(7, 7) = Jtm_8;
        
        // Tm0 ↔ Tm2 nearest neighbors (z=0.75 plane, d ≈ 0.68)
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 2, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 2, Eigen::Vector3i(0, 1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 2, Eigen::Vector3i(-1, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 2, Eigen::Vector3i(-1, 1, 0));
        
        // Tm1 ↔ Tm3 nearest neighbors (z=0.25 plane, d ≈ 0.68)
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(0, 1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(1, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(1, 1, 0));
        
        // Out-of-plane nearest neighbors (between z=0.75 and z=0.25 planes)
        // Tm0 (0.02111, 0.92839, 0.75) ↔ Tm3 (0.97889, 0.07161, 0.25)
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(0, 1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(0, 0, 1));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(0, 1, 1));
        
        // Tm2 (0.47889, 0.42839, 0.75) ↔ Tm1 (0.52111, 0.57161, 0.25)
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, 0, 1));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, -1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, -1, 1));
    }
    
    // Create mixed unit cell
    MixedUnitCell mixed_uc(Fe_atoms, Tm_atoms);
    
    // Set Fe-Tm bilinear coupling (following exact pattern from legacy code)
    if (chi2x != 0.0 || chi2y != 0.0 || chi2z != 0.0 || chi5x != 0.0 || chi5y != 0.0 || chi5z != 0.0 || chi7x != 0.0 || chi7y != 0.0 || chi7z != 0.0) {
        Eigen::MatrixXd chi = Eigen::MatrixXd::Zero(8, 3);
        chi(1, 0) = chi2x; chi(1, 1) = chi2y; chi(1, 2) = chi2z;
        chi(4, 0) = chi5x; chi(4, 1) = chi5y; chi(4, 2) = chi5z;
        chi(6, 0) = chi7x; chi(6, 1) = chi7y; chi(6, 2) = chi7z;
        
        Eigen::MatrixXd chi_inv = Eigen::MatrixXd::Zero(8, 3);
        chi_inv(1, 0) = chi2x; chi_inv(1, 1) = chi2y; chi_inv(1, 2) = chi2z;
        chi_inv(4, 0) = -chi5x; chi_inv(4, 1) = -chi5y; chi_inv(4, 2) = -chi5z;
        chi_inv(6, 0) = -chi7x; chi_inv(6, 1) = -chi7y; chi_inv(6, 2) = -chi7z;
        
        // Fe site 0 - 8 nearest Tm neighbors
        mixed_uc.set_mixed_bilinear(chi, 3, 0, Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_bilinear(chi_inv, 0, 0, Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_bilinear(chi, 2, 0, Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_bilinear(chi_inv, 1, 0, Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_bilinear(chi, 1, 0, Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_bilinear(chi_inv, 2, 0, Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_bilinear(chi, 0, 0, Eigen::Vector3i(0, -1, 0));
        mixed_uc.set_mixed_bilinear(chi_inv, 3, 0, Eigen::Vector3i(-1, 1, 0));
        
        // Fe site 1 - 8 nearest Tm neighbors
        mixed_uc.set_mixed_bilinear(chi, 2, 1, Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_bilinear(chi_inv, 1, 1, Eigen::Vector3i(0, -1, 0));
        mixed_uc.set_mixed_bilinear(chi, 0, 1, Eigen::Vector3i(0, -1, 0));
        mixed_uc.set_mixed_bilinear(chi_inv, 3, 1, Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_bilinear(chi, 0, 1, Eigen::Vector3i(1, -1, 0));
        mixed_uc.set_mixed_bilinear(chi_inv, 3, 1, Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_bilinear(chi, 1, 1, Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_bilinear(chi_inv, 2, 1, Eigen::Vector3i(0, -1, 0));
        
        // Fe site 2 - 8 nearest Tm neighbors
        mixed_uc.set_mixed_bilinear(chi, 2, 2, Eigen::Vector3i(0, 0, -1));
        mixed_uc.set_mixed_bilinear(chi_inv, 1, 2, Eigen::Vector3i(0, -1, 0));
        mixed_uc.set_mixed_bilinear(chi, 0, 2, Eigen::Vector3i(0, -1, -1));
        mixed_uc.set_mixed_bilinear(chi_inv, 3, 2, Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_bilinear(chi, 0, 2, Eigen::Vector3i(1, -1, -1));
        mixed_uc.set_mixed_bilinear(chi_inv, 3, 2, Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_bilinear(chi, 1, 2, Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_bilinear(chi_inv, 2, 2, Eigen::Vector3i(0, -1, -1));
        
        // Fe site 3 - 8 nearest Tm neighbors
        mixed_uc.set_mixed_bilinear(chi, 3, 3, Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_bilinear(chi_inv, 0, 3, Eigen::Vector3i(0, 0, -1));
        mixed_uc.set_mixed_bilinear(chi, 2, 3, Eigen::Vector3i(0, 0, -1));
        mixed_uc.set_mixed_bilinear(chi_inv, 1, 3, Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_bilinear(chi, 1, 3, Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_bilinear(chi_inv, 2, 3, Eigen::Vector3i(-1, 0, -1));
        mixed_uc.set_mixed_bilinear(chi, 0, 3, Eigen::Vector3i(0, -1, -1));
        mixed_uc.set_mixed_bilinear(chi_inv, 3, 3, Eigen::Vector3i(-1, 1, 0));
    }
    
    return mixed_uc;
}

UnitCell build_tmfeo3_fe(const SpinConfig& config) {
    const double Jai = config.get_param("J1ab", 4.92);
    const double Jbi = Jai;
    const double Jci = config.get_param("J1c", 4.92);
    const double J2ai = config.get_param("J2ab", 0.29);
    const double J2bi = J2ai;
    const double J2ci = config.get_param("J2c", 0.29);
    const double Ka = config.get_param("Ka", 0.0);
    const double Kc = config.get_param("Kc", -0.09);
    const double D1 = config.get_param("D1", 0.0);
    const double D2 = config.get_param("D2", 0.0);
    const double h = config.field_strength;
    
    // Use TmFeO3_Fe class from unitcell.h (already has structure)
    TmFeO3_Fe Fe_atoms(3);
    

    // Local frame transformation (following molecular_dynamic_TmFeO3.cpp exactly)
    std::array<std::array<double, 3>, 4> eta = {{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}};
    
    // Original exchange matrices in global frame
    std::array<std::array<double, 3>, 3> Ja_orig = {{{Jai, D2, -D1}, {-D2, Jai, 0}, {D1, 0, Jai}}};
    std::array<std::array<double, 3>, 3> Jb_orig = {{{Jbi, D2, -D1}, {-D2, Jbi, 0}, {D1, 0, Jbi}}};
    std::array<std::array<double, 3>, 3> Jc_orig = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};
    std::array<std::array<double, 3>, 3> J2a_orig = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    std::array<std::array<double, 3>, 3> J2b_orig = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    std::array<std::array<double, 3>, 3> J2c_orig = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};
    
    // Transform to local frames: J_local[i][j][a][b] = J_orig[a][b] * eta[i][a] * eta[j][b]
    std::array<std::array<std::array<std::array<double, 3>, 3>, 4>, 4> Ja, Jb, Jc, J2a, J2b, J2c;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    Ja[i][j][a][b] = Ja_orig[a][b] * eta[i][a] * eta[j][b];
                    Jb[i][j][a][b] = Jb_orig[a][b] * eta[i][a] * eta[j][b];
                    Jc[i][j][a][b] = Jc_orig[a][b] * eta[i][a] * eta[j][b];
                    J2a[i][j][a][b] = J2a_orig[a][b] * eta[i][a] * eta[j][b];
                    J2b[i][j][a][b] = J2b_orig[a][b] * eta[i][a] * eta[j][b];
                    J2c[i][j][a][b] = J2c_orig[a][b] * eta[i][a] * eta[j][b];
                }
            }
        }
    }
    
    // Convert to Eigen matrices for setting interactions
    auto to_eigen = [](const std::array<std::array<double, 3>, 3>& arr) {
        Eigen::Matrix3d mat;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                mat(i, j) = arr[i][j];
            }
        }
        return mat;
    };
    
    // Set Fe-Fe interactions (following exact bond pattern from legacy code)
    // In-plane interactions (J1 type)
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja[1][0]), 1, 0, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja[1][0]), 1, 0, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb[1][0]), 1, 0, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb[1][0]), 1, 0, Eigen::Vector3i(1, 0, 0));
    
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja[2][3]), 2, 3, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja[2][3]), 2, 3, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb[2][3]), 2, 3, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb[2][3]), 2, 3, Eigen::Vector3i(1, 0, 0));
    
    // Next nearest neighbor (J2 type, along a, b, and c axes - same sublattice)
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[0][0]), 0, 0, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[0][0]), 0, 0, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][0]), 0, 0, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[1][1]), 1, 1, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[1][1]), 1, 1, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][1]), 1, 1, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[2][2]), 2, 2, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[2][2]), 2, 2, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[2][2]), 2, 2, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[3][3]), 3, 3, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[3][3]), 3, 3, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[3][3]), 3, 3, Eigen::Vector3i(0, 0, 1));
    
    // Out of plane interactions (J1 type along c-axis)
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[0][3]), 0, 3, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[0][3]), 0, 3, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[1][2]), 1, 2, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[1][2]), 1, 2, Eigen::Vector3i(0, 0, 1));
    
    // J2 out-of-plane interactions (cross-sublattice diagonal)
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(-1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(-1, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(0, 1, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(-1, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[0][2]), 0, 2, Eigen::Vector3i(-1, 1, 1));
    
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(0, -1, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(1, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2c[1][3]), 1, 3, Eigen::Vector3i(1, -1, 1));
    
    
    // Single ion anisotropy (same in all local frames)
    Eigen::MatrixXd K_mat = Eigen::MatrixXd::Zero(3, 3);
    K_mat(0, 0) = Ka;
    K_mat(2, 2) = Kc;
    Fe_atoms.set_onsite_interaction(K_mat, 0);
    Fe_atoms.set_onsite_interaction(K_mat, 1);
    Fe_atoms.set_onsite_interaction(K_mat, 2);
    Fe_atoms.set_onsite_interaction(K_mat, 3);
    
    // External magnetic field
    Eigen::Vector3d field;
    field << config.field_direction[0] * h,
             config.field_direction[1] * h,
             config.field_direction[2] * h;
    Fe_atoms.set_field(field, 0);
    Fe_atoms.set_field(field, 1);
    Fe_atoms.set_field(field, 2);
    Fe_atoms.set_field(field, 3);
    
    return Fe_atoms;
}

UnitCell build_tmfeo3_tm(const SpinConfig& config) {
    // Crystal field parameters for Tm ions
    const double e1 = config.get_param("e1", 0.97);
    const double e2 = config.get_param("e2", 3.97);
    
    // Tm-Tm diagonal bilinear coupling (nearest neighbor) - each component couples λ_a ⊗ λ_a only
    const double Jtm_1 = config.get_param("Jtm_1", 0.0);
    const double Jtm_2 = config.get_param("Jtm_2", 0.0);
    const double Jtm_3 = config.get_param("Jtm_3", 0.0);
    const double Jtm_4 = config.get_param("Jtm_4", 0.0);
    const double Jtm_5 = config.get_param("Jtm_5", 0.0);
    const double Jtm_6 = config.get_param("Jtm_6", 0.0);
    const double Jtm_7 = config.get_param("Jtm_7", 0.0);
    const double Jtm_8 = config.get_param("Jtm_8", 0.0);
    
    // Use TmFeO3_Tm class from unitcell.h (already has structure)
    // SU(3) has 8-dimensional spin space (Gell-Mann basis)
    TmFeO3_Tm Tm_atoms(8);
    
    // Set energy splitting (field in SU(3) space)
    // For a 3-level system with energies 0, e1, e2, the Hamiltonian is H = diag(0, e1, e2)
    // In the Gell-Mann basis with S_a = λ_a/2, we have E = -B · S = -(1/2) B · λ
    // Matching diagonal elements gives:
    //   B_3 = e1 (coefficient of λ_3)
    //   B_8 = (2*e2 - e1) / sqrt(3) (coefficient of λ_8)
    const double tm_alpha_scale = config.get_param("tm_alpha_scale", 1.0);
    const double tm_beta_scale = config.get_param("tm_beta_scale", 1.0);
    double alpha = e1 * tm_alpha_scale;
    double beta = (2.0 * e2 - e1) / sqrt(3.0) * tm_beta_scale;
    Eigen::VectorXd tm_field(8);
    tm_field << 0, 0, alpha, 0, 0, 0, 0, beta;
    
    Tm_atoms.set_field(tm_field, 0);
    Tm_atoms.set_field(tm_field, 1);
    Tm_atoms.set_field(tm_field, 2);
    Tm_atoms.set_field(tm_field, 3);
    
    // Tm-Tm diagonal bilinear interactions (nearest neighbor)
    // Diagonal in Gell-Mann space: J_a * λ_a ⊗ λ_a (no cross terms)
    // Nearest neighbors based on real-space positions:
    //   Tm0 (0.02111, 0.92839, 0.75) ↔ Tm2 (0.47889, 0.42839, 0.75): d ≈ 0.68
    //   Tm1 (0.52111, 0.57161, 0.25) ↔ Tm3 (0.97889, 0.07161, 0.25): d ≈ 0.68
    if (Jtm_1 != 0.0 || Jtm_2 != 0.0 || Jtm_3 != 0.0 || Jtm_4 != 0.0 ||
        Jtm_5 != 0.0 || Jtm_6 != 0.0 || Jtm_7 != 0.0 || Jtm_8 != 0.0) {
        Eigen::MatrixXd J_tm_mat = Eigen::MatrixXd::Zero(8, 8);
        J_tm_mat(0, 0) = Jtm_1;
        J_tm_mat(1, 1) = Jtm_2;
        J_tm_mat(2, 2) = Jtm_3;
        J_tm_mat(3, 3) = Jtm_4;
        J_tm_mat(4, 4) = Jtm_5;
        J_tm_mat(5, 5) = Jtm_6;
        J_tm_mat(6, 6) = Jtm_7;
        J_tm_mat(7, 7) = Jtm_8;
        
        // Tm0 ↔ Tm2 nearest neighbors (z=0.75 plane, d ≈ 0.68)
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 2, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 2, Eigen::Vector3i(0, 1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 2, Eigen::Vector3i(-1, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 2, Eigen::Vector3i(-1, 1, 0));
        
        // Tm1 ↔ Tm3 nearest neighbors (z=0.25 plane, d ≈ 0.68)
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(0, 1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(1, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(1, 1, 0));
        
        // Out-of-plane nearest neighbors (between z=0.75 and z=0.25 planes)
        // Tm0 (0.02111, 0.92839, 0.75) ↔ Tm3 (0.97889, 0.07161, 0.25)
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(0, 1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(0, 0, 1));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(0, 1, 1));
        
        // Tm2 (0.47889, 0.42839, 0.75) ↔ Tm1 (0.52111, 0.57161, 0.25)
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, 0, 1));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, -1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, -1, 1));
    }
    
    return Tm_atoms;
}

