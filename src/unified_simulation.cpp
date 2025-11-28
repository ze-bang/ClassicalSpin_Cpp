#include "unified_config.h"
#include "unitcell.h"
#include "lattice.h"
#include "mixed_lattice.h"
#include <mpi.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <filesystem>
#include <fstream>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using namespace std;

// ============================================================================
// UNIT CELL BUILDERS
// ============================================================================

/**
 * Build BCAO honeycomb unit cell
 */
UnitCell build_bcao_honeycomb(const UnifiedConfig& config) {
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

/**
 * Build Kitaev honeycomb unit cell
 */
UnitCell build_kitaev_honeycomb(const UnifiedConfig& config) {
    const double K = config.get_param("K", -1.0);
    const double Gamma = config.get_param("Gamma", 0.25);
    const double Gammap = config.get_param("Gammap", -0.02);
    const double J = config.get_param("J", 0.0);
    const double h = config.get_param("h", 0.7);
    
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
    
    // Set magnetic field
    Eigen::Vector3d field;
    field << config.field_strength * config.field_direction[0],
             config.field_strength * config.field_direction[1],
             config.field_strength * config.field_direction[2];
    
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    
    return atoms;
}

/**
 * Build pyrochlore unit cell
 */
UnitCell build_pyrochlore(const UnifiedConfig& config) {
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

/**
 * Build TmFeO3 mixed unit cell following molecular_dynamic_TmFeO3.cpp pattern
 */
MixedUnitCell build_tmfeo3(const UnifiedConfig& config) {
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
    const double chii = config.get_param("chii", 0.05);
    const double e1 = config.get_param("e1", 0.97);
    const double e2 = config.get_param("e2", 3.97);
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
    
    // Next nearest neighbor (J2 type, along a and b axes)
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[0][0]), 0, 0, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[0][0]), 0, 0, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[1][1]), 1, 1, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[1][1]), 1, 1, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[2][2]), 2, 2, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[2][2]), 2, 2, Eigen::Vector3i(0, 1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2a[3][3]), 3, 3, Eigen::Vector3i(1, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(J2b[3][3]), 3, 3, Eigen::Vector3i(0, 1, 0));
    
    // Out of plane interactions (J1 type along c-axis)
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[0][3]), 0, 3, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[0][3]), 0, 3, Eigen::Vector3i(0, 0, 1));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[1][2]), 1, 2, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jc[1][2]), 1, 2, Eigen::Vector3i(0, 0, 1));
    
    // J2 out-of-plane interactions
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
    
    // Tm atoms - set energy splitting (field in SU(3) space)
    // Tm field scaling factors from config
    const double tm_alpha_scale = config.get_param("tm_alpha_scale", 1.0);
    const double tm_beta_scale = config.get_param("tm_beta_scale", 1.0);
    double alpha = e1 * tm_alpha_scale;
    double beta = sqrt(3.0) / 3.0 * (2.0 * e2 - e1) * tm_beta_scale;
    Eigen::VectorXd tm_field(8);
    tm_field << 0, 0, alpha, 0, 0, 0, 0, beta;
    
    Tm_atoms.set_field(tm_field, 0);
    Tm_atoms.set_field(tm_field, 1);
    Tm_atoms.set_field(tm_field, 2);
    Tm_atoms.set_field(tm_field, 3);
    
    // Create mixed unit cell
    MixedUnitCell mixed_uc(Fe_atoms, Tm_atoms);
    
    // Set Fe-Tm bilinear coupling (following exact pattern from legacy code)
    if (chii != 0.0) {
        Eigen::MatrixXd chi = Eigen::MatrixXd::Zero(8, 3);
        chi(1, 0) = chii; chi(1, 1) = chii; chi(1, 2) = chii;
        chi(4, 0) = chii; chi(4, 1) = chii; chi(4, 2) = chii;
        chi(6, 0) = chii; chi(6, 1) = chii; chi(6, 2) = chii;
        
        Eigen::MatrixXd chi_inv = Eigen::MatrixXd::Zero(8, 3);
        chi_inv(1, 0) = chii; chi_inv(1, 1) = chii; chi_inv(1, 2) = chii;
        chi_inv(4, 0) = -chii; chi_inv(4, 1) = -chii; chi_inv(4, 2) = -chii;
        chi_inv(6, 0) = -chii; chi_inv(6, 1) = -chii; chi_inv(6, 2) = -chii;
        
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

// ============================================================================
// SIMULATION RUNNERS
// ============================================================================

/**
 * Run simulated annealing
 */
void run_simulated_annealing(Lattice& lattice, const UnifiedConfig& config, int rank) {
    if (rank == 0) {
        cout << "Running simulated annealing..." << endl;
    }
    
    filesystem::create_directories(config.output_dir);
    
    lattice.simulated_annealing(
        config.T_start,
        config.T_end,
        config.annealing_steps,
        config.overrelaxation_rate,
        config.use_twist_boundary,
        true,  // gaussian_move
        config.cooling_rate,
        config.output_dir,
        true   // save_observables
    );
    
    // Save final configuration
    lattice.save_positions(config.output_dir + "/positions.txt");
    lattice.save_spin_config(config.output_dir + "/spins.txt");
    
    if (rank == 0) {
        ofstream energy_file(config.output_dir + "/final_energy.txt");
        energy_file << "Energy Density: " << lattice.energy_density() << "\n";
        energy_file.close();
        
        cout << "Simulated annealing completed. Final energy: " << lattice.energy_density() << endl;
    }
}

/**
 * Run parallel tempering
 */
void run_parallel_tempering(Lattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running parallel tempering with " << size << " replicas..." << endl;
    }
    
    filesystem::create_directories(config.output_dir);
    
    // Generate temperature ladder
    vector<double> temps(size);
    for (int i = 0; i < size; ++i) {
        double log_T = log10(config.T_start) + 
                      (log10(config.T_end) - log10(config.T_start)) * i / (size - 1);
        temps[i] = pow(10, log_T);
    }
    
    lattice.parallel_tempering(
        temps,
        config.annealing_steps,
        config.annealing_steps,
        config.overrelaxation_rate,
        config.pt_exchange_frequency,
        config.probe_rate,
        config.output_dir,
        config.ranks_to_write,
        true  // gaussian_move
    );
    
    if (rank == 0) {
        cout << "Parallel tempering completed." << endl;
    }
}

/**
 * Run molecular dynamics
 */
void run_molecular_dynamics(Lattice& lattice, const UnifiedConfig& config, int rank) {
    if (rank == 0) {
        cout << "Running molecular dynamics..." << endl;
        if (config.use_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
            // Check CUDA device
            int device_count;
            cudaGetDeviceCount(&device_count);
            if (device_count > 0) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);
                cout << "Using GPU: " << prop.name << endl;
                cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
            } else {
                cout << "Warning: No CUDA devices found, falling back to CPU" << endl;
            }
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
    filesystem::create_directories(config.output_dir);
    
    // First equilibrate at low temperature
    if (rank == 0) {
        cout << "Equilibrating system..." << endl;
    }
    lattice.simulated_annealing(
        config.T_start,
        config.T_end,
        config.annealing_steps,
        0,
        config.use_twist_boundary
    );
    
    // Run MD
    if (rank == 0) {
        cout << "Starting MD integration..." << endl;
        cout << "Time range: " << config.md_time_start << " -> " << config.md_time_end << endl;
        cout << "Timestep: " << config.md_timestep << endl;
        cout << "Integration method: " << config.md_integrator << endl;
    }
    
    const size_t save_interval = static_cast<size_t>(config.get_param("md_save_interval", 100.0));
    
    lattice.molecular_dynamics(
        config.md_time_start,
        config.md_time_end,
        config.md_timestep,
        config.output_dir,
        save_interval,
        config.md_integrator,
        config.use_gpu
    );
    
    if (rank == 0) {
        cout << "Molecular dynamics completed." << endl;
        cout << "Results saved to: " << config.output_dir << "/trajectory.h5" << endl;
    }
}

/**
 * Run pump-probe experiment
 */
void run_pump_probe(Lattice& lattice, const UnifiedConfig& config, int rank) {
    if (rank == 0) {
        cout << "Running pump-probe simulation..." << endl;
        if (config.use_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available" << endl;
#endif
        }
    }
    
    filesystem::create_directories(config.output_dir);
    
    // First equilibrate
    if (rank == 0) {
        cout << "Equilibrating system..." << endl;
    }
    lattice.simulated_annealing(
        config.T_start,
        config.T_end,
        config.annealing_steps,
        0,
        config.use_twist_boundary
    );
    
    // Setup pump pulse
    if (rank == 0) {
        cout << "Setting up pump-probe pulses..." << endl;
        cout << "Pump: t=" << config.pump_time << ", A=" << config.pump_amplitude 
             << ", w=" << config.pump_width << ", f=" << config.pump_frequency << endl;
        cout << "Probe: t=" << config.probe_time << ", A=" << config.probe_amplitude << endl;
    }
    
    // Note: Pump-probe functionality needs to be implemented in lattice.h
    // For now, we run standard MD and note that pulse implementation is needed
    if (rank == 0) {
        cout << "Warning: Full pump-probe implementation requires pulse field setup in lattice class" << endl;
        cout << "Running standard MD for now..." << endl;
    }
    
    // Run MD
    const size_t save_interval = static_cast<size_t>(config.get_param("md_save_interval", 100.0));
    
    lattice.molecular_dynamics(
        config.md_time_start,
        config.md_time_end,
        config.md_timestep,
        config.output_dir,
        save_interval,
        config.md_integrator,
        config.use_gpu
    );
    
    if (rank == 0) {
        cout << "Pump-probe simulation completed." << endl;
        cout << "Results saved to: " << config.output_dir << "/trajectory.h5" << endl;
    }
}

/**
 * Run simulated annealing for mixed lattice
 */
void run_simulated_annealing_mixed(MixedLattice& lattice, const UnifiedConfig& config, int rank) {
    if (rank == 0) {
        cout << "Running simulated annealing on mixed lattice..." << endl;
    }
    
    filesystem::create_directories(config.output_dir);
    
    lattice.simulated_annealing(
        config.T_start,
        config.T_end,
        config.annealing_steps,
        true,  // gaussian_move
        config.cooling_rate,
        config.output_dir,
        true   // save_observables
    );
    
    // Save final configuration
    lattice.save_positions(config.output_dir + "/positions.txt");
    lattice.save_spin_config(config.output_dir + "/spins.txt");
    
    if (rank == 0) {
        ofstream energy_file(config.output_dir + "/final_energy.txt");
        energy_file << "Energy Density: " << lattice.energy_density() << "\n";
        energy_file.close();
        
        cout << "Simulated annealing completed. Final energy: " << lattice.energy_density() << endl;
    }
}

/**
 * Run parallel tempering for mixed lattice
 */
void run_parallel_tempering_mixed(MixedLattice& lattice, const UnifiedConfig& config, int rank, int size) {
    if (rank == 0) {
        cout << "Running parallel tempering on mixed lattice with " << size << " replicas..." << endl;
    }
    
    filesystem::create_directories(config.output_dir);
    
    // Generate temperature ladder
    vector<double> temps(size);
    for (int i = 0; i < size; ++i) {
        double log_T = log10(config.T_start) + 
                      (log10(config.T_end) - log10(config.T_start)) * i / (size - 1);
        temps[i] = pow(10, log_T);
    }
    
    lattice.parallel_tempering(
        temps,
        config.annealing_steps,
        config.annealing_steps,
        0,  // overrelaxation_rate
        config.pt_exchange_frequency,
        config.probe_rate,
        config.output_dir,
        config.ranks_to_write,
        true  // gaussian_move
    );
    
    if (rank == 0) {
        cout << "Parallel tempering completed." << endl;
    }
}

/**
 * Run molecular dynamics for mixed lattice
 */
void run_molecular_dynamics_mixed(MixedLattice& lattice, const UnifiedConfig& config, int rank) {
    if (rank == 0) {
        cout << "Running molecular dynamics on mixed lattice..." << endl;
    }
    
    filesystem::create_directories(config.output_dir);
    
    // Equilibrate
    if (rank == 0) {
        cout << "Equilibrating system..." << endl;
    }
    lattice.simulated_annealing(
        config.T_start,
        config.T_end,
        config.annealing_steps,
        true
    );
    
    // Run MD
    if (rank == 0) {
        cout << "Starting MD integration..." << endl;
    }
    
    // Create pump fields
    vector<SpinVector> pump_su2(lattice.lattice_size_SU2, SpinVector::Zero(3));
    vector<SpinVector> pump_su3(lattice.lattice_size_SU3, SpinVector::Zero(8));
    vector<SpinVector> probe_su2(lattice.lattice_size_SU2, SpinVector::Zero(3));
    vector<SpinVector> probe_su3(lattice.lattice_size_SU3, SpinVector::Zero(8));
    
    auto trajectory = lattice.M_B_t(
        pump_su2, pump_su3, config.pump_time,
        config.pump_amplitude, config.pump_width, config.pump_frequency,
        config.probe_amplitude, config.probe_width, config.probe_frequency,
        config.md_time_start, config.md_time_end, config.md_timestep,
        config.md_integrator, config.use_gpu
    );
    
    // Save trajectory
    if (rank == 0) {
        ofstream traj_file(config.output_dir + "/trajectory.txt");
        for (const auto& [t, mag_data] : trajectory) {
            traj_file << t << " "
                     << mag_data.first[0].transpose() << " "  // SU2 mag antiferro
                     << mag_data.first[1].transpose() << " "  // SU2 mag local
                     << mag_data.first[2].transpose() << " "  // SU2 mag global
                     << mag_data.second[0].transpose() << " " // SU3 mag antiferro
                     << mag_data.second[1].transpose() << " " // SU3 mag local
                     << mag_data.second[2].transpose() << "\n"; // SU3 mag global
        }
        traj_file.close();
        cout << "Molecular dynamics completed." << endl;
    }
}

/**
 * Run pump-probe experiment for mixed lattice
 */
void run_pump_probe_mixed(MixedLattice& lattice, const UnifiedConfig& config, int rank) {
    if (rank == 0) {
        cout << "Running pump-probe simulation on mixed lattice..." << endl;
    }
    
    filesystem::create_directories(config.output_dir);
    
    // Equilibrate
    if (rank == 0) {
        cout << "Equilibrating system..." << endl;
    }
    lattice.simulated_annealing(
        config.T_start,
        config.T_end,
        config.annealing_steps,
        true
    );
    
    // Setup pump and probe fields
    if (rank == 0) {
        cout << "Setting up pump-probe pulses..." << endl;
    }
    
    // Create pump fields for SU2 (Fe) and SU3 (Tm)
    vector<SpinVector> pump_su2(lattice.lattice_size_SU2);
    vector<SpinVector> pump_su3(lattice.lattice_size_SU3);
    
    SpinVector pump_dir_su2(3);
    pump_dir_su2 << config.pump_direction[0], config.pump_direction[1], config.pump_direction[2];
    pump_dir_su2.normalize();
    
    // For SU3, pump direction in Gell-Mann basis (default: λ3)
    const int su3_pump_component = static_cast<int>(config.get_param("su3_pump_component", 2.0));
    SpinVector pump_dir_su3 = SpinVector::Zero(8);
    if (su3_pump_component >= 0 && su3_pump_component < 8) {
        pump_dir_su3(su3_pump_component) = 1.0;
    } else {
        pump_dir_su3(2) = 1.0;  // Default to λ3
    }
    
    for (size_t i = 0; i < lattice.lattice_size_SU2; ++i) {
        pump_su2[i] = config.pump_amplitude * pump_dir_su2;
    }
    for (size_t i = 0; i < lattice.lattice_size_SU3; ++i) {
        pump_su3[i] = config.pump_amplitude * pump_dir_su3;
    }
    
    // Run MD with pump-probe
    if (rank == 0) {
        cout << "Running pump-probe dynamics..." << endl;
    }
    
    auto trajectory = lattice.M_B_t(
        pump_su2, pump_su3, config.pump_time,
        config.pump_amplitude, config.pump_width, config.pump_frequency,
        config.probe_amplitude, config.probe_width, config.probe_frequency,
        config.md_time_start, config.md_time_end, config.md_timestep,
        config.md_integrator, config.use_gpu
    );
    
    // Save trajectory
    if (rank == 0) {
        ofstream traj_file(config.output_dir + "/pump_probe_trajectory.txt");
        for (const auto& [t, mag_data] : trajectory) {
            traj_file << t << " "
                     << mag_data.first[0].transpose() << " "  // SU2 mag antiferro
                     << mag_data.first[1].transpose() << " "  // SU2 mag local
                     << mag_data.first[2].transpose() << " "  // SU2 mag global
                     << mag_data.second[0].transpose() << " " // SU3 mag antiferro
                     << mag_data.second[1].transpose() << " " // SU3 mag local
                     << mag_data.second[2].transpose() << "\n"; // SU3 mag global
        }
        traj_file.close();
        cout << "Pump-probe simulation completed." << endl;
    }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char** argv) {
    // Initialize MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    string config_file = "simulation.param";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    // Load configuration
    UnifiedConfig config;
    try {
        config = UnifiedConfig::from_file(config_file);
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Error loading configuration: " << e.what() << endl;
            cerr << "Usage: " << argv[0] << " [config_file]\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Validate configuration
    if (!config.validate()) {
        if (rank == 0) {
            cerr << "Configuration validation failed\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Print configuration on rank 0
    if (rank == 0) {
        config.print();
    }
    
    // Build system and run simulation
    try {
        if (config.system == SystemType::TMFEO3) {
            // Mixed SU(2)+SU(3) system
            if (rank == 0) {
                cout << "\nBuilding TmFeO3 system..." << endl;
            }
            
            auto mixed_uc = build_tmfeo3(config);
            MixedLattice mixed_lattice(mixed_uc, 
                                       config.lattice_size[0],
                                       config.lattice_size[1],
                                       config.lattice_size[2],
                                       config.spin_length,
                                       config.spin_length);
            
            // Initialize spins
            if (config.use_ferromagnetic_init) {
                SpinVector dir_su2(3);
                dir_su2 << config.ferromagnetic_direction[0],
                          config.ferromagnetic_direction[1],
                          config.ferromagnetic_direction[2];
                SpinVector dir_su3 = SpinVector::Zero(8);
                const int su3_init_component = static_cast<int>(config.get_param("su3_init_component", 2.0));
                if (su3_init_component >= 0 && su3_init_component < 8) {
                    dir_su3(su3_init_component) = 1.0;
                } else {
                    dir_su3(2) = 1.0;  // Default to λ3
                }
                mixed_lattice.init_ferromagnetic(dir_su2, dir_su3);
            } else if (!config.initial_spin_config.empty()) {
                mixed_lattice.load_spin_config(config.initial_spin_config);
            } else {
                mixed_lattice.init_random();
            }
            
            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_mixed(mixed_lattice, config, rank);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_mixed(mixed_lattice, config, rank);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_mixed(mixed_lattice, config, rank);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not supported for mixed lattice\n";
                    }
                    break;
            }
        } else {
            // Regular SU(2) system
            if (rank == 0) {
                cout << "\nBuilding unit cell..." << endl;
            }
            
            UnitCell uc(3, 2);  // Default placeholder
            
            switch (config.system) {
                case SystemType::HONEYCOMB_BCAO:
                    uc = build_bcao_honeycomb(config);
                    break;
                case SystemType::HONEYCOMB_KITAEV:
                    uc = build_kitaev_honeycomb(config);
                    break;
                case SystemType::PYROCHLORE:
                    uc = build_pyrochlore(config);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "System type not implemented\n";
                    }
                    MPI_Finalize();
                    return 1;
            }
            
            Lattice lattice(uc, 
                          config.lattice_size[0],
                          config.lattice_size[1],
                          config.lattice_size[2],
                          config.spin_length);
            
            // Initialize spins
            if (config.use_ferromagnetic_init) {
                // Initialize all spins in same direction
                SpinVector dir(3);
                dir << config.ferromagnetic_direction[0],
                      config.ferromagnetic_direction[1],
                      config.ferromagnetic_direction[2];
                dir.normalize();
                dir *= config.spin_length;
                for (size_t i = 0; i < lattice.lattice_size; ++i) {
                    lattice.spins[i] = dir;
                }
            } else if (!config.initial_spin_config.empty()) {
                lattice.load_spin_config(config.initial_spin_config);
            }
            // else: spins already initialized randomly in constructor
            
            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing(lattice, config, rank);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering(lattice, config, rank, size);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics(lattice, config, rank);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe(lattice, config, rank);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not implemented\n";
                    }
                    break;
            }
        }
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Error during simulation: " << e.what() << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Finalize MPI
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
    
    if (rank == 0) {
        cout << "\n=== Simulation completed successfully ===" << endl;
    }
    
    return 0;
}
