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
    // In pyrochlore, second nearest neighbors are same-sublattice atoms
    // connected along the lattice vector directions.
    // Pyrochlore positions from unitcell.h:
    //   Atom 0: (0.125, 0.125, 0.125)
    //   Atom 1: (0.125, -0.125, -0.125)
    //   Atom 2: (-0.125, 0.125, -0.125)
    //   Atom 3: (-0.125, -0.125, 0.125)
    // Lattice vectors: (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    // Each sublattice has 6 second nearest neighbors at offsets ±a₁, ±a₂, ±a₃
    Eigen::Matrix3d J2_mat = Eigen::Matrix3d::Identity() * J2;
    
    // Sublattice 0 -> sublattice 0 (same sublattice in neighboring cells)
    atoms.set_bilinear_interaction(J2_mat, 0, 0, Eigen::Vector3i(1, 0, 0));
    atoms.set_bilinear_interaction(J2_mat, 0, 0, Eigen::Vector3i(0, 1, 0));
    atoms.set_bilinear_interaction(J2_mat, 0, 0, Eigen::Vector3i(0, 0, 1));
    
    // Sublattice 1 -> sublattice 1
    atoms.set_bilinear_interaction(J2_mat, 1, 1, Eigen::Vector3i(1, 0, 0));
    atoms.set_bilinear_interaction(J2_mat, 1, 1, Eigen::Vector3i(0, 1, 0));
    atoms.set_bilinear_interaction(J2_mat, 1, 1, Eigen::Vector3i(0, 0, 1));
    
    // Sublattice 2 -> sublattice 2
    atoms.set_bilinear_interaction(J2_mat, 2, 2, Eigen::Vector3i(1, 0, 0));
    atoms.set_bilinear_interaction(J2_mat, 2, 2, Eigen::Vector3i(0, 1, 0));
    atoms.set_bilinear_interaction(J2_mat, 2, 2, Eigen::Vector3i(0, 0, 1));
    
    // Sublattice 3 -> sublattice 3
    atoms.set_bilinear_interaction(J2_mat, 3, 3, Eigen::Vector3i(1, 0, 0));
    atoms.set_bilinear_interaction(J2_mat, 3, 3, Eigen::Vector3i(0, 1, 0));
    atoms.set_bilinear_interaction(J2_mat, 3, 3, Eigen::Vector3i(0, 0, 1));
    
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

MixedUnitCell build_tmfeo3(const SpinConfig& config) {
    const double Jai = config.get_param("J1ab", 4.74);
    const double Jbi = Jai;
    const double Jci = config.get_param("J1c", 5.15);
    const double J2ai = config.get_param("J2ab", 0.15);
    const double J2bi = J2ai;
    const double J2ci = config.get_param("J2c", 0.30);
    const double Ka = config.get_param("Ka", -0.16221);
    const double Kb = config.get_param("Kb", 0.0);
    const double Kc = config.get_param("Kc", -0.18318);
    const double D1 = config.get_param("D1", 0.12);
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
    // =========================================================================
    // Orbit-dependent Fe(3d)-Tm(4f) superexchange scale factors
    // =========================================================================
    //
    // PHYSICS: The Fe-Tm bilinear coupling is mediated by superexchange through
    // oxygen: Fe(3d^5) — O(2p) — Tm(4f^12). Each Fe has 8 nearest Tm neighbors
    // falling into 4 crystallographic orbits with distinct real-space distances:
    //
    //   Orbit 1:  d = 3.054 Å  (d_frac = 0.497, mostly along b+c, shortest)
    //   Orbit 2:  d = 3.179 Å  (d_frac = 0.545, mostly along a+c)
    //   Orbit 3:  d = 3.357 Å  (d_frac = 0.582, mostly along a+c)
    //   Orbit 4:  d = 3.711 Å  (d_frac = 0.624, mostly along b+c, longest)
    //
    //   (Using Pbnm lattice constants a=5.2534, b=5.5707, c=7.6076 Å)
    //   Distance spread: 21.5% from shortest to longest orbit.
    //
    // The superexchange strength scales as J_SE ∝ (t_pd · t_pf)^2 / Δ^2 where:
    //   t_pd ∝ d^{-3.5}  (Harrison 3d-2p transfer integral)
    //   t_pf ∝ d^{-5}    (4f-2p transfer integral, steeper due to 4f contraction)
    //
    // For 3d-3d systems: J ∝ d^{-7}  (n=7, well-established Harrison scaling)
    // For 4f-3d systems: the 4f orbital is much more contracted than 3d, giving a
    //   steeper effective exponent. With t_pf ∝ d^{-5}, the total superexchange
    //   through a single O bridge gives J ∝ t_pd · t_pf ∝ d^{-(3.5+5)} = d^{-8.5}.
    //   In practice n ≈ 8–12 for 4f-3d RE orthoferrites (path geometry, covalency,
    //   and multi-orbital effects modify the bare estimate).
    //
    // WHY THIS MATTERS: Pbnm inversion symmetry pairs every chi bond with a
    // chi_inv bond (sign-flipped on λ5/λ7 columns). In the minimal chi/chi_inv
    // bond pattern implemented below, the static q=0 field on λ5 and λ7 cancels
    // EXACTLY orbit-by-orbit for a uniform Gamma_2 background, because each chi/
    // chi_inv pair samples Fe sublattices with the same local-frame sign.
    // Distance-dependent orbit scales s_k change the overall bond weights, but do
    // not lift this static q=0 cancellation. The λ5/λ7 channels become active only
    // for bond-resolved finite-q dynamics or if the microscopic bond ansatz is
    // enlarged beyond the minimal inversion-paired pattern. The λ2 channel is
    // UNAFFECTED in the static q=0 limit (all paired contributions add).
    //
    // PRESETS (s_i = (d_min/d_i)^n, normalized to mean = 1):
    //   Conservative (n=5,  3d-3d-like):   1.42, 1.16, 0.88, 0.54
    //   Moderate     (n=8,  ~4f-3d est.):  1.66, 1.21, 0.78, 0.35
    //   Aggressive   (n=10, steep 4f):     1.82, 1.22, 0.71, 0.26
    //
    // Default = 1.0 (uniform) recovers the original code behavior.
    // =========================================================================
    const double chi_orbit1_scale = config.get_param("chi_orbit1_scale", 1.0);  // d=3.054 Å
    const double chi_orbit2_scale = config.get_param("chi_orbit2_scale", 1.0);  // d=3.179 Å
    const double chi_orbit3_scale = config.get_param("chi_orbit3_scale", 1.0);  // d=3.357 Å
    const double chi_orbit4_scale = config.get_param("chi_orbit4_scale", 1.0);  // d=3.711 Å
    // =========================================================================
    // Anisotropy-modulation trilinear coupling: λ^a_Tm · S^b_Fe · S^c_Fe
    // =========================================================================
    // From tmfeo3_notes.tex Eq.10: key mechanism for qFM-linked 2DCS peaks.
    // A1+ sector (mirror-even Tm operators): λ1, λ3, λ8 couple to (S_z² - S_x²)
    //   → projects to I_{A1}^Fe = G_z² - F_x² at q=0
    // A2+ sector (mirror-odd Tm operators): λ4, λ6 couple to S_x·S_z
    //   → projects to I_{A2}^Fe = F_x·G_z at q=0
    // Inversion constraint: chi-type bonds carry full W, chi_inv-type bonds
    //   have A2+ components (v4, v6) sign-flipped.
    // Convention: H += Σ_abc K[a](b,c) S_Fe^a S_Fe^b λ_Tm^c (on-site Fe bilinear)
    //   u-params give coefficient of (S_z²-S_x²)·λ^a, v-params give coeff of 2·S_x·S_z·λ^a
    const double u1 = config.get_param("u1", 0.0);   // λ1 (A1+) aniso-mod coupling
    const double u3 = config.get_param("u3", 0.0);   // λ3 (A1+) aniso-mod coupling
    const double u8 = config.get_param("u8", 0.0);   // λ8 (A1+) aniso-mod coupling
    const double v4 = config.get_param("v4", 0.0);   // λ4 (A2+) aniso-mod coupling
    const double v6 = config.get_param("v6", 0.0);   // λ6 (A2+) aniso-mod coupling
    // =========================================================================
    // General on-site trilinear Fe bilinear channels (tmfeo3_notes.tex §5.3)
    // =========================================================================
    // The full on-site trilinear has 5 Tm-even channels × 6 symmetric Fe bilinears
    // = 30 independent parameters per orbit. The u/v shorthand above covers only
    // the Γ2-dominant channels (S_z²-S_x² and S_x·S_z). These general params
    // ADD to the u/v base, enabling the full symmetric Fe quadrupole tensor:
    //   H_aniso = Σ_a W_a^{bc} λ^a S^b S^c  (a ∈ {1,3,4,6,8}; bc symmetric)
    // Naming: W{channel}_{bilinear}, e.g. W1_yy couples λ1 to S_y².
    // Params here are VALUES placed in K[b](c, a_idx) (and symmetrized in bc).
    // Diagonal (bb): effective coupling = W * S_b².  Off-diag (bc): eff = 2W * S_b S_c.
    struct TrilinearChannel {
        double xx, yy, zz, xy, xz, yz;
    };
    // Antisymmetric Fe bilinear channel: S_i^a S_{i'}^b - S_i^b S_{i'}^a (DM-like)
    // Only meaningful for inter-site (i ≠ i') trilinear; identically zero on-site.
    struct AntiTrilinearChannel {
        double xy, xz, yz;
    };
    auto read_tri_ch = [&](const std::string& pfx, double u_zzmxx, double v_xz) -> TrilinearChannel {
        // u_zzmxx from old u-param: adds +u to zz, -u to xx
        // v_xz from old v-param: adds +v to xz
        return {
            config.get_param(pfx + "_xx", 0.0) - u_zzmxx,   // S_x² coupling
            config.get_param(pfx + "_yy", 0.0),               // S_y² coupling
            config.get_param(pfx + "_zz", 0.0) + u_zzmxx,    // S_z² coupling
            config.get_param(pfx + "_xy", 0.0),               // S_x S_y coupling
            config.get_param(pfx + "_xz", 0.0) + v_xz,       // S_x S_z coupling
            config.get_param(pfx + "_yz", 0.0)                // S_y S_z coupling
        };
    };
    auto read_anti_ch = [&](const std::string& pfx) -> AntiTrilinearChannel {
        return {
            config.get_param(pfx + "_Axy", 0.0),   // [S_x S'_y - S_y S'_x] coupling
            config.get_param(pfx + "_Axz", 0.0),   // [S_x S'_z - S_z S'_x] coupling
            config.get_param(pfx + "_Ayz", 0.0)    // [S_y S'_z - S_z S'_y] coupling
        };
    };
    TrilinearChannel W1_ch = read_tri_ch("W1", u1, 0.0);  // λ1 (A1+)
    TrilinearChannel W3_ch = read_tri_ch("W3", u3, 0.0);  // λ3 (A1+)
    TrilinearChannel W4_ch = read_tri_ch("W4", 0.0, v4);   // λ4 (A2+)
    TrilinearChannel W6_ch = read_tri_ch("W6", 0.0, v6);   // λ6 (A2+)
    TrilinearChannel W8_ch = read_tri_ch("W8", u8, 0.0);  // λ8 (A1+)
    // Orbit-dependent on-site trilinear scaling (analogous to chi_orbit_scale)
    const double W_orbit1_scale = config.get_param("W_orbit1_scale", 1.0);
    const double W_orbit2_scale = config.get_param("W_orbit2_scale", 1.0);
    const double W_orbit3_scale = config.get_param("W_orbit3_scale", 1.0);
    const double W_orbit4_scale = config.get_param("W_orbit4_scale", 1.0);
    // =========================================================================
    // Inter-site anisotropy-modulation trilinear: λ^a_Tm · S^b_{Fe_i} · S^c_{Fe_i'}
    // =========================================================================
    // From tmfeo3_notes.tex Eq.11: extends aniso-mod to inter-site Fe bilinears.
    // Same A1+/A2+ sector decomposition as on-site, but the two Fe legs sit on
    // DIFFERENT sublattice sites (c-axis NN pairs: Fe0↔Fe3, Fe1↔Fe2).
    // Key physics: the on-site A2+ channel (v4,v6) cancels at q=0 for paired
    //   chi/chi_inv bonds, but the inter-site A2+ channel (w4,w6) does NOT—
    //   providing an escape route for λ4/λ6 excitations at zone center.
    // Convention: H += Σ V[a](b,c) S_{Fe_i}^a S_{Fe_i'}^b λ_{Tm_j}^c
    //   w-params couple to the same bilinear forms as u/v but across Fe-Fe NN pairs.
    const double w1 = config.get_param("w1", 0.0);   // λ1 (A1+) inter-site aniso-mod
    const double w3 = config.get_param("w3", 0.0);   // λ3 (A1+) inter-site aniso-mod
    const double w8 = config.get_param("w8", 0.0);   // λ8 (A1+) inter-site aniso-mod
    const double w4 = config.get_param("w4", 0.0);   // λ4 (A2+) inter-site aniso-mod
    const double w6 = config.get_param("w6", 0.0);   // λ6 (A2+) inter-site aniso-mod
    // General inter-site channels (additive with w-params, same structure as on-site)
    TrilinearChannel V1_ch = read_tri_ch("V1", w1, 0.0);
    TrilinearChannel V3_ch = read_tri_ch("V3", w3, 0.0);
    TrilinearChannel V4_ch = read_tri_ch("V4", 0.0, w4);
    TrilinearChannel V6_ch = read_tri_ch("V6", 0.0, w6);
    TrilinearChannel V8_ch = read_tri_ch("V8", w8, 0.0);
    // Antisymmetric inter-site channels: DM-like Fe bilinear × Tm Gell-Mann
    // VA{n}_A{ab}: couples (S_i^a S_{i'}^b - S_i^b S_{i'}^a) to λ_n
    AntiTrilinearChannel VA1_ch = read_anti_ch("V1");  // λ1 (A1+)
    AntiTrilinearChannel VA3_ch = read_anti_ch("V3");  // λ3 (A1+)
    AntiTrilinearChannel VA4_ch = read_anti_ch("V4");  // λ4 (A2+)
    AntiTrilinearChannel VA6_ch = read_anti_ch("V6");  // λ6 (A2+)
    AntiTrilinearChannel VA8_ch = read_anti_ch("V8");  // λ8 (A1+)
    // Orbit-dependent inter-site trilinear scaling
    const double V_orbit1_scale = config.get_param("V_orbit1_scale", 1.0);
    const double V_orbit2_scale = config.get_param("V_orbit2_scale", 1.0);
    const double V_orbit3_scale = config.get_param("V_orbit3_scale", 1.0);
    const double V_orbit4_scale = config.get_param("V_orbit4_scale", 1.0);
    const double e1 = config.get_param("e1", 0.97);
    const double e2 = config.get_param("e2", 3.97);
    
    // Projected magnetic moment matrix: J_α = Σ_a μ_{αa} λ_a  (from CEF wavefunctions)
    // This is a property of the Tm ion CEF states, independent of the Fe-Tm exchange coupling (chi)
    // Used for: sublattice frame construction, Zeeman coupling, magnetization output
    // Defaults from Tm3+ J=6 CEF calculation for TmFeO3 Pbnm:
    //   Jz = 5.264 λ_2,  Jx = 2.3915 λ_5 + 0.9128 λ_7,  Jy = -2.7866 λ_5 + 0.4655 λ_7
    const double mu_2x = config.get_param("mu_2x", 0.0);
    const double mu_2y = config.get_param("mu_2y", 0.0);
    const double mu_2z = config.get_param("mu_2z", 5.264);
    const double mu_5x = config.get_param("mu_5x", 2.3915);
    const double mu_5y = config.get_param("mu_5y", -2.7866);
    const double mu_5z = config.get_param("mu_5z", 0.0);
    const double mu_7x = config.get_param("mu_7x", 0.9128);
    const double mu_7y = config.get_param("mu_7y", 0.4655);
    const double mu_7z = config.get_param("mu_7z", 0.0);
    
    // g-factor ratio: g_Tm / g_Fe = (7/6) / 2 = 7/12 ≈ 0.5833
    // Scales Tm Zeeman coupling relative to Fe so both respond to the same physical field h.
    // Fe Zeeman: -h·S (g_Fe absorbed into h), Tm Zeeman: -g_ratio * h·(μ·λ)
    const double g_ratio_tm = config.get_param("g_ratio_tm", 7.0/12.0);
    
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
    // For bonds 1→0: standard DM with d_y = +D1
    std::array<std::array<double, 3>, 3> Ja_orig = {{{Jai, D2, -D1}, {-D2, Jai, 0}, {D1, 0, Jai}}};
    std::array<std::array<double, 3>, 3> Jb_orig = {{{Jbi, D2, -D1}, {-D2, Jbi, 0}, {D1, 0, Jbi}}};
    // For bonds 2→3: Pbnm symmetry requires opposite DM sign (d_y = -D1)
    std::array<std::array<double, 3>, 3> Ja23_orig = {{{Jai, -D2, D1}, {D2, Jai, 0}, {-D1, 0, Jai}}};
    std::array<std::array<double, 3>, 3> Jb23_orig = {{{Jbi, -D2, D1}, {D2, Jbi, 0}, {-D1, 0, Jbi}}};
    std::array<std::array<double, 3>, 3> Jc_orig = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};
    std::array<std::array<double, 3>, 3> J2a_orig = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    std::array<std::array<double, 3>, 3> J2b_orig = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    std::array<std::array<double, 3>, 3> J2c_orig = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};
    
    // Transform to local frames: J_local[i][j][a][b] = J_orig[a][b] * eta[i][a] * eta[j][b]
    std::array<std::array<std::array<std::array<double, 3>, 3>, 4>, 4> Ja, Jb, Jc, J2a, J2b, J2c;
    std::array<std::array<std::array<std::array<double, 3>, 3>, 4>, 4> Ja23, Jb23;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    Ja[i][j][a][b] = Ja_orig[a][b] * eta[i][a] * eta[j][b];
                    Jb[i][j][a][b] = Jb_orig[a][b] * eta[i][a] * eta[j][b];
                    Ja23[i][j][a][b] = Ja23_orig[a][b] * eta[i][a] * eta[j][b];
                    Jb23[i][j][a][b] = Jb23_orig[a][b] * eta[i][a] * eta[j][b];
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
    
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja23[2][3]), 2, 3, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja23[2][3]), 2, 3, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb23[2][3]), 2, 3, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb23[2][3]), 2, 3, Eigen::Vector3i(1, 0, 0));
    
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
    K_mat(1, 1) = Kb;
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
    
    // Set Bertaut G-mode AFM sublattice signs for orthoferrite: (+,-,+,-)
    Fe_atoms.set_afm_sublattice_signs({1.0, -1.0, 1.0, -1.0});
    
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
    
    // Build SU(3) sublattice frames from projected magnetic moment (mu matrix)
    // J_α = Σ_a μ_{αa} λ_a maps physical angular momentum to Gell-Mann generators
    // This is distinct from the Fe-Tm exchange coupling chi — the mu matrix is a
    // CEF property, while chi is an exchange coupling that can take any symmetry-allowed form.
    // Active generators carrying magnetic moment: λ_2 (idx 1), λ_5 (idx 4), λ_7 (idx 6)
    // Frame F_i = μ_act^{-1} D_i μ_act in the active subspace transforms
    // local SU(3) spins to a global frame where M_α = g_J Σ_a μ_{αa} (F^T S)_a
    // Pbnm sublattice signs (same as Fe): η = {(+,+,+), (+,-,-), (-,+,-), (-,-,+)}
    {
        Eigen::Matrix3d mu_act;
        mu_act << mu_2x, mu_5x, mu_7x,
                  mu_2y, mu_5y, mu_7y,
                  mu_2z, mu_5z, mu_7z;
        
        double mu_det = mu_act.determinant();
        if (std::abs(mu_det) > 1e-12) {
            Eigen::Matrix3d mu_act_inv = mu_act.inverse();
            const int active_idx[3] = {1, 4, 6};
            
            for (int sub = 0; sub < 4; ++sub) {
                Eigen::Matrix3d D = Eigen::Matrix3d::Zero();
                D(0,0) = eta[sub][0];
                D(1,1) = eta[sub][1];
                D(2,2) = eta[sub][2];
                
                Eigen::Matrix3d R_act = mu_act_inv * D * mu_act;
                
                SpinMatrix frame = SpinMatrix::Identity(8, 8);
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        frame(active_idx[a], active_idx[b]) = R_act(a, b);
                    }
                }
                Tm_atoms.set_sublattice_frame(frame, sub);
            }
            
            // Add Zeeman field from external magnetic field
            // B_a^(i) = g_ratio * Σ_α η_{iα} μ_{αa} h_α  (sublattice-dependent projection)
            // g_ratio = g_Tm/g_Fe scales Tm Zeeman relative to Fe for the same physical field
            if (h != 0.0 && config.field_direction.size() >= 3) {
                Eigen::Vector3d h_vec;
                h_vec << config.field_direction[0] * h,
                         config.field_direction[1] * h,
                         config.field_direction[2] * h;
                
                for (int sub = 0; sub < 4; ++sub) {
                    for (int a = 0; a < 3; ++a) {
                        double B_a = 0.0;
                        for (int al = 0; al < 3; ++al) {
                            B_a += eta[sub][al] * mu_act(al, a) * h_vec(al);
                        }
                        Tm_atoms.field[sub](active_idx[a]) += g_ratio_tm * B_a;
                    }
                }
            }
        }
    }
    
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
        
        // Tm1 ↔ Tm3 nearest neighbors (z=0.25 plane)
        // S2 maps Tm0-Tm2@(0,0,0)→Tm1-Tm3@(0,1,0), @(0,1,0)→@(0,0,0),
        //         @(-1,0,0)→@(-1,1,0), @(-1,1,0)→@(-1,0,0)
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(0, 1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(-1, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(-1, 1, 0));
        
        // Out-of-plane nearest neighbors (between z=0.75 and z=0.25 planes)
        // Only 2 NN bonds per pair (d=3.893Å); the other 2 candidate offsets
        // give d=6.107Å which is far beyond NN.
        // S2 maps Tm2-Tm1@(0,0,0)→Tm0-Tm3@(-1,1,0),
        //         Tm2-Tm1@(0,0,1)→Tm0-Tm3@(-1,1,1)
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(-1, 1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(-1, 1, 1));
        
        // Tm2 ↔ Tm1: nearest at d=3.893Å
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, 0, 1));
    }
    
    // Create mixed unit cell
    MixedUnitCell mixed_uc(Fe_atoms, Tm_atoms);
    
    // Set Fe-Tm bilinear coupling (following exact pattern from legacy code)
    if (chi2x != 0.0 || chi2y != 0.0 || chi2z != 0.0 || chi5x != 0.0 || chi5y != 0.0 || chi5z != 0.0 || chi7x != 0.0 || chi7y != 0.0 || chi7z != 0.0) {
        // chi is N_SU2 × N_SU3 = 3×8: rows = spin component (x,y,z), cols = λ index
        Eigen::MatrixXd chi = Eigen::MatrixXd::Zero(3, 8);
        chi(0, 1) = chi2x; chi(1, 1) = chi2y; chi(2, 1) = chi2z;
        chi(0, 4) = chi5x; chi(1, 4) = chi5y; chi(2, 4) = chi5z;
        chi(0, 6) = chi7x; chi(1, 6) = chi7y; chi(2, 6) = chi7z;
        
        Eigen::MatrixXd chi_inv = Eigen::MatrixXd::Zero(3, 8);
        chi_inv(0, 1) = chi2x; chi_inv(1, 1) = chi2y; chi_inv(2, 1) = chi2z;
        chi_inv(0, 4) = -chi5x; chi_inv(1, 4) = -chi5y; chi_inv(2, 4) = -chi5z;
        chi_inv(0, 6) = -chi7x; chi_inv(1, 6) = -chi7y; chi_inv(2, 6) = -chi7z;
        
        // Orbit-specific chi/chi_inv matrices.
        // chi_o[k] = s_k * chi_base: each orbit's superexchange scaled by distance.
        // In the minimal bond pattern used here, the static q=0 field has:
        //   λ2: constructive addition across paired bonds.
        //   λ5, λ7: exact cancellation across each chi/chi_inv pair for a uniform
        //           Gamma_2 background, independent of the orbit scales s_k.
        Eigen::MatrixXd chi_o1 = chi_orbit1_scale * chi;
        Eigen::MatrixXd chi_inv_o1 = chi_orbit1_scale * chi_inv;
        Eigen::MatrixXd chi_o2 = chi_orbit2_scale * chi;
        Eigen::MatrixXd chi_inv_o2 = chi_orbit2_scale * chi_inv;
        Eigen::MatrixXd chi_o3 = chi_orbit3_scale * chi;
        Eigen::MatrixXd chi_inv_o3 = chi_orbit3_scale * chi_inv;
        Eigen::MatrixXd chi_o4 = chi_orbit4_scale * chi;
        Eigen::MatrixXd chi_inv_o4 = chi_orbit4_scale * chi_inv;
        
        // Convention: set_mixed_bilinear(J, source=Fe_idx, partner=Tm_idx, offset)
        //   offset = Tm_cell - Fe_cell (added to Fe cell to find partner Tm cell)
        //
        // Pbnm (No.62) generators acting on fractional coords:
        //   S1: (x,y,z) -> (-x, -y, z+1/2)        [screw axis along c]
        //   S2: (x,y,z) -> (x+1/2, -y+1/2, -z)    [screw axis along a]
        //   I:  (x,y,z) -> (-x, -y, -z)            [inversion]
        //
        // Full D2h point group = {E, S1, S2, S1S2, I, S1I, S2I, S1S2I}
        //   chi-preserving:  E, S1, S2, S1S2
        //   chi-flipping:    I, S1I, S2I, S1S2I  (chi <-> chi_inv)
        //
        // Fe sublattices (Wyckoff 4b):
        //   Fe0=(0, 1/2, 1/2),  Fe1=(1/2, 0, 1/2),  Fe2=(1/2, 0, 0),  Fe3=(0, 1/2, 0)
        // Tm sublattices (Wyckoff 4c):
        //   Tm0=(0.02111, 0.92839, 0.75),  Tm1=(0.52111, 0.57161, 0.25)
        //   Tm2=(0.47889, 0.42839, 0.75),  Tm3=(0.97889, 0.07161, 0.25)
        //
        // 32 total NN bonds = 4 orbits x 8 bonds/orbit (verified by brute-force enumeration)

        // =====================================================================
        // Fe site 0 — 8 nearest Tm neighbors
        // =====================================================================
        // Orbit 1 (d=0.497): E  -> Fe0-Tm3@(-1,0,0)  chi
        mixed_uc.set_mixed_bilinear(chi_o1, 0, 3, Eigen::Vector3i(-1, 0, 0));
        // Orbit 1 (d=0.497): I  -> Fe0-Tm0@(0,0,0)   chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o1, 0, 0, Eigen::Vector3i(0, 0, 0));
        // Orbit 2 (d=0.545): E  -> Fe0-Tm2@(0,0,0)   chi
        mixed_uc.set_mixed_bilinear(chi_o2, 0, 2, Eigen::Vector3i(0, 0, 0));
        // Orbit 2 (d=0.545): I  -> Fe0-Tm1@(-1,0,0)  chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o2, 0, 1, Eigen::Vector3i(-1, 0, 0));
        // Orbit 3 (d=0.582): E  -> Fe0-Tm1@(0,0,0)   chi
        mixed_uc.set_mixed_bilinear(chi_o3, 0, 1, Eigen::Vector3i(0, 0, 0));
        // Orbit 3 (d=0.582): I  -> Fe0-Tm2@(-1,0,0)  chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o3, 0, 2, Eigen::Vector3i(-1, 0, 0));
        // Orbit 4 (d=0.624): E  -> Fe0-Tm0@(0,-1,0)  chi
        mixed_uc.set_mixed_bilinear(chi_o4, 0, 0, Eigen::Vector3i(0, -1, 0));
        // Orbit 4 (d=0.624): I  -> Fe0-Tm3@(-1,1,0)  chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o4, 0, 3, Eigen::Vector3i(-1, 1, 0));

        // =====================================================================
        // Fe site 1 — 8 nearest Tm neighbors
        // =====================================================================
        // Orbit 1 (d=0.497): S2   -> Fe1-Tm2@(0,0,0)   chi
        mixed_uc.set_mixed_bilinear(chi_o1, 1, 2, Eigen::Vector3i(0, 0, 0));
        // Orbit 1 (d=0.497): S2I  -> Fe1-Tm1@(0,-1,0)  chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o1, 1, 1, Eigen::Vector3i(0, -1, 0));
        // Orbit 2 (d=0.545): S2I  -> Fe1-Tm0@(0,-1,0)  chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o2, 1, 0, Eigen::Vector3i(0, -1, 0));
        // Orbit 2 (d=0.545): S2   -> Fe1-Tm3@(0,0,0)   chi
        mixed_uc.set_mixed_bilinear(chi_o2, 1, 3, Eigen::Vector3i(0, 0, 0));
        // Orbit 3 (d=0.582): S2   -> Fe1-Tm0@(1,-1,0)  chi
        mixed_uc.set_mixed_bilinear(chi_o3, 1, 0, Eigen::Vector3i(1, -1, 0));
        // Orbit 3 (d=0.582): S2I  -> Fe1-Tm3@(-1,0,0)  chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o3, 1, 3, Eigen::Vector3i(-1, 0, 0));
        // Orbit 4 (d=0.624): S2   -> Fe1-Tm1@(0,0,0)   chi
        mixed_uc.set_mixed_bilinear(chi_o4, 1, 1, Eigen::Vector3i(0, 0, 0));
        // Orbit 4 (d=0.624): S2I  -> Fe1-Tm2@(0,-1,0)  chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o4, 1, 2, Eigen::Vector3i(0, -1, 0));

        // =====================================================================
        // Fe site 2 — 8 nearest Tm neighbors
        // =====================================================================
        // Orbit 1 (d=0.497): S1S2I -> Fe2-Tm2@(0,0,-1)  chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o1, 2, 2, Eigen::Vector3i(0, 0, -1));
        // Orbit 1 (d=0.497): S1S2  -> Fe2-Tm1@(0,-1,0)  chi
        mixed_uc.set_mixed_bilinear(chi_o1, 2, 1, Eigen::Vector3i(0, -1, 0));
        // Orbit 2 (d=0.545): S1S2  -> Fe2-Tm0@(0,-1,-1) chi
        mixed_uc.set_mixed_bilinear(chi_o2, 2, 0, Eigen::Vector3i(0, -1, -1));
        // Orbit 2 (d=0.545): S1S2I -> Fe2-Tm3@(0,0,0)   chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o2, 2, 3, Eigen::Vector3i(0, 0, 0));
        // Orbit 3 (d=0.582): S1S2I -> Fe2-Tm0@(1,-1,-1) chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o3, 2, 0, Eigen::Vector3i(1, -1, -1));
        // Orbit 3 (d=0.582): S1S2  -> Fe2-Tm3@(-1,0,0)  chi
        mixed_uc.set_mixed_bilinear(chi_o3, 2, 3, Eigen::Vector3i(-1, 0, 0));
        // Orbit 4 (d=0.624): S1S2I -> Fe2-Tm1@(0,0,0)   chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o4, 2, 1, Eigen::Vector3i(0, 0, 0));
        // Orbit 4 (d=0.624): S1S2  -> Fe2-Tm2@(0,-1,-1) chi
        mixed_uc.set_mixed_bilinear(chi_o4, 2, 2, Eigen::Vector3i(0, -1, -1));

        // =====================================================================
        // Fe site 3 — 8 nearest Tm neighbors
        // =====================================================================
        // Orbit 1 (d=0.497): S1I -> Fe3-Tm3@(-1,0,0)  chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o1, 3, 3, Eigen::Vector3i(-1, 0, 0));
        // Orbit 1 (d=0.497): S1  -> Fe3-Tm0@(0,0,-1)  chi
        mixed_uc.set_mixed_bilinear(chi_o1, 3, 0, Eigen::Vector3i(0, 0, -1));
        // Orbit 2 (d=0.545): S1I -> Fe3-Tm2@(0,0,-1)  chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o2, 3, 2, Eigen::Vector3i(0, 0, -1));
        // Orbit 2 (d=0.545): S1  -> Fe3-Tm1@(-1,0,0)  chi
        mixed_uc.set_mixed_bilinear(chi_o2, 3, 1, Eigen::Vector3i(-1, 0, 0));
        // Orbit 3 (d=0.582): S1I -> Fe3-Tm1@(0,0,0)   chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o3, 3, 1, Eigen::Vector3i(0, 0, 0));
        // Orbit 3 (d=0.582): S1  -> Fe3-Tm2@(-1,0,-1)  chi
        mixed_uc.set_mixed_bilinear(chi_o3, 3, 2, Eigen::Vector3i(-1, 0, -1));
        // Orbit 4 (d=0.624): S1I -> Fe3-Tm0@(0,-1,-1) chi_inv
        mixed_uc.set_mixed_bilinear(chi_inv_o4, 3, 0, Eigen::Vector3i(0, -1, -1));
        // Orbit 4 (d=0.624): S1  -> Fe3-Tm3@(-1,1,0)  chi
        mixed_uc.set_mixed_bilinear(chi_o4, 3, 3, Eigen::Vector3i(-1, 1, 0));
    }
    
    // =========================================================================
    // Anisotropy-modulation trilinear coupling (tmfeo3_notes.tex Eq.10,12)
    // =========================================================================
    // H_aniso = Σ K[a](b,c) S_source^a S_partner1^b λ_partner2^c
    // where source=Fe_i, partner1=Fe_i (same site, offset1=0), partner2=Tm_j
    // Uses the same Fe-Tm bond list as the bilinear chi coupling.
    // chi-type bonds: full W tensor; chi_inv-type bonds: A2+ (v4,v6) flipped.
    // Shared helpers for building trilinear SpinTensor3 from TrilinearChannel
    auto fill_channel = [](SpinTensor3& T, int c_idx, const TrilinearChannel& ch, double sign) {
        T[0](0, c_idx) = sign * ch.xx;
        T[1](1, c_idx) = sign * ch.yy;
        T[2](2, c_idx) = sign * ch.zz;
        T[0](1, c_idx) = sign * ch.xy;  T[1](0, c_idx) = sign * ch.xy;
        T[0](2, c_idx) = sign * ch.xz;  T[2](0, c_idx) = sign * ch.xz;
        T[1](2, c_idx) = sign * ch.yz;  T[2](1, c_idx) = sign * ch.yz;
    };
    // Antisymmetric Fe bilinear: T[a](b,c) = -T[b](a,c) for inter-site coupling.
    // Adds to (not overwrites) existing tensor entries, so call after fill_channel.
    auto fill_anti_channel = [](SpinTensor3& T, int c_idx, const AntiTrilinearChannel& ch, double sign) {
        T[0](1, c_idx) += sign * ch.xy;  T[1](0, c_idx) -= sign * ch.xy;
        T[0](2, c_idx) += sign * ch.xz;  T[2](0, c_idx) -= sign * ch.xz;
        T[1](2, c_idx) += sign * ch.yz;  T[2](1, c_idx) -= sign * ch.yz;
    };
    auto scale_tensor3 = [](const SpinTensor3& T, double s) -> SpinTensor3 {
        SpinTensor3 out(T.size());
        for (size_t i = 0; i < T.size(); ++i) out[i] = s * T[i];
        return out;
    };

    {
        // Build the on-site Fe bilinear ⊗ Tm Gell-Mann tensor (general form)
        // K[a](b,c): a,b ∈ {x=0,y=1,z=2} (SU2), c ∈ {λ1..λ8} (SU3, 0-indexed)
        // Full symmetric Fe quadrupole: 6 independent bilinears × 5 Tm-even channels
        // sign_A2 = ±1 distinguishes chi (E-type) vs chi_inv (I-type) bonds
        auto build_W_general = [&](double sign_A2) -> SpinTensor3 {
            SpinTensor3 W(3);
            for (int a = 0; a < 3; ++a) W[a] = Eigen::MatrixXd::Zero(3, 8);
            // A1+ sector: λ1 (idx 0), λ3 (idx 2), λ8 (idx 7) — sign=+1 always
            fill_channel(W, 0, W1_ch, 1.0);
            fill_channel(W, 2, W3_ch, 1.0);
            fill_channel(W, 7, W8_ch, 1.0);
            // A2+ sector: λ4 (idx 3), λ6 (idx 5) — sign flips under inversion
            fill_channel(W, 3, W4_ch, sign_A2);
            fill_channel(W, 5, W6_ch, sign_A2);
            return W;
        };
        SpinTensor3 W_chi_base     = build_W_general(+1.0);
        SpinTensor3 W_chi_inv_base = build_W_general(-1.0);
        // Orbit-scaled on-site trilinear tensors
        SpinTensor3 W_chi_o1 = scale_tensor3(W_chi_base, W_orbit1_scale);
        SpinTensor3 W_chi_inv_o1 = scale_tensor3(W_chi_inv_base, W_orbit1_scale);
        SpinTensor3 W_chi_o2 = scale_tensor3(W_chi_base, W_orbit2_scale);
        SpinTensor3 W_chi_inv_o2 = scale_tensor3(W_chi_inv_base, W_orbit2_scale);
        SpinTensor3 W_chi_o3 = scale_tensor3(W_chi_base, W_orbit3_scale);
        SpinTensor3 W_chi_inv_o3 = scale_tensor3(W_chi_inv_base, W_orbit3_scale);
        SpinTensor3 W_chi_o4 = scale_tensor3(W_chi_base, W_orbit4_scale);
        SpinTensor3 W_chi_inv_o4 = scale_tensor3(W_chi_inv_base, W_orbit4_scale);
        
        // Set trilinear on the same 32 Fe-Tm bonds as the bilinear chi coupling.
        // source=Fe_i, partner1=Fe_i (offset1=0,0,0), partner2=Tm_j (offset2 from bond list)
        // Bond order follows bilinear: orbit 1,1, 2,2, 3,3, 4,4 per Fe site
        
        // Fe site 0
        mixed_uc.set_mixed_trilinear(W_chi_o1,     0, 0, 3, Eigen::Vector3i(0,0,0), Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o1, 0, 0, 0, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_o2,     0, 0, 2, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o2, 0, 0, 1, Eigen::Vector3i(0,0,0), Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_o3,     0, 0, 1, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o3, 0, 0, 2, Eigen::Vector3i(0,0,0), Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_o4,     0, 0, 0, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, -1, 0));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o4, 0, 0, 3, Eigen::Vector3i(0,0,0), Eigen::Vector3i(-1, 1, 0));
        
        // Fe site 1
        mixed_uc.set_mixed_trilinear(W_chi_o1,     1, 1, 2, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o1, 1, 1, 1, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, -1, 0));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o2, 1, 1, 0, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, -1, 0));
        mixed_uc.set_mixed_trilinear(W_chi_o2,     1, 1, 3, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_o3,     1, 1, 0, Eigen::Vector3i(0,0,0), Eigen::Vector3i(1, -1, 0));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o3, 1, 1, 3, Eigen::Vector3i(0,0,0), Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_o4,     1, 1, 1, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o4, 1, 1, 2, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, -1, 0));
        
        // Fe site 2
        mixed_uc.set_mixed_trilinear(W_chi_inv_o1, 2, 2, 2, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, -1));
        mixed_uc.set_mixed_trilinear(W_chi_o1,     2, 2, 1, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, -1, 0));
        mixed_uc.set_mixed_trilinear(W_chi_o2,     2, 2, 0, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, -1, -1));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o2, 2, 2, 3, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o3, 2, 2, 0, Eigen::Vector3i(0,0,0), Eigen::Vector3i(1, -1, -1));
        mixed_uc.set_mixed_trilinear(W_chi_o3,     2, 2, 3, Eigen::Vector3i(0,0,0), Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o4, 2, 2, 1, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_o4,     2, 2, 2, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, -1, -1));
        
        // Fe site 3
        mixed_uc.set_mixed_trilinear(W_chi_inv_o1, 3, 3, 3, Eigen::Vector3i(0,0,0), Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_o1,     3, 3, 0, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, -1));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o2, 3, 3, 2, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, -1));
        mixed_uc.set_mixed_trilinear(W_chi_o2,     3, 3, 1, Eigen::Vector3i(0,0,0), Eigen::Vector3i(-1, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o3, 3, 3, 1, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, 0, 0));
        mixed_uc.set_mixed_trilinear(W_chi_o3,     3, 3, 2, Eigen::Vector3i(0,0,0), Eigen::Vector3i(-1, 0, -1));
        mixed_uc.set_mixed_trilinear(W_chi_inv_o4, 3, 3, 0, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0, -1, -1));
        mixed_uc.set_mixed_trilinear(W_chi_o4,     3, 3, 3, Eigen::Vector3i(0,0,0), Eigen::Vector3i(-1, 1, 0));
    }
    
    // =========================================================================
    // Inter-site anisotropy-modulation trilinear (tmfeo3_notes.tex Eq.11)
    // =========================================================================
    // H_inter = Σ V[a](b,c) S_{Fe_i}^a S_{Fe_i'}^b λ_{Tm_j}^c
    // Same A1+/A2+ tensor structure as on-site W, but the two Fe legs are on
    // different sites: c-axis NN pairs Fe0↔Fe3, Fe1↔Fe2.
    // Each of the 32 Fe-Tm bonds generates 2 inter-site bonds (one per c-axis NN).
    // c-axis Fe NN offsets (offset = partner_cell - source_cell):
    //   Fe0 → Fe3 @ (0,0,0) and (0,0,1)
    //   Fe1 → Fe2 @ (0,0,0) and (0,0,1)
    //   Fe2 → Fe1 @ (0,0,0) and (0,0,-1)
    //   Fe3 → Fe0 @ (0,0,0) and (0,0,-1)
    if (w1 != 0.0 || w3 != 0.0 || w8 != 0.0 || w4 != 0.0 || w6 != 0.0
        || V1_ch.xx != 0.0 || V1_ch.yy != 0.0 || V1_ch.xy != 0.0 || V1_ch.yz != 0.0
        || V3_ch.xx != 0.0 || V3_ch.yy != 0.0 || V3_ch.xy != 0.0 || V3_ch.yz != 0.0
        || V4_ch.xx != 0.0 || V4_ch.yy != 0.0 || V4_ch.xy != 0.0 || V4_ch.yz != 0.0
        || V6_ch.xx != 0.0 || V6_ch.yy != 0.0 || V6_ch.xy != 0.0 || V6_ch.yz != 0.0
        || V8_ch.xx != 0.0 || V8_ch.yy != 0.0 || V8_ch.xy != 0.0 || V8_ch.yz != 0.0
        || VA1_ch.xy != 0.0 || VA1_ch.xz != 0.0 || VA1_ch.yz != 0.0
        || VA3_ch.xy != 0.0 || VA3_ch.xz != 0.0 || VA3_ch.yz != 0.0
        || VA4_ch.xy != 0.0 || VA4_ch.xz != 0.0 || VA4_ch.yz != 0.0
        || VA6_ch.xy != 0.0 || VA6_ch.xz != 0.0 || VA6_ch.yz != 0.0
        || VA8_ch.xy != 0.0 || VA8_ch.xz != 0.0 || VA8_ch.yz != 0.0) {
        // Reuse fill_channel and fill_anti_channel from on-site (captured above)
        auto build_V_general = [&](double sign_A2) -> SpinTensor3 {
            SpinTensor3 V(3);
            for (int a = 0; a < 3; ++a) V[a] = Eigen::MatrixXd::Zero(3, 8);
            // Symmetric Fe bilinear: S_i^a S_{i'}^b + S_i^b S_{i'}^a
            fill_channel(V, 0, V1_ch, 1.0);
            fill_channel(V, 2, V3_ch, 1.0);
            fill_channel(V, 7, V8_ch, 1.0);
            fill_channel(V, 3, V4_ch, sign_A2);
            fill_channel(V, 5, V6_ch, sign_A2);
            // Antisymmetric Fe bilinear: S_i^a S_{i'}^b - S_i^b S_{i'}^a
            // Same A1+/A2+ sign structure — inversion maps each Fe site to itself
            // so the Fe bilinear (symmetric or antisymmetric) is unchanged; only
            // the Tm λ parity (A1+/A2+) determines the sign_A2 flip.
            fill_anti_channel(V, 0, VA1_ch, 1.0);
            fill_anti_channel(V, 2, VA3_ch, 1.0);
            fill_anti_channel(V, 7, VA8_ch, 1.0);
            fill_anti_channel(V, 3, VA4_ch, sign_A2);
            fill_anti_channel(V, 5, VA6_ch, sign_A2);
            return V;
        };
        
        SpinTensor3 V_chi_base     = build_V_general(+1.0);
        SpinTensor3 V_chi_inv_base = build_V_general(-1.0);
        // Orbit-scaled inter-site trilinear tensors
        SpinTensor3 V_chi_o1 = scale_tensor3(V_chi_base, V_orbit1_scale);
        SpinTensor3 V_chi_inv_o1 = scale_tensor3(V_chi_inv_base, V_orbit1_scale);
        SpinTensor3 V_chi_o2 = scale_tensor3(V_chi_base, V_orbit2_scale);
        SpinTensor3 V_chi_inv_o2 = scale_tensor3(V_chi_inv_base, V_orbit2_scale);
        SpinTensor3 V_chi_o3 = scale_tensor3(V_chi_base, V_orbit3_scale);
        SpinTensor3 V_chi_inv_o3 = scale_tensor3(V_chi_inv_base, V_orbit3_scale);
        SpinTensor3 V_chi_o4 = scale_tensor3(V_chi_base, V_orbit4_scale);
        SpinTensor3 V_chi_inv_o4 = scale_tensor3(V_chi_inv_base, V_orbit4_scale);
        
        // c-axis NN partner sublattice index and two offsets for each Fe site
        // Fe_i → partner Fe_p at offsets c_off[0], c_off[1]
        struct CAxisNN { int partner; Eigen::Vector3i off0, off1; };
        CAxisNN c_nn[4] = {
            {3, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0,0,1)},   // Fe0→Fe3
            {2, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0,0,1)},   // Fe1→Fe2
            {1, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0,0,-1)},  // Fe2→Fe1
            {0, Eigen::Vector3i(0,0,0), Eigen::Vector3i(0,0,-1)}   // Fe3→Fe0
        };
        
        // Macro-like lambda to set inter-site trilinear for one Fe-Tm bond
        // with both c-axis Fe NN offsets
        auto set_inter = [&](const SpinTensor3& V, int fe_src, int tm_dst,
                             const Eigen::Vector3i& tm_off) {
            const auto& nn = c_nn[fe_src];
            mixed_uc.set_mixed_trilinear(V, fe_src, nn.partner, tm_dst, nn.off0, tm_off);
            mixed_uc.set_mixed_trilinear(V, fe_src, nn.partner, tm_dst, nn.off1, tm_off);
        };
        
        // Fe site 0 — 8 bonds (same Fe-Tm topology as bilinear/on-site trilinear)
        set_inter(V_chi_o1,     0, 3, Eigen::Vector3i(-1, 0, 0));
        set_inter(V_chi_inv_o1, 0, 0, Eigen::Vector3i(0, 0, 0));
        set_inter(V_chi_o2,     0, 2, Eigen::Vector3i(0, 0, 0));
        set_inter(V_chi_inv_o2, 0, 1, Eigen::Vector3i(-1, 0, 0));
        set_inter(V_chi_o3,     0, 1, Eigen::Vector3i(0, 0, 0));
        set_inter(V_chi_inv_o3, 0, 2, Eigen::Vector3i(-1, 0, 0));
        set_inter(V_chi_o4,     0, 0, Eigen::Vector3i(0, -1, 0));
        set_inter(V_chi_inv_o4, 0, 3, Eigen::Vector3i(-1, 1, 0));
        
        // Fe site 1 — 8 bonds
        set_inter(V_chi_o1,     1, 2, Eigen::Vector3i(0, 0, 0));
        set_inter(V_chi_inv_o1, 1, 1, Eigen::Vector3i(0, -1, 0));
        set_inter(V_chi_inv_o2, 1, 0, Eigen::Vector3i(0, -1, 0));
        set_inter(V_chi_o2,     1, 3, Eigen::Vector3i(0, 0, 0));
        set_inter(V_chi_o3,     1, 0, Eigen::Vector3i(1, -1, 0));
        set_inter(V_chi_inv_o3, 1, 3, Eigen::Vector3i(-1, 0, 0));
        set_inter(V_chi_o4,     1, 1, Eigen::Vector3i(0, 0, 0));
        set_inter(V_chi_inv_o4, 1, 2, Eigen::Vector3i(0, -1, 0));
        
        // Fe site 2 — 8 bonds
        set_inter(V_chi_inv_o1, 2, 2, Eigen::Vector3i(0, 0, -1));
        set_inter(V_chi_o1,     2, 1, Eigen::Vector3i(0, -1, 0));
        set_inter(V_chi_o2,     2, 0, Eigen::Vector3i(0, -1, -1));
        set_inter(V_chi_inv_o2, 2, 3, Eigen::Vector3i(0, 0, 0));
        set_inter(V_chi_inv_o3, 2, 0, Eigen::Vector3i(1, -1, -1));
        set_inter(V_chi_o3,     2, 3, Eigen::Vector3i(-1, 0, 0));
        set_inter(V_chi_inv_o4, 2, 1, Eigen::Vector3i(0, 0, 0));
        set_inter(V_chi_o4,     2, 2, Eigen::Vector3i(0, -1, -1));
        
        // Fe site 3 — 8 bonds
        set_inter(V_chi_inv_o1, 3, 3, Eigen::Vector3i(-1, 0, 0));
        set_inter(V_chi_o1,     3, 0, Eigen::Vector3i(0, 0, -1));
        set_inter(V_chi_inv_o2, 3, 2, Eigen::Vector3i(0, 0, -1));
        set_inter(V_chi_o2,     3, 1, Eigen::Vector3i(-1, 0, 0));
        set_inter(V_chi_inv_o3, 3, 1, Eigen::Vector3i(0, 0, 0));
        set_inter(V_chi_o3,     3, 2, Eigen::Vector3i(-1, 0, -1));
        set_inter(V_chi_inv_o4, 3, 0, Eigen::Vector3i(0, -1, -1));
        set_inter(V_chi_o4,     3, 3, Eigen::Vector3i(-1, 1, 0));
    }
    
    return mixed_uc;
}

UnitCell build_tmfeo3_fe(const SpinConfig& config) {
    const double Jai = config.get_param("J1ab", 4.74);
    const double Jbi = Jai;
    const double Jci = config.get_param("J1c", 5.15);
    const double J2ai = config.get_param("J2ab", 0.15);
    const double J2bi = J2ai;
    const double J2ci = config.get_param("J2c", 0.30);
    const double Ka = config.get_param("Ka", -0.16221);
    const double Kb = config.get_param("Kb", 0.0);
    const double Kc = config.get_param("Kc", -0.18318);
    const double D1 = config.get_param("D1", 0.12);
    const double D2 = config.get_param("D2", 0.0);
    const double h = config.field_strength;
    const int use_local_frame = (int)config.get_param("use_local_frame", 1.0);
    
    // Use TmFeO3_Fe class from unitcell.h (already has structure)
    TmFeO3_Fe Fe_atoms(3);
    

    // Local frame transformation (following molecular_dynamic_TmFeO3.cpp exactly)
    std::array<std::array<double, 3>, 4> eta;
    if (use_local_frame) {
        eta = {{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}};
    } else {
        // Global frame: no eta transformation
        eta = {{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}};
        // Reset sublattice frames to identity
        Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(3, 3);
        Fe_atoms.set_sublattice_frame(identity, 0);
        Fe_atoms.set_sublattice_frame(identity, 1);
        Fe_atoms.set_sublattice_frame(identity, 2);
        Fe_atoms.set_sublattice_frame(identity, 3);
    }
    
    // Original exchange matrices in global frame
    // For bonds 1→0: standard DM with d_y = +D1
    std::array<std::array<double, 3>, 3> Ja_orig = {{{Jai, D2, -D1}, {-D2, Jai, 0}, {D1, 0, Jai}}};
    std::array<std::array<double, 3>, 3> Jb_orig = {{{Jbi, D2, -D1}, {-D2, Jbi, 0}, {D1, 0, Jbi}}};
    // For bonds 2→3: Pbnm symmetry requires opposite DM sign (d_y = -D1)
    // This produces the correct F_x weak ferromagnetism in Gamma_2 phase
    std::array<std::array<double, 3>, 3> Ja23_orig = {{{Jai, -D2, D1}, {D2, Jai, 0}, {-D1, 0, Jai}}};
    std::array<std::array<double, 3>, 3> Jb23_orig = {{{Jbi, -D2, D1}, {D2, Jbi, 0}, {-D1, 0, Jbi}}};
    std::array<std::array<double, 3>, 3> Jc_orig = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};
    std::array<std::array<double, 3>, 3> J2a_orig = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    std::array<std::array<double, 3>, 3> J2b_orig = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    std::array<std::array<double, 3>, 3> J2c_orig = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};
    
    // Transform to local frames: J_local[i][j][a][b] = J_orig[a][b] * eta[i][a] * eta[j][b]
    std::array<std::array<std::array<std::array<double, 3>, 3>, 4>, 4> Ja, Jb, Jc, J2a, J2b, J2c;
    std::array<std::array<std::array<std::array<double, 3>, 3>, 4>, 4> Ja23, Jb23;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    Ja[i][j][a][b] = Ja_orig[a][b] * eta[i][a] * eta[j][b];
                    Jb[i][j][a][b] = Jb_orig[a][b] * eta[i][a] * eta[j][b];
                    Ja23[i][j][a][b] = Ja23_orig[a][b] * eta[i][a] * eta[j][b];
                    Jb23[i][j][a][b] = Jb23_orig[a][b] * eta[i][a] * eta[j][b];
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
    
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja23[2][3]), 2, 3, Eigen::Vector3i(0, 0, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Ja23[2][3]), 2, 3, Eigen::Vector3i(1, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb23[2][3]), 2, 3, Eigen::Vector3i(0, -1, 0));
    Fe_atoms.set_bilinear_interaction(to_eigen(Jb23[2][3]), 2, 3, Eigen::Vector3i(1, 0, 0));
    
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
    K_mat(1, 1) = Kb;
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
    
    // Set Bertaut G-mode AFM sublattice signs for orthoferrite: (+,-,+,-)
    Fe_atoms.set_afm_sublattice_signs({1.0, -1.0, 1.0, -1.0});
    
    return Fe_atoms;
}

UnitCell build_tmfeo3_tm(const SpinConfig& config) {
    // Crystal field parameters for Tm ions
    const double e1 = config.get_param("e1", 0.97);
    const double e2 = config.get_param("e2", 3.97);
    
    // Projected magnetic moment matrix: J_α = Σ_a μ_{αa} λ_a  (from CEF wavefunctions)
    // This is a property of the Tm ion CEF states, used for sublattice frames and Zeeman.
    // Defaults from Tm3+ J=6 CEF calculation for TmFeO3 Pbnm:
    //   Jz = 5.264 λ_2,  Jx = 2.3915 λ_5 + 0.9128 λ_7,  Jy = -2.7866 λ_5 + 0.4655 λ_7
    const double mu_2x = config.get_param("mu_2x", 0.0);
    const double mu_2y = config.get_param("mu_2y", 0.0);
    const double mu_2z = config.get_param("mu_2z", 5.264);
    const double mu_5x = config.get_param("mu_5x", 2.3915);
    const double mu_5y = config.get_param("mu_5y", -2.7866);
    const double mu_5z = config.get_param("mu_5z", 0.0);
    const double mu_7x = config.get_param("mu_7x", 0.9128);
    const double mu_7y = config.get_param("mu_7y", 0.4655);
    const double mu_7z = config.get_param("mu_7z", 0.0);
    
    // g-factor ratio: g_Tm / g_Fe = (7/6) / 2 = 7/12 ≈ 0.5833
    // For standalone Tm system, this scales the Zeeman coupling relative to an implicit Fe scale.
    const double g_ratio_tm = config.get_param("g_ratio_tm", 7.0/12.0);
    
    const double h = config.field_strength;
    
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
    
    // Pbnm sublattice sign factors (pseudovector transformation)
    // Tm0(E): (+,+,+), Tm1(S2): (+,-,-), Tm2(S1S2): (-,+,-), Tm3(S1): (-,-,+)
    std::array<std::array<double, 3>, 4> eta = {{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}};
    
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
    
    // Build SU(3) sublattice frames from projected magnetic moment (mu matrix)
    // Active generators: λ_2 (idx 1), λ_5 (idx 4), λ_7 (idx 6)
    // Frame F_i = μ_act^{-1} D_i μ_act maps local SU(3) spins to global frame
    {
        Eigen::Matrix3d mu_act;
        mu_act << mu_2x, mu_5x, mu_7x,
                  mu_2y, mu_5y, mu_7y,
                  mu_2z, mu_5z, mu_7z;
        
        double mu_det = mu_act.determinant();
        if (std::abs(mu_det) > 1e-12) {
            Eigen::Matrix3d mu_act_inv = mu_act.inverse();
            const int active_idx[3] = {1, 4, 6};
            
            for (int sub = 0; sub < 4; ++sub) {
                Eigen::Matrix3d D = Eigen::Matrix3d::Zero();
                D(0,0) = eta[sub][0];
                D(1,1) = eta[sub][1];
                D(2,2) = eta[sub][2];
                
                Eigen::Matrix3d R_act = mu_act_inv * D * mu_act;
                
                SpinMatrix frame = SpinMatrix::Identity(8, 8);
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        frame(active_idx[a], active_idx[b]) = R_act(a, b);
                    }
                }
                Tm_atoms.set_sublattice_frame(frame, sub);
            }
            
            // Add Zeeman field from external magnetic field
            // B_a^(i) = g_ratio * Σ_α η_{iα} μ_{αa} h_α  (sublattice-dependent projection)
            // g_ratio = g_Tm/g_Fe scales Tm Zeeman relative to Fe for the same physical field
            if (h != 0.0 && config.field_direction.size() >= 3) {
                Eigen::Vector3d h_vec;
                h_vec << config.field_direction[0] * h,
                         config.field_direction[1] * h,
                         config.field_direction[2] * h;
                
                for (int sub = 0; sub < 4; ++sub) {
                    for (int a = 0; a < 3; ++a) {
                        double B_a = 0.0;
                        for (int al = 0; al < 3; ++al) {
                            B_a += eta[sub][al] * mu_act(al, a) * h_vec(al);
                        }
                        Tm_atoms.field[sub](active_idx[a]) += g_ratio_tm * B_a;
                    }
                }
            }
        }
    }
    
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
        
        // Tm1 ↔ Tm3 nearest neighbors (z=0.25 plane)
        // S2 maps Tm0-Tm2@(0,0,0)→Tm1-Tm3@(0,1,0), @(0,1,0)→@(0,0,0),
        //         @(-1,0,0)→@(-1,1,0), @(-1,1,0)→@(-1,0,0)
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(0, 1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(-1, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 1, 3, Eigen::Vector3i(-1, 1, 0));
        
        // Out-of-plane nearest neighbors (between z=0.75 and z=0.25 planes)
        // Only 2 NN bonds per pair (d=3.893Å); the other 2 candidate offsets
        // give d=6.107Å which is far beyond NN.
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(-1, 1, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 0, 3, Eigen::Vector3i(-1, 1, 1));
        
        // Tm2 ↔ Tm1: nearest at d=3.893Å
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, 0, 0));
        Tm_atoms.set_bilinear_interaction(J_tm_mat, 2, 1, Eigen::Vector3i(0, 0, 1));
    }
    
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
    // Same sublattice, 6 neighbors at offsets: (±1,0), (0,±1), (±1,∓1)
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