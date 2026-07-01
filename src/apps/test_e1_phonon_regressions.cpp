/**
 * test_e1_phonon_regressions.cpp
 *
 * End-to-end symmetry + consistency tests for the NCTO E1 magnetoelastic
 * Hamiltonian implemented in PhononLattice.
 *
 * The model under test is the one in docs/tmfeo3_notes.tex:
 *
 *   H = H_spin + H_E1 + H_drive + H_sp-ph,
 *
 * with the leading symmetry-allowed E1 spin-phonon coupling
 *
 *   H_sp-ph = Σ_<ij>γ Σ_X δX_γ(ε) O^{(X)}_{ij,γ},   X ∈ {J, K, Γ, Γ'},
 *   δX_γ(ε) = λ_{X,0}(ε_x²+ε_y²) + λ_{X,2}[(ε_x²-ε_y²) cos 2θ_γ
 *                                          + 2 ε_x ε_y sin 2θ_γ],
 *
 * and bond-axis angles  (θ_x, θ_y, θ_z) = (0, 2π/3, 4π/3).
 *
 * Tests performed:
 *
 *   1. Form-factor sum rule:  Σ_γ (cos 2θ_γ, sin 2θ_γ) = (0, 0).
 *      Required for the bond-anisotropy part of δX_γ to vanish on isotropic
 *      spin configs and for C_3 to permute the three bonds correctly.
 *
 *   2. δX_γ(ε) explicit formula:  numerically compare δX_γ extracted from
 *      one-bond magnetoelastic energy against the boxed formula in the notes.
 *
 *   3. Linear-in-ε coupling is identically zero:  ∂H_sp-ph/∂ε_a |_{ε=0} = 0.
 *      This is the central symmetry statement of the notes — there is no
 *      C_6-allowed linear exchange-striction term in the J–K–Γ–Γ' sector.
 *
 *   4. Phonon force / energy consistency:  ∂H_sp-ph/∂ε_a (analytic) matches
 *      a 2nd-order finite difference of spin_phonon_energy.
 *
 *   5. Spin force / energy consistency:  the E1 contribution to the local
 *      effective field on each spin matches the finite difference of the
 *      total energy with respect to that spin (small-perturbation test).
 *
 *   6. C_3 invariance of one-bond magnetoelastic energy under the combined
 *      transformation
 *           bond label γ → γ' = (γ+1) mod 3,
 *           ε → R(2π/3) ε,
 *           spins (in the local Kitaev frame): cyclic permutation
 *           (S^x, S^y, S^z) → (S^z, S^x, S^y).
 *
 *   7. Bond-modulation pattern matches the notes' Fig. 1: for a linearly
 *      polarized ε = (ε_0, 0), δX_γ ∝ (1, -1/2, -1/2) on (x, y, z) bonds.
 */

#include "classical_spin/lattice/phonon_lattice.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/core/spin_config.h"
#include "classical_spin/lattice/kitaev_bonds.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

constexpr double kPi          = 3.14159265358979323846;
constexpr double kSqrt3       = 1.7320508075688772935;
constexpr double kFDStep      = 1e-5;
constexpr double kFDFieldTol  = 1e-7;
constexpr double kAnalyticTol = 1e-12;

bool nearly_equal(double lhs, double rhs, double abs_tol, double rel_tol = 0.0) {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= abs_tol + rel_tol * scale;
}

// -------------------------------------------------------------------------
//  Reference formulas (mirrors of the boxed equations in tmfeo3_notes.tex).
// -------------------------------------------------------------------------

std::pair<double, double> n_gamma(int bond_type) {
    switch (bond_type) {
        case 0: return {1.0, 0.0};                   // θ_x = 0,    2θ = 0
        case 1: return {-0.5, -0.5 * kSqrt3};        // θ_y = 2π/3, 2θ = 4π/3
        default: return {-0.5,  0.5 * kSqrt3};       // θ_z = 4π/3, 2θ = 8π/3 ≡ 2π/3
    }
}

double delta_X_reference(double lambda0, double lambda2,
                         double qx, double qy, int bond_type) {
    const auto [c2, s2] = n_gamma(bond_type);
    const double q0 = qx * qx + qy * qy;
    const double qc = qx * qx - qy * qy;
    const double qs = 2.0 * qx * qy;
    return lambda0 * q0 + lambda2 * (qc * c2 + qs * s2);
}

// Per-bond magnetoelastic energy in the LOCAL Kitaev frame (notes' Eq. for
// δH_γ summed over channels). Used to check the C++ implementation matches
// the explicit channel expansion.
double bond_energy_reference_local(
    const Eigen::Vector3d& Si_local, const Eigen::Vector3d& Sj_local,
    const SpinPhononCouplingParams& p, double qx, double qy, int bond_type)
{
    const double dJ  = delta_X_reference(p.lambda_E1_J_0,      p.lambda_E1_J_2,      qx, qy, bond_type);
    const double dK  = delta_X_reference(p.lambda_E1_K_0,      p.lambda_E1_K_2,      qx, qy, bond_type);
    const double dG  = delta_X_reference(p.lambda_E1_Gamma_0,  p.lambda_E1_Gamma_2,  qx, qy, bond_type);
    const double dGp = delta_X_reference(p.lambda_E1_Gammap_0, p.lambda_E1_Gammap_2, qx, qy, bond_type);

    const int gamma = bond_type;
    const int alpha = (gamma == 0) ? 1 : 0;
    const int beta  = 3 - gamma - alpha;

    // O^{(J)}, O^{(K)}, O^{(Γ)}, O^{(Γ')} on this bond, in the local frame.
    const double O_J  = Si_local.dot(Sj_local);
    const double O_K  = Si_local(gamma) * Sj_local(gamma);
    const double O_G  = Si_local(alpha) * Sj_local(beta) + Si_local(beta) * Sj_local(alpha);
    const double O_Gp = Si_local(gamma) * (Sj_local(alpha) + Sj_local(beta))
                      + (Si_local(alpha) + Si_local(beta)) * Sj_local(gamma);

    return dJ * O_J + dK * O_K + dG * O_G + dGp * O_Gp;
}

// -------------------------------------------------------------------------
//  Lattice setup helpers.
// -------------------------------------------------------------------------

PhononLattice make_lattice(size_t L) {
    SpinConfig config;
    config.set_param("J",      -0.10);
    config.set_param("K",      -9.00);
    config.set_param("Gamma",   1.80);
    config.set_param("Gammap",  0.30);
    config.set_param("J2_A",    0.30);
    config.set_param("J2_B",    0.30);
    config.set_param("J3",      0.90);
    config.set_param("J7",      0.00);
    config.field_strength = 0.0;

    UnitCell uc = build_phonon_honeycomb(config);
    PhononLattice lattice(uc, L, L, 1, 1.0f);

    SpinPhononCouplingParams sp;
    sp.J = -0.10;  sp.K = -9.00;  sp.Gamma = 1.80;  sp.Gammap = 0.30;
    sp.J2_A = 0.30; sp.J2_B = 0.30; sp.J3 = 0.90; sp.J7 = 0.00;
    sp.lambda_E1_J_0      = 0.13;   sp.lambda_E1_J_2      = -0.21;
    sp.lambda_E1_K_0      = -0.42;  sp.lambda_E1_K_2      = 0.55;
    sp.lambda_E1_Gamma_0  = 0.07;   sp.lambda_E1_Gamma_2  = -0.18;
    sp.lambda_E1_Gammap_0 = -0.09;  sp.lambda_E1_Gammap_2 = 0.11;

    PhononParams ph;
    ph.omega_E1 = 1.5;  ph.gamma_E1 = 0.05;
    ph.lambda_E1_quartic = 0.4;  ph.Z_star = 1.0;

    DriveParams dr;  // all zero — no drive during these tests

    lattice.set_parameters(sp, ph, dr);
    return lattice;
}

void deterministic_spins(PhononLattice& L, double offset = 0.0) {
    for (size_t i = 0; i < L.lattice_size; ++i) {
        const double a = std::sin(0.317 * (i + 1) + offset);
        const double b = std::cos(0.611 * (i + 1) - 0.5 * offset);
        const double c = std::sin(0.233 * (i + 1) + 1.7 * offset);
        Eigen::Vector3d s(a, b, c);
        L.spins[i] = s.normalized() * L.spin_length;
    }
}

// -------------------------------------------------------------------------
//  Tests
// -------------------------------------------------------------------------

bool test_form_factor_sum_rule(std::ostream& out) {
    out << "[1] Form-factor sum rule  Σ_γ (cos 2θ_γ, sin 2θ_γ) = 0\n";
    double sx = 0.0, sy = 0.0;
    for (int g = 0; g < 3; ++g) {
        const auto [c2, s2] = n_gamma(g);
        out << "    γ=" << g << "  (cos 2θ, sin 2θ) = (" << c2 << ", " << s2 << ")\n";
        sx += c2; sy += s2;
    }
    out << "    Σ = (" << sx << ", " << sy << ")\n";
    if (!nearly_equal(sx, 0.0, kAnalyticTol) || !nearly_equal(sy, 0.0, kAnalyticTol)) {
        out << "[FAIL] form factor sum rule violated\n";
        return false;
    }
    out << "[PASS] form factor sum rule\n\n";
    return true;
}

bool test_bond_energy_matches_reference(std::ostream& out) {
    out << "[2] Per-bond magnetoelastic energy matches notes' channel expansion\n";
    PhononLattice L = make_lattice(2);
    deterministic_spins(L, 0.41);

    // Sweep a couple of nontrivial ε values
    const std::array<std::pair<double, double>, 5> eps_list = {{
        {0.0, 0.0}, {0.13, 0.0}, {0.0, 0.21}, {0.17, -0.09}, {-0.22, 0.31}
    }};

    const Eigen::Matrix3d R = SpinPhononCouplingParams::get_kitaev_rotation();
    double max_diff = 0.0;

    for (const auto& [qx, qy] : eps_list) {
        L.phonons.Q_x_E1 = qx;
        L.phonons.Q_y_E1 = qy;
        L.phonons.V_x_E1 = 0.0;
        L.phonons.V_y_E1 = 0.0;

        // Direct sum of per-bond reference energies (in local frame).
        double E_ref = 0.0;
        for (size_t i = 0; i < L.lattice_size; ++i) {
            const Eigen::Vector3d Si_local = R.transpose() * L.spins[i];
            for (size_t n = 0; n < L.nn_partners[i].size(); ++n) {
                const size_t j = L.nn_partners[i][n];
                if (j > i) {
                    const Eigen::Vector3d Sj_local = R.transpose() * L.spins[j];
                    const int g = L.nn_bond_types[i][n];
                    E_ref += bond_energy_reference_local(
                        Si_local, Sj_local, L.spin_phonon_params, qx, qy, g);
                }
            }
        }

        const double E_code = L.spin_phonon_energy();
        max_diff = std::max(max_diff, std::abs(E_code - E_ref));
        out << "    ε=(" << qx << "," << qy << ")  H_sp-ph (code)=" << E_code
            << "  reference=" << E_ref
            << "  diff=" << (E_code - E_ref) << "\n";
        if (!nearly_equal(E_code, E_ref, kAnalyticTol, 1e-10)) {
            out << "[FAIL] H_sp-ph mismatch vs reference channel expansion\n";
            return false;
        }
    }
    out << "    max |code − reference| = " << max_diff << "\n";
    out << "[PASS] per-bond energy formula\n\n";
    return true;
}

bool test_no_linear_coupling(std::ostream& out) {
    out << "[3] Linear-in-ε coupling is forbidden:  ∂H_sp-ph/∂ε_a |_{ε=0} = 0\n";
    PhononLattice L = make_lattice(3);
    deterministic_spins(L, 0.27);

    L.phonons = PhononState();  // ε = 0
    const double dHdqx = L.dH_dQx_E1();
    const double dHdqy = L.dH_dQy_E1();
    out << "    ∂H_sp-ph/∂ε_x|_{ε=0} = " << dHdqx << "\n";
    out << "    ∂H_sp-ph/∂ε_y|_{ε=0} = " << dHdqy << "\n";
    if (!nearly_equal(dHdqx, 0.0, kAnalyticTol) || !nearly_equal(dHdqy, 0.0, kAnalyticTol)) {
        out << "[FAIL] non-zero linear-in-ε coupling — forbidden by C_6\n";
        return false;
    }

    // Also verify the phonon energy itself is stationary at ε = 0:
    // H_sp-ph(ε=0) = 0 exactly (no constant offset).
    const double E0 = L.spin_phonon_energy();
    out << "    H_sp-ph(ε=0) = " << E0 << "\n";
    if (!nearly_equal(E0, 0.0, kAnalyticTol)) {
        out << "[FAIL] H_sp-ph(ε=0) must vanish\n";
        return false;
    }
    out << "[PASS] no linear-in-ε exchange-striction term\n\n";
    return true;
}

bool test_phonon_force_finite_difference(std::ostream& out) {
    out << "[4] Phonon force consistency (analytic vs finite-difference)\n";
    PhononLattice L = make_lattice(3);
    deterministic_spins(L, 0.83);

    const std::array<std::pair<double, double>, 3> eps_list = {{
        {0.05, -0.03}, {0.10, 0.10}, {-0.07, 0.12}
    }};

    auto H_sp_ph_at = [&](double qx, double qy) {
        L.phonons.Q_x_E1 = qx;
        L.phonons.Q_y_E1 = qy;
        return L.spin_phonon_energy();
    };

    for (const auto& [qx0, qy0] : eps_list) {
        L.phonons.Q_x_E1 = qx0;
        L.phonons.Q_y_E1 = qy0;
        const double dHx_an = L.dH_dQx_E1();
        const double dHy_an = L.dH_dQy_E1();

        const double Ep_x = H_sp_ph_at(qx0 + kFDStep, qy0);
        const double Em_x = H_sp_ph_at(qx0 - kFDStep, qy0);
        const double dHx_fd = (Ep_x - Em_x) / (2.0 * kFDStep);

        const double Ep_y = H_sp_ph_at(qx0, qy0 + kFDStep);
        const double Em_y = H_sp_ph_at(qx0, qy0 - kFDStep);
        const double dHy_fd = (Ep_y - Em_y) / (2.0 * kFDStep);

        out << "    ε=(" << qx0 << "," << qy0 << ")\n"
            << "      ∂_x: analytic=" << dHx_an << "  FD=" << dHx_fd
            << "  diff=" << (dHx_an - dHx_fd) << "\n"
            << "      ∂_y: analytic=" << dHy_an << "  FD=" << dHy_fd
            << "  diff=" << (dHy_an - dHy_fd) << "\n";

        if (!nearly_equal(dHx_an, dHx_fd, kFDFieldTol, 1e-6) ||
            !nearly_equal(dHy_an, dHy_fd, kFDFieldTol, 1e-6)) {
            out << "[FAIL] phonon force does not match finite difference\n";
            return false;
        }
    }
    out << "[PASS] phonon force consistency\n\n";
    return true;
}

bool test_spin_force_finite_difference(std::ostream& out) {
    out << "[5] Spin local-field consistency at finite ε (analytic vs finite-difference)\n";
    PhononLattice L = make_lattice(3);
    deterministic_spins(L, 0.71);
    L.phonons.Q_x_E1 = 0.13;
    L.phonons.Q_y_E1 = -0.08;

    // Compare H_eff_i = -∂E_total/∂S_i for a few sites.
    // We use a tangent-space perturbation that ignores |S|=const constraint
    // by directly comparing components 0/1/2 of H_eff against
    // -(E(S+δê_a) - E(S-δê_a))/(2 δ).
    const std::array<size_t, 4> probe_sites = {{0, 3, 7, 11}};
    double max_err = 0.0;

    for (size_t site : probe_sites) {
        const Eigen::Vector3d H_an = L.get_local_field(site);
        const Eigen::Vector3d S0   = L.spins[site];

        Eigen::Vector3d H_fd;
        for (int a = 0; a < 3; ++a) {
            Eigen::Vector3d delta = Eigen::Vector3d::Zero();
            delta(a) = kFDStep;
            L.spins[site] = S0 + delta;
            const double Ep = L.total_energy();
            L.spins[site] = S0 - delta;
            const double Em = L.total_energy();
            L.spins[site] = S0;
            H_fd(a) = -(Ep - Em) / (2.0 * kFDStep);
        }

        const double err = (H_an - H_fd).cwiseAbs().maxCoeff();
        max_err = std::max(max_err, err);
        out << "    site " << site
            << "  H_an=(" << H_an.transpose() << ")"
            << "  H_fd=(" << H_fd.transpose() << ")"
            << "  max|diff|=" << err << "\n";
    }

    if (max_err > 1e-6) {
        out << "[FAIL] H_eff disagrees with -∂E/∂S finite difference (max err=" << max_err << ")\n";
        return false;
    }
    out << "[PASS] spin local-field consistency  (max err=" << max_err << ")\n\n";
    return true;
}

bool test_C3_invariance_per_bond(std::ostream& out) {
    out << "[6] C_3 invariance of one-bond magnetoelastic energy under the\n"
        << "    combined transformation:  γ → γ' = (γ+1) mod 3,\n"
        << "                              ε → R(2π/3) ε,\n"
        << "                              S^x → S^z, S^y → S^x, S^z → S^y\n";
    SpinPhononCouplingParams p;
    p.lambda_E1_J_0      = 0.13;   p.lambda_E1_J_2      = -0.21;
    p.lambda_E1_K_0      = -0.42;  p.lambda_E1_K_2      = 0.55;
    p.lambda_E1_Gamma_0  = 0.07;   p.lambda_E1_Gamma_2  = -0.18;
    p.lambda_E1_Gammap_0 = -0.09;  p.lambda_E1_Gammap_2 = 0.11;

    // Two random spins in the local Kitaev frame (any unit-magnitude is fine).
    const Eigen::Vector3d Si(0.32, -0.84, 0.43);
    const Eigen::Vector3d Sj(-0.51, 0.27, -0.81);

    // ε in the (ε_x, ε_y) frame defined by θ_x = 0.
    const double qx = 0.21, qy = -0.13;

    // Cyclic permutation of spin components: (Sx, Sy, Sz) → (Sz, Sx, Sy)
    // i.e. if γ = 0 (x) is mapped to γ' = 1 (y), then under C_3 the spin
    // axes are permuted in the inverse direction, so that S^γ_new = S^γ_old.
    auto cycle = [](const Eigen::Vector3d& v) {
        return Eigen::Vector3d(v(2), v(0), v(1));
    };

    // R(2π/3) on ε.
    const double cos120 = -0.5;
    const double sin120 = 0.5 * kSqrt3;
    const double qx_new = cos120 * qx - sin120 * qy;
    const double qy_new = sin120 * qx + cos120 * qy;

    double max_diff = 0.0;
    for (int g = 0; g < 3; ++g) {
        const int g_new = (g + 1) % 3;

        const double E_orig = bond_energy_reference_local(Si, Sj, p, qx, qy, g);
        const double E_new  = bond_energy_reference_local(
            cycle(Si), cycle(Sj), p, qx_new, qy_new, g_new);
        const double diff = std::abs(E_orig - E_new);
        max_diff = std::max(max_diff, diff);
        out << "    γ=" << g << " → γ'=" << g_new
            << "  E_orig=" << E_orig
            << "  E_C3=" << E_new
            << "  diff=" << diff << "\n";
    }

    if (max_diff > 1e-12) {
        out << "[FAIL] one-bond energy is NOT C_3 invariant (max diff=" << max_diff << ")\n";
        return false;
    }
    out << "[PASS] C_3 invariance of one-bond magnetoelastic energy\n\n";
    return true;
}

bool test_bond_modulation_pattern(std::ostream& out) {
    out << "[7] Bond modulation pattern for ε = (ε_0, 0):\n"
        << "    δX_γ should be proportional to (1, -1/2, -1/2) on (x, y, z) bonds\n"
        << "    (cf. Fig. 1 of tmfeo3_notes.tex).\n";

    SpinPhononCouplingParams p;
    // Pure λ_X,2 channel (zero λ_X,0) so the bond pattern is purely
    // anisotropic. Use the Kitaev channel as a representative test.
    p.lambda_E1_K_0 = 0.0;
    p.lambda_E1_K_2 = 1.0;

    const double eps0 = 0.37;
    const std::array<double, 3> expected = {1.0, -0.5, -0.5};
    for (int g = 0; g < 3; ++g) {
        const double dK = delta_X_reference(p.lambda_E1_K_0, p.lambda_E1_K_2,
                                            eps0, 0.0, g);
        const double pattern = dK / (eps0 * eps0);
        out << "    γ=" << g << "  δK_γ/ε_0² = " << pattern
            << "   (expected " << expected[g] << ")\n";
        if (!nearly_equal(pattern, expected[g], kAnalyticTol)) {
            out << "[FAIL] bond pattern does not match Fig. 1 of the notes\n";
            return false;
        }
    }

    // For the second sanity check, use the rectified part: ε = (ε_0, 0)
    // → δX_γ = ε_0² (λ_0 + λ_2 cos 2θ_γ). Verify both pieces sum.
    p.lambda_E1_K_0 = 0.27;
    p.lambda_E1_K_2 = -0.31;
    for (int g = 0; g < 3; ++g) {
        const double dK = delta_X_reference(p.lambda_E1_K_0, p.lambda_E1_K_2,
                                            eps0, 0.0, g);
        const auto [c2, s2] = n_gamma(g);
        const double expect = eps0 * eps0 * (p.lambda_E1_K_0 + p.lambda_E1_K_2 * c2);
        if (!nearly_equal(dK, expect, kAnalyticTol)) {
            out << "[FAIL] δX_γ formula failed for γ=" << g
                << "  got=" << dK << " expected=" << expect << "\n";
            return false;
        }
    }
    out << "[PASS] bond modulation pattern matches the notes\n\n";
    return true;
}

bool test_isotropic_only_invariant_part(std::ostream& out) {
    out << "[8] Isotropic-only sweep:  with all λ_X,2 = 0, H_sp-ph depends only\n"
        << "    on |ε|² (rotational invariant).\n";
    PhononLattice L = make_lattice(3);
    deterministic_spins(L, 1.13);

    // Wipe out all λ_X,2 (keep λ_X,0 nonzero).
    L.spin_phonon_params.lambda_E1_J_2      = 0.0;
    L.spin_phonon_params.lambda_E1_K_2      = 0.0;
    L.spin_phonon_params.lambda_E1_Gamma_2  = 0.0;
    L.spin_phonon_params.lambda_E1_Gammap_2 = 0.0;

    const double r = 0.21;
    const std::array<double, 6> phis = {0.0, kPi/6, kPi/3, kPi/2, 2*kPi/3, kPi};
    double E_ref = 0.0;
    for (size_t k = 0; k < phis.size(); ++k) {
        L.phonons.Q_x_E1 = r * std::cos(phis[k]);
        L.phonons.Q_y_E1 = r * std::sin(phis[k]);
        const double E = L.spin_phonon_energy();
        out << "    φ=" << phis[k] << "  H_sp-ph=" << E << "\n";
        if (k == 0) E_ref = E;
        else if (!nearly_equal(E, E_ref, 1e-12, 1e-12)) {
            out << "[FAIL] H_sp-ph depends on direction of ε when λ_X,2 = 0\n";
            return false;
        }
    }
    out << "[PASS] isotropic limit is rotationally invariant\n\n";
    return true;
}

}  // namespace

int main() {
    std::cout << std::scientific << std::setprecision(8);

    bool ok = true;
    ok = test_form_factor_sum_rule(std::cout)            && ok;
    ok = test_bond_energy_matches_reference(std::cout)   && ok;
    ok = test_no_linear_coupling(std::cout)              && ok;
    ok = test_phonon_force_finite_difference(std::cout)  && ok;
    ok = test_spin_force_finite_difference(std::cout)    && ok;
    ok = test_C3_invariance_per_bond(std::cout)          && ok;
    ok = test_bond_modulation_pattern(std::cout)         && ok;
    ok = test_isotropic_only_invariant_part(std::cout)   && ok;

    if (!ok) {
        std::cout << "E1 phonon Hamiltonian regression FAILURES detected.\n";
        return 1;
    }
    std::cout << "All E1 phonon Hamiltonian regressions PASSED.\n";
    return 0;
}
