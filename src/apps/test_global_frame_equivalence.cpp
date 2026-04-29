// Regression test: TmFeO3 use_global_frame=0  ⇔  use_global_frame=1
//
// After the canonical local-frame patch in unitcell_builders.cpp, the two
// frame conventions must produce the *same* physical Hamiltonian.  The test
// below builds two MixedLattice objects from the same SpinConfig, one in
// each mode, instantiates a deterministic configuration of "physical" Fe
// and Tm spins, projects that physical state into each lattice's *stored*
// frame, and verifies that
//
//   1. total_energy() agrees to numerical precision
//   2. the LLG / SU(3) precession derivatives are related by the same
//      per-sublattice rotation that maps one stored representation to the
//      other (i.e. the local-frame dynamics is the rigid rotation of the
//      global-frame dynamics, as it must be for a physically-equivalent
//      Hamiltonian).
//
// Stored-frame mapping (constructed in build_tmfeo3 with
// use_global_frame=0/1):
//   Fe : R_i = diag(η_pbnm[i])         in mode 0,  identity in mode 1
//        S_local^(0)_i = R_i · S_phys_i,  S_local^(1)_i = S_phys_i
//   Tm : R_µ same in both modes (canonical CEF frame, see notes
//        §"Construction of the canonical local frames")
//        ⇒ Tm stored spins are identical between modes.

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using Lattice = MixedLattice;

constexpr double kAbsTol = 5e-11;
constexpr double kRelTol = 5e-10;

// Pbnm sublattice signs that drive the local-frame definition; must match
// `eta_pbnm` in src/core/unitcell_builders.cpp.
constexpr std::array<std::array<double, 3>, 4> kEtaPbnm = {{
    {1, 1, 1},
    {1, -1, -1},
    {-1, 1, -1},
    {-1, -1, 1},
}};

bool nearly_equal(double lhs, double rhs, double abs_tol, double rel_tol) {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= abs_tol + rel_tol * scale;
}

bool vector_nearly_equal(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs,
                         double abs_tol, double rel_tol) {
    if (lhs.size() != rhs.size()) return false;
    for (Eigen::Index i = 0; i < lhs.size(); ++i) {
        if (!nearly_equal(lhs(i), rhs(i), abs_tol, rel_tol)) return false;
    }
    return true;
}

SpinConfig make_config(double field_strength) {
    // Realistic-looking but generic TmFeO3 parameters with everything switched
    // on: bilinear chi, on-site W (u/v shorthand + a few off-diagonal channels),
    // inter-site V (w/wA shorthand), Tm-Tm bilinears, anisotropy, DM, full
    // Heisenberg, external field.  This exercises every code path that depends
    // on use_global_frame.
    SpinConfig config;
    config.field_strength = field_strength;
    config.field_direction = {0.20, -0.30, 0.85};   // arbitrary lab-frame field

    config.set_param("J1ab", 4.74);
    config.set_param("J1c",  5.15);
    config.set_param("J2ab", 0.15);
    config.set_param("J2c",  0.30);
    config.set_param("Ka", -0.16221);
    config.set_param("Kb",  0.0);
    config.set_param("Kc", -0.18318);
    config.set_param("D1", 0.12);
    config.set_param("D2", 0.04);
    config.set_param("e1", 0.97);
    config.set_param("e2", 3.97);

    // Fe-Tm chi
    config.set_param("chi2x", 0.13);
    config.set_param("chi2y", 0.07);
    config.set_param("chi2z", 0.21);
    config.set_param("chi5x", 0.11);
    config.set_param("chi5y", -0.05);
    config.set_param("chi5z", 0.09);
    config.set_param("chi7x", -0.06);
    config.set_param("chi7y", 0.04);
    config.set_param("chi7z", -0.18);
    config.set_param("chi_orbit1_scale", 1.42);
    config.set_param("chi_orbit2_scale", 1.16);
    config.set_param("chi_orbit3_scale", 0.88);
    config.set_param("chi_orbit4_scale", 0.54);

    // On-site W shorthands (u/v) + extra channels
    config.set_param("u1", 0.31);
    config.set_param("u3", -0.22);
    config.set_param("u8", 0.17);
    config.set_param("v4", 0.29);
    config.set_param("v6", -0.11);
    config.set_param("W1_xy", 0.05);
    config.set_param("W1_yz", -0.07);
    config.set_param("W3_xz", 0.04);
    config.set_param("W4_yy", 0.06);
    config.set_param("W6_xx", -0.03);
    config.set_param("W8_yz", 0.02);
    config.set_param("W_orbit1_scale", 0.95);
    config.set_param("W_orbit2_scale", 1.10);
    config.set_param("W_orbit3_scale", 1.02);
    config.set_param("W_orbit4_scale", 0.85);

    // Inter-site V shorthands (w/wA) + extras
    config.set_param("w1", 0.13);
    config.set_param("w3", 0.07);
    config.set_param("w8", -0.19);
    config.set_param("w4", 0.23);
    config.set_param("w6", 0.05);
    config.set_param("V1_Axy", 0.08);
    config.set_param("V3_Ayz", -0.04);
    config.set_param("V4_Axz", 0.06);
    config.set_param("V6_Axy", -0.02);
    config.set_param("V8_Ayz", 0.05);
    config.set_param("V_orbit1_scale", 1.05);
    config.set_param("V_orbit2_scale", 0.95);
    config.set_param("V_orbit3_scale", 1.00);
    config.set_param("V_orbit4_scale", 0.90);

    // Tm-Tm bilinears
    config.set_param("Jtm_2", 0.04);
    config.set_param("Jtm_5", 0.02);
    config.set_param("Jtm_7", -0.03);

    config.spin_length     = 1.0f;
    config.spin_length_su3 = 1.0f;

    return config;
}

void normalize_in_place(Eigen::VectorXd& v, double target) {
    const double n = v.norm();
    if (n > 1e-14) v *= (target / n);
}

// Build a deterministic "physical" spin configuration (Fe in lab Cartesian,
// Tm in its sublattice CEF basis).  The choice is irrelevant; we only need
// the same one in both modes.
void build_phys_state(size_t lattice_size_SU2, size_t lattice_size_SU3,
                      size_t N_atoms_SU2, size_t /*N_atoms_SU3*/,
                      size_t spin_dim_SU2, size_t spin_dim_SU3,
                      double spin_length_SU2, double spin_length_SU3,
                      uint32_t seed,
                      std::vector<Eigen::VectorXd>& fe_phys,
                      std::vector<Eigen::VectorXd>& tm_phys) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    fe_phys.assign(lattice_size_SU2, Eigen::VectorXd::Zero(spin_dim_SU2));
    for (size_t s = 0; s < lattice_size_SU2; ++s) {
        for (size_t d = 0; d < spin_dim_SU2; ++d) fe_phys[s](d) = uni(rng);
        normalize_in_place(fe_phys[s], spin_length_SU2);
        // sublattice info is implicit in s % N_atoms_SU2 — used by the caller.
        (void)N_atoms_SU2;
    }

    tm_phys.assign(lattice_size_SU3, Eigen::VectorXd::Zero(spin_dim_SU3));
    for (size_t s = 0; s < lattice_size_SU3; ++s) {
        for (size_t d = 0; d < spin_dim_SU3; ++d) tm_phys[s](d) = uni(rng);
        normalize_in_place(tm_phys[s], spin_length_SU3);
    }
}

// Project the physical state into a given lattice's stored frame.
//   Fe stored = R_i^T · phys = diag(η_pbnm[i]) · phys  in mode 0
//             = phys                                    in mode 1
//   Tm stored = (R_µ)^T · phys                          (same R_µ in both modes)
void project_into_lattice(Lattice& L, bool global_mode,
                          const std::vector<Eigen::VectorXd>& fe_phys,
                          const std::vector<Eigen::VectorXd>& tm_phys) {
    for (size_t s = 0; s < L.lattice_size_SU2; ++s) {
        const size_t atom = s % L.N_atoms_SU2;
        Eigen::VectorXd v = fe_phys[s];
        if (!global_mode) {
            for (Eigen::Index a = 0; a < v.size(); ++a) v(a) *= kEtaPbnm[atom][a];
        }
        L.spins_SU2[s] = v;
    }
    for (size_t s = 0; s < L.lattice_size_SU3; ++s) {
        const size_t atom = s % L.N_atoms_SU3;
        // S^(stored) = sublattice_frames^T · S^(global).  In our convention the
        // "physical Tm spin" tm_phys is already the *common* basis Bloch vector
        // (i.e. the global-Cartesian-J Bloch vector embedded in the 8d Gell-Mann
        // space), so we apply the frame transpose on each site.
        L.spins_SU3[s] = L.sublattice_frames_SU3[atom].transpose() * tm_phys[s];
    }
}

bool check_energy_equivalence(double field_strength, uint32_t seed,
                              std::ostream& out) {
    SpinConfig cfg0 = make_config(field_strength);
    cfg0.set_param("use_global_frame", 0.0);
    SpinConfig cfg1 = make_config(field_strength);
    cfg1.set_param("use_global_frame", 1.0);

    MixedUnitCell uc0 = build_tmfeo3(cfg0);
    MixedUnitCell uc1 = build_tmfeo3(cfg1);

    Lattice L0(uc0, 2, 2, 2, cfg0.spin_length, cfg0.spin_length_su3);
    Lattice L1(uc1, 2, 2, 2, cfg1.spin_length, cfg1.spin_length_su3);

    if (L0.lattice_size_SU2 != L1.lattice_size_SU2 ||
        L0.lattice_size_SU3 != L1.lattice_size_SU3) {
        out << "[FAIL] lattice sizes differ between modes\n";
        return false;
    }

    std::vector<Eigen::VectorXd> fe_phys, tm_phys;
    build_phys_state(L0.lattice_size_SU2, L0.lattice_size_SU3,
                     L0.N_atoms_SU2, L0.N_atoms_SU3,
                     L0.spin_dim_SU2, L0.spin_dim_SU3,
                     L0.spin_length_SU2, L0.spin_length_SU3,
                     seed, fe_phys, tm_phys);

    project_into_lattice(L0, /*global_mode=*/false, fe_phys, tm_phys);
    project_into_lattice(L1, /*global_mode=*/true,  fe_phys, tm_phys);

    const double E0 = L0.total_energy();
    const double E1 = L1.total_energy();

    if (!nearly_equal(E0, E1, kAbsTol, kRelTol)) {
        out << "[FAIL] total_energy mismatch (h=" << field_strength
            << ", seed=" << seed << "): E_local=" << E0
            << ", E_global=" << E1 << ", |dE|=" << std::abs(E0 - E1) << "\n";
        return false;
    }
    out << "[ok ] total_energy(local)=" << E0
        << "  total_energy(global)=" << E1
        << "  |dE|=" << std::abs(E0 - E1)
        << "  (h=" << field_strength << ", seed=" << seed << ")\n";
    return true;
}

// The LLG / SU(3) precession derivative dx/dt is computed by the runtime in
// the *stored* frame.  Two equivalent Hamiltonians related by S_local = R^T·S
// must produce derivatives related by the same rotation.  For Tm the R_µ
// frames are identical between modes, so the SU(3) derivative must agree
// directly.  For Fe the per-sublattice diag(η) maps mode-0 derivatives to
// mode-1 derivatives.
bool check_derivative_equivalence(double field_strength, uint32_t seed,
                                  std::ostream& out) {
    SpinConfig cfg0 = make_config(field_strength);
    cfg0.set_param("use_global_frame", 0.0);
    SpinConfig cfg1 = make_config(field_strength);
    cfg1.set_param("use_global_frame", 1.0);

    MixedUnitCell uc0 = build_tmfeo3(cfg0);
    MixedUnitCell uc1 = build_tmfeo3(cfg1);

    Lattice L0(uc0, 2, 2, 2, cfg0.spin_length, cfg0.spin_length_su3);
    Lattice L1(uc1, 2, 2, 2, cfg1.spin_length, cfg1.spin_length_su3);

    std::vector<Eigen::VectorXd> fe_phys, tm_phys;
    build_phys_state(L0.lattice_size_SU2, L0.lattice_size_SU3,
                     L0.N_atoms_SU2, L0.N_atoms_SU3,
                     L0.spin_dim_SU2, L0.spin_dim_SU3,
                     L0.spin_length_SU2, L0.spin_length_SU3,
                     seed, fe_phys, tm_phys);

    project_into_lattice(L0, false, fe_phys, tm_phys);
    project_into_lattice(L1, true,  fe_phys, tm_phys);

    L0.reset_pulse();
    L1.reset_pulse();

    auto state0 = L0.spins_to_state();
    auto state1 = L1.spins_to_state();
    Lattice::ODEState d0(state0.size(), 0.0), d1(state1.size(), 0.0);
    L0.ode_system(state0, d0, 0.0);
    L1.ode_system(state1, d1, 0.0);

    const size_t d2  = L0.spin_dim_SU2;
    const size_t d3  = L0.spin_dim_SU3;
    const size_t off3 = L0.lattice_size_SU2 * d2;

    // Fe sector: dx_local^(0) = R_i · dx_local^(1) (with R_i = diag(η_i)),
    // because S_local^(0) = R_i · S_local^(1) and the LLG operator is
    // covariant under rotation of all 3-vectors.
    for (size_t s = 0; s < L0.lattice_size_SU2; ++s) {
        const size_t atom = s % L0.N_atoms_SU2;
        Eigen::VectorXd dx0(d2), dx1(d2);
        for (size_t a = 0; a < d2; ++a) {
            dx0(a) = d0[s * d2 + a];
            dx1(a) = d1[s * d2 + a];
        }
        Eigen::VectorXd dx1_in_local0(d2);
        for (size_t a = 0; a < d2; ++a) dx1_in_local0(a) = kEtaPbnm[atom][a] * dx1(a);
        if (!vector_nearly_equal(dx0, dx1_in_local0, kAbsTol, kRelTol)) {
            out << "[FAIL] Fe derivative mismatch at site " << s
                << " (sublattice " << atom << "): h=" << field_strength
                << ", seed=" << seed << "\n";
            out << "       dx_local0 = " << dx0.transpose() << "\n";
            out << "       R·dx_loc1 = " << dx1_in_local0.transpose() << "\n";
            return false;
        }
    }

    // Tm sector: identical sublattice frames in both modes ⇒ derivatives must
    // match directly.
    for (size_t s = 0; s < L0.lattice_size_SU3; ++s) {
        Eigen::VectorXd dx0(d3), dx1(d3);
        for (size_t a = 0; a < d3; ++a) {
            dx0(a) = d0[off3 + s * d3 + a];
            dx1(a) = d1[off3 + s * d3 + a];
        }
        if (!vector_nearly_equal(dx0, dx1, kAbsTol, kRelTol)) {
            out << "[FAIL] Tm derivative mismatch at site " << s
                << ": h=" << field_strength << ", seed=" << seed << "\n";
            out << "       dx0 = " << dx0.transpose() << "\n";
            out << "       dx1 = " << dx1.transpose() << "\n";
            return false;
        }
    }

    out << "[ok ] LLG derivatives agree under per-sublattice η rotation"
        << "  (h=" << field_strength << ", seed=" << seed << ")\n";
    return true;
}

}  // namespace

int main() {
    bool ok = true;

    // Several seeds + a representative h=0 and h≠0 case; the latter
    // exercises the per-sublattice η on the Fe Zeeman field that the patch
    // also fixes.
    for (uint32_t seed : {1u, 17u, 2025u}) {
        ok = check_energy_equivalence(0.0, seed, std::cout) && ok;
        ok = check_energy_equivalence(0.07, seed, std::cout) && ok;
        ok = check_derivative_equivalence(0.0, seed, std::cout) && ok;
        ok = check_derivative_equivalence(0.07, seed, std::cout) && ok;
    }

    if (!ok) {
        std::cout << "[FAIL] global-frame equivalence regression\n";
        return 1;
    }
    std::cout << "[PASS] use_global_frame=0 ⇔ use_global_frame=1 equivalence\n";
    return 0;
}
