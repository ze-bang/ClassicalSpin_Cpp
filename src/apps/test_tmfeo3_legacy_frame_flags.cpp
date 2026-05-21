// Regression test: legacy TmFeO3 frame-selection flags are ignored.
//
// The current build_tmfeo3() implementation does not branch on
// `use_global_frame` or `use_local_frame`: Fe spins are always stored in lab
// Cartesian coordinates and Tm SU(3) states are always stored in the canonical
// local CEF basis.
//
// This regression verifies that supplying the retired flags does not change
// builder metadata, total energy, or the ODE right-hand side for a fixed
// stored-state configuration.

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using Lattice = MixedLattice;

constexpr double kAbsTol = 5e-11;
constexpr double kRelTol = 5e-10;

bool nearly_equal(double lhs, double rhs, double abs_tol, double rel_tol) {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= abs_tol + rel_tol * scale;
}

bool matrix_nearly_equal(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs,
                         double abs_tol, double rel_tol) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) return false;
    for (Eigen::Index row = 0; row < lhs.rows(); ++row) {
        for (Eigen::Index col = 0; col < lhs.cols(); ++col) {
            if (!nearly_equal(lhs(row, col), rhs(row, col), abs_tol, rel_tol)) {
                return false;
            }
        }
    }
    return true;
}

SpinConfig make_config(double field_strength) {
    SpinConfig config;
    config.field_strength = field_strength;
    config.field_direction = {0.20, -0.30, 0.85};

    config.set_param("J1ab", 4.74);
    config.set_param("J1c",  5.15);
    config.set_param("J2ab", 0.15);
    config.set_param("J2c",  0.30);
    config.set_param("Ka", -0.026);
    config.set_param("Kb",  0.0);
    config.set_param("Kc", -0.029);
    config.set_param("D1", 0.048);
    config.set_param("D2", 0.0);
    config.set_param("e1", 0.97);
    config.set_param("e2", 3.97);

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

    config.set_param("Jtm_2", 0.04);
    config.set_param("Jtm_5", 0.02);
    config.set_param("Jtm_7", -0.03);

    config.spin_length = 1.0f;
    config.spin_length_su3 = 1.0f;
    return config;
}

void normalize_in_place(Eigen::VectorXd& v, double target) {
    const double n = v.norm();
    if (n > 1e-14) v *= (target / n);
}

void build_stored_state(size_t lattice_size_su2, size_t lattice_size_su3,
                        size_t spin_dim_su2, size_t spin_dim_su3,
                        double spin_length_su2, double spin_length_su3,
                        uint32_t seed,
                        std::vector<Eigen::VectorXd>& fe_state,
                        std::vector<Eigen::VectorXd>& tm_state) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    fe_state.assign(lattice_size_su2, Eigen::VectorXd::Zero(spin_dim_su2));
    for (size_t s = 0; s < lattice_size_su2; ++s) {
        for (size_t d = 0; d < spin_dim_su2; ++d) fe_state[s](d) = uni(rng);
        normalize_in_place(fe_state[s], spin_length_su2);
    }

    tm_state.assign(lattice_size_su3, Eigen::VectorXd::Zero(spin_dim_su3));
    for (size_t s = 0; s < lattice_size_su3; ++s) {
        for (size_t d = 0; d < spin_dim_su3; ++d) tm_state[s](d) = uni(rng);
        normalize_in_place(tm_state[s], spin_length_su3);
    }
}

void assign_state(Lattice& L,
                  const std::vector<Eigen::VectorXd>& fe_state,
                  const std::vector<Eigen::VectorXd>& tm_state) {
    for (size_t s = 0; s < L.lattice_size_SU2; ++s) L.spins_SU2[s] = fe_state[s];
    for (size_t s = 0; s < L.lattice_size_SU3; ++s) L.spins_SU3[s] = tm_state[s];
}

bool compare_variant(const SpinConfig& base_cfg,
                     const SpinConfig& flagged_cfg,
                     const std::string& label,
                     uint32_t seed,
                     std::ostream& out) {
    MixedUnitCell uc_base = build_tmfeo3(base_cfg);
    MixedUnitCell uc_flagged = build_tmfeo3(flagged_cfg);

    Lattice L_base(uc_base, 2, 2, 2, base_cfg.spin_length, base_cfg.spin_length_su3);
    Lattice L_flagged(uc_flagged, 2, 2, 2, flagged_cfg.spin_length, flagged_cfg.spin_length_su3);

    if (L_base.lattice_size_SU2 != L_flagged.lattice_size_SU2 ||
        L_base.lattice_size_SU3 != L_flagged.lattice_size_SU3 ||
        L_base.spin_dim_SU2 != L_flagged.spin_dim_SU2 ||
        L_base.spin_dim_SU3 != L_flagged.spin_dim_SU3) {
        out << "[FAIL] lattice metadata mismatch for " << label << "\n";
        return false;
    }

    if (L_base.sublattice_frames_SU2.size() != L_flagged.sublattice_frames_SU2.size() ||
        L_base.sublattice_frames_SU3.size() != L_flagged.sublattice_frames_SU3.size()) {
        out << "[FAIL] sublattice-frame counts differ for " << label << "\n";
        return false;
    }
    for (size_t i = 0; i < L_base.sublattice_frames_SU2.size(); ++i) {
        if (!matrix_nearly_equal(L_base.sublattice_frames_SU2[i],
                                 L_flagged.sublattice_frames_SU2[i],
                                 kAbsTol, kRelTol)) {
            out << "[FAIL] SU2 sublattice frame mismatch for " << label
                << " at atom " << i << "\n";
            return false;
        }
    }
    for (size_t i = 0; i < L_base.sublattice_frames_SU3.size(); ++i) {
        if (!matrix_nearly_equal(L_base.sublattice_frames_SU3[i],
                                 L_flagged.sublattice_frames_SU3[i],
                                 kAbsTol, kRelTol)) {
            out << "[FAIL] SU3 sublattice frame mismatch for " << label
                << " at atom " << i << "\n";
            return false;
        }
    }

    std::vector<Eigen::VectorXd> fe_state, tm_state;
    build_stored_state(L_base.lattice_size_SU2, L_base.lattice_size_SU3,
                       L_base.spin_dim_SU2, L_base.spin_dim_SU3,
                       L_base.spin_length_SU2, L_base.spin_length_SU3,
                       seed, fe_state, tm_state);
    assign_state(L_base, fe_state, tm_state);
    assign_state(L_flagged, fe_state, tm_state);

    const double E_base = L_base.total_energy();
    const double E_flagged = L_flagged.total_energy();
    if (!nearly_equal(E_base, E_flagged, kAbsTol, kRelTol)) {
        out << "[FAIL] energy mismatch for " << label
            << ": E_base=" << E_base
            << ", E_flagged=" << E_flagged
            << ", |dE|=" << std::abs(E_base - E_flagged) << "\n";
        return false;
    }

    auto state_base = L_base.spins_to_state();
    auto state_flagged = L_flagged.spins_to_state();
    Lattice::ODEState d_base(state_base.size(), 0.0), d_flagged(state_flagged.size(), 0.0);
    L_base.ode_system(state_base, d_base, 0.0);
    L_flagged.ode_system(state_flagged, d_flagged, 0.0);

    if (d_base.size() != d_flagged.size()) {
        out << "[FAIL] derivative size mismatch for " << label << "\n";
        return false;
    }
    for (size_t i = 0; i < d_base.size(); ++i) {
        if (!nearly_equal(d_base[i], d_flagged[i], kAbsTol, kRelTol)) {
            out << "[FAIL] derivative mismatch for " << label
                << " at component " << i
                << ": d_base=" << d_base[i]
                << ", d_flagged=" << d_flagged[i] << "\n";
            return false;
        }
    }

    out << "[ok ] legacy flag ignored for " << label
        << "  (seed=" << seed << ", E=" << E_base << ")\n";
    return true;
}

}  // namespace

int main() {
    bool ok = true;

    for (double field_strength : {0.0, 0.07}) {
        for (uint32_t seed : {1u, 17u, 2025u}) {
            SpinConfig cfg_global0 = make_config(field_strength);
            cfg_global0.set_param("use_global_frame", 0.0);

            SpinConfig cfg_global1 = make_config(field_strength);
            cfg_global1.set_param("use_global_frame", 1.0);

            SpinConfig cfg_local1 = make_config(field_strength);
            cfg_local1.set_param("use_local_frame", 1.0);

            ok = compare_variant(make_config(field_strength), cfg_global0,
                                 "use_global_frame=0", seed, std::cout) && ok;
            ok = compare_variant(make_config(field_strength), cfg_global1,
                                 "use_global_frame=1", seed, std::cout) && ok;
            ok = compare_variant(make_config(field_strength), cfg_local1,
                                 "use_local_frame=1", seed, std::cout) && ok;
        }
    }

    if (!ok) {
        std::cout << "[FAIL] tmfeo3 legacy frame-flag regression\n";
        return 1;
    }

    std::cout << "[PASS] tmfeo3 legacy frame-selection flags are ignored\n";
    return 0;
}
