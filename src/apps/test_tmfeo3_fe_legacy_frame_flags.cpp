// Regression test: legacy TmFeO3 Fe-only frame-selection flags are ignored.
//
// The current build_tmfeo3_fe() implementation no longer switches between a
// local and global Fe storage convention. This test verifies that the retired
// `use_global_frame` / `use_local_frame` flags do not change builder metadata,
// total energy, or the LLG derivative for a fixed stored-state configuration.

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/lattice.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

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
    config.set_param("Ka", -0.16221);
    config.set_param("Kb",  0.012);
    config.set_param("Kc", -0.18318);
    config.set_param("D1", 0.12);
    config.set_param("D2", 0.04);
    config.spin_length = 1.0f;
    return config;
}

void normalize_in_place(Eigen::VectorXd& v, double target) {
    const double n = v.norm();
    if (n > 1e-14) v *= (target / n);
}

void build_stored_state(size_t lattice_size, size_t spin_dim, double spin_length,
                        uint32_t seed, std::vector<Eigen::VectorXd>& fe_state) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    fe_state.assign(lattice_size, Eigen::VectorXd::Zero(spin_dim));
    for (size_t s = 0; s < lattice_size; ++s) {
        for (size_t d = 0; d < spin_dim; ++d) fe_state[s](d) = uni(rng);
        normalize_in_place(fe_state[s], spin_length);
    }
}

void assign_state(Lattice& L, const std::vector<Eigen::VectorXd>& fe_state) {
    for (size_t s = 0; s < L.lattice_size; ++s) L.spins[s] = fe_state[s];
}

bool compare_variant(const SpinConfig& base_cfg,
                     const SpinConfig& flagged_cfg,
                     const std::string& label,
                     uint32_t seed,
                     std::ostream& out) {
    UnitCell uc_base = build_tmfeo3_fe(base_cfg);
    UnitCell uc_flagged = build_tmfeo3_fe(flagged_cfg);

    Lattice L_base(uc_base, 2, 2, 2, base_cfg.spin_length);
    Lattice L_flagged(uc_flagged, 2, 2, 2, flagged_cfg.spin_length);

    if (L_base.lattice_size != L_flagged.lattice_size ||
        L_base.spin_dim != L_flagged.spin_dim ||
        L_base.N_atoms != L_flagged.N_atoms) {
        out << "[FAIL] lattice metadata mismatch for " << label << "\n";
        return false;
    }

    if (L_base.sublattice_frames.size() != L_flagged.sublattice_frames.size()) {
        out << "[FAIL] sublattice-frame counts differ for " << label << "\n";
        return false;
    }
    for (size_t i = 0; i < L_base.sublattice_frames.size(); ++i) {
        if (!matrix_nearly_equal(L_base.sublattice_frames[i],
                                 L_flagged.sublattice_frames[i],
                                 kAbsTol, kRelTol)) {
            out << "[FAIL] sublattice frame mismatch for " << label
                << " at atom " << i << "\n";
            return false;
        }
    }

    std::vector<Eigen::VectorXd> fe_state;
    build_stored_state(L_base.lattice_size, L_base.spin_dim, L_base.spin_length,
                       seed, fe_state);
    assign_state(L_base, fe_state);
    assign_state(L_flagged, fe_state);

    const double E_base = L_base.total_energy();
    const double E_flagged = L_flagged.total_energy();
    if (!nearly_equal(E_base, E_flagged, kAbsTol, kRelTol)) {
        out << "[FAIL] energy mismatch for " << label
            << ": E_base=" << E_base
            << ", E_flagged=" << E_flagged
            << ", |dE|=" << std::abs(E_base - E_flagged) << "\n";
        return false;
    }

    auto state_base = L_base.spins_to_state(L_base.spins);
    auto state_flagged = L_flagged.spins_to_state(L_flagged.spins);
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
        std::cout << "[FAIL] tmfeo3 Fe-only legacy frame-flag regression\n";
        return 1;
    }

    std::cout << "[PASS] tmfeo3 Fe-only legacy frame-selection flags are ignored\n";
    return 0;
}
