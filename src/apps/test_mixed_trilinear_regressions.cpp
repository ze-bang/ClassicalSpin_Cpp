#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

namespace {

using Lattice = MixedLattice;

constexpr double kAbsTol = 1e-11;
constexpr double kRelTol = 1e-10;

bool nearly_equal(double lhs, double rhs, double abs_tol, double rel_tol) {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= abs_tol + rel_tol * scale;
}

bool vector_nearly_equal(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs,
                        double abs_tol, double rel_tol) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (Eigen::Index i = 0; i < lhs.size(); ++i) {
        if (!nearly_equal(lhs(i), rhs(i), abs_tol, rel_tol)) {
            return false;
        }
    }

    return true;
}

Eigen::VectorXd su2_precession(const Eigen::VectorXd& field, const Eigen::VectorXd& spin) {
    Eigen::VectorXd dsdt(3);
    dsdt(0) = field(1) * spin(2) - field(2) * spin(1);
    dsdt(1) = field(2) * spin(0) - field(0) * spin(2);
    dsdt(2) = field(0) * spin(1) - field(1) * spin(0);
    return dsdt;
}

Eigen::VectorXd su3_precession(const Eigen::VectorXd& field, const Eigen::VectorXd& spin) {
    const auto& f = get_SU3_structure();
    Eigen::VectorXd dsdt = Eigen::VectorXd::Zero(field.size());

    for (Eigen::Index i = 0; i < field.size(); ++i) {
        for (Eigen::Index j = 0; j < field.size(); ++j) {
            for (Eigen::Index k = 0; k < field.size(); ++k) {
                dsdt(i) += f[static_cast<size_t>(i)](j, k) * field(j) * spin(k);
            }
        }
    }

    return dsdt;
}

Eigen::VectorXd normalized_vector(const std::vector<double>& components, double length) {
    Eigen::VectorXd vec(components.size());
    for (size_t i = 0; i < components.size(); ++i) {
        vec(static_cast<Eigen::Index>(i)) = components[i];
    }

    const double norm = vec.norm();
    if (norm == 0.0) {
        vec.setZero();
        vec(0) = length;
        return vec;
    }
    return vec * (length / norm);
}

SpinConfig make_base_config() {
    SpinConfig config;
    config.field_strength = 0.0;

    config.set_param("J1ab", 0.0);
    config.set_param("J1c", 0.0);
    config.set_param("J2ab", 0.0);
    config.set_param("J2c", 0.0);
    config.set_param("Ka", 0.0);
    config.set_param("Kb", 0.0);
    config.set_param("Kc", 0.0);
    config.set_param("D1", 0.0);
    config.set_param("D2", 0.0);
    config.set_param("e1", 0.0);
    config.set_param("e2", 0.0);
    config.spin_length = 1.0f;
    config.spin_length_su3 = 1.0f;

    return config;
}

void set_reference_symmetric_tensor_terms(SpinConfig& config) {
    // On-site W (a∈{1,3,4,6,8}). u_n→W{n}_zz=+u_n, W{n}_xx=−u_n; v_n→W{n}_xz=+v_n.
    config.set_param("W1_zz",  0.31);  config.set_param("W1_xx", -0.31);   // u1 = 0.31
    config.set_param("W3_zz", -0.22);  config.set_param("W3_xx",  0.22);   // u3 = -0.22
    config.set_param("W8_zz",  0.17);  config.set_param("W8_xx", -0.17);   // u8 = 0.17
    config.set_param("W4_xz",  0.29);                                       // v4 = 0.29
    config.set_param("W6_xz", -0.11);                                       // v6 = -0.11
}

void set_reference_shorthand_terms(SpinConfig& config) {
    config.set_param("u1",  0.31);
    config.set_param("u3", -0.22);
    config.set_param("u8",  0.17);
    config.set_param("v4",  0.29);
    config.set_param("v6", -0.11);
}

SpinConfig make_tensor_config() {
    SpinConfig config = make_base_config();
    set_reference_symmetric_tensor_terms(config);
    return config;
}

SpinConfig make_tensor_reference_config() {
    SpinConfig config = make_base_config();
    set_reference_symmetric_tensor_terms(config);
    return config;
}

SpinConfig make_shorthand_config() {
    SpinConfig config = make_base_config();
    set_reference_shorthand_terms(config);
    return config;
}

MixedUnitCell make_mixed_unit_cell(const SpinConfig& config) {
    return build_tmfeo3(config);
}

Lattice make_lattice_from_config(const SpinConfig& config,
                                 size_t* bilinear_count = nullptr,
                                 size_t* trilinear_count = nullptr) {
    MixedUnitCell mixed_uc = make_mixed_unit_cell(config);
    if (bilinear_count != nullptr) {
        *bilinear_count = mixed_uc.bilinear_SU2_SU3.size();
    }
    if (trilinear_count != nullptr) {
        *trilinear_count = mixed_uc.trilinear_SU2_SU3.size();
    }

    return Lattice(mixed_uc, 1, 1, 1, config.spin_length, config.spin_length_su3);
}

Lattice make_lattice() {
    return make_lattice_from_config(make_tensor_config());
}

void assign_deterministic_spins(Lattice& lattice) {
    for (size_t site = 0; site < lattice.lattice_size_SU2; ++site) {
        const double x = std::sin(0.41 * (site + 1) + 0.2);
        const double y = std::cos(0.67 * (site + 1) - 0.5);
        const double z = std::sin(0.29 * (site + 1) + 1.1);
        lattice.spins_SU2[site] = normalized_vector({x, y, z}, lattice.spin_length_SU2);
    }

    for (size_t site = 0; site < lattice.lattice_size_SU3; ++site) {
        std::vector<double> comps(lattice.spin_dim_SU3);
        for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
            comps[d] = std::sin(0.23 * (site + 1) + 0.37 * (d + 1)) +
                       std::cos(0.19 * (site + 1) - 0.11 * (d + 1));
        }
        lattice.spins_SU3[site] = normalized_vector(comps, lattice.spin_length_SU3);
    }
}

void assign_uniform_q0_spins(Lattice& lattice) {
    const Eigen::VectorXd spin = normalized_vector({1.0, 0.0, 1.0}, lattice.spin_length_SU2);
    for (size_t site = 0; site < lattice.lattice_size_SU2; ++site) {
        lattice.spins_SU2[site] = spin;
    }
    for (size_t site = 0; site < lattice.lattice_size_SU3; ++site) {
        lattice.spins_SU3[site] = Eigen::VectorXd::Zero(lattice.spin_dim_SU3);
    }
}

void set_active_w_orbit(SpinConfig& config, int orbit) {
    config.set_param("W_orbit1_scale", orbit == 1 ? 1.0 : 0.0);
    config.set_param("W_orbit2_scale", orbit == 2 ? 1.0 : 0.0);
    config.set_param("W_orbit3_scale", orbit == 3 ? 1.0 : 0.0);
    config.set_param("W_orbit4_scale", orbit == 4 ? 1.0 : 0.0);
}

bool check_total_energy_flat_consistency(Lattice& lattice, std::ostream& out) {
    const double energy = lattice.total_energy();
    auto state = lattice.spins_to_state();
    const double flat_energy = lattice.total_energy_flat(state.data());

    if (!nearly_equal(energy, flat_energy, kAbsTol, kRelTol)) {
        out << "[FAIL] total_energy and total_energy_flat disagree: E=" << energy
            << ", E_flat=" << flat_energy << "\n";
        return false;
    }

    return true;
}

bool check_w_only_builder_surface(std::ostream& out) {
    MixedUnitCell mixed_uc = make_mixed_unit_cell(make_tensor_reference_config());

    if (!mixed_uc.bilinear_SU2_SU3.empty()) {
        out << "[FAIL] W-only builder emitted unexpected mixed bilinear terms: "
            << mixed_uc.bilinear_SU2_SU3.size() << "\n";
        return false;
    }
    if (mixed_uc.trilinear_SU2_SU3.size() != 32) {
        out << "[FAIL] W-only builder emitted " << mixed_uc.trilinear_SU2_SU3.size()
            << " mixed trilinears instead of 32\n";
        return false;
    }

    for (const auto& entry : mixed_uc.trilinear_SU2_SU3) {
        const size_t source = static_cast<size_t>(entry.first);
        const MixedTrilinear& term = entry.second;
        if (source != term.partner1) {
            out << "[FAIL] Found non-onsite mixed trilinear at Fe source " << source
                << " with partner1=" << term.partner1 << "\n";
            return false;
        }
        if ((term.offset1.array() != 0).any()) {
            out << "[FAIL] Found nonzero Fe-leg offset in W-only builder at Fe source "
                << source << "\n";
            return false;
        }
    }

    return true;
}

bool check_shorthand_equivalence(std::ostream& out) {
    size_t tensor_bilinear_count = 0;
    size_t tensor_trilinear_count = 0;
    size_t shorthand_bilinear_count = 0;
    size_t shorthand_trilinear_count = 0;

    Lattice tensor_lattice = make_lattice_from_config(
        make_tensor_reference_config(), &tensor_bilinear_count, &tensor_trilinear_count);
    Lattice shorthand_lattice = make_lattice_from_config(
        make_shorthand_config(), &shorthand_bilinear_count, &shorthand_trilinear_count);

    if (tensor_bilinear_count != 0 || shorthand_bilinear_count != 0) {
        out << "[FAIL] W-only builder emitted unexpected mixed bilinears during shorthand test"
            << " (tensor=" << tensor_bilinear_count
            << ", shorthand=" << shorthand_bilinear_count << ")\n";
        return false;
    }
    if (tensor_trilinear_count == 0 || shorthand_trilinear_count == 0) {
        out << "[FAIL] Mixed trilinear builder returned no trilinear terms"
            << " (tensor=" << tensor_trilinear_count
            << ", shorthand=" << shorthand_trilinear_count << ")\n";
        return false;
    }
    if (tensor_trilinear_count != shorthand_trilinear_count) {
        out << "[FAIL] Tensor/shorthand trilinear counts differ: tensor="
            << tensor_trilinear_count << ", shorthand=" << shorthand_trilinear_count << "\n";
        return false;
    }

    assign_deterministic_spins(tensor_lattice);
    assign_deterministic_spins(shorthand_lattice);

    const double tensor_energy = tensor_lattice.total_energy();
    const double shorthand_energy = shorthand_lattice.total_energy();
    if (!nearly_equal(tensor_energy, shorthand_energy, kAbsTol, kRelTol)) {
        out << "[FAIL] Tensor/shorthand total energies differ: tensor=" << tensor_energy
            << ", shorthand=" << shorthand_energy << "\n";
        return false;
    }

    for (size_t site = 0; site < tensor_lattice.lattice_size_SU2; ++site) {
        const auto tensor_field = tensor_lattice.get_local_field_SU2(site);
        const auto shorthand_field = shorthand_lattice.get_local_field_SU2(site);
        if (!vector_nearly_equal(tensor_field, shorthand_field, kAbsTol, kRelTol)) {
            out << "[FAIL] Tensor/shorthand SU2 local field mismatch at site " << site << "\n";
            return false;
        }
    }

    for (size_t site = 0; site < tensor_lattice.lattice_size_SU3; ++site) {
        const auto tensor_field = tensor_lattice.get_local_field_SU3(site);
        const auto shorthand_field = shorthand_lattice.get_local_field_SU3(site);
        if (!vector_nearly_equal(tensor_field, shorthand_field, kAbsTol, kRelTol)) {
            out << "[FAIL] Tensor/shorthand SU3 local field mismatch at site " << site << "\n";
            return false;
        }
    }

    return true;
}

bool check_mirror_odd_inversion_pairing(std::ostream& out) {
    // The mirror-odd A2 channels (lambda4, lambda6; inversion parity p=-1) are
    // ANTISYMMETRIC between inversion-paired Tm sites.  Pbnm inversion permutes
    // the Tm sublattices by sigma^Tm(I) = (03)(12) and sends lambda4,6 -> -lambda4,6,
    // while the uniform q=0 Fe spins are invariant (axial, Fe at inversion centers).
    // Hence the static A2 field obeys  h(Tm0) = -h(Tm3),  h(Tm1) = -h(Tm2).
    //
    // (Note: the earlier frame-less W builder produced an accidental per-site
    // cancellation h=0; with the Pbnm-covariant sigma_C Fe frames now applied to
    // W, the correct symmetry statement is this inversion antisymmetry, verified
    // term-by-term in diag_tmfeo3_pbnm_invariance.)
    struct OddChannelCase {
        const char* param_name;
        Eigen::Index lambda_index;
    };
    const OddChannelCase odd_cases[] = {
        {"W4_xz", 3},
        {"W6_xz", 5}
    };
    // Inversion-paired Tm sublattices, sigma^Tm(I) = (03)(12).
    const std::array<std::pair<int, int>, 2> inv_pairs = {{{0, 3}, {1, 2}}};

    for (const auto& odd_case : odd_cases) {
        for (int orbit = 1; orbit <= 4; ++orbit) {
            SpinConfig config = make_base_config();
            config.set_param(odd_case.param_name, 1.0);
            set_active_w_orbit(config, orbit);

            Lattice lattice = make_lattice_from_config(config);
            assign_uniform_q0_spins(lattice);

            std::array<double, 4> h{};
            for (size_t site = 0; site < lattice.lattice_size_SU3 && site < 4; ++site) {
                h[site] = lattice.get_local_field_SU3(site)(odd_case.lambda_index);
            }

            for (const auto& pr : inv_pairs) {
                if (!nearly_equal(h[pr.first], -h[pr.second], kAbsTol, kRelTol)) {
                    out << "[FAIL] Odd W channel " << odd_case.param_name
                        << " not inversion-antisymmetric for orbit " << orbit
                        << ": h(Tm" << pr.first << ")=" << h[pr.first]
                        << ", h(Tm" << pr.second << ")=" << h[pr.second]
                        << " (expected h(Tm" << pr.first << ") = -h(Tm" << pr.second << "))\n";
                    return false;
                }
            }
            // Guard against a trivial all-zero pass: the channel must be active.
            const double max_abs =
                std::max({std::abs(h[0]), std::abs(h[1]), std::abs(h[2]), std::abs(h[3])});
            if (max_abs < kAbsTol) {
                out << "[FAIL] Odd W channel " << odd_case.param_name
                    << " produced no field at all for orbit " << orbit
                    << " (expected a nonzero A2 field)\n";
                return false;
            }
        }
    }

    return true;
}

bool check_state_roundtrip(Lattice& lattice, std::ostream& out) {
    const auto state = lattice.spins_to_state();

    Lattice::SpinConfigSU2 spins2;
    Lattice::SpinConfigSU3 spins3;
    lattice.state_to_spins(state, spins2, spins3);

    for (size_t site = 0; site < lattice.lattice_size_SU2; ++site) {
        if (!vector_nearly_equal(spins2[site], lattice.spins_SU2[site], kAbsTol, kRelTol)) {
            out << "[FAIL] SU2 state roundtrip mismatch at site " << site << "\n";
            return false;
        }
    }

    for (size_t site = 0; site < lattice.lattice_size_SU3; ++site) {
        if (!vector_nearly_equal(spins3[site], lattice.spins_SU3[site], kAbsTol, kRelTol)) {
            out << "[FAIL] SU3 state roundtrip mismatch at site " << site << "\n";
            return false;
        }
    }

    return true;
}

bool check_su2_delta_energy(Lattice& lattice, std::ostream& out) {
    for (size_t site = 0; site < lattice.lattice_size_SU2; ++site) {
        const Eigen::VectorXd old_spin = lattice.spins_SU2[site];
        const Eigen::VectorXd new_spin = normalized_vector(
            {std::cos(0.53 * (site + 1) + 0.1),
             std::sin(0.79 * (site + 1) - 0.4),
             std::cos(0.31 * (site + 1) + 0.8)},
            lattice.spin_length_SU2);

        const double energy_before = lattice.total_energy();
        const double delta_energy = lattice.site_energy_SU2_diff(new_spin, old_spin, site);

        lattice.spins_SU2[site] = new_spin;
        const double energy_after = lattice.total_energy();
        lattice.spins_SU2[site] = old_spin;

        const double exact_delta = energy_after - energy_before;
        if (!nearly_equal(delta_energy, exact_delta, kAbsTol, kRelTol)) {
            out << "[FAIL] SU2 delta-energy mismatch at site " << site
                << ": dE_site=" << delta_energy
                << ", dE_exact=" << exact_delta << "\n";
            return false;
        }
    }

    return true;
}

bool check_su3_delta_energy(Lattice& lattice, std::ostream& out) {
    for (size_t site = 0; site < lattice.lattice_size_SU3; ++site) {
        const Eigen::VectorXd old_spin = lattice.spins_SU3[site];
        std::vector<double> comps(lattice.spin_dim_SU3);
        for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
            comps[d] = std::cos(0.17 * (site + 1) + 0.29 * (d + 1)) -
                       std::sin(0.13 * (site + 1) - 0.07 * (d + 1));
        }
        const Eigen::VectorXd new_spin = normalized_vector(comps, lattice.spin_length_SU3);

        const double energy_before = lattice.total_energy();
        const double delta_energy = lattice.site_energy_SU3_diff(new_spin, old_spin, site);

        lattice.spins_SU3[site] = new_spin;
        const double energy_after = lattice.total_energy();
        lattice.spins_SU3[site] = old_spin;

        const double exact_delta = energy_after - energy_before;
        if (!nearly_equal(delta_energy, exact_delta, kAbsTol, kRelTol)) {
            out << "[FAIL] SU3 delta-energy mismatch at site " << site
                << ": dE_site=" << delta_energy
                << ", dE_exact=" << exact_delta << "\n";
            return false;
        }
    }

    return true;
}

bool check_llg_consistency(Lattice& lattice, std::ostream& out) {
    lattice.reset_pulse();

    const auto state = lattice.spins_to_state();
    Lattice::ODEState dxdt(state.size(), 0.0);
    lattice.ode_system(state, dxdt, 0.0);

    double energy_rate = 0.0;

    for (size_t site = 0; site < lattice.lattice_size_SU2; ++site) {
        const auto field = lattice.get_local_field_SU2(site);
        const auto expected = su2_precession(field, lattice.spins_SU2[site]);

        Eigen::VectorXd actual(lattice.spin_dim_SU2);
        const size_t idx = site * lattice.spin_dim_SU2;
        for (size_t d = 0; d < lattice.spin_dim_SU2; ++d) {
            actual(static_cast<Eigen::Index>(d)) = dxdt[idx + d];
        }

        if (!vector_nearly_equal(expected, actual, kAbsTol, kRelTol)) {
            out << "[FAIL] SU2 LLG derivative mismatch at site " << site << "\n";
            return false;
        }

        const double norm_rate = lattice.spins_SU2[site].dot(actual);
        if (!nearly_equal(norm_rate, 0.0, kAbsTol, kRelTol)) {
            out << "[FAIL] SU2 norm is not preserved at site " << site
                << ": d|S|^2/dt=" << 2.0 * norm_rate << "\n";
            return false;
        }

        energy_rate -= field.dot(actual);
    }

    const size_t offset_su3 = lattice.lattice_size_SU2 * lattice.spin_dim_SU2;
    for (size_t site = 0; site < lattice.lattice_size_SU3; ++site) {
        const auto field = lattice.get_local_field_SU3(site);
        const auto expected = su3_precession(field, lattice.spins_SU3[site]);

        Eigen::VectorXd actual(lattice.spin_dim_SU3);
        const size_t idx = offset_su3 + site * lattice.spin_dim_SU3;
        for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
            actual(static_cast<Eigen::Index>(d)) = dxdt[idx + d];
        }

        if (!vector_nearly_equal(expected, actual, kAbsTol, kRelTol)) {
            out << "[FAIL] SU3 LLG derivative mismatch at site " << site << "\n";
            return false;
        }

        const double norm_rate = lattice.spins_SU3[site].dot(actual);
        if (!nearly_equal(norm_rate, 0.0, kAbsTol, kRelTol)) {
            out << "[FAIL] SU3 norm is not preserved at site " << site
                << ": d|S|^2/dt=" << 2.0 * norm_rate << "\n";
            return false;
        }

        energy_rate -= field.dot(actual);
    }

    if (!nearly_equal(energy_rate, 0.0, 5e-11, 5e-10)) {
        out << "[FAIL] LLG instantaneous energy drift is nonzero: dE/dt="
            << energy_rate << "\n";
        return false;
    }

    return true;
}

}  // namespace

int main() {
    Lattice lattice = make_lattice();
    assign_deterministic_spins(lattice);

    if (!check_w_only_builder_surface(std::cout)) {
        return 1;
    }
    if (!check_shorthand_equivalence(std::cout)) {
        return 1;
    }
    if (!check_mirror_odd_inversion_pairing(std::cout)) {
        return 1;
    }
    if (!check_state_roundtrip(lattice, std::cout)) {
        return 1;
    }
    if (!check_total_energy_flat_consistency(lattice, std::cout)) {
        return 1;
    }
    if (!check_su2_delta_energy(lattice, std::cout)) {
        return 1;
    }
    if (!check_su3_delta_energy(lattice, std::cout)) {
        return 1;
    }
    if (!check_llg_consistency(lattice, std::cout)) {
        return 1;
    }

    std::cout << "[PASS] On-site W builder and LLG regression checks\n";
    return 0;
}