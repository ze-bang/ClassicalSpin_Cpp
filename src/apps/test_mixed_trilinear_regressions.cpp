#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <algorithm>
#include <cmath>
#include <iostream>
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

Lattice make_lattice() {
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

    // On-site W (a∈{1,3,4,6,8}). u_n→W{n}_zz=+u_n, W{n}_xx=−u_n; v_n→W{n}_xz=+v_n.
    config.set_param("W1_zz",  0.31);  config.set_param("W1_xx", -0.31);   // u1 = 0.31
    config.set_param("W3_zz", -0.22);  config.set_param("W3_xx",  0.22);   // u3 = -0.22
    config.set_param("W8_zz",  0.17);  config.set_param("W8_xx", -0.17);   // u8 = 0.17
    config.set_param("W4_xz",  0.29);                                       // v4 = 0.29
    config.set_param("W6_xz", -0.11);                                       // v6 = -0.11
    // Inter-site V. w_n→V{n}_zz=+w_n, V{n}_xx=−w_n (A1+); V{n}_xz=+w_n (A2+).
    config.set_param("V1_zz",  0.13);  config.set_param("V1_xx", -0.13);   // w1 = 0.13
    config.set_param("V3_zz",  0.07);  config.set_param("V3_xx", -0.07);   // w3 = 0.07
    config.set_param("V8_zz", -0.19);  config.set_param("V8_xx",  0.19);   // w8 = -0.19
    config.set_param("V4_xz",  0.23);                                       // w4 = 0.23
    config.set_param("V6_xz",  0.05);                                       // w6 = 0.05
    config.spin_length = 1.0f;
    config.spin_length_su3 = 1.0f;

    MixedUnitCell mixed_uc = build_tmfeo3(config);
    return Lattice(mixed_uc, 1, 1, 1, config.spin_length, config.spin_length_su3);
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

    std::cout << "[PASS] Mixed trilinear and LLG regression checks\n";
    return 0;
}