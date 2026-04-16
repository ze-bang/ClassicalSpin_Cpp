#include "classical_spin/lattice/strain_phonon_lattice.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/core/spin_config.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

using Lattice = StrainPhononLattice;
using SpinVector = Eigen::Vector3d;

constexpr double kFiniteDifferenceStep = 1e-7;
constexpr double kFieldAbsTol = 5e-6;
constexpr double kFieldRelTol = 5e-5;
constexpr double kExactAbsTol = 1e-12;
constexpr double kExactRelTol = 1e-10;

bool nearly_equal(double lhs, double rhs, double abs_tol, double rel_tol) {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= abs_tol + rel_tol * scale;
}

double vector_max_abs_diff(const SpinVector& lhs, const SpinVector& rhs) {
    return (lhs - rhs).cwiseAbs().maxCoeff();
}

SpinVector normalized_spin(double x, double y, double z, double spin_length) {
    SpinVector spin(x, y, z);
    const double norm = spin.norm();
    if (norm == 0.0) {
        return SpinVector(spin_length, 0.0, 0.0);
    }
    return spin * (spin_length / norm);
}

void assign_deterministic_spins(Lattice& lattice, double offset) {
    for (size_t site = 0; site < lattice.lattice_size; ++site) {
        const double x = std::sin(0.37 * (site + 1) + offset);
        const double y = std::cos(0.61 * (site + 1) - 0.5 * offset);
        const double z = std::sin(0.23 * (site + 1) + 1.7 * offset);
        lattice.spins[site] = normalized_spin(x, y, z, lattice.spin_length);
    }
}

Lattice make_lattice(size_t linear_size,
                     float spin_length,
                     const MagnetoelasticParams& me,
                     const ElasticParams& el,
                     const StrainDriveParams& dr) {
    SpinConfig config;
    config.set_param("J", me.J);
    config.set_param("K", me.K);
    config.set_param("Gamma", me.Gamma);
    config.set_param("Gammap", me.Gammap);
    config.set_param("J2_A", me.J2_A);
    config.set_param("J2_B", me.J2_B);
    config.set_param("J3", me.J3);
    config.field_strength = 0.0;

    UnitCell unit_cell = build_strain_honeycomb(config);
    Lattice lattice(unit_cell, linear_size, linear_size, 1, spin_length);
    lattice.set_parameters(me, el, dr);
    return lattice;
}

bool run_ring_exchange_regression(std::ostream& out) {
    MagnetoelasticParams me;
    me.J = 0.0;
    me.K = 0.0;
    me.Gamma = 0.0;
    me.Gammap = 0.0;
    me.J2_A = 0.0;
    me.J2_B = 0.0;
    me.J3 = 0.0;
    me.J7 = -0.37;
    me.lambda_A1g = 0.0;
    me.lambda_Eg = 0.0;
    me.gamma_J7 = 0.0;

    ElasticParams el;
    el.C11 = 1.0;
    el.C12 = 0.3;
    el.C44 = 0.5;
    el.M = 1.0;
    el.gamma_A1g = 0.0;
    el.gamma_Eg = 0.0;

    StrainDriveParams dr;
    dr.E0_1 = 0.0;
    dr.E0_2 = 0.0;

    Lattice lattice = make_lattice(2, 1.0f, me, el, dr);
    lattice.set_strain_Eg(0.0, 0.0);
    assign_deterministic_spins(lattice, 0.31);

    double max_static_dynamic_diff = 0.0;
    double max_local_ring_diff = 0.0;
    double max_fd_diff = 0.0;

    for (size_t site = 0; site < lattice.lattice_size; ++site) {
        const SpinVector ring_static = lattice.get_ring_exchange_field(site);
        const SpinVector ring_dynamic = lattice.get_ring_exchange_field(site, 0.0);
        const SpinVector local_static = lattice.get_local_field(site);
        const SpinVector local_dynamic = lattice.get_local_field(site, 0.0);

        max_static_dynamic_diff = std::max(max_static_dynamic_diff,
                                           vector_max_abs_diff(ring_static, ring_dynamic));
        max_local_ring_diff = std::max(max_local_ring_diff,
                                       vector_max_abs_diff(local_static, ring_static));
        max_static_dynamic_diff = std::max(max_static_dynamic_diff,
                                           vector_max_abs_diff(local_static, local_dynamic));

        if (vector_max_abs_diff(ring_static, ring_dynamic) > kFieldAbsTol ||
            vector_max_abs_diff(local_static, local_dynamic) > kFieldAbsTol) {
            out << "[FAIL] Static and time-dependent field paths diverged at site " << site
                << "\n";
            return false;
        }

        if (vector_max_abs_diff(local_static, ring_static) > kFieldAbsTol) {
            out << "[FAIL] Local field does not reduce to pure ring field at site " << site
                << " when only J7 is nonzero\n";
            return false;
        }

        const SpinVector original_spin = lattice.spins[site];
        for (int component = 0; component < 3; ++component) {
            lattice.spins[site] = original_spin;
            lattice.spins[site](component) += kFiniteDifferenceStep;
            const double energy_plus = lattice.ring_exchange_energy();

            lattice.spins[site] = original_spin;
            lattice.spins[site](component) -= kFiniteDifferenceStep;
            const double energy_minus = lattice.ring_exchange_energy();

            lattice.spins[site] = original_spin;

            const double finite_difference = -(energy_plus - energy_minus) / (2.0 * kFiniteDifferenceStep);
            const double field_component = ring_static(component);
            max_fd_diff = std::max(max_fd_diff, std::abs(finite_difference - field_component));

            if (!nearly_equal(finite_difference, field_component, kFieldAbsTol, kFieldRelTol)) {
                out << "[FAIL] Ring-exchange finite-difference mismatch at site " << site
                    << ", component " << component
                    << ": field=" << field_component
                    << ", fd=" << finite_difference << "\n";
                return false;
            }
        }
    }

    out << "[PASS] Ring-exchange field regression"
        << " (max static/dynamic diff=" << max_static_dynamic_diff
        << ", max local/ring diff=" << max_local_ring_diff
        << ", max finite-difference diff=" << max_fd_diff << ")\n";
    return true;
}

bool run_ode_state_regression(std::ostream& out) {
    MagnetoelasticParams me;
    me.J = 0.42;
    me.K = -1.10;
    me.Gamma = 0.33;
    me.Gammap = -0.18;
    me.J2_A = 0.0;
    me.J2_B = 0.0;
    me.J3 = 0.0;
    me.J7 = -0.12;
    me.lambda_A1g = 0.03;
    me.lambda_Eg = 0.08;
    me.gamma_J7 = 0.15;

    ElasticParams el;
    el.C11 = 1.2;
    el.C12 = 0.35;
    el.C44 = 0.7;
    el.M = 1.0;
    el.gamma_A1g = 0.05;
    el.gamma_Eg = 0.04;

    StrainDriveParams dr;
    dr.E0_1 = 0.0;
    dr.E0_2 = 0.0;

    Lattice lattice_a = make_lattice(2, 1.0f, me, el, dr);
    Lattice lattice_b = make_lattice(2, 1.0f, me, el, dr);

    assign_deterministic_spins(lattice_a, 0.11);
    assign_deterministic_spins(lattice_b, 2.37);

    lattice_a.set_strain_Eg(0.04, -0.015);
    lattice_b.strain = lattice_a.strain;
    for (size_t bond = 0; bond < StrainState::N_BONDS; ++bond) {
        lattice_a.strain.V_xx[bond] = 0.01 * static_cast<double>(bond + 1);
        lattice_a.strain.V_yy[bond] = -0.02 * static_cast<double>(bond + 1);
        lattice_a.strain.V_xy[bond] = 0.03 * static_cast<double>(bond + 1);
    }
    lattice_b.strain = lattice_a.strain;

    Lattice::SpinConfig state_spins(lattice_a.lattice_size);
    for (size_t site = 0; site < lattice_a.lattice_size; ++site) {
        const double x = std::sin(0.41 * (site + 1) + 1.1);
        const double y = std::cos(0.52 * (site + 1) - 0.8);
        const double z = std::sin(0.29 * (site + 1) + 0.6);
        state_spins[site] = normalized_spin(x, y, z, lattice_a.spin_length);
    }

    const size_t spin_offset = Lattice::spin_dim * lattice_a.lattice_size;
    Lattice::ODEState x(spin_offset + StrainState::N_DOF, 0.0);
    for (size_t site = 0; site < lattice_a.lattice_size; ++site) {
        const size_t idx = site * Lattice::spin_dim;
        x[idx] = state_spins[site](0);
        x[idx + 1] = state_spins[site](1);
        x[idx + 2] = state_spins[site](2);
    }
    lattice_a.strain.to_array(&x[spin_offset]);

    Lattice::ODEState dxdt_a(x.size(), 0.0);
    Lattice::ODEState dxdt_b(x.size(), 0.0);
    lattice_a.ode_system(x, dxdt_a, 0.37);
    lattice_b.ode_system(x, dxdt_b, 0.37);

    double max_diff = 0.0;
    size_t max_index = 0;
    for (size_t idx = 0; idx < dxdt_a.size(); ++idx) {
        const double diff = std::abs(dxdt_a[idx] - dxdt_b[idx]);
        if (diff > max_diff) {
            max_diff = diff;
            max_index = idx;
        }
        if (!nearly_equal(dxdt_a[idx], dxdt_b[idx], kExactAbsTol, kExactRelTol)) {
            out << "[FAIL] ode_system depends on pre-existing member spins at state index "
                << idx << ": dx_a=" << dxdt_a[idx] << ", dx_b=" << dxdt_b[idx] << "\n";
            return false;
        }
    }

    out << "[PASS] ODE state-ordering regression"
        << " (max identical-state diff=" << max_diff << " at index " << max_index << ")\n";
    return true;
}

bool run_local_global_equivalence(std::ostream& out) {
    // Test: in the stiff gradient limit (large K_gradient), the local strain
    // relaxation should converge to uniform strain that matches the global mode.
    //
    // For uniform strain, H_gradient = 0 (all neighbors identical), so
    // the equilibrium reduces to: C11 ε_xx + C12 ε_yy = -λ <Σ_Eg1>
    // which is exactly the global mode solution.
    
    out << "\n--- Local-vs-global strain equivalence test ---\n";
    
    MagnetoelasticParams me;
    me.J = -1.0;
    me.K = -6.0;
    me.Gamma = 8.0;
    me.Gammap = -3.5;
    me.J2_A = 0.0;
    me.J2_B = 0.0;
    me.J3 = 0.0;
    me.J7 = -0.1;
    me.lambda_A1g = 0.0;
    me.lambda_Eg = 0.05;
    me.gamma_J7 = 0.0;
    
    ElasticParams el;
    el.C11 = 1.0;
    el.C12 = 0.3;
    el.C44 = 0.35;
    el.M = 1.0;
    el.gamma_A1g = 0.1;
    el.gamma_Eg = 0.1;
    
    StrainDriveParams dr;
    dr.E0_1 = 0.0;
    dr.E0_2 = 0.0;
    
    const size_t L = 6;  // 6×6 lattice = 72 spins, 36 cells
    
    // ---- Global mode ----
    Lattice lattice_global = make_lattice(L, 1.0f, me, el, dr);
    assign_deterministic_spins(lattice_global, 0.42);
    lattice_global.relax_strain(false);
    
    double global_eps_xx = lattice_global.strain.epsilon_xx[0];
    double global_eps_yy = lattice_global.strain.epsilon_yy[0];
    double global_eps_xy = lattice_global.strain.epsilon_xy[0];
    double global_Eg1 = lattice_global.Eg1_amplitude();
    double global_Eg2 = lattice_global.Eg2_amplitude();
    double global_energy = lattice_global.total_energy();
    
    out << "  Global:  ε_xx=" << global_eps_xx << " ε_yy=" << global_eps_yy
        << " ε_xy=" << global_eps_xy << "\n";
    out << "  Global:  Eg1=" << global_Eg1 << " Eg2=" << global_Eg2 << "\n";
    out << "  Global:  E_total=" << global_energy << "\n";
    
    // Collect global effective fields for later comparison
    std::vector<SpinVector> global_fields(lattice_global.lattice_size);
    for (size_t i = 0; i < lattice_global.lattice_size; ++i)
        global_fields[i] = lattice_global.get_local_field(i);
    
    // ---- Local mode with large K_gradient ----
    ElasticParams el_local = el;
    el_local.K_gradient = 1e6;  // Very stiff → uniform
    
    Lattice lattice_local = make_lattice(L, 1.0f, me, el_local, dr);
    assign_deterministic_spins(lattice_local, 0.42);  // Same spins
    lattice_local.init_local_strain();
    lattice_local.relax_strain(false);
    
    // Compute mean strain across all cells
    double mean_xx = 0, mean_yy = 0, mean_xy = 0;
    double std_xx = 0, std_yy = 0, std_xy = 0;
    size_t N_cells = lattice_local.get_N_cells();
    
    // Access cell_strains_ through save_local_strain_map → parse output? 
    // Better: use Eg1_amplitude/Eg2_amplitude which return means in local mode
    double local_Eg1 = lattice_local.Eg1_amplitude();
    double local_Eg2 = lattice_local.Eg2_amplitude();
    double local_energy = lattice_local.total_energy();
    
    out << "  Local:   Eg1=" << local_Eg1 << " Eg2=" << local_Eg2 << "\n";
    out << "  Local:   E_total=" << local_energy << "\n";
    
    // Compare Eg amplitudes
    constexpr double strain_tol = 1e-5;
    constexpr double energy_rel_tol = 1e-4;
    
    if (!nearly_equal(global_Eg1, local_Eg1, strain_tol, strain_tol)) {
        out << "[FAIL] Eg1 mismatch: global=" << global_Eg1 << " local=" << local_Eg1
            << " diff=" << std::abs(global_Eg1 - local_Eg1) << "\n";
        return false;
    }
    
    if (!nearly_equal(global_Eg2, local_Eg2, strain_tol, strain_tol)) {
        out << "[FAIL] Eg2 mismatch: global=" << global_Eg2 << " local=" << local_Eg2
            << " diff=" << std::abs(global_Eg2 - local_Eg2) << "\n";
        return false;
    }
    
    // Compare effective fields on spins (these include ME contribution)
    double max_field_diff = 0.0;
    size_t max_field_site = 0;
    for (size_t i = 0; i < lattice_local.lattice_size; ++i) {
        SpinVector local_field = lattice_local.get_local_field(i);
        double diff = vector_max_abs_diff(global_fields[i], local_field);
        if (diff > max_field_diff) {
            max_field_diff = diff;
            max_field_site = i;
        }
    }
    
    out << "  Max field diff=" << max_field_diff << " at site " << max_field_site << "\n";
    
    // Tolerance: B-sublattice bonds span multiple cells with O(1/K_gradient)
    // strain variations, so perfect matching requires K_gradient → ∞.
    constexpr double stiff_field_tol = 5e-4;
    if (max_field_diff > stiff_field_tol) {
        out << "[FAIL] Effective field mismatch in stiff limit: max_diff=" << max_field_diff
            << " at site " << max_field_site << "\n";
        // Print details for debugging
        SpinVector gf = global_fields[max_field_site];
        SpinVector lf = lattice_local.get_local_field(max_field_site);
        out << "  Global field: " << gf.transpose() << "\n";
        out << "  Local  field: " << lf.transpose() << "\n";
        return false;
    }
    
    // Compare total energies (ME + elastic + spin should match)
    double energy_diff = std::abs(global_energy - local_energy);
    double energy_scale = std::max(std::abs(global_energy), 1.0);
    out << "  Energy diff=" << energy_diff << " (rel=" << energy_diff / energy_scale << ")\n";
    
    if (energy_diff / energy_scale > energy_rel_tol) {
        out << "[FAIL] Total energy mismatch: global=" << global_energy
            << " local=" << local_energy << "\n";
        return false;
    }
    
    // ---- Also test with an ordered spin state (nearer to real use case) ----
    // Initialize lattice to Néel state along z
    out << "  Testing with Néel z state...\n";
    for (size_t i = 0; i < lattice_global.lattice_size; ++i) {
        int atom = static_cast<int>(i) % 2;
        double z = (atom == 0) ? 1.0 : -1.0;
        lattice_global.spins[i] = SpinVector(0, 0, z);
        lattice_local.spins[i] = SpinVector(0, 0, z);
    }
    
    lattice_global.relax_strain(false);
    lattice_local.relax_strain(false);
    
    double neel_global_Eg1 = lattice_global.Eg1_amplitude();
    double neel_local_Eg1 = lattice_local.Eg1_amplitude();
    double neel_global_Eg2 = lattice_global.Eg2_amplitude();
    double neel_local_Eg2 = lattice_local.Eg2_amplitude();
    
    out << "  Néel: global Eg1=" << neel_global_Eg1 << " local Eg1=" << neel_local_Eg1 << "\n";
    out << "  Néel: global Eg2=" << neel_global_Eg2 << " local Eg2=" << neel_local_Eg2 << "\n";
    
    if (!nearly_equal(neel_global_Eg1, neel_local_Eg1, strain_tol, strain_tol)) {
        out << "[FAIL] Néel Eg1 mismatch: " << neel_global_Eg1 << " vs " << neel_local_Eg1 << "\n";
        return false;
    }
    if (!nearly_equal(neel_global_Eg2, neel_local_Eg2, strain_tol, strain_tol)) {
        out << "[FAIL] Néel Eg2 mismatch: " << neel_global_Eg2 << " vs " << neel_local_Eg2 << "\n";
        return false;
    }
    
    // Check fields again with Néel state
    max_field_diff = 0.0;
    for (size_t i = 0; i < lattice_local.lattice_size; ++i) {
        SpinVector gf = lattice_global.get_local_field(i);
        SpinVector lf = lattice_local.get_local_field(i);
        double diff = vector_max_abs_diff(gf, lf);
        max_field_diff = std::max(max_field_diff, diff);
    }
    
    out << "  Néel max field diff=" << max_field_diff << "\n";
    if (max_field_diff > stiff_field_tol) {
        out << "[FAIL] Néel field mismatch: " << max_field_diff << "\n";
        return false;
    }
    
    out << "[PASS] Local-global strain equivalence (stiff K_gradient limit)\n";
    return true;
}

}  // namespace

int main() {
    std::cout << std::scientific << std::setprecision(8);

    bool ok = true;
    ok = run_ring_exchange_regression(std::cout) && ok;
    ok = run_ode_state_regression(std::cout) && ok;
    ok = run_local_global_equivalence(std::cout) && ok;

    if (!ok) {
        std::cout << "Regression failures detected.\n";
        return 1;
    }

    std::cout << "All strain-phonon regressions passed.\n";
    return 0;
}