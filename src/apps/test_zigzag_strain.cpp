/**
 * @file test_zigzag_strain.cpp
 * @brief Test: single-domain zigzag (C3-breaking) → finite equilibrium strain
 *
 * The full NN Hamiltonian is C3-invariant, so triple-Q ground states
 * (which preserve C3) have Σ_Eg = 0 and thus zero equilibrium strain.
 *
 * A single-domain zigzag breaks C3 → Σ_Eg ≠ 0 → finite strain.
 * This test verifies that by initializing each of the 3 zigzag domains
 * and showing that relax_strain produces nonzero ε.
 */

#include "classical_spin/lattice/strain_phonon_lattice.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "==========================================================\n";
    std::cout << "  Test: Zigzag (C3-breaking) → Finite Equilibrium Strain\n";
    std::cout << "==========================================================\n\n";

    // Lattice parameters (same as pump_probe.param)
    const size_t L = 8;
    const float spin_length = 1.0f;

    // Hamiltonian parameters
    MagnetoelasticParams me;
    me.J = -1.0;
    me.K = -6.0;
    me.Gamma = 8.0;
    me.Gammap = -4.0;
    me.J2_A = 0.0;  me.J2_B = 0.0;
    me.J3 = 0.0;    me.J7 = -0.3;
    me.lambda_A1g = 0.0;
    me.lambda_Eg  = 0.1;
    me.gamma_J7   = 0.0;

    ElasticParams el;
    el.C11 = 1.0;  el.C12 = 0.3;  el.C44 = 0.35;
    el.M = 1.0;
    el.gamma_A1g = 0.1;  el.gamma_Eg = 0.1;
    el.lambda_A1g = 0.0;  el.lambda_Eg = 0.0;

    StrainDriveParams dr;
    dr.E0_1 = 0; dr.E0_2 = 0;  // No drive

    // ---- Part 1: Triple-Q (C3-symmetric) → expect zero strain ----
    {
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "  Part 1: Triple-Q state (C3-symmetric)\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

        StrainPhononLattice lattice(L, L, 1, spin_length);
        lattice.set_parameters(me, el, dr);
        lattice.init_triple_q();

        std::cout << "\nSpin energy = " << lattice.spin_energy() << "\n\n";
        lattice.relax_strain(true);

        double eps_Eg1 = lattice.Eg1_amplitude();
        double eps_Eg2 = lattice.Eg2_amplitude();
        double eps_Eg  = std::sqrt(eps_Eg1*eps_Eg1 + eps_Eg2*eps_Eg2);
        std::cout << "\n  |ε_Eg| = " << eps_Eg << "\n";
        std::cout << "  Expected: ~0 (C3-symmetric state)\n\n";
    }

    // ---- Part 2: Three zigzag domains (each breaks C3) → expect finite strain ----
    const char* names[] = {"x-bond zigzag (dir=0)", "y-bond zigzag (dir=1)", "z-bond zigzag (dir=2)"};
    
    for (int dir = 0; dir < 3; ++dir) {
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "  Part 2." << dir << ": " << names[dir] << " (C3-breaking)\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

        StrainPhononLattice lattice(L, L, 1, spin_length);
        lattice.set_parameters(me, el, dr);
        lattice.init_zigzag_pattern(dir);

        std::cout << "\nSpin energy = " << lattice.spin_energy() << "\n\n";
        lattice.relax_strain(true);

        double eps_Eg1 = lattice.Eg1_amplitude();
        double eps_Eg2 = lattice.Eg2_amplitude();
        double eps_Eg  = std::sqrt(eps_Eg1*eps_Eg1 + eps_Eg2*eps_Eg2);
        std::cout << "\n  |ε_Eg| = " << eps_Eg << "\n";
        std::cout << "  Expected: FINITE (C3 broken by single zigzag domain)\n\n";
    }

    // ---- Part 3: SA-relaxed triple-Q (true ground state) → expect zero strain ----
    {
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "  Part 3: SA-annealed ground state (should be C3-symmetric)\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

        StrainPhononLattice lattice(L, L, 1, spin_length);
        lattice.set_parameters(me, el, dr);
        lattice.init_random();

        // Quick anneal + deterministic sweeps
        lattice.anneal(20.0, 1e-3, 50000, 0.9, 5, false, "/tmp/test_zigzag_strain",
                       true, 500);

        std::cout << "\nSpin energy = " << lattice.spin_energy() << "\n\n";
        lattice.relax_strain(true);

        double eps_Eg1 = lattice.Eg1_amplitude();
        double eps_Eg2 = lattice.Eg2_amplitude();
        double eps_Eg  = std::sqrt(eps_Eg1*eps_Eg1 + eps_Eg2*eps_Eg2);
        std::cout << "\n  |ε_Eg| = " << eps_Eg << "\n";
        std::cout << "  Expected: ~0 (SA finds C3-symmetric ground state)\n\n";
    }

    std::cout << "==========================================================\n";
    std::cout << "  Summary:\n";
    std::cout << "  • C3-symmetric states  → Σ_Eg = 0  → zero strain\n";
    std::cout << "  • Single zigzag domain → Σ_Eg ≠ 0  → FINITE strain\n";
    std::cout << "  • This is because H_NN is C3-invariant, so its E_g\n";
    std::cout << "    projection vanishes for C3-symmetric states,\n";
    std::cout << "    but individual K/J/Γ/Γ' pieces are NOT C3-invariant.\n";
    std::cout << "==========================================================\n";

    MPI_Finalize();
    return 0;
}
