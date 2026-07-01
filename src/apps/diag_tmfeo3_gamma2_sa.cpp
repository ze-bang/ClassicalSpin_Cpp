// diag_tmfeo3_gamma2_sa.cpp
// ---------------------------------------------------------------------------
// Two-stage ground-state finder for TmFeO3 (full mixed Fe+Tm, 1x1x1 cell).
//
// PROBLEM
// -------
// Physical Ka=-0.0153, Kc=-0.0187 (calibrated to qFM=0.38 THz, qAFM=0.91 THz)
// give |Ka-Kc|=0.0034 meV.  This is the SAME stiffness as the qFM gap — it sets
// BOTH the magnon frequency AND the Gamma_2/Gamma_4 barrier.  So the xz plane is
// genuinely soft (by design), and random-start SA only reaches Gamma_2 ~40% of
// the time.  You cannot firm Gamma_2 by changing Ka/Kc without detuning the
// magnon.
//
// SOLUTION: two-stage protocol
// ----------------------------
// Stage 1  (BIASED SA, Ka=0 Kc=-0.029)
//   Anneal from random initial state with artificially strong z-easy axis.
//   This gives 100% Gamma_2 convergence.  The biased Hamiltonian is ONLY used
//   for steering; it is discarded after SA.
//
// Stage 2  (PHYSICAL relax, Ka=-0.0153 Kc=-0.0187)
//   Transfer the biased Gamma_2 spin state into a lattice built with the true
//   physical parameters.  Run greedy_quench(tol=1e-14, max=500000) to reach
//   the stationary Gamma_2 minimum consistent with qFM=0.38 THz / qAFM=0.91 THz.
//   The resulting seed has pre-pump Gx std ~1e-8 (suitable for pump-probe).
//
// Output seed conventions (Pbnm a=x, b=y, c=z):
//   Gamma_2 = F_x (uniform weak FM along a) + G_z (G-type AFM along c)
//   Fe0:(+Fx,0,+Gz), Fe1:(+Fx,0,-Gz), Fe2:(+Fx,0,+Gz), Fe3:(+Fx,0,-Gz)
// ---------------------------------------------------------------------------

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Lattice = MixedLattice;

namespace {

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// Base exchange + CEF shared by both stages.
SpinConfig make_base_exchange_config() {
    SpinConfig config;
    config.spin_length     = 2.5f;
    config.spin_length_su3 = 1.0f;
    config.field_strength  = 0.0;

    config.set_param("J1ab", 4.74);   // Fe-Fe exchange (meV)
    config.set_param("J1c",  5.15);
    config.set_param("J2ab", 0.15);
    config.set_param("J2c",  0.30);
    config.set_param("D1",   0.049);   // DM interaction
    config.set_param("D2",   0.0);
    config.set_param("e1",   2.067834); // Tm CEF levels (meV)
    config.set_param("e2",   4.9628);
    return config;
}

// Stage 1: biased anisotropy for reliable Gamma_2 SA steering.
// Ka=0, Kc=-0.029 → z strict easy axis → 100% Gamma_2 from random starts.
// NOT the physical Hamiltonian; only used for steering the SA.
SpinConfig make_biased_config() {
    SpinConfig config = make_base_exchange_config();
    config.set_param("Ka", 0.0);
    config.set_param("Kb", 0.0);
    config.set_param("Kc", -0.029);
    return config;
}

// Stage 2: physical parameters calibrated to qFM=0.38 THz, qAFM=0.91 THz.
//   |Ka-Kc| = 0.0034 meV  →  q_FM² = 42.85·|Ka-Kc|  →  qFM = 0.381 THz
//   |Kc|    = 0.0187 meV  →  qAFM ≈ 0.901 THz  (Kc sets out-of-plane stiffness)
SpinConfig make_physical_config() {
    SpinConfig config = make_base_exchange_config();
    config.set_param("Ka", -0.0153);
    config.set_param("Kb",  0.0);
    config.set_param("Kc", -0.0187);
    return config;
}

// ---------------------------------------------------------------------------
// Bertaut vectors and Gamma_2 check
// ---------------------------------------------------------------------------

struct BertautVectors {
    Eigen::Vector3d F, G, A, C;
};

// Valid only for a 1x1x1 cell where site index == sublattice index.
BertautVectors compute_bertaut(const Lattice& lattice) {
    const Eigen::VectorXd& S0 = lattice.spins_SU2[0];
    const Eigen::VectorXd& S1 = lattice.spins_SU2[1];
    const Eigen::VectorXd& S2 = lattice.spins_SU2[2];
    const Eigen::VectorXd& S3 = lattice.spins_SU2[3];

    BertautVectors bv;
    bv.F = (S0 + S1 + S2 + S3) * 0.25;
    bv.G = (S0 - S1 + S2 - S3) * 0.25;
    bv.A = (S0 + S1 - S2 - S3) * 0.25;
    bv.C = (S0 - S1 - S2 + S3) * 0.25;
    return bv;
}

bool is_gamma2(const BertautVectors& bv) {
    // G_z must dominate (>2.0 out of spin_length=2.5), A and C must be noise.
    return std::abs(bv.G(2)) > 2.0
        && bv.A.norm() < 0.2
        && bv.C.norm() < 0.2;
}

void print_vec3(const char* name, const Eigen::Vector3d& v) {
    std::cout << "    " << name << " = ("
              << std::setw(11) << std::setprecision(6) << std::fixed << v(0) << ", "
              << std::setw(11) << v(1) << ", "
              << std::setw(11) << v(2) << ")   |.|="
              << std::setw(9) << v.norm() << "\n";
}

void print_trial_detail(const Lattice& lattice, const BertautVectors& bv) {
    std::cout << "  per-sublattice Fe magnetisations:\n";
    for (int s = 0; s < 4; ++s) {
        const Eigen::VectorXd& sp = lattice.spins_SU2[s];
        std::cout << "    M" << s << " = ("
                  << std::setw(11) << std::setprecision(6) << std::fixed << sp(0) << ", "
                  << std::setw(11) << sp(1) << ", "
                  << std::setw(11) << sp(2) << ")  |.|="
                  << std::setw(9) << sp.norm() << "\n";
    }
    std::cout << "  Bertaut vectors (lab Pbnm a/b/c frame):\n";
    print_vec3("F", bv.F);
    print_vec3("G", bv.G);
    print_vec3("A", bv.A);
    print_vec3("C", bv.C);
    std::cout << "  Gamma_2: |F_x|=" << std::abs(bv.F(0))
              << "  |C_y|=" << std::abs(bv.C(1))
              << "  |G_z|=" << std::abs(bv.G(2)) << "\n\n";
}

// ---------------------------------------------------------------------------
// Seed I/O
// ---------------------------------------------------------------------------

using SpinArray = std::vector<Eigen::VectorXd>;

bool save_su2_seed(const std::string& path, const SpinArray& spins) {
    std::ofstream out(path);
    if (!out) { std::cerr << "ERROR: cannot write " << path << "\n"; return false; }
    out << std::setprecision(17);
    for (const auto& s : spins) {
        for (int d = 0; d < s.size(); ++d) {
            if (d > 0) out << ' ';
            out << s(d);
        }
        out << '\n';
    }
    return true;
}

bool save_su3_seed(const std::string& path, const SpinArray& spins) {
    std::ofstream out(path);
    if (!out) { std::cerr << "ERROR: cannot write " << path << "\n"; return false; }
    out << std::setprecision(17);
    for (const auto& s : spins) {
        for (int d = 0; d < s.size(); ++d) {
            if (d > 0) out << ' ';
            out << s(d);
        }
        out << '\n';
    }
    return true;
}

void write_summary(const std::string& path, double E_per_site,
                   const BertautVectors& bv, const SpinArray& su2,
                   int n_gamma2, int n_trials) {
    std::ofstream out(path);
    if (!out) { std::cerr << "ERROR: cannot write " << path << "\n"; return; }
    out << std::setprecision(10);
    out << "# TmFeO3 Gamma_2 ground state (1x1x1 mixed lattice)\n";
    out << "# Two-stage protocol: biased SA (Ka=0 Kc=-0.029) + physical relax\n";
    out << "# Physical params: Ka=-0.0153 Kc=-0.0187 D1=0.049 (qFM=0.38 THz, qAFM=0.91 THz)\n";
    out << "# SA convergence: " << n_gamma2 << "/" << n_trials << " biased trials reached Gamma_2\n";
    out << "#\n";
    out << "E_per_site_meV = " << E_per_site << "\n";
    out << "#\n";
    out << "# Fe sublattice magnetisations (lab Cartesian, physical Hamiltonian)\n";
    for (int s = 0; s < (int)su2.size(); ++s) {
        out << "M" << s << " = " << su2[s](0) << " " << su2[s](1) << " " << su2[s](2) << "\n";
    }
    out << "#\n";
    out << "# Bertaut vectors (Pbnm a/b/c)\n";
    out << "F = " << bv.F(0) << " " << bv.F(1) << " " << bv.F(2)
        << "   |F|=" << bv.F.norm() << "\n";
    out << "G = " << bv.G(0) << " " << bv.G(1) << " " << bv.G(2)
        << "   |G|=" << bv.G.norm() << "\n";
    out << "A = " << bv.A(0) << " " << bv.A(1) << " " << bv.A(2)
        << "   |A|=" << bv.A.norm() << "\n";
    out << "C = " << bv.C(0) << " " << bv.C(1) << " " << bv.C(2)
        << "   |C|=" << bv.C.norm() << "\n";
    out << "#\n";
    out << "# Gamma_2 magnitudes\n";
    out << "|F_x| = " << std::abs(bv.F(0)) << "\n";
    out << "|C_y| = " << std::abs(bv.C(1)) << "\n";
    out << "|G_z| = " << std::abs(bv.G(2)) << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    // Output directory: defaults to ../tfo_project (relative to build/).
    // Override by passing a path as argv[1].
    const std::string out_dir = (argc > 1) ? argv[1] : "../tfo_project";

    constexpr int    N_TRIALS        = 10;
    constexpr double T_START         = 15.0;   // K
    constexpr double T_END           = 1e-3;   // K
    constexpr double COOLING_RATE    = 0.9;
    constexpr size_t N_ANNEAL        = 2000;   // MC sweeps per temperature step
    constexpr size_t N_DETERMINISTIC = 2000;   // T=0 deterministic sweeps

    constexpr size_t N_PHYS_QUENCH = 200000;  // fixed det sweeps in physical stage
                                               // (2e5 → pre-pump Gx std ~1e-8)

    std::cout << "================================================================\n"
              << " TmFeO3 Gamma_2 two-stage ground-state finder (1x1x1 mixed)\n"
              << " Stage 1 BIAS  : Ka=0  Kc=-0.029  (z easy axis, 100% Gamma_2)\n"
              << "   SA    : T_start=" << T_START << "K → T_end=" << T_END << "K"
              << "  rate=" << COOLING_RATE << "  n_sweep=" << N_ANNEAL << "\n"
              << "   T=0   : " << N_DETERMINISTIC << " det sweeps\n"
              << " Stage 2 PHYS  : Ka=-0.0153  Kc=-0.0187  (qFM=0.38 THz, qAFM=0.91 THz)\n"
              << "   quench: " << N_PHYS_QUENCH << " fixed deterministic sweeps\n"
              << " Trials : " << N_TRIALS << "\n"
              << "================================================================\n\n";

    const SpinConfig biased_config   = make_biased_config();
    const SpinConfig physical_config = make_physical_config();
    int gamma2_count = 0;

    // Track best (lowest-energy Gamma_2) state across trials.
    double best_energy = std::numeric_limits<double>::max();
    SpinArray best_su2, best_su3;
    BertautVectors best_bv;

    for (int trial = 0; trial < N_TRIALS; ++trial) {
        std::cout << "--------------------------------------------------------\n"
                  << " Trial " << (trial + 1) << " / " << N_TRIALS << "\n"
                  << "--------------------------------------------------------\n";

        // ------ Stage 1: biased SA ----------------------------------------
        Lattice lattice(build_tmfeo3(biased_config), 1, 1, 1,
                        biased_config.spin_length, biased_config.spin_length_su3);
        lattice.init_random();

        const double E_init = lattice.energy_density();
        std::cout << "  [bias] initial E/site = " << std::scientific
                  << std::setprecision(6) << E_init << "\n";

        lattice.simulated_annealing(
            T_START, T_END, N_ANNEAL,
            /*gaussian_move=*/false, COOLING_RATE,
            /*out_dir=*/"", /*save_observables=*/false,
            /*T_zero=*/true, N_DETERMINISTIC, /*twist=*/0);

        lattice.physicalize_SU3_state();

        const BertautVectors bv_biased = compute_bertaut(lattice);
        const bool ok_biased = is_gamma2(bv_biased);
        if (ok_biased) ++gamma2_count;

        std::cout << "  [bias] final  E/site = " << std::scientific
                  << lattice.energy_density()
                  << "  " << (ok_biased ? "[Gamma_2 OK]" : "[NOT Gamma_2 - WARN]") << "\n";

        if (!ok_biased) {
            std::cout << "  Skipping physical stage for this trial.\n\n";
            continue;
        }

        // ------ Stage 2: physical relax ------------------------------------
        std::cout << "  [phys] rebuilding with Ka=-0.0153, Kc=-0.0187...\n";
        Lattice lattice_phys(build_tmfeo3(physical_config), 1, 1, 1,
                             physical_config.spin_length, physical_config.spin_length_su3);

        // Transfer biased Gamma_2 spin state into the physical lattice.
        for (size_t s = 0; s < lattice_phys.lattice_size_SU2; ++s)
            lattice_phys.spins_SU2[s] = lattice.spins_SU2[s];
        for (size_t s = 0; s < lattice_phys.lattice_size_SU3; ++s)
            lattice_phys.spins_SU3[s] = lattice.spins_SU3[s];

        // Hard fixed-count deterministic quench: 2e5 sweeps give pre-pump
        // Gx std ~1e-8 (required for clean pump-probe seeds, per calibration).
        // greedy_quench is NOT used here because its energy-convergence criterion
        // fires at floating-point noise (~hundreds of sweeps) before the soft qFM
        // mode (|Ka-Kc|=0.0034 meV) has settled spatially.
        for (size_t sw = 0; sw < N_PHYS_QUENCH; ++sw)
            lattice_phys.deterministic_sweep();
        lattice_phys.physicalize_SU3_state();

        const double E_final = lattice_phys.energy_density();
        const BertautVectors bv = compute_bertaut(lattice_phys);

        std::cout << "  [phys] final  E/site = " << std::scientific << E_final << "\n\n";
        print_trial_detail(lattice_phys, bv);

        // Update best state.
        if (E_final < best_energy) {
            best_energy = E_final;
            best_bv = bv;
            best_su2.clear();
            for (size_t s = 0; s < lattice_phys.lattice_size_SU2; ++s)
                best_su2.push_back(lattice_phys.spins_SU2[s]);
            best_su3.clear();
            for (size_t s = 0; s < lattice_phys.lattice_size_SU3; ++s)
                best_su3.push_back(lattice_phys.spins_SU3[s]);
        }
    }

    std::cout << "================================================================\n"
              << " Result : Gamma_2 success = " << gamma2_count << " / " << N_TRIALS << "\n";
    if (gamma2_count == N_TRIALS) {
        std::cout << " [PASS] All trials converged to Gamma_2\n";
    } else {
        std::cout << " [WARN] " << (N_TRIALS - gamma2_count)
                  << " trial(s) did NOT converge to Gamma_2\n";
    }
    std::cout << "================================================================\n";

    // ------------------------------------------------------------------
    // Save best Gamma_2 state to out_dir.
    // ------------------------------------------------------------------
    if (!best_su2.empty()) {
        std::filesystem::create_directories(out_dir);

        const std::string su2_path  = out_dir + "/tmfeo3_gamma2_1x1x1_seed_SU2.txt";
        const std::string su3_path  = out_dir + "/tmfeo3_gamma2_1x1x1_seed_SU3.txt";
        const std::string summ_path = out_dir + "/tmfeo3_gamma2_1x1x1_gs_summary.txt";

        save_su2_seed(su2_path, best_su2);
        save_su3_seed(su3_path, best_su3);
        write_summary(summ_path, best_energy, best_bv, best_su2,
                      gamma2_count, N_TRIALS);

        std::cout << "\nSaved best Gamma_2 state (E/site=" << std::scientific
                  << best_energy << ") to:\n"
                  << "  " << su2_path  << "\n"
                  << "  " << su3_path  << "\n"
                  << "  " << summ_path << "\n";
    } else {
        std::cout << "\n[WARN] No Gamma_2 state found; nothing saved.\n";
    }

    return (gamma2_count == N_TRIALS) ? 0 : 1;
}
