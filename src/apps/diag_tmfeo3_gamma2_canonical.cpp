// Standalone diagnostic: TmFeO3 Gamma_2 ground state in the current
// canonical storage convention.
//
// Current build_tmfeo3() storage is fixed:
//   - Fe spins are stored in the lab (Pbnm a/b/c) Cartesian basis.
//   - Tm SU(3) Bloch vectors are stored in the canonical local CEF basis.
//
// The checked-in Gamma_2 seed files were produced before that cleanup, when Fe
// seeds were written in the old local-frame storage. This diagnostic lifts the
// legacy Fe seed into the current storage before polishing, then reports the
// lab-frame Bertaut vectors for hand inspection.

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/core/simple_linear_alg.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::array<std::array<double, 3>, 4> kEtaPbnm = {{
    {1, 1, 1},
    {1, -1, -1},
    {-1, 1, -1},
    {-1, -1, 1},
}};

constexpr std::array<double, 4> kSignF = { 1.0,  1.0,  1.0,  1.0};
constexpr std::array<double, 4> kSignG = { 1.0, -1.0,  1.0, -1.0};
constexpr std::array<double, 4> kSignA = { 1.0,  1.0, -1.0, -1.0};
constexpr std::array<double, 4> kSignC = { 1.0, -1.0, -1.0,  1.0};

struct BertautVectors {
    Eigen::Vector3d F;
    Eigen::Vector3d G;
    Eigen::Vector3d A;
    Eigen::Vector3d C;
};

BertautVectors compute_bertaut(const std::vector<Eigen::VectorXd>& M_sub) {
    BertautVectors b;
    b.F.setZero();
    b.G.setZero();
    b.A.setZero();
    b.C.setZero();
    for (size_t mu = 0; mu < 4; ++mu) {
        for (int a = 0; a < 3; ++a) {
            b.F(a) += kSignF[mu] * M_sub[mu](a);
            b.G(a) += kSignG[mu] * M_sub[mu](a);
            b.A(a) += kSignA[mu] * M_sub[mu](a);
            b.C(a) += kSignC[mu] * M_sub[mu](a);
        }
    }
    b.F *= 0.25;
    b.G *= 0.25;
    b.A *= 0.25;
    b.C *= 0.25;
    return b;
}

SpinConfig make_gamma2_config() {
    SpinConfig config;
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
    config.set_param("chi2x", 0.6);
    config.set_param("chi5x", 0.6);
    config.set_param("chi7x", 0.6);
    config.field_strength = 0.0;
    config.field_direction = {0.0, 1.0, 0.0};
    config.spin_length = 2.5f;
    config.spin_length_su3 = 1.0f;
    return config;
}

void print_vec(std::ostream& out, const std::string& label,
               const Eigen::Vector3d& v) {
    out << "    " << std::left << std::setw(4) << label
        << "= ("
        << std::right << std::setw(12) << std::fixed << std::setprecision(6) << v(0) << ", "
        << std::setw(12) << std::fixed << std::setprecision(6) << v(1) << ", "
        << std::setw(12) << std::fixed << std::setprecision(6) << v(2) << ")"
        << "   |.|=" << std::setw(11) << std::fixed << std::setprecision(6) << v.norm()
        << "\n";
}

void print_per_sublattice(std::ostream& out,
                          const std::vector<Eigen::VectorXd>& M_sub) {
    out << "  per-sublattice (lab-frame) Fe magnetisations:\n";
    for (size_t mu = 0; mu < M_sub.size(); ++mu) {
        std::ostringstream label;
        label << "M" << mu;
        print_vec(out, label.str(), M_sub[mu].head<3>());
    }
}

void print_bertaut(std::ostream& out, const BertautVectors& b) {
    out << "  Bertaut vectors in the lab (Pbnm a/b/c) frame:\n";
    print_vec(out, "F", b.F);
    print_vec(out, "G", b.G);
    print_vec(out, "A", b.A);
    print_vec(out, "C", b.C);
    out << "  Gamma_2 component magnitudes: |F_x|=" << std::abs(b.F(0))
        << "  |C_y|=" << std::abs(b.C(1))
        << "  |G_z|=" << std::abs(b.G(2)) << "\n";
}

void load_gamma2_seed_into_current_storage(MixedLattice& L,
                                           const std::string& seed_base) {
    L.load_spin_config(seed_base);
    for (size_t s = 0; s < L.lattice_size_SU2; ++s) {
        const size_t atom = s % L.N_atoms_SU2;
        for (Eigen::Index a = 0; a < L.spins_SU2[s].size(); ++a) {
            L.spins_SU2[s](a) *= kEtaPbnm[atom][a];
        }
    }
}

void run_seeded(size_t n_polish, std::ostream& out) {
    SpinConfig cfg = make_gamma2_config();
    MixedUnitCell uc = build_tmfeo3(cfg);
    MixedLattice L(uc, 1, 1, 1, cfg.spin_length, cfg.spin_length_su3);

    seed_lehman(20260430);
    load_gamma2_seed_into_current_storage(
        L, "../example_configs/TmFeO3/tmfeo3_gamma2_1x1x1_seed");

    out << "\n--------------------------------------------------------\n";
    out << " SEEDED  polish=" << n_polish << " T=0 sweeps\n";
    out << "--------------------------------------------------------\n";

    const double E_initial = L.total_energy();
    out << "  initial: total_energy = " << std::scientific << E_initial << "\n";
    for (size_t s = 0; s < n_polish; ++s) L.deterministic_sweep();
    const double E_final = L.total_energy();
    out << "  polished: total_energy = " << E_final
        << "  (dE = " << (E_final - E_initial) << ")\n"
        << std::fixed << std::setprecision(6);

    auto M_sub = L.magnetization_sublattice_SU2();
    print_per_sublattice(out, M_sub);
    print_bertaut(out, compute_bertaut(M_sub));
}

void run_anneal(size_t n_anneal, size_t n_polish, std::ostream& out) {
    SpinConfig cfg = make_gamma2_config();
    MixedUnitCell uc = build_tmfeo3(cfg);
    MixedLattice L(uc, 1, 1, 1, cfg.spin_length, cfg.spin_length_su3);

    seed_lehman(20260430);
    L.init_random();

    out << "\n--------------------------------------------------------\n";
    out << " ANNEAL  n_anneal=" << n_anneal << "   n_polish=" << n_polish << "\n";
    out << "--------------------------------------------------------\n";

    L.simulated_annealing(/*T_start=*/10.0, /*T_end=*/0.01, n_anneal,
                          /*gaussian_move=*/false,
                          /*cooling_rate=*/0.9,
                          /*out_dir=*/"",
                          /*save_observables=*/false,
                          /*T_zero=*/true,
                          n_polish,
                          /*twist_sweep_count=*/100);

    auto M_sub = L.magnetization_sublattice_SU2();
    print_per_sublattice(out, M_sub);
    print_bertaut(out, compute_bertaut(M_sub));
}

}  // namespace

int main(int argc, char** argv) {
    bool do_anneal = false;
    bool do_seeded = true;
    size_t n_polish_seeded = 200;
    size_t n_anneal = 1500;
    size_t n_polish_anneal = 1500;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--anneal") == 0) do_anneal = true;
        else if (std::strcmp(argv[i], "--no-seeded") == 0) do_seeded = false;
        else if (std::strcmp(argv[i], "--quick") == 0) {
            n_polish_seeded = 50;
            n_anneal = 200;
            n_polish_anneal = 200;
        }
    }

    std::cout << "TmFeO3 Gamma_2 ground-state diagnostic\n";
    std::cout << "Current storage: Fe in lab Cartesian, Tm in canonical local CEF basis.\n";
    std::cout << "Gamma_2 expectation: F_x and G_z dominate, C_y secondary,\n";
    std::cout << "                     all other components ~ 0.\n";

    if (do_seeded) {
        std::cout << "\n========================================================\n"
                  << " SEEDED PROTOCOL\n"
                  << "   start: example_configs/TmFeO3/tmfeo3_gamma2_1x1x1_seed_{SU2,SU3}.txt\n"
                  << "   Fe seed is lifted from the retired local-frame storage\n"
                  << "   into the current canonical storage before polishing.\n"
                  << "========================================================\n";
        run_seeded(n_polish_seeded, std::cout);
    }

    if (do_anneal) {
        std::cout << "\n========================================================\n"
                  << " ANNEAL PROTOCOL  (informational only; SA on a 1x1x1\n"
                  << " cell can pick any of the four Pbnm Gamma-domain twins)\n"
                  << "========================================================\n";
        run_anneal(n_anneal, n_polish_anneal, std::cout);
    }

    std::cout << "\nDone.\n";
    return 0;
}
