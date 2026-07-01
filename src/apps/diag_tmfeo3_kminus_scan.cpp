// diag_tmfeo3_kminus_scan.cpp
// ---------------------------------------------------------------------------
// Scans the K^- Fe-Tm coupling tensor one component at a time.
//
// For each of the 9 K^- entries (Kminus_{2,5,7}{x,y,z}) set to KMINUS_STRENGTH,
// finds the Gamma_2 ground state via the two-stage protocol and reports both
// the Fe Bertaut vectors and the induced Tm moment (per sublattice and uniform).
//
// Physical J projection (mu_act defaults):
//   Jz = 5.264 * lam2
//   Jx = 2.3915*lam5 + 0.9128*lam7
//   Jy = -2.7866*lam5 + 0.4655*lam7
//
// Output goes to tfo_project/kminus_scan/.
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

// Physical mu_act (Jz=mu_2z*lam2, Jx=mu_5x*lam5+mu_7x*lam7, Jy=...)
static constexpr double MU_2Z =  5.264;
static constexpr double MU_5X =  2.3915,  MU_7X = 0.9128;
static constexpr double MU_5Y = -2.7866,  MU_7Y = 0.4655;

namespace {

// ---------------------------------------------------------------------------
// Configs
// ---------------------------------------------------------------------------

SpinConfig make_base_exchange() {
    SpinConfig c;
    c.spin_length     = 2.5f;
    c.spin_length_su3 = 1.0f;
    c.field_strength  = 0.0;
    c.set_param("J1ab", 4.74); c.set_param("J1c",  5.15);
    c.set_param("J2ab", 0.15); c.set_param("J2c",  0.30);
    c.set_param("D1",   0.049); c.set_param("D2",   0.0);
    c.set_param("e1",   2.067834); c.set_param("e2", 4.9628);
    return c;
}

SpinConfig make_biased_config() {
    SpinConfig c = make_base_exchange();
    c.set_param("Ka", 0.0); c.set_param("Kb", 0.0); c.set_param("Kc", -0.029);
    return c;
}

SpinConfig make_physical_config(const std::string& kminus_key, double value) {
    SpinConfig c = make_base_exchange();
    c.set_param("Ka", -0.0153); c.set_param("Kb", 0.0); c.set_param("Kc", -0.0187);
    if (!kminus_key.empty()) c.set_param(kminus_key, value);
    return c;
}

// ---------------------------------------------------------------------------
// Bertaut vectors (valid for 1x1x1: site==sublattice)
// ---------------------------------------------------------------------------

struct BertautVec { Eigen::Vector3d F, G, A, C; };

BertautVec compute_bertaut(const Lattice& L) {
    BertautVec bv;
    const auto& S = L.spins_SU2;
    bv.F = (S[0]+S[1]+S[2]+S[3])*0.25;
    bv.G = (S[0]-S[1]+S[2]-S[3])*0.25;
    bv.A = (S[0]+S[1]-S[2]-S[3])*0.25;
    bv.C = (S[0]-S[1]-S[2]+S[3])*0.25;
    return bv;
}

bool is_gamma2(const BertautVec& bv) {
    return std::abs(bv.G(2)) > 2.0 && bv.A.norm() < 0.2 && bv.C.norm() < 0.2;
}

// ---------------------------------------------------------------------------
// Tm readout
// ---------------------------------------------------------------------------

struct TmMoment {
    // time-odd active generators
    double lam2, lam5, lam7;
    // time-even off-diagonal
    double lam1, lam4, lam6;
    // diagonal populations
    double lam3, lam8;
    // physical moment
    double Jx, Jy, Jz;
};

TmMoment read_tm(const Eigen::VectorXd& n) {
    TmMoment m;
    m.lam1 = n(0); m.lam2 = n(1); m.lam3 = n(2);
    m.lam4 = n(3); m.lam5 = n(4); m.lam6 = n(5);
    m.lam7 = n(6); m.lam8 = n(7);
    m.Jz   = MU_2Z * m.lam2;
    m.Jx   = MU_5X * m.lam5 + MU_7X * m.lam7;
    m.Jy   = MU_5Y * m.lam5 + MU_7Y * m.lam7;
    return m;
}

// ---------------------------------------------------------------------------
// Seed I/O
// ---------------------------------------------------------------------------

using SpinArr = std::vector<Eigen::VectorXd>;

SpinArr snapshot_SU2(const Lattice& L) {
    SpinArr v;
    for (size_t s = 0; s < L.lattice_size_SU2; ++s) v.push_back(L.spins_SU2[s]);
    return v;
}
SpinArr snapshot_SU3(const Lattice& L) {
    SpinArr v;
    for (size_t s = 0; s < L.lattice_size_SU3; ++s) v.push_back(L.spins_SU3[s]);
    return v;
}
void load_state(Lattice& L, const SpinArr& su2, const SpinArr& su3) {
    for (size_t s = 0; s < su2.size(); ++s) L.spins_SU2[s] = su2[s];
    for (size_t s = 0; s < su3.size(); ++s) L.spins_SU3[s] = su3[s];
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

void print_v3(std::ostream& out, const char* name, const Eigen::Vector3d& v) {
    out << "  " << name << " = ("
        << std::setw(11) << std::setprecision(6) << std::fixed << v(0) << ", "
        << std::setw(11) << v(1) << ", "
        << std::setw(11) << v(2) << ")  |.|="
        << std::setw(9) << v.norm() << "\n";
}

void print_tm_moment(std::ostream& out, int sub, const TmMoment& m) {
    out << std::scientific << std::setprecision(4);
    out << "  Tm" << sub
        << "  lam(2,5,7)=(" << m.lam2 << ", " << m.lam5 << ", " << m.lam7 << ")"
        << "  lam(1,4,6)=(" << m.lam1 << ", " << m.lam4 << ", " << m.lam6 << ")"
        << "  lam(3,8)=("   << m.lam3 << ", " << m.lam8 << ")"
        << "\n       J=(Jx=" << m.Jx << ", Jy=" << m.Jy << ", Jz=" << m.Jz << ")\n";
}

void write_scan_row(std::ofstream& out, const std::string& label,
                    double E, const BertautVec& bv,
                    const std::array<TmMoment, 4>& tm) {
    // Uniform Tm moment (sum over sublattices / 4)
    double Jx_u = 0, Jy_u = 0, Jz_u = 0;
    double lam2_u = 0, lam5_u = 0, lam7_u = 0;
    for (const auto& m : tm) {
        Jx_u   += m.Jx;   Jy_u   += m.Jy;   Jz_u   += m.Jz;
        lam2_u += m.lam2; lam5_u += m.lam5; lam7_u += m.lam7;
    }
    Jx_u /= 4; Jy_u /= 4; Jz_u /= 4;
    lam2_u /= 4; lam5_u /= 4; lam7_u /= 4;

    out << std::fixed << std::setprecision(6);
    out << label
        << "  E=" << E
        << "  Fx=" << bv.F(0) << "  Gz=" << bv.G(2)
        << "  Gx=" << bv.G(0) << "  Gy=" << bv.G(1)
        << "  lam2u=" << lam2_u << "  lam5u=" << lam5_u << "  lam7u=" << lam7_u
        << "  Jxu=" << Jx_u    << "  Jyu=" << Jy_u      << "  Jzu=" << Jz_u
        << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    constexpr double KMINUS_STRENGTH = 0.2;       // meV
    constexpr size_t N_ANNEAL        = 2000;
    constexpr size_t N_DET_BIAS      = 2000;       // Stage 1 T=0 sweeps
    constexpr size_t N_DET_PHYS      = 200000;     // Stage 2 fixed det sweeps
    constexpr double T_START = 15.0, T_END = 1e-3, RATE = 0.9;

    const std::string out_dir = (argc > 1) ? argv[1] : "../tfo_project/kminus_scan";
    std::filesystem::create_directories(out_dir);

    // The 9 K^- parameter keys: lambda column {2,5,7} x Fe row {x,y,z}
    const std::array<std::string, 9> keys = {
        "Kminus_2x", "Kminus_2y", "Kminus_2z",
        "Kminus_5x", "Kminus_5y", "Kminus_5z",
        "Kminus_7x", "Kminus_7y", "Kminus_7z",
    };

    std::cout << "================================================================\n"
              << " TmFeO3 K^- component scan (1x1x1 mixed, Gamma_2 ground state)\n"
              << " Kminus strength = " << KMINUS_STRENGTH << " meV\n"
              << " Two-stage: biased SA (Ka=0 Kc=-0.029) + physical relax\n"
              << "   Stage 2: Ka=-0.0153 Kc=-0.0187 + K^- component, "
              << N_DET_PHYS << " det sweeps\n"
              << "================================================================\n\n";

    const SpinConfig biased_cfg = make_biased_config();

    // Summary table (written to file)
    std::ofstream table(out_dir + "/scan_table.txt");
    table << "# TmFeO3 K^- component scan  (strength=" << KMINUS_STRENGTH << " meV)\n";
    table << "# Columns: label  E/site  Fx  Gz  Gx  Gy  "
             "lam2_uniform  lam5_uniform  lam7_uniform  Jx_uniform  Jy_uniform  Jz_uniform\n";

    // Scan: baseline (K^-=0) first, then each of the 9 components
    std::vector<std::pair<std::string,std::string>> scan_cases;
    scan_cases.push_back({"baseline", ""});
    for (const auto& k : keys) scan_cases.push_back({k, k});

    for (const auto& [label, key] : scan_cases) {
        std::cout << "------------------------------------------------------------\n"
                  << " " << label
                  << (key.empty() ? "" : " = " + std::to_string(KMINUS_STRENGTH) + " meV")
                  << "\n------------------------------------------------------------\n";

        // Stage 1: biased SA (K^-=0, z-easy) → reliable Gamma_2
        Lattice lat_bias(build_tmfeo3(biased_cfg), 1, 1, 1,
                         biased_cfg.spin_length, biased_cfg.spin_length_su3);
        lat_bias.init_random();
        lat_bias.simulated_annealing(T_START, T_END, N_ANNEAL,
            false, RATE, "", false, true, N_DET_BIAS, 0);
        lat_bias.physicalize_SU3_state();

        const BertautVec bv_bias = compute_bertaut(lat_bias);
        if (!is_gamma2(bv_bias)) {
            std::cout << "  [WARN] biased SA did not reach Gamma_2; skipping.\n\n";
            continue;
        }

        // Stage 2: rebuild with physical params + K^- component, transfer state
        const SpinConfig phys_cfg = make_physical_config(key, KMINUS_STRENGTH);
        Lattice lat_phys(build_tmfeo3(phys_cfg), 1, 1, 1,
                         phys_cfg.spin_length, phys_cfg.spin_length_su3);
        load_state(lat_phys, snapshot_SU2(lat_bias), snapshot_SU3(lat_bias));

        for (size_t sw = 0; sw < N_DET_PHYS; ++sw)
            lat_phys.deterministic_sweep();
        lat_phys.physicalize_SU3_state();

        const double E = lat_phys.energy_density();
        const BertautVec bv = compute_bertaut(lat_phys);

        // Collect Tm moments
        std::array<TmMoment, 4> tm;
        for (int s = 0; s < 4; ++s) tm[s] = read_tm(lat_phys.spins_SU3[s]);

        // Uniform Tm (F-channel)
        double Jx_u = 0, Jy_u = 0, Jz_u = 0;
        double lam2_u = 0, lam5_u = 0, lam7_u = 0;
        for (const auto& m : tm) {
            Jx_u   += m.Jx;   Jy_u   += m.Jy;   Jz_u   += m.Jz;
            lam2_u += m.lam2; lam5_u += m.lam5; lam7_u += m.lam7;
        }
        Jx_u /= 4; Jy_u /= 4; Jz_u /= 4;
        lam2_u /= 4; lam5_u /= 4; lam7_u /= 4;

        // Print
        std::cout << "  E/site = " << std::scientific << std::setprecision(6) << E << "\n";
        std::cout << "  Fe Bertaut:\n";
        print_v3(std::cout, "F", bv.F);
        print_v3(std::cout, "G", bv.G);
        print_v3(std::cout, "A", bv.A);
        print_v3(std::cout, "C", bv.C);
        std::cout << "  Tm per-sublattice:\n";
        for (int s = 0; s < 4; ++s) print_tm_moment(std::cout, s, tm[s]);
        std::cout << "  Tm uniform (F-channel):\n"
                  << std::scientific << std::setprecision(4)
                  << "    lam2=" << lam2_u << "  lam5=" << lam5_u << "  lam7=" << lam7_u << "\n"
                  << "    Jx="   << Jx_u   << "  Jy="   << Jy_u   << "  Jz="   << Jz_u   << "\n\n";

        // Save per-case seed files
        const std::string case_dir = out_dir + "/" + label;
        std::filesystem::create_directories(case_dir);
        {
            std::ofstream f(case_dir + "/seed_SU2.txt");
            f << std::setprecision(17);
            for (size_t s = 0; s < lat_phys.lattice_size_SU2; ++s) {
                const auto& sp = lat_phys.spins_SU2[s];
                f << sp(0) << " " << sp(1) << " " << sp(2) << "\n";
            }
        }
        {
            std::ofstream f(case_dir + "/seed_SU3.txt");
            f << std::setprecision(17);
            for (size_t s = 0; s < lat_phys.lattice_size_SU3; ++s) {
                const auto& sp = lat_phys.spins_SU3[s];
                for (int d = 0; d < 8; ++d) { if (d > 0) f << " "; f << sp(d); }
                f << "\n";
            }
        }
        {
            std::ofstream f(case_dir + "/summary.txt");
            f << std::setprecision(10);
            f << "# K^- component: " << (key.empty() ? "none (baseline)" : key)
              << " = " << KMINUS_STRENGTH << " meV\n";
            f << "E_per_site_meV = " << E << "\n";
            f << "# Bertaut\n";
            f << "F = " << bv.F(0) << " " << bv.F(1) << " " << bv.F(2) << "\n";
            f << "G = " << bv.G(0) << " " << bv.G(1) << " " << bv.G(2) << "\n";
            f << "A = " << bv.A(0) << " " << bv.A(1) << " " << bv.A(2) << "\n";
            f << "C = " << bv.C(0) << " " << bv.C(1) << " " << bv.C(2) << "\n";
            f << "# Tm uniform\n";
            f << "lam2_u=" << lam2_u << "  lam5_u=" << lam5_u << "  lam7_u=" << lam7_u << "\n";
            f << "Jx_u=" << Jx_u << "  Jy_u=" << Jy_u << "  Jz_u=" << Jz_u << "\n";
            for (int s = 0; s < 4; ++s) {
                f << "Tm" << s << " lam=(";
                for (int d = 0; d < 8; ++d) {
                    if (d > 0) f << " ";
                    f << lat_phys.spins_SU3[s](d);
                }
                f << ")\n";
            }
        }

        // Append to summary table
        write_scan_row(table, label, E, bv, tm);
    }

    table.close();
    std::cout << "================================================================\n"
              << " Scan complete. Results in " << out_dir << "/\n"
              << "================================================================\n";
    return 0;
}
