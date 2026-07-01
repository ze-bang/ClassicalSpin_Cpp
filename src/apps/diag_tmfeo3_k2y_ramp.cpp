// diag_tmfeo3_k2y_ramp.cpp
// ---------------------------------------------------------------------------
// Adiabatically ramps a selected K^- component from 0 to KMAX in small steps.
// At each step the previous ground state is used as seed (local relaxation),
// so we track the LOCALLY STABLE Gamma_2 phase until it collapses.
//
// Reports: K  E/site  Gz  Gy  lam2_uniform  is_gamma2
// Default output:  tfo_project/kminus_ramp_<component>/ramp_table.txt
// ---------------------------------------------------------------------------

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using Lattice = MixedLattice;

// ---------------------------------------------------------------------------
// Helpers (shared with kminus_scan)
// ---------------------------------------------------------------------------

static constexpr double MU_2Z = 5.264;

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

int main(int argc, char* argv[]) {
    constexpr double KMIN          = 0.0;
    constexpr double DEFAULT_KMAX  = 0.25;
    constexpr double DEFAULT_KSTEP = 0.005;
    constexpr size_t N_DET     = 100000; // sweeps per step (local relaxation only)
    constexpr size_t N_ANNEAL  = 3000;
    constexpr double T_START   = 15.0, T_END = 1e-3, RATE = 0.9;

    const std::string kminus_key = (argc > 1) ? argv[1] : "Kminus_2y";
    if (kminus_key != "Kminus_2x" && kminus_key != "Kminus_2y" && kminus_key != "Kminus_2z") {
        std::cerr << "Usage: " << argv[0]
                  << " [Kminus_2x|Kminus_2y|Kminus_2z] [output_dir] [kmax_meV] [kstep_meV]"
                     " [Ka] [Kb] [Kc] [J1ab] [J1c]\n";
        return 2;
    }

    std::string short_label = kminus_key;
    const std::string prefix = "Kminus_";
    if (short_label.rfind(prefix, 0) == 0) {
        short_label = short_label.substr(prefix.size());
    }

    const std::string out_dir = (argc > 2)
        ? argv[2]
        : "../tfo_project/kminus_ramp_" + short_label;
    const double kmax = (argc > 3) ? std::stod(argv[3]) : DEFAULT_KMAX;
    const double kstep = (argc > 4) ? std::stod(argv[4]) : DEFAULT_KSTEP;
    // Optional Fe anisotropy / exchange overrides (argv 5..9)
    const double Ka_override   = (argc > 5) ? std::stod(argv[5]) : -0.0153;
    const double Kb_override   = (argc > 6) ? std::stod(argv[6]) : 0.0;
    const double Kc_override   = (argc > 7) ? std::stod(argv[7]) : -0.0187;
    const double J1ab_override = (argc > 8) ? std::stod(argv[8]) : 4.74;
    const double J1c_override  = (argc > 9) ? std::stod(argv[9]) : 5.15;
    if (kmax < KMIN || kstep <= 0.0) {
        std::cerr << "[ERROR] Require kmax >= 0 and kstep > 0.\n";
        return 2;
    }
    std::filesystem::create_directories(out_dir);

    std::cout << "Fe params: Ka=" << Ka_override << " Kb=" << Kb_override
              << " Kc=" << Kc_override << " J1ab=" << J1ab_override
              << " J1c=" << J1c_override << "\n";

    // ── Stage 0: biased SA to get clean Gamma_2 seed ─────────────────────
    // Use Kc_override (already negative z-easy) for bias SA; if Kc_override is
    // weaker than -0.029 we use -0.029 for the initial bias to ensure Gamma_2 lock.
    const double Kc_bias = (Kc_override < -0.029) ? Kc_override : -0.029;
    SpinConfig bias_cfg = make_base_exchange();
    bias_cfg.set_param("J1ab", J1ab_override); bias_cfg.set_param("J1c", J1c_override);
    bias_cfg.set_param("Ka", 0.0); bias_cfg.set_param("Kb", 0.0);
    bias_cfg.set_param("Kc", Kc_bias);

    Lattice lat(build_tmfeo3(bias_cfg), 1, 1, 1,
                bias_cfg.spin_length, bias_cfg.spin_length_su3);
    lat.init_random();
    lat.simulated_annealing(T_START, T_END, N_ANNEAL,
                            false, RATE, "", false, true, 2000, 0);
    lat.physicalize_SU3_state();

    BertautVec bv0 = compute_bertaut(lat);
    if (!is_gamma2(bv0)) {
        std::cerr << "[ERROR] Initial biased SA did not reach Gamma_2. Aborting.\n";
        return 1;
    }
    std::cout << "Initial Gamma_2 seed OK: Gz=" << bv0.G(2) << "\n\n";

    // Switch to override anisotropy (K^-=0 baseline), do a short relax
    SpinConfig phys_base = make_base_exchange();
    phys_base.set_param("J1ab", J1ab_override); phys_base.set_param("J1c", J1c_override);
    phys_base.set_param("Ka", Ka_override); phys_base.set_param("Kb", Kb_override);
    phys_base.set_param("Kc", Kc_override);

    {
        Lattice tmp(build_tmfeo3(phys_base), 1, 1, 1,
                    phys_base.spin_length, phys_base.spin_length_su3);
        for (size_t s = 0; s < lat.lattice_size_SU2; ++s) tmp.spins_SU2[s] = lat.spins_SU2[s];
        for (size_t s = 0; s < lat.lattice_size_SU3; ++s) tmp.spins_SU3[s] = lat.spins_SU3[s];
        for (size_t sw = 0; sw < 20000; ++sw) tmp.deterministic_sweep();
        tmp.physicalize_SU3_state();
        lat = std::move(tmp);
    }

    // ── Ramp table ────────────────────────────────────────────────────────
    std::ofstream table(out_dir + "/ramp_table.txt");
    table << "# Adiabatic " << kminus_key << " ramp  step=" << kstep << " meV  N_det=" << N_DET << "\n";
    table << "# Ka=" << Ka_override << " Kb=" << Kb_override << " Kc=" << Kc_override
          << " J1ab=" << J1ab_override << " J1c=" << J1c_override << "\n";
    table << "# K2y_meV   E/site       Gz          Gy          lam2_uniform  gamma2\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << kminus_key << " ramp\n";
    std::cout << "K(meV)     Gz        Gy        lam2_u    Gamma2?\n";
    std::cout << "---------- --------- --------- --------- -------\n";

    const int n_steps = static_cast<int>(std::round((kmax - KMIN) / kstep)) + 1;

    for (int step = 0; step < n_steps; ++step) {
        const double kval = KMIN + step * kstep;

        // Build lattice at this coupling value, transfer current state
        SpinConfig cfg = make_base_exchange();
        cfg.set_param("J1ab", J1ab_override); cfg.set_param("J1c", J1c_override);
        cfg.set_param("Ka", Ka_override); cfg.set_param("Kb", Kb_override);
        cfg.set_param("Kc", Kc_override);
        cfg.set_param(kminus_key, kval);

        Lattice new_lat(build_tmfeo3(cfg), 1, 1, 1,
                        cfg.spin_length, cfg.spin_length_su3);
        for (size_t s = 0; s < lat.lattice_size_SU2; ++s) new_lat.spins_SU2[s] = lat.spins_SU2[s];
        for (size_t s = 0; s < lat.lattice_size_SU3; ++s) new_lat.spins_SU3[s] = lat.spins_SU3[s];

        for (size_t sw = 0; sw < N_DET; ++sw) new_lat.deterministic_sweep();
        new_lat.physicalize_SU3_state();

        const double E    = new_lat.energy_density();
        const BertautVec bv = compute_bertaut(new_lat);

        // Uniform lam2
        double lam2_u = 0;
        for (size_t s = 0; s < new_lat.lattice_size_SU3; ++s)
            lam2_u += new_lat.spins_SU3[s](1);
        lam2_u /= static_cast<double>(new_lat.lattice_size_SU3);

        const bool g2 = is_gamma2(bv);
        const char* flag = g2 ? "YES" : "NO ";

          std::cout << std::setw(10) << kval << " "
                  << std::setw(9)  << bv.G(2) << " "
                  << std::setw(9)  << bv.G(1) << " "
                  << std::setw(9)  << lam2_u  << "  " << flag << "\n";
        std::cout.flush();

          table << std::setw(10) << std::fixed << std::setprecision(5) << kval << "  "
              << std::scientific << std::setprecision(6) << E << "  "
              << std::fixed << std::setprecision(6)
              << bv.G(2) << "  " << bv.G(1) << "  " << lam2_u << "  " << flag << "\n";
        table.flush();

        lat = std::move(new_lat);

        if (!g2) {
            std::cout << "\n*** Left Gamma_2 at " << kminus_key << " = " << kval << " meV ***\n";
            break;
        }
    }

    std::cout << "\nTable written to " << out_dir << "/ramp_table.txt\n";
    return 0;
}
