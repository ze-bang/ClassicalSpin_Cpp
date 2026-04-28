// =====================================================================
// Regression test: pulse-drive trajectories at abs/rel tol = 1e-8 must
// agree with the legacy 1e-10 reference to ~ a few × 1e-7 on the
// magnetisation observable for a small TmFeO3 mixed lattice driven by
// a short Gaussian pulse.
//
// This pins Ingredient XVIII: relaxing the hard-coded 1e-10 default to
// the new `kDefaultPumpProbeAbsTol` / `kDefaultPumpProbeRelTol` (1e-8)
// inside *_pulse_drive() must not measurably change the 2DCS-grade
// observable. We compare every saved time sample of all six
// magnetisation triples (SU(2) AF / local / global; SU(3) AF / local /
// global) at both x/y/z components.
// =====================================================================

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"
#include "classical_spin/lattice/pulse_chunking.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

namespace {

using Lattice = MixedLattice;
using SpinVector = Eigen::VectorXd;

// Physical observable tolerance: with dopri5 going from 1e-10 to 1e-8
// abs/rel tol, we expect errors to grow at worst like O(1e-7) over the
// short (~10 unit-time) integration window used here. We give ourselves
// a generous 5e-6 ceiling which is *still* three to four orders of
// magnitude below a typical 2DCS χ³ feature (~1e-3 of |M|).
constexpr double kPhysicsAbsCeiling = 5e-6;

Lattice make_lattice() {
    SpinConfig config;
    config.field_strength = 0.0;
    config.spin_length = 1.0f;
    config.spin_length_su3 = 1.0f;
    return Lattice(build_tmfeo3(config), 1, 1, 1,
                   config.spin_length, config.spin_length_su3);
}

SpinVector unit3(double x, double y, double z) {
    SpinVector v(3);
    v << x, y, z;
    const double n = v.norm();
    return n > 0.0 ? (v / n).eval() : v;
}

SpinVector make_su3_init(size_t site) {
    SpinVector v(8);
    for (Eigen::Index d = 0; d < 8; ++d) {
        v(d) = std::sin(0.23 * (site + 1) + 0.37 * (d + 1)) +
               std::cos(0.19 * (site + 1) - 0.11 * (d + 1));
    }
    const double n = v.norm();
    return n > 0.0 ? (v / n).eval() : v;
}

void seed_spins(Lattice& lat) {
    for (size_t s = 0; s < lat.lattice_size_SU2; ++s) {
        const double x = std::sin(0.41 * (s + 1) + 0.20);
        const double y = std::cos(0.67 * (s + 1) - 0.50);
        const double z = std::sin(0.29 * (s + 1) + 1.10);
        lat.spins_SU2[s] = unit3(x, y, z);
    }
    for (size_t s = 0; s < lat.lattice_size_SU3; ++s) {
        lat.spins_SU3[s] = make_su3_init(s);
    }
}

Lattice::PumpProbeTrajectory drive(Lattice& lat, double abs_tol, double rel_tol) {
    seed_spins(lat);
    lat.reset_pulse();

    std::vector<SpinVector> field_su2(lat.lattice_size_SU2, unit3(1.0, 0.0, 0.0));
    std::vector<SpinVector> field_su3(lat.lattice_size_SU3);
    for (size_t s = 0; s < lat.lattice_size_SU3; ++s) {
        SpinVector e1(8);
        e1.setZero();
        e1(0) = 1.0;
        field_su3[s] = e1;
    }

    // Short, well-resolved Gaussian pulse and a moderate evolution
    // window. Both are large enough that a 1e-8 vs 1e-10 difference in
    // accepted dt would, if it propagated linearly, produce a
    // trajectory deviation at least an order of magnitude larger than
    // our O(1e-7) target. So if this passes, the relaxation is safe.
    const double t_B = 1.0;
    const double pulse_amp_su2 = 0.5;
    const double pulse_width_su2 = 0.2;
    const double pulse_freq_su2 = 4.0;
    const double pulse_amp_su3 = 0.3;
    const double pulse_width_su3 = 0.2;
    const double pulse_freq_su3 = 5.0;

    const double T_start = -1.0;
    const double T_end = 8.0;
    const double T_step = 0.05;

    return lat.single_pulse_drive(
        field_su2, field_su3, t_B,
        pulse_amp_su2, pulse_width_su2, pulse_freq_su2,
        pulse_amp_su3, pulse_width_su3, pulse_freq_su3,
        T_start, T_end, T_step,
        "dopri5", /*use_gpu=*/false, /*spin_state_out=*/nullptr,
        /*pulse_window_chunking=*/true,
        abs_tol, rel_tol);
}

double max_abs_diff(const Lattice::PumpProbeTrajectory& a,
                    const Lattice::PumpProbeTrajectory& b,
                    double& worst_t,
                    int& worst_block,
                    int& worst_subblock,
                    int& worst_comp) {
    double worst = 0.0;
    worst_t = 0.0;
    worst_block = worst_subblock = worst_comp = -1;
    if (a.size() != b.size()) {
        std::cerr << "[FAIL] trajectory length mismatch: "
                  << a.size() << " vs " << b.size() << "\n";
        return std::numeric_limits<double>::infinity();
    }

    for (size_t i = 0; i < a.size(); ++i) {
        const auto& [ta, mags_a] = a[i];
        const auto& [tb, mags_b] = b[i];
        if (std::abs(ta - tb) > 1e-12) {
            std::cerr << "[FAIL] sample-time grid mismatch at i=" << i
                      << ": " << ta << " vs " << tb << "\n";
            return std::numeric_limits<double>::infinity();
        }

        // Block 0: SU(2) magnetisation triple (AF / local / global).
        // Block 1: SU(3) magnetisation triple.
        const std::array<const std::array<SpinVector, 3>*, 2> blocks_a = {
            &mags_a.first, &mags_a.second};
        const std::array<const std::array<SpinVector, 3>*, 2> blocks_b = {
            &mags_b.first, &mags_b.second};

        for (int blk = 0; blk < 2; ++blk) {
            for (int sub = 0; sub < 3; ++sub) {
                const auto& va = (*blocks_a[blk])[sub];
                const auto& vb = (*blocks_b[blk])[sub];
                if (va.size() != vb.size()) {
                    std::cerr << "[FAIL] mag size mismatch at i=" << i
                              << " blk=" << blk << " sub=" << sub << "\n";
                    return std::numeric_limits<double>::infinity();
                }
                for (Eigen::Index c = 0; c < va.size(); ++c) {
                    const double d = std::abs(va(c) - vb(c));
                    if (d > worst) {
                        worst = d;
                        worst_t = ta;
                        worst_block = blk;
                        worst_subblock = sub;
                        worst_comp = static_cast<int>(c);
                    }
                }
            }
        }
    }
    return worst;
}

}  // namespace

int main() {
    Lattice lat = make_lattice();

    std::cout << "[INFO] Pulse-drive tolerance regression\n"
              << "       reference: abs_tol=rel_tol=1e-10\n"
              << "       relaxed  : abs_tol=rel_tol="
              << classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol
              << "\n";

    using clock = std::chrono::steady_clock;
    // Warm caches once so we don't penalise the first call.
    (void) drive(lat, 1e-10, 1e-10);

    // Reference run at the previous hard-coded tolerance (1e-10).
    const auto t0 = clock::now();
    auto traj_ref = drive(lat, 1e-10, 1e-10);
    const auto t1 = clock::now();
    // Relaxed run at the new default (kDefaultPumpProbe* = 1e-8).
    auto traj_relaxed = drive(lat,
        classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol,
        classical_spin_pulse_chunking::kDefaultPumpProbeRelTol);
    const auto t2 = clock::now();

    const double dt_ref = std::chrono::duration<double>(t1 - t0).count();
    const double dt_rlx = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "[BENCH] single_pulse_drive timings (one shot):\n"
              << "        tol=1e-10 : " << dt_ref << " s\n"
              << "        tol=1e-8  : " << dt_rlx << " s"
              << "  (speedup " << (dt_rlx > 0.0 ? dt_ref / dt_rlx : 0.0)
              << "×)\n";

    double t_w = 0.0;
    int blk_w = -1, sub_w = -1, comp_w = -1;
    const double err = max_abs_diff(traj_ref, traj_relaxed,
                                    t_w, blk_w, sub_w, comp_w);

    std::cout << "[INFO] worst |ΔM| = " << err
              << "  at t=" << t_w
              << " block=" << blk_w
              << " subblock=" << sub_w
              << " component=" << comp_w
              << " (over " << traj_ref.size()
              << " samples × 6 mags × ~3 components)\n";

    if (!(err < kPhysicsAbsCeiling)) {
        std::cerr << "[FAIL] worst |ΔM| = " << err
                  << " exceeds physics ceiling " << kPhysicsAbsCeiling
                  << ". Relaxing the pulse-drive default tolerance from"
                     " 1e-10 to "
                  << classical_spin_pulse_chunking::kDefaultPumpProbeAbsTol
                  << " is no longer safe; revisit Ingredient XVIII.\n";
        return 1;
    }

    std::cout << "[PASS] Pulse-drive 1e-8 trajectories track 1e-10 to "
              << err << " (< " << kPhysicsAbsCeiling << ")\n";
    return 0;
}
