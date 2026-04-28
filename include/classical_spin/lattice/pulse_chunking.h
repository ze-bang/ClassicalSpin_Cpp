/**
 * @file  pulse_chunking.h
 * @brief Pulse-window chunking helper for the 2DCS / pump-probe drivers.
 *
 * Shared between Lattice, MixedLattice, PhononLattice, and (future)
 * StrainPhononLattice. See docs/optimization_notes.tex Ingredient XV
 * (W3) for the full rationale; the executive summary is:
 *
 *   - The Gaussian envelope of a THz / RF pulse decays to ~10⁻¹⁶ of
 *     its peak amplitude past 6σ. Outside that window the LLG (or
 *     coupled spin-strain / phonon) flow is just the unperturbed
 *     deterministic dynamics.
 *   - Boost.Odeint's `integrate_const(controlled_stepper, ..., dt)`
 *     uses `dt` as both the *observer cadence* and the *initial step
 *     hint* for the controlled stepper. With dt = T_step (typically
 *     0.01 in our normalised units) the stepper is forced to start
 *     small even when the dynamics is smooth.
 *   - By splitting the integration into pulse-active and free
 *     segments, and giving the controlled stepper a much larger
 *     dt-hint in the free segments (≈ 20× T_step), we let it grow
 *     its internal step rapidly. The observer keeps firing at exact
 *     T_step boundaries within each segment via interpolation, so
 *     the trajectory grid is unchanged.
 *
 * The helper is purely temporal — it is a pure function of the
 * pulse centres, the integration window, and T_step. It does not
 * touch the lattice and is therefore reusable from every spectroscopy
 * driver.
 */

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

namespace classical_spin_pulse_chunking {

struct Segment {
    double t0;
    double t1;
    double dt_init;
};

/**
 * Build the alternating sequence of free / pulse-active segments
 * covering [T_start, T_end] given the pulse centres.
 *
 * @param T_start         Integration window start.
 * @param T_end           Integration window end.
 * @param pulse_centers   Pulse centre times (one per pulse).
 * @param window          Half-width of the active region (typically
 *                        kPulseWindowSigmas × pulse_width).
 * @param T_step          Observer cadence requested by the caller.
 * @param free_dt_factor  Multiplicative factor for the dt-hint in
 *                        free segments (typically kFreeDtFactor).
 * @return                Ordered list of segments {t0, t1, dt_init}
 *                        with t0 < t1 and contiguous (segments[i+1].t0
 *                        == segments[i].t1).
 */
inline std::vector<Segment> build_pulse_segments(
    double T_start,
    double T_end,
    const std::vector<double>& pulse_centers,
    double window,
    double T_step,
    double free_dt_factor)
{
    std::vector<Segment> segments;
    if (T_end <= T_start) return segments;

    std::vector<std::pair<double, double>> windows;
    windows.reserve(pulse_centers.size());
    for (double tc : pulse_centers) {
        double w0 = std::max(T_start, tc - window);
        double w1 = std::min(T_end,   tc + window);
        if (w0 < w1) windows.emplace_back(w0, w1);
    }
    std::sort(windows.begin(), windows.end());

    std::vector<std::pair<double, double>> merged;
    merged.reserve(windows.size());
    for (const auto& w : windows) {
        if (!merged.empty() && w.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, w.second);
        } else {
            merged.push_back(w);
        }
    }

    const double active_dt   = T_step;
    const double free_dt_raw = std::max(T_step, free_dt_factor * T_step);

    double cursor = T_start;
    for (const auto& [w0, w1] : merged) {
        if (cursor < w0) {
            const double seg_len = w0 - cursor;
            const double dt_hint = std::min(free_dt_raw, seg_len);
            segments.push_back({cursor, w0, dt_hint});
        }
        segments.push_back({w0, w1, active_dt});
        cursor = w1;
    }
    if (cursor < T_end) {
        const double seg_len = T_end - cursor;
        const double dt_hint = std::min(free_dt_raw, seg_len);
        segments.push_back({cursor, T_end, dt_hint});
    }
    return segments;
}

/// Half-width of each pulse-active region as a multiple of the
/// Gaussian σ. 6 σ captures > 99.9999% of the envelope, well below
/// any practical 2DCS noise floor.
constexpr double kPulseWindowSigmas = 6.0;

/// Multiplicative factor for the dt-hint in free segments. 20× is
/// conservative — the controlled stepper still enforces the
/// abs/rel tolerances, so the observable trajectory is unchanged.
constexpr double kFreeDtFactor = 20.0;

/// Default abs/rel tolerances for the pump-probe / 2DCS pulse drivers.
///
/// Historically these were hard-coded to 1e-10 in every
/// `*_pulse_drive` function across the three lattice families. That
/// is far tighter than the 2DCS observable needs:
///   - Typical χ³ responses are ~10⁻³ of |M|, so ~5 significant
///     figures in the trajectory (≈ 10⁻⁸ absolute) is plenty.
///   - dopri5's accepted step scales as dt ∝ tol^{1/5}; relaxing
///     tol from 1e-10 to 1e-8 gives ~100^{1/5} ≈ 2.5× larger steps,
///     i.e. ~2.5× fewer RHS calls per integration.
///   - The new SpinConfig fields `pump_probe_abs_tol` /
///     `pump_probe_rel_tol` let users restore 1e-10 (or tighten
///     further) on a per-run basis.
constexpr double kDefaultPumpProbeAbsTol = 1e-8;
constexpr double kDefaultPumpProbeRelTol = 1e-8;

}  // namespace classical_spin_pulse_chunking
