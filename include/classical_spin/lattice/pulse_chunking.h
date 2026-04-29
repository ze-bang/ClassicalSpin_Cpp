/**
 * @file  pulse_chunking.h
 * @brief Pulse-window chunking helper for the 2DCS / pump-probe drivers.
 *
 * Shared between Lattice, MixedLattice, PhononLattice, and (future)
 * StrainPhononLattice. See docs/optimization_notes.tex Ingredient XV
 * (W3) for the full rationale; the executive summary is:
 *
 *   - The pulse envelope used by every *_pulse_drive() RHS is
 *     exp(−(Δt/(2σ))²) × cos(ω·Δt) (note the factor of 2 in the
 *     denominator — `field_drive_width_*` is *not* the standard
 *     Gaussian σ but rather σ_eff/√2).  At Δt = kPulseWindowSigmas·σ
 *     the envelope is exp(−kPulseWindowSigmas²/4); for the default
 *     value 9 this is ≈ 1.6 × 10⁻⁹, well below any realistic 2DCS
 *     noise floor.  Outside the window the LLG / coupled flow is the
 *     unperturbed deterministic dynamics.
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
 *   - **Grid alignment (audit B1).**  Boost's `integrate_const` for
 *     the controlled-stepper category advances the state in dt-sized
 *     chunks while `time + dt ≤ end_time`, leaves the state at the
 *     largest `start + N·dt ≤ end_time`, and **drops** the trailing
 *     remainder.  If a segment endpoint were not a multiple of
 *     T_step the next segment would re-enter `integrate_const` with
 *     a stale state at every seam, biasing the χ³ response.  We
 *     therefore *snap* every candidate segment endpoint to the
 *     T_step grid before merging, and assert at the bottom that
 *     every (seg.t1 − seg.t0) is an integer multiple of T_step.
 *     For configs whose σ and τ are already multiples of T_step
 *     (the canonical TmFeO3 production runs) this is a no-op; for
 *     drifted configs it converts a silent precision loss into a
 *     correct (slightly shifted) integration window.
 *
 * The helper is purely temporal — it is a pure function of the
 * pulse centres, the integration window, and T_step. It does not
 * touch the lattice and is therefore reusable from every spectroscopy
 * driver.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
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
 *                        == segments[i].t1).  Every segment endpoint
 *                        lies on the T_start + k·T_step grid (audit B1).
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
    if (T_step <= 0.0) return segments;

    // Snap an absolute time to the nearest T_start + k·T_step boundary.
    // Symmetric round → at most T_step/2 off the requested time.  Used to
    // align every candidate window endpoint and the global T_end so each
    // chunk is an exact integer × T_step long (see audit B1).
    auto snap = [&](double t) -> double {
        const double k = std::round((t - T_start) / T_step);
        return T_start + k * T_step;
    };

    // Snap T_end *down* to the largest grid point ≤ T_end.  This is the
    // length the integrator can actually cover with `integrate_const`
    // without dropping a trailing sub-T_step remainder.
    const double T_end_snapped = T_start +
        std::floor((T_end - T_start) / T_step + 1e-9) * T_step;
    if (T_end_snapped <= T_start) return segments;

    std::vector<std::pair<double, double>> windows;
    windows.reserve(pulse_centers.size());
    for (double tc : pulse_centers) {
        double w0 = std::max(T_start, snap(tc - window));
        double w1 = std::min(T_end_snapped, snap(tc + window));
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
    if (cursor < T_end_snapped) {
        const double seg_len = T_end_snapped - cursor;
        const double dt_hint = std::min(free_dt_raw, seg_len);
        segments.push_back({cursor, T_end_snapped, dt_hint});
    }

    // Defence-in-depth: every seam is contiguous, on the T_step grid,
    // and every chunk length is an integer × T_step.  Compiled out in
    // release builds via assert(); in debug a violation indicates a bug
    // in the snapping logic above (not a user-config issue).
    const double tol = 1e-9 * std::max(1.0, std::abs(T_end - T_start));
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i + 1 < segments.size()) {
            assert(std::abs(segments[i].t1 - segments[i + 1].t0) <= tol);
        }
        const double k = (segments[i].t1 - segments[i].t0) / T_step;
        assert(std::abs(k - std::round(k)) <= 1e-6);
    }
    return segments;
}

/// Half-width of each pulse-active region as a multiple of the
/// `field_drive_width_*` parameter σ.
///
/// IMPORTANT: the drive envelope is `exp(−(Δt/(2σ))²)` (see
/// drive_envelopes_SU* in mixed_lattice_md.cpp), i.e. σ here is
/// σ_eff/√2 of the standard Gaussian.  At Δt = kPulseWindowSigmas·σ
/// the envelope amplitude is `exp(−kPulseWindowSigmas²/4)`; for
/// kPulseWindowSigmas = 9 that is ≈ 1.6 × 10⁻⁹.  Audit B2: the old
/// value of 6 only got us `exp(−9) ≈ 10⁻⁴`, which is far above what
/// the comment claimed.
constexpr double kPulseWindowSigmas = 9.0;

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
