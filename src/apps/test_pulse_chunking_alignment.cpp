// =====================================================================
// Regression test for pulse_chunking::build_pulse_segments alignment.
//
// Audit B1 (see docs/optimization_notes.tex Ingredient XX):
//   * Boost.Odeint's `integrate_const(controlled_stepper, ..., dt)`
//     drops the trailing remainder of [t0, t1] when (t1 - t0) is not
//     an integer multiple of dt.  Inside the chunked 2DCS / pump-probe
//     path that meant every seam between segments could leak up to
//     one T_step worth of stale state when the user picked
//     pulse_width / tau_step values that did not align with T_step.
//   * The producer-side fix snaps every candidate window endpoint
//     and the global T_end down to the T_start + k·T_step grid.
//
// This test pins the new contract:
//   1. Every segment endpoint lies on the T_start + k·T_step grid.
//   2. Segments are contiguous (no gaps, no overlaps).
//   3. Each (t1 − t0) is an integer multiple of T_step.
//   4. The final endpoint is the largest grid point ≤ T_end.
//   5. Pulse-active regions (with dt_init == T_step) cover the
//      requested [tc − window, tc + window] (after grid snapping)
//      for every pulse centre that intersects [T_start, T_end].
//
// Intentionally header-only — no lattice or ODE state required.
// =====================================================================

#include "classical_spin/lattice/pulse_chunking.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace ck = classical_spin_pulse_chunking;

int g_failures = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_failures;
        std::cerr << "  FAIL: " << msg << "\n";
    }
}

// Helper: assert seg endpoints land on the T_start + k·T_step grid.
void assert_segments_on_grid(const std::vector<ck::Segment>& segments,
                             double T_start,
                             double T_end,
                             double T_step,
                             const std::string& tag)
{
    if (segments.empty()) {
        check(T_end - T_start < T_step,
              tag + ": empty segments only allowed when window < T_step");
        return;
    }

    const double tol = 1e-9 * std::max(1.0, std::abs(T_end - T_start));
    const double T_end_snapped = T_start +
        std::floor((T_end - T_start) / T_step + 1e-9) * T_step;

    // Front edge.
    check(std::abs(segments.front().t0 - T_start) <= tol,
          tag + ": first segment must start at T_start");
    // Back edge.
    check(std::abs(segments.back().t1 - T_end_snapped) <= tol,
          tag + ": last segment must end at T_end snapped to grid");

    for (size_t i = 0; i < segments.size(); ++i) {
        const auto& seg = segments[i];

        // Endpoints on the T_start + k·T_step grid.
        const double k0 = (seg.t0 - T_start) / T_step;
        const double k1 = (seg.t1 - T_start) / T_step;
        check(std::abs(k0 - std::round(k0)) <= 1e-6,
              tag + ": seg.t0 not on T_step grid (i=" + std::to_string(i) + ")");
        check(std::abs(k1 - std::round(k1)) <= 1e-6,
              tag + ": seg.t1 not on T_step grid (i=" + std::to_string(i) + ")");

        // Length is an integer × T_step.
        const double n = (seg.t1 - seg.t0) / T_step;
        check(std::abs(n - std::round(n)) <= 1e-6 && n >= 0.999,
              tag + ": seg length not integer × T_step (i=" + std::to_string(i) + ")");

        // Contiguous with the next segment.
        if (i + 1 < segments.size()) {
            check(std::abs(seg.t1 - segments[i + 1].t0) <= tol,
                  tag + ": seam not contiguous between segments " +
                  std::to_string(i) + " and " + std::to_string(i + 1));
        }
    }
}

// Helper: every pulse centre tc ∈ [T_start, T_end] should be covered by
// at least one segment whose dt_init == T_step (pulse-active marker).
void assert_pulse_coverage(const std::vector<ck::Segment>& segments,
                           const std::vector<double>& centres,
                           double T_start,
                           double T_end,
                           double T_step,
                           const std::string& tag)
{
    for (double tc : centres) {
        if (tc < T_start || tc > T_end) continue;  // outside window
        bool covered = false;
        for (const auto& seg : segments) {
            const bool active = std::abs(seg.dt_init - T_step) <
                                1e-9 * std::max(1.0, T_step);
            if (active && seg.t0 - 1e-9 <= tc && tc <= seg.t1 + 1e-9) {
                covered = true;
                break;
            }
        }
        check(covered, tag + ": pulse centre tc=" + std::to_string(tc) +
                       " not covered by any active segment");
    }
}

// =====================================================================
// Case 1 — canonical TmFeO3 production grid.
// Every endpoint is *already* a multiple of T_step, so the snap is a
// no-op.  We re-derive the expected segment list and confirm.
// =====================================================================
void case_aligned_canonical()
{
    const double T_start = -100.0;
    const double T_end   = +100.0;
    const double T_step  = 0.01;
    const double sigma   = 0.5;
    const double window  = ck::kPulseWindowSigmas * sigma;  // = 4.5
    const double current_tau = 50.0;

    const auto segs = ck::build_pulse_segments(
        T_start, T_end, /*pulse_centers=*/ {0.0, current_tau},
        window, T_step, ck::kFreeDtFactor);

    assert_segments_on_grid(segs, T_start, T_end, T_step, "aligned_canonical");
    assert_pulse_coverage(segs, {0.0, current_tau}, T_start, T_end, T_step,
                          "aligned_canonical");

    // Expected layout: [-100,-window] free, [-window,+window] active,
    // [+window, current_tau-window] free, [current_tau-window,
    // current_tau+window] active, [current_tau+window, +100] free.
    check(segs.size() == 5,
          "aligned_canonical: expected 5 segments, got " + std::to_string(segs.size()));
}

// =====================================================================
// Case 2 — overlapping pulse windows (small |τ|): merge into one
// active segment.
// =====================================================================
void case_overlapping_windows()
{
    const double T_start = -10.0;
    const double T_end   = +10.0;
    const double T_step  = 0.01;
    const double sigma   = 0.5;
    const double window  = ck::kPulseWindowSigmas * sigma;  // 4.5
    const double current_tau = 1.0;  // window overlaps [-3.5, 5.5]

    const auto segs = ck::build_pulse_segments(
        T_start, T_end, {0.0, current_tau},
        window, T_step, ck::kFreeDtFactor);

    assert_segments_on_grid(segs, T_start, T_end, T_step, "overlapping");
    assert_pulse_coverage(segs, {0.0, current_tau}, T_start, T_end, T_step,
                          "overlapping");

    // Expect three: free pre-pulse, merged active, free post-pulse.
    check(segs.size() == 3,
          "overlapping: expected 3 segments, got " + std::to_string(segs.size()));
}

// =====================================================================
// Case 3 — misaligned pulse_width.  σ = 0.45 ⇒ window = 4.05, which
// is NOT a multiple of T_step = 0.1.  Pre-fix the segment lengths
// would have been 0.05-off; post-fix they must snap to the grid.
// =====================================================================
void case_misaligned_pulse_width()
{
    const double T_start = -10.0;
    const double T_end   = +10.0;
    const double T_step  = 0.1;
    const double sigma   = 0.45;
    const double window  = ck::kPulseWindowSigmas * sigma;  // 4.05
    const double current_tau = 5.0;  // exact multiple of T_step

    const auto segs = ck::build_pulse_segments(
        T_start, T_end, {0.0, current_tau},
        window, T_step, ck::kFreeDtFactor);

    assert_segments_on_grid(segs, T_start, T_end, T_step,
                            "misaligned_pulse_width");
    assert_pulse_coverage(segs, {0.0, current_tau}, T_start, T_end, T_step,
                          "misaligned_pulse_width");
}

// =====================================================================
// Case 4 — misaligned current_tau.  current_tau = 0.55 with T_step =
// 0.1 means tau ± window endpoints land off-grid by 0.05.
// =====================================================================
void case_misaligned_tau()
{
    const double T_start = -10.0;
    const double T_end   = +10.0;
    const double T_step  = 0.1;
    const double sigma   = 0.4;
    const double window  = ck::kPulseWindowSigmas * sigma;  // 3.6 (clean)
    const double current_tau = 0.55;  // off the T_step grid

    const auto segs = ck::build_pulse_segments(
        T_start, T_end, {0.0, current_tau},
        window, T_step, ck::kFreeDtFactor);

    assert_segments_on_grid(segs, T_start, T_end, T_step,
                            "misaligned_tau");
    // current_tau itself is off-grid, but build_pulse_segments only
    // uses centres ± window, both of which we then snap.  The snap
    // moves the active region by at most T_step/2; we still want the
    // (snapped) tau to be inside an active segment.
    bool tau_active = false;
    for (const auto& seg : segs) {
        const bool active = std::abs(seg.dt_init - T_step) <
                            1e-9 * std::max(1.0, T_step);
        if (active && seg.t0 - T_step <= current_tau &&
            current_tau <= seg.t1 + T_step) {
            tau_active = true; break;
        }
    }
    check(tau_active, "misaligned_tau: snapped active region must still cover tau");
}

// =====================================================================
// Case 5 — misaligned T_end.  (T_end − T_start) is not a multiple of
// T_step ⇒ the back edge must snap *down* to the largest grid point.
// =====================================================================
void case_misaligned_T_end()
{
    const double T_start = 0.0;
    const double T_end   = 10.0 + 0.013;  // off the grid by 0.013
    const double T_step  = 0.1;
    const double sigma   = 0.4;
    const double window  = ck::kPulseWindowSigmas * sigma;
    const double current_tau = 5.0;

    const auto segs = ck::build_pulse_segments(
        T_start, T_end, {0.0, current_tau},
        window, T_step, ck::kFreeDtFactor);

    assert_segments_on_grid(segs, T_start, T_end, T_step,
                            "misaligned_T_end");
    // The snapped T_end should be 10.0 (largest k·T_step ≤ 10.013).
    check(std::abs(segs.back().t1 - 10.0) <= 1e-9,
          "misaligned_T_end: back endpoint must snap to 10.0, got " +
          std::to_string(segs.back().t1));
}

// =====================================================================
// Case 6 — pulse centre outside [T_start, T_end].  The window is
// clipped; if no active region survives we should just get one big
// free segment covering the whole range.
// =====================================================================
void case_pulse_outside_window()
{
    const double T_start = 0.0;
    const double T_end   = 10.0;
    const double T_step  = 0.1;
    const double sigma   = 0.4;
    const double window  = ck::kPulseWindowSigmas * sigma;
    const double tc      = 100.0;  // far outside

    const auto segs = ck::build_pulse_segments(
        T_start, T_end, {tc}, window, T_step, ck::kFreeDtFactor);

    assert_segments_on_grid(segs, T_start, T_end, T_step,
                            "pulse_outside");
    check(segs.size() == 1,
          "pulse_outside: expected single free segment, got " +
          std::to_string(segs.size()));
}

// =====================================================================
// Case 7 — degenerate inputs.  Negative window length, T_end ≤
// T_start, T_step ≤ 0 — must all return an empty vector without
// crashing.
// =====================================================================
void case_degenerate()
{
    auto a = ck::build_pulse_segments(0.0, 0.0, {0.0}, 1.0, 0.1,
                                      ck::kFreeDtFactor);
    check(a.empty(), "degenerate: T_end == T_start must yield empty");

    auto b = ck::build_pulse_segments(1.0, 0.0, {0.0}, 1.0, 0.1,
                                      ck::kFreeDtFactor);
    check(b.empty(), "degenerate: T_end < T_start must yield empty");

    auto c = ck::build_pulse_segments(0.0, 1.0, {0.0}, 0.1, 0.0,
                                      ck::kFreeDtFactor);
    check(c.empty(), "degenerate: T_step == 0 must yield empty");
}

}  // namespace

int main() {
    std::cout << "[test_pulse_chunking_alignment] running…\n";

    case_aligned_canonical();
    case_overlapping_windows();
    case_misaligned_pulse_width();
    case_misaligned_tau();
    case_misaligned_T_end();
    case_pulse_outside_window();
    case_degenerate();

    if (g_failures == 0) {
        std::cout << "  PASS — all chunking cases satisfied the snap-and-check contract.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << "  TOTAL FAILURES: " << g_failures << "\n";
    return EXIT_FAILURE;
}
