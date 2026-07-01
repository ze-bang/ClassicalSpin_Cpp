#!/usr/bin/env python3
"""
Route A cross-peak analysis: chi_{2x} S^x_Fe · lambda^2_Tm

Reads the standard 2DCS HDF5 and extracts the nonlinear signal in two channels:
  Lambda^1_loc:  M_local_SU3 lambda^1  (= mean of local-frame lambda^1 over 4 sites)
                 ≡ (1/4) × Lambda^1_G  (where Lambda^1_G is the sigma_G-weighted
                 global-frame lambda^1; proven nonzero when chi2x drives F-mode local
                 oscillations that map to G-mode in global frame).
  Lambda^2_loc:  M_local_SU3 lambda^2  (reference: direct chi2x drive channel)

The cross-peak at (omega_tau, omega_t) = (omega_qAFM, omega_E12) appears in
Lambda^1_loc because:
  1. chi2x drives local lambda^2 uniformly (sigma_F pattern)
  2. SU(3) EOM propagates lambda^2 -> lambda^1 in the same sigma_F (local) pattern
  3. In the global frame this becomes sigma_G (+1,-1,-1,+1): Lambda^1_G is nonzero
  4. The qAFM magnon modulates chi2x effective field during tau -> cross-coherence

Usage:
    python3 util/crosspeak_routeA.py [hdf5_dir] [--output-dir DIR]

Default hdf5_dir:
    build/workflow/v3_grid_chi2x075_wline/v3_grid_chix_0p75_W_0/sample_0
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# meV -> THz conversion (ℏ = 1 unit system; 1 meV ~ 0.2418 THz)
MEV_TO_THZ = 0.241798920

# Expected physics (recalibrated Gamma2 params)
OMEGA_QAFM_THZ = 0.906    # qAFM magnon
OMEGA_QFM_THZ  = 0.378    # qFM magnon (DM-canted)
OMEGA_E12_THZ  = 0.500    # E1-E2 CEF gap (e1 = 2.067834 meV)


def build_apodization(t_pos: np.ndarray, taus: np.ndarray,
                      window: str = "hann",
                      gamma: float = 0.03) -> np.ndarray:
    """
    Apodization window on the (tau, t) grid.

    window='hann'     (default) Hann (raised-cosine) window on both axes.
                      Better for detecting high-frequency tau modulations
                      (e.g. qAFM cross-peak) because it does not cause the
                      DC-sidelobe leakage that the Gaussian window does when
                      sigma_tau * omega_qAFM >> 1.
    window='gaussian' Gaussian with value `gamma` at the boundary (the original
                      reader convention).  NOTE: with the default tau range of
                      100 code-units, sigma_tau * omega_qAFM ~ 141 >> 1, which
                      destroys the qAFM cross-peak via sidelobe leakage.
    window='none'     No apodization (rectangular window, only mean-subtract).
                      Best for maximum sensitivity; may show ringing artefacts.
    """
    n_tau = len(taus)
    n_t   = len(t_pos)
    if window == "hann":
        w_tau = np.hanning(n_tau)
        w_t   = np.hanning(n_t)
    elif window == "gaussian":
        t_rel   = t_pos - t_pos[0]
        tau_abs = np.abs(taus)
        t_range   = t_rel[-1]   if n_t   > 1 else 1.0
        tau_range = tau_abs[-1] if n_tau > 1 else 1.0
        k       = np.sqrt(-2.0 * np.log(gamma))
        w_t   = np.exp(-0.5 * (t_rel   / (t_range   / k)) ** 2)
        w_tau = np.exp(-0.5 * (tau_abs / (tau_range  / k)) ** 2)
    elif window == "none":
        w_tau = np.ones(n_tau)
        w_t   = np.ones(n_t)
    else:
        raise ValueError(f"Unknown window '{window}'. Use 'hann', 'gaussian', or 'none'.")
    return np.outer(w_tau, w_t)   # (n_tau, n_t_pos)


OMEGA_DF2_THZ  = 0.147    # dressed Tm f2 (chi2x=0.75 renormalized; = 0.608 meV / h)
OMEGA_DF1_THZ  = 0.074    # dressed Tm f1


def run_analysis(h5_dir: Path, output_dir: Path,
                 t_cut_ps: float = 1.5,
                 omega_t_lim:   tuple = (0.0, 2.5),
                 omega_tau_lim: tuple = (-2.5, 2.5),
                 window: str = "hann") -> None:

    h5_path = h5_dir / "pump_probe_spectroscopy.h5"
    if not h5_path.exists():
        sys.exit(f"HDF5 not found: {h5_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        meta        = f["metadata"].attrs
        pulse_width = float(meta.get("pulse_width_SU2", 0.3))

        times = f["reference/times"][:]
        taus  = f["tau_scan/tau_values"][:]
        n_tau = len(taus)
        n_t   = len(times)

        print(f"  tau: [{taus[0]:.2f}, {taus[-1]:.2f}]  step={taus[1]-taus[0]:.4f}  N={n_tau}")
        print(f"  t:   [{times[0]:.2f}, {times[-1]:.2f}]  step={times[1]-times[0]:.4f}  N={n_t}")

        # Time cutoff to skip probe pulse region
        t0_idx    = int(np.searchsorted(times, t_cut_ps))
        times_pos = times[t0_idx:]
        n_t_pos   = len(times_pos)
        print(f"  t_cut={t_cut_ps:.2f} ps -> t0_idx={t0_idx}, n_t_pos={n_t_pos}")

        # ── reference M0 (lambda^1 and lambda^2 from M_local_SU3) ──────────
        M0_loc = f["reference/M_local_SU3"][:]    # (n_t, 8)
        M0_l1  = M0_loc[t0_idx:, 0]              # (n_t_pos,)  lambda^1
        M0_l2  = M0_loc[t0_idx:, 1]              # (n_t_pos,)  lambda^2

        print(f"\n  M0 lambda^1 range: [{M0_l1.min():.4e}, {M0_l1.max():.4e}]")
        print(f"  M0 lambda^2 range: [{M0_l2.min():.4e}, {M0_l2.max():.4e}]")

        # ── accumulate NL signal ─────────────────────────────────────────────
        print(f"\nAccumulating NL signal ({n_tau} tau points) ...")
        NL_l1 = np.zeros((n_tau, n_t_pos), dtype=float)
        NL_l2 = np.zeros((n_tau, n_t_pos), dtype=float)

        for i_tau in range(n_tau):
            key = f"tau_scan/tau_{i_tau}"
            M01_loc = f[f"{key}/M01_local_SU3"][t0_idx:, :]   # (n_t_pos, 8)
            M1_loc  = f[f"{key}/M1_local_SU3"][t0_idx:, :]
            NL_l1[i_tau] = M01_loc[:, 0] - M0_l1 - M1_loc[:, 0]
            NL_l2[i_tau] = M01_loc[:, 1] - M0_l2 - M1_loc[:, 1]
            if i_tau % 400 == 0:
                rms1 = np.std(NL_l1[i_tau])
                rms2 = np.std(NL_l2[i_tau])
                print(f"  tau {i_tau:4d}/{n_tau}  rms(NL_l1)={rms1:.3e}  rms(NL_l2)={rms2:.3e}")

    print(f"\nNL_l1 rms over all tau: {np.std(NL_l1):.3e}")
    print(f"NL_l2 rms over all tau: {np.std(NL_l2):.3e}")

    np.save(output_dir / "NL_lambda1_loc.npy", NL_l1)
    np.save(output_dir / "NL_lambda2_loc.npy", NL_l2)
    np.save(output_dir / "times_pos.npy",       times_pos)
    np.save(output_dir / "taus.npy",            taus)

    # ── apodization ───────────────────────────────────────────────────────────
    dt_tau = taus[1] - taus[0]
    dt_t   = times_pos[1] - times_pos[0]
    n_t_pos = len(times_pos)

    print(f"\nApodization window: '{window}'")
    if window == "gaussian":
        print("  WARNING: Gaussian window with gamma=0.03 has sigma_tau*omega_qAFM >> 1")
        print("  → destroys qAFM cross-peak. Use 'hann' for better cross-peak detection.")
    apod = build_apodization(times_pos, taus, window=window)

    # mean-subtract per tau slice (removes DC offset) then apodize
    def prep_signal(arr):
        arr_dm = arr - arr.mean(axis=(-2, -1), keepdims=True)
        return arr_dm * apod

    NL_l1_apod = prep_signal(NL_l1)
    NL_l2_apod = prep_signal(NL_l2)

    # ── 2D FFT ────────────────────────────────────────────────────────────────
    def do_fft2(data):
        spec = np.fft.fftshift(np.fft.fft2(data))
        # Rephasing convention: flip tau axis (tau → -tau sign).
        spec = spec[::-1, :]
        # omega axes in LINEAR THz (not angular).  After the row-flip, row i of
        # spec corresponds to omega_tau = otau_unflipped[n_tau-1-i], so we save
        # the reversed array so that index i → omega_tau[i] stays consistent.
        ot              = np.fft.fftshift(np.fft.fftfreq(n_t_pos, dt_t))   * MEV_TO_THZ
        otau_unflipped  = np.fft.fftshift(np.fft.fftfreq(n_tau,   dt_tau)) * MEV_TO_THZ
        otau            = otau_unflipped[::-1]   # now row 0 ↔ +Nyquist, row N-1 ↔ -Nyquist
        return np.abs(spec), otau, ot

    mag1, omega_tau, omega_t = do_fft2(NL_l1_apod)
    mag2, _,         _       = do_fft2(NL_l2_apod)

    np.save(output_dir / "spec_lambda1_loc.npy", mag1)
    np.save(output_dir / "spec_lambda2_loc.npy", mag2)
    np.save(output_dir / "omega_tau.npy",         omega_tau)
    np.save(output_dir / "omega_t.npy",           omega_t)

    # ── peak search in positive quadrant ──────────────────────────────────────
    tau_pos  = omega_tau >= 0
    t_pos    = omega_t  >= 0
    sub1     = mag1[np.ix_(tau_pos, t_pos)]
    ot_sub   = omega_t[t_pos]
    otau_sub = omega_tau[tau_pos]

    i_cp_tau   = np.argmin(np.abs(otau_sub - OMEGA_QAFM_THZ))
    i_cp_t     = np.argmin(np.abs(ot_sub   - OMEGA_E12_THZ))
    i_cp_t_df2 = np.argmin(np.abs(ot_sub   - OMEGA_DF2_THZ))
    i_dc_tau   = np.argmin(np.abs(otau_sub - 0.0))
    cp_val     = sub1[i_cp_tau, i_cp_t]
    cp_df2_val = sub1[i_cp_tau, i_cp_t_df2]
    dc_df2_val = sub1[i_dc_tau, i_cp_t_df2]

    flat_idx           = np.argmax(sub1)
    itau2d, it2d       = np.unravel_index(flat_idx, sub1.shape)
    peak_tau, peak_t   = otau_sub[itau2d], ot_sub[it2d]
    peak_val           = sub1[itau2d, it2d]

    rms1 = mag1.std()
    print(f"\n=== Lambda^1_loc (M_local_SU3) 2D NL spectrum  [window='{window}'] ===")
    print(f"  Global max: (omega_tau={peak_tau:.3f}, omega_t={peak_t:.3f}) THz  amp={peak_val:.4e}")
    print(f"  rms={rms1:.3e}")
    print(f"  (DC,   dressed_f2={OMEGA_DF2_THZ:.3f}): amp={dc_df2_val:.4e}  S/N={dc_df2_val/rms1:.1f}")
    print(f"  (qAFM, dressed_f2={OMEGA_DF2_THZ:.3f}): amp={cp_df2_val:.4e}  S/N={cp_df2_val/rms1:.1f}")
    print(f"  (qAFM, bare_E12={OMEGA_E12_THZ:.3f}):   amp={cp_val:.4e}      S/N={cp_val/rms1:.1f}")
    print(f"  rms(NL_l1)={np.std(NL_l1):.3e}  rms(NL_l2)={np.std(NL_l2):.3e}")
    print(f"  rms ratio NL_l2/NL_l1 = {np.std(NL_l2)/np.std(NL_l1):.3f}")

    # Top 5 peaks
    print("\n  Top 5 peaks in positive quadrant:")
    flat = sub1.ravel()
    top5 = np.argsort(flat)[-5:][::-1]
    for idx in top5:
        r, c = np.unravel_index(idx, sub1.shape)
        print(f"    (omega_tau={otau_sub[r]:.3f}, omega_t={ot_sub[c]:.3f}) THz  amp={flat[idx]:.4e}")

    # ── plot ─────────────────────────────────────────────────────────────────
    def _window_mask(ot, otau, t_lim, tau_lim):
        tm   = (ot   >= t_lim[0])   & (ot   <= t_lim[1])
        taum = (otau >= tau_lim[0]) & (otau <= tau_lim[1])
        return np.ix_(taum, tm)

    mask = _window_mask(omega_t, omega_tau, omega_t_lim, omega_tau_lim)
    ot_w   = omega_t[mask[1][0]]
    otau_w = omega_tau[mask[0][:, 0]]

    def _plot_spec(ax, data, title, cmap="gnuplot2"):
        d    = data[mask]
        vmax = np.percentile(d, 99.5)
        im   = ax.imshow(
            d, origin="lower", aspect="auto", cmap=cmap,
            norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax),
            extent=[ot_w[0], ot_w[-1], otau_w[0], otau_w[-1]],
        )
        plt.colorbar(im, ax=ax, label="Amplitude")
        ax.axhline(OMEGA_QAFM_THZ, color="cyan",   lw=0.8, ls="--",
                   label=fr"$\omega_{{qAFM}}$={OMEGA_QAFM_THZ} THz")
        ax.axhline(OMEGA_QFM_THZ,  color="lime",   lw=0.8, ls=":")
        ax.axvline(OMEGA_E12_THZ,  color="yellow", lw=0.8, ls="--",
                   label=fr"$\omega_{{E12}}$={OMEGA_E12_THZ} THz")
        ax.axvline(OMEGA_QAFM_THZ, color="cyan",   lw=0.8, ls="--")
        ax.plot(OMEGA_E12_THZ, OMEGA_QAFM_THZ, "r+", ms=12, mew=2,
                label=f"Cross-peak ({OMEGA_QAFM_THZ:.2f}, {OMEGA_E12_THZ:.2f}) THz")
        ax.set_xlabel(r"$\omega_t$ (THz)")
        ax.set_ylabel(r"$\omega_\tau$ (THz)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        r"Route A: $\chi_{2x}=0.75$, W=0 — Lambda^1 and Lambda^2 (M_local_SU3 channel)"
        "\n"
        r"Lambda^1_loc $\equiv\frac{1}{4}\Lambda^1_G$ (sigma_G-weighted global-frame $\lambda^1$)",
        fontsize=10,
    )
    _plot_spec(axes[0], mag1, r"$\Lambda^1_{\rm loc}$ (lambda^1, local-frame mean)")
    _plot_spec(axes[1], mag2, r"$\Lambda^2_{\rm loc}$ (lambda^2, chi2x drive channel)")
    plt.tight_layout()
    out_pdf = output_dir / "routeA_crosspeak_2dcs.pdf"
    plt.savefig(out_pdf, dpi=150)
    print(f"\nPlot saved: {out_pdf}")

    # ── 1D cuts ───────────────────────────────────────────────────────────────
    def nearest(arr, val):
        return int(np.argmin(np.abs(arr - val)))

    i_qAFM   = nearest(omega_tau, OMEGA_QAFM_THZ)
    i_qFM    = nearest(omega_tau, OMEGA_QFM_THZ)
    i_E12    = nearest(omega_t,   OMEGA_E12_THZ)
    i_qAFM_t = nearest(omega_t,   OMEGA_QAFM_THZ)

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle(r"Route A 1D cuts: Lambda^1_loc and Lambda^2_loc", fontsize=10)

    def _cut(ax, x, y1, y2, xlabel, title, xlines=(), xlim=None):
        ax.plot(x, y1, label=r"$\Lambda^1_{\rm loc}$")
        ax.plot(x, y2, label=r"$\Lambda^2_{\rm loc}$", alpha=0.6)
        for xv, col, lab in xlines:
            ax.axvline(xv, color=col, ls="--", lw=0.9, label=lab)
        ax.set_xlabel(xlabel);  ax.set_title(title)
        if xlim:
            ax.set_xlim(xlim)
        ax.legend(fontsize=8)

    _cut(axes2[0, 0], omega_t, mag1[i_qAFM, :], mag2[i_qAFM, :],
         r"$\omega_t$ (THz)",
         fr"Cut at $\omega_\tau$={omega_tau[i_qAFM]:.3f} THz (≈ω_qAFM)",
         [(OMEGA_E12_THZ,  "red",   f"E12={OMEGA_E12_THZ} THz"),
          (OMEGA_QAFM_THZ, "blue",  f"qAFM={OMEGA_QAFM_THZ} THz")],
         xlim=omega_t_lim)

    _cut(axes2[0, 1], omega_t, mag1[i_qFM, :], mag2[i_qFM, :],
         r"$\omega_t$ (THz)",
         fr"Cut at $\omega_\tau$={omega_tau[i_qFM]:.3f} THz (≈ω_qFM)",
         [(OMEGA_E12_THZ,  "red",  ""),
          (OMEGA_QAFM_THZ, "blue", "")],
         xlim=omega_t_lim)

    _cut(axes2[1, 0], omega_tau, mag1[:, i_E12], mag2[:, i_E12],
         r"$\omega_\tau$ (THz)",
         fr"Cut at $\omega_t$={omega_t[i_E12]:.3f} THz (≈ω_E12)",
         [(OMEGA_QAFM_THZ, "blue",  f"qAFM={OMEGA_QAFM_THZ} THz"),
          (OMEGA_QFM_THZ,  "green", f"qFM={OMEGA_QFM_THZ} THz")],
         xlim=omega_tau_lim)

    _cut(axes2[1, 1], omega_tau, mag1[:, i_qAFM_t], mag2[:, i_qAFM_t],
         r"$\omega_\tau$ (THz)",
         fr"Cut at $\omega_t$={omega_t[i_qAFM_t]:.3f} THz (diagonal qAFM)",
         [(OMEGA_QAFM_THZ, "blue", "")],
         xlim=omega_tau_lim)

    plt.tight_layout()
    out_pdf2 = output_dir / "routeA_crosspeak_cuts.pdf"
    plt.savefig(out_pdf2, dpi=150)
    print(f"Cuts plot saved: {out_pdf2}")
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "hdf5_dir", nargs="?",
        default="build/workflow/v3_grid_chi2x075_wline/v3_grid_chix_0p75_W_0/sample_0",
        help="Directory containing pump_probe_spectroscopy.h5 (standard, non-raw-state run)"
    )
    parser.add_argument(
        "--output-dir", default="build/workflow/routeA_analysis",
        help="Output directory for plots and NPY files"
    )
    parser.add_argument(
        "--omega-t",   nargs=2, type=float, default=[0.0, 2.5],
        metavar=("MIN", "MAX"), help="omega_t display range (THz)"
    )
    parser.add_argument(
        "--omega-tau", nargs=2, type=float, default=[-2.5, 2.5],
        metavar=("MIN", "MAX"), help="omega_tau display range (THz)"
    )
    parser.add_argument(
        "--t-cut", type=float, default=1.5,
        help="Time cutoff in ps (skip probe pulse region, default=1.5)"
    )
    parser.add_argument(
        "--window", choices=["hann", "gaussian", "none"], default="hann",
        help="Apodization window for tau and t axes (default=hann).\n"
             "'hann': Hann window — best for high-freq tau modulations (qAFM cross-peak).\n"
             "'gaussian': Gaussian(gamma=0.03) — matches reader convention but destroys qAFM.\n"
             "'none': no window (rectangular) — max sensitivity, may have ringing."
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    h5_dir    = (repo_root / args.hdf5_dir).resolve()
    out_dir   = (repo_root / args.output_dir).resolve()

    run_analysis(
        h5_dir, out_dir,
        t_cut_ps      = args.t_cut,
        omega_t_lim   = tuple(args.omega_t),
        omega_tau_lim = tuple(args.omega_tau),
        window        = args.window,
    )


if __name__ == "__main__":
    main()
