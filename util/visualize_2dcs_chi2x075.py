#!/usr/bin/env python3
"""
Comprehensive visualization of 2DCS results for chi2x=0.75, W=0 sample.

Uses:
  - build/workflow/routeA_analysis/   (crosspeak_routeA.py output)
  - build/workflow/v3_grid_chi2x075_wline/v3_grid_chix_0p75_W_0/sample_0/
    (reader_TmFeO3.py precomputed npy files)

Omega convention: ALL axes displayed in LINEAR THz (nu = omega/(2*pi)).
  - routeA_analysis omega_*.npy: saved in ANGULAR THz (rad/ps) → divide by 2*pi
  - reader npy files: reconstruct from HDF5 metadata, also angular → divide by 2*pi
  - After fix in crosspeak_routeA.py new runs save LINEAR THz directly.
"""
from __future__ import annotations
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
RA_DIR  = Path("build/workflow/routeA_analysis")
S0_DIR  = Path("build/workflow/v3_grid_chi2x075_wline/v3_grid_chix_0p75_W_0/sample_0")
H5_FILE = S0_DIR / "pump_probe_spectroscopy.h5"
OUT_DIR = Path("build/workflow/routeA_analysis")

MEV_TO_THZ = 1.0 / 4.135667696   # 1 meV / h  in THz  (= 0.241799 THz/meV)

# ── reference frequencies (all in LINEAR THz) ─────────────────────────────────
QAFM  = 0.906    # qAFM magnon
QFM   = 0.378    # qFM magnon (DM-canted)
E12   = 0.500    # bare CEF gap E1→E2
E13   = 1.200    # bare CEF gap E1→E3
DF1   = 0.074    # dressed Tm f1  (chi2x mean-field renorm)
DF2   = 0.147    # dressed Tm f2


def load_hdf5_meta() -> tuple:
    """Return (times, taus, dt, dtau, t_cut_idx, n_t_pos, n_tau)."""
    with h5py.File(H5_FILE) as f:
        times = f["reference/times"][:]
        taus  = f["tau_scan/tau_values"][:]
        pw    = float(f["metadata"].attrs.get("pulse_width_SU2", 0.3))
    t_cut     = 5.0 * pw
    t_cut_idx = int(np.searchsorted(times, t_cut))
    n_t_pos   = len(times) - t_cut_idx
    n_tau     = len(taus)
    dt        = times[1] - times[0]
    dtau      = taus[1]  - taus[0]
    return times, taus, dt, dtau, t_cut_idx, n_t_pos, n_tau


def make_linear_axes(n_t_pos: int, n_tau: int,
                     dt: float, dtau: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear-THz omega axes that MATCH the reader_TmFeO3 and routeA angular axes.
    reader saves: omega_ang = fftshift(fftfreq(N, dt)) * 2*pi * MEV_TO_THZ [rad/ps]
    linear THz:   omega_lin = fftshift(fftfreq(N, dt)) * MEV_TO_THZ
    """
    ot   = np.fft.fftshift(np.fft.fftfreq(n_t_pos, dt))   * MEV_TO_THZ
    otau = np.fft.fftshift(np.fft.fftfreq(n_tau,   dtau)) * MEV_TO_THZ
    return ot, otau


def load_routeA(ot_lin: np.ndarray, otau_lin: np.ndarray) -> dict:
    """
    Load routeA spectra.  OLD files store omega in ANGULAR THz; new files
    (after the fix) store LINEAR THz.  We detect which by checking the Nyquist.
    """
    ot_ra   = np.load(RA_DIR / "omega_t.npy")
    otau_ra = np.load(RA_DIR / "omega_tau.npy")
    spec1   = np.load(RA_DIR / "spec_lambda1_loc.npy")
    spec2   = np.load(RA_DIR / "spec_lambda2_loc.npy")

    # Nyquist in LINEAR THz = ot_lin.max()
    nyq_lin = ot_lin.max()
    nyq_ra  = ot_ra.max()
    if nyq_ra > 3.0 * nyq_lin:          # angular axes (factor ≈ 2*pi ≈ 6.28)
        ot_ra   = ot_ra   / (2.0 * np.pi)
        otau_ra = otau_ra / (2.0 * np.pi)

    # After old-format flip the saved otau_ra was the UNFLIPPED array; after our
    # fix it is already the corrected (reversed) array.  Detect by checking whether
    # row 0 (= first row of spec) is at +Ny (corrected) or -Ny (old).
    # For old files: otau_ra[0] < 0 (=-Ny); corrected: otau_ra[0] > 0 (=+Ny).
    if otau_ra[0] < 0:
        # Old file: otau_ra is unflipped (−Ny…+Ny) but spec was flipped (row 0 = +Ny).
        # Reverse otau so that index i ↔ row i of spec.
        otau_ra = otau_ra[::-1]

    return dict(spec1=spec1, spec2=spec2, ot=ot_ra, otau=otau_ra)


def _idx(arr: np.ndarray, val: float) -> int:
    return int(np.argmin(np.abs(arr - val)))


def add_ref_lines(ax, ot, otau, show_cross=True):
    """Add horizontal/vertical reference lines at key frequencies."""
    kw = dict(lw=0.8, alpha=0.8)
    ax.axvline(QAFM, color="cyan",   ls="--", **kw, label=fr"$\nu_{{qAFM}}={QAFM}$ THz")
    ax.axvline(QFM,  color="lime",   ls=":",  **kw, label=fr"$\nu_{{qFM}}={QFM}$ THz")
    ax.axvline(E12,  color="yellow", ls="--", **kw, label=fr"$\nu_{{E12}}={E12}$ THz")
    ax.axhline(QAFM, color="cyan",   ls="--", **kw)
    ax.axhline(QFM,  color="lime",   ls=":",  **kw)
    ax.axhline(E12,  color="yellow", ls="--", **kw)
    ax.axhline(DF2,  color="magenta",ls="-.", **kw, label=fr"dressed $f_2={DF2}$ THz")
    ax.axvline(DF2,  color="magenta",ls="-.", **kw)
    if show_cross:
        ax.plot(E12, QAFM, "r+", ms=14, mew=2,
                label=f"Cross-peak target ({QAFM:.3f}, {E12:.3f}) THz")


def plot_2d_spectrum(ax, spec, ot, otau,
                     lim_t=(0.0, 1.5), lim_tau=(-1.5, 1.5),
                     title="", cmap="gnuplot2", gamma=0.4):
    """Plot 2D spectrum in the given omega window (linear THz)."""
    it_mask   = (ot   >= lim_t[0])   & (ot   <= lim_t[1])
    itau_mask = (otau >= lim_tau[0]) & (otau <= lim_tau[1])
    ot_w    = ot[it_mask]
    otau_w  = otau[itau_mask]
    data    = spec[np.ix_(itau_mask, it_mask)]
    vmax    = np.percentile(data, 99.5)
    im = ax.imshow(
        data, origin="lower", aspect="auto", cmap=cmap,
        norm=PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax),
        extent=[ot_w[0], ot_w[-1], otau_w[0], otau_w[-1]],
    )
    plt.colorbar(im, ax=ax, label="Amplitude (a.u.)")
    add_ref_lines(ax, ot_w, otau_w)
    ax.set_xlabel(r"$\nu_t$ (THz linear)")
    ax.set_ylabel(r"$\nu_\tau$ (THz linear)")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="upper right")
    return im


def plot_1d_spectrum(ax, spec, ot, otau,
                     label="", color="steelblue", lw=1.2):
    """
    Plot 1D (omega_t) spectrum: sum |spec| over positive omega_tau.
    Both axes in linear THz.
    """
    pos_tau = otau >= 0
    s1d     = spec[pos_tau, :].mean(axis=0)
    ax.plot(ot, s1d, color=color, lw=lw, label=label)
    ax.axvline(QAFM, color="cyan",   ls="--", lw=0.7, alpha=0.7)
    ax.axvline(QFM,  color="lime",   ls=":",  lw=0.7, alpha=0.7)
    ax.axvline(E12,  color="yellow", ls="--", lw=0.7, alpha=0.7)
    ax.axvline(DF2,  color="magenta",ls="-.", lw=0.7, alpha=0.7)
    ax.set_xlabel(r"$\nu_t$ (THz linear)")
    ax.set_ylabel("Amplitude (mean over +τ)")


def tau_modulation_plot(ax, NL_path, times, taus):
    """
    Plot the amplitude of the dressed-f2 oscillation as a function of tau.
    Shows whether there is qAFM modulation (genuine cross-peak) or pure DC-in-tau.
    """
    NL    = np.load(NL_path)  # (n_tau, n_t_pos)
    n_tau, n_t_pos = NL.shape
    dt    = times[1] - times[0]

    ot    = np.fft.fftfreq(n_t_pos, dt) * MEV_TO_THZ   # un-shifted, linear THz
    i_df2 = int(np.argmin(np.abs(ot - DF2)))
    i_qafm = int(np.argmin(np.abs(ot - QAFM)))

    # Per-tau amplitude of the t-FFT at dressed_f2
    ft_rows = np.fft.fft(NL, axis=1)   # shape (n_tau, n_t_pos)
    amp_df2  = np.abs(ft_rows[:, i_df2])
    amp_qafm = np.abs(ft_rows[:, i_qafm])

    taus_ps = taus * 0.6582    # code_units → ps
    ax.semilogy(taus_ps, amp_df2,  color="magenta", lw=0.8, label=fr"$|\hat{{M}}(τ,\nu_{{df2}}={DF2}\ \mathrm{{THz}})|$")
    ax.semilogy(taus_ps, amp_qafm, color="cyan",    lw=0.8, alpha=0.7, label=fr"$|\hat{{M}}(τ,\nu_{{qAFM}}={QAFM}\ \mathrm{{THz}})|$")
    ax.set_xlabel(r"$\tau$ (ps)")
    ax.set_ylabel("Per-$\\tau$ t-FFT amplitude")
    ax.set_title(r"$\tau$-dependence of dressed-$f_2$ oscillation (Route A NL $\lambda^1_{loc}$)")
    ax.legend(fontsize=7)
    ax.set_xlim([taus_ps.min(), taus_ps.max()])


def snr_table(ra: dict, ot_lin: np.ndarray, otau_lin: np.ndarray,
              reader_spec: dict) -> None:
    """Print S/N table at key positions using correct linear THz indices."""
    ot   = ra["ot"];    otau = ra["otau"]
    spec1 = ra["spec1"]; spec2 = ra["spec2"]

    def lookup(spec, ot_arr, otau_arr, nu_tau, nu_t):
        i_tau = _idx(otau_arr, nu_tau)
        i_t   = _idx(ot_arr, nu_t)
        return spec[i_tau, i_t]

    print("\n=== S/N table (routeA lambda^1_loc, correct LINEAR THz axes) ===")
    print(f"  rms = {spec1.std():.3e}")
    rms = spec1.std()
    for name, nu_tau, nu_t in [
        ("(DC,    DF2)",   0.0,  DF2),
        ("(DC,    QAFM)",  0.0,  QAFM),
        ("(DC,    E12)",   0.0,  E12),
        ("(QAFM,  DF2)",   QAFM, DF2),
        ("(QAFM,  E12)",   QAFM, E12),
        ("(QAFM,  QAFM)",  QAFM, QAFM),
        ("(DF2,   DF2)",   DF2,  DF2),
    ]:
        v = lookup(spec1, ot, otau, nu_tau, nu_t)
        print(f"  {name:<20}  {v:.3e}  S/N={v/rms:.2f}")

    print("\n=== S/N table (reader M_NL_SU3_FF_λ1, global frame) ===")
    tm1 = reader_spec["λ1"]
    rms_tm = tm1.std()
    if rms_tm > 0:
        print(f"  rms = {rms_tm:.3e}")
        for name, nu_tau, nu_t in [
            ("(DC,    DF1)",   0.0,  DF1),
            ("(DC,    DF2)",   0.0,  DF2),
            ("(QAFM,  DF1)",   QAFM, DF1),
            ("(QAFM,  DF2)",   QAFM, DF2),
            ("(QAFM,  E12)",   QAFM, E12),
        ]:
            v = lookup(tm1, ot_lin, otau_lin, nu_tau, nu_t)
            print(f"  {name:<20}  {v:.3e}  S/N={v/rms_tm:.2f}")


def main():
    print("Loading HDF5 metadata …")
    times, taus, dt, dtau, t0_idx, n_t_pos, n_tau = load_hdf5_meta()
    times_pos = times[t0_idx:]
    ot_lin, otau_lin = make_linear_axes(n_t_pos, n_tau, dt, dtau)
    print(f"  omega_t  linear THz: [{ot_lin.min():.3f}, {ot_lin.max():.3f}]  Nyquist={ot_lin.max():.3f}")
    print(f"  omega_tau linear THz: [{otau_lin.min():.3f}, {otau_lin.max():.3f}]")

    print("\nLoading routeA spectra …")
    ra = load_routeA(ot_lin, otau_lin)
    print(f"  spec1 shape={ra['spec1'].shape}  otau range=[{ra['otau'].min():.3f},{ra['otau'].max():.3f}] THz")

    print("\nLoading reader precomputed spectra …")
    reader = {}
    for lam in range(1, 9):
        fn = S0_DIR / f"M_NL_SU3_FF_λ{lam}.npy"
        arr = np.load(fn)
        reader[f"λ{lam}"] = arr
    su2_y = np.load(S0_DIR / "M_NL_SU2_FF_y.npy")
    su2_x = np.load(S0_DIR / "M_NL_SU2_FF_x.npy")

    snr_table(ra, ot_lin, otau_lin, reader)

    # ── Figure 1: 2D spectra side-by-side ─────────────────────────────────────
    print("\nPlotting Figure 1: 2D spectra …")
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle(
        r"TmFeO$_3$  2DCS Route A — $\chi_{2x}=0.75$ meV, $W=0$  "
        r"(axes in linear THz = $\nu = \omega/2\pi$)"
        "\n"
        r"Dominant feature: dressed-Tm $f_2=0.147$ THz DC-in-$\tau$ (NOT qAFM cross-peak)",
        fontsize=10,
    )

    lim_t   = (0.0, 1.5)
    lim_tau = (-1.5, 1.5)

    plot_2d_spectrum(axes[0], ra["spec1"], ra["ot"], ra["otau"],
                     lim_t=lim_t, lim_tau=lim_tau,
                     title=r"Route A: $\lambda^1_{loc}$ = M_local_SU3[:,0]  (NL signal)")

    plot_2d_spectrum(axes[1], ra["spec2"], ra["ot"], ra["otau"],
                     lim_t=lim_t, lim_tau=lim_tau,
                     title=r"Route A: $\lambda^2_{loc}$ = M_local_SU3[:,1]")

    plot_2d_spectrum(axes[2], reader["λ1"], ot_lin, otau_lin,
                     lim_t=lim_t, lim_tau=lim_tau,
                     title=r"Reader: $\lambda^1$ (global frame, M_NL_SU3_FF)")

    fig1.tight_layout()
    out1 = OUT_DIR / "fig1_2d_spectra.pdf"
    fig1.savefig(out1, dpi=150)
    print(f"  Saved {out1}")
    out1p = OUT_DIR / "fig1_2d_spectra.png"
    fig1.savefig(out1p, dpi=150)
    print(f"  Saved {out1p}")

    # ── Figure 2: 1D projections + tau-modulation analysis ────────────────────
    print("\nPlotting Figure 2: 1D spectra + tau-modulation …")
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
    fig2.suptitle(
        r"TmFeO$_3$  2DCS  — Diagnostic plots ($\chi_{2x}=0.75$ meV, $W=0$)",
        fontsize=11,
    )

    # (0,0) 1D t-spectrum for several channels
    ax = axes2[0, 0]
    ax.set_title(r"1D $\nu_t$ projection (mean over $\nu_\tau > 0$)")
    pos_tau_mask = ra["otau"] >= 0
    s1d_l1 = ra["spec1"][pos_tau_mask, :].mean(axis=0)
    s1d_l2 = ra["spec2"][pos_tau_mask, :].mean(axis=0)
    pos_tau_lin = otau_lin >= 0
    s1d_reader1 = reader["λ1"][pos_tau_lin, :].mean(axis=0)
    for s, lbl, col in [(s1d_l1, r"$\lambda^1_{loc}$ (routeA)", "steelblue"),
                         (s1d_l2, r"$\lambda^2_{loc}$ (routeA)", "darkorange"),
                         (s1d_reader1, r"$\lambda^1$ global (reader)", "green")]:
        norm = s.max() if s.max() > 0 else 1.0
        ax.plot(ra["ot"], s / norm, lw=1.0, label=lbl, alpha=0.85)
    for nu, col, lbl in [(QAFM,"cyan",r"$\nu_{qAFM}$"),(QFM,"lime",r"$\nu_{qFM}$"),
                          (E12,"yellow",r"$\nu_{E12}$"),(DF2,"magenta",r"dressed $f_2$")]:
        ax.axvline(nu, color=col, lw=0.7, ls="--", alpha=0.8)
    ax.set_xlim([-0.05, 1.5]); ax.set_xlabel(r"$\nu_t$ (THz)"); ax.legend(fontsize=7)
    ax.set_ylabel("Norm. amplitude")

    # (0,1) Column at omega_t = DF2 vs omega_tau (shows DC vs qAFM)
    ax = axes2[0, 1]
    ax.set_title(r"Column at $\nu_t=0.147$ THz (dressed $f_2$) vs $\nu_\tau$")
    i_df2_ra  = _idx(ra["ot"],   DF2)
    i_df2_lin = _idx(ot_lin,     DF2)
    col_l1    = ra["spec1"][:, i_df2_ra]
    col_rd1   = reader["λ1"][:, i_df2_lin]
    ax.semilogy(ra["otau"], col_l1  / (col_l1.max()  or 1), lw=1.0, color="steelblue",
                label=r"$\lambda^1_{loc}$ (routeA)")
    ax.semilogy(otau_lin,   col_rd1 / (col_rd1.max() or 1), lw=1.0, color="green",
                label=r"$\lambda^1$ global (reader)")
    for nu, col, lbl in [(QAFM,"cyan","qAFM"),(QFM,"lime","qFM"),(DF2,"magenta","df2")]:
        ax.axvline(nu, color=col, lw=0.7, ls="--", alpha=0.8, label=lbl)
    ax.set_xlabel(r"$\nu_\tau$ (THz)"); ax.set_ylabel("Norm. amplitude (log)")
    ax.set_xlim([-1.5, 1.5])
    ax.legend(fontsize=7)

    # (1,0) tau-modulation of dressed_f2 oscillation
    ax = axes2[1, 0]
    tau_modulation_plot(ax, RA_DIR / "NL_lambda1_loc.npy", times_pos, taus)

    # (1,1) S/N comparison at key positions (bar chart)
    ax = axes2[1, 1]
    ax.set_title("S/N at key (qAFM, *) positions")
    positions = [("(qAFM, DF2)", QAFM, DF2),
                 ("(qAFM, E12)", QAFM, E12),
                 ("(DC,   DF2)", 0.0,  DF2),
                 ("(DC,   E12)", 0.0,  E12)]
    specs_all = {
        r"$\lambda^1_{loc}$": (ra["spec1"], ra["ot"], ra["otau"]),
        r"$\lambda^2_{loc}$": (ra["spec2"], ra["ot"], ra["otau"]),
        r"$\lambda^1$ global": (reader["λ1"], ot_lin, otau_lin),
    }
    x = np.arange(len(positions))
    width = 0.25
    colors = ["steelblue", "darkorange", "green"]
    for i, (lbl, (spec, ot_, otau_)) in enumerate(specs_all.items()):
        rms = spec.std()
        vals = [spec[_idx(otau_, nu_tau), _idx(ot_, nu_t)] / (rms or 1e-30)
                for _, nu_tau, nu_t in positions]
        ax.bar(x + (i - 1) * width, vals, width, label=lbl, color=colors[i], alpha=0.8)
    ax.axhline(1.0, color="red", ls="--", lw=0.8, label="S/N = 1")
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in positions], fontsize=8)
    ax.set_ylabel("S/N (val / rms)"); ax.legend(fontsize=7)
    ax.set_title("S/N at key positions")

    fig2.tight_layout()
    out2 = OUT_DIR / "fig2_diagnostics.pdf"
    fig2.savefig(out2, dpi=150)
    print(f"  Saved {out2}")
    out2p = OUT_DIR / "fig2_diagnostics.png"
    fig2.savefig(out2p, dpi=150)
    print(f"  Saved {out2p}")

    # ── Figure 3: Full-range 2D view (both quadrants) ─────────────────────────
    print("\nPlotting Figure 3: Full-range 2D spectrum …")
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle(
        r"Full-range 2D NL spectrum  ($\chi_{2x}=0.75$ meV, $W=0$)  —  linear THz axes",
        fontsize=11,
    )
    lim_full_t = (-0.5, 2.4); lim_full_tau = (-2.4, 2.4)
    plot_2d_spectrum(axes3[0], ra["spec1"], ra["ot"], ra["otau"],
                     lim_t=lim_full_t, lim_tau=lim_full_tau,
                     title=r"Route A $\lambda^1_{loc}$ (full range)")
    plot_2d_spectrum(axes3[1], reader["λ1"], ot_lin, otau_lin,
                     lim_t=lim_full_t, lim_tau=lim_full_tau,
                     title=r"Reader $\lambda^1$ global (full range)")
    fig3.tight_layout()
    out3 = OUT_DIR / "fig3_full_range.pdf"
    fig3.savefig(out3, dpi=150)
    print(f"  Saved {out3}")
    out3p = OUT_DIR / "fig3_full_range.png"
    fig3.savefig(out3p, dpi=150)
    print(f"  Saved {out3p}")

    print("\nDone.  Figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
