#!/usr/bin/env python3
"""
DM mode-mixing comparison:
  Compare hx and hz pumps with D1=0 vs D1=0.05 to isolate qFM/qAFM modes
  and show DM-induced mixing.

Layout (4 rows x 2 cols):
  Row 0: no DM, hx pump    — expect single Gx peak (qFM only)
  Row 1: no DM, hz pump    — expect zero response (parallel to G)
  Row 2: with DM, hx pump  — expect Gx (qFM) + Gy (qAFM) both present
  Row 3: with DM, hz pump  — expect tiny Gy (qAFM via cant)

Columns: [Gx FFT, Gy FFT]
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

HBAR_MEV_PS = 0.6582119514   # ħ in meV·ps; code time unit = ħ/meV

def load(path):
    d = np.loadtxt(path)
    t = d[:, 0]
    G = d[:, 1:4]   # Gx, Gy, Gz
    # L = d[:, 4:7]
    # F = d[:, 7:10]
    return t, G

def fft(t, sig, t_start=5.0):
    """Return (freq_THz, power) with Hanning window, signal mean-subtracted.
    Time axis in code units (ħ/meV = 0.6582 ps).
    f[THz] = f_code [1/(ħ/meV)] / HBAR_MEV_PS [ps/(ħ/meV)] ... same as:
    f[THz] = f_code / HBAR_MEV_PS  (since 1 THz = 1 cycle/ps).
    """
    mask = t >= t_start
    s = sig[mask] - sig[mask].mean()
    N = len(s)
    dt = (t[mask][-1] - t[mask][0]) / (N - 1)
    win = np.hanning(N)
    S = np.fft.rfft(s * win)
    freq_code = np.fft.rfftfreq(N, d=dt)
    freq_thz  = freq_code / HBAR_MEV_PS   # correct: divide by ħ/meV in ps
    power = np.abs(S)**2 / np.sum(win**2)
    return freq_thz, power

def peak_str(freq, power, fmax=3.0, threshold=0.005):
    mask = freq < fmax
    pk = freq[mask][np.argmax(power[mask])]
    pmax = power[mask].max()
    pglobal = power.max()
    if pmax < threshold * pglobal:
        return "< threshold (noise)"
    return f"{pk:.4f} THz"

# ── paths (relative to build dir) ──────────────────────────────────────────────
BUILD = os.path.join(os.path.dirname(__file__), '..', 'build')

cases = [
    ("no DM, $h_x$ pump",  "fe_nodm_hx",        "D₁=0",    "hx"),
    ("no DM, $h_z$ pump",  "fe_nodm_hz",        "D₁=0",    "hz"),
    ("DM on, $h_x$ pump",  "fe_spectrum_clean", "D₁=0.05", "hx"),
    ("DM on, $h_z$ pump",  "fe_dm_hz",          "D₁=0.05", "hz"),
]

qFM  = 0.3885   # THz (reference)
qAFM = 1.1192   # THz (reference)

fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharey='row')
fig.suptitle("DM Mode Mixing: $h_x$ vs $h_z$ pump, D₁=0 vs D₁=0.05\n"
             "(Fe-only, Γ₂ ground state, pump = 0.001)", fontsize=13)

fmax_plot = 2.0   # THz axis limit

for row, (label, outdir, dm_label, pump) in enumerate(cases):
    traj = os.path.join(BUILD, outdir, "sample_0", "pump_probe_trajectory.txt")
    if not os.path.exists(traj):
        for ax in axes[row]: ax.text(0.5, 0.5, f"missing:\n{traj}", ha='center', va='center', transform=ax.transAxes)
        continue

    t, G = load(traj)

    # pre-pump diagnostics
    pre = t < 0
    gx_pre_std = G[pre, 0].std()
    gy_pre_std = G[pre, 1].std()

    fGx, pGx = fft(t, G[:, 0])
    fGy, pGy = fft(t, G[:, 1])

    # normalise each panel independently for visibility
    norm_x = pGx.max() if pGx.max() > 0 else 1
    norm_y = pGy.max() if pGy.max() > 0 else 1

    for col, (freq, power, norm, comp_label, comp_color) in enumerate([
        (fGx, pGx, norm_x, "Gx", "tab:blue"),
        (fGy, pGy, norm_y, "Gy", "tab:orange"),
    ]):
        ax = axes[row][col]
        mask = freq < fmax_plot
        ax.semilogy(freq[mask], power[mask] / norm + 1e-10,
                    color=comp_color, lw=1.2)

        # reference lines
        ax.axvline(qFM,  color='green',  ls='--', lw=1.0, label=f'qFM={qFM} THz')
        ax.axvline(qAFM, color='red',    ls='--', lw=1.0, label=f'qAFM={qAFM} THz')

        pk = peak_str(freq, power)
        pre_std = gx_pre_std if col == 0 else gy_pre_std
        ax.set_title(f"{label}\n{comp_label}  — peak: {pk}\n"
                     f"pre-pump std: {pre_std:.2e}", fontsize=8)
        ax.set_xlim(0, fmax_plot)
        ax.set_ylim(1e-8, 5)
        ax.set_xlabel("Frequency (THz)", fontsize=8)
        ax.set_ylabel("Norm. power (log)", fontsize=8)
        ax.tick_params(labelsize=7)

        if row == 0 and col == 0:
            ax.legend(fontsize=6)

        # shade zero-response region for visual guidance
        if power.max() < 1e-20:
            ax.text(0.5, 0.5, "no signal", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')

plt.tight_layout()
out = os.path.join(BUILD, "fe_dm_mixing_comparison.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

# ── console summary ────────────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"{'Case':<30} {'Gx peak (THz)':>18} {'Gy peak (THz)':>15}")
print("=" * 65)
for label, outdir, dm_label, pump in cases:
    traj = os.path.join(BUILD, outdir, "sample_0", "pump_probe_trajectory.txt")
    if not os.path.exists(traj):
        print(f"{label:<30} {'missing':>18} {'missing':>15}")
        continue
    t, G = load(traj)
    fGx, pGx = fft(t, G[:, 0])
    fGy, pGy = fft(t, G[:, 1])

    def pk(freq, power, threshold=0.01):
        mask = freq < 2.5
        if power[mask].max() < threshold * power.max():
            return "< noise"
        return f"{freq[mask][np.argmax(power[mask])]:.4f}"

    print(f"{label:<30} {pk(fGx, pGx):>18} {pk(fGy, pGy):>15}")
print("=" * 65)
print(f"\nReference: qFM = {qFM} THz, qAFM = {qAFM} THz")
print("Mode mixing signature: with-DM hx pump shows peaks in BOTH Gx AND Gy")
print("Without DM: hx pump should give ONLY Gx (qFM), zero Gy")
print("hz pump: zero without DM, tiny Gy via cant with DM")
