#!/usr/bin/env python3
"""
Detailed Fe magnon spectrum analysis - decompose into all Cartesian components
and look for sub-dominant peaks, harmonics, mode mixing signatures.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

HBAR_OVER_MEV_IN_PS = 0.6582119514
MEV_TO_THZ = 0.24180
THZ_TO_MEV = 1.0 / MEV_TO_THZ

def load_trajectory(traj_file):
    data = np.loadtxt(traj_file)
    t = data[:, 0]
    G = data[:, 1:4]
    L = data[:, 4:7]
    F = data[:, 7:10]
    return t, G, L, F

def spectrum_1d(t, sig, t_start=5.0):
    mask = t >= t_start
    t_sel = t[mask]
    s = sig[mask] - sig[mask].mean()
    N = len(s)
    dt = t_sel[1] - t_sel[0]
    win = np.hanning(N)
    fft_vals = np.fft.fft(s * win)
    power = np.abs(fft_vals[:N//2])**2
    freq = np.fft.fftfreq(N, d=dt)[:N//2]
    freq_thz = freq / HBAR_OVER_MEV_IN_PS
    return freq_thz, power

def find_all_peaks(freq, power, min_freq=0.02, max_freq=5.0, min_height_frac=1e-4, n_peaks=20):
    """Find peaks with very low threshold to catch sub-dominant structure."""
    mask = (freq >= min_freq) & (freq <= max_freq)
    f, p = freq[mask], power[mask]
    if len(p) < 3:
        return [], []
    threshold = min_height_frac * p.max()
    peaks = []
    for i in range(1, len(p)-1):
        if p[i] > p[i-1] and p[i] > p[i+1] and p[i] > threshold:
            peaks.append((f[i], p[i]))
    peaks.sort(key=lambda x: -x[1])
    return [p[0] for p in peaks[:n_peaks]], [p[1] for p in peaks[:n_peaks]]

# ---- Main ----
traj_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/pc_linux/ClassicalSpin_Cpp/build/fe_spectrum_Ka02545/sample_0'
traj_file = os.path.join(traj_dir, 'pump_probe_trajectory.txt')
t, G, L, F = load_trajectory(traj_file)

# Equilibrium
pre = t < -1.0
G_eq = G[pre].mean(axis=0)
F_eq = F[pre].mean(axis=0)
print(f"Equilibrium: G=({G_eq[0]:.5f}, {G_eq[1]:.5f}, {G_eq[2]:.5f}), F=({F_eq[0]:.5f}, {F_eq[1]:.5f}, {F_eq[2]:.5f})")
print(f"|G|={np.linalg.norm(G_eq):.5f}, |F|={np.linalg.norm(F_eq):.5f}")

# Compute spectra per component
labels = ['Gx','Gy','Gz','Fx','Fy','Fz']
signals = [G[:,0], G[:,1], G[:,2], F[:,0], F[:,1], F[:,2]]
spectra = {}
for lbl, sig in zip(labels, signals):
    freq, power = spectrum_1d(t, sig)
    spectra[lbl] = (freq, power)

# Also L-mode
for i, c in enumerate('xyz'):
    freq, power = spectrum_1d(t, L[:,i])
    spectra[f'L{c}'] = (freq, power)

# Print all peaks for each component
print(f"\n{'='*80}")
print(f"Peak analysis (threshold = 0.01% of max per component)")
print(f"{'='*80}")
for lbl in ['Gx','Gy','Gz','Lx','Ly','Lz','Fx','Fy','Fz']:
    freq, power = spectra[lbl]
    peaks, amps = find_all_peaks(freq, power, min_height_frac=1e-4)
    if not peaks:
        print(f"\n{lbl}: no peaks found")
        continue
    max_amp = max(amps) if amps else 1
    print(f"\n{lbl} peaks:")
    for f, a in sorted(zip(peaks, amps)):
        rel = a / max_amp
        print(f"  {f:8.4f} THz ({f*THZ_TO_MEV:7.3f} meV)  rel_amp={rel:.4e}")

# Comprehensive plot
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
fig.suptitle('Detailed Fe-only TmFeO3 Spectrum Decomposition', fontsize=14)

# Row 1: Time traces (short window)
t_ps = t * HBAR_OVER_MEV_IN_PS
tmask = (t >= -2) & (t <= 60)
for col, (lbl, grp) in enumerate([('G-mode', G), ('L-mode', L), ('F-mode', F)]):
    ax = axes[0, col]
    eq = grp[pre].mean(axis=0)
    for i, c in enumerate('xyz'):
        deviation = grp[tmask, i] - eq[i]
        ax.plot(t_ps[tmask], deviation, alpha=0.7, label=f'$\\delta {lbl[0]}_{c}$')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel(f'{lbl} deviation')
    ax.set_title(f'{lbl} time trace (deviations from equilibrium)')
    ax.legend(fontsize=7)
    ax.axvline(0, color='k', ls='--', alpha=0.3)

# Row 2: Individual component spectra on LOG scale
target_freqs = {0.38: 'qFM', 1.12: 'qAFM'}
for col, grp_labels in enumerate([['Gx','Gy','Gz'], ['Lx','Ly','Lz'], ['Fx','Fy','Fz']]):
    ax = axes[1, col]
    for lbl in grp_labels:
        freq, power = spectra[lbl]
        pmax = power.max() if power.max() > 0 else 1
        ax.semilogy(freq, power/pmax, alpha=0.7, label=lbl)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(1e-8, 5)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Normalized power')
    ax.set_title(f'{grp_labels[0][0]}-mode component spectra')
    ax.legend(fontsize=8)
    for f0, name in target_freqs.items():
        ax.axvline(f0, color='grey', ls=':', alpha=0.5)
        ax.text(f0+0.02, 2, name, fontsize=7, color='grey')

# Row 3: Zoomed views around each mode
# Zoom around qFM
ax = axes[2, 0]
for lbl in ['Gx','Gy','Gz','Fx','Fy','Fz']:
    freq, power = spectra[lbl]
    pmax = power[(freq>0.1) & (freq<0.6)].max() if power[(freq>0.1) & (freq<0.6)].max() > 0 else 1
    mask_zoom = (freq > 0.1) & (freq < 0.7)
    if power[mask_zoom].max() > 1e-6 * pmax:
        ax.semilogy(freq[mask_zoom], power[mask_zoom]/pmax, alpha=0.7, label=lbl)
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('Power (norm to max in window)')
ax.set_title('Zoom: qFM region (0.1–0.7 THz)')
ax.axvline(0.38, color='red', ls=':', alpha=0.7)
ax.legend(fontsize=7)

# Zoom around qAFM
ax = axes[2, 1]
for lbl in ['Gx','Gy','Gz','Fx','Fy','Fz']:
    freq, power = spectra[lbl]
    pmax = power[(freq>0.8) & (freq<1.5)].max() if power[(freq>0.8) & (freq<1.5)].max() > 0 else 1
    mask_zoom = (freq > 0.8) & (freq < 1.5)
    if power[mask_zoom].max() > 1e-6 * pmax:
        ax.semilogy(freq[mask_zoom], power[mask_zoom]/pmax, alpha=0.7, label=lbl)
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('Power (norm to max in window)')
ax.set_title('Zoom: qAFM region (0.8–1.5 THz)')
ax.axvline(1.12, color='blue', ls=':', alpha=0.7)
ax.legend(fontsize=7)

# Zoom high freq: look for harmonics / sum frequencies
ax = axes[2, 2]
for lbl in ['Gx','Gy','Gz','Fx','Fy','Fz']:
    freq, power = spectra[lbl]
    mask_zoom = (freq > 1.2) & (freq < 3.5)
    pmax = power[mask_zoom].max() if power[mask_zoom].max() > 0 else 1
    ax.semilogy(freq[mask_zoom], power[mask_zoom]/pmax, alpha=0.7, label=lbl)
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('Power (norm to max in window)')
ax.set_title('High-freq region (1.2–3.5 THz)')
ax.legend(fontsize=7)
# Mark expected harmonics/sums
for f0, name in [(0.76, '2×qFM'), (1.50, 'qFM+qAFM'), (2.24, '2×qAFM')]:
    ax.axvline(f0, color='orange', ls='--', alpha=0.5)
    ax.text(f0+0.02, 0.5, name, fontsize=7, color='orange', rotation=90)

# Row 4: Mode character analysis
# What drives what: compare G vs F at each frequency
ax = axes[3, 0]
freq_G, pow_G_total = spectrum_1d(t, np.linalg.norm(G - G_eq, axis=1))
freq_F, pow_F_total = spectrum_1d(t, np.linalg.norm(F - F_eq, axis=1))
ax.semilogy(freq_G, pow_G_total/max(pow_G_total.max(),1e-30), 'r-', alpha=0.7, label='|δG| total')
ax.semilogy(freq_F, pow_F_total/max(pow_F_total.max(),1e-30), 'b-', alpha=0.7, label='|δF| total')
ax.set_xlim(0, 3.5)
ax.set_ylim(1e-8, 5)
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('Power')
ax.set_title('Total |δG| vs |δF| spectra')
ax.legend()
for f0, name in target_freqs.items():
    ax.axvline(f0, color='grey', ls=':', alpha=0.5)

# Cross-component analysis: which Cartesian component dominates at each mode?
ax = axes[3, 1]
comp_labels = ['Gx','Gy','Gz','Fx','Fy','Fz']
colors = ['tab:blue','tab:red','tab:green','tab:cyan','tab:orange','tab:purple']
# Evaluate power at specific frequencies
all_peaks = set()
for lbl in comp_labels:
    f, p = spectra[lbl]
    pks, _ = find_all_peaks(f, p, min_height_frac=1e-3)
    all_peaks.update(pks)
all_peaks = sorted(all_peaks)
# Only keep peaks within 0.1-3 THz
all_peaks = [p for p in all_peaks if 0.1 < p < 3.0]

if all_peaks:
    # For each peak, find decomposition
    peak_data = {}
    for pk in all_peaks:
        contrib = {}
        for lbl in comp_labels:
            freq, power = spectra[lbl]
            idx = np.argmin(np.abs(freq - pk))
            contrib[lbl] = power[idx]
        total = sum(contrib.values())
        peak_data[pk] = {k: v/total if total > 0 else 0 for k, v in contrib.items()}
    
    # Plot as stacked bars
    x = np.arange(len(all_peaks))
    bottom = np.zeros(len(all_peaks))
    for i, lbl in enumerate(comp_labels):
        vals = [peak_data[pk][lbl] for pk in all_peaks]
        ax.bar(x, vals, bottom=bottom, color=colors[i], label=lbl, alpha=0.7)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p:.3f}' for p in all_peaks], rotation=45, fontsize=7)
    ax.set_xlabel('Peak frequency (THz)')
    ax.set_ylabel('Fractional power')
    ax.set_title('Mode composition at each peak')
    ax.legend(fontsize=6, ncol=2)

# Print summary: mode character table
ax = axes[3, 2]
ax.axis('off')
summary = "Mode Character Summary\n" + "="*40 + "\n"
if all_peaks:
    for pk in all_peaks:
        contrib = peak_data[pk]
        sorted_c = sorted(contrib.items(), key=lambda x: -x[1])
        dominant = sorted_c[0]
        summary += f"\n{pk:.4f} THz ({pk*THZ_TO_MEV:.2f} meV):\n"
        for lbl, frac in sorted_c:
            if frac > 0.01:
                summary += f"  {lbl}: {frac*100:.1f}%\n"
ax.text(0, 1, summary, fontsize=8, family='monospace', verticalalignment='top', transform=ax.transAxes)

plt.tight_layout()
out = os.path.join(traj_dir, 'fe_spectrum_detailed.png')
plt.savefig(out, dpi=150)
print(f"\nDetailed spectrum saved to: {out}")
