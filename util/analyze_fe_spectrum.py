#!/usr/bin/env python3
"""
Fe-only magnon spectrum analysis for TmFeO3.

Reads pump-probe trajectories, computes FFT, identifies qFM and qAFM modes.
Supports parameter scanning to tune frequencies.

Usage:
    python3 analyze_fe_spectrum.py [trajectory_dir]
    python3 analyze_fe_spectrum.py --scan   # Run parameter scan
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import glob

# Physical constants
HBAR_OVER_MEV_IN_PS = 0.6582119514  # ℏ/meV in ps
MEV_TO_THZ = 0.24180  # 1 meV = 0.24180 THz (ordinary freq)
THZ_TO_MEV = 1.0 / MEV_TO_THZ  # 1 THz = 4.1357 meV


def load_trajectory(traj_file):
    """
    Load pump-probe trajectory.
    Columns: t, Gx, Gy, Gz, Lx, Ly, Lz, Fx, Fy, Fz
    """
    data = np.loadtxt(traj_file)
    t = data[:, 0]
    G = data[:, 1:4]   # Staggered magnetization (Bertaut G-mode)
    L = data[:, 4:7]   # Local-frame average
    F = data[:, 7:10]  # Uniform magnetization (Bertaut F-mode)
    return t, G, L, F


def compute_spectrum(t, signal, t_start=5.0, window='hann'):
    """
    Compute power spectrum of a signal.
    
    Args:
        t: time array (code units: ℏ/meV)
        signal: signal array (can be multi-column)
        t_start: start time for FFT (skip pump transient)
        window: window function ('hann', 'blackman', or None)
    
    Returns:
        freq_thz: frequency axis in THz
        power: power spectral density
    """
    # Select post-pump data
    mask = t >= t_start
    t_sel = t[mask]
    sig_sel = signal[mask]
    
    N = len(t_sel)
    dt_code = t_sel[1] - t_sel[0]
    dt_ps = dt_code * HBAR_OVER_MEV_IN_PS
    
    # Apply window
    if window == 'hann':
        win = np.hanning(N)
    elif window == 'blackman':
        win = np.blackman(N)
    else:
        win = np.ones(N)
    
    # Handle multi-column signals
    if sig_sel.ndim == 1:
        sig_sel = sig_sel[:, None]
    
    # Subtract mean (remove DC offset)
    sig_sel = sig_sel - sig_sel.mean(axis=0)
    
    # FFT
    power_total = np.zeros(N // 2)
    for col in range(sig_sel.shape[1]):
        fft_vals = np.fft.fft(sig_sel[:, col] * win)
        power = np.abs(fft_vals[:N // 2]) ** 2
        power_total += power
    
    # Frequency axis
    freq_code = np.fft.fftfreq(N, d=dt_code)[:N // 2]  # in 1/code-time
    freq_thz = freq_code / HBAR_OVER_MEV_IN_PS  # convert to THz
    
    return freq_thz, power_total


def find_peaks(freq, power, min_freq=0.05, max_freq=3.0, n_peaks=4, min_height_frac=0.01):
    """Find peaks in the spectrum."""
    # Restrict range
    mask = (freq >= min_freq) & (freq <= max_freq)
    freq_r = freq[mask]
    power_r = power[mask]
    
    if len(power_r) < 3:
        return [], []
    
    # Simple peak finding
    peaks = []
    threshold = min_height_frac * power_r.max()
    for i in range(1, len(power_r) - 1):
        if power_r[i] > power_r[i-1] and power_r[i] > power_r[i+1] and power_r[i] > threshold:
            peaks.append((freq_r[i], power_r[i]))
    
    # Sort by amplitude
    peaks.sort(key=lambda x: -x[1])
    
    peak_freqs = [p[0] for p in peaks[:n_peaks]]
    peak_amps = [p[1] for p in peaks[:n_peaks]]
    
    return peak_freqs, peak_amps


def analyze_and_plot(traj_dir, save_dir=None):
    """Load trajectory, compute spectrum, plot, and return peak frequencies."""
    if save_dir is None:
        save_dir = traj_dir
    
    traj_file = os.path.join(traj_dir, 'pump_probe_trajectory.txt')
    if not os.path.exists(traj_file):
        print(f"Error: {traj_file} not found")
        return None
    
    t, G, L, F = load_trajectory(traj_file)
    
    # Print equilibrium state (before pump)
    pre_pump = t < -1.0
    if np.any(pre_pump):
        G_eq = G[pre_pump].mean(axis=0)
        F_eq = F[pre_pump].mean(axis=0)
        print(f"\nEquilibrium state (pre-pump average):")
        print(f"  G = ({G_eq[0]:.4f}, {G_eq[1]:.4f}, {G_eq[2]:.4f})")
        print(f"  F = ({F_eq[0]:.4f}, {F_eq[1]:.4f}, {F_eq[2]:.4f})")
        print(f"  |G| = {np.linalg.norm(G_eq):.4f}")
        print(f"  |F| = {np.linalg.norm(F_eq):.4f}")
        
        # Cant angle
        if abs(G_eq[2]) > 0.01:
            theta_cant = np.arctan2(abs(G_eq[0]), abs(G_eq[2]))
            print(f"  Cant angle: {np.degrees(theta_cant):.3f} deg = {theta_cant:.5f} rad")
    
    # Compute spectra from different observables
    freq_G, pow_G = compute_spectrum(t, G, t_start=5.0)
    freq_F, pow_F = compute_spectrum(t, F, t_start=5.0)
    
    # Also compute component-specific spectra
    freq_Gy, pow_Gy = compute_spectrum(t, G[:, 1], t_start=5.0)
    freq_Gz, pow_Gz = compute_spectrum(t, G[:, 2], t_start=5.0)
    freq_Fx, pow_Fx = compute_spectrum(t, F[:, 0], t_start=5.0)
    freq_Fy, pow_Fy = compute_spectrum(t, F[:, 1], t_start=5.0)
    
    # Find peaks in total spectrum
    peaks_G, amps_G = find_peaks(freq_G, pow_G)
    peaks_F, amps_F = find_peaks(freq_F, pow_F)
    
    print(f"\nPeaks in G-mode spectrum (THz):")
    for f, a in zip(peaks_G, amps_G):
        print(f"  f = {f:.4f} THz ({f * THZ_TO_MEV:.3f} meV)")
    
    print(f"\nPeaks in F-mode spectrum (THz):")
    for f, a in zip(peaks_F, amps_F):
        print(f"  f = {f:.4f} THz ({f * THZ_TO_MEV:.3f} meV)")
    
    # ----- PLOTTING -----
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Fe-only TmFeO3 Magnon Spectrum', fontsize=14)
    
    # 1. Time traces (post-pump)
    t_plot_mask = (t >= -2) & (t <= 50)
    ax = axes[0, 0]
    ax.plot(t[t_plot_mask] * HBAR_OVER_MEV_IN_PS, G[t_plot_mask, 0], 'b-', alpha=0.7, label='$G_x$')
    ax.plot(t[t_plot_mask] * HBAR_OVER_MEV_IN_PS, G[t_plot_mask, 1], 'r-', alpha=0.7, label='$G_y$')
    ax.plot(t[t_plot_mask] * HBAR_OVER_MEV_IN_PS, G[t_plot_mask, 2] - G[pre_pump, 2].mean(), 'g-', alpha=0.7, label='$\\delta G_z$')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('G-mode')
    ax.set_title('G-mode time trace')
    ax.legend(fontsize=8)
    ax.axvline(0, color='k', ls='--', alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(t[t_plot_mask] * HBAR_OVER_MEV_IN_PS, F[t_plot_mask, 0] - F[pre_pump, 0].mean(), 'b-', alpha=0.7, label='$\\delta F_x$')
    ax.plot(t[t_plot_mask] * HBAR_OVER_MEV_IN_PS, F[t_plot_mask, 1], 'r-', alpha=0.7, label='$F_y$')
    ax.plot(t[t_plot_mask] * HBAR_OVER_MEV_IN_PS, F[t_plot_mask, 2], 'g-', alpha=0.7, label='$F_z$')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('F-mode')
    ax.set_title('F-mode time trace')
    ax.legend(fontsize=8)
    ax.axvline(0, color='k', ls='--', alpha=0.3)
    
    # 2. G-mode spectrum (components)
    ax = axes[1, 0]
    ax.semilogy(freq_Gy, pow_Gy / pow_Gy.max(), 'r-', alpha=0.7, label='$G_y$')
    ax.semilogy(freq_Gz, pow_Gz / pow_Gz.max(), 'g-', alpha=0.7, label='$G_z$')
    ax.set_xlim(0, 3.0)
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Power (normalized)')
    ax.set_title('G-mode component spectra')
    ax.legend()
    # Mark target frequencies
    ax.axvline(0.38, color='k', ls=':', alpha=0.5, label='target qFM')
    ax.axvline(1.12, color='k', ls='--', alpha=0.5, label='target qAFM')
    for f in peaks_G[:3]:
        ax.axvline(f, color='blue', ls='-', alpha=0.3)
        ax.text(f, 1.5, f'{f:.3f}', fontsize=7, rotation=90, ha='right')
    
    # 3. F-mode spectrum (components)
    ax = axes[1, 1]
    ax.semilogy(freq_Fx, pow_Fx / max(pow_Fx.max(), 1e-30), 'b-', alpha=0.7, label='$F_x$')
    ax.semilogy(freq_Fy, pow_Fy / max(pow_Fy.max(), 1e-30), 'r-', alpha=0.7, label='$F_y$')
    ax.set_xlim(0, 3.0)
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Power (normalized)')
    ax.set_title('F-mode component spectra')
    ax.legend()
    ax.axvline(0.38, color='k', ls=':', alpha=0.5)
    ax.axvline(1.12, color='k', ls='--', alpha=0.5)
    for f in peaks_F[:3]:
        ax.axvline(f, color='blue', ls='-', alpha=0.3)
        ax.text(f, 1.5, f'{f:.3f}', fontsize=7, rotation=90, ha='right')
    
    # 4. Combined total spectrum
    ax = axes[2, 0]
    pow_total = pow_G + pow_F
    pow_total_norm = pow_total / pow_total.max() if pow_total.max() > 0 else pow_total
    ax.plot(freq_G, pow_total_norm, 'k-', lw=1.5)
    ax.set_xlim(0, 3.0)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Total power (normalized)')
    ax.set_title('Combined G+F spectrum')
    ax.axvline(0.38, color='red', ls=':', alpha=0.7, label='qFM target (0.38 THz)')
    ax.axvline(1.12, color='blue', ls='--', alpha=0.7, label='qAFM target (1.12 THz)')
    peaks_all, _ = find_peaks(freq_G, pow_total)
    for f in peaks_all[:3]:
        ax.axvline(f, color='green', ls='-', alpha=0.4)
        ax.text(f + 0.01, 0.8, f'{f:.3f} THz\n({f*THZ_TO_MEV:.2f} meV)', fontsize=8)
    ax.legend(fontsize=8)
    
    # 5. Long time trace to check for damping/stationarity
    ax = axes[2, 1]
    t_ps = t * HBAR_OVER_MEV_IN_PS
    ax.plot(t_ps, G[:, 1], 'r-', alpha=0.5, lw=0.3, label='$G_y$')
    ax.plot(t_ps, F[:, 1], 'b-', alpha=0.5, lw=0.3, label='$F_y$')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Full time trace')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'fe_spectrum.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSpectrum plot saved to: {fig_path}")
    
    return peaks_G, peaks_F


def run_single_sim(params, build_dir, param_template, output_name):
    """Run a single Fe-only simulation with given parameters."""
    # Create modified param file
    param_file = os.path.join(build_dir, f'param_{output_name}.param')
    with open(param_template) as f:
        content = f.read()
    
    # Override parameters
    for key, val in params.items():
        import re
        pattern = rf'^{re.escape(key)}\s*=.*$'
        replacement = f'{key} = {val}'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Set output directory
    content = re.sub(r'^output_dir\s*=.*$', f'output_dir = {output_name}', content, flags=re.MULTILINE)
    
    with open(param_file, 'w') as f:
        f.write(content)
    
    # Run
    result = subprocess.run(
        ['mpirun', '-np', '1', './spin_solver', param_file],
        cwd=build_dir,
        capture_output=True, text=True, timeout=300
    )
    
    if result.returncode != 0:
        print(f"Simulation failed: {result.stderr[:200]}")
        return None
    
    # Analyze
    traj_dir = os.path.join(build_dir, output_name, 'sample_0')
    traj_file = os.path.join(traj_dir, 'pump_probe_trajectory.txt')
    if not os.path.exists(traj_file):
        print(f"Trajectory not found: {traj_file}")
        return None
    
    t, G, L, F = load_trajectory(traj_file)
    freq, pow_total = compute_spectrum(t, np.hstack([G, F]), t_start=5.0)
    peaks, _ = find_peaks(freq, pow_total, min_freq=0.05, max_freq=3.0)
    
    return peaks


def parameter_scan(build_dir):
    """
    Scan Ka, Kc, D1 parameter space to match target frequencies.
    
    Target: qFM = 0.38 THz, qAFM = 1.12 THz
    """
    template = os.path.join(os.path.dirname(__file__), 
                           '..', 'example_configs', 'TmFeO3', 
                           'pump_probe_fe_only_spectrum.param')
    template = os.path.abspath(template)
    
    print("=" * 60)
    print("Fe-only parameter scan for TmFeO3")
    print(f"Target: qFM = 0.38 THz, qAFM = 1.12 THz")
    print("=" * 60)
    
    # Scan over Ka-Kc difference and D1
    # Ka controls the anisotropy gap (with Kc)
    # D1 controls mode mixing and cant angle
    results = []
    
    # First pass: coarse scan
    Ka_vals = [-0.01, -0.02, -0.03, -0.05, -0.08, -0.12, -0.16]
    Kc_vals = [-0.02, -0.03, -0.05, -0.08, -0.12, -0.18, -0.25]
    D1_vals = [0.02, 0.05, 0.08, 0.12, 0.15]
    
    for Ka in Ka_vals:
        for Kc in Kc_vals:
            if Kc >= Ka:  # Need |Kc| >= |Ka| for Gamma_2
                continue
            for D1 in D1_vals:
                name = f"scan_Ka{Ka:.3f}_Kc{Kc:.3f}_D1{D1:.3f}"
                params = {'Ka': Ka, 'Kc': Kc, 'D1': D1}
                
                print(f"\n--- Ka={Ka}, Kc={Kc}, D1={D1} ---")
                peaks = run_single_sim(params, build_dir, template, name)
                
                if peaks and len(peaks) >= 2:
                    f1, f2 = sorted(peaks[:2])
                    err = np.sqrt((f1 - 0.38)**2 + (f2 - 1.12)**2)
                    results.append({
                        'Ka': Ka, 'Kc': Kc, 'D1': D1,
                        'f_low': f1, 'f_high': f2, 'error': err
                    })
                    print(f"  Peaks: {f1:.4f}, {f2:.4f} THz | Error: {err:.4f}")
    
    # Sort by error
    results.sort(key=lambda x: x['error'])
    
    print("\n" + "=" * 60)
    print("Top 10 parameter sets:")
    print("=" * 60)
    for r in results[:10]:
        print(f"Ka={r['Ka']:.3f}, Kc={r['Kc']:.3f}, D1={r['D1']:.3f} "
              f"-> f_low={r['f_low']:.4f}, f_high={r['f_high']:.4f} THz "
              f"(error={r['error']:.4f})")
    
    return results


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--scan':
        build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
        build_dir = os.path.abspath(build_dir)
        parameter_scan(build_dir)
    else:
        if len(sys.argv) > 1:
            traj_dir = sys.argv[1]
        else:
            traj_dir = os.path.join(os.path.dirname(__file__), '..', 'build',
                                    'fe_spectrum_test', 'sample_0')
            traj_dir = os.path.abspath(traj_dir)
        
        analyze_and_plot(traj_dir)
