"""
Simulated 2DCS Plot Generator

Creates a synthetic 2D coherent spectroscopy plot with specified peak positions,
mimicking the style of actual simulation outputs from ClassicalSpin_Cpp.

Usage:
    python simulate_2dcs_plot.py [output_dir] [--power|--log|--linear] [--gamma VALUE]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, SymLogNorm, Normalize
from matplotlib.cm import ScalarMappable
import os
import sys
from typing import Literal, Optional, Tuple, List

NormType = Literal['log', 'power', 'symlog', 'linear']

# Conversion factor: 1 THz = 4.135667696 meV
MEV_TO_THZ = 1.0 / 4.135667696


def create_lorentzian_peak(omega_tau_grid: np.ndarray, omega_t_grid: np.ndarray,
                            center_tau: float, center_t: float,
                            width_tau: float = 0.03, width_t: float = 0.03,
                            amplitude: float = 1.0) -> np.ndarray:
    """
    Create a 2D Lorentzian peak centered at (center_tau, center_t).
    Lorentzians have sharper peaks and longer tails than Gaussians.
    
    Args:
        omega_tau_grid: 2D meshgrid for omega_tau axis
        omega_t_grid: 2D meshgrid for omega_t axis
        center_tau: Peak center in omega_tau
        center_t: Peak center in omega_t
        width_tau: Lorentzian HWHM in omega_tau direction
        width_t: Lorentzian HWHM in omega_t direction
        amplitude: Peak amplitude
        
    Returns:
        2D array with the Lorentzian peak
    """
    return amplitude / (1 + ((omega_tau_grid - center_tau) / width_tau)**2 + 
                           ((omega_t_grid - center_t) / width_t)**2)


def create_streak(omega_tau_grid: np.ndarray, omega_t_grid: np.ndarray,
                  center_tau: float, center_t: float,
                  direction: str = 'horizontal',
                  width: float = 0.015, length_decay: float = 0.3,
                  amplitude: float = 0.1) -> np.ndarray:
    """
    Create a streak (horizontal, vertical, or diagonal) emanating from a peak position.
    The streak decays exponentially as distance from the peak increases.
    
    Args:
        omega_tau_grid: 2D meshgrid for omega_tau axis
        omega_t_grid: 2D meshgrid for omega_t axis
        center_tau: Streak center in omega_tau
        center_t: Streak center in omega_t
        direction: 'horizontal', 'vertical', 'diagonal_up', or 'diagonal_down'
        width: Width of the streak perpendicular to its direction
        length_decay: Decay rate along the streak direction (higher = faster decay)
        amplitude: Peak amplitude of the streak
        
    Returns:
        2D array with the streak
    """
    if direction == 'horizontal':
        # Streak along omega_t direction (fixed omega_tau)
        # Gaussian profile perpendicular to streak
        perp_profile = np.exp(-((omega_tau_grid - center_tau) / width)**2)
        # Exponential decay along streak direction (away from peak)
        distance_from_peak = np.abs(omega_t_grid - center_t)
        along_decay = np.exp(-distance_from_peak * length_decay)
        # Also use 1/r^2 falloff for more realistic decay
        along_decay *= 1.0 / (1.0 + (distance_from_peak / 0.1)**2)
        return amplitude * perp_profile * along_decay
    elif direction == 'vertical':
        # Streak along omega_tau direction (fixed omega_t)
        perp_profile = np.exp(-((omega_t_grid - center_t) / width)**2)
        distance_from_peak = np.abs(omega_tau_grid - center_tau)
        along_decay = np.exp(-distance_from_peak * length_decay)
        along_decay *= 1.0 / (1.0 + (distance_from_peak / 0.1)**2)
        return amplitude * perp_profile * along_decay
    elif direction == 'diagonal_up':
        # Diagonal streak (slope +1, along omega_tau = omega_t + const)
        # Distance perpendicular to the diagonal line
        perp_dist = np.abs((omega_tau_grid - center_tau) - (omega_t_grid - center_t)) / np.sqrt(2)
        perp_profile = np.exp(-(perp_dist / width)**2)
        # Distance along the diagonal from the peak
        along_dist = np.abs((omega_tau_grid - center_tau) + (omega_t_grid - center_t)) / np.sqrt(2)
        along_decay = np.exp(-along_dist * length_decay)
        along_decay *= 1.0 / (1.0 + (along_dist / 0.15)**2)
        return amplitude * perp_profile * along_decay
    elif direction == 'diagonal_down':
        # Diagonal streak (slope -1, along omega_tau = -omega_t + const)
        # Distance perpendicular to the anti-diagonal line
        perp_dist = np.abs((omega_tau_grid - center_tau) + (omega_t_grid - center_t)) / np.sqrt(2)
        perp_profile = np.exp(-(perp_dist / width)**2)
        # Distance along the anti-diagonal from the peak
        along_dist = np.abs((omega_tau_grid - center_tau) - (omega_t_grid - center_t)) / np.sqrt(2)
        along_decay = np.exp(-along_dist * length_decay)
        along_decay *= 1.0 / (1.0 + (along_dist / 0.15)**2)
        return amplitude * perp_profile * along_decay
    else:
        return np.zeros_like(omega_tau_grid)


def _get_norm(data: np.ndarray, norm_type: NormType = 'power', 
              gamma: float = 0.5, linthresh: float = 1e-10):
    """Get normalization for plotting based on data and requested type."""
    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        return None
    
    data_min = finite_data.min()
    data_max = finite_data.max()
    
    if data_min >= data_max:
        return None
    
    if norm_type == 'log':
        positive_data = data[data > 0]
        if len(positive_data) == 0:
            return None
        vmin = positive_data.min()
        vmax = positive_data.max()
        if vmin >= vmax:
            return None
        return LogNorm(vmin=vmin, vmax=vmax)
    
    elif norm_type == 'power':
        if data_min <= 0:
            vmin = 0
            vmax = data_max - data_min
        else:
            vmin = data_min
            vmax = data_max
        return PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    
    elif norm_type == 'symlog':
        abs_max = max(abs(data_min), abs(data_max))
        if abs_max > 0:
            auto_linthresh = min(linthresh, abs_max * 0.01)
        else:
            auto_linthresh = linthresh
        return SymLogNorm(linthresh=auto_linthresh, vmin=data_min, vmax=data_max)
    
    elif norm_type == 'linear':
        return Normalize(vmin=data_min, vmax=data_max)
    
    else:
        print(f"Warning: Unknown norm_type '{norm_type}', using linear normalization")
        return None


def add_realistic_noise(spectrum: np.ndarray,
                        omega_tau: np.ndarray,
                        omega_t: np.ndarray,
                        base_level: float = 2e-3,
                        seed: int = 123) -> np.ndarray:
    """
    Add realistic simulation-like noise: low-frequency background, banding,
    speckle/multiplicative noise, and sparse spikes.

    Args:
        spectrum: 2D spectrum array
        omega_tau: 1D omega_tau axis
        omega_t: 1D omega_t axis
        base_level: Base noise level as fraction of max intensity
        seed: Random seed for reproducibility

    Returns:
        Spectrum with noise added
    """
    rng = np.random.default_rng(seed)
    noisy = spectrum.copy()
    max_val = float(np.max(noisy)) if np.max(noisy) > 0 else 1.0

    # Low-frequency background (FFT-smoothed random field)
    noise = rng.normal(size=noisy.shape)
    kx = np.fft.fftfreq(noisy.shape[0])
    ky = np.fft.fftfreq(noisy.shape[1])
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    filt = np.exp(-(KX**2 + KY**2) / (2 * (0.08**2)))
    smooth_noise = np.fft.ifft2(np.fft.fft2(noise) * filt).real
    smooth_noise /= (np.std(smooth_noise) + 1e-12)
    noisy += max_val * base_level * 1.2 * smooth_noise

    # Banding noise (row/column correlated offsets)
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    row_noise = rng.normal(size=noisy.shape[0])
    col_noise = rng.normal(size=noisy.shape[1])
    row_noise = np.convolve(row_noise, kernel, mode='same')
    col_noise = np.convolve(col_noise, kernel, mode='same')
    banding = (row_noise[:, None] + col_noise[None, :])
    banding /= (np.std(banding) + 1e-12)
    noisy += max_val * base_level * 0.8 * banding

    # Add faint periodic stripe artifacts (FFT windowing / sampling)
    tau_phase = np.linspace(0, 2*np.pi, noisy.shape[0], endpoint=False)
    t_phase = np.linspace(0, 2*np.pi, noisy.shape[1], endpoint=False)
    stripes = 0.5 * np.sin(6 * tau_phase)[:, None] + 0.3 * np.sin(5 * t_phase)[None, :]
    noisy += max_val * base_level * 0.6 * stripes

    # Speckle-like multiplicative noise
    speckle = rng.normal(scale=0.20, size=noisy.shape)
    noisy *= (1.0 + 0.05 * speckle)

    # Sparse spikes (FFT leakage / numerical artifacts)
    n_spikes = max(15, int(noisy.size * 4e-5))
    spike_idx = rng.integers(0, noisy.size, size=n_spikes)
    spike_vals = rng.uniform(0.015, 0.06, size=n_spikes) * max_val
    flat = noisy.ravel()
    flat[spike_idx] += spike_vals
    noisy = flat.reshape(noisy.shape)

    # Mild baseline tilt (numerical background drift)
    tau_norm = (omega_tau - omega_tau.min()) / (omega_tau.max() - omega_tau.min())
    t_norm = (omega_t - omega_t.min()) / (omega_t.max() - omega_t.min())
    baseline = (tau_norm[:, None] + t_norm[None, :]) * (max_val * base_level * 0.35)
    noisy += baseline

    return np.clip(noisy, 0, None)


def generate_simulated_2dcs(peaks: List[Tuple[float, float, float]],
                             omega_tau_range: Tuple[float, float] = (0, 8),
                             omega_t_range: Tuple[float, float] = (-8, 8),
                             resolution_tau: int = 1001,
                             resolution_t: int = 2001,
                             peak_width: float = 0.03,
                             add_streaks: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a simulated 2DCS spectrum with specified peaks.
    
    Args:
        peaks: List of (omega_tau, omega_t, amplitude) tuples for peak positions
        omega_tau_range: (min, max) for omega_tau axis
        omega_t_range: (min, max) for omega_t axis
        resolution_tau: Number of points for omega_tau axis
        resolution_t: Number of points for omega_t axis
        peak_width: Width of Lorentzian peaks (HWHM, default 0.03 THz)
        add_streaks: Whether to add horizontal/vertical streaks from peaks
        
    Returns:
        omega_tau: 1D array of omega_tau values
        omega_t: 1D array of omega_t values
        spectrum: 2D array of intensities
    """
    omega_tau = np.linspace(omega_tau_range[0], omega_tau_range[1], resolution_tau)
    omega_t = np.linspace(omega_t_range[0], omega_t_range[1], resolution_t)
    
    omega_tau_grid, omega_t_grid = np.meshgrid(omega_tau, omega_t, indexing='ij')
    
    spectrum = np.zeros_like(omega_tau_grid)
    
    for center_tau, center_t, amplitude in peaks:
        # Add main peak
        spectrum += create_lorentzian_peak(omega_tau_grid, omega_t_grid,
                                            center_tau, center_t,
                                            width_tau=peak_width, width_t=peak_width,
                                            amplitude=amplitude)
        
        # Add streaks radiating from the peak (decaying away from peak)
        if add_streaks:
            streak_amp = amplitude * 0.2  # Streaks are ~20% of peak intensity
            # Horizontal streak (along omega_t)
            spectrum += create_streak(omega_tau_grid, omega_t_grid,
                                       center_tau, center_t,
                                       direction='horizontal',
                                       width=0.012, length_decay=2.6,
                                       amplitude=streak_amp)
            # Vertical streak (along omega_tau)
            spectrum += create_streak(omega_tau_grid, omega_t_grid,
                                       center_tau, center_t,
                                       direction='vertical',
                                       width=0.012, length_decay=3.2,
                                       amplitude=streak_amp * 0.75)
            # Diagonal streak (slope +1, faint)
            spectrum += create_streak(omega_tau_grid, omega_t_grid,
                                       center_tau, center_t,
                                       direction='diagonal_up',
                                       width=0.014, length_decay=4.2,
                                       amplitude=streak_amp * 0.5)
    
    # Add realistic simulation-like noise (background, banding, speckle, spikes)
    spectrum = add_realistic_noise(spectrum, omega_tau, omega_t, base_level=2e-3, seed=123)
    
    return omega_tau, omega_t, spectrum


def plot_2dcs_spectrum(omega_tau: np.ndarray, omega_t: np.ndarray, spectrum: np.ndarray,
                       output_dir: str = '.', norm_type: NormType = 'power', gamma: float = 0.5):
    """
    Plot the 2DCS spectrum mimicking the reader_TmFeO3_SU2.py style.
    
    Args:
        omega_tau: 1D array of omega_tau values
        omega_t: 1D array of omega_t values
        spectrum: 2D spectrum array
        output_dir: Directory to save output
        norm_type: Normalization type for colormap
        gamma: Exponent for PowerNorm
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Transpose spectrum for imshow (omega_t on x-axis, omega_tau on y-axis)
    spectrum_plot = spectrum.T
    
    extent = [omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]]
    
    # Important energy levels for reference lines with labels
    omega_tau_lines = [(0.53, '$e_1$'), (0.76, '$e_2 - e_1$'), (1.29, '$e_2$')]
    omega_t_lines = [(0.53, '$e_1$'), (0.91, '$\\omega_{qAFM}$')]
    
    # Main spectrum plot (similar to M_NLSPEC.pdf style)
    # Make the plot rectangular using the axis scale
    omega_tau_range = float(omega_tau[-1] - omega_tau[0])
    omega_t_range = float(omega_t[-1] - omega_t[0])
    aspect_ratio = omega_t_range / omega_tau_range if omega_tau_range > 0 else 1.0
    fig_width = 10
    fig_height = max(4, fig_width * aspect_ratio)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(spectrum_plot, origin='lower', extent=extent,
                   aspect='equal', cmap='gnuplot2',
                   norm=_get_norm(spectrum_plot, norm_type, gamma))
    
    # Add dashed reference lines for omega_tau (vertical lines) with labels
    for val, label in omega_tau_lines:
        ax.axvline(x=val, color='white', linestyle='--', linewidth=0.9, alpha=0.6)
        # Place label at top of plot
        ax.text(val + 0.02, omega_t[-1] * 0.92, label, color='white', fontsize=18,
            ha='left', va='top', bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.4))
    
    # Add dashed reference lines for omega_t (horizontal lines) with labels - both positive and negative
    for val, label in omega_t_lines:
        ax.axhline(y=val, color='white', linestyle='--', linewidth=0.9, alpha=0.6)
        ax.axhline(y=-val, color='white', linestyle='--', linewidth=0.9, alpha=0.6)
        # Place label at right side of plot
        ax.text(omega_tau[-1] * 0.98, val + 0.03, label, color='white', fontsize=18,
            ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.4))
        ax.text(omega_tau[-1] * 0.98, -val + 0.03, f'-{label}', color='white', fontsize=18,
            ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.4))
    
        ax.set_xlabel('$\\omega_{\\tau}$ (THz)', fontsize=24)
        ax.set_ylabel('$\\omega_t$ (THz)', fontsize=24)
        # No colorbar (per request)
        ax.set_title('Simulated 2DCS Spectrum', fontsize=26)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "simulated_M_NLSPEC.pdf"), dpi=150)
    plt.savefig(os.path.join(output_dir, "simulated_M_NLSPEC.png"), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'simulated_M_NLSPEC.pdf')}")
    plt.clf()
    plt.close()
    
    # Save data to text file
    np.savetxt(os.path.join(output_dir, "simulated_M_NL_FF.txt"), spectrum)
    np.savetxt(os.path.join(output_dir, "omega_tau.txt"), omega_tau)
    np.savetxt(os.path.join(output_dir, "omega_t.txt"), omega_t)
    print(f"Saved spectrum data to: {output_dir}")


def main():
    """Main function to generate simulated 2DCS plot with user-specified peaks."""
    
    # Parse command line arguments
    output_dir = "./simulated_2dcs"
    norm_type = 'power'
    gamma = 0.5
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    for i, arg in enumerate(sys.argv):
        if arg == '--power':
            norm_type = 'power'
        elif arg == '--log':
            norm_type = 'log'
        elif arg == '--linear':
            norm_type = 'linear'
        elif arg == '--symlog':
            norm_type = 'symlog'
        elif arg == '--gamma' and i + 1 < len(sys.argv):
            gamma = float(sys.argv[i + 1])
    
    # Energy parameters in meV (from user specification)
    e1_meV = 2.19190387888       # Tm energy level (meV)
    e_magnon_meV = 3.76345760336  # Magnon energy (meV)
    
    # Convert to THz: divide by 4.135667696
    e1 = e1_meV * MEV_TO_THZ           # ~0.53 THz
    e_magnon = e_magnon_meV * MEV_TO_THZ  # ~0.91 THz
    
    # Window ranges in THz (original 0-8 and -8 to 8 meV, divided by 4.135667696)
    omega_tau_max = 8.0 * MEV_TO_THZ  # ~1.93 THz
    omega_t_max = 8.0 * MEV_TO_THZ    # ~1.93 THz
    
    print("=" * 60)
    print("Simulated 2DCS Plot Generator")
    print("=" * 60)
    print(f"Energy parameters:")
    print(f"  e_1 = {e1_meV:.6f} meV = {e1:.6f} THz")
    print(f"  e_magnon = {e_magnon_meV:.6f} meV = {e_magnon:.6f} THz")
    print(f"Window ranges:")
    print(f"  omega_tau: 0 to {omega_tau_max:.4f} THz")
    print(f"  omega_t: -{omega_t_max:.4f} to {omega_t_max:.4f} THz")
    print(f"Output directory: {output_dir}")
    print(f"Normalization: {norm_type} (gamma={gamma})")
    print()
    
    # Define peaks as specified by user: (omega_tau, omega_t, amplitude)
    # Peaks: (0, e1), (e1, 0), (0, -e1), (e1, e1), (e1, -e1), (e1, e_magnon), (0, e_magnon), (0, -e_magnon)
    # Plus additional peaks and higher order peaks for realism
    # All in THz now
    # Negative frequency counterparts are significantly weaker
    # Adding randomization to amplitudes for realism
    np.random.seed(42)  # For reproducibility
    def rand_amp(base_amp):
        """Add ±20% random variation to amplitude"""
        return base_amp * (0.8 + 0.4 * np.random.random())

    def rand_amp_tight(base_amp):
        """Add ±5% random variation to amplitude"""
        return base_amp * (0.95 + 0.10 * np.random.random())
    
    peaks = [
        # Primary e1 peaks
        (0, e1, rand_amp(1.0)),           # (0, e1)
        (e1, 0, rand_amp(0.18)),          # (e1, 0) - really weak
        (0, -e1, rand_amp(0.45)),         # (0, -e1) - much weaker
        (e1, e1, rand_amp(0.7)),          # (e1, e1)
        (e1, -e1, rand_amp(0.35)),        # (e1, -e1) - much weaker
        
        # Magnon peaks (omega_qAFM line at +0.91 THz kept similar intensity)
        (e1, e_magnon, rand_amp_tight(0.18)),   # (e1, e_magnon)
        (0, e_magnon, rand_amp_tight(0.26)),    # (0, e_magnon)
        (0, -e_magnon, rand_amp(0.06)),         # (0, -e_magnon) - much weaker
        
        # User-specified additional peaks (in THz)
        (0.76, 0.53, rand_amp(0.35)),     # (0.76, 0.53)
        (1.29, 0.53, rand_amp(0.22)),     # (1.29, 0.53)
        (0.53, 0.91, rand_amp_tight(0.18)),     # (0.53, 0.91)
        (1.29, 0.91, rand_amp_tight(0.12)),     # (1.29, 0.91)
        
        # Higher order peaks for realism
        # 2*e1 harmonics
        (2*e1, 0, rand_amp(0.12)),        # (2*e1, 0)
        (0, 2*e1, rand_amp(0.10)),        # (0, 2*e1)
        (0, -2*e1, rand_amp(0.035)),      # (0, -2*e1) - much weaker
        (2*e1, e1, rand_amp(0.08)),       # (2*e1, e1)
        (2*e1, -e1, rand_amp(0.028)),     # (2*e1, -e1) - much weaker
        (e1, 2*e1, rand_amp(0.06)),       # (e1, 2*e1)
        (e1, -2*e1, rand_amp(0.02)),      # (e1, -2*e1) - much weaker
        
        # Combination frequencies with magnon
        (e_magnon, 0, rand_amp(0.10)),    # (e_magnon, 0)
        (e_magnon, e1, rand_amp(0.08)),   # (e_magnon, e1)
        (e_magnon, -e1, rand_amp(0.025)), # (e_magnon, -e1) - much weaker
        (e_magnon, e_magnon, rand_amp_tight(0.18)),  # (e_magnon, e_magnon) = (0.91, 0.91)
        (e_magnon, -e_magnon, rand_amp(0.035)), # (e_magnon, -e_magnon) = (0.91, -0.91) - weaker
        
        # Sum/difference frequencies
        (e1 + e_magnon, 0, rand_amp(0.04)),      # sum freq
        (e_magnon - e1, e1, rand_amp(0.05)),     # difference
        (e1, e_magnon - e1, rand_amp(0.04)),     # 
        (e1, e_magnon + e1, rand_amp(0.03)),     # 
        
        # Weak 3rd order peaks
        (3*e1, 0, rand_amp(0.03)),        # 3*e1
        (0, 3*e1, rand_amp(0.02)),        
        (0, -3*e1, rand_amp(0.006)),      # much weaker
        
        # Cross peaks at specified positions (negative omega_t) - much weaker
        (0.76, -0.53, rand_amp(0.12)),    # much weaker
        (1.29, -0.53, rand_amp(0.10)),    # much weaker
        (1.29, -0.91, rand_amp(0.05)),    # much weaker
        
        # Additional combination peaks
        (0.76, 0.91, rand_amp_tight(0.18)),
        (0.76, -0.91, rand_amp(0.035)),   # much weaker
        (1.29, 0, rand_amp(0.15)),
        (0.76, 0, rand_amp(0.18)),
    ]
    
    print("Peak positions (omega_tau, omega_t) in THz:")
    for i, (tau, t, amp) in enumerate(peaks):
        print(f"  Peak {i+1}: ({tau:.4f}, {t:.4f}) THz, amplitude={amp}")
    print()
    
    # Generate the spectrum with high resolution, sharp peaks, and streaks
    print("Generating 2DCS spectrum (high resolution with streaks)...")
    omega_tau, omega_t, spectrum = generate_simulated_2dcs(
        peaks=peaks,
        omega_tau_range=(0, omega_tau_max),      # omega_tau: 0 to ~1.93 THz
        omega_t_range=(-omega_t_max, omega_t_max),  # omega_t: ~-1.93 to 1.93 THz
        resolution_tau=1001,
        resolution_t=2001,
        peak_width=0.015,            # Sharp Lorentzian peaks (~0.015 THz HWHM)
        add_streaks=True             # Add horizontal/vertical streaks
    )
    
    # Plot and save
    plot_2dcs_spectrum(omega_tau, omega_t, spectrum,
                       output_dir=output_dir,
                       norm_type=norm_type,
                       gamma=gamma)
    
    print()
    print("=" * 60)
    print("Done! Generated files:")
    print(f"  - {output_dir}/simulated_M_NLSPEC.pdf")
    print(f"  - {output_dir}/simulated_M_NLSPEC.png")
    print(f"  - {output_dir}/simulated_M_NL_FF.txt")
    print(f"  - {output_dir}/omega_tau.txt")
    print(f"  - {output_dir}/omega_t.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
