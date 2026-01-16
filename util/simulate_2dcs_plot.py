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
    Create a streak (horizontal or vertical) emanating from a peak position.
    The streak decays exponentially as distance from the peak increases.
    
    Args:
        omega_tau_grid: 2D meshgrid for omega_tau axis
        omega_t_grid: 2D meshgrid for omega_t axis
        center_tau: Streak center in omega_tau
        center_t: Streak center in omega_t
        direction: 'horizontal' (along omega_t) or 'vertical' (along omega_tau)
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
    else:
        # Streak along omega_tau direction (fixed omega_t)
        perp_profile = np.exp(-((omega_t_grid - center_t) / width)**2)
        distance_from_peak = np.abs(omega_tau_grid - center_tau)
        along_decay = np.exp(-distance_from_peak * length_decay)
        along_decay *= 1.0 / (1.0 + (distance_from_peak / 0.1)**2)
        return amplitude * perp_profile * along_decay


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
            streak_amp = amplitude * 0.12  # Streaks are ~12% of peak intensity
            # Horizontal streak (along omega_t)
            spectrum += create_streak(omega_tau_grid, omega_t_grid,
                                       center_tau, center_t,
                                       direction='horizontal',
                                       width=0.010, length_decay=3.0,
                                       amplitude=streak_amp)
            # Vertical streak (along omega_tau)
            spectrum += create_streak(omega_tau_grid, omega_t_grid,
                                       center_tau, center_t,
                                       direction='vertical',
                                       width=0.010, length_decay=4.0,
                                       amplitude=streak_amp * 0.6)
    
    # Add very small noise floor for realism (like numerical precision)
    noise_level = 1e-16
    spectrum += noise_level * np.abs(np.random.randn(*spectrum.shape))
    
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
    
    # Main spectrum plot (similar to M_NLSPEC.pdf style)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(spectrum_plot, origin='lower', extent=extent,
                    aspect='auto', cmap='gnuplot2',
                    norm=_get_norm(spectrum_plot, norm_type, gamma))
    plt.xlabel('$\\omega_{\\tau}$ (THz)', fontsize=14)
    plt.ylabel('$\\omega_t$ (THz)', fontsize=14)
    plt.colorbar(im, label='Intensity')
    plt.title('Simulated 2DCS Spectrum', fontsize=16)
    
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
    # All in THz now
    peaks = [
        (0, e1, 1.0),           # (0, e1)
        (e1, 0, 0.8),           # (e1, 0)
        (0, -e1, 0.9),          # (0, -e1)
        (e1, e1, 0.7),          # (e1, e1)
        (e1, -e1, 0.75),        # (e1, -e1)
        (e1, e_magnon, 0.15),   # (e1, e_magnon)
        (0, e_magnon, 0.18),    # (0, e_magnon)
        (0, -e_magnon, 0.15),   # (0, -e_magnon)
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
