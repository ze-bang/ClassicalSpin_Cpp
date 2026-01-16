#!/usr/bin/env python3
"""
Plot archetypical magnetic phases from SSSF data.
Shows representative examples of different phase types found in the exploration.
V2: Fixed real-space plotting and added peak position markers.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).parent))
from phase_classifier import classify_spin_config, PhaseType

# Plotting style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['figure.dpi'] = 120


def load_spin_config(sample_dir: str) -> tuple:
    """Load spin configuration and positions from sample directory."""
    spin_file = os.path.join(sample_dir, 'spins_T=0.txt')
    pos_file = os.path.join(sample_dir, 'positions.txt')
    
    if not os.path.exists(spin_file) or not os.path.exists(pos_file):
        return None, None
    
    spins = np.loadtxt(spin_file)
    positions = np.loadtxt(pos_file)
    
    # Handle 3D positions (take x, y only)
    if positions.ndim == 2 and positions.shape[1] >= 2:
        positions = positions[:, :2]
    
    return spins, positions


def find_sample_dirs() -> list:
    """Find all sample directories with spin configurations."""
    base_paths = [
        '/home/pc_linux/ClassicalSpin_Cpp/Potential_Param_Scan',
        '/home/pc_linux/ClassicalSpin_Cpp/disordered_sample',
        '/home/pc_linux/ClassicalSpin_Cpp/output',
    ]
    
    sample_dirs = []
    for base in base_paths:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                if 'spins_T=0.txt' in files and 'positions.txt' in files:
                    sample_dirs.append(root)
    
    return sample_dirs


def compute_sssf(spins: np.ndarray, positions: np.ndarray, 
                  resolution: int = 100) -> tuple:
    """
    Compute Static Spin Structure Factor S(q) for honeycomb lattice.
    
    S(q) = (1/N) |sum_i S_i exp(i q.r_i)|^2
    
    Returns: (sssf, q1_vals, q2_vals)
    """
    N = len(spins)
    
    # Create q-grid in reduced coordinates [0, 1] x [0, 1]
    q1_vals = np.linspace(0, 1, resolution)
    q2_vals = np.linspace(0, 1, resolution)
    
    # Honeycomb reciprocal lattice vectors (for a=1)
    # b1 = (2π/a) * (1, 1/√3)
    # b2 = (2π/a) * (0, 2/√3)
    a = 1.0
    b1 = (2*np.pi/a) * np.array([1.0, 1.0/np.sqrt(3)])
    b2 = (2*np.pi/a) * np.array([0.0, 2.0/np.sqrt(3)])
    
    sssf = np.zeros((resolution, resolution))
    
    for i, q1 in enumerate(q1_vals):
        for j, q2 in enumerate(q2_vals):
            # q in Cartesian coordinates
            q = q1 * b1 + q2 * b2
            
            # Compute structure factor
            phase = np.exp(1j * (positions[:, 0] * q[0] + positions[:, 1] * q[1]))
            
            # Sum over spin components
            Sq = 0.0
            for comp in range(3):  # x, y, z
                Sq += np.abs(np.sum(spins[:, comp] * phase))**2
            
            sssf[j, i] = Sq / N  # Note: j,i for proper orientation
    
    return sssf, q1_vals, q2_vals


def find_peaks(sssf: np.ndarray, q1_vals: np.ndarray, q2_vals: np.ndarray, 
               n_peaks: int = 5, min_distance: int = 5) -> list:
    """Find peaks in SSSF."""
    # Local maximum filter
    size = min_distance * 2 + 1
    local_max = maximum_filter(sssf, size=size)
    peaks_mask = (sssf == local_max) & (sssf > sssf.max() * 0.05)
    
    # Get peak positions and intensities
    peak_indices = np.argwhere(peaks_mask)
    peak_intensities = sssf[peaks_mask]
    
    # Sort by intensity
    sorted_idx = np.argsort(peak_intensities)[::-1]
    
    peaks = []
    for idx in sorted_idx[:n_peaks]:
        j, i = peak_indices[idx]
        q1, q2 = q1_vals[i], q2_vals[j]
        intensity = sssf[j, i]
        peaks.append((q1, q2, intensity))
    
    return peaks


def plot_sssf(ax, sssf: np.ndarray, q1_vals: np.ndarray, q2_vals: np.ndarray,
              title: str, phase_name: str, peaks: list):
    """Plot SSSF with proper formatting and peak markers."""
    # Use log scale for better visualization
    sssf_plot = sssf.copy()
    sssf_plot[sssf_plot < 1e-10] = 1e-10
    
    im = ax.imshow(sssf_plot, origin='lower', extent=[0, 1, 0, 1],
                   cmap='hot', norm=LogNorm(vmin=sssf_plot.max()/1000, 
                                            vmax=sssf_plot.max()),
                   aspect='equal')
    
    # Mark high symmetry points
    ax.scatter([0], [0], c='cyan', s=80, marker='o', edgecolors='white', 
               linewidths=1.5, label='Γ (0,0)', zorder=10)
    ax.scatter([0.5], [0], c='lime', s=80, marker='s', edgecolors='white',
               linewidths=1.5, label='M (½,0)', zorder=10)
    ax.scatter([2/3], [1/3], c='yellow', s=80, marker='^', edgecolors='white',
               linewidths=1.5, label='K (⅔,⅓)', zorder=10)
    # Equivalent points
    ax.scatter([1/3], [1/3], c='yellow', s=50, marker='^', edgecolors='white',
               linewidths=1, zorder=10)
    ax.scatter([0], [0.5], c='lime', s=50, marker='s', edgecolors='white',
               linewidths=1, zorder=10)
    ax.scatter([0.5], [0.5], c='lime', s=50, marker='s', edgecolors='white',
               linewidths=1, zorder=10)
    
    # Mark detected peaks with crosses
    for i, (q1, q2, intensity) in enumerate(peaks[:3]):
        ax.scatter([q1], [q2], c='white', s=150, marker='x', linewidths=2, zorder=15)
        ax.annotate(f'Q{i+1}\n({q1:.2f},{q2:.2f})', (q1, q2), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    ax.set_xlabel('$Q_1$ (r.l.u.)')
    ax.set_ylabel('$Q_2$ (r.l.u.)')
    ax.set_title(f'{phase_name}', fontweight='bold', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return im


def plot_spin_config(ax, spins: np.ndarray, positions: np.ndarray, 
                      title: str, max_spins: int = 400):
    """Plot spin configuration in real space."""
    # Subsample if too many spins
    if len(spins) > max_spins:
        idx = np.random.choice(len(spins), max_spins, replace=False)
        spins_plot = spins[idx]
        pos_plot = positions[idx]
    else:
        spins_plot = spins
        pos_plot = positions
    
    # Plot lattice sites first
    ax.scatter(pos_plot[:, 0], pos_plot[:, 1], s=10, c='gray', alpha=0.3, zorder=1)
    
    # Plot spin arrows - use in-plane components for arrows
    # Color by out-of-plane (z) component
    colors = spins_plot[:, 2]  # z-component for color
    
    q = ax.quiver(pos_plot[:, 0], pos_plot[:, 1], 
                  spins_plot[:, 0], spins_plot[:, 1],
                  colors, cmap='coolwarm', clim=(-1, 1),
                  scale=20, width=0.005, headwidth=4, headlength=5,
                  zorder=5)
    
    # Add colorbar
    plt.colorbar(q, ax=ax, label='$S_z$', shrink=0.8)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Real space\n(arrows: $S_x, S_y$, color: $S_z$)', fontsize=10)


def main():
    """Main function to plot archetypical phases."""
    print("Finding sample directories...")
    sample_dirs = find_sample_dirs()
    print(f"Found {len(sample_dirs)} sample directories")
    
    # Classify all samples and group by phase
    phases_found = {}
    
    for sample_dir in sample_dirs:
        spins, positions = load_spin_config(sample_dir)
        if spins is None:
            continue
        
        result = classify_spin_config(spins, positions, verbose=False)
        phase = result.phase.value
        
        if phase not in phases_found:
            phases_found[phase] = []
        
        # Use a better name
        parent = os.path.basename(os.path.dirname(sample_dir))
        phases_found[phase].append({
            'dir': sample_dir,
            'spins': spins,
            'positions': positions,
            'result': result,
            'name': parent
        })
    
    print(f"\nPhases found:")
    for phase, samples in phases_found.items():
        print(f"  {phase}: {len(samples)} samples")
    
    # Select one representative from each phase
    representative_phases = []
    for phase, samples in phases_found.items():
        best = max(samples, key=lambda x: x['result'].confidence)
        representative_phases.append((phase, best))
    
    # Sort by phase name
    representative_phases.sort(key=lambda x: x[0])
    
    n_phases = len(representative_phases)
    if n_phases == 0:
        print("No phases found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, n_phases, figsize=(4.5*n_phases, 9))
    if n_phases == 1:
        axes = axes.reshape(2, 1)
    
    print(f"\nPlotting {n_phases} archetypical phases...")
    
    for i, (phase, sample) in enumerate(representative_phases):
        print(f"  Processing {phase}...")
        
        # Compute SSSF
        sssf, q1_vals, q2_vals = compute_sssf(sample['spins'], sample['positions'], 
                                               resolution=100)
        
        # Find peaks
        peaks = find_peaks(sssf, q1_vals, q2_vals, n_peaks=5)
        print(f"    Top peaks: {[(f'{p[0]:.3f}',f'{p[1]:.3f}') for p in peaks[:3]]}")
        
        # Plot SSSF (top row)
        plot_sssf(axes[0, i], sssf, q1_vals, q2_vals, sample['name'], phase, peaks)
        
        # Plot spin config (bottom row)
        plot_spin_config(axes[1, i], sample['spins'], sample['positions'], phase)
    
    # Add legend to first SSSF plot
    axes[0, 0].legend(loc='upper right', fontsize=7, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/home/pc_linux/ClassicalSpin_Cpp/util/classical_solvers/archetypical_phases_v2.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {output_path}")
    
    # Also print peak summary
    print("\n" + "="*60)
    print("PEAK POSITION SUMMARY")
    print("="*60)
    for phase, sample in representative_phases:
        sssf, q1_vals, q2_vals = compute_sssf(sample['spins'], sample['positions'], 
                                               resolution=100)
        peaks = find_peaks(sssf, q1_vals, q2_vals, n_peaks=3)
        print(f"\n{phase}:")
        for j, (q1, q2, intensity) in enumerate(peaks):
            print(f"  Q{j+1}: ({q1:.4f}, {q2:.4f})  I={intensity:.2f}")
    
    plt.show()


if __name__ == '__main__':
    main()
