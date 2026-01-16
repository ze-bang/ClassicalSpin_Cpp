#!/usr/bin/env python3
"""
Plot archetypical magnetic phases from SSSF data.
Shows representative examples of different phase types found in the exploration.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase_classifier import classify_spin_config, PhaseType

# Plotting style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['figure.dpi'] = 120


def load_sssf_file(filepath: str) -> np.ndarray:
    """Load SSSF data from file."""
    return np.loadtxt(filepath)


def load_spin_config(sample_dir: str) -> tuple:
    """Load spin configuration and positions from sample directory."""
    spin_file = os.path.join(sample_dir, 'spins_T=0.txt')
    pos_file = os.path.join(sample_dir, 'positions.txt')
    
    if not os.path.exists(spin_file) or not os.path.exists(pos_file):
        return None, None
    
    spins = np.loadtxt(spin_file)
    positions = np.loadtxt(pos_file)
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
                  resolution: int = 100) -> np.ndarray:
    """
    Compute Static Spin Structure Factor S(q) for honeycomb lattice.
    
    S(q) = (1/N) |sum_i S_i exp(i q.r_i)|^2
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
    
    return sssf


def plot_sssf(ax, sssf: np.ndarray, title: str, phase_name: str):
    """Plot SSSF with proper formatting."""
    # Use log scale for better visualization
    sssf_plot = sssf.copy()
    sssf_plot[sssf_plot < 1e-10] = 1e-10
    
    im = ax.imshow(sssf_plot, origin='lower', extent=[0, 1, 0, 1],
                   cmap='hot', norm=LogNorm(vmin=sssf_plot.max()/1000, 
                                            vmax=sssf_plot.max()))
    
    # Mark high symmetry points
    # Γ = (0, 0), M = (0.5, 0), K = (2/3, 1/3)
    ax.scatter([0], [0], c='cyan', s=50, marker='o', label='Γ', zorder=10)
    ax.scatter([0.5], [0], c='lime', s=50, marker='s', label='M', zorder=10)
    ax.scatter([2/3], [1/3], c='yellow', s=50, marker='^', label='K', zorder=10)
    # Also mark equivalent K points
    ax.scatter([1/3], [1/3], c='yellow', s=30, marker='^', zorder=10)
    
    ax.set_xlabel('$Q_1$ (r.l.u.)')
    ax.set_ylabel('$Q_2$ (r.l.u.)')
    ax.set_title(f'{title}\n({phase_name})', fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return im


def plot_spin_config(ax, spins: np.ndarray, positions: np.ndarray, 
                      title: str, max_spins: int = 200):
    """Plot spin configuration in real space."""
    # Subsample if too many spins
    if len(spins) > max_spins:
        idx = np.random.choice(len(spins), max_spins, replace=False)
        spins = spins[idx]
        positions = positions[idx]
    
    # Plot spin arrows
    ax.quiver(positions[:, 0], positions[:, 1], 
              spins[:, 0], spins[:, 1],
              spins[:, 2],  # Color by z-component
              cmap='coolwarm', clim=(-1, 1),
              scale=15, width=0.008)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)


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
        phases_found[phase].append({
            'dir': sample_dir,
            'spins': spins,
            'positions': positions,
            'result': result,
            'name': os.path.basename(os.path.dirname(sample_dir))
        })
    
    print(f"\nPhases found:")
    for phase, samples in phases_found.items():
        print(f"  {phase}: {len(samples)} samples")
    
    # Select one representative from each phase
    representative_phases = []
    for phase, samples in phases_found.items():
        # Pick the one with highest confidence
        best = max(samples, key=lambda x: x['result'].confidence)
        representative_phases.append((phase, best))
    
    # Sort by phase name for consistent ordering
    representative_phases.sort(key=lambda x: x[0])
    
    n_phases = len(representative_phases)
    if n_phases == 0:
        print("No phases found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, n_phases, figsize=(4*n_phases, 8))
    if n_phases == 1:
        axes = axes.reshape(2, 1)
    
    print(f"\nPlotting {n_phases} archetypical phases...")
    
    for i, (phase, sample) in enumerate(representative_phases):
        print(f"  Processing {phase} from {sample['name']}...")
        
        # Compute SSSF
        sssf = compute_sssf(sample['spins'], sample['positions'], resolution=100)
        
        # Plot SSSF (top row)
        plot_sssf(axes[0, i], sssf, sample['name'], phase)
        
        # Plot spin config (bottom row)
        plot_spin_config(axes[1, i], sample['spins'], sample['positions'], 
                         f"Real space ({phase})")
    
    # Add legend to first SSSF plot
    axes[0, 0].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/home/pc_linux/ClassicalSpin_Cpp/util/classical_solvers/archetypical_phases.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to {output_path}")
    
    plt.show()


if __name__ == '__main__':
    main()
