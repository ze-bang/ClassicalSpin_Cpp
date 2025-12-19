#!/usr/bin/env python3
"""
Analyze and visualize meron double-Q fit results with detailed energy comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys

def load_data(directory, spin_file='spins_T=0.txt', pos_file='positions.txt', 
              fitted_file='fitted_spins.txt'):
    """Load original and fitted spin configurations."""
    spins_path = os.path.join(directory, spin_file)
    pos_path = os.path.join(directory, pos_file)
    fitted_path = os.path.join(directory, fitted_file)
    
    spins_orig = np.loadtxt(spins_path)
    positions = np.loadtxt(pos_path)
    spins_fitted = np.loadtxt(fitted_path)
    
    return positions, spins_orig, spins_fitted


def compute_statistics(spins_orig, spins_fitted):
    """Compute various statistics comparing original and fitted spins."""
    # Normalize both
    norm_orig = np.linalg.norm(spins_orig, axis=1, keepdims=True)
    norm_fitted = np.linalg.norm(spins_fitted, axis=1, keepdims=True)
    
    spins_orig_n = spins_orig / (norm_orig + 1e-12)
    spins_fitted_n = spins_fitted / (norm_fitted + 1e-12)
    
    # Dot products
    dots = np.sum(spins_orig_n * spins_fitted_n, axis=1)
    
    # Component-wise differences
    diff = spins_orig_n - spins_fitted_n
    diff_x = diff[:, 0]
    diff_y = diff[:, 1]
    diff_z = diff[:, 2]
    
    # RMSE
    rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    # Angular error
    angles = np.arccos(np.clip(dots, -1, 1)) * 180 / np.pi
    
    stats = {
        'dots': dots,
        'mean_dot': np.mean(dots),
        'std_dot': np.std(dots),
        'min_dot': np.min(dots),
        'max_dot': np.max(dots),
        'rmse': rmse,
        'diff_x': diff_x,
        'diff_y': diff_y,
        'diff_z': diff_z,
        'rmse_x': np.sqrt(np.mean(diff_x**2)),
        'rmse_y': np.sqrt(np.mean(diff_y**2)),
        'rmse_z': np.sqrt(np.mean(diff_z**2)),
        'angles': angles,
        'mean_angle': np.mean(angles),
        'max_angle': np.max(angles),
    }
    
    return stats


def plot_detailed_comparison(positions, spins_orig, spins_fitted, stats, save_path=None):
    """Create comprehensive comparison plot."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Subsample for quiver plots
    step = max(1, int(np.sqrt(len(positions))) // 20)
    idx = np.arange(0, len(positions), step)
    
    # 1. Original spins (m_z colored)
    ax1 = fig.add_subplot(gs[0, 0])
    q1 = ax1.quiver(positions[idx, 0], positions[idx, 1],
                    spins_orig[idx, 0], spins_orig[idx, 1], 
                    spins_orig[idx, 2], cmap='coolwarm', pivot='mid',
                    clim=(-1, 1))
    ax1.set_title('Original Spins', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    plt.colorbar(q1, ax=ax1, label='$m_z$', fraction=0.046)
    
    # 2. Fitted spins (m_z colored)
    ax2 = fig.add_subplot(gs[0, 1])
    q2 = ax2.quiver(positions[idx, 0], positions[idx, 1],
                    spins_fitted[idx, 0], spins_fitted[idx, 1],
                    spins_fitted[idx, 2], cmap='coolwarm', pivot='mid',
                    clim=(-1, 1))
    ax2.set_title(f'Fitted Spins (RMSE={stats["rmse"]:.3f})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    plt.colorbar(q2, ax=ax2, label='$m_z$', fraction=0.046)
    
    # 3. Dot product distribution
    ax3 = fig.add_subplot(gs[0, 2])
    scatter = ax3.scatter(positions[idx, 0], positions[idx, 1],
                         c=stats['dots'][idx], cmap='RdYlGn',
                         s=20, vmin=-1, vmax=1)
    ax3.set_title(f'Overlap: $\\langle m_{{orig}} \\cdot m_{{fit}} \\rangle$ = {stats["mean_dot"]:.3f}',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal')
    plt.colorbar(scatter, ax=ax3, label='Dot product', fraction=0.046)
    
    # 4. m_z comparison scatter
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(spins_orig[:, 2], spins_fitted[:, 2], alpha=0.3, s=5)
    ax4.plot([-1, 1], [-1, 1], 'r--', lw=2, label='Perfect match')
    ax4.set_xlabel('Original $m_z$', fontsize=11)
    ax4.set_ylabel('Fitted $m_z$', fontsize=11)
    ax4.set_title(f'$m_z$ Component (RMSE={stats["rmse_z"]:.3f})', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # 5. In-plane magnitude comparison
    ax5 = fig.add_subplot(gs[1, 1])
    mag_orig = np.linalg.norm(spins_orig[:, :2], axis=1)
    mag_fitted = np.linalg.norm(spins_fitted[:, :2], axis=1)
    ax5.scatter(mag_orig, mag_fitted, alpha=0.3, s=5)
    ax5.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect match')
    ax5.set_xlabel('Original $|m_{\\perp}|$', fontsize=11)
    ax5.set_ylabel('Fitted $|m_{\\perp}|$', fontsize=11)
    ax5.set_title('In-plane Magnitude', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Angular error distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(stats['angles'], bins=50, alpha=0.7, edgecolor='black')
    ax6.axvline(stats['mean_angle'], color='r', linestyle='--', linewidth=2,
                label=f'Mean: {stats["mean_angle"]:.1f}°')
    ax6.axvline(stats['max_angle'], color='orange', linestyle='--', linewidth=2,
                label=f'Max: {stats["max_angle"]:.1f}°')
    ax6.set_xlabel('Angular error (degrees)', fontsize=11)
    ax6.set_ylabel('Count', fontsize=11)
    ax6.set_title('Angular Error Distribution', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Dot product histogram
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(stats['dots'], bins=50, alpha=0.7, edgecolor='black', color='green')
    ax7.axvline(stats['mean_dot'], color='r', linestyle='--', linewidth=2,
                label=f'Mean: {stats["mean_dot"]:.3f}')
    ax7.set_xlabel('Dot product $m_{orig} \\cdot m_{fit}$', fontsize=11)
    ax7.set_ylabel('Count', fontsize=11)
    ax7.set_title('Overlap Distribution', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Component RMSEs
    ax8 = fig.add_subplot(gs[2, 1])
    components = ['$m_x$', '$m_y$', '$m_z$', 'Total']
    rmses = [stats['rmse_x'], stats['rmse_y'], stats['rmse_z'], stats['rmse']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax8.bar(components, rmses, color=colors, alpha=0.7, edgecolor='black')
    ax8.set_ylabel('RMSE', fontsize=11)
    ax8.set_title('Component-wise RMSE', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 9. Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    summary_text = f"""
    FITTING STATISTICS
    {'='*30}
    
    Overlap (dot product):
      Mean:  {stats['mean_dot']:>8.4f}
      Std:   {stats['std_dot']:>8.4f}
      Min:   {stats['min_dot']:>8.4f}
      Max:   {stats['max_dot']:>8.4f}
    
    RMSE:
      Total: {stats['rmse']:>8.4f}
      m_x:   {stats['rmse_x']:>8.4f}
      m_y:   {stats['rmse_y']:>8.4f}
      m_z:   {stats['rmse_z']:>8.4f}
    
    Angular Error (degrees):
      Mean:  {stats['mean_angle']:>8.2f}°
      Max:   {stats['max_angle']:>8.2f}°
    
    Data points: {len(spins_orig)}
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax9.transAxes)
    
    plt.suptitle('Double-Q Meron Ansatz Fit Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved detailed analysis to {save_path}")
    
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_meron_fit.py <directory> [output_plot.png]")
        print("\nExample:")
        print("  python analyze_meron_fit.py Potential_Param_Scan/fitting_param_4_x/sample_0")
        sys.exit(1)
    
    directory = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(directory, 'fit_analysis.png')
    
    print(f"Loading data from {directory}...")
    positions, spins_orig, spins_fitted = load_data(directory)
    print(f"  Loaded {len(spins_orig)} spins")
    
    print("Computing statistics...")
    stats = compute_statistics(spins_orig, spins_fitted)
    
    print("\n" + "="*60)
    print("FIT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total spins: {len(spins_orig)}")
    print(f"\nOverlap (dot product):")
    print(f"  Mean:  {stats['mean_dot']:.6f}")
    print(f"  Std:   {stats['std_dot']:.6f}")
    print(f"  Range: [{stats['min_dot']:.6f}, {stats['max_dot']:.6f}]")
    print(f"\nRMSE:")
    print(f"  Total: {stats['rmse']:.6f}")
    print(f"  m_x:   {stats['rmse_x']:.6f}")
    print(f"  m_y:   {stats['rmse_y']:.6f}")
    print(f"  m_z:   {stats['rmse_z']:.6f}")
    print(f"\nAngular error:")
    print(f"  Mean: {stats['mean_angle']:.2f}°")
    print(f"  Max:  {stats['max_angle']:.2f}°")
    print("="*60)
    
    print(f"\nGenerating detailed comparison plot...")
    plot_detailed_comparison(positions, spins_orig, spins_fitted, stats, save_path=output_path)
    
    print(f"\nAnalysis complete!")


if __name__ == '__main__':
    main()
