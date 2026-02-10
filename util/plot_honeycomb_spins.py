#!/usr/bin/env python3
"""
Plot spin configuration on honeycomb lattice.

Usage:
    python plot_honeycomb_spins.py <spins.txt> [lattice_size]
    
Example:
    python plot_honeycomb_spins.py NCTO_strain_SA/sample_0/spins.txt 18,18,1
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os


def generate_honeycomb_positions(L1, L2, L3=1):
    """Generate honeycomb lattice positions.
    
    Honeycomb lattice has 2 atoms per unit cell:
    - Atom 0 at (0, 0, 0)
    - Atom 1 at (0, 1/sqrt(3), 0)
    
    Lattice vectors:
    - a1 = (1, 0, 0)
    - a2 = (0.5, sqrt(3)/2, 0)
    - a3 = (0, 0, 1)
    
    Args:
        L1, L2, L3: Number of unit cells in each direction
        
    Returns:
        positions: (N, 3) array of positions
        sublattice: (N,) array of sublattice indices (0 or 1)
    """
    # Lattice vectors
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2, 0.0])
    a3 = np.array([0.0, 0.0, 1.0])
    
    # Basis positions within unit cell
    basis = [
        np.array([0.0, 0.0, 0.0]),            # Sublattice A
        np.array([0.0, 1/np.sqrt(3), 0.0])    # Sublattice B
    ]
    
    positions = []
    sublattice = []
    
    for i3 in range(L3):
        for i2 in range(L2):
            for i1 in range(L1):
                R = i1 * a1 + i2 * a2 + i3 * a3
                for idx, b in enumerate(basis):
                    positions.append(R + b)
                    sublattice.append(idx)
    
    return np.array(positions), np.array(sublattice)


def local_to_global_frame(spins_local):
    """Transform spins from Kitaev local frame to global Cartesian frame.
    
    The Kitaev local frame uses:
    - x_local = (1/√6, 1/√6, -2/√6)
    - y_local = (-1/√2, 1/√2, 0)  
    - z_local = (1/√3, 1/√3, 1/√3)  (pointing along [111])
    
    Args:
        spins_local: (N, 3) array of spins in local frame
        
    Returns:
        spins_global: (N, 3) array of spins in global frame
    """
    # Transformation matrix: columns are local basis vectors in global coords
    # Local frame to global: S_global = R @ S_local
    R = np.array([
        [1/np.sqrt(6), -1/np.sqrt(2), 1/np.sqrt(3)],
        [1/np.sqrt(6),  1/np.sqrt(2), 1/np.sqrt(3)],
        [-2/np.sqrt(6), 0,            1/np.sqrt(3)]
    ])
    
    return (R @ spins_local.T).T


def plot_honeycomb_spins(spins, positions, sublattice, 
                         output_file=None, figsize=(12, 10),
                         arrow_scale=1.2):
    """Plot spin configuration on honeycomb lattice.
    
    Shows XY plane projection with arrows indicating full spin direction.
    Arrow color indicates Sz component (red = +z, blue = -z).
    
    Args:
        spins: (N, 3) array of spin vectors (in global frame)
        positions: (N, 3) array of site positions
        sublattice: (N,) array of sublattice indices
        output_file: If provided, save to this file
        figsize: Figure size
        arrow_scale: Scale factor for spin arrows
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Positions
    x, y = positions[:, 0], positions[:, 1]
    
    # Spin components
    sx, sy, sz = spins[:, 0], spins[:, 1], spins[:, 2]
    
    # Normalize to unit vectors (all same length, direction only)
    spin_mag = np.sqrt(sx**2 + sy**2 + sz**2)
    spin_mag = np.where(spin_mag > 1e-10, spin_mag, 1.0)
    sx_norm = sx / spin_mag
    sy_norm = sy / spin_mag
    
    # Normalize colors by Sz
    norm = Normalize(vmin=-1, vmax=1)
    
    # Draw honeycomb bonds first (as background)
    delta = np.array([0.0, 1/np.sqrt(3)])  # A to B within unit cell
    
    # Draw bonds (light gray background)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if sublattice[i] == 0:  # A sublattice connects to B
            # Bond within unit cell (A -> B)
            ax.plot([xi, xi + delta[0]], [yi, yi + delta[1]], 
                   'k-', alpha=0.3, linewidth=1.0, zorder=1)
    
    # Plot lattice sites as visible dots with black edge
    ax.scatter(x, y, c='white', s=50, alpha=1.0, edgecolors='black', 
               linewidth=0.8, zorder=2)
    
    # Main quiver plot - thin arrows, all same length, color = Sz
    quiver = ax.quiver(x, y, sx_norm, sy_norm,
                      sz, cmap='coolwarm', norm=norm,
                      angles='xy', scale_units='xy', scale=1/arrow_scale,
                      pivot='mid', width=0.003, headwidth=3, headlength=4,
                      headaxislength=3.5, zorder=4)
    
    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r'$S_z$', fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    
    # Labels and formatting
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('y', fontsize=13)
    ax.set_title('Honeycomb Lattice Spin Configuration', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=11)
    
    # Add a subtle grid
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Set axis limits with small padding
    pad = 0.5
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_file}")
    
    plt.show()
    return fig


def compute_structure_factor(spins, positions, kx_vals, ky_vals):
    """Compute static spin structure factor S(q).
    
    Args:
        spins: (N, 3) array of spins
        positions: (N, 2) or (N, 3) array of positions
        kx_vals: 1D array of kx values
        ky_vals: 1D array of ky values
        
    Returns:
        S_q: (len(ky_vals), len(kx_vals)) array of S(q) values
    """
    N = len(spins)
    pos_2d = positions[:, :2] if positions.shape[1] > 2 else positions
    
    S_q = np.zeros((len(ky_vals), len(kx_vals)))
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            q = np.array([kx, ky])
            phase = np.exp(1j * (pos_2d[:, 0] * q[0] + pos_2d[:, 1] * q[1]))
            S_q_vec = np.sum(spins * phase[:, np.newaxis], axis=0) / np.sqrt(N)
            S_q[j, i] = np.sum(np.abs(S_q_vec)**2)
    return S_q


def plot_structure_factor(spins_local, positions, output_file=None, figsize=(16, 16)):
    """Plot S(q) in both local and global frames, in reduced (H,K) and Cartesian (kx,ky) coordinates.
    
    Args:
        spins_local: (N, 3) array of spins in local Kitaev frame
        positions: (N, 3) array of positions
        output_file: If provided, save to this file
        figsize: Figure size
    """
    # Transform to global frame
    spins_global = local_to_global_frame(spins_local)
    
    # Reciprocal lattice vectors
    b1 = 2 * np.pi * np.array([1.0, -1.0 / np.sqrt(3.0)])
    b2 = 2 * np.pi * np.array([0.0, 2.0 / np.sqrt(3.0)])
    
    # High symmetry points in reduced coordinates (H, K)
    M1 = (0.5, 0)
    M2 = (0, 0.5)
    M3 = (0.5, 0.5)
    K_pt = (1/3, 2/3)
    Kp_pt = (2/3, 1/3)
    
    # High symmetry points in Cartesian coordinates
    Gamma_cart = np.array([0.0, 0.0])
    M1_cart = 0.5 * b1
    M2_cart = 0.5 * b2
    M3_cart = 0.5 * (b1 + b2)
    K_cart = (2*b1 + b2) / 3
    Kp_cart = (b1 + 2*b2) / 3
    
    # Compute S(q) on reduced coordinate grid
    n_grid = 150
    h_vals = np.linspace(-0.2, 1.2, n_grid)
    k_vals = np.linspace(-0.2, 1.2, n_grid)
    H_grid, K_grid = np.meshgrid(h_vals, k_vals)
    
    print('Computing S(q) in reduced coordinates...')
    S_q_local = np.zeros((n_grid, n_grid))
    S_q_global = np.zeros((n_grid, n_grid))
    
    pos_2d = positions[:, :2]
    N = len(spins_local)
    
    for i, h in enumerate(h_vals):
        for j, k in enumerate(k_vals):
            q = h * b1 + k * b2
            phase = np.exp(1j * (pos_2d[:, 0] * q[0] + pos_2d[:, 1] * q[1]))
            
            # Local frame
            S_q_vec = np.sum(spins_local * phase[:, np.newaxis], axis=0) / np.sqrt(N)
            S_q_local[j, i] = np.sum(np.abs(S_q_vec)**2)
            
            # Global frame
            S_q_vec = np.sum(spins_global * phase[:, np.newaxis], axis=0) / np.sqrt(N)
            S_q_global[j, i] = np.sum(np.abs(S_q_vec)**2)
    
    # Compute S(q) on Cartesian grid
    print('Computing S(q) in Cartesian coordinates...')
    kmax = 8.0
    n_grid_cart = 150
    kx_vals = np.linspace(-kmax, kmax, n_grid_cart)
    ky_vals = np.linspace(-kmax, kmax, n_grid_cart)
    Kx_grid, Ky_grid = np.meshgrid(kx_vals, ky_vals)
    
    S_q_local_cart = np.zeros((n_grid_cart, n_grid_cart))
    S_q_global_cart = np.zeros((n_grid_cart, n_grid_cart))
    
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            q = np.array([kx, ky])
            phase = np.exp(1j * (pos_2d[:, 0] * q[0] + pos_2d[:, 1] * q[1]))
            
            # Local frame
            S_q_vec = np.sum(spins_local * phase[:, np.newaxis], axis=0) / np.sqrt(N)
            S_q_local_cart[j, i] = np.sum(np.abs(S_q_vec)**2)
            
            # Global frame
            S_q_vec = np.sum(spins_global * phase[:, np.newaxis], axis=0) / np.sqrt(N)
            S_q_global_cart[j, i] = np.sum(np.abs(S_q_vec)**2)
    
    print('Done computing S(q)')
    
    # Create figure with 3x2 layout
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # High symmetry points for annotation (reduced coords)
    sym_pts = {'Γ': (0, 0), 'M₁': M1, 'M₂': M2, 'M₃': M3, 'K': K_pt, "K'": Kp_pt}
    
    # High symmetry points for Cartesian plots
    sym_pts_cart = {'Γ': Gamma_cart, 'M₁': M1_cart, 'M₂': M2_cart, 'M₃': M3_cart, 
                    'K': K_cart, "K'": Kp_cart}
    
    # BZ boundary for Cartesian plots
    bz_corners = [K_cart, M3_cart, Kp_cart, -K_cart, -M3_cart, -Kp_cart, K_cart]
    bz_x = [p[0] for p in bz_corners]
    bz_y = [p[1] for p in bz_corners]
    
    # Panel 1: S(q) in local frame (reduced coords)
    ax = axes[0, 0]
    im = ax.pcolormesh(H_grid, K_grid, S_q_local, shading='auto', cmap='hot')
    plt.colorbar(im, ax=ax, label='S(q)')
    for name, (h, k) in sym_pts.items():
        ax.plot(h, k, 'wo', markersize=6, markeredgecolor='cyan', markeredgewidth=1.5)
        ax.annotate(name, (h, k), xytext=(3, 3), textcoords='offset points', color='cyan', fontsize=9)
    ax.set_xlabel('H (r.l.u.)', fontsize=11)
    ax.set_ylabel('K (r.l.u.)', fontsize=11)
    ax.set_title('S(q) LOCAL frame (H, K)', fontsize=12)
    ax.set_aspect('equal')
    
    # Panel 2: S(q) in global frame (reduced coords)
    ax = axes[0, 1]
    im = ax.pcolormesh(H_grid, K_grid, S_q_global, shading='auto', cmap='hot')
    plt.colorbar(im, ax=ax, label='S(q)')
    for name, (h, k) in sym_pts.items():
        ax.plot(h, k, 'wo', markersize=6, markeredgecolor='cyan', markeredgewidth=1.5)
        ax.annotate(name, (h, k), xytext=(3, 3), textcoords='offset points', color='cyan', fontsize=9)
    ax.set_xlabel('H (r.l.u.)', fontsize=11)
    ax.set_ylabel('K (r.l.u.)', fontsize=11)
    ax.set_title('S(q) GLOBAL frame (H, K)', fontsize=12)
    ax.set_aspect('equal')
    
    # Panel 3: S(q) in local frame (Cartesian)
    ax = axes[1, 0]
    im = ax.pcolormesh(Kx_grid, Ky_grid, S_q_local_cart, shading='auto', cmap='hot')
    plt.colorbar(im, ax=ax, label='S(q)')
    for name, pt in sym_pts_cart.items():
        ax.plot(pt[0], pt[1], 'wo', markersize=5, markeredgecolor='cyan', markeredgewidth=1.5)
        ax.annotate(name, pt, xytext=(3, 3), textcoords='offset points', color='cyan', fontsize=8)
    ax.plot(bz_x, bz_y, 'c--', linewidth=1, alpha=0.7)
    ax.set_xlabel(r'$k_x$', fontsize=11)
    ax.set_ylabel(r'$k_y$', fontsize=11)
    ax.set_title(r'S(q) LOCAL frame ($k_x$, $k_y$)', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(-kmax, kmax)
    ax.set_ylim(-kmax, kmax)
    
    # Panel 4: S(q) in global frame (Cartesian)
    ax = axes[1, 1]
    im = ax.pcolormesh(Kx_grid, Ky_grid, S_q_global_cart, shading='auto', cmap='hot')
    plt.colorbar(im, ax=ax, label='S(q)')
    for name, pt in sym_pts_cart.items():
        ax.plot(pt[0], pt[1], 'wo', markersize=5, markeredgecolor='cyan', markeredgewidth=1.5)
        ax.annotate(name, pt, xytext=(3, 3), textcoords='offset points', color='cyan', fontsize=8)
    ax.plot(bz_x, bz_y, 'c--', linewidth=1, alpha=0.7)
    ax.set_xlabel(r'$k_x$', fontsize=11)
    ax.set_ylabel(r'$k_y$', fontsize=11)
    ax.set_title(r'S(q) GLOBAL frame ($k_x$, $k_y$)', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(-kmax, kmax)
    ax.set_ylim(-kmax, kmax)
    
    # Panel 5: Real-space spins (local frame)
    ax = axes[2, 0]
    n_show = 6
    mask = (positions[:, 0] < n_show) & (positions[:, 1] < n_show * np.sqrt(3)/2)
    pos_show = positions[mask]
    spin_show = spins_local[mask]
    colors = plt.cm.coolwarm((spin_show[:, 2] + 1) / 2)
    ax.quiver(pos_show[:, 0], pos_show[:, 1], spin_show[:, 0], spin_show[:, 1],
              color=colors, scale=15, width=0.008)
    sc = ax.scatter(pos_show[:, 0], pos_show[:, 1], c=spin_show[:, 2], cmap='coolwarm', 
                    s=30, vmin=-1, vmax=1, edgecolors='black', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Sz'")
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title('Real-space spins (LOCAL frame)', fontsize=12)
    ax.set_aspect('equal')
    
    # Panel 6: Real-space spins (global frame)
    ax = axes[2, 1]
    spin_show_g = spins_global[mask]
    colors = plt.cm.coolwarm((spin_show_g[:, 2] + 1) / 2)
    ax.quiver(pos_show[:, 0], pos_show[:, 1], spin_show_g[:, 0], spin_show_g[:, 1],
              color=colors, scale=15, width=0.008)
    sc = ax.scatter(pos_show[:, 0], pos_show[:, 1], c=spin_show_g[:, 2], cmap='coolwarm', 
                    s=30, vmin=-1, vmax=1, edgecolors='black', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='Sz')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title('Real-space spins (GLOBAL frame)', fontsize=12)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.show()
    
    # Print peak intensities at M points
    print("\n=== S(q) at M points ===")
    for name, (h, k) in [('M₁', M1), ('M₂', M2), ('M₃', M3)]:
        q = h * b1 + k * b2
        phase = np.exp(1j * (pos_2d[:, 0] * q[0] + pos_2d[:, 1] * q[1]))
        S_local = np.sum(np.abs(np.sum(spins_local * phase[:, np.newaxis], axis=0) / np.sqrt(N))**2)
        S_global = np.sum(np.abs(np.sum(spins_global * phase[:, np.newaxis], axis=0) / np.sqrt(N))**2)
        print(f"  {name} (H={h}, K={k}): Local={S_local:.1f}, Global={S_global:.1f}")
    
    # Determine order type
    M_intensities = []
    for h, k in [M1, M2, M3]:
        q = h * b1 + k * b2
        phase = np.exp(1j * (pos_2d[:, 0] * q[0] + pos_2d[:, 1] * q[1]))
        S = np.sum(np.abs(np.sum(spins_local * phase[:, np.newaxis], axis=0) / np.sqrt(N))**2)
        M_intensities.append(S)
    
    max_M = max(M_intensities)
    significant = [I > 0.1 * max_M for I in M_intensities]
    n_significant = sum(significant)
    
    print(f"\n=== Order Classification ===")
    if n_significant == 1:
        which = ['M₁', 'M₂', 'M₃'][M_intensities.index(max_M)]
        print(f"  Single-Q Zigzag at {which}")
    elif n_significant == 2:
        print(f"  Double-Q state")
    elif n_significant == 3:
        print(f"  Triple-Q state")
    else:
        print(f"  Unknown/disordered")
    
    return fig


def plot_spin_order_analysis(spins, positions, sublattice, figsize=(14, 10)):
    """Analyze and plot spin order patterns.
    
    Args:
        spins: (N, 3) spins in global frame
        positions: (N, 3) positions
        sublattice: (N,) sublattice indices
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get sublattice masks
    mask_A = sublattice == 0
    mask_B = sublattice == 1
    
    # 1. Histogram of spin components
    ax1 = axes[0, 0]
    ax1.hist(spins[:, 0], bins=30, alpha=0.7, label=r'$S_x$', color='red')
    ax1.hist(spins[:, 1], bins=30, alpha=0.7, label=r'$S_y$', color='green')
    ax1.hist(spins[:, 2], bins=30, alpha=0.7, label=r'$S_z$', color='blue')
    ax1.set_xlabel('Spin component')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Spin Components (Global)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sublattice magnetization comparison
    ax2 = axes[0, 1]
    M_A = np.mean(spins[mask_A], axis=0)
    M_B = np.mean(spins[mask_B], axis=0)
    M_total = np.mean(spins, axis=0)
    M_stag = (M_A - M_B) / 2
    
    x_pos = np.arange(3)
    width = 0.2
    ax2.bar(x_pos - 1.5*width, M_A, width, label='Sublattice A', color='blue')
    ax2.bar(x_pos - 0.5*width, M_B, width, label='Sublattice B', color='red')
    ax2.bar(x_pos + 0.5*width, M_total, width, label='Total', color='green')
    ax2.bar(x_pos + 1.5*width, M_stag, width, label='Staggered', color='purple')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([r'$M_x$', r'$M_y$', r'$M_z$'])
    ax2.set_ylabel('Magnetization')
    ax2.set_title('Sublattice Magnetizations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Spin-spin correlation along x direction
    ax3 = axes[1, 0]
    # Get unique y values and average over them
    unique_y = np.unique(np.round(positions[:, 1], decimals=4))
    if len(unique_y) > 1:
        # Pick a row of sites
        y_mid = unique_y[len(unique_y)//2]
        row_mask = np.abs(positions[:, 1] - y_mid) < 0.1
        row_x = positions[row_mask, 0]
        row_sz = spins[row_mask, 2]
        sort_idx = np.argsort(row_x)
        ax3.plot(row_x[sort_idx], row_sz[sort_idx], 'o-', markersize=6)
        ax3.set_xlabel('x position')
        ax3.set_ylabel(r'$S_z$')
        ax3.set_title(f'Spin profile along x (y ≈ {y_mid:.2f})')
        ax3.grid(True, alpha=0.3)
    
    # 4. Polar plot of spin directions (stereographic)
    ax4 = axes[1, 1]
    # Project spin directions onto unit sphere
    theta = np.arctan2(spins[:, 1], spins[:, 0])  # azimuthal angle
    phi = np.arccos(np.clip(spins[:, 2], -1, 1))  # polar angle from z
    
    ax4.scatter(theta[mask_A], phi[mask_A], c='blue', s=20, alpha=0.5, label='Sublattice A')
    ax4.scatter(theta[mask_B], phi[mask_B], c='red', s=20, alpha=0.5, label='Sublattice B')
    ax4.set_xlabel(r'$\theta$ (azimuthal)')
    ax4.set_ylabel(r'$\phi$ (polar from z)')
    ax4.set_title('Spin Direction Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print summary
    print("\n=== Spin Configuration Analysis ===")
    print(f"Total spins: {len(spins)}")
    print(f"  Sublattice A: {np.sum(mask_A)}")
    print(f"  Sublattice B: {np.sum(mask_B)}")
    print(f"\nMagnetization (global frame):")
    print(f"  Total:     ({M_total[0]:+.4f}, {M_total[1]:+.4f}, {M_total[2]:+.4f})")
    print(f"  |M_total|: {np.linalg.norm(M_total):.4f}")
    print(f"  Sublat A:  ({M_A[0]:+.4f}, {M_A[1]:+.4f}, {M_A[2]:+.4f})")
    print(f"  Sublat B:  ({M_B[0]:+.4f}, {M_B[1]:+.4f}, {M_B[2]:+.4f})")
    print(f"  Staggered: ({M_stag[0]:+.4f}, {M_stag[1]:+.4f}, {M_stag[2]:+.4f})")
    print(f"  |M_stag|:  {np.linalg.norm(M_stag):.4f}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot spin configuration on honeycomb lattice')
    parser.add_argument('spins_file', type=str,
                       help='Path to spins.txt file')
    parser.add_argument('--lattice-size', type=str, default='18,18,1',
                       help='Lattice dimensions L1,L2,L3 (default: 18,18,1)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for figure')
    parser.add_argument('--analyze', action='store_true',
                       help='Show spin order analysis')
    parser.add_argument('--structure-factor', action='store_true',
                       help='Plot S(q) structure factor in local and global frames')
    parser.add_argument('--local-frame', action='store_true',
                       help='Keep spins in local Kitaev frame (no transformation)')
    parser.add_argument('--arrow-scale', type=float, default=1.2,
                       help='Scale factor for spin arrows')
    
    args = parser.parse_args()
    
    # Parse lattice size
    L = [int(x) for x in args.lattice_size.split(',')]
    if len(L) == 2:
        L.append(1)
    L1, L2, L3 = L
    
    print(f"Loading spins from: {args.spins_file}")
    print(f"Lattice size: {L1} x {L2} x {L3}")
    
    # Load spins
    spins = np.loadtxt(args.spins_file)
    print(f"Loaded {len(spins)} spins")
    
    # Generate positions
    positions, sublattice = generate_honeycomb_positions(L1, L2, L3)
    
    expected_n = L1 * L2 * L3 * 2
    if len(spins) != expected_n:
        print(f"Warning: Expected {expected_n} spins but got {len(spins)}")
        print("  Check your lattice size argument.")
    
    # Transform to global frame if requested
    if args.local_frame:
        print("Keeping spins in local Kitaev frame")
        spins_plot = spins
    else:
        print("Transforming spins to global frame")
        spins_plot = local_to_global_frame(spins)
    
    # Plot
    output_file = args.output
    if output_file is None:
        base = os.path.splitext(args.spins_file)[0]
        output_file = f"{base}_plot.png"
    
    plot_honeycomb_spins(spins_plot, positions, sublattice,
                        output_file=output_file,
                        arrow_scale=args.arrow_scale)
    
    if args.structure_factor:
        sf_output = output_file.replace('.png', '_Sq.png') if output_file else None
        plot_structure_factor(spins, positions, output_file=sf_output)
    
    if args.analyze:
        plot_spin_order_analysis(spins_plot, positions, sublattice)


if __name__ == '__main__':
    main()
