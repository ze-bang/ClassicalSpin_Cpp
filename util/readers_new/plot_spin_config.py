"""
NCTO Honeycomb Lattice Spin Configuration Visualization
=========================================================

This script visualizes spin configurations from the NCTO (honeycomb lattice)
simulations. It reads spin and position files and creates various visualizations:

1. 2D quiver plot with arrows showing in-plane spin components in LOCAL Kitaev frame
2. 2D quiver plot with arrows showing in-plane spin components in GLOBAL Cartesian frame
3. Side-by-side comparison of Local vs Global frame (XY projections)
4. 3D vector plot showing full spin orientations
5. Sublattice-resolved views (A and B sublattices)

Frame Transformation:
    The spin configurations are stored in the LOCAL Kitaev frame, where the
    x, y, z axes are aligned with the three types of Kitaev bonds on the 
    honeycomb lattice. The transformation to GLOBAL Cartesian frame uses:
    
    kitaevLocal = [[1/√6, 1/√6, -2/√6],    # x_global
                   [-1/√2, 1/√2, 0],        # y_global  
                   [1/√3, 1/√3, 1/√3]]      # z_global
    
    S_global = kitaevLocal @ S_local

Usage:
    python plot_spin_config.py <directory>
    
Example:
    python plot_spin_config.py ../../NCTO_sa/sample_0/
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import os
import sys
import argparse


# Kitaev local to global frame transformation matrix
# Transforms spins from local Kitaev frame to global Cartesian frame
# S_global = kitaevLocal @ S_local
KITAEV_LOCAL_TO_GLOBAL = np.array([
    [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],   # x_global
    [-1/np.sqrt(2), 1/np.sqrt(2), 0],              # y_global
    [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]     # z_global
])


def transform_local_to_global(spins):
    """Transform spins from local Kitaev frame to global Cartesian frame.
    
    In the Kitaev model, the local frame has:
    - x, y, z axes aligned with the three types of Kitaev bonds
    
    The global frame is the standard Cartesian coordinate system.
    
    Args:
        spins: (N, 3) array of spin vectors in local Kitaev frame
        
    Returns:
        spins_global: (N, 3) array of spin vectors in global frame
    """
    # Transform each spin: S_global = kitaevLocal @ S_local
    # Using einsum: for each spin i, sum over local component j
    spins_global = np.einsum('ij,nj->ni', KITAEV_LOCAL_TO_GLOBAL, spins)
    return spins_global


def generate_honeycomb_positions(n_sites):
    """Generate honeycomb lattice site positions from the number of sites.
    
    This function infers the lattice dimensions from the number of sites
    and generates positions matching the StrainPhononLattice geometry.
    
    The honeycomb lattice has:
    - Lattice vectors: a1 = (1, 0, 0), a2 = (0.5, sqrt(3)/2, 0)
    - 2 atoms per unit cell at: r_A = (0, 0, 0), r_B = (0, 1/sqrt(3), 0)
    - Site indexing: ((i * dim2 + j) * dim3 + k) * 2 + atom
    
    Args:
        n_sites: Total number of sites (must be even, = 2 * dim1 * dim2 * dim3)
        
    Returns:
        positions: (n_sites, 3) array of site positions
    """
    if n_sites % 2 != 0:
        raise ValueError(f"Number of sites ({n_sites}) must be even for honeycomb lattice")
    
    n_unit_cells = n_sites // 2
    
    # Try to infer dimensions - assume square-ish lattice
    dim3 = 1  # Usually 1 for 2D simulations
    
    # Find factors for dim1 * dim2 = n_unit_cells
    sqrt_n = int(np.sqrt(n_unit_cells))
    dim1, dim2 = None, None
    
    for d1 in range(sqrt_n, 0, -1):
        if n_unit_cells % d1 == 0:
            dim1 = d1
            dim2 = n_unit_cells // d1
            break
    
    if dim1 is None:
        raise ValueError(f"Could not factorize {n_unit_cells} unit cells")
    
    print(f"  Inferred lattice dimensions: {dim1} x {dim2} x {dim3} (total {n_sites} sites)")
    
    # Generate positions
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0)/2.0, 0.0])
    a3 = np.array([0.0, 0.0, 1.0])
    
    pos0 = np.array([0.0, 0.0, 0.0])  # Sublattice A
    pos1 = np.array([0.0, 1.0/np.sqrt(3.0), 0.0])  # Sublattice B
    
    positions = np.zeros((n_sites, 3))
    site_idx = 0
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                positions[site_idx] = pos0 + i*a1 + j*a2 + k*a3
                site_idx += 1
                positions[site_idx] = pos1 + i*a1 + j*a2 + k*a3
                site_idx += 1
    
    return positions


def load_spin_config(directory):
    """Load spin configuration and positions from text files.
    
    If positions.txt is not found, positions are inferred from the 
    StrainPhononLattice honeycomb geometry.
    
    Args:
        directory: Path to directory containing spins.txt (and optionally positions.txt)
        
    Returns:
        positions: (N, 3) array of site positions
        spins: (N, 3) array of spin vectors
    """
    # Look for spin config file
    spin_file = os.path.join(directory, 'spins.txt')
    if not os.path.exists(spin_file):
        # Try temperature-specific files
        import glob
        spin_pattern = os.path.join(directory, 'spins_T=*.txt')
        spin_files_T = sorted(glob.glob(spin_pattern))
        if spin_files_T:
            spin_file = spin_files_T[0]  # Use lowest temperature
        else:
            raise FileNotFoundError(f"No spin file found in: {directory}")
    
    spins = np.loadtxt(spin_file)
    n_sites = len(spins)
    
    # Load or generate positions
    pos_file = os.path.join(directory, 'positions.txt')
    if os.path.exists(pos_file):
        positions = np.loadtxt(pos_file)
        print(f"Loaded {n_sites} spins from {os.path.basename(spin_file)}")
    else:
        print(f"Loaded {n_sites} spins from {os.path.basename(spin_file)}")
        print(f"  Position file not found, generating honeycomb positions...")
        positions = generate_honeycomb_positions(n_sites)
    
    print(f"  Spin magnitude range: [{np.linalg.norm(spins, axis=1).min():.4f}, {np.linalg.norm(spins, axis=1).max():.4f}]")
    print(f"  Position range: x=[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
          f"y=[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")
    
    return positions, spins

def identify_sublattices(positions):
    """Identify A and B sublattices based on position pattern.
    
    On honeycomb lattice, sublattice alternates every other site.
    
    Args:
        positions: (N, 3) array of positions
        
    Returns:
        sublattice_A: boolean mask for sublattice A
        sublattice_B: boolean mask for sublattice B
    """
    N = len(positions)
    sublattice_A = np.array([i % 2 == 0 for i in range(N)])
    sublattice_B = ~sublattice_A
    return sublattice_A, sublattice_B


def plot_spin_config_2d(positions, spins, title="Spin Configuration", 
                        ax=None, scale=1.0, color_by='sz'):
    """Plot 2D quiver plot of spin configuration.
    
    Args:
        positions: (N, 3) array of positions
        spins: (N, 3) array of spin vectors
        title: Plot title
        ax: Matplotlib axes (created if None)
        scale: Arrow scale factor
        color_by: 'sz' for z-component, 'sublattice' for sublattice coloring
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.get_figure()
    
    x, y = positions[:, 0], positions[:, 1]
    sx, sy, sz = spins[:, 0], spins[:, 1], spins[:, 2]
    
    # Normalize in-plane components to get fixed arrow length (direction only)
    in_plane_magnitude = np.sqrt(sx**2 + sy**2)
    # Avoid division by zero for spins pointing purely in z direction
    in_plane_magnitude = np.where(in_plane_magnitude > 1e-10, in_plane_magnitude, 1.0)
    sx_norm = sx / in_plane_magnitude
    sy_norm = sy / in_plane_magnitude
    
    if color_by == 'sz':
        # Color by Sz component
        colors = sz
        cmap = 'coolwarm'
        norm = Normalize(vmin=-1, vmax=1)
        label = r'$S_z$'
    elif color_by == 'sublattice':
        # Color by sublattice
        sublattice_A, sublattice_B = identify_sublattices(positions)
        colors = np.where(sublattice_A, 1, 0)
        cmap = 'RdYlBu'
        norm = Normalize(vmin=0, vmax=1)
        label = 'Sublattice (A=red, B=blue)'
    else:
        colors = np.linalg.norm(spins, axis=1)
        cmap = 'viridis'
        norm = Normalize(vmin=colors.min(), vmax=colors.max())
        label = r'$|S|$'
    
    # Plot in-plane arrows with fixed length (normalized direction)
    quiver = ax.quiver(x, y, sx_norm, sy_norm, colors, cmap=cmap, norm=norm,
                       scale=scale, scale_units='xy', angles='xy',
                       pivot='middle', width=0.005)
    
    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
    cbar.set_label(label, fontsize=12)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_spin_config_3d(positions, spins, title="3D Spin Configuration", 
                        ax=None, arrow_length=0.3):
    """Plot 3D vector field of spin configuration.
    
    Args:
        positions: (N, 3) array of positions
        spins: (N, 3) array of spin vectors  
        title: Plot title
        ax: Matplotlib 3D axes (created if None)
        arrow_length: Length scale for arrows
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
    
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    sx, sy, sz = spins[:, 0], spins[:, 1], spins[:, 2]
    
    # Color by Sz
    colors = sz
    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.cm.coolwarm
    
    # Plot arrows
    ax.quiver(x, y, z, sx, sy, sz, 
              length=arrow_length, normalize=True,
              colors=cmap(norm(colors)), arrow_length_ratio=0.3)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add colorbar manually
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(r'$S_z$', fontsize=12)
    
    return fig, ax


def plot_sublattice_comparison(positions, spins, title="Sublattice Comparison"):
    """Plot spin configurations for A and B sublattices separately.
    
    Args:
        positions: (N, 3) array of positions
        spins: (N, 3) array of spin vectors
        title: Plot title
        
    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    sublattice_A, sublattice_B = identify_sublattices(positions)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sublattice A
    ax = axes[0]
    x, y = positions[sublattice_A, 0], positions[sublattice_A, 1]
    sx, sy, sz = spins[sublattice_A, 0], spins[sublattice_A, 1], spins[sublattice_A, 2]
    quiver = ax.quiver(x, y, sx, sy, sz, cmap='coolwarm', 
                       scale=15, pivot='middle', width=0.008)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Sublattice A')
    ax.set_aspect('equal')
    plt.colorbar(quiver, ax=ax, label=r'$S_z$')
    
    # Sublattice B
    ax = axes[1]
    x, y = positions[sublattice_B, 0], positions[sublattice_B, 1]
    sx, sy, sz = spins[sublattice_B, 0], spins[sublattice_B, 1], spins[sublattice_B, 2]
    quiver = ax.quiver(x, y, sx, sy, sz, cmap='coolwarm',
                       scale=15, pivot='middle', width=0.008)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Sublattice B')
    ax.set_aspect('equal')
    plt.colorbar(quiver, ax=ax, label=r'$S_z$')
    
    # Both sublattices with different markers
    ax = axes[2]
    x_A, y_A = positions[sublattice_A, 0], positions[sublattice_A, 1]
    x_B, y_B = positions[sublattice_B, 0], positions[sublattice_B, 1]
    sx_A, sy_A = spins[sublattice_A, 0], spins[sublattice_A, 1]
    sx_B, sy_B = spins[sublattice_B, 0], spins[sublattice_B, 1]
    sz_A, sz_B = spins[sublattice_A, 2], spins[sublattice_B, 2]
    
    ax.quiver(x_A, y_A, sx_A, sy_A, sz_A, cmap='Reds', 
              scale=15, pivot='middle', width=0.006, alpha=0.8)
    ax.quiver(x_B, y_B, sx_B, sy_B, sz_B, cmap='Blues',
              scale=15, pivot='middle', width=0.006, alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Both Sublattices (A=red, B=blue)')
    ax.set_aspect('equal')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig, axes


def plot_local_vs_global_xy(positions, spins, title="Local vs Global Frame (XY Projection)", 
                            scale=15.0):
    """Plot XY projections of spins in both local Kitaev and global frames.
    
    Args:
        positions: (N, 3) array of positions
        spins: (N, 3) array of spin vectors in LOCAL Kitaev frame
        title: Plot title
        scale: Arrow scale factor for quiver plots
        
    Returns:
        fig, axes: Matplotlib figure and axes array
    """
    # Transform spins to global frame
    spins_global = transform_local_to_global(spins)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    x, y = positions[:, 0], positions[:, 1]
    
    # Local frame (left panel)
    ax = axes[0]
    sx_local, sy_local, sz_local = spins[:, 0], spins[:, 1], spins[:, 2]
    
    # Normalize in-plane components to get fixed arrow length (direction only)
    in_plane_mag_local = np.sqrt(sx_local**2 + sy_local**2)
    in_plane_mag_local = np.where(in_plane_mag_local > 1e-10, in_plane_mag_local, 1.0)
    sx_local_norm = sx_local / in_plane_mag_local
    sy_local_norm = sy_local / in_plane_mag_local
    
    norm = Normalize(vmin=-1, vmax=1)
    quiver_local = ax.quiver(x, y, sx_local_norm, sy_local_norm, sz_local, cmap='coolwarm', norm=norm,
                              scale=scale, scale_units='xy', angles='xy',
                              pivot='middle', width=0.005)
    cbar = plt.colorbar(quiver_local, ax=ax, shrink=0.8)
    cbar.set_label(r'$S_z^{local}$', fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Local Kitaev Frame (XY Projection)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Global frame (right panel)
    ax = axes[1]
    sx_global, sy_global, sz_global = spins_global[:, 0], spins_global[:, 1], spins_global[:, 2]
    
    # Normalize in-plane components to get fixed arrow length (direction only)
    in_plane_mag_global = np.sqrt(sx_global**2 + sy_global**2)
    in_plane_mag_global = np.where(in_plane_mag_global > 1e-10, in_plane_mag_global, 1.0)
    sx_global_norm = sx_global / in_plane_mag_global
    sy_global_norm = sy_global / in_plane_mag_global
    
    quiver_global = ax.quiver(x, y, sx_global_norm, sy_global_norm, sz_global, cmap='coolwarm', norm=norm,
                               scale=scale, scale_units='xy', angles='xy',
                               pivot='middle', width=0.005)
    cbar = plt.colorbar(quiver_global, ax=ax, shrink=0.8)
    cbar.set_label(r'$S_z^{global}$', fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Global Cartesian Frame (XY Projection)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig, axes


def plot_global_frame_2d(positions, spins, title="Global Frame Spin Configuration (XY Projection)", 
                         ax=None, scale=None, color_by='sz'):
    """Plot 2D quiver plot of spin configuration in global Cartesian frame.
    
    Args:
        positions: (N, 3) array of positions
        spins: (N, 3) array of spin vectors in LOCAL Kitaev frame
        title: Plot title
        ax: Matplotlib axes (created if None)
        scale: Arrow scale factor (auto-calculated if None)
        color_by: 'sz' for z-component coloring
        
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    # Transform spins to global frame
    spins_global = transform_local_to_global(spins)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.get_figure()
    
    x, y = positions[:, 0], positions[:, 1]
    sx, sy, sz = spins_global[:, 0], spins_global[:, 1], spins_global[:, 2]
    
    # Normalize in-plane components to get fixed arrow length (direction only)
    in_plane_magnitude = np.sqrt(sx**2 + sy**2)
    # Avoid division by zero for spins pointing purely in z direction
    in_plane_magnitude = np.where(in_plane_magnitude > 1e-10, in_plane_magnitude, 1.0)
    sx_norm = sx / in_plane_magnitude
    sy_norm = sy / in_plane_magnitude
    
    if color_by == 'sz':
        colors = sz
        cmap = 'coolwarm'
        norm = Normalize(vmin=-1, vmax=1)
        label = r'$S_z^{global}$'
    else:
        colors = np.linalg.norm(spins_global, axis=1)
        cmap = 'viridis'
        norm = Normalize(vmin=colors.min(), vmax=colors.max())
        label = r'$|S|$'
    
    # Auto-calculate scale if not provided: make arrows ~0.8 units long for unit spin
    if scale is None:
        scale = 1.2
    
    quiver = ax.quiver(x, y, sx_norm, sy_norm, colors, cmap=cmap, norm=norm,
                       scale=scale, scale_units='xy', angles='xy',
                       pivot='middle', width=0.008)
    
    cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
    cbar.set_label(label, fontsize=12)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def compute_order_parameters(positions, spins):
    """Compute various order parameters for the spin configuration.
    
    Args:
        positions: (N, 3) array of positions
        spins: (N, 3) array of spin vectors
        
    Returns:
        dict: Dictionary of order parameters
    """
    N = len(spins)
    sublattice_A, sublattice_B = identify_sublattices(positions)
    
    # Total magnetization
    M_total = np.mean(spins, axis=0)
    
    # Staggered magnetization (Néel order)
    signs = np.where(sublattice_A, 1, -1)[:, np.newaxis]
    M_staggered = np.mean(signs * spins, axis=0)
    
    # Sublattice magnetizations
    M_A = np.mean(spins[sublattice_A], axis=0)
    M_B = np.mean(spins[sublattice_B], axis=0)
    
    # Spin length statistics
    spin_lengths = np.linalg.norm(spins, axis=1)
    
    return {
        'M_total': M_total,
        '|M_total|': np.linalg.norm(M_total),
        'M_staggered': M_staggered,
        '|M_staggered|': np.linalg.norm(M_staggered),
        'M_A': M_A,
        '|M_A|': np.linalg.norm(M_A),
        'M_B': M_B,
        '|M_B|': np.linalg.norm(M_B),
        'avg_spin_length': np.mean(spin_lengths),
        'std_spin_length': np.std(spin_lengths),
    }


def print_order_parameters(order_params):
    """Print order parameters in a formatted way."""
    print("\n" + "="*60)
    print("ORDER PARAMETERS")
    print("="*60)
    print(f"Total Magnetization:     M = ({order_params['M_total'][0]:.4f}, "
          f"{order_params['M_total'][1]:.4f}, {order_params['M_total'][2]:.4f})")
    print(f"                        |M| = {order_params['|M_total|']:.4f}")
    print(f"Staggered Magnetization: N = ({order_params['M_staggered'][0]:.4f}, "
          f"{order_params['M_staggered'][1]:.4f}, {order_params['M_staggered'][2]:.4f})")
    print(f"                        |N| = {order_params['|M_staggered|']:.4f}")
    print(f"Sublattice A:           MA = ({order_params['M_A'][0]:.4f}, "
          f"{order_params['M_A'][1]:.4f}, {order_params['M_A'][2]:.4f})")
    print(f"                       |MA| = {order_params['|M_A|']:.4f}")
    print(f"Sublattice B:           MB = ({order_params['M_B'][0]:.4f}, "
          f"{order_params['M_B'][1]:.4f}, {order_params['M_B'][2]:.4f})")
    print(f"                       |MB| = {order_params['|M_B|']:.4f}")
    print(f"Spin length:          <|S|> = {order_params['avg_spin_length']:.4f} ± "
          f"{order_params['std_spin_length']:.4f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize NCTO honeycomb lattice spin configurations')
    parser.add_argument('directory', type=str, nargs='?', 
                        default='../../NCTO_sa/sample_0/',
                        help='Directory containing spins.txt and positions.txt')
    parser.add_argument('--save', '-s', type=str, default=None,
                        help='Save figures with this prefix (e.g., "spin_config")')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figures (useful for batch processing)')
    parser.add_argument('--scale', type=float, default=1.2,
                        help='Arrow scale for quiver plots (default: 1.2, smaller = larger arrows)')
    
    args = parser.parse_args()
    
    # Load data
    try:
        positions, spins = load_spin_config(args.directory)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Compute and print order parameters (in local frame)
    print("\n--- Order Parameters in LOCAL Kitaev Frame ---")
    order_params_local = compute_order_parameters(positions, spins)
    print_order_parameters(order_params_local)
    
    # Also compute order parameters in global frame
    spins_global = transform_local_to_global(spins)
    print("--- Order Parameters in GLOBAL Cartesian Frame ---")
    order_params_global = compute_order_parameters(positions, spins_global)
    print_order_parameters(order_params_global)
    
    # Create plots
    print("Creating visualizations...")
    
    # Global frame 2D plot (XY projection, colored by Sz)
    fig, ax = plot_global_frame_2d(positions, spins,
                                    title="Global Cartesian Frame - Spin Configuration (XY Projection)",
                                    scale=args.scale)
    
    # Save if requested
    if args.save:
        fig.savefig(f"{args.save}_2d_global.png", dpi=150, bbox_inches='tight')
        print(f"Saved figure: {args.save}_2d_global.png")
    
    if not args.no_show:
        plt.show()
    
    return positions, spins, order_params_local


if __name__ == '__main__':
    main()
