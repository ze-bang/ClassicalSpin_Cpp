"""
Honeycomb Lattice Analysis Tools
=================================

This module provides three main functionalities for analyzing molecular dynamics
and pump-probe spectroscopy data from honeycomb lattice simulations:

1. read_MD_tot(dir)           - Aggregate DSSF from multiple MD runs
2. read_2D_nonlinear_tot(dir) - Aggregate 2D nonlinear spectroscopy from multiple runs
3. parse_spin_config(dir)     - Parse and visualize spin configurations

Supports both HDF5 (new format) and text file (legacy) formats.
"""

import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.colors import PowerNorm, LogNorm

# ============================================================================
# CONSTANTS AND GLOBAL VARIABLES
# ============================================================================

# Kitaev local frame transformation
kitaevLocal = np.array([
    [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
    [-1/np.sqrt(2), 1/np.sqrt(2), 0],
    [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
])

# Reciprocal space basis
kitaevBasis = np.array([
    [6.28318531, -3.62759873, 0.],
    [0., 7.25519746, 0.],
    [0., 0., 6.28318531]
])

# High-symmetry points in reciprocal space
stepN = 100
Gamma = np.array([0, 0, 0])
K1 = np.array([1/3, 2/3, 0])
K2 = -np.array([2/3, 1/3, 0])
M1 = np.array([0, 1/2, 0])
M2 = np.array([1/2, 1/2, 0])
Gamma2 = np.array([0, 1, 0])

# Transform to Cartesian coordinates
Gamma = contract('ij, i->j', kitaevBasis, Gamma)
K1 = contract('ij, i->j', kitaevBasis, K1)
K2 = contract('ij, i->j', kitaevBasis, K2)
M1 = contract('ij, i->j', kitaevBasis, M1)
M2 = contract('ij, i->j', kitaevBasis, M2)
Gamma2 = contract('ij, i->j', kitaevBasis, Gamma2)

def drawLine(A, B, N):
    return np.array([A + (B - A) * i / N for i in range(N)])

# Define k-path through Brillouin zone
P1_2D = drawLine(Gamma, M1, stepN)
P2_2D = drawLine(M2, Gamma, stepN)
P3_2D = drawLine(Gamma, K1, stepN)
P4_2D = drawLine(K2, Gamma, stepN)
P5_2D = drawLine(Gamma2, M1, stepN)
P6_2D = drawLine(M1, K1, int(stepN/2))

# Path segment positions
gGamma = 0
gM1M2 = len(P1_2D)
gGamma1 = gM1M2 + len(P2_2D)
gK1K2 = gGamma1 + len(P3_2D)
gGamma1Gamma2 = gK1K2 + len(P4_2D)
gM1 = gGamma1Gamma2 + len(P5_2D)
gK1 = gM1 + len(P6_2D)

# Complete k-path
DSSF_K = np.concatenate((P1_2D, P2_2D, P3_2D, P4_2D, P5_2D, P6_2D))

# ============================================================================
# HELPER FUNCTIONS - Fourier Transforms
# ============================================================================

def Spin_t(k, S, P):
    """Time-dependent spin structure factor.
    
    Args:
        k: Momentum points (n_k, 3)
        S: Spin configurations (n_time, n_atoms, 3)
        P: Atomic positions (n_atoms, 3)
    
    Returns:
        S(k,t): (n_time, n_k, 3)
    """
    ffact = np.exp(1j * contract('ik,jk->ij', k, P))
    N = S.shape[1]
    return contract('tjs, ij->tis', S, ffact) / np.sqrt(N)

def DSSF(w, k, S, P, T, gb=False):
    """Dynamical Spin Structure Factor.
    
    DSSF(ω, k) = |∫ S(k,t) exp(iωt) dt|²
    
    Args:
        w: Frequency array
        k: Momentum points
        S: Spin configurations (time, atoms, components)
        P: Atomic positions
        T: Time array
        gb: Use global frame (default: local Kitaev frame)
    
    Returns:
        DSSF(ω, k): (len(w), len(k))
    """
    A = Spin_t(k, S, P)
    ffactt = np.exp(1j * contract('w,t->wt', w, T))
    Somega = contract('tis, wt->wis', A, ffactt) / np.sqrt(len(T))
    
    zq = contract('ar, ir->ia', kitaevLocal, k)
    k_norm_sq = contract('ik,ik->i', k, k)
    # Avoid division by zero for k=0
    k_norm_sq = np.where(k_norm_sq < 1e-10, 1.0, k_norm_sq)
    proj = (contract('ar,br,i->iab', kitaevLocal, kitaevLocal, np.ones(len(k))) - 
            contract('ia,ib,i->iab', zq, zq, 1 / k_norm_sq))
    
    read = np.real(contract('wia, wib, iab-> wi', Somega, np.conj(Somega), proj))
    
    # Avoid log(0) or log(negative)
    read = np.where(read > 1e-10, read, 1e-10)
    result = np.log(read)
    
    # Ensure float64 output
    result = np.asarray(result, dtype=np.float64)
    return result

def SSSF_q(k, S, P, gb=False):
    """Static Spin Structure Factor at specific k-points.
    
    Args:
        k: Momentum points (n_k, 3)
        S: Spin configuration (n_atoms, 3)
        P: Atomic positions (n_atoms, 3)
        gb: Use global frame
    
    Returns:
        SSSF(k): (n_k, 3, 3) tensor (real part only)
    """
    ffact = np.exp(1j * contract('ik,jk->ij', k, P))
    N = len(S)
    Sq = contract('js, ij->is', S, ffact) / np.sqrt(N)
    result = contract('ia, ib->iab', Sq, np.conj(Sq))
    # S(q) * S(-q)^* should be real, take real part to avoid casting issues
    return np.real(result)

def SSSF2D(S, P, nK, dir, gb=False, h_range=(-1.5, 1.5), k_range=(-1.5, 1.5)):
    """Compute 2D Static Spin Structure Factor.
    
    Args:
        S: Spin configuration
        P: Atomic positions
        nK: Grid size (nK x nK)
        dir: Output directory
        gb: Use global frame
        h_range: Range of H in reciprocal lattice units (default: -1.5 to 1.5)
        k_range: Range of K in reciprocal lattice units (default: -1.5 to 1.5)
    
    Returns:
        SSSF: (nK, nK, 3, 3)
        H_grid: H values in RLU
        K_grid: K values in RLU
    """
    H = np.linspace(h_range[0], h_range[1], nK)
    L = np.linspace(k_range[0], k_range[1], nK)
    H_grid, K_grid = np.meshgrid(H, L)
    K_momenta = hhknk(H_grid.flatten(), K_grid.flatten()).reshape((nK * nK, 3))
    SSSF = SSSF_q(K_momenta, S, P, gb)
    SSSF = SSSF.reshape((nK, nK, 3, 3))
    return SSSF, H_grid, K_grid

# ============================================================================
# HELPER FUNCTIONS - Reciprocal Space Coordinates
# ============================================================================

def honeycomb_reciprocal_basis():
    """Calculate reciprocal lattice vectors for honeycomb lattice."""
    a1 = np.array([1, 0, 0])
    a2 = np.array([1/2, np.sqrt(3)/2, 0])
    a3 = np.array([0, 0, 1])
    
    a2_cross_a3 = np.cross(a2, a3)
    a3_cross_a1 = np.cross(a3, a1)
    a1_cross_a2 = np.cross(a1, a2)
    
    vol = np.dot(a1, np.cross(a2, a3))
    
    b1 = 2 * np.pi * a2_cross_a3 / vol
    b2 = 2 * np.pi * a3_cross_a1 / vol
    b3 = 2 * np.pi * a1_cross_a2 / vol
    
    return np.array([b1, b2, b3])

def hhknk(H, K):
    """Convert (H, K) to momentum in units of 2π/a."""
    reciprocal_basis = honeycomb_reciprocal_basis()
    b1 = reciprocal_basis[0]
    b2 = reciprocal_basis[1]
    return np.outer(H, b1) + np.outer(K, b2)

# ============================================================================
# HELPER FUNCTIONS - Plotting
# ============================================================================

def get_honeycomb_brillouin_zone(coord_type='cartesian'):
    """Get first Brillouin zone vertices and high-symmetry points for honeycomb lattice.
    
    The honeycomb lattice has a triangular Bravais lattice, so its reciprocal lattice
    is also triangular (rotated 30 degrees). The first BZ is a hexagon.
    
    High-symmetry points (in reciprocal lattice units u*b1 + v*b2):
    - Gamma: (0, 0)
    - K: (2/3, 1/3) and K': (1/3, 2/3)
    - M: (1/2, 0), (0, 1/2), (1/2, 1/2)
    
    Args:
        coord_type: 'cartesian' for k-space (1/Angstrom or 2pi/a units)
                    'rlu' for reciprocal lattice units (H, K)
    
    Returns:
        bz_vertices: (7, 2) array of BZ boundary vertices (closed loop)
        high_sym_points: dict with Gamma, K, K', M coordinates
    """
    if coord_type == 'rlu':
        # In RLU, the BZ is the Wigner-Seitz cell of the reciprocal lattice
        # For triangular lattice, this is a hexagon centered at origin
        bz_vertices = np.array([
            [2/3, 1/3],
            [1/3, 2/3],
            [-1/3, 1/3],
            [-2/3, -1/3],
            [-1/3, -2/3],
            [1/3, -1/3],
            [2/3, 1/3],  # Close the loop
        ])
        high_sym_points = {
            'Gamma': np.array([0.0, 0.0]),
            'K': np.array([2/3, 1/3]),
            'Kp': np.array([1/3, 2/3]),
            'M': np.array([1/2, 0.0]),
            'Mp': np.array([0.0, 1/2]),
        }
    else:  # cartesian
        reciprocal_basis = honeycomb_reciprocal_basis()
        b1 = reciprocal_basis[0, :2]
        b2 = reciprocal_basis[1, :2]
        
        # BZ vertices in Cartesian coordinates
        bz_vertices = np.array([
            (2/3) * b1 + (1/3) * b2,
            (1/3) * b1 + (2/3) * b2,
            (-1/3) * b1 + (1/3) * b2,
            (-2/3) * b1 + (-1/3) * b2,
            (-1/3) * b1 + (-2/3) * b2,
            (1/3) * b1 + (-1/3) * b2,
            (2/3) * b1 + (1/3) * b2,  # Close the loop
        ])
        high_sym_points = {
            'Gamma': np.array([0.0, 0.0]),
            'K': (2/3) * b1 + (1/3) * b2,
            'Kp': (1/3) * b1 + (2/3) * b2,
            'M': (1/2) * b1,
            'Mp': (1/2) * b2,
        }
    
    return bz_vertices, high_sym_points


def SSSFGraph2D(H_grid, K_grid, d1, filename, coord_type='cartesian', cmap='viridis'):
    """Plot 2D Static Structure Factor with Brillouin zone overlay.
    
    Args:
        H_grid: H values in reciprocal lattice units (2D meshgrid)
        K_grid: K values in reciprocal lattice units (2D meshgrid)
        d1: SSSF intensity data (2D array)
        filename: Output filename (without extension)
        coord_type: 'cartesian' for Cartesian k-space plot (k_x, k_y in units of 2pi/a)
                    'rlu' for reciprocal lattice units plot (H, K)
        cmap: Colormap to use (default: 'viridis')
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    reciprocal_basis = honeycomb_reciprocal_basis()
    b1 = reciprocal_basis[0, :2]
    b2 = reciprocal_basis[1, :2]
    
    if coord_type == 'rlu':
        # Plot in RLU coordinates directly (H, K axes are orthogonal in the plot)
        X_plot = H_grid
        Y_plot = K_grid
        xlabel = r'$H$ (r.l.u.)'
        ylabel = r'$K$ (r.l.u.)'
    else:
        # Convert RLU grid to Cartesian k-space
        X_plot = H_grid * b1[0] + K_grid * b2[0]
        Y_plot = H_grid * b1[1] + K_grid * b2[1]
        xlabel = r'$k_x$ $(2\pi/a)$'
        ylabel = r'$k_y$ $(2\pi/a)$'
    
    # Plot the SSSF data
    mesh = ax.pcolormesh(X_plot, Y_plot, d1, shading='auto', cmap=cmap)
    plt.colorbar(mesh, ax=ax, label=r'$\log S(\mathbf{q})$')
    
    # Get and plot Brillouin zone
    bz_vertices, high_sym_points = get_honeycomb_brillouin_zone(coord_type=coord_type)
    
    # Plot first BZ boundary
    ax.plot(bz_vertices[:, 0], bz_vertices[:, 1], 'w-', lw=2.0, label='1st BZ')
    
    # Also plot second BZ (shifted by reciprocal lattice vectors) if in range
    if coord_type == 'rlu':
        shifts = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1)]
    else:
        shifts = [b1, b2, -b1, -b2, b1+b2, -b1-b2]
    
    for shift in shifts:
        shifted_bz = bz_vertices + np.array(shift)
        ax.plot(shifted_bz[:, 0], shifted_bz[:, 1], 'w--', lw=1.0, alpha=0.5)
    
    # Mark high-symmetry points
    for name, pos in high_sym_points.items():
        ax.scatter(pos[0], pos[1], c='white', s=60, zorder=5, edgecolors='black', linewidths=1)
        # Add label with offset
        offset = 0.05 if coord_type == 'rlu' else 0.3
        label_name = r'$\Gamma$' if name == 'Gamma' else (r"$K'$" if name == 'Kp' else (r"$M'$" if name == 'Mp' else f'${name}$'))
        ax.text(pos[0] + offset, pos[1] + offset, label_name, color='white', fontsize=12, fontweight='bold',
                path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_aspect('equal')
    ax.set_title(f'Static Spin Structure Factor ({coord_type.upper()})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filename + f"_{coord_type}.pdf", dpi=150)
    plt.close()
    print(f"    Saved: {filename}_{coord_type}.pdf")

# ============================================================================
# HELPER FUNCTIONS - I/O
# ============================================================================

def read_MD(dir, w):
    """Read molecular dynamics trajectory and compute DSSF.
    
    Supports both HDF5 (new format) and text file (legacy) formats.
    
    Args:
        dir: Directory containing trajectory data
        w: Frequency array for DSSF
    
    Returns:
        DSSF array
    """
    try:
        hdf5_path = os.path.join(dir, "trajectory.h5")
        if os.path.exists(hdf5_path):
            # Read from HDF5 file
            with h5py.File(hdf5_path, 'r') as f:
                T = f['/trajectory/times'][:]
                S = f['/trajectory/spins'][:]
                # Try new location first, fallback to old location for backwards compatibility
                if '/metadata/positions' in f:
                    P = f['/metadata/positions'][:]
                elif '/trajectory/positions' in f:
                    P = f['/trajectory/positions'][:]
                else:
                    raise KeyError("positions dataset not found in HDF5 file")
                
                print(f"  Loaded HDF5: {len(T)} time steps, {S.shape[1]} sites")
        else:
            # Fallback to text file format
            P = np.loadtxt(dir + "/pos.txt")
            T = np.loadtxt(dir + "/Time_steps.txt")
            S = np.loadtxt(dir + "/spin_t.txt").reshape((len(T), len(P), 3))
            print(f"  Loaded text: {len(T)} time steps, {len(P)} sites")
        
        print(f"  Computing DSSF...")
        A = DSSF(w, DSSF_K, S, P, T, True)
        print(f"  DSSF complete: shape={A.shape}")
        
        np.savetxt(dir + "_DSSF.txt", A)
        return A
    except Exception as e:
        print(f"  ERROR in read_MD: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

def read_MD_slice(dir, nK, w):
    """Read MD trajectory and compute sliced DSSF on 2D momentum grid.
    
    Args:
        dir: Directory containing trajectory data
        nK: Grid size (nK x nK)
        w: Frequency array
    
    Returns:
        DSSF array (len(w), nK, nK)
    """
    try:
        hdf5_path = os.path.join(dir, "trajectory.h5")
        if os.path.exists(hdf5_path):
            with h5py.File(hdf5_path, 'r') as f:
                T = f['/trajectory/times'][:]
                S = f['/trajectory/spins'][:]  # Read all time steps
                # Try new location first, fallback to old location for backwards compatibility
                if '/metadata/positions' in f:
                    P = f['/metadata/positions'][:]
                elif '/trajectory/positions' in f:
                    P = f['/trajectory/positions'][:]
                else:
                    raise KeyError("positions dataset not found in HDF5 file")
        else:
            P = np.loadtxt(dir + "/pos.txt")
            T = np.loadtxt(dir + "/Time_steps.txt")
            S = np.loadtxt(dir + "/spin_t.txt").reshape((len(T), len(P), 3))
        
        H = np.linspace(-1.5, 1.5, nK)
        L = np.linspace(-1.0, 1.0, nK)
        A, B = np.meshgrid(H, L)
        K = hhknk(A, B).reshape((nK * nK, 3))
        
        A = DSSF(w, K, S, P, T, True)
        
        np.savetxt(dir + "_DSSF_sliced.txt", A)
        A = A.reshape((len(w), nK, nK))
        print(f"  [read_MD_slice] After reshape: A.shape={A.shape}, A.dtype={A.dtype}")
        
        # Ensure float64 output
        A = np.asarray(A, dtype=np.float64)
        return A
    except Exception as e:
        print(f"  ERROR in read_MD_slice: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

def _validate_window(window, name):
    """Ensure an omega window is either None or a valid (min, max) tuple."""
    if window is None:
        return None
    if (not isinstance(window, (tuple, list)) or len(window) != 2 or
            window[0] >= window[1]):
        raise ValueError(f"{name} must be a tuple (min, max) with min < max")
    return tuple(window)


def read_2D_nonlinear(dir, omega_t_window=None, omega_tau_window=None):
    """Read and compute 2D nonlinear spectroscopy using FFT.
    
    Supports both HDF5 (pump_probe_spectroscopy.h5) and text file formats.
    
    Args:
        dir: Directory containing pump-probe data
        omega_t_window: Optional (min, max) tuple for ω_t axis limits
        omega_tau_window: Optional (min, max) tuple for ω_τ axis limits
    """
    omega_t_window = _validate_window(omega_t_window, "omega_t_window")
    omega_tau_window = _validate_window(omega_tau_window, "omega_tau_window")
    hdf5_path = os.path.join(dir, "pump_probe_spectroscopy.h5")
    component_labels = ['x', 'y', 'z']
    
    # Read from HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        times = f['/reference/times'][:]
        M0_antiferro = f['/reference/M_local'][:]
        tau_values = f['/tau_scan/tau_values'][:]
        tau_step = len(tau_values)
        
        # Print metadata
        print(f"  Loaded HDF5: {hdf5_path}")
        print(f"    Time steps: {len(times)}, t_range: [{times[0]:.4f}, {times[-1]:.4f}]")
        print(f"    Tau values: {tau_step}, tau_range: [{tau_values[0]:.4f}, {tau_values[-1]:.4f}]")
        print(f"    M_local shape: {M0_antiferro.shape}")
        
        # Print pulse and lattice metadata if available
        if '/metadata' in f:
            metadata_grp = f['/metadata']
            print(f"  Pulse parameters:")
            if 'pulse_amp' in metadata_grp.attrs:
                print(f"    Amplitude: {metadata_grp.attrs['pulse_amp']}")
            if 'pulse_width' in metadata_grp.attrs:
                print(f"    Width: {metadata_grp.attrs['pulse_width']}")
            if 'pulse_freq' in metadata_grp.attrs:
                print(f"    Frequency: {metadata_grp.attrs['pulse_freq']}")
            print(f"  Lattice parameters:")
            if 'n_atoms' in metadata_grp.attrs:
                print(f"    n_atoms: {metadata_grp.attrs['n_atoms']}")
            if 'spin_dim' in metadata_grp.attrs:
                print(f"    spin_dim: {metadata_grp.attrs['spin_dim']}")
            if 'lattice_size' in metadata_grp.attrs:
                print(f"    lattice_size: {metadata_grp.attrs['lattice_size']}")
            print(f"  Time evolution:")
            if 'T_start' in metadata_grp.attrs:
                print(f"    T_start: {metadata_grp.attrs['T_start']}")
            if 'T_end' in metadata_grp.attrs:
                print(f"    T_end: {metadata_grp.attrs['T_end']}")
            if 'T_step' in metadata_grp.attrs:
                print(f"    T_step: {metadata_grp.attrs['T_step']}")
            if 'integration_method' in metadata_grp.attrs:
                method = metadata_grp.attrs['integration_method']
                if isinstance(method, bytes):
                    method = method.decode('utf-8')
                print(f"    Integration method: {method}")
        
        # Process all 3 components (x, y, z) for M_NL, M0, M1, M01
        length = len(M0_antiferro[:, 0])
        M_NL_components = np.zeros((3, tau_step, length))
        M0_components = np.zeros((3, tau_step, length))  # M0 broadcast to all tau
        M1_components = np.zeros((3, tau_step, length))
        M01_components = np.zeros((3, tau_step, length))
        
        for i, tau_val in enumerate(tau_values):
            tau_group = f[f'/tau_scan/tau_{i}']
            M1_antiferro = tau_group['M1_local'][:]
            M01_antiferro = tau_group['M01_local'][:]
            
            for comp in range(3):
                M0 = M0_antiferro[:, comp]
                M1 = M1_antiferro[:, comp]
                M01 = M01_antiferro[:, comp]
                
                min_len = min(len(M0), len(M1), len(M01), length)
                M_NL_components[comp, i, :min_len] = M01[:min_len] - M0[:min_len] - M1[:min_len]
                M0_components[comp, i, :min_len] = M0[:min_len]  # Broadcast M0 to all tau slices
                M1_components[comp, i, :min_len] = M1[:min_len]
                M01_components[comp, i, :min_len] = M01[:min_len]
        
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        tau = tau_values

    # Use z-component for main analysis (backwards compatible)
    M_NL = M_NL_components[2]
    
    # Compute omega arrays (needed for all plots)
    omega_tau = np.fft.fftfreq(int(len(tau)), tau[1] - tau[0] if len(tau) > 1 else 1.0) * 2 * np.pi
    omega_tau = np.fft.fftshift(omega_tau)
    omega_t = np.fft.fftfreq(M_NL.shape[1], dt) * 2 * np.pi
    omega_t = np.fft.fftshift(omega_t)
    
    # Helper function for 2D FFT analysis
    def compute_2d_fft(data):
        """Compute 2D FFT with proper shifting and flipping."""
        data_static = np.mean(data)
        data_dynamic = data - data_static
        data_FF = np.fft.fft2(data_dynamic)
        data_FF = np.fft.fftshift(data_FF)
        data_FF = np.abs(data_FF)
        data_FF = np.flip(data_FF, axis=1)  # Flip omega_t axis
        return data_FF
    
    # =========================================================================
    # Analysis for M0, M1, M01 (individual signals)
    # =========================================================================
    signal_names = ['M0', 'M1', 'M01']
    signal_data = [M0_components, M1_components, M01_components]
    
    for sig_name, sig_components in zip(signal_names, signal_data):
        print(f"  Processing {sig_name}...")
        
        # Create debug plot for this signal (3 components x 3 plot types)
        fig_sig, axes_sig = plt.subplots(3, 3, figsize=(15, 12))
        
        for comp in range(3):
            sig_comp = sig_components[comp]
            
            # Time domain plot
            ax_time = axes_sig[comp, 0]
            for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
                ax_time.plot(sig_comp[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
            ax_time.set_xlabel('Time index')
            ax_time.set_ylabel(f'${sig_name}_{{{component_labels[comp]}}}$')
            ax_time.set_title(f'{component_labels[comp]}-component (time domain)')
            ax_time.legend(fontsize=6)
            ax_time.grid(True, alpha=0.3)
            
            # 2D time-tau plot
            ax_2d = axes_sig[comp, 1]
            im = ax_2d.imshow(sig_comp, origin='lower', aspect='auto', cmap='RdBu_r',
                              extent=[0, sig_comp.shape[1], tau[0], tau[-1]])
            ax_2d.set_xlabel('Time index')
            ax_2d.set_ylabel('τ')
            ax_2d.set_title(f'{component_labels[comp]}-component (τ, t)')
            plt.colorbar(im, ax=ax_2d)
            
            # Frequency domain plot (2D FFT)
            sig_comp_FF = compute_2d_fft(sig_comp)
            
            ax_freq = axes_sig[comp, 2]
            im_freq = ax_freq.imshow(sig_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                      norm='linear', extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]])
            ax_freq.set_xlabel('$\\omega_t$ (rad/time)')
            ax_freq.set_ylabel('$\\omega_{\\tau}$ (rad/time)')
            ax_freq.set_title(f'{component_labels[comp]}-component (freq domain)')
            if omega_t_window is not None:
                ax_freq.set_xlim(omega_t_window)
            if omega_tau_window is not None:
                ax_freq.set_ylim(omega_tau_window)
            plt.colorbar(im_freq, ax=ax_freq)
            
            # Save individual component FFT data
            np.savetxt(dir + f"/{sig_name}_FF_{component_labels[comp]}.txt", sig_comp_FF)
        
        plt.tight_layout()
        plt.savefig(dir + f"/{sig_name}_components_debug.pdf")
        plt.clf()
        plt.close()
        
        # Main spectrum plot (z-component)
        sig_z = sig_components[2]
        sig_z_FF = compute_2d_fft(sig_z)
        np.savetxt(dir + f"/{sig_name}_FF.txt", sig_z_FF)
        
        plt.imshow(sig_z_FF, origin='lower',
                   extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
                   aspect='auto', cmap='gnuplot2', norm='linear')
        plt.xlabel('$\\omega_t$ (rad/time)')
        plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
        plt.colorbar(label='Intensity')
        plt.title(f'{sig_name} Spectrum')
        if omega_t_window is not None:
            plt.xlim(omega_t_window)
        if omega_tau_window is not None:
            plt.ylim(omega_tau_window)
        plt.savefig(dir + f"/{sig_name}_SPEC.pdf")
        plt.clf()
    
    # =========================================================================
    # Original M_NL analysis (M01 - M0 - M1)
    # =========================================================================
    
    # Debug plots: Plot M_NL for each component in time domain and frequency domain
    fig_debug, axes_debug = plt.subplots(3, 3, figsize=(15, 12))
    for comp in range(3):
        M_NL_comp = M_NL_components[comp]
        
        # Time domain plot (M_NL vs t for different tau)
        ax_time = axes_debug[comp, 0]
        for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):  # Plot ~5 tau values
            ax_time.plot(M_NL_comp[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
        ax_time.set_xlabel('Time index')
        ax_time.set_ylabel(f'$M_{{NL,{component_labels[comp]}}}$')
        ax_time.set_title(f'{component_labels[comp]}-component (time domain)')
        ax_time.legend(fontsize=6)
        ax_time.grid(True, alpha=0.3)
        
        # 2D time-tau plot
        ax_2d = axes_debug[comp, 1]
        im = ax_2d.imshow(M_NL_comp, origin='lower', aspect='auto', cmap='RdBu_r',
                          extent=[0, M_NL_comp.shape[1], tau[0], tau[-1]])
        ax_2d.set_xlabel('Time index')
        ax_2d.set_ylabel('τ')
        ax_2d.set_title(f'{component_labels[comp]}-component (τ, t)')
        plt.colorbar(im, ax=ax_2d)
        
        # Frequency domain plot (2D FFT)
        M_NL_comp_static = np.mean(M_NL_comp)
        M_NL_comp_dynamic = M_NL_comp - M_NL_comp_static
        M_NL_comp_FF = np.fft.fft2(M_NL_comp_dynamic)
        M_NL_comp_FF = np.fft.fftshift(M_NL_comp_FF)
        M_NL_comp_FF = np.abs(M_NL_comp_FF)
        
        # Flip omega_t axis to correct direction
        M_NL_comp_FF = np.flip(M_NL_comp_FF, axis=1)
        
        ax_freq = axes_debug[comp, 2]
        # Use flipped omega_t for extent (note: we need to reverse the x extent)
        im_freq = ax_freq.imshow(M_NL_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                  norm='linear', extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]])
        ax_freq.set_xlabel('$\\omega_t$ (rad/time)')
        ax_freq.set_ylabel('$\\omega_{\\tau}$ (rad/time)')
        ax_freq.set_title(f'{component_labels[comp]}-component (freq domain)')
        if omega_t_window is not None:
            ax_freq.set_xlim(omega_t_window)
        if omega_tau_window is not None:
            ax_freq.set_ylim(omega_tau_window)
        plt.colorbar(im_freq, ax=ax_freq)
        
        # Save individual component FFT data
        np.savetxt(dir + f"/M_NL_FF_{component_labels[comp]}.txt", M_NL_comp_FF)
    
    plt.tight_layout()
    plt.savefig(dir + "/M_NL_components_debug.pdf")
    plt.clf()
    plt.close()
    
    # Subtract static values for FFT stability (z-component for main output)
    M_NL_static = np.mean(M_NL)
    M_NL_dynamic = M_NL - M_NL_static
    
    # Perform 2D FFT
    M_NL_FF = np.fft.fft2(M_NL_dynamic)
    M_NL_FF = np.fft.fftshift(M_NL_FF)
    M_NL_FF = np.abs(M_NL_FF)
    
    # Flip omega_t axis (second dimension) to correct direction
    M_NL_FF = np.flip(M_NL_FF, axis=1)
    
    np.savetxt(dir + "/M_NL_FF.txt", M_NL_FF)
    
    # Full spectrum plot (omega_t is now flipped, so extent goes from -omega_t[-1] to -omega_t[0])
    plt.imshow(M_NL_FF, origin='lower',
               extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
               aspect='auto', cmap='gnuplot2', norm='linear')
    plt.xlabel('$\\omega_t$ (rad/time)')
    plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
    plt.colorbar(label='Intensity')
    if omega_t_window is not None:
        plt.xlim(omega_t_window)
    if omega_tau_window is not None:
        plt.ylim(omega_tau_window)
    plt.savefig(dir + "/M_NLSPEC.pdf")
    plt.clf()

# ============================================================================
# MAIN FUNCTION 1: Aggregate DSSF from Multiple MD Runs
# ============================================================================

def read_MD_tot(dir):
    """Aggregate DSSF from multiple molecular dynamics runs.
    
    Processes all subdirectories containing trajectory.h5 or text files,
    computes DSSF along high-symmetry path and on 2D momentum grids,
    and generates comprehensive plots.
    
    Args:
        dir: Parent directory containing subdirectories with MD data
    
    Outputs:
        - DSSF_line.pdf: DSSF along high-symmetry k-path
        - DSSF_0.pdf, DSSF_1.pdf: 2D momentum slices at various frequencies
        - SSSF_tot.pdf: Aggregated static structure factor
    """
    directory = os.fsencode(dir)
    nK = 100
    A = np.zeros((8, nK, nK), dtype=np.float64)
    w = np.array([0.1, 1, 2, 3, 4, 5, 6, 7])
    w0 = 0
    wmax = 5
    w_line = np.arange(w0, wmax, 1/100)[1:]
    B = np.zeros((len(w_line), len(DSSF_K)), dtype=np.float64)
    SSSF = np.zeros((nK, nK, 3, 3), dtype=np.float64)
    
    H = np.linspace(0, 1, nK)
    L = np.linspace(0, 1, nK)
    C, D = np.meshgrid(H, L)
    
    # Process each subdirectory
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        # Skip non-directory files and output files
        if not os.path.isdir(dir + "/" + filename) or filename.endswith('.pdf') or filename.endswith('.txt'):
            continue
        print("Processing folder: " + filename)
        try:
            result = read_MD(dir + "/" + filename, w_line)
            B += result
            
            # Read spin configuration for SSSF
            hdf5_path = os.path.join(dir, filename, "trajectory.h5")
            with h5py.File(hdf5_path, 'r') as f:
                S = f['/trajectory/spins'][0]  # First time step
                # Try new location first, fallback to old location
                if '/metadata/positions' in f:
                    P = f['/metadata/positions'][:]
                elif '/trajectory/positions' in f:
                    P = f['/trajectory/positions'][:]
                else:
                    raise KeyError("positions dataset not found in HDF5 file")
        
            sssf_result, _, _ = SSSF2D(S, P, nK, dir + "/" + filename)
            SSSF += sssf_result
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            continue
    
    # Plot SSSF
    SSSFGraph2D(C, D, contract('ijab->ij', SSSF), dir + "/SSSF_tot")
    
    A = np.transpose(A, (0, 2, 1))
    
    # Plot DSSF along k-path
    fig, ax = plt.subplots(figsize=(10, 4))
    C = ax.imshow(B, origin='lower', extent=[0, gK1, 0, 15], aspect='auto',
                  interpolation='lanczos', cmap='gnuplot2')
    ax.axvline(x=gGamma, color='b', linestyle='dashed')
    ax.axvline(x=gM1M2, color='b', linestyle='dashed')
    ax.axvline(x=gGamma1, color='b', linestyle='dashed')
    ax.axvline(x=gK1K2, color='b', linestyle='dashed')
    ax.axvline(x=gM1, color='b', linestyle='dashed')
    ax.axvline(x=gK1, color='b', linestyle='dashed')
    
    xlabpos = [gGamma, gM1M2, gGamma1, gK1K2, gGamma1Gamma2, gM1, gK1]
    labels = [r'$\Gamma_1$', r'$M_1\quad M_2$', r'$\Gamma_1$', r'$K_1\quad K_2$',
              r'$\Gamma_1\quad\Gamma_2$', r'$M_1$', r'$K_1$']
    ax.set_xticks(xlabpos, labels)
    ax.set_xlim([0, gK1])
    fig.colorbar(C)
    plt.savefig(dir + "/DSSF_line.pdf")
    plt.clf()
    
    # Compute 2D momentum slices
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        # Skip non-directory files and output files
        if not os.path.isdir(dir + "/" + filename) or filename.endswith('.pdf') or filename.endswith('.txt'):
            continue
        try:
            print(f"  Processing {filename} for 2D slices...")
            print(f"  Before: A.shape={A.shape}, A.dtype={A.dtype}")
            result = read_MD_slice(dir + "/" + filename, nK, w)
            print(f"  After read_MD_slice: result.shape={result.shape}, result.dtype={result.dtype}")
            A += result
            print(f"  Addition successful")
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Plot 2D momentum slices at different frequencies
    for i in range(2):
        fig11 = plt.figure(figsize=(8, 8), constrained_layout=False)
        grid = fig11.add_gridspec(2, 2, wspace=0, hspace=0)
        ax = [fig11.add_subplot(grid[0, 0]), fig11.add_subplot(grid[0, 1]),
              fig11.add_subplot(grid[1, 1]), fig11.add_subplot(grid[1, 0])]
        ax[0].set_title(str(w[4*i]) + ' meV')
        ax[1].set_title(str(w[4*i+1]) + ' meV')
        ax[2].set_title(str(w[4*i+2]) + ' meV', y=-0.2)
        ax[3].set_title(str(w[4*i+3]) + ' meV', y=-0.2)
        
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[2].set_yticks([])
        
        ax[0].imshow(A[4*i], origin='lower', extent=[-1.0, 1.0, -1.5, 1.5],
                     aspect='auto', cmap='gnuplot2')
        ax[0].set_xlim([-1.0, 0])
        ax[0].set_ylim([0, 1.5])
        
        ax[1].imshow(A[4*i+1], origin='lower', extent=[-1.0, 1.0, -1.5, 1.5],
                     aspect='auto', cmap='gnuplot2')
        ax[1].set_xlim([0, 1.0])
        ax[1].set_ylim([0, 1.5])
        
        ax[2].imshow(A[4*i+2], origin='lower', extent=[-1.0, 1.0, -1.5, 1.5],
                     aspect='auto', cmap='gnuplot2')
        ax[2].set_xlim([0, 1.0])
        ax[2].set_ylim([-1.5, 0])
        
        ax[3].imshow(A[4*i+3], origin='lower', extent=[-1.0, 1.0, -1.5, 1.5],
                     aspect='auto', cmap='gnuplot2')
        ax[3].set_xlim([-1.0, 0])
        ax[3].set_ylim([-1.5, 0])
        
        plt.savefig(dir + "/DSSF_" + str(i) + ".pdf")
        plt.clf()
        plt.close()

# ============================================================================
# MAIN FUNCTION 2: Aggregate 2D Nonlinear Spectroscopy
# ============================================================================

def read_2D_nonlinear_tot(dir, omega_t_window=None, omega_tau_window=None):
    """Aggregate 2D nonlinear spectroscopy from multiple pump-probe runs.
    
    Processes all subdirectories containing pump-probe data, aggregates
    results, and generates combined 2D FFT spectrum.
    
    Args:
        dir: Parent directory containing subdirectories with pump-probe data
        omega_t_window: Optional (min, max) tuple for ω_t axis limits
        omega_tau_window: Optional (min, max) tuple for ω_τ axis limits
    
    Outputs:
        - M_NL_tot.txt: Aggregated nonlinear magnetization in frequency space
        - NLSPEC_tot.pdf: Combined 2D nonlinear spectrum
    """
    omega_t_window = _validate_window(omega_t_window, "omega_t_window")
    omega_tau_window = _validate_window(omega_tau_window, "omega_tau_window")
    directory = os.fsencode(dir)
    A = None
    count = 0
    
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if os.path.isdir(dir + "/" + filename):
            try:
                read_2D_nonlinear(dir + "/" + filename,
                                   omega_t_window=omega_t_window,
                                   omega_tau_window=omega_tau_window)
                M_NL_data = np.loadtxt(dir + "/" + filename + "/M_NL_FF.txt")
                
                if A is None:
                    A = M_NL_data
                else:
                    min_shape = (min(A.shape[0], M_NL_data.shape[0]),
                                min(A.shape[1], M_NL_data.shape[1]))
                    A[:min_shape[0], :min_shape[1]] += M_NL_data[:min_shape[0], :min_shape[1]]
                count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    if A is not None and count > 0:
        A = A / count  # Average over all runs
        np.savetxt(dir + "/M_NL_tot.txt", A)
        
        # Plot aggregated result
        plt.imshow(A, origin='lower', aspect='auto', cmap='gnuplot2', norm='linear')
        plt.xlabel('$\\omega_t$ (rad/time)')
        plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
        plt.colorbar(label='Intensity')
        if omega_t_window is not None:
            plt.xlim(omega_t_window)
        if omega_tau_window is not None:
            plt.ylim(omega_tau_window)
        plt.savefig(dir + "/NLSPEC_tot.pdf")
        plt.clf()
        print(f"Aggregated {count} runs")
    else:
        print("No valid data found")

# ============================================================================
# MAIN FUNCTION 3: Parse and Visualize Spin Configurations
# ============================================================================

def _visualize_single_spin_config(S, P, output_dir, file_prefix, title_suffix):
    """Helper function to visualize a single spin configuration.
    
    Args:
        S: Spin configuration (n_atoms, 3)
        P: Atomic positions (n_atoms, 3)
        output_dir: Directory to save output files
        file_prefix: Prefix for output files
        title_suffix: Suffix for plot title
    """
    # Compute statistics
    n_atoms = len(S)
    spin_mag = np.linalg.norm(S, axis=1)
    avg_mag = np.mean(spin_mag)
    total_mag = np.sum(S, axis=0)
    sz_min = np.min(S[:, 2])
    sz_max = np.max(S[:, 2])
    
    print(f"    Number of atoms: {n_atoms}")
    print(f"    Average spin magnitude: {avg_mag:.6f}")
    print(f"    Total magnetization: [{total_mag[0]:.6f}, {total_mag[1]:.6f}, {total_mag[2]:.6f}]")
    print(f"    |M_total|: {np.linalg.norm(total_mag):.6f}")
    print(f"    S_z range: [{sz_min:.6f}, {sz_max:.6f}]")
    
    # Compute static spin structure factor (SSSF) on 2D grid
    # Sample centered on Gamma to see full Brillouin zones
    print(f"    Computing static spin correlations...")
    nK = 128  # Grid resolution
    h_range = (-1.5, 1.5)  # Cover multiple Brillouin zones
    k_range = (-1.5, 1.5)
    SSSF, H_grid, K_grid = SSSF2D(S, P, nK, output_dir, gb=True, h_range=h_range, k_range=k_range)
    
    # Sum over spin components for total intensity
    SSSF_total = np.sum(np.abs(SSSF), axis=(2, 3))
    
    # Save SSSF data
    sssf_file = os.path.join(output_dir, f"{file_prefix}_SSSF.txt")
    np.savetxt(sssf_file, SSSF_total)
    
    # Save statistics
    info_file = os.path.join(output_dir, f"{file_prefix}_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"Number of atoms: {n_atoms}\n")
        f.write(f"Average spin magnitude: {avg_mag:.6f}\n")
        f.write(f"Total magnetization: [{total_mag[0]:.6f}, {total_mag[1]:.6f}, {total_mag[2]:.6f}]\n")
        f.write(f"|M_total|: {np.linalg.norm(total_mag):.6f}\n")
        f.write(f"S_z range: [{sz_min:.6f}, {sz_max:.6f}]\n")
    
    # Create XY plane projection with z-component colormap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize colormap based on actual S_z range
    sz_values = S[:, 2]
    
    # Color the arrows by z-component (using viridis to keep all spins visible)
    quiver = ax.quiver(P[:, 0], P[:, 1], S[:, 0], S[:, 1], sz_values,
                       cmap='viridis', clim=(sz_min, sz_max),
                       scale=5, scale_units='inches', width=0.004, 
                       headwidth=3, headlength=4, alpha=0.9)
    
    # Add colorbar for z-component
    cbar = plt.colorbar(quiver, ax=ax, label='Spin $S_z$ component')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'Spin Configuration (XY Projection) - {title_suffix} ({n_atoms} atoms)', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    pdf_file = os.path.join(output_dir, f"{file_prefix}.pdf")
    plt.savefig(pdf_file)
    plt.close()
    print(f"    Saved: {pdf_file}")
    
    # Plot static spin structure factor in BOTH coordinate systems
    log_SSSF = np.log(SSSF_total + 1e-10)
    sssf_basename = os.path.join(output_dir, f"{file_prefix}_SSSF")
    
    # Plot in Cartesian coordinates (k_x, k_y in units of 2π/a)
    SSSFGraph2D(H_grid, K_grid, log_SSSF, sssf_basename, coord_type='cartesian', cmap='viridis')
    
    # Plot in reciprocal lattice units (H, K)
    SSSFGraph2D(H_grid, K_grid, log_SSSF, sssf_basename, coord_type='rlu', cmap='viridis')
    
    print(f"    Saved: {info_file}\n")

def parse_spin_config(dir):
    """Parse and visualize spin configurations from simulation outputs.
    
    Processes all subdirectories containing spin configuration data,
    reads from either HDF5 or text files (including simulated annealing outputs),
    and generates 3D visualization plots for each run.
    
    Args:
        dir: Parent directory containing subdirectories with spin configuration data
    
    Outputs (per subdirectory):
        - spin_config_T=*.pdf: 3D plot of spin configuration for each temperature
        - spin_info_T=*.txt: Statistics about spin configuration for each temperature
    """
    directory = os.fsencode(dir)
    
    # Process each subdirectory
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        print("Processing folder: " + filename)
        if os.path.isdir(dir + "/" + filename):
            try:
                subdir = dir + "/" + filename
                
                # Check for HDF5 file first
                hdf5_path = os.path.join(subdir, "trajectory.h5")
                if os.path.exists(hdf5_path):
                    print(f"  Reading from HDF5: {hdf5_path}")
                    with h5py.File(hdf5_path, 'r') as f:
                        # Use last time step
                        S = f['/trajectory/spins'][-1]
                        # Try new location first, fallback to old location
                        if '/metadata/positions' in f:
                            P = f['/metadata/positions'][:]
                        elif '/trajectory/positions' in f:
                            P = f['/trajectory/positions'][:]
                        else:
                            raise KeyError("positions dataset not found in HDF5 file")
                        if '/trajectory/times' in f:
                            times = f['/trajectory/times'][:]
                            print(f"  Time steps: {len(times)}, last time: {times[-1]:.4f}")
                    
                    # Process single configuration from HDF5
                    _visualize_single_spin_config(S, P, subdir, "spin_config", filename)
                    
                else:
                    # Look for simulated annealing output files (spins_T=*.txt)
                    spin_files = sorted([f for f in os.listdir(subdir) if f.startswith('spins_T=') and f.endswith('.txt')])
                    pos_file = os.path.join(subdir, "positions.txt")
                    
                    if spin_files and os.path.exists(pos_file):
                        print(f"  Found {len(spin_files)} simulated annealing temperature files")
                        P = np.loadtxt(pos_file)
                        
                        # Process each temperature file
                        for spin_file in spin_files:
                            spin_path = os.path.join(subdir, spin_file)
                            S = np.loadtxt(spin_path)
                            
                            # Extract temperature from filename (e.g., "spins_T=0.001000.txt" -> "0.001000")
                            T_str = spin_file.replace('spins_T=', '').replace('.txt', '')
                            output_prefix = f"spin_config_T={T_str}"
                            
                            print(f"  Processing T={T_str}")
                            _visualize_single_spin_config(S, P, subdir, output_prefix, f"{filename} (T={T_str})")
                    else:
                        # Fallback to old naming convention
                        spin_file = os.path.join(subdir, "spin.txt")
                        pos_file = os.path.join(subdir, "pos.txt")
                        
                        if os.path.exists(spin_file) and os.path.exists(pos_file):
                            print(f"  Reading from text files: {spin_file}, {pos_file}")
                            S = np.loadtxt(spin_file)
                            P = np.loadtxt(pos_file)
                            _visualize_single_spin_config(S, P, subdir, "spin_config", filename)
                        else:
                            print(f"  Error: No valid spin configuration found in {subdir}")
                            continue
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}\n")
                import traceback
                traceback.print_exc()
                continue

# ============================================================================
# MAIN - Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python reader_honeycomb_clean.py <directory> [function]")
        print("\nAvailable functions:")
        print("  md        - Aggregate DSSF from MD runs (default)")
        print("  pp        - Aggregate 2D nonlinear spectroscopy")
        print("  spin      - Parse and visualize spin configuration")
        print("\nExample:")
        print("  python reader_honeycomb_clean.py ./simulation_runs/ md")
        sys.exit(1)
    
    directory = sys.argv[1]
    function = sys.argv[2] if len(sys.argv) > 2 else "md"
    
    if function == "md":
        print("Running: read_MD_tot()")
        read_MD_tot(directory)
    elif function == "pp":
        print("Running: read_2D_nonlinear_tot()")
        read_2D_nonlinear_tot(directory,(-0.5,0.5),(-0.5,0.5))
    elif function == "spin":
        print("Running: parse_spin_config()")
        parse_spin_config(directory)
    else:
        print(f"Unknown function: {function}")
        print("Available: md, pp, spin")
