"""
HDF5 Reader for TmFeO3 Molecular Dynamics and 2D Coherent Spectroscopy Results

This module reads simulation outputs from the ClassicalSpin_Cpp package stored in HDF5 format.
Mirrors the functionality of the old reader_TmFeO3.py but works with the new HDF5 outputs
as specified in hdf5_io.h and mixed_lattice.h.

HDF5 File Structures:
=====================

1. Molecular Dynamics trajectory.h5:
   /trajectory_SU2/
     - times [n_steps]
     - spins [n_steps, n_sites_SU2, spin_dim_SU2]
     - magnetization_antiferro [n_steps, spin_dim_SU2]
     - magnetization_local [n_steps, spin_dim_SU2]
   /trajectory_SU3/
     - times [n_steps]
     - spins [n_steps, n_sites_SU3, spin_dim_SU3]
     - magnetization_antiferro [n_steps, spin_dim_SU3]
     - magnetization_local [n_steps, spin_dim_SU3]
   /metadata_SU2/, /metadata_SU3/, /metadata_global/
     - positions, lattice_size, spin_dim, etc.

2. Pump-Probe Spectroscopy pump_probe_spectroscopy.h5:
   /metadata/
     - Comprehensive experimental parameters
   /reference/
     - times [n_times]
     - M_antiferro_SU2, M_local_SU2, M_global_SU2 [n_times, spin_dim]
     - M_antiferro_SU3, M_local_SU3, M_global_SU3 [n_times, spin_dim]
   /tau_scan/
     - tau_values [n_tau]
     - tau_0/, tau_1/, ... (each contains M1 and M01 trajectories)
"""

import h5py
import numpy as np
from opt_einsum import contract
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster plotting
import matplotlib.pyplot as plt
import os
from math import gcd
from functools import reduce, lru_cache
from matplotlib.colors import PowerNorm, LogNorm, SymLogNorm, Normalize
from typing import Dict, Tuple, Optional, List, Any, Literal

# Type for norm selection
NormType = Literal['log', 'power', 'symlog', 'linear']

# Conversion factor: meV to THz
# 1 THz = 4.135667696 meV, so meV / 4.135667696 = THz
MEV_TO_THZ = 1.0 / 4.135667696  # ≈ 0.2418 THz per meV

# Try to import scipy.fft at module level for better performance
try:
    from scipy import fft as _scipy_fft
    _USE_SCIPY_FFT = True
except ImportError:
    _USE_SCIPY_FFT = False


def _save_spectrum(path_stem: str, data: np.ndarray) -> None:
    """Save spectrum array in binary .npy format (100x faster than .txt).
    
    Args:
        path_stem: File path without extension (e.g., '/dir/M_NL_SU2_FF')
        data: Array to save
    """
    np.save(path_stem + '.npy', data)


def _load_spectrum(path_stem: str) -> Optional[np.ndarray]:
    """Load spectrum array, preferring .npy over .txt for speed.
    
    Args:
        path_stem: File path without extension (e.g., '/dir/M_NL_SU2_FF')
        
    Returns:
        Loaded array, or None if not found
    """
    npy_path = path_stem + '.npy'
    txt_path = path_stem + '.txt'
    if os.path.exists(npy_path):
        return np.load(npy_path)
    elif os.path.exists(txt_path):
        return np.loadtxt(txt_path)
    return None


def auto_window_from_peaks(energy_levels_mev: Dict[str, float],
                           margin: float = 2.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute plot/FFT windows automatically from known peak frequencies.
    
    Sets ω_t window to [0, max_peak × margin] and ω_τ window to
    [-max_peak × margin, max_peak × margin].
    
    Args:
        energy_levels_mev: Dict of energy level names → values in meV
        margin: Multiplicative margin beyond highest peak (default 2.0)
        
    Returns:
        (omega_t_window_THz, omega_tau_window_THz) — both in THz
    """
    peak_freqs_thz = [v * MEV_TO_THZ for v in energy_levels_mev.values() if v is not None and v > 0]
    if not peak_freqs_thz:
        return None, None
    max_freq = max(peak_freqs_thz) * margin
    omega_t_window = (0.0, max_freq)
    omega_tau_window = (-max_freq, max_freq)
    return omega_t_window, omega_tau_window

# Default energy level line style parameters
ENERGY_LINE_STYLE = {
    'linestyle': '--',
    'linewidth': 0.8,
    'alpha': 0.7
}

# Color scheme for different energy levels
ENERGY_LINE_COLORS = {
    'e1': 'cyan',
    'e2': 'magenta',
    'e2-e1': 'yellow',
    'qAFM': 'lime',
    'qFM': 'orange'
}


def add_energy_level_lines(ax, energy_levels_mev: Dict[str, float], 
                           omega_window: Optional[Tuple[float, float]] = None,
                           show_labels: bool = True):
    """
    Add horizontal and vertical dashed lines at specified energy levels.
    
    Args:
        ax: matplotlib axes object
        energy_levels_mev: Dict mapping level names to energy values in meV
        omega_window: Optional (min, max) tuple for omega axis limits (in THz)
        show_labels: Whether to show labels for the lines
        
    Lines are drawn for both positive and negative frequencies.
    Energy levels are input in meV but converted to THz for plotting.
    """
    if not energy_levels_mev:
        return
    
    # Get current axis limits for positioning labels
    xlim = ax.get_xlim() if omega_window is None else omega_window
    ylim = ax.get_ylim() if omega_window is None else omega_window
    
    for level_name, energy_mev in energy_levels_mev.items():
        if energy_mev is None or energy_mev == 0:
            continue
            
        # Convert meV to THz for plotting
        omega_thz = energy_mev * MEV_TO_THZ
        
        # Get color for this level
        color = ENERGY_LINE_COLORS.get(level_name, 'white')
        
        # Draw lines at +ω and -ω
        for sign, omega_val in [('+', omega_thz), ('-', -omega_thz)]:
            # Horizontal lines (constant ω_t, y-axis)
            ax.axhline(y=omega_val, color=color, **ENERGY_LINE_STYLE)
            # Vertical lines (constant ω_τ, x-axis)
            ax.axvline(x=omega_val, color=color, **ENERGY_LINE_STYLE)
        
        # Add label only once (at positive frequency, on the right side)
        if show_labels:
            # Format label with both meV and THz
            label = f'{level_name}: {omega_thz:.2f} THz'
            # Position label at edge of plot
            label_x = xlim[0] * 0.95 if xlim[0] > xlim[1] else xlim[1] * 0.95  # Account for flipped axis
            ax.annotate(label, xy=(label_x, omega_thz), fontsize=6, color=color,
                       ha='right', va='bottom', alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.3))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calcNumSites(A: np.ndarray, B: np.ndarray, N: int) -> int:
    """Calculate how many grid points are on the path from A to B."""
    A_grid = A * N / (2*np.pi)
    B_grid = B * N / (2*np.pi)
    delta = B_grid - A_grid
    rounded_delta = np.round(delta).astype(int)
    
    if np.all(rounded_delta == 0):
        return 1
    
    non_zero = [abs(x) for x in rounded_delta if x != 0]
    if not non_zero:
        return 1
    
    gcd_all = reduce(gcd, non_zero)
    return gcd_all + 1


def calculate_spectrum_figsize(omega_tau_range: Tuple[float, float], 
                               omega_t_range: Tuple[float, float],
                               base_width: float = 10.0) -> Tuple[float, float]:
    """
    Calculate appropriate figure size for 2D spectrum based on axis ranges.
    
    Args:
        omega_tau_range: (min, max) for omega_tau axis
        omega_t_range: (min, max) for omega_t axis  
        base_width: Base width for the figure in inches
        
    Returns:
        (width, height) tuple for figure size
    """
    tau_span = omega_tau_range[1] - omega_tau_range[0]
    t_span = omega_t_range[1] - omega_t_range[0]
    
    if tau_span <= 0 or t_span <= 0:
        return (base_width, base_width)
    
    # aspect = height / width.  x-axis = ω_t (width), y-axis = ω_τ (height)
    aspect = tau_span / t_span
    
    # Constrain to reasonable values
    aspect = max(0.3, min(aspect, 3.0))
    
    height = base_width * aspect
    return (base_width, height)


def get_reciprocal_lattice_orthorhombic() -> np.ndarray:
    """
    Get reciprocal lattice vectors for orthorhombic TmFeO3.
    
    Returns:
        3x3 array of reciprocal lattice vectors (as rows)
    """
    return np.array([
        [2*np.pi, 0, 0],
        [0, 2*np.pi, 0],
        [0, 0, 2*np.pi]
    ])


def compute_reciprocal_lattice(positions: np.ndarray) -> np.ndarray:
    """
    Compute reciprocal lattice vectors from real-space positions.
    
    Args:
        positions: Nx3 array of atomic positions
        
    Returns:
        3x3 array of reciprocal lattice vectors (as rows)
    """
    pos = positions.copy()
    
    unique_x = np.unique(pos[:, 0])
    unique_y = np.unique(pos[:, 1])
    unique_z = np.unique(pos[:, 2])
    
    Lx = np.diff(unique_x).min() if len(unique_x) > 1 else 1.0
    Ly = np.diff(unique_y).min() if len(unique_y) > 1 else 1.0
    Lz = np.diff(unique_z).min() if len(unique_z) > 1 else 1.0
    
    a1 = np.array([Lx, 0, 0])
    a2 = np.array([0, Ly, 0])
    a3 = np.array([0, 0, Lz])
    
    volume = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume
    
    return np.array([b1, b2, b3])


def generate_k_path(reciprocal_lattice: np.ndarray, graphres: int = 4) -> Tuple[np.ndarray, List[int]]:
    """
    Generate k-path through high-symmetry points for TmFeO3.
    
    Returns:
        DSSF_K: k-point array
        tick_positions: positions for tick labels [g1, g2, g3, g4]
    """
    P1_frac = np.array([1, 0, 3])
    P2_frac = np.array([3, 0, 3])
    P3_frac = np.array([3, 0, 1])
    P4_frac = np.array([3, 2, 1])
    
    P1 = contract('i,ij->j', P1_frac, reciprocal_lattice)
    P2 = contract('i,ij->j', P2_frac, reciprocal_lattice)
    P3 = contract('i,ij->j', P3_frac, reciprocal_lattice)
    P4 = contract('i,ij->j', P4_frac, reciprocal_lattice)
    
    P12 = np.linspace(P1, P2, calcNumSites(P1, P2, graphres))[1:-1]
    P23 = np.linspace(P2, P3, calcNumSites(P2, P3, graphres))[1:-1]
    P34 = np.linspace(P3, P4, calcNumSites(P3, P4, graphres))[1:-1]
    
    g1 = 0
    g2 = g1 + len(P12)
    g3 = g2 + len(P23)
    g4 = g3 + len(P34)
    
    DSSF_K = np.concatenate((P12, P23, P34))
    
    return DSSF_K, [g1, g2, g3, g4]


# =============================================================================
# SPIN STRUCTURE FACTOR COMPUTATIONS
# =============================================================================

# Local frame definitions for TmFeO3 (4 sublattices)
x = np.array([[1, 0, 0], [1, 0, 0], [-1, 0, 0], [-1, 0, 0]])
y = np.array([[0, 1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0]])
z = np.array([[0, 0, 1], [0, 0, -1], [0, 0, -1], [0, 0, 1]])
localframe = np.array([x, y, z])


# =============================================================================
# TmFeO3 Pbnm Bertaut / sublattice-frame helpers
# =============================================================================
# Pbnm Klein-four eta on the 4 Fe/Tm sublattices.  Same ordering as the C++
# builder (`kEtaPbnm` in src/core/unitcell_builders.cpp).
ETA_PBNM = np.array([[1, 1, 1],
                     [1, -1, -1],
                     [-1, 1, -1],
                     [-1, -1, 1]], dtype=float)

# Bertaut sign patterns sigma[sub] for the four irreducible dipole modes of
# the 4-sublattice Pbnm cell (rows = sublattice 0..3, see tmfeo3_notes.tex
# Sec.~"Local sublattice frames").
BERTAUT_SIGNS = {
    'F': np.array([1.0,  1.0,  1.0,  1.0]),   # uniform (qFM magnon)
    'G': np.array([1.0, -1.0, -1.0,  1.0]),   # Neel (qAFM magnon)
    'C': np.array([1.0, -1.0,  1.0, -1.0]),   # C-mode (this is also the
                                              # default `M_antiferro_*` sign
                                              # written by the C++ pipeline)
    'A': np.array([1.0,  1.0, -1.0, -1.0]),   # A-mode
}


def tmfeo3_su3_sublattice_frames(
        mu_2x: float = 0.0, mu_2y: float = 0.0, mu_2z: float = 5.264,
        mu_5x: float = 2.3915, mu_5y: float = -2.7866, mu_5z: float = 0.0,
        mu_7x: float = 0.9128, mu_7y: float = 0.4655, mu_7z: float = 0.0,
        ) -> np.ndarray:
    """Build the four 8x8 SU(3) sublattice frames F_mu in the "Option A"
    pair-locked extension (matches `apply_tmfeo3_tm_sector` in
    `src/core/unitcell_builders.cpp`).

    F_mu acts as R_act = mu_act^{-1} D_mu mu_act on the (lambda_2, lambda_5,
    lambda_7) dipole triplet (indices {1,4,6}) AND on the locked
    (lambda_1, lambda_4, lambda_6) quadrupole partners (indices {0,3,5}),
    where D_mu = diag(eta_pbnm[mu]).  Identity on (lambda_3, lambda_8)
    (indices {2,7}).

    Returns
    -------
    frames : np.ndarray, shape (4, 8, 8)
        F_mu for the four Tm sublattices in C++ index order.
    """
    mu_act = np.array([[mu_2x, mu_5x, mu_7x],
                       [mu_2y, mu_5y, mu_7y],
                       [mu_2z, mu_5z, mu_7z]], dtype=float)
    if abs(np.linalg.det(mu_act)) < 1e-12:
        raise ValueError(
            "mu_act is singular; cannot build pair-locked SU(3) frames. "
            "Check (mu_2*, mu_5*, mu_7*) configuration.")
    mu_inv = np.linalg.inv(mu_act)
    active_im = [1, 4, 6]
    active_re = [0, 3, 5]
    frames = np.empty((4, 8, 8), dtype=float)
    for sub in range(4):
        D = np.diag(ETA_PBNM[sub])
        R_act = mu_inv @ D @ mu_act
        F = np.eye(8)
        for a in range(3):
            for b in range(3):
                F[active_im[a], active_im[b]] = R_act[a, b]
                F[active_re[a], active_re[b]] = R_act[a, b]
        frames[sub] = F
    return frames


def tmfeo3_bertaut_lab(spins_traj: np.ndarray,
                       frames: np.ndarray,
                       bertaut_signs: Optional[np.ndarray] = None,
                       n_sublattices: int = 4) -> np.ndarray:
    """Compute a Bertaut-projected lab-frame trajectory from raw
    sublattice-frame stored spins.

    M^a(t) = (1/N) sum_i sigma[sub(i)] * F[sub(i)]^a_b * spins[t, i, b]

    Works for both SU(2) (frames shape (4,3,3), spin_dim=3) and SU(3)
    (frames shape (4,8,8), spin_dim=8).

    Parameters
    ----------
    spins_traj : (n_t, n_sites, spin_dim)
        Raw stored Bloch vectors, e.g. from `get_spins('SU2')` /
        `get_spins('SU3')`.  Site index maps to sublattice as
        ``sub = site % n_sublattices`` (matches C++ flatten_index).
    frames : (4, spin_dim, spin_dim)
        Per-sublattice frame matrix F_mu.  For Fe (SU(2)), pass the 3x3
        diag(eta_pbnm[mu]).  For Tm (SU(3)), pass
        `tmfeo3_su3_sublattice_frames(...)`.
    bertaut_signs : (4,) or None
        Bertaut sign pattern.  Use `BERTAUT_SIGNS['F']` for the F-mode
        (matches C++ `M_global_*`) or `BERTAUT_SIGNS['G']` for the qAFM
        Neel mode.  ``None`` defaults to F-mode.

    Returns
    -------
    M : (n_t, spin_dim)
        Bertaut-projected lab-frame Bloch trajectory.
    """
    if bertaut_signs is None:
        bertaut_signs = BERTAUT_SIGNS['F']
    bertaut_signs = np.asarray(bertaut_signs, dtype=float)
    n_t, n_sites, spin_dim = spins_traj.shape
    sub_idx = np.arange(n_sites) % n_sublattices
    F_per_site = frames[sub_idx]                       # (n_sites, sd, sd)
    sign_per_site = bertaut_signs[sub_idx]             # (n_sites,)
    # apply F site-wise to spins
    proj = np.einsum('iab,tib->tia', F_per_site, spins_traj, optimize=True)
    # weighted average over sites
    M = np.einsum('i,tia->ta', sign_per_site, proj, optimize=True) / n_sites
    return M


def tmfeo3_fe_sublattice_frames() -> np.ndarray:
    """3x3 sublattice frames D_mu = diag(eta_pbnm[mu]) for Fe (SU(2)).
    Returned shape (4, 3, 3).  Use with `tmfeo3_bertaut_lab` when working
    from raw Fe spin trajectories stored in the legacy local-frame mode."""
    frames = np.zeros((4, 3, 3), dtype=float)
    for sub in range(4):
        frames[sub] = np.diag(ETA_PBNM[sub])
    return frames


def tmfeo3_tm_dipole_lab(
        spins_su3: np.ndarray,
        bertaut_signs: Optional[np.ndarray] = None,
        n_sublattices: int = 4,
        mu_2x: float = 0.0, mu_2y: float = 0.0, mu_2z: float = 5.264,
        mu_5x: float = 2.3915, mu_5y: float = -2.7866, mu_5z: float = 0.0,
        mu_7x: float = 0.9128, mu_7y: float = 0.4655, mu_7z: float = 0.0,
        ) -> np.ndarray:
    """Compute the physical lab-frame Bertaut-projected Tm dipole moment
    J^a(t) directly from raw SU(3) trajectories, bypassing the stored
    `M_global_SU3` / `M_antiferro_SU3` arrays and the C++ sublattice-frame
    convention.

    For each site i with sublattice mu = i mod n_sublattices the lab dipole is
    ``J_lab^a(i, t) = eta_pbnm[mu, a] * sum_b mu_act[a, b] * <lambda_b>(i, t)``
    (b runs over the time-odd Gell-Mann triplet {lambda_2, lambda_5,
    lambda_7}, i.e. spin-component indices {1, 4, 6}).  The output is the
    Bertaut-projected per-site average

        J^a(t) = (1/N) sum_i sigma[mu(i)] * J_lab^a(i, t).

    Use ``bertaut_signs = BERTAUT_SIGNS['F']`` for the qFM net magnetization,
    ``BERTAUT_SIGNS['G']`` for the qAFM Neel vector, etc.

    Parameters
    ----------
    spins_su3 : (n_t, n_sites, 8)
        Raw stored Tm Bloch vectors, e.g. from `get_spins('SU3')`.
    bertaut_signs : (4,) or None
        Bertaut sign pattern.  Defaults to F-mode (uniform).

    Returns
    -------
    J : (n_t, 3)
        Lab-Cartesian (x, y, z) Bertaut-projected Tm dipole moment.
    """
    if bertaut_signs is None:
        bertaut_signs = BERTAUT_SIGNS['F']
    bertaut_signs = np.asarray(bertaut_signs, dtype=float)
    mu_act = np.array([[mu_2x, mu_5x, mu_7x],
                       [mu_2y, mu_5y, mu_7y],
                       [mu_2z, mu_5z, mu_7z]], dtype=float)
    active_im = [1, 4, 6]
    n_t, n_sites, _ = spins_su3.shape
    sub_idx = np.arange(n_sites) % n_sublattices
    eta_per_site = ETA_PBNM[sub_idx]                  # (n_sites, 3)
    sign_per_site = bertaut_signs[sub_idx]            # (n_sites,)
    # canonical-frame dipole on each site: J_can^a = mu_act[a, b] * <lambda_b_active>
    J_can = np.einsum('ab,tib->tia', mu_act,
                      spins_su3[:, :, active_im], optimize=True)  # (n_t, n_sites, 3)
    # transform to lab via eta (D_mu is diagonal so just elementwise sign)
    J_lab = J_can * eta_per_site[None, :, :]          # (n_t, n_sites, 3)
    # Bertaut-weighted per-site average
    J = np.einsum('i,tia->ta', sign_per_site, J_lab, optimize=True) / n_sites
    return J


def Spin_t(k: np.ndarray, S: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Compute time-dependent spin structure factor.
    
    Args:
        k: k-points array [n_k, 3]
        S: Spin configurations [n_times, n_sites, spin_dim]
        P: Site positions [n_sites, 3]
        
    Returns:
        results: [n_times, n_k, spin_dim] complex array
    """
    N = S.shape[1]
    
    # Compute phase factors once (k @ P.T)
    ffact = np.exp(1j * np.dot(k, P.T))  # [n_k, n_sites]
    
    # Vectorized computation: (n_times, n_sites, spin_dim) @ (n_sites, n_k) -> (n_times, spin_dim, n_k)
    # Then transpose to (n_times, n_k, spin_dim)
    results = np.einsum('tjs,kj->tks', S, ffact, optimize=True) / np.sqrt(N)
    
    return results


def Spin_global_t(k: np.ndarray, S: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Compute time-dependent spin structure factor in global frame.
    Transforms spins from local sublattice frames to global frame using eta.
    
    Args:
        k: k-points array [n_k, 3]
        S: Spin configurations [n_times, n_sites, spin_dim]
        P: Site positions [n_sites, 3]
        
    Returns:
        results: [n_times, n_k, 3] complex array (same as Spin_t)
    """
    n_sublattices = 4
    N = S.shape[1]
    
    # eta[sublattice, component] gives the sign for each spin component
    eta = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    
    # Transform spins to global frame: S_global = S_local * eta (elementwise per sublattice)
    S_global = S[:, :, :3].copy()
    for i in range(n_sublattices):
        S_global[:, i::n_sublattices, :] *= eta[i]
    
    # Compute phase factors once (faster than einsum for matrix multiply)
    ffact = np.exp(1j * np.dot(k, P.T))  # [n_k, n_sites]
    
    # Vectorized computation
    results = np.einsum('tjs,kj->tks', S_global, ffact, optimize=True) / np.sqrt(N)
    
    return results


def DSSF(w: np.ndarray, k: np.ndarray, S: np.ndarray, P: np.ndarray, 
         T: np.ndarray, global_frame: bool = False) -> np.ndarray:
    """
    Compute dynamical spin structure factor using FFT over time.
    
    Args:
        w: Frequency points
        k: k-points array
        S: Spin configurations [n_times, n_sites, spin_dim]
        P: Site positions
        T: Time points
        global_frame: If True, transform to global frame (for SU(2) only)
        
    Returns:
        DSSF: [n_w, n_k, spin_dim, spin_dim] array
    """
    if global_frame and S.shape[2] >= 3:
        # Use global frame transformation for 3-component spins
        A = Spin_global_t(k, S, P)
        # A shape: (n_times, n_k, 3) - same as Spin_t
    else:
        # Use local frame (original implementation)
        A = Spin_t(k, S, P)
    
    # Subtract mean and FFT in one step (more cache-friendly)
    A -= A.mean(axis=0, keepdims=True)
    
    # FFT over time dimension
    dt = T[1] - T[0] if len(T) > 1 else 1.0
    A_fft = np.fft.fft(A, axis=0) / np.sqrt(len(T))
    fft_freqs = np.fft.fftfreq(len(T), d=dt) * 2 * np.pi
    
    # Vectorized frequency index selection
    indices = np.argmin(np.abs(fft_freqs[:, None] - w[None, :]), axis=0)
    Somega = A_fft[indices]
    
    # Optimized outer product
    result = np.einsum('wia,wib->wiab', Somega, np.conj(Somega), optimize=True)
    return np.real(result)


# =============================================================================
# HDF5 READING UTILITIES
# =============================================================================

def read_hdf5_attribute(group, attr_name: str, default=None):
    """Safely read an HDF5 attribute."""
    try:
        return group.attrs[attr_name]
    except KeyError:
        return default


def read_hdf5_dataset(group, dataset_name: str) -> Optional[np.ndarray]:
    """Safely read an HDF5 dataset."""
    try:
        return group[dataset_name][:]
    except KeyError:
        return None


class MDTrajectoryHDF5:
    """
    Reader for molecular dynamics trajectory HDF5 files.
    
    File structure:
    /trajectory_SU2/ or /trajectory_SU3/
      - times [n_steps]
      - spins [n_steps, n_sites, spin_dim]
      - magnetization_antiferro [n_steps, spin_dim]
      - magnetization_local [n_steps, spin_dim]
    /metadata_SU2/ or /metadata_SU3/
      - positions [n_sites, 3]
      - lattice_size, spin_dim, n_atoms, spin_length
    /metadata_global/
      - integration_method, dt_initial, T_start, T_end, save_interval, etc.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the MD trajectory reader.
        
        Args:
            filepath: Path to the trajectory.h5 file
        """
        self.filepath = filepath
        self._file = None
        self.metadata_global = {}
        self.metadata_SU2 = {}
        self.metadata_SU3 = {}
        
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self):
        """Open the HDF5 file and load metadata."""
        self._file = h5py.File(self.filepath, 'r')
        self._load_metadata()
        
    def close(self):
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            
    def _load_metadata(self):
        """Load metadata from the HDF5 file."""
        if 'metadata_global' in self._file:
            grp = self._file['metadata_global']
            self.metadata_global = {
                'integration_method': read_hdf5_attribute(grp, 'integration_method'),
                'dt_initial': read_hdf5_attribute(grp, 'dt_initial'),
                'T_start': read_hdf5_attribute(grp, 'T_start'),
                'T_end': read_hdf5_attribute(grp, 'T_end'),
                'save_interval': read_hdf5_attribute(grp, 'save_interval'),
                'creation_time': read_hdf5_attribute(grp, 'creation_time'),
                'dimensions': read_hdf5_dataset(grp, 'dimensions'),
            }
        
        for sublattice in ['SU2', 'SU3']:
            meta_grp_name = f'metadata_{sublattice}'
            if meta_grp_name in self._file:
                grp = self._file[meta_grp_name]
                meta = {
                    'lattice_size': read_hdf5_attribute(grp, 'lattice_size'),
                    'spin_dim': read_hdf5_attribute(grp, 'spin_dim'),
                    'n_atoms': read_hdf5_attribute(grp, 'n_atoms'),
                    'spin_length': read_hdf5_attribute(grp, 'spin_length'),
                    'positions': read_hdf5_dataset(grp, 'positions'),
                }
                if sublattice == 'SU2':
                    self.metadata_SU2 = meta
                else:
                    self.metadata_SU3 = meta
    
    def get_times(self, sublattice: str = 'SU2') -> np.ndarray:
        """Get time points."""
        return self._file[f'trajectory_{sublattice}/times'][:]
    
    def get_spins(self, sublattice: str = 'SU2') -> np.ndarray:
        """Get spin trajectory [n_steps, n_sites, spin_dim]."""
        return self._file[f'trajectory_{sublattice}/spins'][:]
    
    def get_magnetization_antiferro(self, sublattice: str = 'SU2') -> np.ndarray:
        """Get antiferromagnetic magnetization [n_steps, spin_dim]."""
        return self._file[f'trajectory_{sublattice}/magnetization_antiferro'][:]
    
    def get_magnetization_local(self, sublattice: str = 'SU2') -> np.ndarray:
        """Get local magnetization [n_steps, spin_dim]."""
        return self._file[f'trajectory_{sublattice}/magnetization_local'][:]
    
    def get_positions(self, sublattice: str = 'SU2') -> Optional[np.ndarray]:
        """Get site positions [n_sites, 3]."""
        meta = self.metadata_SU2 if sublattice == 'SU2' else self.metadata_SU3
        return meta.get('positions')


class PumpProbeHDF5:
    """
    Reader for pump-probe spectroscopy HDF5 files.
    
    File structure:
    /metadata/
      - All experimental parameters
    /reference/
      - times [n_times]
      - M_antiferro_SU2, M_local_SU2, M_global_SU2 [n_times, spin_dim]
      - M_antiferro_SU3, M_local_SU3, M_global_SU3 [n_times, spin_dim]
    /tau_scan/
      - tau_values [n_tau]
      - tau_0/, tau_1/, ... each containing M1 and M01 trajectories
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the pump-probe spectroscopy reader.
        
        Args:
            filepath: Path to the pump_probe_spectroscopy.h5 file
        """
        self.filepath = filepath
        self._file = None
        self.metadata = {}
        
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self):
        """Open the HDF5 file and load metadata."""
        self._file = h5py.File(self.filepath, 'r')
        self._load_metadata()
        
    def close(self):
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            
    def _load_metadata(self):
        """Load metadata from the HDF5 file."""
        if 'metadata' in self._file:
            grp = self._file['metadata']
            for key in grp.attrs.keys():
                val = grp.attrs[key]
                # Decode bytes to string if needed
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                self.metadata[key] = val
    
    def get_tau_values(self) -> np.ndarray:
        """Get delay time values."""
        return self._file['tau_scan/tau_values'][:]
    
    def get_reference_times(self) -> np.ndarray:
        """Get reference trajectory times."""
        return self._file['reference/times'][:]
    
    def get_reference_magnetization(self, mag_type: str = 'global', 
                                     sublattice: str = 'SU2') -> np.ndarray:
        """
        Get reference magnetization trajectory.
        
        Args:
            mag_type: 'antiferro', 'local', or 'global'
            sublattice: 'SU2' or 'SU3'
            
        Returns:
            [n_times, spin_dim] array
        """
        dataset_name = f'M_{mag_type}_{sublattice}'
        return self._file[f'reference/{dataset_name}'][:]
    
    def get_tau_trajectory(self, tau_index: int, trajectory_type: str = 'M01',
                          mag_type: str = 'global', sublattice: str = 'SU2') -> np.ndarray:
        """
        Get delay-dependent magnetization trajectory.
        
        Args:
            tau_index: Index of the delay time
            trajectory_type: 'M1' (probe only) or 'M01' (pump + probe)
            mag_type: 'antiferro', 'local', or 'global'
            sublattice: 'SU2' or 'SU3'
            
        Returns:
            [n_times, spin_dim] array
        """
        dataset_name = f'{trajectory_type}_{mag_type}_{sublattice}'
        return self._file[f'tau_scan/tau_{tau_index}/{dataset_name}'][:]
    
    def get_tau_value(self, tau_index: int) -> float:
        """Get the tau value for a specific index."""
        tau_group = self._file[f'tau_scan/tau_{tau_index}']
        return tau_group.attrs['tau_value']
    
    def get_n_tau_steps(self) -> int:
        """Get the number of tau steps."""
        return len(self.get_tau_values())
    
    def get_positions(self, sublattice: str = 'SU2') -> Optional[np.ndarray]:
        """Get site positions if available."""
        dataset_name = f'positions_{sublattice}'
        if dataset_name in self._file['metadata']:
            return self._file[f'metadata/{dataset_name}'][:]
        return None


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def read_MD_hdf5(filepath: str, w0: float = 0, wmax: float = 70,
                 output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Read molecular dynamics trajectory from HDF5 and compute DSSF.
    
    Args:
        filepath: Path to trajectory.h5 file
        w0: Minimum frequency
        wmax: Maximum frequency
        output_dir: Directory for output plots (uses filepath directory if None)
        
    Returns:
        Dictionary with DSSF results for SU2 and SU3
    """
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
    
    results = {}
    
    with MDTrajectoryHDF5(filepath) as reader:
        reciprocal_lattice = get_reciprocal_lattice_orthorhombic()
        DSSF_K, tick_positions = generate_k_path(reciprocal_lattice)
        g1, g2, g3, g4 = tick_positions
        
        # Process SU2
        if 'trajectory_SU2' in reader._file:
            T = reader.get_times('SU2')
            S = reader.get_spins('SU2')
            P = reader.get_positions('SU2')
            
            if P is None:
                print("Warning: No positions found for SU2. Using default.")
                P = np.zeros((S.shape[1], 3))
            
            # Compute frequency grid
            dt = T[1] - T[0] if len(T) > 1 else 1.0
            fft_freqs = np.fft.fftfreq(len(T), d=dt) * 2 * np.pi
            w_mask = (fft_freqs >= w0) & (fft_freqs <= wmax)
            w = fft_freqs[w_mask]
            
            # Compute DSSF in local frame
            A_SU2_local = DSSF(w, DSSF_K, S, P, T, global_frame=False)
            results['DSSF_SU2_local'] = A_SU2_local
            results['w'] = w
            
            # Compute DSSF in global frame
            A_SU2_global = DSSF(w, DSSF_K, S, P, T, global_frame=True)
            results['DSSF_SU2_global'] = A_SU2_global
            
            # Plot individual components - local frame
            spin_dim_SU2 = reader.metadata_SU2.get('spin_dim', 3)
            _plot_DSSF_components(A_SU2_local, w, tick_positions, output_dir, 'SU2_local', 
                                  min(3, spin_dim_SU2), w0, wmax)
            
            # Plot individual components - global frame
            _plot_DSSF_components(A_SU2_global, w, tick_positions, output_dir, 'SU2_global', 
                                  3, w0, wmax)
            
            # Compute and save gap at Gamma - local frame
            Gamma_point = np.array([[0, 0, 0]])
            A_Gamma_local = DSSF(w, Gamma_point, S, P, T, global_frame=False)
            DSSF_sum_Gamma_local = contract('wiab->wi', A_Gamma_local)
            _plot_gap_analysis(w, DSSF_sum_Gamma_local, output_dir, 'SU2_local')
            
            # Compute and save gap at Gamma - global frame
            A_Gamma_global = DSSF(w, Gamma_point, S, P, T, global_frame=True)
            DSSF_sum_Gamma_global = contract('wiab->wi', A_Gamma_global)
            _plot_gap_analysis(w, DSSF_sum_Gamma_global, output_dir, 'SU2_global')
        
        # Process SU3
        if 'trajectory_SU3' in reader._file:
            T = reader.get_times('SU3')
            S = reader.get_spins('SU3')
            P = reader.get_positions('SU3')
            
            if P is None:
                print("Warning: No positions found for SU3. Using default.")
                P = np.zeros((S.shape[1], 3))
            
            # Compute frequency grid
            dt = T[1] - T[0] if len(T) > 1 else 1.0
            fft_freqs = np.fft.fftfreq(len(T), d=dt) * 2 * np.pi
            w_mask = (fft_freqs >= w0) & (fft_freqs <= wmax)
            w = fft_freqs[w_mask]
            
            # Compute DSSF
            A_SU3 = DSSF(w, DSSF_K, S, P, T)
            results['DSSF_SU3'] = A_SU3
            
            # Plot individual components
            spin_dim_SU3 = reader.metadata_SU3.get('spin_dim', 8)
            _plot_DSSF_components(A_SU3, w, tick_positions, output_dir, 'SU3',
                                  min(8, spin_dim_SU3), w0, wmax)
            
            # Compute and save gap at Gamma
            Gamma_point = np.array([[0, 0, 0]])
            A_Gamma = DSSF(w, Gamma_point, S, P, T)
            DSSF_sum_Gamma = contract('wiab->wi', A_Gamma)
            _plot_gap_analysis(w, DSSF_sum_Gamma, output_dir, 'SU3')
        
        # Combined DSSF if both exist
        if 'DSSF_SU2_local' in results and 'DSSF_SU3' in results:
            # Combined local frame
            A_combined_local = contract('wiab->wi', results['DSSF_SU2_local']) + \
                              contract('wiab->wi', results['DSSF_SU3'])
            results['DSSF_combined_local'] = A_combined_local
            
            # Save and plot combined local
            _save_spectrum(os.path.join(output_dir, "DSSF_local"), A_combined_local)
            _plot_DSSF_combined(A_combined_local, w, tick_positions, output_dir, w0, wmax)
            
        if 'DSSF_SU2_global' in results and 'DSSF_SU3' in results:
            # Combined with global frame for SU2
            A_combined_global = contract('wiab->wi', results['DSSF_SU2_global']) + \
                               contract('wiab->wi', results['DSSF_SU3'])
            results['DSSF_combined_global'] = A_combined_global
            
            # Plot combined global
            fig, ax = plt.subplots(figsize=(10, 4))
            g1, g2, g3, g4 = tick_positions
            labels = [r'$(0,0,0)$', r'$(0,0,1)$', r'$(0,1,1)$', r'$(1,1,1)$']
            C = ax.imshow(A_combined_global, origin='lower', extent=[0, g4, w0, wmax],
                         aspect='auto', interpolation='gaussian', cmap='gnuplot2')
            ax.axvline(x=g1, color='b', linestyle='dashed')
            ax.axvline(x=g2, color='b', linestyle='dashed')
            ax.axvline(x=g3, color='b', linestyle='dashed')
            ax.axvline(x=g4, color='b', linestyle='dashed')
            ax.set_xticks([g1, g2, g3, g4])
            ax.set_xticklabels(labels)
            ax.set_xlim([0, g4])
            fig.colorbar(C)
            plt.savefig(os.path.join(output_dir, "DSSF_global.pdf"))
            plt.close()
            _save_spectrum(os.path.join(output_dir, "DSSF_global"), A_combined_global)
    
    return results


def _plot_DSSF_components(A: np.ndarray, w: np.ndarray, tick_positions: List[int],
                          output_dir: str, sublattice: str, n_components: int,
                          w0: float, wmax: float):
    """Plot individual DSSF components."""
    g1, g2, g3, g4 = tick_positions
    labels = [r'$(0,0,0)$', r'$(0,0,1)$', r'$(0,1,1)$', r'$(1,1,1)$']
    
    for i in range(n_components):
        fig, ax = plt.subplots(figsize=(10, 4))
        C = ax.imshow(A[:, :, i, i], origin='lower', 
                     extent=[0, g4, w0, wmax],
                     aspect='auto', interpolation='lanczos', cmap='gnuplot2', norm='log')
        ax.axvline(x=g1, color='b', linestyle='dashed')
        ax.axvline(x=g2, color='b', linestyle='dashed')
        ax.axvline(x=g3, color='b', linestyle='dashed')
        ax.axvline(x=g4, color='b', linestyle='dashed')
        ax.set_xticks([g1, g2, g3, g4])
        ax.set_xticklabels(labels)
        ax.set_xlim([0, g4])
        fig.colorbar(C)
        plt.savefig(os.path.join(output_dir, f"DSSF_{sublattice}_{i}_{i}.pdf"))
        plt.close()
    
    # Sum of all components
    DSSF_sum = contract('wiab->wi', A)
    fig, ax = plt.subplots(figsize=(10, 4))
    C = ax.imshow(DSSF_sum, origin='lower', extent=[0, g4, w0, wmax],
                 aspect='auto', interpolation='gaussian', cmap='gnuplot2')
    ax.axvline(x=g1, color='b', linestyle='dashed')
    ax.axvline(x=g2, color='b', linestyle='dashed')
    ax.axvline(x=g3, color='b', linestyle='dashed')
    ax.axvline(x=g4, color='b', linestyle='dashed')
    ax.set_xticks([g1, g2, g3, g4])
    ax.set_xticklabels(labels)
    ax.set_xlim([0, g4])
    fig.colorbar(C)
    plt.savefig(os.path.join(output_dir, f"DSSF_{sublattice}_sum.pdf"))
    plt.close()


def _plot_gap_analysis(w: np.ndarray, DSSF_Gamma: np.ndarray, output_dir: str, 
                       sublattice: str):
    """Plot DSSF at Gamma point for gap analysis."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(w, DSSF_Gamma[:, 0])
    ax.set_xlim([-3, 3])
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$S(\Gamma, \omega)$')
    plt.savefig(os.path.join(output_dir, f"DSSF_{sublattice}_gap_Gamma.pdf"))
    plt.close()
    np.savetxt(os.path.join(output_dir, f"DSSF_{sublattice}_gap_Gamma.txt"),
              np.column_stack((w, DSSF_Gamma[:, 0])))


def _plot_DSSF_combined(A: np.ndarray, w: np.ndarray, tick_positions: List[int],
                        output_dir: str, w0: float, wmax: float):
    """Plot combined DSSF from both sublattices."""
    g1, g2, g3, g4 = tick_positions
    labels = [r'$(0,0,0)$', r'$(0,0,1)$', r'$(0,1,1)$', r'$(1,1,1)$']
    
    fig, ax = plt.subplots(figsize=(10, 4))
    C = ax.imshow(A, origin='lower', extent=[0, g4, w0, wmax],
                 aspect='auto', interpolation='gaussian', cmap='gnuplot2')
    ax.axvline(x=g1, color='b', linestyle='dashed')
    ax.axvline(x=g2, color='b', linestyle='dashed')
    ax.axvline(x=g3, color='b', linestyle='dashed')
    ax.axvline(x=g4, color='b', linestyle='dashed')
    ax.set_xticks([g1, g2, g3, g4])
    ax.set_xticklabels(labels)
    ax.set_xlim([0, g4])
    fig.colorbar(C)
    plt.savefig(os.path.join(output_dir, "DSSF.pdf"))
    plt.close()


def _validate_window(window, name):
    """Ensure an omega window is either None or a valid (min, max) tuple."""
    if window is None:
        return None
    if (not isinstance(window, (tuple, list)) or len(window) != 2 or
            window[0] >= window[1]):
        raise ValueError(f"{name} must be a (min, max) tuple with min < max")
    return tuple(window)


def _get_norm(data: np.ndarray, norm_type: NormType = 'log', gamma: float = 0.5, 
              linthresh: float = 1e-3) -> Optional[Normalize]:
    """Get normalization for imshow based on the specified type.
    
    Args:
        data: The data array to normalize
        norm_type: Type of normalization:
            - 'log': LogNorm (logarithmic scaling, good for data with large dynamic range)
            - 'power': PowerNorm (power-law scaling with gamma parameter)
            - 'symlog': SymLogNorm (symmetric log, good for data with positive and negative values)
            - 'linear': Linear normalization (no scaling)
        gamma: Exponent for PowerNorm (default: 0.5, i.e., square root scaling)
        linthresh: Linear threshold for SymLogNorm (default: 1e-3)
    
    Returns:
        Matplotlib Normalize object or None for default linear normalization
    """
    # Get data range
    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        return None
    
    data_min = finite_data.min()
    data_max = finite_data.max()
    
    if data_min >= data_max:
        # All values are the same, use linear norm
        return None
    
    if norm_type == 'log':
        # Get positive values only for log norm
        positive_data = data[data > 0]
        if len(positive_data) == 0:
            return None
        vmin = positive_data.min()
        vmax = positive_data.max()
        if vmin >= vmax:
            return None
        return LogNorm(vmin=vmin, vmax=vmax)
    
    elif norm_type == 'power':
        # PowerNorm with specified gamma
        # For data that includes zero or negative, shift to positive
        if data_min <= 0:
            vmin = 0
            vmax = data_max - data_min
        else:
            vmin = data_min
            vmax = data_max
        return PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    
    elif norm_type == 'symlog':
        # SymLogNorm for data with both positive and negative values
        # Automatically determine linthresh if data is very small
        abs_max = max(abs(data_min), abs(data_max))
        if abs_max > 0:
            auto_linthresh = min(linthresh, abs_max * 0.01)
        else:
            auto_linthresh = linthresh
        return SymLogNorm(linthresh=auto_linthresh, vmin=data_min, vmax=data_max)
    
    elif norm_type == 'linear':
        return Normalize(vmin=data_min, vmax=data_max)
    
    else:
        # Unknown norm type, return None (linear)
        print(f"Warning: Unknown norm_type '{norm_type}', using linear normalization")
        return None


def _get_safe_log_norm(data: np.ndarray):
    """Get a safe log normalization for imshow, handling zeros and edge cases.
    
    DEPRECATED: Use _get_norm(data, 'log') instead.
    
    Returns None for linear normalization if log is not appropriate.
    """
    return _get_norm(data, 'log')


def _load_and_plot_cached_results(dir: str, omega_t_window: Optional[Tuple[float, float]],
                                   omega_tau_window: Optional[Tuple[float, float]],
                                   norm_type: NormType, 
                                   energy_levels_mev: Optional[Dict[str, float]]) -> Dict[str, np.ndarray]:
    """Load previously computed results from .txt files and regenerate plots only.
    
    Args:
        dir: Directory containing saved .txt files
        omega_t_window: ω_t axis limits
        omega_tau_window: ω_τ axis limits
        norm_type: Normalization type
        energy_levels_mev: Energy level dictionary
        
    Returns:
        Dictionary with loaded results
    """
    results = {}
    
    # Load HDF5 to get omega grids (reconstruct from saved data dimensions)
    hdf5_path = os.path.join(dir, "pump_probe_spectroscopy.h5")
    with h5py.File(hdf5_path, 'r') as f:
        tau_values = f['/tau_scan/tau_values'][:]
        times = f['/reference/times'][:]
        
    # Load a sample FFT file to get actual dimensions (FFT results may have been filtered)
    sample_names = ['M_NL_SU2_FF', 'M_NL_SU3_FF', 'M_NL_combined_FF']
    sample_data = None
    for sample_name in sample_names:
        sample_data = _load_spectrum(os.path.join(dir, sample_name))
        if sample_data is not None:
            break
    
    if sample_data is None:
        print("    Error: No cached FFT results found. Please run without --load-results first.")
        return results
    
    # Reconstruct frequency grids based on actual FFT dimensions
    n_tau_fft, n_t_fft = sample_data.shape
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    dtau = tau_values[1] - tau_values[0] if len(tau_values) > 1 else 1.0
    
    # Create frequency grids matching FFT output dimensions
    omega_tau = np.fft.fftfreq(n_tau_fft, dtau) * 2 * np.pi * MEV_TO_THZ
    omega_tau = np.fft.fftshift(omega_tau)
    
    omega_t = np.fft.fftfreq(n_t_fft, dt) * 2 * np.pi * MEV_TO_THZ
    omega_t = np.fft.fftshift(omega_t)
    
    results['omega_tau'] = omega_tau
    results['omega_t'] = omega_t
    
    # Calculate figure size based on axis ranges
    tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
    t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
    figsize = calculate_spectrum_figsize(tau_range, t_range)
    
    # List of possible result files to load and plot
    print(f"\n  Loaded {len(results)} cached result files (omega grids only; no plots generated)")
    return results


def _reload_and_compute_composites(dir: str, omega_t_window: Optional[Tuple[float, float]],
                                    omega_tau_window: Optional[Tuple[float, float]],
                                    norm_type: NormType,
                                    energy_levels_mev: Optional[Dict[str, float]]) -> Dict[str, np.ndarray]:
    """Load component FFTs from cache and regenerate the two components_debug PDFs.
    
    Loads per-component FFT .npy files (x, y, z for SU2; λ1-λ8 for SU3) and
    regenerates M_NL_SU2_components_debug.pdf and M_NL_SU3_components_debug.pdf.
    
    Args:
        dir: Directory containing saved component .npy files
        omega_t_window: ω_t axis limits
        omega_tau_window: ω_τ axis limits
        norm_type: Normalization type
        energy_levels_mev: Energy level dictionary
        
    Returns:
        Dictionary with omega grids and loaded component arrays
    """
    results = {}
    
    # Load HDF5 to get omega grids
    hdf5_path = os.path.join(dir, "pump_probe_spectroscopy.h5")
    with h5py.File(hdf5_path, 'r') as f:
        tau_values = f['/tau_scan/tau_values'][:]
        times = f['/reference/times'][:]
    
    def try_load_component(name):
        return _load_spectrum(os.path.join(dir, name))
    
    # Load SU(2) component FFTs (x, y, z)
    component_labels_SU2 = ['x', 'y', 'z']
    M_NL_SU2_components = []
    for label in component_labels_SU2:
        comp = try_load_component(f"M_NL_SU2_FF_{label}")
        if comp is not None:
            M_NL_SU2_components.append((label, comp))
    
    # Load SU(3) component FFTs (\u03bb1-\u03bb8)
    component_labels_SU3 = ['\u03bb1', '\u03bb2', '\u03bb3', '\u03bb4', '\u03bb5', '\u03bb6', '\u03bb7', '\u03bb8']
    M_NL_SU3_components = []
    for label in component_labels_SU3:
        comp = try_load_component(f"M_NL_SU3_FF_{label}")
        if comp is not None:
            M_NL_SU3_components.append((label, comp))
    
    # Get shape from first available component
    all_components = M_NL_SU2_components + M_NL_SU3_components
    if not all_components:
        print("    Error: No component FFT files found. Run full calculation first.")
        return results
    sample_data = all_components[0][1]
    
    # Reconstruct frequency grids
    n_tau_fft, n_t_fft = sample_data.shape
    dt   = times[1] - times[0]       if len(times)      > 1 else 1.0
    dtau = tau_values[1] - tau_values[0] if len(tau_values) > 1 else 1.0
    omega_tau = np.fft.fftshift(np.fft.fftfreq(n_tau_fft, dtau) * 2 * np.pi * MEV_TO_THZ)
    omega_t   = np.fft.fftshift(np.fft.fftfreq(n_t_fft,   dt)   * 2 * np.pi * MEV_TO_THZ)
    results['omega_tau'] = omega_tau
    results['omega_t']   = omega_t
    
    # Generate M_NL_SU2_components_debug.pdf
    n_rows_su2 = min(len(M_NL_SU2_components), 3)
    if n_rows_su2 > 0:
        fig_su2, axes_su2 = plt.subplots(n_rows_su2, 1, figsize=(5, 3.5 * n_rows_su2))
        if n_rows_su2 == 1:
            axes_su2 = [axes_su2]
        for row, (label, M_NL_comp_FF) in enumerate(M_NL_SU2_components[:n_rows_su2]):
            ax = axes_su2[row]
            im = ax.imshow(M_NL_comp_FF, origin='lower',
                           extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
                           aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_comp_FF, norm_type))
            ax.set_xlabel('$\\omega_t$ (THz)')
            ax.set_ylabel('$\\omega_{\\tau}$ (THz)')
            ax.set_title(f'$M_{{NL}}$ ({label})')
            if omega_t_window is not None:
                ax.set_xlim(omega_t_window[0], omega_t_window[1])
            if omega_tau_window is not None:
                ax.set_ylim(omega_tau_window)
            if energy_levels_mev:
                add_energy_level_lines(ax, energy_levels_mev, omega_t_window)
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, "M_NL_SU2_components_debug.pdf"), dpi=100)
        plt.close(fig_su2)
        print(f"    Generated: M_NL_SU2_components_debug.pdf")
    
    # Generate M_NL_SU3_components_debug.pdf
    n_rows_su3 = min(len(M_NL_SU3_components), 8)
    if n_rows_su3 > 0:
        fig_su3, axes_su3 = plt.subplots(n_rows_su3, 1, figsize=(5, 3 * n_rows_su3))
        if n_rows_su3 == 1:
            axes_su3 = [axes_su3]
        for row, (label, M_NL_comp_FF) in enumerate(M_NL_SU3_components[:n_rows_su3]):
            ax = axes_su3[row]
            im = ax.imshow(M_NL_comp_FF, origin='lower',
                           extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
                           aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_comp_FF, norm_type))
            ax.set_xlabel('$\\omega_t$ (THz)')
            ax.set_ylabel('$\\omega_{\\tau}$ (THz)')
            ax.set_title(f'$M_{{NL}}$ ({label})')
            if omega_t_window is not None:
                ax.set_xlim(omega_t_window[0], omega_t_window[1])
            if omega_tau_window is not None:
                ax.set_ylim(omega_tau_window)
            if energy_levels_mev:
                add_energy_level_lines(ax, energy_levels_mev, omega_t_window)
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(dir, "M_NL_SU3_components_debug.pdf"), dpi=100)
        plt.close(fig_su3)
        print(f"    Generated: M_NL_SU3_components_debug.pdf")
    
    return results

def read_2D_nonlinear(dir: str, omega_t_window: Optional[Tuple[float, float]] = None,
                      omega_tau_window: Optional[Tuple[float, float]] = None,
                      norm_type: NormType = 'power',
                      apodization_gamma: float = 0.03,
                      pulse_window_sigma: float = 5.0,
                      window_type: str = 'gaussian',
                      energy_levels_mev: Optional[Dict[str, float]] = None,
                      load_from_cache: bool = False,
                      reload_components: bool = False,
                      save_intermediates: bool = False,
                      skip_plots: bool = False,
                      auto_window_margin: float = 2.0) -> Dict[str, np.ndarray]:
    """Read and compute 2D nonlinear spectroscopy using FFT for mixed SU(2)+SU(3) systems.
    
    Reads pump-probe spectroscopy data from HDF5 file and computes the nonlinear
    response M_NL = M01 - M0 - M1, then performs 2D FFT to get the 2D spectrum.
    
    Processes both SU(2) and SU(3) sublattices separately.
    
    Args:
        dir: Directory containing pump_probe_spectroscopy.h5
        omega_t_window: Optional (min, max) tuple for ω_t axis limits
        omega_tau_window: Optional (min, max) tuple for ω_τ axis limits
        norm_type: Normalization type for plots ('log', 'power', 'symlog', 'linear')
        apodization_gamma: Decay parameter for apodization window. For Gaussian: sets
            boundary decay level. For exponential: decay rate. Default: 0.03
        pulse_window_sigma: Number of pulse widths (sigma) after probe center (t=0) 
            to skip before FFT. Default: 5.0
        window_type: Type of apodization window to reduce spectral leakage:
            'gaussian', 'hann', 'hamming', 'blackman', 'tukey', 'cosine', 'exponential', 'none'
            Default: 'gaussian'
        energy_levels_mev: Optional dict of energy levels in meV to mark with dashed lines.
            Keys: 'e1', 'e2', 'e2_e1', 'kc'. Values: energy in meV.
            Lines are drawn at ±ω for each energy level.
        load_from_cache: If True, skip FFT calculations and load previously saved results
            from .txt files. Only regenerates plots. Default: False
        reload_components: If True, load component FFTs (individual λ components, x/y/z) from
            cache and recompute composite spectra. Faster than full calculation. Default: False
        save_intermediates: If True, save intermediate M0, M1, M01 spectra and debug plots.
            Default: False (only saves final M_NL and composite spectra)
        skip_plots: If True, skip all matplotlib figure generation for speed.
            Spectra are still saved as .npy files. Default: False
        auto_window_margin: When energy_levels_mev is provided but windows are None,
            auto-set windows to margin × max_peak_freq. Default: 2.0.
            Set to 0 to disable auto-windowing.
        
    Returns:
        Dictionary with 2DCS results for both sublattices
    """
    omega_t_window = _validate_window(omega_t_window, "omega_t_window")
    omega_tau_window = _validate_window(omega_tau_window, "omega_tau_window")
    
    # Convert windows from meV to THz if provided
    if omega_t_window is not None:
        omega_t_window = (omega_t_window[0] * MEV_TO_THZ, omega_t_window[1] * MEV_TO_THZ)
    if omega_tau_window is not None:
        omega_tau_window = (omega_tau_window[0] * MEV_TO_THZ, omega_tau_window[1] * MEV_TO_THZ)
    
    # Auto-window from energy levels if no explicit window given
    if auto_window_margin > 0 and energy_levels_mev and (omega_t_window is None or omega_tau_window is None):
        auto_t, auto_tau = auto_window_from_peaks(energy_levels_mev, margin=auto_window_margin)
        if omega_t_window is None and auto_t is not None:
            omega_t_window = auto_t
            print(f"  Auto-window ω_t: [{omega_t_window[0]:.2f}, {omega_t_window[1]:.2f}] THz "
                  f"(margin={auto_window_margin}× max peak)")
        if omega_tau_window is None and auto_tau is not None:
            omega_tau_window = auto_tau
            print(f"  Auto-window ω_τ: [{omega_tau_window[0]:.2f}, {omega_tau_window[1]:.2f}] THz")
    
    hdf5_path = os.path.join(dir, "pump_probe_spectroscopy.h5")
    
    # If loading from cache, skip all calculations and just regenerate plots
    if load_from_cache:
        print("\n  Loading results from cached .txt files...")
        return _load_and_plot_cached_results(dir, omega_t_window, omega_tau_window, 
                                             norm_type, energy_levels_mev)
    
    # If reloading components, load component FFTs and recompute composites
    if reload_components:
        print("\n  Loading component FFTs and recomputing composite spectra...")
        return _reload_and_compute_composites(dir, omega_t_window, omega_tau_window,
                                             norm_type, energy_levels_mev)
    
    # Print energy level info if provided
    if energy_levels_mev:
        print(f"  Energy level reference lines (meV → THz):")
        for name, val in energy_levels_mev.items():
            if val is not None:
                print(f"    {name}: {val:.4f} meV → ±{val * MEV_TO_THZ:.4f} THz")
    
    # Component labels
    component_labels_SU2 = ['x', 'y', 'z']
    component_labels_SU3 = ['λ1', 'λ2', 'λ3', 'λ4', 'λ5', 'λ6', 'λ7', 'λ8']
    
    results = {}
    
    # Read from HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        times = f['/reference/times'][:]
        tau_values = f['/tau_scan/tau_values'][:]
        tau_step = len(tau_values)
        
        # Print pulse and lattice metadata if available
        # if '/metadata' in f:
        #     metadata_grp = f['/metadata']
            # print(f"  Pulse parameters:")
            # for attr in ['pulse_amp', 'pulse_width', 'pulse_freq']:
            #     if attr in metadata_grp.attrs:
            #         print(f"    {attr}: {metadata_grp.attrs[attr]}")
            # print(f"  Lattice parameters:")
            # for attr in ['n_atoms', 'spin_dim', 'lattice_size']:
            #     if attr in metadata_grp.attrs:
            #         print(f"    {attr}: {metadata_grp.attrs[attr]}")
        
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        tau = tau_values
        
        # Extract pulse width for time cutoff calculation
        pulse_width = 1.0  # Default fallback
        if '/metadata' in f:
            if 'pulse_width' in f['/metadata'].attrs:
                pulse_width = float(f['/metadata'].attrs['pulse_width'])
            elif 'pulse_width_SU2' in f['/metadata'].attrs:
                pulse_width = float(f['/metadata'].attrs['pulse_width_SU2'])
        
        # Compute time cutoff: the probe is at t=0 (fixed second pulse).
        # For the FFT we only use t > t_cutoff (after the probe transient dies).
        # The time-domain images use the FULL times array including t < 0
        # (pre-probe region where the pump fired at t=τ < 0).
        t_cutoff = pulse_window_sigma * pulse_width
        t_valid_idx = np.where(times > t_cutoff)[0]
        if len(t_valid_idx) > 0:
            t0_idx = t_valid_idx[0]
            times_positive = times[t0_idx:]
            print(f"  Filtering to t > {t_cutoff:.2f} (={pulse_window_sigma}σ × pulse_width={pulse_width}): "
                  f"using {len(times_positive)}/{len(times)} time points (t0_idx={t0_idx})")
        else:
            t0_idx = 0
            times_positive = times
            print(f"  Warning: No t > {t_cutoff:.2f} found, using all time points")
        
        # Compute omega arrays (using filtered time array length)
        # First compute in rad/time, then convert to THz
        omega_tau = np.fft.fftfreq(int(len(tau)), tau[1] - tau[0] if len(tau) > 1 else 1.0) * 2 * np.pi
        omega_tau = np.fft.fftshift(omega_tau)
        omega_tau = omega_tau * MEV_TO_THZ  # Convert meV (=rad/ps with ℏ=1) to THz
        
        omega_t = np.fft.fftfreq(len(times_positive), dt) * 2 * np.pi
        omega_t = np.fft.fftshift(omega_t)
        # omega_t now goes from negative to positive (standard convention)
        omega_t = omega_t * MEV_TO_THZ  # Convert meV (=rad/ps with ℏ=1) to THz
        
        results['omega_tau'] = omega_tau
        results['omega_t'] = omega_t
        results['tau_values'] = tau_values
        
        # Nyquist frequencies
        f_nyq_tau = 1.0 / (2.0 * abs(tau[1] - tau[0])) * 2 * np.pi * MEV_TO_THZ
        f_nyq_t = 1.0 / (2.0 * dt) * 2 * np.pi * MEV_TO_THZ
        print(f"  Nyquist frequencies:  f_Nyq(\u03c4) = {f_nyq_tau:.2f} THz,  f_Nyq(t) = {f_nyq_t:.2f} THz")
        
        # Build apodization window based on window_type
        # For t: use relative time from start of window so decay starts at 1.0
        # For tau: use distance from tau=0 (probe time) so max weight is at tau closest to 0
        n_t = len(times_positive)
        n_tau = len(tau)
        
        if window_type.lower() == 'none' or apodization_gamma <= 0:
            apod_window = None
            print(f"  No apodization applied (window_type='{window_type}')")
        elif window_type.lower() == 'gaussian':
            t_relative = times_positive - times_positive[0]
            t_range = t_relative[-1] if len(t_relative) > 1 else 1.0
            tau_range = max(np.abs(tau[0]), np.abs(tau[-1])) if len(tau) > 1 else 1.0
            
            decay_factor = np.sqrt(-2.0 * np.log(apodization_gamma))
            sigma_t = t_range / decay_factor
            sigma_tau = tau_range / decay_factor
            
            decay_t = np.exp(-0.5 * (t_relative / sigma_t) ** 2)
            decay_tau = np.exp(-0.5 * (np.abs(tau) / sigma_tau) ** 2)
            apod_window = np.outer(decay_tau, decay_t)
            print(f"  Applying Gaussian apodization (σ_t={sigma_t:.2f}, σ_τ={sigma_tau:.2f})")
            print(f"    t window: 1.0 at t={times_positive[0]:.1f} → {decay_t[-1]:.4f} at t={times_positive[-1]:.1f}")
            print(f"    τ window: {decay_tau[0]:.4f} at τ={tau[0]:.1f} → {decay_tau[-1]:.4f} at τ={tau[-1]:.1f}")
        elif window_type.lower() == 'exponential':
            t_relative = times_positive - times_positive[0]
            decay_t = np.exp(-apodization_gamma * t_relative)
            decay_tau = np.exp(-apodization_gamma * np.abs(tau))
            apod_window = np.outer(decay_tau, decay_t)
            print(f"  Applying exponential apodization (γ={apodization_gamma})")
            print(f"    t decay: 1.0 at t={times_positive[0]:.1f} → {decay_t[-1]:.4f} at t={times_positive[-1]:.1f}")
            print(f"    τ decay: {decay_tau[0]:.4f} at τ={tau[0]:.1f} → {decay_tau[-1]:.4f} at τ={tau[-1]:.1f}")
        elif window_type.lower() == 'hann':
            window_t = 0.5 * (1 + np.cos(np.pi * np.arange(n_t) / (n_t - 1)))
            window_tau = 0.5 * (1 + np.cos(np.pi * np.abs(tau) / np.max(np.abs(tau))))
            apod_window = np.outer(window_tau, window_t)
            print(f"  Applying Hann apodization")
            print(f"    t window: {window_t[0]:.4f} at t={times_positive[0]:.1f} → {window_t[-1]:.4f} at t={times_positive[-1]:.1f}")
            print(f"    τ window: {window_tau[0]:.4f} at τ={tau[0]:.1f} → {window_tau[-1]:.4f} at τ={tau[-1]:.1f}")
        elif window_type.lower() == 'hamming':
            window_t = 0.54 + 0.46 * np.cos(np.pi * np.arange(n_t) / (n_t - 1))
            window_tau = 0.54 + 0.46 * np.cos(np.pi * np.abs(tau) / np.max(np.abs(tau)))
            apod_window = np.outer(window_tau, window_t)
            print(f"  Applying Hamming apodization")
            print(f"    t window: {window_t[0]:.4f} at t={times_positive[0]:.1f} → {window_t[-1]:.4f} at t={times_positive[-1]:.1f}")
            print(f"    τ window: {window_tau[0]:.4f} at τ={tau[0]:.1f} → {window_tau[-1]:.4f} at τ={tau[-1]:.1f}")
        elif window_type.lower() == 'blackman':
            n_norm_t = np.arange(n_t) / (n_t - 1)
            n_norm_tau = np.abs(tau) / np.max(np.abs(tau))
            window_t = 0.42 + 0.5 * np.cos(np.pi * n_norm_t) + 0.08 * np.cos(2 * np.pi * n_norm_t)
            window_tau = 0.42 + 0.5 * np.cos(np.pi * n_norm_tau) + 0.08 * np.cos(2 * np.pi * n_norm_tau)
            apod_window = np.outer(window_tau, window_t)
            print(f"  Applying Blackman apodization")
            print(f"    t window: {window_t[0]:.4f} at t={times_positive[0]:.1f} → {window_t[-1]:.4f} at t={times_positive[-1]:.1f}")
            print(f"    τ window: {window_tau[0]:.4f} at τ={tau[0]:.1f} → {window_tau[-1]:.4f} at τ={tau[-1]:.1f}")
        elif window_type.lower() == 'tukey':
            alpha = min(1.0, max(0.0, apodization_gamma * 10))
            window_t = np.ones(n_t)
            window_tau = np.ones(n_tau)
            taper_len_t = int(alpha * n_t / 2)
            if taper_len_t > 0:
                taper = 0.5 * (1 + np.cos(np.pi * np.arange(taper_len_t) / taper_len_t))
                window_t[-taper_len_t:] = taper
            taper_len_tau = int(alpha * n_tau / 2)
            if taper_len_tau > 0:
                taper = 0.5 * (1 + np.cos(np.pi * np.arange(taper_len_tau) / taper_len_tau))
                # Symmetric taper on BOTH τ edges so a centred [-τ_max, +τ_max]
                # grid does not leak from the +τ side (audit C1).
                window_tau[:taper_len_tau] = taper[::-1]
                window_tau[-taper_len_tau:] = taper
            apod_window = np.outer(window_tau, window_t)
            print(f"  Applying Tukey apodization (α={alpha:.2f}, taper={100*alpha/2:.0f}%)")
            print(f"    t window: {window_t[0]:.4f} at t={times_positive[0]:.1f} → {window_t[-1]:.4f} at t={times_positive[-1]:.1f}")
            print(f"    τ window: {window_tau[0]:.4f} at τ={tau[0]:.1f} → {window_tau[-1]:.4f} at τ={tau[-1]:.1f}")
        elif window_type.lower() == 'cosine':
            window_t = np.cos(0.5 * np.pi * np.arange(n_t) / (n_t - 1))
            window_tau = np.cos(0.5 * np.pi * np.abs(tau) / np.max(np.abs(tau)))
            apod_window = np.outer(window_tau, window_t)
            print(f"  Applying Cosine apodization")
            print(f"    t window: {window_t[0]:.4f} at t={times_positive[0]:.1f} → {window_t[-1]:.4f} at t={times_positive[-1]:.1f}")
            print(f"    τ window: {window_tau[0]:.4f} at τ={tau[0]:.1f} → {window_tau[-1]:.4f} at τ={tau[-1]:.1f}")
        else:
            print(f"  Warning: Unknown window_type '{window_type}', using no apodization")
            print(f"    Available: gaussian, exponential, hann, hamming, blackman, tukey, cosine, none")
            apod_window = None
        
        # =====================================================================
        # 2D-FFT helpers (Ingredient XIX)
        # =====================================================================
        # Caches keep BOTH complex spectra and magnitudes:
        #   _cfft_cache[key] -> complex 2D FFT (after fftshift + ω_τ flip)
        #   _fft_cache[key]  -> magnitude    (lazy-derived from _cfft_cache[key])
        #
        # Caching the complex spectrum (instead of just |·|) lets us synthesise
        # any *linear* combination of components (the λ-mode mixtures below) by
        # one weighted sum + one |·|, instead of building the time-domain combo
        # and re-running fft2.  This saves ~20 fft2 calls per 2DCS analysis on
        # the SU(3) lattice.
        _cfft_cache = {}
        _fft_cache = {}

        def _apodize_filter(data):
            """Filter to t >= t_cutoff and apply mean subtraction + apodization."""
            data_filtered = data[..., t0_idx:]
            # Mean subtraction across the full (tau, t) grid (matches axis=None
            # behaviour of np.mean) — preserved per-component for batched arrays.
            mean_axes = (-2, -1)
            data_dynamic = data_filtered - data_filtered.mean(axis=mean_axes, keepdims=True)
            if apod_window is not None:
                data_dynamic = data_dynamic * apod_window  # broadcasts across batch
            return data_dynamic

        def _fft2_shift(data_dynamic, axes=(-2, -1)):
            """Run fft2+fftshift on the trailing two axes (uses scipy when available)."""
            if _USE_SCIPY_FFT:
                cFFT = _scipy_fft.fft2(data_dynamic, axes=axes, workers=-1)
                cFFT = _scipy_fft.fftshift(cFFT, axes=axes)
            else:
                cFFT = np.fft.fft2(data_dynamic, axes=axes)
                cFFT = np.fft.fftshift(cFFT, axes=axes)
            return cFFT

        def _flip_omega_tau(arr):
            """Flip axis -2 (ω_τ) to correct for τ → T = −τ sign convention."""
            return arr[..., ::-1, :]

        def compute_2d_fft(data, cache_key=None):
            """Compute |2D FFT| with apodization, caching the complex spectrum.

            Uses scipy.fft (workers=-1) when available.

            Input shape: (n_tau, n_t)
            Output shape: (n_tau, n_t_positive) after fftshift+ω_τ-flip, so:
              - axis 0 (rows) = ω_τ  →  y-axis in imshow(origin='lower')
              - axis 1 (cols) = ω_t  →  x-axis in imshow(origin='lower')

            The ω_τ axis is flipped after fftshift to convert simulation
            τ ∈ [−τ_max, 0] to physical delay T = −τ ∈ [0, τ_max].
            """
            if cache_key is not None and cache_key in _fft_cache:
                return _fft_cache[cache_key]
            if cache_key is not None and cache_key in _cfft_cache:
                mag = np.abs(_cfft_cache[cache_key])
                _fft_cache[cache_key] = mag
                return mag

            data_dynamic = _apodize_filter(data)
            cFFT = _flip_omega_tau(_fft2_shift(data_dynamic))
            mag = np.abs(cFFT)
            if cache_key is not None:
                _cfft_cache[cache_key] = cFFT
                _fft_cache[cache_key] = mag
            return mag

        def batch_compute_2d_fft(data_3d, key_prefix):
            """Batched fft2 over the leading axis: data_3d has shape
            (n_components, n_tau, n_t).  Returns the per-component magnitude
            spectra as a list, and stores the complex spectra in _cfft_cache
            under keys '{key_prefix}_{c}' for c=0..n_components-1.

            One scipy.fft.fft2 with axes=(-2,-1), workers=-1 is much faster
            than calling compute_2d_fft in a loop because (a) FFT planning
            is amortised, (b) thread fan-out is over the whole batch, and
            (c) Python-level overhead is paid once.
            """
            data_dynamic = _apodize_filter(data_3d)
            cFFT = _flip_omega_tau(_fft2_shift(data_dynamic))
            mags = np.abs(cFFT)
            # Store per-component slices in cache (cheap: views, not copies).
            for c in range(data_3d.shape[0]):
                ck = f'{key_prefix}_{c}'
                _cfft_cache[ck] = cFFT[c]
                _fft_cache[ck] = mags[c]
            return [mags[c] for c in range(data_3d.shape[0])]

        def compute_2d_fft_combo(combos):
            """Magnitude of a linear combination of cached complex spectra.

            ``combos`` is a list of ``(weight, cache_key)`` pairs.  Returns
            ``|sum_i w_i * cFFT[key_i]|`` — exactly equivalent to building
            ``sum_i w_i * x_i`` in the time domain and running compute_2d_fft
            on that, because all of {filter, mean-subtract, apodize, fft2,
            fftshift, ω_τ-flip} are linear.  Falls back to None if any key
            is missing (caller should then build the combo in the time
            domain and call compute_2d_fft).
            """
            cffts = []
            for w, k in combos:
                if k not in _cfft_cache:
                    return None
                cffts.append((w, _cfft_cache[k]))
            acc = cffts[0][0] * cffts[0][1]
            for w, c in cffts[1:]:
                acc = acc + w * c
            return np.abs(acc)

        def clear_fft_cache():
            """Clear the FFT caches to free memory."""
            _fft_cache.clear()
            _cfft_cache.clear()

        def save_intermediate_component_debug(prefix, component_labels, n_rows,
                                              intermediate_spectra, m_nl_spectra):
            """Save per-component M0/M1/M01 spectra and an intermediate debug grid."""
            ordered_groups = [
                ('M0', intermediate_spectra['M0']),
                ('M1', intermediate_spectra['M1']),
                ('M01', intermediate_spectra['M01']),
                ('M_NL = M01 - M0 - M1', m_nl_spectra),
            ]

            for key, spectra in intermediate_spectra.items():
                for comp in range(min(len(spectra), len(component_labels))):
                    label = component_labels[comp]
                    _save_spectrum(os.path.join(dir, f"{key}_{prefix}_FF_{label}"), spectra[comp])

            if skip_plots:
                return

            fig_debug, axes_debug = plt.subplots(
                n_rows,
                len(ordered_groups),
                figsize=(4.4 * len(ordered_groups), 3 * n_rows),
                squeeze=False,
            )

            for comp in range(n_rows):
                label = component_labels[comp] if comp < len(component_labels) else f'comp{comp}'
                for col, (title, spectra) in enumerate(ordered_groups):
                    ax_freq = axes_debug[comp, col]
                    spectrum = spectra[comp]
                    im_freq = ax_freq.imshow(
                        spectrum,
                        origin='lower',
                        aspect='auto',
                        cmap='gnuplot2',
                        norm=_get_norm(spectrum, norm_type),
                        extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
                    )
                    ax_freq.set_xlabel('$\\omega_t$ (THz)')
                    ax_freq.set_ylabel('$\\omega_{\\tau}$ (THz)')
                    ax_freq.set_title(f'{label}: {title}')
                    if omega_t_window is not None:
                        ax_freq.set_xlim(omega_t_window[0], omega_t_window[1])
                    if omega_tau_window is not None:
                        ax_freq.set_ylim(omega_tau_window)
                    if energy_levels_mev:
                        add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                    plt.colorbar(im_freq, ax=ax_freq)

            plt.tight_layout()
            plt.savefig(os.path.join(dir, f"M_intermediates_{prefix}_components_debug.pdf"), dpi=100)
            plt.close(fig_debug)
        
        # =====================================================================
        # Process SU(2) sublattice
        # =====================================================================
        if '/reference/M_global_SU2' in f:
            print("\n  Processing SU(2) sublattice...")
            M0_SU2 = f['/reference/M_global_SU2'][:]
            spin_dim_SU2 = M0_SU2.shape[1]
            length = len(M0_SU2[:, 0])

            # Pre-allocate (spin_dim, n_tau, n_t) arrays.  The M0/M1/M01 component
            # buffers are only consumed downstream when save_intermediates=True;
            # skipping them saves 3 × spin_dim × n_tau × n_t doubles of RAM
            # (~10 GB at production scale).
            M_NL_SU2  = np.zeros((spin_dim_SU2, tau_step, length))
            M01_td_SU2 = np.zeros((spin_dim_SU2, tau_step, length))  # for time-domain plot
            M1_td_SU2  = np.zeros((spin_dim_SU2, tau_step, length))  # for time-domain plot
            if save_intermediates:
                M0_comp_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
                M1_comp_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
                M01_comp_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
            else:
                M0_comp_SU2 = None
                M1_comp_SU2 = None
                M01_comp_SU2 = None

            # Vectorise the per-tau loop: one transpose then one slice per array
            # replaces an inner ``for comp in range(spin_dim_SU2)`` loop.
            M0_SU2_T = M0_SU2.T  # (spin_dim, n_t_M0)
            n_t_M0 = M0_SU2_T.shape[1]
            for i in range(tau_step):
                tau_group = f[f'/tau_scan/tau_{i}']
                M1_T = tau_group['M1_global_SU2'][:].T   # (spin_dim, n_t)
                M01_T = tau_group['M01_global_SU2'][:].T
                n = min(n_t_M0, M1_T.shape[1], M01_T.shape[1], length)
                M_NL_SU2[:, i, :n]  = M01_T[:, :n] - M0_SU2_T[:, :n] - M1_T[:, :n]
                M01_td_SU2[:, i, :n] = M01_T[:, :n]
                M1_td_SU2[:, i, :n]  = M1_T[:, :n]
                if save_intermediates:
                    M0_comp_SU2[:, i, :n] = M0_SU2_T[:, :n]
                    M1_comp_SU2[:, i, :n] = M1_T[:, :n]
                    M01_comp_SU2[:, i, :n] = M01_T[:, :n]
            
            if False:  # windowed_time_evolution_SU2_debug removed
                if not skip_plots:
                    fig_wind, axes_wind = plt.subplots(2, 3, figsize=(15, 8))
                
                    # Select a few representative tau values for plotting
                    n_tau_show = min(4, len(tau))
                    tau_show_indices = np.linspace(0, len(tau)-1, n_tau_show, dtype=int)
                    colors = plt.cm.viridis(np.linspace(0, 0.9, n_tau_show))
                
                    # Filter to t >= t_cutoff (same as FFT uses)
                    M_NL_filtered = M_NL_example[:, t0_idx:]
                    M_NL_dynamic = M_NL_filtered - M_NL_filtered.mean()
                    M_NL_windowed = M_NL_dynamic * apod_window
                
                    # Top-left: Original signal (unwindowed) vs t
                    ax0 = axes_wind[0, 0]
                    for i, idx in enumerate(tau_show_indices):
                        ax0.plot(times_positive, M_NL_dynamic[idx, :], color=colors[i], 
                                label=f'τ={tau[idx]:.1f}', alpha=0.8)
                    ax0.set_xlabel('t (ps)')
                    ax0.set_ylabel(f'$M_{{NL,{comp_label}}}$')
                    ax0.set_title('Original signal (mean subtracted)')
                    ax0.legend(fontsize=7)
                    ax0.grid(True, alpha=0.3)
                
                    # Top-middle: Windowed signal vs t
                    ax1 = axes_wind[0, 1]
                    for i, idx in enumerate(tau_show_indices):
                        ax1.plot(times_positive, M_NL_windowed[idx, :], color=colors[i],
                                label=f'τ={tau[idx]:.1f}', alpha=0.8)
                    ax1.set_xlabel('t (ps)')
                    ax1.set_ylabel(f'$M_{{NL,{comp_label}}}$ × window')
                    ax1.set_title(f'Windowed signal ({window_type})')
                    ax1.legend(fontsize=7)
                    ax1.grid(True, alpha=0.3)
                
                    # Top-right: Overlay comparison at mid-tau
                    ax2 = axes_wind[0, 2]
                    mid_tau = len(tau) // 2
                    ax2.plot(times_positive, M_NL_dynamic[mid_tau, :], 'b-', lw=1.5, 
                            label='Original', alpha=0.8)
                    ax2.plot(times_positive, M_NL_windowed[mid_tau, :], 'r--', lw=1.5,
                            label='Windowed', alpha=0.8)
                    ax2.set_xlabel('t (ps)')
                    ax2.set_ylabel(f'$M_{{NL,{comp_label}}}$')
                    ax2.set_title(f'Comparison @ τ={tau[mid_tau]:.1f}')
                    ax2.legend(fontsize=8)
                    ax2.grid(True, alpha=0.3)
                
                    # Bottom-left: 2D heatmap of original signal
                    ax3 = axes_wind[1, 0]
                    im3 = ax3.imshow(M_NL_dynamic, origin='lower', aspect='auto', cmap='RdBu_r',
                                    extent=[times_positive[0], times_positive[-1], tau[0], tau[-1]])
                    ax3.set_xlabel('t (ps)')
                    ax3.set_ylabel('τ (ps)')
                    ax3.set_title('Original (τ, t) domain')
                    plt.colorbar(im3, ax=ax3)
                
                    # Bottom-middle: 2D heatmap of windowed signal
                    ax4 = axes_wind[1, 1]
                    im4 = ax4.imshow(M_NL_windowed, origin='lower', aspect='auto', cmap='RdBu_r',
                                    extent=[times_positive[0], times_positive[-1], tau[0], tau[-1]])
                    ax4.set_xlabel('t (ps)')
                    ax4.set_ylabel('τ (ps)')
                    ax4.set_title(f'Windowed (τ, t) domain ({window_type})')
                    plt.colorbar(im4, ax=ax4)
                
                    # Bottom-right: Signal at fixed t across tau (shows τ-axis tapering)
                    ax5 = axes_wind[1, 2]
                    n_t_show = min(4, len(times_positive))
                    t_show_indices = np.linspace(0, len(times_positive)-1, n_t_show, dtype=int)
                    colors_t = plt.cm.plasma(np.linspace(0, 0.9, n_t_show))
                    for i, idx in enumerate(t_show_indices):
                        ax5.plot(tau, M_NL_dynamic[:, idx], color=colors_t[i], ls='-',
                                label=f't={times_positive[idx]:.1f} orig', alpha=0.5)
                        ax5.plot(tau, M_NL_windowed[:, idx], color=colors_t[i], ls='--',
                                label=f't={times_positive[idx]:.1f} wind', alpha=0.9)
                    ax5.set_xlabel('τ (ps)')
                    ax5.set_ylabel(f'$M_{{NL,{comp_label}}}$')
                    ax5.set_title('Signal vs τ (solid=orig, dashed=windowed)')
                    ax5.legend(fontsize=6, ncol=2)
                    ax5.grid(True, alpha=0.3)
                
                    plt.tight_layout()
                    plt.close(fig_wind)
            
            # M_NL analysis for SU2
            print("    Processing M_NL_SU2...")
            n_rows = min(spin_dim_SU2, 3)
            # Batched fft2 over the spin_dim axis (caches complex spectra so
            # the SU(2) x-component is reusable in λ-combo synthesis below).
            M_NL_SU2_FFs = batch_compute_2d_fft(M_NL_SU2, 'M_NL_SU2')
            
            # Save individual component FFTs
            for comp in range(min(spin_dim_SU2, len(component_labels_SU2))):
                label = component_labels_SU2[comp]
                _save_spectrum(os.path.join(dir, f"M_NL_SU2_FF_{label}"), M_NL_SU2_FFs[comp])

            if save_intermediates:
                print("    Processing M0/M1/M01_SU2...")
                M0_SU2_FFs = batch_compute_2d_fft(M0_comp_SU2, 'M0_SU2')
                M1_SU2_FFs = batch_compute_2d_fft(M1_comp_SU2, 'M1_SU2')
                M01_SU2_FFs = batch_compute_2d_fft(M01_comp_SU2, 'M01_SU2')
                save_intermediate_component_debug(
                    'SU2',
                    component_labels_SU2,
                    n_rows,
                    {
                        'M0': M0_SU2_FFs,
                        'M1': M1_SU2_FFs,
                        'M01': M01_SU2_FFs,
                    },
                    M_NL_SU2_FFs,
                )
            
            if not skip_plots:
                # Time-domain window masks for the side panel
                _td_tau_mask = tau <= 0
                _td_t_mask   = (times >= -40.0) & (times <= 40.0)
                _td_tau_ax   = tau[_td_tau_mask]
                _td_t_ax     = times[_td_t_mask]
                _td_extent   = [
                    _td_t_ax[0]   if len(_td_t_ax)   else -40.0,
                    _td_t_ax[-1]  if len(_td_t_ax)   else  40.0,
                    _td_tau_ax[0] if len(_td_tau_ax) else -40.0,
                    _td_tau_ax[-1] if len(_td_tau_ax) else  0.0,
                ]

                fig_debug, axes_debug = plt.subplots(n_rows, 5, figsize=(20, 3.5*n_rows), squeeze=False)

                def _td_imshow(ax, data2d, title, extent):
                    vmax = np.abs(data2d).max() or 1.0
                    im = ax.imshow(data2d, origin='lower', aspect='auto', cmap='RdBu_r',
                                   vmin=-vmax, vmax=vmax, extent=extent)
                    plt.colorbar(im, ax=ax, pad=0.02)
                    ax.set_xlabel('$t$ (ps)')
                    ax.set_ylabel('$\\tau$ (ps)')
                    ax.set_title(title)
                    ax.axhline(0, color='k', lw=0.5, ls=':')
                    ax.axvline(0, color='k', lw=0.5, ls=':')
            
                for comp in range(n_rows):
                    label = component_labels_SU2[comp] if comp < len(component_labels_SU2) else f'comp{comp}'
                    M_NL_comp_FF = M_NL_SU2_FFs[comp]
                    ax_freq = axes_debug[comp, 0]
                    im_freq = ax_freq.imshow(M_NL_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                              norm=_get_norm(M_NL_comp_FF, norm_type), extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]])
                    ax_freq.set_xlabel('$\\omega_t$ (THz)')
                    ax_freq.set_ylabel('$\\omega_{\\tau}$ (THz)')
                    ax_freq.set_title(f'{label} (freq domain)')
                    if omega_t_window is not None:
                        ax_freq.set_xlim(omega_t_window[0], omega_t_window[1])
                    if omega_tau_window is not None:
                        ax_freq.set_ylim(omega_tau_window)
                    if energy_levels_mev:
                        add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                    plt.colorbar(im_freq, ax=ax_freq)

                    if _td_tau_mask.any() and _td_t_mask.any():
                        # Col 1: M_NL
                        _td_imshow(axes_debug[comp, 1],
                                   M_NL_SU2[comp][_td_tau_mask][:, _td_t_mask],
                                   f'{label} $M_{{NL}}$', _td_extent)
                        # Col 2: M01
                        _td_imshow(axes_debug[comp, 2],
                                   M01_td_SU2[comp][_td_tau_mask][:, _td_t_mask],
                                   f'{label} $M_{{01}}$', _td_extent)
                        # Col 3: M0 (tiled over τ)
                        m0_row = M0_SU2_T[comp, :length][_td_t_mask]
                        m0_td  = np.tile(m0_row, (_td_tau_mask.sum(), 1))
                        _td_imshow(axes_debug[comp, 3], m0_td,
                                   f'{label} $M_{{0}}$', _td_extent)
                        # Col 4: M1
                        _td_imshow(axes_debug[comp, 4],
                                   M1_td_SU2[comp][_td_tau_mask][:, _td_t_mask],
                                   f'{label} $M_{{1}}$', _td_extent)
            
                plt.tight_layout()
                plt.savefig(os.path.join(dir, "M_NL_SU2_components_debug.pdf"), dpi=100)
                plt.close(fig_debug)
            
            # Main M_NL spectrum for SU2 (z-component) - use cached FFT
            z_idx = 2 if spin_dim_SU2 > 2 else 0
            M_NL_z_FF = M_NL_SU2_FFs[z_idx]
            _save_spectrum(os.path.join(dir, "M_NL_SU2_FF"), M_NL_z_FF)
            results['M_NL_SU2_FF'] = M_NL_z_FF
        
        # =====================================================================
        # Process SU(3) sublattice
        # =====================================================================
        if '/reference/M_global_SU3' in f:
            print("\n  Processing SU(3) sublattice...")
            M0_SU3 = f['/reference/M_global_SU3'][:]
            spin_dim_SU3 = M0_SU3.shape[1]
            length = len(M0_SU3[:, 0])

            # Pre-allocate (spin_dim, n_tau, n_t) arrays.  M0/M1/M01 component
            # buffers are only used in save_intermediates branches below.
            M_NL_SU3   = np.zeros((spin_dim_SU3, tau_step, length))
            M01_td_SU3 = np.zeros((spin_dim_SU3, tau_step, length))  # for time-domain plot
            M1_td_SU3  = np.zeros((spin_dim_SU3, tau_step, length))  # for time-domain plot
            if save_intermediates:
                M0_comp_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
                M1_comp_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
                M01_comp_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
            else:
                M0_comp_SU3 = None
                M1_comp_SU3 = None
                M01_comp_SU3 = None

            # Vectorise the per-tau loop (drops inner ``for comp in range(8)``).
            M0_SU3_T = M0_SU3.T  # (spin_dim, n_t_M0)
            n_t_M0 = M0_SU3_T.shape[1]
            for i in range(tau_step):
                tau_group = f[f'/tau_scan/tau_{i}']
                M1_T = tau_group['M1_global_SU3'][:].T   # (spin_dim, n_t)
                M01_T = tau_group['M01_global_SU3'][:].T
                n = min(n_t_M0, M1_T.shape[1], M01_T.shape[1], length)
                M_NL_SU3[:, i, :n]   = M01_T[:, :n] - M0_SU3_T[:, :n] - M1_T[:, :n]
                M01_td_SU3[:, i, :n] = M01_T[:, :n]
                M1_td_SU3[:, i, :n]  = M1_T[:, :n]
                if save_intermediates:
                    M0_comp_SU3[:, i, :n] = M0_SU3_T[:, :n]
                    M1_comp_SU3[:, i, :n] = M1_T[:, :n]
                    M01_comp_SU3[:, i, :n] = M01_T[:, :n]
            
            if False:  # windowed_time_evolution_SU3_debug removed
                if not skip_plots:
                    print("    Creating windowed time evolution debug plot (SU3)...")
                    # Use λ2 component (typically most relevant) or first if not available
                    comp_idx = 1 if spin_dim_SU3 > 1 else 0
                    M_NL_example = M_NL_SU3[comp_idx]
                    comp_label = component_labels_SU3[comp_idx] if comp_idx < len(component_labels_SU3) else f'comp{comp_idx}'
                
                    fig_wind, axes_wind = plt.subplots(2, 3, figsize=(15, 8))
                
                    # Select a few representative tau values for plotting
                    n_tau_show = min(4, len(tau))
                    tau_show_indices = np.linspace(0, len(tau)-1, n_tau_show, dtype=int)
                    colors = plt.cm.viridis(np.linspace(0, 0.9, n_tau_show))
                
                    # Filter to t >= t_cutoff (same as FFT uses)
                    M_NL_filtered = M_NL_example[:, t0_idx:]
                    M_NL_dynamic = M_NL_filtered - M_NL_filtered.mean()
                    M_NL_windowed = M_NL_dynamic * apod_window
                
                    # Top-left: Original signal (unwindowed) vs t
                    ax0 = axes_wind[0, 0]
                    for i, idx in enumerate(tau_show_indices):
                        ax0.plot(times_positive, M_NL_dynamic[idx, :], color=colors[i], 
                                label=f'τ={tau[idx]:.1f}', alpha=0.8)
                    ax0.set_xlabel('t (ps)')
                    ax0.set_ylabel(f'$M_{{NL,{comp_label}}}$')
                    ax0.set_title('Original signal (mean subtracted)')
                    ax0.legend(fontsize=7)
                    ax0.grid(True, alpha=0.3)
                
                    # Top-middle: Windowed signal vs t
                    ax1 = axes_wind[0, 1]
                    for i, idx in enumerate(tau_show_indices):
                        ax1.plot(times_positive, M_NL_windowed[idx, :], color=colors[i],
                                label=f'τ={tau[idx]:.1f}', alpha=0.8)
                    ax1.set_xlabel('t (ps)')
                    ax1.set_ylabel(f'$M_{{NL,{comp_label}}}$ × window')
                    ax1.set_title(f'Windowed signal ({window_type})')
                    ax1.legend(fontsize=7)
                    ax1.grid(True, alpha=0.3)
                
                    # Top-right: Overlay comparison at mid-tau
                    ax2 = axes_wind[0, 2]
                    mid_tau = len(tau) // 2
                    ax2.plot(times_positive, M_NL_dynamic[mid_tau, :], 'b-', lw=1.5, 
                            label='Original', alpha=0.8)
                    ax2.plot(times_positive, M_NL_windowed[mid_tau, :], 'r--', lw=1.5,
                            label='Windowed', alpha=0.8)
                    ax2.set_xlabel('t (ps)')
                    ax2.set_ylabel(f'$M_{{NL,{comp_label}}}$')
                    ax2.set_title(f'Comparison @ τ={tau[mid_tau]:.1f}')
                    ax2.legend(fontsize=8)
                    ax2.grid(True, alpha=0.3)
                
                    # Bottom-left: 2D heatmap of original signal (transposed: τ on x, t on y)
                    ax3 = axes_wind[1, 0]
                    im3 = ax3.imshow(M_NL_dynamic.T, origin='lower', aspect='auto', cmap='RdBu_r',
                                    extent=[tau[0], tau[-1], times_positive[0], times_positive[-1]])
                    ax3.set_xlabel('τ (ps)')
                    ax3.set_ylabel('t (ps)')
                    ax3.set_title('Original (t, τ) domain')
                    plt.colorbar(im3, ax=ax3)
                
                    # Bottom-middle: 2D heatmap of windowed signal (transposed: τ on x, t on y)
                    ax4 = axes_wind[1, 1]
                    im4 = ax4.imshow(M_NL_windowed.T, origin='lower', aspect='auto', cmap='RdBu_r',
                                    extent=[tau[0], tau[-1], times_positive[0], times_positive[-1]])
                    ax4.set_xlabel('τ (ps)')
                    ax4.set_ylabel('t (ps)')
                    ax4.set_title(f'Windowed (t, τ) domain ({window_type})')
                    plt.colorbar(im4, ax=ax4)
                
                    # Bottom-right: Signal at fixed t across tau (shows τ-axis tapering)
                    ax5 = axes_wind[1, 2]
                    n_t_show = min(4, len(times_positive))
                    t_show_indices = np.linspace(0, len(times_positive)-1, n_t_show, dtype=int)
                    colors_t = plt.cm.plasma(np.linspace(0, 0.9, n_t_show))
                    for i, idx in enumerate(t_show_indices):
                        ax5.plot(tau, M_NL_dynamic[:, idx], color=colors_t[i], ls='-',
                                label=f't={times_positive[idx]:.1f} orig', alpha=0.5)
                        ax5.plot(tau, M_NL_windowed[:, idx], color=colors_t[i], ls='--',
                                label=f't={times_positive[idx]:.1f} wind', alpha=0.9)
                    ax5.set_xlabel('τ (ps)')
                    ax5.set_ylabel(f'$M_{{NL,{comp_label}}}$')
                    ax5.set_title('Signal vs τ (solid=orig, dashed=windowed)')
                    ax5.legend(fontsize=6, ncol=2)
                    ax5.grid(True, alpha=0.3)
                
                    plt.tight_layout()
                    plt.close(fig_wind)
            
            # M_NL analysis for SU3
            print("    Processing M_NL_SU3...")
            n_rows = min(spin_dim_SU3, 8)
            # Batched fft2 over the 8 SU(3) components.
            M_NL_SU3_FFs = batch_compute_2d_fft(M_NL_SU3, 'M_NL_SU3')
            
            # Save individual component FFTs
            for comp in range(min(spin_dim_SU3, len(component_labels_SU3))):
                label = component_labels_SU3[comp]
                _save_spectrum(os.path.join(dir, f"M_NL_SU3_FF_{label}"), M_NL_SU3_FFs[comp])

            if save_intermediates:
                print("    Processing M0/M1/M01_SU3...")
                M0_SU3_FFs = batch_compute_2d_fft(M0_comp_SU3, 'M0_SU3')
                M1_SU3_FFs = batch_compute_2d_fft(M1_comp_SU3, 'M1_SU3')
                M01_SU3_FFs = batch_compute_2d_fft(M01_comp_SU3, 'M01_SU3')
                save_intermediate_component_debug(
                    'SU3',
                    component_labels_SU3,
                    n_rows,
                    {
                        'M0': M0_SU3_FFs,
                        'M1': M1_SU3_FFs,
                        'M01': M01_SU3_FFs,
                    },
                    M_NL_SU3_FFs,
                )
            
            if not skip_plots:
                # Time-domain window masks (cheap to recompute; safe for SU3-only runs)
                _td3_tau_mask = tau <= 0
                _td3_t_mask   = (times >= -40.0) & (times <= 40.0)
                _td3_tau_ax   = tau[_td3_tau_mask]
                _td3_t_ax     = times[_td3_t_mask]
                _td3_extent   = [
                    _td3_t_ax[0]   if len(_td3_t_ax)   else -40.0,
                    _td3_t_ax[-1]  if len(_td3_t_ax)   else  40.0,
                    _td3_tau_ax[0] if len(_td3_tau_ax) else -40.0,
                    _td3_tau_ax[-1] if len(_td3_tau_ax) else  0.0,
                ]

                fig_debug, axes_debug = plt.subplots(n_rows, 5, figsize=(20, 3*n_rows), squeeze=False)

                def _td3_imshow(ax, data2d, title, extent):
                    vmax = np.abs(data2d).max() or 1.0
                    im = ax.imshow(data2d, origin='lower', aspect='auto', cmap='RdBu_r',
                                   vmin=-vmax, vmax=vmax, extent=extent)
                    plt.colorbar(im, ax=ax, pad=0.02)
                    ax.set_xlabel('$t$ (ps)')
                    ax.set_ylabel('$\\tau$ (ps)')
                    ax.set_title(title)
                    ax.axhline(0, color='k', lw=0.5, ls=':')
                    ax.axvline(0, color='k', lw=0.5, ls=':')
            
                for comp in range(n_rows):
                    label = component_labels_SU3[comp] if comp < len(component_labels_SU3) else f'comp{comp}'
                    M_NL_comp_FF = M_NL_SU3_FFs[comp]
                    ax_freq = axes_debug[comp, 0]
                    im_freq = ax_freq.imshow(M_NL_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                              norm=_get_norm(M_NL_comp_FF, norm_type), extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]])
                    ax_freq.set_xlabel('$\\omega_t$ (THz)')
                    ax_freq.set_ylabel('$\\omega_{\\tau}$ (THz)')
                    ax_freq.set_title(f'{label} (freq domain)')
                    if omega_t_window is not None:
                        ax_freq.set_xlim(omega_t_window[0], omega_t_window[1])
                    if omega_tau_window is not None:
                        ax_freq.set_ylim(omega_tau_window)
                    if energy_levels_mev:
                        add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                    plt.colorbar(im_freq, ax=ax_freq)

                    if _td3_tau_mask.any() and _td3_t_mask.any():
                        # Col 1: M_NL
                        _td3_imshow(axes_debug[comp, 1],
                                    M_NL_SU3[comp][_td3_tau_mask][:, _td3_t_mask],
                                    f'{label} $M_{{NL}}$', _td3_extent)
                        # Col 2: M01
                        _td3_imshow(axes_debug[comp, 2],
                                    M01_td_SU3[comp][_td3_tau_mask][:, _td3_t_mask],
                                    f'{label} $M_{{01}}$', _td3_extent)
                        # Col 3: M0 (tiled over τ)
                        m0_row = M0_SU3_T[comp, :length][_td3_t_mask]
                        m0_td  = np.tile(m0_row, (_td3_tau_mask.sum(), 1))
                        _td3_imshow(axes_debug[comp, 3], m0_td,
                                    f'{label} $M_{{0}}$', _td3_extent)
                        # Col 4: M1
                        _td3_imshow(axes_debug[comp, 4],
                                    M1_td_SU3[comp][_td3_tau_mask][:, _td3_t_mask],
                                    f'{label} $M_{{1}}$', _td3_extent)
            
                plt.tight_layout()
                plt.savefig(os.path.join(dir, "M_NL_SU3_components_debug.pdf"), dpi=100)
                plt.close(fig_debug)
            
            # Total M_NL spectrum for SU3 (sum over cached FFTs)
            M_NL_total_FF = sum(M_NL_SU3_FFs)
            _save_spectrum(os.path.join(dir, "M_NL_SU3_FF"), M_NL_total_FF)
            results['M_NL_SU3_FF'] = M_NL_total_FF
    
    # Clear FFT cache to free memory
    clear_fft_cache()
    
    print(f"\n  2D nonlinear spectroscopy analysis complete.")
    print(f"  Output files saved to: {dir}")
    
    return results


def plot_time_evolution(dir: str,
                        t_window: Tuple[float, float] = (-40.0, 40.0),
                        tau_window: Tuple[float, float] = (-40.0, 0.0),
                        energy_levels_mev: Optional[Dict[str, float]] = None,
                        title_suffix: str = '') -> None:
    """Plot magnetisation time evolution from a pump-probe HDF5 file.

    Produces ``time_evolution.pdf`` in *dir* with four rows:

      Row 0: Fe  Sx global  (SU2 comp 0)
      Row 1: Fe  Sz global  (SU2 comp 2)
      Row 2: Tm  λ₂ global  (SU3 comp 1)
      Row 3: Tm  λ₅ global  (SU3 comp 4)

    Each row shows two panels:
      • Left  panel: M0(t) reference trajectory (no pump).
      • Right panel: M_NL(t, τ) as a 2D image (x = t, y = τ).
                     Only the τ ≤ 0 half is shown (``tau_window`` default).

    Args:
        dir:               Directory containing ``pump_probe_spectroscopy.h5``.
        t_window:          (t_min, t_max) in ps.  Default (-40, 40).
        tau_window:        (tau_min, tau_max) in ps for M_NL image.  Default (-40, 0).
        energy_levels_mev: Optional dict of reference energies (meV) drawn as
                           vertical dashed lines on the M0 panel.
        title_suffix:      Appended to the figure suptitle (e.g. "H // a").
    """
    hdf5_path = os.path.join(dir, 'pump_probe_spectroscopy.h5')
    if not os.path.isfile(hdf5_path):
        print(f'  [plot_time_evolution] HDF5 not found: {hdf5_path}')
        return

    # Channels: (label, sublattice, component_index)
    CHANNELS = [
        ('Fe $S_x$ (global)',     'SU2', 0),
        ('Fe $S_z$ (global)',     'SU2', 2),
        ('Tm $\\lambda_2$ (gl.)', 'SU3', 1),
        ('Tm $\\lambda_5$ (gl.)', 'SU3', 4),
    ]

    t_min, t_max   = t_window
    tau_min, tau_max = tau_window

    with h5py.File(hdf5_path, 'r') as f:
        times    = f['/reference/times'][:]
        tau_vals = f['/tau_scan/tau_values'][:]

        t_mask   = (times >= t_min)    & (times <= t_max)
        tau_mask = (tau_vals >= tau_min) & (tau_vals <= tau_max)
        t_plot   = times[t_mask]
        tau_plot = tau_vals[tau_mask]

        # Load reference trajectories
        ref = {}
        for su in ('SU2', 'SU3'):
            key = f'/reference/M_global_{su}'
            if key in f:
                ref[su] = f[key][:]   # (n_t_full, spin_dim)

        # Build M_NL 2D images: dict[(su, ci)] → (n_tau_sel, n_t_sel)
        n_tau_sel = int(tau_mask.sum())
        n_t_sel   = int(t_mask.sum())
        mnl_2d = {(su, ci): np.zeros((n_tau_sel, n_t_sel))
                  for _, su, ci in CHANNELS}

        n_full = len(times)
        for k, ti in enumerate(np.where(tau_mask)[0]):
            grp = f[f'/tau_scan/tau_{ti}']
            for su in ('SU2', 'SU3'):
                k_m1  = f'M1_global_{su}'
                k_m01 = f'M01_global_{su}'
                if k_m1 not in grp or k_m01 not in grp:
                    continue
                m1  = grp[k_m1][:]    # (n_t_full, spin_dim)
                m01 = grp[k_m01][:]
                m0  = ref.get(su)
                if m0 is None:
                    continue
                n = min(n_full, m1.shape[0], m01.shape[0], m0.shape[0])
                t_mask_n = t_mask[:n]
                for _, csu, ci in CHANNELS:
                    if csu != su or ci >= m1.shape[1]:
                        continue
                    mnl_row = (m01[:n, ci] - m0[:n, ci] - m1[:n, ci])[t_mask_n]
                    mnl_2d[(su, ci)][k, :len(mnl_row)] = mnl_row

    # ── figure layout ──────────────────────────────────────────────────────
    N_ROWS = len(CHANNELS)
    fig, axes = plt.subplots(N_ROWS, 2, figsize=(11, 2.8 * N_ROWS), squeeze=False)

    # Reference energy period lines on M0 panel
    def _draw_ref_lines(ax):
        if not energy_levels_mev or not len(t_plot):
            return
        colors_ref = {'e1': '#88ccff', 'e2': '#aaffaa', 'e2_e1': '#ff88ff'}
        for name, val in energy_levels_mev.items():
            if val is None or val <= 0:
                continue
            period = 4.135667696 / val   # ps (h/E; E in meV)
            c = colors_ref.get(name, 'yellow')
            for t0 in np.arange(t_plot[0], t_plot[-1], period):
                ax.axvline(t0, color=c, lw=0.5, ls='--', alpha=0.5)

    td_extent = [
        t_plot[0]   if len(t_plot)   else t_min,
        t_plot[-1]  if len(t_plot)   else t_max,
        tau_plot[0] if len(tau_plot) else tau_min,
        tau_plot[-1] if len(tau_plot) else tau_max,
    ]

    for row, (ch_label, su, ci) in enumerate(CHANNELS):
        ax_ref = axes[row, 0]
        ax_td  = axes[row, 1]

        # Left: M0 reference trace
        m0 = ref.get(su)
        if m0 is not None and ci < m0.shape[1]:
            ax_ref.plot(t_plot, m0[t_mask, ci], color='steelblue', lw=0.8)
        ax_ref.set_ylabel(ch_label, fontsize=7)
        ax_ref.yaxis.set_label_coords(-0.22, 0.5)
        if row == 0:
            ax_ref.set_title('$M_0(t)$ — reference (no pump)', fontsize=8)
        _draw_ref_lines(ax_ref)
        ax_ref.axhline(0, color='grey', lw=0.4, ls=':')
        ax_ref.set_xlim(t_min, t_max)
        ax_ref.tick_params(labelsize=6)
        if row < N_ROWS - 1:
            ax_ref.set_xticklabels([])
        else:
            ax_ref.set_xlabel('$t$ (ps)', fontsize=7)

        # Right: M_NL(τ, t) as 2D image
        img = mnl_2d.get((su, ci))
        if img is not None and img.size > 0:
            vmax = np.abs(img).max() or 1.0
            im = ax_td.imshow(
                img, origin='lower', aspect='auto', cmap='RdBu_r',
                vmin=-vmax, vmax=vmax, extent=td_extent,
            )
            plt.colorbar(im, ax=ax_td, pad=0.02)
        ax_td.set_xlim(t_min, t_max)
        ax_td.set_ylim(tau_min, tau_max)
        ax_td.axhline(0, color='k', lw=0.5, ls=':')
        ax_td.axvline(0, color='k', lw=0.5, ls=':')
        if row == 0:
            ax_td.set_title('$M_{NL}(t,\\tau)$ — time domain', fontsize=8)
        ax_td.tick_params(labelsize=6)
        ax_td.set_ylabel('$\\tau$ (ps)', fontsize=7)
        if row < N_ROWS - 1:
            ax_td.set_xticklabels([])
        else:
            ax_td.set_xlabel('$t$ (ps)', fontsize=7)

    suptitle = 'TmFeO3 time evolution  —  χ=0 baseline'
    if title_suffix:
        suptitle += f'  —  {title_suffix}'
    fig.suptitle(suptitle, fontsize=9, y=1.01)
    fig.tight_layout(rect=[0.08, 0, 1.0, 1.0])

    out = os.path.join(dir, 'time_evolution.pdf')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {out}')


def read_2DCS_combined_hdf5(filepath: str, omega_t_window: Optional[Tuple[float, float]] = None,
                            omega_tau_window: Optional[Tuple[float, float]] = None,
                            norm_type: NormType = 'power',
                            window_type: str = 'gaussian',
                            energy_levels_mev: Optional[Dict[str, float]] = None,
                            load_from_cache: bool = False,
                            reload_components: bool = False,
                            save_intermediates: bool = False,
                            skip_plots: bool = False,
                            auto_window_margin: float = 2.0) -> Dict[str, np.ndarray]:
    """
    Read 2D coherent spectroscopy data and compute nonlinear spectra for both sublattices.
    
    Wrapper that finds the appropriate directory and calls read_2D_nonlinear.
    
    Args:
        filepath: Path to pump_probe_spectroscopy.h5 file
        omega_t_window: Optional (min, max) tuple for ω_t axis limits  
        omega_tau_window: Optional (min, max) tuple for ω_τ axis limits
        norm_type: Normalization type for plots ('log', 'power', 'symlog', 'linear')
        window_type: Type of apodization window ('gaussian', 'hann', 'blackman', etc.)
        energy_levels_mev: Optional dict of energy levels in meV to mark with dashed lines.
            Keys: 'e1', 'e2', 'e2_e1', 'kc'. Values: energy in meV.
        load_from_cache: If True, skip FFT calculations and load from previously saved .txt files
        reload_components: If True, load component FFTs and recompute composites
        save_intermediates: If True, save intermediate M0, M1, M01 spectra. Default: False
        
    Returns:
        Dictionary with 2DCS results for SU(2), SU(3), and combined
    """
    output_dir = os.path.dirname(filepath)
    if output_dir == '':
        output_dir = '.'
    
    return read_2D_nonlinear(output_dir, omega_t_window, omega_tau_window, norm_type, 
                             window_type=window_type, energy_levels_mev=energy_levels_mev,
                             load_from_cache=load_from_cache, reload_components=reload_components,
                             save_intermediates=save_intermediates, skip_plots=skip_plots,
                             auto_window_margin=auto_window_margin)


def run_interactive_2dcs(filepath: str, 
                         omega_t_window: Optional[Tuple[float, float]] = None,
                         omega_tau_window: Optional[Tuple[float, float]] = None,
                         norm_type: NormType = 'power',
                         window_type: str = 'gaussian',
                         energy_levels_mev: Optional[Dict[str, float]] = None):
    """
    Run an interactive 2DCS analysis with sliders to tune component weights and broadening.
    
    Allows real-time tuning of:
    - λ2 (SU3) weight
    - λ5 (SU3) weight  
    - λ7 (SU3) weight
    - x-component (SU2) weight
    - Gaussian broadening (sigma scale factor)
    
    Weights can be negative for subtraction.
    
    Args:
        filepath: Path to pump_probe_spectroscopy.h5 file
        omega_t_window: Optional (min, max) tuple for ω_t axis limits (in meV, converted to THz)
        omega_tau_window: Optional (min, max) tuple for ω_τ axis limits (in meV, converted to THz)
        norm_type: Normalization type for plots
        window_type: Apodization window type
        energy_levels_mev: Optional dict of energy levels in meV
    """
    from matplotlib.widgets import Slider, Button
    
    # Switch to interactive backend
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
    output_dir = os.path.dirname(filepath)
    if output_dir == '':
        output_dir = '.'
    
    # Validate windows
    omega_t_window = _validate_window(omega_t_window, "omega_t_window")
    omega_tau_window = _validate_window(omega_tau_window, "omega_tau_window")
    
    # Convert windows from meV to THz if provided
    if omega_t_window is not None:
        omega_t_window = (omega_t_window[0] * MEV_TO_THZ, omega_t_window[1] * MEV_TO_THZ)
    if omega_tau_window is not None:
        omega_tau_window = (omega_tau_window[0] * MEV_TO_THZ, omega_tau_window[1] * MEV_TO_THZ)
    
    hdf5_path = os.path.join(output_dir, "pump_probe_spectroscopy.h5")
    
    print("Loading data for interactive mode...")
    
    # Store data in a container for closure access
    data = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get metadata (handle missing pulse_width gracefully)
        pulse_window_sigma = 5.0
        if '/metadata/pulse_width' in f:
            pulse_width = f['/metadata/pulse_width'][()]
            t_cutoff = pulse_window_sigma * pulse_width
        else:
            t_cutoff = 0.0  # No pulse cutoff if pulse_width not available
        
        # Get tau values
        tau_keys = [k for k in f['/tau_scan'].keys() if k.startswith('tau_') and k != 'tau_values']
        tau_indices = sorted([int(k.split('_')[1]) for k in tau_keys])
        tau_values = f['/tau_scan/tau_values'][:]
        tau = tau_values
        tau_step = len(tau_indices)
        
        # Get reference time array
        times = f['/reference/times'][:]
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        
        # Find t0 index (where t > t_cutoff)
        t0_candidates = np.where(times > t_cutoff)[0]
        if len(t0_candidates) > 0:
            t0_idx = t0_candidates[0]
            times_positive = times[t0_idx:]
        else:
            t0_idx = 0
            times_positive = times
        
        # Store for later use
        data['tau'] = tau
        data['t0_idx'] = t0_idx
        data['times_positive'] = times_positive
        data['t_relative'] = times_positive - times_positive[0]
        data['t_range'] = data['t_relative'][-1] if len(data['t_relative']) > 1 else 1.0
        data['tau_range'] = max(np.abs(tau[0]), np.abs(tau[-1])) if len(tau) > 1 else 1.0
        
        # Compute omega arrays
        omega_tau = np.fft.fftfreq(int(len(tau)), tau[1] - tau[0] if len(tau) > 1 else 1.0) * 2 * np.pi
        omega_tau = np.fft.fftshift(omega_tau)
        omega_tau = omega_tau * MEV_TO_THZ
        
        omega_t = np.fft.fftfreq(len(times_positive), dt) * 2 * np.pi
        omega_t = np.fft.fftshift(omega_t)
        omega_t = omega_t * MEV_TO_THZ
        
        data['omega_tau'] = omega_tau
        data['omega_t'] = omega_t
        
        # Load SU(2) raw time-domain data
        data['M_NL_SU2'] = None
        if '/reference/M_global_SU2' in f:
            print("  Loading SU(2) data...")
            M0_SU2 = f['/reference/M_global_SU2'][:]
            spin_dim_SU2 = M0_SU2.shape[1]
            length = len(M0_SU2[:, 0])
            
            M_NL_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
            
            for i, tau_idx in enumerate(tau_indices):
                tau_key = f'tau_{tau_idx}'
                # Check for both key formats: M1_global_SU2 (new) or M1_SU2 (old)
                if f'/tau_scan/{tau_key}/M1_global_SU2' in f:
                    M1_SU2 = f[f'/tau_scan/{tau_key}/M1_global_SU2'][:]
                    M01_SU2 = f[f'/tau_scan/{tau_key}/M01_global_SU2'][:]
                else:
                    M1_SU2 = f[f'/tau_scan/{tau_key}/M1_SU2'][:]
                    M01_SU2 = f[f'/tau_scan/{tau_key}/M01_SU2'][:]
                
                for comp in range(spin_dim_SU2):
                    M0 = M0_SU2[:, comp]
                    M1 = M1_SU2[:, comp]
                    M01 = M01_SU2[:, comp]
                    min_len = min(len(M0), len(M1), len(M01), length)
                    M_NL_SU2[comp, i, :min_len] = M01[:min_len] - M0[:min_len] - M1[:min_len]
            
            data['M_NL_SU2'] = M_NL_SU2
            print(f"    Loaded {spin_dim_SU2} SU(2) components")
        
        # Load SU(3) raw time-domain data
        data['M_NL_SU3'] = None
        if '/reference/M_global_SU3' in f:
            print("  Loading SU(3) data...")
            M0_SU3 = f['/reference/M_global_SU3'][:]
            spin_dim_SU3 = M0_SU3.shape[1]
            length = len(M0_SU3[:, 0])
            
            M_NL_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
            
            for i, tau_idx in enumerate(tau_indices):
                tau_key = f'tau_{tau_idx}'
                # Check for both key formats: M1_global_SU3 (new) or M1_SU3 (old)
                if f'/tau_scan/{tau_key}/M1_global_SU3' in f:
                    M1_SU3 = f[f'/tau_scan/{tau_key}/M1_global_SU3'][:]
                    M01_SU3 = f[f'/tau_scan/{tau_key}/M01_global_SU3'][:]
                else:
                    M1_SU3 = f[f'/tau_scan/{tau_key}/M1_SU3'][:]
                    M01_SU3 = f[f'/tau_scan/{tau_key}/M01_SU3'][:]
                
                for comp in range(spin_dim_SU3):
                    M0 = M0_SU3[:, comp]
                    M1 = M1_SU3[:, comp]
                    M01 = M01_SU3[:, comp]
                    min_len = min(len(M0), len(M1), len(M01), length)
                    M_NL_SU3[comp, i, :min_len] = M01[:min_len] - M0[:min_len] - M1[:min_len]
            
            data['M_NL_SU3'] = M_NL_SU3
            print(f"    Loaded {spin_dim_SU3} SU(3) components")
    
    print("Data loaded. Starting interactive mode...")
    
    # Helper function to build apodization window with given broadening
    def build_apod_window(sigma_scale):
        """Build Gaussian apodization window with given sigma scale factor."""
        if sigma_scale <= 0:
            return None
        # sigma_scale = 1.0 gives standard broadening (gamma=0.03 at edges)
        # Larger sigma_scale = less broadening (sharper peaks)
        # Smaller sigma_scale = more broadening (broader peaks)
        base_decay_factor = np.sqrt(-2.0 * np.log(0.03))  # For gamma=0.03
        decay_factor = base_decay_factor * sigma_scale
        
        sigma_t = data['t_range'] / decay_factor
        sigma_tau = data['tau_range'] / decay_factor
        
        decay_t = np.exp(-0.5 * (data['t_relative'] / sigma_t) ** 2)
        decay_tau = np.exp(-0.5 * (np.abs(data['tau']) / sigma_tau) ** 2)
        
        return np.outer(decay_tau, decay_t)
    
    # Helper function for 2D FFT with given apodization
    def compute_2d_fft(raw_data, apod_window):
        """Compute 2D FFT with apodization.
        
        Returns spectrum with shape (n_tau, n_t) for proper plotting
        with omega_t on x-axis and omega_tau on y-axis.
        """
        data_filtered = raw_data[:, data['t0_idx']:]
        data_dynamic = data_filtered - data_filtered.mean()
        if apod_window is not None:
            data_dynamic = data_dynamic * apod_window
        data_FF = np.fft.fft2(data_dynamic)
        data_FF = np.fft.fftshift(data_FF)
        return np.abs(data_FF)
    
    # Compute spectrum with given weights and broadening
    def compute_spectrum(w_lambda2, w_lambda5, w_lambda7, w_su2_x, sigma_scale):
        apod_window = build_apod_window(sigma_scale)
        
        # Get shape from first available data
        if data['M_NL_SU3'] is not None:
            shape = compute_2d_fft(data['M_NL_SU3'][0], apod_window).shape
        elif data['M_NL_SU2'] is not None:
            shape = compute_2d_fft(data['M_NL_SU2'][0], apod_window).shape
        else:
            return None
        
        spectrum = np.zeros(shape)
        
        if data['M_NL_SU3'] is not None:
            if data['M_NL_SU3'].shape[0] > 1:  # λ2 is index 1
                spectrum += w_lambda2 * compute_2d_fft(data['M_NL_SU3'][1], apod_window)
            if data['M_NL_SU3'].shape[0] > 4:  # λ5 is index 4
                spectrum += w_lambda5 * compute_2d_fft(data['M_NL_SU3'][4], apod_window)
            if data['M_NL_SU3'].shape[0] > 6:  # λ7 is index 6
                spectrum += w_lambda7 * compute_2d_fft(data['M_NL_SU3'][6], apod_window)
        
        if data['M_NL_SU2'] is not None:
            if data['M_NL_SU2'].shape[0] > 0:  # x is index 0
                spectrum += w_su2_x * compute_2d_fft(data['M_NL_SU2'][0], apod_window)
        
        return spectrum
    
    # Calculate figure size based on axis ranges for rectangular plot
    omega_tau_range = (data['omega_tau'][0], data['omega_tau'][-1])
    omega_t_range = (data['omega_t'][0], data['omega_t'][-1])
    
    # Use window ranges if provided for aspect ratio calculation
    if omega_tau_window is not None:
        omega_tau_range = omega_tau_window
    if omega_t_window is not None:
        omega_t_range = omega_t_window
    
    fig_width, fig_height = calculate_spectrum_figsize(omega_tau_range, omega_t_range, base_width=10.0)
    # Add extra height for sliders
    fig_height_total = fig_height + 4.0
    
    # Create figure with sliders
    fig, ax = plt.subplots(figsize=(fig_width, fig_height_total))
    plt.subplots_adjust(left=0.12, bottom=0.30, right=0.92, top=0.92)
    
    # Initial values
    init_lambda2 = 0.0
    init_lambda5 = 1.0
    init_lambda7 = 1.0
    init_su2_x = 0.0
    init_broadening = 1.0  # 1.0 = standard broadening
    
    # Compute initial spectrum
    spectrum = compute_spectrum(init_lambda2, init_lambda5, init_lambda7, init_su2_x, init_broadening)
    
    # Plot initial spectrum: omega_t on x-axis, omega_tau on y-axis
    # Extent is [x_min, x_max, y_min, y_max] = [omega_t_min, omega_t_max, omega_tau_min, omega_tau_max]
    im = ax.imshow(spectrum, origin='lower', aspect='auto', cmap='gnuplot2',
                   norm=_get_norm(spectrum, norm_type),
                   extent=[data['omega_t'][0], data['omega_t'][-1], data['omega_tau'][0], data['omega_tau'][-1]])
    ax.set_xlabel('$\\omega_t$ (THz)', fontsize=12)
    ax.set_ylabel('$\\omega_{\\tau}$ (THz)', fontsize=12)
    ax.set_title('Interactive 2DCS Spectrum', fontsize=14)
    
    # Apply window limits: omega_t on x-axis, omega_tau on y-axis
    if omega_t_window is not None:
        ax.set_xlim(omega_t_window[0], omega_t_window[1])
    if omega_tau_window is not None:
        ax.set_ylim(omega_tau_window[0], omega_tau_window[1])
    
    # Add energy level lines (omega_t on x, omega_tau on y)
    energy_lines = []
    if energy_levels_mev:
        for level_name, energy_mev in energy_levels_mev.items():
            if energy_mev is not None and energy_mev != 0:
                omega_thz = energy_mev * MEV_TO_THZ
                color = ENERGY_LINE_COLORS.get(level_name, 'white')
                for omega_val in [omega_thz, -omega_thz]:
                    # axhline for omega_tau (y-axis), axvline for omega_t (x-axis)
                    energy_lines.append(ax.axhline(y=omega_val, color=color, **ENERGY_LINE_STYLE))
                    energy_lines.append(ax.axvline(x=omega_val, color=color, **ENERGY_LINE_STYLE))
    
    cbar = plt.colorbar(im, ax=ax, label='Intensity')
    
    # Create slider axes
    ax_lambda2 = plt.axes([0.15, 0.27, 0.7, 0.025])
    ax_lambda5 = plt.axes([0.15, 0.22, 0.7, 0.025])
    ax_lambda7 = plt.axes([0.15, 0.17, 0.7, 0.025])
    ax_su2_x = plt.axes([0.15, 0.12, 0.7, 0.025])
    ax_broadening = plt.axes([0.15, 0.07, 0.7, 0.025])
    
    # Create sliders
    slider_lambda2 = Slider(ax_lambda2, 'λ₂ weight', -2.0, 2.0, valinit=init_lambda2, valstep=0.1)
    slider_lambda5 = Slider(ax_lambda5, 'λ₅ weight', -2.0, 2.0, valinit=init_lambda5, valstep=0.1)
    slider_lambda7 = Slider(ax_lambda7, 'λ₇ weight', -2.0, 2.0, valinit=init_lambda7, valstep=0.1)
    slider_su2_x = Slider(ax_su2_x, 'SU(2) x weight', -2.0, 2.0, valinit=init_su2_x, valstep=0.1)
    slider_broadening = Slider(ax_broadening, 'Broadening σ', 0.1, 3.0, valinit=init_broadening, valstep=0.1)
    
    # Add reset button
    ax_reset = plt.axes([0.8, 0.01, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset')
    
    # Add save button
    ax_save = plt.axes([0.65, 0.01, 0.1, 0.04])
    button_save = Button(ax_save, 'Save')
    
    # Update function
    def update(val):
        w_lambda2 = slider_lambda2.val
        w_lambda5 = slider_lambda5.val
        w_lambda7 = slider_lambda7.val
        w_su2_x = slider_su2_x.val
        sigma_scale = slider_broadening.val
        
        spectrum = compute_spectrum(w_lambda2, w_lambda5, w_lambda7, w_su2_x, sigma_scale)
        
        im.set_data(spectrum)
        im.set_norm(_get_norm(spectrum, norm_type))
        
        # Update title with current values
        title = f'λ₂={w_lambda2:.1f}, λ₅={w_lambda5:.1f}, λ₇={w_lambda7:.1f}, x={w_su2_x:.1f}, σ={sigma_scale:.1f}'
        ax.set_title(title, fontsize=14)
        
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    slider_lambda2.on_changed(update)
    slider_lambda5.on_changed(update)
    slider_lambda7.on_changed(update)
    slider_su2_x.on_changed(update)
    slider_broadening.on_changed(update)
    
    # Reset function
    def reset(event):
        slider_lambda2.reset()
        slider_lambda5.reset()
        slider_lambda7.reset()
        slider_su2_x.reset()
        slider_broadening.reset()
    
    button_reset.on_clicked(reset)
    
    # Save function
    def save(event):
        w_lambda2 = slider_lambda2.val
        w_lambda5 = slider_lambda5.val
        w_lambda7 = slider_lambda7.val
        w_su2_x = slider_su2_x.val
        sigma_scale = slider_broadening.val
        
        filename = f"interactive_l2={w_lambda2:.1f}_l5={w_lambda5:.1f}_l7={w_lambda7:.1f}_x={w_su2_x:.1f}_s={sigma_scale:.1f}.pdf"
        filepath_out = os.path.join(output_dir, filename)
        
        # Create a clean figure for saving with proper rectangular aspect ratio
        save_fig_width, save_fig_height = calculate_spectrum_figsize(omega_tau_range, omega_t_range, base_width=10.0)
        fig_save, ax_save_plot = plt.subplots(figsize=(save_fig_width, save_fig_height))
        spectrum = compute_spectrum(w_lambda2, w_lambda5, w_lambda7, w_su2_x, sigma_scale)
        # omega_t on x-axis, omega_tau on y-axis
        im_save = ax_save_plot.imshow(spectrum, origin='lower', aspect='auto', cmap='gnuplot2',
                                       norm=_get_norm(spectrum, norm_type),
                                       extent=[data['omega_t'][0], data['omega_t'][-1], 
                                               data['omega_tau'][0], data['omega_tau'][-1]])
        ax_save_plot.set_xlabel('$\\omega_t$ (THz)', fontsize=12)
        ax_save_plot.set_ylabel('$\\omega_{\\tau}$ (THz)', fontsize=12)
        ax_save_plot.set_title(f'λ₂={w_lambda2:.1f}, λ₅={w_lambda5:.1f}, λ₇={w_lambda7:.1f}, x={w_su2_x:.1f}, σ={sigma_scale:.1f}', fontsize=14)
        
        if omega_t_window is not None:
            ax_save_plot.set_xlim(omega_t_window[0], omega_t_window[1])
        if omega_tau_window is not None:
            ax_save_plot.set_ylim(omega_tau_window[0], omega_tau_window[1])
        
        # Add energy level lines (omega_t on x, omega_tau on y)
        if energy_levels_mev:
            for level_name, energy_mev in energy_levels_mev.items():
                if energy_mev is not None and energy_mev != 0:
                    omega_thz = energy_mev * MEV_TO_THZ
                    color = ENERGY_LINE_COLORS.get(level_name, 'white')
                    for omega_val in [omega_thz, -omega_thz]:
                        ax_save_plot.axhline(y=omega_val, color=color, **ENERGY_LINE_STYLE)
                        ax_save_plot.axvline(x=omega_val, color=color, **ENERGY_LINE_STYLE)
        
        plt.colorbar(im_save, ax=ax_save_plot, label='Intensity')
        plt.savefig(filepath_out, dpi=150, bbox_inches='tight')
        plt.close(fig_save)
        print(f"Saved: {filepath_out}")
    
    button_save.on_clicked(save)
    
    print("\nInteractive controls:")
    print("  - Adjust sliders to change component weights (can be negative)")
    print("  - Broadening σ: larger = sharper peaks, smaller = broader peaks")
    print("  - Click 'Reset' to restore initial values")
    print("  - Click 'Save' to save current view as PDF")
    print("  - Close window to exit")
    
    plt.show()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def print_hdf5_structure(filepath: str):
    """Print the structure of an HDF5 file."""
    def print_attrs(name, obj):
        print(f"{name}:")
        if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
            for key, val in obj.attrs.items():
                print(f"  @{key}: {val}")
        if isinstance(obj, h5py.Dataset):
            print(f"  shape: {obj.shape}, dtype: {obj.dtype}")
    
    with h5py.File(filepath, 'r') as f:
        f.visititems(print_attrs)


def get_metadata_from_hdf5(filepath: str) -> Dict[str, Any]:
    """
    Extract metadata from an HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        Dictionary of metadata
    """
    metadata = {}
    
    with h5py.File(filepath, 'r') as f:
        # Check for different possible metadata locations
        for grp_name in ['metadata', 'metadata_global', 'metadata_SU2', 'metadata_SU3']:
            if grp_name in f:
                grp = f[grp_name]
                for key in grp.attrs.keys():
                    val = grp.attrs[key]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    metadata[f'{grp_name}/{key}'] = val
    
    return metadata


# =============================================================================
# MAIN
# =============================================================================

def find_hdf5_file(path: str, analysis_type: str) -> str:
    """
    Find the appropriate HDF5 file given a path.
    
    If path is a directory, search for the expected HDF5 file based on analysis type.
    If path is a file, return it directly.
    
    Args:
        path: Path to file or directory
        analysis_type: 'md' or '2dcs'
        
    Returns:
        Path to the HDF5 file
    """
    if os.path.isfile(path):
        return path
    
    if os.path.isdir(path):
        # Expected filenames based on analysis type
        if analysis_type == '2dcs':
            expected_files = ['pump_probe_spectroscopy.h5']
        else:  # md
            expected_files = ['trajectory.h5']
        
        # Search in the directory and subdirectories
        for root, dirs, files in os.walk(path):
            for expected in expected_files:
                if expected in files:
                    return os.path.join(root, expected)
        
        # If not found, list available .h5 files
        h5_files = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.h5'):
                    h5_files.append(os.path.join(root, f))
        
        if h5_files:
            print(f"Expected HDF5 file not found. Available .h5 files:")
            for f in h5_files:
                print(f"  {f}")
            # Return the first one found
            return h5_files[0]
        
        raise FileNotFoundError(f"No HDF5 files found in {path}")
    
    raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='HDF5 Reader for TmFeO3 Molecular Dynamics and 2D Coherent Spectroscopy Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reader_TmFeO3.py /path/to/data md
  python reader_TmFeO3.py /path/to/data 2dcs --norm power
  python reader_TmFeO3.py /path/to/data 2dcs --norm log --omega-t -10 10 --omega-tau -5 5
  python reader_TmFeO3.py /path/to/data 2dcs --norm power --window blackman
  python reader_TmFeO3.py /path/to/data 2dcs --e1 2.19 --e2 5.34 --kc 3.76  # Add energy level lines
  python reader_TmFeO3.py /path/to/data 2dcs -i --e1 2.19 --e2 5.34 --kc 3.76  # Interactive mode with sliders
        """
    )
    parser.add_argument('path', help='Path to HDF5 file or directory containing it')
    parser.add_argument('analysis_type', nargs='?', default='md', choices=['md', '2dcs'],
                        help="Analysis type: 'md' for molecular dynamics, '2dcs' for 2D spectroscopy (default: md)")
    parser.add_argument('--norm', '-n', type=str, default='power', choices=['log', 'power', 'symlog', 'linear'],
                        help="Normalization type for 2D plots (default: power)")
    parser.add_argument('--omega-t', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='ω_t axis limits for 2DCS plots (e.g., --omega-t -10 10)')
    parser.add_argument('--omega-tau', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='ω_τ axis limits for 2DCS plots (e.g., --omega-tau -5 5)')
    parser.add_argument('--window', '-w', type=str, default='gaussian',
                        choices=['gaussian', 'hann', 'hamming', 'blackman', 'tukey', 'cosine', 'exponential', 'none'],
                        help="Apodization window type (default: gaussian)")
    parser.add_argument('--w0', type=float, default=0,
                        help='Minimum frequency for MD analysis (default: 0)')
    parser.add_argument('--wmax', type=float, default=70,
                        help='Maximum frequency for MD analysis (default: 70)')
    
    # Energy level arguments for 2DCS reference lines (in meV)
    parser.add_argument('--e1', type=float, default=None,
                        help='Tm e1 energy level in meV (e.g., 2.19 meV = 0.53 THz)')
    parser.add_argument('--e2', type=float, default=None,
                        help='Tm e2 energy level in meV (e.g., 5.34 meV = 1.29 THz)')
    parser.add_argument('--kc', type=float, default=None,
                        help='Fe Kc anisotropy energy in meV (e.g., 3.76 meV = 0.91 THz AFM gap)')
    parser.add_argument('--qfm', type=float, default=None,
                        help='Fe qFM magnon frequency in meV (e.g., 1.59 meV = 0.384 THz)')
    
    # Interactive mode
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run interactive mode with sliders to tune λ2, λ5, λ7, and SU(2) x weights')
    
    # Load from cache option
    parser.add_argument('--load-results', '--from-cache', action='store_true',
                        help='Skip FFT calculations and load from previously saved .txt files (only regenerate plots)')
    
    # Reload components option
    parser.add_argument('--reload-components', '--from-components', action='store_true',
                        help='Load component FFTs from cache and recompute composite spectra (faster than full calculation)')

    # Performance flags (Ingredient XIX)
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip all matplotlib figure generation (save NPY spectra only).  '
                             'Roughly 100-300x faster than the default plotting path on large 2DCS jobs.')
    parser.add_argument('--save-intermediates', dest='save_intermediates', action='store_true', default=True,
                        help='Compute and save M0/M1/M01 component spectra and λ-mode mixtures (default: True)')
    parser.add_argument('--no-save-intermediates', dest='save_intermediates', action='store_false',
                        help='Skip M0/M1/M01 components — only compute the M_NL aggregates.  '
                             'Use this for fast scans when only the nonlinear response is needed.')
    parser.add_argument('--fast', action='store_true',
                        help='Convenience: equivalent to --skip-plots --no-save-intermediates.  '
                             'Bare-minimum aggregate spectra only, sub-second on the nocouple_ctrl benchmark.')

    args = parser.parse_args()
    if args.fast:
        args.skip_plots = True
        args.save_intermediates = False
    
    input_path = args.path
    analysis_type = args.analysis_type
    
    if not os.path.exists(input_path):
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
    
    try:
        filepath = find_hdf5_file(input_path, analysis_type)
        print(f"Using HDF5 file: {filepath}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Analyzing {filepath} (type: {analysis_type})")
    print(f"  Normalization: {args.norm}")
    print(f"  Window type: {args.window}")
    if args.omega_t:
        print(f"  ω_t window: {args.omega_t}")
    if args.omega_tau:
        print(f"  ω_τ window: {args.omega_tau}")
    
    # Build energy levels dictionary if any are specified
    energy_levels_mev = None
    if analysis_type == '2dcs' and (args.e1 or args.e2 or args.kc or args.qfm):
        energy_levels_mev = {}
        if args.e1 is not None:
            energy_levels_mev['e1'] = args.e1
        if args.e2 is not None:
            energy_levels_mev['e2'] = args.e2
            # Also add e2-e1 if both are provided
            if args.e1 is not None:
                energy_levels_mev['e2-e1'] = args.e2 - args.e1
        if args.kc is not None:
            energy_levels_mev['qAFM'] = args.kc
        if args.qfm is not None:
            energy_levels_mev['qFM'] = args.qfm
        print(f"  Energy levels: {energy_levels_mev}")
    
    if analysis_type == 'md':
        print(f"\nRunning MD analysis (w0={args.w0}, wmax={args.wmax})...")
        results = read_MD_hdf5(filepath, w0=args.w0, wmax=args.wmax)
        print(f"Results keys: {list(results.keys())}")
    elif analysis_type == '2dcs':
        omega_t_window = tuple(args.omega_t) if args.omega_t else None
        omega_tau_window = tuple(args.omega_tau) if args.omega_tau else None
        
        if args.interactive:
            print("\nStarting interactive 2DCS mode...")
            run_interactive_2dcs(filepath, omega_t_window, omega_tau_window, args.norm,
                                 window_type=args.window, energy_levels_mev=energy_levels_mev)
        else:
            if args.load_results:
                print("\nLoading 2DCS results from cache (skipping calculations)...")
            elif args.reload_components:
                print("\nReloading component FFTs and recomputing composites...")
            else:
                print("\nRunning 2DCS analysis...")
            results = read_2DCS_combined_hdf5(filepath, omega_t_window, omega_tau_window, args.norm,
                                              window_type=args.window, energy_levels_mev=energy_levels_mev,
                                              load_from_cache=args.load_results,
                                              reload_components=args.reload_components,
                                              skip_plots=args.skip_plots,
                                              save_intermediates=args.save_intermediates)
            print(f"Results keys: {list(results.keys())}")
    else:
        print(f"Unknown analysis type: {analysis_type}")
        sys.exit(1)
    
    print("\nAnalysis complete!")
