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
    'qAFM': 'lime'
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
    
    # Calculate aspect ratio (height/width)
    aspect = t_span / tau_span
    
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
            np.savetxt(os.path.join(output_dir, "DSSF_local.txt"), A_combined_local)
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
            np.savetxt(os.path.join(output_dir, "DSSF_global.txt"), A_combined_global)
    
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
    sample_files = ['M_NL_SU2_FF.txt', 'M_NL_SU3_FF.txt', 'M_NL_combined_FF.txt']
    sample_data = None
    for sample_file in sample_files:
        sample_path = os.path.join(dir, sample_file)
        if os.path.exists(sample_path):
            try:
                sample_data = np.loadtxt(sample_path)
                break
            except:
                continue
    
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
    result_files = [
        ('M_NL_SU2_FF', 'M_NL_SU2_SPEC', '$M_{NL}$ Spectrum (SU(2))'),
        ('M_NL_SU3_FF', 'M_NL_SU3_SPEC', '$M_{NL}$ Spectrum (SU(3) total)'),
        ('M_NL_lambda57_FF', 'M_NL_lambda57_SPEC', '$M_{NL}$ Spectrum ($\\lambda_5$ + $\\lambda_7$)'),
        ('M_NL_lambda25_FF', 'M_NL_lambda25_SPEC', '$M_{NL}$ Spectrum ($\\lambda_2$ + $\\lambda_5$)'),
        ('M_NL_lambda257_FF', 'M_NL_lambda257_SPEC', '$M_{NL}$ Spectrum ($\\lambda_2$ + $\\lambda_5$ + $\\lambda_7$)'),
        ('M_NL_lambda27x_FF', 'M_NL_lambda27x_SPEC', '$M_{NL}$ Spectrum ($\\lambda_2$ + $\\lambda_7$ + $S_x$)'),
        ('M_NL_lambda27x_v2_FF', 'M_NL_lambda27x_v2_SPEC', '$M_{NL}$ Spectrum ($\\lambda_2$ - 0.2$\\lambda_7$ + $S_x$)'),
        ('M_NL_combined_FF', 'M_NL_combined_SPEC', '$M_{NL}$ Spectrum (SU(2) + SU(3))'),
    ]
    
    for result_key, pdf_name, title in result_files:
        txt_file = os.path.join(dir, f"{result_key}.txt")
        if os.path.exists(txt_file):
            print(f"    Loading {result_key} from {txt_file}...")
            try:
                data = np.loadtxt(txt_file)
                results[result_key] = data
                
                # Generate plot
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
                im = ax.imshow(data, origin='lower',
                               extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                               aspect='auto', cmap='gnuplot2', norm=_get_norm(data, norm_type))
                ax.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax.set_ylabel('$\\omega_t$ (THz)')
                plt.colorbar(im, ax=ax, label='Intensity')
                ax.set_title(title)
                
                if omega_tau_window is not None:
                    ax.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax.set_ylim(omega_t_window)
                
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax, energy_levels_mev, omega_t_window)
                
                plt.savefig(os.path.join(dir, f"{pdf_name}.pdf"), dpi=100)
                plt.close(fig)
                print(f"      Generated {pdf_name}.pdf")
            except Exception as e:
                print(f"      Warning: Failed to load/plot {result_key}: {e}")
    
    print(f"\n  Loaded {len(results) - 2} result files (excluding omega grids)")
    return results


def _reload_and_compute_composites(dir: str, omega_t_window: Optional[Tuple[float, float]],
                                    omega_tau_window: Optional[Tuple[float, float]],
                                    norm_type: NormType,
                                    energy_levels_mev: Optional[Dict[str, float]]) -> Dict[str, np.ndarray]:
    """Load component FFTs from cache and recompute composite spectra.
    
    Loads individual component FFT results (λ2, λ5, λ7, x, y, z) and recalculates
    composite combinations (λ2+λ7+x, λ2-0.2λ7+x, etc.) with new plots.
    
    Args:
        dir: Directory containing saved component .txt files
        omega_t_window: ω_t axis limits
        omega_tau_window: ω_τ axis limits
        norm_type: Normalization type
        energy_levels_mev: Energy level dictionary
        
    Returns:
        Dictionary with recomputed composite results
    """
    results = {}
    
    # Load HDF5 to get omega grids (reconstruct from saved data dimensions)
    hdf5_path = os.path.join(dir, "pump_probe_spectroscopy.h5")
    with h5py.File(hdf5_path, 'r') as f:
        tau_values = f['/tau_scan/tau_values'][:]
        times = f['/reference/times'][:]
    
    # Helper function to try loading a component file
    def try_load_component(filename):
        filepath = os.path.join(dir, filename)
        if os.path.exists(filepath):
            try:
                return np.loadtxt(filepath)
            except:
                return None
        return None
    
    # Load SU(2) component FFTs (x, y, z)
    print("    Loading SU(2) component FFTs...")
    M_NL_SU2_x = try_load_component("M_NL_SU2_FF_x.txt")
    M_NL_SU2_y = try_load_component("M_NL_SU2_FF_y.txt")
    M_NL_SU2_z = try_load_component("M_NL_SU2_FF_z.txt")
    
    # Store them in array format for easy indexing
    M_NL_SU2_components = []
    for comp, name in [(M_NL_SU2_x, 'x'), (M_NL_SU2_y, 'y'), (M_NL_SU2_z, 'z')]:
        if comp is not None:
            M_NL_SU2_components.append(comp)
            print(f"      Loaded {name} component")
    
    # Load SU(3) component FFTs (λ1-λ8)
    print("    Loading SU(3) component FFTs...")
    M_NL_SU3_components = []
    component_labels = ['λ1', 'λ2', 'λ3', 'λ4', 'λ5', 'λ6', 'λ7', 'λ8']
    for i, label in enumerate(component_labels):
        comp = try_load_component(f"M_NL_SU3_FF_{label}.txt")
        M_NL_SU3_components.append(comp)
        if comp is not None:
            print(f"      Loaded {label} component")
    
    # Get dimensions from first available component
    sample_data = None
    if M_NL_SU2_components:
        sample_data = M_NL_SU2_components[0]
    elif any(c is not None for c in M_NL_SU3_components):
        sample_data = next(c for c in M_NL_SU3_components if c is not None)
    
    if sample_data is None:
        print("    Error: No component FFT files found. Please run full calculation first.")
        return results
    
    # Reconstruct frequency grids
    n_tau_fft, n_t_fft = sample_data.shape
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    dtau = tau_values[1] - tau_values[0] if len(tau_values) > 1 else 1.0
    
    omega_tau = np.fft.fftfreq(n_tau_fft, dtau) * 2 * np.pi * MEV_TO_THZ
    omega_tau = np.fft.fftshift(omega_tau)
    
    omega_t = np.fft.fftfreq(n_t_fft, dt) * 2 * np.pi * MEV_TO_THZ
    omega_t = np.fft.fftshift(omega_t)
    
    results['omega_tau'] = omega_tau
    results['omega_t'] = omega_t
    
    # Calculate figure size
    tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
    t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
    figsize = calculate_spectrum_figsize(tau_range, t_range)
    
    # Helper function to plot and save spectrum
    def plot_spectrum(data, result_key, pdf_name, title):
        results[result_key] = data
        np.savetxt(os.path.join(dir, f"{result_key}.txt"), data)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        im = ax.imshow(data, origin='lower',
                       extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                       aspect='auto', cmap='gnuplot2', norm=_get_norm(data, norm_type))
        ax.set_xlabel('$\\omega_{\\tau}$ (THz)')
        ax.set_ylabel('$\\omega_t$ (THz)')
        plt.colorbar(im, ax=ax, label='Intensity')
        ax.set_title(title)
        
        if omega_tau_window is not None:
            ax.set_xlim(omega_tau_window[0], omega_tau_window[1])
        if omega_t_window is not None:
            ax.set_ylim(omega_t_window)
        
        if energy_levels_mev:
            add_energy_level_lines(ax, energy_levels_mev, omega_t_window)
        
        plt.savefig(os.path.join(dir, f"{pdf_name}.pdf"), dpi=100)
        plt.close(fig)
        print(f"      Generated {pdf_name}.pdf")
    
    # Recompute composite spectra
    print("\\n    Computing composite spectra...")
    
    # λ5 + λ7 (indices 4 and 6)
    if len(M_NL_SU3_components) > 6 and M_NL_SU3_components[4] is not None and M_NL_SU3_components[6] is not None:
        print("      λ5 + λ7...")
        M_NL_lambda57_FF = 1.0 * M_NL_SU3_components[4] + 1.0 * M_NL_SU3_components[6]
        plot_spectrum(M_NL_lambda57_FF, 'M_NL_lambda57_FF', 'M_NL_lambda57_SPEC',
                     '$M_{NL}$ Spectrum ($\\lambda_5$ + $\\lambda_7$)')
    
    # λ2 + λ5 (indices 1 and 4)
    if len(M_NL_SU3_components) > 4 and M_NL_SU3_components[1] is not None and M_NL_SU3_components[4] is not None:
        print("      λ2 + λ5...")
        M_NL_lambda25_FF = 1.0 * M_NL_SU3_components[1] + 1.0 * M_NL_SU3_components[4]
        plot_spectrum(M_NL_lambda25_FF, 'M_NL_lambda25_FF', 'M_NL_lambda25_SPEC',
                     '$M_{NL}$ Spectrum ($\\lambda_2$ + $\\lambda_5$)')
    
    # λ2 + λ5 + λ7 (indices 1, 4, 6)
    if (len(M_NL_SU3_components) > 6 and M_NL_SU3_components[1] is not None and 
        M_NL_SU3_components[4] is not None and M_NL_SU3_components[6] is not None):
        print("      λ2 + λ5 + λ7...")
        M_NL_lambda257_FF = 1.0 * M_NL_SU3_components[1] + 1.0 * M_NL_SU3_components[4] + 1.0 * M_NL_SU3_components[6]
        plot_spectrum(M_NL_lambda257_FF, 'M_NL_lambda257_FF', 'M_NL_lambda257_SPEC',
                     '$M_{NL}$ Spectrum ($\\lambda_2$ + $\\lambda_5$ + $\\lambda_7$)')
    
    # λ2 + λ7 + x (indices: SU3[1], SU3[6], SU2[0])
    if (len(M_NL_SU3_components) > 6 and M_NL_SU3_components[1] is not None and 
        M_NL_SU3_components[6] is not None and len(M_NL_SU2_components) > 0):
        print("      λ2 + λ7 + x...")
        M_NL_lambda27x_FF = 1.0 * M_NL_SU3_components[1] + 1.0 * M_NL_SU3_components[6] + 1.0 * M_NL_SU2_components[0]
        plot_spectrum(M_NL_lambda27x_FF, 'M_NL_lambda27x_FF', 'M_NL_lambda27x_SPEC',
                     '$M_{NL}$ Spectrum ($\\lambda_2$ + $\\lambda_7$ + $S_x$)')
    
    # λ2 - 0.2×λ7 + x (indices: SU3[1], SU3[6], SU2[0])
    if (len(M_NL_SU3_components) > 6 and M_NL_SU3_components[1] is not None and 
        M_NL_SU3_components[6] is not None and len(M_NL_SU2_components) > 0):
        print("      λ2 - 0.2×λ7 + x...")
        M_NL_lambda27x_v2_FF = 1.0 * M_NL_SU3_components[1] - 0.2 * M_NL_SU3_components[6] + 1.0 * M_NL_SU2_components[0]
        plot_spectrum(M_NL_lambda27x_v2_FF, 'M_NL_lambda27x_v2_FF', 'M_NL_lambda27x_v2_SPEC',
                     '$M_{NL}$ Spectrum ($\\lambda_2$ - 0.2$\\lambda_7$ + $S_x$)')
    
    # SU(2) + SU(3) combined (if available)
    M_NL_SU2_FF = try_load_component("M_NL_SU2_FF.txt")
    M_NL_SU3_FF = try_load_component("M_NL_SU3_FF.txt")
    if M_NL_SU2_FF is not None and M_NL_SU3_FF is not None:
        print("      SU(2) + SU(3) combined...")
        M_NL_combined_FF = M_NL_SU2_FF + M_NL_SU3_FF
        plot_spectrum(M_NL_combined_FF, 'M_NL_combined_FF', 'M_NL_combined_SPEC',
                     '$M_{NL}$ Spectrum (SU(2) + SU(3))')
        results['M_NL_SU2_FF'] = M_NL_SU2_FF
        results['M_NL_SU3_FF'] = M_NL_SU3_FF
    
    print(f"\\n  Recomputed {len([k for k in results.keys() if k.endswith('_FF') and k not in ['omega_tau', 'omega_t']])} composite spectra")
    return results


def read_2D_nonlinear(dir: str, omega_t_window: Optional[Tuple[float, float]] = None,
                      omega_tau_window: Optional[Tuple[float, float]] = None,
                      norm_type: NormType = 'power',
                      apodization_gamma: float = 0.03,
                      pulse_window_sigma: float = 5.0,
                      window_type: str = 'gaussian',
                      energy_levels_mev: Optional[Dict[str, float]] = None,
                      load_from_cache: bool = False,
                      reload_components: bool = False) -> Dict[str, np.ndarray]:
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
        
    Returns:
        Dictionary with 2DCS results for both sublattices
    """
    omega_t_window = _validate_window(omega_t_window, "omega_t_window")
    omega_tau_window = _validate_window(omega_tau_window, "omega_tau_window")
    
    # Convert windows from meV to THz if provided
    if omega_tau_window is not None:
        omega_t_window = (omega_t_window[0] * MEV_TO_THZ, omega_t_window[1] * MEV_TO_THZ)
    if omega_t_window is not None:
        omega_tau_window = (omega_tau_window[0] * MEV_TO_THZ, omega_tau_window[1] * MEV_TO_THZ)
    
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
        if '/metadata' in f and 'pulse_width' in f['/metadata'].attrs:
            pulse_width = float(f['/metadata'].attrs['pulse_width'])
        
        # Compute time cutoff: skip the probe pulse region (probe is centered at t=0)
        # We wait pulse_window_sigma * pulse_width after probe center before FFT
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
                window_tau[:taper_len_tau] = taper[::-1]
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
        # DEBUG: Plot apodization window to verify tapering
        # =====================================================================
        if apod_window is not None:
            print("  Creating apodization window debug plot...")
            fig_apod, axes_apod = plt.subplots(2, 2, figsize=(12, 10))
            
            # Top-left: 2D apodization window (transposed: τ on x, t on y)
            ax_2d = axes_apod[0, 0]
            im_apod = ax_2d.imshow(apod_window.T, origin='lower', aspect='auto', cmap='viridis',
                                   extent=[tau[0], tau[-1], times_positive[0], times_positive[-1]])
            ax_2d.set_xlabel('τ (ps)')
            ax_2d.set_ylabel('t (ps)')
            ax_2d.set_title(f'2D Apodization Window ({window_type})')
            plt.colorbar(im_apod, ax=ax_2d, label='Weight')
            
            # Top-right: t-axis window slices at different tau values
            ax_t = axes_apod[0, 1]
            n_tau_slices = min(5, len(tau))
            tau_slice_indices = np.linspace(0, len(tau)-1, n_tau_slices, dtype=int)
            for idx in tau_slice_indices:
                ax_t.plot(times_positive, apod_window[idx, :], label=f'τ={tau[idx]:.1f}', alpha=0.8)
            ax_t.set_xlabel('t (ps)')
            ax_t.set_ylabel('Window weight')
            ax_t.set_title('Window along t-axis (at various τ)')
            ax_t.legend(fontsize=8)
            ax_t.grid(True, alpha=0.3)
            ax_t.set_ylim(-0.05, 1.05)
            
            # Bottom-left: tau-axis window slices at different t values
            ax_tau = axes_apod[1, 0]
            n_t_slices = min(5, len(times_positive))
            t_slice_indices = np.linspace(0, len(times_positive)-1, n_t_slices, dtype=int)
            for idx in t_slice_indices:
                ax_tau.plot(tau, apod_window[:, idx], label=f't={times_positive[idx]:.1f}', alpha=0.8)
            ax_tau.set_xlabel('τ (ps)')
            ax_tau.set_ylabel('Window weight')
            ax_tau.set_title('Window along τ-axis (at various t)')
            ax_tau.legend(fontsize=8)
            ax_tau.grid(True, alpha=0.3)
            ax_tau.set_ylim(-0.05, 1.05)
            
            # Bottom-right: Center slices showing tapering
            ax_center = axes_apod[1, 1]
            mid_tau_idx = len(tau) // 2
            mid_t_idx = len(times_positive) // 2
            ax_center.plot(times_positive, apod_window[mid_tau_idx, :], 'b-', lw=2, 
                          label=f't-slice @ τ={tau[mid_tau_idx]:.1f}')
            ax_center.plot(tau, apod_window[:, mid_t_idx], 'r--', lw=2,
                          label=f'τ-slice @ t={times_positive[mid_t_idx]:.1f}')
            ax_center.axhline(y=apodization_gamma, color='gray', linestyle=':', alpha=0.7,
                             label=f'γ={apodization_gamma} (boundary)')
            ax_center.set_xlabel('Time (ps)')
            ax_center.set_ylabel('Window weight')
            ax_center.set_title('Center slice comparison')
            ax_center.legend(fontsize=8)
            ax_center.grid(True, alpha=0.3)
            ax_center.set_ylim(-0.05, 1.05)
            
            plt.tight_layout()
            plt.savefig(os.path.join(dir, "apodization_window_debug.pdf"), dpi=100)
            plt.close(fig_apod)
            print(f"    Saved: apodization_window_debug.pdf")
        
        # Helper function for 2D FFT analysis (optimized with caching)
        _fft_cache = {}
        
        def compute_2d_fft(data, cache_key=None):
            """Compute 2D FFT with proper shifting, flipping, transposing, and apodization (only t >= 0 data).
            
            Uses scipy.fft for better performance and supports caching to avoid redundant computation.
            
            Returns transposed result so that:
            - x-axis (columns) corresponds to ω_τ
            - y-axis (rows) corresponds to ω_t
            
            Args:
                data: 2D array with shape (n_tau, n_t)
                cache_key: Optional key for caching. If provided and cached, returns cached result.
            """
            # Check cache first
            if cache_key is not None and cache_key in _fft_cache:
                return _fft_cache[cache_key]
            
            # Filter to t >= 0
            data_filtered = data[:, t0_idx:]
            data_dynamic = data_filtered - data_filtered.mean()
            # Apply apodization window to reduce spectral leakage
            if apod_window is not None:
                data_dynamic = data_dynamic * apod_window
            # Use scipy.fft for better performance (can use multi-threading)
            try:
                from scipy import fft as scipy_fft
                data_FF = scipy_fft.fft2(data_dynamic, workers=-1)  # Use all available cores
                data_FF = scipy_fft.fftshift(data_FF)
            except ImportError:
                data_FF = np.fft.fft2(data_dynamic)
                data_FF = np.fft.fftshift(data_FF)
            data_FF = np.abs(data_FF)
            data_FF = np.flip(data_FF, axis=1)  # Flip omega_t axis
            result = data_FF.T  # Transpose so omega_tau is x-axis, omega_t is y-axis
            
            # Cache result
            if cache_key is not None:
                _fft_cache[cache_key] = result
            
            return result
        
        def clear_fft_cache():
            """Clear the FFT cache to free memory."""
            _fft_cache.clear()
        
        # =====================================================================
        # Process SU(2) sublattice
        # =====================================================================
        if '/reference/M_global_SU2' in f:
            print("\n  Processing SU(2) sublattice...")
            M0_SU2 = f['/reference/M_global_SU2'][:]
            spin_dim_SU2 = M0_SU2.shape[1]
            length = len(M0_SU2[:, 0])
            
            # Initialize arrays for all components
            M_NL_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
            M0_comp_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
            M1_comp_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
            M01_comp_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
            
            for i, tau_val in enumerate(tau_values):
                tau_group = f[f'/tau_scan/tau_{i}']
                M1_SU2 = tau_group['M1_global_SU2'][:]
                M01_SU2 = tau_group['M01_global_SU2'][:]
                
                for comp in range(spin_dim_SU2):
                    M0 = M0_SU2[:, comp]
                    M1 = M1_SU2[:, comp]
                    M01 = M01_SU2[:, comp]
                    
                    min_len = min(len(M0), len(M1), len(M01), length)
                    M_NL_SU2[comp, i, :min_len] = M01[:min_len] - M0[:min_len] - M1[:min_len]
                    M0_comp_SU2[comp, i, :min_len] = M0[:min_len]
                    M1_comp_SU2[comp, i, :min_len] = M1[:min_len]
                    M01_comp_SU2[comp, i, :min_len] = M01[:min_len]
            
            # =====================================================================
            # DEBUG: Plot time evolution with windowing applied (SU2)
            # =====================================================================
            if apod_window is not None:
                print("    Creating windowed time evolution debug plot (SU2)...")
                # Use z-component (or first if not available)
                comp_idx = 2 if spin_dim_SU2 > 2 else 0
                M_NL_example = M_NL_SU2[comp_idx]
                comp_label = component_labels_SU2[comp_idx] if comp_idx < len(component_labels_SU2) else f'comp{comp_idx}'
                
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
                plt.savefig(os.path.join(dir, "windowed_time_evolution_SU2_debug.pdf"), dpi=100)
                plt.close(fig_wind)
                print(f"    Saved: windowed_time_evolution_SU2_debug.pdf")
            
            # Process individual signals (M0, M1, M01)
            for sig_name, sig_components in [('M0_SU2', M0_comp_SU2), ('M1_SU2', M1_comp_SU2), ('M01_SU2', M01_comp_SU2)]:
                print(f"    Processing {sig_name}...")
                
                # Pre-compute all FFTs with caching
                sig_comp_FFs = [compute_2d_fft(sig_components[comp], cache_key=f'{sig_name}_{comp}') 
                               for comp in range(spin_dim_SU2)]
                
                # Create debug plot
                n_rows = min(spin_dim_SU2, 3)
                fig_sig, axes_sig = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
                if n_rows == 1:
                    axes_sig = axes_sig.reshape(1, -1)
                
                for comp in range(n_rows):
                    sig_comp = sig_components[comp]
                    label = component_labels_SU2[comp] if comp < len(component_labels_SU2) else f'comp{comp}'
                    
                    # Time domain plot
                    ax_time = axes_sig[comp, 0]
                    for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
                        ax_time.plot(sig_comp[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
                    ax_time.set_xlabel('Time index')
                    ax_time.set_ylabel(f'${sig_name}_{{{label}}}$')
                    ax_time.set_title(f'{label}-component (time domain)')
                    ax_time.legend(fontsize=6)
                    ax_time.grid(True, alpha=0.3)
                    
                    # 2D time-tau plot
                    ax_2d = axes_sig[comp, 1]
                    im = ax_2d.imshow(sig_comp, origin='lower', aspect='auto', cmap='RdBu_r',
                                      extent=[0, sig_comp.shape[1], tau[0], tau[-1]])
                    ax_2d.set_xlabel('Time index')
                    ax_2d.set_ylabel('τ')
                    ax_2d.set_title(f'{label}-component (τ, t)')
                    plt.colorbar(im, ax=ax_2d)
                    
                    # Frequency domain plot (use cached FFT)
                    sig_comp_FF = sig_comp_FFs[comp]
                    ax_freq = axes_sig[comp, 2]
                    im_freq = ax_freq.imshow(sig_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                              norm=_get_norm(sig_comp_FF, norm_type), extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                    ax_freq.set_xlabel('$\\omega_{\\tau}$ (THz)')
                    ax_freq.set_ylabel('$\\omega_t$ (THz)')
                    ax_freq.set_title(f'{label}-component (freq domain)')
                    if omega_tau_window is not None:
                        ax_freq.set_xlim(omega_tau_window[0], omega_tau_window[1])
                    if omega_t_window is not None:
                        ax_freq.set_ylim(omega_t_window)
                    # Add energy level reference lines
                    if energy_levels_mev:
                        add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                    plt.colorbar(im_freq, ax=ax_freq)
                    
                    np.savetxt(os.path.join(dir, f"{sig_name}_FF_{label}.txt"), sig_comp_FF)
                
                plt.tight_layout()
                plt.savefig(os.path.join(dir, f"{sig_name}_components_debug.pdf"))
                plt.clf()
                plt.close()
                
                # Main spectrum plot (z-component for SU2) - use cached FFT
                z_idx = 2 if spin_dim_SU2 > 2 else 0
                sig_z_FF = sig_comp_FFs[z_idx]  # Reuse cached FFT
                np.savetxt(os.path.join(dir, f"{sig_name}_FF.txt"), sig_z_FF)
                results[f'{sig_name}_FF'] = sig_z_FF
                
                # Calculate appropriate figure size
                tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
                t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
                figsize_spec = calculate_spectrum_figsize(tau_range, t_range)
                
                fig_spec = plt.figure(figsize=figsize_spec)
                plt.imshow(sig_z_FF, origin='lower',
                           extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                           aspect='auto', cmap='gnuplot2', norm=_get_norm(sig_z_FF, norm_type))
                plt.xlabel('$\\omega_{\\tau}$ (THz)')
                plt.ylabel('$\\omega_t$ (THz)')
                plt.colorbar(label='Intensity')
                plt.title(f'{sig_name} Spectrum')
                if omega_tau_window is not None:
                    plt.xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    plt.ylim(omega_t_window)
                plt.savefig(os.path.join(dir, f"{sig_name}_SPEC.pdf"), dpi=100)
                plt.close(fig_spec)
            
            # M_NL analysis for SU2
            print("    Processing M_NL_SU2...")
            n_rows = min(spin_dim_SU2, 3)
            fig_debug, axes_debug = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes_debug = axes_debug.reshape(1, -1)
            
            # Pre-compute all M_NL FFTs with caching
            M_NL_SU2_FFs = [compute_2d_fft(M_NL_SU2[comp], cache_key=f'M_NL_SU2_{comp}') 
                           for comp in range(spin_dim_SU2)]
            
            for comp in range(n_rows):
                M_NL_comp = M_NL_SU2[comp]
                label = component_labels_SU2[comp] if comp < len(component_labels_SU2) else f'comp{comp}'
                
                # Time domain plot
                ax_time = axes_debug[comp, 0]
                for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
                    ax_time.plot(M_NL_comp[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
                ax_time.set_xlabel('Time index')
                ax_time.set_ylabel(f'$M_{{NL,{label}}}$')
                ax_time.set_title(f'{label}-component (time domain)')
                ax_time.legend(fontsize=6)
                ax_time.grid(True, alpha=0.3)
                
                # 2D time-tau plot
                ax_2d = axes_debug[comp, 1]
                im = ax_2d.imshow(M_NL_comp, origin='lower', aspect='auto', cmap='RdBu_r',
                                  extent=[0, M_NL_comp.shape[1], tau[0], tau[-1]])
                ax_2d.set_xlabel('Time index')
                ax_2d.set_ylabel('τ')
                ax_2d.set_title(f'{label}-component (τ, t)')
                plt.colorbar(im, ax=ax_2d)
                
                # Frequency domain plot (use cached FFT)
                M_NL_comp_FF = M_NL_SU2_FFs[comp]
                ax_freq = axes_debug[comp, 2]
                im_freq = ax_freq.imshow(M_NL_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                          norm=_get_norm(M_NL_comp_FF, norm_type), extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                ax_freq.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_freq.set_ylabel('$\\omega_t$ (THz)')
                ax_freq.set_title(f'{label}-component (freq domain)')
                if omega_tau_window is not None:
                    ax_freq.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_freq.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                plt.colorbar(im_freq, ax=ax_freq)
                
                np.savetxt(os.path.join(dir, f"M_NL_SU2_FF_{label}.txt"), M_NL_comp_FF)
            
            plt.tight_layout()
            plt.savefig(os.path.join(dir, "M_NL_SU2_components_debug.pdf"), dpi=100)
            plt.close(fig_debug)
            
            # Main M_NL spectrum for SU2 (z-component) - use cached FFT
            z_idx = 2 if spin_dim_SU2 > 2 else 0
            M_NL_z_FF = M_NL_SU2_FFs[z_idx]
            np.savetxt(os.path.join(dir, "M_NL_SU2_FF.txt"), M_NL_z_FF)
            results['M_NL_SU2_FF'] = M_NL_z_FF
            
            # Calculate appropriate figure size based on axis ranges
            tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
            t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
            figsize_nl = calculate_spectrum_figsize(tau_range, t_range)
            
            fig_nl = plt.figure(figsize=figsize_nl)
            ax_nl = fig_nl.add_subplot(111)
            im_nl = ax_nl.imshow(M_NL_z_FF, origin='lower',
                       extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                       aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_z_FF, norm_type))
            ax_nl.set_xlabel('$\\omega_{\\tau}$ (THz)')
            ax_nl.set_ylabel('$\\omega_t$ (THz)')
            plt.colorbar(im_nl, ax=ax_nl, label='Intensity')
            ax_nl.set_title('$M_{NL}$ Spectrum (SU(2))')
            if omega_tau_window is not None:
                ax_nl.set_xlim(omega_tau_window[0], omega_tau_window[1])
            if omega_t_window is not None:
                ax_nl.set_ylim(omega_t_window)
            # Add energy level reference lines
            if energy_levels_mev:
                add_energy_level_lines(ax_nl, energy_levels_mev, omega_t_window)
            plt.savefig(os.path.join(dir, "M_NL_SU2_SPEC.pdf"), dpi=100)
            plt.close(fig_nl)
        
        # =====================================================================
        # Process SU(3) sublattice
        # =====================================================================
        if '/reference/M_global_SU3' in f:
            print("\n  Processing SU(3) sublattice...")
            M0_SU3 = f['/reference/M_global_SU3'][:]
            spin_dim_SU3 = M0_SU3.shape[1]
            length = len(M0_SU3[:, 0])
            
            # Initialize arrays for all components
            M_NL_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
            M0_comp_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
            M1_comp_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
            M01_comp_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
            
            for i, tau_val in enumerate(tau_values):
                tau_group = f[f'/tau_scan/tau_{i}']
                M1_SU3 = tau_group['M1_global_SU3'][:]
                M01_SU3 = tau_group['M01_global_SU3'][:]
                
                for comp in range(spin_dim_SU3):
                    M0 = M0_SU3[:, comp]
                    M1 = M1_SU3[:, comp]
                    M01 = M01_SU3[:, comp]
                    
                    min_len = min(len(M0), len(M1), len(M01), length)
                    M_NL_SU3[comp, i, :min_len] = M01[:min_len] - M0[:min_len] - M1[:min_len]
                    M0_comp_SU3[comp, i, :min_len] = M0[:min_len]
                    M1_comp_SU3[comp, i, :min_len] = M1[:min_len]
                    M01_comp_SU3[comp, i, :min_len] = M01[:min_len]
            
            # =====================================================================
            # DEBUG: Plot time evolution with windowing applied (SU3)
            # =====================================================================
            if apod_window is not None:
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
                plt.savefig(os.path.join(dir, "windowed_time_evolution_SU3_debug.pdf"), dpi=100)
                plt.close(fig_wind)
                print(f"    Saved: windowed_time_evolution_SU3_debug.pdf")
            
            # Process individual signals (M0, M1, M01)
            for sig_name, sig_components in [('M0_SU3', M0_comp_SU3), ('M1_SU3', M1_comp_SU3), ('M01_SU3', M01_comp_SU3)]:
                print(f"    Processing {sig_name}...")
                
                # Pre-compute all FFTs with caching
                sig_comp_FFs = [compute_2d_fft(sig_components[comp], cache_key=f'{sig_name}_{comp}') 
                               for comp in range(spin_dim_SU3)]
                
                # Create debug plot
                n_rows = min(spin_dim_SU3, 8)
                fig_sig, axes_sig = plt.subplots(n_rows, 3, figsize=(15, 3*n_rows))
                if n_rows == 1:
                    axes_sig = axes_sig.reshape(1, -1)
                
                for comp in range(n_rows):
                    sig_comp = sig_components[comp]
                    label = component_labels_SU3[comp] if comp < len(component_labels_SU3) else f'comp{comp}'
                    
                    # Time domain plot
                    ax_time = axes_sig[comp, 0]
                    for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
                        ax_time.plot(sig_comp[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
                    ax_time.set_xlabel('Time index')
                    ax_time.set_ylabel(f'${sig_name}_{{{label}}}$')
                    ax_time.set_title(f'{label}-component (time domain)')
                    ax_time.legend(fontsize=6)
                    ax_time.grid(True, alpha=0.3)
                    
                    # 2D time-tau plot
                    ax_2d = axes_sig[comp, 1]
                    im = ax_2d.imshow(sig_comp, origin='lower', aspect='auto', cmap='RdBu_r',
                                      extent=[0, sig_comp.shape[1], tau[0], tau[-1]])
                    ax_2d.set_xlabel('Time index')
                    ax_2d.set_ylabel('τ')
                    ax_2d.set_title(f'{label}-component (τ, t)')
                    plt.colorbar(im, ax=ax_2d)
                    
                    # Frequency domain plot (use cached FFT)
                    sig_comp_FF = sig_comp_FFs[comp]
                    ax_freq = axes_sig[comp, 2]
                    im_freq = ax_freq.imshow(sig_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                              norm=_get_norm(sig_comp_FF, norm_type), extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                    ax_freq.set_xlabel('$\\omega_{\\tau}$ (THz)')
                    ax_freq.set_ylabel('$\\omega_t$ (THz)')
                    ax_freq.set_title(f'{label}-component (freq domain)')
                    if omega_tau_window is not None:
                        ax_freq.set_xlim(omega_tau_window[0], omega_tau_window[1])
                    if omega_t_window is not None:
                        ax_freq.set_ylim(omega_t_window)
                    # Add energy level reference lines
                    if energy_levels_mev:
                        add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                    plt.colorbar(im_freq, ax=ax_freq)
                    
                    np.savetxt(os.path.join(dir, f"{sig_name}_FF_{label}.txt"), sig_comp_FF)
                
                plt.tight_layout()
                plt.savefig(os.path.join(dir, f"{sig_name}_components_debug.pdf"), dpi=100)
                plt.close(fig_sig)  # Explicitly close figure
                
                # Total spectrum (sum over cached FFTs - no recomputation needed)
                sig_total_FF = sum(sig_comp_FFs)
                np.savetxt(os.path.join(dir, f"{sig_name}_FF.txt"), sig_total_FF)
                results[f'{sig_name}_FF'] = sig_total_FF
                
                # Calculate appropriate figure size
                tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
                t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
                figsize_spec = calculate_spectrum_figsize(tau_range, t_range)
                
                fig_spec = plt.figure(figsize=figsize_spec)
                plt.imshow(sig_total_FF, origin='lower',
                           extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                           aspect='auto', cmap='gnuplot2', norm=_get_norm(sig_total_FF, norm_type))
                plt.xlabel('$\\omega_{\\tau}$ (THz)')
                plt.ylabel('$\\omega_t$ (THz)')
                plt.colorbar(label='Intensity')
                plt.title(f'{sig_name} Spectrum (total)')
                if omega_tau_window is not None:
                    plt.xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    plt.ylim(omega_t_window)
                plt.savefig(os.path.join(dir, f"{sig_name}_SPEC.pdf"), dpi=100)
                plt.close(fig_spec)
            
            # M_NL analysis for SU3
            print("    Processing M_NL_SU3...")
            n_rows = min(spin_dim_SU3, 8)
            fig_debug, axes_debug = plt.subplots(n_rows, 3, figsize=(15, 3*n_rows))
            if n_rows == 1:
                axes_debug = axes_debug.reshape(1, -1)
            
            # Pre-compute all M_NL FFTs with caching
            M_NL_SU3_FFs = [compute_2d_fft(M_NL_SU3[comp], cache_key=f'M_NL_SU3_{comp}') 
                           for comp in range(spin_dim_SU3)]
            
            for comp in range(n_rows):
                M_NL_comp = M_NL_SU3[comp]
                label = component_labels_SU3[comp] if comp < len(component_labels_SU3) else f'comp{comp}'
                
                # Time domain plot
                ax_time = axes_debug[comp, 0]
                for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
                    ax_time.plot(M_NL_comp[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
                ax_time.set_xlabel('Time index')
                ax_time.set_ylabel(f'$M_{{NL,{label}}}$')
                ax_time.set_title(f'{label}-component (time domain)')
                ax_time.legend(fontsize=6)
                ax_time.grid(True, alpha=0.3)
                
                # 2D time-tau plot
                ax_2d = axes_debug[comp, 1]
                im = ax_2d.imshow(M_NL_comp, origin='lower', aspect='auto', cmap='RdBu_r',
                                  extent=[0, M_NL_comp.shape[1], tau[0], tau[-1]])
                ax_2d.set_xlabel('Time index')
                ax_2d.set_ylabel('τ')
                ax_2d.set_title(f'{label}-component (τ, t)')
                plt.colorbar(im, ax=ax_2d)
                
                # Frequency domain plot (use cached FFT)
                M_NL_comp_FF = M_NL_SU3_FFs[comp]
                ax_freq = axes_debug[comp, 2]
                im_freq = ax_freq.imshow(M_NL_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                          norm=_get_norm(M_NL_comp_FF, norm_type), extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                ax_freq.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_freq.set_ylabel('$\\omega_t$ (THz)')
                ax_freq.set_title(f'{label}-component (freq domain)')
                if omega_tau_window is not None:
                    ax_freq.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_freq.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                plt.colorbar(im_freq, ax=ax_freq)
                
                np.savetxt(os.path.join(dir, f"M_NL_SU3_FF_{label}.txt"), M_NL_comp_FF)
            
            plt.tight_layout()
            plt.savefig(os.path.join(dir, "M_NL_SU3_components_debug.pdf"), dpi=100)
            plt.close(fig_debug)
            
            # Total M_NL spectrum for SU3 (sum over cached FFTs)
            M_NL_total_FF = sum(M_NL_SU3_FFs)
            np.savetxt(os.path.join(dir, "M_NL_SU3_FF.txt"), M_NL_total_FF)
            results['M_NL_SU3_FF'] = M_NL_total_FF
            
            # Calculate appropriate figure size
            tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
            t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
            figsize_nl_su3 = calculate_spectrum_figsize(tau_range, t_range)
            
            fig_nl = plt.figure(figsize=figsize_nl_su3)
            ax_nl = fig_nl.add_subplot(111)
            im_nl = ax_nl.imshow(M_NL_total_FF, origin='lower',
                       extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                       aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_total_FF, norm_type))
            ax_nl.set_xlabel('$\\omega_{\\tau}$ (THz)')
            ax_nl.set_ylabel('$\\omega_t$ (THz)')
            plt.colorbar(im_nl, ax=ax_nl, label='Intensity')
            ax_nl.set_title('$M_{NL}$ Spectrum (SU(3) total)')
            if omega_tau_window is not None:
                ax_nl.set_xlim(omega_tau_window[0], omega_tau_window[1])
            if omega_t_window is not None:
                ax_nl.set_ylim(omega_t_window)
            # Add energy level reference lines
            if energy_levels_mev:
                add_energy_level_lines(ax_nl, energy_levels_mev, omega_t_window)
            plt.savefig(os.path.join(dir, "M_NL_SU3_SPEC.pdf"), dpi=100)
            plt.close(fig_nl)
            
            # =====================================================================
            # λ5 + λ7 mode analysis (Tm magnetic dipole observable)
            # λ5 and λ7 are indices 4 and 6 respectively
            # Weighted sum: λ5 + λ7 (equal weight)
            # =====================================================================
            if spin_dim_SU3 >= 8:
                print("    Processing λ5 + λ7 mode (equal weight)...")
                
                # Equal weights for λ5 and λ7 combination
                LAMBDA5_WEIGHT = 1.0
                LAMBDA7_WEIGHT = 1.0
                
                # Combine λ5 (index 4) and λ7 (index 6) in time domain before FFT (equal weight)
                M_NL_lambda57 = LAMBDA5_WEIGHT * M_NL_SU3[4] + LAMBDA7_WEIGHT * M_NL_SU3[6]
                M0_lambda57 = LAMBDA5_WEIGHT * M0_comp_SU3[4] + LAMBDA7_WEIGHT * M0_comp_SU3[6]
                M1_lambda57 = LAMBDA5_WEIGHT * M1_comp_SU3[4] + LAMBDA7_WEIGHT * M1_comp_SU3[6]
                M01_lambda57 = LAMBDA5_WEIGHT * M01_comp_SU3[4] + LAMBDA7_WEIGHT * M01_comp_SU3[6]
                
                # Create debug plot for λ5 + λ7
                fig_l57, axes_l57 = plt.subplots(2, 3, figsize=(15, 8))
                
                # Row 0: M_NL (λ5 + λ7)
                # Time domain
                ax_time = axes_l57[0, 0]
                for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
                    ax_time.plot(M_NL_lambda57[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
                ax_time.set_xlabel('Time index')
                ax_time.set_ylabel(f'$M_{{NL}}$ ({LAMBDA5_WEIGHT}$\\lambda_5$+{LAMBDA7_WEIGHT}$\\lambda_7$)')
                ax_time.set_title(f'{LAMBDA5_WEIGHT}$\\lambda_5$ + {LAMBDA7_WEIGHT}$\\lambda_7$ (time domain)')
                ax_time.legend(fontsize=6)
                ax_time.grid(True, alpha=0.3)
                
                # 2D time-tau plot
                ax_2d = axes_l57[0, 1]
                im = ax_2d.imshow(M_NL_lambda57, origin='lower', aspect='auto', cmap='RdBu_r',
                                  extent=[0, M_NL_lambda57.shape[1], tau[0], tau[-1]])
                ax_2d.set_xlabel('Time index')
                ax_2d.set_ylabel('τ')
                ax_2d.set_title(f'{LAMBDA5_WEIGHT}$\\lambda_5$ + {LAMBDA7_WEIGHT}$\\lambda_7$ (τ, t)')
                plt.colorbar(im, ax=ax_2d)
                
                # Frequency domain
                M_NL_lambda57_FF = compute_2d_fft(M_NL_lambda57)
                ax_freq = axes_l57[0, 2]
                im_freq = ax_freq.imshow(M_NL_lambda57_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                          norm=_get_norm(M_NL_lambda57_FF, norm_type),
                                          extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                ax_freq.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_freq.set_ylabel('$\\omega_t$ (THz)')
                ax_freq.set_title(f'$M_{{NL}}$ ({LAMBDA5_WEIGHT}$\\lambda_5$+{LAMBDA7_WEIGHT}$\\lambda_7$)')
                if omega_tau_window is not None:
                    ax_freq.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_freq.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                plt.colorbar(im_freq, ax=ax_freq)
                
                # Row 1: Individual M0, M1, M01 spectra for λ5 + λ7
                M0_lambda57_FF = compute_2d_fft(M0_lambda57)
                M1_lambda57_FF = compute_2d_fft(M1_lambda57)
                M01_lambda57_FF = compute_2d_fft(M01_lambda57)
                
                for idx, (name, data_FF) in enumerate([('M0', M0_lambda57_FF), ('M1', M1_lambda57_FF), ('M01', M01_lambda57_FF)]):
                    ax = axes_l57[1, idx]
                    im = ax.imshow(data_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                   norm=_get_norm(data_FF, norm_type),
                                   extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                    ax.set_xlabel('$\\omega_{\\tau}$ (THz)')
                    ax.set_ylabel('$\\omega_t$ (THz)')
                    ax.set_title(f'${name}$ ({LAMBDA5_WEIGHT}$\\lambda_5$+{LAMBDA7_WEIGHT}$\\lambda_7$)')
                    if omega_tau_window is not None:
                        ax.set_xlim(omega_tau_window[0], omega_tau_window[1])
                    if omega_t_window is not None:
                        ax.set_ylim(omega_t_window)
                    # Add energy level reference lines
                    if energy_levels_mev:
                        add_energy_level_lines(ax, energy_levels_mev, omega_t_window)
                    plt.colorbar(im, ax=ax)
                
                plt.tight_layout()
                plt.savefig(os.path.join(dir, "M_NL_lambda57_debug.pdf"), dpi=100)
                plt.close(fig_l57)
                
                # Save data
                np.savetxt(os.path.join(dir, "M_NL_lambda57_FF.txt"), M_NL_lambda57_FF)
                np.savetxt(os.path.join(dir, "M0_lambda57_FF.txt"), M0_lambda57_FF)
                np.savetxt(os.path.join(dir, "M1_lambda57_FF.txt"), M1_lambda57_FF)
                np.savetxt(os.path.join(dir, "M01_lambda57_FF.txt"), M01_lambda57_FF)
                results['M_NL_lambda57_FF'] = M_NL_lambda57_FF
                results['M0_lambda57_FF'] = M0_lambda57_FF
                results['M1_lambda57_FF'] = M1_lambda57_FF
                results['M01_lambda57_FF'] = M01_lambda57_FF
                
                # Main λ5 + λ7 spectrum plot
                tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
                t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
                figsize_l57 = calculate_spectrum_figsize(tau_range, t_range)
                
                fig_l57_main = plt.figure(figsize=figsize_l57)
                ax_l57 = fig_l57_main.add_subplot(111)
                im_l57 = ax_l57.imshow(M_NL_lambda57_FF, origin='lower',
                           extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                           aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_lambda57_FF, norm_type))
                ax_l57.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_l57.set_ylabel('$\\omega_t$ (THz)')
                plt.colorbar(im_l57, ax=ax_l57, label='Intensity')
                ax_l57.set_title(f'$M_{{NL}}$ Spectrum ({LAMBDA5_WEIGHT}$\\lambda_5$ + {LAMBDA7_WEIGHT}$\\lambda_7$)')
                if omega_tau_window is not None:
                    ax_l57.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_l57.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_l57, energy_levels_mev, omega_t_window)
                plt.savefig(os.path.join(dir, "M_NL_lambda57_SPEC.pdf"), dpi=100)
                plt.close(fig_l57_main)
            
            # λ2 + λ5 combined mode (equal weight)
            if spin_dim_SU3 >= 8:
                print("    Processing λ2 + λ5 mode (equal weight)...")
                
                # Equal weights for λ2 and λ5 combination
                LAMBDA2_WEIGHT_25 = 1.0
                LAMBDA5_WEIGHT_25 = 1.0
                
                # Combine λ2 (index 1) and λ5 (index 4) in time domain before FFT (equal weight)
                M_NL_lambda25 = LAMBDA2_WEIGHT_25 * M_NL_SU3[1] + LAMBDA5_WEIGHT_25 * M_NL_SU3[4]
                M0_lambda25 = LAMBDA2_WEIGHT_25 * M0_comp_SU3[1] + LAMBDA5_WEIGHT_25 * M0_comp_SU3[4]
                M1_lambda25 = LAMBDA2_WEIGHT_25 * M1_comp_SU3[1] + LAMBDA5_WEIGHT_25 * M1_comp_SU3[4]
                M01_lambda25 = LAMBDA2_WEIGHT_25 * M01_comp_SU3[1] + LAMBDA5_WEIGHT_25 * M01_comp_SU3[4]
                
                # Create debug plot for λ2 + λ5
                fig_l25, axes_l25 = plt.subplots(2, 3, figsize=(15, 8))
                
                # Row 0: M_NL (λ2 + λ5)
                # Time domain
                ax_time = axes_l25[0, 0]
                for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
                    ax_time.plot(M_NL_lambda25[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
                ax_time.set_xlabel('Time index')
                ax_time.set_ylabel(f'$M_{{NL}}$ ($\\lambda_2$+$\\lambda_5$)')
                ax_time.set_title(f'$\\lambda_2$ + $\\lambda_5$ (time domain)')
                ax_time.legend(fontsize=6)
                ax_time.grid(True, alpha=0.3)
                
                # 2D time-tau plot
                ax_2d = axes_l25[0, 1]
                im = ax_2d.imshow(M_NL_lambda25, origin='lower', aspect='auto', cmap='RdBu_r',
                                  extent=[0, M_NL_lambda25.shape[1], tau[0], tau[-1]])
                ax_2d.set_xlabel('Time index')
                ax_2d.set_ylabel('τ')
                ax_2d.set_title(f'$\\lambda_2$ + $\\lambda_5$ (τ, t)')
                plt.colorbar(im, ax=ax_2d)
                
                # Frequency domain
                M_NL_lambda25_FF = compute_2d_fft(M_NL_lambda25)
                ax_freq = axes_l25[0, 2]
                im_freq = ax_freq.imshow(M_NL_lambda25_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                          norm=_get_norm(M_NL_lambda25_FF, norm_type),
                                          extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                ax_freq.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_freq.set_ylabel('$\\omega_t$ (THz)')
                ax_freq.set_title(f'$M_{{NL}}$ ($\\lambda_2$+$\\lambda_5$)')
                if omega_tau_window is not None:
                    ax_freq.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_freq.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                plt.colorbar(im_freq, ax=ax_freq)
                
                # Row 1: Individual M0, M1, M01 spectra for λ2 + λ5
                M0_lambda25_FF = compute_2d_fft(M0_lambda25)
                M1_lambda25_FF = compute_2d_fft(M1_lambda25)
                M01_lambda25_FF = compute_2d_fft(M01_lambda25)
                
                for idx, (name, data_FF) in enumerate([('M0', M0_lambda25_FF), ('M1', M1_lambda25_FF), ('M01', M01_lambda25_FF)]):
                    ax = axes_l25[1, idx]
                    im = ax.imshow(data_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                   norm=_get_norm(data_FF, norm_type),
                                   extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                    ax.set_xlabel('$\\omega_{\\tau}$ (THz)')
                    ax.set_ylabel('$\\omega_t$ (THz)')
                    ax.set_title(f'${name}$ ($\\lambda_2$+$\\lambda_5$)')
                    if omega_tau_window is not None:
                        ax.set_xlim(omega_tau_window[0], omega_tau_window[1])
                    if omega_t_window is not None:
                        ax.set_ylim(omega_t_window)
                    # Add energy level reference lines
                    if energy_levels_mev:
                        add_energy_level_lines(ax, energy_levels_mev, omega_t_window)
                    plt.colorbar(im, ax=ax)
                
                plt.tight_layout()
                plt.savefig(os.path.join(dir, "M_NL_lambda25_debug.pdf"), dpi=100)
                plt.close(fig_l25)
                
                # Save data
                np.savetxt(os.path.join(dir, "M_NL_lambda25_FF.txt"), M_NL_lambda25_FF)
                np.savetxt(os.path.join(dir, "M0_lambda25_FF.txt"), M0_lambda25_FF)
                np.savetxt(os.path.join(dir, "M1_lambda25_FF.txt"), M1_lambda25_FF)
                np.savetxt(os.path.join(dir, "M01_lambda25_FF.txt"), M01_lambda25_FF)
                results['M_NL_lambda25_FF'] = M_NL_lambda25_FF
                results['M0_lambda25_FF'] = M0_lambda25_FF
                results['M1_lambda25_FF'] = M1_lambda25_FF
                results['M01_lambda25_FF'] = M01_lambda25_FF
                
                # Main λ2 + λ5 spectrum plot
                tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
                t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
                figsize_l25 = calculate_spectrum_figsize(tau_range, t_range)
                
                fig_l25_main = plt.figure(figsize=figsize_l25)
                ax_l25 = fig_l25_main.add_subplot(111)
                im_l25 = ax_l25.imshow(M_NL_lambda25_FF, origin='lower',
                           extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                           aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_lambda25_FF, norm_type))
                ax_l25.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_l25.set_ylabel('$\\omega_t$ (THz)')
                plt.colorbar(im_l25, ax=ax_l25, label='Intensity')
                ax_l25.set_title(f'$M_{{NL}}$ Spectrum ($\\lambda_2$ + $\\lambda_5$)')
                if omega_tau_window is not None:
                    ax_l25.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_l25.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_l25, energy_levels_mev, omega_t_window)
                plt.savefig(os.path.join(dir, "M_NL_lambda25_SPEC.pdf"), dpi=100)
                plt.close(fig_l25_main)
            
            # λ2 + λ5 + λ7 combined mode
            if spin_dim_SU3 >= 8:
                print("    Processing λ2 + λ5 + λ7 mode (equal weight)...")
                
                # Equal weights for λ2, λ5 and λ7 combination
                LAMBDA2_WEIGHT = 1.0
                LAMBDA5_WEIGHT_257 = 1.0
                LAMBDA7_WEIGHT_257 = 1.0
                
                # Combine λ2 (index 1), λ5 (index 4) and λ7 (index 6) in time domain before FFT (equal weight)
                M_NL_lambda257 = LAMBDA2_WEIGHT * M_NL_SU3[1] + LAMBDA5_WEIGHT_257 * M_NL_SU3[4] + LAMBDA7_WEIGHT_257 * M_NL_SU3[6]
                M0_lambda257 = LAMBDA2_WEIGHT * M0_comp_SU3[1] + LAMBDA5_WEIGHT_257 * M0_comp_SU3[4] + LAMBDA7_WEIGHT_257 * M0_comp_SU3[6]
                M1_lambda257 = LAMBDA2_WEIGHT * M1_comp_SU3[1] + LAMBDA5_WEIGHT_257 * M1_comp_SU3[4] + LAMBDA7_WEIGHT_257 * M1_comp_SU3[6]
                M01_lambda257 = LAMBDA2_WEIGHT * M01_comp_SU3[1] + LAMBDA5_WEIGHT_257 * M01_comp_SU3[4] + LAMBDA7_WEIGHT_257 * M01_comp_SU3[6]
                
                # Create debug plot for λ2 + λ5 + λ7
                fig_l257, axes_l257 = plt.subplots(2, 3, figsize=(15, 8))
                
                # Row 0: M_NL (λ2 + λ5 + λ7)
                # Time domain
                ax_time = axes_l257[0, 0]
                for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
                    ax_time.plot(M_NL_lambda257[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
                ax_time.set_xlabel('Time index')
                ax_time.set_ylabel(f'$M_{{NL}}$ ({LAMBDA2_WEIGHT}$\\lambda_2$+{LAMBDA5_WEIGHT_257}$\\lambda_5$+{LAMBDA7_WEIGHT_257}$\\lambda_7$)')
                ax_time.set_title(f'{LAMBDA2_WEIGHT}$\\lambda_2$ + {LAMBDA5_WEIGHT_257}$\\lambda_5$ + {LAMBDA7_WEIGHT_257}$\\lambda_7$ (time domain)')
                ax_time.legend(fontsize=6)
                ax_time.grid(True, alpha=0.3)
                
                # 2D time-tau plot
                ax_2d = axes_l257[0, 1]
                im = ax_2d.imshow(M_NL_lambda257, origin='lower', aspect='auto', cmap='RdBu_r',
                                  extent=[0, M_NL_lambda257.shape[1], tau[0], tau[-1]])
                ax_2d.set_xlabel('Time index')
                ax_2d.set_ylabel('τ')
                ax_2d.set_title(f'{LAMBDA2_WEIGHT}$\\lambda_2$ + {LAMBDA5_WEIGHT_257}$\\lambda_5$ + {LAMBDA7_WEIGHT_257}$\\lambda_7$ (τ, t)')
                plt.colorbar(im, ax=ax_2d)
                
                # Frequency domain
                M_NL_lambda257_FF = compute_2d_fft(M_NL_lambda257)
                ax_freq = axes_l257[0, 2]
                im_freq = ax_freq.imshow(M_NL_lambda257_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                          norm=_get_norm(M_NL_lambda257_FF, norm_type),
                                          extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                ax_freq.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_freq.set_ylabel('$\\omega_t$ (THz)')
                ax_freq.set_title(f'$M_{{NL}}$ ({LAMBDA2_WEIGHT}$\\lambda_2$+{LAMBDA5_WEIGHT_257}$\\lambda_5$+{LAMBDA7_WEIGHT_257}$\\lambda_7$)')
                if omega_tau_window is not None:
                    ax_freq.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_freq.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                plt.colorbar(im_freq, ax=ax_freq)
                
                # Row 1: Individual M0, M1, M01 spectra for λ2 + λ5 + λ7
                M0_lambda257_FF = compute_2d_fft(M0_lambda257)
                M1_lambda257_FF = compute_2d_fft(M1_lambda257)
                M01_lambda257_FF = compute_2d_fft(M01_lambda257)
                
                for idx, (name, data_FF) in enumerate([('M0', M0_lambda257_FF), ('M1', M1_lambda257_FF), ('M01', M01_lambda257_FF)]):
                    ax = axes_l257[1, idx]
                    im = ax.imshow(data_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                   norm=_get_norm(data_FF, norm_type),
                                   extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                    ax.set_xlabel('$\\omega_{\\tau}$ (THz)')
                    ax.set_ylabel('$\\omega_t$ (THz)')
                    ax.set_title(f'${name}$ ({LAMBDA2_WEIGHT}$\\lambda_2$+{LAMBDA5_WEIGHT_257}$\\lambda_5$+{LAMBDA7_WEIGHT_257}$\\lambda_7$)')
                    if omega_tau_window is not None:
                        ax.set_xlim(omega_tau_window[0], omega_tau_window[1])
                    if omega_t_window is not None:
                        ax.set_ylim(omega_t_window)
                    # Add energy level reference lines
                    if energy_levels_mev:
                        add_energy_level_lines(ax, energy_levels_mev, omega_t_window)
                    plt.colorbar(im, ax=ax)
                
                plt.tight_layout()
                plt.savefig(os.path.join(dir, "M_NL_lambda257_debug.pdf"), dpi=100)
                plt.close(fig_l257)
                
                # Save data
                np.savetxt(os.path.join(dir, "M_NL_lambda257_FF.txt"), M_NL_lambda257_FF)
                np.savetxt(os.path.join(dir, "M0_lambda257_FF.txt"), M0_lambda257_FF)
                np.savetxt(os.path.join(dir, "M1_lambda257_FF.txt"), M1_lambda257_FF)
                np.savetxt(os.path.join(dir, "M01_lambda257_FF.txt"), M01_lambda257_FF)
                results['M_NL_lambda257_FF'] = M_NL_lambda257_FF
                results['M0_lambda257_FF'] = M0_lambda257_FF
                results['M1_lambda257_FF'] = M1_lambda257_FF
                results['M01_lambda257_FF'] = M01_lambda257_FF
                
                # Main λ2 + λ5 + λ7 spectrum plot
                tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
                t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
                figsize_l257 = calculate_spectrum_figsize(tau_range, t_range)
                
                fig_l257_main = plt.figure(figsize=figsize_l257)
                ax_l257 = fig_l257_main.add_subplot(111)
                im_l257 = ax_l257.imshow(M_NL_lambda257_FF, origin='lower',
                           extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                           aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_lambda257_FF, norm_type))
                ax_l257.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_l257.set_ylabel('$\\omega_t$ (THz)')
                plt.colorbar(im_l257, ax=ax_l257, label='Intensity')
                ax_l257.set_title(f'$M_{{NL}}$ Spectrum ({LAMBDA2_WEIGHT}$\\lambda_2$ + {LAMBDA5_WEIGHT_257}$\\lambda_5$ + {LAMBDA7_WEIGHT_257}$\\lambda_7$)')
                if omega_tau_window is not None:
                    ax_l257.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_l257.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_l257, energy_levels_mev, omega_t_window)
                plt.savefig(os.path.join(dir, "M_NL_lambda257_SPEC.pdf"), dpi=100)
                plt.close(fig_l257_main)
            
            # λ2 + λ7 + x component combined mode
            if spin_dim_SU3 >= 8 and M_NL_SU2 is not None and M_NL_SU2.shape[0] >= 1:
                print("    Processing λ2 + λ7 + x component mode (equal weight)...")
                
                # Equal weights for λ2, λ7 and x combination
                LAMBDA2_WEIGHT_27x = 1.0
                LAMBDA7_WEIGHT_27x = 1.0
                X_WEIGHT_27x = 1.0
                
                # Combine λ2 (index 1), λ7 (index 6) from SU3 and x (index 0) from SU2 in time domain before FFT
                M_NL_lambda27x = LAMBDA2_WEIGHT_27x * M_NL_SU3[1] + LAMBDA7_WEIGHT_27x * M_NL_SU3[6] + X_WEIGHT_27x * M_NL_SU2[0]
                M0_lambda27x = LAMBDA2_WEIGHT_27x * M0_comp_SU3[1] + LAMBDA7_WEIGHT_27x * M0_comp_SU3[6] + X_WEIGHT_27x * M0_comp_SU2[0]
                M1_lambda27x = LAMBDA2_WEIGHT_27x * M1_comp_SU3[1] + LAMBDA7_WEIGHT_27x * M1_comp_SU3[6] + X_WEIGHT_27x * M1_comp_SU2[0]
                M01_lambda27x = LAMBDA2_WEIGHT_27x * M01_comp_SU3[1] + LAMBDA7_WEIGHT_27x * M01_comp_SU3[6] + X_WEIGHT_27x * M01_comp_SU2[0]
                
                # Create debug plot for λ2 + λ7 + x
                fig_l27x, axes_l27x = plt.subplots(2, 3, figsize=(15, 8))
                
                # Row 0: M_NL (λ2 + λ7 + x)
                # Time domain
                ax_time = axes_l27x[0, 0]
                for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
                    ax_time.plot(M_NL_lambda27x[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
                ax_time.set_xlabel('Time index')
                ax_time.set_ylabel(f'$M_{{NL}}$ ({LAMBDA2_WEIGHT_27x}$\\lambda_2$+{LAMBDA7_WEIGHT_27x}$\\lambda_7$+{X_WEIGHT_27x}$S_x$)')
                ax_time.set_title(f'{LAMBDA2_WEIGHT_27x}$\\lambda_2$ + {LAMBDA7_WEIGHT_27x}$\\lambda_7$ + {X_WEIGHT_27x}$S_x$ (time domain)')
                ax_time.legend(fontsize=6)
                ax_time.grid(True, alpha=0.3)
                
                # 2D time-tau plot
                ax_2d = axes_l27x[0, 1]
                im = ax_2d.imshow(M_NL_lambda27x, origin='lower', aspect='auto', cmap='RdBu_r',
                                  extent=[0, M_NL_lambda27x.shape[1], tau[0], tau[-1]])
                ax_2d.set_xlabel('Time index')
                ax_2d.set_ylabel('τ')
                ax_2d.set_title(f'{LAMBDA2_WEIGHT_27x}$\\lambda_2$ + {LAMBDA7_WEIGHT_27x}$\\lambda_7$ + {X_WEIGHT_27x}$S_x$ (τ, t)')
                plt.colorbar(im, ax=ax_2d)
                
                # Frequency domain
                M_NL_lambda27x_FF = compute_2d_fft(M_NL_lambda27x)
                ax_freq = axes_l27x[0, 2]
                im_freq = ax_freq.imshow(M_NL_lambda27x_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                          norm=_get_norm(M_NL_lambda27x_FF, norm_type),
                                          extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                ax_freq.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_freq.set_ylabel('$\\omega_t$ (THz)')
                ax_freq.set_title(f'$M_{{NL}}$ ({LAMBDA2_WEIGHT_27x}$\\lambda_2$+{LAMBDA7_WEIGHT_27x}$\\lambda_7$+{X_WEIGHT_27x}$S_x$)')
                if omega_tau_window is not None:
                    ax_freq.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_freq.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                plt.colorbar(im_freq, ax=ax_freq)
                
                # Row 1: Individual M0, M1, M01 spectra for λ2 + λ7 + x
                M0_lambda27x_FF = compute_2d_fft(M0_lambda27x)
                M1_lambda27x_FF = compute_2d_fft(M1_lambda27x)
                M01_lambda27x_FF = compute_2d_fft(M01_lambda27x)
                
                for idx, (name, data_FF) in enumerate([('M0', M0_lambda27x_FF), ('M1', M1_lambda27x_FF), ('M01', M01_lambda27x_FF)]):
                    ax = axes_l27x[1, idx]
                    im = ax.imshow(data_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                   norm=_get_norm(data_FF, norm_type),
                                   extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                    ax.set_xlabel('$\\omega_{\\tau}$ (THz)')
                    ax.set_ylabel('$\\omega_t$ (THz)')
                    ax.set_title(f'${name}$ ({LAMBDA2_WEIGHT_27x}$\\lambda_2$+{LAMBDA7_WEIGHT_27x}$\\lambda_7$+{X_WEIGHT_27x}$S_x$)')
                    if omega_tau_window is not None:
                        ax.set_xlim(omega_tau_window[0], omega_tau_window[1])
                    if omega_t_window is not None:
                        ax.set_ylim(omega_t_window)
                    # Add energy level reference lines
                    if energy_levels_mev:
                        add_energy_level_lines(ax, energy_levels_mev, omega_t_window)
                    plt.colorbar(im, ax=ax)
                
                plt.tight_layout()
                plt.savefig(os.path.join(dir, "M_NL_lambda27x_debug.pdf"), dpi=100)
                plt.close(fig_l27x)
                
                # Save data
                np.savetxt(os.path.join(dir, "M_NL_lambda27x_FF.txt"), M_NL_lambda27x_FF)
                np.savetxt(os.path.join(dir, "M0_lambda27x_FF.txt"), M0_lambda27x_FF)
                np.savetxt(os.path.join(dir, "M1_lambda27x_FF.txt"), M1_lambda27x_FF)
                np.savetxt(os.path.join(dir, "M01_lambda27x_FF.txt"), M01_lambda27x_FF)
                results['M_NL_lambda27x_FF'] = M_NL_lambda27x_FF
                results['M0_lambda27x_FF'] = M0_lambda27x_FF
                results['M1_lambda27x_FF'] = M1_lambda27x_FF
                results['M01_lambda27x_FF'] = M01_lambda27x_FF
                
                # Main λ2 + λ7 + x spectrum plot
                tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
                t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
                figsize_l27x = calculate_spectrum_figsize(tau_range, t_range)
                
                fig_l27x_main = plt.figure(figsize=figsize_l27x)
                ax_l27x = fig_l27x_main.add_subplot(111)
                im_l27x = ax_l27x.imshow(M_NL_lambda27x_FF, origin='lower',
                           extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                           aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_lambda27x_FF, norm_type))
                ax_l27x.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_l27x.set_ylabel('$\\omega_t$ (THz)')
                plt.colorbar(im_l27x, ax=ax_l27x, label='Intensity')
                ax_l27x.set_title(f'$M_{{NL}}$ Spectrum ({LAMBDA2_WEIGHT_27x}$\\lambda_2$ + {LAMBDA7_WEIGHT_27x}$\\lambda_7$ + {X_WEIGHT_27x}$S_x$)')
                if omega_tau_window is not None:
                    ax_l27x.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_l27x.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_l27x, energy_levels_mev, omega_t_window)
                plt.savefig(os.path.join(dir, "M_NL_lambda27x_SPEC.pdf"), dpi=100)
                plt.close(fig_l27x_main)
            
            # λ2 - 0.2*λ7 + x component combined mode (custom weights)
            if spin_dim_SU3 >= 8 and M_NL_SU2 is not None and M_NL_SU2.shape[0] >= 1:
                print("    Processing λ2 - 0.2*λ7 + x component mode (custom weights)...")
                
                # Custom weights for λ2, λ7 and x combination
                LAMBDA2_WEIGHT_27x_v2 = 1.0
                LAMBDA7_WEIGHT_27x_v2 = 0.2
                X_WEIGHT_27x_v2 = 1.6
                
                # Combine λ2 (index 1), λ7 (index 6) from SU3 and x (index 0) from SU2 in time domain before FFT
                M_NL_lambda27x_v2 = LAMBDA2_WEIGHT_27x_v2 * M_NL_SU3[1] + LAMBDA7_WEIGHT_27x_v2 * M_NL_SU3[6] + X_WEIGHT_27x_v2 * M_NL_SU2[0]
                M0_lambda27x_v2 = LAMBDA2_WEIGHT_27x_v2 * M0_comp_SU3[1] + LAMBDA7_WEIGHT_27x_v2 * M0_comp_SU3[6] + X_WEIGHT_27x_v2 * M0_comp_SU2[0]
                M1_lambda27x_v2 = LAMBDA2_WEIGHT_27x_v2 * M1_comp_SU3[1] + LAMBDA7_WEIGHT_27x_v2 * M1_comp_SU3[6] + X_WEIGHT_27x_v2 * M1_comp_SU2[0]
                M01_lambda27x_v2 = LAMBDA2_WEIGHT_27x_v2 * M01_comp_SU3[1] + LAMBDA7_WEIGHT_27x_v2 * M01_comp_SU3[6] + X_WEIGHT_27x_v2 * M01_comp_SU2[0]
                
                # Create debug plot for λ2 - 0.2*λ7 + x
                fig_l27x_v2, axes_l27x_v2 = plt.subplots(2, 3, figsize=(15, 8))
                
                # Row 0: M_NL (λ2 - 0.2*λ7 + x)
                # Time domain
                ax_time = axes_l27x_v2[0, 0]
                for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
                    ax_time.plot(M_NL_lambda27x_v2[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
                ax_time.set_xlabel('Time index')
                ax_time.set_ylabel(f'$M_{{NL}}$ ({LAMBDA2_WEIGHT_27x_v2}$\\lambda_2${LAMBDA7_WEIGHT_27x_v2:+.1f}$\\lambda_7$+{X_WEIGHT_27x_v2}$S_x$)')
                ax_time.set_title(f'{LAMBDA2_WEIGHT_27x_v2}$\\lambda_2$ {LAMBDA7_WEIGHT_27x_v2:+.1f}$\\lambda_7$ + {X_WEIGHT_27x_v2}$S_x$ (time domain)')
                ax_time.legend(fontsize=6)
                ax_time.grid(True, alpha=0.3)
                
                # 2D time-tau plot
                ax_2d = axes_l27x_v2[0, 1]
                im = ax_2d.imshow(M_NL_lambda27x_v2, origin='lower', aspect='auto', cmap='RdBu_r',
                                  extent=[0, M_NL_lambda27x_v2.shape[1], tau[0], tau[-1]])
                ax_2d.set_xlabel('Time index')
                ax_2d.set_ylabel('τ')
                ax_2d.set_title(f'{LAMBDA2_WEIGHT_27x_v2}$\\lambda_2$ {LAMBDA7_WEIGHT_27x_v2:+.1f}$\\lambda_7$ + {X_WEIGHT_27x_v2}$S_x$ (τ, t)')
                plt.colorbar(im, ax=ax_2d)
                
                # Frequency domain
                M_NL_lambda27x_v2_FF = compute_2d_fft(M_NL_lambda27x_v2)
                ax_freq = axes_l27x_v2[0, 2]
                im_freq = ax_freq.imshow(M_NL_lambda27x_v2_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                          norm=_get_norm(M_NL_lambda27x_v2_FF, norm_type),
                                          extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                ax_freq.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_freq.set_ylabel('$\\omega_t$ (THz)')
                ax_freq.set_title(f'$M_{{NL}}$ ({LAMBDA2_WEIGHT_27x_v2}$\\lambda_2${LAMBDA7_WEIGHT_27x_v2:+.1f}$\\lambda_7$+{X_WEIGHT_27x_v2}$S_x$)')
                if omega_tau_window is not None:
                    ax_freq.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_freq.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_freq, energy_levels_mev, omega_t_window)
                plt.colorbar(im_freq, ax=ax_freq)
                
                # Row 1: Individual M0, M1, M01 spectra for λ2 - 0.2*λ7 + x
                M0_lambda27x_v2_FF = compute_2d_fft(M0_lambda27x_v2)
                M1_lambda27x_v2_FF = compute_2d_fft(M1_lambda27x_v2)
                M01_lambda27x_v2_FF = compute_2d_fft(M01_lambda27x_v2)
                
                for idx, (name, data_FF) in enumerate([('M0', M0_lambda27x_v2_FF), ('M1', M1_lambda27x_v2_FF), ('M01', M01_lambda27x_v2_FF)]):
                    ax = axes_l27x_v2[1, idx]
                    im = ax.imshow(data_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                   norm=_get_norm(data_FF, norm_type),
                                   extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]])
                    ax.set_xlabel('$\\omega_{\\tau}$ (THz)')
                    ax.set_ylabel('$\\omega_t$ (THz)')
                    ax.set_title(f'${name}$ ({LAMBDA2_WEIGHT_27x_v2}$\\lambda_2${LAMBDA7_WEIGHT_27x_v2:+.1f}$\\lambda_7$+{X_WEIGHT_27x_v2}$S_x$)')
                    if omega_tau_window is not None:
                        ax.set_xlim(omega_tau_window[0], omega_tau_window[1])
                    if omega_t_window is not None:
                        ax.set_ylim(omega_t_window)
                    # Add energy level reference lines
                    if energy_levels_mev:
                        add_energy_level_lines(ax, energy_levels_mev, omega_t_window)
                    plt.colorbar(im, ax=ax)
                
                plt.tight_layout()
                plt.savefig(os.path.join(dir, "M_NL_lambda27x_v2_debug.pdf"), dpi=100)
                plt.close(fig_l27x_v2)
                
                # Save data
                np.savetxt(os.path.join(dir, "M_NL_lambda27x_v2_FF.txt"), M_NL_lambda27x_v2_FF)
                np.savetxt(os.path.join(dir, "M0_lambda27x_v2_FF.txt"), M0_lambda27x_v2_FF)
                np.savetxt(os.path.join(dir, "M1_lambda27x_v2_FF.txt"), M1_lambda27x_v2_FF)
                np.savetxt(os.path.join(dir, "M01_lambda27x_v2_FF.txt"), M01_lambda27x_v2_FF)
                results['M_NL_lambda27x_v2_FF'] = M_NL_lambda27x_v2_FF
                results['M0_lambda27x_v2_FF'] = M0_lambda27x_v2_FF
                results['M1_lambda27x_v2_FF'] = M1_lambda27x_v2_FF
                results['M01_lambda27x_v2_FF'] = M01_lambda27x_v2_FF
                
                # Main λ2 - 0.2*λ7 + x spectrum plot
                tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
                t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
                figsize_l27x_v2 = calculate_spectrum_figsize(tau_range, t_range)
                
                fig_l27x_v2_main = plt.figure(figsize=figsize_l27x_v2)
                ax_l27x_v2 = fig_l27x_v2_main.add_subplot(111)
                im_l27x_v2 = ax_l27x_v2.imshow(M_NL_lambda27x_v2_FF, origin='lower',
                           extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                           aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_lambda27x_v2_FF, norm_type))
                ax_l27x_v2.set_xlabel('$\\omega_{\\tau}$ (THz)')
                ax_l27x_v2.set_ylabel('$\\omega_t$ (THz)')
                plt.colorbar(im_l27x_v2, ax=ax_l27x_v2, label='Intensity')
                ax_l27x_v2.set_title(f'$M_{{NL}}$ Spectrum ({LAMBDA2_WEIGHT_27x_v2}$\\lambda_2$ {LAMBDA7_WEIGHT_27x_v2:+.1f}$\\lambda_7$ + {X_WEIGHT_27x_v2}$S_x$)')
                if omega_tau_window is not None:
                    ax_l27x_v2.set_xlim(omega_tau_window[0], omega_tau_window[1])
                if omega_t_window is not None:
                    ax_l27x_v2.set_ylim(omega_t_window)
                # Add energy level reference lines
                if energy_levels_mev:
                    add_energy_level_lines(ax_l27x_v2, energy_levels_mev, omega_t_window)
                plt.savefig(os.path.join(dir, "M_NL_lambda27x_v2_SPEC.pdf"), dpi=100)
                plt.close(fig_l27x_v2_main)
        
        # =====================================================================
        # Combined SU(2) + SU(3) spectrum
        # =====================================================================
        if 'M_NL_SU2_FF' in results and 'M_NL_SU3_FF' in results:
            print("\n  Computing combined SU(2)+SU(3) spectrum...")
            M_NL_combined = results['M_NL_SU2_FF'] + results['M_NL_SU3_FF']
            np.savetxt(os.path.join(dir, "M_NL_combined_FF.txt"), M_NL_combined)
            results['M_NL_combined_FF'] = M_NL_combined
            
            # Calculate appropriate figure size based on axis ranges
            tau_range = (omega_tau_window[0], omega_tau_window[1]) if omega_tau_window else (omega_tau[0], omega_tau[-1])
            t_range = omega_t_window if omega_t_window else (omega_t[0], omega_t[-1])
            figsize_comb = calculate_spectrum_figsize(tau_range, t_range)
            
            fig_comb = plt.figure(figsize=figsize_comb)
            ax_comb = fig_comb.add_subplot(111)
            im_comb = ax_comb.imshow(M_NL_combined, origin='lower',
                       extent=[omega_tau[0], omega_tau[-1], omega_t[0], omega_t[-1]],
                       aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_combined, norm_type))
            ax_comb.set_xlabel('$\\omega_{\\tau}$ (THz)')
            ax_comb.set_ylabel('$\\omega_t$ (THz)')
            plt.colorbar(im_comb, ax=ax_comb, label='Intensity')
            ax_comb.set_title('$M_{NL}$ Spectrum (SU(2) + SU(3))')
            if omega_tau_window is not None:
                ax_comb.set_xlim(omega_tau_window[0], omega_tau_window[1])
            if omega_t_window is not None:
                ax_comb.set_ylim(omega_t_window)
            # Add energy level reference lines
            if energy_levels_mev:
                add_energy_level_lines(ax_comb, energy_levels_mev, omega_t_window)
            plt.savefig(os.path.join(dir, "M_NL_combined_SPEC.pdf"), dpi=100)
            plt.close(fig_comb)
    
    # Clear FFT cache to free memory
    clear_fft_cache()
    
    print(f"\n  2D nonlinear spectroscopy analysis complete.")
    print(f"  Output files saved to: {dir}")
    
    return results


def read_2DCS_combined_hdf5(filepath: str, omega_t_window: Optional[Tuple[float, float]] = None,
                            omega_tau_window: Optional[Tuple[float, float]] = None,
                            norm_type: NormType = 'power',
                            window_type: str = 'gaussian',
                            energy_levels_mev: Optional[Dict[str, float]] = None,
                            load_from_cache: bool = False,
                            reload_components: bool = False) -> Dict[str, np.ndarray]:
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
        
    Returns:
        Dictionary with 2DCS results for SU(2), SU(3), and combined
    """
    output_dir = os.path.dirname(filepath)
    if output_dir == '':
        output_dir = '.'
    
    return read_2D_nonlinear(output_dir, omega_t_window, omega_tau_window, norm_type, 
                             window_type=window_type, energy_levels_mev=energy_levels_mev,
                             load_from_cache=load_from_cache, reload_components=reload_components)


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
    if omega_tau_window is not None:
        omega_t_window = (omega_t_window[0] * MEV_TO_THZ, omega_t_window[1] * MEV_TO_THZ)
    if omega_t_window is not None:
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
        
        Returns spectrum with shape (n_omega_t, n_omega_tau) for proper plotting
        with omega_tau on x-axis and omega_t on y-axis.
        """
        data_filtered = raw_data[:, data['t0_idx']:]
        data_dynamic = data_filtered - data_filtered.mean()
        if apod_window is not None:
            data_dynamic = data_dynamic * apod_window
        data_FF = np.fft.fft2(data_dynamic)
        data_FF = np.fft.fftshift(data_FF)
        data_FF = np.abs(data_FF)
        # Transpose to (n_omega_t, n_omega_tau) for imshow with omega_tau on x, omega_t on y
        data_FF = data_FF.T
        return np.flip(data_FF, axis=0)
    
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
    
    # Plot initial spectrum: omega_tau on x-axis, omega_t on y-axis
    # Extent is [x_min, x_max, y_min, y_max] = [omega_tau_min, omega_tau_max, omega_t_min, omega_t_max]
    im = ax.imshow(spectrum, origin='lower', aspect='auto', cmap='gnuplot2',
                   norm=_get_norm(spectrum, norm_type),
                   extent=[data['omega_tau'][0], data['omega_tau'][-1], data['omega_t'][0], data['omega_t'][-1]])
    ax.set_xlabel('$\\omega_{\\tau}$ (THz)', fontsize=12)
    ax.set_ylabel('$\\omega_t$ (THz)', fontsize=12)
    ax.set_title('Interactive 2DCS Spectrum', fontsize=14)
    
    # Apply window limits: omega_tau on x-axis, omega_t on y-axis
    if omega_tau_window is not None:
        ax.set_xlim(omega_tau_window[0], omega_tau_window[1])
    if omega_t_window is not None:
        ax.set_ylim(omega_t_window[0], omega_t_window[1])
    
    # Add energy level lines (omega_tau on x, omega_t on y)
    energy_lines = []
    if energy_levels_mev:
        for level_name, energy_mev in energy_levels_mev.items():
            if energy_mev is not None and energy_mev != 0:
                omega_thz = energy_mev * MEV_TO_THZ
                color = ENERGY_LINE_COLORS.get(level_name, 'white')
                for omega_val in [omega_thz, -omega_thz]:
                    # axhline for omega_t (y-axis), axvline for omega_tau (x-axis)
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
        # omega_tau on x-axis, omega_t on y-axis
        im_save = ax_save_plot.imshow(spectrum, origin='lower', aspect='auto', cmap='gnuplot2',
                                       norm=_get_norm(spectrum, norm_type),
                                       extent=[data['omega_tau'][0], data['omega_tau'][-1], 
                                               data['omega_t'][0], data['omega_t'][-1]])
        ax_save_plot.set_xlabel('$\\omega_{\\tau}$ (THz)', fontsize=12)
        ax_save_plot.set_ylabel('$\\omega_t$ (THz)', fontsize=12)
        ax_save_plot.set_title(f'λ₂={w_lambda2:.1f}, λ₅={w_lambda5:.1f}, λ₇={w_lambda7:.1f}, x={w_su2_x:.1f}, σ={sigma_scale:.1f}', fontsize=14)
        
        if omega_tau_window is not None:
            ax_save_plot.set_xlim(omega_tau_window[0], omega_tau_window[1])
        if omega_t_window is not None:
            ax_save_plot.set_ylim(omega_t_window[0], omega_t_window[1])
        
        # Add energy level lines (omega_tau on x, omega_t on y)
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
    
    # Interactive mode
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run interactive mode with sliders to tune λ2, λ5, λ7, and SU(2) x weights')
    
    # Load from cache option
    parser.add_argument('--load-results', '--from-cache', action='store_true',
                        help='Skip FFT calculations and load from previously saved .txt files (only regenerate plots)')
    
    # Reload components option
    parser.add_argument('--reload-components', '--from-components', action='store_true',
                        help='Load component FFTs from cache and recompute composite spectra (faster than full calculation)')
    
    args = parser.parse_args()
    
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
    if analysis_type == '2dcs' and (args.e1 or args.e2 or args.kc):
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
                                              reload_components=args.reload_components)
            print(f"Results keys: {list(results.keys())}")
    else:
        print(f"Unknown analysis type: {analysis_type}")
        sys.exit(1)
    
    print("\nAnalysis complete!")
