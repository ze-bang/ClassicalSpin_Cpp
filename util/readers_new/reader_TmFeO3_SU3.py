"""
HDF5 Reader for TmFeO3 Molecular Dynamics Results (SU3-only version)

This module reads simulation outputs from the ClassicalSpin_Cpp package stored in HDF5 format.
Simplified version for SU(3) multipolar moment simulations (no SU(2) spins).

HDF5 File Structure:
=====================

Molecular Dynamics trajectory.h5:
   /metadata/
     - positions [n_sites, 3]
     - Attributes: lattice_size, spin_dim, n_atoms, spin_length,
                   integration_method, dt_initial, T_start, T_end, 
                   save_interval, creation_time, etc.
   /trajectory/
     - times [n_steps]
     - spins [n_steps, n_sites, spin_dim]  (spin_dim=8 for SU(3))
     - magnetization_antiferro [n_steps, spin_dim]
     - magnetization_local [n_steps, spin_dim]
     - magnetization_global [n_steps, spin_dim]
"""

import h5py
import numpy as np
from opt_einsum import contract
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster plotting
import matplotlib.pyplot as plt
import os
import sys
from math import gcd
from functools import reduce, lru_cache
from matplotlib.colors import PowerNorm, LogNorm, SymLogNorm, Normalize
from typing import Dict, Tuple, Optional, List, Any, Literal
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Type for norm selection
NormType = Literal['log', 'power', 'symlog', 'linear']


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
# For SU(3), we use only the first 3 components (dipole moments) for global frame
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
    n_times = S.shape[0]
    spin_dim = S.shape[2]
    
    # Compute phase factors once (k @ P.T)
    ffact = np.exp(1j * np.dot(k, P.T))  # [n_k, n_sites]
    
    # Vectorized computation: (n_times, n_sites, spin_dim) @ (n_sites, n_k) -> (n_times, spin_dim, n_k)
    # Then transpose to (n_times, n_k, spin_dim)
    results = np.einsum('tjs,kj->tks', S, ffact, optimize=True) / np.sqrt(N)
    
    return results


def Spin_global_t(k: np.ndarray, S: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Compute time-dependent spin structure factor in global frame.
    Transforms the first 3 components (dipole moments) from local sublattice frames to global frame,
    and keeps the remaining 5 quadrupole components unchanged.
    
    Args:
        k: k-points array [n_k, 3]
        S: Spin configurations [n_times, n_sites, spin_dim] (spin_dim=8 for SU(3))
        P: Site positions [n_sites, 3]
        
    Returns:
        results: [n_times, n_sublattices, n_k, 8] complex array (all SU(3) components)
    """
    n_sublattices = 4
    size = int(len(P) / n_sublattices)
    n_times = S.shape[0]
    spin_dim = S.shape[2]  # Should be 8 for SU(3)
    tS = np.zeros((n_times, n_sublattices, len(k), spin_dim), dtype=np.complex128)
    
    for i in range(n_sublattices):
        # Compute phase factors once (faster than einsum for matrix multiply)
        ffact = np.exp(1j * np.dot(k, P[i*size:(i+1)*size].T))  # [n_k, size]
        
        # Transform first 3 components (dipole moments) to global frame
        # Use optimized einsum for 3-tensor contraction
        S_sub_dipole = S[:, i*size:(i+1)*size, :3]
        tS[:, i, :, :3] = np.einsum('tjs,kj,sp->tkp', S_sub_dipole, ffact, 
                                     localframe[:, i, :], optimize=True) / np.sqrt(size)
        
        # Keep remaining components (quadrupole moments) unchanged - use vectorized matmul
        if spin_dim > 3:
            S_sub_quad = S[:, i*size:(i+1)*size, 3:]
            tS[:, i, :, 3:] = np.einsum('tjs,kj->tks', S_sub_quad, ffact, optimize=True) / np.sqrt(size)
    
    return tS


def DSSF(w: np.ndarray, k: np.ndarray, S: np.ndarray, P: np.ndarray, 
         T: np.ndarray, global_frame: bool = False) -> np.ndarray:
    """
    Compute dynamical spin structure factor using FFT over time.
    
    For SU(3), the full 8-component structure factor is computed.
    In global frame, the first 3 components (dipole moments) are transformed
    to the global frame, while the remaining 5 (quadrupole) components are kept unchanged.
    
    Args:
        w: Frequency points
        k: k-points array
        S: Spin configurations [n_times, n_sites, spin_dim]
        P: Site positions
        T: Time points
        global_frame: If True, transform dipole components to global frame
        
    Returns:
        DSSF: [n_w, n_k, spin_dim, spin_dim] array (spin_dim=8 for SU(3))
    """
    dt = T[1] - T[0] if len(T) > 1 else 1.0
    fft_freqs = np.fft.fftfreq(len(T), d=dt) * 2 * np.pi
    
    # Vectorized frequency index selection
    indices = np.argmin(np.abs(fft_freqs[:, None] - w[None, :]), axis=0)
    
    if global_frame:
        # Use global frame transformation for all 8 SU(3) components
        A = Spin_global_t(k, S, P)
        # A shape: (n_times, n_sublattices, n_k, 8)
        # Subtract mean and FFT in one step (more cache-friendly)
        A -= A.mean(axis=0, keepdims=True)
        A_fft = np.fft.fft(A, axis=0) / np.sqrt(len(T))
        
        # Select frequencies and compute outer product
        Somega = A_fft[indices]  # [n_w, n_sublattices, n_k, spin_dim]
        # Optimized outer product: S * S^H
        result = np.einsum('wnia,wnib->wiab', Somega, np.conj(Somega), optimize=True)
        return np.real(result)
    else:
        # Use local frame - full 8 components for SU(3)
        A = Spin_t(k, S, P)
        # Subtract mean and FFT
        A -= A.mean(axis=0, keepdims=True)
        A_fft = np.fft.fft(A, axis=0) / np.sqrt(len(T))
        
        # Select frequencies and compute outer product
        Somega = A_fft[indices]  # [n_w, n_k, spin_dim]
        # Optimized outer product
        result = np.einsum('wia,wib->wiab', Somega, np.conj(Somega), optimize=True)
        return np.real(result)


# =============================================================================
# HDF5 READING UTILITIES
# =============================================================================

def _validate_window(window, name):
    """Ensure an omega window is either None or a valid (min, max) tuple."""
    if window is None:
        return None
    if (not isinstance(window, (tuple, list)) or len(window) != 2 or
            window[0] >= window[1]):
        raise ValueError(f"{name} must be a tuple (min, max) with min < max")
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


def read_hdf5_attribute(group, attr_name: str, default=None):
    """Safely read an HDF5 attribute."""
    try:
        val = group.attrs[attr_name]
        # Decode bytes to string if needed
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        return val
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
    Reader for molecular dynamics trajectory HDF5 files (SU3-only version).
    
    File structure:
    /metadata/
      - positions [n_sites, 3]
      - Attributes: lattice_size, spin_dim, n_atoms, spin_length,
                    integration_method, dt_initial, T_start, T_end, etc.
    /trajectory/
      - times [n_steps]
      - spins [n_steps, n_sites, spin_dim]  (spin_dim=8 for SU(3))
      - magnetization_antiferro [n_steps, spin_dim]
      - magnetization_local [n_steps, spin_dim]
      - magnetization_global [n_steps, spin_dim]
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the MD trajectory reader.
        
        Args:
            filepath: Path to the trajectory.h5 file
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
            self.metadata = {
                'integration_method': read_hdf5_attribute(grp, 'integration_method'),
                'dt_initial': read_hdf5_attribute(grp, 'dt_initial'),
                'T_start': read_hdf5_attribute(grp, 'T_start'),
                'T_end': read_hdf5_attribute(grp, 'T_end'),
                'save_interval': read_hdf5_attribute(grp, 'save_interval'),
                'creation_time': read_hdf5_attribute(grp, 'creation_time'),
                'lattice_size': read_hdf5_attribute(grp, 'lattice_size'),
                'spin_dim': read_hdf5_attribute(grp, 'spin_dim', 8),  # Default to 8 for SU(3)
                'n_atoms': read_hdf5_attribute(grp, 'n_atoms'),
                'spin_length': read_hdf5_attribute(grp, 'spin_length'),
                'positions': read_hdf5_dataset(grp, 'positions'),
                'dimensions': read_hdf5_dataset(grp, 'dimensions'),
            }
    
    def get_times(self) -> np.ndarray:
        """Get time points."""
        return self._file['trajectory/times'][:]
    
    def get_spins(self) -> np.ndarray:
        """Get spin trajectory [n_steps, n_sites, spin_dim]."""
        return self._file['trajectory/spins'][:]
    
    def get_magnetization_antiferro(self) -> np.ndarray:
        """Get antiferromagnetic magnetization [n_steps, spin_dim]."""
        return self._file['trajectory/magnetization_antiferro'][:]
    
    def get_magnetization_local(self) -> np.ndarray:
        """Get local magnetization [n_steps, spin_dim]."""
        return self._file['trajectory/magnetization_local'][:]
    
    def get_magnetization_global(self) -> np.ndarray:
        """Get global magnetization [n_steps, spin_dim]."""
        return self._file['trajectory/magnetization_global'][:]
    
    def get_positions(self) -> Optional[np.ndarray]:
        """Get site positions [n_sites, 3]."""
        return self.metadata.get('positions')


class PumpProbeHDF5:
    """
    Reader for pump-probe spectroscopy HDF5 files (SU3-only version).
    
    File structure:
    /metadata/
      - All experimental parameters
    /reference/
      - times [n_times]
      - M_antiferro, M_local, M_global [n_times, spin_dim]
    /tau_scan/
      - tau_values [n_tau]
      - tau_0/, tau_1/, ... each containing M1 and M01 trajectories
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the pump-probe reader.
        
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
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                self.metadata[key] = val
    
    def get_tau_values(self) -> np.ndarray:
        """Get tau delay values."""
        return self._file['tau_scan/tau_values'][:]
    
    def get_reference_times(self) -> np.ndarray:
        """Get time points from reference trajectory."""
        return self._file['reference/times'][:]
    
    def get_reference_magnetization(self, mag_type: str = 'local') -> np.ndarray:
        """
        Get reference magnetization trajectory.
        
        Args:
            mag_type: Type of magnetization ('antiferro', 'local', 'global')
            
        Returns:
            Magnetization array [n_times, spin_dim]
        """
        key = f'M_{mag_type}'
        return self._file[f'reference/{key}'][:]
    
    def get_tau_trajectory(self, tau_index: int, trajectory_type: str = 'M01',
                          mag_type: str = 'local') -> np.ndarray:
        """
        Get magnetization trajectory for a specific tau value.
        
        Args:
            tau_index: Index of the tau value
            trajectory_type: 'M1' or 'M01'
            mag_type: Type of magnetization ('antiferro', 'local', 'global')
            
        Returns:
            Magnetization array [n_times, spin_dim]
        """
        key = f'{trajectory_type}_{mag_type}'
        return self._file[f'tau_scan/tau_{tau_index}/{key}'][:]
    
    def get_tau_value(self, tau_index: int) -> float:
        """Get a specific tau value."""
        return self._file['tau_scan/tau_values'][tau_index]
    
    def get_n_tau_steps(self) -> int:
        """Get the number of tau steps."""
        return len(self._file['tau_scan/tau_values'])


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def read_MD_hdf5(filepath: str, w0: float = 0, wmax: float = 15,
                 output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Read molecular dynamics trajectory from HDF5 and compute DSSF.
    
    Args:
        filepath: Path to trajectory.h5 file
        w0: Minimum frequency
        wmax: Maximum frequency
        output_dir: Directory for output plots (uses filepath directory if None)
        
    Returns:
        Dictionary with DSSF results
    """
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
        if output_dir == '':
            output_dir = '.'
    
    results = {}
    
    with MDTrajectoryHDF5(filepath) as reader:
        reciprocal_lattice = get_reciprocal_lattice_orthorhombic()
        DSSF_K, tick_positions = generate_k_path(reciprocal_lattice)
        g1, g2, g3, g4 = tick_positions
        
        # Get trajectory data
        T = reader.get_times()
        S = reader.get_spins()
        P = reader.get_positions()
        
        if P is None:
            print("Warning: No positions found. Using default positions.")
            P = np.zeros((S.shape[1], 3))
        
        # Compute frequency grid
        dt = T[1] - T[0] if len(T) > 1 else 1.0
        fft_freqs = np.fft.fftfreq(len(T), d=dt) * 2 * np.pi
        w_mask = (fft_freqs >= w0) & (fft_freqs <= wmax)
        w = fft_freqs[w_mask]
        results['w'] = w
        
        # Get spin dimension (should be 8 for SU(3))
        spin_dim = reader.metadata.get('spin_dim', 8)
        if spin_dim is None:
            spin_dim = S.shape[2]
        spin_dim = int(spin_dim)
        
        # Compute DSSF in local frame (all 8 components)
        A_local = DSSF(w, DSSF_K, S, P, T, global_frame=False)
        results['DSSF_local'] = A_local
        
        # Compute DSSF in global frame (dipole components only)
        A_global = DSSF(w, DSSF_K, S, P, T, global_frame=True)
        results['DSSF_global'] = A_global
        
        # Plot individual components - local frame (all 8 SU(3) components)
        _plot_DSSF_components(A_local, w, tick_positions, output_dir, 'local', 
                              spin_dim, w0, wmax)
        
        # Plot individual components - global frame
        global_dim = A_global.shape[2]  # Get dimension from the DSSF result
        _plot_DSSF_components(A_global, w, tick_positions, output_dir, 'global', 
                              global_dim, w0, wmax)
        
        # Compute and save gap at Gamma - local frame
        Gamma_point = np.array([[0, 0, 0]])
        A_Gamma_local = DSSF(w, Gamma_point, S, P, T, global_frame=False)
        DSSF_sum_Gamma_local = contract('wiab->wi', A_Gamma_local)
        _plot_gap_analysis(w, DSSF_sum_Gamma_local, output_dir, 'local')
        
        # Compute and save gap at Gamma - global frame
        A_Gamma_global = DSSF(w, Gamma_point, S, P, T, global_frame=True)
        DSSF_sum_Gamma_global = contract('wiab->wi', A_Gamma_global)
        _plot_gap_analysis(w, DSSF_sum_Gamma_global, output_dir, 'global')
        
        # Save summed DSSF
        DSSF_sum_local = contract('wiab->wi', A_local)
        DSSF_sum_global = contract('wiab->wi', A_global)
        results['DSSF_sum_local'] = DSSF_sum_local
        results['DSSF_sum_global'] = DSSF_sum_global
        
        np.savetxt(os.path.join(output_dir, "DSSF_local.txt"), DSSF_sum_local)
        np.savetxt(os.path.join(output_dir, "DSSF_global.txt"), DSSF_sum_global)
        
        # Plot combined DSSF
        _plot_DSSF_combined(DSSF_sum_local, w, tick_positions, output_dir, 
                            'local', w0, wmax)
        _plot_DSSF_combined(DSSF_sum_global, w, tick_positions, output_dir, 
                            'global', w0, wmax)
    
    return results


def _plot_DSSF_components(A: np.ndarray, w: np.ndarray, tick_positions: List[int],
                          output_dir: str, frame_type: str, n_components: int,
                          w0: float, wmax: float):
    """Plot individual DSSF components."""
    g1, g2, g3, g4 = tick_positions
    labels = [r'$(0,0,0)$', r'$(0,0,1)$', r'$(0,1,1)$', r'$(1,1,1)$']
    
    # Component names for SU(3) - Gell-Mann matrices λ1 through λ8
    component_names = ['λ1', 'λ2', 'λ3', 'λ4', 'λ5', 'λ6', 'λ7', 'λ8']
    
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
        ax.set_ylabel(r'$\omega$')
        if i < len(component_names):
            ax.set_title(f'{component_names[i]} component ({frame_type} frame)')
        fig.colorbar(C)
        plt.savefig(os.path.join(output_dir, f"DSSF_{frame_type}_{i}_{i}.pdf"))
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
    ax.set_ylabel(r'$\omega$')
    ax.set_title(f'Total DSSF ({frame_type} frame)')
    fig.colorbar(C)
    plt.savefig(os.path.join(output_dir, f"DSSF_{frame_type}_sum.pdf"))
    plt.close()


def _plot_gap_analysis(w: np.ndarray, DSSF_Gamma: np.ndarray, output_dir: str, 
                       frame_type: str):
    """Plot DSSF at Gamma point for gap analysis."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(w, DSSF_Gamma[:, 0])
    ax.set_xlim([-10, 20])
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$S(\Gamma, \omega)$')
    ax.set_title(f'Gap analysis at Gamma ({frame_type} frame)')
    plt.savefig(os.path.join(output_dir, f"DSSF_{frame_type}_gap_Gamma.pdf"))
    plt.close()
    np.savetxt(os.path.join(output_dir, f"DSSF_{frame_type}_gap_Gamma.txt"),
              np.column_stack((w, DSSF_Gamma[:, 0])))


def _plot_DSSF_combined(A: np.ndarray, w: np.ndarray, tick_positions: List[int],
                        output_dir: str, frame_type: str, w0: float, wmax: float):
    """Plot combined DSSF."""
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
    ax.set_ylabel(r'$\omega$')
    ax.set_title(f'DSSF ({frame_type} frame)')
    fig.colorbar(C)
    plt.savefig(os.path.join(output_dir, f"DSSF_{frame_type}.pdf"))
    plt.close()


def _plot_first_timesteps(M0_components: np.ndarray, M1_components: np.ndarray, 
                          M01_components: np.ndarray, M_NL_components: np.ndarray,
                          tau_values: np.ndarray, times: np.ndarray, output_dir: str,
                          n_tau_steps: int = 10):
    """Plot the first few time steps for all signal components.
    
    Args:
        M0_components: Reference signal [spin_dim, n_tau, n_times]
        M1_components: First pump signal [spin_dim, n_tau, n_times]
        M01_components: Double pump signal [spin_dim, n_tau, n_times]
        M_NL_components: Nonlinear signal [spin_dim, n_tau, n_times]
        tau_values: Array of tau values
        times: Array of time points
        output_dir: Directory to save plots
        n_tau_steps: Number of tau steps to plot (default: 10)
    """
    spin_dim = M0_components.shape[0]
    component_labels = ['λ1', 'λ2', 'λ3', 'λ4', 'λ5', 'λ6', 'λ7', 'λ8']
    
    # Determine which tau indices to plot (evenly spaced)
    n_tau = len(tau_values)
    tau_indices = np.linspace(0, n_tau - 1, min(n_tau_steps, n_tau), dtype=int)
    
    print(f"  Plotting first {len(tau_indices)} tau steps...")
    
    # Create a figure for each component showing all signals
    for comp in range(spin_dim):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{component_labels[comp]}-component: First {len(tau_indices)} tau steps', 
                     fontsize=14, fontweight='bold')
        
        signal_data = [
            (M0_components[comp], 'M0', axes[0, 0]),
            (M1_components[comp], 'M1', axes[0, 1]),
            (M01_components[comp], 'M01', axes[1, 0]),
            (M_NL_components[comp], 'M_NL', axes[1, 1])
        ]
        
        for data, label, ax in signal_data:
            for tau_idx in tau_indices:
                tau_val = tau_values[tau_idx]
                ax.plot(times, data[tau_idx, :], 
                       label=f'τ={tau_val:.3f}', alpha=0.7, linewidth=1.5)
            
            ax.set_xlabel('Time', fontsize=11)
            ax.set_ylabel(f'${label}_{{{component_labels[comp]}}}$', fontsize=11)
            ax.set_title(f'{label} signal', fontsize=12)
            ax.legend(fontsize=8, ncol=2, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"first_timesteps_{component_labels[comp]}.pdf"), 
                   dpi=100)
        plt.close(fig)
    
    # Create summary plot with all components for M_NL only
    n_rows = min(spin_dim, 4)  # Limit to 4 rows for readability
    n_cols = 2
    fig_summary, axes_summary = plt.subplots(n_rows, n_cols, 
                                            figsize=(14, 3*n_rows))
    fig_summary.suptitle(f'M_NL: First {len(tau_indices)} tau steps (all components)', 
                        fontsize=14, fontweight='bold')
    
    for comp in range(min(spin_dim, n_rows * n_cols)):
        row = comp // n_cols
        col = comp % n_cols
        ax = axes_summary[row, col] if n_rows > 1 else axes_summary[col]
        
        for tau_idx in tau_indices:
            tau_val = tau_values[tau_idx]
            ax.plot(times, M_NL_components[comp][tau_idx, :], 
                   label=f'τ={tau_val:.3f}', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel(f'$M_{{NL,{component_labels[comp]}}}$', fontsize=10)
        ax.set_title(f'{component_labels[comp]}-component', fontsize=11)
        ax.legend(fontsize=7, ncol=2, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "first_timesteps_M_NL_summary.pdf"), dpi=100)
    plt.close(fig_summary)
    
    print(f"  Saved first timestep plots to {output_dir}")


# =============================================================================
# 2D COHERENT SPECTROSCOPY ANALYSIS
# =============================================================================

def read_2D_nonlinear(dir: str, omega_t_window: Optional[Tuple[float, float]] = None, 
                      omega_tau_window: Optional[Tuple[float, float]] = None,
                      norm_type: NormType = 'log', gamma: float = 0.5,
                      demod_omega_t: Optional[float] = None,
                      apodization_gamma: float = 0.03,
                      pulse_window_sigma: float = 5.0,
                      window_type: str = 'gaussian') -> Dict[str, np.ndarray]:
    """Read and compute 2D nonlinear spectroscopy using FFT for SU(3) systems.
    
    Computes the nonlinear response M_NL = M01 - M0 - M1 and performs 2D FFT
    to obtain the frequency-domain spectrum.
    
    Supports HDF5 format (pump_probe_spectroscopy.h5).
    
    Args:
        dir: Directory containing pump-probe data
        omega_t_window: Optional (min, max) tuple for ω_t axis limits
        omega_tau_window: Optional (min, max) tuple for ω_τ axis limits
        norm_type: Normalization type for colormap ('log', 'power', 'symlog', 'linear')
        gamma: Exponent for PowerNorm (default: 0.5)
        demod_omega_t: If provided, demodulate (rotate) along detection time t by
            multiplying data by exp(+i * demod_omega_t * t) before FFT.
        apodization_gamma: Decay parameter for apodization window. For Gaussian: sets
            boundary decay level. For exponential: decay rate. For Tukey: taper fraction.
            Default: 0.03
        pulse_window_sigma: Number of pulse widths (sigma) after probe center (t=0) 
            to skip before FFT. Default: 5.0
        window_type: Type of apodization window to reduce spectral leakage:
            'gaussian', 'hann', 'hamming', 'blackman', 'tukey', 'cosine', 'exponential', 'none'
            Default: 'gaussian'
        
    Returns:
        Dictionary with 2DCS results including M_NL_FF and frequency arrays
    """
    omega_t_window = _validate_window(omega_t_window, "omega_t_window")
    omega_tau_window = _validate_window(omega_tau_window, "omega_tau_window")
    hdf5_path = os.path.join(dir, "pump_probe_spectroscopy.h5")
    
    # Component labels for SU(3) - Gell-Mann matrices λ1 through λ8
    component_labels = ['λ1', 'λ2', 'λ3', 'λ4', 'λ5', 'λ6', 'λ7', 'λ8']
    
    results = {}
    
    pulse_freq: Optional[float] = None

    # Read from HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        times = f['/reference/times'][:]
        M0_global = f['/reference/M_global'][:]
        tau_values = f['/tau_scan/tau_values'][:]
        tau_step = len(tau_values)
        
        spin_dim = M0_global.shape[1]  # Should be 8 for SU(3)
        
        # Print metadata
        print(f"  Loaded HDF5: {hdf5_path}")
        print(f"    Time steps: {len(times)}, t_range: [{times[0]:.4f}, {times[-1]:.4f}]")
        print(f"    Tau values: {tau_step}, tau_range: [{tau_values[0]:.4f}, {tau_values[-1]:.4f}]")
        print(f"    M_global shape: {M0_global.shape}, spin_dim: {spin_dim}")
        
        # Print pulse and lattice metadata if available
        pulse_width = 1.0  # Default fallback
        if '/metadata' in f:
            metadata_grp = f['/metadata']
            print(f"  Pulse parameters:")
            if 'pulse_amp' in metadata_grp.attrs:
                print(f"    Amplitude: {metadata_grp.attrs['pulse_amp']}")
            if 'pulse_width' in metadata_grp.attrs:
                pulse_width = float(metadata_grp.attrs['pulse_width'])
                print(f"    Width: {pulse_width}")
            if 'pulse_freq' in metadata_grp.attrs:
                pulse_freq = float(metadata_grp.attrs['pulse_freq'])
                print(f"    Frequency: {pulse_freq}")
            print(f"  Lattice parameters:")
            if 'n_atoms' in metadata_grp.attrs:
                print(f"    n_atoms: {metadata_grp.attrs['n_atoms']}")
            if 'spin_dim' in metadata_grp.attrs:
                print(f"    spin_dim: {metadata_grp.attrs['spin_dim']}")
            if 'lattice_size' in metadata_grp.attrs:
                print(f"    lattice_size: {metadata_grp.attrs['lattice_size']}")
        
        # Process all spin_dim components for M_NL, M0, M1, M01
        length = len(M0_global[:, 0])
        M_NL_components = np.zeros((spin_dim, tau_step, length))
        M0_components = np.zeros((spin_dim, tau_step, length))
        M1_components = np.zeros((spin_dim, tau_step, length))
        M01_components = np.zeros((spin_dim, tau_step, length))
        
        for i, tau_val in enumerate(tau_values):
            tau_group = f[f'/tau_scan/tau_{i}']
            M1_global = tau_group['M1_global'][:]
            M01_global = tau_group['M01_global'][:]
            
            for comp in range(spin_dim):
                M0 = M0_global[:, comp]
                M1 = M1_global[:, comp]
                M01 = M01_global[:, comp]
                
                min_len = min(len(M0), len(M1), len(M01), length)
                M_NL_components[comp, i, :min_len] = M01[:min_len] - M0[:min_len] - M1[:min_len]
                M0_components[comp, i, :min_len] = M0[:min_len]
                M1_components[comp, i, :min_len] = M1[:min_len]
                M01_components[comp, i, :min_len] = M01[:min_len]
        
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        tau = tau_values

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
    
    # Slice the data arrays to exclude probe pulse region
    M_NL_components = M_NL_components[:, :, t0_idx:]
    M0_components = M0_components[:, :, t0_idx:]
    M1_components = M1_components[:, :, t0_idx:]
    M01_components = M01_components[:, :, t0_idx:]
    
    # Update times for downstream use
    times = times_positive
    
    # Compute omega arrays (using filtered time array)
    omega_tau = np.fft.fftfreq(int(len(tau)), tau[1] - tau[0] if len(tau) > 1 else 1.0) * 2 * np.pi
    omega_tau = np.fft.fftshift(omega_tau)
    omega_t = np.fft.fftfreq(M_NL_components.shape[2], dt) * 2 * np.pi
    omega_t = np.fft.fftshift(omega_t)
    
    results['omega_tau'] = omega_tau
    results['omega_t'] = omega_t
    results['tau_values'] = tau_values
    if pulse_freq is not None:
        results['pulse_freq'] = pulse_freq
    
    # Pre-compute FFT for all components (cache for reuse)
    print("  Pre-computing FFTs for all components...")
    M_NL_FF_cache = np.zeros((spin_dim, len(omega_tau), len(omega_t)), dtype=np.float64)
    M0_FF_cache = np.zeros_like(M_NL_FF_cache)
    M1_FF_cache = np.zeros_like(M_NL_FF_cache)
    M01_FF_cache = np.zeros_like(M_NL_FF_cache)
    
    # Helper function for 2D FFT analysis (optimized)
    t_phase: Optional[np.ndarray] = None
    if demod_omega_t is not None:
        t_phase = np.exp(1j * demod_omega_t * times)

    # Build apodization window based on window_type
    # For t: use relative time from start of window so decay starts at 1.0
    # For tau: use distance from tau=0 (probe time) so max weight is at tau closest to 0
    n_t = len(times)
    n_tau = len(tau)
    
    if window_type.lower() == 'none' or apodization_gamma <= 0:
        apod_window = None
        print(f"  No apodization applied (window_type='{window_type}')")
    elif window_type.lower() == 'gaussian':
        # Gaussian window: exp(-0.5 * (t/sigma)^2)
        t_relative = times - times[0]
        t_range = t_relative[-1] if len(t_relative) > 1 else 1.0
        tau_range = max(np.abs(tau[0]), np.abs(tau[-1])) if len(tau) > 1 else 1.0
        
        decay_factor = np.sqrt(-2.0 * np.log(apodization_gamma))
        sigma_t = t_range / decay_factor
        sigma_tau = tau_range / decay_factor
        
        decay_t = np.exp(-0.5 * (t_relative / sigma_t) ** 2)
        decay_tau = np.exp(-0.5 * (np.abs(tau) / sigma_tau) ** 2)
        apod_window = np.outer(decay_tau, decay_t)
        print(f"  Applying Gaussian apodization (σ_t={sigma_t:.2f}, σ_τ={sigma_tau:.2f})")
        print(f"    t window: 1.0 at t={times[0]:.1f} → {decay_t[-1]:.4f} at t={times[-1]:.1f}")
        print(f"    τ window: {decay_tau[0]:.4f} at τ={tau[0]:.1f} → {decay_tau[-1]:.4f} at τ={tau[-1]:.1f}")
    elif window_type.lower() == 'exponential':
        t_relative = times - times[0]
        decay_t = np.exp(-apodization_gamma * t_relative)
        decay_tau = np.exp(-apodization_gamma * np.abs(tau))
        apod_window = np.outer(decay_tau, decay_t)
        print(f"  Applying exponential apodization (γ={apodization_gamma})")
        print(f"    t decay: 1.0 at t={times[0]:.1f} → {decay_t[-1]:.4f} at t={times[-1]:.1f}")
        print(f"    τ decay: {decay_tau[0]:.4f} at τ={tau[0]:.1f} → {decay_tau[-1]:.4f} at τ={tau[-1]:.1f}")
    elif window_type.lower() == 'hann':
        window_t = 0.5 * (1 + np.cos(np.pi * np.arange(n_t) / (n_t - 1)))
        window_tau = 0.5 * (1 + np.cos(np.pi * np.abs(tau) / np.max(np.abs(tau))))
        apod_window = np.outer(window_tau, window_t)
        print(f"  Applying Hann apodization")
        print(f"    t window: {window_t[0]:.4f} at t={times[0]:.1f} → {window_t[-1]:.4f} at t={times[-1]:.1f}")
        print(f"    τ window: {window_tau[0]:.4f} at τ={tau[0]:.1f} → {window_tau[-1]:.4f} at τ={tau[-1]:.1f}")
    elif window_type.lower() == 'hamming':
        window_t = 0.54 + 0.46 * np.cos(np.pi * np.arange(n_t) / (n_t - 1))
        window_tau = 0.54 + 0.46 * np.cos(np.pi * np.abs(tau) / np.max(np.abs(tau)))
        apod_window = np.outer(window_tau, window_t)
        print(f"  Applying Hamming apodization")
        print(f"    t window: {window_t[0]:.4f} at t={times[0]:.1f} → {window_t[-1]:.4f} at t={times[-1]:.1f}")
        print(f"    τ window: {window_tau[0]:.4f} at τ={tau[0]:.1f} → {window_tau[-1]:.4f} at τ={tau[-1]:.1f}")
    elif window_type.lower() == 'blackman':
        n_norm_t = np.arange(n_t) / (n_t - 1)
        n_norm_tau = np.abs(tau) / np.max(np.abs(tau))
        window_t = 0.42 + 0.5 * np.cos(np.pi * n_norm_t) + 0.08 * np.cos(2 * np.pi * n_norm_t)
        window_tau = 0.42 + 0.5 * np.cos(np.pi * n_norm_tau) + 0.08 * np.cos(2 * np.pi * n_norm_tau)
        apod_window = np.outer(window_tau, window_t)
        print(f"  Applying Blackman apodization")
        print(f"    t window: {window_t[0]:.4f} at t={times[0]:.1f} → {window_t[-1]:.4f} at t={times[-1]:.1f}")
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
        print(f"    t window: {window_t[0]:.4f} at t={times[0]:.1f} → {window_t[-1]:.4f} at t={times[-1]:.1f}")
        print(f"    τ window: {window_tau[0]:.4f} at τ={tau[0]:.1f} → {window_tau[-1]:.4f} at τ={tau[-1]:.1f}")
    elif window_type.lower() == 'cosine':
        window_t = np.cos(0.5 * np.pi * np.arange(n_t) / (n_t - 1))
        window_tau = np.cos(0.5 * np.pi * np.abs(tau) / np.max(np.abs(tau)))
        apod_window = np.outer(window_tau, window_t)
        print(f"  Applying Cosine apodization")
        print(f"    t window: {window_t[0]:.4f} at t={times[0]:.1f} → {window_t[-1]:.4f} at t={times[-1]:.1f}")
        print(f"    τ window: {window_tau[0]:.4f} at τ={tau[0]:.1f} → {window_tau[-1]:.4f} at τ={tau[-1]:.1f}")
    else:
        print(f"  Warning: Unknown window_type '{window_type}', using no apodization")
        print(f"    Available: gaussian, exponential, hann, hamming, blackman, tukey, cosine, none")
        apod_window = None

    def compute_2d_fft(data):
        """Compute 2D FFT with proper shifting, flipping, and apodization (only t >= 0 data)."""
        data_dynamic = data - data.mean()  # In-place mean subtraction equivalent
        # Apply apodization window to reduce spectral leakage
        if apod_window is not None:
            data_dynamic = data_dynamic * apod_window
        if t_phase is not None:
            data_dynamic = data_dynamic * t_phase[None, :]
        data_FF = np.fft.fft2(data_dynamic)
        data_FF = np.fft.fftshift(data_FF)
        data_FF = np.abs(data_FF)
        return np.flip(data_FF, axis=1)  # Flip omega_t axis
    
    # Compute all FFTs once
    for comp in range(spin_dim):
        M_NL_FF_cache[comp] = compute_2d_fft(M_NL_components[comp])
        M0_FF_cache[comp] = compute_2d_fft(M0_components[comp])
        M1_FF_cache[comp] = compute_2d_fft(M1_components[comp])
        M01_FF_cache[comp] = compute_2d_fft(M01_components[comp])
    
    # =========================================================================
    # Plot first few time steps
    # =========================================================================
    print("  Plotting first time steps...")
    _plot_first_timesteps(M0_components, M1_components, M01_components, M_NL_components,
                         tau_values, times, dir, n_tau_steps=10)
    
    # =========================================================================
    # Analysis for M0, M1, M01 (individual signals)
    # =========================================================================
    signal_names = ['M0', 'M1', 'M01']
    signal_data = [M0_components, M1_components, M01_components]
    
    # Map signal names to cached FFTs
    fft_cache_map = {'M0': M0_FF_cache, 'M1': M1_FF_cache, 'M01': M01_FF_cache}
    
    for sig_name, sig_components in zip(signal_names, signal_data):
        print(f"  Processing {sig_name}...")
        sig_fft_cache = fft_cache_map[sig_name]
        
        # Create debug plot for this signal (spin_dim components x 3 plot types)
        n_rows = min(spin_dim, 8)  # Limit rows for readability
        fig_sig, axes_sig = plt.subplots(n_rows, 3, figsize=(15, 3*n_rows))
        
        for comp in range(n_rows):
            sig_comp = sig_components[comp]
            
            # Time domain plot
            ax_time = axes_sig[comp, 0]
            tau_stride = max(1, len(tau) // 5)
            for tau_idx in range(0, len(tau), tau_stride):
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
            
            # Frequency domain plot (use cached FFT)
            sig_comp_FF = sig_fft_cache[comp]
            
            ax_freq = axes_sig[comp, 2]
            im_freq = ax_freq.imshow(sig_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                      norm=_get_norm(sig_comp_FF, norm_type, gamma), 
                                      extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]])
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
        plt.savefig(dir + f"/{sig_name}_components_debug.pdf", dpi=100)
        plt.close(fig_sig)  # Explicitly close figure
        
        # Save total FFT (use cached values - just sum)
        sig_total_FF = sig_fft_cache.sum(axis=0)
        np.savetxt(dir + f"/{sig_name}_FF.txt", sig_total_FF)
        results[f'{sig_name}_FF'] = sig_total_FF
        
        # Main spectrum plot (sum of components)
        fig_spec = plt.figure(figsize=(10, 8))
        plt.imshow(sig_total_FF, origin='lower',
                   extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
                   aspect='auto', cmap='gnuplot2', norm=_get_norm(sig_total_FF, norm_type, gamma))
        plt.xlabel('$\\omega_t$ (rad/time)')
        plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
        plt.colorbar(label='Intensity')
        plt.title(f'{sig_name} Spectrum (SU(3) total)')
        if omega_t_window is not None:
            plt.xlim(omega_t_window)
        if omega_tau_window is not None:
            plt.ylim(omega_tau_window)
        plt.savefig(dir + f"/{sig_name}_SPEC.pdf", dpi=100)
        plt.close(fig_spec)
    
    # =========================================================================
    # M_NL analysis (M01 - M0 - M1)
    # =========================================================================
    
    # Debug plots: Plot M_NL for each component in time domain and frequency domain
    print("  Generating M_NL debug plots...")
    n_rows = min(spin_dim, 8)
    fig_debug, axes_debug = plt.subplots(n_rows, 3, figsize=(15, 3*n_rows))
    tau_stride = max(1, len(tau) // 5)
    
    for comp in range(n_rows):
        M_NL_comp = M_NL_components[comp]
        
        # Time domain plot (M_NL vs t for different tau)
        ax_time = axes_debug[comp, 0]
        for tau_idx in range(0, len(tau), tau_stride):
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
        
        # Frequency domain plot (use cached FFT)
        M_NL_comp_FF = M_NL_FF_cache[comp]
        
        ax_freq = axes_debug[comp, 2]
        im_freq = ax_freq.imshow(M_NL_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                  norm=_get_norm(M_NL_comp_FF, norm_type, gamma), 
                                  extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]])
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
    plt.savefig(dir + "/M_NL_components_debug.pdf", dpi=100)
    plt.close(fig_debug)
    
    # Compute total M_NL (use cached FFTs - just sum)
    M_NL_total_FF = M_NL_FF_cache.sum(axis=0)
    
    np.savetxt(dir + "/M_NL_FF.txt", M_NL_total_FF)
    results['M_NL_FF'] = M_NL_total_FF
    
    # Full spectrum plot (total)
    fig_nl = plt.figure(figsize=(10, 8))
    plt.imshow(M_NL_total_FF, origin='lower',
               extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
               aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_total_FF, norm_type, gamma))
    plt.xlabel('$\\omega_t$ (rad/time)')
    plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
    plt.colorbar(label='Intensity')
    plt.title('$M_{NL}$ Spectrum (SU(3) total)')
    if omega_t_window is not None:
        plt.xlim(omega_t_window)
    if omega_tau_window is not None:
        plt.ylim(omega_tau_window)
    plt.savefig(dir + "/M_NLSPEC.pdf", dpi=100)
    plt.close(fig_nl)
    
    # =====================================================
    # Lambda5 + Lambda7 mode (magnetic dipole observable)
    # λ5 = index 4, λ7 = index 6 in Gell-Mann basis
    # Weighted sum: 2.3915*λ5 + 0.9128*λ7
    # =====================================================
    # Combine λ5 and λ7 components in TIME domain before FFT (weighted)
    LAMBDA5_WEIGHT = 2.3915
    LAMBDA7_WEIGHT = 0.9128
    M_NL_lambda57 = LAMBDA5_WEIGHT * M_NL_components[4] + LAMBDA7_WEIGHT * M_NL_components[6]
    
    # Debug plot for λ5 + λ7 time domain
    fig_lambda57_debug = plt.figure(figsize=(14, 5))
    ax1 = fig_lambda57_debug.add_subplot(131)
    ax1.imshow(M_NL_components[4].T, origin='lower', aspect='auto', cmap='RdBu_r')
    ax1.set_title(f'$\\lambda_5$ (×{LAMBDA5_WEIGHT})')
    ax1.set_xlabel('$\\tau$ index')
    ax1.set_ylabel('$t$ index')
    
    ax2 = fig_lambda57_debug.add_subplot(132)
    ax2.imshow(M_NL_components[6].T, origin='lower', aspect='auto', cmap='RdBu_r')
    ax2.set_title(f'$\\lambda_7$ (×{LAMBDA7_WEIGHT})')
    ax2.set_xlabel('$\\tau$ index')
    ax2.set_ylabel('$t$ index')
    
    ax3 = fig_lambda57_debug.add_subplot(133)
    ax3.imshow(M_NL_lambda57.T, origin='lower', aspect='auto', cmap='RdBu_r')
    ax3.set_title(f'{LAMBDA5_WEIGHT}$\\lambda_5$ + {LAMBDA7_WEIGHT}$\\lambda_7$')
    ax3.set_xlabel('$\\tau$ index')
    ax3.set_ylabel('$t$ index')
    
    plt.tight_layout()
    plt.savefig(dir + "/M_NL_lambda57_debug.pdf", dpi=100)
    plt.close(fig_lambda57_debug)
    
    # Apply apodization and FFT to λ5 + λ7 (use the same compute_2d_fft function)
    M_NL_lambda57_FF = compute_2d_fft(M_NL_lambda57)
    
    np.savetxt(dir + "/M_NL_lambda57_FF.txt", M_NL_lambda57_FF)
    results['M_NL_lambda57_FF'] = M_NL_lambda57_FF
    
    # Spectrum plot for λ5 + λ7
    fig_lambda57 = plt.figure(figsize=(10, 8))
    plt.imshow(M_NL_lambda57_FF, origin='lower',
               extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
               aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_lambda57_FF, norm_type, gamma))
    plt.xlabel('$\\omega_t$ (rad/time)')
    plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
    plt.colorbar(label='Intensity')
    plt.title(f'$M_{{NL}}$ Spectrum ({LAMBDA5_WEIGHT}$\\lambda_5$ + {LAMBDA7_WEIGHT}$\\lambda_7$)')
    if omega_t_window is not None:
        plt.xlim(omega_t_window)
    if omega_tau_window is not None:
        plt.ylim(omega_tau_window)
    plt.savefig(dir + "/M_NL_lambda57_SPEC.pdf", dpi=100)
    plt.close(fig_lambda57)
    
    return results


def _process_subdir_parallel(args):
    """Helper function for parallel processing of subdirectories."""
    subdir, filename, omega_t_window, omega_tau_window, norm_type, gamma, apodization_gamma, pulse_window_sigma = args
    try:
        print(f"Processing: {filename}")
        sub_results = read_2D_nonlinear(subdir,
                                         omega_t_window=omega_t_window,
                                         omega_tau_window=omega_tau_window,
                                         norm_type=norm_type,
                                         gamma=gamma,
                                         apodization_gamma=apodization_gamma,
                                         pulse_window_sigma=pulse_window_sigma)
        M_NL_data = np.loadtxt(os.path.join(subdir, "M_NL_FF.txt"))
        # Also load λ5 + λ7 data
        M_NL_lambda57_path = os.path.join(subdir, "M_NL_lambda57_FF.txt")
        M_NL_lambda57_data = np.loadtxt(M_NL_lambda57_path) if os.path.exists(M_NL_lambda57_path) else None
        return (M_NL_data, M_NL_lambda57_data, sub_results.get('omega_tau'), sub_results.get('omega_t'), filename)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def read_2D_nonlinear_tot(dir: str, omega_t_window: Optional[Tuple[float, float]] = None, 
                          omega_tau_window: Optional[Tuple[float, float]] = None,
                          norm_type: NormType = 'log', gamma: float = 0.5,
                          n_jobs: Optional[int] = None,
                          apodization_gamma: float = 0.03,
                          pulse_window_sigma: float = 5.0) -> Dict[str, np.ndarray]:
    """Aggregate 2D nonlinear spectroscopy from multiple pump-probe runs.
    
    Processes all subdirectories containing pump-probe data in parallel, aggregates
    results, and generates combined 2D FFT spectrum.
    
    Args:
        dir: Parent directory containing subdirectories with pump-probe data
        omega_t_window: Optional (min, max) tuple for ω_t axis limits
        omega_tau_window: Optional (min, max) tuple for ω_τ axis limits
        norm_type: Normalization type for colormap ('log', 'power', 'symlog', 'linear')
        gamma: Exponent for PowerNorm (default: 0.5)
        n_jobs: Number of parallel jobs (default: use all cores)
        apodization_gamma: Exponential decay rate for apodization to reduce spectral
            leakage from truncated oscillations. Applied as exp(-apodization_gamma * |t|).
            Set to 0.0 to disable. Default: 0.03
        pulse_window_sigma: Number of pulse widths (sigma) after probe center (t=0) 
            to skip before FFT. Default: 5.0 (captures >99.99% of Gaussian pulse)
    
    Returns:
        Dictionary with aggregated 2DCS results
        
    Outputs:
        - M_NL_tot.txt: Aggregated nonlinear magnetization in frequency space
        - NLSPEC_tot.pdf: Combined 2D nonlinear spectrum
    """
    omega_t_window = _validate_window(omega_t_window, "omega_t_window")
    omega_tau_window = _validate_window(omega_tau_window, "omega_tau_window")
    
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    # Collect all subdirectories
    subdirs_to_process = []
    for filename in sorted(os.listdir(dir)):
        subdir = os.path.join(dir, filename)
        if os.path.isdir(subdir):
            hdf5_path = os.path.join(subdir, "pump_probe_spectroscopy.h5")
            if os.path.exists(hdf5_path):
                subdirs_to_process.append((subdir, filename, omega_t_window, 
                                          omega_tau_window, norm_type, gamma, 
                                          apodization_gamma, pulse_window_sigma))
    
    if not subdirs_to_process:
        print("No valid subdirectories found")
        return {}
    
    print(f"Processing {len(subdirs_to_process)} subdirectories using {n_jobs} workers...")
    
    A = None
    A_lambda57 = None
    count = 0
    results = {}
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_subdir = {executor.submit(_process_subdir_parallel, args): args[1] 
                           for args in subdirs_to_process}
        
        for future in as_completed(future_to_subdir):
            result = future.result()
            if result is not None:
                M_NL_data, M_NL_lambda57_data, omega_tau, omega_t, filename = result
                
                if A is None:
                    A = M_NL_data.copy()
                    if M_NL_lambda57_data is not None:
                        A_lambda57 = M_NL_lambda57_data.copy()
                    results['omega_tau'] = omega_tau
                    results['omega_t'] = omega_t
                else:
                    min_shape = (min(A.shape[0], M_NL_data.shape[0]),
                                min(A.shape[1], M_NL_data.shape[1]))
                    A[:min_shape[0], :min_shape[1]] += M_NL_data[:min_shape[0], :min_shape[1]]
                    if M_NL_lambda57_data is not None and A_lambda57 is not None:
                        min_shape_57 = (min(A_lambda57.shape[0], M_NL_lambda57_data.shape[0]),
                                       min(A_lambda57.shape[1], M_NL_lambda57_data.shape[1]))
                        A_lambda57[:min_shape_57[0], :min_shape_57[1]] += M_NL_lambda57_data[:min_shape_57[0], :min_shape_57[1]]
                count += 1
    
    if A is not None and count > 0:
        A = A / count  # Average over all runs
        np.savetxt(os.path.join(dir, "M_NL_tot.txt"), A)
        results['M_NL_FF_tot'] = A
        
        # Average λ5 + λ7 if available
        if A_lambda57 is not None:
            A_lambda57 = A_lambda57 / count
            np.savetxt(os.path.join(dir, "M_NL_lambda57_tot.txt"), A_lambda57)
            results['M_NL_lambda57_FF_tot'] = A_lambda57
        
        # Get frequency arrays for plotting
        omega_tau = results.get('omega_tau')
        omega_t = results.get('omega_t')
        
        if omega_tau is not None and omega_t is not None:
            extent = [omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]]
        else:
            extent = None
        
        # Plot aggregated result
        plt.figure(figsize=(10, 8))
        plt.imshow(A, origin='lower', aspect='auto', cmap='gnuplot2', norm=_get_norm(A, norm_type, gamma),
                   extent=extent)
        plt.xlabel('$\\omega_t$ (rad/time)')
        plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
        plt.colorbar(label='Intensity')
        plt.title(f'Aggregated $M_{{NL}}$ Spectrum (SU(3), {count} runs)')
        if omega_t_window is not None:
            plt.xlim(omega_t_window)
        if omega_tau_window is not None:
            plt.ylim(omega_tau_window)
        plt.savefig(os.path.join(dir, "NLSPEC_tot.pdf"))
        plt.clf()
        plt.close()
        print(f"Aggregated {count} runs, saved to {dir}/NLSPEC_tot.pdf")
        
        # Plot aggregated λ5 + λ7 if available
        if A_lambda57 is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(A_lambda57, origin='lower', aspect='auto', cmap='gnuplot2', 
                       norm=_get_norm(A_lambda57, norm_type, gamma), extent=extent)
            plt.xlabel('$\\omega_t$ (rad/time)')
            plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
            plt.colorbar(label='Intensity')
            plt.title(f'Aggregated $\\lambda_5 + \\lambda_7$ Spectrum ({count} runs)')
            if omega_t_window is not None:
                plt.xlim(omega_t_window)
            if omega_tau_window is not None:
                plt.ylim(omega_tau_window)
            plt.savefig(os.path.join(dir, "NLSPEC_lambda57_tot.pdf"))
            plt.clf()
            plt.close()
            print(f"Aggregated λ5+λ7 {count} runs, saved to {dir}/NLSPEC_lambda57_tot.pdf")
    else:
        print("No valid data found")
    
    return results


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
        if 'metadata' in f:
            grp = f['metadata']
            for key in grp.attrs.keys():
                val = grp.attrs[key]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                metadata[key] = val
    
    return metadata


def plot_magnetization_trajectory(filepath: str, output_dir: Optional[str] = None):
    """
    Plot magnetization trajectories from the HDF5 file.
    
    For SU(3), plots both dipole (Jx, Jy, Jz) and quadrupole components.
    
    Args:
        filepath: Path to trajectory.h5 file
        output_dir: Directory for output plots
    """
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
        if output_dir == '':
            output_dir = '.'
    
    with MDTrajectoryHDF5(filepath) as reader:
        T = reader.get_times()
        M_antiferro = reader.get_magnetization_antiferro()
        M_local = reader.get_magnetization_local()
        M_global = reader.get_magnetization_global()
        
        spin_dim = M_antiferro.shape[1] if len(M_antiferro.shape) > 1 else 1
        
        # Component labels for SU(3) - Gell-Mann matrices
        labels_full = ['λ1', 'λ2', 'λ3', 'λ4', 'λ5', 'λ6', 'λ7', 'λ8']
        labels = labels_full[:spin_dim]
        
        # Plot antiferromagnetic magnetization
        fig, axes = plt.subplots(spin_dim, 1, figsize=(12, 2*spin_dim), sharex=True)
        if spin_dim == 1:
            axes = [axes]
        for i in range(spin_dim):
            axes[i].plot(T, M_antiferro[:, i], label=f'$M_{{AF,{labels[i]}}}$')
            axes[i].set_ylabel(f'$M_{{AF,{labels[i]}}}$')
            axes[i].legend()
        axes[-1].set_xlabel('Time')
        fig.suptitle('Antiferromagnetic Magnetization')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'M_antiferro.pdf'))
        plt.close()
        
        # Plot local magnetization
        fig, axes = plt.subplots(spin_dim, 1, figsize=(12, 2*spin_dim), sharex=True)
        if spin_dim == 1:
            axes = [axes]
        for i in range(spin_dim):
            axes[i].plot(T, M_local[:, i], label=f'$M_{{local,{labels[i]}}}$')
            axes[i].set_ylabel(f'$M_{{local,{labels[i]}}}$')
            axes[i].legend()
        axes[-1].set_xlabel('Time')
        fig.suptitle('Local Magnetization')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'M_local.pdf'))
        plt.close()
        
        # Plot global magnetization
        fig, axes = plt.subplots(spin_dim, 1, figsize=(12, 2*spin_dim), sharex=True)
        if spin_dim == 1:
            axes = [axes]
        for i in range(spin_dim):
            axes[i].plot(T, M_global[:, i], label=f'$M_{{global,{labels[i]}}}$')
            axes[i].set_ylabel(f'$M_{{global,{labels[i]}}}$')
            axes[i].legend()
        axes[-1].set_xlabel('Time')
        fig.suptitle('Global Magnetization')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'M_global.pdf'))
        plt.close()
        
        # Save data
        header = 'time ' + ' '.join(labels)
        np.savetxt(os.path.join(output_dir, 'M_antiferro.txt'), 
                   np.column_stack((T, M_antiferro)), 
                   header=header)
        np.savetxt(os.path.join(output_dir, 'M_local.txt'), 
                   np.column_stack((T, M_local)), 
                   header=header)
        np.savetxt(os.path.join(output_dir, 'M_global.txt'), 
                   np.column_stack((T, M_global)), 
                   header=header)


# =============================================================================
# MAIN
# =============================================================================

def find_trajectory_file(path: str) -> str:
    """
    Find the trajectory.h5 file given a path.
    
    If path is an HDF5 file, return it directly.
    If path is a directory, search for trajectory.h5 inside it.
    
    Args:
        path: Path to HDF5 file or directory
        
    Returns:
        Path to the trajectory.h5 file
    """
    if os.path.isfile(path) and path.endswith('.h5'):
        return path
    
    if os.path.isdir(path):
        # Check for trajectory.h5 directly in the directory
        direct_path = os.path.join(path, 'trajectory.h5')
        if os.path.exists(direct_path):
            return direct_path
        
        # Check for sample_0/trajectory.h5
        sample_path = os.path.join(path, 'sample_0', 'trajectory.h5')
        if os.path.exists(sample_path):
            return sample_path
        
        # Search recursively for any trajectory.h5
        for root, dirs, files in os.walk(path):
            if 'trajectory.h5' in files:
                return os.path.join(root, 'trajectory.h5')
        
        raise FileNotFoundError(f"No trajectory.h5 file found in {path}")
    
    raise FileNotFoundError(f"Path does not exist or is not a valid HDF5 file: {path}")


def parse_omega_window(arg: str) -> Optional[Tuple[float, float]]:
    """Parse omega window argument in format 'min,max' or 'None'."""
    if arg.lower() == 'none':
        return None
    try:
        parts = arg.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid omega window format: {arg}. Expected 'min,max'")
        return (float(parts[0]), float(parts[1]))
    except ValueError as e:
        raise ValueError(f"Invalid omega window format: {arg}. Expected 'min,max' (e.g., '-0.5,0.5')")


def parse_norm_arg(arg: str) -> Tuple[NormType, float]:
    """Parse norm argument in format 'type' or 'type:gamma'.
    
    Examples:
        'log' -> ('log', 0.5)
        'power' -> ('power', 0.5)
        'power:0.3' -> ('power', 0.3)
        'symlog' -> ('symlog', 0.5)
        'linear' -> ('linear', 0.5)
    """
    valid_norms = ['log', 'power', 'symlog', 'linear']
    
    if ':' in arg:
        parts = arg.split(':')
        norm_type = parts[0].lower()
        try:
            gamma = float(parts[1])
        except ValueError:
            raise ValueError(f"Invalid gamma value: {parts[1]}. Expected a number.")
    else:
        norm_type = arg.lower()
        gamma = 0.5  # default gamma
    
    if norm_type not in valid_norms:
        raise ValueError(f"Invalid norm type: {norm_type}. Valid options: {valid_norms}")
    
    return norm_type, gamma


def _parse_kv_args(args: List[str]) -> Dict[str, str]:
    """Parse extra CLI args in key=value form."""
    kv: Dict[str, str] = {}
    for a in args:
        if '=' not in a:
            continue
        k, v = a.split('=', 1)
        k = k.strip().lower()
        v = v.strip()
        if k:
            kv[k] = v
    return kv


def _parse_demod_arg(val: str) -> Optional[float]:
    """Parse demod arg. Accepts float string or 'none'."""
    if val.lower() == 'none':
        return None
    return float(val)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reader_TmFeO3_SU3.py <hdf5_file_or_directory> [analysis_type] [omega_t_window] [omega_tau_window] [norm_type] [window_type]")
        print("  analysis_type: 'md' for molecular dynamics (default)")
        print("                 'mag' for magnetization plots")
        print("                 'pp' for 2D nonlinear spectroscopy (pump-probe)")
        print("  omega_t_window: 'min,max' or 'None' for no windowing (default: None)")
        print("  omega_tau_window: 'min,max' or 'None' for no windowing (default: None)")
        print("  norm_type: 'log', 'power', 'power:gamma', 'symlog', or 'linear' (default: 'log')")
        print("             For power norm, optionally specify gamma as 'power:0.3' (default: 0.5)")
        print("  window_type: Apodization window to reduce spectral leakage (default: 'gaussian')")
        print("               'gaussian', 'hann', 'hamming', 'blackman', 'tukey', 'cosine', 'exponential', 'none'")
        print("  Optional extra args (key=value, after window_type):")
        print("    demod=<omega>: demodulate along t by exp(+i*omega*t) before FFT (e.g. demod=0.5)")
        print("                 Use demod=none to disable (default)")
        print("\nExamples:")
        print("  python reader_TmFeO3_SU3.py ./TmFeO3_tm_md/sample_0/trajectory.h5 md")
        print("  python reader_TmFeO3_SU3.py ./TmFeO3_tm_md/ md")
        print("  python reader_TmFeO3_SU3.py ./TmFeO3_2DCS/ pp")
        print("  python reader_TmFeO3_SU3.py ./TmFeO3_2DCS/ pp -0.5,0.5 -0.5,0.5")
        print("  python reader_TmFeO3_SU3.py ./TmFeO3_2DCS/ pp -2,2 -2,2 power blackman")
        print("  python reader_TmFeO3_SU3.py ./TmFeO3_2DCS/ pp -2,2 -2,2 power:0.3 hann")
        print("  python reader_TmFeO3_SU3.py ./TmFeO3_2DCS/ pp -2,2 -2,2 power gaussian demod=0.5")
        print("  python reader_TmFeO3_SU3.py ./TmFeO3_2DCS/ pp None None symlog tukey")
        sys.exit(1)
    
    input_path = sys.argv[1]
    analysis_type = sys.argv[2] if len(sys.argv) > 2 else 'md'
    
    # Parse omega windows from command line arguments (default to None for no windowing)
    omega_t_window = None
    omega_tau_window = None
    norm_type = 'log'
    gamma = 0.5
    window_type = 'gaussian'
    
    if len(sys.argv) > 3:
        omega_t_window = parse_omega_window(sys.argv[3])
    if len(sys.argv) > 4:
        omega_tau_window = parse_omega_window(sys.argv[4])
    if len(sys.argv) > 5:
        norm_type, gamma = parse_norm_arg(sys.argv[5])
    if len(sys.argv) > 6 and '=' not in sys.argv[6]:
        window_type = sys.argv[6]

    extra_kv = _parse_kv_args([arg for arg in sys.argv[6:] if '=' in arg]) if len(sys.argv) > 6 else {}
    demod_omega_t: Optional[float] = None
    if 'demod' in extra_kv:
        # demod=<float> or demod=none
        demod_omega_t = _parse_demod_arg(extra_kv['demod'])
    
    if not os.path.exists(input_path):
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
    
    if analysis_type == 'pp':
        # Pump-probe / 2D nonlinear spectroscopy analysis
        print(f"Running 2D nonlinear spectroscopy analysis on: {input_path}")
        print(f"  omega_t_window: {omega_t_window}")
        print(f"  omega_tau_window: {omega_tau_window}")
        print(f"  norm_type: {norm_type}" + (f" (gamma={gamma})" if norm_type == 'power' else ""))
        if demod_omega_t is not None:
            print(f"  demod_omega_t: {demod_omega_t}")
        print("=" * 50)
        
        if os.path.isdir(input_path):
            # Check if this directory contains subdirectories with pump-probe data
            has_subdirs = False
            for item in os.listdir(input_path):
                subdir = os.path.join(input_path, item)
                if os.path.isdir(subdir):
                    if os.path.exists(os.path.join(subdir, "pump_probe_spectroscopy.h5")):
                        has_subdirs = True
                        break
            
            if has_subdirs:
                # Aggregate from multiple subdirectories
                print("Found subdirectories with pump-probe data. Running aggregation...")
                results = read_2D_nonlinear_tot(input_path, 
                                                 omega_t_window=omega_t_window,
                                                 omega_tau_window=omega_tau_window,
                                                 norm_type=norm_type,
                                                 gamma=gamma)
            elif os.path.exists(os.path.join(input_path, "pump_probe_spectroscopy.h5")):
                # Single directory with pump-probe data
                print("Running single directory analysis...")
                results = read_2D_nonlinear(input_path,
                                            omega_t_window=omega_t_window,
                                            omega_tau_window=omega_tau_window,
                                            norm_type=norm_type,
                                            gamma=gamma,
                                            demod_omega_t=demod_omega_t,
                                            window_type=window_type)
            else:
                print(f"Error: No pump_probe_spectroscopy.h5 found in {input_path}")
                sys.exit(1)
        else:
            print(f"Error: For 'pp' analysis, please provide a directory path")
            sys.exit(1)
    else:
        # MD or magnetization analysis - need trajectory file
        try:
            filepath = find_trajectory_file(input_path)
            print(f"Found trajectory file: {filepath}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        
        print(f"Analyzing {filepath} (type: {analysis_type})")
        print("\nHDF5 Structure:")
        print("-" * 50)
        print_hdf5_structure(filepath)
        print("-" * 50)
        
        if analysis_type == 'md':
            print("\nRunning MD analysis (DSSF computation for SU(3))...")
            results = read_MD_hdf5(filepath)
            print(f"Results keys: {list(results.keys())}")
        elif analysis_type == 'mag':
            print("\nPlotting magnetization trajectories (SU(3) components)...")
            plot_magnetization_trajectory(filepath)
        else:
            print(f"Unknown analysis type: {analysis_type}")
            print("Available: md, mag, pp")
            sys.exit(1)
    
    print("\nAnalysis complete!")
