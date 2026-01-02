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


def read_2D_nonlinear(dir: str, omega_t_window: Optional[Tuple[float, float]] = None,
                      omega_tau_window: Optional[Tuple[float, float]] = None,
                      norm_type: NormType = 'power') -> Dict[str, np.ndarray]:
    """Read and compute 2D nonlinear spectroscopy using FFT for mixed SU(2)+SU(3) systems.
    
    Reads pump-probe spectroscopy data from HDF5 file and computes the nonlinear
    response M_NL = M01 - M0 - M1, then performs 2D FFT to get the 2D spectrum.
    
    Processes both SU(2) and SU(3) sublattices separately.
    
    Args:
        dir: Directory containing pump_probe_spectroscopy.h5
        omega_t_window: Optional (min, max) tuple for ω_t axis limits
        omega_tau_window: Optional (min, max) tuple for ω_τ axis limits
        norm_type: Normalization type for plots ('log', 'power', 'symlog', 'linear')
        
    Returns:
        Dictionary with 2DCS results for both sublattices
    """
    omega_t_window = _validate_window(omega_t_window, "omega_t_window")
    omega_tau_window = _validate_window(omega_tau_window, "omega_tau_window")
    hdf5_path = os.path.join(dir, "pump_probe_spectroscopy.h5")
    
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
        
        # Compute omega arrays
        omega_tau = np.fft.fftfreq(int(len(tau)), tau[1] - tau[0] if len(tau) > 1 else 1.0) * 2 * np.pi
        omega_tau = np.fft.fftshift(omega_tau)
        omega_t = np.fft.fftfreq(len(times), dt) * 2 * np.pi
        omega_t = np.fft.fftshift(omega_t)
        
        results['omega_tau'] = omega_tau
        results['omega_t'] = omega_t
        results['tau_values'] = tau_values
        
        # Helper function for 2D FFT analysis (optimized)
        def compute_2d_fft(data):
            """Compute 2D FFT with proper shifting and flipping."""
            data_dynamic = data - data.mean()  # In-place equivalent
            data_FF = np.fft.fft2(data_dynamic)
            data_FF = np.fft.fftshift(data_FF)
            data_FF = np.abs(data_FF)
            return np.flip(data_FF, axis=1)  # Flip omega_t axis
        
        # =====================================================================
        # Process SU(2) sublattice
        # =====================================================================
        if '/reference/M_local_SU2' in f:
            print("\n  Processing SU(2) sublattice...")
            M0_SU2 = f['/reference/M_local_SU2'][:]
            spin_dim_SU2 = M0_SU2.shape[1]
            length = len(M0_SU2[:, 0])
            
            # Initialize arrays for all components
            M_NL_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
            M0_comp_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
            M1_comp_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
            M01_comp_SU2 = np.zeros((spin_dim_SU2, tau_step, length))
            
            for i, tau_val in enumerate(tau_values):
                tau_group = f[f'/tau_scan/tau_{i}']
                M1_SU2 = tau_group['M1_local_SU2'][:]
                M01_SU2 = tau_group['M01_local_SU2'][:]
                
                for comp in range(spin_dim_SU2):
                    M0 = M0_SU2[:, comp]
                    M1 = M1_SU2[:, comp]
                    M01 = M01_SU2[:, comp]
                    
                    min_len = min(len(M0), len(M1), len(M01), length)
                    M_NL_SU2[comp, i, :min_len] = M01[:min_len] - M0[:min_len] - M1[:min_len]
                    M0_comp_SU2[comp, i, :min_len] = M0[:min_len]
                    M1_comp_SU2[comp, i, :min_len] = M1[:min_len]
                    M01_comp_SU2[comp, i, :min_len] = M01[:min_len]
            
            # Process individual signals (M0, M1, M01)
            for sig_name, sig_components in [('M0_SU2', M0_comp_SU2), ('M1_SU2', M1_comp_SU2), ('M01_SU2', M01_comp_SU2)]:
                print(f"    Processing {sig_name}...")
                
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
                    
                    # Frequency domain plot
                    sig_comp_FF = compute_2d_fft(sig_comp)
                    ax_freq = axes_sig[comp, 2]
                    im_freq = ax_freq.imshow(sig_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                              norm=_get_norm(sig_comp_FF, norm_type), extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]])
                    ax_freq.set_xlabel('$\\omega_t$ (rad/time)')
                    ax_freq.set_ylabel('$\\omega_{\\tau}$ (rad/time)')
                    ax_freq.set_title(f'{label}-component (freq domain)')
                    if omega_t_window is not None:
                        ax_freq.set_xlim(omega_t_window)
                    if omega_tau_window is not None:
                        ax_freq.set_ylim(omega_tau_window)
                    plt.colorbar(im_freq, ax=ax_freq)
                    
                    np.savetxt(os.path.join(dir, f"{sig_name}_FF_{label}.txt"), sig_comp_FF)
                
                plt.tight_layout()
                plt.savefig(os.path.join(dir, f"{sig_name}_components_debug.pdf"))
                plt.clf()
                plt.close()
                
                # Main spectrum plot (z-component for SU2)
                sig_z = sig_components[2] if spin_dim_SU2 > 2 else sig_components[0]
                sig_z_FF = compute_2d_fft(sig_z)
                np.savetxt(os.path.join(dir, f"{sig_name}_FF.txt"), sig_z_FF)
                results[f'{sig_name}_FF'] = sig_z_FF
                
                fig_spec = plt.figure(figsize=(10, 8))
                plt.imshow(sig_z_FF, origin='lower',
                           extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
                           aspect='auto', cmap='gnuplot2', norm=_get_norm(sig_z_FF, norm_type))
                plt.xlabel('$\\omega_t$ (rad/time)')
                plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
                plt.colorbar(label='Intensity')
                plt.title(f'{sig_name} Spectrum')
                if omega_t_window is not None:
                    plt.xlim(omega_t_window)
                if omega_tau_window is not None:
                    plt.ylim(omega_tau_window)
                plt.savefig(os.path.join(dir, f"{sig_name}_SPEC.pdf"), dpi=100)
                plt.close(fig_spec)
            
            # M_NL analysis for SU2
            print("    Processing M_NL_SU2...")
            n_rows = min(spin_dim_SU2, 3)
            fig_debug, axes_debug = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes_debug = axes_debug.reshape(1, -1)
            
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
                
                # Frequency domain plot
                M_NL_comp_FF = compute_2d_fft(M_NL_comp)
                ax_freq = axes_debug[comp, 2]
                im_freq = ax_freq.imshow(M_NL_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                          norm=_get_norm(M_NL_comp_FF, norm_type), extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]])
                ax_freq.set_xlabel('$\\omega_t$ (rad/time)')
                ax_freq.set_ylabel('$\\omega_{\\tau}$ (rad/time)')
                ax_freq.set_title(f'{label}-component (freq domain)')
                if omega_t_window is not None:
                    ax_freq.set_xlim(omega_t_window)
                if omega_tau_window is not None:
                    ax_freq.set_ylim(omega_tau_window)
                plt.colorbar(im_freq, ax=ax_freq)
                
                np.savetxt(os.path.join(dir, f"M_NL_SU2_FF_{label}.txt"), M_NL_comp_FF)
            
            plt.tight_layout()
            plt.savefig(os.path.join(dir, "M_NL_SU2_components_debug.pdf"), dpi=100)
            plt.close(fig_debug)
            
            # Main M_NL spectrum for SU2 (z-component)
            M_NL_z = M_NL_SU2[2] if spin_dim_SU2 > 2 else M_NL_SU2[0]
            M_NL_z_FF = compute_2d_fft(M_NL_z)
            np.savetxt(os.path.join(dir, "M_NL_SU2_FF.txt"), M_NL_z_FF)
            results['M_NL_SU2_FF'] = M_NL_z_FF
            
            fig_nl = plt.figure(figsize=(10, 8))
            plt.imshow(M_NL_z_FF, origin='lower',
                       extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
                       aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_z_FF, norm_type))
            plt.xlabel('$\\omega_t$ (rad/time)')
            plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
            plt.colorbar(label='Intensity')
            plt.title('$M_{NL}$ Spectrum (SU(2))')
            if omega_t_window is not None:
                plt.xlim(omega_t_window)
            if omega_tau_window is not None:
                plt.ylim(omega_tau_window)
            plt.savefig(os.path.join(dir, "M_NL_SU2_SPEC.pdf"), dpi=100)
            plt.close(fig_nl)
        
        # =====================================================================
        # Process SU(3) sublattice
        # =====================================================================
        if '/reference/M_local_SU3' in f:
            print("\n  Processing SU(3) sublattice...")
            M0_SU3 = f['/reference/M_local_SU3'][:]
            spin_dim_SU3 = M0_SU3.shape[1]
            length = len(M0_SU3[:, 0])
            
            # Initialize arrays for all components
            M_NL_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
            M0_comp_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
            M1_comp_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
            M01_comp_SU3 = np.zeros((spin_dim_SU3, tau_step, length))
            
            for i, tau_val in enumerate(tau_values):
                tau_group = f[f'/tau_scan/tau_{i}']
                M1_SU3 = tau_group['M1_local_SU3'][:]
                M01_SU3 = tau_group['M01_local_SU3'][:]
                
                for comp in range(spin_dim_SU3):
                    M0 = M0_SU3[:, comp]
                    M1 = M1_SU3[:, comp]
                    M01 = M01_SU3[:, comp]
                    
                    min_len = min(len(M0), len(M1), len(M01), length)
                    M_NL_SU3[comp, i, :min_len] = M01[:min_len] - M0[:min_len] - M1[:min_len]
                    M0_comp_SU3[comp, i, :min_len] = M0[:min_len]
                    M1_comp_SU3[comp, i, :min_len] = M1[:min_len]
                    M01_comp_SU3[comp, i, :min_len] = M01[:min_len]
            
            # Process individual signals (M0, M1, M01)
            for sig_name, sig_components in [('M0_SU3', M0_comp_SU3), ('M1_SU3', M1_comp_SU3), ('M01_SU3', M01_comp_SU3)]:
                print(f"    Processing {sig_name}...")
                
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
                    
                    # Frequency domain plot
                    sig_comp_FF = compute_2d_fft(sig_comp)
                    ax_freq = axes_sig[comp, 2]
                    im_freq = ax_freq.imshow(sig_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                              norm=_get_norm(sig_comp_FF, norm_type), extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]])
                    ax_freq.set_xlabel('$\\omega_t$ (rad/time)')
                    ax_freq.set_ylabel('$\\omega_{\\tau}$ (rad/time)')
                    ax_freq.set_title(f'{label}-component (freq domain)')
                    if omega_t_window is not None:
                        ax_freq.set_xlim(omega_t_window)
                    if omega_tau_window is not None:
                        ax_freq.set_ylim(omega_tau_window)
                    plt.colorbar(im_freq, ax=ax_freq)
                    
                    np.savetxt(os.path.join(dir, f"{sig_name}_FF_{label}.txt"), sig_comp_FF)
                
                plt.tight_layout()
                plt.savefig(os.path.join(dir, f"{sig_name}_components_debug.pdf"), dpi=100)
                plt.close(fig_sig)  # Explicitly close figure
                
                # Total spectrum (sum over components - cache FFTs)
                sig_total_FF = np.zeros_like(compute_2d_fft(sig_components[0]))
                for comp in range(spin_dim_SU3):
                    sig_total_FF += compute_2d_fft(sig_components[comp])
                np.savetxt(os.path.join(dir, f"{sig_name}_FF.txt"), sig_total_FF)
                results[f'{sig_name}_FF'] = sig_total_FF
                
                fig_spec = plt.figure(figsize=(10, 8))
                plt.imshow(sig_total_FF, origin='lower',
                           extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
                           aspect='auto', cmap='gnuplot2', norm=_get_norm(sig_total_FF, norm_type))
                plt.xlabel('$\\omega_t$ (rad/time)')
                plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
                plt.colorbar(label='Intensity')
                plt.title(f'{sig_name} Spectrum (total)')
                if omega_t_window is not None:
                    plt.xlim(omega_t_window)
                if omega_tau_window is not None:
                    plt.ylim(omega_tau_window)
                plt.savefig(os.path.join(dir, f"{sig_name}_SPEC.pdf"), dpi=100)
                plt.close(fig_spec)
            
            # M_NL analysis for SU3
            print("    Processing M_NL_SU3...")
            n_rows = min(spin_dim_SU3, 8)
            fig_debug, axes_debug = plt.subplots(n_rows, 3, figsize=(15, 3*n_rows))
            if n_rows == 1:
                axes_debug = axes_debug.reshape(1, -1)
            
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
                
                # Frequency domain plot
                M_NL_comp_FF = compute_2d_fft(M_NL_comp)
                ax_freq = axes_debug[comp, 2]
                im_freq = ax_freq.imshow(M_NL_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                          norm=_get_norm(M_NL_comp_FF, norm_type), extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]])
                ax_freq.set_xlabel('$\\omega_t$ (rad/time)')
                ax_freq.set_ylabel('$\\omega_{\\tau}$ (rad/time)')
                ax_freq.set_title(f'{label}-component (freq domain)')
                if omega_t_window is not None:
                    ax_freq.set_xlim(omega_t_window)
                if omega_tau_window is not None:
                    ax_freq.set_ylim(omega_tau_window)
                plt.colorbar(im_freq, ax=ax_freq)
                
                np.savetxt(os.path.join(dir, f"M_NL_SU3_FF_{label}.txt"), M_NL_comp_FF)
            
            plt.tight_layout()
            plt.savefig(os.path.join(dir, "M_NL_SU3_components_debug.pdf"), dpi=100)
            plt.close(fig_debug)
            
            # Total M_NL spectrum for SU3 (sum over components)
            M_NL_total_FF = np.zeros_like(compute_2d_fft(M_NL_SU3[0]))
            for comp in range(spin_dim_SU3):
                M_NL_total_FF += compute_2d_fft(M_NL_SU3[comp])
            np.savetxt(os.path.join(dir, "M_NL_SU3_FF.txt"), M_NL_total_FF)
            results['M_NL_SU3_FF'] = M_NL_total_FF
            
            fig_nl = plt.figure(figsize=(10, 8))
            plt.imshow(M_NL_total_FF, origin='lower',
                       extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
                       aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_total_FF, norm_type))
            plt.xlabel('$\\omega_t$ (rad/time)')
            plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
            plt.colorbar(label='Intensity')
            plt.title('$M_{NL}$ Spectrum (SU(3) total)')
            if omega_t_window is not None:
                plt.xlim(omega_t_window)
            if omega_tau_window is not None:
                plt.ylim(omega_tau_window)
            plt.savefig(os.path.join(dir, "M_NL_SU3_SPEC.pdf"), dpi=100)
            plt.close(fig_nl)
        
        # =====================================================================
        # Combined SU(2) + SU(3) spectrum
        # =====================================================================
        if 'M_NL_SU2_FF' in results and 'M_NL_SU3_FF' in results:
            print("\n  Computing combined SU(2)+SU(3) spectrum...")
            M_NL_combined = results['M_NL_SU2_FF'] + results['M_NL_SU3_FF']
            np.savetxt(os.path.join(dir, "M_NL_combined_FF.txt"), M_NL_combined)
            results['M_NL_combined_FF'] = M_NL_combined
            
            fig_comb = plt.figure(figsize=(10, 8))
            plt.imshow(M_NL_combined, origin='lower',
                       extent=[omega_t[0], omega_t[-1], omega_tau[0], omega_tau[-1]],
                       aspect='auto', cmap='gnuplot2', norm=_get_norm(M_NL_combined, norm_type))
            plt.xlabel('$\\omega_t$ (rad/time)')
            plt.ylabel('$\\omega_{\\tau}$ (rad/time)')
            plt.colorbar(label='Intensity')
            plt.title('$M_{NL}$ Spectrum (SU(2) + SU(3))')
            if omega_t_window is not None:
                plt.xlim(omega_t_window)
            if omega_tau_window is not None:
                plt.ylim(omega_tau_window)
            plt.savefig(os.path.join(dir, "M_NL_combined_SPEC.pdf"), dpi=100)
            plt.close(fig_comb)
    
    print(f"\n  2D nonlinear spectroscopy analysis complete.")
    print(f"  Output files saved to: {dir}")
    
    return results


def read_2DCS_combined_hdf5(filepath: str, omega_t_window: Optional[Tuple[float, float]] = None,
                            omega_tau_window: Optional[Tuple[float, float]] = None,
                            norm_type: NormType = 'power') -> Dict[str, np.ndarray]:
    """
    Read 2D coherent spectroscopy data and compute nonlinear spectra for both sublattices.
    
    Wrapper that finds the appropriate directory and calls read_2D_nonlinear.
    
    Args:
        filepath: Path to pump_probe_spectroscopy.h5 file
        omega_t_window: Optional (min, max) tuple for ω_t axis limits  
        omega_tau_window: Optional (min, max) tuple for ω_τ axis limits
        norm_type: Normalization type for plots ('log', 'power', 'symlog', 'linear')
        
    Returns:
        Dictionary with 2DCS results for SU(2), SU(3), and combined
    """
    output_dir = os.path.dirname(filepath)
    if output_dir == '':
        output_dir = '.'
    
    return read_2D_nonlinear(output_dir, omega_t_window, omega_tau_window, norm_type)


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
    parser.add_argument('--w0', type=float, default=0,
                        help='Minimum frequency for MD analysis (default: 0)')
    parser.add_argument('--wmax', type=float, default=70,
                        help='Maximum frequency for MD analysis (default: 70)')
    
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
    if args.omega_t:
        print(f"  ω_t window: {args.omega_t}")
    if args.omega_tau:
        print(f"  ω_τ window: {args.omega_tau}")
    
    if analysis_type == 'md':
        print(f"\nRunning MD analysis (w0={args.w0}, wmax={args.wmax})...")
        results = read_MD_hdf5(filepath, w0=args.w0, wmax=args.wmax)
        print(f"Results keys: {list(results.keys())}")
    elif analysis_type == '2dcs':
        print("\nRunning 2DCS analysis...")
        omega_t_window = tuple(args.omega_t) if args.omega_t else None
        omega_tau_window = tuple(args.omega_tau) if args.omega_tau else None
        results = read_2DCS_combined_hdf5(filepath, omega_t_window, omega_tau_window, args.norm)
        print(f"Results keys: {list(results.keys())}")
    else:
        print(f"Unknown analysis type: {analysis_type}")
        sys.exit(1)
    
    print("\nAnalysis complete!")
