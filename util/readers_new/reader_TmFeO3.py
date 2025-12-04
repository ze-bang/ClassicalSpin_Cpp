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
import matplotlib.pyplot as plt
import os
from math import gcd
from functools import reduce
from matplotlib.colors import PowerNorm
from typing import Dict, Tuple, Optional, List, Any


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
    n_times = S.shape[0]
    spin_dim = S.shape[2]
    results = np.zeros((n_times, len(k), spin_dim), dtype=np.complex128)
    ffact = np.exp(1j * contract('ik,jk->ij', k, P))
    
    for i in range(n_times):
        results[i] = contract('js, ij->is', S[i], ffact) / np.sqrt(N)
    
    return results


def Spin_global_t(k: np.ndarray, S: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Compute time-dependent spin structure factor in global frame.
    Transforms spins from local sublattice frames to global frame.
    
    Args:
        k: k-points array [n_k, 3]
        S: Spin configurations [n_times, n_sites, spin_dim]
        P: Site positions [n_sites, 3]
        
    Returns:
        results: [n_times, n_sublattices, n_k, 3] complex array
    """
    n_sublattices = 4
    size = int(len(P) / n_sublattices)
    n_times = S.shape[0]
    tS = np.zeros((n_times, n_sublattices, len(k), 3), dtype=np.complex128)
    
    for i in range(n_sublattices):
        ffact = np.exp(1j * contract('ik,jk->ij', k, P[i*size:(i+1)*size]))
        tS[:, i, :, :] = contract('tjs, ij, sp->tip', S[:, i*size:(i+1)*size, :3], 
                                   ffact, localframe[:, i, :]) / np.sqrt(size)
    
    return tS


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
        # A shape: (n_times, n_sublattices, n_k, 3)
        # Subtract mean configuration before FFT
        A_mean = np.mean(A, axis=0, keepdims=True)
        A = A - A_mean
        # FFT over time dimension
        dt = T[1] - T[0] if len(T) > 1 else 1.0
        A_fft = np.fft.fft(A, axis=0)
        fft_freqs = np.fft.fftfreq(len(T), d=dt) * 2 * np.pi
        
        # Select closest frequencies to w
        indices = [np.argmin(np.abs(fft_freqs - w_val)) for w_val in w]
        Somega = A_fft[indices] / np.sqrt(len(T))
        
        read = np.real(contract('wnia, wnib->wiab', Somega, np.conj(Somega)))
        return read
    else:
        # Use local frame (original implementation)
        A = Spin_t(k, S, P)
        
        # Subtract mean configuration before FFT
        A_mean = np.mean(A, axis=0, keepdims=True)
        A = A - A_mean
        
        # FFT over time dimension
        dt = T[1] - T[0] if len(T) > 1 else 1.0
        A_fft = np.fft.fft(A, axis=0)
        fft_freqs = np.fft.fftfreq(len(T), d=dt) * 2 * np.pi
        
        # Select closest frequencies to w
        indices = [np.argmin(np.abs(fft_freqs - w_val)) for w_val in w]
        Somega = A_fft[indices] / np.sqrt(len(T))
        
        read = np.real(contract('wia, wib->wiab', Somega, np.conj(Somega)))
        return read


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


def read_2DCS_hdf5(filepath: str, mag_type: str = 'global', 
                   omega_range: float = 3.0, freq_step: float = 0.1,
                   output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Read 2D coherent spectroscopy data from HDF5 and compute nonlinear response.
    
    The nonlinear response is computed as:
        M_NL(tau, t) = M01(tau, t) - M0(t) - M1(tau, t)
    
    Then Fourier transformed over both tau and t to get the 2D spectrum.
    
    Args:
        filepath: Path to pump_probe_spectroscopy.h5 file
        mag_type: Type of magnetization to use ('antiferro', 'local', 'global')
        omega_range: Maximum frequency range
        freq_step: Frequency step size
        output_dir: Directory for output plots
        
    Returns:
        Dictionary with 2DCS results
    """
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
    
    results = {}
    
    with PumpProbeHDF5(filepath) as reader:
        # Get tau values
        tau_values = reader.get_tau_values()
        n_tau = len(tau_values)
        
        # Get reference times
        times = reader.get_reference_times()
        n_times = len(times)
        
        # Frequency grids
        w_pump = np.arange(0, omega_range, freq_step)  # Pump frequency (tau)
        w_probe = np.arange(-omega_range, omega_range, freq_step)  # Probe frequency (t)
        
        # Process SU2
        for sublattice in ['SU2', 'SU3']:
            spin_dim = reader.metadata.get(f'spin_dim_{sublattice}', 3 if sublattice == 'SU2' else 8)
            if spin_dim is None:
                continue
            spin_dim = int(spin_dim)
            
            try:
                # Get reference (M0) trajectory
                M0 = reader.get_reference_magnetization(mag_type, sublattice)
                
                # Initialize nonlinear spectrum
                M_NL_FF = np.zeros((len(w_pump), len(w_probe), spin_dim), dtype=complex)
                
                # Compute M0 in frequency domain
                M0_phase = np.exp(1j * np.outer(w_probe, times))
                M0_w = contract('ta, wt->wa', M0, M0_phase)
                
                # Process each tau value
                for i_tau in range(n_tau):
                    try:
                        current_tau = reader.get_tau_value(i_tau)
                        
                        # Get M1 and M01 trajectories
                        M1 = reader.get_tau_trajectory(i_tau, 'M1', mag_type, sublattice)
                        M01 = reader.get_tau_trajectory(i_tau, 'M01', mag_type, sublattice)
                        
                        # Transform to frequency domain
                        M1_phase = np.exp(1j * np.outer(w_probe, times))
                        M01_phase = np.exp(1j * np.outer(w_probe, times))
                        
                        M1_w = contract('ta, wt->wa', M1, M1_phase)
                        M01_w = contract('ta, wt->wa', M01, M01_phase)
                        
                        # Compute nonlinear response
                        M_NL_here = M01_w - M0_w - M1_w
                        
                        # Apply tau phase factor
                        ffactau = np.exp(-1j * w_pump * current_tau) / n_tau
                        M_NL_FF += contract('wa, e->ewa', M_NL_here, ffactau)
                        
                    except Exception as e:
                        print(f"Error processing tau_{i_tau}: {e}")
                        continue
                
                # Store results
                M_NL_FF_abs = np.abs(M_NL_FF)
                results[f'M_NL_FF_{sublattice}'] = M_NL_FF_abs
                
                # Plot and save
                _plot_2DCS(M_NL_FF_abs, w_pump, w_probe, output_dir, sublattice, 
                          mag_type, omega_range, spin_dim)
                
            except Exception as e:
                print(f"Error processing {sublattice}: {e}")
                continue
        
        results['w_pump'] = w_pump
        results['w_probe'] = w_probe
        results['tau_values'] = tau_values
    
    return results


def _plot_2DCS(M_NL_FF: np.ndarray, w_pump: np.ndarray, w_probe: np.ndarray,
               output_dir: str, sublattice: str, mag_type: str, 
               omega_range: float, spin_dim: int):
    """Plot 2D coherent spectroscopy results."""
    real_range = omega_range * 4.92 / 4.14  # Convert to THz
    extent = [0, real_range, -real_range, real_range]
    
    for i in range(spin_dim):
        M_NL_here = M_NL_FF[:, :, i]
        
        # Save raw data
        np.savetxt(os.path.join(output_dir, f"M_NL_FF_{sublattice}_{i}.txt"), M_NL_here)
        
        # Linear scale plot
        plt.figure(figsize=(10, 8))
        plt.imshow(M_NL_here.T, origin='lower', extent=extent,
                  aspect='auto', interpolation='lanczos', cmap='gnuplot2')
        plt.colorbar(label='Amplitude')
        plt.xlabel(r'$\omega_{\tau}$ (THz)')
        plt.ylabel(r'$\omega_{t}$ (THz)')
        plt.title(f'2D Nonlinear Spectrum ({sublattice}, component {i})')
        plt.savefig(os.path.join(output_dir, f"NLSPEC_{sublattice}_{i}_{mag_type}.pdf"),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log scale plot
        plt.figure(figsize=(10, 8))
        plt.imshow(M_NL_here.T, origin='lower', extent=extent,
                  aspect='auto', interpolation='lanczos', cmap='gnuplot2',
                  norm=PowerNorm(gamma=0.5))
        plt.colorbar(label='Amplitude (sqrt scale)')
        plt.xlabel(r'$\omega_{\tau}$ (THz)')
        plt.ylabel(r'$\omega_{t}$ (THz)')
        plt.title(f'2D Nonlinear Spectrum ({sublattice}, component {i})')
        plt.savefig(os.path.join(output_dir, f"NLSPEC_{sublattice}_{i}_{mag_type}_log.pdf"),
                   dpi=300, bbox_inches='tight')
        plt.close()


def read_2DCS_combined_hdf5(filepath: str, omega_range: float = 3.0,
                            freq_step: float = 0.1,
                            output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Read 2D coherent spectroscopy data and combine SU(2) and SU(3) contributions.
    
    Args:
        filepath: Path to pump_probe_spectroscopy.h5 file
        omega_range: Maximum frequency range
        freq_step: Frequency step size
        output_dir: Directory for output plots
        
    Returns:
        Dictionary with combined 2DCS results
    """
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
    
    # Get individual results
    results_global = read_2DCS_hdf5(filepath, 'global', omega_range, freq_step, output_dir)
    results_local = read_2DCS_hdf5(filepath, 'local', omega_range, freq_step, output_dir)
    
    # Combine results
    combined = {}
    combined['w_pump'] = results_global.get('w_pump')
    combined['w_probe'] = results_global.get('w_probe')
    combined['tau_values'] = results_global.get('tau_values')
    
    # Sum SU2 and SU3 contributions for global
    for key in ['M_NL_FF_SU2', 'M_NL_FF_SU3']:
        if key in results_global:
            combined[f'{key}_global'] = results_global[key]
        if key in results_local:
            combined[f'{key}_local'] = results_local[key]
    
    return combined


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
    
    if len(sys.argv) < 2:
        print("Usage: python reader_TmFeO3_hdf5.py <hdf5_file_or_directory> [analysis_type]")
        print("  analysis_type: 'md' for molecular dynamics, '2dcs' for 2D spectroscopy")
        print("  If a directory is provided, the script will search for the appropriate HDF5 file.")
        sys.exit(1)
    
    input_path = sys.argv[1]
    analysis_type = sys.argv[2] if len(sys.argv) > 2 else 'md'
    
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
    print("\nHDF5 Structure:")
    print("-" * 50)
    print_hdf5_structure(filepath)
    print("-" * 50)
    
    if analysis_type == 'md':
        print("\nRunning MD analysis...")
        results = read_MD_hdf5(filepath)
        print(f"Results keys: {list(results.keys())}")
    elif analysis_type == '2dcs':
        print("\nRunning 2DCS analysis...")
        results = read_2DCS_combined_hdf5(filepath)
        print(f"Results keys: {list(results.keys())}")
    else:
        print(f"Unknown analysis type: {analysis_type}")
        sys.exit(1)
    
    print("\nAnalysis complete!")
