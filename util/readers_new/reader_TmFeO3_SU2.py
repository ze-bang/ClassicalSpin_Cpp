"""
HDF5 Reader for TmFeO3 Molecular Dynamics Results (SU2-only version)

This module reads simulation outputs from the ClassicalSpin_Cpp package stored in HDF5 format.
Simplified version for SU(2) spin-only simulations (no SU(3) multipolar moments).

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
     - spins [n_steps, n_sites, spin_dim]
     - magnetization_antiferro [n_steps, spin_dim]
     - magnetization_local [n_steps, spin_dim]
     - magnetization_global [n_steps, spin_dim]
"""

import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
import os
import sys
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
        global_frame: If True, transform to global frame
        
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
    Reader for molecular dynamics trajectory HDF5 files (SU2-only version).
    
    File structure:
    /metadata/
      - positions [n_sites, 3]
      - Attributes: lattice_size, spin_dim, n_atoms, spin_length,
                    integration_method, dt_initial, T_start, T_end, etc.
    /trajectory/
      - times [n_steps]
      - spins [n_steps, n_sites, spin_dim]
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
                'spin_dim': read_hdf5_attribute(grp, 'spin_dim'),
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
        
        # Get spin dimension
        spin_dim = reader.metadata.get('spin_dim', 3)
        if spin_dim is None:
            spin_dim = S.shape[2]
        spin_dim = int(spin_dim)
        
        # Compute DSSF in local frame
        A_local = DSSF(w, DSSF_K, S, P, T, global_frame=False)
        results['DSSF_local'] = A_local
        
        # Compute DSSF in global frame
        A_global = DSSF(w, DSSF_K, S, P, T, global_frame=True)
        results['DSSF_global'] = A_global
        
        # Plot individual components - local frame
        _plot_DSSF_components(A_local, w, tick_positions, output_dir, 'local', 
                              min(3, spin_dim), w0, wmax)
        
        # Plot individual components - global frame
        _plot_DSSF_components(A_global, w, tick_positions, output_dir, 'global', 
                              3, w0, wmax)
        
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
    fig.colorbar(C)
    plt.savefig(os.path.join(output_dir, f"DSSF_{frame_type}_sum.pdf"))
    plt.close()


def _plot_gap_analysis(w: np.ndarray, DSSF_Gamma: np.ndarray, output_dir: str, 
                       frame_type: str):
    """Plot DSSF at Gamma point for gap analysis."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(w, DSSF_Gamma[:, 0])
    ax.set_xlim([-3, 3])
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$S(\Gamma, \omega)$')
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
    fig.colorbar(C)
    plt.savefig(os.path.join(output_dir, f"DSSF_{frame_type}.pdf"))
    plt.close()


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
        
        # Plot antiferromagnetic magnetization
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        labels = ['x', 'y', 'z']
        for i in range(3):
            axes[i].plot(T, M_antiferro[:, i], label=f'$M_{{AF,{labels[i]}}}$')
            axes[i].set_ylabel(f'$M_{{AF,{labels[i]}}}$')
            axes[i].legend()
        axes[-1].set_xlabel('Time')
        fig.suptitle('Antiferromagnetic Magnetization')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'M_antiferro.pdf'))
        plt.close()
        
        # Plot local magnetization
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for i in range(3):
            axes[i].plot(T, M_local[:, i], label=f'$M_{{local,{labels[i]}}}$')
            axes[i].set_ylabel(f'$M_{{local,{labels[i]}}}$')
            axes[i].legend()
        axes[-1].set_xlabel('Time')
        fig.suptitle('Local Magnetization')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'M_local.pdf'))
        plt.close()
        
        # Plot global magnetization
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for i in range(3):
            axes[i].plot(T, M_global[:, i], label=f'$M_{{global,{labels[i]}}}$')
            axes[i].set_ylabel(f'$M_{{global,{labels[i]}}}$')
            axes[i].legend()
        axes[-1].set_xlabel('Time')
        fig.suptitle('Global Magnetization')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'M_global.pdf'))
        plt.close()
        
        # Save data
        np.savetxt(os.path.join(output_dir, 'M_antiferro.txt'), 
                   np.column_stack((T, M_antiferro)), 
                   header='time Mx My Mz')
        np.savetxt(os.path.join(output_dir, 'M_local.txt'), 
                   np.column_stack((T, M_local)), 
                   header='time Mx My Mz')
        np.savetxt(os.path.join(output_dir, 'M_global.txt'), 
                   np.column_stack((T, M_global)), 
                   header='time Mx My Mz')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reader_TmFeO3_SU2.py <hdf5_file> [analysis_type]")
        print("  analysis_type: 'md' for molecular dynamics (default), 'mag' for magnetization plots")
        sys.exit(1)
    
    filepath = sys.argv[1]
    analysis_type = sys.argv[2] if len(sys.argv) > 2 else 'md'
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    print(f"Analyzing {filepath} (type: {analysis_type})")
    print("\nHDF5 Structure:")
    print("-" * 50)
    print_hdf5_structure(filepath)
    print("-" * 50)
    
    if analysis_type == 'md':
        print("\nRunning MD analysis (DSSF computation)...")
        results = read_MD_hdf5(filepath)
        print(f"Results keys: {list(results.keys())}")
    elif analysis_type == 'mag':
        print("\nPlotting magnetization trajectories...")
        plot_magnetization_trajectory(filepath)
    else:
        print(f"Unknown analysis type: {analysis_type}")
        sys.exit(1)
    
    print("\nAnalysis complete!")
