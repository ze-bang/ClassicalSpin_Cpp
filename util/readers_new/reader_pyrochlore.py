"""
HDF5 Reader for Pyrochlore Molecular Dynamics and 2D Coherent Spectroscopy Results

This module reads simulation outputs from the ClassicalSpin_Cpp package stored in HDF5 format.
Mirrors the functionality of the old reader_pyrochlore.py but works with the new HDF5 outputs
as specified in hdf5_io.h and mixed_lattice.h.

HDF5 File Structures:
=====================

1. Molecular Dynamics trajectory.h5:
   /trajectory/
     - times [n_steps]
     - spins [n_steps, n_sites, spin_dim]
     - magnetization_antiferro [n_steps, spin_dim]
     - magnetization_local [n_steps, spin_dim]
   /metadata/
     - positions [n_sites, 3], lattice_size, spin_dim, etc.

2. Mixed SU2/SU3 Molecular Dynamics trajectory.h5:
   /trajectory_SU2/, /trajectory_SU3/
   /metadata_SU2/, /metadata_SU3/, /metadata_global/

3. Pump-Probe Spectroscopy pump_probe_spectroscopy.h5:
   /metadata/
   /reference/
   /tau_scan/

Author: Auto-generated for ClassicalSpin_Cpp
"""

import argparse
import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
import os
import re
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Dict, Tuple, Optional, List, Any, Callable


# =============================================================================
# PYROCHLORE LATTICE CONSTANTS AND K-PATH GENERATION
# =============================================================================

# Local frame for pyrochlore
z = np.array([[1,1,1],[1,-1,-1],[-1,1,-1], [-1,-1,1]])/np.sqrt(3)
y = np.array([[0,-1,1],[0,1,-1],[0,-1,-1], [0,1,1]])/np.sqrt(2)
x = np.array([[-2,1,1],[-2,-1,-1],[2,1,-1], [2,-1,1]])/np.sqrt(6)
localframe = np.array([x, y, z])

# Reciprocal lattice basis
BasisBZA = np.array([2*np.pi*np.array([-1,1,1]), 2*np.pi*np.array([1,-1,1]), 2*np.pi*np.array([1,1,-1])])
BasisBZA_reverse = np.array([np.array([0,1,1]), np.array([1,0,1]), np.array([1,1,0])]) / 2

# High symmetry points
Gamma = np.array([0, 0, 0])
K_point = 2 * np.pi * np.array([3/4, -3/4, 0])
W = 2 * np.pi * np.array([1, -1/2, 0])
X = 2 * np.pi * np.array([1, 0, 0])
L = np.pi * np.array([1, 1, 1])
U = 2 * np.pi * np.array([1/4, 1/4, 1])
W1 = 2 * np.pi * np.array([0, 1/2, 1])
X1 = 2 * np.pi * np.array([0, 0, 1])


def drawLine(A: np.ndarray, B: np.ndarray, stepN: float) -> np.ndarray:
    """Draw a line from A to B with given step size."""
    N = np.linalg.norm(A - B)
    num = max(int(N / stepN), 2)
    return np.linspace(A, B, num)


def magnitude_bi(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate magnitude of difference between two vectors."""
    return np.linalg.norm(vector1 - vector2)


def generate_k_path_pyrochlore(graphres: int = 2) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate k-path through high-symmetry points for pyrochlore lattice.
    
    Returns:
        DSSF_K: k-point array along the path
        tick_positions: dictionary with tick positions for plotting
    """
    stepN = np.linalg.norm(U - W1) / graphres
    
    # Path: Γ -> X -> W -> K -> Γ -> L -> U -> W' -> X' -> Γ
    GammaX = drawLine(Gamma, X, stepN)[1:]
    XW = drawLine(X, W, stepN)
    WK = drawLine(W, K_point, stepN)
    KGamma = drawLine(K_point, Gamma, stepN)[:-1]
    
    GammaL = drawLine(Gamma, L, stepN)[1:]
    LU = drawLine(L, U, stepN)
    UW1 = drawLine(U, W1, stepN)
    W1X1 = drawLine(W1, X1, stepN)
    X1Gamma = drawLine(X1, Gamma, stepN)[:-1]
    
    DSSF_K = np.concatenate((GammaX, XW, WK, KGamma, GammaL, LU, UW1, W1X1, X1Gamma))
    
    # Calculate tick positions
    gGamma1 = 0
    gX = magnitude_bi(Gamma, X)
    gW = gX + magnitude_bi(X, W)
    gK = gW + magnitude_bi(W, K_point)
    gGamma2 = gK + magnitude_bi(K_point, Gamma)
    gL = gGamma2 + magnitude_bi(Gamma, L)
    gU = gL + magnitude_bi(L, U)
    gW1 = gU + magnitude_bi(U, W1)
    gX1 = gW1 + magnitude_bi(W1, X1)
    gGamma3 = gX1 + magnitude_bi(X1, Gamma)
    
    tick_positions = {
        'Gamma1': gGamma1, 'X': gX, 'W': gW, 'K': gK,
        'Gamma2': gGamma2, 'L': gL, 'U': gU, 'W1': gW1,
        'X1': gX1, 'Gamma3': gGamma3
    }
    
    return DSSF_K, tick_positions


def genBZ(d: int, m: int = 1) -> np.ndarray:
    """Generate Brillouin zone k-points."""
    dj = d * 1j
    b = np.mgrid[0:m:dj, 0:m:dj, 0:m:dj].reshape(3, -1).T
    # Add high symmetry points
    sym_points = np.mgrid[0:1:9j, 0:1:9j, 0:1:9j].reshape(3, -1).T
    b = np.concatenate((b, sym_points))
    b = contract('ij, jk->ik', b, BasisBZA)
    return b


# =============================================================================
# SPIN STRUCTURE FACTOR COMPUTATIONS
# =============================================================================

def Spin(k: np.ndarray, S: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute spin structure factor."""
    kP = np.dot(k, P.T)
    ffact = np.exp(1j * kP)
    N = len(S)
    return contract('js, ij->is', S, ffact) / np.sqrt(N)


def Spin_t(k: np.ndarray, S: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute time-dependent spin structure factor."""
    kP = np.dot(k, P.T)
    ffact = np.exp(1j * kP)
    N = S.shape[1]
    return contract('tjs, ij->tis', S, ffact) / np.sqrt(N)


def Spin_global_pyrochlore(k: np.ndarray, S: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute spin structure factor in global frame for pyrochlore."""
    size = int(len(P))
    tS = np.zeros((4, len(k), 3), dtype=np.complex128)
    for i in range(4):
        P_sub = P[i::4]
        kP = np.dot(k, P_sub.T)
        ffact = np.exp(1j * kP)
        tS[i, :, :] = contract('js, ij->is', S[i::4], ffact) / np.sqrt(size)
    return tS


def Spin_global_pyrochlore_t(k: np.ndarray, S: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute time-dependent spin structure factor in global frame for pyrochlore."""
    size = int(len(P))
    tS = np.zeros((len(S), 4, len(k), 3), dtype=np.complex128)
    for i in range(4):
        P_sub = P[i::4]
        kP = np.dot(k, P_sub.T)
        ffact = np.exp(1j * kP)
        tS[:, i, :, :] = contract('tjs, ij->tis', S[:, i::4], ffact) / np.sqrt(size)
    return tS


def ggz(q: np.ndarray) -> np.ndarray:
    """Compute g-tensor for z-component."""
    M = np.zeros((len(q), 4, 4))
    for k in range(len(q)):
        for a in range(4):
            for b in range(4):
                qnorm = np.dot(q[k], q[k])
                if qnorm != 0:
                    M[k, a, b] = np.dot(localframe[2][a], localframe[2][b]) - \
                                 np.dot(localframe[2][a], q[k]) * np.dot(localframe[2][b], q[k]) / qnorm
                else:
                    M[k, a, b] = np.dot(localframe[2][a], localframe[2][b])
    return M


def gg(q: np.ndarray) -> np.ndarray:
    """Compute full g-tensor for pyrochlore."""
    M = np.zeros((len(q), 4, 4, 3, 3))
    for k in range(len(q)):
        qnorm = np.dot(q[k], q[k])
        for i in range(4):
            for j in range(4):
                for a in range(3):
                    for b in range(3):
                        if qnorm != 0:
                            M[k, i, j, a, b] = np.dot(localframe[a][i], localframe[b][j]) - \
                                               np.dot(localframe[a][i], q[k]) * np.dot(localframe[b][j], q[k]) / qnorm
                        else:
                            M[k, i, j, a, b] = np.dot(localframe[a][i], localframe[b][j])
    return M


def SSSF_q(k: np.ndarray, S: np.ndarray, P: np.ndarray, gb: bool = False) -> np.ndarray:
    """Compute static spin structure factor."""
    A = Spin_global_pyrochlore(k, S, P)
    if gb:
        read = np.abs(contract('nia, mib, inmab->inmab', A, np.conj(A), gg(k)))
    else:
        read = np.abs(contract('nia, mib->inmab', A, np.conj(A)))
    read = np.where(read <= 1e-8, 1e-8, read)
    return read


def DSSF(w: Optional[np.ndarray], k: np.ndarray, S: np.ndarray, P: np.ndarray, 
         T: np.ndarray, gb: bool = False) -> np.ndarray:
    """
    Compute dynamical spin structure factor using FFT over time.
    
    Args:
        w: Frequency points (if None, returns FFT frequencies)
        k: k-points array
        S: Spin configurations [n_times, n_sites, spin_dim]
        P: Site positions
        T: Time points
        gb: Whether to use global frame
        
    Returns:
        DSSF array
    """
    dT = T[1] - T[0] if len(T) > 1 else 1.0
    A = Spin_global_pyrochlore_t(k, S, P)
    
    nT = len(T)
    nT_padded = 2**int(np.ceil(np.log2(nT)))
    
    if T[0] < 0 and T[-1] > 0:
        A_shifted = np.fft.ifftshift(A, axes=0)
        A_fft = np.fft.fft(A_shifted, n=nT_padded, axis=0)
        A_fft = np.fft.fftshift(A_fft, axes=0)
        freq = np.fft.fftshift(np.fft.fftfreq(nT_padded, d=dT))
    else:
        A_fft = np.fft.fft(A, n=nT_padded, axis=0)
        freq = np.fft.fftfreq(nT_padded, d=dT)
    
    if w is None:
        Somega = dT / (2*np.pi) * A_fft / np.sqrt(nT)
        if gb:
            read = np.abs(contract('wnia, wmib, inm->winmab', Somega, np.conj(Somega), ggz(k)))
        else:
            read = np.abs(contract('wnia, wmib->winmab', Somega, np.conj(Somega)))
        read = np.where(read <= 1e-8, 1e-8, read)
        return freq, read
    
    # Interpolate to desired frequency grid
    Somega = np.zeros((len(w), A.shape[1], A.shape[2], A.shape[3]), dtype=np.complex128)
    for i in range(A.shape[1]):
        for j in range(A.shape[2]):
            for l in range(A.shape[3]):
                idx = np.argsort(freq)
                interp_real = interp1d(freq[idx], A_fft[idx, i, j, l].real, 
                                       kind='cubic', fill_value=0, bounds_error=False)
                interp_imag = interp1d(freq[idx], A_fft[idx, i, j, l].imag, 
                                       kind='cubic', fill_value=0, bounds_error=False)
                Somega[:, i, j, l] = interp_real(w) + 1j * interp_imag(w)
    
    Somega = dT / (2*np.pi) * Somega / np.sqrt(nT)
    
    if gb:
        read = np.abs(contract('wnia, wmib, inm, w->winmab', Somega, np.conj(Somega), ggz(k), w))
    else:
        read = np.abs(contract('wnia, wmib, w->winmab', Somega, np.conj(Somega), w))
    read = np.where(read <= 1e-8, 1e-8, read)
    return read


# =============================================================================
# K-SPACE TRANSFORMATION FUNCTIONS
# =============================================================================

def hhltoK(H: np.ndarray, L: np.ndarray, K: int = 0) -> np.ndarray:
    """Transform (H,H,L) coordinates to k-space."""
    return contract('ij,k->ijk', H, 2*np.array([np.pi, np.pi, 0])) + \
           contract('ij,k->ijk', L, 2*np.array([0, 0, np.pi]))


def hnhltoK(H: np.ndarray, L: np.ndarray, K: int = 0) -> np.ndarray:
    """Transform (H,-H,L) coordinates to k-space."""
    return contract('ij,k->ijk', H, 2*np.array([np.pi, -np.pi, 0])) + \
           contract('ij,k->ijk', L, 2*np.array([0, 0, np.pi]))


def hhztoK(H: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Transform (H,K,0) coordinates to k-space."""
    return contract('ij,k->ijk', H, 2*np.array([np.pi, 0, 0])) + \
           contract('ij,k->ijk', K, 2*np.array([0, np.pi, 0]))


def hnhztoK(H: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Transform (H,-H,K) coordinates to k-space."""
    return contract('ij,k->ijk', H, 2*np.pi*np.array([1, -1, 0])) + \
           contract('ij,k->ijk', K, 2*np.pi*np.array([1, 1, -2]))


# =============================================================================
# SSSF COMPUTATION FUNCTIONS
# =============================================================================

def SSSFHHL(S: np.ndarray, P: np.ndarray, nK: int, filename: str, 
            gb: bool = False) -> np.ndarray:
    """Compute SSSF in (H,H,L) plane."""
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hhltoK(A, B).reshape((nK*nK, 3))
    S_result = SSSF_q(K, S, P, gb)
    return S_result.reshape((nK, nK, 4, 4, 3, 3))


def SSSFHnHL(S: np.ndarray, P: np.ndarray, nK: int, filename: str, 
             gb: bool = False) -> np.ndarray:
    """Compute SSSF in (H,-H,L) plane."""
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hnhltoK(A, B).reshape((nK*nK, 3))
    S_result = SSSF_q(K, S, P, gb)
    return S_result.reshape((nK, nK, 4, 4, 3, 3))


def SSSFHK0(S: np.ndarray, P: np.ndarray, nK: int, filename: str, 
            gb: bool = False) -> np.ndarray:
    """Compute SSSF in (H,K,0) plane."""
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hhztoK(A, B).reshape((nK*nK, 3))
    S_result = SSSF_q(K, S, P, gb)
    return S_result.reshape((nK, nK, 4, 4, 3, 3))


def SSSFHnHn(S: np.ndarray, P: np.ndarray, nK: int, filename: str, 
             gb: bool = False) -> np.ndarray:
    """Compute SSSF in (H,-H,n) plane."""
    H = np.linspace(-2.5, 2.5, nK)
    L = np.linspace(-2.5, 2.5, nK)
    A, B = np.meshgrid(H, L)
    K = hnhztoK(A, B).reshape((nK*nK, 3))
    S_result = SSSF_q(K, S, P, gb)
    return S_result.reshape((nK, nK, 4, 4, 3, 3))


# =============================================================================
# HDF5 READING UTILITIES
# =============================================================================

def read_hdf5_attribute(group, attr_name: str, default=None):
    """Safely read an HDF5 attribute."""
    try:
        val = group.attrs[attr_name]
        if isinstance(val, bytes):
            return val.decode('utf-8')
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
    Reader for molecular dynamics trajectory HDF5 files.
    
    Supports both single-lattice (trajectory/) and mixed-lattice (trajectory_SU2/, trajectory_SU3/) formats.
    """
    
    def __init__(self, filepath: str):
        """Initialize the MD trajectory reader."""
        self.filepath = filepath
        self._file = None
        self.metadata_global = {}
        self.metadata = {}
        self.metadata_SU2 = {}
        self.metadata_SU3 = {}
        self.is_mixed = False
        
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self):
        """Open the HDF5 file and load metadata."""
        self._file = h5py.File(self.filepath, 'r')
        self._detect_format()
        self._load_metadata()
        
    def close(self):
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def _detect_format(self):
        """Detect whether this is a single or mixed lattice file."""
        self.is_mixed = 'trajectory_SU2' in self._file or 'trajectory_SU3' in self._file
            
    def _load_metadata(self):
        """Load metadata from the HDF5 file."""
        if self.is_mixed:
            # Mixed lattice format
            if 'metadata_global' in self._file:
                grp = self._file['metadata_global']
                self.metadata_global = {
                    'integration_method': read_hdf5_attribute(grp, 'integration_method'),
                    'dt_initial': read_hdf5_attribute(grp, 'dt_initial'),
                    'T_start': read_hdf5_attribute(grp, 'T_start'),
                    'T_end': read_hdf5_attribute(grp, 'T_end'),
                    'save_interval': read_hdf5_attribute(grp, 'save_interval'),
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
        else:
            # Single lattice format
            if 'metadata' in self._file:
                grp = self._file['metadata']
                self.metadata = {
                    'lattice_size': read_hdf5_attribute(grp, 'lattice_size'),
                    'spin_dim': read_hdf5_attribute(grp, 'spin_dim'),
                    'n_atoms': read_hdf5_attribute(grp, 'n_atoms'),
                    'integration_method': read_hdf5_attribute(grp, 'integration_method'),
                    'positions': read_hdf5_dataset(grp, 'positions'),
                    'dimensions': read_hdf5_dataset(grp, 'dimensions'),
                }
    
    def get_times(self, sublattice: Optional[str] = None) -> np.ndarray:
        """Get time points."""
        if self.is_mixed and sublattice:
            return self._file[f'trajectory_{sublattice}/times'][:]
        return self._file['trajectory/times'][:]
    
    def get_spins(self, sublattice: Optional[str] = None) -> np.ndarray:
        """Get spin trajectory [n_steps, n_sites, spin_dim]."""
        if self.is_mixed and sublattice:
            return self._file[f'trajectory_{sublattice}/spins'][:]
        return self._file['trajectory/spins'][:]
    
    def get_magnetization_antiferro(self, sublattice: Optional[str] = None) -> np.ndarray:
        """Get antiferromagnetic magnetization [n_steps, spin_dim]."""
        if self.is_mixed and sublattice:
            return self._file[f'trajectory_{sublattice}/magnetization_antiferro'][:]
        return self._file['trajectory/magnetization_antiferro'][:]
    
    def get_magnetization_local(self, sublattice: Optional[str] = None) -> np.ndarray:
        """Get local magnetization [n_steps, spin_dim]."""
        if self.is_mixed and sublattice:
            return self._file[f'trajectory_{sublattice}/magnetization_local'][:]
        return self._file['trajectory/magnetization_local'][:]
    
    def get_positions(self, sublattice: Optional[str] = None) -> Optional[np.ndarray]:
        """Get site positions [n_sites, 3]."""
        if self.is_mixed and sublattice:
            meta = self.metadata_SU2 if sublattice == 'SU2' else self.metadata_SU3
            return meta.get('positions')
        return self.metadata.get('positions')


class PumpProbeHDF5:
    """Reader for pump-probe spectroscopy HDF5 files."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file = None
        self.metadata = {}
        self.is_mixed = False
        
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self):
        """Open the HDF5 file and load metadata."""
        self._file = h5py.File(self.filepath, 'r')
        self._detect_format()
        self._load_metadata()
        
    def close(self):
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def _detect_format(self):
        """Detect whether this is a single or mixed lattice file."""
        if 'metadata' in self._file:
            self.is_mixed = read_hdf5_attribute(self._file['metadata'], 'experiment_type', '') == \
                           'pump_probe_spectroscopy_mixed'
            
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
        """Get delay time values."""
        return self._file['tau_scan/tau_values'][:]
    
    def get_reference_times(self) -> np.ndarray:
        """Get reference trajectory times."""
        return self._file['reference/times'][:]
    
    def get_reference_magnetization(self, mag_type: str = 'global', 
                                     sublattice: str = '') -> np.ndarray:
        """Get reference magnetization trajectory."""
        if self.is_mixed:
            dataset_name = f'M_{mag_type}_{sublattice}'
        else:
            dataset_name = f'M_{mag_type}'
        return self._file[f'reference/{dataset_name}'][:]
    
    def get_tau_trajectory(self, tau_index: int, trajectory_type: str = 'M01',
                          mag_type: str = 'global', sublattice: str = '') -> np.ndarray:
        """Get delay-dependent magnetization trajectory."""
        if self.is_mixed:
            dataset_name = f'{trajectory_type}_{mag_type}_{sublattice}'
        else:
            dataset_name = f'{trajectory_type}_{mag_type}'
        return self._file[f'tau_scan/tau_{tau_index}/{dataset_name}'][:]
    
    def get_tau_value(self, tau_index: int) -> float:
        """Get the tau value for a specific index."""
        tau_group = self._file[f'tau_scan/tau_{tau_index}']
        return tau_group.attrs['tau_value']
    
    def get_n_tau_steps(self) -> int:
        """Get the number of tau steps."""
        return len(self.get_tau_values())


# =============================================================================
# GAUSSIAN RESOLUTION FUNCTION
# =============================================================================

def gaussian_resolution(energy: np.ndarray, intensity: np.ndarray, 
                        resolution_fwhm: float) -> np.ndarray:
    """
    Convolve intensity with Gaussian experimental resolution.
    
    Parameters:
    - energy: array of energy values
    - intensity: array of intensity values
    - resolution_fwhm: Full Width at Half Maximum of the Gaussian
    """
    energy_spacing = np.mean(np.diff(energy))
    sigma = resolution_fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_indices = sigma / energy_spacing
    return gaussian_filter1d(intensity, sigma_indices)


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def read_MD_hdf5(filepath: str, w0: float = 0.03, wmax: float = 10.0,
                 output_dir: Optional[str] = None, 
                 mag: str = 'HnHn') -> Dict[str, np.ndarray]:
    """
    Read molecular dynamics trajectory from HDF5 and compute DSSF.
    
    Args:
        filepath: Path to trajectory.h5 file
        w0: Minimum frequency
        wmax: Maximum frequency
        output_dir: Directory for output plots (uses filepath directory if None)
        mag: Magnetization plane for SSSF ('HnHn', 'HHL', 'HnHL', 'HK0')
        
    Returns:
        Dictionary with DSSF results
    """
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    with MDTrajectoryHDF5(filepath) as reader:
        DSSF_K, tick_positions = generate_k_path_pyrochlore()
        
        # Choose the appropriate sublattice
        if reader.is_mixed:
            sublattice = 'SU2'  # For pyrochlore, typically use SU2 (3-component spins)
        else:
            sublattice = None
        
        T = reader.get_times(sublattice)
        S = reader.get_spins(sublattice)
        P = reader.get_positions(sublattice)
        
        if P is None:
            print("Warning: No positions found. Cannot compute DSSF.")
            return results
        
        # Compute DSSF with FFT (returns freq, data)
        freq, DSSF_local = DSSF(None, DSSF_K, S, P, T, gb=False)
        freq, DSSF_global = DSSF(None, DSSF_K, S, P, T, gb=True)
        
        # Filter to desired frequency range
        w_mask = (freq >= w0) & (freq <= wmax)
        w = freq[w_mask]
        DSSF_local = DSSF_local[w_mask]
        DSSF_global = DSSF_global[w_mask]
        
        results['w'] = w
        results['DSSF_local'] = DSSF_local
        results['DSSF_global'] = DSSF_global
        results['tick_positions'] = tick_positions
        
        # Plot DSSF
        _plot_DSSF_pyrochlore(DSSF_local, w, tick_positions, output_dir, 'local', w0, wmax)
        _plot_DSSF_pyrochlore(DSSF_global, w, tick_positions, output_dir, 'global', w0, wmax)
        
        # Compute SSSF from initial configuration
        S0 = S[0] if len(S) > 0 else S
        
        sssf_func = {
            'HnHn': SSSFHnHn,
            'HHL': SSSFHHL,
            'HnHL': SSSFHnHL,
            'HK0': SSSFHK0,
        }.get(mag, SSSFHnHn)
        
        SSSF_local = sssf_func(S0, P, 50, output_dir, False)
        SSSF_global = sssf_func(S0, P, 50, output_dir, True)
        
        results['SSSF_local'] = SSSF_local
        results['SSSF_global'] = SSSF_global
    
    return results


def _plot_DSSF_pyrochlore(DSSF_data: np.ndarray, w: np.ndarray, 
                          tick_positions: Dict[str, float],
                          output_dir: str, name: str,
                          w0: float, wmax: float):
    """Plot DSSF for pyrochlore lattice."""
    labels = [r'$\Gamma$', r'$X$', r'$W$', r'$K$', r'$\Gamma$', 
              r'$L$', r'$U$', r'$W^\prime$', r'$X^\prime$', r'$\Gamma$']
    tick_pos = [tick_positions['Gamma1'], tick_positions['X'], tick_positions['W'],
                tick_positions['K'], tick_positions['Gamma2'], tick_positions['L'],
                tick_positions['U'], tick_positions['W1'], tick_positions['X1'],
                tick_positions['Gamma3']]
    
    # Sum over sublattice indices
    DSSF_sum = contract('wkxyab->wkab', DSSF_data)
    
    # Plot each component
    com_string = ['x', 'y', 'z']
    for i in range(3):
        for j in range(3):
            fig, ax = plt.subplots(figsize=(10, 4))
            data = np.log(np.maximum(DSSF_sum[:, :, i, j], 1e-10))
            C = ax.imshow(data, origin='lower', 
                         extent=[0, tick_positions['Gamma3'], w0, wmax],
                         aspect='auto', interpolation='lanczos', cmap='gnuplot2')
            
            for pos in tick_pos:
                ax.axvline(x=pos, color='b', linestyle='dashed', alpha=0.5)
            
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(labels)
            ax.set_xlim([0, tick_positions['Gamma3']])
            ax.set_ylabel(r'$\omega$')
            fig.colorbar(C)
            plt.savefig(os.path.join(output_dir, f"DSSF_{name}_{com_string[i]}{com_string[j]}.pdf"))
            plt.close()
    
    # Sum of all components
    DSSF_total = np.sum(DSSF_sum, axis=(2, 3))
    fig, ax = plt.subplots(figsize=(10, 4))
    data = np.log(np.maximum(DSSF_total, 1e-10))
    C = ax.imshow(data, origin='lower', 
                 extent=[0, tick_positions['Gamma3'], w0, wmax],
                 aspect='auto', interpolation='lanczos', cmap='gnuplot2')
    
    for pos in tick_pos:
        ax.axvline(x=pos, color='b', linestyle='dashed', alpha=0.5)
    
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(labels)
    ax.set_xlim([0, tick_positions['Gamma3']])
    ax.set_ylabel(r'$\omega$')
    fig.colorbar(C)
    plt.savefig(os.path.join(output_dir, f"DSSF_{name}_sum.pdf"))
    plt.close()
    
    # Save data
    np.savetxt(os.path.join(output_dir, f"DSSF_{name}_sum.txt"), DSSF_total)


def read_2DCS_hdf5(filepath: str, mag_type: str = 'global', 
                   omega_range: float = 15.0,
                   output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Read 2D coherent spectroscopy data from HDF5 and compute nonlinear response.
    
    Args:
        filepath: Path to pump_probe_spectroscopy.h5 file
        mag_type: Type of magnetization to use ('antiferro', 'local', 'global')
        omega_range: Maximum frequency range
        output_dir: Directory for output plots
        
    Returns:
        Dictionary with 2DCS results
    """
    if output_dir is None:
        output_dir = os.path.dirname(filepath)
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    with PumpProbeHDF5(filepath) as reader:
        tau_values = reader.get_tau_values()
        times = reader.get_reference_times()
        n_tau = len(tau_values)
        n_times = len(times)
        
        # Determine sublattice based on file format
        sublattice = 'SU2' if reader.is_mixed else ''
        
        try:
            spin_dim = int(reader.metadata.get('spin_dim', 3))
        except (KeyError, TypeError):
            spin_dim = 3
        
        # Get M0 trajectory
        M0 = reader.get_reference_magnetization(mag_type, sublattice)
        
        # Setup for 2D FFT
        dT = times[1] - times[0] if len(times) > 1 else 1.0
        dtau = tau_values[1] - tau_values[0] if n_tau > 1 else 1.0
        
        # Initialize nonlinear response
        domain = len(times) // 2
        M_NL = np.zeros((n_tau, domain, spin_dim))
        
        for i_tau in range(n_tau):
            try:
                M1 = reader.get_tau_trajectory(i_tau, 'M1', mag_type, sublattice)
                M01 = reader.get_tau_trajectory(i_tau, 'M01', mag_type, sublattice)
                
                # Compute nonlinear response
                M_NL[i_tau] = M01[-domain:] - M0[-domain:] - M1[-domain:]
                
            except Exception as e:
                print(f"Error processing tau_{i_tau}: {e}")
                continue
        
        # 2D FFT
        n_tau_padded = 2**int(np.ceil(np.log2(n_tau)))
        n_T_padded = 2**int(np.ceil(np.log2(domain)))
        
        M_NL_fft = np.zeros((n_tau_padded, n_T_padded, spin_dim), dtype=complex)
        for d in range(spin_dim):
            M_NL_fft[:, :, d] = np.fft.fftshift(
                np.fft.fft2(M_NL[:, :, d], s=(n_tau_padded, n_T_padded))
            )
        
        freq_tau = np.fft.fftshift(np.fft.fftfreq(n_tau_padded, d=dtau))
        freq_T = np.fft.fftshift(np.fft.fftfreq(n_T_padded, d=dT))
        
        # Filter to desired range
        idx_tau = (freq_tau >= -omega_range) & (freq_tau <= omega_range)
        idx_T = (freq_T >= -omega_range) & (freq_T <= omega_range)
        
        M_NL_FF = np.abs(M_NL_fft[np.ix_(idx_tau, idx_T)])
        
        results['M_NL_FF'] = M_NL_FF
        results['freq_tau'] = freq_tau[idx_tau]
        results['freq_T'] = freq_T[idx_T]
        
        # Plot and save
        for d in range(spin_dim):
            data = np.log(np.maximum(M_NL_FF[:, :, d], 1e-10))
            
            plt.figure(figsize=(10, 8))
            plt.imshow(data.T, origin='lower', 
                      extent=[-omega_range, omega_range, -omega_range, omega_range],
                      aspect='auto', interpolation='lanczos', cmap='gnuplot2')
            plt.colorbar(label='log(Amplitude)')
            plt.xlabel(r'$\omega_{\tau}$')
            plt.ylabel(r'$\omega_{t}$')
            plt.title(f'2D Nonlinear Spectrum (component {d})')
            plt.savefig(os.path.join(output_dir, f"NLSPEC_{mag_type}_{d}.pdf"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            np.savetxt(os.path.join(output_dir, f"M_NL_FF_{mag_type}_{d}.txt"), 
                      M_NL_FF[:, :, d])
    
    return results


# =============================================================================
# UTILITIES
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
    """Extract metadata from an HDF5 file."""
    metadata = {}
    
    with h5py.File(filepath, 'r') as f:
        for grp_name in ['metadata', 'metadata_global', 'metadata_SU2', 'metadata_SU3']:
            if grp_name in f:
                grp = f[grp_name]
                for key in grp.attrs.keys():
                    val = grp.attrs[key]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    metadata[f'{grp_name}/{key}'] = val
    
    return metadata


def magnetization(S: np.ndarray, glob: bool, fielddir: np.ndarray, 
                  theta: float = 0) -> np.ndarray:
    """Compute magnetization in specified direction."""
    if not glob:
        return np.mean(S, axis=0)
    else:
        size = int(len(S) / 4)
        zmag = contract('k,ik->i', fielddir, z)
        mag = np.zeros(3)
        for i in range(4):
            mag = mag + np.mean(S[i*size:(i+1)*size, 2]*np.cos(theta) + 
                               S[i*size:(i+1)*size, 0]*np.cos(theta), axis=0) * zmag[i]
        return mag


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def SSSFGraphHHL(A: np.ndarray, B: np.ndarray, d1: np.ndarray, filename: str):
    """Plot SSSF in (H,H,L) plane."""
    fig, ax = plt.subplots(figsize=(10, 4))
    C = ax.pcolormesh(A, B, d1)
    fig.colorbar(C)
    ax.set_ylabel(r'$(0,0,L)$')
    ax.set_xlabel(r'$(H,H,0)$')
    fig.savefig(filename + ".pdf")
    plt.close()


def SSSFGraphHnHL(A: np.ndarray, B: np.ndarray, d1: np.ndarray, filename: str):
    """Plot SSSF in (H,-H,L) plane."""
    fig, ax = plt.subplots(figsize=(10, 4))
    C = ax.pcolormesh(A, B, d1)
    fig.colorbar(C)
    ax.set_ylabel(r'$(0,0,L)$')
    ax.set_xlabel(r'$(H,-H,0)$')
    fig.savefig(filename + ".pdf")
    plt.close()


def SSSFGraphHK0(A: np.ndarray, B: np.ndarray, d1: np.ndarray, filename: str):
    """Plot SSSF in (H,K,0) plane."""
    fig, ax = plt.subplots(figsize=(10, 4))
    C = ax.pcolormesh(A, B, d1)
    fig.colorbar(C)
    ax.set_ylabel(r'$(0,K,0)$')
    ax.set_xlabel(r'$(H,0,0)$')
    fig.savefig(filename + ".pdf")
    plt.close()


def SSSFGraphHnHn(A: np.ndarray, B: np.ndarray, d1: np.ndarray, filename: str):
    """Plot SSSF in (H,-H,K) plane."""
    fig, ax = plt.subplots(figsize=(10, 4))
    C = ax.pcolormesh(A, B, d1)
    fig.colorbar(C)
    ax.set_ylabel(r'$(K,K,2K)$')
    ax.set_xlabel(r'$(H,-H,0)$')
    fig.savefig(filename + ".pdf")
    plt.close()


def _select_sssf_graph(mag: str) -> Optional[Callable]:
    """Return the SSSF graphing helper that matches the chosen magnetization label."""
    mapping = {
        "001": SSSFGraphHK0,
        "1-10": SSSFGraphHHL,
        "110": SSSFGraphHnHL,
        "111": SSSFGraphHnHn,
        "HnHn": SSSFGraphHnHn,
        "HK0": SSSFGraphHK0,
        "HHL": SSSFGraphHHL,
        "HnHL": SSSFGraphHnHL,
    }
    return mapping.get(mag)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process pyrochlore MD simulation HDF5 outputs.")
    parser.add_argument("filepath", help="Path to the HDF5 file to process.")
    parser.add_argument("--type", choices=['md', '2dcs'], default='md',
                       help="Analysis type: 'md' for molecular dynamics, '2dcs' for 2D spectroscopy")
    parser.add_argument("--mag", default="HnHn",
                       help="Magnetization direction label (e.g. HnHn, HHL, HK0).")
    parser.add_argument("--w0", type=float, default=0.03,
                       help="Minimum frequency for DSSF (default: 0.03)")
    parser.add_argument("--wmax", type=float, default=10.0,
                       help="Maximum frequency for DSSF (default: 10.0)")
    parser.add_argument("--output", "-o", help="Output directory for plots.")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress information.")

    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File not found: {args.filepath}")
        return 1

    if not args.quiet:
        print(f"Analyzing {args.filepath} (type: {args.type})")
        print("\nHDF5 Structure:")
        print("-" * 50)
        print_hdf5_structure(args.filepath)
        print("-" * 50)

    output_dir = args.output if args.output else os.path.dirname(args.filepath)

    if args.type == 'md':
        if not args.quiet:
            print("\nRunning MD analysis...")
        results = read_MD_hdf5(args.filepath, w0=args.w0, wmax=args.wmax,
                              output_dir=output_dir, mag=args.mag)
        if not args.quiet:
            print(f"Results keys: {list(results.keys())}")
    elif args.type == '2dcs':
        if not args.quiet:
            print("\nRunning 2DCS analysis...")
        results = read_2DCS_hdf5(args.filepath, output_dir=output_dir)
        if not args.quiet:
            print(f"Results keys: {list(results.keys())}")

    if not args.quiet:
        print("\nAnalysis complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
