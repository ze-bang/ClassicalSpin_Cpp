"""
Strain Phonon Lattice Analysis Tools
=====================================

This module provides analysis functionalities for magnetoelastic (spin-strain) 
coupled honeycomb lattice simulations (NCTO system with strain field phonons),
including:

1. read_pump_probe(dir)       - Read pump-probe simulation trajectory
2. read_MD(dir)               - Read molecular dynamics trajectory  
3. plot_trajectory(result)    - Plot magnetization and strain evolution
4. compute_fft_spectrum(result) - Compute frequency spectrum from time domain

The StrainPhononLattice simulations track:
- Spin dynamics: magnetization M(t), staggered magnetization M_stag(t)
- Strain dynamics: A1g mode (ε_xx + ε_yy), Eg modes (ε_xx - ε_yy, 2ε_xy)
- Time-dependent J7 modulation: J7_eff(t)
- Energy evolution during MD

Output file formats:
- magnetization.txt: time M_x M_y M_z |M| M_stag_x M_stag_y M_stag_z |M_stag|
- energy.txt: time E
- strain_trajectory.txt: time ε_A1g ε_Eg1 ε_Eg2 |ε_Eg| J7_eff
- strain_per_bond.txt: bond ε_xx ε_yy ε_xy V_xx V_yy V_xy
- initial_spins.txt, final_spins.txt: spin configuration
- initial_strain.txt, final_strain.txt: strain state

The strain field phonon model uses D3d symmetry:
- A1g channel: breathing mode (ε_xx + ε_yy) 
- Eg channel: shear modes (ε_xx - ε_yy, 2ε_xy)
- Time-dependent J7: J7(t) = J7 * (1 - γ*f(t)*λ_Eg/4)^4 * (1 + γ*f(t)*λ_Eg/2)^2
"""

import numpy as np
import os

# Optional imports - plotting and FFT require extra packages
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available. Plotting functions disabled.")

try:
    from scipy.fft import fft, fftfreq
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Note: scipy not available. FFT functions disabled.")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Note: h5py not available. HDF5 reading disabled.")

# ============================================================================
# CONSTANTS
# ============================================================================

# Unit conversion: simulation time (hbar/meV) to seconds
# hbar = 6.58211957e-13 eV·s = 6.58211957e-10 meV·s
# 1 simulation time unit = hbar / (1 meV) = 6.58211957e-13 s = 0.658211957 ps
HBAR_OVER_MEV = 6.58211957e-13  # seconds per simulation time unit
HBAR_OVER_MEV_PS = 0.658211957  # picoseconds per simulation time unit

# ============================================================================
# HONEYCOMB LATTICE GEOMETRY 
# ============================================================================

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
        
    Raises:
        ValueError: If n_sites is not compatible with honeycomb geometry
    """
    if n_sites % 2 != 0:
        raise ValueError(f"Number of sites ({n_sites}) must be even for honeycomb lattice")
    
    n_unit_cells = n_sites // 2
    
    # Try to infer dimensions - assume square-ish lattice for simplicity
    # Try common factorizations
    dim3 = 1  # Usually 1 for 2D simulations
    
    # Find factors for dim1 * dim2 = n_unit_cells
    sqrt_n = int(np.sqrt(n_unit_cells))
    dim1, dim2 = None, None
    
    # Try to find factors close to square
    for d1 in range(sqrt_n, 0, -1):
        if n_unit_cells % d1 == 0:
            dim1 = d1
            dim2 = n_unit_cells // d1
            break
    
    if dim1 is None:
        raise ValueError(f"Could not factorize {n_unit_cells} unit cells into lattice dimensions")
    
    print(f"  Inferred lattice dimensions: {dim1} x {dim2} x {dim3} (total {n_sites} sites)")
    
    # Generate positions
    # Lattice vectors
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0)/2.0, 0.0])
    a3 = np.array([0.0, 0.0, 1.0])
    
    # Sublattice positions within unit cell
    pos0 = np.array([0.0, 0.0, 0.0])  # Sublattice A
    pos1 = np.array([0.0, 1.0/np.sqrt(3.0), 0.0])  # Sublattice B
    
    positions = np.zeros((n_sites, 3))
    site_idx = 0
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                # Sublattice A
                positions[site_idx] = pos0 + i*a1 + j*a2 + k*a3
                site_idx += 1
                # Sublattice B
                positions[site_idx] = pos1 + i*a1 + j*a2 + k*a3
                site_idx += 1
    
    return positions


def read_positions(filepath, n_sites=None):
    """Read site positions from file, or generate them if file not found.
    
    Args:
        filepath: Path to positions.txt
        n_sites: Number of sites (required if file doesn't exist, for position inference)
        
    Returns:
        positions: (n_sites, 3) array of site positions
    """
    if os.path.exists(filepath):
        positions = np.loadtxt(filepath)
        print(f"  Loaded positions from {filepath}")
        return positions
    else:
        if n_sites is None:
            raise ValueError(f"Position file not found: {filepath}. "
                           "Must provide n_sites to generate positions.")
        print(f"  Position file not found: {filepath}")
        print(f"  Generating honeycomb positions for {n_sites} sites...")
        return generate_honeycomb_positions(n_sites)

# ============================================================================
# I/O FUNCTIONS
# ============================================================================

def read_magnetization(filepath):
    """Read magnetization trajectory from text file.
    
    Args:
        filepath: Path to magnetization.txt
        
    Returns:
        dict with keys: 't', 'Mx', 'My', 'Mz', 'M_norm', 
                        'Mx_stag', 'My_stag', 'Mz_stag', 'M_stag_norm'
    """
    data = np.loadtxt(filepath)
    return {
        't': data[:, 0],
        'Mx': data[:, 1],
        'My': data[:, 2],
        'Mz': data[:, 3],
        'M_norm': data[:, 4],
        'Mx_stag': data[:, 5],
        'My_stag': data[:, 6],
        'Mz_stag': data[:, 7],
        'M_stag_norm': data[:, 8]
    }


def read_energy(filepath):
    """Read energy trajectory from text file.
    
    Args:
        filepath: Path to energy.txt
        
    Returns:
        dict with keys: 't', 'E'
    """
    data = np.loadtxt(filepath)
    return {
        't': data[:, 0],
        'E': data[:, 1]
    }


def read_strain_trajectory(filepath):
    """Read strain trajectory from text file.
    
    Args:
        filepath: Path to strain_trajectory.txt
        
    Returns:
        dict with keys: 't', 'eps_A1g', 'eps_Eg1', 'eps_Eg2', 'eps_Eg_norm', 'J7_eff'
    """
    data = np.loadtxt(filepath)
    return {
        't': data[:, 0],
        'eps_A1g': data[:, 1],
        'eps_Eg1': data[:, 2],
        'eps_Eg2': data[:, 3],
        'eps_Eg_norm': data[:, 4],
        'J7_eff': data[:, 5]
    }


def read_spin_config(filepath):
    """Read spin configuration from text file.
    
    Args:
        filepath: Path to spins.txt
        
    Returns:
        ndarray of shape (n_sites, 3)
    """
    return np.loadtxt(filepath)


def read_strain_per_bond(filepath):
    """Read per-bond strain state from text file.
    
    Args:
        filepath: Path to strain_per_bond.txt
        
    Returns:
        dict with keys: 'bond', 'eps_xx', 'eps_yy', 'eps_xy', 'V_xx', 'V_yy', 'V_xy'
    """
    data = np.loadtxt(filepath)
    return {
        'bond': data[:, 0].astype(int),
        'eps_xx': data[:, 1],
        'eps_yy': data[:, 2],
        'eps_xy': data[:, 3],
        'V_xx': data[:, 4],
        'V_yy': data[:, 5],
        'V_xy': data[:, 6]
    }


def read_strain_per_bond_trajectory(filepath):
    """Read per-bond-type strain trajectory from text file.
    
    This is analogous to PhononLattice's per-bond-type phonon trajectory.
    
    Args:
        filepath: Path to strain_per_bond_trajectory.txt
        
    Returns:
        dict with per-bond-type strain components
    """
    data = np.loadtxt(filepath)
    return {
        't': data[:, 0],
        # Strain components per bond type (x=0, y=1, z=2)
        'eps_xx_0': data[:, 1], 'eps_xx_1': data[:, 2], 'eps_xx_2': data[:, 3],
        'eps_yy_0': data[:, 4], 'eps_yy_1': data[:, 5], 'eps_yy_2': data[:, 6],
        'eps_xy_0': data[:, 7], 'eps_xy_1': data[:, 8], 'eps_xy_2': data[:, 9],
        # Velocity components per bond type
        'V_xx_0': data[:, 10], 'V_xx_1': data[:, 11], 'V_xx_2': data[:, 12],
        'V_yy_0': data[:, 13], 'V_yy_1': data[:, 14], 'V_yy_2': data[:, 15],
        'V_xy_0': data[:, 16], 'V_xy_1': data[:, 17], 'V_xy_2': data[:, 18],
    }


def read_hdf5_trajectory(filepath):
    """Read strain lattice trajectory from HDF5 file.
    
    Analogous to read_MD_phonon in reader_phonon_lattice.py.
    
    HDF5 format:
    - /trajectory/times: Time array
    - /trajectory/spins: (n_times, n_sites, 3) full spin configuration
    - /trajectory/magnetization_local: (n_times, 3) local magnetization
    - /trajectory/magnetization_antiferro: (n_times, 3) staggered magnetization
    - /strain_trajectory/eps_xx_0, eps_xx_1, eps_xx_2, ...: Per-bond-type strain
    - /strain_trajectory/eps_A1g, eps_Eg1, eps_Eg2: Mode amplitudes
    - /strain_trajectory/J7_eff, energy: Other observables
    - /metadata/@*: Parameters
    - /metadata/positions: Site positions
    
    Args:
        filepath: Path to trajectory.h5
        
    Returns:
        dict with all trajectory data including:
        - 'spins': (n_times, n_sites, 3) full spin trajectory
        - 't', 'Mx', 'My', 'Mz', 'M_norm', etc.
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for HDF5 reading. Install with: pip install h5py")
    
    result = {}
    
    with h5py.File(filepath, 'r') as f:
        # Read times
        result['t'] = f['/trajectory/times'][:]
        
        # Read magnetization
        M_local = f['/trajectory/magnetization_local'][:]
        result['Mx'] = M_local[:, 0]
        result['My'] = M_local[:, 1]
        result['Mz'] = M_local[:, 2]
        result['M_norm'] = np.linalg.norm(M_local, axis=1)
        
        M_stag = f['/trajectory/magnetization_antiferro'][:]
        result['Mx_stag'] = M_stag[:, 0]
        result['My_stag'] = M_stag[:, 1]
        result['Mz_stag'] = M_stag[:, 2]
        result['M_stag_norm'] = np.linalg.norm(M_stag, axis=1)
        
        # Read full spin configuration trajectory (n_times, n_sites, 3)
        # Like PhononLattice /trajectory/spins
        if '/trajectory/spins' in f:
            result['spins'] = f['/trajectory/spins'][:]  # (n_times, n_sites, 3)
        
        # Read strain mode amplitudes
        result['eps_A1g'] = f['/strain_trajectory/eps_A1g'][:]
        result['eps_Eg1'] = f['/strain_trajectory/eps_Eg1'][:]
        result['eps_Eg2'] = f['/strain_trajectory/eps_Eg2'][:]
        result['eps_Eg_norm'] = np.sqrt(result['eps_Eg1']**2 + result['eps_Eg2']**2)
        result['J7_eff'] = f['/strain_trajectory/J7_eff'][:]
        result['E'] = f['/strain_trajectory/energy'][:]
        
        # Read per-bond-type strain (like PhononLattice per-bond phonon)
        result['has_per_bond_type'] = '/strain_trajectory/eps_xx_0' in f
        
        if result['has_per_bond_type']:
            # Strain tensor components per bond type
            for comp in ['eps_xx', 'eps_yy', 'eps_xy', 'V_xx', 'V_yy', 'V_xy']:
                for b in range(3):
                    key = f'{comp}_{b}'
                    result[key] = f[f'/strain_trajectory/{key}'][:]
            
            # Compute total strain as sum over bond types (for backward compat)
            result['eps_xx'] = result['eps_xx_0'] + result['eps_xx_1'] + result['eps_xx_2']
            result['eps_yy'] = result['eps_yy_0'] + result['eps_yy_1'] + result['eps_yy_2']
            result['eps_xy'] = result['eps_xy_0'] + result['eps_xy_1'] + result['eps_xy_2']
        
        # Read metadata
        if '/metadata' in f:
            meta = f['/metadata']
            result['metadata'] = {}
            for key in meta.attrs.keys():
                result['metadata'][key] = meta.attrs[key]
            
            # Read positions
            if 'positions' in meta:
                result['positions'] = meta['positions'][:]
    
    return result


def read_pump_probe(sample_dir, prefer_hdf5=True):
    """Read complete pump-probe simulation data from a sample directory.
    
    This function reads trajectory data in the following priority:
    1. If prefer_hdf5=True and trajectory.h5 exists, read from HDF5
    2. Otherwise read from text files
    
    Analogous to read_MD_phonon in reader_phonon_lattice.py.
    
    Args:
        sample_dir: Path to sample directory (e.g., 'output/sample_0')
        prefer_hdf5: If True, prefer HDF5 file over text files when available
        
    Returns:
        dict containing all trajectory data and observables, including:
        - 't': Time array
        - 'Mx', 'My', 'Mz', 'M_norm': Magnetization components and norm
        - 'Mx_stag', 'My_stag', 'Mz_stag', 'M_stag_norm': Staggered magnetization
        - 'E': Energy
        - 'eps_A1g', 'eps_Eg1', 'eps_Eg2', 'eps_Eg_norm': Strain mode amplitudes
        - 'J7_eff': Effective J7 coupling
        - 'eps_xx_0', ..., 'eps_xy_2': Per-bond-type strain (if available)
        - 'has_per_bond_type': bool, whether per-bond-type data is available
        - 'metadata': dict of simulation parameters (from HDF5)
    """
    result = {}
    
    # Try HDF5 first if preferred
    hdf5_file = os.path.join(sample_dir, 'trajectory.h5')
    if prefer_hdf5 and os.path.exists(hdf5_file) and HAS_H5PY:
        print(f"  Loading HDF5: {hdf5_file}")
        return read_hdf5_trajectory(hdf5_file)
    
    # Fall back to text files
    result['has_per_bond_type'] = False
    
    # Read magnetization trajectory
    mag_file = os.path.join(sample_dir, 'magnetization.txt')
    if os.path.exists(mag_file):
        mag_data = read_magnetization(mag_file)
        result.update(mag_data)
    else:
        print(f"Warning: {mag_file} not found")
        
    # Read energy trajectory
    energy_file = os.path.join(sample_dir, 'energy.txt')
    if os.path.exists(energy_file):
        energy_data = read_energy(energy_file)
        result['E'] = energy_data['E']
        # Use time from magnetization if available
        if 't' not in result:
            result['t'] = energy_data['t']
    else:
        print(f"Warning: {energy_file} not found")
        
    # Read strain trajectory (mode amplitudes)
    strain_file = os.path.join(sample_dir, 'strain_trajectory.txt')
    if os.path.exists(strain_file):
        strain_data = read_strain_trajectory(strain_file)
        result['eps_A1g'] = strain_data['eps_A1g']
        result['eps_Eg1'] = strain_data['eps_Eg1']
        result['eps_Eg2'] = strain_data['eps_Eg2']
        result['eps_Eg_norm'] = strain_data['eps_Eg_norm']
        result['J7_eff'] = strain_data['J7_eff']
    else:
        print(f"Warning: {strain_file} not found")
    
    # Read per-bond-type strain trajectory (like PhononLattice per-bond phonon)
    strain_bonds_traj_file = os.path.join(sample_dir, 'strain_per_bond_trajectory.txt')
    if os.path.exists(strain_bonds_traj_file):
        strain_bonds_traj = read_strain_per_bond_trajectory(strain_bonds_traj_file)
        # Add per-bond-type data to result
        for key in strain_bonds_traj:
            if key != 't':  # t already in result
                result[key] = strain_bonds_traj[key]
        result['has_per_bond_type'] = True
        
        # Compute total strain as sum over bond types
        result['eps_xx'] = result['eps_xx_0'] + result['eps_xx_1'] + result['eps_xx_2']
        result['eps_yy'] = result['eps_yy_0'] + result['eps_yy_1'] + result['eps_yy_2']
        result['eps_xy'] = result['eps_xy_0'] + result['eps_xy_1'] + result['eps_xy_2']
        
    # Read initial and final spin configurations
    initial_spins_file = os.path.join(sample_dir, 'initial_spins.txt')
    if os.path.exists(initial_spins_file):
        result['initial_spins'] = read_spin_config(initial_spins_file)
        
    final_spins_file = os.path.join(sample_dir, 'final_spins.txt')
    if os.path.exists(final_spins_file):
        result['final_spins'] = read_spin_config(final_spins_file)
        
    # Read per-bond strain state (final state only)
    strain_bonds_file = os.path.join(sample_dir, 'strain_per_bond.txt')
    if os.path.exists(strain_bonds_file):
        result['strain_per_bond'] = read_strain_per_bond(strain_bonds_file)
        
    return result


def read_MD(sample_dir):
    """Alias for read_pump_probe - same output format."""
    return read_pump_probe(sample_dir)


def read_all_samples(output_dir, n_samples=None):
    """Read data from all sample directories.
    
    Args:
        output_dir: Base output directory containing sample_* subdirectories
        n_samples: Maximum number of samples to read (None = all)
        
    Returns:
        list of dicts, one per sample
    """
    samples = []
    sample_idx = 0
    
    while True:
        sample_dir = os.path.join(output_dir, f'sample_{sample_idx}')
        if not os.path.exists(sample_dir):
            break
        if n_samples is not None and sample_idx >= n_samples:
            break
            
        samples.append(read_pump_probe(sample_dir))
        sample_idx += 1
        
    print(f"Read {len(samples)} samples from {output_dir}")
    return samples


# ============================================================================
# GNEB KINETIC BARRIER ANALYSIS
# ============================================================================

def read_mep_result(filepath):
    """Read minimum energy path (MEP) result from GNEB.
    
    Args:
        filepath: Path to mep_Q0.txt or similar MEP file
        
    Returns:
        dict with keys: 'image', 'reaction_coord', 'energy', 'm_3Q', 'm_zigzag', 'f_Eg_amp'
    """
    data = np.loadtxt(filepath)
    return {
        'image': data[:, 0].astype(int),
        'reaction_coord': data[:, 1],
        's': data[:, 1],  # Alias
        'energy': data[:, 2],
        'm_3Q': data[:, 3],
        'm_zigzag': data[:, 4],
        'f_Eg_amp': data[:, 5]
    }


def read_barrier_evolution(filepath):
    """Read barrier evolution vs phonon amplitude from GNEB.
    
    Args:
        filepath: Path to barrier_evolution.txt
        
    Returns:
        dict with keys: 'Q_Eg', 'Delta_E', 'E_saddle', 'E_initial', 'E_final'
    """
    data = np.loadtxt(filepath)
    return {
        'Q_Eg': data[:, 0],
        'Q': data[:, 0],  # Alias
        'Delta_E': data[:, 1],
        'barrier': data[:, 1],  # Alias
        'E_saddle': data[:, 2],
        'E_initial': data[:, 3],
        'E_final': data[:, 4]
    }


def read_barrier_summary(filepath):
    """Read barrier analysis summary.
    
    Args:
        filepath: Path to barrier_summary.txt
        
    Returns:
        dict with summary statistics
    """
    summary = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert to appropriate type
            if value in ['true', 'True', 'TRUE']:
                summary[key] = True
            elif value in ['false', 'False', 'FALSE']:
                summary[key] = False
            else:
                try:
                    summary[key] = float(value)
                except ValueError:
                    summary[key] = value
    return summary


def read_gneb_results(sample_dir):
    """Read all GNEB kinetic barrier analysis results from a sample directory.
    
    Args:
        sample_dir: Path to sample directory (e.g., 'NCTO_Eg_kinetic_barrier/sample_0')
        
    Returns:
        dict containing:
        - 'mep': MEP energy and order parameters at Q=0
        - 'barrier_evolution': Barrier vs phonon amplitude
        - 'summary': Summary statistics
        - 'initial_state': Initial spin configuration
        - 'final_state': Final spin configuration
    """
    result = {}
    
    # Read MEP at Q=0
    mep_file = os.path.join(sample_dir, 'mep_Q0.txt')
    if os.path.exists(mep_file):
        result['mep'] = read_mep_result(mep_file)
        print(f"  Loaded MEP with {len(result['mep']['image'])} images")
    else:
        print(f"  Warning: MEP file not found: {mep_file}")
        result['mep'] = None
    
    # Read barrier evolution
    barrier_file = os.path.join(sample_dir, 'barrier_evolution.txt')
    if os.path.exists(barrier_file):
        result['barrier_evolution'] = read_barrier_evolution(barrier_file)
        print(f"  Loaded barrier evolution with {len(result['barrier_evolution']['Q'])} Q points")
    else:
        print(f"  Warning: Barrier evolution file not found: {barrier_file}")
        result['barrier_evolution'] = None
    
    # Read summary
    summary_file = os.path.join(sample_dir, 'barrier_summary.txt')
    if os.path.exists(summary_file):
        result['summary'] = read_barrier_summary(summary_file)
        print(f"  Loaded barrier summary")
    else:
        result['summary'] = {}
    
    # Read initial and final states
    initial_file = os.path.join(sample_dir, 'triple_q_state.txt')
    if os.path.exists(initial_file):
        result['initial_state'] = read_spin_config(initial_file)
        
    final_file = os.path.join(sample_dir, 'zigzag_state.txt')
    if os.path.exists(final_file):
        result['final_state'] = read_spin_config(final_file)
    
    return result


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_fft_spectrum(result, key='M_norm', window='hann', pad_factor=4):
    """Compute FFT spectrum from a time-domain signal.
    
    Args:
        result: Dictionary from read_pump_probe
        key: Key of signal to transform (default: 'M_norm')
        window: Window function ('hann', 'hamming', 'blackman', None)
        pad_factor: Zero-padding factor for frequency resolution
        
    Returns:
        dict with 'freq' (frequencies in meV) and 'spectrum' (power spectrum)
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for FFT. Install with: pip install scipy")
    
    t = result['t']
    signal = result[key]
    
    # Ensure uniform sampling
    dt = np.mean(np.diff(t))
    
    # Remove DC component
    signal = signal - np.mean(signal)
    
    # Apply window
    N = len(signal)
    if window == 'hann':
        w = np.hanning(N)
    elif window == 'hamming':
        w = np.hamming(N)
    elif window == 'blackman':
        w = np.blackman(N)
    else:
        w = np.ones(N)
    signal = signal * w
    
    # Zero-pad for better frequency resolution
    N_padded = N * pad_factor
    
    # Compute FFT
    spectrum = fft(signal, n=N_padded)
    freq = fftfreq(N_padded, dt)
    
    # Take positive frequencies and power spectrum
    pos_mask = freq >= 0
    freq = freq[pos_mask]
    power = np.abs(spectrum[pos_mask])**2
    
    # Convert frequency from inverse time units to meV
    # f [1/time_unit] = f * hbar/meV [meV/hbar] = f [meV]
    # (time units are already in hbar/meV)
    
    return {
        'freq': freq,
        'spectrum': power,
        'freq_meV': freq  # Already in meV (since time is in hbar/meV)
    }


def find_resonance_peaks(result, key='eps_Eg_norm', prominence=0.1, **kwargs):
    """Find resonance peaks in the frequency spectrum.
    
    Args:
        result: Dictionary from read_pump_probe
        key: Signal key to analyze
        prominence: Minimum prominence for peak detection
        **kwargs: Additional arguments for scipy.signal.find_peaks
        
    Returns:
        dict with 'peak_freqs', 'peak_powers', 'spectrum_result'
    """
    spectrum = compute_fft_spectrum(result, key)
    
    # Normalize spectrum for peak finding
    norm_spectrum = spectrum['spectrum'] / np.max(spectrum['spectrum'])
    
    # Find peaks
    peaks, properties = find_peaks(norm_spectrum, prominence=prominence, **kwargs)
    
    return {
        'peak_freqs': spectrum['freq'][peaks],
        'peak_powers': spectrum['spectrum'][peaks],
        'peak_indices': peaks,
        'spectrum': spectrum
    }


def compute_order_parameter_deviation(result, key='M_stag_norm'):
    """Compute order parameter deviation from equilibrium.
    
    Args:
        result: Dictionary from read_pump_probe
        key: Order parameter key
        
    Returns:
        dict with 't', 'delta_OP' (deviation from initial value)
    """
    t = result['t']
    OP = result[key]
    
    # Find equilibrium value (average before pump)
    t_pump = 0  # Assuming pump arrives at t=0
    pre_pump_mask = t < t_pump
    if np.sum(pre_pump_mask) > 0:
        OP_eq = np.mean(OP[pre_pump_mask])
    else:
        OP_eq = OP[0]
        
    delta_OP = OP - OP_eq
    
    return {
        't': t,
        'delta_OP': delta_OP,
        'OP_eq': OP_eq,
        'OP': OP
    }


# ============================================================================
# KITAEV LOCAL FRAME TRANSFORMATION
# ============================================================================

def get_kitaev_rotation():
    """Get the Kitaev local-to-global rotation matrix R.
    
    The matrix transforms spins from the local Kitaev frame to the global
    Cartesian frame: S_global = R @ S_local
    
    This matches the C++ implementation in strain_phonon_lattice.h
    """
    R = np.array([
        [1.0/np.sqrt(6.0), -1.0/np.sqrt(2.0), 1.0/np.sqrt(3.0)],
        [1.0/np.sqrt(6.0),  1.0/np.sqrt(2.0), 1.0/np.sqrt(3.0)],
        [-2.0/np.sqrt(6.0), 0.0,              1.0/np.sqrt(3.0)]
    ])
    return R


def transform_spins_to_global(spins_local):
    """Transform spins from local Kitaev frame to global Cartesian frame.
    
    Args:
        spins_local: (n_sites, 3) array of spins in local frame
        
    Returns:
        spins_global: (n_sites, 3) array of spins in global frame
    """
    R = get_kitaev_rotation()
    spins_global = spins_local @ R.T  # S_global = R @ S_local, for each row
    return spins_global


# ============================================================================
# STATIC STRUCTURE FACTOR
# ============================================================================

def load_spin_config_with_positions(sample_dir, prefer_global=True):
    """Load spin configuration and positions from a sample directory.
    
    If positions.txt is not found, positions are inferred from the 
    StrainPhononLattice honeycomb geometry.
    
    If spins_global.txt is not found but spins.txt (local frame) exists,
    the spins are transformed to global frame using the Kitaev rotation.
    
    Args:
        sample_dir: Path to sample directory containing spin files
        prefer_global: If True, prefer global frame spins and transform local if needed
        
    Returns:
        dict with 'spins' (n_sites, 3), 'positions' (n_sites, 3), 'frame' ('global'/'local')
    """
    result = {}
    result['frame'] = 'unknown'
    
    # First, try to load global frame spins (most accurate)
    global_spin_file = os.path.join(sample_dir, 'spins_global.txt')
    if prefer_global and os.path.exists(global_spin_file):
        result['spins'] = read_spin_config(global_spin_file)
        result['spin_file'] = 'spins_global.txt'
        result['frame'] = 'global'
        print(f"  Loaded {len(result['spins'])} spins from {result['spin_file']} (global frame)")
    else:
        # Look for spin config files (prefer spins.txt, then try various names)
        spin_files = ['spins.txt', 'final_spins.txt', 'initial_spins.txt']
        for spin_file_name in spin_files:
            spin_file = os.path.join(sample_dir, spin_file_name)
            if os.path.exists(spin_file):
                result['spins'] = read_spin_config(spin_file)
                result['spin_file'] = spin_file_name
                result['frame'] = 'local'
                break
        
        # Also check for temperature-specific spin files like spins_T=0.000010.txt
        if 'spins' not in result:
            import glob
            spin_pattern = os.path.join(sample_dir, 'spins_T=*.txt')
            spin_files_T = sorted(glob.glob(spin_pattern))
            if spin_files_T:
                # Use the lowest temperature one (closest to ground state)
                result['spins'] = read_spin_config(spin_files_T[0])
                result['spin_file'] = os.path.basename(spin_files_T[0])
                result['frame'] = 'local'
        
        if 'spins' not in result:
            raise FileNotFoundError(f"No spin configuration file found in {sample_dir}")
        
        n_sites = len(result['spins'])
        print(f"  Loaded {n_sites} spins from {result['spin_file']} (local Kitaev frame)")
        
        # Transform to global frame if needed
        if prefer_global and result['frame'] == 'local':
            print(f"  Transforming to global Cartesian frame using Kitaev rotation...")
            result['spins'] = transform_spins_to_global(result['spins'])
            result['frame'] = 'global'
    
    n_sites = len(result['spins'])
    
    # Load or generate positions
    pos_file = os.path.join(sample_dir, 'positions.txt')
    result['positions'] = read_positions(pos_file, n_sites)
    
    return result


def compute_static_structure_factor(spins, positions, n_q1=100, n_q2=100,
                                     q1_range=(-2.0, 2.0), q2_range=(-2.0, 2.0)):
    """Compute the static spin structure factor S(q) on a grid.
    
    S^{αβ}(q) = (1/N) Σ_{ij} S_i^α S_j^β exp(-i q·(r_i - r_j))
              = (1/N) |Σ_i S_i^α exp(-i q·r_i)|²  (for α = β)
    
    Uses the honeycomb reciprocal lattice:
    - Real space: a1 = (1, 0), a2 = (0.5, sqrt(3)/2)
    - Reciprocal: b1* = 2π(1, -1/sqrt(3)), b2* = 2π(0, 2/sqrt(3))
    
    Args:
        spins: (n_sites, 3) spin configuration
        positions: (n_sites, 3) site positions
        n_q1: Number of q-points along b1* direction
        n_q2: Number of q-points along b2* direction
        q1_range: (q1_min, q1_max) in units of reciprocal lattice vectors
        q2_range: (q2_min, q2_max) in units of reciprocal lattice vectors
        
    Returns:
        dict with:
        - 'q1': 1D array of q1 values
        - 'q2': 1D array of q2 values
        - 'S_total': (n_q1, n_q2) total structure factor
        - 'S_xx', 'S_yy', 'S_zz': (n_q1, n_q2) diagonal components
    """
    sqrt3 = np.sqrt(3.0)
    
    # Reciprocal lattice vectors
    b1 = np.array([2.0 * np.pi, -2.0 * np.pi / sqrt3, 0.0])
    b2 = np.array([0.0, 4.0 * np.pi / sqrt3, 0.0])
    
    # Generate q-grid
    q1_vals = np.linspace(q1_range[0], q1_range[1], n_q1)
    q2_vals = np.linspace(q2_range[0], q2_range[1], n_q2)
    
    N = len(spins)
    
    # Initialize S(q) arrays
    S_total = np.zeros((n_q1, n_q2))
    S_xx = np.zeros((n_q1, n_q2))
    S_yy = np.zeros((n_q1, n_q2))
    S_zz = np.zeros((n_q1, n_q2))
    
    # Compute S(q) for each q-point
    for i, q1 in enumerate(q1_vals):
        for j, q2 in enumerate(q2_vals):
            # q = q1 * b1* + q2 * b2*
            q = q1 * b1 + q2 * b2
            
            # Compute phase factors for all sites
            phases = np.exp(-1j * (positions @ q))
            
            # Fourier transform of each spin component
            Sq_x = np.sum(spins[:, 0] * phases)
            Sq_y = np.sum(spins[:, 1] * phases)
            Sq_z = np.sum(spins[:, 2] * phases)
            
            # Structure factor components
            S_xx[i, j] = np.abs(Sq_x)**2 / N
            S_yy[i, j] = np.abs(Sq_y)**2 / N
            S_zz[i, j] = np.abs(Sq_z)**2 / N
            S_total[i, j] = S_xx[i, j] + S_yy[i, j] + S_zz[i, j]
    
    return {
        'q1': q1_vals,
        'q2': q2_vals,
        'S_total': S_total,
        'S_xx': S_xx,
        'S_yy': S_yy,
        'S_zz': S_zz,
        'b1': b1,
        'b2': b2
    }


def compute_static_structure_factor_cartesian(spins, positions, n_kx=100, n_ky=100,
                                               kx_range=(-8.0, 8.0), ky_range=(-8.0, 8.0)):
    """Compute the static spin structure factor S(q) on a Cartesian (kx, ky) grid.
    
    S^{αβ}(q) = (1/N) Σ_{ij} S_i^α S_j^β exp(-i q·(r_i - r_j))
              = (1/N) |Σ_i S_i^α exp(-i q·r_i)|²  (for α = β)
    
    Args:
        spins: (n_sites, 3) spin configuration
        positions: (n_sites, 3) site positions
        n_kx: Number of kx points
        n_ky: Number of ky points
        kx_range: (kx_min, kx_max) in units of 1/a (where a=1 is lattice constant)
        ky_range: (ky_min, ky_max) in units of 1/a
        
    Returns:
        dict with:
        - 'kx': 1D array of kx values
        - 'ky': 1D array of ky values
        - 'S_total': (n_kx, n_ky) total structure factor
        - 'S_xx', 'S_yy', 'S_zz': (n_kx, n_ky) diagonal components
    """
    # Generate Cartesian q-grid
    kx_vals = np.linspace(kx_range[0], kx_range[1], n_kx)
    ky_vals = np.linspace(ky_range[0], ky_range[1], n_ky)
    
    N = len(spins)
    
    # Initialize S(q) arrays
    S_total = np.zeros((n_kx, n_ky))
    S_xx = np.zeros((n_kx, n_ky))
    S_yy = np.zeros((n_kx, n_ky))
    S_zz = np.zeros((n_kx, n_ky))
    
    # Compute S(q) for each q-point
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            # q = (kx, ky, 0) in Cartesian coordinates
            q = np.array([kx, ky, 0.0])
            
            # Compute phase factors for all sites
            phases = np.exp(-1j * (positions @ q))
            
            # Fourier transform of each spin component
            Sq_x = np.sum(spins[:, 0] * phases)
            Sq_y = np.sum(spins[:, 1] * phases)
            Sq_z = np.sum(spins[:, 2] * phases)
            
            # Structure factor components
            S_xx[i, j] = np.abs(Sq_x)**2 / N
            S_yy[i, j] = np.abs(Sq_y)**2 / N
            S_zz[i, j] = np.abs(Sq_z)**2 / N
            S_total[i, j] = S_xx[i, j] + S_yy[i, j] + S_zz[i, j]
    
    return {
        'kx': kx_vals,
        'ky': ky_vals,
        'S_total': S_total,
        'S_xx': S_xx,
        'S_yy': S_yy,
        'S_zz': S_zz
    }


def plot_structure_factor(sf_result, output_file=None, component='total', 
                          cmap='hot', vmin=None, vmax=None, log_scale=False):
    """Plot the static spin structure factor.
    
    Args:
        sf_result: Result from compute_static_structure_factor
        output_file: Path to save figure (optional)
        component: 'total', 'xx', 'yy', 'zz', or 'all'
        cmap: Colormap name
        vmin, vmax: Colormap limits (auto if None)
        log_scale: If True, use log scale for colormap
        
    Returns:
        fig, ax (or axes array if component='all')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    q1 = sf_result['q1']
    q2 = sf_result['q2']
    
    if component == 'all':
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for ax, (comp, title) in zip(axes.flat, 
                                     [('S_total', r'$S_{total}(\mathbf{q})$'),
                                      ('S_xx', r'$S_{xx}(\mathbf{q})$'),
                                      ('S_yy', r'$S_{yy}(\mathbf{q})$'),
                                      ('S_zz', r'$S_{zz}(\mathbf{q})$')]):
            data = sf_result[comp]
            if log_scale:
                data = np.log10(data + 1e-10)
            
            im = ax.pcolormesh(q1, q2, data.T, cmap=cmap, shading='auto',
                              vmin=vmin, vmax=vmax)
            ax.set_xlabel(r'$q_1$ (r.l.u.)', fontsize=12)
            ax.set_ylabel(r'$q_2$ (r.l.u.)', fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)
            
            # Mark high-symmetry points
            _mark_high_symmetry_points(ax)
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        return fig, axes
    else:
        fig, ax = plt.subplots(figsize=(8, 7))
        
        comp_key = 'S_total' if component == 'total' else f'S_{component}'
        data = sf_result[comp_key]
        
        if log_scale:
            data = np.log10(data + 1e-10)
        
        im = ax.pcolormesh(q1, q2, data.T, cmap=cmap, shading='auto',
                          vmin=vmin, vmax=vmax)
        ax.set_xlabel(r'$q_1$ (r.l.u.)', fontsize=12)
        ax.set_ylabel(r'$q_2$ (r.l.u.)', fontsize=12)
        
        title_map = {'total': r'$S_{total}(\mathbf{q})$',
                     'xx': r'$S_{xx}(\mathbf{q})$',
                     'yy': r'$S_{yy}(\mathbf{q})$',
                     'zz': r'$S_{zz}(\mathbf{q})$'}
        ax.set_title(f'Static Spin Structure Factor: {title_map.get(component, component)}', fontsize=14)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='S(q)' if not log_scale else 'log₁₀(S(q))')
        
        # Mark high-symmetry points
        _mark_high_symmetry_points(ax)
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        return fig, ax


def _mark_high_symmetry_points(ax):
    """Mark high-symmetry points of the honeycomb BZ on a structure factor plot.
    
    High-symmetry points in reciprocal lattice coordinates (q1, q2):
    - Γ: (0, 0)
    - M: (1/2, 0), (0, 1/2), (1/2, 1/2) and equivalents
    - K: (1/3, 2/3), (2/3, 1/3) and equivalents
    - K': (-1/3, -2/3), (-2/3, -1/3), (2/3, -2/3), (-2/3, 2/3) and equivalents
    """
    from matplotlib.patheffects import withStroke
    
    # Γ point
    ax.plot(0, 0, 'wo', markersize=8, markeredgecolor='k')
    ax.text(0.05, 0.05, r'$\Gamma$', fontsize=10, color='white',
            path_effects=[withStroke(linewidth=2, foreground='black')])
    
    # M points (midpoints of BZ edges)
    m_points = [(0.5, 0), (0, 0.5), (0.5, 0.5), (-0.5, 0), (0, -0.5), (-0.5, -0.5),
                (0.5, -0.5), (-0.5, 0.5)]
    for mp in m_points:
        if ax.get_xlim()[0] <= mp[0] <= ax.get_xlim()[1] and ax.get_ylim()[0] <= mp[1] <= ax.get_ylim()[1]:
            ax.plot(mp[0], mp[1], 'ws', markersize=6, markeredgecolor='k')
    
    # K points (corners of BZ) - both K and K' points
    # K points: (1/3, 2/3), (2/3, 1/3), (-1/3, 1/3), etc.
    # K' points: (2/3, -2/3), (-2/3, 2/3), etc. (related by time-reversal)
    k_points = [
        (1/3, 2/3), (2/3, 1/3), (-1/3, 1/3), (-2/3, -1/3), (1/3, -1/3), (-1/3, -2/3),
        # K' points (time-reversal partners)
        (-1/3, -2/3), (-2/3, -1/3), (1/3, -1/3), (2/3, 1/3), (-1/3, 1/3), (1/3, 2/3),
        # Additional K/K' in extended zone
        (2/3, -2/3), (-2/3, 2/3), (4/3, -1/3), (-4/3, 1/3), (1/3, -4/3), (-1/3, 4/3),
        (4/3, 2/3), (-4/3, -2/3), (2/3, 4/3), (-2/3, -4/3)
    ]
    for kp in k_points:
        if ax.get_xlim()[0] <= kp[0] <= ax.get_xlim()[1] and ax.get_ylim()[0] <= kp[1] <= ax.get_ylim()[1]:
            ax.plot(kp[0], kp[1], 'w^', markersize=6, markeredgecolor='k')


def plot_structure_factor_cartesian(sf_result, output_file=None, component='total', 
                                     cmap='hot', vmin=None, vmax=None, log_scale=False):
    """Plot the static spin structure factor in Cartesian (kx, ky) coordinates.
    
    Args:
        sf_result: Result from compute_static_structure_factor_cartesian
        output_file: Path to save figure (optional)
        component: 'total', 'xx', 'yy', 'zz', or 'all'
        cmap: Colormap name
        vmin, vmax: Colormap limits (auto if None)
        log_scale: If True, use log scale for colormap
        
    Returns:
        fig, ax (or axes array if component='all')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    kx = sf_result['kx']
    ky = sf_result['ky']
    
    if component == 'all':
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for ax, (comp, title) in zip(axes.flat, 
                                     [('S_total', r'$S_{total}(\mathbf{k})$'),
                                      ('S_xx', r'$S_{xx}(\mathbf{k})$'),
                                      ('S_yy', r'$S_{yy}(\mathbf{k})$'),
                                      ('S_zz', r'$S_{zz}(\mathbf{k})$')]):
            data = sf_result[comp]
            if log_scale:
                data = np.log10(data + 1e-10)
            
            im = ax.pcolormesh(kx, ky, data.T, cmap=cmap, shading='auto',
                              vmin=vmin, vmax=vmax)
            ax.set_xlabel(r'$k_x$ (1/a)', fontsize=12)
            ax.set_ylabel(r'$k_y$ (1/a)', fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)
            
            # Mark high-symmetry points in Cartesian coords
            _mark_high_symmetry_points_cartesian(ax)
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        return fig, axes
    else:
        fig, ax = plt.subplots(figsize=(8, 7))
        
        comp_key = 'S_total' if component == 'total' else f'S_{component}'
        data = sf_result[comp_key]
        
        if log_scale:
            data = np.log10(data + 1e-10)
        
        im = ax.pcolormesh(kx, ky, data.T, cmap=cmap, shading='auto',
                          vmin=vmin, vmax=vmax)
        ax.set_xlabel(r'$k_x$ (1/a)', fontsize=12)
        ax.set_ylabel(r'$k_y$ (1/a)', fontsize=12)
        
        title_map = {'total': r'$S_{total}(\mathbf{k})$',
                     'xx': r'$S_{xx}(\mathbf{k})$',
                     'yy': r'$S_{yy}(\mathbf{k})$',
                     'zz': r'$S_{zz}(\mathbf{k})$'}
        ax.set_title(f'Static Spin Structure Factor: {title_map.get(component, component)}', fontsize=14)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='S(k)' if not log_scale else 'log₁₀(S(k))')
        
        # Mark high-symmetry points in Cartesian coords
        _mark_high_symmetry_points_cartesian(ax)
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_file}")
        
        return fig, ax


def _mark_high_symmetry_points_cartesian(ax):
    """Mark high-symmetry points of the honeycomb BZ in Cartesian coordinates.
    
    For honeycomb lattice with a1=(1,0), a2=(0.5, sqrt(3)/2):
    - b1 = 2π(1, -1/sqrt(3)), b2 = 2π(0, 2/sqrt(3))
    
    High-symmetry points in Cartesian (kx, ky):
    - Γ: (0, 0)
    - M: π(1, 1/sqrt(3)), π(1, -1/sqrt(3)), π(0, 2/sqrt(3)), etc.
    - K: (4π/3, 0), (2π/3, 2π/sqrt(3)), etc.
    """
    from matplotlib.patheffects import withStroke
    
    sqrt3 = np.sqrt(3.0)
    
    # Γ point
    ax.plot(0, 0, 'wo', markersize=8, markeredgecolor='k')
    ax.text(0.1, 0.1, r'$\Gamma$', fontsize=10, color='white',
            path_effects=[withStroke(linewidth=2, foreground='black')])
    
    # M points in Cartesian coords
    # M = (1/2)*b1 = π(1, -1/sqrt(3))
    # M = (1/2)*b2 = π(0, 2/sqrt(3))
    # M = (1/2)*(b1+b2) = π(1, 1/sqrt(3))
    m_points_cart = [
        (np.pi, -np.pi/sqrt3), (np.pi, np.pi/sqrt3), (0, 2*np.pi/sqrt3),
        (-np.pi, np.pi/sqrt3), (-np.pi, -np.pi/sqrt3), (0, -2*np.pi/sqrt3)
    ]
    for mp in m_points_cart:
        if ax.get_xlim()[0] <= mp[0] <= ax.get_xlim()[1] and ax.get_ylim()[0] <= mp[1] <= ax.get_ylim()[1]:
            ax.plot(mp[0], mp[1], 'ws', markersize=6, markeredgecolor='k')
    
    # K points in Cartesian coords
    # K = (1/3)*b1 + (2/3)*b2 = 2π(1/3, 1/sqrt(3))
    # K = (2/3)*b1 + (1/3)*b2 = 2π(2/3, 0)
    # K' = (2/3)*b1 - (2/3)*b2 = 2π(2/3, -2/(sqrt(3))) = (4π/3, -4π/(3*sqrt(3)))
    # etc.
    k_points_cart = [
        (4*np.pi/3, 0), (-4*np.pi/3, 0),
        (2*np.pi/3, 2*np.pi/sqrt3), (-2*np.pi/3, -2*np.pi/sqrt3),
        (2*np.pi/3, -2*np.pi/sqrt3), (-2*np.pi/3, 2*np.pi/sqrt3),
        # Additional BZ corners
        (-2*np.pi/3, 2*np.pi/sqrt3), (2*np.pi/3, 2*np.pi/sqrt3),
    ]
    for kp in k_points_cart:
        if ax.get_xlim()[0] <= kp[0] <= ax.get_xlim()[1] and ax.get_ylim()[0] <= kp[1] <= ax.get_ylim()[1]:
            ax.plot(kp[0], kp[1], 'w^', markersize=6, markeredgecolor='k')


def get_high_symmetry_points_rlu():
    """Get inequivalent high-symmetry points in the first Brillouin zone (r.l.u.).
    
    For honeycomb lattice, the first BZ has the following high-symmetry points:
    - Γ (Gamma): (0, 0) - Center of BZ
    - M1, M2, M3: Midpoints of BZ edges (3 inequivalent)
    - K, K': Corners of BZ (2 inequivalent)
    
    In reciprocal lattice units (q1, q2) with b1 = 2π(1, -1/√3), b2 = 2π(0, 2/√3):
    
    Returns:
        dict with point names as keys and (q1, q2) tuples as values
    """
    # All in r.l.u. coordinates
    points = {
        'Γ': (0.0, 0.0),            # BZ center
        'M1': (0.5, 0.0),           # Midpoint along b1
        'M2': (0.0, 0.5),           # Midpoint along b2
        'M3': (0.5, 0.5),           # Midpoint along b1+b2
        'K': (1/3, 2/3),            # BZ corner
        'K\'': (2/3, 1/3),          # BZ corner (distinct from K under C3)
    }
    return points


def find_peak_values_at_high_symmetry(sf_result, tolerance=0.05):
    """Find structure factor values at high-symmetry points by interpolating nearby values.
    
    Args:
        sf_result: Result from compute_static_structure_factor
        tolerance: Radius in r.l.u. to search around high-symmetry point
        
    Returns:
        dict with point names and their S(q) values and components
    """
    q1 = sf_result['q1']
    q2 = sf_result['q2']
    S_total = sf_result['S_total']
    S_xx = sf_result['S_xx']
    S_yy = sf_result['S_yy']
    S_zz = sf_result['S_zz']
    
    hs_points = get_high_symmetry_points_rlu()
    results = {}
    
    for name, (q1_hs, q2_hs) in hs_points.items():
        # Find the grid point closest to the high-symmetry point
        i_closest = np.argmin(np.abs(q1 - q1_hs))
        j_closest = np.argmin(np.abs(q2 - q2_hs))
        
        # Search within tolerance region and find the maximum
        # This accounts for finite grid resolution
        i_min = max(0, i_closest - int(tolerance / (q1[1] - q1[0]) + 1))
        i_max = min(len(q1), i_closest + int(tolerance / (q1[1] - q1[0]) + 1) + 1)
        j_min = max(0, j_closest - int(tolerance / (q2[1] - q2[0]) + 1))
        j_max = min(len(q2), j_closest + int(tolerance / (q2[1] - q2[0]) + 1) + 1)
        
        # Find maximum in local region
        local_S = S_total[i_min:i_max, j_min:j_max]
        if local_S.size == 0:
            continue
            
        local_max_idx = np.unravel_index(np.argmax(local_S), local_S.shape)
        i_peak = i_min + local_max_idx[0]
        j_peak = j_min + local_max_idx[1]
        
        results[name] = {
            'q1': q1[i_peak],
            'q2': q2[j_peak],
            'q1_nominal': q1_hs,
            'q2_nominal': q2_hs,
            'S_total': S_total[i_peak, j_peak],
            'S_xx': S_xx[i_peak, j_peak],
            'S_yy': S_yy[i_peak, j_peak],
            'S_zz': S_zz[i_peak, j_peak],
        }
    
    return results


def plot_peak_strengths(peak_results, output_file=None, show_components=True):
    """Plot a bar chart of structure factor peak strengths at high-symmetry points.
    
    Args:
        peak_results: Result from find_peak_values_at_high_symmetry
        output_file: Path to save figure (optional)
        show_components: If True, show Sxx, Syy, Szz breakdown
        
    Returns:
        fig, ax
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Sort by S_total for better visualization
    sorted_names = sorted(peak_results.keys(), 
                         key=lambda x: peak_results[x]['S_total'], 
                         reverse=True)
    
    n_points = len(sorted_names)
    x = np.arange(n_points)
    width = 0.6 if not show_components else 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if show_components:
        # Stacked/grouped bar chart showing components
        S_total = [peak_results[name]['S_total'] for name in sorted_names]
        S_xx = [peak_results[name]['S_xx'] for name in sorted_names]
        S_yy = [peak_results[name]['S_yy'] for name in sorted_names]
        S_zz = [peak_results[name]['S_zz'] for name in sorted_names]
        
        # Grouped bars
        bars_xx = ax.bar(x - width, S_xx, width, label=r'$S_{xx}$', color='tab:red', alpha=0.8)
        bars_yy = ax.bar(x, S_yy, width, label=r'$S_{yy}$', color='tab:green', alpha=0.8)
        bars_zz = ax.bar(x + width, S_zz, width, label=r'$S_{zz}$', color='tab:blue', alpha=0.8)
        
        # Add total as scatter points on top
        ax.scatter(x, S_total, s=100, c='black', marker='D', zorder=5, label=r'$S_{total}$')
        
        ax.legend(loc='upper right', fontsize=10)
    else:
        # Simple bar chart of total
        S_total = [peak_results[name]['S_total'] for name in sorted_names]
        ax.bar(x, S_total, width, color='steelblue', edgecolor='black')
    
    # Create labels with q values
    labels = []
    for name in sorted_names:
        q1 = peak_results[name]['q1_nominal']
        q2 = peak_results[name]['q2_nominal']
        labels.append(f'{name}\n({q1:.2f},{q2:.2f})')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r'$S(\mathbf{q})$', fontsize=12)
    ax.set_xlabel('High-symmetry point (r.l.u.)', fontsize=12)
    ax.set_title('Structure Factor at High-Symmetry Points (First BZ)', fontsize=14)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for name, xi in zip(sorted_names, x):
        S_val = peak_results[name]['S_total']
        ax.annotate(f'{S_val:.2f}', 
                   xy=(xi, S_val), 
                   xytext=(0, 5), 
                   textcoords='offset points',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig, ax


def print_peak_summary(peak_results):
    """Print a summary table of structure factor peaks at high-symmetry points."""
    print("\n" + "="*80)
    print("Structure Factor at High-Symmetry Points (First Brillouin Zone)")
    print("="*80)
    print(f"{'Point':<8} {'q1 (r.l.u.)':<12} {'q2 (r.l.u.)':<12} {'S_total':<12} {'S_xx':<10} {'S_yy':<10} {'S_zz':<10}")
    print("-"*80)
    
    # Sort by S_total descending
    sorted_names = sorted(peak_results.keys(), 
                         key=lambda x: peak_results[x]['S_total'], 
                         reverse=True)
    
    for name in sorted_names:
        p = peak_results[name]
        print(f"{name:<8} {p['q1']:<12.4f} {p['q2']:<12.4f} {p['S_total']:<12.4f} "
              f"{p['S_xx']:<10.4f} {p['S_yy']:<10.4f} {p['S_zz']:<10.4f}")
    
    print("="*80 + "\n")


def plot_spin_config(spins, positions, output_file=None, title=None, 
                     arrow_scale=0.3, cmap='coolwarm', figsize=(12, 10)):
    """Plot spin configuration on the honeycomb lattice.
    
    Args:
        spins: (N, 3) array of spin vectors
        positions: (N, 2) array of site positions
        output_file: Path to save figure (optional)
        title: Plot title (optional)
        arrow_scale: Scale factor for arrow length
        cmap: Colormap for Sz component coloring
        figsize: Figure size
        
    Returns:
        fig, ax
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract spin components
    Sx, Sy, Sz = spins[:, 0], spins[:, 1], spins[:, 2]
    x, y = positions[:, 0], positions[:, 1]
    
    # Color by Sz component
    norm = plt.Normalize(vmin=-1, vmax=1)
    colors = plt.colormaps.get_cmap(cmap)(norm(Sz))
    
    # Plot site positions
    ax.scatter(x, y, c=Sz, cmap=cmap, s=50, edgecolors='k', linewidths=0.5,
              vmin=-1, vmax=1, zorder=2)
    
    # Plot in-plane spin components as arrows
    scale = arrow_scale
    for i in range(len(spins)):
        # Arrow showing in-plane component (Sx, Sy)
        ax.arrow(x[i], y[i], Sx[i]*scale, Sy[i]*scale,
                head_width=0.08, head_length=0.04, fc=colors[i], ec='k',
                linewidth=0.5, zorder=3)
    
    # Add colorbar for Sz
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(r'$S_z$', fontsize=12)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_aspect('equal')
    ax.set_title(title or 'Spin Configuration (Global Frame)', fontsize=14)
    
    # Add legend for arrow meaning
    ax.text(0.02, 0.98, 'Arrows: in-plane $(S_x, S_y)$\nColor: out-of-plane $S_z$',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig, ax


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_trajectory(result, output_file=None, title=None):
    """Plot magnetization and strain trajectory.
    
    Args:
        result: Dictionary from read_pump_probe
        output_file: Path to save figure (optional)
        title: Plot title (optional)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    t = result['t']
    
    # 1. Magnetization components
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, result['Mx'], 'r-', label=r'$M_x$', alpha=0.7)
    ax1.plot(t, result['My'], 'g-', label=r'$M_y$', alpha=0.7)
    ax1.plot(t, result['Mz'], 'b-', label=r'$M_z$', alpha=0.7)
    ax1.set_xlabel('Time (ℏ/meV)')
    ax1.set_ylabel('Magnetization')
    ax1.legend(loc='best')
    ax1.set_title('Magnetization Components')
    ax1.grid(True, alpha=0.3)
    
    # 2. Magnetization magnitude
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, result['M_norm'], 'k-', label=r'$|M|$')
    ax2.plot(t, result['M_stag_norm'], 'r--', label=r'$|M_{stag}|$')
    ax2.set_xlabel('Time (ℏ/meV)')
    ax2.set_ylabel('|M|')
    ax2.legend(loc='best')
    ax2.set_title('Magnetization Magnitude')
    ax2.grid(True, alpha=0.3)
    
    # 3. Energy
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, result['E'], 'k-')
    ax3.set_xlabel('Time (ℏ/meV)')
    ax3.set_ylabel('Energy')
    ax3.set_title('Total Energy')
    ax3.grid(True, alpha=0.3)
    
    # 4. Strain modes
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, result['eps_A1g'], 'b-', label=r'$\varepsilon_{A1g}$')
    ax4.plot(t, result['eps_Eg1'], 'r-', label=r'$\varepsilon_{Eg1}$', alpha=0.7)
    ax4.plot(t, result['eps_Eg2'], 'g-', label=r'$\varepsilon_{Eg2}$', alpha=0.7)
    ax4.set_xlabel('Time (ℏ/meV)')
    ax4.set_ylabel('Strain')
    ax4.legend(loc='best')
    ax4.set_title('Strain Field Components')
    ax4.grid(True, alpha=0.3)
    
    # 5. Eg magnitude
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t, result['eps_Eg_norm'], 'r-')
    ax5.set_xlabel('Time (ℏ/meV)')
    ax5.set_ylabel(r'$|\varepsilon_{Eg}|$')
    ax5.set_title('Eg Strain Amplitude')
    ax5.grid(True, alpha=0.3)
    
    # 6. Effective J7
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(t, result['J7_eff'], 'b-')
    ax6.set_xlabel('Time (ℏ/meV)')
    ax6.set_ylabel(r'$J_7^{eff}$ (meV)')
    ax6.set_title('Effective Ring Exchange')
    ax6.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.show()


def plot_spectrum(result, keys=['M_norm', 'eps_Eg_norm'], output_file=None):
    """Plot frequency spectra of observables.
    
    Args:
        result: Dictionary from read_pump_probe
        keys: List of observable keys to analyze
        output_file: Path to save figure (optional)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    n_keys = len(keys)
    fig, axes = plt.subplots(n_keys, 1, figsize=(10, 3*n_keys))
    if n_keys == 1:
        axes = [axes]
    
    for ax, key in zip(axes, keys):
        spectrum = compute_fft_spectrum(result, key)
        
        # Only show positive frequencies up to reasonable range
        freq_max = min(50, spectrum['freq'].max())  # meV
        mask = spectrum['freq'] <= freq_max
        
        ax.plot(spectrum['freq'][mask], spectrum['spectrum'][mask])
        ax.set_xlabel('Frequency (meV)')
        ax.set_ylabel('Power')
        ax.set_title(f'Spectrum of {key}')
        ax.set_xlim(0, freq_max)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.show()


def plot_pump_response(result, t_window=(-5, 50), output_file=None):
    """Plot the pump response near t=0.
    
    Args:
        result: Dictionary from read_pump_probe
        t_window: Time window (t_min, t_max)
        output_file: Path to save figure (optional)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    t = result['t']
    mask = (t >= t_window[0]) & (t <= t_window[1])
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Magnetization
    axes[0].plot(t[mask], result['M_norm'][mask], 'k-', label=r'$|M|$')
    axes[0].axvline(0, color='r', linestyle='--', alpha=0.5, label='Pump')
    axes[0].set_ylabel(r'$|M|$')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Strain
    axes[1].plot(t[mask], result['eps_Eg_norm'][mask], 'r-', label=r'$|\varepsilon_{Eg}|$')
    axes[1].axvline(0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_ylabel(r'$|\varepsilon_{Eg}|$')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Energy
    axes[2].plot(t[mask], result['E'][mask], 'b-')
    axes[2].axvline(0, color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time (ℏ/meV)')
    axes[2].set_ylabel('Energy')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.show()


# ============================================================================
# GNEB KINETIC BARRIER PLOTTING
# ============================================================================

def plot_mep_energy_profile(gneb_result, output_file=None):
    """Plot energy profile along the minimum energy path.
    
    Args:
        gneb_result: Dictionary from read_gneb_results
        output_file: Path to save figure (None = display)
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available")
        return
    
    if gneb_result['mep'] is None:
        print("Error: No MEP data available")
        return
    
    mep = gneb_result['mep']
    summary = gneb_result.get('summary', {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    s = mep['reaction_coord']
    E = mep['energy']
    
    # Plot energy profile
    ax.plot(s, E, 'b-o', linewidth=2, markersize=4, label='E(s)')
    
    # Mark initial, saddle, and final points
    ax.plot(s[0], E[0], 'go', markersize=10, label='Initial (triple-Q)')
    ax.plot(s[-1], E[-1], 'ro', markersize=10, label='Final (zigzag)')
    
    # Mark saddle point if available
    if 'saddle_image' in summary:
        saddle_idx = int(summary['saddle_image'])
        if saddle_idx < len(s):
            ax.plot(s[saddle_idx], E[saddle_idx], 'ms', markersize=12, 
                   label=f'Saddle (image {saddle_idx})')
            
            # Draw barrier arrow
            barrier = summary.get('barrier_Q0', E[saddle_idx] - E[0])
            ax.annotate('', xy=(s[saddle_idx], E[saddle_idx]), 
                       xytext=(s[saddle_idx], E[0]),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text(s[saddle_idx] + 0.05, (E[saddle_idx] + E[0])/2, 
                   f'ΔE = {barrier:.2f}',
                   fontsize=12, color='red', weight='bold')
    
    ax.set_xlabel('Reaction Coordinate s', fontsize=14)
    ax.set_ylabel('Energy', fontsize=14)
    ax.set_title('Minimum Energy Path (Q=0)', fontsize=16, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved MEP energy profile to {output_file}")
    else:
        plt.show()


def plot_mep_order_parameters(gneb_result, output_file=None):
    """Plot order parameters along the minimum energy path.
    
    Args:
        gneb_result: Dictionary from read_gneb_results
        output_file: Path to save figure (None = display)
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available")
        return
    
    if gneb_result['mep'] is None:
        print("Error: No MEP data available")
        return
    
    mep = gneb_result['mep']
    summary = gneb_result.get('summary', {})
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    s = mep['reaction_coord']
    
    # Triple-Q order parameter
    axes[0].plot(s, mep['m_3Q'], 'b-o', linewidth=2, markersize=3)
    axes[0].set_ylabel('m_3Q', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Order Parameters Along MEP', fontsize=14, weight='bold')
    
    # Zigzag order parameter
    axes[1].plot(s, mep['m_zigzag'], 'r-o', linewidth=2, markersize=3)
    axes[1].set_ylabel('m_zigzag', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # E_g amplitude
    axes[2].plot(s, mep['f_Eg_amp'], 'g-o', linewidth=2, markersize=3)
    axes[2].set_ylabel('|f_Eg|', fontsize=12)
    axes[2].set_xlabel('Reaction Coordinate s', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    # Mark saddle point on all panels
    if 'saddle_image' in summary:
        saddle_idx = int(summary['saddle_image'])
        if saddle_idx < len(s):
            for ax in axes:
                ax.axvline(s[saddle_idx], color='magenta', linestyle='--', 
                          linewidth=2, alpha=0.7, label='Saddle')
            axes[0].legend(fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved MEP order parameters to {output_file}")
    else:
        plt.show()


def plot_barrier_evolution(gneb_result, output_file=None):
    """Plot barrier height evolution vs phonon amplitude.
    
    Args:
        gneb_result: Dictionary from read_gneb_results
        output_file: Path to save figure (None = display)
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available")
        return
    
    if gneb_result['barrier_evolution'] is None:
        print("Error: No barrier evolution data available")
        return
    
    barrier = gneb_result['barrier_evolution']
    summary = gneb_result.get('summary', {})
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    Q = barrier['Q']
    Delta_E = barrier['barrier']
    
    # Barrier height vs Q
    ax1.plot(Q, Delta_E, 'b-o', linewidth=2, markersize=4)
    ax1.set_ylabel('Barrier Height ΔE', fontsize=14)
    ax1.set_title('Kinetic Barrier Evolution Under E$_g$ Phonon Drive', 
                  fontsize=16, weight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark spinodal point if found
    if 'spinodal_Q' in summary and summary['spinodal_Q'] > 0:
        Q_spinodal = summary['spinodal_Q']
        ax1.axvline(Q_spinodal, color='red', linestyle='--', linewidth=2,
                   label=f'Spinodal Q$_c$ = {Q_spinodal:.3f}')
        ax1.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax1.legend(fontsize=12)
    
    # Annotate Q=0 barrier
    if len(Q) > 0:
        ax1.plot(Q[0], Delta_E[0], 'go', markersize=10, zorder=5)
        ax1.text(Q[0] + 0.05, Delta_E[0], 
                f'ΔE(Q=0) = {Delta_E[0]:.2f}',
                fontsize=11, color='green', weight='bold')
    
    # Energy levels
    ax2.plot(Q, barrier['E_initial'], 'g-', linewidth=2, label='E$_{initial}$ (triple-Q)')
    ax2.plot(Q, barrier['E_saddle'], 'm-', linewidth=2, label='E$_{saddle}$')
    ax2.plot(Q, barrier['E_final'], 'r-', linewidth=2, label='E$_{final}$ (zigzag)')
    ax2.fill_between(Q, barrier['E_initial'], barrier['E_saddle'], 
                     alpha=0.2, color='blue', label='Barrier')
    ax2.set_xlabel('Phonon Amplitude Q$_{Eg}$', fontsize=14)
    ax2.set_ylabel('Energy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Mark spinodal on energy plot too
    if 'spinodal_Q' in summary and summary['spinodal_Q'] > 0:
        ax2.axvline(summary['spinodal_Q'], color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved barrier evolution to {output_file}")
    else:
        plt.show()


def plot_gneb_summary(gneb_result, output_file=None):
    """Plot comprehensive summary of GNEB kinetic barrier analysis.
    
    Args:
        gneb_result: Dictionary from read_gneb_results
        output_file: Path to save figure (None = display)
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not available")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Top left: Energy profile along MEP
    ax1 = fig.add_subplot(gs[0, 0])
    if gneb_result['mep'] is not None:
        mep = gneb_result['mep']
        summary = gneb_result.get('summary', {})
        
        s = mep['reaction_coord']
        E = mep['energy']
        ax1.plot(s, E, 'b-o', linewidth=2, markersize=4)
        ax1.plot(s[0], E[0], 'go', markersize=10, label='Initial')
        ax1.plot(s[-1], E[-1], 'ro', markersize=10, label='Final')
        
        if 'saddle_image' in summary:
            saddle_idx = int(summary['saddle_image'])
            if saddle_idx < len(s):
                ax1.plot(s[saddle_idx], E[saddle_idx], 'ms', markersize=12, label='Saddle')
        
        ax1.set_xlabel('Reaction Coordinate s')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy Profile Along MEP', weight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Top right: Order parameters
    ax2 = fig.add_subplot(gs[0, 1])
    if gneb_result['mep'] is not None:
        mep = gneb_result['mep']
        s = mep['reaction_coord']
        
        ax2.plot(s, mep['m_3Q'], 'b-', linewidth=2, label='m$_{3Q}$')
        ax2.plot(s, mep['m_zigzag'], 'r-', linewidth=2, label='m$_{zigzag}$')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(s, mep['f_Eg_amp'], 'g-', linewidth=2, label='|f$_{Eg}$|', alpha=0.7)
        
        ax2.set_xlabel('Reaction Coordinate s')
        ax2.set_ylabel('m$_{3Q}$, m$_{zigzag}$', color='black')
        ax2_twin.set_ylabel('|f$_{Eg}$|', color='green')
        ax2.set_title('Order Parameters Along MEP', weight='bold')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        ax2.grid(True, alpha=0.3)
    
    # Bottom left: Barrier evolution
    ax3 = fig.add_subplot(gs[1, 0])
    if gneb_result['barrier_evolution'] is not None:
        barrier = gneb_result['barrier_evolution']
        summary = gneb_result.get('summary', {})
        
        Q = barrier['Q']
        Delta_E = barrier['barrier']
        ax3.plot(Q, Delta_E, 'b-o', linewidth=2, markersize=4)
        ax3.axhline(0, color='gray', linestyle=':', linewidth=1)
        
        if 'spinodal_Q' in summary and summary['spinodal_Q'] > 0:
            Q_spinodal = summary['spinodal_Q']
            ax3.axvline(Q_spinodal, color='red', linestyle='--', linewidth=2,
                       label=f'Q$_c$ = {Q_spinodal:.3f}')
            ax3.legend()
        
        ax3.set_xlabel('Phonon Amplitude Q$_{Eg}$')
        ax3.set_ylabel('Barrier Height ΔE')
        ax3.set_title('Barrier vs Phonon Amplitude', weight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Bottom right: Summary text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    if gneb_result.get('summary'):
        summary = gneb_result['summary']
        summary_text = 'GNEB Analysis Summary\\n'
        summary_text += '='*40 + '\\n\\n'
        
        if 'barrier_Q0' in summary:
            summary_text += f"Barrier at Q=0: {summary['barrier_Q0']:.4f}\\n"
        if 'spinodal_Q' in summary:
            if summary['spinodal_Q'] > 0:
                summary_text += f"Spinodal point: Q_c = {summary['spinodal_Q']:.4f}\\n"
            else:
                summary_text += f"Spinodal point: Not found\\n"
        if 'gneb_iterations' in summary:
            summary_text += f"GNEB iterations: {int(summary['gneb_iterations'])}\\n"
        if 'gneb_converged' in summary:
            converged = 'Yes' if summary['gneb_converged'] else 'No'
            summary_text += f"Converged: {converged}\\n"
        if 'saddle_image' in summary:
            summary_text += f"Saddle image: {int(summary['saddle_image'])}\\n"
        
        summary_text += '\\nInitial State (triple-Q):\\n'
        if 'initial_m_3Q' in summary:
            summary_text += f"  m_3Q = {summary['initial_m_3Q']:.4f}\\n"
        if 'initial_m_zigzag' in summary:
            summary_text += f"  m_zigzag = {summary['initial_m_zigzag']:.4f}\\n"
        if 'initial_f_Eg_amp' in summary:
            summary_text += f"  |f_Eg| = {summary['initial_f_Eg_amp']:.4f}\\n"
        
        summary_text += '\\nFinal State (zigzag):\\n'
        if 'final_m_3Q' in summary:
            summary_text += f"  m_3Q = {summary['final_m_3Q']:.4f}\\n"
        if 'final_m_zigzag' in summary:
            summary_text += f"  m_zigzag = {summary['final_m_zigzag']:.4f}\\n"
        if 'final_f_Eg_amp' in summary:
            summary_text += f"  |f_Eg| = {summary['final_f_Eg_amp']:.4f}\\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('GNEB Kinetic Barrier Analysis', fontsize=18, weight='bold', y=0.98)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved GNEB summary to {output_file}")
    else:
        plt.show()


# ============================================================================
# SPIN CONFIGURATION ANIMATION
# ============================================================================

def animate_spin_trajectory(result, output_file='spin_animation.mp4', 
                            fps=10, interval=1, 
                            show_strain=True, show_energy=True, figsize=(12, 10)):
    """Create animation of spin configuration evolution from MD/pump-probe trajectory.
    
    Spins are transformed from local Kitaev frame to global Cartesian frame.
    Shows 2D XY-plane projection with Sz as color.
    
    Args:
        result: Dictionary from read_pump_probe (must contain 'spins' and 'positions')
        output_file: Output filename (supports .mp4, .gif)
        fps: Frames per second
        interval: Use every Nth frame (for long trajectories)
        show_strain: If True, show strain amplitude panel
        show_energy: If True, show energy panel
        figsize: Figure size
    
    Returns:
        matplotlib.animation.FuncAnimation object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for animation. Install with: pip install matplotlib")
    
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from matplotlib.colors import Normalize
    
    # Check required data
    if 'spins' not in result:
        raise ValueError("Spin trajectory not found. Make sure HDF5 file contains /trajectory/spins")
    if 'positions' not in result:
        raise ValueError("Site positions not found. Make sure HDF5 file contains /metadata/positions")
    
    # Kitaev local to global frame transformation
    KITAEV_LOCAL_TO_GLOBAL = np.array([
        [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    ])
    
    S = result['spins']  # (n_steps, n_sites, 3)
    P = result['positions']  # (n_sites, 3)
    T = result['t']
    
    # Strain data
    eps_Eg_norm = result.get('eps_Eg_norm', np.zeros_like(T))
    eps_A1g = result.get('eps_A1g', np.zeros_like(T))
    E = result.get('E', np.zeros_like(T))
    J7_eff = result.get('J7_eff', np.zeros_like(T))
    
    n_steps, n_sites, spin_dim = S.shape
    
    # Transform all spins to global frame
    # S_global[t, i, :] = KITAEV_LOCAL_TO_GLOBAL @ S[t, i, :]
    S_global = np.einsum('ij,tnj->tni', KITAEV_LOCAL_TO_GLOBAL, S)
    
    # Subsample if interval > 1
    frame_indices = list(range(0, n_steps, interval))
    n_frames = len(frame_indices)
    
    print(f"Creating animation: {n_frames} frames from {n_steps} timesteps")
    
    # Count number of bottom panels
    n_bottom_panels = sum([show_strain, show_energy])
    
    # Create figure
    if n_bottom_panels > 0:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1 + n_bottom_panels, 1, height_ratios=[3] + [1]*n_bottom_panels)
        ax_2d = fig.add_subplot(gs[0])
        
        panel_idx = 1
        if show_strain:
            ax_strain = fig.add_subplot(gs[panel_idx])
            panel_idx += 1
        else:
            ax_strain = None
        
        if show_energy:
            ax_energy = fig.add_subplot(gs[panel_idx])
        else:
            ax_energy = None
    else:
        fig, ax_2d = plt.subplots(figsize=(10, 8))
        ax_strain = None
        ax_energy = None
    
    # Get Sz (global) for color mapping
    sz_init = S_global[0, :, 2]
    norm = Normalize(vmin=-1, vmax=1)
    
    # Initialize 2D plot (XY projection in global frame)
    scatter = ax_2d.scatter(P[:, 0], P[:, 1], c=sz_init, cmap='coolwarm', 
                            norm=norm, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    # Normalize in-plane spin components for arrow direction
    sx, sy = S_global[0, :, 0], S_global[0, :, 1]
    in_plane_mag = np.sqrt(sx**2 + sy**2)
    in_plane_mag = np.where(in_plane_mag > 1e-10, in_plane_mag, 1.0)
    sx_norm, sy_norm = sx / in_plane_mag, sy / in_plane_mag
    
    quiver = ax_2d.quiver(P[:, 0], P[:, 1], sx_norm, sy_norm, sz_init,
                          cmap='coolwarm', norm=norm,
                          scale=1.2, scale_units='xy', angles='xy',
                          pivot='middle', width=0.008)
    cbar = plt.colorbar(quiver, ax=ax_2d, shrink=0.8)
    cbar.set_label(r'$S_z^{global}$', fontsize=12)
    
    ax_2d.set_xlabel('x (global)', fontsize=12)
    ax_2d.set_ylabel('y (global)', fontsize=12)
    title_2d = ax_2d.set_title(f'Spin Configuration, t = {T[0]:.3f}', fontsize=12)
    ax_2d.set_aspect('equal')
    ax_2d.grid(True, alpha=0.3)
    
    # Initialize strain plot
    if ax_strain is not None:
        ax_strain.plot(T, eps_Eg_norm, 'r-', alpha=0.3, label=r'$|\varepsilon_{Eg}|$')
        ax_strain.plot(T, eps_A1g, 'b-', alpha=0.3, label=r'$\varepsilon_{A1g}$')
        line_Eg, = ax_strain.plot([], [], 'r-', lw=2)
        line_A1g, = ax_strain.plot([], [], 'b-', lw=2)
        vline_strain = ax_strain.axvline(T[0], color='k', linestyle='--', lw=1)
        ax_strain.set_xlabel('Time (ℏ/meV)')
        ax_strain.set_ylabel('Strain Amplitude')
        ax_strain.legend(loc='upper right', fontsize=8)
        ax_strain.set_xlim(T[0], T[-1])
        ax_strain.grid(True, alpha=0.3)
    else:
        line_Eg = line_A1g = vline_strain = None
    
    # Initialize energy plot
    if ax_energy is not None:
        ax_energy.plot(T, E, 'k-', alpha=0.3, label='Energy')
        line_E, = ax_energy.plot([], [], 'k-', lw=2)
        vline_energy = ax_energy.axvline(T[0], color='k', linestyle='--', lw=1)
        # Add J7_eff on twin axis
        ax_energy_twin = ax_energy.twinx()
        ax_energy_twin.plot(T, J7_eff, 'b-', alpha=0.3)
        line_J7, = ax_energy_twin.plot([], [], 'b-', lw=2)
        ax_energy_twin.set_ylabel(r'$J_7^{eff}$', color='blue')
        ax_energy.set_xlabel('Time (ℏ/meV)')
        ax_energy.set_ylabel('Energy')
        ax_energy.legend(loc='upper right', fontsize=8)
        ax_energy.set_xlim(T[0], T[-1])
        ax_energy.grid(True, alpha=0.3)
    else:
        line_E = vline_energy = line_J7 = ax_energy_twin = None
    
    plt.tight_layout()
    
    def update(frame_idx):
        nonlocal quiver
        
        i = frame_indices[frame_idx]
        
        # Get global spins at this timestep
        sx, sy, sz = S_global[i, :, 0], S_global[i, :, 1], S_global[i, :, 2]
        
        # Normalize in-plane components
        in_plane_mag = np.sqrt(sx**2 + sy**2)
        in_plane_mag = np.where(in_plane_mag > 1e-10, in_plane_mag, 1.0)
        sx_norm, sy_norm = sx / in_plane_mag, sy / in_plane_mag
        
        # Update quiver - need to recreate
        quiver.remove()
        quiver = ax_2d.quiver(P[:, 0], P[:, 1], sx_norm, sy_norm, sz,
                              cmap='coolwarm', norm=norm,
                              scale=1.2, scale_units='xy', angles='xy',
                              pivot='middle', width=0.008)
        
        # Update scatter colors
        scatter.set_array(sz)
        
        title_2d.set_text(f'Spin Configuration, t = {T[i]:.3f}')
        
        # Update strain lines
        if ax_strain is not None and line_Eg is not None:
            line_Eg.set_data(T[:i+1], eps_Eg_norm[:i+1])
            line_A1g.set_data(T[:i+1], eps_A1g[:i+1])
            vline_strain.set_xdata([T[i], T[i]])
        
        # Update energy lines
        if ax_energy is not None and line_E is not None:
            line_E.set_data(T[:i+1], E[:i+1])
            vline_energy.set_xdata([T[i], T[i]])
            if line_J7 is not None:
                line_J7.set_data(T[:i+1], J7_eff[:i+1])
        
        return [quiver, scatter, title_2d]
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    
    # Save animation
    if output_file:
        print(f"Saving animation to {output_file}...")
        if output_file.endswith('.gif'):
            writer = PillowWriter(fps=fps)
        else:
            writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_file, writer=writer)
        print(f"Animation saved to {output_file}")
    
    return anim


def plot_spin_snapshot(result, time_index=0, output_file=None, figsize=(10, 8)):
    """Plot a single snapshot of spin configuration from trajectory.
    
    Spins are transformed from local Kitaev frame to global Cartesian frame.
    Shows 2D XY-plane projection with Sz as color.
    
    Args:
        result: Dictionary from read_pump_probe (must contain 'spins' and 'positions')
        time_index: Index in the time array (or -1 for last frame)
        output_file: If provided, save figure to this file
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    from matplotlib.colors import Normalize
    
    # Check required data
    if 'spins' not in result:
        raise ValueError("Spin trajectory not found")
    if 'positions' not in result:
        raise ValueError("Site positions not found")
    
    # Kitaev local to global frame transformation
    KITAEV_LOCAL_TO_GLOBAL = np.array([
        [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    ])
    
    S = result['spins']  # (n_steps, n_sites, 3)
    P = result['positions']  # (n_sites, 3)
    T = result['t']
    
    # Get the snapshot
    if time_index < 0:
        time_index = len(T) + time_index
    
    # Transform to global frame
    S_global = KITAEV_LOCAL_TO_GLOBAL @ S[time_index].T  # (3, n_sites)
    S_global = S_global.T  # (n_sites, 3)
    
    sx, sy, sz = S_global[:, 0], S_global[:, 1], S_global[:, 2]
    
    # Normalize in-plane components
    in_plane_mag = np.sqrt(sx**2 + sy**2)
    in_plane_mag = np.where(in_plane_mag > 1e-10, in_plane_mag, 1.0)
    sx_norm, sy_norm = sx / in_plane_mag, sy / in_plane_mag
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    norm = Normalize(vmin=-1, vmax=1)
    
    # Scatter plot with color = Sz
    scatter = ax.scatter(P[:, 0], P[:, 1], c=sz, cmap='coolwarm', 
                         norm=norm, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    # Quiver plot for in-plane components
    quiver = ax.quiver(P[:, 0], P[:, 1], sx_norm, sy_norm, sz,
                       cmap='coolwarm', norm=norm,
                       scale=1.2, scale_units='xy', angles='xy',
                       pivot='middle', width=0.008)
    
    cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
    cbar.set_label(r'$S_z^{global}$', fontsize=12)
    
    ax.set_xlabel('x (global)', fontsize=12)
    ax.set_ylabel('y (global)', fontsize=12)
    ax.set_title(f'Spin Configuration, t = {T[time_index]:.3f}', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.show()
    return fig


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Command-line interface for strain lattice analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze StrainPhononLattice simulation data')
    parser.add_argument('directory', type=str,
                        help='Path to sample directory or output directory')
    parser.add_argument('--mode', choices=['pp', 'md', 'all', 'static', 'gneb'], default='pp',
                        help='Analysis mode: pp=pump-probe, md=MD, all=all samples, static=spin config + structure factor, gneb=kinetic barrier')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    parser.add_argument('--spectrum', action='store_true',
                        help='Compute and plot frequency spectrum')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file prefix for plots')
    parser.add_argument('--sample', type=int, default=0,
                        help='Sample index for single-sample analysis')
    # Animation options
    parser.add_argument('--animate', action='store_true',
                        help='Create spin configuration animation')
    parser.add_argument('--animate-output', type=str, default='spin_animation.mp4',
                        help='Animation output filename (.mp4 or .gif)')
    parser.add_argument('--animate-fps', type=int, default=10,
                        help='Animation frames per second')
    parser.add_argument('--animate-interval', type=int, default=1,
                        help='Use every Nth frame in animation')
    # Spin snapshot option
    parser.add_argument('--snapshot', type=int, default=None,
                        help='Plot spin snapshot at given time index (-1 for last frame)')
    # Structure factor options
    parser.add_argument('--sf-nq', type=int, default=100,
                        help='Number of q-points per direction for structure factor')
    parser.add_argument('--sf-range', type=float, default=2.0,
                        help='q-range in reciprocal lattice units (default: 2.0)')
    parser.add_argument('--sf-component', type=str, default='all',
                        choices=['all', 'total', 'xx', 'yy', 'zz'],
                        help='Structure factor component to plot')
    parser.add_argument('--sf-log', action='store_true',
                        help='Use log scale for structure factor plot')
    parser.add_argument('--sf-no-cartesian', action='store_true',
                        help='Skip Cartesian (kx, ky) structure factor plot (included by default)')
    parser.add_argument('--sf-krange', type=float, default=8.0,
                        help='k-range for Cartesian plot (default: 8.0)')
    
    args = parser.parse_args()
    
    # Handle static analysis mode (spin config + structure factor)
    if args.mode == 'static':
        # These modes work directly with spin config, not trajectories
        sample_dir = args.directory
        
        # Check if this is a sample directory
        if not (os.path.exists(os.path.join(sample_dir, 'spins.txt')) or 
                os.path.exists(os.path.join(sample_dir, 'spins_T=0.000010.txt')) or
                len([f for f in os.listdir(sample_dir) if f.startswith('spins_T=')]) > 0):
            # Try as output directory
            sample_dir = os.path.join(args.directory, f'sample_{args.sample}')
        
        if not os.path.exists(sample_dir):
            print(f"Error: Directory not found: {sample_dir}")
            return
        
        # Set default output prefix to sample directory if not specified
        output_base = args.output if args.output else os.path.join(sample_dir, 'analysis')
        
        # Find all spin configuration files to analyze
        import glob
        spin_files_to_analyze = []
        
        # Check for main spins.txt
        if os.path.exists(os.path.join(sample_dir, 'spins.txt')):
            spin_files_to_analyze.append(('spins.txt', ''))
        
        # Check for temperature-specific files (spins_T=*.txt)
        spin_T_pattern = os.path.join(sample_dir, 'spins_T=*.txt')
        spin_T_files = sorted(glob.glob(spin_T_pattern))
        for f in spin_T_files:
            basename = os.path.basename(f)
            # Extract temperature label for output naming
            # e.g., spins_T=0.000010.txt -> _T0.000010
            temp_label = basename.replace('spins_T=', '_T').replace('.txt', '')
            spin_files_to_analyze.append((basename, temp_label))
        
        if not spin_files_to_analyze:
            print(f"Error: No spin configuration files found in {sample_dir}")
            return
        
        print(f"Found {len(spin_files_to_analyze)} spin configuration file(s) to analyze")
        
        # Analyze each spin file
        for spin_filename, label_suffix in spin_files_to_analyze:
            output_prefix = f"{output_base}{label_suffix}"
            spin_filepath = os.path.join(sample_dir, spin_filename)
            
            print(f"\n{'='*80}")
            print(f"Analyzing: {spin_filename}")
            print(f"{'='*80}")
            
            # Load spins
            spins_local = read_spin_config(spin_filepath)
            n_sites = len(spins_local)
            print(f"  Loaded {n_sites} spins (local Kitaev frame)")
            
            # Load or generate positions (only need to do once, reuse for all files)
            pos_file = os.path.join(sample_dir, 'positions.txt')
            positions = read_positions(pos_file, n_sites)
            
            # Transform to global frame
            print(f"  Transforming to global Cartesian frame...")
            spins_global = transform_spins_to_global(spins_local)
            
            print(f"\nSpin configuration summary:")
            print(f"  Number of sites: {n_sites}")
            print(f"  Global frame spin magnitude range: [{np.linalg.norm(spins_global, axis=1).min():.4f}, "
                  f"{np.linalg.norm(spins_global, axis=1).max():.4f}]")
            print(f"  Local frame spin magnitude range: [{np.linalg.norm(spins_local, axis=1).min():.4f}, "
                  f"{np.linalg.norm(spins_local, axis=1).max():.4f}]")
            
            # Plot both local and global spin configurations
            title_suffix = f" ({spin_filename})" if label_suffix else ""
            
            output_file_global = f"{output_prefix}_spin_config_global.png"
            plot_spin_config(spins_global, positions, output_file=output_file_global,
                           title=f'Spin Configuration (Global Cartesian Frame){title_suffix}')
            
            output_file_local = f"{output_prefix}_spin_config_local.png"
            plot_spin_config(spins_local, positions, output_file=output_file_local,
                           title=f'Spin Configuration (Local Kitaev Frame){title_suffix}')
            
            # Use global frame spins for structure factor
            spins = spins_global
            
            # Compute and plot structure factor in r.l.u.
            print(f"\nComputing structure factor on {args.sf_nq}x{args.sf_nq} grid (r.l.u.)...")
            sf_result = compute_static_structure_factor(
                spins, positions, 
                n_q1=args.sf_nq, n_q2=args.sf_nq,
                q1_range=(-args.sf_range, args.sf_range),
                q2_range=(-args.sf_range, args.sf_range)
            )
            
            # Find max S(q) locations
            S_total = sf_result['S_total']
            max_idx = np.unravel_index(np.argmax(S_total), S_total.shape)
            q1_max = sf_result['q1'][max_idx[0]]
            q2_max = sf_result['q2'][max_idx[1]]
            print(f"\nStructure factor max at q = ({q1_max:.3f}, {q2_max:.3f}) r.l.u.")
            print(f"  S_total(q_max) = {S_total.max():.4f}")
            
            output_file = f"{output_prefix}_structure_factor.png"
            plot_structure_factor(sf_result, output_file=output_file,
                                 component=args.sf_component, log_scale=args.sf_log)
            
            # Also compute and plot in Cartesian coordinates (default)
            if not args.sf_no_cartesian:
                print(f"\nComputing structure factor on {args.sf_nq}x{args.sf_nq} grid (Cartesian kx, ky)...")
                sf_cart = compute_static_structure_factor_cartesian(
                    spins, positions,
                    n_kx=args.sf_nq, n_ky=args.sf_nq,
                    kx_range=(-args.sf_krange, args.sf_krange),
                    ky_range=(-args.sf_krange, args.sf_krange)
                )
                
                # Find max in Cartesian
                S_total_cart = sf_cart['S_total']
                max_idx_cart = np.unravel_index(np.argmax(S_total_cart), S_total_cart.shape)
                kx_max = sf_cart['kx'][max_idx_cart[0]]
                ky_max = sf_cart['ky'][max_idx_cart[1]]
                print(f"\nStructure factor max at k = ({kx_max:.3f}, {ky_max:.3f}) [1/a]")
                print(f"  S_total(k_max) = {S_total_cart.max():.4f}")
                
                output_file_cart = f"{output_prefix}_structure_factor_cartesian.png"
                plot_structure_factor_cartesian(sf_cart, output_file=output_file_cart,
                                               component=args.sf_component, log_scale=args.sf_log)
            
            # Find and plot peak strengths at high-symmetry points
            print(f"\nAnalyzing structure factor at high-symmetry points...")
            peak_results = find_peak_values_at_high_symmetry(sf_result)
            print_peak_summary(peak_results)
            
            output_file_peaks = f"{output_prefix}_peak_strengths.png"
            plot_peak_strengths(peak_results, output_file=output_file_peaks, show_components=True)
        
        print(f"\n{'='*80}")
        print(f"Analysis complete. Processed {len(spin_files_to_analyze)} spin configuration(s).")
        print(f"{'='*80}")
        
        return
    
    elif args.mode == 'gneb':
        # GNEB kinetic barrier analysis mode
        process_gneb_mode(args)
        return
    
    # Determine sample directory
    if os.path.exists(os.path.join(args.directory, 'magnetization.txt')):
        # Direct sample directory
        sample_dir = args.directory
    else:
        # Output directory with samples
        sample_dir = os.path.join(args.directory, f'sample_{args.sample}')
        
    if not os.path.exists(sample_dir):
        print(f"Error: Directory not found: {sample_dir}")
        return
    
    # Read data
    print(f"Reading data from {sample_dir}...")
    result = read_pump_probe(sample_dir)
    
    # Print summary
    print(f"\nData summary:")
    print(f"  Time range: {result['t'].min():.2f} to {result['t'].max():.2f}")
    print(f"  Timesteps: {len(result['t'])}")
    print(f"  Energy: {result['E'].min():.2f} to {result['E'].max():.2f}")
    print(f"  |M| range: {result['M_norm'].min():.6f} to {result['M_norm'].max():.6f}")
    print(f"  |ε_Eg| range: {result['eps_Eg_norm'].min():.4f} to {result['eps_Eg_norm'].max():.4f}")
    
    if 'spins' in result:
        print(f"  Spin trajectory shape: {result['spins'].shape}")
    
    # Plot trajectory
    if args.plot:
        output_file = f"{args.output}_trajectory.png" if args.output else None
        plot_trajectory(result, output_file=output_file)
        
        output_file2 = f"{args.output}_pump_response.png" if args.output else None
        plot_pump_response(result, output_file=output_file2)
    
    # Compute spectrum
    if args.spectrum:
        output_file = f"{args.output}_spectrum.png" if args.output else None
        plot_spectrum(result, output_file=output_file)
        
        # Find peaks
        peaks = find_resonance_peaks(result, key='eps_Eg_norm')
        print(f"\nResonance peaks in strain spectrum:")
        for f, p in zip(peaks['peak_freqs'][:5], peaks['peak_powers'][:5]):
            print(f"  {f:.3f} meV (power: {p:.2e})")
    
    # Create spin configuration animation
    if args.animate:
        if 'spins' not in result:
            print("\nWarning: Spin trajectory not found in data. Animation skipped.")
            print("  Make sure the simulation ran with HDF5 output enabled.")
        else:
            output_file = args.animate_output
            if args.output:
                # Use output prefix if specified
                ext = os.path.splitext(args.animate_output)[1]
                output_file = f"{args.output}_animation{ext}"
            
            print(f"\nCreating spin animation with {result['spins'].shape[0]} frames...")
            animate_spin_trajectory(
                result, 
                output_file=output_file,
                fps=args.animate_fps,
                interval=args.animate_interval,
                show_strain=True,
                show_energy=True
            )
    
    # Plot single spin snapshot
    if args.snapshot is not None:
        if 'spins' not in result:
            print("\nWarning: Spin trajectory not found in data. Snapshot skipped.")
        else:
            output_file = None
            if args.output:
                output_file = f"{args.output}_snapshot_t{args.snapshot}.png"
            plot_spin_snapshot(result, time_index=args.snapshot, output_file=output_file)


def process_gneb_mode(args):
    """Process GNEB kinetic barrier analysis mode."""
    print(f"\n{'='*70}")
    print("GNEB Kinetic Barrier Analysis")
    print(f"{'='*70}")
    
    # Determine sample directory
    if os.path.exists(os.path.join(args.directory, 'mep_Q0.txt')):
        # Direct sample directory
        sample_dir = args.directory
    else:
        # Output directory with samples
        sample_dir = os.path.join(args.directory, f'sample_{args.sample}')
    
    if not os.path.exists(sample_dir):
        print(f"Error: Directory not found: {sample_dir}")
        print("Expected to find GNEB output files:")
        print("  - mep_Q0.txt")
        print("  - barrier_evolution.txt")
        print("  - barrier_summary.txt")
        return
    
    # Read GNEB results
    print(f"\\nReading GNEB results from {sample_dir}...")
    gneb_result = read_gneb_results(sample_dir)
    
    if gneb_result['mep'] is None and gneb_result['barrier_evolution'] is None:
        print("\\nError: No GNEB data found in directory")
        print("Expected files: mep_Q0.txt, barrier_evolution.txt")
        return
    
    # Print summary
    print(f"\\n{'-'*70}")
    print("Summary Statistics")
    print(f"{'-'*70}")
    if gneb_result.get('summary'):
        summary = gneb_result['summary']
        
        if 'barrier_Q0' in summary:
            print(f"Barrier at Q=0:        {summary['barrier_Q0']:.4f}")
        if 'spinodal_Q' in summary:
            if summary['spinodal_Q'] > 0:
                print(f"Spinodal point Q_c:    {summary['spinodal_Q']:.4f}")
                print(f"  → Barrier vanishes at Q > {summary['spinodal_Q']:.4f}")
                print(f"  → Deterministic switching possible!")
            else:
                print(f"Spinodal point Q_c:    Not found")
                print(f"  → Barrier remains positive at Q_max")
        if 'gneb_converged' in summary:
            converged = 'Yes' if summary['gneb_converged'] else 'No'
            print(f"GNEB converged:        {converged}")
        if 'gneb_iterations' in summary:
            print(f"GNEB iterations:       {int(summary['gneb_iterations'])}")
        if 'saddle_image' in summary:
            print(f"Saddle image index:    {int(summary['saddle_image'])}")
        
        print(f"\\n{'-'*70}")
        print("Initial State (triple-Q)")
        print(f"{'-'*70}")
        if 'initial_m_3Q' in summary:
            print(f"m_3Q       = {summary['initial_m_3Q']:.4f}")
        if 'initial_m_zigzag' in summary:
            print(f"m_zigzag   = {summary['initial_m_zigzag']:.4f}")
        if 'initial_f_Eg_amp' in summary:
            print(f"|f_Eg|     = {summary['initial_f_Eg_amp']:.4f}")
        
        print(f"\\n{'-'*70}")
        print("Final State (zigzag)")
        print(f"{'-'*70}")
        if 'final_m_3Q' in summary:
            print(f"m_3Q       = {summary['final_m_3Q']:.4f}")
        if 'final_m_zigzag' in summary:
            print(f"m_zigzag   = {summary['final_m_zigzag']:.4f}")
        if 'final_f_Eg_amp' in summary:
            print(f"|f_Eg|     = {summary['final_f_Eg_amp']:.4f}")
    
    # Generate plots
    if args.plot or args.output:
        print(f"\\n{'='*70}")
        print("Generating Plots")
        print(f"{'='*70}")
        
        output_dir = args.output if args.output else '.'
        if args.output and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Comprehensive summary
        summary_file = os.path.join(output_dir, 'gneb_summary.png') if args.output else None
        print("  - Comprehensive summary plot...")
        plot_gneb_summary(gneb_result, output_file=summary_file)
        
        # Individual plots
        if gneb_result['mep'] is not None:
            energy_file = os.path.join(output_dir, 'mep_energy_profile.png') if args.output else None
            print("  - MEP energy profile...")
            plot_mep_energy_profile(gneb_result, output_file=energy_file)
            
            order_file = os.path.join(output_dir, 'mep_order_parameters.png') if args.output else None
            print("  - MEP order parameters...")
            plot_mep_order_parameters(gneb_result, output_file=order_file)
        
        if gneb_result['barrier_evolution'] is not None:
            barrier_file = os.path.join(output_dir, 'barrier_evolution.png') if args.output else None
            print("  - Barrier evolution...")
            plot_barrier_evolution(gneb_result, output_file=barrier_file)
        
        if args.output:
            print(f"\\nAll plots saved to {output_dir}/")
    
    print(f"\\n{'='*70}")
    print("Analysis Complete")
    print(f"{'='*70}\\n")


if __name__ == '__main__':
    main()
