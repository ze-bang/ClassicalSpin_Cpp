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
    parser.add_argument('--mode', choices=['pp', 'md', 'all'], default='pp',
                        help='Analysis mode: pp=pump-probe, md=MD, all=all samples')
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
    
    args = parser.parse_args()
    
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


if __name__ == '__main__':
    main()
