"""
Phonon Lattice Analysis Tools
==============================

This module provides analysis functionalities for spin-phonon coupled honeycomb 
lattice simulations (NCTO/Na2Co2TeO6 system), including:

1. read_MD_tot(dir)           - Aggregate DSSF from multiple MD runs
2. read_2D_nonlinear_tot(dir) - Aggregate 2D nonlinear spectroscopy from multiple runs  
3. parse_spin_config(dir)     - Parse and visualize spin configurations

The PhononLattice simulations track:
- Spin dynamics: magnetization M(t), staggered magnetization M_stag(t)
- Phonon dynamics: E1 mode (Qx_E1, Qy_E1), E2 mode (Qx_E2, Qy_E2), A1 mode (Q_A1)
- Energy evolution during MD

HDF5 output formats:
- trajectory.h5: MD trajectory data (time, M, Q, energy)
- pump_probe_spectroscopy.h5: 2DCS data (M0, M1, M01 trajectories)
- annealing.h5: Simulated annealing thermodynamic data
"""

import h5py
import numpy as np
from opt_einsum import contract
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.colors import PowerNorm, LogNorm

# ============================================================================
# CONSTANTS AND GLOBAL VARIABLES
# ============================================================================

# Unit conversion: simulation time (hbar/meV) to seconds
# hbar = 6.58211957e-13 eV·s = 6.58211957e-10 meV·s
# 1 simulation time unit = hbar / (1 meV) = 6.58211957e-13 s = 0.658211957 ps
HBAR_OVER_MEV = 6.58211957e-13  # seconds per simulation time unit
HBAR_OVER_MEV_PS = 0.658211957  # picoseconds per simulation time unit

# Honeycomb lattice local frame transformation (Kitaev model)
kitaevLocal = np.array([
    [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
    [-1/np.sqrt(2), 1/np.sqrt(2), 0],
    [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
])

# Reciprocal space basis for honeycomb lattice
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
    return np.real(result)

def SSSF2D(S, P, nK, dir, gb=False):
    """Compute 2D Static Spin Structure Factor.
    
    Args:
        S: Spin configuration
        P: Atomic positions
        nK: Grid size (nK x nK)
        dir: Output directory
        gb: Use global frame
    
    Returns:
        SSSF: (nK, nK, 3, 3)
    """
    H = np.linspace(0, 1, nK)
    L = np.linspace(0, 1, nK)
    A, B = np.meshgrid(H, L)
    K = hhknk(A, B).reshape((nK * nK, 3))
    SSSF = SSSF_q(K, S, P, gb)
    SSSF = SSSF.reshape((nK, nK, 3, 3))
    return SSSF

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

def SSSFGraph2D(A, B, d1, filename):
    """Plot 2D Static Structure Factor with Brillouin zone."""
    reciprocal_basis = honeycomb_reciprocal_basis()
    b1 = reciprocal_basis[0]
    b2 = reciprocal_basis[1]
    
    bz_vertices = np.array([
        b1 * (-1/3) + b2 * (1/3),
        b1 * (1/3) + b2 * (2/3),
        b1 * (2/3) + b2 * (1/3),
        b1 * (1/3) + b2 * (-1/3),
        b1 * (-1/3) + b2 * (-2/3),
        b1 * (-2/3) + b2 * (-1/3),
        b1 * (-1/3) + b2 * (1/3),
    ])
    
    gamma_point = np.array([0, 0, 0])
    k_point = b1 * (-1/3) + b2 * (1/3)
    m_point = b1 * 0 + b2 * (1/2)
    
    bz_vertices_plot = bz_vertices[:, :2]
    gamma_plot = gamma_point[:2]
    k_plot = k_point[:2]
    m_plot = m_point[:2]
    
    plt.plot(bz_vertices_plot[:, 0], bz_vertices_plot[:, 1], 'w--', lw=1.5)
    plt.scatter([gamma_plot[0], k_plot[0], m_plot[0]], 
                [gamma_plot[1], k_plot[1], m_plot[1]], c='white', s=50, zorder=5)
    plt.text(gamma_plot[0] + 0.02, gamma_plot[1] + 0.02, r'$\Gamma$', color='white', fontsize=14)
    plt.text(k_plot[0] + 0.02, k_plot[1] + 0.02, 'K', color='white', fontsize=14)
    plt.text(m_plot[0] + 0.02, m_plot[1] + 0.02, 'M', color='white', fontsize=14)
    
    plt.pcolormesh(A, B, d1)
    plt.colorbar()
    plt.ylabel(r'$K_y$')
    plt.xlabel(r'$K_x$')
    plt.savefig(filename + ".pdf")
    plt.clf()

# ============================================================================
# HELPER FUNCTIONS - I/O for PhononLattice
# ============================================================================

def compute_phonon_energy_breakdown(result):
    """Compute detailed phonon energy breakdown from trajectory data.
    
    Computes kinetic and potential energy for each phonon mode:
    - E1 mode (IR active): KE = 0.5*(Vx^2 + Vy^2), PE = 0.5*omega_E1^2*(Qx^2 + Qy^2)
    - E2 mode (Raman active): KE = 0.5*(Vx^2 + Vy^2), PE = 0.5*omega_E2^2*(Qx^2 + Qy^2)
    - A1 mode (Raman active): KE = 0.5*V^2, PE = 0.5*omega_A1^2*Q^2
    
    Also computes quartic, 3-phonon coupling, and spin-phonon coupling terms.
    
    Spin-phonon coupling energy is computed from spin correlations:
    - λ_E1 * Qx_E1 * <Si_x*Sj_z + Si_z*Sj_x>  (summed over NN pairs)
    - λ_E1 * Qy_E1 * <Si_y*Sj_z + Si_z*Sj_y>
    - λ_E2 * Qx_E2 * <Si_x*Sj_x - Si_y*Sj_y>
    - λ_E2 * Qy_E2 * <Si_x*Sj_y + Si_y*Sj_x>
    - λ_A1 * Q_A1  * <Si · Sj>
    
    Args:
        result: Dictionary from read_MD_phonon or read_pump_probe_phonon
        
    Returns:
        dict with energy components including phonon and spin-phonon energies
    """
    metadata = result['metadata']
    
    # Get phonon coordinates
    Qx_E1 = result['Qx_E1']
    Qy_E1 = result['Qy_E1']
    Qx_E2 = result['Qx_E2']
    Qy_E2 = result['Qy_E2']
    Q_A1 = result['Q_A1']
    
    # Get phonon velocities
    Vx_E1 = result.get('Vx_E1')
    Vy_E1 = result.get('Vy_E1')
    Vx_E2 = result.get('Vx_E2')
    Vy_E2 = result.get('Vy_E2')
    V_A1 = result.get('V_A1')
    
    # Get frequencies
    omega_E1 = metadata.get('omega_E1', 0.0)
    omega_E2 = metadata.get('omega_E2', 0.0)
    omega_A1 = metadata.get('omega_A1', 0.0)
    
    # Get quartic coefficients
    lambda_E1_quartic = metadata.get('lambda_E1_quartic', 0.0)
    lambda_E2_quartic = metadata.get('lambda_E2_quartic', 0.0)
    lambda_A1_quartic = metadata.get('lambda_A1_quartic', 0.0)
    
    # Get 3-phonon couplings
    g3_E1A1 = metadata.get('g3_E1A1', 0.0)
    g3_E2A1 = metadata.get('g3_E2A1', 0.0)
    g3_E1E2 = metadata.get('g3_E1E2', 0.0)
    
    # Get spin-phonon couplings
    lambda_E1 = metadata.get('lambda_E1', 0.0)
    lambda_E2 = metadata.get('lambda_E2', 0.0)
    lambda_A1 = metadata.get('lambda_A1', 0.0)
    
    # Mode amplitudes squared
    Q_E1_sq = Qx_E1**2 + Qy_E1**2
    Q_E2_sq = Qx_E2**2 + Qy_E2**2
    Q_A1_sq = Q_A1**2
    
    # --- E1 Mode ---
    if Vx_E1 is not None and Vy_E1 is not None:
        E1_kinetic = 0.5 * (Vx_E1**2 + Vy_E1**2)
    else:
        E1_kinetic = np.zeros_like(Qx_E1)
    E1_potential = 0.5 * omega_E1**2 * Q_E1_sq
    E1_quartic = 0.25 * lambda_E1_quartic * Q_E1_sq**2
    E1_total = E1_kinetic + E1_potential + E1_quartic
    
    # --- E2 Mode ---
    if Vx_E2 is not None and Vy_E2 is not None:
        E2_kinetic = 0.5 * (Vx_E2**2 + Vy_E2**2)
    else:
        E2_kinetic = np.zeros_like(Qx_E2)
    E2_potential = 0.5 * omega_E2**2 * Q_E2_sq
    E2_quartic = 0.25 * lambda_E2_quartic * Q_E2_sq**2
    E2_total = E2_kinetic + E2_potential + E2_quartic
    
    # --- A1 Mode ---
    if V_A1 is not None:
        A1_kinetic = 0.5 * V_A1**2
    else:
        A1_kinetic = np.zeros_like(Q_A1)
    A1_potential = 0.5 * omega_A1**2 * Q_A1_sq
    A1_quartic = 0.25 * lambda_A1_quartic * Q_A1_sq**2
    A1_total = A1_kinetic + A1_potential + A1_quartic
    
    # --- 3-Phonon Coupling Energy ---
    # g3_E1A1: |Q_E1|^2 * Q_A1
    # g3_E2A1: |Q_E2|^2 * Q_A1
    # g3_E1E2: (Qx_E1*Qx_E2 + Qy_E1*Qy_E2) bilinear coupling
    coupling_E1A1 = g3_E1A1 * Q_E1_sq * Q_A1
    coupling_E2A1 = g3_E2A1 * Q_E2_sq * Q_A1
    coupling_E1E2 = g3_E1E2 * (Qx_E1 * Qx_E2 + Qy_E1 * Qy_E2)
    coupling_total = coupling_E1A1 + coupling_E2A1 + coupling_E1E2
    
    # --- Spin-Phonon Coupling Energy ---
    # Compute from spin configurations if available
    # H_sp-ph = Σ_<ij> [λ_E1 * Qx_E1 * (Si_x*Sj_z + Si_z*Sj_x) + ...]
    spins = result.get('spins')  # (n_steps, n_sites, 3)
    positions = result.get('positions')  # (n_sites, 3)
    
    spin_phonon_E1 = np.zeros_like(Qx_E1)
    spin_phonon_E2 = np.zeros_like(Qx_E1)
    spin_phonon_A1 = np.zeros_like(Qx_E1)
    spin_phonon_total = np.zeros_like(Qx_E1)
    
    if spins is not None and positions is not None and (lambda_E1 != 0 or lambda_E2 != 0 or lambda_A1 != 0):
        n_steps, n_sites, _ = spins.shape
        
        # Build nearest-neighbor pairs from positions (honeycomb lattice)
        # For honeycomb: 3 NN per site, but we count each bond once
        # Use distance criterion to find NN pairs
        from scipy.spatial.distance import cdist
        
        # Find NN distance (should be ~1.0 for honeycomb)
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, np.inf)
        nn_dist = np.min(dist_matrix)
        
        # Find all NN pairs (i < j to avoid double counting)
        nn_pairs = []
        for i in range(n_sites):
            for j in range(i+1, n_sites):
                if np.abs(dist_matrix[i, j] - nn_dist) < 0.1:
                    nn_pairs.append((i, j))
        
        n_pairs = len(nn_pairs)
        
        # Compute spin-phonon correlations at each timestep
        for t in range(n_steps):
            S = spins[t]  # (n_sites, 3)
            
            xz_sum = 0.0  # Σ (Si_x*Sj_z + Si_z*Sj_x)
            yz_sum = 0.0  # Σ (Si_y*Sj_z + Si_z*Sj_y)
            xx_yy_sum = 0.0  # Σ (Si_x*Sj_x - Si_y*Sj_y)
            xy_sum = 0.0  # Σ (Si_x*Sj_y + Si_y*Sj_x)
            dot_sum = 0.0  # Σ (Si · Sj)
            
            for i, j in nn_pairs:
                Si, Sj = S[i], S[j]
                xz_sum += Si[0]*Sj[2] + Si[2]*Sj[0]
                yz_sum += Si[1]*Sj[2] + Si[2]*Sj[1]
                xx_yy_sum += Si[0]*Sj[0] - Si[1]*Sj[1]
                xy_sum += Si[0]*Sj[1] + Si[1]*Sj[0]
                dot_sum += np.dot(Si, Sj)
            
            # Normalize per site
            spin_phonon_E1[t] = lambda_E1 * (Qx_E1[t] * xz_sum + Qy_E1[t] * yz_sum) / n_sites
            spin_phonon_E2[t] = lambda_E2 * (Qx_E2[t] * xx_yy_sum + Qy_E2[t] * xy_sum) / n_sites
            spin_phonon_A1[t] = lambda_A1 * Q_A1[t] * dot_sum / n_sites
        
        spin_phonon_total = spin_phonon_E1 + spin_phonon_E2 + spin_phonon_A1
    
    # Total phonon energy (pure phonon, no spin-phonon)
    phonon_total = E1_total + E2_total + A1_total + coupling_total
    
    # Total energy from simulation
    total_energy = result['energy']
    
    # Pure spin energy = total - phonon - spin-phonon
    spin_energy = total_energy - phonon_total - spin_phonon_total
    
    return {
        # E1 mode
        'E1_kinetic': E1_kinetic,
        'E1_potential': E1_potential,
        'E1_quartic': E1_quartic,
        'E1_total': E1_total,
        # E2 mode
        'E2_kinetic': E2_kinetic,
        'E2_potential': E2_potential,
        'E2_quartic': E2_quartic,
        'E2_total': E2_total,
        # A1 mode
        'A1_kinetic': A1_kinetic,
        'A1_potential': A1_potential,
        'A1_quartic': A1_quartic,
        'A1_total': A1_total,
        # 3-Phonon Couplings
        'coupling_E1A1': coupling_E1A1,
        'coupling_E2A1': coupling_E2A1,
        'coupling_E1E2': coupling_E1E2,
        'coupling_total': coupling_total,
        # Spin-Phonon Coupling
        'spin_phonon_E1': spin_phonon_E1,
        'spin_phonon_E2': spin_phonon_E2,
        'spin_phonon_A1': spin_phonon_A1,
        'spin_phonon_total': spin_phonon_total,
        # Totals
        'phonon_total': phonon_total,
        'spin_energy': spin_energy,
        'total_energy': total_energy,
    }


def read_MD_phonon(dir, order_parameter_func=None, use_spin_deviation=True):
    """Read PhononLattice MD trajectory with full spin configuration.
    
    The PhononLattice trajectory.h5 format supports two modes:
    
    Per-bond-type format (new):
    - /phonon_trajectory/Qx_E1_0, Qx_E1_1, Qx_E1_2: E1 x-component per bond type
    - /phonon_trajectory/Qy_E1_0, Qy_E1_1, Qy_E1_2: E1 y-component per bond type
    - /phonon_trajectory/Qx_E2_0, ..., Q_A1_0, Q_A1_1, Q_A1_2: Similar for E2 and A1
    - /metadata/@drive_strength_0, @drive_strength_1, @drive_strength_2: Drive per bond
    
    Scalar format (legacy):
    - /phonon_trajectory/Qx_E1, Qy_E1, Qx_E2, Qy_E2, Q_A1: Phonon coordinates
    - /phonon_trajectory/Vx_E1, Vy_E1, Vx_E2, Vy_E2, V_A1: Phonon velocities
    
    Common:
    - /trajectory/times: Time array
    - /trajectory/spins: Full spin configuration [n_steps, n_sites, spin_dim]
    - /trajectory/magnetization_local, magnetization_antiferro, magnetization_global
    - /phonon_trajectory/energy: Energy density at each timestep
    - /metadata/positions: Site positions [n_sites, 3]
    - /metadata/@omega_E1, @omega_E2, @omega_A1, @g3_E1A1, etc.: Phonon parameters
    
    Args:
        dir: Directory containing trajectory.h5
        order_parameter_func: Optional callable(spins, positions, metadata) -> float
            Function to compute custom order parameter from spin configuration.
            If None and use_spin_deviation=True, uses spin deviation from initial config.
        use_spin_deviation: If True and order_parameter_func is None, compute
            spin deviation from initial configuration as default order parameter.
    
    Returns:
        dict with full trajectory data and computed observables, including:
        - 'has_per_bond_type': bool, whether per-bond-type data is available
        - 'Qx_E1', 'Qy_E1', etc.: Total phonon coordinates (sum over bond types)
        - 'Qx_E1_0', 'Qx_E1_1', 'Qx_E1_2', etc.: Per-bond-type phonon coordinates
        - 'Q_per_bond': (n_time, 5, 3) array of all per-bond-type coordinates
        - 'E1_amp_0', 'E1_amp_1', 'E1_amp_2': Per-bond-type E1 amplitudes
        - 'E1_spectrum_0', etc.: Per-bond-type spectra
    """
    hdf5_path = os.path.join(dir, "trajectory.h5")
    
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"No trajectory.h5 found in {dir}")
    
    print(f"  Loading HDF5: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Read spin trajectory (full configuration)
        T = f['/trajectory/times'][:]
        S = f['/trajectory/spins'][:]  # (n_steps, n_sites, spin_dim)
        
        # Read magnetizations
        M_antiferro = f['/trajectory/magnetization_antiferro'][:]
        M_local = f['/trajectory/magnetization_local'][:]
        M_global = f['/trajectory/magnetization_global'][:]
        
        # Read positions
        if '/metadata/positions' in f:
            P = f['/metadata/positions'][:]
        else:
            raise KeyError("positions dataset not found in HDF5 file")
        
        # Read phonon trajectory - try per-bond-type first, then scalar, then legacy
        # Per-bond-type format: Qx_E1_0, Qx_E1_1, Qx_E1_2, etc.
        # Scalar format: Qx_E1, Qy_E1, etc.
        # Legacy format: Qx, Qy, QR, etc.
        
        N_BONDS = 3  # x-bond, y-bond, z-bond
        
        # E1 mode (IR active, 2-component) - per bond type
        if '/phonon_trajectory/Qx_E1_0' in f:
            # Per-bond-type format
            Qx_E1_0 = f['/phonon_trajectory/Qx_E1_0'][:]
            Qx_E1_1 = f['/phonon_trajectory/Qx_E1_1'][:]
            Qx_E1_2 = f['/phonon_trajectory/Qx_E1_2'][:]
            Qy_E1_0 = f['/phonon_trajectory/Qy_E1_0'][:]
            Qy_E1_1 = f['/phonon_trajectory/Qy_E1_1'][:]
            Qy_E1_2 = f['/phonon_trajectory/Qy_E1_2'][:]
            # Compute total E1 as sum over bond types for backward compat
            Qx_E1 = Qx_E1_0 + Qx_E1_1 + Qx_E1_2
            Qy_E1 = Qy_E1_0 + Qy_E1_1 + Qy_E1_2
            has_per_bond_type = True
        elif '/phonon_trajectory/Qx_E1' in f:
            Qx_E1 = f['/phonon_trajectory/Qx_E1'][:]
            Qy_E1 = f['/phonon_trajectory/Qy_E1'][:]
            # Create dummy per-bond-type arrays (uniform distribution)
            Qx_E1_0 = Qx_E1_1 = Qx_E1_2 = Qx_E1 / 3.0
            Qy_E1_0 = Qy_E1_1 = Qy_E1_2 = Qy_E1 / 3.0
            has_per_bond_type = False
        else:
            # Legacy naming
            Qx_E1 = f['/phonon_trajectory/Qx'][:]
            Qy_E1 = f['/phonon_trajectory/Qy'][:]
            Qx_E1_0 = Qx_E1_1 = Qx_E1_2 = Qx_E1 / 3.0
            Qy_E1_0 = Qy_E1_1 = Qy_E1_2 = Qy_E1 / 3.0
            has_per_bond_type = False
        
        # E2 mode (Raman active, 2-component) - per bond type
        if '/phonon_trajectory/Qx_E2_0' in f:
            Qx_E2_0 = f['/phonon_trajectory/Qx_E2_0'][:]
            Qx_E2_1 = f['/phonon_trajectory/Qx_E2_1'][:]
            Qx_E2_2 = f['/phonon_trajectory/Qx_E2_2'][:]
            Qy_E2_0 = f['/phonon_trajectory/Qy_E2_0'][:]
            Qy_E2_1 = f['/phonon_trajectory/Qy_E2_1'][:]
            Qy_E2_2 = f['/phonon_trajectory/Qy_E2_2'][:]
            Qx_E2 = Qx_E2_0 + Qx_E2_1 + Qx_E2_2
            Qy_E2 = Qy_E2_0 + Qy_E2_1 + Qy_E2_2
        elif '/phonon_trajectory/Qx_E2' in f:
            Qx_E2 = f['/phonon_trajectory/Qx_E2'][:]
            Qy_E2 = f['/phonon_trajectory/Qy_E2'][:]
            Qx_E2_0 = Qx_E2_1 = Qx_E2_2 = Qx_E2 / 3.0
            Qy_E2_0 = Qy_E2_1 = Qy_E2_2 = Qy_E2 / 3.0
        else:
            # Legacy: E2 mode not present, initialize to zeros
            Qx_E2 = np.zeros_like(Qx_E1)
            Qy_E2 = np.zeros_like(Qy_E1)
            Qx_E2_0 = Qx_E2_1 = Qx_E2_2 = np.zeros_like(Qx_E1)
            Qy_E2_0 = Qy_E2_1 = Qy_E2_2 = np.zeros_like(Qy_E1)
        
        # A1 mode (Raman active, 1-component) - per bond type
        if '/phonon_trajectory/Q_A1_0' in f:
            Q_A1_0 = f['/phonon_trajectory/Q_A1_0'][:]
            Q_A1_1 = f['/phonon_trajectory/Q_A1_1'][:]
            Q_A1_2 = f['/phonon_trajectory/Q_A1_2'][:]
            Q_A1 = Q_A1_0 + Q_A1_1 + Q_A1_2
        elif '/phonon_trajectory/Q_A1' in f:
            Q_A1 = f['/phonon_trajectory/Q_A1'][:]
            Q_A1_0 = Q_A1_1 = Q_A1_2 = Q_A1 / 3.0
        else:
            # Legacy naming
            Q_A1 = f['/phonon_trajectory/QR'][:]
            Q_A1_0 = Q_A1_1 = Q_A1_2 = Q_A1 / 3.0
        
        energy = f['/phonon_trajectory/energy'][:]
        
        # Read velocities - try per-bond-type first, then scalar, then legacy
        # E1 velocities
        if '/phonon_trajectory/Vx_E1_0' in f:
            Vx_E1_0 = f['/phonon_trajectory/Vx_E1_0'][:]
            Vx_E1_1 = f['/phonon_trajectory/Vx_E1_1'][:]
            Vx_E1_2 = f['/phonon_trajectory/Vx_E1_2'][:]
            Vy_E1_0 = f['/phonon_trajectory/Vy_E1_0'][:]
            Vy_E1_1 = f['/phonon_trajectory/Vy_E1_1'][:]
            Vy_E1_2 = f['/phonon_trajectory/Vy_E1_2'][:]
            Vx_E1 = Vx_E1_0 + Vx_E1_1 + Vx_E1_2
            Vy_E1 = Vy_E1_0 + Vy_E1_1 + Vy_E1_2
        elif '/phonon_trajectory/Vx_E1' in f:
            Vx_E1 = f['/phonon_trajectory/Vx_E1'][:]
            Vy_E1 = f['/phonon_trajectory/Vy_E1'][:]
            Vx_E1_0 = Vx_E1_1 = Vx_E1_2 = Vx_E1 / 3.0
            Vy_E1_0 = Vy_E1_1 = Vy_E1_2 = Vy_E1 / 3.0
        else:
            Vx_E1 = f['/phonon_trajectory/Vx'][:] if '/phonon_trajectory/Vx' in f else None
            Vy_E1 = f['/phonon_trajectory/Vy'][:] if '/phonon_trajectory/Vy' in f else None
            if Vx_E1 is not None:
                Vx_E1_0 = Vx_E1_1 = Vx_E1_2 = Vx_E1 / 3.0
                Vy_E1_0 = Vy_E1_1 = Vy_E1_2 = Vy_E1 / 3.0
            else:
                Vx_E1_0 = Vx_E1_1 = Vx_E1_2 = None
                Vy_E1_0 = Vy_E1_1 = Vy_E1_2 = None
        
        # E2 velocities
        if '/phonon_trajectory/Vx_E2_0' in f:
            Vx_E2_0 = f['/phonon_trajectory/Vx_E2_0'][:]
            Vx_E2_1 = f['/phonon_trajectory/Vx_E2_1'][:]
            Vx_E2_2 = f['/phonon_trajectory/Vx_E2_2'][:]
            Vy_E2_0 = f['/phonon_trajectory/Vy_E2_0'][:]
            Vy_E2_1 = f['/phonon_trajectory/Vy_E2_1'][:]
            Vy_E2_2 = f['/phonon_trajectory/Vy_E2_2'][:]
            Vx_E2 = Vx_E2_0 + Vx_E2_1 + Vx_E2_2
            Vy_E2 = Vy_E2_0 + Vy_E2_1 + Vy_E2_2
        elif '/phonon_trajectory/Vx_E2' in f:
            Vx_E2 = f['/phonon_trajectory/Vx_E2'][:]
            Vy_E2 = f['/phonon_trajectory/Vy_E2'][:]
            Vx_E2_0 = Vx_E2_1 = Vx_E2_2 = Vx_E2 / 3.0
            Vy_E2_0 = Vy_E2_1 = Vy_E2_2 = Vy_E2 / 3.0
        else:
            Vx_E2 = np.zeros_like(Qx_E1) if Vx_E1 is not None else None
            Vy_E2 = np.zeros_like(Qy_E1) if Vy_E1 is not None else None
            if Vx_E2 is not None:
                Vx_E2_0 = Vx_E2_1 = Vx_E2_2 = Vx_E2 / 3.0
                Vy_E2_0 = Vy_E2_1 = Vy_E2_2 = Vy_E2 / 3.0
            else:
                Vx_E2_0 = Vx_E2_1 = Vx_E2_2 = None
                Vy_E2_0 = Vy_E2_1 = Vy_E2_2 = None
        
        # A1 velocities
        if '/phonon_trajectory/V_A1_0' in f:
            V_A1_0 = f['/phonon_trajectory/V_A1_0'][:]
            V_A1_1 = f['/phonon_trajectory/V_A1_1'][:]
            V_A1_2 = f['/phonon_trajectory/V_A1_2'][:]
            V_A1 = V_A1_0 + V_A1_1 + V_A1_2
        elif '/phonon_trajectory/V_A1' in f:
            V_A1 = f['/phonon_trajectory/V_A1'][:]
            V_A1_0 = V_A1_1 = V_A1_2 = V_A1 / 3.0
        else:
            V_A1 = f['/phonon_trajectory/VR'][:] if '/phonon_trajectory/VR' in f else None
            if V_A1 is not None:
                V_A1_0 = V_A1_1 = V_A1_2 = V_A1 / 3.0
            else:
                V_A1_0 = V_A1_1 = V_A1_2 = None
        
        # Read metadata
        metadata = {}
        if '/metadata' in f:
            meta_grp = f['/metadata']
            for key in meta_grp.attrs.keys():
                metadata[key] = meta_grp.attrs[key]
        
        # Read lattice info
        if 'lattice_size' in meta_grp.attrs:
            metadata['lattice_size'] = meta_grp.attrs['lattice_size']
        if 'spin_dim' in meta_grp.attrs:
            metadata['spin_dim'] = meta_grp.attrs['spin_dim']
        if 'n_atoms' in meta_grp.attrs:
            metadata['n_atoms'] = meta_grp.attrs['n_atoms']
        
        print(f"  Time steps: {len(T)}, t_range: [{T[0]:.4f}, {T[-1]:.4f}]")
        print(f"  Spin config: {S.shape} (n_steps, n_sites, spin_dim)")
        print(f"  Phonon params: omega_E1={metadata.get('omega_E1', 'N/A')}, "
              f"omega_E2={metadata.get('omega_E2', 'N/A')}, "
              f"omega_A1={metadata.get('omega_A1', 'N/A')}")
        if has_per_bond_type:
            print(f"  Per-bond-type phonons: drive_strength_0={metadata.get('drive_strength_0', 'N/A')}, "
                  f"drive_strength_1={metadata.get('drive_strength_1', 'N/A')}, "
                  f"drive_strength_2={metadata.get('drive_strength_2', 'N/A')}")
    
    print(f"  Computing spin deviation from initial configuration...")
    S0 = S[0]  # Initial spin configuration
    # RMS deviation at each timestep
    O_custom = np.array([np.sqrt(np.mean(np.sum((S[t] - S0)**2, axis=1))) 
                            for t in range(len(T))])
    
    # Stack phonon coordinates (all 5 modes - sum over bond types)
    Q = np.stack([Qx_E1, Qy_E1, Qx_E2, Qy_E2, Q_A1], axis=1)  # (n_time, 5)
    
    # Stack per-bond-type phonon coordinates
    # Shape: (n_time, 5 modes, 3 bond types)
    Q_per_bond = np.stack([
        np.stack([Qx_E1_0, Qx_E1_1, Qx_E1_2], axis=1),  # Qx_E1 per bond
        np.stack([Qy_E1_0, Qy_E1_1, Qy_E1_2], axis=1),  # Qy_E1 per bond
        np.stack([Qx_E2_0, Qx_E2_1, Qx_E2_2], axis=1),  # Qx_E2 per bond
        np.stack([Qy_E2_0, Qy_E2_1, Qy_E2_2], axis=1),  # Qy_E2 per bond
        np.stack([Q_A1_0, Q_A1_1, Q_A1_2], axis=1),     # Q_A1 per bond
    ], axis=1)  # (n_time, 5, 3)
    
    # Compute mode amplitudes (total across bond types)
    E1_amp = np.sqrt(Qx_E1**2 + Qy_E1**2)
    E2_amp = np.sqrt(Qx_E2**2 + Qy_E2**2)
    
    # Per-bond-type mode amplitudes
    E1_amp_0 = np.sqrt(Qx_E1_0**2 + Qy_E1_0**2)
    E1_amp_1 = np.sqrt(Qx_E1_1**2 + Qy_E1_1**2)
    E1_amp_2 = np.sqrt(Qx_E1_2**2 + Qy_E1_2**2)
    E2_amp_0 = np.sqrt(Qx_E2_0**2 + Qy_E2_0**2)
    E2_amp_1 = np.sqrt(Qx_E2_1**2 + Qy_E2_1**2)
    E2_amp_2 = np.sqrt(Qx_E2_2**2 + Qy_E2_2**2)
    
    # Compute spectra
    dt = T[1] - T[0] if len(T) > 1 else 1.0
    omega = np.fft.fftfreq(len(T), dt) * 2 * np.pi
    omega = np.fft.fftshift(omega)
    
    # E1 mode spectrum (Qx_E1, Qy_E1) - total
    Qx_E1_fft = np.fft.fftshift(np.fft.fft(Qx_E1 - np.mean(Qx_E1)))
    Qy_E1_fft = np.fft.fftshift(np.fft.fft(Qy_E1 - np.mean(Qy_E1)))
    E1_spectrum = np.abs(Qx_E1_fft)**2 + np.abs(Qy_E1_fft)**2
    
    # E2 mode spectrum (Qx_E2, Qy_E2) - total
    Qx_E2_fft = np.fft.fftshift(np.fft.fft(Qx_E2 - np.mean(Qx_E2)))
    Qy_E2_fft = np.fft.fftshift(np.fft.fft(Qy_E2 - np.mean(Qy_E2)))
    E2_spectrum = np.abs(Qx_E2_fft)**2 + np.abs(Qy_E2_fft)**2
    
    # A1 mode spectrum (Q_A1) - total
    Q_A1_fft = np.fft.fftshift(np.fft.fft(Q_A1 - np.mean(Q_A1)))
    A1_spectrum = np.abs(Q_A1_fft)**2
    
    # Per-bond-type spectra for E1 mode
    E1_spectrum_0 = np.abs(np.fft.fftshift(np.fft.fft(Qx_E1_0 - np.mean(Qx_E1_0))))**2 + \
                    np.abs(np.fft.fftshift(np.fft.fft(Qy_E1_0 - np.mean(Qy_E1_0))))**2
    E1_spectrum_1 = np.abs(np.fft.fftshift(np.fft.fft(Qx_E1_1 - np.mean(Qx_E1_1))))**2 + \
                    np.abs(np.fft.fftshift(np.fft.fft(Qy_E1_1 - np.mean(Qy_E1_1))))**2
    E1_spectrum_2 = np.abs(np.fft.fftshift(np.fft.fft(Qx_E1_2 - np.mean(Qx_E1_2))))**2 + \
                    np.abs(np.fft.fftshift(np.fft.fft(Qy_E1_2 - np.mean(Qy_E1_2))))**2
    
    # Custom order parameter spectrum
    O_custom_spectrum = None
    if O_custom is not None:
        O_custom_fft = np.fft.fftshift(np.fft.fft(O_custom - np.mean(O_custom)))
        O_custom_spectrum = np.abs(O_custom_fft)**2
    
    return {
        'time': T,
        'spins': S,
        'positions': P,
        'energy': energy,
        'M_antiferro': M_antiferro,
        'M_local': M_local,
        'M_global': M_global,
        'Q': Q,
        'Q_per_bond': Q_per_bond,  # (n_time, 5, 3) per-bond-type coordinates
        'has_per_bond_type': has_per_bond_type,
        # E1 mode - total (sum over bond types)
        'Qx_E1': Qx_E1, 'Qy_E1': Qy_E1,
        'Vx_E1': Vx_E1, 'Vy_E1': Vy_E1,
        'E1_amp': E1_amp,
        'E1_spectrum': E1_spectrum,
        # E1 mode - per bond type
        'Qx_E1_0': Qx_E1_0, 'Qx_E1_1': Qx_E1_1, 'Qx_E1_2': Qx_E1_2,
        'Qy_E1_0': Qy_E1_0, 'Qy_E1_1': Qy_E1_1, 'Qy_E1_2': Qy_E1_2,
        'Vx_E1_0': Vx_E1_0, 'Vx_E1_1': Vx_E1_1, 'Vx_E1_2': Vx_E1_2,
        'Vy_E1_0': Vy_E1_0, 'Vy_E1_1': Vy_E1_1, 'Vy_E1_2': Vy_E1_2,
        'E1_amp_0': E1_amp_0, 'E1_amp_1': E1_amp_1, 'E1_amp_2': E1_amp_2,
        'E1_spectrum_0': E1_spectrum_0, 'E1_spectrum_1': E1_spectrum_1, 'E1_spectrum_2': E1_spectrum_2,
        # E2 mode - total
        'Qx_E2': Qx_E2, 'Qy_E2': Qy_E2,
        'Vx_E2': Vx_E2, 'Vy_E2': Vy_E2,
        'E2_amp': E2_amp,
        'E2_spectrum': E2_spectrum,
        # E2 mode - per bond type
        'Qx_E2_0': Qx_E2_0, 'Qx_E2_1': Qx_E2_1, 'Qx_E2_2': Qx_E2_2,
        'Qy_E2_0': Qy_E2_0, 'Qy_E2_1': Qy_E2_1, 'Qy_E2_2': Qy_E2_2,
        'Vx_E2_0': Vx_E2_0, 'Vx_E2_1': Vx_E2_1, 'Vx_E2_2': Vx_E2_2,
        'Vy_E2_0': Vy_E2_0, 'Vy_E2_1': Vy_E2_1, 'Vy_E2_2': Vy_E2_2,
        'E2_amp_0': E2_amp_0, 'E2_amp_1': E2_amp_1, 'E2_amp_2': E2_amp_2,
        # A1 mode - total
        'Q_A1': Q_A1, 'V_A1': V_A1,
        'A1_spectrum': A1_spectrum,
        # A1 mode - per bond type
        'Q_A1_0': Q_A1_0, 'Q_A1_1': Q_A1_1, 'Q_A1_2': Q_A1_2,
        'V_A1_0': V_A1_0, 'V_A1_1': V_A1_1, 'V_A1_2': V_A1_2,
        # Legacy aliases for backward compatibility
        'Qx': Qx_E1, 'Qy': Qy_E1, 'QR': Q_A1,
        'Vx': Vx_E1, 'Vy': Vy_E1, 'VR': V_A1,
        # Order parameter and other
        'omega': omega,
        'O_custom': O_custom,
        'O_custom_spectrum': O_custom_spectrum,
        'metadata': metadata
    }


def compute_DSSF_phonon(dir, w, order_parameter_func=None):
    """Compute Dynamical Spin Structure Factor from PhononLattice MD trajectory.
    
    This function reads the full spin configuration and computes DSSF along
    high-symmetry k-path, similar to the Lattice reader.
    
    Args:
        dir: Directory containing trajectory.h5
        w: Frequency array for DSSF
        order_parameter_func: Optional function to compute custom order parameter
    
    Returns:
        dict with DSSF, phonon spectra, and custom order parameter
    """
    # Read full trajectory
    result = read_MD_phonon(dir, order_parameter_func=order_parameter_func)
    
    T = result['time']
    S = result['spins']
    P = result['positions']
    
    # Compute DSSF along k-path
    print(f"  Computing DSSF...")
    dssf = DSSF(w, DSSF_K, S, P, T, gb=True)
    print(f"  DSSF complete: shape={dssf.shape}")
    
    # Save DSSF
    np.savetxt(os.path.join(dir, "DSSF.txt"), dssf)
    
    result['DSSF'] = dssf
    result['DSSF_omega'] = w
    return result


# ============================================================================
# ORDER PARAMETER EXAMPLES
# ============================================================================

def staggered_magnetization_order(spins, positions, metadata):
    """Compute staggered magnetization order parameter.
    
    Args:
        spins: (n_sites, spin_dim) spin configuration
        positions: (n_sites, 3) site positions
        metadata: dict with lattice info
    
    Returns:
        float: |M_staggered|
    """
    n_atoms = metadata.get('n_atoms', 2)  # Default honeycomb has 2 atoms/cell
    n_sites = len(spins)
    
    M_stag = np.zeros(spins.shape[1])
    for i in range(n_sites):
        sign = 1.0 if (i % n_atoms == 0) else -1.0
        M_stag += sign * spins[i]
    M_stag /= n_sites
    
    return np.linalg.norm(M_stag)


def sublattice_magnetization_order(spins, positions, metadata):
    """Compute sublattice magnetization magnitudes.
    
    Returns tuple of (|M_A|, |M_B|) for two sublattices.
    """
    n_atoms = metadata.get('n_atoms', 2)
    n_sites = len(spins)
    
    M_A = np.zeros(spins.shape[1])
    M_B = np.zeros(spins.shape[1])
    n_A, n_B = 0, 0
    
    for i in range(n_sites):
        if i % n_atoms == 0:
            M_A += spins[i]
            n_A += 1
        else:
            M_B += spins[i]
            n_B += 1
    
    M_A /= n_A if n_A > 0 else 1
    M_B /= n_B if n_B > 0 else 1
    
    return np.linalg.norm(M_A), np.linalg.norm(M_B)


def zigzag_order_parameter(spins, positions, metadata):
    """Compute zigzag order parameter for honeycomb lattice.
    
    Zigzag order: spins aligned along rows, alternating between rows.
    """
    n_sites = len(spins)
    n_atoms = metadata.get('n_atoms', 2)
    dim1 = metadata.get('dimensions', [1, 1, 1])[0] if 'dimensions' in metadata else 1
    
    # Simple zigzag: check if spins are ferromagnetically aligned within sublattice
    M_A = np.zeros(spins.shape[1])
    M_B = np.zeros(spins.shape[1])
    n_A, n_B = 0, 0
    
    for i in range(n_sites):
        if i % n_atoms == 0:
            M_A += spins[i]
            n_A += 1
        else:
            M_B += spins[i]
            n_B += 1
    
    if n_A > 0:
        M_A /= n_A
    if n_B > 0:
        M_B /= n_B
    
    # Zigzag order: sublattices antiparallel
    return np.linalg.norm(M_A - M_B) / 2


def spin_deviation_order_parameter(initial_spins):
    """Factory function to create order parameter measuring deviation from initial config.
    
    Returns a function that computes the RMS deviation of spins from the initial
    configuration. This measures how much the spin configuration has changed.
    
    The deviation is computed as:
        O = sqrt( (1/N) * sum_i |S_i(t) - S_i(0)|^2 )
    
    This equals 0 when spins are identical to initial, and increases as spins deviate.
    Maximum value is 2 (when all spins flip by 180 degrees).
    
    Args:
        initial_spins: np.array of shape (n_sites, spin_dim) - the reference configuration
        
    Returns:
        order_parameter_func: callable(spins, positions, metadata) -> float
        
    Example usage:
        # Read trajectory first to get initial spins
        result = read_MD_phonon(dir)
        initial_spins = result['spins'][0]  # First timestep
        
        # Create order parameter function
        deviation_func = spin_deviation_order_parameter(initial_spins)
        
        # Re-read with order parameter
        result = read_MD_phonon(dir, order_parameter_func=deviation_func)
    """
    S0 = np.array(initial_spins)  # Store reference configuration
    
    def order_param(spins, positions, metadata):
        """Compute RMS deviation from initial spin configuration."""
        delta = spins - S0
        # RMS deviation: sqrt( mean of |delta|^2 )
        return np.sqrt(np.mean(np.sum(delta**2, axis=1)))
    
    return order_param


def spin_deviation_normalized_order_parameter(initial_spins):
    """Factory function for normalized spin deviation order parameter.
    
    Returns a function that computes the mean (1 - S_i · S_i(0)) normalized by 2,
    giving a value between 0 (unchanged) and 1 (all spins flipped 180°).
    
    Args:
        initial_spins: np.array of shape (n_sites, spin_dim) - the reference configuration
        
    Returns:
        order_parameter_func: callable(spins, positions, metadata) -> float
    """
    S0 = np.array(initial_spins)
    
    def order_param(spins, positions, metadata):
        """Compute normalized deviation: mean(1 - S·S0) / 2."""
        # Dot product S_i · S0_i for each site
        dot_products = np.sum(spins * S0, axis=1)
        # (1 - cos(theta)) / 2 = sin^2(theta/2), ranges from 0 to 1
        return np.mean((1 - dot_products) / 2)
    
    return order_param


def read_pump_probe_phonon(dir, order_parameter_func=None):
    """Read pump-probe MD trajectory from PhononLattice (set_pulse + molecular_dynamics).
    
    This reads trajectory.h5 from pump-probe simulations that use set_pulse followed
    by molecular_dynamics (NOT from pump_probe_spectroscopy or single_pulse_drive).
    The trajectory contains full spin configurations at each saved timestep.
    
    The PhononLattice trajectory.h5 format contains:
    - /trajectory/times: Time array
    - /trajectory/spins: Full spin configuration [n_steps, n_sites, spin_dim]
    - /trajectory/magnetization_antiferro, magnetization_local, magnetization_global
    - /phonon_trajectory/Qx_E1, Qy_E1, Qx_E2, Qy_E2, Q_A1: Phonon coordinates
    - /phonon_trajectory/Vx_E1, Vy_E1, Vx_E2, Vy_E2, V_A1: Phonon velocities
    - /phonon_trajectory/energy: Energy density
    - /metadata/positions: Site positions [n_sites, 3]
    - /metadata/@omega_E1, @omega_E2, @omega_A1, @g3_E1A1, etc.: Phonon parameters
    
    Args:
        dir: Directory containing trajectory.h5
        order_parameter_func: Optional callable(spins, positions, metadata) -> float
            Function to compute custom order parameter from spin configuration.
            spins: np.array of shape (n_sites, spin_dim)
            positions: np.array of shape (n_sites, 3)
            metadata: dict of simulation parameters
            Returns: scalar order parameter value
    
    Returns:
        dict with full trajectory data including:
        - 'time': Time array
        - 'spins': Full spin configurations [n_steps, n_sites, spin_dim]
        - 'positions': Site positions [n_sites, 3]
        - 'M_antiferro', 'M_local', 'M_global': Magnetization trajectories
        - 'Qx_E1', 'Qy_E1', 'Qx_E2', 'Qy_E2', 'Q_A1': Phonon coordinates
        - 'Vx_E1', 'Vy_E1', 'Vx_E2', 'Vy_E2', 'V_A1': Phonon velocities
        - 'E1_spectrum', 'E2_spectrum', 'A1_spectrum': Phonon spectra
        - 'O_custom', 'O_custom_spectrum': Custom order parameter (if func provided)
        - 'metadata': Simulation parameters
    """
    return read_MD_phonon(dir, order_parameter_func=order_parameter_func)


# ============================================================================
# SPIN CONFIGURATION ANIMATION
# ============================================================================

def animate_spin_trajectory(result, output_file='spin_animation.mp4', 
                            fps=10, interval=1, 
                            show_phonon=True, show_pulse=True, figsize=(12, 10)):
    """Create animation of spin configuration evolution from MD trajectory.
    
    Spins are transformed from local Kitaev frame to global Cartesian frame.
    Only 2D XY-plane projection is shown (no 3D plots).
    
    Args:
        result: Dictionary from read_MD_phonon or read_pump_probe_phonon
        output_file: Output filename (supports .mp4, .gif)
        fps: Frames per second
        interval: Use every Nth frame (for long trajectories)
        show_phonon: If True, show phonon amplitude panel
        show_pulse: If True, show THz pulse form panel
        figsize: Figure size
    
    Returns:
        matplotlib.animation.FuncAnimation object
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from matplotlib.colors import Normalize
    
    # Kitaev local to global frame transformation
    KITAEV_LOCAL_TO_GLOBAL = np.array([
        [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    ])
    
    S = result['spins']  # (n_steps, n_sites, spin_dim)
    P = result['positions']  # (n_sites, 3)
    T = result['time']
    
    # E1 mode (IR active)
    Qx_E1 = result['Qx_E1']
    Qy_E1 = result['Qy_E1']
    E1_amp = np.sqrt(Qx_E1**2 + Qy_E1**2)
    
    # E2 mode (Raman active)
    Qx_E2 = result['Qx_E2']
    Qy_E2 = result['Qy_E2']
    E2_amp = np.sqrt(Qx_E2**2 + Qy_E2**2)
    
    # A1 mode (Raman active)
    Q_A1 = result['Q_A1']
    
    # Legacy aliases
    Qx, Qy, QR = Qx_E1, Qy_E1, Q_A1
    
    O_custom = result.get('O_custom', None)
    
    n_steps, n_sites, spin_dim = S.shape
    n_atoms = result['metadata'].get('n_atoms', 2)
    
    # Transform all spins to global frame
    # S_global[t, i, :] = KITAEV_LOCAL_TO_GLOBAL @ S[t, i, :]
    S_global = np.einsum('ij,tnj->tni', KITAEV_LOCAL_TO_GLOBAL, S)
    
    # Subsample if interval > 1
    frame_indices = list(range(0, n_steps, interval))
    n_frames = len(frame_indices)
    
    print(f"Creating animation: {n_frames} frames from {n_steps} timesteps")
    
    # Extract pump parameters from metadata
    metadata = result.get('metadata', {})
    pump_amplitude = metadata.get('pump_amplitude', 0.0)
    pump_frequency = metadata.get('pump_frequency', 1.0)
    pump_time = metadata.get('pump_time', 0.0)
    pump_width = metadata.get('pump_width', 1.0)
    pump_phase = metadata.get('pump_phase', 0.0)
    pump_polarization = metadata.get('pump_polarization', 0.0)
    
    # Compute pulse envelope and E-field
    if show_pulse and pump_amplitude > 0:
        envelope = pump_amplitude * np.exp(-(T - pump_time)**2 / (2 * pump_width**2))
        Ex = envelope * np.cos(pump_frequency * (T - pump_time) + pump_phase) * np.cos(pump_polarization)
        Ey = envelope * np.cos(pump_frequency * (T - pump_time) + pump_phase) * np.sin(pump_polarization)
        has_pulse = True
        print(f"  Pulse params: A={pump_amplitude:.2f}, ω={pump_frequency:.2f}, t0={pump_time:.2f}, σ={pump_width:.2f}")
    else:
        has_pulse = False
        Ex = Ey = envelope = None
    
    # Count number of bottom panels
    n_bottom_panels = sum([show_phonon, show_pulse and has_pulse])
    
    # Create figure - only 2D plot, no 3D
    if n_bottom_panels > 0:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1 + n_bottom_panels, 1, height_ratios=[3] + [1]*n_bottom_panels)
        ax_2d = fig.add_subplot(gs[0])
        
        panel_idx = 1
        if show_phonon:
            ax_phonon = fig.add_subplot(gs[panel_idx])
            panel_idx += 1
        else:
            ax_phonon = None
        
        if show_pulse and has_pulse:
            ax_pulse = fig.add_subplot(gs[panel_idx])
        else:
            ax_pulse = None
    else:
        fig, ax_2d = plt.subplots(figsize=(10, 8))
        ax_phonon = None
        ax_pulse = None
    
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
    title_2d = ax_2d.set_title(f'Global Frame Spin Config, t = {T[0]:.3f}', fontsize=12)
    ax_2d.set_aspect('equal')
    ax_2d.grid(True, alpha=0.3)
    
    # Initialize phonon plot
    if ax_phonon is not None:
        ax_phonon.plot(T, E1_amp, 'r-', alpha=0.3, label='|E1|')
        ax_phonon.plot(T, E2_amp, 'g-', alpha=0.3, label='|E2|')
        ax_phonon.plot(T, Q_A1, 'b-', alpha=0.3, label='Q_A1')
        if O_custom is not None:
            ax_phonon_twin = ax_phonon.twinx()
            ax_phonon_twin.plot(T, O_custom, 'purple', alpha=0.3)
            line_O, = ax_phonon_twin.plot([], [], 'purple', lw=2, label='O_custom')
            ax_phonon_twin.set_ylabel('Order Parameter', color='purple')
        else:
            ax_phonon_twin = None
            line_O = None
        line_E1, = ax_phonon.plot([], [], 'r-', lw=2)
        line_E2, = ax_phonon.plot([], [], 'g-', lw=2)
        line_A1, = ax_phonon.plot([], [], 'b-', lw=2)
        vline_phonon = ax_phonon.axvline(T[0], color='k', linestyle='--', lw=1)
        ax_phonon.set_xlabel('Time')
        ax_phonon.set_ylabel('Phonon Amplitude')
        ax_phonon.legend(loc='upper left', fontsize=8)
        ax_phonon.set_xlim(T[0], T[-1])
    else:
        line_E1 = line_E2 = line_A1 = vline_phonon = None
        line_O = None
        ax_phonon_twin = None
    
    # Initialize pulse plot
    if ax_pulse is not None and has_pulse:
        # Plot full pulse in background (transparent)
        ax_pulse.plot(T, envelope, 'k--', alpha=0.3, linewidth=1, label='Envelope')
        ax_pulse.plot(T, -envelope, 'k--', alpha=0.3, linewidth=1)
        ax_pulse.plot(T, Ex, 'b-', alpha=0.3, linewidth=0.5)
        if np.any(np.abs(Ey) > 1e-10):
            ax_pulse.plot(T, Ey, 'r-', alpha=0.3, linewidth=0.5, label='Ey')
        
        # Progressive pulse trace
        line_Ex, = ax_pulse.plot([], [], 'b-', lw=1.2, label='Ex(t)')
        line_Ey, = ax_pulse.plot([], [], 'r-', lw=1.2, label='Ey(t)') if np.any(np.abs(Ey) > 1e-10) else (None,)
        vline_pulse = ax_pulse.axvline(T[0], color='k', linestyle='--', lw=1)
        ax_pulse.axhline(0, color='gray', linewidth=0.5)
        ax_pulse.set_xlabel('Time')
        ax_pulse.set_ylabel('E-field')
        ax_pulse.set_title(f'THz Pulse (A={pump_amplitude:.1f}, ω={pump_frequency:.1f}, σ={pump_width:.1f})')
        ax_pulse.legend(loc='upper right', fontsize=8)
        ax_pulse.set_xlim(T[0], T[-1])
    else:
        line_Ex = line_Ey = vline_pulse = None
    
    plt.tight_layout()
    
    def update(frame_idx):
        i = frame_indices[frame_idx]
        
        # Get global spins at this timestep
        sx, sy, sz = S_global[i, :, 0], S_global[i, :, 1], S_global[i, :, 2]
        
        # Normalize in-plane components
        in_plane_mag = np.sqrt(sx**2 + sy**2)
        in_plane_mag = np.where(in_plane_mag > 1e-10, in_plane_mag, 1.0)
        sx_norm, sy_norm = sx / in_plane_mag, sy / in_plane_mag
        
        # Update quiver - need to recreate
        nonlocal quiver
        quiver.remove()
        quiver = ax_2d.quiver(P[:, 0], P[:, 1], sx_norm, sy_norm, sz,
                              cmap='coolwarm', norm=norm,
                              scale=1.2, scale_units='xy', angles='xy',
                              pivot='middle', width=0.008)
        
        # Update scatter colors
        scatter.set_array(sz)
        
        title_2d.set_text(f'Global Frame Spin Config, t = {T[i]:.3f}')
        
        # Update phonon lines
        if ax_phonon is not None and line_E1 is not None:
            line_E1.set_data(T[:i+1], E1_amp[:i+1])
            line_E2.set_data(T[:i+1], E2_amp[:i+1])
            line_A1.set_data(T[:i+1], Q_A1[:i+1])
            vline_phonon.set_xdata([T[i], T[i]])
            if line_O is not None and O_custom is not None:
                line_O.set_data(T[:i+1], O_custom[:i+1])
        
        # Update pulse lines
        if ax_pulse is not None and line_Ex is not None:
            line_Ex.set_data(T[:i+1], Ex[:i+1])
            if line_Ey is not None:
                line_Ey.set_data(T[:i+1], Ey[:i+1])
            vline_pulse.set_xdata([T[i], T[i]])
        
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


def plot_spin_snapshot(result, time_index=0, output_file=None, figsize=(14, 10)):
    """Plot a single snapshot of spin configuration from trajectory.
    
    Spins are transformed from local Kitaev frame to global Cartesian frame.
    Only 2D XY-plane projection is shown (no 3D plots).
    
    Args:
        result: Dictionary from read_MD_phonon or read_pump_probe_phonon
        time_index: Index in the time array (or -1 for last frame)
        output_file: If provided, save figure to this file
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    from matplotlib.colors import Normalize
    
    # Kitaev local to global frame transformation
    KITAEV_LOCAL_TO_GLOBAL = np.array([
        [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    ])
    
    S = result['spins']
    P = result['positions']
    T = result['time']
    
    n_sites = S.shape[1]
    
    if time_index < 0:
        time_index = len(T) + time_index
    
    spins_local = S[time_index]
    t = T[time_index]
    
    # Transform to global frame
    spins_global = np.einsum('ij,nj->ni', KITAEV_LOCAL_TO_GLOBAL, spins_local)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    
    # Main 2D spin plot (global frame)
    ax = fig.add_subplot(gs[0, :])
    sx, sy, sz = spins_global[:, 0], spins_global[:, 1], spins_global[:, 2]
    
    # Normalize in-plane components
    in_plane_mag = np.sqrt(sx**2 + sy**2)
    in_plane_mag = np.where(in_plane_mag > 1e-10, in_plane_mag, 1.0)
    sx_norm, sy_norm = sx / in_plane_mag, sy / in_plane_mag
    
    norm = Normalize(vmin=-1, vmax=1)
    scatter = ax.scatter(P[:, 0], P[:, 1], c=sz, cmap='coolwarm', norm=norm,
                         s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    quiver = ax.quiver(P[:, 0], P[:, 1], sx_norm, sy_norm, sz,
                       cmap='coolwarm', norm=norm,
                       scale=1.2, scale_units='xy', angles='xy',
                       pivot='middle', width=0.008)
    cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
    cbar.set_label(r'$S_z^{global}$', fontsize=12)
    ax.set_xlabel('x (global)', fontsize=12)
    ax.set_ylabel('y (global)', fontsize=12)
    ax.set_title(f'Global Frame Spin Configuration at t = {t:.4f}', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Order parameter history
    ax3 = fig.add_subplot(gs[1, 0])
    if result['O_custom'] is not None:
        ax3.plot(T, result['O_custom'], 'purple', lw=1, label='O_custom')
    M_stag_norm = np.linalg.norm(result['M_antiferro'], axis=1)
    ax3.plot(T, M_stag_norm, 'b-', lw=1, alpha=0.5, label='|M_stag|')
    ax3.axvline(t, color='r', linestyle='--', label=f't = {t:.3f}')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Order Parameter')
    ax3.set_title('Order Parameter Dynamics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Phonon dynamics
    ax4 = fig.add_subplot(gs[1, 1])
    Qx_E1 = result.get('Qx_E1', result.get('Qx'))
    Qy_E1 = result.get('Qy_E1', result.get('Qy'))
    Q_A1 = result.get('Q_A1', result.get('QR'))
    E1_amp = np.sqrt(Qx_E1**2 + Qy_E1**2)
    ax4.plot(T, E1_amp, 'r-', lw=1, label='|E1|')
    
    # E2 mode if available
    Qx_E2 = result.get('Qx_E2')
    Qy_E2 = result.get('Qy_E2')
    if Qx_E2 is not None and Qy_E2 is not None:
        E2_amp = np.sqrt(Qx_E2**2 + Qy_E2**2)
        ax4.plot(T, E2_amp, 'g-', lw=1, label='|E2|')
    
    ax4.plot(T, Q_A1, 'b-', lw=1, label='Q_A1')
    ax4.axvline(t, color='k', linestyle='--')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Phonon Amplitude')
    ax4.set_title('Phonon Dynamics')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Saved snapshot to {output_file}")
    
    return fig


def plot_spin_trajectory_summary(result, output_file=None, order_parameter_func=None):
    """Create summary plot of spin and phonon dynamics from MD trajectory.
    
    Args:
        result: Dictionary from read_MD_phonon or read_pump_probe_phonon
        output_file: If provided, save figure to this file
        order_parameter_func: Optional function for custom order parameter
    
    Returns:
        matplotlib figure
    """
    T = result['time']
    M_antiferro = result['M_antiferro']
    M_local = result['M_local']
    # Use new naming with fallback to legacy
    Qx_E1 = result.get('Qx_E1', result.get('Qx'))
    Qy_E1 = result.get('Qy_E1', result.get('Qy'))
    Qx_E2 = result.get('Qx_E2', np.zeros_like(Qx_E1))
    Qy_E2 = result.get('Qy_E2', np.zeros_like(Qy_E1))
    Q_A1 = result.get('Q_A1', result.get('QR'))
    energy = result['energy']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Magnetization dynamics
    ax = axes[0, 0]
    M_stag_norm = np.linalg.norm(M_antiferro, axis=1)
    M_local_norm = np.linalg.norm(M_local, axis=1)
    ax.plot(T, M_stag_norm, 'b-', lw=1, label='|M_staggered|')
    ax.plot(T, M_local_norm, 'r-', lw=1, alpha=0.5, label='|M_local|')
    if result['O_custom'] is not None:
        ax.plot(T, result['O_custom'], 'g-', lw=1, label='O_custom')
    ax.set_xlabel('Time')
    ax.set_ylabel('Order Parameter')
    ax.set_title('Magnetization Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phonon dynamics
    ax = axes[0, 1]
    E1_amp = np.sqrt(Qx_E1**2 + Qy_E1**2)
    E2_amp = np.sqrt(Qx_E2**2 + Qy_E2**2)
    ax.plot(T, E1_amp, 'b-', lw=1, label='|E1|')
    ax.plot(T, E2_amp, 'g-', lw=1, label='|E2|')
    ax.plot(T, Q_A1, 'r-', lw=1, label='Q_A1')
    ax.set_xlabel('Time')
    ax.set_ylabel('Phonon Amplitude')
    ax.set_title('Phonon Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy
    ax = axes[1, 0]
    ax.plot(T, energy, 'k-', lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy Density')
    ax.set_title('Energy Evolution')
    ax.grid(True, alpha=0.3)
    
    # Spectra
    ax = axes[1, 1]
    omega = result['omega']
    omega_pos = omega > 0
    ax.semilogy(omega[omega_pos], result['E1_spectrum'][omega_pos], 'b-', lw=1, label='E1')
    if 'E2_spectrum' in result:
        ax.semilogy(omega[omega_pos], result['E2_spectrum'][omega_pos], 'g-', lw=1, label='E2')
    ax.semilogy(omega[omega_pos], result['A1_spectrum'][omega_pos], 'r-', lw=1, label='A1')
    if result['O_custom_spectrum'] is not None:
        ax.semilogy(omega[omega_pos], result['O_custom_spectrum'][omega_pos], 'g-', lw=1, label='O_custom')
    ax.set_xlabel('ω (rad/time)')
    ax.set_ylabel('|FFT|²')
    ax.set_title('Phonon/Order Parameter Spectra')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 20])
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Saved summary to {output_file}")
    
    return fig
    """Ensure an omega window is either None or a valid (min, max) tuple."""
    if window is None:
        return None
    if (not isinstance(window, (tuple, list)) or len(window) != 2 or
            window[0] >= window[1]):
        raise ValueError(f"{name} must be a tuple (min, max) with min < max")
    return tuple(window)

def read_2D_nonlinear_phonon(dir, omega_t_window=None, omega_tau_window=None):
    """Read and compute 2D nonlinear spectroscopy for PhononLattice.
    
    The PhononLattice 2DCS format uses pump_probe_spectroscopy.h5 with structure:
    - /parameters: pulse_amp, pulse_width, pulse_freq, polarization, E_ground
    - /reference: time, M_antiferro, M_local (M0 trajectory)
    - /delay_scan/tau_values: array of tau values
    - /delay_scan/tau_i/M1, /delay_scan/tau_i/M01: delay-dependent trajectories
    
    Args:
        dir: Directory containing pump_probe_spectroscopy.h5
        omega_t_window: Optional (min, max) tuple for ω_t axis limits
        omega_tau_window: Optional (min, max) tuple for ω_τ axis limits
    """
    omega_t_window = _validate_window(omega_t_window, "omega_t_window")
    omega_tau_window = _validate_window(omega_tau_window, "omega_tau_window")
    hdf5_path = os.path.join(dir, "pump_probe_spectroscopy.h5")
    component_labels = ['x', 'y', 'z']
    
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"No pump_probe_spectroscopy.h5 found in {dir}")
    
    print(f"  Loading HDF5: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Read parameters
        pulse_amp = f['/parameters/pulse_amp'][0]
        pulse_width = f['/parameters/pulse_width'][0]
        pulse_freq = f['/parameters/pulse_freq'][0]
        polarization = f['/parameters/polarization'][0]
        E_ground = f['/parameters/E_ground'][0]
        
        print(f"  Pulse parameters:")
        print(f"    Amplitude: {pulse_amp}")
        print(f"    Width: {pulse_width}")
        print(f"    Frequency: {pulse_freq}")
        print(f"    Polarization: {polarization}")
        print(f"    Ground state energy: {E_ground}")
        
        # Read reference trajectory (M0)
        times = f['/reference/time'][:]
        M0_antiferro = f['/reference/M_antiferro'][:]
        M0_local = f['/reference/M_local'][:]
        
        # Read M_global and O_custom if available
        M0_global = None
        if '/reference/M_global' in f:
            M0_global = f['/reference/M_global'][:]
        O0_custom = None
        if '/reference/O_custom' in f:
            O0_custom = f['/reference/O_custom'][:]
        
        print(f"  Reference trajectory:")
        print(f"    Time steps: {len(times)}, t_range: [{times[0]:.4f}, {times[-1]:.4f}]")
        print(f"    M0_antiferro shape: {M0_antiferro.shape}")
        if M0_global is not None:
            print(f"    M0_global shape: {M0_global.shape}")
        if O0_custom is not None:
            print(f"    O_custom: available (shape: {O0_custom.shape})")
        else:
            print(f"    O_custom: not available")
        
        # Read tau values
        tau_values = f['/delay_scan/tau_values'][:]
        tau_step = len(tau_values)
        
        print(f"  Delay scan:")
        print(f"    Tau values: {tau_step}, tau_range: [{tau_values[0]:.4f}, {tau_values[-1]:.4f}]")
        
        # Process all 3 components (x, y, z) for M_NL, M0, M1, M01
        length = len(times)
        M_NL_components = np.zeros((3, tau_step, length))
        M0_components = np.zeros((3, tau_step, length))
        M1_components = np.zeros((3, tau_step, length))
        M01_components = np.zeros((3, tau_step, length))
        
        # Also process M_global components if available
        M_NL_global_components = None
        M0_global_components = None
        M1_global_components = None
        M01_global_components = None
        
        # Also process O_custom if available
        O_NL_custom = None
        O0_custom_arr = None
        O1_custom = None
        O01_custom = None
        
        has_global = M0_global is not None
        has_custom = O0_custom is not None
        
        if has_global:
            M_NL_global_components = np.zeros((3, tau_step, length))
            M0_global_components = np.zeros((3, tau_step, length))
            M1_global_components = np.zeros((3, tau_step, length))
            M01_global_components = np.zeros((3, tau_step, length))
        
        if has_custom:
            O_NL_custom = np.zeros((tau_step, length))
            O0_custom_arr = np.zeros((tau_step, length))
            O1_custom = np.zeros((tau_step, length))
            O01_custom = np.zeros((tau_step, length))
        
        for i in range(tau_step):
            tau_group = f[f'/delay_scan/tau_{i}']
            
            # Read M1 trajectory (probe only)
            M1_antiferro = tau_group['M1/M_antiferro'][:]
            
            # Read M01 trajectory (pump + probe)
            M01_antiferro = tau_group['M01/M_antiferro'][:]
            
            # Read M_global if available
            M1_global = None
            M01_global = None
            if has_global:
                if 'M1/M_global' in tau_group:
                    M1_global = tau_group['M1/M_global'][:]
                if 'M01/M_global' in tau_group:
                    M01_global = tau_group['M01/M_global'][:]
            
            # Read O_custom if available
            O1_cust = None
            O01_cust = None
            if has_custom:
                if 'M1/O_custom' in tau_group:
                    O1_cust = tau_group['M1/O_custom'][:]
                if 'M01/O_custom' in tau_group:
                    O01_cust = tau_group['M01/O_custom'][:]
            
            for comp in range(3):
                M0 = M0_antiferro[:, comp]
                M1 = M1_antiferro[:, comp]
                M01 = M01_antiferro[:, comp]
                
                min_len = min(len(M0), len(M1), len(M01), length)
                M_NL_components[comp, i, :min_len] = M01[:min_len] - M0[:min_len] - M1[:min_len]
                M0_components[comp, i, :min_len] = M0[:min_len]
                M1_components[comp, i, :min_len] = M1[:min_len]
                M01_components[comp, i, :min_len] = M01[:min_len]
                
                # Process M_global components
                if has_global and M1_global is not None and M01_global is not None:
                    M0_g = M0_global[:, comp]
                    M1_g = M1_global[:, comp]
                    M01_g = M01_global[:, comp]
                    min_len_g = min(len(M0_g), len(M1_g), len(M01_g), length)
                    M_NL_global_components[comp, i, :min_len_g] = M01_g[:min_len_g] - M0_g[:min_len_g] - M1_g[:min_len_g]
                    M0_global_components[comp, i, :min_len_g] = M0_g[:min_len_g]
                    M1_global_components[comp, i, :min_len_g] = M1_g[:min_len_g]
                    M01_global_components[comp, i, :min_len_g] = M01_g[:min_len_g]
            
            # Process O_custom
            if has_custom and O1_cust is not None and O01_cust is not None:
                min_len_o = min(len(O0_custom), len(O1_cust), len(O01_cust), length)
                O_NL_custom[i, :min_len_o] = O01_cust[:min_len_o] - O0_custom[:min_len_o] - O1_cust[:min_len_o]
                O0_custom_arr[i, :min_len_o] = O0_custom[:min_len_o]
                O1_custom[i, :min_len_o] = O1_cust[:min_len_o]
                O01_custom[i, :min_len_o] = O01_cust[:min_len_o]
        
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        tau = tau_values

    # Use z-component for main analysis
    M_NL = M_NL_components[2]
    
    # Compute omega arrays in simulation units (rad/sim_time)
    omega_tau = np.fft.fftfreq(int(len(tau)), tau[1] - tau[0] if len(tau) > 1 else 1.0) * 2 * np.pi
    omega_tau = np.fft.fftshift(omega_tau)
    omega_t = np.fft.fftfreq(M_NL.shape[1], dt) * 2 * np.pi
    omega_t = np.fft.fftshift(omega_t)
    
    # Convert to THz for plotting
    omega_tau_THz = omega_tau / (2 * np.pi * HBAR_OVER_MEV_PS)
    omega_t_THz = omega_t / (2 * np.pi * HBAR_OVER_MEV_PS)
    
    # Convert window limits if provided (assumed in simulation units, convert to THz)
    def convert_window_to_THz(window):
        if window is None:
            return None
        return (window[0] / (2 * np.pi * HBAR_OVER_MEV_PS), 
                window[1] / (2 * np.pi * HBAR_OVER_MEV_PS))
    
    omega_t_window_THz = convert_window_to_THz(omega_t_window)
    omega_tau_window_THz = convert_window_to_THz(omega_tau_window)
    
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
                                      norm='linear', extent=[omega_t_THz[0], omega_t_THz[-1], omega_tau_THz[0], omega_tau_THz[-1]])
            ax_freq.set_xlabel('$\\omega_t$ (THz)')
            ax_freq.set_ylabel('$\\omega_{\\tau}$ (THz)')
            ax_freq.set_title(f'{component_labels[comp]}-component (freq domain)')
            if omega_t_window_THz is not None:
                ax_freq.set_xlim(omega_t_window_THz)
            if omega_tau_window_THz is not None:
                ax_freq.set_ylim(omega_tau_window_THz)
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
                   extent=[omega_t_THz[0], omega_t_THz[-1], omega_tau_THz[0], omega_tau_THz[-1]],
                   aspect='auto', cmap='gnuplot2', norm='linear')
        plt.xlabel('$\\omega_t$ (THz)')
        plt.ylabel('$\\omega_{\\tau}$ (THz)')
        plt.colorbar(label='Intensity')
        plt.title(f'{sig_name} Spectrum')
        if omega_t_window_THz is not None:
            plt.xlim(omega_t_window_THz)
        if omega_tau_window_THz is not None:
            plt.ylim(omega_tau_window_THz)
        plt.savefig(dir + f"/{sig_name}_SPEC.pdf")
        plt.clf()
    
    # =========================================================================
    # M_NL analysis (M01 - M0 - M1)
    # =========================================================================
    
    # Debug plots for all components
    fig_debug, axes_debug = plt.subplots(3, 3, figsize=(15, 12))
    for comp in range(3):
        M_NL_comp = M_NL_components[comp]
        
        # Time domain plot
        ax_time = axes_debug[comp, 0]
        for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
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
        
        # Frequency domain plot
        M_NL_comp_FF = compute_2d_fft(M_NL_comp)
        
        ax_freq = axes_debug[comp, 2]
        im_freq = ax_freq.imshow(M_NL_comp_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                  norm='linear', extent=[omega_t_THz[0], omega_t_THz[-1], omega_tau_THz[0], omega_tau_THz[-1]])
        ax_freq.set_xlabel('$\\omega_t$ (THz)')
        ax_freq.set_ylabel('$\\omega_{\\tau}$ (THz)')
        ax_freq.set_title(f'{component_labels[comp]}-component (freq domain)')
        if omega_t_window_THz is not None:
            ax_freq.set_xlim(omega_t_window_THz)
        if omega_tau_window_THz is not None:
            ax_freq.set_ylim(omega_tau_window_THz)
        plt.colorbar(im_freq, ax=ax_freq)
        
        np.savetxt(dir + f"/M_NL_FF_{component_labels[comp]}.txt", M_NL_comp_FF)
    
    plt.tight_layout()
    plt.savefig(dir + "/M_NL_components_debug.pdf")
    plt.clf()
    plt.close()
    
    # Main M_NL spectrum (z-component)
    M_NL_FF = compute_2d_fft(M_NL)
    np.savetxt(dir + "/M_NL_FF.txt", M_NL_FF)
    
    plt.imshow(M_NL_FF, origin='lower',
               extent=[omega_t_THz[0], omega_t_THz[-1], omega_tau_THz[0], omega_tau_THz[-1]],
               aspect='auto', cmap='gnuplot2', norm='linear')
    plt.xlabel('$\\omega_t$ (THz)')
    plt.ylabel('$\\omega_{\\tau}$ (THz)')
    plt.colorbar(label='Intensity')
    if omega_t_window_THz is not None:
        plt.xlim(omega_t_window_THz)
    if omega_tau_window_THz is not None:
        plt.ylim(omega_tau_window_THz)
    plt.savefig(dir + "/M_NLSPEC.pdf")
    plt.clf()
    
    # =========================================================================
    # O_custom analysis (custom order parameter) if available
    # =========================================================================
    if O_NL_custom is not None:
        print("  Processing O_custom (custom order parameter)...")
        
        # Save O_custom time-domain data
        signal_names_O = ['O0_custom', 'O1_custom', 'O01_custom', 'O_NL_custom']
        signal_data_O = [O0_custom_arr, O1_custom, O01_custom, O_NL_custom]
        
        for sig_name, sig_data in zip(signal_names_O, signal_data_O):
            if sig_data is not None:
                sig_FF = compute_2d_fft(sig_data)
                np.savetxt(dir + f"/{sig_name}_FF.txt", sig_FF)
                
                # Create figure for this signal
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                # Time domain
                ax_time = axes[0]
                for tau_idx in range(0, len(tau), max(1, len(tau) // 5)):
                    ax_time.plot(sig_data[tau_idx, :], label=f'τ={tau[tau_idx]:.2f}', alpha=0.7)
                ax_time.set_xlabel('Time index')
                ax_time.set_ylabel(sig_name)
                ax_time.set_title('Time domain')
                ax_time.legend(fontsize=6)
                ax_time.grid(True, alpha=0.3)
                
                # 2D time-tau plot
                ax_2d = axes[1]
                im = ax_2d.imshow(sig_data, origin='lower', aspect='auto', cmap='RdBu_r',
                                  extent=[0, sig_data.shape[1], tau[0], tau[-1]])
                ax_2d.set_xlabel('Time index')
                ax_2d.set_ylabel('τ')
                ax_2d.set_title('(τ, t) plot')
                plt.colorbar(im, ax=ax_2d)
                
                # Frequency domain
                ax_freq = axes[2]
                im_freq = ax_freq.imshow(sig_FF, origin='lower', aspect='auto', cmap='gnuplot2',
                                          norm='linear', extent=[omega_t_THz[0], omega_t_THz[-1], omega_tau_THz[0], omega_tau_THz[-1]])
                ax_freq.set_xlabel('$\\omega_t$ (THz)')
                ax_freq.set_ylabel('$\\omega_{\\tau}$ (THz)')
                ax_freq.set_title('Frequency domain')
                if omega_t_window_THz is not None:
                    ax_freq.set_xlim(omega_t_window_THz)
                if omega_tau_window_THz is not None:
                    ax_freq.set_ylim(omega_tau_window_THz)
                plt.colorbar(im_freq, ax=ax_freq)
                
                plt.tight_layout()
                plt.savefig(dir + f"/{sig_name}_analysis.pdf")
                plt.clf()
                plt.close()
        
        # Main O_NL_custom spectrum
        O_NL_FF = compute_2d_fft(O_NL_custom)
        plt.imshow(O_NL_FF, origin='lower',
                   extent=[omega_t_THz[0], omega_t_THz[-1], omega_tau_THz[0], omega_tau_THz[-1]],
                   aspect='auto', cmap='gnuplot2', norm='linear')
        plt.xlabel('$\\omega_t$ (THz)')
        plt.ylabel('$\\omega_{\\tau}$ (THz)')
        plt.colorbar(label='Intensity')
        plt.title('O_NL Custom Order Parameter Spectrum')
        if omega_t_window_THz is not None:
            plt.xlim(omega_t_window_THz)
        if omega_tau_window_THz is not None:
            plt.ylim(omega_tau_window_THz)
        plt.savefig(dir + "/O_NL_SPEC.pdf")
        plt.clf()

# ============================================================================
# PUMP-PROBE SINGLE RUN ANALYSIS
# ============================================================================

def analyze_pump_probe(dir, omega_max=25.0, order_parameter_func=None,
                       create_animation=True, animation_interval=10, animation_fps=15):
    """Analyze a single pump-probe run from trajectory.h5.
    
    Focused on order parameter analysis (not Mx, My, Mz) and phonon dynamics.
    Creates comprehensive figure and optional spin configuration animation.
    
    Args:
        dir: Directory containing trajectory.h5
        omega_max: Maximum frequency for spectrum plots (default: 25)
        order_parameter_func: Optional callable(spins, positions, metadata) -> float
            If None, uses staggered magnetization norm as default order parameter.
        create_animation: Whether to create spin configuration animation (default: True)
        animation_interval: Use every Nth frame for animation (default: 10)
        animation_fps: Frames per second for animation (default: 15)
    
    Outputs:
        - pump_probe_analysis.pdf: Comprehensive multi-panel plot
        - spin_animation.mp4: Animation of spin configuration (if create_animation=True)
        - Various spectrum .txt files
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    
    # Use staggered magnetization as default order parameter
    if order_parameter_func is None:
        order_parameter_func = staggered_magnetization_order
    
    result = read_pump_probe_phonon(dir, order_parameter_func=order_parameter_func)
    
    T = result['time']
    # Convert time to picoseconds (1 sim unit = hbar/meV = 0.658 ps)
    T_ps = T * HBAR_OVER_MEV_PS
    
    omega = result['omega']
    # Convert angular frequency from rad/(sim time) to THz
    omega_THz = omega / (2 * np.pi * HBAR_OVER_MEV_PS)
    omega_pos = (omega > 0) & (omega < omega_max)
    
    # Extract key quantities
    S = result['spins']  # (n_steps, n_sites, spin_dim)
    P = result['positions']  # (n_sites, 3)
    O_custom = result['O_custom']
    O_custom_spectrum = result['O_custom_spectrum']
    M_antiferro = result['M_antiferro']
    M_antiferro_norm = np.linalg.norm(M_antiferro, axis=1)
    
    # E1 mode (IR active, THz driven)
    Qx_E1 = result['Qx_E1']
    Qy_E1 = result['Qy_E1']
    E1_amp = np.sqrt(Qx_E1**2 + Qy_E1**2)
    
    # E2 mode (Raman active)
    Qx_E2 = result['Qx_E2']
    Qy_E2 = result['Qy_E2']
    E2_amp = np.sqrt(Qx_E2**2 + Qy_E2**2)
    
    # A1 mode (Raman active)
    Q_A1 = result['Q_A1']
    
    # Legacy aliases for backward compatibility
    Qx, Qy, QR = Qx_E1, Qy_E1, Q_A1
    
    # =========================================================================
    # Create comprehensive single figure
    # =========================================================================
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # --- Row 1: Time domain ---
    # Order parameter dynamics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(T_ps, O_custom, 'purple', lw=1.5, label='O_custom')
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Order Parameter')
    ax1.set_title('Order Parameter Dynamics')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Phonon E1 dynamics
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(T_ps, Qx_E1, 'r-', alpha=0.8, label='Qx_E1')
    ax2.plot(T_ps, Qy_E1, 'g-', alpha=0.8, label='Qy_E1')
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Phonon Coordinate')
    ax2.set_title('E1 Mode (Qx, Qy) - IR Active')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Phonon E2 dynamics (Raman active)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(T_ps, Qx_E2, 'c-', alpha=0.8, label='Qx_E2')
    ax3.plot(T_ps, Qy_E2, 'm-', alpha=0.8, label='Qy_E2')
    ax3.set_xlabel('Time (ps)')
    ax3.set_ylabel('Phonon Coordinate')
    ax3.set_title('E2 Mode (Qx, Qy) - Raman Active')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Phonon A1 mode
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(T_ps, Q_A1, 'b-', lw=1.5, label='Q_A1')
    ax4.set_xlabel('Time (ps)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('A1 Mode - Raman Active')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # --- Row 2: Spectra ---
    # Order parameter spectrum
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.semilogy(omega_THz[omega_pos], O_custom_spectrum[omega_pos], 'purple', lw=1.5)
    ax5.set_xlabel('Frequency (THz)')
    ax5.set_ylabel('|FFT|²')
    ax5.set_title('Order Parameter Spectrum')
    ax5.grid(True, alpha=0.3)
    
    # E1 spectrum
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.semilogy(omega_THz[omega_pos], result['E1_spectrum'][omega_pos], 'r-', lw=1.5)
    ax6.set_xlabel('Frequency (THz)')
    ax6.set_ylabel('|FFT|²')
    ax6.set_title('E1 Mode Spectrum')
    ax6.grid(True, alpha=0.3)
    if 'omega_E1' in result['metadata']:
        omega_E1_THz = result['metadata']['omega_E1'] / (2 * np.pi * HBAR_OVER_MEV_PS)
        ax6.axvline(omega_E1_THz, color='k', linestyle='--', 
                    label=f"ω_E1 = {omega_E1_THz:.2f} THz")
        ax6.legend(fontsize=8)
    
    # E2 spectrum
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.semilogy(omega_THz[omega_pos], result['E2_spectrum'][omega_pos], 'g-', lw=1.5)
    ax7.set_xlabel('Frequency (THz)')
    ax7.set_ylabel('|FFT|²')
    ax7.set_title('E2 Mode Spectrum')
    ax7.grid(True, alpha=0.3)
    if 'omega_E2' in result['metadata']:
        omega_E2_THz = result['metadata']['omega_E2'] / (2 * np.pi * HBAR_OVER_MEV_PS)
        ax7.axvline(omega_E2_THz, color='k', linestyle='--',
                    label=f"ω_E2 = {omega_E2_THz:.2f} THz")
        ax7.legend(fontsize=8)
    
    # A1 spectrum
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.semilogy(omega_THz[omega_pos], result['A1_spectrum'][omega_pos], 'b-', lw=1.5)
    ax8.set_xlabel('Frequency (THz)')
    ax8.set_ylabel('|FFT|²')
    ax8.set_title('A1 Mode Spectrum')
    ax8.grid(True, alpha=0.3)
    if 'omega_A1' in result['metadata']:
        omega_A1_THz = result['metadata']['omega_A1'] / (2 * np.pi * HBAR_OVER_MEV_PS)
        ax8.axvline(omega_A1_THz, color='k', linestyle='--',
                    label=f"ω_A1 = {omega_A1_THz:.2f} THz")
        ax8.legend(fontsize=8)
    
    # --- Row 3: Combined spectrum and energy ---
    # Combined phonon + order parameter spectrum
    ax9 = fig.add_subplot(gs[2, 0:2])
    ax9.semilogy(omega_THz[omega_pos], result['E1_spectrum'][omega_pos], 'r-', lw=1.5, label='E1', alpha=0.7)
    ax9.semilogy(omega_THz[omega_pos], result['E2_spectrum'][omega_pos], 'g-', lw=1.5, label='E2', alpha=0.7)
    ax9.semilogy(omega_THz[omega_pos], result['A1_spectrum'][omega_pos], 'b-', lw=1.5, label='A1', alpha=0.7)
    ax9.semilogy(omega_THz[omega_pos], O_custom_spectrum[omega_pos], 'purple', lw=1.5, label='O', alpha=0.7)
    ax9.set_xlabel('Frequency (THz)')
    ax9.set_ylabel('|FFT|²')
    ax9.set_title('Combined Spectrum')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # Compute energy breakdown
    energy_breakdown = compute_phonon_energy_breakdown(result)
    
    # Spin vs Phonon vs Spin-Phonon energy
    ax10 = fig.add_subplot(gs[2, 2:4])
    ax10.plot(T_ps, energy_breakdown['total_energy'], 'k-', lw=1.5, label='Total')
    ax10.plot(T_ps, energy_breakdown['spin_energy'], 'purple', lw=1.5, label='Spin', alpha=0.8)
    ax10.plot(T_ps, energy_breakdown['phonon_total'], 'orange', lw=1.5, label='Phonon', alpha=0.8)
    if np.any(np.abs(energy_breakdown['spin_phonon_total']) > 1e-10):
        ax10.plot(T_ps, energy_breakdown['spin_phonon_total'], 'cyan', lw=1.5, label='Spin-Phonon', alpha=0.8)
    ax10.set_xlabel('Time (ps)')
    ax10.set_ylabel('Energy Density')
    ax10.set_title('Energy Breakdown')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    # --- Row 4: Phonon energy breakdown ---
    # E1, E2, A1 mode energies
    ax11 = fig.add_subplot(gs[3, 0:2])
    ax11.plot(T_ps, energy_breakdown['E1_total'], 'r-', lw=1.5, label='E1 total', alpha=0.8)
    ax11.plot(T_ps, energy_breakdown['E2_total'], 'g-', lw=1.5, label='E2 total', alpha=0.8)
    ax11.plot(T_ps, energy_breakdown['A1_total'], 'b-', lw=1.5, label='A1 total', alpha=0.8)
    if np.any(np.abs(energy_breakdown['coupling_total']) > 1e-10):
        ax11.plot(T_ps, energy_breakdown['coupling_total'], 'm--', lw=1.5, label='3-phonon', alpha=0.8)
    # Add spin-phonon components if nonzero
    if np.any(np.abs(energy_breakdown['spin_phonon_E1']) > 1e-10):
        ax11.plot(T_ps, energy_breakdown['spin_phonon_E1'], 'r:', lw=1.5, label='sp-ph E1', alpha=0.8)
    if np.any(np.abs(energy_breakdown['spin_phonon_E2']) > 1e-10):
        ax11.plot(T_ps, energy_breakdown['spin_phonon_E2'], 'g:', lw=1.5, label='sp-ph E2', alpha=0.8)
    if np.any(np.abs(energy_breakdown['spin_phonon_A1']) > 1e-10):
        ax11.plot(T_ps, energy_breakdown['spin_phonon_A1'], 'b:', lw=1.5, label='sp-ph A1', alpha=0.8)
    ax11.set_xlabel('Time (ps)')
    ax11.set_ylabel('Energy Density')
    ax11.set_title('Phonon & Spin-Phonon Energies')
    ax11.legend(fontsize=7, ncol=2)
    ax11.grid(True, alpha=0.3)
    
    # Kinetic vs Potential for all modes
    ax12 = fig.add_subplot(gs[3, 2:4])
    # E1 mode
    ax12.plot(T_ps, energy_breakdown['E1_kinetic'], 'r-', lw=1, label='E1 KE', alpha=0.7)
    ax12.plot(T_ps, energy_breakdown['E1_potential'], 'r--', lw=1, label='E1 PE', alpha=0.7)
    # E2 mode
    ax12.plot(T_ps, energy_breakdown['E2_kinetic'], 'g-', lw=1, label='E2 KE', alpha=0.7)
    ax12.plot(T_ps, energy_breakdown['E2_potential'], 'g--', lw=1, label='E2 PE', alpha=0.7)
    # A1 mode
    ax12.plot(T_ps, energy_breakdown['A1_kinetic'], 'b-', lw=1, label='A1 KE', alpha=0.7)
    ax12.plot(T_ps, energy_breakdown['A1_potential'], 'b--', lw=1, label='A1 PE', alpha=0.7)
    ax12.set_xlabel('Time (ps)')
    ax12.set_ylabel('Energy Density')
    ax12.set_title('Phonon KE vs PE')
    ax12.legend(fontsize=7, ncol=2)
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "pump_probe_analysis.pdf"), dpi=150)
    plt.close()
    print(f"  Saved: pump_probe_analysis.pdf")
    
    # =========================================================================
    # Create spin configuration animations (global and local frames)
    # =========================================================================
    if create_animation and S is not None:
        from matplotlib.colors import Normalize
        
        n_steps, n_sites, spin_dim = S.shape
        metadata = result.get('metadata', {})
        
        # Kitaev local to global frame transformation
        KITAEV_LOCAL_TO_GLOBAL = np.array([
            [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
            [-1/np.sqrt(2), 1/np.sqrt(2), 0],
            [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
        ])
        
        # Transform all spins to global frame
        S_global = np.einsum('ij,tnj->tni', KITAEV_LOCAL_TO_GLOBAL, S)
        
        # Subsample frames
        frame_indices = list(range(0, n_steps, animation_interval))
        n_frames = len(frame_indices)
        
        norm = Normalize(vmin=-1, vmax=1)
        
        # Extract pump parameters for pulse panel
        pump_amplitude = metadata.get('pump_amplitude', 0.0)
        pump_frequency = metadata.get('pump_frequency', 1.0)
        pump_time = metadata.get('pump_time', 0.0)
        pump_width = metadata.get('pump_width', 1.0)
        pump_phase = metadata.get('pump_phase', 0.0)
        pump_polarization = metadata.get('pump_polarization', 0.0)
        
        # Compute pulse envelope and E-field
        if pump_amplitude > 0:
            envelope = pump_amplitude * np.exp(-(T - pump_time)**2 / (2 * pump_width**2))
            Ex = envelope * np.cos(pump_frequency * (T - pump_time) + pump_phase) * np.cos(pump_polarization)
            Ey = envelope * np.cos(pump_frequency * (T - pump_time) + pump_phase) * np.sin(pump_polarization)
            has_pulse = True
        else:
            has_pulse = False
            envelope = Ex = Ey = np.zeros_like(T)
        
        # =====================================================================
        # Animation with pulse panel (spin + phonon + pulse)
        # =====================================================================
        print(f"  Creating spin animation with pulse...")
        print(f"    {n_frames} frames from {n_steps} timesteps (interval={animation_interval})")
        
        # Create animation figure with 3 panels
        fig_anim = plt.figure(figsize=(12, 12))
        gs_anim = fig_anim.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        ax_2d = fig_anim.add_subplot(gs_anim[0])
        ax_phonon = fig_anim.add_subplot(gs_anim[1])
        ax_pulse = fig_anim.add_subplot(gs_anim[2])
        
        # Get initial global spins
        sx, sy, sz = S_global[0, :, 0], S_global[0, :, 1], S_global[0, :, 2]
        in_plane_mag = np.sqrt(sx**2 + sy**2)
        in_plane_mag = np.where(in_plane_mag > 1e-10, in_plane_mag, 1.0)
        sx_norm, sy_norm = sx / in_plane_mag, sy / in_plane_mag
        
        # Initialize 2D plot (global frame)
        scatter_2d = ax_2d.scatter(P[:, 0], P[:, 1], c=sz, cmap='coolwarm', norm=norm,
                                   s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        quiver_2d = ax_2d.quiver(P[:, 0], P[:, 1], sx_norm, sy_norm, sz,
                                  cmap='coolwarm', norm=norm,
                                  scale=1.2, scale_units='xy', angles='xy',
                                  pivot='middle', width=0.008)
        cbar = plt.colorbar(quiver_2d, ax=ax_2d, shrink=0.8)
        cbar.set_label(r'$S_z^{global}$', fontsize=12)
        ax_2d.set_xlabel('x (global)', fontsize=12)
        ax_2d.set_ylabel('y (global)', fontsize=12)
        title_2d = ax_2d.set_title(f'Global Frame, t = {T_ps[0]:.3f} ps', fontsize=12)
        ax_2d.set_aspect('equal')
        ax_2d.grid(True, alpha=0.3)
        
        # Initialize phonon + order parameter plot
        ax_phonon.plot(T_ps, E1_amp, 'r-', alpha=0.3, label='|E1|')
        ax_phonon.plot(T_ps, E2_amp, 'g-', alpha=0.3, label='|E2|')
        ax_phonon.plot(T_ps, Q_A1, 'b-', alpha=0.3, label='Q_A1')
        ax_phonon_twin = ax_phonon.twinx()
        ax_phonon_twin.plot(T_ps, O_custom, 'purple', alpha=0.3)
        line_E1, = ax_phonon.plot([], [], 'r-', lw=2)
        line_E2, = ax_phonon.plot([], [], 'g-', lw=2)
        line_A1, = ax_phonon.plot([], [], 'b-', lw=2)
        line_O, = ax_phonon_twin.plot([], [], 'purple', lw=2, label='O_custom')
        vline_phonon = ax_phonon.axvline(T_ps[0], color='k', linestyle='--', lw=1)
        ax_phonon.set_xlabel('Time (ps)')
        ax_phonon.set_ylabel('Phonon Amplitude')
        ax_phonon_twin.set_ylabel('Order Parameter', color='purple')
        ax_phonon.legend(loc='upper left', fontsize=8)
        ax_phonon_twin.legend(loc='upper right', fontsize=8)
        ax_phonon.set_xlim(T_ps[0], T_ps[-1])
        
        # Initialize pulse panel
        ax_pulse.plot(T_ps, envelope * HBAR_OVER_MEV_PS, 'k--', alpha=0.3, linewidth=1, label='Envelope')
        ax_pulse.plot(T_ps, -envelope * HBAR_OVER_MEV_PS, 'k--', alpha=0.3, linewidth=1)
        ax_pulse.plot(T_ps, Ex * HBAR_OVER_MEV_PS, 'b-', alpha=0.3, linewidth=0.5)
        if np.any(np.abs(Ey) > 1e-10):
            ax_pulse.plot(T_ps, Ey * HBAR_OVER_MEV_PS, 'r-', alpha=0.3, linewidth=0.5)
        line_Ex, = ax_pulse.plot([], [], 'b-', lw=1.2, label='Ex(t)')
        line_Ey, = ax_pulse.plot([], [], 'r-', lw=1.2, label='Ey(t)') if np.any(np.abs(Ey) > 1e-10) else (None,)
        vline_pulse = ax_pulse.axvline(T_ps[0], color='k', linestyle='--', lw=1)
        ax_pulse.axhline(0, color='gray', linewidth=0.5)
        ax_pulse.set_xlabel('Time (ps)')
        ax_pulse.set_ylabel('E-field')
        omega_THz_pump = pump_frequency / (2 * np.pi * HBAR_OVER_MEV_PS)
        ax_pulse.set_title(f'THz Pulse (A={pump_amplitude:.1f}, ω={omega_THz_pump:.1f} THz, σ={pump_width:.1f})')
        ax_pulse.legend(loc='upper right', fontsize=8)
        ax_pulse.set_xlim(T_ps[0], T_ps[-1])
        
        plt.tight_layout()
        
        def update_with_pulse(frame_idx):
            i = frame_indices[frame_idx]
            
            # Get global spins at this timestep
            sx, sy, sz = S_global[i, :, 0], S_global[i, :, 1], S_global[i, :, 2]
            in_plane_mag = np.sqrt(sx**2 + sy**2)
            in_plane_mag = np.where(in_plane_mag > 1e-10, in_plane_mag, 1.0)
            sx_norm, sy_norm = sx / in_plane_mag, sy / in_plane_mag
            
            # Update 2D quiver
            nonlocal quiver_2d
            quiver_2d.remove()
            quiver_2d = ax_2d.quiver(P[:, 0], P[:, 1], sx_norm, sy_norm, sz,
                                      cmap='coolwarm', norm=norm,
                                      scale=1.2, scale_units='xy', angles='xy',
                                      pivot='middle', width=0.008)
            scatter_2d.set_array(sz)
            title_2d.set_text(f'Global Frame, t = {T_ps[i]:.3f} ps')
            
            # Update phonon/order parameter lines
            line_E1.set_data(T_ps[:i+1], E1_amp[:i+1])
            line_E2.set_data(T_ps[:i+1], E2_amp[:i+1])
            line_A1.set_data(T_ps[:i+1], Q_A1[:i+1])
            line_O.set_data(T_ps[:i+1], O_custom[:i+1])
            vline_phonon.set_xdata([T_ps[i], T_ps[i]])
            
            # Update pulse lines
            line_Ex.set_data(T_ps[:i+1], Ex[:i+1] * HBAR_OVER_MEV_PS)
            if line_Ey is not None:
                line_Ey.set_data(T_ps[:i+1], Ey[:i+1] * HBAR_OVER_MEV_PS)
            vline_pulse.set_xdata([T_ps[i], T_ps[i]])
            
            return [quiver_2d, scatter_2d, title_2d]
        
        anim = FuncAnimation(fig_anim, update_with_pulse, frames=n_frames, 
                             interval=1000/animation_fps, blit=False)

        # Save animation as mp4 only
        mp4_path = os.path.join(dir, "spin_anim_with_pulse.mp4")
        try:
            writer_mp4 = FFMpegWriter(fps=animation_fps, bitrate=2000)
            anim.save(mp4_path, writer=writer_mp4)
            print("  Saved: spin_anim_with_pulse.mp4")
        except Exception as e:
            print(f"  Warning: Could not save spin_anim_with_pulse.mp4 ({e})")
        
        plt.close(fig_anim)
    
    # =========================================================================
    # Save spectra to text files
    # =========================================================================
    np.savetxt(os.path.join(dir, "O_custom_spectrum.txt"),
               np.column_stack([omega, omega_THz, O_custom_spectrum]),
               header="omega(rad/sim_time) frequency(THz) O_custom_spectrum")
    np.savetxt(os.path.join(dir, "E1_spectrum.txt"),
               np.column_stack([omega, omega_THz, result['E1_spectrum']]),
               header="omega(rad/sim_time) frequency(THz) E1_spectrum")
    np.savetxt(os.path.join(dir, "E2_spectrum.txt"),
               np.column_stack([omega, omega_THz, result['E2_spectrum']]),
               header="omega(rad/sim_time) frequency(THz) E2_spectrum")
    np.savetxt(os.path.join(dir, "A1_spectrum.txt"),
               np.column_stack([omega, omega_THz, result['A1_spectrum']]),
               header="omega(rad/sim_time) frequency(THz) A1_spectrum")
    np.savetxt(os.path.join(dir, "time_series.txt"),
               np.column_stack([T, T_ps, O_custom, Qx_E1, Qy_E1, Qx_E2, Qy_E2, Q_A1, result['energy']]),
               header="time(sim_units) time(ps) O_custom Qx_E1 Qy_E1 Qx_E2 Qy_E2 Q_A1 energy")
    
    print(f"  Saved: spectrum and time series files")
    return result

def read_pump_probe_tot(dir, omega_max=25.0, order_parameter_func=None,
                        create_animation=True, animation_interval=10):
    """Aggregate pump-probe results from multiple runs.
    
    Processes all subdirectories containing trajectory.h5 and creates
    individual analysis + aggregated summary.
    
    Args:
        dir: Parent directory containing subdirectories with trajectory.h5
        omega_max: Maximum frequency for spectrum plots
        order_parameter_func: Optional callable(spins, positions, metadata) -> float
            If None, uses staggered magnetization as default.
        create_animation: Whether to create spin animations (default: True)
        animation_interval: Use every Nth frame for animation (default: 10)
    
    Outputs:
        - Individual analysis in each subdirectory
        - Aggregated spectra and summary in parent directory
    """
    directory = os.fsencode(dir)
    
    # Aggregated data
    all_E1_spectra = []
    all_A1_spectra = []
    all_O_custom_spectra = []
    omega_ref = None
    count = 0
    
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        subdir = os.path.join(dir, filename)
        
        if not os.path.isdir(subdir):
            continue
        
        hdf5_path = os.path.join(subdir, "trajectory.h5")
        if not os.path.exists(hdf5_path):
            continue
        
        print(f"Processing folder: {filename}")
        
        try:
            result = analyze_pump_probe(subdir, omega_max, 
                                       order_parameter_func=order_parameter_func,
                                       create_animation=create_animation,
                                       animation_interval=animation_interval)
            
            if omega_ref is None:
                omega_ref = result['omega']
            
            all_E1_spectra.append(result['E1_spectrum'])
            all_A1_spectra.append(result['A1_spectrum'])
            if result['O_custom_spectrum'] is not None:
                all_O_custom_spectra.append(result['O_custom_spectrum'])
            count += 1
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if count == 0:
        print("No valid pump-probe data found")
        return
    
    # Compute averages
    E1_avg = np.mean(all_E1_spectra, axis=0)
    A1_avg = np.mean(all_A1_spectra, axis=0)
    O_custom_avg = np.mean(all_O_custom_spectra, axis=0) if all_O_custom_spectra else None
    
    # Convert omega to THz
    omega_THz = omega_ref / (2 * np.pi * HBAR_OVER_MEV_PS)
    
    # Save aggregated spectra
    np.savetxt(os.path.join(dir, "E1_spectrum_avg.txt"),
               np.column_stack([omega_ref, omega_THz, E1_avg]),
               header="omega(rad/sim_time) frequency(THz) E1_spectrum_avg")
    np.savetxt(os.path.join(dir, "A1_spectrum_avg.txt"),
               np.column_stack([omega_ref, omega_THz, A1_avg]),
               header="omega(rad/sim_time) frequency(THz) A1_spectrum_avg")
    if O_custom_avg is not None:
        np.savetxt(os.path.join(dir, "O_custom_spectrum_avg.txt"),
                   np.column_stack([omega_ref, omega_THz, O_custom_avg]),
                   header="omega(rad/sim_time) frequency(THz) O_custom_spectrum_avg")
    
    # Create summary plot
    omega_pos = (omega_ref > 0) & (omega_ref < omega_max)
    
    n_panels = 3 if O_custom_avg is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 4))
    
    ax = axes[0]
    ax.semilogy(omega_THz[omega_pos], E1_avg[omega_pos], 'r-', lw=2)
    ax.set_xlabel('Frequency (THz)', fontsize=12)
    ax.set_ylabel('|FFT|²', fontsize=12)
    ax.set_title(f'E1 Mode Spectrum (avg of {count} runs)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.semilogy(omega_THz[omega_pos], A1_avg[omega_pos], 'b-', lw=2)
    ax.set_xlabel('Frequency (THz)', fontsize=12)
    ax.set_ylabel('|FFT|²', fontsize=12)
    ax.set_title(f'A1 Mode Spectrum (avg of {count} runs)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if O_custom_avg is not None:
        ax = axes[2]
        ax.semilogy(omega_THz[omega_pos], O_custom_avg[omega_pos], 'purple', lw=2)
        ax.set_xlabel('Frequency (THz)', fontsize=12)
        ax.set_ylabel('|FFT|²', fontsize=12)
        ax.set_title(f'Order Parameter Spectrum (avg of {count} runs)', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "pump_probe_spectra_avg.pdf"))
    plt.close()
    
    print(f"\nAggregated {count} pump-probe runs")
    print(f"Output saved to {dir}")

# ============================================================================
# MAIN FUNCTION 1: Aggregate MD data from Multiple Runs
# ============================================================================

def read_MD_tot(dir, order_parameter_func=None, compute_DSSF=False):
    """Aggregate MD data from multiple PhononLattice runs.
    
    Processes all subdirectories containing trajectory.h5,
    computes phonon spectra, custom order parameter, and optionally DSSF.
    
    Args:
        dir: Parent directory containing subdirectories with MD data
        order_parameter_func: Optional callable(spins, positions, metadata) -> float
            Function to compute custom order parameter from spin configuration.
            See staggered_magnetization_order() for example.
        compute_DSSF: If True, compute DSSF along k-path (expensive)
    
    Outputs:
        - phonon_spectra_avg.pdf: E1 and A1 mode spectra
        - O_custom_spectrum_avg.pdf: Custom order parameter spectrum (if func provided)
        - DSSF_line.pdf: DSSF along k-path (if compute_DSSF=True)
        - dynamics_summary.pdf: Per-run time-domain plots
    """
    directory = os.fsencode(dir)
    
    # Aggregated spectra
    E1_spectra = []
    A1_spectra = []
    O_custom_spectra = []
    DSSF_sum = None
    omega_ref = None
    
    w_line = np.arange(0.1, 5.0, 0.01)  # Frequency array for DSSF
    
    # Process each subdirectory
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        subdir = os.path.join(dir, filename)
        
        if not os.path.isdir(subdir) or filename.endswith('.pdf') or filename.endswith('.txt'):
            continue
        
        hdf5_path = os.path.join(subdir, "trajectory.h5")
        if not os.path.exists(hdf5_path):
            print(f"Skipping {filename}: no trajectory.h5")
            continue
        
        print(f"Processing folder: {filename}")
        
        try:
            # Read trajectory with custom order parameter
            result = read_MD_phonon(subdir, order_parameter_func=order_parameter_func)
            
            if omega_ref is None:
                omega_ref = result['omega']
            
            E1_spectra.append(result['E1_spectrum'])
            A1_spectra.append(result['A1_spectrum'])
            
            if result['O_custom_spectrum'] is not None:
                O_custom_spectra.append(result['O_custom_spectrum'])
            
            # Save individual run spectra
            np.savetxt(os.path.join(subdir, "E1_spectrum.txt"), 
                       np.column_stack([result['omega'], result['E1_spectrum']]))
            np.savetxt(os.path.join(subdir, "A1_spectrum.txt"),
                       np.column_stack([result['omega'], result['A1_spectrum']]))
            
            if result['O_custom'] is not None:
                np.savetxt(os.path.join(subdir, "O_custom.txt"),
                           np.column_stack([result['time'], result['O_custom']]))
                np.savetxt(os.path.join(subdir, "O_custom_spectrum.txt"),
                           np.column_stack([result['omega'], result['O_custom_spectrum']]))
            
            # Compute DSSF if requested
            if compute_DSSF:
                print(f"  Computing DSSF for {filename}...")
                dssf = DSSF(w_line, DSSF_K, result['spins'], result['positions'], 
                           result['time'], gb=True)
                if DSSF_sum is None:
                    DSSF_sum = dssf
                else:
                    DSSF_sum += dssf
                np.savetxt(os.path.join(subdir, "DSSF.txt"), dssf)
            
            # Create individual summary plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Time domain: Order parameter (if available) or magnetization
            ax = axes[0, 0]
            if result['O_custom'] is not None:
                ax.plot(result['time'], result['O_custom'], 'b-', lw=1)
                ax.set_ylabel('Order Parameter')
                ax.set_title('Custom Order Parameter vs Time')
            else:
                M_stag = result['M_antiferro']
                ax.plot(result['time'], np.linalg.norm(M_stag, axis=1), 'b-', lw=1)
                ax.set_ylabel('|M_staggered|')
                ax.set_title('Staggered Magnetization vs Time')
            ax.set_xlabel('Time')
            ax.grid(True, alpha=0.3)
            
            ax = axes[0, 1]
            # Use new naming with fallback to legacy
            Qx_E1 = result.get('Qx_E1', result.get('Qx'))
            Qy_E1 = result.get('Qy_E1', result.get('Qy'))
            Qx_E2 = result.get('Qx_E2')
            Qy_E2 = result.get('Qy_E2')
            Q_A1 = result.get('Q_A1', result.get('QR'))
            
            ax.plot(result['time'], Qx_E1, 'r-', alpha=0.7, label='Qx_E1')
            ax.plot(result['time'], Qy_E1, 'orange', alpha=0.7, label='Qy_E1')
            if Qx_E2 is not None:
                ax.plot(result['time'], Qx_E2, 'g-', alpha=0.7, label='Qx_E2')
                ax.plot(result['time'], Qy_E2, 'c-', alpha=0.7, label='Qy_E2')
            ax.plot(result['time'], Q_A1, 'b-', label='Q_A1')
            ax.set_xlabel('Time')
            ax.set_ylabel('Phonon Coordinate')
            ax.legend(fontsize=8)
            ax.set_title('Phonon Dynamics')
            ax.grid(True, alpha=0.3)
            
            # Frequency domain plots
            ax = axes[1, 0]
            omega_pos = result['omega'] > 0
            ax.semilogy(result['omega'][omega_pos], result['E1_spectrum'][omega_pos], 'r-', label='E1')
            if 'E2_spectrum' in result:
                ax.semilogy(result['omega'][omega_pos], result['E2_spectrum'][omega_pos], 'g-', label='E2')
            ax.semilogy(result['omega'][omega_pos], result['A1_spectrum'][omega_pos], 'b-', label='A1')
            ax.set_xlabel('ω (rad/time)')
            ax.set_ylabel('|FFT|²')
            ax.legend(fontsize=8)
            ax.set_title('Phonon Spectrum')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 20])
            
            ax = axes[1, 1]
            if result['O_custom_spectrum'] is not None:
                ax.semilogy(result['omega'][omega_pos], result['O_custom_spectrum'][omega_pos])
                ax.set_title('Order Parameter Spectrum')
            else:
                # Use staggered magnetization norm spectrum
                M_stag_norm = np.linalg.norm(result['M_antiferro'], axis=1)
                M_stag_fft = np.fft.fftshift(np.fft.fft(M_stag_norm - np.mean(M_stag_norm)))
                ax.semilogy(result['omega'][omega_pos], np.abs(M_stag_fft)[omega_pos]**2)
                ax.set_title('|M_staggered| Spectrum')
            ax.set_xlabel('ω (rad/time)')
            ax.set_ylabel('|FFT|²')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 20])
            
            plt.tight_layout()
            plt.savefig(os.path.join(subdir, "dynamics_summary.pdf"))
            plt.close()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(E1_spectra) == 0:
        print("No valid MD data found")
        return
    
    # Aggregate and average
    E1_avg = np.mean(E1_spectra, axis=0)
    A1_avg = np.mean(A1_spectra, axis=0)
    O_custom_avg = np.mean(O_custom_spectra, axis=0) if O_custom_spectra else None
    
    # Convert omega to THz
    omega_THz = omega_ref / (2 * np.pi * HBAR_OVER_MEV_PS)
    
    # Save aggregated spectra (with both simulation units and THz)
    np.savetxt(os.path.join(dir, "E1_spectrum_avg.txt"), 
               np.column_stack([omega_ref, omega_THz, E1_avg]),
               header="omega(rad/sim_time) frequency(THz) E1_spectrum")
    np.savetxt(os.path.join(dir, "A1_spectrum_avg.txt"),
               np.column_stack([omega_ref, omega_THz, A1_avg]),
               header="omega(rad/sim_time) frequency(THz) A1_spectrum")
    
    if O_custom_avg is not None:
        np.savetxt(os.path.join(dir, "O_custom_spectrum_avg.txt"),
                   np.column_stack([omega_ref, omega_THz, O_custom_avg]),
                   header="omega(rad/sim_time) frequency(THz) O_custom_spectrum")
    
    # Create combined plots
    omega_pos = omega_ref > 0
    
    # Phonon spectra
    n_plots = 3 if O_custom_avg is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    if n_plots == 2:
        axes = list(axes)
    
    ax = axes[0]
    ax.semilogy(omega_THz[omega_pos], E1_avg[omega_pos], 'b-', lw=2)
    ax.set_xlabel('Frequency (THz)', fontsize=12)
    ax.set_ylabel('|FFT|²', fontsize=12)
    ax.set_title('E1 Mode Spectrum (Averaged)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5])  # ~5 THz reasonable for phonon modes
    
    ax = axes[1]
    ax.semilogy(omega_THz[omega_pos], A1_avg[omega_pos], 'r-', lw=2)
    ax.set_xlabel('Frequency (THz)', fontsize=12)
    ax.set_ylabel('|FFT|²', fontsize=12)
    ax.set_title('A1 Mode Spectrum (Averaged)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5])
    
    if O_custom_avg is not None:
        ax = axes[2]
        ax.semilogy(omega_THz[omega_pos], O_custom_avg[omega_pos], 'g-', lw=2)
        ax.set_xlabel('Frequency (THz)', fontsize=12)
        ax.set_ylabel('|FFT|²', fontsize=12)
        ax.set_title('Order Parameter Spectrum (Averaged)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "spectra_avg.pdf"))
    plt.close()
    
    # Plot DSSF if computed
    if DSSF_sum is not None:
        DSSF_avg = DSSF_sum / len(E1_spectra)
        np.savetxt(os.path.join(dir, "DSSF_avg.txt"), DSSF_avg)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        C = ax.imshow(DSSF_avg, origin='lower', extent=[0, len(DSSF_K), w_line[0], w_line[-1]], 
                      aspect='auto', interpolation='lanczos', cmap='gnuplot2')
        ax.axvline(x=gGamma, color='w', linestyle='dashed', alpha=0.5)
        ax.axvline(x=gM1M2, color='w', linestyle='dashed', alpha=0.5)
        ax.axvline(x=gGamma1, color='w', linestyle='dashed', alpha=0.5)
        ax.axvline(x=gK1K2, color='w', linestyle='dashed', alpha=0.5)
        ax.axvline(x=gM1, color='w', linestyle='dashed', alpha=0.5)
        
        xlabpos = [gGamma, gM1M2, gGamma1, gK1K2, gGamma1Gamma2, gM1, gK1]
        labels = [r'$\Gamma$', r'M', r'$\Gamma$', r'K', r'$\Gamma$', r'M', r'K']
        ax.set_xticks(xlabpos)
        ax.set_xticklabels(labels)
        ax.set_ylabel('ω (meV)')
        fig.colorbar(C, label='log(DSSF)')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, "DSSF_line_avg.pdf"))
        plt.close()
    
    print(f"\nAggregated {len(E1_spectra)} runs")
    print(f"Output saved to {dir}")

# ============================================================================
# MAIN FUNCTION 2: Aggregate 2D Nonlinear Spectroscopy
# ============================================================================

def read_2D_nonlinear_tot(dir, omega_t_window=None, omega_tau_window=None):
    """Aggregate 2D nonlinear spectroscopy from multiple PhononLattice runs.
    
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
        subdir = os.path.join(dir, filename)
        
        if not os.path.isdir(subdir):
            continue
        
        hdf5_path = os.path.join(subdir, "pump_probe_spectroscopy.h5")
        if not os.path.exists(hdf5_path):
            print(f"Skipping {filename}: no pump_probe_spectroscopy.h5")
            continue
        
        print(f"Processing folder: {filename}")
        
        try:
            read_2D_nonlinear_phonon(subdir,
                                      omega_t_window=omega_t_window,
                                      omega_tau_window=omega_tau_window)
            M_NL_data = np.loadtxt(os.path.join(subdir, "M_NL_FF.txt"))
            
            if A is None:
                A = M_NL_data
            else:
                min_shape = (min(A.shape[0], M_NL_data.shape[0]),
                            min(A.shape[1], M_NL_data.shape[1]))
                A[:min_shape[0], :min_shape[1]] += M_NL_data[:min_shape[0], :min_shape[1]]
            count += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if A is not None and count > 0:
        A = A / count  # Average over all runs
        np.savetxt(os.path.join(dir, "M_NL_tot.txt"), A)
        
        # Plot aggregated result
        # Note: The extent is now set by the individual processing function in THz
        # For the aggregated plot, we need to recalculate the extent
        plt.imshow(A, origin='lower', aspect='auto', cmap='gnuplot2', norm='linear')
        plt.xlabel('$\\omega_t$ (THz)')
        plt.ylabel('$\\omega_{\\tau}$ (THz)')
        plt.colorbar(label='Intensity')
        # Note: omega_t/tau_window are in simulation units, convert to THz for limits
        if omega_t_window is not None:
            omega_t_window_THz = (omega_t_window[0] / (2 * np.pi * HBAR_OVER_MEV_PS),
                                  omega_t_window[1] / (2 * np.pi * HBAR_OVER_MEV_PS))
            plt.xlim(omega_t_window_THz)
        if omega_tau_window is not None:
            omega_tau_window_THz = (omega_tau_window[0] / (2 * np.pi * HBAR_OVER_MEV_PS),
                                    omega_tau_window[1] / (2 * np.pi * HBAR_OVER_MEV_PS))
            plt.ylim(omega_tau_window_THz)
        plt.savefig(os.path.join(dir, "NLSPEC_tot.pdf"))
        plt.clf()
        print(f"Aggregated {count} runs")
    else:
        print("No valid data found")

# ============================================================================
# MAIN FUNCTION 3: Parse and Visualize Spin Configurations
# ============================================================================

def parse_spin_config(dir):
    """Parse and visualize spin configurations from PhononLattice outputs.
    
    Processes all subdirectories containing spin configuration data
    (from simulated annealing or MD snapshots).
    
    Args:
        dir: Parent directory containing subdirectories with spin data
    
    Outputs (per subdirectory):
        - spin_config.pdf: 3D plot of spin configuration
        - spin_info.txt: Statistics about spin configuration
    """
    directory = os.fsencode(dir)
    
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        subdir = os.path.join(dir, filename)
        
        if not os.path.isdir(subdir):
            continue
        
        print(f"Processing folder: {filename}")
        
        try:
            # Try to load spin configuration
            S = None
            P = None
            
            # Check for various spin config file names
            spin_files = ['spins_final.txt', 'spins.txt', 'spins_initial.txt', 'initial_spins.txt']
            pos_file = 'positions.txt'
            
            for spin_file in spin_files:
                spin_path = os.path.join(subdir, spin_file)
                if os.path.exists(spin_path):
                    S = np.loadtxt(spin_path)
                    print(f"  Loaded spins from: {spin_file}")
                    break
            
            pos_path = os.path.join(subdir, pos_file)
            if os.path.exists(pos_path):
                P = np.loadtxt(pos_path)
                print(f"  Loaded positions from: {pos_file}")
            
            if S is None or P is None:
                print(f"  Skipping: No valid spin/position data found")
                continue
            
            # Compute statistics
            n_atoms = len(S)
            spin_mag = np.linalg.norm(S, axis=1)
            avg_mag = np.mean(spin_mag)
            total_mag = np.sum(S, axis=0)
            
            # Staggered magnetization (alternating sublattices)
            stag_mag = np.zeros(3)
            for i in range(n_atoms):
                sign = 1.0 if i % 2 == 0 else -1.0
                stag_mag += sign * S[i]
            
            print(f"  Spin Configuration Summary:")
            print(f"    Number of atoms: {n_atoms}")
            print(f"    Average spin magnitude: {avg_mag:.6f}")
            print(f"    Total magnetization: [{total_mag[0]:.6f}, {total_mag[1]:.6f}, {total_mag[2]:.6f}]")
            print(f"    |M_total|: {np.linalg.norm(total_mag):.6f}")
            print(f"    Staggered magnetization: [{stag_mag[0]:.6f}, {stag_mag[1]:.6f}, {stag_mag[2]:.6f}]")
            print(f"    |M_stag|: {np.linalg.norm(stag_mag):.6f}")
            
            # Save statistics
            with open(os.path.join(subdir, "spin_info.txt"), 'w') as f:
                f.write(f"Number of atoms: {n_atoms}\n")
                f.write(f"Average spin magnitude: {avg_mag:.6f}\n")
                f.write(f"Total magnetization: [{total_mag[0]:.6f}, {total_mag[1]:.6f}, {total_mag[2]:.6f}]\n")
                f.write(f"|M_total|: {np.linalg.norm(total_mag):.6f}\n")
                f.write(f"Staggered magnetization: [{stag_mag[0]:.6f}, {stag_mag[1]:.6f}, {stag_mag[2]:.6f}]\n")
                f.write(f"|M_stag|: {np.linalg.norm(stag_mag):.6f}\n")
            
            # Create 3D visualization
            fig = plt.figure(figsize=(12, 10))
            
            # 3D spin plot
            ax = fig.add_subplot(221, projection='3d')
            
            # Color by sublattice
            colors = ['blue' if i % 2 == 0 else 'red' for i in range(n_atoms)]
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=colors, s=100, alpha=0.6, edgecolors='k')
            
            # Plot spins as arrows
            ax.quiver(P[:, 0], P[:, 1], P[:, 2],
                      S[:, 0], S[:, 1], S[:, 2],
                      color='green', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.8)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Spin Configuration ({n_atoms} atoms)')
            
            # 2D projection (xy plane)
            ax2 = fig.add_subplot(222)
            colors_arr = np.array([0 if i % 2 == 0 else 1 for i in range(n_atoms)])
            scatter = ax2.scatter(P[:, 0], P[:, 1], c=colors_arr, cmap='coolwarm', s=50, alpha=0.7)
            ax2.quiver(P[:, 0], P[:, 1], S[:, 0], S[:, 1], 
                       scale=20, width=0.005, color='black', alpha=0.8)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('XY Projection')
            ax2.set_aspect('equal')
            
            # Spin magnitude histogram
            ax3 = fig.add_subplot(223)
            ax3.hist(spin_mag, bins=50, edgecolor='black', alpha=0.7)
            ax3.axvline(avg_mag, color='r', linestyle='--', label=f'Mean: {avg_mag:.4f}')
            ax3.set_xlabel('Spin Magnitude')
            ax3.set_ylabel('Count')
            ax3.set_title('Spin Magnitude Distribution')
            ax3.legend()
            
            # Spin component histogram
            ax4 = fig.add_subplot(224)
            ax4.hist(S[:, 0], bins=30, alpha=0.5, label='Sx', color='r')
            ax4.hist(S[:, 1], bins=30, alpha=0.5, label='Sy', color='g')
            ax4.hist(S[:, 2], bins=30, alpha=0.5, label='Sz', color='b')
            ax4.set_xlabel('Spin Component')
            ax4.set_ylabel('Count')
            ax4.set_title('Spin Component Distribution')
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(subdir, "spin_config.pdf"))
            plt.close()
            print(f"  Saved: spin_config.pdf, spin_info.txt\n")
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            continue

# ============================================================================
# ADDITIONAL ANALYSIS: Simulated Annealing Results
# ============================================================================

def read_annealing(dir):
    """Read and plot simulated annealing results from PhononLattice.
    
    Args:
        dir: Directory containing annealing.h5
    
    Outputs:
        - annealing_curve.pdf: Temperature vs Energy plot
    """
    hdf5_path = os.path.join(dir, "annealing.h5")
    
    if not os.path.exists(hdf5_path):
        print(f"No annealing.h5 found in {dir}")
        return None
    
    with h5py.File(hdf5_path, 'r') as f:
        if '/annealing' in f:
            grp = f['/annealing']
            temps = grp['temperature'][:]
            energies = grp['energy'][:]
            acc_rates = grp['acceptance_rate'][:]
            
            print(f"  Annealing data:")
            print(f"    Temperature range: {temps[0]:.4f} → {temps[-1]:.6f}")
            print(f"    Final energy: {energies[-1]:.6f}")
    
    # Plot annealing curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.semilogx(temps, energies, 'b-', lw=2)
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Energy Density', fontsize=12)
    ax.set_title('Annealing: Energy vs Temperature', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    ax = axes[1]
    ax.semilogx(temps, acc_rates, 'r-', lw=2)
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Acceptance Rate', fontsize=12)
    ax.set_title('Annealing: Acceptance vs Temperature', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "annealing_curve.pdf"))
    plt.close()
    
    return {
        'temperature': temps,
        'energy': energies,
        'acceptance_rate': acc_rates
    }

def read_annealing_tot(dir):
    """Aggregate annealing results from multiple runs.
    
    Args:
        dir: Parent directory containing subdirectories with annealing.h5
    """
    directory = os.fsencode(dir)
    
    all_final_energies = []
    
    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        subdir = os.path.join(dir, filename)
        
        if not os.path.isdir(subdir):
            continue
        
        hdf5_path = os.path.join(subdir, "annealing.h5")
        if not os.path.exists(hdf5_path):
            continue
        
        print(f"Processing folder: {filename}")
        
        try:
            result = read_annealing(subdir)
            if result is not None:
                all_final_energies.append(result['energy'][-1])
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    if len(all_final_energies) > 0:
        print(f"\nAnnealing Summary ({len(all_final_energies)} runs):")
        print(f"  Mean final energy: {np.mean(all_final_energies):.6f}")
        print(f"  Std final energy: {np.std(all_final_energies):.6f}")
        print(f"  Min final energy: {np.min(all_final_energies):.6f}")
        print(f"  Max final energy: {np.max(all_final_energies):.6f}")

# ============================================================================
# MAIN - Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Phonon Lattice Analysis Tools")
        print("=" * 50)
        print("\nUsage: python reader_phonon_lattice.py <directory> [function]")
        print("\nAvailable functions:")
        print("  md        - Aggregate MD runs with trajectory.h5 (default)")
        print("  pp        - Analyze pump-probe trajectory.h5 (single THz pulse)")
        print("  2dcs      - Aggregate 2D nonlinear spectroscopy (tau delay scan)")
        print("  spin      - Parse and visualize spin configurations")
        print("  sa        - Aggregate simulated annealing results")
        print("\nExamples:")
        print("  python reader_phonon_lattice.py ./NCTO_pump_probe/ pp")
        print("  python reader_phonon_lattice.py ./NCTO_2dcs/ 2dcs")
        print("  python reader_phonon_lattice.py ./NCTO_sa/ spin")
        print("  python reader_phonon_lattice.py ./NCTO_sa/ sa")
        sys.exit(1)
    
    directory = sys.argv[1]
    function = sys.argv[2] if len(sys.argv) > 2 else "md"
    
    if function == "md":
        print("Running: read_MD_tot()")
        read_MD_tot(directory)
    elif function == "pp":
        print("Running: read_pump_probe_tot()")
        read_pump_probe_tot(directory)
    elif function == "2dcs":
        print("Running: read_2D_nonlinear_tot()")
        read_2D_nonlinear_tot(directory, (-10, 10), (-10, 10))
    elif function == "spin":
        print("Running: parse_spin_config()")
        parse_spin_config(directory)
    elif function == "sa":
        print("Running: read_annealing_tot()")
        read_annealing_tot(directory)
    else:
        print(f"Unknown function: {function}")
        print("Available: md, pp, 2dcs, spin, sa")

