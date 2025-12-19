"""
Feature Extractor for BCAO Honeycomb Spin Configurations.

Extracts normalized features from spin configurations for use in 
machine learning classifiers and active learning.

All parameters are normalized by J1 (taken as the energy scale),
so J1 itself becomes ±1 (sign preserved).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json


@dataclass
class NormalizedParameters:
    """
    BCAO parameters normalized by |J1xy|.
    
    Original parameters: [J1xy, J1z, D, E, F, G, J3xy, J3z]
    Normalized: J1xy_sign ∈ {-1, +1}, all others ∈ [-1, 1]
    """
    # Sign of J1xy (FM vs AFM tendency)
    J1xy_sign: float = -1.0  # Usually -1 for FM
    
    # Normalized parameters (divided by |J1xy|)
    J1z_norm: float = 0.0
    D_norm: float = 0.0
    E_norm: float = 0.0
    F_norm: float = 0.0
    G_norm: float = 0.0
    J3xy_norm: float = 0.0
    J3z_norm: float = 0.0
    
    # Original scale for reference
    J1xy_abs: float = 1.0
    
    @classmethod
    def from_raw(cls, J: List[float]) -> 'NormalizedParameters':
        """
        Create normalized parameters from raw J array.
        
        Args:
            J: [J1xy, J1z, D, E, F, G, J3xy, J3z]
        """
        J1xy, J1z, D, E, F, G, J3xy, J3z = J
        
        J1xy_abs = abs(J1xy)
        if J1xy_abs < 1e-10:
            J1xy_abs = 1.0  # Avoid division by zero
        
        return cls(
            J1xy_sign=np.sign(J1xy) if J1xy != 0 else -1.0,
            J1z_norm=J1z / J1xy_abs,
            D_norm=D / J1xy_abs,
            E_norm=E / J1xy_abs,
            F_norm=F / J1xy_abs,
            G_norm=G / J1xy_abs,
            J3xy_norm=J3xy / J1xy_abs,
            J3z_norm=J3z / J1xy_abs,
            J1xy_abs=J1xy_abs
        )
    
    def to_raw(self) -> List[float]:
        """Convert back to raw J array."""
        J1xy = self.J1xy_sign * self.J1xy_abs
        return [
            J1xy,
            self.J1z_norm * self.J1xy_abs,
            self.D_norm * self.J1xy_abs,
            self.E_norm * self.J1xy_abs,
            self.F_norm * self.J1xy_abs,
            self.G_norm * self.J1xy_abs,
            self.J3xy_norm * self.J1xy_abs,
            self.J3z_norm * self.J1xy_abs,
        ]
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        return np.array([
            self.J1z_norm,
            self.D_norm,
            self.E_norm,
            self.F_norm,
            self.G_norm,
            self.J3xy_norm,
            self.J3z_norm,
        ])
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'J1xy_sign': self.J1xy_sign,
            'J1z_norm': self.J1z_norm,
            'D_norm': self.D_norm,
            'E_norm': self.E_norm,
            'F_norm': self.F_norm,
            'G_norm': self.G_norm,
            'J3xy_norm': self.J3xy_norm,
            'J3z_norm': self.J3z_norm,
            'J1xy_abs': self.J1xy_abs,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'NormalizedParameters':
        """Create from dictionary."""
        return cls(**d)
    
    def __repr__(self):
        return (f"NormalizedParams(J1z={self.J1z_norm:.3f}, D={self.D_norm:.3f}, "
                f"E={self.E_norm:.3f}, F={self.F_norm:.3f}, G={self.G_norm:.3f}, "
                f"J3xy={self.J3xy_norm:.3f}, J3z={self.J3z_norm:.3f})")


# Feature names for ML models
FEATURE_NAMES = [
    'J1z_norm',    # XXZ anisotropy
    'D_norm',      # Single-ion anisotropy (Dx-Dy)/2
    'E_norm',      # Off-diagonal exchange xy
    'F_norm',      # Off-diagonal exchange xz
    'G_norm',      # Off-diagonal exchange yz
    'J3xy_norm',   # Third-neighbor XY
    'J3z_norm',    # Third-neighbor Z
]

# Default bounds for normalized parameters (all |x| < 1)
DEFAULT_BOUNDS = {
    'J1z_norm': (-1.0, 1.0),
    'D_norm': (-1.0, 1.0),
    'E_norm': (-1.0, 1.0),
    'F_norm': (-1.0, 1.0),
    'G_norm': (-1.0, 1.0),
    'J3xy_norm': (-1.0, 1.0),
    'J3z_norm': (-1.0, 1.0),
}


@dataclass
class FeatureSet:
    """
    Complete feature set for a spin configuration.
    
    Includes:
    - Normalized Hamiltonian parameters
    - Observable features (energy, magnetization, etc.)
    - Structure factor features
    - Topological features
    """
    # Hamiltonian parameters
    params: NormalizedParameters = field(default_factory=NormalizedParameters)
    
    # Energy features
    energy_per_site: float = 0.0
    
    # Magnetization features
    total_mag: float = 0.0
    total_mag_x: float = 0.0
    total_mag_y: float = 0.0
    total_mag_z: float = 0.0
    staggered_mag: float = 0.0
    sublattice_A_mag: float = 0.0
    sublattice_B_mag: float = 0.0
    
    # Structure factor features
    dominant_Q1: float = 0.0  # First reduced coordinate
    dominant_Q2: float = 0.0  # Second reduced coordinate
    Q_magnitude: float = 0.0
    num_peaks: int = 0
    peak_ratio_1: float = 0.0  # Ratio of 2nd to 1st peak
    peak_ratio_2: float = 0.0  # Ratio of 3rd to 1st peak
    
    # Q-vector location flags (one-hot style)
    Q_at_gamma: float = 0.0
    Q_at_M: float = 0.0
    Q_at_K: float = 0.0
    Q_incommensurate: float = 0.0
    
    # Multi-Q features
    is_multi_Q: float = 0.0
    Q_perpendicular: float = 0.0
    second_Q_mag: float = 0.0
    
    # Topological features
    skyrmion_number: float = 0.0
    winding_variance: float = 0.0
    vortex_density: float = 0.0
    
    # Real-space pattern features
    is_coplanar: float = 0.0
    is_collinear: float = 0.0
    
    def to_full_vector(self) -> np.ndarray:
        """Convert to full feature vector including Hamiltonian params."""
        param_vec = self.params.to_feature_vector()
        obs_vec = np.array([
            self.energy_per_site,
            self.total_mag,
            self.total_mag_x,
            self.total_mag_y,
            self.total_mag_z,
            self.staggered_mag,
            self.sublattice_A_mag,
            self.sublattice_B_mag,
            self.dominant_Q1,
            self.dominant_Q2,
            self.Q_magnitude,
            self.num_peaks,
            self.peak_ratio_1,
            self.peak_ratio_2,
            self.Q_at_gamma,
            self.Q_at_M,
            self.Q_at_K,
            self.Q_incommensurate,
            self.is_multi_Q,
            self.Q_perpendicular,
            self.second_Q_mag,
            self.skyrmion_number,
            self.winding_variance,
            self.vortex_density,
            self.is_coplanar,
            self.is_collinear,
        ])
        return np.concatenate([param_vec, obs_vec])
    
    def to_param_only_vector(self) -> np.ndarray:
        """Get only Hamiltonian parameter features."""
        return self.params.to_feature_vector()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            'params': self.params.to_dict(),
            'energy_per_site': self.energy_per_site,
            'total_mag': self.total_mag,
            'total_mag_x': self.total_mag_x,
            'total_mag_y': self.total_mag_y,
            'total_mag_z': self.total_mag_z,
            'staggered_mag': self.staggered_mag,
            'sublattice_A_mag': self.sublattice_A_mag,
            'sublattice_B_mag': self.sublattice_B_mag,
            'dominant_Q1': self.dominant_Q1,
            'dominant_Q2': self.dominant_Q2,
            'Q_magnitude': self.Q_magnitude,
            'num_peaks': self.num_peaks,
            'peak_ratio_1': self.peak_ratio_1,
            'peak_ratio_2': self.peak_ratio_2,
            'Q_at_gamma': self.Q_at_gamma,
            'Q_at_M': self.Q_at_M,
            'Q_at_K': self.Q_at_K,
            'Q_incommensurate': self.Q_incommensurate,
            'is_multi_Q': self.is_multi_Q,
            'Q_perpendicular': self.Q_perpendicular,
            'second_Q_mag': self.second_Q_mag,
            'skyrmion_number': self.skyrmion_number,
            'winding_variance': self.winding_variance,
            'vortex_density': self.vortex_density,
            'is_coplanar': self.is_coplanar,
            'is_collinear': self.is_collinear,
        }
        return d


class FeatureExtractor:
    """
    Extract features from spin configurations for ML classification.
    """
    
    def __init__(self, L: int = 12, J1xy_ref: float = None):
        """
        Initialize feature extractor.
        
        Args:
            L: Lattice size (L x L unit cells)
            J1xy_ref: Reference J1xy value for normalization. If None, use |J1xy| from each sample.
        """
        self.L = L
        self.J1xy_ref = J1xy_ref
        
        # Reciprocal lattice vectors
        self.b1 = 2 * np.pi * np.array([1.0, -1.0 / np.sqrt(3.0)])
        self.b2 = 2 * np.pi * np.array([0.0, 2.0 / np.sqrt(3.0)])
    
    def extract(self, spins: np.ndarray, positions: np.ndarray,
                J: List[float], energy: float = None) -> FeatureSet:
        """
        Extract complete feature set from a spin configuration.
        
        Args:
            spins: (N, 3) spin vectors
            positions: (N, 2) site positions
            J: Raw BCAO parameters [J1xy, J1z, D, E, F, G, J3xy, J3z]
            energy: Energy per site (computed if None)
            
        Returns:
            FeatureSet with all extracted features
        """
        features = FeatureSet()
        
        # Normalize parameters
        features.params = NormalizedParameters.from_raw(J)
        
        # Energy
        if energy is not None:
            features.energy_per_site = energy
        
        # Magnetization features
        self._extract_magnetization_features(spins, features)
        
        # Structure factor features
        self._extract_structure_factor_features(spins, positions, features)
        
        # Real-space pattern features
        self._extract_realspace_features(spins, features)
        
        # Topological features (only if multi-Q)
        if features.is_multi_Q > 0.5:
            self._extract_topological_features(spins, positions, features)
        
        return features
    
    def _extract_magnetization_features(self, spins: np.ndarray, 
                                         features: FeatureSet):
        """Extract magnetization-related features."""
        N = len(spins)
        
        # Total magnetization
        total_mag_vec = np.mean(spins, axis=0)
        features.total_mag = np.linalg.norm(total_mag_vec)
        features.total_mag_x = total_mag_vec[0]
        features.total_mag_y = total_mag_vec[1]
        features.total_mag_z = total_mag_vec[2]
        
        # Sublattice magnetizations
        spins_A = spins[::2]
        spins_B = spins[1::2]
        mag_A = np.mean(spins_A, axis=0)
        mag_B = np.mean(spins_B, axis=0)
        
        features.sublattice_A_mag = np.linalg.norm(mag_A)
        features.sublattice_B_mag = np.linalg.norm(mag_B)
        features.staggered_mag = np.linalg.norm(mag_A - mag_B)
    
    def _extract_structure_factor_features(self, spins: np.ndarray,
                                            positions: np.ndarray,
                                            features: FeatureSet):
        """Extract structure factor features."""
        N = len(spins)
        
        # Compute S(q) on a grid
        n_q = min(self.L, 32)
        q1_vals = np.linspace(0, 0.5, n_q)
        q2_vals = np.linspace(0, 0.5, n_q)
        
        S_q = np.zeros((n_q, n_q))
        
        for i, q1 in enumerate(q1_vals):
            for j, q2 in enumerate(q2_vals):
                q = q1 * self.b1 + q2 * self.b2
                phase_factors = np.exp(1j * positions @ q)
                S_q_vec = np.sum(spins * phase_factors[:, np.newaxis], axis=0)
                S_q[i, j] = np.sum(np.abs(S_q_vec)**2) / N
        
        # Find peaks
        peaks = []
        for i in range(1, n_q - 1):
            for j in range(1, n_q - 1):
                if (S_q[i, j] > S_q[i-1, j] and S_q[i, j] > S_q[i+1, j] and
                    S_q[i, j] > S_q[i, j-1] and S_q[i, j] > S_q[i, j+1] and
                    S_q[i, j] > 0.1 * np.max(S_q)):
                    peaks.append((q1_vals[i], q2_vals[j], S_q[i, j]))
        
        # Check Gamma point
        if S_q[0, 0] > 0.5 * np.max(S_q):
            peaks.append((0.0, 0.0, S_q[0, 0]))
        
        # Sort by intensity
        peaks = sorted(peaks, key=lambda x: -x[2])
        
        features.num_peaks = len(peaks)
        
        if len(peaks) > 0:
            features.dominant_Q1 = peaks[0][0]
            features.dominant_Q2 = peaks[0][1]
            features.Q_magnitude = np.linalg.norm(
                peaks[0][0] * self.b1 + peaks[0][1] * self.b2
            )
            
            # Classify Q location
            q_red = np.array([peaks[0][0], peaks[0][1]])
            Q_TOL = 0.05
            
            if np.linalg.norm(q_red) < Q_TOL:
                features.Q_at_gamma = 1.0
            elif self._is_at_m(q_red, Q_TOL):
                features.Q_at_M = 1.0
            elif self._is_at_k(q_red, Q_TOL):
                features.Q_at_K = 1.0
            else:
                features.Q_incommensurate = 1.0
        
        if len(peaks) >= 2:
            features.peak_ratio_1 = peaks[1][2] / peaks[0][2]
            features.second_Q_mag = np.linalg.norm(
                peaks[1][0] * self.b1 + peaks[1][1] * self.b2
            )
            
            # Check if multi-Q
            if features.peak_ratio_1 > 0.3:
                features.is_multi_Q = 1.0
                
                # Check perpendicularity
                q1 = peaks[0][0] * self.b1 + peaks[0][1] * self.b2
                q2 = peaks[1][0] * self.b1 + peaks[1][1] * self.b2
                if np.linalg.norm(q1) > 1e-6 and np.linalg.norm(q2) > 1e-6:
                    dot = abs(np.dot(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2))
                    if dot < 0.2:
                        features.Q_perpendicular = 1.0
        
        if len(peaks) >= 3:
            features.peak_ratio_2 = peaks[2][2] / peaks[0][2]
    
    def _extract_realspace_features(self, spins: np.ndarray,
                                      features: FeatureSet):
        """Extract real-space pattern features."""
        # Coplanarity
        features.is_coplanar = float(self._check_coplanar(spins))
        
        # Collinearity
        features.is_collinear = float(self._check_collinear(spins))
    
    def _extract_topological_features(self, spins: np.ndarray,
                                        positions: np.ndarray,
                                        features: FeatureSet):
        """Extract topological features."""
        # Simplified topological charge calculation
        charges = []
        N = len(spins)
        
        for i in range(0, N - 2, 2):
            S1 = spins[i]
            S2 = spins[i + 1]
            S3 = spins[min(i + 2, N - 1)]
            
            omega = self._solid_angle(S1, S2, S3)
            charges.append(omega / (4 * np.pi))
        
        if len(charges) > 0:
            charges = np.array(charges)
            features.skyrmion_number = np.sum(charges)
            features.winding_variance = np.var(charges)
            sign_changes = np.sum(np.abs(np.diff(np.sign(charges))))
            features.vortex_density = sign_changes / len(charges)
    
    def _solid_angle(self, S1: np.ndarray, S2: np.ndarray, S3: np.ndarray) -> float:
        """Solid angle of three unit vectors."""
        S1 = S1 / (np.linalg.norm(S1) + 1e-10)
        S2 = S2 / (np.linalg.norm(S2) + 1e-10)
        S3 = S3 / (np.linalg.norm(S3) + 1e-10)
        
        num = np.dot(S1, np.cross(S2, S3))
        den = 1 + np.dot(S1, S2) + np.dot(S2, S3) + np.dot(S3, S1)
        
        if abs(den) < 1e-10:
            return 0.0
        return 2 * np.arctan2(num, den)
    
    def _check_coplanar(self, spins: np.ndarray, tol: float = 0.1) -> bool:
        """Check coplanarity."""
        if len(spins) < 3:
            return True
        
        normal = None
        for i in range(1, min(len(spins), 10)):
            cross = np.cross(spins[0], spins[i])
            if np.linalg.norm(cross) > tol:
                normal = cross / np.linalg.norm(cross)
                break
        
        if normal is None:
            return True
        
        for spin in spins[:50]:  # Check subset for speed
            if abs(np.dot(spin, normal)) > tol:
                return False
        return True
    
    def _check_collinear(self, spins: np.ndarray, tol: float = 0.1) -> bool:
        """Check collinearity."""
        if len(spins) < 2:
            return True
        
        ref = spins[0] / (np.linalg.norm(spins[0]) + 1e-10)
        for spin in spins[:50]:  # Check subset for speed
            if abs(np.dot(spin, ref)) < 1 - tol:
                return False
        return True
    
    def _is_at_m(self, q: np.ndarray, tol: float) -> bool:
        """Check if q is at M point."""
        m_points = [np.array([0.5, 0.0]), np.array([0.0, 0.5]), np.array([0.5, 0.5])]
        return any(np.linalg.norm(q - m) < tol for m in m_points)
    
    def _is_at_k(self, q: np.ndarray, tol: float) -> bool:
        """Check if q is at K point."""
        k_points = [np.array([1/3, 1/3]), np.array([2/3, 1/3]), np.array([1/3, 2/3])]
        return any(np.linalg.norm(q - k) < tol for k in k_points)


def generate_random_normalized_params(bounds: Dict[str, Tuple[float, float]] = None,
                                       J1xy_sign: float = -1.0,
                                       J1xy_abs: float = 6.0) -> NormalizedParameters:
    """
    Generate random normalized parameters within bounds.
    
    Args:
        bounds: Dictionary of (min, max) for each parameter. Defaults to DEFAULT_BOUNDS.
        J1xy_sign: Sign of J1xy (+1 or -1)
        J1xy_abs: Absolute value of J1xy for unnormalization
        
    Returns:
        NormalizedParameters with random values
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    
    return NormalizedParameters(
        J1xy_sign=J1xy_sign,
        J1z_norm=np.random.uniform(*bounds['J1z_norm']),
        D_norm=np.random.uniform(*bounds['D_norm']),
        E_norm=np.random.uniform(*bounds['E_norm']),
        F_norm=np.random.uniform(*bounds['F_norm']),
        G_norm=np.random.uniform(*bounds['G_norm']),
        J3xy_norm=np.random.uniform(*bounds['J3xy_norm']),
        J3z_norm=np.random.uniform(*bounds['J3z_norm']),
        J1xy_abs=J1xy_abs,
    )


def generate_latin_hypercube_samples(n_samples: int,
                                      bounds: Dict[str, Tuple[float, float]] = None,
                                      J1xy_sign: float = -1.0,
                                      J1xy_abs: float = 6.0,
                                      seed: int = None) -> List[NormalizedParameters]:
    """
    Generate Latin Hypercube samples in the parameter space.
    
    Args:
        n_samples: Number of samples to generate
        bounds: Parameter bounds
        J1xy_sign: Sign of J1xy
        J1xy_abs: Absolute value of J1xy
        seed: Random seed for reproducibility
        
    Returns:
        List of NormalizedParameters
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS
    
    if seed is not None:
        np.random.seed(seed)
    
    param_names = list(bounds.keys())
    n_params = len(param_names)
    
    # Generate LHS samples
    samples = np.zeros((n_samples, n_params))
    
    for i, name in enumerate(param_names):
        lo, hi = bounds[name]
        # Create n_samples bins
        bin_edges = np.linspace(lo, hi, n_samples + 1)
        # Randomly sample one point from each bin
        perm = np.random.permutation(n_samples)
        for j in range(n_samples):
            samples[perm[j], i] = np.random.uniform(bin_edges[j], bin_edges[j + 1])
    
    # Convert to NormalizedParameters
    result = []
    for s in samples:
        params = NormalizedParameters(
            J1xy_sign=J1xy_sign,
            J1z_norm=s[param_names.index('J1z_norm')],
            D_norm=s[param_names.index('D_norm')],
            E_norm=s[param_names.index('E_norm')],
            F_norm=s[param_names.index('F_norm')],
            G_norm=s[param_names.index('G_norm')],
            J3xy_norm=s[param_names.index('J3xy_norm')],
            J3z_norm=s[param_names.index('J3z_norm')],
            J1xy_abs=J1xy_abs,
        )
        result.append(params)
    
    return result


# Known seed parameters (normalized) from the user's examples
KNOWN_SEEDS = {
    'fitting_param_2': NormalizedParameters.from_raw(
        [-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]
    ),
    'fitting_param_4': NormalizedParameters.from_raw(
        [-6.646, -2.084, 0.675, 1.33, -1.516, 0.21, 1.697, 0.039]
    ),
    'fitting_param_8': NormalizedParameters.from_raw(
        [-6.646, -2.084, 0.675, 1.25, -1.516, -0.21, 1.697, 0.039]
    ),
}


if __name__ == "__main__":
    # Test the feature extraction
    print("=" * 60)
    print("Known seed parameters (normalized by |J1xy|):")
    print("=" * 60)
    
    for name, params in KNOWN_SEEDS.items():
        print(f"\n{name}:")
        print(f"  {params}")
        print(f"  Feature vector: {params.to_feature_vector()}")
        print(f"  Raw J: {params.to_raw()}")
    
    print("\n" + "=" * 60)
    print("Latin Hypercube Samples (5 samples):")
    print("=" * 60)
    
    lhs_samples = generate_latin_hypercube_samples(5, seed=42)
    for i, s in enumerate(lhs_samples):
        print(f"\nSample {i+1}: {s}")
