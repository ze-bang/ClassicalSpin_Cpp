"""
Phase Classifier for Honeycomb BCAO Spin Configurations.

This module classifies spin configurations into distinct magnetic phases based on
Static Spin Structure Factor (SSSF) analysis to identify ordering wavevectors.

Classification scheme (following single_q_BCAO.py):
    - Gamma point (Q ≈ 0): FM or AFM (based on sublattice alignment)
    - M point (Q ≈ (0.5, 0) or equivalent): Zigzag/Stripy order
    - K point (Q ≈ (1/3, 1/3)): 120° order
    - Other Q: Incommensurate order

Extended to Multi-Q phases:
    - Double-Q with parallel Q-vectors: Beating
    - Double-Q with non-parallel Q-vectors: Meron-Antimeron
    - Triple-Q: Triple-Q (no further distinction)

Each classification includes verbose flags explaining why that phase was assigned.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class PhaseType(Enum):
    """Enumeration of possible magnetic phases."""
    FM = "Ferromagnetic"
    AFM = "Antiferromagnetic"
    ZIGZAG = "Zigzag"
    ORDER_120 = "120° Order"
    INCOMM_GAMMA_M = "Incommensurate Γ→M"
    INCOMM_GAMMA_K = "Incommensurate Γ→K"
    BEATING = "Beating"
    MERON_ANTIMERON = "Meron-Antimeron"
    TRIPLE_Q = "Triple-Q"
    INCOMMENSURATE = "Incommensurate"
    DISORDERED = "Disordered"
    TIMEOUT = "Timeout"
    LSWT_REJECTED = "LSWT Rejected"  # Failed LSWT pre-screening
    UNKNOWN = "Unknown"


@dataclass
class ClassificationFlags:
    """Container for all classification flags and their values."""
    # Q-vector information (primary)
    q1: Tuple[float, float] = (0.0, 0.0)  # (Q1, Q2) reduced coordinates
    q1_intensity: float = 0.0
    q1_type: str = "unknown"  # "gamma", "m", "k", "incomm_gm", "incomm_gk", "incomm"
    
    # Q-vector information (secondary, for double-Q)
    q2: Optional[Tuple[float, float]] = None
    q2_intensity: float = 0.0
    q2_type: str = "none"
    
    # Q-vector information (tertiary, for triple-Q)
    q3: Optional[Tuple[float, float]] = None
    q3_intensity: float = 0.0
    q3_type: str = "none"
    
    # Multi-Q analysis
    num_peaks: int = 0
    is_double_q: bool = False
    is_triple_q: bool = False
    q_vectors_parallel: bool = False
    intensity_ratio: float = 0.0  # q2/q1 intensity ratio
    intensity_ratio_3: float = 0.0  # q3/q1 intensity ratio
    
    # All detected peaks
    all_peaks: List[Tuple[float, float, float]] = field(default_factory=list)
    
    # Magnetization (for FM/AFM distinction at Gamma)
    total_mag: float = 0.0
    staggered_mag: float = 0.0
    sublattice_parallel: bool = False
    sublattice_antiparallel: bool = False
    
    # Energy
    energy_per_site: float = 0.0
    
    @staticmethod
    def _to_native(val):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(val, (np.bool_, np.integer)):
            return int(val) if isinstance(val, np.integer) else bool(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, dict):
            return {k: ClassificationFlags._to_native(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [ClassificationFlags._to_native(v) for v in val]
        return val
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert flags to dictionary for logging."""
        raw = {
            "q1": {
                "position": self.q1,
                "intensity": self.q1_intensity,
                "type": self.q1_type,
            },
            "q2": {
                "position": self.q2,
                "intensity": self.q2_intensity,
                "type": self.q2_type,
            },
            "multi_q": {
                "num_peaks": self.num_peaks,
                "is_double_q": self.is_double_q,
                "parallel": self.q_vectors_parallel,
                "intensity_ratio": self.intensity_ratio,
            },
            "all_peaks": self.all_peaks,
            "magnetization": {
                "total": self.total_mag,
                "staggered": self.staggered_mag,
                "sublattice_parallel": self.sublattice_parallel,
                "sublattice_antiparallel": self.sublattice_antiparallel,
            },
            "energy": self.energy_per_site,
        }
        return self._to_native(raw)


@dataclass
class ClassificationResult:
    """Result of phase classification."""
    phase: PhaseType
    confidence: float
    flags: ClassificationFlags
    decision_path: List[str]  # List of decision steps taken
    
    def print_verbose(self):
        """Print verbose classification results."""
        print("\n" + "=" * 70)
        print("PHASE CLASSIFICATION RESULT")
        print("=" * 70)
        print(f"  Phase: {self.phase.value}")
        print(f"  Confidence: {self.confidence:.2%}")
        print("\n  Decision Path:")
        for i, step in enumerate(self.decision_path, 1):
            print(f"    {i}. {step}")
        print("\n  Key Flags:")
        print(f"    Q1: ({self.flags.q1[0]:.4f}, {self.flags.q1[1]:.4f}) [{self.flags.q1_type}], I={self.flags.q1_intensity:.4f}")
        if self.flags.q2 is not None:
            print(f"    Q2: ({self.flags.q2[0]:.4f}, {self.flags.q2[1]:.4f}) [{self.flags.q2_type}], I={self.flags.q2_intensity:.4f}")
            print(f"    Double-Q: {self.flags.is_double_q}, Parallel: {self.flags.q_vectors_parallel}")
        if self.flags.q1_type == "gamma":
            print(f"    Magnetization: |M_tot|={self.flags.total_mag:.4f}, |M_stag|={self.flags.staggered_mag:.4f}")
        print("=" * 70)


class PhaseClassifier:
    """
    Classifier for honeycomb lattice spin configurations using SSSF analysis.
    
    Classification follows the scheme in single_q_BCAO.py:
    - Gamma point → FM or AFM
    - M point → Zigzag
    - K point → 120° order
    - Otherwise → Incommensurate
    
    Extended for multi-Q phases:
    - Double-Q parallel → Beating
    - Double-Q non-parallel → Meron-Antimeron
    - Triple-Q → Triple-Q
    """
    
    # Tolerance for Q-vector identification
    Q_TOL = 0.05
    
    # Threshold for secondary peak (relative to primary)
    DOUBLE_Q_THRESHOLD = 0.3
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the classifier.
        
        Args:
            verbose: Whether to print verbose output during classification.
        """
        self.verbose = verbose
        
        # Reciprocal lattice vectors (2D honeycomb)
        self.b1 = 2 * np.pi * np.array([1.0, -1.0 / np.sqrt(3.0)])
        self.b2 = 2 * np.pi * np.array([0.0, 2.0 / np.sqrt(3.0)])
    
    def classify(self, spins: np.ndarray, positions: np.ndarray, 
                 energy: float = None, L: int = None) -> ClassificationResult:
        """
        Classify the spin configuration based on SSSF analysis.
        
        Args:
            spins: (N, 3) array of spin vectors
            positions: (N, 2) or (N, 3) array of site positions
            energy: Energy per site (optional)
            L: Lattice linear size (optional, inferred if not provided)
            
        Returns:
            ClassificationResult with phase, confidence, flags, and decision path.
        """
        flags = ClassificationFlags()
        decision_path = []
        
        if L is None:
            L = int(np.sqrt(len(spins) / 2))
        
        # Ensure positions are 2D for honeycomb
        if positions.shape[1] == 3:
            positions = positions[:, :2]
        
        # Step 1: Compute SSSF and find peaks
        peaks = self._compute_sssf_peaks(spins, positions, L)
        flags.all_peaks = [(float(p[0]), float(p[1]), float(p[2])) for p in peaks]
        flags.num_peaks = len(peaks)
        
        if len(peaks) == 0:
            decision_path.append("No SSSF peaks found → Disordered")
            flags.q1_type = "none"
            return ClassificationResult(
                phase=PhaseType.DISORDERED,
                confidence=0.7,
                flags=flags,
                decision_path=decision_path
            )
        
        # Step 2: Classify primary Q-vector
        q1 = np.array([peaks[0][0], peaks[0][1]])
        flags.q1 = (float(q1[0]), float(q1[1]))
        flags.q1_intensity = float(peaks[0][2])
        flags.q1_type = self._classify_q_point(q1)
        
        decision_path.append(f"Primary Q: ({q1[0]:.4f}, {q1[1]:.4f}) → {flags.q1_type}")
        
        # Step 3: Check for double-Q/triple-Q (secondary and tertiary peaks)
        if len(peaks) >= 2:
            q2 = np.array([peaks[1][0], peaks[1][1]])
            flags.q2 = (float(q2[0]), float(q2[1]))
            flags.q2_intensity = float(peaks[1][2])
            flags.q2_type = self._classify_q_point(q2)
            flags.intensity_ratio = flags.q2_intensity / flags.q1_intensity
            
            # Check if this is a double-Q state
            if flags.intensity_ratio > self.DOUBLE_Q_THRESHOLD:
                flags.is_double_q = True
                decision_path.append(f"Secondary Q: ({q2[0]:.4f}, {q2[1]:.4f}) → {flags.q2_type}, intensity ratio={flags.intensity_ratio:.3f} → Double-Q")
                
                # Check if Q-vectors are parallel
                q1_norm = np.linalg.norm(q1)
                q2_norm = np.linalg.norm(q2)
                if q1_norm > 1e-6 and q2_norm > 1e-6:
                    cos_angle = np.dot(q1, q2) / (q1_norm * q2_norm)
                    # Parallel if angle is close to 0° or 180° (cos ≈ ±1)
                    flags.q_vectors_parallel = abs(abs(cos_angle) - 1.0) < 0.1
                    angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                    decision_path.append(f"Q-vectors angle: {angle_deg:.1f}° → {'Parallel' if flags.q_vectors_parallel else 'Non-parallel'}")
                
                # Check for triple-Q
                if len(peaks) >= 3:
                    q3 = np.array([peaks[2][0], peaks[2][1]])
                    flags.q3 = (float(q3[0]), float(q3[1]))
                    flags.q3_intensity = float(peaks[2][2])
                    flags.q3_type = self._classify_q_point(q3)
                    flags.intensity_ratio_3 = flags.q3_intensity / flags.q1_intensity
                    
                    if flags.intensity_ratio_3 > self.DOUBLE_Q_THRESHOLD:
                        flags.is_triple_q = True
                        decision_path.append(f"Tertiary Q: ({q3[0]:.4f}, {q3[1]:.4f}) → {flags.q3_type}, intensity ratio={flags.intensity_ratio_3:.3f} → Triple-Q")
        
        # Step 4: For Gamma point, compute magnetization to distinguish FM/AFM
        if flags.q1_type == "gamma":
            self._compute_magnetization(spins, flags, decision_path)
        
        # Step 5: Store energy
        if energy is not None:
            flags.energy_per_site = float(energy)
        
        # Step 6: Make final classification
        phase, confidence = self._classify_from_q_types(flags, decision_path)
        
        result = ClassificationResult(
            phase=phase,
            confidence=confidence,
            flags=flags,
            decision_path=decision_path
        )
        
        if self.verbose:
            result.print_verbose()
        
        return result
    
    def _compute_sssf_peaks(self, spins: np.ndarray, positions: np.ndarray, 
                            L: int) -> List[Tuple[float, float, float]]:
        """
        Compute Static Spin Structure Factor and find peaks.
        
        Following reader_honeycomb.py SSSF_q approach.
        
        Args:
            spins: (N, 3) spin configuration
            positions: (N, 2) site positions
            L: Lattice size
            
        Returns:
            List of (Q1, Q2, intensity) for peaks in reduced coordinates
        """
        N = len(spins)
        
        # Create momentum grid in reduced coordinates (H, K)
        # Cover first Brillouin zone: H in [0, 1], K in [0, 1]
        # Include 1.0 to properly capture M and K points
        n_grid = max(100, 2 * L)
        h_vals = np.linspace(0.0, 1.0, n_grid)  # Extended to full BZ
        k_vals = np.linspace(0.0, 1.0, n_grid)
        
        # Build k-points in Cartesian coordinates
        # k = H * b1 + K * b2
        H_grid, K_grid = np.meshgrid(h_vals, k_vals)
        
        # Reciprocal basis (3D for compatibility)
        b1_3d = np.array([self.b1[0], self.b1[1], 0.0])
        b2_3d = np.array([self.b2[0], self.b2[1], 0.0])
        
        # Positions in 3D
        if positions.shape[1] == 2:
            P_3d = np.column_stack([positions, np.zeros(N)])
        else:
            P_3d = positions
        
        # Compute S(q) on grid
        S_q = np.zeros((n_grid, n_grid))
        
        for i, h in enumerate(h_vals):
            for j, k in enumerate(k_vals):
                q = h * b1_3d + k * b2_3d
                
                # S(q) = |Σ_r S_r exp(i q·r)|² / N
                phase_factors = np.exp(1j * np.dot(P_3d, q))
                S_q_vec = np.sum(spins * phase_factors[:, np.newaxis], axis=0) / np.sqrt(N)
                S_q[j, i] = np.sum(np.abs(S_q_vec)**2)
        
        # Find peaks (local maxima above threshold)
        peaks = []
        threshold = 0.1 * np.max(S_q) if np.max(S_q) > 0 else 0.0
        
        # Include edge pixels by using modular indexing
        for i in range(n_grid):
            for j in range(n_grid):
                val = S_q[j, i]
                if val > threshold:
                    # Check if local maximum using periodic boundary in BZ
                    # For interior points
                    im1 = (i - 1) % n_grid
                    ip1 = (i + 1) % n_grid
                    jm1 = (j - 1) % n_grid
                    jp1 = (j + 1) % n_grid
                    
                    is_max = (val > S_q[j, im1] and val > S_q[j, ip1] and
                              val > S_q[jm1, i] and val > S_q[jp1, i] and
                              val > S_q[jm1, im1] and val > S_q[jp1, ip1] and
                              val > S_q[jm1, ip1] and val > S_q[jp1, im1])
                    
                    if is_max:
                        h, k = h_vals[i], k_vals[j]
                        # Fold back to first BZ quadrant for comparison
                        h_folded = h if h <= 0.5 else 1.0 - h
                        k_folded = k if k <= 0.5 else 1.0 - k
                        peaks.append((h_folded, k_folded, val))
        
        # Also check for peak at Gamma
        if n_grid > 2:
            idx_h0 = np.argmin(np.abs(h_vals))
            idx_k0 = np.argmin(np.abs(k_vals))
            gamma_val = S_q[idx_k0, idx_h0]
            if gamma_val > threshold:
                peaks.append((0.0, 0.0, gamma_val))
        
        # Remove duplicates (peaks at equivalent positions)
        unique_peaks = []
        for p in peaks:
            is_dup = False
            for up in unique_peaks:
                if (abs(p[0] - up[0]) < 0.03 and abs(p[1] - up[1]) < 0.03):
                    # Keep higher intensity
                    if p[2] > up[2]:
                        unique_peaks.remove(up)
                        unique_peaks.append(p)
                    is_dup = True
                    break
            if not is_dup:
                unique_peaks.append(p)
        
        # Sort by intensity (descending)
        unique_peaks = sorted(unique_peaks, key=lambda x: -x[2])
        
        return unique_peaks[:5]  # Return top 5 peaks
    
    def _classify_q_point(self, q: np.ndarray) -> str:
        """
        Classify Q-vector position following single_q_BCAO.py scheme.
        
        Uses mod 1 to map Q to first Brillouin zone, then checks high-symmetry points.
        
        Args:
            q: (Q1, Q2) in reduced coordinates
            
        Returns:
            Type string: "gamma", "m", "k", "commensurate", "incomm_gm", "incomm_gk", "incomm"
        """
        tol = self.Q_TOL
        
        # Map to first BZ using mod 1, keeping in [0, 1)
        Q1 = q[0] % 1.0
        Q2 = q[1] % 1.0
        
        # Handle numerical edge cases near 1.0
        if Q1 > 1.0 - tol:
            Q1 = 0.0
        if Q2 > 1.0 - tol:
            Q2 = 0.0
        
        # Gamma point: Q ≈ (0, 0)
        if abs(Q1) < tol and abs(Q2) < tol:
            return "gamma"
        
        # M points: (0.5, 0), (0, 0.5), (0.5, 0.5)
        # Check if either component is 0.5 and other is 0 or 0.5
        is_half_Q1 = abs(Q1 - 0.5) < tol
        is_half_Q2 = abs(Q2 - 0.5) < tol
        is_zero_Q1 = abs(Q1) < tol
        is_zero_Q2 = abs(Q2) < tol
        
        if (is_half_Q1 and is_zero_Q2) or (is_zero_Q1 and is_half_Q2) or (is_half_Q1 and is_half_Q2):
            return "m"
        
        # K points: (1/3, 1/3), (2/3, 1/3), (1/3, 2/3), (2/3, 2/3)
        # and boundary equivalents (0, 1/3), (0, 2/3), (1/3, 0), (2/3, 0)
        # General rule: Q1 or Q2 is n/3 (n=1,2) and the other is 0 or n/3
        def is_third(x):
            return abs(x - 1/3) < tol or abs(x - 2/3) < tol
        
        if is_third(Q1) and is_third(Q2):
            return "k"
        if (is_third(Q1) and is_zero_Q2) or (is_zero_Q1 and is_third(Q2)):
            return "k"
        
        # Check for commensurate (rational fraction) Q-vectors
        # A Q is commensurate if it can be expressed as p/q where q is small integer
        if self._is_commensurate(Q1, tol) and self._is_commensurate(Q2, tol):
            return "commensurate"
        
        # Incommensurate - check direction
        # Along high-symmetry lines: Γ-M is along (1, 0), (0, 1), or (1, 1)
        if abs(Q2) < tol or abs(Q1) < tol:
            return "incomm_gm"
        if abs(Q1 - Q2) < tol:
            return "incomm_gm"
        
        # Γ-K direction: ratio Q1:Q2 = 2:1 or 1:2
        ratio = Q1 / Q2 if abs(Q2) > tol else float('inf')
        if abs(ratio - 2.0) < 0.2 or abs(ratio - 0.5) < 0.1:
            return "incomm_gk"
        
        return "incomm"
    
    def _is_commensurate(self, q_val: float, tol: float = 0.08) -> bool:
        """
        Check if a Q component is commensurate (rational fraction with small denominator).
        
        Commensurate means Q = p/q where q is a small integer (2, 3, 4, 6, etc.)
        """
        # Common commensurate fractions (denominators up to 6)
        commensurate_values = [
            0.0, 1.0,           # 0/1, 1/1
            0.5,                # 1/2
            1/3, 2/3,           # 1/3, 2/3
            0.25, 0.75,         # 1/4, 3/4
            0.2, 0.4, 0.6, 0.8, # 1/5, 2/5, 3/5, 4/5
            1/6, 5/6,           # 1/6, 5/6
        ]
        for cv in commensurate_values:
            if abs(q_val - cv) < tol:
                return True
        return False
    
    def _compute_magnetization(self, spins: np.ndarray, 
                               flags: ClassificationFlags,
                               decision_path: List[str]):
        """Compute magnetization to distinguish FM from AFM at Gamma point."""
        # Total magnetization
        total_mag = np.mean(spins, axis=0)
        flags.total_mag = float(np.linalg.norm(total_mag))
        
        # Sublattice magnetizations (A: even indices, B: odd indices)
        spins_A = spins[::2]
        spins_B = spins[1::2]
        mag_A = np.mean(spins_A, axis=0)
        mag_B = np.mean(spins_B, axis=0)
        
        # Staggered magnetization
        staggered = mag_A - mag_B
        flags.staggered_mag = float(np.linalg.norm(staggered))
        
        # Check sublattice alignment
        if np.linalg.norm(mag_A) > 0.1 and np.linalg.norm(mag_B) > 0.1:
            dot = np.dot(mag_A, mag_B) / (np.linalg.norm(mag_A) * np.linalg.norm(mag_B))
            flags.sublattice_parallel = dot > 0.9
            flags.sublattice_antiparallel = dot < -0.9
        
        decision_path.append(
            f"Magnetization: |M_tot|={flags.total_mag:.4f}, |M_stag|={flags.staggered_mag:.4f}, "
            f"parallel={flags.sublattice_parallel}, antiparallel={flags.sublattice_antiparallel}"
        )
    
    def _classify_from_q_types(self, flags: ClassificationFlags,
                               decision_path: List[str]) -> Tuple[PhaseType, float]:
        """
        Final classification based on Q-vector types.
        
        Following single_q_BCAO.py classify_phase scheme, extended for double-Q.
        """
        q1_type = flags.q1_type
        is_double_q = flags.is_double_q
        q2_type = flags.q2_type if is_double_q else None
        
        # === Triple-Q phases (check first since they're also double-Q) ===
        if flags.is_triple_q:
            decision_path.append(f"DECISION: Triple-Q detected ({q1_type}, {q2_type}, {flags.q3_type}) → Triple-Q")
            return PhaseType.TRIPLE_Q, 0.85
        
        # === Double-Q phases ===
        if flags.is_double_q:
            # Check if Q-vectors are parallel
            if flags.q_vectors_parallel:
                decision_path.append(f"DECISION: Double-Q with parallel Q-vectors ({q1_type}, {q2_type}) → Beating")
                return PhaseType.BEATING, 0.85
            else:
                decision_path.append(f"DECISION: Double-Q with non-parallel Q-vectors ({q1_type}, {q2_type}) → Meron-Antimeron")
                return PhaseType.MERON_ANTIMERON, 0.85
        
        # === Single-Q phases ===
        # Gamma point → FM or AFM
        if q1_type == "gamma":
            if flags.sublattice_parallel or flags.total_mag > 0.5:
                decision_path.append("DECISION: Q at Γ, sublattices parallel → FM")
                return PhaseType.FM, 0.9
            elif flags.sublattice_antiparallel or flags.staggered_mag > 0.5:
                decision_path.append("DECISION: Q at Γ, sublattices antiparallel → AFM")
                return PhaseType.AFM, 0.9
            else:
                # Default to FM if unclear
                decision_path.append("DECISION: Q at Γ, magnetization unclear → FM (default)")
                return PhaseType.FM, 0.6
        
        # M point → Zigzag
        if q1_type == "m":
            decision_path.append("DECISION: Q at M point → Zigzag")
            return PhaseType.ZIGZAG, 0.9
        
        # K point → 120° order
        if q1_type == "k":
            decision_path.append("DECISION: Q at K point → 120° Order")
            return PhaseType.ORDER_120, 0.9
        
        # Commensurate but not at special point - treat as incommensurate 
        # (this can happen for commensurate fractions like 1/4, 1/5, etc.)
        if q1_type == "commensurate":
            # Check if it's closer to Gamma-M or Gamma-K direction
            q1 = np.array(flags.q1)
            # Check direction
            if abs(q1[1]) < 0.1 or abs(q1[0]) < 0.1 or abs(q1[0] - q1[1]) < 0.1:
                decision_path.append(f"DECISION: Commensurate Q along Γ→M direction → Incommensurate Γ→M")
                return PhaseType.INCOMM_GAMMA_M, 0.8
            else:
                decision_path.append(f"DECISION: Commensurate Q ({q1[0]:.3f}, {q1[1]:.3f}) → Incommensurate")
                return PhaseType.INCOMMENSURATE, 0.8
        
        # Incommensurate
        if q1_type == "incomm_gm":
            decision_path.append("DECISION: Incommensurate Q along Γ→M")
            return PhaseType.INCOMM_GAMMA_M, 0.85
        
        if q1_type == "incomm_gk":
            decision_path.append("DECISION: Incommensurate Q along Γ→K")
            return PhaseType.INCOMM_GAMMA_K, 0.85
        
        if q1_type == "incomm":
            decision_path.append("DECISION: General incommensurate order")
            return PhaseType.INCOMMENSURATE, 0.8
        
        # Fallback for unhandled cases
        if flags.num_peaks == 0 or q1_type == "none":
            decision_path.append("DECISION: No clear peaks → Disordered")
            return PhaseType.DISORDERED, 0.7
        
        # Log the unhandled q1_type for debugging
        decision_path.append(f"DECISION: Unhandled q1_type='{q1_type}' → Incommensurate (fallback)")
        return PhaseType.INCOMMENSURATE, 0.6


def classify_spin_config(spins: np.ndarray, positions: np.ndarray,
                         energy: float = None, L: int = None,
                         verbose: bool = True) -> ClassificationResult:
    """
    Convenience function to classify a spin configuration.
    
    Args:
        spins: (N, 3) array of spin vectors
        positions: (N, 2) or (N, 3) array of site positions
        energy: Energy per site (optional)
        L: Lattice size (optional, inferred)
        verbose: Print verbose output
        
    Returns:
        ClassificationResult
    """
    classifier = PhaseClassifier(verbose=verbose)
    return classifier.classify(spins, positions, energy=energy, L=L)


if __name__ == "__main__":
    # Test the classifier
    print("=" * 70)
    print("PHASE CLASSIFIER TEST")
    print("=" * 70)
    
    try:
        # Try to import lattice creation function
        import sys
        sys.path.insert(0, '.')
        from luttinger_tisza import create_honeycomb_lattice
        
        L = 12
        positions, NN, NN_bonds, NNN, NNNN, a1, a2 = create_honeycomb_lattice(L)
        N = len(positions)
        
        classifier = PhaseClassifier(verbose=True)
        
        # Test 1: Ferromagnetic
        print("\n--- TEST 1: Ferromagnetic ---")
        spins_fm = np.tile([0, 0, 1], (N, 1)).astype(float)
        result_fm = classifier.classify(spins_fm, positions, L=L)
        print(f"Expected: FM, Got: {result_fm.phase.value}")
        
        # Test 2: Antiferromagnetic
        print("\n--- TEST 2: Antiferromagnetic ---")
        spins_afm = np.zeros((N, 3))
        for i in range(N):
            spins_afm[i] = [0, 0, 1] if i % 2 == 0 else [0, 0, -1]
        result_afm = classifier.classify(spins_afm, positions, L=L)
        print(f"Expected: AFM, Got: {result_afm.phase.value}")
        
        # Test 3: Zigzag (alternate along one direction)
        print("\n--- TEST 3: Zigzag-like ---")
        spins_zz = np.zeros((N, 3))
        for i in range(N):
            row = int(positions[i, 1] / (np.sqrt(3)/2))
            spins_zz[i] = [0, 0, 1] if row % 2 == 0 else [0, 0, -1]
        result_zz = classifier.classify(spins_zz, positions, L=L)
        print(f"Expected: Zigzag, Got: {result_zz.phase.value}")
        
        print("\n" + "=" * 70)
        print("Tests completed!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"Could not import lattice module: {e}")
        print("Run from the classical_solvers directory or adjust import path.")
