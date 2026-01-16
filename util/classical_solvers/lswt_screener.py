"""
LSWT Pre-Screening for Active Learning.

This module provides a Linear Spin Wave Theory (LSWT) screening step
that validates proposed parameters against experimental spin wave
dispersion data before running expensive Monte Carlo simulations.

The workflow:
1. Active learning proposes parameter sets
2. LSWT screener quickly evaluates fit quality (R² score)
3. Only parameters that pass the threshold proceed to Monte Carlo

Author: Integrated from workflow/LSWT_fit
Date: 2025-01
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, List
from dataclasses import dataclass
import sys
import os

# Optional pandas import for data loading
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Add workflow path for hamiltonian imports
WORKFLOW_LSWT_PATH = Path(__file__).parent.parent.parent / "workflow" / "LSWT_fit"
sys.path.insert(0, str(WORKFLOW_LSWT_PATH))

try:
    import hamiltonian_init_new as ham
    HAS_LSWT_HAM = True
except ImportError:
    HAS_LSWT_HAM = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LSWTConfig:
    """Configuration for LSWT screening."""
    
    # Physical constants
    SPIN: float = 0.5
    MAGNETIC_FIELD: float = 3.0  # Tesla
    
    # g-factors and magnetic coupling
    G_B: float = 5.0
    G_A: float = 5.0
    G_Z: float = 2.7
    MU_B: float = 0.05788381
    
    # Second-neighbor coupling (typically fixed at 0)
    J2: float = 0.0
    J2Z: float = 0.0
    
    # Crystal structure vectors (honeycomb)
    DELTA_1: np.ndarray = None
    DELTA_2: np.ndarray = None
    DELTA_3: np.ndarray = None
    PHI: np.ndarray = None
    
    # Screening threshold (R² > threshold passes)
    R2_THRESHOLD: float = 0.7
    R2_LOWER_THRESHOLD: float = 0.75  # Lower band R² threshold
    
    # Direction for spin wave calculation
    DIRECTION: str = 'b'
    
    def __post_init__(self):
        """Initialize crystal structure vectors."""
        if self.DELTA_1 is None:
            self.DELTA_1 = np.array([
                [0, 1], 
                [-np.sqrt(3)/2, -1/2], 
                [np.sqrt(3)/2, -1/2]
            ])
        if self.DELTA_2 is None:
            self.DELTA_2 = np.array([
                [np.sqrt(3), 0], 
                [np.sqrt(3)/2, 1.5], 
                [-np.sqrt(3)/2, 1.5],
                [-np.sqrt(3), 0], 
                [-np.sqrt(3)/2, -1.5], 
                [np.sqrt(3)/2, -1.5]
            ])
        if self.DELTA_3 is None:
            self.DELTA_3 = np.array([
                [0, -2], 
                [np.sqrt(3), 1], 
                [-np.sqrt(3), 1]
            ])
        if self.PHI is None:
            self.PHI = np.array([0, 2*np.pi/3, -2*np.pi/3])


# =============================================================================
# DATA LOADING
# =============================================================================

class ExperimentalDataLoader:
    """Loads and processes experimental spin wave dispersion data."""
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing CSV files. Defaults to workflow/LSWT_fit.
        """
        if data_dir is None:
            data_dir = WORKFLOW_LSWT_PATH
        self.data_dir = Path(data_dir)
        self.data_lower = None
        self.data_upper = None
        self._loaded = False
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process experimental data.
        
        Returns:
            (data_lower, data_upper): Arrays of shape (3, N) with [qx, qy, energy]
        """
        if self._loaded:
            return self.data_lower, self.data_upper
        
        if not HAS_PANDAS:
            print("Warning: pandas not available. LSWT screening will be skipped.")
            self.data_lower = np.zeros((3, 1))
            self.data_upper = np.zeros((3, 1))
            self._loaded = True
            return self.data_lower, self.data_upper
        
        try:
            # Load CSV files
            df1 = pd.read_csv(self.data_dir / "plot-data big1.csv")
            df2 = pd.read_csv(self.data_dir / "plot-data big2.csv")
            df3 = pd.read_csv(self.data_dir / "plot-data big3.csv")
            df4 = pd.read_csv(self.data_dir / "plot-data big3 up.csv")
            
            # Extract x, y columns
            data1 = df1[["x", " y"]].values
            data2 = df2[["x", " y"]].values
            data3 = df3[["x", " y"]].values
            data4 = df4[["x", " y"]].values
            
            # Stack all data
            data_all = np.vstack((data1, data2, data3, data4))
            
            # Manual corrections (from original code)
            data_all[48, 0] -= 0.01
            data_all[58, 0] -= 0.01
            
            # Transform to momentum space
            self.data_lower, self.data_upper = self._transform_to_momentum_space(data_all)
            self._loaded = True
            
        except FileNotFoundError as e:
            print(f"Warning: Could not load experimental data: {e}")
            print(f"LSWT screening will be skipped.")
            # Create empty placeholder data
            self.data_lower = np.zeros((3, 1))
            self.data_upper = np.zeros((3, 1))
            self._loaded = True
        
        return self.data_lower, self.data_upper
    
    def _transform_to_momentum_space(self, data_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform linearized path data to (qx, qy) momentum coordinates."""
        pi = np.pi
        sq = np.sqrt(3)
        
        # Total path length in BZ
        L = 4*pi/3 + 8*pi/(3*sq) + 2*pi/3 + 2*pi/(3*sq)
        
        # Normalized positions along path
        r1 = (2*pi/3) / L
        r2 = (4*pi/3) / L
        r3 = (4*pi/3 + 4*pi/(3*sq)) / L
        r4 = (4*pi/3 + 8*pi/(3*sq)) / L
        r5 = (6*pi/3 + 8*pi/(3*sq)) / L
        r6 = (6*pi/3 + 8*pi/(3*sq) + 2*pi/(3*sq)) / L
        
        # Separate into path segments
        segments = self._separate_path_segments(data_all, [r1, r2, r3, r4, r5, r6])
        
        # Transform each segment
        transformed = self._transform_segments(segments, pi, sq, [r1, r2, r3, r4, r5, r6])
        
        # Separate into lower and upper bands
        data_lower = np.hstack([transformed[i] for i in [0, 1, 2, 3, 4, 6]])
        data_upper = np.hstack([transformed[i] for i in [5, 7]])
        
        return data_lower, data_upper
    
    def _separate_path_segments(self, data: np.ndarray, 
                                ratios: List[float]) -> List[np.ndarray]:
        """Separate data into 8 path segments based on position ratios."""
        r1, r2, r3, r4, r5, r6 = ratios
        
        segments = [[] for _ in range(8)]
        
        for i in range(len(data)):
            x, y = data[i, 0], data[i, 1]
            
            if x < r1:
                segments[0].append([x, y])
            elif x < r2:
                segments[1].append([x, y])
            elif x < r3:
                segments[2].append([x, y])
            elif x < r4:
                segments[3].append([x, y])
            elif x < r5 and y < 8:
                segments[4].append([x, y])
            elif x < r5 and y > 8:
                segments[5].append([x, y])
            elif x < r6 and y < 9:
                segments[6].append([x, y])
            elif x < r6 and y > 9:
                segments[7].append([x, y])
        
        return [np.array(seg) if seg else np.zeros((0, 2)) for seg in segments]
    
    def _transform_segments(self, segments: List[np.ndarray], 
                           pi: float, sq: float,
                           ratios: List[float]) -> List[np.ndarray]:
        """Transform each segment to momentum coordinates."""
        r1, r2, r3, r4, r5, r6 = ratios
        transformed = []
        
        for idx, seg in enumerate(segments):
            if len(seg) == 0:
                transformed.append(np.zeros((3, 0)))
                continue
            
            data_seg = np.zeros((3, len(seg)))
            data_seg[2] = seg[:, 1]  # Energy values
            
            if idx == 0:  # Segment 1: Γ → M
                data_seg[1] = 2*pi/3 * seg[:, 0] / r1
                
            elif idx == 1:  # Segment 2: M → K
                x_normalized = 2*pi/3 * (seg[:, 0] - r1) / (r2 - r1) - 2*pi/3
                data_seg[0] = x_normalized * 0.5 * sq
                data_seg[1] = -x_normalized * 0.5
                
            elif idx == 2:  # Segment 3: K → Γ
                x_normalized = 4*pi/(3*sq) * (seg[:, 0] - r2) / (r3 - r2)
                data_seg[0] = -x_normalized * 0.5
                data_seg[1] = x_normalized * 0.5 * sq
                
            elif idx == 3:  # Segment 4: Γ → M'
                data_seg[0] = -(4*pi/(3*sq) * (seg[:, 0] - r3) / (r4 - r3) - 4*pi/(3*sq))
                
            elif idx == 4:  # Segment 5: M' → Γ (lower)
                data_seg[1] = 2*pi/3 * (seg[:, 0] - r4) / (r5 - r4)
                data_seg[1] = data_seg[1][::-1] + 2*pi/3
                
            elif idx == 5:  # Segment 6: M' → Γ (upper)
                data_seg[1] = 2*pi/3 * (seg[:, 0] - r4) / (r5 - r4)
                data_seg[1] = data_seg[1][::-1] + 2*pi/3
                
            elif idx == 6:  # Segment 7: Γ → K (lower)
                data_seg[0] = -2*pi/(3*sq) * (seg[:, 0] - r5) / (r6 - r5)
                data_seg[1] = np.ones(len(seg)) * 2*pi/3
                
            elif idx == 7:  # Segment 8: Γ → K (upper)
                data_seg[0] = -2*pi/(3*sq) * (seg[:, 0] - r5) / (r6 - r5)
                data_seg[1] = np.ones(len(seg)) * 2*pi/3
            
            transformed.append(data_seg)
        
        return transformed


# =============================================================================
# LSWT SCREENER
# =============================================================================

@dataclass
class LSWTScreeningResult:
    """Result of LSWT screening for a parameter set."""
    
    # Whether parameters pass the screening
    passed: bool
    
    # R² values
    r2_total: float
    r2_lower: float
    
    # Reason for pass/fail
    reason: str
    
    # Flag for numerical issues
    numerical_issue: bool = False
    
    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"LSWTResult({status}, R²={self.r2_total:.3f}, R²_lower={self.r2_lower:.3f}, {self.reason})"


class LSWTScreener:
    """
    Pre-screens parameters using Linear Spin Wave Theory.
    
    Computes the theoretical spin wave dispersion and compares to
    experimental data to get an R² goodness-of-fit metric.
    """
    
    def __init__(self, config: LSWTConfig = None, verbose: bool = False):
        """
        Initialize the LSWT screener.
        
        Args:
            config: LSWT configuration. Uses defaults if None.
            verbose: Print verbose output.
        """
        self.config = config or LSWTConfig()
        self.verbose = verbose
        
        # Load experimental data
        self.data_loader = ExperimentalDataLoader()
        self.data_lower, self.data_upper = self.data_loader.load_data()
        
        # Base parameter dictionary for Hamiltonian
        self.base_params = self._create_base_params()
        
        # Check if LSWT Hamiltonian is available
        self.available = HAS_LSWT_HAM
        if not self.available:
            print("Warning: LSWT Hamiltonian module not available. Screening disabled.")
    
    def _create_base_params(self) -> Dict:
        """Create base parameter dictionary for Hamiltonian."""
        return {
            'g_b': self.config.G_B,
            'g_a': self.config.G_A,
            'g_z': self.config.G_Z,
            'mu_B': self.config.MU_B,
            'S': self.config.SPIN,
            'J2': self.config.J2,
            'J2z': self.config.J2Z,
            'delta_1': self.config.DELTA_1,
            'delta_2': self.config.DELTA_2,
            'delta_3': self.config.DELTA_3,
            'phi': self.config.PHI,
        }
    
    def _log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(msg)
    
    def screen(self, params) -> LSWTScreeningResult:
        """
        Screen a parameter set using LSWT.
        
        Args:
            params: NormalizedParameters or raw J array [J1xy, J1z, D, E, F, G, J3xy, J3z]
        
        Returns:
            LSWTScreeningResult with pass/fail and R² scores.
        """
        if not self.available:
            # LSWT not available - always pass
            return LSWTScreeningResult(
                passed=True,
                r2_total=0.0,
                r2_lower=0.0,
                reason="LSWT module not available, skipping screening"
            )
        
        # Convert to raw parameters if needed
        if hasattr(params, 'to_raw'):
            J = params.to_raw()
        else:
            J = list(params)
        
        # Extract individual parameters
        J1xy, J1z, D, E, F, G, J3xy, J3z = J
        
        # Build full parameter dictionary for Hamiltonian
        parameters = self.base_params.copy()
        parameters.update({
            'J1': J1xy,
            'J1z': J1z,
            'J3': J3xy,
            'J3z': J3z,
            'D': D,
            'E': E,
            'F': F,
            'G': G,
        })
        
        try:
            # Compute theoretical dispersion
            energies_lower, energies_upper = self._compute_dispersion(
                parameters, 
                self.config.MAGNETIC_FIELD,
                self.config.DIRECTION
            )
            
            # Check for NaN or complex values
            if np.any(np.isnan(energies_lower)) or np.any(np.isnan(energies_upper)):
                return LSWTScreeningResult(
                    passed=False,
                    r2_total=-1.0,
                    r2_lower=-1.0,
                    reason="NaN in computed dispersion",
                    numerical_issue=True
                )
            
            # Compute R² goodness-of-fit
            r2_total, r2_lower = self._compute_r_squared(energies_lower, energies_upper)
            
            # Apply threshold
            passed = (r2_total >= self.config.R2_THRESHOLD and 
                     r2_lower >= self.config.R2_LOWER_THRESHOLD)
            
            if passed:
                reason = f"Good fit: R²={r2_total:.3f}, R²_lower={r2_lower:.3f}"
            else:
                reasons = []
                if r2_total < self.config.R2_THRESHOLD:
                    reasons.append(f"R²_total={r2_total:.3f} < {self.config.R2_THRESHOLD}")
                if r2_lower < self.config.R2_LOWER_THRESHOLD:
                    reasons.append(f"R²_lower={r2_lower:.3f} < {self.config.R2_LOWER_THRESHOLD}")
                reason = "Poor fit: " + ", ".join(reasons)
            
            return LSWTScreeningResult(
                passed=passed,
                r2_total=r2_total,
                r2_lower=r2_lower,
                reason=reason
            )
            
        except Exception as e:
            self._log(f"  LSWT screening failed with exception: {e}")
            return LSWTScreeningResult(
                passed=False,
                r2_total=-1.0,
                r2_lower=-1.0,
                reason=f"Exception during screening: {str(e)}",
                numerical_issue=True
            )
    
    def _compute_dispersion(self, parameters: Dict, B: float, 
                           direction: str = 'b') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute theoretical dispersion for given parameters.
        
        Args:
            parameters: Full parameter dictionary for Hamiltonian
            B: Magnetic field in Tesla
            direction: 'a' or 'b' for spin orientation
        
        Returns:
            (energies_lower, energies_upper): Arrays of energies at data points
        """
        n_lower = self.data_lower.shape[1]
        n_upper = self.data_upper.shape[1]
        
        energies_lower = np.zeros(n_lower)
        energies_upper = np.zeros(n_upper)
        
        # Compute lower band energies
        for i in range(n_lower):
            kvec = self.data_lower[:2, i]
            eigenvals = ham.hamiltonian_diagonalization(kvec, direction, parameters, B)
            energies_lower[i] = np.real(eigenvals[1])  # Second eigenvalue
        
        # Compute upper band energies
        for i in range(n_upper):
            kvec = self.data_upper[:2, i]
            eigenvals = ham.hamiltonian_diagonalization(kvec, direction, parameters, B)
            energies_upper[i] = np.real(eigenvals[0])  # First eigenvalue
        
        return energies_lower, energies_upper
    
    def _compute_r_squared(self, energies_lower: np.ndarray, 
                          energies_upper: np.ndarray) -> Tuple[float, float]:
        """
        Compute R² goodness-of-fit metric.
        
        Returns:
            (r2_total, r2_lower): Total and lower-band R² values
        """
        # Lower band
        residuals_lower = energies_lower - self.data_lower[2]
        variance_lower = self.data_lower[2] - np.mean(self.data_lower[2])
        ss_res_lower = np.sum(residuals_lower**2)
        ss_tot_lower = np.sum(variance_lower**2)
        
        # Upper band
        residuals_upper = energies_upper - self.data_upper[2]
        variance_upper = self.data_upper[2] - np.mean(self.data_upper[2])
        ss_res_upper = np.sum(residuals_upper**2)
        ss_tot_upper = np.sum(variance_upper**2)
        
        # Avoid division by zero
        if ss_tot_lower < 1e-10 or ss_tot_upper < 1e-10:
            return 0.0, 0.0
        
        # Combined R²
        r2_total = 1 - (ss_res_lower + ss_res_upper) / (ss_tot_lower + ss_tot_upper)
        r2_lower = 1 - ss_res_lower / ss_tot_lower
        
        return r2_total, r2_lower
    
    def set_thresholds(self, r2_total: float = None, r2_lower: float = None):
        """
        Update screening thresholds.
        
        Args:
            r2_total: New threshold for total R² (None to keep current)
            r2_lower: New threshold for lower band R² (None to keep current)
        """
        if r2_total is not None:
            self.config.R2_THRESHOLD = r2_total
        if r2_lower is not None:
            self.config.R2_LOWER_THRESHOLD = r2_lower
        
        self._log(f"Updated thresholds: R²_total >= {self.config.R2_THRESHOLD}, "
                 f"R²_lower >= {self.config.R2_LOWER_THRESHOLD}")


# =============================================================================
# INTEGRATED SIMULATION FUNCTION
# =============================================================================

def create_lswt_screened_simulation(
    L: int = 12,
    use_spin_solver: bool = True,
    fast_mode: bool = True,
    screening_mode: bool = False,
    r2_threshold: float = 0.7,
    r2_lower_threshold: float = 0.75,
    verbose: bool = False
) -> Callable:
    """
    Create a simulation function that first screens with LSWT before Monte Carlo.
    
    This is the main integration point for the active learning workflow.
    
    Args:
        L: Lattice size for Monte Carlo
        use_spin_solver: Use C++ solver vs Python ansatz
        fast_mode: Use fast MC settings
        screening_mode: Use ultra-fast MC settings
        r2_threshold: LSWT R² threshold for passing
        r2_lower_threshold: LSWT lower band R² threshold
        verbose: Print verbose output
    
    Returns:
        Function(NormalizedParameters) -> (spins, positions, energy) or None
        Returns None if LSWT screening fails.
    """
    # Initialize LSWT screener
    config = LSWTConfig()
    config.R2_THRESHOLD = r2_threshold
    config.R2_LOWER_THRESHOLD = r2_lower_threshold
    screener = LSWTScreener(config=config, verbose=verbose)
    
    # Import the MC simulation function
    from active_learning_explorer import create_simulation_function
    mc_simulation = create_simulation_function(
        L=L,
        use_spin_solver=use_spin_solver,
        fast_mode=fast_mode,
        screening_mode=screening_mode,
        verbose=verbose
    )
    
    def screened_simulation(params):
        """
        Run LSWT screening, then Monte Carlo if passed.
        
        Returns:
            (spins, positions, energy, lswt_result) tuple, or
            (None, None, None, lswt_result) if screening failed
        """
        # Step 1: LSWT screening
        lswt_result = screener.screen(params)
        
        if verbose:
            print(f"  LSWT screening: {lswt_result}")
        
        if not lswt_result.passed:
            # Return None to indicate screening failure
            # The caller can check lswt_result for details
            return None, None, None, lswt_result
        
        # Step 2: Monte Carlo simulation
        try:
            spins, positions, energy = mc_simulation(params)
            return spins, positions, energy, lswt_result
        except Exception as e:
            if verbose:
                print(f"  MC simulation failed: {e}")
            return None, None, None, lswt_result
    
    return screened_simulation


def create_lswt_screened_simulation_simple(
    L: int = 12,
    use_spin_solver: bool = True,
    fast_mode: bool = True,
    screening_mode: bool = False,
    r2_threshold: float = 0.7,
    r2_lower_threshold: float = 0.75,
    verbose: bool = False
) -> Callable:
    """
    Create a simple simulation function compatible with existing API.
    
    This returns (spins, positions, energy) or (None, None, None) without
    the LSWT result - for drop-in compatibility with existing code.
    
    The returned function also has an attribute `lswt_stats` that tracks:
    - total_screened: Total number of points screened
    - passed: Number that passed LSWT
    - rejected: Number rejected by LSWT
    - numerical_issues: Number with numerical problems
    
    Args:
        Same as create_lswt_screened_simulation
    
    Returns:
        Function(NormalizedParameters) -> (spins, positions, energy) or (None, None, None)
    """
    full_sim = create_lswt_screened_simulation(
        L=L,
        use_spin_solver=use_spin_solver,
        fast_mode=fast_mode,
        screening_mode=screening_mode,
        r2_threshold=r2_threshold,
        r2_lower_threshold=r2_lower_threshold,
        verbose=verbose
    )
    
    # Statistics tracking
    stats = {
        'total_screened': 0,
        'passed': 0,
        'rejected': 0,
        'numerical_issues': 0,
        'mc_completed': 0,
        'mc_failed': 0,
        'r2_total_values': [],
        'r2_lower_values': [],
    }
    
    def simple_simulation(params):
        result = full_sim(params)
        spins, positions, energy, lswt_result = result
        
        # Update statistics
        stats['total_screened'] += 1
        
        if lswt_result.passed:
            stats['passed'] += 1
            if spins is not None:
                stats['mc_completed'] += 1
            else:
                stats['mc_failed'] += 1
        else:
            stats['rejected'] += 1
            if lswt_result.numerical_issue:
                stats['numerical_issues'] += 1
        
        # Track R² values for analysis
        if lswt_result.r2_total >= 0:
            stats['r2_total_values'].append(lswt_result.r2_total)
            stats['r2_lower_values'].append(lswt_result.r2_lower)
        
        return spins, positions, energy
    
    # Attach stats to the function
    simple_simulation.lswt_stats = stats
    simple_simulation.get_stats_summary = lambda: _format_lswt_stats(stats)
    
    return simple_simulation


def _format_lswt_stats(stats: Dict) -> str:
    """Format LSWT statistics for display."""
    lines = []
    lines.append("LSWT Screening Statistics:")
    lines.append(f"  Total screened: {stats['total_screened']}")
    
    if stats['total_screened'] > 0:
        pass_rate = 100 * stats['passed'] / stats['total_screened']
        lines.append(f"  Passed: {stats['passed']} ({pass_rate:.1f}%)")
        lines.append(f"  Rejected: {stats['rejected']} ({100-pass_rate:.1f}%)")
        
        if stats['numerical_issues'] > 0:
            lines.append(f"  Numerical issues: {stats['numerical_issues']}")
        
        if stats['passed'] > 0:
            mc_success_rate = 100 * stats['mc_completed'] / stats['passed']
            lines.append(f"  MC completed: {stats['mc_completed']} ({mc_success_rate:.1f}% of passed)")
            lines.append(f"  MC failed: {stats['mc_failed']}")
        
        if stats['r2_total_values']:
            r2_mean = np.mean(stats['r2_total_values'])
            r2_std = np.std(stats['r2_total_values'])
            lines.append(f"  R² total: {r2_mean:.3f} ± {r2_std:.3f}")
    
    return "\n".join(lines)


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LSWT SCREENER DEMO")
    print("=" * 70)
    
    # Test with known good parameters
    from feature_extractor import NormalizedParameters, KNOWN_SEEDS
    
    screener = LSWTScreener(verbose=True)
    
    print("\nTesting known seed parameters:")
    print("-" * 70)
    
    for name, params in KNOWN_SEEDS.items():
        result = screener.screen(params)
        print(f"  {name}: {result}")
    
    print("\nTesting random parameters:")
    print("-" * 70)
    
    for i in range(5):
        random_params = NormalizedParameters(
            J1xy_sign=-1.0,
            J1z_norm=np.random.uniform(-0.5, 0.5),
            D_norm=np.random.uniform(-0.3, 0.3),
            E_norm=np.random.uniform(-0.2, 0.2),
            F_norm=np.random.uniform(-0.2, 0.2),
            G_norm=np.random.uniform(-0.2, 0.2),
            J3xy_norm=np.random.uniform(-0.3, 0.3),
            J3z_norm=np.random.uniform(-0.3, 0.3),
            J1xy_abs=6.0
        )
        result = screener.screen(random_params)
        print(f"  Random {i+1}: {result}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
