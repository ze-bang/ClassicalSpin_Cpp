"""
Active Learning Explorer for BCAO Honeycomb Spin Phases.

This module implements an active learning framework for exploring the 
parameter space of the BCAO honeycomb model and discovering regions
that produce different magnetic phases, particularly the double-Q 
meron-antimeron lattice.

Strategy:
1. Start with known seed parameters that produce interesting phases
2. Use Gaussian Process (or Random Forest) as surrogate model
3. Sample new points using acquisition function (uncertainty + exploitation)
4. Run simulations and classify phases
5. Update model and iterate

All parameters are normalized by |J1xy| so that values are in [-1, 1].
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import multiprocessing
import signal
import sys

# Scikit-learn imports (with fallbacks)
try:
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: simple progress tracker
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
        
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
        
        def update(self, n=1):
            self.n += n
        
        def close(self):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()
    warnings.warn("scikit-learn not available. Some features will be disabled.")

from feature_extractor import (
    NormalizedParameters,
    FeatureExtractor,
    FeatureSet,
    FEATURE_NAMES,
    DEFAULT_BOUNDS,
    KNOWN_SEEDS,
    generate_latin_hypercube_samples,
)
from phase_classifier import (
    PhaseClassifier,
    PhaseType,
    ClassificationResult,
)


# ============================================================================
# Module-level worker function for parallel processing
# ============================================================================
# This needs to be at module level for proper pickling in multiprocessing

def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Simulation exceeded time limit")

def _run_simulation_safe(params, simulation_func, timeout: float = None):
    """
    Worker function for parallel simulation execution.
    
    This is a module-level function to ensure proper pickling for multiprocessing.
    Includes timeout handling and exception catching.
    
    Args:
        params: Normalized parameters
        simulation_func: Simulation function
        timeout: Maximum simulation time in seconds (None = no timeout)
        
    Returns:
        (spins, positions, energy) or None if failed
    """
    import sys
    import signal
    import traceback
    
    try:
        # Set up timeout for this specific simulation (only if timeout is set)
        if timeout is not None and timeout > 0:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(timeout))  # Convert to integer seconds
        
        # Run simulation
        result = simulation_func(params)
        
        # Cancel the alarm if simulation completes
        if timeout is not None and timeout > 0:
            signal.alarm(0)
        
        if result is None or result[0] is None:
            # Simulation returned None (timeout or failure)
            sys.stderr.write(f"[Worker PID {os.getpid()}] Simulation returned None for params: {params}\n")
            sys.stderr.flush()
            return None
        
        spins, positions, energy = result
        
        # Validate result
        if spins is None or len(spins) == 0:
            sys.stderr.write(f"[Worker PID {os.getpid()}] Invalid spins result\n")
            sys.stderr.flush()
            return None
        
        return (spins, positions, energy)
    
    except TimeoutError as e:
        # Explicit timeout - cancel alarm and report
        signal.alarm(0)
        sys.stderr.write(f"[Worker PID {os.getpid()}] TIMEOUT after {timeout}s: {e}\n")
        sys.stderr.flush()
        return None
    
    except Exception as e:
        # Catch any exception - cancel alarm and return None
        if timeout is not None and timeout > 0:
            signal.alarm(0)
        error_msg = f"[Worker PID {os.getpid()}] ERROR: {type(e).__name__}: {e}\n"
        error_msg += f"  Params: {params}\n"
        error_msg += f"  Traceback:\n"
        error_msg += traceback.format_exc()
        sys.stderr.write(error_msg)
        sys.stderr.flush()
        return None


# ============================================================================
# Exploration Point Data Structure
# ============================================================================

@dataclass
class ExplorationPoint:
    """A single point in the exploration history."""
    # Unique ID
    point_id: int
    
    # Normalized parameters
    params: NormalizedParameters
    
    # Classification result
    phase: str  # PhaseType.value
    confidence: float
    
    # Key flags that triggered the classification
    decision_flags: Dict[str, Any] = field(default_factory=dict)
    
    # Features extracted
    features: Dict[str, float] = field(default_factory=dict)
    
    # Energy per site
    energy: float = 0.0
    
    # Timestamp
    timestamp: str = ""
    
    # How this point was selected
    selection_method: str = "initial"  # 'initial', 'lhs', 'uncertainty', 'exploitation', 'random'
    
    # Acquisition function value when selected
    acquisition_value: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to serializable dictionary."""
        return {
            'point_id': self.point_id,
            'params': self.params.to_dict(),
            'phase': self.phase,
            'confidence': self.confidence,
            'decision_flags': self.decision_flags,
            'features': self.features,
            'energy': self.energy,
            'timestamp': self.timestamp,
            'selection_method': self.selection_method,
            'acquisition_value': self.acquisition_value,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ExplorationPoint':
        """Create from dictionary."""
        return cls(
            point_id=d['point_id'],
            params=NormalizedParameters.from_dict(d['params']),
            phase=d['phase'],
            confidence=d['confidence'],
            decision_flags=d.get('decision_flags', {}),
            features=d.get('features', {}),
            energy=d.get('energy', 0.0),
            timestamp=d.get('timestamp', ''),
            selection_method=d.get('selection_method', 'unknown'),
            acquisition_value=d.get('acquisition_value', 0.0),
        )


class ActiveLearningExplorer:
    """
    Active learning framework for exploring spin phase space.
    
    Uses a surrogate model to predict phase classification and
    selects new points to explore based on uncertainty or
    expected information gain.
    """
    
    def __init__(self,
                 output_dir: str = "active_learning_results",
                 L: int = 12,
                 J1xy_abs: float = None,
                 J1xy_abs_bounds: Tuple[float, float] = (4.0, 8.0),
                 J1xy_sign: float = -1.0,
                 target_phase: str = "Double-Q Meron-Antimeron",
                 surrogate_type: str = "random_forest",
                 verbose: bool = True):
        """
        Initialize the explorer.
        
        Args:
            output_dir: Directory to save results
            L: Lattice size for simulations
            J1xy_abs: Fixed absolute value of J1xy. If None, sampled from J1xy_abs_bounds.
            J1xy_abs_bounds: Bounds for J1xy_abs sampling (default: 4.0 to 8.0)
            J1xy_sign: Sign of J1xy (usually -1 for FM tendency)
            target_phase: Phase of primary interest for exploitation
            surrogate_type: 'gaussian_process' or 'random_forest'
            verbose: Print verbose output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.L = L
        self.J1xy_abs = J1xy_abs  # None means variable
        self.J1xy_abs_bounds = J1xy_abs_bounds
        self.J1xy_sign = J1xy_sign
        self.target_phase = target_phase
        self.surrogate_type = surrogate_type
        self.verbose = verbose
        
        # History of explored points
        self.history: List[ExplorationPoint] = []
        self.next_id = 0
        
        # Surrogate model
        self.surrogate_model = None
        self.label_encoder = LabelEncoder() if HAS_SKLEARN else None
        self._model_fitted = False
        
        # Feature extractor and classifier
        self.feature_extractor = FeatureExtractor(L=L)
        self.phase_classifier = PhaseClassifier(verbose=False)
        
        # Bounds for normalized parameters
        self.bounds = DEFAULT_BOUNDS.copy()
        
        # Phase counts for monitoring
        self.phase_counts: Dict[str, int] = {}
        
        # Load existing history if available
        self._load_history()
    
    def _log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(msg)
    
    def add_known_seeds(self):
        """Add known parameter sets that produce interesting phases."""
        self._log("\n" + "=" * 60)
        self._log("ADDING KNOWN SEED POINTS")
        self._log("=" * 60)
        
        # Add the known meron-producing parameter sets
        seed_phases = {
            'fitting_param_2': PhaseType.MERON_ANTIMERON.value,
            'fitting_param_4': PhaseType.MERON_ANTIMERON.value,
            'fitting_param_8': PhaseType.MERON_ANTIMERON.value,
        }
        
        for name, expected_phase in seed_phases.items():
            params = KNOWN_SEEDS[name]
            
            # Create exploration point (we assume the phase based on prior knowledge)
            point = ExplorationPoint(
                point_id=self.next_id,
                params=params,
                phase=expected_phase,
                confidence=0.9,  # High confidence since we know these work
                decision_flags={'known_seed': True, 'seed_name': name},
                timestamp=datetime.now().isoformat(),
                selection_method='initial',
            )
            
            self.history.append(point)
            self._update_phase_counts(expected_phase)
            self.next_id += 1
            
            self._log(f"  Added seed '{name}': {params}")
            self._log(f"    Phase: {expected_phase}")
    
    def add_lhs_samples(self, n_samples: int = 50, seed: int = None):
        """
        Add Latin Hypercube samples for initial exploration.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
        """
        self._log(f"\nGenerating {n_samples} Latin Hypercube samples...")
        if self.J1xy_abs is None:
            self._log(f"  J1xy_abs: sampled from {self.J1xy_abs_bounds}")
        else:
            self._log(f"  J1xy_abs: fixed at {self.J1xy_abs}")
        
        samples = generate_latin_hypercube_samples(
            n_samples=n_samples,
            bounds=self.bounds,
            J1xy_sign=self.J1xy_sign,
            J1xy_abs=self.J1xy_abs,
            J1xy_abs_bounds=self.J1xy_abs_bounds,
            seed=seed,
        )
        
        for params in samples:
            point = ExplorationPoint(
                point_id=self.next_id,
                params=params,
                phase=PhaseType.UNKNOWN.value,
                confidence=0.0,
                timestamp=datetime.now().isoformat(),
                selection_method='lhs',
            )
            self.history.append(point)
            self.next_id += 1
        
        self._log(f"  Added {n_samples} LHS samples (pending classification)")
    
    def add_lswt_guided_samples(self, n_samples: int = 50, 
                                 max_attempts_multiplier: int = 20,
                                 r2_threshold: float = 0.7,
                                 r2_lower_threshold: float = 0.75,
                                 seed: int = None):
        """
        Add samples that pass LSWT pre-screening.
        
        Generates random candidates and keeps only those that pass the LSWT
        threshold. This focuses exploration on the physically-relevant region
        of parameter space that fits experimental dispersion data.
        
        Args:
            n_samples: Target number of LSWT-accepted samples to generate
            max_attempts_multiplier: Max attempts = n_samples * this factor
            r2_threshold: LSWT R² threshold for total fit
            r2_lower_threshold: LSWT R² threshold for lower band
            seed: Random seed for reproducibility
        
        Returns:
            Number of samples actually added (may be < n_samples if acceptance rate is very low)
        """
        # Import LSWT screener
        try:
            from lswt_screener import LSWTScreener, LSWTConfig
        except ImportError:
            self._log("  Warning: LSWT screener not available, falling back to regular LHS")
            return self.add_lhs_samples(n_samples, seed)
        
        self._log(f"\nGenerating up to {n_samples} LSWT-guided samples...")
        self._log(f"  LSWT thresholds: R²_total >= {r2_threshold}, R²_lower >= {r2_lower_threshold}")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize LSWT screener
        config = LSWTConfig()
        config.R2_THRESHOLD = r2_threshold
        config.R2_LOWER_THRESHOLD = r2_lower_threshold
        screener = LSWTScreener(config=config, verbose=False)
        
        if not screener.available:
            self._log("  Warning: LSWT screener not functional, falling back to regular LHS")
            return self.add_lhs_samples(n_samples, seed)
        
        max_attempts = n_samples * max_attempts_multiplier
        accepted = 0
        attempted = 0
        r2_values = []
        
        while accepted < n_samples and attempted < max_attempts:
            # Generate a random candidate
            params = self._generate_random_params()
            attempted += 1
            
            # Screen with LSWT
            result = screener.screen(params)
            r2_values.append(result.r2_total)
            
            if result.passed:
                # Add to exploration pool
                point = ExplorationPoint(
                    point_id=self.next_id,
                    params=params,
                    phase=PhaseType.UNKNOWN.value,
                    confidence=0.0,
                    timestamp=datetime.now().isoformat(),
                    selection_method='lswt_guided',
                )
                self.history.append(point)
                self.next_id += 1
                accepted += 1
                
                if accepted % 10 == 0:
                    self._log(f"    Accepted {accepted}/{n_samples} (attempted {attempted}, rate: {100*accepted/attempted:.1f}%)")
        
        acceptance_rate = 100 * accepted / attempted if attempted > 0 else 0
        r2_mean = np.mean(r2_values) if r2_values else 0
        r2_max = np.max(r2_values) if r2_values else 0
        
        self._log(f"  Added {accepted} LSWT-guided samples")
        self._log(f"  Acceptance rate: {acceptance_rate:.1f}% ({accepted}/{attempted})")
        self._log(f"  R² stats: mean={r2_mean:.3f}, max={r2_max:.3f}")
        
        if accepted < n_samples:
            self._log(f"  Warning: Only found {accepted} samples (target was {n_samples})")
            self._log(f"           Consider relaxing thresholds or narrowing parameter bounds")
        
        return accepted
    
    def _generate_random_params(self) -> NormalizedParameters:
        """Generate a single random parameter set within bounds."""
        # Sample normalized parameters uniformly in bounds
        J1z = np.random.uniform(*self.bounds.get('J1z_norm', (-1, 1)))
        D = np.random.uniform(*self.bounds.get('D_norm', (-1, 1)))
        E = np.random.uniform(*self.bounds.get('E_norm', (-1, 1)))
        F = np.random.uniform(*self.bounds.get('F_norm', (-1, 1)))
        G = np.random.uniform(*self.bounds.get('G_norm', (-1, 1)))
        J3xy = np.random.uniform(*self.bounds.get('J3xy_norm', (-1, 1)))
        J3z = np.random.uniform(*self.bounds.get('J3z_norm', (-1, 1)))
        
        # Sample J1xy_abs if variable
        if self.J1xy_abs is None:
            j1xy_abs = np.random.uniform(*self.J1xy_abs_bounds)
        else:
            j1xy_abs = self.J1xy_abs
        
        return NormalizedParameters(
            J1xy_sign=self.J1xy_sign,
            J1z_norm=J1z,
            D_norm=D,
            E_norm=E,
            F_norm=F,
            G_norm=G,
            J3xy_norm=J3xy,
            J3z_norm=J3z,
            J1xy_abs=j1xy_abs,
        )
    
    def add_local_perturbation_samples(self, n_samples: int = 50, 
                                        perturbation_scale: float = 0.1,
                                        seed: int = None):
        """
        Add samples by perturbing known seeds with Gaussian noise.
        
        This focuses exploration in the vicinity of known-good parameters,
        which is useful when the LSWT-accepted region is narrow.
        
        Args:
            n_samples: Number of samples to generate (distributed among seeds)
            perturbation_scale: Std dev of Gaussian perturbation for normalized params
                               (e.g., 0.1 means ~10% variation around each param)
            seed: Random seed for reproducibility
        
        Returns:
            Number of samples added
        """
        self._log(f"\nGenerating {n_samples} local perturbation samples around seeds...")
        self._log(f"  Perturbation scale: {perturbation_scale} (std dev)")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Get seed parameters
        seed_params = list(KNOWN_SEEDS.values())
        n_seeds = len(seed_params)
        
        if n_seeds == 0:
            self._log("  Warning: No known seeds available, falling back to LHS")
            return self.add_lhs_samples(n_samples, seed)
        
        # Distribute samples among seeds (round-robin style)
        samples_per_seed = n_samples // n_seeds
        extra_samples = n_samples % n_seeds
        
        added = 0
        for i, seed_param in enumerate(seed_params):
            n_this_seed = samples_per_seed + (1 if i < extra_samples else 0)
            
            for j in range(n_this_seed):
                # Perturb the normalized parameters
                perturbed = self._perturb_params(seed_param, perturbation_scale)
                
                point = ExplorationPoint(
                    point_id=self.next_id,
                    params=perturbed,
                    phase=PhaseType.UNKNOWN.value,
                    confidence=0.0,
                    timestamp=datetime.now().isoformat(),
                    selection_method='local_perturbation',
                )
                self.history.append(point)
                self.next_id += 1
                added += 1
        
        self._log(f"  Added {added} local perturbation samples (pending classification)")
        return added
    
    def _perturb_params(self, base_params: NormalizedParameters, 
                        scale: float) -> NormalizedParameters:
        """
        Create a perturbed copy of parameters with Gaussian noise.
        
        Args:
            base_params: Base parameter set to perturb
            scale: Standard deviation of Gaussian perturbation
            
        Returns:
            New NormalizedParameters with added noise (clipped to bounds)
        """
        # Perturb each normalized parameter
        def perturb_and_clip(val, key):
            lo, hi = self.bounds.get(key, (-1, 1))
            perturbed = val + np.random.normal(0, scale)
            return np.clip(perturbed, lo, hi)
        
        J1z = perturb_and_clip(base_params.J1z_norm, 'J1z_norm')
        D = perturb_and_clip(base_params.D_norm, 'D_norm')
        E = perturb_and_clip(base_params.E_norm, 'E_norm')
        F = perturb_and_clip(base_params.F_norm, 'F_norm')
        G = perturb_and_clip(base_params.G_norm, 'G_norm')
        J3xy = perturb_and_clip(base_params.J3xy_norm, 'J3xy_norm')
        J3z = perturb_and_clip(base_params.J3z_norm, 'J3z_norm')
        
        # Perturb J1xy_abs if variable
        if self.J1xy_abs is None:
            # Use relative perturbation for J1xy_abs
            j1xy_abs_scale = scale * base_params.J1xy_abs
            j1xy_abs = base_params.J1xy_abs + np.random.normal(0, j1xy_abs_scale)
            j1xy_abs = np.clip(j1xy_abs, *self.J1xy_abs_bounds)
        else:
            j1xy_abs = self.J1xy_abs
        
        return NormalizedParameters(
            J1xy_sign=self.J1xy_sign,
            J1z_norm=J1z,
            D_norm=D,
            E_norm=E,
            F_norm=F,
            G_norm=G,
            J3xy_norm=J3xy,
            J3z_norm=J3z,
            J1xy_abs=j1xy_abs,
        )
    
    def add_adaptive_lswt_samples(self, n_samples: int = 50,
                                   n_initial_screen: int = 500,
                                   n_iterations: int = 5,
                                   exploitation_ratio: float = 0.7,
                                   r2_threshold: float = 0.7,
                                   r2_lower_threshold: float = 0.75,
                                   seed: int = None):
        """
        Adaptive sampling that learns where LSWT-accepted regions are.
        
        Uses a two-phase approach:
        1. Initial broad screening to learn the R² landscape
        2. Surrogate model guides sampling toward high-R² regions
        
        This can find disconnected pockets of LSWT-accepted parameters
        that local perturbation would miss.
        
        Args:
            n_samples: Target number of LSWT-accepted samples
            n_initial_screen: Number of random samples for initial survey
            n_iterations: Number of adaptive refinement iterations
            exploitation_ratio: Fraction of samples near predicted high-R² (vs exploration)
            r2_threshold: LSWT R² threshold for acceptance (total)
            r2_lower_threshold: LSWT R² threshold for lower band
            seed: Random seed
            
        Returns:
            Number of samples actually added
        """
        # Import LSWT screener
        try:
            from lswt_screener import LSWTScreener, LSWTConfig
        except ImportError:
            self._log("  Warning: LSWT screener not available, falling back to LHS")
            return self.add_lhs_samples(n_samples, seed)
        
        if not HAS_SKLEARN:
            self._log("  Warning: sklearn not available for surrogate, falling back to LHS")
            return self.add_lhs_samples(n_samples, seed)
        
        from sklearn.ensemble import RandomForestRegressor
        
        self._log(f"\n{'='*60}")
        self._log(f"ADAPTIVE LSWT SAMPLING")
        self._log(f"{'='*60}")
        self._log(f"  Target samples: {n_samples}")
        self._log(f"  Initial screening: {n_initial_screen} random candidates")
        self._log(f"  Adaptive iterations: {n_iterations}")
        self._log(f"  Exploitation ratio: {exploitation_ratio:.0%}")
        self._log(f"  LSWT thresholds: R²_total >= {r2_threshold}, R²_lower >= {r2_lower_threshold}")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize LSWT screener
        config = LSWTConfig()
        config.R2_THRESHOLD = r2_threshold
        config.R2_LOWER_THRESHOLD = r2_lower_threshold
        screener = LSWTScreener(config=config, verbose=False)
        
        if not screener.available:
            self._log("  Warning: LSWT screener not functional, falling back to LHS")
            return self.add_lhs_samples(n_samples, seed)
        
        # Storage for screening results
        all_params = []  # List of NormalizedParameters
        all_features = []  # Feature vectors for surrogate
        all_r2 = []  # R² values
        all_passed = []  # Pass/fail flags
        accepted_params = []  # Parameters that passed screening
        
        # === Phase 0: Seed with known good parameters ===
        self._log(f"\n[Phase 0] Seeding with known good parameters...")
        for name, seed_params in KNOWN_SEEDS.items():
            # Verify seeds pass LSWT (they should)
            result = screener.screen(seed_params)
            all_params.append(seed_params)
            all_features.append(seed_params.to_feature_vector())
            all_r2.append(result.r2_total)
            all_passed.append(result.passed)
            
            if result.passed:
                accepted_params.append(seed_params)
                self._log(f"    Seed '{name}': R²={result.r2_total:.3f} (PASS)")
            else:
                self._log(f"    Seed '{name}': R²={result.r2_total:.3f} (FAIL - unexpected!)")
        
        self._log(f"  Seeded with {len(accepted_params)} known-good parameters")
        
        # === Phase 1: Initial broad screening ===
        self._log(f"\n[Phase 1] Initial broad screening ({n_initial_screen} samples)...")
        
        for i in range(n_initial_screen):
            params = self._generate_random_params()
            result = screener.screen(params)
            
            all_params.append(params)
            all_features.append(params.to_feature_vector())
            all_r2.append(result.r2_total)
            all_passed.append(result.passed)
            
            if result.passed:
                accepted_params.append(params)
            
            if (i + 1) % 100 == 0:
                n_pass = sum(all_passed)
                self._log(f"    Screened {i+1}/{n_initial_screen}, accepted: {n_pass} ({100*n_pass/(i+1):.1f}%)")
        
        n_initial_accepted = len(accepted_params)
        self._log(f"  Initial screening complete: {n_initial_accepted}/{n_initial_screen} accepted ({100*n_initial_accepted/n_initial_screen:.1f}%)")
        
        if n_initial_accepted >= n_samples:
            # Already have enough, just add the first n_samples
            self._log(f"  Found enough in initial screening, adding {n_samples} samples")
            for params in accepted_params[:n_samples]:
                self._add_exploration_point(params, 'adaptive_lswt')
            return n_samples
        
        # === Phase 2: Train surrogate and adaptively sample ===
        self._log(f"\n[Phase 2] Adaptive refinement ({n_iterations} iterations)...")
        
        X = np.array(all_features)
        y = np.array(all_r2)
        
        samples_per_iter = max(1, (n_samples - n_initial_accepted) // n_iterations)
        candidates_per_iter = samples_per_iter * 20  # Generate 20x candidates per target
        
        for iteration in range(n_iterations):
            if len(accepted_params) >= n_samples:
                break
            
            # Train surrogate model to predict R²
            surrogate = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=seed + iteration if seed else None,
                n_jobs=-1
            )
            surrogate.fit(X, y)
            
            # Generate candidates
            candidates = []
            candidate_features = []
            
            # Exploitation: sample near predicted high-R² regions
            n_exploit = int(candidates_per_iter * exploitation_ratio)
            # Exploration: random samples
            n_explore = candidates_per_iter - n_exploit
            
            # For exploitation, generate candidates near existing accepted points
            # or where surrogate predicts high R²
            if accepted_params and n_exploit > 0:
                # Strategy: perturb accepted parameters with varying scales
                for _ in range(n_exploit):
                    base = accepted_params[np.random.randint(len(accepted_params))]
                    scale = np.random.uniform(0.02, 0.15)  # Variable perturbation
                    params = self._perturb_params(base, scale)
                    candidates.append(params)
                    candidate_features.append(params.to_feature_vector())
            
            # Exploration: uniform random
            for _ in range(n_explore):
                params = self._generate_random_params()
                candidates.append(params)
                candidate_features.append(params.to_feature_vector())
            
            # Predict R² for candidates
            candidate_features_arr = np.array(candidate_features)
            predicted_r2 = surrogate.predict(candidate_features_arr)
            
            # Also get uncertainty (std of tree predictions)
            tree_predictions = np.array([tree.predict(candidate_features_arr) 
                                         for tree in surrogate.estimators_])
            uncertainty = np.std(tree_predictions, axis=0)
            
            # Acquisition: UCB-like (high predicted R² + high uncertainty)
            # Normalized acquisition score
            pred_norm = (predicted_r2 - predicted_r2.mean()) / (predicted_r2.std() + 1e-6)
            unc_norm = (uncertainty - uncertainty.mean()) / (uncertainty.std() + 1e-6)
            acquisition = pred_norm + 0.5 * unc_norm  # Balance exploitation/exploration
            
            # Select top candidates by acquisition score
            top_indices = np.argsort(acquisition)[::-1]
            
            # Screen top candidates until we find enough or run out
            new_accepted = 0
            screened_this_iter = 0
            
            for idx in top_indices:
                if len(accepted_params) >= n_samples:
                    break
                if screened_this_iter >= samples_per_iter * 5:  # Don't screen too many
                    break
                
                params = candidates[idx]
                result = screener.screen(params)
                screened_this_iter += 1
                
                # Add to training data
                all_params.append(params)
                all_features.append(params.to_feature_vector())
                all_r2.append(result.r2_total)
                all_passed.append(result.passed)
                
                if result.passed:
                    accepted_params.append(params)
                    new_accepted += 1
            
            # Update training data
            X = np.array(all_features)
            y = np.array(all_r2)
            
            self._log(f"    Iteration {iteration+1}: screened {screened_this_iter}, "
                     f"accepted {new_accepted}, total accepted: {len(accepted_params)}")
        
        # === Phase 3: Add accepted samples to exploration pool ===
        n_to_add = min(len(accepted_params), n_samples)
        self._log(f"\n[Phase 3] Adding {n_to_add} accepted samples to exploration pool...")
        
        for params in accepted_params[:n_to_add]:
            self._add_exploration_point(params, 'adaptive_lswt')
        
        # Summary
        total_screened = len(all_params)
        acceptance_rate = 100 * len(accepted_params) / total_screened if total_screened > 0 else 0
        self._log(f"\n  Summary:")
        self._log(f"    Total screened: {total_screened}")
        self._log(f"    Total accepted: {len(accepted_params)} ({acceptance_rate:.1f}%)")
        self._log(f"    Added to pool: {n_to_add}")
        self._log(f"    R² range: [{min(all_r2):.3f}, {max(all_r2):.3f}]")
        
        return n_to_add
    
    def _add_exploration_point(self, params: NormalizedParameters, method: str):
        """Helper to add a single exploration point."""
        point = ExplorationPoint(
            point_id=self.next_id,
            params=params,
            phase=PhaseType.UNKNOWN.value,
            confidence=0.0,
            timestamp=datetime.now().isoformat(),
            selection_method=method,
        )
        self.history.append(point)
        self.next_id += 1
    
    def classify_point(self, point: ExplorationPoint,
                       spins: np.ndarray = None,
                       positions: np.ndarray = None,
                       energy: float = None,
                       simulation_func: Callable = None) -> ExplorationPoint:
        """
        Classify a single exploration point.
        
        Either provide spins/positions directly, or provide a simulation_func
        that takes parameters and returns (spins, positions, energy).
        
        Args:
            point: ExplorationPoint to classify
            spins: (N, 3) spin configuration (optional)
            positions: (N, 2) site positions (optional)
            energy: Energy per site (optional)
            simulation_func: Function(params) -> (spins, positions, energy)
            
        Returns:
            Updated ExplorationPoint with classification
        """
        if spins is None and simulation_func is not None:
            # Run simulation to get spin configuration
            try:
                spins, positions, energy = simulation_func(point.params)
                
                # Check for LSWT rejection marker
                if energy == "LSWT_REJECTED":
                    self._log(f"  Point {point.point_id}: LSWT screening rejected")
                    point.phase = PhaseType.LSWT_REJECTED.value
                    point.confidence = 0.0
                    self._update_phase_counts(point.phase)
                    return point
                    
            except (TimeoutError, Exception) as e:
                self._log(f"  Point {point.point_id}: Simulation failed - {e}")
                # Mark as timeout and return
                point.phase = PhaseType.TIMEOUT.value
                point.confidence = 0.0
                self._update_phase_counts(point.phase)
                return point
        
        if spins is None:
            self._log(f"  Point {point.point_id}: No spin data (timeout), marking as TIMEOUT")
            point.phase = PhaseType.TIMEOUT.value
            point.confidence = 0.0
            self._update_phase_counts(point.phase)
            return point
        
        # Extract features
        J_raw = point.params.to_raw()
        features = self.feature_extractor.extract(spins, positions, J_raw, energy)
        
        # Classify
        result = self.phase_classifier.classify(
            spins, positions, energy=energy, L=self.L
        )
        
        # Update point
        point.phase = result.phase.value
        point.confidence = result.confidence
        point.energy = energy if energy is not None else 0.0
        point.features = features.to_dict()
        # Add J1xy_abs to features for MC bookkeeping
        point.features['J1xy_abs'] = point.params.J1xy_abs
        point.features['J1xy'] = point.params.J1xy_sign * point.params.J1xy_abs
        point.decision_flags = result.flags.to_dict()
        
        self._update_phase_counts(point.phase)
        
        return point
    
    def classify_pending_points(self, simulation_func: Callable, n_jobs: int = 1,
                                checkpoint_interval: int = 10, timeout: float = 300):
        """
        Classify all pending (unclassified) points using a simulation function.
        
        Supports parallel execution with robust error handling and checkpointing.
        
        Args:
            simulation_func: Function(NormalizedParameters) -> (spins, positions, energy)
            n_jobs: Number of parallel workers (1=sequential, -1=all CPUs)
            checkpoint_interval: Save progress every N completed simulations
            timeout: Maximum time per simulation in seconds
        """
        pending = [p for p in self.history if p.phase == PhaseType.UNKNOWN.value]
        
        if not pending:
            self._log("No pending points to classify")
            return
        
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        if n_jobs == 1:
            # Sequential execution (existing behavior)
            self._log(f"\nClassifying {len(pending)} pending points (sequential)...")
            for i, point in enumerate(pending):
                self._log(f"  [{i+1}/{len(pending)}] Point {point.point_id}...")
                self.classify_point(point, simulation_func=simulation_func)
                self._log(f"    → {point.phase} (confidence: {point.confidence:.2f})")
                
                # Checkpoint
                if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
                    self.save_history()
        else:
            # Parallel execution
            self._log(f"\nClassifying {len(pending)} points using {n_jobs} parallel workers...")
            self._classify_parallel(pending, simulation_func, n_jobs, checkpoint_interval, timeout)
    
    def _classify_parallel(self, points: List['ExplorationPoint'], 
                          simulation_func: Callable,
                          n_jobs: int,
                          checkpoint_interval: int,
                          timeout: float = 300):
        """
        Internal method for parallel classification with per-simulation timeout.
        
        Args:
            points: List of ExplorationPoint objects to classify
            simulation_func: Simulation function
            n_jobs: Number of parallel workers
            checkpoint_interval: Save interval
            timeout: Maximum time per simulation in seconds (enforced per worker)
        """
        completed = 0
        failed = 0
        
        # Create progress bar
        pbar = tqdm(total=len(points), desc="Classifying", 
                   unit="sim", ncols=100, leave=True)
        
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks with timeout
                future_to_point = {
                    executor.submit(_run_simulation_safe, point.params, simulation_func, timeout): point
                    for point in points
                }
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_point):
                    point = future_to_point[future]
                    
                    try:
                        # Get result (with timeout)
                        result = future.result(timeout=5)  # Short timeout for result retrieval
                        
                        if result is None:
                            # Simulation failed or timed out
                            self._log(f"  Point {point.point_id}: Failed")
                            point.phase = PhaseType.TIMEOUT.value
                            point.confidence = 0.0
                            self._update_phase_counts(point.phase)
                            failed += 1
                        else:
                            spins, positions, energy = result
                            self.classify_point(point, spins=spins, 
                                              positions=positions, energy=energy)
                            completed += 1
                            
                            # Verbose logging
                            if self.verbose:
                                self._log(f"  Point {point.point_id}: {point.phase} "
                                        f"(conf={point.confidence:.2f}, E={energy:.4f})")
                        
                        # Update progress
                        pbar.update(1)
                        pbar.set_postfix({
                            'completed': completed,
                            'failed': failed,
                            'success_rate': f"{100*completed/(completed+failed):.1f}%"
                        })
                        
                        # Checkpoint
                        if checkpoint_interval > 0 and (completed + failed) % checkpoint_interval == 0:
                            self.save_history()
                            self._log(f"  Checkpoint saved ({completed}/{len(points)} completed)")
                    
                    except FutureTimeoutError:
                        self._log(f"  Point {point.point_id}: Result timeout")
                        point.phase = PhaseType.TIMEOUT.value
                        point.confidence = 0.0
                        self._update_phase_counts(point.phase)
                        failed += 1
                        pbar.update(1)
                    
                    except Exception as e:
                        self._log(f"  Point {point.point_id}: Exception - {e}")
                        point.phase = PhaseType.TIMEOUT.value
                        point.confidence = 0.0
                        self._update_phase_counts(point.phase)
                        failed += 1
                        pbar.update(1)
        
        except KeyboardInterrupt:
            self._log("\n\n⚠ Interrupted by user. Saving progress...")
            self.save_history()
            self._log(f"Progress saved: {completed}/{len(points)} completed")
            raise
        
        finally:
            pbar.close()
        
        # Final summary
        self._log(f"\nParallel classification complete:")
        self._log(f"  Completed: {completed}/{len(points)}")
        self._log(f"  Failed: {failed}/{len(points)}")
        self._log(f"  Success rate: {100*completed/(completed+failed):.1f}%" if (completed+failed) > 0 else "  No results")
    
    def fit_surrogate(self):
        """Fit the surrogate model on classified points."""
        if not HAS_SKLEARN:
            self._log("scikit-learn not available, skipping surrogate fitting")
            return
        
        # Get classified points
        classified = [p for p in self.history if p.phase != PhaseType.UNKNOWN.value]
        
        if len(classified) < 5:
            self._log(f"Not enough classified points ({len(classified)}), need at least 5")
            return
        
        # Prepare training data
        X = np.array([p.params.to_feature_vector() for p in classified])
        y_str = [p.phase for p in classified]
        
        # Encode labels
        self.label_encoder.fit(y_str)
        y = self.label_encoder.transform(y_str)
        
        self._log(f"\nFitting surrogate model on {len(classified)} points...")
        self._log(f"  Unique phases: {list(self.label_encoder.classes_)}")
        
        # Fit model
        if self.surrogate_type == 'gaussian_process':
            kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
            self.surrogate_model = GaussianProcessClassifier(
                kernel=kernel,
                n_restarts_optimizer=3,
                random_state=42,
            )
        else:
            self.surrogate_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            )
        
        self.surrogate_model.fit(X, y)
        self._model_fitted = True
        
        # Evaluate on training data
        train_acc = self.surrogate_model.score(X, y)
        self._log(f"  Training accuracy: {train_acc:.2%}")
    
    def predict_phase(self, params: NormalizedParameters) -> Tuple[str, float]:
        """
        Predict phase for given parameters using surrogate.
        
        Returns:
            (predicted_phase, uncertainty)
        """
        if not self._model_fitted:
            return PhaseType.UNKNOWN.value, 1.0
        
        X = params.to_feature_vector().reshape(1, -1)
        
        # Get prediction
        y_pred = self.surrogate_model.predict(X)[0]
        phase = self.label_encoder.inverse_transform([y_pred])[0]
        
        # Get uncertainty (probability of predicted class)
        proba = self.surrogate_model.predict_proba(X)[0]
        uncertainty = 1.0 - np.max(proba)
        
        return phase, uncertainty
    
    def compute_acquisition(self, params: NormalizedParameters,
                            strategy: str = "uncertainty") -> float:
        """
        Compute acquisition function value for candidate parameters.
        
        Args:
            params: Candidate parameters
            strategy: 'uncertainty' (explore) or 'target' (exploit target phase)
            
        Returns:
            Acquisition value (higher = more interesting)
        """
        if not self._model_fitted:
            return np.random.random()  # Random if no model
        
        X = params.to_feature_vector().reshape(1, -1)
        proba = self.surrogate_model.predict_proba(X)[0]
        
        if strategy == "uncertainty":
            # Pure uncertainty sampling
            return 1.0 - np.max(proba)
        
        elif strategy == "target":
            # Probability of target phase
            try:
                target_idx = list(self.label_encoder.classes_).index(self.target_phase)
                return proba[target_idx]
            except ValueError:
                return 0.0
        
        elif strategy == "balanced":
            # Balance uncertainty and target probability
            uncertainty = 1.0 - np.max(proba)
            try:
                target_idx = list(self.label_encoder.classes_).index(self.target_phase)
                target_prob = proba[target_idx]
            except ValueError:
                target_prob = 0.0
            
            return 0.5 * uncertainty + 0.5 * target_prob
        
        else:
            return np.random.random()
    
    def select_next_points(self, n_points: int = 10,
                           strategy: str = "balanced",
                           n_candidates: int = 1000) -> List[NormalizedParameters]:
        """
        Select next points to explore using acquisition function.
        
        Args:
            n_points: Number of points to select
            strategy: Acquisition strategy
            n_candidates: Number of random candidates to evaluate
            
        Returns:
            List of NormalizedParameters for next exploration
        """
        self._log(f"\nSelecting {n_points} points using '{strategy}' strategy...")
        
        # Generate random candidates
        candidates = []
        for _ in range(n_candidates):
            # Sample J1xy_abs if variable
            if self.J1xy_abs is None:
                j1xy_abs_val = np.random.uniform(*self.J1xy_abs_bounds)
            else:
                j1xy_abs_val = self.J1xy_abs
            
            params = NormalizedParameters(
                J1xy_sign=self.J1xy_sign,
                J1z_norm=np.random.uniform(*self.bounds['J1z_norm']),
                D_norm=np.random.uniform(*self.bounds['D_norm']),
                E_norm=np.random.uniform(*self.bounds['E_norm']),
                F_norm=np.random.uniform(*self.bounds['F_norm']),
                G_norm=np.random.uniform(*self.bounds['G_norm']),
                J3xy_norm=np.random.uniform(*self.bounds['J3xy_norm']),
                J3z_norm=np.random.uniform(*self.bounds['J3z_norm']),
                J1xy_abs=j1xy_abs_val,
            )
            acq = self.compute_acquisition(params, strategy)
            candidates.append((params, acq))
        
        # Sort by acquisition value
        candidates.sort(key=lambda x: -x[1])
        
        # Select top candidates
        selected = []
        for params, acq in candidates[:n_points]:
            point = ExplorationPoint(
                point_id=self.next_id,
                params=params,
                phase=PhaseType.UNKNOWN.value,
                confidence=0.0,
                timestamp=datetime.now().isoformat(),
                selection_method=strategy,
                acquisition_value=acq,
            )
            self.history.append(point)
            selected.append(params)
            self.next_id += 1
            
            predicted, unc = self.predict_phase(params)
            self._log(f"  Point {point.point_id}: acq={acq:.3f}, predicted={predicted}, unc={unc:.3f}")
        
        return selected
    
    def get_phase_boundaries(self, param1: str, param2: str,
                              n_grid: int = 50,
                              fixed_params: Dict[str, float] = None
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get phase predictions on a 2D grid for visualization.
        
        Args:
            param1, param2: Parameter names to vary
            n_grid: Grid resolution
            fixed_params: Fixed values for other parameters
            
        Returns:
            (param1_grid, param2_grid, phase_predictions)
        """
        if not self._model_fitted:
            raise ValueError("Surrogate model not fitted yet")
        
        if fixed_params is None:
            fixed_params = {}
        
        # Get bounds
        lo1, hi1 = self.bounds[param1]
        lo2, hi2 = self.bounds[param2]
        
        # Create grid
        p1_vals = np.linspace(lo1, hi1, n_grid)
        p2_vals = np.linspace(lo2, hi2, n_grid)
        P1, P2 = np.meshgrid(p1_vals, p2_vals)
        
        # Get default values for fixed params
        defaults = {
            'J1z_norm': 0.3,
            'D_norm': 0.1,
            'E_norm': 0.2,
            'F_norm': -0.2,
            'G_norm': 0.0,
            'J3xy_norm': 0.25,
            'J3z_norm': 0.0,
        }
        defaults.update(fixed_params)
        
        # Predict on grid
        phases = np.zeros_like(P1, dtype=int)
        
        # Use midpoint of bounds if J1xy_abs is variable
        if self.J1xy_abs is None:
            j1xy_abs_default = 0.5 * (self.J1xy_abs_bounds[0] + self.J1xy_abs_bounds[1])
        else:
            j1xy_abs_default = self.J1xy_abs
        
        for i in range(n_grid):
            for j in range(n_grid):
                params_dict = defaults.copy()
                params_dict[param1] = P1[i, j]
                params_dict[param2] = P2[i, j]
                
                params = NormalizedParameters(
                    J1xy_sign=self.J1xy_sign,
                    J1xy_abs=j1xy_abs_default,
                    **{k: v for k, v in params_dict.items()}
                )
                
                X = params.to_feature_vector().reshape(1, -1)
                phases[i, j] = self.surrogate_model.predict(X)[0]
        
        return P1, P2, phases
    
    def print_summary(self):
        """Print exploration summary."""
        print("\n" + "=" * 70)
        print("EXPLORATION SUMMARY")
        print("=" * 70)
        
        print(f"\nTotal points explored: {len(self.history)}")
        
        classified = [p for p in self.history if p.phase != PhaseType.UNKNOWN.value]
        pending = [p for p in self.history if p.phase == PhaseType.UNKNOWN.value]
        
        print(f"  Classified: {len(classified)}")
        print(f"  Pending: {len(pending)}")
        
        print("\nPhase distribution:")
        for phase, count in sorted(self.phase_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(classified) if classified else 0
            print(f"  {phase}: {count} ({pct:.1f}%)")
        
        print("\nSelection methods:")
        methods = {}
        for p in self.history:
            methods[p.selection_method] = methods.get(p.selection_method, 0) + 1
        for method, count in sorted(methods.items(), key=lambda x: -x[1]):
            print(f"  {method}: {count}")
        
        # Target phase points
        target_points = [p for p in classified if p.phase == self.target_phase]
        print(f"\nTarget phase ({self.target_phase}): {len(target_points)} points")
        
        if target_points:
            print("  Example parameters:")
            for p in target_points[:3]:
                print(f"    {p.params}")
        
        print("=" * 70)
    
    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importances from the surrogate model."""
        if not self._model_fitted:
            return {}
        
        if hasattr(self.surrogate_model, 'feature_importances_'):
            importances = self.surrogate_model.feature_importances_
            return dict(zip(FEATURE_NAMES, importances))
        
        return {}
    
    def print_feature_importances(self):
        """Print feature importances for understanding phase transitions."""
        importances = self.get_feature_importances()
        
        if not importances:
            print("Feature importances not available")
            return
        
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCES (which parameters matter most)")
        print("=" * 60)
        
        sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
        
        for name, imp in sorted_imp:
            bar = "█" * int(40 * imp) + "░" * int(40 * (1 - imp))
            print(f"  {name:12s}: {bar} {imp:.3f}")
        
        print("=" * 60)
    
    def _update_phase_counts(self, phase: str):
        """Update phase count tracking."""
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
    
    def save_history(self, filename: str = None):
        """Save exploration history to JSON."""
        if filename is None:
            filename = self.output_dir / "exploration_history.json"
        
        data = {
            'metadata': {
                'L': self.L,
                'J1xy_abs': self.J1xy_abs,
                'J1xy_sign': self.J1xy_sign,
                'target_phase': self.target_phase,
                'surrogate_type': self.surrogate_type,
                'total_points': len(self.history),
                'timestamp': datetime.now().isoformat(),
            },
            'phase_counts': self.phase_counts,
            'points': [p.to_dict() for p in self.history],
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._log(f"Saved history to {filename}")
    
    def _load_history(self):
        """Load exploration history if exists."""
        filename = self.output_dir / "exploration_history.json"
        
        if not filename.exists():
            return
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.history = [ExplorationPoint.from_dict(p) for p in data['points']]
            self.phase_counts = data.get('phase_counts', {})
            self.next_id = max(p.point_id for p in self.history) + 1 if self.history else 0
            
            self._log(f"Loaded {len(self.history)} points from history")
            
        except Exception as e:
            self._log(f"Warning: Could not load history: {e}")
    
    def save_model(self, filename: str = None):
        """Save the surrogate model."""
        if not self._model_fitted:
            self._log("No model to save")
            return
        
        if filename is None:
            filename = self.output_dir / "surrogate_model.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.surrogate_model,
                'label_encoder': self.label_encoder,
            }, f)
        
        self._log(f"Saved model to {filename}")
    
    def load_model(self, filename: str = None):
        """Load a saved surrogate model."""
        if filename is None:
            filename = self.output_dir / "surrogate_model.pkl"
        
        if not Path(filename).exists():
            self._log(f"No model file found at {filename}")
            return
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.surrogate_model = data['model']
        self.label_encoder = data['label_encoder']
        self._model_fitted = True
        
        self._log(f"Loaded model from {filename}")


def create_simulation_function(L: int = 12, use_spin_solver: bool = True,
                                fast_mode: bool = True, screening_mode: bool = False,
                                verbose: bool = False):
    """
    Create a simulation function for use with the explorer.
    
    Args:
        L: Lattice size (L x L unit cells)
        use_spin_solver: If True, use C++ Monte Carlo solver. If False, use Python ansatz.
        fast_mode: If True, use fewer steps for faster screening.
        screening_mode: If True, use ultra-fast settings (~3s per simulation).
        verbose: Print verbose output during simulation.
    
    Returns:
        Function(NormalizedParameters) -> (spins, positions, energy)
    """
    if use_spin_solver:
        # Use C++ Monte Carlo spin solver
        from spin_solver_runner import (
            create_spin_solver_simulation_func,
            create_fast_simulation_func,
            create_accurate_simulation_func,
            create_screening_simulation_func,
        )
        
        if screening_mode:
            return create_screening_simulation_func(L=L, verbose=verbose)
        elif fast_mode:
            return create_fast_simulation_func(L=L, verbose=verbose)
        else:
            return create_accurate_simulation_func(L=L, verbose=verbose)
    
    # Fallback to Python ansatz optimization
    from luttinger_tisza import create_honeycomb_lattice
    from single_q_BCAO import SingleQ
    from double_q_meron_antimeron import DoubleQMeronAntimeron
    
    def simulate(params: NormalizedParameters) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run simulation for given parameters using Python ansatz."""
        J = params.to_raw()
        
        # Try single-Q first (faster)
        try:
            single_q = SingleQ(L=L, J=J)
            spins_1q = single_q.generate_spin_configuration()
            energy_1q = single_q.opt_energy
        except Exception:
            spins_1q = None
            energy_1q = np.inf
        
        # Try double-Q
        try:
            double_q = DoubleQMeronAntimeron(L=L, J=J)
            spins_2q = double_q.generate_spin_configuration()
            energy_2q = double_q.opt_energy
        except Exception:
            spins_2q = None
            energy_2q = np.inf
        
        # Return lower energy configuration
        if energy_1q <= energy_2q and spins_1q is not None:
            positions = SingleQ(L=L, J=J).positions
            return spins_1q, positions, energy_1q
        elif spins_2q is not None:
            positions = DoubleQMeronAntimeron(L=L, J=J).positions
            return spins_2q, positions, energy_2q
        else:
            # Fallback: random spins
            positions, _, _, _, _, _, _ = create_honeycomb_lattice(L)
            N = len(positions)
            spins = np.random.randn(N, 3)
            spins = spins / np.linalg.norm(spins, axis=1, keepdims=True)
            return spins, positions, 0.0
    
    return simulate


if __name__ == "__main__":
    # Demo of the active learning explorer
    print("=" * 70)
    print("ACTIVE LEARNING EXPLORER DEMO")
    print("=" * 70)
    
    # Create explorer
    explorer = ActiveLearningExplorer(
        output_dir="active_learning_demo",
        L=8,
        verbose=True,
    )
    
    # Add known seeds
    explorer.add_known_seeds()
    
    # Add some LHS samples
    explorer.add_lhs_samples(n_samples=20, seed=42)
    
    # For demo, we'll just classify the seeds (skip simulation for LHS)
    # In practice, you'd use: explorer.classify_pending_points(simulation_func)
    
    # Fit surrogate on known seeds
    explorer.fit_surrogate()
    
    # Select next points
    if explorer._model_fitted:
        explorer.select_next_points(n_points=5, strategy='balanced')
    
    # Print summary
    explorer.print_summary()
    
    # Print feature importances
    explorer.print_feature_importances()
    
    # Save results
    explorer.save_history()
    if explorer._model_fitted:
        explorer.save_model()
