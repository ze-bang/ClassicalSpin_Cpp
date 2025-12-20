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

def _run_simulation_safe(params, simulation_func, timeout: float = 300):
    """
    Worker function for parallel simulation execution.
    
    This is a module-level function to ensure proper pickling for multiprocessing.
    Includes timeout handling and exception catching.
    
    Args:
        params: Normalized parameters
        simulation_func: Simulation function
        timeout: Maximum simulation time in seconds
        
    Returns:
        (spins, positions, energy) or None if failed
    """
    import sys
    import signal
    import traceback
    
    try:
        # Set up timeout for this specific simulation
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout))  # Convert to integer seconds
        
        # Run simulation
        result = simulation_func(params)
        
        # Cancel the alarm if simulation completes
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
                 J1xy_abs: float = 6.0,
                 J1xy_sign: float = -1.0,
                 target_phase: str = "Double-Q Meron-Antimeron",
                 surrogate_type: str = "random_forest",
                 verbose: bool = True):
        """
        Initialize the explorer.
        
        Args:
            output_dir: Directory to save results
            L: Lattice size for simulations
            J1xy_abs: Absolute value of J1xy for unnormalization
            J1xy_sign: Sign of J1xy (usually -1 for FM tendency)
            target_phase: Phase of primary interest for exploitation
            surrogate_type: 'gaussian_process' or 'random_forest'
            verbose: Print verbose output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.L = L
        self.J1xy_abs = J1xy_abs
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
        
        samples = generate_latin_hypercube_samples(
            n_samples=n_samples,
            bounds=self.bounds,
            J1xy_sign=self.J1xy_sign,
            J1xy_abs=self.J1xy_abs,
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
            params = NormalizedParameters(
                J1xy_sign=self.J1xy_sign,
                J1z_norm=np.random.uniform(*self.bounds['J1z_norm']),
                D_norm=np.random.uniform(*self.bounds['D_norm']),
                E_norm=np.random.uniform(*self.bounds['E_norm']),
                F_norm=np.random.uniform(*self.bounds['F_norm']),
                G_norm=np.random.uniform(*self.bounds['G_norm']),
                J3xy_norm=np.random.uniform(*self.bounds['J3xy_norm']),
                J3z_norm=np.random.uniform(*self.bounds['J3z_norm']),
                J1xy_abs=self.J1xy_abs,
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
        
        for i in range(n_grid):
            for j in range(n_grid):
                params_dict = defaults.copy()
                params_dict[param1] = P1[i, j]
                params_dict[param2] = P2[i, j]
                
                params = NormalizedParameters(
                    J1xy_sign=self.J1xy_sign,
                    J1xy_abs=self.J1xy_abs,
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
