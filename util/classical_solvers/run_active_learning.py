#!/usr/bin/env python3
"""
Run Active Learning Exploration for BCAO Spin Phases.

This script runs the active learning exploration to discover the parameter
space regions that produce different magnetic phases, with particular focus
on the double-Q meron-antimeron lattice.

Workflow:
    1. Gaussian random forest proposes parameter sets
    2. LSWT (Linear Spin Wave Theory) pre-screening validates fit quality
    3. Only parameters passing LSWT proceed to Monte Carlo phase classification

By default, uses the C++ Monte Carlo spin solver (simulated annealing) to find
ground states without ansatz constraints.

Usage:
    python run_active_learning.py [options]

Options:
    --output-dir DIR      Output directory for results (default: active_learning_results)
    --n-initial N         Number of initial LHS samples (default: 50)
    --n-iterations N      Number of active learning iterations (default: 10)
    --n-per-iteration N   Points to select per iteration (default: 10)
    --lattice-size L      Lattice size for simulations (default: 24)
    --strategy STR        Acquisition strategy: uncertainty, target, balanced (default: balanced)
    --n-jobs N            Number of parallel workers (1=sequential, -1=all CPUs, default: 1)
    --timeout T           Maximum time per simulation in seconds (default: 300)
    --fresh-start         Start fresh, ignore any existing exploration history
    --skip-simulation     Skip actual simulations (for testing with mock data)
    --use-ansatz          Use Python single-Q/double-Q ansatz instead of C++ solver
    --fast-mode           Use fast screening mode with fewer MC steps (default)
    --accurate            Use accurate mode with more MC steps
    --seed N              Random seed for reproducibility
    --verbose             Print verbose output
    
    LSWT Pre-Screening Options:
    --enable-lswt         Enable LSWT pre-screening (default: enabled)
    --disable-lswt        Disable LSWT pre-screening
    --lswt-r2-threshold   R² threshold for LSWT screening (default: 0.7)
    --lswt-r2-lower       R² threshold for lower band (default: 0.75)
    
Examples:
    # Quick test run with mock simulation
    python run_active_learning.py --n-initial 20 --n-iterations 3 --skip-simulation
    
    # Fast exploration with LSWT screening + C++ Monte Carlo (parallel)
    python run_active_learning.py --n-initial 50 --n-iterations 10 --lattice-size 24 --n-jobs 16
    
    # Use all available CPUs with LSWT pre-screening
    python run_active_learning.py --n-initial 100 --n-iterations 20 --n-jobs -1
    
    # Accurate exploration (slower but more reliable, with parallel processing)
    python run_active_learning.py --n-initial 30 --n-iterations 15 --accurate --lattice-size 36 --n-jobs 8
    
    # Focus on finding more meron phases with strict LSWT threshold
    python run_active_learning.py --strategy target --n-iterations 20 --lswt-r2-threshold 0.8
    
    # Use Python ansatz optimization (faster but constrained to single-Q/double-Q)
    python run_active_learning.py --use-ansatz --n-initial 100 --n-iterations 10
    
    # Disable LSWT screening (use only Monte Carlo)
    python run_active_learning.py --disable-lswt --n-initial 50 --n-iterations 10
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from active_learning_explorer import (
    ActiveLearningExplorer,
    ExplorationPoint,
    create_simulation_function,
)
from phase_classifier import PhaseType
from feature_extractor import NormalizedParameters, KNOWN_SEEDS

# LSWT screening imports
try:
    from lswt_screener import (
        LSWTScreener,
        LSWTConfig,
        LSWTScreeningResult,
        create_lswt_screened_simulation_simple,
    )
    # Check if LSWT is fully functional (hamiltonian + data)
    _test_screener = LSWTScreener(verbose=False)
    LSWT_AVAILABLE = _test_screener.available
    if not LSWT_AVAILABLE:
        LSWT_UNAVAILABLE_REASON = "LSWT hamiltonian or experimental data not available"
    else:
        LSWT_UNAVAILABLE_REASON = None
    del _test_screener
    HAS_LSWT = True
except ImportError as e:
    HAS_LSWT = False
    LSWT_AVAILABLE = False
    LSWT_UNAVAILABLE_REASON = f"LSWT screener module import failed: {e}"


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("  BCAO HONEYCOMB SPIN PHASE EXPLORER")
    print("  Active Learning for Magnetic Phase Discovery")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_phase_legend():
    """Print legend of phase types."""
    print("\n  Phase Types:")
    print("  " + "-" * 40)
    for phase in PhaseType:
        print(f"    {phase.name:20s} : {phase.value}")
    print("  " + "-" * 40)


def create_mock_simulation(L: int = 12):
    """
    Create a mock simulation for testing without running actual simulations.
    
    Returns phase based on simple heuristics from parameters.
    """
    from luttinger_tisza import create_honeycomb_lattice
    
    def mock_simulate(params: NormalizedParameters):
        """Mock simulation that returns synthetic results based on parameters."""
        positions, _, _, _, _, _, _ = create_honeycomb_lattice(L)
        N = len(positions)
        
        # Simple heuristic for phase based on parameters
        J1z = params.J1z_norm
        E = params.E_norm
        F = params.F_norm
        J3xy = params.J3xy_norm
        
        # Create synthetic spin configuration based on expected phase
        if abs(J3xy) > 0.2 and abs(E) > 0.15 and abs(F) > 0.15:
            # Conditions for double-Q meron-antimeron (based on known seeds)
            # Create spiral-like pattern
            spins = np.zeros((N, 3))
            for i, pos in enumerate(positions):
                angle = 0.3 * (pos[0] + pos[1])
                angle2 = 0.3 * (pos[0] - pos[1])
                spins[i] = [np.cos(angle) * np.cos(angle2),
                           np.sin(angle) * np.cos(angle2),
                           np.sin(angle2)]
            energy = -0.8 - 0.1 * abs(J3xy)
            
        elif abs(J1z) > 0.5 and abs(J3xy) < 0.1:
            # Ferromagnetic-like
            spins = np.tile([0, 0, 1], (N, 1)).astype(float)
            energy = -1.0
            
        elif abs(E) < 0.05 and abs(F) < 0.05:
            # Zigzag pattern
            spins = np.zeros((N, 3))
            for i in range(N):
                row = int(positions[i, 1] / 0.5)
                spins[i] = [0, 0, 1] if row % 2 == 0 else [0, 0, -1]
            energy = -0.7
            
        else:
            # General incommensurate
            spins = np.zeros((N, 3))
            q = np.array([0.3, 0.2])
            for i, pos in enumerate(positions):
                angle = np.dot(q, pos)
                spins[i] = [np.cos(angle), np.sin(angle), 0.3]
            spins = spins / np.linalg.norm(spins, axis=1, keepdims=True)
            energy = -0.6
        
        return spins, positions, energy
    
    return mock_simulate


def run_exploration(args):
    """Run the active learning exploration."""
    
    print_banner()
    
    if args.verbose:
        print_phase_legend()
    
    # Handle fresh start - delete old history if requested
    if args.fresh_start:
        history_file = Path(args.output_dir) / "exploration_history.json"
        if history_file.exists():
            print(f"\n[Fresh Start] Removing existing history: {history_file}")
            history_file.unlink()
    
    # Create explorer
    # J1xy_abs=None enables sampling J1xy_abs from J1xy_abs_bounds during LHS
    explorer = ActiveLearningExplorer(
        output_dir=args.output_dir,
        L=args.lattice_size,
        J1xy_abs=None,  # Sample J1xy_abs (not fixed)
        J1xy_abs_bounds=(4.0, 8.0),  # Range for J1xy_abs sampling
        J1xy_sign=-1.0,
        target_phase=PhaseType.MERON_ANTIMERON.value,
        surrogate_type='random_forest',
        verbose=args.verbose,
    )
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Step 1: Add known seeds
    print("\n[Step 1] Adding known seed parameters...")
    explorer.add_known_seeds()
    
    # Step 2: Add initial LHS samples
    print(f"\n[Step 2] Generating {args.n_initial} initial Latin Hypercube samples...")
    explorer.add_lhs_samples(n_samples=args.n_initial, seed=args.seed)
    
    # Determine simulation mode (--accurate overrides --fast-mode, --screening is separate)
    screening_mode = getattr(args, 'screening', False)
    fast_mode = args.fast_mode and not args.accurate and not screening_mode
    
    # Determine LSWT screening settings
    enable_lswt = args.enable_lswt and not args.disable_lswt
    lswt_r2_threshold = args.lswt_r2_threshold
    lswt_r2_lower = args.lswt_r2_lower
    
    # Print LSWT screening status and enforce requirement
    if enable_lswt:
        if not LSWT_AVAILABLE:
            print(f"\n[ERROR] LSWT screening is required but not available!")
            print(f"        Reason: {LSWT_UNAVAILABLE_REASON}")
            print(f"\nTo fix this, either:")
            print(f"  1. Install pandas: pip install pandas")
            print(f"  2. Ensure workflow/LSWT_fit/ contains experimental data CSV files")
            print(f"  3. Use --disable-lswt to skip LSWT screening (not recommended)")
            sys.exit(1)
        print(f"\n[LSWT Screening] Enabled")
        print(f"         R² threshold (total): {lswt_r2_threshold}")
        print(f"         R² threshold (lower band): {lswt_r2_lower}")
    else:
        print(f"\n[LSWT Screening] Disabled by user")
    
    # Step 3: Create simulation function
    if args.skip_simulation:
        print("\n[Step 3] Using mock simulation (--skip-simulation enabled)...")
        simulation_func = create_mock_simulation(L=args.lattice_size)
    elif args.use_ansatz:
        print("\n[Step 3] Using Python single-Q/double-Q ansatz optimization...")
        simulation_func = create_simulation_function(
            L=args.lattice_size, 
            use_spin_solver=False,
            verbose=args.verbose
        )
    else:
        if screening_mode:
            mode_str = "ultra-fast screening (~3s/sim)"
            L = 8  # Override to small lattice
        elif args.accurate:
            mode_str = "accurate (~90s/sim)"
            L = args.lattice_size
        else:
            mode_str = "fast (~15s/sim)"
            L = args.lattice_size
        
        print(f"\n[Step 3] Creating simulation pipeline...")
        print(f"         Monte Carlo Mode: {mode_str}, L={L}")
        
        if enable_lswt:
            # Use LSWT-screened simulation function
            print(f"         Pipeline: LSWT Screening → Monte Carlo → Phase Classification")
            simulation_func = create_lswt_screened_simulation_simple(
                L=L,
                use_spin_solver=True,
                fast_mode=fast_mode,
                screening_mode=screening_mode,
                r2_threshold=lswt_r2_threshold,
                r2_lower_threshold=lswt_r2_lower,
                verbose=args.verbose
            )
        else:
            # Direct Monte Carlo simulation (no LSWT screening)
            print(f"         Pipeline: Monte Carlo → Phase Classification (no LSWT)")
            simulation_func = create_simulation_function(
                L=L,
                use_spin_solver=True,
                fast_mode=fast_mode,
                screening_mode=screening_mode,
                verbose=args.verbose
            )
    
    # Step 4: Classify initial samples
    print("\n[Step 4] Classifying initial samples...")
    explorer.classify_pending_points(simulation_func, n_jobs=args.n_jobs, timeout=args.timeout)
    
    # Step 5: Fit initial surrogate model
    print("\n[Step 5] Fitting surrogate model...")
    explorer.fit_surrogate()
    
    # Step 6: Active learning loop
    print(f"\n[Step 6] Running {args.n_iterations} active learning iterations...")
    print("=" * 70)
    
    for iteration in range(args.n_iterations):
        print(f"\n--- Iteration {iteration + 1}/{args.n_iterations} ---")
        
        # Select new points
        new_points = explorer.select_next_points(
            n_points=args.n_per_iteration,
            strategy=args.strategy,
        )
        
        # Classify new points
        print(f"  Classifying {len(new_points)} new points...")
        explorer.classify_pending_points(simulation_func, n_jobs=args.n_jobs, timeout=args.timeout)
        
        # Update surrogate
        print("  Updating surrogate model...")
        explorer.fit_surrogate()
        
        # Save intermediate results
        explorer.save_history()
        
        # Print current phase distribution
        print("  Current phase distribution:")
        total = sum(explorer.phase_counts.values())
        for phase, count in sorted(explorer.phase_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total if total > 0 else 0
            print(f"    {phase}: {count} ({pct:.1f}%)")
        
        # Print LSWT pass rate if available
        if enable_lswt and hasattr(simulation_func, 'lswt_stats'):
            stats = simulation_func.lswt_stats
            if stats['total_screened'] > 0:
                pass_rate = 100 * stats['passed'] / stats['total_screened']
                print(f"  LSWT pass rate: {stats['passed']}/{stats['total_screened']} ({pass_rate:.1f}%)")
    
    # Step 7: Final summary
    print("\n" + "=" * 70)
    print("[Step 7] FINAL RESULTS")
    print("=" * 70)
    
    # Print LSWT screening statistics if available
    if enable_lswt and hasattr(simulation_func, 'get_stats_summary'):
        print("\n" + simulation_func.get_stats_summary())
        print()
    
    explorer.print_summary()
    explorer.print_feature_importances()
    
    # Save final results
    explorer.save_history()
    explorer.save_model()
    
    # Print interesting findings
    print("\n" + "=" * 70)
    print("INTERESTING FINDINGS")
    print("=" * 70)
    
    # Find all meron points
    meron_points = [p for p in explorer.history 
                   if p.phase == PhaseType.MERON_ANTIMERON.value]
    
    if meron_points:
        print(f"\nFound {len(meron_points)} Double-Q Meron-Antimeron configurations:")
        print("-" * 60)
        
        for i, p in enumerate(meron_points[:10]):
            print(f"\n  Configuration {i+1} (Point {p.point_id}):")
            print(f"    Parameters: {p.params}")
            print(f"    Confidence: {p.confidence:.2%}")
            print(f"    Selection: {p.selection_method}")
            
            # Print key decision flags
            if 'q_vector' in p.decision_flags:
                q_flags = p.decision_flags['q_vector']
                print(f"    Q-vector flags: incomm={q_flags.get('incommensurate', False)}")
            
            if 'multi_q' in p.decision_flags:
                mq = p.decision_flags['multi_q']
                print(f"    Multi-Q: num={mq.get('num_dominant', 0)}, perp={mq.get('perpendicular', False)}")
            
            if 'topology' in p.decision_flags:
                topo = p.decision_flags['topology']
                print(f"    Topology: meron_sig={topo.get('meron_signature', False)}")
    else:
        print("\nNo Double-Q Meron-Antimeron configurations found in this exploration.")
        print("Try:")
        print("  - Increasing --n-iterations")
        print("  - Using --strategy target to focus on meron regions")
        print("  - Refining parameter bounds around known seeds")
    
    # Phase boundary analysis
    print("\n" + "=" * 70)
    print("PHASE BOUNDARY INSIGHTS")
    print("=" * 70)
    
    importances = explorer.get_feature_importances()
    if importances:
        sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
        
        print("\nMost important parameters for phase transitions:")
        for i, (name, imp) in enumerate(sorted_imp[:3], 1):
            print(f"  {i}. {name}: {imp:.3f}")
            
            # Provide interpretation
            if name == 'J1z_norm':
                print("     → Controls XXZ anisotropy (Ising vs XY character)")
            elif name == 'E_norm':
                print("     → Off-diagonal exchange, breaks rotational symmetry")
            elif name == 'F_norm':
                print("     → Dzyaloshinskii-Moriya-like xz coupling")
            elif name == 'G_norm':
                print("     → Dzyaloshinskii-Moriya-like yz coupling")
            elif name == 'J3xy_norm':
                print("     → Third-neighbor frustration, stabilizes incommensurate order")
            elif name == 'D_norm':
                print("     → Easy-axis/easy-plane anisotropy")
    
    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE")
    print(f"Results saved to: {args.output_dir}/")
    print("=" * 70)
    
    return explorer


def main():
    parser = argparse.ArgumentParser(
        description="Active Learning Exploration for BCAO Spin Phases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--output-dir', type=str, default='active_learning_results',
                        help='Output directory for results')
    parser.add_argument('--n-initial', type=int, default=50,
                        help='Number of initial LHS samples')
    parser.add_argument('--n-iterations', type=int, default=10,
                        help='Number of active learning iterations')
    parser.add_argument('--n-per-iteration', type=int, default=10,
                        help='Points to select per iteration')
    parser.add_argument('--lattice-size', type=int, default=12,
                        help='Lattice size for simulations (L x L unit cells)')
    parser.add_argument('--strategy', type=str, default='balanced',
                        choices=['uncertainty', 'target', 'balanced'],
                        help='Acquisition strategy')
    parser.add_argument('--skip-simulation', action='store_true',
                        help='Skip actual simulations (for testing)')
    parser.add_argument('--use-ansatz', action='store_true',
                        help='Use Python single-Q/double-Q ansatz instead of C++ solver')
    parser.add_argument('--fast-mode', action='store_true', default=True,
                        help='Use fast screening mode (fewer MC steps, ~15s/sim)')
    parser.add_argument('--screening', action='store_true',
                        help='Use ultra-fast screening mode (~3s/sim, L=8)')
    parser.add_argument('--accurate', action='store_true',
                        help='Use accurate mode (more MC steps, larger lattice, ~90s/sim)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel workers for simulations (1=sequential, -1=all CPUs)')
    parser.add_argument('--timeout', type=float, default=None,
                        help='Maximum time per simulation in seconds (default: None = no timeout)')
    parser.add_argument('--fresh-start', action='store_true',
                        help='Start fresh, ignore any existing exploration history')
    
    # LSWT Pre-Screening Options
    lswt_group = parser.add_argument_group('LSWT Pre-Screening',
                                           'Options for Linear Spin Wave Theory pre-screening')
    lswt_group.add_argument('--enable-lswt', action='store_true', default=True,
                            help='Enable LSWT pre-screening (default: enabled)')
    lswt_group.add_argument('--disable-lswt', action='store_true',
                            help='Disable LSWT pre-screening')
    lswt_group.add_argument('--lswt-r2-threshold', type=float, default=0.7,
                            help='R² threshold for LSWT total fit (default: 0.7)')
    lswt_group.add_argument('--lswt-r2-lower', type=float, default=0.75,
                            help='R² threshold for LSWT lower band fit (default: 0.75)')
    
    args = parser.parse_args()
    
    explorer = run_exploration(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
