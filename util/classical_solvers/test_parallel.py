#!/usr/bin/env python3
"""
Quick test of parallel simulation capabilities.

Tests the parallel processing infrastructure without running a full
active learning exploration.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from active_learning_explorer import ActiveLearningExplorer, create_simulation_function
from phase_classifier import PhaseType
import multiprocessing


def test_parallel_vs_sequential():
    """Compare parallel vs sequential execution times."""
    
    print("=" * 70)
    print("PARALLEL SIMULATION TEST")
    print("=" * 70)
    
    # Create explorer with small test set
    explorer = ActiveLearningExplorer(
        output_dir="test_parallel_output",
        L=12,
        verbose=False,
    )
    
    # Add a small number of test points
    print("\n[1] Adding 20 LHS test samples...")
    explorer.add_lhs_samples(n_samples=20, seed=42)
    
    # Create fast simulation function
    print("[2] Creating fast simulation function...")
    simulation_func = create_simulation_function(
        L=12,
        use_spin_solver=True,
        fast_mode=True,
        verbose=False
    )
    
    # Test sequential
    print("\n[3] Testing SEQUENTIAL execution (n_jobs=1)...")
    start = time.time()
    explorer.classify_pending_points(simulation_func, n_jobs=1)
    sequential_time = time.time() - start
    
    print(f"\n    Sequential time: {sequential_time:.1f}s")
    print(f"    Time per simulation: {sequential_time/20:.1f}s")
    
    # Reset for parallel test
    for point in explorer.history:
        point.phase = PhaseType.UNKNOWN.value
        point.confidence = 0.0
    
    # Test parallel
    n_cpus = min(8, multiprocessing.cpu_count())  # Use up to 8 cores
    print(f"\n[4] Testing PARALLEL execution (n_jobs={n_cpus})...")
    start = time.time()
    explorer.classify_pending_points(simulation_func, n_jobs=n_cpus)
    parallel_time = time.time() - start
    
    print(f"\n    Parallel time: {parallel_time:.1f}s")
    print(f"    Time per simulation: {parallel_time/20:.1f}s")
    
    # Results
    speedup = sequential_time / parallel_time
    efficiency = speedup / n_cpus * 100
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Sequential time:  {sequential_time:.1f}s")
    print(f"Parallel time:    {parallel_time:.1f}s ({n_cpus} workers)")
    print(f"Speedup:          {speedup:.2f}x")
    print(f"Efficiency:       {efficiency:.1f}%")
    print("\n✓ Parallel processing is working!")
    
    # Show phase distribution
    print("\nPhase distribution:")
    for phase, count in sorted(explorer.phase_counts.items(), key=lambda x: -x[1]):
        print(f"  {phase}: {count}")


if __name__ == "__main__":
    try:
        test_parallel_vs_sequential()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("  Make sure spin_solver is built in build/spin_solver")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
