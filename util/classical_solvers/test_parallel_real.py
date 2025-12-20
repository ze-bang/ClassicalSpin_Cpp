#!/usr/bin/env python3
"""
Test script that replicates the actual parallel execution used in active learning.
This uses ProcessPoolExecutor just like the real code.
"""

import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from spin_solver_runner import create_fast_simulation_func, create_screening_simulation_func
from feature_extractor import NormalizedParameters
import numpy as np

# Copy the actual _run_simulation_safe from active_learning_explorer
def _run_simulation_safe(params, simulation_func, timeout: float = 300):
    """Worker function for parallel simulation execution."""
    import sys
    import traceback
    import os
    
    try:
        # Run simulation
        result = simulation_func(params)
        
        if result is None or result[0] is None:
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
        sys.stderr.write(f"[Worker PID {os.getpid()}] TIMEOUT: {e}\n")
        sys.stderr.flush()
        return None
    
    except Exception as e:
        error_msg = f"[Worker PID {os.getpid()}] ERROR: {type(e).__name__}: {e}\n"
        error_msg += f"  Params: {params}\n"
        error_msg += f"  Traceback:\n"
        error_msg += traceback.format_exc()
        sys.stderr.write(error_msg)
        sys.stderr.flush()
        return None


def test_parallel_with_real_executor():
    """Test using ProcessPoolExecutor exactly like the real code."""
    print("=" * 70)
    print("PARALLEL EXECUTION TEST (matches real active learning code)")
    print("=" * 70)
    
    # Configuration
    n_workers = 4
    n_simulations = 8
    
    # Create simulation function (ultra-fast for testing)
    print(f"\n1. Creating simulation function (L=8, fast mode)...")
    sim_func = create_screening_simulation_func(
        L=8,
        annealing_steps=500,   # Very fast
        n_deterministics=100,
        verbose=False,
        cleanup=True
    )
    
    # Get solver path from the function (it will lazy-init in worker)
    from spin_solver_runner import SpinSolverRunner
    test_runner = SpinSolverRunner()
    print(f"   ✓ Solver: {test_runner.solver_path}")
    
    # Generate test parameters
    print(f"\n2. Generating {n_simulations} test parameters...")
    np.random.seed(42)
    params_list = []
    for i in range(n_simulations):
        params = NormalizedParameters(
            J1z_norm=np.random.uniform(0.2, 0.8),
            D_norm=np.random.uniform(0.0, 0.3),
            E_norm=np.random.uniform(0.0, 0.2),
            F_norm=0.0,
            G_norm=0.0,
            J3xy_norm=0.0,
            J3z_norm=0.0
        )
        params_list.append(params)
        print(f"   Params {i+1}: J1z={params.J1z_norm:.3f}, D={params.D_norm:.3f}, E={params.E_norm:.3f}")
    
    # Run parallel simulations
    print(f"\n3. Running {n_simulations} simulations with {n_workers} workers...")
    print("   (watch stderr for errors)")
    print("-" * 70)
    
    results = []
    start_time = time.time()
    
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(_run_simulation_safe, params, sim_func): i
                for i, params in enumerate(params_list)
            }
            
            # Process as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                
                try:
                    result = future.result(timeout=5)  # Short timeout for getting result
                    
                    if result is None:
                        print(f"   [{idx+1}/{n_simulations}] FAILED (returned None)")
                        results.append(None)
                    else:
                        spins, positions, energy = result
                        print(f"   [{idx+1}/{n_simulations}] ✓ E={energy:.6f}, N={len(spins)}")
                        results.append(energy)
                
                except Exception as e:
                    print(f"   [{idx+1}/{n_simulations}] EXCEPTION: {e}")
                    results.append(None)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        return
    
    total_time = time.time() - start_time
    
    # Results
    print("-" * 70)
    print(f"\n4. Results:")
    print(f"   Total time: {total_time:.2f}s")
    
    successes = sum(1 for r in results if r is not None)
    failures = len(results) - successes
    
    print(f"   Successes: {successes}/{len(results)}")
    print(f"   Failures: {failures}/{len(results)}")
    
    if successes > 0:
        energies = [r for r in results if r is not None]
        avg_time = total_time / len(results)
        print(f"   Avg time per sim: {avg_time:.2f}s")
        print(f"   Energy range: [{min(energies):.4f}, {max(energies):.4f}]")
        print(f"   Success rate: {100*successes/len(results):.1f}%")
    
    if failures > 0:
        print(f"\n   ⚠ {failures} simulations failed - check stderr output above for details")
    
    print("\n" + "=" * 70)
    if successes == len(results):
        print("✓ ALL TESTS PASSED")
    elif successes > 0:
        print(f"⚠ PARTIAL SUCCESS ({successes}/{len(results)})")
    else:
        print("✗ ALL TESTS FAILED")
    print("=" * 70)


if __name__ == "__main__":
    # Make sure stderr is unbuffered
    sys.stderr.reconfigure(line_buffering=True)
    
    try:
        test_parallel_with_real_executor()
    except Exception as e:
        print(f"\n✗ Test crashed: {e}")
        traceback.print_exc()
