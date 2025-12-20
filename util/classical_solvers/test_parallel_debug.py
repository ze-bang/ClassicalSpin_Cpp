#!/usr/bin/env python3
"""
Debug script to test spin_solver_runner in parallel.
Run this to diagnose what's going wrong with parallel execution.
"""

import sys
import time
from pathlib import Path
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from spin_solver_runner import create_fast_simulation_func
from feature_extractor import NormalizedParameters

def test_single_run():
    """Test a single simulation run."""
    print("=" * 60)
    print("TEST 1: Single simulation run")
    print("=" * 60)
    
    try:
        # Create simulation function with verbose output
        sim_func = create_fast_simulation_func(
            L=8,  # Small lattice
            annealing_steps=1000,
            n_deterministics=200,
            verbose=True,
            cleanup=False  # Keep files for inspection
        )
        
        # Test parameters (normalized)
        params = NormalizedParameters(
            J1z_norm=0.5,
            D_norm=0.1,
            E_norm=0.05,
            F_norm=0.0,
            G_norm=0.0,
            J3xy_norm=0.0,
            J3z_norm=0.0
        )
        
        print(f"\nTesting with params: {params}")
        start = time.time()
        
        spins, positions, energy = sim_func(params)
        
        elapsed = time.time() - start
        print(f"\n✓ Success!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Energy: {energy:.6f}")
        print(f"  N_sites: {len(spins)}")
        print(f"  Work dir: {sim_func._runner.work_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        traceback.print_exc()
        return False


def test_sequential_runs():
    """Test multiple sequential runs."""
    print("\n" + "=" * 60)
    print("TEST 2: Sequential runs (3 simulations)")
    print("=" * 60)
    
    try:
        sim_func = create_fast_simulation_func(
            L=8,
            annealing_steps=1000,
            n_deterministics=200,
            verbose=False,  # Less verbose for multiple runs
            cleanup=True
        )
        
        params_list = [
            NormalizedParameters(0.5, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0),
            NormalizedParameters(0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0),
            NormalizedParameters(0.7, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0),
        ]
        
        results = []
        for i, params in enumerate(params_list):
            print(f"\nRun {i+1}/3: {params}")
            start = time.time()
            
            spins, positions, energy = sim_func(params)
            elapsed = time.time() - start
            
            print(f"  ✓ E={energy:.6f}, time={elapsed:.2f}s")
            results.append((energy, elapsed))
        
        print(f"\n✓ All sequential runs completed")
        avg_time = sum(t for _, t in results) / len(results)
        print(f"  Average time: {avg_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        traceback.print_exc()
        return False


def test_parallel_runs():
    """Test parallel execution using multiprocessing."""
    print("\n" + "=" * 60)
    print("TEST 3: Parallel runs (4 workers, 8 simulations)")
    print("=" * 60)
    
    try:
        from multiprocessing import Pool
        
        # Create simulation function
        sim_func = create_fast_simulation_func(
            L=8,
            annealing_steps=1000,
            n_deterministics=200,
            verbose=False,
            cleanup=True
        )
        
        # Generate test parameters
        import numpy as np
        np.random.seed(42)
        params_list = []
        for i in range(8):
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
        
        def run_single(params):
            """Wrapper for parallel execution."""
            try:
                # Need to create a new sim_func in each worker
                local_sim = create_fast_simulation_func(
                    L=8,
                    annealing_steps=1000,
                    n_deterministics=200,
                    verbose=False,
                    cleanup=True
                )
                start = time.time()
                spins, positions, energy = local_sim(params)
                elapsed = time.time() - start
                return {'success': True, 'energy': energy, 'time': elapsed, 'error': None}
            except Exception as e:
                return {'success': False, 'energy': None, 'time': None, 'error': str(e)}
        
        print(f"\nRunning {len(params_list)} simulations with 4 workers...")
        start = time.time()
        
        with Pool(processes=4) as pool:
            results = pool.map(run_single, params_list)
        
        total_time = time.time() - start
        
        # Analyze results
        successes = sum(1 for r in results if r['success'])
        failures = len(results) - successes
        
        print(f"\n✓ Parallel execution completed")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Successes: {successes}/{len(results)}")
        print(f"  Failures: {failures}/{len(results)}")
        
        if successes > 0:
            avg_time = sum(r['time'] for r in results if r['success']) / successes
            print(f"  Avg time per sim: {avg_time:.2f}s")
            energies = [r['energy'] for r in results if r['success']]
            print(f"  Energy range: [{min(energies):.4f}, {max(energies):.4f}]")
        
        if failures > 0:
            print(f"\n  Errors:")
            for i, r in enumerate(results):
                if not r['success']:
                    print(f"    Run {i+1}: {r['error'][:100]}")
        
        return failures == 0
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("SPIN SOLVER PARALLEL DEBUGGING")
    print("=" * 60)
    
    # Check solver exists
    from spin_solver_runner import SpinSolverRunner
    try:
        runner = SpinSolverRunner()
        print(f"✓ Solver found at: {runner.solver_path}")
        print(f"✓ Solver is executable: {runner.solver_path.exists()}")
    except Exception as e:
        print(f"✗ Solver not found: {e}")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Single Run", test_single_run),
        ("Sequential Runs", test_sequential_runs),
        ("Parallel Runs", test_parallel_runs),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    if all(results.values()):
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed - see output above for details")
