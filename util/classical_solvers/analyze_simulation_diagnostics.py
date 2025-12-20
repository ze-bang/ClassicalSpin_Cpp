#!/usr/bin/env python3
"""
Diagnostic Analysis Tool for Failed/Timeout Simulations.

This tool analyzes a batch of simulations to understand why they are
timing out or producing undesirable results (disordered, etc.).

Usage:
    python analyze_simulation_diagnostics.py --exploration-dir ./my_exploration
    
    # Or test with specific parameters
    python analyze_simulation_diagnostics.py --test-params fitting_param_4 --n-test 10
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from feature_extractor import NormalizedParameters, KNOWN_SEEDS
from phase_classifier import PhaseType
from spin_solver_runner import create_fast_simulation_func, create_accurate_simulation_func


def analyze_exploration_history(history_file: Path):
    """
    Analyze completed exploration to identify patterns in failures.
    
    Args:
        history_file: Path to exploration_history.json
    """
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if 'points' in data:
        history = data['points']
        metadata = data.get('metadata', {})
    else:
        history = data
        metadata = {}
    
    print("=" * 70)
    print("EXPLORATION DIAGNOSTICS ANALYSIS")
    print("=" * 70)
    print(f"History file: {history_file}")
    print(f"Total points: {len(history)}")
    if metadata:
        print(f"Lattice size: L={metadata.get('L', '?')}")
        print(f"Target phase: {metadata.get('target_phase', '?')}")
    print()
    
    # Count phases
    phase_counts = defaultdict(int)
    timeout_params = []
    disordered_params = []
    
    for point in history:
        phase = point['phase']
        phase_counts[phase] += 1
        
        if phase == PhaseType.TIMEOUT.value or phase == "Timeout":
            timeout_params.append(point['params'])
        elif phase == PhaseType.DISORDERED.value or phase == "Disordered":
            disordered_params.append(point['params'])
    
    # Print phase distribution
    print("Phase Distribution:")
    print("-" * 40)
    total = len(history)
    for phase, count in sorted(phase_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {phase:25s}: {count:4d} ({pct:5.1f}%)")
    print()
    
    # Analyze timeout patterns
    if timeout_params:
        print("=" * 70)
        print(f"TIMEOUT ANALYSIS ({len(timeout_params)} cases)")
        print("=" * 70)
        
        # Extract parameter ranges for timeouts
        param_names = ['J1z_norm', 'D_norm', 'E_norm', 'F_norm', 'G_norm', 'J3xy_norm', 'J3z_norm']
        
        print("\nParameter ranges for TIMEOUT simulations:")
        print("-" * 40)
        for param_name in param_names:
            values = [p[param_name] for p in timeout_params]
            print(f"  {param_name:12s}: [{min(values):6.3f}, {max(values):6.3f}]  "
                  f"mean={np.mean(values):6.3f}")
        
        print("\nPossible causes:")
        print("  1. Slow equilibration at low T")
        print("  2. Frustrated systems with many local minima")
        print("  3. Too few annealing steps for complex phases")
        print("  4. Phase transitions near these parameters")
        print()
        
        # Check for large coupling values (slow dynamics)
        large_J3xy = [p for p in timeout_params if abs(p['J3xy_norm']) > 0.3]
        large_E = [p for p in timeout_params if abs(p['E_norm']) > 0.3]
        
        if large_J3xy:
            print(f"  → {len(large_J3xy)} timeouts have |J3xy| > 0.3 (long-range frustration)")
        if large_E:
            print(f"  → {len(large_E)} timeouts have |E| > 0.3 (strong off-diagonal coupling)")
    
    # Analyze disordered patterns
    if disordered_params:
        print("\n" + "=" * 70)
        print(f"DISORDERED ANALYSIS ({len(disordered_params)} cases)")
        print("=" * 70)
        
        param_names = ['J1z_norm', 'D_norm', 'E_norm', 'F_norm', 'G_norm', 'J3xy_norm', 'J3z_norm']
        
        print("\nParameter ranges for DISORDERED configurations:")
        print("-" * 40)
        for param_name in param_names:
            values = [p[param_name] for p in disordered_params]
            print(f"  {param_name:12s}: [{min(values):6.3f}, {max(values):6.3f}]  "
                  f"mean={np.mean(values):6.3f}")
        
        print("\nPossible causes:")
        print("  1. Starting temperature too low (trapped in local minima)")
        print("  2. Competing interactions (no clear ground state)")
        print("  3. Need longer equilibration at each T")
        print("  4. True disordered ground state (e.g., spin glass)")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    timeout_rate = phase_counts[PhaseType.TIMEOUT.value] / total
    disorder_rate = phase_counts[PhaseType.DISORDERED.value] / total
    
    if timeout_rate > 0.3:
        print("\n⚠ High timeout rate (>30%)")
        print("  Recommended actions:")
        print("  1. Increase timeout: change timeout=300 to timeout=600 in runner")
        print("  2. Increase annealing_steps: 2000 → 5000 or more")
        print("  3. Slower cooling: cooling_rate 0.9 → 0.95")
        print("  4. Use --accurate mode instead of --fast-mode")
    
    if disorder_rate > 0.2:
        print("\n⚠ High disordered rate (>20%)")
        print("  Recommended actions:")
        print("  1. Increase T_start: 5.0 → 10.0 (better initial randomization)")
        print("  2. More steps per temperature: increase overrelaxation_rate")
        print("  3. Longer deterministic refinement: n_deterministics 500 → 2000")
        print("  4. Multiple annealing runs (take lowest energy)")
    
    if timeout_rate < 0.1 and disorder_rate < 0.1:
        print("\n✓ Simulation parameters appear adequate!")
        print("  Current settings are working well.")
    
    print()


def test_simulation_with_diagnostics(params_name: str, n_tests: int = 5, 
                                     verbose: bool = True):
    """
    Test simulations with specific parameters to see detailed diagnostics.
    
    Args:
        params_name: Name of parameter set (e.g., 'fitting_param_4')
        n_tests: Number of test simulations to run
        verbose: Show detailed output
    """
    print("=" * 70)
    print("DIAGNOSTIC TEST RUN")
    print("=" * 70)
    
    # Get parameters
    if params_name in KNOWN_SEEDS:
        params = KNOWN_SEEDS[params_name]
        print(f"Testing with: {params_name}")
        print(f"Parameters: {params}")
    else:
        print(f"Unknown parameter set: {params_name}")
        print(f"Available: {list(KNOWN_SEEDS.keys())}")
        return
    
    print(f"\nRunning {n_tests} test simulations with diagnostics...")
    print("-" * 70)
    
    # Create simulation function with verbose output
    sim_func = create_fast_simulation_func(L=12, verbose=verbose)
    
    energies = []
    for i in range(n_tests):
        print(f"\nTest {i+1}/{n_tests}:")
        print("-" * 40)
        
        result = sim_func(params)
        
        if result is not None and result[0] is not None:
            spins, positions, energy = result
            energies.append(energy)
            print(f"✓ Success: E/N = {energy:.6f}")
        else:
            print("✗ Failed or timeout")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if energies:
        print(f"Successful runs: {len(energies)}/{n_tests}")
        print(f"Energy statistics:")
        print(f"  Mean:   {np.mean(energies):.6f}")
        print(f"  Std:    {np.std(energies):.6f}")
        print(f"  Min:    {np.min(energies):.6f}")
        print(f"  Max:    {np.max(energies):.6f}")
        print(f"  Range:  {np.max(energies) - np.min(energies):.6f}")
        
        if np.std(energies) > 0.01:
            print("\n⚠ High energy variance - simulations may not be converging well")
        else:
            print("\n✓ Low energy variance - simulations are converging consistently")
    else:
        print("All simulations failed!")
        print("\nRecommended actions:")
        print("  1. Check that spin_solver executable is working")
        print("  2. Try with --accurate mode")
        print("  3. Increase timeout")


def plot_exploration_parameter_space(history_file: Path):
    """
    Plot 2D parameter space colored by phase/timeout.
    
    Args:
        history_file: Path to exploration_history.json
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if 'points' in data:
        history = data['points']
    else:
        history = data
    
    # Extract data
    phases = []
    J3xy = []
    E = []
    
    for point in history:
        phases.append(point['phase'])
        J3xy.append(point['params']['J3xy_norm'])
        E.append(point['params']['E_norm'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map for phases (use string values directly)
    phase_colors = {
        "Timeout": 'red',
        "Disordered": 'gray',
        "Ferromagnetic": 'blue',
        "Antiferromagnetic": 'cyan',
        "Single-Q": 'green',
        "Meron-Antimeron": 'gold',
        "Double-Q": 'orange',
        "Triple-Q": 'purple',
        "Zigzag": 'brown',
        "Beating": 'pink',
        "Incommensurate": 'lime',
        "Incommensurate Γ→M": 'teal',
        "Incommensurate Γ→K": 'olive',
    }
    
    for phase_type in set(phases):
        mask = [p == phase_type for p in phases]
        x = [J3xy[i] for i, m in enumerate(mask) if m]
        y = [E[i] for i, m in enumerate(mask) if m]
        
        color = phase_colors.get(phase_type, 'black')
        label = phase_type.replace('_', ' ').title()
        
        ax.scatter(x, y, c=color, label=label, alpha=0.6, s=50)
    
    ax.set_xlabel('J3xy (normalized)', fontsize=12)
    ax.set_ylabel('E (normalized)', fontsize=12)
    ax.set_title('Parameter Space Exploration: J3xy vs E', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = history_file.parent / 'parameter_space_plot.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic analysis for simulation timeouts and failures"
    )
    
    parser.add_argument('--exploration-dir', type=str,
                        help='Directory containing exploration_history.json')
    parser.add_argument('--test-params', type=str,
                        help='Test specific parameters (e.g., fitting_param_4)')
    parser.add_argument('--n-test', type=int, default=5,
                        help='Number of test simulations')
    parser.add_argument('--plot', action='store_true',
                        help='Generate parameter space plot')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output for test runs')
    
    args = parser.parse_args()
    
    if args.exploration_dir:
        history_file = Path(args.exploration_dir) / 'exploration_history.json'
        if not history_file.exists():
            print(f"Error: {history_file} not found")
            return 1
        
        analyze_exploration_history(history_file)
        
        if args.plot:
            plot_exploration_parameter_space(history_file)
    
    elif args.test_params:
        test_simulation_with_diagnostics(
            args.test_params,
            n_tests=args.n_test,
            verbose=args.verbose
        )
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
