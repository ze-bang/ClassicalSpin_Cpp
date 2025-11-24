import numpy as np
import matplotlib.pyplot as plt
from single_q_BCAO import SingleQ
from double_q_BCAO import DoubleQ
import sys

# filepath: /home/pc_linux/ClassicalSpin_Cpp/util/quick_compare.py

def quick_compare(J, L=6, verbose=True):
    """
    Quick comparison of single-Q vs double-Q for a single parameter set
    
    Args:
        J: BCAO parameter array [J1xy, J1z, D, E, F, G, J3xy, J3z]
        L: Lattice size
        verbose: Print detailed output
    
    Returns:
        Dictionary with comparison results
    """
    if verbose:
        print("="*70)
        print("QUICK COMPARISON: Single-Q vs Double-Q")
        print("="*70)
        print(f"Parameters: J1xy={J[0]}, J1z={J[1]}, D={J[2]}, E={J[3]}")
        print(f"            F={J[4]}, G={J[5]}, J3xy={J[6]}, J3z={J[7]}")
        print(f"Lattice size: {L}x{L}")
        print()
    
    # Run single-Q
    if verbose:
        print("Running Single-Q optimization...")
    try:
        single_q = SingleQ(L=L, J=J)
        E_single = single_q.opt_energy
        Q_single = single_q.opt_params[0:2]
        alpha_A = single_q.opt_params[2]
        alpha_B = single_q.opt_params[3]
        phase_single = single_q.magnetic_order
        
        if verbose:
            print(f"  Energy: {E_single:.8f}")
            print(f"  Q-vector: ({Q_single[0]:.6f}, {Q_single[1]:.6f})")
            print(f"  α_A: {alpha_A:.6f}, α_B: {alpha_B:.6f}")
            print(f"  Phase: {phase_single}")
            print()
        
        single_success = True
    except Exception as e:
        if verbose:
            print(f"  Failed: {str(e)}")
            print()
        E_single = np.nan
        single_success = False
    
    # Run double-Q
    if verbose:
        print("Running Double-Q optimization...")
    try:
        double_q = DoubleQ(L=L, J=J)
        E_double = double_q.opt_energy
        Q1 = double_q.opt_params[0:2]
        Q2 = double_q.opt_params[2:4]
        alpha_A_d = double_q.opt_params[4]
        alpha_B_d = double_q.opt_params[5]
        beta_Q1_A = double_q.opt_params[6]
        beta_Q1_B = double_q.opt_params[7]
        beta_Q2_A = double_q.opt_params[8]
        beta_Q2_B = double_q.opt_params[9]
        phase_double = double_q.magnetic_order
        
        if verbose:
            print(f"  Energy: {E_double:.8f}")
            print(f"  Q1-vector: ({Q1[0]:.6f}, {Q1[1]:.6f})")
            print(f"  Q2-vector: ({Q2[0]:.6f}, {Q2[1]:.6f})")
            print(f"  α_A: {alpha_A_d:.6f}, α_B: {alpha_B_d:.6f}")
            print(f"  β_Q1_A: {beta_Q1_A:.6f}, β_Q1_B: {beta_Q1_B:.6f}")
            print(f"  β_Q2_A: {beta_Q2_A:.6f}, β_Q2_B: {beta_Q2_B:.6f}")
            print(f"  Phase: {phase_double}")
            print()
        
        double_success = True
    except Exception as e:
        if verbose:
            print(f"  Failed: {str(e)}")
            print()
        E_double = np.nan
        double_success = False
    
    # Compare results
    if single_success and double_success:
        energy_diff = E_single - E_double
        energy_gain = -energy_diff  # Positive means double-Q is better
        
        if verbose:
            print("="*70)
            print("COMPARISON RESULTS")
            print("="*70)
            print(f"Single-Q energy:  {E_single:.8f}")
            print(f"Double-Q energy:  {E_double:.8f}")
            print(f"Energy difference (Single - Double): {energy_diff:.8f}")
            print(f"Energy gain from Double-Q: {energy_gain:.8f}")
            print()
            
            if abs(energy_diff) < 1e-6:
                print("Result: Both ansätze give EQUIVALENT energies")
                preferred = "Equivalent"
            elif energy_diff > 0:
                print(f"Result: Double-Q is BETTER by {energy_gain:.8f}")
                preferred = "Double-Q"
            else:
                print(f"Result: Single-Q is BETTER by {-energy_gain:.8f}")
                preferred = "Single-Q"
            
            # Check if double-Q is effectively single-Q
            tol = 1e-5
            if (beta_Q1_A < tol and beta_Q1_B < tol) or (beta_Q2_A < tol and beta_Q2_B < tol):
                print("Note: Double-Q solution is effectively single-Q (one Q-vector has negligible amplitude)")
            
            print("="*70)
    else:
        energy_diff = np.nan
        energy_gain = np.nan
        preferred = "Unknown"
        
        if verbose:
            print("="*70)
            print("One or both optimizations failed")
            print("="*70)
    
    return {
        'J': J,
        'L': L,
        'E_single_q': E_single if single_success else None,
        'E_double_q': E_double if double_success else None,
        'energy_diff': energy_diff if not np.isnan(energy_diff) else None,
        'energy_gain': energy_gain if not np.isnan(energy_gain) else None,
        'preferred': preferred,
        'single_q_model': single_q if single_success else None,
        'double_q_model': double_q if double_success else None
    }


def compare_multiple_parameters(J_list, L=6, labels=None):
    """
    Compare single-Q vs double-Q for multiple parameter sets
    
    Args:
        J_list: List of BCAO parameter arrays
        L: Lattice size
        labels: Optional list of labels for each parameter set
    
    Returns:
        List of comparison results
    """
    if labels is None:
        labels = [f"Set {i+1}" for i in range(len(J_list))]
    
    results = []
    
    print("="*70)
    print(f"COMPARING {len(J_list)} PARAMETER SETS")
    print("="*70)
    print()
    
    for i, (J, label) in enumerate(zip(J_list, labels)):
        print(f"\n{'='*70}")
        print(f"PARAMETER SET {i+1}/{len(J_list)}: {label}")
        print(f"{'='*70}")
        
        result = quick_compare(J, L=L, verbose=True)
        result['label'] = label
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Label':<30} {'Preferred':<15} {'Energy Gain':<15}")
    print("-"*70)
    for result in results:
        label = result['label']
        preferred = result['preferred']
        energy_gain = result['energy_gain']
        if energy_gain is not None:
            print(f"{label:<30} {preferred:<15} {energy_gain:>14.8f}")
        else:
            print(f"{label:<30} {preferred:<15} {'N/A':>14}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    # Example 1: Standard BCAO parameters
    print("EXAMPLE 1: Standard BCAO parameters")
    J_standard = [-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85]
    result1 = quick_compare(J_standard, L=8)
    
    print("\n\n")
    
    # Example 2: Modified J3xy
    print("EXAMPLE 2: Comparing different J3xy values")
    J_sets = [
        [-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157],
        [-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85],
        [-7.6, -1.2, 0.1, -0.1, 0, 0, 4.0, -0.85],
    ]
    labels = ['J3xy=1.0', 'J3xy=2.5', 'J3xy=4.0']
    results = compare_multiple_parameters(J_sets, L=6, labels=labels)
    
    # Visualize energy comparison
    valid_results = [r for r in results if r['energy_gain'] is not None]
    if valid_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels_plot = [r['label'] for r in valid_results]
        energy_gains = [r['energy_gain'] for r in valid_results]
        colors = ['red' if eg > 1e-6 else 'blue' if eg < -1e-6 else 'gray' 
                 for eg in energy_gains]
        
        bars = ax.bar(range(len(labels_plot)), energy_gains, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Parameter Set', fontsize=12)
        ax.set_ylabel('Energy Gain from Double-Q', fontsize=12)
        ax.set_title('Single-Q vs Double-Q Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(labels_plot)))
        ax.set_xticklabels(labels_plot, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Double-Q better'),
            Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='Equivalent'),
            Patch(facecolor='blue', alpha=0.7, edgecolor='black', label='Single-Q better')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig('energy_comparison.png', dpi=300, bbox_inches='tight')
        print("\nEnergy comparison plot saved to: energy_comparison.png")
        plt.show()
