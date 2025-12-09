#!/usr/bin/env python3
"""
Comprehensive analysis of multi-Q states for BCAO
Includes:
1. Side-by-side visualization of 1Q, 2Q, 3Q states
2. Structure factor calculation
3. Parameter space scanning near initial values
"""

import numpy as np
import matplotlib.pyplot as plt
from multi_q_BCAO import MultiQ, visualize_comparison, parameter_scan
from single_q_BCAO import SingleQ

def main_analysis():
    """
    Run complete analysis suite
    """
    # Base BCAO parameters
    L = 8
    J_base = [-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]
    
    print("="*70)
    print("MULTI-Q ANALYSIS SUITE FOR BCAO")
    print("="*70)
    
    # Part 1: Visualization comparison
    print("\n" + "="*70)
    print("PART 1: Visualizing Single-Q vs Multi-Q Configurations")
    print("="*70)
    visualize_comparison(L=L, J=J_base)
    
    # Part 2: Structure factor analysis
    print("\n" + "="*70)
    print("PART 2: Structure Factor Analysis")
    print("="*70)
    
    print("Computing structure factors for all three states...")
    
    # Single-Q
    print("\nSingle-Q structure factor:")
    single_q = SingleQ(L, J_base)
    spins_1q = single_q.generate_spin_configuration()
    
    # Need to create a temporary MultiQ object just for structure factor calculation
    temp_mq = MultiQ(L, 1, J_base)
    temp_mq.positions = single_q.positions
    temp_mq.NN = single_q.NN
    q1, q2, Sq_1q = temp_mq.calculate_structure_factor(spins_1q)
    
    # 2-Q
    print("\n2-Q structure factor:")
    multi_q_2 = MultiQ(L, 2, J_base)
    spins_2q = multi_q_2.generate_spin_configuration()
    multi_q_2.plot_structure_factor(spins_2q, save_path='structure_factor_2q.png')
    
    # 3-Q
    print("\n3-Q structure factor:")
    multi_q_3 = MultiQ(L, 3, J_base)
    spins_3q = multi_q_3.generate_spin_configuration()
    multi_q_3.plot_structure_factor(spins_3q, save_path='structure_factor_3q.png')
    
    # Create comparison plot of all three
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1-Q
    im1 = axes[0].contourf(q1, q2, Sq_1q.T, levels=50, cmap='hot')
    axes[0].set_xlabel('q1 (r.l.u.)')
    axes[0].set_ylabel('q2 (r.l.u.)')
    axes[0].set_title(f'Single-Q Structure Factor\nE={single_q.opt_energy:.3f}')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0])
    
    # 2-Q
    q1, q2, Sq_2q = multi_q_2.calculate_structure_factor(spins_2q)
    im2 = axes[1].contourf(q1, q2, Sq_2q.T, levels=50, cmap='hot')
    axes[1].set_xlabel('q1 (r.l.u.)')
    axes[1].set_ylabel('q2 (r.l.u.)')
    axes[1].set_title(f'2-Q Structure Factor\nE={multi_q_2.opt_energy:.3f}')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1])
    
    # 3-Q
    q1, q2, Sq_3q = multi_q_3.calculate_structure_factor(spins_3q)
    im3 = axes[2].contourf(q1, q2, Sq_3q.T, levels=50, cmap='hot')
    axes[2].set_xlabel('q1 (r.l.u.)')
    axes[2].set_ylabel('q2 (r.l.u.)')
    axes[2].set_title(f'3-Q Structure Factor\nE={multi_q_3.opt_energy:.3f}')
    axes[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('structure_factor_comparison.png', dpi=300, bbox_inches='tight')
    print("\nStructure factor comparison saved to structure_factor_comparison.png")
    plt.show()
    
    # Part 3: Parameter space scan
    print("\n" + "="*70)
    print("PART 3: Parameter Space Scan Near Initial Values")
    print("="*70)
    
    # Scan around the initial J1xy and J3xy values
    J1xy_center = J_base[0]  # -7.6
    J3xy_center = J_base[6]  # 2.5
    
    # Define scan range as ±30% around center values
    J1xy_range = (J1xy_center * 1.3, J1xy_center * 0.7)  # More negative to less negative
    J3xy_range = (J3xy_center * 0.7, J3xy_center * 1.3)
    
    print(f"\nScanning J1xy from {J1xy_range[0]:.2f} to {J1xy_range[1]:.2f}")
    print(f"Scanning J3xy from {J3xy_range[0]:.2f} to {J3xy_range[1]:.2f}")
    
    param1_range, param2_range, E1q, E2q, E3q, gs = parameter_scan(
        L=6,  # Smaller lattice for faster scanning
        base_J=J_base,
        scan_params=['J1xy', 'J3xy'],
        scan_ranges=[J1xy_range, J3xy_range],
        num_points=10
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - spin_config_comparison.png")
    print("  - structure_factor_2q.png")
    print("  - structure_factor_3q.png")
    print("  - structure_factor_comparison.png")
    print("  - parameter_scan_phase_diagram.png")
    print("  - parameter_scan_data.npz")


def quick_visualization():
    """
    Quick visualization for testing
    """
    L = 8
    J = [-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]
    
    print("Running quick visualization comparison...")
    visualize_comparison(L=L, J=J)


def quick_structure_factor():
    """
    Quick structure factor plot
    """
    L = 8
    J = [-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]
    
    print("Computing 3-Q structure factor...")
    multi_q_3 = MultiQ(L, 3, J)
    spins_3q = multi_q_3.generate_spin_configuration()
    multi_q_3.plot_structure_factor(spins_3q)


def quick_parameter_scan():
    """
    Quick parameter scan with fewer points
    """
    J_base = [-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]
    
    # Scan around initial values
    J1xy_center = J_base[0]
    J3xy_center = J_base[6]
    
    J1xy_range = (J1xy_center * 1.3, J1xy_center * 0.7)
    J3xy_range = (J3xy_center * 0.7, J3xy_center * 1.3)
    
    print("Running parameter scan (this will take some time)...")
    parameter_scan(
        L=6,
        base_J=J_base,
        scan_params=['J1xy', 'J3xy'],
        scan_ranges=[J1xy_range, J3xy_range],
        num_points=8
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "vis":
            quick_visualization()
        elif mode == "sf":
            quick_structure_factor()
        elif mode == "scan":
            quick_parameter_scan()
        else:
            print("Usage: python analyze_multi_q.py [vis|sf|scan]")
            print("  vis  - Quick visualization comparison")
            print("  sf   - Quick structure factor plot")
            print("  scan - Quick parameter scan")
            print("  (no argument) - Run full analysis")
    else:
        # Run full analysis
        main_analysis()
