#!/usr/bin/env python3
"""
Compare single-Q and multi-Q methods for BCAO honeycomb lattice
"""
import numpy as np
from single_q_BCAO import SingleQ
from multi_q_BCAO import MultiQ

def compare_methods(L=8, J=[-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85], B_field=np.array([0, 0, 0])):
    """
    Compare single-Q and multi-Q results
    """
    print("\n" + "="*70)
    print("COMPARISON: Single-Q vs Multi-Q Methods for BCAO")
    print("="*70)
    print(f"Lattice size: {L}x{L}")
    print(f"BCAO Parameters: J1xy={J[0]}, J1z={J[1]}, D={J[2]}, E={J[3]},")
    print(f"                 F={J[4]}, G={J[5]}, J3xy={J[6]}, J3z={J[7]}")
    print(f"Magnetic field: [{B_field[0]}, {B_field[1]}, {B_field[2]}]")
    
    # Single-Q calculation
    print("\n" + "-"*70)
    print("SINGLE-Q CALCULATION")
    print("-"*70)
    single_q = SingleQ(L, J, B_field)
    print(f"Energy per unit cell: {single_q.opt_energy:.6f}")
    print(f"Magnetic order: {single_q.magnetic_order}")
    
    Q1, Q2 = single_q.opt_params[0], single_q.opt_params[1]
    print(f"Optimal Q-vector: ({Q1:.4f}, {Q2:.4f})")
    
    spins_1q = single_q.generate_spin_configuration()
    mag_1q = single_q.calculate_magnetization(spins_1q)
    print(f"Total magnetization magnitude: {mag_1q['total_magnitude']:.6f}")
    print(f"Staggered magnetization magnitude: {mag_1q['staggered_magnitude']:.6f}")
    
    # Multi-Q calculations
    results = {'single_q': single_q.opt_energy}
    
    for num_Q in [2, 3]:
        print("\n" + "-"*70)
        print(f"MULTI-Q CALCULATION (num_Q = {num_Q})")
        print("-"*70)
        
        multi_q = MultiQ(L, num_Q, J, B_field)
        print(f"Energy per unit cell: {multi_q.opt_energy:.6f}")
        print(f"Magnetic order: {multi_q.magnetic_order}")
        
        Q_vecs, spin_params_A, spin_params_B = multi_q.parse_params(multi_q.opt_params)
        print(f"\nQ-vectors:")
        for i in range(num_Q):
            Q_reduced = np.array([Q_vecs[i].dot(multi_q.b1) / (2*np.pi), 
                                 Q_vecs[i].dot(multi_q.b2) / (2*np.pi)])
            amp_A = spin_params_A[i][0]
            amp_B = spin_params_B[i][0]
            if amp_A > 0.01 or amp_B > 0.01:  # Only show significant Q-vectors
                print(f"  Q{i+1}: ({Q_reduced[0]:.4f}, {Q_reduced[1]:.4f}) "
                      f"| Amp A: {amp_A:.4f}, B: {amp_B:.4f}")
        
        spins_mq = multi_q.generate_spin_configuration()
        mag_mq = multi_q.calculate_magnetization(spins_mq)
        print(f"\nTotal magnetization magnitude: {mag_mq['total_magnitude']:.6f}")
        print(f"Staggered magnetization magnitude: {mag_mq['staggered_magnitude']:.6f}")
        print(f"Scalar chirality: {mag_mq['scalar_chirality']:.6f}")
        
        results[f'{num_Q}q'] = multi_q.opt_energy
    
    # Summary comparison
    print("\n" + "="*70)
    print("ENERGY COMPARISON SUMMARY")
    print("="*70)
    print(f"Single-Q energy:  {results['single_q']:.6f}")
    print(f"2-Q energy:       {results['2q']:.6f}  (Δ = {results['single_q'] - results['2q']:+.6f})")
    print(f"3-Q energy:       {results['3q']:.6f}  (Δ = {results['single_q'] - results['3q']:+.6f})")
    
    print("\n" + "-"*70)
    if abs(results['single_q'] - results['2q']) < 1e-4 and abs(results['single_q'] - results['3q']) < 1e-4:
        print("CONCLUSION: Single-Q ansatz is sufficient (multi-Q doesn't lower energy)")
    elif results['2q'] < results['single_q'] - 1e-4 or results['3q'] < results['single_q'] - 1e-4:
        print("CONCLUSION: Multi-Q state has LOWER energy than single-Q!")
        if results['3q'] < results['2q']:
            print("            → Triple-Q (3-Q) state is the ground state")
        else:
            print("            → Double-Q (2-Q) state is the ground state")
    else:
        print("CONCLUSION: Energy differences are marginal")
    print("="*70 + "\n")
    
    return results

if __name__ == "__main__":
    # Test with default BCAO parameters
    results = compare_methods(L=8, J=[-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85])
    
    # Optionally test with magnetic field
    # print("\n\nTesting with applied magnetic field...")
    # results_field = compare_methods(L=8, J=[-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85], 
    #                                 B_field=np.array([0, 0, 0.5]))
