#!/usr/bin/env python3
"""
Test that multi-Q with num_Q=1 matches single-Q
"""
import numpy as np
from single_q_BCAO import SingleQ
from multi_q_BCAO import MultiQ

def test_consistency():
    """
    Test that multi-Q with num_Q=1 gives the same energy as single-Q
    """
    L = 6
    J = [-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85]
    
    print("="*70)
    print("CONSISTENCY TEST: Single-Q vs Multi-Q with num_Q=1")
    print("="*70)
    print(f"Lattice size: {L}x{L}")
    print(f"Parameters: {J}")
    
    # Single-Q calculation
    print("\n" + "-"*70)
    print("Running Single-Q...")
    print("-"*70)
    single_q = SingleQ(L, J)
    print(f"Single-Q energy: {single_q.opt_energy:.8f}")
    print(f"Q-vector: ({single_q.opt_params[0]:.6f}, {single_q.opt_params[1]:.6f})")
    print(f"Alpha A: {single_q.opt_params[2]:.6f}, Alpha B: {single_q.opt_params[3]:.6f}")
    
    # Multi-Q with num_Q=1
    print("\n" + "-"*70)
    print("Running Multi-Q with num_Q=1...")
    print("-"*70)
    multi_q_1 = MultiQ(L, num_Q=1, J=J)
    print(f"Multi-Q (n=1) energy: {multi_q_1.opt_energy:.8f}")
    
    Q_vecs, params_A, params_B = multi_q_1.parse_params(multi_q_1.opt_params)
    Q_reduced = np.array([Q_vecs[0].dot(multi_q_1.b1) / (2*np.pi), 
                         Q_vecs[0].dot(multi_q_1.b2) / (2*np.pi)])
    print(f"Q-vector: ({Q_reduced[0]:.6f}, {Q_reduced[1]:.6f})")
    print(f"Amplitude A: {params_A[0][0]:.6f}, B: {params_B[0][0]:.6f}")
    
    # Multi-Q with num_Q=2
    print("\n" + "-"*70)
    print("Running Multi-Q with num_Q=2...")
    print("-"*70)
    multi_q_2 = MultiQ(L, num_Q=2, J=J)
    print(f"Multi-Q (n=2) energy: {multi_q_2.opt_energy:.8f}")
    
    Q_vecs, params_A, params_B = multi_q_2.parse_params(multi_q_2.opt_params)
    for i in range(2):
        Q_reduced = np.array([Q_vecs[i].dot(multi_q_2.b1) / (2*np.pi), 
                             Q_vecs[i].dot(multi_q_2.b2) / (2*np.pi)])
        print(f"Q{i+1}: ({Q_reduced[0]:.6f}, {Q_reduced[1]:.6f}) | "
              f"Amp A: {params_A[i][0]:.6f}, B: {params_B[i][0]:.6f}")
    
    # Multi-Q with num_Q=3
    print("\n" + "-"*70)
    print("Running Multi-Q with num_Q=3...")
    print("-"*70)
    multi_q_3 = MultiQ(L, num_Q=3, J=J)
    print(f"Multi-Q (n=3) energy: {multi_q_3.opt_energy:.8f}")
    
    Q_vecs, params_A, params_B = multi_q_3.parse_params(multi_q_3.opt_params)
    for i in range(3):
        Q_reduced = np.array([Q_vecs[i].dot(multi_q_3.b1) / (2*np.pi), 
                             Q_vecs[i].dot(multi_q_3.b2) / (2*np.pi)])
        print(f"Q{i+1}: ({Q_reduced[0]:.6f}, {Q_reduced[1]:.6f}) | "
              f"Amp A: {params_A[i][0]:.6f}, B: {params_B[i][0]:.6f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Single-Q energy:      {single_q.opt_energy:.8f}")
    print(f"Multi-Q (n=1) energy: {multi_q_1.opt_energy:.8f}  (Δ = {abs(single_q.opt_energy - multi_q_1.opt_energy):.8f})")
    print(f"Multi-Q (n=2) energy: {multi_q_2.opt_energy:.8f}  (Δ = {single_q.opt_energy - multi_q_2.opt_energy:+.8f})")
    print(f"Multi-Q (n=3) energy: {multi_q_3.opt_energy:.8f}  (Δ = {single_q.opt_energy - multi_q_3.opt_energy:+.8f})")
    
    print("\n" + "-"*70)
    if abs(single_q.opt_energy - multi_q_1.opt_energy) < 1e-3:
        print("✓ PASS: Single-Q and Multi-Q(n=1) energies match!")
    else:
        print("✗ FAIL: Single-Q and Multi-Q(n=1) energies do NOT match!")
    
    if multi_q_2.opt_energy < single_q.opt_energy - 0.01:
        print("✓ Multi-Q(n=2) has lower energy than Single-Q")
    else:
        print("  Multi-Q(n=2) does not improve over Single-Q")
    
    if multi_q_3.opt_energy < multi_q_2.opt_energy - 0.01:
        print("✓ Multi-Q(n=3) has lower energy than Multi-Q(n=2)")
    else:
        print("  Multi-Q(n=3) does not improve over Multi-Q(n=2)")
    
    print("="*70)

if __name__ == "__main__":
    test_consistency()
