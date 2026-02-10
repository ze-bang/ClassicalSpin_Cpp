#!/usr/bin/env python3
"""Verify 3rd nearest neighbor connectivity."""

import numpy as np

def verify_j3():
    """Check that the 3rd NN offsets give the correct distance."""
    
    # Lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0)/2.0])
    pos_A = np.array([0.0, 0.0])
    pos_B = np.array([0.0, 1.0/np.sqrt(3.0)])
    
    def get_pos(i, j, sub):
        base = pos_A if sub == 'A' else pos_B
        return base + i * a1 + j * a2
    
    # Reference site
    A_ref = get_pos(0, 0, 'A')
    B_ref = get_pos(0, 0, 'B')
    
    # NN distance
    nn_dist = np.linalg.norm(pos_B - pos_A)
    print(f"NN distance: {nn_dist:.4f}")
    
    # 2nd NN distance (same sublattice)
    j2_dist = np.linalg.norm(a1)  # = 1.0
    print(f"2nd NN distance: {j2_dist:.4f}")
    
    # Expected 3rd NN distance (opposite sublattice, longer than NN)
    # On honeycomb, 3rd NN are across a hexagon
    
    print("\n3rd NN from A(0,0) to B sites:")
    j3_A_to_B_offsets = [(1, -2), (-1, 0), (1, 0)]
    
    for di, dj in j3_A_to_B_offsets:
        B_pos = get_pos(di, dj, 'B')
        dist = np.linalg.norm(B_pos - A_ref)
        print(f"  A(0,0) → B({di},{dj}): distance = {dist:.4f}")
    
    print("\n3rd NN from B(0,0) to A sites:")
    j3_B_to_A_offsets = [(-1, 2), (-1, 0), (1, 0)]
    
    for di, dj in j3_B_to_A_offsets:
        A_pos = get_pos(di, dj, 'A')
        dist = np.linalg.norm(A_pos - B_ref)
        print(f"  B(0,0) → A({di},{dj}): distance = {dist:.4f}")
    
    # What is the true 3rd NN distance?
    # Let's compute all A-B distances and find the 3rd shell
    print("\n\nAll A(0,0)-B(i,j) distances (looking for 3rd NN shell):")
    distances = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            B_pos = get_pos(i, j, 'B')
            dist = np.linalg.norm(B_pos - A_ref)
            distances.append((dist, i, j))
    
    distances.sort()
    
    # Group by distance shells
    shells = {}
    for dist, i, j in distances:
        shell = round(dist, 3)
        if shell not in shells:
            shells[shell] = []
        shells[shell].append((i, j))
    
    print("Distance shells (A→B):")
    for shell_dist, sites in sorted(shells.items())[:5]:
        print(f"  d={shell_dist:.4f}: {sites}")


if __name__ == '__main__':
    verify_j3()
