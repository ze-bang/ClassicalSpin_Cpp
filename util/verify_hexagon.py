#!/usr/bin/env python3
"""Verify hexagon connectivity for ring exchange."""

import numpy as np

def verify_hexagon():
    """Check that the hexagon forms a valid closed loop through NN bonds."""
    
    # Lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0)/2.0])
    pos_A = np.array([0.0, 0.0])
    pos_B = np.array([0.0, 1.0/np.sqrt(3.0)])
    
    def get_pos(i, j, sub):
        base = pos_A if sub == 'A' else pos_B
        return base + i * a1 + j * a2
    
    # Hexagon sites for unit cell (i,j) = (0,0)
    # From C++ code:
    hex_sites = [
        (0, 0, 'A'),   # 0: A(i,j)
        (0, 0, 'B'),   # 1: B(i,j)
        (0, 1, 'A'),   # 2: A(i,j+1)
        (1, 0, 'B'),   # 3: B(i+1,j)
        (1, 0, 'A'),   # 4: A(i+1,j)
        (1, -1, 'B'),  # 5: B(i+1,j-1)
    ]
    
    positions = [get_pos(*s) for s in hex_sites]
    
    print("Hexagon sites and positions:")
    for idx, (site, pos) in enumerate(zip(hex_sites, positions)):
        print(f"  {idx}: {site[2]}({site[0]},{site[1]}) at {pos}")
    
    print("\nEdge lengths (should all be NN distance = 1/√3 ≈ 0.577):")
    nn_dist = 1.0 / np.sqrt(3.0)
    
    for i in range(6):
        p1 = positions[i]
        p2 = positions[(i+1) % 6]
        dist = np.linalg.norm(p2 - p1)
        is_nn = "✓" if np.isclose(dist, nn_dist, atol=0.01) else "✗"
        print(f"  Edge {i}→{(i+1)%6}: distance = {dist:.4f} {is_nn}")
    
    # Check what bond type each edge is
    print("\nBond types for each edge:")
    
    # Bond definitions:
    # x-bond: A(i,j) → B(i,j-1)   [from A perspective]
    # y-bond: A(i,j) → B(i+1,j-1) [from A perspective]
    # z-bond: A(i,j) → B(i,j)     [same unit cell]
    
    for i in range(6):
        s1 = hex_sites[i]
        s2 = hex_sites[(i+1) % 6]
        
        # Determine bond type
        bond_type = "?"
        
        # Check from A's perspective
        if s1[2] == 'A' and s2[2] == 'B':
            di, dj = s2[0] - s1[0], s2[1] - s1[1]
            if di == 0 and dj == 0:
                bond_type = "z-bond"
            elif di == 0 and dj == -1:
                bond_type = "x-bond"
            elif di == 1 and dj == -1:
                bond_type = "y-bond"
        elif s1[2] == 'B' and s2[2] == 'A':
            # Reverse: B → A
            di, dj = s1[0] - s2[0], s1[1] - s2[1]
            if di == 0 and dj == 0:
                bond_type = "z-bond (rev)"
            elif di == 0 and dj == -1:
                bond_type = "x-bond (rev)"
            elif di == 1 and dj == -1:
                bond_type = "y-bond (rev)"
        
        print(f"  {i}→{(i+1)%6}: {s1[2]}({s1[0]},{s1[1]}) → {s2[2]}({s2[0]},{s2[1]}) = {bond_type}")
    
    # Verify the path forms a closed hexagon
    print("\nHexagon verification:")
    total_displacement = sum(positions[i+1] - positions[i] for i in range(5))
    total_displacement += positions[0] - positions[5]
    print(f"  Total displacement around hexagon: {total_displacement}")
    print(f"  Is closed? {np.allclose(total_displacement, 0)}")
    
    # Check alternating sublattices
    sublattices = [s[2] for s in hex_sites]
    print(f"\n  Sublattice pattern: {' → '.join(sublattices)}")
    is_alternating = all(sublattices[i] != sublattices[(i+1)%6] for i in range(6))
    print(f"  Alternating A/B? {is_alternating}")


if __name__ == '__main__':
    verify_hexagon()
