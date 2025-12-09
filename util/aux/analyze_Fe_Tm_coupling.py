#!/usr/bin/env python3
"""
Analyze Fe-Tm coupling from the Fe perspective
Show the 8 nearest Tm neighbors for each Fe site
Group by inversion relations
"""

import numpy as np
from collections import defaultdict

# Fe (SU2) site positions in fractional coordinates
Fe_positions = np.array([
    [0.0, 0.5, 0.5],   # Fe site 0
    [0.5, 0.0, 0.5],   # Fe site 1
    [0.5, 0.0, 0.0],   # Fe site 2
    [0.0, 0.5, 0.0]    # Fe site 3
])

# Tm (SU3) site positions in fractional coordinates
Tm_positions = np.array([
    [0.02111, 0.92839, 0.75],   # Tm site 0
    [0.52111, 0.57161, 0.25],   # Tm site 1
    [0.47889, 0.42839, 0.75],   # Tm site 2
    [0.97889, 0.07161, 0.25]    # Tm site 3
])

print("=" * 90)
print("Fe-Tm Nearest Neighbor Analysis from Fe Perspective")
print("=" * 90)
print()

# For each Fe site, find all Tm neighbors within a reasonable range
for fe_idx in range(4):
    fe_pos = Fe_positions[fe_idx]
    
    print(f"\n{'=' * 90}")
    print(f"Fe site {fe_idx} at position ({fe_pos[0]:.5f}, {fe_pos[1]:.5f}, {fe_pos[2]:.5f})")
    print(f"{'=' * 90}")
    
    # Calculate distances to all Tm sites (considering nearby unit cells)
    neighbors = []
    for tm_idx in range(4):
        for offset in [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], 
                       [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
                       [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
                       [0, -1, -1], [0, -1, 0], [0, -1, 1],
                       [0, 0, -1], [0, 0, 0], [0, 0, 1],
                       [0, 1, -1], [0, 1, 0], [0, 1, 1],
                       [1, -1, -1], [1, -1, 0], [1, -1, 1],
                       [1, 0, -1], [1, 0, 0], [1, 0, 1],
                       [1, 1, -1], [1, 1, 0], [1, 1, 1]]:
            tm_pos = Tm_positions[tm_idx] + np.array(offset)
            displacement = tm_pos - fe_pos
            distance = np.linalg.norm(displacement)
            neighbors.append({
                'tm_idx': tm_idx,
                'offset': offset,
                'tm_pos': tm_pos,
                'displacement': displacement,
                'distance': distance
            })
    
    # Sort by distance and get the 8 nearest
    neighbors.sort(key=lambda x: x['distance'])
    nearest_8 = neighbors[:8]
    
    print(f"\n8 Nearest Tm neighbors:")
    print("-" * 90)
    for i, n in enumerate(nearest_8):
        print(f"{i+1}. Tm{n['tm_idx']} + offset{n['offset']} = "
              f"({n['tm_pos'][0]:.3f}, {n['tm_pos'][1]:.3f}, {n['tm_pos'][2]:.3f})  "
              f"| d = {n['distance']:.4f}  "
              f"| Δ = ({n['displacement'][0]:+.3f}, {n['displacement'][1]:+.3f}, {n['displacement'][2]:+.3f})")
    
    # Group by inversion symmetry
    print(f"\n{'─' * 90}")
    print("Grouped by Inversion Relations:")
    print(f"{'─' * 90}")
    
    # Find inversion pairs
    paired = set()
    inversion_pairs = []
    
    for i, n1 in enumerate(nearest_8):
        if i in paired:
            continue
        
        # Look for inversion partner (displacement should be opposite)
        for j, n2 in enumerate(nearest_8):
            if i == j or j in paired:
                continue
            
            # Check if displacements are approximately opposite
            sum_displacement = n1['displacement'] + n2['displacement']
            if np.linalg.norm(sum_displacement) < 0.01:  # They are inversion related
                inversion_pairs.append((n1, n2, i+1, j+1))
                paired.add(i)
                paired.add(j)
                break
    
    # Print inversion pairs
    for idx, (n1, n2, pos1, pos2) in enumerate(inversion_pairs):
        print(f"\nInversion Pair {idx+1}:")
        print(f"  {pos1}. Tm{n1['tm_idx']} + {n1['offset']}  →  Δ = ({n1['displacement'][0]:+.3f}, {n1['displacement'][1]:+.3f}, {n1['displacement'][2]:+.3f})")
        print(f"  {pos2}. Tm{n2['tm_idx']} + {n2['offset']}  →  Δ = ({n2['displacement'][0]:+.3f}, {n2['displacement'][1]:+.3f}, {n2['displacement'][2]:+.3f})")
        print(f"  Distance: {n1['distance']:.4f}")
        print(f"  Sum of displacements: ({sum(n1['displacement']+n2['displacement']):.6f})")
    
    # Any unpaired neighbors?
    unpaired = [n for i, n in enumerate(nearest_8) if i not in paired]
    if unpaired:
        print(f"\nUnpaired neighbors (no inversion partner):")
        for n in unpaired:
            print(f"  Tm{n['tm_idx']} + {n['offset']}  →  Δ = ({n['displacement'][0]:+.3f}, {n['displacement'][1]:+.3f}, {n['displacement'][2]:+.3f})")

print()
print("=" * 90)
print("Summary: Suggested coupling format for code")
print("=" * 90)
print()

# Generate coupling suggestions for each Fe site
for fe_idx in range(4):
    fe_pos = Fe_positions[fe_idx]
    
    # Calculate distances to all Tm sites
    neighbors = []
    for tm_idx in range(4):
        for offset in [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], 
                       [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
                       [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
                       [0, -1, -1], [0, -1, 0], [0, -1, 1],
                       [0, 0, -1], [0, 0, 0], [0, 0, 1],
                       [0, 1, -1], [0, 1, 0], [0, 1, 1],
                       [1, -1, -1], [1, -1, 0], [1, -1, 1],
                       [1, 0, -1], [1, 0, 0], [1, 0, 1],
                       [1, 1, -1], [1, 1, 0], [1, 1, 1]]:
            tm_pos = Tm_positions[tm_idx] + np.array(offset)
            displacement = tm_pos - fe_pos
            distance = np.linalg.norm(displacement)
            neighbors.append({
                'tm_idx': tm_idx,
                'offset': offset,
                'displacement': displacement,
                'distance': distance
            })
    
    neighbors.sort(key=lambda x: x['distance'])
    nearest_8 = neighbors[:8]
    
    print(f"\n// Fe site {fe_idx} - 8 nearest Tm neighbors")
    print(f"// Fe position: ({fe_pos[0]:.5f}, {fe_pos[1]:.5f}, {fe_pos[2]:.5f})")
    
    # Find and print inversion pairs
    paired = set()
    pair_num = 1
    
    for i, n1 in enumerate(nearest_8):
        if i in paired:
            continue
        
        for j, n2 in enumerate(nearest_8):
            if i == j or j in paired:
                continue
            
            sum_displacement = n1['displacement'] + n2['displacement']
            if np.linalg.norm(sum_displacement) < 0.01:
                print(f"// Inversion pair {pair_num} (distance: {n1['distance']:.4f}):")
                print(f"TFO.set_mix_bilinear_interaction(chi, {n1['tm_idx']}, {fe_idx}, {{{n1['offset'][0]}, {n1['offset'][1]}, {n1['offset'][2]}}});")
                print(f"TFO.set_mix_bilinear_interaction(chi_inv, {n2['tm_idx']}, {fe_idx}, {{{n2['offset'][0]}, {n2['offset'][1]}, {n2['offset'][2]}}});")
                paired.add(i)
                paired.add(j)
                pair_num += 1
                break

print()
print("=" * 90)
print("\nNote: chi and chi_inv should be related by inversion symmetry")
print("For each Fe site, there are 4 inversion pairs = 8 nearest Tm neighbors")
print("=" * 90)
