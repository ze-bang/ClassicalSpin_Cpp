#!/usr/bin/env python3
"""
Visualize Tm-Fe coupling in TmFeO3 mixed bilinear interactions
Shows which Tm and Fe sites are coupled as nearest neighbors
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Mixed bilinear interactions from the code
# Format: (Tm_site, Fe_site, unit_cell_offset)
chi_interactions = [
    (1, 0, [0, 0, 0]),
    (1, 3, [0, 0, 0]),
    (1, 1, [0, 1, 0]),
    (1, 2, [0, 1, 0]),
    
    (0, 0, [0, 0, 0]),
    (0, 3, [0, 0, 1]),
    (0, 1, [0, 1, 0]),
    (0, 2, [0, 1, 1]),
    
    (2, 0, [0, 0, 0]),
    (2, 3, [0, 0, 1]),
    (2, 1, [0, 0, 0]),
    (2, 2, [0, 0, 1]),
    
    (3, 0, [1, 0, 0]),
    (3, 3, [1, 0, 0]),
    (3, 1, [0, 0, 0]),
    (3, 2, [0, 0, 0])
]

chi_inv_interactions = [
    (1, 2, [0, 0, 0]),
    (1, 3, [1, 0, 0]),
    (1, 1, [0, 0, 0]),
    (1, 0, [1, 0, 0]),
    
    (0, 2, [-1, 1, 1]),
    (0, 3, [0, 1, 1]),
    (0, 1, [-1, 1, 0]),
    (0, 0, [0, 1, 0]),
    
    (2, 2, [0, 1, 1]),
    (2, 3, [1, 0, 1]),
    (2, 1, [0, 1, 0]),
    (2, 0, [1, 0, 0]),
    
    (3, 2, [1, 0, 0]),
    (3, 3, [1, -1, 0]),
    (3, 1, [1, 0, 0]),
    (3, 0, [1, -1, 0])
]

print("=" * 80)
print("TmFeO3 Tm-Fe Mixed Bilinear Coupling Visualization")
print("=" * 80)
print()

print("Fe (SU2) Site Positions (fractional coordinates):")
print("-" * 60)
for i, pos in enumerate(Fe_positions):
    print(f"  Fe site {i}: ({pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f})")
print()

print("Tm (SU3) Site Positions (fractional coordinates):")
print("-" * 60)
for i, pos in enumerate(Tm_positions):
    print(f"  Tm site {i}: ({pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f})")
print()

print("=" * 80)
print("CHI Interactions (coupling between Tm and Fe):")
print("=" * 80)
for tm_site, fe_site, offset in chi_interactions:
    tm_pos = Tm_positions[tm_site]
    fe_pos = Fe_positions[fe_site] + np.array(offset)
    distance = np.linalg.norm(fe_pos - tm_pos)
    print(f"Tm{tm_site} ({tm_pos[0]:.3f}, {tm_pos[1]:.3f}, {tm_pos[2]:.3f}) <--> "
          f"Fe{fe_site} ({fe_pos[0]:.3f}, {fe_pos[1]:.3f}, {fe_pos[2]:.3f})  "
          f"| offset: {offset}  | distance: {distance:.4f}")
print()

print("=" * 80)
print("CHI_INV Interactions (inverse coupling between Tm and Fe):")
print("=" * 80)
for tm_site, fe_site, offset in chi_inv_interactions:
    tm_pos = Tm_positions[tm_site]
    fe_pos = Fe_positions[fe_site] + np.array(offset)
    distance = np.linalg.norm(fe_pos - tm_pos)
    print(f"Tm{tm_site} ({tm_pos[0]:.3f}, {tm_pos[1]:.3f}, {tm_pos[2]:.3f}) <--> "
          f"Fe{fe_site} ({fe_pos[0]:.3f}, {fe_pos[1]:.3f}, {fe_pos[2]:.3f})  "
          f"| offset: {offset}  | distance: {distance:.4f}")
print()

# Calculate all distances and find nearest neighbors
print("=" * 80)
print("Distance Analysis:")
print("=" * 80)
all_distances = []
for tm_idx in range(4):
    for fe_idx in range(4):
        for offset in [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], 
                       [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
                       [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
                       [0, -1, -1], [0, -1, 0], [0, -1, 1],
                       [0, 0, -1], [0, 0, 0], [0, 0, 1],
                       [0, 1, -1], [0, 1, 0], [0, 1, 1],
                       [1, -1, -1], [1, -1, 0], [1, -1, 1],
                       [1, 0, -1], [1, 0, 0], [1, 0, 1],
                       [1, 1, -1], [1, 1, 0], [1, 1, 1]]:
            tm_pos = Tm_positions[tm_idx]
            fe_pos = Fe_positions[fe_idx] + np.array(offset)
            distance = np.linalg.norm(fe_pos - tm_pos)
            all_distances.append((distance, tm_idx, fe_idx, offset))

# Sort by distance
all_distances.sort(key=lambda x: x[0])

print(f"\nMinimum Tm-Fe distance: {all_distances[0][0]:.4f}")
print(f"Maximum Tm-Fe distance (within ±1 unit cell): {all_distances[-1][0]:.4f}")

# Show the 20 nearest Tm-Fe pairs
print(f"\n20 Nearest Tm-Fe pairs:")
print("-" * 60)
for i, (dist, tm_idx, fe_idx, offset) in enumerate(all_distances[:20]):
    tm_pos = Tm_positions[tm_idx]
    fe_pos = Fe_positions[fe_idx] + np.array(offset)
    in_chi = (tm_idx, fe_idx, offset) in chi_interactions
    in_chi_inv = (tm_idx, fe_idx, offset) in chi_inv_interactions
    coupled = "✓ CHI" if in_chi else ("✓ CHI_INV" if in_chi_inv else "")
    print(f"{i+1:2d}. Tm{tm_idx} - Fe{fe_idx} offset{offset}: {dist:.4f}  {coupled}")

print()
print("=" * 80)
print("Verification: Are all coupled pairs nearest neighbors?")
print("=" * 80)

# Check if all chi interactions are among the nearest neighbors
coupled_distances = []
for tm_site, fe_site, offset in chi_interactions + chi_inv_interactions:
    tm_pos = Tm_positions[tm_site]
    fe_pos = Fe_positions[fe_site] + np.array(offset)
    distance = np.linalg.norm(fe_pos - tm_pos)
    coupled_distances.append(distance)

print(f"Number of CHI interactions: {len(chi_interactions)}")
print(f"Number of CHI_INV interactions: {len(chi_inv_interactions)}")
print(f"Total coupled pairs: {len(chi_interactions) + len(chi_inv_interactions)}")
print(f"Distance range of coupled pairs: {min(coupled_distances):.4f} - {max(coupled_distances):.4f}")
print(f"Mean distance of coupled pairs: {np.mean(coupled_distances):.4f}")

# Count unique Tm-Fe pairs (considering all unit cell offsets)
unique_pairs = set()
for tm_site, fe_site, offset in chi_interactions + chi_inv_interactions:
    unique_pairs.add((tm_site, fe_site, tuple(offset)))

print(f"\nTotal unique Tm-Fe coupled pairs (with offsets): {len(unique_pairs)}")

# Check if coupled pairs are nearest neighbors
threshold = all_distances[0][0] + 0.1  # Within 10% of minimum distance
nearest_neighbor_count = sum(1 for d in coupled_distances if d < threshold)
print(f"\nCoupled pairs within nearest neighbor threshold ({threshold:.4f}): {nearest_neighbor_count}/{len(coupled_distances)}")

if nearest_neighbor_count == len(coupled_distances):
    print("\n✓ SUCCESS: All coupled pairs ARE nearest neighbors!")
else:
    print(f"\n⚠ WARNING: {len(coupled_distances) - nearest_neighbor_count} coupled pairs are NOT nearest neighbors!")

print()
print("=" * 80)
