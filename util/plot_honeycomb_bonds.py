#!/usr/bin/env python3
"""
Visualize honeycomb lattice bond structure to verify Kitaev bond assignments.

Bond types (from the C++ code):
- x-bond (type 0): A(i,j,k) → B(i,j-1,k)   [offset (0,-1,0)]
- y-bond (type 1): A(i,j,k) → B(i+1,j-1,k) [offset (1,-1,0)]  
- z-bond (type 2): A(i,j,k) → B(i,j,k)     [same unit cell]

Lattice vectors:
- a1 = (1, 0, 0)
- a2 = (0.5, sqrt(3)/2, 0)

Sublattice positions within unit cell:
- A: (0, 0, 0)
- B: (0, 1/sqrt(3), 0)

Hexagon sites (from C++ code):
- 0: A(i,j)      - central A site
- 1: B(i,j)      - z-bond neighbor (same unit cell)
- 2: A(i,j+1)    - next unit cell in j direction
- 3: B(i+1,j)    - B site in (i+1,j) unit cell
- 4: A(i+1,j)    - A site in (i+1,j) unit cell
- 5: B(i+1,j-1)  - B site in (i+1,j-1) unit cell
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection
import argparse

def build_honeycomb_lattice(dim1, dim2):
    """Build honeycomb lattice sites and bonds."""
    
    # Lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0)/2.0])
    
    # Sublattice positions
    pos_A = np.array([0.0, 0.0])
    pos_B = np.array([0.0, 1.0/np.sqrt(3.0)])
    
    sites = []
    sublattices = []
    site_indices = {}  # (i, j, sub) -> flat index
    
    # Build sites
    for i in range(dim1):
        for j in range(dim2):
            # Sublattice A
            idx = len(sites)
            pos = pos_A + i * a1 + j * a2
            sites.append(pos)
            sublattices.append('A')
            site_indices[(i, j, 'A')] = idx
            
            # Sublattice B
            idx = len(sites)
            pos = pos_B + i * a1 + j * a2
            sites.append(pos)
            sublattices.append('B')
            site_indices[(i, j, 'B')] = idx
    
    sites = np.array(sites)
    
    # Build bonds with types
    bonds = []  # (site_i, site_j, bond_type, bond_name)
    
    def flatten_index(i, j, sublattice):
        """Convert (i,j,sublattice) to flat index."""
        sub = 0 if sublattice == 'A' else 1
        return 2 * (i * dim2 + j) + sub
    
    def flatten_periodic(i, j, sublattice):
        """Periodic boundary conditions."""
        i_p = i % dim1
        j_p = j % dim2
        sub = 0 if sublattice == 'A' else 1
        return 2 * (i_p * dim2 + j_p) + sub
    
    for i in range(dim1):
        for j in range(dim2):
            site_A = flatten_index(i, j, 'A')
            site_B = flatten_index(i, j, 'B')
            
            # x-bond (type 0): A(i,j) → B(i,j-1)
            partner_x = flatten_periodic(i, j-1, 'B')
            bonds.append((site_A, partner_x, 0, 'x'))
            
            # y-bond (type 1): A(i,j) → B(i+1,j-1)
            partner_y = flatten_periodic(i+1, j-1, 'B')
            bonds.append((site_A, partner_y, 1, 'y'))
            
            # z-bond (type 2): A(i,j) → B(i,j) [same unit cell]
            bonds.append((site_A, site_B, 2, 'z'))
    
    # Build 2nd nearest neighbor bonds (same sublattice)
    j2_bonds = []
    j2_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    for i in range(dim1):
        for j in range(dim2):
            site_A = flatten_index(i, j, 'A')
            site_B = flatten_index(i, j, 'B')
            for di, dj in j2_offsets:
                partner_A = flatten_periodic(i+di, j+dj, 'A')
                partner_B = flatten_periodic(i+di, j+dj, 'B')
                if partner_A > site_A:
                    j2_bonds.append((site_A, partner_A))
                if partner_B > site_B:
                    j2_bonds.append((site_B, partner_B))
    
    # Build 3rd nearest neighbor bonds (opposite sublattice)
    j3_bonds = []
    # From A to B: offsets that give 3rd NN distance
    j3_A_to_B_offsets = [(1, -2), (-1, 0), (1, 0)]  # Need to verify these
    j3_B_to_A_offsets = [(-1, 2), (-1, 0), (1, 0)]
    for i in range(dim1):
        for j in range(dim2):
            site_A = flatten_index(i, j, 'A')
            site_B = flatten_index(i, j, 'B')
            for di, dj in j3_A_to_B_offsets:
                partner = flatten_periodic(i+di, j+dj, 'B')
                if partner > site_A:
                    j3_bonds.append((site_A, partner))
            for di, dj in j3_B_to_A_offsets:
                partner = flatten_periodic(i+di, j+dj, 'A')
                if partner > site_B:
                    j3_bonds.append((site_B, partner))
    
    # Build hexagons
    hexagons = []
    for i in range(dim1):
        for j in range(dim2):
            hex_sites = [
                flatten_index(i, j, 'A'),           # 0: A(i,j)
                flatten_index(i, j, 'B'),           # 1: B(i,j)
                flatten_periodic(i, j+1, 'A'),      # 2: A(i,j+1)
                flatten_periodic(i+1, j, 'B'),      # 3: B(i+1,j)
                flatten_periodic(i+1, j, 'A'),      # 4: A(i+1,j)
                flatten_periodic(i+1, j-1, 'B'),    # 5: B(i+1,j-1)
            ]
            hexagons.append(hex_sites)
    
    return sites, sublattices, bonds, j2_bonds, j3_bonds, hexagons, a1, a2


def plot_honeycomb_bonds(dim1=4, dim2=4, show_labels=True, output_file=None):
    """Plot honeycomb lattice with color-coded Kitaev bonds."""
    
    sites, sublattices, bonds, j2_bonds, j3_bonds, hexagons, a1, a2 = build_honeycomb_lattice(dim1, dim2)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Bond colors for Kitaev bond types
    bond_colors = {
        0: 'red',      # x-bond
        1: 'green',    # y-bond  
        2: 'blue'      # z-bond
    }
    bond_labels = {
        0: 'x-bond (S^x S^x)',
        1: 'y-bond (S^y S^y)',
        2: 'z-bond (S^z S^z)'
    }
    
    # Plot bonds
    plotted_labels = set()
    for site_i, site_j, bond_type, bond_name in bonds:
        pos_i = sites[site_i]
        pos_j = sites[site_j]
        
        # Handle periodic boundary - only draw if bond is short
        diff = pos_j - pos_i
        bond_length = np.linalg.norm(diff)
        
        # Skip long bonds (these are periodic wrap-arounds)
        if bond_length > 1.5:
            continue
        
        color = bond_colors[bond_type]
        label = bond_labels[bond_type] if bond_type not in plotted_labels else None
        
        ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                color=color, linewidth=2.5, label=label, alpha=0.8)
        
        if label:
            plotted_labels.add(bond_type)
    
    # Plot sites
    for idx, (pos, sub) in enumerate(zip(sites, sublattices)):
        if sub == 'A':
            ax.scatter(pos[0], pos[1], c='black', s=100, zorder=5, marker='o')
        else:
            ax.scatter(pos[0], pos[1], c='white', s=100, zorder=5, marker='o', 
                      edgecolors='black', linewidths=1.5)
        
        if show_labels and idx < 20:  # Only label first few sites
            ax.annotate(f'{idx}', pos, fontsize=7, ha='center', va='center',
                       color='gray' if sub == 'A' else 'black')
    
    # Draw lattice vectors
    origin = np.array([0, 0])
    ax.annotate('', xy=a1, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax.annotate('', xy=a2, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(a1[0]/2, a1[1]-0.15, r'$\mathbf{a}_1$', fontsize=12, color='purple')
    ax.text(a2[0]-0.15, a2[1]/2, r'$\mathbf{a}_2$', fontsize=12, color='orange')
    
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_title(f'Honeycomb Lattice with Kitaev Bond Types ({dim1}×{dim2})\n'
                 f'● = Sublattice A, ○ = Sublattice B', fontsize=13)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    # Add text box with bond definitions
    textstr = '\n'.join([
        'Bond definitions (from C++ code):',
        '• x-bond: A(i,j) → B(i,j-1)',
        '• y-bond: A(i,j) → B(i+1,j-1)', 
        '• z-bond: A(i,j) → B(i,j)',
        '',
        'Kitaev interaction on γ-bond:',
        r'$H_K = K \sum_{\langle ij \rangle_\gamma} S_i^\gamma S_j^\gamma$'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    
    plt.show()


def plot_bond_directions(dim1=3, dim2=3, output_file=None):
    """Plot a cleaner diagram showing bond directions from A sites."""
    
    sites, sublattices, bonds, j2_bonds, j3_bonds, hexagons, a1, a2 = build_honeycomb_lattice(dim1, dim2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Bond colors
    bond_colors = {0: 'red', 1: 'green', 2: 'blue'}
    
    # Find a central A site
    center_i, center_j = dim1 // 2, dim2 // 2
    center_A = 2 * (center_i * dim2 + center_j)
    
    # Plot all bonds, highlighting those from central A site
    for site_i, site_j, bond_type, bond_name in bonds:
        pos_i = sites[site_i]
        pos_j = sites[site_j]
        
        diff = pos_j - pos_i
        if np.linalg.norm(diff) > 1.5:
            continue
        
        color = bond_colors[bond_type]
        lw = 4 if site_i == center_A else 1.5
        alpha = 1.0 if site_i == center_A else 0.3
        
        ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                color=color, linewidth=lw, alpha=alpha)
        
        # Add arrow and label for central bonds
        if site_i == center_A:
            mid = (pos_i + pos_j) / 2
            ax.annotate(f'{bond_name}-bond', mid + np.array([0.05, 0.05]), 
                       fontsize=11, fontweight='bold', color=color)
    
    # Plot sites
    for idx, (pos, sub) in enumerate(zip(sites, sublattices)):
        if sub == 'A':
            size = 200 if idx == center_A else 80
            ax.scatter(pos[0], pos[1], c='black', s=size, zorder=5, marker='o')
        else:
            ax.scatter(pos[0], pos[1], c='white', s=80, zorder=5, marker='o', 
                      edgecolors='black', linewidths=1.5)
    
    # Label central site
    ax.annotate('A', sites[center_A] + np.array([-0.15, -0.15]), 
               fontsize=14, fontweight='bold')
    
    ax.set_aspect('equal')
    ax.set_title('Kitaev Bond Types from Central A Site\n'
                 'x-bond (red): A→B(j-1), y-bond (green): A→B(i+1,j-1), z-bond (blue): A→B(same cell)',
                 fontsize=11)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='x-bond (γ=0): $S^x_i S^x_j$'),
        Line2D([0], [0], color='green', lw=3, label='y-bond (γ=1): $S^y_i S^y_j$'),
        Line2D([0], [0], color='blue', lw=3, label='z-bond (γ=2): $S^z_i S^z_j$'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    
    plt.show()


def plot_all_interactions(dim1=4, dim2=4, output_file=None):
    """Plot NN bonds, 2nd NN, 3rd NN, and one hexagon."""
    
    sites, sublattices, bonds, j2_bonds, j3_bonds, hexagons, a1, a2 = build_honeycomb_lattice(dim1, dim2)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # --- Panel 1: NN bonds with Kitaev types ---
    ax = axes[0, 0]
    bond_colors = {0: 'red', 1: 'green', 2: 'blue'}
    
    for site_i, site_j, bond_type, bond_name in bonds:
        pos_i, pos_j = sites[site_i], sites[site_j]
        if np.linalg.norm(pos_j - pos_i) < 1.5:
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                    color=bond_colors[bond_type], linewidth=2, alpha=0.7)
    
    for idx, (pos, sub) in enumerate(zip(sites, sublattices)):
        color = 'black' if sub == 'A' else 'white'
        ax.scatter(pos[0], pos[1], c=color, s=60, zorder=5, 
                  edgecolors='black', linewidths=1)
    
    ax.set_aspect('equal')
    ax.set_title('1st NN: Kitaev bonds\nx=red, y=green, z=blue')
    
    # --- Panel 2: 2nd NN bonds ---
    ax = axes[0, 1]
    
    # Plot NN bonds faintly
    for site_i, site_j, bond_type, bond_name in bonds:
        pos_i, pos_j = sites[site_i], sites[site_j]
        if np.linalg.norm(pos_j - pos_i) < 1.5:
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                    color='lightgray', linewidth=1, alpha=0.5)
    
    # Plot 2nd NN bonds
    for site_i, site_j in j2_bonds:
        pos_i, pos_j = sites[site_i], sites[site_j]
        if np.linalg.norm(pos_j - pos_i) < 2.5:
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                    color='purple', linewidth=1.5, alpha=0.6)
    
    for idx, (pos, sub) in enumerate(zip(sites, sublattices)):
        color = 'black' if sub == 'A' else 'white'
        ax.scatter(pos[0], pos[1], c=color, s=60, zorder=5, 
                  edgecolors='black', linewidths=1)
    
    ax.set_aspect('equal')
    ax.set_title('2nd NN bonds (purple)\nSame sublattice: A↔A, B↔B')
    
    # --- Panel 3: 3rd NN bonds ---
    ax = axes[1, 0]
    
    # Plot NN bonds faintly
    for site_i, site_j, bond_type, bond_name in bonds:
        pos_i, pos_j = sites[site_i], sites[site_j]
        if np.linalg.norm(pos_j - pos_i) < 1.5:
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                    color='lightgray', linewidth=1, alpha=0.5)
    
    # Plot 3rd NN bonds
    for site_i, site_j in j3_bonds:
        pos_i, pos_j = sites[site_i], sites[site_j]
        if np.linalg.norm(pos_j - pos_i) < 3.0:
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                    color='orange', linewidth=1.5, alpha=0.6)
    
    for idx, (pos, sub) in enumerate(zip(sites, sublattices)):
        color = 'black' if sub == 'A' else 'white'
        ax.scatter(pos[0], pos[1], c=color, s=60, zorder=5, 
                  edgecolors='black', linewidths=1)
    
    ax.set_aspect('equal')
    ax.set_title('3rd NN bonds (orange)\nOpposite sublattice: A↔B')
    
    # --- Panel 4: Hexagon (ring exchange) ---
    ax = axes[1, 1]
    
    # Plot NN bonds
    for site_i, site_j, bond_type, bond_name in bonds:
        pos_i, pos_j = sites[site_i], sites[site_j]
        if np.linalg.norm(pos_j - pos_i) < 1.5:
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                    color=bond_colors[bond_type], linewidth=1.5, alpha=0.4)
    
    # Highlight one hexagon
    hex_idx = dim2 + 1  # Pick a central hexagon
    if hex_idx < len(hexagons):
        hex_sites = hexagons[hex_idx]
        hex_positions = [sites[s] for s in hex_sites]
        
        # Check if hexagon is valid (not wrapping around)
        center = np.mean(hex_positions, axis=0)
        max_dist = max(np.linalg.norm(p - center) for p in hex_positions)
        
        if max_dist < 1.5:  # Valid hexagon
            # Draw filled polygon
            polygon = Polygon(hex_positions, closed=True, 
                            facecolor='yellow', edgecolor='black', 
                            alpha=0.3, linewidth=2, zorder=2)
            ax.add_patch(polygon)
            
            # Draw hexagon edges with arrows showing order
            for i in range(6):
                p1 = hex_positions[i]
                p2 = hex_positions[(i+1) % 6]
                ax.annotate('', xy=p2, xytext=p1,
                           arrowprops=dict(arrowstyle='->', color='black', lw=2))
            
            # Label the sites
            for i, (site, pos) in enumerate(zip(hex_sites, hex_positions)):
                sub = sublattices[site]
                ax.scatter(pos[0], pos[1], c='yellow', s=150, zorder=10,
                          edgecolors='black', linewidths=2)
                ax.annotate(f'{i}', pos, fontsize=12, ha='center', va='center',
                           fontweight='bold', zorder=11)
    
    # Plot all sites
    for idx, (pos, sub) in enumerate(zip(sites, sublattices)):
        color = 'black' if sub == 'A' else 'white'
        ax.scatter(pos[0], pos[1], c=color, s=40, zorder=5, 
                  edgecolors='black', linewidths=0.5, alpha=0.5)
    
    ax.set_aspect('equal')
    ax.set_title('Hexagon for Ring Exchange J7\n'
                 '0:A(i,j) → 1:B(i,j) → 2:A(i,j+1) →\n'
                 '3:B(i+1,j) → 4:A(i+1,j) → 5:B(i+1,j-1)')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    
    plt.show()


def plot_single_hexagon(dim1=4, dim2=4, output_file=None):
    """Plot just one hexagon with detailed labeling."""
    
    sites, sublattices, bonds, j2_bonds, j3_bonds, hexagons, a1, a2 = build_honeycomb_lattice(dim1, dim2)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    bond_colors = {0: 'red', 1: 'green', 2: 'blue'}
    
    # Plot all NN bonds
    for site_i, site_j, bond_type, bond_name in bonds:
        pos_i, pos_j = sites[site_i], sites[site_j]
        if np.linalg.norm(pos_j - pos_i) < 1.5:
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                    color=bond_colors[bond_type], linewidth=2, alpha=0.3)
    
    # Find and highlight hexagon at (1,1)
    hex_idx = dim2 + 1
    hex_sites = hexagons[hex_idx]
    hex_positions = [sites[s] for s in hex_sites]
    
    # Labels for hexagon sites
    hex_labels = [
        '0: A(i,j)',
        '1: B(i,j)',
        '2: A(i,j+1)',
        '3: B(i+1,j)',
        '4: A(i+1,j)',
        '5: B(i+1,j-1)'
    ]
    
    # Draw the hexagon path
    for i in range(6):
        p1 = hex_positions[i]
        p2 = hex_positions[(i+1) % 6]
        
        # Determine bond type for this edge
        s1, s2 = hex_sites[i], hex_sites[(i+1) % 6]
        edge_color = 'black'
        edge_label = ''
        for si, sj, bt, bn in bonds:
            if (si == s1 and sj == s2) or (si == s2 and sj == s1):
                edge_color = bond_colors[bt]
                edge_label = f'{bn}-bond'
                break
        
        mid = (p1 + p2) / 2
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge_color, linewidth=4, zorder=3)
        
        # Add edge label
        offset = (p2 - p1)
        offset = np.array([-offset[1], offset[0]]) * 0.15
        ax.annotate(edge_label, mid + offset, fontsize=9, ha='center', color=edge_color)
    
    # Draw and label hexagon vertices
    for i, (site, pos, label) in enumerate(zip(hex_sites, hex_positions, hex_labels)):
        sub = sublattices[site]
        color = 'lightblue' if sub == 'A' else 'lightyellow'
        ax.scatter(pos[0], pos[1], c=color, s=400, zorder=10,
                  edgecolors='black', linewidths=2)
        ax.annotate(f'{i}', pos, fontsize=14, ha='center', va='center',
                   fontweight='bold', zorder=11)
        
        # Add detailed label
        offset_dir = pos - np.mean(hex_positions, axis=0)
        offset_dir = offset_dir / np.linalg.norm(offset_dir) * 0.4
        ax.annotate(label, pos + offset_dir, fontsize=10, ha='center')
    
    # Plot other sites dimly
    for idx, (pos, sub) in enumerate(zip(sites, sublattices)):
        if idx not in hex_sites:
            color = 'gray' if sub == 'A' else 'lightgray'
            ax.scatter(pos[0], pos[1], c=color, s=50, zorder=2, 
                      edgecolors='gray', linewidths=0.5, alpha=0.3)
    
    ax.set_aspect('equal')
    ax.set_title('Ring Exchange Hexagon\n'
                 'Path: 0→1→2→3→4→5→0\n'
                 'Blue=A sublattice, Yellow=B sublattice', fontsize=12)
    
    # Add legend for bond types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='x-bond'),
        Line2D([0], [0], color='green', lw=3, label='y-bond'),
        Line2D([0], [0], color='blue', lw=3, label='z-bond'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    
    plt.show()


def verify_c3_symmetry():
    """Verify that the three bond types are related by C3 rotation."""
    
    # Lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0)/2.0])
    
    # Bond vectors (from A site at origin)
    # z-bond: A(0,0) → B(0,0) at (0, 1/√3)
    z_bond = np.array([0.0, 1.0/np.sqrt(3.0)])
    
    # x-bond: A(0,0) → B(0,-1) at -a2 + (0, 1/√3)
    x_bond = -a2 + np.array([0.0, 1.0/np.sqrt(3.0)])
    
    # y-bond: A(0,0) → B(1,-1) at a1 - a2 + (0, 1/√3)
    y_bond = a1 - a2 + np.array([0.0, 1.0/np.sqrt(3.0)])
    
    print("Bond vectors from A site at origin:")
    print(f"  z-bond: {z_bond} (length = {np.linalg.norm(z_bond):.4f})")
    print(f"  x-bond: {x_bond} (length = {np.linalg.norm(x_bond):.4f})")
    print(f"  y-bond: {y_bond} (length = {np.linalg.norm(y_bond):.4f})")
    
    # C3 rotation matrix (120 degrees counterclockwise)
    theta = 2 * np.pi / 3
    C3 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    
    print("\nC3 rotation check:")
    print(f"  C3 @ z_bond = {C3 @ z_bond}")
    print(f"  C3 @ x_bond = {C3 @ x_bond}")
    print(f"  C3 @ y_bond = {C3 @ y_bond}")
    
    # Check if C3 permutes the bonds: z → x → y → z
    print("\nExpected C3 permutation: z → ? → ? → z")
    z_rotated = C3 @ z_bond
    print(f"  C3(z) ≈ x? diff = {np.linalg.norm(z_rotated - x_bond):.6f}")
    print(f"  C3(z) ≈ y? diff = {np.linalg.norm(z_rotated - y_bond):.6f}")
    
    x_rotated = C3 @ x_bond
    print(f"  C3(x) ≈ y? diff = {np.linalg.norm(x_rotated - y_bond):.6f}")
    print(f"  C3(x) ≈ z? diff = {np.linalg.norm(x_rotated - z_bond):.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot honeycomb lattice bonds')
    parser.add_argument('--dim', type=int, nargs=2, default=[4, 4],
                       help='Lattice dimensions (dim1 dim2)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file for figure')
    parser.add_argument('--simple', action='store_true',
                       help='Show simplified bond direction diagram')
    parser.add_argument('--verify', action='store_true',
                       help='Verify C3 symmetry of bonds')
    parser.add_argument('--all', action='store_true',
                       help='Show all interactions (NN, 2nd NN, 3rd NN, hexagon)')
    parser.add_argument('--hexagon', action='store_true',
                       help='Show detailed hexagon plot')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_c3_symmetry()
    elif args.all:
        plot_all_interactions(dim1=args.dim[0], dim2=args.dim[1], output_file=args.output)
    elif args.hexagon:
        plot_single_hexagon(dim1=args.dim[0], dim2=args.dim[1], output_file=args.output)
    elif args.simple:
        plot_bond_directions(dim1=args.dim[0], dim2=args.dim[1], output_file=args.output)
    else:
        plot_honeycomb_bonds(dim1=args.dim[0], dim2=args.dim[1], output_file=args.output)
