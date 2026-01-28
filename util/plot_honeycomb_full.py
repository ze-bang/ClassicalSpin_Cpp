#!/usr/bin/env python3
"""
Comprehensive visualization of honeycomb lattice structure:
- NN bonds (x, y, z Kitaev types)
- 2nd NN bonds (within same sublattice)
- 3rd NN bonds (between sublattices)
- Hexagonal plaquettes for ring exchange

Verifies the lattice construction matches the C++ code exactly.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.collections import LineCollection
import argparse

def build_honeycomb_lattice(dim1, dim2):
    """Build honeycomb lattice matching C++ implementation exactly."""
    
    # Lattice vectors (from C++ code)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3.0)/2.0])
    
    # Sublattice positions (from C++ code)
    pos_A = np.array([0.0, 0.0])
    pos_B = np.array([0.0, 1.0/np.sqrt(3.0)])
    
    sites = []
    sublattices = []
    unit_cell_indices = []  # (i, j, sublattice)
    
    # Build sites - matching C++ flatten_index order
    for i in range(dim1):
        for j in range(dim2):
            # Sublattice A (even index in pair)
            pos = pos_A + i * a1 + j * a2
            sites.append(pos)
            sublattices.append('A')
            unit_cell_indices.append((i, j, 0))
            
            # Sublattice B (odd index in pair)
            pos = pos_B + i * a1 + j * a2
            sites.append(pos)
            sublattices.append('B')
            unit_cell_indices.append((i, j, 1))
    
    sites = np.array(sites)
    
    def flatten_index(i, j, sublattice):
        """Convert (i,j,sublattice) to flat index - matches C++ exactly."""
        sub = 0 if sublattice == 'A' else 1
        return 2 * (i * dim2 + j) + sub
    
    def flatten_periodic(i, j, sublattice):
        """Periodic boundary conditions."""
        i_p = i % dim1
        j_p = j % dim2
        sub = 0 if sublattice == 'A' else 1
        return 2 * (i_p * dim2 + j_p) + sub
    
    # ============================================
    # NN BONDS (from C++ build_honeycomb)
    # ============================================
    # Bond assignment from C++ code:
    #   - z-bond (type 2): A(i,j,k) → B(i,j,k)     [same unit cell]
    #   - x-bond (type 0): A(i,j,k) → B(i,j-1,k)   [offset (0,-1,0)]
    #   - y-bond (type 1): A(i,j,k) → B(i+1,j-1,k) [offset (1,-1,0)]
    
    nn_bonds = []  # (site_i, site_j, bond_type, label)
    
    for i in range(dim1):
        for j in range(dim2):
            site_A = flatten_index(i, j, 'A')
            site_B = flatten_index(i, j, 'B')
            
            # x-bond (type 0): A(i,j) → B(i,j-1)
            partner_x = flatten_periodic(i, j-1, 'B')
            nn_bonds.append((site_A, partner_x, 0, 'x'))
            
            # y-bond (type 1): A(i,j) → B(i+1,j-1)
            partner_y = flatten_periodic(i+1, j-1, 'B')
            nn_bonds.append((site_A, partner_y, 1, 'y'))
            
            # z-bond (type 2): A(i,j) → B(i,j) [same unit cell]
            nn_bonds.append((site_A, site_B, 2, 'z'))
    
    # ============================================
    # 2ND NN BONDS (from C++ code)
    # ============================================
    # 2nd NN connect same sublattice (A↔A or B↔B)
    # Offsets: (±1, 0), (0, ±1), (±1, ∓1) in lattice coordinates
    
    j2_bonds = []  # (site_i, site_j, sublattice)
    j2_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    for i in range(dim1):
        for j in range(dim2):
            site_A = flatten_index(i, j, 'A')
            site_B = flatten_index(i, j, 'B')
            
            # 2nd NN for sublattice A
            for di, dj in j2_offsets:
                partner = flatten_periodic(i+di, j+dj, 'A')
                if partner > site_A:  # Avoid double counting
                    j2_bonds.append((site_A, partner, 'A'))
            
            # 2nd NN for sublattice B
            for di, dj in j2_offsets:
                partner = flatten_periodic(i+di, j+dj, 'B')
                if partner > site_B:
                    j2_bonds.append((site_B, partner, 'B'))
    
    # ============================================
    # 3RD NN BONDS (from C++ code)
    # ============================================
    # 3rd NN connect OPPOSITE sublattices (A↔B)
    # From C++ code:
    #   A → B offsets: {1, -2}, {-1, 0}, {1, 0}
    #   B → A offsets: {-1, 2}, {-1, 0}, {1, 0}
    
    j3_bonds = []  # (site_i, site_j)
    j3_A_to_B_offsets = [(1, -2), (-1, 0), (1, 0)]
    j3_B_to_A_offsets = [(-1, 2), (-1, 0), (1, 0)]
    
    for i in range(dim1):
        for j in range(dim2):
            site_A = flatten_index(i, j, 'A')
            site_B = flatten_index(i, j, 'B')
            
            # 3rd NN from sublattice A to B
            for di, dj in j3_A_to_B_offsets:
                partner = flatten_periodic(i+di, j+dj, 'B')
                if partner > site_A:
                    j3_bonds.append((site_A, partner))
            
            # 3rd NN from sublattice B to A
            for di, dj in j3_B_to_A_offsets:
                partner = flatten_periodic(i+di, j+dj, 'A')
                if partner > site_B:
                    j3_bonds.append((site_B, partner))
    
    # ============================================
    # HEXAGONAL PLAQUETTES (from C++ code)
    # ============================================
    # From C++ build_honeycomb (actual code, not comments):
    #   0: A(i,j,k)      - central A site
    #   1: B(i,j,k)      - z-bond neighbor (same unit cell)
    #   2: A(i,j+1,k)    - next unit cell in j direction
    #   3: B(i+1,j,k)    - B site in (i+1,j,k) unit cell
    #   4: A(i+1,j,k)    - A site in (i+1,j,k) unit cell
    #   5: B(i,j-1,k)    - B site in (i,j-1,k) unit cell (x-bond neighbor)
    
    hexagons = []
    
    for i in range(dim1):
        for j in range(dim2):
            hex_sites = [
                flatten_periodic(i, j, 'A'),      # 0
                flatten_periodic(i, j, 'B'),      # 1
                flatten_periodic(i, j+1, 'A'),    # 2
                flatten_periodic(i+1, j, 'B'),    # 3
                flatten_periodic(i+1, j, 'A'),    # 4
                flatten_periodic(i, j-1, 'B'),    # 5
            ]
            hexagons.append(hex_sites)
    
    return {
        'sites': sites,
        'sublattices': sublattices,
        'unit_cell_indices': unit_cell_indices,
        'nn_bonds': nn_bonds,
        'j2_bonds': j2_bonds,
        'j3_bonds': j3_bonds,
        'hexagons': hexagons,
        'a1': a1,
        'a2': a2,
        'dim1': dim1,
        'dim2': dim2
    }


def plot_full_lattice(dim1=4, dim2=4, show_nn=True, show_j2=False, show_j3=False, 
                      show_hex=False, highlight_hex_idx=0, output_file=None):
    """Plot honeycomb lattice with all connectivity."""
    
    lattice = build_honeycomb_lattice(dim1, dim2)
    sites = lattice['sites']
    sublattices = lattice['sublattices']
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Colors
    nn_colors = {0: 'red', 1: 'green', 2: 'blue'}  # x, y, z bonds
    j2_color = 'orange'
    j3_color = 'purple'
    hex_color = 'cyan'
    
    def is_short_bond(pos_i, pos_j, max_len=2.0):
        """Filter out periodic wrap-around bonds."""
        return np.linalg.norm(pos_j - pos_i) < max_len
    
    # ============================================
    # Plot J2 bonds (2nd NN)
    # ============================================
    if show_j2:
        for site_i, site_j, sub in lattice['j2_bonds']:
            pos_i, pos_j = sites[site_i], sites[site_j]
            if is_short_bond(pos_i, pos_j, 1.8):
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                       color=j2_color, linewidth=1.0, alpha=0.4, linestyle='--')
    
    # ============================================
    # Plot J3 bonds (3rd NN)
    # ============================================
    if show_j3:
        for site_i, site_j in lattice['j3_bonds']:
            pos_i, pos_j = sites[site_i], sites[site_j]
            if is_short_bond(pos_i, pos_j, 2.5):
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                       color=j3_color, linewidth=1.0, alpha=0.4, linestyle=':')
    
    # ============================================
    # Highlight hexagon
    # ============================================
    if show_hex and len(lattice['hexagons']) > highlight_hex_idx:
        hex_sites = lattice['hexagons'][highlight_hex_idx]
        hex_positions = [sites[s] for s in hex_sites]
        
        # Check if hexagon is not wrapped around boundary
        center = np.mean(hex_positions, axis=0)
        max_dist = max(np.linalg.norm(p - center) for p in hex_positions)
        
        if max_dist < 1.5:  # Only draw if hexagon is compact
            # Draw hexagon polygon
            polygon = Polygon(hex_positions, fill=True, facecolor=hex_color, 
                            edgecolor='black', alpha=0.3, linewidth=2)
            ax.add_patch(polygon)
            
            # Label hexagon vertices
            for idx, (site, pos) in enumerate(zip(hex_sites, hex_positions)):
                ax.annotate(f'{idx}', pos + np.array([0.05, 0.05]), 
                           fontsize=10, fontweight='bold', color='black',
                           bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.8))
    
    # ============================================
    # Plot NN bonds (on top)
    # ============================================
    if show_nn:
        plotted_labels = set()
        bond_labels = {0: 'x-bond (K·S^x S^x)', 1: 'y-bond (K·S^y S^y)', 2: 'z-bond (K·S^z S^z)'}
        
        for site_i, site_j, bond_type, bond_name in lattice['nn_bonds']:
            pos_i, pos_j = sites[site_i], sites[site_j]
            
            if not is_short_bond(pos_i, pos_j, 1.2):
                continue
            
            color = nn_colors[bond_type]
            label = bond_labels[bond_type] if bond_type not in plotted_labels else None
            
            ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                   color=color, linewidth=2.5, label=label, alpha=0.9)
            
            if label:
                plotted_labels.add(bond_type)
    
    # ============================================
    # Plot sites
    # ============================================
    for idx, (pos, sub) in enumerate(zip(sites, sublattices)):
        if sub == 'A':
            ax.scatter(pos[0], pos[1], c='black', s=80, zorder=5, marker='o')
        else:
            ax.scatter(pos[0], pos[1], c='white', s=80, zorder=5, marker='o', 
                      edgecolors='black', linewidths=1.5)
    
    # ============================================
    # Legend and formatting
    # ============================================
    legend_elements = []
    
    if show_nn:
        from matplotlib.lines import Line2D
        legend_elements.extend([
            Line2D([0], [0], color='red', lw=2.5, label='x-bond: K·S^x·S^x'),
            Line2D([0], [0], color='green', lw=2.5, label='y-bond: K·S^y·S^y'),
            Line2D([0], [0], color='blue', lw=2.5, label='z-bond: K·S^z·S^z'),
        ])
    
    if show_j2:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], color=j2_color, lw=1, linestyle='--', label='2nd NN (J2)'))
    
    if show_j3:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], color=j3_color, lw=1, linestyle=':', label='3rd NN (J3)'))
    
    if show_hex:
        from matplotlib.patches import Patch
        legend_elements.append(
            Patch(facecolor=hex_color, edgecolor='black', alpha=0.3, label='Hexagon (J7)'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_aspect('equal')
    title_parts = [f'Honeycomb Lattice ({dim1}×{dim2})']
    if show_nn: title_parts.append('NN')
    if show_j2: title_parts.append('J2')
    if show_j3: title_parts.append('J3')
    if show_hex: title_parts.append('Hex')
    
    ax.set_title(' + '.join(title_parts) + '\n● = Sublattice A, ○ = Sublattice B', fontsize=12)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    
    plt.show()


def plot_single_hexagon(dim1=5, dim2=5, hex_i=2, hex_j=2, output_file=None):
    """Plot a single hexagon with all its bonds labeled."""
    
    lattice = build_honeycomb_lattice(dim1, dim2)
    sites = lattice['sites']
    sublattices = lattice['sublattices']
    
    # Find the hexagon at (hex_i, hex_j)
    hex_idx = hex_i * dim2 + hex_j
    if hex_idx >= len(lattice['hexagons']):
        print(f"Invalid hexagon index. Max is {len(lattice['hexagons'])-1}")
        return
    
    hex_sites = lattice['hexagons'][hex_idx]
    hex_positions = np.array([sites[s] for s in hex_sites])
    center = np.mean(hex_positions, axis=0)
    
    # Check it's a valid (non-wrapped) hexagon
    max_dist = max(np.linalg.norm(p - center) for p in hex_positions)
    if max_dist > 1.5:
        print(f"Hexagon at ({hex_i}, {hex_j}) wraps around boundary. Try different indices.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get surrounding sites for context
    nearby_sites = []
    for idx, pos in enumerate(sites):
        if np.linalg.norm(pos - center) < 2.5:
            nearby_sites.append(idx)
    
    # Draw nearby NN bonds (faded)
    nn_colors = {0: 'red', 1: 'green', 2: 'blue'}
    for site_i, site_j, bond_type, _ in lattice['nn_bonds']:
        if site_i in nearby_sites or site_j in nearby_sites:
            pos_i, pos_j = sites[site_i], sites[site_j]
            if np.linalg.norm(pos_j - pos_i) < 1.2:
                in_hex = (site_i in hex_sites) and (site_j in hex_sites)
                alpha = 0.9 if in_hex else 0.2
                lw = 3 if in_hex else 1
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                       color=nn_colors[bond_type], linewidth=lw, alpha=alpha)
    
    # Draw hexagon outline
    polygon = Polygon(hex_positions, fill=True, facecolor='cyan', 
                     edgecolor='black', alpha=0.2, linewidth=2)
    ax.add_patch(polygon)
    
    # Draw and label hexagon vertices
    hex_labels = ['0: A(i,j)', '1: B(i,j)', '2: A(i,j+1)', 
                  '3: B(i+1,j)', '4: A(i+1,j)', '5: B(i,j-1)']
    
    for idx, (site, pos) in enumerate(zip(hex_sites, hex_positions)):
        # Site marker
        sub = sublattices[site]
        if sub == 'A':
            ax.scatter(pos[0], pos[1], c='black', s=200, zorder=10, marker='o')
        else:
            ax.scatter(pos[0], pos[1], c='white', s=200, zorder=10, marker='o', 
                      edgecolors='black', linewidths=2)
        
        # Label
        offset = (pos - center) * 0.4
        ax.annotate(hex_labels[idx], pos + offset, fontsize=11, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
    
    # Draw nearby sites (faded)
    for idx in nearby_sites:
        if idx not in hex_sites:
            pos = sites[idx]
            sub = sublattices[idx]
            if sub == 'A':
                ax.scatter(pos[0], pos[1], c='gray', s=60, zorder=5, marker='o', alpha=0.3)
            else:
                ax.scatter(pos[0], pos[1], c='lightgray', s=60, zorder=5, marker='o', 
                          edgecolors='gray', linewidths=1, alpha=0.3)
    
    # Add bond type annotations on hexagon edges
    hex_bonds = []
    for site_i, site_j, bond_type, label in lattice['nn_bonds']:
        if site_i in hex_sites and site_j in hex_sites:
            hex_bonds.append((site_i, site_j, bond_type, label))
    
    for site_i, site_j, bond_type, label in hex_bonds:
        pos_i, pos_j = sites[site_i], sites[site_j]
        mid = (pos_i + pos_j) / 2
        perp = np.array([-(pos_j[1] - pos_i[1]), pos_j[0] - pos_i[0]])
        perp = perp / np.linalg.norm(perp) * 0.15
        ax.annotate(f'{label}', mid + perp, fontsize=9, color=nn_colors[bond_type],
                   fontweight='bold', ha='center', va='center')
    
    ax.set_aspect('equal')
    ax.set_title(f'Hexagonal Plaquette at (i={hex_i}, j={hex_j})\n'
                 f'Ring exchange: H_7 = (J_7/6) × [triple products of S_i·S_j]', fontsize=12)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='x-bond'),
        Line2D([0], [0], color='green', lw=3, label='y-bond'),
        Line2D([0], [0], color='blue', lw=3, label='z-bond'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    
    plt.show()


def plot_j2_j3_detail(dim1=5, dim2=5, center_i=2, center_j=2, output_file=None):
    """Plot 2nd and 3rd NN connectivity from a central site."""
    
    lattice = build_honeycomb_lattice(dim1, dim2)
    sites = lattice['sites']
    sublattices = lattice['sublattices']
    
    def flatten_index(i, j, sublattice):
        sub = 0 if sublattice == 'A' else 1
        return 2 * (i * dim2 + j) + sub
    
    center_A = flatten_index(center_i, center_j, 'A')
    center_B = flatten_index(center_i, center_j, 'B')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # ============================================
    # Left: J2 (2nd NN, same sublattice)
    # ============================================
    ax = axes[0]
    
    center = sites[center_A]
    
    # Draw all NN bonds (faded)
    for site_i, site_j, bond_type, _ in lattice['nn_bonds']:
        pos_i, pos_j = sites[site_i], sites[site_j]
        if np.linalg.norm(pos_j - pos_i) < 1.2:
            dist_from_center = min(np.linalg.norm(pos_i - center), np.linalg.norm(pos_j - center))
            if dist_from_center < 2.5:
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                       color='gray', linewidth=1, alpha=0.3)
    
    # Draw J2 bonds from center_A
    j2_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    j2_labels = ['+a1', '-a1', '+a2', '-a2', '+a1-a2', '-a1+a2']
    
    for (di, dj), label in zip(j2_offsets, j2_labels):
        i_p, j_p = (center_i + di) % dim1, (center_j + dj) % dim2
        partner = flatten_index(i_p, j_p, 'A')
        pos_partner = sites[partner]
        
        # Only draw if not wrapped
        if np.linalg.norm(pos_partner - center) < 2.0:
            ax.plot([center[0], pos_partner[0]], [center[1], pos_partner[1]], 
                   color='orange', linewidth=2.5, alpha=0.8)
            # Label
            mid = (center + pos_partner) / 2
            ax.annotate(label, mid, fontsize=9, fontweight='bold', color='darkorange',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Draw sites
    for idx, (pos, sub) in enumerate(zip(sites, sublattices)):
        dist = np.linalg.norm(pos - center)
        if dist < 3.0:
            is_center = (idx == center_A)
            is_j2_neighbor = any(
                flatten_index((center_i + di) % dim1, (center_j + dj) % dim2, 'A') == idx 
                for di, dj in j2_offsets
            )
            
            size = 200 if is_center else (120 if is_j2_neighbor and sub == 'A' else 60)
            
            if sub == 'A':
                color = 'red' if is_center else ('orange' if is_j2_neighbor else 'black')
                ax.scatter(pos[0], pos[1], c=color, s=size, zorder=5, marker='o')
            else:
                ax.scatter(pos[0], pos[1], c='white', s=size, zorder=5, marker='o', 
                          edgecolors='black', linewidths=1, alpha=0.5)
    
    ax.set_aspect('equal')
    ax.set_title('2nd NN (J2): Same Sublattice (A↔A or B↔B)\n'
                 '6 neighbors per site, offsets: (±1,0), (0,±1), (±1,∓1)', fontsize=11)
    
    # ============================================
    # Right: J3 (3rd NN, opposite sublattice)
    # ============================================
    ax = axes[1]
    
    center = sites[center_A]
    
    # Draw all NN bonds (faded)
    for site_i, site_j, bond_type, _ in lattice['nn_bonds']:
        pos_i, pos_j = sites[site_i], sites[site_j]
        if np.linalg.norm(pos_j - pos_i) < 1.2:
            dist_from_center = min(np.linalg.norm(pos_i - center), np.linalg.norm(pos_j - center))
            if dist_from_center < 3.0:
                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                       color='gray', linewidth=1, alpha=0.3)
    
    # Draw J3 bonds from center_A to sublattice B
    # From C++ code: j3_A_to_B_offsets = {(1, -2), (-1, 0), (1, 0)}
    j3_offsets = [(1, -2), (-1, 0), (1, 0)]
    j3_labels = ['(+1,-2)', '(-1,0)', '(+1,0)']
    
    for (di, dj), label in zip(j3_offsets, j3_labels):
        i_p, j_p = (center_i + di) % dim1, (center_j + dj) % dim2
        partner = flatten_index(i_p, j_p, 'B')
        pos_partner = sites[partner]
        
        # Only draw if not wrapped
        if np.linalg.norm(pos_partner - center) < 2.5:
            ax.plot([center[0], pos_partner[0]], [center[1], pos_partner[1]], 
                   color='purple', linewidth=2.5, alpha=0.8)
            # Label
            mid = (center + pos_partner) / 2
            ax.annotate(label, mid, fontsize=9, fontweight='bold', color='purple',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Draw sites
    for idx, (pos, sub) in enumerate(zip(sites, sublattices)):
        dist = np.linalg.norm(pos - center)
        if dist < 3.5:
            is_center = (idx == center_A)
            is_j3_neighbor = any(
                flatten_index((center_i + di) % dim1, (center_j + dj) % dim2, 'B') == idx 
                for di, dj in j3_offsets
            )
            
            size = 200 if is_center else (120 if is_j3_neighbor else 60)
            
            if sub == 'A':
                color = 'red' if is_center else 'black'
                alpha = 1.0 if is_center else 0.5
                ax.scatter(pos[0], pos[1], c=color, s=size, zorder=5, marker='o', alpha=alpha)
            else:
                color = 'purple' if is_j3_neighbor else 'white'
                ax.scatter(pos[0], pos[1], c=color, s=size, zorder=5, marker='o', 
                          edgecolors='black', linewidths=1.5)
    
    ax.set_aspect('equal')
    ax.set_title('3rd NN (J3): Opposite Sublattice (A↔B)\n'
                 '3 neighbors per site, offsets (from A): (+1,-2), (±1,0)', fontsize=11)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    
    plt.show()


def verify_neighbor_counts(dim1=4, dim2=4):
    """Verify the neighbor counts match expected values."""
    
    lattice = build_honeycomb_lattice(dim1, dim2)
    
    n_sites = len(lattice['sites'])
    n_nn = len(lattice['nn_bonds'])
    n_j2 = len(lattice['j2_bonds'])
    n_j3 = len(lattice['j3_bonds'])
    n_hex = len(lattice['hexagons'])
    
    print(f"Lattice: {dim1}×{dim2} = {dim1*dim2} unit cells, {n_sites} sites")
    print()
    print(f"NN bonds (z=3 per site, each counted once from A): {n_nn}")
    print(f"  Expected: 3 × {dim1*dim2} = {3*dim1*dim2}")
    print()
    print(f"J2 bonds (z=6 per site, counted once, half from A, half from B): {n_j2}")
    # Each site has 6 J2 neighbors, but we only count j > i, and bonds are within same sublattice
    # For each sublattice: n_sites/2 sites, each with ~3 neighbors counted (avoiding double count)
    print(f"  Expected: ~{6 * n_sites // 2} (depends on boundary conditions)")
    print()
    print(f"J3 bonds (3 per A site to B, counted once): {n_j3}")
    print(f"  Expected: ~{3 * dim1 * dim2} (from A sites)")
    print()
    print(f"Hexagons: {n_hex}")
    print(f"  Expected: {dim1*dim2} (one per unit cell)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive honeycomb lattice visualization')
    parser.add_argument('--dim', type=int, nargs=2, default=[5, 5], help='Lattice dimensions')
    parser.add_argument('--nn', action='store_true', default=True, help='Show NN bonds')
    parser.add_argument('--j2', action='store_true', help='Show 2nd NN bonds')
    parser.add_argument('--j3', action='store_true', help='Show 3rd NN bonds')
    parser.add_argument('--hex', action='store_true', help='Show hexagon')
    parser.add_argument('--hex-idx', type=int, default=6, help='Hexagon index to highlight')
    parser.add_argument('--hex-detail', action='store_true', help='Detailed hexagon plot')
    parser.add_argument('--j2j3-detail', action='store_true', help='Detailed J2/J3 plot')
    parser.add_argument('--verify', action='store_true', help='Verify neighbor counts')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_neighbor_counts(args.dim[0], args.dim[1])
    elif args.hex_detail:
        plot_single_hexagon(args.dim[0], args.dim[1], hex_i=2, hex_j=2, output_file=args.output)
    elif args.j2j3_detail:
        plot_j2_j3_detail(args.dim[0], args.dim[1], output_file=args.output)
    else:
        plot_full_lattice(args.dim[0], args.dim[1], 
                         show_nn=args.nn, show_j2=args.j2, show_j3=args.j3,
                         show_hex=args.hex, highlight_hex_idx=args.hex_idx,
                         output_file=args.output)
