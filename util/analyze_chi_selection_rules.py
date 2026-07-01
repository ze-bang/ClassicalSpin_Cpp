#!/usr/bin/env python3
"""
analyze_chi_selection_rules.py
==============================
Comprehensive analysis of ALL chi coupling channels in TmFeO3 Gamma_2 phase.

Covers two scenarios:
  A) Ground state: which channels produce nonzero lambda^2 in the Gamma_2
     background, and why. Exact algebra via the sigma product rule.
  B) qAFM dynamics: equation-of-motion analysis for each channel.
     The full Gell-Mann EOM is derived step by step, and the selection-rule
     cancellation is shown at the level of individual sigma projections.

Also produces 3D visualisation figures:
  Fig 1: TmFeO3 crystal structure with all Fe–Tm chi bonds, coloured by orbit
  Fig 2: Local Fe frame axes (eta^x, eta^y, eta^z signs visualised)
  Fig 3: Local Tm frame axes
  Fig 4: Summary table — ground-state activity and qAFM detectability
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')        # headless / file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (side-effect register)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec

# ============================================================================
# Crystal structure data (fractional coordinates, Pbnm setting)
# ============================================================================

# Lattice parameters [Angstrom] (Herrmann 2020 / Wu 2011 TmFeO3 values)
a, b, c = 5.302, 5.582, 7.616

# Fe Wyckoff 4b positions — from TmFeO3_Fe class in unitcell.h
# sub 0 = E, sub 1 = C2x, sub 2 = C2z, sub 3 = C2y (local frame ordering)
Fe_frac = np.array([
    [0.0,     0.5,     0.5],   # sub 0
    [0.5,     0.0,     0.5],   # sub 1
    [0.5,     0.0,     0.0],   # sub 2
    [0.0,     0.5,     0.0],   # sub 3
])

# Tm Wyckoff 4c positions — from TmFeO3_Tm class in unitcell.h
# sub 0 = E, sub 1 = C2x, sub 2 = C2y, sub 3 = C2z (local frame ordering)
Tm_frac = np.array([
    [0.02111, 0.92839, 0.75],  # sub 0
    [0.52111, 0.57161, 0.25],  # sub 1
    [0.47889, 0.42839, 0.75],  # sub 2
    [0.97889, 0.07161, 0.25],  # sub 3
])

# Convert to Cartesian [Angstrom]
A = np.diag([a, b, c])
Fe_cart = Fe_frac @ A
Tm_cart = Tm_frac @ A

# ============================================================================
# Sublattice sign conventions (CODE convention, used throughout unitcell.cpp)
# ============================================================================
sigma_F = np.array([+1, +1, +1, +1], dtype=float)
sigma_G = np.array([+1, -1, +1, -1], dtype=float)   # kSignG in diagnostic
sigma_A = np.array([+1, +1, -1, -1], dtype=float)   # kSignA in diagnostic
sigma_C = np.array([+1, -1, -1, +1], dtype=float)   # kSignC in diagnostic

sigma_names = {'F': sigma_F, 'G': sigma_G, 'A': sigma_A, 'C': sigma_C}

def sigma_prod(s1, s2):
    """Component-wise product and identify the result."""
    prod = s1 * s2
    for name, sig in sigma_names.items():
        if np.allclose(prod, sig):
            return prod, f'sigma_{name}'
    return prod, str(prod)

# ============================================================================
# Fe local frame diagonal elements (from R_fe_frames in unitcell_builders.cpp)
# ============================================================================
#   sub 0: E    -> diag(+,+,+)
#   sub 1: C2x  -> diag(+,-,-)
#   sub 2: C2z  -> diag(-,-,+)
#   sub 3: C2y  -> diag(-,+,-)
Fe_eta = np.array([
    [+1, +1, +1],   # sub 0: eta^{x,y,z}
    [+1, -1, -1],   # sub 1
    [-1, -1, +1],   # sub 2
    [-1, +1, -1],   # sub 3
], dtype=float)

# eta^x_Fe = (+,+,-,-) = sigma_A
# eta^y_Fe = (+,-,-,+) = sigma_C
# eta^z_Fe = (+,-,+,-) = sigma_G

# Tm local frame diagonal elements (from tmfeo3_tm_local_frames_xyz())
#   sub 0: diag(+,+,+)
#   sub 1: diag(+,-,-)
#   sub 2: diag(-,+,-)
#   sub 3: diag(-,-,+)
Tm_eta = np.array([
    [+1, +1, +1],   # sub 0
    [+1, -1, -1],   # sub 1
    [-1, +1, -1],   # sub 2
    [-1, -1, +1],   # sub 3
], dtype=float)

# eta^x_Tm = (+,+,-,-) = sigma_A
# eta^y_Tm = (+,-,+,-) = sigma_G
# eta^z_Tm = (+,-,-,+) = sigma_C   <-- NOT sigma_A despite earlier labelling confusion

# ============================================================================
# Bond pairs (from fe_tm_w_bond_pairs in unitcell_builders.cpp)
# Each entry: (orbit, fe_sub, tm_sub, cell_offset)
# All 16 even bonds (one from each FeTmBondPair.even)
# ============================================================================
even_bonds = [
    # orbit, fe_sub, tm_sub, offset
    (1, 0, 3, (-1,  0,  0)),
    (2, 0, 2, ( 0,  0,  0)),
    (3, 0, 1, ( 0,  0,  0)),
    (4, 0, 0, ( 0, -1,  0)),
    (1, 1, 2, ( 0,  0,  0)),
    (2, 1, 3, ( 0,  0,  0)),
    (3, 1, 0, ( 1, -1,  0)),
    (4, 1, 1, ( 0,  0,  0)),
    (1, 2, 1, ( 0, -1,  0)),
    (2, 2, 0, ( 0, -1, -1)),
    (3, 2, 3, (-1,  0,  0)),
    (4, 2, 2, ( 0, -1, -1)),
    (1, 3, 0, ( 0,  0, -1)),
    (2, 3, 1, (-1,  0,  0)),
    (3, 3, 2, (-1,  0, -1)),
    (4, 3, 3, (-1,  1,  0)),
]

odd_bonds = [
    # orbit, fe_sub, tm_sub, offset — sign_57 = -1
    (1, 0, 0, ( 0,  0,  0)),
    (2, 0, 1, (-1,  0,  0)),
    (3, 0, 2, (-1,  0,  0)),
    (4, 0, 3, (-1,  1,  0)),
    (1, 1, 1, ( 0, -1,  0)),
    (2, 1, 0, ( 0, -1,  0)),
    (3, 1, 3, (-1,  0,  0)),
    (4, 1, 2, ( 0, -1,  0)),
    (1, 2, 2, ( 0,  0, -1)),
    (2, 2, 3, ( 0,  0,  0)),
    (3, 2, 0, ( 1, -1, -1)),
    (4, 2, 1, ( 0,  0,  0)),
    (1, 3, 3, (-1,  0,  0)),
    (2, 3, 2, ( 0,  0, -1)),
    (3, 3, 1, ( 0,  0,  0)),
    (4, 3, 0, ( 0, -1, -1)),
]

orbit_colors = {1: '#E74C3C', 2: '#3498DB', 3: '#2ECC71', 4: '#F39C12'}
orbit_labels = {1: 'Orbit 1', 2: 'Orbit 2', 3: 'Orbit 3', 4: 'Orbit 4'}

# ============================================================================
# Helper: Cartesian position with cell offset
# ============================================================================
def cart_pos(frac, sub, off):
    return (frac[sub] + np.array(off, dtype=float)) @ A

# ============================================================================
# FIGURE 1: Crystal structure with Fe–Tm chi bonds (3D, 2x2 viewpoints)
# ============================================================================

def draw_unit_cell_box(ax, alpha=0.07):
    """Draw the Pbnm unit cell box."""
    corners = np.array([[i, j, k] for i in [0,1] for j in [0,1] for k in [0,1]])
    corners_cart = corners @ A
    edges = [
        (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    ]
    for i, j in edges:
        ax.plot(*zip(corners_cart[i], corners_cart[j]), 'k-', lw=0.6, alpha=0.4)

def make_bond_figure():
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        'TmFeO3 crystal structure — Fe–Tm chi coupling bonds\n'
        '(Pbnm Γ₂ phase, 4 orbits × 2 bonds per orbit = 8 unique bond types)',
        fontsize=13, fontweight='bold')

    viewpoints = [
        ('Looking down c-axis (a–b plane)',  (90,  0)),
        ('Looking down b-axis (a–c plane)',  ( 0, 90)),
        ('Perspective 1 (ab-plane tilt)',     (25, 45)),
        ('Perspective 2 (bc-plane tilt)',     (20,135)),
    ]

    Fe_colors = ['#C0392B', '#8E44AD', '#1ABC9C', '#D35400']  # one per sublattice
    Tm_colors = ['#2980B9', '#16A085', '#7F8C8D', '#F1C40F']

    axes = []
    for panel, (title, (elev, azim)) in enumerate(viewpoints):
        ax = fig.add_subplot(2, 2, panel + 1, projection='3d')
        ax.set_title(title, fontsize=10)
        draw_unit_cell_box(ax)

        # Draw all bonds (both even and odd, within the unit cell / nearest image)
        for bond_set, sign_label in [(even_bonds, 'E'), (odd_bonds, 'O')]:
            for orbit, fe_sub, tm_sub, off in bond_set:
                p_fe = cart_pos(Fe_frac, fe_sub, (0, 0, 0))
                p_tm = cart_pos(Tm_frac, tm_sub, off)
                # Clip to stay within ~1 unit cell for clarity
                mid = (p_fe + p_tm) / 2
                c = orbit_colors[orbit]
                lw = 1.5 if sign_label == 'E' else 0.8
                ls = '-' if sign_label == 'E' else '--'
                ax.plot(*zip(p_fe, p_tm), color=c, lw=lw, ls=ls, alpha=0.7)

        # Draw Fe atoms
        for sub in range(4):
            p = cart_pos(Fe_frac, sub, (0, 0, 0))
            ax.scatter(*p, color=Fe_colors[sub], s=200, zorder=5,
                       edgecolors='black', linewidths=0.7, marker='o',
                       label=f'Fe{sub}' if panel == 0 else '')
            if panel == 0:
                ax.text(p[0]+0.1, p[1]+0.1, p[2]+0.1, f'Fe{sub}', fontsize=8,
                        color=Fe_colors[sub], fontweight='bold')

        # Draw Tm atoms
        for sub in range(4):
            p = cart_pos(Tm_frac, sub, (0, 0, 0))
            ax.scatter(*p, color=Tm_colors[sub], s=140, zorder=5,
                       edgecolors='black', linewidths=0.7, marker='s',
                       label=f'Tm{sub}' if panel == 0 else '')
            if panel == 0:
                ax.text(p[0]+0.1, p[1]+0.1, p[2]+0.1, f'Tm{sub}', fontsize=8,
                        color=Tm_colors[sub], fontweight='bold')

        ax.set_xlabel('x (Å)', fontsize=8)
        ax.set_ylabel('y (Å)', fontsize=8)
        ax.set_zlabel('z (Å)', fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        axes.append(ax)

    # Legend for orbits
    legend_handles = [
        mpatches.Patch(color=orbit_colors[o], label=f'Orbit {o}') for o in range(1, 5)
    ]
    legend_handles += [
        plt.Line2D([0],[0], color='k', lw=1.5, ls='-', label='even bond (sign₅₇=+1)'),
        plt.Line2D([0],[0], color='k', lw=0.8, ls='--', label='odd bond (sign₅₇=−1)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=6,
               fontsize=9, framealpha=0.9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ============================================================================
# FIGURE 2: Fe local frame axes (eta signs) shown as 3D arrows
# ============================================================================

def make_fe_frame_figure():
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(
        'Fe sublattice local frames (R_fe_frames in unitcell_builders.cpp)\n'
        'Arrows: η^x (red), η^y (green), η^z (blue). '
        'Solid = +1, dashed = −1',
        fontsize=12, fontweight='bold')

    Fe_colors = ['#C0392B', '#8E44AD', '#1ABC9C', '#D35400']
    frame_labels = ['E → diag(+,+,+)', 'C₂x → diag(+,−,−)',
                    'C₂z → diag(−,−,+)', 'C₂y → diag(−,+,−)']
    sigma_labels = [
        'η^x=+1  η^y=+1  η^z=+1',
        'η^x=+1  η^y=−1  η^z=−1',
        'η^x=−1  η^y=−1  η^z=+1',
        'η^x=−1  η^y=+1  η^z=−1',
    ]

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title('Fe local frame arrows at each sublattice position', fontsize=10)
    draw_unit_cell_box(ax)

    arrow_len = 1.2
    for sub in range(4):
        p = cart_pos(Fe_frac, sub, (0, 0, 0))
        ax.scatter(*p, color=Fe_colors[sub], s=250, zorder=5,
                   edgecolors='black', linewidths=0.8)
        ax.text(p[0]+0.05, p[1]+0.05, p[2]+0.3,
                f'Fe{sub}\n{frame_labels[sub]}', fontsize=7.5,
                color=Fe_colors[sub], ha='left')

        eta = Fe_eta[sub]
        dirs = np.eye(3) * arrow_len  # x, y, z directions
        axis_colors = ['#E74C3C', '#27AE60', '#2980B9']  # x=red, y=green, z=blue
        axis_names = ['x', 'y', 'z']
        for ax_i in range(3):
            d = eta[ax_i] * dirs[ax_i]
            style = '-' if eta[ax_i] > 0 else '--'
            ax.quiver(*p, *d, color=axis_colors[ax_i], lw=2, alpha=0.85,
                      arrow_length_ratio=0.25, linestyle=style)
            tip = p + d * 1.05
            ax.text(*tip, f'η^{axis_names[ax_i]}={eta[ax_i]:+.0f}',
                    fontsize=6.5, color=axis_colors[ax_i], ha='center')

    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)'); ax.set_zlabel('z (Å)')
    ax.view_init(elev=20, azim=45)

    # Table panel: sigma vector summary
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    table_data = [
        ['Component', 'sub 0', 'sub 1', 'sub 2', 'sub 3', 'Pattern'],
        ['η^x (R_xx)', '+1', '+1', '−1', '−1', 'σ_A'],
        ['η^y (R_yy)', '+1', '−1', '−1', '+1', 'σ_C'],
        ['η^z (R_zz)', '+1', '−1', '+1', '−1', 'σ_G'],
    ]
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11)
    table.scale(1.5, 2.5)

    # Color the Pattern column
    pattern_colors = {'σ_A': '#E8D5B7', 'σ_C': '#D5E8D4', 'σ_G': '#DAE8FC'}
    for i, row in enumerate(table_data[1:]):
        cell = table[i + 1, 5]
        cell.set_facecolor(pattern_colors.get(row[-1], 'white'))

    ax2.set_title(
        'Fe frame diagonal elements = rotated Fe spin components\n'
        'H[λ²]_j += R^(k)_αα · χ₂α · S^α_Fe,k  →  η^α_Fe = R^(k)_αα',
        fontsize=10, pad=15)

    # Add sigma pattern legend
    text = (
        'CODE convention sigma vectors:\n'
        '  σ_F = (+,+,+,+)  ferromagnetic\n'
        '  σ_G = (+,−,+,−)  G-type AFM  ← Γ₂ ordering\n'
        '  σ_A = (+,+,−,−)  A-type AFM\n'
        '  σ_C = (+,−,−,+)  C-type AFM\n\n'
        'Dot products (key for selection rules):\n'
        '  σ_G · σ_G = 4   σ_G · σ_A = 0\n'
        '  σ_G · σ_C = 0   σ_G · σ_F = 0\n'
        '  σ_A · σ_F = 0   σ_C · σ_F = 0'
    )
    ax2.text(0.02, 0.02, text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8))

    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 3: Tm local frame axes and the M_α_Tm detector patterns
# ============================================================================

def make_tm_frame_figure():
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(
        'Tm sublattice local frames (tmfeo3_tm_local_frames_xyz)\n'
        'The η^z_Tm pattern is the detector sensitivity for H//z (c-axis) spectroscopy',
        fontsize=12, fontweight='bold')

    Tm_colors = ['#2980B9', '#16A085', '#7F8C8D', '#F1C40F']
    frame_labels = ['diag(+,+,+)', 'diag(+,−,−)', 'diag(−,+,−)', 'diag(−,−,+)']

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title('Tm local frame arrows at each sublattice position', fontsize=10)
    draw_unit_cell_box(ax)

    arrow_len = 1.0
    for sub in range(4):
        p = cart_pos(Tm_frac, sub, (0, 0, 0))
        ax.scatter(*p, color=Tm_colors[sub], s=200, zorder=5,
                   edgecolors='black', linewidths=0.8, marker='s')
        ax.text(p[0]+0.05, p[1]+0.05, p[2]+0.3,
                f'Tm{sub}\n{frame_labels[sub]}', fontsize=7.5,
                color=Tm_colors[sub], ha='left')

        eta = Tm_eta[sub]
        dirs = np.eye(3) * arrow_len
        axis_colors = ['#E74C3C', '#27AE60', '#2980B9']
        axis_names = ['x', 'y', 'z']
        for ax_i in range(3):
            d = eta[ax_i] * dirs[ax_i]
            ax.quiver(*p, *d, color=axis_colors[ax_i], lw=2, alpha=0.85,
                      arrow_length_ratio=0.3,
                      linestyle='-' if eta[ax_i] > 0 else '--')

    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)'); ax.set_zlabel('z (Å)')
    ax.view_init(elev=25, azim=55)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    table_data = [
        ['Component', 'sub 0', 'sub 1', 'sub 2', 'sub 3', 'Pattern'],
        ['η^x_Tm (R_xx)', '+1', '+1', '−1', '−1', 'σ_A'],
        ['η^y_Tm (R_yy)', '+1', '−1', '+1', '−1', 'σ_G'],
        ['η^z_Tm (R_zz)', '+1', '−1', '−1', '+1', 'σ_C'],
    ]
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc='upper center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11)
    table.scale(1.5, 2.5)

    pattern_colors = {'σ_A': '#E8D5B7', 'σ_C': '#DAE8FC', 'σ_G': '#D5E8D4'}
    for i, row in enumerate(table_data[1:]):
        cell = table[i + 1, 5]
        cell.set_facecolor(pattern_colors.get(row[-1], 'white'))

    mz_text = (
        'Detector coupling for H//z (c-axis field / detection):\n\n'
        '  M_z^Tm = Σ_j η^z_Tm,j · c_2 · λ²_j\n'
        '         = σ_C · λ²_vector\n\n'
        'If λ² has sigma pattern σ_P, then:\n'
        '  M_z^Tm = σ_C · σ_P · λ²_avg\n\n'
        'Nonzero only if σ_C · σ_P = 4, i.e. σ_P = σ_C\n\n'
        'chi2z drives λ² with σ_F pattern:\n'
        '   σ_C · σ_F = 0  →  M_z^Tm = 0  (FORBIDDEN)\n\n'
        'chi2x drives λ² with σ_A pattern (if F_x ≠ 0):\n'
        '   σ_C · σ_A = 0  →  M_z^Tm = 0  (FORBIDDEN)\n\n'
        'For M_z^Tm to be nonzero, λ² must carry σ_C pattern.\n'
        'chi2y would drive λ² with σ_C pattern (η^y_Fe = σ_C),\n'
        'because σ_C · σ_G ≠ 0... but wait: in Γ₂, S^y_Fe = 0.\n'
        'So even chi2y cannot generate λ² in the equilibrium state.'
    )
    ax2.text(0.02, 0.05, mz_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#EAF4FB', alpha=0.9))

    fig.tight_layout()
    return fig


# ============================================================================
# Ground-state selection rule analysis (algebraic)
# ============================================================================

def analyze_ground_state():
    """
    Compute which chi channels generate nonzero H[λ²] in the Gamma_2 ground state.

    In Gamma_2:
      S^x_Fe,k = F_x · sigma_F[k]          (small ferromagnetic canting)
      S^y_Fe,k = 0                           (C_y component absent)
      S^z_Fe,k = G_z · sigma_G[k]           (main antiferromagnetic order)

    For a chi_2α channel, the effective field on Tm site j bonded to Fe site k is:
      H[λ²]_j ∝ η^α_Fe[k] · chi_2α · S^α_Fe,k
              = R^(k)_αα · chi_2α · S^α_Fe,k

    When summed over the 4 bonds per Tm site (one from each orbit), the pattern
    of H[λ²] across the 4 Tm sublattices is determined by:
      sigma_pattern(H[λ²]) = eta^α_Fe · S^α_Fe_pattern (component-wise product)
    """
    print('\n' + '='*72)
    print('GROUND STATE ANALYSIS: which chi channels activate lambda^2')
    print('='*72)
    print("""
Gamma_2 Fe spin state (Pbnm, kSignG=(+,-,+,-)):
  S^z_Fe = G_z * sigma_G  (large, ~2.5 mu_B per Fe)
  S^x_Fe = F_x * sigma_F  (small DM canting, ~0.014 mu_B per Fe)
  S^y_Fe = 0

The effective field coupling chi_2alpha to lambda^2 at Tm site j is:
  H[lambda^2]_j = sum_{k in bonds(j)} R_fe^(k)_aa * chi_2alpha * S^alpha_Fe,k

Since each Tm site has one bond per orbit (4 orbits, 4 bonds total):
  H[lambda^2]_j = chi_2alpha * sum_{orbits} eta^alpha_Fe[k_orbit(j)] * S^alpha_Fe[k_orbit(j)]

The sigma-pattern of H[lambda^2] across j=0..3 is:
  H[lambda^2] ~ chi_2alpha * (eta^alpha_Fe) . (S^alpha_Fe pattern) * 4
""")

    cases = [
        ('chi2z', 2, 'sigma_G', sigma_G, 'G_z * sigma_G',  'sigma_G'),
        ('chi2x', 0, 'sigma_A', sigma_A, 'F_x * sigma_F',  'sigma_F'),
        ('chi2y', 1, 'sigma_C', sigma_C, '0',               'zero'),
    ]

    for chi_name, alpha, eta_fe_name, eta_fe, S_pattern_str, S_sigma_name in cases:
        print(f'  {chi_name}: eta^alpha_Fe = {eta_fe_name} = {eta_fe.astype(int)}')
        print(f'    S^alpha_Fe in Gamma_2 = {S_pattern_str}')
        if S_sigma_name == 'zero':
            print(f'    -> H[lambda^2] = 0  (S^y_Fe = 0 in Gamma_2, no chi2y coupling)')
        else:
            # Compute product eta^alpha_Fe * S_sigma
            S_sigma = sigma_names[S_sigma_name[-1]] if 'sigma_' in S_sigma_name else sigma_F
            prod, name = sigma_prod(eta_fe, S_sigma)
            print(f'    eta^alpha_Fe * S_sigma = {eta_fe_name} * {S_sigma_name}')
            print(f'      = {prod.astype(int)} = {name}')
            if 'F' in name:
                print(f'    -> H[lambda^2] = {chi_name} * G_z * sigma_F  (UNIFORM, active!)')
                print(f'    -> lambda^2 ~ sigma_F at all Tm sites')
            else:
                print(f'    -> H[lambda^2] = {chi_name} * F_x * {name}  '
                      f'(staggered, but F_x ~ 0)')
                print(f'    -> lambda^2 ~ negligible (F_x/G_z ~ 0.006)')
        print()

    print("""
Summary for chi5 and chi7:
  These couple to lambda^5 and lambda^7 respectively, not lambda^2.
  In the ground state they can generate lambda^5 or lambda^7 via the
  same sigma product rule.
  However: each Tm site is connected to the SAME Fe sublattice via both
  the even bond (sign_57=+1) and odd bond (sign_57=-1) within each orbit
  pair. The even and odd bonds share pair.even.fe == pair.odd.fe.
  -> H[lambda^5]_j = sum_{orbits} [+1*eta + -1*eta] * chi5 * S_Fe = 0
  -> H[lambda^7]_j = 0 identically (same cancellation)
  Chi5 and chi7 are dead for ALL q=0 Fe configurations.
""")


# ============================================================================
# Ground-state chi2y analysis: is it really dead?
# ============================================================================

def analyze_chi2y_detail():
    """
    Detailed check: chi2y should be dead because S^y_Fe = 0 in Gamma_2.
    But what if we have a C_y component? What sigma pattern would it drive?
    """
    print('='*72)
    print('CHI2Y DETAIL: would be active if S^y_Fe != 0')
    print('='*72)
    eta_y_fe = Fe_eta[:, 1]  # sigma_C
    eta_z_tm = Tm_eta[:, 2]  # sigma_C
    print(f"""
  eta^y_Fe = {eta_y_fe.astype(int)} = sigma_C

  If there were a C_y ordering component S^y_Fe = C_y * sigma_C:
    H[lambda^2]_j = chi2y * eta^y_Fe * C_y * sigma_C
                   = chi2y * C_y * (sigma_C * sigma_C)
                   = chi2y * C_y * sigma_F  (UNIFORM)
  -> lambda^2 would be sigma_F (same as chi2z with G_z!)

  Then: M_z^Tm = eta^z_Tm · lambda^2 = sigma_C · sigma_F = {int(np.dot(Tm_eta[:,2], sigma_F))}
  -> Still zero! chi2y with C_y would be ACTIVE but also UNDETECTABLE by H//z.

  What sigma pattern of lambda^2 would be detectable by H//z?
  Need: eta^z_Tm · lambda^2 = sigma_C · lambda^2_sigma = 4
  -> lambda^2 must have sigma_C pattern.

  Which chi channel drives sigma_C lambda^2?
    chi2y if S^y_Fe ~ sigma_A (A_y component): sigma_C * sigma_A = ?
""")
    prod, name = sigma_prod(Fe_eta[:, 1], sigma_A)
    print(f'    sigma_C * sigma_A = {prod.astype(int)} = {name}')
    print(f"""
  None of the Gamma_2 components (sigma_G, sigma_F, sigma_A for F_x) are
  sigma_A for y.  The only way to get sigma_C lambda^2 (detectable by H//z)
  is through chi2y with an A_y magnetic component.
  A_y is absent in Gamma_2 by symmetry (requires C_2y breaking).
""")


# ============================================================================
# FIGURE 4: Comprehensive chi channel activity table
# ============================================================================

def make_activity_table_figure():
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(
        'Chi coupling channel activity in TmFeO3 Gamma_2 phase\n'
        'Left: ground-state lambda generation  |  Right: H//z detectability (M_z^Tm)',
        fontsize=13, fontweight='bold')

    # --- Left panel: ground-state lambda^2 generation ---
    ax = axes[0]
    ax.axis('off')
    ax.set_title('Ground state: H[λ²] pattern generated by each channel',
                 fontsize=11, pad=15)

    headers_gs = ['Channel', 'η^α_Fe', 'Fe spin\npattern', 'η·S\nproduct', 'λ² sigma\npattern', 'Active?']
    rows_gs = [
        ['chi2z',  'σ_G=(+−+−)', 'G_z·σ_G', 'σ_G·σ_G=σ_F', 'σ_F (uniform)', 'YES ★'],
        ['chi2x',  'σ_A=(++−−)', 'F_x·σ_F', 'σ_A·σ_F=0',   'none (F_x≈0)', 'no'],
        ['chi2y',  'σ_C=(+−−+)', 'S^y=0',   '—',            'none',          'no'],
        ['chi5z',  'σ_G=(+−+−)', '(any)',    'cancels',      'none',          'no ✗'],
        ['chi5x',  'σ_A=(++−−)', '(any)',    'cancels',      'none',          'no ✗'],
        ['chi5y',  'σ_C=(+−−+)', '(any)',    'cancels',      'none',          'no ✗'],
        ['chi7z',  'σ_G=(+−+−)', '(any)',    'cancels',      'none',          'no ✗'],
        ['chi7x',  'σ_A=(++−−)', '(any)',    'cancels',      'none',          'no ✗'],
        ['chi7y',  'σ_C=(+−−+)', '(any)',    'cancels',      'none',          'no ✗'],
    ]
    gs_colors = [
        ['#D5F5E3','#D5F5E3','#D5F5E3','#D5F5E3','#D5F5E3','#27AE60'],
        ['#FDFEFE','#FDFEFE','#FDFEFE','#FDFEFE','#FEF9E7','#F5CBA7'],
        ['#FDFEFE','#FDFEFE','#FDFEFE','#FDFEFE','#FDFEFE','#F5CBA7'],
        ['#FADBD8','#FADBD8','#FADBD8','#FADBD8','#FADBD8','#E74C3C'],
        ['#FADBD8','#FADBD8','#FADBD8','#FADBD8','#FADBD8','#E74C3C'],
        ['#FADBD8','#FADBD8','#FADBD8','#FADBD8','#FADBD8','#E74C3C'],
        ['#FADBD8','#FADBD8','#FADBD8','#FADBD8','#FADBD8','#E74C3C'],
        ['#FADBD8','#FADBD8','#FADBD8','#FADBD8','#FADBD8','#E74C3C'],
        ['#FADBD8','#FADBD8','#FADBD8','#FADBD8','#FADBD8','#E74C3C'],
    ]
    t1 = ax.table(cellText=rows_gs, colLabels=headers_gs,
                  loc='center', cellLoc='center')
    t1.auto_set_font_size(False); t1.set_fontsize(9.5)
    t1.scale(1.3, 2.8)
    for (r, c), cell in t1.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r >= 1:
            cell.set_facecolor(gs_colors[r-1][c])

    note_gs = (
        '★ chi5/chi7 cancel because even/odd bonds share the same Fe sublattice\n'
        '  within each orbit pair: sign_57=(+1)+(-1)=0 identically.\n'
        '  chi2y inactive because S^y_Fe=0 in Γ₂ by symmetry.\n'
        '  chi2x inactive because σ_A·σ_F=0 (small F_x canting is σ_F, not σ_A).'
    )
    ax.text(0.01, 0.01, note_gs, transform=ax.transAxes, fontsize=8.5,
            va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.9))

    # --- Right panel: H//z detectability ---
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('M_z^Tm = η^z_Tm · λ² = σ_C · λ²   (H//z detector coupling)',
                  fontsize=11, pad=15)

    headers_det = ['Channel', 'λ² sigma\npattern', 'σ_C · λ²_sigma', 'M_z^Tm', 'Detectable\nby H//z?']
    rows_det = [
        ['chi2z',  'σ_F=(++++)', 'σ_C·σ_F = 0', '= 0', 'NO ✗'],
        ['chi2x',  'σ_A (tiny)', 'σ_C·σ_A = 0', '= 0', 'NO ✗'],
        ['chi2y',  'dead',       '—',            '0',   'NO ✗'],
        ['chi5*',  'dead',       '—',            '0',   'NO ✗'],
        ['chi7*',  'dead',       '—',            '0',   'NO ✗'],
        ['—',      'need σ_C',   'σ_C·σ_C = 4',  '≠ 0', 'YES'],
    ]
    det_colors = [
        ['#FDFEFE','#FDFEFE','#FDFEFE','#FDFEFE','#E74C3C'],
        ['#FDFEFE','#FDFEFE','#FDFEFE','#FDFEFE','#E74C3C'],
        ['#FDFEFE','#FDFEFE','#FDFEFE','#FDFEFE','#E74C3C'],
        ['#FDFEFE','#FDFEFE','#FDFEFE','#FDFEFE','#E74C3C'],
        ['#FDFEFE','#FDFEFE','#FDFEFE','#FDFEFE','#E74C3C'],
        ['#D5F5E3','#D5F5E3','#D5F5E3','#D5F5E3','#27AE60'],
    ]
    t2 = ax2.table(cellText=rows_det, colLabels=headers_det,
                   loc='center', cellLoc='center')
    t2.auto_set_font_size(False); t2.set_fontsize(10)
    t2.scale(1.4, 3.2)
    for (r, c), cell in t2.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r >= 1:
            cell.set_facecolor(det_colors[r-1][c])

    note_det = (
        'H//z couples to M_z^Tm = Σ_j η^z_Tm,j · J^z_Tm,j(local)\n'
        '  J^z_Tm,j(local) = c_2 · λ²_j  (leading-order matrix element)\n'
        '  η^z_Tm = σ_C = (+,−,−,+)  [from Tm frame diagonals R_zz]\n\n'
        'For any chi channel, λ² must carry σ_C pattern to be detected by H//z.\n'
        'No chi2 channel produces σ_C lambda^2 from Gamma_2 background:\n'
        '  chi2z → σ_F,  chi2x → 0,  chi2y → dead.\n\n'
        'CONCLUSION: The (omega_qAFM, omega_E12) cross-peak is FORBIDDEN\n'
        'under H//z for ALL chi channels in the Gamma_2 phase.'
    )
    ax2.text(0.01, 0.01, note_det, transform=ax2.transAxes, fontsize=9,
             va='bottom', ha='left',
             bbox=dict(boxstyle='round', facecolor='#EAF4FB', alpha=0.9))

    fig.tight_layout()
    return fig


# ============================================================================
# EOM ANALYSIS (printed to stdout)
# ============================================================================

def print_eom_analysis():
    print('\n' + '='*72)
    print('EQUATION OF MOTION ANALYSIS: qAFM excitation and chi coupling')
    print('='*72)
    print(r"""
─────────────────────────────────────────────────────────────────────────
I.  THE FULL SU(3) EQUATION OF MOTION
─────────────────────────────────────────────────────────────────────────

The Tm^3+ non-Kramers ion is modelled as a 3-level system (qutrits) with
Gell-Mann generators lambda^a (a = 1..8).  The equation of motion is the
quantum Liouville equation projected onto the Gell-Mann basis:

  d/dt lambda^a_j = -(i/hbar) Tr([H_eff_j, rho_j] * F^a)
                  = (1/hbar) f^{abc} H_eff^b_j lambda^c_j

where f^{abc} is the fully antisymmetric SU(3) structure constant and
H_eff^b_j is the b-th component of the effective field acting on site j.

The effective field decomposes as:
  H_eff^b_j = H_CEF^b  +  H_chi^b_j  +  H_Jtm^b_j  +  H_field^b_j

For the chi coupling only (all other terms set to zero for clarity):
  H_chi^b_j = sum_{k in bonds(j)} W^{alpha,b}_{jk} S^alpha_Fe,k

The tensor W comes from build_chi_bond:

  For chi2z:   W^{z, lambda^2}_{jk} = R^(k)_zz * chi2z = eta^z_Fe[k] * chi2z
  For chi2x:   W^{x, lambda^2}_{jk} = R^(k)_xx * chi2x = eta^x_Fe[k] * chi2x
  For chi5z:   W^{z, lambda^5}_{jk} = sign_57(k) * eta^z_Fe[k] * chi5z
  For chi7z:   W^{z, lambda^7}_{jk} = sign_57(k) * eta^z_Fe[k] * chi7z

─────────────────────────────────────────────────────────────────────────
II.  WHICH LAMBDA COMPONENT IS DRIVEN?
─────────────────────────────────────────────────────────────────────────

From the SU(3) structure constants, the relevant commutators are:
  [lambda^2, lambda^3] ∝ lambda^1,   [lambda^2, lambda^8] ∝ lambda^2, ...

The driven channel for chi2z is H_eff^{lambda^2}.
The equation of motion for lambda^2_j then reads:
  d/dt lambda^2_j = (1/hbar) sum_{c!=2} f^{2bc} H_eff^b_j lambda^c_j

The lambda^2 component represents the IMAGINARY PART of the off-diagonal
coherence between |E1> and |E2> in the Tm CEF eigenbasis.

Key coupling: [lambda^2, H_chi] drives time evolution of lambda^1 (the
real part of the same coherence), and lambda^1 feeds back to lambda^2
through the CEF Hamiltonian.  This is the E12 (Tm resonance) mode.

─────────────────────────────────────────────────────────────────────────
III. GROUND STATE: chi2z COUPLING
─────────────────────────────────────────────────────────────────────────

In the Gamma_2 ground state with static Fe spins:
  S^z_Fe[k] = G_z * sigma_G[k] = G_z * eta^z_Fe[k]

The chi coupling effective field at Tm site j:
  H_chi^{lambda^2}_j = sum_{k in bonds(j)} eta^z_Fe[k] * chi2z * G_z * sigma_G[k]
                     = chi2z * G_z * sum_{k} eta^z_Fe[k] * sigma_G[k]
                     = chi2z * G_z * sum_{k} sigma_G[k] * sigma_G[k]   [since eta^z_Fe = sigma_G]
                     = chi2z * G_z * 4   [since sigma_G[k]^2 = 1 for each k]

This is UNIFORM at all 4 Tm sublattices.
Pattern: H_chi^{lambda^2} = +chi2z * G_z * 4  (sigma_F, same at every site)

Tm relaxes to equilibrium lambda^2_j = f(H_eff^{lambda^2}_j), same at all sites.
Numerical value: lambda^2 ~ -0.997 * sigma_F  (from diagnostic)

─────────────────────────────────────────────────────────────────────────
IV.  qAFM KICK: WHAT CHANGES IN THE Fe SPIN STATE?
─────────────────────────────────────────────────────────────────────────

A qAFM kick (rotation of all Fe spins by delta_theta around y-axis) creates:
  S^x_Fe[k](t) = G_z * sin(delta_theta) * sigma_G[k]  +  O(F_x)
  S^z_Fe[k](t) = G_z * cos(delta_theta) * sigma_G[k]

At short times (delta_theta << 1):
  delta S^x_Fe[k] = G_z * delta_theta * sigma_G[k]    (small G_x perturbation)
  delta S^z_Fe[k] = -G_z * (1-cos(delta_theta)) * sigma_G[k] ≈ 0

So the kick introduces a G_x component: delta(G_x) = G_z * sin(delta_theta).

─────────────────────────────────────────────────────────────────────────
V.  CHI2z WITH qAFM KICK: TIME EVOLUTION OF H_chi^{lambda^2}
─────────────────────────────────────────────────────────────────────────

With the kicked state:
  H_chi^{lambda^2}_j(t) = chi2z * sum_k eta^z_Fe[k] * S^z_Fe[k](t)
                         = chi2z * G_z * cos(omega_qAFM * t)  [sum sigma_G^2 = 4]

This oscillates in amplitude, but REMAINS UNIFORM across all Tm sublattices.
The lambda^2 oscillation amplitude is the SAME at all 4 sites.

Pattern at all times:  lambda^2_j(t) = lambda^2(t) * sigma_F (same function,
same sign at all sites).

─────────────────────────────────────────────────────────────────────────
VI.  WHY M_z^Tm REMAINS ZERO: THE DETECTOR COUPLING EOM
─────────────────────────────────────────────────────────────────────────

The observable in H//z 2DCS is the z-component of the Tm magnetic moment:
  M_z^Tm = sum_j g_J * mu_B * J^z_Tm,j

In the local Tm frame, the z-axis couples to lambda^2 via the matrix element:
  J^z_j(local) = c_2 * lambda^2_j  (where c_2 is the CEF matrix element)

To convert to the global z-axis, we use the Tm local frame rotation:
  J^z_j(global) = eta^z_Tm[j] * J^z_j(local) = sigma_C[j] * c_2 * lambda^2_j

Therefore:
  M_z^Tm = c_2 * sum_j sigma_C[j] * lambda^2_j
          = c_2 * (sigma_C . lambda^2_vector)

With lambda^2_vector = lambda^2(t) * sigma_F:
  M_z^Tm(t) = c_2 * lambda^2(t) * (sigma_C . sigma_F)
             = c_2 * lambda^2(t) * 0    [since sigma_C . sigma_F = (+,-,-,+).(+,+,+,+) = 0]
             = 0

This holds at ALL times, for ANY time-dependence lambda^2(t), as long as:
  1. chi2z is the only active coupling channel, AND
  2. The Fe spin pattern preserves sigma_G at each instant (i.e., q=0 motion)

The cancellation is EXACT by the orthogonality of Bertaut vectors:
  sigma_C . sigma_F = 1 - 1 - 1 + 1 = 0

─────────────────────────────────────────────────────────────────────────
VII. WHAT sigma PATTERN OF lambda^2 WOULD BE DETECTABLE?
─────────────────────────────────────────────────────────────────────────

  M_z^Tm != 0  requires  sigma_C . lambda^2_sigma = 4  =>  lambda^2_sigma = sigma_C

The question is: which Fe spin pattern P produces sigma_C pattern lambda^2?
  H[lambda^2]_j = chi2z * sum_k sigma_G[k] * S^z_Fe[k] sigma pattern
               = chi2z * sigma_G . S_Fe_sigma

For this to give sigma_C:  sigma_G * S_sigma_Fe = sigma_C
  => S_sigma_Fe = sigma_G * sigma_C = sigma_C * sigma_G
  let's compute: sigma_G . sigma_C = (+,-,+,-) * (+,-,-,+) = (+,+,-,-) = sigma_A

So we need S^z_Fe ~ sigma_A (A-type AFM) to get sigma_C lambda^2 from chi2z.
A-type AFM is not Gamma_2. In Gamma_2, S^z_Fe ~ sigma_G (G-type).

Alternatively, using chi2y (eta^y_Fe = sigma_C):
  H[lambda^2]_j = chi2y * sigma_C[k] * S^y_Fe[k] sigma pattern
  For sigma_C lambda^2: sigma_C * S_sigma_Fe = sigma_C
  => S_sigma_Fe = sigma_F (ferromagnetic S^y)
  But in Gamma_2, S^y_Fe = 0. So chi2y is also dead.

─────────────────────────────────────────────────────────────────────────
VIII. WHAT ABOUT THE qAFM KICK FOR chi2z? THE G_x COMPONENT
─────────────────────────────────────────────────────────────────────────

After the qAFM kick, we have a G_x component:
  S^x_Fe[k] = G_x * sigma_G[k]   (sigma_G pattern!)

But chi2z only couples to S^z_Fe, not S^x_Fe. So the G_x component does
NOT enter the chi2z effective field.

What if we had chi2x? Then:
  H_chi2x^{lambda^2}_j = chi2x * eta^x_Fe[k] * S^x_Fe[k]
                        = chi2x * sigma_A[k] * G_x * sigma_G[k]
                        = chi2x * G_x * (sigma_A . sigma_G)
                        = chi2x * G_x * 0    [sigma_A . sigma_G = (+,+,-,-).( +,-,+,-) = 0]
  => Also zero!

The kick creates G_x in sigma_G pattern. chi2x has eta^x_Fe = sigma_A.
sigma_A . sigma_G = 0, so chi2x ALSO cannot be activated by a qAFM kick.

─────────────────────────────────────────────────────────────────────────
IX.  COMPLETE CHANNEL SURVEY: WHY EVERY CHI IS DEAD FOR (qAFM, E12, H//z)
─────────────────────────────────────────────────────────────────────────

For the cross-peak (omega_qAFM, omega_E12) to be nonzero under H//z:
  Need: M_z^Tm oscillates at omega_qAFM
  Need: M_z^Tm = c_2 * sigma_C . lambda^2_vector != 0
  Need: lambda^2 carries sigma_C pattern
  Need: some chi channel driven by G-type (sigma_G) S_Fe to produce sigma_C lambda^2

Test all chi2 channels for the qAFM kick (S^x ~ sigma_G, S^z ~ sigma_G):
  chi2z:  eta^z_Fe = sigma_G; sigma_G . sigma_G = sigma_F != sigma_C  FAIL
  chi2x:  eta^x_Fe = sigma_A; sigma_A . sigma_G = 0                  FAIL (zero)
  chi2y:  eta^y_Fe = sigma_C; sigma_C . sigma_G = ?
""")
    prod, name = sigma_prod(Fe_eta[:, 1], sigma_G)
    print(f'    sigma_C . sigma_G = {prod.astype(int)} = {name}')
    print("""
  chi2y: sigma_C . sigma_G = sigma_A != sigma_C  FAIL (also S^y=0 in Gamma_2)

Chi5/chi7: identically zero (sign_57 cancellation), irrelevant.

RESULT: No chi channel can produce sigma_C lambda^2 from sigma_G Fe spin
fluctuations. The cross-peak (omega_qAFM, omega_E12) is FORBIDDEN under H//z
by the sigma_C vs sigma_G orthogonality, which is enforced by the Fe local
frame structure (R_fe_frames in unitcell_builders.cpp).

─────────────────────────────────────────────────────────────────────────
X.   WHAT GEOMETRY / CHANNEL WOULD ALLOW THE CROSS-PEAK?
─────────────────────────────────────────────────────────────────────────

For M_alpha_Tm to be nonzero with chi2z:
  Need eta^alpha_Tm . sigma_F = sigma_alpha_Tm . sigma_F = 4
  => sigma_alpha_Tm = sigma_F  => need M_F_Tm (ferromagnetic Tm magnetisation)
  But M_F is the DIAGONAL magnetisation, not the off-diagonal E12 coherence.

For off-diagonal (E12) coherence (lambda^2) to be detectable:
  Need detector coupling to M_alpha with sigma_alpha_Tm = sigma_C
  Only M_z satisfies this (eta^z_Tm = sigma_C).

For M_z^Tm to see chi2z lambda^2 (which is sigma_F):
  Need sigma_C . sigma_F = 0 — this is the irremovable obstacle.

CONCLUSION: H//z 2DCS cannot detect the (qAFM, E12) cross-peak via chi2z.
The forbidden combination is:
  [chi2z source: sigma_G * sigma_G = sigma_F]
  [H//z detector: sigma_C]
  [sigma_C . sigma_F = 0]

The ONLY way to observe this cross-peak would require one of:
  (a) H//x or H//y geometry (M_x or M_y as detector):
      eta^x_Tm = sigma_A: sigma_A . sigma_F = 0  still zero
      eta^y_Tm = sigma_G: sigma_G . sigma_F = 0  still zero
      => No standard geometry works for chi2z cross-peak!
  (b) A different ordering (e.g. Gamma_4 with G_x) changing which Fe
      components are active.
  (c) Higher-order pathways (chi^2 processes, not linear chi2z) that can
      mix sigma patterns via nonlinear combinations.
  (d) A chi coupling with mixed sigma structure (e.g. chi tensor off-diagonal
      elements combining eta^z_Fe with eta^x_Tm in the bond geometry).
""")

# ============================================================================
# Bond connectivity analysis: verify which Tm connects to which Fe
# ============================================================================

def analyze_bond_connectivity():
    """
    For each Tm sublattice (0-3), list which Fe sublattices it bonds to
    across the 4 orbits (even bonds only). This is crucial for verifying
    the sigma product analysis.
    """
    print('\n' + '='*72)
    print('BOND CONNECTIVITY: which Fe connects to which Tm (even bonds)')
    print('='*72)
    print()

    # For each Tm sublattice, collect all Fe sublattices bonded to it
    tm_to_fe = {tm: [] for tm in range(4)}
    for orbit, fe_sub, tm_sub, off in even_bonds:
        tm_to_fe[tm_sub].append((orbit, fe_sub, off))

    for tm_sub in range(4):
        print(f'  Tm{tm_sub} (eta^z_Tm = {int(Tm_eta[tm_sub, 2]):+d} = sigma_C):')
        fe_subs = []
        for orbit, fe_sub, off in sorted(tm_to_fe[tm_sub]):
            eta_z_fe = Fe_eta[fe_sub, 2]
            sz_fe_sign = sigma_G[fe_sub]        # S^z_Fe ~ sigma_G[k] * G_z
            contribution = eta_z_fe * sz_fe_sign  # = sigma_G[k] * sigma_G[k] = +1
            print(f'    orbit {orbit}: Fe{fe_sub}  eta^z_Fe = {eta_z_fe:+.0f} = sigma_G[{fe_sub}]'
                  f',  S^z ~ {sz_fe_sign:+.0f}*G_z'
                  f'  =>  eta*S^z = {contribution:+.0f}*G_z  (each = +1 since sigma_G^2=1)')
            fe_subs.append(contribution)
        total = sum(fe_subs)
        print(f'    -> sum(eta^z_Fe * sigma_G_k) = {total:.0f}')
        print(f'    -> H[lambda^2]_Tm{tm_sub} = chi2z * G_z * {total:.0f}  (UNIFORM, nonzero!)')
        print()

    print("""
Note: Each Tm site is bonded to all 4 Fe sublattices (one per orbit).
sigma_G[k]^2 = 1 for each k, so the chi2z field is:
  H[lambda^2]_j = chi2z * G_z * (1+1+1+1) = 4 * chi2z * G_z  (uniform!)
This is why lambda^2 has sigma_F pattern.
""")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import os
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
    os.makedirs(out_dir, exist_ok=True)

    print('TmFeO3 chi coupling: selection rule analysis')
    print('Generated by util/analyze_chi_selection_rules.py')
    print()

    # Print text analyses
    analyze_ground_state()
    analyze_chi2y_detail()
    analyze_bond_connectivity()
    print_eom_analysis()

    # Generate figures
    print('\nGenerating 3D structure figures...')

    fig1 = make_bond_figure()
    p1 = os.path.join(out_dir, 'chi_bonds_3d_structure.pdf')
    fig1.savefig(p1, dpi=150, bbox_inches='tight')
    fig1.savefig(p1.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f'  Fig 1 saved: {p1}')

    fig2 = make_fe_frame_figure()
    p2 = os.path.join(out_dir, 'chi_fe_frames.pdf')
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    fig2.savefig(p2.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f'  Fig 2 saved: {p2}')

    fig3 = make_tm_frame_figure()
    p3 = os.path.join(out_dir, 'chi_tm_frames.pdf')
    fig3.savefig(p3, dpi=150, bbox_inches='tight')
    fig3.savefig(p3.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f'  Fig 3 saved: {p3}')

    fig4 = make_activity_table_figure()
    p4 = os.path.join(out_dir, 'chi_activity_table.pdf')
    fig4.savefig(p4, dpi=150, bbox_inches='tight')
    fig4.savefig(p4.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f'  Fig 4 saved: {p4}')

    plt.close('all')
    print('\nDone.')
