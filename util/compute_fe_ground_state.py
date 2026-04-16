#!/usr/bin/env python3
"""
Compute exact classical ground state of Fe-only TmFeO3.
Uses scipy optimization constrained to |S_i| = S for each sublattice.
"""
import numpy as np
from scipy.optimize import minimize

# Parameters (from tuned param file)
S = 2.5
J1ab = 4.74
J1c  = 5.15
J2ab = 0.15
J2c  = 0.30
Ka   = -0.02545
Kb   = 0.0
Kc   = -0.029
D1   = 0.050
D2   = 0.0

# Sublattice frames η
frames = [
    np.diag([1, 1, 1]),       # sub 0: (+,+,+)
    np.diag([1, -1, -1]),     # sub 1: (+,-,-)
    np.diag([-1, 1, -1]),     # sub 2: (-,+,-)
    np.diag([-1, -1, 1]),     # sub 3: (-,-,+)
]

def to_global(i, s_local):
    """Transform local-frame spin to global frame."""
    return frames[i] @ s_local

def to_local(i, s_global):
    """Transform global-frame spin to local frame (frame is its own inverse for diagonal)."""
    return frames[i] @ s_global

def angles_to_spins(angles):
    """Convert 4 pairs of (theta, phi) to 4 spins on the sphere with |S|=S."""
    spins = np.zeros((4, 3))
    for i in range(4):
        th = angles[2*i]
        ph = angles[2*i+1]
        spins[i] = S * np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
    return spins

def compute_energy(spins_local):
    """Compute total energy for a 1x1x1 cell.
    spins_local[i] = local-frame spin of sublattice i.
    """
    # Get global-frame spins
    sg = np.array([to_global(i, spins_local[i]) for i in range(4)])
    
    E = 0.0
    
    # Single-ion anisotropy: E = Σ_i [Ka*Sx_i^2 + Kb*Sy_i^2 + Kc*Sz_i^2] (global frame)
    for i in range(4):
        E += Ka * sg[i,0]**2 + Kb * sg[i,1]**2 + Kc * sg[i,2]**2
    
    # Exchange interactions
    # The exchange matrices in local frame are:
    # J_local[i][j][a][b] = J_orig[a][b] * η_i^a * η_j^b
    # The energy for bond i->j is: E = S_i_local · J_local[i][j] · S_j_local
    # which equals (in global frame): S_i_global · J_orig · S_j_global
    # (because the η factors cancel: η_i^a * S_i_local^a = S_i_global^a)
    
    # NN bonds (from build_tmfeo3_fe):
    # Bonds 1→0: J_orig = {{J, D2, -D1}, {-D2, J, 0}, {D1, 0, J}}
    # Bonds 2→3: J_orig_alt = {{J, -D2, D1}, {D2, J, 0}, {-D1, 0, J}}
    
    J1_10 = np.array([
        [J1ab, D2, -D1],
        [-D2, J1ab, 0],
        [D1, 0, J1ab]
    ])
    J1_23 = np.array([
        [J1ab, -D2, D1],
        [D2, J1ab, 0],
        [-D1, 0, J1ab]
    ])
    
    # In-plane NN: each bond appears 4 times in 1x1x1 (due to PBC)
    # Bond types 1→0: 4 in-plane bonds
    # Bond types 2→3: 4 in-plane bonds
    E += 4 * sg[1] @ J1_10 @ sg[0]
    E += 4 * sg[2] @ J1_23 @ sg[3]
    
    # c-axis NN: bonds between (0,1)↔(2,3)
    # From build_tmfeo3_fe: c-axis bonds use J1c (scalar exchange, no DM)
    # Bond: sub0→sub2 and sub1→sub3, with 2 bonds each
    J1c_mat = J1c * np.eye(3)
    E += 2 * sg[0] @ J1c_mat @ sg[2]
    E += 2 * sg[1] @ J1c_mat @ sg[3]
    
    # NNN bonds (J2): same-sublattice with 12 bonds per sublattice
    # and cross-sublattice with 16 bonds. Using isotropic J2.
    # Same sublattice: 0↔0 (12 bonds, but in 1x1x1 this is self-loops → 0 contribution)
    # Actually in 1x1x1, same-sublattice NNN wraps around. With 12 NNN per sublattice:
    # In 1x1x1 PBC: each atom's same-sublattice NNN all map to itself → E_NNN_same = 12*J2*S·S per atom
    for i in range(4):
        E += 12 * J2ab * np.dot(sg[i], sg[i]) # This is a constant
    
    # Cross-sublattice NNN: 16 bonds connecting different sublattice pairs
    # In 1x1x1, these connect e.g. sub0↔sub1, sub0↔sub3, etc. with J2c
    # 4 bonds per unique pair
    for i in range(4):
        for j in range(4):
            if i != j:
                E += J2c * np.dot(sg[i], sg[j])  # approximate: 1 bond per pair in 1x1x1
    
    return E

def energy_from_angles(angles):
    spins = angles_to_spins(angles)
    return compute_energy(spins)

# Start from approximate Γ2 state
# Sub 0: local n pointing in (-x, 0, -z) → global (-x, 0, -z)
# Sub 1: local n pointing in (+x, 0, -z) → global (+x, 0, +z) 
# Sub 2: local n pointing in (+x, 0, +z) → global (-x, 0, -z)
# Sub 3: local n pointing in (-x, 0, +z) → global (+x, 0, +z)
# Wait, this doesn't match. Let me start from the SA result.

sa_spins = np.array([
    [-2.6045303918115670e-01, 1.2157018954879471e-10, -2.4863958281780678e+00],
    [ 2.4339321579977014e-01, 1.2132555646135138e-10, -2.4881237393873015e+00],
    [ 2.6031487439265599e-01, 1.2101504017480789e-10,  2.4864102972297095e+00],
    [-2.4339311301410105e-01, 1.2132556365852411e-10,  2.4881237494419977e+00]
])

# Convert to angles
init_angles = []
for i in range(4):
    s = sa_spins[i]
    r = np.linalg.norm(s)
    th = np.arccos(np.clip(s[2]/r, -1, 1))
    ph = np.arctan2(s[1], s[0])
    init_angles.extend([th, ph])
init_angles = np.array(init_angles)

E_sa = energy_from_angles(init_angles)
print(f"SA seed energy: {E_sa:.6f} meV (E/N = {E_sa/4:.6f})")

# Optimize
result = minimize(energy_from_angles, init_angles, method='Nelder-Mead', 
                  options={'maxiter': 100000, 'xatol': 1e-12, 'fatol': 1e-12})
print(f"Optimized energy: {result.fun:.6f} meV (E/N = {result.fun/4:.6f})")

opt_spins = angles_to_spins(result.x)
print(f"\nOptimized local-frame spins:")
for i in range(4):
    print(f"  Sub {i}: ({opt_spins[i,0]:.10f}, {opt_spins[i,1]:.10f}, {opt_spins[i,2]:.10f})  |S|={np.linalg.norm(opt_spins[i]):.6f}")

# Compute G, F
G_opt = np.zeros(3); F_opt = np.zeros(3)
signs = [1,-1,1,-1]
for i in range(4):
    sg = to_global(i, opt_spins[i])
    G_opt += signs[i] * sg
    F_opt += sg
G_opt /= 4; F_opt /= 4
print(f"\nG = ({G_opt[0]:.8f}, {G_opt[1]:.8f}, {G_opt[2]:.8f}), |G|={np.linalg.norm(G_opt):.6f}")
print(f"F = ({F_opt[0]:.8f}, {F_opt[1]:.8f}, {F_opt[2]:.8f}), |F|={np.linalg.norm(F_opt):.8f}")

if abs(G_opt[2]) > 0.01:
    cant = np.degrees(np.arctan2(abs(G_opt[0]), abs(G_opt[2])))
    print(f"Cant angle: {cant:.4f} deg")

# Save seed
seed_path = 'example_configs/TmFeO3/fe_gamma2_seed_exact.txt'
np.savetxt(seed_path, opt_spins, fmt='%.16e')
print(f"\nSeed saved to: {seed_path}")
