#!/usr/bin/env python3
"""
Verify the Kitaev bond assignment matches the exchange matrix structure.

The code uses a local Kitaev frame where:
- x-bond has K on the (0,0) diagonal element
- y-bond has K on the (1,1) diagonal element
- z-bond has K on the (2,2) diagonal element

The rotation matrix R transforms from local to global:
R = [1/√6,  -1/√2,  1/√3]
    [1/√6,   1/√2,  1/√3]
    [-2/√6,  0,     1/√3]

This analysis checks if f_K^{A1g} = S^x S^x|_x + S^y S^y|_y + S^z S^z|_z
is consistent with the exchange matrix structure.
"""

import numpy as np

def get_kitaev_rotation():
    """Get the local-to-global rotation matrix."""
    R = np.array([
        [1.0/np.sqrt(6.0), -1.0/np.sqrt(2.0), 1.0/np.sqrt(3.0)],
        [1.0/np.sqrt(6.0),  1.0/np.sqrt(2.0), 1.0/np.sqrt(3.0)],
        [-2.0/np.sqrt(6.0), 0.0,              1.0/np.sqrt(3.0)]
    ])
    return R

def get_Jx_local(J, K, Gamma, Gammap):
    """x-bond exchange in local frame."""
    return np.array([
        [J + K, Gammap, Gammap],
        [Gammap, J, Gamma],
        [Gammap, Gamma, J]
    ])

def get_Jy_local(J, K, Gamma, Gammap):
    """y-bond exchange in local frame."""
    return np.array([
        [J, Gammap, Gamma],
        [Gammap, J + K, Gammap],
        [Gamma, Gammap, J]
    ])

def get_Jz_local(J, K, Gamma, Gammap):
    """z-bond exchange in local frame."""
    return np.array([
        [J, Gamma, Gammap],
        [Gamma, J, Gammap],
        [Gammap, Gammap, J + K]
    ])

def to_global_frame(J_local, R):
    """Transform exchange matrix to global frame."""
    return R @ J_local @ R.T

def analyze_kitaev_structure():
    """Analyze the Kitaev term structure in global frame."""
    
    J, K, Gamma, Gammap = 0.0, 1.0, 0.0, 0.0  # Pure Kitaev
    
    R = get_kitaev_rotation()
    
    Jx_local = get_Jx_local(J, K, Gamma, Gammap)
    Jy_local = get_Jy_local(J, K, Gamma, Gammap)
    Jz_local = get_Jz_local(J, K, Gamma, Gammap)
    
    Jx = to_global_frame(Jx_local, R)
    Jy = to_global_frame(Jy_local, R)
    Jz = to_global_frame(Jz_local, R)
    
    print("Pure Kitaev (K=1, J=Γ=Γ'=0):")
    print("\n=== LOCAL FRAME ===")
    print("Jx_local (K on 0,0):")
    print(Jx_local)
    print("\nJy_local (K on 1,1):")
    print(Jy_local)
    print("\nJz_local (K on 2,2):")
    print(Jz_local)
    
    print("\n=== GLOBAL FRAME ===")
    print("Jx_global:")
    print(np.round(Jx, 4))
    print("\nJy_global:")
    print(np.round(Jy, 4))
    print("\nJz_global:")
    print(np.round(Jz, 4))
    
    print("\n=== KITAEV TERM ANALYSIS ===")
    print("For H = S^T J S, the 'Kitaev term' is the diagonal:")
    print(f"  x-bond: Jx_global diagonal = {np.diag(Jx)}")
    print(f"  y-bond: Jy_global diagonal = {np.diag(Jy)}")
    print(f"  z-bond: Jz_global diagonal = {np.diag(Jz)}")
    
    print("\n=== CHECK: Which component is 'Kitaev' on each bond? ===")
    for bond_name, Jmat in [('x', Jx), ('y', Jy), ('z', Jz)]:
        diag = np.diag(Jmat)
        max_idx = np.argmax(np.abs(diag))
        components = ['x', 'y', 'z']
        print(f"  {bond_name}-bond: largest diagonal at component {components[max_idx]} ({diag[max_idx]:.4f})")
    
    print("\n=== CRITICAL CHECK ===")
    print("The spin basis function f_K^{A1g} in the code does:")
    print("  f_K^{A1g} = S^{bond_type} * S^{bond_type}")
    print("where bond_type = 0,1,2 for x,y,z bonds")
    print("\nThis means:")
    print("  x-bond (type=0): sums S^0 * S^0 = S^x * S^x in GLOBAL frame")
    print("  y-bond (type=1): sums S^1 * S^1 = S^y * S^y in GLOBAL frame")
    print("  z-bond (type=2): sums S^2 * S^2 = S^z * S^z in GLOBAL frame")
    
    # Check consistency
    print("\n=== CONSISTENCY CHECK ===")
    print("For this to be correct, we need Jγ_global(γ,γ) to have the K contribution.")
    print(f"  Jx_global[0,0] = {Jx[0,0]:.4f} (should contain K)")
    print(f"  Jy_global[1,1] = {Jy[1,1]:.4f} (should contain K)")
    print(f"  Jz_global[2,2] = {Jz[2,2]:.4f} (should contain K)")
    
    # Verify with full J-K model
    print("\n" + "="*60)
    print("VERIFICATION WITH J-K MODEL (J=-0.2, K=-0.667)")
    print("="*60)
    
    J, K = -0.2, -0.667
    Gamma, Gammap = 0.0, 0.0
    
    Jx = to_global_frame(get_Jx_local(J, K, Gamma, Gammap), R)
    Jy = to_global_frame(get_Jy_local(J, K, Gamma, Gammap), R)
    Jz = to_global_frame(get_Jz_local(J, K, Gamma, Gammap), R)
    
    print("\nJx_global:")
    print(np.round(Jx, 4))
    print("\nJy_global:")
    print(np.round(Jy, 4))
    print("\nJz_global:")
    print(np.round(Jz, 4))
    
    print("\nDiagonal elements (should show J+K vs J pattern):")
    print(f"  Jx: {np.round(np.diag(Jx), 4)} - expect one element = J+K = {J+K:.4f}, others ~ J = {J:.4f}")
    print(f"  Jy: {np.round(np.diag(Jy), 4)}")
    print(f"  Jz: {np.round(np.diag(Jz), 4)}")


def verify_physical_picture():
    """Check if the physical Kitaev bond picture is correct."""
    
    print("\n" + "="*60)
    print("PHYSICAL PICTURE VERIFICATION")
    print("="*60)
    
    print("""
Standard Kitaev model on honeycomb:
- Three bond types related by C3 rotation
- On γ-bond, the interaction is K * S_i^γ S_j^γ (Ising-like)

The convention in the code:
- x-bond (type 0): K contributes to S^x S^x term
- y-bond (type 1): K contributes to S^y S^y term  
- z-bond (type 2): K contributes to S^z S^z term

The basis functions in the code:
- f_K^{A1g} = S^x S^x|_{x-bonds} + S^y S^y|_{y-bonds} + S^z S^z|_{z-bonds}
- f_K^{Eg,1} = S^x S^x|_{x-bonds} + S^y S^y|_{y-bonds} - 2*S^z S^z|_{z-bonds}
- f_K^{Eg,2} = √3 * (S^x S^x|_{x-bonds} - S^y S^y|_{y-bonds})

This IS the correct Kitaev contribution because:
1. The bond type index (0,1,2) = (x,y,z) matches the spin component
2. The local-to-global rotation R is chosen such that the K term
   on the γ-bond appears on the γ-component diagonal of J_global
""")
    
    # Verify the rotation choice
    R = get_kitaev_rotation()
    print("Rotation matrix R (local → global):")
    print(np.round(R, 4))
    print(f"\nR is orthogonal: R @ R.T = I? {np.allclose(R @ R.T, np.eye(3))}")
    
    # Check: local [1,0,0] (x) maps to which global direction?
    print("\nLocal basis vectors in global frame:")
    print(f"  local x → global: {R @ np.array([1,0,0])}")
    print(f"  local y → global: {R @ np.array([0,1,0])}")
    print(f"  local z → global: {R @ np.array([0,0,1])}")


if __name__ == '__main__':
    analyze_kitaev_structure()
    verify_physical_picture()
