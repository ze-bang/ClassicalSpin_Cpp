#!/usr/bin/env python3
"""
Analyze the CORRECT Kitaev basis functions for D3d magnetoelastic coupling.

The issue: The code uses bond_type (0,1,2) as spin component index,
but the spins are in the GLOBAL frame while the Kitaev structure is
defined in a LOCAL frame related by rotation R.

Two possible interpretations:
1. The basis functions should use LOCAL spin components (R^T @ S)
2. The basis functions should be redefined for GLOBAL frame

The correct physics requires that f_K^{A1g} captures the Kitaev exchange
energy, which in the code is computed as S_i^T @ J_global @ S_j.
"""

import numpy as np

def get_kitaev_rotation():
    """Local-to-global rotation matrix."""
    R = np.array([
        [1.0/np.sqrt(6.0), -1.0/np.sqrt(2.0), 1.0/np.sqrt(3.0)],
        [1.0/np.sqrt(6.0),  1.0/np.sqrt(2.0), 1.0/np.sqrt(3.0)],
        [-2.0/np.sqrt(6.0), 0.0,              1.0/np.sqrt(3.0)]
    ])
    return R

def get_J_local(J, K, Gamma, Gammap, bond_type):
    """Get exchange matrix in local frame for given bond type."""
    if bond_type == 0:  # x-bond
        return np.array([
            [J + K, Gammap, Gammap],
            [Gammap, J, Gamma],
            [Gammap, Gamma, J]
        ])
    elif bond_type == 1:  # y-bond
        return np.array([
            [J, Gammap, Gamma],
            [Gammap, J + K, Gammap],
            [Gamma, Gammap, J]
        ])
    else:  # z-bond
        return np.array([
            [J, Gamma, Gammap],
            [Gamma, J, Gammap],
            [Gammap, Gammap, J + K]
        ])

def analyze_kitaev_basis():
    """
    The Kitaev Hamiltonian in local frame is:
    H_K = K * sum_{<ij>_x} S_i^{local,0} S_j^{local,0}
        + K * sum_{<ij>_y} S_i^{local,1} S_j^{local,1}
        + K * sum_{<ij>_z} S_i^{local,2} S_j^{local,2}
    
    In global frame, S^{local} = R^T @ S^{global}, so:
    S_i^{local,γ} S_j^{local,γ} = (R^T @ S_i)_γ (R^T @ S_j)_γ
                                = (R_{αγ} S_i^α)(R_{βγ} S_j^β)
                                = R_{αγ} R_{βγ} S_i^α S_j^β
    
    This is NOT simply S_i^γ S_j^γ in the global frame!
    """
    
    R = get_kitaev_rotation()
    print("Rotation matrix R (local → global):")
    print(np.round(R, 4))
    print()
    
    print("=== CORRECT KITAEV BASIS FUNCTIONS ===")
    print()
    print("In the LOCAL frame, the Kitaev term on γ-bond is:")
    print("  K * S_i^{local,γ} S_j^{local,γ}")
    print()
    print("Converting to GLOBAL frame using S^{local} = R^T @ S^{global}:")
    print("  K * (R^T @ S_i)_γ * (R^T @ S_j)_γ")
    print("  = K * Σ_{αβ} R_{αγ} R_{βγ} S_i^α S_j^β")
    print()
    
    # Compute the effective "Kitaev projector" for each bond type
    for gamma, bond_name in enumerate(['x', 'y', 'z']):
        # The projector P_γ satisfies: S_i^{local,γ} S_j^{local,γ} = S_i^T P_γ S_j
        # P_γ = R[:, γ] @ R[:, γ].T = outer product of γ-th column of R
        P = np.outer(R[:, gamma], R[:, gamma])
        
        print(f"{bond_name}-bond (γ={gamma}):")
        print(f"  Projector P_{bond_name} = R[:, {gamma}] ⊗ R[:, {gamma}]:")
        print(f"  {np.round(P, 4)}")
        print(f"  Trace = {np.trace(P):.4f} (should be 1)")
        print()
    
    print("=== WHAT THE CODE CURRENTLY DOES ===")
    print()
    print("f_K^{A1g} = S^0 S^0|_{x-bond} + S^1 S^1|_{y-bond} + S^2 S^2|_{z-bond}")
    print("         = S^x S^x|_{x-bond} + S^y S^y|_{y-bond} + S^z S^z|_{z-bond}")
    print()
    print("This uses the IDENTITY projector P = e_γ ⊗ e_γ instead of R[:, γ] ⊗ R[:, γ]")
    print()
    
    print("=== COMPARISON ===")
    print()
    
    # Compare the two approaches for A1g
    P_correct_sum = sum(np.outer(R[:, g], R[:, g]) for g in range(3))
    P_code_sum = np.eye(3)  # sum of e_γ ⊗ e_γ = I
    
    print("Sum of projectors for A1g (should be identity for orthonormal basis):")
    print(f"  Correct: Σ_γ R[:, γ] ⊗ R[:, γ] = R @ R^T = ")
    print(f"  {np.round(P_correct_sum, 4)}")
    print(f"  Code:    Σ_γ e_γ ⊗ e_γ = I = ")
    print(f"  {P_code_sum}")
    print()
    print("They are the same! This means f_K^{A1g} IS correct because:")
    print("  Σ_γ (R^T S)_γ (R^T S')_γ = (R^T S)^T (R^T S') = S^T R R^T S' = S^T S' = S·S'")
    print("  Σ_γ S^γ S'^γ = S·S'")
    print()
    print("HOWEVER, the Eg basis functions are NOT the same!")
    
    # Check Eg1 
    print()
    print("=== Eg,1 ANALYSIS ===")
    
    # Correct Eg,1: P_x + P_y - 2*P_z where P_γ = R[:, γ] ⊗ R[:, γ]
    P_correct_Eg1 = (np.outer(R[:, 0], R[:, 0]) + np.outer(R[:, 1], R[:, 1]) 
                    - 2*np.outer(R[:, 2], R[:, 2]))
    
    # Code Eg,1: e_x ⊗ e_x + e_y ⊗ e_y - 2*e_z ⊗ e_z
    P_code_Eg1 = np.diag([1, 1, -2])
    
    print("Correct Eg,1 projector:")
    print(np.round(P_correct_Eg1, 4))
    print()
    print("Code Eg,1 projector (diag[1,1,-2]):")
    print(P_code_Eg1)
    print()
    print(f"Are they equal? {np.allclose(P_correct_Eg1, P_code_Eg1)}")
    
    # Check Eg2
    print()
    print("=== Eg,2 ANALYSIS ===")
    
    # Correct Eg,2: √3 * (P_x - P_y)
    P_correct_Eg2 = np.sqrt(3) * (np.outer(R[:, 0], R[:, 0]) - np.outer(R[:, 1], R[:, 1]))
    
    # Code Eg,2: √3 * (e_x ⊗ e_x - e_y ⊗ e_y)
    P_code_Eg2 = np.sqrt(3) * np.diag([1, -1, 0])
    
    print("Correct Eg,2 projector:")
    print(np.round(P_correct_Eg2, 4))
    print()
    print("Code Eg,2 projector (√3 * diag[1,-1,0]):")
    print(np.round(P_code_Eg2, 4))
    print()
    print(f"Are they equal? {np.allclose(P_correct_Eg2, P_code_Eg2)}")


def correct_basis_functions():
    """Show the CORRECT basis function expressions."""
    
    R = get_kitaev_rotation()
    
    print("\n" + "="*70)
    print("CORRECT KITAEV BASIS FUNCTIONS IN GLOBAL FRAME")
    print("="*70)
    print()
    print("Let S^{(γ)} = S · R[:, γ] be the projection onto local axis γ")
    print()
    print("Then on γ-bond, the Kitaev term is K * S_i^{(γ)} * S_j^{(γ)}")
    print()
    print("The columns of R (local basis vectors in global frame) are:")
    for g, name in enumerate(['x', 'y', 'z']):
        col = R[:, g]
        print(f"  local {name}: {np.round(col, 4)}")
    print()
    
    # These are the 'Kitaev axes' in global frame
    print("So the CORRECT Kitaev components are:")
    print("  S^{(x)} = S · [0.408, 0.408, -0.816] = (S^x + S^y - 2S^z)/√6")
    print("  S^{(y)} = S · [-0.707, 0.707, 0] = (-S^x + S^y)/√2")
    print("  S^{(z)} = S · [0.577, 0.577, 0.577] = (S^x + S^y + S^z)/√3")
    print()
    print("The correct f_K^{Eg,1} would be:")
    print("  f_K^{Eg,1} = S^{(x)} S^{(x)} + S^{(y)} S^{(y)} - 2 S^{(z)} S^{(z)}")
    print()
    print("NOT the code's version:")
    print("  f_K^{Eg,1} = S^x S^x + S^y S^y - 2 S^z S^z")


if __name__ == '__main__':
    analyze_kitaev_basis()
    correct_basis_functions()
