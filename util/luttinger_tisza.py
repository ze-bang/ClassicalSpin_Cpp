import numpy as np
from scipy.linalg import eigh
from itertools import product
from scipy.optimize import minimize

# filepath: /home/pc_linux/ClassicalSpin_Cpp/util/luttinger_tizsza.py
import matplotlib.pyplot as plt

def create_honeycomb_lattice(L):
    """Create a honeycomb lattice with L x L unit cells."""
    # Each unit cell contains 2 sites (A and B)
    N = 2 * L * L
    positions = np.zeros((N, 2))
    NN = [[] for _ in range(N)]
    NN_bonds = np.zeros((N, N), dtype=int)
    NNN = [[] for _ in range(N)]
    NNNN = [[] for _ in range(N)]

    # Lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
        
    # Nearest neighbor vectors from A site
    delta1 = np.array([0, 0])  # Z bond
    delta2 = np.array([0, -1]) # X bond
    delta3 = np.array([1,-1])  # Y bond
    deltas = [delta1, delta2, delta3]

    # Second nearest neighbor vectors from A site
    deltaNN1 = np.array([1, 0])  
    deltaNN2 = np.array([0, 1])  
    deltaNN3 = np.array([1, -1]) 
    deltaNN4 = np.array([-1, 0])  
    deltaNN5 = np.array([0, -1])  
    deltaNN6 = np.array([-1, 1]) 
    deltasNN = [deltaNN1, deltaNN2, deltaNN3, deltaNN4, deltaNN5, deltaNN6]

    # Third nearest neighbor vectors from A site
    deltaNNN1 = np.array([1, 0])
    deltaNNN2 = np.array([-1, 0])
    deltaNNN3 = np.array([1, -2])
    deltasNNN = [deltaNNN1, deltaNNN2, deltaNNN3]
    
    for i in range(L):
        for j in range(L):
            # Compute unit cell position
            cell_pos = i * a1 + j * a2
            
            # A site (even index)
            site_a = 2 * (i * L + j)
            positions[site_a] = cell_pos
            
            # B site (odd index)
            site_b = site_a + 1
            positions[site_b] = cell_pos + np.array([0, 1/np.sqrt(3)])
            
            # Connect A site to its three neighboring B sites
            for delta in deltas:
                # Apply periodic boundaries
                ni = (i + delta[0]) % L
                nj = (j + delta[1]) % L
                neigh_b = 2 * (ni * L + nj) + 1
                NN[site_a].append(neigh_b)
                NN[neigh_b].append(site_a)
                bondtype = 0
                if (delta==delta1).all():
                    bondtype = 2
                elif (delta==delta2).all(): 
                    bondtype = 0
                elif (delta==delta3).all():
                    bondtype = 1

                NN_bonds[site_a, neigh_b] = bondtype
                NN_bonds[neigh_b, site_a] = bondtype

            for delta in deltasNN:
                # Apply periodic boundaries for A
                ni = (i + delta[0]) % L
                nj = (j + delta[1]) % L
                neigh_b = 2 * (ni * L + nj)
                NNN[site_a].append(neigh_b)
                NNN[neigh_b].append(site_a)

                ni = (i + delta[0]) % L
                nj = (j + delta[1]) % L
                neigh_b = 2 * (ni * L + nj) + 1
                NNN[site_a+1].append(neigh_b)
                NNN[neigh_b].append(site_a+1)

            for delta in deltasNNN:
                # Apply periodic boundaries for A
                ni = (i + delta[0]) % L
                nj = (j + delta[1]) % L
                neigh_b = 2 * (ni * L + nj) + 1
                NNNN[site_a].append(neigh_b)
                NNNN[neigh_b].append(site_a)
    
    return positions, NN, NN_bonds, NNN, NNNN, a1, a2

def construct_interaction_matrices(J=[-6.54, 0.15, -3.76, 0.36, -0.21, 1.70, 0.03]):
    """Construct interaction matrices for different bond types."""
    J1, Jpmpm, Jzp, Delta1, J2, J3, Delta3 = J
    
    # Nearest-neighbor interactions
    J1x = np.array([
        [J1+2*Jpmpm*np.cos(2*np.pi/3), -2*Jpmpm*np.sin(2*np.pi/3), Jzp*np.sin(2*np.pi/3)],
        [-2*Jpmpm*np.sin(2*np.pi/3), J1-2*Jpmpm*np.cos(2*np.pi/3), -Jzp*np.cos(2*np.pi/3)],
        [Jzp*np.sin(2*np.pi/3), -Jzp*np.cos(2*np.pi/3), J1*Delta1]
    ])
    
    J1y = np.array([
        [J1+2*Jpmpm*np.cos(-2*np.pi/3), -2*Jpmpm*np.sin(-2*np.pi/3), Jzp*np.sin(-2*np.pi/3)],
        [-2*Jpmpm*np.sin(-2*np.pi/3), J1-2*Jpmpm*np.cos(-2*np.pi/3), -Jzp*np.cos(-2*np.pi/3)],
        [Jzp*np.sin(-2*np.pi/3), -Jzp*np.cos(-2*np.pi/3), J1*Delta1]
    ])
    
    J1z = np.array([
        [J1+2*Jpmpm*np.cos(0), -2*Jpmpm*np.sin(0), Jzp*np.sin(0)],
        [-2*Jpmpm*np.sin(0), J1-2*Jpmpm*np.cos(0), -Jzp*np.cos(0)],
        [Jzp*np.sin(0), -Jzp*np.cos(0), J1*Delta1]
    ])
    
    J1 = [J1x, J1y, J1z]
    
    # Second-neighbor interactions
    J2_mat = np.array([
        [J2, 0, 0],
        [0, J2, 0],
        [0, 0, J2]
    ])
    
    # Third-neighbor interactions
    J3_mat = np.array([
        [J3, 0, 0],
        [0, J3, 0],
        [0, 0, Delta3*J3]
    ])
    
    return J1, J2_mat, J3_mat

def get_bond_vectors(a1, a2):
    """Get bond vectors for different neighbor types."""
    # A to B site vectors (nearest neighbors)
    delta3 = np.array([0, 1/np.sqrt(3)])           # Z bond
    delta1 = -a2 + delta3                          # X bond
    delta2 = a1 - a2 + delta3                      # Y bond
    nn_vectors = [delta1, delta2, delta3]
    
    # Second nearest neighbor vectors (A to A or B to B)
    nnn_vectors = [a1, a2, a1-a2, -a1, -a2, a2-a1]
    
    # Third nearest neighbor vectors (A to B)
    nnnn_vectors = [a1 + delta3, -a1 + delta3, a1-2*a2 + delta3]
    
    return nn_vectors, nnn_vectors, nnnn_vectors

def fourier_transform_interactions(k, a1, a2, J=[-6.54, 0.15, -3.76, 0.36, -0.21, 1.70, 0.03]):
    """Compute the Fourier transformed interaction matrix J(k)."""
    J1, J2_mat, J3_mat = construct_interaction_matrices(J)
    nn_vectors, nnn_vectors, nnnn_vectors = get_bond_vectors(a1, a2)
    
    # Initialize J(k) matrix for a 2-sublattice system with 3 spin components
    J_k = np.zeros((6, 6), dtype=complex)
    
    # J(k) is a 2x2 block matrix, each block is 3x3
    # [JAA(k), JAB(k)]
    # [JBA(k), JBB(k)]
    
    # A-B interactions (nearest neighbor)
    JAB = np.zeros((3, 3), dtype=complex)
    for i, delta in enumerate(nn_vectors):
        phase = np.exp(1j * np.dot(k, delta))
        JAB += J1[i] * phase
    
    # B-A interactions
    JBA = np.conjugate(JAB.T)
    
    # A-A and B-B interactions (second nearest neighbors)
    JAA = np.zeros((3, 3), dtype=complex)
    JBB = np.zeros((3, 3), dtype=complex)
    for delta in nnn_vectors:
        phase = np.exp(1j * np.dot(k, delta))
        JAA += J2_mat * phase
        JBB += J2_mat * phase
    
    # A-B interactions (third nearest neighbors)
    for delta in nnnn_vectors:
        phase = np.exp(1j * np.dot(k, delta))
        JAB += J3_mat * phase
        JBA += J3_mat * np.conjugate(phase)
    
    # Construct the full J(k) matrix
    J_k[:3, :3] = JAA
    J_k[:3, 3:] = JAB
    J_k[3:, :3] = JBA
    J_k[3:, 3:] = JBB
    
    return J_k

def luttinger_tisza_method(L, J=[-6.54, 0.15, -3.76, 0.36, -0.21, 1.70, 0.03]):
    """
    Implement the Luttinger-Tisza method to find the ground state.
    
    Args:
        L: Size of the lattice
        J: Exchange coupling parameters
        k_mesh_density: Number of k-points along each direction in the Brillouin zone
    """
    positions, NN, NN_bonds, NNN, NNNN, a1, a2 = create_honeycomb_lattice(L)
    
    # Reciprocal lattice vectors
    # b1 · a1 = 2π, b1 · a2 = 0
    # b2 · a1 = 0, b2 · a2 = 2π
    b1 = 2 * np.pi * np.array([1, -1/np.sqrt(3)])
    b2 = 2 * np.pi * np.array([0, 2/np.sqrt(3)])
    # Define an objective function that returns the minimum eigenvalue for a given k
    def objective_function(k_params):
        k = k_params[0] * b1 + k_params[1] * b2
        J_k = fourier_transform_interactions(k, a1, a2, J)
        eigenvalues = eigh(J_k, eigvals_only=True)
        return eigenvalues[0]  # Return the smallest eigenvalue
    
    # Initial guess (center of Brillouin zone)
    initial_guess = np.array([0.3, 0.3])
    
    # Use scipy's minimize function to find the minimum eigenvalue
    result = minimize(objective_function, initial_guess, 
                    method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
    
    # Get the optimized k-point
    opt_k_params = result.x
    min_k = opt_k_params[0] * b1 + opt_k_params[1] * b2
    
    # Get the corresponding eigenvector
    J_k = fourier_transform_interactions(min_k, a1, a2, J)
    min_eigenvalue, eigenvectors = eigh(J_k)
    min_eigenvector = eigenvectors[:, 0]
    
    # Ground state energy per site
    ground_state_energy = min_eigenvalue[0] / 2  # Factor of 1/2 from Hamiltonian definition
    # Generate the real-space spin configuration
    spins = reconstruct_spin_configuration(min_k, min_eigenvector, positions)
    
    return np.array([opt_k_params[0], opt_k_params[1]]), ground_state_energy, spins, positions

def reconstruct_spin_configuration(k, eigenvector, positions):
    """
    Reconstruct the real-space spin configuration from the k-space eigenvector.
    
    Args:
        k: Optimal k-vector
        eigenvector: Corresponding eigenvector from Luttinger-Tisza method
        positions: Real-space positions of spins
    
    Returns:
        spins: Real-space spin configuration
    """
    N = positions.shape[0]
    spins = np.zeros((N, 3))
    
    # The eigenvector has 6 components: 3 for sublattice A and 3 for sublattice B
    # Extract the spin directions for each sublattice
    S_A = eigenvector[:3]
    S_B = eigenvector[3:]
    
    # Normalize to ensure unit length spins
    S_A = S_A / np.linalg.norm(S_A)
    S_B = S_B / np.linalg.norm(S_B)
    
    # Reconstruct the real space configuration
    for i in range(N):
        sublattice = i % 2
        if sublattice == 0:  # A sublattice
            phase = np.exp(1j * np.dot(k, positions[i]))
            S_i = phase * S_A
        else:  # B sublattice
            phase = np.exp(1j * np.dot(k, positions[i]))
            S_i = phase * S_B
        
        # Take the real part for physical spins
        spins[i] = np.real(S_i)
        
        # Normalize each spin
        spins[i] = spins[i] / np.linalg.norm(spins[i])
    
    return spins

def visualize_spins(positions, spins, L, save=False, filename=None):
    """Visualize the spin configuration"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Separate sublattice A and B
    pos_A = positions[::2]
    pos_B = positions[1::2]
    spins_A = spins[::2]
    spins_B = spins[1::2]
    
    # Plot lattice sites
    ax.scatter(pos_A[:, 0], pos_A[:, 1], color='black', s=5)
    ax.scatter(pos_B[:, 0], pos_B[:, 1], color='black', s=5)
    
    # Plot the spins
    ax.quiver(pos_A[:, 0], pos_A[:, 1], spins_A[:, 0], spins_A[:, 1], color='red', pivot='mid')
    ax.quiver(pos_B[:, 0], pos_B[:, 1], spins_B[:, 0], spins_B[:, 1], color='blue', pivot='mid')
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Spin Configuration for {L}x{L} Lattice")
    
    if save and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    elif save:
        plt.savefig('spin_configuration.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, ax

if __name__ == "__main__":
    # Size of lattice (L x L unit cells)
    L = 12  # Smaller for faster computation
    
    # J parameters: [J1, Jpmpm, Jzp, Delta1, J2, J3, Delta3]
    J = [-6.54, 0.15, -3.76, 0.36, -0.21, 1.9, 0.03]
    
    # Find ground state using Luttinger-Tisza method
    k_opt, energy, spins, positions = luttinger_tisza_method(L, J)
    
    # Print results
    print(f"Optimal k-vector: ({k_opt[0]:.4f}, {k_opt[1]:.4f})")
    print(f"Ground state energy per site: {energy:.6f}")
    
    # Visualize the ground state
    visualize_spins(positions, spins, L)