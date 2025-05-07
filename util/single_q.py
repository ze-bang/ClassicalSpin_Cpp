import numpy as np
from scipy.optimize import minimize

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
                ni = (i + delta[0])% L
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
                ni = (i + delta[0])% L
                nj = (j + delta[1]) % L
                neigh_b = 2 * (ni * L + nj)
                NNN[site_a].append(neigh_b)
                NNN[neigh_b].append(site_a)

                ni = (i + delta[0])% L
                nj = (j + delta[1]) % L
                neigh_b = 2 * (ni * L + nj) + 1
                NNN[site_a+1].append(neigh_b)
                NNN[neigh_b].append(site_a+1)

            for delta in deltasNNN:
                # Apply periodic boundaries for A
                ni = (i + delta[0])% L
                nj = (j + delta[1]) % L
                neigh_b = 2 * (ni * L + nj) + 1
                NNNN[site_a].append(neigh_b)
                NNNN[neigh_b].append(site_a)
    
    return positions, NN, NN_bonds, NNN, NNNN

def single_q_ansatz(params, positions):
    """
    Generate spin configuration based on single-q variational ansatz.
    
    Args:
        params: [qx, qy, theta, phi] - Parameters defining the ansatz
        positions: Site positions
    
    Returns:
        spins: Spin configuration (Nx3 array)
    """
    kitaevLocal_a = np.array([[0,1,-1]/np.sqrt(2),[-2,1,1]/np.sqrt(6),[1,1,1]/np.sqrt(3)])
    kitaevLocal_b = -kitaevLocal_a

    kitaevLocal = np.array([kitaevLocal_a, kitaevLocal_b])

    qx, qy, cant_a, cant_b, = params
    cants = np.array([cant_a, cant_b])
    q = np.array([qx, qy])
    N = positions.shape[0]
    spins = np.zeros((N, 3))
    
    for i in range(N):
        ex = kitaevLocal[i%2][0]
        ey = kitaevLocal[i%2][1]
        ez = kitaevLocal[i%2][2]
        cant = cants[i%2]
        # Phase at this site depends on position and q-vector
        spins[i] = np.cos(np.dot(q, positions[i])) * ex * np.sin(cant) + \
                    np.sin(np.dot(q, positions[i])) * ey * np.sin(cant) + \
                    np.cos(cant) * ez


    return spins

def compute_energy(params, positions, NN, NN_bonds, NNN, NNNN, J=[-6.54, 0.15, -3.76, 0.36, -0.21, 1.70, 0.03]):
    """Compute energy of the Heisenberg model with given parameters."""
    spins = single_q_ansatz(params, positions)
    energy = 0.0

    J1, Jpmpm, Jzp, Delta1, J2, J3, Delta3 = J
    
    
    J1x_ = np.array([[J1+2*Jpmpm*np.cos(2*np.pi/3),-2*Jpmpm*np.sin(2*np.pi/3),Jzp*np.sin(2*np.pi/3)],[-2*Jpmpm*np.sin(2*np.pi/3),J1-2*Jpmpm*np.cos(2*np.pi/3),-Jzp*np.cos(2*np.pi/3)],[Jzp*np.sin(2*np.pi/3),-Jzp*np.cos(2*np.pi/3),J1*Delta1]])
    J1y_ = np.array([[J1+2*Jpmpm*np.cos(-2*np.pi/3),-2*Jpmpm*np.sin(-2*np.pi/3),Jzp*np.sin(-2*np.pi/3)],[-2*Jpmpm*np.sin(-2*np.pi/3),J1-2*Jpmpm*np.cos(-2*np.pi/3),-Jzp*np.cos(-2*np.pi/3)],[Jzp*np.sin(-2*np.pi/3),-Jzp*np.cos(-2*np.pi/3),J1*Delta1]])
    J1z_ = np.array([[J1+2*Jpmpm*np.cos(0),-2*Jpmpm*np.sin(0),Jzp*np.sin(0)],[-2*Jpmpm*np.sin(0),J1-2*Jpmpm*np.cos(0),-Jzp*np.cos(0)],[Jzp*np.sin(0),-Jzp*np.cos(0),J1*Delta1]])

    J1_ = np.array([J1x_, J1y_, J1z_])

    J2_ = np.array([[J2,0,0],[0,J2,0],[0,0,0]])
    J3_ = np.array([[J3,0,0],[0,J3,0],[0,0,Delta3*J3]])

    # Sum over bonds (avoiding double counting)

    for i in range(len(NN)):
        for j in NN[i]:
            if j > i:  # Count each bond only once
                bondtype = NN_bonds[i][j]
                energy += np.matmul(spins[i], np.matmul(J1_[bondtype], spins[j]))

    for i in range(len(NNN)):
        for j in NNN[i]:
            if j > i:
                bondtype = NN_bonds[i][j]
                energy += np.matmul(spins[i], np.matmul(J2_, spins[j]))
    
    for i in range(len(NNNN)):
        for j in NNNN[i]:
            if j > i:
                bondtype = NN_bonds[i][j]
                energy += np.matmul(spins[i], np.matmul(J3_, spins[j]))

    return energy/len(positions)  # Normalize by number of spins

def find_ground_state(L, J=[-6.54, 0.15, -3.76, 0.36, -0.21, 1.70, 0.03], initial_params=None):
    """Find ground state by minimizing energy."""
    positions, NN, NN_bonds, NNN, NNNN  = create_honeycomb_lattice(L)
    
    if initial_params is None:
        # For antiferromagnetic case (J>0), start with q near K point
        initial_params = [2*np.pi/3, 2*np.pi/3, 2*np.pi/3, 2*np.pi/3]
    
    # Minimize energy
    result = minimize(
        compute_energy,
        initial_params,
        args=(positions, NN, NN_bonds, NNN, NNNN, J),
        method='L-BFGS-B',
        options={'disp': True}
    )
    
    optimal_params = result.x
    optimal_energy = result.fun
    optimal_spins = single_q_ansatz(optimal_params, positions)
    
    return optimal_params, optimal_energy, positions, optimal_spins

def visualize_spins(positions, spins, L):
    """Visualize spin configuration on the lattice."""
    plt.figure(figsize=(8, 8))
    
    # Plot lattice points
    plt.scatter(positions[:, 0], positions[:, 1], c='lightgray', s=50)
    
    # Plot spin projections onto xy-plane
    for i in range(len(positions)):
        plt.arrow(
            positions[i, 0], positions[i, 1],
            spins[i, 0] * 0.3, spins[i, 1] * 0.3,
            head_width=0.05, head_length=0.1, 
            fc='blue', ec='blue'
        )
    
    plt.title('Classical Ground State on Honeycomb Lattice')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(f'honeycomb_ground_state_L[L].png')
    plt.show()

if __name__ == "__main__":
    # Size of lattice (L x L unit cells)
    L = 24
    
    # Find ground state
    params, energy, positions, spins = find_ground_state(L, J=[-6.54, 0.15, -3.76, 0.36, -0.21, 1.70, 0.03])
    
    # Print results
    qx, qy, canta, cantb = params
    print(f"Optimized parameters:")
    print(f"  q-vector: ({qx:.4f}, {qy:.4f})")
    print(f"  cant A sublattice: {canta:.4f}")
    print(f"  cant B sublattice: {cantb:.4f}")
    print(f"Ground state energy: {energy:.6f}")
    

    # Visualize the ground state
    # visualize_spins(positions, spins, L)