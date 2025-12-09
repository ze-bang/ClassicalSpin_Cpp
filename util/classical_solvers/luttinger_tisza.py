import numpy as np
from scipy.linalg import eigh
from itertools import product
from scipy.optimize import minimize

# filepath: /home/pc_linux/ClassicalSpin_Cpp/util/luttinger_tizsza.py
import matplotlib.pyplot as plt

def check_luttinger_tisza_constraints(spins, positions, NN, NN_bonds, NNN, NNNN, 
                                    k_opt, eigenvector, J=[-6.54, 0.15, -3.76, 0.36, -0.21, 1.70, 0.03],
                                    B_field=None, tolerance=1e-6):
    """
    Check if the solution satisfies the Luttinger-Tisza constraints.
    
    Args:
        spins: Real-space spin configuration
        positions: Positions of lattice sites
        NN, NN_bonds, NNN, NNNN: Neighbor lists
        k_opt: Optimal k-vector from LT method
        eigenvector: Corresponding eigenvector
        J: Exchange parameters
        B_field: External magnetic field vector (3D array)
        tolerance: Numerical tolerance for constraint checking
        
    Returns:
        constraints_satisfied: Dictionary with constraint results
    """
    constraints = {}
    
    # 1. Check spin length constraint |S_i| = 1
    spin_lengths = np.linalg.norm(spins, axis=1)
    unit_length_satisfied = np.all(np.abs(spin_lengths - 1.0) < tolerance)
    max_deviation_length = np.max(np.abs(spin_lengths - 1.0))
    
    constraints['unit_length'] = {
        'satisfied': unit_length_satisfied,
        'max_deviation': max_deviation_length,
        'all_lengths': spin_lengths
    }
    
    # 2. Calculate energies (exchange, Zeeman, and total)
    total_energy, exchange_energy, zeeman_energy = calculate_total_energy(spins, NN, NN_bonds, NNN, NNNN, J, B_field)
    
    constraints['exchange_energy'] = exchange_energy
    constraints['zeeman_energy'] = zeeman_energy
    constraints['total_energy'] = total_energy
    
    # 3. Check eigenvector constraint (verify it's the lowest eigenvalue)
    eigenvector_check = check_eigenvector_constraint(k_opt, eigenvector, J, tolerance)
    constraints['eigenvector'] = eigenvector_check
    
    # 4. Check Fourier space constraint
    fourier_constraint = check_fourier_space_constraint(spins, positions, k_opt, eigenvector, tolerance)
    constraints['fourier_space'] = fourier_constraint
    
    # 5. Check if spins are coplanar (additional check for some systems)
    coplanar_check = check_coplanarity(spins, tolerance)
    constraints['coplanarity'] = coplanar_check
    
    # Overall satisfaction
    all_satisfied = (constraints['unit_length']['satisfied'] and 
                    constraints['eigenvector']['satisfied'] and
                    constraints['fourier_space']['satisfied'])
    
    constraints['all_constraints_satisfied'] = all_satisfied
    
    return constraints

def calculate_classical_energy(spins, NN, NN_bonds, NNN, NNNN, J):
    """Calculate the classical energy of the spin configuration."""
    J1_mats, J2_mat, J3_mat = construct_interaction_matrices(J)
    
    total_energy = 0.0
    N = len(spins)
    
    # Nearest neighbor interactions
    for i in range(N):
        for j in NN[i]:
            if i < j:  # Avoid double counting
                bond_type = NN_bonds[i, j]
                interaction_matrix = J1_mats[bond_type]
                energy = spins[i] @ interaction_matrix @ spins[j]
                total_energy += energy
    
    # Second nearest neighbor interactions
    for i in range(N):
        for j in NNN[i]:
            if i < j:  # Avoid double counting
                energy = spins[i] @ J2_mat @ spins[j]
                total_energy += energy
    
    # Third nearest neighbor interactions
    for i in range(N):
        for j in NNNN[i]:
            if i < j:  # Avoid double counting
                energy = spins[i] @ J3_mat @ spins[j]
                total_energy += energy
    
    return total_energy / N  # Energy per site

def calculate_zeeman_energy(spins, B_field):
    """
    Calculate the Zeeman energy of the spin configuration in an external magnetic field.
    
    Args:
        spins: Spin configuration (N x 3 array)
        B_field: External magnetic field vector (3D array) [Bx, By, Bz]
    
    Returns:
        zeeman_energy: Zeeman energy per site
    """
    if B_field is None or np.linalg.norm(B_field) == 0:
        return 0.0
    
    N = len(spins)
    total_zeeman_energy = 0.0
    
    for i in range(N):
        # Zeeman energy is -μ·B = -S·B (assuming g=2, μ_B=1 units)
        total_zeeman_energy -= np.dot(spins[i], B_field)
    
    return total_zeeman_energy / N  # Energy per site

def calculate_total_energy(spins, NN, NN_bonds, NNN, NNNN, J, B_field=None):
    """
    Calculate the total classical energy including exchange and Zeeman terms.
    
    Args:
        spins: Spin configuration
        NN, NN_bonds, NNN, NNNN: Neighbor lists
        J: Exchange parameters
        B_field: External magnetic field vector (3D array)
    
    Returns:
        total_energy: Total energy per site
        exchange_energy: Exchange energy per site
        zeeman_energy: Zeeman energy per site
    """
    exchange_energy = calculate_classical_energy(spins, NN, NN_bonds, NNN, NNNN, J)
    zeeman_energy = calculate_zeeman_energy(spins, B_field)
    total_energy = exchange_energy + zeeman_energy
    
    return total_energy, exchange_energy, zeeman_energy

def calculate_magnetization(spins):
    """
    Calculate the magnetization vector (average spin) of the configuration.
    
    Args:
        spins: Spin configuration (N x 3 array)
    
    Returns:
        magnetization: Magnetization vector [Mx, My, Mz]
        magnetization_magnitude: Magnitude of magnetization
        sublattice_magnetizations: Dictionary with A and B sublattice magnetizations
    """
    N = len(spins)
    
    # Total magnetization
    magnetization = np.mean(spins, axis=0)
    magnetization_magnitude = np.linalg.norm(magnetization)
    
    # Sublattice magnetizations
    spins_A = spins[::2]   # A sublattice (even indices)
    spins_B = spins[1::2]  # B sublattice (odd indices)
    
    mag_A = np.mean(spins_A, axis=0)
    mag_B = np.mean(spins_B, axis=0)
    
    sublattice_magnetizations = {
        'A': mag_A,
        'B': mag_B,
        'A_magnitude': np.linalg.norm(mag_A),
        'B_magnitude': np.linalg.norm(mag_B),
        'staggered': mag_A - mag_B,
        'staggered_magnitude': np.linalg.norm(mag_A - mag_B)
    }
    
    return magnetization, magnetization_magnitude, sublattice_magnetizations

def check_eigenvector_constraint(k_opt, eigenvector, J, tolerance):
    """Check if the eigenvector corresponds to the lowest eigenvalue of J(k)."""
    # Get lattice vectors (assuming honeycomb)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    # Reconstruct J(k) matrix
    J_k = fourier_transform_interactions(k_opt, a1, a2, J)
    
    # Get all eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(J_k)
    
    # Check if our eigenvector corresponds to the smallest eigenvalue
    min_eigenvalue = eigenvalues[0]
    expected_eigenvector = eigenvectors[:, 0]
    
    # Compare eigenvectors (they can differ by a phase)
    dot_product = np.abs(np.dot(np.conj(eigenvector), expected_eigenvector))
    is_correct_eigenvector = np.abs(dot_product - 1.0) < tolerance
    
    return {
        'satisfied': is_correct_eigenvector,
        'eigenvalue': min_eigenvalue,
        'eigenvector_overlap': dot_product,
        'all_eigenvalues': eigenvalues
    }

def check_fourier_space_constraint(spins, positions, k_opt, eigenvector, tolerance):
    """Check if the spin configuration corresponds to the Fourier space solution."""
    N = len(spins)
    
    # Extract sublattice spin directions from eigenvector
    S_A = eigenvector[:3]
    S_B = eigenvector[3:]
    
    # Normalize
    S_A = S_A / np.linalg.norm(S_A)
    S_B = S_B / np.linalg.norm(S_B)
    
    max_deviation = 0.0
    constraint_satisfied = True
    
    for i in range(N):
        sublattice = i % 2
        if sublattice == 0:  # A sublattice
            phase = np.exp(1j * np.dot(k_opt, positions[i]))
            expected_spin = np.real(phase * S_A)
        else:  # B sublattice
            phase = np.exp(1j * np.dot(k_opt, positions[i]))
            expected_spin = np.real(phase * S_B)
        
        # Normalize expected spin
        if np.linalg.norm(expected_spin) > 1e-10:
            expected_spin = expected_spin / np.linalg.norm(expected_spin)
        
        # Calculate deviation
        deviation = np.linalg.norm(spins[i] - expected_spin)
        max_deviation = max(max_deviation, deviation)
        
        if deviation > tolerance:
            constraint_satisfied = False
    
    return {
        'satisfied': constraint_satisfied,
        'max_deviation': max_deviation
    }

def check_coplanarity(spins, tolerance):
    """Check if all spins lie in the same plane."""
    N = len(spins)
    
    if N < 3:
        return {'satisfied': True, 'reason': 'Too few spins to determine plane'}
    
    # Find two non-parallel spins to define a plane
    reference_spin = spins[0]
    second_spin = None
    
    for i in range(1, N):
        cross_product = np.cross(reference_spin, spins[i])
        if np.linalg.norm(cross_product) > tolerance:
            second_spin = spins[i]
            break
    
    if second_spin is None:
        return {'satisfied': True, 'reason': 'All spins are parallel'}
    
    # Define plane normal
    normal = np.cross(reference_spin, second_spin)
    normal = normal / np.linalg.norm(normal)
    
    # Check if all spins lie in this plane
    max_deviation = 0.0
    for spin in spins:
        deviation = np.abs(np.dot(spin, normal))
        max_deviation = max(max_deviation, deviation)
    
    coplanar = max_deviation < tolerance
    
    return {
        'satisfied': coplanar,
        'max_deviation': max_deviation,
        'plane_normal': normal
    }

def print_constraint_results(constraints):
    """Print a summary of constraint checking results."""
    print("\n" + "="*60)
    print("LUTTINGER-TISZA CONSTRAINT VERIFICATION")
    print("="*60)
    
    # Unit length constraint
    unit_check = constraints['unit_length']
    status = "✓ PASSED" if unit_check['satisfied'] else "✗ FAILED"
    print(f"1. Unit Spin Length Constraint: {status}")
    print(f"   Max deviation from |S| = 1: {unit_check['max_deviation']:.2e}")
    
    # Energy breakdown
    print(f"\n2. Energy Components (per site):")
    print(f"   Exchange energy: {constraints['exchange_energy']:.6f}")
    print(f"   Zeeman energy: {constraints['zeeman_energy']:.6f}")
    print(f"   Total energy: {constraints['total_energy']:.6f}")
    
    # Eigenvector constraint
    eigen_check = constraints['eigenvector']
    status = "✓ PASSED" if eigen_check['satisfied'] else "✗ FAILED"
    print(f"\n3. Eigenvector Constraint: {status}")
    print(f"   Eigenvalue: {eigen_check['eigenvalue']:.6f}")
    print(f"   Eigenvector overlap: {eigen_check['eigenvector_overlap']:.6f}")
    
    # Fourier space constraint
    fourier_check = constraints['fourier_space']
    status = "✓ PASSED" if fourier_check['satisfied'] else "✗ FAILED"
    print(f"\n4. Fourier Space Constraint: {status}")
    print(f"   Max deviation from k-space solution: {fourier_check['max_deviation']:.2e}")
    
    # Coplanarity
    coplanar_check = constraints['coplanarity']
    status = "✓ PASSED" if coplanar_check['satisfied'] else "✗ FAILED"
    print(f"\n5. Coplanarity Check: {status}")
    if 'reason' in coplanar_check:
        print(f"   Reason: {coplanar_check['reason']}")
    else:
        print(f"   Max deviation from plane: {coplanar_check['max_deviation']:.2e}")
    
    # Overall result
    overall = constraints['all_constraints_satisfied']
    status = "✓ ALL CONSTRAINTS SATISFIED" if overall else "✗ SOME CONSTRAINTS FAILED"
    print(f"\n{status}")
    print("="*60)

def print_magnetization_results(magnetization, magnetization_magnitude, sublattice_magnetizations):
    """Print a summary of magnetization results."""
    print("\n" + "="*60)
    print("MAGNETIZATION ANALYSIS")
    print("="*60)
    
    # Total magnetization
    print(f"Total Magnetization:")
    print(f"   Vector: [{magnetization[0]:.6f}, {magnetization[1]:.6f}, {magnetization[2]:.6f}]")
    print(f"   Magnitude: {magnetization_magnitude:.6f}")
    
    # Sublattice magnetizations
    print(f"\nSublattice A Magnetization:")
    mag_A = sublattice_magnetizations['A']
    print(f"   Vector: [{mag_A[0]:.6f}, {mag_A[1]:.6f}, {mag_A[2]:.6f}]")
    print(f"   Magnitude: {sublattice_magnetizations['A_magnitude']:.6f}")
    
    print(f"\nSublattice B Magnetization:")
    mag_B = sublattice_magnetizations['B']
    print(f"   Vector: [{mag_B[0]:.6f}, {mag_B[1]:.6f}, {mag_B[2]:.6f}]")
    print(f"   Magnitude: {sublattice_magnetizations['B_magnitude']:.6f}")
    
    print(f"\nStaggered Magnetization (A - B):")
    staggered = sublattice_magnetizations['staggered']
    print(f"   Vector: [{staggered[0]:.6f}, {staggered[1]:.6f}, {staggered[2]:.6f}]")
    print(f"   Magnitude: {sublattice_magnetizations['staggered_magnitude']:.6f}")
    
    print("="*60)

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
    delta2 = np.array([0, -1]) # Y bond
    delta3 = np.array([1,-1])  # X bond
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
                    bondtype = 1
                elif (delta==delta3).all():
                    bondtype = 0

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
        [0, 0, 0]
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
    delta2 = -a2 + delta3                          # Y bond
    delta1 = a1 - a2 + delta3                      # X bond
    nn_vectors = [delta1, delta2, delta3]
    # print("Nearest neighbor vectors:", nn_vectors)
    
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

def luttinger_tisza_method(L, J=[-6.54, 0.15, -3.76, 0.36, -0.21, 1.70, 0.03], constraints=''):
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
    def objective_function_with_constraints(k_params):
        k = k_params[0] * b1 + k_params[1] * b2
        J_k = fourier_transform_interactions(k, a1, a2, J)
        eigenvalues, eigenvectors = eigh(J_k)
        min_eigenvalue = eigenvalues[0]
        min_eigenvector = eigenvectors[:, 0]
        
        # Check constraints
        spins = reconstruct_spin_configuration(k, min_eigenvector, positions)
        constraints = check_luttinger_tisza_constraints(spins, positions, NN, NN_bonds, NNN, NNNN, 
                                                        k, min_eigenvector, J)
        
        if not constraints['all_constraints_satisfied']:
            return 1e10
        return min_eigenvalue / 2

    def objective_function_Gamma_to_K_path_with_constraints(k_params):
        """
        Objective function for the Gamma to K path with constraints.
        This function returns the minimum eigenvalue for a given k-point.
        """
        k = k_params * b1 + k_params * b2
        J_k = fourier_transform_interactions(k, a1, a2, J)
        eigenvalues, eigenvectors = eigh(J_k)
        min_eigenvalue = eigenvalues[0]
        min_eigenvector = eigenvectors[:, 0]
        
        # Check constraints
        spins = reconstruct_spin_configuration(k, min_eigenvector, positions)
        constraints = check_luttinger_tisza_constraints(spins, positions, NN, NN_bonds, NNN, NNNN, 
                                                        k, min_eigenvector, J)
        
        if not constraints['all_constraints_satisfied']:
            return 1e10
        
        return min_eigenvalue / 2
    
    def objective_function_Gamma_to_M_path_with_constraints(k_params):
        """
        Objective function for the Gamma to M path.
        This function returns the minimum eigenvalue for a given k-point.
        """
        k = k_params * b1
        J_k = fourier_transform_interactions(k, a1, a2, J)
        eigenvalues, eigenvectors = eigh(J_k)
        min_eigenvalue = eigenvalues[0]
        min_eigenvector = eigenvectors[:, 0]

        spins = reconstruct_spin_configuration(k, min_eigenvector, positions)
        constraints = check_luttinger_tisza_constraints(spins, positions, NN, NN_bonds, NNN, NNNN, 
                                                        k, min_eigenvector, J)
        
        if not constraints['all_constraints_satisfied']:
            return 1e10
        
        return min_eigenvalue / 2

    # Initial guess (center of Brillouin zone)
    opt_energy = 1e10
    opt_k_params = None
    min_eigenvector = None
    min_k = None

    for i in range(10):
        initial_guess = np.random.uniform(0, 0.5, size=2)
        
        # Use scipy's minimize function to find the minimum eigenvalue
        if constraints == 'Gamma_to_K':
            objective_function = objective_function_Gamma_to_K_path_with_constraints
            result = minimize(objective_function, np.random.uniform(0, 0.5), 
                        method='Nelder-Mead', bounds=[(0, 0.5)], tol=1e-8)
            opt_k_params_ = result.x * np.ones(2)
        elif constraints == 'Gamma_to_M':
            objective_function = objective_function_Gamma_to_M_path_with_constraints
            result = minimize(objective_function, np.random.uniform(0, 0.5), 
                        method='Nelder-Mead', bounds=[(0, 0.5)], tol=1e-8)
            opt_k_params_ = result.x * np.array([1, 0])
        else:
            objective_function = objective_function_with_constraints
            result = minimize(objective_function, initial_guess, 
                        method='Nelder-Mead', bounds=[(0, 0.5), (0, 0.5)], tol=1e-8)
            opt_k_params_ = result.x

        min_k_ = opt_k_params_[0] * b1 + opt_k_params_[1] * b2
        # Get the corresponding eigenvector
        J_k_ = fourier_transform_interactions(min_k_, a1, a2, J)
        min_eigenvalue_, eigenvectors_ = eigh(J_k_)
        min_eigenvector_ = eigenvectors_[:, 0]
        
        # Ground state energy per site
        ground_state_energy_ = min_eigenvalue_[0] / 2  # Factor of 1/2 from Hamiltonian definition
        if ground_state_energy_ < opt_energy:
            opt_energy = ground_state_energy_
            opt_k_params = opt_k_params_
            min_k = min_k_
            min_eigenvector = min_eigenvector_

    # Optimize again with the best k found
    if opt_k_params is not None:
        result = minimize(objective_function, opt_k_params, 
                method='L-BFGS-B', bounds=[(0, 0.5-1e-9), (0, 0.5)], tol=1e-10)
        opt_k_params = result.x
        min_k = opt_k_params[0] * b1 + opt_k_params[1] * b2
        J_k = fourier_transform_interactions(min_k, a1, a2, J)
        min_eigenvalue, eigenvectors = eigh(J_k)
        min_eigenvector = eigenvectors[:, 0]
        opt_energy = min_eigenvalue[0] / 2

    # Generate the real-space spin configuration
    spins = reconstruct_spin_configuration(min_k, min_eigenvector, positions)
    
    # Check Luttinger-Tisza constraints
    constraints = check_luttinger_tisza_constraints(spins, positions, NN, NN_bonds, NNN, NNNN, 
                                                  min_k, min_eigenvector, J)
    
    return np.array([opt_k_params[0], opt_k_params[1]]), opt_energy, spins, positions, constraints

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
    J = [-6.54, 0.15, -3.76, 0.36, -0.21, 1.7, 0.03]
    
    # Find ground state using Luttinger-Tisza method
    k_opt, energy, spins, positions, constraints = luttinger_tisza_method(L, J, constraints='Gamma_to_K')
    
    # Print results
    print(f"Optimal k-vector: ({k_opt[0]:.4f}, {k_opt[1]:.4f})")
    print(f"Ground state energy per site: {energy:.6f}")
    
    # Print constraint verification results
    print_constraint_results(constraints)
    
    # Calculate and print magnetization results
    magnetization, magnetization_magnitude, sublattice_magnetizations = calculate_magnetization(spins)
    print_magnetization_results(magnetization, magnetization_magnitude, sublattice_magnetizations)
    
    # Visualize the ground state
    visualize_spins(positions, spins, L)