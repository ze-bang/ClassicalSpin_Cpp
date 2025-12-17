import numpy as np
from scipy.optimize import minimize
from luttinger_tisza import (
    create_honeycomb_lattice,
    construct_interaction_matrices,
    get_bond_vectors,
    visualize_spins,
    calculate_energy_from_matrices,
)

# filepath: /home/pc_linux/ClassicalSpin_Cpp/util/single_q.py

import matplotlib.pyplot as plt

class SingleQ:
    """
    Class to perform single-Q ansatz simulation on a BCAO honeycomb lattice
    to determine ground state spin configuration and energy.
    
    Uses the BCAO parameter convention: J = [J1xy, J1z, D, E, F, G, J3xy, J3z]
    matching the C++ implementation in simulated_annealing_BCAO_emily.cpp
    """
    # Define parameter bounds
    eta_small = 10**-9
    min_Q1, max_Q1 = (0, 0.5-eta_small)
    min_Q2, max_Q2 = (0, 0.5)
    min_phi_A, max_phi_A = (0, (2-eta_small)*np.pi)
    min_theta_A, max_theta_A = (0, (2-eta_small)*np.pi)
    min_psi_A, max_psi_A = (0, (2-eta_small)*np.pi)
    min_phi_B, max_phi_B = (0, (2-eta_small)*np.pi)
    min_theta_B, max_theta_B = (0, (2-eta_small)*np.pi)
    min_psi_B, max_psi_B = (0, (2-eta_small)*np.pi)
    min_alpha_A, max_alpha_A = (0, 1.0)
    min_alpha_B, max_alpha_B = (0, 1.0)
    parameter_bounds = [
        (min_Q1, max_Q1), (min_Q2, max_Q2), 
        (min_alpha_A, max_alpha_A), (min_alpha_B, max_alpha_B),
        (min_phi_A, max_phi_A), (min_theta_A, max_theta_A), (min_psi_A, max_psi_A),
        (min_phi_B, max_phi_B), (min_theta_B, max_theta_B), (min_psi_B, max_psi_B),
    ]
    
    def __init__(self, L=4, J=[-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85], B_field=np.array([0, 0, 0])):
        """
        Initialize the single-Q model
        
        Args:
            L: Size of the lattice (L x L unit cells)
            J: BCAO exchange coupling parameters [J1xy, J1z, D, E, F, G, J3xy, J3z]
            B_field: External magnetic field
        """
        self.L = L
        self.J = J
        self.J1xy = J[0]
        self.J1z = J[1]
        self.D = J[2]
        self.E = J[3]
        self.F = J[4]
        self.G = J[5]
        self.J3xy = J[6]
        self.J3z = J[7] 
        self.B_field = B_field
        
        # Create lattice and compute interaction matrices
        self.positions, self.NN, self.NN_bonds, self.NNN, self.NNNN, self.a1, self.a2 = create_honeycomb_lattice(L)
        self.J1, self.J2_mat, self.J3_mat = self.construct_BCAO_interaction_matrices()
        self.nn_vectors, self.nnn_vectors, self.nnnn_vectors = get_bond_vectors(self.a1, self.a2)
        
        # Reciprocal lattice vectors
        self.b1 = 2 * np.pi * np.array([1, -1/np.sqrt(3)])
        self.b2 = 2 * np.pi * np.array([0, 2/np.sqrt(3)])
        
        # Find optimal spin configuration
        self.opt_params, self.opt_energy = self.find_minimum_energy()
        
        # Determine spin configuration type
        self.magnetic_order = self.classify_phase()

    @classmethod
    def RotatedBasis(cls, phi, theta, psi):
        """Define rotated basis with Euler angles (phi, theta, psi) in X-Y-Z convention"""
        RotMat = np.array([
            [np.cos(theta)*np.cos(phi)*np.cos(psi) - np.sin(phi)*np.sin(psi),
             -np.cos(psi)*np.sin(phi) - np.cos(theta)*np.cos(phi)*np.sin(psi),
             np.cos(phi)*np.sin(theta)],
            [np.cos(theta)*np.cos(psi)*np.sin(phi) + np.cos(phi)*np.sin(psi),
             np.cos(phi)*np.cos(psi) - np.cos(theta)*np.sin(phi)*np.sin(psi),
             np.sin(theta)*np.sin(phi)],
            [-np.cos(psi)*np.sin(theta),
             np.sin(theta)*np.sin(psi),
             np.cos(theta)]
        ])
        ex = RotMat[:, 0]
        ey = RotMat[:, 1]
        ez = RotMat[:, 2]
        return ex, ey, ez

    @classmethod
    def uniform_sampling(cls, a, b):
        """Generate a random number between a and b"""
        return (b - a) * np.random.random() + a
    
    def construct_BCAO_interaction_matrices(self):
        """Construct interaction matrices for BCAO using the same convention as the C++ code"""
        # J1z matrix (the base interaction matrix)
        J1z_mat = np.array([
            [self.J1xy + self.D, self.E, self.F],
            [self.E, self.J1xy - self.D, self.G],
            [self.F, self.G, self.J1z]
        ])
        
        # Rotation matrix for 2π/3
        cos_2pi3 = np.cos(2*np.pi/3)
        sin_2pi3 = np.sin(2*np.pi/3)
        U_2pi_3 = np.array([
            [cos_2pi3, sin_2pi3, 0],
            [-sin_2pi3, cos_2pi3, 0],
            [0, 0, 1]
        ])
        
        # Construct J1x and J1y through rotations
        J1x_mat = U_2pi_3 @ J1z_mat @ U_2pi_3.T
        J1y_mat = U_2pi_3.T @ J1z_mat @ U_2pi_3
        
        # List of J1 matrices for the three bond types
        J1 = [J1x_mat, J1y_mat, J1z_mat]
        
        # Second nearest neighbor interactions (set to zero for BCAO)
        J2_mat = np.zeros((3, 3))
        
        # Third nearest neighbor interactions
        J3_mat = np.array([
            [self.J3xy, 0, 0],
            [0, self.J3xy, 0],
            [0, 0, self.J3z]
        ])
        
        return J1, J2_mat, J3_mat
    
    def HAB(self, q):
        """Compute the Fourier transformed interaction matrix between A and B sublattices"""
        JAB = np.zeros((3, 3), dtype=complex)
        
        # Nearest-neighbor interactions
        for i, delta in enumerate(self.nn_vectors):
            phase = np.exp(1j * np.dot(q, delta))
            JAB += self.J1[i] * phase
        
        # Third-neighbor interactions
        for delta in self.nnnn_vectors:
            phase = np.exp(1j * np.dot(q, delta))
            JAB += self.J3_mat * phase
        
        return JAB
    
    def HBA(self, q):
        """Compute the Fourier transformed interaction matrix between B and A sublattices"""
        return self.HAB(q).conj().T
    
    def HAA(self, q):
        """Compute the Fourier transformed interaction matrix for A-A interactions"""
        JAA = np.zeros((3, 3), dtype=complex)
        
        # Second-neighbor interactions
        for delta in self.nnn_vectors:
            phase = np.exp(1j * np.dot(q, delta))
            JAA += self.J2_mat * phase
        
        return JAA
    
    def HBB(self, q):
        """Compute the Fourier transformed interaction matrix for B-B interactions"""
        # Second-neighbor interactions for B sites are the same as for A sites
        return self.HAA(q)
    
    def E_per_UC(self, params):
        """Calculate the energy per unit cell for the single-Q ansatz"""
        Q1, Q2, alpha_A, alpha_B, phi_A, theta_A, psi_A, phi_B, theta_B, psi_B = params
        Qvec = Q1 * self.b1 + Q2 * self.b2
        
        # Get rotated basis vectors
        exA, eyA, ezA = self.RotatedBasis(phi_A, theta_A, psi_A)
        exB, eyB, ezB = self.RotatedBasis(phi_B, theta_B, psi_B)
        
        # Uniform component contribution
        E_q0 = alpha_A * alpha_B * (ezA.dot(self.HAB(np.array([0, 0]))).dot(ezB) + 
                                    ezB.dot(self.HBA(np.array([0, 0]))).dot(ezA))
        E_q0 += alpha_A**2 * ezA.dot(self.HAA(np.array([0, 0]))).dot(ezA)
        E_q0 += alpha_B**2 * ezB.dot(self.HBB(np.array([0, 0]))).dot(ezB)
        
        # Modulated component contribution
        E_q = np.sqrt(1-alpha_A**2) * np.sqrt(1-alpha_B**2) / 4 * (
            (exA - 1.0j * eyA).dot(self.HAB(Qvec)).dot(exB + 1.0j * eyB) +
            (exB - 1.0j * eyB).dot(self.HBA(Qvec)).dot(exA + 1.0j * eyA) +
            (exA + 1.0j * eyA).dot(self.HAB(-Qvec)).dot(exB - 1.0j * eyB) +
            (exB + 1.0j * eyB).dot(self.HBA(-Qvec)).dot(exA - 1.0j * eyA)
        )
        
        E_q += np.sqrt(1-alpha_A**2)**2 / 4 * (
            (exA - 1.0j * eyA).dot(self.HAA(Qvec)).dot(exA + 1.0j * eyA) +
            (exA + 1.0j * eyA).dot(self.HAA(-Qvec)).dot(exA - 1.0j * eyA)
        )
        
        E_q += np.sqrt(1-alpha_B**2)**2 / 4 * (
            (exB - 1.0j * eyB).dot(self.HBB(Qvec)).dot(exB + 1.0j * eyB) +
            (exB + 1.0j * eyB).dot(self.HBB(-Qvec)).dot(exB - 1.0j * eyB)
        )
        
        # Zeeman energy contribution
        E_zeeman = self.B_field.dot(ezA * alpha_A + ezB * alpha_B)
        
        return np.real(E_q0 / 4 + E_q / 4 + E_zeeman / 2)
    
    def find_minimum_energy(self, N_ITERATIONS=20, tol_first_opt=10**-8, tol_second_opt=10**-10):
        """Find the optimal parameters that minimize the energy"""
        opt_params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        opt_energy = 10**10
        
        for i in range(N_ITERATIONS):
            # Random initial guess
            Q1_guess = self.uniform_sampling(self.min_Q1, self.max_Q1)
            Q2_guess = self.uniform_sampling(self.min_Q2, self.max_Q2)
            alpha_A_guess = self.uniform_sampling(self.min_alpha_A, self.max_alpha_A)
            alpha_B_guess = self.uniform_sampling(self.min_alpha_B, self.max_alpha_B)
            phi_A_guess = self.uniform_sampling(self.min_phi_A, self.max_phi_A)
            theta_A_guess = self.uniform_sampling(self.min_theta_A, self.max_theta_A)
            psi_A_guess = self.uniform_sampling(self.min_psi_A, self.max_psi_A)
            phi_B_guess = self.uniform_sampling(self.min_phi_B, self.max_phi_B)
            theta_B_guess = self.uniform_sampling(self.min_theta_B, self.max_theta_B)
            psi_B_guess = self.uniform_sampling(self.min_psi_B, self.max_psi_B)
            
            initial_guess = [Q1_guess, Q2_guess, alpha_A_guess, alpha_B_guess, 
                            phi_A_guess, theta_A_guess, psi_A_guess, 
                            phi_B_guess, theta_B_guess, psi_B_guess]
            
            # Minimize at that point
            res = minimize(self.E_per_UC, x0=initial_guess, bounds=self.parameter_bounds, 
                          method='Nelder-Mead', tol=tol_first_opt)

            if res.fun < opt_energy:
                opt_params = res.x
                opt_energy = res.fun
        
        # Final optimization run on best parameters
        res = minimize(self.E_per_UC, x0=opt_params, bounds=self.parameter_bounds, 
                      method='L-BFGS-B', tol=tol_second_opt)
        opt_params = res.x
        opt_energy = res.fun
        
        # If we have uniform magnetization, set Q to zero
        if np.abs(opt_params[2]-1) < 10**-10 and np.abs(opt_params[3]-1) < 10**-10:
            opt_params[0] = 0
            opt_params[1] = 0
        
        return opt_params, opt_energy
    
    def classify_phase(self, tol=10**-6):
        """Classify the magnetic ordering based on the optimal parameters"""
        Q1, Q2, alphaA, alphaB, phiA, thetaA, psiA, phiB, thetaB, psiB = self.opt_params
        Q_vec = Q1 * self.b1 + Q2 * self.b2
        
        # Gamma point order
        if np.abs(Q1) < tol and np.abs(Q2) < tol:
            exA, eyA, ezA = self.RotatedBasis(phiA, thetaA, psiA)
            exB, eyB, ezB = self.RotatedBasis(phiB, thetaB, psiB)
            spin_a_0 = ezA * alphaA + np.sqrt(1 - alphaA**2) * exA
            spin_b_0 = ezB * alphaB + np.sqrt(1 - alphaB**2) * exB
            dot_product = spin_a_0.dot(spin_b_0)
            if dot_product > 0:
                return "FM"
            else:
                return "AFM"
        
        # M point order
        elif (np.abs(Q1-0.5) < tol and np.abs(Q2) < tol) or (np.abs(Q2-0.5) < tol and np.abs(Q1) < tol):
            return "Zigzag/Stripy"
        
        # K point order
        elif np.abs(Q1-1/3) < tol and np.abs(Q2-1/3) < tol:
            return "120° order"
        
        # Incommensurate order
        else:
            return "Incommensurate order"
    
    def generate_spin_configuration(self):
        """Generate the spin configuration from the optimal parameters"""
        Q1, Q2, alphaA, alphaB, phiA, thetaA, psiA, phiB, thetaB, psiB = self.opt_params
        Q_vec = Q1 * self.b1 + Q2 * self.b2
        
        N = len(self.positions)
        spins = np.zeros((N, 3))
        
        exA, eyA, ezA = self.RotatedBasis(phiA, thetaA, psiA)
        exB, eyB, ezB = self.RotatedBasis(phiB, thetaB, psiB)
        
        for i in range(N):
            pos = self.positions[i]
            if i % 2 == 0:  # A sublattice
                spin = ezA * alphaA + np.sqrt(1 - alphaA**2) * (exA * np.cos(Q_vec.dot(pos)) + eyA * np.sin(Q_vec.dot(pos)))
            else:  # B sublattice
                spin = ezB * alphaB + np.sqrt(1 - alphaB**2) * (exB * np.cos(Q_vec.dot(pos)) + eyB * np.sin(Q_vec.dot(pos)))
            
            # Normalize the spin
            spins[i] = spin / np.linalg.norm(spin)
        
        return spins

    def calculate_magnetization(self, spins):
        """
        Calculate the magnetization components of the spin configuration.
        
        Args:
            spins: Spin configuration (N x 3 array)
        
        Returns:
            magnetization: Dictionary with magnetization components
        """
        N = len(spins)
        
        # Total magnetization
        total_magnetization = np.mean(spins, axis=0)
        total_magnitude = np.linalg.norm(total_magnetization)
        
        # Sublattice magnetizations
        spins_A = spins[::2]   # A sublattice (even indices)
        spins_B = spins[1::2]  # B sublattice (odd indices)
        
        mag_A = np.mean(spins_A, axis=0)
        mag_B = np.mean(spins_B, axis=0)
        
        mag_A_magnitude = np.linalg.norm(mag_A)
        mag_B_magnitude = np.linalg.norm(mag_B)
        
        # Staggered magnetization
        staggered = mag_A - mag_B
        staggered_magnitude = np.linalg.norm(staggered)
        
        return {
            'total': total_magnetization,
            'total_magnitude': total_magnitude,
            'A': mag_A,
            'B': mag_B,
            'A_magnitude': mag_A_magnitude,
            'B_magnitude': mag_B_magnitude,
            'staggered': staggered,
            'staggered_magnitude': staggered_magnitude
        }

    def real_space_energy(self, spins=None):
        """Energy per site computed directly from the BCAO Hamiltonian."""
        if spins is None:
            spins = self.generate_spin_configuration()
        return calculate_energy_from_matrices(
            spins, self.NN, self.NN_bonds, self.NNN, self.NNNN, self.J1, self.J2_mat, self.J3_mat
        )


if __name__ == "__main__":
    # Size of lattice (L x L unit cells)
    L = 12
    
    # BCAO parameters matching the C++ convention: [J1xy, J1z, D, E, F, G, J3xy, J3z]
    J = [-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85]
    
    # Create single-Q model
    model = SingleQ(L, J)
    
    # Print results
    Q1, Q2 = model.opt_params[0], model.opt_params[1]
    print(f"Optimal Q-vector: ({Q1:.4f}, {Q2:.4f})")
    print(f"Ground state energy: {model.opt_energy:.6f}")
    print(f"Magnetic order: {model.magnetic_order}")
    print(f"BCAO Parameters: J1xy={J[0]}, J1z={J[1]}, D={J[2]}, E={J[3]}, F={J[4]}, G={J[5]}, J3xy={J[6]}, J3z={J[7]}")
    
    # Generate and analyze the spin configuration
    spins = model.generate_spin_configuration()
    
    # Calculate magnetization
    magnetization = model.calculate_magnetization(spins)
    
    # Print magnetization results
    print("\nMagnetization Analysis:")
    print("="*50)
    print(f"Total magnetization: [{magnetization['total'][0]:.6f}, {magnetization['total'][1]:.6f}, {magnetization['total'][2]:.6f}]")
    print(f"Total magnetization magnitude: {magnetization['total_magnitude']:.6f}")
    print(f"Sublattice A magnetization: [{magnetization['A'][0]:.6f}, {magnetization['A'][1]:.6f}, {magnetization['A'][2]:.6f}]")
    print(f"Sublattice A magnitude: {magnetization['A_magnitude']:.6f}")
    print(f"Sublattice B magnetization: [{magnetization['B'][0]:.6f}, {magnetization['B'][1]:.6f}, {magnetization['B'][2]:.6f}]")
    print(f"Sublattice B magnitude: {magnetization['B_magnitude']:.6f}")
    print(f"Staggered magnetization: [{magnetization['staggered'][0]:.6f}, {magnetization['staggered'][1]:.6f}, {magnetization['staggered'][2]:.6f}]")
    print(f"Staggered magnetization magnitude: {magnetization['staggered_magnitude']:.6f}")
    
    # Visualize the spin configuration
    visualize_spins(model.positions, spins, L)