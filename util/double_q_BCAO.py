import numpy as np
from scipy.optimize import minimize
from luttinger_tisza import create_honeycomb_lattice, construct_interaction_matrices, get_bond_vectors, visualize_spins

# filepath: /home/pc_linux/ClassicalSpin_Cpp/util/double_q_BCAO.py

import matplotlib.pyplot as plt

class DoubleQ:
    """
    Class to perform double-Q ansatz simulation on a BCAO honeycomb lattice
    to determine ground state spin configuration and energy.
    
    The double-Q ansatz allows for more complex magnetic orderings by superposing
    two different Q-vectors. The spin configuration is:
    
    S_A(r) = α_A * e_z^A + Σ_{Q∈{Q1,Q2}} √(β_Q^A/2) * [e_x^{A,Q} * cos(Q·r + φ_Q^A) + e_y^{A,Q} * sin(Q·r + φ_Q^A)]
    S_B(r) = α_B * e_z^B + Σ_{Q∈{Q1,Q2}} √(β_Q^B/2) * [e_x^{B,Q} * cos(Q·r + φ_Q^B) + e_y^{B,Q} * sin(Q·r + φ_Q^B)]
    
    where:
    - Q1, Q2 are two independent ordering wavevectors
    - α_A, α_B control the uniform (Q=0) component
    - β_Q^A, β_Q^B control the amplitude of each Q component
    - Constraint: α_A² + Σ_Q β_Q^A = 1 (and similarly for B)
    - e_z, e_x, e_y are orthonormal basis vectors defined by Euler angles
    - φ_Q are phase factors for each Q component
    
    Uses the BCAO parameter convention: J = [J1xy, J1z, D, E, F, G, J3xy, J3z]
    """
    
    # Define parameter bounds
    eta_small = 10**-9
    min_Q1, max_Q1 = (0, 0.5-eta_small)
    min_Q2, max_Q2 = (0, 0.5)
    min_Q3, max_Q3 = (0, 0.5-eta_small)
    min_Q4, max_Q4 = (0, 0.5)
    min_phi_A, max_phi_A = (0, (2-eta_small)*np.pi)
    min_theta_A, max_theta_A = (0, (2-eta_small)*np.pi)
    min_psi_A, max_psi_A = (0, (2-eta_small)*np.pi)
    min_phi_B, max_phi_B = (0, (2-eta_small)*np.pi)
    min_theta_B, max_theta_B = (0, (2-eta_small)*np.pi)
    min_psi_B, max_psi_B = (0, (2-eta_small)*np.pi)
    min_phi_A_Q1, max_phi_A_Q1 = (0, (2-eta_small)*np.pi)
    min_theta_A_Q1, max_theta_A_Q1 = (0, (2-eta_small)*np.pi)
    min_psi_A_Q1, max_psi_A_Q1 = (0, (2-eta_small)*np.pi)
    min_phi_B_Q1, max_phi_B_Q1 = (0, (2-eta_small)*np.pi)
    min_theta_B_Q1, max_theta_B_Q1 = (0, (2-eta_small)*np.pi)
    min_psi_B_Q1, max_psi_B_Q1 = (0, (2-eta_small)*np.pi)
    min_phi_A_Q2, max_phi_A_Q2 = (0, (2-eta_small)*np.pi)
    min_theta_A_Q2, max_theta_A_Q2 = (0, (2-eta_small)*np.pi)
    min_psi_A_Q2, max_psi_A_Q2 = (0, (2-eta_small)*np.pi)
    min_phi_B_Q2, max_phi_B_Q2 = (0, (2-eta_small)*np.pi)
    min_theta_B_Q2, max_theta_B_Q2 = (0, (2-eta_small)*np.pi)
    min_psi_B_Q2, max_psi_B_Q2 = (0, (2-eta_small)*np.pi)
    min_alpha_A, max_alpha_A = (0, 1.0)
    min_alpha_B, max_alpha_B = (0, 1.0)
    min_beta_Q1_A, max_beta_Q1_A = (0, 1.0)
    min_beta_Q1_B, max_beta_Q1_B = (0, 1.0)
    min_beta_Q2_A, max_beta_Q2_A = (0, 1.0)
    min_beta_Q2_B, max_beta_Q2_B = (0, 1.0)
    min_phase_Q1_A, max_phase_Q1_A = (0, (2-eta_small)*np.pi)
    min_phase_Q1_B, max_phase_Q1_B = (0, (2-eta_small)*np.pi)
    min_phase_Q2_A, max_phase_Q2_A = (0, (2-eta_small)*np.pi)
    min_phase_Q2_B, max_phase_Q2_B = (0, (2-eta_small)*np.pi)
    
    parameter_bounds = [
        # Q-vectors
        (min_Q1, max_Q1), (min_Q2, max_Q2), 
        (min_Q3, max_Q3), (min_Q4, max_Q4),
        # Amplitudes
        (min_alpha_A, max_alpha_A), (min_alpha_B, max_alpha_B),
        (min_beta_Q1_A, max_beta_Q1_A), (min_beta_Q1_B, max_beta_Q1_B),
        (min_beta_Q2_A, max_beta_Q2_A), (min_beta_Q2_B, max_beta_Q2_B),
        # Uniform component Euler angles
        (min_phi_A, max_phi_A), (min_theta_A, max_theta_A), (min_psi_A, max_psi_A),
        (min_phi_B, max_phi_B), (min_theta_B, max_theta_B), (min_psi_B, max_psi_B),
        # Q1 component Euler angles
        (min_phi_A_Q1, max_phi_A_Q1), (min_theta_A_Q1, max_theta_A_Q1), (min_psi_A_Q1, max_psi_A_Q1),
        (min_phi_B_Q1, max_phi_B_Q1), (min_theta_B_Q1, max_theta_B_Q1), (min_psi_B_Q1, max_psi_B_Q1),
        # Q2 component Euler angles
        (min_phi_A_Q2, max_phi_A_Q2), (min_theta_A_Q2, max_theta_A_Q2), (min_psi_A_Q2, max_psi_A_Q2),
        (min_phi_B_Q2, max_phi_B_Q2), (min_theta_B_Q2, max_theta_B_Q2), (min_psi_B_Q2, max_psi_B_Q2),
        # Phase factors
        (min_phase_Q1_A, max_phase_Q1_A), (min_phase_Q1_B, max_phase_Q1_B),
        (min_phase_Q2_A, max_phase_Q2_A), (min_phase_Q2_B, max_phase_Q2_B),
    ]
    
    def __init__(self, L=4, J=[-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85], B_field=np.array([0, 0, 0])):
        """
        Initialize the double-Q model
        
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
        """
        Calculate the energy per unit cell for the double-Q ansatz
        
        Parameters are organized as:
        [Q1_1, Q1_2, Q2_1, Q2_2,
         alpha_A, alpha_B, beta_Q1_A, beta_Q1_B, beta_Q2_A, beta_Q2_B,
         phi_A, theta_A, psi_A, phi_B, theta_B, psi_B,
         phi_A_Q1, theta_A_Q1, psi_A_Q1, phi_B_Q1, theta_B_Q1, psi_B_Q1,
         phi_A_Q2, theta_A_Q2, psi_A_Q2, phi_B_Q2, theta_B_Q2, psi_B_Q2,
         phase_Q1_A, phase_Q1_B, phase_Q2_A, phase_Q2_B]
        """
        (Q1_1, Q1_2, Q2_1, Q2_2,
         alpha_A, alpha_B, beta_Q1_A, beta_Q1_B, beta_Q2_A, beta_Q2_B,
         phi_A, theta_A, psi_A, phi_B, theta_B, psi_B,
         phi_A_Q1, theta_A_Q1, psi_A_Q1, phi_B_Q1, theta_B_Q1, psi_B_Q1,
         phi_A_Q2, theta_A_Q2, psi_A_Q2, phi_B_Q2, theta_B_Q2, psi_B_Q2,
         phase_Q1_A, phase_Q1_B, phase_Q2_A, phase_Q2_B) = params
        
        # Check normalization constraint with soft penalty
        norm_A = alpha_A**2 + beta_Q1_A + beta_Q2_A
        norm_B = alpha_B**2 + beta_Q1_B + beta_Q2_B
        
        # Soft penalty for constraint violations
        penalty = 0.0
        if norm_A > 1.0:
            penalty += 1e6 * (norm_A - 1.0)**2
        if norm_B > 1.0:
            penalty += 1e6 * (norm_B - 1.0)**2
        if norm_A < 0.0 or norm_B < 0.0:
            return 1e10
        
        # Construct Q-vectors
        Q1 = Q1_1 * self.b1 + Q1_2 * self.b2
        Q2 = Q2_1 * self.b1 + Q2_2 * self.b2
        
        # Get rotated basis vectors for uniform component
        exA, eyA, ezA = self.RotatedBasis(phi_A, theta_A, psi_A)
        exB, eyB, ezB = self.RotatedBasis(phi_B, theta_B, psi_B)
        
        # Get rotated basis vectors for Q1 component
        exA_Q1, eyA_Q1, ezA_Q1 = self.RotatedBasis(phi_A_Q1, theta_A_Q1, psi_A_Q1)
        exB_Q1, eyB_Q1, ezB_Q1 = self.RotatedBasis(phi_B_Q1, theta_B_Q1, psi_B_Q1)
        
        # Get rotated basis vectors for Q2 component
        exA_Q2, eyA_Q2, ezA_Q2 = self.RotatedBasis(phi_A_Q2, theta_A_Q2, psi_A_Q2)
        exB_Q2, eyB_Q2, ezB_Q2 = self.RotatedBasis(phi_B_Q2, theta_B_Q2, psi_B_Q2)
        
        # Energy calculation
        E = 0.0
        
        # === Q=0 (uniform) component contributions ===
        E_q0_AB = alpha_A * alpha_B * (ezA.dot(self.HAB(np.array([0, 0]))).dot(ezB))
        E_q0_BA = alpha_A * alpha_B * (ezB.dot(self.HBA(np.array([0, 0]))).dot(ezA))
        E_q0_AA = alpha_A**2 * ezA.dot(self.HAA(np.array([0, 0]))).dot(ezA)
        E_q0_BB = alpha_B**2 * ezB.dot(self.HBB(np.array([0, 0]))).dot(ezB)
        
        E += (E_q0_AB + E_q0_BA + E_q0_AA + E_q0_BB) / 4
        
        # === Q1 component contributions ===
        # Define complex spin vectors for Q1
        SA_Q1_plus = exA_Q1 * np.cos(phase_Q1_A) + eyA_Q1 * np.sin(phase_Q1_A)
        SA_Q1_minus = exA_Q1 * np.cos(-phase_Q1_A) + eyA_Q1 * np.sin(-phase_Q1_A)
        SB_Q1_plus = exB_Q1 * np.cos(phase_Q1_B) + eyB_Q1 * np.sin(phase_Q1_B)
        SB_Q1_minus = exB_Q1 * np.cos(-phase_Q1_B) + eyB_Q1 * np.sin(-phase_Q1_B)
        
        # Q1 inter-sublattice
        E_Q1_AB = np.sqrt(beta_Q1_A * beta_Q1_B) / 4 * (
            SA_Q1_plus.dot(self.HAB(Q1)).dot(SB_Q1_minus) +
            SA_Q1_minus.dot(self.HAB(-Q1)).dot(SB_Q1_plus)
        )
        E_Q1_BA = np.sqrt(beta_Q1_A * beta_Q1_B) / 4 * (
            SB_Q1_plus.dot(self.HBA(Q1)).dot(SA_Q1_minus) +
            SB_Q1_minus.dot(self.HBA(-Q1)).dot(SA_Q1_plus)
        )
        
        # Q1 intra-sublattice
        E_Q1_AA = beta_Q1_A / 4 * (
            SA_Q1_plus.dot(self.HAA(Q1)).dot(SA_Q1_minus) +
            SA_Q1_minus.dot(self.HAA(-Q1)).dot(SA_Q1_plus)
        )
        E_Q1_BB = beta_Q1_B / 4 * (
            SB_Q1_plus.dot(self.HBB(Q1)).dot(SB_Q1_minus) +
            SB_Q1_minus.dot(self.HBB(-Q1)).dot(SB_Q1_plus)
        )
        
        E += (E_Q1_AB + E_Q1_BA + E_Q1_AA + E_Q1_BB) / 4
        
        # === Q2 component contributions ===
        SA_Q2_plus = exA_Q2 * np.cos(phase_Q2_A) + eyA_Q2 * np.sin(phase_Q2_A)
        SA_Q2_minus = exA_Q2 * np.cos(-phase_Q2_A) + eyA_Q2 * np.sin(-phase_Q2_A)
        SB_Q2_plus = exB_Q2 * np.cos(phase_Q2_B) + eyB_Q2 * np.sin(phase_Q2_B)
        SB_Q2_minus = exB_Q2 * np.cos(-phase_Q2_B) + eyB_Q2 * np.sin(-phase_Q2_B)
        
        # Q2 inter-sublattice
        E_Q2_AB = np.sqrt(beta_Q2_A * beta_Q2_B) / 4 * (
            SA_Q2_plus.dot(self.HAB(Q2)).dot(SB_Q2_minus) +
            SA_Q2_minus.dot(self.HAB(-Q2)).dot(SB_Q2_plus)
        )
        E_Q2_BA = np.sqrt(beta_Q2_A * beta_Q2_B) / 4 * (
            SB_Q2_plus.dot(self.HBA(Q2)).dot(SA_Q2_minus) +
            SB_Q2_minus.dot(self.HBA(-Q2)).dot(SA_Q2_plus)
        )
        
        # Q2 intra-sublattice
        E_Q2_AA = beta_Q2_A / 4 * (
            SA_Q2_plus.dot(self.HAA(Q2)).dot(SA_Q2_minus) +
            SA_Q2_minus.dot(self.HAA(-Q2)).dot(SA_Q2_plus)
        )
        E_Q2_BB = beta_Q2_B / 4 * (
            SB_Q2_plus.dot(self.HBB(Q2)).dot(SB_Q2_minus) +
            SB_Q2_minus.dot(self.HBB(-Q2)).dot(SB_Q2_plus)
        )
        
        E += (E_Q2_AB + E_Q2_BA + E_Q2_AA + E_Q2_BB) / 4
        
        # === Cross terms between Q=0 and Q1 ===
        E_cross_0_Q1_AB = alpha_A * np.sqrt(beta_Q1_B) / 2 * (
            ezA.dot(self.HAB(Q1)).dot(SB_Q1_minus) +
            ezA.dot(self.HAB(-Q1)).dot(SB_Q1_plus)
        )
        E_cross_0_Q1_BA = alpha_B * np.sqrt(beta_Q1_A) / 2 * (
            ezB.dot(self.HBA(Q1)).dot(SA_Q1_minus) +
            ezB.dot(self.HBA(-Q1)).dot(SA_Q1_plus)
        )
        E_cross_0_Q1_AA = alpha_A * np.sqrt(beta_Q1_A) / 2 * (
            ezA.dot(self.HAA(Q1)).dot(SA_Q1_minus) +
            ezA.dot(self.HAA(-Q1)).dot(SA_Q1_plus)
        )
        E_cross_0_Q1_BB = alpha_B * np.sqrt(beta_Q1_B) / 2 * (
            ezB.dot(self.HBB(Q1)).dot(SB_Q1_minus) +
            ezB.dot(self.HBB(-Q1)).dot(SB_Q1_plus)
        )
        
        E += (E_cross_0_Q1_AB + E_cross_0_Q1_BA + E_cross_0_Q1_AA + E_cross_0_Q1_BB) / 4
        
        # === Cross terms between Q=0 and Q2 ===
        E_cross_0_Q2_AB = alpha_A * np.sqrt(beta_Q2_B) / 2 * (
            ezA.dot(self.HAB(Q2)).dot(SB_Q2_minus) +
            ezA.dot(self.HAB(-Q2)).dot(SB_Q2_plus)
        )
        E_cross_0_Q2_BA = alpha_B * np.sqrt(beta_Q2_A) / 2 * (
            ezB.dot(self.HBA(Q2)).dot(SA_Q2_minus) +
            ezB.dot(self.HBA(-Q2)).dot(SA_Q2_plus)
        )
        E_cross_0_Q2_AA = alpha_A * np.sqrt(beta_Q2_A) / 2 * (
            ezA.dot(self.HAA(Q2)).dot(SA_Q2_minus) +
            ezA.dot(self.HAA(-Q2)).dot(SA_Q2_plus)
        )
        E_cross_0_Q2_BB = alpha_B * np.sqrt(beta_Q2_B) / 2 * (
            ezB.dot(self.HBB(Q2)).dot(SB_Q2_minus) +
            ezB.dot(self.HBB(-Q2)).dot(SB_Q2_plus)
        )
        
        E += (E_cross_0_Q2_AB + E_cross_0_Q2_BA + E_cross_0_Q2_AA + E_cross_0_Q2_BB) / 4
        
        # === Cross terms between Q1 and Q2 ===
        Q_sum = Q1 + Q2
        Q_diff = Q1 - Q2
        
        # Q1 + Q2 contributions
        E_cross_Q1_Q2_AB = np.sqrt(beta_Q1_A * beta_Q2_B) / 4 * (
            SA_Q1_plus.dot(self.HAB(Q_sum)).dot(SB_Q2_minus) +
            SA_Q1_minus.dot(self.HAB(-Q_sum)).dot(SB_Q2_plus)
        )
        E_cross_Q1_Q2_BA = np.sqrt(beta_Q2_A * beta_Q1_B) / 4 * (
            SB_Q1_plus.dot(self.HBA(Q_sum)).dot(SA_Q2_minus) +
            SB_Q1_minus.dot(self.HBA(-Q_sum)).dot(SA_Q2_plus)
        )
        E_cross_Q1_Q2_AA = np.sqrt(beta_Q1_A * beta_Q2_A) / 4 * (
            SA_Q1_plus.dot(self.HAA(Q_sum)).dot(SA_Q2_minus) +
            SA_Q1_minus.dot(self.HAA(-Q_sum)).dot(SA_Q2_plus)
        )
        E_cross_Q1_Q2_BB = np.sqrt(beta_Q1_B * beta_Q2_B) / 4 * (
            SB_Q1_plus.dot(self.HBB(Q_sum)).dot(SB_Q2_minus) +
            SB_Q1_minus.dot(self.HBB(-Q_sum)).dot(SB_Q2_plus)
        )
        
        E += (E_cross_Q1_Q2_AB + E_cross_Q1_Q2_BA + E_cross_Q1_Q2_AA + E_cross_Q1_Q2_BB) / 4
        
        # Q1 - Q2 contributions
        E_cross_Q1_Q2_diff_AB = np.sqrt(beta_Q1_A * beta_Q2_B) / 4 * (
            SA_Q1_plus.dot(self.HAB(Q_diff)).dot(SB_Q2_plus) +
            SA_Q1_minus.dot(self.HAB(-Q_diff)).dot(SB_Q2_minus)
        )
        E_cross_Q1_Q2_diff_BA = np.sqrt(beta_Q2_A * beta_Q1_B) / 4 * (
            SB_Q2_plus.dot(self.HBA(Q_diff)).dot(SA_Q1_plus) +
            SB_Q2_minus.dot(self.HBA(-Q_diff)).dot(SA_Q1_minus)
        )
        E_cross_Q1_Q2_diff_AA = np.sqrt(beta_Q1_A * beta_Q2_A) / 4 * (
            SA_Q1_plus.dot(self.HAA(Q_diff)).dot(SA_Q2_plus) +
            SA_Q1_minus.dot(self.HAA(-Q_diff)).dot(SA_Q2_minus)
        )
        E_cross_Q1_Q2_diff_BB = np.sqrt(beta_Q1_B * beta_Q2_B) / 4 * (
            SB_Q1_plus.dot(self.HBB(Q_diff)).dot(SB_Q2_plus) +
            SB_Q1_minus.dot(self.HBB(-Q_diff)).dot(SB_Q2_minus)
        )
        
        E += (E_cross_Q1_Q2_diff_AB + E_cross_Q1_Q2_diff_BA + E_cross_Q1_Q2_diff_AA + E_cross_Q1_Q2_diff_BB) / 4
        
        # === Zeeman energy contribution ===
        # Average spin components
        avg_spin_A = alpha_A * ezA
        avg_spin_B = alpha_B * ezB
        E_zeeman = self.B_field.dot(avg_spin_A + avg_spin_B) / 2
        
        E += E_zeeman
        
        return np.real(E) + penalty
    
    def find_minimum_energy(self, N_ITERATIONS=20, tol_first_opt=10**-5, tol_second_opt=10**-7, 
                           start_from_single_q=True):
        """
        Find the optimal parameters that minimize the energy
        
        Args:
            N_ITERATIONS: Number of random initializations
            tol_first_opt: Tolerance for first optimization phase
            tol_second_opt: Tolerance for final optimization phase
            start_from_single_q: Whether to seed some initial guesses from single-Q solution
        """
        opt_params = None
        opt_energy = 10**10
        
        print(f"Starting optimization with {N_ITERATIONS} random initial guesses...")
        
        # If requested, try starting from single-Q solution
        if start_from_single_q:
            try:
                from single_q_BCAO import SingleQ
                single_q = SingleQ(L=self.L, J=self.J, B_field=self.B_field)
                
                # Use single-Q solution as one of the initial guesses
                Q1_sq, Q2_sq = single_q.opt_params[0], single_q.opt_params[1]
                alpha_A_sq, alpha_B_sq = single_q.opt_params[2], single_q.opt_params[3]
                phi_A_sq, theta_A_sq, psi_A_sq = single_q.opt_params[4], single_q.opt_params[5], single_q.opt_params[6]
                phi_B_sq, theta_B_sq, psi_B_sq = single_q.opt_params[7], single_q.opt_params[8], single_q.opt_params[9]
                
                # Create initial guess from single-Q (set Q2 components to zero)
                initial_guess_sq = [
                    Q1_sq, Q2_sq, 0.0, 0.0,  # Q1 from single-Q, Q2 = 0
                    alpha_A_sq, alpha_B_sq,   # Amplitudes from single-Q
                    1 - alpha_A_sq**2, 1 - alpha_B_sq**2, 0.0, 0.0,  # All weight on Q1
                    phi_A_sq, theta_A_sq, psi_A_sq, phi_B_sq, theta_B_sq, psi_B_sq,  # From single-Q
                    self.uniform_sampling(0, 2*np.pi), self.uniform_sampling(0, 2*np.pi), self.uniform_sampling(0, 2*np.pi),
                    self.uniform_sampling(0, 2*np.pi), self.uniform_sampling(0, 2*np.pi), self.uniform_sampling(0, 2*np.pi),
                    self.uniform_sampling(0, 2*np.pi), self.uniform_sampling(0, 2*np.pi), self.uniform_sampling(0, 2*np.pi),
                    self.uniform_sampling(0, 2*np.pi), self.uniform_sampling(0, 2*np.pi), self.uniform_sampling(0, 2*np.pi),
                    0.0, 0.0, 0.0, 0.0  # Phases
                ]
                
                res = minimize(self.E_per_UC, x0=initial_guess_sq, bounds=self.parameter_bounds,
                             method='Nelder-Mead', options={'maxiter': 5000}, tol=tol_first_opt)
                
                if res.fun < opt_energy:
                    opt_params = res.x
                    opt_energy = res.fun
                    print(f"  Single-Q seeded guess: energy = {res.fun:.8f}")
            except Exception as e:
                print(f"  Warning: Could not seed from single-Q: {str(e)}")
        
        for i in range(N_ITERATIONS):
            if (i+1) % 5 == 0:
                print(f"  Iteration {i+1}/{N_ITERATIONS}, current best energy: {opt_energy:.8f}")
            
            # Random initial guess
            Q1_1_guess = self.uniform_sampling(self.min_Q1, self.max_Q1)
            Q1_2_guess = self.uniform_sampling(self.min_Q2, self.max_Q2)
            Q2_1_guess = self.uniform_sampling(self.min_Q3, self.max_Q3)
            Q2_2_guess = self.uniform_sampling(self.min_Q4, self.max_Q4)
            
            alpha_A_guess = self.uniform_sampling(self.min_alpha_A, self.max_alpha_A)
            alpha_B_guess = self.uniform_sampling(self.min_alpha_B, self.max_alpha_B)
            
            # Ensure normalization constraint: α² + β_Q1 + β_Q2 = 1
            remaining_A = max(0, 1 - alpha_A_guess**2)
            remaining_B = max(0, 1 - alpha_B_guess**2)
            
            if remaining_A > 0:
                beta_Q1_A_guess = self.uniform_sampling(0, remaining_A)
                beta_Q2_A_guess = remaining_A - beta_Q1_A_guess
            else:
                beta_Q1_A_guess = 0
                beta_Q2_A_guess = 0
            
            if remaining_B > 0:
                beta_Q1_B_guess = self.uniform_sampling(0, remaining_B)
                beta_Q2_B_guess = remaining_B - beta_Q1_B_guess
            else:
                beta_Q1_B_guess = 0
                beta_Q2_B_guess = 0
            
            phi_A_guess = self.uniform_sampling(self.min_phi_A, self.max_phi_A)
            theta_A_guess = self.uniform_sampling(self.min_theta_A, self.max_theta_A)
            psi_A_guess = self.uniform_sampling(self.min_psi_A, self.max_psi_A)
            phi_B_guess = self.uniform_sampling(self.min_phi_B, self.max_phi_B)
            theta_B_guess = self.uniform_sampling(self.min_theta_B, self.max_theta_B)
            psi_B_guess = self.uniform_sampling(self.min_psi_B, self.max_psi_B)
            
            phi_A_Q1_guess = self.uniform_sampling(self.min_phi_A_Q1, self.max_phi_A_Q1)
            theta_A_Q1_guess = self.uniform_sampling(self.min_theta_A_Q1, self.max_theta_A_Q1)
            psi_A_Q1_guess = self.uniform_sampling(self.min_psi_A_Q1, self.max_psi_A_Q1)
            phi_B_Q1_guess = self.uniform_sampling(self.min_phi_B_Q1, self.max_phi_B_Q1)
            theta_B_Q1_guess = self.uniform_sampling(self.min_theta_B_Q1, self.max_theta_B_Q1)
            psi_B_Q1_guess = self.uniform_sampling(self.min_psi_B_Q1, self.max_psi_B_Q1)
            
            phi_A_Q2_guess = self.uniform_sampling(self.min_phi_A_Q2, self.max_phi_A_Q2)
            theta_A_Q2_guess = self.uniform_sampling(self.min_theta_A_Q2, self.max_theta_A_Q2)
            psi_A_Q2_guess = self.uniform_sampling(self.min_psi_A_Q2, self.max_psi_A_Q2)
            phi_B_Q2_guess = self.uniform_sampling(self.min_phi_B_Q2, self.max_phi_B_Q2)
            theta_B_Q2_guess = self.uniform_sampling(self.min_theta_B_Q2, self.max_theta_B_Q2)
            psi_B_Q2_guess = self.uniform_sampling(self.min_psi_B_Q2, self.max_psi_B_Q2)
            
            phase_Q1_A_guess = self.uniform_sampling(self.min_phase_Q1_A, self.max_phase_Q1_A)
            phase_Q1_B_guess = self.uniform_sampling(self.min_phase_Q1_B, self.max_phase_Q1_B)
            phase_Q2_A_guess = self.uniform_sampling(self.min_phase_Q2_A, self.max_phase_Q2_A)
            phase_Q2_B_guess = self.uniform_sampling(self.min_phase_Q2_B, self.max_phase_Q2_B)
            
            initial_guess = [
                Q1_1_guess, Q1_2_guess, Q2_1_guess, Q2_2_guess,
                alpha_A_guess, alpha_B_guess, beta_Q1_A_guess, beta_Q1_B_guess, beta_Q2_A_guess, beta_Q2_B_guess,
                phi_A_guess, theta_A_guess, psi_A_guess, phi_B_guess, theta_B_guess, psi_B_guess,
                phi_A_Q1_guess, theta_A_Q1_guess, psi_A_Q1_guess, phi_B_Q1_guess, theta_B_Q1_guess, psi_B_Q1_guess,
                phi_A_Q2_guess, theta_A_Q2_guess, psi_A_Q2_guess, phi_B_Q2_guess, theta_B_Q2_guess, psi_B_Q2_guess,
                phase_Q1_A_guess, phase_Q1_B_guess, phase_Q2_A_guess, phase_Q2_B_guess
            ]
            
            # Minimize at that point
            res = minimize(self.E_per_UC, x0=initial_guess, bounds=self.parameter_bounds, 
                          method='Nelder-Mead', options={'maxiter': 5000}, tol=tol_first_opt)

            if res.fun < opt_energy:
                opt_params = res.x
                opt_energy = res.fun
        
        print(f"First phase complete. Best energy: {opt_energy:.8f}")
        print(f"Running final optimization with L-BFGS-B...")
        
        # Final optimization run on best parameters
        res = minimize(self.E_per_UC, x0=opt_params, bounds=self.parameter_bounds, 
                      method='L-BFGS-B', options={'maxiter': 10000}, tol=tol_second_opt)
        opt_params = res.x
        opt_energy = res.fun
        
        print(f"Optimization complete. Final energy: {opt_energy:.8f}")
        
        return opt_params, opt_energy
    
    def classify_phase(self, tol=10**-5):
        """Classify the magnetic ordering based on the optimal parameters"""
        (Q1_1, Q1_2, Q2_1, Q2_2,
         alpha_A, alpha_B, beta_Q1_A, beta_Q1_B, beta_Q2_A, beta_Q2_B,
         *_) = self.opt_params
        
        Q1_vec = Q1_1 * self.b1 + Q1_2 * self.b2
        Q2_vec = Q2_1 * self.b1 + Q2_2 * self.b2
        
        # Check if it's effectively single-Q
        if beta_Q1_A < tol and beta_Q1_B < tol:
            # Only Q2 is active
            return self._classify_single_q(Q2_1, Q2_2, alpha_A, alpha_B, tol)
        elif beta_Q2_A < tol and beta_Q2_B < tol:
            # Only Q1 is active
            return self._classify_single_q(Q1_1, Q1_2, alpha_A, alpha_B, tol)
        
        # Check for special double-Q points
        Q1_norm = np.linalg.norm(Q1_vec)
        Q2_norm = np.linalg.norm(Q2_vec)
        
        # Both Q vectors are non-zero - true double-Q order
        phase_info = f"Double-Q order: Q1=({Q1_1:.4f}, {Q1_2:.4f}), Q2=({Q2_1:.4f}, {Q2_2:.4f})"
        
        # Check for specific symmetric configurations
        if (np.abs(Q1_1 - 1/3) < tol and np.abs(Q1_2 - 1/3) < tol and 
            np.abs(Q2_1 - 1/3) < tol and np.abs(Q2_2) < tol):
            return "Double-Q at K-points (tetrahedral)"
        
        return phase_info
    
    def _classify_single_q(self, Q1, Q2, alphaA, alphaB, tol):
        """Helper to classify single-Q phases"""
        # Gamma point order
        if np.abs(Q1) < tol and np.abs(Q2) < tol:
            return "FM/AFM (Gamma point)"
        
        # M point order
        elif (np.abs(Q1-0.5) < tol and np.abs(Q2) < tol) or (np.abs(Q2-0.5) < tol and np.abs(Q1) < tol):
            return "Zigzag/Stripy (M point)"
        
        # K point order
        elif np.abs(Q1-1/3) < tol and np.abs(Q2-1/3) < tol:
            return "120° order (K point)"
        
        # Incommensurate order
        else:
            return f"Incommensurate single-Q: ({Q1:.4f}, {Q2:.4f})"
    
    def generate_spin_configuration(self):
        """Generate the spin configuration from the optimal parameters"""
        (Q1_1, Q1_2, Q2_1, Q2_2,
         alpha_A, alpha_B, beta_Q1_A, beta_Q1_B, beta_Q2_A, beta_Q2_B,
         phi_A, theta_A, psi_A, phi_B, theta_B, psi_B,
         phi_A_Q1, theta_A_Q1, psi_A_Q1, phi_B_Q1, theta_B_Q1, psi_B_Q1,
         phi_A_Q2, theta_A_Q2, psi_A_Q2, phi_B_Q2, theta_B_Q2, psi_B_Q2,
         phase_Q1_A, phase_Q1_B, phase_Q2_A, phase_Q2_B) = self.opt_params
        
        Q1 = Q1_1 * self.b1 + Q1_2 * self.b2
        Q2 = Q2_1 * self.b1 + Q2_2 * self.b2
        
        N = len(self.positions)
        spins = np.zeros((N, 3))
        
        exA, eyA, ezA = self.RotatedBasis(phi_A, theta_A, psi_A)
        exB, eyB, ezB = self.RotatedBasis(phi_B, theta_B, psi_B)
        
        exA_Q1, eyA_Q1, ezA_Q1 = self.RotatedBasis(phi_A_Q1, theta_A_Q1, psi_A_Q1)
        exB_Q1, eyB_Q1, ezB_Q1 = self.RotatedBasis(phi_B_Q1, theta_B_Q1, psi_B_Q1)
        
        exA_Q2, eyA_Q2, ezA_Q2 = self.RotatedBasis(phi_A_Q2, theta_A_Q2, psi_A_Q2)
        exB_Q2, eyB_Q2, ezB_Q2 = self.RotatedBasis(phi_B_Q2, theta_B_Q2, psi_B_Q2)
        
        for i in range(N):
            pos = self.positions[i]
            if i % 2 == 0:  # A sublattice
                spin = alpha_A * ezA
                spin += np.sqrt(beta_Q1_A) * (exA_Q1 * np.cos(Q1.dot(pos) + phase_Q1_A) + 
                                               eyA_Q1 * np.sin(Q1.dot(pos) + phase_Q1_A))
                spin += np.sqrt(beta_Q2_A) * (exA_Q2 * np.cos(Q2.dot(pos) + phase_Q2_A) + 
                                               eyA_Q2 * np.sin(Q2.dot(pos) + phase_Q2_A))
            else:  # B sublattice
                spin = alpha_B * ezB
                spin += np.sqrt(beta_Q1_B) * (exB_Q1 * np.cos(Q1.dot(pos) + phase_Q1_B) + 
                                               eyB_Q1 * np.sin(Q1.dot(pos) + phase_Q1_B))
                spin += np.sqrt(beta_Q2_B) * (exB_Q2 * np.cos(Q2.dot(pos) + phase_Q2_B) + 
                                               eyB_Q2 * np.sin(Q2.dot(pos) + phase_Q2_B))
            
            # Normalize the spin
            spin_norm = np.linalg.norm(spin)
            if spin_norm > 1e-10:
                spins[i] = spin / spin_norm
            else:
                spins[i] = np.array([0, 0, 1])  # Default direction if spin is zero
        
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


if __name__ == "__main__":
    # Size of lattice (L x L unit cells)
    L = 8  # Use smaller lattice due to increased complexity
    
    # BCAO parameters matching the C++ convention: [J1xy, J1z, D, E, F, G, J3xy, J3z]
    J = [-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85]
    
    # Create double-Q model
    print("="*60)
    print("DOUBLE-Q ANSATZ FOR BCAO HONEYCOMB LATTICE")
    print("="*60)
    print(f"Lattice size: {L}x{L}")
    print(f"BCAO Parameters: J1xy={J[0]}, J1z={J[1]}, D={J[2]}, E={J[3]}, F={J[4]}, G={J[5]}, J3xy={J[6]}, J3z={J[7]}")
    print()
    
    model = DoubleQ(L, J)
    
    # Print results
    Q1_1, Q1_2, Q2_1, Q2_2 = model.opt_params[0], model.opt_params[1], model.opt_params[2], model.opt_params[3]
    alpha_A, alpha_B = model.opt_params[4], model.opt_params[5]
    beta_Q1_A, beta_Q1_B = model.opt_params[6], model.opt_params[7]
    beta_Q2_A, beta_Q2_B = model.opt_params[8], model.opt_params[9]
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Ground state energy per unit cell: {model.opt_energy:.8f}")
    print(f"\nQ-vectors:")
    print(f"  Q1 = ({Q1_1:.6f}, {Q1_2:.6f})")
    print(f"  Q2 = ({Q2_1:.6f}, {Q2_2:.6f})")
    print(f"\nAmplitudes (A sublattice):")
    print(f"  α_A (uniform) = {alpha_A:.6f}")
    print(f"  β_Q1_A = {beta_Q1_A:.6f}")
    print(f"  β_Q2_A = {beta_Q2_A:.6f}")
    print(f"  Normalization check: α_A² + β_Q1_A + β_Q2_A = {alpha_A**2 + beta_Q1_A + beta_Q2_A:.6f}")
    print(f"\nAmplitudes (B sublattice):")
    print(f"  α_B (uniform) = {alpha_B:.6f}")
    print(f"  β_Q1_B = {beta_Q1_B:.6f}")
    print(f"  β_Q2_B = {beta_Q2_B:.6f}")
    print(f"  Normalization check: α_B² + β_Q1_B + β_Q2_B = {alpha_B**2 + beta_Q1_B + beta_Q2_B:.6f}")
    print(f"\nMagnetic order: {model.magnetic_order}")
    
    # Generate and analyze the spin configuration
    spins = model.generate_spin_configuration()
    
    # Calculate magnetization
    magnetization = model.calculate_magnetization(spins)
    
    # Print magnetization results
    print("\n" + "="*60)
    print("MAGNETIZATION ANALYSIS")
    print("="*60)
    print(f"Total magnetization: [{magnetization['total'][0]:.6f}, {magnetization['total'][1]:.6f}, {magnetization['total'][2]:.6f}]")
    print(f"Total magnetization magnitude: {magnetization['total_magnitude']:.6f}")
    print(f"\nSublattice A magnetization: [{magnetization['A'][0]:.6f}, {magnetization['A'][1]:.6f}, {magnetization['A'][2]:.6f}]")
    print(f"Sublattice A magnitude: {magnetization['A_magnitude']:.6f}")
    print(f"\nSublattice B magnetization: [{magnetization['B'][0]:.6f}, {magnetization['B'][1]:.6f}, {magnetization['B'][2]:.6f}]")
    print(f"Sublattice B magnitude: {magnetization['B_magnitude']:.6f}")
    print(f"\nStaggered magnetization: [{magnetization['staggered'][0]:.6f}, {magnetization['staggered'][1]:.6f}, {magnetization['staggered'][2]:.6f}]")
    print(f"Staggered magnetization magnitude: {magnetization['staggered_magnitude']:.6f}")
    print("="*60)
    
    # Visualize the spin configuration
    visualize_spins(model.positions, spins, L)
