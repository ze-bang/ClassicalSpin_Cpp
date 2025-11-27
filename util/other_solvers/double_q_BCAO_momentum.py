import numpy as np
from scipy.optimize import minimize
from luttinger_tisza import create_honeycomb_lattice, construct_interaction_matrices, get_bond_vectors, visualize_spins

# filepath: /home/pc_linux/ClassicalSpin_Cpp/util/double_q_BCAO_momentum.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DoubleQ_Momentum:
    """
    Class to perform fully general double-Q ansatz simulation on a BCAO honeycomb lattice
    using momentum-space energy evaluation (Luttinger-Tisza approach).
    
    Uses general double-Q ansatz with Q^(1) || b1 and Q^(2) || b2:
        S_X(r) = alpha_X * e_z^X 
               + gamma_X^(1) * [e_x^X * cos(Q^(1)·r) + e_y^X * sin(Q^(1)·r)]
               + gamma_X^(2) * [e_x^X * cos(Q^(2)·r) + e_y^X * sin(Q^(2)·r)]
    where:
        alpha_X = cos(theta_X)
        gamma_X^(1) = sin(theta_X) * cos(phi_X)
        gamma_X^(2) = sin(theta_X) * sin(phi_X)
    
    This parameterization ensures alpha_X^2 + (gamma_X^(1))^2 + (gamma_X^(2))^2 = 1.
    
    This is a direct extension of single_q_BCAO.py using the same momentum-space
    machinery. The energy decomposes into independent contributions from q=0, ±Q^(1), ±Q^(2).
    
    Uses the BCAO parameter convention: J = [J1xy, J1z, D, E, F, G, J3xy, J3z]
    matching the C++ implementation in simulated_annealing_BCAO_emily.cpp
    """
    # Define parameter bounds
    eta_small = 10**-9
    min_Q1, max_Q1 = (0, 0.5-eta_small)
    min_Q2, max_Q2 = (0, 0.5)
    min_theta_amp_A, max_theta_amp_A = (0, np.pi/2)  # theta for amplitude splitting
    min_theta_amp_B, max_theta_amp_B = (0, np.pi/2)
    min_phi_amp_A, max_phi_amp_A = (0, np.pi/2)      # phi for amplitude splitting
    min_phi_amp_B, max_phi_amp_B = (0, np.pi/2)
    min_phi_A, max_phi_A = (0, (2-eta_small)*np.pi)
    min_theta_A, max_theta_A = (0, (2-eta_small)*np.pi)
    min_psi_A, max_psi_A = (0, (2-eta_small)*np.pi)
    min_phi_B, max_phi_B = (0, (2-eta_small)*np.pi)
    min_theta_B, max_theta_B = (0, (2-eta_small)*np.pi)
    min_psi_B, max_psi_B = (0, (2-eta_small)*np.pi)
    parameter_bounds = [
        (min_Q1, max_Q1), (min_Q2, max_Q2), 
        (min_theta_amp_A, max_theta_amp_A), (min_phi_amp_A, max_phi_amp_A),
        (min_theta_amp_B, max_theta_amp_B), (min_phi_amp_B, max_phi_amp_B),
        (min_phi_A, max_phi_A), (min_theta_A, max_theta_A), (min_psi_A, max_psi_A),
        (min_phi_B, max_phi_B), (min_theta_B, max_theta_B), (min_psi_B, max_psi_B),
    ]
    
    def __init__(self, L=4, J=[-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85], B_field=np.array([0, 0, 0]), equal_amplitudes=False, orthogonal_mode=False):
        """
        Initialize the double-Q model with momentum-space energy evaluation
        
        Args:
            L: Size of the lattice (L x L unit cells)
            J: BCAO exchange coupling parameters [J1xy, J1z, D, E, F, G, J3xy, J3z]
            B_field: External magnetic field
            equal_amplitudes: If True, constrain gamma_X^(1) = gamma_X^(2) for symmetric double-Q states
            orthogonal_mode: If True, use orthogonal ansatz where x spirals along Q1 and y spirals along Q2
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
        self.equal_amplitudes = equal_amplitudes
        self.orthogonal_mode = orthogonal_mode
        
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
    
    def E_single_Q_block(self, Qvec, exA, eyA, ezA, exB, eyB, ezB, gammaA, gammaB):
        """
        Calculate the energy contribution from a single Q-vector component.
        
        This evaluates the modulated part of the energy for one wavevector Q using
        the momentum-space formulation:
        
        E(Q) = (gamma_A * gamma_B / 4) * [
                 (e_-^A · H_AB(Q) · e_+^B) + (e_-^B · H_BA(Q) · e_+^A) +
                 (e_+^A · H_AB(-Q) · e_-^B) + (e_+^B · H_BA(-Q) · e_-^A)
               ]
             + (gamma_A^2 / 4) * [
                 (e_-^A · H_AA(Q) · e_+^A) + (e_+^A · H_AA(-Q) · e_-^A)
               ]
             + (gamma_B^2 / 4) * [
                 (e_-^B · H_BB(Q) · e_+^B) + (e_+^B · H_BB(-Q) · e_-^B)
               ]
        
        where e_± = e_x ± i*e_y are circular basis vectors.
        
        Args:
            Qvec: The wavevector
            exA, eyA, ezA: Local basis vectors for sublattice A
            exB, eyB, ezB: Local basis vectors for sublattice B
            gammaA, gammaB: Transverse amplitude factors for each sublattice
        
        Returns:
            Energy contribution from this Q-vector
        """
        HAB_Q = self.HAB(Qvec)
        HBA_Q = self.HBA(Qvec)
        HAA_Q = self.HAA(Qvec)
        HBB_Q = self.HBB(Qvec)
        
        HAB_mQ = self.HAB(-Qvec)
        HBA_mQ = self.HBA(-Qvec)
        HAA_mQ = self.HAA(-Qvec)
        HBB_mQ = self.HBB(-Qvec)
        
        # Complex unit
        i = 1j
        
        # A-B cross terms
        E_q = gammaA * gammaB / 4.0 * (
            np.dot(exA - i * eyA, np.dot(HAB_Q, exB + i * eyB)) +
            np.dot(exB - i * eyB, np.dot(HBA_Q, exA + i * eyA)) +
            np.dot(exA + i * eyA, np.dot(HAB_mQ, exB - i * eyB)) +
            np.dot(exB + i * eyB, np.dot(HBA_mQ, exA - i * eyA))
        )
        
        # A-A terms
        E_q += gammaA**2 / 4.0 * (
            np.dot(exA - i * eyA, np.dot(HAA_Q, exA + i * eyA)) +
            np.dot(exA + i * eyA, np.dot(HAA_mQ, exA - i * eyA))
        )
        
        # B-B terms
        E_q += gammaB**2 / 4.0 * (
            np.dot(exB - i * eyB, np.dot(HBB_Q, exB + i * eyB)) +
            np.dot(exB + i * eyB, np.dot(HBB_mQ, exB - i * eyB))
        )
        
        return E_q
    
    def E_per_UC(self, params):
        """
        Calculate the energy per unit cell for the fully general double-Q ansatz.
        
        The total energy decomposes into three independent pieces:
        
        E_total = E_q0 + E_Q1 + E_Q2 + E_Zeeman
        
        where:
        - E_q0: Uniform (q=0) component from alpha_A, alpha_B
        - E_Q1: Modulated component from Q^(1) = Q1*b1 with amplitudes gamma_X^(1)
        - E_Q2: Modulated component from Q^(2) = Q2*b2 with amplitudes gamma_X^(2)
        - E_Zeeman: Zeeman coupling to magnetic field
        
        The amplitudes are parameterized as:
            alpha_X = cos(theta_amp_X)
            gamma_X^(1) = sin(theta_amp_X) * cos(phi_amp_X)
            gamma_X^(2) = sin(theta_amp_X) * sin(phi_amp_X)
        ensuring alpha_X^2 + (gamma_X^(1))^2 + (gamma_X^(2))^2 = 1.
        
        If equal_amplitudes=True, then phi_amp_X is fixed to pi/4, giving gamma_X^(1) = gamma_X^(2).
        If orthogonal_mode=True, uses orthogonal ansatz with independent x and y spirals.
        """
        if self.orthogonal_mode:
            # For orthogonal mode, fall back to regular calculation for now
            pass
        
        if self.equal_amplitudes:
            Q1, Q2, theta_amp_A, theta_amp_B, phi_A, theta_A, psi_A, phi_B, theta_B, psi_B = params
            phi_amp_A = np.pi / 4
            phi_amp_B = np.pi / 4
        else:
            Q1, Q2, theta_amp_A, phi_amp_A, theta_amp_B, phi_amp_B, phi_A, theta_A, psi_A, phi_B, theta_B, psi_B = params
        
        # The two wavevectors: Q^(1) = Q1*b1, Q^(2) = Q2*b2
        Qvec1 = Q1 * self.b1
        Qvec2 = Q2 * self.b2
        
        # Get rotated basis vectors
        exA, eyA, ezA = self.RotatedBasis(phi_A, theta_A, psi_A)
        exB, eyB, ezB = self.RotatedBasis(phi_B, theta_B, psi_B)
        
        # Compute amplitudes using spherical parameterization
        alpha_A = np.cos(theta_amp_A)
        gamma_A1 = np.sin(theta_amp_A) * np.cos(phi_amp_A)
        gamma_A2 = np.sin(theta_amp_A) * np.sin(phi_amp_A)
        
        alpha_B = np.cos(theta_amp_B)
        gamma_B1 = np.sin(theta_amp_B) * np.cos(phi_amp_B)
        gamma_B2 = np.sin(theta_amp_B) * np.sin(phi_amp_B)
        
        # Uniform component contribution (q=0, unchanged from single-Q)
        E_q0 = alpha_A * alpha_B * (ezA.dot(self.HAB(np.array([0, 0]))).dot(ezB) + 
                                    ezB.dot(self.HBA(np.array([0, 0]))).dot(ezA))
        E_q0 += alpha_A**2 * ezA.dot(self.HAA(np.array([0, 0]))).dot(ezA)
        E_q0 += alpha_B**2 * ezB.dot(self.HBB(np.array([0, 0]))).dot(ezB)
        
        # Modulated component: sum of two single-Q pieces with independent amplitudes
        E_q = 0.0
        E_q += self.E_single_Q_block(Qvec1, exA, eyA, ezA, exB, eyB, ezB, gamma_A1, gamma_B1)
        E_q += self.E_single_Q_block(Qvec2, exA, eyA, ezA, exB, eyB, ezB, gamma_A2, gamma_B2)
        
        # Zeeman energy contribution (unchanged)
        E_zeeman = self.B_field.dot(ezA * alpha_A + ezB * alpha_B)
        
        return np.real(E_q0 / 4 + E_q / 4 + E_zeeman / 2)
    
    def find_minimum_energy(self, N_ITERATIONS=20, tol_first_opt=10**-8, tol_second_opt=10**-10):
        """Find the optimal parameters that minimize the energy"""
        if self.equal_amplitudes:
            # Reduced parameter space: no phi_amp_A and phi_amp_B
            opt_params = [0] * 10
            opt_energy = 10**10
            
            # Reduced bounds
            bounds = [
                (self.min_Q1, self.max_Q1), (self.min_Q2, self.max_Q2),
                (self.min_theta_amp_A, self.max_theta_amp_A),
                (self.min_theta_amp_B, self.max_theta_amp_B),
                (self.min_phi_A, self.max_phi_A), (self.min_theta_A, self.max_theta_A), (self.min_psi_A, self.max_psi_A),
                (self.min_phi_B, self.max_phi_B), (self.min_theta_B, self.max_theta_B), (self.min_psi_B, self.max_psi_B),
            ]
            
            for i in range(N_ITERATIONS):
                Q1_guess = self.uniform_sampling(self.min_Q1, self.max_Q1)
                Q2_guess = self.uniform_sampling(self.min_Q2, self.max_Q2)
                theta_amp_A_guess = self.uniform_sampling(self.min_theta_amp_A, self.max_theta_amp_A)
                theta_amp_B_guess = self.uniform_sampling(self.min_theta_amp_B, self.max_theta_amp_B)
                phi_A_guess = self.uniform_sampling(self.min_phi_A, self.max_phi_A)
                theta_A_guess = self.uniform_sampling(self.min_theta_A, self.max_theta_A)
                psi_A_guess = self.uniform_sampling(self.min_psi_A, self.max_psi_A)
                phi_B_guess = self.uniform_sampling(self.min_phi_B, self.max_phi_B)
                theta_B_guess = self.uniform_sampling(self.min_theta_B, self.max_theta_B)
                psi_B_guess = self.uniform_sampling(self.min_psi_B, self.max_psi_B)
                
                initial_guess = [Q1_guess, Q2_guess, theta_amp_A_guess, theta_amp_B_guess,
                                phi_A_guess, theta_A_guess, psi_A_guess,
                                phi_B_guess, theta_B_guess, psi_B_guess]
                
                res = minimize(self.E_per_UC, x0=initial_guess, bounds=bounds,
                              method='Nelder-Mead', tol=tol_first_opt)
                
                if res.fun < opt_energy:
                    opt_params = res.x
                    opt_energy = res.fun
            
            res = minimize(self.E_per_UC, x0=opt_params, bounds=bounds,
                          method='L-BFGS-B', tol=tol_second_opt)
            opt_params = res.x
            opt_energy = res.fun
            
            if np.abs(opt_params[2]) < 10**-10 and np.abs(opt_params[3]) < 10**-10:
                opt_params[0] = 0
                opt_params[1] = 0
        else:
            # Full parameter space
            opt_params = [0] * 12
            opt_energy = 10**10
            
            for i in range(N_ITERATIONS):
                # Random initial guess
                Q1_guess = self.uniform_sampling(self.min_Q1, self.max_Q1)
                Q2_guess = self.uniform_sampling(self.min_Q2, self.max_Q2)
                theta_amp_A_guess = self.uniform_sampling(self.min_theta_amp_A, self.max_theta_amp_A)
                phi_amp_A_guess = self.uniform_sampling(self.min_phi_amp_A, self.max_phi_amp_A)
                theta_amp_B_guess = self.uniform_sampling(self.min_theta_amp_B, self.max_theta_amp_B)
                phi_amp_B_guess = self.uniform_sampling(self.min_phi_amp_B, self.max_phi_amp_B)
                phi_A_guess = self.uniform_sampling(self.min_phi_A, self.max_phi_A)
                theta_A_guess = self.uniform_sampling(self.min_theta_A, self.max_theta_A)
                psi_A_guess = self.uniform_sampling(self.min_psi_A, self.max_psi_A)
                phi_B_guess = self.uniform_sampling(self.min_phi_B, self.max_phi_B)
                theta_B_guess = self.uniform_sampling(self.min_theta_B, self.max_theta_B)
                psi_B_guess = self.uniform_sampling(self.min_psi_B, self.max_psi_B)
                
                initial_guess = [Q1_guess, Q2_guess, theta_amp_A_guess, phi_amp_A_guess,
                                theta_amp_B_guess, phi_amp_B_guess,
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
            
            # If we have uniform magnetization (theta_amp ≈ 0), set Q to zero
            if np.abs(opt_params[2]) < 10**-10 and np.abs(opt_params[4]) < 10**-10:
                opt_params[0] = 0
                opt_params[1] = 0
        
        return opt_params, opt_energy
    
    def classify_phase(self, tol=10**-6):
        """Classify the magnetic ordering based on the optimal parameters"""
        if self.equal_amplitudes:
            Q1, Q2, theta_amp_A, theta_amp_B, phiA, thetaA, psiA, phiB, thetaB, psiB = self.opt_params
            phi_amp_A = np.pi / 4
            phi_amp_B = np.pi / 4
        else:
            Q1, Q2, theta_amp_A, phi_amp_A, theta_amp_B, phi_amp_B, phiA, thetaA, psiA, phiB, thetaB, psiB = self.opt_params
        Qvec1 = Q1 * self.b1
        Qvec2 = Q2 * self.b2
        
        # Compute amplitudes
        alphaA = np.cos(theta_amp_A)
        alphaB = np.cos(theta_amp_B)
        
        # Check if both Q's are zero (uniform order)
        if np.abs(Q1) < tol and np.abs(Q2) < tol:
            exA, eyA, ezA = self.RotatedBasis(phiA, thetaA, psiA)
            exB, eyB, ezB = self.RotatedBasis(phiB, thetaB, psiB)
            spin_a_0 = ezA * alphaA
            spin_b_0 = ezB * alphaB
            dot_product = spin_a_0.dot(spin_b_0)
            if dot_product > 0:
                return "FM"
            else:
                return "AFM"
        
        # Check if only one Q is active (single-Q order)
        if np.abs(Q1) < tol and np.abs(Q2) > tol:
            # Only Q2 active
            if np.abs(Q2 - 0.5) < tol:
                return "Single-Q: Zigzag/Stripy (Q||b2)"
            elif np.abs(Q2 - 1/3) < tol:
                return "Single-Q: 120° order (Q||b2)"
            else:
                return f"Single-Q: Incommensurate (Q||b2, Q2={Q2:.4f})"
        
        if np.abs(Q2) < tol and np.abs(Q1) > tol:
            # Only Q1 active
            if np.abs(Q1 - 0.5) < tol:
                return "Single-Q: Zigzag/Stripy (Q||b1)"
            elif np.abs(Q1 - 1/3) < tol:
                return "Single-Q: 120° order (Q||b1)"
            else:
                return f"Single-Q: Incommensurate (Q||b1, Q1={Q1:.4f})"
        
        # Both Q's are active (true double-Q order)
        # Check for special high-symmetry cases
        if np.abs(Q1 - 0.5) < tol and np.abs(Q2 - 0.5) < tol:
            return "Double-Q: M-point order"
        elif np.abs(Q1 - 1/3) < tol and np.abs(Q2 - 1/3) < tol:
            return "Double-Q: K-point order"
        else:
            return f"Double-Q: General (Q1={Q1:.4f}, Q2={Q2:.4f})"
    
    def generate_spin_configuration(self):
        """Generate the spin configuration from the optimal parameters"""
        if self.equal_amplitudes:
            Q1, Q2, theta_amp_A, theta_amp_B, phiA, thetaA, psiA, phiB, thetaB, psiB = self.opt_params
            phi_amp_A = np.pi / 4
            phi_amp_B = np.pi / 4
        else:
            Q1, Q2, theta_amp_A, phi_amp_A, theta_amp_B, phi_amp_B, phiA, thetaA, psiA, phiB, thetaB, psiB = self.opt_params
        Qvec1 = Q1 * self.b1
        Qvec2 = Q2 * self.b2
        
        N = len(self.positions)
        spins = np.zeros((N, 3))
        
        exA, eyA, ezA = self.RotatedBasis(phiA, thetaA, psiA)
        exB, eyB, ezB = self.RotatedBasis(phiB, thetaB, psiB)
        
        # General amplitudes from spherical parameterization
        alphaA = np.cos(theta_amp_A)
        gamma_A1 = np.sin(theta_amp_A) * np.cos(phi_amp_A)
        gamma_A2 = np.sin(theta_amp_A) * np.sin(phi_amp_A)
        
        alphaB = np.cos(theta_amp_B)
        gamma_B1 = np.sin(theta_amp_B) * np.cos(phi_amp_B)
        gamma_B2 = np.sin(theta_amp_B) * np.sin(phi_amp_B)
        
        for i in range(N):
            pos = self.positions[i]
            phase1 = Qvec1.dot(pos)
            phase2 = Qvec2.dot(pos)
            
            if i % 2 == 0:  # A sublattice
                spin = (
                    ezA * alphaA +
                    gamma_A1 * (exA * np.cos(phase1) + eyA * np.sin(phase1)) +
                    gamma_A2 * (exA * np.cos(phase2) + eyA * np.sin(phase2))
                )
            else:  # B sublattice
                spin = (
                    ezB * alphaB +
                    gamma_B1 * (exB * np.cos(phase1) + eyB * np.sin(phase1)) +
                    gamma_B2 * (exB * np.cos(phase2) + eyB * np.sin(phase2))
                )
            
            # Normalize the spin for visualization
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


    def visualize_with_arbitrary_Q(self, Q1, Q2):
        """
        Visualize the spin configuration for arbitrary Q1 and Q2 values.
        This uses the optimal amplitudes and basis orientations from the ground state,
        but allows you to explore different Q-vectors.
        
        Args:
            Q1: Q-vector component along b1 (0 to 0.5)
            Q2: Q-vector component along b2 (0 to 0.5)
        """
        # Use the optimal parameters but replace Q1 and Q2
        params = self.opt_params.copy()
        params[0] = Q1
        params[1] = Q2
        
        # Generate configuration with these Q values
        if self.equal_amplitudes:
            _, _, theta_amp_A, theta_amp_B, phiA, thetaA, psiA, phiB, thetaB, psiB = params
            phi_amp_A = np.pi / 4
            phi_amp_B = np.pi / 4
        else:
            _, _, theta_amp_A, phi_amp_A, theta_amp_B, phi_amp_B, phiA, thetaA, psiA, phiB, thetaB, psiB = params
        Qvec1 = Q1 * self.b1
        Qvec2 = Q2 * self.b2
        
        N = len(self.positions)
        spins = np.zeros((N, 3))
        
        exA, eyA, ezA = self.RotatedBasis(phiA, thetaA, psiA)
        exB, eyB, ezB = self.RotatedBasis(phiB, thetaB, psiB)
        
        alphaA = np.cos(theta_amp_A)
        gamma_A1 = np.sin(theta_amp_A) * np.cos(phi_amp_A)
        gamma_A2 = np.sin(theta_amp_A) * np.sin(phi_amp_A)
        
        alphaB = np.cos(theta_amp_B)
        gamma_B1 = np.sin(theta_amp_B) * np.cos(phi_amp_B)
        gamma_B2 = np.sin(theta_amp_B) * np.sin(phi_amp_B)
        
        for i in range(N):
            pos = self.positions[i]
            phase1 = Qvec1.dot(pos)
            phase2 = Qvec2.dot(pos)
            
            if i % 2 == 0:  # A sublattice
                spin = (
                    ezA * alphaA +
                    gamma_A1 * (exA * np.cos(phase1) + eyA * np.sin(phase1)) +
                    gamma_A2 * (exA * np.cos(phase2) + eyA * np.sin(phase2))
                )
            else:  # B sublattice
                spin = (
                    ezB * alphaB +
                    gamma_B1 * (exB * np.cos(phase1) + eyB * np.sin(phase1)) +
                    gamma_B2 * (exB * np.cos(phase2) + eyB * np.sin(phase2))
                )
            
            spins[i] = spin / np.linalg.norm(spin)
        
        # Calculate energy and magnetization
        energy = self.E_per_UC(params)
        magnetization = self.calculate_magnetization(spins)
        
        # Print info
        print(f"\nExploring Q1={Q1:.4f}, Q2={Q2:.4f}")
        print(f"Energy per unit cell: {energy:.6f}")
        print(f"Total magnetization magnitude: {magnetization['total_magnitude']:.6f}")
        print(f"Staggered magnetization magnitude: {magnetization['staggered_magnitude']:.6f}")
        
        # Visualize
        visualize_spins(self.positions, spins, self.L)
        
        return spins, energy, magnetization


def interactive_slider_mode(equal_amplitudes=False, orthogonal_mode=False):
    """
    Interactive slider mode to explore spin configurations with all parameters.
    Uses matplotlib sliders to interactively adjust all parameters in real-time.
    
    Args:
        equal_amplitudes: If True, constrain gamma_X^(1) = gamma_X^(2)
        orthogonal_mode: If True, use orthogonal ansatz
    """
    from matplotlib.widgets import Slider, Button
    from mpl_toolkits.mplot3d import Axes3D
    
    print("="*60)
    print("Double-Q BCAO Interactive Slider Mode")
    if equal_amplitudes:
        print("(Equal amplitude mode: γ^(1) = γ^(2))")
    if orthogonal_mode:
        print("(Orthogonal mode: x spirals along Q1, y spirals along Q2)")
    print("="*60)
    
    # Size of lattice (L x L unit cells)
    L = 8  # Smaller for better interactivity
    
    # BCAO parameters matching the C++ convention
    J = [-6.646, -2.084, 0.675, 1.33, -1.516, 0.21, 1.697, 0.039]
    
    print("\nInitializing model and finding ground state...")
    model = DoubleQ_Momentum(L, J, equal_amplitudes=equal_amplitudes, orthogonal_mode=orthogonal_mode)
    
    # Get optimal parameters as initial values
    opt_params = model.opt_params
    
    # Create figure with subplots for spin visualization and sliders
    fig = plt.figure(figsize=(16, 10))
    
    # 3D plot for spin configuration
    ax_spins = fig.add_subplot(121, projection='3d')
    
    # Info text area
    ax_info = fig.add_axes([0.55, 0.85, 0.4, 0.1])
    ax_info.axis('off')
    
    # Slider axes - arrange based on mode
    if equal_amplitudes:
        # 10 parameters
        slider_height = 0.015
        slider_gap = 0.025
        slider_left = 0.55
        slider_width = 0.35
        slider_bottom_start = 0.05
        
        ax_Q1 = fig.add_axes([slider_left, slider_bottom_start + 9*slider_gap, slider_width, slider_height])
        ax_Q2 = fig.add_axes([slider_left, slider_bottom_start + 8*slider_gap, slider_width, slider_height])
        ax_theta_amp_A = fig.add_axes([slider_left, slider_bottom_start + 7*slider_gap, slider_width, slider_height])
        ax_theta_amp_B = fig.add_axes([slider_left, slider_bottom_start + 6*slider_gap, slider_width, slider_height])
        ax_phi_A = fig.add_axes([slider_left, slider_bottom_start + 5*slider_gap, slider_width, slider_height])
        ax_theta_A = fig.add_axes([slider_left, slider_bottom_start + 4*slider_gap, slider_width, slider_height])
        ax_psi_A = fig.add_axes([slider_left, slider_bottom_start + 3*slider_gap, slider_width, slider_height])
        ax_phi_B = fig.add_axes([slider_left, slider_bottom_start + 2*slider_gap, slider_width, slider_height])
        ax_theta_B = fig.add_axes([slider_left, slider_bottom_start + 1*slider_gap, slider_width, slider_height])
        ax_psi_B = fig.add_axes([slider_left, slider_bottom_start + 0*slider_gap, slider_width, slider_height])
        
        slider_Q1 = Slider(ax_Q1, 'Q1', 0.0, 0.499, valinit=opt_params[0])
        slider_Q2 = Slider(ax_Q2, 'Q2', 0.0, 0.5, valinit=opt_params[1])
        slider_theta_amp_A = Slider(ax_theta_amp_A, 'θ_amp_A', 0.0, np.pi/2, valinit=opt_params[2])
        slider_theta_amp_B = Slider(ax_theta_amp_B, 'θ_amp_B', 0.0, np.pi/2, valinit=opt_params[3])
        slider_phi_A = Slider(ax_phi_A, 'φ_A', 0.0, 2*np.pi, valinit=opt_params[4])
        slider_theta_A = Slider(ax_theta_A, 'θ_A', 0.0, 2*np.pi, valinit=opt_params[5])
        slider_psi_A = Slider(ax_psi_A, 'ψ_A', 0.0, 2*np.pi, valinit=opt_params[6])
        slider_phi_B = Slider(ax_phi_B, 'φ_B', 0.0, 2*np.pi, valinit=opt_params[7])
        slider_theta_B = Slider(ax_theta_B, 'θ_B', 0.0, 2*np.pi, valinit=opt_params[8])
        slider_psi_B = Slider(ax_psi_B, 'ψ_B', 0.0, 2*np.pi, valinit=opt_params[9])
        
        sliders = [slider_Q1, slider_Q2, slider_theta_amp_A, slider_theta_amp_B,
                  slider_phi_A, slider_theta_A, slider_psi_A,
                  slider_phi_B, slider_theta_B, slider_psi_B]
    else:
        # 12 parameters
        slider_height = 0.015
        slider_gap = 0.022
        slider_left = 0.55
        slider_width = 0.35
        slider_bottom_start = 0.05
        
        ax_Q1 = fig.add_axes([slider_left, slider_bottom_start + 11*slider_gap, slider_width, slider_height])
        ax_Q2 = fig.add_axes([slider_left, slider_bottom_start + 10*slider_gap, slider_width, slider_height])
        ax_theta_amp_A = fig.add_axes([slider_left, slider_bottom_start + 9*slider_gap, slider_width, slider_height])
        ax_phi_amp_A = fig.add_axes([slider_left, slider_bottom_start + 8*slider_gap, slider_width, slider_height])
        ax_theta_amp_B = fig.add_axes([slider_left, slider_bottom_start + 7*slider_gap, slider_width, slider_height])
        ax_phi_amp_B = fig.add_axes([slider_left, slider_bottom_start + 6*slider_gap, slider_width, slider_height])
        ax_phi_A = fig.add_axes([slider_left, slider_bottom_start + 5*slider_gap, slider_width, slider_height])
        ax_theta_A = fig.add_axes([slider_left, slider_bottom_start + 4*slider_gap, slider_width, slider_height])
        ax_psi_A = fig.add_axes([slider_left, slider_bottom_start + 3*slider_gap, slider_width, slider_height])
        ax_phi_B = fig.add_axes([slider_left, slider_bottom_start + 2*slider_gap, slider_width, slider_height])
        ax_theta_B = fig.add_axes([slider_left, slider_bottom_start + 1*slider_gap, slider_width, slider_height])
        ax_psi_B = fig.add_axes([slider_left, slider_bottom_start + 0*slider_gap, slider_width, slider_height])
        
        slider_Q1 = Slider(ax_Q1, 'Q1', 0.0, 0.499, valinit=opt_params[0])
        slider_Q2 = Slider(ax_Q2, 'Q2', 0.0, 0.5, valinit=opt_params[1])
        slider_theta_amp_A = Slider(ax_theta_amp_A, 'θ_amp_A', 0.0, np.pi/2, valinit=opt_params[2])
        slider_phi_amp_A = Slider(ax_phi_amp_A, 'φ_amp_A', 0.0, np.pi/2, valinit=opt_params[3])
        slider_theta_amp_B = Slider(ax_theta_amp_B, 'θ_amp_B', 0.0, np.pi/2, valinit=opt_params[4])
        slider_phi_amp_B = Slider(ax_phi_amp_B, 'φ_amp_B', 0.0, np.pi/2, valinit=opt_params[5])
        slider_phi_A = Slider(ax_phi_A, 'φ_A', 0.0, 2*np.pi, valinit=opt_params[6])
        slider_theta_A = Slider(ax_theta_A, 'θ_A', 0.0, 2*np.pi, valinit=opt_params[7])
        slider_psi_A = Slider(ax_psi_A, 'ψ_A', 0.0, 2*np.pi, valinit=opt_params[8])
        slider_phi_B = Slider(ax_phi_B, 'φ_B', 0.0, 2*np.pi, valinit=opt_params[9])
        slider_theta_B = Slider(ax_theta_B, 'θ_B', 0.0, 2*np.pi, valinit=opt_params[10])
        slider_psi_B = Slider(ax_psi_B, 'ψ_B', 0.0, 2*np.pi, valinit=opt_params[11])
        
        sliders = [slider_Q1, slider_Q2, slider_theta_amp_A, slider_phi_amp_A,
                  slider_theta_amp_B, slider_phi_amp_B,
                  slider_phi_A, slider_theta_A, slider_psi_A,
                  slider_phi_B, slider_theta_B, slider_psi_B]
    
    # Reset button
    ax_reset = fig.add_axes([0.75, 0.75, 0.1, 0.03])
    btn_reset = Button(ax_reset, 'Reset to GS')
    
    def update(val):
        """Update the spin configuration based on slider values"""
        # Get current parameter values
        if equal_amplitudes:
            params = [slider_Q1.val, slider_Q2.val, slider_theta_amp_A.val, slider_theta_amp_B.val,
                     slider_phi_A.val, slider_theta_A.val, slider_psi_A.val,
                     slider_phi_B.val, slider_theta_B.val, slider_psi_B.val]
            phi_amp_A = np.pi / 4
            phi_amp_B = np.pi / 4
            _, _, theta_amp_A, theta_amp_B, phiA, thetaA, psiA, phiB, thetaB, psiB = params
        else:
            params = [slider_Q1.val, slider_Q2.val, slider_theta_amp_A.val, slider_phi_amp_A.val,
                     slider_theta_amp_B.val, slider_phi_amp_B.val,
                     slider_phi_A.val, slider_theta_A.val, slider_psi_A.val,
                     slider_phi_B.val, slider_theta_B.val, slider_psi_B.val]
            _, _, theta_amp_A, phi_amp_A, theta_amp_B, phi_amp_B, phiA, thetaA, psiA, phiB, thetaB, psiB = params
        
        Q1, Q2 = params[0], params[1]
        Qvec1 = Q1 * model.b1
        Qvec2 = Q2 * model.b2
        
        # Generate spins with current parameters
        N = len(model.positions)
        spins = np.zeros((N, 3))
        
        exA, eyA, ezA = model.RotatedBasis(phiA, thetaA, psiA)
        exB, eyB, ezB = model.RotatedBasis(phiB, thetaB, psiB)
        
        if equal_amplitudes:
            alphaA = np.cos(theta_amp_A)
            gamma_A1 = np.sin(theta_amp_A) * np.cos(phi_amp_A)
            gamma_A2 = np.sin(theta_amp_A) * np.sin(phi_amp_A)
            alphaB = np.cos(theta_amp_B)
            gamma_B1 = np.sin(theta_amp_B) * np.cos(phi_amp_B)
            gamma_B2 = np.sin(theta_amp_B) * np.sin(phi_amp_B)
        else:
            alphaA = np.cos(theta_amp_A)
            gamma_A1 = np.sin(theta_amp_A) * np.cos(phi_amp_A)
            gamma_A2 = np.sin(theta_amp_A) * np.sin(phi_amp_A)
            alphaB = np.cos(theta_amp_B)
            gamma_B1 = np.sin(theta_amp_B) * np.cos(phi_amp_B)
            gamma_B2 = np.sin(theta_amp_B) * np.sin(phi_amp_B)
        
        for i in range(N):
            pos = model.positions[i]
            phase1 = Qvec1.dot(pos)
            phase2 = Qvec2.dot(pos)
            
            if i % 2 == 0:  # A sublattice
                spin = (
                    ezA * alphaA +
                    gamma_A1 * (exA * np.cos(phase1) + eyA * np.sin(phase1)) +
                    gamma_A2 * (exA * np.cos(phase2) + eyA * np.sin(phase2))
                )
            else:  # B sublattice
                spin = (
                    ezB * alphaB +
                    gamma_B1 * (exB * np.cos(phase1) + eyB * np.sin(phase1)) +
                    gamma_B2 * (exB * np.cos(phase2) + eyB * np.sin(phase2))
                )
            
            spins[i] = spin / np.linalg.norm(spin)
        
        # Calculate energy and magnetization
        energy = model.E_per_UC(params)
        magnetization = model.calculate_magnetization(spins)
        
        # Update 3D plot
        ax_spins.clear()
        positions = model.positions
        
        # Plot spins as arrows
        for i in range(N):
            pos = positions[i]
            spin = spins[i]
            color = 'blue' if i % 2 == 0 else 'red'
            ax_spins.quiver(pos[0], pos[1], 0, spin[0], spin[1], spin[2],
                           length=0.3, color=color, arrow_length_ratio=0.3, linewidth=1.5)
        
        
        ax_spins.set_xlabel('X')
        ax_spins.set_ylabel('Y')
        ax_spins.set_zlabel('Z')
        ax_spins.set_title(f'Spin Configuration (L={L})', fontsize=12)
        ax_spins.set_box_aspect([1, 1, 0.5])
        
        # Update info text
        ax_info.clear()
        ax_info.axis('off')
        info_text = f"Energy: {energy:.6f}\n"
        info_text += f"Q1={Q1:.4f}, Q2={Q2:.4f}\n"
        info_text += f"|M_total|={magnetization['total_magnitude']:.4f}, "
        info_text += f"|M_stag|={magnetization['staggered_magnitude']:.4f}"
        ax_info.text(0, 0.5, info_text, fontsize=10, verticalalignment='center',
                    family='monospace')
        
        fig.canvas.draw_idle()
    
    def reset(event):
        """Reset sliders to ground state values"""
        for i, slider in enumerate(sliders):
            slider.set_val(opt_params[i])
    
    # Connect sliders to update function
    for slider in sliders:
        slider.on_changed(update)
    
    btn_reset.on_clicked(reset)
    
    # Initial plot
    update(None)
    
    plt.suptitle('Interactive Double-Q BCAO Explorer', fontsize=14, fontweight='bold')
    plt.show()


def graph_mode(equal_amplitudes=False):
    """
    Interactive graph mode to explore spin configurations with arbitrary Q1 and Q2.
    
    Args:
        equal_amplitudes: If True, constrain gamma_X^(1) = gamma_X^(2)
    """
    import sys
    
    print("="*60)
    print("Double-Q BCAO Exploration Mode")
    if equal_amplitudes:
        print("(Equal amplitude mode: γ^(1) = γ^(2))")
    print("="*60)
    
    # Size of lattice (L x L unit cells)
    L = 12
    
    # BCAO parameters matching the C++ convention: [J1xy, J1z, D, E, F, G, J3xy, J3z]
    J = [-6.646, -2.084, 0.675, 1.33, -1.516, 0.21, 1.697, 0.039]
    
    print("\nInitializing model and finding ground state...")
    print(f"BCAO Parameters: J1xy={J[0]}, J1z={J[1]}, D={J[2]}, E={J[3]}, F={J[4]}, G={J[5]}, J3xy={J[6]}, J3z={J[7]}")
    
    # Create model and find ground state
    model = DoubleQ_Momentum(L, J, equal_amplitudes=equal_amplitudes)
    
    Q1_opt, Q2_opt = model.opt_params[0:2]
    print(f"\nGround state: Q1={Q1_opt:.4f}, Q2={Q2_opt:.4f}")
    print(f"Ground state energy: {model.opt_energy:.6f}")
    print(f"Magnetic order: {model.magnetic_order}")
    
    print("\n" + "="*60)
    print("Enter Q1 and Q2 values to visualize (or 'q' to quit)")
    print("Q1 range: [0, 0.5), Q2 range: [0, 0.5]")
    print("Special points: 0.5 (zigzag/stripy), 0.333 (120° order)")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nEnter Q1 Q2 (e.g., '0.5 0.5' or 'q' to quit): ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Exiting graph mode.")
                break
            
            # Parse input
            parts = user_input.split()
            if len(parts) != 2:
                print("Please enter exactly two values: Q1 Q2")
                continue
            
            Q1 = float(parts[0])
            Q2 = float(parts[1])
            
            # Validate range
            if Q1 < 0 or Q1 >= 0.5:
                print(f"Q1 must be in range [0, 0.5), got {Q1}")
                continue
            if Q2 < 0 or Q2 > 0.5:
                print(f"Q2 must be in range [0, 0.5], got {Q2}")
                continue
            
            # Visualize with these Q values
            model.visualize_with_arbitrary_Q(Q1, Q2)
            
        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            print("\nExiting graph mode.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    # Check if interactive slider mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        equal_amp = '--equal' in sys.argv
        orthogonal = '--orthogonal' in sys.argv
        interactive_slider_mode(equal_amplitudes=equal_amp, orthogonal_mode=orthogonal)
    # Check if graph mode is requested
    elif len(sys.argv) > 1 and sys.argv[1] == '--graph':
        equal_amp = '--equal' in sys.argv
        graph_mode(equal_amplitudes=equal_amp)
    elif len(sys.argv) > 1 and sys.argv[1] == '--equal':
        # Run optimization with equal amplitude constraint
        L = 12
        J = [-6.646, -2.084, 0.675, 1.33, -1.516, 0.21, 1.697, 0.039]
        
        print("Running optimization with equal amplitude constraint (γ^(1) = γ^(2))...")
        model = DoubleQ_Momentum(L, J, equal_amplitudes=True)
        
        Q1, Q2, theta_amp_A, theta_amp_B = model.opt_params[0:4]
        
        alpha_A = np.cos(theta_amp_A)
        gamma_A = np.sin(theta_amp_A) / np.sqrt(2)  # gamma^(1) = gamma^(2) = sin(theta)/sqrt(2)
        alpha_B = np.cos(theta_amp_B)
        gamma_B = np.sin(theta_amp_B) / np.sqrt(2)
        
        print(f"\nOptimal Q-vectors: Q^(1) = {Q1:.4f}*b1, Q^(2) = {Q2:.4f}*b2")
        print(f"Ground state energy per unit cell: {model.opt_energy:.6f}")
        print(f"Magnetic order: {model.magnetic_order}")
        print(f"\nAmplitude distribution (equal mode):")
        print(f"  Sublattice A: alpha={alpha_A:.4f}, gamma^(1)=gamma^(2)={gamma_A:.4f}")
        print(f"  Sublattice B: alpha={alpha_B:.4f}, gamma^(1)=gamma^(2)={gamma_B:.4f}")
        print(f"  Normalization check A: {alpha_A**2 + 2*gamma_A**2:.6f}")
        print(f"  Normalization check B: {alpha_B**2 + 2*gamma_B**2:.6f}")
        print(f"\nBCAO Parameters: J1xy={J[0]}, J1z={J[1]}, D={J[2]}, E={J[3]}, F={J[4]}, G={J[5]}, J3xy={J[6]}, J3z={J[7]}")
        
        spins = model.generate_spin_configuration()
        magnetization = model.calculate_magnetization(spins)
        
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
        
        visualize_spins(model.positions, spins, L)
    else:
        # Original optimization mode
        # Size of lattice (L x L unit cells)
        L = 12
        
        # BCAO parameters matching the C++ convention: [J1xy, J1z, D, E, F, G, J3xy, J3z]
        # J = [-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85]
        # J = [-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]
        J = [-6.646, -2.084, 0.675, 1.33, -1.516, 0.21, 1.697, 0.039]
        
        # Create fully general double-Q model using momentum-space formulation
        model = DoubleQ_Momentum(L, J)
        
        # Extract optimal parameters
        Q1, Q2, theta_amp_A, phi_amp_A, theta_amp_B, phi_amp_B = model.opt_params[0:6]
        
        # Compute amplitudes
        alpha_A = np.cos(theta_amp_A)
        gamma_A1 = np.sin(theta_amp_A) * np.cos(phi_amp_A)
        gamma_A2 = np.sin(theta_amp_A) * np.sin(phi_amp_A)
        alpha_B = np.cos(theta_amp_B)
        gamma_B1 = np.sin(theta_amp_B) * np.cos(phi_amp_B)
        gamma_B2 = np.sin(theta_amp_B) * np.sin(phi_amp_B)
        
        # Print results
        print(f"Optimal Q-vectors: Q^(1) = {Q1:.4f}*b1, Q^(2) = {Q2:.4f}*b2")
        print(f"Ground state energy per unit cell: {model.opt_energy:.6f}")
        print(f"Magnetic order: {model.magnetic_order}")
        print(f"\nAmplitude distribution:")
        print(f"  Sublattice A: alpha={alpha_A:.4f}, gamma^(1)={gamma_A1:.4f}, gamma^(2)={gamma_A2:.4f}")
        print(f"  Sublattice B: alpha={alpha_B:.4f}, gamma^(1)={gamma_B1:.4f}, gamma^(2)={gamma_B2:.4f}")
        print(f"  Normalization check A: {alpha_A**2 + gamma_A1**2 + gamma_A2**2:.6f}")
        print(f"  Normalization check B: {alpha_B**2 + gamma_B1**2 + gamma_B2**2:.6f}")
        print(f"\nBCAO Parameters: J1xy={J[0]}, J1z={J[1]}, D={J[2]}, E={J[3]}, F={J[4]}, G={J[5]}, J3xy={J[6]}, J3z={J[7]}")
        
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
