import numpy as np
from scipy.optimize import minimize
from luttinger_tisza import (
    create_honeycomb_lattice,
    construct_interaction_matrices,
    get_bond_vectors,
    visualize_spins,
    calculate_energy_from_matrices,
)

"""
Double-Q ansatz for BCAO honeycomb lattice spin systems.

The spin configuration is parameterized as:
S_i = α * e_z + β₁ * (e_x1 * cos(Q1·r) + e_y1 * sin(Q1·r)) 
              + β₂ * (e_x2 * cos(Q2·r) + e_y2 * sin(Q2·r))

where α² + β₁² + β₂² = 1 (normalization constraint)

This allows capturing:
- Single-Q spirals (when β₂ = 0)
- Multi-Q phases like vortex crystals
- Skyrmion lattices (when Q1 and Q2 are related by symmetry)
"""

import matplotlib.pyplot as plt


class DoubleQ:
    """
    Class to perform double-Q ansatz simulation on a BCAO honeycomb lattice
    to determine ground state spin configuration and energy.
    
    Uses the BCAO parameter convention: J = [J1xy, J1z, D, E, F, G, J3xy, J3z]
    matching the C++ implementation.
    """
    # Define parameter bounds
    eta_small = 10**-9
    
    # Q-vector bounds (in reciprocal lattice units)
    min_Q1a, max_Q1a = (0, 0.5 - eta_small)
    min_Q1b, max_Q1b = (0, 0.5)
    min_Q2a, max_Q2a = (0, 0.5 - eta_small)
    min_Q2b, max_Q2b = (0, 0.5)
    
    # Euler angle bounds
    min_phi, max_phi = (0, (2 - eta_small) * np.pi)
    min_theta, max_theta = (0, (2 - eta_small) * np.pi)
    min_psi, max_psi = (0, (2 - eta_small) * np.pi)
    
    # Amplitude bounds - we use (alpha, beta1) and compute beta2 = sqrt(1 - alpha^2 - beta1^2)
    # To ensure valid amplitudes: alpha^2 + beta1^2 <= 1
    min_alpha, max_alpha = (0, 1.0)
    min_beta1, max_beta1 = (0, 1.0)
    
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
        
        # Parameter bounds for optimization
        # Parameters: [Q1a, Q1b, Q2a, Q2b, 
        #              alpha_A, beta1_A, alpha_B, beta1_B,
        #              phi_A1, theta_A1, psi_A1, phi_A2, theta_A2, psi_A2,
        #              phi_B1, theta_B1, psi_B1, phi_B2, theta_B2, psi_B2,
        #              phi_A0, theta_A0, phi_B0, theta_B0]
        self.parameter_bounds = [
            (self.min_Q1a, self.max_Q1a), (self.min_Q1b, self.max_Q1b),
            (self.min_Q2a, self.max_Q2a), (self.min_Q2b, self.max_Q2b),
            (self.min_alpha, self.max_alpha), (self.min_beta1, self.max_beta1),
            (self.min_alpha, self.max_alpha), (self.min_beta1, self.max_beta1),
            (self.min_phi, self.max_phi), (self.min_theta, self.max_theta), (self.min_psi, self.max_psi),
            (self.min_phi, self.max_phi), (self.min_theta, self.max_theta), (self.min_psi, self.max_psi),
            (self.min_phi, self.max_phi), (self.min_theta, self.max_theta), (self.min_psi, self.max_psi),
            (self.min_phi, self.max_phi), (self.min_theta, self.max_theta), (self.min_psi, self.max_psi),
            (self.min_phi, self.max_phi), (self.min_theta, self.max_theta),
            (self.min_phi, self.max_phi), (self.min_theta, self.max_theta),
        ]
        
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
    def uniform_direction(cls, phi, theta):
        """Get a unit vector from spherical angles"""
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

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
        return self.HAA(q)
    
    def parse_params(self, params):
        """Parse the parameter array into named components"""
        Q1a, Q1b, Q2a, Q2b = params[0:4]
        alpha_A, beta1_A, alpha_B, beta1_B = params[4:8]
        phi_A1, theta_A1, psi_A1 = params[8:11]
        phi_A2, theta_A2, psi_A2 = params[11:14]
        phi_B1, theta_B1, psi_B1 = params[14:17]
        phi_B2, theta_B2, psi_B2 = params[17:20]
        phi_A0, theta_A0 = params[20:22]
        phi_B0, theta_B0 = params[22:24]
        
        # Compute Q vectors
        Q1_vec = Q1a * self.b1 + Q1b * self.b2
        Q2_vec = Q2a * self.b1 + Q2b * self.b2
        
        # Compute beta2 from normalization constraint
        # alpha^2 + beta1^2 + beta2^2 = 1
        beta2_A_sq = max(0, 1 - alpha_A**2 - beta1_A**2)
        beta2_B_sq = max(0, 1 - alpha_B**2 - beta1_B**2)
        beta2_A = np.sqrt(beta2_A_sq)
        beta2_B = np.sqrt(beta2_B_sq)
        
        # Get rotated basis vectors for each spiral
        exA1, eyA1, ezA1 = self.RotatedBasis(phi_A1, theta_A1, psi_A1)
        exA2, eyA2, ezA2 = self.RotatedBasis(phi_A2, theta_A2, psi_A2)
        exB1, eyB1, ezB1 = self.RotatedBasis(phi_B1, theta_B1, psi_B1)
        exB2, eyB2, ezB2 = self.RotatedBasis(phi_B2, theta_B2, psi_B2)
        
        # Uniform magnetization directions
        ez_A0 = self.uniform_direction(phi_A0, theta_A0)
        ez_B0 = self.uniform_direction(phi_B0, theta_B0)
        
        return {
            'Q1_vec': Q1_vec, 'Q2_vec': Q2_vec,
            'Q1a': Q1a, 'Q1b': Q1b, 'Q2a': Q2a, 'Q2b': Q2b,
            'alpha_A': alpha_A, 'beta1_A': beta1_A, 'beta2_A': beta2_A,
            'alpha_B': alpha_B, 'beta1_B': beta1_B, 'beta2_B': beta2_B,
            'exA1': exA1, 'eyA1': eyA1, 'ezA1': ezA1,
            'exA2': exA2, 'eyA2': eyA2, 'ezA2': ezA2,
            'exB1': exB1, 'eyB1': eyB1, 'ezB1': ezB1,
            'exB2': exB2, 'eyB2': eyB2, 'ezB2': ezB2,
            'ez_A0': ez_A0, 'ez_B0': ez_B0
        }
    
    def check_valid_amplitudes(self, alpha, beta1):
        """Check if amplitudes satisfy normalization constraint"""
        return alpha**2 + beta1**2 <= 1.0
    
    def E_per_UC(self, params):
        """
        Calculate the energy per unit cell for the double-Q ansatz.
        
        The spin ansatz is:
        S_A(r) = α_A * n_A0 + β1_A * (ex_A1 * cos(Q1·r) + ey_A1 * sin(Q1·r))
                           + β2_A * (ex_A2 * cos(Q2·r) + ey_A2 * sin(Q2·r))
        
        and similarly for sublattice B.
        """
        p = self.parse_params(params)
        
        # Check normalization constraint
        if not self.check_valid_amplitudes(p['alpha_A'], p['beta1_A']):
            return 1e10
        if not self.check_valid_amplitudes(p['alpha_B'], p['beta1_B']):
            return 1e10
        
        Q1 = p['Q1_vec']
        Q2 = p['Q2_vec']
        zero = np.array([0, 0])
        
        # Extract components
        alpha_A, beta1_A, beta2_A = p['alpha_A'], p['beta1_A'], p['beta2_A']
        alpha_B, beta1_B, beta2_B = p['alpha_B'], p['beta1_B'], p['beta2_B']
        
        ez_A0 = p['ez_A0']
        ez_B0 = p['ez_B0']
        
        exA1, eyA1 = p['exA1'], p['eyA1']
        exA2, eyA2 = p['exA2'], p['eyA2']
        exB1, eyB1 = p['exB1'], p['eyB1']
        exB2, eyB2 = p['exB2'], p['eyB2']
        
        # Helper for complex spiral vectors
        def spiral_plus(ex, ey):
            return ex + 1j * ey
        
        def spiral_minus(ex, ey):
            return ex - 1j * ey
        
        E_total = 0.0
        
        # ===== Uniform component (q=0) =====
        # Contributions from α terms
        E_q0 = alpha_A * alpha_B * (ez_A0.dot(self.HAB(zero)).dot(ez_B0) + 
                                     ez_B0.dot(self.HBA(zero)).dot(ez_A0))
        E_q0 += alpha_A**2 * ez_A0.dot(self.HAA(zero)).dot(ez_A0)
        E_q0 += alpha_B**2 * ez_B0.dot(self.HBB(zero)).dot(ez_B0)
        E_total += E_q0 / 4
        
        # ===== First spiral Q1 contributions =====
        # A-B interactions at Q1
        E_Q1_AB = beta1_A * beta1_B / 4 * (
            spiral_minus(exA1, eyA1).dot(self.HAB(Q1)).dot(spiral_plus(exB1, eyB1)) +
            spiral_plus(exB1, eyB1).dot(self.HBA(Q1)).dot(spiral_minus(exA1, eyA1)) +
            spiral_plus(exA1, eyA1).dot(self.HAB(-Q1)).dot(spiral_minus(exB1, eyB1)) +
            spiral_minus(exB1, eyB1).dot(self.HBA(-Q1)).dot(spiral_plus(exA1, eyA1))
        )
        
        # A-A and B-B interactions at Q1
        E_Q1_AA = beta1_A**2 / 4 * (
            spiral_minus(exA1, eyA1).dot(self.HAA(Q1)).dot(spiral_plus(exA1, eyA1)) +
            spiral_plus(exA1, eyA1).dot(self.HAA(-Q1)).dot(spiral_minus(exA1, eyA1))
        )
        E_Q1_BB = beta1_B**2 / 4 * (
            spiral_minus(exB1, eyB1).dot(self.HBB(Q1)).dot(spiral_plus(exB1, eyB1)) +
            spiral_plus(exB1, eyB1).dot(self.HBB(-Q1)).dot(spiral_minus(exB1, eyB1))
        )
        
        E_total += (E_Q1_AB + E_Q1_AA + E_Q1_BB) / 4
        
        # ===== Second spiral Q2 contributions =====
        # A-B interactions at Q2
        E_Q2_AB = beta2_A * beta2_B / 4 * (
            spiral_minus(exA2, eyA2).dot(self.HAB(Q2)).dot(spiral_plus(exB2, eyB2)) +
            spiral_plus(exB2, eyB2).dot(self.HBA(Q2)).dot(spiral_minus(exA2, eyA2)) +
            spiral_plus(exA2, eyA2).dot(self.HAB(-Q2)).dot(spiral_minus(exB2, eyB2)) +
            spiral_minus(exB2, eyB2).dot(self.HBA(-Q2)).dot(spiral_plus(exA2, eyA2))
        )
        
        # A-A and B-B interactions at Q2
        E_Q2_AA = beta2_A**2 / 4 * (
            spiral_minus(exA2, eyA2).dot(self.HAA(Q2)).dot(spiral_plus(exA2, eyA2)) +
            spiral_plus(exA2, eyA2).dot(self.HAA(-Q2)).dot(spiral_minus(exA2, eyA2))
        )
        E_Q2_BB = beta2_B**2 / 4 * (
            spiral_minus(exB2, eyB2).dot(self.HBB(Q2)).dot(spiral_plus(exB2, eyB2)) +
            spiral_plus(exB2, eyB2).dot(self.HBB(-Q2)).dot(spiral_minus(exB2, eyB2))
        )
        
        E_total += (E_Q2_AB + E_Q2_AA + E_Q2_BB) / 4
        
        # ===== Cross terms between Q1 and Q2 spirals =====
        # These arise from products like cos(Q1·r)cos(Q2·r) which give contributions at Q1±Q2
        
        # Q1 + Q2 terms
        Q_plus = Q1 + Q2
        E_cross_plus_AB = beta1_A * beta2_B / 4 * (
            spiral_minus(exA1, eyA1).dot(self.HAB(Q_plus)).dot(spiral_plus(exB2, eyB2)) +
            spiral_plus(exA1, eyA1).dot(self.HAB(-Q_plus)).dot(spiral_minus(exB2, eyB2))
        )
        E_cross_plus_AB += beta2_A * beta1_B / 4 * (
            spiral_minus(exA2, eyA2).dot(self.HAB(Q_plus)).dot(spiral_plus(exB1, eyB1)) +
            spiral_plus(exA2, eyA2).dot(self.HAB(-Q_plus)).dot(spiral_minus(exB1, eyB1))
        )
        
        E_cross_plus_AA = beta1_A * beta2_A / 4 * (
            spiral_minus(exA1, eyA1).dot(self.HAA(Q_plus)).dot(spiral_plus(exA2, eyA2)) +
            spiral_plus(exA1, eyA1).dot(self.HAA(-Q_plus)).dot(spiral_minus(exA2, eyA2))
        )
        E_cross_plus_BB = beta1_B * beta2_B / 4 * (
            spiral_minus(exB1, eyB1).dot(self.HBB(Q_plus)).dot(spiral_plus(exB2, eyB2)) +
            spiral_plus(exB1, eyB1).dot(self.HBB(-Q_plus)).dot(spiral_minus(exB2, eyB2))
        )
        
        # Q1 - Q2 terms
        Q_minus = Q1 - Q2
        E_cross_minus_AB = beta1_A * beta2_B / 4 * (
            spiral_minus(exA1, eyA1).dot(self.HAB(Q_minus)).dot(spiral_minus(exB2, eyB2)) +
            spiral_plus(exA1, eyA1).dot(self.HAB(-Q_minus)).dot(spiral_plus(exB2, eyB2))
        )
        E_cross_minus_AB += beta2_A * beta1_B / 4 * (
            spiral_plus(exA2, eyA2).dot(self.HAB(Q_minus)).dot(spiral_plus(exB1, eyB1)) +
            spiral_minus(exA2, eyA2).dot(self.HAB(-Q_minus)).dot(spiral_minus(exB1, eyB1))
        )
        
        E_cross_minus_AA = beta1_A * beta2_A / 4 * (
            spiral_minus(exA1, eyA1).dot(self.HAA(Q_minus)).dot(spiral_minus(exA2, eyA2)) +
            spiral_plus(exA1, eyA1).dot(self.HAA(-Q_minus)).dot(spiral_plus(exA2, eyA2))
        )
        E_cross_minus_BB = beta1_B * beta2_B / 4 * (
            spiral_minus(exB1, eyB1).dot(self.HBB(Q_minus)).dot(spiral_minus(exB2, eyB2)) +
            spiral_plus(exB1, eyB1).dot(self.HBB(-Q_minus)).dot(spiral_plus(exB2, eyB2))
        )
        
        E_total += (E_cross_plus_AB + E_cross_plus_AA + E_cross_plus_BB) / 4
        E_total += (E_cross_minus_AB + E_cross_minus_AA + E_cross_minus_BB) / 4
        
        # ===== Zeeman energy =====
        E_zeeman = self.B_field.dot(ez_A0 * alpha_A + ez_B0 * alpha_B)
        E_total += E_zeeman / 2
        
        return np.real(E_total)
    
    def random_initial_guess(self):
        """Generate a random initial guess satisfying constraints"""
        Q1a = self.uniform_sampling(self.min_Q1a, self.max_Q1a)
        Q1b = self.uniform_sampling(self.min_Q1b, self.max_Q1b)
        Q2a = self.uniform_sampling(self.min_Q2a, self.max_Q2a)
        Q2b = self.uniform_sampling(self.min_Q2b, self.max_Q2b)
        
        # Sample amplitudes satisfying constraint
        alpha_A = self.uniform_sampling(0, 1)
        max_beta1_A = np.sqrt(max(0, 1 - alpha_A**2))
        beta1_A = self.uniform_sampling(0, max_beta1_A)
        
        alpha_B = self.uniform_sampling(0, 1)
        max_beta1_B = np.sqrt(max(0, 1 - alpha_B**2))
        beta1_B = self.uniform_sampling(0, max_beta1_B)
        
        # Euler angles for first spiral
        phi_A1 = self.uniform_sampling(self.min_phi, self.max_phi)
        theta_A1 = self.uniform_sampling(self.min_theta, self.max_theta)
        psi_A1 = self.uniform_sampling(self.min_psi, self.max_psi)
        
        phi_A2 = self.uniform_sampling(self.min_phi, self.max_phi)
        theta_A2 = self.uniform_sampling(self.min_theta, self.max_theta)
        psi_A2 = self.uniform_sampling(self.min_psi, self.max_psi)
        
        phi_B1 = self.uniform_sampling(self.min_phi, self.max_phi)
        theta_B1 = self.uniform_sampling(self.min_theta, self.max_theta)
        psi_B1 = self.uniform_sampling(self.min_psi, self.max_psi)
        
        phi_B2 = self.uniform_sampling(self.min_phi, self.max_phi)
        theta_B2 = self.uniform_sampling(self.min_theta, self.max_theta)
        psi_B2 = self.uniform_sampling(self.min_psi, self.max_psi)
        
        # Uniform direction angles
        phi_A0 = self.uniform_sampling(self.min_phi, self.max_phi)
        theta_A0 = self.uniform_sampling(0, np.pi)
        phi_B0 = self.uniform_sampling(self.min_phi, self.max_phi)
        theta_B0 = self.uniform_sampling(0, np.pi)
        
        return [Q1a, Q1b, Q2a, Q2b,
                alpha_A, beta1_A, alpha_B, beta1_B,
                phi_A1, theta_A1, psi_A1, phi_A2, theta_A2, psi_A2,
                phi_B1, theta_B1, psi_B1, phi_B2, theta_B2, psi_B2,
                phi_A0, theta_A0, phi_B0, theta_B0]
    
    def find_minimum_energy(self, N_ITERATIONS=30, tol_first_opt=1e-7, tol_second_opt=1e-10):
        """Find the optimal parameters that minimize the energy"""
        opt_params = None
        opt_energy = 1e10
        
        for i in range(N_ITERATIONS):
            initial_guess = self.random_initial_guess()
            
            # First optimization with Nelder-Mead (robust for many parameters)
            try:
                res = minimize(self.E_per_UC, x0=initial_guess, 
                              method='Nelder-Mead', 
                              options={'maxiter': 5000, 'xatol': tol_first_opt, 'fatol': tol_first_opt})
                
                if res.fun < opt_energy:
                    opt_params = res.x
                    opt_energy = res.fun
            except Exception as e:
                print(f"Warning: Optimization iteration {i} failed: {e}")
                continue
        
        if opt_params is None:
            raise RuntimeError("All optimization attempts failed")
        
        # Final refinement with L-BFGS-B
        try:
            res = minimize(self.E_per_UC, x0=opt_params, 
                          method='L-BFGS-B', 
                          options={'maxiter': 10000, 'ftol': tol_second_opt})
            if res.fun < opt_energy:
                opt_params = res.x
                opt_energy = res.fun
        except Exception as e:
            print(f"Warning: Final refinement failed: {e}")
        
        return opt_params, opt_energy
    
    def classify_phase(self, tol=1e-6):
        """Classify the magnetic ordering based on the optimal parameters"""
        p = self.parse_params(self.opt_params)
        
        Q1a, Q1b = p['Q1a'], p['Q1b']
        Q2a, Q2b = p['Q2a'], p['Q2b']
        beta1_A, beta2_A = p['beta1_A'], p['beta2_A']
        beta1_B, beta2_B = p['beta1_B'], p['beta2_B']
        
        # Check if effectively single-Q
        is_single_q = (beta2_A < tol and beta2_B < tol) or (beta1_A < tol and beta1_B < tol)
        
        if is_single_q:
            # Delegate to single-Q classification
            if beta2_A < tol and beta2_B < tol:
                Qa, Qb = Q1a, Q1b
            else:
                Qa, Qb = Q2a, Q2b
            
            if np.abs(Qa) < tol and np.abs(Qb) < tol:
                return "Uniform (FM/AFM)"
            elif (np.abs(Qa - 0.5) < tol and np.abs(Qb) < tol) or (np.abs(Qb - 0.5) < tol and np.abs(Qa) < tol):
                return "Single-Q: Zigzag/Stripy"
            elif np.abs(Qa - 1/3) < tol and np.abs(Qb - 1/3) < tol:
                return "Single-Q: 120° order"
            else:
                return "Single-Q: Incommensurate"
        
        # Check for commensurate double-Q
        Q1_commensurate = (np.abs(Q1a - 0.5) < tol or np.abs(Q1a) < tol) and \
                          (np.abs(Q1b - 0.5) < tol or np.abs(Q1b) < tol)
        Q2_commensurate = (np.abs(Q2a - 0.5) < tol or np.abs(Q2a) < tol) and \
                          (np.abs(Q2b - 0.5) < tol or np.abs(Q2b) < tol)
        
        if Q1_commensurate and Q2_commensurate:
            return "Double-Q: Commensurate"
        else:
            return "Double-Q: Incommensurate"
    
    def generate_spin_configuration(self):
        """Generate the spin configuration from the optimal parameters"""
        p = self.parse_params(self.opt_params)
        
        N = len(self.positions)
        spins = np.zeros((N, 3))
        
        for i in range(N):
            pos = self.positions[i]
            Q1_phase = np.dot(p['Q1_vec'], pos)
            Q2_phase = np.dot(p['Q2_vec'], pos)
            
            if i % 2 == 0:  # A sublattice
                spin = (p['alpha_A'] * p['ez_A0'] +
                       p['beta1_A'] * (p['exA1'] * np.cos(Q1_phase) + p['eyA1'] * np.sin(Q1_phase)) +
                       p['beta2_A'] * (p['exA2'] * np.cos(Q2_phase) + p['eyA2'] * np.sin(Q2_phase)))
            else:  # B sublattice
                spin = (p['alpha_B'] * p['ez_B0'] +
                       p['beta1_B'] * (p['exB1'] * np.cos(Q1_phase) + p['eyB1'] * np.sin(Q1_phase)) +
                       p['beta2_B'] * (p['exB2'] * np.cos(Q2_phase) + p['eyB2'] * np.sin(Q2_phase)))
            
            # Normalize
            norm = np.linalg.norm(spin)
            if norm > 1e-10:
                spins[i] = spin / norm
            else:
                spins[i] = np.array([0, 0, 1])
        
        return spins

    def calculate_magnetization(self, spins):
        """Calculate magnetization components of the spin configuration."""
        N = len(spins)
        
        total_magnetization = np.mean(spins, axis=0)
        total_magnitude = np.linalg.norm(total_magnetization)
        
        spins_A = spins[::2]
        spins_B = spins[1::2]
        
        mag_A = np.mean(spins_A, axis=0)
        mag_B = np.mean(spins_B, axis=0)
        
        staggered = mag_A - mag_B
        
        return {
            'total': total_magnetization,
            'total_magnitude': total_magnitude,
            'A': mag_A,
            'B': mag_B,
            'A_magnitude': np.linalg.norm(mag_A),
            'B_magnitude': np.linalg.norm(mag_B),
            'staggered': staggered,
            'staggered_magnitude': np.linalg.norm(staggered)
        }

    def real_space_energy(self, spins=None):
        """Energy per site from the real-space Hamiltonian (pair-counted once)."""
        if spins is None:
            spins = self.generate_spin_configuration()
        return calculate_energy_from_matrices(
            spins, self.NN, self.NN_bonds, self.NNN, self.NNNN, self.J1, self.J2_mat, self.J3_mat
        )
    
    def print_optimal_parameters(self):
        """Print the optimal parameters in a readable format"""
        p = self.parse_params(self.opt_params)
        
        print("=" * 60)
        print("Double-Q Ansatz Optimization Results")
        print("=" * 60)
        print(f"\nGround state energy: {self.opt_energy:.8f}")
        print(f"Magnetic order: {self.magnetic_order}")
        
        print(f"\nQ-vectors:")
        print(f"  Q1 = ({p['Q1a']:.6f}, {p['Q1b']:.6f}) in r.l.u.")
        print(f"  Q2 = ({p['Q2a']:.6f}, {p['Q2b']:.6f}) in r.l.u.")
        
        print(f"\nAmplitudes (sublattice A):")
        print(f"  α_A = {p['alpha_A']:.6f} (uniform)")
        print(f"  β1_A = {p['beta1_A']:.6f} (Q1 spiral)")
        print(f"  β2_A = {p['beta2_A']:.6f} (Q2 spiral)")
        print(f"  Check: α² + β1² + β2² = {p['alpha_A']**2 + p['beta1_A']**2 + p['beta2_A']**2:.6f}")
        
        print(f"\nAmplitudes (sublattice B):")
        print(f"  α_B = {p['alpha_B']:.6f} (uniform)")
        print(f"  β1_B = {p['beta1_B']:.6f} (Q1 spiral)")
        print(f"  β2_B = {p['beta2_B']:.6f} (Q2 spiral)")
        print(f"  Check: α² + β1² + β2² = {p['alpha_B']**2 + p['beta1_B']**2 + p['beta2_B']**2:.6f}")
        
        print(f"\nUniform magnetization directions:")
        print(f"  n_A0 = [{p['ez_A0'][0]:.4f}, {p['ez_A0'][1]:.4f}, {p['ez_A0'][2]:.4f}]")
        print(f"  n_B0 = [{p['ez_B0'][0]:.4f}, {p['ez_B0'][1]:.4f}, {p['ez_B0'][2]:.4f}]")
        
        print(f"\nBCAO Parameters: J1xy={self.J[0]}, J1z={self.J[1]}, D={self.J[2]}, " +
              f"E={self.J[3]}, F={self.J[4]}, G={self.J[5]}, J3xy={self.J[6]}, J3z={self.J[7]}")


def compare_single_and_double_q(L=8, J=[-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85]):
    """Compare single-Q and double-Q ansatz results"""
    from single_q_BCAO import SingleQ
    
    print("Running single-Q optimization...")
    single_q = SingleQ(L, J)
    
    print("\nRunning double-Q optimization...")
    double_q = DoubleQ(L, J)
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\nSingle-Q energy: {single_q.opt_energy:.8f}")
    print(f"Double-Q energy: {double_q.opt_energy:.8f}")
    print(f"Energy difference: {double_q.opt_energy - single_q.opt_energy:.8e}")
    
    if double_q.opt_energy < single_q.opt_energy - 1e-8:
        print("\n*** Double-Q finds lower energy! ***")
    elif abs(double_q.opt_energy - single_q.opt_energy) < 1e-8:
        print("\n*** Energies are equivalent (single-Q is likely the ground state) ***")
    else:
        print("\n*** Single-Q finds lower energy (double-Q may be in local minimum) ***")
    
    return single_q, double_q


if __name__ == "__main__":
    # Size of lattice (L x L unit cells)
    L = 8
    
    # BCAO parameters matching the C++ convention: [J1xy, J1z, D, E, F, G, J3xy, J3z]
    J = [-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85]
    
    # Create double-Q model
    print("Initializing Double-Q model...")
    print(f"Lattice size: {L}x{L} unit cells")
    print(f"BCAO Parameters: J1xy={J[0]}, J1z={J[1]}, D={J[2]}, E={J[3]}, F={J[4]}, G={J[5]}, J3xy={J[6]}, J3z={J[7]}")
    print()
    
    model = DoubleQ(L, J)
    model.print_optimal_parameters()
    
    # Generate and analyze the spin configuration
    spins = model.generate_spin_configuration()
    magnetization = model.calculate_magnetization(spins)
    
    print("\nMagnetization Analysis:")
    print("=" * 50)
    print(f"Total magnetization: [{magnetization['total'][0]:.6f}, {magnetization['total'][1]:.6f}, {magnetization['total'][2]:.6f}]")
    print(f"Total magnetization magnitude: {magnetization['total_magnitude']:.6f}")
    print(f"Staggered magnetization magnitude: {magnetization['staggered_magnitude']:.6f}")
    
    # Optionally compare with single-Q
    print("\n" + "=" * 60)
    print("Comparing with Single-Q Ansatz...")
    print("=" * 60)
    single_q, double_q = compare_single_and_double_q(L, J)
    
    # Visualize the spin configuration
    visualize_spins(model.positions, spins, L)
