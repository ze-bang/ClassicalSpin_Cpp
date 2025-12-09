import numpy as np
from scipy.optimize import minimize
from luttinger_tisza import create_honeycomb_lattice, construct_interaction_matrices, get_bond_vectors, visualize_spins

# filepath: /home/pc_linux/ClassicalSpin_Cpp/util/double_q_BCAO_orthogonal.py

import matplotlib.pyplot as plt

class DoubleQ_Orthogonal:
    """
    Class to perform orthogonal double-Q ansatz simulation on a BCAO honeycomb lattice
    using momentum-space energy evaluation (Luttinger-Tisza approach).
    
    Uses orthogonal double-Q ansatz where x and y components spiral independently:
        S_X(r) = alpha_X * e_z^X 
               + beta_X * [e_x^X * cos(Q^(1)·r) + e_x^X * sin(Q^(1)·r)]
               + gamma_X * [e_y^X * cos(Q^(2)·r) + e_y^X * sin(Q^(2)·r)]
    
    where Q^(1) = Q1*b1 and Q^(2) = Q2*b2.
    
    The x-component modulates with Q^(1) along b1.
    The y-component modulates with Q^(2) along b2.
    The z-component (alpha_X * e_z^X) is uniform.
    
    Parameterization ensures <|S|^2> = 1 on average:
        alpha_X = cos(theta_X)
        beta_X = sin(theta_X) * cos(phi_X)
        gamma_X = sin(theta_X) * sin(phi_X)
    
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
    
    def __init__(self, L=4, J=[-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85], B_field=np.array([0, 0, 0])):
        """
        Initialize the orthogonal double-Q model with momentum-space energy evaluation
        
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
        Calculate the energy per unit cell for the orthogonal double-Q ansatz.
        
        The ansatz is:
            S_X(r) = alpha_X * e_z^X 
                   + beta_X * [e_x^X * cos(Q1·b1·r) + e_x^X * sin(Q1·b1·r)]
                   + gamma_X * [e_y^X * cos(Q2·b2·r) + e_y^X * sin(Q2·b2·r)]
        
        The Fourier components are:
        - q=0: uniform z-component (alpha)
        - q=±Q1*b1: x-component modulation (beta)
        - q=±Q2*b2: y-component modulation (gamma)
        
        The amplitudes are parameterized as:
            alpha_X = cos(theta_amp_X)
            beta_X = sin(theta_amp_X) * cos(phi_amp_X)    [couples to Q1]
            gamma_X = sin(theta_amp_X) * sin(phi_amp_X)   [couples to Q2]
        """
        Q1, Q2, theta_amp_A, phi_amp_A, theta_amp_B, phi_amp_B, phi_A, theta_A, psi_A, phi_B, theta_B, psi_B = params
        
        # The two wavevectors: Q^(1) = Q1*b1, Q^(2) = Q2*b2
        Qvec1 = Q1 * self.b1
        Qvec2 = Q2 * self.b2
        
        # Get rotated basis vectors
        exA, eyA, ezA = self.RotatedBasis(phi_A, theta_A, psi_A)
        exB, eyB, ezB = self.RotatedBasis(phi_B, theta_B, psi_B)
        
        # Compute amplitudes using spherical parameterization
        alpha_A = np.cos(theta_amp_A)
        beta_A = np.sin(theta_amp_A) * np.cos(phi_amp_A)   # x-component amplitude (couples to Q1)
        gamma_A = np.sin(theta_amp_A) * np.sin(phi_amp_A)  # y-component amplitude (couples to Q2)
        
        alpha_B = np.cos(theta_amp_B)
        beta_B = np.sin(theta_amp_B) * np.cos(phi_amp_B)
        gamma_B = np.sin(theta_amp_B) * np.sin(phi_amp_B)
        
        # Uniform component contribution (q=0)
        E_q0 = alpha_A * alpha_B * (ezA.dot(self.HAB(np.array([0, 0]))).dot(ezB) + 
                                    ezB.dot(self.HBA(np.array([0, 0]))).dot(ezA))
        E_q0 += alpha_A**2 * ezA.dot(self.HAA(np.array([0, 0]))).dot(ezA)
        E_q0 += alpha_B**2 * ezB.dot(self.HBB(np.array([0, 0]))).dot(ezB)
        
        # Q1 component: x-direction modulation
        # For S_A = beta_A * [ex_A * cos(Q1·r) + ex_A * sin(Q1·r)]
        # In Fourier space: S_A(Q1) = beta_A * ex_A, S_A(-Q1) = beta_A * ex_A (real spiral)
        # Energy contribution: beta_A * beta_B * ex_A · H_AB(Q1) · ex_B + c.c.
        HAB_Q1 = self.HAB(Qvec1)
        HBA_Q1 = self.HBA(Qvec1)
        HAA_Q1 = self.HAA(Qvec1)
        HBB_Q1 = self.HBB(Qvec1)
        
        E_Q1 = beta_A * beta_B * (
            exA.dot(HAB_Q1).dot(exB) + exB.dot(HBA_Q1).dot(exA)
        )
        E_Q1 += beta_A**2 * exA.dot(HAA_Q1).dot(exA)
        E_Q1 += beta_B**2 * exB.dot(HBB_Q1).dot(exB)
        
        # Q2 component: y-direction modulation
        # For S_A = gamma_A * [ey_A * cos(Q2·r) + ey_A * sin(Q2·r)]
        HAB_Q2 = self.HAB(Qvec2)
        HBA_Q2 = self.HBA(Qvec2)
        HAA_Q2 = self.HAA(Qvec2)
        HBB_Q2 = self.HBB(Qvec2)
        
        E_Q2 = gamma_A * gamma_B * (
            eyA.dot(HAB_Q2).dot(eyB) + eyB.dot(HBA_Q2).dot(eyA)
        )
        E_Q2 += gamma_A**2 * eyA.dot(HAA_Q2).dot(eyA)
        E_Q2 += gamma_B**2 * eyB.dot(HBB_Q2).dot(eyB)
        
        # Zeeman energy contribution
        E_zeeman = self.B_field.dot(ezA * alpha_A + ezB * alpha_B)
        
        return np.real(E_q0 / 4 + E_Q1 / 4 + E_Q2 / 4 + E_zeeman / 2)
    
    def find_minimum_energy(self, N_ITERATIONS=20, tol_first_opt=10**-8, tol_second_opt=10**-10):
        """Find the optimal parameters that minimize the energy"""
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
        Q1, Q2, theta_amp_A, phi_amp_A, theta_amp_B, phi_amp_B, phiA, thetaA, psiA, phiB, thetaB, psiB = self.opt_params
        
        # Compute amplitudes
        alphaA = np.cos(theta_amp_A)
        alphaB = np.cos(theta_amp_B)
        beta_A = np.sin(theta_amp_A) * np.cos(phi_amp_A)
        gamma_A = np.sin(theta_amp_A) * np.sin(phi_amp_A)
        beta_B = np.sin(theta_amp_B) * np.cos(phi_amp_B)
        gamma_B = np.sin(theta_amp_B) * np.sin(phi_amp_B)
        
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
        
        # Check if only one Q is active
        if np.abs(Q1) < tol and np.abs(Q2) > tol:
            # Only Q2 active (y-component modulation)
            if np.abs(Q2 - 0.5) < tol:
                return "Single-Q: y-modulated at M-point (Q||b2)"
            elif np.abs(Q2 - 1/3) < tol:
                return "Single-Q: y-modulated at K-point (Q||b2)"
            else:
                return f"Single-Q: y-modulated incommensurate (Q||b2, Q2={Q2:.4f})"
        
        if np.abs(Q2) < tol and np.abs(Q1) > tol:
            # Only Q1 active (x-component modulation)
            if np.abs(Q1 - 0.5) < tol:
                return "Single-Q: x-modulated at M-point (Q||b1)"
            elif np.abs(Q1 - 1/3) < tol:
                return "Single-Q: x-modulated at K-point (Q||b1)"
            else:
                return f"Single-Q: x-modulated incommensurate (Q||b1, Q1={Q1:.4f})"
        
        # Both Q's are active (true orthogonal double-Q order)
        if np.abs(Q1 - 0.5) < tol and np.abs(Q2 - 0.5) < tol:
            return "Orthogonal Double-Q: M-point (x↔Q1||b1, y↔Q2||b2)"
        elif np.abs(Q1 - 1/3) < tol and np.abs(Q2 - 1/3) < tol:
            return "Orthogonal Double-Q: K-point (x↔Q1||b1, y↔Q2||b2)"
        else:
            return f"Orthogonal Double-Q (x↔Q1={Q1:.4f}*b1, y↔Q2={Q2:.4f}*b2)"
    
    def generate_spin_configuration(self):
        """Generate the spin configuration from the optimal parameters"""
        Q1, Q2, theta_amp_A, phi_amp_A, theta_amp_B, phi_amp_B, phiA, thetaA, psiA, phiB, thetaB, psiB = self.opt_params
        Qvec1 = Q1 * self.b1
        Qvec2 = Q2 * self.b2
        
        N = len(self.positions)
        spins = np.zeros((N, 3))
        
        exA, eyA, ezA = self.RotatedBasis(phiA, thetaA, psiA)
        exB, eyB, ezB = self.RotatedBasis(phiB, thetaB, psiB)
        
        # Orthogonal amplitudes
        alphaA = np.cos(theta_amp_A)
        beta_A = np.sin(theta_amp_A) * np.cos(phi_amp_A)   # x-component
        gamma_A = np.sin(theta_amp_A) * np.sin(phi_amp_A)  # y-component
        
        alphaB = np.cos(theta_amp_B)
        beta_B = np.sin(theta_amp_B) * np.cos(phi_amp_B)
        gamma_B = np.sin(theta_amp_B) * np.sin(phi_amp_B)
        
        for i in range(N):
            pos = self.positions[i]
            phase1 = Qvec1.dot(pos)  # Q1 couples to x
            phase2 = Qvec2.dot(pos)  # Q2 couples to y
            
            if i % 2 == 0:  # A sublattice
                spin = (
                    ezA * alphaA +
                    beta_A * exA * np.cos(phase1) +     # x-component with Q1
                    gamma_A * eyA * np.cos(phase2)      # y-component with Q2
                )
            else:  # B sublattice
                spin = (
                    ezB * alphaB +
                    beta_B * exB * np.cos(phase1) +
                    gamma_B * eyB * np.cos(phase2)
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


def interactive_graph_mode(L=12, J=[-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]):
    """
    Interactive mode to explore spin configurations with arbitrary Q1 and Q2 values.
    Uses sliders to adjust Q1 and Q2 and visualize the resulting spin configuration.
    """
    from matplotlib.widgets import Slider, Button
    
    # Create a model instance (we'll override its parameters)
    model = DoubleQ_Orthogonal(L, J)
    
    # Initial Q values
    Q1_init = 1/3
    Q2_init = 1/3
    
    # Set initial parameters with default angles
    model.opt_params = [
        Q1_init, Q2_init,
        np.pi/4, np.pi/4,  # theta_amp_A, phi_amp_A
        np.pi/4, np.pi/4,  # theta_amp_B, phi_amp_B
        0, 0, 0,           # phi_A, theta_A, psi_A
        0, 0, 0            # phi_B, theta_B, psi_B
    ]
    
    # Generate initial spin configuration
    spins = model.generate_spin_configuration()
    
    # Create the figure and 3D axis
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.1, bottom=0.35)
    
    # Plot initial configuration
    positions = model.positions
    # Create z=0 for all positions (honeycomb lattice is 2D)
    z_positions = np.zeros(len(positions))
    quiver = ax.quiver(positions[:, 0], positions[:, 1], z_positions,
                       spins[:, 0], spins[:, 1], spins[:, 2],
                       length=0.5, normalize=True, arrow_length_ratio=0.3)
    
    # Set up the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Orthogonal Double-Q Spin Configuration\nQ1={Q1_init:.3f}, Q2={Q2_init:.3f}')
    
    # Create slider axes
    ax_Q1 = plt.axes([0.1, 0.25, 0.8, 0.03])
    ax_Q2 = plt.axes([0.1, 0.20, 0.8, 0.03])
    ax_theta_A = plt.axes([0.1, 0.15, 0.8, 0.03])
    ax_phi_A = plt.axes([0.1, 0.10, 0.8, 0.03])
    ax_theta_B = plt.axes([0.1, 0.05, 0.8, 0.03])
    ax_phi_B = plt.axes([0.1, 0.00, 0.8, 0.03])
    
    # Create sliders
    slider_Q1 = Slider(ax_Q1, 'Q1', 0.0, 0.5, valinit=Q1_init, valstep=0.01)
    slider_Q2 = Slider(ax_Q2, 'Q2', 0.0, 0.5, valinit=Q2_init, valstep=0.01)
    slider_theta_A = Slider(ax_theta_A, 'θ_amp_A', 0.0, np.pi/2, valinit=np.pi/4, valstep=0.01)
    slider_phi_A = Slider(ax_phi_A, 'φ_amp_A', 0.0, np.pi/2, valinit=np.pi/4, valstep=0.01)
    slider_theta_B = Slider(ax_theta_B, 'θ_amp_B', 0.0, np.pi/2, valinit=np.pi/4, valstep=0.01)
    slider_phi_B = Slider(ax_phi_B, 'φ_amp_B', 0.0, np.pi/2, valinit=np.pi/4, valstep=0.01)
    
    def update(val):
        """Update the spin configuration when sliders change"""
        Q1 = slider_Q1.val
        Q2 = slider_Q2.val
        theta_A = slider_theta_A.val
        phi_A = slider_phi_A.val
        theta_B = slider_theta_B.val
        phi_B = slider_phi_B.val
        
        # Update model parameters
        model.opt_params = [
            Q1, Q2,
            theta_A, phi_A,
            theta_B, phi_B,
            0, 0, 0,  # Keep basis vectors along xyz
            0, 0, 0
        ]
        
        # Generate new spin configuration
        spins = model.generate_spin_configuration()
        
        # Calculate energy and magnetization
        energy = model.E_per_UC(model.opt_params)
        mag = model.calculate_magnetization(spins)
        
        # Update the quiver plot
        ax.clear()
        z_positions = np.zeros(len(positions))
        ax.quiver(positions[:, 0], positions[:, 1], z_positions,
                 spins[:, 0], spins[:, 1], spins[:, 2],
                 length=0.5, normalize=True, arrow_length_ratio=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Compute amplitudes for display
        alpha_A = np.cos(theta_A)
        beta_A = np.sin(theta_A) * np.cos(phi_A)
        gamma_A = np.sin(theta_A) * np.sin(phi_A)
        alpha_B = np.cos(theta_B)
        beta_B = np.sin(theta_B) * np.cos(phi_B)
        gamma_B = np.sin(theta_B) * np.sin(phi_B)
        
        title = f'Orthogonal Double-Q: Q1={Q1:.3f}, Q2={Q2:.3f}\n'
        title += f'A: α={alpha_A:.2f}, β={beta_A:.2f}, γ={gamma_A:.2f} | '
        title += f'B: α={alpha_B:.2f}, β={beta_B:.2f}, γ={gamma_B:.2f}\n'
        title += f'E/UC={energy:.4f}, |M|={mag["total_magnitude"]:.4f}'
        ax.set_title(title)
        
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    slider_Q1.on_changed(update)
    slider_Q2.on_changed(update)
    slider_theta_A.on_changed(update)
    slider_phi_A.on_changed(update)
    slider_theta_B.on_changed(update)
    slider_phi_B.on_changed(update)
    
    plt.show()


if __name__ == "__main__":
    import sys
    
    # Size of lattice (L x L unit cells)
    L = 12
    
    # BCAO parameters matching the C++ convention: [J1xy, J1z, D, E, F, G, J3xy, J3z]
    # J = [-7.6, -1.2, 0.1, -0.1, 0, 0, 2.5, -0.85]
    J = [-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]
    # J = [-6.646, -2.084, 0.675, 1.33, -1.516, 0.21, 1.697, 0.039]
    
    # Check for graph mode
    if len(sys.argv) > 1 and sys.argv[1] == '--graph':
        print("Starting interactive graph mode...")
        print("Use the sliders to explore different Q1 and Q2 values")
        print("θ_amp and φ_amp control the amplitude distribution:")
        print("  α (uniform z) = cos(θ)")
        print("  β (x-modulation with Q1) = sin(θ)cos(φ)")
        print("  γ (y-modulation with Q2) = sin(θ)sin(φ)")
        interactive_graph_mode(L, J)
    else:
        # Original optimization mode
        # Create orthogonal double-Q model using momentum-space formulation
        model = DoubleQ_Orthogonal(L, J)
        
        # Extract optimal parameters
        Q1, Q2, theta_amp_A, phi_amp_A, theta_amp_B, phi_amp_B = model.opt_params[0:6]
        
        # Compute amplitudes
        alpha_A = np.cos(theta_amp_A)
        beta_A = np.sin(theta_amp_A) * np.cos(phi_amp_A)
        gamma_A = np.sin(theta_amp_A) * np.sin(phi_amp_A)
        alpha_B = np.cos(theta_amp_B)
        beta_B = np.sin(theta_amp_B) * np.cos(phi_amp_B)
        gamma_B = np.sin(theta_amp_B) * np.sin(phi_amp_B)
        
        # Print results
        print(f"Optimal Q-vectors:")
        print(f"  Q^(1) = {Q1:.4f}*b1  [couples to x-component]")
        print(f"  Q^(2) = {Q2:.4f}*b2  [couples to y-component]")
        print(f"Ground state energy per unit cell: {model.opt_energy:.6f}")
        print(f"Magnetic order: {model.magnetic_order}")
        print(f"\nAmplitude distribution:")
        print(f"  Sublattice A: alpha(z)={alpha_A:.4f}, beta(x)={beta_A:.4f}, gamma(y)={gamma_A:.4f}")
        print(f"  Sublattice B: alpha(z)={alpha_B:.4f}, beta(x)={beta_B:.4f}, gamma(y)={gamma_B:.4f}")
        print(f"  Normalization check A: {alpha_A**2 + beta_A**2 + gamma_A**2:.6f}")
        print(f"  Normalization check B: {alpha_B**2 + beta_B**2 + gamma_B**2:.6f}")
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
        
        print("\nRun with '--graph' flag to enter interactive exploration mode")
        
        # Visualize the spin configuration
        visualize_spins(model.positions, spins, L)
