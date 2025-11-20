import numpy as np
from scipy.optimize import minimize, differential_evolution
from luttinger_tisza import create_honeycomb_lattice, get_bond_vectors, visualize_spins
import matplotlib.pyplot as plt

class MultiQ:
    """
    Class to perform multi-Q ansatz simulation on a BCAO honeycomb lattice
    to determine ground state spin configuration and energy.
    
    The multi-Q ansatz generalizes the single-Q approach by allowing the spin
    configuration to be a superposition of multiple wave vectors:
    
    S_i = sum_q [ alpha_q * n_q * cos(q·r_i + phi_q) ]
    
    This can capture more complex magnetic orders including:
    - Triple-Q states (e.g., skyrmion lattices)
    - Double-Q states
    - Non-coplanar spin structures
    
    Uses the BCAO parameter convention: J = [J1xy, J1z, D, E, F, G, J3xy, J3z]
    """
    
    def __init__(self, L=4, num_Q=2, J=[-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157], 
                 B_field=np.array([0, 0, 0])):
        """
        Initialize the multi-Q model
        
        Args:
            L: Size of the lattice (L x L unit cells)
            num_Q: Number of Q-vectors in the ansatz
            J: BCAO exchange coupling parameters [J1xy, J1z, D, E, F, G, J3xy, J3z]
            B_field: External magnetic field
        """
        self.L = L
        self.num_Q = num_Q
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
        
        # Set up parameter bounds for optimization
        self.setup_parameter_bounds()
        
        # Find optimal spin configuration
        self.opt_params, self.opt_energy = self.find_minimum_energy()
        
        # Determine spin configuration type
        self.magnetic_order = self.classify_phase()
    
    def setup_parameter_bounds(self):
        """Setup parameter bounds for the multi-Q optimization"""
        eta_small = 10**-9
        bounds = []
        
        # For each Q-vector: Q1, Q2 (2 parameters)
        for i in range(self.num_Q):
            bounds.extend([
                (0, 0.5-eta_small),  # Q1
                (0, 0.5)             # Q2
            ])
        
        # For each Q-vector on each sublattice: amplitude, phi, theta, psi (4 parameters × num_Q × 2 sublattices)
        for i in range(self.num_Q):
            for sublattice in ['A', 'B']:
                bounds.extend([
                    (0, 1.0),                    # amplitude
                    (0, (2-eta_small)*np.pi),   # phi
                    (0, (2-eta_small)*np.pi),   # theta
                    (0, (2-eta_small)*np.pi)    # psi
                ])
        
        self.parameter_bounds = bounds
        self.num_params = len(bounds)
    
    @staticmethod
    def RotatedBasis(phi, theta, psi):
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
        """Parse the parameter array into Q-vectors and spin parameters"""
        Q_vecs = []
        spin_params_A = []
        spin_params_B = []
        
        idx = 0
        # Extract Q-vectors
        for i in range(self.num_Q):
            Q1 = params[idx]
            Q2 = params[idx + 1]
            Q_vecs.append(Q1 * self.b1 + Q2 * self.b2)
            idx += 2
        
        # Extract spin parameters for each Q on each sublattice
        for i in range(self.num_Q):
            # A sublattice
            amplitude_A = params[idx]
            phi_A = params[idx + 1]
            theta_A = params[idx + 2]
            psi_A = params[idx + 3]
            spin_params_A.append((amplitude_A, phi_A, theta_A, psi_A))
            idx += 4
            
            # B sublattice
            amplitude_B = params[idx]
            phi_B = params[idx + 1]
            theta_B = params[idx + 2]
            psi_B = params[idx + 3]
            spin_params_B.append((amplitude_B, phi_B, theta_B, psi_B))
            idx += 4
        
        return Q_vecs, spin_params_A, spin_params_B
    
    def E_per_UC(self, params):
        """
        Calculate the energy per unit cell for the multi-Q ansatz
        
        Following the same structure as single-Q but generalized to multiple Q-vectors.
        The spin ansatz is: S = Σ_q amplitude_q * (e_x cos(q·r) + e_y sin(q·r))
        
        Energy includes:
        1. Exchange interactions between spins (computed in Fourier space)
        2. Zeeman energy from external magnetic field
        3. Normalization constraint (sum of amplitude^2 should equal 1)
        """
        Q_vecs, spin_params_A, spin_params_B = self.parse_params(params)
        
        # Initialize energy
        E_total = 0.0
        
        # Construct spin basis vectors for each Q-vector
        basis_A = []
        basis_B = []
        
        for i in range(self.num_Q):
            amp_A, phi_A, theta_A, psi_A = spin_params_A[i]
            amp_B, phi_B, theta_B, psi_B = spin_params_B[i]
            
            ex_A, ey_A, ez_A = self.RotatedBasis(phi_A, theta_A, psi_A)
            ex_B, ey_B, ez_B = self.RotatedBasis(phi_B, theta_B, psi_B)
            
            basis_A.append((amp_A, ex_A, ey_A, ez_A))
            basis_B.append((amp_B, ex_B, ey_B, ez_B))
        
        # Compute energy contributions from all Q-vector pairs
        # Energy = (1/2) Σ_ij J_ij S_i · S_j
        # In Fourier space: (1/2N) Σ_q S(-q) · J(q) · S(q)
        # For multi-Q: S(q) = Σ_k amplitude_k * basis_k * δ(q - Q_k) + c.c.
        
        # First compute contributions from same-Q terms (i==j)
        for i in range(self.num_Q):
            amp_A_i, ex_A_i, ey_A_i, ez_A_i = basis_A[i]
            amp_B_i, ex_B_i, ey_B_i, ez_B_i = basis_B[i]
            Q_i = Q_vecs[i]
            
            # Fourier components for +Q and -Q
            spin_A_plus = ex_A_i + 1j * ey_A_i
            spin_A_minus = ex_A_i - 1j * ey_A_i
            spin_B_plus = ex_B_i + 1j * ey_B_i
            spin_B_minus = ex_B_i - 1j * ey_B_i
            
            # A-B interactions: S_A(Q) · J_AB(Q) · S_B(-Q) + S_A(-Q) · J_AB(-Q) · S_B(Q)
            HAB_Q = self.HAB(Q_i)
            HAB_mQ = self.HAB(-Q_i)
            
            E_AB_same = amp_A_i * amp_B_i * (
                spin_A_minus.dot(HAB_Q).dot(spin_B_plus) +
                spin_A_plus.dot(HAB_mQ).dot(spin_B_minus)
            ) / 4
            
            # B-A interactions (should equal A-B by symmetry)
            HBA_Q = self.HBA(Q_i)
            HBA_mQ = self.HBA(-Q_i)
            
            E_BA_same = amp_B_i * amp_A_i * (
                spin_B_minus.dot(HBA_Q).dot(spin_A_plus) +
                spin_B_plus.dot(HBA_mQ).dot(spin_A_minus)
            ) / 4
            
            # A-A interactions
            HAA_Q = self.HAA(Q_i)
            HAA_mQ = self.HAA(-Q_i)
            
            E_AA_same = amp_A_i**2 * (
                spin_A_minus.dot(HAA_Q).dot(spin_A_plus) +
                spin_A_plus.dot(HAA_mQ).dot(spin_A_minus)
            ) / 4
            
            # B-B interactions
            HBB_Q = self.HBB(Q_i)
            HBB_mQ = self.HBB(-Q_i)
            
            E_BB_same = amp_B_i**2 * (
                spin_B_minus.dot(HBB_Q).dot(spin_B_plus) +
                spin_B_plus.dot(HBB_mQ).dot(spin_B_minus)
            ) / 4
            
            E_total += np.real(E_AB_same + E_BA_same + E_AA_same + E_BB_same)
        
        # Cross-terms between different Q-vectors (i ≠ j)
        # These arise from interactions at Q_i ± Q_j
        for i in range(self.num_Q):
            amp_A_i, ex_A_i, ey_A_i, ez_A_i = basis_A[i]
            amp_B_i, ex_B_i, ey_B_i, ez_B_i = basis_B[i]
            Q_i = Q_vecs[i]
            
            for j in range(i+1, self.num_Q):  # Only count each pair once
                amp_A_j, ex_A_j, ey_A_j, ez_A_j = basis_A[j]
                amp_B_j, ex_B_j, ey_B_j, ez_B_j = basis_B[j]
                Q_j = Q_vecs[j]
                
                spin_A_i_plus = ex_A_i + 1j * ey_A_i
                spin_A_i_minus = ex_A_i - 1j * ey_A_i
                spin_A_j_plus = ex_A_j + 1j * ey_A_j
                spin_A_j_minus = ex_A_j - 1j * ey_A_j
                
                spin_B_i_plus = ex_B_i + 1j * ey_B_i
                spin_B_i_minus = ex_B_i - 1j * ey_B_i
                spin_B_j_plus = ex_B_j + 1j * ey_B_j
                spin_B_j_minus = ex_B_j - 1j * ey_B_j
                
                # Interactions at Q_i - Q_j
                Q_diff = Q_i - Q_j
                HAB_diff = self.HAB(Q_diff)
                HBA_diff = self.HBA(Q_diff)
                HAA_diff = self.HAA(Q_diff)
                HBB_diff = self.HBB(Q_diff)
                
                # And at Q_j - Q_i = -(Q_i - Q_j)
                Q_diff_neg = Q_j - Q_i
                HAB_diff_neg = self.HAB(Q_diff_neg)
                HBA_diff_neg = self.HBA(Q_diff_neg)
                HAA_diff_neg = self.HAA(Q_diff_neg)
                HBB_diff_neg = self.HBB(Q_diff_neg)
                
                # Cross terms (factor of 2 because we only loop over i < j)
                E_AB_cross = 2 * amp_A_i * amp_B_j * (
                    spin_A_i_minus.dot(HAB_diff).dot(spin_B_j_plus) +
                    spin_A_i_plus.dot(HAB_diff_neg).dot(spin_B_j_minus)
                ) / 4
                
                E_BA_cross = 2 * amp_B_i * amp_A_j * (
                    spin_B_i_minus.dot(HBA_diff).dot(spin_A_j_plus) +
                    spin_B_i_plus.dot(HBA_diff_neg).dot(spin_A_j_minus)
                ) / 4
                
                E_AA_cross = 2 * amp_A_i * amp_A_j * (
                    spin_A_i_minus.dot(HAA_diff).dot(spin_A_j_plus) +
                    spin_A_i_plus.dot(HAA_diff_neg).dot(spin_A_j_minus)
                ) / 4
                
                E_BB_cross = 2 * amp_B_i * amp_B_j * (
                    spin_B_i_minus.dot(HBB_diff).dot(spin_B_j_plus) +
                    spin_B_i_plus.dot(HBB_diff_neg).dot(spin_B_j_minus)
                ) / 4
                
                E_total += np.real(E_AB_cross + E_BA_cross + E_AA_cross + E_BB_cross)
        
        # Add Zeeman energy contribution
        # The Zeeman term couples to the spatially-averaged magnetization
        # For oscillating components, the average is zero; only Q=0 contributes
        mag_A = np.zeros(3)
        mag_B = np.zeros(3)
        
        for i in range(self.num_Q):
            amp_A, ex_A, ey_A, ez_A = basis_A[i]
            amp_B, ex_B, ey_B, ez_B = basis_B[i]
            
            # Only Q=0 components contribute (uniform magnetization)
            if np.linalg.norm(Q_vecs[i]) < 1e-10:
                # For Q=0, the spin is just amplitude * ez (no oscillation)
                mag_A += amp_A * ez_A
                mag_B += amp_B * ez_B
        
        # Zeeman energy: -B · M, with factor of 1/2 for two sublattices
        E_zeeman = -self.B_field.dot(mag_A + mag_B) / 2
        
        # Add soft constraint to enforce spin normalization
        # The constraint is: sum_i amplitude_i^2 = 1 for each sublattice
        norm_A = sum(amp**2 for amp, _, _, _ in basis_A)
        norm_B = sum(amp**2 for amp, _, _, _ in basis_B)
        
        penalty_weight = 100.0  # Large penalty to enforce constraint
        E_penalty = penalty_weight * ((norm_A - 1.0)**2 + (norm_B - 1.0)**2)
        
        # Final energy: divide exchange part by 4 (following single-Q convention)
        # and divide Zeeman by 2, then add penalty
        return E_total / 4 + E_zeeman / 2 + E_penalty
    
    def find_minimum_energy(self, N_ITERATIONS=50, use_global=True):
        """
        Find the optimal parameters that minimize the energy
        
        Args:
            N_ITERATIONS: Number of random initializations for local optimization
            use_global: If True, use differential evolution for global optimization first
        """
        if use_global:
            # Use differential evolution for global optimization
            print(f"Running global optimization with differential evolution...")
            result = differential_evolution(
                self.E_per_UC,
                bounds=self.parameter_bounds,
                maxiter=500,
                popsize=15,
                tol=1e-8,
                seed=42,
                workers=1,
                updating='deferred',
                polish=True
            )
            opt_params = result.x
            opt_energy = result.fun
            print(f"Global optimization complete. Energy: {opt_energy:.6f}")
        else:
            # Use multiple random initializations with local optimization
            opt_params = None
            opt_energy = 10**10
            
            for i in range(N_ITERATIONS):
                if i % 10 == 0:
                    print(f"Iteration {i}/{N_ITERATIONS}...")
                
                # Random initial guess
                initial_guess = []
                for (lb, ub) in self.parameter_bounds:
                    initial_guess.append(np.random.uniform(lb, ub))
                
                # Minimize at that point
                result = minimize(
                    self.E_per_UC,
                    x0=initial_guess,
                    bounds=self.parameter_bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 1000}
                )
                
                if result.fun < opt_energy:
                    opt_params = result.x
                    opt_energy = result.fun
                    print(f"  New best energy: {opt_energy:.6f}")
            
            # Final refinement
            print("Performing final refinement...")
            result = minimize(
                self.E_per_UC,
                x0=opt_params,
                bounds=self.parameter_bounds,
                method='L-BFGS-B',
                options={'ftol': 1e-10, 'maxiter': 2000}
            )
            opt_params = result.x
            opt_energy = result.fun
        
        return opt_params, opt_energy
    
    def classify_phase(self, tol=1e-6):
        """Classify the magnetic ordering based on the optimal parameters"""
        Q_vecs, spin_params_A, spin_params_B = self.parse_params(self.opt_params)
        
        # Determine number of significant Q-vectors
        significant_Q = []
        for i, Q in enumerate(Q_vecs):
            amp_A = spin_params_A[i][0]
            amp_B = spin_params_B[i][0]
            if amp_A > tol or amp_B > tol:
                Q_reduced = np.array([Q.dot(self.b1) / (2*np.pi), Q.dot(self.b2) / (2*np.pi)])
                significant_Q.append(Q_reduced)
        
        num_sig_Q = len(significant_Q)
        
        if num_sig_Q == 0:
            return "Disordered/Paramagnetic"
        elif num_sig_Q == 1:
            Q = significant_Q[0]
            if np.linalg.norm(Q) < tol:
                return "Ferromagnetic/Uniform"
            elif np.abs(np.abs(Q[0]) - 0.5) < tol or np.abs(np.abs(Q[1]) - 0.5) < tol:
                return "Zigzag/Stripy (Single-Q)"
            elif np.abs(Q[0] - 1/3) < tol and np.abs(Q[1] - 1/3) < tol:
                return "120° order (Single-Q)"
            else:
                return f"Incommensurate (Single-Q at Q={Q})"
        elif num_sig_Q == 2:
            return f"Double-Q state with {num_sig_Q} Q-vectors"
        elif num_sig_Q == 3:
            # Check if it's a triple-Q state (possibly skyrmion lattice)
            return f"Triple-Q state (possible skyrmion lattice)"
        else:
            return f"Multi-Q state with {num_sig_Q} Q-vectors"
    
    def generate_spin_configuration(self):
        """Generate the spin configuration from the optimal parameters"""
        Q_vecs, spin_params_A, spin_params_B = self.parse_params(self.opt_params)
        
        N = len(self.positions)
        spins = np.zeros((N, 3))
        
        for i in range(N):
            pos = self.positions[i]
            spin = np.zeros(3)
            
            if i % 2 == 0:  # A sublattice
                for q_idx in range(self.num_Q):
                    amp, phi, theta, psi = spin_params_A[q_idx]
                    ex, ey, ez = self.RotatedBasis(phi, theta, psi)
                    Q = Q_vecs[q_idx]
                    
                    spin += amp * (ex * np.cos(Q.dot(pos)) + ey * np.sin(Q.dot(pos)))
            else:  # B sublattice
                for q_idx in range(self.num_Q):
                    amp, phi, theta, psi = spin_params_B[q_idx]
                    ex, ey, ez = self.RotatedBasis(phi, theta, psi)
                    Q = Q_vecs[q_idx]
                    
                    spin += amp * (ex * np.cos(Q.dot(pos)) + ey * np.sin(Q.dot(pos)))
            
            # Normalize the spin
            norm = np.linalg.norm(spin)
            if norm > 1e-10:
                spins[i] = spin / norm
            else:
                spins[i] = np.array([0, 0, 1])  # Default to z-direction
        
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
        
        # Scalar chirality (measure of non-coplanarity)
        chirality = self.calculate_scalar_chirality(spins)
        
        return {
            'total': total_magnetization,
            'total_magnitude': total_magnitude,
            'A': mag_A,
            'B': mag_B,
            'A_magnitude': mag_A_magnitude,
            'B_magnitude': mag_B_magnitude,
            'staggered': staggered,
            'staggered_magnitude': staggered_magnitude,
            'scalar_chirality': chirality
        }
    
    def calculate_scalar_chirality(self, spins):
        """
        Calculate the scalar chirality, which measures the non-coplanarity
        of the spin configuration. Non-zero chirality is a signature of
        skyrmion-like or non-coplanar magnetic textures.
        
        Chirality = S_i · (S_j × S_k) for triangular plaquettes
        """
        chirality_sum = 0.0
        count = 0
        
        # Iterate over triangular plaquettes
        for i in range(0, len(spins), 2):  # A sublattice sites
            if i < len(spins) - 1:
                # Find nearest neighbors
                for nn_idx in self.NN[i]:
                    if nn_idx < len(spins) and nn_idx % 2 == 1:  # B sublattice
                        # Find common neighbor to form triangle
                        for nn2_idx in self.NN[nn_idx]:
                            if nn2_idx < len(spins) and nn2_idx != i and nn2_idx % 2 == 0:
                                # Calculate chirality for this triangle
                                s_i = spins[i]
                                s_j = spins[nn_idx]
                                s_k = spins[nn2_idx]
                                
                                chirality = np.dot(s_i, np.cross(s_j, s_k))
                                chirality_sum += chirality
                                count += 1
        
        if count > 0:
            return chirality_sum / count
        else:
            return 0.0
    
    def print_summary(self):
        """Print a summary of the multi-Q state"""
        Q_vecs, spin_params_A, spin_params_B = self.parse_params(self.opt_params)
        
        print("="*60)
        print(f"Multi-Q Analysis Results (num_Q = {self.num_Q})")
        print("="*60)
        print(f"Ground state energy: {self.opt_energy:.6f}")
        print(f"Magnetic order: {self.magnetic_order}")
        print(f"\nBCAO Parameters: J1xy={self.J[0]}, J1z={self.J[1]}, D={self.J[2]}, "
              f"E={self.J[3]}, F={self.J[4]}, G={self.J[5]}, J3xy={self.J[6]}, J3z={self.J[7]}")
        
        print("\nQ-vectors and amplitudes:")
        print("-"*60)
        for i in range(self.num_Q):
            Q_reduced = np.array([Q_vecs[i].dot(self.b1) / (2*np.pi), 
                                 Q_vecs[i].dot(self.b2) / (2*np.pi)])
            amp_A = spin_params_A[i][0]
            amp_B = spin_params_B[i][0]
            print(f"  Q{i+1}: ({Q_reduced[0]:.4f}, {Q_reduced[1]:.4f}) "
                  f"| Amplitude A: {amp_A:.4f}, B: {amp_B:.4f}")
    
    def calculate_structure_factor(self, spins, q_points=None, num_q_points=100):
        """
        Calculate the magnetic structure factor S(q) to identify the dominant Q-vectors
        
        S(q) = (1/N) |Σ_i S_i e^{iq·r_i}|^2
        
        Args:
            spins: Spin configuration (N x 3 array)
            q_points: Specific q-points to evaluate (optional)
            num_q_points: Number of q-points along each direction for 2D scan
        
        Returns:
            If q_points provided: array of S(q) values
            Otherwise: q_grid, structure_factor_2d for plotting
        """
        N = len(spins)
        
        if q_points is not None:
            # Evaluate at specific q-points
            S_q = np.zeros(len(q_points))
            for idx, q in enumerate(q_points):
                phase = np.exp(1j * np.dot(self.positions, q))
                S_vec = np.sum(spins * phase[:, np.newaxis], axis=0) / N
                S_q[idx] = np.linalg.norm(S_vec)**2
            return S_q
        else:
            # 2D scan in reciprocal space
            q1_range = np.linspace(-0.5, 0.5, num_q_points)
            q2_range = np.linspace(-0.5, 0.5, num_q_points)
            
            S_q_2d = np.zeros((num_q_points, num_q_points))
            
            for i, q1 in enumerate(q1_range):
                for j, q2 in enumerate(q2_range):
                    q = q1 * self.b1 + q2 * self.b2
                    phase = np.exp(1j * np.dot(self.positions, q))
                    S_vec = np.sum(spins * phase[:, np.newaxis], axis=0) / N
                    S_q_2d[i, j] = np.linalg.norm(S_vec)**2
            
            return q1_range, q2_range, S_q_2d
    
    def plot_structure_factor(self, spins, save_path=None):
        """
        Plot the magnetic structure factor in 2D reciprocal space
        """
        q1_range, q2_range, S_q = self.calculate_structure_factor(spins)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.contourf(q1_range, q2_range, S_q.T, levels=50, cmap='hot')
        ax.contour(q1_range, q2_range, S_q.T, levels=10, colors='black', linewidths=0.5, alpha=0.3)
        
        # Mark high-symmetry points
        ax.plot(0, 0, 'wo', markersize=10, markeredgecolor='blue', markeredgewidth=2, label='Γ')
        ax.plot(0.5, 0, 'ws', markersize=10, markeredgecolor='blue', markeredgewidth=2, label='M')
        ax.plot(1/3, 1/3, 'w^', markersize=10, markeredgecolor='blue', markeredgewidth=2, label='K')
        
        # Mark predicted Q-vectors from optimization
        Q_vecs, spin_params_A, spin_params_B = self.parse_params(self.opt_params)
        for i, Q in enumerate(Q_vecs):
            Q_reduced = np.array([Q.dot(self.b1) / (2*np.pi), Q.dot(self.b2) / (2*np.pi)])
            amp_A = spin_params_A[i][0]
            amp_B = spin_params_B[i][0]
            if amp_A > 0.1 or amp_B > 0.1:
                ax.plot(Q_reduced[0], Q_reduced[1], 'c*', markersize=15, 
                       markeredgecolor='white', markeredgewidth=1.5,
                       label=f'Q{i+1} (opt)')
        
        ax.set_xlabel('q1 (r.l.u.)', fontsize=14)
        ax.set_ylabel('q2 (r.l.u.)', fontsize=14)
        ax.set_title(f'Magnetic Structure Factor S(q) - {self.num_Q}-Q state', fontsize=16)
        ax.legend(fontsize=10, loc='upper right')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('S(q) Intensity', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Structure factor plot saved to {save_path}")
        
        plt.show()
        
        return fig, ax


def visualize_comparison(L=8, J=[-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]):
    """
    Create side-by-side visualization of single-Q vs multi-Q spin configurations
    """
    from single_q_BCAO import SingleQ
    
    print("Generating single-Q configuration...")
    single_q = SingleQ(L, J)
    spins_1q = single_q.generate_spin_configuration()
    
    print("Generating 2-Q configuration...")
    multi_q_2 = MultiQ(L, 2, J)
    spins_2q = multi_q_2.generate_spin_configuration()
    
    print("Generating 3-Q configuration...")
    multi_q_3 = MultiQ(L, 3, J)
    spins_3q = multi_q_3.generate_spin_configuration()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Single-Q plot
    ax1 = fig.add_subplot(131, projection='3d')
    for i, pos in enumerate(single_q.positions):
        color = 'red' if i % 2 == 0 else 'blue'
        ax1.quiver(pos[0], pos[1], 0, spins_1q[i, 0], spins_1q[i, 1], spins_1q[i, 2],
                  color=color, arrow_length_ratio=0.3, linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(f'Single-Q (E={single_q.opt_energy:.3f})\n{single_q.magnetic_order}', fontsize=12)
    ax1.set_xlim([-1, L+1])
    ax1.set_ylim([-1, L+1])
    ax1.set_zlim([-1, 1])
    
    # 2-Q plot
    ax2 = fig.add_subplot(132, projection='3d')
    for i, pos in enumerate(multi_q_2.positions):
        color = 'red' if i % 2 == 0 else 'blue'
        ax2.quiver(pos[0], pos[1], 0, spins_2q[i, 0], spins_2q[i, 1], spins_2q[i, 2],
                  color=color, arrow_length_ratio=0.3, linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title(f'2-Q (E={multi_q_2.opt_energy:.3f})\n{multi_q_2.magnetic_order}', fontsize=12)
    ax2.set_xlim([-1, L+1])
    ax2.set_ylim([-1, L+1])
    ax2.set_zlim([-1, 1])
    
    # 3-Q plot
    ax3 = fig.add_subplot(133, projection='3d')
    for i, pos in enumerate(multi_q_3.positions):
        color = 'red' if i % 2 == 0 else 'blue'
        ax3.quiver(pos[0], pos[1], 0, spins_3q[i, 0], spins_3q[i, 1], spins_3q[i, 2],
                  color=color, arrow_length_ratio=0.3, linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.set_title(f'3-Q (E={multi_q_3.opt_energy:.3f})\n{multi_q_3.magnetic_order}', fontsize=12)
    ax3.set_xlim([-1, L+1])
    ax3.set_ylim([-1, L+1])
    ax3.set_zlim([-1, 1])
    
    plt.tight_layout()
    plt.savefig('spin_config_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison saved to spin_config_comparison.png")
    plt.show()
    
    # Plot structure factors
    print("\nPlotting structure factors...")
    multi_q_3.plot_structure_factor(spins_3q, save_path='structure_factor_3q.png')
    
    return fig


def parameter_scan(L=6, base_J=[-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157], 
                   scan_params=['J1xy', 'J3xy'], scan_ranges=[(-10, -5), (1, 4)], 
                   num_points=15):
    """
    Scan parameter space near the initial values to map out the phase diagram
    and determine when multi-Q becomes favorable over single-Q
    
    Args:
        L: Lattice size
        base_J: Base BCAO parameters [J1xy, J1z, D, E, F, G, J3xy, J3z]
        scan_params: List of parameter names to scan (e.g., ['J1xy', 'J3xy'])
        scan_ranges: List of (min, max) tuples for each scan parameter
        num_points: Number of points along each scan direction
    """
    from single_q_BCAO import SingleQ
    
    param_map = {
        'J1xy': 0, 'J1z': 1, 'D': 2, 'E': 3, 
        'F': 4, 'G': 5, 'J3xy': 6, 'J3z': 7
    }
    
    if len(scan_params) != 2:
        raise ValueError("Currently only supports 2D parameter scans")
    
    idx1 = param_map[scan_params[0]]
    idx2 = param_map[scan_params[1]]
    
    param1_range = np.linspace(scan_ranges[0][0], scan_ranges[0][1], num_points)
    param2_range = np.linspace(scan_ranges[1][0], scan_ranges[1][1], num_points)
    
    energy_1q = np.zeros((num_points, num_points))
    energy_2q = np.zeros((num_points, num_points))
    energy_3q = np.zeros((num_points, num_points))
    
    print(f"\nParameter Space Scan: {scan_params[0]} vs {scan_params[1]}")
    print("="*70)
    print(f"Base parameters: J = {base_J}")
    print(f"{scan_params[0]} range: {scan_ranges[0]}")
    print(f"{scan_params[1]} range: {scan_ranges[1]}")
    print(f"Grid: {num_points} x {num_points} = {num_points**2} points")
    print("="*70)
    
    total_calcs = num_points * num_points
    calc_count = 0
    
    for i, p1 in enumerate(param1_range):
        for j, p2 in enumerate(param2_range):
            calc_count += 1
            print(f"\nProgress: {calc_count}/{total_calcs} "
                  f"({100*calc_count/total_calcs:.1f}%) - "
                  f"{scan_params[0]}={p1:.3f}, {scan_params[1]}={p2:.3f}")
            
            # Construct parameter set
            J = base_J.copy()
            J[idx1] = p1
            J[idx2] = p2
            
            try:
                # Single-Q
                single_q = SingleQ(L, J)
                energy_1q[i, j] = single_q.opt_energy
                
                # 2-Q
                multi_q_2 = MultiQ(L, 2, J)
                energy_2q[i, j] = multi_q_2.opt_energy
                
                # 3-Q
                multi_q_3 = MultiQ(L, 3, J)
                energy_3q[i, j] = multi_q_3.opt_energy
                
                print(f"  Energies: 1Q={energy_1q[i,j]:.3f}, "
                      f"2Q={energy_2q[i,j]:.3f}, 3Q={energy_3q[i,j]:.3f}")
            except Exception as e:
                print(f"  Error at this point: {e}")
                energy_1q[i, j] = np.nan
                energy_2q[i, j] = np.nan
                energy_3q[i, j] = np.nan
    
    # Determine ground state at each point
    ground_state = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            energies = [energy_1q[i, j], energy_2q[i, j], energy_3q[i, j]]
            if not any(np.isnan(energies)):
                ground_state[i, j] = np.argmin(energies)  # 0=1Q, 1=2Q, 2=3Q
            else:
                ground_state[i, j] = -1
    
    # Plot phase diagram
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Energy difference: 1Q - 3Q
    ax1 = axes[0, 0]
    energy_diff = energy_1q - energy_3q
    im1 = ax1.contourf(param1_range, param2_range, energy_diff.T, levels=20, cmap='RdBu_r')
    ax1.contour(param1_range, param2_range, energy_diff.T, levels=[0], colors='black', linewidths=2)
    ax1.set_xlabel(scan_params[0], fontsize=12)
    ax1.set_ylabel(scan_params[1], fontsize=12)
    ax1.set_title('Energy Difference: E(1-Q) - E(3-Q)', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='ΔE')
    
    # Ground state phase diagram
    ax2 = axes[0, 1]
    im2 = ax2.contourf(param1_range, param2_range, ground_state.T, 
                       levels=[-0.5, 0.5, 1.5, 2.5], cmap='viridis')
    ax2.set_xlabel(scan_params[0], fontsize=12)
    ax2.set_ylabel(scan_params[1], fontsize=12)
    ax2.set_title('Ground State Phase Diagram', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
    cbar2.ax.set_yticklabels(['1-Q', '2-Q', '3-Q'])
    
    # Mark base parameter point
    base_p1 = base_J[idx1]
    base_p2 = base_J[idx2]
    ax2.plot(base_p1, base_p2, 'r*', markersize=20, markeredgecolor='white', 
            markeredgewidth=2, label='Base params')
    ax2.legend()
    
    # Energy landscapes for each method
    ax3 = axes[1, 0]
    im3 = ax3.contourf(param1_range, param2_range, energy_1q.T, levels=20, cmap='viridis')
    ax3.set_xlabel(scan_params[0], fontsize=12)
    ax3.set_ylabel(scan_params[1], fontsize=12)
    ax3.set_title('Single-Q Energy Landscape', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='E(1-Q)')
    
    ax4 = axes[1, 1]
    im4 = ax4.contourf(param1_range, param2_range, energy_3q.T, levels=20, cmap='viridis')
    ax4.set_xlabel(scan_params[0], fontsize=12)
    ax4.set_ylabel(scan_params[1], fontsize=12)
    ax4.set_title('Triple-Q Energy Landscape', fontsize=12)
    plt.colorbar(im4, ax=ax4, label='E(3-Q)')
    
    plt.tight_layout()
    plt.savefig('parameter_scan_phase_diagram.png', dpi=300, bbox_inches='tight')
    print("\n\nPhase diagram saved to parameter_scan_phase_diagram.png")
    plt.show()
    
    # Save data
    np.savez('parameter_scan_data.npz',
             param1_range=param1_range, param2_range=param2_range,
             energy_1q=energy_1q, energy_2q=energy_2q, energy_3q=energy_3q,
             ground_state=ground_state,
             scan_params=scan_params, base_J=base_J)
    print("Data saved to parameter_scan_data.npz")
    
    return param1_range, param2_range, energy_1q, energy_2q, energy_3q, ground_state


def compare_single_and_multi_Q():
    """
    Compare single-Q and multi-Q results to see if multi-Q provides lower energy
    """
    from single_q_BCAO import SingleQ
    
    L = 8
    J = [-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]
    
    print("\n" + "="*60)
    print("COMPARISON: Single-Q vs Multi-Q")
    print("="*60)
    
    # Single-Q calculation
    print("\nRunning Single-Q calculation...")
    single_q = SingleQ(L, J)
    print(f"Single-Q energy: {single_q.opt_energy:.6f}")
    print(f"Single-Q order: {single_q.magnetic_order}")
    
    # Multi-Q calculations
    for num_Q in [2, 3]:
        print(f"\nRunning Multi-Q calculation with {num_Q} Q-vectors...")
        multi_q = MultiQ(L, num_Q, J, use_global=True)
        multi_q.print_summary()
        
        energy_gain = single_q.opt_energy - multi_q.opt_energy
        print(f"\nEnergy gain over single-Q: {energy_gain:.6f}")
        
        if energy_gain > 1e-4:
            print("→ Multi-Q state has LOWER energy!")
        else:
            print("→ Single-Q state is already optimal")


if __name__ == "__main__":
    # Example usage
    L = 8
    num_Q = 2  # Try 2-Q or 3-Q ansatz
    J = [-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]

    print("Creating Multi-Q model...")
    model = MultiQ(L, num_Q, J)
    
    # Print summary
    model.print_summary()
    
    # Generate and analyze the spin configuration
    spins = model.generate_spin_configuration()
    
    # Calculate magnetization
    magnetization = model.calculate_magnetization(spins)
    
    # Print magnetization results
    print("\nMagnetization Analysis:")
    print("="*60)
    print(f"Total magnetization: [{magnetization['total'][0]:.6f}, "
          f"{magnetization['total'][1]:.6f}, {magnetization['total'][2]:.6f}]")
    print(f"Total magnetization magnitude: {magnetization['total_magnitude']:.6f}")
    print(f"Sublattice A magnitude: {magnetization['A_magnitude']:.6f}")
    print(f"Sublattice B magnitude: {magnetization['B_magnitude']:.6f}")
    print(f"Staggered magnetization magnitude: {magnetization['staggered_magnitude']:.6f}")
    print(f"Scalar chirality (non-coplanarity): {magnetization['scalar_chirality']:.6f}")
    
    # Visualize the spin configuration
    print("\nGenerating visualization...")
    visualize_spins(model.positions, spins, L)
    
    # Optional: Compare with single-Q
    # compare_single_and_multi_Q()
