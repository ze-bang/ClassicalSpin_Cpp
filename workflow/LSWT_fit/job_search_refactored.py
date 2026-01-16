"""
Linear Spin Wave Theory (LSWT) Parameter Fitting for BCAO

This script fits experimental spin wave dispersion data to a magnetic Hamiltonian
model using Linear Spin Wave Theory. It systematically searches parameter space
to find the best-fit exchange coupling parameters.

Author: Refactored version
Date: 2025-11-01
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import sys
import hamiltonian_init_new as ham


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class Config:
    """Configuration parameters for the fitting procedure."""
    
    # Grid resolution
    N_J1_POINTS = 50          # Number of J1 values to scan
    N_M1_A_POINTS = 50        # Number of b_M1_a values
    N_M1_B_POINTS = 50        # Number of b_M1_b values
    N_J1_RANGE = 100          # Points for computing valid J1 ranges
    
    # Physical constants
    SPIN = 0.5
    MAGNETIC_FIELD = 3.0      # Tesla
    
    # Fixed parameters from experiments
    A_GAMMA = -3.067          # Gamma-point gap parameter
    B_GAMMA = -7.423          # Gamma-point gap parameter
    
    # M1 eigenvalue ranges
    B_M1_A_MIN = 30.0
    B_M1_A_MAX = 60.0
    B_M1_B_MIN = 50.0
    B_M1_B_MAX = 80.0
    
    # E parameter (from command line)
    E_MIN = 0.0
    E_MAX = 2.0
    E_POINTS = 50
    
    # Fixed Hamiltonian parameters
    G_B = 5.0
    G_A = 5.0
    G_Z = 2.7
    MU_B = 0.05788381
    J2 = 0.0
    J2Z = 0.0
    
    # Crystal structure vectors
    DELTA_1 = np.array([[0, 1], 
                        [-np.sqrt(3)/2, -1/2], 
                        [np.sqrt(3)/2, -1/2]])
    
    DELTA_2 = np.array([[np.sqrt(3), 0], 
                        [np.sqrt(3)/2, 1.5], 
                        [-np.sqrt(3)/2, 1.5],
                        [-np.sqrt(3), 0], 
                        [-np.sqrt(3)/2, -1.5], 
                        [np.sqrt(3)/2, -1.5]])
    
    DELTA_3 = np.array([[0, -2], 
                        [np.sqrt(3), 1], 
                        [-np.sqrt(3), 1]])
    
    PHI = np.array([0, 2*np.pi/3, -2*np.pi/3])


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

class DataLoader:
    """Handles loading and preprocessing of experimental dispersion data."""
    
    def __init__(self):
        self.data_lower = None
        self.data_upper = None
        
    def load_csv_files(self):
        """Load experimental data from CSV files."""
        print("Loading experimental data...")
        
        df1 = pd.read_csv("plot-data big1.csv")
        df2 = pd.read_csv("plot-data big2.csv")
        df3 = pd.read_csv("plot-data big3.csv")
        df4 = pd.read_csv("plot-data big3 up.csv")
        
        # Extract x, y columns
        data1 = df1[["x", " y"]].values
        data2 = df2[["x", " y"]].values
        data3 = df3[["x", " y"]].values
        data4 = df4[["x", " y"]].values
        
        # Stack all data
        data_all = np.vstack((data1, data2, data3, data4))
        
        # Manual corrections
        data_all[48, 0] -= 0.01
        data_all[58, 0] -= 0.01
        
        return data_all
    
    def transform_to_momentum_space(self, data_all):
        """
        Transform linearized dispersion data to proper momentum space coordinates.
        
        The data is along a path in the Brillouin zone. This function converts
        the normalized path coordinate to actual (qx, qy) momentum coordinates.
        """
        print("Transforming to momentum space...")
        
        pi = np.pi
        sq = np.sqrt(3)
        
        # Total path length in BZ
        L = 4*pi/3 + 8*pi/(3*sq) + 2*pi/3 + 2*pi/(3*sq)
        
        # Normalized positions along path
        r1 = (2*pi/3) / L
        r2 = (4*pi/3) / L
        r3 = (4*pi/3 + 4*pi/(3*sq)) / L
        r4 = (4*pi/3 + 8*pi/(3*sq)) / L
        r5 = (6*pi/3 + 8*pi/(3*sq)) / L
        r6 = (6*pi/3 + 8*pi/(3*sq) + 2*pi/(3*sq)) / L
        
        # Separate data into path segments
        segments = self._separate_path_segments(data_all, [r1, r2, r3, r4, r5, r6])
        
        # Transform each segment to momentum coordinates
        transformed_segments = self._transform_segments(segments, pi, sq)
        
        # Separate into lower and upper bands
        data_lower = np.hstack([transformed_segments[i] for i in [0, 1, 2, 3, 4, 6]])
        data_upper = np.hstack([transformed_segments[i] for i in [5, 7]])
        
        print(f"  Lower band: {data_lower.shape[1]} points")
        print(f"  Upper band: {data_upper.shape[1]} points")
        
        return data_lower, data_upper
    
    def _separate_path_segments(self, data, ratios):
        """Separate data into 8 path segments based on position ratios."""
        r1, r2, r3, r4, r5, r6 = ratios
        
        segments = [[] for _ in range(8)]
        
        for i in range(len(data)):
            x, y = data[i, 0], data[i, 1]
            
            if x < r1:
                segments[0].append([x, y])
            elif x < r2:
                segments[1].append([x, y])
            elif x < r3:
                segments[2].append([x, y])
            elif x < r4:
                segments[3].append([x, y])
            elif x < r5 and y < 8:
                segments[4].append([x, y])
            elif x < r5 and y > 8:
                segments[5].append([x, y])
            elif x < r6 and y < 9:
                segments[6].append([x, y])
            elif x < r6 and y > 9:
                segments[7].append([x, y])
        
        return [np.array(seg) for seg in segments]
    
    def _transform_segments(self, segments, pi, sq):
        """Transform each segment to momentum coordinates."""
        transformed = []
        
        # Precompute ratios
        r1 = (2*pi/3) / (4*pi/3 + 8*pi/(3*sq) + 2*pi/3 + 2*pi/(3*sq))
        r2 = (4*pi/3) / (4*pi/3 + 8*pi/(3*sq) + 2*pi/3 + 2*pi/(3*sq))
        r3 = (4*pi/3 + 4*pi/(3*sq)) / (4*pi/3 + 8*pi/(3*sq) + 2*pi/3 + 2*pi/(3*sq))
        r4 = (4*pi/3 + 8*pi/(3*sq)) / (4*pi/3 + 8*pi/(3*sq) + 2*pi/3 + 2*pi/(3*sq))
        r5 = (6*pi/3 + 8*pi/(3*sq)) / (4*pi/3 + 8*pi/(3*sq) + 2*pi/3 + 2*pi/(3*sq))
        r6 = (6*pi/3 + 8*pi/(3*sq) + 2*pi/(3*sq)) / (4*pi/3 + 8*pi/(3*sq) + 2*pi/3 + 2*pi/(3*sq))
        
        for idx, seg in enumerate(segments):
            if len(seg) == 0:
                transformed.append(np.zeros((3, 0)))
                continue
            
            data_seg = np.zeros((3, len(seg)))
            data_seg[2] = seg[:, 1]  # Energy values
            
            if idx == 0:  # Segment 1: Γ → M
                data_seg[1] = 2*pi/3 * seg[:, 0] / r1
                
            elif idx == 1:  # Segment 2: M → K
                x_normalized = 2*pi/3 * (seg[:, 0] - r1) / (r2 - r1) - 2*pi/3
                data_seg[0] = x_normalized * 0.5 * sq
                data_seg[1] = -x_normalized * 0.5
                
            elif idx == 2:  # Segment 3: K → Γ
                x_normalized = 4*pi/(3*sq) * (seg[:, 0] - r2) / (r3 - r2)
                data_seg[0] = -x_normalized * 0.5
                data_seg[1] = x_normalized * 0.5 * sq
                
            elif idx == 3:  # Segment 4: Γ → M'
                data_seg[0] = -(4*pi/(3*sq) * (seg[:, 0] - r3) / (r4 - r3) - 4*pi/(3*sq))
                
            elif idx == 4:  # Segment 5: M' → Γ (lower)
                data_seg[1] = 2*pi/3 * (seg[:, 0] - r4) / (r5 - r4)
                data_seg[1] = data_seg[1][::-1] + 2*pi/3
                
            elif idx == 5:  # Segment 6: M' → Γ (upper)
                data_seg[1] = 2*pi/3 * (seg[:, 0] - r4) / (r5 - r4)
                data_seg[1] = data_seg[1][::-1] + 2*pi/3
                
            elif idx == 6:  # Segment 7: Γ → K (lower)
                data_seg[0] = -2*pi/(3*sq) * (seg[:, 0] - r5) / (r6 - r5)
                data_seg[1] = np.ones(len(seg)) * 2*pi/3
                
            elif idx == 7:  # Segment 8: Γ → K (upper)
                data_seg[0] = -2*pi/(3*sq) * (seg[:, 0] - r5) / (r6 - r5)
                data_seg[1] = np.ones(len(seg)) * 2*pi/3
            
            transformed.append(data_seg)
        
        return transformed
    
    def load_and_process(self):
        """Complete data loading and processing pipeline."""
        data_all = self.load_csv_files()
        self.data_lower, self.data_upper = self.transform_to_momentum_space(data_all)
        return self.data_lower, self.data_upper


# =============================================================================
# PARAMETER CONSTRAINTS
# =============================================================================

class ParameterConstraints:
    """
    Implements the LSWT-derived constraints on exchange parameters.
    
    These constraints come from eigenvalue analysis at high-symmetry points
    (particularly the M1 point) in the Brillouin zone.
    """
    
    @staticmethod
    def c_M1_a(b_M1_a):
        """Linear constraint: c_M1_a as function of b_M1_a."""
        return 0.12817723 * b_M1_a - 0.04713861
    
    @staticmethod
    def c_M1_b(b_M1_b):
        """Linear constraint: c_M1_b as function of b_M1_b."""
        return -(0.12153832 * b_M1_b - 1.09268781)
    
    @staticmethod
    def J3(J1, S, b_g):
        """Third-neighbor exchange from Γ-point constraint."""
        return (b_g - 3*J1*S) / (3*S)
    
    @staticmethod
    def J1z(J1, S, a_g, b_g, c_M1_b, c_M1_a):
        """Anisotropic first-neighbor exchange from M1 constraints."""
        return (c_M1_b - c_M1_a + 2*a_g + 2*b_g - 8*J1*S) / (8*S)
    
    @staticmethod
    def J3z(J1, S, a_g, b_g, c_M1_b, c_M1_a):
        """Anisotropic third-neighbor exchange from M1 constraints."""
        return (c_M1_a - c_M1_b + 2*a_g/3 - 2*b_g + 8*J1*S) / (8*S)
    
    @staticmethod
    def D(S, c_M1_a, c_M1_b):
        """Symmetric anisotropy parameter."""
        return (c_M1_b + c_M1_a) / (4*S)
    
    @staticmethod
    def F_squared(J1, S, b_g, c_M1_a, c_M1_b, b_M1_a):
        """
        Off-diagonal Kitaev interaction F squared.
        Must be non-negative for physical parameters.
        """
        value = (c_M1_a**2 - 4*b_M1_a - 
                (c_M1_b - 8*J1*S) * (4*b_g + c_M1_b - 8*J1*S)) / (16*S**2)
        return np.maximum(value, 0.0)
    
    @staticmethod
    def G_squared(J1, S, b_g, c_M1_a, c_M1_b, b_M1_b):
        """
        Off-diagonal Kitaev interaction G squared.
        Must be non-negative for physical parameters.
        """
        value = (c_M1_b**2 - 4*b_M1_b + 
                (c_M1_a + 8*J1*S) * (4*b_g - c_M1_a - 8*J1*S)) / (16*S**2)
        return np.maximum(value, 0.0)
    
    @staticmethod
    def compute_valid_J1_range(b_M1_a, c_M1_a, b_M1_b, c_M1_b, 
                               S, a_g, b_g, n_points=100):
        """
        Compute the range of J1 values where both F² ≥ 0 and G² ≥ 0.
        
        Returns:
            (J1_min, J1_max): Valid range for J1
        """
        J1_test = np.linspace(-9, -2, n_points)
        
        F2_vals = np.array([ParameterConstraints.F_squared(
            j1, S, b_g, c_M1_a, c_M1_b, b_M1_a) for j1 in J1_test])
        G2_vals = np.array([ParameterConstraints.G_squared(
            j1, S, b_g, c_M1_a, c_M1_b, b_M1_b) for j1 in J1_test])
        
        # Find where F² is non-negative
        valid_F = F2_vals > 0
        if np.any(valid_F):
            F_start = J1_test[np.argmax(valid_F)]
            F_end = J1_test[len(valid_F) - 1 - np.argmax(valid_F[::-1])]
        else:
            F_start = F_end = J1_test[0]
        
        # Find where G² is non-negative
        valid_G = G2_vals > 0
        if np.any(valid_G):
            G_start = J1_test[np.argmax(valid_G)]
            G_end = J1_test[len(valid_G) - 1 - np.argmax(valid_G[::-1])]
        else:
            G_start = G_end = J1_test[0]
        
        # Valid range is intersection
        J1_min = np.maximum(F_start, G_start)
        J1_max = np.minimum(F_end, G_end)
        
        return J1_min, J1_max


# =============================================================================
# HAMILTONIAN CALCULATION AND FITTING
# =============================================================================

class SpinWaveFitter:
    """Performs spin wave dispersion fitting."""
    
    def __init__(self, data_lower, data_upper, config):
        self.data_lower = data_lower
        self.data_upper = data_upper
        self.config = config
        
        # Create base parameter dictionary
        self.base_params = self._create_base_params()
    
    def _create_base_params(self):
        """Create base parameter dictionary for Hamiltonian."""
        return {
            'g_b': self.config.G_B,
            'g_a': self.config.G_A,
            'g_z': self.config.G_Z,
            'mu_B': self.config.MU_B,
            'S': self.config.SPIN,
            'J2': self.config.J2,
            'J2z': self.config.J2Z,
            'delta_1': self.config.DELTA_1,
            'delta_2': self.config.DELTA_2,
            'delta_3': self.config.DELTA_3,
            'phi': self.config.PHI
        }
    
    def compute_dispersion(self, parameters, B, direction='b'):
        """
        Compute theoretical dispersion for given parameters.
        
        Returns:
            (energies_lower, energies_upper): Arrays of energies at data points
        """
        n_lower = self.data_lower.shape[1]
        n_upper = self.data_upper.shape[1]
        
        energies_lower = np.zeros(n_lower)
        energies_upper = np.zeros(n_upper)
        
        # Compute lower band
        for i in range(n_lower):
            kvec = self.data_lower[:2, i]
            eigenvals = ham.hamiltonian_diagonalization(kvec, direction, parameters, B)
            energies_lower[i] = np.real(eigenvals[1])  # Second eigenvalue
        
        # Compute upper band
        for i in range(n_upper):
            kvec = self.data_upper[:2, i]
            eigenvals = ham.hamiltonian_diagonalization(kvec, direction, parameters, B)
            energies_upper[i] = np.real(eigenvals[0])  # First eigenvalue
        
        return energies_lower, energies_upper
    
    def compute_r_squared(self, energies_lower, energies_upper):
        """
        Compute R² goodness-of-fit metric.
        
        Returns:
            (r2_total, r2_lower): Total and lower-band R² values
        """
        # Lower band
        residuals_lower = energies_lower - self.data_lower[2]
        variance_lower = self.data_lower[2] - np.mean(self.data_lower[2])
        ss_res_lower = np.sum(residuals_lower**2)
        ss_tot_lower = np.sum(variance_lower**2)
        
        # Upper band
        residuals_upper = energies_upper - self.data_upper[2]
        variance_upper = self.data_upper[2] - np.mean(self.data_upper[2])
        ss_res_upper = np.sum(residuals_upper**2)
        ss_tot_upper = np.sum(variance_upper**2)
        
        # Combined R²
        r2_total = 1 - (ss_res_lower + ss_res_upper) / (ss_tot_lower + ss_tot_upper)
        r2_lower = 1 - ss_res_lower / ss_tot_lower
        
        return r2_total, r2_lower
    
    def fit_single_point(self, b_M1_a_val, c_M1_a_val, b_M1_b_val, c_M1_b_val,
                         J1_val, E_val, F_sign, G_sign):
        """
        Compute fit quality for a single point in parameter space.
        
        Args:
            b_M1_a_val, c_M1_a_val: M1 eigenvalues for sublattice a
            b_M1_b_val, c_M1_b_val: M1 eigenvalues for sublattice b
            J1_val: First-neighbor exchange
            E_val: Asymmetric anisotropy parameter
            F_sign, G_sign: Signs for F and G parameters (+1 or -1)
        
        Returns:
            (kitaev_vals, r2_total, r2_lower): Kitaev parameters and fit quality
        """
        # Compute derived parameters
        J3_val = ParameterConstraints.J3(J1_val, self.config.SPIN, self.config.B_GAMMA)
        J1z_val = ParameterConstraints.J1z(J1_val, self.config.SPIN, 
                                           self.config.A_GAMMA, self.config.B_GAMMA,
                                           c_M1_b_val, c_M1_a_val)
        J3z_val = ParameterConstraints.J3z(J1_val, self.config.SPIN,
                                           self.config.A_GAMMA, self.config.B_GAMMA,
                                           c_M1_b_val, c_M1_a_val)
        D_val = ParameterConstraints.D(self.config.SPIN, c_M1_a_val, c_M1_b_val)
        
        F_val = F_sign * np.sqrt(ParameterConstraints.F_squared(
            J1_val, self.config.SPIN, self.config.B_GAMMA,
            c_M1_a_val, c_M1_b_val, b_M1_a_val))
        
        G_val = G_sign * np.sqrt(ParameterConstraints.G_squared(
            J1_val, self.config.SPIN, self.config.B_GAMMA,
            c_M1_a_val, c_M1_b_val, b_M1_b_val))
        
        # Kitaev eigenvalues
        kitaev_vals = [D_val - np.sqrt(2) * F_val, D_val + np.sqrt(2) * F_val]
        
        # Create parameter dictionary
        parameters = self.base_params.copy()
        parameters.update({
            'J1': J1_val,
            'J3': J3_val,
            'J1z': J1z_val,
            'J3z': J3z_val,
            'D': D_val,
            'F': F_val,
            'G': G_val,
            'E': E_val
        })
        
        # Compute dispersion
        energies_lower, energies_upper = self.compute_dispersion(
            parameters, self.config.MAGNETIC_FIELD)
        
        # Compute fit quality
        r2_total, r2_lower = self.compute_r_squared(energies_lower, energies_upper)
        
        return kitaev_vals, r2_total, r2_lower


# =============================================================================
# PARAMETER SCAN
# =============================================================================

class ParameterScanner:
    """Orchestrates the systematic parameter space search."""
    
    def __init__(self, data_lower, data_upper, config):
        self.data_lower = data_lower
        self.data_upper = data_upper
        self.config = config
        self.fitter = SpinWaveFitter(data_lower, data_upper, config)
    
    def compute_valid_J1_ranges(self):
        """
        Pre-compute valid J1 ranges for all (b_M1_a, b_M1_b) combinations.
        
        Returns:
            J1_ranges: Array of shape (N_M1_A, N_M1_B, 2) with [min, max] J1 values
        """
        print("Computing valid J1 ranges...")
        
        b_M1_a_grid = np.linspace(self.config.B_M1_A_MIN, self.config.B_M1_A_MAX, 
                                  self.config.N_M1_A_POINTS)
        b_M1_b_grid = np.linspace(self.config.B_M1_B_MIN, self.config.B_M1_B_MAX,
                                  self.config.N_M1_B_POINTS)
        
        J1_ranges = np.zeros((self.config.N_M1_A_POINTS, 
                             self.config.N_M1_B_POINTS, 2))
        
        for i, b_a in enumerate(b_M1_a_grid):
            if i % 10 == 0:
                print(f"  Progress: {i}/{self.config.N_M1_A_POINTS}")
            
            c_a = ParameterConstraints.c_M1_a(b_a)
            
            for j, b_b in enumerate(b_M1_b_grid):
                c_b = ParameterConstraints.c_M1_b(b_b)
                
                J1_min, J1_max = ParameterConstraints.compute_valid_J1_range(
                    b_a, c_a, b_b, c_b,
                    self.config.SPIN, self.config.A_GAMMA, self.config.B_GAMMA,
                    self.config.N_J1_RANGE)
                
                J1_ranges[i, j, 0] = J1_min
                J1_ranges[i, j, 1] = J1_max
        
        print("Valid J1 ranges computed.")
        return J1_ranges, b_M1_a_grid, b_M1_b_grid
    
    def scan_single_sign_combination(self, J1_ranges, b_M1_a_grid, b_M1_b_grid,
                                     E_val, F_sign, G_sign, label):
        """
        Scan parameter space for a single sign combination of F and G.
        
        Args:
            J1_ranges: Pre-computed valid J1 ranges
            b_M1_a_grid, b_M1_b_grid: Grid values for M1 parameters
            E_val: Value of E parameter
            F_sign, G_sign: Signs for F and G (+1 or -1)
            label: Descriptive label for this combination
        
        Returns:
            results_dict: Dictionary with 'kitaev' and 'err_all' arrays
        """
        print(f"\nScanning sign combination: {label}")
        
        # Prepare storage
        kitaev_all = np.zeros((self.config.N_J1_POINTS, 
                              self.config.N_M1_A_POINTS,
                              self.config.N_M1_B_POINTS, 2))
        err_all = np.zeros((self.config.N_J1_POINTS,
                           self.config.N_M1_A_POINTS,
                           self.config.N_M1_B_POINTS, 2))
        
        # Create tasks for parallel execution
        tasks = []
        for i, b_a in enumerate(b_M1_a_grid):
            c_a = ParameterConstraints.c_M1_a(b_a)
            
            for j, b_b in enumerate(b_M1_b_grid):
                c_b = ParameterConstraints.c_M1_b(b_b)
                
                # Get valid J1 range for this (i, j) combination
                J1_min, J1_max = J1_ranges[i, j]
                
                if J1_max <= J1_min:
                    continue  # Skip if no valid range
                
                J1_vals = np.linspace(J1_min, J1_max, self.config.N_J1_POINTS)
                
                for k, J1_val in enumerate(J1_vals):
                    tasks.append((i, j, k, b_a, c_a, b_b, c_b, J1_val))
        
        print(f"  Total tasks: {len(tasks)}")
        
        # Parallel execution
        def process_task(task):
            i, j, k, b_a, c_a, b_b, c_b, J1_val = task
            kitaev, r2_total, r2_lower = self.fitter.fit_single_point(
                b_a, c_a, b_b, c_b, J1_val, E_val, F_sign, G_sign)
            return (i, j, k, kitaev, r2_total, r2_lower)
        
        results = Parallel(n_jobs=-1, verbose=5)(
            delayed(process_task)(task) for task in tasks)
        
        # Store results
        for i, j, k, kitaev, r2_total, r2_lower in results:
            kitaev_all[k, i, j, 0] = kitaev[0]
            kitaev_all[k, i, j, 1] = kitaev[1]
            err_all[k, i, j, 0] = r2_total
            err_all[k, i, j, 1] = r2_lower
        
        return {'kitaev': kitaev_all, 'err_all': err_all}
    
    def run_full_scan(self, E_index):
        """
        Run complete parameter scan for given E parameter index.
        
        Args:
            E_index: Index into E_array (from command line argument)
        
        Saves:
            Four .npy files with results for each sign combination
        """
        # Get E value
        E_array = np.linspace(self.config.E_MIN, self.config.E_MAX, 
                             self.config.E_POINTS)
        E_val = E_array[E_index]
        
        print(f"\n{'='*70}")
        print(f"Starting parameter scan for E = {E_val:.4f} (index {E_index})")
        print(f"{'='*70}\n")
        
        # Compute valid J1 ranges
        J1_ranges, b_M1_a_grid, b_M1_b_grid = self.compute_valid_J1_ranges()
        
        # Define sign combinations
        sign_combinations = [
            ('+1', '+1', 'all_+ve', 1, 1),
            ('+1', '-1', '+ve_-ve', 1, -1),
            ('-1', '+1', '-ve_+ve', -1, 1),
            ('-1', '-1', '-ve_-ve', -1, -1)
        ]
        
        # Scan each combination
        for F_label, G_label, filename_suffix, F_sign, G_sign in sign_combinations:
            print(f"\n{'='*70}")
            print(f"Sign combination: F={F_label}, G={G_label}")
            print(f"{'='*70}")
            
            results = self.scan_single_sign_combination(
                J1_ranges, b_M1_a_grid, b_M1_b_grid,
                E_val, F_sign, G_sign, f"F={F_label}, G={G_label}")
            
            # Save results
            filename = f'data_{filename_suffix}_E_{E_index}.npy'
            np.save(filename, results)
            print(f"\nResults saved to: {filename}")
        
        print(f"\n{'='*70}")
        print(f"Scan complete for E index {E_index}")
        print(f"{'='*70}\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    # Parse command line argument
    if len(sys.argv) < 2:
        print("Usage: python job_search_refactored.py <E_index>")
        print("  E_index: Integer from 0 to 49")
        sys.exit(1)
    
    E_index = int(sys.argv[1])
    
    if not (0 <= E_index < 50):
        print(f"Error: E_index must be between 0 and 49, got {E_index}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("LSWT Parameter Fitting for BCAO")
    print("="*70 + "\n")
    
    # Initialize configuration
    config = Config()
    
    # Load and process data
    print("Stage 1: Data Loading and Processing")
    print("-" * 70)
    loader = DataLoader()
    data_lower, data_upper = loader.load_and_process()
    
    # Run parameter scan
    print("\n\nStage 2: Parameter Space Scan")
    print("-" * 70)
    scanner = ParameterScanner(data_lower, data_upper, config)
    scanner.run_full_scan(E_index)
    
    print("\n" + "="*70)
    print("All processing complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
