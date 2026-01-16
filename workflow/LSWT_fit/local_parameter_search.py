"""
Local Parameter Search Around Known Solution

Given a parameter set (either from a previous fit or user-specified), this script
performs a local search in parameter space to find nearby solutions that meet
a specified R² threshold.

This is useful for:
1. Exploring parameter degeneracy/uncertainty
2. Finding alternative solutions near a good fit
3. Understanding the parameter space landscape

Author: Parameter landscape explorer
Date: 2025-11-01
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import sys
import yaml
import matplotlib.pyplot as plt
import hamiltonian_init_new as ham
from job_search_refactored import (
    Config, DataLoader, ParameterConstraints, SpinWaveFitter
)


# =============================================================================
# LOCAL SEARCH CONFIGURATION
# =============================================================================

class LocalSearchConfig(Config):
    """Extended configuration for local parameter search."""
    
    # Search grid resolution (points per parameter)
    N_LOCAL_POINTS = 15
    
    # Search range as fraction of parameter value
    # e.g., 0.1 means ±10% around the center value
    B_M1_A_RANGE_FRACTION = 0.15
    B_M1_B_RANGE_FRACTION = 0.15
    J1_RANGE_FRACTION = 0.15
    E_RANGE_FRACTION = 0.20
    
    # Absolute ranges (fallback if fractional gives too small range)
    B_M1_A_MIN_RANGE = 3.0
    B_M1_B_MIN_RANGE = 3.0
    J1_MIN_RANGE = 0.5
    E_MIN_RANGE = 0.2
    
    # R² threshold for accepting solutions
    R2_THRESHOLD = 0.90
    
    # Maximum number of solutions to return
    MAX_SOLUTIONS = 100
    
    # Sign combinations to test (can restrict to subset)
    TEST_SIGN_COMBINATIONS = [
        ('all_+ve', 1, 1),
        ('+ve_-ve', 1, -1),
        ('-ve_+ve', -1, 1),
        ('-ve_-ve', -1, -1)
    ]


# =============================================================================
# PARAMETER SET REPRESENTATION
# =============================================================================

class ParameterSet:
    """Represents a complete set of fitting parameters."""
    
    def __init__(self, J1xy, J1z, D, E, F, G, J3xy, J3z,
                 r2_total=None, r2_lower=None):
        """
        Initialize parameter set using physical exchange parameters.
        
        Args:
            J1xy: First-neighbor in-plane exchange (J1 in original code)
            J1z: First-neighbor anisotropic exchange
            D: Symmetric anisotropy
            E: Asymmetric anisotropy
            F: Off-diagonal Kitaev interaction
            G: Off-diagonal Kitaev interaction
            J3xy: Third-neighbor in-plane exchange (J3 in original code)
            J3z: Third-neighbor anisotropic exchange
            r2_total, r2_lower: Optional R² values
        """
        # Store physical parameters
        self.J1xy = J1xy
        self.J1z = J1z
        self.D = D
        self.E = E
        self.F = F
        self.G = G
        self.J3xy = J3xy
        self.J3z = J3z
        self.r2_total = r2_total
        self.r2_lower = r2_lower
        
        # Compute internal representation (b_M1_a, b_M1_b, J1)
        self._compute_internal_parameters()
        
        # Compute Kitaev eigenvalues
        self.K1 = self.D - np.sqrt(2) * self.F
        self.K2 = self.D + np.sqrt(2) * self.F
        
        # Store signs
        self.F_sign = 1 if self.F >= 0 else -1
        self.G_sign = 1 if self.G >= 0 else -1
    
    def _compute_internal_parameters(self):
        """
        Compute internal M1 eigenvalue parameters from physical parameters.
        
        This inverts the constraint equations to get b_M1_a, b_M1_b, J1 (which is J1xy).
        """
        config = Config()
        
        # J1 in the code is actually J1xy
        self.J1 = self.J1xy
        
        # From constraint: J3 = (b_g - 3*J1*S) / (3*S)
        # Invert: b_g = J3 * 3*S + 3*J1*S
        # But we use the constraint directly, so J3xy is already consistent
        self.J3 = self.J3xy
        
        # From constraints, we can solve for c_M1_a and c_M1_b:
        # D = (c_M1_b + c_M1_a) / (4*S)
        # J1z = (c_M1_b - c_M1_a + 2*a_g + 2*b_g - 8*J1*S) / (8*S)
        
        # From D: c_M1_b + c_M1_a = 4*S*D
        # From J1z: c_M1_b - c_M1_a = 8*S*J1z - 2*a_g - 2*b_g + 8*J1*S
        
        sum_c = 4 * config.SPIN * self.D
        diff_c = 8 * config.SPIN * self.J1z - 2*config.A_GAMMA - 2*config.B_GAMMA + 8*self.J1*config.SPIN
        
        self.c_M1_b = (sum_c + diff_c) / 2
        self.c_M1_a = (sum_c - diff_c) / 2
        
        # Now invert the linear constraints to get b_M1_a and b_M1_b
        # c_M1_a = 0.12817723 * b_M1_a - 0.04713861
        # c_M1_b = -(0.12153832 * b_M1_b - 1.09268781)
        
        self.b_M1_a = (self.c_M1_a + 0.04713861) / 0.12817723
        self.b_M1_b = (1.09268781 - self.c_M1_b) / 0.12153832
    
    @classmethod
    def from_internal(cls, b_M1_a, b_M1_b, J1, E, F_sign, G_sign,
                     r2_total=None, r2_lower=None):
        """
        Alternative constructor from internal M1 eigenvalue representation.
        
        This is used internally by the search engine.
        """
        config = Config()
        
        # Compute physical parameters from internal representation
        c_M1_a = ParameterConstraints.c_M1_a(b_M1_a)
        c_M1_b = ParameterConstraints.c_M1_b(b_M1_b)
        
        J1xy = J1
        J3xy = ParameterConstraints.J3(J1, config.SPIN, config.B_GAMMA)
        J1z = ParameterConstraints.J1z(
            J1, config.SPIN, config.A_GAMMA, config.B_GAMMA,
            c_M1_b, c_M1_a)
        J3z = ParameterConstraints.J3z(
            J1, config.SPIN, config.A_GAMMA, config.B_GAMMA,
            c_M1_b, c_M1_a)
        D = ParameterConstraints.D(config.SPIN, c_M1_a, c_M1_b)
        
        F2 = ParameterConstraints.F_squared(
            J1, config.SPIN, config.B_GAMMA,
            c_M1_a, c_M1_b, b_M1_a)
        G2 = ParameterConstraints.G_squared(
            J1, config.SPIN, config.B_GAMMA,
            c_M1_a, c_M1_b, b_M1_b)
        
        F = F_sign * np.sqrt(F2)
        G = G_sign * np.sqrt(G2)
        
        return cls(J1xy, J1z, D, E, F, G, J3xy, J3z, r2_total, r2_lower)
    
    def to_dict(self):
        """Convert to dictionary for easy saving/loading."""
        return {
            'J1xy': self.J1xy,
            'J1z': self.J1z,
            'D': self.D,
            'E': self.E,
            'F': self.F,
            'G': self.G,
            'J3xy': self.J3xy,
            'J3z': self.J3z,
            'K1': self.K1,
            'K2': self.K2,
            'r2_total': self.r2_total,
            'r2_lower': self.r2_lower,
            'F_sign': self.F_sign,
            'G_sign': self.G_sign,
            # Also include internal parameters for reference
            'b_M1_a': self.b_M1_a,
            'b_M1_b': self.b_M1_b,
            'c_M1_a': self.c_M1_a,
            'c_M1_b': self.c_M1_b
        }
    
    def __str__(self):
        """String representation."""
        r2_str = f"{self.r2_total:.6f}" if self.r2_total is not None else "N/A"
        return (f"ParameterSet(\n"
                f"  # Exchange interaction parameters\n"
                f"  J1xy = {self.J1xy:.3f}\n"
                f"  J1z  = {self.J1z:.3f}\n"
                f"  D    = {self.D:.3f}\n"
                f"  E    = {self.E:.3f}\n"
                f"  F    = {self.F:.3f}\n"
                f"  G    = {self.G:.3f}\n"
                f"  \n"
                f"  # Third nearest neighbor parameters\n"
                f"  J3xy = {self.J3xy:.3f}\n"
                f"  J3z  = {self.J3z:.3f}\n"
                f"  \n"
                f"  # Kitaev eigenvalues\n"
                f"  K1   = {self.K1:.3f}\n"
                f"  K2   = {self.K2:.3f}\n"
                f"  \n"
                f"  R²_total = {r2_str}\n"
                f")")


# =============================================================================
# LOCAL SEARCH ENGINE
# =============================================================================

class LocalSearchEngine:
    """Performs local search around a given parameter set."""
    
    def __init__(self, data_lower, data_upper, config):
        self.data_lower = data_lower
        self.data_upper = data_upper
        self.config = config
        self.fitter = SpinWaveFitter(data_lower, data_upper, config)
    
    def define_search_grid(self, center_params):
        """
        Define a local search grid around center parameters.
        
        Args:
            center_params: ParameterSet object
        
        Returns:
            Dictionary with parameter grids
        """
        print("\nDefining local search grid...")
        
        # Compute search ranges (use internal representation for grid)
        b_M1_a_range = max(
            center_params.b_M1_a * self.config.B_M1_A_RANGE_FRACTION,
            self.config.B_M1_A_MIN_RANGE
        )
        b_M1_b_range = max(
            center_params.b_M1_b * self.config.B_M1_B_RANGE_FRACTION,
            self.config.B_M1_B_MIN_RANGE
        )
        J1xy_range = max(
            abs(center_params.J1xy) * self.config.J1_RANGE_FRACTION,
            self.config.J1_MIN_RANGE
        )
        E_range = max(
            center_params.E * self.config.E_RANGE_FRACTION,
            self.config.E_MIN_RANGE
        )
        
        # Create grids
        b_M1_a_grid = np.linspace(
            center_params.b_M1_a - b_M1_a_range,
            center_params.b_M1_a + b_M1_a_range,
            self.config.N_LOCAL_POINTS
        )
        b_M1_b_grid = np.linspace(
            center_params.b_M1_b - b_M1_b_range,
            center_params.b_M1_b + b_M1_b_range,
            self.config.N_LOCAL_POINTS
        )
        J1xy_grid = np.linspace(
            center_params.J1xy - J1xy_range,
            center_params.J1xy + J1xy_range,
            self.config.N_LOCAL_POINTS
        )
        E_grid = np.linspace(
            max(0, center_params.E - E_range),
            center_params.E + E_range,
            self.config.N_LOCAL_POINTS
        )
        
        # Apply physical bounds
        b_M1_a_grid = np.clip(b_M1_a_grid, 
                             self.config.B_M1_A_MIN, self.config.B_M1_A_MAX)
        b_M1_b_grid = np.clip(b_M1_b_grid,
                             self.config.B_M1_B_MIN, self.config.B_M1_B_MAX)
        J1xy_grid = np.clip(J1xy_grid, -10, -1)  # Reasonable J1xy range
        E_grid = np.clip(E_grid, self.config.E_MIN, self.config.E_MAX)
        
        print(f"  J1xy:   [{J1xy_grid[0]:.4f}, {J1xy_grid[-1]:.4f}]")
        print(f"  E:      [{E_grid[0]:.4f}, {E_grid[-1]:.4f}]")
        print(f"  (Internal: b_M1_a: [{b_M1_a_grid[0]:.4f}, {b_M1_a_grid[-1]:.4f}])")
        print(f"  (Internal: b_M1_b: [{b_M1_b_grid[0]:.4f}, {b_M1_b_grid[-1]:.4f}])")
        
        return {
            'b_M1_a': b_M1_a_grid,
            'b_M1_b': b_M1_b_grid,
            'J1xy': J1xy_grid,
            'E': E_grid
        }
    
    def validate_parameters(self, b_M1_a, b_M1_b, J1):
        """
        Check if parameters satisfy physical constraints:
        - F² ≥ 0, G² ≥ 0
        - max(|K1|, |K2|) < |J1|/2
        
        Returns:
            bool: True if valid, False otherwise
        """
        c_M1_a = ParameterConstraints.c_M1_a(b_M1_a)
        c_M1_b = ParameterConstraints.c_M1_b(b_M1_b)
        
        # Check F² and G² are non-negative
        F2 = ParameterConstraints.F_squared(
            J1, self.config.SPIN, self.config.B_GAMMA,
            c_M1_a, c_M1_b, b_M1_a)
        G2 = ParameterConstraints.G_squared(
            J1, self.config.SPIN, self.config.B_GAMMA,
            c_M1_a, c_M1_b, b_M1_b)
        
        if F2 < 0 or G2 < 0:
            return False
        
        # Check Kitaev eigenvalue constraint: max(|K1|, |K2|) < |J1|/2
        # K1 = D - sqrt(2)*F, K2 = D + sqrt(2)*F
        # We need to compute D and F to check this
        D = ParameterConstraints.D(self.config.SPIN, c_M1_a, c_M1_b)
        F = np.sqrt(F2)  # Take positive root for checking
        
        K1 = D - np.sqrt(2) * F
        K2 = D + np.sqrt(2) * F
        
        max_K = max(abs(K1), abs(K2))
        
        if max_K >= abs(J1) / 2:
            return False
        
        return True
    
    def search_around_point(self, center_params, test_all_signs=True):
        """
        Perform local search around a parameter set.
        
        Args:
            center_params: ParameterSet to search around
            test_all_signs: If True, test all sign combinations.
                           If False, only test the original signs.
        
        Returns:
            list of ParameterSet objects meeting R² threshold
        """
        print(f"\n{'='*70}")
        print("LOCAL PARAMETER SEARCH")
        print(f"{'='*70}")
        print(f"\nCenter parameters:")
        print(center_params)
        print(f"\nR² threshold: {self.config.R2_THRESHOLD:.4f}")
        
        # Define search grid
        grids = self.define_search_grid(center_params)
        
        # Determine which sign combinations to test
        if test_all_signs:
            sign_combos = self.config.TEST_SIGN_COMBINATIONS
        else:
            # Only test the original sign combination
            sign_combos = [('original', center_params.F_sign, center_params.G_sign)]
        
        print(f"\nTesting {len(sign_combos)} sign combination(s)")
        
        # Generate all parameter combinations
        tasks = []
        for label, F_sign, G_sign in sign_combos:
            for b_a in grids['b_M1_a']:
                for b_b in grids['b_M1_b']:
                    for J1xy_val in grids['J1xy']:
                        # Check validity
                        if not self.validate_parameters(b_a, b_b, J1xy_val):
                            continue
                        
                        for E_val in grids['E']:
                            tasks.append((b_a, b_b, J1xy_val, E_val, F_sign, G_sign))
        
        print(f"\nTotal parameter combinations to test: {len(tasks)}")
        print("Starting parallel evaluation...")
        
        # Parallel execution
        def evaluate_point(task):
            b_a, b_b, J1xy_val, E_val, F_sign, G_sign = task
            
            c_a = ParameterConstraints.c_M1_a(b_a)
            c_b = ParameterConstraints.c_M1_b(b_b)
            
            kitaev, r2_total, r2_lower = self.fitter.fit_single_point(
                b_a, c_a, b_b, c_b, J1xy_val, E_val, F_sign, G_sign)
            
            if r2_total >= self.config.R2_THRESHOLD:
                return ParameterSet.from_internal(
                    b_a, b_b, J1xy_val, E_val, F_sign, G_sign,
                    r2_total, r2_lower)
            return None
        
        results = Parallel(n_jobs=-1, verbose=5)(
            delayed(evaluate_point)(task) for task in tasks)
        
        # Filter out None results and sort by R²
        solutions = [r for r in results if r is not None]
        solutions.sort(key=lambda x: x.r2_total, reverse=True)
        
        # Limit to max solutions
        if len(solutions) > self.config.MAX_SOLUTIONS:
            solutions = solutions[:self.config.MAX_SOLUTIONS]
        
        print(f"\n{'='*70}")
        print(f"Found {len(solutions)} solutions meeting R² ≥ {self.config.R2_THRESHOLD:.4f}")
        print(f"{'='*70}\n")
        
        return solutions
    
    def save_solutions(self, solutions, filename):
        """
        Save solutions to CSV file.
        
        Args:
            solutions: List of ParameterSet objects
            filename: Output filename
        """
        if not solutions:
            print("No solutions to save.")
            return
        
        # Convert to DataFrame
        data = [sol.to_dict() for sol in solutions]
        df = pd.DataFrame(data)
        
        # Sort by R² total
        df = df.sort_values('r2_total', ascending=False)
        
        # Save to CSV
        df.to_csv(filename, index=False, float_format='%.8f')
        print(f"\nSolutions saved to: {filename}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Number of solutions: {len(solutions)}")
        print(f"  Best R² (total):     {df['r2_total'].max():.6f}")
        print(f"  Worst R² (total):    {df['r2_total'].min():.6f}")
        print(f"  Mean R² (total):     {df['r2_total'].mean():.6f}")
        print(f"  Std R² (total):      {df['r2_total'].std():.6f}")
        
        print("\n  Parameter ranges in solutions:")
        for param in ['J1xy', 'J1z', 'J3xy', 'J3z', 'D', 'E', 'F', 'G', 'K1', 'K2']:
            if param in df.columns:
                print(f"    {param:8s}: [{df[param].min():.4f}, {df[param].max():.4f}]")
    
    def plot_fit_comparison(self, params_set, filename="fit_comparison.png"):
        """
        Plot the theoretical dispersion vs experimental data for a parameter set.
        
        Args:
            params_set: ParameterSet object to plot
            filename: Output filename for the plot
        """
        print(f"\nGenerating fit comparison plot...")
        
        # Compute theoretical dispersion
        fitter = SpinWaveFitter(self.data_lower, self.data_upper, self.config)
        
        # Create parameter dictionary
        parameters = fitter.base_params.copy()
        parameters.update({
            'J1': params_set.J1xy,
            'J3': params_set.J3xy,
            'J1z': params_set.J1z,
            'J3z': params_set.J3z,
            'D': params_set.D,
            'E': params_set.E,
            'F': params_set.F,
            'G': params_set.G
        })
        
        # Compute dispersions
        energies_lower, energies_upper = fitter.compute_dispersion(
            parameters, self.config.MAGNETIC_FIELD)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot lower band
        ax1.scatter(range(len(self.data_lower[2])), self.data_lower[2], 
                   c='blue', marker='o', s=30, alpha=0.6, label='Experimental')
        ax1.plot(range(len(energies_lower)), energies_lower, 
                'r-', linewidth=2, label='Theoretical fit')
        ax1.set_xlabel('Data Point Index', fontsize=12)
        ax1.set_ylabel('Energy (meV)', fontsize=12)
        ax1.set_title(f'Lower Band (R² = {params_set.r2_lower:.4f})', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot upper band
        ax2.scatter(range(len(self.data_upper[2])), self.data_upper[2], 
                   c='blue', marker='o', s=30, alpha=0.6, label='Experimental')
        ax2.plot(range(len(energies_upper)), energies_upper, 
                'r-', linewidth=2, label='Theoretical fit')
        ax2.set_xlabel('Data Point Index', fontsize=12)
        ax2.set_ylabel('Energy (meV)', fontsize=12)
        ax2.set_title('Upper Band', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Overall title with parameters
        fig.suptitle(
            f'LSWT Fit Comparison (Overall R² = {params_set.r2_total:.4f})\n' +
            f'J1xy={params_set.J1xy:.3f}, D={params_set.D:.3f}, E={params_set.E:.3f}, ' +
            f'F={params_set.F:.3f}, G={params_set.G:.3f}, ' +
            f'K1={params_set.K1:.3f}, K2={params_set.K2:.3f}',
            fontsize=13, y=1.00)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
        
        return fig
    
    def plot_dispersion_path(self, params_set, filename="dispersion_path.png"):
        """
        Plot the dispersion along the high-symmetry path in the Brillouin zone.
        
        This shows the energy vs momentum path (more physical representation).
        
        Args:
            params_set: ParameterSet object to plot
            filename: Output filename for the plot
        """
        print(f"\nGenerating dispersion path plot...")
        
        # Compute theoretical dispersion
        fitter = SpinWaveFitter(self.data_lower, self.data_upper, self.config)
        
        parameters = fitter.base_params.copy()
        parameters.update({
            'J1': params_set.J1xy,
            'J3': params_set.J3xy,
            'J1z': params_set.J1z,
            'J3z': params_set.J3z,
            'D': params_set.D,
            'E': params_set.E,
            'F': params_set.F,
            'G': params_set.G
        })
        
        energies_lower, energies_upper = fitter.compute_dispersion(
            parameters, self.config.MAGNETIC_FIELD)
        
        # Create path coordinate (normalized 0 to 1)
        n_lower = len(self.data_lower[2])
        n_upper = len(self.data_upper[2])
        
        # For lower band: use normalized position
        path_lower = np.linspace(0, 1, n_lower)
        
        # Upper band offset to continue from lower
        path_upper = np.linspace(0, 1, n_upper)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot experimental data
        ax.scatter(path_lower, self.data_lower[2], 
                  c='blue', marker='o', s=40, alpha=0.6, 
                  label='Experimental (lower band)', zorder=5)
        ax.scatter(path_upper, self.data_upper[2], 
                  c='cyan', marker='s', s=40, alpha=0.6, 
                  label='Experimental (upper band)', zorder=5)
        
        # Plot theoretical fit
        ax.plot(path_lower, energies_lower, 
               'r-', linewidth=2.5, label='LSWT fit (lower)', zorder=4)
        ax.plot(path_upper, energies_upper, 
               'orange', linewidth=2.5, linestyle='--', 
               label='LSWT fit (upper)', zorder=4)
        
        # Labels and styling
        ax.set_xlabel('Path in Brillouin Zone (normalized)', fontsize=14)
        ax.set_ylabel('Energy (meV)', fontsize=14)
        ax.set_title(
            f'Spin Wave Dispersion - LSWT Fit (R² = {params_set.r2_total:.4f})\n' +
            f'J1xy={params_set.J1xy:.3f}, J1z={params_set.J1z:.3f}, ' +
            f'D={params_set.D:.3f}, E={params_set.E:.3f}, ' +
            f'F={params_set.F:.3f}, G={params_set.G:.3f}, ' +
            f'K1={params_set.K1:.3f}, K2={params_set.K2:.3f}',
            fontsize=12)
        
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add high-symmetry point markers (approximate positions)
        # This depends on your actual path - adjust as needed
        sym_points = {'Γ': 0.0, 'M': 0.15, 'K': 0.35, 'Γ\'': 0.55, 'M\'': 0.75}
        for label, pos in sym_points.items():
            if pos <= 1.0:
                ax.axvline(pos, color='gray', linestyle=':', alpha=0.5, linewidth=1)
                ax.text(pos, ax.get_ylim()[1]*0.98, label, 
                       ha='center', va='top', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Dispersion plot saved to: {filename}")
        plt.close()
        
        return fig
    
    def print_top_solutions(self, solutions, n=10):
        """Print the top n solutions."""
        print(f"\n{'='*70}")
        print(f"TOP {min(n, len(solutions))} SOLUTIONS")
        print(f"{'='*70}\n")
        
        for i, sol in enumerate(solutions[:n], 1):
            print(f"Solution {i}:")
            print(f"  R² (total): {sol.r2_total:.6f}, R² (lower): {sol.r2_lower:.6f}")
            print(f"  # Exchange interaction parameters")
            print(f"  J1xy = {sol.J1xy:.3f}")
            print(f"  J1z  = {sol.J1z:.3f}")
            print(f"  D    = {sol.D:.3f}")
            print(f"  E    = {sol.E:.3f}")
            print(f"  F    = {sol.F:.3f}")
            print(f"  G    = {sol.G:.3f}")
            print(f"  # Third nearest neighbor parameters")
            print(f"  J3xy = {sol.J3xy:.3f}")
            print(f"  J3z  = {sol.J3z:.3f}")
            print(f"  # Kitaev eigenvalues")
            print(f"  K1   = {sol.K1:.3f}")
            print(f"  K2   = {sol.K2:.3f}")
            print()


# =============================================================================
# CONFIGURATION FILE HANDLING
# =============================================================================

def load_config_file(filename):
    """
    Load parameters from YAML configuration file.
    
    Args:
        filename: Path to YAML config file
    
    Returns:
        Dictionary with parameters and search configuration
    """
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config_file(params, filename, search_config=None):
    """
    Save parameters to YAML configuration file.
    
    Args:
        params: ParameterSet object or dictionary
        filename: Output filename
        search_config: Optional search configuration dictionary
    """
    if isinstance(params, ParameterSet):
        params_dict = {
            'J1xy': float(params.J1xy),
            'J1z': float(params.J1z),
            'D': float(params.D),
            'E': float(params.E),
            'F': float(params.F),
            'G': float(params.G),
            'J3xy': float(params.J3xy),
            'J3z': float(params.J3z)
        }
    else:
        params_dict = params
    
    config = params_dict.copy()
    
    if search_config:
        config['search_config'] = search_config
    
    # Add comments through ordered structure
    output = "# BCAO LSWT Parameter Configuration File\n"
    output += "# Generated from local parameter search\n\n"
    output += "# Exchange interaction parameters\n"
    
    with open(filename, 'w') as f:
        f.write(output)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nConfiguration saved to: {filename}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_best_from_scan(filename, sign_combination='all_+ve'):
    """
    Load the best parameter set from a previous full scan.
    
    Args:
        filename: Path to .npy file from full scan
        sign_combination: Which sign combination was used
    
    Returns:
        ParameterSet object
    """
    print(f"\nLoading best parameters from: {filename}")
    
    data = np.load(filename, allow_pickle=True).item()
    kitaev = data['kitaev']
    err_all = data['err_all']
    
    # Find maximum R²
    max_idx = np.unravel_index(np.argmax(err_all[:, :, :, 0]), err_all[:, :, :, 0].shape)
    k_idx, i_idx, j_idx = max_idx
    
    max_r2 = err_all[k_idx, i_idx, j_idx, 0]
    
    print(f"  Best R² found: {max_r2:.6f}")
    print(f"  Array indices: k={k_idx}, i={i_idx}, j={j_idx}")
    
    # Reconstruct parameters
    # Need to know the grids used
    config = Config()
    b_M1_a_grid = np.linspace(config.B_M1_A_MIN, config.B_M1_A_MAX, config.N_M1_A_POINTS)
    b_M1_b_grid = np.linspace(config.B_M1_B_MIN, config.B_M1_B_MAX, config.N_M1_B_POINTS)
    
    b_M1_a = b_M1_a_grid[i_idx]
    b_M1_b = b_M1_b_grid[j_idx]
    
    # J1 is trickier - it's the k_idx'th point in a custom range
    # For now, we'll need to recompute or pass in separately
    print("\nWarning: J1 and E values need to be provided separately")
    print("         as they depend on the valid range computation.")
    
    return None  # Placeholder


def create_manual_parameter_set(J1xy, J1z, D, E, F, G, J3xy, J3z):
    """
    Create a parameter set manually from physical parameters.
    
    Args:
        J1xy: First-neighbor in-plane exchange
        J1z: First-neighbor anisotropic exchange
        D: Symmetric anisotropy
        E: Asymmetric anisotropy
        F: Off-diagonal Kitaev interaction
        G: Off-diagonal Kitaev interaction
        J3xy: Third-neighbor in-plane exchange
        J3z: Third-neighbor anisotropic exchange
    
    Returns:
        ParameterSet object
    """
    return ParameterSet(J1xy, J1z, D, E, F, G, J3xy, J3z)


def create_parameter_set_from_config(config):
    """
    Create a parameter set from a configuration dictionary.
    
    Args:
        config: Dictionary loaded from YAML file
    
    Returns:
        ParameterSet object
    """
    return ParameterSet(
        J1xy=config['J1xy'],
        J1z=config['J1z'],
        D=config['D'],
        E=config['E'],
        F=config['F'],
        G=config['G'],
        J3xy=config['J3xy'],
        J3z=config['J3z']
    )


def update_search_config_from_file(search_config_obj, config_dict):
    """
    Update LocalSearchConfig object from configuration dictionary.
    
    Args:
        search_config_obj: LocalSearchConfig object to update
        config_dict: Dictionary with search_config section from YAML
    """
    if 'search_config' not in config_dict:
        return
    
    sc = config_dict['search_config']
    
    if 'r2_threshold' in sc:
        search_config_obj.R2_THRESHOLD = sc['r2_threshold']
    if 'n_local_points' in sc:
        search_config_obj.N_LOCAL_POINTS = sc['n_local_points']
    if 'J1xy_range_fraction' in sc:
        search_config_obj.J1_RANGE_FRACTION = sc['J1xy_range_fraction']
    if 'E_range_fraction' in sc:
        search_config_obj.E_RANGE_FRACTION = sc['E_range_fraction']
    if 'J1xy_min_range' in sc:
        search_config_obj.J1_MIN_RANGE = sc['J1xy_min_range']
    if 'E_min_range' in sc:
        search_config_obj.E_MIN_RANGE = sc['E_min_range']
    if 'max_solutions' in sc:
        search_config_obj.MAX_SOLUTIONS = sc['max_solutions']


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution with config file support."""
    
    print("\n" + "="*70)
    print("LOCAL PARAMETER SEARCH")
    print("="*70 + "\n")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python local_parameter_search.py <config_file.yaml>")
        print("  python local_parameter_search.py <J1xy> <J1z> <D> <E> <F> <G> <J3xy> <J3z> [R2_threshold]")
        print("\nExamples:")
        print("  # Using config file:")
        print("  python local_parameter_search.py example_parameters.yaml")
        print("\n  # Using command line arguments:")
        print("  python local_parameter_search.py -6.772 -1.887 0.815 1.292 -0.091 0.627 1.823 -0.157")
        print("  python local_parameter_search.py -6.772 -1.887 0.815 1.292 -0.091 0.627 1.823 -0.157 0.92")
        print("\nArguments (if not using config file):")
        print("  J1xy:         First-neighbor in-plane exchange")
        print("  J1z:          First-neighbor anisotropic exchange")
        print("  D:            Symmetric anisotropy")
        print("  E:            Asymmetric anisotropy")
        print("  F:            Off-diagonal Kitaev interaction")
        print("  G:            Off-diagonal Kitaev interaction")
        print("  J3xy:         Third-neighbor in-plane exchange")
        print("  J3z:          Third-neighbor anisotropic exchange")
        print("  R2_threshold: Minimum R² to accept (default: 0.90)")
        sys.exit(1)
    
    # Initialize configuration
    config = LocalSearchConfig()
    test_all_signs = True
    
    # Check if first argument is a config file
    if sys.argv[1].endswith('.yaml') or sys.argv[1].endswith('.yml'):
        # Load from config file
        print(f"Loading parameters from config file: {sys.argv[1]}")
        config_dict = load_config_file(sys.argv[1])
        
        # Create parameter set
        center_params = create_parameter_set_from_config(config_dict)
        
        # Update search configuration if provided
        update_search_config_from_file(config, config_dict)
        
        if 'search_config' in config_dict and 'test_all_signs' in config_dict['search_config']:
            test_all_signs = config_dict['search_config']['test_all_signs']
        
        print("\nParameters loaded from config file:")
        print(center_params)
        
    else:
        # Parse from command line
        if len(sys.argv) < 9:
            print("Error: Not enough command line arguments")
            print("Need 8 parameters: J1xy J1z D E F G J3xy J3z")
            sys.exit(1)
        
        J1xy = float(sys.argv[1])
        J1z = float(sys.argv[2])
        D = float(sys.argv[3])
        E = float(sys.argv[4])
        F = float(sys.argv[5])
        G = float(sys.argv[6])
        J3xy = float(sys.argv[7])
        J3z = float(sys.argv[8])
        
        if len(sys.argv) > 9:
            config.R2_THRESHOLD = float(sys.argv[9])
        
        center_params = create_manual_parameter_set(
            J1xy, J1z, D, E, F, G, J3xy, J3z)
        
        print("\nParameters from command line:")
        print(center_params)
    
    print(f"\nSearch configuration:")
    print(f"  R² threshold:     {config.R2_THRESHOLD}")
    print(f"  Grid points:      {config.N_LOCAL_POINTS}")
    print(f"  Max solutions:    {config.MAX_SOLUTIONS}")
    print(f"  Test all signs:   {test_all_signs}")
    
    # Load data
    print("\nLoading experimental data...")
    loader = DataLoader()
    data_lower, data_upper = loader.load_and_process()
    
    # First, evaluate the center point
    print("\n" + "-"*70)
    print("Evaluating center point...")
    print("-"*70)
    fitter = SpinWaveFitter(data_lower, data_upper, config)
    kitaev, r2_total, r2_lower = fitter.fit_single_point(
        center_params.b_M1_a, center_params.c_M1_a, 
        center_params.b_M1_b, center_params.c_M1_b, 
        center_params.J1xy, center_params.E, 
        center_params.F_sign, center_params.G_sign)
    center_params.r2_total = r2_total
    center_params.r2_lower = r2_lower
    
    print(f"\nCenter point R²: {r2_total:.6f} (total), {r2_lower:.6f} (lower)")
    
    # Perform local search
    engine = LocalSearchEngine(data_lower, data_upper, config)
    solutions = engine.search_around_point(center_params, test_all_signs=test_all_signs)
    
    # Print top solutions
    engine.print_top_solutions(solutions, n=10)
    
    # Save all solutions
    output_file = f"local_search_solutions_R2_{config.R2_THRESHOLD:.2f}.csv"
    engine.save_solutions(solutions, output_file)
    
    # Save best solution as config file
    if solutions:
        best_params = solutions[0]
        best_config_file = f"best_solution_R2_{best_params.r2_total:.4f}.yaml"
        save_config_file(best_params, best_config_file)
        
        # Plot best fit comparison
        plot_filename = f"fit_comparison_R2_{best_params.r2_total:.4f}.png"
        dispersion_filename = f"dispersion_path_R2_{best_params.r2_total:.4f}.png"
        try:
            engine.plot_fit_comparison(best_params, plot_filename)
            engine.plot_dispersion_path(best_params, dispersion_filename)
        except Exception as e:
            print(f"\nWarning: Could not generate plots: {e}")
    
    print("\n" + "="*70)
    print("Local search complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
