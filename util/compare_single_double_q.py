import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
from single_q_BCAO import SingleQ
from double_q_BCAO import DoubleQ
from datetime import datetime
import os

# filepath: /home/pc_linux/ClassicalSpin_Cpp/util/compare_single_double_q.py

class ParameterSpaceExplorer:
    """
    Explore parameter space to determine whether single-Q or double-Q ansatz
    provides a better ground state energy for BCAO honeycomb lattice.
    
    This class systematically varies exchange coupling parameters and compares
    the ground state energies obtained from single-Q and double-Q variational ansätze.
    """
    
    def __init__(self, L=6, B_field=np.array([0, 0, 0])):
        """
        Initialize the parameter space explorer
        
        Args:
            L: Lattice size (L x L unit cells)
            B_field: External magnetic field
        """
        self.L = L
        self.B_field = B_field
        self.results = []
        
    def scan_parameter_space(self, param_ranges, fixed_params=None, verbose=True):
        """
        Scan through parameter space and compare single-Q vs double-Q
        
        Args:
            param_ranges: Dictionary with parameter names and (min, max, num_points)
                         e.g., {'J1xy': (-10, -5, 6), 'J3xy': (0, 5, 6)}
            fixed_params: Dictionary with fixed parameter values
                         e.g., {'J1z': -1.2, 'D': 0.1, 'E': -0.1, 'F': 0, 'G': 0, 'J3z': -0.85}
            verbose: Print progress updates
        
        Returns:
            results: List of dictionaries containing scan results
        """
        # Default fixed parameters (BCAO standard)
        if fixed_params is None:
            fixed_params = {
                'J1z': -1.2,
                'D': 0.1,
                'E': -0.1,
                'F': 0.0,
                'G': 0.0,
                'J3z': -0.85
            }
        
        # Generate parameter grid
        param_names = list(param_ranges.keys())
        param_grids = []
        for param_name in param_names:
            min_val, max_val, num_points = param_ranges[param_name]
            param_grids.append(np.linspace(min_val, max_val, num_points))
        
        # Create meshgrid for all parameters
        meshgrids = np.meshgrid(*param_grids, indexing='ij')
        total_points = np.prod([len(grid) for grid in param_grids])
        
        if verbose:
            print("="*70)
            print("PARAMETER SPACE EXPLORATION: Single-Q vs Double-Q Ansatz")
            print("="*70)
            print(f"Lattice size: {self.L}x{self.L}")
            print(f"Varying parameters: {param_names}")
            print(f"Total parameter combinations: {total_points}")
            print(f"Fixed parameters: {fixed_params}")
            print("="*70)
        
        results = []
        count = 0
        
        # Iterate through all parameter combinations
        for indices in np.ndindex(*[len(grid) for grid in param_grids]):
            count += 1
            
            # Build parameter vector
            J_params = [fixed_params.get(key, 0.0) for key in 
                       ['J1xy', 'J1z', 'D', 'E', 'F', 'G', 'J3xy', 'J3z']]
            
            # Update with scanned parameters
            for i, param_name in enumerate(param_names):
                param_value = meshgrids[i][indices]
                if param_name == 'J1xy':
                    J_params[0] = param_value
                elif param_name == 'J1z':
                    J_params[1] = param_value
                elif param_name == 'D':
                    J_params[2] = param_value
                elif param_name == 'E':
                    J_params[3] = param_value
                elif param_name == 'F':
                    J_params[4] = param_value
                elif param_name == 'G':
                    J_params[5] = param_value
                elif param_name == 'J3xy':
                    J_params[6] = param_value
                elif param_name == 'J3z':
                    J_params[7] = param_value
            
            if verbose and count % max(1, total_points // 20) == 0:
                print(f"\nProgress: {count}/{total_points} ({100*count/total_points:.1f}%)")
                print(f"Current parameters: {dict(zip(param_names, [meshgrids[i][indices] for i in range(len(param_names))]))}")
            
            # Run single-Q optimization
            try:
                if verbose:
                    print("  Running single-Q optimization...")
                single_q_model = SingleQ(L=self.L, J=J_params, B_field=self.B_field)
                E_single_q = single_q_model.opt_energy
                Q_single = single_q_model.opt_params[0:2]
                phase_single = single_q_model.magnetic_order
                if verbose:
                    print(f"    Single-Q energy: {E_single_q:.8f}, Q=({Q_single[0]:.4f}, {Q_single[1]:.4f}), Phase: {phase_single}")
            except Exception as e:
                if verbose:
                    print(f"    Single-Q failed: {str(e)}")
                E_single_q = np.nan
                Q_single = [np.nan, np.nan]
                phase_single = "Failed"
            
            # Run double-Q optimization
            try:
                if verbose:
                    print("  Running double-Q optimization...")
                double_q_model = DoubleQ(L=self.L, J=J_params, B_field=self.B_field)
                E_double_q = double_q_model.opt_energy
                Q1_double = double_q_model.opt_params[0:2]
                Q2_double = double_q_model.opt_params[2:4]
                beta_Q1_A = double_q_model.opt_params[6]
                beta_Q1_B = double_q_model.opt_params[7]
                beta_Q2_A = double_q_model.opt_params[8]
                beta_Q2_B = double_q_model.opt_params[9]
                phase_double = double_q_model.magnetic_order
                if verbose:
                    print(f"    Double-Q energy: {E_double_q:.8f}")
                    print(f"    Q1=({Q1_double[0]:.4f}, {Q1_double[1]:.4f}), Q2=({Q2_double[0]:.4f}, {Q2_double[1]:.4f})")
                    print(f"    β_Q1_A={beta_Q1_A:.4f}, β_Q1_B={beta_Q1_B:.4f}, β_Q2_A={beta_Q2_A:.4f}, β_Q2_B={beta_Q2_B:.4f}")
                    print(f"    Phase: {phase_double}")
            except Exception as e:
                if verbose:
                    print(f"    Double-Q failed: {str(e)}")
                E_double_q = np.nan
                Q1_double = [np.nan, np.nan]
                Q2_double = [np.nan, np.nan]
                beta_Q1_A = np.nan
                beta_Q1_B = np.nan
                beta_Q2_A = np.nan
                beta_Q2_B = np.nan
                phase_double = "Failed"
            
            # Determine which ansatz is better
            energy_diff = E_single_q - E_double_q
            if not np.isnan(energy_diff):
                if energy_diff > 1e-6:
                    preferred = "Double-Q"
                elif energy_diff < -1e-6:
                    preferred = "Single-Q"
                else:
                    preferred = "Equivalent"
            else:
                preferred = "Unknown"
            
            if verbose:
                print(f"  Energy difference (Single - Double): {energy_diff:.8f}")
                print(f"  Preferred ansatz: {preferred}")
            
            # Store results
            result = {
                'parameters': dict(zip(['J1xy', 'J1z', 'D', 'E', 'F', 'G', 'J3xy', 'J3z'], J_params)),
                'scanned_params': dict(zip(param_names, [meshgrids[i][indices] for i in range(len(param_names))])),
                'E_single_q': float(E_single_q) if not np.isnan(E_single_q) else None,
                'E_double_q': float(E_double_q) if not np.isnan(E_double_q) else None,
                'energy_diff': float(energy_diff) if not np.isnan(energy_diff) else None,
                'Q_single': [float(q) for q in Q_single] if not any(np.isnan(Q_single)) else None,
                'Q1_double': [float(q) for q in Q1_double] if not any(np.isnan(Q1_double)) else None,
                'Q2_double': [float(q) for q in Q2_double] if not any(np.isnan(Q2_double)) else None,
                'beta_Q1_A': float(beta_Q1_A) if not np.isnan(beta_Q1_A) else None,
                'beta_Q1_B': float(beta_Q1_B) if not np.isnan(beta_Q1_B) else None,
                'beta_Q2_A': float(beta_Q2_A) if not np.isnan(beta_Q2_A) else None,
                'beta_Q2_B': float(beta_Q2_B) if not np.isnan(beta_Q2_B) else None,
                'phase_single': phase_single,
                'phase_double': phase_double,
                'preferred': preferred
            }
            results.append(result)
        
        self.results = results
        
        if verbose:
            print("\n" + "="*70)
            print("SCAN COMPLETE")
            print("="*70)
            self._print_summary()
        
        return results
    
    def _print_summary(self):
        """Print a summary of the scan results"""
        if not self.results:
            print("No results to summarize")
            return
        
        total = len(self.results)
        prefer_single = sum(1 for r in self.results if r['preferred'] == 'Single-Q')
        prefer_double = sum(1 for r in self.results if r['preferred'] == 'Double-Q')
        equivalent = sum(1 for r in self.results if r['preferred'] == 'Equivalent')
        unknown = sum(1 for r in self.results if r['preferred'] == 'Unknown')
        
        print(f"\nTotal parameter points: {total}")
        print(f"Single-Q preferred: {prefer_single} ({100*prefer_single/total:.1f}%)")
        print(f"Double-Q preferred: {prefer_double} ({100*prefer_double/total:.1f}%)")
        print(f"Equivalent: {equivalent} ({100*equivalent/total:.1f}%)")
        print(f"Unknown/Failed: {unknown} ({100*unknown/total:.1f}%)")
        
        # Find largest energy gain from double-Q
        valid_results = [r for r in self.results if r['energy_diff'] is not None]
        if valid_results:
            max_gain_idx = np.argmin([r['energy_diff'] for r in valid_results])
            max_gain_result = valid_results[max_gain_idx]
            print(f"\nLargest energy gain from double-Q:")
            print(f"  Parameters: {max_gain_result['scanned_params']}")
            print(f"  Energy gain: {-max_gain_result['energy_diff']:.8f}")
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"parameter_scan_results_{timestamp}.json"
        
        output_data = {
            'metadata': {
                'lattice_size': self.L,
                'B_field': self.B_field.tolist(),
                'timestamp': datetime.now().isoformat(),
                'num_points': len(self.results)
            },
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        return filename
    
    def plot_phase_diagram(self, param_names=None, save_file=None):
        """
        Plot phase diagram showing regions where single-Q or double-Q is preferred
        
        Args:
            param_names: List of parameter names to plot (must be 2 for 2D plot)
            save_file: Filename to save the plot
        """
        if not self.results:
            print("No results to plot")
            return
        
        # If param_names not specified, try to infer from results
        if param_names is None:
            # Get scanned parameters from first result
            scanned = list(self.results[0]['scanned_params'].keys())
            if len(scanned) == 2:
                param_names = scanned
            else:
                print("Please specify which 2 parameters to plot")
                return
        
        if len(param_names) != 2:
            print("Can only plot 2D phase diagrams (2 parameters)")
            return
        
        # Extract data
        param1_vals = []
        param2_vals = []
        preference_vals = []
        energy_diff_vals = []
        
        for result in self.results:
            if result['preferred'] != 'Unknown':
                p1 = result['scanned_params'].get(param_names[0])
                p2 = result['scanned_params'].get(param_names[1])
                if p1 is not None and p2 is not None:
                    param1_vals.append(p1)
                    param2_vals.append(p2)
                    
                    # Map preference to number
                    if result['preferred'] == 'Single-Q':
                        preference_vals.append(0)
                    elif result['preferred'] == 'Double-Q':
                        preference_vals.append(2)
                    else:  # Equivalent
                        preference_vals.append(1)
                    
                    energy_diff_vals.append(result['energy_diff'] if result['energy_diff'] is not None else 0)
        
        # Reshape for contour plot
        unique_p1 = np.unique(param1_vals)
        unique_p2 = np.unique(param2_vals)
        
        preference_grid = np.full((len(unique_p2), len(unique_p1)), np.nan)
        energy_diff_grid = np.full((len(unique_p2), len(unique_p1)), np.nan)
        
        for i, (p1, p2, pref, ediff) in enumerate(zip(param1_vals, param2_vals, preference_vals, energy_diff_vals)):
            i1 = np.where(unique_p1 == p1)[0][0]
            i2 = np.where(unique_p2 == p2)[0][0]
            preference_grid[i2, i1] = pref
            energy_diff_grid[i2, i1] = ediff
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Phase diagram
        ax1 = axes[0]
        cmap = ListedColormap(['#3498db', '#95a5a6', '#e74c3c'])  # Blue, Gray, Red
        im1 = ax1.contourf(unique_p1, unique_p2, preference_grid, 
                          levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.8)
        ax1.contour(unique_p1, unique_p2, preference_grid, 
                   levels=[-0.5, 0.5, 1.5, 2.5], colors='black', linewidths=0.5)
        
        # Add colorbar with labels
        cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2])
        cbar1.ax.set_yticklabels(['Single-Q', 'Equivalent', 'Double-Q'])
        
        ax1.set_xlabel(param_names[0], fontsize=12)
        ax1.set_ylabel(param_names[1], fontsize=12)
        ax1.set_title('Phase Diagram: Preferred Ansatz', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Energy difference
        ax2 = axes[1]
        im2 = ax2.contourf(unique_p1, unique_p2, energy_diff_grid, 
                          levels=20, cmap='RdBu_r')
        ax2.contour(unique_p1, unique_p2, energy_diff_grid, 
                   levels=[0], colors='black', linewidths=2)
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('E(Single-Q) - E(Double-Q)', fontsize=11)
        
        ax2.set_xlabel(param_names[0], fontsize=12)
        ax2.set_ylabel(param_names[1], fontsize=12)
        ax2.set_title('Energy Difference', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Phase diagram saved to: {save_file}")
        
        plt.show()
        
        return fig, axes


def run_example_scan():
    """Run an example parameter space scan"""
    # Initialize explorer with smaller lattice for faster computation
    explorer = ParameterSpaceExplorer(L=6)
    
    # Define parameter ranges to scan
    # Scanning J1xy and J3xy while keeping others fixed
    param_ranges = {
        'J1xy': (-10.0, -5.0, 6),  # (min, max, num_points)
        'J3xy': (0.5, 4.0, 6)
    }
    
    # Fixed parameters
    fixed_params = {
        'J1z': -1.2,
        'D': 0.1,
        'E': -0.1,
        'F': 0.0,
        'G': 0.0,
        'J3z': -0.85
    }
    
    # Run scan
    results = explorer.scan_parameter_space(param_ranges, fixed_params, verbose=True)
    
    # Save results
    explorer.save_results()
    
    # Plot phase diagram
    explorer.plot_phase_diagram(param_names=['J1xy', 'J3xy'], 
                               save_file='phase_diagram_J1xy_vs_J3xy.png')
    
    return explorer


if __name__ == "__main__":
    print("Starting parameter space exploration...")
    print("This will compare single-Q and double-Q ansätze across parameter space")
    print()
    
    # Run example scan
    explorer = run_example_scan()
    
    print("\n" + "="*70)
    print("EXPLORATION COMPLETE")
    print("="*70)
    print("Results have been saved and phase diagram has been generated.")
