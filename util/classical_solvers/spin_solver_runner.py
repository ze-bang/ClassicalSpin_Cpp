"""
Spin Solver Runner - Interface to C++ Monte Carlo Solver.

This module provides functions to run the C++ spin_solver executable
and parse its output for use in the active learning framework.

The spin_solver uses simulated annealing or parallel tempering to find
the ground state spin configuration without ansatz constraints.
"""

import numpy as np
import subprocess
import tempfile
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import time

# Handle imports when running from different directories
try:
    from feature_extractor import NormalizedParameters
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_extractor import NormalizedParameters


@dataclass
class SimulationConfig:
    """Configuration for running the spin solver."""
    # Lattice
    lattice_size: Tuple[int, int, int] = (12, 12, 1)
    system: str = "honeycomb_bcao"
    
    # Simulation mode
    simulation_mode: str = "simulated_annealing"  # or "parallel_tempering"
    
    # Temperature schedule
    T_start: float = 5.0
    T_end: float = 0.01
    
    # Simulated annealing specific
    annealing_steps: int = 5000  # Fast default for testing
    cooling_rate: float = 0.95  # Slower cooling for better equilibration
    T_zero: bool = True
    n_deterministics: int = 2000  # Modest zero-T sweeps
    
    # Parallel tempering specific
    num_replicas: int = 20
    thermalization_sweeps: int = 20000
    measurement_sweeps: int = 10000
    swap_interval: int = 50
    
    # Common
    overrelaxation_rate: int = 10
    use_twist_boundary: bool = False
    gaussian_move: bool = False
    
    # Diagnostics
    save_diagnostics: bool = False  # Save T, E, acceptance rate vs step
    
    # Field (default zero)
    field_strength: float = 0.0
    field_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    
    # Output
    save_observables: bool = False


class SpinSolverRunner:
    """
    Interface to the C++ spin_solver executable.
    
    Creates parameter files, runs simulations, and parses output.
    """
    
    def __init__(self, 
                 solver_path: str = None,
                 work_dir: str = None,
                 config: SimulationConfig = None,
                 cleanup: bool = True,
                 verbose: bool = False):
        """
        Initialize the spin solver runner.
        
        Args:
            solver_path: Path to spin_solver executable. Auto-detected if None.
            work_dir: Working directory for temporary files. Uses temp dir if None.
            config: Simulation configuration. Uses defaults if None.
            cleanup: Whether to clean up temporary files after simulation.
            verbose: Print verbose output.
        """
        # Find solver executable
        if solver_path is None:
            solver_path = self._find_solver()
        self.solver_path = Path(solver_path)
        
        if not self.solver_path.exists():
            raise FileNotFoundError(f"Spin solver not found at {self.solver_path}")
        
        if not os.access(self.solver_path, os.X_OK):
            raise PermissionError(f"Spin solver is not executable: {self.solver_path}")
        
        # Work directory (unique per instance for parallel safety)
        if work_dir is None:
            # Create unique temp directory with process ID for multiprocessing safety
            prefix = f"spin_solver_pid{os.getpid()}_"
            self.work_dir = Path(tempfile.mkdtemp(prefix=prefix))
            self._temp_work_dir = True
        else:
            self.work_dir = Path(work_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self._temp_work_dir = False
        
        self.config = config or SimulationConfig()
        self.cleanup = cleanup
        self.verbose = verbose
        
        self._run_count = 0
    
    def _find_solver(self) -> str:
        """Find the spin_solver executable."""
        # Try common locations
        candidates = [
            Path(__file__).parent.parent.parent / "build" / "spin_solver",
            Path.cwd() / "build" / "spin_solver",
            Path.home() / "ClassicalSpin_Cpp" / "build" / "spin_solver",
        ]
        
        for path in candidates:
            if path.exists():
                return str(path)
        
        # Try which
        try:
            result = subprocess.run(["which", "spin_solver"], 
                                    capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        raise FileNotFoundError(
            "Could not find spin_solver executable. "
            "Please specify solver_path or build the solver."
        )
    
    def _log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(f"[SpinSolver] {msg}")
    
    def create_param_file(self, params: NormalizedParameters,
                          output_dir: str) -> str:
        """
        Create a parameter file for the spin solver.
        
        Args:
            params: Normalized BCAO parameters
            output_dir: Directory for simulation output
            
        Returns:
            Path to the created parameter file
        """
        # Use normalized parameters directly
        J1xy = -1
        J1z = params.J1z_norm
        D = params.D_norm
        E = params.E_norm
        F = params.F_norm
        G = params.G_norm
        J3xy = params.J3xy_norm
        J3z = params.J3z_norm
        
        config = self.config
        L = config.lattice_size
        
        lines = [
            "# Auto-generated parameter file for active learning",
            f"# Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "# System",
            f"system = {config.system}",
            f"simulation_mode = {config.simulation_mode}",
            "",
            "# Lattice Size",
            f"lattice_size = {L[0]},{L[1]},{L[2]}",
            "",
            "# Exchange Interaction Parameters (BCAO)",
            f"J1xy = {J1xy:.6f}",
            f"J1z = {J1z:.6f}",
            f"D = {D:.6f}",
            f"E = {E:.6f}",
            f"F = {F:.6f}",
            f"G = {G:.6f}",
            f"J3xy = {J3xy:.6f}",
            f"J3z = {J3z:.6f}",
            "",
            "# Magnetic Field",
            f"field_strength = {config.field_strength:.6f}",
            f"field_direction = {config.field_direction[0]},{config.field_direction[1]},{config.field_direction[2]}",
            "",
        ]
        
        if config.simulation_mode == "simulated_annealing":
            lines.extend([
                "# Simulated Annealing",
                f"T_start = {config.T_start}",
                f"T_end = {config.T_end}",
                f"annealing_steps = {config.annealing_steps}",
                f"overrelaxation_rate = {config.overrelaxation_rate}",
                f"use_twist_boundary = {'true' if config.use_twist_boundary else 'false'}",
                f"gaussian_move = {'true' if config.gaussian_move else 'false'}",
                f"cooling_rate = {config.cooling_rate}",
                f"T_zero = {'true' if config.T_zero else 'true'}",
                f"n_deterministics = {config.n_deterministics}",
            ])
        else:  # parallel_tempering
            lines.extend([
                "# Parallel Tempering",
                f"num_replicas = {config.num_replicas}",
                f"T_start = {config.T_start}",
                f"T_end = {config.T_end}",
                f"thermalization_sweeps = {config.thermalization_sweeps}",
                f"measurement_sweeps = {config.measurement_sweeps}",
                f"overrelaxation_rate = {config.overrelaxation_rate}",
                f"swap_interval = {config.swap_interval}",
                f"use_twist_boundary = {'true' if config.use_twist_boundary else 'false'}",
            ])
        
        lines.extend([
            "",
            "# Output",
            f"output_dir = {output_dir}",
            f"save_observables = 'false'",
        ])
        
        # Write parameter file
        param_file = self.work_dir / f"params_{self._run_count}.param"
        with open(param_file, 'w') as f:
            f.write('\n'.join(lines))
        
        return str(param_file)
    
    def run_simulation(self, params: NormalizedParameters,
                       timeout: float = 600) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run a simulation with the given parameters.
        
        Args:
            params: Normalized BCAO parameters
            timeout: Maximum time in seconds for the simulation
            
        Returns:
            (spins, positions, energy_per_site)
        """
        self._run_count += 1
        run_id = self._run_count
        
        # Create output directory
        output_dir = self.work_dir / f"output_{run_id}" / "sample_0"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create parameter file
        param_file = self.create_param_file(
            params, 
            str(self.work_dir / f"output_{run_id}")
        )
        
        self._log(f"Running simulation {run_id} with params: {params}")
        self._log(f"  Parameter file: {param_file}")
        
        # Run solver
        start_time = time.time()
        try:
            result = subprocess.run(
                [str(self.solver_path), param_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.work_dir)
            )
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                error_msg = (
                    f"Spin solver failed after {elapsed:.2f}s\n"
                    f"  Return code: {result.returncode}\n"
                    f"  Command: {self.solver_path} {param_file}\n"
                    f"  Working dir: {self.work_dir}\n"
                    f"  stdout: {result.stdout[:1000]}\n"
                    f"  stderr: {result.stderr[:1000]}"
                )
                print(error_msg)  # Always print errors, not just when verbose
                raise RuntimeError(f"Spin solver failed (exit {result.returncode}): {result.stderr[:200]}")
            
            # Parse diagnostics from stdout
            diagnostics = self._parse_diagnostics(result.stdout)
            self._log(f"  Solver completed in {elapsed:.2f}s")
            
            # Check if run was suspiciously fast
            if elapsed < 1.0:
                print(f"WARNING: Simulation completed very fast ({elapsed:.2f}s). Output may be invalid.")
                print(f"  stdout: {result.stdout[:500]}")
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            error_msg = (
                f"Simulation timed out after {elapsed:.2f}s (limit: {timeout}s)\n"
                f"  Command: {self.solver_path} {param_file}\n"
                f"  Working dir: {self.work_dir}\n"
                f"  This may indicate the solver is stuck or the timeout is too short."
            )
            print(error_msg)  # Always print timeouts
            raise TimeoutError(f"Spin solver timed out after {timeout}s")
        
        # Parse output
        try:
            spins, positions, energy = self._parse_output(output_dir)
        except Exception as e:
            print(f"ERROR parsing output from {output_dir}: {e}")
            print(f"  Available files: {list(output_dir.glob('**/*'))}")
            raise
        
        self._log(f"  Completed: E={energy:.6f}, N_sites={len(spins)}, time={elapsed:.2f}s")
        
        # Log diagnostics if verbose
        if self.verbose and diagnostics:
            self._log_diagnostics(diagnostics)
        
        # Cleanup if requested
        if self.cleanup:
            try:
                shutil.rmtree(self.work_dir / f"output_{run_id}")
                os.remove(param_file)
            except:
                pass
        
        return spins, positions, energy
    
    def _parse_diagnostics(self, stdout: str) -> Dict[str, Any]:
        """
        Parse diagnostic information from solver stdout.
        
        Looks for:
        - Temperature vs energy
        - Acceptance rates
        - Convergence information
        
        Args:
            stdout: Standard output from spin_solver
            
        Returns:
            Dictionary with diagnostic data
        """
        diagnostics = {
            'temperatures': [],
            'energies': [],
            'acceptance_rates': [],
            'num_steps': [],
        }
        
        lines = stdout.split('\n')
        
        for line in lines:
            # Parse temperature and energy lines
            # Example: "T = 2.500, E/N = -0.7234, acceptance = 0.45"
            if 'T =' in line and 'E/N' in line:
                try:
                    parts = line.split(',')
                    temp = float(parts[0].split('=')[1].strip())
                    energy = float(parts[1].split('=')[1].strip())
                    
                    diagnostics['temperatures'].append(temp)
                    diagnostics['energies'].append(energy)
                    
                    # Look for acceptance rate if present
                    if 'acceptance' in line:
                        acc = float(parts[2].split('=')[1].strip())
                        diagnostics['acceptance_rates'].append(acc)
                    
                except (IndexError, ValueError):
                    continue
            
            # Parse step counts
            elif 'Step' in line or 'sweep' in line:
                try:
                    # Extract step number
                    import re
                    match = re.search(r'(\d+)', line)
                    if match:
                        diagnostics['num_steps'].append(int(match.group(1)))
                except:
                    continue
        
        # Convert to numpy arrays for easier analysis
        for key in ['temperatures', 'energies', 'acceptance_rates']:
            if diagnostics[key]:
                diagnostics[key] = np.array(diagnostics[key])
        
        return diagnostics
    
    def _log_diagnostics(self, diagnostics: Dict[str, Any]):
        """
        Log diagnostic information in a readable format.
        
        Args:
            diagnostics: Dictionary from _parse_diagnostics
        """
        self._log("\n  === Simulation Diagnostics ===")
        
        temps = diagnostics.get('temperatures', [])
        energies = diagnostics.get('energies', [])
        acc_rates = diagnostics.get('acceptance_rates', [])
        
        if len(temps) > 0:
            self._log(f"  Temperature range: {temps.min():.4f} → {temps.max():.4f}")
            self._log(f"  Energy range: {energies.min():.6f} → {energies.max():.6f}")
            self._log(f"  Energy change: Δ = {abs(energies[-1] - energies[0]):.6f}")
            
            if len(acc_rates) > 0:
                self._log(f"  Acceptance rate: {acc_rates.mean():.3f} ± {acc_rates.std():.3f}")
                self._log(f"  Acceptance range: [{acc_rates.min():.3f}, {acc_rates.max():.3f}]")
                
                # Check for common issues
                if acc_rates.mean() < 0.1:
                    self._log("  ⚠ WARNING: Very low acceptance rate - system may not be equilibrating")
                elif acc_rates.mean() > 0.9:
                    self._log("  ⚠ WARNING: Very high acceptance rate - moves may be too small")
                
                # Check if energy is still changing significantly at low T
                if len(energies) > 5:
                    final_energies = energies[-5:]
                    energy_std = final_energies.std()
                    if energy_std > 0.01:
                        self._log(f"  ⚠ WARNING: Energy still fluctuating at low T (σ={energy_std:.4f})")
                        self._log("     → Consider more annealing steps or slower cooling")
        
        self._log("  ==============================\n")
    
    def _parse_output(self, output_dir: Path) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Parse the simulation output files.
        
        Args:
            output_dir: Directory containing output files
            
        Returns:
            (spins, positions, energy_per_site)
        """
        # Check for sample_0 subdirectory
        sample_dir = output_dir / "sample_0"
        if sample_dir.exists():
            output_dir = sample_dir
        
        # Read spins - try various filename patterns
        spins_file = None
        candidates = [
            output_dir / "spins.txt",
            output_dir / "spins_T=0.txt",  # T=0 output
        ]
        # Also check for temperature-specific files
        for f in output_dir.glob("spins_T=*.txt"):
            candidates.append(f)
        
        for candidate in candidates:
            if candidate.exists():
                spins_file = candidate
                break
        
        if spins_file is None:
            available = list(output_dir.glob("**/*.txt"))
            all_files = list(output_dir.glob("**/*"))
            raise FileNotFoundError(
                f"Spins file not found in {output_dir}\n"
                f"  Checked: {[str(c.relative_to(output_dir.parent)) for c in candidates if c.exists()]}\n"
                f"  Available .txt files: {[str(f.relative_to(output_dir.parent)) for f in available]}\n"
                f"  All files: {[str(f.relative_to(output_dir.parent)) for f in all_files[:20]]}"
            )
        
        self._log(f"  Reading spins from: {spins_file.name}")
        spins = np.loadtxt(spins_file)
        if spins.ndim == 1:
            spins = spins.reshape(-1, 3)
        
        # Read positions
        positions_file = output_dir / "positions.txt"
        if positions_file.exists():
            positions = np.loadtxt(positions_file)
            if positions.ndim == 1:
                positions = positions.reshape(-1, 3)
            # Take only x,y for 2D honeycomb
            positions = positions[:, :2]
        else:
            # Generate default honeycomb positions
            N = len(spins)
            L = int(np.sqrt(N / 2))
            positions = self._generate_honeycomb_positions(L)
        
        # Read energy
        energy_file = output_dir / "final_energy.txt"
        energy = 0.0
        if energy_file.exists():
            with open(energy_file, 'r') as f:
                for line in f:
                    line_stripped = line.strip()
                    # Try parsing as just a number first
                    try:
                        energy = float(line_stripped)
                        break
                    except ValueError:
                        pass
                    # Otherwise look for labeled format
                    if "Energy Density" in line or "Energy" in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            energy = float(parts[1].strip())
                            break
        
        return spins, positions, energy
    
    def _generate_honeycomb_positions(self, L: int) -> np.ndarray:
        """Generate honeycomb lattice positions."""
        N = 2 * L * L
        positions = np.zeros((N, 2))
        
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.5, np.sqrt(3)/2])
        delta_B = np.array([0, 1/np.sqrt(3)])
        
        for i in range(L):
            for j in range(L):
                cell_pos = i * a1 + j * a2
                site_a = 2 * (i * L + j)
                positions[site_a] = cell_pos
                positions[site_a + 1] = cell_pos + delta_B
        
        return positions
    
    def cleanup_all(self):
        """Clean up all temporary files."""
        if self._temp_work_dir and self.work_dir.exists():
            shutil.rmtree(self.work_dir)
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_temp_work_dir') and self._temp_work_dir:
            try:
                self.cleanup_all()
            except:
                pass


class _PicklableSimulation:
    """
    A picklable simulation callable for multiprocessing.
    
    This wraps the SpinSolverRunner configuration so it can be passed
    to worker processes in parallel execution.
    """
    def __init__(self, config_dict: Dict[str, Any], solver_path: str = None, 
                 cleanup: bool = True, verbose: bool = False):
        """
        Initialize with configuration dictionary.
        
        Args:
            config_dict: Dictionary of SimulationConfig parameters
            solver_path: Path to solver executable
            cleanup: Whether to cleanup temp files
            verbose: Verbose output
        """
        self.config_dict = config_dict
        self.solver_path = solver_path
        self.cleanup = cleanup
        self.verbose = verbose
        self.timeout = config_dict.get('timeout', None)  # None = no timeout
        self._runner = None
    
    def _get_runner(self):
        """Lazy initialization of runner in worker process."""
        if self._runner is None:
            # Create config from dict
            config = SimulationConfig(**{k: v for k, v in self.config_dict.items() 
                                        if k != 'timeout'})
            
            # Create runner in this process
            self._runner = SpinSolverRunner(
                solver_path=self.solver_path,
                config=config,
                cleanup=self.cleanup,
                verbose=self.verbose
            )
        return self._runner
    
    def __call__(self, params: NormalizedParameters) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run simulation."""
        runner = self._get_runner()
        return runner.run_simulation(params, timeout=self.timeout)
    
    def __getstate__(self):
        """Control pickling - don't pickle the runner."""
        state = self.__dict__.copy()
        state['_runner'] = None
        return state
    
    def __setstate__(self, state):
        """Control unpickling."""
        self.__dict__.update(state)
        self._runner = None


def create_spin_solver_simulation_func(
    L: int = 24,
    simulation_mode: str = "simulated_annealing",
    annealing_steps: int = 50000,
    n_deterministics: int = 5000,
    T_start: float = 5.0,
    T_end: float = 0.01,
    timeout: float = None,  # None = no timeout
    solver_path: str = None,
    verbose: bool = False,
    cleanup: bool = True
):
    """
    Create a simulation function that uses the C++ spin solver.
    
    This function returns a callable that takes NormalizedParameters
    and returns (spins, positions, energy).
    
    Args:
        L: Lattice size (L x L unit cells)
        simulation_mode: 'simulated_annealing' or 'parallel_tempering'
        annealing_steps: Number of annealing steps
        n_deterministics: Number of zero-temperature optimization steps
        T_start: Starting temperature
        T_end: Ending temperature  
        timeout: Maximum time per simulation in seconds (None = no timeout)
        solver_path: Path to spin_solver executable
        verbose: Print verbose output
        cleanup: Clean up temporary files
        
    Returns:
        Picklable function(NormalizedParameters) -> (spins, positions, energy)
    """
    # Convert config to dictionary for pickling
    config_dict = {
        'lattice_size': (L, L, 1),
        'simulation_mode': simulation_mode,
        'annealing_steps': annealing_steps,
        'n_deterministics': n_deterministics,
        'T_start': T_start,
        'T_end': T_end,
        'cooling_rate': 0.9,
        'T_zero': True,
        'overrelaxation_rate': 2,
        'timeout': timeout,
    }
    
    # Create picklable simulation function
    simulate = _PicklableSimulation(
        config_dict=config_dict,
        solver_path=solver_path,
        cleanup=cleanup,
        verbose=verbose
    )
    
    return simulate


def create_fast_simulation_func(
    L: int = 12,
    annealing_steps: int = 5000,
    n_deterministics: int = 2000,
    **kwargs
):
    """
    Create a fast simulation function for initial screening.
    
    Uses smaller lattice with minimal steps for quick phase identification.
    Target: ~5 seconds per simulation for L=12.
    """
    return create_spin_solver_simulation_func(
        L=L,
        annealing_steps=annealing_steps,
        n_deterministics=n_deterministics,
        timeout=None,  # No timeout
        **kwargs
    )


def create_accurate_simulation_func(
    L: int = 16,
    annealing_steps: int = 20000,
    n_deterministics: int = 5000,
    **kwargs
):
    """
    Create an accurate simulation function for confirmation runs.
    
    Uses larger lattice and more steps for reliable classification.
    Target: ~15-30 seconds per simulation for L=16.
    """
    return create_spin_solver_simulation_func(
        L=L,
        annealing_steps=annealing_steps,
        n_deterministics=n_deterministics,
        timeout=None,  # No timeout
        **kwargs
    )


def create_screening_simulation_func(
    L: int = 8,
    annealing_steps: int = 3000,
    n_deterministics: int = 1000,
    **kwargs
):
    """
    Create a screening simulation function for initial exploration.
    
    Uses minimal settings for very fast phase identification.
    Target: ~1-2 seconds per simulation.
    Good for: Rapid exploration to identify promising regions.
    """
    return create_spin_solver_simulation_func(
        L=L,
        annealing_steps=annealing_steps,
        n_deterministics=n_deterministics,
        timeout=None,  # No timeout
        **kwargs
    )


if __name__ == "__main__":
    # Test the spin solver runner
    print("=" * 60)
    print("SPIN SOLVER RUNNER TEST")
    print("=" * 60)
    
    # Create a test simulation function
    try:
        sim_func = create_fast_simulation_func(L=12, verbose=True)
        
        # Test with known parameter set
        from feature_extractor import KNOWN_SEEDS
        params = KNOWN_SEEDS['fitting_param_4']
        
        print(f"\nRunning test simulation with: {params}")
        print(f"Raw J: {params.to_raw()}")
        
        spins, positions, energy = sim_func(params)
        
        print(f"\nResults:")
        print(f"  N_sites: {len(spins)}")
        print(f"  Energy per site: {energy:.6f}")
        print(f"  Spin shape: {spins.shape}")
        print(f"  Total magnetization: {np.mean(spins, axis=0)}")
        
        print("\n✓ Spin solver test passed!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Spin solver not found: {e}")
        print("  Make sure the spin_solver is built in the build/ directory")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
