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
    T_end: float = 0.001
    
    # Simulated annealing specific
    annealing_steps: int = 50000
    cooling_rate: float = 0.9
    T_zero: bool = True
    n_deterministics: int = 10000
    
    # Parallel tempering specific
    num_replicas: int = 20
    thermalization_sweeps: int = 100000
    measurement_sweeps: int = 100000
    swap_interval: int = 50
    
    # Common
    overrelaxation_rate: int = 10
    use_twist_boundary: bool = False
    gaussian_move: bool = False
    
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
            f"save_observables = {'true' if config.save_observables else 'false'}",
        ])
        
        # Write parameter file
        param_file = self.work_dir / f"params_{self._run_count}.param"
        with open(param_file, 'w') as f:
            f.write('\n'.join(lines))
        
        return str(param_file)
    
    def run_simulation(self, params: NormalizedParameters,
                       timeout: float = 300) -> Tuple[np.ndarray, np.ndarray, float]:
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
        try:
            result = subprocess.run(
                [str(self.solver_path), param_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.work_dir)
            )
            
            if result.returncode != 0:
                self._log(f"  Solver returned error code {result.returncode}")
                self._log(f"  stdout: {result.stdout[:500]}")
                self._log(f"  stderr: {result.stderr[:500]}")
                raise RuntimeError(f"Spin solver failed: {result.stderr[:200]}")
            
        except subprocess.TimeoutExpired:
            self._log(f"  Simulation timed out after {timeout}s")
            # Return None to indicate timeout instead of crashing
            return None, None, None
        
        # Parse output
        spins, positions, energy = self._parse_output(output_dir)
        
        self._log(f"  Completed: E={energy:.6f}, N_sites={len(spins)}")
        
        # Cleanup if requested
        if self.cleanup:
            try:
                shutil.rmtree(self.work_dir / f"output_{run_id}")
                os.remove(param_file)
            except:
                pass
        
        return spins, positions, energy
    
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
            available = list(output_dir.glob("*.txt"))
            raise FileNotFoundError(
                f"Spins file not found in {output_dir}. "
                f"Available files: {[f.name for f in available]}"
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


def create_spin_solver_simulation_func(
    L: int = 24,
    simulation_mode: str = "simulated_annealing",
    annealing_steps: int = 30000,
    n_deterministics: int = 5000,
    T_start: float = 5.0,
    T_end: float = 0.001,
    timeout: float = 240,
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
        timeout: Maximum time per simulation in seconds
        solver_path: Path to spin_solver executable
        verbose: Print verbose output
        cleanup: Clean up temporary files
        
    Returns:
        Function(NormalizedParameters) -> (spins, positions, energy)
    """
    config = SimulationConfig(
        lattice_size=(L, L, 1),
        simulation_mode=simulation_mode,
        annealing_steps=annealing_steps,
        n_deterministics=n_deterministics,
        T_start=T_start,
        T_end=T_end,
        cooling_rate=0.9,
        T_zero=True,
        overrelaxation_rate=10,
    )
    
    runner = SpinSolverRunner(
        solver_path=solver_path,
        config=config,
        cleanup=cleanup,
        verbose=verbose,
    )
    
    def simulate(params: NormalizedParameters) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run spin solver simulation."""
        return runner.run_simulation(params, timeout=timeout)
    
    # Attach runner to function for cleanup access
    simulate._runner = runner
    
    return simulate


def create_fast_simulation_func(
    L: int = 12,
    annealing_steps: int = 2000,
    n_deterministics: int = 500,
    **kwargs
):
    """
    Create a fast simulation function for initial screening.
    
    Uses smaller lattice and fewer steps for quick phase identification.
    Target: ~3-5 seconds per simulation for L=12.
    """
    return create_spin_solver_simulation_func(
        L=L,
        annealing_steps=annealing_steps,
        n_deterministics=n_deterministics,
        timeout=30,
        **kwargs
    )


def create_accurate_simulation_func(
    L: int = 24,
    annealing_steps: int = 20000,
    n_deterministics: int = 5000,
    **kwargs
):
    """
    Create an accurate simulation function for confirmation runs.
    
    Uses larger lattice and more steps for reliable classification.
    Target: ~60-90 seconds per simulation for L=24.
    """
    return create_spin_solver_simulation_func(
        L=L,
        annealing_steps=annealing_steps,
        n_deterministics=n_deterministics,
        timeout=300,
        **kwargs
    )


def create_screening_simulation_func(
    L: int = 8,
    annealing_steps: int = 1000,
    n_deterministics: int = 200,
    **kwargs
):
    """
    Create an ultra-fast simulation function for initial screening.
    
    Uses very small lattice for quick rough phase identification.
    Target: ~3 seconds per simulation.
    Good for: Initial exploration to identify promising regions.
    """
    return create_spin_solver_simulation_func(
        L=L,
        annealing_steps=annealing_steps,
        n_deterministics=n_deterministics,
        timeout=15,
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
        
        # Cleanup
        sim_func._runner.cleanup_all()
        
        print("\n✓ Spin solver test passed!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Spin solver not found: {e}")
        print("  Make sure the spin_solver is built in the build/ directory")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
