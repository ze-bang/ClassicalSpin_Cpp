# Example Configuration Files

This directory contains example configuration files for the unified simulation framework, organized by physical system. These examples replicate the functionality of the legacy `run_scripts` from `/legacy/run_scripts/`.

## Directory Structure

```
example_configs/
├── BCAO/           # Ba₃CoSb₂O₉ honeycomb configurations
├── Kitaev/         # Kitaev honeycomb model configurations
├── Pyrochlore/     # Pyrochlore lattice configurations
├── TmFeO3/         # TmFeO3 mixed lattice configurations
└── param_sweeps/   # N-dimensional parameter sweep examples
```

## System Coverage

### BCAO (Ba₃CoSb₂O₉ Honeycomb Lattice)

The unified simulation covers all BCAO functionality from legacy run_scripts:

| Legacy Script | Example Config | Simulation Mode | Description |
|---------------|----------------|-----------------|-------------|
| `molecular_dynamic_BCAO_emily.cpp` | `BCAO/md_emily.param` | `molecular_dynamics` | MD with Emily model parameters |
| `simulated_annealing_BCAO_emily.cpp` | `BCAO/sa_emily.param` | `simulated_annealing` | SA with Emily model |
| `parallel_tempering_BCAO_emily.cpp` | `BCAO/pt_emily.param` | `parallel_tempering` | PT with Emily model |
| `molecular_dynamic_BCAO_songvilay.cpp` | `BCAO/md_songvilay.param` | `molecular_dynamics` | MD with Songvilay model |
| `molecular_dynamic_BCAO.cpp` | - | Various modes | Generic BCAO simulations |

**Key Features:**
- Emily model: Uses J1xy, J1z, D, E, F, G parameters with anisotropic g-factors
- Songvilay model: Kitaev-like parameterization with J1, K, Gamma, Gammap, J2, J3, J4
- Support for simulated annealing, molecular dynamics, and parallel tempering
- Anisotropic magnetic field with g-factor tensors

### Kitaev (Kitaev Honeycomb Model)

| Legacy Script | Example Config | Simulation Mode | Description |
|---------------|----------------|-----------------|-------------|
| `molecular_dynamic_kitaev_honeycomb.cpp` | `Kitaev/md_kitaev.param` | `molecular_dynamics` | Standard Kitaev MD |
| `nonlinearspectroscopy_kitaev_honeycomb` | `Kitaev/2dcs_kitaev.param` | `pump_probe_2dcs` | 2D coherent spectroscopy |

**Key Features:**
- Pure Kitaev interactions with K, Gamma, Gammap parameters
- Optional Heisenberg exchange (J)
- Pump-probe and 2DCS capabilities
- Field direction typically along [111]

### Pyrochlore

| Legacy Script | Example Config | Simulation Mode | Description |
|---------------|----------------|-----------------|-------------|
| `molecular_dynamic_pyrochlore.cpp` | `Pyrochlore/md_pyrochlore.param` | `molecular_dynamics` | Pyrochlore MD |
| Field scan functionality | `Pyrochlore/field_scan.param` | `simulated_annealing` | Magnetic field scan |

**Key Features:**
- Exchange anisotropy (Jxx, Jyy, Jzz)
- g-tensor anisotropy (gxx, gyy, gzz, theta)
- Magnetic field scans with MPI parallelization
- Multiple field direction options (001, 110, 1-10, 111)

### TmFeO3 (Mixed Fe/Tm Lattice)

| Legacy Script | Example Config | Simulation Mode | Description |
|---------------|----------------|-----------------|-------------|
| `molecular_dynamic_TmFeO3.cpp` | `TmFeO3/md_tmfeo3.param` | `molecular_dynamics` | TmFeO3 MD |
| `TmFeO3_2DCS.cpp` | `TmFeO3/2dcs_tmfeo3.param` | `pump_probe_2dcs` | 2DCS with Fe-Tm coupling |
| - | `TmFeO3/sa_tmfeo3_chii_scan.param` | `simulated_annealing` | Parameter scanning |

**Key Features:**
- Mixed SU(2) (Fe) and SU(3) (Tm) spins
- Fe nearest/next-nearest neighbor exchange (J1ab, J1c, J2ab, J2c)
- Fe single-ion anisotropy (Ka, Kc)
- DM interactions (D1, D2)
- Tm energy level splitting (e1, e2)
- Fe-Tm bilinear coupling (chii parameter)
- CUDA-accelerated 2DCS option

### Parameter Sweeps (N-Dimensional)

The `param_sweeps/` directory contains examples for systematic parameter space exploration:

| Example Config | Dimensions | Parameters | Base Simulation |
|----------------|------------|------------|-----------------|
| `parameter_sweep_example.param` | 1D | J1xy | Simulated annealing |
| `field_sweep_example.param` | 1D | field_strength | Molecular dynamics |
| `2d_parameter_sweep_example.param` | 2D | J1xy × field | Simulated annealing |
| `3d_parameter_sweep_example.param` | 3D | K × Γ × field | Simulated annealing |
| `2dcs_chii_sweep.param` | 1D | chii | 2DCS spectroscopy |
| `pump_amplitude_sweep.param` | 1D | pump_amplitude | Pump-probe |
| `2d_pump_probe_sweep_example.param` | 2D | pump × probe | Pump-probe |

**Key Features:**
- Support for 1D, 2D, 3D, and higher-dimensional sweeps
- Any Hamiltonian or simulation parameter can be swept
- MPI-parallelized distribution of sweep points
- Each sweep point runs independently
- Organized output directory structure

## How to Use

### 1. Single Simulation

```bash
# Compile first
cmake --build build -j$(nproc)

# Run simulation
./build/spin_solver example_configs/BCAO/md_emily.param
```

### 2. Parallel Tempering (MPI)

```bash
# Run with N MPI processes (one per temperature replica)
mpirun -np 48 ./build/spin_solver example_configs/BCAO/pt_emily.param
```

### 3. Parameter Sweeps (MPI)

```bash
# 1D sweep (sweep points distributed across MPI ranks)
mpirun -np 8 ./build/spin_solver example_configs/param_sweeps/parameter_sweep_example.param

# 2D sweep (creates grid of sweep points)
mpirun -np 16 ./build/spin_solver example_configs/param_sweeps/2d_parameter_sweep_example.param

# 3D phase diagram sweep
mpirun -np 32 ./build/spin_solver example_configs/param_sweeps/3d_parameter_sweep_example.param
```

### 4. Field Scans

```bash
# MPI processes will distribute field values
mpirun -np 20 ./build/spin_solver example_configs/Pyrochlore/field_scan.param
```

### 4. GPU-Accelerated Simulations

```bash
# Set use_gpu = true in config file
./build/spin_solver example_configs/TmFeO3/2dcs_tmfeo3.param
```

## Configuration File Format

Config files use a simple `key = value` format:

```ini
# Comments start with #
parameter_name = value

# Arrays use commas
field_direction = 0,1,0

# Booleans
use_gpu = true
```

### Common Parameters

**All Systems:**
- `system`: `bcao_honeycomb`, `kitaev_honeycomb`, `pyrochlore`, `tmfeo3`
- `simulation_mode`: `simulated_annealing`, `molecular_dynamics`, `parallel_tempering`, `pump_probe`, `pump_probe_2dcs`
- `lattice_size`: Lx,Ly,Lz (integers)
- `output_dir`: Directory for output files

**Temperature/Annealing:**
- `T_start`: Starting temperature (K)
- `T_end`: Final temperature (K)
- `annealing_steps`: Number of MC sweeps
- `cooling_rate`: Exponential cooling factor (0-1)
- `overrelaxation_rate`: Frequency of overrelaxation moves

**Molecular Dynamics:**
- `md_time_start`, `md_time_end`, `md_timestep`
- `md_integrator`: `rk4` or `verlet`
- `md_save_interval`: Frames between saves
- `use_gpu`: Enable CUDA acceleration

**Pump-Probe/2DCS:**
- `pump_direction`: Field direction vector
- `pump_amplitude`, `pump_width`, `pump_frequency`
- `pump_time`: Pump pulse center time
- `tau_start`, `tau_end`, `tau_step`: Delay time scan

## Mapping Legacy to Unified

### Function Call Equivalence

**Legacy:**
```cpp
MD_BCAO_honeycomb(num_trials=5, h=0.0, field_dir={0,1,0}, 
                  dir="output", J1xy=-7.6, ...);
```

**Unified Config:**
```ini
system = bcao_honeycomb
simulation_mode = molecular_dynamics
field_strength = 0.0
field_direction = 0,1,0
output_dir = output
J1xy = -7.6
```

### Parameter Renaming

Some parameters have been renamed for clarity:

| Legacy | Unified | Notes |
|--------|---------|-------|
| `h` | `field_strength` | Magnetic field magnitude |
| `field_dir` | `field_direction` | Unit vector |
| `dir` | `output_dir` | Output directory |
| `num_trials` | Implicit loop | Run config multiple times |
| Temperature args | `T_start`, `T_end` | More descriptive |

## Advanced Features

### Twist Boundary Conditions

```ini
use_twist_boundary = true
```

### Gaussian Spin Updates

```ini
gaussian_move = true
```

### Custom g-tensors (BCAO)

```ini
g_factor = 4.8,4.85,2.5  # gx, gy, gz
```

### Mixed Lattice Parameters (TmFeO3)

```ini
# SU(2) spins (Fe)
J1ab = 4.92
chii = 0.05

# SU(3) spins (Tm)
e1 = 0.97
e2 = 3.97
tm_alpha_scale = 1.0
tm_beta_scale = 1.0
```

## Validation

These configurations have been verified to produce equivalent behavior to the legacy scripts. Key checks:

1. ✅ **Energy Matching**: Ground state energies match within numerical precision
2. ✅ **Spin Configurations**: Equilibrium spin patterns are identical
3. ✅ **Observables**: Magnetization, structure factors, correlation functions agree
4. ✅ **Dynamics**: MD trajectories follow the same evolution

## Troubleshooting

### MPI Issues
```bash
# Check MPI processes
mpirun -np 4 hostname

# Run with debugging
mpirun -np 4 --mca btl_base_verbose 10 ./build/spin_solver config.param
```

### CUDA Issues
```bash
# Check available GPUs
nvidia-smi

# Disable CUDA if needed
use_gpu = false
```

### File Not Found
```bash
# Use absolute paths
./build/spin_solver $(pwd)/example_configs/BCAO/md_emily.param
```

## Creating New Configurations

1. Copy an existing config from the appropriate system directory
2. Modify parameters as needed
3. Run with `./build/spin_solver your_config.param`
4. Check `output_dir/simulation_parameters.txt` for verification

## References

- Legacy scripts: `/legacy/run_scripts/`
- Source code: `/src/apps/spin_solver.cpp`
- Parameter documentation: `SPIN_SOLVER_QUICKREF.md`
- Build instructions: `CMakeLists.txt`
