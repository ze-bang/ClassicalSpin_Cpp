# Parameter Sweep Examples

This directory contains example configurations for N-dimensional parameter sweeps. Parameter sweeps enable systematic exploration of parameter space by running a base simulation (SA, PT, MD, pump-probe, or 2DCS) at each point in a grid.

## Quick Start

```bash
# 1D sweep over K parameter
mpirun -np 8 ./build/spin_solver example_configs/param_sweeps/parameter_sweep_example.param

# 2D sweep over J1xy and field_strength
mpirun -np 16 ./build/spin_solver example_configs/param_sweeps/2d_parameter_sweep_example.param

# 3D phase diagram (K × Γ × field)
mpirun -np 32 ./build/spin_solver example_configs/param_sweeps/3d_parameter_sweep_example.param
```
```

## Available Examples

### 1D Sweeps

| File | System | Parameter | Range | Base Simulation |
|------|--------|-----------|-------|-----------------|
| `parameter_sweep_example.param` | BCAO | J1xy | -10.0 → -5.0 | SA |
| `field_sweep_example.param` | Kitaev | field_strength | 0.0 → 2.0 | MD |
| `pump_amplitude_sweep.param` | Kitaev | pump_amplitude | 0.0 → 2.0 | Pump-probe |
| `probe_delay_sweep_example.param` | BCAO | probe_time | 10.0 → 100.0 | Pump-probe |
| `2dcs_chii_sweep.param` | TmFeO₃ | chii | 0.0 → 0.1 | 2DCS |
| `pump_probe_sweep_example.param` | Kitaev | pump_amplitude | 0.0 → 2.0 | Pump-probe |
| `2dcs_sweep_example.param` | TmFeO₃ | pump_frequency | 0.0 → 5.0 | 2DCS |

### 2D Sweeps

| File | System | Parameters | Grid Size | Base Simulation |
|------|--------|------------|-----------|-----------------|
| `2d_parameter_sweep_example.param` | BCAO | J1xy × field | 6 × 5 | SA |
| `2d_pump_probe_sweep_example.param` | Kitaev | pump_amp × probe_time | 5 × 6 | Pump-probe |

### 3D Sweeps

| File | System | Parameters | Grid Size | Base Simulation |
|------|--------|------------|-----------|-----------------|
| `3d_parameter_sweep_example.param` | Kitaev | K × Γ × field | 5 × 6 × 5 | SA |

## Configuration Syntax

### 1D Sweep (Legacy Syntax - still supported)

```ini
simulation_mode = parameter_sweep

# Single parameter sweep
sweep_parameter = K
sweep_start = -2.0
sweep_end = 0.0
sweep_step = 0.1
sweep_base_simulation = simulated_annealing
```

### N-Dimensional Sweep (New Syntax)

```ini
simulation_mode = parameter_sweep

# Multi-parameter sweep (comma-separated lists)
sweep_parameters = K, Gamma, field_strength
sweep_starts = -2.0, 0.0, 0.0
sweep_ends = 0.0, 0.5, 1.0
sweep_steps = 0.5, 0.1, 0.25
sweep_base_simulation = simulated_annealing
```

## Sweepable Parameters

### Exchange Interactions
- `J1xy`, `J1z` - Nearest-neighbor exchange (BCAO)
- `J3xy`, `J3z` - Third-neighbor exchange (BCAO)
- `K`, `J` - Kitaev and Heisenberg exchange
- `Gamma`, `Gammap` - Symmetric off-diagonal terms
- `Jxx`, `Jyy`, `Jzz` - Pyrochlore exchange anisotropy
- `J1ab`, `J1c`, `J2ab`, `J2c` - TmFeO₃ Fe-Fe exchange

### DM and Anisotropy
- `D`, `E`, `F`, `G` - DM interaction components (BCAO)
- `Ka`, `Kc` - Single-ion anisotropy (TmFeO₃)
- `D1`, `D2` - DM interactions (TmFeO₃)

### Field and Temperature
- `field_strength` - Magnetic field magnitude
- `T_end` - Final temperature

### Mixed Lattice (TmFeO₃)
- `chii` - Fe-Tm bilinear coupling
- `e1`, `e2` - Tm crystal field levels
- `tm_alpha_scale`, `tm_beta_scale` - Tm wavefunction scaling

### Pump-Probe / 2DCS
- `pump_amplitude` - Pump pulse strength
- `pump_frequency` - Pump pulse frequency
- `pump_time` - Pump pulse center time
- `probe_time` - Probe pulse center time
- `probe_amplitude` - Probe pulse strength
- `tau_start`, `tau_end` - 2DCS delay scan range

## Base Simulation Types

| Type | Aliases | Best For |
|------|---------|----------|
| `simulated_annealing` | `SA`, `annealing` | Ground state phase diagrams |
| `parallel_tempering` | `PT`, `tempering` | Complex energy landscapes |
| `molecular_dynamics` | `MD`, `dynamics` | Dynamic properties |
| `pump_probe` | `PUMP_PROBE`, `pump-probe` | Ultrafast response |
| `2dcs` | `2DCS`, `spectroscopy` | Coherent spectroscopy |

## Output Structure

Each sweep point creates its own subdirectory:

```
output_dir/
├── K_-2.000000e+00/           # 1D sweep
│   ├── md_trajectory.h5
│   └── simulation_parameters.txt
├── K_-1.500000e+00/
│   └── ...
│
└── K_-2.000000e+00_Gamma_0.000000e+00/  # 2D sweep
    └── ...
```

## MPI Parallelization

Sweep points are automatically distributed across MPI ranks:

```bash
# 21 sweep points across 8 ranks (~3 points each)
mpirun -np 8 ./build/spin_solver sweep.param
```

For 2D/3D sweeps, the total number of grid points is:
- 2D: `n1 × n2` points
- 3D: `n1 × n2 × n3` points

The grid points are flattened and distributed round-robin across MPI ranks.

## Tips for Efficient Sweeps

1. **Match MPI ranks to sweep points**: Ideal when `n_ranks` divides evenly into total points
2. **Use smaller lattices for exploration**: Start with 8×8×1 or 12×12×1
3. **Reduce annealing steps for coarse scans**: Use 10000-30000 for initial exploration
4. **Increase precision near phase boundaries**: Use smaller step sizes in critical regions
5. **Consider disk space**: Each sweep point generates its own output files

## Example Workflow: Phase Diagram

```bash
# 1. Coarse scan to identify phases
mpirun -np 32 ./build/spin_solver coarse_scan.param

# 2. Analyze results to identify phase boundaries
python analyze_phase_diagram.py output_coarse/

# 3. Fine scan near phase boundary
mpirun -np 64 ./build/spin_solver fine_scan.param

# 4. Generate final phase diagram
python plot_phase_diagram.py output_fine/
```

## See Also

- [`PARAMETER_SWEEP_IMPLEMENTATION.md`](../../PARAMETER_SWEEP_IMPLEMENTATION.md) - Implementation details
- [`PUMP_PROBE_2DCS_SWEEP_IMPLEMENTATION.md`](../../PUMP_PROBE_2DCS_SWEEP_IMPLEMENTATION.md) - Spectroscopy sweeps
- [`../README.md`](../README.md) - Main examples documentation
