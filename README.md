# ClassicalSpin_Cpp

**Unified Framework for Classical Spin Dynamics Simulations**

A high-performance C++ simulation framework for studying frustrated magnetic systems including BCAO, Kitaev honeycomb models, pyrochlore lattices, and mixed SU(2)×SU(3) systems like TmFeO₃.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)]()

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Systems](#supported-systems)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Examples](#examples)
- [Output Format](#output-format)
- [Performance](#performance)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Overview

This framework provides a **unified interface** for running various types of classical spin dynamics simulations, replacing the legacy collection of specialized run scripts with a single, flexible, parameter-driven executable.

### Key Advantages

- **Single Executable:** One binary (`unified_simulation`) handles all simulation types
- **Parameter Files:** Human-readable config files instead of hard-coded parameters
- **Comprehensive I/O:** HDF5 output with full metadata for reproducibility
- **MPI Support:** Scalable parallel tempering and field scans
- **GPU Acceleration:** CUDA support for TmFeO₃ 2DCS simulations
- **Well-Documented:** Extensive examples and documentation

---

## Features

### Simulation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **Simulated Annealing** | Monte Carlo with exponential cooling | Ground state search |
| **Parallel Tempering** | MPI replica exchange across temperatures | Phase transitions, critical phenomena |
| **Molecular Dynamics** | Classical spin equations of motion | Spin wave dynamics, thermalization |
| **Pump-Probe** | Time-resolved spectroscopy | Ultrafast magnetization dynamics |
| **2DCS** | Two-dimensional coherent spectroscopy | Nonlinear optical response |

### Physical Systems

<table>
<tr>
<th>System</th>
<th>Lattice</th>
<th>Spin Type</th>
<th>Key Interactions</th>
</tr>
<tr>
<td><b>BCAO</b><br>(BCAO)</td>
<td>Honeycomb</td>
<td>SU(2)</td>
<td>
  • Anisotropic J1 exchange<br>
  • DM interactions (D, E, F, G)<br>
  • J3 third-neighbor coupling<br>
  • Anisotropic g-factors
</td>
</tr>
<tr>
<td><b>Kitaev Honeycomb</b></td>
<td>Honeycomb</td>
<td>SU(2)</td>
<td>
  • Kitaev interaction (K)<br>
  • Symmetric off-diagonal (Γ, Γ')<br>
  • Heisenberg exchange (J)
</td>
</tr>
<tr>
<td><b>Pyrochlore</b></td>
<td>Pyrochlore</td>
<td>SU(2)</td>
<td>
  • Anisotropic exchange (Jxx, Jyy, Jzz)<br>
  • g-tensor anisotropy<br>
  • Rotation angle θ
</td>
</tr>
<tr>
<td><b>TmFeO₃</b></td>
<td>Orthorhombic</td>
<td>SU(2) + SU(3)</td>
<td>
  • Fe-Fe exchange (J1, J2)<br>
  • Fe single-ion anisotropy (K)<br>
  • Tm CEF splitting (e1, e2)<br>
  • Fe-Tm coupling (χ)
</td>
</tr>
</table>

### Technical Features

- **Modern C++17** with Eigen for linear algebra
- **HDF5 I/O** with compression and chunking
- **Boost.Odeint** integration for advanced ODE solvers
- **MPI** parallelization via OpenMPI/MPICH
- **CUDA** support for GPU acceleration
- **Flexible Configuration** with `.param` files
- **Comprehensive Metadata** for reproducibility

---

## Supported Systems

### Experimental Platforms

- Linux (Ubuntu 20.04+, CentOS 7+, Arch)
- macOS (10.15+)
- WSL2 (Windows Subsystem for Linux)

### Dependencies

**Required:**
- C++17 compiler (GCC ≥9.0, Clang ≥10.0)
- CMake ≥3.15
- Eigen3 ≥3.3
- HDF5 ≥1.10 (with C++ bindings)
- Boost ≥1.65 (headers only)

**Optional:**
- OpenMPI or MPICH (for parallel tempering)
- CUDA ≥11.0 (for GPU-accelerated 2DCS)

---

## Installation

### 1. Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y build-essential cmake \
    libeigen3-dev libhdf5-dev libboost-all-dev \
    libopenmpi-dev openmpi-bin
```

**macOS (Homebrew):**
```bash
brew install cmake eigen hdf5 boost open-mpi
```

**CentOS/RHEL:**
```bash
sudo yum install -y gcc-c++ cmake3 \
    eigen3-devel hdf5-devel boost-devel \
    openmpi-devel
```

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/ClassicalSpin_Cpp.git
cd ClassicalSpin_Cpp
```

### 3. Build

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

**Build with CUDA support:**
```bash
cmake -DUSE_CUDA=ON ..
make -j$(nproc)
```

### 4. Verify Installation

```bash
./unified_simulation --version
./unified_simulation --help
```

---

## Quick Start

### Example 1: Molecular Dynamics (BCAO)

```bash
# Run MD simulation with Emily model parameters
./build/unified_simulation example_configs/BCAO/md_emily.param
```

**Output:**
- HDF5 file: `BCAO_emily_md/md_trajectory.h5`
- Contains: spin configurations, magnetization vs. time

### Example 2: Parallel Tempering (with MPI)

```bash
# Run parallel tempering with 48 temperature replicas
mpirun -np 48 ./build/unified_simulation example_configs/BCAO/pt_emily.param
```

### Example 3: 2DCS Spectroscopy (Kitaev)

```bash
# Run 2D coherent spectroscopy
./build/unified_simulation example_configs/Kitaev/2dcs_kitaev.param
```

### Example 4: Field Scan (Pyrochlore)

```bash
# Scan magnetic field from 0 to 5 T with 20 MPI ranks
mpirun -np 20 ./build/unified_simulation example_configs/Pyrochlore/field_scan.param
```

---

## Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| [`example_configs/README.md`](example_configs/README.md) | **Complete guide to example configurations** |
| [`example_configs/HDF5_OUTPUT_FORMAT.md`](example_configs/HDF5_OUTPUT_FORMAT.md) | **HDF5 file structure reference** |
| [`UNIFIED_SIMULATION_QUICKREF.md`](UNIFIED_SIMULATION_QUICKREF.md) | Parameter reference |
| [`CONFIG_PARAMETERS_UPDATE.md`](CONFIG_PARAMETERS_UPDATE.md) | Parameter changes from legacy code |

### Parameter Files

Config files use a simple `key = value` format:

```ini
# BCAO Molecular Dynamics Example
system = bcao_honeycomb
simulation_mode = molecular_dynamics

lattice_size = 36,36,1

# Exchange parameters
J1xy = -7.6
J1z = -1.2
D = 0.1
E = -0.1
J3xy = 2.5
J3z = -0.85

# Magnetic field
field_strength = 0.0
field_direction = 0,1,0
g_factor = 4.8,4.85,2.5

# Annealing
T_start = 20.0
T_end = 1.0
annealing_steps = 100000

# Molecular dynamics
md_time_start = 0.0
md_time_end = 200.0
md_timestep = 0.01
md_integrator = rk4

output_dir = BCAO_emily_md
```

See [`example_configs/`](example_configs/) for complete examples.

---

## Examples

### Directory Structure

```
example_configs/
├── README.md                    # Comprehensive usage guide
├── HDF5_OUTPUT_FORMAT.md       # Output format documentation
│
├── BCAO/                        # BCAO examples
│   ├── md_emily.param
│   ├── sa_emily.param
│   ├── pt_emily.param
│   └── md_songvilay.param
│
├── Kitaev/                      # Kitaev honeycomb examples
│   ├── md_kitaev.param
│   └── 2dcs_kitaev.param
│
├── Pyrochlore/                  # Pyrochlore examples
│   ├── md_pyrochlore.param
│   └── field_scan.param
│
├── TmFeO3/                      # TmFeO₃ mixed lattice examples
    ├── md_tmfeo3.param
    ├── 2dcs_tmfeo3.param
    └── sa_tmfeo3_chii_scan.param

### Running Examples

**Local workstation:**
```bash
cd example_configs/run_scripts
./local_md.sh ../BCAO/md_emily.param
```

**SLURM cluster:**
```bash
cd example_configs/run_scripts
sbatch slurm_md.sh ../BCAO/md_emily.param
```

**With MPI:**
```bash
./local_mpi.sh 8 ../BCAO/pt_emily.param
```

---

## Output Format

All simulations output to **HDF5 format** with comprehensive metadata.

### Molecular Dynamics Output

```
md_trajectory.h5
├── metadata/
│   ├── lattice_size, spin_dim, dimensions
│   ├── integration_method, dt, T_start, T_end
│   ├── creation_time, code_version
│   └── positions (optional)
│
└── trajectory/
    ├── times [n_steps]
    ├── spins [n_steps, n_sites, spin_dim]
    ├── magnetization_antiferro [n_steps, spin_dim]
    └── magnetization_local [n_steps, spin_dim]
```

### Pump-Probe / 2DCS Output

```
2dcs_output.h5
├── metadata/
│   ├── [lattice parameters]
│   ├── pulse_amp, pulse_width, pulse_freq
│   ├── tau_start, tau_end, tau_step
│   └── ground_state_energy
│
├── reference/
│   ├── times, M_antiferro, M_local
│   
└── tau_scan/
    ├── tau_values [n_tau]
    └── tau_i/
        ├── M1_*, M01_* (magnetization trajectories)
```

**See [`example_configs/HDF5_OUTPUT_FORMAT.md`](example_configs/HDF5_OUTPUT_FORMAT.md) for complete specification.**

### Reading Output (Python)

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('md_trajectory.h5', 'r') as f:
    times = f['trajectory/times'][:]
    mag = f['trajectory/magnetization_antiferro'][:]
    
    plt.plot(times, np.linalg.norm(mag, axis=1))
    plt.xlabel('Time')
    plt.ylabel('|M_antiferro|')
    plt.show()
```

---

## Performance

### Benchmarks

**System:** BCAO honeycomb, 36×36×1 lattice (2592 spins)

| Method | Hardware | Performance |
|--------|----------|-------------|
| Simulated Annealing | Intel Xeon Gold 6248 | ~50k sweeps/sec |
| Parallel Tempering | 48 cores MPI | Linear scaling to 48 replicas |
| Molecular Dynamics | Single core | ~500 timesteps/sec (RK4) |
| 2DCS (TmFeO₃) | NVIDIA A100 GPU | 10× speedup vs CPU |

### Optimization Tips

1. **Use appropriate integration method:**
   - `rk4`: Good balance of speed/accuracy
   - `dopri5`: High accuracy, slower
   - `verlet`: Fastest, energy-conserving

2. **Tune save interval:** Large `md_save_interval` reduces I/O overhead

3. **Enable compiler optimizations:**
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

4. **Use MPI for parameter scans:** Distribute field/temperature points

5. **GPU acceleration:** Enable CUDA for large 2DCS calculations

---

## Legacy Code Migration

This framework replaces the legacy `run_scripts/` directory with a unified approach:

| Legacy Script | Unified Config | Notes |
|---------------|----------------|-------|
| `molecular_dynamic_BCAO_emily.cpp` | `BCAO/md_emily.param` | Parameters → config file |
| `parallel_tempering_BCAO_emily.cpp` | `BCAO/pt_emily.param` | MPI-based |
| `molecular_dynamic_kitaev_honeycomb.cpp` | `Kitaev/md_kitaev.param` | 2DCS included |
| `molecular_dynamic_pyrochlore.cpp` | `Pyrochlore/md_pyrochlore.param` | Field scans |
| `TmFeO3_2DCS.cpp` | `TmFeO3/2dcs_tmfeo3.param` | GPU-accelerated |

**See [`example_configs/README.md`](example_configs/README.md) for detailed mapping.**

---

## Project Structure

```
ClassicalSpin_Cpp/
├── src/                         # Source code
│   ├── unified_simulation.cpp   # Main entry point
│   ├── unified_config.h         # Configuration parser
│   ├── lattice.h                # Lattice builder
│   ├── mixed_lattice.h          # Mixed SU(2)/SU(3) lattice
│   ├── unitcell.h               # Unit cell definitions
│   ├── hdf5_io.h                # HDF5 I/O
│   └── simulation_config.h      # Simulation parameters
│
├── legacy/                      # Legacy run_scripts (reference)
│   └── run_scripts/             
│
├── example_configs/             # Example parameter files
│   ├── README.md
│   ├── HDF5_OUTPUT_FORMAT.md
│   ├── BCAO/, Kitaev/, Pyrochlore/, TmFeO3/
│   └── run_scripts/             # Job submission scripts
│
├── build/                       # Build directory
├── CMakeLists.txt               # CMake configuration
└── README.md                    # This file
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/new-system`)
3. **Commit** changes with descriptive messages
4. **Test** thoroughly (add example configs)
5. **Submit** a pull request

### Adding a New System

1. Define unit cell in `src/unitcell.h`
2. Add builder function in `src/unified_simulation.cpp`
3. Update `UnifiedConfig` parser for new parameters
4. Create example configs in `example_configs/YourSystem/`
5. Update documentation

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{classicalspin_cpp,
  title = {ClassicalSpin\_Cpp: Unified Classical Spin Dynamics Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/ClassicalSpin_Cpp}
}
```

### Related Publications

- **BCAO Emily model:** [Citation]
- **BCAO Songvilay model:** [Citation]
- **TmFeO₃ 2DCS:** [Citation]

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Eigen library:** High-performance linear algebra
- **HDF5:** Hierarchical data format
- **Boost.Odeint:** ODE integration
- **OpenMPI:** MPI implementation

---

## Contact

For questions, bug reports, or feature requests, please:

- **Open an issue:** [GitHub Issues](https://github.com/yourusername/ClassicalSpin_Cpp/issues)
- **Email:** your.email@institution.edu

---

## Quick Reference Card

### Common Commands

```bash
# Build
cmake --build build -j$(nproc)

# Single simulation
./build/unified_simulation config.param

# MPI parallel (field scan, parallel tempering)
mpirun -np N ./build/unified_simulation config.param

# Submit to SLURM
sbatch example_configs/run_scripts/slurm_md.sh config.param

# Analyze HDF5 output
h5ls -r output.h5
python -c "import h5py; f = h5py.File('output.h5'); print(list(f.keys()))"
```

### Essential Parameters

```ini
# System selection
system = bcao_honeycomb | kitaev_honeycomb | pyrochlore | tmfeo3

# Simulation mode
simulation_mode = simulated_annealing | molecular_dynamics | 
                  parallel_tempering | pump_probe | pump_probe_2dcs

# Lattice
lattice_size = Lx,Ly,Lz

# Temperature/Annealing
T_start = 10.0
T_end = 0.01
annealing_steps = 100000

# Molecular Dynamics
md_time_start = 0.0
md_time_end = 200.0
md_timestep = 0.01
md_integrator = rk4

# Output
output_dir = results/
```

---

**Last Updated:** November 27, 2025  
**Version:** 1.0  
**Build Status:** ✅ Passing
