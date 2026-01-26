# ClassicalSpin_Cpp

A high-performance C++/CUDA framework for simulating classical spin systems on various lattice geometries. This software package provides efficient implementations of Monte Carlo methods (simulated annealing, parallel tempering) and molecular dynamics simulations for studying magnetic materials.

## Features

- **Multiple Lattice Types**
  - Honeycomb lattice (BCAO, Kitaev models)
  - Pyrochlore lattice
  - TmFeO3 mixed lattice (SU(2) + SU(3) spins)
  - Extensible architecture for custom lattices

- **Simulation Methods**
  - Simulated Annealing (SA)
  - Parallel Tempering (PT) with MPI
  - Molecular Dynamics (MD)
  - Pump-Probe spectroscopy
  - 2D Coherent Spectroscopy (2DCS)
  - N-dimensional parameter sweeps

- **High Performance**
  - CUDA GPU acceleration for MD and spectroscopy
  - MPI parallelization for PT and parameter sweeps
  - OpenMP for shared-memory parallelism
  - Optimized numerical routines with Eigen3

- **Flexible Configuration**
  - Simple parameter file format
  - Support for arbitrary Hamiltonian parameters
  - Field scans and parameter sweeps
  - HDF5 output for large datasets

## Requirements

- CMake 3.18+
- C++20 compatible compiler (GCC 10+, Clang 12+)
- CUDA Toolkit 11.0+ (for GPU support)
- MPI implementation (OpenMPI, MPICH, etc.)
- Eigen3 3.3+
- Boost 1.65+
- HDF5 with C++ bindings

### Ubuntu/Debian Installation

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake \
    libopenmpi-dev openmpi-bin \
    libeigen3-dev \
    libboost-all-dev \
    libhdf5-dev libhdf5-cpp-103
```

For CUDA support, install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

## Building

```bash
# Clone the repository
git clone https://github.com/ze-bang/ClassicalSpin_Cpp.git
cd ClassicalSpin_Cpp

# Create build directory
mkdir build && cd build

# Configure (Release build recommended)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build (use all available cores)
cmake --build . -j$(nproc)
```

The main executable `spin_solver` will be created in the `build/` directory.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug/Release) |
| `CMAKE_CUDA_ARCHITECTURES` | 60;75;80;86;89 | Target GPU architectures |

## Quick Start

### Basic Usage

```bash
# Run a simulation with a parameter file
./build/spin_solver config.param

# Run with MPI for parallel tempering
mpirun -np 16 ./build/spin_solver example_configs/BCAO/pt_emily.param

# Run parameter sweep
mpirun -np 8 ./build/spin_solver example_configs/param_sweeps/parameter_sweep_example.param
```

### Example: BCAO Honeycomb Simulated Annealing

```bash
./build/spin_solver example_configs/BCAO/sa_emily.param
```

### Example: Kitaev Model Molecular Dynamics

```bash
./build/spin_solver example_configs/Kitaev/md_kitaev.param
```

### Example: Pyrochlore Field Scan

```bash
mpirun -np 20 ./build/spin_solver example_configs/Pyrochlore/field_scan.param
```

## Configuration Files

Configuration files use a simple `key = value` format:

```ini
# System selection
system = honeycomb_bcao
simulation_mode = simulated_annealing

# Lattice size (Lx, Ly, Lz)
lattice_size = 24, 24, 1

# Temperature range
T_start = 10.0
T_end = 0.001
annealing_steps = 100000

# Hamiltonian parameters
J1xy = -7.6
J1z = -1.2
D = 0.1

# Magnetic field
field_strength = 0.0
field_direction = 0, 1, 0
g_factor = 4.8, 4.85, 2.5

# Output
output_dir = output
save_observables = true
```

### Supported Systems

| System | Key | Description |
|--------|-----|-------------|
| BCAO Honeycomb | `honeycomb_bcao` | Ba₃CoSb₂O₉ with anisotropic exchange |
| Kitaev Honeycomb | `honeycomb_kitaev` | Kitaev model with K, Γ, Γ' interactions |
| Pyrochlore | `pyrochlore` | Pyrochlore lattice with exchange anisotropy |
| TmFeO3 | `tmfeo3` | Mixed Fe (SU2) + Tm (SU3) spins |

### Simulation Modes

| Mode | Key | Description |
|------|-----|-------------|
| Simulated Annealing | `simulated_annealing` | Single replica cooling |
| Parallel Tempering | `parallel_tempering` | Multi-replica with exchanges |
| Molecular Dynamics | `molecular_dynamics` | Spin dynamics integration |
| Pump-Probe | `pump_probe` | Magnetic pulse response |
| 2D Spectroscopy | `2dcs` | Two-pulse coherent spectroscopy |
| Parameter Sweep | `parameter_sweep` | Systematic parameter exploration |

### Optimized Parallel Tempering

The code implements the feedback-optimized temperature grid algorithm from [Bittner et al., Phys. Rev. Lett. 101, 130603 (2008)](https://arxiv.org/abs/0809.0571). This automatically generates optimal temperature spacing to:

- Achieve **uniform 50% acceptance rate** across all temperature pairs
- **Minimize round-trip time** in temperature space (maximizes diffusivity)
- Adapt to system-specific energy landscape

Configuration options:

```ini
# Enable optimized temperature grid (default: true)
pt_optimize_temperatures = true

# Target acceptance rate (0.5 = optimal per Bittner et al.)
pt_target_acceptance = 0.5

# Optimization parameters
pt_optimization_warmup = 500      # Warmup sweeps per replica
pt_optimization_sweeps = 500      # MC sweeps per feedback iteration
pt_optimization_iterations = 20    # Number of feedback iterations
```

The algorithm outputs diagnostic information including:
- Final acceptance rates for each temperature pair
- Local diffusivities D(T) = A(1-A)
- Estimated round-trip time
- Convergence status

Set `pt_optimize_temperatures = false` to use simple geometric (logarithmic) spacing instead.

## Example Configurations

The `example_configs/` directory contains ready-to-use configurations:

```
example_configs/
├── BCAO/           # Ba₃CoSb₂O₉ honeycomb configurations
├── Kitaev/         # Kitaev honeycomb model
├── Pyrochlore/     # Pyrochlore lattice
├── TmFeO3/         # TmFeO3 mixed lattice
└── param_sweeps/   # N-dimensional parameter sweeps
```

See [example_configs/README.md](example_configs/README.md) for detailed documentation.

## Project Structure

```
ClassicalSpin_Cpp/
├── CMakeLists.txt           # Build configuration
├── README.md                 # This file
├── LICENSE                   # MIT License
├── include/
│   └── classical_spin/
│       ├── core/             # Core classes (SpinConfig, UnitCell, etc.)
│       ├── lattice/          # Lattice implementations
│       ├── gpu/              # CUDA kernels and GPU helpers
│       └── io/               # HDF5 I/O utilities
├── src/
│   ├── apps/                 # Main executables
│   │   └── spin_solver.cpp   # Main simulation driver
│   ├── core/                 # Core implementations
│   └── gpu/                  # CUDA implementations
├── example_configs/          # Example parameter files
├── util/                     # Utility scripts (Python readers, etc.)
└── legacy/                   # Legacy run scripts (deprecated)
```

## Key Classes

### SpinConfig
Configuration structure that holds all simulation parameters. Parse from file with:
```cpp
SpinConfig config = SpinConfig::from_file("config.param");
```

### UnitCell / MixedUnitCell
Defines the magnetic unit cell including:
- Lattice vectors
- Atom positions
- Exchange interactions
- Single-ion anisotropies
- Magnetic fields

### Lattice / MixedLattice
Full simulation lattice built from unit cells. Provides:
- Spin initialization
- Energy calculations
- Monte Carlo updates
- Molecular dynamics integration
- Observable measurements

## Output Files

Simulations produce output in the specified `output_dir`:

```
output/
├── sample_0/
│   ├── positions.txt       # Spin positions
│   ├── spins.txt          # Final spin configuration
│   ├── final_energy.txt   # Ground state energy
│   ├── observables.txt    # Temperature-dependent observables
│   └── trajectory.h5      # MD trajectory (HDF5)
├── rank_0/
│   └── parallel_tempering_data.h5  # PT replica data (HDF5)
├── parallel_tempering_aggregated.h5  # PT temperature scan (HDF5)
└── simulation_parameters.txt  # Copy of input parameters
```

**HDF5 Output:**  
Parallel tempering simulations use structured HDF5 format to minimize file count on cluster filesystems. See [PARALLEL_TEMPERING_HDF5.md](PARALLEL_TEMPERING_HDF5.md) for detailed documentation.

## GPU Acceleration

Enable GPU acceleration for molecular dynamics:

```ini
use_gpu = true
```

Supported operations:
- Spin dynamics integration (RK4, Dopri5)
- Energy calculations
- Pump-probe simulations
- 2DCS spectroscopy

## Python Utilities

Analysis scripts are provided in `util/`:

```python
# Read honeycomb lattice results
python util/readers_new/reader_honeycomb.py output_dir/ plot

# Read pyrochlore results
python util/readers_new/reader_pyrochlore.py output_dir/ analyze

# Read and visualize parallel tempering HDF5 output
python util/read_pt_hdf5.py output_parallel_tempering/
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Eigen3](https://eigen.tuxfamily.org/) for linear algebra
- [Thrust](https://thrust.github.io/) for GPU primitives
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) for data storage
- [Boost](https://www.boost.org/) for utilities
