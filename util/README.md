# Utility Scripts

This directory contains helper scripts for analyzing and visualizing simulation output.

## Parallel Tempering HDF5 Analysis

### `read_pt_hdf5.py`

Python script to read and visualize parallel tempering HDF5 output files.

**Requirements:**
```bash
pip install h5py numpy matplotlib
```

**Usage:**
```bash
python util/read_pt_hdf5.py <output_directory>
```

**Example:**
```bash
python util/read_pt_hdf5.py output_parallel_tempering/
```

**Output:**
- Prints detailed summary of simulation parameters and observables
- Generates `timeseries_rank0.png`: Time series plots for the first replica
  - Energy evolution
  - Magnetization magnitude
  - Energy histogram
  - Sublattice magnetizations
- Generates `specific_heat_vs_T.png`: Temperature-dependent specific heat

**Features:**
- Reads comprehensive HDF5 parallel tempering data
- Displays metadata and simulation parameters
- Calculates derived quantities (magnetization magnitudes, etc.)
- Produces publication-quality plots

## Other Utility Scripts

### `simulate_2dcs_plot.py`

Visualization script for 2D coherent spectroscopy simulations.

### Classical Solvers (`classical_solvers/`)

Additional Monte Carlo solvers and analysis tools.

### Readers (`readers/` and `readers_new/`)

Data reading utilities for various output formats.
