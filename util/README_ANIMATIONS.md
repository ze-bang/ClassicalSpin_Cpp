# DSSF Animation for Pyrochlore Classical Spin Simulations

This directory contains tools for creating animations of DSSF (Dynamical Spin Structure Factor) spinon+photon data across magnetic field strengths for pyrochlore classical spin simulations.

## Features

The animation tools create animated GIFs showing how the DSSF spinon+photon signal (S_xx + S_zz) evolves as a function of magnetic field strength. Separate animations are created for:
- **Local channel**: DSSF_spinon_photon_local.gif
- **Global channel**: DSSF_spinon_photon_global.gif

Each animation shows:
- The total spinon+photon signal (S_xx + S_zz) in red
- Individual S_xx component in blue dashed line
- Individual S_zz component in green dashed line

## Requirements

- Python 3.x
- numpy
- matplotlib
- Pillow (for GIF generation)
- scipy
- opt_einsum

Install missing dependencies:
```bash
pip install numpy matplotlib Pillow scipy opt_einsum
```

## Usage

### Method 1: Using the standalone animation script

The easiest way to create animations:

```bash
python animate_DSSF_pyrochlore.py /path/to/root_dir
```

With optional arguments:
```bash
python animate_DSSF_pyrochlore.py /path/to/root_dir --output /path/to/output --fps 10 --energy-conversion 0.063
```

Arguments:
- `root_dir`: Root directory containing field subdirectories (e.g., field_0.0, field_0.1, etc.)
- `--output`: Output directory for animations (default: root_dir/animations)
- `--fps`: Frames per second for animation (default: 5)
- `--energy-conversion`: Energy conversion factor to meV (default: 0.063)

### Method 2: Using the main reader script

You can also use the main `reader_pyrochlore.py` script:

```bash
python reader_pyrochlore.py /path/to/root_dir --animate
```

With optional arguments:
```bash
python reader_pyrochlore.py /path/to/root_dir --animate --animation-fps 10 --animation-output /path/to/output
```

### Method 3: Programmatic usage

Import and use the functions in your own Python scripts:

```python
from reader_pyrochlore import animate_DSSF_spinon_photon

# Create animations
animate_DSSF_spinon_photon(
    root_dir="/path/to/root_dir",
    output_dir="/path/to/output",
    fps=5,
    energy_conversion=0.063
)
```

## Input Data Structure

The script expects the following directory structure:

```
root_dir/
├── field_0.0/  (or h=0.0, B=0.0, etc.)
│   └── results/
│       ├── DSSF_local_xx.txt
│       ├── DSSF_local_zz.txt
│       ├── DSSF_global_xx.txt
│       └── DSSF_global_zz.txt
├── field_0.1/
│   └── results/
│       ├── DSSF_local_xx.txt
│       ├── DSSF_local_zz.txt
│       ├── DSSF_global_xx.txt
│       └── DSSF_global_zz.txt
└── ...
```

Each DSSF file should be a 2-column text file:
- Column 1: Frequency/energy (ω)
- Column 2: DSSF intensity

## Output

The animations will be saved as:
- `DSSF_spinon_photon_local.gif`: Animation for local channel
- `DSSF_spinon_photon_global.gif`: Animation for global channel

## Field Directory Naming

The script automatically extracts field values from directory names using these patterns:
- `field_X.X` or `field=X.X`
- `h_X.X` or `h=X.X`
- `B_X.X` or `B=X.X`

where X.X is the numerical field value.

## Notes

- The script automatically sorts field directories by field strength
- Energy values are converted to meV using the energy_conversion factor (default: 0.063)
- Y-axis limits are automatically determined from the data to ensure consistency across all frames
- Animations loop continuously by default

## Troubleshooting

**Missing files error**: Ensure all field directories have the required DSSF data files in their `results/` subdirectory.

**Could not extract field value**: Ensure your field directory names follow one of the supported naming patterns.

**Import errors**: Install missing Python packages using pip.

## Examples

Create animations with 10 fps:
```bash
python animate_DSSF_pyrochlore.py /scratch/data/MD_pi_flux_sweep --fps 10
```

Create animations with custom output directory:
```bash
python animate_DSSF_pyrochlore.py /scratch/data/MD_pi_flux_sweep --output ~/animations
```

Process data and create animations in one go:
```bash
# First process the data to generate DSSF files
python reader_pyrochlore.py /scratch/data/MD_pi_flux_sweep --mag HnHn

# Then create animations
python animate_DSSF_pyrochlore.py /scratch/data/MD_pi_flux_sweep
```

## Related Functions

- `extract_field_value(field_dir)`: Extract field value from directory name
- `collect_DSSF_data_all_fields(root_dir)`: Collect DSSF data from all field directories
- `animate_DSSF_spinon_photon(root_dir, ...)`: Create animations for both channels
