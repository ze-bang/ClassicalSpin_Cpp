# Summary of Animation Features Added to reader_pyrochlore.py

## Overview
Added comprehensive animation capabilities to visualize DSSF (Dynamical Spin Structure Factor) spinon+photon data across magnetic field strengths for pyrochlore classical spin simulations.

## Files Modified/Created

### 1. Modified: `reader_pyrochlore.py`

#### New Imports
- Added `from matplotlib.animation import FuncAnimation, PillowWriter`
- Added `import re` for pattern matching

#### New Functions

##### `extract_field_value(field_dir)`
- Extracts numerical field values from directory names
- Supports multiple naming patterns: `field_X`, `h=X`, `B=X`
- Returns float value or None if no match found

##### `collect_DSSF_data_all_fields(root_dir, verbose=True)`
- Collects DSSF data from all field subdirectories
- Reads S_xx and S_zz components for both local and global channels
- Computes spinon+photon signal (S_xx + S_zz)
- Returns sorted list of data dictionaries by field strength
- Handles missing files and provides informative error messages

##### `animate_DSSF_spinon_photon(root_dir, output_dir=None, fps=5, energy_conversion=0.063)`
- Creates animated GIFs showing DSSF evolution across field strengths
- Generates two animations:
  - `DSSF_spinon_photon_local.gif`: Local channel animation
  - `DSSF_spinon_photon_global.gif`: Global channel animation
- Each animation shows:
  - Total spinon+photon (S_xx + S_zz) in red
  - S_xx component in blue dashed line
  - S_zz component in green dashed line
- Features consistent y-axis scaling across all frames
- Converts energy units to meV

#### Modified: `main()` function
- Added `--animate` flag to trigger animation creation
- Added `--animation-fps` option (default: 5)
- Added `--animation-output` option for custom output directory
- Conditional execution: either processes data OR creates animations

### 2. Created: `animate_DSSF_pyrochlore.py`
Standalone script for creating animations without running full data processing.

**Features:**
- Can be run independently
- Same animation functionality as main script
- Command-line interface with argparse
- Easier to use for just creating animations from existing data

**Usage:**
```bash
python animate_DSSF_pyrochlore.py /path/to/data --fps 10 --output /path/to/output
```

### 3. Created: `README_ANIMATIONS.md`
Comprehensive documentation covering:
- Feature overview
- Installation requirements
- Usage examples (3 different methods)
- Expected input data structure
- Output format details
- Field directory naming conventions
- Troubleshooting guide
- Example commands

### 4. Created: `example_animations.py`
Example script demonstrating:
- Basic animation creation
- Custom settings usage
- Data collection and inspection
- Field value extraction testing
- Requirements checking
- Error handling patterns

## Usage Examples

### Method 1: Using main reader script
```bash
# Create animations after processing
python reader_pyrochlore.py /data/MD_pi_flux --animate --animation-fps 10
```

### Method 2: Using standalone script
```bash
# Just create animations from existing processed data
python animate_DSSF_pyrochlore.py /data/MD_pi_flux --fps 10
```

### Method 3: Programmatic
```python
from reader_pyrochlore import animate_DSSF_spinon_photon

animate_DSSF_spinon_photon(
    root_dir="/data/MD_pi_flux",
    fps=10,
    energy_conversion=0.063
)
```

## Input Data Requirements

Expected directory structure:
```
root_dir/
├── field_0.0/
│   └── results/
│       ├── DSSF_local_xx.txt
│       ├── DSSF_local_zz.txt
│       ├── DSSF_global_xx.txt
│       └── DSSF_global_zz.txt
├── field_0.1/
│   └── results/
│       └── ...
└── ...
```

Each DSSF file format:
```
# Column 1: ω (frequency/energy)
# Column 2: DSSF intensity
0.0000  0.123456
0.0001  0.234567
...
```

## Output Files

Animations saved to `{root_dir}/animations/` (or custom directory):
- `DSSF_spinon_photon_local.gif`
- `DSSF_spinon_photon_global.gif`

## Key Features

1. **Automatic field detection**: Extracts field values from directory names
2. **Robust error handling**: Continues if some directories lack data
3. **Consistent visualization**: Same y-axis scale across all frames
4. **Energy unit conversion**: Converts to meV automatically
5. **Component visualization**: Shows both individual and combined signals
6. **Flexible configuration**: FPS, output directory, energy conversion all adjustable

## Technical Details

### Animation Parameters
- Default FPS: 5 (adjustable)
- Default energy conversion: 0.063 (Jzz units to meV)
- Figure size: 10x6 inches
- DPI: Default matplotlib settings
- Format: GIF (using PillowWriter)

### Data Processing
- Sorts field directories numerically by field strength
- Uses first directory's ω array as reference
- Stacks all DSSF data into numpy arrays for efficient animation
- Determines global y-limits from all data

### Supported Field Directory Patterns
- `field_X.X` or `field=X.X`
- `h_X.X` or `h=X.X`
- `B_X.X` or `B=X.X`

## Dependencies

Required Python packages:
- numpy
- matplotlib (with animation support)
- Pillow (for GIF writer)
- scipy
- opt_einsum
- re (standard library)
- os, argparse (standard library)

## Error Handling

The code handles:
- Missing directories gracefully
- Missing DSSF files in some field directories
- Invalid field directory names
- File read errors
- Empty data directories

All errors are reported with informative messages while processing continues where possible.

## Future Enhancements (Potential)

Possible additions:
1. Support for MP4/WebM video formats
2. 2D heatmap animations (field vs. energy)
3. Multiple component animations (all 9 tensor components)
4. Comparison animations (multiple datasets)
5. Interactive HTML5 animations
6. Overlay with experimental data
