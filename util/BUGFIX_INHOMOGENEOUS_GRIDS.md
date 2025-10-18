# Bug Fix: Handling Inhomogeneous Frequency Grids

## Issue
The animation and plotting scripts failed with the following error when processing data from different field directories:

```
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (21,) + inhomogeneous part.
```

## Root Cause
Different field directories had DSSF data with different frequency grid sizes. This happens when:
- Different simulations use different time steps
- FFT produces different frequency resolutions
- Data processing uses different parameters across field strengths

When trying to stack these arrays into a single numpy array for animation, numpy cannot create a regular array from arrays of different lengths.

## Solution
Added automatic detection and interpolation to handle varying frequency grids:

1. **Detection**: Check if all `w` (frequency) arrays have the same length
2. **Interpolation**: If different lengths are detected, interpolate all data to a common grid
3. **Reference Grid**: Use the longest frequency array as the reference grid
4. **Linear Interpolation**: Use `np.interp()` for fast, smooth interpolation

## Implementation Details

### Modified Functions

#### `animate_DSSF_spinon_photon()` (in `reader_pyrochlore.py` and `animate_DSSF_pyrochlore.py`)
```python
# Check if all w arrays have the same length; if not, interpolate to a common grid
w_lengths = [len(d['w']) for d in field_data]
if len(set(w_lengths)) > 1:
    print(f"Warning: Different frequency grid sizes detected: {set(w_lengths)}")
    print("Interpolating all data to common frequency grid...")
    
    # Use the longest w array as the reference
    max_idx = np.argmax(w_lengths)
    w_common = field_data[max_idx]['w']
    
    # Interpolate all data to common grid
    for i, data in enumerate(field_data):
        if len(data['w']) != len(w_common):
            data['dssf_local_spinon_photon'] = np.interp(w_common, data['w'], data['dssf_local_spinon_photon'])
            data['dssf_global_spinon_photon'] = np.interp(w_common, data['w'], data['dssf_global_spinon_photon'])
            data['dssf_local_xx'] = np.interp(w_common, data['w'], data['dssf_local_xx'])
            data['dssf_local_zz'] = np.interp(w_common, data['w'], data['dssf_local_zz'])
            data['dssf_global_xx'] = np.interp(w_common, data['w'], data['dssf_global_xx'])
            data['dssf_global_zz'] = np.interp(w_common, data['w'], data['dssf_global_zz'])
            data['w'] = w_common
```

#### `create_comparison_plots()` (in `create_comparison_plots.py`)
Same interpolation logic added to ensure all data can be plotted together.

## Benefits

1. **Robustness**: Scripts now work with any combination of frequency grids
2. **Automatic**: No manual intervention required
3. **Informative**: Warns users when interpolation is performed
4. **Preserves Data**: Uses the most detailed grid (longest array) as reference
5. **Smooth**: Linear interpolation maintains smooth spectral features

## Performance
- Minimal overhead: Interpolation is only performed when needed
- Fast: `np.interp()` is optimized for 1D interpolation
- One-time cost: Interpolation done once during data loading

## Testing
The fix handles various scenarios:
- All grids same size: No interpolation, original behavior
- Some grids different: Only different ones interpolated
- All grids different: All interpolated to common reference
- Large size differences: Uses longest grid to preserve detail

## Files Modified
- `reader_pyrochlore.py` - Main animation function
- `animate_DSSF_pyrochlore.py` - Standalone animation script
- `create_comparison_plots.py` - Comparison plotting script

## Usage
No changes needed! The fix is automatic and transparent to users.

```bash
# Works seamlessly now, even with varying frequency grids
python animate_DSSF_pyrochlore.py /path/to/data
```

## Future Enhancements
Potential improvements:
- Option to choose reference grid (smallest, largest, or specific field)
- Spline interpolation for higher accuracy
- Warning if interpolation error exceeds threshold
- Option to skip fields with very different grids
