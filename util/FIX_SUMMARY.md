# Fix Applied: Inhomogeneous Frequency Grid Handling

## Problem Solved
The animation scripts were failing with a `ValueError` when trying to stack DSSF data arrays of different lengths. This occurred because different field directories had frequency grids of different sizes (e.g., 239 points vs 479 points).

## Solution Implemented
Added automatic interpolation to a common frequency grid when different-sized arrays are detected.

## Changes Made

### 1. `reader_pyrochlore.py`
Modified `animate_DSSF_spinon_photon()` function to:
- Detect when frequency grids have different lengths
- Interpolate all data to the longest frequency grid (preserves maximum detail)
- Print informative warnings when interpolation is performed

### 2. `animate_DSSF_pyrochlore.py`
Applied the same fix to the standalone animation script.

### 3. `create_comparison_plots.py`
Applied the same fix to ensure comparison plots work with varying grids.

### 4. `BUGFIX_INHOMOGENEOUS_GRIDS.md`
Created comprehensive documentation of the issue and solution.

## How It Works

```python
# Before stacking arrays, check if they're all the same length
w_lengths = [len(d['w']) for d in field_data]

if len(set(w_lengths)) > 1:
    # Different lengths detected - interpolate to common grid
    max_idx = np.argmax(w_lengths)
    w_common = field_data[max_idx]['w']  # Use longest as reference
    
    for data in field_data:
        if len(data['w']) != len(w_common):
            # Interpolate all DSSF components to common grid
            data['dssf_local_spinon_photon'] = np.interp(w_common, data['w'], data['dssf_local_spinon_photon'])
            # ... (same for all other components)
            data['w'] = w_common
```

## Benefits
✅ Handles any combination of frequency grid sizes  
✅ Automatic - no user intervention needed  
✅ Preserves maximum detail (uses longest grid)  
✅ Fast linear interpolation  
✅ Informative warnings  
✅ Backward compatible (no-op when all grids match)  

## Testing
The fix should now allow animations to be created successfully:

```bash
# Should now work without errors
python animate_DSSF_pyrochlore.py /path/to/data
```

## Expected Output
When different grid sizes are detected, you'll see:
```
[animate_DSSF_spinon_photon] Warning: Different frequency grid sizes detected: {239, 479}
[animate_DSSF_spinon_photon] Interpolating all data to common frequency grid...
```

Then animation creation proceeds normally.

## Notes
- Uses numpy's `interp()` for 1D linear interpolation
- Chooses the longest grid as reference to preserve maximum frequency resolution
- All 6 DSSF components are interpolated consistently
- Minimal performance impact (interpolation is fast)

## Status
✅ **FIXED** - The inhomogeneous array error should no longer occur.
