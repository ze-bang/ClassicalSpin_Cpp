# Bug Fix: Skip Output/Metadata Directories

## Issue
The processing and animation scripts were trying to process output directories (like `animations`, `comparison_plots`, `results`) as if they were field data directories, causing errors:

```
RuntimeError: No simulation data could be processed in /scratch/zhouzb79/MD_pi_flux/animations.
```

## Root Cause
When `read_MD_pi_flux_all()` and `collect_DSSF_data_all_fields()` iterate over all directories in the root folder, they would encounter directories created by previous runs:
- `animations/` - Created by animation scripts
- `comparison_plots/` - Created by comparison plot scripts  
- `results/` - Created by processing scripts
- `__pycache__/` - Created by Python
- `.git/` - Git repository metadata

These directories don't contain field data and should be skipped.

## Solution
Added a `skip_dirs` set to explicitly exclude known output/metadata directories from processing.

## Implementation

### Modified Functions

#### `read_MD_pi_flux_all()` in `reader_pyrochlore.py`
```python
# Directories to skip (output/metadata directories, not field data)
skip_dirs = {'animations', 'comparison_plots', 'results', '__pycache__', '.git'}

field_dirs = sorted(
    entry for entry in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, entry)) and entry not in skip_dirs
)
```

Also added error handling to continue processing other directories if one fails:
```python
try:
    read_MD_int(field_path, mag, SSSFGraph, run_ids=run_ids)
except RuntimeError as e:
    if verbose:
        print(f"[read_MD_pi_flux_all] Warning: Skipping {field_dir}: {e}")
    continue
```

#### `collect_DSSF_data_all_fields()` 
Applied the same fix in:
- `reader_pyrochlore.py`
- `animate_DSSF_pyrochlore.py`
- `create_comparison_plots.py`

## Benefits
✅ No longer tries to process output directories  
✅ Continues processing even if one field directory fails  
✅ Provides informative warnings when skipping directories  
✅ Clean separation of data vs. output directories  
✅ Works with existing directory structures  

## Directories Skipped
- `animations` - Animation output
- `comparison_plots` - Comparison plot output
- `results` - Processing results (when at root level)
- `__pycache__` - Python bytecode cache
- `.git` - Git repository metadata

## Usage
No changes needed - the fix is automatic!

```bash
# Now works even if animations/ directory exists
python reader_pyrochlore.py /path/to/data --mag HnHn
python animate_DSSF_pyrochlore.py /path/to/data
```

## Files Modified
- ✅ `reader_pyrochlore.py` - Both functions updated
- ✅ `animate_DSSF_pyrochlore.py` - Collection function updated
- ✅ `create_comparison_plots.py` - Collection function updated

## Testing
The scripts now correctly:
1. Skip `animations/` and other output directories
2. Process only actual field directories (field_0.0, h=0.1, etc.)
3. Continue processing if one field directory has issues
4. Provide clear warnings about skipped directories

## Future Extensions
If you create other output directories, add them to `skip_dirs`:
```python
skip_dirs = {'animations', 'comparison_plots', 'results', '__pycache__', '.git', 'your_new_dir'}
```

## Status
✅ **FIXED** - Output directories are now properly skipped.
