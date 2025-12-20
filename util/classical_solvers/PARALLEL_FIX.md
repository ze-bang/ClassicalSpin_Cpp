# Parallel Execution Fix Summary

## Problem

When running active learning with `--n-jobs=16`, simulations were:
1. Completing extremely fast (~0.1s) 
2. Timing out despite 300 second timeout
3. Producing mostly "disorder" or "unknown" classifications

## Root Cause

**The simulation function could not be pickled for multiprocessing.**

The function returned by `create_spin_solver_simulation_func()` was a nested closure that captured the `SpinSolverRunner` instance. Python's multiprocessing requires pickling objects to send them to worker processes, but nested functions with closures cannot be pickled.

When ProcessPoolExecutor tried to send the simulation function to workers, it failed immediately with a pickling error. However, this error was being caught and converted to `None`, making it appear as if simulations ran but failed.

## Solution

Refactored the code to use a **picklable class-based callable**:

### 1. Created `_PicklableSimulation` class
- Stores configuration as a dictionary (picklable)
- Lazy-initializes the `SpinSolverRunner` in each worker process
- Implements `__call__` to act like a function
- Implements `__getstate__` and `__setstate__` for proper pickling

### 2. Updated `create_spin_solver_simulation_func()`
- Returns `_PicklableSimulation` instance instead of nested function
- Configuration is stored as dictionary, not live objects

### 3. Improved error reporting
- Added stderr output in worker processes (visible in parallel runs)
- Added timing checks to detect suspiciously fast runs
- Better error messages showing command, working directory, stdout/stderr
- Distinguished between timeouts, errors, and parsing failures

## Changes Made

### Files Modified:
1. **spin_solver_runner.py**
   - Added `_PicklableSimulation` class (lines 553-614)
   - Refactored `create_spin_solver_simulation_func()` to return picklable callable
   - Enhanced error reporting and timing diagnostics
   - Added validation for executable permissions

2. **active_learning_explorer.py**
   - Improved error reporting in `_run_simulation_safe()` worker function
   - Added stderr logging for visibility in parallel execution
   - Better distinction between timeout and other errors

### Test Files Added:
- `test_parallel_debug.py` - Sequential diagnostics
- `test_parallel_real.py` - Parallel execution test matching actual active learning usage

## Verification

Test results with 8 simulations, 4 workers, L=8 lattice:
```
Total time: 4.60s
Successes: 8/8
Failures: 0/8
Avg time per sim: 0.57s
Success rate: 100.0%
```

## Usage

The API remains the same - no changes needed to calling code:

```python
# Still works exactly as before
sim_func = create_fast_simulation_func(L=12, verbose=True)
spins, positions, energy = sim_func(params)

# Now also works in parallel!
with ProcessPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(sim_func, p) for p in params_list]
    results = [f.result() for f in futures]
```

## Performance Notes

Expected timings for different configurations:

| Lattice | Steps      | Sequential | Parallel (16 cores) |
|---------|------------|------------|---------------------|
| L=8     | 500/100    | ~2-3s      | ~0.3s per sim       |
| L=12    | 1000/200   | ~3-4s      | ~0.5s per sim       |
| L=24    | 50k/5k     | ~60-90s    | ~10-15s per sim     |

## Troubleshooting

If you still see failures:

1. **Check stderr output** - errors now print to stderr in worker processes
2. **Run test script**: `python3 test_parallel_real.py`
3. **Check solver exists**: `ls -lh ../../build/spin_solver`
4. **Try verbose mode**: Set `verbose=True` in simulation config
5. **Reduce parallelism**: Try `--n-jobs=4` first to see if it's a resource issue
