# Quick Start Guide - Parallel Active Learning

## ✅ Fixed: scikit-learn Warning

The scikit-learn warning has been resolved. The packages are installed in the virtual environment at `pystandard/`.

## Running Active Learning

### Option 1: Using the Wrapper Script (Recommended)

The wrapper script automatically activates the virtual environment:

```bash
# Basic parallel run (uses all 32 CPUs on your system)
./util/classical_solvers/run_parallel_active_learning.sh \
    --n-initial 100 \
    --n-iterations 20 \
    --n-jobs -1

# Custom number of workers
./util/classical_solvers/run_parallel_active_learning.sh \
    --n-initial 500 \
    --n-iterations 30 \
    --n-per-iteration 50 \
    --n-jobs 16 \
    --output-dir my_exploration
```

### Option 2: Manual Virtual Environment Activation

```bash
# Activate virtual environment
source pystandard/bin/activate

# Run active learning
python util/classical_solvers/run_active_learning.py \
    --n-initial 100 \
    --n-iterations 20 \
    --n-jobs 16

# Deactivate when done
deactivate
```

## Your System Specs

- **CPUs**: 32 cores
- **Recommended `--n-jobs`**: 16-24 (for best efficiency)
- **Expected speedup**: 12-18x for large runs

## Example Commands for Your System

### Fast Exploration (500 sims in ~5 minutes)
```bash
./util/classical_solvers/run_parallel_active_learning.sh \
    --n-initial 100 \
    --n-iterations 8 \
    --n-per-iteration 50 \
    --fast-mode \
    --n-jobs 20
```

### Large Exploration (2000 sims in ~20 minutes)
```bash
./util/classical_solvers/run_parallel_active_learning.sh \
    --n-initial 500 \
    --n-iterations 30 \
    --n-per-iteration 50 \
    --fast-mode \
    --n-jobs 24 \
    --strategy balanced
```

### Accurate Mode (slower but more reliable)
```bash
./util/classical_solvers/run_parallel_active_learning.sh \
    --n-initial 200 \
    --n-iterations 15 \
    --accurate \
    --lattice-size 36 \
    --n-jobs 16
```

## Performance Estimates (Your System)

| Mode | Time/sim | 20 workers | 1000 sims |
|------|----------|------------|-----------|
| **Fast** | ~15s | ~0.8s | **13 min** |
| **Accurate** | ~90s | ~4.5s | **75 min** |
| **Screening** | ~3s | ~0.15s | **2.5 min** |

## Installed Packages

- ✅ scikit-learn 1.8.0
- ✅ tqdm 4.67.1 (progress bars)
- ✅ numpy 2.3.5
- ✅ scipy 1.16.3
- ✅ joblib 1.5.3

## Troubleshooting

### No warnings anymore!
The virtual environment has all required packages. Just use the wrapper script or activate manually.

### Testing Parallel Performance
```bash
source pystandard/bin/activate
python util/classical_solvers/test_parallel.py
```

### Resuming Interrupted Runs
Progress is automatically saved every 10 simulations. Just re-run the same command - already completed points are skipped.
