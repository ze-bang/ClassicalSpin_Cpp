# Simulation Diagnostics Report

## Summary of Issues

Your exploration shows:
- **Timeout rate: 57.6%** (752/1306 simulations)
- **Disordered rate: 21.1%** (276/1306 simulations)
- **Success rate: Only ~21%**

This indicates **inadequate simulation parameters** for the parameter space being explored.

## Root Causes Identified

### 1. **High Frustration Regions**
- **60% of timeouts** (451/752) have `|J3xy| > 0.3`
  - Strong third-neighbor coupling creates long-range frustration
  - Requires many more MC steps to equilibrate

- **70% of timeouts** (525/752) have `|E| > 0.3`  
  - Strong off-diagonal exchange breaks symmetries
  - Creates complex energy landscapes with many local minima

### 2. **Fast Mode Too Aggressive**
Current settings (fast mode):
```python
annealing_steps = 2000
n_deterministics = 500
timeout = 30s
```

This is insufficient for:
- Frustrated systems (need 5-10x more steps)
- Complex phases like meron-antimeron
- Parameter regions near phase boundaries

### 3. **Disordered Results**
21% disordered suggests:
- Not enough high-T equilibration (trapped in local minima)
- Competing interactions without clear ground state
- Need longer relaxation at each temperature

## Recommendations

### Option 1: **Use Accurate Mode** (Recommended)
```bash
./util/classical_solvers/run_parallel_active_learning.sh \
    --n-initial 500 \
    --n-iterations 30 \
    --n-per-iteration 50 \
    --accurate \              # ← Use this instead of --fast-mode
    --lattice-size 24 \
    --n-jobs 16 \
    --output-dir my_exploration_accurate
```

**Accurate mode settings:**
- `annealing_steps = 20000` (10x more)
- `n_deterministics = 5000` (10x more)
- `L = 24` (larger lattice)
- `timeout = 300s` (10x longer)

**Expected impact:**
- Timeout rate: 57% → ~10-15%
- Disordered rate: 21% → ~5-10%
- Time per simulation: 15s → 90s
- With 16 workers: Still only ~6s effective time per simulation

### Option 2: **Custom Improved Fast Mode**

Modify `spin_solver_runner.py`:

```python
def create_improved_fast_simulation_func(
    L: int = 12,
    annealing_steps: int = 5000,      # 2.5x more
    n_deterministics: int = 1500,     # 3x more
    T_start: float = 10.0,            # Higher start (better randomization)
    cooling_rate: float = 0.93,       # Slower cooling
    overrelaxation_rate: int = 20,    # More steps per T
    **kwargs
):
    return create_spin_solver_simulation_func(
        L=L,
        annealing_steps=annealing_steps,
        n_deterministics=n_deterministics,
        T_start=T_start,
        cooling_rate=cooling_rate,
        timeout=120,  # Longer timeout
        **kwargs
    )
```

**Expected impact:**
- Timeout rate: 57% → ~20-25%
- Time per simulation: 15s → 35s
- Better balance of speed vs quality

### Option 3: **Adaptive Strategy**

For regions with high J3xy or E, automatically use more steps:

```python
def adaptive_simulation(params):
    # Check if parameters suggest frustration
    if abs(params.J3xy_norm) > 0.3 or abs(params.E_norm) > 0.3:
        # Use accurate mode for hard cases
        return accurate_sim_func(params)
    else:
        # Use fast mode for easy cases
        return fast_sim_func(params)
```

## Detailed Parameter Adjustments

### Current (Fast Mode):
| Parameter | Current | Issue |
|-----------|---------|-------|
| annealing_steps | 2000 | Too few for frustrated systems |
| n_deterministics | 500 | Insufficient T=0 relaxation |
| T_start | 5.0 | May trap in local minima |
| T_end | 0.001 | Good |
| cooling_rate | 0.9 | Too fast |
| timeout | 30s | Too short |

### Recommended (Improved):
| Parameter | Improved | Justification |
|-----------|----------|---------------|
| annealing_steps | 5000-10000 | Better equilibration at each T |
| n_deterministics | 1500-3000 | Thorough T=0 refinement |
| T_start | 10.0 | Escape local minima |
| T_end | 0.001 | Keep |
| cooling_rate | 0.93-0.95 | Slower, more careful cooling |
| timeout | 120-180s | Allow time to finish |
| overrelaxation_rate | 20 | More updates per T |

## Quick Fix for Current Run

If you want to continue with existing data but improve future points:

```bash
# Stop current run (Ctrl+C)

# Resume with accurate mode
./util/classical_solvers/run_parallel_active_learning.sh \
    --output-dir ./my_exploration \    # Same directory!
    --n-initial 500 \                   # Will skip already completed
    --n-iterations 30 \
    --n-per-iteration 50 \
    --accurate \                        # Better parameters
    --n-jobs 16
```

The explorer automatically skips already-classified points, so this will only run simulations for remaining pending points with better settings.

## Monitoring Improvements

To verify improvements are working, after restarting with new settings:

```bash
# Check progress periodically
python util/classical_solvers/analyze_simulation_diagnostics.py \
    --exploration-dir ./my_exploration
```

**Target metrics:**
- Timeout rate: < 15%
- Disordered rate: < 10%
- Total success rate: > 75%

## Understanding the Trade-offs

| Mode | Time/sim | Timeout % | Total Time (2000 sims, 16 workers) |
|------|----------|-----------|-------------------------------------|
| **Fast (current)** | 15s | 57% | 30 min | 
| **Improved** | 35s | 20-25% | 75 min |
| **Accurate** | 90s | 10-15% | 3 hours |

**Recommendation:** Use **Accurate mode** with parallel processing. Even at 90s per simulation, with 16 workers you get effective ~6s per simulation, completing 2000 simulations in ~3 hours with high quality results.

## Implementation

### Step 1: Stop current run
```bash
# Press Ctrl+C if still running
```

### Step 2: Restart with accurate mode
```bash
cd /home/pc_linux/ClassicalSpin_Cpp

./util/classical_solvers/run_parallel_active_learning.sh \
    --output-dir ./my_exploration_accurate \
    --n-initial 500 \
    --n-iterations 30 \
    --n-per-iteration 50 \
    --accurate \
    --lattice-size 24 \
    --strategy balanced \
    --n-jobs 16 \
    --seed 42
```

### Step 3: Monitor progress
```bash
# In another terminal, check periodically:
watch -n 60 'python util/classical_solvers/analyze_simulation_diagnostics.py --exploration-dir ./my_exploration_accurate | grep -A 10 "Phase Distribution"'
```

This will dramatically improve your success rate while still completing in reasonable time with parallelization!
