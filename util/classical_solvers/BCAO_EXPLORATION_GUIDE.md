# BCAO Honeycomb Phase Exploration Guide

An active learning framework for discovering magnetic phases in the BCAO honeycomb model by systematically exploring the 7-dimensional parameter space.

## Overview

This tool uses Monte Carlo simulated annealing combined with machine learning to efficiently explore the parameter space and identify distinct magnetic phases based on their Static Spin Structure Factor (SSSF).

### Phases Detected

| Phase | Description | SSSF Signature |
|-------|-------------|----------------|
| Ferromagnetic (FM) | All spins aligned | Peak at Γ point (0, 0) |
| Antiferromagnetic (AFM) | Sublattices antiparallel | Peak at Γ, staggered magnetization |
| Zigzag | Stripy/zigzag order | Peak at M point (0.5, 0) |
| 120° Order | Three-sublattice order | Peak at K point (1/3, 1/3) |
| Incommensurate | Spiral order | Peak between high-symmetry points |
| Double-Q Commensurate | Two rational Q-vectors | Two peaks at commensurate (p/q) positions |
| Double-Q Incommensurate | Two incommensurate Q-vectors | Two peaks, not perpendicular in RLU |
| **Double-Q Meron-Antimeron** | Topological vortex lattice | Two **perpendicular in RLU** incomm. Q-vectors |
| Triple-Q Commensurate | Three rational Q-vectors | Three peaks at commensurate positions |
| **Triple-Q Skyrmion** | Magnetic skyrmion lattice | Three Q-vectors at **120° angles** |
| Triple-Q Incommensurate | Three incommensurate Q-vectors | Three peaks, general pattern |

---

## Quick Start

### 1. Basic Training Run (~30 minutes)

```bash
cd /home/pc_linux/ClassicalSpin_Cpp
source pystandard/bin/activate

python3 util/classical_solvers/run_active_learning.py \
    --screening \
    --n-initial 100 \
    --n-iterations 30 \
    --n-per-iteration 10 \
    --output-dir ./my_exploration \
    --strategy balanced \
    --seed 42
```

### 2. Analyze Results

```bash
python3 util/classical_solvers/analyze_exploration.py ./my_exploration
```

This generates:
- `phase_distribution.png` - Pie chart of discovered phases
- `parameter_scatter.png` - 6 parameter-pair scatter plots
- `feature_importance.png` - Which parameters matter most
- `exploration_timeline.png` - Phase discoveries over time
- `q_vector_distribution.png` - Q-vectors in reciprocal space
- `energy_landscape.png` - Energy vs phase/parameters
- `summary_report.txt` - Text summary

---

## Command Line Options

### Simulation Quality

| Option | Description | Time per sim | Lattice Size |
|--------|-------------|--------------|--------------|
| `--screening` | Ultra-fast screening | ~3.5s | L=8 |
| `--fast` | Quick exploration | ~15s | L=12 |
| `--accurate` | High quality | ~90s | L=24 |
| (default) | Standard | ~30s | L=16 |

### Exploration Parameters

```bash
--n-initial N       # Initial Latin Hypercube samples (default: 50)
--n-iterations N    # Active learning iterations (default: 20)
--n-per-iteration N # Points per iteration (default: 5)
--output-dir DIR    # Output directory
--seed N            # Random seed for reproducibility
```

### Acquisition Strategy

```bash
--strategy balanced     # Balance exploration and exploitation (default)
--strategy uncertainty  # Focus on uncertain regions
--strategy exploration  # Maximize coverage
--strategy exploitation # Focus on rare phases
```

---

## Parameter Space

The exploration covers 7 normalized parameters (all divided by |J1xy|):

| Parameter | Range | Physical Meaning |
|-----------|-------|------------------|
| J1z | [-1, 1] | First-neighbor z-coupling (XXZ anisotropy) |
| D | [-1, 1] | DM-like antisymmetric exchange |
| E | [-1, 1] | Bond-dependent symmetric exchange |
| F | [-1, 1] | Off-diagonal exchange |
| G | [-1, 1] | Off-diagonal exchange |
| J3xy | [-1, 1] | Third-neighbor xy-coupling |
| J3z | [-1, 1] | Third-neighbor z-coupling |

The reference scale |J1xy| is fixed at 6.0 (from BCAO experiments).

---

## Output Files

After training, your output directory contains:

```
my_exploration/
├── exploration_history.json   # Full exploration data
├── surrogate_model.pkl        # Trained Random Forest model
├── phase_distribution.png     # Pie chart
├── parameter_scatter.png      # Parameter space plots
├── feature_importance.png     # Feature importance
├── exploration_timeline.png   # Discovery timeline
├── q_vector_distribution.png  # Reciprocal space
├── energy_landscape.png       # Energy landscape
└── summary_report.txt         # Text summary
```

### Using the Saved Model

```python
import pickle
import numpy as np

# Load model
with open('my_exploration/surrogate_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']           # RandomForestClassifier
label_encoder = model_data['label_encoder']  # Phase labels

# Predict phase for new parameters
# Order: [J1z, D, E, F, G, J3xy, J3z] (normalized)
params = np.array([[-0.3, 0.1, 0.2, -0.01, 0.09, 0.27, -0.02]])
phase_idx = model.predict(params)
phase_name = label_encoder.inverse_transform(phase_idx)
print(f"Predicted phase: {phase_name[0]}")

# Get prediction probabilities
probs = model.predict_proba(params)
for i, phase in enumerate(label_encoder.classes_):
    print(f"  {phase}: {probs[0][i]*100:.1f}%")
```

---

## Understanding the Classification

### How Phases are Identified

1. **Run Monte Carlo** simulation with simulated annealing
2. **Compute SSSF** $S(\mathbf{Q}) = \sum_{ij} e^{i\mathbf{Q}\cdot(\mathbf{r}_i-\mathbf{r}_j)} \mathbf{S}_i \cdot \mathbf{S}_j$
3. **Find peaks** in S(Q) via local maximum detection
4. **Classify Q-vectors** by proximity to Γ, M, K points
5. **Detect double-Q** if secondary peak > 30% of primary
6. **Check perpendicularity** for meron-antimeron

### Double-Q Meron-Antimeron Detection

A configuration is classified as meron-antimeron when:
- ✅ Two significant SSSF peaks (intensity ratio > 0.3)
- ✅ Both Q-vectors are incommensurate (not at Γ, M, or K)
- ✅ Q-vectors are approximately perpendicular (cos θ < 0.3)

---

## Example Workflows

### 1. Quick Phase Survey

```bash
# Fast survey with minimal resources
python3 util/classical_solvers/run_active_learning.py \
    --screening \
    --n-initial 50 \
    --n-iterations 10 \
    --output-dir ./quick_survey
```

### 2. Focused Search for Rare Phases

```bash
# Focus on finding Double-Q Meron-Antimeron
python3 util/classical_solvers/run_active_learning.py \
    --fast \
    --n-initial 100 \
    --n-iterations 50 \
    --n-per-iteration 10 \
    --strategy exploitation \
    --output-dir ./meron_search
```

### 3. High-Quality Exploration

```bash
# Overnight run for publication-quality data
python3 util/classical_solvers/run_active_learning.py \
    --accurate \
    --n-initial 200 \
    --n-iterations 100 \
    --n-per-iteration 20 \
    --output-dir ./full_exploration \
    --seed 12345
```

### 4. Continue from Previous Run

```python
# Load previous results and add more points
import json

with open('my_exploration/exploration_history.json') as f:
    data = json.load(f)

# Extract Meron-Antimeron parameter regions
meron_points = [p for p in data['points'] 
                if p['phase'] == 'Double-Q Meron-Antimeron']

for pt in meron_points:
    params = pt['params']
    print(f"J1z={params['J1z_norm']:.3f}, D={params['D_norm']:.3f}, ...")
```

---

## Feature Importance Interpretation

The Random Forest model provides feature importance, telling you which parameters most strongly influence phase transitions:

| Rank | Parameter | Typical Importance | Physical Meaning |
|------|-----------|-------------------|------------------|
| 1 | J3xy | 0.25 | Third-neighbor frustration, stabilizes incommensurate order |
| 2 | J3z | 0.19 | z-anisotropy of third-neighbor |
| 3 | J1z | 0.17 | XXZ anisotropy (Ising vs XY character) |
| 4 | G | 0.10 | Off-diagonal exchange |
| 5 | F | 0.10 | Off-diagonal exchange |
| 6 | E | 0.10 | Bond-dependent exchange |
| 7 | D | 0.09 | DM-like interaction |

**Key insight**: Third-neighbor interactions (J3xy, J3z) are the most important for determining phase boundaries in this model.

---

## Known Seeds

The exploration starts with 3 known parameter sets that produce Double-Q Meron-Antimeron lattices:

| Seed | Source | Key Parameters |
|------|--------|----------------|
| fitting_param_2 | Experiment | J1z≈-0.28, J3xy≈0.27 |
| fitting_param_4 | Experiment | J1z≈-0.31, J3xy≈0.26 |
| fitting_param_8 | Experiment | J1z≈-0.31, J3xy≈0.26 |

These serve as anchors to guide the exploration toward the meron-antimeron region.

---

## Troubleshooting

### Spin Solver Timeout
If simulations timeout, try:
```bash
--screening  # Use faster simulation mode
```

### Not Finding Rare Phases
Try:
```bash
--strategy exploitation  # Focus on rare phases
--n-iterations 100       # More iterations
```

### Memory Issues
Reduce batch size:
```bash
--n-per-iteration 5
```

### Reproducibility
Always set a seed:
```bash
--seed 42
```

---

## File Locations

| File | Purpose |
|------|---------|
| `util/classical_solvers/run_active_learning.py` | Main entry point |
| `util/classical_solvers/active_learning_explorer.py` | Core exploration logic |
| `util/classical_solvers/phase_classifier.py` | SSSF-based classification |
| `util/classical_solvers/spin_solver_runner.py` | C++ solver interface |
| `util/classical_solvers/analyze_exploration.py` | Visualization & analysis |
| `build/spin_solver` | Compiled Monte Carlo solver |

---

## Citation

If you use this tool, please cite:
- The BCAO experimental papers for the parameter values
- This active learning framework for the exploration methodology
