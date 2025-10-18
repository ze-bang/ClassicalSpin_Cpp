# Complete Guide: DSSF Animation and Visualization Tools

## Quick Start

### 1. Process your MD simulation data
```bash
python reader_pyrochlore.py /path/to/MD_pi_flux_root --mag HnHn
```

### 2. Create animations
```bash
python animate_DSSF_pyrochlore.py /path/to/MD_pi_flux_root --fps 5
```

### 3. Create comparison plots (optional)
```bash
python create_comparison_plots.py /path/to/MD_pi_flux_root --max-fields 10
```

## Detailed Workflow

### Step 1: Organize Your Data

Ensure your directory structure looks like this:

```
MD_pi_flux_root/
├── field_0.0/          # or h=0.0, B=0.0, etc.
│   ├── 0/              # Run 0
│   │   ├── pos.txt
│   │   ├── Time_steps.txt
│   │   ├── spin_t.txt
│   │   └── spin_0.txt
│   ├── 1/              # Run 1
│   │   └── ...
│   └── results/        # Created by processing
│       ├── DSSF_local_xx.txt
│       ├── DSSF_local_zz.txt
│       ├── DSSF_global_xx.txt
│       └── DSSF_global_zz.txt
├── field_0.1/
│   └── ...
└── field_0.2/
    └── ...
```

### Step 2: Process Raw Data (if not already done)

Process all field directories to generate DSSF files:

```bash
python reader_pyrochlore.py /path/to/MD_pi_flux_root --mag HnHn
```

Options:
- `--mag`: Magnetization direction (HnHn, 001, 110, 111, 1-10)
- `--quiet`: Suppress progress messages
- `--run-ids`: Process only specific run IDs (e.g., `--run-ids 0 1 2`)

This will create `results/` directories in each field folder with processed DSSF data.

### Step 3: Create Animations

#### Option A: Standalone animation script (recommended)
```bash
python animate_DSSF_pyrochlore.py /path/to/MD_pi_flux_root
```

Advanced options:
```bash
python animate_DSSF_pyrochlore.py /path/to/MD_pi_flux_root \
    --output /custom/output/dir \
    --fps 10 \
    --energy-conversion 0.063
```

#### Option B: Using main reader script
```bash
python reader_pyrochlore.py /path/to/MD_pi_flux_root --animate --animation-fps 10
```

### Step 4: Create Static Comparison Plots (optional)

For publications or quick overview:

```bash
python create_comparison_plots.py /path/to/MD_pi_flux_root --max-fields 10
```

This creates:
- **Stacked comparison plots**: Shows multiple field strengths in one figure
- **2D heatmaps**: Color map of DSSF vs. energy vs. field strength

## Output Files

### Animations (in `animations/` directory)
- `DSSF_spinon_photon_local.gif`: Local channel animation
- `DSSF_spinon_photon_global.gif`: Global channel animation

### Comparison Plots (in `comparison_plots/` directory)
- `DSSF_comparison_local.pdf/png`: Stacked plots for local channel
- `DSSF_comparison_global.pdf/png`: Stacked plots for global channel
- `DSSF_heatmap.pdf/png`: 2D heatmap visualization

## Animation Features

Each animation shows:
- **Red solid line**: Total spinon+photon signal (S_xx + S_zz)
- **Blue dashed line**: S_xx component
- **Green dashed line**: S_zz component
- **Title**: Current field strength
- **X-axis**: Energy ω in meV
- **Y-axis**: DSSF intensity

### Why S_xx + S_zz?

In pyrochlore quantum spin ice systems:
- S_xx and S_zz couple to photons in the quantum spin ice Coulomb phase
- Their sum represents the "photon-like" or "spinon+photon" excitations
- This is the key signature for observing these exotic quasiparticles

## Examples

### Example 1: Basic workflow
```bash
# Process data for all fields
python reader_pyrochlore.py /scratch/my_simulation --mag HnHn

# Create animations with default settings
python animate_DSSF_pyrochlore.py /scratch/my_simulation

# Results in: /scratch/my_simulation/animations/
```

### Example 2: High-quality animations
```bash
# Create smooth animations with more frames per second
python animate_DSSF_pyrochlore.py /scratch/my_simulation \
    --fps 15 \
    --output ~/presentations/animations
```

### Example 3: Subset of runs
```bash
# Process only specific runs (e.g., converged ones)
python reader_pyrochlore.py /scratch/my_simulation \
    --mag HnHn \
    --run-ids 0 1 2 3 4

# Then create animations
python animate_DSSF_pyrochlore.py /scratch/my_simulation
```

### Example 4: Publication-ready figures
```bash
# Create static comparison plots
python create_comparison_plots.py /scratch/my_simulation \
    --max-fields 8 \
    --output ~/paper_figures

# This gives you high-resolution PDFs for publication
```

### Example 5: Programmatic usage
```python
from reader_pyrochlore import animate_DSSF_spinon_photon, collect_DSSF_data_all_fields

# Collect data
data = collect_DSSF_data_all_fields("/path/to/root", verbose=True)

# Inspect
for d in data:
    print(f"Field {d['field']}: max intensity = {d['dssf_local_spinon_photon'].max()}")

# Create animations
animate_DSSF_spinon_photon("/path/to/root", fps=10)
```

## Troubleshooting

### "No field directories found"
- Check that your directory names contain "field", "h", or "B" followed by a number
- Examples: `field_0.1`, `h=0.5`, `B_1.0`

### "Missing DSSF files"
- Run `python reader_pyrochlore.py /path/to/root` first to process raw data
- Check that each field directory has simulation output files

### "Import errors"
```bash
pip install numpy matplotlib scipy opt_einsum Pillow h5py
```

### Animations too slow/fast
- Adjust FPS: `--fps 10` for smoother (slower playback)
- Adjust FPS: `--fps 3` for faster playback

### Too many fields in comparison plot
- Use `--max-fields` to limit: `--max-fields 5`
- Script will automatically subsample evenly

### Different energy units
- Adjust conversion factor: `--energy-conversion 0.1` (depends on your Jzz value)

## Performance Tips

1. **Large datasets**: Process fields in parallel using separate terminal sessions
2. **Memory issues**: Process fields one at a time with `--run-ids`
3. **Quick preview**: Use `create_comparison_plots.py` before full animation
4. **Storage**: GIFs can be large; reduce FPS or convert to MP4 if needed

## Advanced: Customization

### Modify animation appearance
Edit `animate_DSSF_pyrochlore.py`:
- Change colors: Modify `'r-'`, `'b--'`, `'g--'` in plot commands
- Change figure size: Modify `figsize=(10, 6)`
- Change labels: Edit `xlabel`, `ylabel`, `title` strings

### Add more components
To animate other DSSF components (S_xy, S_yy, etc.), modify `collect_DSSF_data_all_fields` to read additional files and update animation functions accordingly.

### Export to video
```python
# In animate_DSSF_pyrochlore.py, change:
# from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Then change:
# anim.save(output_file, writer=PillowWriter(fps=fps))
anim.save(output_file.replace('.gif', '.mp4'), writer=FFMpegWriter(fps=fps))
```

## Integration with Other Tools

### Use with Jupyter notebooks
```python
from IPython.display import Image
from reader_pyrochlore import animate_DSSF_spinon_photon

# Create animations
local_gif, global_gif = animate_DSSF_spinon_photon("/path/to/root")

# Display in notebook
Image(filename=local_gif)
```

### Batch processing multiple datasets
```bash
#!/bin/bash
for dataset in /scratch/simulations/*/; do
    echo "Processing $dataset"
    python reader_pyrochlore.py "$dataset" --mag HnHn
    python animate_DSSF_pyrochlore.py "$dataset"
done
```

## Summary of Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `reader_pyrochlore.py` | Main processing script | DSSF data files |
| `animate_DSSF_pyrochlore.py` | Create animations | GIF animations |
| `create_comparison_plots.py` | Static comparison plots | PDF/PNG figures |
| `example_animations.py` | Usage examples and tests | None (demonstration) |

## Support and Documentation

- Full documentation: `README_ANIMATIONS.md`
- Feature summary: `ANIMATION_FEATURES_SUMMARY.md`
- Example code: `example_animations.py`
- This guide: `USAGE_GUIDE.md`

## Citation

If you use these animation tools in your research, please cite:
- The original ClassicalSpin_Cpp simulation code
- Any relevant publications for the physics (quantum spin ice, spinon-photon coupling, etc.)
