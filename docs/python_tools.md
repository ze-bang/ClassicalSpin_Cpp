# Python analysis toolkit

The `util/` directory contains companion scripts for post-processing the binary and text outputs produced by the C++ drivers. Install NumPy, SciPy, and Matplotlib in your Python environment before running the tools described here.【F:util/plot_convergence.py†L1-L159】【F:util/single_q.py†L1-L160】

## Readers and geometry helpers

- **`reader_pyrochlore.py`** assembles common high-symmetry paths through the pyrochlore Brillouin zone, defines local coordinate frames, and exposes Fourier transforms that convert real-space spin textures into reciprocal-space observables.【F:util/reader_pyrochlore.py†L1-L123】 Use it when plotting dynamical structure factors or when interpolating Monte Carlo snapshots onto experimental momentum cuts.

## Diagnostics and visualisation

- **`plot_convergence.py`** reads `convergence_data.txt` files emitted by enhanced simulated annealing runs and generates publication-ready plots for energy, acceptance rates, configuration stability, and rolling variances. It also prints convergence statistics that make it easy to judge whether additional sweeps are needed.【F:util/plot_convergence.py†L7-L128】
- **`example_animations.py`** demonstrates how to load spin trajectories and animate them frame-by-frame, providing a template for creating quick visual sanity checks before launching long scans.【F:util/example_animations.py†L1-L144】

## Variational ansätze and scans

- **`single_q.py`** minimises a single-$Q$ ansatz on the honeycomb lattice by constructing Luttinger–Tisza interaction matrices and optimising nine angular/weight parameters with SciPy. The class reports the optimal energy and classifies the resulting magnetic order.【F:util/single_q.py†L1-L160】
- **`SingleQAnsatzHoneycomb.py`** offers a closely related interface tuned for honeycomb materials, including helper methods for sampling Euler angles, evaluating the energy per unit cell, and extracting order parameters from the optimal configuration.【F:util/SingleQAnsatzHoneycomb.py†L1-L160】
- **`single_q_field_scan.py`** and `single_q_fixed_q.py` build on these classes to sweep over field strengths or hold $Q$ fixed while mapping magnetisation responses; adapt them when comparing against mean-field or variational baselines.【F:util/single_q_field_scan.py†L1-L69】【F:util/single_q_fixed_q.py†L1-L140】

## Batch plotting and comparisons

Scripts such as `plot_mag.py`, `plot_spin_config_2d.py`, and `compare_energy.py` automate the generation of magnetisation curves, spin-configuration heatmaps, and energy comparisons across parameter grids. Combine them with the output files described in [docs/experiments.md](experiments.md) to streamline analysis pipelines.【F:util/plot_mag.py†L1-L138】【F:util/compare_energy.py†L1-L90】

## Working with large data sets

For large clusters, prefer the `reader_*` modules (`reader_honeycomb.py`, `reader_pyrochlore.py`, `reader_TmFeO3.py`) which provide structured loading routines and take care of unit conversions. They are designed to handle the multi-file outputs written by the MPI-aware drivers without manual parsing.【F:util/reader_honeycomb.py†L1-L150】【F:util/reader_TmFeO3.py†L1-L140】
