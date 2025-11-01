# Experiment driver reference

This document summarises the supplied executables in `run_scripts/` and the data products they generate. Every driver writes its results to a user-specified directory and inherits the lattice write helpers that emit spin trajectories, magnetisation traces, and lattice coordinates for later analysis.【F:src/lattice.h†L949-L1014】

## Molecular dynamics of TmFeO$_3$

The `molecular_dynamic_TmFeO3` executable combines a four-sublattice Fe SU(2) manifold with four Tm SU(3) sites and evolves both using the CUDA integrator.【F:run_scripts/molecular_dynamic_TmFeO3.cpp†L677-L688】 Key behaviours:

- **Parameter loading.** The driver first interprets `<output_dir>/params.txt`, then falls back to `./params.txt`, and finally to positional command-line arguments. The file accepts simple `key value` pairs (e.g. `J1ab 4.92`).【F:run_scripts/molecular_dynamic_TmFeO3.cpp†L718-L748】
- **Normalisation.** Exchange constants, anisotropies, and couplings are scaled by `J1ab` before the simulation starts so that the GPU integrator works with a dimensionless Hamiltonian.【F:run_scripts/molecular_dynamic_TmFeO3.cpp†L773-L780】
- **Trial management.** MPI ranks divide `num_trials` evenly; each run optionally reloads an initial spin state from `spin0` before annealing into the molecular-dynamics trajectory.【F:run_scripts/molecular_dynamic_TmFeO3.cpp†L677-L687】
- **Metadata.** The resolved parameter set is always written into `<output_dir>/parameters.txt` for provenance tracking.【F:run_scripts/molecular_dynamic_TmFeO3.cpp†L789-L810】

Result directories contain magnetisation traces (`M_t*.txt`), optional spin snapshots, and any restart files requested in the parameter file.

## Pyrochlore parallel tempering

`parallel_tempering_pyrochlore` expects a parameter file whose keys mirror the structure shown below and runs `num_trials` independent replica-exchange campaigns in subdirectories of `output_dir`.【F:run_scripts/parallel_tempering_pyrochlore.cpp†L5-L70】

```
T_start: 5.0
T_end: 0.05
Jxx: 1.0
Jyy: 1.0
Jzz: 1.0
gxx: 1.0
gyy: 1.0
gzz: 1.0
h: 0.0
field_dir: 0 0 1
output_dir: results/pyrochlore
rank_to_write: 0
theta: 0.0
num_trials: 4
```

For each trial, the driver instantiates the pyrochlore lattice, sets the exchange matrices and g-tensor, and calls `parallel_tempering` to equilibrate and sample at the specified temperature ladder.【F:run_scripts/experiments.h†L13-L102】 MPI ranks cooperate in the same ladder, so launch the executable with `mpirun` and ensure that the parameter file contains enough temperatures for all ranks.

## Pyrochlore molecular dynamics and field scans

The `molecular_dynamic_pyrochlore` driver mixes simulated annealing with deterministic time evolution. It can distribute trial indices across MPI ranks, record the minimum energy density, and optionally sweep over magnetic-field values.【F:run_scripts/molecular_dynamic_pyrochlore.cpp†L11-L200】 Highlights:

- **Trial partitioning.** Each rank draws either an explicit list of trial indices or an automatic range based on the world size so that high-throughput scans can run without manual bookkeeping.【F:run_scripts/molecular_dynamic_pyrochlore.cpp†L111-L139】
- **Energy ranking.** After running, ranks report their local minima, and rank 0 records the best trial and its energy in `best_configuration.txt`.【F:run_scripts/molecular_dynamic_pyrochlore.cpp†L152-L189】
- **Field sweeps.** The optional `magnetic_field_scan` helper sweeps a range of field magnitudes, forwarding the settings to `MD_pyrochlore` and relying on MPI to parallelise the job.【F:run_scripts/molecular_dynamic_pyrochlore.cpp†L193-L217】

Use this executable when you want to initialise molecular dynamics from multiple annealed configurations and identify the most stable state across a parameter grid.

## BCAO honeycomb replica exchange

`parallel_tempering_BCAO` configures an anisotropic honeycomb Hamiltonian with nearest-, next-nearest-, and third-neighbour couplings before launching a long-temperature-ladder run.【F:run_scripts/parallel_tempering_BCAO.cpp†L4-L66】 Temperatures are sampled logarithmically between `T_start` and `T_end` using the size of `MPI_COMM_WORLD`, so each rank occupies exactly one rung.【F:run_scripts/parallel_tempering_BCAO.cpp†L55-L60】 The current `main` hard-codes the output path; adjust it or add argument parsing prior to submitting cluster jobs.【F:run_scripts/parallel_tempering_BCAO.cpp†L70-L86】

## Phase-diagram utilities

Additional helpers such as `phase_diagram_pyrochlore.cpp` provide parameter sweeps over exchange constants and field directions, delegating the heavy lifting to the functions defined in `experiments.h`. The executable reads nine positional arguments: a `J_pm` range, the number of samples, a fixed `J_{±±}`, a field range, and the scan direction (`001`, `110`, `1-10`, or `111`).【F:run_scripts/phase_diagram_pyrochlore.cpp†L4-L26】 The resulting directory contains subfolders for each grid point, enabling offline comparison with experimental phase diagrams.

## Creating new drivers

When you add a new file to `run_scripts/`, it is automatically picked up by CMake and built into its own executable. Follow the patterns above: parse parameters, instantiate the appropriate unit-cell class, configure interactions, and call into the shared Monte Carlo or dynamics routines in `experiments.h` and `src/lattice.h`.【F:CMakeLists.txt†L84-L142】【F:run_scripts/experiments.h†L13-L190】
