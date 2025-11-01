# ClassicalSpin_Cpp

ClassicalSpin_Cpp is a high-performance C++/CUDA toolkit for simulating classical spin models on complex lattices. The code supports hybrid SU(2)/SU(3) degrees of freedom, large-scale Monte Carlo workflows, and GPU-accelerated molecular dynamics so that researchers can explore phase diagrams, dynamical responses, and field-driven behaviour in materials such as pyrochlores, honeycomb magnets, and TmFeO$_3$.

## Key capabilities

- **Template-based lattice construction.** `UnitCell` helpers collect on-site terms together with bi- and tri-linear couplings, making it straightforward to encode material-specific Hamiltonians and print debug summaries of the resulting unit cell definitions.【F:src/unitcell.h†L209-L320】
- **Production-ready Monte Carlo engines.** The `lattice` driver exposes simulated annealing, cluster updates, and replica-exchange parallel tempering with built-in MPI/OpenMP support, including extensive logging and acceptance-rate control to keep large runs stable.【F:src/lattice.h†L1245-L1289】【F:src/lattice.h†L1916-L1965】
- **GPU-accelerated real-time dynamics.** CUDA integrators evolve coupled SU(2)/SU(3) spins, stream observables to disk, and optionally record microscopic trajectories for later analysis.【F:src/molecular_dynamics.cuh†L1000-L1052】
- **Mixed-lattice simulations.** Hybrid SU(2)/SU(3) lattices share magnetisation buffers, support CPU-based SSPRK time stepping, and write spin snapshots for both manifolds, enabling co-evolution of Fe and Tm sublattices in TmFeO$_3$.【F:src/mixed_lattice.h†L2743-L2790】
- **Rich output for post-processing.** Every lattice exposes helpers that emit spin configurations, magnetisation traces, and lattice positions, which the Python utilities in `util/` can ingest for visualisation and further analysis.【F:src/lattice.h†L956-L1014】【F:util/plot_convergence.py†L7-L116】

## Repository layout

| Path | Description |
| --- | --- |
| `src/` | Core headers and CUDA kernels implementing lattice dynamics, Monte Carlo moves, and molecular dynamics solvers. |
| `run_scripts/` | Stand-alone drivers that instantiate specific material models (pyrochlore, BCAO, TmFeO$_3$) and orchestrate full simulation campaigns. |
| `util/` | Python tools for analysing simulation output, creating plots, and exploring ansatz-based reductions. |
| `docs/` | Additional documentation covering experiment workflows and analysis helpers. |

## Building the project

### Dependencies

The build system requires a C++20 toolchain, CUDA 11+ (with support for SM60 through SM89 devices), MPI, and OpenMP. These dependencies are validated by CMake at configure time and the release profile enables aggressive CPU and GPU optimisation flags.【F:CMakeLists.txt†L1-L82】

### Configure & build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target all_executables
```

CMake generates one executable per `run_scripts/*.cpp` file. Set `CMAKE_CUDA_ARCHITECTURES` to a space-delimited list if your hardware differs from the default `60 75 80 86 89` selection.【F:CMakeLists.txt†L27-L28】 When using Ninja or Make, the build automatically fans out across all local CPU cores.【F:CMakeLists.txt†L168-L178】

## Running simulations

### General workflow

1. Choose a driver under `run_scripts/` that matches the experiment you want to run.
2. Configure model parameters via a text file or command-line arguments (see the tables below or the corresponding source file comments).
3. Launch the compiled executable from the build directory. MPI-aware drivers should be started with `mpirun`/`mpiexec` so that each rank participates in replica exchange or sampling.
4. Inspect the output directory for magnetisation traces, spin snapshots, energy logs, and convergence summaries.

Lattice helpers automatically create the requested output directories and emit spin configurations (`spin.txt`), lattice coordinates (`pos.txt`), and magnetisation time series (`M_t*.txt`).【F:src/lattice.h†L949-L1014】

### Example: CUDA molecular dynamics for TmFeO$_3$

The `molecular_dynamic_TmFeO3` driver loads parameters from `<run_dir>/params.txt` (or `./params.txt` as a fallback) and normalises the Hamiltonian before launching a hybrid SU(2)/SU(3) molecular dynamics run.【F:run_scripts/molecular_dynamic_TmFeO3.cpp†L718-L771】 Each MPI rank runs a subset of trials and writes time-resolved magnetisation files using the GPU integrator.【F:run_scripts/molecular_dynamic_TmFeO3.cpp†L677-L688】 A minimal parameter file looks like:

```
J1ab 4.92
J1c 4.92
J2ab 0.29
J2c 0.29
Ka 0.0
Kc -0.09
D1 0.0
D2 0.0
chii 0.05
xii 0.0
e1 2.2
e2 4.8
h 0.0
T_start 0.0
T_end 50.0
T_step_size 0.01
```

Invoke the executable with `mpirun -n <ranks> ./molecular_dynamic_TmFeO3 <output_dir>` to write results into `<output_dir>` and automatically log the resolved parameters in `parameters.txt` for reproducibility.【F:run_scripts/molecular_dynamic_TmFeO3.cpp†L729-L769】

### Replica-exchange and annealing campaigns

- **Pyrochlore parallel tempering.** Provide a parameter file with temperature bounds, exchange tensors, and MPI output ranks; the driver will spawn the requested number of trials and call `parallel_tempering_pyrochlore` with the configured replica ladder.【F:run_scripts/parallel_tempering_pyrochlore.cpp†L5-L70】
- **BCAO honeycomb magnet.** `parallel_tempering_BCAO` assembles anisotropic nearest-, next-nearest-, and third-neighbour couplings, samples a logarithmic temperature ladder based on the MPI world size, and launches long parallel-tempering runs. Update the hard-coded output directory in `main` before submitting production jobs.【F:run_scripts/parallel_tempering_BCAO.cpp†L4-L66】【F:run_scripts/parallel_tempering_BCAO.cpp†L70-L86】
- **Pyrochlore molecular dynamics scans.** `molecular_dynamic_pyrochlore` supports batch field scans, trial subsets per MPI rank, and best-energy reductions to identify the lowest-energy configuration across trials.【F:run_scripts/molecular_dynamic_pyrochlore.cpp†L11-L190】

Additional drivers cover phase-diagram sweeps, finite-temperature Monte Carlo of other models, and single-purpose experiments; inspect the source under `run_scripts/` for argument details.

## Python analysis tools

The `util/` directory complements the C++ drivers with analysis scripts:

- **Convergence diagnostics.** `plot_convergence.py` summarises simulated annealing runs by plotting energy, acceptance rates, and configuration changes while reporting convergence statistics in the terminal.【F:util/plot_convergence.py†L7-L128】
- **Reciprocal-space utilities.** `reader_pyrochlore.py` defines Brillouin-zone paths, local frames, and Fourier transforms used to visualise dynamical structure factors for pyrochlore lattices.【F:util/reader_pyrochlore.py†L1-L109】
- **Variational ansätze.** `single_q.py` implements a single-$Q$ optimisation for honeycomb magnets, constructing interaction matrices and minimising the energy with SciPy to classify magnetic orders.【F:util/single_q.py†L1-L160】

These scripts expect NumPy/SciPy/Matplotlib; install them into a Python environment of your choice before post-processing large simulation dumps.

## Extending the codebase

1. **Define the microscopic model.** Start from an existing unit cell or create a new `UnitCell` subclass by chaining `set_bilinear_interaction`, `set_trilinear_interaction`, and `set_onsite_interaction` calls to capture the relevant couplings.【F:src/unitcell.h†L241-L264】
2. **Embed the model in a lattice.** Instantiate the templated `lattice` (or `mixed_lattice`) with the desired dimensions and spin length, then reuse the Monte Carlo or dynamics routines provided in `src/lattice.h` and `src/mixed_lattice.h`.
3. **Create a driver.** Add a `.cpp` file under `run_scripts/` that parses parameters, sets up MPI if necessary, and invokes the simulation routines. CMake will automatically build a matching executable thanks to the globbing rules in the top-level `CMakeLists.txt` file.【F:CMakeLists.txt†L84-L142】
4. **Document output.** Use the provided write helpers to emit spin configurations, observables, and metadata so that the Python tooling can ingest the results without additional glue code.【F:src/lattice.h†L956-L1014】

See `docs/` for walkthroughs that cover the bundled experiment drivers and analysis notebooks.
