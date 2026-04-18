# Refactor backlog

This file tracks structural work that has been scoped but deliberately
deferred out of a single refactor pass, usually because landing it safely
needs its own PR / day-long effort. Items listed here came out of the
Tier 1 and Tier 2 audit passes.

## Deferred (Tier 2)

### Split `lattice.h` (~8.6 kLOC) and `mixed_lattice.h` (~7.8 kLOC)

Both headers are god-objects: they declare the `Lattice` / `MixedLattice`
class *and* inline every method body — Monte Carlo moves, MD integration,
observables, HDF5 I/O, GPU plumbing, EwaldHamiltonian helpers, etc.
Any change that touches one method causes the whole compilation unit to
rebuild.

Recommended split (per file):

1. `lattice_core.h` — class declaration, data members, constructors,
   geometry helpers.
2. `lattice_energy.h` — `total_energy`, per-site energy, gradient.
3. `lattice_mc.h` — metropolis, overrelaxation, deterministic sweep,
   `molecular_dynamics_*`.
4. `lattice_observables.h` — magnetisation, structure factor, binning.
5. `lattice_io.h` — `save_spin_config`, `save_positions`, HDF5 glue.
6. `lattice_gpu.h` — `ensure_gpu_data_initialized` and friends (already
   guards trilinear interactions as of Tier 1).

Each split becomes a translation unit compiled once per executable
instead of inlined at every include site. Expected compile-time win is
large (a full build currently spends most of its time re-instantiating
the two lattice headers from `spin_solver.cpp`).

The reason this is deferred: there are ~150 member functions, many with
subtle cross-references (e.g. `molecular_dynamics_*` → `total_energy` →
`get_Jx` on magnetoelastic params), and several template helpers in
`mc_common.h` call methods across multiple groups. The split must be
done in one coordinated change to keep CTest green, which is a PR of
its own.

### Decompose `spin_solver.cpp::main()` — **partially landed**

The 4.9 kLOC `spin_solver.cpp` has been split into one TU per lattice
family plus a dedicated TU for the parameter-sweep driver. `main()` is
now a thin (~370-line) dispatcher on top of forward declarations in
`src/apps/spin_solver_runners.h`:

* `src/apps/runners_lattice.cpp`           — SU(2) `Lattice` runners
* `src/apps/runners_phonon.cpp`            — `PhononLattice` runners
* `src/apps/runners_strain.cpp`            — `StrainPhononLattice` runners
  (incl. GNEB kinetic-barrier analysis and its file-local helpers)
* `src/apps/runners_mixed.cpp`             — `MixedLattice` (SU(2)+SU(3))
* `src/apps/runners_parameter_sweep.cpp`   — multi-lattice sweep driver
* `src/apps/spin_solver.cpp`               — MPI lifetime + dispatch only

Still deferred, but separately from the TU split:

* Extracting CLI parsing into `spin_solver_cli.{h,cpp}` and replacing
  the hand-rolled option loop with `CLI11`. Needs a smoke-test suite
  for the CLI first (currently no test coverage).
* Introducing a `run_<method>()` factory keyed by `SimulationType` so the
  per-lattice 4-way `switch` blocks in `main` collapse to a single
  dispatch. Low priority — the switches are small and easy to read.

### Drop `using std::X` from public headers

`lattice.h`, `mixed_lattice.h`, `phonon_lattice.h`,
`strain_phonon_lattice.h`, and `mc_common.h` each contain a cluster of
`using std::vector;` / `using std::string;` / `using std::cout;` /
`using std::endl;` declarations at namespace scope. These leak names
into the global namespace of every translation unit that includes them —
standard header-hygiene violation, and a latent source of name-clash
bugs the day someone adds their own `vector` symbol.

Removing them is mechanical but invasive: every unqualified `vector<…>`,
`string`, `cout << …` inside those headers (many thousands of sites)
must be qualified. Easier to do with a clang-tidy pass
(`readability-avoid-unused-using-decls`, `google-global-names-in-headers`)
and a single build than in a hand edit. Deferred for now.

## Not deferred — landed in Tier 2

* Library no longer calls `MPI_Init` on behalf of the caller
  (`include/classical_spin/mc/mc_common.h`): the routine now throws if
  MPI is not initialised.
* Kitaev rotation + local-frame bond matrices live in a single header
  (`include/classical_spin/lattice/kitaev_bonds.h`) and both
  `phonon_lattice.h` and `strain_phonon_lattice.h` delegate to it.
* SLERP / geodesic-angle logic for the GNEB optimizers lives in
  `include/classical_spin/core/gneb_math.h`. Both `gneb.cpp` and
  `gneb_strain.cpp` now share the same implementation instead of three
  copy-pasted copies.
* Dead `include/classical_spin/core/simulation_config.h` has been
  deleted (it was only listed in CMake, never included), which also
  removes the duplicate `trim()` definition that would have become an
  ODR violation if anyone ever included both config headers.
* `spin_solver.cpp` was split from 4945 lines into a 408-line `main` TU
  plus five per-lattice runner TUs. The shared private header
  `src/apps/spin_solver_runners.h` carries the forward declarations so
  the `run_*()` functions can live in separate compilation units and
  build in parallel.
* `util/readers/` is marked deprecated (see `util/readers/DEPRECATED.md`)
  and `workflow/LSWT_fit/job_search.py` emits a `DeprecationWarning`
  pointing callers at `job_search_refactored.py`.
