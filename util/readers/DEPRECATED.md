# `util/readers/` — deprecated

**Status:** legacy. No active code path in this repository imports from
`util/readers/`. The maintained implementations live in
[`util/readers_new/`](../readers_new).

## Why both directories exist

When the simulation engine migrated its output format from per-file CSV /
binary dumps to HDF5, the reader scripts were forked:

* `util/readers/` — original CSV / `*.txt` / `*.npy` readers for the
  pre-HDF5 outputs and ad-hoc plotting utilities written over the years.
  Kept around so old `local_runs/` trees can still be replotted without
  digging through git history.
* `util/readers_new/` — the current HDF5-based readers (`reader_honeycomb.py`,
  `reader_phonon_lattice.py`, `reader_strain_lattice.py`, etc.). Used by
  the `NCTO_project/*` scripts and documented in the top-level `README.md`.

## What to do

New analysis code should import from `util/readers_new/`. If you find a
feature here (e.g. a plot type) that does **not** exist in `readers_new/`,
port the relevant helper into `readers_new/` and delete the copy here.

Once every plot type has a `readers_new/` equivalent this directory will
be removed in a follow-up cleanup.
