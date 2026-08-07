# tmfeo3_2dcs_final — organized 2026-08-07

- `FINAL_SCENARIO/` — THE working model (κB-free): runs, params, trajectory
  datasets (time / mixed ω_τ-vs-t / 2D-frequency domains, all magnetization
  components), exact readout-channel definitions (READOUT_CHANNELS.md),
  atlas + figures. Start here.
- `ARCHIVE_pre_kBfree/` — the entire prior campaign (gate-era runs, iteration
  ladder, isolation/exclusion record in EXPERIMENTAL_FINAL/w3_isolation/).
  All bulky HDF5 deleted (PRUNED_H5_MANIFEST.tsv, 617.8 GB / 102 files);
  params, READMEs and figures retained — any run regenerates in ~40 s from
  its param file.
