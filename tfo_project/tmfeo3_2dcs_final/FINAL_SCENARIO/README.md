# FINAL SCENARIO — TmFeO3 2DCS, κB-free model (frozen 2026-08-07)

The working model of the full 2DCS atlas with NO field-assisted (BSλ / κB)
vertex. Geometry-B transfer mechanism: the E∥a electric dipole writes the E12
coherence (d_x·λ1, polarisation-owned), the static W1_xz converts it to the
qAFM magnon. w_E1 = 0.54 (μ units) is fixed by the measured transfer ratio.

## Contents
- `READOUT_CHANNELS.md` — EXACT drive and detection operator for each of the
  four channels, with dataset column indices. Read this first.
- `runs/`
  - `geomB_gs1/`        — geometry B (E∥a, H∥c), ground seed, τ∈[−120,0] step 0.2
  - `geomB_gs1_timing/` — same, single τ, t→400 (post-pulse timing)
  - `geomA_gs{1,2,3}/`  — geometry A (E∥c, H∥a), ground + level-2/3 seeds
                          (Boltzmann-mix for finite T), τ step 0.1
- `params/` — the exact solver inputs, one per run (regenerate: ~40 s each,
  `mpirun -np 16 build/spin_solver <param>` from the repo root).
- `trajectories/` — the distilled magnetization datasets (see below).
- `figures/` — atlas_kBfree.png (+ census json + generator script),
  kBfree_final.png (scenario vs gate referee), Bsame_decomposition.png.

## trajectories/*.h5 layout (self-documented attrs in each file)
- `/axes/` — t_full, t (decimated ×5, dt = 0.1 units = 0.0658 ps), tau,
  omega_tau (PHYSICAL ω_T, ascending; + = non-rephasing), omega_t (THz).
- `/reference/` — single-pulse M0(t) at FULL time resolution:
  M_global_SU2 (S_x,S_y,S_z), M_global_SU3 (λ1..λ8), M_antiferro_SU2,
  M_local_SU3, quad (= local λ1·λ2, the 2E12 emission coordinate).
- `/time_domain/` — M_NL(τ, t) = M01−M1−M0 for all components
  (MNL_SU2 [τ,t,3], MNL_SU3 [τ,t,8], QNL [τ,t]).
- `/mixed_domain/` — FFT along τ ONLY, time along t (complex64):
  M_NL(ω_τ, t). Processing: per-t τ-mean removed (ω_τ=0 row), half-Hann τ
  apodization, reordered to physical ω_T. Use |·| for row-amplitude vs
  detection time; phase retained.
- `/freq_domain/` — |M_NL(ω_τ, ω_t)| via the standard analysis pipeline
  (t≥3 window, Hann both axes, NO smoothing) — the atlas maps are these
  with (2,1.5)-bin Gaussian smoothing and the census on top.

To reproduce the atlas: `python3 figures/atlas_kBfree.py` from the repo root
(paths inside point at runs/).

## Provenance
Established by the w3_isolation campaign (2026-08-06/07): exhaustive exclusion
of W3 (all components), W1_yz (wrong irrep), recalibrated static mixers, and
the κB gate replaced by the two-quadrature drive. Full audit trail:
`../ARCHIVE_pre_kBfree/EXPERIMENTAL_FINAL/w3_isolation/README.md`.
