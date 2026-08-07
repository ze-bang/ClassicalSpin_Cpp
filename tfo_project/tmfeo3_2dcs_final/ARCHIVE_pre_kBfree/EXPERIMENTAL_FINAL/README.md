# EXPERIMENTAL_FINAL — the final picture from experimental inputs only

Cross-polarization scenario computed with ZERO free lineshape parameters:
every linewidth is a damping rate, every emission weight an oscillator strength,
and the pump is the measured THz waveform. High-res (t→160), τ∈[-120,0].

## Couplings — ALL THREE ON simultaneously (one Hamiltonian, both geometries)
- W1_yy = 0.01 (two-magnon exchange, static) → G-A (qAFM,E12) peak.
- kappaB_1y = 0.01 (field-gated, impulsive) → G-B (E12,qAFM) transfer peak.
- W1_xz = 0.01 (static hybridization) → G-B (qFM,qAFM) peak + post-drive buildup.
Isolated-vertex tests (gmB_kBonly, gmB_Wxzonly, gmB_a0_*) are DIAGNOSTIC only,
proving none is redundant: κB-only makes (E12,qAFM) but is flat (no buildup);
W1_xz-only builds up but loses (E12,qAFM). The scenario runs gmA/gmB have all on.

## Experimental inputs used
- Pulse: measured THz waveform (experimental_pulse_codeunits.dat).
- Linewidths → damping:
  - CEF Bloch γ_su3 [meV] = linewidth[THz]×h(4.136):
    λ1,2 (E12, 0.038 THz)=0.157; λ4,5 (E13, 0.05)=0.207; λ6,7 (E23, 0.10)=0.414.
  - Magnon Gilbert α = Δν/2ν = 0.003/(2·0.9) = 0.001667 (qAFM-matched).
- Oscillator strengths → emission weights: CEF via √f (E1), magnon via M1;
  qAFM E1 (f=7e-4) negligible.

## Runs
- gmA / gmA_gs2 / gmA_gs3 — geometry A (H∥a, E∥c), ground + level-2/3 seeds.
- gmB / gmB_gs2 / gmB_gs3 — geometry B (H∥c, E∥a), ground + level-2/3 seeds.
- Thermal = Boltzmann mix e^{-E_i/kT} of the three seed runs.

## Results (Hann-apodized, prominence census)
- G-A cross (λ2, emit @E12 E1): (+qAFM,E12)=1.00, (−qAFM,E12)=0.33,
  (qAFM,qAFM)=0.18. Temperature-flat to 10 K.
- G-B cross (Sx, emit @qAFM M1): (+qFM,qAFM)=1.00, (+E12,qAFM)=0.43;
  (E12,qAFM) falls 0.43/0.41/0.35/0.29 at 0/5/8/10 K (∝ n1−n2).
- Figure: crosspol_experimental_final.png (also in ../figures/).

Written up in tmfeo3_foundation.tex §sec:fullscenario (fig:crosspol).

## Regeneration 2026-07-23 (consolidated Hamiltonian, runs vA_gs1-3 / vB2_gs1-3)
One Hamiltonian for ALL channels (incl. same-pol it14hr): v4 + D2=0.014,
W1_yy=0.01, W1_xz=0.01, kappaB_1y=0.0035 (recalibrated once; B_z-gated so
both H||a channels provably unaffected). Verified blind censuses:
A (lambda2): (+0.91,0.50)=1.00, (-0.90,0.50)=0.46, (+0.90,0.89)=0.24, T-flat.
B (Sx): (+0.38,0.90)=1.00, (+0.53,0.89)=0.46 -> 0.45/0.40/0.36 @5/8/10K, res<=0.17.
Params + distilled data in ../../paper/{params,data}; figure paper/figs/crosspol_final.png.
NOTE: vB_gs1-3 (kappaB=0.01 first regeneration) show the inverted ratio 0.76/1.00 —
the B ratio is the kappaB calibration observable.
