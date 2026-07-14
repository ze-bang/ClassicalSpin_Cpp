# TmFeO3 2DCS campaign — organized run index

All runs: 1×1×1 mixed lattice, Γ₂ seed, T=0 RK4 (Δt=0.02), τ∈[−120,0] (Δτ=0.1),
t∈[−130,40], honest M1 (`reuse_m0_for_m1=false`), `pulse_window_chunking=false`,
pump amp 0.1 / width 0.5, g_ratio_tm=0.058333, γ(λ4..7)=0.15 unless noted.
Shared Hamiltonian: J1ab=4.74, J1c=5.15, J2ab=0.15, J2c=0.30, Ka=−0.0153,
Kc=−0.0187, D1=0.049, e1=2.067834, e2=4.9628 (meV).
Analysis: reader_TmFeO3 with `--tau-gate 6 --dc-remove --norm linear`.

## EXPERIMENTAL_FINAL/  ← THE FINAL PICTURE (2026-07-14, all-experimental-input)
Cross-pol, both geometries, 0K+finite T. Zero free lineshape params:
ALL THREE couplings ON (W1_yy 0.01 → G-A peak; κB 0.01 → G-B transfer peak; W1_xz 0.01 → G-B magnon peak + buildup; isolated-vertex runs diagnostic only). linewidths→damping (CEF Bloch γ=lw[THz]×4.136 meV: λ1,2=0.157/λ4,5=0.207/
λ6,7=0.414; magnon Gilbert α=0.001667), oscillator strengths→emission
(√f E1 CEF, M1 magnon), measured pulse. G-A (qAFM,E12)=1.00/R 0.33, T-flat;
G-B (qFM,qAFM)=1.00,(E12,qAFM)=0.43→0.29(10K). Figure crosspol_experimental_final.png.
See EXPERIMENTAL_FINAL/README.md.

## (earlier) cross-pol lean scenario — §sec:fullscenario in tex
Cross-polarization only, both geometries, 0K + finite T, experimental readout.
G-A cross (expA_longt, λ2, emit @E12 E1): (+qAFM,E12)=1.00, (−qAFM,E12)=0.68.
G-B cross (xB_gs1, Sx, emit @qAFM M1): (+qFM,qAFM)=1.00, (+E12,qAFM)=0.65.
Finite T (xA_gs2/3, xB_gs2/3, Boltzmann): window 5-8K; G-A gains (qFM,E12),
G-B loses (E12,qAFM)~(n1-n2). High-res t→160 (Δω_t=0.010 THz).
Readout: CEF E1 (√f), magnon M1; qAFM E1-dark (f=7e-4) => G-A reads magnon
through Tm E12 line, G-B via magnon M1 dipole. Figure crosspol_scenario.png.
Same-pol DROPPED. (was: exppulse_A/B v6 reference)
UPDATED PEAK LISTS (signed, delay conv., + = NR): G-A cross (+qAFM,E12);
G-A same (+E12,E13) AND (+E12,qAFM) [reassigned from (E12,E23) — experimental
0.7~0.9 peak is the W-mediated Tm→magnon transfer, unconjugated ⇒ NR side
×2.6, matches E²·E scaling and the same-side observation]; G-B cross
(+qFM,qAFM)+(+E12,qAFM); G-B same prediction (+qAFM,E12)=1.00,
(+qFM,qAFM)=0.97, (+E12,qAFM)=0.59. Predictions: weak (−E12,E23) mirror at
(−0.5,0.70); line center 0.90 not 0.70; position tracks qAFM with T.
Figure: figures/final_scenario_v6.png. (originally the rolled-back reference)
v4 master Hamiltonian verbatim + experimental pulse table, zero retuning:
W1_yy=0.01, W1_xz=0.01, κB_1y=0.01, d_z^eff=3.0, amp=0.1, g_ratio_tm=0.058333.
Gap vs designated lists (folded, physically weighted detections):
- G-A cross m_z: (qAFM,E12)=1.00 ✓ sole dominant; extras ≤0.24.
- G-A same m_x: per-channel perfect (λ5 (E12,E13)=1.00; λ7 (E12,E23)=1.00,
  soft-mode sidebands 0.3-0.36) BUT combined channel is the Fe (qAFM,qAFM)
  diagonal ×~300 over the CEF peaks — Fe/Tm detection-weight gap (the g knob).
- G-B cross m_x: (qFM,qAFM)=1.00, (E12,qAFM)=0.62 ✓; extras ≤0.12; λ4-7 ≡ 0
  (SU(2) subalgebra closure — algebraic).
- G-B same m_z PREDICTION: (qFM,qAFM)=1.00, (qAFM,E12)=0.95, (E12,qAFM)=0.62,
  (qFM,E12)=0.61, (E12,E12)=0.57, (qFM,E23)=0.22.
Open physics gaps: signed side of (E12,E23) [see scan_esa_sides/], soft-mode
±0.12 THz satellites (checkable in data).
E23>E13 ORDERING (expA_gasym/, 2026-07-14): level-differentiated linewidths
γ(λ4,λ5)=0.25, γ(λ6,λ7)=0.05 (E13 line broader — higher CEF level, denser
phonon decay; peak∝1/γ verified: λ7 ×2.5, λ5 ×0.46) + detection operator
ω_t²·[d·(λ5+λ7) + 2S_x M1] with equal E1 dipoles d23=d13, Tm/Fe scale R≈300.
Detected cross-peak list: (E12,E23)=1.00, (E12,E13)=0.43, all else ≤0.30
(magnon diagonal excluded as expected AFM feature). Falsifiable: E23 peak must
be ~5× NARROWER than E13 in the experimental ω_t cuts; μ ratio can't do this
(cancels — both peaks share the μ5·μ7 product). Geometry B untouched (λ4-7≡0).
Figure: figures/gasym_detected_GA_same.png.

## master_exppulse/  (v5.1, 2026-07-13 — retuned variant, superseded by rollback)
Pulse: pump_table_file experimental_pulse_codeunits.dat (E_field_20240716,
quasi-single-cycle, peak 0.53 THz, no DC), amp=0.01337, g_ratio_tm=0.262.
Params: W1_yy=0.003 (reduced from 0.01 to cut Tm→Fe back-action combs at
E12±qAFM), W1_xz=0.0025, κB_1y=0.0035, d_z^eff=3.0
(A: su3 manual dir 0,3.0,0,0,2.3915,0,0.9128,0, amp 0.013816; B: auto).
- geomA — λ2: (qAFM,E12)=1.00; λ4: (E12,E13)=1.00; λ6: (E12,E23)=1.00;
  Fe S_x abs down ×3.5 vs W=0.01, residual tones confined to ω_t=0.38/0.9/1.4.
- geomB — F_x: (qFM,qAFM)=1.00, (E12,qAFM)=0.97 (both dominant); rest ≤0.15.

## expA_gs{2,3}/, expB_gs{2,3}/  (exp-pulse thermal level components)
Level seeds |2⟩/|3⟩ run verbatim (annealing_steps=0, n_deterministics=0) with
the v5.1 configs; Boltzmann-combined by scratchpad final/thermal_sweep.py.
Result: perfect temperature T*=5 K, window T≲8 K (λ2 loses to quasi-elastic
(−0.10,E12) at 12 K; B (E12,qAFM) 0.94@5K→0.44@12K). λ4/λ6 targets stay 1.00
through 18 K with thermal partners (E23,E13)/(E13,E23) growing.
Figure: figures/exppulse_T0_vs_T5K.png.

## master_v4/  (2026-07-10, Gaussian-pulse hybrid master)
Params: W1_yy=0.01, **W1_xz=0.01** (static, post-pulse-buildup-required),
κB_1y=0.01 (B_z-gated), d_z^eff=3.0
(A: pump 1,0,0; su3 dir 0,3.0,0,0,2.3915,0,0.9128,0, amp 0.023006;
 B: pump 0,0,1, auto_su3_pump).
- geomA_master — G1: (qAFM,E12)=1.00 sole peak (leak 0.09);
  G3 λ4: (E12,E13)=1.00, (qAFM,E13)=0.08; λ6: (E12,E23)=1.00, cascade 0.75.
- geomB_master — F_x: (qFM,qAFM)=1.00, (E12,qAFM)=0.62; residuals ≤0.14.
  Post-pulse qAFM buildup via triad qFM+E12≈qAFM (κB-only is flat).

## verification_pulsewidth/  (displacive-mechanism test, §19.6)
verify_item1_w0p{25,50,75,100}: width sweep, g_tm=0.0583; τ∈[−40,0], t∈[−50,40].
verify_item1_g0_w0p{25,50,75,100}: same with g_tm=0 (pure W).
Result: peak collapses 51.7→1.7e−3 (4.5 decades); Tm-drive independent <1e−4.

## scan_gamma/  (level-3 linewidth trade-off, §19.7)
geomA_gs04 / geomA_gs08: γ(λ4..7)=0.04 / 0.08 on the v3 master config
(W=0.01, d=1.6, κB=0.01). Result: peak∼1/γ, FWHM∼γ, (E23,E13) stray 0.34→0.06.

## scan_W1_components/  (static replacement scan for κB)
Geometry B, κB=0, W1_yy=0.01 + one component at 0.03: test_B_W{xx,zz,xz,yz},
test_B_Wxy{01,03}. Verdicts: yz/xy inert, zz negligible, xx floods junk,
**xz alive** → operating-point scan test_B_Wxz{05,06,09} and geometry-A leak
checks test_A_Wxz{05,09}, test_A_Wxy01. No clean κB replacement exists;
W1_xz adopted small (0.01) in v4 for the buildup only.

## scan_W4_components/  (the (qAFM,E13)-enabler question)
test_W4{yy,xy,yz}: W4=0.02 on v3 master → ALL exactly inert (Δ≈1e−15,
Γ₁-cancelled). test_kE5x: κE_5x=0.02 → inert. ⇒ no same-pol (qAFM,E12);
(qAFM,E13) only via cascade.

## scan_strongdrive/  (drive-regime bracket, §19.x)
test_strongdrive (amp_su3 ×5), test_strong25 (×14): cross-peak inventory
drive-invariant; λ6 same-side ratio 11:1→2.3:1; G1 gains (E12,E12) DIAGONAL 11×.
test_dec5/test_dec7: λ5-only/λ7-only decomposition → no interference;
E23 dimness is a creation bottleneck. geomB_strong: strong-B fails (2E12 line).

## controls_earlier/  (v1/v2-era corrected runs, superseded parameters)
geom1and3_Ha_Ec (W=0.05, κB=0.01, d=1.0-era), geom2_Hc_Ea,
control_pure_qutrit_Ha (all couplings off — single-ion transfer proof).

## figures/
master_spectra.png (v3-era), focus_channels.png, samepol_combined.png,
pulsewidth_verification.png, gamma_sharpening.png, summary_*.png (v1-era).

Legacy pre-window-fix campaigns (contaminated by the md_time_start/M1-reuse
artifact; kept for history): ../archive_prefix_legacy/

## scan_esa_sides/  (signed-side / ESA investigation, 2026-07-13 — NOT adopted)
Experiment reports (E12,E13) and (E12,E23) both NON-REPHASING; model at master
drive puts λ6 on the rephasing side (29:1, drive-robust; leakage floor 3.4%).
Scan of Tm drive alone: λ6 flips NR at ~100× (θ12≈0.5 rad), both-NR window
120–200×, but that operating point buries (qAFM,E12) under the (E12,E12)
diagonal (×2000) and (qFM,qAFM) under W-fed back-action (×70) → rolled back.
v5.1 master unchanged. Open discrepancy: see figures/rabi_point_verification.png.
Refuted along the way: W1_xz static rotation, Γ₁ orbit cancellation (staggered
λ6 = 1e-15), d-boost (su3 dir vector is normalized — components reweight).
