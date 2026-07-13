# TmFeO3 2DCS campaign — organized run index

All runs: 1×1×1 mixed lattice, Γ₂ seed, T=0 RK4 (Δt=0.02), τ∈[−120,0] (Δτ=0.1),
t∈[−130,40], honest M1 (`reuse_m0_for_m1=false`), `pulse_window_chunking=false`,
pump amp 0.1 / width 0.5, g_ratio_tm=0.058333, γ(λ4..7)=0.15 unless noted.
Shared Hamiltonian: J1ab=4.74, J1c=5.15, J2ab=0.15, J2c=0.30, Ka=−0.0153,
Kc=−0.0187, D1=0.049, e1=2.067834, e2=4.9628 (meV).
Analysis: reader_TmFeO3 with `--tau-gate 6 --dc-remove --norm linear`.

## master_v4/  ← NEWEST (2026-07-10, hybrid master)
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
