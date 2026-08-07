# w3_isolation — is the G-B (E12,qAFM) transfer peak field-gated (κB) or population-mediated (W3)?

Question (2026-08-06): the experimental signature of (E12,qAFM) suggested a
NON-field-mediated conversion. Candidate field-free vertices from the exhaustive
symmetry analysis: (i) W3 SSλ³ population kick (coherence → τ-labelled population
grating → static molecular-field step on Fe), (ii) W3 thermal feeding (in-solver
`thermal_heat` channel), (iii) static W1 components contracting the field-following
tilt (W1_yz — the microscopic candidate for κB_eff = W1_yz·χ⊥).

All runs: fin_B_gs1 base (paper master: W1_yy=W1_xz=0.01, κB_1y=0.0035, W3_yz=0.01,
thermal 2000/0.05/0.00333, exp pulse, geometry B H∥c), τ∈[−120,0] step 0.2
(vs 0.1 in fin_B; analysis subsamples fin_B ::2 for exact comparability),
t∈[−130,160]. Census: crosspol_final2.py pipeline (Hann, blind prominence≥1.5),
S_x readout. bu_* runs: τ=−0.2 only, t→400.

## Verdict — the peak is irreducibly field-gated
ratio = (E12,qAFM)/(qFM,qAFM) at target point; "peak" = resolved blind-census max.

| run          | config                          | ratio | census peak (+0.53,0.89)? |
|--------------|---------------------------------|-------|---------------------------|
| fin_B_gs1    | FULL (κB + W3_yz + thermal)     | 0.448 | YES 0.457 @ ω_t=0.894     |
| iso_kB       | κB only (no W3, no thermal)     | 0.449 | YES 0.459 @ ω_t=0.894     |
| iso_noKB     | W3_yz + thermal, κB=0           | 0.239 | NO (background shoulder)  |
| iso_W3coh    | W3_yz coherent only, κB=0       | 0.253 | NO — identical to none to 2.4e-8 (ODE tol): orbit-cancelled |
| iso_none     | κB=0, W3=0, thermal=0           | 0.253 | NO                        |
| iso_W3xz     | W3_xz=0.01, κB=0                | 0.234 | NO (weak, destructive)    |
| iso_W3xy     | W3_xy=0.01, κB=0                | 0.254 | NO — identical to none (orbit-cancelled) |
| iso_W3zz     | W3_zz=0.01, κB=0                | 0.305 | NO (diagonal renormalizer; anchor amp changes 0.58→0.86) |
| iso_W1yz001  | W1_yz=0.01, κB=0                | 0.254 | NO                        |
| iso_W1yz005  | W1_yz=0.05, κB=0                | 0.529 | NO — merged shoulder, mixer signature (rows coalesce, no resolved 2nd max) |

Why W3 fails: the probe-written population grating is UNIFORM over the 4 Tm
sublattices (dipole signs square), so the orbit algebra R_i·W³·R_i projects its
molecular field onto Bertaut patterns orthogonal to both magnon irreps for
yz/xy (exact null within solver tolerance — same Γ1-block mechanism as K⁻);
xz survives weakly (pattern-allowed) but is ~30× too small and destructive at
the target; zz only renormalizes frequencies/amplitudes. W1_yz at large value
behaves as a static mixer (raises the inter-row region without a resolved peak),
NOT as a converter — the data's two-resolved-unshifted-rows criterion kills it,
same as the W1_xz sweep.

## Buildup / timing (bu_full, bu_noKB, bu_kB, bb2/bb3 comparison)
- Single-pulse qAFM Gabor: decays after pulse in ALL configs incl. bb3_W3 — no growth.
- τ-labelled E12 row vs detection window (iso runs, t≤160): decays monotonically in
  ALL configs; W3+thermal shows NO recovery; fin_B ≈ iso_kB (thermal adds nothing there).
- M_NL @0.90 THz late-time: thermal runs (bu_full/bu_noKB) show a real κB-independent
  tail at t≳300 units, ~16× above the no-thermal floor — but FOUR DECADES below the
  impulsive peak. It is a tail, not the peak.
- FLAG: the foundation's buildup paragraph ("recovers on the repopulation time,
  persists beyond 100 ps, 50–100× above coherent-only") did NOT reproduce with the
  fin_B master parameters in any of the three metrics above. Either it rests on
  different thermal settings/runs (bb5-series analysis scripts were in /tmp, now lost)
  or the text overstates. Needs revisiting before publication.

Figure: w3_settled.png. Params in params/. Analysis: scratchpad w3settle/{analyze,final_fig}.py.

## W1_yz sweep (2026-08-06, runs swB_/swA_W1yz*, figure w1yz_sweep.png)
Dose–response, κB=0, W1_yz ∈ {0, .01, .02, .03, .05, .07}, geometry B target vs
geometry A cost (both readouts + 1D linear line):

| W1_yz | B target/anchor | B resolved row?        | A 1D E12 line | A stray (±0.47,0.13) | A same-pol (E12,qAFM) |
|-------|-----------------|------------------------|---------------|----------------------|-----------------------|
| 0     | 0.253           | no                     | 0.503 THz     | 0.01                 | 0.91 of max           |
| 0.01  | 0.254           | no                     | 0.494         | 0.01                 | 0.90                  |
| 0.02  | 0.257           | no                     | 0.494         | 0.03                 | 0.88                  |
| 0.03  | 0.278           | no                     | 0.484         | 0.07 (census 0.57)   | 0.84                  |
| 0.05  | 0.529           | no (shoulder)          | 0.474         | 0.21 (census 0.98!)  | 0.73                  |
| 0.07  | 0.642           | YES but @ ω_τ=0.46, amp 0.84 | 0.435   | 0.22 (census 0.41)   | 0.56 (row moved to 0.44) |
| κB ref| 0.449           | YES @ ω_τ=0.53, amp 0.46 (= measured) | 0.503 | 0.01        | 0.91                  |

Verdict: gain and damage are the SAME matrix element. Below 0.05 B gains nothing;
at 0.07 the row finally resolves but at the level-repelled position 0.46 THz (not
E12=0.52) with 2× the measured amplitude, while geometry A's LINEAR E12 line has
moved 0.07 THz (≈1.8 linewidths — visible in plain absorption) and the (±0.47,0.13)
stray pair reaches up to parity with the main cross peak (census 0.98 at W=0.05).
No operating window exists: a static bilinear's conversion amplitude IS its level
repulsion; only a gated vertex decouples them. Experimental discriminator: the
measured delay row at the unshifted CEF energy (= the 1D absorption line) excludes
the mixer at ALL strengths.

## W1_yz cost curves at the ATLAS operating point (runs swAf_W1yz*, fig w1yz_sweep_atlasOP.png)
Repeat of the A-side sweep on the flu0.12 base (A_Fe=0.12, su3 0.0220, κB=0.01, no
W3 — the run set behind the atlas figure), because the earlier A-side sweep used
fin_A2 (A_Fe=0.006!) where the weak Fe drive lets the diagonal dominate.
NOTE the provenance issue this exposed: atlas + quoted clean census = flu0.12/vA
generation at A_Fe=0.10-0.12; census_final_picture/crosspol_composite jsons =
fin_A2 at A_Fe=0.006 with a diagonal-dominated inventory. Needs unifying.

| W1_yz | A cross (qAFM,E12) | diag | stray (±.47,.13) | 1D E12 line | 2D emission ω_t | same-pol (E12,qAFM) |
|-------|--------------------|------|------------------|-------------|-----------------|---------------------|
| 0     | 1.00               | 0.21 | 0.02             | 0.503 THz   | 0.50            | 0.26                |
| 0.01  | 1.00               | 0.21 | 0.02             | 0.494 (1 bin)| 0.49           | 0.26                |
| 0.02  | 0.98               | 0.21 | 0.02             | 0.494       | 0.49            | 0.25                |
| 0.03  | 0.99               | 0.21 | 0.02             | 0.484       | 0.48            | 0.23                |
| 0.05  | 0.96               | 0.22 | 0.03             | 0.464       | 0.47            | 0.15                |
| 0.07  | 0.61               | 0.25 | 0.05             | 0.435       | 0.43            | 0.09                |

Refined verdict: at the paper fluence the inventory RATIOS are robust to W1_yz ≤
0.03-0.05 (the weak-drive "mess" was operating-point specific; strays stay ≤0.05).
The un-hideable cost is the LINE POSITION: every E12 feature — 1D absorption and
2D emission alike — slides together (0.50→0.43 over the sweep; FFT bin 0.0097 THz,
shifts ≥0.02 are ≥2 bins). The value geometry B needs for a resolved transfer row
(W1_yz ≥ 0.07) puts the E12 line at 0.435 THz, 1.8 linewidths below the measured
0.50/0.52 — excluded by plain 1D absorption before any 2DCS argument — and erodes
the same-pol (E12,qAFM) peak 0.26→0.09. Tolerable window (≤0.02) and gain window
(≥0.07) do not overlap; they cannot, since repulsion and conversion are the same
matrix element for a static bilinear.

## Recalibrated W1_yz scenario (2026-08-06, runs nscA/B_W05..W15) — the pivot attempt, exhausted
Attempt: adopt static W1_yz as the (E12,qAFM) generator instead of κB, absorbing
the hybridization shift into the bare splitting: iterate e1 until the DRESSED
E12 = 0.50 THz (legitimate — only dressed lines are measured; sum rule survives:
raising |2⟩'s bare energy keeps E12+E23=E13, verified: dressed 0.498/0.709/1.201).
Calibrations: W=0.05→e1=2.1949, 0.06→2.2521, 0.07→2.3267, 0.10→2.6549,
0.12→2.9323, 0.15→3.5331 (bare E12 up to 0.85 THz).

RESULT — the transfer peak NEVER appears once the dressed line is pinned:
- B census at every W ∈ {0.05..0.15}: no row anywhere near ω_τ=0.50; map identical
  in structure to the no-conversion control. The naive-sweep "row at 0.46" was the
  DETUNED E12 line itself sliding toward qFM — proximity mixing into the Fe sector
  through qFM, then the existing (qFM,qAFM) engine. Pin the line, kill the peak.
- The true hybridization partner of W1_yz is NOT qAFM: qFM/qAFM stay at 0.38/0.90
  at all W (they never repel), while a ghost delay row grows at 1.40 THz in A-cross
  (0.67/0.65/0.61 at W=0.10/0.12/0.15) — the upper branch of E12 ⊗ a ~1.06 THz
  zone-centre Fe exchange mode (A_y/C_y Bertaut pattern, per the orbit algebra
  (R_yy R_zz)σ_i). W1_yz couples the E12 coherence to the WRONG magnon irrep, so
  no strength can produce the transfer.
- Strong coupling also wrecks geometry A: at W≥0.10 the hierarchy inverts
  (rephasing 1.00 > non-rephasing 0.56-0.30) + the 1.40 THz ghost row, nowhere in data.

CONCLUSION: the static-vertex space is now exhausted WITH recalibration freedom
included. κB's gate remains the unique (E12,qAFM) generator. New falsifiable
statement for the paper: a static mixer either moves the 1D line (excluded by
absorption) or, recalibrated, betrays itself by its true partner (ghost row at
the upper branch + no transfer); the gated vertex alone puts the row at the 1D
line position with no partner. Figures/params: nsc*, calibration in scratchpad
w3settle/calibrate.py.

## Recalibrated W1_xz — the RIGHT-irrep static candidate (2026-08-07, runs xzB/A_W*)
W1_yz is symmetry-blocked (wrong partner), but W1_xz feeds the qAFM pattern
(July scan: on-target at 0.09 bare). Its July rejection predates recalibration —
retested with dressed E12 pinned at 0.50 (calibrations: W=0.02→e1 bare ok,
0.03→2.0882, 0.04→2.1009, 0.05→2.1185, 0.07→2.1650, 0.09→2.2248).

W1_xz = 0.05/0.07/0.09 (recalibrated): the transfer row appears AT THE CORRECT
POSITION — B top census peak (+0.49, 0.90) at all three — the recalibration
methodology works and the irrep is right. BUT it overshoots by 10-20×: the
(qFM,qAFM) anchor (measured 1.00) vanishes from the census (≤0.09), and strays
flood both geometries (B: E12-emission leakage at ω_t=0.44-0.48 up to 0.20,
reciprocal (0.90, 0.48) up to 0.17; A cross: rephasing E12 diagonal 0.80-0.94 vs
master 0.22, magnon diagonal 0.46-0.61 vs 0.24-0.31, (0.49,0.90) transfer leak
0.27-0.49). Transfer ∝ W² ⇒ measured ratio 0.45 → sweet-spot scan at 0.02/0.03/0.04
(this section to be updated with the verdict).

DISK NOTE: h5 files of all runs with recorded conclusions pruned 2026-08-07
(campaign hit 160 GB / disk 100%); params retained, each regenerates in ~40 s.

## W1_xz fine scan + referee (runs xzB/A_W015/018/02/025/03/04, bu_xz015) — FINAL
Sharp-metric (sigma 1,1) delay-cut at ω_t=0.90, identical analysis for all:

| config          | rows resolved (ω_τ, rel)          | transfer/anchor | A diag costs (reph/mag; master 0.22/0.27) |
|-----------------|-----------------------------------|-----------------|-------------------------------------------|
| fin_B (κB)      | (0.379, 1.00) + (0.518, 0.43)     | 0.43 ✓          | untouched (gate silent in A)              |
| xz W=0.015      | (0.379, 1.00) only — NOT resolved | —               | 0.33 / 0.35                               |
| xz W=0.018      | (0.392, 1.00) only — NOT resolved | —               | 0.39 / 0.40                               |
| xz W=0.020      | (0.392, 1.00) + (0.468, 0.80)     | 0.80            | 0.43 / 0.42                               |
| xz W=0.030      | (0.404, 0.91) + (0.493, 1.00)     | 1.10            | 0.59 / 0.54                               |

Timing (bu_xz015 vs bu_kB/bu_full/bu_none, M_NL qAFM Gabor): ALL peak at 5.3 ps
and decay with identical shape — the static vertex gives NO post-pulse growth
at viable strengths; timing does not discriminate and does not rescue.

VERDICT — the static W1_xz scenario fails on three quantitative counts, all
robust under recalibration freedom:
1. ROW POSITION: static row born at 0.468-0.493 (BELOW the E12 line, W-dependent);
   κB row at 0.518 (at/above). The data's row sits at the 1D-absorption line.
2. EMPTY WINDOW: the row resolves only for W ≥ 0.02 — where its ratio is already
   ≥ 0.80, ~2× the measured 0.43-0.46. Any W matching the measured ratio gives a
   single merged row, contradicting the two resolved rows in the data. Root cause:
   the static row is born only 0.08 THz from qFM (vs 0.14 for the gated row), so
   resolvability and excess amplitude are inseparable.
3. A-SIDE: 1.5-1.8× diagonal elevations at threshold, with no timing benefit.

NEWEST SCENARIO therefore keeps κB as the unique (E12,qAFM) generator. The
campaign's paper-grade products: (i) dressed-line recalibration methodology
(sum rule E12+E23=E13 auto-preserved); (ii) two new falsifiable discriminators —
transfer-row delay position vs the 1D line (static: below by 0.03-0.05 THz;
gated: at/above), and the joint constraint that a static mixer cannot show two
resolved rows at ratio < 0.8; (iii) full static-family exclusion incl. W3 (all
components), W1_yz (wrong irrep, partner = ~1.06 THz A_y/C_y mode), W1_xz
(right irrep, empty window).

## ADOPTED κB-FREE SCENARIO: W1_yz (2026-08-07, figure w1yz_scenario.png)
Per project decision: the working scenario for the geometry-B (E12,qAFM) transfer
is the static W1_yz S_yS_z λ1 vertex at 0.05-0.07 (dial to the measured ratio:
0.60 @0.05, 0.87 @0.07 under (1.5,1.2)-smoothing; ~0.045 → 0.45), κB = 0
everywhere, bare e1. Runs: iso_W1yz005, swB_W1yz007, swA_W1yz007.
Its five testable fingerprints (vs the gate alternative):
1. E12 line down ~13%: 1D absorption at ~0.45 THz lab units; 2D rows co-located.
2. Transfer-row delay position = the (shifted) line position 0.46; gate: 0.52.
3. E23 line UP by the same shift (level-|2⟩ sum rule).
4. Line position ∝ W·G_z ⇒ strong E12 softening through the spin reorientation.
5. Geometry A: E12-family emission at 0.44-0.46 in both polarisations; rephasing
   E12 diagonal ~0.9.
Decisive data check: do the measured 1D E12 line and the measured B delay row
agree at the SAME value? Both down-shifted → this scenario. Line at 0.52 with
row at 0.52 → gate. The constraint results recorded above stand as the ledger
of what each alternative predicts; the scenario choice is falsified/confirmed
by the fingerprints, not by the ledger.

## ★ FINAL κB-FREE SCENARIO (2026-08-07): λ1-drive + master W1_xz — the user's
## message-1 proposal, verified. Runs l1d_*, figure kBfree_final.png.
Hamiltonian: master statics ONLY (W1_yy=0.01, W1_xz=0.01, d_z_eff=3.0,
W3_yz+thermal), kappaB = 0, bare e1. Drive: the one physical geometry-B pulse
carrying BOTH quadratures: H∥c → μ_z λ2 (amp 0.030711, auto value) AND
E∥a → d_x λ1 (amp 0.0165, NEGATIVE relative sign):
  pump_direction_su3 = -0.0165,0.030711,0,0,0,0,0,0 (normalized),
  pump_amplitude_su3 = 0.034864, auto_su3_pump = false.
Result (l1d_final, sharp cut): (qFM,qAFM) = (0.379, 1.00) anchor dominant,
transfer (0.518, 0.42) — SAME row position as the κB master (0.518, 0.43) and
the measured 0.43-0.46. Census clean; strays ≤ 0.12.

Mechanism: E∥a writes the E12 coherence through the electric dipole d_x —
polarisation-OWNED, so the drive exists in geometry B only (geometry A: E∥c,
no d_x; W1_xz stays at master ⇒ geometry A untouched by construction). The
static W1_xz converts the coherence to the magnon. The E×B quadrature
interference (− sign) places the row at 0.518.
Dose ladder (λ1-only drive): ratio 0.41/0.52/0.66/2.13 at k=0.6/0.8/1.0/3.0 of
0.0307; λ2-quadrature-only writing converts dirtily (diagonals ∝ k²) — the
electric quadrature is essential. Combined-drive signs: −: row @0.518 (adopted);
+: row @0.48, amp 0.76.

The E1/M1 balance is now MEASURED, not free: w_E1 = 0.0165/0.0307 = 0.54 in μ
units, fixed by the B transfer ratio. Predictions: (i) sign(d_x·μ_z) < 0 from
the CEF wavefunctions; (ii) the geometry-B same-pol rectified (E12,0) line
strength is now fixed by w_E1=0.54 — independent cross-check; (iii) transfer
∝ G_z ⇒ collapses at the spin reorientation; (iv) timing: static conversion
persists ~T2(E12) after the pulse (see bu_l1dfinal vs bu_kB).
κB is fully retired from the model.

## FULL ATLAS under the κB-free scenario (atlas_kBfree.png + census_atlas_kBfree.json)
Exact atlas_final.py pipeline, B runs → l1d_final (A runs unchanged: flu0.12_gs1-3).
Blind censuses vs the published atlas targets:
- A cross:  (+0.90,0.49)=1.00, (E12,2E12)=(0.49,1.00)=0.97, reph 0.50, diag 0.35 — unchanged ✓
- B cross:  (+0.38,0.90)=1.00, (+0.53,0.90)=0.45 ✓ = the paper inventory, now with NO κB;
            extras ≤0.17 ((0.88,0.90)=0.17, reph 0.13, (1.19/1.39,0.90)=0.10)
- A same:   (E12,E23)=1.00, (E12,E13)=0.80, reph 0.67, (E13,E23-row)=0.32, soft ±0.19=0.12 ✓
- B same PREDICTION: led by (E12,2E12)=(0.49,1.00)=1.00 ✓ with rectified pair (±0.49,0.13)=0.44
            — the w_E1 readout line, now a HARD prediction (w_E1=0.54 fixed by B cross).

## Detection-operator correction (2026-08-07, user physics call): NO λ3 readout
The (E∥a,H∥c) analyser operator must NOT contain the population term λ3 (nor λ8):
P2 = F_z + 5.264λ2 (m_z) + w_E1·λ1 (d_x) + β·λ1λ2. Atlas regenerated (C3=0).
Consequences: A cross essentially unchanged (λ3 had no resonant content there);
B same-pol prediction LOSES the rectified (±E12, ω_t→0) pair (0.44 → 0.07) and
becomes a single-feature prediction: (E12, 2E12) = 1.00, all else ≤ 0.07.
This retracts the "rectified line measures w_E1" claim and the population row of
the main-text polarisation table — w_E1 is instead fixed (=0.54) by the B-cross
transfer ratio in the κB-free scenario. Tex edits pending.
