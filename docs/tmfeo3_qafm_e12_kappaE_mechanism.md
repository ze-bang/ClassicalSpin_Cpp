# TmFeO3 (qAFM, E12) 2DCS cross-peak via E∥c-assisted exchange (χ dropped)

**Date:** 2026 · **Geometry:** pump H∥a + E∥c, detection H∥c + E∥a · **Model:** χ=0, W/κ^E/κ^B only

## Premise
The bilinear magnetoelectric coupling χ (the `chi2*/chi5*/chi7*` terms) is **Pbnm-forbidden
for the Γ₂ (Gₓ) Fe order at q=0** — proven this session (`diag_tmfeo3_pbnm_invariance`,
`diag_tmfeo3_qafm_e12_xpeak`): a uniform (σ_F) driven λ pattern is orthogonal to the staggered
(σ_A) Tm dipole detector, and λ⁵/λ⁷ are dead channels. So χ cannot produce the (qAFM, E12)
cross-peak. The prior campaign confirmed empirically that static χ (and static W) yield only the
**transpose** (E12 exc, qAFM det) or a DC-rectified ridge — never the requested ordering.

## The working mechanism: a time-dependent E∥c bridge (κ^E)
The symmetry-allowed **electric-field-assisted exchange** (`tmfeo3_foundation.tex`, §"Specialization
to H∥a, E∥c") is
$$ H_{E\chi}(t) = -E_c(t)\sum_{(k,j)}\big[\kappa^E_{c;5y}\,S^y_{\mathrm{Fe},k}\lambda^5_{\mathrm{Tm},j}
   +\kappa^E_{c;7y}\,S^y_{\mathrm{Fe},k}\lambda^7_{\mathrm{Tm},j}\big], $$
plus the direct E∥c dipole $H_E^{\mathrm{Tm}}=-E_c(t)\sum_j(d_{c4}\lambda^4_j+d_{c6}\lambda^6_j)$.
SU(3) commutators $[\lambda^5,\lambda^6]\!\sim\!\lambda^1$, $[\lambda^7,\lambda^4]\!\sim\!\lambda^1$
build the E12 coherence ($\lambda^1$).

This is **Route C**: the H∥a pulse creates a qAFM coherence (Fe $S^y$) that evolves freely during τ
(→ $\omega_\tau=q_{\rm AFM}$, *linear* in the qAFM coherence); the *separate, pulsed* E∥c field then
converts it to E12 via the κ^E vertex (the E-pulse supplies the frequency "idler"). A *static* vertex
(χ or static W) cannot do this — it only hybridizes/rectifies, giving the transpose or DC.

## Numerical confirmation (`tmfeo3_qafm_e12_kappa/`, 1×1×1, 201 τ-points)
Production Fe params (qFM≈0.374, qAFM≈0.905 THz), e1=2.0678 meV (E12=0.500 THz), χ=0,
`kappaE_5y=kappaE_7y=0.30`. NL = M01−M0−M1, 2D-FFT (Hann), rigorous local-max detection.

| run | drive amp | Tm λ¹ (E12, E∥a) target | transpose | κ^E control |
|-----|-----------|--------------------------|-----------|-------------|
| `amp01` | 0.01 (linear) | **9.3 %, LOCAL-MAX** at (ω_τ=0.907, ω_t=0.503) | 0 % | — |
| `amp01_ctrl` | 0.01, κ^E=0 | not a local max; max\|F\| **1800× smaller** | 0 % | peak gone |
| `kappaE_route` | 0.10 (over-driven) | buried under DC; Fe afSz target = 9.9 % LOCAL-MAX | 0 % | — |
| `all_terms` | 0.10, +W+κ^B | Tm λ¹ target = 4.0 % LOCAL-MAX | 0 % | — |

- **κ^E is causal:** zeroing κ^E removes the cross-peak entirely (max|F| drops ×1800) and leaves only
  the DC-E12 ridge.
- **Correct ordering:** the peak is the *target* (qAFM exc, E12 det); the transpose is 0 %.
- **Detection channel:** cleanest in the **Tm electric dipole / E12 channel (λ¹/λ²)** at linear drive
  (= the E∥a detector). At strong drive it also shows in the **magnetic H∥c channel (Fe af-$S^z$)**.
- **Nonlinearity:** NLstd scales ≈A⁵ (multi-step cascade: qAFM creation × κ^E bridge × dipole λ⁴/λ⁶ ×
  commutator), so it is a genuine high-order coherent response, vanishing in the κ^E=0 control.

## Files
- configs: `tmfeo3_qafm_e12_kappa/{kappaE_route,amp01,amp01_ctrl,loamp,all_terms}.param`
- analysis: `tmfeo3_qafm_e12_kappa/analyze_xpeak.py`, `plot_xpeak.py`
- figure: `tmfeo3_qafm_e12_kappa/xpeak_amp01_detrend.png`

## Bottom line
Dropping χ is correct, and the (qAFM, E12) cross-peak is **not** lost: it is carried by the
**time-dependent E∥c-assisted exchange κ^E**, which is exactly the Route-C bridge the static-coupling
analysis predicted was the only viable path. It appears with the requested ordering in the E∥a (Tm
E12) detection channel and is causally tied to κ^E.
