# FINAL SCENARIO — exact drive and readout channel definitions (2026-08-07)

Model: κB-FREE. One Hamiltonian, all-static Fe–Tm vertices:
W1_yy = 0.01, W1_xz = 0.01, W3_yz = 0.01 + thermal (2000/0.05/0.00333),
d_z^eff = 3.0, kappaB = 0, e1 = 2.067834 meV, e2 = 4.9628 meV.
Pulse: measured THz waveform (experimental_pulse_codeunits.dat) in every run.
Units: energies meV; frequency conversion ω[THz] = (value)/4.135667696;
1 code time unit = 0.6582 ps.

## Component index convention (all datasets)
- SU2 (Fe, per-site sum / uniform F): columns 0,1,2 = S_x, S_y, S_z
  (crystallographic a, b, c).
- SU3 (Tm, transported local Gell-Mann basis): columns 0..7 = λ1..λ8.
  λ1,λ2 = Re/Im of the 1↔2 coherence (E12 = 0.50 THz); λ3 = 1/2 population
  imbalance; λ4,λ5 = 1↔3 (E13 = 1.20); λ6,λ7 = 2↔3 (E23 = 0.70);
  λ8 = diagonal.
- quad = local product channel (λ1·λ2 summed over Tm sites, computed from
  M_local_SU3): the hyperpolarizability emission coordinate, radiates at 2E12.

## GEOMETRY A (one experiment, two analysers)  —  runs/geomA_gs{1,2,3}
DRIVE (single pulse, E∥c / H∥a):
  - Fe: Zeeman B∥a · f(t), amplitude 0.12.
  - Tm (su3 vector, manual): E∥c d_z^eff → 3.0·λ2 ; B∥a μ_x → 0.9128·λ7
    (dark-μ13: λ5 component = 0). Amplitude 0.021950.
CHANNEL A-cross — analyser (E∥a, H∥c) out:
  S(t) = F_z + 5.264·λ2 (m_z, M1) + 5.264·λ1 (d_x, E1) + β·quad,  β = 66
  [F_z ≡ 0 in this channel by symmetry — verified 1e-15;
   NO λ3/λ8 (population readout removed 2026-08-07); β calibrated from the
   measured A-cross parity (E12,2E12)/(qAFM,E12) = 0.97.]
  Dataset columns: SU2[2], SU3[1], SU3[0], quad.
CHANNEL A-same-pol — analyser (E∥c, H∥a) out:
  S(t) = F_x + 2.3915·λ5 + 0.9128·λ7 (m_x, M1) + 0.006·λ4 + 4.4·λ6 (d_z, E1)
  [mirror-odd analyser ⇒ NO λ3/λ8. Atlas panel plots the CEF part
   0.006·λ4 + 4.4·λ6; the Fe/Tm relative weight (R≈300) is the one
   uncalibrated detection scale.]
  Dataset columns: SU2[0], SU3[4], SU3[6], SU3[3], SU3[5].

## GEOMETRY B (one experiment, two analysers)  —  runs/geomB_gs1
DRIVE (single pulse, E∥a / H∥c):
  - Fe: Zeeman B∥c · f(t), amplitude 0.10.
  - Tm (su3 vector, manual, auto_su3_pump = false):
      B∥c μ_z → +0.030711·λ2   (magnetic quadrature)
      E∥a d_x → −0.016500·λ1   (electric quadrature; RELATIVE SIGN − REQUIRED)
    pump_direction_su3 = -0.0165, 0.030711, 0,0,0,0,0,0 ; amplitude 0.034864.
    ⇒ w_E1 = 0.0165/0.0307 = 0.54 in μ units — the E1/M1 balance, FIXED by
    the measured (E12,qAFM)/(qFM,qAFM) = 0.43–0.46.
CHANNEL B-cross — analyser (E∥c, H∥a) out:
  S(t) = m_x = Fe S_x (M1).  Tm part 2.3915·λ5 + 0.9128·λ7 ≡ 0 exactly
  (SU(2) subalgebra closure: drive touches only λ1, λ2 ⇒ λ4..λ7 never move;
  verified 0.00 in this run). d_z part (λ4, λ6) ≡ 0 likewise.
  Dataset column: SU2[0]. This is the (qFM,qAFM)=1.00 / (E12,qAFM)=0.45 map.
CHANNEL B-same-pol — analyser (E∥a, H∥c) out (same operator as A-cross):
  S(t) = S_z (F_z, alive here) + 5.264·λ2 + 5.264·λ1 + β·quad,  β = 66,
  NO λ3/λ8. In practice the channel is >99% the β·quad harmonic term
  (see figures/Bsame_decomposition.png): prediction = single broad ridge at
  ω_τ = E12 peaked at ω_t = 2E12 = 1.00 THz.
  Dataset columns: SU2[2], SU3[1], SU3[0], quad.

## Mechanism of the geometry-B transfer peak (κB-free)
E∥a writes the E12 coherence through d_x·λ1 (polarisation-owned: this drive
does not exist in geometry A). The static W1_xz S_xS_z λ1 converts the stored
coherence into the qAFM magnon, which radiates through Fe m_x. The E×B
quadrature interference (− sign) places the transfer row at ω_τ = 0.52,
identical to the measured position; ratio dialed by w_E1.
