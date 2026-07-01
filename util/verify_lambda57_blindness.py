#!/usr/bin/env python3
"""verify_lambda57_blindness.py

Explicit verification of the EOM-level selection rules behind the
qAFM <-> E1-E2 (lambda^2) cross-peak in TmFeO3.

Central claims tested (single-site SU(3) Tm qutrit, equilibrium under H2=chi2z):

 (1) A pure "chi-form" drive (only delta H^2, conjugate to lambda^2) leaves
     lambda^5 and lambda^7 IDENTICALLY zero  ==>  they are "blind".
 (2) The torque on lambda^2 that comes from the OTHER (off-diagonal) sectors
     is  +1/2 ( H^5 lambda^7 - H^7 lambda^5 ).
     With lambda^5 = lambda^7 = 0 this term is exactly zero, so a chi-form
     drive can only move lambda^2 through the *within-sector* (1-2) route,
     which is longitudinal and suppressed by e1/H2.
 (3) Only a transverse drive H^5 (or H^7) seeds lambda^7 (or lambda^5) and
     thereby opens a finite ( H^5 lambda^7 - H^7 lambda^5 ) torque on lambda^2.

This isolates exactly which Hamiltonian vertex the current model is missing
in order to convert a qAFM magnon into an E1-E2 (lambda^2) coherence.
"""
import numpy as np
from scipy.integrate import solve_ivp

# ---- physical parameters (same as verify_xpeak_eom.py) ----
e1 = 0.97
e2 = 3.97
H2 = 11.999939          # chi2z effective field on a single Tm site
omega_qAFM = 3.84
A_drive = 0.05
h8 = (2 * e2 - e1) / np.sqrt(3.)


def su3_eom(h, lam):
    dl = np.zeros(8); s3 = np.sqrt(3.) / 2.
    dl[0] = (h[1]*lam[2]-h[2]*lam[1]+.5*(h[3]*lam[6]-h[6]*lam[3])-.5*(h[4]*lam[5]-h[5]*lam[4]))
    dl[1] = (-(h[0]*lam[2]-h[2]*lam[0])+.5*(h[3]*lam[5]-h[5]*lam[3])+.5*(h[4]*lam[6]-h[6]*lam[4]))
    dl[2] = (h[0]*lam[1]-h[1]*lam[0]+.5*(h[3]*lam[4]-h[4]*lam[3])-.5*(h[5]*lam[6]-h[6]*lam[5]))
    dl[3] = (-0.5*(h[0]*lam[6]-h[6]*lam[0])-0.5*(h[1]*lam[5]-h[5]*lam[1])-0.5*(h[2]*lam[4]-h[4]*lam[2])+s3*(h[4]*lam[7]-h[7]*lam[4]))
    dl[4] = (+0.5*(h[0]*lam[5]-h[5]*lam[0])-0.5*(h[1]*lam[6]-h[6]*lam[1])+0.5*(h[2]*lam[3]-h[3]*lam[2])-s3*(h[3]*lam[7]-h[7]*lam[3]))
    dl[5] = (-0.5*(h[0]*lam[4]-h[4]*lam[0])+0.5*(h[1]*lam[3]-h[3]*lam[1])+0.5*(h[2]*lam[6]-h[6]*lam[2])+s3*(h[6]*lam[7]-h[7]*lam[6]))
    dl[6] = (+0.5*(h[0]*lam[3]-h[3]*lam[0])+0.5*(h[1]*lam[4]-h[4]*lam[1])-0.5*(h[2]*lam[5]-h[5]*lam[2])-s3*(h[5]*lam[7]-h[7]*lam[5]))
    dl[7] = (s3*(h[3]*lam[4]-h[4]*lam[3])+s3*(h[5]*lam[6]-h[6]*lam[5]))
    return dl


def find_eq_single(H2val):
    """Ground-state Gell-Mann vector of one Tm site by exact diagonalization."""
    s3 = np.sqrt(3.)
    M = np.zeros((3, 3), dtype=complex)
    M[0, 0] = e1 + h8 / s3
    M[1, 1] = -e1 + h8 / s3
    M[2, 2] = -2 * h8 / s3
    M[0, 1] = -1j * H2val
    M[1, 0] = 1j * H2val
    ev, ec = np.linalg.eigh(M)
    psi = ec[:, np.argmin(ev)]
    lam = np.zeros(8)
    lam[0] = 2. * np.real(np.conj(psi[0]) * psi[1])
    lam[1] = (np.conj(psi[0]) * (-1j) * psi[1] + np.conj(psi[1]) * (1j) * psi[0]).real
    lam[2] = abs(psi[0])**2 - abs(psi[1])**2
    lam[3] = 2. * np.real(np.conj(psi[0]) * psi[2])
    lam[4] = (np.conj(psi[0]) * (-1j) * psi[2] + np.conj(psi[2]) * (1j) * psi[0]).real
    lam[5] = 2. * np.real(np.conj(psi[1]) * psi[2])
    lam[6] = (np.conj(psi[1]) * (-1j) * psi[2] + np.conj(psi[2]) * (1j) * psi[1]).real
    lam[7] = (abs(psi[0])**2 + abs(psi[1])**2 - 2 * abs(psi[2])**2) / s3
    return lam


def static_field():
    h = np.zeros(8)
    h[1] = H2          # H^2 = chi2z
    h[2] = e1          # H^3 = bare CEF gap
    h[7] = h8          # H^8
    return h


def run(drive_channel, t_end=400.0, dt=0.01):
    """Integrate one Tm site with an oscillating drive on one H-channel.
    drive_channel in {2,5,7} (1-based Gell-Mann index)."""
    lam0 = find_eq_single(H2)
    h_static = static_field()
    ch = drive_channel - 1  # 0-based

    def rhs(t, lam):
        h = h_static.copy()
        h[ch] += A_drive * np.sin(omega_qAFM * t)
        return su3_eom(h, lam)

    ts = np.arange(0, t_end, dt)
    sol = solve_ivp(rhs, (0, t_end), lam0, t_eval=ts, rtol=1e-10, atol=1e-12, method="DOP853")
    return ts, sol.y, lam0


def rms(x):
    return float(np.sqrt(np.mean(x**2)))


print("=" * 72)
print("TmFeO3  qAFM -> E1-E2  EOM selection-rule verification (single Tm site)")
print("=" * 72)
lam0 = find_eq_single(H2)
print("Equilibrium Gell-Mann vector (H2=%.4f):" % H2)
print("  lam1=%.4f lam2=%.4f lam3=%.4f lam4=%.4f lam5=%.4f lam6=%.4f lam7=%.4f lam8=%.4f"
      % tuple(lam0))
print("  dressed E1-E2 gap  omega_12 = sqrt(e1^2+H2^2) = %.4f meV" % np.hypot(e1, H2))
print("  longitudinal-tilt factor e1/omega_12 = %.4f" % (e1 / np.hypot(e1, H2)))

# -------- TEST 1: chi-form drive (delta H^2 only) --------
print("\n" + "-" * 72)
print("TEST 1:  pure chi-form drive  delta H^2(t) = A sin(omega_qAFM t)")
print("-" * 72)
t, y, _ = run(drive_channel=2)
i = len(t) // 2  # discard transient
print("  RMS lambda^5 = %.3e   RMS lambda^7 = %.3e   (expect ~0  -> BLIND)"
      % (rms(y[4, i:]), rms(y[6, i:])))
print("  RMS delta lambda^2 = %.3e   RMS delta lambda^1 = %.3e   (within-1-2-sector only)"
      % (rms(y[1, i:] - lam0[1]), rms(y[0, i:] - lam0[0])))
blind = rms(y[4, i:]) < 1e-11 and rms(y[6, i:]) < 1e-11
print("  ==> lambda^5, lambda^7 BLIND to chi-form drive: %s" % blind)

# -------- TEST 2: transverse drive (delta H^5 only) --------
print("\n" + "-" * 72)
print("TEST 2:  transverse drive  delta H^5(t) = A sin(omega_qAFM t)")
print("-" * 72)
t, y, _ = run(drive_channel=5)
i = len(t) // 2
print("  RMS lambda^7 = %.3e   (seeded by  -1/2 H^5 lambda^2,  lambda^2~%.3f)"
      % (rms(y[6, i:]), lam0[1]))
print("  RMS lambda^5 = %.3e" % rms(y[4, i:]))
print("  RMS delta lambda^2 = %.3e   (torqued by +1/2 (H^5 lambda^7 - H^7 lambda^5))"
      % rms(y[1, i:] - lam0[1]))

# -------- TEST 3: term-by-term torque on lambda^2 --------
print("\n" + "-" * 72)
print("TEST 3:  decomposition of  d(lambda^2)/dt  at a sample time (TEST-2 run)")
print("-" * 72)
ts = t[i]
lam = y[:, i]
h = static_field(); h[4] += A_drive * np.sin(omega_qAFM * ts)
term_intra = -(h[0]*lam[2] - h[2]*lam[0])          # within 1-2 sector
term_46 = 0.5 * (h[3]*lam[5] - h[5]*lam[3])          # 1-3 / 2-3 (H4,H6)
term_57 = 0.5 * (h[4]*lam[6] - h[6]*lam[4])          # H5 lam7 - H7 lam5
print("  -(H1 lam3 - H3 lam1)           = %+.4e   (intra 1-2, longitudinal)" % term_intra)
print("  +1/2 (H4 lam6 - H6 lam4)       = %+.4e" % term_46)
print("  +1/2 (H5 lam7 - H7 lam5)       = %+.4e   <-- cross-sector torque" % term_57)
print("  d(lambda^2)/dt total           = %+.4e" % (term_intra + term_46 + term_57))

print("\n" + "=" * 72)
print("CONCLUSION")
print("=" * 72)
print("""  * chi-form (H^2) drive cannot touch lambda^5/lambda^7  -> they stay 0.
  * the cross-sector torque +1/2(H^5 lam7 - H^7 lam5) on lambda^2 is therefore
    identically zero for any chi-only coupling.
  * a finite H^5 (or H^7) vertex is REQUIRED to seed lambda^7 (lambda^5) and
    open the qAFM -> E1-E2 conversion channel.""")
