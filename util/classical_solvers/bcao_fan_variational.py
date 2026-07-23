"""Continuous E(q) for a bunched planar fan — settles the true incommensurate q.

The SA ground state is an in-plane fan: transverse modulation about a common
easy-plane axis, with 2q/3q harmonics (bunching). A fan
    S_i = cos(theta_i) e1 + sin(theta_i) e2,   theta_i = A cos(q.r_i + chi_s)
satisfies |S_i| = S exactly and the cos-of-cos generates the harmonics on its own.

Because q lies along b1, q.r depends only on n1 and sublattice, so the energy is
translation-invariant along a2 and can be summed on a 1-D chain of M cells for ANY
q_val = n/M with M as large as we like. That removes the finite-size lock entirely.
"""
import numpy as np
from scipy.optimize import minimize
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# exchange matrices in the code frame (x=b, y=a*, z=c) -- identical to build_bcao_honeycomb
J1xy, J1z = -6.54, -2.3544
D, E_, F, G = 0.30, 0.0, 0.0, 3.76
J2xy, J2z = -0.21, 0.0
J3xy, J3z = 1.70, 0.051
SLEN = 0.5

J1z_mat = np.array([[J1xy + D, E_, F], [E_, J1xy - D, G], [F, G, J1z]])
c_, s_ = np.cos(2*np.pi/3), np.sin(2*np.pi/3)
U = np.array([[c_, s_, 0], [-s_, c_, 0], [0, 0, 1]])
J1x_mat = U @ J1z_mat @ U.T
J1y_mat = U.T @ J1z_mat @ U
J3_mat = np.diag([J3xy, J3xy, J3z])
J2_mat = np.diag([J2xy, J2xy, J2z])

a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, np.sqrt(3)/2])
posA = np.array([0.0, 0.0]); posB = np.array([0.0, 1/np.sqrt(3)])
POS = [posA, posB]
Bm = 2*np.pi*np.linalg.inv(np.array([a1, a2]).T).T
b1 = Bm[:, 0]

# (matrix, sub_from, sub_to, cell offset d1)  -- d2 is irrelevant to the energy
BONDS = [(J1x_mat, 0, 1, 0), (J1y_mat, 0, 1, 1), (J1z_mat, 0, 1, 0),
         (J3_mat, 0, 1, 1), (J3_mat, 0, 1, -1), (J3_mat, 0, 1, 1),
         (J2_mat, 0, 0, 1), (J2_mat, 0, 0, 0), (J2_mat, 0, 0, 1),
         (J2_mat, 1, 1, 1), (J2_mat, 1, 1, 0), (J2_mat, 1, 1, 1)]
# phase that q.r picks up going from sublattice sa's cell to sb's cell + d1 along a1:
#   dphase = q_val*( 2pi*d1 + b1.(pos_sb - pos_sa) )
DPHASE = [(Jm, sa, sb, 2*np.pi*d1 + np.dot(b1, POS[sb] - POS[sa])) for Jm, sa, sb, d1 in BONDS]
# absolute phase of q.r at sublattice s in cell n1:  q_val*(2pi*n1 + b1.pos_s)
PHI0 = [np.dot(b1, POS[0]), np.dot(b1, POS[1])]


def fan_spins(qv, params, M):
    """Planar fan with a small out-of-plane fan; |S|=SLEN exactly."""
    alpha, A, chiB, Az, chiBz = params
    e1 = np.array([np.cos(alpha), np.sin(alpha), 0.0])   # in-plane mean axis
    e2 = np.array([-np.sin(alpha), np.cos(alpha), 0.0])  # in-plane transverse
    ez = np.array([0.0, 0.0, 1.0])
    n1 = np.arange(M)
    S = np.zeros((M, 2, 3))
    for s in (0, 1):
        ph = qv*(2*np.pi*n1 + PHI0[s]) + (chiB if s else 0.0)
        th = A*np.cos(ph)                       # in-plane fan angle
        z = Az*np.cos(ph + (chiBz if s else 0.0))  # out-of-plane wobble
        z = np.clip(z, -0.999, 0.999)
        ip = np.sqrt(1 - z*z)
        S[:, s, :] = SLEN*(ip[:, None]*(np.cos(th)[:, None]*e1 + np.sin(th)[:, None]*e2)
                           + z[:, None]*ez)
    return S


def energy_density_fan(qv, params, M):
    S = fan_spins(qv, params, M)
    tot = 0.0
    n1 = np.arange(M)
    for (Jm, sa, sb, d1) in BONDS:
        Si = S[:, sa, :]
        Sj = S[(n1 + d1) % M, sb, :]
        tot += np.einsum('ni,ij,nj->', Si, Jm, Sj)
    return tot/(2*M)


def best_at_q(qv, M=600, restarts=12, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = np.array([rng.uniform(0, 2*np.pi), rng.uniform(0, 1.5),
                       rng.uniform(0, 2*np.pi), rng.uniform(0, 0.2),
                       rng.uniform(0, 2*np.pi)])
        r = minimize(lambda p: energy_density_fan(qv, p, M), x0, method="Nelder-Mead",
                     options=dict(maxiter=8000, xatol=1e-9, fatol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    return best.fun, best.x


if __name__ == "__main__":
    # sanity: A=0 must reproduce FM = -1.972500
    e_fm = energy_density_fan(0.0, [np.radians(150), 0, 0, 0, 0], 600)
    print(f"check: fan with A=0  ->  E/site = {e_fm:.6f}  (FM ref -1.972500)")
    print()
    qs = np.linspace(0.0, 0.22, 45)
    Es = []
    for qv in qs:
        e, x = best_at_q(qv, M=600, restarts=8)
        Es.append(e)
    Es = np.array(Es)
    imin = np.argmin(Es)
    # parabolic refine around the discrete minimum
    lo = max(imin-1, 0); hi = min(imin+1, len(qs)-1)
    print(f"{'q (r.l.u.)':>10s} {'E/site (meV)':>14s}")
    for qv, e in zip(qs, Es):
        mark = "  <-- min" if abs(qv-qs[imin]) < 1e-9 else ""
        print(f"{qv:10.4f} {e:14.6f}{mark}")
    # fine scan near the minimum
    qf = np.linspace(qs[lo], qs[hi], 41)
    Ef = np.array([best_at_q(qv, M=1200, restarts=16)[0] for qv in qf])
    qstar = qf[np.argmin(Ef)]
    print()
    print(f"refined optimum:  q* = {qstar:.4f} r.l.u.   E/site = {Ef.min():.6f} meV")
    print(f"period = {1/qstar:.2f} unit cells")
    print(f"SA (L=40, best of 8): q=0.100, E/site=-1.973906")
    np.save("/tmp/claude-1000/-home-pc-linux-ClassicalSpin-Cpp/"
            "2b408709-2c40-4543-b476-93f4f4727eab/scratchpad/fan_Eq.npy",
            np.vstack([qs, Es]))
