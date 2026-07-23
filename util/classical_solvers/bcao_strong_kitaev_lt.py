"""Luttinger-Tisza for the BCAO "strong Kitaev" model.

Maksimov, Jiang, Regnault & Chernyshev, "BaCo2(AsO4)2: Strong Kitaev, After All"
(arXiv:2503.20859), parameter set Eq. (3).

Built directly from the paper's Eq. (1)+(2) in the paper's crystallographic frame
(x = a*, y = b, z = c), independently of the C++ builder's conventions, so it serves
as a cross-check on the exchange matrices used by build_bcao_honeycomb.
"""
import numpy as np
from scipy.optimize import minimize_scalar

# --- paper set (3); Delta2 = 0 ("kept XY-like, to avoid overfitting") ---
J1, D1, Jpm, Jzpm = -6.54, 0.36, 0.15, -3.76
J2, D2 = -0.21, 0.0
J3, D3 = 1.70, 0.03
S = 0.5


def J1_bond(phi):
    """NN exchange matrix for a bond at angle phi to the x = a* axis (Eq. 1 + Eq. 2)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([
        [J1 + 2*Jpm*c,   -2*Jpm*s,      Jzpm*s],
        [-2*Jpm*s,       J1 - 2*Jpm*c, -Jzpm*c],
        [Jzpm*s,         -Jzpm*c,       J1*D1 ],
    ])


phis = np.array([0.0, 2*np.pi/3, -2*np.pi/3])          # bond angles to a*
delta = np.array([[np.cos(p), np.sin(p)] for p in phis])  # A -> B, unit NN length
a1, a2 = delta[0] - delta[1], delta[0] - delta[2]       # Bravais vectors
nnn = np.array([a1, a2, a1 - a2, -a1, -a2, -(a1 - a2)])  # 6 second neighbours
third = -2 * delta                                       # 3 third neighbours A->B

J2_mat = np.diag([J2, J2, J2*D2])
J3_mat = np.diag([J3, J3, J3*D3])

A = np.array([a1, a2]).T
B = 2*np.pi*np.linalg.inv(A).T
b1, b2 = B[:, 0], B[:, 1]


def M(q):
    """6x6 Fourier-transformed exchange (2 sublattices x 3 spin components)."""
    JAB = sum(J1_bond(phis[i])*np.exp(1j*np.dot(q, d)) for i, d in enumerate(delta))
    JAB = JAB + sum(J3_mat*np.exp(1j*np.dot(q, d)) for d in third)
    JAA = sum(J2_mat*np.exp(1j*np.dot(q, d)) for d in nnn)
    return np.block([[JAA, JAB], [JAB.conj().T, JAA]])


def lam(k):
    """Lowest LT eigenvalue for q = k*b2 (k in r.l.u.)."""
    return np.linalg.eigvalsh(M(k*b2))[0]


def kitaev_notation():
    """Translate set (3) to {J,K,Gamma,Gamma'} via the paper's Eq. (A2)."""
    s2 = np.sqrt(2)
    T = np.array([
        [ 2/3, 1/3,  2/3, -s2/3],
        [ 0,   0,   -2,    s2  ],
        [-1/3, 1/3, -4/3, -s2/3],
        [-1/3, 1/3,  2/3,  s2/6],
    ])
    return T @ np.array([J1, J1*D1, Jpm, Jzpm])


if __name__ == "__main__":
    print("Kitaev notation {J,K,Gamma,Gamma'} =", np.round(kitaev_notation(), 3))
    print("paper Eq. (4)                     = [-3.3, -5.6, 3.0, 0.6]")
    print()

    # full 2D scan, then refine along the b2 axis
    n = 201
    hs = np.linspace(-0.5, 0.5, n)
    grid = np.array([[np.linalg.eigvalsh(M(h*b1 + k*b2))[0] for k in hs] for h in hs])
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    print(f"2D coarse argmin (h,k) r.l.u. = ({hs[i]:+.4f}, {hs[j]:+.4f})")

    r = minimize_scalar(lam, bracket=(0.10, 0.15, 0.20))
    print(f"LT optimum k = {r.x:.5f} r.l.u.  lambda_min = {r.fun:.6f} meV")
    print(f"LT bound on E/site = {0.5*r.fun*S**2:.6f} meV  (|S| = {S})")
    print("paper: classical spiral k ~ (0.16, 0) r.l.u.")
    print()
    print("commensurate locks allowed by a finite lattice (k = n/L):")
    for L in (36, 40):
        print(f"  L = {L}:")
        for nn_ in range(int(0.10*L), int(0.21*L)+1):
            k = nn_/L
            mark = "   <-- closest" if abs(k - r.x) < 0.5/L else ""
            print(f"    {nn_:2d}/{L}  k = {k:.4f}  lambda = {lam(k):+.6f}  "
                  f"E/site_LT = {0.5*lam(k)*S**2:+.6f}{mark}")
