"""Exact enumeration of collinear orders for the BCAO strong-Kitaev model.

For a collinear state S_i = S*sigma_i*n (sigma_i = +-1), the energy is
    E = S^2 * n^T [sum_bonds sigma_i sigma_j J_ij] n,
so the best spin axis n is the lowest eigenvector of that 3x3 matrix. We can
therefore enumerate every sign pattern in a magnetic cell exactly.

Covers FM, Neel, zigzag, stripe and double-zigzag (and every other pattern that
fits the cell), for all magnetic cells with p1*p2*2 <= 16 sites.
"""
import itertools
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

J1xy, J1z = -6.54, -2.3544
D, E_, F, G = 0.30, 0.0, 0.0, 3.76
J2xy, J2z = -0.21, 0.0
J3xy, J3z = 1.70, 0.051
SLEN = 0.5

J1z_mat = np.array([[J1xy + D, E_, F], [E_, J1xy - D, G], [F, G, J1z]])
c, s = np.cos(2*np.pi/3), np.sin(2*np.pi/3)
U = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
J1x_mat = U @ J1z_mat @ U.T
J1y_mat = U.T @ J1z_mat @ U
J3_mat = np.diag([J3xy, J3xy, J3z])
J2_mat = np.diag([J2xy, J2xy, J2z])

BONDS = [
    (J1x_mat, 0, 1, (0, -1)), (J1y_mat, 0, 1, (1, -1)), (J1z_mat, 0, 1, (0, 0)),
    (J3_mat, 0, 1, (1, 0)), (J3_mat, 0, 1, (-1, 0)), (J3_mat, 0, 1, (1, -2)),
    (J2_mat, 0, 0, (1, 0)), (J2_mat, 0, 0, (0, 1)), (J2_mat, 0, 0, (1, -1)),
    (J2_mat, 1, 1, (1, 0)), (J2_mat, 1, 1, (0, 1)), (J2_mat, 1, 1, (1, -1)),
]

NAMES = {}


def scan(p1, p2, L=12):
    """Enumerate sign patterns on a p1 x p2 magnetic cell, tiled onto an L x L lattice."""
    assert L % p1 == 0 and L % p2 == 0
    nsite = p1*p2*2
    idx = lambda n1, n2, sub: ((n1 % p1)*p2 + (n2 % p2))*2 + sub
    # accumulate, for each bond, the pair (site_a, site_b) in the magnetic cell
    pairs = []
    for Jm, sa, sb, off in BONDS:
        for n1 in range(L):
            for n2 in range(L):
                ia = idx(n1, n2, sa)
                ib = idx(n1 + off[0], n2 + off[1], sb)
                pairs.append((ia, ib, Jm))
    # group: coefficient matrix C[a,b] = number of bonds linking cell-sites a,b (with J)
    acc = np.zeros((nsite, nsite, 3, 3))
    for ia, ib, Jm in pairs:
        acc[ia, ib] += Jm
    N = L*L*2

    best = None
    for bits in itertools.product([1, -1], repeat=nsite-1):
        sig = np.array((1,) + bits)          # fix global sign
        M = np.einsum('a,b,abij->ij', sig, sig, acc)
        w, v = np.linalg.eigh(M)
        e = SLEN**2*w[0]/N
        if best is None or e < best[0]:
            best = (e, sig.copy(), v[:, 0])
    return best


if __name__ == "__main__":
    print(f"{'cell':>8s} {'E/site (meV)':>14s}   pattern (sublattice-interleaved)")
    results = []
    for (p1, p2) in [(1, 1), (1, 2), (2, 1), (2, 2), (4, 1), (1, 4), (4, 2), (2, 4)]:
        if p1*p2*2 > 16:
            continue
        e, sig, n = scan(p1, p2)
        results.append((e, p1, p2, sig, n))
        print(f"{p1}x{p2:<6d} {e:14.6f}   {''.join('+' if x > 0 else '-' for x in sig)}")
    results.sort()
    e, p1, p2, sig, n = results[0]
    print(f"\nbest collinear order: {p1}x{p2} cell, E/site = {e:.6f} meV")
    print(f"  spin axis n = ({n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f})  [code frame: x=b, y=a*, z=c]")
    print(f"  out-of-plane angle = {np.degrees(np.arcsin(abs(n[2]))):.2f} deg")
    print(f"\nreference: FM = -1.972500,  SA best = -1.973700,  LT bound = -1.984557")
