"""Extract ordering wavevector, energy and out-of-plane tilt from annealed BCAO configs.

Usage: python3 bcao_analyze_gs.py <run_dir> <L>

Code frame note: build_bcao_honeycomb places the paper's phi=0 exchange matrix on the
bond along the lattice y axis, so in these outputs spin x = b, y = a*, z = c.
"""
import sys, glob, os
import numpy as np

run_dir = sys.argv[1]
L = int(sys.argv[2])

a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2])
A = np.array([a1, a2]).T
B = 2*np.pi*np.linalg.inv(A).T
b1, b2 = B[:, 0], B[:, 1]


def analyze(spin_file, pos_file):
    S = np.loadtxt(spin_file)
    P = np.loadtxt(pos_file)[:, :2]
    N = len(S)
    # scan the allowed q grid (all sublattices together)
    best = (-1.0, 0.0, 0.0)
    for n1 in range(L):
        for n2 in range(L):
            q = (n1/L)*b1 + (n2/L)*b2
            ph = np.exp(1j*(P @ q))
            val = float((np.abs(S.T @ ph)**2).sum()/N)
            if val > best[0]:
                best = (val, n1/L, n2/L)
    tilt = np.degrees(np.arcsin(np.clip(np.abs(S[:, 2])/np.linalg.norm(S, axis=1), 0, 1)))
    m = float(np.linalg.norm(S.mean(axis=0)))
    return best, tilt, m, S


rows = []
print(f"{'sample':9s} {'E/site':>11s} {'h':>7s} {'k':>7s} {'S(q)/N':>8s} "
      f"{'tilt_mean':>9s} {'tilt_max':>8s} {'m_tot':>7s}")
for sample in sorted(glob.glob(os.path.join(run_dir, "sample_*"))):
    pos = os.path.join(sample, "positions.txt")
    spin = os.path.join(sample, "spins_T=0.txt")
    if not os.path.exists(spin):                 # fall back to post-anneal config
        cands = sorted(glob.glob(os.path.join(sample, "spins_T=*.txt")))
        if not cands or not os.path.exists(pos):
            continue
        spin = cands[0]
    best, tilt, m, S = analyze(spin, pos)
    ef = os.path.join(sample, "final_energy.txt")
    E = float(open(ef).read().split(":")[1]) if os.path.exists(ef) else float("nan")
    print(f"{os.path.basename(sample):9s} {E:11.6f} {best[1]:7.4f} {best[2]:7.4f} "
          f"{best[0]:8.3f} {tilt.mean():9.2f} {tilt.max():8.2f} {m:7.4f}")
    rows.append((E, best[1], best[2], tilt.mean(), os.path.basename(sample)))

if rows:
    rows.sort(key=lambda r: (np.isnan(r[0]), r[0]))
    E, h, k, t, name = rows[0]
    print(f"\nlowest-energy sample: {name}  E/site = {E:.6f} meV")
    print(f"  ordering vector (h,k) = ({h:.4f}, {k:.4f}) r.l.u.,  mean |tilt| = {t:.2f} deg")
    print(f"  LT bound = -1.984557 meV/site at k = 0.15038 r.l.u.")
