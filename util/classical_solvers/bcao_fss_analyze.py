"""Finite-size scan: best energy and fundamental wavevector vs L.

Separates the true incommensurate q from an L-grid lock: the fundamental peak
position and best E/site are reported for each lattice size.
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCRATCH = ("/tmp/claude-1000/-home-pc-linux-ClassicalSpin-Cpp/"
           "2b408709-2c40-4543-b476-93f4f4727eab/scratchpad")
sys.path.insert(0, SCRATCH)
import varspiral as vs

REPO = "/home/pc_linux/ClassicalSpin_Cpp"
a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, np.sqrt(3)/2])
B = 2*np.pi*np.linalg.inv(np.array([a1, a2]).T).T
b1, b2 = B[:, 0], B[:, 1]


def analyze_run(run, L):
    P0, I, Jn, Js = vs.build(L)
    key = {(round(p[0], 6), round(p[1], 6)): i for i, p in enumerate(P0)}
    best = None
    for s in sorted(glob.glob(f"{REPO}/output/{run}/sample_*")):
        f = f"{s}/spins_T=0.txt"
        if not os.path.exists(f):
            continue
        S = np.loadtxt(f); P = np.loadtxt(f"{s}/positions.txt")[:, :2]
        perm = np.array([key[(round(p[0], 6), round(p[1], 6))] for p in P])
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        e = vs.energy_density(S[inv], I, Jn, Js)
        if best is None or e < best[0]:
            best = (e, S, P)
    if best is None:
        return None
    e, S, P = best
    # fundamental q: strongest peak that is not q=0
    peaks = []
    for n1 in range(L):
        for n2 in range(L):
            if n1 == 0 and n2 == 0:
                continue
            q = (n1/L)*b1 + (n2/L)*b2
            v = float((np.abs(S.T @ np.exp(1j*(P @ q)))**2).sum()/len(S))
            h = n1/L - (1 if n1/L > 0.5 else 0)
            k = n2/L - (1 if n2/L > 0.5 else 0)
            peaks.append((v, h, k, np.hypot(h, k)))
    peaks.sort(reverse=True)
    v, h, k, qmag = peaks[0]
    m = np.linalg.norm(S.mean(axis=0))/0.5
    return dict(E=e, h=h, k=k, qmag=qmag, Sq=v, m=m, N=len(S))


if __name__ == "__main__":
    print(f"{'L':>4s} {'E/site':>12s} {'|m|/S':>7s} {'fund (h,k)':>16s} {'|q|':>7s} {'S(q)/N':>9s}")
    for L in (36, 40, 48, 60):
        run = "bcao_slow_L40" if L == 40 else f"bcao_fss_L{L}"
        r = analyze_run(run, L)
        if r:
            print(f"{L:4d} {r['E']:12.6f} {r['m']:7.3f} "
                  f"({r['h']:+.4f},{r['k']:+.4f}) {r['qmag']:7.4f} {r['Sq']:9.2f}")
    print("\nIf |q| is ~constant across L -> genuine incommensurate order.")
    print("If it tracks the nearest n/L grid point -> finite-size lock.")
