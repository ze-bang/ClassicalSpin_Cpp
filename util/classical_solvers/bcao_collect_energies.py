"""Recompute every run's quenched energy exactly and cache to JSON.

The solver's logs are unreliable for this: final_energy.txt is only written by
rank 0, and the twist run's log prints 3 decimals. This re-evaluates every saved
spin config with the real-space evaluator that reproduces the C++
energy_density() to ~1e-6.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCRATCH = ("/tmp/claude-1000/-home-pc-linux-ClassicalSpin-Cpp/"
           "2b408709-2c40-4543-b476-93f4f4727eab/scratchpad")
sys.path.insert(0, SCRATCH)
import varspiral as vs

REPO = "/home/pc_linux/ClassicalSpin_Cpp"
RUNS = {
    "SA, fast cooling (0.90)": "bcao_strong_kitaev_sa_L40",
    "SA, slow cooling (0.97)": "bcao_slow_L40",
    "SA + twisted boundaries": "bcao_twist_L40",
    "LT seed after T=0 quench": "bcao_spiral_seed",
}

P, I, Jn, Js = vs.build(40)
key = {(round(p[0], 6), round(p[1], 6)): i for i, p in enumerate(P)}

out = {}
for label, run in RUNS.items():
    d = os.path.join(REPO, "output", run)
    if not os.path.isdir(d):
        continue
    es = []
    for s in sorted(glob.glob(d + "/sample_*")):
        f = os.path.join(s, "spins_T=0.txt")
        if not os.path.exists(f):
            continue
        S = np.loadtxt(f)
        Pc = np.loadtxt(os.path.join(s, "positions.txt"))[:, :2]
        perm = np.array([key[(round(p[0], 6), round(p[1], 6))] for p in Pc])
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        es.append(float(vs.energy_density(S[inv], I, Jn, Js)))
    if es:
        out[label] = dict(best=min(es), worst=max(es), n=len(es), run=run)
        print(f"{label:28s} n={len(es)}  best={min(es):.6f}  worst={max(es):.6f}")

dst = os.path.join(REPO, "output", "bcao_energies.json")
json.dump(out, open(dst, "w"), indent=2)
print("wrote", dst)
