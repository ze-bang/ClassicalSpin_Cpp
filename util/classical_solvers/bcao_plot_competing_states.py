"""Visualize the competing classical states of the BCAO strong-Kitaev model.

Rows: real-space spin texture, static spin structure factor (SSSF), and an energy ladder.

Usage: python3 bcao_plot_competing_states.py <outfile.png>
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm, LogNorm

REPO = "/home/pc_linux/ClassicalSpin_Cpp"
SP = ("/tmp/claude-1000/-home-pc-linux-ClassicalSpin-Cpp/"
      "2b408709-2c40-4543-b476-93f4f4727eab/scratchpad")
L = 40
SLEN = 0.5
LT_BOUND = -1.984557
E_FM = -1.972500

a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, np.sqrt(3)/2])
Bm = 2*np.pi*np.linalg.inv(np.array([a1, a2]).T).T
b1, b2 = Bm[:, 0], Bm[:, 1]

# --- states: (label, spins, positions, energy, note) -------------------------
def load(sp, po):
    return np.loadtxt(sp), np.loadtxt(po)[:, :2]


states = []

S, P = load(f"{SP}/varspiral_seed_raw.txt", f"{SP}/varspiral_pos.txt")
states.append(dict(name="Best circular spiral\n(variational)", S=S, P=P, E=E_FM,
                   note="collapses to q=0 (= FM)"))

S, P = load(f"{SP}/spiral_seed.txt",
            f"{REPO}/output/bcao_strong_kitaev_sa_L40/sample_0/positions.txt")
states.append(dict(name="LT eigenvector k=0.15\n(hard-normalized seed)", S=S, P=P, E=None,
                   note="elliptical SDW, not realizable"))

S, P = load(f"{REPO}/output/bcao_spiral_seed/sample_0/spins_T=0.txt",
            f"{REPO}/output/bcao_spiral_seed/sample_0/positions.txt")
states.append(dict(name="LT seed after T=0 quench", S=S, P=P, E=-1.95704,
                   note="relaxes to a worse minimum"))

# the SA runs: pick each run's *best* sample, scored with the verified evaluator
sys.path.insert(0, SP)
import varspiral as vs
_P, _I, _Jn, _Js = vs.build(L)
_key = {(round(p[0], 6), round(p[1], 6)): i for i, p in enumerate(_P)}


def best_sample(run):
    """Return (spins, positions, E) for the lowest-energy trial of a run."""
    import glob
    out = []
    for s in sorted(glob.glob(f"{REPO}/output/{run}/sample_*")):
        f = f"{s}/spins_T=0.txt"
        if not os.path.exists(f):
            continue
        Sx = np.loadtxt(f); Px = np.loadtxt(f"{s}/positions.txt")[:, :2]
        perm = np.array([_key[(round(p[0], 6), round(p[1], 6))] for p in Px])
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        out.append((float(vs.energy_density(Sx[inv], _I, _Jn, _Js)), Sx, Px))
    if not out:
        return None
    out.sort(key=lambda r: r[0])
    return out[0]


for run, nm, note in (("bcao_strong_kitaev_sa_L40", "SA, fast cooling\n(best of 4)", "under-converged: diffuse"),
                      ("bcao_slow_L40", "SA, slow cooling\n(best of 8)", "FM + q=0.10 + harmonics 2q,3q"),
                      ("bcao_twist_L40", "SA, twisted BCs\n(best of 8)", "same, rotated q; twist→0")):
    r = best_sample(run)
    if r:
        e, Sx, Px = r
        states.append(dict(name=nm, S=Sx, P=Px, E=e, note=note))


def sssf(S, P):
    """|S(q)|^2 / N on the allowed q grid, returned on an (L,L) grid of (h,k)."""
    out = np.zeros((L, L))
    for n1 in range(L):
        for n2 in range(L):
            q = (n1/L)*b1 + (n2/L)*b2
            out[n1, n2] = (np.abs(S.T @ np.exp(1j*(P @ q)))**2).sum()/len(S)
    return out


n = len(states)
fig = plt.figure(figsize=(3.3*n, 9.4))
gs = GridSpec(3, n, figure=fig, height_ratios=[1.15, 1.15, 0.85],
              hspace=0.42, wspace=0.28)

# ---------------- row 1: real-space spin texture ----------------
for i, st in enumerate(states):
    ax = fig.add_subplot(gs[0, i])
    S, P = st["S"], st["P"]
    # a rectangular patch from the middle of the lattice (avoids the a2 shear)
    m = ((P[:, 0] > 18) & (P[:, 0] < 32) & (P[:, 1] > 8) & (P[:, 1] < 20))
    sz = S[m, 2]/SLEN
    q = ax.quiver(P[m, 0], P[m, 1], S[m, 0], S[m, 1], sz,
                  cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1),
                  pivot="mid", scale=9, width=0.008, edgecolor="none")
    ax.set_aspect("equal")
    ax.set_title(st["name"], fontsize=9.5, pad=7)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c9ced6")
    if i == n-1:
        cb = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("$S_z/S$ (out of plane)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    tilt = np.degrees(np.arcsin(np.clip(np.abs(S[:, 2])/np.linalg.norm(S, axis=1), 0, 1)))
    mtot = np.linalg.norm(S.mean(axis=0))/SLEN
    ax.set_xlabel(f"$|m|$ = {mtot:.2f}$S$   mean tilt = {tilt.mean():.1f}°", fontsize=8)

# ---------------- row 2: SSSF ----------------
maps = [sssf(st["S"], st["P"]) for st in states]
vmax = max(mm.max() for mm in maps)
for i, (st, mm) in enumerate(zip(states, maps)):
    ax = fig.add_subplot(gs[1, i])
    # plot on (h,k) axes centred at 0, wrapped to [-0.5, 0.5)
    hk = np.fft.fftshift(mm)
    ext = [-0.5, 0.5, -0.5, 0.5]
    # log scale: a single Bragg pixel otherwise saturates everything else to black
    im = ax.imshow(np.maximum(hk.T, 1e-2), origin="lower", extent=ext, cmap="magma",
                   norm=LogNorm(vmin=1e-2, vmax=vmax), aspect="equal",
                   interpolation="nearest")
    ax.set_xlabel("$h$ (r.l.u.)", fontsize=8.5)
    if i == 0:
        ax.set_ylabel("$k$ (r.l.u.)", fontsize=8.5)
    ax.tick_params(labelsize=7)
    # mark the LT wavevector
    ax.plot([-0.15, 0.15], [0, 0], ls="none", marker="o", mfc="none",
            mec="#39d3c3", mew=1.4, ms=9)
    ax.set_title(st["note"], fontsize=8.5, color="#4a5261", pad=5)
    if i == n-1:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("$|S(q)|^2/N$", fontsize=8)
        cb.ax.tick_params(labelsize=7)

# ---------------- row 3: energy ladder ----------------
ax = fig.add_subplot(gs[2, :])
lab, val = [], []
for st in states:
    if st["E"] is not None:
        lab.append(st["name"].replace("\n", " "))
        val.append(st["E"])
lab.append("FM (realizable, k=0)"); val.append(E_FM)
order = np.argsort(val)
lab = [lab[i] for i in order]; val = [val[i] for i in order]
ypos = np.arange(len(val))
bars = ax.barh(ypos, np.array(val) - LT_BOUND, left=LT_BOUND, height=0.55,
               color="#7c8698")
bars[0].set_color("#2f6fed")
ax.axvline(LT_BOUND, color="#d1495b", lw=2)
ax.annotate(f"Luttinger–Tisza bound {LT_BOUND:.4f}\n"
            "NOT achievable — its eigenvector is an\nelliptical SDW (hard-spin constraint violated)",
            xy=(LT_BOUND, -1.6), xytext=(14, 0), textcoords="offset points",
            color="#d1495b", fontsize=8.5, va="center", ha="left")
ax.set_yticks(ypos); ax.set_yticklabels(lab, fontsize=9)
ax.set_xlabel("energy density  $E/N$  (meV per site,  $|S|=1/2$)", fontsize=9.5)
ax.set_xlim(LT_BOUND - 0.0015, max(val) + 0.007)
ax.set_ylim(len(val) - 0.5, -2.7)
for y, v in zip(ypos, val):
    ax.text(v + 0.0004, y, f"{v:.6f}", va="center", fontsize=8.5, color="#22262e")
ax.grid(axis="x", color="#e6e9ee", lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c9ced6")
ax.tick_params(labelsize=8)

fig.suptitle("BCAO strong-Kitaev model (arXiv:2503.20859, set 3) — competing classical states, $L=40$",
             fontsize=12.5, y=0.985)
out = sys.argv[1] if len(sys.argv) > 1 else "bcao_competing_states.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("wrote", out)
for st in states:
    e = f"{st['E']:.6f}" if st["E"] is not None else "n/a"
    print(f"  {st['name'][:38]:40s} E/site = {e}")
