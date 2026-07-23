"""Capstone: how the BCAO strong-Kitaev classical ground state was settled.

A  E(q) for three variational ansaetze -- only the bunched fan beats FM.
B  finite-size scan -- energy is L-stable, the fundamental locks near q~0.105.
C  the real-space bunched fan -- in-plane angle sweeps and bunches with period ~1/q*.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = "/home/pc_linux/ClassicalSpin_Cpp"
SCRATCH = ("/tmp/claude-1000/-home-pc-linux-ClassicalSpin-Cpp/"
           "2b408709-2c40-4543-b476-93f4f4727eab/scratchpad")
sys.path.insert(0, os.path.join(REPO, "util/classical_solvers"))
sys.path.insert(0, SCRATCH)

E_FM = -1.972500
LT_BOUND = -1.984557
INK = "#22262e"; MUTED = "#5b6472"; GRID = "#e6e9ee"
BLUE = "#2f6fed"; RED = "#d1495b"; TEAL = "#0f9b8e"; AMBER = "#e0952a"

fig = plt.figure(figsize=(15.5, 5.4))
gs = GridSpec(1, 3, figure=fig, wspace=0.30, width_ratios=[1.15, 1.0, 1.05])

# ---------- A: E(q) for the three ansaetze ----------
axA = fig.add_subplot(gs[0, 0])
fan = np.load(f"{SCRATCH}/fan_Eq.npy")
cone = np.load(f"{SCRATCH}/cone_Eq.npy")
mh = np.load(f"{SCRATCH}/multiharm_Eq.npy")
axA.axhline(E_FM, color=MUTED, ls="--", lw=1.3, zorder=2)
axA.text(0.007, E_FM, "  ferromagnet", color=MUTED, fontsize=8.5, va="bottom")
axA.plot(fan[0], fan[1], color=AMBER, lw=2, marker="o", ms=3,
         label="symmetric planar fan")
axA.plot(cone[0], cone[1], color=TEAL, lw=2, marker="s", ms=3,
         label="general single-$q$ cone")
axA.plot(mh[0], mh[1], color=BLUE, lw=2.4, marker="o", ms=4,
         label="bunched fan (3 harmonics)")
imin = int(np.argmin(mh[1]))
axA.plot(mh[0][imin], mh[1][imin], marker="o", ms=10, mfc="none", mec=RED, mew=2, zorder=6)
axA.annotate(f"$q^*\\approx${mh[0][imin]:.2f}\n{mh[1][imin]:.5f}",
             xy=(mh[0][imin], mh[1][imin]), xytext=(10, -6),
             textcoords="offset points", color=RED, fontsize=8.5)
axA.set_title("A · only bunching beats the ferromagnet", fontsize=11, color=INK)
axA.set_xlabel("modulation wavevector $q$ (r.l.u.)", fontsize=9.5)
axA.set_ylabel("$E/N$ (meV per site)", fontsize=9.5)
axA.legend(fontsize=8.5, frameon=False, loc="upper right")
axA.grid(color=GRID, lw=0.8); axA.set_axisbelow(True)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
axA.tick_params(labelsize=8.5)
axA.set_ylim(-1.9745, -1.9715)

# ---------- B: finite-size scan ----------
axB = fig.add_subplot(gs[0, 1])
# (L, E/site, fundamental |q| along the ordering direction)
fss = [(36, -1.973907, 0.1111), (40, -1.973906, 0.1000),
       (48, -1.973872, 0.1042), (60, -1.973906, 0.1000)]
Ls = [f[0] for f in fss]; Es = [f[1] for f in fss]; qq = [f[2] for f in fss]
axB.plot(Ls, Es, color=BLUE, lw=2, marker="o", ms=7, zorder=4)
axB.axhline(E_FM, color=MUTED, ls="--", lw=1.3)
axB.text(37, E_FM, "ferromagnet", color=MUTED, fontsize=8.5, va="bottom")
for L, E, _ in fss:
    axB.annotate(f"{E:.5f}", (L, E), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=7.8, color=INK)
axB.set_title("B · energy is stable with lattice size", fontsize=11, color=INK)
axB.set_xlabel("lattice size $L$", fontsize=9.5)
axB.set_ylabel("best $E/N$ (meV per site)", fontsize=9.5)
axB.set_ylim(-1.9742, -1.9722)
axB.grid(color=GRID, lw=0.8); axB.set_axisbelow(True)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.tick_params(labelsize=8.5)
# inset: q-lock vs L
axBi = axB.inset_axes([0.50, 0.50, 0.46, 0.40])
axBi.axhspan(0.10, 0.11, color=RED, alpha=0.12)
axBi.plot(Ls, qq, color=RED, lw=1.5, marker="D", ms=5)
axBi.set_title("fundamental $q$", fontsize=7.5, color=MUTED, pad=2)
axBi.set_xlabel("$L$", fontsize=7); axBi.set_ylabel("$q$", fontsize=7)
axBi.tick_params(labelsize=6.5)
axBi.set_ylim(0.085, 0.16)
for s in ("top", "right"):
    axBi.spines[s].set_visible(False)

# ---------- C: the real-space bunched fan wave ----------
axC = fig.add_subplot(gs[0, 2])
# reconstruct the winning bunched fan angle profile from the fitted harmonics
# in-plane |A_m| = [0.158, 0.768, 0.0, 0.033]; dominated by the fundamental (a bunched fan)
qstar = 0.11
phi = np.linspace(0, 4*np.pi, 600)          # two spatial periods
A = [0.158, 0.768, 0.0, 0.033]
theta = A[0] + A[1]*np.cos(phi) + A[3]*np.cos(3*phi)   # in-plane fan angle (rad)
axC.plot(phi/(2*np.pi), np.degrees(theta), color=BLUE, lw=2.4, label="bunched fan (actual)")
axC.plot(phi/(2*np.pi), np.degrees(A[1]*np.cos(phi)), color=MUTED, lw=1.4, ls=":",
         label="pure sinusoid (for contrast)")
axC.axhline(0, color=MUTED, lw=1)
axC.set_title("C · the moment fans back and forth, and bunches", fontsize=11, color=INK)
axC.set_xlabel("position along $q$  (modulation periods)", fontsize=9.5)
axC.set_ylabel("in-plane angle from mean axis (deg)", fontsize=9.5)
axC.legend(fontsize=8.5, frameon=False, loc="upper right")
axC.grid(color=GRID, lw=0.8); axC.set_axisbelow(True)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)
axC.tick_params(labelsize=8.5)
axC.text(0.02, -52, "period $1/q^*\\approx9$ cells · net moment in-plane · 2.8° out-of-plane cant",
         fontsize=8, color=MUTED)
axC.set_ylim(-62, 62)

fig.suptitle("Settling the classical ground state of BCAO set (3): a bunched in-plane fan at $q^*\\approx0.11$, "
             "$E/N=-1.9739$ meV (0.0014 below the ferromagnet)",
             fontsize=12, y=1.02)
out = sys.argv[1] if len(sys.argv) > 1 else "bcao_settle.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("wrote", out)
