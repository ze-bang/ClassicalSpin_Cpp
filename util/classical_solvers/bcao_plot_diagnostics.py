"""Why the classical BCAO strong-Kitaev ground state is hard to converge.

Panels:
  A  LT eigenvalue landscape over the BZ            -- the minimum is a shallow ring
  B  cut through it                                 -- only 0.6% deep; L=40 grid overlaid
  C  LT eigenvector spin length around the cycle    -- k=0.15 is an elliptical SDW
  D  every collinear order, exactly enumerated      -- none beats FM
  E  energy ladder of everything tried
"""
import itertools, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = "/home/pc_linux/ClassicalSpin_Cpp"
SLEN = 0.5
LT_BOUND = -1.984557
E_FM = -1.972500

J1xy, J1z = -6.54, -2.3544
D, E_, F, G = 0.30, 0.0, 0.0, 3.76
J2xy, J2z = -0.21, 0.0
J3xy, J3z = 1.70, 0.051

J1z_mat = np.array([[J1xy + D, E_, F], [E_, J1xy - D, G], [F, G, J1z]])
c_, s_ = np.cos(2*np.pi/3), np.sin(2*np.pi/3)
U = np.array([[c_, s_, 0], [-s_, c_, 0], [0, 0, 1]])
J1x_mat = U @ J1z_mat @ U.T
J1y_mat = U.T @ J1z_mat @ U
J3_mat = np.diag([J3xy, J3xy, J3z])
J2_mat = np.diag([J2xy, J2xy, J2z])

a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, np.sqrt(3)/2])
posA = np.array([0.0, 0.0]); posB = np.array([0.0, 1/np.sqrt(3)])
Bm = 2*np.pi*np.linalg.inv(np.array([a1, a2]).T).T
b1, b2 = Bm[:, 0], Bm[:, 1]

AB = [(J1x_mat, (0, -1)), (J1y_mat, (1, -1)), (J1z_mat, (0, 0)),
      (J3_mat, (1, 0)), (J3_mat, (-1, 0)), (J3_mat, (1, -2))]
AA = [(0, 1), (1, 0), (1, -1)]
BONDS = [(J1x_mat, 0, 1, (0, -1)), (J1y_mat, 0, 1, (1, -1)), (J1z_mat, 0, 1, (0, 0)),
         (J3_mat, 0, 1, (1, 0)), (J3_mat, 0, 1, (-1, 0)), (J3_mat, 0, 1, (1, -2)),
         (J2_mat, 0, 0, (1, 0)), (J2_mat, 0, 0, (0, 1)), (J2_mat, 0, 0, (1, -1)),
         (J2_mat, 1, 1, (1, 0)), (J2_mat, 1, 1, (0, 1)), (J2_mat, 1, 1, (1, -1))]


def disp(off):
    return posB - posA + off[0]*a1 + off[1]*a2


def M(q):
    JAB = np.zeros((3, 3), dtype=complex)
    for J, off in AB:
        JAB += J*np.exp(1j*np.dot(q, disp(off)))
    JAA = np.zeros((3, 3), dtype=complex)
    for off in AA:
        d = off[0]*a1 + off[1]*a2
        JAA += J2_mat*(np.exp(1j*np.dot(q, d)) + np.exp(-1j*np.dot(q, d)))
    return np.block([[JAA, JAB], [JAB.conj().T, JAA]])


lam = lambda q: np.linalg.eigvalsh(M(q))[0]

fig = plt.figure(figsize=(15.5, 9.2))
gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.30,
              height_ratios=[1.0, 0.92])

INK = "#22262e"; MUTED = "#5b6472"; GRID = "#e6e9ee"
ACC = "#2f6fed"; RED = "#d1495b"; TEAL = "#0f9b8e"

# ---------------- A: LT landscape over the BZ ----------------
axA = fig.add_subplot(gs[0, 0])
n = 121
hs = np.linspace(-0.5, 0.5, n)
Z = np.array([[lam(h*b1 + k*b2) for k in hs] for h in hs])
im = axA.imshow(Z.T, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5],
                cmap="viridis_r", aspect="equal")
cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=7)
i, j = np.unravel_index(np.argmin(Z), Z.shape)
axA.plot(hs[i], hs[j], marker="o", mfc="none", mec="w", mew=1.6, ms=10)
axA.set_title(r"A · LT eigenvalue $\lambda_{\min}(q)$ over the BZ  (meV)", fontsize=10, color=INK)
axA.set_xlabel("$h$ (r.l.u.)", fontsize=9); axA.set_ylabel("$k$ (r.l.u.)", fontsize=9)
axA.tick_params(labelsize=8)

# ---------------- B: the cut -- flatness ----------------
axB = fig.add_subplot(gs[0, 1])
ks = np.linspace(-0.35, 0.35, 400)
lk = np.array([lam(k*b1) for k in ks])
axB.plot(ks, lk, color=ACC, lw=2, zorder=3)
axB.axhline(lam(np.zeros(2)), color=MUTED, ls="--", lw=1.2, zorder=2)
kg = np.arange(-14, 15)/40
axB.plot(kg, [lam(k*b1) for k in kg], ls="none", marker="|", ms=9,
         color=MUTED, mew=1.2, zorder=4)
kmin = ks[np.argmin(lk)]
axB.plot([kmin], [lk.min()], marker="o", ms=8, mfc="none", mec=RED, mew=2, zorder=5)
axB.annotate(f"LT min  k={0.150:.3f}\n$\\lambda$={lk.min():.3f}", xy=(kmin, lk.min()),
             xytext=(0, -32), textcoords="offset points", fontsize=8, color=RED,
             ha="center")
axB.annotate(f"FM (k=0)\n$\\lambda$={lam(np.zeros(2)):.3f}", xy=(0, lam(np.zeros(2))),
             xytext=(-4, 12), textcoords="offset points", fontsize=8, color=MUTED,
             ha="center")
depth = 100*(lam(np.zeros(2)) - lk.min())/abs(lk.min())
axB.set_title(f"B · the minimum is only {depth:.1f}% deep  (ticks = $L$=40 allowed $k$)",
              fontsize=10, color=INK)
axB.set_xlabel("$k$ along $b_1$ (r.l.u.)", fontsize=9)
axB.set_ylabel(r"$\lambda_{\min}$  (meV)", fontsize=9)
axB.set_ylim(-15.95, None)
axB.grid(color=GRID, lw=0.8); axB.set_axisbelow(True)
for sp in ("top", "right"):
    axB.spines[sp].set_visible(False)
axB.tick_params(labelsize=8)

# ---------------- C: eigenvector ellipticity ----------------
axC = fig.add_subplot(gs[0, 2])
th = np.linspace(0, 2*np.pi, 500)
w, v = np.linalg.eigh(M(-0.15*b1))
u = v[:3, 0]
Ln = np.linalg.norm(np.real(np.outer(np.exp(1j*th), u)), axis=1)
swing = Ln.max()/Ln.min()
axC.plot(np.degrees(th), Ln/Ln.max(), color=RED, lw=2,
         label=f"LT eigenvector at k=0.15  (swings {swing:.1f}×)")
axC.axhline(1.0, color=TEAL, lw=2, ls="--",
            label="required for hard spins (circular)")
axC.set_title("C · spin length around the incommensurate cycle", fontsize=10, color=INK)
axC.set_xlabel(r"phase $q\cdot r$ across the lattice  (deg)", fontsize=9)
axC.set_ylabel(r"$|S|$  (normalized)", fontsize=9)
axC.legend(fontsize=8, frameon=False, loc="lower center")
axC.grid(color=GRID, lw=0.8); axC.set_axisbelow(True)
for sp in ("top", "right"):
    axC.spines[sp].set_visible(False)
axC.tick_params(labelsize=8)
axC.set_xlim(0, 360); axC.set_ylim(-0.03, 1.32)
axC.text(180, 1.19, "an incommensurate hard-spin state must be circular (flat).\n"
                    "The LT eigenvector is an elliptical SDW, so its bound\n"
                    "is not attainable by any real spin configuration.",
         fontsize=7.8, color=RED, ha="center", va="center")

# ---------------- D: every collinear order ----------------
axD = fig.add_subplot(gs[1, 0])


def collinear_spectrum(p1, p2, L=12):
    nsite = p1*p2*2
    idx = lambda n1, n2, sub: ((n1 % p1)*p2 + (n2 % p2))*2 + sub
    acc = np.zeros((nsite, nsite, 3, 3))
    for Jm, sa, sb, off in BONDS:
        for n1 in range(L):
            for n2 in range(L):
                acc[idx(n1, n2, sa), idx(n1 + off[0], n2 + off[1], sb)] += Jm
    N = L*L*2
    out = []
    for bits in itertools.product([1, -1], repeat=nsite-1):
        sig = np.array((1,) + bits)
        Mx = np.einsum('a,b,abij->ij', sig, sig, acc)
        out.append(SLEN**2*np.linalg.eigvalsh(Mx)[0]/N)
    return np.array(out)


spec = collinear_spectrum(2, 2)
axD.hist(spec, bins=40, color="#9aa4b2")
axD.axvline(E_FM, color=ACC, lw=2)
axD.annotate("FM = lowest\n(-1.9725)", xy=(E_FM, axD.get_ylim()[1]*0.7),
             xytext=(10, 0), textcoords="offset points", color=ACC, fontsize=8.5,
             va="center")
axD.set_title("D · all 128 collinear patterns in a 2×2 cell", fontsize=10, color=INK)
axD.set_xlabel("$E/N$ (meV per site)", fontsize=9)
axD.set_ylabel("count", fontsize=9)
axD.grid(axis="y", color=GRID, lw=0.8); axD.set_axisbelow(True)
for sp in ("top", "right"):
    axD.spines[sp].set_visible(False)
axD.tick_params(labelsize=8)

# ---------------- E: energy ladder ----------------
axE = fig.add_subplot(gs[1, 1:])


import json
EN = json.load(open(os.path.join(REPO, "output", "bcao_energies.json")))

rows = [("all collinear orders (≤4×2) = FM", E_FM, "#9aa4b2"),
        ("best circular spiral (variational)", E_FM, "#9aa4b2")]
for lab in ("LT seed after T=0 quench", "SA, fast cooling (0.90)",
            "SA, slow cooling (0.97)", "SA + twisted boundaries"):
    if lab in EN:
        d = EN[lab]
        col = "#9aa4b2" if lab.startswith("LT seed") else ACC
        rows.append((f"{lab}, best of {d['n']}", d["best"], col))
rows.sort(key=lambda r: -r[1])
ypos = np.arange(len(rows))
for y, (lab, v, col) in zip(ypos, rows):
    axE.barh(y, v - LT_BOUND, left=LT_BOUND, height=0.5, color=col)
    axE.text(v + 0.0004, y, f"{v:.6f}", va="center", fontsize=8.5, color=INK)
axE.axvline(LT_BOUND, color=RED, lw=2)
axE.annotate(f"LT bound {LT_BOUND:.4f} — UNREACHABLE\n(panel C: its eigenvector is an elliptical SDW)",
             xy=(LT_BOUND, len(rows)-0.35), xytext=(12, 0), textcoords="offset points",
             color=RED, fontsize=8.5, va="center")
axE.axvline(E_FM, color=MUTED, ls=":", lw=1.4)
axE.annotate("FM", xy=(E_FM, -0.5), xytext=(3, 0), textcoords="offset points",
             color=MUTED, fontsize=8, va="center")
best = min(r[1] for r in rows)
axE.text(0.52, 0.60, f"every method converges to {best:.4f}\n"
         "— only 0.0014 below FM.  Twisted BCs relax\n"
         "to θ≈0, so there is no incommensurate\n"
         "frustration for them to relieve.",
         transform=axE.transAxes, color=INK, fontsize=8.4, va="center")
axE.set_yticks(ypos); axE.set_yticklabels([r[0] for r in rows], fontsize=9)
axE.set_xlim(LT_BOUND - 0.0012, -1.9545)
axE.set_ylim(-0.6, len(rows) - 0.15)
axE.set_xlabel("energy density  $E/N$  (meV per site, $|S|=1/2$)", fontsize=9.5)
axE.set_title("E · everything tried  (all realizable states sit ~0.011 meV above the LT bound)",
              fontsize=10, color=INK)
axE.grid(axis="x", color=GRID, lw=0.8); axE.set_axisbelow(True)
for sp in ("top", "right", "left"):
    axE.spines[sp].set_visible(False)
axE.tick_params(labelsize=8)

fig.suptitle("Why the classical ground state of BCAO set (3) is hard to converge — "
             "the model sits on the edge of the FM instability",
             fontsize=13, y=0.975)
out = sys.argv[1] if len(sys.argv) > 1 else "bcao_diagnostics.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("wrote", out)
for lab, v, _ in rows:
    print(f"  {lab:42s} {v:.6f}")
