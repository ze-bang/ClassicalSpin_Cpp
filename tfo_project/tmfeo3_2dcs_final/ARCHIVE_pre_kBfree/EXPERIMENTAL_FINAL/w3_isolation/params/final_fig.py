import numpy as np, h5py, os
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SCALE = 2*np.pi/4.135667696; PS = 0.6582119569
ROOT = "/home/pc_linux/ClassicalSpin_Cpp/tfo_project/tmfeo3_2dcs_final/EXPERIMENTAL_FINAL"
ISO  = f"{ROOT}/w3_isolation"
OUT  = os.path.dirname(os.path.abspath(__file__))
BLUE, RED, GREEN, PURPLE, GRAY = "#31549E", "#C6303E", "#1E7239", "#8A5A9E", "#8a8a86"

def load(path, stride=1):
    with h5py.File(f"{path}/sample_0/pump_probe_spectroscopy.h5") as f:
        t = f['/reference/times'][:]; tau = f['/tau_scan/tau_values'][:]
        M0 = f['/reference/M_global_SU2'][:, 0]
        idx = np.arange(len(tau))[::stride]
        M = np.zeros((len(idx), len(t)))
        for k, i in enumerate(idx):
            g = f[f'/tau_scan/tau_{i}']
            M[k] = g['M01_global_SU2'][:, 0] - g['M1_global_SU2'][:, 0] - M0
    return t, tau[idx], M

def spec(t, tau, M):
    tm = t >= 3.0
    at = np.hanning(2*tm.sum())[tm.sum():]; atau = np.hanning(2*len(tau))[:len(tau)]
    Md = (M[:, tm] - M[:, tm].mean(axis=0, keepdims=True)) * atau[:, None] * at[None, :]
    wt  = np.fft.fftshift(np.fft.fftfreq(tm.sum(), t[1]-t[0])) * SCALE
    wta = np.fft.fftshift(np.fft.fftfreq(len(tau), tau[1]-tau[0])) * SCALE
    mx = (wt > 0.12) & (wt < 1.6); my = (np.abs(wta) < 1.6)
    A = gaussian_filter(np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my, mx)], sigma=(2, 1.5))
    return wt[mx], -wta[my], A

def anchor_of(wtb, wTb, A, wT, wt):
    return A[np.argmin(np.abs(wTb-wT))][np.argmin(np.abs(wtb-wt))]

MAPS = [  # (panel title, path, stride, ratio_label)
    ("FULL master\n($\\kappa^B$ + W$_3$ + thermal)", f"{ROOT}/fin_B_gs1", 2),
    ("$\\kappa^B$ only  (field-gated)", f"{ISO}/iso_kB", 1),
    ("no $\\kappa^B$  (W$_3^{yz}$ + thermal)", f"{ISO}/iso_noKB", 1),
    ("none  (no $\\kappa^B$, no W$_3$)", f"{ISO}/iso_none", 1),
    ("W$_3^{yz}$ coherent grating only", f"{ISO}/iso_W3coh", 1),
    ("W$_3^{xz}$ (reorientation comp.)", f"{ISO}/iso_W3xz", 1),
    ("W$_3^{zz}$ (diagonal)", f"{ISO}/iso_W3zz", 1),
    ("W$_1^{yz}$ = 0.05 (static, no gate)", f"{ISO}/iso_W1yz005", 1),
]

cache = {}
for title, path, st in MAPS:
    t, tau, M = load(path, st)
    cache[title] = spec(t, tau, M)
    print("spectrum done:", title)

fig = plt.figure(figsize=(17, 13.2))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9], hspace=0.44, wspace=0.30)

for i, (title, path, st) in enumerate(MAPS):
    ax = fig.add_subplot(gs[i//4, i%4])
    wtb, wTb, A = cache[title]
    anch = anchor_of(wtb, wTb, A, 0.38, 0.90)
    targ = anchor_of(wtb, wTb, A, 0.52, 0.90)
    Z = A.copy(); Z[np.abs(wTb) < 0.18, :] = 0
    ax.pcolormesh(wtb, wTb, Z/anch, shading="auto", cmap="inferno", vmin=0, vmax=1.25, rasterized=True)
    for nm, v in [("qFM",0.38),("E12",0.50),("E23",0.70),("qAFM",0.90),("E13",1.20)]:
        ax.axvline(v, color="w", ls="--", lw=0.6, alpha=0.28)
        ax.axhline(v, color="w", ls="--", lw=0.6, alpha=0.28)
        ax.axhline(-v, color="w", ls="--", lw=0.6, alpha=0.28)
    ax.plot(0.90, 0.52, "o", mfc="none", mec="cyan", ms=17, mew=1.7)
    ax.annotate("(E$_{12}$,q$_{AFM}$)", (0.90, 0.52), textcoords="offset points",
                xytext=(9, 8), color="cyan", fontsize=7.5)
    ax.set_xlim(0.15, 1.55); ax.set_ylim(-1.4, 1.4)
    ax.set_title(f"{title}\ntarget/(qFM,qAFM) anchor = {targ/anch:.2f}", fontsize=9.3)
    if i % 4 == 0: ax.set_ylabel("$\\omega_\\tau$ (THz)", fontsize=9)
    ax.set_xlabel("$\\omega_t$ (THz)", fontsize=9)
    ax.tick_params(labelsize=7.5)

# ---------- row 3a: omega_tau cut at omega_t = 0.90 ----------
axc = fig.add_subplot(gs[2, 0:2])
cuts = [("$\\kappa^B$ only", "$\\kappa^B$ only  (field-gated)", BLUE, "-"),
        ("W$_1^{yz}$=0.05 static", "W$_1^{yz}$ = 0.05 (static, no gate)", RED, "-"),
        ("W$_3^{yz}$+thermal", "no $\\kappa^B$  (W$_3^{yz}$ + thermal)", GREEN, "-"),
        ("none", "none  (no $\\kappa^B$, no W$_3$)", GRAY, "--")]
for lab, key, c, ls in cuts:
    wtb, wTb, A = cache[key]
    j = np.argmin(np.abs(wtb - 0.90))
    sel = (wTb > 0.16) & (wTb < 0.80)
    y = A[sel, j]; ref = y.max()
    axc.plot(wTb[sel], y/ref, color=c, ls=ls, lw=1.9)
xlbl = {"$\\kappa^B$ only": (0.585, 0.50), "W$_1^{yz}$=0.05 static": (0.60, 0.30),
        "W$_3^{yz}$+thermal": (0.545, 0.115), "none": (0.235, 0.55)}
for lab, key, c, ls in cuts:
    axc.text(*xlbl[lab], lab, color=c, fontsize=8.6,
             fontweight="bold" if c != GRAY else "normal")
axc.axvline(0.52, color="k", ls=":", lw=1.0, alpha=0.6)
axc.text(0.52, 1.02, "E$_{12}$", ha="center", fontsize=8)
axc.axvline(0.38, color="k", ls=":", lw=1.0, alpha=0.6)
axc.text(0.38, 1.02, "q$_{FM}$", ha="center", fontsize=8)
axc.set_xlabel("$\\omega_\\tau$ (THz)", fontsize=9.5)
axc.set_ylabel("amplitude at $\\omega_t$=0.90 (norm.)", fontsize=9.5)
axc.set_title("Delay-axis cut through the magnon emission line: only the gate makes a resolved second row;\n"
              "the static W$_1^{yz}$ raises a merged shoulder (mixer), W$_3$ adds nothing", fontsize=9.5)
axc.grid(alpha=0.25); axc.set_ylim(0, 1.08)

# ---------- row 3b: late-time M_NL @ qAFM ----------
axb = fig.add_subplot(gs[2, 2])
def mnl_gabor(path):
    with h5py.File(f"{path}/sample_0/pump_probe_spectroscopy.h5") as f:
        t = f['/reference/times'][:]
        g = f['/tau_scan/tau_0']
        m = g['M01_global_SU2'][:,0]-g['M1_global_SU2'][:,0]-f['/reference/M_global_SU2'][:,0]
    w = 2*np.pi*0.90/SCALE; m = m - m.mean()
    cs = np.arange(12, 392, 6.0); out = []
    for tc in cs:
        gw = np.exp(-0.5*((t-tc)/10.0)**2)
        out.append(abs(np.sum(m*gw*np.exp(-1j*w*t)))*(t[1]-t[0]))
    return cs*PS, np.array(out)
for lab, run, c, ls in [("full", "bu_full", BLUE, "-"),
                        ("$\\kappa^B$ only", "bu_kB", BLUE, ":"),
                        ("W$_3$+thermal (no $\\kappa^B$)", "bu_noKB", GREEN, "-"),
                        ("none", "bu_none", GRAY, "--")]:
    tt, A = mnl_gabor(f"{ISO}/{run}")
    axb.semilogy(tt, np.maximum(A, 1e-12), color=c, ls=ls, lw=1.7)
axb.text(9, 2.5e-4, "impulsive peak\n(all runs)", fontsize=7.8, color="0.25")
axb.annotate("thermal W$_3$ tail:\nreal but 4 decades down", xy=(56, 3.6e-8), xytext=(21, 3e-9),
             fontsize=7.8, color=GREEN, arrowprops=dict(arrowstyle="->", lw=0.8, color=GREEN))
axb.text(40, 2.4e-10, "none / $\\kappa^B$: floor", fontsize=7.5, color=GRAY)
axb.set_xlabel("detection time t (ps)", fontsize=9.5)
axb.set_ylabel("|M$_{NL}$| @ 0.90 THz (Gabor)", fontsize=9.5)
axb.set_title("Post-pulse timing ($\\tau$=$-$0.2)", fontsize=9.5)
axb.set_ylim(1e-11, 1e-3); axb.grid(alpha=0.25, which="both")

# ---------- row 3c: verdict bars ----------
axr = fig.add_subplot(gs[2, 3])
bars = [("FULL", 0.448, BLUE), ("$\\kappa^B$\nonly", 0.449, BLUE),
        ("no $\\kappa^B$\nW$_3$+th", 0.239, GREEN), ("W$_3^{yz}$\ncoh", 0.253, GREEN),
        ("W$_3^{xz}$", 0.234, GREEN), ("W$_1^{yz}$\n0.05", 0.529, RED),
        ("none", 0.253, GRAY)]
xs = range(len(bars))
axr.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
axr.axhspan(0.43, 0.46, color="k", alpha=0.12, lw=0)
axr.text(6.35, 0.445, "measured\n0.43–0.46", ha="right", va="center", fontsize=7.4)
axr.axhline(0.253, color=GRAY, ls="--", lw=1.0)
axr.text(0.0, 0.262, "background shoulder (no peak)", fontsize=7.0, color="0.35")
for x, (lab, v, c) in zip(xs, bars):
    star = {"$\\kappa^B$\nonly": "peak ✓", "FULL": "peak ✓", "W$_1^{yz}$\n0.05": "no peak\n(mixer)"}.get(lab, "")
    if star: axr.text(x, v+0.012, star, ha="center", fontsize=6.8)
axr.set_xticks(list(xs)); axr.set_xticklabels([b[0] for b in bars], fontsize=7.6)
axr.set_ylabel("(E$_{12}$,q$_{AFM}$) / (q$_{FM}$,q$_{AFM}$)", fontsize=9)
axr.set_title("Verdict: value at target point\n('peak ✓' = resolved blind-census maximum)", fontsize=9)
axr.set_ylim(0, 0.62)

fig.suptitle("Settling the (E$_{12}$, q$_{AFM}$) transfer peak, geometry B cross-polarised ($S_x$ readout):\n"
             "every field-free configuration tested (W$_3$ population channel in all components, coherent and thermal; static W$_1$ mixer) fails to create the resolved peak — the $B_z(t)$-gated $\\kappa^B$ vertex is required",
             fontsize=12.5, y=0.998)
fig.savefig(f"{OUT}/w3_settled.png", dpi=150, bbox_inches="tight")
print("saved", f"{OUT}/w3_settled.png")
