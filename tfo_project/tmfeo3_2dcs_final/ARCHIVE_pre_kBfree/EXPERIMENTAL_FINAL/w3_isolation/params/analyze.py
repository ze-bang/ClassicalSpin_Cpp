import numpy as np, h5py, json, os
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
from scipy.signal import hilbert
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SCALE = 2*np.pi/4.135667696
ROOT  = "/home/pc_linux/ClassicalSpin_Cpp/tfo_project/tmfeo3_2dcs_final/EXPERIMENTAL_FINAL"
ISO   = f"{ROOT}/w3_isolation"
OUT   = os.path.dirname(os.path.abspath(__file__))

def load(path, sp="SU2", l=0, tau_stride=1):
    with h5py.File(f"{path}/sample_0/pump_probe_spectroscopy.h5") as f:
        t   = f['/reference/times'][:]
        tau = f['/tau_scan/tau_values'][:]
        M0  = f[f'/reference/M_global_{sp}'][:, l]
        idx = np.arange(len(tau))[::tau_stride]
        M   = np.zeros((len(idx), len(t)))
        for k, i in enumerate(idx):
            g = f[f'/tau_scan/tau_{i}']
            M[k] = g[f'M01_global_{sp}'][:, l] - g[f'M1_global_{sp}'][:, l] - M0
    return t, tau[idx], M

def spec(t, tau, M):
    tm = t >= 3.0
    at   = np.hanning(2*tm.sum())[tm.sum():]
    atau = np.hanning(2*len(tau))[:len(tau)]
    Md = (M[:, tm] - M[:, tm].mean(axis=0, keepdims=True)) * atau[:, None] * at[None, :]
    wt  = np.fft.fftshift(np.fft.fftfreq(tm.sum(), t[1]-t[0])) * SCALE
    wta = np.fft.fftshift(np.fft.fftfreq(len(tau), tau[1]-tau[0])) * SCALE
    mx = (wt > 0.12) & (wt < 1.6); my = (np.abs(wta) < 1.6)
    F  = np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my, mx)]
    A  = gaussian_filter(F, sigma=(2, 1.5))
    return wt[mx], -wta[my], A, F   # A smoothed (census), F raw (position)

def blind(wtb, wTb, A, amp_min=0.05):
    B = A.copy(); B[np.abs(wTb) < 0.18, :] = 0
    dyx = abs(wTb[1]-wTb[0]); dxx = abs(wtb[1]-wtb[0])
    n = B.max(); fp = (int(0.10/dyx)|1, int(0.10/dxx)|1)
    ismax = (B == maximum_filter(B, size=fp))
    bg = median_filter(B, size=(int(0.34/dyx)|1, int(0.34/dxx)|1))
    ys, xs = np.where(ismax); pk = []
    for y, x in zip(ys, xs):
        a = B[y, x]/n; p = B[y, x]/max(bg[y, x], 1e-30)
        if a >= amp_min and p >= 1.5:
            pk.append((round(float(a),3), round(float(p),1),
                       round(float(wTb[y]),2), round(float(wtb[x]),2)))
    pk.sort(reverse=True); out = []
    for q in pk:
        if all(abs(q[2]-r[2]) > 0.09 or abs(q[3]-r[3]) > 0.09 for r in out):
            out.append(q)
    return out[:10]

def at(wtb, wTb, A, wT, wt):
    return A[np.argmin(np.abs(wTb-wT))][np.argmin(np.abs(wtb-wt))]

def refine_wt(wtb, wTb, F, wT0, wt0, half=0.12):
    """peak omega_t position on the RAW spectrum near (wT0, wt0)"""
    iy = np.argmin(np.abs(wTb - wT0))
    sel = (wtb > wt0-half) & (wtb < wt0+half)
    row = F[iy, :]; xs = np.where(sel)[0]
    j = xs[np.argmax(row[xs])]
    # parabolic sub-bin refinement
    if 0 < j < len(wtb)-1:
        y0,y1,y2 = row[j-1],row[j],row[j+1]
        d = (y0-y2)/(2*(y0-2*y1+y2)) if (y0-2*y1+y2)!=0 else 0.0
        return wtb[j] + d*(wtb[1]-wtb[0])
    return wtb[j]

RUNS = [  # (label, path, tau_stride)
    ("FULL master (κB+W3+thermal)", f"{ROOT}/fin_B_gs1",   2),
    ("no κB  (W3+thermal only)",    f"{ISO}/iso_noKB",     1),
    ("W3 coherent only (no thermal)",f"{ISO}/iso_W3coh",   1),
    ("κB only (no W3)",             f"{ISO}/iso_kB",       1),
    ("none (all conversion off)",   f"{ISO}/iso_none",     1),
    ("W3_xz (κB=0)",                f"{ISO}/iso_W3xz",     1),
    ("W3_xy (κB=0)",                f"{ISO}/iso_W3xy",     1),
    ("W3_zz (κB=0)",                f"{ISO}/iso_W3zz",     1),
]

results = {}
maps = {}
for label, path, stride in RUNS:
    if not os.path.exists(f"{path}/sample_0/pump_probe_spectroscopy.h5"):
        print(f"[skip] {label}: no h5"); continue
    t, tau, M = load(path, tau_stride=stride)
    wtb, wTb, A, F = spec(t, tau, M)
    pk = blind(wtb, wTb, A)
    anchor = at(wtb, wTb, A, 0.38, 0.90)          # (qFM,qAFM)
    target = at(wtb, wTb, A, 0.52, 0.90)          # (E12,qAFM)
    pos = refine_wt(wtb, wTb, F, 0.52, 0.90)
    results[label] = dict(census=pk,
                          ratio=float(target/anchor) if anchor>0 else np.nan,
                          target_abs=float(target), anchor_abs=float(anchor),
                          wt_position=float(pos))
    maps[label] = (wtb, wTb, A)
    print(f"\n=== {label}")
    print(f"  (E12,qAFM)/(qFM,qAFM) = {target/anchor:.3f}   [abs {target:.3g}/{anchor:.3g}]")
    print(f"  omega_t position of E12-row peak: {pos:.4f} THz")
    for a,p,yy,xx in pk[:6]: print(f"   {a:5.2f} x{p:5.1f}  ({yy:+.2f}, {xx:.2f})")

# ---------------- buildup curves ----------------
def buildup(path):
    t, tau, M = load(path)
    m = M[0]                       # tau = -0.2 row
    tm = t > 3.0
    tt = t[tm]; y = m[tm]
    # bandpass 0.75-1.05 THz around qAFM
    Y = np.fft.rfft(y); fr = np.fft.rfftfreq(len(y), t[1]-t[0])*SCALE
    Y[(fr < 0.75) | (fr > 1.05)] = 0
    yb = np.fft.irfft(Y, n=len(y))
    env = np.abs(hilbert(yb))
    # smooth envelope
    k = int(8/(t[1]-t[0])); k += (k+1)%2
    envs = np.convolve(env, np.ones(k)/k, mode="same")
    return tt, envs

BURUNS = [("full (κB+W3+thermal)", f"{ISO}/bu_full",  "C0"),
          ("no κB (W3+thermal)",   f"{ISO}/bu_noKB",  "C2"),
          ("κB only",              f"{ISO}/bu_kB",    "C1"),
          ("none",                 f"{ISO}/bu_none",  "C7")]
bu = {}
for label, path, c in BURUNS:
    if os.path.exists(f"{path}/sample_0/pump_probe_spectroscopy.h5"):
        bu[label] = buildup(path) + (c,)

json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
          open(f"{OUT}/census.json", "w"), indent=1)

# ---------------- verdict figure ----------------
fig = plt.figure(figsize=(16, 12.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.85], hspace=0.42, wspace=0.32)
order = [r[0] for r in RUNS]
for i, label in enumerate(order):
    if label not in maps: continue
    ax = fig.add_subplot(gs[i//4, i%4])
    wtb, wTb, A = maps[label]
    Z = A.copy(); Z[np.abs(wTb) < 0.18, :] = 0
    anchor = results[label]["anchor_abs"]
    ax.pcolormesh(wtb, wTb, Z/max(anchor,1e-30), shading="auto", cmap="inferno",
                  vmin=0, vmax=1.3, rasterized=True)
    for nm, v in [("qFM",0.38),("E12",0.50),("E23",0.70),("qAFM",0.90),("E13",1.20)]:
        ax.axvline(v, color="w", ls="--", lw=0.6, alpha=0.3)
        ax.axhline(v, color="w", ls="--", lw=0.6, alpha=0.3)
        ax.axhline(-v, color="w", ls="--", lw=0.6, alpha=0.3)
    r = results[label]
    ax.plot(0.90, 0.52, "o", mfc="none", mec="cyan", ms=16, mew=1.6)
    ax.plot(0.90, 0.38, "s", mfc="none", mec="lime", ms=14, mew=1.2)
    ax.set_xlim(0.15, 1.55); ax.set_ylim(-1.4, 1.4)
    ax.set_title(f"{label}\n(E12,qAFM)/(qFM,qAFM) = {r['ratio']:.2f}", fontsize=9)
    ax.set_xlabel("$\\omega_t$ (THz)", fontsize=8); ax.set_ylabel("$\\omega_\\tau$ (THz)", fontsize=8)
    ax.tick_params(labelsize=7)

# buildup panel
axb = fig.add_subplot(gs[2, 0:2])
for label, (tt, env, c) in bu.items():
    axb.semilogy(tt, env/1e-6, c, label=label, lw=1.4)
axb.set_xlabel("detection time t (code units; 1 unit ≈ 0.152 ps)")
axb.set_ylabel("|M$_{NL}$ S$_x$| envelope @ qAFM band (×10$^{-6}$)")
axb.set_title("Post-drive buildup: qAFM-band envelope at $\\tau=-0.2$", fontsize=10)
axb.legend(fontsize=8); axb.grid(alpha=0.3)

# ratio bar chart
axr = fig.add_subplot(gs[2, 2:4])
labs = [l for l in order if l in results]
vals = [results[l]["ratio"] for l in labs]
short = ["FULL", "no κB\n(W3+th)", "W3 coh", "κB only", "none", "W3_xz", "W3_xy", "W3_zz"][:len(labs)]
cols = ["C0", "C2", "C2", "C1", "C7", "C4", "C4", "C4"][:len(labs)]
axr.bar(range(len(vals)), vals, color=cols)
axr.axhline(0.46, color="k", ls=":", lw=1.2)
axr.text(len(vals)-0.4, 0.47, "measured 0.45–0.46", ha="right", fontsize=8)
axr.set_xticks(range(len(vals))); axr.set_xticklabels(short, fontsize=8)
axr.set_ylabel("(E12,qAFM) / (qFM,qAFM)")
axr.set_title("The isolation matrix: who makes the transfer peak?", fontsize=10)
fig.suptitle("Geometry B cross-polarised ($S_x$ readout): is the (E$_{12}$, q$_{AFM}$) transfer peak field-gated (κ$^B$) or population-mediated (W$_3$)?",
             fontsize=12, y=0.995)
fig.savefig(f"{OUT}/w3_verdict.png", dpi=160, bbox_inches="tight")
print(f"\nsaved {OUT}/w3_verdict.png")
