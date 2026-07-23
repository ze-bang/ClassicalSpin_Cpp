import numpy as np, h5py
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCALE=2*np.pi/4.135667696
BASE="tfo_project/tmfeo3_2dcs_final"
def load(run):
    d=f"{BASE}/{run}/sample_0"
    with h5py.File(f"{d}/pump_probe_spectroscopy.h5") as f:
        t=f['/reference/times'][:]; tau=f['/tau_scan/tau_values'][:]
        M0=f['/reference/M_global_SU3'][:,1]
        M=np.zeros((len(tau),len(t)))
        for i in range(len(tau)):
            g=f[f'/tau_scan/tau_{i}']
            M[i]=g['M01_global_SU3'][:,1]-g['M1_global_SU3'][:,1]-M0
    return t,tau,M
def spec(t,tau,M):
    tm=t>=3.0
    at=np.hanning(2*tm.sum())[tm.sum():]; atau=np.hanning(2*len(tau))[:len(tau)]
    Md=(M[:,tm]-M[:,tm].mean())*atau[:,None]*at[None,:]
    wt=np.fft.fftshift(np.fft.fftfreq(tm.sum(),t[1]-t[0]))*SCALE
    wta=np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
    mx=(wt>0.12)&(wt<1.6); my=(np.abs(wta)<1.6)
    A=gaussian_filter(np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my,mx)],sigma=(2,1.5))
    return wt[mx],-wta[my],A
runs=[("EXPERIMENTAL_FINAL/vA_gs1","baseline ($W^{xz}_1{=}0.01$)"),
      ("EXPERIMENTAL_FINAL/vA_noWxz","$W^{xz}_1=0$"),
      ("EXPERIMENTAL_FINAL/vA_su3x2","Tm drive $\\times2$")]
fig,axs=plt.subplots(1,4,figsize=(16.5,4.4),gridspec_kw={"width_ratios":[1,1,1,1.15]})
mainref=None; rows={}
for ax,(run,lab) in zip(axs[:3],runs):
    t,tau,M=load(run)
    wtb,wTb,A=spec(t,tau,M)
    main=A[np.argmin(np.abs(wTb-0.90))][np.argmin(np.abs(wtb-0.50))]
    sy=(wTb>0.2)&(wTb<0.8); sx=(wtb>0.6)&(wtb<1.4)
    Z=A[np.ix_(sy,sx)]/main
    ax.pcolormesh(wtb[sx],wTb[sy],Z,shading="auto",cmap="inferno",vmin=0,vmax=0.13,rasterized=True)
    for v in [0.70,0.90,1.00,1.20]:
        ax.axvline(v,color="w",ls="--",lw=0.7,alpha=0.4)
    ax.axhline(0.50,color="w",ls="--",lw=0.7,alpha=0.4)
    ax.set_xlabel("omega_t (THz)"); ax.set_ylabel("physical omega_T (THz)")
    ax.set_title(lab+f"\n(0.49,0.90)={A[np.argmin(np.abs(wTb-0.49))][np.argmin(np.abs(wtb-0.90))]/main:.3f} of main",fontsize=9)
    iy=np.argmin(np.abs(wTb-0.49))
    rows[lab]=(wtb,A[iy]/main)
ax=axs[3]
for (lab,(wtb,row)),c in zip(rows.items(),["C0","C3","C2"]):
    sel=(wtb>0.6)&(wtb<1.4)
    ax.plot(wtb[sel],row[sel],color=c,lw=1.9,label=lab)
for v,nm in [(0.70,"E23"),(0.90,"qAFM"),(1.00,"2E12"),(1.20,"E13")]:
    ax.axvline(v,color="0.75",ls="--",lw=0.8)
    ax.text(v,ax.get_ylim()[1]*0.0+0.128,nm,ha="center",fontsize=7,color="0.4")
ax.set_xlabel("omega_t (THz)"); ax.set_ylabel("amplitude / main peak")
ax.set_title("cut along $\\omega_T=+E_{12}$: the reverse-transfer row",fontsize=9.5)
ax.legend(fontsize=8); ax.set_ylim(0,0.135)
fig.suptitle("Geometry A cross channel: the reverse-W transfer peak $(E_{12},q_{\\rm AFM})$ near the reported $(0.5,\\,1.0)$ "
             "— dies with $W^{xz}_1$, linear in Tm drive",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/reverseW_peak.png",dpi=115)
print("saved paper/figs/reverseW_peak.png")
