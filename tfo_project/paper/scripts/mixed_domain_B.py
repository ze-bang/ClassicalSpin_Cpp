import numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCALE=2*np.pi/4.135667696; PS=0.6582119569
BASE="tfo_project/tmfeo3_2dcs_final"
d=f"{BASE}/EXPERIMENTAL_FINAL/vB2_gs1/sample_0"
with h5py.File(f"{d}/pump_probe_spectroscopy.h5") as f:
    t=f['/reference/times'][:]; tau=f['/tau_scan/tau_values'][:]
    M0=f['/reference/M_global_SU2'][:,0]
    M=np.zeros((len(tau),len(t)))
    for i in range(len(tau)):
        g=f[f'/tau_scan/tau_{i}']
        M[i]=g['M01_global_SU2'][:,0]-g['M1_global_SU2'][:,0]-M0
# FFT along tau ONLY; keep t as real time
w=np.ones_like(tau); m1=tau>-6; w[m1]=0.5*(1-np.cos(np.pi*(-tau[m1])/6))
m2=tau<-110; w[m2]=0.5*(1-np.cos(np.pi*(120+tau[m2])/10))
Md=(M-M.mean())*w[:,None]
S=np.abs(np.fft.fftshift(np.fft.fft(Md,axis=0),axes=0))   # (omega_tau, t)
wta=np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
wT=-wta
# smooth along t with a 2 ps window to remove carrier oscillation of the emission
from scipy.ndimage import uniform_filter1d
Senv=uniform_filter1d(S,size=int(3.0/(t[1]-t[0])),axis=1)
my=(np.abs(wT)<1.5)
fig,axs=plt.subplots(1,2,figsize=(12.8,4.9),gridspec_kw={"width_ratios":[1.35,1]})
ax=axs[0]
sel_t=(t>=-15)&(t<=120)
Z=Senv[np.ix_(my,sel_t)]
ax.pcolormesh(t[sel_t]*PS,wT[my],Z/Z.max(),shading="auto",cmap="inferno",vmin=0,vmax=1,rasterized=True)
for nm,v in [("qFM",0.38),("E12",0.50),("qAFM",0.90),("E23",0.70),("E13",1.20)]:
    ax.axhline(v,color="w",ls="--",lw=0.7,alpha=0.3)
    ax.axhline(-v,color="w",ls="--",lw=0.7,alpha=0.3)
    ax.text(80,v+0.02,nm,color="w",alpha=0.6,fontsize=6)
ax.axvspan(-11*PS,11*PS,color="w",alpha=0.12,lw=0)
ax.set_xlabel("real time t (ps)"); ax.set_ylabel("physical $\\omega_T$ (THz)")
ax.set_title("Geometry B: $|{\\rm FFT}_\\tau\\,M_{\\rm NL}|(\\omega_\\tau, t)$ — delay rows vs real time",fontsize=10)
ax=axs[1]
for v,lab,c in [(0.38,"$\\omega_\\tau=q_{\\rm FM}$ row (magnon--magnon peak)","C1"),
                (0.50,"$\\omega_\\tau=E_{12}$ row ($\\kappa^B$ transfer peak)","C4")]:
    iy=np.argmin(np.abs(wT-v))
    row=Senv[iy]; sel=(t>=-15)&(t<=120)
    ax.plot(t[sel]*PS,row[sel]/row[sel].max(),color=c,lw=1.9,label=lab)
ax.axvspan(-11*PS,11*PS,color="gold",alpha=0.2,lw=0)
ax.text(0,1.02,"pulse",ha="center",fontsize=8,color="0.45")
ax.set_xlabel("real time t (ps)"); ax.set_ylabel("row amplitude / max")
ax.set_title("Row cuts: model predicts decay after the pulse\n(a population-fed buildup would rise here on $T_1$)",fontsize=9.5)
ax.legend(fontsize=8)
fig.suptitle("Mixed-domain view of geometry B (run vB2, consolidated Hamiltonian): the buildup discriminator",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/mixed_domain_B.png",dpi=115)
print("saved paper/figs/mixed_domain_B.png")
for v in [0.38,0.50]:
    iy=np.argmin(np.abs(wT-v))
    row=Senv[iy]
    for tp in [10,20,40,60,90,120]:
        print(f"row {v}: t={tp*PS:5.1f}ps amp={row[np.argmin(np.abs(t-tp))]:.3e}")
    print()
