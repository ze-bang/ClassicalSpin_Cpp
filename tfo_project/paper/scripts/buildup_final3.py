import numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCALE=2*np.pi/4.135667696; PS=0.6582119569
BASE="tfo_project/tmfeo3_2dcs_final/EXPERIMENTAL_FINAL"
def ref(run,ds,l):
    with h5py.File(f"{BASE}/{run}/sample_0/pump_probe_spectroscopy.h5") as f:
        return f['/reference/times'][:], f[f'/reference/{ds}'][:,l]
def gabor(t,M,f_thz,centers,sig=8.0):
    w=2*np.pi*f_thz/SCALE; M0=M-M.mean(); out=[]
    for tc in centers:
        g=np.exp(-0.5*((t-tc)/sig)**2)
        out.append(abs(np.sum(M0*g*np.exp(-1j*w*t)))*(t[1]-t[0]))
    return np.array(out)
centers=np.arange(-14,120,1.5)
configs=[("bb2_full","full: $\\kappa^B{+}W^{yy}_1{+}W^{xz}_1$","C0",2.2),
         ("bb2_wxz","$W^{xz}_1$ only","C2",1.6),
         ("bb2_kB","$\\kappa^B$ only","C3",1.6),
         ("bb2_none","no Fe--Tm vertex","0.45",1.3)]
fig,axs=plt.subplots(1,2,figsize=(12.8,4.9))
ax=axs[0]
for run,lab,c,lw in configs:
    t,M=ref(run,"M_global_SU2",0)
    A=gabor(t,M,0.90,centers)
    ax.plot(centers*PS,1e4*A,color=c,lw=lw,label=lab)
ax.axvspan(-11*PS,11*PS,color="gold",alpha=0.2,lw=0)
ax.text(0,7.9,"pulse",ha="center",fontsize=8,color="0.45")
ax.annotate("peaks AT pulse end;\nno post-pulse growth",xy=(9,7.2),xytext=(28,7.0),
            fontsize=8,arrowprops=dict(arrowstyle="->",lw=0.8))
ax.annotate("$\\tau_{env}\\approx8$ ps $=T_2(E_{12})$:\nconverted signal mirrors\nthe reservoir lifetime",
            xy=(16,2.2),xytext=(38,3.6),fontsize=8,arrowprops=dict(arrowstyle="->",lw=0.8))
ax.set_xlabel("t (ps)"); ax.set_ylabel("$|A|(0.90\\,{\\rm THz})$ of $S_x$  ($\\times10^{-4}$)")
ax.set_title("Geometry B, single measured pulse: $q_{\\rm AFM}$ amplitude (Gabor, $\\sigma$=5 ps)",fontsize=10)
ax.legend(fontsize=8); ax.set_xlim(-12,80)
ax=axs[1]
t,_=ref("bb2_full","M_global_SU2",0)
for run,ds,l,f0,lab,c in [("bb2_full","M_global_SU2",0,0.90,"$q_{\\rm AFM}$ (0.90)","C0"),
                          ("bb2_full","M_global_SU3",1,0.50,"$E_{12}$ reservoir (0.50), $T_2{=}8$ ps","C4"),
                          ("bb2_full","M_global_SU2",1,0.38,"$q_{\\rm FM}$ reservoir (0.38)","C1")]:
    t,M=ref(run,ds,l)
    A=gabor(t,M,f0,centers)
    ax.semilogy(centers*PS,A/A.max(),color=c,lw=1.8,label=lab)
ax.axvspan(-11*PS,11*PS,color="gold",alpha=0.2,lw=0)
ax.set_ylim(1e-4,1.4)
ax.set_xlabel("t (ps)"); ax.set_ylabel("normalized amplitude (log)")
ax.set_title("All coherent reservoirs die within $T_2$: nothing left to pump slowly",fontsize=10)
ax.legend(fontsize=8,loc="upper right"); ax.set_xlim(-12,80)
fig.suptitle("No coherence-fed post-pulse buildup with measured linewidths (runs bb2_*, consolidated Hamiltonian)",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/postdrive_buildup_final.png",dpi=115)
print("saved paper/figs/postdrive_buildup_final.png")
