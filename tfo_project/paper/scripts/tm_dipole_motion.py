import numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCALE=2*np.pi/4.135667696; PS=0.6582119569
BASE="tfo_project/tmfeo3_2dcs_final"
def ref(run,ds,l):
    with h5py.File(f"{BASE}/{run}/sample_0/pump_probe_spectroscopy.h5") as f:
        return f['/reference/times'][:], f[f'/reference/M_{ds}'][:,l]
t,l2_0 =ref("hybF_00","global_SU3",1)
_,l2_1 =ref("flu0.12_gs1","global_SU3",1)
_,l2_2 =ref("hybF_002","global_SU3",1)
_,mx   =ref("flu0.12_gs1","global_SU2",0)
def bp(M,f0,df=0.05):
    """Gaussian band-pass around f0 (THz) — no brick-wall ringing"""
    F=np.fft.rfft(M-M.mean()); fr=np.fft.rfftfreq(len(M),t[1]-t[0])*SCALE
    F*=np.exp(-0.5*((fr-f0)/df)**2)
    return np.fft.irfft(F,n=len(M))
fig,axs=plt.subplots(1,3,figsize=(15.5,4.6))
# (a) the dipole motion: big E12 ring, small forced wobble riding on it
ax=axs[0]; sel=(t>10)&(t<70)
ax.plot(t[sel]*PS,l2_2[sel]-l2_2[t>150].mean(),color="C0",lw=1.2,label="$\\lambda^2(t)$ (full)")
ax.plot(t[sel]*PS,20*bp(l2_2-l2_0,0.90)[sel],color="C3",lw=1.5,
        label="the $W^{xz}_1$-induced 0.90 THz part ($\\times20$)")
ax.set_xlabel("t (ps)"); ax.set_ylabel("Tm $m_z\\propto\\lambda^2$")
ax.set_title("(a) the dipole rings at its own $E_{12}$=0.50 THz\nwith a small forced 0.90 THz wobble on top",fontsize=10)
ax.legend(fontsize=8)
# (b) the wobble grows with W1_xz and is phase-locked to the magnon
ax=axs[1]; sel2=(t>12)&(t<45)
for M,lab,c in [(l2_1-l2_0,"$W^{xz}_1=0.01$","C2"),(l2_2-l2_0,"$0.02$","C0")]:
    ax.plot(t[sel2]*PS,bp(M,0.90)[sel2],color=c,lw=1.6,label=lab)
b=bp(mx,0.90); sc=np.abs(bp(l2_2-l2_0,0.90)[sel2]).max()/np.abs(b[sel2]).max()
ax.plot(t[sel2]*PS,sc*b[sel2],color="C1",lw=1.1,ls="--",label="Fe $m_x$ source (scaled)")
ax.set_xlabel("t (ps)"); ax.set_ylabel("$W^{xz}_1$-induced 0.90 THz part of $\\lambda^2$")
ax.set_title("(b) the wobble is linear in $W^{xz}_1$ and\nphase-locked to the magnon that drives it",fontsize=10)
ax.legend(fontsize=8,ncol=2)
# (c) line inventory
ax=axs[2]
selt=t>3.0; w=np.hanning(selt.sum())
fr=np.fft.rfftfreq(selt.sum(),t[1]-t[0])*SCALE
for M,lab,c in [(l2_0,"$W^{xz}_1=0$","0.55"),(l2_1,"$0.01$","C2"),(l2_2,"$0.02$","C0")]:
    S=np.abs(np.fft.rfft((M[selt]-M[selt].mean())*w))
    ax.semilogy(fr,S/S.max(),color=c,lw=1.4,label=lab)
for f0,nm,c in [(0.50,"$E_{12}$ own",  "C2"),(0.90,"$q_{\\rm AFM}$ forced","C3"),
                (1.00,"$2E_{12}$ harmonic","C4"),(1.80,"$2q_{\\rm AFM}$","C5")]:
    ax.axvline(f0,color=c,ls=":",lw=1.0)
    ax.text(f0-0.015,3e-5,nm,rotation=90,fontsize=7,ha="right",va="bottom",color=c)
ax.set_xlim(0.2,2.0); ax.set_ylim(1e-5,3)
ax.set_xlabel("frequency (THz)"); ax.set_ylabel("$|\\lambda^2(\\omega)|$ (norm.)")
ax.set_title("(c) the line sits at 0.90 (the magnon), not at\nthe $2E_{12}$=1.00 harmonic, which is 12$\\times$ weaker",fontsize=10)
ax.legend(fontsize=8,loc="upper right")
fig.suptitle("The Tm dipole emits at the magnon frequency by off-resonant forced response through $G_z\\,\\delta S_x\\,\\lambda^1$ — not by harmonic generation",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/tm_dipole_forced.png",dpi=115)
print("saved")
