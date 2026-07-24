import numpy as np, h5py
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCALE=2*np.pi/4.135667696
BASE="tfo_project/tmfeo3_2dcs_final"
runs={"gs1":"fin_S_gs1","gs2":"fin_S_gs2","gs3":"fin_S_gs3"}
E=[0.0,2.067834,4.9628]
data={}
for k,r in runs.items():
    d=f"{BASE}/{r}/sample_0"; ch={}
    with h5py.File(f"{d}/pump_probe_spectroscopy.h5") as f:
        t=f['/reference/times'][:]; tau=f['/tau_scan/tau_values'][:]
        for sp,l,nm in [("SU3",3,"l4"),("SU3",5,"l6"),("SU2",0,"fe")]:
            M0=f[f'/reference/M_global_{sp}'][:,l]
            M=np.zeros((len(tau),len(t)))
            for i in range(len(tau)):
                g=f[f'/tau_scan/tau_{i}']
                M[i]=g[f'M01_global_{sp}'][:,l]-g[f'M1_global_{sp}'][:,l]-M0
            ch[nm]=M
    data[k]=ch
tm=t>=3.0; tk=t[tm]
apod=np.exp(np.log(0.03)*((tk-tk[0])/(tk[-1]-tk[0]))**2)
w=np.ones_like(tau); m1=tau>-6; w[m1]=0.5*(1-np.cos(np.pi*(-tau[m1])/6))
m2=tau<-110; w[m2]=0.5*(1-np.cos(np.pi*(120+tau[m2])/10))
wt=np.fft.fftshift(np.fft.fftfreq(tm.sum(),t[1]-t[0]))*SCALE
wta=np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
mx=(wt>0.12)&(wt<1.6); my=(np.abs(wta)<1.6)
wtb=wt[mx]; wTb=-wta[my]
dyx=abs(wta[1]-wta[0]); dxx=abs(wt[1]-wt[0])
def spec(M):
    Md=(M[:,tm]-M[:,tm].mean(axis=0,keepdims=True))*w[:,None]*apod[None,:]
    return gaussian_filter(np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my,mx)],sigma=(2,1.5))
def blind(A,amp_min=0.04,prom_min=1.5,nmax=12):
    n=A.max(); fp=(int(0.10/dyx)|1,int(0.10/dxx)|1)
    ismax=(A==maximum_filter(A,size=fp))
    bg=median_filter(A,size=(int(0.34/dyx)|1,int(0.34/dxx)|1))
    ys,xs=np.where(ismax); pk=[]
    for y,x in zip(ys,xs):
        a=A[y,x]/n; p=A[y,x]/max(bg[y,x],1e-30)
        if a>=amp_min and p>=prom_min: pk.append((a,p,wTb[y],wtb[x]))
    pk.sort(reverse=True); out=[]
    for q in pk:
        if all(abs(q[2]-r[2])>0.09 or abs(q[3]-r[3])>0.09 for r in out): out.append(q)
    return out[:nmax]
def mixM(nm,T):
    kT=T*0.086173
    ws=np.array([np.exp(-e/kT) for e in E]); ws/=ws.sum()
    return ws[0]*data["gs1"][nm]+ws[1]*data["gs2"][nm]+ws[2]*data["gs3"][nm], ws
fig,axs=plt.subplots(1,4,figsize=(21.5,5.2))
for ax,T in zip(axs[:3],[0.0,10.0,20.0]):
    if T==0: Ml4,Ml6,Mfe=data["gs1"]["l4"],data["gs1"]["l6"],data["gs1"]["fe"]; ws=[1,0,0]
    else:
        Ml4,ws=mixM("l4",T); Ml6,_=mixM("l6",T); Mfe,_=mixM("fe",T)
    A=0.05*spec(Ml4)+4.4*spec(Ml6)
    pks=blind(A)
    print(f"\nT={T}K (weights {np.round(ws,3)}) — composite 0.05*A4+4.4*A6 (Fe=0), blind census:")
    for a,p,yy,xx in pks: print(f"   {a:5.2f} x{p:4.1f}  ({yy:+.2f}, {xx:.2f})")
    n=A.max()
    ax.pcolormesh(wtb,wTb,A/n,shading="auto",cmap="inferno",vmin=0,vmax=1,rasterized=True)
    for nm,v in [("qFM",0.38),("E12",0.50),("E23",0.70),("qAFM",0.90),("E13",1.20)]:
        ax.axvline(v,color="w",ls="--",lw=0.7,alpha=0.35)
        ax.axhline(v,color="w",ls="--",lw=0.7,alpha=0.35)
        ax.axhline(-v,color="w",ls="--",lw=0.7,alpha=0.35)
        ax.text(v,1.30,nm,ha="center",va="top",color="w",alpha=0.55,fontsize=6)
        ax.text(1.565,v,nm,ha="left",va="center",color="0.35",fontsize=6)
        ax.text(1.565,-v,nm,ha="left",va="center",color="0.35",fontsize=6)
    for a,p,yy,xx in pks:
        ax.plot(xx,yy,"x",color="cyan",ms=9,mew=1.8)
        if a>0.05: ax.annotate(f"({yy:+.1f},{xx:.2f})",(xx,yy),textcoords="offset points",xytext=(4,5),color="cyan",fontsize=7)
    ax.set_xlim(0.15,1.55); ax.set_ylim(-1.4,1.4)
    ax.set_xlabel("omega_t (THz)"); ax.set_ylabel("physical omega_T (THz)")
    ax.set_title(f"T = {T:.0f} K",fontsize=10)
# Fe channel alone at 20K — hunt (0.9, 0.38)
Mfe,_=mixM("fe",20.0)
Afe=spec(Mfe)
print("\nFe channel alone (T=20K) — blind census:")
for a,p,yy,xx in blind(Afe): print(f"   {a:5.2f} x{p:4.1f}  ({yy:+.2f}, {xx:.2f})")
# S_DC(tau) line -> (0.5, 0) peak
Ml4,_=mixM("l4",10.0); Ml6,_=mixM("l6",10.0)
ax=axs[3]
for nm,M in [("l4",Ml4),("l6",Ml6)]:
    S=M[:,tm].mean(axis=1)
    c=np.polyfit(tau,S,3); S=S-np.polyval(c,tau)
    Sw=np.abs(np.fft.fftshift(np.fft.fft(S*w)))
    f_ax=-wta
    sel=np.abs(f_ax)<1.6
    ax.plot(f_ax[sel],Sw[sel]/Sw[sel].max(),label=f"S_DC from {nm}")
    ipk=np.argmax(Sw[sel]); print(f"\nS_DC({nm}): strongest line at omega_T = {f_ax[sel][ipk]:+.2f} THz (rel amp 1.00)")
ax.axvline(0.5,color="gray",ls=":"); ax.axvline(-0.5,color="gray",ls=":")
ax.set_xlabel("physical omega_T (THz)"); ax.set_title("S_DC(tau) FFT — the (0.5, 0) line"); ax.legend(fontsize=8)
fig.suptitle("Same-pol H||a E||c, dark-mu13 model (it14hr): blind census, thermal ensemble",fontsize=12)
fig.tight_layout(); fig.savefig(f"{BASE}/figures/samepol_darkmu13_final.png",dpi=112)
print("\nsaved figures/samepol_darkmu13_final.png")
