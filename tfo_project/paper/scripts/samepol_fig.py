import numpy as np, h5py
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCALE=2*np.pi/4.135667696
BASE="tfo_project/tmfeo3_2dcs_final"
E=[0.0,2.067834,4.9628]
RUNS=[f"flu0.12_{g}" for g in ["gs1","gs2","gs3"]]
W4=0.006      # residual mu13^z admixture, fixed by the observed hierarchy
MODES=[("qFM",0.38),("E12",0.50),("E23",0.70),("qAFM",0.90),("E13",1.20)]
def load(run,l):
    with h5py.File(f"{BASE}/{run}/sample_0/pump_probe_spectroscopy.h5") as f:
        t=f['/reference/times'][:]; tau=f['/tau_scan/tau_values'][:]
        M0=f['/reference/M_global_SU3'][:,l]
        M=np.zeros((len(tau),len(t)))
        for i in range(len(tau)):
            g=f[f'/tau_scan/tau_{i}']
            M[i]=g['M01_global_SU3'][:,l]-g['M1_global_SU3'][:,l]-M0
    return t,tau,M
def spec(t,tau,M):
    tm=t>=3.0; tk=t[tm]
    at=np.exp(np.log(0.03)*((tk-tk[0])/(tk[-1]-tk[0]))**2)
    atau=np.ones_like(tau); m1=tau>-6; atau[m1]=0.5*(1-np.cos(np.pi*(-tau[m1])/6))
    m2=tau<-110; atau[m2]=0.5*(1-np.cos(np.pi*(120+tau[m2])/10))
    Md=(M[:,tm]-M[:,tm].mean(axis=0,keepdims=True))*atau[:,None]*at[None,:]
    wt=np.fft.fftshift(np.fft.fftfreq(tm.sum(),t[1]-t[0]))*SCALE
    wta=np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
    mx=(wt>0.12)&(wt<1.6); my=(np.abs(wta)<1.6)
    return wt[mx],-wta[my],gaussian_filter(np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my,mx)],sigma=(2,1.5))
def blind(wtb,wTb,A,amp_min=0.05,n=8):
    B=A.copy(); B[np.abs(wTb)<0.18,:]=0
    dy=abs(wTb[1]-wTb[0]); dx=abs(wtb[1]-wtb[0]); nm=B.max()
    ismax=(B==maximum_filter(B,size=(int(0.10/dy)|1,int(0.10/dx)|1)))
    bg=median_filter(B,size=(int(0.34/dy)|1,int(0.34/dx)|1))
    ys,xs=np.where(ismax); pk=[]
    for y,x in zip(ys,xs):
        a=B[y,x]/nm; p=B[y,x]/max(bg[y,x],1e-30)
        if a>=amp_min and p>=1.5: pk.append((a,wTb[y],wtb[x]))
    pk.sort(reverse=True); out=[]
    for q in pk:
        if all(abs(q[1]-r[1])>0.09 or abs(q[2]-r[2])>0.09 for r in out): out.append(q)
    return out[:n]
def mix(l,T):
    ws=np.array([1.,0,0]) if T<=0 else np.array([np.exp(-e/(T*0.086173)) for e in E]); ws=ws/ws.sum()
    tot=None
    for w,r in zip(ws,RUNS):
        if w<1e-6: continue
        t,tau,M=load(r,l); tot=w*M if tot is None else tot+w*M
    return t,tau,tot
fig,axs=plt.subplots(1,4,figsize=(21.0,5.2))
for ax,T in zip(axs[:3],[0.0,10.0,20.0]):
    t,tau,M4=mix(3,T); _,_,M6=mix(5,T)
    wtb,wTb,A4=spec(t,tau,M4); _,_,A6=spec(t,tau,M6)
    A=W4*A4+4.4*A6
    pk=blind(wtb,wTb,A)
    print(f"T={T:.0f}K:", [(round(a,2),round(y,2),round(x,2)) for a,y,x in pk])
    Z=A.copy(); Z[np.abs(wTb)<0.18,:]=0
    ax.pcolormesh(wtb,wTb,Z/Z.max(),shading="auto",cmap="inferno",vmin=0,vmax=1,rasterized=True)
    for nm,v in MODES:
        ax.axvline(v,color="w",ls="--",lw=0.7,alpha=0.32)
        ax.axhline(v,color="w",ls="--",lw=0.7,alpha=0.32); ax.axhline(-v,color="w",ls="--",lw=0.7,alpha=0.32)
        ax.text(v,1.31,nm,ha="center",va="top",color="w",alpha=0.6,fontsize=6)
        ax.text(1.575,v,nm,ha="left",va="center",color="0.4",fontsize=6)
        ax.text(1.575,-v,nm,ha="left",va="center",color="0.4",fontsize=6)
    for a,yy,xx in pk[:6]:
        ax.plot(xx,yy,"x",color="cyan",ms=9,mew=1.8)
        ax.annotate(f"({yy:+.2f},{xx:.2f})",(xx,yy),textcoords="offset points",xytext=(5,6),color="cyan",fontsize=7)
    ax.set_xlim(0.15,1.55); ax.set_ylim(-1.4,1.4)
    ax.set_xlabel("$\\omega_t$ (THz)"); ax.set_ylabel("physical $\\omega_T$ (THz)")
    ax.set_title(f"$T$ = {T:.0f} K",fontsize=10)
# S_DC panel: the (E12, 0) rectified line
ax=axs[3]
t,tau,M4=mix(3,10.0); _,_,M6=mix(5,10.0)
tm=t>=3.0
w=np.ones_like(tau); m1=tau>-6; w[m1]=0.5*(1-np.cos(np.pi*(-tau[m1])/6))
m2=tau<-110; w[m2]=0.5*(1-np.cos(np.pi*(120+tau[m2])/10))
wta=np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
for nm,M,c in [("$\\lambda^4$",M4,"C0"),("$\\lambda^6$",M6,"C1")]:
    S=M[:,tm].mean(axis=1)
    S=S-np.polyval(np.polyfit(tau,S,3),tau)
    Sw=np.abs(np.fft.fftshift(np.fft.fft(S*w)))
    f_ax=-wta; sel=np.abs(f_ax)<1.6
    ax.plot(f_ax[sel],Sw[sel]/Sw[sel].max(),color=c,lw=1.7,label=f"$S_{{\\rm DC}}$ from {nm}")
    print(f"S_DC({nm}) line at {f_ax[sel][np.argmax(Sw[sel])]:+.2f} THz")
for v in [0.5,-0.5]: ax.axvline(v,color="0.6",ls=":",lw=1.0)
ax.set_xlabel("physical $\\omega_T$ (THz)"); ax.set_ylabel("normalized")
ax.set_title("$S_{\\rm DC}(\\tau)$ spectrum — the $(E_{12},0)$ line",fontsize=10); ax.legend(fontsize=8)
fig.suptitle("Geometry A, same-polarised ($m_x$ readout) at the consolidated operating point: "
             "$A_{\\rm Fe}$=0.12, $\\mu^z_{13}/\\mu^z_{23}$=0.14%, blind census",fontsize=12)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/samepol_darkmu13_final.png",dpi=112)
print("saved samepol_darkmu13_final.png")
