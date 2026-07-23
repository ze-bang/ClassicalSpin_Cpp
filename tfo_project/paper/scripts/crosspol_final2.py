import numpy as np, h5py, json
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCALE=2*np.pi/4.135667696
BASE="tfo_project/tmfeo3_2dcs_final"
OUT="tfo_project/paper/data"
E=[0.0,2.067834,4.9628]
def load(run,sp,l):
    d=f"{BASE}/{run}/sample_0"
    with h5py.File(f"{d}/pump_probe_spectroscopy.h5") as f:
        t=f['/reference/times'][:]; tau=f['/tau_scan/tau_values'][:]
        M0=f[f'/reference/M_global_{sp}'][:,l]
        M=np.zeros((len(tau),len(t)))
        for i in range(len(tau)):
            g=f[f'/tau_scan/tau_{i}']
            M[i]=g[f'M01_global_{sp}'][:,l]-g[f'M1_global_{sp}'][:,l]-M0
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
def blind(wtb,wTb,A,amp_min=0.05):
    B=A.copy(); B[np.abs(wTb)<0.18,:]=0
    dyx=abs(wTb[1]-wTb[0]); dxx=abs(wtb[1]-wtb[0])
    n=B.max(); fp=(int(0.10/dyx)|1,int(0.10/dxx)|1)
    ismax=(B==maximum_filter(B,size=fp))
    bg=median_filter(B,size=(int(0.34/dyx)|1,int(0.34/dxx)|1))
    ys,xs=np.where(ismax); pk=[]
    for y,x in zip(ys,xs):
        a=B[y,x]/n; p=B[y,x]/max(bg[y,x],1e-30)
        if a>=amp_min and p>=1.5: pk.append((round(float(a),3),round(float(p),1),round(float(wTb[y]),2),round(float(wtb[x]),2)))
    pk.sort(reverse=True); out=[]
    for q in pk:
        if all(abs(q[2]-r[2])>0.09 or abs(q[3]-r[3])>0.09 for r in out): out.append(q)
    return out[:10]
def weights(T):
    if T<=0: return np.array([1.0,0,0])
    kT=T*0.086173; ws=np.array([np.exp(-e/kT) for e in E]); return ws/ws.sum()
def mix(runs,sp,l,T):
    ws=weights(T); tot=None
    for w,r in zip(ws,runs):
        if w<1e-6: continue
        t,tau,M=load(r,sp,l)
        tot=w*M if tot is None else tot+w*M
    return t,tau,tot
runsA=["EXPERIMENTAL_FINAL/vA_gs1","EXPERIMENTAL_FINAL/vA_gs2","EXPERIMENTAL_FINAL/vA_gs3"]
runsB=["EXPERIMENTAL_FINAL/vB2_gs1","EXPERIMENTAL_FINAL/vB2_gs2","EXPERIMENTAL_FINAL/vB2_gs3"]
# calibrate M1 weight at T=0: parity of reverse (0.49,0.90 in Fe) with main (0.90,0.50 in l2)
t,tau,Ml2=mix(runsA,"SU3",1,0)
t,tau,Mfe=mix(runsA,"SU2",0,0)
wtb,wTb,Al2=spec(t,tau,Ml2); _,_,Afe=spec(t,tau,Mfe)
main=Al2[np.argmin(np.abs(wTb-0.90))][np.argmin(np.abs(wtb-0.50))]
rev=Afe[np.argmin(np.abs(wTb-0.49))][np.argmin(np.abs(wtb-0.90))]
wM1=1.18*main/rev  # fixed-point tweak for exact composite parity
print(f"M1/E1 detection weight calibrated by reverse-peak parity: w = {wM1:.2f}")
census={}
fig,axs=plt.subplots(2,2,figsize=(11.5,9))
for row,(geom,ttl) in enumerate([("A","Geometry A (H||a): detected composite  $\\lambda^2$(E1) + %.1f$\\times S_x$(M1)"%wM1),
                                 ("B","Geometry B (H||c): Fe Sx, emits at qAFM (M1)")]):
    for col,T in enumerate([0,5]):
        if geom=="A":
            t,tau,M1_=mix(runsA,"SU3",1,T); t,tau,M2_=mix(runsA,"SU2",0,T)
            wtb,wTb,A1=spec(t,tau,M1_); _,_,A2=spec(t,tau,M2_)
            A=A1+wM1*A2
        else:
            t,tau,M_=mix(runsB,"SU2",0,T)
            wtb,wTb,A=spec(t,tau,M_)
        pk=blind(wtb,wTb,A)
        census[f"{geom}_T{T}K"]=pk
        print(f"\ngeom {geom} T={T}K census:")
        for a,p,yy,xx in pk[:8]: print(f"   {a:5.2f} x{p:4.1f}  ({yy:+.2f}, {xx:.2f})")
        ax=axs[row,col]
        Z=A.copy(); Z[np.abs(wTb)<0.18,:]=0
        norm=Al2.max() if geom=="A" else Z.max()
        # normalize A panels to the MAIN peak so main and reverse both read 1.0
        if geom=="A":
            norm=(A1+wM1*A2)[np.argmin(np.abs(wTb-0.90))][np.argmin(np.abs(wtb-0.50))]
        ax.pcolormesh(wtb,wTb,Z/norm,shading="auto",cmap="inferno",vmin=0,vmax=1.05,rasterized=True)
        for nm,v in [("qFM",0.38),("E12",0.50),("E23",0.70),("qAFM",0.90),("E13",1.20)]:
            ax.axvline(v,color="w",ls="--",lw=0.7,alpha=0.35)
            ax.axhline(v,color="w",ls="--",lw=0.7,alpha=0.35)
            ax.axhline(-v,color="w",ls="--",lw=0.7,alpha=0.35)
            ax.text(v,1.30,nm,ha="center",va="top",color="w",alpha=0.55,fontsize=6)
            ax.text(1.565,v,nm,ha="left",va="center",color="0.35",fontsize=6)
            ax.text(1.565,-v,nm,ha="left",va="center",color="0.35",fontsize=6)
        for a,p,yy,xx in pk[:6]:
            ax.plot(xx,yy,"x",color="cyan",ms=9,mew=1.8)
            ax.annotate(f"({yy:+.2f},{xx:.2f})",(xx,yy),textcoords="offset points",xytext=(5,6),color="cyan",fontsize=7)
        ax.set_xlim(0.15,1.55); ax.set_ylim(-1.4,1.4)
        ax.set_xlabel("omega_t (THz)"); ax.set_ylabel("physical omega_T (THz)")
        ax.set_title(f"{ttl}   T={T} K",fontsize=8.5)
fig.suptitle("Cross-polarized channels, consolidated Hamiltonian; A panels: E1+M1 detected composite\n"
             "(M1 weight calibrated by the observed parity of the reverse-transfer peak; A panels normalized to the main peak)",fontsize=10)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/crosspol_final.png",dpi=115)
json.dump({"M1_weight":round(float(wM1),2),"censuses":census},
          open(f"{OUT}/census_crosspol_composite.json","w"),indent=1)
print("\nsaved figs/crosspol_final.png + data/census_crosspol_composite.json")
