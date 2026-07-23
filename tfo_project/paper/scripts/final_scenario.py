import numpy as np, h5py, json, os
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCALE=2*np.pi/4.135667696
BASE="tfo_project/tmfeo3_2dcs_final"
OUT="tfo_project/paper/data"; os.makedirs(OUT,exist_ok=True)
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
def weights(T):
    if T<=0: return np.array([1.0,0,0])
    kT=T*0.086173
    ws=np.array([np.exp(-e/kT) for e in E]); return ws/ws.sum()
def spec2d(t,tau,M,hann=True):
    tm=t>=3.0
    if hann:
        at=np.hanning(2*tm.sum())[tm.sum():]; atau=np.hanning(2*len(tau))[:len(tau)]
    else:
        tk=t[tm]; at=np.exp(np.log(0.03)*((tk-tk[0])/(tk[-1]-tk[0]))**2)
        atau=np.ones_like(tau); m1=tau>-6; atau[m1]=0.5*(1-np.cos(np.pi*(-tau[m1])/6))
        m2=tau<-110; atau[m2]=0.5*(1-np.cos(np.pi*(120+tau[m2])/10))
    Md=(M[:,tm]-M[:,tm].mean())*atau[:,None]*at[None,:]
    wt=np.fft.fftshift(np.fft.fftfreq(tm.sum(),t[1]-t[0]))*SCALE
    wta=np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
    mx=(wt>0.12)&(wt<1.6); my=(np.abs(wta)<1.6)
    A=gaussian_filter(np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my,mx)],sigma=(2,1.5))
    return wt[mx],-wta[my],A
def blind(wtb,wTb,A,mask_qe=False,amp_min=0.05):
    B=A.copy()
    if mask_qe: B[np.abs(wTb)<0.1,:]=0
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
census_all={}
def channel(tag,runs,sp,l,Ts,hann,mask_qe,comp=None):
    res={}
    for T in Ts:
        ws=weights(T); tot=None
        for w,r in zip(ws,runs):
            if w<1e-6: continue
            t,tau,M=load(r,sp,l)
            tot=w*M if tot is None else tot+w*M
        if comp is not None:   # composite: second channel with weight
            sp2,l2,w4,w6=comp
            tot2=None
            for w,r in zip(ws,runs):
                if w<1e-6: continue
                t,tau,M=load(r,sp2,l2)
                tot2=w*M if tot2 is None else tot2+w*M
            wtb,wTb,A1=spec2d(t,tau,tot,hann); _,_,A2=spec2d(t,tau,tot2,hann)
            A=w4*A1+w6*A2
        else:
            wtb,wTb,A=spec2d(t,tau,tot,hann)
        pk=blind(wtb,wTb,A,mask_qe)
        res[f"T={T}K"]=pk
        np.savez_compressed(f"{OUT}/{tag}_T{T}K.npz",wt=wtb.astype(np.float32),
                            wT=wTb.astype(np.float32),A=(A/A.max()).astype(np.float32))
        print(f"\n[{tag}] T={T}K census:")
        for a,p,yy,xx in pk: print(f"   {a:5.2f} x{p:4.1f}  ({yy:+.2f}, {xx:.2f})")
    census_all[tag]=res
    return wtb,wTb,A
# Geometry A cross (lambda2), Hann, mask QE band
channel("geomA_cross_lambda2",["EXPERIMENTAL_FINAL/vA_gs1","EXPERIMENTAL_FINAL/vA_gs2","EXPERIMENTAL_FINAL/vA_gs3"],"SU3",1,[0,5],True,True)
# Geometry B cross (Fe Sx), kappaB=0.0035 runs
channel("geomB_cross_Sx",["EXPERIMENTAL_FINAL/vB2_gs1","EXPERIMENTAL_FINAL/vB2_gs2","EXPERIMENTAL_FINAL/vB2_gs3"],"SU2",0,[0,5],True,True)
# Same-pol composite (0.05 l4 + 4.4 l6), same-pol windows, no QE mask (bands excluded in reading)
channel("geomA_samepol_composite",["it14hr_gs1","it14hr_gs2","it14hr_gs3"],"SU3",3,[0,10,20],False,False,comp=("SU3",5,0.05,4.4))
with open(f"{OUT}/census_final_scenario.json","w") as f:
    json.dump({"convention":"(amp_rel, prominence, physical omega_T [THz], omega_t [THz]); +omega_T = non-rephasing",
               "hamiltonian":"consolidated: v4 exchange/anisotropy/DM(D1=0.049,D2=0.014) + W1_yy=0.01, W1_xz=0.01, kappaB_1y=0.0035; e1=2.067834, e2=4.9628; measured pulse; measured-linewidth dampings",
               "censuses":census_all},f,indent=1)
print(f"\nwrote {OUT}/census_final_scenario.json")
# ---- regenerated cross-pol figure for the paper ----
fig,axs=plt.subplots(2,2,figsize=(11.5,9))
for row,(tag,ttl) in enumerate([("geomA_cross_lambda2","Geometry A (H||a): Tm lambda2, emits at E12"),
                                ("geomB_cross_Sx","Geometry B (H||c): Fe Sx, emits at qAFM (M1)")]):
    for col,T in enumerate([0,5]):
        z=np.load(f"{OUT}/{tag}_T{T}K.npz")
        ax=axs[row,col]
        A=z["A"].copy(); A[np.abs(z["wT"])<0.1,:]=0; A/=A.max()
        ax.pcolormesh(z["wt"],z["wT"],A,shading="auto",cmap="inferno",vmin=0,vmax=1,rasterized=True)
        for a,p,yy,xx in census_all[tag][f"T={T}K"][:4]:
            ax.plot(xx,yy,"x",color="cyan",ms=9,mew=1.8)
            ax.annotate(f"({yy:+.2f},{xx:.2f})",(xx,yy),textcoords="offset points",xytext=(5,6),color="cyan",fontsize=7)
        ax.set_xlim(0.15,1.55); ax.set_ylim(-1.4,1.4)
        ax.set_xlabel("omega_t (THz)"); ax.set_ylabel("physical omega_T (THz)")
        ax.set_title(f"{ttl}   T={T} K",fontsize=9)
fig.suptitle("Cross-polarized channels, consolidated Hamiltonian (blind census; quasi-elastic band masked)",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/crosspol_final.png",dpi=115)
print("saved paper/figs/crosspol_final.png")
