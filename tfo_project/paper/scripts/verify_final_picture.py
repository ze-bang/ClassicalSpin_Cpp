import numpy as np, h5py, json
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter, uniform_filter1d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCALE=2*np.pi/4.135667696; PS=0.6582119569
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
def spec_hann(t,tau,M):
    tm=t>=3.0
    at=np.hanning(2*tm.sum())[tm.sum():]; atau=np.hanning(2*len(tau))[:len(tau)]
    Md=(M[:,tm]-M[:,tm].mean(axis=0,keepdims=True))*atau[:,None]*at[None,:]
    wt=np.fft.fftshift(np.fft.fftfreq(tm.sum(),t[1]-t[0]))*SCALE
    wta=np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
    mx=(wt>0.12)&(wt<1.6); my=(np.abs(wta)<1.6)
    A=gaussian_filter(np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my,mx)],sigma=(2,1.5))
    return wt[mx],-wta[my],A
def spec_samepol(t,tau,M):
    tm=t>=3.0; tk=t[tm]
    at=np.exp(np.log(0.03)*((tk-tk[0])/(tk[-1]-tk[0]))**2)
    atau=np.ones_like(tau); m1=tau>-6; atau[m1]=0.5*(1-np.cos(np.pi*(-tau[m1])/6))
    m2=tau<-110; atau[m2]=0.5*(1-np.cos(np.pi*(120+tau[m2])/10))
    Md=(M[:,tm]-M[:,tm].mean(axis=0,keepdims=True))*atau[:,None]*at[None,:]
    wt=np.fft.fftshift(np.fft.fftfreq(tm.sum(),t[1]-t[0]))*SCALE
    wta=np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
    mx=(wt>0.12)&(wt<1.6); my=(np.abs(wta)<1.6)
    A=gaussian_filter(np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my,mx)],sigma=(2,1.5))
    return wt[mx],-wta[my],A
def blind(wtb,wTb,A,qe=0.12,amp_min=0.05,nmax=10):
    B=A.copy(); B[np.abs(wTb)<qe,:]=0
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
    return out[:nmax]
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
census={}
# ---------- sanity: NaN + lambda3 shift in each geometry ----------
print("=== sanity (reference trajectories) ===")
for run in ["EXPERIMENTAL_FINAL/fin_A_gs1","EXPERIMENTAL_FINAL/fin_B_gs1","fin_S_gs1"]:
    with h5py.File(f"{BASE}/{run}/sample_0/pump_probe_spectroscopy.h5") as f:
        t=f['/reference/times'][:]; l3=f['/reference/M_global_SU3'][:,2]
    pre=l3[t<-20].mean(); late=l3[np.argmin(np.abs(t-t.max()+5))]
    print(f"{run:32s}: NaN={np.isnan(l3).any()}  dl3(late)={late-pre:+.3e}")
# ---------- geometry A cross: composite ----------
runsA=["EXPERIMENTAL_FINAL/fin_A_gs1","EXPERIMENTAL_FINAL/fin_A_gs2","EXPERIMENTAL_FINAL/fin_A_gs3"]
t,tau,Ml2=mix(runsA,"SU3",1,0); t,tau,Mfe=mix(runsA,"SU2",0,0)
wtb,wTb,Al2=spec_hann(t,tau,Ml2); _,_,Afe=spec_hann(t,tau,Mfe)
main=Al2[np.argmin(np.abs(wTb-0.90))][np.argmin(np.abs(wtb-0.50))]
rev =Afe[np.argmin(np.abs(wTb-0.49))][np.argmin(np.abs(wtb-0.90))]
wM1=main/rev
A=Al2+wM1*Afe
pk=blind(wtb,wTb,A)
census["A_cross_T0"]= {"wM1":round(float(wM1),2),"peaks":pk}
print(f"\n=== A cross, composite (wM1={wM1:.2f}), T=0 ===")
for a,p,yy,xx in pk[:8]: print(f"   {a:5.2f} x{p:4.1f}  ({yy:+.2f}, {xx:.2f})")
# ---------- geometry B cross ----------
runsB=["EXPERIMENTAL_FINAL/fin_B_gs1","EXPERIMENTAL_FINAL/fin_B_gs2","EXPERIMENTAL_FINAL/fin_B_gs3"]
for T in [0,5]:
    t,tau,M=mix(runsB,"SU2",0,T)
    wtb,wTb,B_=spec_hann(t,tau,M)
    pk=blind(wtb,wTb,B_)
    census[f"B_cross_T{T}"]=pk
    print(f"\n=== B cross (Fe Sx), T={T} ===")
    for a,p,yy,xx in pk[:7]: print(f"   {a:5.2f} x{p:4.1f}  ({yy:+.2f}, {xx:.2f})")
# ---------- geometry B: buildup row check ----------
t,tau,M=load("EXPERIMENTAL_FINAL/fin_B_gs1","SU2",0)
w=np.hanning(2*len(tau))[:len(tau)]
S=np.abs(np.fft.fftshift(np.fft.fft((M-M.mean())*w[:,None],axis=0),axes=0))
wT=-np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
S=uniform_filter1d(S,size=int(3.0/(t[1]-t[0])),axis=1)
iy=np.argmin(np.abs(wT-0.50))
print("\n=== B buildup: E12 row vs t ===")
vals=[]
for tp in [5,20,40,60,80,105]:
    v=S[iy][np.argmin(np.abs(t-tp/PS))]; vals.append(v)
    print(f"   t={tp:3d}ps: {v:.3e}")
census["B_buildup_E12row"]={f"t{tp}ps":float(f"{v:.3e}") for tp,v in zip([5,20,40,60,80,105],vals)}
# ---------- geometry A same-pol: composite ----------
runsS=["fin_S_gs1","fin_S_gs2","fin_S_gs3"]
for T in [0,10]:
    t,tau,M4=mix(runsS,"SU3",3,T); _,_,M6=mix(runsS,"SU3",5,T)
    wtb,wTb,A4=spec_samepol(t,tau,M4); _,_,A6=spec_samepol(t,tau,M6)
    A=0.05*A4+4.4*A6
    pk=blind(wtb,wTb,A,qe=0.12,amp_min=0.04)
    census[f"A_same_T{T}"]=pk
    print(f"\n=== A same-pol composite, T={T} ===")
    for a,p,yy,xx in pk[:9]: print(f"   {a:5.2f} x{p:4.1f}  ({yy:+.2f}, {xx:.2f})")
# ---------- geometry B SAME-pol prediction: detect E||a = x-E1 (l4/l6) + m_z M1 (Sz) ----------
for T in [0]:
    t,tau,M4=mix(runsB,"SU3",3,T); _,_,M6=mix(runsB,"SU3",5,T); _,_,Mz=mix(runsB,"SU2",2,T)
    wtb,wTb,A4=spec_hann(t,tau,M4); _,_,A6=spec_hann(t,tau,M6); _,_,Az=spec_hann(t,tau,Mz)
    print(f"\n=== B same-pol prediction, T={T} ===")
    print(f"   CEF x-dipole channels: max|A4|={A4.max():.3e}, max|A6|={A6.max():.3e}  vs  m_z M1: max|Az|={Az.max():.3e}")
    comp=0.05*A4+4.4*A6+0.69*Az
    pk=blind(wtb,wTb,comp)
    census["B_same_prediction_T0"]={"maxA4":float(f"{A4.max():.3e}"),"maxA6":float(f"{A6.max():.3e}"),
                                    "maxAz":float(f"{Az.max():.3e}"),"peaks":pk}
    for a,p,yy,xx in pk[:8]: print(f"   {a:5.2f} x{p:4.1f}  ({yy:+.2f}, {xx:.2f})")
    pkz=blind(wtb,wTb,Az)
    census["B_same_Sz_only_T0"]=pkz
    print("   Sz (m_z M1) channel alone:")
    for a,p,yy,xx in pkz[:6]: print(f"   {a:5.2f} x{p:4.1f}  ({yy:+.2f}, {xx:.2f})")
json.dump(census,open(f"{OUT}/census_final_picture.json","w"),indent=1)
print(f"\nwrote {OUT}/census_final_picture.json")
