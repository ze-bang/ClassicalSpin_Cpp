import numpy as np, h5py, json
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SCALE=2*np.pi/4.135667696
BASE="tfo_project/tmfeo3_2dcs_final"; OUT="tfo_project/paper/data"
E=[0.0,2.067834,4.9628]
MODES=[("qFM",0.38),("E12",0.50),("E23",0.70),("qAFM",0.90),("E13",1.20)]
def load(run,sp,l):
    with h5py.File(f"{BASE}/{run}/sample_0/pump_probe_spectroscopy.h5") as f:
        t=f['/reference/times'][:]; tau=f['/tau_scan/tau_values'][:]
        M0=f[f'/reference/M_global_{sp}'][:,l]
        M=np.zeros((len(tau),len(t)))
        for i in range(len(tau)):
            g=f[f'/tau_scan/tau_{i}']
            M[i]=g[f'M01_global_{sp}'][:,l]-g[f'M1_global_{sp}'][:,l]-M0
    return t,tau,M
def spec(t,tau,M,samepol=False):
    tm=t>=3.0
    if samepol:
        tk=t[tm]; at=np.exp(np.log(0.03)*((tk-tk[0])/(tk[-1]-tk[0]))**2)
        atau=np.ones_like(tau); m1=tau>-6; atau[m1]=0.5*(1-np.cos(np.pi*(-tau[m1])/6))
        m2=tau<-110; atau[m2]=0.5*(1-np.cos(np.pi*(120+tau[m2])/10))
    else:
        at=np.hanning(2*tm.sum())[tm.sum():]; atau=np.hanning(2*len(tau))[:len(tau)]
    Md=(M[:,tm]-M[:,tm].mean(axis=0,keepdims=True))*atau[:,None]*at[None,:]   # tau-mean removal
    wt=np.fft.fftshift(np.fft.fftfreq(tm.sum(),t[1]-t[0]))*SCALE
    wta=np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
    mx=(wt>0.12)&(wt<1.9); my=(np.abs(wta)<1.6)
    A=gaussian_filter(np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my,mx)],sigma=(2,1.5))
    return wt[mx],-wta[my],A
def blind(wtb,wTb,A,amp_min=0.05):
    B=A.copy(); B[np.abs(wTb)<0.18,:]=0
    dy=abs(wTb[1]-wTb[0]); dx=abs(wtb[1]-wtb[0]); n=B.max()
    ismax=(B==maximum_filter(B,size=(int(0.10/dy)|1,int(0.10/dx)|1)))
    bg=median_filter(B,size=(int(0.34/dy)|1,int(0.34/dx)|1))
    ys,xs=np.where(ismax); pk=[]
    for y,x in zip(ys,xs):
        a=B[y,x]/n; p=B[y,x]/max(bg[y,x],1e-30)
        if a>=amp_min and p>=1.5: pk.append((round(float(a),3),round(float(p),1),round(float(wTb[y]),2),round(float(wtb[x]),2)))
    pk.sort(reverse=True); out=[]
    for q in pk:
        if all(abs(q[2]-r[2])>0.09 or abs(q[3]-r[3])>0.09 for r in out): out.append(q)
    return out[:8]
def mix(runs,sp,l,T,samepol=False):
    ws=np.array([1.,0,0]) if T<=0 else np.array([np.exp(-e/(T*0.086173)) for e in E])
    ws=ws/ws.sum(); tot=None
    for w,r in zip(ws,runs):
        if w<1e-6: continue
        t,tau,M=load(r,sp,l); tot=w*M if tot is None else tot+w*M
    return spec(t,tau,tot,samepol)
A2=[f"flu0.12_{g}" for g in ["gs1","gs2","gs3"]]   # geometry A: ONE drive for both its channels
B =[f"EXPERIMENTAL_FINAL/fin_B_{g}"  for g in ["gs1","gs2","gs3"]]
S = A2   # same-pol is the SAME experiment as A-cross, only the detected polarization differs
panels=[]
# 1. A cross: detect H||c  ->  m_z (5.264 l2 + Sz) + w_E1 * x-E1 (2.39 l5 + 0.91 l7)
wtb,wTb,a=mix(A2,"SU3",1,10); _,_,b=mix(A2,"SU2",2,10)
# quadratic (hyperpolarizability) emission, allowed because Tm 4c lacks inversion:
# m_z  =  mu*lambda2  +  beta*lambda1*lambda2   -> emits at 2*E12
def quad(runs,T):
    ws=np.array([1.,0,0]) if T<=0 else np.array([np.exp(-e/(T*0.086173)) for e in E]); ws=ws/ws.sum()
    tot=None
    for w,r in zip(ws,runs):
        if w<1e-6: continue
        with h5py.File(f"{BASE}/{r}/sample_0/pump_probe_spectroscopy.h5") as f:
            t=f['/reference/times'][:]; tau=f['/tau_scan/tau_values'][:]
            R=f['/reference/M_local_SU3'][:]; q0=R[:,0]*R[:,1]
            Q=np.zeros((len(tau),len(t)))
            for i in range(len(tau)):
                g=f[f'/tau_scan/tau_{i}']
                A_=g['M01_local_SU3'][:]; B_=g['M1_local_SU3'][:]
                Q[i]=A_[:,0]*A_[:,1]-B_[:,0]*B_[:,1]-q0
        tot=w*Q if tot is None else tot+w*Q
    return spec(t,tau,tot)
_,_,aq=quad(A2,10)
beta=5.264*a[np.argmin(np.abs(wTb-0.90))][np.argmin(np.abs(wtb-0.50))]/aq[np.argmin(np.abs(wTb-0.49))][np.argmin(np.abs(wtb-1.00))]
mz=5.264*a+b+beta*aq
panels.append(("A cross  ($H\\parallel a$ in, $H\\parallel c$ out)\n$m_z$=5.264$\\lambda^2$+%.0f$\\lambda^1\\lambda^2$ (Tm has no inversion),  $T$=10 K"%beta,wtb,wTb,mz,"A_cross"))
# 2. B cross: detect m_x = Fe Sx
wtb,wTb,e=mix(B,"SU2",0,0)
panels.append(("B cross  ($H\\parallel c$ in, $H\\parallel a$ out)\nFe $m_x$(M1),  $T$=0",wtb,wTb,e,"B_cross"))
# 3. A same-pol: detect E||c -> CEF composite
wtb,wTb,f4=mix(S,"SU3",3,10,True); _,_,f6=mix(S,"SU3",5,10,True)
panels.append(("A same-pol  ($H\\parallel a$ out): Tm $m_x$\n0.006$\\lambda^4$+4.4$\\lambda^6$,  $T$=10 K",wtb,wTb,0.006*f4+4.4*f6,"A_same"))
# 4. B same-pol PREDICTION: detect m_z (CEF channels are machine-zero here)
wtb,wTb,g_=mix(B,"SU2",2,0)
panels.append(("B same-pol  PREDICTION\nFe $m_z$(M1); CEF channels $\\equiv$0,  $T$=0",wtb,wTb,g_,"B_same_pred"))
census={"beta_note":"m_z = 5.264*l2 + beta*l1*l2, beta from observed parity","drive":"A_Fe=0.12 tied su3=0.02195, dark-mu13","mu13_admixture_pct":0.14}
fig,axs=plt.subplots(2,2,figsize=(12.4,9.4))
for ax,(ttl,wtb,wTb,A,key) in zip(axs.ravel(),panels):
    pk=blind(wtb,wTb,A); census[key]=pk
    print(f"\n=== {key} ===")
    for a_,p,yy,xx in pk: print(f"   {a_:5.2f} x{p:4.1f}  ({yy:+.2f}, {xx:.2f})")
    Z=A.copy(); Z[np.abs(wTb)<0.18,:]=0
    ax.pcolormesh(wtb,wTb,Z/Z.max(),shading="auto",cmap="inferno",vmin=0,vmax=1,rasterized=True)
    for nm,v in MODES:
        ax.axvline(v,color="w",ls="--",lw=0.7,alpha=0.32)
        ax.axhline(v,color="w",ls="--",lw=0.7,alpha=0.32); ax.axhline(-v,color="w",ls="--",lw=0.7,alpha=0.32)
        ax.text(v,1.31,nm,ha="center",va="top",color="w",alpha=0.6,fontsize=6)
        ax.text(1.875,v,nm,ha="left",va="center",color="0.4",fontsize=6)
        ax.text(1.575,-v,nm,ha="left",va="center",color="0.4",fontsize=6)
    for a_,p,yy,xx in pk[:6]:
        ax.plot(xx,yy,"x",color="cyan",ms=9,mew=1.8)
        ax.annotate(f"({yy:+.2f},{xx:.2f})",(xx,yy),textcoords="offset points",xytext=(5,6),color="cyan",fontsize=7)
    ax.set_xlim(0.15,1.85); ax.set_ylim(-1.4,1.4)
    ax.set_xlabel("$\\omega_t$ (THz)"); ax.set_ylabel("physical $\\omega_T$ (THz)")
    ax.set_title(ttl,fontsize=9)
fig.suptitle("The TmFeO$_3$ 2DCS atlas — one Hamiltonian, all four polarization channels\n"
             "(magnetic-resolution detection; $\\omega_\\tau$=0 pump--probe row removed; blind census)",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/atlas_final.png",dpi=118)
json.dump(census,open(f"{OUT}/census_atlas_final.json","w"),indent=1)
print("\nsaved paper/figs/atlas_final.png + data/census_atlas_final.json")
