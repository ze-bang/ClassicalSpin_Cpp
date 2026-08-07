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
B =["EXPERIMENTAL_FINAL/w3_isolation/l1d_final"]
S = A2   # same-pol is the SAME experiment as A-cross, only the detected polarization differs
# ---------------------------------------------------------------------------
# The two OUTPUT polarizations, and the full operator each one analyses.
# Tm 4c site mirror m (z->-z):  m_z, d_x EVEN ;  m_x, d_z ODD.
# Time reversal in a real singlet basis puts the MAGNETIC moment purely on
# lambda^{2,5,7}; the ELECTRIC dipole is the T-even complement lambda^{1,3,4,6,8}.
#   P2 = (E||a, H||c)  [A cross, B same-pol] : both operators mirror-EVEN
#        -> m_z: F_z + mu*l2      d_x: l1 AND the diagonals l3, l8 (populations)
#   P1 = (E||c, H||a)  [A same-pol, B cross] : both operators mirror-ODD
#        -> m_x: F_x + l5,l7      d_z: l4, l6      (no diagonal -> no population)
# The P2 population readout is the asymmetry: only that polarization can see
# lambda3/lambda8.  W_E1 is the E1/M1 balance -- an O(1) free parameter that
# the atlas is insensitive to everywhere except the geometry-B rectified line.
W_E1, C3 = 5.264, 0.0   # lambda3 detection REMOVED (user physics call 2026-08-07): no population readout
# ---------------------------------------------------------------------------
panels=[]
# 1. A cross: detect (E||a,H||c) -> F_z + 5.264 l2 + W_E1(l1 + C3 l3) + beta l1 l2
wtb,wTb,a=mix(A2,"SU3",1,10); _,_,b=mix(A2,"SU2",2,10)
_,_,a1=mix(A2,"SU3",0,10); _,_,a3=mix(A2,"SU3",2,10)
# quadratic (hyperpolarizability) emission, allowed because Tm 4c lacks inversion:
# m_z  =  mu*lambda2  +  beta*lambda1*lambda2   -> emits at 2*E12
def quad(runs,T):  # quadratic (hyperpolarisability) channel
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
lin=5.264*a+b+W_E1*(a1+C3*a3)
beta=lin[np.argmin(np.abs(wTb-0.90))][np.argmin(np.abs(wtb-0.50))]/aq[np.argmin(np.abs(wTb-0.49))][np.argmin(np.abs(wtb-1.00))]
mz=lin+beta*aq
panels.append(("A cross   DRIVE: B$\\parallel$a$\\,f(t)$ Zeeman on Fe (0.12) + Tm: E$\\parallel$c d$_z^{eff}$3.0$\\lambda^2$, B$\\parallel$a $\\mu_x$0.913$\\lambda^7$ (amp 0.0220)\n"
               "DETECT (E$\\parallel$a,H$\\parallel$c) out: $F_z{\\equiv}$0 + 5.264$\\lambda^2$($m_z$) + %.1f$\\lambda^1$($d_x$) + %.0f$\\lambda^1\\lambda^2$;  NO $\\lambda^{3,8}$,  $T$=10 K"%(W_E1,beta),
               wtb,wTb,mz,"A_cross"))
# 2. B cross: detect m_x = Fe Sx
wtb,wTb,e=mix(B,"SU2",0,0)
panels.append(("B cross   DRIVE: B$\\parallel$c$\\,f(t)$ Zeeman on Fe (0.10) + Tm: B$\\parallel$c $\\mu_z$0.0307$\\lambda^2$, E$\\parallel$a d$_x$($-$0.0165)$\\lambda^1$ [w$_{E1}$=0.54]\nDETECT (E$\\parallel$c,H$\\parallel$a) out: $m_x$ = Fe S$_x$ (M1); Tm $\\lambda^{5,7}{\\equiv}$0 (closure), $\\kappa^B$=0,  $T$=0",wtb,wTb,e,"B_cross"))
# 3. A same-pol: detect E||c -> CEF composite
wtb,wTb,f4=mix(S,"SU3",3,10,True); _,_,f6=mix(S,"SU3",5,10,True)
panels.append(("A same-pol   DRIVE: same single geometry-A pulse as A cross (second analyser on one experiment)\n"
               "DETECT (E$\\parallel$c,H$\\parallel$a) out, plotted: d$_z$ CEF 0.006$\\lambda^4$+4.4$\\lambda^6$ (F$_x$+$\\lambda^{5,7}$ m$_x$ part omitted);  no $\\lambda^{3,8}$ (mirror-odd),  $T$=10 K",
               wtb,wTb,0.006*f4+4.4*f6,"A_same"))
# 4. B same-pol PREDICTION: SAME operator as A cross -- both analyse (E||a,H||c)
wtb,wTb,g_=mix(B,"SU2",2,0); _,_,bl2=mix(B,"SU3",1,0); _,_,bq=quad(B,0)
_,_,bl1=mix(B,"SU3",0,0); _,_,bl3=mix(B,"SU3",2,0)
panels.append(("B same-pol PREDICTION   DRIVE: same single geometry-B pulse as B cross\n"
               "DETECT (E$\\parallel$a,H$\\parallel$c) out: S$_z$($F_z$)+5.264$\\lambda^2$($m_z$)+%.1f$\\lambda^1$($d_x$)+%.0f$\\lambda^1\\lambda^2$;  NO $\\lambda^3$; $\\lambda^{4-7},\\lambda^8{\\equiv}$0,  $T$=0"%(W_E1,beta),
               wtb,wTb,g_+5.264*bl2+W_E1*(bl1+C3*bl3)+beta*bq,"B_same_pred"))
census={"detection_P2":"(E||a,H||c) = F_z + 5.264*l2 (m_z) + W_E1*l1 (d_x) + beta*l1*l2; NO l3/l8 — population readout removed (user physics call 2026-08-07)",
        "detection_P1":"(E||c,H||a) = F_x + l5,l7 (m_x) + 0.006*l4 + 4.4*l6 (d_z); mirror-ODD so no l3/l8 population term",
        "W_E1":W_E1,"C3":C3,
        "beta_note":"beta re-fitted from the observed A-cross parity (E12,2E12)=(qAFM,E12)",
        "anchor_insensitivity":"A-cross observables move by <=0.04 over a 50x range of W_E1 (see p2_operator_fix.py)",
        "drive":"A_Fe=0.12 tied su3=0.02195, dark-mu13","mu13_admixture_pct":0.14}
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
    ax.set_title(ttl,fontsize=8.2)
fig.suptitle("The TmFeO$_3$ 2DCS atlas — $\\kappa^B$-FREE scenario: one Hamiltonian (all-static Fe-Tm vertices), one physical pulse per geometry\n"
             "(geometry-B coherence written electrically via d$_x$, converted by master W$_1^{xz}$; $\\omega_\\tau$=0 row removed; blind census)",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/tmfeo3_2dcs_final/EXPERIMENTAL_FINAL/w3_isolation/atlas_kBfree.png",dpi=118)
json.dump(census,open("tfo_project/tmfeo3_2dcs_final/EXPERIMENTAL_FINAL/w3_isolation/census_atlas_kBfree.json","w"),indent=1)
print("\nsaved w3_isolation/atlas_kBfree.png + census_atlas_kBfree.json")
