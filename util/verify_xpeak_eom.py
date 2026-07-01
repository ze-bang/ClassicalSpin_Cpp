#!/usr/bin/env python3
"""verify_xpeak_eom.py — TmFeO3 cross-peak numerical verification"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Parameters
e1=0.97; e2=3.97; g_Tm=7./12.; mu_z2=5.264
omega_qAFM=3.84; kick_amp=0.05; H2_chi_uniform=11.999939

sigma_F=np.array([+1.,+1.,+1.,+1.])
sigma_G=np.array([+1.,-1.,+1.,-1.])
sigma_C=np.array([+1.,-1.,-1.,+1.])
sigma_A=np.array([+1.,+1.,-1.,-1.])
eta_z_Tm=sigma_C

even_bonds_full=[(1,0,3),(2,0,2),(3,0,1),(4,0,0),(1,1,2),(2,1,3),(3,1,0),(4,1,1),(1,2,1),(2,2,0),(3,2,3),(4,2,2),(1,3,0),(2,3,1),(3,3,2),(4,3,3)]
odd_bonds_full=[(1,0,0),(2,0,1),(3,0,2),(4,0,3),(1,1,1),(2,1,0),(3,1,3),(4,1,2),(1,2,2),(2,2,3),(3,2,0),(4,2,1),(1,3,3),(2,3,2),(3,3,1),(4,3,0)]

def build_tm_bonds():
    tb=[[] for _ in range(4)]
    for (o,f,t) in even_bonds_full: tb[t].append((o,f,+1))
    for (o,f,t) in odd_bonds_full: tb[t].append((o,f,-1))
    return tb
TM_BONDS=build_tm_bonds()

# ---- PART A: bond filter ----
def part_a():
    print("="*68)
    print("PART A: Bond-filter — which Fe spin patterns activate H^5?")
    print("="*68)
    sc_eq={1:1.,2:1.,3:1.,4:1.}; sc_uq={1:1.,2:1.,3:2.,4:2.}
    def H5(pat,sc):
        h=np.zeros(4)
        for tm in range(4):
            for (o,fe,s57) in TM_BONDS[tm]: h[tm]+=sc[o]*s57*sigma_C[fe]*pat[fe]
        return h
    def lab(v):
        if np.allclose(v,0): return "ZERO"
        for n,s in [("F",sigma_F),("G",sigma_G),("C",sigma_C),("A",sigma_A)]:
            if np.allclose(v/abs(v[0]),s/abs(s[0])): return "sigma_{0}(amp={1:.3f})".format(n,v[0])
        return "mixed"
    for nm,pat in [("sigma_G (qAFM Sy)",sigma_G),("sigma_F",sigma_F),("sigma_C",sigma_C),("sigma_A",sigma_A)]:
        eq=H5(pat,sc_eq); uq=H5(pat,sc_uq)
        print("  Sy~{0}:  equal={1},  unequal(1,1,2,2)={2}".format(nm,lab(eq),lab(uq)))
    H5q=H5(sigma_G,sc_uq)
    print("  KEY: qAFM Sy~sigma_G, unequal orbits -> H5 has pattern: {0}".format(lab(H5q)))
    return H5q

# ---- SU(3) EOM (exact from tex, 0-based) ----
def su3_eom(h,lam):
    dl=np.zeros(8); s3=np.sqrt(3.)/2.
    dl[0]=(h[1]*lam[2]-h[2]*lam[1]+.5*(h[3]*lam[6]-h[6]*lam[3])-.5*(h[4]*lam[5]-h[5]*lam[4]))
    dl[1]=(-(h[0]*lam[2]-h[2]*lam[0])+.5*(h[3]*lam[5]-h[5]*lam[3])+.5*(h[4]*lam[6]-h[6]*lam[4]))
    dl[2]=(h[0]*lam[1]-h[1]*lam[0]+.5*(h[3]*lam[4]-h[4]*lam[3])-.5*(h[5]*lam[6]-h[6]*lam[5]))
    dl[3]=(-0.5*(h[0]*lam[6]-h[6]*lam[0])-0.5*(h[1]*lam[5]-h[5]*lam[1])-0.5*(h[2]*lam[4]-h[4]*lam[2])+s3*(h[4]*lam[7]-h[7]*lam[4]))
    dl[4]=(+0.5*(h[0]*lam[5]-h[5]*lam[0])-0.5*(h[1]*lam[6]-h[6]*lam[1])+0.5*(h[2]*lam[3]-h[3]*lam[2])-s3*(h[3]*lam[7]-h[7]*lam[3]))
    dl[5]=(-0.5*(h[0]*lam[4]-h[4]*lam[0])+0.5*(h[1]*lam[3]-h[3]*lam[1])+0.5*(h[2]*lam[6]-h[6]*lam[2])+s3*(h[6]*lam[7]-h[7]*lam[6]))
    dl[6]=(+0.5*(h[0]*lam[3]-h[3]*lam[0])+0.5*(h[1]*lam[4]-h[4]*lam[1])-0.5*(h[2]*lam[5]-h[5]*lam[2])-s3*(h[5]*lam[7]-h[7]*lam[5]))
    dl[7]=(s3*(h[3]*lam[4]-h[4]*lam[3])+s3*(h[5]*lam[6]-h[6]*lam[5]))
    return dl

# ---- Exact equilibrium via matrix diagonalization ----
def find_eq_exact(h_z):
    h8=(2*e2-e1)/np.sqrt(3.); s3=np.sqrt(3.)
    H2s=H2_chi_uniform+g_Tm*mu_z2*eta_z_Tm*h_z
    lam_eq=np.zeros((4,8))
    for mu in range(4):
        H2=H2s[mu]
        M=np.zeros((3,3),dtype=complex)
        M[0,0]=e1+h8/s3; M[1,1]=-e1+h8/s3; M[2,2]=-2*h8/s3
        M[0,1]=-1j*H2; M[1,0]=1j*H2
        ev,ec=np.linalg.eigh(M); psi=ec[:,np.argmin(ev)]
        lam_eq[mu,0]=2.*np.real(np.conj(psi[0])*psi[1])
        lam_eq[mu,1]=(np.conj(psi[0])*(-1j)*psi[1]+np.conj(psi[1])*(1j)*psi[0]).real
        lam_eq[mu,2]=abs(psi[0])**2-abs(psi[1])**2
        lam_eq[mu,3]=2.*np.real(np.conj(psi[0])*psi[2])
        lam_eq[mu,4]=(np.conj(psi[0])*(-1j)*psi[2]+np.conj(psi[2])*(1j)*psi[0]).real
        lam_eq[mu,5]=2.*np.real(np.conj(psi[1])*psi[2])
        lam_eq[mu,6]=(np.conj(psi[1])*(-1j)*psi[2]+np.conj(psi[2])*(1j)*psi[1]).real
        lam_eq[mu,7]=(abs(psi[0])**2+abs(psi[1])**2-2*abs(psi[2])**2)/s3
    return lam_eq

# ---- PART B: equilibrium sigma_C vs h_z ----
def part_b():
    print("")
    print("="*68)
    print("PART B: Equilibrium sigma_C.lambda^2 vs h_z  (predict: linear in h_z)")
    print("="*68)
    print("  {:>6}  {:>10}  {:>10}  {:>10}  {:>12}".format("h_z","c_F(l2)","c_C(l2)","Mz_Tm","dMz/dh_z"))
    h_zs=[0.,0.25,0.5,1.,2.,4.]; cC=[]
    for hz in h_zs:
        L=find_eq_exact(hz); l2=L[:,1]
        cF=np.dot(l2,sigma_F)/4.; cc=np.dot(l2,sigma_C)/4.; Mz=np.dot(eta_z_Tm,l2)
        rat=Mz/hz if hz>0 else float("nan")
        print("  {:>6.2f}  {:>10.6f}  {:>10.6f}  {:>10.6f}  {:>12.6f}".format(hz,cF,cc,Mz,rat))
        cC.append(cc)
    print("  --> sigma_C component grows with h_z (confirms Zeeman injects sigma_C into lambda^2)")
    return h_zs,cC

# ---- Dynamics ----
def run_dyn(h_z,amp=kick_amp,t_end=400.,dt=0.02):
    h8=(2*e2-e1)/np.sqrt(3.)
    H2s=H2_chi_uniform+g_Tm*mu_z2*eta_z_Tm*h_z
    lam_eq=find_eq_exact(h_z); y0=lam_eq.ravel().copy()
    def rhs(t,y):
        lam=y.reshape(4,8); dl=np.zeros((4,8))
        fac=amp*np.sin(omega_qAFM*t)
        for mu in range(4):
            hv=np.zeros(8); hv[1]=H2s[mu]; hv[2]=e1
            hv[4]=fac*sigma_A[mu]; hv[6]=fac*sigma_A[mu]; hv[7]=h8
            dl[mu]=su3_eom(hv,lam[mu])
        return dl.ravel()
    tev=np.arange(0,t_end+dt,dt)
    sol=solve_ivp(rhs,[0,t_end],y0,method="RK45",t_eval=tev,rtol=1e-8,atol=1e-11,max_step=0.04)
    t=sol.t; lt=sol.y.reshape(4,8,-1)
    sC=np.einsum("s,st->t",sigma_C,lt[:,1,:])
    sF=np.einsum("s,st->t",sigma_F,lt[:,1,:])
    return t,sC,sF

# ---- PART C ----
def part_c():
    print(""); print("="*68)
    print("PART C: Dynamics driven by H^5/H^7~sigma_A*sin(omega_qAFM*t)")
    print("="*68)
    print("  {:>5}  {:>22}  {:>22}".format("h_z","RMS(sigma_C.lam2)","RMS(sigma_F.lam2)"))
    res={}
    for hz in [0.,0.5,1.,2.]:
        t,sC,sF=run_dyn(hz)
        i0=np.searchsorted(t,150.)
        rC=np.sqrt(np.mean(sC[i0:]**2)); rF=np.sqrt(np.mean(sF[i0:]**2))
        print("  {:>5.1f}  {:>22.10f}  {:>22.10f}".format(hz,rC,rF))
        res[hz]=(t,sC,sF)
    return res

# ---- PART D ----
def part_d():
    print(""); print("="*68); print("PART D: Amplitude vs h_z (predict: linear)"); print("="*68)
    hzs=[0.,0.25,0.5,0.75,1.,1.5,2.,3.,4.]; amps=[]
    for hz in hzs:
        t,sC,_=run_dyn(hz,t_end=300.)
        i0=np.searchsorted(t,100.); a=np.sqrt(np.mean(sC[i0:]**2)); amps.append(a)
        print("  h_z={0:.3f}: {1:.10f}".format(hz,a))
    ha=np.array(hzs); aa=np.array(amps); mask=ha>0
    sl=np.polyfit(ha[mask],aa[mask],1)
    res=aa[mask]-np.polyval(sl,ha[mask]); ss_r=np.sum(res**2); ss_t=np.sum((aa[mask]-np.mean(aa[mask]))**2)
    R2=1-ss_r/ss_t if ss_t>0 else 0
    print("  Linear fit: {0:.4e}*h_z + {1:.3e},  R^2={2:.6f}".format(sl[0],sl[1],R2))
    return hzs,amps,sl

# ---- PART E: spectral analysis ----
def part_e(hz=1.0):
    print(""); print("="*68); print("PART E: Spectrum at h_z={0}".format(hz)); print("="*68)
    t,sC,_=run_dyn(hz,t_end=1500.,dt=0.01)
    i0=np.searchsorted(t,400.); sig=sC[i0:]; dta=t[i0+1]-t[i0]
    N=len(sig); fr=np.fft.rfftfreq(N,d=dta)*2*np.pi; sp=np.abs(np.fft.rfft(sig))/N
    thr=0.03*sp.max(); pk=np.where(sp>thr)[0]
    hb=4.13567
    print("  {:>10}  {:>8}  {:>10}  label".format("w(meV)","THz","|FFT|"))
    for idx in pk[:25]:
        w,s=fr[idx],sp[idx]; lb=""
        if abs(w-e1)<0.06: lb="<-- omega_E12 ***"
        elif abs(w-omega_qAFM)<0.06: lb="<-- omega_qAFM"
        elif abs(w-(omega_qAFM-e1))<0.1: lb="<-- qAFM-E12"
        elif abs(w-(omega_qAFM+e1))<0.1: lb="<-- qAFM+E12"
        elif abs(w-2*omega_qAFM)<0.1: lb="<-- 2*qAFM"
        print("  {:>10.4f}  {:>8.4f}  {:>10.6f}  {}".format(w,w/hb,s,lb))
    print("  omega_E12={0:.3f} meV={1:.4f} THz".format(e1,e1/hb))
    print("  2*omega_qAFM={0:.3f} meV={1:.4f} THz".format(2*omega_qAFM,2*omega_qAFM/hb))
    return t[i0:],sig,fr,sp

# ---- Plot ----
def plot_all(res_c,hzs,amps,t_f,s_f,fr,sp):
    fig,axes=plt.subplots(3,1,figsize=(10,13))
    fig.suptitle("TmFeO3 cross-peak verification: (omega_qAFM, omega_E12) under H//c",fontsize=12,fontweight="bold")
    colors=["#555","#e74c3c","#2980b9","#27ae60"]
    ax=axes[0]
    for hz,col in zip([0.,0.5,1.,2.],colors):
        t,sC,_=res_c[hz]; mask=(t>=50)&(t<=350)
        ax.plot(t[mask],sC[mask],lw=1.5,color=col,label="h_z={0} meV".format(hz),ls="--" if hz==0 else "-")
    ax.axhline(0,color="gray",lw=0.5,ls=":"); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    ax.set_xlabel("Time (hbar/meV)"); ax.set_ylabel("sigma_C . lam2(t)")
    ax.set_title("Cross-peak channel M_z_Tm driven by H^5/H^7~sigma_A*sin(omega_qAFM*t)")
    ax=axes[1]
    ha=np.array(hzs); aa=np.array(amps)
    ax.plot(ha,aa,"o-",ms=6,color="#e74c3c",lw=2)
    mask=ha>0
    if mask.sum()>1:
        sl=np.polyfit(ha[mask],aa[mask],1); hf=np.linspace(0,ha.max(),100)
        ax.plot(hf,np.polyval(sl,hf),"--",color="#e74c3c",alpha=.5,label="Linear fit slope={0:.3e}".format(sl[0]))
        ax.legend(fontsize=9)
    ax.axhline(0,color="gray",lw=0.5,ls=":"); ax.grid(True,alpha=0.3)
    ax.set_xlabel("h_z (meV)"); ax.set_ylabel("RMS|sigma_C.lam2|"); ax.set_title("Amplitude vs field")
    ax=axes[2]
    hb=4.13567; fTHz=fr/hb; mask2=fTHz<2.0
    ax.semilogy(fTHz[mask2],sp[mask2]+1e-12,color="#2980b9",lw=1.5)
    ax.axvline(e1/hb,color="#e74c3c",ls="--",lw=2,label="omega_E12={0:.3f} THz".format(e1/hb))
    ax.axvline(omega_qAFM/hb,color="#27ae60",ls="--",lw=1.5,label="omega_qAFM={0:.3f} THz".format(omega_qAFM/hb))
    ax.axvline(2*omega_qAFM/hb,color="#9b59b6",ls=":",lw=1.5,label="2*qAFM={0:.3f} THz".format(2*omega_qAFM/hb))
    ax.set_xlabel("Frequency (THz)"); ax.set_ylabel("|FFT(M_z_Tm)|"); ax.legend(fontsize=9); ax.grid(True,which="both",alpha=0.3)
    ax.set_title("Spectrum at h_z=1 meV")
    plt.tight_layout()
    out="/home/pc_linux/ClassicalSpin_Cpp/util/xpeak_verification.png"
    plt.savefig(out,dpi=130); print("Saved: "+out)

if __name__=="__main__":
    np.set_printoptions(precision=6,suppress=True)
    print("TmFeO3 cross-peak verification (hbar=1, meV)")
    H5q=part_a()
    h_zl,cC=part_b()
    res_c=part_c()
    hzs,amps,sl=part_d()
    t_f,s_f,fr,sp=part_e(hz=1.)
    plot_all(res_c,hzs,amps,t_f,s_f,fr,sp)
    print(""); print("="*68)
    t0,sC0,_=res_c[0.]; t1,sC1,_=res_c[1.]
    i0=np.searchsorted(t0,150.)
    r0=np.sqrt(np.mean(sC0[i0:]**2)); r1=np.sqrt(np.mean(sC1[i0:]**2))
    print("  h_z=0: RMS(Mz_Tm)={0:.2e}  {1}".format(r0,"OK" if r0<1e-7 else "FAIL"))
    print("  h_z=1: RMS(Mz_Tm)={0:.6f}  {1}".format(r1,"OK" if r1>1e-6 else "FAIL"))
    ok=(r0<1e-7)and(r1>1e-6)
    print("STATUS: {0}".format("PASSED" if ok else "FAILED"))
