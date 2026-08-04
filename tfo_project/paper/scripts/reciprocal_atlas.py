import numpy as np, h5py
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
SCALE=2*np.pi/4.135667696
BASE="tfo_project/tmfeo3_2dcs_final"
def ch(run,sp,l,quad=False):
    with h5py.File(f"{BASE}/{run}/sample_0/pump_probe_spectroscopy.h5") as f:
        t=f['/reference/times'][:]; tau=f['/tau_scan/tau_values'][:]
        if quad:
            R=f['/reference/M_local_SU3'][:]; ref=R[:,0]*R[:,1]
        else:
            ref=f[f'/reference/M_global_{sp}'][:,l]
        M=np.zeros((len(tau),len(t)))
        for i in range(len(tau)):
            g=f[f'/tau_scan/tau_{i}']
            if quad:
                A_=g['M01_local_SU3'][:]; B_=g['M1_local_SU3'][:]
                M[i]=A_[:,0]*A_[:,1]-B_[:,0]*B_[:,1]-ref
            else:
                M[i]=g[f'M01_global_{sp}'][:,l]-g[f'M1_global_{sp}'][:,l]-ref
    return t,tau,M
def spec(t,tau,M):
    tm=t>=3.0
    at=np.hanning(2*tm.sum())[tm.sum():]; atau=np.hanning(2*len(tau))[:len(tau)]
    Md=(M[:,tm]-M[:,tm].mean(axis=0,keepdims=True))*atau[:,None]*at[None,:]
    wt=np.fft.fftshift(np.fft.fftfreq(tm.sum(),t[1]-t[0]))*SCALE
    wta=np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
    mx=(wt>0.12)&(wt<1.9); my=(np.abs(wta)<1.6)
    return wt[mx],-wta[my],gaussian_filter(np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my,mx)],sigma=(2,1.5))
def blind(wtb,wTb,A,amp_min=0.08,n=6):
    B=A.copy(); B[np.abs(wTb)<0.18,:]=0
    dy=abs(wTb[1]-wTb[0]); dx=abs(wtb[1]-wtb[0]); nm=B.max()
    ismax=(B==maximum_filter(B,size=(int(0.10/dy)|1,int(0.10/dx)|1)))
    bg=median_filter(B,size=(int(0.34/dy)|1,int(0.34/dx)|1))
    ys,xs=np.where(ismax); pk=[]
    for y,x in zip(ys,xs):
        a=B[y,x]/nm; p=B[y,x]/max(bg[y,x],1e-30)
        if a>=amp_min and p>=1.5: pk.append((round(float(a),2),round(float(wTb[y]),2),round(float(wtb[x]),2)))
    pk.sort(reverse=True); out=[]
    for q in pk:
        if all(abs(q[1]-r[1])>0.09 or abs(q[2]-r[2])>0.09 for r in out): out.append(q)
    return out[:n]
A="recip_Adark"; B="EXPERIMENTAL_FINAL/fin_B_gs1"; beta=37.1
print("="*76); print("STRICTLY RECIPROCAL ATLAS  (drive block == detection block per polarisation)"); print("="*76)
# ---- A same-pol: detect its OWN block  m_x(l5,l7) + d_z(l4 dark, l6)
t,tau,m5=ch(A,"SU3",4); _,_,m7=ch(A,"SU3",6); _,_,m6=ch(A,"SU3",5); _,_,fx=ch(A,"SU2",0)
wtb,wTb,S5=spec(t,tau,m5); _,_,S7=spec(t,tau,m7); _,_,S6=spec(t,tau,m6); _,_,FX=spec(t,tau,fx)
Asame=2.39*S5+0.91*S7+2.57*S6
print("\nA SAME-POL  (Tm m_x + d_z^{23}; d_z^{13} dark):")
print("   ", blind(wtb,wTb,Asame))
print("    with Fe m_x added:", blind(wtb,wTb,Asame+FX))
# ---- A cross: detect the OTHER block  m_z(l2) + SHG
t,tau,l2=ch(A,"SU3",1); _,_,q=ch(A,"SU3",1,quad=True); _,_,fz=ch(A,"SU2",2)
_,_,L2=spec(t,tau,l2); _,_,Q=spec(t,tau,q); _,_,FZ=spec(t,tau,fz)
Across=5.264*L2+beta*Q+FZ
print("\nA CROSS  (m_z = 5.264*l2 + 37.1*l1l2 + F_z):")
print("   ", blind(wtb,wTb,Across))
print(f"    Fe F_z max = {FZ.max():.2e}  (radiatively silent)")
# ---- B cross: detect the OTHER block  Fe m_x + Tm m_x
t,tau,bfx=ch(B,"SU2",0); _,_,b5=ch(B,"SU3",4); _,_,b7=ch(B,"SU3",6)
wtb2,wTb2,BFX=spec(t,tau,bfx); _,_,B5=spec(t,tau,b5); _,_,B7=spec(t,tau,b7)
print("\nB CROSS  (Fe m_x + Tm m_x):")
print("   ", blind(wtb2,wTb2,BFX+2.39*B5+0.91*B7))
print(f"    Tm m_x max = {(2.39*B5+0.91*B7).max():.2e}  (subalgebra: identically zero)")
# ---- B same-pol: detect its OWN block  F_z + m_z(l2) + SHG
t,tau,bfz=ch(B,"SU2",2); _,_,bl2=ch(B,"SU3",1); _,_,bq=ch(B,"SU3",1,quad=True)
_,_,BFZ=spec(t,tau,bfz); _,_,BL2=spec(t,tau,bl2); _,_,BQ=spec(t,tau,bq)
print("\nB SAME-POL  (F_z + 5.264*l2 + 37.1*l1l2):")
print("   ", blind(wtb2,wTb2,BFZ+5.264*BL2+beta*BQ))
