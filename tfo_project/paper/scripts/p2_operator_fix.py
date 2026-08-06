"""Corrected (E||a,H||c) detection operator, applied to BOTH channels that
share that output polarization (geometry-A cross, geometry-B same-pol):

    O_2 = F_z + mu*lambda2                       magnetic m_z  (mu = 5.264)
        + w_E1*( lambda1 + c3*lambda3 )          electric d_x  (mirror-even)
        + beta*lambda1*lambda2                   hyperpolarisability

Mirror parity: on the Tm 4c site (m: z->-z) both m_z and d_x are EVEN, so the
polarization they share reads the 1<->2 block AND, through d_x only, the
diagonal (population) entries lambda3, lambda8.  In the orthogonal
polarization m_x and d_z are both mirror-ODD, hence purely off-diagonal --
which is why the same-polarised geometry-A operator needs no population term.

beta is re-fitted at each w_E1 from the OBSERVED A-cross parity
(E12,2E12) = (qAFM,E12).  The A-cross E12 DIAGONAL is the calibration
observable: the fluence scan pinned it at 0.16 of the main cross peak."""
import numpy as np, h5py, json
exec(open("tfo_project/paper/scripts/atlas_final.py").read().split("A2=[f")[0])

A2=[f"flu0.12_{g}" for g in ["gs1","gs2","gs3"]]
B =[f"EXPERIMENTAL_FINAL/fin_B_{g}"  for g in ["gs1","gs2","gs3"]]
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
def pack(runs,T):
    wt,wT,l1=mix(runs,"SU3",0,T); _,_,l2=mix(runs,"SU3",1,T)
    _,_,l3=mix(runs,"SU3",2,T);  _,_,fz=mix(runs,"SU2",2,T); _,_,q=quad(runs,T)
    return wt,wT,l1,l2,l3,fz,q
PA=pack(A2,10); PB=pack(B,0)
def at(w,W,A,y,x): return A[np.argmin(np.abs(W-y))][np.argmin(np.abs(w-x))]

print("               ---------------- A cross (ANCHOR) ----------------   -- B same-pol --")
print(" w_E1  E1/M1  beta   (qAFM,E12) (E12,2E12) (-qAFM,E12) (E12,E12)    (E12,2E12) (E12,~0)")
print("                                                        exp 0.16")
rows=[]
for w1 in [0.0,0.5,1.0,2.0,2.5,5.264,10.0,26.3]:
    for c3 in ([0.0,1.0] if w1 else [0.0]):
        wt,wT,l1,l2,l3,fz,q=PA
        lin=fz+5.264*l2+w1*(l1+c3*l3)
        beta=at(wt,wT,lin,0.90,0.50)/at(wt,wT,q,0.49,1.00)
        A_=lin+beta*q; n=at(wt,wT,A_,0.90,0.50)
        wt2,wT2,b1,b2,b3,bfz,bq=PB
        B_=bfz+5.264*b2+w1*(b1+c3*b3)+beta*bq; nb=B_.max()
        print(f"{w1:5.2f} {w1/5.264:5.2f} {beta:6.1f}   "
              f"{at(wt,wT,A_,0.90,0.50)/n:9.2f} {at(wt,wT,A_,0.49,1.00)/n:10.2f} "
              f"{at(wt,wT,A_,-0.90,0.50)/n:11.2f} {at(wt,wT,A_,0.49,0.50)/n:9.2f}    "
              f"{at(wt2,wT2,B_,0.49,1.00)/nb:9.2f} {at(wt2,wT2,B_,0.49,0.13)/nb:8.2f}"
              f"   c3={c3}")
