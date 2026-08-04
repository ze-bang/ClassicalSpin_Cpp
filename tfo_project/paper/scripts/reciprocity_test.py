exec(open('/tmp/shg.py').read().split('t,tau,L,Q=channels')[0])
import numpy as np, h5py
def ch(run,sp,l):
    with h5py.File(f"{BASE}/{run}/sample_0/pump_probe_spectroscopy.h5") as f:
        t=f['/reference/times'][:]; tau=f['/tau_scan/tau_values'][:]
        M0=f[f'/reference/M_global_{sp}'][:,l]
        M=np.zeros((len(tau),len(t)))
        for i in range(len(tau)):
            g=f[f'/tau_scan/tau_{i}']
            M[i]=g[f'M01_global_{sp}'][:,l]-g[f'M1_global_{sp}'][:,l]-M0
    return t,tau,M
for run,lab in [("flu0.12_gs1","current drive (lambda2 3.0 + lambda7)"),
                ("recip_A","RECIPROCAL: d_z on l4,l6 + m_x on l5,l7"),
                ("recip_Adark","RECIPROCAL, E13 electric leg removed")]:
    print(f"\n########## {lab} ##########")
    t,tau,m4=ch(run,"SU3",3); _,_,m6=ch(run,"SU3",5)
    wtb,wTb,A4=spec(t,tau,m4); _,_,A6=spec(t,tau,m6)
    S=0.006*A4+4.4*A6
    print("  SAME-POL (d_z/m_x block: 0.006*l4+4.4*l6):", blind(wtb,wTb,S,0.07,6))
    t,tau,m2=ch(run,"SU3",1)
    _,_,A2=spec(t,tau,m2)
    print("  CROSS   (m_z = lambda2)                 :", blind(wtb,wTb,A2,0.07,6))
    print(f"     abs: same-pol max={S.max():.3e}   cross max={A2.max():.3e}   ratio={A2.max()/S.max():.3f}")
