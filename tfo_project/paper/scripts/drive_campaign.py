exec(open('/tmp/recip_atlas.py').read().split('A="recip_Adark"')[0])
import numpy as np
beta=37.1
def report(run,geom,lab):
    t,tau,l2=ch(run,"SU3",1); _,_,q=ch(run,"SU3",1,quad=True)
    _,_,l4=ch(run,"SU3",3); _,_,l6=ch(run,"SU3",5); _,_,l5=ch(run,"SU3",4); _,_,l7=ch(run,"SU3",6)
    _,_,fx=ch(run,"SU2",0); _,_,fz=ch(run,"SU2",2)
    wtb,wTb,L2=spec(t,tau,l2); _,_,Q=spec(t,tau,q)
    _,_,L4=spec(t,tau,l4); _,_,L6=spec(t,tau,l6); _,_,L5=spec(t,tau,l5); _,_,L7=spec(t,tau,l7)
    _,_,FX=spec(t,tau,fx); _,_,FZ=spec(t,tau,fz)
    mz=5.264*L2+beta*Q; mx_tm=2.39*L5+0.91*L7; cef=0.006*L4+4.4*L6
    print(f"\n===== {lab}  ({geom}) =====")
    if geom=="A":
        print(f"  A-cross  (m_z):  {blind(wtb,wTb,mz+FZ,0.10,5)}")
        print(f"  A-same   (Tm CEF + Fe m_x): {blind(wtb,wTb,cef+FX,0.10,5)}")
        print(f"           (Tm CEF only)    : {blind(wtb,wTb,cef,0.10,5)}")
        print(f"     abs: l2={L2.max():.2e}  l4={L4.max():.2e}  l6={L6.max():.2e}  Fe m_x={FX.max():.2e}")
    else:
        print(f"  B-cross  (Fe m_x + Tm m_x): {blind(wtb,wTb,FX+mx_tm,0.10,5)}")
        print(f"  B-same   (m_z):             {blind(wtb,wTb,FZ+mz,0.10,5)}")
        print(f"     abs: l1..l7 -> l2={L2.max():.2e} l4={L4.max():.2e} l6={L6.max():.2e}  Fe m_x={FX.max():.2e}")
report("flu0.12_gs1","A","REFERENCE  geomA  magnetic drive (l2=3.0, l7)")
report("cA_e","A","CAMPAIGN   geomA  ELECTRIC drive (l4=2.95, l6=2.57)")
report("EXPERIMENTAL_FINAL/fin_B_gs1","B","REFERENCE  geomB  magnetic drive (l2 only)")
report("EXPERIMENTAL_FINAL/cB_e1","B","CAMPAIGN   geomB  ELECTRIC drive (l1 = d_x^12)")
report("EXPERIMENTAL_FINAL/cB_e1m","B","CAMPAIGN   geomB  ELECTRIC + magnetic (l1 + l2)")
