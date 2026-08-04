exec(open('/tmp/recip_atlas.py').read().split('A="recip_Adark"')[0])
import numpy as np
beta=37.1
print("FULLY RECIPROCAL: drive (0,d_eff,0,0,2.3915,0,0.9128,0); detection uses the SAME coefficients")
print("  same-pol = 2.3915*l5 + 0.9128*l7 + d_eff*l2     cross = 5.264*l2 + 37.1*l1l2")
print("="*104)
for run,D in [("full10",1.0),("full30",3.0)]:
    t,tau,m5=ch(run,"SU3",4); _,_,m7=ch(run,"SU3",6); _,_,l2=ch(run,"SU3",1)
    _,_,q=ch(run,"SU3",1,quad=True); _,_,fx=ch(run,"SU2",0)
    wtb,wTb,S5=spec(t,tau,m5); _,_,S7=spec(t,tau,m7); _,_,L2=spec(t,tau,l2)
    _,_,Q=spec(t,tau,q); _,_,FX=spec(t,tau,fx)
    same=2.3915*S5+0.9128*S7 + D*L2
    cross=5.264*L2+beta*Q
    g=lambda A,y,x: A[np.argmin(np.abs(wTb-y))][np.argmin(np.abs(wtb-x))]
    print(f"\n---- d_eff = {D} ----")
    print(f" SAME-POL: {blind(wtb,wTb,same,0.10,6)}")
    n=g(same,0.49,0.70)
    print(f"    normalised to (E12,E23): (E12,E13)={g(same,0.49,1.20)/n:5.2f}  "
          f"(qAFM,E13)={g(same,0.90,1.20)/n:5.2f}  (E13 delay row)={g(same,1.20,0.70)/n:5.2f}  "
          f"leak(.,E12)={g(same,0.49,0.50)/n:5.2f}")
    print(f" CROSS   : {blind(wtb,wTb,cross,0.10,5)}")
    m=g(cross,0.90,0.50)
    print(f"    normalised to (qAFM,E12): (E12,2E12)={g(cross,0.49,1.00)/m:5.2f}  "
          f"(E13 delay row)={g(cross,1.20,0.50)/m:5.2f}")
