"""The (E||a, H||c) output polarization is shared by geometry-A cross and
geometry-B same-pol.  Mirror parity of the Tm 4c site (m: z->-z):

   magnetic m_z  even  -> allowed 1<->2 only (lambda2).  Diagonal entries are
                          forbidden not by the mirror but by TIME REVERSAL:
                          in a real singlet basis J is imaginary-antisymmetric,
                          so P J P has NO lambda^{1,3,4,6,8} component at all.
   electric  d_x  even  -> allowed between SAME-parity states: 1<->2 (lambda1)
                          AND all three diagonals (lambda3, lambda8).
                          d is T-even and real-symmetric: it lives exactly on
                          the complement lambda^{1,3,4,6,8}.

So the (E||a,H||c) analyser reads  F_z + mu*lambda2  (magnetic)
                                 + d_x*(c1 lambda1 + c3 lambda3 + c8 lambda8)
                                 + beta*lambda1*lambda2  (hyperpolarisability).
The atlas keeps only F_z + mu*lambda2 + beta*l1*l2.  Quantify what is missing.

Contrast (E||c, H||a):  m_x and d_z are both mirror-ODD, so BOTH are purely
off-diagonal (lambda^{4,5,6,7}) -- no population readout.  That channel's
operator is already complete."""
import numpy as np, h5py
exec(open("tfo_project/paper/scripts/atlas_final.py").read().split("A2=[f")[0])

A2=[f"flu0.12_{g}" for g in ["gs1","gs2","gs3"]]
B =[f"EXPERIMENTAL_FINAL/fin_B_{g}"  for g in ["gs1","gs2","gs3"]]

def cen(w,W,A,n=4):
    return ", ".join(f"({y:+.2f},{x:.2f}):{a:.2f}" for a,p,y,x in blind(w,W,A)[:n])

for tag,runs,T in [("A cross",A2,10),("B same-pol",B,0)]:
    wt,wT,l1=mix(runs,"SU3",0,T)
    _ ,_ ,l2=mix(runs,"SU3",1,T)
    _ ,_ ,l3=mix(runs,"SU3",2,T)
    _ ,_ ,l8=mix(runs,"SU3",7,T)
    _ ,_ ,fz=mix(runs,"SU2",2,T)
    print(f"\n=== {tag}:  what radiates into (E||a, H||c) ===")
    print(f"  MAGNETIC  Fe F_z            {fz.max():.4e}")
    print(f"  MAGNETIC  Tm lambda2 (E12)  {l2.max():.4e}   [{cen(wt,wT,l2)}]")
    print(f"  ELECTRIC  Tm lambda1 (E12)  {l1.max():.4e}   [{cen(wt,wT,l1)}]")
    print(f"  ELECTRIC  Tm lambda3 (pop)  {l3.max():.4e}   [{cen(wt,wT,l3)}]")
    print(f"  ELECTRIC  Tm lambda8 (pop)  {l8.max():.4e}   [{cen(wt,wT,l8)}]")
    print(f"  --> electric/magnetic Tm amplitude ratio at equal weight: "
          f"l1/l2={l1.max()/l2.max():.2f}   l3/l2={l3.max()/l2.max():.2f}   l8/l2={l8.max()/l2.max():.2e}")

# and the contrast channel, to show it needs no repair
print("\n=== A same-pol: what radiates into (E||c, H||a) -- mirror-ODD operators ===")
wt,wT,s4=mix(A2,"SU3",3,10,True); _,_,s6=mix(A2,"SU3",5,10,True)
_,_,s3=mix(A2,"SU3",2,10,True); _,_,s8=mix(A2,"SU3",7,10,True)
print(f"  ELECTRIC d_z on lambda4 {s4.max():.3e}   lambda6 {s6.max():.3e}   (both kept)")
print(f"  population lambda3 {s3.max():.3e}   lambda8 {s8.max():.3e}"
      f"   -- FORBIDDEN here: <i|d_z|i>=0 by the mirror, so correctly absent")
