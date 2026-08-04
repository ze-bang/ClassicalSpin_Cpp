exec(open('/tmp/target_fit.py').read().split('cef=0.006*A4')[0])
import numpy as np
from scipy.optimize import minimize
# detection operator for the same-pol channel, three physical contributions:
#   CEF   : w4*lambda4 (E13 emission) + 4.4*lambda6 (E23 emission)
#   Fe    : c_Fe * F_x                (qFM/qAFM emission)
#   d_eff : c_E12 * lambda2           (E12 emission = the reciprocity partner)
targets=[("(E12, E23)",   0.49,0.70,1.00),
         ("(E12, qFM)",   0.49,0.38,0.45),
         ("(E12, qAFM)",  0.49,0.90,0.40),
         ("(qAFM, E12)",  0.90,0.50,0.30),
         ("(qAFM, E13)",  0.90,1.20,0.30),
         ("(E12, E13)",   0.49,1.20,0.15)]
def build(p):
    w4,cFe,cE12=np.exp(p)
    return w4*A4+4.4*A6+cFe*FX+cE12*A2
def cost(p):
    T=build(p); r=g(T,0.49,0.70)
    if r<=0: return 1e9
    return sum((g(T,y,x)/r-e)**2 for _,y,x,e in targets)
best=None
for s in [(-5,-7,-8),(-6,-6,-9),(-4,-8,-7)]:
    res=minimize(cost,np.array(s),method="Nelder-Mead",options={"maxiter":4000,"xatol":1e-4,"fatol":1e-10})
    if best is None or res.fun<best.fun: best=res
w4,cFe,cE12=np.exp(best.x)
T=build(best.x); r=g(T,0.49,0.70)
print("FITTED same-pol detection weights (three physical contributions):")
print(f"   CEF   : {w4:.4g}*lambda4 + 4.4*lambda6      -> mu13/mu23 admixture = {w4/4.4*100:.3f}%")
print(f"   Fe    : {cFe:.4g}*F_x")
print(f"   d_eff : {cE12:.4g}*lambda2   (E12 emission, the reciprocity partner)")
print(f"   residual = {best.fun:.4f}\n")
print(f"{'peak':>16s} {'experiment':>11s} {'model':>7s} {'ratio':>7s}")
for nm,y,x,e in targets:
    m=g(T,y,x)/r
    print(f"{nm:>16s} {e:11.2f} {m:7.2f} {m/e:7.2f}")
print("\n(the rectified omega_t~0 line is excluded: our detection window t>=3 with")
print(" Gaussian apodisation suppresses it by construction; it is analysed via S_DC)")
