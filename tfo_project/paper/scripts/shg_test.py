import numpy as np, h5py
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
SCALE=2*np.pi/4.135667696
BASE="tfo_project/tmfeo3_2dcs_final"
def channels(run,frame="local"):
    """returns t, tau, M_NL for the linear (l2) and the quadratic (l1*l2) emitters"""
    with h5py.File(f"{BASE}/{run}/sample_0/pump_probe_spectroscopy.h5") as f:
        t=f['/reference/times'][:]; tau=f['/tau_scan/tau_values'][:]
        R=f[f'/reference/M_{frame}_SU3'][:]
        lin0=R[:,1]; quad0=R[:,0]*R[:,1]
        L=np.zeros((len(tau),len(t))); Q=np.zeros((len(tau),len(t)))
        for i in range(len(tau)):
            g=f[f'/tau_scan/tau_{i}']
            a=g[f'M01_{frame}_SU3'][:]; b=g[f'M1_{frame}_SU3'][:]
            L[i]=a[:,1]-b[:,1]-lin0
            Q[i]=a[:,0]*a[:,1] - b[:,0]*b[:,1] - quad0
    return t,tau,L,Q
def spec(t,tau,M):
    tm=t>=3.0
    at=np.hanning(2*tm.sum())[tm.sum():]; atau=np.hanning(2*len(tau))[:len(tau)]
    Md=(M[:,tm]-M[:,tm].mean(axis=0,keepdims=True))*atau[:,None]*at[None,:]
    wt=np.fft.fftshift(np.fft.fftfreq(tm.sum(),t[1]-t[0]))*SCALE
    wta=np.fft.fftshift(np.fft.fftfreq(len(tau),tau[1]-tau[0]))*SCALE
    mx=(wt>0.12)&(wt<1.8); my=(np.abs(wta)<1.6)
    return wt[mx],-wta[my],gaussian_filter(np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my,mx)],sigma=(2,1.5))
def blind(wtb,wTb,A,amp_min=0.07,n=7):
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
t,tau,L,Q=channels("flu0.12_gs1")
wtb,wTb,AL=spec(t,tau,L); _,_,AQ=spec(t,tau,Q)
gl=lambda A,y,x: A[np.argmin(np.abs(wTb-y))][np.argmin(np.abs(wtb-x))]
print("LINEAR emitter  m_z ~ lambda2      census:", blind(wtb,wTb,AL))
print("QUADRATIC emitter  ~ lambda1*lambda2 census:", blind(wtb,wTb,AQ))
print(f"\nquadratic channel, (E12, 2E12)=(0.50,1.00): {gl(AQ,0.49,1.00):.3e}   its own max {AQ.max():.3e}")
print(f"   relative to its (E12,E12) diagonal: {gl(AQ,0.49,1.00)/gl(AQ,0.49,0.50):.2f}")
print(f"\nlinear channel main (qAFM,E12) = {gl(AL,0.90,0.50):.3e}")
print(f"=> beta needed for the SHG peak to match the main peak: {gl(AL,0.90,0.50)/gl(AQ,0.49,1.00):.3f}")

print("\n#### composite m_z = 5.264*lambda2 + beta*lambda1*lambda2, beta set by observed parity ####")
beta=5.264*gl(AL,0.90,0.50)/gl(AQ,0.49,1.00)
print(f"beta = {beta:.1f}  (i.e. {beta/5.264:.1f}x the linear moment)")
for frame in ["local","global"]:
    t,tau,L,Q=channels("flu0.12_gs1",frame)
    wtb,wTb,AL2=spec(t,tau,L); _,_,AQ2=spec(t,tau,Q)
    C=5.264*AL2+beta*AQ2
    print(f"\n  frame={frame}: census =", blind(wtb,wTb,C,0.06,8))
