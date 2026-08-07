import numpy as np, sys
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
HBAR=1.0   # code units: hbar=1, time in meV^-1 (census SCALE converts to THz)
SCALE=2*np.pi/4.135667696
# Gell-Mann
L=np.zeros((8,3,3),complex)
L[0][0,1]=L[0][1,0]=1
L[1][0,1]=-1j; L[1][1,0]=1j
L[2][0,0]=1; L[2][1,1]=-1
L[3][0,2]=L[3][2,0]=1
L[4][0,2]=-1j; L[4][2,0]=1j
L[5][1,2]=L[5][2,1]=1
L[6][1,2]=-1j; L[6][2,1]=1j
L[7]=np.diag([1,1,-2])/np.sqrt(3)
def liou(H):     # Omega_ab = Tr(L_a * (-i/hbar)[H, L_b]) / 2
    O=np.zeros((8,8))
    for b in range(8):
        C=(-1j/HBAR)*(H@L[b]-L[b]@H)
        for a in range(8): O[a,b]=np.real(np.trace(L[a]@C))/2
    return O
e1,e2=2.067834,4.9628
if "--e2" in sys.argv: e2=float(sys.argv[sys.argv.index("--e2")+1])
H0=np.diag([0.0,e1,e2])
v=np.array([0,3.0,0,0,2.3915,0,0.9128,0])
if "--v" in sys.argv: v=np.array([float(x) for x in sys.argv[sys.argv.index("--v")+1].split(",")])
amp=0.138027
if "--amp" in sys.argv: amp=float(sys.argv[sys.argv.index("--amp")+1])
V=sum(v[a]/np.linalg.norm(v)*L[a] for a in range(8))
O0=liou(H0); O1=liou(V)*amp
gam=np.array([0.157,0.157,0.0,0.207,0.207,0.414,0.414,0.0])
lam0=np.array([0,0,1,0,0,0,0,1/np.sqrt(3)])
# pulse table, centered at its |f| max
tab=np.loadtxt("tfo_project/experimental_pulse_codeunits.dat")
tt,ff=tab[:,0],tab[:,1]; tt=tt-tt[np.argmax(np.abs(ff))]
pol=1.0
if "--flip" in sys.argv: pol=-1.0
def pulse(t,t0):   # vectorized over t scalar; returns scalar envelope
    return pol*np.interp(t-t0,tt,ff,left=0.0,right=0.0)
dt=0.02; t0,t1=-130.0,40.0
ts=np.arange(t0,t1+dt/2,dt); nt=len(ts)
taus=np.arange(-120.0,0.0+1e-9,0.2)   # 0.2 ps step (vs 0.1 in sim) for speed; Nyquist 2.5 THz ok
ntau=len(taus)
def run(tau_arr,probe):
    # batched: state (nb,8); drive f = sum of pulses
    nb=len(tau_arr)
    lam=np.tile(lam0,(nb,1))
    out4=np.zeros((nb,nt)); out6=np.zeros((nb,nt))
    def rhs(lam,t):
        f1=pol*np.interp(t-tau_arr,tt,ff,left=0.0,right=0.0) if tau_arr is not None else 0.0
        f2=pulse(t,0.0) if probe else 0.0
        d=lam@O0.T
        if np.ndim(f1)>0: d+= (f1[:,None]*(lam@O1.T))
        elif f1: d+= f1*(lam@O1.T)
        if f2: d+= f2*(lam@O1.T)
        d-= gam[None,:]*lam
        return d
    for i,t in enumerate(ts):
        out4[:,i]=lam[:,3]; out6[:,i]=lam[:,5]
        k1=rhs(lam,t); k2=rhs(lam+dt/2*k1,t+dt/2); k3=rhs(lam+dt/2*k2,t+dt/2); k4=rhs(lam+dt*k3,t+dt)
        lam=lam+dt/6*(k1+2*k2+2*k3+k4)
    return out4,out6
# M01: both pulses (batch over tau); M1: pump only (batch over tau); M0: probe only (single)
M01_4,M01_6=run(taus,True)
M1_4, M1_6 =run(taus,False)
M0_4,M0_6  =run(np.array([500.0]),True)   # pump far away => probe-only reference
M0_4,M0_6=M0_4[0],M0_6[0]
Ml4=M01_4-M1_4-M0_4[None,:]; Ml6=M01_6-M1_6-M0_6[None,:]
# analysis identical to census pipeline
t=ts; tau=taus
tm=t>=3.0; tk=t[tm]
apod=np.exp(np.log(0.03)*((tk-tk[0])/(tk[-1]-tk[0]))**2)
w=np.ones_like(tau); m1=tau>-6; w[m1]=0.5*(1-np.cos(np.pi*(-tau[m1])/6))
m2=tau<-110; w[m2]=0.5*(1-np.cos(np.pi*(120+tau[m2])/10))
wt=np.fft.fftshift(np.fft.fftfreq(tm.sum(),dt))*SCALE
wta=np.fft.fftshift(np.fft.fftfreq(ntau,tau[1]-tau[0]))*SCALE
mx=(wt>0.12)&(wt<1.6); my=(np.abs(wta)<1.6)
wtb=wt[mx]; wTb=-wta[my]
dyx=abs(wta[1]-wta[0]); dxx=abs(wt[1]-wt[0])
def spec(M):
    Md=(M[:,tm]-M[:,tm].mean())*w[:,None]*apod[None,:]
    return gaussian_filter(np.abs(np.fft.fftshift(np.fft.fft2(Md)))[np.ix_(my,mx)],sigma=(2,1.5))
def blind(A,amp_min=0.05,prom_min=1.5,nmax=12):
    n=A.max(); fp=(int(0.10/dyx)|1,int(0.10/dxx)|1)
    ismax=(A==maximum_filter(A,size=fp))
    bg=median_filter(A,size=(int(0.34/dyx)|1,int(0.34/dxx)|1))
    ys,xs=np.where(ismax); pk=[]
    for y,x in zip(ys,xs):
        a=A[y,x]/n; p=A[y,x]/max(bg[y,x],1e-30)
        if a>=amp_min and p>=prom_min: pk.append((a,p,wTb[y],wtb[x]))
    pk.sort(reverse=True); out=[]
    for q in pk:
        if all(abs(q[2]-r[2])>0.09 or abs(q[3]-r[3])>0.09 for r in out): out.append(q)
    return out[:nmax]
w4=1.0
if "--w4" in sys.argv: w4=float(sys.argv[sys.argv.index("--w4")+1])
A=w4*spec(Ml4)+4.4*spec(Ml6)
tag=" ".join(sys.argv[1:]) or "baseline"
print(f"TOY QUTRIT [{tag}] — composite {w4}*A4+4.4A6, blind census:")
for a,p,yy,xx in blind(A): print(f"   {a:5.2f} x{p:4.1f}  ({yy:+.2f}, {xx:.2f})")
