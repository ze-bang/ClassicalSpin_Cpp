import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import odeint
# minimal quantitative model of an incoherent-repopulation-driven buildup:
# magnon x'' + 2G x' + w^2 x = F(t),  F = F0 (1-exp(-t/tr)) exp(-t/T1) theta(t)
# w = qAFM 0.90 THz; G = magnon amplitude decay (measured linewidth 0.003 THz)
PS=1.0  # work directly in ps
w=2*np.pi*0.90     # rad/ps
G=np.pi*0.003      # amplitude decay rate from measured linewidth
T1=100.0
t=np.linspace(-10,250,26000)
fig,axs=plt.subplots(1,2,figsize=(12.6,4.7))
ax=axs[0]
for tr,c in [(5,"C0"),(15,"C2"),(40,"C3")]:
    def F(tt): return np.where(tt>0,(1-np.exp(-tt/tr))*np.exp(-tt/T1),0.0)
    def rhs(y,tt): return [y[1],F(tt)-2*G*y[1]-w*w*y[0]]
    sol=odeint(rhs,[0,0],t)
    x=sol[:,0]
    ax.plot(t,x/np.abs(x).max(),lw=1.4,color=c,label=f"$\\tau_{{\\rm rise}}={tr}$ ps")
ax.axvspan(-5,5,color="gold",alpha=0.2,lw=0)
ax.set_xlabel("t (ps)"); ax.set_ylabel("magnon coordinate (norm.)")
ax.set_title("Incoherent repopulation drive: signal GROWS on $\\tau_{\\rm rise}$, decays on $T_1$\n"
             "$F(t)\\propto\\Delta n(t)=(1-e^{-t/\\tau_{\\rm rise}})e^{-t/T_1}$",fontsize=9.5)
ax.legend(fontsize=8); ax.set_xlim(-10,250)
ax=axs[1]
for tr,c in [(5,"C0"),(15,"C2"),(40,"C3")]:
    def F(tt): return np.where(tt>0,(1-np.exp(-tt/tr))*np.exp(-tt/T1),0.0)
    def rhs(y,tt): return [y[1],F(tt)-2*G*y[1]-w*w*y[0]]
    sol=odeint(rhs,[0,0],t)
    x=np.abs(sol[:,0]);
    from scipy.ndimage import maximum_filter1d
    env=maximum_filter1d(x,int(2.5/(t[1]-t[0])))
    ax.plot(t,env/env.max(),lw=1.6,color=c,label=f"$\\tau_{{\\rm rise}}={tr}$ ps")
# coherent-model comparison: 8 ps decay from pulse end
ax.plot(t,np.where(t>0,np.exp(-t/8.0),np.exp(t/2)*0+np.nan),ls="--",color="0.4",lw=1.6,
        label="coherent model (all vertices):\ndecays with $T_2(E_{12})=8$ ps")
ax.axvspan(-5,5,color="gold",alpha=0.2,lw=0)
ax.set_xlabel("t (ps)"); ax.set_ylabel("signal envelope (norm.)")
ax.set_title("The buildup discriminator: envelope growth time $=\\tau_{\\rm rise}$ (measurable),\n"
             "persistence $=\\min(T_1,$ magnon lifetime$)$",fontsize=9.5)
ax.legend(fontsize=7.5); ax.set_xlim(-10,250)
fig.suptitle("Geometry B post-pulse buildup requires an incoherent population ramp "
             "(minimal quantitative model; coherent model shown dashed)",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/buildup_model.png",dpi=115)
print("saved paper/figs/buildup_model.png")
