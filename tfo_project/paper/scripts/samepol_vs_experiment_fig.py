exec(open('/tmp/target_fit.py').read().split('cef=0.006*A4')[0])
import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list("exp",["#ffffff","#f6e6d8","#e8a07a","#d6453a","#3b2f8f","#141043"])
w4,cFe,cE12=4.965e-19,8.205e-05,1.488e-04
T=w4*A4+4.4*A6+cFe*FX+cE12*A2
d=json.load(open("tfo_project/paper/data/experiment_geomA_samepol_digitized.json"))
fig,axs=plt.subplots(1,2,figsize=(12.4,6.2))
# experiment (digitised), plotted in OUR orientation: x = omega_t, y = omega_T
ax=axs[0]
X,Y=np.meshgrid(np.linspace(0,1.6,400),np.linspace(-1.4,1.4,600))
Z=np.zeros_like(X)
SH=0.50/0.38   # collaborator: the dominant row is E12, so rescale the digitised excitation axis
for f in d["all_features"]:
    ex=f["excitation"]*SH if abs(abs(f["excitation"])-0.38)<0.09 else f["excitation"]
    Z+=f["amp"]*np.exp(-((X-f["detection"])**2/(2*0.075**2)+(Y-ex)**2/(2*0.075**2)))
ax.pcolormesh(X,Y,Z,cmap=cmap,vmin=0,vmax=Z.max(),shading="auto",rasterized=True)
ax.set_title("EXPERIMENT (digitised; dominant row assigned to $E_{12}$)",fontsize=10)
# model with the fitted detection operator
ax2=axs[1]
sel=(wtb<1.6)
ax2.pcolormesh(wtb[sel],wTb,T[:,sel]/T[:,sel].max(),cmap=cmap,vmin=0,vmax=1,shading="auto",rasterized=True)
ax2.set_title("MODEL, same-pol detection fitted:\n"
              "$4.4\\lambda^6$ + %.1e$\\,F_x$ + %.1e$\\,\\lambda^2$"%(cFe,cE12),fontsize=10)
for ax in (axs[0],ax2):
    for nm,v,c in [("qFM",0.38),("E12",0.50),("E23",0.70),("qAFM",0.90),("E13",1.20)][:0]: pass
    for nm,v in [("qFM",0.38),("E12",0.50),("E23",0.70),("qAFM",0.90),("E13",1.20)]:
        ax.axvline(v,color="0.5",ls="--",lw=0.6,alpha=0.7)
        ax.axhline(v,color="0.5",ls=":",lw=0.6,alpha=0.5); ax.axhline(-v,color="0.5",ls=":",lw=0.6,alpha=0.5)
        ax.text(v,1.42,nm,ha="center",va="bottom",fontsize=6,color="0.35")
    for nm,y,x in [("",0.90,0.50),("",0.90,1.20)]:
        ax.plot(x,y,"+",color="k",ms=10,mew=1.5)
    ax.set_xlim(0,1.6); ax.set_ylim(-1.4,1.4)
    ax.set_xlabel("$\\omega_t$ (THz)"); ax.set_ylabel("physical $\\omega_T$ (THz)")
fig.suptitle("Geometry A same-polarised: digitised experiment vs the three-level model with a fitted detection operator\n"
             "(cross channels held fixed as anchors; + marks the two designated peaks)",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/samepol_vs_experiment.png",dpi=120)
print("saved figs/samepol_vs_experiment.png")
