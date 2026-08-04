import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open("tfo_project/paper/data/experiment_geomA_samepol_digitized.json"))
fig,axs=plt.subplots(1,2,figsize=(11.5,6.4))
X,Y=np.meshgrid(np.linspace(0,2,400),np.linspace(-2,2,700))
Z=np.zeros_like(X)
for f in d["all_features"]:
    s=1.0 if f["sign"]=="+" else -1.0
    Z+=s*f["amp"]*np.exp(-((X-f["detection"])**2/(2*0.07**2)+(Y-f["excitation"])**2/(2*0.07**2)))
m=np.abs(Z).max()
for k,ax in enumerate(axs):
    ax.pcolormesh(X,Y,Z,cmap="RdBu_r",vmin=-m,vmax=m,shading="auto",rasterized=True)
    for p in d["marked_peaks"]:
        ax.plot(p["detection"],p["excitation"],"+",color="k",ms=11,mew=1.6)
    ax.set_xlabel("Detection frequency (THz)"); ax.set_ylabel("Excitation frequency (THz)")
    ax.set_xlim(0,2); ax.set_ylim(-2,2)
axs[0].set_title("my digitisation of the supplied map\n(same axes and orientation)",fontsize=10)
ax=axs[1]
for nm,v,c in [("qFM 0.38",0.38,"C2"),("E12 0.50",0.50,"C0"),("E23 0.70",0.70,"C4"),
               ("qAFM 0.90",0.90,"C3"),("E13 1.20",1.20,"C1")]:
    ax.axvline(v,color=c,ls="--",lw=1.0,alpha=0.85)
    ax.axhline(v,color=c,ls=":",lw=0.9,alpha=0.6); ax.axhline(-v,color=c,ls=":",lw=0.9,alpha=0.6)
    ax.text(v,2.03,nm,rotation=90,fontsize=7,color=c,va="bottom",ha="center")
    ax.text(2.03,v,nm,fontsize=7,color=c,va="center")
ax.set_title("with the five mode energies overlaid\n(dashed = detection axis, dotted = excitation axis)",fontsize=10)
fig.suptitle("Digitisation of the supplied geometry-A same-polarised experimental map — please check against the original",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/experiment_digitised.png",dpi=120)
print("saved figs/experiment_digitised.png")
print("\nMARKED peaks, in our convention (omega_tau, omega_t):")
for p in d["marked_peaks"]:
    print(f"   ({p['excitation']:.2f}, {p['detection']:.2f})  = {p['assignment']}")
rows=sorted({round(f["excitation"],2) for f in d["all_features"] if f["amp"]>=1.5})
print("\nDominant unmarked structure sits on excitation rows:",rows)
print("   -> qFM (0.38) and 0, NOT E12 (0.50)")
