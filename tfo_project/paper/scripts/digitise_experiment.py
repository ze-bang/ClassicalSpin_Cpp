import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
d=json.load(open("tfo_project/paper/data/experiment_geomA_samepol_digitized.json"))
# colour scale of the source figure: white -> tan -> red -> dark blue (darker = larger)
cmap=LinearSegmentedColormap.from_list("exp",["#ffffff","#f6e6d8","#e8a07a","#d6453a","#3b2f8f","#141043"])
fig,axs=plt.subplots(1,2,figsize=(11.6,6.4))
X,Y=np.meshgrid(np.linspace(0,2,500),np.linspace(-2,2,800))
Z=np.zeros_like(X)
for f in d["all_features"]:
    Z+=f["amp"]*np.exp(-((X-f["detection"])**2/(2*0.075**2)+(Y-f["excitation"])**2/(2*0.075**2)))
for ax in axs:
    ax.pcolormesh(X,Y,Z,cmap=cmap,vmin=0,vmax=Z.max(),shading="auto",rasterized=True)
    for p in d["marked_peaks"]:
        ax.plot(p["detection"],p["excitation"],"+",color="k",ms=11,mew=1.6)
    ax.set_xlabel("Detection frequency (THz)"); ax.set_ylabel("Excitation frequency (THz)")
    ax.set_xlim(0,2); ax.set_ylim(-2,2)
axs[0].set_title("digitisation as an INTENSITY map\n(white $\\to$ tan $\\to$ red $\\to$ blue = increasing)",fontsize=10)
ax=axs[1]
for nm,v,c in [("qFM 0.38",0.38,"C2"),("E12 0.50",0.50,"C0"),("E23 0.70",0.70,"C4"),
               ("qAFM 0.90",0.90,"C3"),("E13 1.20",1.20,"C1")]:
    ax.axvline(v,color=c,ls="--",lw=1.0,alpha=0.9)
    ax.axhline(v,color=c,ls=":",lw=1.0,alpha=0.7); ax.axhline(-v,color=c,ls=":",lw=1.0,alpha=0.7)
    ax.text(v,2.03,nm,rotation=90,fontsize=7,color=c,va="bottom",ha="center")
    ax.text(2.03,v,nm,fontsize=7,color=c,va="center")
ax.set_title("with mode energies overlaid\n(dashed = detection, dotted = excitation)",fontsize=10)
fig.suptitle("Digitisation of the geometry-A same-polarised experimental map (intensity, darker = stronger)",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/experiment_digitised.png",dpi=120)
print("saved figs/experiment_digitised.png\n")
print("PEAK LIST in our convention (omega_tau, omega_t), normalised:")
for f in sorted(d["all_features"],key=lambda z:-z["amp"]):
    tag=" <-- MARKED" if "MARKED" in f.get("note","") else ""
    print(f"   {f['amp']:4.2f}   ({f['excitation']:+.2f}, {f['detection']:.2f}){tag}")
