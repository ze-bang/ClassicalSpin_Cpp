import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="tfo_project/paper/data"
census=json.load(open(f"{OUT}/census_final_scenario.json"))["censuses"]
MODES=[("qFM",0.38),("E12",0.50),("E23",0.70),("qAFM",0.90),("E13",1.20)]
fig,axs=plt.subplots(2,2,figsize=(11.5,9))
for row,(tag,ttl) in enumerate([("geomA_cross_lambda2","Geometry A (H||a): Tm lambda2, emits at E12"),
                                ("geomB_cross_Sx","Geometry B (H||c): Fe Sx, emits at qAFM (M1)")]):
    for col,T in enumerate([0,5]):
        z=np.load(f"{OUT}/{tag}_T{T}K.npz")
        ax=axs[row,col]
        A=z["A"].copy(); A[np.abs(z["wT"])<0.1,:]=0; A/=A.max()
        ax.pcolormesh(z["wt"],z["wT"],A,shading="auto",cmap="inferno",vmin=0,vmax=1,rasterized=True)
        for nm,v in MODES:
            ax.axvline(v,color="w",ls="--",lw=0.7,alpha=0.35)
            ax.axhline(v,color="w",ls="--",lw=0.7,alpha=0.35)
            ax.axhline(-v,color="w",ls="--",lw=0.7,alpha=0.35)
            ax.text(v,1.30,nm,ha="center",va="top",color="w",alpha=0.55,fontsize=6)
            ax.text(1.565,v,nm,ha="left",va="center",color="0.35",fontsize=6)
            ax.text(1.565,-v,nm,ha="left",va="center",color="0.35",fontsize=6)
        for a,p,yy,xx in census[tag][f"T={T}K"][:4]:
            ax.plot(xx,yy,"x",color="cyan",ms=9,mew=1.8)
            ax.annotate(f"({yy:+.2f},{xx:.2f})",(xx,yy),textcoords="offset points",xytext=(5,6),color="cyan",fontsize=7)
        ax.set_xlim(0.15,1.55); ax.set_ylim(-1.4,1.4)
        ax.set_xlabel("omega_t (THz)"); ax.set_ylabel("physical omega_T (THz)")
        ax.set_title(f"{ttl}   T={T} K",fontsize=9)
fig.suptitle("Cross-polarized channels, consolidated Hamiltonian (blind census; quasi-elastic band masked)",fontsize=11)
fig.tight_layout(); fig.savefig("tfo_project/paper/figs/crosspol_final.png",dpi=115)
print("saved paper/figs/crosspol_final.png")
