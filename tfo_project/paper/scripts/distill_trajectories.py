import numpy as np, h5py, os
BASE="tfo_project/tmfeo3_2dcs_final"
OUT="tfo_project/paper/data/trajectories"; os.makedirs(OUT,exist_ok=True)
DEC=10   # t decimation: dt 0.02 -> 0.2 code units, Nyquist 3.8 THz (all content < 2.5 THz)
# channels per geometry: (dataset kind, species, component index, short name)
CH={
 "fin_A2":[("global","SU2",0,"Sx"),("global","SU2",1,"Sy"),("global","SU2",2,"Sz"),
          ("global","SU3",1,"l2"),("global","SU3",4,"l5"),("global","SU3",6,"l7"),("antiferro","SU2",1,"stagSy")],
 "fin_B":[("global","SU2",0,"Sx"),("global","SU2",1,"Sy"),("global","SU2",2,"Sz"),
          ("antiferro","SU2",1,"stagSy"),("global","SU3",1,"l2")],
 "fin_S":[("global","SU3",3,"l4"),("global","SU3",5,"l6"),("global","SU2",0,"Sx")],
}
LOC={"fin_A2":"EXPERIMENTAL_FINAL/","fin_B":"EXPERIMENTAL_FINAL/","fin_S":""}
for geom,chans in CH.items():
    for g in ["gs1","gs2","gs3"]:
        run=f"{LOC[geom]}{geom}_{g}"
        d=f"{BASE}/{run}/sample_0"
        out={}
        with h5py.File(f"{d}/pump_probe_spectroscopy.h5") as f:
            t=f['/reference/times'][:]; tau=f['/tau_scan/tau_values'][:]
            out["t"]=t[::DEC].astype(np.float32); out["tau"]=tau.astype(np.float32)
            # reference (single-pulse) trajectories, all requested channels
            for kind,sp,l,nm in chans:
                out[f"M0_{nm}"]=f[f'/reference/M_{kind}_{sp}'][::DEC,l].astype(np.float32)
            # full M_NL(tau,t) per channel
            for kind,sp,l,nm in chans:
                M0=f[f'/reference/M_{kind}_{sp}'][:,l]
                M=np.zeros((len(tau),len(t[::DEC])),dtype=np.float32)
                for i in range(len(tau)):
                    gr=f[f'/tau_scan/tau_{i}']
                    m=gr[f'M01_{kind}_{sp}'][:,l]-gr[f'M1_{kind}_{sp}'][:,l]-M0
                    M[i]=m[::DEC].astype(np.float32)
                out[f"MNL_{nm}"]=M
        fn=f"{OUT}/{geom}_{g}.npz"
        np.savez_compressed(fn,**out)
        print(f"{fn}: {os.path.getsize(fn)/1e6:.1f} MB, channels: {[c[3] for c in chans]}")
with open(f"{OUT}/README.md","w") as f:
    f.write("""# Full time-domain trajectories of the final scenario

One npz per run (`fin_A|fin_B|fin_S` x Boltzmann seeds `gs1|gs2|gs3`;
exact .param files in ../params/). Contents:

- `t`      : detection-time axis, code units (1 unit = 0.6582 ps),
             decimated x10 from the integration grid (dt = 0.2 units,
             Nyquist 3.8 THz — above all physical content).
- `tau`    : pump-probe delay axis (negative = pump first), full resolution.
- `M0_<ch>`: single-pulse (probe-only) reference trajectory, channel <ch>.
- `MNL_<ch>`: full nonlinear signal M_NL(tau, t) = M01 - M1 - M0,
             shape (len(tau), len(t)), float32.

Channels: Sx/Sy/Sz = uniform Fe magnetization; stagSy = staggered Fe S_y
(qAFM coordinate); l1..l6 = Tm Gell-Mann components (global frame).
Frequency convention: f[THz] = (cycles per code unit) x 1.5192.

Any spectrum in the paper regenerates from these via FFT with the window
recipes in ../scripts/ (verify_final_picture.py); raw 8.6 GB/run h5 stay
on the lab machine (regenerable bit-exact from ../params/).
""")
print("wrote README.md")
