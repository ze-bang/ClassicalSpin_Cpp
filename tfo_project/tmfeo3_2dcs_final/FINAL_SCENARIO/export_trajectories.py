# Export the FINAL SCENARIO magnetization trajectories in three domains:
#   /time_domain   M_NL(tau, t)           — all components, t decimated x5
#   /mixed_domain  M_NL(omega_tau, t)     — FFT along tau ONLY (complex), time along t
#   /freq_domain   |M_NL(omega_tau, omega_t)| — standard 2D analysis pipeline
# plus /reference M0(t) single-pulse trajectories at full resolution.
# Readout operators: see READOUT_CHANNELS.md. Run from tmfeo3_2dcs_final/FINAL_SCENARIO.
import numpy as np, h5py, os

SCALE = 2*np.pi/4.135667696       # code angular frequency -> THz
PS    = 0.6582119569              # code time unit -> ps
HERE  = os.path.dirname(os.path.abspath(__file__))
DEC   = 5                         # t decimation for tau-resolved datasets

def export(run, out):
    src = f"{HERE}/runs/{run}/sample_0/pump_probe_spectroscopy.h5"
    with h5py.File(src) as f, h5py.File(out, "w") as g:
        t   = f['/reference/times'][:]
        tau = f['/tau_scan/tau_values'][:]
        nt, ntau = len(t), len(tau)
        td_idx = np.arange(0, nt, DEC)

        g.attrs["scenario"] = "kappaB-FREE final scenario 2026-08-07; see READOUT_CHANNELS.md"
        g.attrs["units"] = ("time axes in code units (1 unit = 0.6582 ps); "
                            "frequency axes in THz; SU2 cols = S_x,S_y,S_z; SU3 cols = lambda1..lambda8; "
                            "quad = sum over Tm sites of local lambda1*lambda2 (hyperpolarizability coordinate)")
        g.attrs["MNL_definition"] = "M_NL = M01 - M1 - M0 (two-pulse minus single-pulse responses)"
        g.attrs["sign_convention"] = ("omega_tau axes are PHYSICAL omega_T (positive = non-rephasing), "
                                      "i.e. -fftfreq of the raw tau transform, ascending")

        ax = g.create_group("axes")
        ax.create_dataset("t_full", data=t)
        ax.create_dataset("t", data=t[td_idx])
        ax.create_dataset("tau", data=tau)

        # ---------- reference single-pulse trajectories (full resolution) ----------
        ref = g.create_group("reference")
        for name in ["M_global_SU2", "M_global_SU3", "M_antiferro_SU2", "M_local_SU3"]:
            ref.create_dataset(name, data=f[f'/reference/{name}'][:], compression="gzip", compression_opts=4)
        Rl = f['/reference/M_local_SU3'][:]
        ref.create_dataset("quad", data=(Rl[:, 0]*Rl[:, 1]), compression="gzip", compression_opts=4)

        # ---------- assemble M_NL ----------
        M0su2 = f['/reference/M_global_SU2'][:]
        M0su3 = f['/reference/M_global_SU3'][:]
        q0    = Rl[:, 0]*Rl[:, 1]
        MNL2 = np.zeros((ntau, len(td_idx), 3), np.float32)
        MNL3 = np.zeros((ntau, len(td_idx), 8), np.float32)
        QNL  = np.zeros((ntau, len(td_idx)),    np.float32)
        for i in range(ntau):
            grp = f[f'/tau_scan/tau_{i}']
            MNL2[i] = (grp['M01_global_SU2'][:] - grp['M1_global_SU2'][:] - M0su2)[td_idx].astype(np.float32)
            MNL3[i] = (grp['M01_global_SU3'][:] - grp['M1_global_SU3'][:] - M0su3)[td_idx].astype(np.float32)
            A_ = grp['M01_local_SU3'][:]; B_ = grp['M1_local_SU3'][:]
            QNL[i] = (A_[:, 0]*A_[:, 1] - B_[:, 0]*B_[:, 1] - q0)[td_idx].astype(np.float32)

        tdg = g.create_group("time_domain")
        tdg.attrs["axes"] = "(tau, t, component); t decimated x%d (dt = %.3f units)" % (DEC, (t[1]-t[0])*DEC)
        tdg.create_dataset("MNL_SU2", data=MNL2, compression="gzip", compression_opts=4)
        tdg.create_dataset("MNL_SU3", data=MNL3, compression="gzip", compression_opts=4)
        tdg.create_dataset("QNL",     data=QNL,  compression="gzip", compression_opts=4)

        # ---------- mixed domain: FFT along tau only ----------
        # per-t tau-mean removed (kills the omega_tau = 0 pump-probe row), Hann over tau
        wta = np.fft.fftshift(np.fft.fftfreq(ntau, tau[1]-tau[0]))*SCALE
        order = np.argsort(-wta)                     # physical omega_T = -wta, ascending
        ax.create_dataset("omega_tau", data=(-wta)[order])
        atau = np.hanning(2*ntau)[:ntau]
        mg = g.create_group("mixed_domain")
        mg.attrs["axes"] = "(omega_tau_physical ascending, t decimated, component); complex64"
        mg.attrs["processing"] = "per-t tau-mean removed; half-Hann tau apodization (max weight at tau=0); FFT over tau; fftshift; reordered to physical omega_T"
        for name, arr in [("MNL_SU2", MNL2), ("MNL_SU3", MNL3), ("QNL", QNL[..., None])]:
            X = arr - arr.mean(axis=0, keepdims=True)
            F = np.fft.fftshift(np.fft.fft(X*atau[:, None, None], axis=0), axes=0)[order]
            mg.create_dataset(name, data=F.astype(np.complex64).squeeze(),
                              compression="gzip", compression_opts=4)

        # ---------- frequency domain: standard 2D analysis pipeline ----------
        tm = t >= 3.0
        at = np.hanning(2*tm.sum())[tm.sum():]
        wt = np.fft.fftshift(np.fft.fftfreq(tm.sum(), t[1]-t[0]))*SCALE
        mx = (wt > 0.0) & (wt < 2.0)
        ax.create_dataset("omega_t", data=wt[mx])
        fg = g.create_group("freq_domain")
        fg.attrs["axes"] = "(omega_tau_physical ascending, omega_t 0..2 THz, component); |FFT2| float32"
        fg.attrs["processing"] = ("t >= 3 window, per-t tau-mean removed, Hann on both axes "
                                  "(same pipeline as the atlas/census scripts; NO smoothing applied)")
        # recompute at FULL t resolution for the spectra (uses the run file directly)
        for name, cols, getter in [
            ("MNL_SU2", 3, lambda grp: grp['M01_global_SU2'][:] - grp['M1_global_SU2'][:] - M0su2),
            ("MNL_SU3", 8, lambda grp: grp['M01_global_SU3'][:] - grp['M1_global_SU3'][:] - M0su3)]:
            out_arr = np.zeros((ntau, mx.sum(), cols), np.float32)
            buf = np.zeros((ntau, tm.sum()))
            for c in range(cols):
                for i in range(ntau):
                    grp = f[f'/tau_scan/tau_{i}']
                    buf[i] = getter(grp)[tm, c]
                X = (buf - buf.mean(axis=0, keepdims=True))*atau[:, None]*at[None, :]
                F = np.abs(np.fft.fftshift(np.fft.fft2(X)))[:, mx][order]
                out_arr[:, :, c] = F.astype(np.float32)
            fg.create_dataset(name, data=out_arr, compression="gzip", compression_opts=4)
        # quad channel spectrum
        bufq = np.zeros((ntau, tm.sum()))
        for i in range(ntau):
            grp = f[f'/tau_scan/tau_{i}']
            A_ = grp['M01_local_SU3'][:]; B_ = grp['M1_local_SU3'][:]
            bufq[i] = (A_[:, 0]*A_[:, 1] - B_[:, 0]*B_[:, 1] - q0)[tm]
        Xq = (bufq - bufq.mean(axis=0, keepdims=True))*atau[:, None]*at[None, :]
        fg.create_dataset("QNL", data=np.abs(np.fft.fftshift(np.fft.fft2(Xq)))[:, mx][order].astype(np.float32),
                          compression="gzip", compression_opts=4)
    print("wrote", out, f"({os.path.getsize(out)/1e6:.0f} MB)")

export("geomB_gs1", f"{HERE}/trajectories/geomB_gs1_trajectories.h5")
export("geomA_gs1", f"{HERE}/trajectories/geomA_gs1_trajectories.h5")

# thermal level seeds: reference trajectories only (full M_NL available in runs/)
for run in ["geomA_gs2", "geomA_gs3"]:
    src = f"{HERE}/runs/{run}/sample_0/pump_probe_spectroscopy.h5"
    out = f"{HERE}/trajectories/{run}_reference_only.h5"
    with h5py.File(src) as f, h5py.File(out, "w") as g:
        g.attrs["note"] = "level-seed run: single-pulse reference only; full tau scan in runs/" + run
        g.create_dataset("t", data=f['/reference/times'][:])
        for name in ["M_global_SU2", "M_global_SU3", "M_local_SU3"]:
            g.create_dataset(name, data=f[f'/reference/{name}'][:], compression="gzip", compression_opts=4)
    print("wrote", out)
print("DONE")
