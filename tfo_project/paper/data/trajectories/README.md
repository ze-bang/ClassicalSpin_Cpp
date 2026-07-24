# Full time-domain trajectories of the final scenario

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
