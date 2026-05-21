# TmFeO3 Geometry-II 2DCS pipeline — end-to-end specification

This document describes, in full detail, the chain of operations that produced
the Tm-dipole 2DCS spectra (F, G modes) shown for the Geom-II pilot point
`geomII_chi_0p01_W_0p001` from the parameter file all the way down to the
plotted lab-frame observable. It is meant to be sufficient to re-derive the
output by hand or in an independent implementation.

The reference files are:

- Config: [tmfeo3_2dcs_geomII_geomIII/configs/scan_geomII_chiW1/geomII_chi_0p01_W_0p001.param](tmfeo3_2dcs_geomII_geomIII/configs/scan_geomII_chiW1/geomII_chi_0p01_W_0p001.param)
- Hamiltonian builder: [src/core/unitcell_builders.cpp](src/core/unitcell_builders.cpp) (TmFeO3 helpers, lines ~419–1080)
- Time evolution: [src/core/mixed_lattice_md.cpp](src/core/mixed_lattice_md.cpp) (drive envelopes lines ~257–273; LLG lines ~314+)
- Python observable: [util/readers_new/reader_TmFeO3.py](util/readers_new/reader_TmFeO3.py) (lines 320–510)
- Plotting: [tmfeo3_2dcs_geomII_geomIII/scripts/plot_tm_F_dipole_2dcs.py](tmfeo3_2dcs_geomII_geomIII/scripts/plot_tm_F_dipole_2dcs.py)
- Pilot output: [build/workflow/scan_geomII_chiW1/geomII_chi_0p01_W_0p001/sample_0/pump_probe_spectroscopy.h5](build/workflow/scan_geomII_chiW1/geomII_chi_0p01_W_0p001/sample_0/pump_probe_spectroscopy.h5)
- Theory background: [docs/tmfeo3_notes.tex](docs/tmfeo3_notes.tex)

Units used throughout: energy in meV, time in ps, frequency in THz. (The code
uses $\hbar = 1$ with the identification $1\,\text{meV} = 2\pi \cdot 0.24180\,\text{THz}$, so spectra reported in raw FFT-frequency units (`1/T_step`) are in THz.)

---

## 1. Lattice and site indexing

4-sublattice Pbnm cell. Both Fe and Tm sublattice lists are ordered identically
in C++ and Python, and the Klein-four sublattice characters (`ETA_PBNM`,
[unitcell_builders.cpp `kEtaPbnm`](src/core/unitcell_builders.cpp); [reader_TmFeO3.py L327](util/readers_new/reader_TmFeO3.py#L327)) are

$$
\boldsymbol{\eta}_\mu \;=\;
\begin{pmatrix} +1 & +1 & +1 \\ +1 & -1 & -1 \\ -1 & +1 & -1 \\ -1 & -1 & +1 \end{pmatrix},
\quad \mu \in \{0,1,2,3\}.
$$

Bertaut sign vectors for the four irreducible single-cell modes
([reader_TmFeO3.py L335](util/readers_new/reader_TmFeO3.py#L335)):

| Mode | $\sigma_0$ | $\sigma_1$ | $\sigma_2$ | $\sigma_3$ | Meaning |
|------|-----------|-----------|-----------|-----------|---------|
| F | $+1$ | $+1$ | $+1$ | $+1$ | qFM net |
| G | $+1$ | $-1$ | $-1$ | $+1$ | qAFM Néel |
| C | $+1$ | $-1$ | $+1$ | $-1$ | C-mode |
| A | $+1$ | $+1$ | $-1$ | $-1$ | A-mode |

`lattice_size = 1,1,1` ⇒ 4 Fe sites and 4 Tm sites; `n_sublattices = 4`; site
index $i$ maps to sublattice $\mu(i) = i \bmod 4$.

Stored degrees of freedom per site:

- Fe: SU(2) spin, $\mathbf{S}_i \in \mathbb{R}^3$ with $|\mathbf{S}_i| = s_{\rm SU2} = 2.5$, `spin_dim_SU2 = 3`.
- Tm: SU(3) Bloch vector $\boldsymbol{\lambda}_i \in \mathbb{R}^8$ (Gell-Mann components), with effective $|\boldsymbol{\lambda}_i| = s_{\rm SU3} = 1$, `spin_dim_SU3 = 8`.

State vector per time slice in the HDF5 file:

```
state[t, : L2*sd2]                 = Fe spins flattened (4 sites × 3) = 12 floats
state[t, L2*sd2 :]                 = Tm Bloch vectors flattened (4 sites × 8) = 32 floats
state_dim_total                    = 12 + 32 = 44
```

Current TmFeO3 storage is fixed: Fe spins are stored in lab Cartesian
(per-sublattice frame is identity) and Tm Bloch vectors are stored in the
local CEF basis. The legacy `use_global_frame` / `use_local_frame` flags are
ignored by the current builders. The SU(3) frame `F_µ` is applied only when
projecting to the lab dipole; see §5.

---

## 2. Hamiltonian (the exact expression that is simulated)

The full Hamiltonian is the sum of three blocks, $H = H_{\rm Fe} + H_{\rm Tm} + H_{\rm Fe-Tm}$, all in meV.

### 2.1 Fe sector ([apply_tmfeo3_fe_sector](src/core/unitcell_builders.cpp#L508))

Bilinear exchange + Dzyaloshinskii–Moriya + single-ion anisotropy + Zeeman:

$$
H_{\rm Fe} \;=\;
\tfrac12 \sum_{\langle ij\rangle} S^a_i\,J^{ab}_{ij}\,S^b_j
\;+\; \sum_i \mathbf S_i^\top K \,\mathbf S_i
\;-\; \sum_i \mathbf{h}_i^{\rm stored}\!\cdot \mathbf S_i,
$$

with $J^{ab}_{ij} = J_{\rm iso} \delta^{ab} + \varepsilon^{abc} D^c_{ij}$ on
each bond and the stored Zeeman field equal to the lab field,
$\mathbf{h}_i^{\rm stored} = \mathbf{h}_{\rm lab}$. The Bertaut / AFM
observables still use the Pbnm sign pattern via `set_afm_sublattice_signs`,
but the Hamiltonian itself no longer has a frame-mode branch.

Bond list (per 4-site Pbnm cell — see [unitcell_builders.cpp L569–L607](src/core/unitcell_builders.cpp#L569-L607)):

| Bond | Sites | Offsets (cells) | $J_{\rm iso}$ | $D_y$ | $D_z$ |
|------|------|----------------|--------------|-------|-------|
| In-plane NN (z=½ plane, Fe1↔Fe0) | (1,0) | (0,0,0), (1,−1,0), (0,−1,0), (1,0,0) | $J_{1ab}$ | $+D_1$ | $+D_2$ |
| In-plane NN (z=0 plane, Fe2↔Fe3) | (2,3) | same 4 | $J_{1ab}$ | $-D_1$ | $+D_2$ |
| Out-of-plane NN ‖ c | (0,3),(1,2) | (0,0,0), (0,0,1) | $J_{1c}$ | 0 | 0 |
| 2nd-neighbour intra-sublattice | (i,i) ∀i | (1,0,0),(0,1,0),(0,0,1) | $J_{2ab}, J_{2ab}, J_{2c}$ | 0 | 0 |
| 2nd-neighbour cross-sublattice ‖ c | (0,2),(1,3) | 8 c-axis offsets | $J_{2c}$ | 0 | 0 |

Single-ion anisotropy: $K = \mathrm{diag}(K_a, K_b, K_c)$ (frame-invariant since $\eta^2 = 1$).

Values used in this run (from the config):

$$
\begin{aligned}
J_{1ab}&=4.74,\; J_{1c}=5.15,\; J_{2ab}=0.15,\; J_{2c}=0.30,\\
K_a&=-0.0153,\; K_b=0,\; K_c=-0.0187,\\
D_1&=0.049,\; D_2=0,\quad h=0 .
\end{aligned}
$$

These reproduce the experimental $\Gamma_2$ ground state (canted-AFM with
$F_x$, $G_z$, no $C_y$). The DM term $D_1$ is what produces the weak FM
canting; the sign flip between the z=½ and z=0 planes is the Pbnm-mandated
glide-plane mapping.

### 2.2 Tm sector ([apply_tmfeo3_tm_sector](src/core/unitcell_builders.cpp#L667))

The Tm CEF ground multiplet is a $J=6$ singlet plus a low-lying doublet,
projected onto the lowest three levels. The on-site Hamiltonian in the CEF
basis is

$$
H_{\rm CEF}^{(i)} \;=\; \mathrm{diag}\!\big(0,\;\varepsilon_1,\;\varepsilon_2\big)
\quad\Longleftrightarrow\quad
H^{(i)}_{\rm CEF}
= h_3 \lambda^3_i + h_8 \lambda^8_i,
$$

with the conversion
$h_3 = (\varepsilon_1 - 0)/2$,
$h_8 = (\varepsilon_1 + \varepsilon_2 - 2\cdot 0)/(2\sqrt3)$,
giving the three CEF gaps

$$
\omega_{12} = \varepsilon_1 = 2.067834\,\text{meV} \;\equiv\; 0.500\,\text{THz},\qquad
\omega_{13} = \varepsilon_2 = 4.9628\,\text{meV} \;\equiv\; 1.200\,\text{THz},
$$

and $\omega_{23} = \varepsilon_2 - \varepsilon_1 = 0.700\,\text{THz}$.

The projected magnetic-moment matrix (which connects the Gell-Mann time-odd
triplet $\{\lambda^2,\lambda^5,\lambda^7\}$ to the physical dipole) is

$$
J_\alpha^{(i, {\rm can})} = \mu_{\alpha b}\,\lambda^b_i,\qquad
\mu \;=\;
\begin{pmatrix}
\mu_{2x} & \mu_{5x} & \mu_{7x}\\
\mu_{2y} & \mu_{5y} & \mu_{7y}\\
\mu_{2z} & \mu_{5z} & \mu_{7z}
\end{pmatrix},
$$

with defaults (from the J=6 CEF calculation in [tmfeo3_notes.tex](docs/tmfeo3_notes.tex)):

$$
\mu \;=\;
\begin{pmatrix}
0 & 2.3915 & 0.9128\\
0 & -2.7866 & 0.4655\\
5.264 & 0 & 0
\end{pmatrix}.
$$

These three columns correspond to the dominant dipole channels of the
$\lambda^2, \lambda^5, \lambda^7$ generators. No Tm-Tm bilinears
(`Jtm_2 = Jtm_5 = Jtm_7 = 0`) and no external Zeeman are active in this run.

So the full Tm sector in this run is

$$
H_{\rm Tm} = \sum_i \big(\tfrac12\varepsilon_1 \lambda^3_i + \tfrac{\varepsilon_1+\varepsilon_2}{2\sqrt3} \lambda^8_i\big).
$$

The sublattice frame $F_\mu \in \mathbb{R}^{8\times 8}$ that maps the local
CEF basis to the lab is constructed in the **Option A pair-locked**
extension:

$$
F_\mu \;=\; R_\mu^{\rm act} \text{ on } \{\lambda^2,\lambda^5,\lambda^7\} \;\&\; \{\lambda^1,\lambda^4,\lambda^6\},
\quad \mathbf 1 \text{ on } \{\lambda^3,\lambda^8\},
$$

with

$$
R_\mu^{\rm act} \;=\; \mu^{-1}\, D_\mu\, \mu,\quad D_\mu = \mathrm{diag}(\boldsymbol\eta_\mu).
$$

Diagonal $(\lambda^3,\lambda^8)$ are Klein-invariant (populations). The
quadrupole partners $(\lambda^1,\lambda^4,\lambda^6)$ are pair-locked to
their dipole partners so that each coherence pair $|E_a\rangle\langle E_b|$
carries a single bulk character $\chi_a \chi_b$. This is the construction
implemented in [unitcell_builders.cpp lines 744-755](src/core/unitcell_builders.cpp#L744-L755).

### 2.3 Fe–Tm couplings ([apply_tmfeo3_fe_tm_couplings](src/core/unitcell_builders.cpp#L901))

Two channels are active in this run:

**Linear (χ) on the Fe–Tm bonds** (set by `chi_orbit{1..4}_scale`, 8 NN Fe–Tm bonds per Fe partitioned into 4 orbits, with type E/I = identity-/inversion-paired):

$$
H_\chi \;=\; \sum_{(i_{\rm Fe}, j_{\rm Tm}) \in {\rm bonds}}
S^a_{i_{\rm Fe}}\, \chi^{\rm sign}_{ab}\, \lambda^b_{j_{\rm Tm}},
$$

where the $3\times 8$ matrix $\chi$ has nonzero entries only in columns $b \in \{1,4,6\}$ (the time-odd Gell-Mann triplet), with

- $\chi_{a,1} = \chi_{2a}$ (no sign flip in I-bonds)
- $\chi_{a,4} = \mathrm{sign}_{57}\cdot\chi_{5a}$
- $\chi_{a,6} = \mathrm{sign}_{57}\cdot\chi_{7a}$

and $\mathrm{sign}_{57} = +1$ on E-bonds, $-1$ on I-bonds (inversion-paired).

This run sets $\chi_{2z} = 0.01$ meV and all other $\chi_{\alpha\beta} = 0$.

**On-site trilinear (W) on the Fe–Fe–Tm three-leg vertex** ([unitcell_builders.cpp L932–L941](src/core/unitcell_builders.cpp#L932-L941)):

$$
H_W \;=\; \sum_{(i_{\rm Fe}, j_{\rm Tm})\in{\rm bonds}}\sum_{ac}
S^a_{i_{\rm Fe}}\, S^c_{i_{\rm Fe}}\,
W^{(b)}_{ac}\, \lambda^b_{j_{\rm Tm}},
$$

with $W^{(b)}_{ac}$ symmetric in $(a,c)$ and only the diagonal Tm channels
$b\in\{1,3,8\}$ (A₁⁺) and $\{4,6\}$ (A₂⁺) populated. This run sets

$$
W_{1,xx} = +0.001,\quad W_{1,zz} = -0.001,
$$

i.e. a small symmetric-traceless piece on the $\lambda^1$ Tm quadrupole
channel. All V (inter-site Fe–Fe–Tm), all higher W, V_A, and all other Tm
couplings are zero in this run.

This is the **smoking-gun coupling**: $\chi_{2z}$ is what transfers Fe
qAFM dynamics into the Tm $\lambda^2$ dipole, and $W_1$ provides a small
quadrupole channel that lifts certain selection rules.

---

## 3. Pump-probe protocol (Geometry II)

### 3.1 Pulse geometry

Geometry II is defined by

- pump field $\mathbf H \parallel \hat a$ (Fe Zeeman drive, x-polarized at the Fe)
- detection polarization VH (cross-polarized → selects the even Tm sector and qAFM)

From the config:

```
pump_direction = (1, 0, 0)   # Fe pump along a
pump_amplitude = 0.1         # meV
pump_width     = 0.30        # ps  (Gaussian σ-like)
pump_frequency = 0.5         # carrier ≈ 0.121 THz · 0.5 = 0.500 THz
pump_amplitude_su3 = 0.0     # SU3 pump disabled in this scan
```

### 3.2 Drive form (the exact $\mathbf B(t)$ that is added to the LLG RHS)

Per [`drive_envelopes_SU2`](src/core/mixed_lattice_md.cpp#L263) the two pulse
envelopes at time $t$ are

$$
\phi_k(t) \;=\; A\;\exp\!\Big[-\!\Big(\tfrac{t - t_k}{2\sigma}\Big)^{\!2}\Big]\;\cos\!\big(\omega_c (t - t_k)\big),\quad k\in\{1,2\},
$$

with $A = $ `pump_amplitude` $= 0.1$, $\sigma = $ `pump_width` $= 0.30$ ps,
$\omega_c = $ `pump_frequency` $= 0.5$ (rad/ps in code units).

For each MD trajectory the drive field at each Fe site is

$$
\mathbf B_i(t) \;=\; \phi_1(t)\,\mathbf b^{(1)}_{\mu(i)} \;+\; \phi_2(t)\,\mathbf b^{(2)}_{\mu(i)},
$$

where $\mathbf b^{(k)}_\mu$ is the stored per-sublattice pulse polarization
vector. For Geom II, $\mathbf b^{(k)} = (A,0,0)$ in the lab and, in the
current canonical storage, the stored Fe polarization is the same lab vector
at every site. Older notes that discuss an additional `use_global_frame`
branch refer to the retired local-frame path.

### 3.3 Three-trajectory protocol (the 2DCS subtraction)

For each pump–probe delay $\tau$ in the scan, three MD trajectories are run:

| Trajectory | Pulses applied | t-grid |
|------------|----------------|--------|
| M0 | none (reference, same initial state, no pulses) | $t\in[-100, +100]$ ps |
| M1 | only pulse-2 (probe, centred at $t_2 = 0$) | same |
| M01 | both pulses: pump centred at $t_1 = -\tau$, probe at $t_2 = 0$ | same |

All three start from the same equilibrated initial state ([seed file:
`tmfeo3_2dcs_geomII_geomIII/seeds/tmfeo3_gamma2_global_1x1x1`]).

The pure third-order non-linear response is

$$
\boxed{\;M_{\rm NL}(\tau, t)\;=\;M_{01}(\tau, t) - M_0(t) - M_1(t)\;}
$$

This isolates the cross-pulse-mixing term; the linear responses to either
pulse alone (M0 ≡ 0 if no pulses; M1 is the linear-probe-only signal) are
subtracted out.

### 3.4 Scan parameters

```
tau_start = -100.0 ps
tau_end   =    0.0 ps
tau_step  =    0.05 ps     →  n_tau = 2001
md_time_start = -100.0
md_time_end   = +100.0
md_timestep   = 0.05 ps    →  n_t   = 4001
integrator   = dopri5     (adaptive Dormand–Prince RK45)
parallel_tau = true       (each τ point in a separate MPI rank)
```

The pilot run was launched with 4 MPI ranks (`mpirun -np 4 spin_solver …`).

### 3.5 Time evolution: LLG / SU(3) Bloch equations

Per [`landau_lifshitz`](src/core/mixed_lattice_md.cpp#L317) the EoM is

$$
\dot{\mathbf S}_i \;=\; \mathbf S_i \times \mathbf B^{\rm eff}_i(t),\qquad
\dot{\boldsymbol\lambda}_i \;=\; \boldsymbol\lambda_i \star \mathbf h^{\rm eff}_i(t),
$$

with $\mathbf B^{\rm eff}_i = -\partial H/\partial \mathbf S_i + \mathbf B^{\rm drive}_i(t)$ on Fe, and the SU(3) Poisson bracket $\star = -f^{abc}$ on Tm, with $\mathbf h^{\rm eff}_i = -\partial H/\partial \boldsymbol\lambda_i + \mathbf h^{\rm drive}_i(t)$.

The integrator is dopri5 with absolute/relative tolerances at the default
$10^{-9}$ (set in the spin-solver MD harness). For each $\tau$ the three
trajectories share the same RNG-free initial condition, so the only
difference between M0, M1, M01 is the drive term.

### 3.6 What is written to disk

With `save_spin_trajectories = 1.0` the HDF5 layout is

```
/metadata                  attrs: lattice_size_SU2, spin_dim_SU2,
                                  lattice_size_SU3, spin_dim_SU3,
                                  T_step, T_start, T_end, tau_step, tau_start, …
/reference/times           (n_t,)               t-grid in ps
/reference/M0_spin_state   (n_t, 44)            raw spins, no pulses
/reference/M_global_SU2    (n_t, 3)             C++-projected Fe lab dipole
/reference/M_global_SU3    (n_t, 8)             C++-projected Tm Gell-Mann avg
/tau_scan/tau_i/M1_spin_state    (n_t, 44)     probe only
/tau_scan/tau_i/M01_spin_state   (n_t, 44)     pump + probe
/tau_scan/tau_i/M_global_SU2     (n_t, 3)      C++ projection (M01 - M0 - M1)
/tau_scan/tau_i/M_global_SU3     (n_t, 8)      "
```

The raw `*_spin_state` arrays are the canonical source of truth — they
contain the unprojected per-site state and bypass any C++-side
sublattice-frame ambiguity. The C++ `M_global_SU3` projection uses the
$F_\mu^\top$ convention in `mixed_lattice.h` (which differs from `lattice.h`'s
$F_\mu$); rather than fix that long-standing inconsistency we reconstruct the
observable in Python from the raw spins.

---

## 4. Observable definition: physical lab Tm dipole

The Bertaut-projected lab Tm dipole for mode $X \in \{F,G,C,A\}$ is

$$
\boxed{\;\;J^a_X(t) \;=\;\frac{1}{N}\sum_{i=1}^{N}\sigma^X_{\mu(i)}\;
\eta^a_{\mu(i)}\;\;\sum_{b\in\{2,5,7\}} \mu_{a,b}\,\lambda^b_i(t).\;\;}
$$

Breakdown of the three factors:

1. $\sum_b \mu_{a,b}\,\lambda^b_i(t)$ — projects the (canonical-frame) SU(3)
   Bloch vector onto the physical magnetic-moment basis. Only the time-odd
   Gell-Mann triplet $\{\lambda^2,\lambda^5,\lambda^7\}$ contributes (other
   components carry no dipole). This is the canonical-frame ion dipole.

2. $\eta^a_{\mu(i)}$ — pushes the canonical-frame dipole into the lab
   Cartesian for site $i$ on sublattice $\mu$ ($D_\mu = \mathrm{diag}(\boldsymbol\eta_\mu)$).
   This is the on-site contribution of the SU(3) sublattice frame $F_\mu$
   projected onto the dipole subspace, where it acts diagonally as $\boldsymbol\eta_\mu$.

3. $\sigma^X_{\mu(i)}$ — Bertaut sign pattern selecting the irrep (F, G, C, A).

This factorisation works because the lattice has 4 distinct sublattices each
with a 1-d $\eta_\mu$ character, so $F_\mu$ acts diagonally on $J$ even
though it is non-trivial in the full 8-d Gell-Mann space.

The Python implementation is
[`tmfeo3_tm_dipole_lab`](util/readers_new/reader_TmFeO3.py#L444). The
equivalent SU(2) helper for Fe is
[`tmfeo3_fe_sublattice_frames`](util/readers_new/reader_TmFeO3.py#L434).

### 4.1 Why F-mode vanishes by symmetry

For any Cartesian axis $a\in\{x,y,z\}$, summing $\eta^a_\mu$ over the four
sublattices gives

$$
\sum_\mu \eta^a_\mu \;=\; 0\quad \forall\,a,
$$

(each column of $\eta$ has two $+1$ and two $-1$). With $\sigma^F_\mu = +1$
this forces $J^a_F(t) = 0$ identically, independent of dynamics, microscopic
parameters, or numerical noise. The simulator confirms this to the expected
$\sim 10^{-13}$ floating-point floor.

The G-mode pattern $(+,-,-,+)$ combined with $\eta^z = (+,-,-,+)$ gives
$\sigma^G \cdot \eta^z = +4$ on the z-component, so the G-z channel is the
correct "qAFM along c" observable that selects the geometry's response.

### 4.2 M_NL projection

For each Cartesian component $a$ and each $\tau$-slice, the protocol-level
non-linear signal is

$$
M^{(X)}_{\rm NL}(\tau, t)^a \;=\; J^a_X[\,\mathrm{M01}(\tau)\,](t)\;-\;J^a_X[\,\mathrm{M0}\,](t)\;-\;J^a_X[\,\mathrm{M1}\,](t),
$$

each $J_X[\cdot]$ being the lab projection of the corresponding stored raw
trajectory.

---

## 5. 2DCS spectrum

### 5.1 Apodization

A 2D Hann window is applied:

$$
w(\tau, t) \;=\; w_{\rm Hann}(\tau)\, w_{\rm Hann}(t),
$$

with `scipy.signal.windows.hann(n, sym=False)`. (Disable with `--no-apodize`.)

### 5.2 FFT and frequency-axis convention

Mean-subtraction → apodize → 2D FFT → fftshift → flip $\omega_\tau$ (because in
the simulation $\tau$ is the pump-arrival delay relative to probe; the
experimentally natural T-axis is $T = -\tau$):

$$
\tilde M_{\rm NL}(\omega_\tau, \omega_t)\;=\;
\mathrm{FT}_{2D}\!\Big[\,\big(M_{\rm NL}(\tau,t)-\bar M_{\rm NL}\big)\,w(\tau,t)\,\Big],
$$

and the plotted spectrum is $|\tilde M_{\rm NL}|$.

### 5.3 Frequency grid

With `T_step = tau_step = 0.05 ps`,

$$
\Delta\omega_t = \tfrac{1}{0.05\cdot 4001}\approx 5.0\,\mathrm{mTHz},\quad
\Delta\omega_\tau = \tfrac{1}{0.05\cdot 2001}\approx 10.0\,\mathrm{mTHz},\quad
|\omega|_{\rm Nyq} = \pm 10\,\text{THz}.
$$

---

## 6. Verified output

### 6.1 F-mode (`--mode F`)

All Cartesian components are zero to numerical precision:

```
x: time-domain max|M_NL| = 0.000e+00     spectrum max = 0.000e+00
y: time-domain max|M_NL| = 0.000e+00     spectrum max = 0.000e+00
z: time-domain max|M_NL| = 1.696e-13     spectrum max = 1.660e-07
```

The z value is a Kahan-sum residual from $\sum_\mu \sigma^F_\mu \eta^z_\mu \cdot (\cdots) = 0$ summed in finite precision. This is the symmetry result described in §4.1.

### 6.2 G-mode (`--mode G`)

x and y zero by symmetry; z carries the physical signal:

```
z: time-domain max|M_NL| = 7.249e-05   at tau = -99.35 ps, t = +97.75 ps
   spectrum    max = 17.88
```

The 2D spectrum shows two clear peaks at $\omega_\tau \approx \pm 0.6$ THz with $\omega_t \approx 0.33$ THz (Tm E1 cross-coupled to the pump) and $\omega_t \approx 1.2$ THz (Tm E2 = $\varepsilon_2 = 4.9628$ meV). This is the (qAFM, E1→E2) cross-peak structure predicted by the inter-model summary in [tmfeo3_notes.tex](docs/tmfeo3_notes.tex) (Sec. "Geom II").

---

## 7. Reproduction (start-to-finish)

```bash
# 1. Build
cd build && cmake .. && make -j8 spin_solver

# 2. Run a single Geom II point (requires save_spin_trajectories=1 in .param)
mpirun -np 4 ./spin_solver \
   ../tmfeo3_2dcs_geomII_geomIII/configs/scan_geomII_chiW1/geomII_chi_0p01_W_0p001.param

# 3. Plot a Bertaut-mode 2DCS spectrum from the raw spin trajectories
source ../.venv/bin/activate
python ../tmfeo3_2dcs_geomII_geomIII/scripts/plot_tm_F_dipole_2dcs.py \
   workflow/scan_geomII_chiW1/geomII_chi_0p01_W_0p001/sample_0/pump_probe_spectroscopy.h5 \
   --mode G --omega-t-max 2.5 --omega-tau-max 1.5
```

Each invocation writes `M_NL_Tm_<MODE>_dipole_2dcs.pdf` and a matching
`.npz` (spectra, M_NL, omega_t, omega_tau, times, tau, labels) into the same
directory as the HDF5 file.

---

## 8. Provenance and outstanding issues

- **C++ `M_global_SU3` / `M_antiferro_SU3` arrays** in
  `mixed_lattice.h` apply $F_\mu^\top$ where `lattice.h` applies $F_\mu$. For
  diagonal $D_\mu$ (i.e. Fe) these agree; for the Option-A SU(3) `R_act` they
  do not. This pipeline sidesteps the issue entirely by recomputing the
  observable in Python from raw spins.

- **Tm `M_antiferro_SU3`** is identically zero in the HDF5 file because the
  Tm sector ships without `afm_sublattice_signs_SU3` (defaults to all-ones,
  i.e. F-mode, which vanishes by §4.1). The Python helper provides any
  Bertaut sign pattern on demand.

- The 16-point χ × W scan
  `tmfeo3_2dcs_geomII_geomIII/configs/scan_geomII_chiW1/*.param` has been
  patched with `save_spin_trajectories = 1.0` so the same Python analysis
  applies to every point.
