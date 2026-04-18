# TmFeO₃ Hamiltonian — Full Implementation Reference

This document describes the **exact** Hamiltonian implemented in the code.
Every term, index convention, sign, and counting factor is derived from the source files:

- `src/core/unitcell_builders.cpp` → `build_tmfeo3()`
- `include/classical_spin/lattice/mixed_lattice.h` → EOM, energy, MC
- `include/classical_spin/unitcell/unitcell.h` → bond storage

---

## 1. Crystal Structure and Sublattice Conventions

**Space group**: Pbnm (No. 62)  
**Lattice constants**: $a = 5.2534$ Å, $b = 5.5707$ Å, $c = 7.6076$ Å

### Fe sublattices (Wyckoff 4b, SU(2) sector, $S = 5/2$, `spin_dim = 3`)

| Index | Fractional coords |
|-------|-------------------|
| Fe₀ | $(0,\;\tfrac{1}{2},\;\tfrac{1}{2})$ |
| Fe₁ | $(\tfrac{1}{2},\;0,\;\tfrac{1}{2})$ |
| Fe₂ | $(\tfrac{1}{2},\;0,\;0)$ |
| Fe₃ | $(0,\;\tfrac{1}{2},\;0)$ |

### Tm sublattices (Wyckoff 4c, SU(3) sector, non-Kramers qutrit, `spin_dim = 8`)

| Index | Fractional coords |
|-------|-------------------|
| Tm₀ | $(0.02111,\;0.92839,\;0.75)$ |
| Tm₁ | $(0.52111,\;0.57161,\;0.25)$ |
| Tm₂ | $(0.47889,\;0.42839,\;0.75)$ |
| Tm₃ | $(0.97889,\;0.07161,\;0.25)$ |

### Sublattice frame signs $\eta_\mu$

All spins are stored in **local frames** related to the global frame by the staggering signs:

$$
\eta_\mu = \begin{cases}
(+1,+1,+1) & \mu = 0 \\
(+1,-1,-1) & \mu = 1 \\
(-1,+1,-1) & \mu = 2 \\
(-1,-1,+1) & \mu = 3
\end{cases}
$$

For Fe: $\mathbf{S}_\mu^{\text{global},\alpha} = \eta_{\mu\alpha}\, \mathbf{S}_\mu^{\text{local},\alpha}$ (diagonal rotation).

For Tm: The sublattice frame is built from the projected magnetic moment matrix $\mu$ as:

$$
F_\mu = \mu_{\text{act}}^{-1}\, D_\mu\, \mu_{\text{act}}, \qquad D_\mu = \text{diag}(\eta_{\mu,x},\,\eta_{\mu,y},\,\eta_{\mu,z})
$$

acting in the 3-dimensional active subspace spanned by $(\lambda_2,\,\lambda_5,\,\lambda_7)$ (indices 1, 4, 6 in 0-based C++ arrays), with the remaining 5 generators forming the identity.  
The `mu_act` matrix is:

$$
\mu_{\text{act}} = \begin{pmatrix}
\mu_{2x} & \mu_{5x} & \mu_{7x} \\
\mu_{2y} & \mu_{5y} & \mu_{7y} \\
\mu_{2z} & \mu_{5z} & \mu_{7z}
\end{pmatrix}
$$

with defaults:

$$
\mu_{\text{act}} = \begin{pmatrix}
0 & 2.3915 & 0.9128 \\
0 & -2.7866 & 0.4655 \\
5.264 & 0 & 0
\end{pmatrix}
$$

---

## 2. Full Hamiltonian

$$
\mathcal{H} = \mathcal{H}_{\text{Fe-Fe}} + \mathcal{H}_{\text{Fe}}^{\text{Zeeman}} + \mathcal{H}_{\text{Fe}}^{\text{SIA}} + \mathcal{H}_{\text{Tm}}^{\text{CEF}} + \mathcal{H}_{\text{Tm}}^{\text{Zeeman}} + \mathcal{H}_{\text{Tm-Tm}} + \mathcal{H}_{\chi} + \mathcal{H}_{W} + \mathcal{H}_{V}
$$

Each term is described below with the **exact** formula used in the code.

---

## 3. Fe–Fe Bilinear ($\mathcal{H}_{\text{Fe-Fe}}$)

### 3.1 Energy

$$
\mathcal{H}_{\text{Fe-Fe}} = \frac{1}{2}\sum_{i}\sum_{n \in \text{nbrs}(i)} \sum_{\alpha\beta} J_{i,n}^{\alpha\beta}\, S_i^\alpha\, S_n^\beta
$$

The factor $\frac{1}{2}$ avoids double counting. In the code (`total_energy_SU2`):

```
energy += 0.5 * spin.dot(J * spins_SU2[partner])
```

### 3.2 Exchange matrices (global frame)

**NN in-plane** (bonds between $z = 0$ and $z = \tfrac{1}{2}$ Fe layers):

For Fe₁→Fe₀ bonds (along $\hat{a}$, $\hat{b}$):

$$
J_a^{\text{glob}} = \begin{pmatrix}
J_{1ab} & D_2 & -D_1 \\
-D_2 & J_{1ab} & 0 \\
D_1 & 0 & J_{1ab}
\end{pmatrix}, \qquad
J_b^{\text{glob}} = \begin{pmatrix}
J_{1ab} & D_2 & -D_1 \\
-D_2 & J_{1ab} & 0 \\
D_1 & 0 & J_{1ab}
\end{pmatrix}
$$

For Fe₂→Fe₃ bonds (Pbnm partner layer, DM sign reversed):

$$
J_{a,23}^{\text{glob}} = \begin{pmatrix}
J_{1ab} & -D_2 & D_1 \\
D_2 & J_{1ab} & 0 \\
-D_1 & 0 & J_{1ab}
\end{pmatrix}
$$

**NN c-axis** (Fe₀↔Fe₃, Fe₁↔Fe₂):

$$
J_c^{\text{glob}} = J_{1c}\, \mathbb{1}_{3\times 3}
$$

**NNN** (same sublattice, $J_2$ type):

$$
J_{2a}^{\text{glob}} = J_{2ab}\,\mathbb{1}, \quad J_{2b}^{\text{glob}} = J_{2ab}\,\mathbb{1}, \quad J_{2c}^{\text{glob}} = J_{2c}\,\mathbb{1}
$$

### 3.3 Local frame transformation

All matrices are transformed to the local frame before storage:

$$
\tilde{J}_{ij}^{\alpha\beta} = J_{ij}^{\alpha\beta}\,\eta_{i,\alpha}\,\eta_{j,\beta}
$$

This absorbs the sublattice staggering so that the code can treat all spins as if ferromagnetically aligned.

### 3.4 Bond topology

| Bond type | Source → Partner | Offsets | Count |
|-----------|-----------------|---------|-------|
| $J_a$ (NN ab-plane) | Fe₁ → Fe₀ | $(0,0,0)$, $(1,-1,0)$ | 2 |
| $J_b$ (NN ab-plane) | Fe₁ → Fe₀ | $(0,-1,0)$, $(1,0,0)$ | 2 |
| $J_a$ (NN ab-plane) | Fe₂ → Fe₃ | $(0,0,0)$, $(1,-1,0)$ | 2 |
| $J_b$ (NN ab-plane) | Fe₂ → Fe₃ | $(0,-1,0)$, $(1,0,0)$ | 2 |
| $J_c$ (NN c-axis) | Fe₀ → Fe₃ | $(0,0,0)$, $(0,0,1)$ | 2 |
| $J_c$ (NN c-axis) | Fe₁ → Fe₂ | $(0,0,0)$, $(0,0,1)$ | 2 |
| $J_{2a}$ (NNN) | Fe$_\mu$ → Fe$_\mu$ | $(1,0,0)$ | 4 (one per sublattice) |
| $J_{2b}$ (NNN) | Fe$_\mu$ → Fe$_\mu$ | $(0,1,0)$ | 4 |
| $J_{2c}$ (NNN) | Fe$_\mu$ → Fe$_\mu$ | $(0,0,1)$ | 4 |
| $J_{2c}$ (NNN cross) | Fe₀ → Fe₂ | 8 offsets (all $ab$-plane + $c$) | 8 |
| $J_{2c}$ (NNN cross) | Fe₁ → Fe₃ | 8 offsets (all $ab$-plane + $c$) | 8 |

**Parameters**: `J1ab` (default 4.74), `J1c` (5.15), `J2ab` (0.15), `J2c` (0.30), `D1` (0.12), `D2` (0.0).

---

## 4. Fe Single-Ion Anisotropy ($\mathcal{H}_{\text{Fe}}^{\text{SIA}}$)

$$
\mathcal{H}_{\text{Fe}}^{\text{SIA}} = \sum_{i} \sum_{\alpha\beta} K_{\alpha\beta}\,S_i^\alpha\,S_i^\beta = \sum_{i} \left(K_a\,S_{i,x}^2 + K_b\,S_{i,y}^2 + K_c\,S_{i,z}^2\right)
$$

In the code (`total_energy_SU2`):

```
energy += spin.dot(onsite_interaction * spin)
```

Note: the 2× factor in the `get_local_field` ($H_\alpha = 2 K_{\alpha\beta} S_\beta$) is correct because $\partial(S^T K S)/\partial S_\alpha = 2(KS)_\alpha$ for symmetric $K$.

**Parameters**: `Ka` (default −0.16221), `Kb` (0.0), `Kc` (−0.18318). The anisotropy is diagonal and **identical in all local frames** (since $K$ commutes with $\eta_\mu\otimes\eta_\mu$).

---

## 5. Fe Zeeman ($\mathcal{H}_{\text{Fe}}^{\text{Zeeman}}$)

$$
\mathcal{H}_{\text{Fe}}^{\text{Zeeman}} = -\sum_{i} \mathbf{h} \cdot \mathbf{S}_i = -\sum_{i} \sum_\alpha h_\alpha\, S_i^\alpha
$$

The field $\mathbf{h} = h \cdot \hat{n}$ where $h$ = `field_strength` and $\hat{n}$ = `field_direction`.

In the code (`total_energy_SU2`):

```
energy -= spin.dot(field_SU2[i])
```

and in `get_local_field_SU2_flat`:

```
H = -field_SU2[site]  (contributes −h to energy, ×S gives -h·S)
```

The same field vector is applied to all 4 Fe sublattices (the $g$-factor is absorbed into $h$).

---

## 6. Tm CEF On-Site ($\mathcal{H}_{\text{Tm}}^{\text{CEF}}$)

The Tm non-Kramers doublet ground state splits into 3 singlets $\{|E_1(A_1)\rangle, |E_2(A_1)\rangle, |E_3(A_2)\rangle\}$ with energies controlled by:

$$
\mathcal{H}_{\text{Tm}}^{\text{CEF}} = -\sum_{j} \left( \alpha\,\lambda_3^{(j)} + \beta\,\lambda_8^{(j)} \right)
$$

where

$$
\alpha = e_1 \cdot \texttt{tm\_alpha\_scale}, \qquad \beta = \frac{2\,e_2 - e_1}{\sqrt{3}} \cdot \texttt{tm\_beta\_scale}
$$

In the diagonal basis: $E_1 - E_2 = 2\alpha$, $E_3 - \tfrac{1}{2}(E_1+E_2) = -\beta\sqrt{3}$.

**Parameters**: `e1` (default 0.97 meV), `e2` (3.97 meV), `tm_alpha_scale` (1.0), `tm_beta_scale` (1.0).

This enters as a **field** (not as an on-site quadratic), so the energy is:

```
energy -= spin.dot(field_SU3[i])
```

---

## 7. Tm Zeeman ($\mathcal{H}_{\text{Tm}}^{\text{Zeeman}}$)

$$
\mathcal{H}_{\text{Tm}}^{\text{Zeeman}} = -g_{\text{ratio}} \sum_{j} \sum_{a\in\{2,5,7\}} B_a^{(j)}\, \lambda_a^{(j)}
$$

where

$$
B_a^{(\mu)} = \sum_{\alpha \in \{x,y,z\}} \eta_{\mu,\alpha}\, \mu_{\alpha,a}\, h_\alpha
$$

The Zeeman contribution is **added** to the CEF field vector on each Tm sublattice. The $g$-factor ratio $g_{\text{ratio}} = g_{\text{Tm}}/g_{\text{Fe}}$ (default $7/12 \approx 0.583$) scales the Tm response relative to Fe.

**Parameters**: `g_ratio_tm` (default 7/12), `mu_{2,5,7}{x,y,z}` (9 parameters).

---

## 8. Tm–Tm Bilinear ($\mathcal{H}_{\text{Tm-Tm}}$)

$$
\mathcal{H}_{\text{Tm-Tm}} = \frac{1}{2}\sum_{\langle j,j'\rangle} \sum_{a=1}^{8} J_a^{\text{Tm}}\, \lambda_a^{(j)}\,\lambda_a^{(j')}
$$

Diagonal in Gell-Mann space. Nearest-neighbor Tm pairs: Tm₀↔Tm₂ (same $z = 0.75$ plane, 4 bond offsets) and Tm₁↔Tm₃ (same $z = 0.25$ plane, 4 bond offsets).

**Parameters**: `Jtm_1` through `Jtm_8` (all default 0.0).

---

## 9. Fe–Tm Bilinear: $\chi$ Exchange ($\mathcal{H}_\chi$)

### 9.1 Definition

$$
\mathcal{H}_\chi = \frac{1}{2}\sum_{i,j} \sum_{\alpha=1}^{3}\sum_{a=1}^{8} \chi_{ij}^{\alpha a}\, S_i^\alpha\, \lambda_j^a
$$

The $\frac{1}{2}$ is from double-counting: each bond contributes once from the SU(2) side (as `mixed_bilinear_interaction_SU2`) and once from the SU(3) side (as `mixed_bilinear_interaction_SU3` with transposed matrix).

### 9.2 Matrix structure

The $\chi$ matrix is $3 \times 8$ (rows = Fe spin component $\alpha \in \{x,y,z\}$, cols = Gell-Mann index $a \in \{1,\ldots,8\}$). Only the **T-odd** (time-reversal odd) generators $\lambda_2, \lambda_5, \lambda_7$ carry non-zero entries:

$$
\chi = \begin{pmatrix}
0 & \chi_{2x} & 0 & 0 & \chi_{5x} & 0 & \chi_{7x} & 0 \\
0 & \chi_{2y} & 0 & 0 & \chi_{5y} & 0 & \chi_{7y} & 0 \\
0 & \chi_{2z} & 0 & 0 & \chi_{5z} & 0 & \chi_{7z} & 0
\end{pmatrix}
$$

### 9.3 Inversion symmetry: $\chi$ vs $\chi^{\text{inv}}$

Pbnm inversion maps each Fe-Tm bond to a partner bond. The inversion-related bond carries $\chi^{\text{inv}}$:

$$
\chi^{\text{inv}} = \begin{pmatrix}
0 & \chi_{2x} & 0 & 0 & -\chi_{5x} & 0 & -\chi_{7x} & 0 \\
0 & \chi_{2y} & 0 & 0 & -\chi_{5y} & 0 & -\chi_{7y} & 0 \\
0 & \chi_{2z} & 0 & 0 & -\chi_{5z} & 0 & -\chi_{7z} & 0
\end{pmatrix}
$$

Rule: $\lambda_2$ is **A₁⁻** (inversion even among T-odd generators), $\lambda_5$ and $\lambda_7$ are **A₂⁻** (inversion odd) → sign flip under inversion.

### 9.4 Orbit scaling

Each of the 4 orbits (32 total bonds = 4 orbits × 8 bonds/orbit) carries a scale factor:

$$
\chi_{\text{orbit}\,k} = s_k^{(\chi)} \cdot \chi_{\text{base}}, \qquad \chi^{\text{inv}}_{\text{orbit}\,k} = s_k^{(\chi)} \cdot \chi^{\text{inv}}_{\text{base}}
$$

**Parameters**: `chi{2,5,7}{x,y,z}` (9 couplings), `chi_orbit{1,2,3,4}_scale` (4 scales, default 1.0).

### 9.5 EOM contribution

For SU(2) site $i$:

$$
H_\alpha^{(\text{Fe},i)} \mathrel{+}= \sum_{n} \sum_a \chi_{in}^{\alpha a}\, \lambda_n^a
$$

For SU(3) site $j$:

$$
H_a^{(\text{Tm},j)} \mathrel{+}= \sum_{n} \sum_\alpha (\chi^T)_{jn}^{a\alpha}\, S_n^\alpha
$$

where $\chi^T$ is the transposed interaction stored on the SU(3) side during bond initialization.

---

## 10. On-Site Trilinear: $W$ Coupling ($\mathcal{H}_W$)

### 10.1 Definition

$$
\mathcal{H}_W = \frac{1}{3}\sum_{\text{bonds}} \sum_{\alpha,\beta=1}^{3}\sum_{a=1}^{8} W^{\alpha\beta a}\, S_i^\alpha\, S_i^\beta\, \lambda_j^a
$$

Here $i$ is the Fe site (source = partner1, same site, offset = $(0,0,0)$), $j$ is the Tm partner.

The $\frac{1}{3}$ avoids triple-counting: each trilinear bond generates 3 stored copies (one per leg).

### 10.2 Tensor storage convention

The code stores $K[\alpha](\beta, a)$ as a `SpinTensor3` = `std::vector<MatrixXd>`:

- `K[α]` is a $3 \times 8$ matrix
- `K[α](β, a)` = coefficient of $S^\alpha S^\beta \lambda^a$

### 10.3 Fe bilinear: symmetric 6-parameter channels

`fill_channel` populates $K[\alpha](\beta, a_{\text{idx}})$ with **symmetry in $(\alpha,\beta)$**:

$$
K[\alpha](\beta, a) = K[\beta](\alpha, a)
$$

This is correct because the source and partner1 Fe are the **same site**, so $S_i^\alpha S_i^\beta = S_i^\beta S_i^\alpha$.

For each T-even Gell-Mann channel $a \in \{1, 3, 4, 6, 8\}$ (0-indexed: 0, 2, 3, 5, 7), there are 6 independent symmetric Fe bilinear components:

| Component | Tensor entries |
|-----------|---------------|
| `xx` | $K[0](0, a) = \text{sign} \cdot \texttt{xx}$ |
| `yy` | $K[1](1, a) = \text{sign} \cdot \texttt{yy}$ |
| `zz` | $K[2](2, a) = \text{sign} \cdot \texttt{zz}$ |
| `xy` | $K[0](1, a) = K[1](0, a) = \text{sign} \cdot \texttt{xy}$ |
| `xz` | $K[0](2, a) = K[2](0, a) = \text{sign} \cdot \texttt{xz}$ |
| `yz` | $K[1](2, a) = K[2](1, a) = \text{sign} \cdot \texttt{yz}$ |

### 10.4 A₁⁺ / A₂⁺ sector decomposition

`build_W_general(sign_A2)` constructs the full $W$ tensor:

- **A₁⁺ sector** ($\lambda_1, \lambda_3, \lambda_8$): `sign = +1` always
- **A₂⁺ sector** ($\lambda_4, \lambda_6$): `sign = sign_A2`

For a $\chi$-type bond: `sign_A2 = +1` → full $W$.  
For a $\chi^{\text{inv}}$-type bond: `sign_A2 = -1$ → A₂⁺ columns negated.

### 10.5 Legacy parameter compatibility

Legacy `u` and `v` parameters add to the general channels:

$$
\texttt{W1\_xx} \to \texttt{W1\_xx} - u_1, \quad \texttt{W1\_zz} \to \texttt{W1\_zz} + u_1 \quad (\text{i.e., } u_1 \text{ couples to } S_z^2 - S_x^2)
$$
$$
\texttt{W4\_xz} \to \texttt{W4\_xz} + v_4 \quad (\text{i.e., } v_4 \text{ couples to } 2\,S_x\,S_z)
$$

### 10.6 Orbit scaling

$$
W_{\text{orbit}\,k} = s_k^{(W)} \cdot W_{\text{base}}
$$

**Parameters**: `W{1,3,4,6,8}_{xx,yy,zz,xy,xz,yz}` (30 general), `u{1,3,8}`, `v{4,6}` (5 legacy), `W_orbit{1,2,3,4}_scale` (4 scales).  
**Total on-site trilinear**: 30 independent couplings + 4 orbit scales.

### 10.7 Bond topology

32 bonds (identical topology to the bilinear $\chi$ bonds), 8 per Fe site, 2 per orbit per Fe.

### 10.8 Three stored copies per bond

During bond initialization, each stored bond $K[\alpha](\beta, a)$ with `source=Fe_i, partner1=Fe_i, partner2=Tm_j` generates:

1. **Source Fe$_i$**: stores $K_{\alpha\beta a}$ with partners = (Fe$_i$, Tm$_j$)  
2. **Partner1 Fe$_i$**: stores $K_{\beta\alpha a}$ with partners = (Fe$_i$, Tm$_j$) — since source = partner1 for on-site, this doubles the entry on Fe$_i$  
3. **Partner2 Tm$_j$**: stores $K_{a,\alpha\beta}$ (cyclic transpose) with partners = (Fe$_i$, Fe$_i$)

In code: $K_{\text{bac}}[b](\alpha, a) = K[\alpha](b, a)$ and $K_{\text{cab}}[a](\alpha, \beta) = K[\alpha](\beta, a)$.

### 10.9 EOM contribution

For SU(2) site $i$ (from `get_local_field_SU2_flat`):

$$
H_\alpha^{(\text{Fe},i)} \mathrel{+}= \sum_{\beta,a} T[\alpha](\beta, a)\, S_{p_1}^\beta\, \lambda_{p_2}^a
$$

For SU(3) site $j$ (from `get_local_field_SU3_flat`):

$$
H_a^{(\text{Tm},j)} \mathrel{+}= \sum_{\alpha,\beta} T[a](\alpha, \beta)\, S_{p_1}^\alpha\, S_{p_2}^\beta
$$

where $T$ is the appropriately transposed tensor stored on each site.

---

## 11. Inter-Site Trilinear: $V$ Coupling ($\mathcal{H}_V$)

### 11.1 Definition

$$
\mathcal{H}_V = \frac{1}{3}\sum_{\text{bonds}} \sum_{\alpha,\beta=1}^{3}\sum_{a=1}^{8} V^{\alpha\beta a}\, S_i^\alpha\, S_{i'}^\beta\, \lambda_j^a
$$

Here $i$ and $i'$ are **different** Fe sites forming a c-axis NN pair, and $j$ is a Tm neighbor of Fe$_i$.

### 11.2 Symmetric part (30 parameters)

Same structure as $W$: `fill_channel` with `TrilinearChannel` gives $V[\alpha](\beta, a) = V[\beta](\alpha, a)$.

**Parameters**: `V{1,3,4,6,8}_{xx,yy,zz,xy,xz,yz}` (30 general), `w{1,3,8,4,6}` (5 legacy).

### 11.3 Antisymmetric part (15 parameters, DM-like)

Since $i \neq i'$, the antisymmetric Fe bilinear $S_i^\alpha S_{i'}^\beta - S_i^\beta S_{i'}^\alpha$ is **nonzero**. `fill_anti_channel` adds:

$$
V[\alpha](\beta, a) \mathrel{+}= \text{sign} \cdot A_{\alpha\beta}, \qquad V[\beta](\alpha, a) \mathrel{-}= \text{sign} \cdot A_{\alpha\beta}
$$

so the antisymmetric part satisfies $V[\alpha](\beta, a) = -V[\beta](\alpha, a)$.

| Component | Tensor entries |
|-----------|---------------|
| `Axy` | $V[0](1, a) \mathrel{+}= \text{sign} \cdot \texttt{Axy}$, $\;V[1](0, a) \mathrel{-}= \text{sign} \cdot \texttt{Axy}$ |
| `Axz` | $V[0](2, a) \mathrel{+}= \text{sign} \cdot \texttt{Axz}$, $\;V[2](0, a) \mathrel{-}= \text{sign} \cdot \texttt{Axz}$ |
| `Ayz` | $V[1](2, a) \mathrel{+}= \text{sign} \cdot \texttt{Ayz}$, $\;V[2](1, a) \mathrel{-}= \text{sign} \cdot \texttt{Ayz}$ |

Same A₁⁺/A₂⁺ sign structure as the symmetric part: the inversion acts on the Tm Gell-Mann generator, not on the Fe bilinear ordering, so:

- A₁⁺ ($\lambda_1, \lambda_3, \lambda_8$): `sign = +1` always
- A₂⁺ ($\lambda_4, \lambda_6$): `sign = sign_A2 = ±1`

**Parameters**: `V{1,3,4,6,8}_A{xy,xz,yz}` (15 antisymmetric), `V_orbit{1,2,3,4}_scale` (4 scales).

### 11.4 Combined tensor

The stored $V[\alpha](\beta, a)$ in total is:

$$
V[\alpha](\beta, a) = V^{(\text{sym})}_{\alpha\beta,a} + V^{(\text{anti})}_{\alpha\beta,a}
$$

where $V^{(\text{sym})}_{\alpha\beta,a} = V^{(\text{sym})}_{\beta\alpha,a}$ and $V^{(\text{anti})}_{\alpha\beta,a} = -V^{(\text{anti})}_{\beta\alpha,a}$.

**Total inter-site trilinear**: $30 + 15 = 45$ independent couplings + 4 orbit scales.

### 11.5 Transposition behavior (partner Fe)

Bond initialization creates $K_{\text{bac}}[\beta](\alpha, a) = V[\alpha](\beta, a)$, which is the **transpose** of the first two Fe indices.

For the combined tensor:
- Symmetric part: $K_{\text{bac}} = V^{(\text{sym})}$ (unchanged)
- Antisymmetric part: $K_{\text{bac}} = -V^{(\text{anti})}$ (sign flip)

So from Fe$_i$'s viewpoint the coupling is $V^{(\text{sym})} + V^{(\text{anti})}$, while from Fe$_{i'}$'s viewpoint it is $V^{(\text{sym})} - V^{(\text{anti})}$.

### 11.6 C-axis NN pairs

| Fe source | Fe partner | Offsets |
|-----------|-----------|---------|
| Fe₀ | Fe₃ | $(0,0,0)$, $(0,0,1)$ |
| Fe₁ | Fe₂ | $(0,0,0)$, $(0,0,1)$ |
| Fe₂ | Fe₁ | $(0,0,0)$, $(0,0,-1)$ |
| Fe₃ | Fe₀ | $(0,0,0)$, $(0,0,-1)$ |

### 11.7 Bond topology

64 bonds total = 32 Fe-Tm bonds × 2 c-axis Fe NN offsets. Each Fe site has 16 inter-site trilinear bonds (4 orbits × 4 bonds/orbit).

---

## 12. Equations of Motion

### 12.1 SU(2) Fe dynamics (Landau-Lifshitz)

$$
\frac{d\mathbf{S}_i}{dt} = \mathbf{H}_i \times \mathbf{S}_i
$$

where the local field $\mathbf{H}_i = -\partial\mathcal{H}/\partial\mathbf{S}_i$ collects all contributions:

$$
H_\alpha^{(i)} = -h_\alpha + 2\sum_\beta K_{\alpha\beta} S_i^\beta + \sum_{n,\beta} J_{in}^{\alpha\beta} S_n^\beta + \sum_{n,a} \chi_{in}^{\alpha a} \lambda_n^a + \sum_{\text{tri}} \sum_{\beta,a} T[\alpha](\beta,a)\, S_{p_1}^\beta\, \lambda_{p_2}^a
$$

In code (`get_local_field_SU2_flat`): accumulates field, on-site ($2A S$), bilinear, mixed bilinear, pure trilinear, and mixed trilinear contributions, then subtracts the time-dependent drive.

### 12.2 SU(3) Tm dynamics (Bloch)

$$
\frac{d\lambda_i^a}{dt} = \sum_{b,c} f_{abc}\, H_b^{(i)}\, \lambda_i^c \;-\; \Gamma_a\left(\lambda_i^a - \lambda_i^{a,\text{eq}}\right)
$$

where $f_{abc}$ are the SU(3) Gell-Mann structure constants, and $\Gamma_a$ are the Bloch damping rates (`damping_rates_SU3`).

The local field $H_a^{(j)}$ collects:

$$
H_a^{(j)} = -\left(\alpha\,\delta_{a3} + \beta\,\delta_{a8} + g_{\text{ratio}} B_a^{(\mu)}\right) + 2\sum_b A_{ab}\,\lambda_j^b + \sum_{n,b} J_{jn}^{ab}\,\lambda_n^b + \sum_{n,\alpha} \chi_{jn}^{a\alpha}\,S_n^\alpha + \sum_{\text{tri}} \sum_{\alpha,\beta} T[a](\alpha,\beta)\,S_{p_1}^\alpha\,S_{p_2}^\beta
$$

---

## 13. Energy Counting

### 13.1 Bilinear terms

Factor $\frac{1}{2}$ for both SU(2)–SU(2) and SU(3)–SU(3) bilinear, and also for mixed SU(2)–SU(3) bilinear. Each bond is stored with both source and partner having their own interaction entry, so the $\frac{1}{2}$ in `total_energy` avoids double counting.

### 13.2 Trilinear terms

Factor $\frac{1}{3}$ for all trilinear (pure and mixed). Each bond creates 3 stored copies (one per leg), so summing over all sites with weight $\frac{1}{3}$ recovers the correct energy.

In code (`total_energy_SU2`, `total_energy_SU3`):

```
energy += (1.0 / 3.0) * spin(a) * temp
```

### 13.3 MC energy difference

For an SU(2) update at site $i$:

- **Bilinear**: $\Delta E = (S_i^{\text{new}} - S_i^{\text{old}}) \cdot J \cdot S_{\text{partner}}$ (linear in the changed spin)
- **Mixed trilinear**: multiplicity factor $1 + \delta(p_1, i)$ accounts for bonds where both Fe legs coincide (on-site W gives multiplicity 2; inter-site V gives multiplicity 1 when $p_1 \neq i$)
- **Pure trilinear**: multiplicity = $1 + \delta(p_1,i) + \delta(p_2,i)$

For an SU(3) update at site $j$:

- **Mixed trilinear**: multiplicity = 1 (Fe spins don't change when updating a Tm site, so the two Fe partners in $K_{cab}[a](\alpha,\beta) S^\alpha S^\beta$ are constant).

---

## 14. Complete Parameter List

### Fe–Fe sector (6 params)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `J1ab` | 4.74 | NN exchange in ab-plane |
| `J1c` | 5.15 | NN exchange along c-axis |
| `J2ab` | 0.15 | NNN exchange in ab-plane |
| `J2c` | 0.30 | NNN exchange along c-axis |
| `D1` | 0.12 | DM interaction (y-component) |
| `D2` | 0.0 | DM interaction (x-component) |

### Fe single-ion anisotropy (3 params)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Ka` | −0.16221 | Anisotropy along $a$ |
| `Kb` | 0.0 | Anisotropy along $b$ |
| `Kc` | −0.18318 | Anisotropy along $c$ |

### Tm CEF (4 params)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `e1` | 0.97 | CEF singlet splitting |
| `e2` | 3.97 | CEF singlet splitting |
| `tm_alpha_scale` | 1.0 | Scale for $\alpha = e_1$ |
| `tm_beta_scale` | 1.0 | Scale for $\beta$ |

### Tm magnetic moment (9 params)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mu_2x` | 0.0 | $\mu_{x,\lambda_2}$ |
| `mu_2y` | 0.0 | $\mu_{y,\lambda_2}$ |
| `mu_2z` | 5.264 | $\mu_{z,\lambda_2}$ |
| `mu_5x` | 2.3915 | $\mu_{x,\lambda_5}$ |
| `mu_5y` | −2.7866 | $\mu_{y,\lambda_5}$ |
| `mu_5z` | 0.0 | $\mu_{z,\lambda_5}$ |
| `mu_7x` | 0.9128 | $\mu_{x,\lambda_7}$ |
| `mu_7y` | 0.4655 | $\mu_{y,\lambda_7}$ |
| `mu_7z` | 0.0 | $\mu_{z,\lambda_7}$ |

### Tm Zeeman (1 param)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_ratio_tm` | 7/12 | $g_{\text{Tm}}/g_{\text{Fe}}$ |

### Tm–Tm bilinear (8 params)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Jtm_1`…`Jtm_8` | 0.0 | Diagonal Gell-Mann coupling |

### Fe–Tm bilinear $\chi$ (9 + 4 params)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chi{2,5,7}{x,y,z}` | 0.0 | 9 bilinear couplings |
| `chi_orbit{1,2,3,4}_scale` | 1.0 | 4 orbit scales |

### On-site trilinear $W$ (30 + 5 legacy + 4 params)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `W{1,3,4,6,8}_{xx,yy,zz,xy,xz,yz}` | 0.0 | 30 general symmetric |
| `u{1,3,8}` | 0.0 | 3 legacy $S_z^2 - S_x^2$ |
| `v{4,6}` | 0.0 | 2 legacy $S_x S_z$ |
| `W_orbit{1,2,3,4}_scale` | 1.0 | 4 orbit scales |

### Inter-site trilinear $V$ (45 + 5 legacy + 4 params)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `V{1,3,4,6,8}_{xx,yy,zz,xy,xz,yz}` | 0.0 | 30 general symmetric |
| `V{1,3,4,6,8}_A{xy,xz,yz}` | 0.0 | 15 antisymmetric (DM-like) |
| `w{1,3,8,4,6}` | 0.0 | 5 legacy (additive to symmetric) |
| `V_orbit{1,2,3,4}_scale` | 1.0 | 4 orbit scales |

### External field (direction + magnitude)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `field_strength` | — | $h$ magnitude |
| `field_direction` | — | $\hat{n}$ unit vector |

---

## 15. Summary of Bond Counts

| Interaction | Type | Bonds stored | Effective bonds | Factor |
|-------------|------|-------------|-----------------|--------|
| Fe–Fe NN ab-plane | bilinear | 8 (per direction) | 4 per orbit | $\frac{1}{2}$ |
| Fe–Fe NN c-axis | bilinear | 4 | 2 | $\frac{1}{2}$ |
| Fe–Fe NNN | bilinear | 12 + 16 | — | $\frac{1}{2}$ |
| Tm–Tm NN | bilinear | 8 | 4 per pair | $\frac{1}{2}$ |
| Fe–Tm $\chi$ | mixed bilinear | 32 | 32 | $\frac{1}{2}$ |
| On-site $W$ | mixed trilinear | 32 (→ 96 stored) | 32 | $\frac{1}{3}$ |
| Inter-site $V$ | mixed trilinear | 64 (→ 192 stored) | 64 | $\frac{1}{3}$ |

---

## 16. Summary of Independent Coupling Parameters

| Sector | Count |
|--------|-------|
| Fe–Fe bilinear | 6 |
| Fe SIA | 3 |
| Tm CEF | 4 |
| Tm moment / Zeeman | 10 |
| Tm–Tm bilinear | 8 |
| Fe–Tm bilinear $\chi$ | 9 + 4 scales |
| On-site trilinear $W$ | 30 + 4 scales (+ 5 legacy aliases) |
| Inter-site trilinear $V$ | 45 + 4 scales (+ 5 legacy aliases) |
| **Total** | **84 couplings + 12 orbit scales + 31 other** |
