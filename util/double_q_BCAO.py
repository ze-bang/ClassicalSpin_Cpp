import numpy as np
from scipy.optimize import minimize
from luttinger_tisza import create_honeycomb_lattice, construct_interaction_matrices, get_bond_vectors, visualize_spins

"""Double-Q variational ansatz for BCAO on the honeycomb lattice.

This is a minimal extension of `single_q_BCAO.SingleQ` where the spiral
part of the spin texture is allowed to contain two independent Fourier
components with wave–vectors

    Q1_vec = q1 * b1   (along first reciprocal basis vector)
    Q2_vec = q2 * b2   (along second reciprocal basis vector)

The real-space spin on site i belonging to sublattice s=A,B is taken as

    S_i^s = m_s * e_z^s
             + sqrt(1-m_s^2) * [
                 a1_s * e_x^s cos(Q1·r_i) + b1_s * e_y^s sin(Q1·r_i)
               + a2_s * e_x^s cos(Q2·r_i) + b2_s * e_y^s sin(Q2·r_i)
             ]

with separate amplitudes for the two Q-components on each sublattice.
The energy is evaluated from the microscopic BCAO Hamiltonian by
explicit real‑space summation over bonds (same lattice and couplings as
in the single‑Q code).
"""


class DoubleQ:
    # Parameter ranges
    eta_small = 1e-9
    min_q1, max_q1 = (0.0, 0.5 - eta_small)
    min_q2, max_q2 = (0.0, 0.5)

    # uniform magnetization fraction on each sublattice
    min_mA, max_mA = (0.0, 1.0)
    min_mB, max_mB = (0.0, 1.0)

    # Euler angles for local frames
    min_phi, max_phi = (0.0, (2.0 - eta_small) * np.pi)
    min_theta, max_theta = (0.0, (2.0 - eta_small) * np.pi)
    min_psi, max_psi = (0.0, (2.0 - eta_small) * np.pi)

    # amplitudes for double-Q on each sublattice; restricted so that
    # |S| ≈ 1 but we enforce this only approximately and re-normalise.
    min_a, max_a = (-1.0, 1.0)

    parameter_bounds = [
        (min_q1, max_q1),  # 0: q1 along b1
        (min_q2, max_q2),  # 1: q2 along b2
        (min_mA, max_mA),  # 2: m_A
        (min_mB, max_mB),  # 3: m_B
        (min_phi, max_phi),   # 4: phi_A
        (min_theta, max_theta),  # 5: theta_A
        (min_psi, max_psi),   # 6: psi_A
        (min_phi, max_phi),   # 7: phi_B
        (min_theta, max_theta),  # 8: theta_B
        (min_psi, max_psi),   # 9: psi_B
        # amplitudes for Q1 on A sublattice
        (min_a, max_a),  # 10: a1_A (ex cos Q1)
        (min_a, max_a),  # 11: b1_A (ey sin Q1)
        # amplitudes for Q2 on A sublattice
        (min_a, max_a),  # 12: a2_A
        (min_a, max_a),  # 13: b2_A
        # amplitudes for Q1 on B sublattice
        (min_a, max_a),  # 14: a1_B
        (min_a, max_a),  # 15: b1_B
        # amplitudes for Q2 on B sublattice
        (min_a, max_a),  # 16: a2_B
        (min_a, max_a),  # 17: b2_B
    ]

    def __init__(self, L=4, J=None, B_field=None):
        if J is None:
            # Default BCAO parameters, same convention as single_Q code
            J = [-7.6, -1.2, 0.1, -0.1, 0.0, 0.0, 2.5, -0.85]
        if B_field is None:
            B_field = np.array([0.0, 0.0, 0.0])

        self.L = L
        self.J = J
        self.J1xy, self.J1z, self.D, self.E, self.F, self.G, self.J3xy, self.J3z = J
        self.B_field = np.asarray(B_field, dtype=float)

        # lattice and geometry
        (
            self.positions,
            self.NN,
            self.NN_bonds,
            self.NNN,
            self.NNNN,
            self.a1,
            self.a2,
        ) = create_honeycomb_lattice(L)

        self.J1_list, self.J2_mat, self.J3_mat = self._construct_BCAO_interaction_matrices()
        self.nn_vectors, self.nnn_vectors, self.nnnn_vectors = get_bond_vectors(self.a1, self.a2)

        # reciprocal lattice (same convention as in single_q_BCAO)
        self.b1 = 2.0 * np.pi * np.array([1.0, -1.0 / np.sqrt(3.0)])
        self.b2 = 2.0 * np.pi * np.array([0.0, 2.0 / np.sqrt(3.0)])

        # optimise variational parameters
        self.opt_params, self.opt_energy = self._find_minimum_energy()

    # --- local frame utilities -------------------------------------------------

    @staticmethod
    def rotated_basis(phi, theta, psi):
        """Return orthonormal basis (ex, ey, ez) defined by Euler angles.

        Same convention as in `SingleQ.RotatedBasis`.
        """
        RotMat = np.array([
            [
                np.cos(theta) * np.cos(phi) * np.cos(psi)
                - np.sin(phi) * np.sin(psi),
                -np.cos(psi) * np.sin(phi)
                - np.cos(theta) * np.cos(phi) * np.sin(psi),
                np.cos(phi) * np.sin(theta),
            ],
            [
                np.cos(theta) * np.cos(psi) * np.sin(phi)
                + np.cos(phi) * np.sin(psi),
                np.cos(phi) * np.cos(psi)
                - np.cos(theta) * np.sin(phi) * np.sin(psi),
                np.sin(theta) * np.sin(phi),
            ],
            [
                -np.cos(psi) * np.sin(theta),
                np.sin(theta) * np.sin(psi),
                np.cos(theta),
            ],
        ])
        ex = RotMat[:, 0]
        ey = RotMat[:, 1]
        ez = RotMat[:, 2]
        return ex, ey, ez

    @staticmethod
    def _uniform(a, b):
        return (b - a) * np.random.random() + a

    # --- interactions ----------------------------------------------------------

    def _construct_BCAO_interaction_matrices(self):
        J1z_mat = np.array(
            [
                [self.J1xy + self.D, self.E, self.F],
                [self.E, self.J1xy - self.D, self.G],
                [self.F, self.G, self.J1z],
            ]
        )

        cos_2pi3 = np.cos(2.0 * np.pi / 3.0)
        sin_2pi3 = np.sin(2.0 * np.pi / 3.0)
        U = np.array(
            [
                [cos_2pi3, sin_2pi3, 0.0],
                [-sin_2pi3, cos_2pi3, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        J1x_mat = U @ J1z_mat @ U.T
        J1y_mat = U.T @ J1z_mat @ U
        J1_list = [J1x_mat, J1y_mat, J1z_mat]

        J2_mat = np.zeros((3, 3))
        J3_mat = np.array(
            [
                [self.J3xy, 0.0, 0.0],
                [0.0, self.J3xy, 0.0],
                [0.0, 0.0, self.J3z],
            ]
        )
        return J1_list, J2_mat, J3_mat

    # --- variational spin texture ---------------------------------------------

    def _build_spins(self, params):
        (
            q1,
            q2,
            mA,
            mB,
            phiA,
            thetaA,
            psiA,
            phiB,
            thetaB,
            psiB,
            a1A,
            b1A,
            a2A,
            b2A,
            a1B,
            b1B,
            a2B,
            b2B,
        ) = params

        Q1_vec = q1 * self.b1
        Q2_vec = q2 * self.b2

        exA, eyA, ezA = self.rotated_basis(phiA, thetaA, psiA)
        exB, eyB, ezB = self.rotated_basis(phiB, thetaB, psiB)

        N = len(self.positions)
        spins = np.zeros((N, 3), dtype=float)

        for i, r in enumerate(self.positions):
            phase1 = Q1_vec.dot(r)
            phase2 = Q2_vec.dot(r)

            if i % 2 == 0:  # A sublattice
                m = mA
                ex, ey, ez = exA, eyA, ezA
                a1, b1, a2, b2 = a1A, b1A, a2A, b2A
            else:  # B sublattice
                m = mB
                ex, ey, ez = exB, eyB, ezB
                a1, b1, a2, b2 = a1B, b1B, a2B, b2B

            mod = (
                a1 * ex * np.cos(phase1)
                + b1 * ey * np.sin(phase1)
                + a2 * ex * np.cos(phase2)
                + b2 * ey * np.sin(phase2)
            )

            spin = m * ez + np.sqrt(max(0.0, 1.0 - m * m)) * mod
            # normalise to unit length to keep spins classical of length 1
            norm = np.linalg.norm(spin)
            if norm > 0:
                spin /= norm
            spins[i] = spin

        return spins

    # --- energy evaluation -----------------------------------------------------

    def energy_per_site(self, params):
        spins = self._build_spins(params)
        N = len(spins)

        # exchange part from NN and NNNN (J1_list and J3_mat), mirroring
        # the structure of the BCAO C++ code as closely as possible.
        E = 0.0

        # nearest neighbours (A-B). `NN_bonds` may store more than just a
        # simple integer; mirror the access pattern from the single-Q code by
        # taking only the first entry as the bond-type index.
        for (i, j, *rest), bond_info in zip(self.NN, self.NN_bonds):
            S_i = spins[i]
            S_j = spins[j]
            # support both scalar and tuple/list bond descriptors
            bond_type = bond_info[0] if isinstance(bond_info, (list, tuple, np.ndarray)) else bond_info
            Jmat = self.J1_list[int(bond_type)]
            E += S_i @ (Jmat @ S_j)

        # third neighbours (same sublattice). `NNNN` may also contain
        # additional metadata per bond; only first two entries are site
        # indices.
        for (i, j, *rest) in self.NNNN:
            S_i = spins[i]
            S_j = spins[j]
            E += S_i @ (self.J3_mat @ S_j)

        # Zeeman term
        if np.linalg.norm(self.B_field) > 0:
            for S in spins:
                E -= self.B_field.dot(S)

        # Each bond counted once; divide by number of sites for energy/site
        return E / float(N)

    # --- optimisation ---------------------------------------------------------

    def _find_minimum_energy(self, N_ITER=20, tol_first=1e-6, tol_second=1e-8):
        best_params = None
        best_E = np.inf

        for _ in range(N_ITER):
            guess = [
                self._uniform(self.min_q1, self.max_q1),
                self._uniform(self.min_q2, self.max_q2),
                self._uniform(self.min_mA, self.max_mA),
                self._uniform(self.min_mB, self.max_mB),
                self._uniform(self.min_phi, self.max_phi),
                self._uniform(self.min_theta, self.max_theta),
                self._uniform(self.min_psi, self.max_psi),
                self._uniform(self.min_phi, self.max_phi),
                self._uniform(self.min_theta, self.max_theta),
                self._uniform(self.min_psi, self.max_psi),
                self._uniform(self.min_a, self.max_a),
                self._uniform(self.min_a, self.max_a),
                self._uniform(self.min_a, self.max_a),
                self._uniform(self.min_a, self.max_a),
                self._uniform(self.min_a, self.max_a),
                self._uniform(self.min_a, self.max_a),
                self._uniform(self.min_a, self.max_a),
                self._uniform(self.min_a, self.max_a),
            ]

            res = minimize(
                self.energy_per_site,
                x0=guess,
                bounds=self.parameter_bounds,
                method="Nelder-Mead",
                tol=tol_first,
            )

            if res.fun < best_E:
                best_E = res.fun
                best_params = res.x

        # refine with gradient-based method starting from best guess
        res = minimize(
            self.energy_per_site,
            x0=best_params,
            bounds=self.parameter_bounds,
            method="L-BFGS-B",
            tol=tol_second,
        )
        return res.x, res.fun

    # --- helpers for analysis --------------------------------------------------

    def generate_spin_configuration(self):
        return self._build_spins(self.opt_params)

    def calculate_magnetization(self, spins):
        N = len(spins)
        total = np.mean(spins, axis=0)
        total_mag = np.linalg.norm(total)
        spins_A = spins[::2]
        spins_B = spins[1::2]
        mag_A = np.mean(spins_A, axis=0)
        mag_B = np.mean(spins_B, axis=0)
        mag_A_mag = np.linalg.norm(mag_A)
        mag_B_mag = np.linalg.norm(mag_B)
        staggered = mag_A - mag_B
        stag_mag = np.linalg.norm(staggered)
        return {
            "total": total,
            "total_magnitude": total_mag,
            "A": mag_A,
            "B": mag_B,
            "A_magnitude": mag_A_mag,
            "B_magnitude": mag_B_mag,
            "staggered": staggered,
            "staggered_magnitude": stag_mag,
        }


if __name__ == "__main__":
    L = 6

    # J = [-7.6, -1.2, 0.1, -0.1, 0.0, 0.0, 2.5, -0.85]
    # J = [-6.772, -1.887, 0.815, 1.292, -0.091, 0.627, 1.823, -0.157]
    J = [-6.646, -2.084, 0.675, 1.33, -1.516, 0.21, 1.697, 0.039]
    model = DoubleQ(L=L, J=J, B_field=np.array([0.0, 0.0, 0.0]))

    print("Optimal parameters (q1, q2, ...):")
    print(model.opt_params)
    print(f"Ground-state energy per site: {model.opt_energy:.6f}")

    spins = model.generate_spin_configuration()
    mags = model.calculate_magnetization(spins)
    print("Total magnetization:", mags["total"], "|M|=", mags["total_magnitude"])

    visualize_spins(model.positions, spins, L)
