import numpy as np
from scipy.optimize import minimize
from luttinger_tisza import (
    create_honeycomb_lattice,
    get_bond_vectors,
    calculate_energy_from_matrices,
    visualize_spins,
)


class DoubleQMeronAntimeron:
    """
    Double-Q meron-antimeron ansatz on the BCAO honeycomb lattice.

    Ansatz (per sublattice s = A,B):
        m_perp(r)_s = sum_j a_{s,j} sin(Q_j·r + theta_{s,j}) e_j
        m_z(r)_s = m0_s + sum_j b_{s,j} sin(Q_j·r + theta_{s,j})
        S_s(r) = m_perp + m_z e_z with |S_s(r)|=1 after normalization.

    Constraints: Q1 ⟂ Q2 in the plane, e1 ⟂ e2 ⟂ ez (from Euler angles).
    BCAO parameters follow the C++ convention: [J1xy, J1z, D, E, F, G, J3xy, J3z].
    """

    eta_small = 1e-9
    min_q1a, max_q1a = (0.0, 0.5 - eta_small)
    min_q1b, max_q1b = (0.0, 0.5)
    min_q2mag, max_q2mag = (0.0, 0.5)
    min_m0, max_m0 = (-1.0, 1.0)
    min_amp, max_amp = (0.0, 1.0)
    min_angle, max_angle = (0.0, (2.0 - eta_small) * np.pi)

    parameter_bounds = [
        (min_q1a, max_q1a),  # q1 coefficient along b1
        (min_q1b, max_q1b),  # q1 coefficient along b2
        (min_q2mag, max_q2mag),  # |Q2| scaled by |b1|
        (min_m0, max_m0),  # m0_A
        (min_m0, max_m0),  # m0_B
        (min_amp, max_amp),  # a1_perp_A
        (min_amp, max_amp),  # a2_perp_A
        (min_amp, max_amp),  # a1_perp_B
        (min_amp, max_amp),  # a2_perp_B
        (min_amp, max_amp),  # b1_z_A
        (min_amp, max_amp),  # b2_z_A
        (min_amp, max_amp),  # b1_z_B
        (min_amp, max_amp),  # b2_z_B
        (min_angle, max_angle),  # theta1_A
        (min_angle, max_angle),  # theta2_A
        (min_angle, max_angle),  # theta1_B
        (min_angle, max_angle),  # theta2_B
        (min_angle, max_angle),  # phi_A
        (min_angle, max_angle),  # theta_A
        (min_angle, max_angle),  # psi_A
        (min_angle, max_angle),  # phi_B
        (min_angle, max_angle),  # theta_B
        (min_angle, max_angle),  # psi_B
    ]

    def __init__(
        self,
        L=4,
        J=None,
        B_field=None,
    ):
        if J is None:
            J = [-7.6, -1.2, 0.1, -0.1, 0.0, 0.0, 2.5, -0.85]
        if B_field is None:
            B_field = np.array([0.0, 0.0, 0.0])

        self.L = L
        self.J = J
        self.J1xy = J[0]
        self.J1z = J[1]
        self.D = J[2]
        self.E = J[3]
        self.F = J[4]
        self.G = J[5]
        self.J3xy = J[6]
        self.J3z = J[7]
        self.B_field = B_field

        (
            self.positions,
            self.NN,
            self.NN_bonds,
            self.NNN,
            self.NNNN,
            self.a1,
            self.a2,
        ) = create_honeycomb_lattice(L)
        self.J1, self.J2_mat, self.J3_mat = self._construct_BCAO_interaction_matrices()
        self.nn_vectors, self.nnn_vectors, self.nnnn_vectors = get_bond_vectors(
            self.a1, self.a2
        )

        self.b1 = 2 * np.pi * np.array([1.0, -1.0 / np.sqrt(3.0)])
        self.b2 = 2 * np.pi * np.array([0.0, 2.0 / np.sqrt(3.0)])

        self.opt_params, self.opt_energy = self.find_minimum_energy()

    @staticmethod
    def _rotated_basis(phi, theta, psi):
        rot = np.array(
            [
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
            ]
        )
        e1 = rot[:, 0]
        e2 = rot[:, 1]
        ez = rot[:, 2]
        return e1, e2, ez

    @staticmethod
    def _uniform(a, b):
        return (b - a) * np.random.random() + a

    def _construct_BCAO_interaction_matrices(self):
        j1z_mat = np.array(
            [[self.J1xy + self.D, self.E, self.F], [self.E, self.J1xy - self.D, self.G], [self.F, self.G, self.J1z]]
        )

        cos_2pi3 = np.cos(2.0 * np.pi / 3.0)
        sin_2pi3 = np.sin(2.0 * np.pi / 3.0)
        u_2pi_3 = np.array([[cos_2pi3, sin_2pi3, 0.0], [-sin_2pi3, cos_2pi3, 0.0], [0.0, 0.0, 1.0]])

        j1x_mat = u_2pi_3 @ j1z_mat @ u_2pi_3.T
        j1y_mat = u_2pi_3.T @ j1z_mat @ u_2pi_3

        j1 = [j1x_mat, j1y_mat, j1z_mat]
        j2_mat = np.zeros((3, 3))
        j3_mat = np.array([[self.J3xy, 0.0, 0.0], [0.0, self.J3xy, 0.0], [0.0, 0.0, self.J3z]])
        return j1, j2_mat, j3_mat

    def _build_Q_vectors(self, q1a, q1b, q2_mag):
        q1 = q1a * self.b1 + q1b * self.b2
        q1_norm = np.linalg.norm(q1)
        if q1_norm < 1e-12:
            q1_dir = np.array([1.0, 0.0])
            q1_norm = np.linalg.norm(self.b1)
        else:
            q1_dir = q1 / q1_norm
        q2_dir = np.array([-q1_dir[1], q1_dir[0]])
        q2 = q2_mag * np.linalg.norm(self.b1) * q2_dir
        return q1, q2

    def _spin_field_from_params(self, params):
        (
            q1a,
            q1b,
            q2_mag,
            m0_a,
            m0_b,
            a1_perp_a,
            a2_perp_a,
            a1_perp_b,
            a2_perp_b,
            b1_z_a,
            b2_z_a,
            b1_z_b,
            b2_z_b,
            theta1_a,
            theta2_a,
            theta1_b,
            theta2_b,
            phi_a,
            theta_a,
            psi_a,
            phi_b,
            theta_b,
            psi_b,
        ) = params

        q1, q2 = self._build_Q_vectors(q1a, q1b, q2_mag)
        e1_a, e2_a, ez_a = self._rotated_basis(phi_a, theta_a, psi_a)
        e1_b, e2_b, ez_b = self._rotated_basis(phi_b, theta_b, psi_b)

        spins = np.zeros((len(self.positions), 3))
        for idx, pos in enumerate(self.positions):
            phase1 = np.dot(q1, pos)
            phase2 = np.dot(q2, pos)
            if idx % 2 == 0:
                spin = (
                    a1_perp_a * np.sin(phase1 + theta1_a) * e1_a
                    + a2_perp_a * np.sin(phase2 + theta2_a) * e2_a
                    + (m0_a + b1_z_a * np.sin(phase1 + theta1_a) + b2_z_a * np.sin(phase2 + theta2_a)) * ez_a
                )
            else:
                spin = (
                    a1_perp_b * np.sin(phase1 + theta1_b) * e1_b
                    + a2_perp_b * np.sin(phase2 + theta2_b) * e2_b
                    + (m0_b + b1_z_b * np.sin(phase1 + theta1_b) + b2_z_b * np.sin(phase2 + theta2_b)) * ez_b
                )
            norm = np.linalg.norm(spin)
            spins[idx] = spin / norm if norm > 0 else ez_a
        return spins

    def energy_per_site(self, params):
        spins = self._spin_field_from_params(params)
        exchange = calculate_energy_from_matrices(
            spins, self.NN, self.NN_bonds, self.NNN, self.NNNN, self.J1, self.J2_mat, self.J3_mat
        )
        zeeman = -np.mean(spins @ self.B_field)
        return float(np.real(exchange + zeeman))

    def generate_spin_configuration(self, params=None):
        if params is None:
            params = self.opt_params
        return self._spin_field_from_params(params)

    def real_space_energy(self, spins=None):
        if spins is None:
            spins = self.generate_spin_configuration()
        exchange = calculate_energy_from_matrices(
            spins, self.NN, self.NN_bonds, self.NNN, self.NNNN, self.J1, self.J2_mat, self.J3_mat
        )
        zeeman = -np.mean(spins @ self.B_field)
        return float(np.real(exchange + zeeman))

    def calculate_magnetization(self, spins):
        total = np.mean(spins, axis=0)
        spins_a = spins[::2]
        spins_b = spins[1::2]
        mag_a = np.mean(spins_a, axis=0)
        mag_b = np.mean(spins_b, axis=0)
        staggered = mag_a - mag_b
        return {
            "total": total,
            "total_magnitude": np.linalg.norm(total),
            "A": mag_a,
            "B": mag_b,
            "A_magnitude": np.linalg.norm(mag_a),
            "B_magnitude": np.linalg.norm(mag_b),
            "staggered": staggered,
            "staggered_magnitude": np.linalg.norm(staggered),
        }

    def find_minimum_energy(self, n_restarts=3, tol_primary=1e-4, tol_refine=1e-6):
        opt_params = None
        opt_energy = np.inf
        print(f"Starting optimization with {n_restarts} restarts...")
        for restart_idx in range(n_restarts):
            print(f"  Restart {restart_idx+1}/{n_restarts}...", end=" ", flush=True)
            guess = [
                self._uniform(self.min_q1a, self.max_q1a),
                self._uniform(self.min_q1b, self.max_q1b),
                self._uniform(self.min_q2mag, self.max_q2mag),
                self._uniform(self.min_m0, self.max_m0),
                self._uniform(self.min_m0, self.max_m0),
                self._uniform(self.min_amp, self.max_amp),
                self._uniform(self.min_amp, self.max_amp),
                self._uniform(self.min_amp, self.max_amp),
                self._uniform(self.min_amp, self.max_amp),
                self._uniform(self.min_amp, self.max_amp),
                self._uniform(self.min_amp, self.max_amp),
                self._uniform(self.min_amp, self.max_amp),
                self._uniform(self.min_amp, self.max_amp),
                self._uniform(self.min_angle, self.max_angle),
                self._uniform(self.min_angle, self.max_angle),
                self._uniform(self.min_angle, self.max_angle),
                self._uniform(self.min_angle, self.max_angle),
                self._uniform(self.min_angle, self.max_angle),
                self._uniform(self.min_angle, self.max_angle),
                self._uniform(self.min_angle, self.max_angle),
                self._uniform(self.min_angle, self.max_angle),
                self._uniform(self.min_angle, self.max_angle),
                self._uniform(self.min_angle, self.max_angle),
            ]
            primary = minimize(
                self.energy_per_site,
                x0=guess,
                bounds=self.parameter_bounds,
                method="Nelder-Mead",
                tol=tol_primary,
                options={'maxiter': 5000, 'maxfev': 10000}
            )
            print(f"E={primary.fun:.4f}")
            if primary.fun < opt_energy:
                opt_params = primary.x
                opt_energy = primary.fun

        print(f"Refining best solution (E={opt_energy:.4f})...", flush=True)
        refined = minimize(
            self.energy_per_site,
            x0=opt_params,
            bounds=self.parameter_bounds,
            method="L-BFGS-B",
            tol=tol_refine,
            options={'maxiter': 1000}
        )
        print(f"Optimization complete. Final E={refined.fun:.6f}")
        return refined.x, float(refined.fun)


if __name__ == "__main__":
    L = 8
    J = [-7.6, -1.2, 0.1, -0.1, 0.0, 0.0, 2.5, -0.85]
    model = DoubleQMeronAntimeron(L=L, J=J)
    spins = model.generate_spin_configuration()
    mags = model.calculate_magnetization(spins)

    print("Double-Q meron-antimeron ansatz")
    print(f"Optimal energy per site: {model.opt_energy:.6f}")
    print(f"Total magnetization: {mags['total']}")

    visualize_spins(model.positions, spins, L)
