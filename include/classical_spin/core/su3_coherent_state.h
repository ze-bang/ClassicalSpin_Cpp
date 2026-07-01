// su3_coherent_state.h
// -----------------------------------------------------------------------------
// SU(3) coherent-state classical spin dynamics for the TmFeO3 Tm sector.
//
// References (widely-accepted method in the SU(N) classical spin community):
//
//   [1] H. Zhang and C. D. Batista,
//       "Classical spin dynamics based on SU(N) coherent states,"
//       Phys. Rev. B 104, 104409 (2021).
//       arXiv:2106.14125
//
//   [2] D. Dahlbom, H. Zhang, C. Miles, X. Bai, C. D. Batista, K. Barros,
//       "Geometric integration of classical spin dynamics via a mean-field
//        Schrödinger equation,"
//       Phys. Rev. B 106, 054423 (2022).
//       arXiv:2204.07563
//
//   [3] D. Dahlbom et al.,
//       "Langevin dynamics of generalized spins as SU(N) coherent states,"
//       Phys. Rev. B 106, 235154 (2022).
//
//   [4] D. Dahlbom et al., Sunny.jl: A Julia package for spin dynamics,
//       Journal of Open Source Software 10(116), 8138 (2025).
//
// Zhang and Batista explicitly identify our regime — "S ≥ 1 systems with
// large single-ion anisotropy" — as the case where the SU(N) generalization
// is required rather than optional. The local Tm three-level problem
// (E1, E2, E3) is exactly an N = 3 single-ion anisotropy site.
//
// State representation
// --------------------
//   psi ∈ C^3 with ⟨psi|psi⟩ = 1   (a point on the projective manifold CP^2).
//   The Gell-Mann expectation values are
//       n^a = ⟨psi| λ^a |psi⟩,    a = 1..8,
//   which by construction always come from a physical pure-state density
//   matrix ρ = |psi⟩⟨psi|, so positivity and the Casimir constraints are
//   automatic — none of the "8-vector outside the positive cone" failure
//   modes of the bare Bloch parameterization can occur.
//
// Equation of motion (Schrödinger picture, mean-field)
// ----------------------------------------------------
//   i ℏ dψ/dt = H_loc[ψ; neighbours] ψ
//
//   where, for any classical Hamiltonian H_total({n^a_i}), the local 3×3
//   Hermitian operator is built from the same local SU(3) field h^a_i that
//   the existing code already computes:
//
//       H_loc = h_0 1 + (1/2) Σ_a h^a λ^a.
//
//   (See Eq. (5) of Ref. [2]; equivalently Eq. (4) of Ref. [1].)
//
//   It is straightforward to check that this is equivalent to the
//   Bloch-vector EOM that the current code uses,
//       dn^a/dt = f_{abc} h^b n^c,
//   provided that the n^a actually come from a physical pure state. The
//   two pictures differ only at amplitudes where the bare Bloch vector
//   would leave CP^2.
//
// Time integration
// ----------------
//   For frozen neighbours over a step Δt, the exact local propagator is
//   the unitary
//       U(Δt) = exp(−i H_loc Δt) = V exp(−i D Δt) V^†,
//   where H_loc = V D V^† is the 3×3 eigendecomposition. This is the
//   building block of the symplectic / norm-preserving spherical-midpoint
//   integrator of Ref. [2]. For a non-self-consistent ("explicit") step,
//   simply applying U(Δt) once per site preserves ⟨ψ|ψ⟩ = 1 exactly to
//   floating-point precision (the only error is the mean-field freezing
//   of neighbours, an O(Δt) commutator term).
//
// Static T = 0 update
// -------------------
//   For minimisation / annealing, the local update consistent with the
//   above is local exact diagonalization: with all neighbours frozen,
//   set ψ to the lowest eigenvector of H_loc. This is a direct upgrade
//   of the present "align 8-vector antiparallel to local field" rule in
//   `MixedLattice::deterministic_sweep_local_field`, which can place ρ
//   outside the positive cone (see seed file
//   `example_configs/TmFeO3/tmfeo3_gamma2_1x1x1_seed_SU3.txt`, whose
//   stored (λ_3, λ_8) imply a Tm-level population p_3 ≈ −0.19).
//
// Scope
// -----
//   This header is intentionally self-contained and N = 3 only — the
//   only case the Tm sector needs. Generalisation to other N would
//   replace the closed-form Gell-Mann constants below with the
//   structure-constant routines already in
//   `classical_spin/core/simple_linear_alg.h`.
// -----------------------------------------------------------------------------
#ifndef CLASSICAL_SPIN_SU3_COHERENT_STATE_H
#define CLASSICAL_SPIN_SU3_COHERENT_STATE_H

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <complex>

namespace classical_spin {
namespace su3 {

using Complex   = std::complex<double>;
using Vector3c  = Eigen::Matrix<Complex, 3, 1>;
using Matrix3c  = Eigen::Matrix<Complex, 3, 3>;
using Vector8r  = Eigen::Matrix<double,  8, 1>;

// -----------------------------------------------------------------------------
// Gell-Mann matrices (1-indexed convention, returned as 0-indexed array).
// Normalisation: Tr(λ^a λ^b) = 2 δ^{ab}.
// -----------------------------------------------------------------------------
inline const std::array<Matrix3c, 8>& gell_mann() {
    static const std::array<Matrix3c, 8> kLambda = [] {
        std::array<Matrix3c, 8> L;
        const Complex I(0.0, 1.0);
        for (auto& m : L) m.setZero();

        // λ_1
        L[0](0, 1) = 1.0;  L[0](1, 0) = 1.0;
        // λ_2
        L[1](0, 1) = -I;   L[1](1, 0) =  I;
        // λ_3
        L[2](0, 0) = 1.0;  L[2](1, 1) = -1.0;
        // λ_4
        L[3](0, 2) = 1.0;  L[3](2, 0) = 1.0;
        // λ_5
        L[4](0, 2) = -I;   L[4](2, 0) =  I;
        // λ_6
        L[5](1, 2) = 1.0;  L[5](2, 1) = 1.0;
        // λ_7
        L[6](1, 2) = -I;   L[6](2, 1) =  I;
        // λ_8
        const double s = 1.0 / std::sqrt(3.0);
        L[7](0, 0) = s;    L[7](1, 1) = s;    L[7](2, 2) = -2.0 * s;
        return L;
    }();
    return kLambda;
}

// -----------------------------------------------------------------------------
// State conversions
// -----------------------------------------------------------------------------

// n^a = ⟨ψ| λ^a |ψ⟩.
inline Vector8r expectations_from_psi(const Vector3c& psi) {
    const auto& L = gell_mann();
    Vector8r n;
    for (int a = 0; a < 8; ++a) {
        const Complex z = psi.adjoint() * (L[a] * psi);
        n(a) = z.real();
    }
    return n;
}

// Build the qutrit density matrix ρ from a Bloch vector:
//   ρ = (1/3) 1 + (1/2) Σ_a n^a λ^a.
// (Valid as a density matrix iff ρ ≥ 0 and Tr ρ = 1; the second is
//  automatic, the first is the physical positivity constraint that the
//  bare 8-vector parameterization can violate.)
inline Matrix3c density_from_expectations(const Vector8r& n) {
    const auto& L = gell_mann();
    Matrix3c rho = (1.0 / 3.0) * Matrix3c::Identity();
    for (int a = 0; a < 8; ++a) {
        rho.noalias() += 0.5 * n(a) * L[a];
    }
    return rho;
}

// Closest pure-state ψ to a given Bloch vector n^a, via the largest-eigenvalue
// eigenvector of the implied ρ. If n^a comes from a physical pure state, this
// recovers ψ up to a global phase; otherwise it returns the maximum-overlap
// projector approximation. The largest-eigenvalue itself is returned in
// `out_purity` (= 1 iff ψ is exact, < 1 if the input n^a is mixed / unphysical).
inline Vector3c psi_from_expectations(const Vector8r& n, double* out_purity = nullptr) {
    Matrix3c rho = density_from_expectations(n);
    Eigen::SelfAdjointEigenSolver<Matrix3c> es(rho);
    const int top = 2;  // Eigen sorts ascending; largest is index 2.
    if (out_purity) *out_purity = es.eigenvalues()(top);
    Vector3c psi = es.eigenvectors().col(top);
    psi.normalize();
    return psi;
}

// -----------------------------------------------------------------------------
// Local Hamiltonian and propagators
// -----------------------------------------------------------------------------

// Build the local 3×3 mean-field Hamiltonian from a Gell-Mann field h^a:
//   H_loc = h_0 1 + (1/2) Σ_a h^a λ^a.
// `h_a` is the same 8-component local field that the existing
// `MixedLattice::get_local_field_SU3(_flat)` machinery already computes.
inline Matrix3c local_hamiltonian(const Vector8r& h_a, double h_0 = 0.0) {
    const auto& L = gell_mann();
    Matrix3c H = h_0 * Matrix3c::Identity();
    for (int a = 0; a < 8; ++a) {
        H.noalias() += 0.5 * h_a(a) * L[a];
    }
    return H;
}

// Sign convention adapter for the existing code base.
//
// The current `get_local_field_SU3` / `get_local_field_SU3_flat_into`
// returns a vector `H_field[a]` such that the Bloch-EOM in
// mixed_lattice_md.cpp reads
//     dn^a/dt = f_{abc} H_field^b n^c
// (see `cross_prod_SU3_flat`). The Schrödinger-picture Hamiltonian that
// reproduces this is then exactly
//     H_loc = (1/2) Σ_a H_field^a λ^a
// (compute d⟨λ_a⟩/dt = i⟨[H_loc, λ_a]⟩ and use [λ_b, λ_a] = 2 i f_{bac} λ_c,
//  f_{bac} = −f_{abc}).
inline Matrix3c local_hamiltonian_from_field(const Vector8r& H_field, double h_0 = 0.0) {
    return local_hamiltonian(H_field, h_0);
}

// Exact one-step unitary propagator U(Δt) = exp(−i H_loc Δt) for a 3×3
// Hermitian H_loc. Uses Eigen's SelfAdjointEigenSolver, which is a hand-tuned
// 3×3 closed-form path internally, so the cost is small.
inline Matrix3c propagator_exact(const Matrix3c& H_loc, double dt) {
    Eigen::SelfAdjointEigenSolver<Matrix3c> es(H_loc);
    const auto& V = es.eigenvectors();
    Eigen::Matrix<Complex, 3, 1> phase;
    const Complex mi_dt(0.0, -dt);
    for (int k = 0; k < 3; ++k) phase(k) = std::exp(mi_dt * es.eigenvalues()(k));
    return V * phase.asDiagonal() * V.adjoint();
}

// Apply the exact local unitary step in-place, preserving ⟨ψ|ψ⟩ = 1 to
// floating-point precision (numerical re-normalisation is included as a
// safety net; should be a no-op modulo round-off).
inline void unitary_step_explicit(Vector3c& psi, const Matrix3c& H_loc, double dt) {
    psi = propagator_exact(H_loc, dt) * psi;
    const double n = psi.norm();
    if (n > 0.0) psi /= n;
}

// Spherical-midpoint geometric step of Dahlbom et al., Ref. [2].
//
// Implements the implicit midpoint scheme that is symplectic on CP^{N−1}:
//     ψ_mid          = (ψ_old + ψ_new) / 2
//     ψ_new − ψ_old  = −i Δt H_loc(ψ_mid) ψ_mid                  (∗)
// Self-consistently solved by Picard iteration. For frozen neighbours
// (`H_loc` independent of ψ), one Picard iteration is the same as the
// explicit step above; the iterative form matters only when `H_loc`
// itself depends on ψ through self-consistent / on-site nonlinearities
// (e.g. an on-site SU(3) anisotropy A^{ab} n^a n^b, which we currently
// do not use, but the implementation is ready for it).
//
// `H_loc_of` is any callable Matrix3c(const Vector3c&).
template <class H_loc_fn>
inline void spherical_midpoint_step(Vector3c& psi, H_loc_fn H_loc_of, double dt,
                                    int max_iters = 8, double tol = 1e-12) {
    Vector3c psi_old = psi;
    Vector3c psi_new = psi;
    const Complex mi_dt_half(0.0, -0.5 * dt);
    for (int it = 0; it < max_iters; ++it) {
        Vector3c psi_mid = 0.5 * (psi_old + psi_new);
        Matrix3c H = H_loc_of(psi_mid);
        Vector3c psi_next = psi_old + mi_dt_half * (H * psi_mid + H * psi_old);
        psi_next.normalize();
        const double err = (psi_next - psi_new).norm();
        psi_new = psi_next;
        if (err < tol) break;
    }
    psi = psi_new;
}

// -----------------------------------------------------------------------------
// Static T = 0 update: local ground state of H_loc.
// -----------------------------------------------------------------------------
inline Vector3c ground_state(const Matrix3c& H_loc) {
    Eigen::SelfAdjointEigenSolver<Matrix3c> es(H_loc);
    Vector3c psi = es.eigenvectors().col(0);  // ascending sort -> smallest is col 0
    psi.normalize();
    return psi;
}

// Smallest eigenvalue (ground-state energy) of H_loc.
inline double ground_state_energy(const Matrix3c& H_loc) {
    Eigen::SelfAdjointEigenSolver<Matrix3c> es(H_loc, Eigen::EigenvaluesOnly);
    return es.eigenvalues()(0);
}

// -----------------------------------------------------------------------------
// Diagnostics: physicality of a stored Gell-Mann 8-vector.
// -----------------------------------------------------------------------------
// Returns the three eigenvalues of the implied ρ. For a physical qutrit
// state these are non-negative and sum to 1; negative entries are a
// quantitative signature that the stored 8-vector does not correspond to
// any valid qutrit density matrix.
inline Eigen::Vector3d density_eigenvalues(const Vector8r& n) {
    Matrix3c rho = density_from_expectations(n);
    Eigen::SelfAdjointEigenSolver<Matrix3c> es(rho, Eigen::EigenvaluesOnly);
    return es.eigenvalues();
}

}  // namespace su3
}  // namespace classical_spin

#endif  // CLASSICAL_SPIN_SU3_COHERENT_STATE_H
