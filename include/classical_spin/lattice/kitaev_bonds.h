#pragma once
/**
 * kitaev_bonds.h — Shared helpers for the honeycomb Kitaev bond Hamiltonian.
 *
 * `PhononLattice::MagnetoelasticParams` and
 * `StrainPhononLattice::MagnetoelasticParams` historically duplicated the
 * exact same rotation matrix and three bond-exchange matrices (get_Jx_local,
 * get_Jy_local, get_Jz_local). The two copies have already drifted once
 * (global vs. local frame semantics in `get_Jx`) and are a long-standing
 * source of subtle physics bugs. This header centralises the pure functions
 * so both parameter classes now delegate here.
 *
 * What stays class-local:
 *   - `get_Jx / get_Jy / get_Jz` (frame choice differs between the two
 *     lattices — phonon uses global, strain uses local).
 *
 * What lives here:
 *   - `kitaev_rotation()` — the (1,1,-2)/√6, (-1,1,0)/√2, (1,1,1)/√3 frame.
 *   - `to_global_frame(J_local)` — R * J_local * Rᵀ.
 *   - `make_J{x,y,z}_local(J, K, Gamma, Gammap)` — per-bond 3x3 exchange
 *     matrix in the local Kitaev frame.
 *   - `heisenberg_matrix(J)` — J * I_3 for isotropic exchange (J2_A, J2_B, J3).
 */

#include "classical_spin/core/simple_linear_alg.h"
#include <cmath>

namespace classical_spin::kitaev {

/// Local-to-global rotation matrix for the honeycomb Kitaev frame.
inline SpinMatrix kitaev_rotation() {
    SpinMatrix R(3, 3);
    R <<  1.0 / std::sqrt(6.0), -1.0 / std::sqrt(2.0), 1.0 / std::sqrt(3.0),
          1.0 / std::sqrt(6.0),  1.0 / std::sqrt(2.0), 1.0 / std::sqrt(3.0),
         -2.0 / std::sqrt(6.0),  0.0,                  1.0 / std::sqrt(3.0);
    return R;
}

/// Transform a 3x3 exchange matrix from the local Kitaev frame to global.
inline SpinMatrix to_global_frame(const SpinMatrix& J_local) {
    const SpinMatrix R = kitaev_rotation();
    return R * J_local * R.transpose();
}

/// x-bond (local frame) exchange matrix with Kitaev term on diagonal (0,0).
inline SpinMatrix make_Jx_local(double J, double K, double Gamma, double Gammap) {
    SpinMatrix Jx = SpinMatrix::Zero(3, 3);
    Jx << J + K, Gammap, Gammap,
          Gammap, J,     Gamma,
          Gammap, Gamma, J;
    return Jx;
}

/// y-bond (local frame) exchange matrix with Kitaev term on diagonal (1,1).
inline SpinMatrix make_Jy_local(double J, double K, double Gamma, double Gammap) {
    SpinMatrix Jy = SpinMatrix::Zero(3, 3);
    Jy << J,      Gammap, Gamma,
          Gammap, J + K,  Gammap,
          Gamma,  Gammap, J;
    return Jy;
}

/// z-bond (local frame) exchange matrix with Kitaev term on diagonal (2,2).
inline SpinMatrix make_Jz_local(double J, double K, double Gamma, double Gammap) {
    SpinMatrix Jz = SpinMatrix::Zero(3, 3);
    Jz << J,      Gamma,  Gammap,
          Gamma,  J,      Gammap,
          Gammap, Gammap, J + K;
    return Jz;
}

/// Isotropic Heisenberg exchange (J_ij = J * I_3). Used by J2_A, J2_B, J3.
inline SpinMatrix heisenberg_matrix(double J) {
    return J * SpinMatrix::Identity(3, 3);
}

} // namespace classical_spin::kitaev
