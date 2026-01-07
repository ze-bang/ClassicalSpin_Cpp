/**
 * @file phonon_lattice.h
 * @brief Spin-phonon coupled honeycomb lattice
 * 
 * This implements a honeycomb lattice with:
 * - Generic NN, 2nd NN, and 3rd NN spin-spin interactions (bilinear matrix form)
 * - Sublattice-dependent 2nd NN coupling (J2_A for sublattice A, J2_B for sublattice B)
 * - Three phonon modes: E1 (Qx_E1, Qy_E1), E2 (Qx_E2, Qy_E2), and A1 (Q_A1)
 * - Three-phonon coupling: g3_E1A1 * (Qx_E1² + Qy_E1²) * Q_A1 + g3_E2A1 * (Qx_E2² + Qy_E2²) * Q_A1
 * - E1-E2 bilinear coupling: g3_E1E2 * (Qx_E1*Qx_E2 + Qy_E1*Qy_E2)
 * - Spin-phonon E1 coupling: λ_E1 * [Qx_E1*(SxSz + SzSx) + Qy_E1*(SySz + SzSy)]
 * - Spin-phonon E2 coupling: λ_E2 * [Qx_E2*(SxSx - SySy) + Qy_E2*(SxSy - SySx)]
 * - Spin-phonon A1 coupling: λ_A1 * Q_A1 * (Si·Sj)
 * - THz drive coupling to E1 mode (only E1 is IR active)
 * 
 * COORDINATE FRAMES:
 * -----------------
 * The spin Hamiltonian (Kitaev-Heisenberg-Γ-Γ') is defined in a LOCAL Kitaev frame
 * where the Kitaev exchange has a simple diagonal form on each bond type. The exchange
 * matrices are then TRANSFORMED to the GLOBAL Cartesian frame using:
 *   J_global = R * J_local * R^T
 * 
 * where R is the Kitaev local-to-global rotation matrix with columns:
 *   x' = (1, 1, -2)/√6
 *   y' = (-1, 1, 0)/√2  
 *   z' = (1, 1, 1)/√3
 * 
 * This means spins are stored and evolved in the GLOBAL Cartesian frame, while the
 * physical Kitaev model parameters (J, K, Γ, Γ') are specified in the standard form.
 * 
 * Hamiltonian:
 * H = H_spin + H_phonon + H_sp-ph + H_drive
 * 
 * H_spin = Σ_<ij> Si · J1_global · Sj + Σ_<<ij>>_A J2_A Si·Sj + Σ_<<ij>>_B J2_B Si·Sj
 *        + Σ_<<<ij>>> Si · J3 · Sj - Σ_i B · Si
 *   (NN J1_global = R * J1_local * R^T, where J1_local is bond-dependent Kitaev-Heisenberg-Γ-Γ')
 *   (2nd and 3rd NN are isotropic Heisenberg, invariant under rotation)
 * 
 * H_phonon = (1/2)(Vx_E1² + Vy_E1²) + (1/2)ω_E1²(Qx_E1² + Qy_E1²) + (λ_E1/4)(Qx_E1² + Qy_E1²)²
 *          + (1/2)(Vx_E2² + Vy_E2²) + (1/2)ω_E2²(Qx_E2² + Qy_E2²) + (λ_E2/4)(Qx_E2² + Qy_E2²)²
 *          + (1/2)V_A1² + (1/2)ω_A1²*Q_A1² + (λ_A1/4)*Q_A1⁴
 *          + g3_E1A1*(Qx_E1² + Qy_E1²)*Q_A1 + g3_E2A1*(Qx_E2² + Qy_E2²)*Q_A1
 *          + g3_E1E2*(Qx_E1*Qx_E2 + Qy_E1*Qy_E2)
 * 
 * H_sp-ph = Σ_<ij> [λ_E1 * Qx_E1 * (Si_x*Sj_z + Si_z*Sj_x)    // E1 coupling (global frame)
 *                 + λ_E1 * Qy_E1 * (Si_y*Sj_z + Si_z*Sj_y)
 *                 + λ_E2 * Qx_E2 * (Si_x*Sj_x - Si_y*Sj_y)    // E2 coupling (global frame)
 *                 + λ_E2 * Qy_E2 * (Si_x*Sj_y - Si_y*Sj_x)
 *                 + λ_A1 * Q_A1  * (Si · Sj)]                  // A1 coupling (invariant)
 * 
 * H_drive = -E_x(t) * Qx_E1 - E_y(t) * Qy_E1   (only E1 is IR active)
 * 
 * Equations of motion (Euler-Lagrange):
 * - Spins: dS/dt = S × H_eff (LLG with optional Gilbert damping)
 * - Phonons: d²Q/dt² = -∂H/∂Q - γ*dQ/dt
 */

#ifndef PHONON_LATTICE_H
#define PHONON_LATTICE_H

#include "unitcell.h"
#include "simple_linear_alg.h"
#include <vector>
#include <array>
#include <functional>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <complex>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <mpi.h>
#include <boost/numeric/odeint.hpp>

// Include Boost uBLAS for implicit solvers (rosenbrock4, implicit_euler)
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_controller.hpp>
#include <boost/numeric/odeint/stepper/rosenbrock4_dense_output.hpp>
#include <boost/numeric/odeint/stepper/implicit_euler.hpp>

#ifdef HDF5_ENABLED
#include "classical_spin/io/hdf5_io.h"
#endif

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::function;
using std::array;

/**
 * Phonon state for the spin-phonon model
 * 
 * E1 mode: 2-component (Qx_E1, Qy_E1) - IR active, couples to SxSz+SzSx, SySz+SzSy
 * E2 mode: 2-component (Qx_E2, Qy_E2) - Raman active, couples to SxSx-SySy, SxSy-SySx
 * A1 mode: 1-component (Q_A1) - Raman active, couples to Si·Sj
 */
struct PhononState {
    // E1 mode (infrared active, 2-component)
    double Q_x_E1 = 0.0;   // x-component of E1 normal coordinate
    double Q_y_E1 = 0.0;   // y-component of E1 normal coordinate
    
    // E2 mode (Raman active, 2-component)
    double Q_x_E2 = 0.0;   // x-component of E2 normal coordinate
    double Q_y_E2 = 0.0;   // y-component of E2 normal coordinate
    
    // A1 mode (Raman active, 1-component)
    double Q_A1 = 0.0;     // A1 normal coordinate
    
    // Velocities (dQ/dt)
    double V_x_E1 = 0.0;   // dQx_E1/dt
    double V_y_E1 = 0.0;   // dQy_E1/dt
    double V_x_E2 = 0.0;   // dQx_E2/dt
    double V_y_E2 = 0.0;   // dQy_E2/dt
    double V_A1 = 0.0;     // dQ_A1/dt
    
    // Total DOF: 5 coordinates + 5 velocities = 10
    static constexpr size_t N_DOF = 10;
    
    // Pack to flat array: [Qx_E1, Qy_E1, Qx_E2, Qy_E2, Q_A1, Vx_E1, Vy_E1, Vx_E2, Vy_E2, V_A1]
    void to_array(double* arr) const {
        arr[0] = Q_x_E1; arr[1] = Q_y_E1; 
        arr[2] = Q_x_E2; arr[3] = Q_y_E2;
        arr[4] = Q_A1;
        arr[5] = V_x_E1; arr[6] = V_y_E1;
        arr[7] = V_x_E2; arr[8] = V_y_E2;
        arr[9] = V_A1;
    }
    
    // Unpack from flat array
    void from_array(const double* arr) {
        Q_x_E1 = arr[0]; Q_y_E1 = arr[1];
        Q_x_E2 = arr[2]; Q_y_E2 = arr[3];
        Q_A1 = arr[4];
        V_x_E1 = arr[5]; V_y_E1 = arr[6];
        V_x_E2 = arr[7]; V_y_E2 = arr[8];
        V_A1 = arr[9];
    }
    
    // Kinetic energy (unit mass)
    double kinetic_energy() const {
        return 0.5 * (V_x_E1*V_x_E1 + V_y_E1*V_y_E1 + 
                      V_x_E2*V_x_E2 + V_y_E2*V_y_E2 + 
                      V_A1*V_A1);
    }
    
    // E1 amplitude
    double E1_amplitude() const {
        return std::sqrt(Q_x_E1*Q_x_E1 + Q_y_E1*Q_y_E1);
    }
    
    // E2 amplitude
    double E2_amplitude() const {
        return std::sqrt(Q_x_E2*Q_x_E2 + Q_y_E2*Q_y_E2);
    }
};

/**
 * Phonon parameters
 */
struct PhononParams {
    // E1 mode (IR active, 2-component)
    double omega_E1 = 1.0;   // E1 mode frequency
    double gamma_E1 = 0.1;   // E1 mode damping
    double lambda_E1 = 0.0;  // E1 mode quartic coefficient
    
    // E2 mode (Raman active, 2-component)
    double omega_E2 = 0.8;   // E2 mode frequency
    double gamma_E2 = 0.1;   // E2 mode damping
    double lambda_E2 = 0.0;  // E2 mode quartic coefficient
    
    // A1 mode (Raman active, 1-component)
    double omega_A1 = 0.5;   // A1 mode frequency
    double gamma_A1 = 0.05;  // A1 mode damping
    double lambda_A1 = 0.0;  // A1 mode quartic coefficient
    
    // Three-phonon coupling: g3_E1A1 * (Qx_E1² + Qy_E1²) * Q_A1
    //                      + g3_E2A1 * (Qx_E2² + Qy_E2²) * Q_A1
    double g3_E1A1 = 0.0;   // E1-A1 cubic coupling
    double g3_E2A1 = 0.0;   // E2-A1 cubic coupling
    
    // E1-E2 bilinear coupling: g3_E1E2 * (Qx_E1 * Qx_E2 + Qy_E1 * Qy_E2)
    double g3_E1E2 = 0.0;   // E1-E2 bilinear coupling
    
    // THz coupling strength (only E1 is IR active)
    double Z_star = 1.0;    // Effective charge for E(t) coupling to E1
};

/**
 * Spin-phonon coupling parameters
 * 
 * Uses Kitaev-Heisenberg-Γ-Γ' model with bond-dependent exchange:
 * H_spin = Σ_<ij>_γ Si · J_γ · Sj  where γ ∈ {x, y, z} is the bond type
 * 
 * The exchange matrices are defined in the LOCAL Kitaev frame and then
 * transformed to the GLOBAL Cartesian frame using:
 *   J_global = R * J_local * R^T
 * 
 * where R is the Kitaev local-to-global rotation matrix:
 *   R = [[1/√6, -1/√2, 1/√3],      (columns are local x', y', z' in global coords)
 *        [1/√6,  1/√2, 1/√3],
 *        [-2/√6,  0,   1/√3]]
 * 
 * Local frame Kitaev model:
 *   Jx (x-bond): K on x'x', Γ on y'z'/z'y', Γ' on x'y'/x'z'  
 *   Jy (y-bond): K on y'y', Γ on x'z'/z'x', Γ' on x'y'/y'z'
 *   Jz (z-bond): K on z'z', Γ on x'y'/y'x', Γ' on x'z'/y'z'
 * 
 * Spin-phonon coupling form on each bond <ij> (in LOCAL frame):
 *   λ_E1 * Qx_E1 * (Si_x'*Sj_z' + Si_z'*Sj_x') + λ_E1 * Qy_E1 * (Si_y'*Sj_z' + Si_z'*Sj_y')  [E1 coupling]
 * + λ_E2 * Qx_E2 * (Si_x'*Sj_x' - Si_y'*Sj_y') + λ_E2 * Qy_E2 * (Si_x'*Sj_y' - Si_y'*Sj_x')  [E2 coupling]
 * + λ_A1 * Q_A1  * (Si · Sj)                                                                  [A1 coupling]
 */
struct SpinPhononCouplingParams {
    // Kitaev-Heisenberg-Γ-Γ' parameters (Songvilay defaults)
    double J = -0.1;       // Heisenberg coupling
    double K = -9.0;       // Kitaev coupling
    double Gamma = 1.8;    // Γ (off-diagonal symmetric) 
    double Gammap = 0.3;   // Γ' (off-diagonal asymmetric)
    
    // 2nd NN exchange (isotropic Heisenberg, sublattice-dependent)
    double J2_A = 0.3;     // 2nd NN coupling on sublattice A
    double J2_B = 0.3;     // 2nd NN coupling on sublattice B
    
    // 3rd NN exchange (isotropic Heisenberg)
    double J3 = 0.9;
    
    // Six-spin ring exchange on hexagonal plaquettes
    double J7 = 0.0;
    
    // Spin-phonon coupling strengths (in LOCAL Kitaev frame)
    double lambda_E1 = 0.0;  // E1 coupling: Qx_E1*(Sx'Sz'+Sz'Sx') + Qy_E1*(Sy'Sz'+Sz'Sy')
    double lambda_E2 = 0.0;  // E2 coupling: Qx_E2*(Sx'Sx'-Sy'Sy') + Qy_E2*(Sx'Sy'-Sy'Sx')
    double lambda_A1 = 0.0;  // A1 coupling: Q_A1*(Si·Sj)
    
    /**
     * Get the Kitaev local-to-global rotation matrix R.
     * 
     * The columns of R are the local frame basis vectors (x', y', z') expressed
     * in global Cartesian coordinates:
     *   x' = (1, 1, -2)/√6
     *   y' = (-1, 1, 0)/√2
     *   z' = (1, 1, 1)/√3
     * 
     * This transforms spins from local to global: S_global = R * S_local
     */
    static SpinMatrix get_kitaev_rotation() {
        SpinMatrix R(3, 3);
        R << 1.0/std::sqrt(6.0), -1.0/std::sqrt(2.0), 1.0/std::sqrt(3.0),
             1.0/std::sqrt(6.0),  1.0/std::sqrt(2.0), 1.0/std::sqrt(3.0),
            -2.0/std::sqrt(6.0),  0.0,                1.0/std::sqrt(3.0);
        return R;
    }
    
    /**
     * Transform exchange matrix from local Kitaev frame to global Cartesian frame.
     * J_global = R * J_local * R^T
     */
    static SpinMatrix to_global_frame(const SpinMatrix& J_local) {
        SpinMatrix R = get_kitaev_rotation();
        return R * J_local * R.transpose();
    }
    
    // Build bond-dependent exchange matrices in LOCAL Kitaev frame
    SpinMatrix get_Jx_local() const {
        SpinMatrix Jx = SpinMatrix::Zero(3, 3);
        Jx << J + K, Gammap, Gammap,
              Gammap, J, Gamma,
              Gammap, Gamma, J;
        return Jx;
    }
    
    SpinMatrix get_Jy_local() const {
        SpinMatrix Jy = SpinMatrix::Zero(3, 3);
        Jy << J, Gammap, Gamma,
              Gammap, J + K, Gammap,
              Gamma, Gammap, J;
        return Jy;
    }
    
    SpinMatrix get_Jz_local() const {
        SpinMatrix Jz = SpinMatrix::Zero(3, 3);
        Jz << J, Gamma, Gammap,
              Gamma, J, Gammap,
              Gammap, Gammap, J + K;
        return Jz;
    }
    
    // Build bond-dependent exchange matrices in GLOBAL Cartesian frame
    // These are used for spin dynamics with spins stored in global coordinates
    SpinMatrix get_Jx() const {
        return to_global_frame(get_Jx_local());
    }
    
    SpinMatrix get_Jy() const {
        return to_global_frame(get_Jy_local());
    }
    
    SpinMatrix get_Jz() const {
        return to_global_frame(get_Jz_local());
    }
    
    // J2 and J3 are isotropic Heisenberg, so they are invariant under rotation
    SpinMatrix get_J3_matrix() const {
        return J3 * SpinMatrix::Identity(3, 3);
    }
    
    SpinMatrix get_J2_A_matrix() const {
        return J2_A * SpinMatrix::Identity(3, 3);
    }
    
    SpinMatrix get_J2_B_matrix() const {
        return J2_B * SpinMatrix::Identity(3, 3);
    }
};

/**
 * Time-dependent spin-phonon coupling parameters
 * 
 * Allows the spin-phonon coupling strengths (λ_E1, λ_E2, λ_A1) to be time-dependent.
 * Default mode is "constant" which uses the values from SpinPhononCouplingParams.
 * 
 * Available modes:
 * - "constant": λ(t) = λ (from SpinPhononCouplingParams, default)
 * - "window": λ(t) = λ_target for t_start <= t <= t_end, λ_static otherwise
 * 
 * Each mode (E1, E2, A1) can have independent window parameters.
 */
struct TimeDependentSpinPhononParams {
    // Mode selection: "constant" or "window"
    std::string mode = "constant";
    
    // Window function parameters for E1 mode
    double t_start_E1 = 0.0;       // Time at which E1 coupling changes to target
    double t_end_E1 = 1e30;        // Time at which E1 coupling reverts to static
    double lambda_E1_target = 0.0; // Value of λ_E1 inside the window
    
    // Window function parameters for E2 mode
    double t_start_E2 = 0.0;       // Time at which E2 coupling changes to target
    double t_end_E2 = 1e30;        // Time at which E2 coupling reverts to static
    double lambda_E2_target = 0.0; // Value of λ_E2 inside the window
    
    // Window function parameters for A1 mode
    double t_start_A1 = 0.0;       // Time at which A1 coupling changes to target
    double t_end_A1 = 1e30;        // Time at which A1 coupling reverts to static
    double lambda_A1_target = 0.0; // Value of λ_A1 inside the window
    
    /**
     * Get effective λ_E1 at time t
     */
    double get_lambda_E1(double t, double lambda_E1_static) const {
        if (mode == "constant") {
            return lambda_E1_static;
        } else if (mode == "window") {
            return (t >= t_start_E1 && t <= t_end_E1) ? lambda_E1_target : lambda_E1_static;
        }
        return lambda_E1_static;  // fallback
    }
    
    /**
     * Get effective λ_E2 at time t
     */
    double get_lambda_E2(double t, double lambda_E2_static) const {
        if (mode == "constant") {
            return lambda_E2_static;
        } else if (mode == "window") {
            return (t >= t_start_E2 && t <= t_end_E2) ? lambda_E2_target : lambda_E2_static;
        }
        return lambda_E2_static;  // fallback
    }
    
    /**
     * Get effective λ_A1 at time t
     */
    double get_lambda_A1(double t, double lambda_A1_static) const {
        if (mode == "constant") {
            return lambda_A1_static;
        } else if (mode == "window") {
            return (t >= t_start_A1 && t <= t_end_A1) ? lambda_A1_target : lambda_A1_static;
        }
        return lambda_A1_static;  // fallback
    }
};

/**
 * THz drive parameters (two-pulse)
 */
struct DriveParams {
    // Pulse 1
    double E0_1 = 0.0;      // Amplitude
    double omega_1 = 1.0;   // Frequency
    double t_1 = 0.0;       // Center time
    double sigma_1 = 1.0;   // Gaussian width
    double phi_1 = 0.0;     // Phase
    double theta_1 = 0.0;   // Polarization angle (0 = x, π/2 = y)
    
    // Pulse 2
    double E0_2 = 0.0;
    double omega_2 = 1.0;
    double t_2 = 0.0;
    double sigma_2 = 1.0;
    double phi_2 = 0.0;
    double theta_2 = 0.0;
    
    // Compute E-field components at time t
    void E_field(double t, double& Ex, double& Ey) const {
        double dt1 = t - t_1;
        double dt2 = t - t_2;
        
        double env1 = std::exp(-0.5 * dt1*dt1 / (sigma_1*sigma_1));
        double osc1 = std::cos(omega_1 * dt1 + phi_1);
        double E1 = E0_1 * env1 * osc1;
        
        double env2 = std::exp(-0.5 * dt2*dt2 / (sigma_2*sigma_2));
        double osc2 = std::cos(omega_2 * dt2 + phi_2);
        double E2 = E0_2 * env2 * osc2;
        
        Ex = E1 * std::cos(theta_1) + E2 * std::cos(theta_2);
        Ey = E1 * std::sin(theta_1) + E2 * std::sin(theta_2);
    }
};

/**
 * PhononLattice: Honeycomb lattice with spin-phonon coupling
 * 
 * Degrees of freedom:
 * - N_spin = 2 * dim1 * dim2 * dim3 classical spins (honeycomb, spin_dim = 3)
 * - 10 global phonon DOF: (Qx_E1, Qy_E1, Qx_E2, Qy_E2, Q_A1, Vx_E1, Vy_E1, Vx_E2, Vy_E2, V_A1)
 * 
 * Total ODE state size: 3 * N_spin + 10
 */
class PhononLattice {
public:
    using SpinConfig = vector<SpinVector>;
    using ODEState = vector<double>;
    
    // Lattice properties
    static constexpr size_t spin_dim = 3;    // 3D classical spins
    static constexpr size_t N_atoms = 2;     // 2 atoms per honeycomb unit cell
    size_t dim1, dim2, dim3;                 // Lattice dimensions
    size_t lattice_size;                     // Total spin sites
    float spin_length = 1.0;                 // Spin magnitude
    
    // Spin configuration
    SpinConfig spins;
    vector<Eigen::Vector3d> site_positions;
    
    // Phonon state
    PhononState phonons;
    
    // NN interactions (stored per site to avoid double counting)
    vector<vector<SpinMatrix>> nn_interaction;      // J1 matrices
    vector<vector<size_t>> nn_partners;             // NN partner indices
    vector<vector<int>> nn_bond_types;              // Bond type (0,1,2 for x,y,z bonds)
    
    // 2nd NN interactions (sublattice-dependent)
    vector<vector<SpinMatrix>> j2_interaction;      // J2 matrices (J2_A or J2_B depending on sublattice)
    vector<vector<size_t>> j2_partners;             // 2nd NN partner indices
    
    // 3rd NN interactions
    vector<vector<SpinMatrix>> j3_interaction;      // J3 matrices  
    vector<vector<size_t>> j3_partners;             // 3rd NN partner indices
    
    // Hexagonal plaquettes for ring exchange
    // Each hexagon stores 6 site indices in order (i,j,k,l,m,n) going around the ring
    vector<std::array<size_t, 6>> hexagons;
    // For each site, list of hexagons it belongs to and its position (0-5) within each hexagon
    vector<vector<std::pair<size_t, size_t>>> site_hexagons;
    
    // External field
    vector<SpinVector> field;
    
    // Parameters
    PhononParams phonon_params;
    SpinPhononCouplingParams spin_phonon_params;
    TimeDependentSpinPhononParams time_dep_spin_phonon_params;
    DriveParams drive_params;
    
    // LLG damping
    double alpha_gilbert = 0.0;
    
    // ODE state size
    size_t state_size;
    
    // Sublattice local frames for global-to-local spin transformations
    // For Kitaev honeycomb, transforms from local Kitaev basis to global cubic frame
    // sublattice_frames[atom] is a 3x3 rotation matrix: S_global = R * S_local
    std::array<SpinMatrix, N_atoms> sublattice_frames;
    
    // Custom ordering vector (set from initial spin configuration)
    // Used to compute order parameter along the ground state ordering direction
    SpinConfig ordering_pattern;
    bool has_ordering_pattern = false;
    
    /**
     * Constructor
     */
    PhononLattice(size_t d1, size_t d2, size_t d3 = 1, float spin_l = 1.0);
    
    // ============================================================
    // LATTICE CONSTRUCTION
    // ============================================================
    
    /**
     * Build honeycomb lattice structure
     */
    void build_honeycomb();
    
    /**
     * Flatten multi-index to linear site index
     */
    size_t flatten_index(size_t i, size_t j, size_t k, size_t atom) const {
        return ((i * dim2 + j) * dim3 + k) * N_atoms + atom;
    }
    
    /**
     * Periodic boundary condition
     */
    int periodic_boundary(int coord, size_t dim_size) const {
        if (coord < 0) return coord + dim_size;
        if (coord >= (int)dim_size) return coord - dim_size;
        return coord;
    }
    
    /**
     * Flatten with periodic boundaries
     */
    size_t flatten_index_periodic(int i, int j, int k, size_t atom) const {
        return flatten_index(
            periodic_boundary(i, dim1),
            periodic_boundary(j, dim2),
            periodic_boundary(k, dim3),
            atom
        );
    }
    
    // ============================================================
    // PARAMETER SETTING
    // ============================================================
    
    /**
     * Set all parameters and rebuild interaction matrices
     */
    void set_parameters(const SpinPhononCouplingParams& sp_params,
                       const PhononParams& ph_params,
                       const DriveParams& dr_params);
    
    /**
     * Set time-dependent spin-phonon coupling parameters
     */
    void set_time_dependent_spin_phonon(const TimeDependentSpinPhononParams& td_params) {
        time_dep_spin_phonon_params = td_params;
        if (td_params.mode != "constant") {
            std::cout << "Time-dependent spin-phonon coupling enabled (mode: " 
                      << td_params.mode << ")" << std::endl;
            if (td_params.mode == "window") {
                std::cout << "  E1: λ=" << td_params.lambda_E1_target 
                          << " for t∈[" << td_params.t_start_E1 << ", " << td_params.t_end_E1 << "]" << std::endl;
                std::cout << "  E2: λ=" << td_params.lambda_E2_target 
                          << " for t∈[" << td_params.t_start_E2 << ", " << td_params.t_end_E2 << "]" << std::endl;
                std::cout << "  A1: λ=" << td_params.lambda_A1_target 
                          << " for t∈[" << td_params.t_start_A1 << ", " << td_params.t_end_A1 << "]" << std::endl;
            }
        }
    }
    
    /**
     * Get effective λ_E1 at time t
     */
    double get_lambda_E1(double t) const {
        return time_dep_spin_phonon_params.get_lambda_E1(t, spin_phonon_params.lambda_E1);
    }
    
    /**
     * Get effective λ_E2 at time t
     */
    double get_lambda_E2(double t) const {
        return time_dep_spin_phonon_params.get_lambda_E2(t, spin_phonon_params.lambda_E2);
    }
    
    /**
     * Get effective λ_A1 at time t
     */
    double get_lambda_A1(double t) const {
        return time_dep_spin_phonon_params.get_lambda_A1(t, spin_phonon_params.lambda_A1);
    }
    
    /**
     * Set external magnetic field (uniform)
     */
    void set_field(const Eigen::Vector3d& B) {
        for (size_t i = 0; i < lattice_size; ++i) {
            field[i] = B;
        }
    }
    
    // ============================================================
    // INITIALIZATION
    // ============================================================
    
    /**
     * Generate random spin on 2-sphere
     */
    SpinVector gen_random_spin() {
        SpinVector spin(3);
        double z = random_double_lehman(-1.0, 1.0);
        double phi = random_double_lehman(0.0, 2.0 * M_PI);
        double r = std::sqrt(1.0 - z*z);
        spin(0) = r * std::cos(phi);
        spin(1) = r * std::sin(phi);
        spin(2) = z;
        return spin * spin_length;
    }
    
    /**
     * Initialize random spins
     */
    void init_random() {
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = gen_random_spin();
        }
        phonons = PhononState();
    }
    
    /**
     * Initialize ferromagnetic state
     */
    void init_ferromagnetic(const Eigen::Vector3d& direction) {
        Eigen::Vector3d dir = direction.normalized() * spin_length;
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = dir;
        }
        phonons = PhononState();
    }
    
    /**
     * Initialize Néel state (antiferromagnetic on sublattices)
     */
    void init_neel(const Eigen::Vector3d& direction) {
        Eigen::Vector3d dir = direction.normalized() * spin_length;
        for (size_t i = 0; i < lattice_size; ++i) {
            double sign = (i % 2 == 0) ? 1.0 : -1.0;
            spins[i] = sign * dir;
        }
        phonons = PhononState();
    }
    
    // ============================================================
    // ENERGY CALCULATIONS
    // ============================================================
    
    /**
     * Compute local energy for a spin at a given site
     * E = -B·S + Σ_j S^T J_ij S_j (for NN, 2nd NN, 3rd NN)
     *   + spin-phonon coupling terms
     * 
     * Note: This counts the full bond energy for each neighbor, so summing
     * site_energy over all sites will double-count pair interactions.
     * Use total_energy() for the correctly counted total energy.
     */
    double site_energy(const Eigen::Vector3d& spin_here, size_t site) const {
        double E = -spin_here.dot(field[site]);
        
        // Phonon coordinates (for spin-phonon coupling)
        double Qx_E1 = phonons.Q_x_E1;
        double Qy_E1 = phonons.Q_y_E1;
        double Qx_E2 = phonons.Q_x_E2;
        double Qy_E2 = phonons.Q_y_E2;
        double Q_A1 = phonons.Q_A1;
        double l_E1 = spin_phonon_params.lambda_E1;
        double l_E2 = spin_phonon_params.lambda_E2;
        double l_A1 = spin_phonon_params.lambda_A1;
        
        // NN interactions (includes spin-phonon coupling)
        for (size_t n = 0; n < nn_partners[site].size(); ++n) {
            size_t j = nn_partners[site][n];
            const Eigen::Vector3d& Sj = spins[j];
            
            // Pure spin-spin interaction
            E += spin_here.dot(nn_interaction[site][n] * Sj);
            
            // Spin-phonon coupling: E1 terms
            // λ_E1 * Qx_E1 * (Si_x * Sj_z + Si_z * Sj_x)
            E += l_E1 * Qx_E1 * (spin_here(0) * Sj(2) + spin_here(2) * Sj(0));
            // λ_E1 * Qy_E1 * (Si_y * Sj_z + Si_z * Sj_y)
            E += l_E1 * Qy_E1 * (spin_here(1) * Sj(2) + spin_here(2) * Sj(1));
            
            // Spin-phonon coupling: E2 terms
            // λ_E2 * Qx_E2 * (Si_x * Sj_x - Si_y * Sj_y)
            E += l_E2 * Qx_E2 * (spin_here(0) * Sj(0) - spin_here(1) * Sj(1));
            // λ_E2 * Qy_E2 * (Si_x * Sj_y - Si_y * Sj_x)
            E += l_E2 * Qy_E2 * (spin_here(0) * Sj(1) - spin_here(1) * Sj(0));
            
            // Spin-phonon coupling: A1 term
            // λ_A1 * Q_A1 * (Si · Sj)
            E += l_A1 * Q_A1 * spin_here.dot(Sj);
        }
        // 2nd NN interactions
        for (size_t n = 0; n < j2_partners[site].size(); ++n) {
            size_t j = j2_partners[site][n];
            E += spin_here.dot(j2_interaction[site][n] * spins[j]);
        }
        // 3rd NN interactions
        for (size_t n = 0; n < j3_partners[site].size(); ++n) {
            size_t j = j3_partners[site][n];
            E += spin_here.dot(j3_interaction[site][n] * spins[j]);
        }
        return E;
    }
    
    /**
     * Compute energy difference for a proposed spin flip (optimized for Metropolis)
     * dE = E(new_spin) - E(old_spin)
     * 
     * Includes:
     * - Zeeman energy change
     * - NN, 2nd NN, 3rd NN spin-spin interaction changes
     * - Spin-phonon coupling energy change (if phonons are non-zero)
     * - Ring exchange energy change
     */
    double site_energy_diff(const Eigen::Vector3d& new_spin, 
                           const Eigen::Vector3d& old_spin, 
                           size_t site) const;
    
    /**
     * Pure spin energy (NN + 2nd NN + 3rd NN + Zeeman + ring exchange)
     */
    double spin_energy() const;
    
    /**
     * Six-spin ring exchange energy on hexagonal plaquettes
     * 
     * H_7 = (J_7/6) Σ_{hex} [2(S_i·S_j)(S_k·S_l)(S_m·S_n)
     *                       -6(S_i·S_k)(S_j·S_l)(S_m·S_n)
     *                       +3(S_i·S_l)(S_j·S_k)(S_m·S_n)
     *                       +3(S_i·S_k)(S_j·S_m)(S_l·S_n)
     *                       -(S_i·S_l)(S_j·S_m)(S_k·S_n)
     *                       + cyclic permutations of (i,j,k,l,m,n)]
     */
    double ring_exchange_energy() const;
    
    /**
     * Phonon energy (kinetic + potential + cubic coupling)
     * 
     * E_ph = (1/2)(Vx_E1² + Vy_E1²) + (1/2)ω_E1²(Qx_E1² + Qy_E1²) + (λ_E1/4)(Qx_E1² + Qy_E1²)²
     *      + (1/2)(Vx_E2² + Vy_E2²) + (1/2)ω_E2²(Qx_E2² + Qy_E2²) + (λ_E2/4)(Qx_E2² + Qy_E2²)²
     *      + (1/2)V_A1² + (1/2)ω_A1²*Q_A1² + (λ_A1/4)*Q_A1⁴
     *      + g3_E1A1*(Qx_E1² + Qy_E1²)*Q_A1 + g3_E2A1*(Qx_E2² + Qy_E2²)*Q_A1
     *      + g3_E1E2*(Qx_E1*Qx_E2 + Qy_E1*Qy_E2)
     */
    double phonon_energy() const;
    
    /**
     * Spin-phonon coupling energy
     * 
     * H_sp-ph = Σ_<ij> [λ_E1 * Qx_E1 * (Si_x*Sj_z + Si_z*Sj_x)   // E1 coupling
     *                 + λ_E1 * Qy_E1 * (Si_y*Sj_z + Si_z*Sj_y)
     *                 + λ_E2 * Qx_E2 * (Si_x*Sj_x - Si_y*Sj_y)   // E2 coupling
     *                 + λ_E2 * Qy_E2 * (Si_x*Sj_y - Si_y*Sj_x)
     *                 + λ_A1 * Q_A1  * (Si · Sj)]                 // A1 coupling
     */
    double spin_phonon_energy() const;
    
    /**
     * Total energy
     */
    double total_energy() const {
        return spin_energy() + phonon_energy() + spin_phonon_energy();
    }
    
    /**
     * Energy per site
     */
    double energy_density() const {
        return total_energy() / lattice_size;
    }
    
    // ============================================================
    // DERIVATIVES FOR EQUATIONS OF MOTION
    // ============================================================
    
    /**
     * Compute ∂H_sp-ph/∂Qx_E1 = Σ_<ij> λ_E1 * (Si_x*Sj_z + Si_z*Sj_x)
     */
    double dH_dQx_E1() const;
    
    /**
     * Compute ∂H_sp-ph/∂Qy_E1 = Σ_<ij> λ_E1 * (Si_y*Sj_z + Si_z*Sj_y)
     */
    double dH_dQy_E1() const;
    
    /**
     * Compute ∂H_sp-ph/∂Qx_E2 = Σ_<ij> λ_E2 * (Si_x*Sj_x - Si_y*Sj_y)
     */
    double dH_dQx_E2() const;
    
    /**
     * Compute ∂H_sp-ph/∂Qy_E2 = Σ_<ij> λ_E2 * (Si_x*Sj_y - Si_y*Sj_x)
     */
    double dH_dQy_E2() const;
    
    /**
     * Compute ∂H_sp-ph/∂Q_A1 = Σ_<ij> λ_A1 * (Si · Sj)
     */
    double dH_dQ_A1() const;
    
    /**
     * Compute ring exchange contribution to effective field on spin at given site
     * 
     * H_eff_ring = -∂H_7/∂S_site
     * 
     * For each hexagon containing the site, computes the derivative of the
     * ring exchange term with respect to that spin.
     */
    SpinVector get_ring_exchange_field(size_t site) const;
    
    /**
     * Compute effective field on spin i (for spin EOM)
     * 
     * H_eff = -∂H/∂Si = B + Σ_j [NN contributions] + [spin-phonon contributions] + [ring exchange]
     */
    SpinVector get_local_field(size_t site) const;
    
    // ============================================================
    // EQUATIONS OF MOTION
    // ============================================================
    
    /**
     * Phonon EOM derivatives
     * 
     * E1 mode (IR active, THz driven):
     *   dQx_E1/dt = Vx_E1
     *   dVx_E1/dt = -ω_E1² Qx_E1 - λ_E1 (Qx_E1²+Qy_E1²) Qx_E1 
     *              - 2*g3_E1A1*Qx_E1*Q_A1 - γ_E1*Vx_E1 - ∂H_sp-ph/∂Qx_E1 + Z*Ex(t)
     *   dQy_E1/dt = Vy_E1
     *   dVy_E1/dt = -ω_E1² Qy_E1 - λ_E1 (Qx_E1²+Qy_E1²) Qy_E1 
     *              - 2*g3_E1A1*Qy_E1*Q_A1 - γ_E1*Vy_E1 - ∂H_sp-ph/∂Qy_E1 + Z*Ey(t)
     * 
     * E2 mode (Raman active, not directly driven by THz):
     *   dQx_E2/dt = Vx_E2
     *   dVx_E2/dt = -ω_E2² Qx_E2 - λ_E2 (Qx_E2²+Qy_E2²) Qx_E2 
     *              - 2*g3_E2A1*Qx_E2*Q_A1 - γ_E2*Vx_E2 - ∂H_sp-ph/∂Qx_E2
     *   dQy_E2/dt = Vy_E2
     *   dVy_E2/dt = -ω_E2² Qy_E2 - λ_E2 (Qx_E2²+Qy_E2²) Qy_E2 
     *              - 2*g3_E2A1*Qy_E2*Q_A1 - γ_E2*Vy_E2 - ∂H_sp-ph/∂Qy_E2
     * 
     * A1 mode (Raman active, not directly driven by THz):
     *   dQ_A1/dt = V_A1
     *   dV_A1/dt = -ω_A1² Q_A1 - λ_A1 Q_A1³ 
     *             - g3_E1A1 (Qx_E1²+Qy_E1²) - g3_E2A1 (Qx_E2²+Qy_E2²) 
     *             - γ_A1*V_A1 - ∂H_sp-ph/∂Q_A1
     */
    void phonon_derivatives(const PhononState& ph, double t,
                           double dH_dQx_E1_val, double dH_dQy_E1_val,
                           double dH_dQx_E2_val, double dH_dQy_E2_val,
                           double dH_dQ_A1_val,
                           PhononState& dph_dt) const;
    
    /**
     * Full ODE system for coupled spin-phonon dynamics
     * State: [S0_x, S0_y, S0_z, ..., SN_z, Qx, Qy, Q_R, Vx, Vy, V_R]
     */
    void ode_system(const ODEState& x, ODEState& dxdt, double t);
    
    /**
     * Spin derivative (LLG equation)
     * dS/dt = S × H_eff + α S × (S × H_eff)
     */
    Eigen::Vector3d spin_derivative(const Eigen::Vector3d& S, 
                                    const Eigen::Vector3d& H_eff) const {
        Eigen::Vector3d dSdt = S.cross(H_eff);
        if (alpha_gilbert > 0) {
            dSdt += alpha_gilbert * S.cross(S.cross(H_eff)) / spin_length;
        }
        return dSdt;
    }
    
    // ============================================================
    // STATE CONVERSION
    // ============================================================
    
    /**
     * Pack current state to flat ODE state vector
     */
    ODEState to_state() const {
        ODEState state(state_size);
        size_t idx = 0;
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t d = 0; d < spin_dim; ++d) {
                state[idx++] = spins[i](d);
            }
        }
        phonons.to_array(&state[idx]);
        return state;
    }
    
    /**
     * Unpack flat ODE state to internal variables
     */
    void from_state(const ODEState& state) {
        size_t idx = 0;
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t d = 0; d < spin_dim; ++d) {
                spins[i](d) = state[idx++];
            }
            // Renormalize spins
            spins[i] = spins[i].normalized() * spin_length;
        }
        phonons.from_array(&state[idx]);
    }
    
    // ============================================================
    // OBSERVABLES
    // ============================================================
    
    /**
     * Total magnetization (per spin)
     */
    Eigen::Vector3d magnetization() const {
        Eigen::Vector3d M = Eigen::Vector3d::Zero();
        for (const auto& s : spins) {
            M += s;
        }
        return M / lattice_size;
    }
    
    /**
     * Staggered magnetization
     */
    Eigen::Vector3d staggered_magnetization() const {
        Eigen::Vector3d M = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < lattice_size; ++i) {
            double sign = (i % 2 == 0) ? 1.0 : -1.0;
            M += sign * spins[i];
        }
        return M / lattice_size;
    }
    
    /**
     * Global magnetization (transformed from local Kitaev frame to global cubic frame)
     * M_global = Σ R * S_local / N
     * where R is the sublattice frame transformation matrix
     */
    Eigen::Vector3d magnetization_global() const {
        Eigen::Vector3d M = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < lattice_size; ++i) {
            size_t atom = i % N_atoms;
            // Transform spin from local to global frame
            M += sublattice_frames[atom] * spins[i];
        }
        return M / lattice_size;
    }
    
    /**
     * Set the ordering pattern from current spin configuration
     * This should be called after simulated annealing/equilibration to capture
     * the ground state ordering for computing custom order parameters
     */
    void set_ordering_pattern() {
        ordering_pattern = spins;
        has_ordering_pattern = true;
    }
    
    /**
     * Set ordering pattern from provided spin configuration
     */
    void set_ordering_pattern(const SpinConfig& pattern) {
        if (pattern.size() != lattice_size) {
            throw std::invalid_argument("Ordering pattern size mismatch");
        }
        ordering_pattern = pattern;
        has_ordering_pattern = true;
    }
    
    /**
     * Compute custom order parameter based on the ordering pattern
     * Projects current spin configuration onto the initial ordering pattern
     * O = Σ S_i · S_i^(0) / N
     * where S_i^(0) is the initial ordering pattern
     */
    double custom_order_parameter() const {
        if (!has_ordering_pattern) {
            return 0.0;
        }
        double O = 0.0;
        for (size_t i = 0; i < lattice_size; ++i) {
            O += spins[i].dot(ordering_pattern[i]);
        }
        return O / lattice_size;
    }
    
    /**
     * Compute custom magnetization projected onto ordering pattern (per sublattice)
     * Returns vector of order parameters: [O_total, O_A, O_B]
     * where O_A = Σ_{i∈A} S_i · S_i^(0) / N_A and similarly for O_B
     */
    Eigen::Vector3d custom_order_parameter_sublattice() const {
        if (!has_ordering_pattern) {
            return Eigen::Vector3d::Zero();
        }
        double O_total = 0.0;
        double O_A = 0.0;
        double O_B = 0.0;
        size_t N_A = 0, N_B = 0;
        
        for (size_t i = 0; i < lattice_size; ++i) {
            double proj = spins[i].dot(ordering_pattern[i]);
            O_total += proj;
            if (i % N_atoms == 0) {
                O_A += proj;
                N_A++;
            } else {
                O_B += proj;
                N_B++;
            }
        }
        
        Eigen::Vector3d result;
        result << O_total / lattice_size,
                  (N_A > 0) ? O_A / N_A : 0.0,
                  (N_B > 0) ? O_B / N_B : 0.0;
        return result;
    }
    
    /**
     * E1 phonon amplitude
     */
    double E1_amplitude() const {
        return phonons.E1_amplitude();
    }
    
    /**
     * E2 phonon amplitude
     */
    double E2_amplitude() const {
        return phonons.E2_amplitude();
    }
    
    // ============================================================
    // SIMULATION
    // ============================================================
    
private:
    /**
     * Generic ODE integrator with support for multiple methods
     * 
     * Available methods:
     * 
     * EXPLICIT METHODS (recommended for non-stiff problems):
     * - "euler": Explicit Euler (1st order)
     * - "rk2" or "midpoint": Runge-Kutta 2nd order
     * - "rk4": Classic Runge-Kutta 4th order (fixed step)
     * - "rk5" or "rkck54": Cash-Karp 5(4) adaptive
     * - "rk54" or "rkf54": Runge-Kutta-Fehlberg 5(4) adaptive
     * - "dopri5": Dormand-Prince 5(4) adaptive (default, recommended)
     * - "rk78" or "rkf78": Runge-Kutta-Fehlberg 7(8) (high accuracy)
     * - "bulirsch_stoer" or "bs": Bulirsch-Stoer (very high accuracy)
     * - "adams_bashforth" or "ab": Adams-Bashforth 5-step multistep
     * - "adams_moulton" or "am": Adams-Bashforth-Moulton predictor-corrector
     * 
     * IMPLICIT METHODS (recommended for stiff problems):
     * - "rosenbrock4" or "rb4": Rosenbrock 4th order (stiff systems, uses numerical Jacobian)
     * - "implicit_euler" or "ie": Implicit Euler (1st order, very stable for stiff systems)
     * 
     * Note: Implicit methods use numerical Jacobian approximation via finite differences.
     * They are more stable for stiff problems but computationally more expensive.
     */
    template<typename System, typename Observer>
    void integrate_ode_system(System system_func, ODEState& state,
                             double T_start, double T_end, double dt_step,
                             Observer observer, const string& method,
                             bool use_adaptive = false,
                             double abs_tol = 1e-6, double rel_tol = 1e-6);

public:
    /**
     * Run molecular dynamics simulation
     * 
     * @param T_start        Start time
     * @param T_end          End time
     * @param dt_initial     Initial/fixed time step
     * @param out_dir        Output directory for trajectories
     * @param save_interval  Steps between saves
     * @param method         Integration method: euler, rk2, rk4, rk5, dopri5 (default),
     *                       rk78, bulirsch_stoer, adams_bashforth, adams_moulton
     */
    void molecular_dynamics(double T_start, double T_end, double dt_initial,
                           string out_dir = "", size_t save_interval = 100,
                           string method = "dopri5");
    
    // ============================================================
    // MONTE CARLO METHODS
    // ============================================================
    
    /**
     * Single Metropolis sweep over all spins
     * @param T  Temperature
     * @return Number of accepted moves
     */
    size_t metropolis_sweep(double T);
    
    /**
     * Single overrelaxation sweep over all spins
     * Reflects each spin about its local field (energy-conserving)
     */
    void overrelaxation_sweep();
    
    /**
     * Simulated annealing for spin subsystem
     * 
     * @param T_start              Starting temperature
     * @param T_end                Final temperature
     * @param n_steps              Number of MC sweeps per temperature
     * @param overrelax_rate       Overrelaxation frequency (0 = disabled)
     * @param cooling_rate         Temperature cooling factor (T *= cooling_rate each step)
     * @param out_dir              Output directory for saving configs
     * @param save_observables     Whether to save observables to HDF5
     * @param T_zero               Whether to perform deterministic sweeps at T=0
     * @param n_deterministics     Number of deterministic sweeps at T=0
     * @param adiabatic_phonons    If true, relax phonons to equilibrium at each temperature step
     *                             (Born-Oppenheimer approximation for phonons)
     */
    void simulated_annealing(double T_start, double T_end, size_t n_steps,
                            size_t overrelax_rate = 0,
                            double cooling_rate = 0.9,
                            string out_dir = "",
                            bool save_observables = true,
                            bool T_zero = false,
                            size_t n_deterministics = 1000,
                            bool adiabatic_phonons = false);
    
    /**
     * Deterministic T=0 sweep: align each spin with its local field
     * 
     * @param num_sweeps  Number of sweeps to perform
     */
    void deterministic_sweep(size_t num_sweeps);
    
    /**
     * Relax phonon coordinates to equilibrium for current spin configuration.
     * 
     * Finds the static equilibrium Q values that minimize the total energy
     * for fixed spins. This should be called after simulated annealing and
     * before molecular dynamics to ensure a proper steady state.
     * 
     * The equilibrium satisfies (for each mode):
     * E1: ω_E1² Qx_E1 + λ_E1 (Qx_E1²+Qy_E1²) Qx_E1 + 2 g3_E1A1 Qx_E1 Q_A1 + ∂H_sp/∂Qx_E1 = 0
     *     ω_E1² Qy_E1 + λ_E1 (Qx_E1²+Qy_E1²) Qy_E1 + 2 g3_E1A1 Qy_E1 Q_A1 + ∂H_sp/∂Qy_E1 = 0
     * E2: ω_E2² Qx_E2 + λ_E2 (Qx_E2²+Qy_E2²) Qx_E2 + 2 g3_E2A1 Qx_E2 Q_A1 + ∂H_sp/∂Qx_E2 = 0
     *     ω_E2² Qy_E2 + λ_E2 (Qx_E2²+Qy_E2²) Qy_E2 + 2 g3_E2A1 Qy_E2 Q_A1 + ∂H_sp/∂Qy_E2 = 0
     * A1: ω_A1² Q_A1 + λ_A1 Q_A1³ + g3_E1A1 (Qx_E1²+Qy_E1²) 
     *     + g3_E2A1 (Qx_E2²+Qy_E2²) + ∂H_sp/∂Q_A1 = 0
     * 
     * Uses damped dynamics to find equilibrium (overdamped relaxation).
     * 
     * @param tol         Convergence tolerance for |dQ/dt|
     * @param max_iter    Maximum relaxation iterations
     * @param damping     Damping coefficient for overdamped dynamics
     * @return true if converged, false if max_iter reached
     */
    bool relax_phonons(double tol = 1e-8, size_t max_iter = 10000, double damping = 1.0);
    
    /**
     * Joint spin-phonon relaxation to find true steady state.
     * 
     * Iterates between:
     * 1. Relaxing phonons to equilibrium for current spin configuration
     * 2. Relaxing spins (deterministic sweeps) for current phonon configuration
     * 
     * This finds the self-consistent equilibrium where both spins and phonons
     * are stationary. Required for proper energy conservation in dynamics.
     * 
     * @param tol                   Convergence tolerance for energy and Q changes
     * @param max_iter              Maximum joint relaxation iterations
     * @param spin_sweeps_per_iter  Number of deterministic spin sweeps per iteration
     * @param phonon_only           If true, only relax phonons (keep spins fixed)
     * @return true if converged, false if max_iter reached
     */
    bool relax_joint(double tol = 1e-6, size_t max_iter = 100, size_t spin_sweeps_per_iter = 10, bool phonon_only = false);
    
    // ============================================================
    // SINGLE/DOUBLE PULSE DRIVE (for 2DCS)
    // ============================================================
    
    /**
     * Magnetization trajectory data type
     * Returns: (time, [M_antiferro, M_local, M_global, (O_custom, 0, 0)])
     * The 4th element stores the custom order parameter in the x-component
     */
    using MagTrajectory = vector<std::pair<double, std::array<Eigen::Vector3d, 4>>>;
    
    /**
     * Single pulse THz drive on phonon E1 mode
     * Matches lattice.h::single_pulse_drive signature (adapted for phonon drive)
     * 
     * @param polarization  THz field polarization angle (0=x, π/2=y)
     * @param t_B           Center time of pulse
     * @param pulse_amp     Pulse amplitude (E-field strength)
     * @param pulse_width   Gaussian width (sigma)
     * @param pulse_freq    Carrier frequency
     * @param T_start       Integration start time
     * @param T_end         Integration end time
     * @param step_size     Integration timestep
     * @param method        ODE integration method
     * @return Trajectory of (time, [M_antiferro, M_local, M_global])
     */
    MagTrajectory single_pulse_drive(double polarization, double t_B,
                                     double pulse_amp, double pulse_width, double pulse_freq,
                                     double T_start, double T_end, double step_size,
                                     const string& method = "dopri5");
    
    /**
     * Double pulse THz drive (pump + probe)
     * Matches lattice.h::double_pulse_drive signature (adapted for phonon drive)
     * Both pulses share the same amplitude, width, and frequency
     * 
     * @param polarization_1  Pump pulse polarization angle
     * @param t_B_1           Pump pulse center time
     * @param polarization_2  Probe pulse polarization angle
     * @param t_B_2           Probe pulse center time
     * @param pulse_amp       Pulse amplitude (shared)
     * @param pulse_width     Gaussian width (shared)
     * @param pulse_freq      Carrier frequency (shared)
     * @param T_start         Integration start time
     * @param T_end           Integration end time
     * @param step_size       Integration timestep
     * @param method          ODE integration method
     * @return Trajectory of (time, [M_antiferro, M_local, M_global])
     */
    MagTrajectory double_pulse_drive(double polarization_1, double t_B_1,
                                     double polarization_2, double t_B_2,
                                     double pulse_amp, double pulse_width, double pulse_freq,
                                     double T_start, double T_end, double step_size,
                                     const string& method = "dopri5");
    
    /**
     * Complete 2D coherent spectroscopy (2DCS) workflow
     * Matches lattice.h::pump_probe_spectroscopy signature (adapted for phonon drive)
     * 
     * Performs pump-probe spectroscopy with THz pulses driving the E1 phonon mode:
     * 1. Uses current spin configuration as ground state
     * 2. Runs reference single-pulse dynamics M0 (pump at t=0)
     * 3. Scans delay times (tau) to measure:
     *    - M1(t, tau): Response to probe pulse at time tau only
     *    - M01(t, tau): Response to pump (t=0) + probe (t=tau)
     * 
     * Nonlinear signal extraction: M_NL = M01 - M0 - M1
     * 
     * @param polarization  THz field polarization angle
     * @param pulse_amp     THz pulse amplitude
     * @param pulse_width   Gaussian pulse width
     * @param pulse_freq    Pulse carrier frequency
     * @param tau_start     Initial delay time
     * @param tau_end       Final delay time  
     * @param tau_step      Delay time step
     * @param T_start       Integration start time
     * @param T_end         Integration end time
     * @param T_step        Integration timestep
     * @param dir_name      Output directory
     * @param method        ODE integration method
     */
    void pump_probe_spectroscopy(double polarization,
                                double pulse_amp, double pulse_width, double pulse_freq,
                                double tau_start, double tau_end, double tau_step,
                                double T_start, double T_end, double T_step,
                                const string& dir_name = "spectroscopy",
                                const string& method = "dopri5");
    
    /**
     * MPI-parallelized 2DCS spectroscopy
     * Distributes tau values across MPI ranks
     */
    void pump_probe_spectroscopy_mpi(double polarization,
                                    double pulse_amp, double pulse_width, double pulse_freq,
                                    double tau_start, double tau_end, double tau_step,
                                    double T_start, double T_end, double T_step,
                                    const string& dir_name = "spectroscopy",
                                    const string& method = "dopri5");
    
    // ============================================================
    // I/O
    // ============================================================
    
#ifdef HDF5_ENABLED
    void save_spin_config_hdf5(const string& filename) const;
    void load_spin_config_hdf5(const string& filename);
    void save_state_hdf5(const string& filename) const;
    void load_state_hdf5(const string& filename);
#endif
    
    void save_spin_config(const string& filename) const;
    void load_spin_config(const string& filename);
    void save_positions(const string& filename) const;
    
    void print_state() const {
        cout << "=== PhononLattice State ===" << endl;
        cout << "E1: Qx=" << phonons.Q_x_E1 << ", Qy=" << phonons.Q_y_E1 
             << ", Vx=" << phonons.V_x_E1 << ", Vy=" << phonons.V_y_E1 << endl;
        cout << "E2: Qx=" << phonons.Q_x_E2 << ", Qy=" << phonons.Q_y_E2 
             << ", Vx=" << phonons.V_x_E2 << ", Vy=" << phonons.V_y_E2 << endl;
        cout << "A1: Q=" << phonons.Q_A1 << ", V=" << phonons.V_A1 << endl;
        cout << "E1 amplitude: " << E1_amplitude() << ", E2 amplitude: " << E2_amplitude() << endl;
        cout << "Magnetization: " << magnetization().transpose() << endl;
        cout << "Staggered M: " << staggered_magnetization().transpose() << endl;
        cout << "Energy: " << energy_density() << " per site" << endl;
        cout << "===========================" << endl;
    }
};

#endif // PHONON_LATTICE_H
