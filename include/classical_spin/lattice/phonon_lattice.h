/**
 * @file phonon_lattice.h
 * @brief Spin-phonon coupled honeycomb lattice
 * 
 * This implements a honeycomb lattice with:
 * - Generic NN, 2nd NN, and 3rd NN spin-spin interactions (bilinear matrix form)
 * - Sublattice-dependent 2nd NN coupling (J2_A for sublattice A, J2_B for sublattice B)
 * - Two phonon modes: E1 (Qx, Qy) and A1 (Q_R)
 * - Three-phonon coupling: g3 * (Qx² + Qy²) * Q_R
 * - Spin-phonon coupling: Qx*(SxSz + SzSx) + Qy*(SySz + SzSy) + Q_R*(SxSx + SySy + SzSz)
 * - THz drive coupling to E1 mode
 * 
 * Hamiltonian:
 * H = H_spin + H_phonon + H_sp-ph + H_drive
 * 
 * H_spin = Σ_<ij> Si · J1 · Sj + Σ_<<ij>>_A J2_A Si·Sj + Σ_<<ij>>_B J2_B Si·Sj
 *        + Σ_<<<ij>>> Si · J3 · Sj - Σ_i B · Si
 *   (NN J1 is bond-dependent Kitaev-Heisenberg-Γ-Γ', 2nd and 3rd NN are isotropic Heisenberg)
 * 
 * H_phonon = (1/2)(Vx² + Vy²) + (1/2)ω_E²(Qx² + Qy²) + (λ_E/4)(Qx² + Qy²)²
 *          + (1/2)V_R² + (1/2)ω_A²*Q_R² + (λ_A/4)*Q_R⁴
 *          + g3*(Qx² + Qy²)*Q_R
 * 
 * H_sp-ph = Σ_<ij> [λ_xy * Qx * (Si_x*Sj_z + Si_z*Sj_x)
 *                 + λ_xy * Qy * (Si_y*Sj_z + Si_z*Sj_y)
 *                 + λ_R  * Q_R * (Si_x*Sj_x + Si_y*Sj_y + Si_z*Sj_z)]
 * 
 * H_drive = -E_x(t) * Qx - E_y(t) * Qy
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
 * E1 mode: 2-component (Qx, Qy) with velocities (Vx, Vy)
 * A1 mode: 1-component (Q_R) with velocity (V_R)
 */
struct PhononState {
    // E1 mode (infrared active, 2-component)
    double Q_x = 0.0;   // x-component of E1 normal coordinate
    double Q_y = 0.0;   // y-component of E1 normal coordinate
    
    // A1 mode (Raman active, 1-component)
    double Q_R = 0.0;   // A1 normal coordinate
    
    // Velocities (dQ/dt)
    double V_x = 0.0;   // dQx/dt
    double V_y = 0.0;   // dQy/dt
    double V_R = 0.0;   // dQ_R/dt
    
    // Total DOF: 3 coordinates + 3 velocities = 6
    static constexpr size_t N_DOF = 6;
    
    // Pack to flat array: [Qx, Qy, Q_R, Vx, Vy, V_R]
    void to_array(double* arr) const {
        arr[0] = Q_x; arr[1] = Q_y; arr[2] = Q_R;
        arr[3] = V_x; arr[4] = V_y; arr[5] = V_R;
    }
    
    // Unpack from flat array
    void from_array(const double* arr) {
        Q_x = arr[0]; Q_y = arr[1]; Q_R = arr[2];
        V_x = arr[3]; V_y = arr[4]; V_R = arr[5];
    }
    
    // Kinetic energy (unit mass)
    double kinetic_energy() const {
        return 0.5 * (V_x*V_x + V_y*V_y + V_R*V_R);
    }
};

/**
 * Phonon parameters
 */
struct PhononParams {
    // E1 mode (IR active)
    double omega_E = 1.0;   // E1 mode frequency
    double gamma_E = 0.1;   // E1 mode damping
    double mass_E = 1.0;    // E1 mode effective mass (set to 1, absorb in other params)
    
    // A1 mode (Raman active)
    double omega_A = 0.5;   // A1 mode frequency
    double gamma_A = 0.05;  // A1 mode damping
    double mass_A = 1.0;    // A1 mode effective mass
    
    // Three-phonon coupling: g3 * (Qx² + Qy²) * Q_R
    double g3 = 0.0;        // Cubic phonon-phonon coupling
    
    // Quartic stabilization terms: (λ_E/4)(Qx² + Qy²)² + (λ_A/4)*Q_R⁴
    double lambda_E = 0.0;  // E1 mode quartic coefficient (restores stability)
    double lambda_A = 0.0;  // A1 mode quartic coefficient (restores stability)
    
    // THz coupling strength
    double Z_star = 1.0;    // Effective charge for E(t) coupling
};

/**
 * Spin-phonon coupling parameters
 * 
 * Uses Kitaev-Heisenberg-Γ-Γ' model with bond-dependent exchange:
 * H_spin = Σ_<ij>_γ Si · J_γ · Sj  where γ ∈ {x, y, z} is the bond type
 * 
 * For honeycomb lattice:
 *   Jx (x-bond): K on xx, Γ on yz/zy, Γ' on xy/xz  
 *   Jy (y-bond): K on yy, Γ on xz/zx, Γ' on xy/yz
 *   Jz (z-bond): K on zz, Γ on xy/yx, Γ' on xz/yz
 * 
 * Spin-phonon coupling form on each bond <ij>:
 *   λ_xy * Qx * (Si_x*Sj_z + Si_z*Sj_x)
 * + λ_xy * Qy * (Si_y*Sj_z + Si_z*Sj_y)  
 * + λ_R  * Q_R * (Si_x*Sj_x + Si_y*Sj_y + Si_z*Sj_z)
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
    
    // Spin-phonon coupling strengths
    double lambda_xy = 0.0;  // Coupling for E1 mode (Qx, Qy)
    double lambda_R = 0.0;   // Coupling for A1 mode (Q_R)
    
    // Build bond-dependent exchange matrices
    SpinMatrix get_Jx() const {
        SpinMatrix Jx = SpinMatrix::Zero(3, 3);
        Jx << J + K, Gammap, Gammap,
              Gammap, J, Gamma,
              Gammap, Gamma, J;
        return Jx;
    }
    
    SpinMatrix get_Jy() const {
        SpinMatrix Jy = SpinMatrix::Zero(3, 3);
        Jy << J, Gammap, Gamma,
              Gammap, J + K, Gammap,
              Gamma, Gammap, J;
        return Jy;
    }
    
    SpinMatrix get_Jz() const {
        SpinMatrix Jz = SpinMatrix::Zero(3, 3);
        Jz << J, Gamma, Gammap,
              Gamma, J, Gammap,
              Gammap, Gammap, J + K;
        return Jz;
    }
    
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
 * - 6 global phonon DOF: (Qx, Qy, Q_R, Vx, Vy, V_R)
 * 
 * Total ODE state size: 3 * N_spin + 6
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
    
    // External field
    vector<SpinVector> field;
    
    // Parameters
    PhononParams phonon_params;
    SpinPhononCouplingParams spin_phonon_params;
    DriveParams drive_params;
    
    // LLG damping
    double alpha_gilbert = 0.0;
    
    // ODE state size
    size_t state_size;
    
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
     * Pure spin energy (NN + 3rd NN + Zeeman)
     */
    double spin_energy() const;
    
    /**
     * Phonon energy (kinetic + potential + cubic coupling)
     * 
     * E_ph = (1/2)(Vx² + Vy²) + (1/2)ω_E²(Qx² + Qy²)
     *      + (1/2)V_R² + (1/2)ω_A²*Q_R²
     *      + g3*(Qx² + Qy²)*Q_R
     */
    double phonon_energy() const;
    
    /**
     * Spin-phonon coupling energy
     * 
     * H_sp-ph = Σ_<ij> [λ_xy * Qx * (Si_x*Sj_z + Si_z*Sj_x)
     *                 + λ_xy * Qy * (Si_y*Sj_z + Si_z*Sj_y)
     *                 + λ_R  * Q_R * (Si · Sj)]
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
     * Compute ∂H_sp-ph/∂Qx = Σ_<ij> λ_xy * (Si_x*Sj_z + Si_z*Sj_x)
     */
    double dH_dQx() const;
    
    /**
     * Compute ∂H_sp-ph/∂Qy = Σ_<ij> λ_xy * (Si_y*Sj_z + Si_z*Sj_y)
     */
    double dH_dQy() const;
    
    /**
     * Compute ∂H_sp-ph/∂Q_R = Σ_<ij> λ_R * (Si · Sj)
     */
    double dH_dQR() const;
    
    /**
     * Compute effective field on spin i (for spin EOM)
     * 
     * H_eff = -∂H/∂Si = B + Σ_j [NN contributions] + [spin-phonon contributions]
     */
    SpinVector get_local_field(size_t site) const;
    
    // ============================================================
    // EQUATIONS OF MOTION
    // ============================================================
    
    /**
     * Phonon EOM derivatives
     * 
     * dQx/dt = Vx
     * dVx/dt = -ω_E² Qx - 2*g3*Qx*Q_R - γ_E*Vx - ∂H_sp-ph/∂Qx + Z*Ex(t)
     * 
     * dQy/dt = Vy  
     * dVy/dt = -ω_E² Qy - 2*g3*Qy*Q_R - γ_E*Vy - ∂H_sp-ph/∂Qy + Z*Ey(t)
     * 
     * dQ_R/dt = V_R
     * dV_R/dt = -ω_A² Q_R - g3*(Qx² + Qy²) - γ_A*V_R - ∂H_sp-ph/∂Q_R
     */
    void phonon_derivatives(const PhononState& ph, double t,
                           double dH_dQx_val, double dH_dQy_val, double dH_dQR_val,
                           double& dQx, double& dQy, double& dQR,
                           double& dVx, double& dVy, double& dVR) const;
    
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
     * E1 phonon amplitude
     */
    double E1_amplitude() const {
        return std::sqrt(phonons.Q_x * phonons.Q_x + phonons.Q_y * phonons.Q_y);
    }
    
    // ============================================================
    // SIMULATION
    // ============================================================
    
private:
    template<typename System, typename Observer>
    void integrate_ode_system(System system_func, ODEState& state,
                             double T_start, double T_end, double dt_step,
                             Observer observer, const string& method,
                             bool use_adaptive = false,
                             double abs_tol = 1e-6, double rel_tol = 1e-6);

public:
    /**
     * Run molecular dynamics simulation
     */
    void molecular_dynamics(double T_start, double T_end, double dt_initial,
                           string out_dir = "", size_t save_interval = 100,
                           string method = "dopri5");
    
    /**
     * Simulated annealing for spin subsystem
     */
    void simulated_annealing(double T_start, double T_end, size_t n_steps,
                            size_t overrelax_rate = 0,
                            double cooling_rate = 0.9,
                            string out_dir = "",
                            bool save_observables = true);
    
    // ============================================================
    // SINGLE/DOUBLE PULSE DRIVE (for 2DCS)
    // ============================================================
    
    /**
     * Magnetization trajectory data type
     * Returns: (time, [M_antiferro, M_local, M_global])
     */
    using MagTrajectory = vector<std::pair<double, std::array<Eigen::Vector3d, 3>>>;
    
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
        cout << "Phonon Q: Qx=" << phonons.Q_x << ", Qy=" << phonons.Q_y 
             << ", Q_R=" << phonons.Q_R << endl;
        cout << "Phonon V: Vx=" << phonons.V_x << ", Vy=" << phonons.V_y 
             << ", V_R=" << phonons.V_R << endl;
        cout << "E1 amplitude: " << E1_amplitude() << endl;
        cout << "Magnetization: " << magnetization().transpose() << endl;
        cout << "Staggered M: " << staggered_magnetization().transpose() << endl;
        cout << "Energy: " << energy_density() << " per site" << endl;
        cout << "===========================" << endl;
    }
};

#endif // PHONON_LATTICE_H
