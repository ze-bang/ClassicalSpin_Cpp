#ifndef NCTO_LATTICE_H
#define NCTO_LATTICE_H

/**
 * NCTO Lattice: Spin-Phonon coupled system on honeycomb lattice
 * 
 * This class implements a model with:
 * - Classical spins on a honeycomb lattice with Kitaev-Heisenberg-Γ-J3 interactions
 * - Global IR (E1) phonon mode: 2-component (Qx, Qy)
 * - Global Raman (A1) phonon mode: 1-component Q_R
 * - Spin-phonon coupling: exchange parameters depend on Raman mode Q_R
 * - Nonlinear phononics: cubic coupling g * Q_R * (Qx^2 + Qy^2)
 * - THz drive: E(t) couples to IR mode
 * 
 * Hamiltonian:
 * H = H_spin + H_ph + H_sp-ph + H_drive
 * 
 * H_spin = Σ [J1(Q_R) Si·Sj + K(Q_R) Si^γ Sj^γ + Γ(Q_R)(Si^α Sj^β + Si^β Sj^α)] + J3 Σ Si·Sj
 *   where:
 *     J1(Q_R) = J1^0 + λ_J * Q_R
 *     K(Q_R)  = K^0  + λ_K * Q_R  
 *     Γ(Q_R)  = Γ^0  + λ_Γ * Q_R
 * 
 * H_sp-ph = Q_R * Σ [λJ Si·Sj + λK Si^γ Sj^γ + λΓ(Si^α Sj^β + Si^β Sj^α)]
 *   (this is the Q_R-dependent part of H_spin)
 * 
 * H_ph = (1/2)ω_IR^2(Qx^2 + Qy^2) + (1/2)ω_R^2 Q_R^2 + (β/4)Q_R^4 + g*Q_R*(Qx^2+Qy^2)
 * 
 * H_drive = -Z* [Ex(t) Qx + Ey(t) Qy]
 * 
 * Equations of motion:
 * - Spins: LLG (dS/dt = S × H_eff + α S × (S × H_eff))
 * - Phonons: Driven damped harmonic oscillator (Newton + damping)
 *   d²Qx/dt² = -ω_IR² Qx - 2g*Q_R*Qx - γ_IR*(dQx/dt) + Z*Ex(t)
 *   d²Qy/dt² = -ω_IR² Qy - 2g*Q_R*Qy - γ_IR*(dQy/dt) + Z*Ey(t)
 *   d²Q_R/dt² = -ω_R² Q_R - β*Q_R³ - g*(Qx²+Qy²) - γ_R*(dQ_R/dt) - dH_sp-ph/dQ_R
 * 
 * State vector layout: [S0_x, S0_y, S0_z, ..., SN_z, Qx, Qy, Q_R, dQx/dt, dQy/dt, dQ_R/dt]
 */

#include "unitcell.h"
#include "simple_linear_alg.h"
#include <vector>
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
 * Phonon state for NCTO model
 * Stores global phonon coordinates and velocities
 * 
 * IR E1 mode: 2-component (Qx, Qy) with velocities (Vx, Vy)
 * Raman A1 mode: 1-component (Q_R) with velocity (V_R)
 */
struct PhononState {
    // IR E1 mode (2-component)
    double Q_x = 0.0;   // x-component of IR normal coordinate
    double Q_y = 0.0;   // y-component of IR normal coordinate
    
    // Raman A1 mode (1-component)
    double Q_R = 0.0;   // Raman normal coordinate
    
    // Velocities (dQ/dt)
    double V_x = 0.0;   // dQx/dt
    double V_y = 0.0;   // dQy/dt
    double V_R = 0.0;   // dQ_R/dt
    
    // Total number of phonon degrees of freedom
    // 3 coordinates + 3 velocities = 6 DOF for second-order ODE system
    static constexpr size_t N_DOF = 6;
    
    // Convert to flat array (for ODE integration)
    // Layout: [Qx, Qy, Q_R, Vx, Vy, V_R]
    void to_array(double* arr) const {
        arr[0] = Q_x;
        arr[1] = Q_y;
        arr[2] = Q_R;
        arr[3] = V_x;
        arr[4] = V_y;
        arr[5] = V_R;
    }
    
    // Load from flat array
    void from_array(const double* arr) {
        Q_x = arr[0];
        Q_y = arr[1];
        Q_R = arr[2];
        V_x = arr[3];
        V_y = arr[4];
        V_R = arr[5];
    }
    
    // Kinetic energy T = (1/2)(Vx² + Vy²) + (1/2)V_R²
    // (with unit effective masses)
    double kinetic_energy() const {
        return 0.5 * (V_x * V_x + V_y * V_y + V_R * V_R);
    }
};

/**
 * NCTO phonon parameters
 * 
 * Effective masses are set to 1 (absorbed into frequency/coupling definitions)
 */
struct NCTOPhononParams {
    // Frequencies
    double omega_IR = 1.0;  // IR mode frequency ω_IR
    double omega_R = 0.5;   // Raman mode frequency ω_R
    
    // Damping coefficients
    double gamma_IR = 0.1;  // IR mode damping γ_IR
    double gamma_R = 0.05;  // Raman mode damping γ_R
    
    // Anharmonicity
    double beta = 0.0;      // Q_R^4 anharmonic coefficient (stabilization)
    
    // Nonlinear phonon coupling (ionic Raman mechanism)
    double g = 0.0;         // g * Q_R * (Qx^2 + Qy^2) coupling
    
    // Effective charge for THz coupling
    double Z_star = 1.0;    // Z* for THz-IR coupling: -Z* [Ex Qx + Ey Qy]
};

/**
 * NCTO spin-phonon coupling parameters
 */
struct NCTOSpinPhononParams {
    // Base exchange parameters (at Q_R = 0)
    double J1_0 = 0.0;      // Heisenberg exchange
    double K_0 = -1.0;      // Kitaev exchange
    double Gamma_0 = 0.25;  // Gamma exchange
    double Gammap_0 = 0.0;  // Gamma' exchange
    double J3 = 0.0;        // Third-neighbor Heisenberg
    
    // Spin-phonon coupling strengths (linear in Q_R)
    double lambda_J = 0.0;      // dJ1/dQ_R
    double lambda_K = 0.0;      // dK/dQ_R
    double lambda_Gamma = 0.0;  // dΓ/dQ_R
    double lambda_Gammap = 0.0; // dΓ'/dQ_R
    
    // Compute effective exchange parameters at given Q_R
    double J1(double Q_R) const { return J1_0 + lambda_J * Q_R; }
    double K(double Q_R) const { return K_0 + lambda_K * Q_R; }
    double Gamma(double Q_R) const { return Gamma_0 + lambda_Gamma * Q_R; }
    double Gammap(double Q_R) const { return Gammap_0 + lambda_Gammap * Q_R; }
};

/**
 * NCTO THz drive parameters
 */
struct NCTODriveParams {
    // Pulse 1 parameters
    double E0_1 = 0.0;          // Amplitude of pulse 1
    double omega_1 = 1.0;       // Frequency of pulse 1
    double t_1 = 0.0;           // Center time of pulse 1
    double sigma_1 = 1.0;       // Width of pulse 1 (Gaussian)
    double phi_1 = 0.0;         // Phase of pulse 1
    double theta_1 = 0.0;       // Polarization angle of pulse 1 (0 = x, π/2 = y)
    
    // Pulse 2 parameters
    double E0_2 = 0.0;          // Amplitude of pulse 2
    double omega_2 = 1.0;       // Frequency of pulse 2
    double t_2 = 0.0;           // Center time of pulse 2
    double sigma_2 = 1.0;       // Width of pulse 2 (Gaussian)
    double phi_2 = 0.0;         // Phase of pulse 2
    double theta_2 = 0.0;       // Polarization angle of pulse 2
    
    // Compute E-field at time t
    void E_field(double t, double& Ex, double& Ey) const {
        double dt1 = t - t_1;
        double dt2 = t - t_2;
        
        double env1 = std::exp(-0.5 * dt1 * dt1 / (sigma_1 * sigma_1));
        double osc1 = std::cos(omega_1 * dt1 + phi_1);
        double E1 = E0_1 * env1 * osc1;
        
        double env2 = std::exp(-0.5 * dt2 * dt2 / (sigma_2 * sigma_2));
        double osc2 = std::cos(omega_2 * dt2 + phi_2);
        double E2 = E0_2 * env2 * osc2;
        
        Ex = E1 * std::cos(theta_1) + E2 * std::cos(theta_2);
        Ey = E1 * std::sin(theta_1) + E2 * std::sin(theta_2);
    }
};

/**
 * NCTOLattice class: Spin-phonon coupled honeycomb lattice
 * 
 * Degrees of freedom:
 * - N_spin = 2 * dim1 * dim2 classical spins (honeycomb, spin_dim = 3)
 * - 6 global phonon DOF: coordinates (Qx, Qy, Q_R) + velocities (Vx, Vy, V_R)
 * 
 * Total ODE state size: 3 * N_spin + 6
 */
class NCTOLattice {
public:
    using SpinConfig = vector<SpinVector>;
    using ODEState = vector<double>;
    
    // Lattice properties
    size_t spin_dim = 3;         // 3D spins
    size_t N_atoms = 2;          // 2 atoms per honeycomb unit cell
    size_t dim1, dim2, dim3;     // Lattice dimensions (dim3 typically 1)
    size_t lattice_size;         // Total number of spin sites
    float spin_length = 1.0;     // Magnitude of spin vectors
    
    // Spin configuration
    SpinConfig spins;
    vector<Eigen::Vector3d> site_positions;
    
    // Phonon state
    PhononState phonons;
    
    // Spin-spin interactions (stored per site)
    // These are Q_R-dependent, so we store base values
    vector<vector<SpinMatrix>> bilinear_base;       // Base coupling (Q_R = 0)
    vector<vector<SpinMatrix>> bilinear_coupling;   // Spin-phonon coupling strength
    vector<vector<size_t>> bilinear_partners;       // Partner indices
    vector<vector<int>> bilinear_bond_types;        // 0=x, 1=y, 2=z bond for Kitaev
    
    // J3 interactions (not Q_R dependent)
    vector<vector<SpinMatrix>> j3_interaction;
    vector<vector<size_t>> j3_partners;
    
    // External magnetic field
    vector<SpinVector> field;
    
    // Parameters
    NCTOPhononParams phonon_params;
    NCTOSpinPhononParams spin_phonon_params;
    NCTODriveParams drive_params;
    
    // Spin damping for LLG (Gilbert damping)
    double alpha_gilbert = 0.0;  // α in LLG: dS/dt = -γ S × H + α S × (S × H)
    
    // Size of ODE state vector
    size_t state_size;
    
    /**
     * Constructor: Build NCTO lattice
     * 
     * @param d1    Lattice size in first dimension
     * @param d2    Lattice size in second dimension  
     * @param d3    Lattice size in third dimension (usually 1 for 2D)
     * @param spin_l Magnitude of spin vectors
     */
    NCTOLattice(size_t d1, size_t d2, size_t d3 = 1, float spin_l = 1.0)
        : dim1(d1), dim2(d2), dim3(d3), spin_length(spin_l)
    {
        lattice_size = N_atoms * dim1 * dim2 * dim3;
        state_size = spin_dim * lattice_size + PhononState::N_DOF;
        
        // Initialize arrays
        spins.resize(lattice_size);
        site_positions.resize(lattice_size);
        field.resize(lattice_size);
        
        bilinear_base.resize(lattice_size);
        bilinear_coupling.resize(lattice_size);
        bilinear_partners.resize(lattice_size);
        bilinear_bond_types.resize(lattice_size);
        j3_interaction.resize(lattice_size);
        j3_partners.resize(lattice_size);
        
        // Initialize random seed
        seed_lehman(std::chrono::system_clock::now().time_since_epoch().count() * 2 + 1);
        
        cout << "Initializing NCTO lattice with dimensions: " << dim1 << " x " << dim2 << " x " << dim3 << endl;
        cout << "Total spin sites: " << lattice_size << endl;
        cout << "Phonon DOF: " << PhononState::N_DOF << " (Qx, Qy, Q_R, Vx, Vy, V_R)" << endl;
        cout << "Total ODE state size: " << state_size << endl;
        
        // Build honeycomb lattice structure
        build_honeycomb();
        
        // Initialize spins randomly
        init_random();
        
        cout << "NCTO lattice initialization complete!" << endl;
    }
    
    /**
     * Flatten multi-index to linear site index
     */
    size_t flatten_index(size_t i, size_t j, size_t k, size_t atom) const {
        return ((i * dim2 + j) * dim3 + k) * N_atoms + atom;
    }
    
    /**
     * Apply periodic boundary condition
     */
    int periodic_boundary(int coord, size_t dim_size) const {
        if (coord < 0) {
            return coord + dim_size;
        } else if (coord >= (int)dim_size) {
            return coord - dim_size;
        }
        return coord;
    }
    
    /**
     * Flatten with periodic boundaries
     */
    size_t flatten_index_periodic(int i, int j, int k, size_t atom) const {
        return flatten_index(periodic_boundary(i, dim1),
                            periodic_boundary(j, dim2),
                            periodic_boundary(k, dim3),
                            atom);
    }
    
    /**
     * Generate random spin on 2-sphere
     */
    SpinVector gen_random_spin() {
        SpinVector spin(3);
        double z = random_double_lehman(-1.0, 1.0);
        double phi = random_double_lehman(0.0, 2.0 * M_PI);
        double r = std::sqrt(1.0 - z * z);
        spin(0) = r * std::cos(phi);
        spin(1) = r * std::sin(phi);
        spin(2) = z;
        return spin * spin_length;
    }
    
    /**
     * Initialize spins randomly
     */
    void init_random() {
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = gen_random_spin();
        }
        // Reset phonons to equilibrium
        phonons = PhononState();
    }
    
    /**
     * Initialize spins to ferromagnetic state
     */
    void init_ferromagnetic(const Eigen::Vector3d& direction) {
        Eigen::Vector3d dir = direction.normalized() * spin_length;
        for (size_t i = 0; i < lattice_size; ++i) {
            spins[i] = dir;
        }
        phonons = PhononState();
    }
    
    /**
     * Build honeycomb lattice structure and interactions
     */
    void build_honeycomb();
    
    /**
     * Set spin-phonon parameters and rebuild interaction matrices
     */
    void set_parameters(const NCTOSpinPhononParams& sp_params,
                       const NCTOPhononParams& ph_params,
                       const NCTODriveParams& dr_params);
    
    /**
     * Set magnetic field
     */
    void set_field(const Eigen::Vector3d& B) {
        for (size_t i = 0; i < lattice_size; ++i) {
            field[i] = B;
        }
    }
    
    // ============================================================
    // ENERGY CALCULATIONS
    // ============================================================
    
    /**
     * Compute spin energy (including Q_R-dependent exchanges)
     */
    double spin_energy() const;
    
    /**
     * Compute spin-phonon coupling energy
     * H_sp-ph = Q_R * Σ [λJ Si·Sj + λK Si^γ Sj^γ + λΓ(Si^α Sj^β + Si^β Sj^α)]
     */
    double spin_phonon_energy() const;
    
    /**
     * Compute phonon energy (kinetic + potential + nonlinear)
     * 
     * H_ph = (1/2)(Vx² + Vy²) + (1/2)ω_IR²(Qx² + Qy²)
     *      + (1/2)V_R² + (1/2)ω_R² Q_R² + (β/4)Q_R⁴
     *      + g Q_R (Qx² + Qy²)
     */
    double phonon_energy() const {
        // Kinetic energy: T = (1/2)(Vx² + Vy² + V_R²)
        double T = phonons.kinetic_energy();
        
        // IR potential: (1/2)ω_IR²(Qx² + Qy²)
        double Q_IR_sq = phonons.Q_x * phonons.Q_x + phonons.Q_y * phonons.Q_y;
        double V_IR = 0.5 * phonon_params.omega_IR * phonon_params.omega_IR * Q_IR_sq;
        
        // Raman potential: (1/2)ω_R² Q_R²
        double V_R = 0.5 * phonon_params.omega_R * phonon_params.omega_R * phonons.Q_R * phonons.Q_R;
        
        // Anharmonic: (β/4)Q_R⁴
        double V_anharm = 0.25 * phonon_params.beta * std::pow(phonons.Q_R, 4);
        
        // Nonlinear phonon coupling: g Q_R (Qx² + Qy²)
        double V_coupling = phonon_params.g * phonons.Q_R * Q_IR_sq;
        
        return T + V_IR + V_R + V_anharm + V_coupling;
    }
    
    /**
     * Compute total energy
     */
    double total_energy() const {
        return spin_energy() + phonon_energy();
    }
    
    /**
     * Energy per spin site
     */
    double energy_density() const {
        return total_energy() / lattice_size;
    }
    
    // ============================================================
    // EQUATIONS OF MOTION
    // ============================================================
    
    /**
     * Compute effective field on spin at site i
     * H_eff = -dH_spin/dS_i (includes Q_R-dependent exchanges)
     */
    SpinVector get_local_field(size_t site, double Q_R) const;
    
    /**
     * Compute derivative of spin-phonon energy with respect to Q_R
     * Used for Raman mode equation of motion
     */
    double dH_spin_dQR() const;
    
    /**
     * ODE system for spin-phonon dynamics
     * State layout: [S0_x, S0_y, S0_z, S1_x, ..., SN_z, Qx, Qy, Q_R, Vx, Vy, V_R]
     */
    void ode_system(const ODEState& x, ODEState& dxdt, double t);
    
    /**
     * Spin LLG equation: dS/dt = -γ S × H_eff + α S × (S × H_eff)
     * Returns derivative for spin at given site
     */
    Eigen::Vector3d spin_derivative(const Eigen::Vector3d& S, const Eigen::Vector3d& H_eff) const {
        Eigen::Vector3d dSdt = S.cross(H_eff);  // -γ S × H (γ = 1 in our units)
        
        if (alpha_gilbert > 0) {
            // Gilbert damping term
            dSdt += alpha_gilbert * S.cross(S.cross(H_eff)) / spin_length;
        }
        
        return dSdt;
    }
    
    /**
     * Phonon equations of motion (driven damped harmonic oscillator)
     * Second-order Newton form with damping:
     * 
     * dQx/dt = Vx
     * dVx/dt = -ω_IR² Qx - 2g*Q_R*Qx - γ_IR*Vx + Z*Ex(t)
     * 
     * dQy/dt = Vy
     * dVy/dt = -ω_IR² Qy - 2g*Q_R*Qy - γ_IR*Vy + Z*Ey(t)
     * 
     * dQ_R/dt = V_R
     * dV_R/dt = -ω_R² Q_R - β*Q_R³ - g*(Qx²+Qy²) - γ_R*V_R - dH_sp-ph/dQ_R
     * 
     * @param ph      Current phonon state
     * @param t       Current time
     * @param dH_dQR  Derivative of spin-phonon energy w.r.t. Q_R
     * @param dQx, dQy, dQR  Output: time derivatives of coordinates
     * @param dVx, dVy, dVR  Output: time derivatives of velocities (accelerations)
     */
    void phonon_derivatives(const PhononState& ph, double t, double dH_dQR,
                           double& dQx, double& dQy, double& dQR,
                           double& dVx, double& dVy, double& dVR) const;
    
    // ============================================================
    // STATE CONVERSION
    // ============================================================
    
    /**
     * Convert current state to flat ODE state vector
     */
    ODEState to_state() const {
        ODEState state(state_size);
        
        // Pack spins
        size_t idx = 0;
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t d = 0; d < spin_dim; ++d) {
                state[idx++] = spins[i](d);
            }
        }
        
        // Pack phonons
        phonons.to_array(&state[idx]);
        
        return state;
    }
    
    /**
     * Update lattice from flat ODE state vector
     */
    void from_state(const ODEState& state) {
        // Unpack spins
        size_t idx = 0;
        for (size_t i = 0; i < lattice_size; ++i) {
            for (size_t d = 0; d < spin_dim; ++d) {
                spins[i](d) = state[idx++];
            }
            // Normalize spins to maintain unit length
            spins[i] = spins[i].normalized() * spin_length;
        }
        
        // Unpack phonons
        phonons.from_array(&state[idx]);
    }
    
    // ============================================================
    // OBSERVABLES
    // ============================================================
    
    /**
     * Compute total magnetization
     */
    Eigen::Vector3d magnetization() const {
        Eigen::Vector3d M = Eigen::Vector3d::Zero();
        for (const auto& s : spins) {
            M += s;
        }
        return M / lattice_size;
    }
    
    /**
     * Compute staggered magnetization (antiferromagnetic order parameter)
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
     * Compute IR phonon amplitude |Q_IR| = sqrt(Qx^2 + Qy^2)
     */
    double IR_amplitude() const {
        return std::sqrt(phonons.Q_x * phonons.Q_x + phonons.Q_y * phonons.Q_y);
    }
    
    // ============================================================
    // SIMULATION METHODS
    // ============================================================
    
private:
    /**
     * Helper: Execute ODE integration with selected method
     * Centralizes integrator selection logic to reduce code duplication
     * 
     * @param system_func   ODE system function (dx/dt = f(x, t))
     * @param state         Initial state vector (modified in-place)
     * @param T_start       Integration start time
     * @param T_end         Integration end time
     * @param dt_step       Time step (fixed for const methods, initial for adaptive)
     * @param observer      Observer function called at each step
     * @param method        Integration method (see list below)
     * @param use_adaptive  If true, use integrate_adaptive; if false, use integrate_const
     * @param abs_tol       Absolute tolerance for adaptive methods
     * @param rel_tol       Relative tolerance for adaptive methods
     * 
     * Available methods:
     * - "euler": Explicit Euler (1st order, simple, inaccurate)
     * - "rk2" or "midpoint": Runge-Kutta 2nd order
     * - "rk4": Classic Runge-Kutta 4th order (good balance, fixed step)
     * - "dopri5": Dormand-Prince 5(4) (default, recommended for general use)
     * - "rk78" or "rkf78": Runge-Kutta-Fehlberg 7(8) (high accuracy, expensive)
     * - "bulirsch_stoer" or "bs": Bulirsch-Stoer (very high accuracy, expensive)
     */
    template<typename System, typename Observer>
    void integrate_ode_system(System system_func, ODEState& state,
                             double T_start, double T_end, double dt_step,
                             Observer observer, const string& method,
                             bool use_adaptive = false,
                             double abs_tol = 1e-6, double rel_tol = 1e-6);

public:
    /**
     * Run molecular dynamics simulation using Boost.Odeint
     * Requires HDF5 for output.
     * 
     * @param T_start       Start time
     * @param T_end         End time  
     * @param dt_initial    Initial step size (adaptive methods will adjust)
     * @param out_dir       Output directory for trajectory HDF5 file
     * @param save_interval Number of steps between saves
     * @param method        Integration method: "euler", "rk2", "rk4", "dopri5" (default),
     *                      "rk78", "bulirsch_stoer"
     */
    void molecular_dynamics(double T_start, double T_end, double dt_initial,
                           string out_dir = "", size_t save_interval = 100,
                           string method = "dopri5");
    
    /**
     * Run simulated annealing for spin subsystem only
     * (Phonons kept at equilibrium during thermal equilibration)
     * 
     * @param T_start        Starting temperature
     * @param T_end          Ending temperature
     * @param n_steps        Number of annealing steps
     * @param overrelax_rate Overrelaxation sweep frequency (0 = disabled)
     * @param cooling_rate   Geometric cooling factor per step
     * @param out_dir        Output directory for HDF5 file
     * @param save_observables Whether to save observables during annealing
     */
    void simulated_annealing(double T_start, double T_end, size_t n_steps,
                            size_t overrelax_rate = 0,
                            double cooling_rate = 0.9,
                            string out_dir = "",
                            bool save_observables = true);
    
    // ============================================================
    // I/O
    // ============================================================
    
#ifdef HDF5_ENABLED
    /**
     * Save spin configuration to HDF5 file
     */
    void save_spin_config_hdf5(const string& filename) const;
    
    /**
     * Load spin configuration from HDF5 file
     */
    void load_spin_config_hdf5(const string& filename);
    
    /**
     * Save complete state (spins + phonons) to HDF5 file
     */
    void save_state_hdf5(const string& filename) const;
    
    /**
     * Load complete state (spins + phonons) from HDF5 file
     */
    void load_state_hdf5(const string& filename);
#endif
    
    /**
     * Save spin configuration to text file (legacy format)
     */
    void save_spin_config(const string& filename) const;
    
    /**
     * Load spin configuration from text file (legacy format)
     */
    void load_spin_config(const string& filename);
    
    /**
     * Save positions to text file
     */
    void save_positions(const string& filename) const;
    
    /**
     * Print current state summary
     */
    void print_state() const {
        cout << "=== NCTO State ===" << endl;
        cout << "Phonon Q: Qx=" << phonons.Q_x << ", Qy=" << phonons.Q_y 
             << ", Q_R=" << phonons.Q_R << endl;
        cout << "Phonon V: Vx=" << phonons.V_x << ", Vy=" << phonons.V_y 
             << ", V_R=" << phonons.V_R << endl;
        cout << "IR amplitude |Q_IR|: " << IR_amplitude() << endl;
        cout << "Magnetization: " << magnetization().transpose() << endl;
        cout << "Staggered M: " << staggered_magnetization().transpose() << endl;
        cout << "Energy: " << energy_density() << " per site" << endl;
        cout << "=================" << endl;
    }
};

#endif // NCTO_LATTICE_H
