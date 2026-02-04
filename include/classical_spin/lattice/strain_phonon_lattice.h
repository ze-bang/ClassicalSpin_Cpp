/**
 * @file strain_phonon_lattice.h
 * @brief Spin-strain coupled honeycomb lattice with magnetoelastic coupling
 * 
 * This implements a honeycomb lattice with:
 * - Generic NN, 2nd NN, and 3rd NN spin-spin interactions (Kitaev-Heisenberg-Γ-Γ')
 * - Strain field phonons: A1g (ε_xx + ε_yy) and Eg (ε_xx - ε_yy, 2ε_xy)
 * - Magnetoelastic coupling through A1g and Eg symmetry channels
 * 
 * Hamiltonian:
 * H = H_spin + H_strain + H_magnetoelastic + H_drive
 * 
 * H_spin = Σ_<ij> Si · J_global · Sj + Σ_<<ij>>_A J2_A Si·Sj + Σ_<<ij>>_B J2_B Si·Sj
 *        + Σ_<<<ij>>> Si · J3 · Sj - Σ_i B · Si
 * 
 * H_strain = (1/2) Σ_r [C11(ε_xx² + ε_yy²) + 2C12 ε_xx ε_yy + C44 ε_xy²]
 *          + (1/2) Σ_r M [V_xx² + V_yy² + V_xy²]   (kinetic energy)
 *          (V_ij = dε_ij/dt are strain velocities)
 * 
 * Magnetoelastic coupling (D3d point group symmetry):
 * H_c = H_c^{A1g} + H_c^{Eg}
 * 
 * H_c^{A1g} = λ_{A1g} Σ_r (ε_xx + ε_yy) {(J+K)f_K^{A1g} + J f_J^{A1g} + Γ f_Γ^{A1g}}
 * 
 * H_c^{Eg} = λ_{Eg} Σ_r {(ε_xx - ε_yy)[(J+K)f_K^{Eg,1} + J f_J^{Eg,1} + Γ f_Γ^{Eg,1}]
 *                       + 2ε_xy[(J+K)f_K^{Eg,2} + J f_J^{Eg,2} + Γ f_Γ^{Eg,2}]}
 * 
 * where f_X^{IRR} are the spin basis functions for the D3d point group irreps.
 * 
 * Equations of motion (Euler-Lagrange):
 * - Spins: dS/dt = S × H_eff + α S × (S × H_eff)  (LLG with Gilbert damping)
 * - Strain: M d²ε_ij/dt² = -∂H/∂ε_ij - γ dε_ij/dt  (damped oscillator)
 */

#ifndef STRAIN_PHONON_LATTICE_H
#define STRAIN_PHONON_LATTICE_H

#include "unitcell.h"
#include "simple_linear_alg.h"
#include "classical_spin/core/spin_config.h"  // For should_rank_write
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
 * Strain field state
 * 
 * The strain tensor in 2D has 3 independent components: ε_xx, ε_yy, ε_xy
 * These transform under the D3d point group as:
 * - A1g: ε_xx + ε_yy (breathing mode)
 * - Eg: (ε_xx - ε_yy, 2ε_xy) (shear modes)
 * 
 * Each bond type can have independent strain fields (for local coupling).
 */
struct StrainState {
    static constexpr size_t N_BONDS = 3;  // Number of bond types (x=0, y=1, z=2)
    
    // Strain components per bond type
    double epsilon_xx[N_BONDS] = {0.0, 0.0, 0.0};
    double epsilon_yy[N_BONDS] = {0.0, 0.0, 0.0};
    double epsilon_xy[N_BONDS] = {0.0, 0.0, 0.0};
    
    // Strain velocities (time derivatives)
    double V_xx[N_BONDS] = {0.0, 0.0, 0.0};
    double V_yy[N_BONDS] = {0.0, 0.0, 0.0};
    double V_xy[N_BONDS] = {0.0, 0.0, 0.0};
    
    // Total DOF: 3 strain components × 3 bond types × 2 (coords + velocities) = 18
    static constexpr size_t N_DOF = 18;
    
    // Pack to flat array
    void to_array(double* arr) const {
        size_t idx = 0;
        for (size_t b = 0; b < N_BONDS; ++b) arr[idx++] = epsilon_xx[b];
        for (size_t b = 0; b < N_BONDS; ++b) arr[idx++] = epsilon_yy[b];
        for (size_t b = 0; b < N_BONDS; ++b) arr[idx++] = epsilon_xy[b];
        for (size_t b = 0; b < N_BONDS; ++b) arr[idx++] = V_xx[b];
        for (size_t b = 0; b < N_BONDS; ++b) arr[idx++] = V_yy[b];
        for (size_t b = 0; b < N_BONDS; ++b) arr[idx++] = V_xy[b];
    }
    
    // Unpack from flat array
    void from_array(const double* arr) {
        size_t idx = 0;
        for (size_t b = 0; b < N_BONDS; ++b) epsilon_xx[b] = arr[idx++];
        for (size_t b = 0; b < N_BONDS; ++b) epsilon_yy[b] = arr[idx++];
        for (size_t b = 0; b < N_BONDS; ++b) epsilon_xy[b] = arr[idx++];
        for (size_t b = 0; b < N_BONDS; ++b) V_xx[b] = arr[idx++];
        for (size_t b = 0; b < N_BONDS; ++b) V_yy[b] = arr[idx++];
        for (size_t b = 0; b < N_BONDS; ++b) V_xy[b] = arr[idx++];
    }
    
    // Kinetic energy (with unit mass)
    double kinetic_energy() const {
        double T = 0.0;
        for (size_t b = 0; b < N_BONDS; ++b) {
            T += V_xx[b] * V_xx[b] + V_yy[b] * V_yy[b] + V_xy[b] * V_xy[b];
        }
        return 0.5 * T;
    }
    
    // A1g amplitude (ε_xx + ε_yy) for specific bond type
    double A1g_amplitude(size_t bond_type) const {
        return epsilon_xx[bond_type] + epsilon_yy[bond_type];
    }
    
    // Eg amplitude sqrt((ε_xx - ε_yy)² + 4ε_xy²) for specific bond type
    double Eg_amplitude(size_t bond_type) const {
        double d = epsilon_xx[bond_type] - epsilon_yy[bond_type];
        return std::sqrt(d * d + 4.0 * epsilon_xy[bond_type] * epsilon_xy[bond_type]);
    }
    
    // Total A1g amplitude (averaged over bonds)
    double A1g_amplitude() const {
        double sum = 0.0;
        for (size_t b = 0; b < N_BONDS; ++b) {
            sum += A1g_amplitude(b);
        }
        return sum / N_BONDS;
    }
    
    // Total Eg amplitude (averaged over bonds)
    double Eg_amplitude() const {
        double sum = 0.0;
        for (size_t b = 0; b < N_BONDS; ++b) {
            sum += Eg_amplitude(b);
        }
        return sum / N_BONDS;
    }
};

/**
 * Elastic parameters (in units of stiffness)
 */
struct ElasticParams {
    // Elastic constants (Voigt notation)
    double C11 = 1.0;    // Longitudinal stiffness
    double C12 = 0.3;    // Off-diagonal coupling
    double C44 = 0.35;   // Shear stiffness (for cubic: C44 = (C11-C12)/2)
    
    // Effective mass for strain dynamics
    double M = 1.0;
    
    // Damping coefficients
    double gamma_A1g = 0.1;   // A1g mode damping
    double gamma_Eg = 0.1;    // Eg mode damping
    
    // Characteristic frequencies (derived from elastic constants)
    // ω_A1g² ≈ (C11 + C12) / M
    // ω_Eg² ≈ (C11 - C12) / M ≈ 2C44 / M
    double omega_A1g() const { return std::sqrt((C11 + C12) / M); }
    double omega_Eg() const { return std::sqrt(2.0 * C44 / M); }
    
    // Optional: quartic anharmonicity
    double lambda_A1g = 0.0;  // A1g quartic coefficient
    double lambda_Eg = 0.0;   // Eg quartic coefficient
};

/**
 * Magnetoelastic coupling parameters
 * 
 * Coupling is through the spin basis functions f_X^{IRR} for D3d irreps.
 * These are bilinear forms in the spins on nearest-neighbor bonds.
 */
struct MagnetoelasticParams {
    // Kitaev-Heisenberg-Γ-Γ' parameters
    double J = -0.1;       // Heisenberg coupling
    double K = -9.0;       // Kitaev coupling
    double Gamma = 1.8;    // Γ (off-diagonal symmetric)
    double Gammap = 0.3;   // Γ' (off-diagonal asymmetric)
    
    // 2nd NN exchange (isotropic Heisenberg, sublattice-dependent)
    double J2_A = 0.3;
    double J2_B = 0.3;
    
    // 3rd NN exchange (isotropic Heisenberg)
    double J3 = 0.9;
    
    // Six-spin ring exchange
    double J7 = 0.0;
    
    // Gamma parameter for time-dependent J7 modulation via strain
    // J7(t) = J7 * (1 - γ*|δε_Eg|/4)^4 * (1 + γ*|δε_Eg|/2)^2
    // where |δε_Eg| is the Eg strain deviation from equilibrium
    double gamma_J7 = 0.0;  // Set to 0 to disable time dependence
    
    // Magnetoelastic coupling constants
    double lambda_A1g = 0.0;  // A1g channel coupling
    double lambda_Eg = 0.0;   // Eg channel coupling
    
    /**
     * Get the Kitaev local-to-global rotation matrix R.
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
    
    // Build bond-dependent exchange matrices
    // NOTE: We work in the LOCAL Kitaev frame where:
    //   - x-bond has Kitaev term K*S^x*S^x (K on diagonal element 0,0)
    //   - y-bond has Kitaev term K*S^y*S^y (K on diagonal element 1,1)
    //   - z-bond has Kitaev term K*S^z*S^z (K on diagonal element 2,2)
    // This ensures the spin basis functions f_K, f_J, f_Gamma are correctly computed
    // using the bond_type index directly as the spin component.
    SpinMatrix get_Jx() const { return get_Jx_local(); }
    SpinMatrix get_Jy() const { return get_Jy_local(); }
    SpinMatrix get_Jz() const { return get_Jz_local(); }
    
    // J2 and J3 matrices
    SpinMatrix get_J3_matrix() const { return J3 * SpinMatrix::Identity(3, 3); }
    SpinMatrix get_J2_A_matrix() const { return J2_A * SpinMatrix::Identity(3, 3); }
    SpinMatrix get_J2_B_matrix() const { return J2_B * SpinMatrix::Identity(3, 3); }
};

/**
 * Strain drive parameters (acoustic excitation)
 */
struct StrainDriveParams {
    // Pulse 1
    double E0_1 = 0.0;      // Amplitude
    double omega_1 = 1.0;   // Frequency
    double t_1 = 0.0;       // Center time
    double sigma_1 = 1.0;   // Gaussian width
    double phi_1 = 0.0;     // Phase
    
    // Pulse 2
    double E0_2 = 0.0;
    double omega_2 = 1.0;
    double t_2 = 0.0;
    double sigma_2 = 1.0;
    double phi_2 = 0.0;
    
    // Drive strength for A1g mode
    double drive_strength_A1g = 1.0;
    
    // Drive strength for Eg mode (separate for Eg1 and Eg2)
    double drive_strength_Eg1 = 0.0;  // ε_xx - ε_yy component
    double drive_strength_Eg2 = 0.0;  // 2ε_xy component
    
    // Compute drive force at time t
    double drive_force(double t) const {
        double dt1 = t - t_1;
        double dt2 = t - t_2;
        
        double env1 = std::exp(-0.5 * dt1 * dt1 / (sigma_1 * sigma_1));
        double osc1 = std::cos(omega_1 * dt1 + phi_1);
        double F1 = E0_1 * env1 * osc1;
        
        double env2 = std::exp(-0.5 * dt2 * dt2 / (sigma_2 * sigma_2));
        double osc2 = std::cos(omega_2 * dt2 + phi_2);
        double F2 = E0_2 * env2 * osc2;
        
        return F1 + F2;
    }
    
    // Get A1g drive force
    double A1g_force(double t) const {
        return drive_strength_A1g * drive_force(t);
    }
    
    // Get Eg1 drive force (ε_xx - ε_yy)
    double Eg1_force(double t) const {
        return drive_strength_Eg1 * drive_force(t);
    }
    
    // Get Eg2 drive force (2ε_xy)
    double Eg2_force(double t) const {
        return drive_strength_Eg2 * drive_force(t);
    }
};

// ============================================================
// HELPER STRUCTS FOR PARALLEL TEMPERING
// ============================================================

/**
 * Binning analysis result for error estimation
 */
struct SPL_BinningResult {
    double mean;
    double error;
    double tau_int;  // Integrated autocorrelation time estimate
    size_t optimal_bin_level;
    vector<double> errors_by_level;
};

/**
 * Observable with uncertainty (mean ± error)
 */
struct SPL_Observable {
    double value;
    double error;
    
    SPL_Observable(double v = 0.0, double e = 0.0) : value(v), error(e) {}
};

/**
 * Vector observable with uncertainty for each component
 */
struct SPL_VectorObservable {
    vector<double> values;
    vector<double> errors;
    
    SPL_VectorObservable() = default;
    SPL_VectorObservable(size_t dim) : values(dim, 0.0), errors(dim, 0.0) {}
};

/**
 * Complete set of thermodynamic observables with uncertainties
 */
struct SPL_ThermodynamicObservables {
    double temperature;
    SPL_Observable energy;                      // <E>/N
    SPL_Observable specific_heat;               // C_V = (<E²> - <E>²) / (T² N)
    vector<SPL_VectorObservable> sublattice_magnetization;  // <S_α> for each sublattice α
    vector<SPL_VectorObservable> energy_sublattice_cross;   // <E * S_α> - <E><S_α>
};

/**
 * Result from optimized temperature grid generation
 * Based on Bittner et al., Phys. Rev. Lett. 101, 130603 (2008)
 */
struct SPL_OptimizedTempGridResult {
    vector<double> temperatures;              // Optimized temperature ladder
    vector<double> acceptance_rates;          // Final acceptance rates between adjacent pairs
    vector<double> local_diffusivities;       // Local diffusivity D(T) ∝ A(1-A) at each T
    double mean_acceptance_rate;              // Average acceptance rate across all pairs
    double round_trip_estimate;               // Estimated round-trip time in sweeps
    size_t feedback_iterations_used;          // Number of feedback iterations performed
    bool converged;                           // Whether the algorithm converged
};

/**
 * StrainPhononLattice: Honeycomb lattice with magnetoelastic (spin-strain) coupling
 * 
 * Degrees of freedom:
 * - N_spin = 2 * dim1 * dim2 * dim3 classical spins (honeycomb, spin_dim = 3)
 * - 18 strain DOF: (ε_xx, ε_yy, ε_xy, V_xx, V_yy, V_xy) × 3 bond types
 * 
 * Total ODE state size: 3 * N_spin + 18
 */
class StrainPhononLattice {
public:
    using SpinConfig = vector<SpinVector>;
    using ODEState = vector<double>;
    
    // Lattice properties
    static constexpr size_t spin_dim = 3;
    static constexpr size_t N_atoms = 2;  // Honeycomb unit cell
    size_t dim1, dim2, dim3;
    size_t lattice_size;
    float spin_length = 1.0;
    
    // Spin configuration
    SpinConfig spins;
    vector<Eigen::Vector3d> site_positions;
    
    // Strain state
    StrainState strain;
    StrainState strain_equilibrium;  // Equilibrium strain (set by relax_strain)
    
    // NN interactions
    vector<vector<SpinMatrix>> nn_interaction;
    vector<vector<size_t>> nn_partners;
    vector<vector<int>> nn_bond_types;
    
    // 2nd NN interactions
    vector<vector<SpinMatrix>> j2_interaction;
    vector<vector<size_t>> j2_partners;
    
    // 3rd NN interactions
    vector<vector<SpinMatrix>> j3_interaction;
    vector<vector<size_t>> j3_partners;
    
    // Hexagons for ring exchange
    vector<std::array<size_t, 6>> hexagons;
    vector<vector<std::pair<size_t, size_t>>> site_hexagons;
    
    // External field
    vector<SpinVector> field;
    
    // Parameters
    ElasticParams elastic_params;
    MagnetoelasticParams magnetoelastic_params;
    StrainDriveParams drive_params;
    
    // LLG damping
    double alpha_gilbert = 0.0;
    
    // ODE state size
    size_t state_size;
    
    // Sublattice local frames
    std::array<SpinMatrix, N_atoms> sublattice_frames;
    
    // Custom ordering vector
    SpinConfig ordering_pattern;
    bool has_ordering_pattern = false;
    
    /**
     * Constructor
     */
    StrainPhononLattice(size_t d1, size_t d2, size_t d3 = 1, float spin_l = 1.0);
    
    // ============================================================
    // LATTICE CONSTRUCTION
    // ============================================================
    
    void build_honeycomb();
    
    size_t flatten_index(size_t i, size_t j, size_t k, size_t atom) const {
        // MUST match phonon_lattice.h: ((i * dim2 + j) * dim3 + k) * N_atoms + atom
        return ((i * dim2 + j) * dim3 + k) * N_atoms + atom;
    }
    
    int periodic_boundary(int coord, size_t dim_size) const {
        return ((coord % (int)dim_size) + (int)dim_size) % (int)dim_size;
    }
    
    size_t flatten_periodic(int i, int j, int k, size_t atom) const {
        return flatten_index(periodic_boundary(i, dim1),
                           periodic_boundary(j, dim2),
                           periodic_boundary(k, dim3), atom);
    }
    
    // ============================================================
    // PARAMETER SETTING
    // ============================================================
    
    void set_parameters(const MagnetoelasticParams& me_params,
                       const ElasticParams& el_params,
                       const StrainDriveParams& dr_params);
    
    void set_nn_coupling(size_t site1, size_t site2, const SpinMatrix& J, int bond_type);
    void set_j2_coupling(size_t site, size_t partner, const SpinMatrix& J);
    void set_j3_coupling(size_t site, size_t partner, const SpinMatrix& J);
    void set_field(size_t site, const SpinVector& B);
    void set_uniform_field(const SpinVector& B);
    
    // ============================================================
    // INITIALIZATION
    // ============================================================
    
    void init_random();
    void init_neel();
    void init_zigzag();
    void set_spin(size_t site, const SpinVector& s);
    
    /**
     * Generate random spin on unit sphere (efficient cylindrical projection for 3D)
     */
    SpinVector gen_random_spin(float spin_l);
    
    // ============================================================
    // TIME-DEPENDENT PARAMETERS
    // ============================================================
    
    /**
     * Get effective J7 at time t with strain modulation
     * J7(t) = J7 * (1 - γ*|δε_Eg|/4)^4 * (1 + γ*|δε_Eg|/2)^2
     * where |δε_Eg| is the Eg strain deviation from equilibrium
     */
    double get_effective_J7(double t) const;
    
    /**
     * Get ring exchange energy without the J7 prefactor (for computing ∂H/∂ε)
     */
    double get_ring_exchange_normalized() const;
    
    /**
     * Get derivatives of J7 with respect to strain components
     * ∂J7/∂ε_xx, ∂J7/∂ε_yy, ∂J7/∂ε_xy for each bond type
     */
    void get_dJ7_deps(double* dJ7_deps_xx, double* dJ7_deps_yy, 
                      double* dJ7_deps_xy) const;
    
    // Current simulation time (for time-dependent parameters)
    mutable double current_time = 0.0;
    
    // ============================================================
    // ENERGY CALCULATIONS
    // ============================================================
    
    double spin_energy() const;
    double spin_energy(double t) const;  // Time-dependent version
    double ring_exchange_energy() const;
    double ring_exchange_energy(double t) const;  // Time-dependent version
    double strain_energy() const;
    double magnetoelastic_energy() const;
    
    double total_energy() const {
        return spin_energy() + strain_energy() + magnetoelastic_energy();
    }
    
    double total_energy(double t) const {
        return spin_energy(t) + strain_energy() + magnetoelastic_energy();
    }
    
    double energy_density() const {
        return total_energy() / lattice_size;
    }
    
    /**
     * Compute energy contribution of a single site
     * Includes: Zeeman, NN/2nd NN/3rd NN exchange, magnetoelastic, ring exchange
     * 
     * @param spin_here The spin at this site
     * @param site_index The site index
     * @return Energy contribution from this site's interactions
     */
    double site_energy(const SpinVector& spin_here, size_t site_index) const;
    
    /**
     * Compute energy difference for local spin update (optimized for Metropolis)
     * dE = E(new_spin) - E(old_spin)
     * 
     * @param new_spin Proposed new spin
     * @param old_spin Current spin at site
     * @param site_index The site index
     * @return Energy change if the spin were updated
     */
    double site_energy_diff(const SpinVector& new_spin, const SpinVector& old_spin,
                           size_t site_index) const;
    
    /**
     * Compute ring exchange energy contribution for a single site
     * Returns the portion of H_7 that involves this site's spin
     * 
     * @param spin_here The spin at this site
     * @param site_index The site index
     * @return Ring exchange energy contribution from hexagons containing this site
     */
    double site_ring_exchange_energy(const SpinVector& spin_here, size_t site_index) const;
    
    // ============================================================
    // SPIN BASIS FUNCTIONS FOR D3d IRREPS
    // ============================================================
    
    /**
     * A1g spin basis function: f_K^{A1g}
     * Sum over nearest-neighbor bonds of Kitaev-type term
     * f_K^{A1g} = Σ_<ij>_γ S_i^γ S_j^γ (where γ is the bond direction)
     */
    double f_K_A1g() const;
    
    /**
     * A1g spin basis function: f_J^{A1g}
     * Sum over nearest-neighbor bonds of Heisenberg-type term
     * f_J^{A1g} = Σ_<ij> S_i · S_j
     */
    double f_J_A1g() const;
    
    /**
     * A1g spin basis function: f_Γ^{A1g}
     * Sum over nearest-neighbor bonds of Γ-type term
     * f_Γ^{A1g} = Σ_<ij>_γ (S_i^α S_j^β + S_i^β S_j^α) where α,β ≠ γ
     */
    double f_Gamma_A1g() const;
    
    /**
     * Eg spin basis functions: f_K^{Eg,1}, f_K^{Eg,2}
     * The Eg representation is 2-dimensional
     */
    double f_K_Eg1() const;
    double f_K_Eg2() const;
    
    double f_J_Eg1() const;
    double f_J_Eg2() const;
    
    double f_Gamma_Eg1() const;
    double f_Gamma_Eg2() const;
    
    // ============================================================
    // DERIVATIVES
    // ============================================================
    
    /**
     * Derivative of magnetoelastic energy with respect to strain components
     * ∂H_c/∂ε_xx, ∂H_c/∂ε_yy, ∂H_c/∂ε_xy
     */
    double dH_deps_xx(size_t bond_type) const;
    double dH_deps_yy(size_t bond_type) const;
    double dH_deps_xy(size_t bond_type) const;
    
    /**
     * Derivative of spin basis functions w.r.t. spins (for effective field)
     */
    SpinVector df_K_A1g_dS(size_t site) const;
    SpinVector df_J_A1g_dS(size_t site) const;
    SpinVector df_Gamma_A1g_dS(size_t site) const;
    SpinVector df_K_Eg1_dS(size_t site) const;
    SpinVector df_K_Eg2_dS(size_t site) const;
    SpinVector df_J_Eg1_dS(size_t site) const;
    SpinVector df_J_Eg2_dS(size_t site) const;
    SpinVector df_Gamma_Eg1_dS(size_t site) const;
    SpinVector df_Gamma_Eg2_dS(size_t site) const;
    
    /**
     * Effective field on spin from magnetoelastic coupling
     */
    SpinVector get_magnetoelastic_field(size_t site) const;
    
    /**
     * Total effective field on spin
     */
    SpinVector get_local_field(size_t site) const;
    SpinVector get_local_field(size_t site, double t) const;  // Time-dependent version
    
    SpinVector get_ring_exchange_field(size_t site) const;
    SpinVector get_ring_exchange_field(size_t site, double t) const;  // Time-dependent version
    
    // ============================================================
    // EQUATIONS OF MOTION
    // ============================================================
    
    /**
     * Spin derivative (LLG equation)
     * 
     * dS/dt = S × H_eff - (α/|S|) S × (S × H_eff)
     * 
     * where H_eff = -∂H/∂S is the effective field.
     * The first term is the precessional motion.
     * The second term is damping that relaxes S towards H_eff.
     */
    Eigen::Vector3d spin_derivative(const Eigen::Vector3d& S,
                                   const Eigen::Vector3d& H_eff) const {
        Eigen::Vector3d dSdt = S.cross(H_eff);
        if (alpha_gilbert > 0) {
            // Damping term: -α S × (S × H) drives S towards H (energy dissipation)
            dSdt -= alpha_gilbert * S.cross(S.cross(H_eff)) / spin_length;
        }
        return dSdt;
    }
    
    /**
     * Strain EOM derivatives
     * 
     * For the elastic Hamiltonian:
     * H_elastic = (1/2)[C11(ε_xx² + ε_yy²) + 2C12 ε_xx ε_yy + C44(2ε_xy)²]
     * 
     * The equations of motion are:
     * M d²ε_xx/dt² = -∂H/∂ε_xx - γ dε_xx/dt
     *              = -(C11 ε_xx + C12 ε_yy) - ∂H_c/∂ε_xx - γ V_xx
     * 
     * M d²ε_yy/dt² = -∂H/∂ε_yy - γ dε_yy/dt
     *              = -(C11 ε_yy + C12 ε_xx) - ∂H_c/∂ε_yy - γ V_yy
     * 
     * M d²ε_xy/dt² = -∂H/∂ε_xy - γ dε_xy/dt
     *              = -4C44 ε_xy - ∂H_c/∂ε_xy - γ V_xy
     */
    void strain_derivatives(const StrainState& eps, double t,
                           const double* dH_deps_xx_arr,
                           const double* dH_deps_yy_arr,
                           const double* dH_deps_xy_arr,
                           StrainState& deps_dt) const;
    
    /**
     * Full ODE system for coupled spin-strain dynamics
     */
    void ode_system(const ODEState& x, ODEState& dxdt, double t);
    
    // ============================================================
    // INTEGRATION AND SIMULATION
    // ============================================================
    
    void integrate_rk4(double dt, double t_start, double t_final, 
                      size_t output_every = 100,
                      const string& output_dir = "output");
    
    void integrate_adaptive(double dt_init, double t_start, double t_final,
                           double abs_tol = 1e-8, double rel_tol = 1e-8,
                           size_t output_every = 100,
                           const string& output_dir = "output");
    
    /**
     * Relax strain to equilibrium given current spin configuration.
     * 
     * This finds the strain state that minimizes the total energy 
     * (elastic + magnetoelastic) for the current spin configuration.
     * In the adiabatic limit, phonons quickly equilibrate to the instantaneous
     * spin configuration.
     * 
     * For the elastic Hamiltonian:
     * H = (1/2)[C11(ε_xx² + ε_yy²) + 2C12 ε_xx ε_yy + 4C44 ε_xy²] + H_c(ε, S)
     * 
     * Setting ∂H/∂ε = 0 gives the equilibrium strain.
     * 
     * @param verbose If true, print equilibrium strain values
     */
    void relax_strain(bool verbose = true);
    
    // ============================================================
    // MONTE CARLO
    // ============================================================
    
    /**
     * Perform one Monte Carlo sweep using Metropolis algorithm
     * @param temperature  Temperature for Boltzmann weights
     * @param gaussian_move If true, use Gaussian perturbation; if false, propose random spin
     * @param sigma        Width of Gaussian perturbation (only used if gaussian_move=true)
     * @return Acceptance rate (0.0 to 1.0)
     */
    double mc_sweep(double temperature, bool gaussian_move = false, double sigma = 60.0);
    
    /**
     * Gaussian move around current spin
     */
    SpinVector gaussian_spin_move(const SpinVector& current_spin, double sigma);
    
    /**
     * Simulated annealing with progress reporting
     * @param T_start           Starting temperature
     * @param T_end             Final temperature
     * @param n_sweeps          Sweeps per temperature
     * @param cooling_rate      Multiplicative cooling factor (default 0.9)
     * @param overrelaxation_rate If > 0, do overrelaxation every N sweeps (0 = disabled)
     * @param gaussian_move     Use Gaussian moves (default false)
     * @param out_dir           Output directory for configs (empty = no save)
     * @param T_zero            If true, perform deterministic sweeps at end
     * @param n_deterministics  Number of deterministic sweeps at T=0
     */
    void anneal(double T_start, double T_end, size_t n_sweeps,
                double cooling_rate = 0.9,
                size_t overrelaxation_rate = 0,
                bool gaussian_move = false,
                const string& out_dir = "",
                bool T_zero = false,
                size_t n_deterministics = 1000);
    
    /**
     * Deterministic sweep: align each spin parallel to its local field.
     * This minimizes energy and ensures S × H_eff = 0, eliminating precession.
     * Should be called after annealing to get a true energy minimum.
     * 
     * @param num_sweeps Number of full sweeps over all sites
     */
    void deterministic_sweep(size_t num_sweeps);
    
    /**
     * Over-relaxation sweep (microcanonical, zero acceptance rate)
     * Reflects spins about local field: S' = 2(S·H)H/|H|² - S
     * This preserves energy but accelerates decorrelation.
     */
    void overrelaxation();
    
    // ============================================================
    // PARALLEL TEMPERING
    // ============================================================
    
    /**
     * Parallel tempering with MPI
     * Collects: energy, specific heat, sublattice magnetizations, and cross-correlations
     * All with binning analysis for error estimation
     * 
     * @param temp              Temperature ladder (one per MPI rank)
     * @param n_anneal          Number of equilibration sweeps
     * @param n_measure         Number of measurement sweeps
     * @param overrelaxation_rate Apply overrelaxation every N sweeps (0 = disabled)
     * @param swap_rate         Attempt replica exchange every N sweeps
     * @param probe_rate        Record observables every N sweeps
     * @param dir_name          Output directory
     * @param rank_to_write     List of ranks that should write output (-1 = all)
     * @param gaussian_move     Use Gaussian moves (true) or uniform (false)
     * @param comm              MPI communicator (default: MPI_COMM_WORLD)
     * @param verbose           If true, save spin configurations
     */
    void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move = true, MPI_Comm comm = MPI_COMM_WORLD,
                           bool verbose = false);
    
    /**
     * Attempt replica exchange between neighboring temperatures
     * Uses checkerboard pattern for non-blocking exchanges
     * 
     * @param rank          Current MPI rank
     * @param size          Total number of MPI ranks
     * @param temp          Temperature ladder
     * @param curr_Temp     Current temperature for this rank
     * @param swap_parity   Parity for checkerboard pattern (even/odd)
     * @param comm          MPI communicator
     * @return 1 if exchange was accepted, 0 otherwise
     */
    int attempt_replica_exchange(int rank, int size, const vector<double>& temp,
                                double curr_Temp, size_t swap_parity, 
                                MPI_Comm comm = MPI_COMM_WORLD);
    
    /**
     * Gather and save comprehensive statistics with binning analysis
     */
    void gather_and_save_statistics_comprehensive(int rank, int size, double curr_Temp,
                                   const vector<double>& energies,
                                   const vector<SpinVector>& magnetizations,
                                   const vector<vector<SpinVector>>& sublattice_mags,
                                   vector<double>& heat_capacity, vector<double>& dHeat,
                                   const vector<double>& temp, const string& dir_name,
                                   const vector<int>& rank_to_write,
                                   size_t n_anneal, size_t n_measure,
                                   double curr_accept, int swap_accept,
                                   size_t swap_rate, size_t overrelaxation_rate,
                                   size_t probe_rate, MPI_Comm comm = MPI_COMM_WORLD,
                                   bool verbose = false);
    
    /**
     * Generate optimized temperature grid for parallel tempering
     * Based on Bittner et al., Phys. Rev. Lett. 101, 130603 (2008)
     * 
     * MPI-distributed version: each rank handles one replica
     * 
     * @param Tmin              Minimum (coldest) temperature
     * @param Tmax              Maximum (hottest) temperature  
     * @param warmup_sweeps     MC sweeps for initial equilibration
     * @param sweeps_per_iter   MC sweeps per feedback iteration
     * @param feedback_iters    Number of feedback optimization iterations
     * @param gaussian_move     Use Gaussian moves (true) or uniform (false)
     * @param overrelaxation_rate Apply overrelaxation every N sweeps (0 = disabled)
     * @param target_acceptance Target acceptance rate (default: 0.5 per Bittner)
     * @param convergence_tol   Convergence tolerance for acceptance rate uniformity
     * @param comm              MPI communicator (default: MPI_COMM_WORLD)
     * @return OptimizedTempGridResult containing temperatures and diagnostics
     */
    SPL_OptimizedTempGridResult generate_optimized_temperature_grid_mpi(
        double Tmin, double Tmax,
        size_t warmup_sweeps = 500,
        size_t sweeps_per_iter = 500,
        size_t feedback_iters = 20,
        bool gaussian_move = false,
        size_t overrelaxation_rate = 0,
        double target_acceptance = 0.5,
        double convergence_tol = 0.05,
        MPI_Comm comm = MPI_COMM_WORLD);
    
    /**
     * Generate geometric temperature ladder (simple, no optimization)
     * Uses logarithmic spacing: T_i = T_min * (T_max/T_min)^(i/(R-1))
     * 
     * @param Tmin  Minimum temperature
     * @param Tmax  Maximum temperature
     * @param R     Number of temperatures
     * @return Vector of temperatures in ascending order
     */
    static vector<double> generate_geometric_temperature_ladder(
        double Tmin, double Tmax, size_t R);
    
    // ============================================================
    // HELPER METHODS FOR PARALLEL TEMPERING
    // ============================================================
    
    /**
     * Compute magnetization for each sublattice separately
     * @return Vector of SpinVectors, one per sublattice (N_atoms sublattices)
     */
    vector<SpinVector> magnetization_sublattice() const;
    
    /**
     * Compute global magnetization: M = Σ S_i / N (transformed to global frame)
     */
    SpinVector magnetization_global() const;
    
    /**
     * Binning analysis for error estimation of a scalar observable
     * @param data Vector of observable measurements
     * @return BinningResult containing mean, error, and binning information
     */
    static SPL_BinningResult binning_analysis(const vector<double>& data);
    
    /**
     * Compute comprehensive thermodynamic observables with binning error analysis
     */
    SPL_ThermodynamicObservables compute_thermodynamic_observables(
        const vector<double>& energies,
        const vector<vector<SpinVector>>& sublattice_mags,
        double temperature) const;
    
#ifdef HDF5_ENABLED
    /**
     * Save thermodynamic observables to HDF5 format
     */
    void save_thermodynamic_observables_hdf5(const string& out_dir,
                                              const SPL_ThermodynamicObservables& obs,
                                              const vector<double>& energies,
                                              const vector<SpinVector>& magnetizations,
                                              const vector<vector<SpinVector>>& sublattice_mags,
                                              size_t n_anneal,
                                              size_t n_measure,
                                              size_t probe_rate,
                                              size_t swap_rate,
                                              size_t overrelaxation_rate,
                                              double acceptance_rate,
                                              double swap_acceptance_rate) const;
    
    /**
     * Save aggregated heat capacity data from all temperatures to HDF5 format
     */
    void save_heat_capacity_hdf5(const string& out_dir,
                                  const vector<double>& temperatures,
                                  const vector<double>& heat_capacity,
                                  const vector<double>& dHeat) const;
#endif
    
    // ============================================================
    // I/O
    // ============================================================
    
    void save_spin_config(const string& filename) const;
    void save_spin_config_global(const string& filename) const;
    void load_spin_config(const string& filename);
    void save_strain_state(const string& filename) const;
    void save_positions(const string& filename) const;
    void save_trajectory_txt(const string& output_dir,
                             const vector<double>& times,
                             const vector<Eigen::Vector3d>& M_traj,
                             const vector<Eigen::Vector3d>& M_stag_traj,
                             const vector<double>& energy_traj,
                             const vector<double>& eps_A1g_traj,
                             const vector<double>& eps_Eg1_traj,
                             const vector<double>& eps_Eg2_traj,
                             const vector<double>& J7_eff_traj) const;
    
    // ============================================================
    // OBSERVABLES
    // ============================================================
    
    Eigen::Vector3d total_magnetization() const;
    Eigen::Vector3d staggered_magnetization() const;
    double order_parameter() const;
    
    // Strain mode amplitudes (averaged over bond types)
    double A1g_amplitude() const;
    double Eg1_amplitude() const;
    double Eg2_amplitude() const;
    
    // ============================================================
    // GNEB AND TRANSITION PATH ANALYSIS
    // ============================================================
    
    /**
     * Wrapper to compute energy for GNEB
     * Takes a vector of Eigen::Vector3d (spin config)
     */
    double energy_for_gneb(const vector<Eigen::Vector3d>& config) const;
    
    /**
     * Wrapper to compute gradient (∂E/∂S) for GNEB
     * Returns the gradient at each site (NOT the effective field, but the actual gradient)
     */
    vector<Eigen::Vector3d> gradient_for_gneb(const vector<Eigen::Vector3d>& config) const;
    
    /**
     * Compute the Eg phonon-driven force on spins: -∂H_sp-ph/∂S_i
     * 
     * H_sp-ph = -g Q_E1(t) · f_E1[{S_i}]
     * 
     * Force = +g Q_E1 · ∂f_E1/∂S_i
     * 
     * This is the "symmetry-directed push" in the E1 channel.
     * Even if f_E1 = 0 at the triple-Q state, ∂f_E1/∂S ≠ 0 generically.
     * 
     * @param config Spin configuration
     * @param Q_Eg   Eg phonon amplitude (can be Q_E1 for IR modes)
     * @return Force vector at each site
     */
    vector<Eigen::Vector3d> compute_Eg_phonon_force(
        const vector<Eigen::Vector3d>& config, double Q_Eg) const;
    
    /**
     * Compute the derivative of Eg spin basis functions w.r.t. spins
     * ∂f_Eg/∂S_i = (∂f_K_Eg1/∂S_i, ∂f_K_Eg2/∂S_i, ∂f_J_Eg1/∂S_i, ...)
     * 
     * This gives the direction in spin space that the Eg phonon "pushes"
     * 
     * @param config Spin configuration
     * @return Vector of (df_Eg1/dS, df_Eg2/dS) at each site, combining K, J, Γ contributions
     */
    std::pair<vector<Eigen::Vector3d>, vector<Eigen::Vector3d>> 
    compute_Eg_derivatives(const vector<Eigen::Vector3d>& config) const;
    
    /**
     * Get the current spin configuration as a vector for GNEB
     */
    vector<Eigen::Vector3d> get_spin_config() const;
    
    /**
     * Set spins from a GNEB-style configuration vector
     */
    void set_spin_config(const vector<Eigen::Vector3d>& config);
    
    /**
     * Initialize spins to a zigzag pattern
     * @param direction  Zigzag direction: 0=x-bond, 1=y-bond, 2=z-bond
     */
    void init_zigzag_pattern(int direction = 2);
    
    /**
     * Initialize spins to triple-Q pattern
     */
    void init_triple_q();
    
    /**
     * Structure factor at a specific wavevector
     * S(q) = |Σ_i S_i exp(-i q·r_i)|² / N
     */
    double structure_factor(const Eigen::Vector3d& q) const;
    
    /**
     * Compute the full static spin structure factor S(q) on a grid
     * Returns S^{αβ}(q) = (1/N) Σ_{ij} S_i^α S_j^β exp(-i q·(r_i - r_j))
     * 
     * @param n_q1  Number of q-points along b1* direction
     * @param n_q2  Number of q-points along b2* direction  
     * @param q1_range  Range of q1 in units of 2π (default: [-2, 2])
     * @param q2_range  Range of q2 in units of 2π (default: [-2, 2])
     * @return struct with q-grid and S(q) tensor components
     */
    struct StructureFactorResult {
        vector<double> q1_vals;  // q-points along b1*
        vector<double> q2_vals;  // q-points along b2*
        vector<vector<double>> S_total;   // Total S(q) = Σ_α S^{αα}(q)
        vector<vector<double>> S_xx;      // S^{xx}(q)
        vector<vector<double>> S_yy;      // S^{yy}(q)
        vector<vector<double>> S_zz;      // S^{zz}(q)
    };
    
    StructureFactorResult compute_static_structure_factor(
        size_t n_q1 = 100, size_t n_q2 = 100,
        double q1_min = -2.0, double q1_max = 2.0,
        double q2_min = -2.0, double q2_max = 2.0) const;
    
    /**
     * Save structure factor to file
     */
    void save_structure_factor(const string& filename, 
                               const StructureFactorResult& sf) const;
    
    /**
     * Compute collective variables for MEP analysis:
     * - m_3Q: triple-Q order parameter (structure factor at M points)
     * - m_zz: zigzag order parameter
     * - f_Eg: Eg symmetry breaking bilinear |f_Eg| = sqrt(f_Eg1² + f_Eg2²)
     */
    struct CollectiveVars {
        double m_3Q;
        double m_zigzag;
        double f_Eg_amplitude;
        double E_total;
    };
    
    CollectiveVars compute_collective_variables() const;
    
private:
    // Random number generation
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
};

#endif // STRAIN_PHONON_LATTICE_H
