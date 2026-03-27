/**
 * @file strain_phonon_lattice.h
 * @brief Spin-strain coupled honeycomb lattice with magnetoelastic coupling
 * 
 * This implements a honeycomb lattice with:
 * - Generic NN, 2nd NN, and 3rd NN spin-spin interactions (Kitaev-Heisenberg-Γ-Γ')
 * - Strain field phonons: Eg (ε_xx - ε_yy, 2ε_xy)
 * - Magnetoelastic coupling through the Eg symmetry channel
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
 * Magnetoelastic coupling (D3d point group symmetry, Eg channel only):
 * H_c = λ_{Eg} Σ_r {(ε_xx - ε_yy)[(J+K)f_K^{Eg,1} + J f_J^{Eg,1} + Γ f_Γ^{Eg,1}]
 *                  + 2ε_xy[(J+K)f_K^{Eg,2} + J f_J^{Eg,2} + Γ f_Γ^{Eg,2}]}
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
#include "unitcell_builders.h"
#include "simple_linear_alg.h"
#include "classical_spin/core/spin_config.h"  // For should_rank_write
#include "classical_spin/mc/mc_common.h"      // Common MC structs & templates
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
 * - Eg: (ε_xx - ε_yy, 2ε_xy) (shear modes)
 * 
 * Unit-cell strain (Eg irrep of D3d point group).
 */
struct StrainState {
    // Strain tensor components (unit-cell level)
    double epsilon_xx = 0.0;
    double epsilon_yy = 0.0;
    double epsilon_xy = 0.0;
    
    // Strain velocities (time derivatives)
    double V_xx = 0.0;
    double V_yy = 0.0;
    double V_xy = 0.0;
    
    // Total DOF: 3 strain components × 2 (coords + velocities) = 6
    static constexpr size_t N_DOF = 6;
    
    // Pack to flat array
    void to_array(double* arr) const {
        arr[0] = epsilon_xx; arr[1] = epsilon_yy; arr[2] = epsilon_xy;
        arr[3] = V_xx;       arr[4] = V_yy;       arr[5] = V_xy;
    }
    
    // Unpack from flat array
    void from_array(const double* arr) {
        epsilon_xx = arr[0]; epsilon_yy = arr[1]; epsilon_xy = arr[2];
        V_xx       = arr[3]; V_yy       = arr[4]; V_xy       = arr[5];
    }
    
    // Kinetic energy (with unit mass)
    double kinetic_energy() const {
        return 0.5 * (V_xx * V_xx + V_yy * V_yy + V_xy * V_xy);
    }
    
    // Eg amplitude sqrt((ε_xx - ε_yy)² + 4ε_xy²)
    double Eg_amplitude() const {
        double d = epsilon_xx - epsilon_yy;
        return std::sqrt(d * d + 4.0 * epsilon_xy * epsilon_xy);
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
    
    // Damping coefficient (Eg mode)
    double gamma_Eg = 0.1;
    
    // Characteristic Eg frequency (derived from elastic constants)
    // ω_Eg² ≈ (C11 - C12) / M ≈ 2C44 / M
    double omega_Eg() const { return std::sqrt(2.0 * C44 / M); }
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
    
    // Magnetoelastic coupling constant (Eg channel)
    double lambda_Eg = 0.0;
    
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

// Common MC structs (from mc_common.h)
using mc::BinningResult;
using mc::Observable;
using mc::VectorObservable;
using mc::ThermodynamicObservables;
using mc::OptimizedTempGridResult;
using mc::AutocorrelationResult;

// Legacy type aliases for backward compatibility
using SPL_BinningResult = mc::BinningResult;
using SPL_Observable = mc::Observable;
using SPL_VectorObservable = mc::VectorObservable;
using SPL_ThermodynamicObservables = mc::ThermodynamicObservables;
using SPL_OptimizedTempGridResult = mc::OptimizedTempGridResult;

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
    size_t N_atoms;  // Atoms per unit cell (from UnitCell)
    size_t dim1, dim2, dim3;
    size_t lattice_size;
    float spin_length = 1.0;
    
    // UnitCell
    UnitCell unit_cell;
    
    // Spin configuration
    SpinConfig spins;
    vector<Eigen::Vector3d> site_positions;
    
    // Strain state
    StrainState strain;
    StrainState strain_equilibrium;  // Equilibrium strain (set by relax_strain)
    bool fix_strain_ = false;       // If true, relax_strain() is a no-op
    double drive_F_Eg1_ = 0.0;     // Static drive force along Eg1: H_drive = -Σ_b F1*Q_Eg1(b)
    double drive_F_Eg2_ = 0.0;     // Static drive force along Eg2: H_drive = -Σ_b F2*Q_Eg2(b)
    
    // ============================================================
    // EXTRA DOF INTERFACE (for mc::attempt_replica_exchange)
    // Ensures strain state is exchanged together with spins in PT
    // ============================================================
    
    /** Number of extra (non-spin) degrees of freedom to exchange in PT */
    size_t extra_dof_size() const { return StrainState::N_DOF; }
    
    /** Pack strain state into flat array for MPI exchange */
    void pack_extra_dof(double* buf) const { strain.to_array(buf); }
    
    /** Unpack strain state from flat array after MPI exchange */
    void unpack_extra_dof(const double* buf) { strain.from_array(buf); }
    
    /**
     * Set uniform Eg strain on all bond types.
     * @param Eg1  Eg1 component: ε_xx = Eg1, ε_yy = -Eg1
     * @param Eg2  Eg2 component: ε_xy = Eg2
     */
    void set_strain_Eg(double Eg1, double Eg2) {
        strain.epsilon_xx = Eg1;
        strain.epsilon_yy = -Eg1;
        strain.epsilon_xy = Eg2;
    }
    
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
    
    // Langevin temperature for stochastic dynamics (k_B T)
    // Set > 0 to enable thermal fluctuations in integrate_langevin
    double langevin_temperature = 0.0;
    
    // ODE state size
    size_t state_size;
    
    // Sublattice local frames
    vector<SpinMatrix> sublattice_frames;
    
    // Custom ordering vector
    SpinConfig ordering_pattern;
    bool has_ordering_pattern = false;
    
    /**
     * Constructor: takes a UnitCell (built by e.g. build_strain_honeycomb)
     */
    StrainPhononLattice(const UnitCell& uc, size_t d1, size_t d2, size_t d3 = 1, float spin_l = 1.0);
    
    // ============================================================
    // LATTICE CONSTRUCTION
    // ============================================================
    
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
    // QUENCHED DISORDER
    // ============================================================
    
    /**
     * Save a snapshot of the clean (disorder-free) interaction matrices.
     * Must be called once after set_parameters() and before apply_exchange_disorder().
     */
    void store_clean_interactions();
    
    /**
     * Restore interactions to the clean state saved by store_clean_interactions().
     */
    void restore_clean_interactions();
    
    /**
     * Apply quenched exchange disorder to ALL bond interaction matrices.
     *
     * Each 3×3 bond matrix J_ij is replaced by J_ij * (1 + σ * ξ_ij)
     * where ξ_ij ~ N(0,1) is drawn independently for each bond (but
     * shared between J_ij and J_ji for hermiticity).
     *
     * @param disorder_strength  σ — fractional standard deviation (0.1 = 10%)
     * @param seed               RNG seed for reproducibility (different per trial)
     */
    void apply_exchange_disorder(double disorder_strength, unsigned int seed);
    
    /**
     * Apply site dilution: randomly zero out all bonds connected to a site.
     *
     * @param dilution_fraction  Fraction of sites to remove (0.0–1.0)
     * @param seed               RNG seed for reproducibility
     */
    void apply_site_dilution(double dilution_fraction, unsigned int seed);
    
    /** Whether clean interactions have been stored */
    bool has_clean_interactions_ = false;
    
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
     * ∂J7/∂ε_xx, ∂J7/∂ε_yy, ∂J7/∂ε_xy
     */
    void get_dJ7_deps(double& dJ7_deps_xx, double& dJ7_deps_yy, 
                      double& dJ7_deps_xy) const;
    
    // Current simulation time (for time-dependent parameters)
    mutable double current_time = 0.0;
    
    // ============================================================
    // ENERGY CALCULATIONS
    // ============================================================
    
    /** Bilinear exchange energy (NN + J2 + J3 + Zeeman).  Excludes ring exchange. */
    double bilinear_energy() const;
    
    /** Ring exchange hexagon sum for a given J7 value. */
    double ring_exchange_sum(double J7_val) const;
    
    double spin_energy() const;
    double spin_energy(double t) const;  // Time-dependent version
    double ring_exchange_energy() const;
    double ring_exchange_energy(double t) const;  // Time-dependent version
    double strain_energy() const;
    double magnetoelastic_energy() const;
    
    double drive_energy() const {
        if (std::abs(drive_F_Eg1_) < 1e-15 && std::abs(drive_F_Eg2_) < 1e-15) return 0.0;
        double Eg1_b = (strain.epsilon_xx - strain.epsilon_yy) / 2.0;
        double Eg2_b = strain.epsilon_xy;
        return -(drive_F_Eg1_ * Eg1_b + drive_F_Eg2_ * Eg2_b);
    }
    
    double total_energy() const {
        return spin_energy() + strain_energy() + magnetoelastic_energy() + drive_energy();
    }
    
    double total_energy(double t) const {
        return spin_energy(t) + strain_energy() + magnetoelastic_energy() + drive_energy();
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
    // Eg SPIN BASIS FUNCTIONS (D3d irrep)
    // ============================================================
    
    /**
     * All 8 Eg spin basis function values, computed in a single pass
     * over nearest-neighbor bonds.  Each interaction type (K, J, Γ, Γ')
     * has an Eg1 and Eg2 component following the D3d point-group decomposition.
     */
    struct EgBasisValues {
        double K1, K2;     // Kitaev
        double J1, J2;     // Heisenberg (non-Kitaev diagonal)
        double G1, G2;     // Gamma (off-diagonal symmetric)
        double Gp1, Gp2;   // Gamma' (off-diagonal asymmetric)
    };
    
    /**
     * Spin gradients of all 8 Eg basis functions at a given site,
     * computed in a single pass over the site's nearest neighbors.
     */
    struct EgBasisGradients {
        SpinVector K1, K2;
        SpinVector J1, J2;
        SpinVector G1, G2;
        SpinVector Gp1, Gp2;
    };
    
    /** Compute all Eg basis function values in one bond loop. */
    EgBasisValues compute_Eg_basis() const;
    
    /** Compute all Eg basis function gradients ∂f/∂S_site in one neighbor loop. */
    EgBasisGradients compute_Eg_basis_dS(size_t site) const;
    
    /**
     * Weighted Eg spin factors:
     *   Σ_Eg1 = (J+K)·f_K1 + J·f_J1 + Γ·f_Γ1 + Γ'·f_Γ'1
     *   Σ_Eg2 = (J+K)·f_K2 + J·f_J2 + Γ·f_Γ2 + Γ'·f_Γ'2
     * These are the coupling-weighted bilinears that enter the ME energy/force.
     */
    std::pair<double, double> Eg_spin_factors() const;
    
    /**
     * Weighted Eg spin factor gradients at a given site:
     *   ∂Σ_Eg1/∂S = (J+K)·∂f_K1/∂S + J·∂f_J1/∂S + Γ·∂f_Γ1/∂S + Γ'·∂f_Γ'1/∂S
     *   ∂Σ_Eg2/∂S = ...
     * These are the coupling-weighted spin gradients that enter the ME field.
     */
    std::pair<SpinVector, SpinVector> Eg_spin_factor_gradients(size_t site) const;
    
    /**
     * Effective field on spin from magnetoelastic coupling
     */
    SpinVector get_magnetoelastic_field(size_t site) const;
    
    /** Bilinear effective field (NN + J2 + J3 + Zeeman). Excludes ring exchange & ME. */
    SpinVector bilinear_field(size_t site) const;
    
    /**
     * Total effective field on spin
     */
    SpinVector get_local_field(size_t site) const;
    SpinVector get_local_field(size_t site, double t) const;  // Time-dependent version
    
    SpinVector get_ring_exchange_field(size_t site) const;
    SpinVector get_ring_exchange_field(size_t site, double t) const;  // Time-dependent version
    
    /**
     * Finite-difference verification of the ring exchange effective field.
     * Computes -∂H_7/∂S_site using central differences on ring_exchange_energy().
     * Use to validate the analytical get_ring_exchange_field() against energy.
     */
    SpinVector ring_exchange_field_numerical(size_t site, double delta = 1e-5) const;

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
     * Strain EOM derivatives (Eg damping only)
     * 
     * M d²ε_xx/dt² = -(C11 ε_xx + C12 ε_yy) - ∂H_c/∂ε_xx - γ_Eg V_xx + F_Eg1
     * M d²ε_yy/dt² = -(C11 ε_yy + C12 ε_xx) - ∂H_c/∂ε_yy - γ_Eg V_yy - F_Eg1
     * M d²ε_xy/dt² = -4C44 ε_xy - ∂H_c/∂ε_xy - γ_Eg V_xy + F_Eg2
     */
    void strain_derivatives(const StrainState& eps, double t,
                           double dH_deps_xx,
                           double dH_deps_yy,
                           double dH_deps_xy,
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
     * Stochastic Langevin dynamics using the Heun (improved Euler) scheme.
     *
     * Spins obey the stochastic LLG (sLLG) equation:
     *   dS/dt = S × (H_eff + ξ) − (α/|S|) S × [S × (H_eff + ξ)]
     * with white-noise field ξ satisfying the fluctuation-dissipation relation:
     *   <ξ_i^a(t) ξ_j^b(t')> = (2 α k_B T / |S|) δ_{ij} δ_{ab} δ(t−t')
     * 
     * Strain DOF obey the Langevin equation:
     *   M d²ε/dt² = −∂H/∂ε − γ dε/dt + η(t)
     * with thermal noise η satisfying:
     *   <η(t) η(t')> = 2 γ k_B T δ(t−t')
     *
     * The stochastic Heun method (second-order predictor-corrector) is used
     * to correctly handle the multiplicative noise in the spin equation.
     *
     * At each step:
     *   1. Draw noise vectors ξ, η once (held constant over the step)
     *   2. Predictor: compute deterministic + noise RHS at current state
     *   3. Euler predict: x̃ = x + dt * f(x, ξ)
     *   4. Corrector: recompute RHS at predicted state with SAME noise
     *   5. Heun update: x_{n+1} = x + (dt/2) [f(x, ξ) + f(x̃, ξ)]
     *   6. Re-normalize spins to preserve |S| = spin_length
     *
     * Requires alpha_gilbert > 0 (damping is essential for proper thermalization).
     * langevin_temperature sets k_B T.
     *
     * @param dt            Integration timestep
     * @param t_start       Start time
     * @param t_final       End time
     * @param output_every  Save observables every N steps
     * @param output_dir    Directory for output files
     */
    void integrate_langevin(double dt, double t_start, double t_final,
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
    double metropolis(double temperature, bool gaussian_move = false, double sigma = 60.0);
    
    /** @brief Legacy alias for metropolis() */
    inline double mc_sweep(double temperature, bool gaussian_move = false, double sigma = 60.0) {
        return metropolis(temperature, gaussian_move, sigma);
    }
    
    /**
     * Gaussian move around current spin
     */
    SpinVector gaussian_spin_move(const SpinVector& current_spin, double sigma);
    
    /**
     * Greedy quench: deterministic sweep until energy converges
     * @param rel_tol    Relative energy tolerance for convergence
     * @param max_sweeps Maximum number of sweeps
     */
    inline void greedy_quench(double rel_tol = 1e-12, size_t max_sweeps = 10000) {
        mc::greedy_quench(*this, rel_tol, max_sweeps);
    }
    
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
    void simulated_annealing(double T_start, double T_end, size_t n_sweeps,
                double cooling_rate = 0.9,
                size_t overrelaxation_rate = 0,
                bool gaussian_move = false,
                const string& out_dir = "",
                bool T_zero = false,
                size_t n_deterministics = 1000);
    
    /** @brief Legacy alias for simulated_annealing() */
    inline void anneal(double T_start, double T_end, size_t n_sweeps,
                double cooling_rate = 0.9,
                size_t overrelaxation_rate = 0,
                bool gaussian_move = false,
                const string& out_dir = "",
                bool T_zero = false,
                size_t n_deterministics = 1000) {
        simulated_annealing(T_start, T_end, n_sweeps, cooling_rate, overrelaxation_rate,
                           gaussian_move, out_dir, T_zero, n_deterministics);
    }
    
    /**
     * Deterministic sweep: align each spin parallel to its local field.
     * This minimizes energy and ensures S × H_eff = 0, eliminating precession.
     * Should be called after annealing to get a true energy minimum.
     * 
     * @param num_sweeps Number of full sweeps over all sites
     * @param output_dir Optional directory to save torque convergence diagnostics
     */
    void deterministic_sweep(size_t num_sweeps, const std::string& output_dir = "");
    
    /**
     * Over-relaxation sweep (microcanonical, zero acceptance rate)
     * Reflects spins about local field: S' = 2(S·H)H/|H|² - S
     * This preserves energy but accelerates decorrelation.
     */
    void overrelaxation();
    
    // ============================================================
    // PARALLEL TEMPERING (delegated to mc::* templates)
    // ============================================================
    
    /**
     * Parallel tempering with MPI — delegates to mc::parallel_tempering
     * SPL version also saves strain state via extra_save callback.
     * Strain is relaxed before PT begins and saved after measurements.
     * During PT, strain is exchanged with spins via extra_dof interface.
     */
    inline void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure,
                           size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                           string dir_name, const vector<int>& rank_to_write,
                           bool gaussian_move = true, MPI_Comm comm = MPI_COMM_WORLD,
                           bool verbose = false, const vector<size_t>& sweeps_per_temp = {}) {
        int rank; MPI_Comm_rank(comm, &rank);
        rng.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count() + rank * 12345);
        
        // Relax strain to equilibrium before starting PT
        relax_strain(rank == 0);
        
        auto extra_save = [this](const string& dir, double) {
            // Relax strain to match final spin configuration, then save
            relax_strain(false);
            save_strain_state(dir + "/strain_state.txt");
        };
        mc::parallel_tempering(*this, temp, n_anneal, n_measure, overrelaxation_rate,
                               swap_rate, probe_rate, dir_name, rank_to_write,
                               gaussian_move, comm, verbose, sweeps_per_temp, extra_save);
    }
    
    /**
     * Generate optimized temperature grid — delegates to mc::generate_optimized_temperature_grid_mpi
     */
    inline mc::OptimizedTempGridResult generate_optimized_temperature_grid_mpi(
        double Tmin, double Tmax,
        size_t warmup_sweeps = 500,
        size_t sweeps_per_iter = 500,
        size_t feedback_iters = 20,
        bool gaussian_move = false,
        size_t overrelaxation_rate = 0,
        double target_acceptance = 0.45,
        double convergence_tol = 0.05,
        MPI_Comm comm = MPI_COMM_WORLD,
        bool use_gradient = true) {
        int rank; MPI_Comm_rank(comm, &rank);
        rng.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count() + rank * 12345);
        return mc::generate_optimized_temperature_grid_mpi(*this, Tmin, Tmax,
            warmup_sweeps, sweeps_per_iter, feedback_iters, gaussian_move,
            overrelaxation_rate, target_acceptance, convergence_tol, comm, use_gradient);
    }
    
    /** Generate geometric temperature ladder */
    static inline vector<double> generate_geometric_temperature_ladder(
        double Tmin, double Tmax, size_t R) {
        return mc::generate_geometric_temperature_ladder(Tmin, Tmax, R);
    }
    
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
    
    /** Binning analysis — delegates to mc::binning_analysis */
    static inline mc::BinningResult binning_analysis(const vector<double>& data) {
        return mc::binning_analysis(data);
    }
    
    /** Estimate autocorrelation time — delegates to mc::estimate_autocorrelation_time */
    static inline void estimate_autocorrelation_time(const vector<double>& energies,
                                               size_t base_interval,
                                               double& tau_int_out,
                                               size_t& sampling_interval_out) {
        mc::estimate_autocorrelation_time(energies, base_interval, tau_int_out, sampling_interval_out);
    }
    
    /** Compute thermodynamic observables — delegates to mc::compute_thermodynamic_observables */
    inline mc::ThermodynamicObservables compute_thermodynamic_observables(
        const vector<double>& energies,
        const vector<vector<SpinVector>>& sublattice_mags,
        double temperature) const {
        return mc::compute_thermodynamic_observables<SpinVector>(energies, sublattice_mags, temperature, lattice_size);
    }
    
    // ============================================================
    // I/O
    // ============================================================
    
    void save_spin_config(const string& filename) const;
    void save_spin_config_global(const string& filename) const;
    void load_spin_config(const string& filename);
    void save_strain_state(const string& filename) const;
    void save_positions(const string& filename) const;
    
    /**
     * Save combined (spin, strain) configuration for GNEB
     * Format includes both spin vectors and Eg strain components
     */
    void save_spin_strain_config(const string& filename) const;
    
    /**
     * Load combined (spin, strain) configuration for GNEB
     * Reads both spin vectors and Eg strain components
     */
    void load_spin_strain_config(const string& filename);
    
    void save_trajectory_txt(const string& output_dir,
                             const vector<double>& times,
                             const vector<Eigen::Vector3d>& M_traj,
                             const vector<Eigen::Vector3d>& M_stag_traj,
                             const vector<double>& energy_traj,
                             const vector<double>& eps_Eg1_traj,
                             const vector<double>& eps_Eg2_traj,
                             const vector<double>& J7_eff_traj) const;
    
    // ============================================================
    // OBSERVABLES
    // ============================================================
    
    Eigen::Vector3d magnetization_local() const;
    Eigen::Vector3d magnetization_local_antiferro() const;
    
    // Legacy aliases
    Eigen::Vector3d total_magnetization() const { return magnetization_local(); }
    Eigen::Vector3d staggered_magnetization() const { return magnetization_local_antiferro(); }
    double order_parameter() const;
    
    // Strain mode amplitudes (averaged over bond types)
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
    
    // ============================================================
    // GNEB WITH STRAIN: Combined spin + strain configuration space
    // ============================================================
    
    /**
     * Compute total energy for GNEB with strain degrees of freedom
     * E(spins, ε_Eg1, ε_Eg2) = E_spin(spins; ε) + E_elastic(ε)
     * 
     * The strain is applied uniformly to all bond types.
     * 
     * @param spins      Spin configuration (N unit vectors)
     * @param strain_Eg1 Eg1 strain component (ε_xx - ε_yy)/2
     * @param strain_Eg2 Eg2 strain component (ε_xy)
     * @return Total energy including spin-strain coupling
     */
    double energy_for_gneb_with_strain(const vector<Eigen::Vector3d>& spins,
                                        double strain_Eg1, double strain_Eg2) const;
    
    /**
     * Compute gradients for GNEB with strain
     * Returns (∂E/∂S_i, ∂E/∂ε_Eg1, ∂E/∂ε_Eg2)
     * 
     * The spin gradients are the standard effective field gradients.
     * The strain gradients come from elastic + magnetoelastic contributions.
     * 
     * @param spins       Spin configuration
     * @param strain_Eg1  Eg1 strain
     * @param strain_Eg2  Eg2 strain
     * @return Tuple of (spin_gradients, dE/dε_Eg1, dE/dε_Eg2)
     */
    std::tuple<vector<Eigen::Vector3d>, double, double>
    gradient_for_gneb_with_strain(const vector<Eigen::Vector3d>& spins,
                                   double strain_Eg1, double strain_Eg2) const;
    
    /**
     * Relax strain at fixed spin configuration
     * Finds ε* = argmin_ε E(spins, ε)
     * 
     * This is useful for finding equilibrium strain for a given spin state,
     * e.g., to set up initial/final states for GNEB.
     * 
     * @param spins       Fixed spin configuration
     * @param max_iter    Maximum optimization iterations
     * @param tolerance   Convergence tolerance for strain force
     * @return Pair of (ε_Eg1*, ε_Eg2*) at equilibrium
     */
    std::pair<double, double> relax_strain_at_fixed_spins(
        const vector<Eigen::Vector3d>& spins,
        size_t max_iter = 1000,
        double tolerance = 1e-6) const;
    
    /**
     * Relax strain at fixed spins with warm-start from initial guess.
     * Uses L-BFGS-like conjugate gradient for fast convergence.
     */
    std::pair<double, double> relax_strain_at_fixed_spins(
        const vector<Eigen::Vector3d>& spins,
        double init_Eg1, double init_Eg2,
        size_t max_iter = 200,
        double tolerance = 1e-6) const;
    
    /**
     * Compute energy with an external (fixed) strain offset plus internal Eg strain
     * Total strain = external + internal: ε_total = ε_ext + ε_int
     * Used for computing kinetic barriers at fixed applied strain.
     * 
     * @param spins          Spin configuration
     * @param internal_Eg1   Internal (relaxable) Eg1 strain
     * @param internal_Eg2   Internal (relaxable) Eg2 strain
     * @param external_Eg1   External (fixed) Eg1 strain offset
     * @param external_Eg2   External (fixed) Eg2 strain offset
     * @return Total energy at the combined strain
     */
    double energy_for_gneb_with_external_strain(
        const vector<Eigen::Vector3d>& spins,
        double internal_Eg1, double internal_Eg2,
        double external_Eg1, double external_Eg2) const;
    
    /**
     * Compute gradients with external strain offset
     * Returns (∂E/∂S_i, ∂E/∂ε_int_Eg1, ∂E/∂ε_int_Eg2)
     * The gradients are w.r.t. internal strain (external is fixed).
     */
    std::tuple<vector<Eigen::Vector3d>, double, double>
    gradient_for_gneb_with_external_strain(
        const vector<Eigen::Vector3d>& spins,
        double internal_Eg1, double internal_Eg2,
        double external_Eg1, double external_Eg2) const;
    
    /**
     * Relax internal strain at fixed spins with an external strain offset
     * Finds ε_int* = argmin_{ε_int} E(spins, ε_ext + ε_int)
     * 
     * @param spins         Fixed spin configuration
     * @param external_Eg1  External Eg1 strain (fixed)
     * @param external_Eg2  External Eg2 strain (fixed)
     * @return Pair of (ε_int_Eg1*, ε_int_Eg2*) for internal strain at equilibrium
     *         Total strain = external + internal
     */
    std::pair<double, double> relax_strain_with_external(
        const vector<Eigen::Vector3d>& spins,
        double external_Eg1, double external_Eg2,
        size_t max_iter = 1000,
        double tolerance = 1e-6) const;
    
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
    std::normal_distribution<double> normal_dist{0.0, 1.0};
    
    // Clean interaction backup (for applying fresh disorder per trial)
    vector<vector<SpinMatrix>> clean_nn_interaction_;
    vector<vector<SpinMatrix>> clean_j2_interaction_;
    vector<vector<SpinMatrix>> clean_j3_interaction_;
};

#endif // STRAIN_PHONON_LATTICE_H