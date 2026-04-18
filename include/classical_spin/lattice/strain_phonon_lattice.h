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
#include "unitcell_builders.h"
#include "simple_linear_alg.h"
#include "kitaev_bonds.h"
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
 * Per-unit-cell strain state for local strain mode.
 * Each honeycomb unit cell (2 sites) carries its own strain field.
 */
struct CellStrain {
    double epsilon_xx = 0.0;
    double epsilon_yy = 0.0;
    double epsilon_xy = 0.0;
    double V_xx = 0.0;
    double V_yy = 0.0;
    double V_xy = 0.0;
    
    static constexpr size_t N_DOF = 6;
    
    void to_array(double* arr) const {
        arr[0] = epsilon_xx; arr[1] = epsilon_yy; arr[2] = epsilon_xy;
        arr[3] = V_xx; arr[4] = V_yy; arr[5] = V_xy;
    }
    void from_array(const double* arr) {
        epsilon_xx = arr[0]; epsilon_yy = arr[1]; epsilon_xy = arr[2];
        V_xx = arr[3]; V_yy = arr[4]; V_xy = arr[5];
    }
    
    double Eg1() const { return epsilon_xx - epsilon_yy; }
    double Eg2() const { return 2.0 * epsilon_xy; }
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
    
    // Damping coefficients
    double gamma_A1g = 0.1;   // A1g mode damping
    double gamma_Eg = 0.1;    // Eg mode damping
    
    // Characteristic frequencies (derived from elastic constants)
    // ω_A1g² ≈ (C11 + C12) / M
    // ω_Eg² ≈ (C11 - C12) / M ≈ 2C44 / M
    double omega_A1g() const { return std::sqrt((C11 + C12) / M); }
    double omega_Eg() const { return std::sqrt(2.0 * C44 / M); }
    
    // Optional: quartic anharmonicity (V += ¼κ·|ε|⁴ prevents ME runaway)
    double kappa_A1g = 0.0;  // A1g quartic coefficient
    double kappa_Eg = 0.0;   // Eg quartic coefficient
    
    // Gradient stiffness coupling neighboring cells (local strain mode)
    // E_grad = (K_gradient/2) Σ_{<cc'>} |ε(c) - ε(c')|²
    double K_gradient = 0.0;
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
    
    // Kitaev rotation / local-exchange helpers live in kitaev_bonds.h so
    // PhononLattice and StrainPhononLattice can't drift again. The only
    // class-specific choice here is that `get_J{x,y,z}` returns the LOCAL
    // frame, because the magnetoelastic energy and the f_K/f_J/f_Γ basis
    // functions used downstream are expressed in that frame.

    /// Kitaev local-to-global rotation matrix.
    static SpinMatrix get_kitaev_rotation() {
        return classical_spin::kitaev::kitaev_rotation();
    }

    /// Transform a local-frame exchange matrix to the global Cartesian frame.
    static SpinMatrix to_global_frame(const SpinMatrix& J_local) {
        return classical_spin::kitaev::to_global_frame(J_local);
    }

    // Bond-dependent exchange matrices in LOCAL Kitaev frame.
    SpinMatrix get_Jx_local() const {
        return classical_spin::kitaev::make_Jx_local(J, K, Gamma, Gammap);
    }
    SpinMatrix get_Jy_local() const {
        return classical_spin::kitaev::make_Jy_local(J, K, Gamma, Gammap);
    }
    SpinMatrix get_Jz_local() const {
        return classical_spin::kitaev::make_Jz_local(J, K, Gamma, Gammap);
    }

    // Exchange matrices as used by StrainPhononLattice — kept in LOCAL frame
    // (different from PhononLattice, which returns the GLOBAL frame here).
    // The spin basis functions f_K / f_J / f_Γ used downstream index the
    // bond_type directly as the spin component, which requires the local
    // frame.
    SpinMatrix get_Jx() const { return get_Jx_local(); }
    SpinMatrix get_Jy() const { return get_Jy_local(); }
    SpinMatrix get_Jz() const { return get_Jz_local(); }

    // Isotropic Heisenberg (rotation-invariant).
    SpinMatrix get_J3_matrix()   const { return classical_spin::kitaev::heisenberg_matrix(J3);   }
    SpinMatrix get_J2_A_matrix() const { return classical_spin::kitaev::heisenberg_matrix(J2_A); }
    SpinMatrix get_J2_B_matrix() const { return classical_spin::kitaev::heisenberg_matrix(J2_B); }
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
    // LOCAL STRAIN (per-unit-cell strain DOF)
    // ============================================================
    bool use_local_strain_ = false;  // If true, use per-cell strain instead of global
    size_t N_cells_ = 0;            // Number of unit cells = dim1 * dim2 * dim3
    
    // Per-cell strain arrays
    vector<CellStrain> cell_strains_;          // Current cell strains [N_cells]
    vector<CellStrain> cell_strains_eq_;       // Equilibrium cell strains
    
    // Cell topology
    vector<size_t> site_to_cell_;              // Map site index → cell index
    vector<std::array<size_t, 2>> cell_sites_; // Map cell index → {site_A, site_B}
    
    // Bonds belonging to each cell: list of (site_i, neighbor_idx, bond_type)
    struct CellBond {
        size_t site_i;
        size_t site_j;
        size_t nn_idx;   // index into nn_partners[site_i]
        int bond_type;
    };
    vector<vector<CellBond>> cell_bonds_;      // cell → bonds
    
    // Cell neighbors (triangular lattice: 6 neighbors per cell)
    vector<vector<size_t>> cell_neighbors_;
    
    // ============================================================
    // EXTRA DOF INTERFACE (for mc::attempt_replica_exchange)
    // Ensures strain state is exchanged together with spins in PT
    // ============================================================
    
    /** Number of extra (non-spin) degrees of freedom to exchange in PT */
    size_t extra_dof_size() const { 
        return use_local_strain_ ? CellStrain::N_DOF * N_cells_ : StrainState::N_DOF; 
    }
    
    /** Pack strain state into flat array for MPI exchange */
    void pack_extra_dof(double* buf) const {
        if (use_local_strain_) {
            for (size_t c = 0; c < N_cells_; ++c)
                cell_strains_[c].to_array(buf + c * CellStrain::N_DOF);
        } else {
            strain.to_array(buf);
        }
    }
    
    /** Unpack strain state from flat array after MPI exchange */
    void unpack_extra_dof(const double* buf) {
        if (use_local_strain_) {
            for (size_t c = 0; c < N_cells_; ++c)
                cell_strains_[c].from_array(buf + c * CellStrain::N_DOF);
        } else {
            strain.from_array(buf);
        }
    }
    
    /**
     * Set uniform Eg strain on all bond types.
     * @param Eg1  Eg1 component: ε_xx = Eg1, ε_yy = -Eg1
     * @param Eg2  Eg2 component: ε_xy = Eg2
     */
    void set_strain_Eg(double Eg1, double Eg2) {
        for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
            strain.epsilon_xx[b] = Eg1;
            strain.epsilon_yy[b] = -Eg1;
            strain.epsilon_xy[b] = Eg2;
        }
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
    
    double drive_energy() const {
        if (std::abs(drive_F_Eg1_) < 1e-15 && std::abs(drive_F_Eg2_) < 1e-15) return 0.0;
        double E = 0.0;
        for (size_t b = 0; b < StrainState::N_BONDS; ++b) {
            double Eg1_b = (strain.epsilon_xx[b] - strain.epsilon_yy[b]) / 2.0;
            double Eg2_b = strain.epsilon_xy[b];
            E -= drive_F_Eg1_ * Eg1_b + drive_F_Eg2_ * Eg2_b;
        }
        return E;
    }
    
    double total_energy() const {
        if (use_local_strain_) {
            return spin_energy() + local_strain_energy() + local_magnetoelastic_energy();
        }
        return spin_energy() + strain_energy() + magnetoelastic_energy() + drive_energy();
    }
    
    double total_energy(double t) const {
        if (use_local_strain_) {
            return spin_energy(t) + local_strain_energy() + local_magnetoelastic_energy();
        }
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
    
    /**
     * Eg spin basis functions for Gammap (Γ') term: f_Γ'^{Eg,1}, f_Γ'^{Eg,2}
     * The Γ' operator on γ-bond: S_i^γ (S_j^α + S_j^β) + (S_i^α + S_i^β) S_j^γ where α,β ≠ γ
     */
    double f_Gammap_Eg1() const;
    double f_Gammap_Eg2() const;
    
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
    SpinVector df_Gammap_Eg1_dS(size_t site) const;
    SpinVector df_Gammap_Eg2_dS(size_t site) const;
    
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
    // LOCAL STRAIN METHODS
    // ============================================================
    
    /**
     * Initialize local strain mode: build cell topology, allocate per-cell arrays.
     * Must be called after constructor and set_parameters() if local_strain=1.
     */
    void init_local_strain();
    
    /** Number of unit cells (valid after init_local_strain). */
    size_t get_N_cells() const { return N_cells_; }
    
    /**
     * Compute Eg spin factors for a single cell (3 NN bonds per cell).
     * Returns (Σ_Eg1, Σ_Eg2) for the cell's bonds only.
     */
    std::pair<double, double> compute_cell_Eg_spin_factors(size_t cell) const;
    
    /**
     * Local magnetoelastic energy: Σ_c λ_Eg [ε_Eg1(c) Σ_Eg1(c) + ε_Eg2(c) Σ_Eg2(c)]
     */
    double local_magnetoelastic_energy() const;
    
    /**
     * Local strain energy: elastic + kinetic + gradient stiffness
     */
    double local_strain_energy() const;
    
    /**
     * Get magnetoelastic field on a spin using its cell's local strain
     */
    SpinVector get_local_magnetoelastic_field(size_t site) const;
    
    /**
     * Compute per-cell strain derivatives (local strain EOM).
     * Includes elastic, ME, gradient, damping, and drive forces.
     */
    void local_strain_derivatives(double t, vector<CellStrain>& dcell_dt) const;
    
    /**
     * Relax all cell strains to local equilibrium (Jacobi iteration).
     * Each cell's strain minimizes E_elastic(c) + E_ME(c) + E_gradient(c).
     */
    void relax_local_strain(bool verbose = true);
    
    /**
     * Save local strain map to file (one line per cell with position + strain)
     */
    void save_local_strain_map(const string& filename) const;
    
    /**
     * Load local strain state from file
     */
    void load_local_strain_state(const string& filename);
    
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
                             const vector<double>& eps_A1g_traj,
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