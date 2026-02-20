/**
 * @file gneb.h
 * @brief Geodesic Nudged Elastic Band (GNEB) method for minimum energy path finding
 * 
 * Implements GNEB on the manifold of unit spin vectors |S_i| = 1 for finding
 * transition paths between spin configurations.
 * 
 * Key concepts:
 * - Minimum Energy Path (MEP): the optimal path connecting two minima
 * - Saddle point: maximum along the MEP defines the activation barrier
 * - Geodesic interpolation: path on the sphere S^2 instead of linear interpolation
 * - Climbing Image NEB: highest energy image climbs to the saddle point
 * 
 * References:
 * - Bessarab et al., Comp. Phys. Comm. 196, 335 (2015) - GNEB for spin systems
 * - Henkelman et al., J. Chem. Phys. 113, 9901 (2000) - CI-NEB
 * - Dittrich et al., J. Magn. Magn. Mater. 250, 12 (2002) - Geodesics on spin manifolds
 * 
 * The MEP provides the proper "reaction coordinate" for transition state theory,
 * and allows projection of driven forces onto the escape direction.
 */

#ifndef GNEB_H
#define GNEB_H

#include <vector>
#include <array>
#include <functional>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>

using std::vector;
using std::string;
using std::cout;
using std::endl;

/**
 * Spin configuration for GNEB: array of unit 3-vectors
 */
using GNEBSpinConfig = vector<Eigen::Vector3d>;

/**
 * Result of GNEB optimization
 */
struct GNEBResult {
    // The minimum energy path (vector of images)
    vector<GNEBSpinConfig> images;
    
    // Energy of each image along the path
    vector<double> energies;
    
    // Reaction coordinate (arc length) for each image
    vector<double> arc_lengths;
    
    // Tangent vectors at each image (normalized)
    vector<GNEBSpinConfig> tangents;
    
    // Index of the saddle point image (highest energy interior image)
    size_t saddle_index;
    
    // Energy barrier: E_saddle - E_initial
    double barrier;
    
    // Energy difference: E_final - E_initial
    double delta_E;
    
    // Convergence diagnostics
    double max_force;           // Maximum force component at convergence
    size_t iterations_used;     // Number of iterations to converge
    bool converged;             // Whether optimization converged
    
    // Curvature at the saddle (negative = unstable mode)
    double saddle_curvature;
};

/**
 * Parameters for GNEB optimization
 */
struct GNEBParams {
    // Number of images in the path (including endpoints)
    size_t n_images = 20;
    
    // Spring constant for image spacing
    double spring_constant = 1.0;
    
    // Use climbing image for saddle point refinement
    bool climbing_image = true;
    
    // Climbing image starts after this many iterations
    size_t climbing_start = 100;
    
    // Optimization parameters
    double step_size = 0.01;     // Initial step size for steepest descent
    double max_step = 0.1;       // Maximum step size per iteration
    double force_tolerance = 1e-5;  // Convergence criterion for max force
    size_t max_iterations = 5000;
    
    // Use velocity-Verlet with FIRE dynamics (usually faster)
    bool use_fire = true;
    
    // FIRE parameters (Fast Inertial Relaxation Engine)
    double fire_dtmax = 0.5;     // Maximum timestep
    double fire_dtmin = 0.01;    // Minimum timestep
    double fire_alpha_start = 0.1;
    double fire_alpha_decrease = 0.99;
    size_t fire_N_min = 5;       // Steps before increasing dt
    double fire_f_inc = 1.1;
    double fire_f_dec = 0.5;
    
    // Verbosity: 0 = silent, 1 = summary, 2 = per-iteration
    int verbosity = 1;
    
    // Save intermediate paths
    bool save_intermediate = false;
    string output_dir = "gneb_output";
    size_t save_every = 100;
};

/**
 * GNEB optimizer for finding minimum energy paths in spin systems
 */
class GNEBOptimizer {
public:
    using EnergyFunc = std::function<double(const GNEBSpinConfig&)>;
    using GradientFunc = std::function<GNEBSpinConfig(const GNEBSpinConfig&)>;
    
    /**
     * Constructor
     * @param energy_func  Function computing total energy for a spin configuration
     * @param gradient_func Function computing gradient ∂E/∂S for each spin
     *                     (returns the 3D gradient vector for each site)
     * @param n_sites      Number of spin sites in the system
     */
    GNEBOptimizer(EnergyFunc energy_func, GradientFunc gradient_func, size_t n_sites)
        : compute_energy(energy_func), compute_gradient(gradient_func), n_sites(n_sites) {}
    
    /**
     * Find the minimum energy path between two configurations
     * 
     * @param initial  Initial spin configuration (local minimum)
     * @param final    Final spin configuration (local minimum)
     * @param params   GNEB optimization parameters
     * @return GNEBResult containing the MEP and diagnostics
     */
    GNEBResult find_mep(const GNEBSpinConfig& initial, const GNEBSpinConfig& final,
                       const GNEBParams& params = GNEBParams());
    
    /**
     * Find the minimum energy path starting from an initial path guess
     * 
     * Use this when you have a good initial guess (e.g., converged path at
     * a nearby parameter value). The endpoints will be kept fixed.
     * 
     * @param initial_path  Initial guess for the path (vector of images)
     * @param params        GNEB optimization parameters
     * @return GNEBResult containing the MEP and diagnostics
     */
    GNEBResult find_mep_from_path(const vector<GNEBSpinConfig>& initial_path,
                                   const GNEBParams& params = GNEBParams());
    
    /**
     * Generate initial path by geodesic interpolation
     * 
     * For each spin, interpolate along the great circle on S^2:
     * S(t) = sin((1-t)θ)/sin(θ) * S_i + sin(tθ)/sin(θ) * S_f
     * where θ = arccos(S_i · S_f)
     * 
     * @param initial Initial configuration
     * @param final   Final configuration
     * @param n_images Number of images (including endpoints)
     * @return Vector of images forming the initial path
     */
    vector<GNEBSpinConfig> geodesic_interpolation(const GNEBSpinConfig& initial,
                                               const GNEBSpinConfig& final,
                                               size_t n_images) const;
    
    /**
     * Compute the projection of a force onto the MEP tangent
     * 
     * This is the key quantity for analyzing driven transitions:
     * P(t,s) = Σ_i (-∂H_drive/∂S_i) · t^_i(s)
     * 
     * If positive, the drive pushes "forward" along the path toward the product.
     * 
     * @param force_per_site  The force vector at each site (e.g., from phonon drive)
     * @param image_index     Which image to evaluate at
     * @return Projection magnitude (positive = forward, negative = backward)
     */
    double project_force_onto_path(const GNEBSpinConfig& force_per_site,
                                   size_t image_index) const;
    
    /**
     * Get the tangent vector at a specific image
     */
    const GNEBSpinConfig& get_tangent(size_t image_index) const {
        return current_tangents[image_index];
    }
    
    /**
     * Get the current path (all images)
     */
    const vector<GNEBSpinConfig>& get_path() const {
        return images;
    }
    
    /**
     * Save the current path to files
     */
    void save_path(const string& output_dir, const string& prefix = "mep") const;
    
    /**
     * Load a path from files
     */
    void load_path(const string& output_dir, const string& prefix = "mep");
    
private:
    EnergyFunc compute_energy;
    GradientFunc compute_gradient;
    size_t n_sites;
    
    // Current state of the optimization
    vector<GNEBSpinConfig> images;
    vector<double> energies;
    vector<GNEBSpinConfig> current_tangents;
    
    // FIRE state
    vector<GNEBSpinConfig> velocities;
    double fire_dt;
    double fire_alpha;
    size_t fire_n_positive;
    
    /**
     * Project gradient onto the tangent space of the spin sphere
     * ∇_⊥ = ∇ - (∇ · S) S
     * This ensures the gradient is orthogonal to S (on the tangent plane of S^2)
     */
    GNEBSpinConfig project_to_tangent_space(const GNEBSpinConfig& grad, 
                                        const GNEBSpinConfig& config) const;
    
    /**
     * Compute tangent vectors for all images using upwind finite differences
     * Follows Henkelman & Jonsson for improved stability at the saddle
     */
    void compute_tangents();
    
    /**
     * Compute the NEB force for a single image
     * F_NEB = F_⊥ + F_spring_‖
     * 
     * F_⊥ = ∇E - (∇E · τ̂)τ̂  (perpendicular energy gradient)
     * F_spring_‖ = k(|R_{i+1} - R_i| - |R_i - R_{i-1}|)τ̂  (parallel spring force)
     */
    GNEBSpinConfig compute_neb_force(size_t image_index);
    
    /**
     * Compute geodesic distance between two spin configurations
     * d = Σ_i arccos(S_i^(1) · S_i^(2))
     */
    double geodesic_distance(const GNEBSpinConfig& c1, const GNEBSpinConfig& c2) const;
    
    /**
     * Compute arc length parameter along the path
     */
    vector<double> compute_arc_lengths() const;
    
    /**
     * Normalize all spins in a configuration to unit length
     */
    void normalize_spins(GNEBSpinConfig& config) const;
    
    /**
     * FIRE optimization step
     */
    double fire_step(vector<GNEBSpinConfig>& forces, bool climbing);
    
    /**
     * Steepest descent step
     */
    double steepest_descent_step(vector<GNEBSpinConfig>& forces, double step_size, bool climbing);
    
    /**
     * Find the climbing image (highest energy interior image)
     */
    size_t find_climbing_image() const;
};

// ============================================================================
// COLLECTIVE VARIABLE ANALYSIS
// ============================================================================

/**
 * Collective variable definitions for triple-Q ↔ zigzag transition analysis
 */
struct CollectiveVariables {
    // Structure factor weights at ordering wavevectors
    double m_3Q;      // Triple-Q structure factor weight
    double m_zigzag;  // Zigzag structure factor weight
    
    // Symmetry order parameters
    double f_E1_amplitude;  // |f_E1| = sqrt(f_K_Eg1² + f_K_Eg2²) (E1/Eg breaking)
    double f_A1g;           // A1g bilinear
    
    // Energy decomposition
    double E_kitaev;
    double E_heisenberg;
    double E_gamma;
    double E_ring;
    double E_total;
};

/**
 * Free energy surface estimator using collective variables
 * 
 * Computes F(m_3Q, m_zz, f_E1) from constrained minimizations or
 * sampling with umbrella potentials.
 */
class FreeEnergySurface {
public:
    struct GridPoint {
        double m_3Q;
        double m_zz;
        double f_E1;
        double energy;
        GNEBSpinConfig config;  // Representative configuration
    };
    
    using EnergyFunc = std::function<double(const GNEBSpinConfig&)>;
    using CVFunc = std::function<CollectiveVariables(const GNEBSpinConfig&)>;
    
    FreeEnergySurface(EnergyFunc energy, CVFunc cv_calculator)
        : compute_energy(energy), compute_cv(cv_calculator) {}
    
    /**
     * Compute constrained energy surface
     * 
     * For each grid point (m_3Q, m_zz), find the minimum energy configuration
     * with those collective variables constrained.
     * 
     * @param m_3Q_grid  Grid values for m_3Q
     * @param m_zz_grid  Grid values for m_zz
     * @param initial    Initial configuration for constrained minimization
     * @param constraint_strength  Strength of harmonic constraint
     * @return Grid of (m_3Q, m_zz, E) points
     */
    vector<GridPoint> compute_energy_surface(
        const vector<double>& m_3Q_grid,
        const vector<double>& m_zz_grid,
        const GNEBSpinConfig& initial,
        double constraint_strength = 100.0);
    
    /**
     * Find the minimum energy path on the CV surface using string method
     * 
     * @param initial_cv  Initial CV values (e.g., triple-Q minimum)
     * @param final_cv    Final CV values (e.g., zigzag minimum)
     * @param n_images    Number of string images
     * @return Path in CV space with energies
     */
    vector<GridPoint> string_method_2d(
        const std::array<double, 2>& initial_cv,
        const std::array<double, 2>& final_cv,
        size_t n_images = 20);
    
private:
    EnergyFunc compute_energy;
    CVFunc compute_cv;
};

// ============================================================================
// DRIVEN TRANSITION ANALYSIS
// ============================================================================

/**
 * Analysis tools for phonon-driven transitions
 * 
 * Key quantity: projection of the driven force onto the MEP tangent
 * P(t,s) = Σ_i (-∂H_sp-ph/∂S_i) · τ̂_i(s)
 * 
 * This quantifies whether the drive "pushes along the reaction coordinate"
 * versus shaking orthogonal directions.
 */
class DrivenTransitionAnalyzer {
public:
    using SpinForceFunc = std::function<GNEBSpinConfig(const GNEBSpinConfig&, double Q_E1)>;
    
    /**
     * Constructor
     * @param mep_result  The minimum energy path from GNEB
     * @param force_func  Function computing -∂H_sp-ph/∂S_i for given Q_E1
     */
    DrivenTransitionAnalyzer(const GNEBResult& mep_result, SpinForceFunc force_func)
        : mep(mep_result), compute_phonon_force(force_func) {}
    
    /**
     * Compute the projection of driven force onto MEP tangent
     * 
     * @param image_index  Which image along the path
     * @param Q_E1         Phonon amplitude (can be time-dependent drive value)
     * @return P(s, Q) = Σ_i F_i(Q) · τ̂_i(s)
     */
    double force_projection(size_t image_index, double Q_E1) const;
    
    /**
     * Compute the effective barrier reduction from the drive
     * 
     * At each point along the path, the drive adds:
     * δE(s) = -∫ ds' P(s', Q)
     * 
     * The effective barrier is:
     * ΔE_eff = max_s[E(s) + δE(s)] - [E(0) + δE(0)]
     * 
     * @param Q_E1  Phonon amplitude
     * @return Effective barrier at this drive amplitude
     */
    double effective_barrier(double Q_E1) const;
    
    /**
     * Find the critical Q_E1 where barrier vanishes (spinodal)
     * 
     * If the drive is strong enough, it can make the transition barrierless.
     * 
     * @param Q_max  Maximum Q to search up to
     * @param tol    Tolerance for barrier ≈ 0
     * @return Critical Q_E1 (or Q_max if not found)
     */
    double find_spinodal_Q(double Q_max, double tol = 1e-4) const;
    
    /**
     * Analyze the overlap between E1 eigenmodes and escape direction
     * 
     * Compute the projection of ∂f_E1/∂S onto the tangent direction at each
     * point along the MEP. This tells you how well the E1 channel couples
     * to the transition.
     * 
     * @return Vector of overlaps at each image
     */
    vector<double> e1_eigenmode_overlap() const;
    
    /**
     * Compute time-dependent energy along the path during driving
     * 
     * @param Q_E1_trajectory  Time series of Q_E1(t) values
     * @return Matrix of energies: E[time_index][image_index]
     */
    vector<vector<double>> energy_landscape_dynamics(
        const vector<double>& Q_E1_trajectory) const;
    
    /**
     * Save analysis results to file
     */
    void save_analysis(const string& output_dir) const;
    
private:
    GNEBResult mep;
    SpinForceFunc compute_phonon_force;
    
    // Cached analysis results
    mutable vector<double> cached_projections;
    mutable double cached_Q;
    mutable bool cache_valid = false;
};

// ============================================================================
// HESSIAN AND NORMAL MODE ANALYSIS
// ============================================================================

/**
 * Compute the Hessian matrix at a spin configuration
 * 
 * The Hessian is ∂²E/∂S_i^α ∂S_j^β projected onto the tangent space
 * of the spin manifold (since |S| = 1 is constrained).
 */
class HessianAnalyzer {
public:
    using EnergyFunc = std::function<double(const GNEBSpinConfig&)>;
    
    HessianAnalyzer(EnergyFunc energy, size_t n_sites)
        : compute_energy(energy), n_sites(n_sites) {}
    
    /**
     * Compute the Hessian matrix using finite differences
     * 
     * @param config  Spin configuration (should be at a stationary point)
     * @param delta   Step size for finite differences
     * @return Hessian matrix projected to tangent space (2N × 2N)
     */
    Eigen::MatrixXd compute_hessian(const GNEBSpinConfig& config, double delta = 1e-5);
    
    /**
     * Eigenvalue decomposition of the Hessian
     * 
     * @param config  Spin configuration
     * @return Pair of (eigenvalues, eigenvectors)
     */
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> 
    eigen_decomposition(const GNEBSpinConfig& config);
    
    /**
     * Compute overlap of eigenmodes with a given direction
     * 
     * @param config    Spin configuration
     * @param direction Target direction (e.g., ∂f_E1/∂S)
     * @return Vector of |⟨v_n | direction⟩|² for each eigenmode
     */
    vector<double> eigenmode_overlap(const GNEBSpinConfig& config,
                                     const GNEBSpinConfig& direction);
    
    /**
     * Find the unstable mode at a saddle point
     * 
     * @param config  Configuration (should be at saddle)
     * @return Pair of (negative eigenvalue, eigenvector direction)
     */
    std::pair<double, GNEBSpinConfig> 
    find_unstable_mode(const GNEBSpinConfig& config);
    
private:
    EnergyFunc compute_energy;
    size_t n_sites;
    
    /**
     * Convert tangent space vector to GNEBSpinConfig perturbation
     * Each spin has 2 tangent directions (θ, φ in spherical coords)
     */
    GNEBSpinConfig tangent_to_perturbation(const Eigen::VectorXd& tangent_vec,
                                        const GNEBSpinConfig& base_config) const;
    
    /**
     * Convert GNEBSpinConfig perturbation to tangent space vector
     */
    Eigen::VectorXd perturbation_to_tangent(const GNEBSpinConfig& perturbation,
                                            const GNEBSpinConfig& base_config) const;
};

#endif // GNEB_H
