/**
 * @file gneb_strain.h
 * @brief GNEB method for combined spin + strain configuration space
 * 
 * This extends the standard GNEB algorithm to include strain degrees of freedom
 * alongside spins. The configuration space is now:
 * 
 *   X = (S_1, S_2, ..., S_N, ε_Eg1, ε_Eg2)
 * 
 * where S_i are unit spins on S^2 and ε_Eg = (ε_xx - ε_yy, 2ε_xy) are the 
 * Eg strain components (2D Euclidean space).
 * 
 * Key modifications:
 * - Spins use geodesic interpolation and geodesic distance
 * - Strain uses linear interpolation and Euclidean distance  
 * - Combined metric: d² = d²_spin + w_strain * d²_strain
 * - Forces include both spin and strain gradients
 * 
 * The minimum energy path now includes both the spin transition AND the
 * accompanying lattice distortion, giving the true transition pathway.
 * 
 * Reference: Bessarab et al., Comp. Phys. Comm. 196, 335 (2015)
 */

#ifndef GNEB_STRAIN_H
#define GNEB_STRAIN_H

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
 * @brief Strain configuration for Eg mode (2 components)
 * 
 * ε_Eg1 = (ε_xx - ε_yy) / 2  -- stretching along x vs y
 * ε_Eg2 = ε_xy               -- shear strain
 * 
 * These transform as the E_g irrep of the D3d point group.
 */
struct StrainEg {
    double Eg1 = 0.0;  // (ε_xx - ε_yy) / 2
    double Eg2 = 0.0;  // ε_xy
    
    StrainEg() = default;
    StrainEg(double e1, double e2) : Eg1(e1), Eg2(e2) {}
    
    // Euclidean operations
    StrainEg operator+(const StrainEg& other) const {
        return StrainEg(Eg1 + other.Eg1, Eg2 + other.Eg2);
    }
    
    StrainEg operator-(const StrainEg& other) const {
        return StrainEg(Eg1 - other.Eg1, Eg2 - other.Eg2);
    }
    
    StrainEg operator*(double scalar) const {
        return StrainEg(Eg1 * scalar, Eg2 * scalar);
    }
    
    double dot(const StrainEg& other) const {
        return Eg1 * other.Eg1 + Eg2 * other.Eg2;
    }
    
    double norm() const {
        return std::sqrt(Eg1 * Eg1 + Eg2 * Eg2);
    }
    
    double squaredNorm() const {
        return Eg1 * Eg1 + Eg2 * Eg2;
    }
    
    // Amplitude (for convenience)
    double amplitude() const { return norm(); }
};

inline StrainEg operator*(double scalar, const StrainEg& strain) {
    return strain * scalar;
}

/**
 * @brief Combined spin + strain configuration for GNEB
 * 
 * The full phase space is (S_1, ..., S_N, ε_Eg1, ε_Eg2).
 * Spins live on S^2 (geodesic manifold), strain is Euclidean.
 */
struct SpinStrainConfig {
    vector<Eigen::Vector3d> spins;  // N unit spins
    StrainEg strain;                 // Eg strain (2 DOF)
    
    SpinStrainConfig() = default;
    
    SpinStrainConfig(size_t n_sites) : spins(n_sites, Eigen::Vector3d::UnitZ()) {}
    
    SpinStrainConfig(const vector<Eigen::Vector3d>& s, const StrainEg& e)
        : spins(s), strain(e) {}
    
    size_t n_sites() const { return spins.size(); }
    
    // Total DOF: 2N (spins on S^2) + 2 (strain)
    size_t total_dof() const { return 2 * spins.size() + 2; }
};

/**
 * @brief Gradient of energy w.r.t. spin + strain configuration
 */
struct SpinStrainGradient {
    vector<Eigen::Vector3d> d_spins;  // ∂E/∂S_i for each site
    StrainEg d_strain;                 // (∂E/∂ε_Eg1, ∂E/∂ε_Eg2)
    
    SpinStrainGradient() = default;
    
    SpinStrainGradient(size_t n_sites) 
        : d_spins(n_sites, Eigen::Vector3d::Zero()) {}
    
    size_t n_sites() const { return d_spins.size(); }
};

/**
 * @brief Result of GNEB optimization with strain
 */
struct GNEBStrainResult {
    // The minimum energy path
    vector<SpinStrainConfig> images;
    
    // Energy of each image
    vector<double> energies;
    
    // Reaction coordinate (arc length) for each image
    vector<double> arc_lengths;
    
    // Index of saddle point (highest energy interior image)
    size_t saddle_index;
    
    // Energy barrier: E_saddle - E_initial
    double barrier;
    
    // Energy difference: E_final - E_initial
    double delta_E;
    
    // Strain at saddle point
    StrainEg saddle_strain;
    
    // Strain at initial and final states
    StrainEg initial_strain;
    StrainEg final_strain;
    
    // Maximum strain amplitude along path
    double max_strain_amplitude;
    
    // Convergence diagnostics
    double max_force;
    size_t iterations_used;
    bool converged;
};

/**
 * @brief Parameters for GNEB with strain
 */
struct GNEBStrainParams {
    // Number of images (including endpoints)
    size_t n_images = 20;
    
    // Spring constant for image spacing
    double spring_constant = 1.0;
    
    // Use climbing image for saddle refinement
    bool climbing_image = true;
    size_t climbing_start = 100;
    
    // Optimization parameters
    double step_size = 0.01;
    double force_tolerance = 1e-5;
    size_t max_iterations = 5000;
    
    // Metric weighting for strain vs spin distance
    // d² = d²_spin + weight_strain * d²_strain
    // Higher weight makes strain "more expensive" to change
    double weight_strain = 1.0;
    
    // Maximum allowed strain amplitude (for safety)
    double max_strain_amplitude = 10.0;
    
    // FIRE optimization
    bool use_fire = true;
    double fire_dtmax = 0.5;
    double fire_alpha_start = 0.1;
    
    // Verbosity: 0 = silent, 1 = summary, 2 = per-iteration
    int verbosity = 1;
    
    // Output
    bool save_intermediate = false;
    string output_dir = "gneb_strain_output";
    size_t save_every = 100;
};

/**
 * @brief GNEB optimizer for combined spin + strain systems
 * 
 * Finds the minimum energy path in the joint (spin, strain) configuration
 * space. The strain coordinates relax along with spins to find the true
 * transition pathway including lattice distortion.
 */
class GNEBStrainOptimizer {
public:
    using EnergyFunc = std::function<double(const SpinStrainConfig&)>;
    using GradientFunc = std::function<SpinStrainGradient(const SpinStrainConfig&)>;
    
    /**
     * @brief Constructor
     * @param energy_func   Computes total energy E(spins, strain)
     * @param gradient_func Computes (∂E/∂S, ∂E/∂ε)
     * @param n_sites       Number of spin sites
     */
    GNEBStrainOptimizer(EnergyFunc energy_func, GradientFunc gradient_func, size_t n_sites)
        : compute_energy(energy_func), compute_gradient(gradient_func), n_sites(n_sites) {}
    
    /**
     * @brief Find minimum energy path between two configurations
     * 
     * @param initial Initial state (e.g., triple-Q with equilibrium strain)
     * @param final   Final state (e.g., zigzag with equilibrium strain)
     * @param params  GNEB optimization parameters
     * @return GNEBStrainResult containing MEP and diagnostics
     */
    GNEBStrainResult find_mep(const SpinStrainConfig& initial, 
                               const SpinStrainConfig& final,
                               const GNEBStrainParams& params = GNEBStrainParams());
    
    /**
     * @brief Generate initial path by interpolation
     * 
     * Spins: geodesic interpolation on S^2
     * Strain: linear interpolation in Euclidean space
     */
    vector<SpinStrainConfig> interpolate_path(const SpinStrainConfig& initial,
                                               const SpinStrainConfig& final,
                                               size_t n_images) const;
    
    /**
     * @brief Compute distance between two configurations
     * 
     * d² = Σ_i θ_i² + w_strain * |Δε|²
     * where θ_i = arccos(S_i^1 · S_i^2) is geodesic distance on sphere
     */
    double configuration_distance(const SpinStrainConfig& c1, 
                                   const SpinStrainConfig& c2,
                                   double weight_strain = 1.0) const;
    
    /**
     * @brief Get current path
     */
    const vector<SpinStrainConfig>& get_path() const { return images; }
    
    /**
     * @brief Save path to files
     */
    void save_path(const string& output_dir, const string& prefix = "mep") const;
    
private:
    EnergyFunc compute_energy;
    GradientFunc compute_gradient;
    size_t n_sites;
    double weight_strain = 1.0;  // Will be set from params
    
    // Current optimization state
    vector<SpinStrainConfig> images;
    vector<double> energies;
    vector<SpinStrainGradient> tangents;  // Tangent vectors along path
    
    // FIRE state
    vector<SpinStrainGradient> velocities;
    double fire_dt;
    double fire_alpha;
    size_t fire_n_positive;
    
    /**
     * @brief Project spin gradient to tangent space of S^2
     * ∇_⊥ = ∇ - (∇ · S) S
     */
    SpinStrainGradient project_to_tangent_space(const SpinStrainGradient& grad,
                                                 const SpinStrainConfig& config) const;
    
    /**
     * @brief Compute tangent vectors at each image
     * Uses energy-weighted upwind scheme (Henkelman-Jonsson)
     */
    void compute_tangents();
    
    /**
     * @brief Compute GNEB force at an image
     * F_GNEB = F_⊥ + F_spring_∥
     */
    SpinStrainGradient compute_neb_force(size_t image_index, double spring_constant);
    
    /**
     * @brief Compute arc lengths along path
     */
    vector<double> compute_arc_lengths() const;
    
    /**
     * @brief Normalize spins to unit length
     */
    void normalize_spins(SpinStrainConfig& config) const;
    
    /**
     * @brief Clamp strain to maximum amplitude
     */
    void clamp_strain(SpinStrainConfig& config, double max_amplitude) const;
    
    /**
     * @brief Find highest energy interior image
     */
    size_t find_climbing_image() const;
    
    /**
     * @brief FIRE optimization step
     */
    double fire_step(vector<SpinStrainGradient>& forces, 
                     const GNEBStrainParams& params,
                     bool climbing);
    
    /**
     * @brief Inner product of two gradients
     * ⟨g1, g2⟩ = Σ_i g1_spin_i · g2_spin_i + w * g1_strain · g2_strain
     */
    double gradient_inner_product(const SpinStrainGradient& g1,
                                   const SpinStrainGradient& g2) const;
    
    /**
     * @brief Norm of gradient
     */
    double gradient_norm(const SpinStrainGradient& g) const;
    
    /**
     * @brief Add scaled gradient: result = a + scale * b
     */
    SpinStrainGradient gradient_axpy(const SpinStrainGradient& a,
                                      double scale,
                                      const SpinStrainGradient& b) const;
};

#endif // GNEB_STRAIN_H
