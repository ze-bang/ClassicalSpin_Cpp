/**
 * @file gneb.cpp
 * @brief Implementation of Geodesic Nudged Elastic Band method
 * 
 * This implements the GNEB algorithm for finding minimum energy paths
 * in classical spin systems with unit-length constraints.
 */

#include "classical_spin/core/gneb.h"
#include <filesystem>
#include <algorithm>
#include <numeric>

// ============================================================================
// GNEBOptimizer Implementation
// ============================================================================

GNEBResult GNEBOptimizer::find_mep(const GNEBSpinConfig& initial, const GNEBSpinConfig& final,
                                    const GNEBParams& params) {
    if (params.verbosity >= 1) {
        cout << "================================================" << endl;
        cout << "GNEB: Finding Minimum Energy Path" << endl;
        cout << "================================================" << endl;
        cout << "Number of images: " << params.n_images << endl;
        cout << "Spring constant: " << params.spring_constant << endl;
        cout << "Climbing image: " << (params.climbing_image ? "enabled" : "disabled") << endl;
        cout << "Optimizer: " << (params.use_fire ? "FIRE" : "Steepest descent") << endl;
        cout << "Force tolerance: " << params.force_tolerance << endl;
        cout << "Max iterations: " << params.max_iterations << endl;
        cout << endl;
    }
    
    // Generate initial path by geodesic interpolation
    images = geodesic_interpolation(initial, final, params.n_images);
    
    // Compute initial energies
    energies.resize(params.n_images);
    for (size_t i = 0; i < params.n_images; ++i) {
        energies[i] = compute_energy(images[i]);
    }
    
    if (params.verbosity >= 1) {
        cout << "Initial path energies:" << endl;
        cout << "  E_initial = " << energies.front() << endl;
        cout << "  E_final = " << energies.back() << endl;
        double E_max = *std::max_element(energies.begin(), energies.end());
        cout << "  E_max (initial) = " << E_max << endl;
        cout << endl;
    }
    
    // Initialize tangent vectors
    current_tangents.resize(params.n_images);
    
    // Initialize FIRE state if using FIRE
    if (params.use_fire) {
        velocities.resize(params.n_images);
        for (size_t i = 0; i < params.n_images; ++i) {
            velocities[i].resize(n_sites, Eigen::Vector3d::Zero());
        }
        fire_dt = params.step_size;
        fire_alpha = params.fire_alpha_start;
        fire_n_positive = 0;
    }
    
    // Main optimization loop
    GNEBResult result;
    result.converged = false;
    result.iterations_used = 0;
    
    vector<GNEBSpinConfig> forces(params.n_images);
    for (size_t i = 0; i < params.n_images; ++i) {
        forces[i].resize(n_sites, Eigen::Vector3d::Zero());
    }
    
    for (size_t iter = 0; iter < params.max_iterations; ++iter) {
        // Compute tangent vectors
        compute_tangents();
        
        // Check if we should enable climbing image
        bool use_climbing = params.climbing_image && (iter >= params.climbing_start);
        
        // Compute NEB forces for all interior images
        double max_force = 0.0;
        
        for (size_t i = 1; i < params.n_images - 1; ++i) {
            forces[i] = compute_neb_force(i);
            
            // For climbing image, modify the force
            if (use_climbing && i == find_climbing_image()) {
                // Climbing image: invert force along tangent
                // F_CI = F - 2(F · τ̂)τ̂
                GNEBSpinConfig grad = compute_gradient(images[i]);
                grad = project_to_tangent_space(grad, images[i]);
                
                // Compute (grad · tangent)
                double grad_dot_tau = 0.0;
                for (size_t s = 0; s < n_sites; ++s) {
                    grad_dot_tau += grad[s].dot(current_tangents[i][s]);
                }
                
                // F_CI = -grad + 2*(grad · τ̂)τ̂
                for (size_t s = 0; s < n_sites; ++s) {
                    forces[i][s] = -grad[s] + 2.0 * grad_dot_tau * current_tangents[i][s];
                }
            }
            
            // Track maximum force for convergence
            for (size_t s = 0; s < n_sites; ++s) {
                max_force = std::max(max_force, forces[i][s].norm());
            }
        }
        
        // Check convergence
        if (max_force < params.force_tolerance) {
            result.converged = true;
            result.iterations_used = iter;
            if (params.verbosity >= 1) {
                cout << "Converged at iteration " << iter << " with max force = " 
                     << max_force << endl;
            }
            break;
        }
        
        // Take optimization step
        double actual_step;
        if (params.use_fire) {
            actual_step = fire_step(forces, use_climbing);
        } else {
            actual_step = steepest_descent_step(forces, params.step_size, use_climbing);
        }
        
        // Normalize all spins
        for (size_t i = 1; i < params.n_images - 1; ++i) {
            normalize_spins(images[i]);
        }
        
        // Recompute energies
        for (size_t i = 0; i < params.n_images; ++i) {
            energies[i] = compute_energy(images[i]);
        }
        
        // Progress output
        if (params.verbosity >= 2 && iter % 100 == 0) {
            size_t ci = find_climbing_image();
            cout << "Iter " << std::setw(5) << iter 
                 << ": max_force = " << std::scientific << std::setprecision(3) << max_force
                 << ", E_saddle = " << std::fixed << std::setprecision(6) << energies[ci]
                 << ", barrier = " << energies[ci] - energies[0]
                 << endl;
        }
        
        // Save intermediate
        if (params.save_intermediate && iter % params.save_every == 0) {
            std::filesystem::create_directories(params.output_dir);
            save_path(params.output_dir, "mep_iter_" + std::to_string(iter));
        }
        
        result.iterations_used = iter;
    }
    
    if (!result.converged && params.verbosity >= 1) {
        cout << "Warning: GNEB did not converge within " << params.max_iterations 
             << " iterations" << endl;
    }
    
    // Compute final results
    compute_tangents();
    
    result.images = images;
    result.energies = energies;
    result.tangents = current_tangents;
    result.arc_lengths = compute_arc_lengths();
    result.saddle_index = find_climbing_image();
    result.barrier = energies[result.saddle_index] - energies[0];
    result.delta_E = energies.back() - energies.front();
    result.max_force = 0.0;
    
    for (size_t i = 1; i < params.n_images - 1; ++i) {
        GNEBSpinConfig grad = compute_gradient(images[i]);
        grad = project_to_tangent_space(grad, images[i]);
        for (size_t s = 0; s < n_sites; ++s) {
            result.max_force = std::max(result.max_force, grad[s].norm());
        }
    }
    
    if (params.verbosity >= 1) {
        cout << endl;
        cout << "================================================" << endl;
        cout << "GNEB Results:" << endl;
        cout << "================================================" << endl;
        cout << "Converged: " << (result.converged ? "yes" : "no") << endl;
        cout << "Iterations: " << result.iterations_used << endl;
        cout << "Max force: " << result.max_force << endl;
        cout << "Saddle image index: " << result.saddle_index << endl;
        cout << "Energy barrier: " << result.barrier << endl;
        cout << "ΔE (final - initial): " << result.delta_E << endl;
        cout << endl;
        
        cout << "Energy profile along MEP:" << endl;
        for (size_t i = 0; i < params.n_images; ++i) {
            cout << "  s = " << std::fixed << std::setprecision(4) << result.arc_lengths[i]
                 << "  E = " << std::setprecision(6) << result.energies[i];
            if (i == result.saddle_index) cout << "  <-- saddle";
            if (i == 0) cout << "  <-- initial";
            if (i == params.n_images - 1) cout << "  <-- final";
            cout << endl;
        }
    }
    
    return result;
}

vector<GNEBSpinConfig> GNEBOptimizer::geodesic_interpolation(const GNEBSpinConfig& initial,
                                                          const GNEBSpinConfig& final,
                                                          size_t n_images) const {
    vector<GNEBSpinConfig> path(n_images);
    
    for (size_t img = 0; img < n_images; ++img) {
        double t = static_cast<double>(img) / (n_images - 1);
        path[img].resize(n_sites);
        
        for (size_t s = 0; s < n_sites; ++s) {
            const Eigen::Vector3d& Si = initial[s];
            const Eigen::Vector3d& Sf = final[s];
            
            // Geodesic (great circle) interpolation on S^2
            double cos_theta = Si.dot(Sf);
            
            // Handle numerical issues
            if (cos_theta > 1.0 - 1e-10) {
                // Spins are nearly parallel
                path[img][s] = Si;
            } else if (cos_theta < -1.0 + 1e-10) {
                // Spins are antiparallel - use linear interpolation through origin
                // and renormalize (will pass through zero, so use perpendicular)
                Eigen::Vector3d perp = Eigen::Vector3d(1, 0, 0).cross(Si);
                if (perp.norm() < 1e-10) {
                    perp = Eigen::Vector3d(0, 1, 0).cross(Si);
                }
                perp.normalize();
                
                double theta_total = M_PI;
                double theta_t = t * theta_total;
                path[img][s] = std::cos(theta_t) * Si + std::sin(theta_t) * perp;
            } else {
                // Standard geodesic interpolation
                double theta = std::acos(cos_theta);
                double sin_theta = std::sin(theta);
                
                double w1 = std::sin((1.0 - t) * theta) / sin_theta;
                double w2 = std::sin(t * theta) / sin_theta;
                
                path[img][s] = w1 * Si + w2 * Sf;
            }
            
            // Ensure unit length
            path[img][s].normalize();
        }
    }
    
    return path;
}

double GNEBOptimizer::project_force_onto_path(const GNEBSpinConfig& force_per_site,
                                               size_t image_index) const {
    if (image_index >= current_tangents.size()) {
        throw std::runtime_error("Invalid image index for force projection");
    }
    
    double projection = 0.0;
    for (size_t s = 0; s < n_sites; ++s) {
        projection += force_per_site[s].dot(current_tangents[image_index][s]);
    }
    
    return projection;
}

void GNEBOptimizer::save_path(const string& output_dir, const string& prefix) const {
    std::filesystem::create_directories(output_dir);
    
    // Save energies and arc lengths
    std::ofstream efile(output_dir + "/" + prefix + "_energies.txt");
    efile << "# image  arc_length  energy" << endl;
    vector<double> arcs = compute_arc_lengths();
    for (size_t i = 0; i < images.size(); ++i) {
        efile << i << " " << arcs[i] << " " << energies[i] << endl;
    }
    efile.close();
    
    // Save each image configuration
    for (size_t i = 0; i < images.size(); ++i) {
        std::ofstream cfile(output_dir + "/" + prefix + "_image_" + std::to_string(i) + ".txt");
        cfile << "# site  Sx  Sy  Sz" << endl;
        for (size_t s = 0; s < n_sites; ++s) {
            cfile << s << " " 
                  << images[i][s](0) << " " 
                  << images[i][s](1) << " " 
                  << images[i][s](2) << endl;
        }
        cfile.close();
    }
}

GNEBSpinConfig GNEBOptimizer::project_to_tangent_space(const GNEBSpinConfig& grad, 
                                                    const GNEBSpinConfig& config) const {
    GNEBSpinConfig projected(n_sites);
    for (size_t s = 0; s < n_sites; ++s) {
        // Project out the component along S: grad_⊥ = grad - (grad · S) S
        projected[s] = grad[s] - grad[s].dot(config[s]) * config[s];
    }
    return projected;
}

void GNEBOptimizer::compute_tangents() {
    // Compute tangent vectors using improved tangent definition
    // from Henkelman & Jonsson, JCP 113, 9978 (2000)
    
    for (size_t i = 0; i < images.size(); ++i) {
        current_tangents[i].resize(n_sites);
        
        if (i == 0) {
            // Initial endpoint: forward difference
            for (size_t s = 0; s < n_sites; ++s) {
                current_tangents[i][s] = images[1][s] - images[0][s];
            }
        } else if (i == images.size() - 1) {
            // Final endpoint: backward difference
            for (size_t s = 0; s < n_sites; ++s) {
                current_tangents[i][s] = images[i][s] - images[i-1][s];
            }
        } else {
            // Interior images: use energy-weighted bisection
            GNEBSpinConfig tau_plus(n_sites), tau_minus(n_sites);
            for (size_t s = 0; s < n_sites; ++s) {
                tau_plus[s] = images[i+1][s] - images[i][s];
                tau_minus[s] = images[i][s] - images[i-1][s];
            }
            
            double dE_plus = energies[i+1] - energies[i];
            double dE_minus = energies[i] - energies[i-1];
            
            if ((dE_plus > 0 && dE_minus > 0) || (dE_plus < 0 && dE_minus < 0)) {
                // Same sign: use energy-weighted average
                double dE_max = std::max(std::abs(dE_plus), std::abs(dE_minus));
                double dE_min = std::min(std::abs(dE_plus), std::abs(dE_minus));
                
                if (dE_plus > dE_minus) {
                    for (size_t s = 0; s < n_sites; ++s) {
                        current_tangents[i][s] = dE_max * tau_plus[s] + dE_min * tau_minus[s];
                    }
                } else {
                    for (size_t s = 0; s < n_sites; ++s) {
                        current_tangents[i][s] = dE_min * tau_plus[s] + dE_max * tau_minus[s];
                    }
                }
            } else {
                // Different signs (at extremum): use upwind direction
                if (dE_plus > dE_minus) {
                    current_tangents[i] = tau_plus;
                } else {
                    current_tangents[i] = tau_minus;
                }
            }
        }
        
        // Project tangent to tangent space of sphere and normalize
        current_tangents[i] = project_to_tangent_space(current_tangents[i], images[i]);
        
        double norm = 0.0;
        for (size_t s = 0; s < n_sites; ++s) {
            norm += current_tangents[i][s].squaredNorm();
        }
        norm = std::sqrt(norm);
        
        if (norm > 1e-10) {
            for (size_t s = 0; s < n_sites; ++s) {
                current_tangents[i][s] /= norm;
            }
        }
    }
}

GNEBSpinConfig GNEBOptimizer::compute_neb_force(size_t image_index) {
    if (image_index == 0 || image_index == images.size() - 1) {
        // Endpoints are fixed
        return GNEBSpinConfig(n_sites, Eigen::Vector3d::Zero());
    }
    
    GNEBSpinConfig force(n_sites);
    
    // Compute true gradient and project to tangent space
    GNEBSpinConfig grad = compute_gradient(images[image_index]);
    grad = project_to_tangent_space(grad, images[image_index]);
    
    // Compute tangent-parallel component of gradient
    double grad_dot_tau = 0.0;
    for (size_t s = 0; s < n_sites; ++s) {
        grad_dot_tau += grad[s].dot(current_tangents[image_index][s]);
    }
    
    // Perpendicular force: F_⊥ = -∇E + (∇E · τ̂)τ̂
    for (size_t s = 0; s < n_sites; ++s) {
        force[s] = -grad[s] + grad_dot_tau * current_tangents[image_index][s];
    }
    
    // Spring force (parallel to tangent)
    double d_plus = geodesic_distance(images[image_index + 1], images[image_index]);
    double d_minus = geodesic_distance(images[image_index], images[image_index - 1]);
    
    // Spring constant from params would be nice but we use member variable
    double k_spring = 1.0;  // Will be set by params in caller
    
    double spring_mag = k_spring * (d_plus - d_minus);
    
    for (size_t s = 0; s < n_sites; ++s) {
        force[s] += spring_mag * current_tangents[image_index][s];
    }
    
    // Project final force to tangent space
    force = project_to_tangent_space(force, images[image_index]);
    
    return force;
}

double GNEBOptimizer::geodesic_distance(const GNEBSpinConfig& c1, const GNEBSpinConfig& c2) const {
    double dist_sq = 0.0;
    for (size_t s = 0; s < n_sites; ++s) {
        double cos_theta = c1[s].dot(c2[s]);
        // Clamp for numerical stability
        cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
        double theta = std::acos(cos_theta);
        dist_sq += theta * theta;
    }
    return std::sqrt(dist_sq);
}

vector<double> GNEBOptimizer::compute_arc_lengths() const {
    vector<double> arcs(images.size());
    arcs[0] = 0.0;
    
    for (size_t i = 1; i < images.size(); ++i) {
        arcs[i] = arcs[i-1] + geodesic_distance(images[i], images[i-1]);
    }
    
    // Normalize to [0, 1]
    double total = arcs.back();
    if (total > 1e-10) {
        for (size_t i = 0; i < images.size(); ++i) {
            arcs[i] /= total;
        }
    }
    
    return arcs;
}

void GNEBOptimizer::normalize_spins(GNEBSpinConfig& config) const {
    for (size_t s = 0; s < n_sites; ++s) {
        double norm = config[s].norm();
        if (norm > 1e-10) {
            config[s] /= norm;
        }
    }
}

size_t GNEBOptimizer::find_climbing_image() const {
    // Find the highest energy interior image
    size_t ci = 1;
    double max_E = energies[1];
    
    for (size_t i = 2; i < images.size() - 1; ++i) {
        if (energies[i] > max_E) {
            max_E = energies[i];
            ci = i;
        }
    }
    
    return ci;
}

double GNEBOptimizer::fire_step(vector<GNEBSpinConfig>& forces, bool climbing) {
    // FIRE algorithm: Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006)
    
    // Compute P = F · V
    double P = 0.0;
    double F_norm_sq = 0.0;
    double V_norm_sq = 0.0;
    
    for (size_t i = 1; i < images.size() - 1; ++i) {
        for (size_t s = 0; s < n_sites; ++s) {
            P += forces[i][s].dot(velocities[i][s]);
            F_norm_sq += forces[i][s].squaredNorm();
            V_norm_sq += velocities[i][s].squaredNorm();
        }
    }
    
    double F_norm = std::sqrt(F_norm_sq);
    double V_norm = std::sqrt(V_norm_sq);
    
    // Adjust velocity: V = (1-α)V + α|V|F̂
    if (F_norm > 1e-10 && V_norm > 1e-10) {
        for (size_t i = 1; i < images.size() - 1; ++i) {
            for (size_t s = 0; s < n_sites; ++s) {
                velocities[i][s] = (1.0 - fire_alpha) * velocities[i][s] 
                                 + fire_alpha * V_norm * forces[i][s] / F_norm;
            }
        }
    }
    
    // Adaptive timestep
    if (P > 0) {
        fire_n_positive++;
        if (fire_n_positive > 5) {  // fire_N_min
            fire_dt = std::min(fire_dt * 1.1, 0.5);  // fire_f_inc, fire_dtmax
            fire_alpha *= 0.99;  // fire_alpha_decrease
        }
    } else {
        // Reset
        fire_n_positive = 0;
        fire_dt *= 0.5;  // fire_f_dec
        fire_dt = std::max(fire_dt, 0.01);  // fire_dtmin
        fire_alpha = 0.1;  // fire_alpha_start
        
        // Zero velocities
        for (size_t i = 1; i < images.size() - 1; ++i) {
            for (size_t s = 0; s < n_sites; ++s) {
                velocities[i][s].setZero();
            }
        }
    }
    
    // Velocity Verlet step
    // x += v*dt + 0.5*F*dt^2
    // v += 0.5*(F_old + F_new)*dt (but we just use F*dt for simplicity)
    for (size_t i = 1; i < images.size() - 1; ++i) {
        for (size_t s = 0; s < n_sites; ++s) {
            images[i][s] += velocities[i][s] * fire_dt + 0.5 * forces[i][s] * fire_dt * fire_dt;
            velocities[i][s] += forces[i][s] * fire_dt;
        }
    }
    
    return fire_dt;
}

double GNEBOptimizer::steepest_descent_step(vector<GNEBSpinConfig>& forces, 
                                             double step_size, bool climbing) {
    for (size_t i = 1; i < images.size() - 1; ++i) {
        for (size_t s = 0; s < n_sites; ++s) {
            images[i][s] += step_size * forces[i][s];
        }
    }
    return step_size;
}

// ============================================================================
// DrivenTransitionAnalyzer Implementation
// ============================================================================

double DrivenTransitionAnalyzer::force_projection(size_t image_index, double Q_E1) const {
    if (image_index >= mep.images.size()) {
        throw std::runtime_error("Invalid image index");
    }
    
    // Compute the phonon-induced force at this configuration
    GNEBSpinConfig force = compute_phonon_force(mep.images[image_index], Q_E1);
    
    // Project onto the MEP tangent
    double projection = 0.0;
    for (size_t s = 0; s < force.size(); ++s) {
        projection += force[s].dot(mep.tangents[image_index][s]);
    }
    
    return projection;
}

double DrivenTransitionAnalyzer::effective_barrier(double Q_E1) const {
    // Compute the effective energy landscape with the drive
    // E_eff(s) = E(s) - ∫_0^s ds' P(s', Q)
    
    size_t n = mep.images.size();
    vector<double> E_eff(n);
    
    // Trapezoidal integration of force projection
    double integral = 0.0;
    E_eff[0] = mep.energies[0];
    
    for (size_t i = 1; i < n; ++i) {
        double ds = mep.arc_lengths[i] - mep.arc_lengths[i-1];
        double P_i = force_projection(i, Q_E1);
        double P_im1 = force_projection(i-1, Q_E1);
        
        integral += 0.5 * (P_i + P_im1) * ds;
        E_eff[i] = mep.energies[i] - integral;
    }
    
    // Find the effective barrier
    double E_eff_max = *std::max_element(E_eff.begin() + 1, E_eff.end() - 1);
    
    return E_eff_max - E_eff[0];
}

double DrivenTransitionAnalyzer::find_spinodal_Q(double Q_max, double tol) const {
    // Binary search for the critical Q where barrier vanishes
    
    double Q_low = 0.0;
    double Q_high = Q_max;
    
    // Check if spinodal exists
    double barrier_at_max = effective_barrier(Q_max);
    if (barrier_at_max > tol) {
        // Barrier still positive at Q_max
        return Q_max;
    }
    
    // Binary search
    while (Q_high - Q_low > tol * Q_max) {
        double Q_mid = 0.5 * (Q_low + Q_high);
        double barrier_mid = effective_barrier(Q_mid);
        
        if (barrier_mid > tol) {
            Q_low = Q_mid;
        } else {
            Q_high = Q_mid;
        }
    }
    
    return 0.5 * (Q_low + Q_high);
}

vector<double> DrivenTransitionAnalyzer::e1_eigenmode_overlap() const {
    // For each image, compute how well the E1 coupling direction
    // overlaps with the tangent direction
    
    vector<double> overlaps(mep.images.size());
    
    for (size_t i = 0; i < mep.images.size(); ++i) {
        // Get the E1 coupling direction at unit Q
        GNEBSpinConfig e1_dir = compute_phonon_force(mep.images[i], 1.0);
        
        // Normalize
        double norm = 0.0;
        for (size_t s = 0; s < e1_dir.size(); ++s) {
            norm += e1_dir[s].squaredNorm();
        }
        norm = std::sqrt(norm);
        
        if (norm > 1e-10) {
            for (size_t s = 0; s < e1_dir.size(); ++s) {
                e1_dir[s] /= norm;
            }
        }
        
        // Compute overlap with tangent
        double overlap = 0.0;
        for (size_t s = 0; s < e1_dir.size(); ++s) {
            overlap += e1_dir[s].dot(mep.tangents[i][s]);
        }
        
        overlaps[i] = std::abs(overlap);  // Take absolute value
    }
    
    return overlaps;
}

vector<vector<double>> DrivenTransitionAnalyzer::energy_landscape_dynamics(
    const vector<double>& Q_E1_trajectory) const {
    
    size_t n_times = Q_E1_trajectory.size();
    size_t n_images = mep.images.size();
    
    vector<vector<double>> E_dynamics(n_times, vector<double>(n_images));
    
    for (size_t t = 0; t < n_times; ++t) {
        double Q = Q_E1_trajectory[t];
        
        // Compute effective energies at this Q
        double integral = 0.0;
        E_dynamics[t][0] = mep.energies[0];
        
        for (size_t i = 1; i < n_images; ++i) {
            double ds = mep.arc_lengths[i] - mep.arc_lengths[i-1];
            double P_i = force_projection(i, Q);
            double P_im1 = force_projection(i-1, Q);
            
            integral += 0.5 * (P_i + P_im1) * ds;
            E_dynamics[t][i] = mep.energies[i] - integral;
        }
    }
    
    return E_dynamics;
}

void DrivenTransitionAnalyzer::save_analysis(const string& output_dir) const {
    std::filesystem::create_directories(output_dir);
    
    // Save MEP energies
    std::ofstream mep_file(output_dir + "/mep_energies.txt");
    mep_file << "# image  arc_length  energy  tangent_norm" << endl;
    for (size_t i = 0; i < mep.images.size(); ++i) {
        double tau_norm = 0.0;
        for (size_t s = 0; s < mep.tangents[i].size(); ++s) {
            tau_norm += mep.tangents[i][s].squaredNorm();
        }
        tau_norm = std::sqrt(tau_norm);
        
        mep_file << i << " " << mep.arc_lengths[i] << " " << mep.energies[i] 
                 << " " << tau_norm << endl;
    }
    mep_file.close();
    
    // Save E1 overlap analysis
    vector<double> overlaps = e1_eigenmode_overlap();
    std::ofstream overlap_file(output_dir + "/e1_overlap.txt");
    overlap_file << "# image  arc_length  |<E1|tangent>|" << endl;
    for (size_t i = 0; i < mep.images.size(); ++i) {
        overlap_file << i << " " << mep.arc_lengths[i] << " " << overlaps[i] << endl;
    }
    overlap_file.close();
    
    // Save effective barrier vs Q
    std::ofstream barrier_file(output_dir + "/effective_barrier_vs_Q.txt");
    barrier_file << "# Q_E1  effective_barrier" << endl;
    for (double Q = 0.0; Q <= 1.0; Q += 0.01) {
        barrier_file << Q << " " << effective_barrier(Q) << endl;
    }
    barrier_file.close();
    
    // Save force projections
    std::ofstream proj_file(output_dir + "/force_projections.txt");
    proj_file << "# image  arc_length  P(s, Q=0.1)  P(s, Q=0.5)  P(s, Q=1.0)" << endl;
    for (size_t i = 0; i < mep.images.size(); ++i) {
        proj_file << i << " " << mep.arc_lengths[i] << " "
                  << force_projection(i, 0.1) << " "
                  << force_projection(i, 0.5) << " "
                  << force_projection(i, 1.0) << endl;
    }
    proj_file.close();
}

// ============================================================================
// HessianAnalyzer Implementation
// ============================================================================

Eigen::MatrixXd HessianAnalyzer::compute_hessian(const GNEBSpinConfig& config, double delta) {
    // Dimension of tangent space: 2 per spin (θ, φ in spherical coords)
    size_t dim = 2 * n_sites;
    Eigen::MatrixXd H(dim, dim);
    
    double E0 = compute_energy(config);
    
    // For each pair of tangent directions
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = i; j < dim; ++j) {
            // Create perturbation vectors
            Eigen::VectorXd ei = Eigen::VectorXd::Zero(dim);
            Eigen::VectorXd ej = Eigen::VectorXd::Zero(dim);
            ei(i) = 1.0;
            ej(j) = 1.0;
            
            // Compute E(+δi, +δj), E(+δi, -δj), E(-δi, +δj), E(-δi, -δj)
            GNEBSpinConfig config_pp = config;
            GNEBSpinConfig config_pm = config;
            GNEBSpinConfig config_mp = config;
            GNEBSpinConfig config_mm = config;
            
            // Apply perturbations
            GNEBSpinConfig pert_i = tangent_to_perturbation(ei * delta, config);
            GNEBSpinConfig pert_j = tangent_to_perturbation(ej * delta, config);
            
            for (size_t s = 0; s < n_sites; ++s) {
                config_pp[s] = config[s] + pert_i[s] + pert_j[s];
                config_pm[s] = config[s] + pert_i[s] - pert_j[s];
                config_mp[s] = config[s] - pert_i[s] + pert_j[s];
                config_mm[s] = config[s] - pert_i[s] - pert_j[s];
                
                // Renormalize
                config_pp[s].normalize();
                config_pm[s].normalize();
                config_mp[s].normalize();
                config_mm[s].normalize();
            }
            
            double Epp = compute_energy(config_pp);
            double Epm = compute_energy(config_pm);
            double Emp = compute_energy(config_mp);
            double Emm = compute_energy(config_mm);
            
            // Second derivative by finite difference
            double Hessian_ij = (Epp - Epm - Emp + Emm) / (4.0 * delta * delta);
            
            H(i, j) = Hessian_ij;
            H(j, i) = Hessian_ij;
        }
    }
    
    return H;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> 
HessianAnalyzer::eigen_decomposition(const GNEBSpinConfig& config) {
    Eigen::MatrixXd H = compute_hessian(config);
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
    
    return {solver.eigenvalues(), solver.eigenvectors()};
}

vector<double> HessianAnalyzer::eigenmode_overlap(const GNEBSpinConfig& config,
                                                   const GNEBSpinConfig& direction) {
    auto [eigenvalues, eigenvectors] = eigen_decomposition(config);
    
    // Convert direction to tangent space
    Eigen::VectorXd dir_tangent = perturbation_to_tangent(direction, config);
    
    // Normalize
    dir_tangent.normalize();
    
    // Compute overlaps
    vector<double> overlaps(eigenvalues.size());
    for (int i = 0; i < eigenvalues.size(); ++i) {
        double overlap = dir_tangent.dot(eigenvectors.col(i));
        overlaps[i] = overlap * overlap;  // |⟨v|d⟩|²
    }
    
    return overlaps;
}

std::pair<double, GNEBSpinConfig> 
HessianAnalyzer::find_unstable_mode(const GNEBSpinConfig& config) {
    auto [eigenvalues, eigenvectors] = eigen_decomposition(config);
    
    // Find most negative eigenvalue
    int min_idx = 0;
    double min_val = eigenvalues(0);
    for (int i = 1; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < min_val) {
            min_val = eigenvalues(i);
            min_idx = i;
        }
    }
    
    if (min_val >= 0) {
        cout << "Warning: No unstable mode found (all eigenvalues non-negative)" << endl;
    }
    
    // Convert eigenvector to spin perturbation
    GNEBSpinConfig mode = tangent_to_perturbation(eigenvectors.col(min_idx), config);
    
    return {min_val, mode};
}

GNEBSpinConfig HessianAnalyzer::tangent_to_perturbation(const Eigen::VectorXd& tangent_vec,
                                                     const GNEBSpinConfig& base_config) const {
    // Convert tangent space coordinates to 3D spin perturbations
    // Each spin has 2 tangent directions, which we choose as:
    // - e_theta: perpendicular to S in the plane containing S and z-axis
    // - e_phi: perpendicular to both S and e_theta
    
    GNEBSpinConfig perturbation(n_sites);
    
    for (size_t s = 0; s < n_sites; ++s) {
        const Eigen::Vector3d& S = base_config[s];
        
        // Construct orthonormal basis in tangent plane
        Eigen::Vector3d e1, e2;
        
        // e1 = S × z (unless S is parallel to z)
        e1 = S.cross(Eigen::Vector3d(0, 0, 1));
        if (e1.norm() < 1e-10) {
            e1 = S.cross(Eigen::Vector3d(1, 0, 0));
        }
        e1.normalize();
        
        // e2 = S × e1
        e2 = S.cross(e1);
        e2.normalize();
        
        // Perturbation = c1 * e1 + c2 * e2
        double c1 = tangent_vec(2 * s);
        double c2 = tangent_vec(2 * s + 1);
        
        perturbation[s] = c1 * e1 + c2 * e2;
    }
    
    return perturbation;
}

Eigen::VectorXd HessianAnalyzer::perturbation_to_tangent(const GNEBSpinConfig& perturbation,
                                                          const GNEBSpinConfig& base_config) const {
    Eigen::VectorXd tangent_vec(2 * n_sites);
    
    for (size_t s = 0; s < n_sites; ++s) {
        const Eigen::Vector3d& S = base_config[s];
        
        // Same orthonormal basis
        Eigen::Vector3d e1, e2;
        e1 = S.cross(Eigen::Vector3d(0, 0, 1));
        if (e1.norm() < 1e-10) {
            e1 = S.cross(Eigen::Vector3d(1, 0, 0));
        }
        e1.normalize();
        e2 = S.cross(e1);
        e2.normalize();
        
        // Project perturbation onto tangent basis
        // First remove any component along S
        Eigen::Vector3d pert_tangent = perturbation[s] - perturbation[s].dot(S) * S;
        
        tangent_vec(2 * s) = pert_tangent.dot(e1);
        tangent_vec(2 * s + 1) = pert_tangent.dot(e2);
    }
    
    return tangent_vec;
}
