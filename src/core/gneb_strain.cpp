/**
 * @file gneb_strain.cpp
 * @brief Implementation of GNEB method for spin + strain configuration space
 */

#include "classical_spin/core/gneb_strain.h"
#include <filesystem>
#include <algorithm>
#include <numeric>

// ============================================================================
// GNEBStrainOptimizer Implementation
// ============================================================================

GNEBStrainResult GNEBStrainOptimizer::find_mep(const SpinStrainConfig& initial,
                                                 const SpinStrainConfig& final,
                                                 const GNEBStrainParams& params) {
    weight_strain = params.weight_strain;
    
    if (params.verbosity >= 1) {
        cout << "================================================" << endl;
        cout << "GNEB with Strain: Finding Minimum Energy Path" << endl;
        cout << "================================================" << endl;
        cout << "Number of images: " << params.n_images << endl;
        cout << "Spring constant: " << params.spring_constant << endl;
        cout << "Strain weight: " << params.weight_strain << endl;
        cout << "Climbing image: " << (params.climbing_image ? "enabled" : "disabled") << endl;
        cout << "Force tolerance: " << params.force_tolerance << endl;
        cout << "Max iterations: " << params.max_iterations << endl;
        cout << endl;
        cout << "Initial strain: (" << initial.strain.Eg1 << ", " << initial.strain.Eg2 << ")" << endl;
        cout << "Final strain:   (" << final.strain.Eg1 << ", " << final.strain.Eg2 << ")" << endl;
        cout << endl;
    }
    
    // Generate initial path
    images = interpolate_path(initial, final, params.n_images);
    
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
    tangents.resize(params.n_images);
    for (size_t i = 0; i < params.n_images; ++i) {
        tangents[i] = SpinStrainGradient(n_sites);
    }
    
    // Initialize FIRE state
    if (params.use_fire) {
        velocities.resize(params.n_images);
        for (size_t i = 0; i < params.n_images; ++i) {
            velocities[i] = SpinStrainGradient(n_sites);
        }
        fire_dt = params.step_size;
        fire_alpha = params.fire_alpha_start;
        fire_n_positive = 0;
    }
    
    // Main optimization loop
    GNEBStrainResult result;
    result.converged = false;
    result.iterations_used = 0;
    
    vector<SpinStrainGradient> forces(params.n_images);
    for (size_t i = 0; i < params.n_images; ++i) {
        forces[i] = SpinStrainGradient(n_sites);
    }
    
    for (size_t iter = 0; iter < params.max_iterations; ++iter) {
        // Compute tangent vectors
        compute_tangents();
        
        // Check if climbing image should be enabled
        bool use_climbing = params.climbing_image && (iter >= params.climbing_start);
        
        // Compute GNEB forces for all interior images
        double max_force = 0.0;
        
        for (size_t i = 1; i < params.n_images - 1; ++i) {
            forces[i] = compute_neb_force(i, params.spring_constant);
            
            // Climbing image modification
            if (use_climbing && i == find_climbing_image()) {
                // F_CI = F - 2(F · τ̂)τ̂  (invert component along tangent)
                SpinStrainGradient grad = compute_gradient(images[i]);
                grad = project_to_tangent_space(grad, images[i]);
                
                double grad_dot_tau = gradient_inner_product(grad, tangents[i]);
                
                // F_CI = -grad + 2*(grad · τ̂)τ̂
                forces[i] = gradient_axpy(
                    gradient_axpy(SpinStrainGradient(n_sites), -1.0, grad),
                    2.0 * grad_dot_tau,
                    tangents[i]
                );
            }
            
            // Track maximum force
            double f_norm = gradient_norm(forces[i]);
            max_force = std::max(max_force, f_norm);
        }
        
        // Check convergence
        if (max_force < params.force_tolerance) {
            result.converged = true;
            result.iterations_used = iter;
            if (params.verbosity >= 1) {
                cout << "Converged at iteration " << iter 
                     << " with max force = " << max_force << endl;
            }
            break;
        }
        
        // Take optimization step
        double actual_step;
        if (params.use_fire) {
            actual_step = fire_step(forces, params, use_climbing);
        } else {
            // Simple steepest descent
            for (size_t i = 1; i < params.n_images - 1; ++i) {
                for (size_t s = 0; s < n_sites; ++s) {
                    images[i].spins[s] += params.step_size * forces[i].d_spins[s];
                }
                images[i].strain.Eg1 += params.step_size * forces[i].d_strain.Eg1;
                images[i].strain.Eg2 += params.step_size * forces[i].d_strain.Eg2;
            }
            actual_step = params.step_size;
        }
        
        // Normalize spins and clamp strain
        for (size_t i = 1; i < params.n_images - 1; ++i) {
            normalize_spins(images[i]);
            clamp_strain(images[i], params.max_strain_amplitude);
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
                 << ", E_saddle = " << std::fixed << std::setprecision(4) << energies[ci]
                 << ", barrier = " << energies[ci] - energies[0]
                 << ", ε_saddle = (" << images[ci].strain.Eg1 
                 << ", " << images[ci].strain.Eg2 << ")"
                 << endl;
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
    result.arc_lengths = compute_arc_lengths();
    result.saddle_index = find_climbing_image();
    result.barrier = energies[result.saddle_index] - energies[0];
    result.delta_E = energies.back() - energies.front();
    
    result.initial_strain = images.front().strain;
    result.final_strain = images.back().strain;
    result.saddle_strain = images[result.saddle_index].strain;
    
    // Find max strain amplitude along path
    result.max_strain_amplitude = 0.0;
    for (const auto& img : images) {
        result.max_strain_amplitude = std::max(result.max_strain_amplitude, 
                                                img.strain.amplitude());
    }
    
    // Compute final max force
    result.max_force = 0.0;
    for (size_t i = 1; i < params.n_images - 1; ++i) {
        SpinStrainGradient grad = compute_gradient(images[i]);
        grad = project_to_tangent_space(grad, images[i]);
        result.max_force = std::max(result.max_force, gradient_norm(grad));
    }
    
    if (params.verbosity >= 1) {
        cout << endl;
        cout << "================================================" << endl;
        cout << "GNEB with Strain Results:" << endl;
        cout << "================================================" << endl;
        cout << "Converged: " << (result.converged ? "yes" : "no") << endl;
        cout << "Iterations: " << result.iterations_used << endl;
        cout << "Max force: " << result.max_force << endl;
        cout << "Saddle image index: " << result.saddle_index << endl;
        cout << "Energy barrier: " << result.barrier << endl;
        cout << "ΔE (final - initial): " << result.delta_E << endl;
        cout << endl;
        cout << "Strain along path:" << endl;
        cout << "  Initial: ε_Eg = (" << result.initial_strain.Eg1 
             << ", " << result.initial_strain.Eg2 << ")" << endl;
        cout << "  Saddle:  ε_Eg = (" << result.saddle_strain.Eg1 
             << ", " << result.saddle_strain.Eg2 << ")" << endl;
        cout << "  Final:   ε_Eg = (" << result.final_strain.Eg1 
             << ", " << result.final_strain.Eg2 << ")" << endl;
        cout << "  Max amplitude: " << result.max_strain_amplitude << endl;
        cout << endl;
    }
    
    return result;
}

vector<SpinStrainConfig> GNEBStrainOptimizer::interpolate_path(
    const SpinStrainConfig& initial,
    const SpinStrainConfig& final,
    size_t n_images) const {
    
    vector<SpinStrainConfig> path(n_images);
    
    for (size_t img = 0; img < n_images; ++img) {
        double t = static_cast<double>(img) / (n_images - 1);
        path[img] = SpinStrainConfig(n_sites);
        
        // Geodesic interpolation for spins
        for (size_t s = 0; s < n_sites; ++s) {
            const Eigen::Vector3d& Si = initial.spins[s];
            const Eigen::Vector3d& Sf = final.spins[s];
            
            double cos_theta = Si.dot(Sf);
            cos_theta = std::max(-1.0, std::min(1.0, cos_theta));  // Clamp
            
            if (cos_theta > 1.0 - 1e-10) {
                // Nearly parallel
                path[img].spins[s] = Si;
            } else if (cos_theta < -1.0 + 1e-10) {
                // Antiparallel - find perpendicular direction
                Eigen::Vector3d perp = Eigen::Vector3d(1, 0, 0).cross(Si);
                if (perp.norm() < 1e-10) {
                    perp = Eigen::Vector3d(0, 1, 0).cross(Si);
                }
                perp.normalize();
                
                double theta_t = t * M_PI;
                path[img].spins[s] = std::cos(theta_t) * Si + std::sin(theta_t) * perp;
            } else {
                // Standard geodesic (slerp)
                double theta = std::acos(cos_theta);
                double sin_theta = std::sin(theta);
                double w1 = std::sin((1.0 - t) * theta) / sin_theta;
                double w2 = std::sin(t * theta) / sin_theta;
                path[img].spins[s] = w1 * Si + w2 * Sf;
            }
            
            path[img].spins[s].normalize();
        }
        
        // Linear interpolation for strain
        path[img].strain.Eg1 = (1.0 - t) * initial.strain.Eg1 + t * final.strain.Eg1;
        path[img].strain.Eg2 = (1.0 - t) * initial.strain.Eg2 + t * final.strain.Eg2;
    }
    
    return path;
}

double GNEBStrainOptimizer::configuration_distance(const SpinStrainConfig& c1,
                                                    const SpinStrainConfig& c2,
                                                    double w_strain) const {
    // Spin distance: sum of geodesic distances squared
    double d_spin_sq = 0.0;
    for (size_t s = 0; s < n_sites; ++s) {
        double cos_theta = c1.spins[s].dot(c2.spins[s]);
        cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
        double theta = std::acos(cos_theta);
        d_spin_sq += theta * theta;
    }
    
    // Strain distance: Euclidean
    double d_strain_sq = (c1.strain - c2.strain).squaredNorm();
    
    // Combined metric
    return std::sqrt(d_spin_sq + w_strain * d_strain_sq);
}

void GNEBStrainOptimizer::save_path(const string& output_dir, const string& prefix) const {
    std::filesystem::create_directories(output_dir);
    
    // Save energies, arc lengths, and strain
    std::ofstream efile(output_dir + "/" + prefix + "_energies.txt");
    efile << "# image  arc_length  energy  strain_Eg1  strain_Eg2  strain_amp" << endl;
    vector<double> arcs = compute_arc_lengths();
    for (size_t i = 0; i < images.size(); ++i) {
        efile << i << " " << arcs[i] << " " << energies[i] 
              << " " << images[i].strain.Eg1 
              << " " << images[i].strain.Eg2
              << " " << images[i].strain.amplitude() << endl;
    }
    efile.close();
    
    // Save each image configuration
    for (size_t i = 0; i < images.size(); ++i) {
        std::ofstream cfile(output_dir + "/" + prefix + "_image_" + std::to_string(i) + ".txt");
        cfile << "# Spin-strain image " << i << endl;
        cfile << "# strain_Eg1 = " << images[i].strain.Eg1 << endl;
        cfile << "# strain_Eg2 = " << images[i].strain.Eg2 << endl;
        cfile << "# site  Sx  Sy  Sz" << endl;
        for (size_t s = 0; s < n_sites; ++s) {
            cfile << s << " " 
                  << images[i].spins[s](0) << " " 
                  << images[i].spins[s](1) << " " 
                  << images[i].spins[s](2) << endl;
        }
        cfile.close();
    }
}

SpinStrainGradient GNEBStrainOptimizer::project_to_tangent_space(
    const SpinStrainGradient& grad,
    const SpinStrainConfig& config) const {
    
    SpinStrainGradient projected(n_sites);
    
    // Spin: project out radial component
    for (size_t s = 0; s < n_sites; ++s) {
        projected.d_spins[s] = grad.d_spins[s] 
                               - grad.d_spins[s].dot(config.spins[s]) * config.spins[s];
    }
    
    // Strain: no constraint, just copy
    projected.d_strain = grad.d_strain;
    
    return projected;
}

void GNEBStrainOptimizer::compute_tangents() {
    // Compute tangent vectors using energy-weighted upwind scheme
    for (size_t i = 0; i < images.size(); ++i) {
        tangents[i] = SpinStrainGradient(n_sites);
        
        SpinStrainGradient tau_plus(n_sites), tau_minus(n_sites);
        
        if (i == 0) {
            // Forward difference
            for (size_t s = 0; s < n_sites; ++s) {
                tangents[i].d_spins[s] = images[1].spins[s] - images[0].spins[s];
            }
            tangents[i].d_strain = images[1].strain - images[0].strain;
        } else if (i == images.size() - 1) {
            // Backward difference
            for (size_t s = 0; s < n_sites; ++s) {
                tangents[i].d_spins[s] = images[i].spins[s] - images[i-1].spins[s];
            }
            tangents[i].d_strain = images[i].strain - images[i-1].strain;
        } else {
            // Interior: energy-weighted bisection
            for (size_t s = 0; s < n_sites; ++s) {
                tau_plus.d_spins[s] = images[i+1].spins[s] - images[i].spins[s];
                tau_minus.d_spins[s] = images[i].spins[s] - images[i-1].spins[s];
            }
            tau_plus.d_strain = images[i+1].strain - images[i].strain;
            tau_minus.d_strain = images[i].strain - images[i-1].strain;
            
            double dE_plus = energies[i+1] - energies[i];
            double dE_minus = energies[i] - energies[i-1];
            
            if ((dE_plus > 0 && dE_minus > 0) || (dE_plus < 0 && dE_minus < 0)) {
                // Same sign: energy-weighted average
                double dE_max = std::max(std::abs(dE_plus), std::abs(dE_minus));
                double dE_min = std::min(std::abs(dE_plus), std::abs(dE_minus));
                
                if (dE_plus > dE_minus) {
                    tangents[i] = gradient_axpy(
                        gradient_axpy(SpinStrainGradient(n_sites), dE_max, tau_plus),
                        dE_min, tau_minus
                    );
                } else {
                    tangents[i] = gradient_axpy(
                        gradient_axpy(SpinStrainGradient(n_sites), dE_min, tau_plus),
                        dE_max, tau_minus
                    );
                }
            } else {
                // Different signs: upwind
                if (dE_plus > dE_minus) {
                    tangents[i] = tau_plus;
                } else {
                    tangents[i] = tau_minus;
                }
            }
        }
        
        // Project to tangent space and normalize
        tangents[i] = project_to_tangent_space(tangents[i], images[i]);
        
        double norm = gradient_norm(tangents[i]);
        if (norm > 1e-10) {
            for (size_t s = 0; s < n_sites; ++s) {
                tangents[i].d_spins[s] /= norm;
            }
            tangents[i].d_strain.Eg1 /= norm;
            tangents[i].d_strain.Eg2 /= norm;
        }
    }
}

SpinStrainGradient GNEBStrainOptimizer::compute_neb_force(size_t image_index,
                                                          double spring_constant) {
    if (image_index == 0 || image_index == images.size() - 1) {
        return SpinStrainGradient(n_sites);  // Endpoints fixed
    }
    
    SpinStrainGradient force(n_sites);
    
    // True gradient projected to tangent space
    SpinStrainGradient grad = compute_gradient(images[image_index]);
    grad = project_to_tangent_space(grad, images[image_index]);
    
    // Component along tangent
    double grad_dot_tau = gradient_inner_product(grad, tangents[image_index]);
    
    // Perpendicular force: F_⊥ = -∇E + (∇E · τ̂)τ̂
    for (size_t s = 0; s < n_sites; ++s) {
        force.d_spins[s] = -grad.d_spins[s] 
                           + grad_dot_tau * tangents[image_index].d_spins[s];
    }
    force.d_strain.Eg1 = -grad.d_strain.Eg1 
                         + grad_dot_tau * tangents[image_index].d_strain.Eg1;
    force.d_strain.Eg2 = -grad.d_strain.Eg2 
                         + grad_dot_tau * tangents[image_index].d_strain.Eg2;
    
    // Spring force (parallel to tangent)
    double d_plus = configuration_distance(images[image_index + 1], images[image_index], weight_strain);
    double d_minus = configuration_distance(images[image_index], images[image_index - 1], weight_strain);
    double spring_mag = spring_constant * (d_plus - d_minus);
    
    for (size_t s = 0; s < n_sites; ++s) {
        force.d_spins[s] += spring_mag * tangents[image_index].d_spins[s];
    }
    force.d_strain.Eg1 += spring_mag * tangents[image_index].d_strain.Eg1;
    force.d_strain.Eg2 += spring_mag * tangents[image_index].d_strain.Eg2;
    
    // Project final force
    force = project_to_tangent_space(force, images[image_index]);
    
    return force;
}

vector<double> GNEBStrainOptimizer::compute_arc_lengths() const {
    vector<double> arcs(images.size());
    arcs[0] = 0.0;
    
    for (size_t i = 1; i < images.size(); ++i) {
        arcs[i] = arcs[i-1] + configuration_distance(images[i], images[i-1], weight_strain);
    }
    
    // Normalize to [0, 1]
    double total = arcs.back();
    if (total > 1e-10) {
        for (auto& a : arcs) a /= total;
    }
    
    return arcs;
}

void GNEBStrainOptimizer::normalize_spins(SpinStrainConfig& config) const {
    for (auto& S : config.spins) {
        double norm = S.norm();
        if (norm > 1e-10) S /= norm;
    }
}

void GNEBStrainOptimizer::clamp_strain(SpinStrainConfig& config, double max_amplitude) const {
    double amp = config.strain.amplitude();
    if (amp > max_amplitude) {
        double scale = max_amplitude / amp;
        config.strain.Eg1 *= scale;
        config.strain.Eg2 *= scale;
    }
}

size_t GNEBStrainOptimizer::find_climbing_image() const {
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

double GNEBStrainOptimizer::gradient_inner_product(const SpinStrainGradient& g1,
                                                    const SpinStrainGradient& g2) const {
    double result = 0.0;
    for (size_t s = 0; s < n_sites; ++s) {
        result += g1.d_spins[s].dot(g2.d_spins[s]);
    }
    result += weight_strain * g1.d_strain.dot(g2.d_strain);
    return result;
}

double GNEBStrainOptimizer::gradient_norm(const SpinStrainGradient& g) const {
    return std::sqrt(gradient_inner_product(g, g));
}

SpinStrainGradient GNEBStrainOptimizer::gradient_axpy(const SpinStrainGradient& a,
                                                       double scale,
                                                       const SpinStrainGradient& b) const {
    SpinStrainGradient result(n_sites);
    for (size_t s = 0; s < n_sites; ++s) {
        result.d_spins[s] = a.d_spins[s] + scale * b.d_spins[s];
    }
    result.d_strain.Eg1 = a.d_strain.Eg1 + scale * b.d_strain.Eg1;
    result.d_strain.Eg2 = a.d_strain.Eg2 + scale * b.d_strain.Eg2;
    return result;
}

double GNEBStrainOptimizer::fire_step(vector<SpinStrainGradient>& forces,
                                       const GNEBStrainParams& params,
                                       bool climbing) {
    // FIRE: Fast Inertial Relaxation Engine
    // Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006)
    
    // P = F · V
    double P = 0.0;
    double F_norm_sq = 0.0;
    double V_norm_sq = 0.0;
    
    for (size_t i = 1; i < images.size() - 1; ++i) {
        P += gradient_inner_product(forces[i], velocities[i]);
        F_norm_sq += gradient_inner_product(forces[i], forces[i]);
        V_norm_sq += gradient_inner_product(velocities[i], velocities[i]);
    }
    
    double F_norm = std::sqrt(F_norm_sq);
    double V_norm = std::sqrt(V_norm_sq);
    
    // Mix velocity with force direction
    if (F_norm > 1e-10 && V_norm > 1e-10) {
        for (size_t i = 1; i < images.size() - 1; ++i) {
            for (size_t s = 0; s < n_sites; ++s) {
                velocities[i].d_spins[s] = (1.0 - fire_alpha) * velocities[i].d_spins[s]
                                           + fire_alpha * (V_norm / F_norm) * forces[i].d_spins[s];
            }
            velocities[i].d_strain.Eg1 = (1.0 - fire_alpha) * velocities[i].d_strain.Eg1
                                         + fire_alpha * (V_norm / F_norm) * forces[i].d_strain.Eg1;
            velocities[i].d_strain.Eg2 = (1.0 - fire_alpha) * velocities[i].d_strain.Eg2
                                         + fire_alpha * (V_norm / F_norm) * forces[i].d_strain.Eg2;
        }
    }
    
    // Check power sign
    if (P > 0) {
        fire_n_positive++;
        if (fire_n_positive > 5) {
            fire_dt = std::min(fire_dt * 1.1, params.fire_dtmax);
            fire_alpha *= 0.99;
        }
    } else {
        // Reset
        fire_n_positive = 0;
        fire_dt *= 0.5;
        fire_alpha = params.fire_alpha_start;
        
        // Zero velocities
        for (size_t i = 1; i < images.size() - 1; ++i) {
            for (size_t s = 0; s < n_sites; ++s) {
                velocities[i].d_spins[s].setZero();
            }
            velocities[i].d_strain = StrainEg(0.0, 0.0);
        }
    }
    
    // Velocity Verlet: update velocities then positions
    for (size_t i = 1; i < images.size() - 1; ++i) {
        // v += dt * F
        for (size_t s = 0; s < n_sites; ++s) {
            velocities[i].d_spins[s] += fire_dt * forces[i].d_spins[s];
        }
        velocities[i].d_strain.Eg1 += fire_dt * forces[i].d_strain.Eg1;
        velocities[i].d_strain.Eg2 += fire_dt * forces[i].d_strain.Eg2;
        
        // x += dt * v
        for (size_t s = 0; s < n_sites; ++s) {
            images[i].spins[s] += fire_dt * velocities[i].d_spins[s];
        }
        images[i].strain.Eg1 += fire_dt * velocities[i].d_strain.Eg1;
        images[i].strain.Eg2 += fire_dt * velocities[i].d_strain.Eg2;
    }
    
    return fire_dt;
}
