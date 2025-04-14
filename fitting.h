#ifndef FITTING_H
#define FITTING_H

#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace optimization {

class NelderMead {
public:
    // Constructor with default Nelder-Mead parameters
    NelderMead(double alpha = 1.0, double gamma = 2.0, double rho = 0.5, double sigma = 0.5)
        : alpha_(alpha), gamma_(gamma), rho_(rho), sigma_(sigma) {}

    // Main optimization function with variadic arguments
    template<typename Func, typename... Args>
    std::vector<double> minimize(Func func, 
                                const std::vector<double>& initial_guess,
                                double tolerance = 1e-6, 
                                int max_iterations = 1000,
                                Args&&... args) {
        int n = initial_guess.size();
        
        // Create initial simplex
        std::vector<std::vector<double>> simplex = createInitialSimplex(initial_guess);
        std::vector<double> values(n + 1);
        
        // Evaluate function at each simplex point
        for (int i = 0; i <= n; ++i) {
            values[i] = func(simplex[i], std::forward<Args>(args)...);
        }
        
        int iteration = 0;
        while (iteration < max_iterations) {
            // Sort points by function value
            std::vector<int> indices(n + 1);
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), 
                [&values](int a, int b) { return values[a] < values[b]; });
            
            // Check for convergence
            double std_dev = calculateStdDev(values);
            if (std_dev < tolerance) {
                return simplex[indices[0]];
            }
            
            // Calculate centroid of all points except worst
            std::vector<double> centroid = calculateCentroid(simplex, indices, n);
            
            // Reflection
            std::vector<double> reflection = reflect(simplex[indices[n]], centroid);
            double reflection_value = func(reflection, std::forward<Args>(args)...);
            
            if (reflection_value < values[indices[0]]) {
                // Try expansion
                std::vector<double> expansion = expand(centroid, reflection);
                double expansion_value = func(expansion, std::forward<Args>(args)...);
                
                if (expansion_value < reflection_value) {
                    simplex[indices[n]] = expansion;
                    values[indices[n]] = expansion_value;
                } else {
                    simplex[indices[n]] = reflection;
                    values[indices[n]] = reflection_value;
                }
            } else if (reflection_value < values[indices[n - 1]]) {
                // Accept reflection
                simplex[indices[n]] = reflection;
                values[indices[n]] = reflection_value;
            } else {
                // Contraction
                bool contracted = false;
                
                if (reflection_value < values[indices[n]]) {
                    // Outside contraction
                    std::vector<double> contraction = contractOutside(centroid, reflection);
                    double contraction_value = func(contraction, std::forward<Args>(args)...);
                    
                    if (contraction_value <= reflection_value) {
                        simplex[indices[n]] = contraction;
                        values[indices[n]] = contraction_value;
                        contracted = true;
                    }
                } else {
                    // Inside contraction
                    std::vector<double> contraction = contractInside(centroid, simplex[indices[n]]);
                    double contraction_value = func(contraction, std::forward<Args>(args)...);
                    
                    if (contraction_value < values[indices[n]]) {
                        simplex[indices[n]] = contraction;
                        values[indices[n]] = contraction_value;
                        contracted = true;
                    }
                }
                
                // If contraction failed, shrink the simplex
                if (!contracted) {
                    shrink(simplex, indices, func, values, std::forward<Args>(args)...);
                }
            }
            // Print the best fitting parameters after each iteration
            if (iteration % 1 == 0) {  // Print every 10 iterations to avoid too much output
                std::cout << "Iteration " << iteration << ", Value: " << values[indices[0]] << ", Parameters: ";
                for (double param : simplex[indices[0]]) {
                    std::cout << param << " ";
                }
                std::cout << std::endl;
            }
            ++iteration;
        }
        
        // Return best point if max iterations reached
        std::vector<int> indices(n + 1);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), 
            [&values](int a, int b) { return values[a] < values[b]; });
        
        return simplex[indices[0]];
    }
    
private:
    double alpha_; // reflection coefficient
    double gamma_; // expansion coefficient
    double rho_;   // contraction coefficient
    double sigma_; // shrink coefficient
    
    std::vector<std::vector<double>> createInitialSimplex(const std::vector<double>& initial_guess) {
        int n = initial_guess.size();
        std::vector<std::vector<double>> simplex(n + 1, std::vector<double>(n));
        
        simplex[0] = initial_guess;
        
        for (int i = 0; i < n; ++i) {
            simplex[i + 1] = initial_guess;
            if (std::abs(simplex[i + 1][i]) < 1e-10) {
                simplex[i + 1][i] = 0.01;
            } else {
                simplex[i + 1][i] *= 1.05;
            }
        }
        
        return simplex;
    }
    
    double calculateStdDev(const std::vector<double>& values) {
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double variance = 0.0;
        
        for (double value : values) {
            variance += (value - mean) * (value - mean);
        }
        
        return std::sqrt(variance / values.size());
    }
    
    std::vector<double> calculateCentroid(const std::vector<std::vector<double>>& simplex, 
                                         const std::vector<int>& indices, int n) {
        std::vector<double> centroid(n, 0.0);
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                centroid[j] += simplex[indices[i]][j];
            }
        }
        
        for (int j = 0; j < n; ++j) {
            centroid[j] /= n;
        }
        
        return centroid;
    }
    
    std::vector<double> reflect(const std::vector<double>& worst, const std::vector<double>& centroid) {
        std::vector<double> result(worst.size());
        for (size_t i = 0; i < worst.size(); ++i) {
            result[i] = centroid[i] + alpha_ * (centroid[i] - worst[i]);
        }
        return result;
    }
    
    std::vector<double> expand(const std::vector<double>& centroid, const std::vector<double>& reflection) {
        std::vector<double> result(reflection.size());
        for (size_t i = 0; i < reflection.size(); ++i) {
            result[i] = centroid[i] + gamma_ * (reflection[i] - centroid[i]);
        }
        return result;
    }
    
    std::vector<double> contractOutside(const std::vector<double>& centroid, const std::vector<double>& reflection) {
        std::vector<double> result(reflection.size());
        for (size_t i = 0; i < reflection.size(); ++i) {
            result[i] = centroid[i] + rho_ * (reflection[i] - centroid[i]);
        }
        return result;
    }
    
    std::vector<double> contractInside(const std::vector<double>& centroid, const std::vector<double>& worst) {
        std::vector<double> result(worst.size());
        for (size_t i = 0; i < worst.size(); ++i) {
            result[i] = centroid[i] - rho_ * (centroid[i] - worst[i]);
        }
        return result;
    }
    
    template<typename Func, typename... Args>
    void shrink(std::vector<std::vector<double>>& simplex, 
               const std::vector<int>& indices, 
               Func func, 
               std::vector<double>& values,
               Args&&... args) {
        int n = simplex[0].size();
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j < n; ++j) {
                simplex[indices[i]][j] = simplex[indices[0]][j] + 
                                        sigma_ * (simplex[indices[i]][j] - simplex[indices[0]][j]);
            }
            values[indices[i]] = func(simplex[indices[i]], std::forward<Args>(args)...);
        }
    }
};

} // namespace optimization

#endif // FITTING_H