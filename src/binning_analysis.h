#ifndef BINNING_ANALYSIS_H
#define BINNING_ANALYSIS_H

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <cmath>
#include <tuple>
#include <numeric>
#include <memory>
#include <thread>
#include <future>
#include <execution>

// Enhanced statistical results structure
struct BinningResults {
    double mean_variance;
    double std_variance;
    double autocorrelation_time;
    double effective_sample_size;
    double error_estimate;
    std::vector<double> bin_means;
    std::vector<double> bin_variances;
    int optimal_num_bins;
    double convergence_ratio;
};

// Fast bin class with optimized statistics calculation
struct FastBin {
    int index;
    std::vector<double> values;
    mutable double cached_mean = std::numeric_limits<double>::quiet_NaN();
    mutable double cached_variance = std::numeric_limits<double>::quiet_NaN();
    mutable bool stats_dirty = true;
    
    std::function<double(const std::vector<double>&)> metric;
    std::function<std::vector<double>(const std::vector<double>&)> gradient;

    FastBin(int idx) : index(idx), metric(nullptr), gradient(nullptr) {
        values.reserve(1000); // Pre-allocate for better performance
    }
    
    FastBin(int idx, std::function<double(const std::vector<double>&)> m, 
            std::function<std::vector<double>(const std::vector<double>&)> g) 
        : index(idx), metric(m), gradient(g) {
        values.reserve(1000);
    }

    void add(double value) {
        values.push_back(value);
        stats_dirty = true;
    }
    
    void reserve(size_t size) {
        values.reserve(size);
    }

    // Optimized single-pass mean and variance calculation
    void calculate_stats() const {
        if (!stats_dirty) return;
        
        if (values.empty()) {
            cached_mean = 0.0;
            cached_variance = 0.0;
            stats_dirty = false;
            return;
        }
        
        // Welford's online algorithm for numerical stability
        double mean = 0.0;
        double m2 = 0.0;
        size_t n = 0;
        
        for (double x : values) {
            ++n;
            double delta = x - mean;
            mean += delta / n;
            double delta2 = x - mean;
            m2 += delta * delta2;
        }
        
        cached_mean = mean;
        cached_variance = (n > 1) ? m2 / (n - 1) : 0.0;
        stats_dirty = false;
    }

    double mean() const {
        calculate_stats();
        return cached_mean;
    }

    double variance() const {
        calculate_stats();
        return cached_variance;
    }

    double std_error() const {
        double var = variance();
        return values.size() > 1 ? std::sqrt(var / values.size()) : 0.0;
    }

    size_t count() const {
        return values.size();
    }

    double metric_value() const {
        if (metric && !values.empty()) {
            return metric(values);
        }
        return 0.0;
    }
};

// Calculate autocorrelation function
std::vector<double> calculate_autocorrelation(const std::vector<double>& data, int max_lag = -1) {
    if (data.size() < 2) return {};
    
    if (max_lag < 0) max_lag = std::min(static_cast<int>(data.size() / 4), 100);
    
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double variance = 0.0;
    for (double x : data) {
        variance += (x - mean) * (x - mean);
    }
    variance /= data.size();
    
    if (variance == 0.0) return std::vector<double>(max_lag + 1, 0.0);
    
    std::vector<double> autocorr(max_lag + 1, 0.0);
    
    for (int lag = 0; lag <= max_lag; ++lag) {
        double sum = 0.0;
        for (size_t i = 0; i < data.size() - lag; ++i) {
            sum += (data[i] - mean) * (data[i + lag] - mean);
        }
        autocorr[lag] = sum / ((data.size() - lag) * variance);
    }
    
    return autocorr;
}

// Estimate integrated autocorrelation time
double estimate_autocorr_time(const std::vector<double>& autocorr) {
    if (autocorr.empty()) return 1.0;
    
    double tau_int = 0.5; // tau_int = 0.5 + sum_{t=1}^{W} rho(t)
    
    // Automatic windowing procedure
    for (size_t t = 1; t < autocorr.size(); ++t) {
        tau_int += autocorr[t];
        
        // Stop when we have enough data (W >= c * tau_int with c ~ 6-8)
        if (t >= 6.0 * tau_int) break;
        
        // Stop if autocorrelation becomes negative (often noise)
        if (autocorr[t] <= 0) break;
    }
    
    return std::max(tau_int, 0.5);
}

// Determine optimal number of bins based on autocorrelation
int calculate_optimal_bins(const std::vector<double>& data, double tau_int) {
    // Rule of thumb: each bin should contain ~2*tau_int independent samples
    double independent_samples = data.size() / (2.0 * tau_int);
    int optimal_bins = static_cast<int>(std::sqrt(independent_samples));
    
    // Constrain to reasonable range
    return std::max(2, std::min(optimal_bins, static_cast<int>(data.size() / 10)));
}

// Enhanced binning analysis with automatic optimization
BinningResults enhanced_binning_analysis(
    const std::vector<double>& data, 
    int num_bins = -1,  // -1 for automatic selection
    std::function<double(const std::vector<double>&)> metric = nullptr,
    std::function<std::vector<double>(const std::vector<double>&)> gradient = nullptr,
    bool parallel = true) {
    
    BinningResults results;
    
    if (data.empty()) {
        std::cerr << "Error: Empty data provided to binning analysis.\n";
        return results;
    }
    
    if (data.size() < 10) {
        std::cerr << "Warning: Very small dataset for binning analysis.\n";
    }
    
    // Calculate autocorrelation properties
    auto autocorr = calculate_autocorrelation(data);
    double tau_int = estimate_autocorr_time(autocorr);
    results.autocorrelation_time = tau_int;
    results.effective_sample_size = data.size() / (2.0 * tau_int);
    
    // Determine optimal number of bins if not specified
    if (num_bins <= 0) {
        num_bins = calculate_optimal_bins(data, tau_int);
    }
    results.optimal_num_bins = num_bins;
    
    if (num_bins > static_cast<int>(data.size() / 2)) {
        std::cerr << "Warning: Too many bins for dataset size. Reducing bin count.\n";
        num_bins = std::max(2, static_cast<int>(data.size() / 10));
    }
    
    // Create bins with pre-allocated size
    std::vector<FastBin> bins;
    bins.reserve(num_bins);
    size_t expected_bin_size = data.size() / num_bins + 1;
    
    for (int i = 0; i < num_bins; ++i) {
        bins.emplace_back(i, metric, gradient);
        bins[i].reserve(expected_bin_size);
    }
    
    // Use block-based binning for better statistical properties
    // (consecutive blocks preserve local correlations better than random shuffle)
    size_t block_size = data.size() / num_bins;
    size_t remainder = data.size() % num_bins;
    
    size_t data_idx = 0;
    for (int i = 0; i < num_bins; ++i) {
        size_t current_block_size = block_size + (i < remainder ? 1 : 0);
        
        for (size_t j = 0; j < current_block_size && data_idx < data.size(); ++j) {
            bins[i].add(data[data_idx++]);
        }
    }
    
    // Calculate statistics (potentially in parallel)
    results.bin_means.reserve(num_bins);
    results.bin_variances.reserve(num_bins);
    
    if (parallel && num_bins > 4) {
        // Parallel calculation of bin statistics
        std::vector<std::future<std::pair<double, double>>> futures;
        futures.reserve(num_bins);
        
        for (int i = 0; i < num_bins; ++i) {
            futures.emplace_back(std::async(std::launch::async, [&bins, i]() {
                return std::make_pair(bins[i].mean(), bins[i].variance());
            }));
        }
        
        for (int i = 0; i < num_bins; ++i) {
            auto [mean, var] = futures[i].get();
            results.bin_means.push_back(mean);
            results.bin_variances.push_back(var);
        }
    } else {
        // Sequential calculation
        for (int i = 0; i < num_bins; ++i) {
            results.bin_means.push_back(bins[i].mean());
            results.bin_variances.push_back(bins[i].variance());
        }
    }
    
    // Calculate mean and standard deviation of bin variances
    if (!results.bin_variances.empty()) {
        results.mean_variance = std::accumulate(results.bin_variances.begin(), 
                                              results.bin_variances.end(), 0.0) / num_bins;
        
        double sum_sq_diff = 0.0;
        for (double var : results.bin_variances) {
            sum_sq_diff += (var - results.mean_variance) * (var - results.mean_variance);
        }
        results.std_variance = std::sqrt(sum_sq_diff / std::max(1, num_bins - 1));
    }
    
    // Improved error estimate using autocorrelation information
    double total_variance = 0.0;
    for (const auto& bin : bins) {
        if (bin.count() > 0) {
            total_variance += bin.variance() * bin.count();
        }
    }
    total_variance /= data.size();
    
    results.error_estimate = std::sqrt(total_variance * 2.0 * tau_int / data.size());
    
    // Convergence check: ratio of error estimates from different bin sizes
    if (num_bins > 4) {
        // Quick test with half the bins
        auto half_result = enhanced_binning_analysis(data, num_bins / 2, metric, gradient, false);
        results.convergence_ratio = results.error_estimate / half_result.error_estimate;
    } else {
        results.convergence_ratio = 1.0;
    }
    
    return results;
}

// Legacy interface for backward compatibility
std::tuple<double, double> binning_analysis(
    const std::vector<double>& data, 
    int num_bins, 
    std::function<double(const std::vector<double>&)> metric = nullptr, 
    std::function<std::vector<double>(const std::vector<double>&)> gradient = nullptr) {
    
    auto results = enhanced_binning_analysis(data, num_bins, metric, gradient, false);
    return std::make_tuple(results.mean_variance, results.std_variance);
}

// Utility function to print detailed results
void print_binning_results(const BinningResults& results, std::ostream& os = std::cout) {
    os << std::fixed << std::setprecision(6);
    os << "=== Enhanced Binning Analysis Results ===\n";
    os << "Number of bins: " << results.optimal_num_bins << "\n";
    os << "Autocorrelation time: " << results.autocorrelation_time << "\n";
    os << "Effective sample size: " << results.effective_sample_size << "\n";
    os << "Mean bin variance: " << results.mean_variance << " ± " << results.std_variance << "\n";
    os << "Error estimate: " << results.error_estimate << "\n";
    os << "Convergence ratio: " << results.convergence_ratio << "\n";
    
    if (results.convergence_ratio > 1.2) {
        os << "Warning: Poor convergence detected. Consider more bins or longer simulation.\n";
    }
    if (results.effective_sample_size < 10) {
        os << "Warning: Very few independent samples. Results may be unreliable.\n";
    }
    os << "==========================================\n";
}

#endif // BINNING_ANALYSIS_H