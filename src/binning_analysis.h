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

struct Bin {
    int index;
    std::vector<double> values;
    std::function<double(const std::vector<double>&)> metric;
    std::function<std::vector<double>(const std::vector<double>&)> gradient;

    Bin(int idx) : index(idx) {
        metric = nullptr;
        gradient = nullptr;
    }
    Bin(int idx, std::function<double(const std::vector<double>&)> metric, std::function<std::vector<double>(const std::vector<double>&)> gradient) : index(idx), metric(metric), gradient(gradient) {}

    void add(double value) {
        values.push_back(value);
    }

    double mean() const {
        if (values.empty()) return 0.0;
        double sum = 0.0;
        for (double v : values) sum += v;
        return sum / values.size();
    }

    double variance() const {
        if (values.size() < 2) return 0.0;
        double mean_value = mean();
        double sum_of_squares = 0.0;
        for (double v : values) {
            sum_of_squares += (v - mean_value) * (v - mean_value);
        }
        return sum_of_squares / (values.size() - 1);
    }

    double std_error() const {
        double var = variance();
        return values.size() > 1 ? std::sqrt(var / values.size()) : 0.0;
    }

    int count() const {
        return values.size();
    }

    double metric_value() const {
        if (metric) {
            return metric(values);
        }
        return 0.0;
    }
};

const std::tuple<double,double> binning_analysis(const std::vector<double>& data, int num_bins, std::function<double(const std::vector<double>&)> metric = nullptr, std::function<std::vector<double>(const std::vector<double>&)> gradient = nullptr) {
    if (data.empty() || num_bins <= 0) {
        std::cerr << "Invalid data or number of bins.\n";
        return std::make_tuple(0.0, 0.0);
    }

    // Create bins
    std::vector<Bin> bins;
    for (int i = 0; i < num_bins; ++i) {
        bins.push_back(Bin(i, metric, gradient));
    }

    // Randomly shuffle the data
    std::vector<double> shuffled_data = data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), gen);

    // Distribute data points into bins
    size_t bin_size = data.size() / num_bins;
    auto it = shuffled_data.begin();
    for (int i = 0; i < num_bins; ++i) {
        size_t current_bin_size = (i == num_bins - 1) ? (shuffled_data.end() - it) : bin_size;
        for (size_t j = 0; j < current_bin_size; ++j) {
            bins[i].add(*it++);
        }
    }

    // Print results
    double mean_var = 0;
    double std_var = 0;
    for (int i = 0; i < num_bins; ++i) {
        mean_var += bins[i].variance();
    }
    mean_var /= num_bins;
    for (int i = 0; i < num_bins; ++i) {
        std_var += (bins[i].variance() - mean_var) * (bins[i].variance() - mean_var);
    }
    std_var /= num_bins - 1;
    std_var = std::sqrt(std_var);
    return std::make_tuple(mean_var, std_var);
}

#endif // BINNING_ANALYSIS_H