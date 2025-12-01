#ifndef LIN_ALG_REFACTORED_H
#define LIN_ALG_REFACTORED_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

// Type aliases for clarity
using SpinVector = VectorXd;
using SpinMatrix = MatrixXd;
using SpinTensor3 = std::vector<MatrixXd>;  // For trilinear interactions

// Structure constants for SU(2) and SU(3)
class StructureConstants {
public:
    static SpinTensor3 SU2_structure_constant();
    static SpinTensor3 SU3_structure_constant();
    
private:
    static void set_permutation(SpinTensor3& A, size_t a, size_t b, size_t c, double val);
};

// Global structure constants (lazy initialization) - defined in simple_linear_alg.cpp
const SpinTensor3& get_SU2_structure();
const SpinTensor3& get_SU3_structure();

// Matrix-vector multiplication using Eigen
inline SpinVector multiply(const SpinMatrix& M, const SpinVector& v) {
    return M * v;
}

// Bilinear contraction: a^T * M * b
inline double contract(const SpinVector& a, const SpinMatrix& M, const SpinVector& b) {
    return a.dot(M * b);
}

// Trilinear contraction functions - defined in simple_linear_alg.cpp
double contract_trilinear(const SpinTensor3& M, const SpinVector& a, 
                         const SpinVector& b, const SpinVector& c);
SpinVector contract_trilinear_field(const SpinTensor3& M, 
                                    const SpinVector& b, 
                                    const SpinVector& c);

// Cross product functions - defined in simple_linear_alg.cpp
SpinVector cross_prod_SU2(const SpinVector& a, const SpinVector& b);
SpinVector cross_prod_SU3(const SpinVector& a, const SpinVector& b);
SpinVector cross_product(const SpinVector& a, const SpinVector& b);

// Transpose 2D matrix
inline SpinMatrix transpose2D(const SpinMatrix& M) {
    return M.transpose();
}

// Transpose 3D tensor - defined in simple_linear_alg.cpp
SpinTensor3 transpose3D(const SpinTensor3& T, size_t N1, size_t N2, size_t N3);

// Random number generation using Lehman generator - defined in simple_linear_alg.cpp
extern unsigned __int128 lehman_state;

void seed_lehman(unsigned __int128 seed);
uint64_t lehman_next();
double random_double_lehman(double min, double max);
int random_int_lehman(int size);

// Standard C++ random number generators
inline double random_double(double min, double max, std::mt19937& gen) {
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

inline int random_int(int min, int max, std::mt19937& gen) {
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

// Utility functions
template<typename T>
std::vector<T> logspace(T start, T end, int num) {
    std::vector<T> result;
    if (num <= 0) return result;
    
    if (num == 1) {
        result.push_back(std::pow(10, start));
        return result;
    }
    
    T step = (end - start) / (num - 1);
    
    for (int i = 0; i < num; ++i) {
        T exponent = start + i * step;
        result.push_back(std::pow(10, exponent));
    }
    
    return result;
}

template<typename T>
T variance(const std::vector<T>& data) {
    T mean = 0;
    for (const auto& val : data) {
        mean += val;
    }
    mean /= data.size();
    
    T var = 0;
    for (const auto& val : data) {
        var += (val - mean) * (val - mean);
    }
    var /= data.size();
    
    return var;
}

// Norm functions for arrays of vectors - defined in simple_linear_alg.cpp
double norm_average(const std::vector<SpinVector>& vecs);

#endif // LIN_ALG_REFACTORED_H
