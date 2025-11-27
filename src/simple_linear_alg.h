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
    static SpinTensor3 SU2_structure_constant() {
        SpinTensor3 result(3, MatrixXd::Zero(3, 3));
        
        // Set the antisymmetric structure constants for SU(2)
        // f^{ijk} = epsilon^{ijk}
        set_permutation(result, 0, 1, 2, 1.0);
        
        return result;
    }
    
    static SpinTensor3 SU3_structure_constant() {
        SpinTensor3 result(8, MatrixXd::Zero(8, 8));
        
        // Set the Gell-Mann structure constants
        set_permutation(result, 0, 1, 2, 1.0);
        set_permutation(result, 0, 3, 6, 0.5);
        set_permutation(result, 0, 4, 5, -0.5);
        set_permutation(result, 1, 3, 5, 0.5);
        set_permutation(result, 1, 4, 6, 0.5);
        set_permutation(result, 2, 3, 4, 0.5);
        set_permutation(result, 2, 5, 6, -0.5);
        set_permutation(result, 3, 4, 7, sqrt(3.0)/2.0);
        set_permutation(result, 5, 6, 7, sqrt(3.0)/2.0);
        
        return result;
    }
    
private:
    static void set_permutation(SpinTensor3& A, size_t a, size_t b, size_t c, double val) {
        if (a >= A.size() || b >= A[a].cols() || c >= A[a].rows()) {
            throw out_of_range("Index out of range in set_permutation");
        }
        if (a == b || b == c || a == c) {
            throw invalid_argument("Indices must be distinct in set_permutation");
        }
        
        // Set all permutations with appropriate signs
        A[a](b, c) = val;
        A[a](c, b) = -val;
        A[b](a, c) = -val;
        A[b](c, a) = val;
        A[c](a, b) = val;
        A[c](b, a) = -val;
    }
};

// Global structure constants (lazy initialization)
inline const SpinTensor3& get_SU2_structure() {
    static const SpinTensor3 SU2_structure = StructureConstants::SU2_structure_constant();
    return SU2_structure;
}

inline const SpinTensor3& get_SU3_structure() {
    static const SpinTensor3 SU3_structure = StructureConstants::SU3_structure_constant();
    return SU3_structure;
}

// Matrix-vector multiplication using Eigen
inline SpinVector multiply(const SpinMatrix& M, const SpinVector& v) {
    return M * v;
}

// Bilinear contraction: a^T * M * b
inline double contract(const SpinVector& a, const SpinMatrix& M, const SpinVector& b) {
    return a.dot(M * b);
}

// Trilinear contraction: sum_ijk M_ijk * a_i * b_j * c_k
inline double contract_trilinear(const SpinTensor3& M, const SpinVector& a, 
                                 const SpinVector& b, const SpinVector& c) {
    double result = 0.0;
    size_t N1 = a.size();
    size_t N2 = b.size();
    size_t N3 = c.size();
    
    for (size_t i = 0; i < N1; ++i) {
        for (size_t j = 0; j < N2; ++j) {
            result += a(i) * b(j) * M[i].row(j).dot(c);
        }
    }
    
    return result;
}

// Trilinear field contraction: returns vector from trilinear * two vectors
inline SpinVector contract_trilinear_field(const SpinTensor3& M, 
                                          const SpinVector& b, 
                                          const SpinVector& c) {
    size_t N1 = M.size();
    SpinVector result = SpinVector::Zero(N1);
    
    for (size_t i = 0; i < N1; ++i) {
        result(i) = (M[i] * c).dot(b);
    }
    
    return result;
}

// Cross product for SU(2) (standard 3D cross product)
inline SpinVector cross_prod_SU2(const SpinVector& a, const SpinVector& b) {
    if (a.size() != 3 || b.size() != 3) {
        throw std::invalid_argument("SU2 cross product requires 3D vectors");
    }
    
    SpinVector result(3);
    result(0) = a(1) * b(2) - a(2) * b(1);
    result(1) = a(2) * b(0) - a(0) * b(2);
    result(2) = a(0) * b(1) - a(1) * b(0);
    
    return result;
}

// Cross product for SU(3) using structure constants
inline SpinVector cross_prod_SU3(const SpinVector& a, const SpinVector& b) {
    if (a.size() != 8 || b.size() != 8) {
        throw std::invalid_argument("SU3 cross product requires 8D vectors");
    }
    
    const auto& f = get_SU3_structure();
    SpinVector result = SpinVector::Zero(8);
    
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            result(i) += (f[i].row(j).array() * b.array()).matrix().dot(a);
        }
    }
    
    return result;
}

// General cross product that dispatches based on dimension
inline SpinVector cross_product(const SpinVector& a, const SpinVector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have same dimension for cross product");
    }
    
    if (a.size() == 3) {
        return cross_prod_SU2(a, b);
    } else if (a.size() == 8) {
        return cross_prod_SU3(a, b);
    } else {
        throw std::invalid_argument("Cross product only defined for N=3 (SU2) or N=8 (SU3)");
    }
}

// Transpose 2D matrix
inline SpinMatrix transpose2D(const SpinMatrix& M) {
    return M.transpose();
}

// Transpose 3D tensor (rearrange indices)
inline SpinTensor3 transpose3D(const SpinTensor3& T, size_t N1, size_t N2, size_t N3) {
    SpinTensor3 result(N2, MatrixXd::Zero(N3, N1));
    
    for (size_t i = 0; i < N1; ++i) {
        for (size_t j = 0; j < N2; ++j) {
            for (size_t k = 0; k < N3; ++k) {
                result[j](k, i) = T[i](j, k);
            }
        }
    }
    
    return result;
}

// Random number generation using Lehman generator (from original code)
static unsigned __int128 lehman_state;

inline void seed_lehman(unsigned __int128 seed) {
    lehman_state = seed << 1 | 1;
}

inline uint64_t lehman_next() {
    uint64_t result = lehman_state >> 64;
    const unsigned __int128 mult =
        (unsigned __int128)0x12e15e35b500f16e << 64 |
        0x2e714eb2b37916a5;
    lehman_state *= mult;
    return result;
}

inline double random_double_lehman(double min, double max) {
    return min + (max - min) * lehman_next() / ((uint64_t)-1);
}

inline int random_int_lehman(int size) {
    return lehman_next() % size;
}

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

// Norm functions for arrays of vectors
inline double norm_average(const std::vector<SpinVector>& vecs) {
    double total = 0.0;
    size_t count = 0;
    
    for (const auto& v : vecs) {
        total += v.squaredNorm();
        count += v.size();
    }
    
    return total / count;
}

#endif // LIN_ALG_REFACTORED_H
