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

// Random number generation using Lehman generator - defined in simple_linear_alg.cpp.
//
// Thread safety:
//   `lehman_state` is declared `thread_local`, so each OpenMP / std::thread
//   thread has its own independent state. Calling `seed_lehman(s)` from any
//   thread updates a shared master seed and also reseeds the calling thread.
//   The first time a thread (other than the one that called `seed_lehman`)
//   invokes `lehman_next` / `random_*_lehman`, its state is lazily derived
//   from (master_seed, thread_id) via a splitmix64 mix, producing a distinct
//   stream per thread. This prevents the pre-existing data race where the
//   OpenMP PT temperature optimizer in `lattice.h` (see ~5200-5286) was
//   calling the generator from parallel regions against a single global
//   state.
extern thread_local unsigned __int128 lehman_state;

void seed_lehman(unsigned __int128 seed);
uint64_t lehman_next();
double random_double_lehman(double min, double max);
int random_int_lehman(int size);

// splitmix64 finalizer — used everywhere we need to derive a deterministic
// stream from (master_seed, key) with good avalanche properties.
inline unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Deterministically reseed the calling thread from the current master seed +
// `key` (typically MPI rank, replica index, or another stream identifier).
//
// This is the **reproducible** alternative to seeding from
// `std::chrono::system_clock::now()` inside parallel-tempering / temperature-
// grid optimizers. Set the master seed once via `seed_lehman(master)` (or
// leave it at its default), then call `seed_lehman_from_rank(rank)` from each
// rank/thread before performing MC work.
//
// The same call also publishes a derived seed as the new master so that any
// lazily-spawned worker thread inheriting `lehman_state == 0` will pick up a
// distinct stream consistent with this rank.
void seed_lehman_from_rank(unsigned long long key);

// Derive a 64-bit seed from the current master seed + key without modifying
// any thread-local or master state. Useful for seeding e.g. a separate
// `std::mt19937` exchange-decision RNG with deterministic, rank-distinct
// values.
unsigned long long derive_seed_from_master(unsigned long long key);

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
