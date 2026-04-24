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

// -----------------------------------------------------------------------------
// Sparse SU(3) cross product, hot-loop variant.
//
// Computes  out_i  =  sum_{j,k} f_{ijk} * H_j * S_k    (i = 0..7)
// where f_{ijk} are the Gell-Mann SU(3) structure constants. The general
// implementation in `cross_prod_SU3` walks all 8 x 8 x 8 = 512 (j,k,i)
// triples and pays a multiply-add for every single one, even though the
// structure constants are extremely sparse: there are only 9 distinct
// non-zero (a<b<c) triples (012, 036, 045, 135, 146, 234, 256, 347, 567)
// — i.e. 54 non-zero entries after antisymmetric permutations, and 458 of
// the 512 multiplies in the textbook implementation are 0 * something.
//
// This routine evaluates only the 9 non-zero triples directly, writing
// straight into a caller-supplied `double[8]` so there is no heap
// allocation in the LLG hot loop.
//
// In-place semantics:
//   * accumulate=false  →  out[0..7]  = sum  (overwrite)
//   * accumulate=true   →  out[0..7] += sum  (add to existing)
//
// Both `H` and `S` must point to 8 contiguous doubles. The kernel touches
// no other state and is fully thread-safe.
inline void cross_prod_SU3_flat(const double* __restrict H,
                                const double* __restrict S,
                                double* __restrict out,
                                bool accumulate = false) {
    // Read all 8 components of H and S once into registers; the compiler
    // turns the rest into a flat sequence of FMAs.
    const double H0 = H[0], H1 = H[1], H2 = H[2], H3 = H[3];
    const double H4 = H[4], H5 = H[5], H6 = H[6], H7 = H[7];
    const double S0 = S[0], S1 = S[1], S2 = S[2], S3 = S[3];
    const double S4 = S[4], S5 = S[5], S6 = S[6], S7 = S[7];

    // For each non-zero triple (a,b,c) with f_{abc} = v, antisymmetry of
    // f gives:
    //   d_a += v * (H_b * S_c - H_c * S_b)
    //   d_b += v * (H_c * S_a - H_a * S_c)
    //   d_c += v * (H_a * S_b - H_b * S_a)
    // (this is the i=a, i=b, i=c slices of  d_i = sum_{jk} f_{ijk} H_j S_k.)
    //
    // We pre-fold the value v into the temporaries so the compiler sees
    // pure FMA chains.

    // Triple (0,1,2), f = 1
    const double t012_a = (H1*S2 - H2*S1);
    const double t012_b = (H2*S0 - H0*S2);
    const double t012_c = (H0*S1 - H1*S0);

    // Triple (0,3,6), f = 1/2
    const double t036_a = 0.5 * (H3*S6 - H6*S3);
    const double t036_b = 0.5 * (H6*S0 - H0*S6);
    const double t036_c = 0.5 * (H0*S3 - H3*S0);

    // Triple (0,4,5), f = -1/2
    const double t045_a = -0.5 * (H4*S5 - H5*S4);
    const double t045_b = -0.5 * (H5*S0 - H0*S5);
    const double t045_c = -0.5 * (H0*S4 - H4*S0);

    // Triple (1,3,5), f = 1/2
    const double t135_a = 0.5 * (H3*S5 - H5*S3);
    const double t135_b = 0.5 * (H5*S1 - H1*S5);
    const double t135_c = 0.5 * (H1*S3 - H3*S1);

    // Triple (1,4,6), f = 1/2
    const double t146_a = 0.5 * (H4*S6 - H6*S4);
    const double t146_b = 0.5 * (H6*S1 - H1*S6);
    const double t146_c = 0.5 * (H1*S4 - H4*S1);

    // Triple (2,3,4), f = 1/2
    const double t234_a = 0.5 * (H3*S4 - H4*S3);
    const double t234_b = 0.5 * (H4*S2 - H2*S4);
    const double t234_c = 0.5 * (H2*S3 - H3*S2);

    // Triple (2,5,6), f = -1/2
    const double t256_a = -0.5 * (H5*S6 - H6*S5);
    const double t256_b = -0.5 * (H6*S2 - H2*S6);
    const double t256_c = -0.5 * (H2*S5 - H5*S2);

    // Triple (3,4,7), f = sqrt(3)/2
    static constexpr double SQRT3_OVER_2 = 0.8660254037844386;
    const double t347_a = SQRT3_OVER_2 * (H4*S7 - H7*S4);
    const double t347_b = SQRT3_OVER_2 * (H7*S3 - H3*S7);
    const double t347_c = SQRT3_OVER_2 * (H3*S4 - H4*S3);

    // Triple (5,6,7), f = sqrt(3)/2
    const double t567_a = SQRT3_OVER_2 * (H6*S7 - H7*S6);
    const double t567_b = SQRT3_OVER_2 * (H7*S5 - H5*S7);
    const double t567_c = SQRT3_OVER_2 * (H5*S6 - H6*S5);

    // Index 0 receives contributions from (012, 036, 045) — at the 'a' slot
    const double d0 = t012_a + t036_a + t045_a;
    // Index 1 receives from (012)_b + (135)_a + (146)_a
    const double d1 = t012_b + t135_a + t146_a;
    // Index 2 receives from (012)_c + (234)_a + (256)_a
    const double d2 = t012_c + t234_a + t256_a;
    // Index 3 receives from (036)_b + (135)_b + (234)_b + (347)_a
    const double d3 = t036_b + t135_b + t234_b + t347_a;
    // Index 4 receives from (045)_b + (146)_b + (234)_c + (347)_b
    const double d4 = t045_b + t146_b + t234_c + t347_b;
    // Index 5 receives from (045)_c + (135)_c + (256)_b + (567)_a
    const double d5 = t045_c + t135_c + t256_b + t567_a;
    // Index 6 receives from (036)_c + (146)_c + (256)_c + (567)_b
    const double d6 = t036_c + t146_c + t256_c + t567_b;
    // Index 7 receives from (347)_c + (567)_c
    const double d7 = t347_c + t567_c;

    if (accumulate) {
        out[0] += d0; out[1] += d1; out[2] += d2; out[3] += d3;
        out[4] += d4; out[5] += d5; out[6] += d6; out[7] += d7;
    } else {
        out[0] = d0;  out[1] = d1;  out[2] = d2;  out[3] = d3;
        out[4] = d4;  out[5] = d5;  out[6] = d6;  out[7] = d7;
    }
}

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
