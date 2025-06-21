#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <omp.h>
#include <array>

using namespace std;

template<size_t N>
void set_permutation(array<array<array<double, N>, N>,N> &A, const size_t a, const size_t b, const size_t c, double val){
    if (a >= N || b >= N || c >= N) {
        throw out_of_range("Index out of range in set_permutation");
    }
    if (a == b || b == c || a == c) {
        throw invalid_argument("Indices must be distinct in set_permutation");
    }
    // Set the structure constant for the permutation
    A[a][b][c] = val;
    A[a][c][b] = -val;
    A[b][a][c] = -val;
    A[b][c][a] = val;
    A[c][a][b] = val;
    A[c][b][a] = -val;
}

inline const array<array<array<double, 3>, 3>,3> SU2_structure_constant(){
    array<array<array<double, 3>, 3>,3> result;
    result = {{{{{0}}}}};
    set_permutation(result, 0, 1, 2, 1);
    return result;
}

inline const array<array<array<double, 8>, 8>,8> SU3_structure_constant(){
    array<array<array<double, 8>, 8>,8> result;
    result = {{{{{0}}}}};
    set_permutation(result, 0, 1, 2, 1);
    set_permutation(result, 0, 3, 6, 0.5);
    set_permutation(result, 0, 4, 5, -0.5);
    set_permutation(result, 1, 3, 5, 0.5);
    set_permutation(result, 1, 4, 6, 0.5);
    set_permutation(result, 2, 3, 4, 0.5);
    set_permutation(result, 2, 5, 6, -0.5);
    set_permutation(result, 3, 4, 7, sqrt(3)/2);
    set_permutation(result, 5, 6, 7, sqrt(3)/2);
    return result;
}


// Due to the number of different basis SU(3) can take, the structure constant is not always the same
// Even though the Cartan algebra has Gell-mann matrices as the standard basis for SU(3),
// typical systems obtain their SU(3) nature by promoting SU(2) spins to SU(3) spins via the construction of 
// some quadropolar operators such as Q_{ij} = S_i S_j + S_j S_i - 2/3 \delta_{ij} where S is the spin of the SU(2) system
// There is also further promotion by constructing another set of basis according to 
// N. Papanicolaou, Unusual phases in quantum spin-1 systems, Nucl. Phys. B 305, 367 (1988).
// the basis for this is (A^{xx}, A^{xy}, A^{xz}, A^{yx}, A^{yy}, A^{yz}, A^{zx}, A^{zy}, A^{zz})

inline const array<array<array<double, 8>, 8>,8> SU3_structure = SU3_structure_constant();
inline const array<array<array<double, 3>, 3>,3> SU2_structure = SU2_structure_constant();


template<typename T, typename T1, size_t N>
array<T, N> operator*(const array<T, N> &a, const T1 n) {
    array<T, N> result;
    const T converted_n = T(n);  // Convert type once outside the loop
    
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] * converted_n;
    }
    
    return result;
}

template<typename T, size_t N>
const array<array<T, N>,N> operator*(const array<array<T, N>,N> &a, const array<array<T, N>,N> &b) {
    array<array<T, N>,N> result = {};  // Zero-initialize
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < N; ++i) {
        // Process row i of the result
        for (size_t k = 0; k < N; ++k) {
            const T a_ik = a[i][k];  // Cache this value
            
            #pragma omp simd
            for (size_t j = 0; j < N; ++j) {
                result[i][j] += a_ik * b[k][j];
            }
        }
    }
    
    return result;
}


template<typename T, typename T1, size_t N>
array<T, N> operator*= (array<T, N> &a, const T1 n) {
    const T converted_n = T(n);  // Convert type once outside the loop

    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        a[i] *= converted_n;
    }
    return a;
}


template<typename T, typename T1, size_t N>
array<T, N> operator/(const array<T, N> &a, const T1 n) {
    // Check for division by zero to ensure data integrity
    if (n == T1(0)) {
        throw std::invalid_argument("Division by zero in array division");
    }
    
    array<T, N> result;
    const T converted_n = T(n);  // Convert type once outside the loop
    
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] / converted_n;
    }
    
    return result;
}

template<typename T, size_t N>
array<T, N> operator+(const array<T, N> &a, const array<T, N> &b) {
    array<T, N> result;
    
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] + b[i];
    }
    
    return result;
}

template<typename T, size_t N>
array<T, N> operator+=(array<T, N> &a, const array<T, N>  &b) {
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        a[i] += b[i];
    }
    return a;
}

template<typename T, size_t N>
array<T, N> operator-=(array<T, N> &a,const array<T, N>  &b) {
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        a[i] -= b[i];
    }
    return a;
}

template<typename T, size_t N>
array<T, N> operator-(const array<T, N> &a,const array<T, N>  &b) {
    array<T, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}


template<typename T, size_t N>
T dot(const array<T, N>  &a, const array<T, N>  &b) {
    T result = 0;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result += a[i] * b[i];
    }
    return result;
}


inline array<double, 3> multiply_SU2(const array<double, 9> &M, const array<double, 3> &a) {
    // Complete loop unrolling for this small fixed-size operation
    // This eliminates loop overhead and branch prediction misses
    return {
        M[0] * a[0] + M[1] * a[1] + M[2] * a[2],
        M[3] * a[0] + M[4] * a[1] + M[5] * a[2],
        M[6] * a[0] + M[7] * a[1] + M[8] * a[2]
    };
}

inline array<double, 8> multiply_SU3(const array<double, 64> &M, const array<double, 8> &a) {
    array<double, 8> result;
    
    // Cache a values to improve memory access patterns and reduce reloads
    const double a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    const double a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    
    #pragma omp simd
    for (int i = 0; i < 8; i++) {
        const int base = i * 8;
        result[i] = M[base]   * a0 + 
                   M[base+1] * a1 + 
                   M[base+2] * a2 + 
                   M[base+3] * a3 + 
                   M[base+4] * a4 + 
                   M[base+5] * a5 + 
                   M[base+6] * a6 + 
                   M[base+7] * a7;
    }
    
    return result;
}


inline double contract_SU2(const array<double, 3> &a, const array<double, 9> &M, const array<double, 3> &b) {
    double result = 0.0;
    
    // Compute directly without temporary array for better cache efficiency
    #pragma omp simd reduction(+:result)
    for (int i = 0; i < 3; i++) {
        // Calculate M*b for this component and multiply by a[i] in one step
        result += a[i] * (M[i*3] * b[0] + M[i*3 + 1] * b[1] + M[i*3 + 2] * b[2]);
    }
    
    return result;
}

inline double contract_SU3(const array<double, 8> &a, const array<double, 64> &M, const array<double, 8> &b) {
    double result = 0.0;
    
    // Compute directly without temporary array for better cache efficiency
    #pragma omp simd reduction(+:result)
    for (int i = 0; i < 8; i++) {
        const int base = i * 8;
        // Calculate M*b for this component and multiply by a[i] in one step
        result += a[i] * (M[base]   * b[0] + 
                          M[base+1] * b[1] + 
                          M[base+2] * b[2] + 
                          M[base+3] * b[3] + 
                          M[base+4] * b[4] + 
                          M[base+5] * b[5] + 
                          M[base+6] * b[6] + 
                          M[base+7] * b[7]);
    }
    
    return result;
}

template<size_t N>
inline double contract(const array<double, N>  &a, const array<double, N*N>  &M, const array<double, N>  &b) {
    double result = 0;
    if constexpr (N == 3){
        result = contract_SU2(a, M, b);
    }
    else if constexpr (N == 8){
        result = contract_SU3(a, M, b);
    }
    return result;
}


template<size_t N_1, size_t N_2, size_t N_3>
inline double contract_trilinear(const array<double, N_1*N_2*N_3> &M, const array<double, N_1> &a, const array<double, N_2> &b, const array<double, N_3> &c) {
    double result = 0.0;
    
    #pragma omp parallel reduction(+:result)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < N_1; ++i) {
            const double a_i = a[i];
            
            for (size_t j = 0; j < N_2; ++j) {
                const double a_i_b_j = a_i * b[j];
                const size_t base_idx = i*N_2*N_3 + j*N_3;
                
                double k_sum = 0.0;
                #pragma omp simd reduction(+:k_sum)
                for (size_t k = 0; k < N_3; ++k) {
                    k_sum += M[base_idx + k] * c[k];
                }
                
                result += a_i_b_j * k_sum;
            }
        }
    }
    
    return result;
}


template<size_t N_1, size_t N_2, size_t N_3>
inline array<double, N_1> contract_trilinear_field(const array<double, N_1*N_2*N_3> &M, const array<double, N_2> &b, const array<double, N_3> &c) {
    array<double, N_1> result = {0};
    
    #pragma omp parallel for schedule(dynamic, 1)
    for(size_t i = 0; i < N_1; i++) {
        const size_t i_base = i * N_2 * N_3;
        double sum_i = 0.0;
        
        for(size_t j = 0; j < N_2; j++) {
            const double b_j = b[j];
            const size_t base_idx = i_base + j * N_3;
            
            double k_sum = 0.0;
            #pragma omp simd reduction(+:k_sum)
            for(size_t k = 0; k < N_3; k++) {
                k_sum += M[base_idx + k] * c[k];
            }
            
            sum_i += b_j * k_sum;
        }
        result[i] = sum_i;
    }
    
    return result;
}


template <size_t N>
inline array<double, N>  multiply(const array<double, N*N>  &M, const array<double, N>  &a){
    array<double, N>  result;
    if constexpr (N == 3){
        result = multiply_SU2(M, a);
    }else if constexpr (N == 8){
        result = multiply_SU3(M, a);
    }
    return result;
}


inline array<double, 3> cross_prod_SU2(const array<double, 3> &a, const array<double, 3> &b) {
    array<double, 3> result;
    
    #pragma omp simd
    for (int i = 0; i < 3; ++i) {
        const int j = (i + 1) % 3;
        const int k = (i + 2) % 3;
        result[i] = a[j] * b[k] - a[k] * b[j];
    }
    
    return result;
}



inline array<double, 8> cross_prod_SU3(const array<double, 8> &a, const array<double, 8> &b) {
    array<double, 8> result = {0};
    
    // Cache the structure constants reference for better performance
    const auto &f = SU3_structure;
    
    // Static scheduling works better for uniformly distributed workloads
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < 8; ++i) {
        double sum_i = 0.0;
        
        for(size_t j = 0; j < 8; ++j) {
            const double a_j = a[j];
            
            // Use SIMD for the innermost loop to enable vectorization
            #pragma omp simd reduction(+:sum_i)
            for(size_t k = 0; k < 8; ++k) {
                sum_i += f[i][j][k] * a_j * b[k];
            }
        }
        
        // Atomic write to ensure data integrity
        result[i] = sum_i;
    }
    
    return result;
}


// array<double, 9> cross_prod_U3(const array<double, 9>  &a,const array<double, 9> &b){
//     array<double, 9> result;
//     for(size_t i=0; i<9; i++){
//         for(size_t j=0; j<9; j++){

            

//             for(size_t k=0; k <9; k++){
                
//                 result[i] += SU3_structure[i][j][k]*a[j]*b[k];
//             }
//         }
//     }

//     return result;
// }


template <size_t N, size_t M>
array<array<double, M>, N> operator+(const array<array<double, M>, N> &a, const array<array<double, M>, N> &b) {
    array<array<double, M>, N> result;
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

template <size_t N, size_t M>
array<array<double, M>, N> operator*(const array<array<double, M>, N> &a, const double &b) {
    array<array<double, M>, N> result;
    
    // Use static scheduling for more efficient workload distribution
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        // Enable vectorization with SIMD
        #pragma omp simd
        for (size_t j = 0; j < M; ++j) {
            result[i][j] = a[i][j] * b;
        }
    }
    return result;
}

template <size_t N, size_t M>
double norm_average_2D(const array<array<double, M>, N> &a) {
    double result = 0;
    
    #pragma omp parallel for collapse(2) reduction(+:result) schedule(dynamic, 1)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result += a[i][j] * a[i][j];
        }
    }
    
    return result / (N * M);
}

template <size_t N1, size_t N2, size_t M1, size_t M2>
double norm_average_2D_tuple(const tuple<array<array<double,N1>, M1>, array<array<double,N2>, M2>> &a) {
    double result = 0;
    for (size_t i = 0; i < N1; ++i) {
        for (size_t j = 0; j < M1; ++j) {
            result += get<0>(a)[i][j]*get<0>(a)[i][j];
        }
    }

    for (size_t i = 0; i < N2; ++i) {
        for (size_t j = 0; j < M2; ++j) {
            result += get<1>(a)[i][j]*get<1>(a)[i][j];
        }
    }
    return result/(N1*M1*N2*M2);
}

template <size_t N, size_t M>
array<array<double, M>, N> operator-(const array<array<double, M>, N> &a, const array<array<double, M>, N> &b) {
    array<array<double, M>, N> result;
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        #pragma omp simd
        for (size_t j = 0; j < M; ++j) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    return result;
}

static unsigned __int128 state;

inline void seed_lehman(unsigned __int128 seed)
{
	state = seed << 1 | 1;
}


inline uint64_t lehman_next(void)
{
	uint64_t result = state >> 64;
	// GCC cannot write 128-bit literals, so we use an expression
	const unsigned __int128 mult =
		(unsigned __int128)0x12e15e35b500f16e << 64 |
		0x2e714eb2b37916a5;
	state *= mult;
	return result;
}


inline double random_double_lehman(double min, double max){
    return min + (max - min) * (lehman_next()) / ((uint64_t)-1);
}

inline int random_int_lehman(int size){
    return lehman_next() % size;
}

inline double random_double(double min, double max, std::mt19937 &gen){
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

inline int random_int(int min, int max, std::mt19937 &gen){
    std::uniform_int_distribution dis(min, max);
    return dis(gen);
}

template<size_t N>
array<double, N> transpose2D(const array<double, N>& matrix) {
    array<double, N> transposed;
    int size = int(sqrt(N));
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            transposed[j*size + i] = matrix[i*size + j];
        }
    }
    return transposed;
}


template<size_t N>
array<double, N> transpose3D(const array<double, N>& matrix, size_t N_1, size_t N_2, size_t N_3) {
    array<double, N> transposed;
    #pragma omp parallel for 
    for (size_t i = 0; i < N_1; ++i) {
        for (size_t j = 0; j < N_2; ++j) {
            for (size_t k = 0; k < N_3; ++k) {
                transposed[j*N_1*N_3 + k*N_1 + i] = matrix[i*N_2*N_3 + j*N_3 + k];
            }
        }
    }
    return transposed; // Overwrite the original matrix with the transposed one
}


template<size_t N>
array<double, N> swap_axis_3D(const array<double, N>& matrix, size_t N_1, size_t N_2, size_t N_3) {
    array<double, N> transposed;
    #pragma omp parallel for collapse(3)
    for (size_t i = 0; i < N_1; ++i) {
        for (size_t j = 0; j < N_2; ++j) {
            for (size_t k = 0; k < N_3; ++k) {
                transposed[j*N_1*N_3 + i*N_3 + k] = matrix[i*N_2*N_3 + j*N_3 + k];
            }
        }
    }
    
    return transposed; // Overwrite the original matrix with the transposed one
}

template<typename T>
std::vector<T> logspace(T start, T end, int num) {
    std::vector<T> result;
    if (num <= 0) return result; // Handle invalid input

    if (num == 1) { // Special case: single value
        result.push_back(std::pow(10, start));
        return result;
    }

    // Step size for the exponent
    T step = (end - start) / (num - 1);

    // Generate the values
    for (int i = 0; i < num; ++i) {
        T exponent = start + i * step;
        result.push_back(std::pow(10, exponent));
    }

    return result;
}


template<typename T>
T variance(vector<T> &data){
    T mean = 0;
    T variance = 0;
    for (size_t i = 0; i < data.size(); ++i){
        mean += data[i];
    }
    mean /= data.size();
    for (size_t i = 0; i < data.size(); ++i){
        variance += (data[i] - mean)*(data[i] - mean);
    }
    variance /= data.size();
    return variance;
}

template<typename T, size_t N>
array<T, N> operator/= (array<T, N> &a, const T &b){
    #pragma omp simd
    for (size_t i = 0; i < N; ++i){
        a[i] /= b;
    }
    return a;
}

#endif // LIN_ALG_H