#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <omp.h>
#include <array>

using namespace std;

void set_permutation(array<array<array<double, 8>, 8>,8> &A, const size_t a, const size_t b, const size_t c, double val){
    A[a][b][c] = val;
    A[a][c][b] = -val;
    A[b][a][c] = -val;
    A[b][c][a] = val;
    A[c][a][b] = val;
    A[c][b][a] = -val;
}



const array<array<array<double, 8>, 8>,8> SU3_structure_constant(){
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

const extern array<array<array<double, 8>, 8>,8> SU3_structure = SU3_structure_constant();


// Somehow BLAS is slower, shelving for now until figured out a better implementation
// template<typename T, size_t N>
// array<double, N> operator*(const array<double, N> &a, const T n) {
//     array<double, N> result;
//     cblas_dcopy(N, a.data(), 1, result.data(), 1);
//     cblas_dscal(N, double(n), result.data(), 1);
//     return result;
// }


// template<typename T, size_t N>
// array<double, N> operator/(const array<double, N> &a, const T n) {
//     array<T, N> result;
//     cblas_dcopy(N, a.data(), 1, result.data(), 1);
//     cblas_dscal(N, 1/double(n), result.data(), 1);
//     return result;
// }

// template<size_t N>
// array<double, N> operator+(const array<double, N> &a, const array<double, N>  &b) {
//     array<double, N> result;
//     cblas_dcopy(N, a.data(), 1, result.data(), 1);
//     cblas_daxpy(N, 1.0, b.data(), 1, result.data(), 1);
//     return result;
// }

// template<size_t N>
// array<double, N> operator-(const array<double, N> &a,const array<double, N>  &b) {
//     array<double, N> result;
//     cblas_dcopy(N, a.data(), 1, result.data(), 1);
//     cblas_daxpy(N, -1.0, b.data(), 1, result.data(), 1);
//     return result;
// }


// template<size_t N>
// double dot(const array<double, N>  &a, const array<double, N>  &b) {
//     return cblas_ddot(N, a.data(), 1, b.data(), 1);
// }

// template<size_t N>
// array<double, N> multiply(const array<double, N*N>  &M, const array<double, N>  &a){
//     array<double, N>  result;
//     cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, M.data(), N, a.data(), 1, 0.0, result.data(), 1);
//     return result;
// }


// template<size_t N>
// double contract(const array<double, N>  &a, const array<double, N*N>  &M, const array<double, N>  &b) {
//     array<double, N> temp = multiply(M, b);
//     return dot(a, temp);
// }

template<typename T, typename T1, size_t N>
array<T, N> operator*(const array<T, N> &a, const T1 n) {
    array<T, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i]*T(n);
    }
    return result;
}


template<typename T, typename T1, size_t N>
array<T, N> operator/(const array<T, N> &a, const T1 n) {
    array<T, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i]/T(n);
    }
    return result;
}

template<typename T, size_t N>
array<T, N> operator+(const array<T, N> &a,const array<T, N>  &b) {
    array<T, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
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
    for (size_t i = 0; i < N; ++i) {
        result += a[i] * b[i];
    }
    return result;
}


array<double, 3> multiply_SU2(const array<double, 9>  &M, const array<double, 3>  &a){
    array<double, 3>  result;
    result[0] = M[0] * a[0] + M[1] * a[1] + M[2] * a[2];
    result[1] = M[3] * a[0] + M[4] * a[1] + M[5] * a[2];
    result[2] = M[6] * a[0] + M[7] * a[1] + M[8] * a[2];
    return result;
}

array<double, 8> multiply_SU3(const array<double, 64>  &M, const array<double, 8>  &a){
    array<double, 8>  result;
    result[0] = M[0] * a[0] + M[1] * a[1] + M[2] * a[2] + M[3] * a[3] + M[4] * a[4] + M[5] * a[5] + M[6] * a[6]+ M[7] * a[7];
    result[1] = M[8] * a[0] + M[9] * a[1] + M[10] * a[2] + M[11] * a[3] + M[12] * a[4] + M[13] * a[5] + M[14] * a[6]+ M[15] * a[7];
    result[2] = M[16] * a[0] + M[17] * a[1] + M[18] * a[2] + M[19] * a[3] + M[20] * a[4] + M[21] * a[5] + M[22] * a[6]+ M[23] * a[7];
    result[3] = M[24] * a[0] + M[25] * a[1] + M[26] * a[2] + M[27] * a[3] + M[28] * a[4] + M[29] * a[5] + M[30] * a[6]+ M[31] * a[7];
    result[4] = M[32] * a[0] + M[33] * a[1] + M[34] * a[2] + M[35] * a[3] + M[36] * a[4] + M[37] * a[5] + M[38] * a[6]+ M[39] * a[7];
    result[5] = M[40] * a[0] + M[41] * a[1] + M[42] * a[2] + M[43] * a[3] + M[44] * a[4] + M[45] * a[5] + M[46] * a[6]+ M[47] * a[7];
    result[6] = M[48] * a[0] + M[49] * a[1] + M[50] * a[2] + M[51] * a[3] + M[52] * a[4] + M[53] * a[5] + M[54] * a[6]+ M[55] * a[7];
    result[7] = M[56] * a[0] + M[57] * a[1] + M[58] * a[2] + M[59] * a[3] + M[60] * a[4] + M[61] * a[5] + M[62] * a[6]+ M[63] * a[7];
    return result;
}


double contract_SU2(const array<double, 3>  &a, const array<double, 9>  &M, const array<double, 3>  &b){
    return a[0] * (M[0] * b[0] + M[1] * b[1] + M[2] * b[2]) + a[1] * (M[3] * b[0] + M[4] * b[1] + M[5] * b[2]) + a[2] * (M[6] * b[0] + M[7] * b[1] + M[8] * b[2]);
}

double contract_SU3(const array<double, 8> &a, const array<double, 64> &M, const array<double, 8>  &b){
    return a[0] * (M[0] * b[0] + M[1] * b[1] + M[2] * b[2] + M[3] * b[3] + M[4] * b[4] + M[5] * b[5] + M[6] * b[6]+ M[7] * b[7])
    + a[1] * (M[8] * b[0] + M[9] * b[1] + M[10] * b[2] + M[11] * b[3] + M[12] * b[4] + M[13] * b[5] + M[14] * b[6]+ M[15] * b[7])
    + a[2] * (M[16] * b[0] + M[17] * b[1] + M[18] * b[2] + M[19] * b[3] + M[20] * b[4] + M[21] * b[5] + M[22] * b[6]+ M[23] * b[7])
    + a[3] * (M[24] * b[0] + M[25] * b[1] + M[26] * b[2] + M[27] * b[3] + M[28] * b[4] + M[29] * b[5] + M[30] * b[6]+ M[31] * b[7])
    + a[4] * (M[32] * b[0] + M[33] * b[1] + M[34] * b[2] + M[35] * b[3] + M[36] * b[4] + M[37] * b[5] + M[38] * b[6]+ M[39] * b[7])
    + a[5] * (M[40] * b[0] + M[41] * b[1] + M[42] * b[2] + M[43] * b[3] + M[44] * b[4] + M[45] * b[5] + M[46] * b[6]+ M[47] * b[7])
    + a[6] * (M[48] * b[0] + M[49] * b[1] + M[50] * b[2] + M[51] * b[3] + M[52] * b[4] + M[53] * b[5] + M[54] * b[6]+ M[55] * b[7])
    + a[7] * (M[56] * b[0] + M[57] * b[1] + M[58] * b[2] + M[59] * b[3] + M[60] * b[4] + M[61] * b[5] + M[62] * b[6]+ M[63] * b[7]);
}

template<size_t N>
double contract(const array<double, N>  &a, const array<double, N*N>  &M, const array<double, N>  &b) {
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
double contract_trilinear(const array<array<array<double, N_3>,N_2>, N_1>  &M, const array<double, N_1>  &a, const array<double, N_2>  &b, const array<double, N_3>  &c) {
    double result = 0;
    #pragma omp simd  
    for(size_t i = 0; i < N_1; i++){
        for(size_t j = 0; j < N_2; j++){
            for(size_t k = 0; k < N_3; k++){
                result += M[i][j][k]*a[i]*b[j]*c[k];
            }
        }
    }
    return result;
}


template<size_t N_1, size_t N_2, size_t N_3>
array<double, N_1> contract_trilinear_field(const array<array<array<double, N_3>,N_2>, N_1>  &M, const array<double, N_2>  &b, const array<double, N_3>  &c) {
    array<double, N_1>  result = {0};  
    #pragma omp simd  
    for(size_t i = 0; i < N_1; i++){
        for(size_t j = 0; j < N_2; j++){
            for(size_t k = 0; k < N_3; k++){
                result[i] += M[i][j][k]*b[j]*c[k];
            }
        }
    }
    return result;
}


template <size_t N>
array<double, N>  multiply(const array<double, N*N>  &M, const array<double, N>  &a){
    array<double, N>  result;
    if constexpr (N == 3){
        result = multiply_SU2(M, a);
    }else if constexpr (N == 8){
        result = multiply_SU3(M, a);
    }
    return result;
}



array<double, 3> cross_prod_SU2(const array<double, 3>  &a,const array<double, 3> &b){
    array<double, 3> result;
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2];
    result[2] = a[0]*b[1] - a[1]*b[0];
    return result;
}



array<double, 8> cross_prod_SU3(const array<double, 8>  &a,const array<double, 8> &b){
    array<double, 8> result;
    for(size_t i=0; i<8; i++){
        for(size_t j=0; j <8; j++){
            for(size_t k=0; k <8; k++){
                result[i] += SU3_structure[i][j][k]*a[j]*b[k];
            }
        }
    }

    return result;
}


array<double, 9> cross_prod_U3(const array<double, 9>  &a,const array<double, 9> &b){
    array<double, 9> result;
    for(size_t i=0; i<9; i++){
        for(size_t j=0; j<9; j++){

            

            for(size_t k=0; k <9; k++){
                
                result[i] += SU3_structure[i][j][k]*a[j]*b[k];
            }
        }
    }

    return result;
}


template <size_t N, size_t M>
array<array<double, M>, N> operator+(const array<array<double, M>, N> &a,const array<array<double, M>, N> &b) {
    array<array<double, M>, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

template <size_t N, size_t M>
array<array<double, M>, N> operator*(const array<array<double, M>, N> &a,const double &b) {
    array<array<double, M>, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[i][j] = a[i][j]*b;
        }
    }
    return result;
}

template <size_t N, size_t M>
double norm_average_2D(const array<array<double, M>, N> &a) {
    double result = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result += a[i][j]*a[i][j];
        }
    }
    return result/(N*M);
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
array<array<double, M>, N> operator-(const array<array<double, M>, N> &a,const array<array<double, M>, N>  &b) {
    array<array<double, M>, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    return result;
}

static unsigned __int128 state;

void seed_lehman(unsigned __int128 seed)
{
	state = seed << 1 | 1;
}


uint64_t lehman_next(void)
{
	uint64_t result = state >> 64;
	// GCC cannot write 128-bit literals, so we use an expression
	const unsigned __int128 mult =
		(unsigned __int128)0x12e15e35b500f16e << 64 |
		0x2e714eb2b37916a5;
	state *= mult;
	return result;
}


double random_double_lehman(double min, double max){
    return min + (max - min) * (lehman_next()) / ((uint64_t)-1);
}

int random_int_lehman(int size){
    return lehman_next() % size;
}

double random_double(double min, double max, std::mt19937 &gen){
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

int random_int(int min, int max, std::mt19937 &gen){
    std::uniform_int_distribution dis(min, max);
    return dis(gen);
}

template<size_t N>
array<double, N> transpose2D(const array<double, N>& matrix) {
    array<double, N> transposed;
    int size = int(sqrt(N));
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            transposed[j*size + i] = matrix[i*size + j];
        }
    }
    return transposed;
}


template<size_t N_1, size_t N_2, size_t N_3>
array<array<array<double, N_1>, N_3>, N_2> transpose3D(const array<array<array<double, N_3>, N_2>, N_1>& matrix) {
    array<array<array<double, N_1>, N_3>, N_2> transposed;
    for (size_t i = 0; i < N_1; ++i) {
        for (size_t j = 0; j < N_2; ++j) {
            for (size_t k = 0; k < N_3; ++k) {
                transposed[j][k][i] = matrix[i][j][k];
            }
        }
    }
    
    return transposed; // Overwrite the original matrix with the transposed one
}


template<size_t N_1, size_t N_2, size_t N_3>
array<array<array<double, N_3>, N_1>, N_2> swap_axis_3D(const array<array<array<double, N_3>, N_2>, N_1>& matrix) {
    array<array<array<double, N_3>, N_1>, N_2> transposed;
    for (size_t i = 0; i < N_1; ++i) {
        for (size_t j = 0; j < N_2; ++j) {
            for (size_t k = 0; k < N_3; ++k) {
                transposed[j][i][k] = matrix[i][j][k];
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
    for (size_t i = 0; i < N; ++i){
        a[i] /= b;
    }
    return a;
}

#endif // LIN_ALG_H