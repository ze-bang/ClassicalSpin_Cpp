#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <omp.h>

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


template<size_t N>
array<double, N> operator*(const array<double, N> &a,const double n) {
    array<double, N> result;
    #pragma omp simd
    for (size_t i = 0; i < 3; ++i) {
        result[i] = a[i]*n;
    }
    return result;
}


template<size_t N>
array<double, N> operator/(const array<double, N> &a,const double n) {
    array<double, N> result;
    #pragma omp simd
    for (size_t i = 0; i < 3; ++i) {
        result[i] = a[i]/n;
    }
    return result;
}


template<size_t N>
array<double, N> operator*(const array<double, N> &a,const int n) {
    array<double, N> result;
    #pragma omp simd
    for (size_t i = 0; i < 3; ++i) {
        result[i] = a[i]*double(n);
    }
    return result;
}

template<size_t N>
array<double, N> operator+(const array<double, N> &a,const array<double, N>  &b) {
    array<double, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

template<size_t N>
array<double, N> operator-(const array<double, N> &a,const array<double, N>  &b) {
    array<double, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}


template<size_t N>
double dot(const array<double, N>  &a, const array<double, N>  &b) {
    double result = 0;
    for (size_t i = 0; i < N; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

array<double, 3> multiply_SU2(const array<array<double, 3>,3>  &M, const array<double, 3>  &a){
    array<double, 3>  result;
    result[0] = M[0][0] * a[0] + M[0][1] * a[1] + M[0][2] * a[2];
    result[1] = M[1][0] * a[0] + M[1][1] * a[1] + M[1][2] * a[2];
    result[2] = M[2][0] * a[0] + M[2][1] * a[1] + M[2][2] * a[2];
    return result;
}

array<double, 8> multiply_SU3(const array<array<double, 8>,8>  &M, const array<double, 8>  &a){
    array<double, 8>  result;
    result[0] = M[0][0] * a[0] + M[0][1] * a[1] + M[0][2] * a[2];
    result[1] = M[1][0] * a[0] + M[1][1] * a[1] + M[1][2] * a[2];
    result[2] = M[2][0] * a[0] + M[2][1] * a[1] + M[2][2] * a[2];
    result[3] = M[3][0] * a[0] + M[3][1] * a[1] + M[3][2] * a[2];
    result[4] = M[4][0] * a[0] + M[4][1] * a[1] + M[4][2] * a[2];
    result[5] = M[5][0] * a[0] + M[5][1] * a[1] + M[5][2] * a[2];
    result[6] = M[6][0] * a[0] + M[6][1] * a[1] + M[6][2] * a[2];
    result[7] = M[7][0] * a[0] + M[7][1] * a[1] + M[7][2] * a[2];
    return result;
}


double contract_SU2(const array<double, 3>  &a, const array<array<double, 3>,3>  &M, const array<double, 3>  &b){
    return a[0] * (M[0][0] * b[0] + M[0][1] * b[1] + M[0][2] * b[2]) + a[1] * (M[1][0] * b[0] + M[1][1] * b[1] + M[1][2] * b[2]) + a[2] * (M[2][0] * b[0] + M[2][1] * b[1] + M[2][2] * b[2]);
}

double contract_SU3(const array<double, 8> &a, const array<array<double, 8>, 8> &M, const array<double, 8>  &b){
    return a[0] * (M[0][0] * b[0] + M[0][1] * b[1] + M[0][2] * b[2] + M[0][3] * b[3] + M[0][4] * b[4] + M[0][5] * b[5] + M[0][6] * b[6]+ M[0][7] * b[7]) 
    + a[1] * (M[1][0] * b[0] + M[1][1] * b[1] + M[1][2] * b[2] + M[1][3] * b[3] + M[1][4] * b[4] + M[1][5] * b[5] + M[1][6] * b[6]+ M[1][7] * b[7])
    + a[2] * (M[2][0] * b[0] + M[2][1] * b[1] + M[2][2] * b[2] + M[2][3] * b[3] + M[2][4] * b[4] + M[2][5] * b[5] + M[2][6] * b[6]+ M[2][7] * b[7])
    + a[3] * (M[3][0] * b[0] + M[3][1] * b[1] + M[3][2] * b[2] + M[3][3] * b[3] + M[3][4] * b[4] + M[3][5] * b[5] + M[3][6] * b[6]+ M[3][7] * b[7])
    + a[4] * (M[4][0] * b[0] + M[4][1] * b[1] + M[4][2] * b[2] + M[4][3] * b[3] + M[4][4] * b[4] + M[4][5] * b[5] + M[4][6] * b[6]+ M[4][7] * b[7])
    + a[5] * (M[5][0] * b[0] + M[5][1] * b[1] + M[5][2] * b[2] + M[5][3] * b[3] + M[5][4] * b[4] + M[5][5] * b[5] + M[5][6] * b[6]+ M[5][7] * b[7])
    + a[6] * (M[6][0] * b[0] + M[6][1] * b[1] + M[6][2] * b[2] + M[6][3] * b[3] + M[6][4] * b[4] + M[6][5] * b[5] + M[6][6] * b[6]+ M[6][7] * b[7])
    + a[7] * (M[7][0] * b[0] + M[7][1] * b[1] + M[7][2] * b[2] + M[7][3] * b[3] + M[7][4] * b[4] + M[7][5] * b[5] + M[7][6] * b[6]+ M[7][7] * b[7]);
}

template<size_t N>
double contract(const array<double, N>  &a, const array<array<double, N>,N>  &M, const array<double, N>  &b) {
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
array<double, N>  multiply(const array<array<double, N>,N>  &M, const array<double, N>  &a){
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

double random_double(double min, double max, std::mt19937 &gen){
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

int random_int(int min, int max, std::mt19937 &gen){
    std::uniform_int_distribution dis(min, max);
    return dis(gen);
}
template<size_t N, size_t M>
array<array<double, M>, N> transpose2D(const array<array<double, M>, N>& matrix) {
    array<array<double, N>, M> transposed;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            transposed[j][i] = matrix[i][j];
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




#endif // LIN_ALG_H