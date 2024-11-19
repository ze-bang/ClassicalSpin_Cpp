#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <omp.h>

template<size_t N>
array<float, N> operator*(const array<float, N> &a,const float n) {
    array<float, N> result;
    #pragma omp simd
    for (size_t i = 0; i < 3; ++i) {
        result[i] = a[i]*n;
    }
    return result;
}

template<size_t N>
array<float, N> operator*(const array<float, N> &a,const int n) {
    array<float, N> result;
    #pragma omp simd
    for (size_t i = 0; i < 3; ++i) {
        result[i] = a[i]*float(n);
    }
    return result;
}

template<size_t N>
array<float, N> operator+(const array<float, N> &a,const array<float, N>  &b) {
    array<float, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

template<size_t N>
array<float, N> operator-(const array<float, N> &a,const array<float, N>  &b) {
    array<float, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}


template<size_t N>
float dot(const array<float, N>  &a, const array<float, N>  &b) {
    float result = 0;
    for (size_t i = 0; i < N; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

array<float, 3> multiply_SU2(const array<array<float, 3>,3>  &M, const array<float, 3>  &a){
    array<float, 3>  result;
    result[0] = M[0][0] * a[0] + M[0][1] * a[1] + M[0][2] * a[2];
    result[1] = M[1][0] * a[0] + M[1][1] * a[1] + M[1][2] * a[2];
    result[2] = M[2][0] * a[0] + M[2][1] * a[1] + M[2][2] * a[2];
    return result;
}

array<float, 8> multiply_SU3(const array<array<float, 8>,8>  &M, const array<float, 8>  &a){
    array<float, 8>  result;
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


float contract_SU2(const array<float, 3>  &a, const array<array<float, 3>,3>  &M, const array<float, 3>  &b){
    return a[0] * (M[0][0] * b[0] + M[0][1] * b[1] + M[0][2] * b[2]) + a[1] * (M[1][0] * b[0] + M[1][1] * b[1] + M[1][2] * b[2]) + a[2] * (M[2][0] * b[0] + M[2][1] * b[1] + M[2][2] * b[2]);
}

float contract_SU3(const array<float, 8> &a, const array<array<float, 8>, 8> &M, const array<float, 8>  &b){
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
float contract(const array<float, N>  &a, const array<array<float, N>,N>  &M, const array<float, N>  &b) {
    float result = 0;
    if constexpr (N == 3){
        result = contract_SU2(a, M, b);
    }
    else if constexpr (N == 8){
        result = contract_SU3(a, M, b);
    }
    return result;
}


template<size_t N_1, size_t N_2, size_t N_3>
float contract_trilinear(const array<array<array<float, N_3>,N_2>, N_1>  &M, const array<float, N_1>  &a, const array<float, N_2>  &b, const array<float, N_3>  &c) {
    float result = 0;
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
array<float, N_1> contract_trilinear_field(const array<array<array<float, N_3>,N_2>, N_1>  &M, const array<float, N_2>  &b, const array<float, N_3>  &c) {
    array<float, N_1>  result = {0};    
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
array<float, N>  multiply(const array<array<float, N>,N>  &M, const array<float, N>  &a){
    array<float, N>  result;
    if constexpr (N == 3){
        result = multiply_SU2(M, a);
    }else if constexpr (N == 8){
        result = multiply_SU3(M, a);
    }
    return result;
}



template <size_t N>
array<float, N> cross_prod_SU2(const array<float, N>  &a,const array<float, N> &b){
    array<float, N> result;
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2];
    result[2] = a[0]*b[1] - a[1]*b[0];
    return result;
}

template <size_t N, size_t M>
array<array<float, M>, N> operator+(const array<array<float, M>, N> &a,const array<array<float, M>, N> &b) {
    array<array<float, M>, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

template <size_t N, size_t M>
array<array<float, M>, N> operator*(const array<array<float, M>, N> &a,const float &b) {
    array<array<float, M>, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[i][j] = a[i][j]*b;
        }
    }
    return result;
}

template <size_t N, size_t M>
float norm_average_2D(const array<array<float, M>, N> &a) {
    float result = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result += a[i][j]*a[i][j];
        }
    }
    return result/(N*M);
}

template <size_t N, size_t M>
array<array<float, M>, N> operator-(const array<array<float, M>, N> &a,const array<array<float, M>, N>  &b) {
    array<array<float, M>, N> result;
    #pragma omp simd
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < M; ++j) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    return result;
}

float random_float(float min, float max, std::mt19937 &gen){
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

int random_int(int min, int max, std::mt19937 &gen){
    std::uniform_int_distribution dis(min, max);
    return dis(gen);
}


template<size_t N_1, size_t N_2, size_t N_3>
array<array<array<float, N_1>, N_3>, N_2> transpose3D(const array<array<array<float, N_3>, N_2>, N_1>& matrix) {
    array<array<array<float, N_1>, N_3>, N_2> transposed;
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
array<array<array<float, N_3>, N_1>, N_2> swap_axis_3D(const array<array<array<float, N_3>, N_2>, N_1>& matrix) {
    array<array<array<float, N_3>, N_1>, N_2> transposed;
    for (size_t i = 0; i < N_1; ++i) {
        for (size_t j = 0; j < N_2; ++j) {
            for (size_t k = 0; k < N_3; ++k) {
                transposed[j][i][k] = matrix[i][j][k];
            }
        }
    }
    
    return transposed; // Overwrite the original matrix with the transposed one
}





#endif // LIN_ALG_H