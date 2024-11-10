#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
template<size_t N>
array<float, N> operator*(const array<float, N> &a,const float n) {
    // cout << "Begin s multiplication with " << i << " ";
    array<float, N> result;
    for (int i = 0; i < 3; ++i) {
        result[i] = a[i]*float(n);
        // cout << a[i] << "x" << i << "="<< result[i] << " ";
    }
    // cout << endl;
    return result;
}

template<size_t N, size_t M>
array<array<float, M>, N> s_prod_2D(const array<array<float, M>, N> &a, const float n) {
    array<array<float, M>, N> result;
    for (int i = 0; i < N; ++i) {
        for(int j =0; j< M; ++j){
            result[i][j] = a[i][j]*float(n);
        }
    }
    return result;
}

template<size_t N>
array<float, N> operator+(const array<float, N> &a,const array<float, N>  &b) {
    array<float, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

template<size_t N>
array<float, N> operator-(const array<float, N> &a,const array<float, N>  &b) {
    array<float, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}


template<size_t N>
float dot(const array<float, N>  &a, const array<float, N>  &b) {
    float result = 0;
    for (int i = 0; i < N; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

template <size_t N>
array<float, N>  multiply(const array<array<float, N>,N>  &M, const array<float, N>  &a){
    array<float, N>  result;
    result = {0};
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i] += M[i][j] * a[j];
        }
    }
    return result;
}

template <size_t N>
array<float, N> cross_prod(const array<float, N>  &a,const array<float, N> &b){
    array<float, N> result;
    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2];
    result[2] = a[0]*b[1] - a[1]*b[0];
    return result;
}

template <size_t N, size_t M>
array<array<float, M>, N> array_add_2d(const array<array<float, M>, N> &a,const array<array<float, M>, N> &b) {
    array<array<float, M>, N> result;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

template <size_t N, size_t M>
array<array<float, M>, N> array_add_2d_mult(const array<array<float, M>, N> &a,const array<array<float, M>, N> &b,const array<array<float, M>, N> &c, const array<array<float, M>, N> &d) {
    array<array<float, M>, N> result;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            result[i][j] = a[i][j] + b[i][j] + c[i][j] + d[i][j];
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
#endif // LIN_ALG_H