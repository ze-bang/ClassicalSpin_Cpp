#ifndef FILE_IO_H
#define FILE_IO_H

#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

using std::ofstream;
using std::ios;
using std::string;
using std::vector;
using std::array;
using std::min;

// Generic file-writing utilities to reduce duplication across simulation code

// Write a spin configuration (2D array) to file
// Each row represents one site's spin components, space-separated
template<size_t N, typename SpinConfig>
void write_spin_config_to_file(const string& filename, const SpinConfig& spins, size_t lattice_size, bool append = false) {
    ofstream myfile(filename, append ? ios::app : ios::out);
    if (!myfile) return;
    
    size_t limit = min(lattice_size, spins.size());
    for (size_t i = 0; i < limit; ++i) {
        for (size_t j = 0; j < N; ++j) {
            myfile << spins[i][j];
            if (j + 1 < N) myfile << " ";
        }
        myfile << '\n';
    }
    myfile.close();
}

// Write a single scalar value to file
inline void write_single_value_to_file(const string& filename, double value, bool append = false) {
    ofstream myfile(filename, append ? ios::app : ios::out);
    if (!myfile) return;
    myfile << value << " \n";
    myfile.close();
}

// Write a fixed-size array (e.g., magnetization vector) to file
template<size_t N>
void write_arrayN_to_file(const string& filename, const array<double, N>& arr, bool append = true) {
    ofstream myfile(filename, append ? ios::app : ios::out);
    if (!myfile) return;
    
    for (size_t j = 0; j < N; ++j) {
        myfile << arr[j];
        if (j + 1 < N) myfile << " ";
    }
    myfile << '\n';
    myfile.close();
}

// Write a vector of fixed-size arrays (e.g., time series of magnetization vectors)
template<size_t N>
void write_2d_vector_array_to_file(const string& filename, const vector<array<double, N>>& data, bool append = true) {
    ofstream myfile(filename, append ? ios::app : ios::out);
    if (!myfile) return;
    
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < N; ++j) {
            myfile << data[i][j];
            if (j + 1 < N) myfile << " ";
        }
        myfile << '\n';
    }
    myfile.close();
}

// Write a column vector (one value per line)
inline void write_column_vector_to_file(const string& filename, const vector<double>& data, bool append = false) {
    ofstream myfile(filename, append ? ios::app : ios::out);
    if (!myfile) return;
    
    for (size_t i = 0; i < data.size(); ++i) {
        myfile << data[i] << '\n';
    }
    myfile.close();
}

#endif // FILE_IO_H
