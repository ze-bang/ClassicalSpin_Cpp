// Simple test program to verify HDF5 PT writer functionality
// This is a standalone test that doesn't require full MPI run

#include <iostream>
#include <vector>
#include <ctime>
#include "classical_spin/core/simple_linear_alg.h"

#ifdef HDF5_ENABLED
#include "classical_spin/io/hdf5_io.h"

void test_hdf5_pt_writer() {
    std::cout << "Testing HDF5 Parallel Tempering Writer..." << std::endl;
    
    // Simulation parameters
    double temperature = 0.5;
    size_t lattice_size = 100;
    size_t spin_dim = 3;
    size_t n_sublattices = 2;
    size_t n_samples = 50;
    size_t n_anneal = 1000;
    size_t n_measure = 2000;
    size_t probe_rate = 10;
    size_t swap_rate = 20;
    size_t overrelaxation_rate = 5;
    double acceptance_rate = 0.65;
    double swap_acceptance_rate = 0.48;
    
    // Generate synthetic data
    std::vector<double> energies(n_samples);
    std::vector<SpinVector> magnetizations(n_samples);
    std::vector<std::vector<SpinVector>> sublattice_mags(n_samples);
    
    for (size_t i = 0; i < n_samples; ++i) {
        // Random-ish energy
        energies[i] = -1.5 + 0.1 * std::sin(i * 0.1);
        
        // Random-ish magnetization
        SpinVector mag(spin_dim);
        for (size_t d = 0; d < spin_dim; ++d) {
            mag(d) = 0.3 * std::cos(i * 0.2 + d);
        }
        magnetizations[i] = mag;
        
        // Sublattice magnetizations
        std::vector<SpinVector> sub_mags(n_sublattices);
        for (size_t alpha = 0; alpha < n_sublattices; ++alpha) {
            SpinVector sub_mag(spin_dim);
            for (size_t d = 0; d < spin_dim; ++d) {
                sub_mag(d) = 0.2 * std::sin(i * 0.15 + alpha + d);
            }
            sub_mags[alpha] = sub_mag;
        }
        sublattice_mags[i] = sub_mags;
    }
    
    // Create HDF5 writer
    std::cout << "  Creating HDF5 file: test_pt_output.h5" << std::endl;
    HDF5PTWriter writer("test_pt_output.h5", temperature, lattice_size, spin_dim,
                       n_sublattices, n_samples, n_anneal, n_measure, probe_rate,
                       swap_rate, overrelaxation_rate, acceptance_rate,
                       swap_acceptance_rate);
    
    // Write time series
    std::cout << "  Writing time series data..." << std::endl;
    writer.write_timeseries(energies, magnetizations, sublattice_mags);
    
    // Prepare observables data
    std::vector<std::vector<double>> sublattice_mag_means(n_sublattices);
    std::vector<std::vector<double>> sublattice_mag_errors(n_sublattices);
    std::vector<std::vector<double>> energy_cross_means(n_sublattices);
    std::vector<std::vector<double>> energy_cross_errors(n_sublattices);
    
    for (size_t alpha = 0; alpha < n_sublattices; ++alpha) {
        sublattice_mag_means[alpha] = std::vector<double>(spin_dim, 0.15);
        sublattice_mag_errors[alpha] = std::vector<double>(spin_dim, 0.02);
        energy_cross_means[alpha] = std::vector<double>(spin_dim, 0.05);
        energy_cross_errors[alpha] = std::vector<double>(spin_dim, 0.01);
    }
    
    // Write observables
    std::cout << "  Writing observables data..." << std::endl;
    writer.write_observables(-1.48, 0.03, 1.25, 0.15,
                            sublattice_mag_means, sublattice_mag_errors,
                            energy_cross_means, energy_cross_errors);
    
    writer.close();
    std::cout << "  ✓ HDF5 file created successfully!" << std::endl;
    std::cout << std::endl;
    std::cout << "To inspect the file, run:" << std::endl;
    std::cout << "  h5dump -H test_pt_output.h5" << std::endl;
    std::cout << "  h5ls -r test_pt_output.h5" << std::endl;
}

void test_hdf5_aggregated_writer() {
    std::cout << "Testing HDF5 Aggregated Writer..." << std::endl;
    
    // Temperature scan data
    std::vector<double> temperatures = {0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> heat_capacity = {0.52, 1.23, 1.85, 1.45, 0.98};
    std::vector<double> dHeat = {0.05, 0.08, 0.12, 0.10, 0.07};
    
    size_t n_temps = temperatures.size();
    
    // Create HDF5 file
    std::cout << "  Creating HDF5 file: test_aggregated.h5" << std::endl;
    H5::H5File file("test_aggregated.h5", H5F_ACC_TRUNC);
    
    // Create groups
    H5::Group data_group = file.createGroup("/temperature_scan");
    H5::Group metadata_group = file.createGroup("/metadata");
    
    // Write metadata
    std::time_t now = std::time(nullptr);
    char time_str[100];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::Attribute n_temps_attr = metadata_group.createAttribute(
        "n_temperatures", H5::PredType::NATIVE_HSIZE, scalar_space);
    n_temps_attr.write(H5::PredType::NATIVE_HSIZE, &n_temps);
    
    H5::StrType str_type(H5::PredType::C_S1, strlen(time_str) + 1);
    H5::Attribute time_attr = metadata_group.createAttribute(
        "creation_time", str_type, scalar_space);
    time_attr.write(str_type, time_str);
    
    // Write datasets
    hsize_t dims[1] = {n_temps};
    H5::DataSpace dataspace(1, dims);
    
    H5::DataSet temp_dataset = data_group.createDataSet(
        "temperature", H5::PredType::NATIVE_DOUBLE, dataspace);
    temp_dataset.write(temperatures.data(), H5::PredType::NATIVE_DOUBLE);
    
    H5::DataSet heat_dataset = data_group.createDataSet(
        "specific_heat", H5::PredType::NATIVE_DOUBLE, dataspace);
    heat_dataset.write(heat_capacity.data(), H5::PredType::NATIVE_DOUBLE);
    
    H5::DataSet dheat_dataset = data_group.createDataSet(
        "specific_heat_error", H5::PredType::NATIVE_DOUBLE, dataspace);
    dheat_dataset.write(dHeat.data(), H5::PredType::NATIVE_DOUBLE);
    
    temp_dataset.close();
    heat_dataset.close();
    dheat_dataset.close();
    data_group.close();
    metadata_group.close();
    file.close();
    
    std::cout << "  ✓ Aggregated HDF5 file created successfully!" << std::endl;
}

#endif

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "HDF5 Parallel Tempering Writer Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
#ifdef HDF5_ENABLED
    std::cout << "HDF5 support: ENABLED" << std::endl;
    std::cout << std::endl;
    
    try {
        test_hdf5_pt_writer();
        std::cout << std::endl;
        test_hdf5_aggregated_writer();
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "All tests passed!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
#else
    std::cout << "HDF5 support: DISABLED" << std::endl;
    std::cout << "Please compile with -DHDF5_ENABLED to run tests" << std::endl;
    return 1;
#endif
}
