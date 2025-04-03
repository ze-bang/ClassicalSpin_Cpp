#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include "experiments.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <data_file> <output_dir> [field_strength] [weight_heat]" << std::endl;
        std::cout << "  data_file: File containing temperature and specific heat data (two columns)" << std::endl;
        std::cout << "  output_dir: Directory to save fitting results" << std::endl;
        std::cout << "  field_strength: Optional magnetic field strength (default: 0.0)" << std::endl;
        std::cout << "  weight_heat: Optional weight for specific heat in fitting (default: 1.0)" << std::endl;
        return 1;
    }
    
    std::string data_file = argv[1];
    std::string output_dir = argv[2];
    double field_strength = (argc > 3) ? std::stod(argv[3]) : 0.0;
    double weight_heat = (argc > 4) ? std::stod(argv[4]) : 1.0;
    
    // Read experimental data
    std::vector<double> exp_temps;
    std::vector<double> exp_specific_heat;
    std::vector<double> exp_magnetization; // We'll initialize this but not use it
    
    std::ifstream infile(data_file);
    if (!infile) {
        std::cerr << "Error: Cannot open data file " << data_file << std::endl;
        return 1;
    }
    
    double temp, heat;
    while (infile >> temp >> heat) {
        exp_temps.push_back(temp);
        exp_specific_heat.push_back(heat);
    }
    
    if (exp_temps.empty()) {
        std::cerr << "Error: No data points read from file" << std::endl;
        return 1;
    }
    
    std::cout << "Read " << exp_temps.size() << " data points from " << data_file << std::endl;
    
    // Set up field direction (default along z-axis)
    std::array<double, 3> field_dir = {0.0, 0.0, 1.0};
    
    // Run the fitting procedure
    std::cout << "Starting pyrochlore model fitting..." << std::endl;
    std::cout << "Field strength: " << field_strength << std::endl;
    std::cout << "Field direction: [" << field_dir[0] << ", " << field_dir[1] << ", " << field_dir[2] << "]" << std::endl;
    std::cout << "Weight for specific heat: " << weight_heat << std::endl;
    
    fit_pyrochlore_model(
        exp_temps,
        exp_specific_heat,
        exp_magnetization,
        field_dir,
        field_strength,
        output_dir,
        weight_heat,
        0.0 // Not using magnetization data
    );
    
    std::cout << "Fitting complete. Results saved to " << output_dir << std::endl;
    
    return 0;
}