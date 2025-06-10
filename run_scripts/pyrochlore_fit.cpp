#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include "experiments.h"

void fit_pyrochlore_model(const vector<double>& exp_temps, 
    const vector<double>& exp_specific_heat,
    const vector<double>& exp_magnetization,
    array<double, 3> field_dir,
    double field_strength,
    string output_dir,
    double weight_heat = 1.0,
    double weight_mag = 1.0) {
filesystem::create_directory(output_dir);
double k_B = 1.380649e-23; // Boltzmann constant in J/K
double k_B_meV = 8.617333262145e-2; // Boltzmann constant in meV/K
double avogadro = 6.02214076e23; // Avogadro's number in mol^-1
// Define the error function to minimize
    auto error_function = [&](const std::vector<double>& params, 
         const vector<double>& exp_temps,
         const vector<double>& exp_specific_heat,
         const vector<double>& exp_magnetization,
         array<double, 3> field_dir,
         double field_strength,
         double weight_heat,
         double weight_mag) -> double {

        double Jxx = params[0];
        double Jyy = params[1];
        double Jzz = params[2];

        double total_error = 0.0;

        // For each temperature point
        for (size_t i = 0; i < exp_temps.size(); i++) {
        // Create a temporary directory for this simulation
            string temp_dir = output_dir + "/temp_sim";
            filesystem::create_directory(temp_dir);

            // Run simulation at this temperature
            Pyrochlore<3> atoms;

            // Set up the lattice with current parameters
            array<double,3> z1 = {1, 1, 1};
            array<double,3> z2 = {1,-1,-1};
            array<double,3> z3 = {-1,1,-1};
            array<double,3> z4 = {-1,-1,1};

            z1 /= double(sqrt(3));
            z2 /= double(sqrt(3));
            z3 /= double(sqrt(3));
            z4 /= double(sqrt(3));

            array<array<double,3>, 3> J = {{{Jxx,0,0},{0,Jyy,0},{0,0,Jzz}}};
            array<double, 3> field = field_dir * field_strength;

            atoms.set_bilinear_interaction(J, 0, 1, {0, 0, 0}); 
            atoms.set_bilinear_interaction(J, 0, 2, {0, 0, 0}); 
            atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 0}); 
            atoms.set_bilinear_interaction(J, 1, 2, {0, 0, 0}); 
            atoms.set_bilinear_interaction(J, 1, 3, {0, 0, 0}); 
            atoms.set_bilinear_interaction(J, 2, 3, {0, 0, 0});

            atoms.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); 
            atoms.set_bilinear_interaction(J, 0, 2, {0, 1, 0}); 
            atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 1}); 
            atoms.set_bilinear_interaction(J, 1, 2, {-1, 1, 0}); 
            atoms.set_bilinear_interaction(J, 1, 3, {-1, 0, 1}); 
            atoms.set_bilinear_interaction(J, 2, 3, {0, 1, -1});

            atoms.set_field(field, 0);
            atoms.set_field(field, 1);
            atoms.set_field(field, 2);
            atoms.set_field(field, 3);

            lattice<3, 4, 4, 4, 4> MC(&atoms, 0.5);

            // Run MC to equilibrate
            // cout << "Running simulation for temperature: " << exp_temps[i] << endl;

            if (i == 0) {
                MC.simulated_annealing(20*k_B_meV, exp_temps[i]*k_B_meV, 100000, 1000, true, temp_dir, true);
            } else {
                MC.read_spin_from_file(temp_dir + "/spin.txt");
                filesystem::remove_all(temp_dir);
                MC.simulated_annealing(exp_temps[i-1]*k_B_meV, exp_temps[i]*k_B_meV, 100000, 1000, true, temp_dir, true);
            }            
            // Collect measurements
            ifstream data_file(temp_dir + "/specific_heat.txt");
            double T, sim_specific_heat, dheat;
            string line;
            if (!data_file.is_open()) {
                cerr << "Error opening file: " << temp_dir << "/specific_heat.txt" << endl;
            } else {
                // Read the last line (assuming the last measurement is what we want)
                while (getline(data_file, line)) {
                    istringstream iss(line);
                    iss >> T >> sim_specific_heat >> dheat;
                }
                data_file.close();
            }

            sim_specific_heat *= k_B*avogadro*2;
            dheat *= k_B*avogadro*2;

            cout << "Simulated specific heat at T = " << exp_temps[i] << ": " << sim_specific_heat << " Experimental: " << exp_specific_heat[i] << endl;

            // Calculate squared errors
            double heat_error = pow(sim_specific_heat - exp_specific_heat[i], 2);            
            // Add weighted errors
            total_error += weight_heat * heat_error;

            // Clean up temporary directory
        }
        filesystem::remove_all(output_dir + "/temp_sim");
        total_error = sqrt(total_error / exp_temps.size());
        cout << "Total error: " << total_error << endl;
        return total_error;
    };

    // Initialize Nelder-Mead optimizer
    optimization::NelderMead optimizer;

    // Initial guess for parameters
    std::vector<double> initial_params = {0.257363, 0.252536, 2.4};  // Initial guess for Jxx, Jyy, Jzz

    // Run optimization
    std::vector<double> best_params = optimizer.minimize(
    error_function, initial_params, 1e-5, 1000,
    exp_temps, exp_specific_heat, exp_magnetization,
    field_dir, field_strength, weight_heat, weight_mag
    );

    // Output best fit parameters
    cout << "Best fit parameters:" << endl;
    cout << "Jxx = " << best_params[0] << endl;
    cout << "Jyy = " << best_params[1] << endl;
    cout << "Jzz = " << best_params[2] << endl;

    // Save best parameters to file
    ofstream param_file(output_dir + "/best_params.txt");
    param_file << "Jxx = " << best_params[0] << endl;
    param_file << "Jyy = " << best_params[1] << endl;
    param_file << "Jzz = " << best_params[2] << endl;
    param_file.close();

    // Run a final simulation with best parameters to generate comparison data
    for (size_t i = 0; i < exp_temps.size(); i++) {
    // Create a temporary directory for this simulation
        string temp_dir = output_dir + "/temp_sim_" + to_string(i);
        filesystem::create_directory(temp_dir);
        Pyrochlore<3> atoms;

        array<double,3> z1 = {1, 1, 1};
        array<double,3> z2 = {1,-1,-1};
        array<double,3> z3 = {-1,1,-1};
        array<double,3> z4 = {-1,-1,1};

        z1 /= double(sqrt(3));
        z2 /= double(sqrt(3));
        z3 /= double(sqrt(3));
        z4 /= double(sqrt(3));

        array<array<double,3>, 3> J = {{{best_params[0],0,0},{0,best_params[1],0},{0,0,best_params[2]}}};
        array<double, 3> field = field_dir * field_strength;

        atoms.set_bilinear_interaction(J, 0, 1, {0, 0, 0}); 
        atoms.set_bilinear_interaction(J, 0, 2, {0, 0, 0}); 
        atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 0}); 
        atoms.set_bilinear_interaction(J, 1, 2, {0, 0, 0}); 
        atoms.set_bilinear_interaction(J, 1, 3, {0, 0, 0}); 
        atoms.set_bilinear_interaction(J, 2, 3, {0, 0, 0});

        atoms.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); 
        atoms.set_bilinear_interaction(J, 0, 2, {0, 1, 0}); 
        atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 1}); 
        atoms.set_bilinear_interaction(J, 1, 2, {-1, 1, 0}); 
        atoms.set_bilinear_interaction(J, 1, 3, {-1, 0, 1}); 
        atoms.set_bilinear_interaction(J, 2, 3, {0, 1, -1});

        atoms.set_field(field, 0);
        atoms.set_field(field, 1);
        atoms.set_field(field, 2);
        atoms.set_field(field, 3);
        lattice<3, 4, 6, 6, 6> MC(&atoms, 0.5);

        MC.simulated_annealing(20*k_B_meV, exp_temps[i]*k_B_meV, 100000, 1000, true, temp_dir, true);

        // Collect measurements
        ifstream infile(temp_dir + "/specific_heat.txt");
        double sim_specific_heat, dheat;
        infile >> sim_specific_heat >> dheat;
        infile.close();


        int lattice_size = 3*3*3*4;

        sim_specific_heat *= k_B*avogadro*2;
        dheat *= k_B*avogadro*2;

        ofstream outfile(output_dir + "/results_T" + to_string(exp_temps[i]) + ".txt");
        outfile << "Temperature: " << exp_temps[i] << endl;
        outfile << "Experimental specific heat: " << exp_specific_heat[i] << endl;
        outfile << "Simulated specific heat: " << sim_specific_heat << endl;
        outfile.close();
    }
}




void fit_pyrochlore_model_parallel_temp(vector<double>& exp_temps, 
const vector<double>& exp_specific_heat,
const vector<double>& exp_magnetization,
array<double, 3> field_dir,
double field_strength,
string output_dir,
double weight_heat = 1.0,
double weight_mag = 1.0) {

sort(exp_temps.begin(), exp_temps.end());

int initialized;
MPI_Initialized(&initialized);
if (!initialized){
MPI_Init(NULL, NULL);
}
int rank, size, partner_rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

filesystem::create_directory(output_dir);
double k_B = 1.380649e-23; // Boltzmann constant in J/K
double k_B_meV = 8.617333262145e-2; // Boltzmann constant in meV/K
double avogadro = 6.02214076e23; // Avogadro's number in mol^-1
// Define the error function to minimize

for (size_t i = 0; i < exp_temps.size(); i++) {
// Create a temporary directory for this simulation
exp_temps[i] = exp_temps[i] * k_B_meV;
}

auto error_function = [&](const std::vector<double>& params, 
const vector<double>& exp_temps,
const vector<double>& exp_specific_heat,
const vector<double>& exp_magnetization,
array<double, 3> field_dir,
double field_strength,
double weight_heat,
double weight_mag) -> double {

double Jxx = params[0];
double Jyy = params[1];
double Jzz = params[2];

double total_error = 0.0;

// For each temperature point
string temp_dir = output_dir + "/temp_sim_";
filesystem::create_directory(temp_dir);

// Run simulation at this temperature
Pyrochlore<3> atoms;

// Set up the lattice with current parameters
array<double,3> z1 = {1, 1, 1};
array<double,3> z2 = {1,-1,-1};
array<double,3> z3 = {-1,1,-1};
array<double,3> z4 = {-1,-1,1};

z1 /= double(sqrt(3));
z2 /= double(sqrt(3));
z3 /= double(sqrt(3));
z4 /= double(sqrt(3));

array<array<double,3>, 3> J = {{{Jxx,0,0},{0,Jyy,0},{0,0,Jzz}}};
array<double, 3> field = field_dir * field_strength;

atoms.set_bilinear_interaction(J, 0, 1, {0, 0, 0}); 
atoms.set_bilinear_interaction(J, 0, 2, {0, 0, 0}); 
atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 0}); 
atoms.set_bilinear_interaction(J, 1, 2, {0, 0, 0}); 
atoms.set_bilinear_interaction(J, 1, 3, {0, 0, 0}); 
atoms.set_bilinear_interaction(J, 2, 3, {0, 0, 0});

atoms.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); 
atoms.set_bilinear_interaction(J, 0, 2, {0, 1, 0}); 
atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 1}); 
atoms.set_bilinear_interaction(J, 1, 2, {-1, 1, 0}); 
atoms.set_bilinear_interaction(J, 1, 3, {-1, 0, 1}); 
atoms.set_bilinear_interaction(J, 2, 3, {0, 1, -1});

atoms.set_field(field, 0);
atoms.set_field(field, 1);
atoms.set_field(field, 2);
atoms.set_field(field, 3);

lattice<3, 4, 2, 2, 2> MC(&atoms, 0.5);

// Run MC to equilibrate
cout << "Running simulation for temperature: " << exp_temps[0] << " to " << exp_temps[exp_temps.size()-1] << endl;

MC.parallel_tempering(exp_temps, 1e6, 1e6, 1e3, 50, 2e3, temp_dir, {0}, true);
// Collect measurements
int lattice_size = 2*2*2*4;
if (rank == 0){
ifstream data_file(temp_dir + "/specific_heat.txt");
for (size_t i = 0; i < exp_temps.size(); i++) {
double t, sim_heat, dheat_temp;
data_file >> t >> sim_heat >> dheat_temp;
total_error += pow(sim_heat*k_B*avogadro/lattice_size- exp_specific_heat[i], 2);
cout << "Temperature: " << exp_temps[i] << " Simulated specific heat: " << sim_heat*k_B*avogadro/lattice_size << " Experimental specific heat: " << exp_specific_heat[i] << endl;
}
std::cout << "Total error: " << total_error << std::endl;
// Clean up temporary directory
filesystem::remove_all(temp_dir);
return sqrt(total_error)/exp_temps.size();
}
};

// Initialize Nelder-Mead optimizer
optimization::NelderMead optimizer;

// Initial guess for parameters
std::vector<double> initial_params = {0.05, 0.05, 0.1};  // Initial guess for Jxx, Jyy, Jzz

// Run optimization
std::vector<double> best_params = optimizer.minimize(
error_function, initial_params, 1e-5, 1000,
exp_temps, exp_specific_heat, exp_magnetization,
field_dir, field_strength, weight_heat, weight_mag
);

// Output best fit parameters
cout << "Best fit parameters:" << endl;
cout << "Jxx = " << best_params[0] << endl;
cout << "Jyy = " << best_params[1] << endl;
cout << "Jzz = " << best_params[2] << endl;

// Save best parameters to file
ofstream param_file(output_dir + "/best_params.txt");
param_file << "Jxx = " << best_params[0] << endl;
param_file << "Jyy = " << best_params[1] << endl;
param_file << "Jzz = " << best_params[2] << endl;
param_file.close();

// Run a final simulation with best parameters to generate comparison data
string temp_dir = output_dir + "/temp_sim";
filesystem::create_directory(temp_dir);
Pyrochlore<3> atoms;

array<double,3> z1 = {1, 1, 1};
array<double,3> z2 = {1,-1,-1};
array<double,3> z3 = {-1,1,-1};
array<double,3> z4 = {-1,-1,1};

z1 /= double(sqrt(3));
z2 /= double(sqrt(3));
z3 /= double(sqrt(3));
z4 /= double(sqrt(3));

array<array<double,3>, 3> J = {{{best_params[0],0,0},{0,best_params[1],0},{0,0,best_params[2]}}};
array<double, 3> field = field_dir * field_strength;

atoms.set_bilinear_interaction(J, 0, 1, {0, 0, 0}); 
atoms.set_bilinear_interaction(J, 0, 2, {0, 0, 0}); 
atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 0}); 
atoms.set_bilinear_interaction(J, 1, 2, {0, 0, 0}); 
atoms.set_bilinear_interaction(J, 1, 3, {0, 0, 0}); 
atoms.set_bilinear_interaction(J, 2, 3, {0, 0, 0});

atoms.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); 
atoms.set_bilinear_interaction(J, 0, 2, {0, 1, 0}); 
atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 1}); 
atoms.set_bilinear_interaction(J, 1, 2, {-1, 1, 0}); 
atoms.set_bilinear_interaction(J, 1, 3, {-1, 0, 1}); 
atoms.set_bilinear_interaction(J, 2, 3, {0, 1, -1});

atoms.set_field(field, 0);
atoms.set_field(field, 1);
atoms.set_field(field, 2);
atoms.set_field(field, 3);
lattice<3, 4, 2, 2, 2> MC(&atoms, 0.5);

// Run MC to equilibrate
cout << "Running simulation for temperature: " << exp_temps[0] << " to " << exp_temps[exp_temps.size()-1] << endl;

MC.parallel_tempering(exp_temps, 1e6, 1e6, 1e3, 50, 2e3, temp_dir, {0}, true);
// Collect measurements
ifstream data_file(temp_dir + "/specific_heat.txt");
int lattice_size = 2*2*2*4;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
if (rank == 0){
ofstream outfile(output_dir + "/results_T.txt");
for (size_t i = 0; i < exp_temps.size(); i++) {
double sim_heat, dheat_temp;
data_file >> sim_heat >> dheat_temp;
outfile << "Temperature: " << exp_temps[i] << endl;
outfile << "Experimental specific heat: " << exp_specific_heat[i] << endl;
outfile << "Simulated specific heat: " << sim_heat << endl;
}
outfile.close();
}


int finalized;
if (!MPI_Finalized(&finalized)){
MPI_Finalize();
}

}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <data_file> <output_dir> [field_strength] [weight_heat]" << std::endl;
        std::cout << "  data_file: File containing temperature and specific heat data (two columns)" << std::endl;
        std::cout << "  output_dir: Directory to save fitting results" << std::endl;
        std::cout << "  field_strength: Optional magnetic field strength (default: 0.0)" << std::endl;
        std::cout << "  weight_heat: Optional weight for specific heat in fitting (default: 1.0)" << std::endl;
        return 1;
    }
    
    std::string data_file = argv[1] ? argv[1] : "specific_heat_Pr2Zr2O7.txt";
    std::string output_dir = argv[2] ? argv[2] : "output_Pr2Zr2O7";
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