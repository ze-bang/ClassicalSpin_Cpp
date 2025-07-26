#include "experiments.h"
#include <math.h>
#include <mpi.h>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace std;

struct SimulationParams {
    size_t num_trials = 5;
    double h = 0.0;
    array<double, 3> field_dir = {0, 1, 0};
    string dir = "BCAO_simulation";
    double J1xy = -7.6, J1z = -1.2, D = 0.1, E = -0.1, F = 0, G = 0;
    double J3xy = 2.5, J3z = -0.85;
};

SimulationParams read_parameters(const string& filename) {
    SimulationParams params;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cout << "Warning: Could not open parameter file '" << filename << "'. Using default parameters.\n";
        return params;
    }
    
    string line;
    while (getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#' || line[0] == '/') continue;
        
        istringstream iss(line);
        string key, value;
        if (getline(iss, key, '=') && getline(iss, value)) {
            // Remove whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (key == "num_trials") params.num_trials = stoi(value);
            else if (key == "h") params.h = stod(value);
            else if (key == "field_dir") {
                stringstream ss(value);
                string item;
                int i = 0;
                while (getline(ss, item, ',') && i < 3) {
                    params.field_dir[i++] = stod(item);
                }
            }
            else if (key == "dir") params.dir = value;
            else if (key == "J1xy") params.J1xy = stod(value);
            else if (key == "J1z") params.J1z = stod(value);
            else if (key == "D") params.D = stod(value);
            else if (key == "E") params.E = stod(value);
            else if (key == "F") params.F = stod(value);
            else if (key == "G") params.G = stod(value);
            else if (key == "J3xy") params.J3xy = stod(value);
            else if (key == "J3z") params.J3z = stod(value);
        }
    }
    
    file.close();
    return params;
}

void create_default_parameter_file(const string& filename) {
    ofstream file(filename);
    file << "# BCAO Honeycomb Molecular Dynamics Parameters\n";
    file << "# Lines starting with # are comments\n";
    file << "# Format: parameter_name = value\n\n";
    file << "# Number of simulation trials\n";
    file << "num_trials = 5\n\n";
    file << "# Magnetic field strength in mu_B units\n";
    file << "h = 0.0\n\n";
    file << "# Field direction (x,y,z)\n";
    file << "field_dir = 0,1,0\n\n";
    file << "# Output directory\n";
    file << "dir = BCAO_simulation\n\n";
    file << "# Exchange interaction parameters\n";
    file << "J1xy = -7.6\n";
    file << "J1z = -1.2\n";
    file << "D = 0.1\n";
    file << "E = -0.1\n";
    file << "F = 0\n";
    file << "G = 0\n\n";
    file << "# Third nearest neighbor parameters\n";
    file << "J3xy = 2.5\n";
    file << "J3z = -0.85\n";
    file.close();
}


void MD_BCAO_honeycomb(size_t num_trials, double h, array<double, 3> field_dir, string dir, double J1xy=-7.6, double J1z=-1.2, double D=0.1, double E=-0.1, double F=0, double G=0, double J3xy=2.5, double J3z = -0.85){
    filesystem::create_directory(dir);
    HoneyComb_standarx<3> atoms;


    array<array<double,3>, 3> J1z_ = {{{J1xy+D, E, F},{-E, J1xy-D, -G},{F, -G, J1z}}};
    array<array<double,3>, 3> U_2pi_3 = {{{cos(2*M_PI/3), -sin(2*M_PI/3), 0},{sin(2*M_PI/3), cos(2*M_PI/3), 0},{0, 0, 1}}};

    auto transpose = [](const array<array<double,3>, 3>& m) {
        array<array<double,3>, 3> res;
        for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) res[i][j] = m[j][i];
        return res;
    };

    auto multiply = [](const array<array<double,3>, 3>& A, const array<array<double,3>, 3>& B) {
        array<array<double,3>, 3> C;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                C[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    };

    array<array<double,3>, 3> J1x_ = multiply(multiply(U_2pi_3, J1z_), transpose(U_2pi_3));
    array<array<double,3>, 3> J1y_ = multiply(multiply(transpose(U_2pi_3), J1z_), U_2pi_3);

    array<array<double,3>, 3> J3_ = {{{J3xy,0,0},{0,J3xy,0},{0,0,J3z}}};

    std::cout << field_dir[0] << " " << field_dir[1] << " " << field_dir[2] << std::endl;
    array<double, 3> field = {4.8*h*field_dir[0],4.85*h*field_dir[1],2.5*h*field_dir[2]};
    

    //nearest neighbour
    atoms.set_bilinear_interaction(J1x_, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J1y_, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});

    //third nearest neighbour
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});

    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    double k_B = 0.08620689655;

    // Save simulation parameters
    ofstream param_file(dir + "/simulation_parameters.txt");
    param_file << "Simulation Parameters for BCAO Honeycomb MD\n";
    param_file << "==========================================\n";
    param_file << "Number of trials: " << num_trials << "\n";
    param_file << "Magnetic field strength (h): " << h << "\n";
    param_file << "Field direction: [" << field_dir[0] << ", " << field_dir[1] << ", " << field_dir[2] << "]\n";
    param_file << "Applied field: [" << field[0] << ", " << field[1] << ", " << field[2] << "]\n";
    param_file << "J1xy: " << J1xy << "\n";
    param_file << "J1z: " << J1z << "\n";
    param_file << "D: " << D << "\n";
    param_file << "E: " << E << "\n";
    param_file << "F: " << F << "\n";
    param_file << "G: " << G << "\n";
    param_file << "J3xy: " << J3xy << "\n";
    param_file << "J3z: " << J3z << "\n";
    param_file << "Lattice size: 24x24x1\n";
    param_file << "Temperature for SA: 10K to 0.001K\n";
    param_file << "SA steps: 100000\n";
    param_file << "MD steps: 100\n";
    param_file << "MD timestep: 0.01\n";
    param_file.close();

    for(size_t i=0; i<num_trials;++i){

        lattice<3, 2, 36, 36, 1> MC(&atoms, 1);
        MC.simulated_annealing(20, 1, 100000, 10, true, 0.9, dir+"/"+std::to_string(i));
        MC.molecular_dynamics(0, 200, 1e-2, dir+"/"+std::to_string(i));
    }
}


int main(int argc, char** argv) {
    double mu_B = 5.7883818012e-2;
    
    // Default parameter file name
    string param_file = "bcao_parameters.txt";
    
    // Parse command line arguments for parameter file
    if (argc > 1) {
        if (string(argv[1]) == "--help" || string(argv[1]) == "-h") {
            cout << "Usage: " << argv[0] << " [parameter_file]\n";
            cout << "  parameter_file    Path to parameter file (default: bcao_parameters.txt)\n";
            cout << "  --help, -h        Show this help message\n";
            cout << "  --create-params   Create default parameter file\n\n";
            cout << "If parameter file doesn't exist, default parameters will be used.\n";
            cout << "Use --create-params to generate a template parameter file.\n";
            return 0;
        } else if (string(argv[1]) == "--create-params") {
            create_default_parameter_file("bcao_parameters.txt");
            cout << "Created default parameter file: bcao_parameters.txt\n";
            cout << "Edit this file and run the simulation again.\n";
            return 0;
        } else {
            param_file = argv[1];
        }
    }
    
    // Read parameters from file
    SimulationParams params = read_parameters(param_file);
    
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(&argc, &argv);
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    cout << "Running MD_BCAO_honeycomb with parameters from: " << param_file << "\n";
    cout << "Trials: " << params.num_trials << ", Field: " << params.h << " mu_B\n";
    cout << "Field direction: [" << params.field_dir[0] << "," << params.field_dir[1] << "," << params.field_dir[2] << "]\n";
    cout << "Output directory: " << params.dir << "\n";
    
    // Convert field strength to internal units
    double h_internal = params.h;
    
    MD_BCAO_honeycomb(params.num_trials, h_internal, params.field_dir, params.dir, 
                      params.J1xy, params.J1z, params.D, params.E, params.F, params.G, 
                      params.J3xy, params.J3z);
    
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}
