#include "simulation_config.h"
#include "experiments.h"
#include <mpi.h>
#include <iostream>
#include <memory>
#include <math.h>

using namespace std;

// Helper function to setup BCAO honeycomb Hamiltonian
template<size_t Lx, size_t Ly, size_t Lz>
void setup_BCAO_honeycomb(HoneyComb<3>& atoms, const SimulationConfig& config) {
    // Extract parameters with defaults
    double J1xy = config.get_param("J1xy", -7.6);
    double J1z = config.get_param("J1z", -1.2);
    double D = config.get_param("D", 0.1);
    double E = config.get_param("E", -0.1);
    double F = config.get_param("F", 0.0);
    double G = config.get_param("G", 0.0);
    double J3xy = config.get_param("J3xy", 2.5);
    double J3z = config.get_param("J3z", -0.85);
    
    // Energy scale
    double k_B = 1.0;
    J1xy *= k_B;
    J1z *= k_B;
    D *= k_B;
    E *= k_B;
    F *= k_B;
    G *= k_B;
    J3xy *= k_B;
    J3z *= k_B;
    
    // Build interaction matrices
    array<array<double,3>, 3> J1z_ = {{{J1xy+D, E, F},
                                        {E, J1xy-D, G},
                                        {F, G, J1z}}};
    array<array<double,3>, 3> U_2pi_3 = {{{cos(2*M_PI/3), sin(2*M_PI/3), 0},
                                          {-sin(2*M_PI/3), cos(2*M_PI/3), 0},
                                          {0, 0, 1}}};
    
    auto transpose = [](const array<array<double,3>, 3>& m) {
        array<array<double,3>, 3> res;
        for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) res[i][j] = m[j][i];
        return res;
    };
    
    auto multiply = [](const array<array<double,3>, 3>& A, const array<array<double,3>, 3>& B) {
        array<array<double,3>, 3> C = {{{0,0,0},{0,0,0},{0,0,0}}};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
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
    
    // Set nearest neighbor interactions
    atoms.set_bilinear_interaction(J1x_, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J1y_, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});
    
    // Set third nearest neighbor interactions
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});
    
    // Set magnetic field
    double mu_B = 0.05788; // meV/T
    array<double, 3> field = {
        config.field_scaling[0] * config.field_strength * mu_B * config.field_direction[0],
        config.field_scaling[1] * config.field_strength * mu_B * config.field_direction[1],
        config.field_scaling[2] * config.field_strength * mu_B * config.field_direction[2]
    };
    
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
}

// Helper function to setup Pyrochlore Hamiltonian
template<size_t Lx, size_t Ly, size_t Lz>
void setup_pyrochlore(Pyrochlore<3>& atoms, const SimulationConfig& config) {
    double Jxx = config.get_param("Jxx", 1.0);
    double Jyy = config.get_param("Jyy", 1.0);
    double Jzz = config.get_param("Jzz", 1.0);
    double gxx = config.get_param("gxx", 0.01);
    double gyy = config.get_param("gyy", 4e-4);
    double gzz = config.get_param("gzz", 1.0);
    double theta = config.get_param("theta", 0.0);
    
    array<array<double,3>, 3> J = {{{Jxx,0,0},{0,Jyy,0},{0,0,Jzz}}};
    
    // Set up sublattice local axes
    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};
    z1 /= sqrt(3);
    z2 /= sqrt(3);
    z3 /= sqrt(3);
    z4 /= sqrt(3);
    
    array<double, 3> y1 = {0,1,-1};
    array<double, 3> y2 = {0,-1,1};
    array<double, 3> y3 = {0,-1,-1};
    array<double, 3> y4 = {0,1,1};
    y1 /= sqrt(2);
    y2 /= sqrt(2);
    y3 /= sqrt(2);
    y4 /= sqrt(2);
    
    array<double, 3> x1 = {-2,1,1};
    array<double, 3> x2 = {-2,-1,-1};
    array<double, 3> x3 = {2,1,-1};
    array<double, 3> x4 = {2,-1,1};
    x1 /= sqrt(6);
    x2 /= sqrt(6);
    x3 /= sqrt(6);
    x4 /= sqrt(6);
    
    // Set bilinear interactions
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
    
    // Set up fields with local anisotropy
    auto dot = [](const array<double,3>& a, const array<double,3>& b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    };
    
    array<double, 3> field = config.field_direction * config.field_strength;
    array<double, 3> rot_field = {gzz*sin(theta)+gxx*cos(theta), 0, gzz*cos(theta)-gxx*sin(theta)};
    
    array<double, 3> By1 = {0, gyy*(pow(dot(field,y1),3) - 3*pow(dot(field,x1),2)*dot(field,y1)), 0};
    array<double, 3> By2 = {0, gyy*(pow(dot(field,y2),3) - 3*pow(dot(field,x2),2)*dot(field,y2)), 0};
    array<double, 3> By3 = {0, gyy*(pow(dot(field,y3),3) - 3*pow(dot(field,x3),2)*dot(field,y3)), 0};
    array<double, 3> By4 = {0, gyy*(pow(dot(field,y4),3) - 3*pow(dot(field,x4),2)*dot(field,y4)), 0};
    
    atoms.set_field(rot_field*dot(field, z1) + By1, 0);
    atoms.set_field(rot_field*dot(field, z2) + By2, 1);
    atoms.set_field(rot_field*dot(field, z3) + By3, 2);
    atoms.set_field(rot_field*dot(field, z4) + By4, 3);
}

// Unified simulation runner for honeycomb lattices
template<size_t Lx, size_t Ly, size_t Lz>
void run_honeycomb_simulation(const SimulationConfig& config, int rank, int size) {
    HoneyComb<3> atoms;
    setup_BCAO_honeycomb<Lx, Ly, Lz>(atoms, config);
    
    lattice<3, 2, Lx, Ly, Lz> MC(&atoms, config.initial_step_size, config.use_twist_boundary);
    
    double k_B = 1.0;
    
    switch (config.method) {
        case SimulationMethod::SIMULATED_ANNEALING: {
            filesystem::create_directories(config.output_dir);
            MC.simulated_annealing(
                config.T_start * k_B,
                config.T_end * k_B,
                config.annealing_steps,
                config.equilibration_steps,
                config.use_twist_boundary,
                false,
                config.cooling_rate,
                config.output_dir,
                true
            );
            
            // Save final configuration
            MC.write_to_file_pos(config.output_dir + "/pos.txt");
            MC.write_to_file_spin(config.output_dir + "/spins.txt", MC.spins);
            
            double energy = MC.energy_density(MC.spins);
            if (rank == 0) {
                ofstream energy_file(config.output_dir + "/energy_density.txt");
                energy_file << "Energy Density: " << energy << "\n";
                energy_file.close();
            }
            break;
        }
        
        case SimulationMethod::MOLECULAR_DYNAMICS: {
            filesystem::create_directories(config.output_dir);
            // First anneal to get ground state
            MC.simulated_annealing(
                config.T_start * k_B,
                config.T_end * k_B,
                config.annealing_steps,
                config.equilibration_steps,
                config.use_twist_boundary
            );
            // Then run MD
            MC.molecular_dynamics(0, config.md_time, config.md_timestep, config.output_dir);
            break;
        }
        
        case SimulationMethod::PARALLEL_TEMPERING: {
            filesystem::create_directories(config.output_dir);
            vector<double> temps;
            for (int i = 0; i < size; ++i) {
                double log_T = log10(config.T_start) + 
                              (log10(config.T_end) - log10(config.T_start)) * i / (size - 1);
                temps.push_back(pow(10, log_T) * k_B);
            }
            
            vector<int> ranks_to_write = {0}; // Only write from rank 0
            MC.parallel_tempering(
                temps,
                config.annealing_steps,
                config.annealing_steps,
                config.pt_sweeps_per_exchange,
                config.pt_exchange_frequency,
                2000,
                config.output_dir,
                ranks_to_write,
                true
            );
            break;
        }
        
        default:
            if (rank == 0) {
                cerr << "Simulation method not implemented for this geometry\n";
            }
            break;
    }
}

// Unified simulation runner for pyrochlore lattices
template<size_t Lx, size_t Ly, size_t Lz>
void run_pyrochlore_simulation(const SimulationConfig& config, int rank, int size) {
    Pyrochlore<3> atoms;
    setup_pyrochlore<Lx, Ly, Lz>(atoms, config);
    
    lattice<3, 4, Lx, Ly, Lz> MC(&atoms, config.initial_step_size);
    
    switch (config.method) {
        case SimulationMethod::SIMULATED_ANNEALING: {
            filesystem::create_directories(config.output_dir);
            MC.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.equilibration_steps,
                false,
                false,
                config.cooling_rate,
                config.output_dir,
                true
            );
            
            MC.write_to_file_pos(config.output_dir + "/pos.txt");
            MC.write_to_file_spin(config.output_dir + "/spins.txt", MC.spins);
            break;
        }
        
        case SimulationMethod::MOLECULAR_DYNAMICS: {
            filesystem::create_directories(config.output_dir);
            MC.simulated_annealing(
                config.T_start,
                config.T_end,
                config.annealing_steps,
                config.equilibration_steps,
                false
            );
            MC.molecular_dynamics(0, config.md_time, config.md_timestep, config.output_dir);
            break;
        }
        
        case SimulationMethod::PARALLEL_TEMPERING: {
            filesystem::create_directories(config.output_dir);
            vector<double> temps;
            for (int i = 0; i < size; ++i) {
                double log_T = log10(config.T_start) + 
                              (log10(config.T_end) - log10(config.T_start)) * i / (size - 1);
                temps.push_back(pow(10, log_T));
            }
            
            vector<int> ranks_to_write = {0};
            MC.parallel_tempering(
                temps,
                config.annealing_steps,
                config.annealing_steps,
                config.pt_sweeps_per_exchange,
                config.pt_exchange_frequency,
                2000,
                config.output_dir,
                ranks_to_write,
                true
            );
            break;
        }
        
        default:
            if (rank == 0) {
                cerr << "Simulation method not implemented for this geometry\n";
            }
            break;
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    string config_file = "simulation.conf";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    // Load configuration
    SimulationConfig config;
    try {
        config = SimulationConfig::from_file(config_file);
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Error loading configuration: " << e.what() << endl;
            cerr << "Usage: " << argv[0] << " [config_file]\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Validate configuration
    if (!config.validate()) {
        if (rank == 0) {
            cerr << "Configuration validation failed\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Print configuration on rank 0
    if (rank == 0) {
        config.print();
    }
    
    // Dispatch to appropriate simulation based on geometry and lattice size
    try {
        if (config.geometry == GeometryType::HONEYCOMB) {
            if (config.lattice_size[0] == 24 && config.lattice_size[1] == 24 && config.lattice_size[2] == 1) {
                run_honeycomb_simulation<24, 24, 1>(config, rank, size);
            } else if (config.lattice_size[0] == 36 && config.lattice_size[1] == 36 && config.lattice_size[2] == 1) {
                run_honeycomb_simulation<36, 36, 1>(config, rank, size);
            } else if (config.lattice_size[0] == 60 && config.lattice_size[1] == 60 && config.lattice_size[2] == 1) {
                run_honeycomb_simulation<60, 60, 1>(config, rank, size);
            } else {
                if (rank == 0) {
                    cerr << "Unsupported lattice size for honeycomb. Supported: 24x24x1, 36x36x1, 60x60x1\n";
                }
            }
        }
        else if (config.geometry == GeometryType::PYROCHLORE) {
            if (config.lattice_size[0] == 2 && config.lattice_size[1] == 2 && config.lattice_size[2] == 2) {
                run_pyrochlore_simulation<2, 2, 2>(config, rank, size);
            } else if (config.lattice_size[0] == 8 && config.lattice_size[1] == 8 && config.lattice_size[2] == 8) {
                run_pyrochlore_simulation<8, 8, 8>(config, rank, size);
            } else if (config.lattice_size[0] == 16 && config.lattice_size[1] == 16 && config.lattice_size[2] == 16) {
                run_pyrochlore_simulation<16, 16, 16>(config, rank, size);
            } else {
                if (rank == 0) {
                    cerr << "Unsupported lattice size for pyrochlore. Supported: 2x2x2, 8x8x8, 16x16x16\n";
                }
            }
        }
        else {
            if (rank == 0) {
                cerr << "Unsupported geometry type\n";
            }
        }
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Error during simulation: " << e.what() << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Finalize MPI
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
    
    return 0;
}
