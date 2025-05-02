#include "experiments.h"
#include <math.h>


void MD_BCAO_honeycomb(size_t num_trials, double h, array<double, 3> field_dir, string dir, double J1=-6.54, double Jzp=-3.76, double Jpmpm=0.15, double J2=-0.21, double J3=1.70, double Delta1=0.36, double Delta2=0, double Delta3=0.03){
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;

    // double J1_ = (2*J1 + Delta1*J1 + 2*Jpmpm - sqrt(2)*Jzp)/3;
    // double K = -2*Jpmpm + sqrt(2)*Jzp;
    // double Gamma = (-J1 + Delta1*J1 - 4*Jpmpm - sqrt(2)*Jzp)/3;
    // double Gammap = (-2*J1 + 2*Delta1*J1 + 4*Jpmpm + sqrt(2)*Jzp)/6;
    // array<array<double,3>, 3> J1x_ = {{{J1_+K,Gammap,Gammap},{Gammap,J1_,Gamma},{Gammap,Gamma,J1_}}};
    // array<array<double,3>, 3> J1y_ = {{{J1_,Gammap,Gamma},{Gammap,K+J1_,Gammap},{Gamma,Gammap,J1_}}};
    // array<array<double,3>, 3> J1z_ = {{{J1_,Gamma,Gammap},{Gamma,J1_,Gammap},{Gammap,Gammap,K+J1_}}};

    array<array<double,3>, 3> J1x_ = {{{J1+2*Jpmpm*cos(2*M_PI/3),-2*Jpmpm*sin(2*M_PI/3),Jzp*sin(2*M_PI/3)},{-2*Jpmpm*sin(2*M_PI/3),J1-2*Jpmpm*cos(2*M_PI/3),-Jzp*cos(2*M_PI/3)},{Jzp*sin(2*M_PI/3),-Jzp*cos(2*M_PI/3),J1*Delta1}}};
    array<array<double,3>, 3> J1y_ = {{{J1+2*Jpmpm*cos(-2*M_PI/3),-2*Jpmpm*sin(-2*M_PI/3),Jzp*sin(-2*M_PI/3)},{-2*Jpmpm*sin(-2*M_PI/3),J1-2*Jpmpm*cos(-2*M_PI/3),-Jzp*cos(-2*M_PI/3)},{Jzp*sin(-2*M_PI/3),-Jzp*cos(-2*M_PI/3),J1*Delta1}}};
    array<array<double,3>, 3> J1z_ = {{{J1+2*Jpmpm*cos(0),-2*Jpmpm*sin(0),Jzp*sin(0)},{-2*Jpmpm*sin(0),J1-2*Jpmpm*cos(0),-Jzp*cos(0)},{Jzp*sin(0),-Jzp*cos(0),J1*Delta1}}};


    array<array<double,3>, 3> J2_ = {{{J2,0,0},{0,J2,0},{0,0,Delta2*J2}}};
    array<array<double,3>, 3> J3_ = {{{J3,0,0},{0,J3,0},{0,0,Delta3*J3}}};

    std::cout << field_dir[0] << " " << field_dir[1] << " " << field_dir[2] << std::endl;
    array<double, 3> field = {4.8*h*field_dir[0],4.85*h*field_dir[1],2.5*h*field_dir[2]};
    

    //nearest neighbour
    atoms.set_bilinear_interaction(J1x_, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J1y_, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});

    //next nearest neighbour
    atoms.set_bilinear_interaction(J2_, 0, 0, {1,0,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {0,1,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {1,-1,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {-1,0,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {0,-1,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {-1,1,0});

    atoms.set_bilinear_interaction(J2_, 1, 1, {1,0,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {0,1,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {-1,1,0});
    //third nearest neighbour
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});

    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    double k_B = 0.08620689655;

    for(size_t i=0; i<num_trials;++i){

        lattice<3, 2, 24, 24, 1> MC(&atoms, 0.5);
        MC.simulated_annealing(100*k_B, 5*k_B, 100000, 1e3, true);
        MC.molecular_dynamics(0, 100, 1e-2, dir+"/"+std::to_string(i));
    }
}

void simulated_annealing_BCAO_honeycomb(string filename, double h, array<double, 3> field_dir, string dir, double J1=-6.54, double Jzp=-3.76, double Jpmpm=0.15, double J2=-0.21, double J3=1.70, double Delta1=0.36, double Delta2=0, double Delta3=0.03){
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;

    array<array<double,3>, 3> J1x_ = {{{J1+2*Jpmpm*cos(2*M_PI/3),-2*Jpmpm*sin(2*M_PI/3),Jzp*sin(2*M_PI/3)},{-2*Jpmpm*sin(2*M_PI/3),J1-2*Jpmpm*cos(2*M_PI/3),-Jzp*cos(2*M_PI/3)},{Jzp*sin(2*M_PI/3),-Jzp*cos(2*M_PI/3),J1*Delta1}}};
    array<array<double,3>, 3> J1y_ = {{{J1+2*Jpmpm*cos(-2*M_PI/3),-2*Jpmpm*sin(-2*M_PI/3),Jzp*sin(-2*M_PI/3)},{-2*Jpmpm*sin(-2*M_PI/3),J1-2*Jpmpm*cos(-2*M_PI/3),-Jzp*cos(-2*M_PI/3)},{Jzp*sin(-2*M_PI/3),-Jzp*cos(-2*M_PI/3),J1*Delta1}}};
    array<array<double,3>, 3> J1z_ = {{{J1+2*Jpmpm*cos(0),-2*Jpmpm*sin(0),Jzp*sin(0)},{-2*Jpmpm*sin(0),J1-2*Jpmpm*cos(0),-Jzp*cos(0)},{Jzp*sin(0),-Jzp*cos(0),J1*Delta1}}};


    array<array<double,3>, 3> J2_ = {{{J2,0,0},{0,J2,0},{0,0,Delta2*J2}}};
    array<array<double,3>, 3> J3_ = {{{J3,0,0},{0,J3,0},{0,0,Delta3*J3}}};

    std::cout << field_dir[0] << " " << field_dir[1] << " " << field_dir[2] << std::endl;
    array<double, 3> field = {4.8*h*field_dir[0],4.85*h*field_dir[1],2.5*h*field_dir[2]};
    

    //nearest neighbour
    atoms.set_bilinear_interaction(J1x_, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J1y_, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});

    //next nearest neighbour
    atoms.set_bilinear_interaction(J2_, 0, 0, {1,0,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {0,1,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {1,-1,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {-1,0,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {0,-1,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {-1,1,0});

    atoms.set_bilinear_interaction(J2_, 1, 1, {1,0,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {0,1,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {-1,1,0});
    //third nearest neighbour
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});

    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    double k_B = 0.08620689655;


    lattice<3, 2, 24, 24, 1> MC(&atoms, 0.5);
    MC.simulated_annealing(100*k_B, 1e-2*k_B, 100000, 0, true);
    for (size_t i=0; i<1e6;++i){
        MC.deterministic_sweep();
    }
    MC.write_to_file_pos(filename+"/pos.txt");
    MC.write_to_file_spin(filename+"/spins.txt", MC.spins);
}


void field_sweep_BCAO_honeycomb(size_t num_angles, size_t num_fields, double max_field, string output_dir, 
                               double J1=-6.54, double Jzp=-3.76, double Jpmpm=0.15, 
                               double J2=-0.21, double J3=1.70, double Delta1=0.36, 
                               double Delta2=0, double Delta3=0.03) {
    // Initialize MPI if not already initialized
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(NULL, NULL);
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create output directory
    if (rank == 0) {
        filesystem::create_directory(output_dir);
    }
    
    // Prepare for output file
    ofstream critical_fields;
    if (rank == 0) {
        critical_fields.open(output_dir + "/critical_fields.txt");
        critical_fields << "# Angle(deg) Critical_Field Mx My Mz Energy" << endl;
    }
    
    // Divide work among processes
    size_t total_angles = num_angles + 1;
    size_t angles_per_rank = total_angles / size;
    size_t remainder = total_angles % size;
    size_t start_idx = rank * angles_per_rank + min(rank, (int)remainder);
    size_t end_idx = start_idx + angles_per_rank + (rank < remainder ? 1 : 0);
    
    double k_B = 0.08620689655;
    
    // Local results storage
    vector<double> angles, critical_fields_values, energies;
    vector<array<double, 3>> magnetizations;
    
    // Process assigned angles
    for (size_t i = start_idx; i < end_idx; ++i) {
        double angle = i * M_PI / (2 * num_angles); // 0 to 90 degrees
        array<double, 3> field_dir = {cos(angle), sin(angle), 0.0};
        
        cout << "Rank " << rank << " processing angle: " << angle * 180.0 / M_PI << " degrees" << endl;
        
        string angle_dir = output_dir + "/angle_" + to_string(int(angle * 180.0 / M_PI));
        filesystem::create_directory(angle_dir);
        
        double critical_field = -1;
        array<double, 3> critical_magnetization = {0, 0, 0};
        double critical_energy = 0;
        double max_change = 0;
        
        vector<double> field_energies;
        vector<array<double, 3>> field_magnetizations;
        
        // Sweep field strengths
        for (size_t j = 0; j < num_fields; ++j) {
            double field_strength = max_field * (j + 1) / num_fields;
            
            string field_dir_path = angle_dir + "/field_" + to_string(field_strength);
            filesystem::create_directory(field_dir_path);
            
            // Setup simulation
            HoneyComb<3> atoms;
            
            // Setup interactions as in the original code
            array<array<double,3>, 3> J1x_ = {{{J1+2*Jpmpm*cos(2*M_PI/3),-2*Jpmpm*sin(2*M_PI/3),Jzp*sin(2*M_PI/3)},
                                             {-2*Jpmpm*sin(2*M_PI/3),J1-2*Jpmpm*cos(2*M_PI/3),-Jzp*cos(2*M_PI/3)},
                                             {Jzp*sin(2*M_PI/3),-Jzp*cos(2*M_PI/3),J1*Delta1}}};
            array<array<double,3>, 3> J1y_ = {{{J1+2*Jpmpm*cos(-2*M_PI/3),-2*Jpmpm*sin(-2*M_PI/3),Jzp*sin(-2*M_PI/3)},
                                             {-2*Jpmpm*sin(-2*M_PI/3),J1-2*Jpmpm*cos(-2*M_PI/3),-Jzp*cos(-2*M_PI/3)},
                                             {Jzp*sin(-2*M_PI/3),-Jzp*cos(-2*M_PI/3),J1*Delta1}}};
            array<array<double,3>, 3> J1z_ = {{{J1+2*Jpmpm*cos(0),-2*Jpmpm*sin(0),Jzp*sin(0)},
                                             {-2*Jpmpm*sin(0),J1-2*Jpmpm*cos(0),-Jzp*cos(0)},
                                             {Jzp*sin(0),-Jzp*cos(0),J1*Delta1}}};
            array<array<double,3>, 3> J2_ = {{{J2,0,0},{0,J2,0},{0,0,Delta2*J2}}};
            array<array<double,3>, 3> J3_ = {{{J3,0,0},{0,J3,0},{0,0,Delta3*J3}}};

            array<double, 3> field = {4.8*field_strength*field_dir[0], 
                                      4.85*field_strength*field_dir[1], 
                                      2.5*field_strength*field_dir[2]};
            
            // Setup all interactions
            atoms.set_bilinear_interaction(J1x_, 0, 1, {0,-1,0});
            atoms.set_bilinear_interaction(J1y_, 0, 1, {1,-1,0});
            atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});
            
            atoms.set_bilinear_interaction(J2_, 0, 0, {1,0,0});
            atoms.set_bilinear_interaction(J2_, 0, 0, {0,1,0});
            atoms.set_bilinear_interaction(J2_, 0, 0, {1,-1,0});
            atoms.set_bilinear_interaction(J2_, 0, 0, {-1,0,0});
            atoms.set_bilinear_interaction(J2_, 0, 0, {0,-1,0});
            atoms.set_bilinear_interaction(J2_, 0, 0, {-1,1,0});
            
            atoms.set_bilinear_interaction(J2_, 1, 1, {1,0,0});
            atoms.set_bilinear_interaction(J2_, 1, 1, {0,1,0});
            atoms.set_bilinear_interaction(J2_, 1, 1, {1,-1,0});
            atoms.set_bilinear_interaction(J2_, 1, 1, {-1,0,0});
            atoms.set_bilinear_interaction(J2_, 1, 1, {0,-1,0});
            atoms.set_bilinear_interaction(J2_, 1, 1, {-1,1,0});
            
            atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
            atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
            atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});
            
            atoms.set_field(field, 0);
            atoms.set_field(field, 1);
            
            // Run simulation
            lattice<3, 2, 24, 24, 1> MC(&atoms, 0.5);
            MC.simulated_annealing(100*k_B, 1e-2*k_B, 10000, 0, true);
            
            // Additional sweeps for convergence
            for (size_t k = 0; k < 1e4; ++k) {
                MC.deterministic_sweep();
            }
            
            double energy = MC.total_energy(MC.spins);
            array<double, 3> magnetization = MC.magnetization_local(MC.spins);
            
            field_energies.push_back(energy);
            field_magnetizations.push_back(magnetization);
            
            // Save results
            MC.write_to_file_spin(field_dir_path + "/spins.txt", MC.spins);
            MC.write_to_file_pos(field_dir_path + "/pos.txt");
            
            // Look for phase transition
            if (j > 0) {
                double energy_change = fabs(energy - field_energies[j-1]);
                double rel_change = energy_change / fabs(field_energies[j-1]);
                
                if (rel_change > max_change) {
                    max_change = rel_change;
                    critical_field = field_strength;
                    critical_magnetization = magnetization;
                    critical_energy = energy;
                }
            }
        }
        
        // Store results for this angle
        angles.push_back(angle * 180.0 / M_PI);
        critical_fields_values.push_back(critical_field);
        magnetizations.push_back(critical_magnetization);
        energies.push_back(critical_energy);
        
        // Write individual angle results
        ofstream angle_results(angle_dir + "/critical_field.txt");
        angle_results << "Angle: " << angle * 180.0 / M_PI << " deg" << endl;
        angle_results << "Critical_Field: " << critical_field << endl;
        angle_results << "Magnetization: " << critical_magnetization[0] << " " 
                    << critical_magnetization[1] << " " << critical_magnetization[2] << endl;
        angle_results.close();
    }
    
    // Gather results
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Write results from all processes
    if (rank == 0) {
        for (size_t i = 0; i < angles.size(); ++i) {
            critical_fields << angles[i] << " " << critical_fields_values[i] << " "
                          << magnetizations[i][0] << " " << magnetizations[i][1] << " " 
                          << magnetizations[i][2] << " " << energies[i] << endl;
        }
        
        // Gather results from other processes
        for (int src = 1; src < size; ++src) {
            int count;
            MPI_Recv(&count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            vector<double> recv_angles(count), recv_fields(count), recv_energies(count);
            vector<double> recv_mag_x(count), recv_mag_y(count), recv_mag_z(count);
            
            MPI_Recv(recv_angles.data(), count, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_fields.data(), count, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_mag_x.data(), count, MPI_DOUBLE, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_mag_y.data(), count, MPI_DOUBLE, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_mag_z.data(), count, MPI_DOUBLE, src, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_energies.data(), count, MPI_DOUBLE, src, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < count; ++i) {
                critical_fields << recv_angles[i] << " " << recv_fields[i] << " "
                              << recv_mag_x[i] << " " << recv_mag_y[i] << " " 
                              << recv_mag_z[i] << " " << recv_energies[i] << endl;
            }
        }
        
        critical_fields.close();
        cout << "Field sweep completed. Results saved to " << output_dir << endl;
    } else {
        // Send results to rank 0
        int count = angles.size();
        MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(angles.data(), count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send(critical_fields_values.data(), count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        
        vector<double> mag_x(count), mag_y(count), mag_z(count);
        for (int i = 0; i < count; ++i) {
            mag_x[i] = magnetizations[i][0];
            mag_y[i] = magnetizations[i][1];
            mag_z[i] = magnetizations[i][2];
        }
        
        MPI_Send(mag_x.data(), count, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
        MPI_Send(mag_y.data(), count, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        MPI_Send(mag_z.data(), count, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
        MPI_Send(energies.data(), count, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
    }
}


void field_sweep_BCAO_honeycomb_ac(size_t num_angles, size_t num_fields, double max_field, string output_dir, 
    double J1=-6.54, double Jzp=-3.76, double Jpmpm=0.15, 
    double J2=-0.21, double J3=1.70, double Delta1=0.36, 
    double Delta2=0, double Delta3=0.03) {
// Initialize MPI if not already initialized
int initialized;
MPI_Initialized(&initialized);
if (!initialized) {
MPI_Init(NULL, NULL);
}

int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

// Create output directory
if (rank == 0) {
filesystem::create_directory(output_dir);
}

// Prepare for output file
ofstream critical_fields;
if (rank == 0) {
critical_fields.open(output_dir + "/critical_fields.txt");
critical_fields << "# Angle(deg) Critical_Field Mx My Mz Energy" << endl;
}

// Divide work among processes
size_t total_angles = num_angles + 1;
size_t angles_per_rank = total_angles / size;
size_t remainder = total_angles % size;
size_t start_idx = rank * angles_per_rank + min(rank, (int)remainder);
size_t end_idx = start_idx + angles_per_rank + (rank < remainder ? 1 : 0);

double k_B = 0.08620689655;

// Local results storage
vector<double> angles, critical_fields_values, energies;
vector<array<double, 3>> magnetizations;

// Process assigned angles
for (size_t i = start_idx; i < end_idx; ++i) {
double angle = i * M_PI / (2 * num_angles); // 0 to 90 degrees
array<double, 3> field_dir = {cos(angle), 0.0, sin(angle)};

cout << "Rank " << rank << " processing angle: " << angle * 180.0 / M_PI << " degrees" << endl;

string angle_dir = output_dir + "/angle_" + to_string(int(angle * 180.0 / M_PI));
filesystem::create_directory(angle_dir);

double critical_field = -1;
array<double, 3> critical_magnetization = {0, 0, 0};
double critical_energy = 0;
double max_change = 0;

vector<double> field_energies;
vector<array<double, 3>> field_magnetizations;

// Sweep field strengths
for (size_t j = 0; j < num_fields; ++j) {
double field_strength = max_field * (j + 1) / num_fields;

string field_dir_path = angle_dir + "/field_" + to_string(field_strength);
filesystem::create_directory(field_dir_path);

// Setup simulation
HoneyComb<3> atoms;

// Setup interactions as in the original code
array<array<double,3>, 3> J1x_ = {{{J1+2*Jpmpm*cos(2*M_PI/3),-2*Jpmpm*sin(2*M_PI/3),Jzp*sin(2*M_PI/3)},
                  {-2*Jpmpm*sin(2*M_PI/3),J1-2*Jpmpm*cos(2*M_PI/3),-Jzp*cos(2*M_PI/3)},
                  {Jzp*sin(2*M_PI/3),-Jzp*cos(2*M_PI/3),J1*Delta1}}};
array<array<double,3>, 3> J1y_ = {{{J1+2*Jpmpm*cos(-2*M_PI/3),-2*Jpmpm*sin(-2*M_PI/3),Jzp*sin(-2*M_PI/3)},
                  {-2*Jpmpm*sin(-2*M_PI/3),J1-2*Jpmpm*cos(-2*M_PI/3),-Jzp*cos(-2*M_PI/3)},
                  {Jzp*sin(-2*M_PI/3),-Jzp*cos(-2*M_PI/3),J1*Delta1}}};
array<array<double,3>, 3> J1z_ = {{{J1+2*Jpmpm*cos(0),-2*Jpmpm*sin(0),Jzp*sin(0)},
                  {-2*Jpmpm*sin(0),J1-2*Jpmpm*cos(0),-Jzp*cos(0)},
                  {Jzp*sin(0),-Jzp*cos(0),J1*Delta1}}};
array<array<double,3>, 3> J2_ = {{{J2,0,0},{0,J2,0},{0,0,Delta2*J2}}};
array<array<double,3>, 3> J3_ = {{{J3,0,0},{0,J3,0},{0,0,Delta3*J3}}};

array<double, 3> field = {4.8*field_strength*field_dir[0], 
           4.85*field_strength*field_dir[1], 
           2.5*field_strength*field_dir[2]};

// Setup all interactions
atoms.set_bilinear_interaction(J1x_, 0, 1, {0,-1,0});
atoms.set_bilinear_interaction(J1y_, 0, 1, {1,-1,0});
atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});

atoms.set_bilinear_interaction(J2_, 0, 0, {1,0,0});
atoms.set_bilinear_interaction(J2_, 0, 0, {0,1,0});
atoms.set_bilinear_interaction(J2_, 0, 0, {1,-1,0});
atoms.set_bilinear_interaction(J2_, 0, 0, {-1,0,0});
atoms.set_bilinear_interaction(J2_, 0, 0, {0,-1,0});
atoms.set_bilinear_interaction(J2_, 0, 0, {-1,1,0});

atoms.set_bilinear_interaction(J2_, 1, 1, {1,0,0});
atoms.set_bilinear_interaction(J2_, 1, 1, {0,1,0});
atoms.set_bilinear_interaction(J2_, 1, 1, {1,-1,0});
atoms.set_bilinear_interaction(J2_, 1, 1, {-1,0,0});
atoms.set_bilinear_interaction(J2_, 1, 1, {0,-1,0});
atoms.set_bilinear_interaction(J2_, 1, 1, {-1,1,0});

atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});

atoms.set_field(field, 0);
atoms.set_field(field, 1);

// Run simulation
lattice<3, 2, 24, 24, 1> MC(&atoms, 0.5);
MC.simulated_annealing(100*k_B, 1e-2*k_B, 10000, 0, true);

// Additional sweeps for convergence
for (size_t k = 0; k < 1e4; ++k) {
MC.deterministic_sweep();
}

double energy = MC.total_energy(MC.spins);
array<double, 3> magnetization = MC.magnetization_local(MC.spins);

field_energies.push_back(energy);
field_magnetizations.push_back(magnetization);

// Save results
MC.write_to_file_spin(field_dir_path + "/spins.txt", MC.spins);
MC.write_to_file_pos(field_dir_path + "/pos.txt");

// Look for phase transition
if (j > 0) {
double energy_change = fabs(energy - field_energies[j-1]);
double rel_change = energy_change / fabs(field_energies[j-1]);

if (rel_change > max_change) {
max_change = rel_change;
critical_field = field_strength;
critical_magnetization = magnetization;
critical_energy = energy;
}
}
}

// Store results for this angle
angles.push_back(angle * 180.0 / M_PI);
critical_fields_values.push_back(critical_field);
magnetizations.push_back(critical_magnetization);
energies.push_back(critical_energy);

// Write individual angle results
ofstream angle_results(angle_dir + "/critical_field.txt");
angle_results << "Angle: " << angle * 180.0 / M_PI << " deg" << endl;
angle_results << "Critical_Field: " << critical_field << endl;
angle_results << "Magnetization: " << critical_magnetization[0] << " " 
<< critical_magnetization[1] << " " << critical_magnetization[2] << endl;
angle_results.close();
}

// Gather results
MPI_Barrier(MPI_COMM_WORLD);

// Write results from all processes
if (rank == 0) {
for (size_t i = 0; i < angles.size(); ++i) {
critical_fields << angles[i] << " " << critical_fields_values[i] << " "
<< magnetizations[i][0] << " " << magnetizations[i][1] << " " 
<< magnetizations[i][2] << " " << energies[i] << endl;
}

// Gather results from other processes
for (int src = 1; src < size; ++src) {
int count;
MPI_Recv(&count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

vector<double> recv_angles(count), recv_fields(count), recv_energies(count);
vector<double> recv_mag_x(count), recv_mag_y(count), recv_mag_z(count);

MPI_Recv(recv_angles.data(), count, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(recv_fields.data(), count, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(recv_mag_x.data(), count, MPI_DOUBLE, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(recv_mag_y.data(), count, MPI_DOUBLE, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(recv_mag_z.data(), count, MPI_DOUBLE, src, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(recv_energies.data(), count, MPI_DOUBLE, src, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

for (int i = 0; i < count; ++i) {
critical_fields << recv_angles[i] << " " << recv_fields[i] << " "
   << recv_mag_x[i] << " " << recv_mag_y[i] << " " 
   << recv_mag_z[i] << " " << recv_energies[i] << endl;
}
}

critical_fields.close();
cout << "Field sweep completed. Results saved to " << output_dir << endl;
} else {
// Send results to rank 0
int count = angles.size();
MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
MPI_Send(angles.data(), count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
MPI_Send(critical_fields_values.data(), count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

vector<double> mag_x(count), mag_y(count), mag_z(count);
for (int i = 0; i < count; ++i) {
mag_x[i] = magnetizations[i][0];
mag_y[i] = magnetizations[i][1];
mag_z[i] = magnetizations[i][2];
}

MPI_Send(mag_x.data(), count, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
MPI_Send(mag_y.data(), count, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
MPI_Send(mag_z.data(), count, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
MPI_Send(energies.data(), count, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
}
}

// int main(int argc, char** argv) {
//     simulated_annealing_BCAO_honeycomb("BCAO_sasha_ground_state", 0, {0,1,0}, "BCAO_sasha_ground_state");
// }

// int main(int argc, char** argv) {
//     double k_B = 0.08620689655;
//     double mu_B = 5.7883818012e-2;
//     int initialized;
//     MPI_Initialized(&initialized);
//     if (!initialized){
//         MPI_Init(NULL, NULL);
//     }
//     int size;
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     MD_BCAO_honeycomb(30, 0*mu_B, {0,1,0}, "BCAO_zero_field_5K_sasha");
//     int finalized;
//     MPI_Finalized(&finalized);
//     if (!finalized){
//         MPI_Finalize();
//     }
//     return 0;
// }

int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    field_sweep_BCAO_honeycomb_ac(10, 10, 15*mu_B, "BCAO_field_sweep_ac");
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}