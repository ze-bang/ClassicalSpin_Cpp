#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include "unitcell.h"
#include "lattice.h"
#include "mixed_lattice.h"
// #include "lattice_strain_field.h"
#include <iostream>
#include <mpi.h>
#include "simple_linear_alg.h"
#include "fitting.h"

void  simulated_annealing_pyrochlore(double TargetT, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, double theta=0, bool theta_or_Jxz=true, bool save=false){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};

    z1 /= double(sqrt(3));
    z2 /= double(sqrt(3));
    z3 /= double(sqrt(3));
    z4 /= double(sqrt(3));


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
    double Jx, Jy, Jz, theta_in;
    cout << "Begin simulated annealing with parameters: " << Jxx << " " << Jyy << " " << Jzz << " " << theta_in << endl;
    if (theta_or_Jxz){

        Jx = Jxx;
        Jy = Jyy;
        Jz = Jzz;
        theta_in = theta;
    }
    else{
        Jy = Jyy;
        theta_in = atan(2*theta/(Jxx-Jzz))/2;
        Jx = cos(theta_in)*cos(theta_in)*Jxx + sin(theta_in)*sin(theta_in)*Jzz + sin(2*theta_in)*theta;
        Jz = sin(theta_in)*sin(theta_in)*Jxx + cos(theta_in)*cos(theta_in)*Jzz - sin(2*theta_in)*theta;
        cout << "Begin simulated annealing with parameters: " << Jx << " " << Jy << " " << Jz << " " << theta_in << endl;
        // Jx = (Jxx + Jzz)/2 - sqrt((Jxx-Jzz)*(Jxx-Jzz) + 4*theta*theta)/2;
        // Jz = (Jxx + Jzz)/2 + sqrt((Jxx-Jzz)*(Jxx-Jzz) + 4*theta*theta)/2;
        // cout << "Begin simulated annealing with parameters: " << Jx << " " << Jy << " " << Jz << " " << theta_in << endl;
        double maxJ = max(Jx, max(Jy, Jz));
        Jx /= maxJ;
        Jy /= maxJ;
        Jz /= maxJ;
    }
    cout << "Renormalized parameters: " << Jx << " " << Jy << " " << Jz << " " << theta_in << endl;
    array<array<double,3>, 3> J = {{{Jx,0,0},{0,Jy,0},{0,0,Jz}}};
    array<double, 3> field = field_dir*h;


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

    array<double, 3> rot_field = {gzz*sin(theta_in)+gxx*cos(theta_in),0,gzz*cos(theta_in)-gxx*sin(theta_in)};
    array<double, 3> By1 = {0, gyy*(pow(dot(field,y1),3) - 3*pow(dot(field,x1),2)*dot(field,y1)),0};
    array<double, 3> By2 = {0, gyy*(pow(dot(field,y2),3) - 3*pow(dot(field,x2),2)*dot(field,y2)),0};
    array<double, 3> By3 = {0, gyy*(pow(dot(field,y3),3) - 3*pow(dot(field,x3),2)*dot(field,y3)),0};
    array<double, 3> By4 = {0, gyy*(pow(dot(field,y4),3) - 3*pow(dot(field,x4),2)*dot(field,y4)),0};


    atoms.set_field(rot_field*dot(field, z1)+ By1, 0);
    atoms.set_field(rot_field*dot(field, z2)+ By2, 1);
    atoms.set_field(rot_field*dot(field, z3)+ By3, 2);
    atoms.set_field(rot_field*dot(field, z4)+ By4, 3);

    lattice<3, 4, 8, 8, 8> MC(&atoms, 0.5);
    // MC.simulated_annealing_deterministic(5, 1e-7, 10000, 10000, 0, dir);
    MC.simulated_annealing(5, 1e-4, 1e4, 0, true, dir, save);
}

void parallel_tempering_pyrochlore(double T_start, double T_end, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, const vector<int> &rank_to_write, double Jxz=0){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};

    z1 /= double(sqrt(3));
    z2 /= double(sqrt(3));
    z3 /= double(sqrt(3));
    z4 /= double(sqrt(3));


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


    double Jz_Jx = Jzz-Jxx;
    double Jz_Jz_sign = (Jzz-Jxx < 0) ? -1 : 1;
    double Jx = (Jxx+Jzz)/2 - Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*Jxz*Jxz)/2; 
    double Jy = Jyy;
    double Jz = (Jxx+Jzz)/2 + Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*Jxz*Jxz)/2; 
    double theta = atan(2*Jxz/(Jzz-Jxx));

    array<array<double,3>, 3> J = {{{Jx,0,0},{0,Jy,0},{0,0,Jz}}};
    array<double, 3> field = field_dir*h;


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

    array<double, 3> rot_field = {gzz*sin(theta)+gxx*cos(theta),0,gzz*cos(theta)-gxx*sin(theta)};
    array<double, 3> By1 = {0, gyy*(pow(dot(field,y1),3) - 3*pow(dot(field,x1),2)*dot(field,y1)),0};
    array<double, 3> By2 = {0, gyy*(pow(dot(field,y2),3) - 3*pow(dot(field,x2),2)*dot(field,y2)),0};
    array<double, 3> By3 = {0, gyy*(pow(dot(field,y3),3) - 3*pow(dot(field,x3),2)*dot(field,y3)),0};
    array<double, 3> By4 = {0, gyy*(pow(dot(field,y4),3) - 3*pow(dot(field,x4),2)*dot(field,y4)),0};


    atoms.set_field(rot_field*dot(field, z1)+ By1, 0);
    atoms.set_field(rot_field*dot(field, z2)+ By2, 1);
    atoms.set_field(rot_field*dot(field, z3)+ By3, 2);
    atoms.set_field(rot_field*dot(field, z4)+ By4, 3);

    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    lattice<3, 4, 16, 16, 16> MC(&atoms, 0.5);

    vector<double> temps = logspace(log10(T_start), log10(T_end), size);

    MC.parallel_tempering(temps, 1e6, 1e6, 10, 50, 2e3, dir, rank_to_write, true);

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

void phase_diagram_pyrochlore(double Jpm_min, double Jpm_max, int num_Jpm, double h_min, double h_max, int num_h, double Jpmpm, array<double, 3> field_dir, string dir){
    filesystem::create_directory(dir);
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int totaljob_num = num_Jpm*num_h;

    int start = rank*totaljob_num/size;
    int end = (rank+1)*totaljob_num/size;

    for(int i=start; i<end; ++i){
        int Jpm_ind = i % num_Jpm;
        int h_ind = i / num_Jpm;
        double Jpm = Jpm_min + Jpm_ind*(Jpm_max-Jpm_min)/num_Jpm;
        double h = h_min + h_ind*(h_max-h_min)/num_h;
        cout << "Jpm: " << Jpm << " h: " << h << "i: " << i << endl;
        string subdir = dir + "/Jpm_" + std::to_string(Jpm) + "_h_" + std::to_string(h) + "_index_" + std::to_string(Jpm_ind) + "_" + std::to_string(h_ind);
        simulated_annealing_pyrochlore(1e-4, -2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0.01, 4e-4, 1, h, field_dir, subdir);
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }

}

void pyrochlore_line_scan(double TargetT, double Jxx, double Jyy, double Jzz, double h_min, double h_max, int num_h, array<double, 3> field_dir, string dir, double theta, bool theta_or_Jxz, bool save){
    filesystem::create_directory(dir);
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int totaljob_num = num_h;

    int start = rank*totaljob_num/size;
    int end = (rank+1)*totaljob_num/size;

    for(int i=start; i<end; ++i){
        double h = h_min + i*(h_max-h_min)/num_h;
        cout << "h: " << h << "i: " << i << endl;
        string subdir = dir + "/h_" + std::to_string(h) + "_index_" + std::to_string(i);
        simulated_annealing_pyrochlore(TargetT, Jxx, Jyy, Jzz, 0.01, 4e-4, 1, h, field_dir, subdir, theta, theta_or_Jxz, save);
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }

}


void phase_diagram_pyrochlore_0_field(int num_Jpm, string dir){
    filesystem::create_directory(dir);
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int totaljob_num = num_Jpm*num_Jpm;

    int start = rank*totaljob_num/size;
    int end = (rank+1)*totaljob_num/size;

    for(int i=start; i<end; ++i){
        int Jpm_ind = i % num_Jpm;
        int h_ind = i / num_Jpm;
        double Jxx = -1 + double(Jpm_ind*2)/double(num_Jpm);
        double Jzz = -1 + double(h_ind*2)/double(num_Jpm);
        cout << "Jxx: " << Jxx << " Jzz: " << Jzz << "i: " << i << endl;
        string subdir = dir + "/Jxx_" + std::to_string(Jxx) + "_Jzz_" + std::to_string(Jzz) + "_index_" + std::to_string(Jpm_ind) + "_" + std::to_string(h_ind);
        simulated_annealing_pyrochlore(1e-4, Jxx, 1, Jzz, 0.01, 4e-4, 1, 0, {0,0,1}, subdir);
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

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
            string temp_dir = output_dir + "/temp_sim_" + to_string(i);
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
            MC.simulated_annealing(20*k_B_meV, exp_temps[i], 100000, 1000, true, temp_dir, true);
            
            // Collect measurements
            ifstream data_file(temp_dir + "/specific_heat.txt");
            double sim_specific_heat, dheat;
            data_file >> sim_specific_heat >> dheat;
            data_file.close();
            
            int lattice_size = 2*2*2*4;

            sim_specific_heat *= k_B*avogadro/lattice_size;
            dheat *= k_B*avogadro/lattice_size;
            // Calculate squared errors
            double heat_error = pow(sim_specific_heat - exp_specific_heat[i], 2);            
            // Add weighted errors
            total_error += weight_heat * heat_error;
            
            // Clean up temporary directory
            filesystem::remove_all(temp_dir);
        }
        
        return sqrt(total_error)/exp_temps.size();
    };
    
    // Initialize Nelder-Mead optimizer
    optimization::NelderMead optimizer;
    
    // Initial guess for parameters
    std::vector<double> initial_params = {0.010, 0.010, 0.050};  // Initial guess for Jxx, Jyy, Jzz
    
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
        lattice<3, 4, 2, 2, 2> MC(&atoms, 0.5);

        MC.simulated_annealing(20*k_B_meV, exp_temps[i], 100000, 1000, true, temp_dir, true);
            
        // Collect measurements
        ifstream infile(temp_dir + "/specific_heat.txt");
        double sim_specific_heat, dheat;
        infile >> sim_specific_heat >> dheat;
        infile.close();
        
        int lattice_size = 2*2*2*4;

        sim_specific_heat *= k_B*avogadro/lattice_size;
        dheat *= k_B*avogadro/lattice_size;
        
        ofstream outfile(output_dir + "/results_T" + to_string(exp_temps[i]) + ".txt");
        outfile << "Temperature: " << exp_temps[i] << endl;
        outfile << "Experimental specific heat: " << exp_specific_heat[i] << endl;
        outfile << "Simulated specific heat: " << sim_specific_heat << endl;
        outfile.close();
    }
}

#endif