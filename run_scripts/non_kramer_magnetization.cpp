// filepath: /home/pc_linux/ClassicalSpin_Cpp/src/non_kramer_magnetization.cpp
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include <mpi.h>
#include <cmath>
#include "experiments.h"

using namespace std;

void non_kramer_pyrochlore_field_scan(double Tstart, double TargetT, double Jpm, double Jpmpm, double Jzz, 
                                      double gxx, double gyy, double gzz, 
                                      double h_min, double h_max, int num_h,
                                      array<double, 3> field_dir, string dir, bool save=false) {
    
    filesystem::create_directory(dir);
    
    // Initialize MPI if not already initialized
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(NULL, NULL);
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Distribute the work among MPI processes
    int total_jobs = num_h;
    int jobs_per_proc = total_jobs / size;
    int remainder = total_jobs % size;
    
    int start_job = rank * jobs_per_proc + min(rank, remainder);
    int end_job = start_job + jobs_per_proc + (rank < remainder ? 1 : 0);
    
    // Loop through assigned field values
    for (int i = start_job; i < end_job; ++i) {
        double h = h_min + (h_max - h_min) * i / (num_h - 1);
        
        cout << "Rank " << rank << " processing h = " << h << " (job " << i << " of " << total_jobs << ")" << endl;
        
        // Create subdirectory for this field strength
        string subdir = dir + "/h_" + to_string(h) + "_index_" + to_string(i);
        
        // Run simulation
        simulated_annealing_pyrochlore_non_kramer(Tstart, TargetT, Jpm, Jpmpm, Jzz, 
                                                 gxx, gyy, gzz, h, field_dir, subdir, save);
    }
    
    // Finalize MPI if we initialized it
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized && initialized) {
        MPI_Finalize();
    }
}

int main(int argc, char** argv) {
    if (argc < 16) {
        cout << "Usage: " << argv[0] << " Tstart TargetT Jpm Jpmpm Jzz gxx gyy gzz "
             << "h_min h_max num_h field_dir_x field_dir_y field_dir_z output_dir [save]" << endl;
        return 1;
    }
    
    // Parse command-line arguments
    double Tstart = stod(argv[1]);
    double TargetT = stod(argv[2]);
    double Jpm = stod(argv[3]);
    double Jpmpm = stod(argv[4]);
    double Jzz = stod(argv[5]);
    double gxx = stod(argv[6]);
    double gyy = stod(argv[7]);
    double gzz = stod(argv[8]);
    double h_min = stod(argv[9]);
    double h_max = stod(argv[10]);
    int num_h = stoi(argv[11]);
    
    // Field direction
    array<double, 3> field_dir = {stod(argv[12]), stod(argv[13]), stod(argv[14])};
    double norm = sqrt(field_dir[0]*field_dir[0] + field_dir[1]*field_dir[1] + field_dir[2]*field_dir[2]);
    field_dir[0] /= norm;
    field_dir[1] /= norm;
    field_dir[2] /= norm;
    
    string output_dir = argv[15];
    
    // Optional save parameter
    bool save = false;
    if (argc > 16) {
        save = (string(argv[16]) == "true" || string(argv[16]) == "1" || string(argv[16]) == "yes");
    }
    
    // Run the scan
    non_kramer_pyrochlore_field_scan(Tstart, TargetT, Jpm, Jpmpm, Jzz, gxx, gyy, gzz,
                                    h_min, h_max, num_h, field_dir, output_dir, save);
    
    return 0;
}