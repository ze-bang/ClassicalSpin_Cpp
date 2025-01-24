#include "experiments.h"

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
    vector<int> rank_to_write = {size-1};
    bool T_zero = argv[1] ? atoi(argv[1]) : 0;
    double Temp_start = argv[2] ? atof(argv[2]) : 0.0;
    double Temp_end = argv[3] ? atof(argv[3]) : 0.0;
    double tau_start = argv[4] ? atof(argv[4]) : 0.0;
    double tau_end = argv[5] ? atof(argv[5]) : 0.0;
    double tau_step_size = argv[6] ? atof(argv[6]) : 0.0;
    double T_start = argv[7] ? atof(argv[7]) : 0.0;
    double T_end = argv[8] ? atof(argv[8]) : 0.0;
    double T_step_size = argv[9] ? atof(argv[9]) : 0.0;
    string field_extern_string = argv[10] ? argv[10] : "001";
    double Jpm = argv[11] ? atof(argv[11]) : 0.0;
    double Jpmpm = argv[12] ? atof(argv[12]) : 0.0;
    double h = argv[13] ? atof(argv[13]) : 0.0;
    string dir_string = argv[14] ? argv[14] : "001";
    double theta = argv[15] ? atof(argv[15]) : 0.0;
    array<double, 3> field_dir;
    array<double, 3> field_extern;
    if (dir_string == "001"){
        field_dir = {0,0,1};
    }else if(dir_string == "110"){
        field_dir = {1/sqrt(2), 1/sqrt(2), 0};
    }else{
        field_dir = {1/sqrt(3),1/sqrt(3),1/sqrt(3)};
    }

    if (field_extern_string == "001"){
        field_extern = {0,0,1};
    }else if(field_extern_string == "110"){
        field_extern = {1/sqrt(2), 1/sqrt(2), 0};
    }else{
        field_extern = {1/sqrt(3),1/sqrt(3),1/sqrt(3)};
    }

    string dir_name = argv[16] ? argv[16] : "";
    filesystem::create_directory(dir_name);
    int num_trials = argv[17] ? atoi(argv[17]) : 1;
    std::cout << "Initializing 2DCS calculation with parameters: Jpm: " << Jpm << " Jpmpm: " << Jpmpm << " H: " << h << " field direction : " << dir_string << " with angle:" << theta << "driven by field in direction " << field_extern_string << " saving to: " << dir_name << endl;
    pyrochlore_2DCS(num_trials, T_zero, Temp_start, Temp_end, tau_start, tau_end, tau_step_size, T_start, T_end, T_step_size, field_extern, -2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0, 0, 1, h, field_dir, dir_name, theta);
    return 0;
}