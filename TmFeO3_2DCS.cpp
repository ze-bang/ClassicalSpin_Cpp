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


    double J1ab = argv[10] ? atof(argv[10]) : 0.0;
    double J1c = argv[11] ? atof(argv[11]) : 0.0;
    double J2ab = argv[12] ? atof(argv[12]) : 0.0;
    double J2c = argv[13] ? atof(argv[13]) : 0.0;
    double Ka = argv[14] ? atof(argv[14]) : 0.0;
    double Kc = argv[15] ? atof(argv[15]) : 0.0;
    double D1 = argv[16] ? atof(argv[16]) : 0.0;
    double D2 = argv[17] ? atof(argv[17]) : 0.0;
    double h = argv[18] ? atof(argv[18]) : 0.0;
    string dir_name = argv[19] ? argv[19] : "";
    filesystem::create_directory(dir_name);
    int num_trials = argv[20] ? atoi(argv[20]) : 1;

    TmFeO3_2DCS(num_trials, Temp_start, Temp_end, tau_start, tau_end, tau_step_size, T_start, T_end, T_step_size, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, h, {1,0,0}, dir_name, T_zero);
    return 0;
}