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
    double J1ab = argv[1] ? atof(argv[1]) : 0.0;
    double J1c = argv[2] ? atof(argv[2]) : 0.0;
    double J2ab = argv[3] ? atof(argv[3]) : 0.0;
    double J2c = argv[4] ? atof(argv[4]) : 0.0;
    double Ka = argv[5] ? atof(argv[5]) : 0.0;
    double Kc = argv[6] ? atof(argv[6]) : 0.0;
    double D1 = argv[7] ? atof(argv[7]) : 0.0;
    double D2 = argv[8] ? atof(argv[8]) : 0.0;
    double h = argv[9] ? atof(argv[9]) : 0.0;
    string dir_name = argv[10] ? argv[10] : "";
    int num_trials = argv[11] ? atoi(argv[11]) : 0;
    MD_TmFeO3_Fe(num_trials, 20, 1e-2, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, h, {1,0,0}, dir_name);
    return 0;
}