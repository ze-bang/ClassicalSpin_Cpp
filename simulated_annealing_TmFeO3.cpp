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
    double J1c = argv[2] ? atof(argv[2])/J1ab : 0.0;
    double J2ab = argv[3] ? atof(argv[3])/J1ab : 0.0;
    double J2c = argv[4] ? atof(argv[4])/J1ab : 0.0;
    double Ka = argv[5] ? atof(argv[5])/J1ab : 0.0;
    double Kc = argv[6] ? atof(argv[6])/J1ab : 0.0;
    double D1 = argv[7] ? atof(argv[7])/J1ab : 0.0;
    double D2 = argv[8] ? atof(argv[8])/J1ab : 0.0;
    double e1 = argv[9] ? atof(argv[9])/J1ab : 0.0;
    double e2 = argv[10] ? atof(argv[10])/J1ab : 0.0;
    double xii = argv[11] ? atof(argv[11])/J1ab : 0.0;
    double h = argv[12] ? atof(argv[12])/J1ab : 0.0;
    J1ab = 1;
    string dir_name = argv[13] ? argv[13] : "";
    int T_zero = argv[14] ? atoi(argv[14]) : 0;
    int num_trials = argv[15] ? atoi(argv[15]) : 0;
    // simulated_annealing_TmFeO3_Fe(num_trials, 20, 1e-2, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, h, {1,0,0}, dir_name, T_zero);
    simulated_annealing_TmFeO3(num_trials, 20, 1e-2, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, e1, e2, xii, h, {1,0,0}, dir_name, T_zero);
    return 0;
}
//J1ab=J1c=4.92meV J2ab=J2c=0.29meV Ka=0meV Kc=-0.09meV D1=D2=0 xii is the modeling param E1 = 0.97meV E2=3.89134081434meV