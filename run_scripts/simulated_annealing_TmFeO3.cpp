#include "experiments.h"

void simulated_annealing_TmFeO3(double T_start, double T_end, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double e1, double e2, double xii, double h, const array<double,3> &fielddir, string dir){
       int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    mixed_lattice<3, 4, 8, 4, 8, 8, 8> MC(&TFO, 2.5, 1.0);
    MC.simulated_annealing(T_start, T_end, 10000, 0, 100, true, dir);

    for (size_t i = 0; i < 1e5; ++i) {
        MC.deterministic_sweep();
    }

    // Write the zero temperature spin configuration
    cout << "Writing zero temperature spin configuration to " << dir + "/Tzero" << endl;
    MC.write_to_file_spin(dir + "/spin_zero.txt");

}

int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    std::vector<int> rank_to_write = {{0}};
    double J1ab = argv[1] ? atof(argv[1]) : 0.0;
    double J1c = argv[2] ? atof(argv[2])/J1ab : 0.0;
    double J2ab = argv[3] ? atof(argv[3])/J1ab  : 0.0;
    double J2c = argv[4] ? atof(argv[4])/J1ab  : 0.0;
    double Ka = argv[5] ? atof(argv[5])/J1ab  : 0.0;
    double Kc = argv[6] ? atof(argv[6])/J1ab  : 0.0;
    double D1 = argv[7] ? atof(argv[7])/J1ab  : 0.0;
    double D2 = argv[8] ? atof(argv[8])/J1ab  : 0.0;
    double xii = argv[9] ? atof(argv[9])/J1ab  : 0.0;
    double e1 = argv[10] ? atof(argv[10])/J1ab  : 0.0;
    double e2 = argv[11] ? atof(argv[11])/J1ab  : 0.0;
    double h = argv[12] ? atof(argv[12])/J1ab  : 0.0;
    J1ab = 1;
    string dir_name = argv[13] ? argv[13] : "";
    double T_start = argv[14] ? atof(argv[14]) : 0.0;
    double T_end = argv[15] ? atof(argv[15]) : 0.0;
    cout << "Begin simulated annealing on TmFeO3 with parameters:" << J1ab << " " << J1c << " " << J2ab << " " << J2c << " " << Ka << " " << Kc << " " << D1 << " " << D2 << " " << xii << " " << e1 << " " << e2 << " " << h << " " << dir_name << endl;

    simulated_annealing_TmFeO3(T_start, T_end, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, e1, e2, xii, h, {0,0,1}, dir_name);
}