#include "experiments.h"

int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;

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
    double J1c = argv[11] ? atof(argv[11])/J1ab : 0.0;
    double J2ab = argv[12] ? atof(argv[12])/J1ab : 0.0;
    double J2c = argv[13] ? atof(argv[13])/J1ab : 0.0;
    double Ka = argv[14] ? atof(argv[14])/J1ab : 0.0;
    double Kc = argv[15] ? atof(argv[15])/J1ab : 0.0;
    double D1 = argv[16] ? atof(argv[16])/J1ab : 0.0;
    double D2 = argv[17] ? atof(argv[17])/J1ab : 0.0;
    double e1 = argv[18] ? atof(argv[18])/J1ab : 0.0;
    double e2 = argv[19] ? atof(argv[19])/J1ab : 0.0;
    double xii = argv[20] ? atof(argv[20])/J1ab : 0.0;
    double h = argv[21] ? atof(argv[21])/J1ab : 0.0;
    J1ab = 1;
    string dir_name = argv[22] ? argv[22] : "";
    filesystem::create_directory(dir_name);
    int num_trials = argv[23] ? atoi(argv[23]) : 1;
    string spin_config_file = argv[24] ? argv[24] : "";
    cout << "Initializing TmFeO3 2DCS calculation with parameters: J1ab: " << J1ab << " J1c: " << J1c << " J2ab: " << J2ab << " J2c: " << J2c << " Ka: " << Ka << " Kc: " << Kc << " D1: " << D1 << " D2: " << D2 << " H: " << h << " saving to: " << dir_name << endl;
    // MD_TmFeO3(num_trials, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, xii, h, {1,0,0}, e1, e2, dir_name);
    TmFeO3_2DCS(num_trials, Temp_start, Temp_end, tau_start, tau_end, tau_step_size, T_start, T_end, T_step_size, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, e1, e2, xii, h, {1,0,0}, dir_name, T_zero, spin_config_file);
    return 0;
}

//J1ab=J1c=4.92meV J2ab=J2c=0.29meV Ka=0meV Kc=-0.09meV D1=D2=0 xii is the modeling param E1 = -0.97meV E2=-3.89134081434meV