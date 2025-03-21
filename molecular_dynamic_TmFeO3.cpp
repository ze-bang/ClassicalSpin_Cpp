#include "experiments.h"


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    vector<int> rank_to_write = {0};
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
    double offset = argv[12] ? atof(argv[12]) : 0.0;
    double h = argv[13] ? atof(argv[13])/J1ab  : 0.0;
    J1ab = 1;
    string dir_name = argv[14] ? argv[14] : "";
    int num_trials = argv[15] ? atoi(argv[15]) : 0;
    double T_start = argv[16] ? atof(argv[16]) : 0.0;
    double T_end = argv[17] ? atof(argv[17]) : 0.0;
    double T_step_size = argv[18] ? atof(argv[18]) : 0.0;
    string spin_config_file = argv[19] ? argv[19] : "";
    cout << "Begin MD on TmFeO3 with parameters:" << J1ab << " " << J1c << " " << J2ab << " " << J2c << " " << Ka << " " << Kc << " " << D1 << " " << D2 << " " << xii << " " << e1 << " " << e2 << " " << h << " " << dir_name << " " << num_trials << endl;
    filesystem::create_directory(dir_name);

    ofstream myfile;
    myfile.open(dir_name + "/parameters.txt");
    myfile << "J1ab: " << J1ab << endl;
    myfile << "J1c: " << J1c << endl;
    myfile << "J2ab: " << J2ab << endl;
    myfile << "J2c: " << J2c << endl;
    myfile << "Ka: " << Ka << endl;
    myfile << "Kc: " << Kc << endl;
    myfile << "D1: " << D1 << endl;
    myfile << "D2: " << D2 << endl;
    myfile << "xii: " << xii << endl;
    myfile << "e1: " << e1 << endl;
    myfile << "e2: " << e2 << endl;
    myfile << "h: " << h << endl;
    myfile << "dir_name: " << dir_name << endl;
    myfile << "num_trials: " << num_trials << endl;
    myfile << "T_start: " << T_start << endl;
    myfile << "T_end: " << T_end << endl;
    myfile << "T_step_size: " << T_step_size << endl;
    myfile << "spin_config_file: " << spin_config_file << endl;
    myfile.close();
    // MD_TmFeO3_Fe(num_trials, 20, 1e-2, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, h, {1,0,0}, dir_name);
    MD_TmFeO3(num_trials, 20, 1e-2, T_start, T_end, T_step_size, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, xii, h, {1,0,0}, e1, e2, offset, dir_name, spin_config_file);
    
    return 0;
}
//J1ab=J1c=4.92meV J2ab=J2c=0.29meV Ka=0meV Kc=-0.09meV D1=D2=0 xii is the modeling param E1 = 0.97meV E2=3.9744792531meV