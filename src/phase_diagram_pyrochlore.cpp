#include "experiments.h"


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    double Jpm_min = argv[1] ? atof(argv[1]) : 0.0;
    double Jpm_max = argv[2] ? atof(argv[2]) : 0.0;
    int num_Jpm = argv[3] ? atoi(argv[3]) : 0;
    double Jpmpm = argv[4] ? atof(argv[4]) : 0.0;
    double h_min = argv[5] ? atof(argv[5]) : 0.0;
    double h_max = argv[6] ? atof(argv[6]) : 0.0;
    int num_h = argv[7] ? atoi(argv[7]) : 0;
    string dir_string = argv[8] ? argv[8] : "001";
    array<double, 3> field_dir;
    if (dir_string == "001"){
        field_dir = {0,0,1};
    }else if(dir_string == "110"){
        field_dir = {1/sqrt(2), 1/sqrt(2), 0};
    }else if(dir_string == "1-10"){
        field_dir = {1/sqrt(2), -1/sqrt(2), 0};
    }else{
        field_dir = {1/sqrt(3),1/sqrt(3),1/sqrt(3)};
    }
    string dir = argv[9] ? argv[9] : "";
    phase_diagram_pyrochlore(Jpm_min, Jpm_max, num_Jpm, h_min, h_max, num_h, Jpmpm, field_dir, dir);
    return 0;
}