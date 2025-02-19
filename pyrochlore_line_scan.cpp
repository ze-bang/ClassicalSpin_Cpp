#include "experiments.h"


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    double Jxx = argv[1] ? atof(argv[1]) : 0.0;
    double Jyy = argv[2] ? atof(argv[2]) : 0.0;
    double Jzz = argv[3] ? atof(argv[3]) : 0;
    double h_min = argv[4] ? atof(argv[4]) : 0.0;
    double h_max = argv[5] ? atof(argv[5]) : 0.0;
    int num_h = argv[6] ? atoi(argv[6]) : 0;
    string dir_string = argv[7] ? argv[7] : "001";
    double Jxz = argv[8] ? atof(argv[8]) : 0.0;
    bool theta_or_Jxz = argv[9] ? atoi(argv[9]) : false;
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
    string dir = argv[10] ? argv[10] : "";
    bool save = argv[11] ? atoi(argv[11]) : false;
    pyrochlore_line_scan(Jxx, Jyy, Jzz, h_min, h_max, num_h, field_dir, dir, Jxz, theta_or_Jxz, save);
    return 0;
}

//CSO 1 -0.02222222222 -0.26666666666 -0.02276652512 or 0.49275362318 1 0.49275362318
//CHO 0.25 1 0.36363636363 -0.04545454545 or 1 0.47826086956 0.23913043478 -0.02173913043
//CZO 0.98412698412 1 0.1746031746