#include "experiments.h"


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;

    double Jxx = argv[1] ? atof(argv[1]) : 0.0;
    double Jyy = argv[2] ? atof(argv[2]) : 0.0;
    double Jzz = argv[3] ? atof(argv[3]) : 0.0;
    double h = argv[4] ? atof(argv[4]) : 0.0;
    string dir_string = argv[5] ? argv[5] : "001";
    double Jxz = argv[6] ? atof(argv[6]) : 0.0;
    array<double, 3> field_dir;
    if (dir_string == "001"){
        field_dir = {0,0,1};
    }else if(dir_string == "110"){
        field_dir = {1/sqrt(2), 1/sqrt(2), 0};
    }else if(dir_string == "1-10"){
        field_dir = {1/sqrt(2), -1/sqrt(2), 0};
    }
    else{
        field_dir = {1/sqrt(3),1/sqrt(3),1/sqrt(3)};
    }
    string dir_name = argv[7] ? argv[7] : "";
    filesystem::create_directory(dir_name);
    int num_trials = argv[8] ? atoi(argv[8]) : 1;
    std::cout << "Initializing molecular dynamic calculation with parameters: Jxx: " << Jxx << " Jyy: " << Jyy << " Jzz: " << Jzz << " Jxz: " << Jxz << " H: " << h << " field direction : " << dir_string << " with angle:" << theta << " saving to: " << dir_name << endl;
    MD_pyrochlore(num_trials, Jxx, Jyy, Jzz, 0.01, 4e-4, 1, h, field_dir, dir_name, Jxz);
    return 0;
}