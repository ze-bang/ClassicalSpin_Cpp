#include "experiments.h"


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;

    double Jpm = argv[1] ? atof(argv[1]) : 0.0;
    double Jpmpm = argv[2] ? atof(argv[2]) : 0.0;
    double h = argv[3] ? atof(argv[3]) : 0.0;
    string dir_string = argv[4] ? argv[4] : "001";
    double theta = argv[5] ? atof(argv[5]) : 0.0;
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
    string dir_name = argv[6] ? argv[6] : "";
    filesystem::create_directory(dir_name);
    int num_trials = argv[7] ? atoi(argv[7]) : 1;
    std::cout << "Initializing molecular dynamic calculation with parameters: Jpm: " << Jpm << " Jpmpm: " << Jpmpm << " H: " << h << " field direction : " << dir_string << " with angle:" << theta << " saving to: " << dir_name << endl;
    MD_pyrochlore(num_trials, -2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0, 0, 1, h, field_dir, dir_name, theta);
    return 0;
}