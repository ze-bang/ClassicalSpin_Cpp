#include "experiments.h"


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    // MD_TmFeO3(1, -1.0, -0.06, "test_L=12");
    // MD_kitaev_honeycomb(1, -1.0, 0.0, -0.0, 0.06, "Pure_Kitaev_h=0.06");
    // string dir = "pure_kitaev_2DCS_h=0.7_pulse_amp=0.1/";
    // full_nonlinearspectroscopy_kitaev_honeycomb(1, 1, 1e-4, 0, -600, 0.25, -600, 600, 0.25, -1.0, 0.25, -0.02, 0.7, dir, true);
    // array<double, 3> field_in = {0,0,1};
    // MD_pyrochlore(1, 0.062/0.063, 1.0, 0.011/0.063, 0, 0, 2.24, 6, field_in*mu_B/0.063, "CZO_h=6T_001_theta=0.0", 0.0);
    // TmFeO3_2DCS(1,1000*k_B, 2*k_B, 0, -200/4.625, 0.05/4.625, -200/4.625, 200/4.625, 0.05/4.625, 4.625, 4.625, 4.625, 0.158, 0.158, 0.158, 0, -0.023, 0.0, 0.0, 0.0, {0,0,1}, "TmFeO3_Fe_Magnon_MD_real_meV", true, "TmFeO3_spin_config.txt");
    // MD_TmFeO3_Fe(1, 1000*k_B, 2*k_B, 4.625, 4.625, 4.625, 0.158, 0.158, 0.158, 0, -0.023, 0.0, 0.0, 0, {0,0,1}, "TmFeO3_Fe_Magnon_MD_real_meV");
    // MD_TmFeO3_2DCS(1000*k_B, 2*k_B, 0, -200/4.625, 0.2/4.625, -200/4.625, 200/4.625, 0.2/4.625, 4.625, 4.625, 4.625, 0.158, 0.158, 0.158, 0, -0.023, 0.0, 0.0, 0, {0,0,1}, "TmFeO3_Fe_Magnon_2DCS_real_meV");
    // simulated_annealing_honeycomb(1, 1e-6, -1, 0.25, -0.02, 0.7, "test_simulated_annealing");
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<int> rank_to_write = {size-1};
    // parallel_tempering_honeycomb(1, 1e-6, -1, 0.25, -0.02, 0.7, "test_parallel", rank_to_write);
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
    }else{
        field_dir = {1/sqrt(3),1/sqrt(3),1/sqrt(3)};
    }
    string dir_name = argv[6] ? argv[6] : "";
    filesystem::create_directory(dir_name);
    int num_trials = argv[7] ? atoi(argv[7]) : 1;
    std::cout << "Initializing molecular dynamic calculation with parameters: Jpm: " << Jpm << " Jpmpm: " << Jpmpm << " H: " << h << " field direction : " << dir_string << " with angle:" << theta << " saving to: " << dir_name << endl;
    parallel_tempering_pyrochlore(1e-3, 10, -2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0, 0, 1, h, field_dir, sub_dir, {0});
    MD_pyrochlore(num_trials, -2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0, 0, 1, h, field_dir, dir_name, theta);
    return 0;
}