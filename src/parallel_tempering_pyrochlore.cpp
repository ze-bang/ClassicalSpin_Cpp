#include "experiments.h"


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    // int initialized;
    // MPI_Initialized(&initialized);
    // if (!initialized){
    //     MPI_Init(NULL, NULL);
    // }
    // int size;
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    // vector<int> rank_to_write = {0};
    // // parallel_tempering_honeycomb(1, 1e-6, -1, 0.25, -0.02, 0.7, "test_parallel", rank_to_write);
    // double Jpm_start = argv[1] ? atof(argv[1]) : 0.0;
    // double Jpm_end = argv[2] ? atof(argv[2]) : 0.0;
    // int num_Jpm = argv[3] ? atoi(argv[3]) : 0;
    // double Jpmpm = argv[4] ? atof(argv[4]) : 0.0;
    // double h_min = argv[5] ? atof(argv[5]) : 0.0;
    // double h_max = argv[6] ? atof(argv[6]) : 0.0;
    // double num_H = argv[7] ? atoi(argv[7]) : 0;
    // string dir_string = argv[8] ? argv[8] : "001";
    // array<double, 3> field_dir;
    // if (dir_string == "001"){
    //     field_dir = {0,0,1};
    // }else if(dir_string == "110"){
    //     field_dir = {1/sqrt(2), 1/sqrt(2), 0};
    // }else if(dir_string == "1-10"){
    //     field_dir = {1/sqrt(2), -1/sqrt(2), 0};
    // }else{
    //     field_dir = {1/sqrt(3),1/sqrt(3),1/sqrt(3)};
    // }
    // int SLURM_TASK_ID = argv[9] ? atoi(argv[9]) : 0;
    // int Jpm_ind = SLURM_TASK_ID % num_Jpm;
    // int h_ind = SLURM_TASK_ID / num_Jpm;
    // double Jpm = Jpm_start + Jpm_ind*(Jpm_end-Jpm_start)/num_Jpm;
    // double h = h_min + h_ind*(h_max-h_min)/num_H;
    // string dir_name = argv[10] ? argv[10] : "";
    // filesystem::create_directory(dir_name);
    // string sub_dir = dir_name + "/Jpm_" + std::to_string(Jpm) + "_h_" + std::to_string(h) + "_" + std::to_string(Jpm_ind) + "_" + std::to_string(h_ind);
    // int MPI_n_tasks = argv[11] ? atoi(argv[11]) : 1;
    // std::cout << "Initializing parallel tempering calculation with parameters: " << "T = " << 10 << "-" << 1e-3 << " Jpm: " << Jpm << " Jpmpm: " << Jpmpm << " H: " << h << " field direction : " << dir_string << " saving to: " << dir_name << endl;
    // parallel_tempering_pyrochlore(1e-3, 10, -2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0, 0, 1, h, field_dir, sub_dir, {0});
    simulated_annealing_pyrochlore(20*k_B, 1e-2*k_B, 0.257363, 0.252536, 1.73305, 0, 0, 1, 0, {0,0,1}, "p_test", 0, true, true);

    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.2, {0,0,1}, "MC_phase_diagram_CZO_001");
    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.0, {1,1,1}, "MC_phase_diagram_CZO_111");
    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.0, {1,1,0}, "MC_phase_diagram_CZO_110");

    // MD_pyrochlore(1, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0, 0, 2.24, 1.5*mu_B/0.063, {1/sqrt(2), 1/sqrt(2), 0}, "pyrochlore_test_110");
    // MD_pyrochlore(20, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0, 0, 2.24, 1.5*mu_B/0.063, {0,0,1}, "pyrochlore_test_001");
    // std::cout << "finished" << std::endl;   
    return 0;
}