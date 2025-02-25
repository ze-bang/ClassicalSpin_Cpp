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
    double K = argv[1] ? atof(argv[1]) : 0.0;
    double Gamma = argv[2] ? atof(argv[2]) : 0.0;
    double Gammap = argv[3] ? atof(argv[3]) : 0.0;
    double h = argv[4] ? atof(argv[4]) : 0.0;
    string dir_name = argv[5] ? argv[5] : "";
    filesystem::create_directory(dir_name);
    int deterministic = argv[6] ? atoi(argv[6]) : 1;
    simulated_annealing_honeycomb(1, 1e-3, K, Gamma, Gammap, h, dir_name, deterministic);
    return 0;
}