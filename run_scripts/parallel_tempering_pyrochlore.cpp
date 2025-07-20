#include "experiments.h"


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;

    simulated_annealing_pyrochlore(20*k_B, 1e-2*k_B, 0.257363, 0.252536, 1.73305, 0, 0, 1, 0, {0,0,1}, "p_test", 0, true, true);
    return 0;
}