#include "experiments.h"


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    int resolution = argv[1] ? atoi(argv[1]) : 0;
    string dir = argv[2] ? argv[2] : "";
    phase_diagram_pyrochlore_0_field(resolution, dir);
    return 0;
}