#include "experiments.h"

void simulated_annealing_honeycomb(double T_start, double T_end, double K, double Gamma, double Gammap, double h, string dir="", bool deterministic=false){
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;
    array<array<double,3>, 3> Jx = {{{K,Gammap,Gammap},{Gammap,0,Gamma},{Gammap,Gamma,0}}};
    array<array<double,3>, 3> Jy = {{{0,Gammap,Gamma},{Gammap,K,Gammap},{Gamma,Gammap,0}}};
    array<array<double,3>, 3> Jz = {{{0,Gamma,Gammap},{Gamma,0,Gammap},{Gammap,Gammap,K}}};


    array<double, 3> z = {1, 1, 1};
    array<double, 3> y = {1, 1, -2};
    array<double, 3> x = {-1, 1, 0};

    z /= double(sqrt(3));
    y /= double(sqrt(6));
    x /= double(sqrt(2));
    double theta = 150 * M_PI / 180;

    array<double, 3> field = y*cos(theta) + x*sin(theta);
    // array<double, 3> field = y*(-0.15) + x*0.06;

    
    cout << "Setting up honeycomb lattice with parameters: " << endl;
    cout << "Jx: " << Jx[0][0] << " " << Jx[0][1] << " " << Jx[0][2] << endl;
    cout << "Jy: " << Jy[1][0] << " " << Jy[1][1] << " " << Jy[1][2] << endl;
    cout << "Jz: " << Jz[2][0] << " " << Jz[2][1] << " " << Jz[2][2] << endl;
    cout << "Field: " << field[0] << " " << field[1] << " " << field[2] << endl;

    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    lattice<3, 2, 36, 36, 1> MC(&atoms);
    if (deterministic == false){
        MC.simulated_annealing(T_start, T_end, 10000, 0, false, 0.9, dir);
    }
    else{
        MC.simulated_annealing_deterministic(T_start, T_end, 10000, 10000, 0, dir);
    }
}

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
    simulated_annealing_honeycomb(5, 1e-3, K, Gamma, Gammap, h, dir_name, deterministic);

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    cout << "Finished!" << endl;

    return 0;
}