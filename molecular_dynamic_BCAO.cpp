#include "experiments.h"

void MD_BCAO_honeycomb(size_t num_trials, double J1, double Jzp, double Jpmpm, double J2, double J3, double Delta1, double Delta2, double Delta3, double h, string dir){
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;

    // double J1_ = (2*J1 + Delta1*J1 + 2*Jpmpm - sqrt(2)*Jzp)/3;
    // double K = -2*Jpmpm + sqrt(2)*Jzp;
    // double Gamma = (-J1 + Delta1*J1 - 4*Jpmpm - sqrt(2)*Jzp)/3;
    // double Gammap = (-2*J1 + 2*Delta1*J1 + 4*Jpmpm + sqrt(2)*Jzp)/6;
    // array<array<double,3>, 3> J1x_ = {{{J1_+K,Gammap,Gammap},{Gammap,J1_,Gamma},{Gammap,Gamma,J1_}}};
    // array<array<double,3>, 3> J1y_ = {{{J1_,Gammap,Gamma},{Gammap,K+J1_,Gammap},{Gamma,Gammap,J1_}}};
    // array<array<double,3>, 3> J1z_ = {{{J1_,Gamma,Gammap},{Gamma,J1_,Gammap},{Gammap,Gammap,K+J1_}}};

    array<array<double,3>, 3> J1x_ = {{{J1+2*Jpmpm*cos(alpha),-2*Jpmpm*sin(alpha),Jzp*sin(alpha)},{-2*Jpmpm*sin(alpha),J1-2*Jpmpm*cos(alpha),-Jzp*cos(alpha)},{Jzp*sin(alpha),-Jzp*cos(alpha),J1*Delta1}}};
    array<array<double,3>, 3> J1y_ = {{{J1+2*Jpmpm*cos(alpha),-2*Jpmpm*sin(alpha),Jzp*sin(alpha)},{-2*Jpmpm*sin(alpha),J1-2*Jpmpm*cos(alpha),-Jzp*cos(alpha)},{Jzp*sin(alpha),-Jzp*cos(alpha),J1*Delta1}}};
    array<array<double,3>, 3> J1z_ = {{{J1+2*Jpmpm*cos(0),-2*Jpmpm*sin(0),Jzp*sin(0)},{-2*Jpmpm*sin(0),J1-2*Jpmpm*cos(0),-Jzp*cos(0)},{Jzp*sin(0),-Jzp*cos(0),J1*Delta1}}};


    array<array<double,3>, 3> J2_ = {{{J2,0,0},{0,J2,0},{0,0,Delta2*J2}}};
    array<array<double,3>, 3> J3_ = {{{J3,0,0},{0,J3,0},{0,0,Delta3*J3}}};, 

    array<double, 3> field = {5*h/double(sqrt(3)),5*h/double(sqrt(3)),2.5*h/double(sqrt(3))};
    

    //nearest neighbour
    atoms.set_bilinear_interaction(J1x_, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J1y_, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J1z_, 0, 1, {0,0,0});

    //next nearest neighbour
    atoms.set_bilinear_interaction(J2_, 0, 0, {1,0,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {0,1,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {1,-1,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {-1,0,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {0,-1,0});
    atoms.set_bilinear_interaction(J2_, 0, 0, {-1,1,0});
    //third nearest neighbour
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});

    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    for(size_t i=0; i<num_trials;++i){

        lattice<3, 2, 12, 12, 1> MC(&atoms);
        MC.simulated_annealing(1, 1e-4, 100000, 0, true);
        // std::random_device rd;
        // std::mt19937 gen(rd());
        // for (size_t i = 0; i<100000; ++i){
        //     MC.deterministic_sweep(gen);
        // }
        MC.molecular_dynamics(0, 600, 0.25, dir+"/"+std::to_string(i));
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
    int num_trials = argv[6] ? atoi(argv[6]) : 1;
    MD_kitaev_honeycomb(num_trials, K, Gamma, Gammap, h, dir_name);
    return 0;
}