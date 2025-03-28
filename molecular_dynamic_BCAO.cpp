#include "experiments.h"
#include <math.h>


void MD_BCAO_honeycomb(size_t num_trials, double h, array<double, 3> field_dir, string dir, double J1=-6.54, double Jzp=-3.76, double Jpmpm=0.15, double J2=-0.21, double J3=1.70, double Delta1=0.36, double Delta2=0, double Delta3=0.03){
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;

    // double J1_ = (2*J1 + Delta1*J1 + 2*Jpmpm - sqrt(2)*Jzp)/3;
    // double K = -2*Jpmpm + sqrt(2)*Jzp;
    // double Gamma = (-J1 + Delta1*J1 - 4*Jpmpm - sqrt(2)*Jzp)/3;
    // double Gammap = (-2*J1 + 2*Delta1*J1 + 4*Jpmpm + sqrt(2)*Jzp)/6;
    // array<array<double,3>, 3> J1x_ = {{{J1_+K,Gammap,Gammap},{Gammap,J1_,Gamma},{Gammap,Gamma,J1_}}};
    // array<array<double,3>, 3> J1y_ = {{{J1_,Gammap,Gamma},{Gammap,K+J1_,Gammap},{Gamma,Gammap,J1_}}};
    // array<array<double,3>, 3> J1z_ = {{{J1_,Gamma,Gammap},{Gamma,J1_,Gammap},{Gammap,Gammap,K+J1_}}};

    array<array<double,3>, 3> J1x_ = {{{J1+2*Jpmpm*cos(2*M_PI/3),-2*Jpmpm*sin(2*M_PI/3),Jzp*sin(2*M_PI/3)},{-2*Jpmpm*sin(2*M_PI/3),J1-2*Jpmpm*cos(2*M_PI/3),-Jzp*cos(2*M_PI/3)},{Jzp*sin(2*M_PI/3),-Jzp*cos(2*M_PI/3),J1*Delta1}}};
    array<array<double,3>, 3> J1y_ = {{{J1+2*Jpmpm*cos(-2*M_PI/3),-2*Jpmpm*sin(-2*M_PI/3),Jzp*sin(-2*M_PI/3)},{-2*Jpmpm*sin(-2*M_PI/3),J1-2*Jpmpm*cos(-2*M_PI/3),-Jzp*cos(-2*M_PI/3)},{Jzp*sin(-2*M_PI/3),-Jzp*cos(-2*M_PI/3),J1*Delta1}}};
    array<array<double,3>, 3> J1z_ = {{{J1+2*Jpmpm*cos(0),-2*Jpmpm*sin(0),Jzp*sin(0)},{-2*Jpmpm*sin(0),J1-2*Jpmpm*cos(0),-Jzp*cos(0)},{Jzp*sin(0),-Jzp*cos(0),J1*Delta1}}};


    array<array<double,3>, 3> J2_ = {{{J2,0,0},{0,J2,0},{0,0,Delta2*J2}}};
    array<array<double,3>, 3> J3_ = {{{J3,0,0},{0,J3,0},{0,0,Delta3*J3}}};

    std::cout << field_dir[0] << " " << field_dir[1] << " " << field_dir[2] << std::endl;
    array<double, 3> field = {4.8*h*field_dir[0],4.85*h*field_dir[1],2.5*h*field_dir[2]};
    

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

    atoms.set_bilinear_interaction(J2_, 1, 1, {1,0,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {0,1,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {1,-1,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {0,-1,0});
    atoms.set_bilinear_interaction(J2_, 1, 1, {-1,1,0});
    //third nearest neighbour
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {-1,0,0});
    atoms.set_bilinear_interaction(J3_, 0, 1, {1,-2,0});

    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    double k_B = 0.08620689655;

    for(size_t i=0; i<num_trials;++i){

        lattice<3, 2, 12, 12, 1> MC(&atoms, 0.5);
        MC.simulated_annealing(100*k_B, k_B, 100000, 0, true);
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
    MD_BCAO_honeycomb(1, 3*mu_B, {0,1,0}, "BCAO_test");
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}