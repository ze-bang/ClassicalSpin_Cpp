#include "experiments.h"
#include <math.h>

void parallel_tempering_BCAO(double T_start, double T_end, double h, array<double, 3> field_dir, string dir, double J1=-6.54, double Jzp=-3.76, double Jpmpm=0.15, double J2=-0.21, double J3=1.70, double Delta1=0.36, double Delta2=0, double Delta3=0.03){
    filesystem::create_directory(dir);
    HoneyComb_standarx<3> atoms;



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
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<double> temps = logspace(log10(T_start), log10(T_end), size);

    lattice<3, 2, 36, 36, 1> MC(&atoms, 0.5);
    MC.parallel_tempering(temps, 1e6, 1e6, 10, 50, 2e3, dir, {0});

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
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
    parallel_tempering_BCAO(1*k_B, 25*k_B, 0*mu_B, {0,1,0}, "/scratch/y/ybkim/zhouzb79/parallel_tempering_BCAO_zero_field_sasha");
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}