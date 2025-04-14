#include "experiments.h"
#include <math.h>


void MD_BCAO(int num_trials, double h, array<double, 3> field_dir, string dir, double J1=-3.3, double K=2.0, double Gamma=-0.9, double Gammap=3.3, double J2=-0.5, double J3=0.6, double J4=0.15){
    filesystem::create_directory(dir);
    HoneyComb_standarx<3> atoms;

    array<array<double,3>, 3> J1x_ = {{{J1+K,Gammap,Gammap},{Gammap,J1,Gamma},{Gammap,Gamma,J1}}};
    array<array<double,3>, 3> J1y_ = {{{J1,Gammap,Gamma},{Gammap,K+J1,Gammap},{Gamma,Gammap,J1}}};
    array<array<double,3>, 3> J1z_ = {{{J1,Gamma,Gammap},{Gamma,J1,Gammap},{Gammap,Gammap,K+J1}}};


    array<array<double,3>, 3> J2_ = {{{J2,0,0},{0,J2,0},{0,0,0.5*J2}}};
    array<array<double,3>, 3> J3_ = {{{J3,0,0},{0,J3,0},{0,0,0.5*J3}}};
    array<array<double,3>, 3> J4_ = {{{J4,0,0},{0,J4,0},{0,0,0.5*J4}}};

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

    //fourth nearest neighbour
    atoms.set_bilinear_interaction(J4_, 0, 1, {0,1,0});
    atoms.set_bilinear_interaction(J4_, 0, 1, {-1,1,0});
    atoms.set_bilinear_interaction(J4_, 0, 1, {2,-1,0});
    atoms.set_bilinear_interaction(J4_, 0, 1, {-1,-1,0});
    atoms.set_bilinear_interaction(J4_, 0, 1, {2,-2,0});
    atoms.set_bilinear_interaction(J4_, 0, 1, {0,-2,0});

    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    double k_B = 0.08620689655;

    for(size_t i=0; i<num_trials;++i){
        lattice<3, 2, 24, 24, 1> MC(&atoms, 0.5);
        MC.simulated_annealing(100*k_B, 8*k_B, 100000, 1e3, true);
        MC.molecular_dynamics(0, 100, 1e-2, dir+"/"+std::to_string(i));
    }
}



void simulated_annealing_BCAO(double h, array<double, 3> field_dir, string dir, double J1=-3.3, double K=2.0, double Gamma=-0.9, double Gammap=3.3, double J2=-0.5, double J3=0.6, double J4=0.15){
    filesystem::create_directory(dir);
    HoneyComb_standarx<3> atoms;

    array<array<double,3>, 3> J1x_ = {{{J1+K,Gammap,Gammap},{Gammap,J1,Gamma},{Gammap,Gamma,J1}}};
    array<array<double,3>, 3> J1y_ = {{{J1,Gammap,Gamma},{Gammap,K+J1,Gammap},{Gamma,Gammap,J1}}};
    array<array<double,3>, 3> J1z_ = {{{J1,Gamma,Gammap},{Gamma,J1,Gammap},{Gammap,Gammap,K+J1}}};


    array<array<double,3>, 3> J2_ = {{{J2,0,0},{0,J2,0},{0,0,J2}}};
    array<array<double,3>, 3> J3_ = {{{J3,0,0},{0,J3,0},{0,0,J3}}};
    array<array<double,3>, 3> J4_ = {{{J4,0,0},{0,J4,0},{0,0,J4}}};

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

    //fourth nearest neighbour
    atoms.set_bilinear_interaction(J4_, 0, 1, {0,1,0});
    atoms.set_bilinear_interaction(J4_, 0, 1, {-1,1,0});
    atoms.set_bilinear_interaction(J4_, 0, 1, {2,-1,0});
    atoms.set_bilinear_interaction(J4_, 0, 1, {-1,-1,0});
    atoms.set_bilinear_interaction(J4_, 0, 1, {2,-2,0});
    atoms.set_bilinear_interaction(J4_, 0, 1, {0,-2,0});

    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    double k_B = 0.08620689655;

    lattice<3, 2, 24, 24, 1> MC(&atoms, 0.5);
    MC.simulated_annealing(20, 1e-2, 100000, 1e3, true);
    for (size_t i=0; i<1e6;++i){
        MC.deterministic_sweep();
    }
    MC.write_to_file_pos(dir+"/pos.txt");
    MC.write_to_file_spin(dir+"/spin.txt", MC.spins);
}

int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    simulated_annealing_BCAO(0*mu_B, {0,1,0}, "BCAO_ground_state_songvilay");
    return 0;
}

// int main(int argc, char** argv) {
//     double k_B = 0.08620689655;
//     double mu_B = 5.7883818012e-2;
//     int initialized;
//     MPI_Initialized(&initialized);
//     if (!initialized){
//         MPI_Init(NULL, NULL);
//     }
//     int size;
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     MD_BCAO(20, 0*mu_B, {0,1,0}, "BCAO_zero_field_8K_songvilay");
//     int finalized;
//     MPI_Finalized(&finalized);
//     if (!finalized){
//         MPI_Finalize();
//     }
//     return 0;
// }