#include "unitcell.h"
#include "lattice.h"
#include "mixed_lattice.h"
#include <iostream>


void MD_kitaev_honeycomb(size_t num_trials, float K, float h, string dir){
    srand(time(NULL));
    filesystem::create_directory(dir);
    HoneyComb<3> atoms;
    array<array<float,3>, 3> Jx = {{{K,0,0},{0,0,0},{0,0,0}}};
    array<array<float,3>, 3> Jy = {{{0,0,0},{0,K,0},{0,0,0}}};
    array<array<float,3>, 3> Jz = {{{0,0,0},{0,0,0},{0,0,K}}};

    array<float, 3> field = {h/float(sqrt(3)),h/float(sqrt(3)),h/float(sqrt(3))};

    int offsetx[3] = {0,-1,0};
    int offsety[3] = {1,-1,0};
    int offsetz[3] = {0,0,0};
    
    atoms.set_bilinear_interaction(Jx, 0, 1, offsetx);
    atoms.set_bilinear_interaction(Jy, 0, 1, offsety);
    atoms.set_bilinear_interaction(Jz, 0, 1, offsetz);
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    for(size_t i=0; i<num_trials;++i){

        lattice<3, 2, 24, 24, 1> MC(&atoms);
        // MC.simulated_annealing(1, 0.001, 1000, 10000, 1000000, 0, dir+"/"+std::to_string(i));
        MC.molecular_dynamics(1,0.001, 1000, 10000, 0, 1000, 1e-2, dir+"/"+std::to_string(i));
    }
}


void MD_TmFeO3(int num_trials, float J, float xi, string dir){
    HoneyComb<3> atoms_SU2;
    HoneyComb<8> atoms_SU3;
    
    mixed_UnitCell<3, 2, 8, 2> atoms(&atoms_SU2, &atoms_SU3);
    mixed_lattice<3, 2, 8, 2, 16, 16, 1> MC(&atoms);
}


int main() {
    MD_TmFeO3(1, -1.0, -0.06, "test_L=12");
    // std::cout << "finished" << std::endl;   
    return 0;
}