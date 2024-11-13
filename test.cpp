#include "unitcell.h"
#include "lattice.h"
#include <iostream>


void MD_kitaev_honeycomb(int num_trials, float K, float h, string dir){
    srand(time(NULL));
    filesystem::create_directory(dir);
    HoneyComb<3, 3, 0> atoms;
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

    int offsetx2[3] = {0,1,0};
    int offsety2[3] = {-1,1,0};

    atoms.set_bilinear_interaction(Jx, 1, 0, offsetx2);
    atoms.set_bilinear_interaction(Jy, 1, 0, offsety2);
    atoms.set_bilinear_interaction(Jz, 1, 0, offsetz);


    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    cout << "Finished setting up UC. Starting MD simulation" << endl;

    for(int i=0; i<num_trials;++i){
        lattice<3, 2, 36, 36, 1, 3, 0> MC(&atoms);
        cout << "Lattice initialized" << endl;
        MC.molecular_dynamics_SU2(1,0.001, 1000, 10000, 0, 1000, 1, dir+"/"+std::to_string(i));
    }
}

int main() {
    MD_kitaev_honeycomb(25, -1.0, -0.06, "test_L=24");
    return 0;
}