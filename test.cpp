#include "unitcell.h"
#include "lattice.h"
#include "mixed_lattice.h"
#include <iostream>
#include <mpi.h>


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
        MC.molecular_dynamics(1,0.001, 1000, 10000, 0, 1000, 1e-1, dir+"/"+std::to_string(i));
    }
}


void nonlinearspectroscopy_kitaev_honeycomb(float tau_end, size_t tau_steps, float K, float h, string dir){
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

    array<array<float, 3>,2> field_drive = {{{0,0,1},{0,0,1}}};

    float pulse_width = 0.5;

    lattice<3, 2, 24, 24, 1> MC(&atoms);
    MC.simulated_annealing(1.0, 0.001, 1000, 10000, 0);
    MC.write_to_file_pos(dir+"/pos.txt");
    MC.write_to_file_spin(dir+"/spin_0.txt", MC.spins);

    lattice<3, 2, 24, 24, 1> temp_lattice(&MC);
    filesystem::create_directory(dir+"/M_time_0.000000");
    temp_lattice.M_B_t(field_drive, 0.0, pulse_width, 1000, 1e-1, dir+"/M_time_0.000000/M0");

    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start = rank*tau_steps/size;
    int end = (rank+1)*tau_steps/size;

    for(int i=start; i< end;++i){
        float current_time = i*tau_end/tau_steps;
        cout << "Time: " << current_time << " Rank: " << rank << endl;
        filesystem::create_directory(dir+"/M_time_"+ std::to_string(current_time));
        temp_lattice.reset_lattice(&MC);
        MC.M_B_t(field_drive, current_time, pulse_width, 1000, 1e-1, dir+"/M_time_"+ std::to_string(current_time) + "/M1");
        temp_lattice.reset_lattice(&MC);
        MC.M_BA_BB_t(field_drive, current_time, field_drive, pulse_width, 1000, 1e-1, dir+"/M_time_"+ std::to_string(current_time)+ "/M01");
    }

    MPI_Finalize();
}

void MD_TmFeO3(int num_trials, float J, float xi, string dir){
    HoneyComb<3> atoms_SU2;
    HoneyComb<8> atoms_SU3;
    
    mixed_UnitCell<3, 2, 8, 2> atoms(&atoms_SU2, &atoms_SU3);
    mixed_lattice<3, 2, 8, 2, 16, 16, 1> MC(&atoms);
}


int main(int argc, char** argv) {
    // MD_TmFeO3(1, -1.0, -0.06, "test_L=12");
    // MD_kitaev_honeycomb(1, -1.0, -0.06, "integrity_test");
    nonlinearspectroscopy_kitaev_honeycomb(1000, 5000, -1.0, -0.06, "nonlinear_spec_test");
    // std::cout << "finished" << std::endl;   
    return 0;
}