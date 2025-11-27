#include "../src/unitcell.h"
#include "../src/lattice.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

using namespace std;

void test_old_md(const string& initial_spins_file, double J=0, double K=-1, double Gamma=0.25, double Gammap=-0.02, double h=0.7) {
    std::cout << "========================================" << std::endl;
    std::cout << "Running OLD MD Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    HoneyComb_standarx<3> atoms;
    array<array<double,3>, 3> Jx = {{{J+K,Gammap,Gammap},{Gammap,J,Gamma},{Gammap,Gamma,J}}};
    array<array<double,3>, 3> Jy = {{{J,Gammap,Gamma},{Gammap,J+K,Gammap},{Gamma,Gammap,J}}};
    array<array<double,3>, 3> Jz = {{{J,Gamma,Gammap},{Gamma,J,Gammap},{Gammap,Gammap,J+K}}};
    array<double, 3> field = {h/double(sqrt(3)),h/double(sqrt(3)),h/double(sqrt(3))};
    
    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);

    lattice<3, 2, 24, 24, 1> MC(&atoms);
    
    // Load initial spins
    std::cout << "Loading initial spins from: " << initial_spins_file << std::endl;
    ifstream spin_file(initial_spins_file);
    if (!spin_file.is_open()) {
        std::cerr << "Error: Cannot open " << initial_spins_file << std::endl;
        return;
    }
    
    for(size_t site = 0; site < MC.lattice_size; ++site) {
        double x, y, z;
        if (!(spin_file >> x >> y >> z)) {
            std::cerr << "Error reading spin at site " << site << std::endl;
            return;
        }
        MC.spins[site][0] = x;
        MC.spins[site][1] = y;
        MC.spins[site][2] = z;
    }
    spin_file.close();
    
    // Compute initial energy
    double initial_energy = MC.total_energy(MC.spins);
    std::cout << "Initial energy: " << fixed << setprecision(12) << initial_energy << std::endl;
    std::cout << "Lattice size: " << MC.lattice_size << std::endl;
    
    // Run molecular dynamics
    filesystem::create_directory("MD_test_output");
    std::cout << "Running molecular dynamics..." << std::endl;
    MC.molecular_dynamics(0, 50, 1e-2, "MD_test_output/old");
    
    std::cout << "Old MD complete. Output saved to MD_test_output/old/" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main(int argc, char** argv) {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    
    string initial_spins = (argc > 1) ? argv[1] : "KITAEV/spins_initial_0.txt";
    double K = (argc > 2) ? atof(argv[2]) : -1.0;
    double Gamma = (argc > 3) ? atof(argv[3]) : 0.25;
    double Gammap = (argc > 4) ? atof(argv[4]) : -0.02;
    double h = (argc > 5) ? atof(argv[5]) : 0.7;
    double J = (argc > 6) ? atof(argv[6]) : 0.0;
    
    test_old_md(initial_spins, J, K, Gamma, Gammap, h);
    
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}
