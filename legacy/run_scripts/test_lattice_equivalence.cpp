#include "../src/unitcell.h"
#include "../src/lattice.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>

using namespace std;

void dump_old_lattice_info(double J=0, double K=-1, double Gamma=0.25, double Gammap=-0.02, double h=0.7) {
    std::cout << "========================================" << std::endl;
    std::cout << "Old Lattice Implementation Info" << std::endl;
    std::cout << "========================================" << std::endl;
    
    HoneyComb<3> atoms;
    array<array<double,3>, 3> Jx = {{{J+K,Gammap,Gammap},{Gammap,J,Gamma},{Gammap,Gamma,J}}};
    array<array<double,3>, 3> Jy = {{{J,Gammap,Gamma},{Gammap,J+K,Gammap},{Gamma,Gammap,J}}};
    array<array<double,3>, 3> Jz = {{{J,Gamma,Gammap},{Gamma,J,Gammap},{Gammap,Gammap,J+K}}};
    array<double, 3> field = {h/sqrt(3), h/sqrt(3), h/sqrt(3)};
    
    atoms.set_bilinear_interaction(Jx, 0, 1, {0,-1,0});
    atoms.set_bilinear_interaction(Jy, 0, 1, {1,-1,0});
    atoms.set_bilinear_interaction(Jz, 0, 1, {0,0,0});
    atoms.set_field(field, 0);
    atoms.set_field(field, 1);
    
    lattice<3, 2, 24, 24, 1> MC(&atoms);
    
    // Output basic properties
    std::cout << "Lattice size: " << MC.lattice_size << std::endl;
    std::cout << "num_bi: " << MC.num_bi << std::endl;
    std::cout << "num_tri: " << MC.num_tri << std::endl;
    std::cout << "spin_length: " << MC.spin_length << std::endl;
    
    // Save detailed info to file
    filesystem::create_directory("lattice_test_output");
    ofstream out("lattice_test_output/old_lattice_info.txt");
    
    out << "lattice_size: " << MC.lattice_size << "\n";
    out << "num_bi: " << MC.num_bi << "\n";
    out << "num_tri: " << MC.num_tri << "\n\n";
    
    // Sample first 10 sites
    for(size_t site = 0; site < min(MC.lattice_size, size_t(10)); ++site) {
        out << "Site " << site << ":\n";
        out << "  Position: " << MC.site_pos[site][0] << " " 
            << MC.site_pos[site][1] << " " << MC.site_pos[site][2] << "\n";
        out << "  Field: " << MC.field[site][0] << " " 
            << MC.field[site][1] << " " << MC.field[site][2] << "\n";
        out << "  Spin: " << MC.spins[site][0] << " " 
            << MC.spins[site][1] << " " << MC.spins[site][2] << "\n";
        out << "  Num bilinear partners: " << MC.bilinear_partners[site].size() << "\n";
        out << "  Partners: ";
        for(size_t p = 0; p < MC.bilinear_partners[site].size(); ++p) {
            out << MC.bilinear_partners[site][p] << " ";
        }
        out << "\n";
        
        // Output interaction matrices
        for(size_t i = 0; i < min(MC.bilinear_interaction[site].size(), size_t(3)); ++i) {
            out << "  Interaction " << i << " matrix:\n";
            for(size_t row = 0; row < 3; ++row) {
                out << "    ";
                for(size_t col = 0; col < 3; ++col) {
                    out << fixed << setprecision(8) << MC.bilinear_interaction[site][i][row*3 + col] << " ";
                }
                out << "\n";
            }
        }
        
        // Compute site energy
        double site_energy = MC.site_energy(MC.spins[site], site);
        out << "  Site energy: " << fixed << setprecision(12) << site_energy << "\n\n";
    }
    
    // Total energy
    double total_energy = MC.total_energy(MC.spins);
    out << "Total energy: " << fixed << setprecision(12) << total_energy << "\n";
    out << "Energy density: " << fixed << setprecision(12) << MC.energy_density(MC.spins) << "\n";
    
    // Save spin configuration for comparison (set to simple test config)
    // Set all spins to [0, 0, 1] for deterministic comparison
    for(size_t site = 0; site < MC.lattice_size; ++site) {
        MC.spins[site][0] = 0.0;
        MC.spins[site][1] = 0.0;
        MC.spins[site][2] = 1.0;
    }
    
    // Recompute energy with deterministic spins
    total_energy = MC.total_energy(MC.spins);
    std::cout << "Total energy (all spins=[0,0,1]): " << fixed << setprecision(12) << total_energy << std::endl;
    
    ofstream spin_out("lattice_test_output/old_spins.txt");
    for(size_t site = 0; site < MC.lattice_size; ++site) {
        spin_out << MC.spins[site][0] << " " << MC.spins[site][1] << " " << MC.spins[site][2] << "\n";
    }
    spin_out.close();
    
    out.close();
    
    std::cout << "Detailed info saved to lattice_test_output/old_lattice_info.txt" << std::endl;
    std::cout << "Spin configuration saved to lattice_test_output/old_spins.txt" << std::endl;
    std::cout << "Total energy: " << fixed << setprecision(12) << total_energy << std::endl;
    std::cout << "========================================" << std::endl;
}

int main(int argc, char** argv) {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    
    double K = (argc > 1) ? atof(argv[1]) : -1.0;
    double Gamma = (argc > 2) ? atof(argv[2]) : 0.25;
    double Gammap = (argc > 3) ? atof(argv[3]) : -0.02;
    double h = (argc > 4) ? atof(argv[4]) : 0.7;
    double J = (argc > 5) ? atof(argv[5]) : 0.0;
    
    dump_old_lattice_info(J, K, Gamma, Gammap, h);
    
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}
