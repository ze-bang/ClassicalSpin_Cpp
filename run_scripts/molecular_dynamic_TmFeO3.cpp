#include "experiments.h"
#include "../src/molecular_dynamics.cuh"
#include <fstream>
#include <sstream>
#include <unordered_map>

static bool load_params_from_file(const std::string &path, std::unordered_map<std::string, double> &out) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        size_t pos = line.find_first_not_of(" \t");
        if (pos == std::string::npos) continue;
        if (line[pos] == '#' || (line[pos] == '/' && pos + 1 < line.size() && line[pos + 1] == '/')) continue;
        std::istringstream iss(line);
        std::string key; double val;
        if (iss >> key >> val) out[key] = val;
    }
    return true;
}

void MD_TmFeO3_Fe(int num_trials, double T_start, double T_end, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double h, const array<double,3> &fielddir, string dir){
    filesystem::create_directory(dir);
    TmFeO3_Fe<3> Fe_atoms;
    TmFeO3_Tm<8> Tm_atoms;

    array<array<double, 3>, 3> Ja = {{{Jai, 0, 0}, {0, Jai, 0}, {0, 0, Jai}}};
    array<array<double, 3>, 3> Jb = {{{Jbi, 0, 0}, {0, Jbi, 0}, {0, 0, Jbi}}};
    array<array<double, 3>, 3> Jc = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};

    array<array<double, 3>, 3> J2a = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    array<array<double, 3>, 3> J2b = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    array<array<double, 3>, 3> J2c = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};

    array<double, 9> K = {{Ka, 0, 0, 0, 0, 0, 0, 0, Kc}};

    array<array<double, 3>,3> D = {{{0, D2, -D1}, {-D2, 0, 0}, {D1, 0, 0}}};
    //In plane interactions
    //Nearest Neighbours
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {1,-1,0});
    //Next Nearest Neighbours
    Fe_atoms.set_bilinear_interaction(J2a, 0, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 0, 0, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 1, 1, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 1, 1, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 2, 2, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 2, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a, 3, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b, 3, 3, {0,1,0});
    //Out of plane interaction
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc, 0, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc, 1, 2, {0,0,1});

    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,1,0});

    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {0,1,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 0, 2, {-1,1,1});

    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {0,-1,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c, 1, 3, {1,-1,1});

    //single ion anisotropy
    Fe_atoms.set_onsite_interaction(K, 0);
    Fe_atoms.set_onsite_interaction(K, 1);
    Fe_atoms.set_onsite_interaction(K, 2);
    Fe_atoms.set_onsite_interaction(K, 3);

    //Dzyaloshinskii-Moriya interaction
    Fe_atoms.set_bilinear_interaction(D, 0, 0, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 0, 0, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 1, 1, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 1, 1, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 2, 2, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 2, 2, {1,-1,0});
    Fe_atoms.set_bilinear_interaction(D, 3, 3, {1,1,0});
    Fe_atoms.set_bilinear_interaction(D, 3, 3, {1,-1,0});

    Fe_atoms.set_field(fielddir*h, 0);
    Fe_atoms.set_field(fielddir*h, 1);
    Fe_atoms.set_field(fielddir*h, 2);
    Fe_atoms.set_field(fielddir*h, 3);

    for(size_t i = 0; i < num_trials; ++i){
        lattice<3, 4, 8, 8, 8> MC_FE(&Fe_atoms, 2.5);
        MC_FE.simulated_annealing(T_start, T_end, 10000, 0, false);
        MC_FE.molecular_dynamics(0, 200/Jai, 5e-2/Jai, dir+"/"+std::to_string(i));
    }
}

void MD_TmFeO3(int num_trials, double Temp_start, double Temp_end, double T_start, double T_end, double T_step_size, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double chii, double xii, double h, const array<double,3> &fielddir, double e1, double e2, double offset, string dir, string spin_config_filename){
    filesystem::create_directory(dir);
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    TmFeO3_Fe<3> Fe_atoms;
    TmFeO3_Tm<8> Tm_atoms;
    // Define eta vectors
    array<array<double, 3>, 4> eta = {{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}};

    // Original exchange matrices
    array<array<double, 3>, 3> Ja_orig = {{{Jai, D2, -D1}, {-D2, Jai, 0}, {D1, 0, Jai}}};
    array<array<double, 3>, 3> Jb_orig = {{{Jbi, D2, -D1}, {-D2, Jbi, 0}, {D1, 0, Jbi}}};
    array<array<double, 3>, 3> Jc_orig = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};

    array<array<double, 3>, 3> J2a_orig = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    array<array<double, 3>, 3> J2b_orig = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    array<array<double, 3>, 3> J2c_orig = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};
    // Create 4x4 sublattice versions for each exchange matrix
    // First two indices denote sublattices i,j; last two are matrix indices
    array<array<array<array<double, 3>, 3>, 4>, 4> Ja;
    array<array<array<array<double, 3>, 3>, 4>, 4> Jb;
    array<array<array<array<double, 3>, 3>, 4>, 4> Jc;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2a;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2b;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2c;
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    Ja[i][j][a][b] = Ja_orig[a][b] * eta[i][a] * eta[j][b];
                    Jb[i][j][a][b] = Jb_orig[a][b] * eta[i][a] * eta[j][b];
                    Jc[i][j][a][b] = Jc_orig[a][b] * eta[i][a] * eta[j][b];
                    J2a[i][j][a][b] = J2a_orig[a][b] * eta[i][a] * eta[j][b];
                    J2b[i][j][a][b] = J2b_orig[a][b] * eta[i][a] * eta[j][b];
                    J2c[i][j][a][b] = J2c_orig[a][b] * eta[i][a] * eta[j][b];
                }
            }
        }
    }

    array<double, 9> K = {{Ka, 0, 0, 0, 0, 0, 0, 0, Kc}};
    //In plane interactions
    Fe_atoms.set_bilinear_interaction(Ja[1][0], 1, 0, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb[1][0], 1, 0, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb[1][0], 1, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja[1][0], 1, 0, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(Ja[2][3], 2, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb[2][3], 2, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb[2][3], 2, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja[2][3], 2, 3, {1,-1,0});

    //Next Nearest Neighbour
    Fe_atoms.set_bilinear_interaction(J2a[0][0], 0, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[0][0], 0, 0, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a[1][1], 1, 1, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[1][1], 1, 1, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a[2][2], 2, 2, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[2][2], 2, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a[3][3], 3, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[3][3], 3, 3, {0,1,0});

    //Out of plane interaction
    Fe_atoms.set_bilinear_interaction(Jc[0][3], 0, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc[0][3], 0, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(Jc[1][2], 1, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc[1][2], 1, 2, {0,0,1});

    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,1,0});

    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,1,1});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,1,1});

    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,-1,1});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,-1,1});

    //single ion anisotropy
    Fe_atoms.set_onsite_interaction(K, 0);
    Fe_atoms.set_onsite_interaction(K, 1);
    Fe_atoms.set_onsite_interaction(K, 2);
    Fe_atoms.set_onsite_interaction(K, 3);

    Fe_atoms.set_field(fielddir*h, 0);
    Fe_atoms.set_field(fielddir*h, 1);
    Fe_atoms.set_field(fielddir*h, 2);
    Fe_atoms.set_field(fielddir*h, 3);

    //Tm atoms
    //Set energy splitting for Tm atoms
    //\alpha\lambda3 + \beta\lambda8 + \gamma\identity
    double alpha = e1/2;
    double beta = sqrt(3)/6*(2*e2-e1);
    double gamma = -(e1+e2)/3 *3/16;

    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 0);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 1);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 2);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 3);



    TmFeO3<3, 8> TFO(&Fe_atoms, &Tm_atoms);
    // I have finally cracked the correct model...
    // Here we go!

    if (chii != 0.0){
        array<array<double,3>,8> chi = {{{0}}};
        chi[1] = {{0, 0, 5.264*chii}};
        chi[4] = {{2.3915*chii,2.7866*chii,0}};
        chi[6] = {{0.9128*chii,-0.4655*chii,0}};

        array<array<double,3>,8> chi_inv = {{{0}}};
        chi_inv[1] = {{0, 0, 5.264*chii}};
        chi_inv[4] = {{-2.3915*chii,-2.7866*chii,0}};
        chi_inv[6] = {{-0.9128*chii,0.4655*chii,0}};

        TFO.set_mix_bilinear_interaction(chi, 1, 0, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 1, 3, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 1, 1, {0,1,0});
        TFO.set_mix_bilinear_interaction(chi, 1, 2, {0,1,0});


        TFO.set_mix_bilinear_interaction(chi_inv, 1, 2, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 3, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 1, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 0, {1,0,0});

        ///////////////
        TFO.set_mix_bilinear_interaction(chi, 0, 0, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 0, 3, {0,0,1});
        TFO.set_mix_bilinear_interaction(chi, 0, 1, {0,1,0});
        TFO.set_mix_bilinear_interaction(chi, 0, 2, {0,1,1});

        TFO.set_mix_bilinear_interaction(chi_inv, 0, 2, {-1,1,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 0, 3, {0,1,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 0, 1, {-1,1,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 0, 0, {0,1,0});

        ///////////////
        TFO.set_mix_bilinear_interaction(chi, 2, 0, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 2, 3, {0,0,1});
        TFO.set_mix_bilinear_interaction(chi, 2, 1, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 2, 2, {0,0,1});

        TFO.set_mix_bilinear_interaction(chi_inv, 2, 2, {0,1,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 3, {1,0,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 1, {0,1,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 0, {1,0,0});

        ///////////////
        TFO.set_mix_bilinear_interaction(chi, 3, 0, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi, 3, 3, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi, 3, 1, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 3, 2, {0,0,0});

        TFO.set_mix_bilinear_interaction(chi_inv, 3, 2, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 3, {1,-1,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 1, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 0, {1,-1,0});
    }

    if (xii != 0.0){

        array<array<array<double,3>,3>,8> xi = {{{0}}};
        xi[0] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
        // xi[2] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
        // xi[7] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};

        ////////// Trilinear coupling/Oxygen path way
        TFO.set_mix_trilinear_interaction(xi, 1, 0, 3, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 1, 2, {0,1,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 2, 3, {0,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 1, 0, {0,0,0}, {1,0,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 1, 0, {0,1,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 2, 3, {0,1,0}, {1,0,0});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 2, 2, 3, {0,0,1}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 2, 2, 3, {0,1,1}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 2, 1, 2, {0,0,0}, {0,0,1});
        TFO.set_mix_trilinear_interaction(xi, 2, 0, 3, {1,0,0}, {1,0,1});
        //////////////////

        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,0,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,0,1}, {0,1,1});

        TFO.set_mix_trilinear_interaction(xi, 0, 1, 2, {-1,1,0}, {-1,1,1});
        TFO.set_mix_trilinear_interaction(xi, 0, 0, 3, {0,0,0}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,1,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,1,1}, {0,1,1});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,0,0});
        
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,-1,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,-1,0});

        TFO.set_mix_trilinear_interaction(xi, 3, 0, 3, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 2, {1,0,0}, {1,0,0});

        ///////////// Trilinear Interaction - Nearest neighbours

        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {1,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {1,0,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {1,0,1}, {0,0,1});
        TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {1,0,1}, {0,1,1});

        TFO.set_mix_trilinear_interaction(xi, 2, 0, 3, {0,0,0}, {0,0,1});
        TFO.set_mix_trilinear_interaction(xi, 2, 1, 2, {0,1,0}, {0,1,1});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,0,0}, {-1,1,0});
        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,1,0}, {-1,1,0});

        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,0,1}, {-1,1,1});
        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,1,1}, {-1,1,1});

        TFO.set_mix_trilinear_interaction(xi, 0, 0, 3, {0,1,0}, {0,1,1});
        TFO.set_mix_trilinear_interaction(xi, 0, 1, 2, {0,1,0}, {0,1,1});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {0,0,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {0,0,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 0, 3, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 2, 1, {0,0,0}, {0,0,0});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {1,0,0}, {1,-1,0});

        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {1,0,0}, {1,-1,0});

        TFO.set_mix_trilinear_interaction(xi, 3, 1, 2, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 0, 3, {1,-1,0}, {1,-1,0});
    }
    
    //
    cout << "Finished setting up TmFeO3 model." << endl;
    cout << "Starting calculations..." << endl;

    int trial_section = int(num_trials/size);

    for(size_t i = rank*trial_section; i < (rank+1)*trial_section; ++i){
        mixed_lattice<3, 4, 8, 4, 4, 4, 4> MC(&TFO, 2.5, 1.0);
        if (spin_config_filename != ""){
            MC.read_spin_from_file(spin_config_filename);
        }
        else{
            MC.simulated_annealing(Temp_start, Temp_end , 10000, 0, 100, false);
        }
        MC.molecular_dynamics(T_start, T_end, T_step_size, dir+"/"+std::to_string(i));
    }

}

void MD_TmFeO3_cuda(int num_trials, double Temp_start, double Temp_end, double T_start, double T_end, double T_step_size, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double chii, double xii, double h, const array<double,3> &fielddir, double e1, double e2, double offset, string dir, string spin_config_filename){
    filesystem::create_directories(dir);
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    TmFeO3_Fe<3> Fe_atoms;
    TmFeO3_Tm<8> Tm_atoms;
    // Define eta vectors
    array<array<double, 3>, 4> eta = {{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}};

    // Original exchange matrices
    array<array<double, 3>, 3> Ja_orig = {{{Jai, D2, -D1}, {-D2, Jai, 0}, {D1, 0, Jai}}};
    array<array<double, 3>, 3> Jb_orig = {{{Jbi, D2, -D1}, {-D2, Jbi, 0}, {D1, 0, Jbi}}};
    array<array<double, 3>, 3> Jc_orig = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};

    array<array<double, 3>, 3> J2a_orig = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    array<array<double, 3>, 3> J2b_orig = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    array<array<double, 3>, 3> J2c_orig = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};
    // Create 4x4 sublattice versions for each exchange matrix
    // First two indices denote sublattices i,j; last two are matrix indices
    array<array<array<array<double, 3>, 3>, 4>, 4> Ja;
    array<array<array<array<double, 3>, 3>, 4>, 4> Jb;
    array<array<array<array<double, 3>, 3>, 4>, 4> Jc;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2a;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2b;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2c;
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    Ja[i][j][a][b] = Ja_orig[a][b] * eta[i][a] * eta[j][b];
                    Jb[i][j][a][b] = Jb_orig[a][b] * eta[i][a] * eta[j][b];
                    Jc[i][j][a][b] = Jc_orig[a][b] * eta[i][a] * eta[j][b];
                    J2a[i][j][a][b] = J2a_orig[a][b] * eta[i][a] * eta[j][b];
                    J2b[i][j][a][b] = J2b_orig[a][b] * eta[i][a] * eta[j][b];
                    J2c[i][j][a][b] = J2c_orig[a][b] * eta[i][a] * eta[j][b];
                }
            }
        }
    }

    array<double, 9> K = {{Ka, 0, 0, 0, 0, 0, 0, 0, Kc}};
    //In plane interactions
    Fe_atoms.set_bilinear_interaction(Ja[1][0], 1, 0, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb[1][0], 1, 0, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb[1][0], 1, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja[1][0], 1, 0, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(Ja[2][3], 2, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb[2][3], 2, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb[2][3], 2, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja[2][3], 2, 3, {1,-1,0});

    //Next Nearest Neighbour
    Fe_atoms.set_bilinear_interaction(J2a[0][0], 0, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[0][0], 0, 0, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a[1][1], 1, 1, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[1][1], 1, 1, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a[2][2], 2, 2, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[2][2], 2, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2a[3][3], 3, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2b[3][3], 3, 3, {0,1,0});

    //Out of plane interaction
    Fe_atoms.set_bilinear_interaction(Jc[0][3], 0, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc[0][3], 0, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(Jc[1][2], 1, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jc[1][2], 1, 2, {0,0,1});

    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,1,0});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,1,0});

    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,1,1});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,1,1});

    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,-1,1});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,0,1});
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,-1,1});

    //single ion anisotropy
    Fe_atoms.set_onsite_interaction(K, 0);
    Fe_atoms.set_onsite_interaction(K, 1);
    Fe_atoms.set_onsite_interaction(K, 2);
    Fe_atoms.set_onsite_interaction(K, 3);

    Fe_atoms.set_field(fielddir*h, 0);
    Fe_atoms.set_field(fielddir*h, 1);
    Fe_atoms.set_field(fielddir*h, 2);
    Fe_atoms.set_field(fielddir*h, 3);

    //Tm atoms
    //Set energy splitting for Tm atoms
    //\alpha\lambda3 + \beta\lambda8 + \gamma\identity
    double alpha = e1/2;
    double beta = sqrt(3)/6*(2*e2-e1);
    double gamma = -(e1+e2)/3 *3/16;

    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 0);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 1);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 2);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 3);



    TmFeO3<3, 8> TFO(&Fe_atoms, &Tm_atoms);
    // I have finally cracked the correct model...
    // Here we go!

    if (chii != 0.0){
        array<array<double,3>,8> chi = {{{0}}};
        chi[1] = {{0, 0, 5.264*chii}};
        chi[4] = {{2.3915*chii,2.7866*chii,0}};
        chi[6] = {{0.9128*chii,-0.4655*chii,0}};

        array<array<double,3>,8> chi_inv = {{{0}}};
        chi_inv[1] = {{0, 0, 5.264*chii}};
        chi_inv[4] = {{-2.3915*chii,-2.7866*chii,0}};
        chi_inv[6] = {{-0.9128*chii,0.4655*chii,0}};

        TFO.set_mix_bilinear_interaction(chi, 1, 0, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 1, 3, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 1, 1, {0,1,0});
        TFO.set_mix_bilinear_interaction(chi, 1, 2, {0,1,0});


        TFO.set_mix_bilinear_interaction(chi_inv, 1, 2, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 3, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 1, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 0, {1,0,0});

        ///////////////
        TFO.set_mix_bilinear_interaction(chi, 0, 0, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 0, 3, {0,0,1});
        TFO.set_mix_bilinear_interaction(chi, 0, 1, {0,1,0});
        TFO.set_mix_bilinear_interaction(chi, 0, 2, {0,1,1});

        TFO.set_mix_bilinear_interaction(chi_inv, 0, 2, {-1,1,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 0, 3, {0,1,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 0, 1, {-1,1,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 0, 0, {0,1,0});

        ///////////////
        TFO.set_mix_bilinear_interaction(chi, 2, 0, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 2, 3, {0,0,1});
        TFO.set_mix_bilinear_interaction(chi, 2, 1, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 2, 2, {0,0,1});

        TFO.set_mix_bilinear_interaction(chi_inv, 2, 2, {0,1,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 3, {1,0,1});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 1, {0,1,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 0, {1,0,0});

        ///////////////
        TFO.set_mix_bilinear_interaction(chi, 3, 0, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi, 3, 3, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi, 3, 1, {0,0,0});
        TFO.set_mix_bilinear_interaction(chi, 3, 2, {0,0,0});

        TFO.set_mix_bilinear_interaction(chi_inv, 3, 2, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 3, {1,-1,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 1, {1,0,0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 0, {1,-1,0});
    }

    if (xii != 0.0){

        array<array<array<double,3>,3>,8> xi = {{{0}}};
        xi[0] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
        // xi[2] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
        // xi[7] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};

        ////////// Trilinear coupling/Oxygen path way
        TFO.set_mix_trilinear_interaction(xi, 1, 0, 3, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 1, 2, {0,1,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 2, 3, {0,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 1, 0, {0,0,0}, {1,0,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 1, 0, {0,1,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 2, 3, {0,1,0}, {1,0,0});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 2, 2, 3, {0,0,1}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 2, 2, 3, {0,1,1}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 2, 1, 2, {0,0,0}, {0,0,1});
        TFO.set_mix_trilinear_interaction(xi, 2, 0, 3, {1,0,0}, {1,0,1});
        //////////////////

        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,0,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,0,1}, {0,1,1});

        TFO.set_mix_trilinear_interaction(xi, 0, 1, 2, {-1,1,0}, {-1,1,1});
        TFO.set_mix_trilinear_interaction(xi, 0, 0, 3, {0,0,0}, {0,0,1});

        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,1,0}, {0,1,0});
        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,1,1}, {0,1,1});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,0,0});
        
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,-1,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,-1,0});

        TFO.set_mix_trilinear_interaction(xi, 3, 0, 3, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 2, {1,0,0}, {1,0,0});

        ///////////// Trilinear Interaction - Nearest neighbours

        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {1,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {1,0,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {1,0,1}, {0,0,1});
        TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {1,0,1}, {0,1,1});

        TFO.set_mix_trilinear_interaction(xi, 2, 0, 3, {0,0,0}, {0,0,1});
        TFO.set_mix_trilinear_interaction(xi, 2, 1, 2, {0,1,0}, {0,1,1});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,0,0}, {-1,1,0});
        TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,1,0}, {-1,1,0});

        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,0,1}, {-1,1,1});
        TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,1,1}, {-1,1,1});

        TFO.set_mix_trilinear_interaction(xi, 0, 0, 3, {0,1,0}, {0,1,1});
        TFO.set_mix_trilinear_interaction(xi, 0, 1, 2, {0,1,0}, {0,1,1});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {0,0,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {0,0,0}, {0,1,0});

        TFO.set_mix_trilinear_interaction(xi, 1, 0, 3, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 1, 2, 1, {0,0,0}, {0,0,0});

        //////////////////
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {1,0,0}, {1,-1,0});

        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {1,0,0}, {1,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {1,0,0}, {1,-1,0});

        TFO.set_mix_trilinear_interaction(xi, 3, 1, 2, {0,0,0}, {0,0,0});
        TFO.set_mix_trilinear_interaction(xi, 3, 0, 3, {1,-1,0}, {1,-1,0});
    }
    
    //
    cout << "Finished setting up TmFeO3 model." << endl;
    cout << "Starting calculations..." << endl;

    
    int trial_section = int(num_trials/size);

    for(size_t i = rank*trial_section; i < (rank+1)*trial_section; ++i){
        mixed_lattice_cuda<3, 4, 8, 4, 4, 4, 4> MC(&TFO, 2.5, 1.0);
        if (spin_config_filename != ""){
            MC.read_spin_from_file(spin_config_filename);
        }
        else{
            MC.simulated_annealing(Temp_start, Temp_end , 10000, 0, 100, false);
        }
        MC.molecular_dynamics_cuda(T_start, T_end, T_step_size, dir+"/"+std::to_string(i), 1, false, true);
    }
}


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    vector<int> rank_to_write = {0};

    // Defaults
    double J1ab = 4.92;
    double J1c = 4.92;
    double J2ab = 0.29;
    double J2c = 0.29;
    double Ka = 0.0;
    double Kc = -0.09;
    double D1 = 0.0;
    double D2 = 0.0;
    double chii = 0.05; // Tm-O coupling
    double xii = 0.0;
    double e1 = 2.2;
    double e2 = 4.8;
    double offset = 0.0;
    double h = 0.0;

    int num_trials = 1;
    double T_start = 0.0;
    double T_end = 50.0;
    double T_step_size = 1e-2;

    // Try to load from parameter file (argv[1] or ./params.txt). If not found, fall back to CLI args.
    std::unordered_map<std::string,double> par;
    std::string dir_name = (argc > 1) ? argv[1] : "TmFeO3_2DCS";
    string param_path = dir_name + "/params.txt";
    string spin_config_file = dir_name + "/spin0";


    bool loaded = load_params_from_file(param_path, par);
    if (!loaded && argc > 1) {
        // If first arg was not a path, try default params.txt
        loaded = load_params_from_file("params.txt", par);
    }
    if (loaded) {
        auto get = [&](const char* k, double &v){ auto it = par.find(k); if (it != par.end()) v = it->second; };
        get("J1ab", J1ab);
        get("J1c", J1c);
        get("J2ab", J2ab);
        get("J2c", J2c);
        get("Ka", Ka);
        get("Kc", Kc);
        get("D1", D1);
        get("D2", D2);
        get("chii", chii);
        get("xii", xii);
        get("e1", e1);
        get("e2", e2);
        get("h", h);
        get("T_start", T_start);
        get("T_end", T_end);
        get("T_step_size", T_step_size);
    } else {
        // Backward-compatible CLI parsing
        J1ab = (argc > 1) ? atof(argv[1]) : J1ab;
        J1c = (argc > 2) ? atof(argv[2]) : J1c;
        J2ab = (argc > 3) ? atof(argv[3]) : J2ab;
        J2c = (argc > 4) ? atof(argv[4]) : J2c;
        Ka = (argc > 5) ? atof(argv[5]) : Ka;
        Kc = (argc > 6) ? atof(argv[6]) : Kc;
        D1 = (argc > 7) ? atof(argv[7]) : D1;
        D2 = (argc > 8) ? atof(argv[8]) : D2;
        chii = (argc > 9) ? atof(argv[9]) : chii;
        xii = (argc > 10) ? atof(argv[10]) : xii;
        e1 = (argc > 11) ? atof(argv[11]) : e1;
        e2 = (argc > 12) ? atof(argv[12]) : e2;
        offset = (argc > 13) ? atof(argv[13]) : offset;
        h = (argc > 14) ? atof(argv[14]) : h;

        dir_name = (argc > 15) ? argv[15] : dir_name;
        num_trials = (argc > 16) ? atoi(argv[16]) : num_trials;
        T_start = (argc > 17) ? atof(argv[17]) : T_start;
        T_end = (argc > 18) ? atof(argv[18]) : T_end;
        T_step_size = (argc > 19) ? atof(argv[19]) : T_step_size;
        spin_config_file = (argc > 20) ? argv[20] : spin_config_file;
    }

    J1c /= J1ab;
    J2ab /= J1ab;
    J2c /= J1ab;
    Ka /= J1ab;
    Kc /= J1ab;
    D1 /= J1ab;
    D2 /= J1ab;
    xii /= J1ab;
    e1 /= J1ab;
    e2 /= J1ab;
    h /= J1ab;
    J1ab = 1;

    cout << "Begin MD on TmFeO3 with parameters:" << J1ab << " " << J1c << " " << J2ab << " " << J2c << " " << Ka << " " << Kc << " " << D1 << " " << D2 << " " << xii << " " << e1 << " " << e2 << " " << h << " " << dir_name << " " << num_trials << endl;
    filesystem::create_directory(dir_name);

    ofstream myfile;
    myfile.open(dir_name + "/parameters.txt");
    myfile << "J1ab: " << J1ab << endl;
    myfile << "J1c: " << J1c << endl;
    myfile << "J2ab: " << J2ab << endl;
    myfile << "J2c: " << J2c << endl;
    myfile << "Ka: " << Ka << endl;
    myfile << "Kc: " << Kc << endl;
    myfile << "D1: " << D1 << endl;
    myfile << "D2: " << D2 << endl;
    myfile << "chii: " << chii << endl;
    myfile << "xii: " << xii << endl;
    myfile << "e1: " << e1 << endl;
    myfile << "e2: " << e2 << endl;
    myfile << "h: " << h << endl;
    myfile << "dir_name: " << dir_name << endl;
    myfile << "num_trials: " << num_trials << endl;
    myfile << "T_start: " << T_start << endl;
    myfile << "T_end: " << T_end << endl;
    myfile << "T_step_size: " << T_step_size << endl;
    myfile << "spin_config_file: " << spin_config_file << endl;
    myfile.close();

    // MD_TmFeO3(num_trials, 20, 1e-2, T_start, T_end, T_step_size, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, xii, h, {0,1,0}, e1, e2, offset, dir_name, spin_config_file);
    MD_TmFeO3_cuda(num_trials, 20, 1e-2, T_start, T_end, T_step_size, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, chii, xii, h, {0,1,0}, e1, e2, offset, dir_name, spin_config_file);

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}
//J1ab=J1c=4.92meV J2ab=J2c=0.29meV Ka=0meV Kc=-0.09meV D1=D2=0 xii is the modeling param E1 = 0.97meV E2=3.9744792531meV