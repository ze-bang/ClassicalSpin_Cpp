#include "experiments.h"

void simulated_annealing_TmFeO3(double T_start, double T_end, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double e1, double e2, double xii, double h, const array<double,3> &fielddir, string dir){
    filesystem::create_directories(dir);
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
    double gamma = -(e1+e2)/3;

    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 0);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 1);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 2);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 3);



    TmFeO3<3, 8> TFO(&Fe_atoms, &Tm_atoms);
    // I have finally cracked the correct model...
    // Here we go!

    if (chii != 0.0){
        // Actually, for any non-linear response, one must also consider
        // some type of DM interaction between the SU(2) and SU(3) spin.
        // The idea is essentially, for the ordered phase, clearly we have
        // a condensate of lambda3 and lambda8. As such, these are the ordered
        // moment to expand upon. As such, a bilinear term can only yield a 
        // bilinear term S\lambda or a quartic term like SS\lambda\lambda.
        // So what do we do here? Well, either we want coupling of the form
        // S lambda3 or S lambda 8. But these operators are time reversal even
        // and inversion even.
        // So what about a trilinear interaction like SS\lambda?
        // Well in this case then the only lambda operator allowed to couple is 
        // \lambda1,3,and 8. So therefore, we can only consider \lambda1.
        // Then we must consider the process: Namely, \lambda1 only has |E0><E1|
        // Therefore, we must have a bilinear term in \lambda that contains |E1><E2|
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
    
    cout << "Finished setting up TmFeO3" << endl;

    mixed_lattice<3, 4, 8, 4, 8, 8, 8> MC(&TFO, 2.5, 1.0);
    MC.simulated_annealing(T_start, T_end, 10000, 0, 100, true, dir);

    for (size_t i = 0; i < 1e5; ++i) {
        MC.deterministic_sweep();
    }

    // Write the zero temperature spin configuration
    cout << "Writing zero temperature spin configuration to " << dir + "/Tzero" << endl;
    MC.write_to_file_spin(dir + "/spin_zero.txt");

}

int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    std::vector<int> rank_to_write = {{0}};
    double J1ab = argv[1] ? atof(argv[1]) : 0.0;
    double J1c = argv[2] ? atof(argv[2])/J1ab : 0.0;
    double J2ab = argv[3] ? atof(argv[3])/J1ab  : 0.0;
    double J2c = argv[4] ? atof(argv[4])/J1ab  : 0.0;
    double Ka = argv[5] ? atof(argv[5])/J1ab  : 0.0;
    double Kc = argv[6] ? atof(argv[6])/J1ab  : 0.0;
    double D1 = argv[7] ? atof(argv[7])/J1ab  : 0.0;
    double D2 = argv[8] ? atof(argv[8])/J1ab  : 0.0;
    double xii = argv[9] ? atof(argv[9])/J1ab  : 0.0;
    double e1 = argv[10] ? atof(argv[10])/J1ab  : 0.0;
    double e2 = argv[11] ? atof(argv[11])/J1ab  : 0.0;
    double h = argv[12] ? atof(argv[12])/J1ab  : 0.0;
    J1ab = 1;
    string dir_name = argv[13] ? argv[13] : "";
    double T_start = argv[14] ? atof(argv[14]) : 0.0;
    double T_end = argv[15] ? atof(argv[15]) : 0.0;
    cout << "Begin simulated annealing on TmFeO3 with parameters:" << J1ab << " " << J1c << " " << J2ab << " " << J2c << " " << Ka << " " << Kc << " " << D1 << " " << D2 << " " << xii << " " << e1 << " " << e2 << " " << h << " " << dir_name << endl;

    simulated_annealing_TmFeO3(T_start, T_end, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, e1, e2, xii, h, {0,0,1}, dir_name);
}