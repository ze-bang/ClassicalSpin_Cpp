#include "experiments.h"

void parallel_tempering_TmFeO3(double T_start, double T_end, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double e1, double e2, double xii, double h, const array<double,3> &fielddir, string dir, const vector<int> &rank_to_write){
        filesystem::create_directories(dir);
    TmFeO3_Fe<3> Fe_atoms;
    TmFeO3_Tm<8> Tm_atoms;

    array<array<double, 3>, 3> Ja = {{{Jai, D2, -D1}, {-D2, Jai, 0}, {D1, 0, Jai}}};
    array<array<double, 3>, 3> Jb = {{{Jbi, D2, -D1}, {-D2, Jbi, 0}, {D1, 0, Jbi}}};
    array<array<double, 3>, 3> Jc = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};

    array<array<double, 3>, 3> J2a = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    array<array<double, 3>, 3> J2b = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    array<array<double, 3>, 3> J2c = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};

    array<double, 9> K = {{Ka, 0, 0, 0, 0, 0, 0, 0, Kc}};
    //In plane interactions

    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 1, 0, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 1, 0, {1,-1,0});

    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {0,0,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {0,-1,0});
    Fe_atoms.set_bilinear_interaction(Jb, 2, 3, {1,0,0});
    Fe_atoms.set_bilinear_interaction(Ja, 2, 3, {1,-1,0});

    //Next Nearest Neighbour
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

    Fe_atoms.set_field(fielddir*h, 0);
    Fe_atoms.set_field(fielddir*h, 1);
    Fe_atoms.set_field(fielddir*h, 2);
    Fe_atoms.set_field(fielddir*h, 3);

    //Tm atoms
    //Set energy splitting for Tm atoms
    //\alpha\lambda3 + \beta\lambda8 + \gamma\identity
    double alpha = -e1/2;
    double beta = -sqrt(3)/6*(2*e2-e1);
    double gamma = (e1+e2)/3 *3/16;

    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 0);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 1);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 2);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 3);



    TmFeO3<3, 8> TFO(&Fe_atoms, &Tm_atoms);

    if (xii != 0.0){

        array<array<array<double,3>,3>,8> xi = {{{0}}};
        xi[0] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
        xi[1] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
        xi[2] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
        xi[7] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
        
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

    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<double> temps = logspace(log10(T_start), log10(T_end), size);


    mixed_lattice<3, 4, 8, 4, 4, 4, 4> MC(&TFO, 2.5, 1.0);
    MC.parallel_tempering(temps, 4e5, 1e3, 1000, 50, 2e3, dir, rank_to_write);

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
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
    cout << "Begin parallel tempering on TmFeO3 with parameters:" << J1ab << " " << J1c << " " << J2ab << " " << J2c << " " << Ka << " " << Kc << " " << D1 << " " << D2 << " " << xii << " " << e1 << " " << e2 << " " << h << " " << dir_name << endl;
    cout << "T_start: " << T_start << " T_end: " << T_end << endl;
    parallel_tempering_TmFeO3(T_start, T_end, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, e1, e2, xii, h, {0,1,0}, dir_name, rank_to_write);    
    return 0;
}