#include "experiments.h"

void simulated_annealing_TmFeO3_Fe(int num_trials, double T_start, double T_end, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double h, const array<double,3> &fielddir, string dir, bool T_zero=false){
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
        lattice<3, 4, 12, 12, 12> MC_FE(&Fe_atoms, 2.5);
        MC_FE.simulated_annealing(T_start, T_end, 10000, 0, false);
        if (T_zero){
            // std::random_device rd;
            // std::mt19937 gen(rd());
            for(size_t i = 0; i<100000; ++i){
                MC_FE.deterministic_sweep();
            }   
        }
        MC_FE.write_to_file_pos(dir+"/pos.txt");
        MC_FE.write_to_file_spin(dir+"/spin"+ to_string(i) +".txt", MC_FE.spins);
    }
}

void simulated_annealing_TmFeO3(int num_trials, double Temp_start, double Temp_end, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double e1, double e2, double xii, double h, const array<double,3> &fielddir, string dir, bool T_zero=false){
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

    //Tm atoms
    Tm_atoms.set_field({0,0,e1,0,0,0,0,e2}, 0);
    Tm_atoms.set_field({0,0,e1,0,0,0,0,e2}, 1);
    Tm_atoms.set_field({0,0,e1,0,0,0,0,e2}, 2);
    Tm_atoms.set_field({0,0,e1,0,0,0,0,e2}, 3);


    TmFeO3<3, 8> TFO(&Fe_atoms, &Tm_atoms);

    array<array<array<double,3>,3>,8> xi = {{{0}}};

    xi[0] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};
    xi[1] = {{{xii,0,0},{0,xii,0},{0,0,xii}}};

    ///////////////////
    TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {1,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {0,0,0}, {0,1,0});
    TFO.set_mix_trilinear_interaction(xi, 2, 0, 1, {1,0,0}, {0,1,0});

    TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {0,0,1}, {0,0,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {1,0,1}, {0,0,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {0,0,1}, {0,1,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 3, 2, {1,0,1}, {0,1,1});

    TFO.set_mix_trilinear_interaction(xi, 2, 0, 3, {0,0,0}, {0,0,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 0, 3, {1,0,0}, {1,0,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 1, 2, {0,0,0}, {0,0,1});
    TFO.set_mix_trilinear_interaction(xi, 2, 1, 2, {0,1,0}, {0,1,1});
    //////////////////
    TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,0,0}, {0,1,0});
    TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,1,0}, {0,1,0});
    TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,0,0}, {-1,1,0});
    TFO.set_mix_trilinear_interaction(xi, 0, 0, 1, {0,1,0}, {-1,1,0});

    TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,0,1}, {0,1,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,1,1}, {0,1,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,0,1}, {-1,1,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 3, 2, {0,1,1}, {-1,1,1});

    TFO.set_mix_trilinear_interaction(xi, 0, 0, 3, {0,0,0}, {0,0,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 0, 3, {0,1,0}, {0,1,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 1, 2, {0,1,0}, {0,1,1});
    TFO.set_mix_trilinear_interaction(xi, 0, 1, 2, {-1,1,0}, {-1,1,1});
    //////////////////
    TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {1,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {0,0,0}, {0,1,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 3, 2, {1,0,0}, {0,1,0});

    TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {1,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {0,0,0}, {0,1,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 0, 1, {1,0,0}, {0,1,0});

    TFO.set_mix_trilinear_interaction(xi, 1, 0, 3, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 0, 3, {1,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 2, 1, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 1, 2, 1, {0,1,0}, {0,1,0});
    //////////////////
    TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {1,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {0,0,0}, {1,-1,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 1, 0, {1,0,0}, {1,-1,0});

    TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {1,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {0,0,0}, {1,-1,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 2, 3, {1,0,0}, {1,-1,0});

    TFO.set_mix_trilinear_interaction(xi, 3, 1, 2, {0,0,0}, {0,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 1, 2, {1,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 0, 3, {1,0,0}, {1,0,0});
    TFO.set_mix_trilinear_interaction(xi, 3, 0, 3, {1,-1,0}, {1,-1,0});

    for(size_t i = 0; i < num_trials; ++i){
        mixed_lattice<3, 4, 8, 4, 12, 12, 12> MC(&TFO, 2.5, 1.0);
        MC.simulated_annealing(Temp_start, Temp_end, 10000, 0, 0, true);
        if (T_zero){
            for(size_t i = 0; i<100000; ++i){
                MC.deterministic_sweep();
            }
        }
        MC.write_to_file_pos(dir+"/pos");
        MC.write_to_file_spin(dir+"/spin"+ to_string(i));
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
    vector<int> rank_to_write = {size-1};
    double J1ab = argv[1] ? atof(argv[1]) : 0.0;
    double J1c = argv[2] ? atof(argv[2])/J1ab : 0.0;
    double J2ab = argv[3] ? atof(argv[3])/J1ab : 0.0;
    double J2c = argv[4] ? atof(argv[4])/J1ab : 0.0;
    double Ka = argv[5] ? atof(argv[5])/J1ab : 0.0;
    double Kc = argv[6] ? atof(argv[6])/J1ab : 0.0;
    double D1 = argv[7] ? atof(argv[7])/J1ab : 0.0;
    double D2 = argv[8] ? atof(argv[8])/J1ab : 0.0;
    double e1 = argv[9] ? atof(argv[9])/J1ab : 0.0;
    double e2 = argv[10] ? atof(argv[10])/J1ab : 0.0;
    double xii = argv[11] ? atof(argv[11])/J1ab : 0.0;
    double h = argv[12] ? atof(argv[12])/J1ab : 0.0;
    J1ab = 1;
    string dir_name = argv[13] ? argv[13] : "";
    int T_zero = argv[14] ? atoi(argv[14]) : 0;
    int num_trials = argv[15] ? atoi(argv[15]) : 0;
    // simulated_annealing_TmFeO3_Fe(num_trials, 20, 1e-2, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, h, {1,0,0}, dir_name, T_zero);
    simulated_annealing_TmFeO3(num_trials, 20, 1e-2, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, e1, e2, xii, h, {1,0,0}, dir_name, T_zero);
    return 0;
}
//J1ab=J1c=4.92meV J2ab=J2c=0.29meV Ka=0meV Kc=-0.09meV D1=D2=0 xii is the modeling param E1 = 0.97meV E2=3.89134081434meV