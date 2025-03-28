#include "experiments.h"


void  simulated_annealing_pyrochlore(double TargetT, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, double theta=0, bool theta_or_Jxz=true, bool save=false){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};

    z1 /= double(sqrt(3));
    z2 /= double(sqrt(3));
    z3 /= double(sqrt(3));
    z4 /= double(sqrt(3));


    array<double, 3> y1 = {0,1,-1};
    array<double, 3> y2 = {0,-1,1};
    array<double, 3> y3 = {0,-1,-1};
    array<double, 3> y4 = {0,1,1};
    y1 /= sqrt(2);
    y2 /= sqrt(2);
    y3 /= sqrt(2);
    y4 /= sqrt(2);

    array<double, 3> x1 = {-2,1,1};
    array<double, 3> x2 = {-2,-1,-1};
    array<double, 3> x3 = {2,1,-1};
    array<double, 3> x4 = {2,-1,1};
    x1 /= sqrt(6);
    x2 /= sqrt(6);
    x3 /= sqrt(6);
    x4 /= sqrt(6);
    double Jx, Jy, Jz, theta_in;
    cout << "Begin simulated annealing with parameters: " << Jxx << " " << Jyy << " " << Jzz << " " << theta_in << endl;
    if (theta_or_Jxz){

        Jx = Jxx;
        Jy = Jyy;
        Jz = Jzz;
        theta_in = theta;
    }
    else{
        Jy = Jyy;
        theta_in = atan(2*theta/(Jxx-Jzz))/2;
        Jx = cos(theta_in)*cos(theta_in)*Jxx + sin(theta_in)*sin(theta_in)*Jzz + sin(2*theta_in)*theta;
        Jz = sin(theta_in)*sin(theta_in)*Jxx + cos(theta_in)*cos(theta_in)*Jzz - sin(2*theta_in)*theta;
        cout << "Begin simulated annealing with parameters: " << Jx << " " << Jy << " " << Jz << " " << theta_in << endl;
        // Jx = (Jxx + Jzz)/2 - sqrt((Jxx-Jzz)*(Jxx-Jzz) + 4*theta*theta)/2;
        // Jz = (Jxx + Jzz)/2 + sqrt((Jxx-Jzz)*(Jxx-Jzz) + 4*theta*theta)/2;
        // cout << "Begin simulated annealing with parameters: " << Jx << " " << Jy << " " << Jz << " " << theta_in << endl;
        double maxJ = max(Jx, max(Jy, Jz));
        Jx /= maxJ;
        Jy /= maxJ;
        Jz /= maxJ;
    }
    cout << "Renormalized parameters: " << Jx << " " << Jy << " " << Jz << " " << theta_in << endl;
    array<array<double,3>, 3> J = {{{Jx,0,0},{0,Jy,0},{0,0,Jz}}};
    array<double, 3> field = field_dir*h;


    atoms.set_bilinear_interaction(J, 0, 1, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 0, 0}); 

    atoms.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 1, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 1}); 
    atoms.set_bilinear_interaction(J, 1, 2, {-1, 1, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {-1, 0, 1}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 1, -1}); 

    array<double, 3> rot_field = {gzz*sin(theta_in)+gxx*cos(theta_in),0,gzz*cos(theta_in)-gxx*sin(theta_in)};
    array<double, 3> By1 = {0, gyy*(pow(dot(field,y1),3) - 3*pow(dot(field,x1),2)*dot(field,y1)),0};
    array<double, 3> By2 = {0, gyy*(pow(dot(field,y2),3) - 3*pow(dot(field,x2),2)*dot(field,y2)),0};
    array<double, 3> By3 = {0, gyy*(pow(dot(field,y3),3) - 3*pow(dot(field,x3),2)*dot(field,y3)),0};
    array<double, 3> By4 = {0, gyy*(pow(dot(field,y4),3) - 3*pow(dot(field,x4),2)*dot(field,y4)),0};


    atoms.set_field(rot_field*dot(field, z1)+ By1, 0);
    atoms.set_field(rot_field*dot(field, z2)+ By2, 1);
    atoms.set_field(rot_field*dot(field, z3)+ By3, 2);
    atoms.set_field(rot_field*dot(field, z4)+ By4, 3);

    lattice<3, 4, 8, 8, 8> MC(&atoms, 0.5);
    // MC.simulated_annealing_deterministic(5, 1e-7, 10000, 10000, 0, dir);
    MC.simulated_annealing(5, 1e-4, 1e4, 0, true, dir, save);
}

void parallel_tempering_pyrochlore(double T_start, double T_end, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, const vector<int> &rank_to_write, double Jxz=0){
    filesystem::create_directory(dir);
    Pyrochlore<3> atoms;

    array<double,3> z1 = {1, 1, 1};
    array<double,3> z2 = {1,-1,-1};
    array<double,3> z3 = {-1,1,-1};
    array<double,3> z4 = {-1,-1,1};

    z1 /= double(sqrt(3));
    z2 /= double(sqrt(3));
    z3 /= double(sqrt(3));
    z4 /= double(sqrt(3));


    array<double, 3> y1 = {0,1,-1};
    array<double, 3> y2 = {0,-1,1};
    array<double, 3> y3 = {0,-1,-1};
    array<double, 3> y4 = {0,1,1};
    y1 /= sqrt(2);
    y2 /= sqrt(2);
    y3 /= sqrt(2);
    y4 /= sqrt(2);

    array<double, 3> x1 = {-2,1,1};
    array<double, 3> x2 = {-2,-1,-1};
    array<double, 3> x3 = {2,1,-1};
    array<double, 3> x4 = {2,-1,1};
    x1 /= sqrt(6);
    x2 /= sqrt(6);
    x3 /= sqrt(6);
    x4 /= sqrt(6);


    double Jz_Jx = Jzz-Jxx;
    double Jz_Jz_sign = (Jzz-Jxx < 0) ? -1 : 1;
    double Jx = (Jxx+Jzz)/2 - Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*Jxz*Jxz)/2; 
    double Jy = Jyy;
    double Jz = (Jxx+Jzz)/2 + Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*Jxz*Jxz)/2; 
    double theta = atan(2*Jxz/(Jzz-Jxx));

    array<array<double,3>, 3> J = {{{Jx,0,0},{0,Jy,0},{0,0,Jz}}};
    array<double, 3> field = field_dir*h;


    atoms.set_bilinear_interaction(J, 0, 1, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 2, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {0, 0, 0}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 0, 0}); 

    atoms.set_bilinear_interaction(J, 0, 1, {1, 0, 0}); 
    atoms.set_bilinear_interaction(J, 0, 2, {0, 1, 0}); 
    atoms.set_bilinear_interaction(J, 0, 3, {0, 0, 1}); 
    atoms.set_bilinear_interaction(J, 1, 2, {-1, 1, 0}); 
    atoms.set_bilinear_interaction(J, 1, 3, {-1, 0, 1}); 
    atoms.set_bilinear_interaction(J, 2, 3, {0, 1, -1}); 

    array<double, 3> rot_field = {gzz*sin(theta)+gxx*cos(theta),0,gzz*cos(theta)-gxx*sin(theta)};
    array<double, 3> By1 = {0, gyy*(pow(dot(field,y1),3) - 3*pow(dot(field,x1),2)*dot(field,y1)),0};
    array<double, 3> By2 = {0, gyy*(pow(dot(field,y2),3) - 3*pow(dot(field,x2),2)*dot(field,y2)),0};
    array<double, 3> By3 = {0, gyy*(pow(dot(field,y3),3) - 3*pow(dot(field,x3),2)*dot(field,y3)),0};
    array<double, 3> By4 = {0, gyy*(pow(dot(field,y4),3) - 3*pow(dot(field,x4),2)*dot(field,y4)),0};


    atoms.set_field(rot_field*dot(field, z1)+ By1, 0);
    atoms.set_field(rot_field*dot(field, z2)+ By2, 1);
    atoms.set_field(rot_field*dot(field, z3)+ By3, 2);
    atoms.set_field(rot_field*dot(field, z4)+ By4, 3);

    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    lattice<3, 4, 16, 16, 16> MC(&atoms, 0.5);

    vector<double> temps = logspace(log10(T_start), log10(T_end), size);

    MC.parallel_tempering(temps, 1e6, 1e6, 10, 50, 2e3, dir, rank_to_write, true);

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}

int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    // MD_TmFeO3(1, -1.0, -0.06, "test_L=12");
    // MD_kitaev_honeycomb(1, -1.0, 0.0, -0.0, 0.06, "Pure_Kitaev_h=0.06");
    // string dir = "pure_kitaev_2DCS_h=0.7_pulse_amp=0.1/";
    // full_nonlinearspectroscopy_kitaev_honeycomb(1, 1, 1e-4, 0, -600, 0.25, -600, 600, 0.25, -1.0, 0.25, -0.02, 0.7, dir, true);
    // array<double, 3> field_in = {0,0,1};
    // MD_pyrochlore(1, 0.062/0.063, 1.0, 0.011/0.063, 0, 0, 2.24, 6, field_in*mu_B/0.063, "CZO_h=6T_001_theta=0.0", 0.0);
    // TmFeO3_2DCS(1,1000*k_B, 2*k_B, 0, -200/4.625, 0.05/4.625, -200/4.625, 200/4.625, 0.05/4.625, 4.625, 4.625, 4.625, 0.158, 0.158, 0.158, 0, -0.023, 0.0, 0.0, 0.0, {0,0,1}, "TmFeO3_Fe_Magnon_MD_real_meV", true, "TmFeO3_spin_config.txt");
    // MD_TmFeO3_Fe(1, 1000*k_B, 2*k_B, 4.625, 4.625, 4.625, 0.158, 0.158, 0.158, 0, -0.023, 0.0, 0.0, 0, {0,0,1}, "TmFeO3_Fe_Magnon_MD_real_meV");
    // MD_TmFeO3_2DCS(1000*k_B, 2*k_B, 0, -200/4.625, 0.2/4.625, -200/4.625, 200/4.625, 0.2/4.625, 4.625, 4.625, 4.625, 0.158, 0.158, 0.158, 0, -0.023, 0.0, 0.0, 0, {0,0,1}, "TmFeO3_Fe_Magnon_2DCS_real_meV");
    // simulated_annealing_honeycomb(1, 1e-6, -1, 0.25, -0.02, 0.7, "test_simulated_annealing");
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<int> rank_to_write = {0};
    // parallel_tempering_honeycomb(1, 1e-6, -1, 0.25, -0.02, 0.7, "test_parallel", rank_to_write);
    double Jpm_start = argv[1] ? atof(argv[1]) : 0.0;
    double Jpm_end = argv[2] ? atof(argv[2]) : 0.0;
    int num_Jpm = argv[3] ? atoi(argv[3]) : 0;
    double Jpmpm = argv[4] ? atof(argv[4]) : 0.0;
    double h_min = argv[5] ? atof(argv[5]) : 0.0;
    double h_max = argv[6] ? atof(argv[6]) : 0.0;
    double num_H = argv[7] ? atoi(argv[7]) : 0;
    string dir_string = argv[8] ? argv[8] : "001";
    array<double, 3> field_dir;
    if (dir_string == "001"){
        field_dir = {0,0,1};
    }else if(dir_string == "110"){
        field_dir = {1/sqrt(2), 1/sqrt(2), 0};
    }else if(dir_string == "1-10"){
        field_dir = {1/sqrt(2), -1/sqrt(2), 0};
    }else{
        field_dir = {1/sqrt(3),1/sqrt(3),1/sqrt(3)};
    }
    int SLURM_TASK_ID = argv[9] ? atoi(argv[9]) : 0;
    int Jpm_ind = SLURM_TASK_ID % num_Jpm;
    int h_ind = SLURM_TASK_ID / num_Jpm;
    double Jpm = Jpm_start + Jpm_ind*(Jpm_end-Jpm_start)/num_Jpm;
    double h = h_min + h_ind*(h_max-h_min)/num_H;
    string dir_name = argv[10] ? argv[10] : "";
    filesystem::create_directory(dir_name);
    string sub_dir = dir_name + "/Jpm_" + std::to_string(Jpm) + "_h_" + std::to_string(h) + "_" + std::to_string(Jpm_ind) + "_" + std::to_string(h_ind);
    int MPI_n_tasks = argv[11] ? atoi(argv[11]) : 1;
    std::cout << "Initializing parallel tempering calculation with parameters: " << "T = " << 10 << "-" << 1e-3 << " Jpm: " << Jpm << " Jpmpm: " << Jpmpm << " H: " << h << " field direction : " << dir_string << " saving to: " << dir_name << endl;
    parallel_tempering_pyrochlore(1e-3, 10, -2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0, 0, 1, h, field_dir, sub_dir, {0});
    // simulated_annealing_pyrochlore(-0.4, 1, 0.4, 0, 0, 1, 0, {0,0,1}, "test");

    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.2, {0,0,1}, "MC_phase_diagram_CZO_001");
    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.0, {1,1,1}, "MC_phase_diagram_CZO_111");
    // phase_diagram_pyrochlore(-0.3, 0.3, 70, 0.0, 3.0, 30, 0.0, {1,1,0}, "MC_phase_diagram_CZO_110");

    // MD_pyrochlore(1, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0, 0, 2.24, 1.5*mu_B/0.063, {1/sqrt(2), 1/sqrt(2), 0}, "pyrochlore_test_110");
    // MD_pyrochlore(20, 0.062/0.063, 0.063/0.063, 0.011/0.063, 0, 0, 2.24, 1.5*mu_B/0.063, {0,0,1}, "pyrochlore_test_001");
    // std::cout << "finished" << std::endl;   
    return 0;
}