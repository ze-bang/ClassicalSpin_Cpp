#include "experiments.h"

void pyrochlore_2DCS(size_t num_trials, bool T_zero, double Temp_start, double Temp_end, double tau_start, double tau_end, double tau_step_size, double T_start, double T_end, double T_step_size, array<double, 3> field_extern, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, double Jxz=0, string spin_config=""){
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

    array<array<double, 3>,4> field_drive = {{{0,0,dot(field, z1)},{0,0,dot(field, z2)},{0,0,dot(field, z3)},{0,0,dot(field, z4)}}};

    double pulse_amp = 0.1;
    double pulse_width = 0.38;
    double pulse_freq = 0.33;

    int T_steps = abs(int((T_end-T_start)/T_step_size))+1;
    int tau_steps = abs(int((tau_end-tau_start)/tau_step_size))+1;
    tau_step_size = tau_end - tau_start < 0 ? - abs(tau_step_size) : abs(tau_step_size);
    T_step_size = T_end - T_start < 0 ? - abs(T_step_size) : abs(T_step_size);
    lattice<3, 4, 12, 12, 12> MC(&atoms, 0.5);
    if (spin_config != ""){
        MC.read_spin_from_file(spin_config);
    }else{
        MC.simulated_annealing(Temp_start, Temp_end, 10000, 0, true);
        if (T_zero){
            for (size_t i = 0; i<100000; ++i){
                MC.deterministic_sweep();
            }
        }
    }
    MC.write_to_file_pos(dir+"/pos.txt");
    MC.write_to_file_spin(dir+"/spin_0.txt", MC.spins);

    filesystem::create_directory(dir+"/M_time_0");
    MC.M_B_t(field_drive, 0.0, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_0/M0");

    ofstream run_param;
    run_param.open(dir + "/param.txt");
    run_param << tau_start << " " << tau_end << " " << tau_steps  << " " << T_start << " " << T_end << " " << T_steps << endl;
    run_param.close();

    double current_tau = tau_start;

    for(int i=0; i< tau_steps;++i){
        filesystem::create_directory(dir+"/M_time_"+ std::to_string(i));
        cout << "Time: " << current_tau << endl;
        MC.M_B_t(field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i) + "/M1");
        MC.M_BA_BB_t(field_drive, 0.0, field_drive, current_tau, pulse_amp, pulse_width, pulse_freq, T_start, T_end, T_step_size, dir+"/M_time_"+ std::to_string(i)+ "/M01");
        current_tau += tau_step_size;
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
    bool T_zero = argv[1] ? atoi(argv[1]) : 0;
    double Temp_start = argv[2] ? atof(argv[2]) : 0.0;
    double Temp_end = argv[3] ? atof(argv[3]) : 0.0;
    double tau_start = argv[4] ? atof(argv[4]) : 0.0;
    double tau_end = argv[5] ? atof(argv[5]) : 0.0;
    double tau_step_size = argv[6] ? atof(argv[6]) : 0.0;
    double T_start = argv[7] ? atof(argv[7]) : 0.0;
    double T_end = argv[8] ? atof(argv[8]) : 0.0;
    double T_step_size = argv[9] ? atof(argv[9]) : 0.0;
    string field_extern_string = argv[10] ? argv[10] : "001";
    double Jpm = argv[11] ? atof(argv[11]) : 0.0;
    double Jpmpm = argv[12] ? atof(argv[12]) : 0.0;
    double h = argv[13] ? atof(argv[13]) : 0.0;
    string dir_string = argv[14] ? argv[14] : "001";
    double theta = argv[15] ? atof(argv[15]) : 0.0;
    array<double, 3> field_dir;
    array<double, 3> field_extern;
    if (dir_string == "001"){
        field_dir = {0,0,1};
    }else if(dir_string == "110"){
        field_dir = {1/sqrt(2), 1/sqrt(2), 0};
    }else{
        field_dir = {1/sqrt(3),1/sqrt(3),1/sqrt(3)};
    }

    if (field_extern_string == "001"){
        field_extern = {0,0,1};
    }else if(field_extern_string == "110"){
        field_extern = {1/sqrt(2), 1/sqrt(2), 0};
    }else{
        field_extern = {1/sqrt(3),1/sqrt(3),1/sqrt(3)};
    }

    string dir_name = argv[16] ? argv[16] : "";
    filesystem::create_directory(dir_name);
    int num_trials = argv[17] ? atoi(argv[17]) : 1;
    std::cout << "Initializing 2DCS calculation with parameters: Jpm: " << Jpm << " Jpmpm: " << Jpmpm << " H: " << h << " field direction : " << dir_string << " with angle:" << theta << "driven by field in direction " << field_extern_string << " saving to: " << dir_name << endl;
    pyrochlore_2DCS(num_trials, T_zero, Temp_start, Temp_end, tau_start, tau_end, tau_step_size, T_start, T_end, T_step_size, field_extern, -2*Jpm - 2*Jpmpm, 1, -2*Jpm + 2*Jpmpm, 0, 0, 1, h, field_dir, dir_name, theta);
    return 0;
}