#include "experiments.h"
void MD_pyrochlore(size_t num_trials, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, double theta=0, bool theta_or_Jxz=false){
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
    if (theta_or_Jxz){
        Jx = Jxx;
        Jy = Jyy;
        Jz = Jzz;
        theta_in = theta;
        cout << "Hi" << endl;
    }
    else{
        double Jz_Jx = Jzz-Jxx;
        double Jz_Jz_sign = (Jzz-Jxx < 0) ? -1 : 1;
        Jx = (Jxx+Jzz)/2 - Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*theta*theta)/2; 
        Jy = Jyy;
        Jz = (Jxx+Jzz)/2 + Jz_Jz_sign*sqrt(Jz_Jx*Jz_Jx+4*theta*theta)/2; 
        theta_in = atan(2*theta/(Jzz-Jxx))/2;
        double maxJ = max(Jx, max(Jy, Jz));
        Jx /= maxJ;
        Jy /= maxJ;
        Jz /= maxJ;
    }
    cout << Jx << " " << Jy << " " << Jz << " " << theta << endl;

    array<array<double,3>, 3> J = {{{Jxx,0,0},{0,Jyy,0},{0,0,Jzz}}};
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

    // array<double, 3> temp = rot_field*dot(field, z1)+ By1;
    // cout << gxx << " " << gyy << " " << gzz << " " << temp[0] << " " << temp[1] << " " << temp[2] << endl;
    // array<double, 3> temp1 = rot_field*dot(field, z2)+ By2;
    // cout << gxx << " " << gyy << " " << gzz << " " << temp1[0] << " " << temp1[1] << " " << temp1[2] << endl;
    // array<double, 3> temp2 = rot_field*dot(field, z3)+ By3;
    // cout << gxx << " " << gyy << " " << gzz << " " << temp2[0] << " " << temp2[1] << " " << temp2[2] << endl;
    // array<double, 3> temp3 = rot_field*dot(field, z4)+ By4;
    // cout << gxx << " " << gyy << " " << gzz << " " << temp3[0] << " " << temp3[1] << " " << temp3[2] << endl;

    atoms.set_field(rot_field*dot(field, z1)+ By1, 0);
    atoms.set_field(rot_field*dot(field, z2)+ By2, 1);
    atoms.set_field(rot_field*dot(field, z3)+ By3, 2);
    atoms.set_field(rot_field*dot(field, z4)+ By4, 3);

    int initialized;

    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start = rank*num_trials/size;
    int end = (rank+1)*num_trials/size;
    double k_B = 0.08620689655;
    for(int i=start; i<end;++i){
        lattice<3, 4, 8, 8, 8> MC(&atoms, 0.5);
        MC.simulated_annealing(5, 1e-3, 1e4, 10, true);
        MC.molecular_dynamics(0, 600, 0.25, dir+"/"+std::to_string(i));
        for(int i=0; i<1e6; ++i){
            MC.deterministic_sweep();
        }
        if(dir != ""){
            filesystem::create_directory(dir);
            ofstream myfile;
            myfile.open(dir+"/"+std::to_string(i)+"/spin_0.txt");
            for(size_t i = 0; i<MC.lattice_size; ++i){
                for(size_t j = 0; j<3; ++j){
                    myfile << MC.spins[i][j] << " ";
                }
                myfile << endl;
            }
            myfile.close();
        }
    }
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
}


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;

    double Jxx = argv[1] ? atof(argv[1]) : 0.0;
    double Jyy = argv[2] ? atof(argv[2]) : 0.0;
    double Jzz = argv[3] ? atof(argv[3]) : 0.0;
    double h = argv[4] ? atof(argv[4]) : 0.0;
    string dir_string = argv[5] ? argv[5] : "001";
    double Jxz = argv[6] ? atof(argv[6]) : 0.0;
    array<double, 3> field_dir;
    if (dir_string == "001"){
        field_dir = {0,0,1};
    }else if(dir_string == "110"){
        field_dir = {1/sqrt(2), 1/sqrt(2), 0};
    }else if(dir_string == "1-10"){
        field_dir = {1/sqrt(2), -1/sqrt(2), 0};
    }
    else{
        field_dir = {1/sqrt(3),1/sqrt(3),1/sqrt(3)};
    }
    string dir_name = argv[7] ? argv[7] : "";
    filesystem::create_directory(dir_name);
    int num_trials = argv[8] ? atoi(argv[8]) : 1;
    std::cout << "Initializing molecular dynamic calculation with parameters: Jxx: " << Jxx << " Jyy: " << Jyy << " Jzz: " << Jzz << " Jxz: " << Jxz << " H: " << h << " field direction : " << dir_string << " saving to: " << dir_name << endl;
    MD_pyrochlore(num_trials, Jxx, Jyy, Jzz, 0.0, 0.0, 1, h, field_dir, dir_name, Jxz);
    return 0;
}