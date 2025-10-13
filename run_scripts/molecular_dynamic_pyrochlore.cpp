#include "experiments.h"

void MD_pyrochlore(double T_target, size_t num_trials, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, double h, array<double, 3> field_dir, string dir, double theta=0, bool theta_or_Jxz=false, bool field_scan=false){
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
        MC.simulated_annealing(5, T_target, 1e5, 10, true);
        MC.molecular_dynamics(0, 600, 0.01, dir+"/"+std::to_string(i));
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
    
    // Only call barrier if not in field_scan mode
    if (!field_scan) {
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized){
            MPI_Finalize();
        }
    }
}

void magnetic_field_scan(double T_target, double Jxx, double Jyy, double Jzz, double gxx, double gyy, double gzz, 
                        size_t num_steps, double h_start, double h_end, array<double, 3> field_dir, 
                        string dir, double theta=0, bool theta_or_Jxz=false) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<double> h_values;
    if (num_steps > 0) {
        double step = (h_end - h_start) / num_steps;
        for (size_t i = 0; i <= num_steps; ++i) {
            h_values.push_back(h_start + i * step);
        }
    } else {
        h_values.push_back(h_start);
    }

    // Distribute field values across MPI processes
    for (size_t i = rank; i < h_values.size(); i += size) {
        double h = h_values[i];
        string subdir = dir + "/h_" + to_string(h);
        filesystem::create_directory(subdir);
        
        if (rank == 0 || true) {  // All ranks can print their assigned work
            std::cout << "Running simulation for h = " << h << " on process " << rank << std::endl;
        }
        
        // Run single trial (no multiple trials in field scan mode)
        MD_pyrochlore(T_target, 1, Jxx, Jyy, Jzz, gxx, gyy, gzz, h, field_dir, subdir, theta, theta_or_Jxz, true);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;

    // Check for --help or --field-scan flags
    bool do_field_scan = false;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [options] Jxx Jyy Jzz h dir_string Jxz dir_name num_trials T_target\n\n";
            cout << "Options:\n";
            cout << "  --help, -h          Show this help message\n";
            cout << "  --field-scan        Perform a magnetic field scan from h_start to h_end\n";
            cout << "                      When using --field-scan, provide h_start h_end num_steps instead of h\n\n";
            cout << "Arguments:\n";
            cout << "  Jxx                 Exchange parameter Jxx\n";
            cout << "  Jyy                 Exchange parameter Jyy\n";
            cout << "  Jzz                 Exchange parameter Jzz\n";
            cout << "  h (or h_start)      Magnetic field strength (or starting field for scan)\n";
            cout << "  dir_string          Field direction: '001', '110', '1-10', or '111'\n";
            cout << "  Jxz                 Exchange parameter Jxz\n";
            cout << "  dir_name            Output directory name\n";
            cout << "  num_trials          Number of trials (ignored in field-scan mode)\n";
            cout << "  T_target            Target temperature\n";
            cout << "  h_end               (field-scan only) Ending field strength\n";
            cout << "  num_steps           (field-scan only) Number of field steps\n";
            return 0;
        } else if (arg == "--field-scan") {
            do_field_scan = true;
        }
    }

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
    double T_target = argv[9] ? atof(argv[9]) : 0.0;
    
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(&argc, &argv);
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (do_field_scan) {
        double h_start = h;  // First argument is now h_start
        double h_end = argv[10] ? atof(argv[10]) : h_start;
        int num_steps = argv[11] ? atoi(argv[11]) : 0;
        
        if (rank == 0) {
            std::cout << "Initializing magnetic field scan with parameters:\n";
            std::cout << "  Jxx: " << Jxx << " Jyy: " << Jyy << " Jzz: " << Jzz << " Jxz: " << Jxz << "\n";
            std::cout << "  Field range: " << h_start << " to " << h_end << " in " << num_steps << " steps\n";
            std::cout << "  Field direction: " << dir_string << "\n";
            std::cout << "  Temperature: " << T_target << "\n";
            std::cout << "  Output directory: " << dir_name << endl;
        }
        
        magnetic_field_scan(T_target, Jxx, Jyy, Jzz, 0.0, 0.0, 1, num_steps, h_start, h_end, field_dir, dir_name, Jxz);
    } else {
        if (rank == 0) {
            std::cout << "Initializing molecular dynamic calculation with parameters: Jxx: " << Jxx << " Jyy: " << Jyy << " Jzz: " << Jzz << " Jxz: " << Jxz << " H: " << h << " field direction : " << dir_string << " at Temperature: " << T_target << " saving to: " << dir_name << endl;
        }
        MD_pyrochlore(T_target, num_trials, Jxx, Jyy, Jzz, 0.0, 0.0, 1, h, field_dir, dir_name, Jxz);
    }
    
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    
    return 0;
}