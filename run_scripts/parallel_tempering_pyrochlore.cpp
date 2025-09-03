#include "experiments.h"

using namespace std;

struct Simulation_Param{
    double T_start;
    double T_end;
    double Jxx;
    double Jyy;
    double Jzz;
    double gxx;
    double gyy;
    double gzz;
    double h;
    array<double, 3> field_dir;
    string output_dir;
    vector<int> rank_to_write;
    double theta;
    int num_trials;
    void parse_param_file(string file){
        ifstream infile(file);
        string line;
        while (getline(infile, line)) {
            istringstream iss(line);
            string key;
            if (!(iss >> key)) { continue; } // skip empty lines
            if (key[0] == '#') { continue; } // skip comments
            if (key == "T_start:") { iss >> T_start; }
            else if (key == "T_end:") { iss >> T_end; }
            else if (key == "Jxx:") { iss >> Jxx; }
            else if (key == "Jyy:") { iss >> Jyy; }
            else if (key == "Jzz:") { iss >> Jzz; }
            else if (key == "gxx:") { iss >> gxx; }
            else if (key == "gyy:") { iss >> gyy; }
            else if (key == "gzz:") { iss >> gzz; }
            else if (key == "h:") { iss >> h; }
            else if (key == "field_dir:") { 
                double x, y, z;
                iss >> x >> y >> z;
                field_dir = {x, y, z};
            }
            else if (key == "output_dir:") { iss >> output_dir; }
            else if (key == "rank_to_write:") { 
                rank_to_write.clear();
                int rank;
                while (iss >> rank) {
                    rank_to_write.push_back(rank);
                }
            }
            else if (key == "theta:") { iss >> theta; }
            else if (key == "num_trials:") { iss >> num_trials; }
            else {
                cerr << "Unknown parameter: " << key << endl;
            }
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <parameter_file>" << endl;
        return 1;
    }

    Simulation_Param params;
    params.parse_param_file(argv[1]);
    for (int i = 0; i < params.num_trials; i++) {
        string output_dir = params.output_dir + "/trial_" + std::to_string(i);
        parallel_tempering_pyrochlore(params.T_start, params.T_end, params.Jxx, params.Jyy, params.Jzz, params.gxx, params.gyy, params.gzz, params.h, params.field_dir, params.output_dir, params.rank_to_write, params.theta);
    }
    return 0;
}