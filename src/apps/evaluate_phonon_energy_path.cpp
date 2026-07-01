#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "spin_solver_runners.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cctype>

using namespace std;

struct Row {
    string spin_file;
    double qx;
    double qy;
    double time;
    double s;
};

static string trim_copy(string value) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " config.param path_table.csv output.csv\n";
        return 1;
    }

    const string config_file = argv[1];
    const string table_file = argv[2];
    const string output_file = argv[3];

    SpinConfig config = SpinConfig::from_file(config_file);
    UnitCell uc = build_phonon_honeycomb(config);
    PhononLattice lattice(uc, config.lattice_size[0], config.lattice_size[1],
                          config.lattice_size[2], config.spin_length);

    SpinPhononCouplingParams sp_params;
    PhononParams ph_params;
    DriveParams dr_params;
    TimeDependentSpinPhononParams td_sp_params;
    build_phonon_params(config, sp_params, ph_params, dr_params, td_sp_params);
    lattice.set_parameters(sp_params, ph_params, dr_params);

    Eigen::Vector3d B;
    B << config.field_strength * config.field_direction[0],
         config.field_strength * config.field_direction[1],
         config.field_strength * config.field_direction[2];
    lattice.set_field(B);

    ifstream in(table_file);
    if (!in) {
        throw runtime_error("Cannot open input table: " + table_file);
    }
    ofstream out(output_file);
    if (!out) {
        throw runtime_error("Cannot open output file: " + output_file);
    }
    out << "time,s,Qx,Qy,spin_energy_per_site,spin_phonon_energy_per_site,total_spin_sector_energy_per_site\n";
    out << scientific << setprecision(12);

    string line;
    getline(in, line);  // header
    while (getline(in, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string time_s, s_s, qx_s, qy_s, spin_file;
        getline(ss, time_s, ',');
        getline(ss, s_s, ',');
        getline(ss, qx_s, ',');
        getline(ss, qy_s, ',');
        getline(ss, spin_file, ',');
        spin_file = trim_copy(spin_file);

        const double time = stod(time_s);
        const double s = stod(s_s);
        const double qx = stod(qx_s);
        const double qy = stod(qy_s);

        lattice.load_spin_config(spin_file);
        lattice.phonons.Q_x_E1 = qx;
        lattice.phonons.Q_y_E1 = qy;
        lattice.phonons.V_x_E1 = 0.0;
        lattice.phonons.V_y_E1 = 0.0;

        const double n = static_cast<double>(lattice.lattice_size);
        const double spin_e = lattice.spin_energy() / n;
        const double sp_e = lattice.spin_phonon_energy() / n;
        out << time << "," << s << "," << qx << "," << qy << ","
            << spin_e << "," << sp_e << "," << spin_e + sp_e << "\n";
    }

    return 0;
}
