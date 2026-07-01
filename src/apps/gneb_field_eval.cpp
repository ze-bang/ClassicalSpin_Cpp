// GNEB force/energy evaluator for the PhononLattice (NCTO).
//
// Given a manifest of spin configurations (one per band image) with optional
// prescribed E1 strain, this evaluates, for each image in a single process:
//   * the total energy per site (spin + phonon + spin-phonon),
//   * the per-site effective field  H_eff,i = -dH/dS_i  (the energy gradient
//     on the spin manifold, used to build nudged GNEB forces in Python),
//   * the generalized strain forces  -dH/deps_x, -dH/deps_y,
// optionally relaxing the E1 phonon adiabatically (Born-Oppenheimer) first.
//
// Usage:
//   gneb_field_eval config.param manifest.txt energy_out.csv
//
// manifest.txt lines (whitespace separated):
//   <spin_file> <eps_x> <eps_y> <relax_flag 0|1> <field_out_file>
//
// energy_out.csv columns:
//   idx,energy_per_site,eps_x,eps_y,Fqx,Fqy
//
#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "spin_solver_runners.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0]
             << " config.param manifest.txt energy_out.csv\n";
        return 1;
    }
    const string config_file = argv[1];
    const string manifest_file = argv[2];
    const string energy_out = argv[3];

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

    // Apply the same quenched-disorder channels the SA/MD runners use, so the
    // evaluated energy/gradient match the in-model Hamiltonian (e.g. a line
    // defect supplied via nn_exchange_channel_disorder_config).
    if (!config.nn_exchange_disorder_config.empty()) {
        lattice.apply_nn_exchange_disorder_from_file(
            config.nn_exchange_disorder_config);
    }
    if (!config.nn_exchange_channel_disorder_config.empty()) {
        lattice.apply_nn_exchange_channel_disorder_from_file(
            config.nn_exchange_channel_disorder_config);
    }
    if (!config.plaquette_j7_disorder_config.empty()) {
        lattice.apply_plaquette_j7_disorder_from_file(
            config.plaquette_j7_disorder_config);
    }

    Eigen::Vector3d B;
    B << config.field_strength * config.field_direction[0],
         config.field_strength * config.field_direction[1],
         config.field_strength * config.field_direction[2];
    lattice.set_field(B);

    ifstream in(manifest_file);
    if (!in) throw runtime_error("Cannot open manifest: " + manifest_file);
    ofstream out(energy_out);
    if (!out) throw runtime_error("Cannot open output: " + energy_out);
    out << "idx,energy_per_site,eps_x,eps_y,Fqx,Fqy\n";
    out << scientific << setprecision(12);

    const size_t N = lattice.lattice_size;
    string line;
    int idx = 0;
    while (getline(in, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string spin_file, field_out;
        double eps_x, eps_y;
        int relax_flag;
        ss >> spin_file >> eps_x >> eps_y >> relax_flag >> field_out;

        lattice.load_spin_config(spin_file);
        lattice.phonons.Q_x_E1 = eps_x;
        lattice.phonons.Q_y_E1 = eps_y;
        lattice.phonons.V_x_E1 = 0.0;
        lattice.phonons.V_y_E1 = 0.0;

        if (relax_flag) {
            lattice.relax_phonons(1e-11, 5000, 1.0);
        }

        const double E = lattice.total_energy() / static_cast<double>(N);
        const double Fqx = -lattice.dH_dQx_E1();
        const double Fqy = -lattice.dH_dQy_E1();

        // Write per-site effective fields.
        ofstream ff(field_out);
        if (!ff) throw runtime_error("Cannot open field out: " + field_out);
        ff << scientific << setprecision(12);
        for (size_t i = 0; i < N; ++i) {
            const Eigen::Vector3d H = lattice.get_local_field(i);
            ff << H[0] << " " << H[1] << " " << H[2] << "\n";
        }
        ff.close();

        out << idx << "," << E << ","
            << lattice.phonons.Q_x_E1 << "," << lattice.phonons.Q_y_E1 << ","
            << Fqx << "," << Fqy << "\n";
        ++idx;
    }
    return 0;
}
