#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "spin_solver_runners.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0]
             << " config.param spins_1.txt [spins_2.txt ...]\n";
        return 1;
    }

    const string config_file = argv[1];
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
    lattice.set_time_dependent_spin_phonon(td_sp_params);

    Eigen::Vector3d B;
    B << config.field_strength * config.field_direction[0],
         config.field_strength * config.field_direction[1],
         config.field_strength * config.field_direction[2];
    lattice.set_field(B);

    const Eigen::Vector3d M1(M_PI, M_PI / std::sqrt(3.0), 0.0);
    const Eigen::Vector3d M2(0.0, 2.0 * M_PI / std::sqrt(3.0), 0.0);
    const Eigen::Vector3d M3(-M_PI, M_PI / std::sqrt(3.0), 0.0);

    cout << "spin_file,S_M1,S_M2,S_M3,m_3Q,m_zigzag,m_min_over_max,dominant_domain\n";
    cout << scientific << setprecision(12);

    for (int argi = 2; argi < argc; ++argi) {
        const string spin_file = argv[argi];
        lattice.load_spin_config(spin_file);

        const double s1 = lattice.structure_factor(M1);
        const double s2 = lattice.structure_factor(M2);
        const double s3 = lattice.structure_factor(M3);
        const double m_3Q = (s1 + s2 + s3) / 3.0;
        const double m_zigzag = std::max({s1, s2, s3});
        const double m_min = std::min({s1, s2, s3});
        const double min_over_max = (m_zigzag > 0.0) ? (m_min / m_zigzag) : 0.0;

        string dominant_domain = "M1";
        if (s2 >= s1 && s2 >= s3) dominant_domain = "M2";
        if (s3 >= s1 && s3 >= s2) dominant_domain = "M3";

        cout << spin_file << ","
             << s1 << ","
             << s2 << ","
             << s3 << ","
             << m_3Q << ","
             << m_zigzag << ","
             << min_over_max << ","
             << dominant_domain << "\n";
    }

    return 0;
}