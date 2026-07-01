// fix_tmfeo3_su3_seed.cpp
// -----------------------------------------------------------------------------
// Phase 2 utility for the SU(3) coherent-state refactor.
//
// Loads an existing TmFeO3 SU(3) seed file (raw Gell-Mann <lambda^a>
// expectation values, 8 numbers per line, one line per site), audits its
// physicality as a qutrit density matrix, projects every site to the
// closest physical pure-state representation via
//   MixedLattice::physicalize_SU3_state ...
// equivalent for a flat file (we do not need to build the full lattice
// just to fix the seed), and writes a new file alongside it.
//
// Sign / normalisation conventions match
//   include/classical_spin/core/su3_coherent_state.h
// and the discussion in
//   docs/tmfeo3_notes.tex (positive-cone audit, ~ line 1686).
//
// References:
//   Zhang & Batista,  PRB 104, 104409 (2021)
//   Dahlbom et al.,   PRB 106, 054423 (2022); PRB 106, 235154 (2022)
//
// Usage:
//   fix_tmfeo3_su3_seed <input_path>_SU3.txt [<output_path>_SU3.txt]
// If the output is omitted, writes
//   <input>_SU3.physical.txt
// next to the input, leaving the original untouched.
// -----------------------------------------------------------------------------
#include "classical_spin/core/su3_coherent_state.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cs = classical_spin::su3;

namespace {

bool read_seed(const std::string& path, std::vector<cs::Vector8r>& out) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        return false;
    }
    std::string line;
    out.clear();
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        cs::Vector8r n;
        for (int a = 0; a < 8; ++a) {
            if (!(iss >> n(a))) {
                std::cerr << "ERROR: malformed line in " << path << " (need 8 values, got "
                          << a << "): " << line << "\n";
                return false;
            }
        }
        out.push_back(n);
    }
    return !out.empty();
}

bool write_seed(const std::string& path, const std::vector<cs::Vector8r>& v) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "ERROR: cannot write " << path << "\n";
        return false;
    }
    out << std::setprecision(17);
    for (const auto& n : v) {
        for (int a = 0; a < 8; ++a) {
            if (a > 0) out << " ";
            out << n(a);
        }
        out << "\n";
    }
    return true;
}

// Worst (minimum) eigenvalue of rho = (1/3)I + (1/2) sum n^a lambda^a across sites.
double worst_eigenvalue(const std::vector<cs::Vector8r>& v) {
    double w = 1.0;
    for (const auto& n : v) {
        const Eigen::Vector3d ev = cs::density_eigenvalues(n);
        if (ev(0) < w) w = ev(0);
    }
    return w;
}

double worst_purity(const std::vector<cs::Vector8r>& v) {
    double worst = 1.0;
    for (const auto& n : v) {
        const Eigen::Vector3d ev = cs::density_eigenvalues(n);
        if (ev(2) < worst) worst = ev(2);
    }
    return worst;
}

double max_purity_change(const std::vector<cs::Vector8r>& a,
                         const std::vector<cs::Vector8r>& b) {
    double mx = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const double d = (a[i] - b[i]).norm();
        if (d > mx) mx = d;
    }
    return mx;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: " << argv[0]
                  << " <input>_SU3.txt [<output>_SU3.txt]\n";
        return 2;
    }
    const std::string in_path  = argv[1];
    std::string       out_path;
    if (argc == 3) {
        out_path = argv[2];
    } else {
        const std::string suffix = "_SU3.txt";
        if (in_path.size() > suffix.size() &&
            in_path.compare(in_path.size() - suffix.size(), suffix.size(),
                            suffix) == 0) {
            out_path = in_path.substr(0, in_path.size() - suffix.size())
                       + "_SU3.physical.txt";
        } else {
            out_path = in_path + ".physical";
        }
    }

    std::vector<cs::Vector8r> seed;
    if (!read_seed(in_path, seed)) return 1;

    const double w_before = worst_eigenvalue(seed);
    const double p_before = worst_purity(seed);

    std::cout << "loaded " << seed.size() << " sites from " << in_path << "\n";
    std::cout << "  worst rho eigenvalue (before): " << std::scientific
              << std::setprecision(6) << w_before << "\n";
    std::cout << "  worst purity (top eigval, before): "
              << p_before << "\n";

    if (w_before < -1e-10) {
        std::cout << "  --> input is unphysical (negative eigenvalue present)\n";
    } else {
        std::cout << "  --> input is a valid density matrix at all sites\n";
    }

    // Project to closest pure state at every site.
    std::vector<cs::Vector8r> proj = seed;
    for (auto& n : proj) {
        double purity = 0.0;
        auto psi = cs::psi_from_expectations(n, &purity);
        n = cs::expectations_from_psi(psi);
    }

    const double w_after  = worst_eigenvalue(proj);
    const double p_after  = worst_purity(proj);
    const double max_diff = max_purity_change(seed, proj);

    std::cout << "after pure-state projection:\n";
    std::cout << "  worst rho eigenvalue : " << w_after  << "\n";
    std::cout << "  worst purity         : " << p_after  << "\n";
    std::cout << "  max |Delta n| (per site): " << max_diff << "\n";

    if (!write_seed(out_path, proj)) return 1;
    std::cout << "wrote " << proj.size() << " sites to " << out_path << "\n";

    // Sanity round-trip: read back the file we just wrote and re-audit.
    std::vector<cs::Vector8r> reread;
    if (!read_seed(out_path, reread)) return 1;
    const double w_reread = worst_eigenvalue(reread);
    const double p_reread = worst_purity(reread);
    std::cout << "round-trip:\n";
    std::cout << "  worst rho eigenvalue : " << w_reread << "\n";
    std::cout << "  worst purity         : " << p_reread << "\n";

    const bool ok = (w_after  >= -1e-12) &&
                    (1.0 - p_after  < 1e-10) &&
                    (w_reread >= -1e-12) &&
                    (1.0 - p_reread < 1e-10);
    std::cout << (ok ? "PASS" : "FAIL") << ": physical pure-state seed\n";
    return ok ? 0 : 1;
}
