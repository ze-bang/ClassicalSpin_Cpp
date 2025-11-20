#include "experiments.h"
#include <unordered_map>
#include <sstream>
#include <cctype>
#include <iomanip>

static inline std::string trim(const std::string &s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static std::unordered_map<std::string, std::string> read_params_from_file(const std::string &path) {
    std::unordered_map<std::string, std::string> kv;
    std::ifstream in(path);
    if (!in) {
        std::cout << "Warning: could not open param file: " << path << std::endl;
        return kv;
    }
    std::string line;
    while (std::getline(in, line)) {
        // strip comments (# or //)
        size_t pos_hash = line.find('#');
        if (pos_hash != std::string::npos) line = line.substr(0, pos_hash);
        size_t pos_slashes = line.find("//");
        if (pos_slashes != std::string::npos) line = line.substr(0, pos_slashes);
        line = trim(line);
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (!key.empty()) kv[key] = val;
    }
    return kv;
}

static bool getBool(const std::unordered_map<std::string, std::string> &m, const std::string &key, bool defVal) {
    auto it = m.find(key);
    if (it == m.end()) return defVal;
    std::string v = it->second;
    for (auto &c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    return defVal;
}

static int getInt(const std::unordered_map<std::string, std::string> &m, const std::string &key, int defVal) {
    auto it = m.find(key);
    if (it == m.end()) return defVal;
    try { return std::stoi(it->second); } catch (...) { return defVal; }
}

static double getDouble(const std::unordered_map<std::string, std::string> &m, const std::string &key, double defVal) {
    auto it = m.find(key);
    if (it == m.end()) return defVal;
    try { return std::stod(it->second); } catch (...) { return defVal; }
}

static std::string getString(const std::unordered_map<std::string, std::string> &m, const std::string &key, const std::string &defVal) {
    auto it = m.find(key);
    if (it == m.end()) return defVal;
    return it->second;
}


mixed_UnitCell<3, 4, 8, 4> setup_lattice(double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double e1, double e2, double chii, double xii, double h, const array<double,3> &fielddir, string dir){
    filesystem::create_directories(dir);
    TmFeO3_Fe<3> Fe_atoms;
    TmFeO3_Tm<8> Tm_atoms;

    // ========================================================================
    // GLOBAL TO LOCAL FRAME TRANSFORMATION
    // ========================================================================
    // Global Hamiltonian: H = -J1 Σ_<ij> S_i·S_j - J2 Σ_<ij>' S_i·S_j 
    //                        - D1 Σ ŷ·(S_i × S_j) - D2 Σ ẑ·(S_i × S_j)
    //                        - Ka Σ(S_i^x)² - Kc Σ(S_i^z)²
    //
    // Local sublattice frames (sign patterns applied to {x,y,z}):
    //   Site 0: { x,  y,  z} → η₀ = { 1,  1,  1}
    //   Site 1: { x, -y, -z} → η₁ = { 1, -1, -1}
    //   Site 2: {-x,  y, -z} → η₂ = {-1,  1, -1}
    //   Site 3: {-x, -y,  z} → η₃ = {-1, -1,  1}
    //
    // Transformation: J_ij^(ab,local) = J_orig^(ab) × η_i^a × η_j^b
    // ========================================================================

    array<array<double, 3>, 4> eta = {{{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}}};

    // Original exchange matrices in global frame
    // Antisymmetric parts encode DM interactions:
    //   (J_xy - J_yx)/2 = D2  →  ẑ·(S_i × S_j)
    //   (J_zx - J_xz)/2 = -D1 →  ŷ·(S_i × S_j)
    array<array<double, 3>, 3> Ja_orig = {{{Jai, D2, -D1}, {-D2, Jai, 0}, {D1, 0, Jai}}};
    array<array<double, 3>, 3> Jb_orig = {{{Jbi, D2, -D1}, {-D2, Jbi, 0}, {D1, 0, Jbi}}};
    array<array<double, 3>, 3> Jc_orig = {{{Jci, 0, 0}, {0, Jci, 0}, {0, 0, Jci}}};

    array<array<double, 3>, 3> J2a_orig = {{{J2ai, 0, 0}, {0, J2ai, 0}, {0, 0, J2ai}}};
    array<array<double, 3>, 3> J2b_orig = {{{J2bi, 0, 0}, {0, J2bi, 0}, {0, 0, J2bi}}};
    array<array<double, 3>, 3> J2c_orig = {{{J2ci, 0, 0}, {0, J2ci, 0}, {0, 0, J2ci}}};

    // Lambda to perform matrix transformation: J_local = diag(η_i) × J_global × diag(η_j)
    auto transform_exchange = [&eta](const array<array<double, 3>, 3>& J_orig, int i, int j) {
        array<array<double, 3>, 3> J_local = {{{0}}};
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                J_local[a][b] = J_orig[a][b] * eta[i][a] * eta[j][b];
            }
        }
        return J_local;
    };

    // Verification lambda: check that trace is preserved (important invariant)
    auto verify_trace = [](const array<array<double, 3>, 3>& J_orig, 
                           const array<array<double, 3>, 3>& J_local,
                           const string& name) {
        double trace_orig = J_orig[0][0] + J_orig[1][1] + J_orig[2][2];
        double trace_local = J_local[0][0] + J_local[1][1] + J_local[2][2];
        if (abs(trace_orig - trace_local) > 1e-10) {
            cout << "WARNING: Trace not preserved for " << name << "!" << endl;
            cout << "  Original trace: " << trace_orig << ", Local trace: " << trace_local << endl;
        }
    };

    // Create 4x4 sublattice versions for each exchange matrix
    // Indices: [site_i][site_j][component_a][component_b]
    array<array<array<array<double, 3>, 3>, 4>, 4> Ja;
    array<array<array<array<double, 3>, 3>, 4>, 4> Jb;
    array<array<array<array<double, 3>, 3>, 4>, 4> Jc;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2a;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2b;
    array<array<array<array<double, 3>, 3>, 4>, 4> J2c;
    
    cout << "Transforming exchange matrices to local sublattice frames..." << endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            // Transform using explicit formula
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    Ja[i][j][a][b] = Ja_orig[a][b] * eta[i][a] * eta[j][b];
                    Jb[i][j][a][b] = Jb_orig[a][b] * eta[i][a] * eta[j][b];
                    Jc[i][j][a][b] = Jc_orig[a][b] * eta[i][a] * eta[j][b];
                    J2a[i][j][a][b] = J2a_orig[a][b] * eta[i][a] * eta[j][b];
                    J2b[i][j][a][b] = J2b_orig[a][b] * eta[i][a] * eta[j][b];
                    J2c[i][j][a][b] = J2c_orig[a][b] * eta[i][a] * eta[j][b];
                }
            }
            
            // Verify key transformations
            if (i == 0 && j == 0) {
                verify_trace(Ja_orig, Ja[i][j], "Ja[0][0]");
                cout << "  Sample verification: Ja[0][0] trace preserved ✓" << endl;
            }
        }
    }
    
    // Additional verification: Check antisymmetric part is transformed correctly
    auto check_dm_term = [](const array<array<double, 3>, 3>& J, double D_expected, int comp, const string& msg) {
        const int pairs[3][2] = {{1, 2}, {2, 0}, {0, 1}}; // (y,z), (z,x), (x,y)
        int a = pairs[comp][0], b = pairs[comp][1];
        double DM = (J[a][b] - J[b][a]) / 2.0;
        if (abs(abs(DM) - abs(D_expected)) > 1e-10) {
            cout << "WARNING: " << msg << " DM term mismatch!" << endl;
            cout << "  Expected: " << D_expected << ", Got: " << DM << endl;
        }
    };
    
    // Verify DM interactions are encoded correctly in Ja[0][0] (should preserve original)
    check_dm_term(Ja[0][0], D2, 2, "Ja[0][0] D2(z)");
    check_dm_term(Ja[0][0], -D1, 1, "Ja[0][0] D1(y)");
    cout << "  DM interaction verification passed ✓" << endl;

    // Optional: Print sample transformed matrices for verification
    if (false) { // Set to true to enable detailed output
        cout << "\n=== Sample Transformed Matrices ===" << endl;
        cout << "Ja[1][0] (site 1 → site 0):" << endl;
        for (int a = 0; a < 3; a++) {
            cout << "  [";
            for (int b = 0; b < 3; b++) {
                printf("%8.4f", Ja[1][0][a][b]);
            }
            cout << " ]" << endl;
        }
        cout << "\nJa[2][3] (site 2 → site 3):" << endl;
        for (int a = 0; a < 3; a++) {
            cout << "  [";
            for (int b = 0; b < 3; b++) {
                printf("%8.4f", Ja[2][3][a][b]);
            }
            cout << " ]" << endl;
        }
    }

    // ========================================================================
    // SETTING UP FE-FE INTERACTIONS
    // ========================================================================
    // Now apply the transformed matrices to the actual lattice bonds
    
    array<double, 9> K = {{Ka, 0, 0, 0, 0, 0, 0, 0, Kc}};
    
    cout << "\nSetting up Fe-Fe interactions..." << endl;
    //In plane interactions (J1 type, nearest neighbor within ab-plane)
    // J1 bonds: Along a±b directions (diagonal in ab-plane)
    // Bond type 'a': R_j = R_i + a(x̂ + ŷ)  and  R_j = R_i + a(x̂ - ŷ)
    Fe_atoms.set_bilinear_interaction(Ja[1][0], 1, 0, {0,0,0});    // site 1 → site 0
    Fe_atoms.set_bilinear_interaction(Ja[1][0], 1, 0, {1,-1,0});   // site 1 → site 0 (translated)
    // Bond type 'b': Along perpendicular diagonal
    Fe_atoms.set_bilinear_interaction(Jb[1][0], 1, 0, {0,-1,0});   // site 1 → site 0
    Fe_atoms.set_bilinear_interaction(Jb[1][0], 1, 0, {1,0,0});    // site 1 → site 0 (translated)

    Fe_atoms.set_bilinear_interaction(Ja[2][3], 2, 3, {0,0,0});    // site 2 → site 3
    Fe_atoms.set_bilinear_interaction(Ja[2][3], 2, 3, {1,-1,0});   // site 2 → site 3 (translated)
    Fe_atoms.set_bilinear_interaction(Jb[2][3], 2, 3, {0,-1,0});   // site 2 → site 3
    Fe_atoms.set_bilinear_interaction(Jb[2][3], 2, 3, {1,0,0});    // site 2 → site 3 (translated)

    //Next Nearest Neighbour (J2 type, along a and b axes)
    Fe_atoms.set_bilinear_interaction(J2a[0][0], 0, 0, {1,0,0});   // site 0 → site 0 (along x)
    Fe_atoms.set_bilinear_interaction(J2b[0][0], 0, 0, {0,1,0});   // site 0 → site 0 (along y)
    Fe_atoms.set_bilinear_interaction(J2a[1][1], 1, 1, {1,0,0});   // site 1 → site 1 (along x)
    Fe_atoms.set_bilinear_interaction(J2b[1][1], 1, 1, {0,1,0});   // site 1 → site 1 (along y)
    Fe_atoms.set_bilinear_interaction(J2a[2][2], 2, 2, {1,0,0});   // site 2 → site 2 (along x)
    Fe_atoms.set_bilinear_interaction(J2b[2][2], 2, 2, {0,1,0});   // site 2 → site 2 (along y)
    Fe_atoms.set_bilinear_interaction(J2a[3][3], 3, 3, {1,0,0});   // site 3 → site 3 (along x)
    Fe_atoms.set_bilinear_interaction(J2b[3][3], 3, 3, {0,1,0});   // site 3 → site 3 (along y)

    //Out of plane interaction (J1 type along c-axis)
    Fe_atoms.set_bilinear_interaction(Jc[0][3], 0, 3, {0,0,0});    // site 0 → site 3 (same z)
    Fe_atoms.set_bilinear_interaction(Jc[0][3], 0, 3, {0,0,1});    // site 0 → site 3 (z+1)
    Fe_atoms.set_bilinear_interaction(Jc[1][2], 1, 2, {0,0,0});    // site 1 → site 2 (same z)
    Fe_atoms.set_bilinear_interaction(Jc[1][2], 1, 2, {0,0,1});    // site 1 → site 2 (z+1)

    // J2 out-of-plane interactions (4 bonds per pair, forming square around z-axis)
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,0,0});   // z=0 layer
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,1,0});   
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,0,0});  
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,1,0});  

    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,0,1});   // z=1 layer
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {0,1,1});   
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,0,1});  
    Fe_atoms.set_bilinear_interaction(J2c[0][2], 0, 2, {-1,1,1});  

    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,0,0});   // z=0 layer
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,-1,0});  
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,0,0});   
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,-1,0});  

    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,0,1});   // z=1 layer
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {0,-1,1});  
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,0,1});   
    Fe_atoms.set_bilinear_interaction(J2c[1][3], 1, 3, {1,-1,1});  

    //Single ion anisotropy (same in all local frames due to quadratic form)
    Fe_atoms.set_onsite_interaction(K, 0);
    Fe_atoms.set_onsite_interaction(K, 1);
    Fe_atoms.set_onsite_interaction(K, 2);
    Fe_atoms.set_onsite_interaction(K, 3);

    //External magnetic field (same in all frames - global field direction)
    Fe_atoms.set_field(fielddir*h, 0);
    Fe_atoms.set_field(fielddir*h, 1);
    Fe_atoms.set_field(fielddir*h, 2);
    Fe_atoms.set_field(fielddir*h, 3);

    //Tm atoms
    //Set energy splitting for Tm atoms
    //\alpha\lambda3 + \beta\lambda8 + \gamma\identity
    double alpha = e1/2;
    double beta = sqrt(3)/6*(2*e2-e1);
    double gamma = -(e1+e2)/3;

    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 0);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 1);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 2);
    Tm_atoms.set_field({0,0,alpha,0,0,0,0,beta}, 3);



    TmFeO3<3, 8> TFO(&Fe_atoms, &Tm_atoms);
    // I have finally cracked the correct model...
    // Here we go!

    // Bilinear coupling (chii parameter)
    if (chii != 0.0){
        array<array<double,3>,8> chi = {{{0}}};
        chi[1] = {{chii, chii, chii}};
        chi[4] = {{chii, chii, chii}};
        chi[6] = {{chii, chii, chii}};

        array<array<double,3>,8> chi_inv = {{{0}}};
        chi_inv[1] = {{chii, chii, chii}};
        chi_inv[4] = {{-chii, -chii, -chii}};
        chi_inv[6] = {{-chii, -chii, -chii}};

        // Structure is SU(3) sites then SU(2) sites then unitcell offset        
        // Fe site 0 - 8 nearest Tm neighbors
        // Fe position: (0.00000, 0.50000, 0.50000)
        // Inversion pair 1 (distance: 0.4965):
        TFO.set_mix_bilinear_interaction(chi, 3, 0, {-1, 0, 0});
        TFO.set_mix_bilinear_interaction(chi_inv, 0, 0, {0, 0, 0});
        // Inversion pair 2 (distance: 0.5449):
        TFO.set_mix_bilinear_interaction(chi, 2, 0, {0, 0, 0});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 0, {-1, 0, 0});
        // Inversion pair 3 (distance: 0.5824):
        TFO.set_mix_bilinear_interaction(chi, 1, 0, {0, 0, 0});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 0, {-1, 0, 0});
        // Inversion pair 4 (distance: 0.6242):
        TFO.set_mix_bilinear_interaction(chi, 0, 0, {0, -1, 0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 0, {-1, 1, 0});

        // Fe site 1 - 8 nearest Tm neighbors
        // Fe position: (0.50000, 0.00000, 0.50000)
        // Inversion pair 1 (distance: 0.4965):
        TFO.set_mix_bilinear_interaction(chi, 2, 1, {0, 0, 0});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 1, {0, -1, 0});
        // Inversion pair 2 (distance: 0.5449):
        TFO.set_mix_bilinear_interaction(chi, 0, 1, {0, -1, 0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 1, {0, 0, 0});
        // Inversion pair 3 (distance: 0.5824):
        TFO.set_mix_bilinear_interaction(chi, 0, 1, {1, -1, 0});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 1, {-1, 0, 0});
        // Inversion pair 4 (distance: 0.6242):
        TFO.set_mix_bilinear_interaction(chi, 1, 1, {0, 0, 0});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 1, {0, -1, 0});

        // Fe site 2 - 8 nearest Tm neighbors
        // Fe position: (0.50000, 0.00000, 0.00000)
        // Inversion pair 1 (distance: 0.4965):
        TFO.set_mix_bilinear_interaction(chi, 2, 2, {0, 0, -1});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 2, {0, -1, 0});
        // Inversion pair 2 (distance: 0.5449):
        TFO.set_mix_bilinear_interaction(chi, 0, 2, {0, -1, -1});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 2, {0, 0, 0});
        // Inversion pair 3 (distance: 0.5824):
        TFO.set_mix_bilinear_interaction(chi, 0, 2, {1, -1, -1});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 2, {-1, 0, 0});
        // Inversion pair 4 (distance: 0.6242):
        TFO.set_mix_bilinear_interaction(chi, 1, 2, {0, 0, 0});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 2, {0, -1, -1});

        // Fe site 3 - 8 nearest Tm neighbors
        // Fe position: (0.00000, 0.50000, 0.00000)
        // Inversion pair 1 (distance: 0.4965):
        TFO.set_mix_bilinear_interaction(chi, 3, 3, {-1, 0, 0});
        TFO.set_mix_bilinear_interaction(chi_inv, 0, 3, {0, 0, -1});
        // Inversion pair 2 (distance: 0.5449):
        TFO.set_mix_bilinear_interaction(chi, 2, 3, {0, 0, -1});
        TFO.set_mix_bilinear_interaction(chi_inv, 1, 3, {-1, 0, 0});
        // Inversion pair 3 (distance: 0.5824):
        TFO.set_mix_bilinear_interaction(chi, 1, 3, {0, 0, 0});
        TFO.set_mix_bilinear_interaction(chi_inv, 2, 3, {-1, 0, -1});
        // Inversion pair 4 (distance: 0.6242):
        TFO.set_mix_bilinear_interaction(chi, 0, 3, {0, -1, -1});
        TFO.set_mix_bilinear_interaction(chi_inv, 3, 3, {-1, 1, 0});

    }

    cout << "\n========================================" << endl;
    cout << "Finished setting up TmFeO3 model." << endl;
    cout << "========================================" << endl;
    cout << "Summary of interactions:" << endl;
    cout << "  J1 nearest-neighbor: Ja=" << Jai << ", Jb=" << Jbi << ", Jc=" << Jci << endl;
    cout << "  J2 next-nearest:     J2a=" << J2ai << ", J2b=" << J2bi << ", J2c=" << J2ci << endl;
    cout << "  DM interactions:     D1=" << D1 << ", D2=" << D2 << endl;
    cout << "  Single-ion:          Ka=" << Ka << ", Kc=" << Kc << endl;
    cout << "  Fe-Tm coupling:      χ=" << chii << ", ξ=" << xii << endl;
    cout << "  Tm splitting:        e1=" << e1 << ", e2=" << e2 << endl;
    cout << "  External field:      h=" << h << " along (" << fielddir[0] << "," 
         << fielddir[1] << "," << fielddir[2] << ")" << endl;
    cout << "========================================\n" << endl;

    // Save transformation details to file for reference
    ofstream param_log(dir + "/hamiltonian_setup.log");
    param_log << "HAMILTONIAN TRANSFORMATION LOG" << endl;
    param_log << "==============================" << endl;
    param_log << "\nGlobal Frame Hamiltonian:" << endl;
    param_log << "H = -J1·Σ_<ij> S_i·S_j - J2·Σ_<ij>' S_i·S_j" << endl;
    param_log << "    - D1·Σ ŷ·(S_i × S_j) - D2·Σ ẑ·(S_i × S_j)" << endl;
    param_log << "    - Ka·Σ(S_i^x)² - Kc·Σ(S_i^z)²" << endl;
    param_log << "\nLocal Sublattice Frames:" << endl;
    param_log << "  Site 0: { x,  y,  z} → η₀ = { 1,  1,  1}" << endl;
    param_log << "  Site 1: { x, -y, -z} → η₁ = { 1, -1, -1}" << endl;
    param_log << "  Site 2: {-x,  y, -z} → η₂ = {-1,  1, -1}" << endl;
    param_log << "  Site 3: {-x, -y,  z} → η₃ = {-1, -1,  1}" << endl;
    param_log << "\nParameters (in units of J1ab):" << endl;
    param_log << "  J1ab = " << Jai << ", J1c = " << Jci << endl;
    param_log << "  J2ab = " << J2ai << ", J2c = " << J2ci << endl;
    param_log << "  Ka = " << Ka << ", Kc = " << Kc << endl;
    param_log << "  D1 = " << D1 << ", D2 = " << D2 << endl;
    param_log << "  e1 = " << e1 << ", e2 = " << e2 << endl;
    param_log << "  χ = " << chii << ", ξ = " << xii << endl;
    param_log << "  h = " << h << " T" << endl;
    param_log << "\nSample Transformed Matrix [Ja site1→site0]:" << endl;
    for (int a = 0; a < 3; a++) {
        param_log << "  [";
        for (int b = 0; b < 3; b++) {
            param_log << " " << setw(10) << setprecision(6) << Ja[1][0][a][b];
        }
        param_log << " ]" << endl;
    }
    param_log.close();

    // ========================================================================
    // OUTPUT UNIT CELL INFORMATION
    // ========================================================================
    ofstream unitcell_info(dir + "/unitcell_info.txt");
    unitcell_info << "========================================" << endl;
    unitcell_info << "UNIT CELL INFORMATION - TmFeO3" << endl;
    unitcell_info << "========================================\n" << endl;
    
    // Lattice vectors (same for both Fe and Tm sublattices)
    unitcell_info << "Lattice Vectors (in units of lattice constant):" << endl;
    unitcell_info << "  a1 = (" << TFO.SU2.lattice_vectors[0][0] << ", " 
                  << TFO.SU2.lattice_vectors[0][1] << ", " 
                  << TFO.SU2.lattice_vectors[0][2] << ")" << endl;
    unitcell_info << "  a2 = (" << TFO.SU2.lattice_vectors[1][0] << ", " 
                  << TFO.SU2.lattice_vectors[1][1] << ", " 
                  << TFO.SU2.lattice_vectors[1][2] << ")" << endl;
    unitcell_info << "  a3 = (" << TFO.SU2.lattice_vectors[2][0] << ", " 
                  << TFO.SU2.lattice_vectors[2][1] << ", " 
                  << TFO.SU2.lattice_vectors[2][2] << ")" << endl;
    
    // Fe sublattice information
    unitcell_info << "\n----------------------------------------" << endl;
    unitcell_info << "Fe Sublattice (SU(2) spins, S=5/2)" << endl;
    unitcell_info << "----------------------------------------" << endl;
    unitcell_info << "Number of Fe sites per unit cell: 4" << endl;
    unitcell_info << "Spin dimension: 3" << endl;
    unitcell_info << "\nFe site positions (fractional coordinates):" << endl;
    for (int i = 0; i < 4; i++) {
        unitcell_info << "  Fe[" << i << "] = (" 
                      << fixed << setprecision(5) << TFO.SU2.lattice_pos[i][0] << ", " 
                      << TFO.SU2.lattice_pos[i][1] << ", " 
                      << TFO.SU2.lattice_pos[i][2] << ")" << endl;
    }
    
    unitcell_info << "\nFe local sublattice frames:" << endl;
    unitcell_info << "  (transformation from global to local coordinates)" << endl;
    for (int i = 0; i < 4; i++) {
        unitcell_info << "  Fe[" << i << "] frame:" << endl;
        unitcell_info << "    x_local = (" << TFO.SU2.sublattice_frames[i][0][0] << ", "
                      << TFO.SU2.sublattice_frames[i][0][1] << ", "
                      << TFO.SU2.sublattice_frames[i][0][2] << ")" << endl;
        unitcell_info << "    y_local = (" << TFO.SU2.sublattice_frames[i][1][0] << ", "
                      << TFO.SU2.sublattice_frames[i][1][1] << ", "
                      << TFO.SU2.sublattice_frames[i][1][2] << ")" << endl;
        unitcell_info << "    z_local = (" << TFO.SU2.sublattice_frames[i][2][0] << ", "
                      << TFO.SU2.sublattice_frames[i][2][1] << ", "
                      << TFO.SU2.sublattice_frames[i][2][2] << ")" << endl;
    }
    
    // Tm sublattice information
    unitcell_info << "\n----------------------------------------" << endl;
    unitcell_info << "Tm Sublattice (SU(3) pseudospins, J=1)" << endl;
    unitcell_info << "----------------------------------------" << endl;
    unitcell_info << "Number of Tm sites per unit cell: 4" << endl;
    unitcell_info << "Spin dimension: 8 (Gell-Mann matrices)" << endl;
    unitcell_info << "\nTm site positions (fractional coordinates):" << endl;
    for (int i = 0; i < 4; i++) {
        unitcell_info << "  Tm[" << i << "] = (" 
                      << fixed << setprecision(5) << TFO.SU3.lattice_pos[i][0] << ", " 
                      << TFO.SU3.lattice_pos[i][1] << ", " 
                      << TFO.SU3.lattice_pos[i][2] << ")" << endl;
    }
    
    // Total unit cell summary
    unitcell_info << "\n========================================" << endl;
    unitcell_info << "UNIT CELL SUMMARY" << endl;
    unitcell_info << "========================================" << endl;
    unitcell_info << "Total sites per unit cell: 8 (4 Fe + 4 Tm)" << endl;
    unitcell_info << "Crystal structure: Orthorhombic (Pbnm space group)" << endl;
    unitcell_info << "Lattice type: Simple orthorhombic with basis" << endl;
    unitcell_info << "\nNotes:" << endl;
    unitcell_info << "  - Fe sites form a distorted perovskite B-site sublattice" << endl;
    unitcell_info << "  - Tm sites occupy the perovskite A-site sublattice" << endl;
    unitcell_info << "  - Local frames for Fe sites account for crystallographic" << endl;
    unitcell_info << "    symmetry operations of the Pbnm space group" << endl;
    unitcell_info << "  - All coordinates given in fractional (reduced) units" << endl;
    unitcell_info << "========================================" << endl;
    
    unitcell_info.close();
    
    cout << "Unit cell information written to " << dir << "/unitcell_info.txt" << endl;

    return TFO;
}

void simulated_annealing_TmFeO3(double T_start, double T_end, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double e1, double e2, double chii, double xii, double h, const array<double,3> &fielddir, string dir){
    mixed_UnitCell<3, 4, 8, 4> TFO = setup_lattice(Jai, Jbi, Jci, J2ai, J2bi, J2ci, Ka, Kc, D1, D2, e1, e2, chii, xii, h, fielddir, dir);
    
    // Write all parameters to params.txt
    ofstream params_file(dir + "/params.txt");
    params_file << "# TmFeO3 Simulated Annealing Parameters" << endl;
    params_file << "# All parameters normalized to J1ab = 1" << endl;
    params_file << "# ========================================" << endl;
    params_file << endl;
    params_file << "# Temperature range" << endl;
    params_file << "T_start = " << T_start << endl;
    params_file << "T_end = " << T_end << endl;
    params_file << endl;
    params_file << "# Exchange interactions (J1 nearest neighbor)" << endl;
    params_file << "J1ab = " << Jai << "  # In-plane (a,b directions)" << endl;
    params_file << "J1c = " << Jci << "   # Out-of-plane (c direction)" << endl;
    params_file << endl;
    params_file << "# Next-nearest neighbor exchange (J2)" << endl;
    params_file << "J2ab = " << J2ai << endl;
    params_file << "J2c = " << J2ci << endl;
    params_file << endl;
    params_file << "# Single-ion anisotropy" << endl;
    params_file << "Ka = " << Ka << "  # In-plane anisotropy" << endl;
    params_file << "Kc = " << Kc << "  # Out-of-plane anisotropy" << endl;
    params_file << endl;
    params_file << "# Dzyaloshinskii-Moriya interactions" << endl;
    params_file << "D1 = " << D1 << "  # DM along y-axis" << endl;
    params_file << "D2 = " << D2 << "  # DM along z-axis" << endl;
    params_file << endl;
    params_file << "# Tm crystal field splitting" << endl;
    params_file << "e1 = " << e1 << endl;
    params_file << "e2 = " << e2 << endl;
    params_file << endl;
    params_file << "# Fe-Tm coupling" << endl;
    params_file << "chii = " << chii << "  # Bilinear coupling" << endl;
    params_file << "xii = " << xii << "   # (Additional coupling parameter)" << endl;
    params_file << endl;
    params_file << "# External magnetic field" << endl;
    params_file << "h = " << h << "  # Field magnitude" << endl;
    params_file << "fielddir = (" << fielddir[0] << ", " << fielddir[1] << ", " << fielddir[2] << ")  # Field direction" << endl;
    params_file << endl;
    params_file << "# Output directory" << endl;
    params_file << "dir_name = " << dir << endl;
    params_file.close();
    cout << "Parameters written to " << dir << "/params.txt" << endl;
    
    mixed_lattice<3, 4, 8, 4, 4, 4, 4> MC(&TFO, 2.5, 1.0);
    cout << "Starting simulated annealing from T=" << T_start << " to T=" << T_end << endl;
    MC.simulated_annealing(T_start, T_end, 10000, 0, 10, false);

    MC.write_to_file_spin(dir + "/spin");
    
    // Write atom positions
    cout << "Writing atom positions to " << dir + "/pos_SU2.txt and " << dir + "/pos_SU3.txt" << endl;
    MC.write_to_file_pos(dir + "/pos");
    
    cout << "Running zero temperature relaxation sweeps..." << endl;
    for (size_t i = 0; i < 100000; ++i) {
        MC.deterministic_sweep();
    }

    // Write the zero temperature spin configuration
    cout << "Writing zero temperature spin configuration to " << dir + "/spin_zero" << endl;
    MC.write_to_file_spin(dir + "/spin_zero");

}

int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;

    // Prefer loading parameters from a key=value param file if argv[1] is a valid file path.
    double T_start, T_end;
    double J1ab, J1c, J2ab, J2c, Ka, Kc, D1, D2, e1, e2, chii, xii, h;
    std::string dir_name;

    std::string param_file = (argc > 1 && filesystem::exists(argv[1])) ? argv[1] : "";
    if (!param_file.empty()) {
        std::cout << "Loading parameters from file: " << param_file << std::endl;
        auto p = read_params_from_file(param_file);
        // Defaults (same as TmFeO3_2DCS.cpp)
        T_start = getDouble(p, "T_start", 20.0);
        T_end = getDouble(p, "T_end", 0.01);

        J1ab = getDouble(p, "J1ab", 4.92);
        J1c  = getDouble(p, "J1c", 4.92);
        J2ab = getDouble(p, "J2ab", 0.29);
        J2c  = getDouble(p, "J2c", 0.29);
        Ka   = getDouble(p, "Ka", 0.0);
        Kc   = getDouble(p, "Kc", -0.09);
        D1   = getDouble(p, "D1", 0.0);
        D2   = getDouble(p, "D2", 0.0);
        e1   = getDouble(p, "e1", 4.0);
        e2   = getDouble(p, "e2", 0.0);
        chii = getDouble(p, "chii", 0.0);
        xii  = getDouble(p, "xii", 0.0);
        h    = getDouble(p, "h", 0.0);

        dir_name = getString(p, "dir_name", "TmFeO3_annealing");
    } else {
        // Fallback to original CLI parsing with better defaults
        J1ab = (argc > 1) ? atof(argv[1]) : 4.92;
        J1c = (argc > 2) ? atof(argv[2]) : 4.92;
        J2ab = (argc > 3) ? atof(argv[3]) : 0.29;
        J2c = (argc > 4) ? atof(argv[4]) : 0.29;
        Ka = (argc > 5) ? atof(argv[5]) : 0.0;
        Kc = (argc > 6) ? atof(argv[6]) : -0.09;
        D1 = (argc > 7) ? atof(argv[7]) : 0.0;
        D2 = (argc > 8) ? atof(argv[8]) : 0.0;
        chii = (argc > 9) ? atof(argv[9]) : 0.0;
        xii = (argc > 10) ? atof(argv[10]) : 0.0;
        e1 = (argc > 11) ? atof(argv[11]) : 4.0;
        e2 = (argc > 12) ? atof(argv[12]) : 0.0;
        h = (argc > 13) ? atof(argv[13]) : 0.0;
        dir_name = (argc > 14) ? argv[14] : std::string("TmFeO3_annealing");
        T_start = (argc > 15) ? atof(argv[15]) : 20.0;
        T_end = (argc > 16) ? atof(argv[16]) : 0.01;
    }

    // Normalize to J1ab
    if (J1ab != 0){
        J1c /= J1ab;
        J2ab /= J1ab;
        J2c /= J1ab;
        Ka /= J1ab;
        Kc /= J1ab;
        D1 /= J1ab;
        D2 /= J1ab;
        e1 /= J1ab;
        e2 /= J1ab;
        h /= J1ab;
        J1ab = 1;
    }
    filesystem::create_directories(dir_name);

    cout << "Begin simulated annealing on TmFeO3 with parameters:" << endl;
    cout << "  J1ab=" << J1ab << " J1c=" << J1c << " J2ab=" << J2ab << " J2c=" << J2c << endl;
    cout << "  Ka=" << Ka << " Kc=" << Kc << " D1=" << D1 << " D2=" << D2 << endl;
    cout << "  e1=" << e1 << " e2=" << e2 << " chii=" << chii << " xii=" << xii << endl;
    cout << "  h=" << h << " T_start=" << T_start << " T_end=" << T_end << endl;
    cout << "  Output directory: " << dir_name << endl;

    simulated_annealing_TmFeO3(T_start, T_end, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, e1, e2, chii, xii, h, {0,0,1}, dir_name);
    
    return 0;
}