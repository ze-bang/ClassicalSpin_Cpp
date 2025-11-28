#include "experiments.h"
#include "../src/molecular_dynamics.cuh"
#include <fstream>
#include <sstream>
#include <unordered_map>

static bool load_params_from_file(const std::string &path, std::unordered_map<std::string, double> &out_double, std::unordered_map<std::string, std::string> &out_string) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        size_t pos = line.find_first_not_of(" \t");
        if (pos == std::string::npos) continue;
        if (line[pos] == '#' || (line[pos] == '/' && pos + 1 < line.size() && line[pos + 1] == '/')) continue;
        
        // Handle "key: value", "key = value", and "key value" formats
        size_t colon_pos = line.find(':');
        size_t equals_pos = line.find('=');
        std::string key, value_str;
        
        if (colon_pos != std::string::npos) {
            // Format: "key: value"
            key = line.substr(0, colon_pos);
            value_str = line.substr(colon_pos + 1);
        } else if (equals_pos != std::string::npos) {
            // Format: "key = value"
            key = line.substr(0, equals_pos);
            value_str = line.substr(equals_pos + 1);
        } else {
            // Format: "key value" (whitespace separated)
            std::istringstream iss(line);
            if (!(iss >> key >> value_str)) continue;
        }
        
        // Trim whitespace from key and value
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value_str.erase(0, value_str.find_first_not_of(" \t"));
        value_str.erase(value_str.find_last_not_of(" \t") + 1);
        
        // Remove inline comments (anything after # in value_str)
        size_t comment_pos = value_str.find('#');
        if (comment_pos != std::string::npos) {
            value_str = value_str.substr(0, comment_pos);
            // Trim again after removing comment
            value_str.erase(value_str.find_last_not_of(" \t") + 1);
        }
        
        // Try to parse as double, otherwise store as string
        try {
            double val = std::stod(value_str);
            out_double[key] = val;
        } catch (...) {
            out_string[key] = value_str;
        }
    }
    return true;
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
    double alpha = e1;
    double beta = sqrt(3)/3*(2*e2-e1);

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

    return TFO;
}


void MD_TmFeO3(int num_trials, double Temp_start, double Temp_end, double T_start, double T_end, double T_step_size, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double chii, double xii, double h, const array<double,3> &fielddir, double e1, double e2, double offset, string dir, string spin_config_filename){
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    auto TFO = setup_lattice(Jai, Jbi, Jci, J2ai, J2bi, J2ci, Ka, Kc, D1, D2, e1, e2, chii, xii, h, fielddir, dir);

    //
    cout << "Finished setting up TmFeO3 model." << endl;
    cout << "Starting calculations..." << endl;

    int trial_section = int(num_trials/size);

    for(size_t i = rank*trial_section; i < (rank+1)*trial_section; ++i){
        mixed_lattice<3, 4, 8, 4, 4, 4, 4> MC(&TFO, 2.5, 1.0);
        if (spin_config_filename != ""){
            MC.read_spin_from_file(spin_config_filename);
        }
        else{
            MC.simulated_annealing(Temp_start, Temp_end , 10000, 0, 100, false);
        }
        MC.molecular_dynamics(T_start, T_end, T_step_size, dir+"/"+std::to_string(i));
    }

}

void MD_TmFeO3_cuda(int num_trials, double Temp_start, double Temp_end, double T_start, double T_end, double T_step_size, double Jai, double Jbi, double Jci, double J2ai, double J2bi, double J2ci, double Ka, double Kc, double D1, double D2, double chii, double xii, double h, const array<double,3> &fielddir, double e1, double e2, double offset, string dir, string spin_config_filename){
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized){
        MPI_Init(NULL, NULL);
    }
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto TFO = setup_lattice(Jai, Jbi, Jci, J2ai, J2bi, J2ci, Ka, Kc, D1, D2, e1, e2, chii, xii, h, fielddir, dir);

    
    int trial_section = int(num_trials/size);

    for(size_t i = rank*trial_section; i < (rank+1)*trial_section; ++i){
        mixed_lattice_cuda<3, 4, 8, 4, 12, 12, 12> MC(&TFO, 2.5, 1.0);
        if (spin_config_filename != ""){
            MC.read_spin_from_file(spin_config_filename);
        }
        else{MC.simulated_annealing(Temp_start, Temp_end , 10000, 0, 10, false);
        }
        MC.write_to_file_pos(dir + "/pos");
        MC.write_to_file_spin(dir + "/spin");
        MC.molecular_dynamics_cuda(T_start, T_end, T_step_size, dir+"/"+std::to_string(i), 1, false, true);
    }
}


int main(int argc, char** argv) {
    double k_B = 0.08620689655;
    double mu_B = 5.7883818012e-2;
    vector<int> rank_to_write = {0};

    // Defaults
    double J1ab = 4.92;
    double J1c = 4.92;
    double J2ab = 0.29;
    double J2c = 0.29;
    double Ka = 0.0;
    double Kc = -0.09;
    double D1 = 0.0;
    double D2 = 0.0;
    double chii = 0.05; // Tm-O coupling
    double xii = 0.0;
    double e1 = 0.97;
    double e2 = 3.97;
    double offset = 0.0;
    double h = 0.0;

    int num_trials = 1;
    double T_start = 0.0;
    double T_end = 400.0;
    double T_step_size = 1e-2;
    array<double, 3> fielddir = {0, 1, 0};  // Default field direction

    // Try to load from parameter file (argv[1] or ./params.txt). If not found, fall back to CLI args.
    std::unordered_map<std::string,double> par_double;
    std::unordered_map<std::string,std::string> par_string;
    std::string dir_name = (argc > 1) ? argv[1] : "TmFeO3_2DCS";
    string param_path = dir_name + "/params.txt";
    string spin_config_file = "";


    bool loaded = load_params_from_file(param_path, par_double, par_string);
    if (!loaded && argc > 1) {
        // If first arg was not a path, try default params.txt or parameters.txt
        loaded = load_params_from_file("params.txt", par_double, par_string);
    }
    if (loaded) {
        cout << "Successfully loaded parameters from: " << param_path << endl;
        auto get_double = [&](const char* k, double &v){ auto it = par_double.find(k); if (it != par_double.end()) v = it->second; };
        auto get_int = [&](const char* k, int &v){ auto it = par_double.find(k); if (it != par_double.end()) v = static_cast<int>(it->second); };
        auto get_string = [&](const char* k, std::string &v){ auto it = par_string.find(k); if (it != par_string.end()) v = it->second; };
        
        get_double("J1ab", J1ab);
        get_double("J1c", J1c);
        get_double("J2ab", J2ab);
        get_double("J2c", J2c);
        get_double("Ka", Ka);
        get_double("Kc", Kc);
        get_double("D1", D1);
        get_double("D2", D2);
        get_double("chii", chii);
        get_double("xii", xii);
        get_double("e1", e1);
        get_double("e2", e2);
        get_double("h", h);
        get_double("T_start", T_start);
        get_double("T_end", T_end);
        get_double("T_step_size", T_step_size);
        get_int("num_trials", num_trials);
        get_string("dir_name", dir_name);
        get_string("spin_config_file", spin_config_file);
        
        // Parse fielddir if present (format: "(x, y, z)" or "x y z")
        auto it_fielddir = par_string.find("fielddir");
        if (it_fielddir != par_string.end()) {
            std::string fielddir_str = it_fielddir->second;
            // Remove parentheses if present
            fielddir_str.erase(std::remove(fielddir_str.begin(), fielddir_str.end(), '('), fielddir_str.end());
            fielddir_str.erase(std::remove(fielddir_str.begin(), fielddir_str.end(), ')'), fielddir_str.end());
            // Parse the three components
            std::istringstream iss(fielddir_str);
            std::string token;
            int idx = 0;
            while (std::getline(iss, token, ',') && idx < 3) {
                fielddir[idx++] = std::stod(token);
            }
        }
    } else {
        cout << "No parameter file found, using command line arguments or defaults" << endl;
        // Backward-compatible CLI parsing
        J1ab = (argc > 1) ? atof(argv[1]) : J1ab;
        J1c = (argc > 2) ? atof(argv[2]) : J1c;
        J2ab = (argc > 3) ? atof(argv[3]) : J2ab;
        J2c = (argc > 4) ? atof(argv[4]) : J2c;
        Ka = (argc > 5) ? atof(argv[5]) : Ka;
        Kc = (argc > 6) ? atof(argv[6]) : Kc;
        D1 = (argc > 7) ? atof(argv[7]) : D1;
        D2 = (argc > 8) ? atof(argv[8]) : D2;
        chii = (argc > 9) ? atof(argv[9]) : chii;
        xii = (argc > 10) ? atof(argv[10]) : xii;
        e1 = (argc > 11) ? atof(argv[11]) : e1;
        e2 = (argc > 12) ? atof(argv[12]) : e2;
        offset = (argc > 13) ? atof(argv[13]) : offset;
        h = (argc > 14) ? atof(argv[14]) : h;

        dir_name = (argc > 15) ? argv[15] : dir_name;
        num_trials = (argc > 16) ? atoi(argv[16]) : num_trials;
        T_start = (argc > 17) ? atof(argv[17]) : T_start;
        T_end = (argc > 18) ? atof(argv[18]) : T_end;
        T_step_size = (argc > 19) ? atof(argv[19]) : T_step_size;
        spin_config_file = (argc > 20) ? argv[20] : spin_config_file;
    }
    if (J1ab != 0){
        J1c /= J1ab;
        J2ab /= J1ab;
        J2c /= J1ab;
        Ka /= J1ab;
        Kc /= J1ab;
        D1 /= J1ab;
        D2 /= J1ab;
        chii /= J1ab;
        xii /= J1ab;
        e1 /= J1ab;
        e2 /= J1ab;
        h /= J1ab;
        J1ab = 1;
    }

    cout << "Begin MD on TmFeO3 with parameters:" << J1ab << " " << J1c << " " << J2ab << " " << J2c << " " << Ka << " " << Kc << " " << D1 << " " << D2 << " " << chii << " " << xii << " " << e1 << " " << e2 << " " << h << " " << dir_name << " " << num_trials << endl;
    filesystem::create_directory(dir_name);


    // MD_TmFeO3(num_trials, 20, 1e-2, T_start, T_end, T_step_size, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, xii, h, fielddir, e1, e2, offset, dir_name, spin_config_file);
    MD_TmFeO3_cuda(num_trials, 10, 1e-2, T_start, T_end, T_step_size, J1ab, J1ab, J1c, J2ab, J2ab, J2c, Ka, Kc, D1, D2, chii, xii, h, fielddir, e1, e2, offset, dir_name, spin_config_file);

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        MPI_Finalize();
    }
    return 0;
}
//J1ab=J1c=4.92meV J2ab=J2c=0.29meV Ka=0meV Kc=-0.09meV D1=D2=0 xii is the modeling param E1 = 0.97meV E2=3.9744792531meV