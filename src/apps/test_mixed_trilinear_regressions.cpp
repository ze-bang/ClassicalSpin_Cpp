#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/mixed_lattice.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

using Lattice = MixedLattice;

constexpr double kAbsTol = 1e-11;
constexpr double kRelTol = 1e-10;

struct FeTmBond {
    int fe;
    int tm;
    Eigen::Vector3i off;
};

struct FeTmBondPair {
    int orbit;
    FeTmBond even;
    FeTmBond odd;
};

bool nearly_equal(double lhs, double rhs, double abs_tol, double rel_tol) {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= abs_tol + rel_tol * scale;
}

bool matrix_nearly_equal(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs,
                         double abs_tol, double rel_tol) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        return false;
    }
    for (Eigen::Index row = 0; row < lhs.rows(); ++row) {
        for (Eigen::Index col = 0; col < lhs.cols(); ++col) {
            if (!nearly_equal(lhs(row, col), rhs(row, col), abs_tol, rel_tol)) {
                return false;
            }
        }
    }
    return true;
}

bool vector_nearly_equal(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs,
                         double abs_tol, double rel_tol) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (Eigen::Index i = 0; i < lhs.size(); ++i) {
        if (!nearly_equal(lhs(i), rhs(i), abs_tol, rel_tol)) {
            return false;
        }
    }
    return true;
}

std::array<Eigen::Matrix3d, 4> transported_frames() {
    std::array<Eigen::Matrix3d, 4> frames;
    frames[0] = Eigen::Vector3d(+1.0, +1.0, +1.0).asDiagonal();
    frames[1] = Eigen::Vector3d(+1.0, -1.0, -1.0).asDiagonal();
    frames[2] = Eigen::Vector3d(-1.0, +1.0, -1.0).asDiagonal();
    frames[3] = Eigen::Vector3d(-1.0, -1.0, +1.0).asDiagonal();
    return frames;
}

const std::array<FeTmBondPair, 16>& kminus_bond_pairs() {
    static const std::array<FeTmBondPair, 16> pairs = {{
        {1, {0, 3, { -1,  0,  0}}, {0, 0, {  0,  0,  0}}},
        {2, {0, 2, {  0,  0,  0}}, {0, 1, { -1,  0,  0}}},
        {3, {0, 1, {  0,  0,  0}}, {0, 2, { -1,  0,  0}}},
        {4, {0, 0, {  0, -1,  0}}, {0, 3, { -1,  1,  0}}},
        {1, {1, 2, {  0,  0,  0}}, {1, 1, {  0, -1,  0}}},
        {2, {1, 3, {  0,  0,  0}}, {1, 0, {  0, -1,  0}}},
        {3, {1, 0, {  1, -1,  0}}, {1, 3, { -1,  0,  0}}},
        {4, {1, 1, {  0,  0,  0}}, {1, 2, {  0, -1,  0}}},
        {1, {2, 1, {  0, -1,  0}}, {2, 2, {  0,  0, -1}}},
        {2, {2, 0, {  0, -1, -1}}, {2, 3, {  0,  0,  0}}},
        {3, {2, 3, { -1,  0,  0}}, {2, 0, {  1, -1, -1}}},
        {4, {2, 2, {  0, -1, -1}}, {2, 1, {  0,  0,  0}}},
        {1, {3, 0, {  0,  0, -1}}, {3, 3, { -1,  0,  0}}},
        {2, {3, 1, { -1,  0,  0}}, {3, 2, {  0,  0, -1}}},
        {3, {3, 2, { -1,  0, -1}}, {3, 1, {  0,  0,  0}}},
        {4, {3, 3, { -1,  1,  0}}, {3, 0, {  0, -1, -1}}}
    }};
    return pairs;
}

SpinConfig make_base_config() {
    SpinConfig config;
    config.field_strength = 0.0;
    config.spin_length = 1.0f;
    config.spin_length_su3 = 1.0f;

    config.set_param("J1ab", 0.0);
    config.set_param("J1c", 0.0);
    config.set_param("J2ab", 0.0);
    config.set_param("J2c", 0.0);
    config.set_param("Ka", 0.0);
    config.set_param("Kb", 0.0);
    config.set_param("Kc", 0.0);
    config.set_param("D1", 0.0);
    config.set_param("D2", 0.0);
    config.set_param("e1", 0.0);
    config.set_param("e2", 0.0);

    return config;
}

Eigen::Matrix3d reference_kminus() {
    Eigen::Matrix3d K;
    K <<  0.31, -0.23,  0.17,
         -0.19,  0.29, -0.11,
          0.07, -0.13,  0.37;
    return K;
}

Eigen::Matrix3d reference_kappaE() {
    Eigen::Matrix3d K;
    K <<  0.09, -0.07,  0.05,
          0.04,  0.11, -0.03,
         -0.08,  0.02,  0.13;
    return K;
}

Eigen::MatrixXd reference_kappaB() {
    Eigen::MatrixXd K(3, 5);
    K <<  0.06, -0.04,  0.03, -0.02,  0.05,
         -0.01,  0.08, -0.07,  0.09, -0.06,
          0.10, -0.03,  0.04,  0.02,  0.07;
    return K;
}

std::array<Eigen::Matrix3d, 5> reference_W() {
    std::array<Eigen::Matrix3d, 5> W;
    W[0] <<  0.05,  0.01, -0.02,
              0.01, -0.03,  0.04,
             -0.02,  0.04,  0.07;
    W[1] << -0.02,  0.03,  0.01,
              0.03,  0.06, -0.05,
              0.01, -0.05,  0.04;
    W[2] <<  0.08, -0.01,  0.06,
             -0.01,  0.02,  0.03,
              0.06,  0.03, -0.04;
    W[3] << -0.06,  0.05, -0.02,
              0.05,  0.01,  0.07,
             -0.02,  0.07,  0.03;
    W[4] <<  0.03, -0.04,  0.02,
             -0.04,  0.09, -0.01,
              0.02, -0.01, -0.05;
    return W;
}

void set_reference_kminus_terms(SpinConfig& config) {
    const Eigen::Matrix3d K = reference_kminus();
    const char* axes[3] = {"x", "y", "z"};
    const char* lambdas[3] = {"2", "5", "7"};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            config.set_param(std::string("Kminus_") + lambdas[col] + axes[row], K(row, col));
        }
    }
}

void set_reference_higher_order_terms(SpinConfig& config) {
    const Eigen::Matrix3d KE = reference_kappaE();
    const Eigen::MatrixXd KB = reference_kappaB();
    const auto W = reference_W();
    const char* axes[3] = {"x", "y", "z"};
    const char* minus_lambdas[3] = {"2", "5", "7"};
    const char* plus_lambdas[5] = {"1", "3", "4", "6", "8"};
    const char* comps[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            config.set_param(std::string("kappaE_") + minus_lambdas[col] + axes[row], KE(row, col));
        }
        for (int col = 0; col < 5; ++col) {
            config.set_param(std::string("kappaB_") + plus_lambdas[col] + axes[row], KB(row, col));
        }
    }

    for (int channel = 0; channel < 5; ++channel) {
        const Eigen::Matrix3d& M = W[channel];
        const double values[6] = {M(0, 0), M(1, 1), M(2, 2), M(0, 1), M(0, 2), M(1, 2)};
        for (int comp = 0; comp < 6; ++comp) {
            config.set_param(std::string("W") + plus_lambdas[channel] + "_" + comps[comp], values[comp]);
        }
    }
}

void set_active_kminus_orbit(SpinConfig& config, int orbit) {
    config.set_param("Kminus_orbit1_scale", orbit == 1 ? 1.0 : 0.0);
    config.set_param("Kminus_orbit2_scale", orbit == 2 ? 1.0 : 0.0);
    config.set_param("Kminus_orbit3_scale", orbit == 3 ? 1.0 : 0.0);
    config.set_param("Kminus_orbit4_scale", orbit == 4 ? 1.0 : 0.0);
}

SpinConfig make_kminus_config() {
    SpinConfig config = make_base_config();
    set_reference_kminus_terms(config);
    return config;
}

SpinConfig make_higher_order_config() {
    SpinConfig config = make_base_config();
    set_reference_higher_order_terms(config);
    return config;
}

Eigen::MatrixXd expected_kminus_tensor(int fe_sub, bool odd, double scale = 1.0) {
    constexpr int LAM2 = 1;
    constexpr int LAM5 = 4;
    constexpr int LAM7 = 6;

    const auto R = transported_frames();
    Eigen::Matrix3d M = Eigen::Matrix3d::Identity();
    if (odd) {
        M(1, 1) = -1.0;
        M(2, 2) = -1.0;
    }

    const Eigen::Matrix3d K_bond = scale * R[fe_sub] * reference_kminus() * M;
    Eigen::MatrixXd tensor = Eigen::MatrixXd::Zero(3, 8);
    tensor.col(LAM2) = K_bond.col(0);
    tensor.col(LAM5) = K_bond.col(1);
    tensor.col(LAM7) = K_bond.col(2);
    return tensor;
}

Eigen::MatrixXd expected_kappaE_tensor(int fe_sub, bool odd, double scale = 1.0) {
    constexpr int LAM2 = 1;
    constexpr int LAM5 = 4;
    constexpr int LAM7 = 6;
    const auto R = transported_frames();
    Eigen::Matrix3d M = Eigen::Matrix3d::Identity();
    if (odd) {
        M(0, 0) = -1.0;
    }
    const Eigen::Matrix3d K_bond = scale * R[fe_sub] * reference_kappaE() * M;
    Eigen::MatrixXd tensor = Eigen::MatrixXd::Zero(3, 8);
    tensor.col(LAM2) = K_bond.col(0);
    tensor.col(LAM5) = K_bond.col(1);
    tensor.col(LAM7) = K_bond.col(2);
    return tensor;
}

Eigen::MatrixXd expected_kappaB_tensor(int fe_sub, bool odd, double scale = 1.0) {
    constexpr int cols[5] = {0, 2, 3, 5, 7};
    const auto R = transported_frames();
    Eigen::MatrixXd M = Eigen::MatrixXd::Identity(5, 5);
    if (odd) {
        M(2, 2) = -1.0;
        M(3, 3) = -1.0;
    }
    const Eigen::MatrixXd K_bond = scale * R[fe_sub] * reference_kappaB() * M;
    Eigen::MatrixXd tensor = Eigen::MatrixXd::Zero(3, 8);
    for (int col = 0; col < 5; ++col) {
        tensor.col(cols[col]) = K_bond.col(col);
    }
    return tensor;
}

SpinTensor3 expected_W_tensor(int fe_sub, bool odd, double scale = 1.0) {
    constexpr int cols[5] = {0, 2, 3, 5, 7};
    const auto R = transported_frames();
    const auto W = reference_W();
    double signs[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    if (odd) {
        signs[2] = -1.0;
        signs[3] = -1.0;
    }
    SpinTensor3 tensor(3);
    for (int alpha = 0; alpha < 3; ++alpha) {
        tensor[alpha] = Eigen::MatrixXd::Zero(3, 8);
    }
    for (int channel = 0; channel < 5; ++channel) {
        const Eigen::Matrix3d W_bond = scale * signs[channel] * R[fe_sub] * W[channel] * R[fe_sub];
        for (int alpha = 0; alpha < 3; ++alpha) {
            for (int beta = 0; beta < 3; ++beta) {
                tensor[alpha](beta, cols[channel]) = W_bond(alpha, beta);
            }
        }
    }
    return tensor;
}

const MixedBilinear* find_bilinear(const MixedUnitCell& mixed_uc, const FeTmBond& bond) {
    const auto range = mixed_uc.bilinear_SU2_SU3.equal_range(bond.fe);
    for (auto iter = range.first; iter != range.second; ++iter) {
        const MixedBilinear& term = iter->second;
        if (static_cast<int>(term.partner) == bond.tm && term.offset == bond.off) {
            return &term;
        }
    }
    return nullptr;
}

const MixedBilinearDrive* find_drive_bilinear(const MixedUnitCell& mixed_uc,
                                              const FeTmBond& bond,
                                              int envelope) {
    const auto range = mixed_uc.bilinear_drive_SU2_SU3.equal_range(bond.fe);
    for (auto iter = range.first; iter != range.second; ++iter) {
        const MixedBilinearDrive& term = iter->second;
        if (static_cast<int>(term.partner) == bond.tm && term.offset == bond.off
            && term.envelope == envelope) {
            return &term;
        }
    }
    return nullptr;
}

const MixedTrilinear* find_trilinear(const MixedUnitCell& mixed_uc, const FeTmBond& bond) {
    const auto range = mixed_uc.trilinear_SU2_SU3.equal_range(bond.fe);
    for (auto iter = range.first; iter != range.second; ++iter) {
        const MixedTrilinear& term = iter->second;
        if (static_cast<int>(term.partner1) == bond.fe
            && static_cast<int>(term.partner2) == bond.tm
            && term.offset1 == Eigen::Vector3i::Zero()
            && term.offset2 == bond.off) {
            return &term;
        }
    }
    return nullptr;
}

bool tensor_nearly_equal(const SpinTensor3& lhs, const SpinTensor3& rhs,
                         double abs_tol, double rel_tol) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!matrix_nearly_equal(lhs[i], rhs[i], abs_tol, rel_tol)) {
            return false;
        }
    }
    return true;
}

Eigen::VectorXd normalized_vector(const std::vector<double>& components, double length) {
    Eigen::VectorXd vec(components.size());
    for (size_t i = 0; i < components.size(); ++i) {
        vec(static_cast<Eigen::Index>(i)) = components[i];
    }

    const double norm = vec.norm();
    if (norm == 0.0) {
        vec.setZero();
        vec(0) = length;
        return vec;
    }
    return vec * (length / norm);
}

Eigen::VectorXd su2_precession(const Eigen::VectorXd& field, const Eigen::VectorXd& spin) {
    Eigen::VectorXd dsdt(3);
    dsdt(0) = field(1) * spin(2) - field(2) * spin(1);
    dsdt(1) = field(2) * spin(0) - field(0) * spin(2);
    dsdt(2) = field(0) * spin(1) - field(1) * spin(0);
    return dsdt;
}

Eigen::VectorXd su3_precession(const Eigen::VectorXd& field, const Eigen::VectorXd& spin) {
    const auto& f = get_SU3_structure();
    Eigen::VectorXd dsdt = Eigen::VectorXd::Zero(field.size());
    for (Eigen::Index i = 0; i < field.size(); ++i) {
        for (Eigen::Index j = 0; j < field.size(); ++j) {
            for (Eigen::Index k = 0; k < field.size(); ++k) {
                dsdt(i) += f[static_cast<size_t>(i)](j, k) * field(j) * spin(k);
            }
        }
    }
    return dsdt;
}

Lattice make_lattice_from_config(const SpinConfig& config) {
    return Lattice(build_tmfeo3(config), 1, 1, 1,
                   config.spin_length, config.spin_length_su3);
}

void assign_deterministic_spins(Lattice& lattice) {
    for (size_t site = 0; site < lattice.lattice_size_SU2; ++site) {
        const double x = std::sin(0.41 * (site + 1) + 0.2);
        const double y = std::cos(0.67 * (site + 1) - 0.5);
        const double z = std::sin(0.29 * (site + 1) + 1.1);
        lattice.spins_SU2[site] = normalized_vector({x, y, z}, lattice.spin_length_SU2);
    }

    for (size_t site = 0; site < lattice.lattice_size_SU3; ++site) {
        std::vector<double> comps(lattice.spin_dim_SU3);
        for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
            comps[d] = std::sin(0.23 * (site + 1) + 0.37 * (d + 1))
                     + std::cos(0.19 * (site + 1) - 0.11 * (d + 1));
        }
        lattice.spins_SU3[site] = normalized_vector(comps, lattice.spin_length_SU3);
    }
}

bool check_kminus_builder_surface(std::ostream& out) {
    const MixedUnitCell mixed_uc = build_tmfeo3(make_kminus_config());
    if (mixed_uc.bilinear_SU2_SU3.size() != 32) {
        out << "[FAIL] Kminus builder emitted " << mixed_uc.bilinear_SU2_SU3.size()
            << " mixed bilinears instead of 32\n";
        return false;
    }
    if (!mixed_uc.trilinear_SU2_SU3.empty()) {
        out << "[FAIL] Kminus-only builder emitted mixed trilinears: "
            << mixed_uc.trilinear_SU2_SU3.size() << "\n";
        return false;
    }
    if (!mixed_uc.bilinear_drive_SU2_SU3.empty()) {
        out << "[FAIL] Kminus-only builder emitted driven mixed bilinears: "
            << mixed_uc.bilinear_drive_SU2_SU3.size() << "\n";
        return false;
    }

    for (const auto& pair : kminus_bond_pairs()) {
        const MixedBilinear* even = find_bilinear(mixed_uc, pair.even);
        const MixedBilinear* odd = find_bilinear(mixed_uc, pair.odd);
        if (even == nullptr || odd == nullptr) {
            out << "[FAIL] Missing Kminus bond for orbit " << pair.orbit
                << " Fe" << pair.even.fe << "\n";
            return false;
        }
        if (!matrix_nearly_equal(even->interaction,
                                 expected_kminus_tensor(pair.even.fe, false),
                                 kAbsTol, kRelTol)) {
            out << "[FAIL] Even Kminus tensor mismatch for orbit " << pair.orbit
                << " Fe" << pair.even.fe << " -> Tm" << pair.even.tm << "\n";
            return false;
        }
        if (!matrix_nearly_equal(odd->interaction,
                                 expected_kminus_tensor(pair.odd.fe, true),
                                 kAbsTol, kRelTol)) {
            out << "[FAIL] Odd Kminus tensor mismatch for orbit " << pair.orbit
                << " Fe" << pair.odd.fe << " -> Tm" << pair.odd.tm << "\n";
            return false;
        }
    }

    return true;
}

bool check_orbit_specific_override(std::ostream& out) {
    SpinConfig config = make_base_config();
    config.set_param("Kminus_2x", 100.0);
    config.set_param("Kminus2_2x", 0.42);
    set_active_kminus_orbit(config, 2);

    const MixedUnitCell mixed_uc = build_tmfeo3(config);
    if (mixed_uc.bilinear_SU2_SU3.size() != 8) {
        out << "[FAIL] Single active Kminus orbit emitted "
            << mixed_uc.bilinear_SU2_SU3.size() << " bonds instead of 8\n";
        return false;
    }

    Eigen::Matrix3d K_ref = Eigen::Matrix3d::Zero();
    K_ref(0, 0) = 0.42;
    const auto R = transported_frames();
    const Eigen::Matrix3d M_odd = Eigen::Vector3d(+1.0, -1.0, -1.0).asDiagonal();

    for (const auto& pair : kminus_bond_pairs()) {
        if (pair.orbit != 2) {
            continue;
        }
        const MixedBilinear* even = find_bilinear(mixed_uc, pair.even);
        const MixedBilinear* odd = find_bilinear(mixed_uc, pair.odd);
        if (even == nullptr || odd == nullptr) {
            out << "[FAIL] Missing orbit-specific Kminus bond for Fe" << pair.even.fe << "\n";
            return false;
        }
        Eigen::MatrixXd expected_even = Eigen::MatrixXd::Zero(3, 8);
        expected_even.col(1) = (R[pair.even.fe] * K_ref).col(0);
        Eigen::MatrixXd expected_odd = Eigen::MatrixXd::Zero(3, 8);
        expected_odd.col(1) = (R[pair.odd.fe] * K_ref * M_odd).col(0);
        if (!matrix_nearly_equal(even->interaction, expected_even, kAbsTol, kRelTol)
            || !matrix_nearly_equal(odd->interaction, expected_odd, kAbsTol, kRelTol)) {
            out << "[FAIL] Orbit-specific Kminus override was not applied correctly\n";
            return false;
        }
    }

    return true;
}

bool check_higher_order_builder_surface(std::ostream& out) {
    const MixedUnitCell mixed_uc = build_tmfeo3(make_higher_order_config());
    if (!mixed_uc.bilinear_SU2_SU3.empty()) {
        out << "[FAIL] Higher-order-only builder emitted static mixed bilinears: "
            << mixed_uc.bilinear_SU2_SU3.size() << "\n";
        return false;
    }
    if (mixed_uc.bilinear_drive_SU2_SU3.size() != 64) {
        out << "[FAIL] Higher-order builder emitted " << mixed_uc.bilinear_drive_SU2_SU3.size()
            << " driven mixed bilinears instead of 64\n";
        return false;
    }
    if (mixed_uc.trilinear_SU2_SU3.size() != 32) {
        out << "[FAIL] Higher-order builder emitted " << mixed_uc.trilinear_SU2_SU3.size()
            << " mixed trilinears instead of 32\n";
        return false;
    }

    for (const auto& pair : kminus_bond_pairs()) {
        const MixedBilinearDrive* even_E = find_drive_bilinear(mixed_uc, pair.even, 0);
        const MixedBilinearDrive* odd_E = find_drive_bilinear(mixed_uc, pair.odd, 0);
        const MixedBilinearDrive* even_B = find_drive_bilinear(mixed_uc, pair.even, 1);
        const MixedBilinearDrive* odd_B = find_drive_bilinear(mixed_uc, pair.odd, 1);
        const MixedTrilinear* even_W = find_trilinear(mixed_uc, pair.even);
        const MixedTrilinear* odd_W = find_trilinear(mixed_uc, pair.odd);
        if (even_E == nullptr || odd_E == nullptr || even_B == nullptr || odd_B == nullptr
            || even_W == nullptr || odd_W == nullptr) {
            out << "[FAIL] Missing higher-order term for orbit " << pair.orbit
                << " Fe" << pair.even.fe << "\n";
            return false;
        }
        if (!matrix_nearly_equal(even_E->interaction,
                                 expected_kappaE_tensor(pair.even.fe, false),
                                 kAbsTol, kRelTol)
            || !matrix_nearly_equal(odd_E->interaction,
                                    expected_kappaE_tensor(pair.odd.fe, true),
                                    kAbsTol, kRelTol)) {
            out << "[FAIL] kappaE tensor mismatch for orbit " << pair.orbit
                << " Fe" << pair.even.fe << "\n";
            return false;
        }
        if (!matrix_nearly_equal(even_B->interaction,
                                 expected_kappaB_tensor(pair.even.fe, false),
                                 kAbsTol, kRelTol)
            || !matrix_nearly_equal(odd_B->interaction,
                                    expected_kappaB_tensor(pair.odd.fe, true),
                                    kAbsTol, kRelTol)) {
            out << "[FAIL] kappaB tensor mismatch for orbit " << pair.orbit
                << " Fe" << pair.even.fe << "\n";
            return false;
        }
        if (!tensor_nearly_equal(even_W->interaction,
                                 expected_W_tensor(pair.even.fe, false),
                                 kAbsTol, kRelTol)
            || !tensor_nearly_equal(odd_W->interaction,
                                    expected_W_tensor(pair.odd.fe, true),
                                    kAbsTol, kRelTol)) {
            out << "[FAIL] W tensor mismatch for orbit " << pair.orbit
                << " Fe" << pair.even.fe << "\n";
            return false;
        }
    }

    return true;
}

bool check_higher_order_orbit_specific_override(std::ostream& out) {
    SpinConfig config = make_base_config();
    config.set_param("kappaB_1x", 100.0);
    config.set_param("kappaB3_1x", 0.25);
    config.set_param("kappaB_orbit1_scale", 0.0);
    config.set_param("kappaB_orbit2_scale", 0.0);
    config.set_param("kappaB_orbit3_scale", 1.0);
    config.set_param("kappaB_orbit4_scale", 0.0);

    const MixedUnitCell mixed_uc = build_tmfeo3(config);
    if (mixed_uc.bilinear_drive_SU2_SU3.size() != 8) {
        out << "[FAIL] Orbit-specific kappaB emitted " << mixed_uc.bilinear_drive_SU2_SU3.size()
            << " driven bonds instead of 8\n";
        return false;
    }

    Eigen::MatrixXd K_ref = Eigen::MatrixXd::Zero(3, 5);
    K_ref(0, 0) = 0.25;
    const auto R = transported_frames();
    Eigen::MatrixXd M_odd = Eigen::MatrixXd::Identity(5, 5);
    M_odd(2, 2) = -1.0;
    M_odd(3, 3) = -1.0;

    for (const auto& pair : kminus_bond_pairs()) {
        if (pair.orbit != 3) {
            continue;
        }
        const MixedBilinearDrive* even = find_drive_bilinear(mixed_uc, pair.even, 1);
        const MixedBilinearDrive* odd = find_drive_bilinear(mixed_uc, pair.odd, 1);
        if (even == nullptr || odd == nullptr) {
            out << "[FAIL] Missing orbit-specific kappaB bond\n";
            return false;
        }
        Eigen::MatrixXd expected_even = Eigen::MatrixXd::Zero(3, 8);
        expected_even.col(0) = (R[pair.even.fe] * K_ref).col(0);
        Eigen::MatrixXd expected_odd = Eigen::MatrixXd::Zero(3, 8);
        expected_odd.col(0) = (R[pair.odd.fe] * K_ref * M_odd).col(0);
        if (!matrix_nearly_equal(even->interaction, expected_even, kAbsTol, kRelTol)
            || !matrix_nearly_equal(odd->interaction, expected_odd, kAbsTol, kRelTol)) {
            out << "[FAIL] Orbit-specific kappaB override was not applied correctly\n";
            return false;
        }
    }

    return true;
}

bool check_driven_bilinear_field_consistency(Lattice& lattice, std::ostream& out) {
    assign_deterministic_spins(lattice);
    const auto state = lattice.spins_to_state();
    const size_t offset_su3 = lattice.lattice_size_SU2 * lattice.spin_dim_SU2;
    const double env_E = 0.37;
    const double env_B = -0.23;

    for (size_t site = 0; site < lattice.lattice_size_SU2; ++site) {
        double H0[3] = {0.0, 0.0, 0.0};
        double Hd[3] = {0.0, 0.0, 0.0};
        lattice.get_local_field_SU2_flat_into(site, state, offset_su3, 0.0, 0.0, H0, 0.0, 0.0);
        lattice.get_local_field_SU2_flat_into(site, state, offset_su3, 0.0, 0.0, Hd, env_E, env_B);
        Eigen::Vector3d expected = Eigen::Vector3d::Zero();
        for (size_t n = 0; n < lattice.mixed_bilinear_drive_partners_SU2[site].size(); ++n) {
            const double env = (lattice.mixed_bilinear_drive_envelope_SU2[site][n] == 0) ? env_E : env_B;
            const size_t partner = lattice.mixed_bilinear_drive_partners_SU2[site][n];
            expected += env * lattice.mixed_bilinear_drive_interaction_SU2[site][n]
                            * lattice.spins_SU3[partner];
        }
        Eigen::Vector3d actual(Hd[0] - H0[0], Hd[1] - H0[1], Hd[2] - H0[2]);
        if (!vector_nearly_equal(actual, expected, kAbsTol, kRelTol)) {
            out << "[FAIL] Driven SU2 local-field contribution mismatch at site " << site << "\n";
            return false;
        }
    }

    for (size_t site = 0; site < lattice.lattice_size_SU3; ++site) {
        double H0[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double Hd[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        lattice.get_local_field_SU3_flat_into(site, state, offset_su3, 0.0, 0.0, H0, 0.0, 0.0);
        lattice.get_local_field_SU3_flat_into(site, state, offset_su3, 0.0, 0.0, Hd, env_E, env_B);
        Eigen::VectorXd expected = Eigen::VectorXd::Zero(8);
        for (size_t n = 0; n < lattice.mixed_bilinear_drive_partners_SU3[site].size(); ++n) {
            const double env = (lattice.mixed_bilinear_drive_envelope_SU3[site][n] == 0) ? env_E : env_B;
            const size_t partner = lattice.mixed_bilinear_drive_partners_SU3[site][n];
            expected += env * lattice.mixed_bilinear_drive_interaction_SU3[site][n]
                            * lattice.spins_SU2[partner];
        }
        Eigen::VectorXd actual(8);
        for (int component = 0; component < 8; ++component) {
            actual(component) = Hd[component] - H0[component];
        }
        if (!vector_nearly_equal(actual, expected, kAbsTol, kRelTol)) {
            out << "[FAIL] Driven SU3 local-field contribution mismatch at site " << site << "\n";
            return false;
        }
    }

    return true;
}

bool check_total_energy_flat_consistency(Lattice& lattice, std::ostream& out) {
    const double energy = lattice.total_energy();
    const auto state = lattice.spins_to_state();
    const double flat_energy = lattice.total_energy_flat(state.data());
    if (!nearly_equal(energy, flat_energy, kAbsTol, kRelTol)) {
        out << "[FAIL] total_energy and total_energy_flat disagree: E=" << energy
            << ", E_flat=" << flat_energy << "\n";
        return false;
    }
    return true;
}

bool check_state_roundtrip(Lattice& lattice, std::ostream& out) {
    const auto state = lattice.spins_to_state();
    Lattice::SpinConfigSU2 spins2;
    Lattice::SpinConfigSU3 spins3;
    lattice.state_to_spins(state, spins2, spins3);

    for (size_t site = 0; site < lattice.lattice_size_SU2; ++site) {
        if (!vector_nearly_equal(spins2[site], lattice.spins_SU2[site], kAbsTol, kRelTol)) {
            out << "[FAIL] SU2 state roundtrip mismatch at site " << site << "\n";
            return false;
        }
    }
    for (size_t site = 0; site < lattice.lattice_size_SU3; ++site) {
        if (!vector_nearly_equal(spins3[site], lattice.spins_SU3[site], kAbsTol, kRelTol)) {
            out << "[FAIL] SU3 state roundtrip mismatch at site " << site << "\n";
            return false;
        }
    }
    return true;
}

bool check_su2_delta_energy(Lattice& lattice, std::ostream& out) {
    for (size_t site = 0; site < lattice.lattice_size_SU2; ++site) {
        const Eigen::VectorXd old_spin = lattice.spins_SU2[site];
        const Eigen::VectorXd new_spin = normalized_vector(
            {std::cos(0.53 * (site + 1) + 0.1),
             std::sin(0.79 * (site + 1) - 0.4),
             std::cos(0.31 * (site + 1) + 0.8)},
            lattice.spin_length_SU2);

        const double energy_before = lattice.total_energy();
        const double delta_energy = lattice.site_energy_SU2_diff(new_spin, old_spin, site);
        lattice.spins_SU2[site] = new_spin;
        const double exact_delta = lattice.total_energy() - energy_before;
        lattice.spins_SU2[site] = old_spin;

        if (!nearly_equal(delta_energy, exact_delta, kAbsTol, kRelTol)) {
            out << "[FAIL] SU2 delta-energy mismatch at site " << site
                << ": dE_site=" << delta_energy << ", dE_exact=" << exact_delta << "\n";
            return false;
        }
    }
    return true;
}

bool check_su3_delta_energy(Lattice& lattice, std::ostream& out) {
    for (size_t site = 0; site < lattice.lattice_size_SU3; ++site) {
        const Eigen::VectorXd old_spin = lattice.spins_SU3[site];
        std::vector<double> comps(lattice.spin_dim_SU3);
        for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
            comps[d] = std::cos(0.17 * (site + 1) + 0.29 * (d + 1))
                     - std::sin(0.13 * (site + 1) - 0.07 * (d + 1));
        }
        const Eigen::VectorXd new_spin = normalized_vector(comps, lattice.spin_length_SU3);

        const double energy_before = lattice.total_energy();
        const double delta_energy = lattice.site_energy_SU3_diff(new_spin, old_spin, site);
        lattice.spins_SU3[site] = new_spin;
        const double exact_delta = lattice.total_energy() - energy_before;
        lattice.spins_SU3[site] = old_spin;

        if (!nearly_equal(delta_energy, exact_delta, kAbsTol, kRelTol)) {
            out << "[FAIL] SU3 delta-energy mismatch at site " << site
                << ": dE_site=" << delta_energy << ", dE_exact=" << exact_delta << "\n";
            return false;
        }
    }
    return true;
}

bool check_llg_consistency(Lattice& lattice, std::ostream& out) {
    lattice.reset_pulse();

    const auto state = lattice.spins_to_state();
    Lattice::ODEState dxdt(state.size(), 0.0);
    lattice.ode_system(state, dxdt, 0.0);

    double energy_rate = 0.0;
    for (size_t site = 0; site < lattice.lattice_size_SU2; ++site) {
        const auto field = lattice.get_local_field_SU2(site);
        const auto expected = su2_precession(field, lattice.spins_SU2[site]);
        Eigen::VectorXd actual(lattice.spin_dim_SU2);
        const size_t idx = site * lattice.spin_dim_SU2;
        for (size_t d = 0; d < lattice.spin_dim_SU2; ++d) {
            actual(static_cast<Eigen::Index>(d)) = dxdt[idx + d];
        }
        if (!vector_nearly_equal(expected, actual, kAbsTol, kRelTol)) {
            out << "[FAIL] SU2 LLG derivative mismatch at site " << site << "\n";
            return false;
        }
        const double norm_rate = lattice.spins_SU2[site].dot(actual);
        if (!nearly_equal(norm_rate, 0.0, kAbsTol, kRelTol)) {
            out << "[FAIL] SU2 norm is not preserved at site " << site << "\n";
            return false;
        }
        energy_rate -= field.dot(actual);
    }

    const size_t offset_su3 = lattice.lattice_size_SU2 * lattice.spin_dim_SU2;
    for (size_t site = 0; site < lattice.lattice_size_SU3; ++site) {
        const auto field = lattice.get_local_field_SU3(site);
        const auto expected = su3_precession(field, lattice.spins_SU3[site]);
        Eigen::VectorXd actual(lattice.spin_dim_SU3);
        const size_t idx = offset_su3 + site * lattice.spin_dim_SU3;
        for (size_t d = 0; d < lattice.spin_dim_SU3; ++d) {
            actual(static_cast<Eigen::Index>(d)) = dxdt[idx + d];
        }
        if (!vector_nearly_equal(expected, actual, kAbsTol, kRelTol)) {
            out << "[FAIL] SU3 LLG derivative mismatch at site " << site << "\n";
            return false;
        }
        const double norm_rate = lattice.spins_SU3[site].dot(actual);
        if (!nearly_equal(norm_rate, 0.0, kAbsTol, kRelTol)) {
            out << "[FAIL] SU3 norm is not preserved at site " << site << "\n";
            return false;
        }
        energy_rate -= field.dot(actual);
    }

    if (!nearly_equal(energy_rate, 0.0, 5e-11, 5e-10)) {
        out << "[FAIL] LLG instantaneous energy drift is nonzero: dE/dt="
            << energy_rate << "\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!check_kminus_builder_surface(std::cout)) {
        return 1;
    }
    if (!check_orbit_specific_override(std::cout)) {
        return 1;
    }
    if (!check_higher_order_builder_surface(std::cout)) {
        return 1;
    }
    if (!check_higher_order_orbit_specific_override(std::cout)) {
        return 1;
    }

    Lattice lattice = make_lattice_from_config(make_kminus_config());
    assign_deterministic_spins(lattice);

    if (!check_state_roundtrip(lattice, std::cout)) {
        return 1;
    }
    if (!check_total_energy_flat_consistency(lattice, std::cout)) {
        return 1;
    }
    if (!check_su2_delta_energy(lattice, std::cout)) {
        return 1;
    }
    if (!check_su3_delta_energy(lattice, std::cout)) {
        return 1;
    }
    if (!check_llg_consistency(lattice, std::cout)) {
        return 1;
    }

    Lattice higher_order_lattice = make_lattice_from_config(make_higher_order_config());
    assign_deterministic_spins(higher_order_lattice);
    if (!check_driven_bilinear_field_consistency(higher_order_lattice, std::cout)) {
        return 1;
    }
    if (!check_total_energy_flat_consistency(higher_order_lattice, std::cout)) {
        return 1;
    }
    if (!check_su2_delta_energy(higher_order_lattice, std::cout)) {
        return 1;
    }
    if (!check_su3_delta_energy(higher_order_lattice, std::cout)) {
        return 1;
    }
    if (!check_llg_consistency(higher_order_lattice, std::cout)) {
        return 1;
    }

    std::cout << "[PASS] Kminus and higher-order mixed coupling regression checks\n";
    return 0;
}
