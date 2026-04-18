#pragma once
/**
 * gneb_math.h — Manifold helpers shared by the two GNEB optimizers.
 *
 * Before this header the SLERP (spherical linear interpolation) on S² was
 * copy-pasted three times across `src/core/gneb.cpp` and
 * `src/core/gneb_strain.cpp` (in `geodesic_interpolation`,
 * `interpolate_path`, and `redistribute_images`). All three copies used the
 * same numeric thresholds for the parallel / antiparallel branches, so a
 * drift between them would silently change the reaction coordinate in one
 * optimizer relative to the other.
 *
 * Both optimizers now delegate here. The full GNEB unification (templating
 * the two optimizer classes on a state-type trait) is tracked as Tier 2
 * deferred work; this header covers the low-risk, high-duplication subset.
 */

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace classical_spin::gneb_math {

/// Safe arccos of a dot product that may have drifted slightly outside
/// [-1, 1] due to floating-point rounding.
inline double safe_acos(double x) {
    return std::acos(std::max(-1.0, std::min(1.0, x)));
}

/// Spherical linear interpolation on S². `a` and `b` must have unit norm.
/// Handles the two degenerate branches (nearly parallel, nearly
/// antiparallel) with the same tolerances the original GNEB code used.
///
/// t = 0 returns a, t = 1 returns b. The result is unit-norm up to the
/// rounding of the underlying sin/cos.
inline Eigen::Vector3d slerp_s2(const Eigen::Vector3d& a,
                                const Eigen::Vector3d& b,
                                double t) {
    const double cos_theta = a.dot(b);

    if (cos_theta > 1.0 - 1e-10) {
        // Nearly parallel — any interpolation is a good approximation.
        return a;
    }
    if (cos_theta < -1.0 + 1e-10) {
        // Nearly antiparallel — build a perpendicular axis and rotate by πt.
        Eigen::Vector3d perp = Eigen::Vector3d(1.0, 0.0, 0.0).cross(a);
        if (perp.norm() < 1e-10) {
            perp = Eigen::Vector3d(0.0, 1.0, 0.0).cross(a);
        }
        perp.normalize();
        const double theta_t = t * M_PI;
        return std::cos(theta_t) * a + std::sin(theta_t) * perp;
    }

    const double theta     = std::acos(cos_theta);
    const double sin_theta = std::sin(theta);
    const double w1 = std::sin((1.0 - t) * theta) / sin_theta;
    const double w2 = std::sin(t        * theta) / sin_theta;
    return w1 * a + w2 * b;
}

/// Geodesic angle (in radians) between two unit vectors, with safe clamp.
inline double geodesic_angle(const Eigen::Vector3d& a,
                             const Eigen::Vector3d& b) {
    return safe_acos(a.dot(b));
}

} // namespace classical_spin::gneb_math
