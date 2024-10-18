/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/definitions/math.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/seed.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/edm/track_parameters.hpp"

// System include(s).
#include <cmath>

namespace traccc::details {

/// helper functions (for both cpu and gpu) to perform conformal transformation
///
/// @param x is the x value
/// @param y is the y value
/// @return is the conformal transformation result
///
inline TRACCC_HOST_DEVICE vector2 uv_transform(scalar x, scalar y) {

    const scalar denominator = x * x + y * y;
    return {x / denominator, y / denominator};
}

/// helper functions (for both cpu and gpu) to calculate bound track parameters
/// at the bottom spacepoint
///
/// @param measurements The measurement collection
/// @param spacepoints  The spacepoint collection
/// @param seed         The seed to calculate the bound track parameters for
/// @param bfield       The magnetic field
/// @return The bound track parameters at the bottom spacepoint of a track seed
///
inline TRACCC_HOST_DEVICE bound_vector seed_to_bound_vector(
    const edm::measurement_collection::const_device& measurements,
    const spacepoint_collection_types::const_device& spacepoints,
    const seed& seed, const vector3& bfield) {

    const auto& spB =
        spacepoints.at(static_cast<unsigned int>(seed.spB_link));
    const auto& spM =
        spacepoints.at(static_cast<unsigned int>(seed.spM_link));
    const auto& spT =
        spacepoints.at(static_cast<unsigned int>(seed.spT_link));

    darray<vector3, 3> sp_global_positions;
    sp_global_positions[0] = spB.global;
    sp_global_positions[1] = spM.global;
    sp_global_positions[2] = spT.global;

    // Define a new coordinate frame with its origin at the bottom space
    // point, z axis long the magnetic field direction and y axis
    // perpendicular to vector from the bottom to middle space point.
    // Hence, the projection of the middle space point on the tranverse
    // plane will be located at the x axis of the new frame.
    const vector3 relVec = sp_global_positions[1] - sp_global_positions[0];
    const vector3 newZAxis = vector::normalize(bfield);
    const vector3 newYAxis = vector::normalize(vector::cross(newZAxis, relVec));
    const vector3 newXAxis = vector::cross(newYAxis, newZAxis);

    // The center of the new frame is at the bottom space point
    const transform3 trans(sp_global_positions[0], newZAxis, newXAxis);

    // The coordinate of the middle and top space point in the new frame
    const auto local1 = trans.point_to_local(sp_global_positions[1]);
    const auto local2 = trans.point_to_local(sp_global_positions[2]);

    // The uv1.y() should be zero
    const vector2 uv1 = uv_transform(local1[0], local1[1]);
    const vector2 uv2 = uv_transform(local2[0], local2[1]);

    // A,B are slope and intercept of the straight line in the u,v plane
    // connecting the three points
    const scalar A = (uv2[1] - uv1[1]) / (uv2[0] - uv1[0]);
    const scalar B = uv2[1] - A * uv2[0];

    // Radius (with a sign)
    const scalar R = -getter::perp(vector2{1.f, A}) / (2.f * B);
    // The (1/tanTheta) of momentum in the new frame
    const scalar invTanTheta =
        local2[2] / (2.f * R * math::asin(getter::perp(local2) / (2.f * R)));

    // The momentum direction in the new frame (the center of the circle
    // has the coordinate (-1.*A/(2*B), 1./(2*B)))
    const vector3 transDirection =
        vector3({1.f, A, scalar(getter::perp(vector2{1.f, A})) * invTanTheta});
    // Transform it back to the original frame
    const vector3 direction =
        transform3::rotate(trans._data, vector::normalize(transDirection));

    // Create the result object.
    bound_vector params;

    // The estimated phi and theta
    getter::element(params, e_bound_phi, 0) = getter::phi(direction);
    getter::element(params, e_bound_theta, 0) = getter::theta(direction);

    // The measured loc0 and loc1
    const auto meas_for_spB = measurements.at(spB.measurement_index);
    getter::element(params, e_bound_loc0, 0) = meas_for_spB.local()[0];
    getter::element(params, e_bound_loc1, 0) = meas_for_spB.local()[1];

    // The estimated q/pt in [GeV/c]^-1 (note that the pt is the
    // projection of momentum on the transverse plane of the new frame)
    scalar qOverPt = 1.f / (R * getter::norm(bfield));
    // The estimated q/p in [GeV/c]^-1
    getter::element(params, e_bound_qoverp, 0) =
        qOverPt / getter::perp(vector2{1.f, invTanTheta});

    // Make sure the time is a finite value
    assert(std::isfinite(getter::element(params, e_bound_time, 0)));

    return params;
}

}  // namespace traccc::details
