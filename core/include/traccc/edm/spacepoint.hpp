/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/math.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"

// System include(s).
#include <cmath>

namespace traccc {

/// A spacepoint expressed in global coordinates
///
/// It describes the 3D position of the spacepoint, and the 2D measurement that
/// the spacepoint was created from.
///
struct spacepoint {

    /// The global position of the spacepoint in 3D space
    point3 global{0.f, 0.f, 0.f};
    /// Index of the local measurement of the spacepoint on the detector surface
    unsigned int measurement_index = 0u;

    TRACCC_HOST_DEVICE
    const scalar& x() const { return global[0]; }
    TRACCC_HOST_DEVICE
    const scalar& y() const { return global[1]; }
    TRACCC_HOST_DEVICE
    const scalar& z() const { return global[2]; }
    TRACCC_HOST_DEVICE
    scalar radius() const {
        return std::sqrt(global[0] * global[0] + global[1] * global[1]);
    }
};

/// Comparison / ordering operator for spacepoints
TRACCC_HOST_DEVICE
inline bool operator<(const spacepoint& lhs, const spacepoint& rhs) {

    if (math::fabs(lhs.x() - rhs.x()) > float_epsilon) {
        return (lhs.x() < rhs.x());
    } else if (math::fabs(lhs.y() - rhs.y()) > float_epsilon) {
        return (lhs.y() < rhs.y());
    } else {
        return (lhs.z() < rhs.z());
    }
}

/// Equality operator for spacepoints
TRACCC_HOST_DEVICE
inline bool operator==(const spacepoint& lhs, const spacepoint& rhs) {

    return ((math::fabs(lhs.x() - rhs.x()) < float_epsilon) &&
            (math::fabs(lhs.y() - rhs.y()) < float_epsilon) &&
            (math::fabs(lhs.z() - rhs.z()) < float_epsilon) &&
            (lhs.measurement_index == rhs.measurement_index));
}

/// Declare all spacepoint collection types
using spacepoint_collection_types = collection_types<spacepoint>;

}  // namespace traccc
