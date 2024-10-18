/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/spacepoint.hpp"

// Detray include(s).
#include "detray/geometry/tracking_surface.hpp"

namespace traccc::details {

template <typename detector_t, typename measurement_base_t,
          typename spacepoint_collection_t>
TRACCC_HOST_DEVICE bool add_spacepoint(
    const detector_t& det,
    const edm::measurement<measurement_base_t>& measurement,
    unsigned int measurement_index, spacepoint_collection_t& spacepoints) {

    // We use 2D (pixel) measurements only for spacepoint creation
    if (measurement.dimensions() != 2u) {
        return false;
    }

    // Helper object for doing the local-to-global transformation with.
    const detray::tracking_surface sf{det, measurement.geometry_id()};

    // Create the spacepoint.
    spacepoints.push_back(spacepoint{
        sf.bound_to_global({}, measurement.local(), {}), measurement_index});
    return true;
}

}  // namespace traccc::details
