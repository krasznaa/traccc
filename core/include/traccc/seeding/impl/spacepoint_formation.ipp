/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"

// Detray include(s).
#include "detray/geometry/tracking_surface.hpp"

// System include(s).
#include <cassert>

namespace traccc::details {

template <typename measurement_base_t>
TRACCC_HOST_DEVICE bool is_valid_measurement(
    const edm::measurement<measurement_base_t>& meas) {
    // We use 2D (pixel) measurements only for spacepoint creation
    return (meas.dimensions() == 2u);
}

template <typename detector_t>
TRACCC_HOST_DEVICE spacepoint
create_spacepoint(const detector_t& det,
                  const edm::measurement_collection::const_device& measurements,
                  edm::measurement_collection::const_device::size_type index) {

    // Get the measurement in question.
    const auto meas = measurements.at(index);
    assert(meas.dimensions() == 2u);

    // Make the tracking surface that the measurement sits on.
    const detray::tracking_surface sf{det, meas.geometry_id()};

    // Calculate the global 3D position of the measurement.
    const point3 global = sf.bound_to_global({}, meas.local(), {});

    // Return the spacepoint with this spacepoint
    return spacepoint{global, index};
}

}  // namespace traccc::details
