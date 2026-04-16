/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::edm {

TRACCC_HOST_DEVICE
track_state_collection::device::object_type make_track_state(
    [[maybe_unused]] const measurement_collection::const_device& measurements,
    unsigned int mindex) {

    // Create the result object.
    track_state_collection::device::object_type state;

    // Set it not to be a hole by default, with the appropriate (measurement)
    // index.
    state.set_hole(false);
    state.measurement_index() = mindex;

    // Return the initialized state.
    return state;
}

}  // namespace traccc::edm
