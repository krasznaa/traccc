/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint.hpp"

namespace traccc::details {

/// Function helping with checking a measurement obejct for spacepoint creation
///
/// @tparam measurement_base_t Base type of the measurement proxy object
///
/// @param[in]  measurement The input measurement
///
/// @return @c true if the measurement is valid for spacepoint creation
///
template <typename measurement_base_t>
TRACCC_HOST_DEVICE bool is_valid_measurement(
    const edm::measurement<measurement_base_t>& meas);

/// Function helping with filling/setting up a spacepoint object
///
/// @tparam detector_t Type of the detector object
///
/// @param[in]  det          The tracking geometry
/// @param[in]  measurements All measurements in the event
/// @param[in]  index        Index of the measurement to convert
///
/// @return The created spacepoint object
///
template <typename detector_t>
TRACCC_HOST_DEVICE spacepoint
create_spacepoint(const detector_t& det,
                  const edm::measurement_collection::const_device& measurements,
                  edm::measurement_collection::const_device::size_type index);

}  // namespace traccc::details

// Include the implementation.
#include "traccc/seeding/impl/spacepoint_formation.ipp"
