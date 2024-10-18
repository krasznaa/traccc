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

namespace traccc::details {

/// (Possibly) Add a spacepoint to a collection
///
/// @param[in]  det         The tracking geometry
/// @param[in]  measurement The measurement to (possibly) create a spacepoints
///                         out of
/// @param[in]  meas_index  The index of the measurement in its collection
/// @param[out] spacepoints The collection to (possibly) add the spacepoint to
/// @return @c true if a spacepoint was added, @c false otherwise
///
template <typename detector_t, typename measurement_base_t,
          typename spacepoint_collection_t>
TRACCC_HOST_DEVICE bool add_spacepoint(
    const detector_t& det,
    const edm::measurement<measurement_base_t>& measurement,
    unsigned int meas_index, spacepoint_collection_t& spacepoints);

}  // namespace traccc::details

// Include the implementation.
#include "traccc/seeding/impl/spacepoint_formation.ipp"
