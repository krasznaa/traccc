/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/seeding/detail/spacepoint_formation.hpp"

namespace traccc::host {

template <typename detector_t>
spacepoint_formation_algorithm<detector_t>::spacepoint_formation_algorithm(
    vecmem::memory_resource& mr)
    : m_mr(mr) {}

template <typename detector_t>
spacepoint_formation_algorithm<detector_t>::output_type
spacepoint_formation_algorithm<detector_t>::operator()(
    const detector_t& det,
    const edm::measurement_collection::const_view& measurements_view) const {

    // Create device containers for the inputs.
    const edm::measurement_collection::const_device measurements{
        measurements_view};

    // Create the result container.
    spacepoint_collection_types::host result(&(m_mr.get()));
    result.reserve(measurements.size());

    // Set up each spacepoint in the result container.
    for (edm::measurement_collection::const_device::size_type i = 0;
         i < measurements.size(); ++i) {
        details::add_spacepoint(det, measurements[i], i, result);
    }

    // Return the created container.
    return result;
}

}  // namespace traccc::host
