/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/spacepoint_formation.hpp"

namespace traccc {

spacepoint_formation::spacepoint_formation(vecmem::memory_resource& mr)
    : m_mr(mr) {}

spacepoint_formation::output_type spacepoint_formation::operator()(
    const measurement_collection_types::host& measurements,
    const edm::pixel_module_container::host& modules) const {

    // Create the result container.
    output_type result(&(m_mr.get()));

    // Iterate over the measurements.
    for (std::size_t i = 0; i < measurements.size(); ++i) {

        // Access the measurements of the current module.
        const measurement& this_measurement = measurements.at(i);

        // Transform measurement position to 3D
        point3 local_3d = {this_measurement.local[0], this_measurement.local[1],
                           0.};
        point3 global = edm::pixel_module_container::placement::get(modules)
                            .at(this_measurement.module_link)
                            .point_to_global(local_3d);

        // Fill result with this spacepoint
        result.push_back({global, this_measurement});
    }

    // Return the created container.
    return result;
}

}  // namespace traccc
