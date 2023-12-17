/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/component_connection.hpp"

#include "traccc/clusterization/detail/sparse_ccl.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/containers/vector.hpp>

namespace traccc {

component_connection::output_type component_connection::operator()(
    const edm::cell_container::host& cells) const {

    std::vector<unsigned int> CCL_indices(cells.size());

    // Run SparseCCL to fill CCL indices
    const unsigned int num_clusters = detail::sparse_ccl(
        vecmem::get_data(cells), vecmem::get_data(CCL_indices));

    // Create the result container.
    output_type result(num_clusters, &(m_mr.get()));

    // Add cells to their clusters
    for (std::size_t i = 0; i < CCL_indices.size(); ++i) {
        result.get_items()[CCL_indices[i]].push_back(i);
    }

    return result;
}

}  // namespace traccc
