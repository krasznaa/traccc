/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"

namespace traccc {

clusterization_algorithm::clusterization_algorithm(vecmem::memory_resource& mr)
    : m_cc(mr), m_mc(mr), m_mr(mr) {}

clusterization_algorithm::output_type clusterization_algorithm::operator()(
    const edm::cell_container::host& cells,
    const edm::cell_module_container::host& modules) const {

    return m_mc(m_cc(cells), cells, modules);
}

}  // namespace traccc
