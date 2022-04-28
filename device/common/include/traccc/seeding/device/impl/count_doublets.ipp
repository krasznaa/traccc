/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_HOST_DEVICE
void count_doublets(std::size_t globalIndex, const seedfinder_config& config, const const_sp_grid_view& sp_view, doublet_counter_container_view& doublet_view, const vecmem::data::vector_view<const prefix_sum_element_t>& sp_prefix_sum) {


}

}  // namespace traccc::device
