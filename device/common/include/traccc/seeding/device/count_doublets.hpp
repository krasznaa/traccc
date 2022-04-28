/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/get_prefix_sum.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"

// System include(s).
#include <cstddef>

namespace traccc::device {

TRACCC_HOST_DEVICE
void count_doublets(
    std::size_t globalIndex, const seedfinder_config& config,
    const sp_grid_const_view& sp_view,
    doublet_counter_container_view& doublet_view,
    const vecmem::data::vector_view<const prefix_sum_element_t>& sp_prefix_sum);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/seeding/device/impl/count_doublets.ipp"
