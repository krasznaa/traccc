/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/sycl/seeding/seeding_algorithm.hpp"

// Project include(s).
#include "traccc/seeding/detail/seeding_config.hpp"

// System include(s).
#include <cmath>

namespace traccc::sycl {

seeding_algorithm::seeding_algorithm(const seedfinder_config& finder_config,
                                     const spacepoint_grid_config& grid_config,
                                     const seedfilter_config& filter_config,
                                     const traccc::memory_resource& mr,
                                     vecmem::copy& copy,
                                     const queue_wrapper& queue,
                                     std::unique_ptr<const Logger> logger)
    : messaging(logger->clone()),
      m_binning(finder_config, grid_config, mr, copy, queue,
                logger->cloneWithSuffix("BinningAlg")),
      m_finding(finder_config, filter_config, mr, copy, queue,
                logger->cloneWithSuffix("SeedFindingAlg")) {}

seeding_algorithm::output_type seeding_algorithm::operator()(
    const edm::spacepoint_collection::const_view& spacepoints_view) const {

    return m_finding(spacepoints_view, m_binning(spacepoints_view));
}

}  // namespace traccc::sycl
