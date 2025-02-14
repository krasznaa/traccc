/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/cuda/seeding/seeding_algorithm.hpp"

#include "seed_finding.hpp"
#include "spacepoint_binning.hpp"

// Project include(s).
#include "traccc/seeding/detail/seeding_config.hpp"

// System include(s).
#include <cmath>

namespace traccc::cuda {

struct seeding_algorithm::impl {
    /// Sub-algorithm performing the spacepoint binning
    spacepoint_binning m_binning;
    /// Sub-algorithm performing the seed finding
    seed_finding m_finding;
};

seeding_algorithm::seeding_algorithm(const seedfinder_config& finder_config,
                                     const spacepoint_grid_config& grid_config,
                                     const seedfilter_config& filter_config,
                                     const traccc::memory_resource& mr,
                                     vecmem::copy& copy, stream& str)
    : m_impl{std::make_unique<impl>(
          spacepoint_binning{finder_config, grid_config, mr, copy, str},
          seed_finding{finder_config, filter_config, mr, copy, str})} {}

seeding_algorithm::~seeding_algorithm() = default;

seeding_algorithm::output_type seeding_algorithm::operator()(
    const edm::spacepoint_collection::const_view& spacepoints_view) const {

    return m_impl->m_finding(spacepoints_view,
                             m_impl->m_binning(spacepoints_view));
}

}  // namespace traccc::cuda
