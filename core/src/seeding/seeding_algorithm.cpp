/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/seeding_algorithm.hpp"

#include "seed_finding.hpp"
#include "spacepoint_binning.hpp"

// System include(s).
#include <functional>

namespace traccc::host {

struct seeding_algorithm::impl {

    spacepoint_binning m_binning;
    seed_finding m_finding;

};  // struct seeding_algorithm::impl

seeding_algorithm::seeding_algorithm(const seedfinder_config& finder_config,
                                     const spacepoint_grid_config& grid_config,
                                     const seedfilter_config& filter_config,
                                     vecmem::memory_resource& mr)
    : m_impl{std::make_unique<impl>(
          spacepoint_binning{finder_config, grid_config, mr},
          seed_finding{finder_config, filter_config, mr})} {}

seeding_algorithm::seeding_algorithm(seeding_algorithm&&) = default;

seeding_algorithm::~seeding_algorithm() = default;

seeding_algorithm& seeding_algorithm::operator=(seeding_algorithm&&) = default;

seeding_algorithm::output_type seeding_algorithm::operator()(
    const edm::spacepoint_collection::const_view& spacepoints) const {

    return m_impl->m_finding(spacepoints, m_impl->m_binning(spacepoints));
}

}  // namespace traccc::host
