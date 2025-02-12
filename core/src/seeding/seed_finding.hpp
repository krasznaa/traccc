/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "doublet_finding.hpp"
#include "seed_filtering.hpp"
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/seeding/detail/spacepoint_grid.hpp"
#include "triplet_finding.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <functional>

namespace traccc::host {

/// Seed finding
class seed_finding {

    public:
    /// Constructor for the seed finding
    ///
    /// @param find_config is seed finder configuration parameters
    /// @param filter_config is the seed filter configuration
    /// @param mr The memory resource to use
    ///
    seed_finding(const seedfinder_config& find_config,
                 const seedfilter_config& filter_config,
                 vecmem::memory_resource& mr);

    /// Callable operator for the seed finding
    ///
    /// @param spacepoints All spacepoints in the event
    /// @param sp_grid The same spacepoints arranged in a 2D Phi-Z grid
    /// @return The spacepoint triplets that form the track seeds
    ///
    edm::seed_collection::host operator()(
        const edm::spacepoint_collection::const_view& spacepoints,
        const details::spacepoint_grid_types::host& sp_grid) const;

    private:
    /// Algorithm performing the mid bottom doublet finding
    doublet_finding<details::spacepoint_type::bottom> m_midBot_finding;
    /// Algorithm performing the mid top doublet finding
    doublet_finding<details::spacepoint_type::top> m_midTop_finding;
    /// Algorithm performing the triplet finding
    triplet_finding m_triplet_finding;
    /// Algorithm performing the seed selection
    seed_filtering m_seed_filtering;
    /// The memory resource to use
    std::reference_wrapper<vecmem::memory_resource> m_mr;

};  // class seed_finding

}  // namespace traccc::host
