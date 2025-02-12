/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "seed_finding.hpp"

namespace traccc::host {

seed_finding::seed_finding(const seedfinder_config& finder_config,
                           const seedfilter_config& filter_config,
                           vecmem::memory_resource& mr)
    : m_midBot_finding(finder_config, mr),
      m_midTop_finding(finder_config, mr),
      m_triplet_finding(finder_config, filter_config, mr),
      m_seed_filtering(filter_config, mr),
      m_mr{mr} {}

edm::seed_collection::host seed_finding::operator()(
    const edm::spacepoint_collection::const_view& sp_view,
    const details::spacepoint_grid_types::host& sp_grid) const {

    // Create the result collection.
    edm::seed_collection::host seeds{m_mr.get()};

    // Create a device container for the spacepoints.
    const edm::spacepoint_collection::const_device spacepoints{sp_view};

    // Iterate over the spacepoint grid's bins.
    for (unsigned int i = 0; i < sp_grid.nbins(); ++i) {

        // Consider all spacepoints in this bin as "middle" spacepoints in the
        // seed.
        const auto& middle_indices = sp_grid.bin(i);

        // Evaluate these middle spacepoints one-by-one.
        for (unsigned int j = 0; j < middle_indices.size(); ++j) {

            // Internal identifier for this middle spacepoint.
            sp_location spM_location({i, j});

            // middule-bottom doublet search
            const auto mid_bot =
                m_midBot_finding(spacepoints, sp_grid, spM_location);

            if (mid_bot.first.empty()) {
                continue;
            }

            // middule-top doublet search
            const auto mid_top =
                m_midTop_finding(spacepoints, sp_grid, spM_location);

            if (mid_top.first.empty()) {
                continue;
            }

            triplet_collection_types::host triplets{&(m_mr.get())};

            // triplet search from the combinations of two doublets which
            // share middle spacepoint
            for (unsigned int k = 0; k < mid_bot.first.size(); ++k) {

                const doublet& mid_bot_doublet = mid_bot.first[k];
                const lin_circle& mid_bot_lc = mid_bot.second[k];

                const triplet_collection_types::host triplets_for_mid_bot =
                    m_triplet_finding(spacepoints, sp_grid, mid_bot_doublet,
                                      mid_bot_lc, mid_top.first,
                                      mid_top.second);

                triplets.insert(triplets.end(), triplets_for_mid_bot.begin(),
                                triplets_for_mid_bot.end());
            }

            // seed filtering
            m_seed_filtering(spacepoints, sp_grid, triplets, seeds);
        }
    }

    return seeds;
}

}  // namespace traccc::host
