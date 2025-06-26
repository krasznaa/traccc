/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/finding/finding_config.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/bfield.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

// System include(s).
#include <memory>

namespace traccc::cuda {

/// Algorithm performing the full chain of track reconstruction
///
/// At least as much as is implemented in the project at any given moment.
///
class full_chain_algorithm
    : public algorithm<vecmem::vector<fitting_result<default_algebra>>(
          const edm::silicon_cell_collection::host&)>,
      public messaging {

    public:
    /// @name Type declaration(s)
    /// @{

    /// (Host) Detector type used during track finding and fitting
    using host_detector_type = traccc::default_detector::host;

    /// @}

    /// Algorithm constructor
    ///
    /// @param mr The memory resource to use for the intermediate and result
    ///           objects
    ///
    full_chain_algorithm(vecmem::memory_resource& host_mr,
                         const clustering_config& clustering_config,
                         const seedfinder_config& finder_config,
                         const spacepoint_grid_config& grid_config,
                         const seedfilter_config& filter_config,
                         const finding_config& finding_config,
                         const fitting_config& fitting_config,
                         const silicon_detector_description::host& det_descr,
                         const bfield& field, host_detector_type* detector,
                         std::unique_ptr<const traccc::Logger> logger);

    /// Move constructor
    full_chain_algorithm(full_chain_algorithm&&);

    /// Algorithm destructor
    ~full_chain_algorithm();

    /// Move assignment operator
    full_chain_algorithm& operator=(full_chain_algorithm&&);

    /// Reconstruct track parameters in the entire detector
    ///
    /// @param cells The cells for every detector module in the event
    /// @return The track parameters reconstructed
    ///
    output_type operator()(
        const edm::silicon_cell_collection::host& cells) const override;

    private:
    /// Implementation type for the class
    struct impl;
    /// Implementation object
    std::unique_ptr<impl> m_impl;

};  // class full_chain_algorithm

}  // namespace traccc::cuda
