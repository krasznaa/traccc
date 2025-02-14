/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/cuda/utils/stream.hpp"

// Project include(s).
#include "traccc/edm/seed_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/seeding/detail/seeding_config.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <memory>

namespace traccc::cuda {

/// Main algorithm for performing the track seeding on an NVIDIA GPU
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
class seeding_algorithm : public algorithm<edm::seed_collection::buffer(
                              const edm::spacepoint_collection::const_view&)> {

    public:
    /// Constructor for the seed finding algorithm
    ///
    /// @param mr The memory resource to use
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    ///
    seeding_algorithm(const seedfinder_config& finder_config,
                      const spacepoint_grid_config& grid_config,
                      const seedfilter_config& filter_config,
                      const traccc::memory_resource& mr, vecmem::copy& copy,
                      stream& str);
    /// Destructor
    ~seeding_algorithm() override;

    /// Operator executing the algorithm.
    ///
    /// @param spacepoints is a view of all spacepoints in the event
    /// @return the buffer of track seeds reconstructed from the spacepoints
    ///
    output_type operator()(const edm::spacepoint_collection::const_view&
                               spacepoints) const override;

    private:
    /// Internal data for the algorithm
    struct impl;
    /// Pointer to the internal data
    std::unique_ptr<impl> m_impl;

};  // class seeding_algorithm

}  // namespace traccc::cuda
