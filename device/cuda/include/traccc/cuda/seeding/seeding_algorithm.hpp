/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/seeding/device/seeding_algorithm.hpp"

namespace traccc::cuda {

/// Main algorithm for performing the track seeding on an NVIDIA GPU
///
/// This algorithm returns a buffer which is not necessarily filled yet. A
/// synchronisation statement is required before destroying this buffer.
///
class seeding_algorithm : public device::seeding_algorithm {

    public:
    /// Constructor for the seed finding algorithm
    ///
    /// @param mr The memory resource to use
    /// @param mr The memory resource(s) to use in the algorithm
    /// @param copy The copy object to use for copying data between device
    ///             and host memory blocks
    /// @param str The CUDA stream to perform the operations in
    ///
    seeding_algorithm(
        const seedfinder_config& finder_config,
        const spacepoint_grid_config& grid_config,
        const seedfilter_config& filter_config, const memory_resource& mr,
        vecmem::copy& copy, stream& str,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    private:
    /// @name Function(s) inherited from @c traccc::device::seeding_algorithm
    /// @{

    /// Spacepoint grid capacity counting kernel launcher
    ///
    /// @param n_spacepoints   Number of spacepoints in the event
    /// @param config          The seed finding configuration
    /// @param phi_axis        The phi axis of the spacepoint grid
    /// @param z_axis          The z axis of the spacepoint grid
    /// @param spacepoints     The view of spacepoints in the event
    /// @param grid_capacities The buffer to write the grid capacities into
    ///
    void count_grid_capacities_kernel(
        edm::spacepoint_collection::const_view::size_type n_spacepoints,
        const seedfinder_config& config,
        const details::spacepoint_grid_types::host::axis_p0_type& phi_axis,
        const details::spacepoint_grid_types::host::axis_p1_type& z_axis,
        const edm::spacepoint_collection::const_view& spacepoints,
        vecmem::data::vector_view<unsigned int>& grid_capacities)
        const override;

    /// Spacepoint grid population kernel launcher
    ///
    /// @param n_spacepoints   Number of spacepoints in the event
    /// @param config          The seed finding configuration
    /// @param spacepoints     The view of spacepoints in the event
    /// @param grid            The spacepoint grid to populate
    /// @param grid_prefix_sum A prefix sum describing the grid contents
    ///
    void populate_grid_kernel(
        edm::spacepoint_collection::const_view::size_type n_spacepoints,
        const seedfinder_config& config,
        const edm::spacepoint_collection::const_view& spacepoints,
        details::spacepoint_grid_types::view& grid,
        vecmem::data::vector_view<prefix_sum_element_t>& grid_prefix_sum)
        const override;

    /// Doublet counting kernel launcher
    ///
    /// @param n_spacepoints   Number of spacepoints in the spacepoint grid
    /// @param config          The seed finding configuration
    /// @param spacepoints     The view of spacepoints in the event
    /// @param grid            The populated spacepoint grid
    /// @param grid_prefix_sum A prefix sum describing the grid contents
    /// @param doublet_counter The doublet counter collection to fill
    /// @param nMidBot         The number of middle-bottom doublets found
    /// @param nMidTop         The number of middle-top doublets found
    ///
    void count_doublets_kernel(
        edm::spacepoint_collection::const_view::size_type n_spacepoints,
        const seedfinder_config& config,
        const edm::spacepoint_collection::const_view& spacepoints,
        const details::spacepoint_grid_types::const_view& grid,
        const vecmem::data::vector_view<const device::prefix_sum_element_t>&
            grid_prefix_sum,
        doublet_counter_collection_types::view& doublet_counter,
        unsigned int& nMidBot, unsigned int& nMidTop) const override;

    /// Doublet finding kernel launcher
    ///
    /// @param n_doublets      Number of doublets counted earlier
    /// @param config          The seed finding configuration
    /// @param spacepoints     The view of spacepoints in the event
    /// @param grid            The populated spacepoint grid
    /// @param doublet_counter The doublet counter collection
    /// @param mb_doublets     The middle-bottom doublet collection to fill
    /// @param mt_doublets     The middle-top doublet collection to fill
    ///
    void find_doublets_kernel(
        device::doublet_counter_collection_types::const_view::size_type
            n_doublets,
        const seedfinder_config& config,
        const edm::spacepoint_collection::const_view& spacepoints,
        const details::spacepoint_grid_types::const_view& grid,
        const doublet_counter_collection_types::const_view& doublet_counter,
        device_doublet_collection_types::view& mb_doublets,
        device_doublet_collection_types::view& mt_doublets) const override;

    /// Triplet counting kernel launcher
    ///
    /// @param nMidBot         Number of middle-bottom doublets found earlier
    /// @param config          The seed finding configuration
    /// @param spacepoints     The view of spacepoints in the event
    /// @param grid            The populated spacepoint grid
    /// @param doublet_counter The doublet counter collection
    /// @param mb_doublets     The middle-bottom doublet collection
    /// @param mt_doublets     The middle-top doublet collection
    /// @param spM_counter     The triplet counter per middle spacepoint to fill
    /// @param midBot_counter  The triplet counter per middle-bottom doublet to
    ///                        fill
    ///
    void count_triplets_kernel(
        unsigned int nMidBot, const seedfinder_config& config,
        const edm::spacepoint_collection::const_view& spacepoints,
        const details::spacepoint_grid_types::const_view& grid,
        const doublet_counter_collection_types::const_view& doublet_counter,
        const device_doublet_collection_types::const_view& mb_doublets,
        const device_doublet_collection_types::const_view& mt_doublets,
        triplet_counter_spM_collection_types::view& spM_counter,
        triplet_counter_collection_types::view& midBot_counter) const override;

    /// Triplet count reduction kernel launcher
    ///
    /// @param n_doublets      Number of doublets found earlier
    /// @param doublet_counter The doublet counter collection
    /// @param spM_counter     The triplet counter per middle spacepoint
    /// @param nTriplets       The total number of triplets found
    ///
    void triplet_counts_reduction_kernel(
        device::doublet_counter_collection_types::const_view::size_type
            n_doublets,
        const doublet_counter_collection_types::const_view& doublet_counter,
        triplet_counter_spM_collection_types::view& spM_counter,
        unsigned int& nTriplets) const override;

    /// Triplet finding kernel launcher
    ///
    /// @param nMidBot         Number of trilets per middle-bottom doublets
    /// @param finding_config  The seed finding configuration
    /// @param filter_config   The seed filtering configuration
    /// @param spacepoints     The view of spacepoints in the event
    /// @param grid            The populated spacepoint grid
    /// @param doublet_counter The doublet counter collection
    /// @param mt_doublets     The middle-top doublet collection
    /// @param spM_tc          The triplet counter per middle spacepoint
    /// @param midBot_tc       The triplet counter per middle-bottom doublet
    /// @param triplets        The triplet collection to fill
    ///
    void find_triplets_kernel(
        unsigned int nMidBot, const seedfinder_config& finding_config,
        const seedfilter_config& filter_config,
        const edm::spacepoint_collection::const_view& spacepoints,
        const details::spacepoint_grid_types::const_view& grid,
        const doublet_counter_collection_types::const_view& doublet_counter,
        const device_doublet_collection_types::const_view& mt_doublets,
        const triplet_counter_spM_collection_types::const_view& spM_tc,
        const triplet_counter_collection_types::const_view& midBot_tc,
        device_triplet_collection_types::view& triplets) const override;

    /// Triplet weight updater/filler kernel launcher
    ///
    /// @param n_triplets     Number of triplets found earlier
    /// @param config         The seed filtering configuration
    /// @param spacepoints    The view of spacepoints in the event
    /// @param spM_tc         The triplet counter per middle spacepoint
    /// @param midBot_tc      The triplet counter per middle-bottom doublet
    /// @param triplets       The triplet collection to update
    ///
    void update_triplet_weights_kernel(
        device_triplet_collection_types::const_view::size_type n_triplets,
        const seedfilter_config& config,
        const edm::spacepoint_collection::const_view& spacepoints,
        const triplet_counter_spM_collection_types::const_view& spM_tc,
        const triplet_counter_collection_types::const_view& midBot_tc,
        device_triplet_collection_types::view& triplets) const override;

    /// Seed selection/filling kernel launcher
    ///
    /// @param n_doublets      Number of doublets found earlier
    /// @param finder_config   The seed finding configuration
    /// @param filter_config   The seed filtering configuration
    /// @param spacepoints     The view of spacepoints in the event
    /// @param grid            The populated spacepoint grid
    /// @param spM_tc          The triplet counter per middle spacepoint
    /// @param midBot_tc       The triplet counter per middle-bottom doublet
    /// @param triplets        The triplet collection
    /// @param seeds           The seed collection to fill
    ///
    void select_seeds_kernel(
        device::doublet_counter_collection_types::const_view::size_type
            n_doublets,
        const seedfinder_config& finder_config,
        const seedfilter_config& filter_config,
        const edm::spacepoint_collection::const_view& spacepoints,
        const details::spacepoint_grid_types::const_view& grid,
        const triplet_counter_spM_collection_types::const_view& spM_tc,
        const triplet_counter_collection_types::const_view& midBot_tc,
        const device_triplet_collection_types::const_view& triplets,
        edm::seed_collection::view& seeds) const override;

    /// @}

    /// The CUDA stream to use
    std::reference_wrapper<stream> m_stream;

};  // class seeding_algorithm

}  // namespace traccc::cuda
