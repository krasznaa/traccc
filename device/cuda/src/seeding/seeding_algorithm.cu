/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/global_index.hpp"
#include "../utils/utils.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"

// Project include(s).
#include "traccc/seeding/device/count_grid_capacities.hpp"
#include "traccc/seeding/device/populate_grid.hpp"

namespace traccc::cuda {
namespace kernels {

/// CUDA kernel for running @c traccc::device::count_grid_capacities
__global__ void count_grid_capacities(
    seedfinder_config config,
    details::spacepoint_grid_types::host::axis_p0_type phi_axis,
    details::spacepoint_grid_types::host::axis_p1_type z_axis,
    edm::spacepoint_collection::const_view spacepoints,
    vecmem::data::vector_view<unsigned int> grid_capacities) {

    device::count_grid_capacities(details::global_index1(), config, phi_axis,
                                  z_axis, spacepoints, grid_capacities);
}

/// CUDA kernel for running @c traccc::device::populate_grid
__global__ void populate_grid(
    seedfinder_config config,
    edm::spacepoint_collection::const_view spacepoints,
    traccc::details::spacepoint_grid_types::view grid) {

    device::populate_grid(details::global_index1(), config, spacepoints, grid);
}

}  // namespace kernels

seeding_algorithm::seeding_algorithm(const seedfinder_config& finder_config,
                                     const spacepoint_grid_config& grid_config,
                                     const seedfilter_config& filter_config,
                                     const traccc::memory_resource& mr,
                                     vecmem::copy& copy, stream& str,
                                     std::unique_ptr<const Logger> logger)
    : device::seeding_algorithm(finder_config, grid_config, filter_config, mr,
                                copy, std::move(logger)),
      m_stream{str} {}

void seeding_algorithm::count_grid_capacities_kernel(
    edm::spacepoint_collection::const_view::size_type n_spacepoints,
    const seedfinder_config& config,
    const details::spacepoint_grid_types::host::axis_p0_type& phi_axis,
    const details::spacepoint_grid_types::host::axis_p1_type& z_axis,
    const edm::spacepoint_collection::const_view& spacepoints,
    vecmem::data::vector_view<unsigned int>& grid_capacities) const {

    const unsigned int num_threads = m_warp_size * 8;
    const unsigned int num_blocks =
        (n_spacepoints + num_threads - 1) / num_threads;
    kernels::count_grid_capacities<<<num_blocks, num_threads, 0, stream>>>(
        m_config, m_axes.first, m_axes.second, spacepoints_view,
        grid_capacities_view);
    TRACCC_CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace traccc::cuda
