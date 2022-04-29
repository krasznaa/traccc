/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/cuda/seeding/seed_finding.hpp"
#include "traccc/cuda/utils/definitions.hpp"

// Project include(s).
#include "traccc/cuda/seeding/detail/doublet_counter.hpp"
#include "traccc/cuda/seeding/doublet_counting.hpp"
#include "traccc/device/get_prefix_sum.hpp"
#include "traccc/edm/device/doublet_counter.hpp"
#include "traccc/seeding/device/count_doublets.hpp"
// #include "traccc/cuda/seeding/doublet_finding.hpp"
// #include "traccc/cuda/seeding/seed_selecting.hpp"
// #include "traccc/cuda/seeding/triplet_counting.hpp"
// #include "traccc/cuda/seeding/triplet_finding.hpp"
// #include "traccc/cuda/seeding/weight_updating.hpp"

// VecMem include(s).
#include "vecmem/utils/copy.hpp"

// System include(s).
#include <iostream>
#include <string>
#include <vector>

/// Temporary macro for comparing the old and new code
#define CHECK(EXP)                                        \
    do {                                                  \
        const bool result = EXP;                          \
        if (!result) {                                    \
            std::cerr << "Failed: " << #EXP << std::endl; \
        }                                                 \
    } while (false)

namespace traccc::cuda {
namespace kernels {

/// CUDA kernel for running @c traccc::device::count_doublets
__global__ void count_doublets(
    seedfinder_config config, sp_grid_const_view sp_grid,
    vecmem::data::vector_view<const device::prefix_sum_element_t> sp_prefix_sum,
    device::doublet_counter_container_view doublet_counter) {

    device::count_doublets(threadIdx.x + blockIdx.x * blockDim.x, config,
                           sp_grid, sp_prefix_sum, doublet_counter);
}

}  // namespace kernels

seed_finding::seed_finding(const seedfinder_config& config,
                           vecmem::memory_resource& mr)
    : m_seedfinder_config(config), m_mr(mr) {}

seed_finding::output_type seed_finding::operator()(
    const host_spacepoint_container& spacepoints,
    const sp_grid_const_view& g2_view) const {

    // Helper object for the data management.
    vecmem::copy copy;

    // Get the prefix sum for the spacepoint grid.
    const device::prefix_sum_t sp_grid_prefix_sum =
        device::get_prefix_sum(g2_view._data_view, m_mr.get(), copy);
    auto sp_grid_prefix_sum_view = vecmem::get_data(sp_grid_prefix_sum);

    // Set up the doublet counter buffer.
    auto sp_grid_sizes = copy.get_sizes(g2_view._data_view);
    const device::doublet_counter_container_buffer::header_vector::size_type
        doublet_counter_buffer_size = sp_grid_sizes.size();
    device::doublet_counter_container_buffer doublet_counter_buffer{
        {doublet_counter_buffer_size, m_mr.get()},
        {std::vector<std::size_t>(doublet_counter_buffer_size, 0),
         std::vector<std::size_t>(sp_grid_sizes.begin(), sp_grid_sizes.end()),
         m_mr.get()}};
    copy.setup(doublet_counter_buffer.headers);
    copy.setup(doublet_counter_buffer.items);

    // Make sure that the summary values are set to zero in the counter buffer.
    std::memset(
        doublet_counter_buffer.headers.ptr(), 0,
        doublet_counter_buffer_size * sizeof(device::doublet_counter_header));

    // Calculate the number of threads and thread blocks to run the kernels for.
    const unsigned int nThreads = WARP_SIZE * 2;
    const unsigned int nBlocks = sp_grid_prefix_sum.size() / nThreads + 1;

    // Count the number of doublets that we need to produce.
    kernels::count_doublets<<<nThreads, nBlocks>>>(
        m_seedfinder_config, g2_view, vecmem::get_data(sp_grid_prefix_sum),
        doublet_counter_buffer);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Get the summary values per bin.
    vecmem::vector<device::doublet_counter_header> doublet_counts(
        &(m_mr.get()));
    copy(doublet_counter_buffer.headers, doublet_counts);

    {
        unsigned int nbins = g2_view._data_view.m_size;

        // Fill the size vector for double counter container
        std::vector<size_t> n_spm_per_bin;
        n_spm_per_bin.reserve(nbins);
        for (unsigned int i = 0; i < nbins; ++i) {
            n_spm_per_bin.push_back(g2_view._data_view.m_ptr[i].size());
        }

        // Create doublet counter container buffer
        doublet_counter_container_buffer dcc_buffer{
            {nbins, m_mr.get()}, {n_spm_per_bin, m_mr.get()}};
        copy.setup(dcc_buffer.headers);

        // Run doublet counting
        traccc::cuda::doublet_counting(m_seedfinder_config, g2_view, dcc_buffer,
                                       m_mr.get());

        // Take header of the doublet counter container into host
        vecmem::vector<doublet_counter_per_bin> dcc_headers(&m_mr.get());
        copy(dcc_buffer.headers, dcc_headers);

        // Compare the "new" implementation with the "old" one.
        CHECK(doublet_counts.size() == dcc_headers.size());
        for (std::size_t i = 0; i < doublet_counts.size(); ++i) {
            CHECK(doublet_counts[i].m_nSpM == dcc_headers[i].n_spM);
            CHECK(doublet_counts[i].m_nMidBot == dcc_headers[i].n_mid_bot);
            CHECK(doublet_counts[i].m_nMidTop == dcc_headers[i].n_mid_top);
        }
    }

    // // Fill the size vectors for doublet containers
    // std::vector<size_t> n_mid_bot_per_bin;
    // std::vector<size_t> n_mid_top_per_bin;
    // n_mid_bot_per_bin.reserve(nbins);
    // n_mid_top_per_bin.reserve(nbins);
    // for (const auto& h : dcc_headers) {
    //     n_mid_bot_per_bin.push_back(h.n_mid_bot);
    //     n_mid_top_per_bin.push_back(h.n_mid_top);
    // }

    // // Create doublet container buffer
    // doublet_container_buffer mbc_buffer{{nbins, m_mr.get()},
    //                                     {n_mid_bot_per_bin, m_mr.get()}};
    // doublet_container_buffer mtc_buffer{{nbins, m_mr.get()},
    //                                     {n_mid_top_per_bin, m_mr.get()}};
    // copy.setup(mbc_buffer.headers);

    // // Run doublet finding
    // traccc::cuda::doublet_finding(m_seedfinder_config, dcc_headers, g2_view,
    //                               dcc_buffer, mbc_buffer, mtc_buffer,
    //                               m_mr.get());

    // // Take header of the middle-bottom doublet container buffer into host
    // vecmem::vector<doublet_per_bin> mbc_headers(&m_mr.get());
    // copy(mbc_buffer.headers, mbc_headers);

    // // Create triplet counter container buffer
    // triplet_counter_container_buffer tcc_buffer{
    //     {nbins, m_mr.get()}, {n_mid_bot_per_bin, m_mr.get()}};
    // copy.setup(tcc_buffer.headers);

    // // Run triplet counting
    // traccc::cuda::triplet_counting(m_seedfinder_config, mbc_headers, g2_view,
    //                                dcc_buffer, mbc_buffer, mtc_buffer,
    //                                tcc_buffer, m_mr.get());

    // // Take header of the triplet counter container buffer into host
    // vecmem::vector<triplet_counter_per_bin> tcc_headers(&m_mr.get());
    // copy(tcc_buffer.headers, tcc_headers);

    // // Fill the size vector for triplet container
    // std::vector<size_t> n_triplets_per_bin;
    // n_triplets_per_bin.reserve(nbins);
    // for (const auto& h : tcc_headers) {
    //     n_triplets_per_bin.push_back(h.n_triplets);
    // }

    // // Create triplet container buffer
    // triplet_container_buffer tc_buffer{{nbins, m_mr.get()},
    //                                    {n_triplets_per_bin, m_mr.get()}};
    // copy.setup(tc_buffer.headers);

    // // Run triplet finding
    // traccc::cuda::triplet_finding(
    //     m_seedfinder_config, m_seedfilter_config, tcc_headers, g2_view,
    //     dcc_buffer, mbc_buffer, mtc_buffer, tcc_buffer, tc_buffer,
    //     m_mr.get());

    // // Take header of the triplet container buffer into host
    // vecmem::vector<triplet_per_bin> tc_headers(&m_mr.get());
    // copy(tc_buffer.headers, tc_headers);

    // // Run weight updating
    // traccc::cuda::weight_updating(m_seedfilter_config, tc_headers, g2_view,
    //                               tcc_buffer, tc_buffer, m_mr.get());

    // // Get the number of seeds (triplets)
    // auto n_triplets = std::accumulate(n_triplets_per_bin.begin(),
    //                                   n_triplets_per_bin.end(), 0);

    // // Create seed buffer
    // vecmem::data::vector_buffer<seed> seed_buffer(n_triplets, 0, m_mr.get());
    // copy.setup(seed_buffer);

    // // Run seed selecting
    // traccc::cuda::seed_selecting(m_seedfilter_config, dcc_headers,
    // spacepoints,
    //                              g2_view, dcc_buffer, tcc_buffer, tc_buffer,
    //                              seed_buffer, m_mr.get());

    // // Take seed buffer into seed collection
    // host_seed_collection seed_collection(&m_mr.get());
    // copy(seed_buffer, seed_collection);

    host_seed_collection seed_collection(&m_mr.get());
    return seed_collection;
}

}  // namespace traccc::cuda
