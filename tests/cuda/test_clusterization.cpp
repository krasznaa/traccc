/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/cuda/clusterization/experimental/clusterization_algorithm.hpp"
#include "traccc/definitions/common.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// GTest include(s).
#include <gtest/gtest.h>

using namespace traccc;

TEST(clusterization, cuda) {

    // Memory resource used by the EDM.
    vecmem::cuda::managed_memory_resource mng_mr;
    traccc::memory_resource mr{mng_mr};

    // Cuda stream
    traccc::cuda::stream stream;

    // Cuda copy objects
    vecmem::cuda::async_copy copy{stream.cudaStream()};

    // Create cell collection
    traccc::edm::cell_container::host cells{mng_mr};
    cells.resize(8);
    cells.channel0() = {1u, 2u, 3u, 5u, 6u, 6u, 6u, 7u};
    cells.channel1() = {2u, 2u, 2u, 5u, 4u, 5u, 6u, 5u};
    cells.activation() = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};

    // Create module collection
    traccc::edm::cell_module_container::host modules{mng_mr};
    modules.resize(1);

    // Run Clusterization
    traccc::cuda::experimental::clusterization_algorithm ca_cuda(mr, copy,
                                                                 stream, 1024);

    auto measurements_buffer =
        ca_cuda(vecmem::get_data(cells), vecmem::get_data(modules));

    measurement_collection_types::device measurements(measurements_buffer);

    // Check the results
    EXPECT_EQ(copy.get_size(measurements_buffer), 2u);
    std::set<measurement> test;
    test.insert(measurements[0]);
    test.insert(measurements[1]);

    std::set<measurement> ref;
    ref.insert({{2.f, 2.f}, {0.75, 0.0833333}, detray::geometry::barcode{0u}});
    ref.insert(
        {{6.f, 5.f}, {0.483333, 0.483333}, detray::geometry::barcode{0u}});

    EXPECT_EQ(test, ref);
}