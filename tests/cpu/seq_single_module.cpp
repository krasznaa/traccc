/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/measurement_creation_algorithm.hpp"
#include "traccc/clusterization/sparse_ccl_algorithm.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/silicon_cluster_collection.hpp"
#include "traccc/geometry/silicon_detector_description.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

TEST(algorithms, seq_single_module) {

    // Memory resource used in the test.
    vecmem::host_memory_resource resource;

    traccc::host::sparse_ccl_algorithm cc(resource);
    traccc::host::measurement_creation_algorithm mc(resource);

    /// Following [DOI: 10.1109/DASIP48288.2019.9049184]
    traccc::edm::silicon_cell_collection::host cells{resource};
    cells.push_back({1, 0, 1.f, 0.f, 0});
    cells.push_back({8, 1, 2.f, 0.f, 0});
    cells.push_back({10, 1, 3.f, 0.f, 0});
    cells.push_back({9, 2, 4.f, 0.f, 0});
    cells.push_back({10, 2, 5.f, 0.f, 0});
    cells.push_back({12, 3, 6.f, 0.f, 0});
    cells.push_back({3, 4, 7.f, 0.f, 0});
    cells.push_back({11, 4, 8.f, 0.f, 0});
    cells.push_back({4, 5, 9.f, 0.f, 0});
    traccc::silicon_detector_description::host dd{resource};
    dd.resize(1);

    auto cells_data = vecmem::get_data(cells);
    auto clusters = cc(cells_data);
    EXPECT_EQ(clusters.size(), 4u);

    auto clusters_data = vecmem::get_data(clusters);
    auto dd_data = vecmem::get_data(dd);
    auto measurements = mc(cells_data, clusters_data, dd_data);

    EXPECT_EQ(measurements.size(), 4u);
}
