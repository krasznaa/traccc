/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/component_connection.hpp"
#include "traccc/clusterization/measurement_creation.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/pixel_cell_container.hpp"
#include "traccc/edm/pixel_module_container.hpp"
#include "traccc/edm/spacepoint.hpp"
#include "traccc/geometry/pixel_data.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

TEST(algorithms, seq_single_module) {

    // Memory resource used in the test.
    vecmem::host_memory_resource resource;

    traccc::component_connection cc(resource);
    traccc::measurement_creation mc(resource);

    /// Following [DOI: 10.1109/DASIP48288.2019.9049184]
    traccc::edm::pixel_cell_container::host cells{resource};
    cells.resize(9);
    traccc::edm::pixel_cell_container::channel0::get(cells) = {1,  8, 10, 9, 10,
                                                               12, 3, 11, 4};
    traccc::edm::pixel_cell_container::channel1::get(cells) = {0,  4,  4,  4, 5,
                                                               12, 13, 13, 14};
    traccc::edm::pixel_cell_container::activation::get(cells) = {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};

    traccc::edm::pixel_module_container::host modules{resource};
    modules.resize(1);

    auto clusters = cc(cells);
    EXPECT_EQ(clusters.size(), 4u);

    auto measurements = mc(clusters, cells, modules);

    EXPECT_EQ(measurements.size(), 4u);
}
