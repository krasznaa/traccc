/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include <functional>

#include "tests/cca_test.hpp"
#include "traccc/cuda/clusterization/experimental/clusterization_algorithm.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/utils/cuda/copy.hpp>

namespace {
vecmem::cuda::host_memory_resource host_mr;
vecmem::cuda::device_memory_resource device_mr;
traccc::memory_resource mr{device_mr, &host_mr};
vecmem::cuda::copy copy;
traccc::cuda::stream stream;
traccc::cuda::experimental::clusterization_algorithm cc{mr, copy, stream, 1024};

cca_function_t f = [](const traccc::edm::pixel_cell_container::host& cells,
                      const traccc::edm::pixel_module_container::host&
                          modules) {
    std::map<traccc::geometry_id, vecmem::vector<traccc::measurement>> result;

    traccc::edm::pixel_cell_container::buffer cells_buffer{
        static_cast<traccc::edm::pixel_cell_container::buffer::size_type>(
            cells.size()),
        device_mr};
    copy(vecmem::get_data(cells), cells_buffer);
    traccc::edm::pixel_module_container::buffer modules_buffer(
        static_cast<traccc::edm::pixel_module_container::buffer::size_type>(
            modules.size()),
        device_mr);
    copy(vecmem::get_data(modules), modules_buffer);
    traccc::measurement_collection_types::buffer measurements_buffer =
        cc(cells_buffer, modules_buffer);
    traccc::measurement_collection_types::host measurements;
    copy(measurements_buffer, measurements);

    for (std::size_t i = 0; i < measurements.size(); i++) {
        result[traccc::edm::pixel_module_container::surface_link::get(modules)
                   .at(measurements.at(i).module_link)
                   .value()]
            .push_back(measurements.at(i));
    }

    return result;
};
}  // namespace

TEST_P(ConnectedComponentAnalysisTests, Run) {
    test_connected_component_analysis(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    FastSvAlgorithm, ConnectedComponentAnalysisTests,
    ::testing::Combine(
        ::testing::Values(f),
        ::testing::ValuesIn(ConnectedComponentAnalysisTests::get_test_files())),
    ConnectedComponentAnalysisTests::get_test_name);