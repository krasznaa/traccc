/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/write.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// GTest include(s).
#include <gtest/gtest.h>

// System
#include <cstdio>
#include <fstream>

/// Helper function comparing the contents of two pixel cell containers.
void compareCellContainers(const traccc::edm::pixel_cell_container::host& c1,
                           const traccc::edm::pixel_cell_container::host& c2) {
    ASSERT_EQ(c1.size(), c2.size());
    for (std::size_t i = 0; i < c1.size(); i++) {
        EXPECT_EQ(traccc::edm::pixel_cell_container::channel0::get(c1).at(i),
                  traccc::edm::pixel_cell_container::channel0::get(c2).at(i));
        EXPECT_EQ(traccc::edm::pixel_cell_container::channel1::get(c1).at(i),
                  traccc::edm::pixel_cell_container::channel1::get(c2).at(i));
        EXPECT_FLOAT_EQ(
            traccc::edm::pixel_cell_container::activation::get(c1).at(i),
            traccc::edm::pixel_cell_container::activation::get(c2).at(i));
        EXPECT_FLOAT_EQ(traccc::edm::pixel_cell_container::time::get(c1).at(i),
                        traccc::edm::pixel_cell_container::time::get(c2).at(i));
        EXPECT_EQ(
            traccc::edm::pixel_cell_container::module_index::get(c1).at(i),
            traccc::edm::pixel_cell_container::module_index::get(c2).at(i));
    }
}

/// Helper function comapring the contents of two pixel module containers.
void compareModuleContainers(
    const traccc::edm::pixel_module_container::host& m1,
    const traccc::edm::pixel_module_container::host& m2) {
    ASSERT_EQ(m1.size(), m2.size());
    for (std::size_t i = 0; i < m1.size(); i++) {
        EXPECT_EQ(
            traccc::edm::pixel_module_container::surface_link::get(m1).at(i),
            traccc::edm::pixel_module_container::surface_link::get(m2).at(i));
        EXPECT_EQ(
            traccc::edm::pixel_module_container::placement::get(m1).at(i),
            traccc::edm::pixel_module_container::placement::get(m2).at(i));
        EXPECT_FLOAT_EQ(
            traccc::edm::pixel_module_container::threshold::get(m1).at(i),
            traccc::edm::pixel_module_container::threshold::get(m2).at(i));
    }
}

// This defines the local frame test suite for binary cell container
TEST(io_binary, cell) {

    // Set event configuration
    const std::size_t event = 0;
    const std::string cells_directory = "tml_full/ttbar_mu200/";

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read the surface transforms
    auto surface_transforms =
        traccc::io::read_geometry("tml_detector/trackml-detector.csv");

    // Read the digitization configuration file
    auto digi_cfg = traccc::io::read_digitization_config(
        "tml_detector/default-geometric-config-generic.json");

    // Read csv file
    traccc::edm::pixel_cell_container::host cells_csv{host_mr};
    traccc::edm::pixel_module_container::host modules_csv{host_mr};
    traccc::io::read_cells(cells_csv, modules_csv, event, cells_directory,
                           traccc::data_format::csv, &surface_transforms,
                           &digi_cfg);

    // Write binary file
    traccc::io::write(event, cells_directory, traccc::data_format::binary,
                      vecmem::get_data(cells_csv),
                      vecmem::get_data(modules_csv));

    // Read binary file
    traccc::edm::pixel_cell_container::host cells_binary{host_mr};
    traccc::edm::pixel_module_container::host modules_binary{host_mr};
    traccc::io::read_cells(cells_binary, modules_binary, event, cells_directory,
                           traccc::data_format::binary, &surface_transforms,
                           &digi_cfg);

    // Delete binary file
    std::string io_cells_file =
        traccc::io::data_directory() + cells_directory +
        traccc::io::get_event_filename(event, "-cells.dat");
    std::remove(io_cells_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_cells_file));

    std::string io_modules_file =
        traccc::io::data_directory() + cells_directory +
        traccc::io::get_event_filename(event, "-modules.dat");
    std::remove(io_modules_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_modules_file));

    // Check cells size
    EXPECT_GT(cells_csv.size(), 0);
    EXPECT_EQ(cells_csv.size(), cells_binary.size());

    // Check modules size
    EXPECT_GT(modules_csv.size(), 0);
    EXPECT_EQ(modules_csv.size(), modules_binary.size());

    // Compare the cells and the modules.
    compareCellContainers(cells_csv, cells_binary);
    compareModuleContainers(modules_csv, modules_binary);
}

// This defines the local frame test suite for binary spacepoint container
TEST(io_binary, spacepoint) {

    // Set event configuration
    const std::size_t event = 0;
    const std::string hits_directory = "tml_full/ttbar_mu200/";

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read the surface transforms
    auto surface_transforms =
        traccc::io::read_geometry("tml_detector/trackml-detector.csv");

    // Read csv file
    traccc::io::spacepoint_reader_output reader_csv(host_mr);
    traccc::io::read_spacepoints(reader_csv, event, hits_directory,
                                 surface_transforms, traccc::data_format::csv);
    const traccc::spacepoint_collection_types::host& spacepoints_csv =
        reader_csv.spacepoints;
    const traccc::edm::pixel_module_container::host& modules_csv =
        reader_csv.modules;

    // // Write binary file
    traccc::io::write(event, hits_directory, traccc::data_format::binary,
                      vecmem::get_data(spacepoints_csv),
                      vecmem::get_data(modules_csv));

    // Read binary file
    traccc::io::spacepoint_reader_output reader_binary(host_mr);
    traccc::io::read_spacepoints(reader_binary, event, hits_directory,
                                 surface_transforms,
                                 traccc::data_format::binary);
    const traccc::spacepoint_collection_types::host& spacepoints_binary =
        reader_binary.spacepoints;
    const traccc::edm::pixel_module_container::host& modules_binary =
        reader_binary.modules;

    // Delete binary file
    std::string io_spacepoints_file =
        traccc::io::data_directory() + hits_directory +
        traccc::io::get_event_filename(event, "-hits.dat");
    std::remove(io_spacepoints_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_spacepoints_file));

    std::string io_modules_file =
        traccc::io::data_directory() + hits_directory +
        traccc::io::get_event_filename(event, "-modules.dat");
    std::remove(io_modules_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_modules_file));

    // Check spacepoints size
    EXPECT_GT(spacepoints_csv.size(), 0);
    ASSERT_EQ(spacepoints_csv.size(), spacepoints_binary.size());

    // Check modules size
    EXPECT_GT(modules_csv.size(), 0);
    ASSERT_EQ(modules_csv.size(), modules_binary.size());

    for (std::size_t i = 0; i < spacepoints_csv.size(); i++) {
        ASSERT_EQ(spacepoints_csv[i], spacepoints_binary[i]);
    }
    compareModuleContainers(modules_csv, modules_binary);
}

// This defines the local frame test suite for binary measurement container
TEST(io_binary, measurement) {
    // Set event configuration
    const std::size_t event = 0;
    const std::string measurements_directory = "tml_full/ttbar_mu200/";

    // Memory resource used by the EDM.
    vecmem::host_memory_resource host_mr;

    // Read csv file
    traccc::io::measurement_reader_output reader_csv(host_mr);
    traccc::io::read_measurements(reader_csv, event, measurements_directory,
                                  traccc::data_format::csv);
    const traccc::measurement_collection_types::host& measurements_csv =
        reader_csv.measurements;
    const traccc::edm::pixel_module_container::host& modules_csv =
        reader_csv.modules;

    // Write binary file
    traccc::io::write(
        event, measurements_directory, traccc::data_format::binary,
        vecmem::get_data(measurements_csv), vecmem::get_data(modules_csv));

    // Read binary file
    traccc::io::measurement_reader_output reader_binary(host_mr);
    traccc::io::read_measurements(reader_binary, event, measurements_directory,
                                  traccc::data_format::binary);
    const traccc::measurement_collection_types::host& measurements_binary =
        reader_binary.measurements;
    const traccc::edm::pixel_module_container::host& modules_binary =
        reader_binary.modules;

    // Delete binary file
    std::string io_measurements_file =
        traccc::io::data_directory() + measurements_directory +
        traccc::io::get_event_filename(event, "-measurements.dat");
    std::remove(io_measurements_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_measurements_file));

    std::string io_modules_file =
        traccc::io::data_directory() + measurements_directory +
        traccc::io::get_event_filename(event, "-modules.dat");
    std::remove(io_modules_file.c_str());

    ASSERT_TRUE(!std::ifstream(io_modules_file));

    // Check header size
    ASSERT_TRUE(measurements_csv.size() > 0);
    ASSERT_EQ(measurements_csv.size(), measurements_binary.size());

    // Check modules size
    ASSERT_TRUE(modules_csv.size() > 0);
    ASSERT_EQ(modules_csv.size(), modules_binary.size());

    for (std::size_t i = 0; i < measurements_csv.size(); i++) {
        ASSERT_EQ(measurements_csv[i], measurements_binary[i]);
    }
    compareModuleContainers(modules_csv, modules_binary);
}