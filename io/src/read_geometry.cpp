/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_geometry.hpp"

#include "traccc/io/details/read_surfaces.hpp"
#include "traccc/io/utils.hpp"

// Detray include(s).
#include <detray/io/frontend/detector_reader.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <stdexcept>

namespace {

/// Helper function constructing @c traccc::geometry from Detray JSON
std::tuple<traccc::geometry,
           std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>>,
           std::unique_ptr<detray::detector<>>>
read_json_geometry(std::string_view filename) {

    // Memory resource used while reading the detector JSON.
    vecmem::host_memory_resource host_mr;

    // Construct a detector object.
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(std::string{filename});
    auto [detector, _] =
        detray::io::read_detector<detray::detector<>>(host_mr, reader_cfg);

    // Construct an "old style geometry" from the detector object.
    traccc::geometry surface_transforms =
        traccc::io::alt_read_geometry(detector);

    // Construct a map from Acts surface identifiers to Detray barcodes.
    auto barcode_map =
        std::make_unique<std::map<std::uint64_t, detray::geometry::barcode>>();
    for (const auto& surface : detector.surfaces()) {
        (*barcode_map)[surface.source] = surface.barcode();
    }

    // Return the created objects.
    return {surface_transforms, std::move(barcode_map),
            std::make_unique<detray::detector<>>(std::move(detector))};
}

}  // namespace

namespace traccc::io {

std::tuple<geometry,
           std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>>,
           std::unique_ptr<detray::detector<>>>
read_geometry(std::string_view filename, data_format format) {

    // Construct the full file name.
    const std::string full_filename = data_directory() + filename.data();

    // Decide how to read the file.
    switch (format) {
        case data_format::csv:
            return {geometry{details::read_surfaces(full_filename, format)},
                    nullptr, nullptr};
        case data_format::json:
            return ::read_json_geometry(full_filename);
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
