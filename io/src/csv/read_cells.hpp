/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/pixel_cell_container.hpp"
#include "traccc/edm/pixel_module_container.hpp"
#include "traccc/geometry/geometry.hpp"
#include "traccc/io/digitization_config.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::csv {

/// Read cell information from a specific CSV file
///
/// @param cells A pixel cell (host) container
/// @param modules A pixel module (host) container
/// @param filename The file to read the cell data from
/// @param geom The description of the detector geometry
/// @param dconfig The detector's digitization configuration
///
void read_cells(traccc::edm::pixel_cell_container::host &cells,
                traccc::edm::pixel_module_container::host &modules,
                std::string_view filename, const geometry* geom = nullptr,
                const digitization_config* dconfig = nullptr);

}  // namespace traccc::io::csv
