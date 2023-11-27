/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/io/data_format.hpp"

// Project include(s).
#include "traccc/edm/pixel_cell_container.hpp"
#include "traccc/edm/pixel_module_container.hpp"
#include "traccc/geometry/geometry.hpp"
#include "traccc/io/digitization_config.hpp"

// System include(s).
#include <cstddef>
#include <string_view>

namespace traccc::io {

/// Read cell data into memory
///
/// The file to read is selected according the naming conventions used in
/// our data.
///
/// @param cells A pixel cell (host) container
/// @param modules A pixel module (host) container
/// @param event The event ID to read in the cells for
/// @param directory The directory holding the cell data files
/// @param format The format of the cell data files (to read)
/// @param geom The description of the detector geometry
/// @param dconfig The detector's digitization configuration
///
void read_cells(edm::pixel_cell_container::host &cells,
                edm::pixel_module_container::host &modules, std::size_t event,
                std::string_view directory,
                data_format format = data_format::csv,
                const geometry *geom = nullptr,
                const digitization_config *dconfig = nullptr);

/// Read cell data into memory
///
/// The file name is selected explicitly by the user.
///
/// @param cells A pixel cell (host) container
/// @param modules A pixel module (host) container
/// @param filename The file to read the cell data from
/// @param format The format of the cell data files (to read)
/// @param geom The description of the detector geometry
/// @param dconfig The detector's digitization configuration
///
void read_cells(edm::pixel_cell_container::host &cells,
                edm::pixel_module_container::host &modules,
                std::string_view filename,
                data_format format = data_format::csv,
                const geometry *geom = nullptr,
                const digitization_config *dconfig = nullptr);

}  // namespace traccc::io
