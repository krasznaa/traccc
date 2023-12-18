/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "traccc/io/read_cells.hpp"

#include "csv/read_cells.hpp"
#include "read_binary.hpp"
#include "traccc/io/utils.hpp"

namespace traccc::io {

void read_cells(edm::cell_container::host& cells,
                edm::cell_module_container::host& modules, std::size_t event,
                std::string_view directory, data_format format,
                const geometry* geom, const digitization_config* dconfig) {

    switch (format) {
        case data_format::csv: {
            read_cells(cells, modules,
                       data_directory() + directory.data() +
                           get_event_filename(event, "-cells.csv"),
                       format, geom, dconfig);
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

void read_cells(edm::cell_container::host& cells,
                edm::cell_module_container::host& modules,
                std::string_view filename, data_format format,
                const geometry* geom, const digitization_config* dconfig) {

    switch (format) {
        case data_format::csv:
            return csv::read_cells(cells, modules, filename, geom, dconfig);

        default:
            throw std::invalid_argument("Unsupported data format");
    }
}

}  // namespace traccc::io
