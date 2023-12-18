/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_cells.hpp"

#include "traccc/io/csv/make_cell_reader.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <numeric>
#include <utility>
#include <vector>

namespace {

/// Helper function which finds module from csv::cell in the geometry and
/// digitization config, and initializes the module limits with the cell's
/// properties
std::size_t fill_module(traccc::edm::cell_module_container::host& modules,
                        const traccc::io::csv::cell& c,
                        const traccc::geometry* geom,
                        const traccc::digitization_config* dconfig) {

    // Add a new module. Remembering its position.
    const std::size_t pos = modules.size();
    modules.resize(pos + 1);

    // Set the module's surface link.
    const detray::geometry::barcode surface_link{c.geometry_id};
    modules.surface_link()[pos] = surface_link;

    // Find/set the 3D position of the detector module.
    if (geom != nullptr) {

        // Check if the module ID is known.
        if (!geom->contains(surface_link.value())) {
            throw std::runtime_error(
                "Could not find placement for geometry ID " +
                std::to_string(surface_link.value()));
        }

        // Set the value on the module description.
        modules.placement()[pos] = (*geom)[surface_link.value()];
    }

    // Find/set the digitization configuration of the detector module.
    if (dconfig != nullptr) {

        // Check if the module ID is known.
        const traccc::digitization_config::Iterator geo_it =
            dconfig->find(surface_link.value());
        if (geo_it == dconfig->end()) {
            throw std::runtime_error(
                "Could not find digitization config for geometry ID " +
                std::to_string(surface_link.value()));
        }

        // Set the value on the module description.
        const auto& binning_data = geo_it->segmentation.binningData();
        assert(binning_data.size() >= 2);
        modules.pixel_data()[pos] = {binning_data[0].min, binning_data[1].min,
                                     binning_data[0].step,
                                     binning_data[1].step};
    }

    // Return the position of the new element.
    return pos;
}

}  // namespace

namespace traccc::io::csv {

void read_cells(traccc::edm::cell_container::host& cells,
                traccc::edm::cell_module_container::host& modules,
                std::string_view filename, const geometry* geom,
                const digitization_config* dconfig) {

    // Construct the cell reader object.
    auto reader = make_cell_reader(filename);

    // Create a cell counter vector.
    std::vector<std::size_t> cellCounts;
    cellCounts.reserve(5000);

    // Reserve a reasonable amount of space for the modules.
    modules.reserve(5000);

    // Create a cell collection, which holds on to a flat list of all the cells
    // and the position of their respective cell counter & module.
    std::vector<std::pair<csv::cell, std::size_t>> allCells;
    allCells.reserve(50000);

    // Read all cells from input file.
    csv::cell iocell;
    while (reader.read(iocell)) {

        // Check whether this cell's module is already known to us.
        const auto& surface_links = modules.surface_link();
        auto rit = std::find(surface_links.rbegin(), surface_links.rend(),
                             detray::geometry::barcode{iocell.geometry_id});
        std::size_t pos = 0;
        if (rit == surface_links.rend()) {
            // Add a new pixel cell module.
            pos = fill_module(modules, iocell, geom, dconfig);
        } else {
            // Find the existing position of this pixel cell module.
            pos = std::distance(surface_links.begin(), rit.base()) - 1;
        }
        allCells.push_back({iocell, pos});
        if (cellCounts.size() <= pos) {
            cellCounts.resize(pos + 1);
            cellCounts[pos] = 0;
        }
        ++(cellCounts[pos]);
    }

    // Transform the cellCounts vector into a prefix sum for accessing
    // positions in the result vector.
    std::partial_sum(cellCounts.begin(), cellCounts.end(), cellCounts.begin());

    // Sort the cells in each module, individually. This is needed for
    // clusterization to work correctly.
    for (std::size_t i = 0; i < cellCounts.size(); ++i) {
        const std::size_t module_start = i == 0 ? 0 : cellCounts[i - 1];
        assert(module_start <= cellCounts[i]);
        const std::size_t module_end = cellCounts[i];
        assert(module_end <= allCells.size());
        assert(module_start <= module_end);
        std::sort(allCells.begin() + module_start,
                  allCells.begin() + module_end,
                  [](const std::pair<csv::cell, std::size_t>& c1,
                     const std::pair<csv::cell, std::size_t>& c2) {
                      return c1.first.channel1 < c2.first.channel1;
                  });
    }

    // The total number cells.
    const unsigned int totalCells = allCells.size();

    // Construct the result collection.
    cells.resize(totalCells);

    // Member "-1" of the prefix sum vector
    std::size_t nCellsZero = 0;
    // Fill the result object with the read csv cells
    for (unsigned int i = 0; i < totalCells; ++i) {

        // The pixel cell to be added to the result container.
        const csv::cell& cell = allCells[i].first;
        // The position of the pixel cell module in the result container.
        const std::size_t module_pos = allCells[i].second;

        // The pixel cell counter for this pixel module.
        std::size_t& cell_pos_ref =
            module_pos == 0 ? nCellsZero : cellCounts[module_pos - 1];
        const std::size_t cell_pos = cell_pos_ref++;

        // Set the properties of the pixel cell.
        cells.channel0()[cell_pos] = cell.channel0;
        cells.channel1()[cell_pos] = cell.channel1;
        cells.activation()[cell_pos] = cell.value;
        cells.time()[cell_pos] = cell.timestamp;
        cells.module_index()[cell_pos] = module_pos;
    }
}

}  // namespace traccc::io::csv
