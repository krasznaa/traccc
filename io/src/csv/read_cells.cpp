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
#include <utility>
#include <vector>

namespace {

/// Comparator used for sorting cells. This sorting is one of the assumptions
/// made in the clusterization algorithm
const auto comp = [](const traccc::cell& c1, const traccc::cell& c2) {
    return c1.channel1 < c2.channel1;
};

/// Helper function which finds module from csv::cell in the geometry and
/// digitization config, and initializes the module limits with the cell's
/// properties
std::size_t fill_module(edm::pixel_module_container::host& modules,
                        const traccc::io::csv::cell& c,
                        const traccc::geometry* geom,
                        const traccc::digitization_config* dconfig) {

    // Add a new module. Remembering its position.
    const std::size_t pos = modules.size();
    modules.resize(pos + 1);

    // Set the module's surface link.
    const detray::geometry::barcode surface_link{c.geometry_id};
    edm::pixel_module_container::surface_link::get(modules)[pos] = surface_link;

    // Find/set the 3D position of the detector module.
    if (geom != nullptr) {

        // Check if the module ID is known.
        if (!geom->contains(surface_link.value())) {
            throw std::runtime_error(
                "Could not find placement for geometry ID " +
                std::to_string(result.surface_link.value()));
        }

        // Set the value on the module description.
        edm::pixel_module_container::placement::get(modules)[pos] =
            (*geom)[result.surface_link.value()];
    }

    // Find/set the digitization configuration of the detector module.
    if (dconfig != nullptr) {

        // Check if the module ID is known.
        const traccc::digitization_config::Iterator geo_it =
            dconfig->find(surface_link.value());
        if (geo_it == dconfig->end()) {
            throw std::runtime_error(
                "Could not find digitization config for geometry ID " +
                std::to_string(result.surface_link.value()));
        }

        // Set the value on the module description.
        const auto& binning_data = geo_it->segmentation.binningData();
        assert(binning_data.size() >= 2);
        edm::pixel_module_container::pixel_data::get(modules)[pos] =
            {binning_data[0].min, binning_data[1].min,
             binning_data[0].step, binning_data[1].step};
    }

    return result;
}

}  // namespace

namespace traccc::io::csv {

void read_cells(edm::pixel_cell_container::host &cells,
                edm::pixel_module_container::host &modules,
                std::string_view filename,
                const geometry* geom, const digitization_config* dconfig) {

    // Construct the cell reader object.
    auto reader = make_cell_reader(filename);

    // Create a cell counter vector.
    std::vector<unsigned int> cellCounts;
    cellCounts.reserve(5000);

    // Reserve a reasonable amount of space for the modules.
    modules.reserve(5000);

    // Create a cell collection, which holds on to a flat list of all the cells
    // and the position of their respective cell counter & module.
    std::vector<std::pair<csv::cell, unsigned int>> allCells;
    allCells.reserve(50000);

    // Read all cells from input file.
    csv::cell iocell;
    while (reader.read(iocell)) {

        // Check whether this cell's module is already known to us.
        auto& surface_links =
            edm::pixel_module_container::surface_link::get(modules);
        auto rit = std::find(surface_links.rbegin(), surface_links.rend(),
                             iocell.geometry_id);
        if (rit == surface_links.rend()) {
            // Add new cell and new cell counter if a new module is found
            const cell_module mod = get_module(iocell, geom, dconfig);
            allCells.push_back({iocell, result_modules.size()});
            result_modules.push_back(mod);
            cellCounts.push_back(1);
        } else {
            // Add a new cell and update cell counter if repeat module is found
            const unsigned int pos =
                std::distance(result_modules.begin(), rit.base()) - 1;
            allCells.push_back({iocell, pos});
            ++(cellCounts[pos]);
        }
    }

    // Transform the cellCounts vector into a prefix sum for accessing
    // positions in the result vector.
    std::partial_sum(cellCounts.begin(), cellCounts.end(), cellCounts.begin());

    // The total number cells.
    const unsigned int totalCells = allCells.size();

    // Construct the result collection.
    cell_collection_types::host& result_cells = out.cells;
    result_cells.resize(totalCells);

    // Member "-1" of the prefix sum vector
    unsigned int nCellsZero = 0;
    // Fill the result object with the read csv cells
    for (unsigned int i = 0; i < totalCells; ++i) {
        const csv::cell& c = allCells[i].first;

        // The position of the cell counter this cell belongs to
        const unsigned int& counterPos = allCells[i].second;

        unsigned int& prefix_sum_previous =
            counterPos == 0 ? nCellsZero : cellCounts[counterPos - 1];
        result_cells[prefix_sum_previous++] = traccc::cell{
            c.channel0, c.channel1, c.value, c.timestamp, counterPos};
    }

    if (cellCounts.size() == 0) {
        return;
    }
    /* This is might look a bit overcomplicated, and could be made simpler by
     * having a copy of the prefix sum vector before incrementing its value when
     * filling the vector. however this seems more efficient, but requires
     * manually setting the 1st & 2nd modules instead of just the 1st.
     */

    // Sort the cells belonging to the first module.
    std::sort(result_cells.begin(), result_cells.begin() + nCellsZero, comp);
    // Sort the cells belonging to the second module.
    std::sort(result_cells.begin() + nCellsZero,
              result_cells.begin() + cellCounts[0], comp);

    // Sort cells belonging to all other modules.
    for (unsigned int i = 1; i < cellCounts.size() - 1; ++i) {
        std::sort(result_cells.begin() + cellCounts[i - 1],
                  result_cells.begin() + cellCounts[i], comp);
    }
}

}  // namespace traccc::io::csv
