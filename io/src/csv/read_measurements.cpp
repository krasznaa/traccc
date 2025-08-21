/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_measurements.hpp"

#include "traccc/io/csv/make_measurement_edm.hpp"
#include "traccc/io/csv/make_measurement_reader.hpp"

// System include(s).
#include <numeric>
#include <ranges>

namespace traccc::io::csv {

std::vector<measurement_id_type> read_measurements(
    edm::measurement_collection<default_algebra>::host& measurements,
    std::string_view filename, const traccc::default_detector::host* detector,
    const bool do_sort) {

    // Construct the measurement reader object.
    auto reader = make_measurement_reader(filename);

    // For Acts data, build a map of acts->detray geometry IDs
    std::map<geometry_id, geometry_id> acts_to_detray_id;

    if (detector) {
        for (const auto& surface_desc : detector->surfaces()) {
            acts_to_detray_id[surface_desc.source] =
                surface_desc.barcode().value();
        }
    }

    // Read the measurements from the input file.
    csv::measurement iomeas;
    while (reader.read(iomeas)) {

        // Construct the measurement object.
        measurements.resize(measurements.size() + 1u);
        auto meas = measurements.at(measurements.size() - 1u);
        make_measurement_edm(iomeas, meas, &acts_to_detray_id);
    }

    // Contains the index of the new position at the entry of the old position
    std::vector<measurement_id_type> new_idx_map(measurements.size());
    if (do_sort) {
        // Remeber index locations
        std::vector<unsigned int> idx(measurements.size());
        std::iota(idx.begin(), idx.end(), 0u);

        // Sort the indices the way the measurements will be sorted
        // https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
        std::ranges::sort(idx.begin(), idx.end(),
                          [&measurements](unsigned int i, unsigned int j) {
                              return measurements[i] < measurements[j];
                          });

        // Create a sorted measurement collection.
        vecmem::host_memory_resource host_mr;
        edm::measurement_collection<default_algebra>::host sorted_measurements(
            host_mr);
        sorted_measurements.resize(measurements.size());

        // Map the indices to the new positions. While creating a sorted
        // measurement collection.
        for (std::size_t i = 0u; i < idx.size(); ++i) {
            new_idx_map[idx[i]] = static_cast<measurement_id_type>(i);
            sorted_measurements.at(i) = measurements.at(idx[i]);
        }

        // Override the measurements with the sorted ones.
        measurements = sorted_measurements;
    }

    return new_idx_map;
}

}  // namespace traccc::io::csv
