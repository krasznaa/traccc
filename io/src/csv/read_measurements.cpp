/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "read_measurements.hpp"

#include "traccc/io/csv/make_measurement_reader.hpp"

// Detray include(s).
#include "detray/geometry/barcode.hpp"

// System include(s).
#include <algorithm>

namespace traccc::io::csv {

void read_measurements(measurement_reader_output& out,
                       std::string_view filename, const bool do_sort) {

    // Construct the measurement reader object.
    auto reader = make_measurement_reader(filename);

    // Create the result collection.
    measurement_collection_types::host& result_measurements = out.measurements;
    edm::pixel_module_container::host& result_modules = out.modules;

    std::map<geometry_id, unsigned int> m;

    // Read the measurements from the input file.
    csv::measurement iomeas;
    while (reader.read(iomeas)) {

        unsigned int link;
        auto it = m.find(iomeas.geometry_id);

        if (it != m.end()) {
            link = (*it).second;
        } else {
            link = result_modules.size();
            m[iomeas.geometry_id] = link;
            result_modules.resize(link + 1);
            edm::pixel_module_container::surface_link::get(
                result_modules)[link] =
                detray::geometry::barcode{iomeas.geometry_id};
        }

        // Construct the measurement object.
        traccc::measurement meas;
        std::array<typename transform3::size_type, 2u> indices{0u, 0u};
        meas.meas_dim = 0u;

        // Local key is a 8 bit char and first and last bit are dummy value. 2 -
        // 7th bits are for 6 bound track parameters.
        // Ex1) 0000010 or 2 -> meas dim = 1 and [loc0] active -> strip or wire
        // Ex2) 0000110 or 6 -> meas dim = 2 and [loc0, loc1] active -> pixel
        // Ex3) 0000100 or 4 -> meas dim = 1 and [loc1] active -> annulus
        for (unsigned int ipar = 0; ipar < 2u; ++ipar) {
            if (((iomeas.local_key) & (1 << (ipar + 1))) != 0) {

                switch (ipar) {
                    case e_bound_loc0: {
                        meas.local[0] = iomeas.local0;
                        meas.variance[0] = iomeas.var_local0;
                        indices[meas.meas_dim++] = ipar;
                    }; break;
                    case e_bound_loc1: {
                        meas.local[1] = iomeas.local1;
                        meas.variance[1] = iomeas.var_local1;
                        indices[meas.meas_dim++] = ipar;
                    }; break;
                }
            }
        }

        meas.subs.set_indices(indices);
        meas.surface_link = detray::geometry::barcode{iomeas.geometry_id};
        meas.module_link = link;

        result_measurements.push_back(meas);
    }

    if (do_sort) {
        std::sort(result_measurements.begin(), result_measurements.end(),
                  measurement_sort_comp());
    }
}

}  // namespace traccc::io::csv
