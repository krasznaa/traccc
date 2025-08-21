/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/edm/measurement_collection.hpp"

// Local include(s).
#include "traccc/io/csv/measurement.hpp"

#pragma once

namespace traccc::io::csv {

/// Make measurement EDM from csv measurement
///
/// @param[in] csv_meas input csv measurement
/// @param[out] meas output measurement proxy to fill
/// @param[in] acts_to_detray_id Map for acts-to-detray geometry ID converision
///
void make_measurement_edm(
    const traccc::io::csv::measurement& csv_meas,
    edm::measurement_collection<default_algebra>::host::proxy_type& meas,
    const std::map<geometry_id, geometry_id>* acts_to_detray_id);

}  // namespace traccc::io::csv
