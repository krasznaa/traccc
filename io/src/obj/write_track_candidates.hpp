/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_candidate_container.hpp"
#include "traccc/geometry/detector.hpp"

// System include(s).
#include <string_view>

namespace traccc::io::obj {

/// Write a track candidate container into a Wavefront OBJ file.
///
/// @param filename is the name of the output file
/// @param tracks is the track candidate container to write
/// @param measurements is the collection of all measurements in the event
/// @param detector is the Detray detector describing the geometry
///
void write_track_candidates(
    std::string_view filename,
    edm::track_candidate_container<default_algebra>::const_view tracks,
    const traccc::default_detector::host& detector);

}  // namespace traccc::io::obj
