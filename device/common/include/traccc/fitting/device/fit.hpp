/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_candidate_collection.hpp"
#include "traccc/edm/track_state.hpp"

namespace traccc::device {

/// Function used for fitting a track for a given track candidates
///
/// @param[in] globalIndex   The index of the current thread
/// @param[in] det_data      Detector view object
/// @param[in] track_candidates_view The input track candidates
/// @param[in] measurements_view All measurements in the event
/// @param[in] param_ids_view The input parameter ids
/// @param[out] track_states_view The output of fitted track states
///
template <typename fitter_t>
TRACCC_HOST_DEVICE inline void fit(
    global_index_t globalIndex,
    typename fitter_t::detector_type::view_type det_data,
    const typename fitter_t::bfield_type field_data,
    const typename fitter_t::config_type cfg,
    const edm::track_candidate_collection<default_algebra>::const_view
        track_candidates_view,
    const measurement_collection_types::const_view& measurements_view,
    const vecmem::data::vector_view<const unsigned int>& param_ids_view,
    track_state_container_types::view track_states_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/fitting/device/impl/fit.ipp"
