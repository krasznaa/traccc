/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_collection.hpp"
#include "traccc/edm/track_container.hpp"
#include "traccc/edm/track_state_collection.hpp"
#include "traccc/edm/track_state_helpers.hpp"
#include "traccc/fitting/status_codes.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

namespace traccc::host::details {

/// Templated implementation of the track fitting algorithm.
///
/// Concrete track fitting algorithms can use this function with the appropriate
/// specializations, to fit tracks on top of a specific detector type, magnetic
/// field type, and track fitting configuration.
///
/// @note The memory resource received by this function is not used thoroughly
///       for the setup of the output container. Inner vectors in the output's
///       jagged vector are created using the default memory resource.
///
/// @tparam fitter_t The fitter type used for the track fitting
///
/// @param[in] fitter           The fitter object to use on the track candidates
/// @param[in] track_container  All track candidates to fit
/// @param[in] mr               Memory resource to use for the output container
///
/// @return A container of the fitted track states
///
template <typename algebra_t, typename fitter_t>
typename edm::track_container<algebra_t>::host kalman_fitting(
    fitter_t& fitter,
    const typename edm::track_container<algebra_t>::const_view& track_container,
    vecmem::memory_resource& mr, vecmem::copy& copy) {

    // Create the input container(s).
    const measurement_collection_types::const_device measurements{
        track_container.measurements};
    const typename edm::track_collection<algebra_t>::const_device
        track_candidates{track_container.tracks};

    // Create the output containers.
    typename edm::track_container<algebra_t>::host result{mr};

    // Iterate over the tracks,
    for (unsigned int i = 0; i < track_candidates.size(); ++i) {

        // Create the objects that will describe this track fit.
        result.tracks.push_back({});
        auto fitted_track = result.tracks.at(result.tracks.size() - 1);
        for (const edm::track_constituent_link& link :
             track_candidates.constituent_links().at(i)) {
            assert(link.type == edm::track_constituent_link::measurement);
            fitted_track.constituent_links().push_back(
                {edm::track_constituent_link::track_state,
                 static_cast<unsigned int>(result.states.size())});
            result.states.push_back(
                edm::make_track_state<algebra_t>(measurements, link.index));
        }

        vecmem::data::vector_buffer<detray::geometry::barcode> seqs_buffer{
            static_cast<vecmem::data::vector_buffer<
                detray::geometry::barcode>::size_type>(
                std::max(fitted_track.constituent_links().size() *
                             fitter.config().barcode_sequence_size_factor,
                         fitter.config().min_barcode_sequence_capacity)),
            mr, vecmem::data::buffer_type::resizable};
        copy.setup(seqs_buffer)->wait();

        // Make a fitter state
        auto result_tracks_view = vecmem::get_data(result.tracks);
        typename edm::track_collection<algebra_t>::device result_tracks_device{
            result_tracks_view};
        typename fitter_t::state fitter_state(
            result_tracks_device.at(result_tracks_device.size() - 1),
            typename edm::track_state_collection<algebra_t>::device{
                vecmem::get_data(result.states)},
            measurements, seqs_buffer);

        // Run the fitter. The status that it returns is not used here. The main
        // failure modes are saved onto the fitted track itself. Not sure what
        // we may want to do with the more detailed status codes in the future.
        (void)fitter.fit(track_candidates.params().at(i), fitter_state);
    }

    // Return the fitted track states.
    return result;
}

}  // namespace traccc::host::details
