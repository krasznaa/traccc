/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::device {

TRACCC_DEVICE inline void prune_tracks(const global_index_t globalIndex,
                                       const prune_tracks_payload& payload) {

    const edm::track_candidate_collection<default_algebra>::const_device
        track_candidates(payload.track_candidates_view);
    const vecmem::device_vector<const unsigned int> valid_indices(
        payload.valid_indices_view);
    edm::track_candidate_collection<default_algebra>::device prune_candidates(
        payload.prune_candidates_view);

    assert(valid_indices.size() >= prune_candidates.size());

    if (globalIndex >= prune_candidates.size()) {
        return;
    }

    const auto idx = valid_indices.at(globalIndex);

    // Copy candidates
    prune_candidates.at(globalIndex) = track_candidates.at(idx);
}

}  // namespace traccc::device
