/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_collection.hpp"
#include "traccc/edm/track_state_collection.hpp"

// VecMem include(s).
#include <vecmem/memory/memory_resource.hpp>

namespace traccc::edm {

/// Types collecting all the collections used to describe (fitted) tracks
template <typename ALGEBRA>
struct track_container {

    struct view {
        /// The tracks
        track_collection<ALGEBRA>::view tracks;
        /// The track states used by the tracks
        track_state_collection<ALGEBRA>::view states;
        /// The measurements used by the tracks
        measurement_collection_types::const_view measurements;
    };

    struct const_view {
        /// The tracks
        track_collection<ALGEBRA>::const_view tracks;
        /// The track states used by the tracks
        track_state_collection<ALGEBRA>::const_view states;
        /// The measurements used by the tracks
        measurement_collection_types::const_view measurements;
    };

    struct host {
        /// Constructor using a memory resource
        explicit host(vecmem::memory_resource& mr) : tracks{mr}, states{mr} {}

        /// The tracks
        track_collection<ALGEBRA>::host tracks;
        /// The track states used by the tracks
        track_state_collection<ALGEBRA>::host states;
    };

    struct buffer {
        /// The tracks
        track_collection<ALGEBRA>::buffer tracks;
        /// The track states used by the tracks
        track_state_collection<ALGEBRA>::buffer states;
    };

    struct data {
        /// The tracks
        track_collection<ALGEBRA>::data tracks;
        /// The track states used by the tracks
        track_state_collection<ALGEBRA>::data states;
    };

    struct const_data {
        /// The tracks
        track_collection<ALGEBRA>::const_data tracks;
        /// The track states used by the tracks
        track_state_collection<ALGEBRA>::const_data states;
    };

    struct device {
        /// Constructor from a view
        TRACCC_HOST_DEVICE
        explicit device(const view& v)
            : tracks{v.tracks},
              states{v.states},
              measurements{v.measurements} {}
        /// The tracks
        track_collection<ALGEBRA>::device tracks;
        /// The track states used by the tracks
        track_state_collection<ALGEBRA>::device states;
        /// The measurements used by the tracks
        measurement_collection_types::const_device measurements;
    };

    struct const_device {
        /// Constructor from a view
        TRACCC_HOST_DEVICE
        explicit const_device(const const_view& v)
            : tracks{v.tracks},
              states{v.states},
              measurements{v.measurements} {}
        /// The tracks
        track_collection<ALGEBRA>::const_device tracks;
        /// The track states used by the tracks
        track_state_collection<ALGEBRA>::const_device states;
        /// The measurements used by the tracks
        measurement_collection_types::const_device measurements;
    };

};  // struct track_container

}  // namespace traccc::edm
