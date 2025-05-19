/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/track_fit_collection.hpp"
#include "traccc/edm/track_state_collection.hpp"
#include "traccc/fitting/kalman_filter/gain_matrix_updater.hpp"
#include "traccc/fitting/kalman_filter/two_filters_smoother.hpp"
#include "traccc/fitting/status_codes.hpp"
#include "traccc/utils/particle.hpp"

// detray include(s).
#include <detray/propagator/base_actor.hpp>

// vecmem include(s)
#include <vecmem/containers/device_vector.hpp>

namespace traccc {

/// Detray actor for Kalman filtering
template <typename algebra_t>
struct kalman_actor : detray::actor {

    // Actor state
    struct state {

        /// Constructor with the vector of track states
        TRACCC_HOST_DEVICE
        state(const typename edm::track_fit_collection<
                  algebra_t>::device::proxy_type& track,
              const typename edm::track_state_collection<algebra_t>::device&
                  track_states,
              const measurement_collection_types::const_device& measurements)
            : m_track{track},
              m_track_states{track_states},
              m_measurements{measurements} {

            reset();
        }

        /// @return the reference of track state pointed by the iterator
        TRACCC_HOST_DEVICE
        typename edm::track_state_collection<algebra_t>::device::proxy_type
        operator()() {
            if (!backward_mode) {
                return m_track_states.at(*m_it);
            } else {
                return m_track_states.at(*m_it_rev);
            }
        }

        /// Reset the iterator
        TRACCC_HOST_DEVICE
        void reset() {
            m_it = m_track.state_indices().begin();
            m_it_rev = m_track.state_indices().rbegin();
        }

        /// Advance the iterator
        TRACCC_HOST_DEVICE
        void next() {
            if (!backward_mode) {
                m_it++;
            } else {
                m_it_rev++;
            }
        }

        /// @return true if the iterator reaches the end of vector
        TRACCC_HOST_DEVICE
        bool is_complete() {
            if (!backward_mode && m_it == m_track.state_indices().end()) {
                return true;
            } else if (backward_mode &&
                       m_it_rev == m_track.state_indices().rend()) {
                return true;
            }
            return false;
        }

        /// Object describing the track fit
        typename edm::track_fit_collection<algebra_t>::device::proxy_type
            m_track;
        /// All track states in the event
        typename edm::track_state_collection<algebra_t>::device m_track_states;
        /// All measurements in the event
        measurement_collection_types::const_device m_measurements;

        /// Iterator for forward filtering over the track states
        vecmem::device_vector<unsigned int>::iterator m_it;

        /// Iterator for backward filtering over the track states
        vecmem::device_vector<unsigned int>::reverse_iterator m_it_rev;

        // The number of holes (The number of sensitive surfaces which do not
        // have a measurement for the track pattern)
        unsigned int n_holes{0u};

        // Run back filtering for smoothing, if true
        bool backward_mode = false;
    };

    /// Actor operation to perform the Kalman filtering
    ///
    /// @param actor_state the actor state
    /// @param propagation the propagator state
    template <typename propagator_state_t>
    TRACCC_HOST_DEVICE void operator()(state& actor_state,
                                       propagator_state_t& propagation) const {

        auto& stepping = propagation._stepping;
        auto& navigation = propagation._navigation;

        // If the iterator reaches the end, terminate the propagation
        if (actor_state.is_complete()) {
            propagation._heartbeat &= navigation.exit();
            return;
        }

        // triggered only for sensitive surfaces
        if (navigation.is_on_sensitive()) {

            typename edm::track_state_collection<algebra_t>::device::proxy_type
                trk_state = actor_state();

            // Increase the hole counts if the propagator fails to find the next
            // measurement
            if (navigation.barcode() !=
                actor_state.m_measurements.at(trk_state.measurement_index())
                    .surface_link) {
                if (!actor_state.backward_mode) {
                    actor_state.n_holes++;
                }
                return;
            }

            // This track state is not a hole
            if (!actor_state.backward_mode) {
                trk_state.set_hole(false);
            }

            // Run Kalman Gain Updater
            const auto sf = navigation.get_surface();

            kalman_fitter_status res = kalman_fitter_status::SUCCESS;

            if (!actor_state.backward_mode) {
                // Forward filter
                res = sf.template visit_mask<gain_matrix_updater<algebra_t>>(
                    trk_state, actor_state.m_measurements,
                    propagation._stepping.bound_params());

                // Update the propagation flow
                stepping.bound_params() = trk_state.filtered_params();

            } else {
                // Backward filter for smoothing
                res = sf.template visit_mask<two_filters_smoother<algebra_t>>(
                    trk_state, actor_state.m_measurements,
                    propagation._stepping.bound_params());
            }

            // Abort if the Kalman update fails
            if (res != kalman_fitter_status::SUCCESS) {
                propagation._heartbeat &=
                    navigation.abort(fitter_debug_msg{res});
                return;
            }

            // Change the charge of hypothesized particles when the sign of qop
            // is changed (This rarely happens when qop is set with a poor seed
            // resolution)
            propagation.set_particle(detail::correct_particle_hypothesis(
                stepping.particle_hypothesis(),
                propagation._stepping.bound_params()));

            // Update iterator
            actor_state.next();

            // Flag renavigation of the current candidate
            navigation.set_high_trust();
        }
    }
};

}  // namespace traccc
