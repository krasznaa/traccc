/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// SYCL library include(s).
#include "traccc/sycl/utils/queue_wrapper.hpp"

// Project include(s).
#include "traccc/bfield/magnetic_field.hpp"
#include "traccc/edm/track_candidate_container.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/fitting_config.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/messaging.hpp"

// VecMem include(s).
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <functional>

namespace traccc::sycl {

/// Kalman filter based track fitting algorithm
class kalman_fitting_algorithm
    : public algorithm<track_state_container_types::buffer(
          const default_detector::view&, const magnetic_field&,
          const edm::track_candidate_container<default_algebra>::const_view&)>,
      public algorithm<track_state_container_types::buffer(
          const telescope_detector::view&, const magnetic_field&,
          const edm::track_candidate_container<default_algebra>::const_view&)>,
      public messaging {

    public:
    /// Configuration type
    using config_type = fitting_config;
    /// Output type
    using output_type = track_state_container_types::buffer;

    /// Constructor with the algorithm's configuration
    ///
    /// @param config The configuration object
    ///
    kalman_fitting_algorithm(
        const config_type& config, const traccc::memory_resource& mr,
        vecmem::copy& copy, queue_wrapper queue,
        std::unique_ptr<const Logger> logger = getDummyLogger().clone());

    /// Execute the algorithm
    ///
    /// @param det             The (default) detector object
    /// @param bfield          The magnetic field object
    /// @param track_candidates All track candidates to fit
    ///
    /// @return A container of the fitted track states
    ///
    output_type operator()(
        const default_detector::view& det, const magnetic_field& bfield,
        const edm::track_candidate_container<default_algebra>::const_view&
            track_candidates) const override;

    /// Execute the algorithm
    ///
    /// @param det             The (telescope) detector object
    /// @param bfield          The magnetic field object
    /// @param track_candidates All track candidates to fit
    ///
    /// @return A container of the fitted track states
    ///
    output_type operator()(
        const telescope_detector::view& det, const magnetic_field& bfield,
        const edm::track_candidate_container<default_algebra>::const_view&
            track_candidates) const override;

    private:
    /// Algorithm configuration
    config_type m_config;
    /// Memory resource used by the algorithm
    traccc::memory_resource m_mr;
    /// Copy object used by the algorithm
    std::reference_wrapper<vecmem::copy> m_copy;
    /// Queue wrapper
    mutable queue_wrapper m_queue;

};  // class kalman_fitting_algorithm

}  // namespace traccc::sycl
