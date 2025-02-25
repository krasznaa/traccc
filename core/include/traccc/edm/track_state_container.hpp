/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_quality.hpp"

// Detray include(s).
#include <detray/definitions/track_parametrization.hpp>
#include <detray/geometry/barcode.hpp>
#include <detray/tracks/bound_track_parameters.hpp>

// VecMem include(s).
#include <vecmem/edm/container.hpp>

// System include(s).
#include <cstdint>

namespace traccc::edm {

/// Fitting outcome for one track
enum class track_state_fitting_result : std::uint32_t {
    SUCCESS = 0,
    UNKNOWN,
    FAILURE_NON_POSITIVE_NDF,
    FAILURE_NOT_ALL_SMOOTHED,
    MAX_OUTCOME
};

/// Interface for the @c traccc::edm::track_state_collection class.
///
/// It provides the API that users would interact with, while using the
/// columns/arrays of the SoA containers, or the variables of the AoS proxies
/// created on top of the SoA containers.
///
template <typename BASE>
class track_state : public BASE {

    public:
    /// @name Functions inherited from the base class
    /// @{

    /// Inherit the base class's constructor(s)
    using BASE::BASE;
    /// Inherit the base class's assignment operator(s).
    using BASE::operator=;

    /// @}

    /// @name Track State Information
    /// @{

    /// The outcome of the track fit (non-const)
    ///
    /// @return A (non-const) vector of @c track_state_fitting_result values
    ///
    TRACCC_HOST_DEVICE
    auto& fit_outcome() { return BASE::template get<0>(); }
    /// The outcome of the track fit (const)
    ///
    /// @return A (const) vector of @c track_state_fitting_result values
    ///
    TRACCC_HOST_DEVICE
    const auto& fit_outcome() const { return BASE::template get<0>(); }

    /// The fitted track parameters at the track's origin (non-const)
    ///
    /// @return A (non-const) vector of bound track parameters
    ///
    TRACCC_HOST_DEVICE
    auto& fit_params() { return BASE::template get<1>(); }
    /// The fitted track parameters at the track's origin (const)
    ///
    /// @return A (const) vector of bound track parameters
    ///
    TRACCC_HOST_DEVICE
    const auto& fit_params() const { return BASE::template get<1>(); }

    /// The quality of the track fit (non-const)
    ///
    /// @return A (non-const) vector of @c track_quality values
    ///
    TRACCC_HOST_DEVICE
    auto& trk_quality() { return BASE::template get<2>(); }
    /// The quality of the track fit (const)
    ///
    /// @return A (const) vector of @c track_quality values
    ///
    TRACCC_HOST_DEVICE
    const auto& trk_quality() const { return BASE::template get<2>(); }

    /// Whether the track has a hole on the surface (non-const)
    ///
    /// @return A (non-const) jagged vector of @c std::uint8_t values
    ///
    TRACCC_HOST_DEVICE
    auto& is_hole() { return BASE::template get<3>(); }
    /// Whether the track has a hole on the surface (const)
    ///
    /// @return A (const) jagged vector of @c std::uint8_t values
    ///
    TRACCC_HOST_DEVICE
    const auto& is_hole() const { return BASE::template get<3>(); }

    /// Whether the track is smoothed on the surface (non-const)
    ///
    /// @return A (non-const) jagged vector of @c std::uint8_t values
    ///
    TRACCC_HOST_DEVICE
    auto& is_smoothed() { return BASE::template get<4>(); }
    /// Whether the track is smoothed on the surface (const)
    ///
    /// @return A (const) jagged vector of @c std::uint8_t values
    ///
    TRACCC_HOST_DEVICE
    const auto& is_smoothed() const { return BASE::template get<4>(); }

    /// The identifier of the surface that the fit is tied to (non-const)
    ///
    /// @return A (non-const) jagged vector of
    ///         @c detray::geometry::barcode values
    ///
    TRACCC_HOST_DEVICE
    auto& surface_link() { return BASE::template get<5>(); }
    /// The identifier of the surface that the fit is tied to (const)
    ///
    /// @return A (const) jagged vector of @c detray::geometry::barcode values
    ///
    TRACCC_HOST_DEVICE
    const auto& surface_link() const { return BASE::template get<5>(); }

    /// The index of the track's measurement on the surface (non-const)
    ///
    /// @return A (non-const) vector of <tt>unsigned int</tt> values
    ///
    TRACCC_HOST_DEVICE
    auto& measurement_index() { return BASE::template get<6>(); }
    /// The index of the track's measurement on the surface (const)
    ///
    /// @return A (const) vector of <tt>unsigned int</tt> values
    ///
    TRACCC_HOST_DEVICE
    const auto& measurement_index() const { return BASE::template get<6>(); }

    /// The transport jacobian of the track fit on the surface (non-const)
    ///
    /// @return A (non-const) jagged vector of bound matrix values
    ///
    TRACCC_HOST_DEVICE
    auto& jacobian() { return BASE::template get<7>(); }
    /// The transport jacobian of the track fit on the surface (const)
    ///
    /// @return A (const) jagged vector of bound matrix values
    ///
    TRACCC_HOST_DEVICE
    const auto& jacobian() const { return BASE::template get<7>(); }

    /// The predicted track state on the surface (non-const)
    ///
    /// @return A (non-const) jagged vector of bound track parameters
    ///
    TRACCC_HOST_DEVICE
    auto& predicted() { return BASE::template get<8>(); }
    /// The predicted track state on the surface (const)
    ///
    /// @return A (const) jagged vector of bound track parameters
    ///
    TRACCC_HOST_DEVICE
    const auto& predicted() const { return BASE::template get<8>(); }

    /// The chi square of the filtered track parameters on the surface
    /// (non-const)
    ///
    /// @return A (non-const) jagged vector of scalar values
    ///
    TRACCC_HOST_DEVICE
    auto& filtered_chi2() { return BASE::template get<9>(); }
    /// The chi square of the filtered track parameters on the surface (const)
    ///
    /// @return A (const) jagged vector of scalar values
    ///
    TRACCC_HOST_DEVICE
    const auto& filtered_chi2() const { return BASE::template get<9>(); }

    /// The filtered track parameters on the surface (non-const)
    ///
    /// @return A (non-const) jagged vector of bound track parameters
    ///
    TRACCC_HOST_DEVICE
    auto& filtered() { return BASE::template get<10>(); }
    /// The filtered track parameters on the surface (const)
    ///
    /// @return A (const) jagged vector of bound track parameters
    ///
    TRACCC_HOST_DEVICE
    const auto& filtered() const { return BASE::template get<10>(); }

    /// The chi square of the smoothed track parameters on the surface
    /// (non-const)
    ///
    /// @return A (non-const) jagged vector of scalar values
    ///
    TRACCC_HOST_DEVICE
    auto& smoothed_chi2() { return BASE::template get<11>(); }
    /// The chi square of the smoothed track parameters on the surface (const)
    ///
    /// @return A (const) jagged vector of scalar values
    ///
    TRACCC_HOST_DEVICE
    const auto& smoothed_chi2() const { return BASE::template get<11>(); }

    /// The smoothed track parameters on the surface (non-const)
    ///
    /// @return A (non-const) jagged vector of bound track parameters
    ///
    TRACCC_HOST_DEVICE
    auto& smoothed() { return BASE::template get<12>(); }
    /// The smoothed track parameters on the surface (const)
    ///
    /// @return A (const) jagged vector of bound track parameters
    ///
    TRACCC_HOST_DEVICE
    const auto& smoothed() const { return BASE::template get<12>(); }

    /// The chi square of backward filter on the surface (non-const)
    ///
    /// @return A (non-const) jagged vector of scalar values
    ///
    TRACCC_HOST_DEVICE
    auto& backward_chi2() { return BASE::template get<13>(); }
    /// The chi square of backward filter on the surface (const)
    ///
    /// @return A (const) jagged vector of scalar values
    ///
    TRACCC_HOST_DEVICE
    const auto& backward_chi2() const { return BASE::template get<13>(); }

    /// @}

};  // class track_state

/// SoA container describing fitted tracks
///
/// @tparam ALGEBRA The algebra type used to describe the tracks
///
template <typename ALGEBRA>
using track_state_container = vecmem::edm::container<
    traccc::edm::track_state,
    vecmem::edm::type::vector<traccc::edm::track_state_fitting_result>,
    vecmem::edm::type::vector<detray::bound_track_parameters<ALGEBRA>>,
    vecmem::edm::type::vector<traccc::track_quality>,
    vecmem::edm::type::jagged_vector<std::uint8_t>,
    vecmem::edm::type::jagged_vector<std::uint8_t>,
    vecmem::edm::type::jagged_vector<detray::geometry::barcode>,
    vecmem::edm::type::jagged_vector<unsigned int>,
    vecmem::edm::type::jagged_vector<detray::bound_matrix<ALGEBRA>>,
    vecmem::edm::type::jagged_vector<detray::bound_track_parameters<ALGEBRA>>,
    vecmem::edm::type::jagged_vector<detray::dscalar<ALGEBRA>>,
    vecmem::edm::type::jagged_vector<detray::bound_track_parameters<ALGEBRA>>,
    vecmem::edm::type::jagged_vector<detray::dscalar<ALGEBRA>>,
    vecmem::edm::type::jagged_vector<detray::bound_track_parameters<ALGEBRA>>,
    vecmem::edm::type::jagged_vector<detray::dscalar<ALGEBRA>>>;

}  // namespace traccc::edm
