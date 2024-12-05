/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/track_parameters.hpp"

// VecMem include(s).
#include <vecmem/edm/container.hpp>

namespace traccc::edm {

/// Interface for the @c traccc::edm::track_candidate_collection class.
template <typename BASE>
class track_candidate : public BASE {

    public:
    /// @name Constructors
    /// @{

    /// Inherit the base class's constructor(s)
    using BASE::BASE;
    /// Use a default copy constructor
    track_candidate(const track_candidate& other) = default;
    /// Use a default move constructor
    track_candidate(track_candidate&& other) = default;

    /// @}

    /// @name Track Candidate Information
    /// @{

    /// (Bound) Track parameters for a track candidate (non-const)
    ///
    /// @return A (non-const) (vector) of @c traccc::bound_track_parameters
    ///         value(s)
    ///
    TRACCC_HOST_DEVICE auto& parameters() { return BASE::template get<0>(); }
    /// (Bound) Track parameters for a track candidate (const)
    ///
    /// @return A (const) (vector) of @c traccc::bound_track_parameters value(s)
    ///
    TRACCC_HOST_DEVICE const auto& parameters() const {
        return BASE::template get<0>();
    }

    /// Links to the track's measurements (non-const)
    ///
    /// @return A (non-const) (jagged vector) of @c uint32_t value(s)
    ///
    TRACCC_HOST_DEVICE auto& measurements() { return BASE::template get<1>(); }
    /// Links to the track's measurements (const)
    ///
    /// @return A (const) (jagged vector) of @c uint32_t value(s)
    ///
    TRACCC_HOST_DEVICE const auto& measurements() const {
        return BASE::template get<1>();
    }

    /// @}

};  // class track_candidate

/// SoA container describing track candidates
using track_candidate_collection =
    vecmem::edm::container<track_candidate,
                           vecmem::edm::type::vector<bound_track_parameters>,
                           vecmem::edm::type::jagged_vector<unsigned int>>;

}  // namespace traccc::edm
