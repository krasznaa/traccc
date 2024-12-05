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
#include "traccc/utils/subspace.hpp"

// Detray include(s).
#include "detray/geometry/barcode.hpp"

// VecMem include(s).
#include <vecmem/edm/container.hpp>

namespace traccc::edm {

/// Interface for the @c traccc::edm::measurement_collection class.
template <typename BASE>
class measurement : public BASE {

    public:
    /// @name Constructors
    /// @{

    /// Inherit the base class's constructor(s)
    using BASE::BASE;
    /// Use a default copy constructor
    measurement(const measurement& other) = default;
    /// Use a default move constructor
    measurement(measurement&& other) = default;

    /// @}

    /// @name Measurement Information
    /// @{

    /// Local 2D coordinates for a measurement on a detector module (non-const)
    ///
    /// @return A (non-const) (vector) of @c traccc::point2 value(s)
    ///
    TRACCC_HOST_DEVICE auto& local() { return BASE::template get<0>(); }
    /// Local 2D coordinates for a measurement on a detector module (const)
    ///
    /// @return A (const) (vector) of @c traccc::point2 value(s)
    ///
    TRACCC_HOST_DEVICE const auto& local() const {
        return BASE::template get<0>();
    }

    /// Variance on the 2D coordinates of the measurement (non-const)
    ///
    /// @return A (non-const) (vector) of @c traccc::variance2 value(s)
    ///
    TRACCC_HOST_DEVICE auto& variance() { return BASE::template get<1>(); }
    /// Variance on the 2D coordinates of the measurement (const)
    ///
    /// @return A (const) (vector) of @c traccc::variance2 value(s)
    ///
    TRACCC_HOST_DEVICE const auto& variance() const {
        return BASE::template get<1>();
    }

    /// Geometry ID (non-const)
    ///
    /// @return A (non-const) (vector) of @c detray::geometry::barcode value(s)
    ///
    TRACCC_HOST_DEVICE auto& geometry_id() { return BASE::template get<2>(); }
    /// Geometry ID (const)
    ///
    /// @return A (const) (vector) of @c detray::geometry::barcode value(s)
    ///
    TRACCC_HOST_DEVICE const auto& geometry_id() const {
        return BASE::template get<2>();
    }

    /// Unique measurement ID (non-const)
    ///
    /// @return A (non-const) (vector) of @c uint32_t value(s)
    ///
    TRACCC_HOST_DEVICE auto& measurement_id() {
        return BASE::template get<3>();
    }
    /// Unique measurement ID (const)
    ///
    /// @return A (const) (vector) of @c uint32_t value(s)
    ///
    TRACCC_HOST_DEVICE const auto& measurement_id() const {
        return BASE::template get<3>();
    }

    /// Link to the cluster that the measurement was made from (non-const)
    ///
    /// @return A (non-const) (vector) of @c uint32_t value(s)
    ///
    TRACCC_HOST_DEVICE auto& cluster_link() { return BASE::template get<4>(); }
    /// Link to the cluster that the measurement was made from (const)
    ///
    /// @return A (const) (vector) of @c uint32_t value(s)
    ///
    TRACCC_HOST_DEVICE const auto& cluster_link() const {
        return BASE::template get<4>();
    }

    /// Measurement dimensions (non-const)
    ///
    /// @return A (non-const) (vector) of @c uint8_t value(s)
    ///
    TRACCC_HOST_DEVICE auto& dimensions() { return BASE::template get<5>(); }
    /// Measurement dimensions (const)
    ///
    /// @return A (const) (vector) of @c uint8_t value(s)
    ///
    TRACCC_HOST_DEVICE const auto& dimensions() const {
        return BASE::template get<5>();
    }

    /// Measurement subspace (non-const)
    ///
    /// @return A (non-const) (vector) of
    ///         @c traccc::subspace<default_algebra,e_bound_size,2u> value(s)
    ///
    TRACCC_HOST_DEVICE auto& subs() { return BASE::template get<6>(); }
    /// Measurement subspace (const)
    ///
    /// @return A (const) (vector) of
    ///         @c traccc::subspace<default_algebra,e_bound_size,2u> value(s)
    ///
    TRACCC_HOST_DEVICE const auto& subs() const {
        return BASE::template get<6>();
    }

    /// @}

};  // class measurement

/// SoA container describing measurements
using measurement_collection = vecmem::edm::container<
    measurement, vecmem::edm::type::vector<point2>,
    vecmem::edm::type::vector<variance2>,
    vecmem::edm::type::vector<detray::geometry::barcode>,
    vecmem::edm::type::vector<unsigned int>,
    vecmem::edm::type::vector<unsigned int>,
    vecmem::edm::type::vector<unsigned char>,
    vecmem::edm::type::vector<subspace<default_algebra, e_bound_size, 2u>>>;

}  // namespace traccc::edm
