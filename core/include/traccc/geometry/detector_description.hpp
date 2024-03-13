/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/qualifiers.hpp"

// VecMem include(s).
#include <vecmem/edm/container.hpp>

namespace traccc {

/// Interface for the @c traccc::detector_description class.
///
/// It provides the API that users would interact with, while using the
/// columns/arrays defined in @c traccc::detector_description.
///
template <typename BASE>
class detector_description_interface : public BASE {

    public:
    /// Inherit the base class's constructor(s)
    using BASE::BASE;

    /// @name Detector geometry information
    /// @{

    /// The identifier of the detector module's surface (non-const)
    ///
    /// Can be used to look up the module in a @c detray::detector object.
    ///
    /// @return A (non-const) vector of @c detray::geometry::barcode objects
    ///
    TRACCC_HOST_DEVICE
    auto& surface_link() { return BASE::template get<0>(); }
    /// The identifier of the detector module's surface (const)
    ///
    /// Can be used to look up the module in a @c detray::detector object.
    ///
    /// @return A (const) vector of @c detray::geometry::barcode objects
    ///
    TRACCC_HOST_DEVICE
    const auto& surface_link() const { return BASE::template get<0>(); }

    /// The placement of the detector module "in the world frame" (non-const)
    ///
    /// To be used for local-to-global transformations during spacepoint
    /// creation.
    ///
    /// @return A (non-const) vector of @c traccc::transform3 objects
    ///
    TRACCC_HOST_DEVICE
    auto& placement() { return BASE::template get<1>(); }
    /// The placement of the detector module "in the world frame" (const)
    ///
    /// To be used for local-to-global transformations during spacepoint
    /// creation.
    ///
    /// @return A (const) vector of @c traccc::transform3 objects
    ///
    TRACCC_HOST_DEVICE
    const auto& placement() const { return BASE::template get<1>(); }

    /// @}

    /// @name Detector module information
    /// @{

    /// Signal threshold for detection elements (non-const)
    ///
    /// It controls which elements (pixels and strips) are considered during
    /// clusterization.
    ///
    /// @return A (non-const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    auto& threshold() { return BASE::template get<2>(); }
    /// Signal threshold for detection elements (const)
    ///
    /// It controls which elements (pixels and strips) are considered during
    /// clusterization.
    ///
    /// @return A (const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    const auto& threshold() const { return BASE::template get<2>(); }

    /// Reference for local position calculation in X direction (non-const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the X axis with the formula:
    /// @f$pos_x = reference_x + pitch_x * index_x@f$
    ///
    /// @return A (non-const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    auto& reference_x() { return BASE::template get<3>(); }
    /// Reference for local position calculation in X direction (const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the X axis with the formula:
    /// @f$pos_x = reference_x + pitch_x * index_x@f$
    ///
    /// @return A (const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    const auto& reference_x() const { return BASE::template get<3>(); }

    /// Reference for local position calculation in Y direction (non-const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the Y axis with the formula:
    /// @f$pos_y = reference_y + pitch_y * index_y@f$
    ///
    /// @return A (non-const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    auto& reference_y() { return BASE::template get<4>(); }
    /// Reference for local position calculation in Y direction (const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the Y axis with the formula:
    /// @f$pos_y = reference_y + pitch_y * index_y@f$
    ///
    /// @return A (const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    const auto& reference_y() const { return BASE::template get<4>(); }

    /// Pitch for local position calculation in X direction (non-const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the X axis with the formula:
    /// @f$pos_x = reference_x + pitch_x * index_x@f$
    ///
    /// @return A (non-const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    auto& pitch_x() { return BASE::template get<5>(); }
    /// Pitch for local position calculation in X direction (const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the X axis with the formula:
    /// @f$pos_x = reference_x + pitch_x * index_x@f$
    ///
    /// @return A (const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    const auto& pitch_x() const { return BASE::template get<5>(); }

    /// Pitch for local position calculation in Y direction (non-const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the Y axis with the formula:
    /// @f$pos_y = reference_y + pitch_y * index_y@f$
    ///
    /// @return A (non-const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    auto& pitch_y() { return BASE::template get<6>(); }
    /// Pitch for local position calculation in Y direction (const)
    ///
    /// The position of a detector element (pixel or strip) is calculated
    /// along the Y axis with the formula:
    /// @f$pos_y = reference_y + pitch_y * index_y@f$
    ///
    /// @return A (const) vector of @c traccc::scalar objects
    ///
    TRACCC_HOST_DEVICE
    const auto& pitch_y() const { return BASE::template get<6>(); }

    /// @}

};  // class detector_description_interface

/// SoA container describing the detector
using detector_description = vecmem::edm::container<
    detector_description_interface,
    vecmem::edm::type::vector<detray::geometry::barcode>,
    vecmem::edm::type::vector<transform3>, vecmem::edm::type::vector<scalar>,
    vecmem::edm::type::vector<scalar>, vecmem::edm::type::vector<scalar>,
    vecmem::edm::type::vector<scalar>, vecmem::edm::type::vector<scalar>>;

}  // namespace traccc
