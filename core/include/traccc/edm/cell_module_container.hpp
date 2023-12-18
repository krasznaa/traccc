/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/geometry/pixel_data.hpp"

// Detray include(s).
#include <detray/geometry/barcode.hpp>

// VecMem include(s).
#include <vecmem/edm/container.hpp>

namespace traccc::edm {

/// Interface to the cell module (SoA) container
///
/// This container describes the properties of the (pixel) detector modules that
/// had at least one active cell in an event.
///
template <typename BASE>
struct cell_module_interface : public BASE {

    // Inherit the base class's constructor(s)
    using BASE::BASE;

    /// Barcode of the surface that is on this module (non-const)
    TRACCC_HOST_DEVICE
    auto& surface_link() { return BASE::template get<0>(); }
    /// Barcode of the surface that is on this module (const)
    TRACCC_HOST_DEVICE
    auto const& surface_link() const { return BASE::template get<0>(); }

    /// Local<->Global transformation of the module's surface (non-const)
    TRACCC_HOST_DEVICE
    auto& placement() { return BASE::template get<1>(); }
    /// Local<->Global transformation of the module's surface (const)
    TRACCC_HOST_DEVICE
    auto const& placement() const { return BASE::template get<1>(); }

    /// Pixel activation threshold for the module (non-const)
    TRACCC_HOST_DEVICE
    auto& threshold() { return BASE::template get<2>(); }
    /// Pixel activation threshold for the module (const)
    TRACCC_HOST_DEVICE
    auto const& threshold() const { return BASE::template get<2>(); }

    /// Information about the module's segmentation (non-const)
    TRACCC_HOST_DEVICE
    auto& pixel_data() { return BASE::template get<3>(); }
    /// Information about the module's segmentation (const)
    TRACCC_HOST_DEVICE
    auto const& pixel_data() const { return BASE::template get<3>(); }

};  // struct cell_module_interface

/// Cell module (SoA) container
using cell_module_container = vecmem::edm::container<
    cell_module_interface, vecmem::edm::type::vector<detray::geometry::barcode>,
    vecmem::edm::type::vector<transform3>, vecmem::edm::type::vector<scalar>,
    vecmem::edm::type::vector<pixel_data> >;

}  // namespace traccc::edm
