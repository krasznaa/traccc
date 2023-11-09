/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/geometry/pixel_data.hpp"

// Detray include(s).
#include "detray/geometry/barcode.hpp"

// VecMem include(s).
#include "vecmem/edm/accessor.hpp"
#include "vecmem/edm/container.hpp"

namespace traccc::edm {

/// Pixel module (SoA) container
///
/// This container describes the properties of the (pixel) detector modules that
/// had at least one active cell in an event.
///
struct pixel_module_container
    : public vecmem::edm::container<vecmem::edm::schema<
          vecmem::edm::type::vector<detray::geometry::barcode>,
          vecmem::edm::type::vector<transform3>,
          vecmem::edm::type::vector<scalar>,
          vecmem::edm::type::vector<pixel_data> > > {

    /// @name Accessors for the individual container variables
    /// @{

    /// Barcode of the surface that is on this module.
    using surface_link = vecmem::edm::accessor<0, schema>;
    /// Local<->Global transformation of the module's surface
    using placement = vecmem::edm::accessor<1, schema>;
    /// Pixel activation threshold for the module
    using threshold = vecmem::edm::accessor<2, schema>;
    /// Information about the module's segmentation
    using pixel_data = vecmem::edm::accessor<3, schema>;

    /// @}

};  // struct pixel_module

}  // namespace traccc::edm
