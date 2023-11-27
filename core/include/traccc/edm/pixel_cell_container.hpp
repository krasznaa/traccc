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

// VecMem include(s).
#include "vecmem/edm/accessor.hpp"
#include "vecmem/edm/container.hpp"

namespace traccc::edm {

/// Pixel cell (SoA) container
struct pixel_cell_container
    : vecmem::edm::container<vecmem::edm::type::vector<channel_id>,
                             vecmem::edm::type::vector<channel_id>,
                             vecmem::edm::type::vector<scalar>,
                             vecmem::edm::type::vector<scalar>,
                             vecmem::edm::type::vector<unsigned int> > {

    /// @name Accessors for the individual container variables
    /// @{

    /// First (x) channel identifier of the cell
    using channel0 = vecmem::edm::accessor<0, schema_type>;
    /// Second (y) channel identifier of the cell
    using channel1 = vecmem::edm::accessor<1, schema_type>;
    /// Activation / signal strength of the cell
    using activation = vecmem::edm::accessor<2, schema_type>;
    /// Time of the cell signal
    using time = vecmem::edm::accessor<3, schema_type>;
    /// Index of the pixel module that the pixel cell belongs to
    using module_index = vecmem::edm::accessor<4, schema_type>;

    /// @}

};  // struct cell_container

}  // namespace traccc::edm
