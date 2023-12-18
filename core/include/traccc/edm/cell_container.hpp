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

// VecMem include(s).
#include <vecmem/edm/container.hpp>

namespace traccc::edm {

/// Interface to the cell (SoA) container
template <typename BASE>
struct cell_interface : public BASE {

    // Inherit the base class's constructor(s)
    using BASE::BASE;

    /// First (x) channel identifier of the cell (non-const)
    TRACCC_HOST_DEVICE
    auto& channel0() { return BASE::template get<0>(); }
    /// First (x) channel identifier of the cell (const)
    TRACCC_HOST_DEVICE
    auto const& channel0() const { return BASE::template get<0>(); }

    /// Second (y) channel identifier of the cell (non-const)
    TRACCC_HOST_DEVICE
    auto& channel1() { return BASE::template get<1>(); }
    /// Second (y) channel identifier of the cell (const)
    TRACCC_HOST_DEVICE
    auto const& channel1() const { return BASE::template get<1>(); }

    /// Activation / signal strength of the cell (non-const)
    TRACCC_HOST_DEVICE
    auto& activation() { return BASE::template get<2>(); }
    /// Activation / signal strength of the cell (const)
    TRACCC_HOST_DEVICE
    auto const& activation() const { return BASE::template get<2>(); }

    /// Time of the cell signal (non-const)
    TRACCC_HOST_DEVICE
    auto& time() { return BASE::template get<3>(); }
    /// Time of the cell signal (const)
    TRACCC_HOST_DEVICE
    auto const& time() const { return BASE::template get<3>(); }

    /// Index of the pixel module that the pixel cell belongs to (non-const)
    TRACCC_HOST_DEVICE
    auto& module_index() { return BASE::template get<4>(); }
    /// Index of the pixel module that the pixel cell belongs to (const)
    TRACCC_HOST_DEVICE
    auto const& module_index() const { return BASE::template get<4>(); }

};  // struct cell_interface

/// Cell (SoA) container
using cell_container = vecmem::edm::container<
    cell_interface, vecmem::edm::type::vector<channel_id>,
    vecmem::edm::type::vector<channel_id>, vecmem::edm::type::vector<scalar>,
    vecmem::edm::type::vector<scalar>,
    vecmem::edm::type::vector<unsigned int> >;

}  // namespace traccc::edm
