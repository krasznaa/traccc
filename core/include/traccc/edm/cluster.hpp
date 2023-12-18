/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/edm/container.hpp"

// System include(s).
#include <variant>

namespace traccc {

/// Declare all cluster container types
using cluster_container_types = container_types<std::monostate, unsigned int>;

}  // namespace traccc
