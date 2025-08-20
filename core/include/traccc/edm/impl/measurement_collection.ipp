/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace traccc::edm {

template <typename BASE>
template <typename T>
TRACCC_HOST_DEVICE bool measurement<BASE>::operator==(
    const measurement<T>& other) const {

    return (identifier() == other.identifier());
}

template <typename BASE>
template <typename T>
TRACCC_HOST_DEVICE std::weak_ordering measurement<BASE>::operator<=>(
    const measurement<T>& other) const {

    if (surface_link() != other.surface_link()) {
        return (surface_link() <=> other.surface_link());
    } else {
        return (identifier() <=> other.identifier());
    }
}

}  // namespace traccc::edm
