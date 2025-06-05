/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/edm/track_fit_collection.hpp"

namespace traccc::details {

/// @c traccc::is_same_object specialisation for @c traccc::<edm::track_fit<T>
template <typename T>
class is_same_object<edm::track_fit<T>> {

    public:
    /// Constructor with a reference object, and an allowed uncertainty
    is_same_object(const edm::track_fit<T>& ref, scalar unc = float_epsilon)
        : m_ref(ref), m_unc(unc) {}

    /// Specialised implementation for @c traccc::measurement
    bool operator()(const edm::track_fit<T>& obj) const {

        return ((obj.fit_outcome() == m_ref.fit_outcome()) &&
                is_same_object<bound_track_parameters<>>(
                    obj.params(), m_unc)(m_ref.params()) &&
                is_same_scalar(obj.ndf(), m_ref.ndf(), m_unc) &&
                is_same_scalar(obj.chi2(), m_ref.chi2(), m_unc) &&
                is_same_scalar(obj.pval(), m_ref.pval(), m_unc) &&
                (obj.nholes() == m_ref.nholes()));
    }

    private:
    /// The reference object
    const edm::track_fit<T> m_ref;
    /// The uncertainty
    const scalar m_unc;

};  // class is_same_object<edm::track_fit<T>>

}  // namespace traccc::details
