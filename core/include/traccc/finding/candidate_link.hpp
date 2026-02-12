/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/utils/pair.hpp"

namespace traccc {

// A link that contains the index of corresponding measurement and the index of
// a link from a previous step of track finding
struct candidate_link {
    // Step on which this link was found
    unsigned int step = 0u;

    // Index of the previous candidate
    unsigned int previous_candidate_idx = 0u;

    // Measurement index
    unsigned int meas_idx = 0u;

    // Index to the initial seed
    unsigned int seed_idx = 0u;

    // How many times it skipped a surface
    unsigned int n_skipped = 0u;

    // Number of consecutive holes; reset on measurement
    unsigned int n_consecutive_skipped = 0u;

    // chi2
    traccc::scalar chi2 = 0.f;

    // chi2 sum
    traccc::scalar chi2_sum = 0.f;

    // degrees of freedom
    unsigned int ndf_sum = 0u;
};

}  // namespace traccc
