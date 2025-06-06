/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/options/accelerator.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/output_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_resolution.hpp"
#include "traccc/options/track_seeding.hpp"

namespace traccc::apps {

/// Command line options used in the sequence example applications
struct program_opts {

    /// Detector options
    opts::detector detector;

    /// Input data options
    opts::input_data input;
    /// Output data options
    opts::output_data output{data_format::obj, ""};

    /// @name Algorithm options
    /// @{

    /// Clusterization options
    opts::clusterization clusterization;
    /// Seeding options
    opts::track_seeding seeding;
    /// Track finding options
    opts::track_finding finding;
    /// Track propagation options
    opts::track_propagation propagation;
    /// Track ambiguity resolution options
    opts::track_resolution resolution;
    /// Track fitting options
    opts::track_fitting fitting;

    /// @}

    /// Performance measurement options
    opts::performance performance;
    /// Accelerator options
    opts::accelerator accelerator;

};  // struct program_opts

/// Parse the command line options for the algorithm sequence application
///
/// @param argc The @c argc argument received by @c main(...)
/// @param argv The @c argv argument received by @c main(...)
///
/// @return The parsed command line arguments
///
program_opts parse_cmdl(int argc, char* argv[]);

}  // namespace traccc::apps
