/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "seq_example.hpp"

// Project include(s).
#include "traccc/options/program_options.hpp"
#include "traccc/utils/logging.hpp"

namespace traccc::apps {

program_opts parse_cmdl(int argc, char* argv[]) {

    // Create the program options object.
    program_opts opts;

    // Create and run the program options parser.
    opts::program_options po{
        "TRACCC Full Tracking Chain",
        {opts.detector, opts.input, opts.output, opts.clusterization,
         opts.seeding, opts.finding, opts.propagation, opts.resolution,
         opts.fitting, opts.performance, opts.accelerator},
        argc,
        argv,
        traccc::getDefaultLogger("traccc::apps::parse_cmdl",
                                 traccc::Logging::Level::INFO)};

    // Return the parsed options.
    return opts;
}

}  // namespace traccc::apps
