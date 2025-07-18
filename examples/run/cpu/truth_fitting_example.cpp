/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../common/make_magnetic_field.hpp"
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/geometry/detector.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/propagation.hpp"
#include "traccc/utils/seed_generator.hpp"

// Detray include(s).
#include <detray/io/frontend/detector_reader.hpp>

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

using namespace traccc;
namespace po = boost::program_options;

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> ilogger = traccc::getDefaultLogger(
        "TracccExampleTruthFitting", traccc::Logging::Level::INFO);

    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::magnetic_field bfield_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::track_fitting fitting_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::program_options program_opts{
        "Truth Track Fitting on the Host",
        {detector_opts, bfield_opts, input_opts, propagation_opts, fitting_opts,
         performance_opts},
        argc,
        argv,
        logger().cloneWithSuffix("Options")};

    /// Type declarations
    using host_detector_type = traccc::default_detector::host;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    // Copy obejct
    vecmem::copy copy;

    // Performance writer
    traccc::fitting_performance_writer fit_performance_writer(
        traccc::fitting_performance_writer::config{},
        logger().clone("FittingPerformanceWriter"));

    /*****************************
     * Build a geometry
     *****************************/

    // B field value
    const auto field = traccc::details::make_magnetic_field(bfield_opts);

    // Read the detector
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(
        traccc::io::get_absolute_path(detector_opts.detector_file));
    if (!detector_opts.material_file.empty()) {
        reader_cfg.add_file(
            traccc::io::get_absolute_path(detector_opts.material_file));
    }
    if (!detector_opts.grid_file.empty()) {
        reader_cfg.add_file(
            traccc::io::get_absolute_path(detector_opts.grid_file));
    }
    const auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(host_mr, reader_cfg);

    /*****************************
     * Do the reconstruction
     *****************************/

    /// Standard deviations for seed track parameters
    static constexpr std::array<scalar, e_bound_size> stddevs = {
        0.03f * traccc::unit<scalar>::mm,
        0.03f * traccc::unit<scalar>::mm,
        0.017f,
        0.017f,
        0.001f / traccc::unit<scalar>::GeV,
        1.f * traccc::unit<scalar>::ns};

    // Fitting algorithm object
    traccc::fitting_config fit_cfg(fitting_opts);
    fit_cfg.propagation = propagation_opts;

    traccc::host::kalman_fitting_algorithm host_fitting(
        fit_cfg, host_mr, copy, logger().clone("FittingAlg"));

    // Seed generator
    traccc::seed_generator<host_detector_type> sg(host_det, stddevs);

    // Iterate over events
    for (auto event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Truth Track Candidates
        traccc::event_data evt_data(input_opts.directory, event, host_mr,
                                    input_opts.use_acts_geom_source, &host_det,
                                    input_opts.format, false);

        traccc::edm::track_candidate_container<traccc::default_algebra>::host
            truth_track_candidates{host_mr};
        evt_data.generate_truth_candidates(truth_track_candidates, sg, host_mr);

        // Run fitting
        auto track_states = host_fitting(
            host_det, field,
            {vecmem::get_data(truth_track_candidates.tracks),
             vecmem::get_data(truth_track_candidates.measurements)});

        print_fitted_tracks_statistics(track_states);

        const decltype(track_states)::size_type n_fitted_tracks =
            track_states.size();

        if (performance_opts.run) {

            for (unsigned int i = 0; i < n_fitted_tracks; i++) {
                const auto& trk_states_per_track = track_states.at(i).items;

                const auto& fit_res = track_states[i].header;

                fit_performance_writer.write(trk_states_per_track, fit_res,
                                             host_det, evt_data);
            }
        }
    }

    if (performance_opts.run) {
        fit_performance_writer.finalize();
    }

    return EXIT_SUCCESS;
}
