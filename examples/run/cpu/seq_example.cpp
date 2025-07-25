/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// core
#include "traccc/geometry/detector.hpp"
#include "traccc/utils/memory_resource.hpp"
#include "traccc/utils/propagation.hpp"

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/write.hpp"

// algorithms
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// performance
#include "traccc/efficiency/finding_performance_writer.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"

// options
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/output_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_resolution.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/options/truth_finding.hpp"

// examples
#include "../common/make_magnetic_field.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/copy.hpp>

// System include(s).
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <memory>

int seq_run(const traccc::opts::input_data& input_opts,
            const traccc::opts::output_data& output_opts,
            const traccc::opts::detector& detector_opts,
            const traccc::opts::magnetic_field& bfield_opts,
            const traccc::opts::clusterization& /*clusterization_opts*/,
            const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::track_finding& finding_opts,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::track_resolution& resolution_opts,
            const traccc::opts::track_fitting& fitting_opts,
            const traccc::opts::performance& performance_opts,
            const traccc::opts::truth_finding& truth_finding_opts,
            std::unique_ptr<const traccc::Logger> ilogger) {
    TRACCC_LOCAL_LOGGER(std::move(ilogger));

    // Memory resource used by the application.
    vecmem::host_memory_resource host_mr;

    // Copy obejct
    vecmem::copy copy;

    // Construct the detector description object.
    traccc::silicon_detector_description::host det_descr{host_mr};
    traccc::io::read_detector_description(
        det_descr, detector_opts.detector_file, detector_opts.digitization_file,
        (detector_opts.use_detray_detector ? traccc::data_format::json
                                           : traccc::data_format::csv));
    traccc::silicon_detector_description::data det_descr_data{
        vecmem::get_data(det_descr)};

    // Construct a Detray detector object, if supported by the configuration.
    traccc::default_detector::host detector{host_mr};
    if (detector_opts.use_detray_detector) {
        traccc::io::read_detector(
            detector, host_mr, detector_opts.detector_file,
            detector_opts.material_file, detector_opts.grid_file);
    }

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_seeds = 0;
    uint64_t n_found_tracks = 0;
    uint64_t n_ambiguity_free_tracks = 0;
    uint64_t n_fitted_tracks = 0;

    // Type definitions
    using spacepoint_formation_algorithm =
        traccc::host::silicon_pixel_spacepoint_formation_algorithm;
    using finding_algorithm =
        traccc::host::combinatorial_kalman_filter_algorithm;
    using fitting_algorithm = traccc::host::kalman_fitting_algorithm;

    // Constant B field for the track finding and fitting
    const traccc::vector3 field_vec(seeding_opts);
    const auto field = traccc::details::make_magnetic_field(bfield_opts);

    // Algorithm configuration(s).
    detray::propagation::config propagation_config(propagation_opts);

    const traccc::seedfinder_config seedfinder_config(seeding_opts);
    const traccc::seedfilter_config seedfilter_config(seeding_opts);
    const traccc::spacepoint_grid_config spacepoint_grid_config(seeding_opts);

    finding_algorithm::config_type finding_cfg(finding_opts);
    finding_cfg.propagation = propagation_config;

    traccc::host::greedy_ambiguity_resolution_algorithm::config_type
        resolution_config(resolution_opts);

    fitting_algorithm::config_type fitting_cfg(fitting_opts);
    fitting_cfg.propagation = propagation_config;

    // Algorithms
    traccc::host::sparse_ccl_algorithm cc(host_mr,
                                          logger().clone("SparseCclAlg"));
    traccc::host::measurement_creation_algorithm mc(
        host_mr, logger().clone("MeasCreationAlg"));
    spacepoint_formation_algorithm sf(host_mr,
                                      logger().clone("SpFormationAlg"));
    traccc::host::seeding_algorithm sa(
        seedfinder_config, spacepoint_grid_config, seedfilter_config, host_mr,
        logger().clone("SeedingAlg"));
    traccc::host::track_params_estimation tp(host_mr,
                                             logger().clone("TrackParEstAlg"));

    finding_algorithm finding_alg(finding_cfg, host_mr,
                                  logger().clone("FindingAlg"));
    traccc::host::greedy_ambiguity_resolution_algorithm resolution_alg(
        resolution_config, host_mr, logger().clone("AmbiguityResolutionAlg"));
    fitting_algorithm fitting_alg(fitting_cfg, host_mr, copy,
                                  logger().clone("FittingAlg"));

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{},
        logger().clone("SeedingPerformanceWriter"));
    traccc::finding_performance_writer find_performance_writer(
        traccc::finding_performance_writer::config{.truth_config =
                                                       truth_finding_opts},
        logger().clone("FindingPerformanceWriter"));
    traccc::finding_performance_writer::config ar_writer_cfg;
    ar_writer_cfg.file_path = "performance_track_ambiguity_resolution.root";
    ar_writer_cfg.algorithm_name = "ambiguity_resolution";
    traccc::finding_performance_writer ar_performance_writer(
        ar_writer_cfg, logger().clone("AmbiResFindingPerformanceWriter"));
    traccc::fitting_performance_writer fit_performance_writer(
        traccc::fitting_performance_writer::config{},
        logger().clone("FittingPerformanceWriter"));

    // Timers
    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (std::size_t event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        traccc::edm::silicon_cell_collection::host cells_per_event{host_mr};
        traccc::host::sparse_ccl_algorithm::output_type clusters_per_event{
            host_mr};
        traccc::host::measurement_creation_algorithm::output_type
            measurements_per_event{&host_mr};
        spacepoint_formation_algorithm::output_type spacepoints_per_event{
            host_mr};
        traccc::host::seeding_algorithm::output_type seeds{host_mr};
        traccc::host::track_params_estimation::output_type params{&host_mr};
        finding_algorithm::output_type track_candidates{host_mr};
        traccc::host::greedy_ambiguity_resolution_algorithm::output_type
            resolved_track_candidates{host_mr};
        fitting_algorithm::output_type track_states{&host_mr};

        {  // Start measuring wall time.
            traccc::performance::timer timer_wall{"Wall time", elapsedTimes};

            {
                traccc::performance::timer timer{"Read cells", elapsedTimes};
                // Read the cells from the relevant event file
                static constexpr bool DEDUPLICATE = true;
                traccc::io::read_cells(
                    cells_per_event, event, input_opts.directory,
                    logger().clone(), &det_descr, input_opts.format,
                    DEDUPLICATE, input_opts.use_acts_geom_source);
            }

            /*-------------------
                Clusterization
              -------------------*/

            {
                traccc::performance::timer timer{"Clusterization",
                                                 elapsedTimes};

                clusters_per_event = cc(vecmem::get_data(cells_per_event));
                measurements_per_event =
                    mc(vecmem::get_data(cells_per_event),
                       vecmem::get_data(clusters_per_event), det_descr_data);
            }

            // Perform seeding, track finding and fitting only when using a
            // Detray geometry.
            if (detector_opts.use_detray_detector) {

                /*------------------------
                    Spacepoint formation
                  ------------------------*/

                {
                    traccc::performance::timer timer{"Spacepoint formation",
                                                     elapsedTimes};
                    spacepoints_per_event =
                        sf(detector, vecmem::get_data(measurements_per_event));
                }
                if (output_opts.directory != "") {
                    traccc::io::write(event, output_opts.directory,
                                      output_opts.format,
                                      vecmem::get_data(spacepoints_per_event),
                                      vecmem::get_data(measurements_per_event));
                }

                /*-----------------------
                  Seeding algorithm
                  -----------------------*/

                {
                    traccc::performance::timer timer{"Seeding", elapsedTimes};
                    seeds = sa(vecmem::get_data(spacepoints_per_event));
                }
                if (output_opts.directory != "") {
                    traccc::io::write(event, output_opts.directory,
                                      output_opts.format,
                                      vecmem::get_data(seeds),
                                      vecmem::get_data(spacepoints_per_event));
                }

                /*----------------------------
                  Track params estimation
                  ----------------------------*/

                {
                    traccc::performance::timer timer{"Track params estimation",
                                                     elapsedTimes};
                    params = tp(vecmem::get_data(measurements_per_event),
                                vecmem::get_data(spacepoints_per_event),
                                vecmem::get_data(seeds), field_vec);
                }

                {
                    traccc::performance::timer timer{"Track finding",
                                                     elapsedTimes};
                    track_candidates =
                        finding_alg(detector, field,
                                    vecmem::get_data(measurements_per_event),
                                    vecmem::get_data(params));
                }
                if (output_opts.directory != "") {
                    traccc::io::write(
                        event, output_opts.directory, output_opts.format,
                        vecmem::get_data(track_candidates),
                        vecmem::get_data(measurements_per_event), detector);
                }

                {
                    // Perform ambiguity resolution only if asked for.
                    traccc::performance::timer timer{
                        "Track ambiguity resolution", elapsedTimes};
                    resolved_track_candidates = resolution_alg(
                        {vecmem::get_data(track_candidates),
                         vecmem::get_data(measurements_per_event)});
                }

                {
                    traccc::performance::timer timer{"Track fitting",
                                                     elapsedTimes};
                    track_states = fitting_alg(
                        detector, field,
                        {vecmem::get_data(resolved_track_candidates),
                         vecmem::get_data(measurements_per_event)});
                }
            }

            /*----------------------------
              Statistics
              ----------------------------*/

            n_cells += cells_per_event.size();
            n_measurements += measurements_per_event.size();
            n_spacepoints += spacepoints_per_event.size();
            n_seeds += seeds.size();
            n_found_tracks += track_candidates.size();
            n_ambiguity_free_tracks += resolved_track_candidates.size();
            n_fitted_tracks += track_states.size();

        }  // Stop measuring Wall time.

        /*------------
             Writer
          ------------*/

        if (performance_opts.run) {

            traccc::event_data evt_data(input_opts.directory, event, host_mr,
                                        input_opts.use_acts_geom_source,
                                        &detector, input_opts.format, true);
            evt_data.fill_cca_result(cells_per_event, clusters_per_event,
                                     measurements_per_event, det_descr);

            sd_performance_writer.write(
                vecmem::get_data(seeds),
                vecmem::get_data(spacepoints_per_event),
                vecmem::get_data(measurements_per_event), evt_data);
            find_performance_writer.write(
                vecmem::get_data(track_candidates),
                vecmem::get_data(measurements_per_event), evt_data);
            ar_performance_writer.write(
                vecmem::get_data(resolved_track_candidates),
                vecmem::get_data(measurements_per_event), evt_data);

            for (unsigned int i = 0; i < track_states.size(); i++) {
                const auto& trk_states_per_track = track_states.at(i).items;

                const auto& fit_res = track_states[i].header;

                fit_performance_writer.write(trk_states_per_track, fit_res,
                                             detector, evt_data);
            }
        }
    }

    if (performance_opts.run) {
        sd_performance_writer.finalize();
        find_performance_writer.finalize();
        ar_performance_writer.finalize();
        fit_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read     " << n_cells << " cells" << std::endl;
    std::cout << "- created  " << n_measurements << " measurements. "
              << std::endl;
    std::cout << "- created  " << n_spacepoints << " space points. "
              << std::endl;
    std::cout << "- created  " << n_seeds << " seeds" << std::endl;
    std::cout << "- found    " << n_found_tracks << " tracks" << std::endl;
    std::cout << "- resolved " << n_ambiguity_free_tracks << " tracks"
              << std::endl;
    std::cout << "- fitted   " << n_fitted_tracks << " tracks" << std::endl;
    std::cout << "==> Elapsed times...\n" << elapsedTimes << std::endl;

    return EXIT_SUCCESS;
}

// The main routine
//
int main(int argc, char* argv[]) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "TracccExampleSeqCpu", traccc::Logging::Level::INFO);

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::magnetic_field bfield_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::output_data output_opts{traccc::data_format::obj, ""};
    traccc::opts::clusterization clusterization_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::track_resolution resolution_opts;
    traccc::opts::track_fitting fitting_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::truth_finding truth_finding_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain on the Host",
        {detector_opts, bfield_opts, input_opts, output_opts,
         clusterization_opts, seeding_opts, finding_opts, resolution_opts,
         fitting_opts, propagation_opts, performance_opts, truth_finding_opts},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Run the application.
    return seq_run(input_opts, output_opts, detector_opts, bfield_opts,
                   clusterization_opts, seeding_opts, finding_opts,
                   propagation_opts, resolution_opts, fitting_opts,
                   performance_opts, truth_finding_opts, logger->clone());
}
