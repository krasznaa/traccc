/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "make_magnetic_field.hpp"

// Project include(s)
#include "traccc/geometry/detector.hpp"

// Command line option include(s).
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/magnetic_field.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/throughput.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_fitting.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"

// I/O include(s).
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"
#include "traccc/io/utils.hpp"

// Performance measurement include(s).
#include "traccc/performance/throughput.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/performance/timing_info.hpp"

// VecMem include(s).
#include <vecmem/memory/binary_page_memory_resource.hpp>

// Indicators include(s).
#include <indicators/progress_bar.hpp>

// System include(s).
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <memory>

namespace traccc {

template <typename FULL_CHAIN_ALG, typename HOST_MR>
int throughput_st(std::string_view description, int argc, char* argv[],
                  bool use_host_caching) {
    std::unique_ptr<const traccc::Logger> logger = traccc::getDefaultLogger(
        "ThroughputExample", traccc::Logging::Level::INFO);

    // Program options.
    opts::detector detector_opts;
    opts::magnetic_field bfield_opts;
    opts::input_data input_opts;
    opts::clusterization clusterization_opts;
    opts::track_seeding seeding_opts;
    opts::track_finding finding_opts;
    opts::track_propagation propagation_opts;
    opts::track_fitting fitting_opts;
    opts::throughput throughput_opts;
    opts::program_options program_opts{
        description,
        {detector_opts, bfield_opts, input_opts, clusterization_opts,
         seeding_opts, finding_opts, propagation_opts, fitting_opts,
         throughput_opts},
        argc,
        argv,
        logger->cloneWithSuffix("Options")};

    // Set up the timing info holder.
    performance::timing_info times;

    // Memory resource to use in the test.
    HOST_MR uncached_host_mr;
    std::unique_ptr<vecmem::binary_page_memory_resource> cached_host_mr =
        std::make_unique<vecmem::binary_page_memory_resource>(uncached_host_mr);

    // Construct the detector description object.
    traccc::silicon_detector_description::host det_descr{uncached_host_mr};
    traccc::io::read_detector_description(
        det_descr, detector_opts.detector_file, detector_opts.digitization_file,
        (detector_opts.use_detray_detector ? traccc::data_format::json
                                           : traccc::data_format::csv));

    // Construct a Detray detector object, if supported by the configuration.
    traccc::default_detector::host detector{uncached_host_mr};
    if (detector_opts.use_detray_detector) {
        traccc::io::read_detector(
            detector, uncached_host_mr, detector_opts.detector_file,
            detector_opts.material_file, detector_opts.grid_file);
    }

    // Construct the magnetic field object.
    const auto field = details::make_magnetic_field(bfield_opts);

    vecmem::memory_resource& alg_host_mr =
        use_host_caching
            ? static_cast<vecmem::memory_resource&>(*cached_host_mr)
            : static_cast<vecmem::memory_resource&>(uncached_host_mr);

    // Read in all input events into memory.
    vecmem::vector<edm::silicon_cell_collection::host> input{&uncached_host_mr};
    {
        performance::timer t{"File reading", times};
        // Read the input cells into memory event-by-event.
        input.reserve(input_opts.events);
        for (std::size_t i = input_opts.skip;
             i < input_opts.skip + input_opts.events; ++i) {
            input.emplace_back(uncached_host_mr);
            static constexpr bool DEDUPLICATE = true;
            io::read_cells(input.back(), i, input_opts.directory,
                           logger->clone(), &det_descr, input_opts.format,
                           DEDUPLICATE, input_opts.use_acts_geom_source);
        }
    }

    // Algorithm configuration(s).
    detray::propagation::config propagation_config(propagation_opts);

    typename FULL_CHAIN_ALG::clustering_algorithm::config_type clustering_cfg(
        clusterization_opts);

    const traccc::seedfinder_config seedfinder_config(seeding_opts);
    const traccc::seedfilter_config seedfilter_config(seeding_opts);
    const traccc::spacepoint_grid_config spacepoint_grid_config(seeding_opts);

    typename FULL_CHAIN_ALG::finding_algorithm::config_type finding_cfg(
        finding_opts);
    finding_cfg.propagation = propagation_config;

    typename FULL_CHAIN_ALG::fitting_algorithm::config_type fitting_cfg(
        fitting_opts);
    fitting_cfg.propagation = propagation_config;

    // Set up the full-chain algorithm.
    std::unique_ptr<FULL_CHAIN_ALG> alg = std::make_unique<FULL_CHAIN_ALG>(
        alg_host_mr, clustering_cfg, seedfinder_config, spacepoint_grid_config,
        seedfilter_config, finding_cfg, fitting_cfg, det_descr, field,
        (detector_opts.use_detray_detector ? &detector : nullptr),
        logger->clone("FullChainAlg"));

    // Seed the random number generator.
    if (throughput_opts.random_seed == 0) {
        std::srand(static_cast<unsigned int>(std::time(0)));
    } else {
        std::srand(throughput_opts.random_seed);
    }

    // Set up a lambda that calls the correct function on the algorithm.
    std::function<std::size_t(const edm::silicon_cell_collection::host&)>
        process_event;
    if (throughput_opts.reco_stage == opts::throughput::stage::seeding) {
        process_event = [&](const edm::silicon_cell_collection::host& cells)
            -> std::size_t { return alg->seeding(cells).size(); };
    } else if (throughput_opts.reco_stage == opts::throughput::stage::full) {
        process_event = [&](const edm::silicon_cell_collection::host& cells)
            -> std::size_t { return (*alg)(cells).size(); };
    } else {
        throw std::invalid_argument("Unknown reconstruction stage");
    }

    // Dummy count uses output of tp algorithm to ensure the compiler
    // optimisations don't skip any step
    std::size_t rec_track_params = 0;

    // Cold Run events. To discard any "initialisation issues" in the
    // measurements.
    {
        // Set up a progress bar for the warm-up processing.
        indicators::ProgressBar progress_bar{
            indicators::option::BarWidth{50},
            indicators::option::PrefixText{"Warm-up processing "},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::MaxProgress{throughput_opts.cold_run_events}};

        // Measure the time of execution.
        performance::timer t{"Warm-up processing", times};

        // Process the requested number of events.
        for (std::size_t i = 0; i < throughput_opts.cold_run_events; ++i) {

            // Choose which event to process.
            const std::size_t event =
                (throughput_opts.deterministic_event_order
                     ? i
                     : static_cast<std::size_t>(std::rand())) %
                input_opts.events;

            // Process one event.
            rec_track_params += process_event(input[event]);
            progress_bar.tick();
        }
    }

    // Reset the dummy counter.
    rec_track_params = 0;

    {
        // Set up a progress bar for the event processing.
        indicators::ProgressBar progress_bar{
            indicators::option::BarWidth{50},
            indicators::option::PrefixText{"Event processing   "},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::MaxProgress{throughput_opts.processed_events}};

        // Measure the total time of execution.
        performance::timer t{"Event processing", times};

        // Process the requested number of events.
        for (std::size_t i = 0; i < throughput_opts.processed_events; ++i) {

            // Choose which event to process.
            const std::size_t event =
                (throughput_opts.deterministic_event_order
                     ? i
                     : static_cast<std::size_t>(std::rand())) %
                input_opts.events;

            // Process one event.
            rec_track_params += (*alg)(input[event]).size();
            progress_bar.tick();
        }
    }

    // Explicitly delete the objects in the correct order.
    alg.reset();
    cached_host_mr.reset();

    // Print some results.
    std::cout << "Reconstructed track parameters: " << rec_track_params
              << std::endl;
    std::cout << "Time totals:" << std::endl;
    std::cout << times << std::endl;
    std::cout << "Throughput:" << std::endl;
    std::cout << performance::throughput{throughput_opts.cold_run_events, times,
                                         "Warm-up processing"}
              << "\n"
              << performance::throughput{throughput_opts.processed_events,
                                         times, "Event processing"}
              << std::endl;

    // Return gracefully.
    return 0;
}

}  // namespace traccc
