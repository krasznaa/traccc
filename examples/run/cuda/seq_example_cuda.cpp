/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/clusterization/spacepoint_formation.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/accelerator.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/cuda/managed_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// System include(s).
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>

int seq_run(const traccc::opts::detector& detector_opts,
            const traccc::opts::input_data& input_opts,
            const traccc::opts::clusterization& clusterization_opts,
            const traccc::opts::track_seeding& seeding_opts,
            const traccc::opts::track_finding& finding_opts,
            const traccc::opts::track_propagation& propagation_opts,
            const traccc::opts::performance& performance_opts,
            const traccc::opts::accelerator& accelerator_opts) {

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    vecmem::cuda::managed_memory_resource mng_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};

    // Read in the geometry.
    auto [surface_transforms, barcode_map] = traccc::io::read_geometry(
        detector_opts.detector_file,
        (detector_opts.use_detray_detector ? traccc::data_format::json
                                           : traccc::data_format::csv));

    using host_detector_type = detray::detector<detray::default_metadata,
                                                detray::host_container_types>;
    host_detector_type host_detector{mng_mr};
    host_detector_type::view_type host_detector_view;
    if (detector_opts.use_detray_detector) {
        // Set up the detector reader configuration.
        detray::io::detector_reader_config cfg;
        cfg.add_file(traccc::io::data_directory() +
                     detector_opts.detector_file);
        if (detector_opts.material_file.empty() == false) {
            cfg.add_file(traccc::io::data_directory() +
                         detector_opts.material_file);
        }
        if (detector_opts.grid_file.empty() == false) {
            cfg.add_file(traccc::io::data_directory() +
                         detector_opts.grid_file);
        }

        // Read the detector.
        auto det = detray::io::read_detector<host_detector_type>(mng_mr, cfg);
        host_detector = std::move(det.first);
        host_detector_view = detray::get_data(host_detector);
    }

    // Read the digitization configuration file
    auto digi_cfg =
        traccc::io::read_digitization_config(detector_opts.digitization_file);

    // Output stats
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;
    // uint64_t n_clusters = 0;
    uint64_t n_measurements = 0;
    uint64_t n_spacepoints = 0;
    uint64_t n_spacepoints_cuda = 0;
    uint64_t n_seeds = 0;
    uint64_t n_seeds_cuda = 0;
    uint64_t n_found_tracks = 0;
    uint64_t n_found_tracks_cuda = 0;
    uint64_t n_fitted_tracks = 0;
    uint64_t n_fitted_tracks_cuda = 0;

    // Type definitions
    using host_stepper_type =
        detray::rk_stepper<detray::bfield::const_field_t::view_t,
                           host_detector_type::transform3,
                           detray::constrained_step<>>;
    using host_navigator_type = detray::navigator<const host_detector_type>;
    using host_finding_algorithm =
        traccc::finding_algorithm<host_stepper_type, host_navigator_type>;
    using host_fitting_algorithm = traccc::fitting_algorithm<
        traccc::kalman_fitter<host_stepper_type, host_navigator_type>>;

    using device_detector_type =
        detray::detector<detray::default_metadata,
                         detray::device_container_types>;
    using device_stepper_type =
        detray::rk_stepper<detray::bfield::const_field_t::view_t,
                           device_detector_type::transform3,
                           detray::constrained_step<>>;
    using device_navigator_type = detray::navigator<const device_detector_type>;
    using device_finding_algorithm =
        traccc::cuda::finding_algorithm<device_stepper_type,
                                        device_navigator_type>;
    using device_fitting_algorithm = traccc::cuda::fitting_algorithm<
        traccc::kalman_fitter<device_stepper_type, device_navigator_type>>;

    // Constant B field for the track finding and fitting
    const traccc::vector3 field_vec = {0.f, 0.f,
                                       seeding_opts.seedfinder.bFieldInZ};
    const detray::bfield::const_field_t field =
        detray::bfield::create_const_field(field_vec);

    // Algorithm configuration(s).
    host_finding_algorithm::config_type finding_cfg;
    finding_cfg.min_track_candidates_per_track =
        finding_opts.track_candidates_range[0];
    finding_cfg.max_track_candidates_per_track =
        finding_opts.track_candidates_range[1];
    finding_cfg.chi2_max = finding_opts.chi2_max;
    finding_cfg.propagation = propagation_opts.config;

    host_fitting_algorithm::config_type fitting_cfg;
    fitting_cfg.propagation = propagation_opts.config;

    // Algorithms.
    traccc::clusterization_algorithm ca(host_mr);
    traccc::spacepoint_formation sf(host_mr);
    traccc::seeding_algorithm sa(seeding_opts.seedfinder,
                                 {seeding_opts.seedfinder},
                                 seeding_opts.seedfilter, host_mr);
    traccc::track_params_estimation tp(host_mr);
    host_finding_algorithm finding_alg(finding_cfg);
    host_fitting_algorithm fitting_alg(fitting_cfg);

    traccc::cuda::stream stream;

    vecmem::cuda::async_copy copy{stream.cudaStream()};

    traccc::cuda::clusterization_algorithm ca_cuda(
        mr, copy, stream, clusterization_opts.target_cells_per_partition);
    traccc::cuda::seeding_algorithm sa_cuda(
        seeding_opts.seedfinder, {seeding_opts.seedfinder},
        seeding_opts.seedfilter, mr, copy, stream);
    traccc::cuda::track_params_estimation tp_cuda(mr, copy, stream);
    device_finding_algorithm finding_alg_cuda(finding_cfg, mr, copy, stream);
    device_fitting_algorithm fitting_alg_cuda(fitting_cfg, mr, copy, stream);

    // performance writer
    traccc::seeding_performance_writer sd_performance_writer(
        traccc::seeding_performance_writer::config{});

    traccc::performance::timing_info elapsedTimes;

    // Loop over events
    for (unsigned int event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Instantiate host containers/collections
        traccc::io::cell_reader_output read_out_per_event(mr.host);
        traccc::clusterization_algorithm::output_type measurements_per_event{
            &host_mr};
        traccc::spacepoint_formation::output_type spacepoints_per_event{
            &host_mr};
        traccc::seeding_algorithm::output_type seeds{&host_mr};
        traccc::track_params_estimation::output_type params{&host_mr};
        host_finding_algorithm::output_type track_candidates{&host_mr};
        host_fitting_algorithm::output_type track_states{&host_mr};

        // Instantiate cuda containers/collections
        traccc::spacepoint_collection_types::buffer spacepoints_cuda_buffer(
            0, *mr.host);
        traccc::seed_collection_types::buffer seeds_cuda_buffer(0, *mr.host);
        traccc::bound_track_parameters_collection_types::buffer
            params_cuda_buffer(0, *mr.host);
        device_finding_algorithm::output_type track_candidates_cuda_buffer;
        device_fitting_algorithm::output_type track_states_cuda_buffer;

        {
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            {
                traccc::performance::timer t("File reading  (cpu)",
                                             elapsedTimes);
                // Read the cells from the relevant event file into host memory.
                traccc::io::read_cells(read_out_per_event, event,
                                       input_opts.directory, input_opts.format,
                                       &surface_transforms, &digi_cfg,
                                       barcode_map.get());
            }  // stop measuring file reading timer

            const traccc::cell_collection_types::host& cells_per_event =
                read_out_per_event.cells;
            const traccc::cell_module_collection_types::host&
                modules_per_event = read_out_per_event.modules;

            /*-----------------------------
                Clusterization and Spacepoint Creation (cuda)
            -----------------------------*/
            // Create device copy of input collections
            traccc::cell_collection_types::buffer cells_buffer(
                cells_per_event.size(), mr.main);
            copy(vecmem::get_data(cells_per_event), cells_buffer);
            traccc::cell_module_collection_types::buffer modules_buffer(
                modules_per_event.size(), mr.main);
            copy(vecmem::get_data(modules_per_event), modules_buffer);

            {
                traccc::performance::timer t("Clusterization (cuda)",
                                             elapsedTimes);
                // Reconstruct it into spacepoints on the device.
                spacepoints_cuda_buffer =
                    ca_cuda(cells_buffer, modules_buffer).first;
                stream.synchronize();
            }  // stop measuring clusterization cuda timer

            if (accelerator_opts.compare_with_cpu) {

                /*-----------------------------
                    Clusterization (cpu)
                -----------------------------*/

                {
                    traccc::performance::timer t("Clusterization  (cpu)",
                                                 elapsedTimes);
                    measurements_per_event =
                        ca(cells_per_event, modules_per_event);
                }  // stop measuring clusterization cpu timer

                /*---------------------------------
                    Spacepoint formation (cpu)
                ---------------------------------*/

                {
                    traccc::performance::timer t("Spacepoint formation  (cpu)",
                                                 elapsedTimes);
                    spacepoints_per_event =
                        sf(measurements_per_event, modules_per_event);
                }  // stop measuring spacepoint formation cpu timer
            }

            /*----------------------------
                Seeding algorithm
            ----------------------------*/

            // CUDA

            {
                traccc::performance::timer t("Seeding (cuda)", elapsedTimes);
                seeds_cuda_buffer = sa_cuda(spacepoints_cuda_buffer);
                stream.synchronize();
            }  // stop measuring seeding cuda timer

            // CPU

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Seeding  (cpu)", elapsedTimes);
                seeds = sa(spacepoints_per_event);
            }  // stop measuring seeding cpu timer

            /*----------------------------
            Track params estimation
            ----------------------------*/

            // CUDA

            {
                traccc::performance::timer t("Track params (cuda)",
                                             elapsedTimes);
                params_cuda_buffer = tp_cuda(spacepoints_cuda_buffer,
                                             seeds_cuda_buffer, field_vec);
                stream.synchronize();
            }  // stop measuring track params timer

            // CPU

            if (accelerator_opts.compare_with_cpu) {
                traccc::performance::timer t("Track params  (cpu)",
                                             elapsedTimes);
                params = tp(spacepoints_per_event, seeds, field_vec);
            }  // stop measuring track params cpu timer

            /*----------------------------
            Track finding
            ----------------------------*/

            // CUDA

            {
                traccc::performance::timer t("Track finding (cuda)",
                                             elapsedTimes);

                // Create a navigation buffer.
                auto navigation_buffer = detray::create_candidates_buffer(
                    host_detector,
                    finding_alg_cuda.get_config().max_num_branches_per_seed *
                        copy.get_size(seeds_cuda_buffer),
                    mr.main, mr.host);
                track_candidates_cuda_buffer = finding_alg_cuda(
                    host_detector_view, field, navigation_buffer,
                    measurements_cuda_buffer, params_cuda_buffer);
            }  // stop measuring track finding cuda timer

        }  // Stop measuring wall time

        /*----------------------------------
          compare cpu and cuda result
          ----------------------------------*/

        traccc::spacepoint_collection_types::host spacepoints_per_event_cuda;
        traccc::seed_collection_types::host seeds_cuda;
        traccc::bound_track_parameters_collection_types::host params_cuda;

        copy(spacepoints_cuda_buffer, spacepoints_per_event_cuda)->wait();
        copy(seeds_cuda_buffer, seeds_cuda)->wait();
        copy(params_cuda_buffer, params_cuda)->wait();

        if (accelerator_opts.compare_with_cpu) {

            // Show which event we are currently presenting the results for.
            std::cout << "===>>> Event " << event << " <<<===" << std::endl;

            // Compare the spacepoints made on the host and on the device.
            traccc::collection_comparator<traccc::spacepoint>
                compare_spacepoints{"spacepoints"};
            compare_spacepoints(vecmem::get_data(spacepoints_per_event),
                                vecmem::get_data(spacepoints_per_event_cuda));

            // Compare the seeds made on the host and on the device
            traccc::collection_comparator<traccc::seed> compare_seeds{
                "seeds", traccc::details::comparator_factory<traccc::seed>{
                             vecmem::get_data(spacepoints_per_event),
                             vecmem::get_data(spacepoints_per_event_cuda)}};
            compare_seeds(vecmem::get_data(seeds),
                          vecmem::get_data(seeds_cuda));

            // Compare the track parameters made on the host and on the device.
            traccc::collection_comparator<traccc::bound_track_parameters>
                compare_track_parameters{"track parameters"};
            compare_track_parameters(vecmem::get_data(params),
                                     vecmem::get_data(params_cuda));
        }
        /// Statistics
        n_modules += read_out_per_event.modules.size();
        n_cells += read_out_per_event.cells.size();
        n_measurements += measurements_per_event.size();
        n_spacepoints += spacepoints_per_event.size();
        n_seeds += seeds.size();
        n_spacepoints_cuda += spacepoints_per_event_cuda.size();
        n_seeds_cuda += seeds_cuda.size();

        if (performance_opts.run) {

            traccc::event_map evt_map(
                event, detector_opts.detector_file,
                detector_opts.digitization_file, input_opts.directory,
                input_opts.directory, input_opts.directory, host_mr);
            sd_performance_writer.write(
                vecmem::get_data(seeds_cuda),
                vecmem::get_data(spacepoints_per_event_cuda), evt_map);
        }
    }

    if (performance_opts.run) {
        sd_performance_writer.finalize();
    }

    std::cout << "==> Statistics ... " << std::endl;
    std::cout << "- read    " << n_cells << " cells from " << n_modules
              << " modules" << std::endl;
    std::cout << "- created (cpu)  " << n_measurements << " measurements     "
              << std::endl;
    std::cout << "- created (cpu)  " << n_spacepoints << " spacepoints     "
              << std::endl;
    std::cout << "- created (cuda) " << n_spacepoints_cuda
              << " spacepoints     " << std::endl;

    std::cout << "- created  (cpu) " << n_seeds << " seeds" << std::endl;
    std::cout << "- created (cuda) " << n_seeds_cuda << " seeds" << std::endl;
    std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 0;
}

// The main routine
//
int main(int argc, char* argv[]) {

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::clusterization clusterization_opts;
    traccc::opts::track_seeding seeding_opts;
    traccc::opts::track_finding finding_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::accelerator accelerator_opts;
    traccc::opts::program_options program_opts{
        "Full Tracking Chain Using CUDA",
        {detector_opts, input_opts, clusterization_opts, seeding_opts,
         finding_opts, propagation_opts, performance_opts, accelerator_opts},
        argc,
        argv};

    // Run the application.
    return seq_run(detector_opts, input_opts, clusterization_opts, seeding_opts,
                   finding_opts, propagation_opts, performance_opts,
                   accelerator_opts);
}
