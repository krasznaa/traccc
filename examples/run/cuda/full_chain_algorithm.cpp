/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "full_chain_algorithm.hpp"

// Project include(s).
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/finding/combinatorial_kalman_filter_algorithm.hpp"
#include "traccc/cuda/fitting/kalman_fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/make_bfield.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/utils/propagation.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// CUDA include(s).
#include <cuda_runtime_api.h>

// System include(s).
#include <iostream>
#include <stdexcept>

/// Helper macro for checking the return value of CUDA function calls
#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

namespace traccc::cuda {

struct full_chain_algorithm::impl {

    /// Constructor
    impl(vecmem::memory_resource& host_mr,
         const clustering_config& clustering_config,
         const seedfinder_config& finder_config,
         const spacepoint_grid_config& grid_config,
         const seedfilter_config& filter_config,
         const finding_config& finding_config,
         const fitting_config& fitting_config,
         const silicon_detector_description::host& det_descr,
         const bfield& field, host_detector_type* detector,
         const traccc::Logger& logger)
        : m_host_mr(host_mr),
          m_stream(),
          m_device_mr(),
          m_cached_device_mr(m_device_mr),
          m_copy(m_stream.cudaStream()),
          m_field_vec{0.f, 0.f, finder_config.bFieldInZ},
          m_field(traccc::cuda::make_bfield(field)),
          m_device_det_descr(
              static_cast<silicon_detector_description::buffer::size_type>(
                  det_descr.size()),
              m_device_mr),
          m_detector(detector),
          m_clusterization(memory_resource{m_cached_device_mr, &m_host_mr},
                           m_copy, m_stream, clustering_config),
          m_measurement_sorting(memory_resource{m_cached_device_mr, &m_host_mr},
                                m_copy, m_stream,
                                logger.cloneWithSuffix("MeasSortingAlg")),
          m_spacepoint_formation(
              memory_resource{m_cached_device_mr, &m_host_mr}, m_copy, m_stream,
              logger.cloneWithSuffix("SpFormationAlg")),
          m_seeding(finder_config, grid_config, filter_config,
                    memory_resource{m_cached_device_mr, &m_host_mr}, m_copy,
                    m_stream, logger.cloneWithSuffix("SeedingAlg")),
          m_track_parameter_estimation(
              memory_resource{m_cached_device_mr, &m_host_mr}, m_copy, m_stream,
              logger.cloneWithSuffix("TrackParEstAlg")),
          m_finding(finding_config,
                    memory_resource{m_cached_device_mr, &m_host_mr}, m_copy,
                    m_stream, logger.cloneWithSuffix("TrackFindingAlg")),
          m_fitting(fitting_config,
                    memory_resource{m_cached_device_mr, &m_host_mr}, m_copy,
                    m_stream, logger.cloneWithSuffix("TrackFittingAlg")) {}

    /// Host memory resource
    vecmem::memory_resource& m_host_mr;
    /// CUDA stream to use
    stream m_stream;
    /// Device memory resource
    vecmem::cuda::device_memory_resource m_device_mr;
    /// Device caching memory resource
    vecmem::binary_page_memory_resource m_cached_device_mr;
    /// (Asynchronous) Memory copy object
    vecmem::cuda::async_copy m_copy;

    /// Constant B field for the (seed) track parameter estimation
    traccc::vector3 m_field_vec;
    /// Constant B field for the track finding and fitting
    bfield m_field;

    /// Detector description buffer
    silicon_detector_description::buffer m_device_det_descr;
    /// Host detector
    host_detector_type* m_detector;
    /// Buffer holding the detector's payload on the device
    host_detector_type::buffer_type m_device_detector;
    /// View of the detector's payload on the device
    host_detector_type::view_type m_device_detector_view;

    /// @name Sub-algorithms used by this full-chain algorithm
    /// @{

    /// Clusterization algorithm
    clusterization_algorithm m_clusterization;
    /// Measurement sorting algorithm
    measurement_sorting_algorithm m_measurement_sorting;
    /// Spacepoint formation algorithm
    spacepoint_formation_algorithm<default_detector::device>
        m_spacepoint_formation;
    /// Seeding algorithm
    seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    track_params_estimation m_track_parameter_estimation;

    /// Track finding algorithm
    combinatorial_kalman_filter_algorithm m_finding;
    /// Track fitting algorithm
    kalman_fitting_algorithm m_fitting;

    /// @}

};  // struct full_chain_algorithm::impl

full_chain_algorithm::full_chain_algorithm(
    vecmem::memory_resource& host_mr,
    const clustering_config& clustering_config,
    const seedfinder_config& finder_config,
    const spacepoint_grid_config& grid_config,
    const seedfilter_config& filter_config,
    const finding_config& finding_config, const fitting_config& fitting_config,
    const silicon_detector_description::host& det_descr, const bfield& field,
    host_detector_type* detector, std::unique_ptr<const traccc::Logger> logger)
    : messaging(logger->clone()),
      m_impl(std::make_unique<impl>(host_mr, clustering_config, finder_config,
                                    grid_config, filter_config, finding_config,
                                    fitting_config, det_descr, field, detector,
                                    *logger)) {

    // Tell the user what device is being used.
    int device = 0;
    CUDA_ERROR_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_ERROR_CHECK(cudaGetDeviceProperties(&props, device));
    std::cout << "Using CUDA device: " << props.name << " [id: " << device
              << ", bus: " << props.pciBusID
              << ", device: " << props.pciDeviceID << "]" << std::endl;

    // Copy the detector (description) to the device.
    m_impl->m_copy(vecmem::get_data(det_descr), m_impl->m_device_det_descr)
        ->ignore();
    if (m_impl->m_detector != nullptr) {
        m_impl->m_device_detector = detray::get_buffer(
            *(m_impl->m_detector), m_impl->m_device_mr, m_impl->m_copy);
        m_impl->m_device_detector_view =
            detray::get_data(m_impl->m_device_detector);
    }
}

full_chain_algorithm::full_chain_algorithm(full_chain_algorithm&&) = default;

full_chain_algorithm::~full_chain_algorithm() = default;

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const edm::silicon_cell_collection::host& cells) const {

    // Create device copy of input collections
    edm::silicon_cell_collection::buffer cells_buffer(
        static_cast<unsigned int>(cells.size()), m_impl->m_cached_device_mr);
    m_impl->m_copy(vecmem::get_data(cells), cells_buffer)->ignore();

    // Run the clusterization (asynchronously).
    const measurement_collection_types::buffer measurements =
        m_impl->m_clusterization(cells_buffer, m_impl->m_device_det_descr);
    m_impl->m_measurement_sorting(measurements);

    // If we have a Detray detector, run the seeding, track finding and fitting.
    if (m_impl->m_detector != nullptr) {

        // Run the seed-finding (asynchronously).
        const auto spacepoints = m_impl->m_spacepoint_formation(
            m_impl->m_device_detector_view, measurements);
        const auto track_params = m_impl->m_track_parameter_estimation(
            measurements, spacepoints, m_impl->m_seeding(spacepoints),
            m_impl->m_field_vec);

        // Run the track finding (asynchronously).
        const auto track_candidates =
            m_impl->m_finding(m_impl->m_device_detector_view, m_impl->m_field,
                              measurements, track_params);

        // Run the track fitting (asynchronously).
        const auto track_states =
            m_impl->m_fitting(m_impl->m_device_detector_view, m_impl->m_field,
                              {track_candidates, measurements});

        // Copy a limited amount of result data back to the host.
        output_type result{&(m_impl->m_host_mr)};
        m_impl->m_copy(track_states.headers, result)->wait();
        return result;

    }
    // If not, copy the measurements back to the host, and return a dummy
    // object.
    else {

        // Copy the measurements back to the host.
        measurement_collection_types::host measurements_host(
            &(m_impl->m_host_mr));
        m_impl->m_copy(measurements, measurements_host)->wait();

        // Return an empty object.
        return {};
    }
}

}  // namespace traccc::cuda
