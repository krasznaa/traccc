/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/primitives.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cluster.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/edm/pixel_cell_container.hpp"
#include "traccc/edm/pixel_module_container.hpp"

// VecMem include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/device_vector.hpp"

namespace traccc::detail {

/// Function used for retrieving the cell signal based on the module id
TRACCC_HOST_DEVICE
inline scalar signal_cell_modelling(
    scalar signal_in,
    const edm::pixel_module_container::const_device& /*modules*/,
    unsigned int /*module_index*/) {

    return signal_in;
}

/// Function for pixel segmentation
TRACCC_HOST_DEVICE
inline vector2 position_from_cell(
    const edm::pixel_cell_container::const_device& cells,
    unsigned int cell_index,
    const edm::pixel_module_container::const_device& modules,
    unsigned int module_index) {

    const pixel_data& pixel =
        edm::pixel_module_container::pixel_data::get(modules)[module_index];
    const auto channel0 =
        edm::pixel_cell_container::channel0::get(cells)[cell_index];
    const auto channel1 =
        edm::pixel_cell_container::channel1::get(cells)[cell_index];
    return {pixel.min_center_x + channel0 * pixel.pitch_x,
            pixel.min_center_y + channel1 * pixel.pitch_y};
}

/// Function used for calculating the properties of the cluster during
/// measurement creation
///
/// @param[in] cluster The vector of indices to the cells describing the
///                    identified cluster
/// @param[in] cells   The cell container
/// @param[in] modules The cell module container
/// @param[in] module_index The index of the module that the cluster belongs to
/// @param[out] mean   The mean position of the cluster/measurement
/// @param[out] var    The variation on the mean position of the
///                    cluster/measurement
/// @param[out] totalWeight The total weight of the cluster/measurement
///
TRACCC_HOST inline void calc_cluster_properties(
    const vecmem::device_vector<const unsigned int>& cluster,
    const edm::pixel_cell_container::const_device& cells,
    const edm::pixel_module_container::const_device& modules,
    unsigned int module_index, point2& mean, point2& var, scalar& totalWeight) {

    // Helper aliases.
    using cell_acc = edm::pixel_cell_container;
    using module_acc = edm::pixel_module_container;

    // Loop over the cells of the cluster.
    for (unsigned int cell_index : cluster) {

        // Translate the cell readout value into a weight.
        const scalar weight =
            signal_cell_modelling(cell_acc::activation::get(cells)[cell_index],
                                  modules, module_index);

        // Only consider cells over a minimum threshold.
        if (weight > module_acc::threshold::get(modules)[module_index]) {

            // Update all output properties with this cell.
            totalWeight += weight;
            const point2 cell_position =
                position_from_cell(cells, cell_index, modules, module_index);
            const point2 prev = mean;
            const point2 diff = cell_position - prev;

            mean = prev + (weight / totalWeight) * diff;
            for (std::size_t i = 0; i < 2; ++i) {
                var[i] =
                    var[i] + weight * (diff[i]) * (cell_position[i] - mean[i]);
            }
        }
    }
}

/// Function used for calculating the properties of the cluster during
/// measurement creation
///
/// @param[out] measurements is the measurement collection where the measurement
/// object will be filled
/// @param[in] cluster The indices to the input cells
/// @param[in] cells   The cell container
/// @param[in] modules The cell module container
///
TRACCC_HOST inline void fill_measurement(
    measurement_collection_types::host& measurements,
    cluster_container_types::host::item_vector::const_reference cluster,
    const edm::pixel_cell_container::host& cells,
    const edm::pixel_module_container::host& modules) {

    // To calculate the mean and variance with high numerical stability
    // we use a weighted variant of Welford's algorithm. This is a
    // single-pass online algorithm that works well for large numbers
    // of samples, as well as samples with very high values.
    //
    // To learn more about this algorithm please refer to:
    // [1] https://doi.org/10.1080/00401706.1962.10490022
    // [2] The Art of Computer Programming, Donald E. Knuth, second
    //     edition, chapter 4.2.2.

    // A security check.
    assert(cluster.empty() == false);

    // Get the module index.
    const unsigned int module_index =
        edm::pixel_cell_container::module_index::get(cells)[cluster.front()];

    // Calculate the cluster properties
    scalar totalWeight = 0.;
    point2 mean{0., 0.}, var{0., 0.};
    const vecmem::device_vector<const unsigned int> cluster_device{
        vecmem::get_data(cluster)};
    const edm::pixel_cell_container::const_device cells_device{
        vecmem::get_data(cells)};
    const edm::pixel_module_container::const_device modules_device{
        vecmem::get_data(modules)};
    calc_cluster_properties(cluster_device, cells_device, modules_device,
                            module_index, mean, var, totalWeight);

    if (totalWeight > 0.) {
        measurement m;
        m.module_link = module_index;
        m.surface_link = edm::pixel_module_container::surface_link::get(
            modules)[module_index];
        // normalize the cell position
        m.local = mean;
        // normalize the variance
        m.variance[0] = var[0] / totalWeight;
        m.variance[1] = var[1] / totalWeight;
        // plus pitch^2 / 12
        const auto pitch =
            edm::pixel_module_container::pixel_data::get(modules)[module_index]
                .get_pitch();
        m.variance =
            m.variance + point2{pitch[0] * pitch[0] / static_cast<scalar>(12.),
                                pitch[1] * pitch[1] / static_cast<scalar>(12.)};
        // @todo add variance estimation

        measurements.push_back(std::move(m));
    }
}

}  // namespace traccc::detail
