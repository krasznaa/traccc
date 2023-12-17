/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s)
#include "traccc/clusterization/detail/measurement_creation_helper.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void aggregate_cluster(
    const edm::cell_container::const_device& cells,
    const edm::cell_module_container::const_device& modules,
    const vecmem::data::vector_view<unsigned short> f_view,
    const unsigned int start, const unsigned int end, const unsigned short cid,
    measurement& out, vecmem::data::vector_view<unsigned int> cell_links,
    const unsigned int link) {

    const vecmem::device_vector<unsigned short> f(f_view);
    vecmem::device_vector<unsigned int> cell_links_device(cell_links);

    /*
     * Now, we iterate over all other cells to check if they belong
     * to our cluster. Note that we can start at the current index
     * because no cell is ever a child of a cluster owned by a cell
     * with a higher ID.
     */
    scalar totalWeight = 0.;
    point2 mean{0., 0.}, var{0., 0.};
    const unsigned int module_index = cells.module_index()[cid + start];
    const unsigned short partition_size = end - start;

    channel_id maxChannel1 = std::numeric_limits<channel_id>::min();

    for (unsigned short j = cid; j < partition_size; j++) {

        assert(j < f.size());

        const unsigned int pos = j + start;
        /*
         * Terminate the process earlier if we have reached a cell sufficiently
         * in a different module.
         */
        if (cells.module_index()[pos] != module_index) {
            break;
        }

        /*
         * If the value of this cell is equal to our, that means it
         * is part of our cluster. In that case, we take its values
         * for position and add them to our accumulators.
         */
        if (f[j] == cid) {

            const channel_id channel1 = cells.channel1()[pos];
            if (channel1 > maxChannel1) {
                maxChannel1 = channel1;
            }

            const scalar activation = cells.activation()[pos];
            const float weight = traccc::detail::signal_cell_modelling(
                activation, modules, module_index);

            if (weight > modules.threshold()[module_index]) {
                totalWeight += activation;
                const point2 cell_position = traccc::detail::position_from_cell(
                    cells, pos, modules, module_index);
                const point2 prev = mean;
                const point2 diff = cell_position - prev;

                mean = prev + (weight / totalWeight) * diff;
                for (char i = 0; i < 2; ++i) {
                    var[i] = var[i] +
                             weight * (diff[i]) * (cell_position[i] - mean[i]);
                }
            }

            cell_links_device.at(pos) = link;
        }

        /*
         * Terminate the process earlier if we have reached a cell sufficiently
         * far away from the cluster in the dominant axis.
         */
        if (cells.channel1()[pos] > maxChannel1 + 1) {
            break;
        }
    }
    if (totalWeight > static_cast<scalar>(0.)) {
        for (char i = 0; i < 2; ++i) {
            var[i] /= totalWeight;
        }
        const auto pitch = modules.pixel_data()[module_index].get_pitch();
        var = var + point2{pitch[0] * pitch[0] / static_cast<scalar>(12.),
                           pitch[1] * pitch[1] / static_cast<scalar>(12.)};
    }

    /*
     * Fill output vector with calculated cluster properties
     */
    out.local = mean;
    out.variance = var;
    out.surface_link = modules.surface_link()[module_index];
    out.module_link = module_index;
    // The following will need to be filled properly "soon".
    out.meas_dim = 2u;
}

}  // namespace traccc::device
