/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "edm/cell.hpp"
#include "edm/cluster.hpp"
#include "detail/sparse_ccl.hpp"

namespace traccc {

    /// Connected component labelling.
    struct component_connection {

        /// Callable operator for the connected component, based on one single module
        ///
        /// @param cells are the input cells into the connected component, they are
        ///              per module and unordered
        /// @param module The description of the module that the cells belong to
        ///
        /// c++20 piping interface:
        /// @return a cluster collection
        cluster_collection operator()(const cell_collection& cells,
                                      const cell_module& module) const {
            cluster_collection clusters;
            clusters.placement = module.placement;
            this->operator()(cells, module, clusters);
            return clusters;
        }

        /// Callable operator for the connected component, based on one single module
        ///
        /// @param cells are the input cells into the connected component, they are
        ///              per module and unordered
        /// @param module The description of the module that the cells belong to
        /// @param clusters[in,out] are the output clusters
        ///
        /// void interface
        void operator()(const cell_collection& cells, const cell_module& module,
                        cluster_collection& clusters) const {
            // Assign the module id
            clusters.module = module.module;
            // Run the algorithm
            auto connected_cells = detail::sparse_ccl(cells);
            std::vector<cluster> cluster_items(std::get<0>(connected_cells),cluster{});
            unsigned int icell = 0;
            for (auto cell_label : std::get<1>(connected_cells)){
                auto cindex = static_cast<unsigned int>(cell_label-1);
                if (cindex < cluster_items.size()){
                    cluster_items[cindex].cells.push_back(cells[icell++]);
                }
            }
            clusters.items = cluster_items;
        }

    };

}
