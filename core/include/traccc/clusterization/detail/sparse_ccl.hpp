/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Library include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/cell_container.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>
#include <vecmem/containers/device_vector.hpp>

namespace traccc {

/// Implemementation of SparseCCL, following
/// [DOI: 10.1109/DASIP48288.2019.9049184]
///
/// Requires cells to be sorted in column major
namespace detail {

/// Find root of the tree for entry @param e
///
/// @param L an equivalance table
///
/// @return the root of @param e
///
TRACCC_HOST_DEVICE inline unsigned int find_root(
    const vecmem::device_vector<unsigned int>& L, unsigned int e) {

    unsigned int r = e;
    assert(r < L.size());
    while (L[r] != r) {
        r = L[r];
        assert(r < L.size());
    }
    return r;
}

/// Create a union of two entries @param e1 and @param e2
///
/// @param L an equivalance table
///
/// @return the rleast common ancestor of the entries
///
TRACCC_HOST_DEVICE inline unsigned int make_union(
    vecmem::device_vector<unsigned int>& L, unsigned int e1, unsigned int e2) {

    int e;
    if (e1 < e2) {
        e = e1;
        assert(e2 < L.size());
        L[e2] = e;
    } else {
        e = e2;
        assert(e1 < L.size());
        L[e1] = e;
    }
    return e;
}

/// Helper method to find adjacent cells
///
/// @param cells The (device) cell container
/// @param index_a the index of the first cell
/// @param index_b the index of the second cell
///
/// @return boolan to indicate 8-cell connectivity
///
TRACCC_HOST_DEVICE inline bool is_adjacent(
    const edm::cell_container::const_device& cells, unsigned int index_a,
    unsigned int index_b) {

    const channel_id channel0_a = cells.channel0()[index_a];
    const channel_id channel0_b = cells.channel0()[index_b];
    const channel_id channel1_a = cells.channel1()[index_a];
    const channel_id channel1_b = cells.channel1()[index_b];
    return ((((channel0_a - channel0_b) * (channel0_a - channel0_b)) <= 1) &&
            (((channel1_a - channel1_b) * (channel1_a - channel1_b)) <= 1) &&
            (cells.module_index()[index_a] == cells.module_index()[index_b]));
}

/// Helper method to find define distance,
/// does not need abs, as channels are sorted in
/// column major
///
/// @param cells The (device) cell container
/// @param index_a the index of the first cell
/// @param index_b the index of the second cell
///
/// @return boolan to indicate !8-cell connectivity
///
TRACCC_HOST_DEVICE inline bool is_far_enough(
    const edm::cell_container::const_device& cells, unsigned int index_a,
    unsigned int index_b) {

    return (((cells.channel1()[index_a] - cells.channel1()[index_b]) > 1) ||
            (cells.module_index()[index_a] != cells.module_index()[index_b]));
}

/// Sparce CCL algorithm
///
/// @param cells_view View of the cells to clusterize
/// @param labels_view View of the vector of the output indices (to which
/// cluster a cell belongs to)
/// @return number of clusters
///
TRACCC_HOST_DEVICE inline unsigned int sparse_ccl(
    const edm::cell_container::const_view& cells_view,
    vecmem::data::vector_view<unsigned int> labels_view) {

    // The number of labels (clusters).
    unsigned int n_labels = 0;

    // Create "data objects" on top of the views.
    const edm::cell_container::const_device cells{cells_view};
    vecmem::device_vector<unsigned int> labels{labels_view};

    // As an aggressive optimization, get the number of cells into a local
    // variable.
    const unsigned int n_cells = cells.size();

    // first scan: pixel association
    unsigned int start_j = 0;
    for (unsigned int i = 0; i < n_cells; ++i) {
        labels[i] = i;
        unsigned int ai = i;
        for (unsigned int j = start_j; j < i; ++j) {
            if (is_adjacent(cells, i, j)) {
                ai = make_union(labels, ai, find_root(labels, j));
            } else if (is_far_enough(cells, i, j)) {
                ++start_j;
            }
        }
    }

    // second scan: transitive closure
    for (unsigned int i = 0; i < n_cells; ++i) {
        if (labels[i] == i) {
            labels[i] = n_labels++;
        } else {
            labels[i] = labels[labels[i]];
        }
    }

    return n_labels;
}

}  // namespace detail

}  // namespace traccc
