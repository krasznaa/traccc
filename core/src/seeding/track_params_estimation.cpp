/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Library include(s).
#include "traccc/seeding/track_params_estimation.hpp"

#include "traccc/edm/seed.hpp"
#include "traccc/seeding/track_params_estimation_helper.hpp"

namespace traccc {

track_params_estimation::track_params_estimation(vecmem::memory_resource& mr)
    : m_mr(mr) {}

track_params_estimation::output_type track_params_estimation::operator()(
    const edm::measurement_collection::const_view& measurements_view,
    const spacepoint_collection_types::const_view& spacepoints_view,
    const seed_collection_types::const_view& seeds_view, const vector3& bfield,
    const std::array<traccc::scalar, traccc::e_bound_size>& stddev) const {

    // Create device containers for the inputs.
    const edm::measurement_collection::const_device measurements{
        measurements_view};
    const spacepoint_collection_types::const_device spacepoints{
        spacepoints_view};
    const seed_collection_types::const_device seeds{seeds_view};

    const seed_collection_types::const_device::size_type num_seeds =
        seeds.size();
    output_type result(num_seeds, &m_mr.get());

    for (seed_collection_types::const_device::size_type i = 0; i < num_seeds;
         ++i) {
        bound_track_parameters& track_params = result[i];
        track_params.set_vector(details::seed_to_bound_vector(
            measurements, spacepoints, seeds[i], bfield));

        // Set Covariance
        for (std::size_t j = 0; j < e_bound_size; ++j) {
            getter::element(track_params.covariance(), j, j) =
                stddev[j] * stddev[j];
        }

        // Get geometry ID for bottom spacepoint
        const auto& spB = spacepoints.at(
            static_cast<spacepoint_collection_types::const_device::size_type>(
                seeds[i].spB_link));
        track_params.set_surface_link(
            measurements.at(spB.measurement_index).geometry_id());
    }

    return result;
}

}  // namespace traccc
