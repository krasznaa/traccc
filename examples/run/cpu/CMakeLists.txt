# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

traccc_add_executable( seeding_example "seeding_example.cpp"
   LINK_LIBRARIES vecmem::core traccc::core traccc::io
   traccc::performance traccc::options detray::detectors detray::io
   traccc_examples_common )

traccc_add_executable( seq_example "seq_example.cpp"
   LINK_LIBRARIES vecmem::core traccc::core traccc::io
   traccc::performance traccc::options detray::detectors detray::io
   traccc_examples_common )

traccc_add_executable( truth_finding_example "truth_finding_example.cpp"
   LINK_LIBRARIES vecmem::core detray::detectors traccc::core traccc::io
   traccc::performance traccc::options traccc_examples_common )

traccc_add_executable( truth_fitting_example "truth_fitting_example.cpp"
   LINK_LIBRARIES vecmem::core detray::io detray::detectors traccc::core
   traccc::io traccc::performance traccc::options traccc_examples_common )

traccc_add_executable( misaligned_truth_fitting_example "misaligned_truth_fitting_example.cpp"
   LINK_LIBRARIES vecmem::core detray::io detray::detectors traccc::core
   traccc::io traccc::performance traccc::options traccc_examples_common )

#
# Set up the "throughput applications".
#
add_library( traccc_examples_cpu STATIC
   "full_chain_algorithm.hpp"
   "full_chain_algorithm.cpp" )
target_link_libraries( traccc_examples_cpu
   PUBLIC vecmem::core detray::core detray::detectors traccc::core
   traccc_examples_common )

traccc_add_executable( throughput_st "throughput_st.cpp"
   LINK_LIBRARIES indicators::indicators vecmem::core detray::detectors
                  detray::io traccc::core traccc::io traccc::performance
                  traccc::options traccc_examples_cpu )

traccc_add_executable( throughput_mt "throughput_mt.cpp"
   LINK_LIBRARIES indicators::indicators TBB::tbb vecmem::core detray::detectors
                  detray::io traccc::core traccc::io traccc::performance
                  traccc::options traccc_examples_cpu )
